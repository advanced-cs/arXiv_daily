# 自然语言处理 cs.CL

- **最新发布 80 篇**

- **更新 56 篇**

## 最新发布

#### [new 001] Just Use XML: Revisiting Joint Translation and Label Projection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于跨语言迁移任务，解决标签投影与翻译分离导致的效率问题。通过XML标签联合进行翻译和标签投影，提升跨语言迁移效果。**

- **链接: [https://arxiv.org/pdf/2603.12021](https://arxiv.org/pdf/2603.12021)**

> **作者:** Thennal D K; Chris Biemann; Hans Ole Hatzel
>
> **摘要:** Label projection is an effective technique for cross-lingual transfer, extending span-annotated datasets from a high-resource language to low-resource ones. Most approaches perform label projection as a separate step after machine translation, and prior work that combines the two reports degraded translation quality. We re-evaluate this claim with LabelPigeon, a novel framework that jointly performs translation and label projection via XML tags. We design a direct evaluation scheme for label projection, and find that LabelPigeon outperforms baselines and actively improves translation quality in 11 languages. We further assess translation quality across 203 languages and varying annotation complexity, finding consistent improvement attributed to additional fine-tuning. Finally, across 27 languages and three downstream tasks, we report substantial gains in cross-lingual transfer over comparable work, up to +39.9 F1 on NER. Overall, our results demonstrate that XML-tagged label projection provides effective and efficient label transfer without compromising translation quality.
>
---
#### [new 002] DatedGPT: Preventing Lookahead Bias in Large Language Models with Time-Aware Pretraining
- **分类: cs.CL; q-fin.GN**

- **简介: 该论文属于金融领域的时间敏感任务，旨在解决大语言模型在财务回测中的前瞻偏差问题。通过时间感知预训练和严格年份分割数据，构建了DatedGPT模型族。**

- **链接: [https://arxiv.org/pdf/2603.11838](https://arxiv.org/pdf/2603.11838)**

> **作者:** Yutong Yan; Raphael Tang; Zhenyu Gao; Wenxi Jiang; Yao Lu
>
> **摘要:** In financial backtesting, large language models pretrained on internet-scale data risk introducing lookahead bias that undermines their forecasting validity, as they may have already seen the true outcome during training. To address this, we present DatedGPT, a family of twelve 1.3B-parameter language models, each trained from scratch on approximately 100 billion tokens of temporally partitioned data with strict annual cutoffs spanning 2013 to 2024. We further enhance each model with instruction fine-tuning on both general-domain and finance-specific datasets curated to respect the same temporal boundaries. Perplexity-based probing confirms that each model's knowledge is effectively bounded by its data cutoff year, while evaluation on standard benchmarks shows competitive performance with existing models of similar scale. We provide an interactive web demo that allows users to query and compare responses from models across different cutoff years.
>
---
#### [new 003] A technology-oriented mapping of the language and translation industry: Analysing stakeholder values and their potential implication for translation pedagogy
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于行业分析任务，探讨自动化背景下语言与翻译行业价值构建问题。通过访谈分析，揭示技术与人类价值的互动关系及其对翻译教育的影响。**

- **链接: [https://arxiv.org/pdf/2603.11667](https://arxiv.org/pdf/2603.11667)**

> **作者:** María Isabel Rivas Ginel; Janiça Hackenbuchner; Alina Secară; Ralph Krüger; Caroline Rossi
>
> **备注:** Under review
>
> **摘要:** This paper examines how value is constructed and negotiated in today's increasingly automated language and translation industry. Drawing on interview data from twenty-nine industry stakeholders collected within the LT-LiDER project, the study analyses how human value, technological value, efficiency, and adaptability are articulated across different professional roles. Using Chesterman's framework of translation ethics and associated values as an analytical lens, the paper shows that efficiency-oriented technological values aligned with the ethics of service have become baseline expectations in automated production environments, where speed, scalability, and deliverability dominate evaluation criteria. At the same time, human value is not displaced but repositioned, emerging primarily through expertise, oversight, accountability, and contextual judgment embedded within technology-mediated workflows. A central finding is the prominence of adaptability as a mediating value linking human and technological domains. Adaptability is constructed as a core professional requirement, reflecting expectations that translators continuously adjust their skills, roles, and identities in response to evolving tools and organisational demands. The paper argues that automation reshapes rather than replaces translation value, creating an interdependent configuration in which technological efficiency enables human communicative work.
>
---
#### [new 004] CHiL(L)Grader: Calibrated Human-in-the-Loop Short-Answer Grading
- **分类: cs.CL**

- **简介: 该论文属于教育评估任务，旨在解决自动评分中模型可靠性不足的问题。通过引入置信度校准和人机协作，提升评分准确性与安全性。**

- **链接: [https://arxiv.org/pdf/2603.11957](https://arxiv.org/pdf/2603.11957)**

> **作者:** Pranav Raikote; Korbinian Randl; Ioanna Miliou; Athanasios Lakes; Panagiotis Papapetrou
>
> **摘要:** Scaling educational assessment with large language models requires not just accuracy, but the ability to recognize when predictions are trustworthy. Instruction-tuned models tend to be overconfident, and their reliability deteriorates as curricula evolve, making fully autonomous deployment unsafe in high-stakes settings. We introduce CHiL(L)Grader, the first automated grading framework that incorporates calibrated confidence estimation into a human-in-the-loop workflow. Using post-hoc temperature scaling, confidence-based selective prediction, and continual learning, CHiL(L)Grader automates only high-confidence predictions while routing uncertain cases to human graders, and adapts to evolving rubrics and unseen questions. Across three short-answer grading datasets, CHiL(L)Grader automatically scores 35-65% of responses at expert-level quality (QWK >= 0.80). A QWK gap of 0.347 between accepted and rejected predictions confirms the effectiveness of the confidence-based routing. Each correction cycle strengthens the model's grading capability as it learns from teacher feedback. These results show that uncertainty quantification is key for reliable AI-assisted grading.
>
---
#### [new 005] Evaluating Explainable AI Attribution Methods in Neural Machine Translation via Attention-Guided Knowledge Distillation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于神经机器翻译任务，旨在评估XAI方法在seq2seq模型中的有效性。通过注意力引导的知识蒸馏，研究不同归因方法对模型性能的影响。**

- **链接: [https://arxiv.org/pdf/2603.11342](https://arxiv.org/pdf/2603.11342)**

> **作者:** Aria Nourbakhsh; Salima Lamsiyah; Adelaide Danilov; Christoph Schommer
>
> **备注:** 37 pages, 11 figures
>
> **摘要:** The study of the attribution of input features to the output of neural network models is an active area of research. While numerous Explainable AI (XAI) techniques have been proposed to interpret these models, the systematic and automated evaluation of these methods in sequence-to-sequence (seq2seq) models is less explored. This paper introduces a new approach for evaluating explainability methods in transformer-based seq2seq models. We use teacher-derived attribution maps as a structured side signal to guide a student model, and quantify the utility of different attribution methods through the student's ability to simulate targets. Using the Inseq library, we extract attribution scores over source-target sequence pairs and inject these scores into the attention mechanism of a student transformer model under four composition operators (addition, multiplication, averaging, and replacement). Across three language pairs (de-en, fr-en, ar-en) and attributions from Marian-MT and mBART models, Attention, Value Zeroing, and Layer Gradient $\times$ Activation consistently yield the largest gains in BLEU (and corresponding improvements in chrF) relative to baselines. In contrast, other gradient-based methods (Saliency, Integrated Gradients, DeepLIFT, Input $\times$ Gradient, GradientShap) lead to smaller and less consistent improvements. These results suggest that different attribution methods capture distinct signals and that attention-derived attributions better capture alignment between source and target representations in seq2seq models. Finally, we introduce an Attributor transformer that, given a source-target pair, learns to reconstruct the teacher's attribution map. Our findings demonstrate that the more accurately the Attributor can reproduce attribution maps, the more useful an injection of those maps is for the downstream task. The source code can be found on GitHub.
>
---
#### [new 006] Temporal Text Classification with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于时间文本分类任务，旨在评估大语言模型在文本年代预测中的表现。研究对比了多种模型，发现微调可提升开源模型效果，但仍不及专有模型。**

- **链接: [https://arxiv.org/pdf/2603.11295](https://arxiv.org/pdf/2603.11295)**

> **作者:** Nishat Raihan; Marcos Zampieri
>
> **摘要:** Languages change over time. Computational models can be trained to recognize such changes enabling them to estimate the publication date of texts. Despite recent advancements in Large Language Models (LLMs), their performance on automatic dating of texts, also known as Temporal Text Classification (TTC), has not been explored. This study provides the first systematic evaluation of leading proprietary (Claude 3.5, GPT-4o, Gemini 1.5) and open-source (LLaMA 3.2, Gemma 2, Mistral, Nemotron 4) LLMs on TTC using three historical corpora, two in English and one in Portuguese. We test zero-shot and few-shot prompting, and fine-tuning settings. Our results indicate that proprietary models perform well, especially with few-shot prompting. They also indicate that fine-tuning substantially improves open-source models but that they still fail to match the performance delivered by proprietary LLMs.
>
---
#### [new 007] MDER-DR: Multi-Hop Question Answering with Entity-Centric Summaries
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于问答任务，解决知识图谱检索增强生成中的上下文丢失问题。提出MDER-DR框架，通过实体中心摘要和分解解析提升多跳问答性能。**

- **链接: [https://arxiv.org/pdf/2603.11223](https://arxiv.org/pdf/2603.11223)**

> **作者:** Riccardo Campi; Nicolò Oreste Pinciroli Vago; Mathyas Giudici; Marco Brambilla; Piero Fraternali
>
> **备注:** Our code is available at this https URL
>
> **摘要:** Retrieval-Augmented Generation (RAG) over Knowledge Graphs (KGs) suffers from the fact that indexing approaches may lose important contextual nuance when text is reduced to triples, thereby degrading performance in downstream Question-Answering (QA) tasks, particularly for multi-hop QA, which requires composing answers from multiple entities, facts, or relations. We propose a domain-agnostic, KG-based QA framework that covers both the indexing and retrieval/inference phases. A new indexing approach called Map-Disambiguate-Enrich-Reduce (MDER) generates context-derived triple descriptions and subsequently integrates them with entity-level summaries, thus avoiding the need for explicit traversal of edges in the graph during the QA retrieval phase. Complementing this, we introduce Decompose-Resolve (DR), a retrieval mechanism that decomposes user queries into resolvable triples and grounds them in the KG via iterative reasoning. Together, MDER and DR form an LLM-driven QA pipeline that is robust to sparse, incomplete, and complex relational data. Experiments show that on standard and domain specific benchmarks, MDER-DR achieves substantial improvements over standard RAG baselines (up to 66%), while maintaining cross-lingual robustness. Our code is available at this https URL.
>
---
#### [new 008] Try, Check and Retry: A Divide-and-Conquer Framework for Boosting Long-context Tool-Calling Performance of LLMs
- **分类: cs.CL**

- **简介: 该论文属于长文本工具调用任务，旨在提升LLMs在复杂场景下的工具调用性能。提出Tool-DC框架，通过“试错-检查-重试”策略降低推理难度，增强模型自省能力。**

- **链接: [https://arxiv.org/pdf/2603.11495](https://arxiv.org/pdf/2603.11495)**

> **作者:** Kunfeng Chen; Qihuang Zhong; Juhua Liu; Bo Du; Dacheng Tao
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** Tool-calling empowers Large Language Models (LLMs) to interact with external environments. However, current methods often struggle to handle massive and noisy candidate tools in long-context tool-calling tasks, limiting their real-world application. To this end, we propose Tool-DC, a Divide-and-Conquer framework for boosting tool-calling performance of LLMs. The core of Tool-DC is to reduce the reasoning difficulty and make full use of self-reflection ability of LLMs via a "Try-Check-Retry" paradigm. Specifically, Tool-DC involves two variants: 1) the training-free Tool-DC (TF), which is plug-and-play and flexible; 2) the training-based Tool-DC (TB), which is more inference-efficient. Extensive experiments show that both Tool-DC methods outperform their counterparts by a clear margin. Tool-DC (TF) brings up to +25.10% average gains against the baseline on BFCL and ACEBench benchmarks, while Tool-DC (TB) enables Qwen2.5-7B to achieve comparable or even better performance than proprietary LLMs, e.g., OpenAI o3 and Claude-Haiku-4.5.
>
---
#### [new 009] UtilityMax Prompting: A Formal Framework for Multi-Objective Large Language Model Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出UtilityMax Prompting框架，解决多目标LLM优化问题。通过数学语言明确任务，提升推荐系统的精度和NDCG指标。**

- **链接: [https://arxiv.org/pdf/2603.11583](https://arxiv.org/pdf/2603.11583)**

> **作者:** Ofir Marom
>
> **摘要:** The success of a Large Language Model (LLM) task depends heavily on its prompt. Most use-cases specify prompts using natural language, which is inherently ambiguous when multiple objectives must be simultaneously satisfied. In this paper we introduce UtilityMax Prompting, a framework that specifies tasks using formal mathematical language. We reconstruct the task as an influence diagram in which the LLM's answer is the sole decision variable. A utility function is defined over the conditional probability distributions within the diagram, and the LLM is instructed to find the answer that maximises expected utility. This constrains the LLM to reason explicitly about each component of the objective, directing its output toward a precise optimization target rather than a subjective natural language interpretation. We validate our approach on the MovieLens 1M dataset across three frontier models (Claude Sonnet 4.6, GPT-5.4, and Gemini 2.5 Pro), demonstrating consistent improvements in precision and Normalized Discounted Cumulative Gain (NDCG) over natural language baselines in a multi-objective movie recommendation task.
>
---
#### [new 010] BTZSC: A Benchmark for Zero-Shot Text Classification Across Cross-Encoders, Embedding Models, Rerankers and LLMs
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **简介: 该论文属于零样本文本分类任务，旨在解决不同模型在无标注数据下的性能比较问题。提出BTZSC基准，评估四种模型家族的表现。**

- **链接: [https://arxiv.org/pdf/2603.11991](https://arxiv.org/pdf/2603.11991)**

> **作者:** Ilias Aarab
>
> **备注:** Accepted at ICLR 2026. 31 pages, 5 figures, 9 tables. Code: this https URL ; Dataset: this https URL ; Leaderboard: this https URL . Proceedings of the Fourteenth International Conference on Learning Representations (ICLR 2026), 2026
>
> **摘要:** Zero-shot text classification (ZSC) offers the promise of eliminating costly task-specific annotation by matching texts directly to human-readable label descriptions. While early approaches have predominantly relied on cross-encoder models fine-tuned for natural language inference (NLI), recent advances in text-embedding models, rerankers, and instruction-tuned large language models (LLMs) have challenged the dominance of NLI-based architectures. Yet, systematically comparing these diverse approaches remains difficult. Existing evaluations, such as MTEB, often incorporate labeled examples through supervised probes or fine-tuning, leaving genuine zero-shot capabilities underexplored. To address this, we introduce BTZSC, a comprehensive benchmark of 22 public datasets spanning sentiment, topic, intent, and emotion classification, capturing diverse domains, class cardinalities, and document lengths. Leveraging BTZSC, we conduct a systematic comparison across four major model families, NLI cross-encoders, embedding models, rerankers and instruction-tuned LLMs, encompassing 38 public and custom checkpoints. Our results show that: (i) modern rerankers, exemplified by Qwen3-Reranker-8B, set a new state-of-the-art with macro F1 = 0.72; (ii) strong embedding models such as GTE-large-en-v1.5 substantially close the accuracy gap while offering the best trade-off between accuracy and latency; (iii) instruction-tuned LLMs at 4--12B parameters achieve competitive performance (macro F1 up to 0.67), excelling particularly on topic classification but trailing specialized rerankers; (iv) NLI cross-encoders plateau even as backbone size increases; and (v) scaling primarily benefits rerankers and LLMs over embedding models. BTZSC and accompanying evaluation code are publicly released to support fair and reproducible progress in zero-shot text understanding.
>
---
#### [new 011] CLASP: Defending Hybrid Large Language Models Against Hidden State Poisoning Attacks
- **分类: cs.CL**

- **简介: 该论文属于安全防护任务，旨在防御隐状态污染攻击。针对SSM模型的漏洞，提出CLASP模型通过分类识别恶意标记，有效提升安全性。**

- **链接: [https://arxiv.org/pdf/2603.12206](https://arxiv.org/pdf/2603.12206)**

> **作者:** Alexandre Le Mercier; Thomas Demeester; Chris Develder
>
> **备注:** 22 pages, 6 figures
>
> **摘要:** State space models (SSMs) like Mamba have gained significant traction as efficient alternatives to Transformers, achieving linear complexity while maintaining competitive performance. However, Hidden State Poisoning Attacks (HiSPAs), a recently discovered vulnerability that corrupts SSM memory through adversarial strings, pose a critical threat to these architectures and their hybrid variants. Framing the HiSPA mitigation task as a binary classification problem at the token level, we introduce the CLASP model to defend against this threat. CLASP exploits distinct patterns in Mamba's block output embeddings (BOEs) and uses an XGBoost classifier to identify malicious tokens with minimal computational overhead. We consider a realistic scenario in which both SSMs and HiSPAs are likely to be used: an LLM screening résumés to identify the best candidates for a role. Evaluated on a corpus of 2,483 résumés totaling 9.5M tokens with controlled injections, CLASP achieves 95.9% token-level F1 score and 99.3% document-level F1 score on malicious tokens detection. Crucially, the model generalizes to unseen attack patterns: under leave-one-out cross-validation, performance remains high (96.9% document-level F1), while under clustered cross-validation with structurally novel triggers, it maintains useful detection capability (91.6% average document-level F1). Operating independently of any downstream model, CLASP processes 1,032 tokens per second with under 4GB VRAM consumption, potentially making it suitable for real-world deployment as a lightweight front-line defense for SSM-based and hybrid architectures. All code and detailed results are available at this https URL.
>
---
#### [new 012] QAQ: Bidirectional Semantic Coherence for Selecting High-Quality Synthetic Code Instructions
- **分类: cs.CL**

- **简介: 该论文属于代码生成任务，旨在解决合成数据质量差的问题。提出QAQ框架，通过双向语义一致性筛选高质量数据，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.12165](https://arxiv.org/pdf/2603.12165)**

> **作者:** Jiayin Lei; Ming Ma; Yunxi Duan; Chenxi Li; Tianming Yang
>
> **备注:** 12 pages, 5 figures. Under review at ACL 2026
>
> **摘要:** Synthetic data has become essential for training code generation models, yet it introduces significant noise and hallucinations that are difficult to detect with current metrics. Existing data selection methods like Instruction-Following Difficulty (IFD) typically assess how hard a model generates an answer given a query ($A|Q$). However, this metric is ambiguous on noisy synthetic data, where low probability can distinguish between intrinsic task complexity and model-generated hallucinations. Here, we propose QAQ, a novel data selection framework that evaluates data quality from the reverse direction: how well can the answer predict the query ($Q|A$)? We define Reverse Mutual Information (RMI) to quantify the information gain about the query conditioned on the answer. Our analyses reveal that both extremes of RMI signal quality issues: low RMI indicates semantic misalignment, while excessively high RMI may contain defect patterns that LLMs easily recognize. Furthermore, we introduce a selection strategy based on the disagreement between strong and weak models to identify samples that are valid yet challenging. Experiments on the WarriorCoder dataset demonstrate that selecting just 25% of data using stratified RMI achieves comparable performance to full-data training, significantly outperforming existing data selection methods. Our approach highlights the importance of bidirectional semantic coherence in synthetic data curation, offering a scalable pathway to reduce computational costs without sacrificing model capability.
>
---
#### [new 013] Algorithmic Consequences of Particle Filters for Sentence Processing: Amplified Garden-Paths and Digging-In Effects
- **分类: cs.CL**

- **简介: 该论文属于语言处理任务，探讨结构歧义对句法加工的影响。通过粒子滤波模型，研究者揭示了歧义处理中的增强花园路径效应和实时挖深效应，指出粒子数量影响歧义处理难度。**

- **链接: [https://arxiv.org/pdf/2603.11412](https://arxiv.org/pdf/2603.11412)**

> **作者:** Amani Maina-Kilaas; Roger Levy
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Under surprisal theory, linguistic representations affect processing difficulty only through the bottleneck of surprisal. Our best estimates of surprisal come from large language models, which have no explicit representation of structural ambiguity. While LLM surprisal robustly predicts reading times across languages, it systematically underpredicts difficulty when structural expectations are violated -- suggesting that representations of ambiguity are causally implicated in sentence processing. Particle filter models offer an alternative where structural hypotheses are explicitly represented as a finite set of particles. We prove several algorithmic consequences of particle filter models, including the amplification of garden-path effects. Most critically, we demonstrate that resampling, a common practice with these models, inherently produces real-time digging-in effects -- where disambiguation difficulty increases with ambiguous region length. Digging-in magnitude scales inversely with particle count: fully parallel models predict no such effect.
>
---
#### [new 014] CoMMET: To What Extent Can LLMs Perform Theory of Mind Tasks?
- **分类: cs.CL**

- **简介: 该论文属于理论心智（ToM）任务，旨在评估大语言模型的社会推理能力。提出多模态基准数据集CoMMET，扩展了评测范围并引入多轮对话测试。**

- **链接: [https://arxiv.org/pdf/2603.11915](https://arxiv.org/pdf/2603.11915)**

> **作者:** Ruirui Chen; Weifeng Jiang; Chengwei Qin; Cheston Tan
>
> **摘要:** Theory of Mind (ToM)-the ability to reason about the mental states of oneself and others-is a cornerstone of human social intelligence. As Large Language Models (LLMs) become ubiquitous in real-world applications, validating their capacity for this level of social reasoning is essential for effective and natural interactions. However, existing benchmarks for assessing ToM in LLMs are limited; most rely solely on text inputs and focus narrowly on belief-related tasks. In this paper, we propose a new multimodal benchmark dataset, CoMMET, a Comprehensive Mental states and Moral Evaluation Task inspired by the Theory of Mind Booklet Task. CoMMET expands the scope of evaluation by covering a broader range of mental states and introducing multi-turn testing. To the best of our knowledge, this is the first multimodal dataset to evaluate ToM in a multi-turn conversational setting. Through a comprehensive assessment of LLMs across different families and sizes, we analyze the strengths and limitations of current models and identify directions for future improvement. Our work offers a deeper understanding of the social cognitive capabilities of modern LLMs.
>
---
#### [new 015] SommBench: Assessing Sommelier Expertise of Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SommBench，用于评估语言模型的品酒专家能力。针对语言模型在感官判断上的局限性，设计三个任务测试其葡萄酒知识和搭配能力。**

- **链接: [https://arxiv.org/pdf/2603.12117](https://arxiv.org/pdf/2603.12117)**

> **作者:** William Brach; Tomas Bedej; Jacob Nielsen; Jacob Pichna; Juraj Bedej; Eemeli Saarensilta; Julie Dupouy; Gianluca Barmina; Andrea Blasi Núñez; Peter Schneider-Kamp; Kristian Košťál; Michal Ries; Lukas Galke Poech
>
> **摘要:** With the rapid advances of large language models, it becomes increasingly important to systematically evaluate their multilingual and multicultural capabilities. Previous cultural evaluation benchmarks focus mainly on basic cultural knowledge that can be encoded in linguistic form. Here, we propose SommBench, a multilingual benchmark to assess sommelier expertise, a domain deeply grounded in the senses of smell and taste. While language models learn about sensory properties exclusively through textual descriptions, SommBench tests whether this textual grounding is sufficient to emulate expert-level sensory judgment. SommBench comprises three main tasks: Wine Theory Question Answering (WTQA), Wine Feature Completion (WFC), and Food-Wine Pairing (FWP). SommBench is available in multiple languages: English, Slovak, Swedish, Finnish, German, Danish, Italian, and Spanish. This helps separate a language model's wine expertise from its language skills. The benchmark datasets were developed in close collaboration with a professional sommelier and native speakers of the respective languages, resulting in 1,024 wine theory question-answering questions, 1,000 wine feature-completion examples, and 1,000 food-wine pairing examples. We provide results for the most popular language models, including closed-weights models such as Gemini 2.5, and open-weights models, such as GPT-OSS and Qwen 3. Our results show that the most capable models perform well on wine theory question answering (up to 97% correct with a closed-weights model), yet feature completion (peaking at 65%) and food-wine pairing show (MCC ranging between 0 and 0.39) turn out to be more challenging. These results position SommBench as an interesting and challenging benchmark for evaluating the sommelier expertise of language models. The benchmark is publicly available at this https URL.
>
---
#### [new 016] Semi-Synthetic Parallel Data for Translation Quality Estimation: A Case Study of Dataset Building for an Under-Resourced Language Pair
- **分类: cs.CL**

- **简介: 该论文属于机器翻译质量评估任务，旨在解决资源匮乏语言对的QE系统开发难题。通过构建半合成平行数据集，训练神经网络模型以提升翻译质量评估效果。**

- **链接: [https://arxiv.org/pdf/2603.11743](https://arxiv.org/pdf/2603.11743)**

> **作者:** Assaf Siani; Anna Kernerman; Ilan Kernerman
>
> **摘要:** Quality estimation (QE) plays a crucial role in machine translation (MT) workflows, as it serves to evaluate generated outputs that have no reference translations and to determine whether human post-editing or full retranslation is necessary. Yet, developing highly accurate, adaptable and reliable QE systems for under-resourced language pairs remains largely unsolved, due mainly to limited parallel corpora and to diverse language-dependent factors, such as with morphosyntactically complex languages. This study presents a semi-synthetic parallel dataset for English-to-Hebrew QE, generated by creating English sentences based on examples of usage that illustrate typical linguistic patterns, translating them to Hebrew using multiple MT engines, and filtering outputs via BLEU-based selection. Each translated segment was manually evaluated and scored by a linguist, and we also incorporated professionally translated English-Hebrew segments from our own resources, which were assigned the highest quality score. Controlled translation errors were introduced to address linguistic challenges, particularly regarding gender and number agreement, and we trained neural QE models, including BERT and XLM-R, on this dataset to assess sentence-level MT quality. Our findings highlight the impact of dataset size, distributed balance, and error distribution on model performance. We will describe the challenges, methodology and results of our experiments, and specify future directions aimed at improving QE performance. This research contributes to advancing QE models for under resourced language pairs, including morphology-rich languages.
>
---
#### [new 017] To Words and Beyond: Probing Large Language Models for Sentence-Level Psycholinguistic Norms of Memorability and Reading Times
- **分类: cs.CL**

- **简介: 该论文研究LLM在句子层面心理语言规范（如记忆性和阅读时间）上的表现，旨在评估其能否准确预测这些人类认知指标。通过微调提升模型性能，验证其有效性。**

- **链接: [https://arxiv.org/pdf/2603.12105](https://arxiv.org/pdf/2603.12105)**

> **作者:** Thomas Hikaru Clark; Carlos Arriaga; Javier Conde; Gonzalo Martínez; Pedro Reviriego
>
> **摘要:** Large Language Models (LLMs) have recently been shown to produce estimates of psycholinguistic norms, such as valence, arousal, or concreteness, for words and multiword expressions, that correlate with human judgments. These estimates are obtained by prompting an LLM, in zero-shot fashion, with a question similar to those used in human studies. Meanwhile, for other norms such as lexical decision time or age of acquisition, LLMs require supervised fine-tuning to obtain results that align with ground-truth values. In this paper, we extend this approach to the previously unstudied features of sentence memorability and reading times, which involve the relationship between multiple words in a sentence-level context. Our results show that via fine-tuning, models can provide estimates that correlate with human-derived norms and exceed the predictive power of interpretable baseline predictors, demonstrating that LLMs contain useful information about sentence-level features. At the same time, our results show very mixed zero-shot and few-shot performance, providing further evidence that care is needed when using LLM-prompting as a proxy for human cognitive measures.
>
---
#### [new 018] Translationese as a Rational Response to Translation Task Difficulty
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解释翻译现象（translationese）的成因。通过分析翻译难度因素，探讨其与翻译特征的关系，提出基于信息理论的度量方法进行预测。**

- **链接: [https://arxiv.org/pdf/2603.12050](https://arxiv.org/pdf/2603.12050)**

> **作者:** Maria Kunilovskaya
>
> **备注:** 17 pages, submitted to ARR March 2026
>
> **摘要:** Translations systematically diverge from texts originally produced in the target language, a phenomenon widely referred to as translationese. Translationese has been attributed to production tendencies (e.g. interference, simplification), socio-cultural variables, and language-pair effects, yet a unified explanatory account is still lacking. We propose that translationese reflects cognitive load inherent in the translation task itself. We test whether observable translationese can be predicted from quantifiable measures of translation task difficulty. Translationese is operationalised as a segment-level translatedness score produced by an automatic classifier. Translation task difficulty is conceptualised as comprising source-text and cross-lingual transfer components, operationalised mainly through information-theoretic metrics based on LLM surprisal, complemented by established syntactic and semantic alternatives. We use a bidirectional English-German corpus comprising written and spoken subcorpora. Results indicate that translationese can be partly explained by translation task difficulty, especially in English-to-German. For most experiments, cross-lingual transfer difficulty contributes more than source-text complexity. Information-theoretic indicators match or outperform traditional features in written mode, but offer no advantage in spoken mode. Source-text syntactic complexity and translation-solution entropy emerged as the strongest predictors of translationese across language pairs and modes.
>
---
#### [new 019] LifeSim: Long-Horizon User Life Simulator for Personalized Assistant Evaluation
- **分类: cs.CL**

- **简介: 该论文提出LifeSim，用于模拟用户认知和交互行为，解决个性化助手评估中真实场景不足的问题。构建了LifeSim-Eval基准，评估模型处理长期意图和用户偏好能力。**

- **链接: [https://arxiv.org/pdf/2603.12152](https://arxiv.org/pdf/2603.12152)**

> **作者:** Feiyu Duan; Xuanjing Huang; Zhongyu Wei
>
> **摘要:** The rapid advancement of large language models (LLMs) has accelerated progress toward universal AI assistants. However, existing benchmarks for personalized assistants remain misaligned with real-world user-assistant interactions, failing to capture the complexity of external contexts and users' cognitive states. To bridge this gap, we propose LifeSim, a user simulator that models user cognition through the Belief-Desire-Intention (BDI) model within physical environments for coherent life trajectories generation, and simulates intention-driven user interactive behaviors. Based on LifeSim, we introduce LifeSim-Eval, a comprehensive benchmark for multi-scenario, long-horizon personalized assistance. LifeSim-Eval covers 8 life domains and 1,200 diverse scenarios, and adopts a multi-turn interactive method to assess models' abilities to complete explicit and implicit intentions, recover user profiles, and produce high-quality responses. Under both single-scenario and long-horizon settings, our experiments reveal that current LLMs face significant limitations in handling implicit intention and long-term user preference modeling.
>
---
#### [new 020] Large Language Models for Biomedical Article Classification
- **分类: cs.CL**

- **简介: 该论文属于文本分类任务，研究大语言模型在生物医学文章分类中的应用。通过多种提示策略和输出处理方法，评估模型性能，并与传统算法比较，提出有效配置建议。**

- **链接: [https://arxiv.org/pdf/2603.11780](https://arxiv.org/pdf/2603.11780)**

> **作者:** Jakub Proboszcz; Paweł Cichosz
>
> **备注:** 63 pages, 25 tables, 4 figures
>
> **摘要:** This work presents a systematic and in-depth investigation of the utility of large language models as text classifiers for biomedical article classification. The study uses several small and mid-size open source models, as well as selected closed source ones, and is more comprehensive than most prior work with respect to the scope of evaluated configurations: different types of prompts, output processing methods for generating both class and class probability predictions, as well as few-shot example counts and selection methods. The performance of the most successful configurations is compared to that of conventional classification algorithms. The obtained average PR AUC over 15 challenging datasets above 0.4 for zero-shot prompting and nearly 0.5 for few-shot prompting comes close to that of the naïve Bayes classifier (0.5), the random forest algorithm (0.5 with default settings or 0.55 with hyperparameter tuning) and fine-tuned transformer models (0.5). These results confirm the utility of large language models as text classifiers for non-trivial domains and provide practical recommendations of the most promising setups, including in particular using output token probabilities for class probability prediction.
>
---
#### [new 021] In the LLM era, Word Sense Induction remains unsolved
- **分类: cs.CL**

- **简介: 该论文属于词义消歧任务，探讨无标注数据下的词义归纳问题。研究评估了预训练嵌入和聚类方法，提出基于大模型的WIS方法，并验证数据增强与Wiktionary的作用。**

- **链接: [https://arxiv.org/pdf/2603.11686](https://arxiv.org/pdf/2603.11686)**

> **作者:** Anna Mosolova; Marie Candito; Carlos Ramisch
>
> **备注:** Accepted at ACL 2025 (Findings)
>
> **摘要:** In the absence of sense-annotated data, word sense induction (WSI) is a compelling alternative to word sense disambiguation, particularly in low-resource or domain-specific settings. In this paper, we emphasize methodological problems in current WSI evaluation. We propose an evaluation on a SemCor-derived dataset, respecting the original corpus polysemy and frequency distributions. We assess pre-trained embeddings and clustering algorithms across parts of speech, and propose and evaluate an LLM-based WSI method for English. We evaluate data augmentation sources (LLM-generated, corpus and lexicon), and semi-supervised scenarios using Wiktionary for data augmentation, must-link constraints, number of clusters per lemma. We find that no unsupervised method (whether ours or previous) surpasses the strong "one cluster per lemma" heuristic (1cpl). We also show that (i) results and best systems may vary across POS, (ii) LLMs have troubles performing this task, (iii) data augmentation is beneficial and (iv) capitalizing on Wiktionary does help. It surpasses previous SOTA system on our test set by 3.3\%. WSI is not solved, and calls for a better articulation of lexicons and LLMs' lexical semantics capabilities.
>
---
#### [new 022] ThReadMed-QA: A Multi-Turn Medical Dialogue Benchmark from Real Patient Questions
- **分类: cs.CL**

- **简介: 该论文提出ThReadMed-QA，一个用于医疗对话问答的多轮基准数据集，解决真实患者与医生对话中多轮交互的评估问题。通过分析五种大模型的表现，揭示其多轮对话可靠性不足的问题。**

- **链接: [https://arxiv.org/pdf/2603.11281](https://arxiv.org/pdf/2603.11281)**

> **作者:** Monica Munnangi; Saiph Savage
>
> **摘要:** Medical question-answering benchmarks predominantly evaluate single-turn exchanges, failing to capture the iterative, clarification-seeking nature of real patient consultations. We introduce ThReadMed-QA, a benchmark of 2,437 fully-answered patient-physician conversation threads extracted from r/AskDocs, comprising 8,204 question-answer pairs across up to 9 turns. Unlike prior work relying on simulated dialogues, adversarial prompts, or exam-style questions, ThReadMed-QA captures authentic patient follow-up questions and verified physician responses, reflecting how patients naturally seek medical information online. We evaluate five state-of-the-art LLMs -- GPT-5, GPT-4o, Claude Haiku, Gemini 2.5 Flash, and Llama 3.3 70B -- on a stratified test split of 238 conversations (948 QA pairs) using a calibrated LLM-as-a-judge rubric grounded in physician ground truth. Even the strongest model, GPT-5, achieves only 41.2% fully-correct responses. All five models degrade significantly from turn 0 to turn 2 (p < 0.001), with wrong-answer rates roughly tripling by the third turn. We identify a fundamental tension between single-turn capability and multi-turn reliability: models with the strongest initial performance (GPT-5: 75.2; Claude Haiku: 72.3 out of 100) exhibit the steepest declines by turn 2 (dropping 16.2 and 25.0 points respectively), while weaker models plateau or marginally improve. We introduce two metrics to quantify multi-turn failure modes: Conversational Consistency Score (CCS) and Error Propagation Rate (EPR). CCS reveals that nearly one in three Claude Haiku conversations swings between a fully correct and a completely wrong response within the same thread. EPR shows that a single wrong turn raises the probability of a subsequent wrong turn by 1.9-6.1x across all models.
>
---
#### [new 023] IndexCache: Accelerating Sparse Attention via Cross-Layer Index Reuse
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在提升稀疏注意力机制的效率。通过跨层索引复用减少计算量，提升推理速度。**

- **链接: [https://arxiv.org/pdf/2603.12201](https://arxiv.org/pdf/2603.12201)**

> **作者:** Yushi Bai; Qian Dong; Ting Jiang; Xin Lv; Zhengxiao Du; Aohan Zeng; Jie Tang; Juanzi Li
>
> **摘要:** Long-context agentic workflows have emerged as a defining use case for large language models, making attention efficiency critical for both inference speed and serving cost. Sparse attention addresses this challenge effectively, and DeepSeek Sparse Attention (DSA) is a representative production-grade solution: a lightweight lightning indexer selects the top-k most relevant tokens per query, reducing core attention from $O(L^2)$ to $O(Lk)$. However, the indexer itself retains $O(L^2)$ complexity and must run independently at every layer, despite the fact that the resulting top-k selections are highly similar across consecutive layers. We present IndexCache, which exploits this cross-layer redundancy by partitioning layers into a small set of Full layers that run their own indexers and a majority of Shared layers that simply reuse the nearest Full layer's top-k indices. We propose two complementary approaches to determine and optimize this configuration. Training-free IndexCache applies a greedy search algorithm that selects which layers to retain indexers by directly minimizing language modeling loss on a calibration set, requiring no weight updates. Training-aware IndexCache introduces a multi-layer distillation loss that trains each retained indexer against the averaged attention distributions of all layers it serves, enabling even simple interleaved patterns to match full-indexer accuracy. Experimental results on a 30B DSA model show that IndexCache can remove 75% of indexer computations with negligible quality degradation, achieving up to 1.82$\times$ prefill speedup and 1.48$\times$ decode speedup compared to standard DSA. These positive results are further confirmed by our preliminary experiments on the production-scale GLM-5 model (Figure 1).
>
---
#### [new 024] BLooP: Zero-Shot Abstractive Summarization using Large Language Models with Bigram Lookahead Promotion
- **分类: cs.CL**

- **简介: 该论文属于摘要生成任务，旨在解决大语言模型生成摘要时遗漏关键信息的问题。提出BLooP方法，在解码过程中通过双词查找提升摘要准确性。**

- **链接: [https://arxiv.org/pdf/2603.11415](https://arxiv.org/pdf/2603.11415)**

> **作者:** Varun Iyer; Cornelia Caragea
>
> **备注:** LREC 2026
>
> **摘要:** Abstractive summarization requires models to generate summaries that convey information in the source document. While large language models can generate summaries without fine-tuning, they often miss key details and include extraneous information. We propose BLooP (Bigram Lookahead Promotion), a simple training-free decoding intervention that encourages large language models (LLMs) to generate tokens that form bigrams from the source document. BLooP operates through a hash table lookup at each decoding step, requiring no training, fine-tuning, or model modification. We demonstrate improvements in ROUGE and BARTScore for Llama-3.1-8B-Instruct, Mistral-Nemo-Instruct-2407, and Gemma-2-9b-it on CNN/DM, CCSum, Multi-News, and SciTLDR. Human evaluation shows that BLooP significantly improves faithfulness without reducing readability. We make the code available at this https URL
>
---
#### [new 025] QChunker: Learning Question-Aware Text Chunking for Domain RAG via Multi-Agent Debate
- **分类: cs.CL**

- **简介: 该论文属于文本分块任务，旨在解决RAG中文本块语义不连贯和信息粒度问题。提出QChunker框架，通过多智能体辩论优化文本分块质量。**

- **链接: [https://arxiv.org/pdf/2603.11650](https://arxiv.org/pdf/2603.11650)**

> **作者:** Jihao Zhao; Daixuan Li; Pengfei Li; Shuaishuai Zu; Biao Qin; Hongyan Liu
>
> **摘要:** The effectiveness upper bound of retrieval-augmented generation (RAG) is fundamentally constrained by the semantic integrity and information granularity of text chunks in its knowledge base. To address these challenges, this paper proposes QChunker, which restructures the RAG paradigm from retrieval-augmentation to understanding-retrieval-augmentation. Firstly, QChunker models the text chunking as a composite task of text segmentation and knowledge completion to ensure the logical coherence and integrity of text chunks. Drawing inspiration from Hal Gregersen's "Questions Are the Answer" theory, we design a multi-agent debate framework comprising four specialized components: a question outline generator, text segmenter, integrity reviewer, and knowledge completer. This framework operates on the principle that questions serve as catalysts for profound insights. Through this pipeline, we successfully construct a high-quality dataset of 45K entries and transfer this capability to small language models. Additionally, to handle long evaluation chains and low efficiency in existing chunking evaluation methods, which overly rely on downstream QA tasks, we introduce a novel direct evaluation metric, ChunkScore. Both theoretical and experimental validations demonstrate that ChunkScore can directly and efficiently discriminate the quality of text chunks. Furthermore, during the text segmentation phase, we utilize document outlines for multi-path sampling to generate multiple candidate chunks and select the optimal solution employing ChunkScore. Extensive experimental results across four heterogeneous domains exhibit that QChunker effectively resolves aforementioned issues by providing RAG with more logically coherent and information-rich text chunks.
>
---
#### [new 026] Can Small Language Models Use What They Retrieve? An Empirical Study of Retrieval Utilization Across Model Scale
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究小模型（7B以下）在检索增强生成（RAG）中的信息利用问题。工作包括多模型、多检索条件实验及知识分割分析，发现小模型存在上下文利用瓶颈。**

- **链接: [https://arxiv.org/pdf/2603.11513](https://arxiv.org/pdf/2603.11513)**

> **作者:** Sanchit Pandey
>
> **备注:** 10 pages, 5 figures, planning to submit to arr march 2026. Code and evaluation data: this https URL . Earlier draft preprint available on Zenodo: this https URL (note: this arXiv submission is an updated draft)
>
> **摘要:** Retrieval augmented generation RAG is widely deployed to improve factual accuracy in language models yet it remains unclear whether smaller models of size 7B parameters or less can effectively utilize retrieved information. To investigate this question we evaluate five model sizes from 360M to 8B across three architecture families SmolLM2 Qwen2.5 and Llama 3.1 under four retrieval conditions including no retrieval BM25 dense retrieval using E5 large v2 and oracle retrieval where the retrieved passage is guaranteed to contain the answer. We introduce a parametric knowledge split that separates questions a model can already answer from those that require external knowledge which allows us to isolate utilization failure from retrieval quality failure. We find three main results. First even with oracle retrieval models of size 7B or smaller fail to extract the correct answer 85 to 100 percent of the time on questions they cannot answer alone which indicates a fundamental utilization bottleneck. Second adding retrieval context destroys 42 to 100 percent of answers the model previously knew suggesting a distraction effect driven by the presence of context rather than its quality. Third an error analysis of 2588 oracle failures shows that the dominant failure mode is irrelevant generation where the model ignores the provided context entirely. These patterns hold across multiple prompt templates and retrieval methods. The results indicate that for models below 7B parameters the main limitation of RAG is context utilization rather than retrieval quality and that deploying RAG at this scale can lead to a net negative trade off under standard evaluation conditions.
>
---
#### [new 027] Bielik-Minitron-7B: Compressing Large Language Models via Structured Pruning and Knowledge Distillation for the Polish Language
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型压缩任务，旨在减少大语言模型参数量以降低部署成本。通过结构化剪枝和知识蒸馏方法，将Bielik-11B-v3.0压缩至7.35B参数，保持90%性能。**

- **链接: [https://arxiv.org/pdf/2603.11881](https://arxiv.org/pdf/2603.11881)**

> **作者:** Remigiusz Kinas; Paweł Kiszczak; Sergio P. Perez; Krzysztof Ociepa; Łukasz Flis; Krzysztof Wróbel; Adrian Gwoździej
>
> **摘要:** This report details the creation of Bielik-Minitron-7B, a compressed 7.35B parameter version of the Bielik-11B-v3.0 model, specifically optimized for European languages. By leveraging a two-stage compression methodology inspired by the NVIDIA Minitron approach, we combined structured hybrid pruning and knowledge distillation to reduce the model's parameter count by 33.4%, from 11.04B to 7.35B. We utilized the NVIDIA Model Optimizer for structural pruning and the NVIDIA NeMo Framework for logit-based distillation for quality recovery. Following distillation, the model underwent a rigorous alignment pipeline consisting of Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO-P), and Reinforcement Learning (GRPO). Our final model successfully recovered approximately 90% of the baseline model's performance while providing up to 50% inference speedup. This approach demonstrates an efficient pathway to create language models for less-represented languages, preserving the original model quality while reducing inference deployment costs.
>
---
#### [new 028] Summarize Before You Speak with ARACH: A Training-Free Inference-Time Plug-In for Enhancing LLMs via Global Attention Reallocation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型性能。解决的问题是在不更新参数的情况下优化模型推理过程。工作是提出ARACH方法，通过调整注意力机制实现高效推理。**

- **链接: [https://arxiv.org/pdf/2603.11067](https://arxiv.org/pdf/2603.11067)**

> **作者:** Jingtao Wang; Yucong Wang; Jun Ding; Rui Cai; Xun Wang
>
> **摘要:** Large language models (LLMs) achieve remarkable performance, yet further gains often require costly training. This has motivated growing interest in post-training techniques-especially training-free approaches that improve models at inference time without updating weights. Most training-free methods treat the model as a black box and improve outputs via input/output-level interventions, such as prompt design and test-time scaling through repeated sampling, reranking/verification, or search. In contrast, they rarely offer a plug-and-play mechanism to intervene in a model's internal computation. We propose ARACH(Attention Reallocation via an Adaptive Context Hub), a training-free inference-time plug-in that augments LLMs with an adaptive context hub to aggregate context and reallocate attention. Extensive experiments across multiple language modeling tasks show consistent improvements with modest inference overhead and no parameter updates. Attention analyses further suggest that ARACH mitigates the attention sink phenomenon. These results indicate that engineering a model's internal computation offers a distinct inference-time strategy, fundamentally different from both prompt-based test-time methods and training-based post-training approaches.
>
---
#### [new 029] PersonaTrace: Synthesizing Realistic Digital Footprints with LLM Agents
- **分类: cs.CL**

- **简介: 该论文属于数据生成任务，旨在解决数字足迹数据稀缺问题。通过LLM代理生成真实可信的用户行为数据，提升数据多样性和实用性。**

- **链接: [https://arxiv.org/pdf/2603.11955](https://arxiv.org/pdf/2603.11955)**

> **作者:** Minjia Wang; Yunfeng Wang; Xiao Ma; Dexin Lv; Qifan Guo; Lynn Zheng; Benliang Wang; Lei Wang; Jiannan Li; Yongwei Xing; David Xu; Zheng Sun
>
> **备注:** EACL 2026 Industry Track
>
> **摘要:** Digital footprints (records of individuals' interactions with digital systems) are essential for studying behavior, developing personalized applications, and training machine learning models. However, research in this area is often hindered by the scarcity of diverse and accessible data. To address this limitation, we propose a novel method for synthesizing realistic digital footprints using large language model (LLM) agents. Starting from a structured user profile, our approach generates diverse and plausible sequences of user events, ultimately producing corresponding digital artifacts such as emails, messages, calendar entries, reminders, etc. Intrinsic evaluation results demonstrate that the generated dataset is more diverse and realistic than existing baselines. Moreover, models fine-tuned on our synthetic data outperform those trained on other synthetic datasets when evaluated on real-world out-of-distribution tasks.
>
---
#### [new 030] LLM-Assisted Causal Structure Disambiguation and Factor Extraction for Legal Judgment Prediction
- **分类: cs.CL**

- **简介: 该论文属于法律判决预测任务，旨在解决传统方法依赖统计相关性、缺乏因果建模的问题。通过结合大语言模型与因果推理，提升模型的准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.11446](https://arxiv.org/pdf/2603.11446)**

> **作者:** Yuzhi Liang; Lixiang Ma; Xinrong Zhu
>
> **摘要:** Mainstream methods for Legal Judgment Prediction (LJP) based on Pre-trained Language Models (PLMs) heavily rely on the statistical correlation between case facts and judgment results. This paradigm lacks explicit modeling of legal constituent elements and underlying causal logic, making models prone to learning spurious correlations and suffering from poor robustness. While introducing causal inference can mitigate this issue, existing causal LJP methods face two critical bottlenecks in real-world legal texts: inaccurate legal factor extraction with severe noise, and significant uncertainty in causal structure discovery due to Markov equivalence under sparse features. To address these challenges, we propose an enhanced causal inference framework that integrates Large Language Model (LLM) priors with statistical causal discovery. First, we design a coarse-to-fine hybrid extraction mechanism combining statistical sampling and LLM semantic reasoning to accurately identify and purify standard legal constituent elements. Second, to resolve structural uncertainty, we introduce an LLM-assisted causal structure disambiguation mechanism. By utilizing the LLM as a constrained prior knowledge base, we conduct probabilistic evaluation and pruning on ambiguous causal directions to generate legally compliant candidate causal graphs. Finally, a causal-aware judgment prediction model is constructed by explicitly constraining text attention intensity via the generated causal graphs. Extensive experiments on multiple benchmark datasets, including LEVEN , QA, and CAIL, demonstrate that our proposed method significantly outperforms state-of-the-art baselines in both predictive accuracy and robustness, particularly in distinguishing confusing charges.
>
---
#### [new 031] Speculative Decoding Scaling Laws (SDSL): Throughput Optimization Made Simple
- **分类: cs.CL; cs.IT; cs.LG**

- **简介: 该论文属于模型推理优化任务，旨在解决如何通过理论分析预测最优超参数以提升推理吞吐量的问题。工作包括建立理论模型，连接超参数与吞吐效率。**

- **链接: [https://arxiv.org/pdf/2603.11053](https://arxiv.org/pdf/2603.11053)**

> **作者:** Amirhossein Bozorgkhoo; Igor Molybog
>
> **摘要:** Speculative decoding is a technique that uses multiple language models to accelerate infer- ence. Previous works have used an experi- mental approach to optimize the throughput of the inference pipeline, which involves LLM training and can be costly. This study of spec- ulative decoding proposes a theory that ana- lytically connects the key hyperparameters of pre-trained LLMs to the throughput efficiency of a downstream SD-based inference system. The theory allows the prediction of throughput- optimal hyperparameters for the components of an inference system before their pre-training.
>
---
#### [new 032] MaterialFigBENCH: benchmark dataset with figures for evaluating college-level materials science problem-solving abilities of multimodal large language models
- **分类: cs.CL; cond-mat.mtrl-sci**

- **简介: 该论文提出MaterialFigBench，用于评估多模态大语言模型在材料科学问题中的图表理解能力，解决模型视觉推理与数值解析不足的问题。**

- **链接: [https://arxiv.org/pdf/2603.11414](https://arxiv.org/pdf/2603.11414)**

> **作者:** Michiko Yoshitake; Yuta Suzuki; Ryo Igarashi; Yoshitaka Ushiku; Keisuke Nagato
>
> **备注:** 27 pages, 4 tables, 6 figures
>
> **摘要:** We present MaterialFigBench, a benchmark dataset designed to evaluate the ability of multimodal large language models (LLMs) to solve university-level materials science problems that require accurate interpretation of figures. Unlike existing benchmarks that primarily rely on textual representations, MaterialFigBench focuses on problems in which figures such as phase diagrams, stress-strain curves, Arrhenius plots, diffraction patterns, and microstructural schematics are indispensable for deriving correct answers. The dataset consists of 137 free-response problems adapted from standard materials science textbooks, covering a broad range of topics including crystal structures, mechanical properties, diffusion, phase diagrams, phase transformations, and electronic properties of materials. To address unavoidable ambiguity in reading numerical values from images, expert-defined answer ranges are provided where appropriate. We evaluate several state-of-the-art multimodal LLMs, including ChatGPT and GPT models accessed via OpenAI APIs, and analyze their performance across problem categories and model versions. The results reveal that, although overall accuracy improves with model updates, current LLMs still struggle with genuine visual understanding and quantitative interpretation of materials science figures. In many cases, correct answers are obtained by relying on memorized domain knowledge rather than by reading the provided images. MaterialFigBench highlights persistent weaknesses in visual reasoning, numerical precision, and significant-digit handling, while also identifying problem types where performance has improved. This benchmark provides a systematic and domain-specific foundation for advancing multimodal reasoning capabilities in materials science and for guiding the development of future LLMs with stronger figure-based understanding.
>
---
#### [new 033] Strategic Navigation or Stochastic Search? How Agents and Humans Reason Over Document Collections
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于智能代理行为分析任务，旨在区分代理的策略性推理与随机搜索。通过构建MADQA基准和新评估协议，研究发现最佳代理在准确率上接近人类，但依赖暴力搜索，缺乏有效策略。**

- **链接: [https://arxiv.org/pdf/2603.12180](https://arxiv.org/pdf/2603.12180)**

> **作者:** Łukasz Borchmann; Jordy Van Landeghem; Michał Turski; Shreyansh Padarha; Ryan Othniel Kearns; Adam Mahdi; Niels Rogge; Clémentine Fourrier; Siwei Han; Huaxiu Yao; Artemis Llabrés; Yiming Xu; Dimosthenis Karatzas; Hao Zhang; Anupam Datta
>
> **摘要:** Multimodal agents offer a promising path to automating complex document-intensive workflows. Yet, a critical question remains: do these agents demonstrate genuine strategic reasoning, or merely stochastic trial-and-error search? To address this, we introduce MADQA, a benchmark of 2,250 human-authored questions grounded in 800 heterogeneous PDF documents. Guided by Classical Test Theory, we design it to maximize discriminative power across varying levels of agentic abilities. To evaluate agentic behaviour, we introduce a novel evaluation protocol measuring the accuracy-effort trade-off. Using this framework, we show that while the best agents can match human searchers in raw accuracy, they succeed on largely different questions and rely on brute-force search to compensate for weak strategic planning. They fail to close the nearly 20% gap to oracle performance, persisting in unproductive loops. We release the dataset and evaluation harness to help facilitate the transition from brute-force retrieval to calibrated, efficient reasoning.
>
---
#### [new 034] Compression Favors Consistency, Not Truth: When and Why Language Models Prefer Correct Information
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究语言模型为何在混合数据中偏好正确信息，属于自然语言处理任务。通过实验验证压缩与一致性原则，揭示“真理偏差”实为压缩压力所致。**

- **链接: [https://arxiv.org/pdf/2603.11749](https://arxiv.org/pdf/2603.11749)**

> **作者:** Konstantin Krestnikov
>
> **备注:** v1: initial release. Full code, synthetic datasets and experiments available at this https URL This work was done independently
>
> **摘要:** Why do language models sometimes prefer correct statements even when trained on mixed-quality data? We introduce the Compression--Consistency Principle: next-token prediction favors hypotheses that allow shorter and more internally consistent descriptions of the training data. Truth bias emerges only when false alternatives are structurally harder to compress. We test this using small GPT-2-style character-level transformers (3.5M--86M parameters) on synthetic math corpora with controlled mixtures of correct and incorrect rules. In the random-error setting, models strongly prefer correct completions in paired evaluation: 83.1% accuracy at balanced data and 67.0% even when correct rules appear in only 10% of the corpus. Replacing random errors with a coherent but mathematically incorrect rule system largely eliminates the preference (near-chance accuracy). In a more natural-language-like synthetic world, the effect is weaker but still present (57.7%). Additional experiments show that embedding verification steps can restore preference for correctness even at small scale, while increasing the number of consistent rules produces a graded improvement in accuracy. Our results suggest that what appears as a "truth bias" is largely a side effect of compression pressure and preference for internal consistency, rather than an intrinsic drive toward truth. Full code and data are available at this https URL.
>
---
#### [new 035] Multi-Task Reinforcement Learning for Enhanced Multimodal LLM-as-a-Judge
- **分类: cs.CL**

- **简介: 该论文属于多任务强化学习领域，旨在解决MLLM作为评判者在多任务场景下泛化能力不足的问题。通过联合优化多个任务，提升模型的一致性和与人类偏好的相关性。**

- **链接: [https://arxiv.org/pdf/2603.11665](https://arxiv.org/pdf/2603.11665)**

> **作者:** Junjie Wu; Xuan Kan; Zihao He; Shunwen Tan; Bo Pan; Kaitai Zhang
>
> **摘要:** Multimodal Large Language Models (MLLMs) have been widely adopted as MLLM-as-a-Judges due to their strong alignment with human judgment across various visual tasks. However, most existing judge models are optimized for single-task scenarios and struggle to generalize to diverse contexts, which is a critical requirement for reliable evaluation. To address this limitation, we propose Multi-Task Reinforcement Learning for MLLM-as-a-Judge (MT-RL-Judge), a framework that jointly optimizes the judge model across multiple tasks, leveraging the generalization capabilities of RL. Experimental results against several strong baselines demonstrate that MT-RL-Judge outperforms strong baselines in both judgment consistency and correlation with human preferences. Furthermore, our approach exhibits robust generalization on out-of-distribution tasks, further validating its effectiveness.
>
---
#### [new 036] SciMDR: Benchmarking and Advancing Scientific Multimodal Document Reasoning
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出SciMDR，解决科学多模态文档推理数据集构建难题，通过合成与再校准框架生成高质量QA对，提升模型在复杂科学任务上的表现。**

- **链接: [https://arxiv.org/pdf/2603.12249](https://arxiv.org/pdf/2603.12249)**

> **作者:** Ziyu Chen; Yilun Zhao; Chengye Wang; Rilyn Han; Manasi Patwardhan; Arman Cohan
>
> **摘要:** Constructing scientific multimodal document reasoning datasets for foundation model training involves an inherent trade-off among scale, faithfulness, and realism. To address this challenge, we introduce the synthesize-and-reground framework, a two-stage pipeline comprising: (1) Claim-Centric QA Synthesis, which generates faithful, isolated QA pairs and reasoning on focused segments, and (2) Document-Scale Regrounding, which programmatically re-embeds these pairs into full-document tasks to ensure realistic complexity. Using this framework, we construct SciMDR, a large-scale training dataset for cross-modal comprehension, comprising 300K QA pairs with explicit reasoning chains across 20K scientific papers. We further construct SciMDR-Eval, an expert-annotated benchmark to evaluate multimodal comprehension within full-length scientific workflows. Experiments demonstrate that models fine-tuned on SciMDR achieve significant improvements across multiple scientific QA benchmarks, particularly in those tasks requiring complex document-level reasoning.
>
---
#### [new 037] Cross-Context Review: Improving LLM Output Quality by Separating Production and Review Sessions
- **分类: cs.CL**

- **简介: 该论文属于模型输出质量提升任务，旨在解决LLM自检效果差的问题。通过分离生成与评审上下文，提出Cross-Context Review方法，显著提升错误检测效果。**

- **链接: [https://arxiv.org/pdf/2603.12123](https://arxiv.org/pdf/2603.12123)**

> **作者:** Tae-Eun Song
>
> **备注:** 10 pages, 2 figures, 8 tables
>
> **摘要:** Large language models struggle to catch errors in their own outputs when the review happens in the same session that produced them. This paper introduces Cross-Context Review (CCR), a straightforward method where the review is conducted in a fresh session with no access to the production conversation history. We ran a controlled experiment: 30 artifacts (code, technical documents, presentation scripts) with 150 injected errors, tested under four review conditions -- same-session Self-Review (SR), repeated Self-Review (SR2), context-aware Subagent Review (SA), and Cross-Context Review (CCR). Over 360 reviews, CCR reached an F1 of 28.6%, outperforming SR (24.6%, p=0.008, d=0.52), SR2 (21.7%, p<0.001, d=0.72), and SA (23.8%, p=0.004, d=0.57). The SR2 result matters most for interpretation: reviewing twice in the same session did not beat reviewing once (p=0.11), which rules out repetition as an explanation for CCR's advantage. The benefit comes from context separation itself. CCR works with any model, needs no infrastructure, and costs only one extra session.
>
---
#### [new 038] Long-Context Encoder Models for Polish Language Understanding
- **分类: cs.CL**

- **简介: 该论文针对波兰语长文本理解任务，解决传统编码器上下文窗口短的问题。通过两阶段训练和知识蒸馏，构建了支持8192词的高效模型，并在多个任务中取得最佳性能。**

- **链接: [https://arxiv.org/pdf/2603.12191](https://arxiv.org/pdf/2603.12191)**

> **作者:** Sławomir Dadas; Rafał Poświata; Marek Kozłowski; Małgorzata Grębowiec; Michał Perełkiewicz; Paweł Klimiuk; Przemysław Boruta
>
> **摘要:** While decoder-only Large Language Models (LLMs) have recently dominated the NLP landscape, encoder-only architectures remain a cost-effective and parameter-efficient standard for discriminative tasks. However, classic encoders like BERT are limited by a short context window, which is insufficient for processing long documents. In this paper, we address this limitation for the Polish by introducing a high-quality Polish model capable of processing sequences of up to 8192 tokens. The model was developed by employing a two-stage training procedure that involves positional embedding adaptation and full parameter continuous pre-training. Furthermore, we propose compressed model variants trained via knowledge distillation. The models were evaluated on 25 tasks, including the KLEJ benchmark, a newly introduced financial task suite (FinBench), and other classification and regression tasks, specifically those requiring long-document understanding. The results demonstrate that our model achieves the best average performance among Polish and multilingual models, significantly outperforming competitive solutions in long-context tasks while maintaining comparable quality on short texts.
>
---
#### [new 039] Performance Evaluation of Open-Source Large Language Models for Assisting Pathology Report Writing in Japanese
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学文本生成任务，旨在评估开源大语言模型在辅助日语病理科报告写作中的表现，解决模型在结构化报告、错别字修正及解释性文本生成方面的有效性问题。**

- **链接: [https://arxiv.org/pdf/2603.11597](https://arxiv.org/pdf/2603.11597)**

> **作者:** Masataka Kawai; Singo Sakashita; Shumpei Ishikawa; Shogo Watanabe; Anna Matsuoka; Mikio Sakurai; Yasuto Fujimoto; Yoshiyuki Takahara; Atsushi Ohara; Hirohiko Miyake; Genichiro Ishii
>
> **备注:** 9 pages (including bibliography), 2 figures, 6 tables
>
> **摘要:** The performance of large language models (LLMs) for supporting pathology report writing in Japanese remains unexplored. We evaluated seven open-source LLMs from three perspectives: (A) generation and information extraction of pathology diagnosis text following predefined formats, (B) correction of typographical errors in Japanese pathology reports, and (C) subjective evaluation of model-generated explanatory text by pathologists and clinicians. Thinking models and medical-specialized models showed advantages in structured reporting tasks that required reasoning and in typo correction. In contrast, preferences for explanatory outputs varied substantially across raters. Although the utility of LLMs differed by task, our findings suggest that open-source LLMs can be useful for assisting Japanese pathology report writing in limited but clinically relevant scenarios.
>
---
#### [new 040] Markovian Generation Chains in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究LLM迭代生成过程，属于自然语言处理任务。解决文本在多次LLM处理中的演化问题，通过实验和建模分析其多样性变化。**

- **链接: [https://arxiv.org/pdf/2603.11228](https://arxiv.org/pdf/2603.11228)**

> **作者:** Mingmeng Geng; Amr Mohamed; Guokan Shang; Michalis Vazirgiannis; Thierry Poibeau
>
> **摘要:** The widespread use of large language models (LLMs) raises an important question: how do texts evolve when they are repeatedly processed by LLMs? In this paper, we define this iterative inference process as Markovian generation chains, where each step takes a specific prompt template and the previous output as input, without including any prior memory. In iterative rephrasing and round-trip translation experiments, the output either converges to a small recurrent set or continues to produce novel sentences over a finite horizon. Through sentence-level Markov chain modeling and analysis of simulated data, we show that iterative process can either increase or reduce sentence diversity depending on factors such as the temperature parameter and the initial input sentence. These results offer valuable insights into the dynamics of iterative LLM inference and their implications for multi-agent LLM systems.
>
---
#### [new 041] One Supervisor, Many Modalities: Adaptive Tool Orchestration for Autonomous Queries
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出一种多模态查询处理框架，通过中央监督器协调不同工具，解决多模态AI部署效率低的问题。工作包括动态任务分解、自适应路由和成本优化。**

- **链接: [https://arxiv.org/pdf/2603.11545](https://arxiv.org/pdf/2603.11545)**

> **作者:** Mayank Saini Arit Kumar Bishwas
>
> **备注:** 19 pages, 3 figures
>
> **摘要:** We present an agentic AI framework for autonomous multimodal query processing that coordinates specialized tools across text, image, audio, video, and document modalities. A central Supervisor dynamically decomposes user queries, delegates subtasks to modality-appropriate tools (e.g., object detection, OCR, speech transcription), and synthesizes results through adaptive routing strategies rather than predetermined decision trees. For text-only queries, the framework uses learned routing via RouteLLM, while non-text paths use SLM-assisted modality decomposition. Evaluated on 2,847 queries across 15 task categories, our framework achieves 72% reduction in time-to-accurate-answer, 85% reduction in conversational rework, and 67% cost reduction compared to the matched hierarchical baseline while maintaining accuracy parity. These results demonstrate that intelligent centralized orchestration fundamentally improves multimodal AI deployment economics.
>
---
#### [new 042] Legal-DC: Benchmarking Retrieval-Augmented Generation for Legal Documents
- **分类: cs.CL**

- **简介: 该论文属于法律文档问答任务，旨在解决中文法律RAG系统评估不足与结构处理不善的问题。构建了Legal-DC基准数据集，提出LegRAG框架提升答案准确性与条款完整性。**

- **链接: [https://arxiv.org/pdf/2603.11772](https://arxiv.org/pdf/2603.11772)**

> **作者:** Yaocong Li; Qiang Lan; Leihan Zhang; Le Zhang
>
> **备注:** 20 pages, 4 figures, to be submitted to a conference/journal
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a promising technology for legal document consultation, yet its application in Chinese legal scenarios faces two key limitations: existing benchmarks lack specialized support for joint retriever-generator evaluation, and mainstream RAG systems often fail to accommodate the structured nature of legal provisions. To address these gaps, this study advances two core contributions: First, we constructed the Legal-DC benchmark dataset, comprising 480 legal documents (covering areas such as market regulation and contract management) and 2,475 refined question-answer pairs, each annotated with clause-level references, filling the gap for specialized evaluation resources in Chinese legal RAG. Second, we propose the LegRAG framework, which integrates legal adaptive indexing (clause-boundary segmentation) with a dual-path self-reflection mechanism to ensure clause integrity while enhancing answer accuracy. Third, we introduce automated evaluation methods for large language models to meet the high-reliability demands of legal retrieval scenarios. LegRAG outperforms existing state-of-the-art methods by 1.3% to 5.6% across key evaluation metrics. This research provides a specialized benchmark, practical framework, and empirical insights to advance the development of Chinese legal RAG systems. Our code and data are available at this https URL.
>
---
#### [new 043] SemBench: A Universal Semantic Framework for LLM Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SemBench，用于评估大语言模型的语义理解能力。解决跨语言评估难题，通过词典定义和句编码生成基准，无需人工标注数据。**

- **链接: [https://arxiv.org/pdf/2603.11687](https://arxiv.org/pdf/2603.11687)**

> **作者:** Mikel Zubillaga; Naiara Perez; Oscar Sainz; German Rigau
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** Recent progress in Natural Language Processing (NLP) has been driven by the emergence of Large Language Models (LLMs), which exhibit remarkable generative and reasoning capabilities. However, despite their success, evaluating the true semantic understanding of these models remains a persistent challenge. Traditional benchmarks such as Word-in-Context (WiC) effectively probe this capability, but their creation is resource-intensive and often limited to high-resource languages. In this paper, we introduce SemBench, a framework for automatically generating synthetic benchmarks that assess the semantic competence of LLMs using only dictionary sense definitions and a sentence encoder. This approach eliminates the need for curated example sentences, making it both scalable and language-independent. We evaluate SemBench in three languages (English, Spanish, and Basque) spanning different levels of linguistic resources, and across a wide range of LLMs. Our results show that rankings derived from SemBench strongly correlate with those obtained from standard WiC datasets. Furthermore, our analysis demonstrates that only a small number of examples is required to achieve stable and meaningful rankings. Overall, SemBench provides a lightweight, adaptable, and data-efficient framework for cross-lingual evaluation of semantic understanding in LLMs.
>
---
#### [new 044] Where Matters More Than What: Decoding-aligned KV Cache Compression via Position-aware Pseudo Queries
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理优化任务，解决KV缓存内存占用过高的问题。通过位置感知的伪查询构建观察窗口，实现更精准的token淘汰。**

- **链接: [https://arxiv.org/pdf/2603.11564](https://arxiv.org/pdf/2603.11564)**

> **作者:** Zhenxu Tian; Yi Su; Juntao Li; Min Zhang
>
> **摘要:** The Key-Value (KV) cache is crucial for efficient Large Language Models (LLMs) inference, but excessively long contexts drastically increase KV cache memory footprint. Existing KV cache compression methods typically rely on input-side attention patterns within a prompt observation window to estimate token importance during the prefill stage. They fail to preserve critical tokens for future generation since these assessments are not derived from the decoding process. Intuitively, an effective observation window should mirror the decoding-stage queries to accurately reflect which tokens the generation process will attend to. However, ground-truth decoding queries are inherently unavailable during inference. For constructing pseudo queries to approximate them, we find that positional information plays a more critical role than semantic content. Motivated by this insight, we propose decoding-aligned KV cache compression via position-aware pseudo queries (DapQ), a novel and lightweight eviction framework that leverages position-aware pseudo queries to simulate the output tokens, thereby establishing an effective observation window for importance assessment. It aligns closely with the actual generation context and enables precise token eviction. Extensive evaluations across multiple benchmarks and LLMs demonstrate that DapQ achieves superior performance, particularly under strict memory constraints (e.g., up to nearly lossless performance 99.5% on NIAH with 3% KV cache budgets).
>
---
#### [new 045] Tiny Aya: Bridging Scale and Multilingual Depth
- **分类: cs.CL**

- **简介: 该论文提出Tiny Aya，一个小型多语言模型，解决多语言AI效率与性能平衡问题，通过优化训练和区域适应提升翻译与生成质量。**

- **链接: [https://arxiv.org/pdf/2603.11510](https://arxiv.org/pdf/2603.11510)**

> **作者:** Alejandro R. Salamanca; Diana Abagyan; Daniel D'souza; Ammar Khairi; David Mora; Saurabh Dash; Viraat Aryabumi; Sara Rajaee; Mehrnaz Mofakhami; Ananya Sahu; Thomas Euyang; Brittawnya Prince; Madeline Smith; Hangyu Lin; Acyr Locatelli; Sara Hooker; Tom Kocmi; Aidan Gomez; Ivan Zhang; Phil Blunsom; Nick Frosst; Joelle Pineau; Beyza Ermis; Ahmet Üstün; Julia Kreutzer; Marzieh Fadaee
>
> **摘要:** Tiny Aya redefines what a small multilingual language model can achieve. Trained on 70 languages and refined through region-aware posttraining, it delivers state-of-the-art in translation quality, strong multilingual understanding, and high-quality target-language generation, all with just 3.35B parameters. The release includes a pretrained foundation model, a globally balanced instruction-tuned variant, and three region-specialized models targeting languages from Africa, South Asia, Europe, Asia-Pacific, and West Asia. This report details the training strategy, data composition, and comprehensive evaluation framework behind Tiny Aya, and presents an alternative scaling path for multilingual AI: one centered on efficiency, balanced performance across languages, and practical deployment.
>
---
#### [new 046] Streaming Translation and Transcription Through Speech-to-Text Causal Alignment
- **分类: cs.CL**

- **简介: 该论文提出Hikari模型，解决实时语音到文本的翻译与转录问题，通过概率等待机制实现端到端处理，提升质量与延迟平衡。**

- **链接: [https://arxiv.org/pdf/2603.11578](https://arxiv.org/pdf/2603.11578)**

> **作者:** Roman Koshkin; Jeon Haesung; Lianbo Liu; Hao Shi; Mengjie Zhao; Yusuke Fujita; Yui Sudo
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** Simultaneous machine translation (SiMT) has traditionally relied on offline machine translation models coupled with human-engineered heuristics or learned policies. We propose Hikari, a policy-free, fully end-to-end model that performs simultaneous speech-to-text translation and streaming transcription by encoding READ/WRITE decisions into a probabilistic WAIT token mechanism. We also introduce Decoder Time Dilation, a mechanism that reduces autoregressive overhead and ensures a balanced training distribution. Additionally, we present a supervised fine-tuning strategy that trains the model to recover from delays, significantly improving the quality-latency trade-off. Evaluated on English-to-Japanese, German, and Russian, Hikari achieves new state-of-the-art BLEU scores in both low- and high-latency regimes, outperforming recent baselines.
>
---
#### [new 047] Trust Oriented Explainable AI for Fake News Detection
- **分类: cs.CL**

- **简介: 该论文属于虚假新闻检测任务，旨在提升模型的可信度。通过应用XAI技术，增强模型的可解释性，同时保持高检测准确率。**

- **链接: [https://arxiv.org/pdf/2603.11778](https://arxiv.org/pdf/2603.11778)**

> **作者:** Krzysztof Siwek; Daniel Stankowski; Maciej Stodolski
>
> **备注:** 9 pages, 4 figures, 2 tables
>
> **摘要:** This article examines the application of Explainable Artificial Intelligence (XAI) in NLP based fake news detection and compares selected interpretability methods. The work outlines key aspects of disinformation, neural network architectures, and XAI techniques, with a focus on SHAP, LIME, and Integrated Gradients. In the experimental study, classification models were implemented and interpreted using these methods. The results show that XAI enhances model transparency and interpretability while maintaining high detection accuracy. Each method provides distinct explanatory value: SHAP offers detailed local attributions, LIME provides simple and intuitive explanations, and Integrated Gradients performs efficiently with convolutional models. The study also highlights limitations such as computational cost and sensitivity to parameterization. Overall, the findings demonstrate that integrating XAI with NLP is an effective approach to improving the reliability and trustworthiness of fake news detection systems.
>
---
#### [new 048] Stop Listening to Me! How Multi-turn Conversations Can Degrade Diagnostic Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于医疗诊断任务，研究多轮对话对诊断推理的影响。旨在解决多轮交互是否降低模型诊断准确性的问题，通过实验发现模型在多轮中易放弃正确判断。**

- **链接: [https://arxiv.org/pdf/2603.11394](https://arxiv.org/pdf/2603.11394)**

> **作者:** Kevin H. Guo; Chao Yan; Avinash Baidya; Katherine Brown; Xiang Gao; Juming Xiong; Zhijun Yin; Bradley A. Malin
>
> **摘要:** Patients and clinicians are increasingly using chatbots powered by large language models (LLMs) for healthcare inquiries. While state-of-the-art LLMs exhibit high performance on static diagnostic reasoning benchmarks, their efficacy across multi-turn conversations, which better reflect real-world usage, has been understudied. In this paper, we evaluate 17 LLMs across three clinical datasets to investigate how partitioning the decision-space into multiple simpler turns of conversation influences their diagnostic reasoning. Specifically, we develop a "stick-or-switch" evaluation framework to measure model conviction (i.e., defending a correct diagnosis or safe abstention against incorrect suggestions) and flexibility (i.e., recognizing a correct suggestion when it is introduced) across conversations. Our experiments reveal the conversation tax, where multi-turn interactions consistently degrade performance when compared to single-shot baselines. Notably, models frequently abandon initial correct diagnoses and safe abstentions to align with incorrect user suggestions. Additionally, several models exhibit blind switching, failing to distinguish between signal and incorrect suggestions.
>
---
#### [new 049] DeReason: A Difficulty-Aware Curriculum Improves Decoupled SFT-then-RL Training for General Reasoning
- **分类: cs.CL**

- **简介: 该论文属于语言模型训练任务，旨在解决SFT与RL协同训练中的数据分配问题。通过难度分层策略提升通用推理性能。**

- **链接: [https://arxiv.org/pdf/2603.11193](https://arxiv.org/pdf/2603.11193)**

> **作者:** Hanxu Hu; Yuxuan Wang; Maggie Huan; Jannis Vamvas; Yinya Huang; Zhijiang Guo; Rico Sennrich
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** Reinforcement learning with Verifiable Rewards (RLVR) has emerged as a powerful paradigm for eliciting reasoning capabilities in large language models, particularly in mathematics and coding. While recent efforts have extended this paradigm to broader general scientific (STEM) domains, the complex interplay between supervised fine-tuning (SFT) and RL in these contexts remains underexplored. In this paper, we conduct controlled experiments revealing a critical challenge: for general STEM domains, RL applied directly to base models is highly sample-inefficient and is consistently surpassed by supervised fine-tuning (SFT) on moderate-quality responses. Yet sequential SFT followed by RL can further improve performance, suggesting that the two stages play complementary roles, and that how training data is allocated between them matters. Therefore, we propose DeReason, a difficulty-based data decoupling strategy for general reasoning. DeReason partitions training data by reasoning intensity estimated via LLM-based scoring into reasoning-intensive and non-reasoning-intensive subsets. It allocates broad-coverage, non-reasoning-intensive problems to SFT to establish foundational domain knowledge, and reserves a focused subset of difficult problems for RL to cultivate complex reasoning. We demonstrate that this principled decoupling yields better performance than randomly splitting the data for sequential SFT and RL. Extensive experiments on general STEM and mathematical benchmarks demonstrate that our decoupled curriculum training significantly outperforms SFT-only, RL-only, and random-split baselines. Our work provides a systematic study of the interplay between SFT and RL for general reasoning, offering a highly effective and generalized post-training recipe.
>
---
#### [new 050] Artificial Intelligence for Sentiment Analysis of Persian Poetry
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感分析任务，旨在利用AI模型分析波斯诗歌的情感与韵律关系。研究对比了鲁米与帕尔文的诗作，验证了大语言模型在文学分析中的有效性。**

- **链接: [https://arxiv.org/pdf/2603.11254](https://arxiv.org/pdf/2603.11254)**

> **作者:** Arash Zargar; Abolfazl Moshiri; Mitra Shafaei; Shabnam Rahimi-Golkhandan; Mohamad Tavakoli-Targhi; Farzad Khalvati
>
> **摘要:** Recent advancements of the Artificial Intelligence (AI) have led to the development of large language models (LLMs) that are capable of understanding, analysing, and creating textual data. These language models open a significant opportunity in analyzing the literature and more specifically poetry. In the present work, we employ multiple Bidirectional encoder representations from transformers (BERT) and Generative Pre-trained Transformer (GPT) based language models to analyze the works of two prominent Persian poets: Jalal al-Din Muhammad Rumi (Rumi) and Parvin E'tesami. The main objective of this research is to investigate the capability of the modern language models in grasping complexities of the Persian poetry and explore potential correlations between the poems' sentiment and their meters. Our findings in this study indicates that GPT4o language model can reliably be used in analysis of Persian poetry. Furthermore, the results of our sentiment analysis revealed that in general, Rumi's poems express happier sentiments compared to Parvin E'tesami's poems. Furthermore, comparing the utilization of poetic meters highlighted Rumi's poems superiority in using meters to express a wider variety of sentiments. These findings are significant as they confirm that LLMs can be effectively applied in conducting computer-based semantic studies, where human interpretations are not required, and thereby significantly reducing potential biases in the analysis.
>
---
#### [new 051] Sparking Scientific Creativity via LLM-Driven Interdisciplinary Inspiration
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于科学创造力增强任务，旨在解决跨学科研究中创新不足的问题。提出Idea-Catalyst框架，通过跨领域知识整合提升创意推理。**

- **链接: [https://arxiv.org/pdf/2603.12226](https://arxiv.org/pdf/2603.12226)**

> **作者:** Priyanka Kargupta; Shuhaib Mehri; Dilek Hakkani-Tur; Jiawei Han
>
> **备注:** Code and dataset provided at this https URL
>
> **摘要:** Despite interdisciplinary research leading to larger and longer-term impact, most work remains confined to single-domain academic silos. Recent AI-based approaches to scientific discovery show promise for interdisciplinary research, but many prioritize rapidly designing experiments and solutions, bypassing the exploratory, collaborative reasoning processes that drive creative interdisciplinary breakthroughs. As a result, prior efforts largely prioritize automating scientific discovery rather than augmenting the reasoning processes that underlie scientific disruption. We present Idea-Catalyst, a novel framework that systematically identifies interdisciplinary insights to support creative reasoning in both humans and large language models. Starting from an abstract research goal, Idea-Catalyst is designed to assist the brainstorming stage, explicitly avoiding premature anchoring on specific solutions. The framework embodies key metacognitive features of interdisciplinary reasoning: (a) defining and assessing research goals, (b) awareness of a domain's opportunities and unresolved challenges, and (c) strategic exploration of interdisciplinary ideas based on impact potential. Concretely, Idea-Catalyst decomposes an abstract goal (e.g., improving human-AI collaboration) into core target-domain research questions that guide the analysis of progress and open challenges within that domain. These challenges are reformulated as domain-agnostic conceptual problems, enabling retrieval from external disciplines (e.g., Psychology, Sociology) that address analogous issues. By synthesizing and recontextualizing insights from these domains back into the target domain, Idea-Catalyst ranks source domains by their interdisciplinary potential. Empirically, this targeted integration improves average novelty by 21% and insightfulness by 16%, while remaining grounded in the original research problem.
>
---
#### [new 052] From Debate to Deliberation: Structured Collective Reasoning with Typed Epistemic Acts
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出DCI框架，解决多智能体系统在复杂推理中的协作问题。通过结构化推理流程提升决策质量，适用于非常规任务。**

- **链接: [https://arxiv.org/pdf/2603.11781](https://arxiv.org/pdf/2603.11781)**

> **作者:** Sunil Prakash
>
> **备注:** 26 pages, 6 tables, 2 figures, 2 listings
>
> **摘要:** Multi-agent LLM systems increasingly tackle complex reasoning, yet their interaction patterns remain limited to voting, unstructured debate, or pipeline orchestration. None model deliberation: a phased process where differentiated participants exchange typed reasoning moves, preserve disagreements, and converge on accountable outcomes. We introduce Deliberative Collective Intelligence (DCI), specifying four reasoning archetypes, 14 typed epistemic acts, a shared workspace, and DCI-CF, a convergent flow algorithm that guarantees termination with a structured decision packet containing the selected option, residual objections, minority report, and reopen conditions. We evaluate on 45 tasks across seven domains using Gemini 2.5 Flash. On non-routine tasks (n=40), DCI significantly improves over unstructured debate (+0.95, 95% CI [+0.41, +1.54]). DCI excels on hidden-profile tasks requiring perspective integration (9.56, highest of any system on any domain) while failing on routine decisions (5.39), confirming task-dependence. DCI produces 100% structured decision packets and 98% minority reports, artifacts absent from all baselines. However, DCI consumes ~62x single-agent tokens, and single-agent generation outperforms DCI on overall quality. DCI's contribution is not that more agents are better, but that consequential decisions benefit from deliberative structure when process accountability justifies the cost.
>
---
#### [new 053] Huntington Disease Automatic Speech Recognition with Biomarker Supervision
- **分类: cs.LG; cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，解决亨廷顿病患者语音识别问题。通过分析HD语音特征，提出改进模型和生物标志物辅助方法，提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2603.11168](https://arxiv.org/pdf/2603.11168)**

> **作者:** Charles L. Wang; Cady Chen; Ziwei Gong; Julia Hirschberg
>
> **摘要:** Automatic speech recognition (ASR) for pathological speech remains underexplored, especially for Huntington's disease (HD), where irregular timing, unstable phonation, and articulatory distortion challenge current models. We present a systematic HD-ASR study using a high-fidelity clinical speech corpus not previously used for end-to-end ASR training. We compare multiple ASR families under a unified evaluation, analyzing WER as well as substitution, deletion, and insertion patterns. HD speech induces architecture-specific error regimes, with Parakeet-TDT outperforming encoder-decoder and CTC baselines. HD-specific adaptation reduces WER from 6.99% to 4.95% and we also propose a method for using biomarker-based auxiliary supervision and analyze how error behavior is reshaped in severity-dependent ways rather than uniformly improving WER. We open-source all code and models.
>
---
#### [new 054] Linking Perception, Confidence and Accuracy in MLLMs
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究多模态大语言模型的置信度与准确性关系，解决模型无法正确判断自身不确定性的置信度校准问题。通过CDRL和CA-TTS方法提升模型感知能力和测试阶段表现。**

- **链接: [https://arxiv.org/pdf/2603.12149](https://arxiv.org/pdf/2603.12149)**

> **作者:** Yuetian Du; Yucheng Wang; Rongyu Zhang; Zhijie Xu; Boyu Yang; Ming Kong; Jie Liu; Qiang Zhu
>
> **备注:** Accepted by CVPR2026
>
> **摘要:** Recent advances in Multi-modal Large Language Models (MLLMs) have predominantly focused on enhancing visual perception to improve accuracy. However, a critical question remains unexplored: Do models know when they do not know? Through a probing experiment, we reveal a severe confidence miscalibration problem in MLLMs. To address this, we propose Confidence-Driven Reinforcement Learning (CDRL), which uses original-noise image pairs and a novel confidence-based reward to enhance perceptual sensitivity and robustly calibrate the model's confidence. Beyond training benefits, calibrated confidence enables more effective test-time scaling as a free lunch. We further propose Confidence-Aware Test-Time Scaling (CA-TTS), which dynamically coordinates Self-Consistency, Self-Reflection, and Visual Self-Check modules guided by confidence signals. An Expert Model acts in multiple roles (e.g., Planner, Critic, Voter) to schedule these modules and provide external verification. Our integrated framework establishes new state-of-the-art results with consistent 8.8% gains across four benchmarks. More ablation studies demonstrate the effectiveness of each module and scaling superiority.
>
---
#### [new 055] Speak or Stay Silent: Context-Aware Turn-Taking in Multi-Party Dialogue
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究多方对话中的上下文感知发言权切换任务，解决AI助手在多方对话中误判沉默时机的问题。通过构建数据集并改进模型训练方法提升判断准确性。**

- **链接: [https://arxiv.org/pdf/2603.11409](https://arxiv.org/pdf/2603.11409)**

> **作者:** Kratika Bhagtani; Mrinal Anand; Yu Chen Xu; Amit Kumar Singh Yadav
>
> **备注:** Submitted for review to Interspeech 2026
>
> **摘要:** Existing voice AI assistants treat every detected pause as an invitation to speak. This works in dyadic dialogue, but in multi-party settings, where an AI assistant participates alongside multiple speakers, pauses are abundant and ambiguous. An assistant that speaks on every pause becomes disruptive rather than useful. In this work, we formulate context-aware turn-taking: at every detected pause, given the full conversation context, our method decides whether the assistant should speak or stay silent. We introduce a benchmark of over 120K labeled conversations spanning three multi-party corpora. Evaluating eight recent large language models, we find that they consistently fail at context-aware turn-taking under zero-shot prompting. We then propose a supervised fine-tuning approach with reasoning traces, improving balanced accuracy by up to 23 percentage points. Our findings suggest that context-aware turn-taking is not an emergent capability; it must be explicitly trained.
>
---
#### [new 056] LongFlow: Efficient KV Cache Compression for Reasoning M
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决长输出导致的KV缓存占用过大的问题。提出LongFlow方法，通过高效重要性估计压缩KV缓存，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2603.11504](https://arxiv.org/pdf/2603.11504)**

> **作者:** Yi Su; Zhenxu Tian; Dan Qiao; Yuechi Zhou; Juntao Li; Min Zhang
>
> **摘要:** Recent reasoning models such as OpenAI-o1 and DeepSeek-R1 have shown strong performance on complex tasks including mathematical reasoning and code generation. However, this performance gain comes with substantially longer output sequences, leading to significantly increased deployment costs. In particular, long outputs require large KV caches, resulting in high memory consumption and severe bandwidth pressure during attention computation. Most existing KV cache optimization methods are designed for long-input, short-output scenarios and are ineffective for the long-output setting of reasoning models. Moreover, importance estimation in prior work is computationally expensive and becomes prohibitive when continuous re-evaluation is required during long generation. To address these challenges, we propose LongFlow, a KV cache compression method with an efficient importance estimation metric derived from an intermediate result of attention computation using only the current query. This design introduces negligible computational overhead and requires no auxiliary storage. We further develop a custom kernel that fuses FlashAttention, importance estimation, and token eviction into a single optimized operator, improving system-level efficiency. Experiments show that LongFlow achieves up to an 11.8 times throughput improvement with 80% KV cache compression with minimal impact on model accuracy.
>
---
#### [new 057] Expert Threshold Routing for Autoregressive Language Modeling with Dynamic Computation Allocation and Load Balancing
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于语言模型优化任务，解决TC-MoE动态计算分配和负载平衡问题。提出ET路由机制，通过阈值动态分配计算资源，提升效率。**

- **链接: [https://arxiv.org/pdf/2603.11535](https://arxiv.org/pdf/2603.11535)**

> **作者:** Hanchi Sun; Yixin Liu; Yonghui Wu; Lichao Sun
>
> **摘要:** Token-choice Mixture-of-Experts (TC-MoE) routes each token to a fixed number of experts, limiting dynamic computation allocation and requiring auxiliary losses to maintain load balance. We propose Expert Threshold (ET) routing, where each expert maintains an exponential moving average (EMA) threshold estimated from the global token distribution. At both training and inference, each token is independently routed to an expert if its score exceeds the expert's threshold, enabling dynamic computation allocation while achieving load balance without auxiliary losses. This fully causal mechanism eliminates dependence on other tokens in the batch, making it well-suited for autoregressive language modeling. In pretraining experiments scaling to 2.4B parameters on FineWeb-Edu, ET achieves 0.067 lower cross-entropy loss than TC-MoE, equivalent to reaching the same performance with 1.6$\times$ fewer tokens.
>
---
#### [new 058] Scaling Reasoning Efficiently via Relaxed On-Policy Distillation
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型压缩任务，解决知识蒸馏中的不稳定和负迁移问题。提出REOPOLD框架，通过奖励调整和动态采样提升训练效率与推理性能。**

- **链接: [https://arxiv.org/pdf/2603.11137](https://arxiv.org/pdf/2603.11137)**

> **作者:** Jongwoo Ko; Sara Abdali; Young Jin Kim; Tianyi Chen; Pashmina Cameron
>
> **备注:** Code will be available soon
>
> **摘要:** On-policy distillation is pivotal for transferring reasoning capabilities to capacity-constrained models, yet remains prone to instability and negative transfer. We show that on-policy distillation can be interpreted, both theoretically and empirically, as a form of policy optimization, where the teacher-student log-likelihood ratio acts as a token reward. From this insight, we introduce REOPOLD (Relaxed On-Policy Distillation) a framework that stabilizes optimization by relaxing the strict imitation constraints of standard on-policy distillation. Specifically, REOPOLD temperately and selectively leverages rewards from the teacher through mixture-based reward clipping, entropy-based token-level dynamic sampling, and a unified exploration-to-refinement training strategy. Empirically, REOPOLD surpasses its baselines with superior sample efficiency during training and enhanced test-time scaling at inference, across mathematical, visual, and agentic tool-use reasoning tasks. Specifically, REOPOLD outperforms recent RL approaches achieving 6.7~12x greater sample efficiency and enables a 7B student to match a 32B teacher in visual reasoning with a ~3.32x inference speedup.
>
---
#### [new 059] Beyond Polarity: Multi-Dimensional LLM Sentiment Signals for WTI Crude Oil Futures Return Prediction
- **分类: q-fin.ST; cs.CL**

- **简介: 该论文属于金融预测任务，旨在提升WTI原油期货收益预测。通过多维情感信号分析，结合大语言模型与传统模型，提升预测效果。**

- **链接: [https://arxiv.org/pdf/2603.11408](https://arxiv.org/pdf/2603.11408)**

> **作者:** Dehao Dai; Ding Ma; Dou Liu; Kerui Geng; Yiqing Wang
>
> **备注:** 28 pages, 4 figures, 4 tables
>
> **摘要:** Forecasting crude oil prices remains challenging because market-relevant information is embedded in large volumes of unstructured news and is not fully captured by traditional polarity-based sentiment measures. This paper examines whether multi-dimensional sentiment signals extracted by large language models improve the prediction of weekly WTI crude oil futures returns. Using energy-sector news articles from 2020 to 2025, we construct five sentiment dimensions covering relevance, polarity, intensity, uncertainty, and forwardness based on GPT-4o, Llama 3.2-3b, and two benchmark models, FinBERT and AlphaVantage. We aggregate article-level signals to the weekly level and evaluate their predictive performance in a classification framework. The best results are achieved by combining GPT-4o and FinBERT, suggesting that LLM-based and conventional financial sentiment models provide complementary predictive information. SHAP analysis further shows that intensity- and uncertainty-related features are among the most important predictors, indicating that the predictive value of news sentiment extends beyond simple polarity. Overall, the results suggest that multi-dimensional LLM-based sentiment measures can improve commodity return forecasting and support energy-market risk monitoring.
>
---
#### [new 060] Think While Watching: Online Streaming Segment-Level Memory for Multi-Turn Video Reasoning in Multimodal Large Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态大模型的在线视频推理任务，解决在线交互中记忆衰减和长距离依赖问题。提出Think While Watching框架，实现连续段级记忆保持。**

- **链接: [https://arxiv.org/pdf/2603.11896](https://arxiv.org/pdf/2603.11896)**

> **作者:** Lu Wang; Zhuoran Jin; Yupu Hao; Yubo Chen; Kang Liu; Yulong Ao; Jun Zhao
>
> **摘要:** Multimodal large language models (MLLMs) have shown strong performance on offline video understanding, but most are limited to offline inference or have weak online reasoning, making multi-turn interaction over continuously arriving video streams difficult. Existing streaming methods typically use an interleaved perception-generation paradigm, which prevents concurrent perception and generation and leads to early memory decay as streams grow, hurting long-range dependency modeling. We propose Think While Watching, a memory-anchored streaming video reasoning framework that preserves continuous segment-level memory during multi-turn interaction. We build a three-stage, multi-round chain-of-thought dataset and adopt a stage-matched training strategy, while enforcing strict causality through a segment-level streaming causal mask and streaming positional encoding. During inference, we introduce an efficient pipeline that overlaps watching and thinking and adaptively selects the best attention backend. Under both single-round and multi-round streaming input protocols, our method achieves strong results. Built on Qwen3-VL, it improves single-round accuracy by 2.6% on StreamingBench and by 3.79% on OVO-Bench. In the multi-round setting, it maintains performance while reducing output tokens by 56%. Code is available at: this https URL
>
---
#### [new 061] Hindsight-Anchored Policy Optimization: Turning Failure into Feedback in Sparse Reward Settings
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决稀疏奖励下的策略优化问题。提出HAPO方法，通过 hindsight 机制和自适应课程学习，提升策略性能。**

- **链接: [https://arxiv.org/pdf/2603.11321](https://arxiv.org/pdf/2603.11321)**

> **作者:** Yuning Wu; Ke Wang; Devin Chen; Kai Wei
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has emerged as a promising paradigm for post-training reasoning models. However, group-based methods such as Group Relative Policy Optimization (GRPO) face a critical dilemma in sparse-reward settings: pure Reinforcement Learning (RL) suffers from advantage collapse and high-variance gradient estimation, while mixed-policy optimization introduces persistent distributional bias. To resolve this dilemma, we introduce Hindsight-Anchored Policy Optimization (HAPO). HAPO employs the Synthetic Success Injection (SSI) operator, a hindsight mechanism that selectively anchors optimization to teacher demonstrations during failure. This injection is governed by a Thompson sampling-inspired gating mechanism, creating an autonomous, self-paced curriculum. Theoretically, we demonstrate that HAPO achieves \textit{asymptotic consistency}: by naturally annealing the teacher signal as the policy improves, HAPO recovers the unbiased on-policy gradient. This ensures off-policy guidance acts as a temporary scaffold rather than a persistent ceiling, enabling the model to surpass the limitations of static teacher forcing.
>
---
#### [new 062] Uni-ASR: Unified LLM-Based Architecture for Non-Streaming and Streaming Automatic Speech Recognition
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于自动语音识别任务，解决LLM与ASR融合在低延迟流式场景中的部署难题。提出Uni-ASR框架，实现非流式与流式识别的统一，提升流式识别准确率。**

- **链接: [https://arxiv.org/pdf/2603.11123](https://arxiv.org/pdf/2603.11123)**

> **作者:** Yinfeng Xia; Jian Tang; Junfeng Hou; Gaopeng Xu; Haitao Yao
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Although the deep integration of the Automatic Speech Recognition (ASR) system with Large Language Models (LLMs) has significantly improved accuracy, the deployment of such systems in low-latency streaming scenarios remains challenging. In this paper, we propose Uni-ASR, a unified framework based on LLMs that integrates both non-streaming and streaming speech recognition capabilities. We propose a joint training paradigm that enables the system to seamlessly transition between two recognition modes without any architectural modifications. Furthermore, we introduce a context-aware training paradigm and a co-designed fallback decoding strategy, which can enhance streaming recognition accuracy without introducing additional latency. The experimental results demonstrate that Uni-ASR not only achieves competitive performance within non-streaming mode, but also demonstrates strong effectiveness in streaming scenarios under diverse latency constraints.
>
---
#### [new 063] CR-Bench: Evaluating the Real-World Utility of AI Code Review Agents
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于代码审查任务，旨在解决AI代码审查代理评估不足的问题。提出CR-Bench和CR-Evaluator，用于更细致地评估代码审查代理的表现。**

- **链接: [https://arxiv.org/pdf/2603.11078](https://arxiv.org/pdf/2603.11078)**

> **作者:** Kristen Pereira; Neelabh Sinha; Rajat Ghosh; Debojyoti Dutta
>
> **摘要:** Recent advances in frontier large language models have enabled code review agents that operate in open-ended, reasoning-intensive settings. However, the lack of standardized benchmarks and granular evaluation protocols makes it difficult to assess behavior of code review agents beyond coarse success metrics, particularly for tasks where false positives are costly. To address this gap, we introduce CR-Bench, a benchmarking dataset, and CR-Evaluator, a fine-grained evaluation pipeline for code review agents. Using these tools, we conduct a preliminary study evaluating both a single-shot agent and a Reflexion-based agent across two frontier models. We find that code review agents can exhibit a low signal-to-noise ratio when designed to identify all hidden issues, obscuring true progress and developer productivity when measured solely by resolution rates. Our analysis identifies the hidden trade-off between issue resolution and spurious findings, revealing a frontier that constrains effective agent design. Together, CR-Bench and CR-Evaluator provide a timely foundation for studying and developing code review agents as LLM-based systems transition from controlled benchmarks to real-world software engineering workflows.
>
---
#### [new 064] LLMs Can Infer Political Alignment from Online Conversations
- **分类: cs.SI; cs.CL; cs.CY**

- **简介: 该论文属于隐私泄露任务，旨在解决LLMs从对话中推断政治倾向的问题。研究显示LLMs能有效利用非政治词汇进行预测，揭示潜在风险。**

- **链接: [https://arxiv.org/pdf/2603.11253](https://arxiv.org/pdf/2603.11253)**

> **作者:** Byunghwee Lee; Sangyeon Kim; Filippo Menczer; Yong-Yeol Ahn; Haewoon Kwak; Jisun An
>
> **备注:** 55 pages; 4 figures in the main text and 18 supplementary figures, 11 supplementary tables
>
> **摘要:** Due to the correlational structure in our traits such as identities, cultures, and political attitudes, seemingly innocuous preferences such as following a band or using a specific slang, can reveal private traits. This possibility, especially when combined with massive, public social data and advanced computational methods, poses a fundamental privacy risk. Given our increasing data exposure online and the rapid advancement of AI are increasing the misuse potential of such risk, it is therefore critical to understand capacity of large language models (LLMs) to exploit it. Here, using online discussions on this http URL and Reddit, we show that LLMs can reliably infer hidden political alignment, significantly outperforming traditional machine learning models. Prediction accuracy further improves as we aggregate multiple text-level inferences into a user-level prediction, and as we use more politics-adjacent domains. We demonstrate that LLMs leverage the words that can be highly predictive of political alignment while not being explicitly political. Our findings underscore the capacity and risks of LLMs for exploiting socio-cultural correlates.
>
---
#### [new 065] AnimeScore: A Preference-Based Dataset and Framework for Evaluating Anime-Like Speech Style
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文属于语音评估任务，旨在解决缺乏客观评价动漫风格语音标准的问题。通过构建AnimeScore框架，利用偏好排序进行自动评估。**

- **链接: [https://arxiv.org/pdf/2603.11482](https://arxiv.org/pdf/2603.11482)**

> **作者:** Joonyong Park; Jerry Li
>
> **摘要:** Evaluating 'anime-like' voices currently relies on costly subjective judgments, yet no standardized objective metric exists. A key challenge is that anime-likeness, unlike naturalness, lacks a shared absolute scale, making conventional Mean Opinion Score (MOS) protocols unreliable. To address this gap, we propose AnimeScore, a preference-based framework for automatic anime-likeness evaluation via pairwise ranking. We collect 15,000 pairwise judgments from 187 evaluators with free-form descriptions, and acoustic analysis reveals that perceived anime-likeness is driven by controlled resonance shaping, prosodic continuity, and deliberate articulation rather than simple heuristics such as high pitch. We show that handcrafted acoustic features reach a 69.3% AUC ceiling, while SSL-based ranking models achieve up to 90.8% AUC, providing a practical metric that can also serve as a reward signal for preference-based optimization of generative speech models.
>
---
#### [new 066] An Automatic Text Classification Method Based on Hierarchical Taxonomies, Neural Networks and Document Embedding: The NETHIC Tool
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于文本分类任务，旨在提升分类效果与效率。提出NETHIC工具，结合神经网络、层次分类体系和文档嵌入技术，优化分类性能。**

- **链接: [https://arxiv.org/pdf/2603.11770](https://arxiv.org/pdf/2603.11770)**

> **作者:** Luigi Lomasto; Rosario Di Florio; Andrea Ciapetti; Giuseppe Miscione; Giulia Ruggiero; Daniele Toti
>
> **备注:** ICEIS 2019 Conference
>
> **摘要:** This work describes an automatic text classification method implemented in a software tool called NETHIC, which takes advantage of the inner capabilities of highly-scalable neural networks combined with the expressiveness of hierarchical taxonomies. As such, NETHIC succeeds in bringing about a mechanism for text classification that proves to be significantly effective as well as efficient. The tool had undergone an experimentation process against both a generic and a domain-specific corpus, outputting promising results. On the basis of this experimentation, NETHIC has been now further refined and extended by adding a document embedding mechanism, which has shown improvements in terms of performance on the individual networks and on the whole hierarchical model.
>
---
#### [new 067] TopoBench: Benchmarking LLMs on Hard Topological Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出TopoBench，用于评估大语言模型在拓扑推理任务中的表现。任务是解决需要全局空间推理的网格谜题，解决的问题是模型在处理复杂空间约束时的局限性。工作包括构建基准、分析失败原因及测试缓解策略。**

- **链接: [https://arxiv.org/pdf/2603.12133](https://arxiv.org/pdf/2603.12133)**

> **作者:** Mayug Maniparambil; Nils Hoehing; Janak Kapuriya; Arjun Karuvally; Ellen Rushe; Anthony Ventresque; Noel O'Connor; Fergal Reid
>
> **备注:** Accepted, Workshop on Logical Reasoning of Large Language Models at ICLR 2026
>
> **摘要:** Solving topological grid puzzles requires reasoning over global spatial invariants such as connectivity, loop closure, and region symmetry and remains challenging for even the most powerful large language models (LLMs). To study these abilities under controlled settings, we introduce TopoBench, a benchmark of six puzzle families across three difficulty levels. We evaluate strong reasoning LLMs on TopoBench and find that even frontier models solve fewer than one quarter of hard instances, with two families nearly unsolved. To investigate whether these failures stem from reasoning limitations or from difficulty extracting and maintaining spatial constraints, we annotate 750 chain of thought traces with an error taxonomy that surfaces four candidate causal failure modes, then test them with targeted interventions simulating each error type. These interventions show that certain error patterns like premature commitment and constraint forgetting have a direct impact on the ability to solve the puzzle, while repeated reasoning is a benign effect of search. Finally we study mitigation strategies including prompt guidance, cell-aligned grid representations and tool-based constraint checking, finding that the bottleneck lies in extracting constraints from spatial representations and not in reasoning over them. Code and data are available at this http URL.
>
---
#### [new 068] XSkill: Continual Learning from Experience and Skills in Multimodal Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于多模态智能体的持续学习任务，旨在解决开放场景下工具使用效率低和策略不灵活的问题。提出XSkill框架，通过经验与技能双流机制实现持续学习与优化。**

- **链接: [https://arxiv.org/pdf/2603.12056](https://arxiv.org/pdf/2603.12056)**

> **作者:** Guanyu Jiang; Zhaochen Su; Xiaoye Qu; Yi R.; Fung
>
> **摘要:** Multimodal agents can now tackle complex reasoning tasks with diverse tools, yet they still suffer from inefficient tool use and inflexible orchestration in open-ended settings. A central challenge is enabling such agents to continually improve without parameter updates by learning from past trajectories. We identify two complementary forms of reusable knowledge essential for this goal: experiences, providing concise action-level guidance for tool selection and decision making, and skills, providing structured task-level guidance for planning and tool use. To this end, we propose XSkill, a dual-stream framework for continual learning from experience and skills in multimodal agents. XSkill grounds both knowledge extraction and retrieval in visual observations. During accumulation, XSkill distills and consolidates experiences and skills from multi-path rollouts via visually grounded summarization and cross-rollout critique. During inference, it retrieves and adapts this knowledge to the current visual context and feeds usage history back into accumulation to form a continual learning loop. Evaluated on five benchmarks across diverse domains with four backbone models, XSkill consistently and substantially outperforms both tool-only and learning-based baselines. Further analysis reveals that the two knowledge streams play complementary roles in influencing the reasoning behaviors of agents and show superior zero-shot generalization.
>
---
#### [new 069] Fractional Rotation, Full Potential? Investigating Performance and Convergence of Partial RoPE
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究Transformer中部分应用RoPE对模型性能和收敛的影响，旨在平衡效率与稳定性。任务为模型优化，解决如何减少内存消耗同时保持效果的问题。工作包括系统分析不同RoPE比例的影响。**

- **链接: [https://arxiv.org/pdf/2603.11611](https://arxiv.org/pdf/2603.11611)**

> **作者:** Mohammad Aflah Khan; Krishna P. Gummadi; Manish Gupta; Abhilasha Ravichander
>
> **摘要:** Rotary Positional Embedding (RoPE) is a common choice in transformer architectures for encoding relative positional information. Although earlier work has examined omitting RoPE in specific layers, the effect of varying the fraction of hidden dimensions that receive rotary transformations remains largely unexplored. This design choice can yield substantial memory savings, which becomes especially significant at long context lengths. We find up to 10x memory savings over the standard RoPE cache, while achieving comparable final loss. In this work, we present a systematic study examining the impact of partial RoPE on training dynamics and convergence across architectures and datasets. Our findings uncover several notable patterns: (1) applying RoPE to only a small fraction of dimensions (around 10%) achieves convergence comparable to using full RoPE; (2) these trends hold consistently across model size, sequence lengths and datasets of varying quality and architectures, with higher-quality data resulting in lower overall loss and similar benchmark performance; and (3) some models trained with NoPE (No Positional Encoding) showcase unstable learning trajectories, which can be alleviated through minimal RoPE application or QK-Norm which converges to a higher loss. Together, these results offer practical guidance for model designers aiming to balance efficiency and training stability, while emphasizing the previously overlooked importance of partial RoPE.
>
---
#### [new 070] Enhancing Value Alignment of LLMs with Multi-agent system and Combinatorial Fusion
- **分类: cs.MA; cs.CL**

- **简介: 该论文属于人工智能伦理任务，旨在解决LLMs与人类价值观对齐的问题。通过多智能体融合方法提升模型响应的伦理多样性与准确性。**

- **链接: [https://arxiv.org/pdf/2603.11126](https://arxiv.org/pdf/2603.11126)**

> **作者:** Yuanhong Wu; Djallel Bouneffouf; D. Frank Hsu
>
> **备注:** 5 pages, 3 figures, accepted to 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)
>
> **摘要:** Aligning large language models (LLMs) with human values is a central challenge for ensuring trustworthy and safe deployment. While existing methods such as Reinforcement Learning from Human Feedback (RLHF) and its variants have improved alignment, they often rely on a single evaluator or narrowly defined reward signals, limiting their ability to capture ethical pluralism. In this work, we propose the Value Alignment System using Combinatorial Fusion Analysis (VAS-CFA), a framework that operationalizes multi-agent fusion alignment. It instantiates multiple moral agents, each fine-tuned to represent a distinct normative perspective, and fuses their outputs using CFA with both rank- and score-based aggregation. This design leverages cognitive diversity, between agents, to mitigate conflicts and redundancies across multiple agents, producing responses that better reflect human values. Empirical evaluation demonstrates that VAS-CFA outperforms both single agent baselines and prior aggregation approaches on standard metrics, showing that multi-agent fusion provides a robust and effective mechanism for advancing value alignment in LLMs.
>
---
#### [new 071] EndoCoT: Scaling Endogenous Chain-of-Thought Reasoning in Diffusion Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出EndoCoT框架，解决扩散模型中文本编码器推理深度不足和引导不变的问题，通过迭代思考和终端对齐提升复杂任务处理能力。**

- **链接: [https://arxiv.org/pdf/2603.12252](https://arxiv.org/pdf/2603.12252)**

> **作者:** Xuanlang Dai; Yujie Zhou; Long Xing; Jiazi Bu; Xilin Wei; Yuhong Liu; Beichen Zhang; Kai Chen; Yuhang Zang
>
> **备注:** 23 pages, 18 figures
>
> **摘要:** Recently, Multimodal Large Language Models (MLLMs) have been widely integrated into diffusion frameworks primarily as text encoders to tackle complex tasks such as spatial reasoning. However, this paradigm suffers from two critical limitations: (i) MLLMs text encoder exhibits insufficient reasoning depth. Single-step encoding fails to activate the Chain-of-Thought process, which is essential for MLLMs to provide accurate guidance for complex tasks. (ii) The guidance remains invariant during the decoding process. Invariant guidance during decoding prevents DiT from progressively decomposing complex instructions into actionable denoising steps, even with correct MLLM encodings. To this end, we propose Endogenous Chain-of-Thought (EndoCoT), a novel framework that first activates MLLMs' reasoning potential by iteratively refining latent thought states through an iterative thought guidance module, and then bridges these states to the DiT's denoising process. Second, a terminal thought grounding module is applied to ensure the reasoning trajectory remains grounded in textual supervision by aligning the final state with ground-truth answers. With these two components, the MLLM text encoder delivers meticulously reasoned guidance, enabling the DiT to execute it progressively and ultimately solve complex tasks in a step-by-step manner. Extensive evaluations across diverse benchmarks (e.g., Maze, TSP, VSP, and Sudoku) achieve an average accuracy of 92.1%, outperforming the strongest baseline by 8.3 percentage points.
>
---
#### [new 072] Meta-Reinforcement Learning with Self-Reflection for Agentic Search
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出MR-Search，属于强化学习任务，解决agentic search中探索效率低的问题。通过自反思机制和跨回合探索，提升测试时的泛化能力与效果。**

- **链接: [https://arxiv.org/pdf/2603.11327](https://arxiv.org/pdf/2603.11327)**

> **作者:** Teng Xiao; Yige Yuan; Hamish Ivison; Huaisheng Zhu; Faeze Brahman; Nathan Lambert; Pradeep Dasigi; Noah A. Smith; Hannaneh Hajishirzi
>
> **备注:** 23 pages, Preprint
>
> **摘要:** This paper introduces MR-Search, an in-context meta reinforcement learning (RL) formulation for agentic search with self-reflection. Instead of optimizing a policy within a single independent episode with sparse rewards, MR-Search trains a policy that conditions on past episodes and adapts its search strategy across episodes. MR-Search learns to learn a search strategy with self-reflection, allowing search agents to improve in-context exploration at test-time. Specifically, MR-Search performs cross-episode exploration by generating explicit self-reflections after each episode and leveraging them as additional context to guide subsequent attempts, thereby promoting more effective exploration during test-time. We further introduce a multi-turn RL algorithm that estimates a dense relative advantage at the turn level, enabling fine-grained credit assignment on each episode. Empirical results across various benchmarks demonstrate the advantages of MR-Search over baselines based RL, showing strong generalization and relative improvements of 9.2% to 19.3% across eight benchmarks. Our code and data are available at this https URL.
>
---
#### [new 073] Examining Reasoning LLMs-as-Judges in Non-Verifiable LLM Post-Training
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 论文研究在非验证领域使用推理型LLM作为裁判的对齐方法，旨在解决政策训练中裁判有效性问题。通过对比分析，发现推理裁判能提升性能，但也可能生成欺骗性输出。**

- **链接: [https://arxiv.org/pdf/2603.12246](https://arxiv.org/pdf/2603.12246)**

> **作者:** Yixin Liu; Yue Yu; DiJia Su; Sid Wang; Xuewei Wang; Song Jiang; Bo Liu; Arman Cohan; Yuandong Tian; Zhengxing Chen
>
> **摘要:** Reasoning LLMs-as-Judges, which can benefit from inference-time scaling, provide a promising path for extending the success of reasoning models to non-verifiable domains where the output correctness/quality cannot be directly checked. However, while reasoning judges have shown better performance on static evaluation benchmarks, their effectiveness in actual policy training has not been systematically examined. Therefore, we conduct a rigorous study to investigate the actual impact of non-reasoning and reasoning judges in reinforcement-learning-based LLM alignment. Our controlled synthetic setting, where a "gold-standard" judge (gpt-oss-120b) provides preference annotations to train smaller judges, reveals key differences between non-reasoning and reasoning judges: non-reasoning judges lead to reward hacking easily, while reasoning judges can lead to policies that achieve strong performance when evaluated by the gold-standard judge. Interestingly, we find that the reasoning-judge-trained policies achieve such strong performance by learning to generate highly effective adversarial outputs that can also score well on popular benchmarks such as Arena-Hard by deceiving other LLM-judges. Combined with our further analysis, our study highlights both important findings and room for improvements for applying (reasoning) LLM-judges in non-verifiable LLM post-training.
>
---
#### [new 074] OpenSanctions Pairs: Large-Scale Entity Matching with LLMs
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出OpenSanctions Pairs数据集，用于实体匹配任务，解决跨国制裁数据中的重复识别问题，对比了规则系统与LLMs的性能。**

- **链接: [https://arxiv.org/pdf/2603.11051](https://arxiv.org/pdf/2603.11051)**

> **作者:** Chandler Smith; Magnus Sesodia; Friedrich Lindenberg; Christian Schroeder de Witt
>
> **摘要:** We release OpenSanctions Pairs, a large-scale entity matching benchmark derived from real-world international sanctions aggregation and analyst deduplication. The dataset contains 755,540 labeled pairs spanning 293 heterogeneous sources across 31 countries, with multilingual and cross-script names, noisy and missing attributes, and set-valued fields typical of compliance workflows. We benchmark a production rule-based matcher (nomenklatura RegressionV1 algorithm) against open- and closed-source LLMs in zero- and few-shot settings. Off-the-shelf LLMs substantially outperform the production rule-based baseline (91.33\% F1), reaching up to 98.95\% F1 (GPT-4o) and 98.23\% F1 with a locally deployable open model (DeepSeek-R1-Distill-Qwen-14B). DSPy MIPROv2 prompt optimization yields consistent but modest gains, while adding in-context examples provides little additional benefit and can degrade performance. Error analysis shows complementary failure modes: the rule-based system over-matches (high false positives), whereas LLMs primarily fail on cross-script transliteration and minor identifier/date inconsistencies. These results indicate that pairwise matching performance is approaching a practical ceiling in this setting, and motivate shifting effort toward pipeline components such as blocking, clustering, and uncertainty-aware review. Code available at this https URL
>
---
#### [new 075] Frequency-Modulated Visual Restoration for Matryoshka Large Multimodal Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态模型任务，解决视觉token减少导致语义丢失的问题。提出FMVR方法，通过频率调制恢复视觉语义，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.11220](https://arxiv.org/pdf/2603.11220)**

> **作者:** Qingtao Pan; Zhihao Dou; Shuo Li
>
> **摘要:** Large Multimodal Models (LMMs) struggle to adapt varying computational budgets due to numerous visual tokens. Previous methods attempted to reduce the number of visual tokens before or within LLMs. However, these strategies inevitably result in the loss of visual semantic. To address these issues, we introduce FMVR, a plug-and-play and extremely simple Frequency-Modulated Visual Restoration strategy to boost the reasoning ability of LMMs under visual token reduction. Specifically, FMVR disentangles the visual representation of fewer visual tokens into low- and high-frequency components through AvgPool and MaxPool. The derived frequencies are subsequently modulated using lightweight learnable parameters. The high-frequency from AvgPool acts as a saliency filter to enhance saliency visual semantics, while the low-frequency from MaxPool acts as an anti-saliency filter to strengthen weak visual semantics. It enables the preservation of visual semantics dominated by few visual tokens and the restoration of diluted visual semantics. Additionally, we inject FMVR into Matryoshka Representation Learning to learn coarse-to-fine visual token sets, thus enabling to elastically adjust the number of visual tokens during inference while maintaining comparable performance. Experiments across 10 image-based and 4 video-based bench marks demonstrate that FMVR-LLaVA reduce the FLOPs of LLaVA-1.5-7B by 89%, while maintaining almost 100% of the original accuracy. The code will be open.
>
---
#### [new 076] OSCBench: Benchmarking Object State Change in Text-to-Video Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于文本到视频生成任务，旨在解决对象状态变化（OSC）评估问题。提出OSCBench基准，评估模型在不同场景下的OSC性能。**

- **链接: [https://arxiv.org/pdf/2603.11698](https://arxiv.org/pdf/2603.11698)**

> **作者:** Xianjing Han; Bin Zhu; Shiqi Hu; Franklin Mingzhe Li; Patrick Carrington; Roger Zimmermann; Jingjing Chen
>
> **备注:** Project page: this https URL
>
> **摘要:** Text-to-video (T2V) generation models have made rapid progress in producing visually high-quality and temporally coherent videos. However, existing benchmarks primarily focus on perceptual quality, text-video alignment, or physical plausibility, leaving a critical aspect of action understanding largely unexplored: object state change (OSC) explicitly specified in the text prompt. OSC refers to the transformation of an object's state induced by an action, such as peeling a potato or slicing a lemon. In this paper, we introduce OSCBench, a benchmark specifically designed to assess OSC performance in T2V models. OSCBench is constructed from instructional cooking data and systematically organizes action-object interactions into regular, novel, and compositional scenarios to probe both in-distribution performance and generalization. We evaluate six representative open-source and proprietary T2V models using both human user study and multimodal large language model (MLLM)-based automatic evaluation. Our results show that, despite strong performance on semantic and scene alignment, current T2V models consistently struggle with accurate and temporally consistent object state changes, especially in novel and compositional settings. These findings position OSC as a key bottleneck in text-to-video generation and establish OSCBench as a diagnostic benchmark for advancing state-aware video generation models.
>
---
#### [new 077] From Control to Foresight: Simulation as a New Paradigm for Human-Agent Collaboration
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于人机协作任务，旨在解决用户缺乏决策 foresight 的问题。提出“模拟闭环”框架，通过模拟未来轨迹提升协作效果。**

- **链接: [https://arxiv.org/pdf/2603.11677](https://arxiv.org/pdf/2603.11677)**

> **作者:** Gaole He; Brian Y. Lim
>
> **备注:** CHI 2026 Workshop on Human-Agent Collaboration
>
> **摘要:** Large Language Models (LLMs) are increasingly used to power autonomous agents for complex, multi-step tasks. However, human-agent interaction remains pointwise and reactive: users approve or correct individual actions to mitigate immediate risks, without visibility into subsequent consequences. This forces users to mentally simulate long-term effects, a cognitively demanding and often inaccurate process. Users have control over individual steps but lack the foresight to make informed decisions. We argue that effective collaboration requires foresight, not just control. We propose simulation-in-the-loop, an interaction paradigm that enables users and agents to explore simulated future trajectories before committing to decisions. Simulation transforms intervention from reactive guesswork into informed exploration, while helping users discover latent constraints and preferences along the way. This perspective paper characterizes the limitations of current paradigms, introduces a conceptual framework for simulation-based collaboration, and illustrates its potential through concrete human-agent collaboration scenarios.
>
---
#### [new 078] Chem4DLLM: 4D Multimodal LLMs for Chemical Dynamics Understanding
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出ChemDU任务，解决化学动态理解问题，通过4D分子轨迹生成自然语言解释，并构建了相应数据集和模型Chem4DLLM。**

- **链接: [https://arxiv.org/pdf/2603.11924](https://arxiv.org/pdf/2603.11924)**

> **作者:** Xinyu Li; Zhen Zhang; Qi Chen; Anton van den Hengel; Lina Yao; Javen Qinfeng Shi
>
> **备注:** 18 pages
>
> **摘要:** Existing chemical understanding tasks primarily rely on static molecular representations, limiting their ability to model inherently dynamic phenomena such as bond breaking or conformational changes, which are essential for a chemist to understand chemical reactions. To address this gap, we introduce Chemical Dynamics Understanding (ChemDU), a new task that translates 4D molecular trajectories into interpretable natural-language explanations. ChemDU focuses on fundamental dynamic scenarios, including gas-phase and catalytic reactions, and requires models to reason about key events along molecular trajectories, such as bond formation and dissociation, and to generate coherent, mechanistically grounded narratives. To benchmark this capability, we construct Chem4DBench, the first dataset pairing 4D molecular trajectories with expert-authored explanations across these settings. We further propose Chem4DLLM, a unified model that integrates an equivariant graph encoder with a pretrained large language model to explicitly capture molecular geometry and rotational dynamics. We hope that ChemDU, together with Chem4DBench and Chem4DLLM, will stimulate further research in dynamic chemical understanding and multimodal scientific reasoning.
>
---
#### [new 079] Resurfacing Paralinguistic Awareness in Large Audio Language Models
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **简介: 该论文属于语音语言模型任务，旨在解决LALMs忽视语气等副语言信息的问题。通过分析层结构并提出增强微调协议，提升模型对副语言线索的感知能力。**

- **链接: [https://arxiv.org/pdf/2603.11947](https://arxiv.org/pdf/2603.11947)**

> **作者:** Hao Yang; Minghan Wang; Tongtong Wu; Lizhen Qu; Ehsan Shareghi; Gholamreza Haffari
>
> **备注:** Submitted to Interspeech 2026
>
> **摘要:** Large Audio Language Models (LALMs) have expanded the interaction with human to speech modality, which introduces great interactive potential, due to the paralinguistic cues implicitly indicating the user context. However, building on the current content-centred paradigm, LALMs usually neglect such paralinguistic cues and respond solely based on query content. In this work, to resurface the paralinguistic awareness in LALMs, we introduce five diverse layer-wise analyses to jointly identify paralinguistic layers and semantic understanding layers. Based on these insights, we propose a paralinguistic-enhanced fine-tuning (PE-FT) protocol accordingly to equip LALMs with paralinguistic-aware capabilities, including (1) selective-layer fine-tuning, and (2) an auxiliary dual-level classification head. Our experiments demonstrate that PE-FT protocol efficiently and effectively resurfaces the paralinguistic awareness, even surpassing the performance of the all-layer fine-tuning strategy.
>
---
#### [new 080] Human-Centred LLM Privacy Audits: Findings and Frictions
- **分类: cs.HC; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于隐私审计任务，旨在解决用户无法检查大语言模型对其个人信息的关联问题。研究提出工具LMP2，通过用户实验验证方法有效性，并指出隐私评估中的挑战与改进方向。**

- **链接: [https://arxiv.org/pdf/2603.12094](https://arxiv.org/pdf/2603.12094)**

> **作者:** Dimitri Staufer; Kirsten Morehouse; David Hartmann; Bettina Berendt
>
> **摘要:** Large language models (LLMs) learn statistical associations from massive training corpora and user interactions, and deployed systems can surface or infer information about individuals. Yet people lack practical ways to inspect what a model associates with their name. We report interim findings from an ongoing study and introduce LMP2, a browser-based self-audit tool. In two user studies ($N_{total}{=}458$), GPT-4o predicts 11 of 50 features for everyday people with $\ge$60\% accuracy, and participants report wanting control over LLM-generated associations despite not considering all outputs privacy violations. To validate our probing method, we evaluate eight LLMs on public figures and non-existent names, observing clear separation between stable name-conditioned associations and model defaults. Our findings also contribute to exposing a broader generative AI evaluation crisis: when outputs are probabilistic, context-dependent, and user-mediated through elicitation, what model--individual associations even include is under-specified and operationalisation relies on crafting probes and metrics that are hard to validate or compare. To move towards reliable, actionable human-centred LLM privacy audits, we identify nine frictions that emerged in our study and offer recommendations for future work and the design of human-centred LLM privacy audits.
>
---
## 更新

#### [replaced 001] Swiss Parliaments Corpus Re-Imagined (SPC_R): Enhanced Transcription with RAG-based Correction and Predicted BLEU
- **分类: cs.CL**

- **简介: 该论文属于语音转文本任务，旨在提升低资源领域语料质量。通过ASR、LLM修正和数据过滤，构建高质量的瑞士议会语料库。**

- **链接: [https://arxiv.org/pdf/2506.07726](https://arxiv.org/pdf/2506.07726)**

> **作者:** Vincenzo Timmel; Manfred Vogel; Daniel Perruchoud; Reza Kakooee
>
> **备注:** Change: Updated number of hours for train/test
>
> **摘要:** This paper presents a new long-form release of the Swiss Parliaments Corpus, converting entire multi-hour Swiss German debate sessions (each aligned with the official session protocols) into high-quality speech-text pairs. Our pipeline starts by transcribing all session audio into Standard German using Whisper Large-v3 under high-compute settings. We then apply a two-step GPT-4o correction process: first, GPT-4o ingests the raw Whisper output alongside the official protocols to refine misrecognitions, mainly named entities. Second, a separate GPT-4o pass evaluates each refined segment for semantic completeness. We filter out any segments whose Predicted BLEU score (derived from Whisper's average token log-probability) and GPT-4o evaluation score fall below a certain threshold. The final corpus contains 801 hours of audio, of which 555 hours pass our quality control. Compared to the original sentence-level SPC release, our long-form dataset achieves a 6-point BLEU improvement, demonstrating the power of combining robust ASR, LLM-based correction, and data-driven filtering for low-resource, domain-specific speech corpora.
>
---
#### [replaced 002] NeuralOS: Towards Simulating Operating Systems via Neural Generative Models
- **分类: cs.CV; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文提出NeuralOS，属于操作系统模拟任务，旨在通过神经网络生成GUI响应用户输入，解决真实系统数据依赖问题，利用合成数据训练模型模拟未安装应用。**

- **链接: [https://arxiv.org/pdf/2507.08800](https://arxiv.org/pdf/2507.08800)**

> **作者:** Luke Rivard; Sun Sun; Hongyu Guo; Wenhu Chen; Yuntian Deng
>
> **备注:** ICLR 2026
>
> **摘要:** We introduce NeuralOS, a neural framework that simulates graphical user interfaces (GUIs) of operating systems by directly predicting screen frames in response to user inputs such as mouse movements, clicks, and keyboard events. NeuralOS combines a recurrent neural network (RNN), which tracks computer state, with a diffusion-based neural renderer that generates screen images. The model is trained on a dataset of Ubuntu XFCE recordings, which include both randomly generated interactions and realistic interactions produced by AI agents. Experiments show that NeuralOS successfully renders realistic GUI sequences, accurately captures mouse interactions, and reliably predicts state transitions like application launches. Beyond reproducing existing systems, NeuralOS shows that synthesized training data can teach the model to simulate applications that were never installed, as illustrated by a Doom application, and suggests a path toward learning user interfaces purely from synthetic demonstrations.
>
---
#### [replaced 003] TURA: Tool-Augmented Unified Retrieval Agent for AI Search
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出TURA框架，解决AI搜索中静态与动态信息融合问题，通过三阶段设计实现高效实时查询。**

- **链接: [https://arxiv.org/pdf/2508.04604](https://arxiv.org/pdf/2508.04604)**

> **作者:** Zhejun Zhao; Yuchen Li; Alley Liu; Yuehu Dong; Xiaolong Wei; Lixue Zheng; Pingsheng Liu; Dongdong Shen; Long Xia; Jiashu Zhao; Dawei Yin
>
> **摘要:** The advent of Large Language Models (LLMs) is transforming search engines into conversational AI search products, primarily using Retrieval-Augmented Generation (RAG) on web corpora. However, this paradigm has significant industrial limitations. Traditional RAG approaches struggle with real-time needs and structured queries that require accessing dynamically generated content like ticket availability or inventory. Limited to indexing static pages, search engines cannot perform the interactive queries needed for such time-sensitive data. Academic research has focused on optimizing RAG for static content, overlooking complex intents and the need for dynamic sources like databases and real-time APIs. To bridge this gap, we introduce TURA (Tool-Augmented Unified Retrieval Agent for AI Search), a novel three-stage framework that combines RAG with agentic tool-use to access both static content and dynamic, real-time information. TURA has three key components: an Intent-Aware Retrieval module to decompose queries and retrieve information sources encapsulated as Model Context Protocol (MCP) Servers, a DAG-based Task Planner that models task dependencies as a Directed Acyclic Graph (DAG) for optimal parallel execution, and a lightweight Distilled Agent Executor for efficient tool calling. TURA is the first architecture to systematically bridge the gap between static RAG and dynamic information sources for a world-class AI search product. Serving tens of millions of users, it leverages an agentic framework to deliver robust, real-time answers while meeting the low-latency demands of a large-scale industrial system.
>
---
#### [replaced 004] AraModernBERT: Transtokenized Initialization and Long-Context Encoder Modeling for Arabic
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AraModernBERT，针对阿拉伯语进行编码器优化，解决长文本建模和嵌入初始化问题，提升语言模型性能。**

- **链接: [https://arxiv.org/pdf/2603.09982](https://arxiv.org/pdf/2603.09982)**

> **作者:** Omar Elshehy; Omer Nacar; Abdelbasset Djamai; Muhammed Ragab; Khloud Al Jallad; Mona Abdelazim
>
> **备注:** 9 pages, 1 figure. Accepted at AbjadNLP Workshop, EACL 2026
>
> **摘要:** Encoder-only transformer models remain widely used for discriminative NLP tasks, yet recent architectural advances have largely focused on English. In this work, we present AraModernBERT, an adaptation of the ModernBERT encoder architecture to Arabic, and study the impact of transtokenized embedding initialization and native long-context modeling up to 8,192 tokens. We show that transtokenization is essential for Arabic language modeling, yielding dramatic improvements in masked language modeling performance compared to non-transtokenized initialization. We further demonstrate that AraModernBERT supports stable and effective long-context modeling, achieving improved intrinsic language modeling performance at extended sequence lengths. Downstream evaluations on Arabic natural language understanding tasks, including inference, offensive language detection, question-question similarity, and named entity recognition, confirm strong transfer to discriminative and sequence labeling settings. Our results highlight practical considerations for adapting modern encoder architectures to Arabic and other languages written in Arabic-derived scripts.
>
---
#### [replaced 005] Mock Worlds, Real Skills: Building Small Agentic Language Models with Synthetic Tasks, Simulated Environments, and Rubric-Based Rewards
- **分类: cs.CL**

- **简介: 该论文提出SYNTHAGENT框架，解决小模型缺乏代理能力的问题。通过合成任务和模拟环境，提升小模型在数学、搜索等任务中的表现。**

- **链接: [https://arxiv.org/pdf/2601.22511](https://arxiv.org/pdf/2601.22511)**

> **作者:** Yuanjie Lyu; Chengyu Wang; Lei Shen; Jun Huang; Tong Xu
>
> **备注:** The first author prefers the more commonly used English name "Yuanjie Lyu" over "Yuan-Jay Lü", so we have updated it; both refer to the same person
>
> **摘要:** Small LLMs often struggle to match the agentic capabilities of large, costly models. While reinforcement learning can help, progress has been limited by two structural bottlenecks: existing open-source agentic training data are narrow in task variety and easily solved; real-world APIs lack diversity and are unstable for large-scale reinforcement learning rollout processes. We address these challenges with SYNTHAGENT, a framework that jointly synthesizes diverse tool-use training data and simulates complete environments. Specifically, a strong teacher model creates novel tasks and tool ecosystems, then rewrites them into intentionally underspecified instructions. This compels agents to actively query users for missing details. When handling synthetic tasks, an LLM-based user simulator provides user-private information, while a mock tool system delivers stable tool responses. For rewards, task-level rubrics are constructed based on required subgoals, user-agent interactions, and forbidden behaviors. Across 14 challenging datasets in math, search, and tool use, models trained on our synthetic data achieve substantial gains, with small models outperforming larger baselines.
>
---
#### [replaced 006] Evaluating LLM-Based Grant Proposal Review via Structured Perturbations
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于AI辅助评审任务，探讨LLM在高风险资助提案评审中的能力与局限。通过结构化扰动分析，评估不同评审架构的效果，发现分段评审更有效，但整体仍存在偏差与不一致。**

- **链接: [https://arxiv.org/pdf/2603.08281](https://arxiv.org/pdf/2603.08281)**

> **作者:** William Thorne; Joseph James; Yang Wang; Chenghua Lin; Diana Maynard
>
> **摘要:** As AI-assisted grant proposals outpace manual review capacity in a kind of ``Malthusian trap'' for the research ecosystem, this paper investigates the capabilities and limitations of LLM-based grant reviewing for high-stakes evaluation. Using six EPSRC proposals, we develop a perturbation-based framework probing LLM sensitivity across six quality axes: funding, timeline, competency, alignment, clarity, and impact. We compare three review architectures: single-pass review, section-by-section analysis, and a 'Council of Personas' ensemble emulating expert panels. The section-level approach significantly outperforms alternatives in both detection rate and scoring reliability, while the computationally expensive council method performs no better than baseline. Detection varies substantially by perturbation type, with alignment issues readily identified but clarity flaws largely missed by all systems. Human evaluation shows LLM feedback is largely valid but skewed toward compliance checking over holistic assessment. We conclude that current LLMs may provide supplementary value within EPSRC review but exhibit high variability and misaligned review priorities. We release our code and any non-protected data.
>
---
#### [replaced 007] On the Theoretical Limitations of Embedding-Based Retrieval
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文属于信息检索领域，研究嵌入模型在检索任务中的理论局限性。论文指出即使在简单查询下，嵌入模型也存在固有限制，并通过实验验证了这一结论。**

- **链接: [https://arxiv.org/pdf/2508.21038](https://arxiv.org/pdf/2508.21038)**

> **作者:** Orion Weller; Michael Boratko; Iftekhar Naim; Jinhyuk Lee
>
> **备注:** Accepted to ICLR'26
>
> **摘要:** Vector embeddings have been tasked with an ever-increasing set of retrieval tasks over the years, with a nascent rise in using them for reasoning, instruction-following, coding, and more. These new benchmarks push embeddings to work for any query and any notion of relevance that could be given. While prior works have pointed out theoretical limitations of vector embeddings, there is a common assumption that these difficulties are exclusively due to unrealistic queries, and those that are not can be overcome with better training data and larger models. In this work, we demonstrate that we may encounter these theoretical limitations in realistic settings with extremely simple queries. We connect known results in learning theory, showing that the number of top-k subsets of documents capable of being returned as the result of some query is limited by the dimension of the embedding. We empirically show that this holds true even if we directly optimize on the test set with free parameterized embeddings. Using free embeddings, we then demonstrate that returning all pairs of documents requires a relatively high dimension. We then create a realistic dataset called LIMIT that stress tests embedding models based on these theoretical results, and observe that even state-of-the-art models fail on this dataset despite the simple nature of the task. Our work shows the limits of embedding models under the existing single vector paradigm and calls for future research to develop new techniques that can resolve this fundamental limitation.
>
---
#### [replaced 008] ConCISE: A Reference-Free Conciseness Evaluation Metric for LLM-Generated Answers
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM生成回答冗长的问题。提出一种无参考的简洁性评估指标，通过压缩比和词移除方法量化冗余内容。**

- **链接: [https://arxiv.org/pdf/2511.16846](https://arxiv.org/pdf/2511.16846)**

> **作者:** Seyed Mohssen Ghafari; Ronny Kol; Juan C. Quiroz; Nella Luan; Monika Patial; Chanaka Rupasinghe; Herman Wandabwa; Luiz Pizzato
>
> **摘要:** Large language models (LLMs) frequently generate responses that are lengthy and verbose, filled with redundant or unnecessary details. This diminishes clarity and user satisfaction, and it increases costs for model developers, especially with well-known proprietary models that charge based on the number of output tokens. In this paper, we introduce a novel reference-free metric for evaluating the conciseness of responses generated by LLMs. Our method quantifies non-essential content without relying on gold standard references and calculates the average of three calculations: i) a compression ratio between the original response and an LLM abstractive summary; ii) a compression ratio between the original response and an LLM extractive summary; and iii) wordremoval compression, where an LLM removes as many non-essential words as possible from the response while preserving its meaning, with the number of tokens removed indicating the conciseness score. Experimental results demonstrate that our proposed metric identifies redundancy in LLM outputs, offering a practical tool for automated evaluation of response brevity in conversational AI systems without the need for ground truth human annotations.
>
---
#### [replaced 009] PsihoRo: Depression and Anxiety Romanian Text Corpus
- **分类: cs.CL**

- **简介: 该论文构建了首个罗马尼亚语抑郁与焦虑文本语料库PsihoRo，旨在填补罗马尼亚心理NLP资源的空白。通过开放问题和问卷收集数据，进行文本分析与情绪检测。**

- **链接: [https://arxiv.org/pdf/2602.18324](https://arxiv.org/pdf/2602.18324)**

> **作者:** Alexandra Ciobotaru; Ana-Maria Bucur; Liviu P. Dinu
>
> **备注:** This article was accepted at LREC 2026
>
> **摘要:** Psychological corpora in NLP are collections of texts used to analyze human psychology, emotions, and mental health. These texts allow researchers to study psychological constructs, identify patterns related to mental health problems and analyze emotional language. However, collecting accurate mental health data from social media can be challenging due to the assumptions made by data collectors. A more effective approach involves gathering data through open-ended questions and then assessing participants' mental health status using self-report screening surveys. This method was successfully employed for English, a language with a lot of psychological NLP resources. However, the same cannot be stated for Romanian, which currently has no open-source mental health corpus. To address this gap, we have collected the first open-source corpus focused on depression and anxiety in Romanian, by utilizing a form with 6 open-ended questions along with the standardized PHQ-9 and GAD-7 screening questionnaires. Although the PsihoRo corpus contains texts from only 205 respondents, it represents an important first step toward understanding and analyzing mental health issues within the Romanian population. We employ statistical analysis, text analysis using Romanian LIWC, emotion detection, and topic modeling to identify the most important features of this newly introduced resource for the NLP community. The data is publicly available at this https URL.
>
---
#### [replaced 010] Beyond the Black Box: A Survey on the Theory and Mechanism of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在解决LLM理论理解不足的问题。通过构建生命周期分类框架，系统分析了LLM的理论与机制，提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2601.02907](https://arxiv.org/pdf/2601.02907)**

> **作者:** Zeyu Gan; Ruifeng Ren; Wei Yao; Xiaolin Hu; Gengze Xu; Chen Qian; Huayi Tang; Zixuan Gong; Xinhao Yao; Pengwei Tang; Zhenxing Dou; Yong Liu
>
> **摘要:** The rapid emergence of Large Language Models (LLMs) has precipitated a profound paradigm shift in Artificial Intelligence, delivering monumental engineering successes that increasingly impact modern society. However, a critical paradox persists within the current field: despite the empirical efficacy, our theoretical understanding of LLMs remains disproportionately nascent, forcing these systems to be treated largely as ``black boxes''. To address this theoretical fragmentation, this survey proposes a unified lifecycle-based taxonomy that organizes the research landscape into six distinct stages: Data Preparation, Model Preparation, Training, Alignment, Inference, and Evaluation. Within this framework, we provide a systematic review of the foundational theories and internal mechanisms driving LLM performance. Specifically, we analyze core theoretical issues such as the mathematical justification for data mixtures, the representational limits of various architectures, and the optimization dynamics of alignment algorithms. Moving beyond current best practices, we identify critical frontier challenges, including the theoretical limits of synthetic data self-improvement, the mathematical bounds of safety guarantees, and the mechanistic origins of emergent intelligence. By connecting empirical observations with rigorous scientific inquiry, this work provides a structured roadmap for transitioning LLM development from engineering heuristics toward a principled scientific discipline.
>
---
#### [replaced 011] X-GS: An Extensible Open Framework for Perceiving and Thinking via 3D Gaussian Splatting
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出X-GS框架，解决3DGS应用孤立问题，通过统一方法实现实时SLAM和多模态任务，提升效率与扩展性。**

- **链接: [https://arxiv.org/pdf/2603.09632](https://arxiv.org/pdf/2603.09632)**

> **作者:** Yueen Ma; Zenglin Xu; Irwin King
>
> **摘要:** 3D Gaussian Splatting (3DGS) has emerged as a powerful technique for novel view synthesis, subsequently extending into numerous spatial AI applications. However, most existing 3DGS methods operate in isolation, focusing on specific domains such as pose-free 3DGS, online SLAM, and semantic enrichment. In this paper, we introduce X-GS, an extensible open framework consisting of two major components: the X-GS-Perceiver, which unifies a broad range of 3DGS techniques to enable real-time online SLAM and distill semantic features; and the X-GS-Thinker, which interfaces with downstream multimodal models. In our implementation of the Perceiver, we integrate various 3DGS methods through three novel mechanisms: an online Vector Quantization (VQ) module, a GPU-accelerated grid-sampling scheme, and a highly parallelized pipeline design. The Thinker accommodates vision-language models and utilizes the resulting 3D semantic Gaussians, enabling downstream applications such as object detection, caption generation, and potentially embodied tasks. Experimental results on real-world datasets demonstrate the efficiency and newly unlocked multimodal capabilities of the X-GS framework.
>
---
#### [replaced 012] Hidden State Poisoning Attacks against Mamba-based Language Models
- **分类: cs.CL**

- **简介: 该论文研究了针对Mamba模型的隐藏状态污染攻击（HiSPA），揭示其在对抗性环境下的脆弱性。属于安全与鲁棒性任务，旨在解决模型易受特定输入干扰的问题。**

- **链接: [https://arxiv.org/pdf/2601.01972](https://arxiv.org/pdf/2601.01972)**

> **作者:** Alexandre Le Mercier; Chris Develder; Thomas Demeester
>
> **备注:** 29 pages, 4 figures
>
> **摘要:** State space models (SSMs) like Mamba offer efficient alternatives to Transformer-based language models, with linear time complexity. Yet, their adversarial robustness remains critically unexplored. This paper studies the phenomenon whereby specific short input phrases induce a partial amnesia effect in such models, by irreversibly overwriting information in their hidden states, referred to as a Hidden State Poisoning Attack (HiSPA). Our benchmark RoBench25 allows evaluating a model's information retrieval capabilities when subject to HiSPAs, and confirms the vulnerability of SSMs against such attacks. Even a recent 52B hybrid SSM-Transformer model from the Jamba family collapses on RoBench25 under optimized HiSPA triggers, whereas pure Transformers do not. We also observe that HiSPA triggers significantly weaken the Jamba model on the popular Open-Prompt-Injections benchmark, unlike pure Transformers. Finally, our interpretability study reveals patterns in Mamba's hidden layers during HiSPAs that could be used to build a HiSPA mitigation system. The full code and data to reproduce the experiments can be found at this https URL.
>
---
#### [replaced 013] Ultra-Fast Language Generation via Discrete Diffusion Divergence Instruct
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出DiDi-Instruct，用于加速语言生成任务。通过知识蒸馏，提升生成速度并保持质量，解决高效高质量生成难题。**

- **链接: [https://arxiv.org/pdf/2509.25035](https://arxiv.org/pdf/2509.25035)**

> **作者:** Haoyang Zheng; Xinyang Liu; Cindy Xiangrui Kong; Nan Jiang; Zheyuan Hu; Weijian Luo; Wei Deng; Guang Lin
>
> **备注:** [ICLR 2026] 38 pages, 7 figures, 13 tables
>
> **摘要:** Fast and high-quality language generation is the holy grail that people pursue in the age of AI. In this work, we introduce Discrete Diffusion Divergence Instruct (DiDi-Instruct), a training-based method that initializes from a pre-trained diffusion large language model (dLLM) and distills a few-step student for fast generation. The model distilled with DiDi-Instruct matches or surpasses its dLLM teacher and the GPT-2 baseline while providing up to 64$\times$ acceleration. The theoretical foundation of DiDi-Instruct is a novel framework based on integral KL-divergence minimization, which leads to a practical training algorithm. We further introduce grouped reward normalization, intermediate-state matching, and the reward-guided ancestral sampler to improve training stability, model coverage, and inference quality. On the OpenWebText benchmark, DiDi-Instruct achieves perplexity ranging from 62.2 (8 NFEs) to 18.4 (128 NFEs), outperforming prior accelerated dLLMs and the GPT-2 baseline. These gains incur a negligible entropy loss (around $1$%) and reduce additional training wall-clock time by more than $20\times$ compared to competing dLLM distillation methods. We further validate the robustness and effectiveness of DiDi-Instruct through extensive ablation studies, model scaling, downstream task evaluations, and unconditional protein sequence generation. In conclusion, DiDi-Instruct enables efficient and effective distillation for language generation in the blink of an eye.
>
---
#### [replaced 014] Can Theoretical Physics Research Benefit from Language Agents?
- **分类: cs.CL; cs.AI; math-ph; quant-ph**

- **简介: 论文探讨了语言模型在理论物理中的应用，指出其在物理直觉和推理上的不足，提出需专门训练和工具支持。任务是提升AI在物理研究中的实用性，解决模型缺乏物理知识的问题，工作包括构建专用数据集和验证框架。**

- **链接: [https://arxiv.org/pdf/2506.06214](https://arxiv.org/pdf/2506.06214)**

> **作者:** Sirui Lu; Zhijing Jin; Terry Jingchen Zhang; Pavel Kos; J. Ignacio Cirac; Bernhard Schölkopf
>
> **备注:** 8+2 pages + references
>
> **摘要:** Large Language Models (LLMs) are rapidly advancing across diverse domains, yet their application in theoretical physics remains inadequate. While current models show competence in mathematical reasoning and code generation, we identify critical gaps in physical intuition, constraint satisfaction, and reliable reasoning that cannot be addressed through prompting alone. Physics demands approximation judgment, symmetry exploitation, and physical grounding that require AI agents specifically trained on physics reasoning patterns and equipped with physics-aware verification tools. We argue that LLM would require such domain-specialized training and tooling to be useful in real-world for physics research. We envision physics-specialized AI agents that seamlessly handle multimodal data, propose physically consistent hypotheses, and autonomously verify theoretical results. Realizing this vision requires developing physics-specific training datasets, reward signals that capture physical reasoning quality, and verification frameworks encoding fundamental principles. We call for collaborative efforts between physics and AI communities to build the specialized infrastructure necessary for AI-driven scientific discovery.
>
---
#### [replaced 015] FrugalPrompt: Reducing Contextual Overhead in Large Language Models via Token Attribution
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型因冗长输入导致的效率问题。通过保留语义关键token，提出FrugalPrompt框架以减少上下文开销。**

- **链接: [https://arxiv.org/pdf/2510.16439](https://arxiv.org/pdf/2510.16439)**

> **作者:** Syed Rifat Raiyan; Md Farhan Ishmam; Abdullah Al Imran; Mohammad Ali Moni
>
> **摘要:** Human communication heavily relies on laconism and inferential pragmatics, allowing listeners to successfully reconstruct rich meaning from sparse, telegraphic speech. In contrast, large language models (LLMs) owe much of their stellar performance to expansive input contexts, yet such verbosity inflates monetary costs, carbon footprint, and inference-time latency. This overhead manifests from the redundant low-utility tokens present in typical prompts, as only a fraction of tokens typically carries the majority of the semantic weight. Inspired by the aforementioned cognitive psycholinguistic processes, we address this inefficiency by introducing FrugalPrompt, a novel prompt compression framework for LLMs, which retains only the most semantically significant tokens. Leveraging two state-of-the-art token attribution methods, GlobEnc and DecompX, we assign salience scores to every token in an input sequence, rank them to retain the top-k% tokens, and obtain a sparse frugalized prompt. We establish the theoretical stability of our approach and provide strong empirical results across a suite of four NLP tasks to study the trade-off between the portion of retained tokens and performance. Experimental findings across retention settings reveal asymmetric performance patterns that suggest potential task contamination effects. We posit that our work contributes to a more nuanced understanding of LLM behavior in performance-efficiency trade-offs and delineates the boundary between tasks tolerant of contextual sparsity and those requiring exhaustive context.
>
---
#### [replaced 016] Beyond the Prompt in Large Language Models: Comprehension, In-Context Learning, and Chain-of-Thought
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大语言模型的提示理解、上下文学习和思维链机制，旨在揭示其理论基础。通过分析模型如何解码提示、提升性能及分解任务，提供新的理论见解。**

- **链接: [https://arxiv.org/pdf/2603.10000](https://arxiv.org/pdf/2603.10000)**

> **作者:** Yuling Jiao; Yanming Lai; Huazhen Lin; Wensen Ma; Houduo Qi; Defeng Sun
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable proficiency across diverse tasks, exhibiting emergent properties such as semantic prompt comprehension, In-Context Learning (ICL), and Chain-of-Thought (CoT) reasoning. Despite their empirical success, the theoretical mechanisms driving these phenomena remain poorly understood. This study dives into the foundations of these observations by addressing three critical questions: (1) How do LLMs accurately decode prompt semantics despite being trained solely on a next-token prediction objective? (2) Through what mechanism does ICL facilitate performance gains without explicit parameter updates? and (3) Why do intermediate reasoning steps in CoT prompting effectively unlock capabilities for complex, multi-step problems? Our results demonstrate that, through the autoregressive process, LLMs are capable of exactly inferring the transition probabilities between tokens across distinct tasks using provided prompts. We show that ICL enhances performance by reducing prompt ambiguity and facilitating posterior concentration on the intended task. Furthermore, we find that CoT prompting activates the model's capacity for task decomposition, breaking complex problems into a sequence of simpler sub-tasks that the model has mastered during the pretraining phase. By comparing their individual error bounds, we provide novel theoretical insights into the statistical superiority of advanced prompt engineering techniques.
>
---
#### [replaced 017] Learning Through Dialogue: Engagement and Efficacy Matter More Than Explanations
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于人机交互任务，探讨LLM对话中学习效果的影响因素。研究解决LLM如何通过对话促进用户学习的问题，分析了互动特征与学习成效的关系。**

- **链接: [https://arxiv.org/pdf/2601.07796](https://arxiv.org/pdf/2601.07796)**

> **作者:** Shaz Furniturewala; Gerard Christopher Yeo; Kokil Jaidka
>
> **摘要:** Large language models (LLMs) are increasingly used as conversational partners for learning, yet the interactional dynamics supporting users' learning and engagement are understudied. We analyze the linguistic and interactional features from both LLM and participant chats across 397 human-LLM conversations about socio-political issues to identify the mechanisms and conditions under which LLM explanations shape changes in political knowledge and confidence. Mediation analyses reveal that LLM explanatory richness partially supports confidence by fostering users' reflective insight, whereas its effect on knowledge gain operates entirely through users' cognitive engagement. Moderation analyses show that these effects are highly conditional and vary by political efficacy. Confidence gains depend on how high-efficacy users experience and resolve uncertainty. Knowledge gains depend on high-efficacy users' ability to leverage extended interaction, with longer conversations benefiting primarily reflective users. In summary, we find that learning from LLMs is an interactional achievement, not a uniform outcome of better explanations. The findings underscore the importance of aligning LLM explanatory behavior with users' engagement states to support effective learning in designing Human-AI interactive systems.
>
---
#### [replaced 018] Prompting Underestimates LLM Capability for Time Series Classification
- **分类: cs.CL**

- **简介: 该论文研究LLM在时间序列分类任务中的表现，指出prompt评估低估了模型能力。通过对比提示输出与线性探测器，发现模型内部已具备时间结构信息。**

- **链接: [https://arxiv.org/pdf/2601.03464](https://arxiv.org/pdf/2601.03464)**

> **作者:** Dan Schumacher; Erfan Nourbakhsh; Rocky Slavin; Anthony Rios
>
> **备注:** 8 pages + Appendix and References, 9 figures
>
> **摘要:** Prompt-based evaluations suggest that large language models (LLMs) perform poorly on time series classification, raising doubts about whether they encode meaningful temporal structure. We show that this conclusion reflects limitations of prompt-based generation rather than the model's representational capacity by directly comparing prompt outputs with linear probes over the same internal representations. While zero-shot prompting performs near chance, linear probes improve average F1 from 0.15-0.26 to 0.61-0.67, often matching or exceeding specialized time series models. Layer-wise analyses further show that class-discriminative time series information emerges in early transformer layers and is amplified by visual and multimodal inputs. Together, these results demonstrate a systematic mismatch between what LLMs internally represent and what prompt-based evaluation reveals, leading current evaluations to underestimate their time series understanding.
>
---
#### [replaced 019] PosIR: Position-Aware Heterogeneous Information Retrieval Benchmark
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决位置偏差问题。通过构建PosIR基准，分析不同文档位置对检索的影响，揭示模型的偏好机制。**

- **链接: [https://arxiv.org/pdf/2601.08363](https://arxiv.org/pdf/2601.08363)**

> **作者:** Ziyang Zeng; Dun Zhang; Yu Yan; Xu Sun; Cuiqiaoshu Pan; Yudong Zhou; Yuqing Yang
>
> **备注:** Work in progress
>
> **摘要:** In real-world documents, the information relevant to a user query may reside anywhere from the beginning to the end. This makes position bias -- a systematic tendency of retrieval models to favor or neglect content based on its location -- a critical concern. Although recent studies have identified such bias, existing analyses focus predominantly on English, fail to disentangle document length from information position, and lack a standardized framework for systematic diagnosis. To address these limitations, we introduce PosIR (Position-Aware Information Retrieval), the first standardized benchmark designed to systematically diagnose position bias in diverse retrieval scenarios. PosIR comprises 310 datasets spanning 10 languages and 31 domains, with relevance tied to precise reference spans. At its methodological core, PosIR employs a length-controlled bucketing strategy that groups queries by positive document length and analyzes positional effects within each bucket. This design strictly isolates position bias from length-induced performance degradation. Extensive experiments on 10 state-of-the-art embedding-based retrieval models reveal that: (1) retrieval performance on PosIR with documents exceeding 1536 tokens correlates poorly with the MMTEB benchmark, exposing limitations of current short-text evaluations; (2) position bias is pervasive in embedding models and even increases with document length, with most models exhibiting primacy bias while certain models show unexpected recency bias; (3) as an exploratory investigation, gradient-based saliency analysis further uncovers two distinct internal mechanisms that correlate with these positional preferences. We hope that PosIR can serve as a valuable diagnostic framework to advance the development of position-robust retrieval systems.
>
---
#### [replaced 020] Efficient Compositional Multi-tasking for On-device Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究在设备端进行高效的组合多任务处理，解决传统方法仅支持单任务的问题。提出基准和高效方法，提升资源受限环境下的多任务性能。**

- **链接: [https://arxiv.org/pdf/2507.16083](https://arxiv.org/pdf/2507.16083)**

> **作者:** Ondrej Bohdal; Mete Ozay; Jijoong Moon; Kyeng-Hun Lee; Hyeonmok Ko; Umberto Michieli
>
> **备注:** Accepted at EMNLP 2025 (main track, long paper)
>
> **摘要:** Adapter parameters provide a mechanism to modify the behavior of machine learning models and have gained significant popularity in the context of large language models (LLMs) and generative AI. These parameters can be merged to support multiple tasks via a process known as task merging. However, prior work on merging in LLMs, particularly in natural language processing, has been limited to scenarios where each test example addresses only a single task. In this paper, we focus on on-device settings and study the problem of text-based compositional multi-tasking, where each test example involves the simultaneous execution of multiple tasks. For instance, generating a translated summary of a long text requires solving both translation and summarization tasks concurrently. To facilitate research in this setting, we propose a benchmark comprising four practically relevant compositional tasks. We also present an efficient method (Learnable Calibration) tailored for on-device applications, where computational resources are limited, emphasizing the need for solutions that are both resource-efficient and high-performing. Our contributions lay the groundwork for advancing the capabilities of LLMs in real-world multi-tasking scenarios, expanding their applicability to complex, resource-constrained use cases.
>
---
#### [replaced 021] AudioTrust: Benchmarking the Multifaceted Trustworthiness of Audio Large Language Models
- **分类: cs.SD; cs.AI; cs.CL; eess.AS**

- **简介: 该论文属于音频大模型可信性评估任务，旨在解决现有框架无法有效评估音频模型安全性和可靠性的问题。工作包括构建AudioTrust框架，涵盖六维评估指标及真实场景数据集，全面测试模型在多种高风险音频场景下的表现。**

- **链接: [https://arxiv.org/pdf/2505.16211](https://arxiv.org/pdf/2505.16211)**

> **作者:** Kai Li; Can Shen; Yile Liu; Jirui Han; Kelong Zheng; Xuechao Zou; Lionel Z. Wang; Shun Zhang; Xingjian Du; Hanjun Luo; Yingbin Jin; Xinxin Xing; Ziyang Ma; Yue Liu; Yifan Zhang; Junfeng Fang; Kun Wang; Yibo Yan; Gelei Deng; Haoyang Li; Yiming Li; Xiaobin Zhuang; Tianlong Chen; Qingsong Wen; Tianwei Zhang; Yang Liu; Haibo Hu; Zhizheng Wu; Xiaolin Hu; Eng-Siong Chng; Wenyuan Xu; XiaoFeng Wang; Wei Dong; Xinfeng Li
>
> **备注:** Accepted to ICLR 2026
>
> **摘要:** The rapid development and widespread adoption of Audio Large Language Models (ALLMs) demand rigorous evaluation of their trustworthiness. However, existing evaluation frameworks are primarily designed for text and fail to capture vulnerabilities introduced by the acoustic properties of audio. We find that significant trustworthiness risks in ALLMs arise from non-semantic acoustic cues, such as timbre, accent, and background noise, which can be exploited to manipulate model behavior. To address this gap, we propose AudioTrust, the first large-scale and systematic framework for evaluating ALLM trustworthiness under audio-specific risks. AudioTrust covers six key dimensions: fairness, hallucination, safety, privacy, robustness, and authenticition. It includes 26 sub-tasks and a curated dataset of more than 4,420 audio samples collected from real-world scenarios, including daily conversations, emergency calls, and voice assistant interactions, and is specifically designed to probe trustworthiness across multiple dimensions. Our comprehensive evaluation spans 18 experimental settings and uses human-validated automated pipelines to enable objective and scalable assessment of model outputs. Experimental results on 14 state-of-the-art open-source and closed-source ALLMs reveal important limitations and failure boundaries under diverse high-risk audio scenarios, providing critical insights for the secure and trustworthy deployment of future audio models. Our platform and benchmark are publicly available at this https URL.
>
---
#### [replaced 022] Seq vs Seq: An Open Suite of Paired Encoders and Decoders
- **分类: cs.CL; cs.IR; cs.LG**

- **简介: 该论文对比了编码器和解码器模型，解决模型架构选择问题，通过统一训练方法提升性能，并开源相关数据与模型。**

- **链接: [https://arxiv.org/pdf/2507.11412](https://arxiv.org/pdf/2507.11412)**

> **作者:** Orion Weller; Kathryn Ricci; Marc Marone; Antoine Chaffin; Dawn Lawrie; Benjamin Van Durme
>
> **备注:** Accepted to ICLR'26
>
> **摘要:** The large language model (LLM) community focuses almost exclusively on decoder-only language models, since they are easier to use for text generation. However, a large subset of the community still uses encoder-only models for tasks such as classification or retrieval. Previous work has attempted to compare these architectures, but is forced to make comparisons with models that have different numbers of parameters, training techniques, and datasets. We introduce the SOTA open-data Ettin suite of models: paired encoder-only and decoder-only models ranging from 17 million parameters to 1 billion, trained on up to 2 trillion tokens. Using the same recipe for both encoder-only and decoder-only models produces SOTA recipes in both categories for their respective sizes, beating ModernBERT as an encoder and Llama 3.2 and SmolLM2 as decoders. Like previous work, we find that encoder-only models excel at classification and retrieval tasks while decoders excel at generative tasks. However, we show that adapting a decoder model to encoder tasks (and vice versa) through continued training is subpar compared to using only the reverse objective (i.e. a 400M encoder outperforms a 1B decoder on MNLI, and vice versa for generative tasks). We open-source all artifacts of this study including training data, training order segmented by checkpoint, and 200+ checkpoints to allow future work to analyze or extend all aspects of training.
>
---
#### [replaced 023] LLLMs: A Data-Driven Survey of Evolving Research on Limitations of Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于文献综述任务，旨在分析大语言模型的局限性研究。通过数据驱动方法，梳理2022至2025年的相关论文，揭示研究趋势与重点问题。**

- **链接: [https://arxiv.org/pdf/2505.19240](https://arxiv.org/pdf/2505.19240)**

> **作者:** Aida Kostikova; Zhipin Wang; Deidamea Bajri; Ole Pütz; Benjamin Paaßen; Steffen Eger
>
> **备注:** ACM Computing Surveys (CSUR); 56 pages
>
> **摘要:** Large language model (LLM) research has grown rapidly, along with increasing concern about their limitations. In this survey, we conduct a data-driven, semi-automated review of research on limitations of LLMs (LLLMs) from 2022 to early 2025 using a bottom-up approach. From a corpus of 250,000 ACL and arXiv papers, we identify 14,648 relevant papers using keyword filtering, LLM-based classification, validated against expert labels, and topic clustering (via two approaches, HDBSCAN+BERTopic and LlooM). We find that the share of LLM-related papers increases over fivefold in ACL and nearly eightfold in arXiv between 2022 and 2025. Since 2022, LLLMs research grows even faster, reaching over 30% of LLM papers by 2025. Reasoning remains the most studied limitation, followed by generalization, hallucination, bias, and security. The distribution of topics in the ACL dataset stays relatively stable over time, while arXiv shifts toward security risks, alignment, hallucinations, knowledge editing, and multimodality. We offer a quantitative view of trends in LLLMs research and release a dataset of annotated abstracts and a validated methodology, available at: this https URL.
>
---
#### [replaced 024] Do LLMs Truly Benefit from Longer Context in Automatic Post-Editing?
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于自动后编辑任务，探讨LLMs在文档级上下文下的表现。研究对比了专有与开源LLMs，分析其质量、鲁棒性及效率，发现专有模型虽性能强但缺乏上下文利用能力。**

- **链接: [https://arxiv.org/pdf/2601.19410](https://arxiv.org/pdf/2601.19410)**

> **作者:** Ahrii Kim; Seong-heum Kim
>
> **摘要:** Automatic post-editing (APE) aims to refine machine translations by correcting residual errors. Although recent large language models (LLMs) demonstrate strong translation capabilities, their effectiveness for APE--especially under document-level context--remains insufficiently understood. We present a systematic comparison of proprietary and open-weight LLMs under a naive document-level prompting setup, analyzing APE quality, contextual behavior, robustness, and efficiency. Our results show that proprietary LLMs achieve near human-level APE quality even with simple one-shot prompting, regardless of whether document context is provided. While these models exhibit higher robustness to data poisoning attacks than open-weight counterparts, this robustness also reveals a limitation: they largely fail to exploit document-level context for contextual error correction. Furthermore, standard automatic metrics do not reliably reflect these qualitative improvements, highlighting the continued necessity of human evaluation. Despite their strong performance, the substantial cost and latency overheads of proprietary LLMs render them impractical for real-world APE deployment. Overall, our findings elucidate both the promise and current limitations of LLM-based document-aware APE, and point toward the need for more efficient long-context modeling approaches for translation refinement.
>
---
#### [replaced 025] Let It Flow: Agentic Crafting on Rock and Roll, Building the ROME Model within an Open Agentic Learning Ecosystem
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出ALE生态系统，解决开放环境中智能体开发问题。构建ROME模型，采用IPA算法提升长序列训练稳定性。**

- **链接: [https://arxiv.org/pdf/2512.24873](https://arxiv.org/pdf/2512.24873)**

> **作者:** Weixun Wang; XiaoXiao Xu; Wanhe An; Fangwen Dai; Wei Gao; Yancheng He; Ju Huang; Qiang Ji; Hanqi Jin; Xiaoyang Li; Yang Li; Zhongwen Li; Shirong Lin; Jiashun Liu; Zenan Liu; Tao Luo; Dilxat Muhtar; Yuanbin Qu; Jiaqiang Shi; Qinghui Sun; Yingshui Tan; Hao Tang; Runze Wang; Yi Wang; Zhaoguo Wang; Yanan Wu; Shaopan Xiong; Binchen Xu; Xander Xu; Yuchi Xu; Qipeng Zhang; Xixia Zhang; Haizhou Zhao; Jie Zhao; Shuaibing Zhao; Baihui Zheng; Jianhui Zheng; Suhang Zheng; Yanni Zhu; Mengze Cai; Kerui Cao; Xitong Chen; Yue Dai; Lifan Du; Tao Feng; Tao He; Jin Hu; Yijie Hu; Ziyu Jiang; Cheng Li; Xiang Li; Jing Liang; Xin Lin; Chonghuan Liu; ZhenDong Liu; Zhiqiang Lv; Haodong Mi; Yanhu Mo; Junjia Ni; Shixin Pei; Jingyu Shen; XiaoShuai Song; Cecilia Wang; Chaofan Wang; Kangyu Wang; Pei Wang; Tao Wang; Wei Wang; Ke Xiao; Mingyu Xu; Tiange Xu; Nan Ya; Siran Yang; Jianan Ye; Yaxing Zang; Duo Zhang; Junbo Zhang; Boren Zheng; Wanxi Deng; Ling Pan; Lin Qu; Wenbo Su; Jiamang Wang; Wei Wang; Hu Wei; Minggang Wu; Cheng Yu; Bing Zhao; Zhicheng Zheng; Bo Zheng
>
> **备注:** 36 pages, 15 figures
>
> **摘要:** Agentic crafting requires LLMs to operate in real-world environments over multiple turns by taking actions, observing outcomes, and iteratively refining artifacts. Despite its importance, the open-source community lacks a principled, end-to-end ecosystem to streamline agent development. We introduce the Agentic Learning Ecosystem (ALE), a foundational infrastructure that optimizes the production pipeline for agentic model. ALE consists of three components: ROLL, a post-training framework for weight optimization; ROCK, a sandbox environment manager for trajectory generation; and iFlow CLI, an agent framework for efficient context engineering. We release ROME, an open-source agent grounded by ALE and trained on over one million trajectories. Our approach includes data composition protocols for synthesizing complex behaviors and a novel policy optimization algorithm, Interaction-Perceptive Agentic Policy Optimization (IPA), which assigns credit over semantic interaction chunks rather than individual tokens to improve long-horizon training stability. Empirically, we evaluate ROME within a structured setting and introduce Terminal Bench Pro, a benchmark with improved scale and contamination control. ROME demonstrates strong performance across benchmarks like SWE-bench Verified and Terminal Bench, proving the effectiveness of ALE.
>
---
#### [replaced 026] Expert Selections In MoE Models Reveal (Almost) As Much As Text
- **分类: cs.CL; cs.CR**

- **简介: 该论文研究MoE模型中专家选择的隐私泄露问题，通过分析路由信息恢复文本内容，属于安全与隐私任务。工作包括提出文本重建攻击方法，验证其有效性，并讨论实际泄露风险。**

- **链接: [https://arxiv.org/pdf/2602.04105](https://arxiv.org/pdf/2602.04105)**

> **作者:** Amir Nuriyev; Gabriel Kulp
>
> **摘要:** We present a text-reconstruction attack on mixture-of-experts (MoE) language models that recovers tokens from expert selections alone. In MoE models, each token is routed to a subset of expert subnetworks; we show these routing decisions leak substantially more information than previously understood. Prior work using logistic regression achieves limited reconstruction; we show that a 3-layer MLP improves this to 63.1% top-1 accuracy, and that a transformer-based sequence decoder recovers 91.2% of tokens top-1 (94.8% top-10) on 32-token sequences from OpenWebText after training on 100M tokens. These results connect MoE routing to the broader literature on embedding inversion. We outline practical leakage scenarios (e.g., distributed inference and side channels) and show that adding noise reduces but does not eliminate reconstruction. Our findings suggest that expert selections in MoE deployments should be treated as sensitive as the underlying text.
>
---
#### [replaced 027] NormGenesis: Multicultural Dialogue Generation via Exemplar-Guided Social Norm Modeling and Violation Recovery
- **分类: cs.CL**

- **简介: 该论文提出NormGenesis框架，解决跨文化对话生成中的社会规范问题。通过V2R对话类型和示例引导方法，提升对话的自然度与文化适应性。**

- **链接: [https://arxiv.org/pdf/2509.18395](https://arxiv.org/pdf/2509.18395)**

> **作者:** Minki Hong; Jangho Choi; Jihie Kim
>
> **备注:** 39 pages, 17 figures, EMNLP 2025 Main Conference, Senior Area Chair (SAC) Highlights Award
>
> **摘要:** Social norms govern culturally appropriate behavior in communication, enabling dialogue systems to produce responses that are not only coherent but also socially acceptable. We present NormGenesis, a multicultural framework for generating and annotating socially grounded dialogues across English, Chinese, and Korean. To model the dynamics of social interaction beyond static norm classification, we propose a novel dialogue type, Violation-to-Resolution (V2R), which models the progression of conversations following norm violations through recognition and socially appropriate repair. To improve pragmatic consistency in underrepresented languages, we implement an exemplar-based iterative refinement early in the dialogue synthesis process. This design introduces alignment with linguistic, emotional, and sociocultural expectations before full dialogue generation begins. Using this framework, we construct a dataset of 10,800 multi-turn dialogues annotated at the turn level for norm adherence, speaker intent, and emotional response. Human and LLM-based evaluations demonstrate that NormGenesis significantly outperforms existing datasets in refinement quality, dialogue naturalness, and generalization performance. We show that models trained on our V2R-augmented data exhibit improved pragmatic competence in ethically sensitive contexts. Our work establishes a new benchmark for culturally adaptive dialogue modeling and provides a scalable methodology for norm-aware generation across linguistically and culturally diverse languages.
>
---
#### [replaced 028] Belief Dynamics Reveal the Dual Nature of In-Context Learning and Activation Steering
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文研究大语言模型的控制方法，解决如何统一解释提示学习和激活操控的问题。通过贝叶斯视角构建模型，揭示二者在信念动态上的共性与差异。**

- **链接: [https://arxiv.org/pdf/2511.00617](https://arxiv.org/pdf/2511.00617)**

> **作者:** Eric Bigelow; Daniel Wurgaft; YingQiao Wang; Noah Goodman; Tomer Ullman; Hidenori Tanaka; Ekdeep Singh Lubana
>
> **摘要:** Large language models (LLMs) can be controlled at inference time through prompts (in-context learning) and internal activations (activation steering). Different accounts have been proposed to explain these methods, yet their common goal of controlling model behavior raises the question of whether these seemingly disparate methodologies can be seen as specific instances of a broader framework. Motivated by this, we develop a unifying, predictive account of LLM control from a Bayesian perspective. Specifically, we posit that both context- and activation-based interventions impact model behavior by altering its belief in latent concepts: steering operates by changing concept priors, while in-context learning leads to an accumulation of evidence. This results in a closed-form Bayesian model that is highly predictive of LLM behavior across context- and activation-based interventions in a set of domains inspired by prior work on many-shot in-context learning. This model helps us explain prior empirical phenomena - e.g., sigmoidal learning curves as in-context evidence accumulates - while predicting novel ones - e.g., additivity of both interventions in log-belief space, which results in distinct phases such that sudden and dramatic behavioral shifts can be induced by slightly changing intervention controls. Taken together, this work offers a unified account of prompt-based and activation-based control of LLM behavior, and a methodology for empirically predicting the effects of these interventions.
>
---
#### [replaced 029] Leveraging Wikidata for Geographically Informed Sociocultural Bias Dataset Creation: Application to Latin America
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM在非英语文化背景下的偏见问题。通过Wikidata构建拉美文化偏见数据集，评估模型在不同拉美国家的表现差异。**

- **链接: [https://arxiv.org/pdf/2603.10001](https://arxiv.org/pdf/2603.10001)**

> **作者:** Yannis Karmim; Renato Pino; Hernan Contreras; Hernan Lira; Sebastian Cifuentes; Simon Escoffier; Luis Martí; Djamé Seddah; Valentin Barrière
>
> **摘要:** Large Language Models (LLMs) exhibit inequalities with respect to various cultural contexts. Most prominent open-weights models are trained on Global North data and show prejudicial behavior towards other cultures. Moreover, there is a notable lack of resources to detect biases in non-English languages, especially from Latin America (Latam), a continent containing various cultures, even though they share a common cultural ground. We propose to leverage the content of Wikipedia, the structure of the Wikidata knowledge graph, and expert knowledge from social science in order to create a dataset of question/answer (Q/As) pairs, based on the different popular and social cultures of various Latin American countries. We create the LatamQA database of over 26k questions and associated answers extracted from 26k Wikipedia articles, and transformed into multiple-choice questions (MCQ) in Spanish and Portuguese, in turn translated to English. We use this MCQ to quantify the degree of knowledge of various LLMs and find out (i) a discrepancy in performances between the Latam countries, ones being easier than others for the majority of the models, (ii) that the models perform better in their original language, and (iii) that Iberian Spanish culture is better known than Latam one.
>
---
#### [replaced 030] CARROT: A Learned Cost-Constrained Retrieval Optimization System for RAG
- **分类: cs.DB; cs.CL; cs.IR**

- **简介: 该论文属于RAG系统优化任务，旨在解决检索碎片冗余、非单调效用及查询适应性差的问题。提出CARROT框架，结合MCTS与配置代理，提升检索效率与准确性。**

- **链接: [https://arxiv.org/pdf/2411.00744](https://arxiv.org/pdf/2411.00744)**

> **作者:** Ziting Wang; Haitao Yuan; Wei Dong; Gao Cong; Feifei Li
>
> **备注:** Accepted to ICDE 2026. Updated title (previously "CORAG: A Cost-Constrained Retrieval Optimization System for Retrieval-Augmented Generation")
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive ability in generation and reasoning tasks but struggle with handling up-to-date knowledge, leading to inaccuracies or hallucinations. Retrieval-Augmented Generation (RAG) mitigates this by retrieving and incorporating external knowledge into input prompts. In particular, due to LLMs' context window limitations and long-context hallucinations, only the most relevant "chunks" are retrieved. However, current RAG systems face three key challenges: (1) chunks are often retrieved independently without considering their relationships, such as redundancy and ordering; (2) the utility of chunks is non-monotonic, as adding more chunks can degrade quality; and (3) retrieval strategies fail to adapt to the unique characteristics of different queries. To overcome these challenges, we design a cost-constrained retrieval optimization framework for RAG. We adopt a Monte Carlo Tree Search (MCTS) based strategy to find the optimal chunk combination order, which considers the chunks' correlations. In addition, to address the non-monotonicity of chunk utility, instead of treating budget exhaustion as the termination condition, we design a utility computation strategy to identify the optimal chunk combination without necessarily exhausting the budget. Furthermore, we propose a configuration agent that predicts optimal configurations for each query domain, improving our framework's adaptability and efficiency. Experimental results demonstrate up to a 30% improvement over baseline models, highlighting the framework's effectiveness, scalability, and suitability. Our source code has been released at this https URL.
>
---
#### [replaced 031] SwissGov-RSD: A Human-annotated, Cross-lingual Benchmark for Token-level Recognition of Semantic Differences Between Related Documents
- **分类: cs.CL**

- **简介: 该论文提出SwissGov-RSD，首个跨语言文档级语义差异识别基准，解决多语言文本对齐与评估问题，通过人工标注数据评估模型表现。**

- **链接: [https://arxiv.org/pdf/2512.07538](https://arxiv.org/pdf/2512.07538)**

> **作者:** Michelle Wastl; Jannis Vamvas; Rico Sennrich
>
> **备注:** 30 pages; v2 contains re-annotated subset of EN-DE data
>
> **摘要:** Recognizing semantic differences across documents, especially in different languages, is crucial for text generation evaluation and multilingual content alignment. However, as a standalone task it has received little attention. We address this by introducing SwissGov-RSD, the first naturalistic, document-level, cross-lingual dataset for semantic difference recognition. It encompasses a total of 224 multi-parallel documents in English-German, English-French, and English-Italian with token-level difference annotations by human annotators. We evaluate a variety of open-source and closed source large language models as well as encoder models across different fine-tuning settings on this new benchmark. Our results show that current automatic approaches perform poorly compared to their performance on monolingual, sentence-level, and synthetic benchmarks, revealing a considerable gap for both LLMs and encoder models. We make our code and datasets publicly available.
>
---
#### [replaced 032] Do LLMs Judge Distantly Supervised Named Entity Labels Well? Constructing the JudgeWEL Dataset
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于命名实体识别任务，解决低资源语言数据不足的问题。通过自动标注并验证，构建了更大的Luxembourgish NER数据集。**

- **链接: [https://arxiv.org/pdf/2601.00411](https://arxiv.org/pdf/2601.00411)**

> **作者:** Alistair Plum; Laura Bernardy; Tharindu Ranasinghe
>
> **备注:** Accepted at LREC 2026
>
> **摘要:** We present judgeWEL, a dataset for named entity recognition (NER) in Luxembourgish, automatically labelled and subsequently verified using large language models (LLM) in a novel pipeline. Building datasets for under-represented languages remains one of the major bottlenecks in natural language processing, where the scarcity of resources and linguistic particularities make large-scale annotation costly and potentially inconsistent. To address these challenges, we propose and evaluate a novel approach that leverages Wikipedia and Wikidata as structured sources of weak supervision. By exploiting internal links within Wikipedia articles, we infer entity types based on their corresponding Wikidata entries, thereby generating initial annotations with minimal human intervention. Because such links are not uniformly reliable, we mitigate noise by employing and comparing several LLMs to identify and retain only high-quality labelled sentences. The resulting corpus is approximately five times larger than the currently available Luxembourgish NER dataset and offers broader and more balanced coverage across entity categories, providing a substantial new resource for multilingual and low-resource NER research.
>
---
#### [replaced 033] Knowledge Distillation with Structured Chain-of-Thought for Text-to-SQL
- **分类: cs.CL; cs.AI; cs.DB**

- **简介: 该论文属于Text-to-SQL任务，旨在解决企业部署SQL生成系统时的成本、安全与性能矛盾。通过结构化推理蒸馏方法提升小模型性能。**

- **链接: [https://arxiv.org/pdf/2512.17053](https://arxiv.org/pdf/2512.17053)**

> **作者:** Khushboo Thaker; Yony Bresler
>
> **备注:** Accepted at the 39th Canadian Conference on Artificial Intelligence (Canadian AI 2026). This is the extended version containing additional details and appendices omitted from the camera-ready proceedings due to space constraints
>
> **摘要:** Deploying accurate Text-to-SQL systems at the enterprise level faces a difficult trilemma involving cost, security and performance. Current solutions force enterprises to choose between expensive, proprietary Large Language Models (LLMs) and low-performing Small Language Models (SLMs). Efforts to improve SLMs often rely on distilling reasoning from large LLMs using unstructured Chain-of-Thought (CoT) traces, a process that remains inherently ambiguous. Instead, we hypothesize that a formal, structured reasoning representation provides a clearer, more reliable teaching signal, as the Text-to-SQL task requires explicit and precise logical steps. To evaluate this hypothesis, we propose Struct-SQL, a novel Knowledge Distillation (KD) framework that trains an SLM to emulate a powerful large LLM. Consequently, we adopt a query execution plan as a formal blueprint to derive this structured reasoning. Our SLM, distilled with structured CoT, achieves an absolute improvement of 8.1% over an unstructured CoT distillation baseline. A detailed error analysis reveals that a key factor in this gain is a marked reduction in syntactic errors. This demonstrates that teaching a model to reason using a structured logical blueprint is beneficial for reliable SQL generation in SLMs.
>
---
#### [replaced 034] Llettuce: An Open Source Natural Language Processing Tool for the Translation of Medical Terms into Uniform Clinical Encoding
- **分类: cs.CL**

- **简介: 该论文介绍Llettuce，一个用于将医学术语转换为标准化概念的开源自然语言处理工具，解决医疗术语映射难题。**

- **链接: [https://arxiv.org/pdf/2410.09076](https://arxiv.org/pdf/2410.09076)**

> **作者:** James Mitchell-White; Reza Omdivar; Benjamin Partridge; Esmond Urwin; Karthikeyan Sivakumar; Ruizhe Li; Andy Rae; Xiaoyan Wang; Theresia Mina; Tom Giles; Diego Garcia-Gil; Tim Beck; John Chambers; Grazziela Figueredo; Philip R Quinlan
>
> **摘要:** This paper introduces Llettuce, an open-source tool designed to address the complexities of converting medical terms into OMOP standard concepts. Unlike existing solutions such as the Athena database search and Usagi, which struggle with semantic nuances and require substantial manual input, Llettuce leverages advanced natural language processing, including large language models and fuzzy matching, to automate and enhance the mapping process. Developed with a focus on GDPR compliance, Llettuce can be deployed locally, ensuring data protection while maintaining high performance in converting informal medical terms to standardised concepts.
>
---
#### [replaced 035] EXPLORE-Bench: Egocentric Scene Prediction with Long-Horizon Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出EXPLORE-Bench任务，解决长时序视角下的场景预测问题。通过真实第一人称视频构建基准，评估模型在长期动作序列后的场景预测能力。**

- **链接: [https://arxiv.org/pdf/2603.09731](https://arxiv.org/pdf/2603.09731)**

> **作者:** Chengjun Yu; Xuhan Zhu; Chaoqun Du; Pengfei Yu; Wei Zhai; Yang Cao; Zheng-Jun Zha
>
> **摘要:** Multimodal large language models (MLLMs) are increasingly considered as a foundation for embodied agents, yet it remains unclear whether they can reliably reason about the long-term physical consequences of actions from an egocentric viewpoint. We study this gap through a new task, Egocentric Scene Prediction with LOng-horizon REasoning: given an initial-scene image and a sequence of atomic action descriptions, a model is asked to predict the final scene after all actions are executed. To enable systematic evaluation, we introduce EXPLORE-Bench, a benchmark curated from real first-person videos spanning diverse scenarios. Each instance pairs long action sequences with structured final-scene annotations, including object categories, visual attributes, and inter-object relations, which supports fine-grained, quantitative assessment. Experiments on a range of proprietary and open-source MLLMs reveal a significant performance gap to humans, indicating that long-horizon egocentric reasoning remains a major challenge. We further analyze test-time scaling via stepwise reasoning and show that decomposing long action sequences can improve performance to some extent, while incurring non-trivial computational overhead. Overall, EXPLORE-Bench provides a principled testbed for measuring and advancing long-horizon reasoning for egocentric embodied perception.
>
---
#### [replaced 036] Model-Dowser: Data-Free Importance Probing to Mitigate Catastrophic Forgetting in Multimodal Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于多模态大语言模型的持续学习任务，旨在解决微调过程中导致泛化能力下降的灾难性遗忘问题。提出Model-Dowser方法，通过重要性评估实现高效参数更新。**

- **链接: [https://arxiv.org/pdf/2602.04509](https://arxiv.org/pdf/2602.04509)**

> **作者:** Hyeontaek Hwang; Nguyen Dinh Son; Daeyoung Kim
>
> **摘要:** Fine-tuning Multimodal Large Language Models (MLLMs) on task-specific data is an effective way to improve performance on downstream applications. However, such adaptation often leads to a degradation in generalization on pretrained tasks, a phenomenon known as Catastrophic Forgetting. Existing methods that aim to mitigate this issue either become ineffective when fine-tuning deeper layers of the language decoder or scale poorly with increasing model size. To address these limitations, we propose Model-Dowser, a novel sparse fine-tuning approach for MLLMs. Model-Dowser measures a principled importance score for each model parameter with respect to pretrained generalization (prior to downstream adaptation) by jointly considering weight magnitudes, input activations, and output sensitivities. During fine-tuning, Model-Dowser selectively preserves high-importance parameters and updates the remaining. Comprehensive experiments on two representative MLLMs, LLaVA and NVILA, demonstrate that Model-Dowser effectively mitigates catastrophic forgetting and consistently outperforms prior methods, while remaining resource-efficient and scalable to multi-billion-parameter models.
>
---
#### [replaced 037] Structured Agent Distillation for Large Language Model
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型部署成本高的问题。通过结构化代理蒸馏，将大模型压缩为小模型，保持推理和行动的一致性。**

- **链接: [https://arxiv.org/pdf/2505.13820](https://arxiv.org/pdf/2505.13820)**

> **作者:** Jun Liu; Zhenglun Kong; Peiyan Dong; Changdi Yang; Tianqi Li; Hao Tang; Geng Yuan; Wei Niu; Wenbin Zhang; Pu Zhao; Xue Lin; Dong Huang; Yanzhi Wang
>
> **摘要:** Large language models (LLMs) exhibit strong capabilities as decision-making agents by interleaving reasoning and actions, as seen in ReAct-style frameworks. Yet, their practical deployment is constrained by high inference costs and large model sizes. We propose Structured Agent Distillation, a framework that compresses large LLM-based agents into smaller student models while preserving both reasoning fidelity and action consistency. Unlike standard token-level distillation, our method segments trajectories into {[REASON]} and {[ACT]} spans, applying segment-specific losses to align each component with the teacher's behavior. This structure-aware supervision enables compact agents to better replicate the teacher's decision process. Experiments on ALFWorld, HotPotQA-ReAct, and WebShop show that our approach consistently outperforms token-level and imitation learning baselines, achieving significant compression with minimal performance drop. Scaling and ablation results further highlight the importance of span-level alignment for efficient and deployable agents.
>
---
#### [replaced 038] Reasoning Theater: Disentangling Model Beliefs from Chain-of-Thought
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究模型在推理过程中的行为，旨在区分真实推理与“表演性推理”。通过分析模型激活和回答时机，提出一种基于探测的早期退出方法，提升效率。任务属于模型推理分析与优化。**

- **链接: [https://arxiv.org/pdf/2603.05488](https://arxiv.org/pdf/2603.05488)**

> **作者:** Siddharth Boppana; Annabel Ma; Max Loeffler; Raphael Sarfati; Eric Bigelow; Atticus Geiger; Owen Lewis; Jack Merullo
>
> **摘要:** We provide evidence of performative chain-of-thought (CoT) in reasoning models, where a model becomes strongly confident in its final answer, but continues generating tokens without revealing its internal belief. Our analysis compares activation probing, early forced answering, and a CoT monitor across two large models (DeepSeek-R1 671B & GPT-OSS 120B) and find task difficulty-specific differences: The model's final answer is decodable from activations far earlier in CoT than a monitor is able to say, especially for easy recall-based MMLU questions. We contrast this with genuine reasoning in difficult multihop GPQA-Diamond questions. Despite this, inflection points (e.g., backtracking, 'aha' moments) occur almost exclusively in responses where probes show large belief shifts, suggesting these behaviors track genuine uncertainty rather than learned "reasoning theater." Finally, probe-guided early exit reduces tokens by up to 80% on MMLU and 30% on GPQA-Diamond with similar accuracy, positioning attention probing as an efficient tool for detecting performative reasoning and enabling adaptive computation.
>
---
#### [replaced 039] Hope Speech Detection in code-mixed Roman Urdu tweets: A Positive Turn in Natural Language Processing
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于希望话语检测任务，旨在解决低资源、非正式语言如罗马乌尔都语中的希望话语识别问题。研究构建了首个多类标注数据集，并提出优化模型以提升检测效果。**

- **链接: [https://arxiv.org/pdf/2506.21583](https://arxiv.org/pdf/2506.21583)**

> **作者:** Muhammad Ahmad; Muhammad Waqas; Ameer Hamza; Ildar Batyrshin; Grigori Sidorov
>
> **备注:** We are withdrawing this preprint because it contains initial experimental results and an early version of the manuscript. We are currently improving the methodology, conducting additional experiments, and refining the analysis. A substantially revised version will be submitted in the future
>
> **摘要:** Hope is a positive emotional state involving the expectation of favorable future outcomes, while hope speech refers to communication that promotes optimism, resilience, and support, particularly in adverse contexts. Although hope speech detection has gained attention in Natural Language Processing (NLP), existing research mainly focuses on high-resource languages and standardized scripts, often overlooking informal and underrepresented forms such as Roman Urdu. To the best of our knowledge, this is the first study to address hope speech detection in code-mixed Roman Urdu by introducing a carefully annotated dataset, thereby filling a critical gap in inclusive NLP research for low-resource, informal language varieties. This study makes four key contributions: (1) it introduces the first multi-class annotated dataset for Roman Urdu hope speech, comprising Generalized Hope, Realistic Hope, Unrealistic Hope, and Not Hope categories; (2) it explores the psychological foundations of hope and analyzes its linguistic patterns in code-mixed Roman Urdu to inform dataset development; (3) it proposes a custom attention-based transformer model optimized for the syntactic and semantic variability of Roman Urdu, evaluated using 5-fold cross-validation; and (4) it verifies the statistical significance of performance gains using a t-test. The proposed model, XLM-R, achieves the best performance with a cross-validation score of 0.78, outperforming the baseline SVM (0.75) and BiLSTM (0.76), with gains of 4% and 2.63% respectively.
>
---
#### [replaced 040] Multi-lingual Functional Evaluation for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于多语言模型评估任务，旨在解决静态基准无法准确反映模型多语言性能的问题。通过构建跨语言功能基准，比较不同模型在多种语言中的表现与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2506.20793](https://arxiv.org/pdf/2506.20793)**

> **作者:** Victor Ojewale; Inioluwa Deborah Raji; Suresh Venkatasubramanian
>
> **备注:** This is an updated version with details of the CL-GSM Symbolic and CL-IFEval datasets validation
>
> **摘要:** Multi-lingual competence in large language models is often evaluated via static data benchmarks such as Belebele, M-MMLU and M-GSM. However, these evaluations often fail to provide an adequate understanding of the practical performance and robustness of models across multi-lingual settings. In response, we create multi-lingual functional benchmarks -- Cross-Lingual Grade School Math Symbolic (CL-GSM Symbolic) and Cross-Lingual Instruction-Following Eval (CL-IFEval)-- by translating existing functional benchmark templates from English to five additional languages that span the range of resources available for NLP: French, Spanish, Hindi, Arabic and Yoruba. Our results reveal that some static multi-lingual benchmarks capture functional performance much more closely than others (i.e. across models, there is a 24%, 17% and 18% decrease in performance between M-GSM and CL-GSM Symbolic in English, French and Spanish respectively; similarly there's a 15 - 24% performance drop across languages between Belebele and CL-IFEval, and only a 0.5% to 3% performance drop between M-MMLU and CL-IFEval). Similarly, we find that model robustness across languages varies significantly, with certain languages (eg. Arabic, English) being the most consistently well performing across evaluation iterations.
>
---
#### [replaced 041] [b]=[d]-[t]+[p]: Self-supervised Speech Models Discover Phonological Vector Arithmetic
- **分类: eess.AS; cs.CL; cs.LG; cs.SD**

- **简介: 该论文研究自监督语音模型的表征结构，解决语音信息如何编码的问题。通过分析96种语言，发现模型中存在可解释的音系向量，并展示其可计算性。**

- **链接: [https://arxiv.org/pdf/2602.18899](https://arxiv.org/pdf/2602.18899)**

> **作者:** Kwanghee Choi; Eunjung Yeo; Cheol Jun Cho; David Harwath; David R. Mortensen
>
> **备注:** Submitted to ACL, code planned to release after acceptance
>
> **摘要:** Self-supervised speech models (S3Ms) are known to encode rich phonetic information, yet how this information is structured remains underexplored. We conduct a comprehensive study across 96 languages to analyze the underlying structure of S3M representations, with particular attention to phonological vectors. We first show that there exist linear directions within the model's representation space that correspond to phonological features. We further demonstrate that the scale of these phonological vectors correlate to the degree of acoustic realization of their corresponding phonological features in a continuous manner. For example, the difference between [d] and [t] yields a voicing vector: adding this vector to [p] produces [b], while scaling it results in a continuum of voicing. Together, these findings indicate that S3Ms encode speech using phonologically interpretable and compositional vectors, demonstrating phonological vector arithmetic. All code and interactive demos are available at this https URL .
>
---
#### [replaced 042] Text-only adaptation in LLM-based ASR through text denoising
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文属于语音识别任务，解决LLM在新领域适应的问题。通过文本去噪方法，在不破坏跨模态对齐的情况下，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.20900](https://arxiv.org/pdf/2601.20900)**

> **作者:** Andrés Carofilis; Sergio Burdisso; Esaú Villatoro-Tello; Shashi Kumar; Kadri Hacioglu; Srikanth Madikeri; Pradeep Rangappa; Manjunath K E; Petr Motlicek; Shankar Venkatesan; Andreas Stolcke
>
> **摘要:** Adapting large language model (LLM)-based automatic speech recognition (ASR) systems to new domains using text-only data is a significant yet underexplored challenge. Standard fine-tuning of the LLM on the target domain text often disrupts the critical alignment between the speech and text modality learned by the projector, degrading performance. We introduce a novel text-only adaptation method that frames this process as a text denoising task. Our approach trains the LLM to recover clean transcripts from noisy inputs. This process effectively adapts the model to a target domain while preserving cross-modal alignment. Our solution is lightweight, requiring no architectural changes or additional parameters. Extensive evaluation on two datasets demonstrates up to 22.1% relative improvement, outperforming recent state-of-the-art text-only adaptation methods.
>
---
#### [replaced 043] Measuring Intent Comprehension in LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决LLMs理解用户意图的问题。通过构建框架评估模型在不同提示下的输出一致性，以判断其是否真正理解用户意图。**

- **链接: [https://arxiv.org/pdf/2506.16584](https://arxiv.org/pdf/2506.16584)**

> **作者:** Nadav Kunievsky; James A. Evans
>
> **摘要:** People judge interactions with large language models (LLMs) as successful when outputs match what they want, not what they type. Yet LLMs are trained to predict the next token solely from text input, not underlying intent. Because written language is an imperfect proxy for intent, and correlations between phrasing and desired outcomes can break down in training data, models that rely too heavily on surface cues may respond inconsistently to semantically equivalent prompts. This makes it essential to evaluate whether LLMs can reliably infer user intent-especially in high-stakes settings where robustness and generalization are critical. We introduce a formal framework for assessing intent comprehension in LLMs: whether a model demonstrates robust understanding of user intent by producing consistent outputs across semantically equivalent prompts while differentiating between prompts with distinct intents. Our evaluation approach is based on a variance decomposition of model responses into three components: variability due to user intent, user articulation, and model uncertainty. Models that understand what users want, and are not overly sensitive to textual cues, should attribute most output variance to intent differences, rather than articulation style. Applying this framework across diverse domains, we find that, within the five LLaMA and Gemma models we evaluate, larger models typically assign a greater share of variance to intent, indicating stronger comprehension of intent, although gains are uneven and often modest with increasing model size. These results motivate moving beyond accuracy-only benchmarks toward semantic diagnostics that directly assess whether models understand what users intend.
>
---
#### [replaced 044] Truncated Step-Level Sampling with Process Rewards for Retrieval-Augmented Reasoning
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于自然语言处理中的问答任务，解决大模型与搜索引擎结合时的信用分配问题。提出SLATE框架，通过截断采样和密集奖励机制，提升推理与检索效果。**

- **链接: [https://arxiv.org/pdf/2602.23440](https://arxiv.org/pdf/2602.23440)**

> **作者:** Chris Samarinas; Haw-Shiuan Chang; Hamed Zamani
>
> **摘要:** Training large language models to reason with search engines via reinforcement learning is hindered by a fundamental credit assignment problem: existing methods such as Search-R1 provide only a sparse outcome reward after an entire multi-step trajectory, making it infeasible to attribute success or failure to individual reasoning and retrieval decisions. Process-reward methods like StepSearch alleviate this by introducing step-level supervision, but rely on heuristic rewards such as TF-IDF overlap with gold documents, and still sample $k$ complete trajectories per example, retaining high gradient variance. We propose SLATE, a framework built on two complementary ideas: (1) truncated step-level sampling, which generates $k$ trajectories that share a common prefix and differ only at the next step, isolating variation to a single decision point; and (2) dense, decomposed LLM-as-judge rewards, which score each reasoning step, search query, and answer on a ternary scale with separate quality dimensions, providing richer supervision than binary outcome signals or undifferentiated step-level judgments. We theoretically prove that under the same dense reward structure, truncated sampling reduces the variance of advantage estimates by up to a factor of $T$ compared to full-trajectory sampling for $T$-step trajectories, yielding lower-variance and better-targeted policy gradients. Experiments on seven QA benchmarks confirm that SLATE consistently outperforms both sparse-reward and process-reward baselines, with the largest gains on harder multi-hop tasks and smaller models.
>
---
#### [replaced 045] Let's Verify Math Questions Step by Step
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于数学问题验证任务，旨在解决数学问题质量评估问题。通过构建ValiMath基准和MathQ-Verify管道，提升数学数据集的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2505.13903](https://arxiv.org/pdf/2505.13903)**

> **作者:** Chengyu Shen; Zhen Hao Wong; Runming He; Hao Liang; Meiyi Qiang; Zimo Meng; Zhengyang Zhao; Bohan Zeng; Zhengzhou Zhu; Bin Cui; Wentao Zhang
>
> **摘要:** Large Language Models (LLMs) have recently achieved remarkable progress in mathematical reasoning. To enable such capabilities, many existing works distill strong reasoning models into long chains of thought or design algorithms to construct high-quality math question-answer (QA) data for training. However, these efforts primarily focus on generating correct reasoning paths and answers, while largely overlooking the correctness of the questions themselves. In this work, we present ValiMath, a benchmark consisting of 2147 human-verified mathematical questions covering a wide range of domains such as arithmetic, algebra, and geometry, which are synthesized and curated from the NuminaMath dataset. Each question is annotated with its logical structure, domain coverage, and question correctness, enabling fine-grained evaluation of question quality. ValiMath serves as a high-quality gold-standard test set for validating mathematical questions in LLM training corpora. Building upon this benchmark, we further propose MathQ-Verify, a pipeline that performs fine-grained parsing of mathematical questions into atomic assumptions and conclusions, and evaluates their semantic soundness through consistency checks. This pipeline achieves high precision in detecting flawed questions and provides a reliable foundation for cleaning noisy mathematical datasets. Experiments show that MathQ-Verify achieves state-of-the-art performance across multiple benchmarks, improving the F1 score by up to 25 percentage points over the direct verification baseline. MathQ-Verify offers a scalable and accurate solution for curating reliable mathematical datasets, reducing label noise and avoiding unnecessary computation on invalid questions. Our code and data are available at the repository this https URL.
>
---
#### [replaced 046] Jr. AI Scientist and Its Risk Report: Autonomous Scientific Exploration from a Baseline Paper
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文属于AI科学探索任务，旨在研究AI科学家系统的性能与风险。工作包括开发Jr. AI Scientist，模拟新手研究流程，生成新论文，并评估其效果与局限性。**

- **链接: [https://arxiv.org/pdf/2511.04583](https://arxiv.org/pdf/2511.04583)**

> **作者:** Atsuyuki Miyai; Mashiro Toyooka; Takashi Otonari; Zaiying Zhao; Kiyoharu Aizawa
>
> **备注:** TMLR2026. Issues, comments, and questions are all welcome in this https URL
>
> **摘要:** Understanding the current capabilities and risks of AI Scientist systems (autoresearch) is essential for ensuring trustworthy and sustainable AI-driven scientific progress while preserving the integrity of the academic ecosystem. To this end, we develop Jr. AI Scientist, a state-of-the-art autonomous AI scientist system that mimics the core research workflow of a novice student researcher: Given the baseline paper from the human mentor, it analyzes its limitations, formulates novel hypotheses for improvement, iteratively experiments until improvements are achieved, and writes a paper with the results. Unlike previous approaches that assume full automation or operate on small-scale code, Jr. AI Scientist follows a well-defined research workflow and leverages modern coding agents to handle complex, multi-file implementations, leading to scientifically valuable contributions. Through our experiments, the Jr. AI Scientist successfully generated new research papers that build upon real NeurIPS, IJCV, and ICLR works by proposing and implementing novel methods. For evaluation, we conducted automated assessments using AI Reviewers, author-led evaluations, and submissions to Agents4Science, a venue dedicated to AI-driven contributions. The findings demonstrate that Jr. AI Scientist generates papers receiving higher review scores by DeepReviewer than existing fully automated systems. Nevertheless, we identify important limitations from the author evaluation and the Agents4Science reviews, indicating the potential risks of directly applying current AI Scientist systems and key challenges for future research. Finally, we comprehensively report various risks identified during development. We believe this study clarifies the current role and limitations of AI Scientist systems, offering insights into the areas that still require human expertise and the risks that may emerge as these systems evolve.
>
---
#### [replaced 047] Consistency of Large Reasoning Models Under Multi-Turn Attacks
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于安全与鲁棒性研究，探讨大推理模型在多轮攻击下的一致性问题。工作包括评估模型脆弱性、分析失败模式，并提出对信心机制的重新设计。**

- **链接: [https://arxiv.org/pdf/2602.13093](https://arxiv.org/pdf/2602.13093)**

> **作者:** Yubo Li; Ramayya Krishnan; Rema Padman
>
> **摘要:** Large reasoning models with reasoning capabilities achieve state-of-the-art performance on complex tasks, but their robustness under multi-turn adversarial pressure remains underexplored. We evaluate nine frontier reasoning models under adversarial attacks. Our findings reveal that reasoning confers meaningful but incomplete robustness: most reasoning models studied significantly outperform instruction-tuned baselines, yet all exhibit distinct vulnerability profiles, with misleading suggestions universally effective and social pressure showing model-specific efficacy. Through trajectory analysis, we identify five failure modes (Self-Doubt, Social Conformity, Suggestion Hijacking, Emotional Susceptibility, and Reasoning Fatigue) with the first two accounting for 50% of failures. We further demonstrate that Confidence-Aware Response Generation (CARG), effective for standard LLMs, fails for reasoning models due to overconfidence induced by extended reasoning traces; counterintuitively, random confidence embedding outperforms targeted extraction. Our results highlight that reasoning capabilities do not automatically confer adversarial robustness and that confidence-based defenses require fundamental redesign for reasoning models.
>
---
#### [replaced 048] Mechanistic Indicators of Steering Effectiveness in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于模型解释任务，旨在解决激活干预有效性的机制理解问题。通过分析内部信号，如NBF和KL散度，评估 steering 成功与否。**

- **链接: [https://arxiv.org/pdf/2602.01716](https://arxiv.org/pdf/2602.01716)**

> **作者:** Mehdi Jafari; Hao Xue; Flora Salim
>
> **摘要:** Activation-based steering enables Large Language Models (LLMs) to exhibit targeted behaviors by intervening on intermediate activations without retraining. Despite its widespread use, the mechanistic factors that govern when steering succeeds or fails remain poorly understood, as prior work has relied primarily on black-box outputs or LLM-based judges. In this study, we investigate whether the reliability of steering can be diagnosed using internal model signals. We focus on two information-theoretic measures: the entropy-derived Normalized Branching Factor (NBF), and the Kullback-Leibler (KL) divergence between steered activations and targeted concepts in the vocabulary space. We hypothesize that effective steering corresponds to structured entropy preservation and coherent KL alignment across decoding steps. Building on a reliability study demonstrating high inter-judge agreement between two architecturally distinct LLMs, we use LLM-generated annotations as ground truth and show that these mechanistic signals provide meaningful predictive power for identifying successful steering and estimating failure probability. We further introduce a stronger evaluation baseline for Contrastive Activation Addition (CAA) and Sparse Autoencoder-based steering, the two most widely adopted activation-steering methods.
>
---
#### [replaced 049] Reasoning Boosts Opinion Alignment in LLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于意见建模任务，旨在提升大语言模型在政治观点上的一致性。研究通过结构化推理减少偏见，验证了其有效性，但未完全消除偏差。**

- **链接: [https://arxiv.org/pdf/2603.01214](https://arxiv.org/pdf/2603.01214)**

> **作者:** Frédéric Berdoz; Yann Billeter; Yann Vonlanthen; Roger Wattenhofer
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Opinion modeling aims to capture individual or group political preferences, enabling applications such as digital democracies, where models could help shape fairer and more popular policies. Given their versatility, strong generalization capabilities, and demonstrated success across diverse text-to-text applications, large language models (LLMs) are natural candidates for this task. However, due to their statistical nature and limited causal understanding, they tend to produce biased opinions when prompted naively. In this work, we study whether reasoning can improve opinion alignment. Motivated by the recent advancement in mathematical reasoning enabled by reinforcement learning (RL), we train models to produce profile-consistent answers through structured reasoning. We evaluate our approach on three datasets covering U.S., European, and Swiss politics. Results indicate that reasoning enhances opinion modeling and is competitive with strong baselines, but does not fully remove bias, highlighting the need for additional mechanisms to build faithful political digital twins using LLMs. By releasing both our method and datasets, we establish a solid baseline to support future research on LLM opinion alignment.
>
---
#### [replaced 050] RECAP: Reproducing Copyrighted Data from LLMs Training with an Agentic Pipeline
- **分类: cs.CL**

- **简介: 该论文属于数据提取任务，旨在从大语言模型中还原训练数据。通过构建RECAP系统，利用反馈机制和破解模块提高提取效果。**

- **链接: [https://arxiv.org/pdf/2510.25941](https://arxiv.org/pdf/2510.25941)**

> **作者:** André V. Duarte; Xuying li; Bin Zeng; Arlindo L. Oliveira; Lei Li; Zhuo Li
>
> **摘要:** If we cannot inspect the training data of a large language model (LLM), how can we ever know what it has seen? We believe the most compelling evidence arises when the model itself freely reproduces the target content. As such, we propose RECAP, an agentic pipeline designed to elicit and verify memorized training data from LLM outputs. At the heart of RECAP is a feedback-driven loop, where an initial extraction attempt is evaluated by a secondary language model, which compares the output against a reference passage and identifies discrepancies. These are then translated into minimal correction hints, which are fed back into the target model to guide subsequent generations. In addition, to address alignment-induced refusals, RECAP includes a jailbreaking module that detects and overcomes such barriers. We evaluate RECAP on EchoTrace, a new benchmark spanning over 30 full books, and the results show that RECAP leads to substantial gains over single-iteration approaches. For instance, with GPT-4.1, the average ROUGE-L score for the copyrighted text extraction improved from 0.38 to 0.47 - a nearly 24% increase.
>
---
#### [replaced 051] LatentChem: From Textual CoT to Latent Thinking in Chemical Reasoning
- **分类: physics.chem-ph; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于化学推理任务，旨在解决传统方法在化学推理中效率低、性能受限的问题。工作是提出LatentChem，通过连续潜在空间进行推理，提升效率和效果。**

- **链接: [https://arxiv.org/pdf/2602.07075](https://arxiv.org/pdf/2602.07075)**

> **作者:** Xinwu Ye; Yicheng Mao; Jia Zhang; Yimeng Liu; Li Hao; Fang Wu; Zhiwei Li; Yuxuan Liao; Zehong Wang; Zhiyuan Liu; Zhenfei Yin; Li Yuan; Philip Torr; Huan Sun; Xiangxiang Zeng; Mengdi Wang; Le Cong; Shenghua Gao; Xiangru Tang
>
> **摘要:** Chemical large language models (LLMs) predominantly rely on explicit Chain-of-Thought (CoT) in natural language to perform complex reasoning. However, chemical reasoning is inherently continuous and structural, and forcing it into discrete linguistic tokens introduces a fundamental representation mismatch that constrains both efficiency and performance. We introduce LatentChem, a latent reasoning interface that decouples chemical computation from textual generation, enabling models to perform multi-step reasoning directly in continuous latent space while emitting language only for final outputs. Remarkably, we observe a consistent emergent behavior: when optimized solely for task success, models spontaneously internalize reasoning, progressively abandoning verbose textual derivations in favor of implicit latent computation. This shift is not merely stylistic but computationally advantageous. Across diverse chemical reasoning benchmarks, LatentChem achieves a 59.88\% non-tie win rate over strong CoT-based baselines on ChemCoTBench, while delivering a 10.84$\times$ average inference speedup. Our results provide empirical evidence that chemical reasoning is more naturally and effectively realized as continuous latent dynamics rather than discretized linguistic trajectories.
>
---
#### [replaced 052] Evolving Beyond Snapshots: Harmonizing Structure and Sequence via Entity State Tuning for Temporal Knowledge Graph Forecasting
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于时间知识图谱预测任务，解决现有方法因无状态导致的长期依赖衰减问题。提出Entity State Tuning框架，通过持续更新实体状态提升预测性能。**

- **链接: [https://arxiv.org/pdf/2602.12389](https://arxiv.org/pdf/2602.12389)**

> **作者:** Siyuan Li; Yunjia Wu; Yiyong Xiao; Pingyang Huang; Peize Li; Ruitong Liu; Yan Wen; Te Sun; Fangyi Pei
>
> **摘要:** Temporal knowledge graph (TKG) forecasting requires predicting future facts by jointly modeling structural dependencies within each snapshot and temporal evolution across snapshots. However, most existing methods are stateless: they recompute entity representations at each timestamp from a limited query window, leading to episodic amnesia and rapid decay of long-term dependencies. To address this limitation, we propose Entity State Tuning (EST), an encoder-agnostic framework that endows TKG forecasters with persistent and continuously evolving entity states. EST maintains a global state buffer and progressively aligns structural evidence with sequential signals via a closed-loop design. Specifically, a topology-aware state perceiver first injects entity-state priors into structural encoding. Then, a unified temporal context module aggregates the state-enhanced events with a pluggable sequence backbone. Subsequently, a dual-track evolution mechanism writes the updated context back to the global entity state memory, balancing plasticity against stability. Experiments on multiple benchmarks show that EST consistently improves diverse backbones and achieves state-of-the-art performance, highlighting the importance of state persistence for long-horizon TKG forecasting. The code is published at this https URL.
>
---
#### [replaced 053] Critique-Coder: Enhancing Coder Models by Critique Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于语言模型训练任务，旨在提升模型的推理与批判能力。通过引入批判强化学习（CRL），结合标准RL数据，提出Critique-Coder模型，显著提升了代码生成和逻辑推理性能。**

- **链接: [https://arxiv.org/pdf/2509.22824](https://arxiv.org/pdf/2509.22824)**

> **作者:** Chi Ruan; Dongfu Jiang; Yubo Wang; Wenhu Chen
>
> **摘要:** Reinforcement Learning (RL) has emerged as a popular training paradigm, particularly when paired with reasoning models. While effective, it primarily focuses on generating responses and lacks mechanisms to explicitly foster critique or reflection. Several recent studies, like Critique-Fine-Tuning (CFT) and Critique-Guided-Distillation (CGD) have shown the benefits of explicitly teaching LLMs how to critique. Motivated by them, we propose Critique Reinforcement Learning (CRL), where the model is tasked with generating a critique for a given (question, solution) pair. The reward is determined solely by whether the final judgment label $c \in \{\texttt{True}, \texttt{False}\}$ of the generated critique aligns with the ground-truth judgment $c^*$. Building on this point, we introduce Critique-Coder, which is trained on a hybrid of RL and CRL by substituting 20% of the standard RL data with CRL data. We fine-tune multiple models (Critique-Coder) and evaluate them on different benchmarks to show their advantages over RL-only models. We show that Critique-Coder consistently outperforms RL-only baselines on all the evaluated benchmarks. Notably, our Critique-Coder-8B can reach over 60% on LiveCodeBench (v5), outperforming other reasoning models like DeepCoder-14B and GPT-o1. Beyond code generation, Critique-Coder also demonstrates enhanced general reasoning abilities, as evidenced by its better performance on logic reasoning tasks from the BBEH dataset. This indicates that the application of CRL on coding datasets enhances general reasoning and critique abilities, which are transferable across a broad range of tasks. Hence, we believe that CRL works as a great complement to standard RL for LLM reasoning.
>
---
#### [replaced 054] SENS-ASR: Semantic Embedding injection in Neural-transducer for Streaming Automatic Speech Recognition
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语音识别任务，旨在提升流式ASR的准确性。针对低延迟下语义信息不足的问题，提出SENS-ASR方法，通过注入语义信息增强声学特征。**

- **链接: [https://arxiv.org/pdf/2603.10005](https://arxiv.org/pdf/2603.10005)**

> **作者:** Youness Dkhissi; Valentin Vielzeuf; Elys Allesiardo; Anthony Larcher
>
> **摘要:** Many Automatic Speech Recognition (ASR) applications require streaming processing of the audio data. In streaming mode, ASR systems need to start transcribing the input stream before it is complete, i.e., the systems have to process a stream of inputs with a limited (or no) future context. Compared to offline mode, this reduction of the future context degrades the performance of Streaming-ASR systems, especially while working with low-latency constraint. In this work, we present SENS-ASR, an approach to enhance the transcription quality of Streaming-ASR by reinforcing the acoustic information with semantic information. This semantic information is extracted from the available past frame-embeddings by a context module. This module is trained using knowledge distillation from a sentence embedding Language Model fine-tuned on the training dataset transcriptions. Experiments on standard datasets show that SENS-ASR significantly improves the Word Error Rate on small-chunk streaming scenarios.
>
---
#### [replaced 055] Partially Recentralization Softmax Loss for Vision-Language Models Robustness
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态自然语言处理任务，旨在提升视觉-语言模型的对抗鲁棒性。通过修改损失函数，限制softmax输出，增强模型对对抗攻击的抵抗能力。**

- **链接: [https://arxiv.org/pdf/2402.03627](https://arxiv.org/pdf/2402.03627)**

> **作者:** Hao Wang; Jinzhe Jiang; Xin Zhang; Chen Li
>
> **备注:** The study described in Section 4 was conducted without required institutional review board approval. The paper is withdrawn pending completion of the approval process
>
> **摘要:** As Large Language Models make a breakthrough in natural language processing tasks (NLP), multimodal technique becomes extremely popular. However, it has been shown that multimodal NLP are vulnerable to adversarial attacks, where the outputs of a model can be dramatically changed by a perturbation to the input. While several defense techniques have been proposed both in computer vision and NLP models, the multimodal robustness of models have not been fully explored. In this paper, we study the adversarial robustness provided by modifying loss function of pre-trained multimodal models, by restricting top K softmax outputs. Based on the evaluation and scoring, our experiments show that after a fine-tuning, adversarial robustness of pre-trained models can be significantly improved, against popular attacks. Further research should be studying, such as output diversity, generalization and the robustness-performance trade-off of this kind of loss functions. Our code will be available after this paper is accepted
>
---
#### [replaced 056] ReasonMap: Towards Fine-Grained Visual Reasoning from Transit Maps
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出ReasonMap基准，用于评估多模态大模型在交通图细粒度视觉推理的能力。旨在解决视觉推理与模型性能差异问题，通过设计数据集和评估方法进行分析。**

- **链接: [https://arxiv.org/pdf/2505.18675](https://arxiv.org/pdf/2505.18675)**

> **作者:** Sicheng Feng; Song Wang; Shuyi Ouyang; Lingdong Kong; Zikai Song; Jianke Zhu; Huan Wang; Xinchao Wang
>
> **备注:** CVPR 2026, website: this https URL
>
> **摘要:** Multimodal large language models (MLLMs) have demonstrated significant progress in semantic scene understanding and text-image alignment, with reasoning variants enhancing performance on more complex tasks involving mathematics and logic. To bridge this gap, we introduce ReasonMap, a novel benchmark specifically designed to evaluate these capabilities. ReasonMap encompasses high-resolution transit maps from 30 cities and includes 1,008 question-answer pairs spanning two question types and three templates. Furthermore, we design a two-level evaluation pipeline that properly assesses answer correctness and quality. Our comprehensive evaluation of 16 popular MLLMs reveals a counterintuitive pattern: among open-source models, base variants outperform their reasoning-tuned counterparts, whereas the opposite trend is observed in closed-source models. Further analysis under the visual-masking setting confirms that strong performance necessitates direct visual grounding, rather than relying solely on language priors. We further establish a training baseline with reinforcement fine-tuning, providing a reference for future exploration. We hope this benchmark study offers new insights into visual reasoning and helps investigate the gap between open- and closed-source models.
>
---
