# 自然语言处理 cs.CL

- **最新发布 83 篇**

- **更新 82 篇**

## 最新发布

#### [new 001] MUSCAT: MUltilingual, SCientific ConversATion Benchmark
- **分类: cs.CL**

- **简介: 该论文提出MUSCAT基准，用于评估多语言语音识别系统处理多语言输入、专业词汇和语码转换的能力，旨在解决多语言交流中的技术挑战。**

- **链接: [https://arxiv.org/pdf/2604.15929](https://arxiv.org/pdf/2604.15929)**

> **作者:** Supriti Sinhamahapatra; Thai-Binh Nguyen; Yiğit Oğuz; Enes Ugan; Jan Niehues; Alexander Waibel
>
> **摘要:** The goal of multilingual speech technology is to facilitate seamless communication between individuals speaking different languages, creating the experience as though everyone were a multilingual speaker. To create this experience, speech technology needs to address several challenges: Handling mixed multilingual input, specific vocabulary, and code-switching. However, there is currently no dataset benchmarking this situation. We propose a new benchmark to evaluate current Automatic Speech Recognition (ASR) systems, whether they are able to handle these challenges. The benchmark consists of bilingual discussions on scientific papers between multiple speakers, each conversing in a different language. We provide a standard evaluation framework, beyond Word Error Rate (WER) enabling consistent comparison of ASR performance across languages. Experimental results demonstrate that the proposed dataset is still an open challenge for state-of-the-art ASR systems. The dataset is available in this https URL \\ \newline \Keywords{multilingual, speech recognition, audio segmentation, speaker diarization}
>
---
#### [new 002] Sentiment Analysis of German Sign Language Fairy Tales
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于情感分析任务，旨在解决德语手语童话的情感分类问题。通过提取面部和身体动作特征，训练可解释模型进行情感预测。**

- **链接: [https://arxiv.org/pdf/2604.16138](https://arxiv.org/pdf/2604.16138)**

> **作者:** Fabrizio Nunnari; Siddhant Jain; Patrick Gebhard
>
> **摘要:** We present a dataset and a model for sentiment analysis of German sign language (DGS) fairy tales. First, we perform sentiment analysis for three levels of valence (negative, neutral, positive) on German fairy tales text segments using four large language models (LLMs) and majority voting, reaching an inter-annotator agreement of 0.781 Krippendorff's alpha. Second, we extract face and body motion features from each corresponding DGS video segment using MediaPipe. Finally, we train an explainable model (based on XGBoost) to predict negative, neutral or positive sentiment from video features. Results show an average balanced accuracy of 0.631. A thorough analysis of the most important features reveal that, in addition to eyebrows and mouth motion on the face, also the motion of hips, elbows, and shoulders considerably contribute in the discrimination of the conveyed sentiment, indicating an equal importance of face and body for sentiment communication in sign language.
>
---
#### [new 003] Language, Place, and Social Media: Geographic Dialect Alignment in New Zealand
- **分类: cs.CL**

- **简介: 该论文研究社交媒体中语言与地域的关联，分析新西兰Reddit社区的语言特征。任务是探索语言变化与地方认同的关系，通过语料分析和语言模型揭示语义差异。**

- **链接: [https://arxiv.org/pdf/2604.15744](https://arxiv.org/pdf/2604.15744)**

> **作者:** Sidney Wong
>
> **备注:** PhD thesis
>
> **摘要:** This thesis investigates geographic dialect alignment in place-informed social media communities, focussing on New Zealand-related Reddit communities. By integrating qualitative analyses of user perceptions with computational methods, the study examines how language use reflects place identity and patterns of language variation and change based on user-informed lexical, morphosyntactic, and semantic variables. The findings show that users generally associate language with place, and place-related communities form a contiguous speech community, though alignment between geographic dialect communities and place-related communities remains complex. Advanced language modelling, including static and diachronic Word2Vec language embeddings, revealed semantic variation across place-based communities and meaningful semantic shifts within New Zealand English. The research involved the creation of a corpus containing 4.26 billion unprocessed words, which offers a valuable resource for future study. Overall, the results highlight the potential of social media as a natural laboratory for sociolinguistic inquiry.
>
---
#### [new 004] Brain Score Tracks Shared Properties of Languages: Evidence from Many Natural Languages and Structured Sequences
- **分类: cs.CL**

- **简介: 该论文属于语言模型与人类语言处理比较任务，旨在评估模型与人类处理的相似性。通过Brain Score框架，研究不同数据训练的模型表现，探讨其共性结构提取能力。**

- **链接: [https://arxiv.org/pdf/2604.15503](https://arxiv.org/pdf/2604.15503)**

> **作者:** Jingnong Qu; Ashvin Ranjan; Shane Steinert-Threlkeld
>
> **摘要:** Recent breakthroughs in language models (LMs) using neural networks have raised the question: how similar are these models' processing to human language processing? Results using a framework called Brain Score (BS) -- predicting fMRI activations during reading from LM activations -- have been used to argue for a high degree of similarity. To understand this similarity, we conduct experiments by training LMs on various types of input data and evaluate them on BS. We find that models trained on various natural languages from many different language families have very similar BS performance. LMs trained on other structured data -- the human genome, Python, and pure hierarchical structure (nested parentheses) -- also perform reasonably well and close to natural languages in some cases. These findings suggest that BS can highlight language models' ability to extract common structure across natural languages, but that the metric may not be sensitive enough to allow us to infer human-like processing from a high BS score alone.
>
---
#### [new 005] Disentangling Mathematical Reasoning in LLMs: A Methodological Investigation of Internal Mechanisms
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究LLMs在数学推理中的内部机制。旨在揭示模型如何执行算术任务，通过分析注意力和MLP模块的作用，发现高效模型存在分工协作现象。**

- **链接: [https://arxiv.org/pdf/2604.15842](https://arxiv.org/pdf/2604.15842)**

> **作者:** Tanja Baeumel; Josef van Genabith; Simon Ostermann
>
> **备注:** MathNLP 2025
>
> **摘要:** Large language models (LLMs) have demonstrated impressive capabilities, yet their internal mechanisms for handling reasoning-intensive tasks remain underexplored. To advance the understanding of model-internal processing mechanisms, we present an investigation of how LLMs perform arithmetic operations by examining internal mechanisms during task execution. Using early decoding, we trace how next-token predictions are constructed across layers. Our experiments reveal that while the models recognize arithmetic tasks early, correct result generation occurs only in the final layers. Notably, models proficient in arithmetic exhibit a clear division of labor between attention and MLP modules, where attention propagates input information and MLP modules aggregate it. This division is absent in less proficient models. Furthermore, successful models appear to process more challenging arithmetic tasks functionally, suggesting reasoning capabilities beyond factual recall.
>
---
#### [new 006] AtManRL: Towards Faithful Reasoning via Differentiable Attention Saliency
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型推理过程不透明的问题。通过引入AtManRL方法，利用注意力机制和强化学习提升推理的可解释性与准确性。**

- **链接: [https://arxiv.org/pdf/2604.16158](https://arxiv.org/pdf/2604.16158)**

> **作者:** Max Henning Höth; Kristian Kersting; Björn Deiseroth; Letitia Parcalabescu
>
> **备注:** 14 pages, 8 figures, 1 table
>
> **摘要:** Large language models (LLMs) increasingly rely on chain-of-thought (CoT) reasoning to solve complex tasks. Yet ensuring that the reasoning trace both contributes to and faithfully reflects the processes underlying the model's final answer, rather than merely accompanying it, remains challenging. We introduce AtManRL, a method that leverages differentiable attention manipulation to learn more faithful reasoning through reinforcement learning. By training an additive attention mask that identifies tokens in the CoT crucial for producing correct answers, we derive a saliency reward signal that encourages the model to generate reasoning traces that genuinely influence its final predictions. We integrate this saliency reward with outcome-based rewards within the GRPO framework to jointly optimize for correctness and interpretability. Experiments on GSM8K and MMLU with Llama-3.2-3B-Instruct demonstrate that our approach can identify influential reasoning tokens and enable training more transparent reasoning models.
>
---
#### [new 007] HyperGVL: Benchmarking and Improving Large Vision-Language Models in Hypergraph Understanding and Reasoning
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型任务，旨在解决LVLM在超图理解与推理上的能力评估问题。提出首个基准HyperGVL，并引入优化方法WiseHyGR提升性能。**

- **链接: [https://arxiv.org/pdf/2604.15648](https://arxiv.org/pdf/2604.15648)**

> **作者:** Yanbin Wei; Chun Kang; Siwei Li; Haoxuan Che; Yang Chen; Hua Liu; Jian Liu; Zhuang Liu; Can Ouyang; Fei Xing; Lei Sha; Rui Liu; Yu Zhang; James Kwok
>
> **备注:** Under Review; Opensource after accepted
>
> **摘要:** Large Vision-Language Models (LVLMs) consistently require new arenas to guide their expanding boundaries, yet their capabilities with hypergraphs remain unexplored. In the real world, hypergraphs have significant practical applications in areas such as life sciences and social communities. Recent advancements in LVLMs have shown promise in understanding complex topologies, yet there remains a lack of a benchmark to delineate the capabilities of LVLMs with hypergraphs, leaving the boundaries of their abilities unclear. To fill this gap, in this paper, we introduce $\texttt{HyperGVL}$, the first benchmark to evaluate the proficiency of LVLMs in hypergraph understanding and reasoning. $\texttt{HyperGVL}$ provides a comprehensive assessment of 12 advanced LVLMs across 84,000 vision-language question-answering (QA) samples spanning 12 tasks, ranging from basic component counting to complex NP-hard problem reasoning. The involved hypergraphs contain multiscale synthetic structures and real-world citation and protein networks. Moreover, we examine the effects of 12 textual and visual hypergraph representations and introduce a generalizable router $\texttt{WiseHyGR}$ that improves LVLMs in hypergraph via learning adaptive representations. We believe that this work is a step forward in connecting hypergraphs with LVLMs.
>
---
#### [new 008] CHOP: Chunkwise Context-Preserving Framework for RAG on Multi Documents
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，解决多文档RAG系统中因相似文档导致的检索混淆问题。提出CHOP框架，通过Chunkwise方法提升检索准确性和文档连贯性。**

- **链接: [https://arxiv.org/pdf/2604.15802](https://arxiv.org/pdf/2604.15802)**

> **作者:** Hyunseok Park; Jihyeon Kim; Jongeun Kim; Dongsik Yoon
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems lose retrieval accuracy when similar documents coexist in the vector database, causing unnecessary information, hallucinations, and factual errors. To alleviate this issue, we propose CHOP, a framework that iteratively evaluates chunk relevance with Large Language Models (LLMs) and progressively reconstructs documents by determining their association with specific topics or query types. CHOP integrates two key components: the CNM-Extractor, which generates compact per-chunk signatures capturing categories, key nouns, and model names, and the Continuity Decision Module, which preserves contextual coherence by deciding whether consecutive chunks belong to the same document flow. By prefixing each chunk with context-aware metadata, CHOP reduces semantic conflicts among similar documents and enhances retriever discrimination. Experiments on benchmark datasets show that CHOP alleviates retrieval confusion and provides a scalable approach for building high-quality knowledge bases, achieving a Top-1 Hit Rate of 90.77% and notable gains in ranking quality metrics.
>
---
#### [new 009] DALM: A Domain-Algebraic Language Model via Three-Phase Structured Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DALM，一种基于领域代数的生成模型，解决多领域知识干扰问题。通过三阶段结构化去噪，实现领域隔离与约束生成。属于自然语言生成任务。**

- **链接: [https://arxiv.org/pdf/2604.15593](https://arxiv.org/pdf/2604.15593)**

> **作者:** Chao Li
>
> **摘要:** Large language models compress heterogeneous knowledge into a single parameter space, allowing facts from different domains to interfere during generation. We propose DALM, a Domain-Algebraic Language Model that replaces unconstrained token generation with structured denoising over a domain lattice. DALM follows a three-phase generation path: it first resolves domain uncertainty, then relation uncertainty, and finally concept uncertainty, so each stage operates under explicit algebraic constraints. The framework requires only three ingredients: a lattice of domains with computable meet, join, and implication; a typing function over relations that controls inheritance across domains; and a fiber partition that localizes knowledge to domain-specific subsets. Given these ingredients, DALM yields a three-phase encoder-decoder architecture in which generation is confined to a domain fiber, cross-domain contamination is structurally prevented in closed-vocabulary mode and auditably bounded in open-vocabulary mode, and a single query can produce a domain-indexed multi-perspective answer space. We instantiate the framework with the CDC knowledge representation system and outline training and evaluation on validated domain-annotated crystal libraries. DALM reframes language generation as algebraically constrained structured denoising rather than unconstrained decoding over a flat token space.
>
---
#### [new 010] Improving Reasoning Capabilities in Small Models through Mixture-of-Layers Distillation with Stepwise Attention on Key Information
- **分类: cs.CL**

- **简介: 该论文属于模型压缩任务，旨在提升小模型的推理能力。通过迁移教师模型在关键信息上的逐步注意力，增强学生模型的推理效果。**

- **链接: [https://arxiv.org/pdf/2604.15701](https://arxiv.org/pdf/2604.15701)**

> **作者:** Yao Chen; Jiawei Sheng; Wenyuan Zhang; Tingwen Liu
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** The significant computational demands of large language models have increased interest in distilling reasoning abilities into smaller models via Chain-of-Thought (CoT) distillation. Current CoT distillation methods mainly focus on transferring teacher-generated rationales for complex reasoning to student models. However, they do not adequately explore teachers' dynamic attention toward critical information during reasoning. We find that language models exhibit progressive attention shifts towards key information during reasoning, which implies essential clues for drawing conclusions. Building on this observation and analysis, we introduce a novel CoT distillation framework that transfers the teacher's stepwise attention on key information to the student model. This establishes structured guidance for the student's progressive concentration on key information during reasoning. More importantly, we develop a Mixture of Layers module enabling dynamic alignment that adapts to different layers between the teacher and student. Our method achieves consistent performance improvements across multiple mathematical and commonsense reasoning datasets. To our knowledge, it is the first method to leverage stepwise attention within CoT distillation to improve small model reasoning.
>
---
#### [new 011] CiPO: Counterfactual Unlearning for Large Reasoning Models through Iterative Preference Optimization
- **分类: cs.CL**

- **简介: 该论文属于机器学习中的模型去隐私任务，旨在解决大型推理模型中难以有效删除特定知识的问题。提出CiPO框架，通过迭代优化实现精准去学习，同时保持模型推理能力。**

- **链接: [https://arxiv.org/pdf/2604.15847](https://arxiv.org/pdf/2604.15847)**

> **作者:** Junyi Li; Yongqiang Chen; Ningning Ding
>
> **备注:** Accepted by ACL 2026 Main Conference
>
> **摘要:** Machine unlearning has gained increasing attention in recent years, as a promising technique to selectively remove unwanted privacy or copyrighted information from Large Language Models that are trained on a massive scale of human data. However, the emergence of Large Reasoning Models (LRMs), which emphasize long chain-of-thought (CoT) reasoning to address complex questions, presents a dilemma to unlearning: existing methods either struggle to completely eliminate undesired knowledge from the CoT traces or degrade the reasoning performances due to the interference with the reasoning process. To this end, we introduce Counterfactual Unlearning through iterative Preference Optimization (CiPO), a novel framework that redefines unlearning as the targeted intervention of the CoT reasoning in LRMs. More specifically, given a desired unlearning target answer, CiPO instructs LRMs to generate a logically valid counterfactual reasoning trace for preference tuning. As the LRM adjusts to the counterfactual trace, CiPO iteratively updates the preference learning data to increase the discrepancy from the original model. This iterative loop ensures both desirable unlearning and smooth optimization, effectively mitigating the dilemma. Experiments on challenging benchmarks demonstrate that CiPO excels at unlearning, completely removing knowledge from both the intermediate CoT steps and the final answer, while preserving the reasoning abilities of LRMs.
>
---
#### [new 012] CIG: Measuring Conversational Information Gain in Deliberative Dialogues with Semantic Memory Dynamics
- **分类: cs.CL**

- **简介: 该论文属于对话质量评估任务，旨在衡量公共讨论中的信息进展。通过构建动态语义记忆，量化话语的创新性、相关性和影响范围，提升对话分析的准确性。**

- **链接: [https://arxiv.org/pdf/2604.15647](https://arxiv.org/pdf/2604.15647)**

> **作者:** Ming-Bin Chen; Jey Han Lau; Lea Frermann
>
> **备注:** 24 pages, 5 figures
>
> **摘要:** Measuring the quality of public deliberation requires evaluating not only civility or argument structure, but also the informational progress of a conversation. We introduce a framework for Conversational Information Gain (CIG) that evaluates each utterance in terms of how it advances collective understanding of the target topic. To operationalize CIG, we model an evolving semantic memory of the discussion: the system extracts atomic claims from utterances and incrementally consolidates them into a structured memory state. Using this memory, we score each utterance along three interpretable dimensions: Novelty, Relevance, and Implication Scope. We annotate 80 segments from two moderated deliberative settings (TV debates and community discussions) with these dimensions and show that memory-derived dynamics (e.g., the number of claim updates) correlate more strongly with human-perceived CIG than traditional heuristics such as utterance length or TF--IDF. We develop effective LLM-based CIG predictors paving the way for information-focused conversation quality analysis in dialogues and deliberative success.
>
---
#### [new 013] On the Rejection Criterion for Proxy-based Test-time Alignment
- **分类: cs.CL**

- **简介: 该论文属于测试时对齐任务，解决大模型生成质量不足的问题。通过改进拒绝准则，提升对齐效果。**

- **链接: [https://arxiv.org/pdf/2604.16146](https://arxiv.org/pdf/2604.16146)**

> **作者:** Ayoub Hammal; Pierre Zweigenbaum; Caio Corro
>
> **备注:** ACL 2026 Main
>
> **摘要:** Recent works proposed test-time alignment methods that rely on a small aligned model as a proxy that guides the generation of a larger base (unaligned) model. The implicit reward approach skews the large model distribution, whereas the nudging approach defers the generation of the next token to the small aligned model when the large base one is unconfident about its outcome. In this work, we first show that both approaches can be reduced to sampling from similar graphical models, where they differ only in the definition of a rejection criterion (or distribution). Moreover, we argue that the confidence criterion is ill-motivated due to linguistic phenomena like ambiguous phrasing. We propose a novel rejection criterion based on a conservative confidence bet. Experimentally, our novel approach outperforms previous work on several datasets.
>
---
#### [new 014] C-Mining: Unsupervised Discovery of Seeds for Cultural Data Synthesis via Geometric Misalignment
- **分类: cs.CL**

- **简介: 该论文属于文化数据合成任务，解决种子选择缺乏量化标准的问题。提出C-Mining框架，通过几何错位发现文化种子，提升文化理解能力。**

- **链接: [https://arxiv.org/pdf/2604.15675](https://arxiv.org/pdf/2604.15675)**

> **作者:** Pufan Zeng; Yilun Liu; Mingchen Dai; Mengyao Piao; Chunguang Zhao; Lingqi Miao; Shimin Tao; Weibin Meng; Minggui He; Chenxin Liu; Zhenzhen Qin; Li Zhang; Hongxia Ma; Boxing Chen; Daimeng Wei
>
> **摘要:** Achieving cultural alignment in Large Language Models (LLMs) increasingly depends on synthetic data generation. For such synthesis, the most vital initial step is seed curation; however, current methods lack quantifiable standards for selecting these seeds. Existing approaches rely on unscalable manual curation or bias-prone LLM extraction, treating cultural specificity as an abstract concept rather than a measurable signal. In this paper, we address this "quantification gap" by proposing C-Mining, an unsupervised framework that transforms the discovery of cultural seeds from a subjective selection process into a computable data mining formulation. Our approach exploits a novel geometric insight, leveraging the cross-lingual misalignment of cultural concepts within pre-trained embedding spaces as a quantifiable discovery signal. By systematically identifying these regions characterized by pronounced linguistic exclusivity and geometric isolation, while actively filtering out noise, C-Mining automatically extracts high-fidelity Culture Points (CPs) from raw multilingual corpora without reliance on human or LLM supervision, reducing preparation costs by more than 150-fold. We further leverage the mined knowledge to steer the synthesis of diverse instruction-tuning datasets. Extensive experiments demonstrate that this seed-centric approach significantly enhances cultural understanding and reasoning capabilities, achieving a +6.03 point improvement on CulturalBench-Hard and surpassing state-of-the-art baselines, providing a scalable, quantifiable solution for high-quality cultural data synthesis.
>
---
#### [new 015] LLM attribution analysis across different fine-tuning strategies and model scales for automated code compliance
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究LLM在代码合规任务中的可解释性，比较不同微调策略和模型规模对模型解释行为的影响，旨在提升模型透明度。**

- **链接: [https://arxiv.org/pdf/2604.15589](https://arxiv.org/pdf/2604.15589)**

> **作者:** Jack Wei Lun Shi; Minghao Dang; Wawan Solihin; Justin K.W. Yeoh
>
> **备注:** 8 pages, 9 figures. Accepted at ICCCBE 2026 (International Conference on Computing in Civil and Building Engineering)
>
> **摘要:** Existing research on large language models (LLMs) for automated code compliance has primarily focused on performance, treating the models as black boxes and overlooking how training decisions affect their interpretive behavior. This paper addresses this gap by employing a perturbation-based attribution analysis to compare the interpretive behaviors of LLMs across different fine-tuning strategies such as full fine-tuning (FFT), low-rank adaptation (LoRA) and quantized LoRA fine-tuning, as well as the impact of model scales which include varying LLM parameter sizes. Our results show that FFT produces attribution patterns that are statistically different and more focused than those from parameter-efficient fine-tuning methods. Furthermore, we found that as model scale increases, LLMs develop specific interpretive strategies such as prioritizing numerical constraints and rule identifiers in the building text, albeit with performance gains in semantic similarity of the generated and reference computer-processable rules plateauing for models larger than 7B. This paper provides crucial insights into the explainability of these models, taking a step toward building more transparent LLMs for critical, regulation-based tasks in the Architecture, Engineering, and Construction industry.
>
---
#### [new 016] PIIBench: A Unified Multi-Source Benchmark Corpus for Personally Identifiable Information Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PIIBench，一个统一的PII检测基准语料库，解决多源数据不兼容问题，整合多个数据集并标准化标注，为PII检测提供更全面的评估挑战。**

- **链接: [https://arxiv.org/pdf/2604.15776](https://arxiv.org/pdf/2604.15776)**

> **作者:** Pritesh Jha
>
> **摘要:** We present PIIBench, a unified benchmark corpus for Personally Identifiable Information (PII) detection in natural language text. Existing resources for PII detection are fragmented across domain-specific corpora with mutually incompatible annotation schemes, preventing systematic comparison of detection systems. We consolidate ten publicly available datasets spanning synthetic PII corpora, multilingual Named Entity Recognition (NER) benchmarks, and financial domain annotated text, yielding a corpus of 2,369,883 annotated sequences and 3.35 million entity mentions across 48 canonical PII entity types. We develop a principled normalization pipeline that maps 80+ source-specific label variants to a standardized BIO tagging scheme, applies frequency-based suppression of near absent entity types, and produces stratified 80/10/10 train/validation/test splits preserving source distribution. To establish baseline difficulty, we evaluate eight published systems spanning rule-based engines (Microsoft Presidio), general purpose NER models (spaCy, BERT-base NER, XLM-RoBERTa NER, SpanMarker mBERT, SpanMarker BERT), a PII-specific model (Piiranha DeBERTa), and a financial NER specialist (XtremeDistil FiNER). All systems achieve span-level F1 below 0.14, with the best system (Presidio, F1=0.1385) still producing zero recall on most entity types. These results directly quantify the domain-silo problem and demonstrate that PIIBench presents a substantially harder and more comprehensive evaluation challenge than any existing single source PII dataset. The dataset construction pipeline and benchmark evaluation code are publicly available at this https URL.
>
---
#### [new 017] Beyond Surface Statistics: Robust Conformal Prediction for LLMs via Internal Representations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型可靠性任务，解决输出不确定性不稳定问题。通过内部表示构建非一致性分数，提升 conformal 预测的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.16217](https://arxiv.org/pdf/2604.16217)**

> **作者:** Yanli Wang; Peng Kuang; Xiaoyu Han; Kaidi Xu; Haohan Wang
>
> **摘要:** Large language models are increasingly deployed in settings where reliability matters, yet output-level uncertainty signals such as token probabilities, entropy, and self-consistency can become brittle under calibration--deployment mismatch. Conformal prediction provides finite-sample validity under exchangeability, but its practical usefulness depends on the quality of the nonconformity score. We propose a conformal framework for LLM question answering that uses internal representations rather than output-facing statistics: specifically, we introduce Layer-Wise Information (LI) scores, which measure how conditioning on the input reshapes predictive entropy across model depth, and use them as nonconformity scores within a standard split conformal pipeline. Across closed-ended and open-domain QA benchmarks, with the clearest gains under cross-domain shift, our method achieves a better validity--efficiency trade-off than strong text-level baselines while maintaining competitive in-domain reliability at the same nominal risk level. These results suggest that internal representations can provide more informative conformal scores when surface-level uncertainty is unstable under distribution shift.
>
---
#### [new 018] BAGEL: Benchmarking Animal Knowledge Expertise in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出BAGEL基准，用于评估语言模型在动物知识方面的表现。任务是测试模型在封闭环境下处理专业动物知识的能力，解决其在生物多样性应用中的可靠性问题。**

- **链接: [https://arxiv.org/pdf/2604.16241](https://arxiv.org/pdf/2604.16241)**

> **作者:** Jiacheng Shen; Masato Hagiwara; Milad Alizadeh; Ellen Gilsenan-McMahon; Marius Miron; David Robinson; Emmanuel Chemla; Sara Keen; Gagan Narula; Mathieu Laurière; Matthieu Geist; Olivier Pietquin
>
> **备注:** 28 pages, 3 figures
>
> **摘要:** Large language models have shown strong performance on broad-domain knowledge and reasoning benchmarks, but it remains unclear how well language models handle specialized animal-related knowledge under a unified closed-book evaluation protocol. We introduce BAGEL, a benchmark for evaluating animal knowledge expertise in language models. BAGEL is constructed from diverse scientific and reference sources, including bioRxiv, Global Biotic Interactions, Xeno-canto, and Wikipedia, using a combination of curated examples and automatically generated closed-book question-answer pairs. The benchmark covers multiple aspects of animal knowledge, including taxonomy, morphology, habitat, behavior, vocalization, geographic distribution, and species interactions. By focusing on closed-book evaluation, BAGEL measures animal-related knowledge of models without external retrieval at inference time. BAGEL further supports fine-grained analysis across source domains, taxonomic groups, and knowledge categories, enabling a more precise characterization of model strengths and systematic failure modes. Our benchmark provides a new testbed for studying domain-specific knowledge generalization in language models and for improving their reliability in biodiversity-related applications.
>
---
#### [new 019] From Benchmarking to Reasoning: A Dual-Aspect, Large-Scale Evaluation of LLMs on Vietnamese Legal Text
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律文本处理任务，旨在评估大语言模型在越南法律文本中的表现。通过基准测试和错误分析，解决模型在准确性与可读性间的权衡问题。**

- **链接: [https://arxiv.org/pdf/2604.16270](https://arxiv.org/pdf/2604.16270)**

> **作者:** Van-Truong Le
>
> **备注:** 7 pages, 2 figures. Accepted at the FISU Joint Conference on Artificial Intelligence (FJCAI 2026), Vietnam
>
> **摘要:** The complexity of Vietnam's legal texts presents a significant barrier to public access to justice. While Large Language Models offer a promising solution for legal text simplification, evaluating their true capabilities requires a multifaceted approach that goes beyond surface-level metrics. This paper introduces a comprehensive dual-aspect evaluation framework to address this need. First, we establish a performance benchmark for four state-of-the-art large language models (GPT-4o, Claude 3 Opus, Gemini 1.5 Pro, and Grok-1) across three key dimensions: Accuracy, Readability, and Consistency. Second, to understand the "why" behind these performance scores, we conduct a large-scale error analysis on a curated dataset of 60 complex Vietnamese legal articles, using a novel, expert-validated error typology. Our results reveal a crucial trade-off: models like Grok-1 excel in Readability and Consistency but compromise on fine-grained legal Accuracy, while models like Claude 3 Opus achieve high Accuracy scores that mask a significant number of subtle but critical reasoning errors. The error analysis pinpoints \textit{Incorrect Example} and \textit{Misinterpretation} as the most prevalent failures, confirming that the primary challenge for current LLMs is not summarization but controlled, accurate legal reasoning. By integrating a quantitative benchmark with a qualitative deep dive, our work provides a holistic and actionable assessment of LLMs for legal applications.
>
---
#### [new 020] Towards Intrinsic Interpretability of Large Language Models:A Survey of Design Principles and Architectures
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于可解释AI任务，旨在解决LLM透明性不足的问题。通过系统综述，提出五种内在可解释性设计范式，推动模型架构的透明化。**

- **链接: [https://arxiv.org/pdf/2604.16042](https://arxiv.org/pdf/2604.16042)**

> **作者:** Yutong Gao; Qinglin Meng; Yuan Zhou; Liangming Pan
>
> **备注:** Accepted to the Main Conference of ACL 2026. 14 pages, 4 figures, 1 table
>
> **摘要:** While Large Language Models (LLMs) have achieved strong performance across many NLP tasks, their opaque internal mechanisms hinder trustworthiness and safe deployment. Existing surveys in explainable AI largely focus on post-hoc explanation methods that interpret trained models through external approximations. In contrast, intrinsic interpretability, which builds transparency directly into model architectures and computations, has recently emerged as a promising alternative. This paper presents a systematic review of the recent advances in intrinsic interpretability for LLMs, categorizing existing approaches into five design paradigms: functional transparency, concept alignment, representational decomposability, explicit modularization, and latent sparsity induction. We further discuss open challenges and outline future research directions in this emerging field. The paper list is available at: this https URL.
>
---
#### [new 021] "Excuse me, may I say something..." CoLabScience, A Proactive AI Assistant for Biomedical Discovery and LLM-Expert Collaborations
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于人工智能与生物医学协作任务，旨在解决LLM反应性不足的问题。提出CoLabScience系统及PULI框架，实现主动干预，提升合作效率。**

- **链接: [https://arxiv.org/pdf/2604.15588](https://arxiv.org/pdf/2604.15588)**

> **作者:** Yang Wu; Jinhong Yu; Jingwei Xiong; Zhimin Tao; Xiaozhong Liu
>
> **备注:** ACL 2026 Main Conference
>
> **摘要:** The integration of Large Language Models (LLMs) into scientific workflows presents exciting opportunities to accelerate biomedical discovery. However, the reactive nature of LLMs, which respond only when prompted, limits their effectiveness in collaborative settings that demand foresight and autonomous engagement. In this study, we introduce CoLabScience, a proactive LLM assistant designed to enhance biomedical collaboration between AI systems and human experts through timely, context-aware interventions. At the core of our method is PULI (Positive-Unlabeled Learning-to-Intervene), a novel framework trained with a reinforcement learning objective to determine when and how to intervene in streaming scientific discussions, by leveraging the team's project proposal and long- and short-term conversational memory. To support this work, we introduce BSDD (Biomedical Streaming Dialogue Dataset), a new benchmark of simulated research discussion dialogues with intervention points derived from PubMed articles. Experimental results show that PULI significantly outperforms existing baselines in both intervention precision and collaborative task utility, highlighting the potential of proactive LLMs as intelligent scientific assistants.
>
---
#### [new 022] Consistency Analysis of Sentiment Predictions using Syntactic & Semantic Context Assessment Summarization (SSAS)
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感分析任务，旨在解决LLM在情感预测中的不一致性问题。通过构建SSAS框架，提升数据质量与预测稳定性。**

- **链接: [https://arxiv.org/pdf/2604.15547](https://arxiv.org/pdf/2604.15547)**

> **作者:** Sharookh Daruwalla; Nitin Mayande; Shreeya Verma Kathuria; Nitin Joglekar; Charles Weber
>
> **备注:** 27 pages, 2 figures. arXiv admin note: text overlap with arXiv:2604.12049
>
> **摘要:** The fundamental challenge of using Large Language Models (LLMs) for reliable, enterprise-grade analytics, such as sentiment prediction, is the conflict between the LLMs' inherent stochasticity (generative, non-deterministic nature) and the analytical requirement for consistency. The LLM inconsistency, coupled with the noisy nature of chaotic modern datasets, renders sentiment predictions too volatile for strategic business decisions. To resolve this, we present a Syntactic & Semantic Context Assessment Summarization (SSAS) framework for establishing context. Context established by SSAS functions as a sophisticated data pre-processing framework that enforces a bounded attention mechanism on LLMs. It achieves this by applying a hierarchical classification structure (Themes, Stories, Clusters) and an iterative Summary-of-Summaries (SoS) based context computation architecture. This endows the raw text with high-signal, sentiment-dense prompts, that effectively mitigate both irrelevant data and analytical variance. We empirically evaluated the efficacy of SSAS, using Gemini 2.0 Flash Lite, against a direct-LLM approach across three industry-standard datasets - Amazon Product Reviews, Google Business Reviews, Goodreads Book Reviews - and multiple robustness scenarios. Our results show that our SSAS framework is capable of significantly improving data quality, up to 30%, through a combination of noise removal and improvement in the estimation of sentiment prediction. Ultimately, consistency in our context-estimation capabilities provides a stable and reliable evidence base for decision-making.
>
---
#### [new 023] LLMs Corrupt Your Documents When You Delegate
- **分类: cs.CL; cs.HC**

- **简介: 该论文研究AI在文档协作中的可靠性问题，发现LLMs在长期任务中会破坏文档内容，提出DELEGATE-52测试集验证模型缺陷。**

- **链接: [https://arxiv.org/pdf/2604.15597](https://arxiv.org/pdf/2604.15597)**

> **作者:** Philippe Laban; Tobias Schnabel; Jennifer Neville
>
> **摘要:** Large Language Models (LLMs) are poised to disrupt knowledge work, with the emergence of delegated work as a new interaction paradigm (e.g., vibe coding). Delegation requires trust - the expectation that the LLM will faithfully execute the task without introducing errors into documents. We introduce DELEGATE-52 to study the readiness of AI systems in delegated workflows. DELEGATE-52 simulates long delegated workflows that require in-depth document editing across 52 professional domains, such as coding, crystallography, and music notation. Our large-scale experiment with 19 LLMs reveals that current models degrade documents during delegation: even frontier models (Gemini 3.1 Pro, Claude 4.6 Opus, GPT 5.4) corrupt an average of 25% of document content by the end of long workflows, with other models failing more severely. Additional experiments reveal that agentic tool use does not improve performance on DELEGATE-52, and that degradation severity is exacerbated by document size, length of interaction, or presence of distractor files. Our analysis shows that current LLMs are unreliable delegates: they introduce sparse but severe errors that silently corrupt documents, compounding over long interaction.
>
---
#### [new 024] CoEvolve: Training LLM Agents via Agent-Data Mutual Evolution
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM代理在静态数据分布下适应性差的问题。通过构建代理与数据的协同进化框架，提升代理性能。**

- **链接: [https://arxiv.org/pdf/2604.15840](https://arxiv.org/pdf/2604.15840)**

> **作者:** Shidong Yang; Ziyu Ma; Tongwen Huang; Yiming Hu; Yong Wang; Xiangxiang Chu
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** Reinforcement learning for LLM agents is typically conducted on a static data distribution, which fails to adapt to the agent's evolving behavior and leads to poor coverage of complex environment interactions. To address these challenges, we propose CoEvolve, an agent-data mutual evolution framework that enables LLM agents to improve through closed-loop, interaction-driven training. Specifically, CoEvolve extracts feedback signals such as forgetting and uncertainty from rollout trajectories to identify failure-prone interaction patterns, and utilizes them to guide LLM-based task synthesis. The synthesized tasks are validated through environment interaction and utilized to update the data distribution, enabling joint adaptation of the agent and its data. Extensive experiments on AppWorld and BFCL across Qwen2.5-7B, Qwen3-4B, and Qwen3-30B-A3B demonstrate consistent and significant improvements over strong base models, yielding absolute gains of 19.43%, 15.58%, and 18.14%, respectively.
>
---
#### [new 025] SwanNLP at SemEval-2026 Task 5: An LLM-based Framework for Plausibility Scoring in Narrative Word Sense Disambiguation
- **分类: cs.CL**

- **简介: 该论文属于叙事词义消歧任务，旨在解决词语在故事中合理性评分问题。提出基于大语言模型的框架，通过结构化推理和动态提示提升词义判断准确性。**

- **链接: [https://arxiv.org/pdf/2604.16262](https://arxiv.org/pdf/2604.16262)**

> **作者:** Deshan Sumanathilaka; Nicholas Micallef; Julian Hough; Saman Jayasinghe
>
> **备注:** 6 pages, 5 Tables, 1 figure, Accepted to SemEval 2026
>
> **摘要:** Recent advances in language models have substantially improved Natural Language Understanding (NLU). Although widely used benchmarks suggest that Large Language Models (LLMs) can effectively disambiguate, their practical applicability in real-world narrative contexts remains underexplored. SemEval-2026 Task 5 addresses this gap by introducing a task that predicts the human-perceived plausibility of a word sense within a short story. In this work, we propose an LLM-based framework for plausibility scoring of homonymous word senses in narrative texts using a structured reasoning mechanism. We examine the impact of fine-tuning low-parameter LLMs with diverse reasoning strategies, alongside dynamic few-shot prompting for large-parameter models, on accurate sense identification and plausibility estimation. Our results show that commercial large-parameter LLMs with dynamic few-shot prompting closely replicate human-like plausibility judgments. Furthermore, model ensembling slightly improves performance, better simulating the agreement patterns of five human annotators compared to single-model predictions
>
---
#### [new 026] TTL: Test-time Textual Learning for OOD Detection with Pretrained Vision-Language Models
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于OOD检测任务，旨在解决预训练视觉-语言模型在测试时无法适应开放语义空间的问题。通过动态学习未标记测试数据的文本语义，提升检测性能。**

- **链接: [https://arxiv.org/pdf/2604.15756](https://arxiv.org/pdf/2604.15756)**

> **作者:** Jinlun Ye; Jiang Liao; Runhe Lai; Xinhua Lu; Jiaxin Zhuang; Zhiyong Gan; Ruixuan Wang
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Vision-language models (VLMs) such as CLIP exhibit strong Out-of-distribution (OOD) detection capabilities by aligning visual and textual representations. Recent CLIP-based test-time adaptation methods further improve detection performance by incorporating external OOD labels. However, such labels are finite and fixed, while the real OOD semantic space is inherently open-ended. Consequently, fixed labels fail to represent the diverse and evolving OOD semantics encountered in test streams. To address this limitation, we introduce Test-time Textual Learning (TTL), a framework that dynamically learns OOD textual semantics from unlabeled test streams, without relying on external OOD labels. TTL updates learnable prompts using pseudo-labeled test samples to capture emerging OOD knowledge. To suppress noise introduced by pseudo-labels, we introduce an OOD knowledge purification strategy that selects reliable OOD samples for adaptation while suppressing noise. In addition, TTL maintains an OOD Textual Knowledge Bank that stores high-quality textual features, providing stable score calibration across batches. Extensive experiments on two standard benchmarks with nine OOD datasets demonstrate that TTL consistently achieves state-of-the-art performance, highlighting the value of textual adaptation for robust test-time OOD detection. Our code is available at this https URL.
>
---
#### [new 027] Qwen3.5-Omni Technical Report
- **分类: cs.CL; eess.AS**

- **简介: 该论文介绍Qwen3.5-Omni，解决多模态理解和交互问题，通过大规模数据训练，提升音频视频处理及语音合成能力。**

- **链接: [https://arxiv.org/pdf/2604.15804](https://arxiv.org/pdf/2604.15804)**

> **作者:** Qwen Team
>
> **摘要:** In this work, we present Qwen3.5-Omni, the latest advancement in the Qwen-Omni model family. Representing a significant evolution over its predecessor, Qwen3.5-Omni scales to hundreds of billions of parameters and supports a 256k context length. By leveraging a massive dataset comprising heterogeneous text-vision pairs and over 100 million hours of audio-visual content, the model demonstrates robust omni-modality capabilities. Qwen3.5-Omni-plus achieves SOTA results across 215 audio and audio-visual understanding, reasoning, and interaction subtasks and benchmarks, surpassing Gemini-3.1 Pro in key audio tasks and matching it in comprehensive audio-visual understanding. Architecturally, Qwen3.5-Omni employs a Hybrid Attention Mixture-of-Experts (MoE) framework for both Thinker and Talker, enabling efficient long-sequence inference. The model facilitates sophisticated interaction, supporting over 10 hours of audio understanding and 400 seconds of 720P video (at 1 FPS). To address the inherent instability and unnaturalness in streaming speech synthesis, often caused by encoding efficiency discrepancies between text and speech tokenizers, we introduce ARIA. ARIA dynamically aligns text and speech units, significantly enhancing the stability and prosody of conversational speech with minimal latency impact. Furthermore, Qwen3.5-Omni expands linguistic boundaries, supporting multilingual understanding and speech generation across 10 languages with human-like emotional nuance. Finally, Qwen3.5-Omni exhibits superior audio-visual grounding capabilities, generating script-level structured captions with precise temporal synchronization and automated scene segmentation. Remarkably, we observed the emergence of a new capability in omnimodal models: directly performing coding based on audio-visual instructions, which we call Audio-Visual Vibe Coding.
>
---
#### [new 028] Can LLMs Understand the Impact of Trauma? Costs and Benefits of LLMs Coding the Interviews of Firearm Violence Survivors
- **分类: cs.CL; cs.AI**

- **简介: 论文探讨了使用大语言模型（LLM）对枪支暴力幸存者访谈进行编码的可行性，旨在解决传统人工编码耗时的问题。研究发现LLM效果有限且存在伦理风险。**

- **链接: [https://arxiv.org/pdf/2604.16132](https://arxiv.org/pdf/2604.16132)**

> **作者:** Jessica H. Zhu; Shayla Stringfield; Vahe Zaprosyan; Michael Wagner; Michel Cukier; Joseph B. Richardson Jr
>
> **备注:** Accepted to Findings of the Association for Computational Linguistics (2026)
>
> **摘要:** Firearm violence is a pressing public health issue, yet research into survivors' lived experiences remains underfunded and difficult to scale. Qualitative research, including in-depth interviews, is a valuable tool for understanding the personal and societal consequences of community firearm violence and designing effective interventions. However, manually analyzing these narratives through thematic analysis and inductive coding is time-consuming and labor-intensive. Recent advancements in large language models (LLMs) have opened the door to automating this process, though concerns remain about whether these models can accurately and ethically capture the experiences of vulnerable populations. In this study, we assess the use of open-source LLMs to inductively code interviews with 21 Black men who have survived community firearm violence. Our results demonstrate that while some configurations of LLMs can identify important codes, overall relevance remains low and is highly sensitive to data processing. Furthermore, LLM guardrails lead to substantial narrative erasure. These findings highlight both the potential and limitations of LLM-assisted qualitative coding and underscore the ethical challenges of applying AI in research involving marginalized communities.
>
---
#### [new 029] DiZiNER: Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot Named Entity Recognition
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于命名实体识别任务，解决零样本NER的系统性错误问题。通过模拟标注过程，利用模型间分歧优化指令，提升性能。**

- **链接: [https://arxiv.org/pdf/2604.15866](https://arxiv.org/pdf/2604.15866)**

> **作者:** Siun Kim; Hyung-Jin Yoon
>
> **备注:** 9 pages, 3 figures; Accepted to the ACL 2026 Main Conference
>
> **摘要:** Large language models (LLMs) have advanced information extraction (IE) by enabling zero-shot and few-shot named entity recognition (NER), yet their generative outputs still show persistent and systematic errors. Despite progress through instruction fine-tuning, zero-shot NER still lags far behind supervised systems. These recurring errors mirror inconsistencies observed in early-stage human annotation processes that resolve disagreements through pilot annotation. Motivated by this analogy, we introduce DiZiNER (Disagreement-guided Instruction Refinement via Pilot Annotation Simulation for Zero-shot Named Entity Recognition), a framework that simulates the pilot annotation process, employing LLMs to act as both annotators and supervisors. Multiple heterogeneous LLMs annotate shared texts, and a supervisor model analyzes inter-model disagreements to refine task instructions. Across 18 benchmarks, DiZiNER achieves zero-shot SOTA results on 14 datasets, improving prior bests by +8.0 F1 and reducing the zero-shot to supervised gap by over +11 points. It also consistently outperforms its supervisor, GPT-5 mini, indicating that improvements stem from disagreement-guided instruction refinement rather than model capacity. Pairwise agreement between models shows a strong correlation with NER performance, further supporting this finding.
>
---
#### [new 030] The Metacognitive Monitoring Battery: A Cross-Domain Benchmark for LLM Self-Monitoring
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于LLM自监控评估任务，旨在解决模型元认知监测能力的衡量问题。通过设计跨领域测试电池，评估模型在不同认知任务中的自我监控表现。**

- **链接: [https://arxiv.org/pdf/2604.15702](https://arxiv.org/pdf/2604.15702)**

> **作者:** Jon-Paul Cacioli
>
> **备注:** 11 pages, 6 figures, 3 tables. Submitted to NeurIPS 2026 Evaluations and Datasets Track. Code, data, and Croissant metadata: this https URL
>
> **摘要:** We introduce a cross-domain behavioural assay of monitoring-control coupling in LLMs, grounded in the Nelson and Narens (1990) metacognitive framework and applying human psychometric methodology to LLM evaluation. The battery comprises 524 items across six cognitive domains (learning, metacognitive calibration, social cognition, attention, executive function, prospective regulation), each grounded in an established experimental paradigm. Tasks T1-T5 were pre-registered on OSF prior to data collection; T6 was added as an exploratory extension. After every forced-choice response, dual probes adapted from Koriat and Goldsmith (1996) ask the model to KEEP or WITHDRAW its answer and to BET or decline. The critical metric is the withdraw delta: the difference in withdrawal rate between incorrect and correct items. Applied to 20 frontier LLMs (10,480 evaluations), the battery discriminates three profiles consistent with the Nelson-Narens architecture: blanket confidence, blanket withdrawal, and selective sensitivity. Accuracy rank and metacognitive sensitivity rank are largely inverted. Retrospective monitoring and prospective regulation appear dissociable (r = .17, 95% CI wide given n=20; exemplar-based evidence is the primary support). Scaling on metacognitive calibration is architecture-dependent: monotonically decreasing (Qwen), monotonically increasing (GPT-5.4), or flat (Gemma). Behavioural findings converge structurally with an independent Type-2 SDT approach, providing preliminary cross-method construct validity. All items, data, and code: this https URL.
>
---
#### [new 031] Skill-RAG: Failure-State-Aware Retrieval Augmentation via Hidden-State Probing and Skill Routing
- **分类: cs.CL**

- **简介: 该论文提出Skill-RAG，解决RAG系统中查询与证据对齐失败的问题，通过检测失败状态并选择相应技能进行修正，提升复杂任务的准确性。**

- **链接: [https://arxiv.org/pdf/2604.15771](https://arxiv.org/pdf/2604.15771)**

> **作者:** Kai Wei; Raymond Li; Xi Zhu; Zhaoqian Xue; Jiaojiao Han; Jingcheng Niu; Fan Yang
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a foundational paradigm for grounding large language models in external knowledge. While adaptive retrieval mechanisms have improved retrieval efficiency, existing approaches treat post-retrieval failure as a signal to retry rather than to diagnose -- leaving the structural causes of query-evidence misalignment unaddressed. We observe that a significant portion of persistent retrieval failures stem not from the absence of relevant evidence but from an alignment gap between the query and the evidence space. We propose Skill-RAG, a failure-aware RAG framework that couples a lightweight hidden-state prober with a prompt-based skill router. The prober gates retrieval at two pipeline stages; upon detecting a failure state, the skill router diagnoses the underlying cause and selects among four retrieval skills -- query rewriting, question decomposition, evidence focusing, and an exit skill for truly irreducible cases -- to correct misalignment before the next generation attempt. Experiments across multiple open-domain QA and complex reasoning benchmarks show that Skill-RAG substantially improves accuracy on hard cases persisting after multi-turn retrieval, with particularly strong gains on out-of-distribution datasets. Representation-space analyses further reveal that the proposed skills occupy structured, separable regions of the failure state space, supporting the view that query-evidence misalignment is a typed rather than monolithic phenomenon.
>
---
#### [new 032] A Systematic Study of Training-Free Methods for Trustworthy Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，旨在评估训练-free方法在大语言模型中的可信度影响。研究系统分析了这些方法对模型实用性、鲁棒性及计算开销的影响，并提出改进建议。**

- **链接: [https://arxiv.org/pdf/2604.15789](https://arxiv.org/pdf/2604.15789)**

> **作者:** Wai Man Si; Mingjie Li; Michael Backes; Yang Zhang
>
> **摘要:** As Large Language Models (LLMs) receive increasing attention and are being deployed across various domains, their potential risks, including generating harmful or biased content, producing unsupported claims, and exhibiting vulnerabilities to adversarial attacks, have drawn significant attention. To enable quick and low-cost adaptation, training-free methods have recently emerged as cost-effective alternatives to post-training alignment techniques. Despite their promising results, these methods are evaluated inconsistently across the literature, cover limited dimensions of trustworthiness, and can introduce undesirable side effects, such as utility degradation and increased brittleness. To fully assess the impacts of these training-free methods, we take a step back and systematically re-evaluate the effectiveness of existing training-free methods against various trustworthy settings and their influence on utility, robustness, and computational overhead. We also categorize these methods into three levels (input, internal, and output) based on where they intervene in the model's information flow during inference. Using this taxonomy, we conduct a comprehensive analysis of various representative and effective methods from each level across different LLM families and sizes. Our analysis highlights several trade-offs and unresolved challenges in current approaches. We summarize key findings and limitations in the existing literature, and propose practical recommendations for balancing trustworthiness, utility, and robustness in LLMs without the need for additional training.
>
---
#### [new 033] Optimizing Korean-Centric LLMs via Token Pruning
- **分类: cs.CL**

- **简介: 该论文研究多语言大模型的优化，针对韩语相关任务，通过删除无关语言token提升模型性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.16235](https://arxiv.org/pdf/2604.16235)**

> **作者:** Hoyeol Kim; Hyeonwoo Kim
>
> **备注:** 5 pages
>
> **摘要:** This paper presents a systematic benchmark of state-of-the-art multilingual large language models (LLMs) adapted via token pruning - a compression technique that eliminates tokens and embedding parameters corresponding to languages irrelevant to the target application. Focusing on Korean-centric natural language processing (NLP) tasks, we evaluate architectures including Qwen3, Gemma-3, Llama-3, and Aya across three vocabulary configurations: Original, English-Korean (EnKo), and English-Korean-Chinese (EnKoZh). Performance is assessed using established benchmarks for general aptitude, cultural literacy, instruction following, and machine translation. Our findings indicate that token pruning significantly improves generation stability by eliminating language confusion, and in the case of machine translation, frequently enhances performance on Korean-specific tasks. While instruction-following capabilities display architecture-dependent variance linked to latent cross-lingual representations, the significant reduction in vocabulary size validates token pruning as a highly effective optimization strategy for memory-constrained, domain-specific deployments, despite modest gains in inference latency.
>
---
#### [new 034] Target-Oriented Pretraining Data Selection via Neuron-Activated Graph
- **分类: cs.CL**

- **简介: 该论文属于目标导向的预训练任务，旨在提升语言模型在特定目标上的性能。通过引入NAG-based Ranking方法，选择高影响力神经元构建图结构，优化数据选择，提高模型效果。**

- **链接: [https://arxiv.org/pdf/2604.15706](https://arxiv.org/pdf/2604.15706)**

> **作者:** Zijun Wang; Haoqin Tu; Weidong Zhou; Yiyang Zhou; Xiaohuan Zhou; Bingni Zhang; Weiguo Feng; Taifeng Wang; Cihang Xie; Fengze Liu
>
> **摘要:** Everyday tasks come with a target, and pretraining models around this target is what turns them into experts. In this paper, we study target-oriented language model (LM) pretraining by introducing Neuron-Activated Graph Ranking (NAG-based Ranking), a training-free and interpretable framework for target pretraining data selection. Rather than using black-box representations, our approach directly characterizes each target input by a sparse set of high-impact neurons in any off-the-shelf LLMs. Concretely, we quantify neuron impact and select the most influential neurons across layers into a compact Neuron-Activated Graph (NAG), and rank candidate data by NAG similarity to target examples. We conduct experiments across six benchmarks, where our NAG-based Ranking improves target-oriented pretraining by 4.9% on average over random sampling, and also outperforms state-of-the-art baselines by 5.3% accuracy on HellaSwag. It also remains effective under a more applicable multi-target setting, where our best setup surpasses two baselines by 1.1% and 4.1%, respectively. Furthermore, we provide a comprehensive analysis on why and how our NAG works, e.g., deactivating NAG-selected neurons (only 0.12% of all) causes a 23.5% performance collapse, and restricting NAG to the final layer incurs a 4.1% average drop, indicating that NAG captures a sparse "functional backbone" for learning target features. We release the code at this https URL.
>
---
#### [new 035] Applied Explainability for Large Language Models: A Comparative Study
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究大语言模型的可解释性问题。通过比较三种解释方法，评估其在情感分类中的表现，探讨不同方法的优缺点。**

- **链接: [https://arxiv.org/pdf/2604.15371](https://arxiv.org/pdf/2604.15371)**

> **作者:** Venkata Abhinandan Kancharla
>
> **备注:** 14 pages, 3 figures, comparative study of explainability methods for transformer-based NLP models; also available on Zenodo
>
> **摘要:** Large language models (LLMs) achieve strong performance across many natural language processing tasks, yet their decision processes remain difficult to interpret. This lack of transparency creates challenges for trust, debugging, and deployment in real-world systems. This paper presents an applied comparative study of three explainability techniques: Integrated Gradients, Attention Rollout, and SHAP, on a fine-tuned DistilBERT model for SST-2 sentiment classification. Rather than proposing new methods, the focus is on evaluating the practical behavior of existing approaches under a consistent and reproducible setup. The results show that gradient-based attribution provides more stable and intuitive explanations, while attention-based methods are computationally efficient but less aligned with prediction-relevant features. Model-agnostic approaches offer flexibility but introduce higher computational cost and variability. This work highlights key trade-offs between explainability methods and emphasizes their role as diagnostic tools rather than definitive explanations. The findings provide practical insights for researchers and engineers working with transformer-based NLP systems. This is a preprint and has not undergone peer review.
>
---
#### [new 036] GroupDPO: Memory efficient Group-wise Direct Preference Optimization
- **分类: cs.CL**

- **简介: 该论文属于大语言模型对齐任务，解决单对训练效率低的问题，提出GroupDPO算法，通过多响应对比提升性能并降低内存消耗。**

- **链接: [https://arxiv.org/pdf/2604.15602](https://arxiv.org/pdf/2604.15602)**

> **作者:** Jixuan Leng; Si Si; Hsiang-Fu Yu; Vinod Raman; Inderjit S. Dhillon
>
> **摘要:** Preference optimization is widely used to align Large Language Models (LLMs) with preference feedback. However, most existing methods train on a single positive-negative pair per prompt, discarding additional supervision available in preference datasets that typically contain multiple candidate responses. Motivated by this limitation, recent work explores group-wise preference optimization, which jointly contrasts multiple responses for the same prompt, but its empirical behavior and scalability remain underexplored due to the memory overhead of group-coupled objectives. In this work, we introduce a memory-efficient group-wise preference optimization algorithm that preserves gradients while decoupling samples during backpropagation, substantially reducing peak memory usage, which enables scalable training with larger group sizes. Across both offline and online alignment settings, we show that leveraging multiple responses consistently outperforms single-pair training. Furthermore, incorporating a negative log-likelihood (NLL) term on positive responses is critical for both performance gains and training stability.
>
---
#### [new 037] SCHK-HTC: Sibling Contrastive Learning with Hierarchical Knowledge-Aware Prompt Tuning for Hierarchical Text Classification
- **分类: cs.CL**

- **简介: 该论文属于少样本层次文本分类任务，旨在解决因数据稀缺导致的相似兄弟类难以区分的问题。提出SCHK-HTC方法，结合层次知识与对比学习提升分类性能。**

- **链接: [https://arxiv.org/pdf/2604.15998](https://arxiv.org/pdf/2604.15998)**

> **作者:** Ke Xiong; Qian Wu; Wangjie Gan; Yuke Li; Xuhong Zhang
>
> **备注:** 5pages,3 figures,ICASSP 2026
>
> **摘要:** Few-shot Hierarchical Text Classification (few-shot HTC) is a challenging task that involves mapping texts to a predefined tree-structured label hierarchy under data-scarce conditions. While current approaches utilize structural constraints from the label hierarchy to maintain parent-child prediction consistency, they face a critical bottleneck, the difficulty in distinguishing semantically similar sibling classes due to insufficient domain knowledge. We introduce an innovative method named Sibling Contrastive Learning with Hierarchical Knowledge-aware Prompt Tuning for few-shot HTC tasks (SCHK-HTC). Our work enhances the model's perception of subtle differences between sibling classes at deeper levels, rather than just enforcing hierarchical rules. Specifically, we propose a novel framework featuring two core components: a hierarchical knowledge extraction module and a sibling contrastive learning mechanism. This design guides model to encode discriminative features at each hierarchy level, thus improving the separability of confusable classes. Our approach achieves superior performance across three benchmark datasets, surpassing existing state-of-the-art methods in most cases. Our code is available at this https URL.
>
---
#### [new 038] Preference Estimation via Opponent Modeling in Multi-Agent Negotiation
- **分类: cs.CL**

- **简介: 该论文属于多智能体协商任务，旨在解决传统方法无法有效整合自然语言信息的问题。通过将语言信息转化为概率形式，提升偏好估计的准确性。**

- **链接: [https://arxiv.org/pdf/2604.15687](https://arxiv.org/pdf/2604.15687)**

> **作者:** Yuta Konishi; Kento Yamamoto; Eisuke Sonomoto; Rikuho Takeda; Ryo Furukawa; Yusuke Muraki; Takafumi Shimizu; Kazuma Fukumura; Yuya Kanemoto; Takayuki Ito; Shiyao Ding
>
> **备注:** This paper is accepted as a Findings of ACL 2026
>
> **摘要:** Automated negotiation in complex, multi-party and multi-issue settings critically depends on accurate opponent modeling. However, conventional numerical-only approaches fail to capture the qualitative information embedded in natural language interactions, resulting in unstable and incomplete preference estimation. Although Large Language Models (LLMs) enable rich semantic understanding of utterances, it remains challenging to quantitatively incorporate such information into a consistent opponent modeling. To tackle this issue, we propose a novel preference estimation method integrating natural language information into a structured Bayesian opponent modeling framework. Our approach leverages LLMs to extract qualitative cues from utterances and converts them into probabilistic formats for dynamic belief tracking. Experimental results on a multi-party benchmark demonstrate that our framework improves the full agreement rate and preference estimation accuracy by integrating probabilistic reasoning with natural language understanding.
>
---
#### [new 039] Exploring the Capability Boundaries of LLMs in Mastering of Chinese Chouxiang Language
- **分类: cs.CL**

- **简介: 该论文研究LLMs在中文“嘲讽语言”上的能力边界，属于NLP任务。旨在解决其在该子文化语言上的表现不足问题，并构建基准测试。**

- **链接: [https://arxiv.org/pdf/2604.15841](https://arxiv.org/pdf/2604.15841)**

> **作者:** Dianqing Lin; Tian Lan; Jiali Zhu; Jiang Li; Wei Chen; Xu Liu; Aruukhan; Xiangdong Su; Hongxu Hou; Guanglai Gao
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** While large language models (LLMs) have achieved remarkable success in general language tasks, their performance on Chouxiang Language, a representative subcultural language in the Chinese internet context, remains largely unexplored. In this paper, we introduce Mouse, a specialized benchmark designed to evaluate the capabilities of LLMs on NLP tasks involving Chouxiang Language across six tasks. Experimental results show that, current state-of-the-art (SOTA) LLMs exhibit clear limitations on multiple tasks, while performing well on tasks that involve contextual semantic understanding. In addition, we further discuss the reasons behind the generally low performance of SOTA LLMs on Chouxiang Language, examine whether the LLM-as-a-judge approach adopted for translation tasks aligns with human judgments and values, and analyze the key factors that influence Chouxiang translation. Our study aims to promote further research in the NLP community on multicultural integration and the dynamics of evolving internet languages. Our code and data are publicly available.
>
---
#### [new 040] Stochasticity in Tokenisation Improves Robustness
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的鲁棒性。针对输入分词扰动带来的脆弱性问题，通过引入随机分词方法增强模型对对抗攻击和随机扰动的抵抗能力。实验表明，使用随机分词进行预训练和微调可提升鲁棒性且不增加推理成本。**

- **链接: [https://arxiv.org/pdf/2604.16037](https://arxiv.org/pdf/2604.16037)**

> **作者:** Sophie Steger; Rui Li; Sofiane Ennadir; Anya Sims; Arno Solin; Franz Pernkopf; Martin Trapp
>
> **摘要:** The widespread adoption of large language models (LLMs) has increased concerns about their robustness. Vulnerabilities in perturbations of tokenisation of the input indicate that models trained with a deterministic canonical tokenisation can be brittle to adversarial attacks. Recent studies suggest that stochastic tokenisation can deliver internal representations that are less sensitive to perturbations. In this paper, we analyse how stochastic tokenisations affect robustness to adversarial attacks and random perturbations. We systematically study this over a range of learning regimes (pre-training, supervised fine-tuning, and in-context learning), data sets, and model architectures. We show that pre-training and fine-tuning with uniformly sampled stochastic tokenisations improve robustness to random and adversarial perturbations. Evaluating on uniformly sampled non-canonical tokenisations reduces the accuracy of a canonically trained Llama-1b model by 29.8%. We find that training with stochastic tokenisation preserves accuracy without increasing inference cost.
>
---
#### [new 041] Imperfectly Cooperative Human-AI Interactions: Comparing the Impacts of Human and AI Attributes in Simulated and User Studies
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 该论文研究人机协作中的影响因素，比较模拟与真实用户数据，分析性格与AI特性对交互结果的影响，旨在提升人机交互质量。**

- **链接: [https://arxiv.org/pdf/2604.15607](https://arxiv.org/pdf/2604.15607)**

> **作者:** Myke C. Cohen; Mingqian Zheng; Neel Bhandari; Hsien-Te Kao; Xuhui Zhou; Daniel Nguyen; Laura Cassani; Maarten Sap; Svitlana Volkova
>
> **备注:** Will be presented at ACL 2026 and published in the Findings of the Association for Computational Linguistics: ACL 2026
>
> **摘要:** AI design characteristics and human personality traits each impact the quality and outcomes of human-AI interactions. However, their relative and joint impacts are underexplored in imperfectly cooperative scenarios, where people and AI only have partially aligned goals and objectives. This study compares a purely simulated dataset comprising 2,000 simulations and a parallel human subjects experiment involving 290 human participants to investigate these effects across two scenario categories: (1) hiring negotiations between human job candidates and AI hiring agents; and (2) human-AI transactions wherein AI agents may conceal information to maximize internal goals. We examine user Extraversion and Agreeableness alongside AI design characteristics, including Adaptability, Expertise, and chain-of-thought Transparency. Our causal discovery analysis extends performance-focused evaluations by integrating scenario-based outcomes, communication analysis, and questionnaire measures. Results reveal divergences between purely simulated and human study datasets, and between scenario types. In simulation experiments, personality traits and AI attributes were comparatively influential. Yet, with actual human subjects, AI attributes -- particularly transparency -- were much more impactful. We discuss how these divergences vary across different interaction contexts, offering crucial insights for the future of human-centered AI agents.
>
---
#### [new 042] Why Fine-Tuning Encourages Hallucinations and How to Fix It
- **分类: cs.CL; cs.AI; cs.LG; cs.NE**

- **简介: 该论文属于自然语言处理任务，解决大模型在微调中产生幻觉的问题。通过自蒸馏和参数冻结等方法减少幻觉，提升事实准确性。**

- **链接: [https://arxiv.org/pdf/2604.15574](https://arxiv.org/pdf/2604.15574)**

> **作者:** Guy Kaplan; Zorik Gekhman; Zhen Zhu; Lotem Rozner; Yuval Reif; Swabha Swayamdipta; Derek Hoiem; Roy Schwartz
>
> **摘要:** Large language models are prone to hallucinating factually incorrect statements. A key source of these errors is exposure to new factual information through supervised fine-tuning (SFT), which can increase hallucinations w.r.t. knowledge acquired during pre-training. In this work, we explore whether SFT-induced hallucinations can be mitigated using established tools from the continual learning literature, since they arise as a by-product of knowledge degradation during training. We propose a self-distillation-based SFT method that facilitates effective factual learning while minimizing hallucinations w.r.t. pre-existing knowledge by regularizing output-distribution drift. We also show that, in settings where new knowledge acquisition is unnecessary, suppressing factual plasticity by freezing parameter groups, can preserve task performance while reducing hallucinations. Lastly, we investigate the mechanism behind SFT-induced hallucinations through three hypotheses: capacity limitations, behavior cloning, and localized interference. Our experiments show that a main driver is interference among overlapping semantic representations, and that self-distillation succeeds by mitigating this interference.
>
---
#### [new 043] No Universal Courtesy: A Cross-Linguistic, Multi-Model Study of Politeness Effects on LLMs Using the PLUM Corpus
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究不同礼貌程度对大语言模型响应的影响。通过多语言、多模型实验，分析礼貌与模型表现的关系，揭示其非普遍性。**

- **链接: [https://arxiv.org/pdf/2604.16275](https://arxiv.org/pdf/2604.16275)**

> **作者:** Hitesh Mehta; Arjit Saxena; Garima Chhikara; Rohit Kumar
>
> **摘要:** This paper explores the response of Large Language Models (LLMs) to user prompts with different degrees of politeness and impoliteness. The Politeness Theory by Brown and Levinson and the Impoliteness Framework by Culpeper form the basis of experiments conducted across three languages (English, Hindi, Spanish), five models (Gemini-Pro, GPT-4o Mini, Claude 3.7 Sonnet, DeepSeek-Chat, and Llama 3), and three interaction histories between users (raw, polite, and impolite). Our sample consists of 22,500 pairs of prompts and responses of various types, evaluated across five levels of politeness using an eight-factor assessment framework: coherence, clarity, depth, responsiveness, context retention, toxicity, conciseness, and readability. The findings show that model performance is highly influenced by tone, dialogue history, and language. While polite prompts enhance the average response quality by up to ~11% and impolite tones worsen it, these effects are neither consistent nor universal across languages and models. English is best served by courteous or direct tones, Hindi by deferential and indirect tones, and Spanish by assertive tones. Among the models, Llama is the most tone-sensitive (11.5% range), whereas GPT is more robust to adversarial tone. These results indicate that politeness is a quantifiable computational variable that affects LLM behaviour, though its impact is language- and model-dependent rather than universal. To support reproducibility and future work, we additionally release PLUM (Politeness Levels in Utterances, Multilingual), a publicly available corpus of 1,500 human-validated prompts across three languages and five politeness categories, and provide a formal supplementary analysis of six falsifiable hypotheses derived from politeness theory, empirically assessed against the dataset.
>
---
#### [new 044] Where does output diversity collapse in post-training?
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究后训练语言模型输出多样性下降的问题，分析不同训练方法对多样性的影响，揭示多样性损失由训练数据决定，而非生成格式。**

- **链接: [https://arxiv.org/pdf/2604.16027](https://arxiv.org/pdf/2604.16027)**

> **作者:** Constantinos Karouzos; Xingwei Tan; Nikolaos Aletras
>
> **摘要:** Post-trained language models produce less varied outputs than their base counterparts. This output diversity collapse undermines inference-time scaling methods that rely on varied samples, and risks homogenizing model outputs on creative and value-laden tasks. Prior work attributes collapse to specific post-training methods, without separating the role of training data composition from the method, or the generation format from the model weights. We trace output diversity through three parallel post-training lineages of Olmo 3, Think (chain-of-thought distillation), Instruct (broad multi-source data), and RL-Zero, across 15 tasks and four text diversity metrics. We find that the location of collapse co-varies with data composition: the Think lineage loses most semantic diversity at supervised fine-tuning, and the effect of DPO is larger in Instruct than in Think. Suppressing chain-of-thought reasoning at inference in Think models drops accuracy on hard tasks, yet leaves answer-level diversity unchanged, showing that the collapse is embedded in the model weights by training data, not imposed by the generation format. Decomposing diversity loss on six verifiable tasks into a quality-control component (removal of incorrect outputs) and a residual component (genuine narrowing among correct outputs) reveals that the split is task-dependent, and Think models retain more correct-answer diversity than Instruct despite collapsing more in aggregate. Our results indicate that diversity collapse is determined during training by data composition and cannot be addressed at inference time alone.
>
---
#### [new 045] MemEvoBench: Benchmarking Memory MisEvolution in LLM Agents
- **分类: cs.CL**

- **简介: 该论文属于LLM安全任务，解决记忆误进化问题。提出MemEvoBench基准，评估模型在对抗性记忆注入下的安全性能。**

- **链接: [https://arxiv.org/pdf/2604.15774](https://arxiv.org/pdf/2604.15774)**

> **作者:** Weiwei Xie; Shaoxiong Guo; Fan Zhang; Tian Xia; Xue Yang; Lizhuang Ma; Junchi Yan; Qibing Ren
>
> **摘要:** Equipping Large Language Models (LLMs) with persistent memory enhances interaction continuity and personalization but introduces new safety risks. Specifically, contaminated or biased memory accumulation can trigger abnormal agent behaviors. Existing evaluation methods have not yet established a standardized framework for measuring memory misevolution. This phenomenon refers to the gradual behavioral drift resulting from repeated exposure to misleading information. To address this gap, we introduce MemEvoBench, the first benchmark evaluating long-horizon memory safety in LLM agents against adversarial memory injection, noisy tool outputs, and biased feedback. The framework consists of QA-style tasks across 7 domains and 36 risk types, complemented by workflow-style tasks adapted from 20 Agent-SafetyBench environments with noisy tool returns. Both settings employ mixed benign and misleading memory pools within multi-round interactions to simulate memory evolution. Experiments on representative models reveal substantial safety degradation under biased memory updates. Our analysis suggests that memory evolution is a significant contributor to these failures. Furthermore, static prompt-based defenses prove insufficient, underscoring the urgency of securing memory evolution in LLM agents.
>
---
#### [new 046] RAGognizer: Hallucination-Aware Fine-Tuning via Detection Head Integration
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的幻觉检测任务，旨在解决生成内容与检索信息不一致的问题。通过引入检测头进行微调，提升模型的幻觉识别能力并减少生成错误。**

- **链接: [https://arxiv.org/pdf/2604.15945](https://arxiv.org/pdf/2604.15945)**

> **作者:** Fabian Ridder; Laurin Lessel; Malte Schilling
>
> **备注:** accepted at IJCNN 2026
>
> **摘要:** Retrieval-Augmented Generation (RAG) is widely used to augment the input to Large Language Models (LLMs) with external information, such as recent or domain-specific knowledge. Nonetheless, current models still produce closed-domain hallucinations and generate content that is unsupported by the retrieved context. Current detection approaches typically treat hallucination as a post-hoc problem, relying on black-box consistency checks or probes over frozen internal representations. In this work, we demonstrate that hallucination detection based on internal state representation can also serve as a direct training signal. We introduce RAGognize, a dataset of naturally occurring closed-domain hallucinations with token-level annotations, and RAGognizer, a hallucination-aware fine-tuning approach that integrates a lightweight detection head into an LLM, allowing for the joint optimization of language modeling and hallucination detection. This joint objective forces the model to improve the separability of its internal states regarding hallucinations while simultaneously learning to generate well-formed and meaningful responses. Across multiple benchmarks, RAGognizer achieves state-of-the-art token-level hallucination detection while substantially reducing hallucination rates during generation, without degrading language quality or relevance.
>
---
#### [new 047] Cut Your Losses! Learning to Prune Paths Early for Efficient Parallel Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于高效推理任务，旨在解决并行推理中因早期错误导致的无效路径问题。通过提出STOP方法实现路径剪枝，提升效率与效果。**

- **链接: [https://arxiv.org/pdf/2604.16029](https://arxiv.org/pdf/2604.16029)**

> **作者:** Jiaxi Bi; Tongxu Luo; Wenyu Du; Zhengyang Tang; Benyou Wang
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Parallel reasoning enhances Large Reasoning Models (LRMs) but incurs prohibitive costs due to futile paths caused by early errors. To mitigate this, path pruning at the prefix level is essential, yet existing research remains fragmented without a standardized framework. In this work, we propose the first systematic taxonomy of path pruning, categorizing methods by their signal source (internal vs. external) and learnability (learnable vs. non-learnable). This classification reveals the unexplored potential of learnable internal methods, motivating our proposal of STOP (Super TOken for Pruning). Extensive evaluations across LRMs ranging from 1.5B to 20B parameters demonstrate that STOP achieves superior effectiveness and efficiency compared to existing baselines. Furthermore, we rigorously validate the scalability of STOP under varying compute budgets - for instance, boosting GPT-OSS-20B accuracy on AIME25 from 84% to nearly 90% under fixed compute budgets. Finally, we distill our findings into formalized empirical guidelines to facilitate optimal real-world deployment. Code, data and models are available at this https URL
>
---
#### [new 048] AgentV-RL: Scaling Reward Modeling with Agentic Verifier
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Agentic Verifier框架，解决复杂领域中验证器的可靠性问题，通过双向代理和强化学习提升奖励建模效果。**

- **链接: [https://arxiv.org/pdf/2604.16004](https://arxiv.org/pdf/2604.16004)**

> **作者:** Jiazheng Zhang; Ziche Fu; Zhiheng Xi; Wenqing Jing; Mingxu Chai; Wei He; Guoqiang Zhang; Chenghao Fan; Chenxin An; Wenxiang Chen; Zhicheng Liu; Haojie Pan; Dingwei Zhu; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **备注:** ACL 2026
>
> **摘要:** Verifiers have been demonstrated to enhance LLM reasoning via test-time scaling (TTS). Yet, they face significant challenges in complex domains. Error propagation from incorrect intermediate reasoning can lead to false positives for seemingly plausible solutions, while lacking external grounding makes verifiers unreliable on computation or knowledge-intensive tasks. To address these challenges, we propose Agentic Verifier, a framework that transforms reward modeling into a multi-turn, tool-augmented deliberative process. We introduce complementary forward and backward agents: one traces solutions from premises to conclusions, while the other re-checks conclusions against their underlying premises. This bidirectional process enables a comprehensive, reliable, and interpretable assessment of solutions. To facilitate practical deployment, we propose AgentV-RL. Through proactive exploration and reinforcement learning, the verifier autonomously interleaves tool-use with internal reasoning. Extensive experiments show that Agentic Verifier yields consistent performance gains under both parallel and sequential TTS. Notably, our 4B variant surpasses state-of-the-art ORMs by 25.2%, positioning it as a promising paradigm for agentic reward modeling.
>
---
#### [new 049] GTA-2: Benchmarking General Tool Agents from Atomic Tool-Use to Open-Ended Workflows
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于智能代理任务，旨在解决工具使用基准与现实需求不匹配的问题。提出GTA-2基准，涵盖原子工具使用和开放工作流，评估模型与执行框架性能。**

- **链接: [https://arxiv.org/pdf/2604.15715](https://arxiv.org/pdf/2604.15715)**

> **作者:** Jize Wang; Xuanxuan Liu; Yining Li; Songyang Zhang; Yijun Wang; Zifei Shan; Xinyi Le; Cailian Chen; Xinping Guan; Dacheng Tao
>
> **摘要:** The development of general-purpose agents requires a shift from executing simple instructions to completing complex, real-world productivity workflows. However, current tool-use benchmarks remain misaligned with real-world requirements, relying on AI-generated queries, dummy tools, and limited system-level coordination. To address this, we propose GTA-2, a hierarchical benchmark for General Tool Agents (GTA) spanning atomic tool use and open-ended workflows. Built on real-world authenticity, it leverages real user queries, deployed tools, and multimodal contexts. (i) GTA-Atomic, inherited from our prior GTA benchmark, evaluates short-horizon, closed-ended tool-use precision. (ii) GTA-Workflow introduces long-horizon, open-ended tasks for realistic end-to-end completion. To evaluate open-ended deliverables, we propose a recursive checkpoint-based evaluation mechanism that decomposes objectives into verifiable sub-goals, enabling unified evaluation of both model capabilities and agent execution frameworks (i.e., execution harnesses). Experiments reveal a pronounced capability cliff: while frontier models already struggle on atomic tasks (below 50%), they largely fail on workflows, with top models achieving only 14.39% success. Further analysis shows that checkpoint-guided feedback improves performance, while advanced frameworks such as Manus and OpenClaw substantially enhance workflow completion, highlighting the importance of execution harness design beyond the underlying model capacity. These findings provide guidance for developing reliable personal and professional assistants. Dataset and code will be available at this https URL.
>
---
#### [new 050] Think Multilingual, Not Harder: A Data-Efficient Framework for Teaching Reasoning Models to Code-Switch
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决模型代码切换行为的优化问题。通过构建数据集和微调框架，提升模型在推理中有效代码切换的能力。**

- **链接: [https://arxiv.org/pdf/2604.15490](https://arxiv.org/pdf/2604.15490)**

> **作者:** Eleanor M. Lin; David Jurgens
>
> **摘要:** Recent developments in reasoning capabilities have enabled large language models to solve increasingly complex mathematical, symbolic, and logical tasks. Interestingly, while reasoning models are often trained to generate monolingual text, these models have also been observed to code-switch (i.e., mix languages). Prior works have either viewed code-switching as an undesirable error, attempted to control code-switching through modifications to input prompts or the output decoding process, or focus on narrow subsets of languages, domains, tasks, and models. We address these gaps by introducing the first linguistically and behaviorally motivated fine-tuning framework for identifying beneficial code-switched reasoning behaviors in large language models and teaching these models to code-switch more effectively for reasoning. First, we create and systematically analyze a dataset of reasoning traces from diverse models, languages, tasks, and domains to understand the types of code-switching behaviors found in existing reasoning models. Then, we develop fine-tuning interventions that teach reasoning models to code-switch based on our observations of helpful behaviors in existing models. We find that our framework can significantly increase beneficial code-switched reasoning behaviors in a data-efficient manner. Interestingly, we also find that code-switching behaviors in reasoning models can be modified by fine-tuning for tasks that do not directly demonstrate code-switching in reasoning (e.g., machine translation). Our work suggests that data-efficient interventions can instill helpful forms of code-switching behavior in reasoning models.
>
---
#### [new 051] PolicyBank: Evolving Policy Understanding for LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于LLM代理的政策理解任务，解决自然语言政策中的歧义与逻辑漏洞问题。通过PolicyBank机制，代理在测试中不断修正政策理解，提升合规性。**

- **链接: [https://arxiv.org/pdf/2604.15505](https://arxiv.org/pdf/2604.15505)**

> **作者:** Jihye Choi; Jinsung Yoon; Long T. Le; Somesh Jha; Tomas Pfister
>
> **摘要:** LLM agents operating under organizational policies must comply with authorization constraints typically specified in natural language. In practice, such specifications inevitably contain ambiguities and logical or semantic gaps that cause the agent's behavior to systematically diverge from the true requirements. We ask: by letting an agent evolve its policy understanding through interaction and corrective feedback from pre-deployment testing, can it autonomously refine its interpretation to close specification gaps? We propose PolicyBank, a memory mechanism that maintains structured, tool-level policy insights and iteratively refines them -- unlike existing memory mechanisms that treat the policy as immutable ground truth, reinforcing "compliant but wrong" behaviors. We also contribute a systematic testbed by extending a popular tool-calling benchmark with controlled policy gaps that isolate alignment failures from execution failures. While existing memory mechanisms achieve near-zero success on policy-gap scenarios, PolicyBank closes up to 82% of the gap toward a human oracle.
>
---
#### [new 052] Learning Uncertainty from Sequential Internal Dispersion in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于 hallucination 检测任务，旨在解决 LLM 中不确定性估计问题。提出 SIVR 方法，通过分析隐藏状态的层间方差来检测幻觉，避免信息丢失并提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.15741](https://arxiv.org/pdf/2604.15741)**

> **作者:** Ponhvoan Srey; Xiaobao Wu; Cong-Duy Nguyen; Anh Tuan Luu
>
> **备注:** Accepted at ACL 2026 (Main Conference)
>
> **摘要:** Uncertainty estimation is a promising approach to detect hallucinations in large language models (LLMs). Recent approaches commonly depend on model internal states to estimate uncertainty. However, they suffer from strict assumptions on how hidden states should evolve across layers, and from information loss by solely focusing on last or mean tokens. To address these issues, we present Sequential Internal Variance Representation (SIVR), a supervised hallucination detection framework that leverages token-wise, layer-wise features derived from hidden states. SIVR adopts a more basic assumption that uncertainty manifests in the degree of dispersion or variance of internal representations across layers, rather than relying on specific assumptions, which makes the method model and task agnostic. It additionally aggregates the full sequence of per-token variance features, learning temporal patterns indicative of factual errors and thereby preventing information loss. Experimental results demonstrate SIVR consistently outperforms strong baselines. Most importantly, SIVR enjoys stronger generalisation and avoids relying on large training sets, highlighting the potential for practical deployment. Our code repository is available online at this https URL.
>
---
#### [new 053] FD-NL2SQL: Feedback-Driven Clinical NL2SQL that Improves with Use
- **分类: cs.CL**

- **简介: 该论文属于自然语言到SQL（NL2SQL）任务，旨在帮助临床医生高效查询肿瘤学数据库。通过反馈机制不断优化生成的SQL语句，提升查询准确性和实用性。**

- **链接: [https://arxiv.org/pdf/2604.15646](https://arxiv.org/pdf/2604.15646)**

> **作者:** Suparno Roy Chowdhury; Tejas Anvekar; Manan Roy Choudhury; Muhammad Ali Khan; Kaneez Zahra Rubab Khakwani; Mohamad Bassam Sonbol; Irbaz Bin Riaz; Vivek Gupta
>
> **摘要:** Clinicians exploring oncology trial repositories often need ad-hoc, multi-constraint queries over biomarkers, endpoints, interventions, and time, yet writing SQL requires schema expertise. We demo FD-NL2SQL, a feedback-driven clinical NL2SQL assistant for SQLite-based oncology databases. Given a natural-language question, a schema-aware LLM decomposes it into predicate-level sub-questions, retrieves semantically similar expert-verified NL2SQL exemplars via sentence embeddings, and synthesizes executable SQL conditioned on the decomposition, retrieved exemplars, and schema, with post-processing validity checks. To improve with use, FD-NL2SQL incorporates two update signals: (i) clinician edits of generated SQL are approved and added to the exemplar bank; and (ii) lightweight logic-based SQL augmentation applies a single atomic mutation (e.g., operator or column change), retaining variants only if they return non-empty results. A second LLM generates the corresponding natural-language question and predicate decomposition for accepted variants, automatically expanding the exemplar bank without additional annotation. The demo interface exposes decomposition, retrieval, synthesis, and execution results to support interactive refinement and continuous improvement.
>
---
#### [new 054] How Hypocritical Is Your LLM judge? Listener-Speaker Asymmetries in the Pragmatic Competence of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，探讨LLM在语用能力上的听者与说者角色差异。研究发现模型作为听者比作为说者表现更优，揭示了评价与生成任务的弱关联，呼吁更综合的评估方式。**

- **链接: [https://arxiv.org/pdf/2604.15873](https://arxiv.org/pdf/2604.15873)**

> **作者:** Judith Sieker; Sina Zarrieß
>
> **备注:** Accepted at ACL 2026 (findings)
>
> **摘要:** Large language models (LLMs) are increasingly studied as repositories of linguistic knowledge. In this line of work, models are commonly evaluated both as generators of language and as judges of linguistic output, yet these two roles are rarely examined in direct relation to one another. As a result, it remains unclear whether success in one role aligns with success in the other. In this paper, we address this question for pragmatic competence by comparing LLMs' performance as pragmatic listeners, judging the appropriateness of linguistic outputs, and as pragmatic speakers, generating pragmatically appropriate language. We evaluate multiple open-weight and proprietary LLMs across three pragmatic settings. We find a robust asymmetry between pragmatic evaluation and pragmatic generation: many models perform substantially better as listeners than as speakers. Our results suggest that pragmatic judging and pragmatic generation are only weakly aligned in current LLMs, calling for more integrated evaluation practices.
>
---
#### [new 055] Discover and Prove: An Open-source Agentic Framework for Hard Mode Automated Theorem Proving in Lean 4
- **分类: cs.AI; cs.CL; cs.LO**

- **简介: 该论文提出DAP框架，解决Hard Mode自动定理证明问题。通过释放新的基准和引入自主推理机制，提升定理证明效果，揭示LLM与形式化证明工具间的差距。**

- **链接: [https://arxiv.org/pdf/2604.15839](https://arxiv.org/pdf/2604.15839)**

> **作者:** Chengwu Liu; Yichun Yin; Ye Yuan; Jiaxuan Xie; Botao Li; Siqi Li; Jianhao Shen; Yan Xu; Lifeng Shang; Ming Zhang
>
> **备注:** ACL 2026 Main Conference
>
> **摘要:** Most ATP benchmarks embed the final answer within the formal statement -- a convention we call "Easy Mode" -- a design that simplifies the task relative to what human competitors face and may lead to optimistic estimates of model capability. We call the stricter, more realistic setting "Hard Mode": the system must independently discover the answer before constructing a formal proof. To enable Hard Mode research, we make two contributions. First, we release MiniF2F-Hard and FIMO-Hard, expert-reannotated Hard Mode variants of two widely-used ATP benchmarks. Second, we introduce Discover And Prove (DAP), an agentic framework that uses LLM natural-language reasoning with explicit self-reflection to discover answers, then rewrites Hard Mode statements into Easy Mode ones for existing ATP provers. DAP sets the state of the art: on CombiBench it raises solved problems from 7 (previous SOTA, Pass@16) to 10; on PutnamBench it is the first system to formally prove 36 theorems in Hard Mode -- while simultaneously revealing that state-of-the-art LLMs exceed 80% answer accuracy on the same problems where formal provers manage under 10%, exposing a substantial gap that Hard Mode benchmarks are uniquely suited to measure.
>
---
#### [new 056] Learning to Reason with Insight for Informal Theorem Proving
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于数学定理证明任务，旨在解决非形式化证明中缺乏洞察力的问题。通过构建包含核心技巧的数据集和渐进式训练策略，提升模型的数学推理能力。**

- **链接: [https://arxiv.org/pdf/2604.16278](https://arxiv.org/pdf/2604.16278)**

> **作者:** Yunhe Li; Hao Shi; Bowen Deng; Wei Wang; Mengzhe Ruan; Hanxu Hou; Zhongxiang Dai; Siyang Gao; Chao Wang; Shuang Qiu; Linqi Song
>
> **摘要:** Although most of the automated theorem-proving approaches depend on formal proof systems, informal theorem proving can align better with large language models' (LLMs) strength in natural language processing. In this work, we identify a primary bottleneck in informal theorem proving as a lack of insight, namely the difficulty of recognizing the core techniques required to solve complex problems. To address this, we propose a novel framework designed to cultivate this essential reasoning skill and enable LLMs to perform insightful reasoning. We propose $\mathtt{DeepInsightTheorem}$, a hierarchical dataset that structures informal proofs by explicitly extracting core techniques and proof sketches alongside the final proof. To fully exploit this dataset, we design a Progressive Multi-Stage SFT strategy that mimics the human learning process, guiding the model from basic proof writing to insightful thinking. Our experiments on challenging mathematical benchmarks demonstrate that this insight-aware generation strategy significantly outperforms baselines. These results demonstrate that teaching models to identify and apply core techniques can substantially improve their mathematical reasoning.
>
---
#### [new 057] LLMSniffer: Detecting LLM-Generated Code via GraphCodeBERT and Supervised Contrastive Learning
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于代码生成检测任务，旨在区分AI生成与人工编写的代码。通过改进GraphCodeBERT模型，结合对比学习和评论去除技术，提升了检测效果。**

- **链接: [https://arxiv.org/pdf/2604.16058](https://arxiv.org/pdf/2604.16058)**

> **作者:** Mahir Labib Dihan; Abir Muhtasim
>
> **摘要:** The rapid proliferation of Large Language Models (LLMs) in software development has made distinguishing AI-generated code from human-written code a critical challenge with implications for academic integrity, code quality assurance, and software security. We present LLMSniffer, a detection framework that fine-tunes GraphCodeBERT using a two-stage supervised contrastive learning pipeline augmented with comment removal preprocessing and an MLP classifier. Evaluated on two benchmark datasets - GPTSniffer and Whodunit - LLMSniffer achieves substantial improvements over prior baselines: accuracy increases from 70% to 78% on GPTSniffer (F1: 68% to 78%) and from 91% to 94.65% on Whodunit (F1: 91% to 94.64%). t-SNE visualizations confirm that contrastive fine-tuning yields well-separated, compact embeddings. We release our model checkpoints, datasets, codes and a live interactive demo to facilitate further research.
>
---
#### [new 058] Self-Distillation as a Performance Recovery Mechanism for LLMs: Counteracting Compression and Catastrophic Forgetting
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型优化任务，旨在解决LLMs在微调、量化和剪枝中性能下降的问题。通过自蒸馏方法恢复模型能力，理论分析其与高维流形对齐的关系。**

- **链接: [https://arxiv.org/pdf/2604.15794](https://arxiv.org/pdf/2604.15794)**

> **作者:** Chi Liu; Xin Chen; Xu Zhou; Fangbo Tu; Srinivasan Manoharan
>
> **备注:** 14 pages, 8 figures
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable success, underpinning diverse AI applications. However, they often suffer from performance degradation due to factors such as catastrophic forgetting during Supervised Fine-Tuning (SFT), quantization, and pruning. In this work, we introduce a performance recovery framework based on Self-Distillation Fine-Tuning (SDFT) that effectively restores model capabilities. Complementing this practical contribution, we provide a rigorous theoretical explanation for the underlying recovery mechanism. We posit that an LLM's generative capability fundamentally relies on the high-dimensional manifold constructed by its hidden layers. To investigate this, we employ Centered Kernel Alignment (CKA) to quantify the alignment between student and teacher activation trajectories, leveraging its invariance to orthogonal transformations and scaling. Our experiments demonstrate a strong correlation between performance recovery and manifold alignment, substantiating the claim that self-distillation effectively aligns the student's high-dimensional manifold with the optimal structure represented by the teacher. This study bridges the gap between practical recovery frameworks and geometric representation theory, offering new insights into the internal mechanisms of self-distillation.
>
---
#### [new 059] JFinTEB: Japanese Financial Text Embedding Benchmark
- **分类: cs.IR; cs.CL**

- **简介: 该论文提出JFinTEB，首个针对日语金融文本嵌入的基准。解决日语金融文本表示不足的问题，涵盖检索与分类任务，评估多种模型并公开数据集。**

- **链接: [https://arxiv.org/pdf/2604.15882](https://arxiv.org/pdf/2604.15882)**

> **作者:** Masahiro Suzuki; Hiroki Sakaji
>
> **备注:** 5 pages. Accepted at SIGIR 2026 Resource Track
>
> **摘要:** We introduce JFinTEB, the first comprehensive benchmark specifically designed for evaluating Japanese financial text embeddings. Existing embedding benchmarks provide limited coverage of language-specific and domain-specific aspects found in Japanese financial texts. Our benchmark encompasses diverse task categories including retrieval and classification tasks that reflect realistic and well-defined financial text processing scenarios. The retrieval tasks leverage instruction-following datasets and financial text generation queries, while classification tasks cover sentiment analysis, document categorization, and domain-specific classification challenges derived from economic survey data. We conduct extensive evaluations across a wide range of embedding models, including Japanese-specific models of various sizes, multilingual models, and commercial embedding services. We publicly release JFinTEB datasets and evaluation framework at this https URL to facilitate future research and provide a standardized evaluation protocol for the Japanese financial text mining community. This work addresses a critical gap in Japanese financial text processing resources and establishes a foundation for advancing domain-specific embedding research.
>
---
#### [new 060] JumpLoRA: Sparse Adapters for Continual Learning in Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于持续学习任务，旨在解决大语言模型中的灾难性遗忘问题。提出JumpLoRA框架，通过引入稀疏性机制提升模型性能。**

- **链接: [https://arxiv.org/pdf/2604.16171](https://arxiv.org/pdf/2604.16171)**

> **作者:** Alexandra Dragomir; Ioana Pintilie; Antonio Barbalau; Marius Dragoi; Florin Brad; Cristian Daniel Paduraru; Alexandru Tifrea; Elena Burceanu; Radu Tudor Ionescu
>
> **摘要:** Adapter-based methods have become a cost-effective approach to continual learning (CL) for Large Language Models (LLMs), by sequentially learning a low-rank update matrix for each task. To mitigate catastrophic forgetting, state-of-the-art approaches impose constraints on new adapters with respect to the previous ones, by targeting either subspace or coordinate-wise interference. In this paper, we propose JumpLoRA, a novel framework to adaptively induce sparsity in the Low-Rank Adaptation (LoRA) blocks through the use of JumpReLU gating. The method achieves dynamic parameter isolation, which helps prevent task interference. We demonstrate that our method is highly modular and compatible with LoRA-based CL approaches. Specifically, it significantly boosts the performance of IncLoRA and outperforms the leading state-of-the-art CL method, ELLA.
>
---
#### [new 061] Evaluating LLM Simulators as Differentially Private Data Generators
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文属于隐私数据生成任务，旨在评估LLM模拟器作为差分隐私数据生成器的性能。研究发现LLM存在分布偏移问题，需解决其对输入统计的忠实再现。**

- **链接: [https://arxiv.org/pdf/2604.15461](https://arxiv.org/pdf/2604.15461)**

> **作者:** Nassima M. Bouzid; Dehao Yuan; Nam H. Nguyen; Mayana Pereira
>
> **备注:** Submitted to ICLR 2026. 6 pages + appendix
>
> **摘要:** LLM-based simulators offer a promising path for generating complex synthetic data where traditional differentially private (DP) methods struggle with high-dimensional user profiles. But can LLMs faithfully reproduce statistical distributions from DP-protected inputs? We evaluate this using PersonaLedger, an agentic financial simulator, seeded with DP synthetic personas derived from real user statistics. We find that PersonaLedger achieves promising fraud detection utility (AUC 0.70 at epsilon=1) but exhibits significant distribution drift due to systematic LLM biases--learned priors overriding input statistics for temporal and demographic features. These failure modes must be addressed before LLM-based methods can handle the richer user representations where they might otherwise excel.
>
---
#### [new 062] Weak-Link Optimization for Multi-Agent Reasoning and Collaboration
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于多智能体协作任务，旨在解决协作中因弱代理导致的推理不稳定问题。提出WORC框架，通过识别并优化弱代理提升系统整体性能。**

- **链接: [https://arxiv.org/pdf/2604.15972](https://arxiv.org/pdf/2604.15972)**

> **作者:** Haoyu Bian; Chaoning Zhang; Jiaquan Zhang; Xingyao Li; Yuanfang Guo; Wei Dong; Yang Yang
>
> **备注:** 13 pages, 4 figures. Submitted to CAAI Transactions on Intelligence Technology
>
> **摘要:** LLM-driven multi-agent frameworks address complex reasoning tasks through multi-role collaboration. However, existing approaches often suffer from reasoning instability, where individual agent errors are amplified through collaboration, undermining overall performance. Current research mainly focuses on enhancing high-capability agents or suppressing unreliable outputs to improve framework effectiveness, while systematic identification and reinforcement of performance-limiting agents receive less attention. To address this gap, we propose WORC, a \underline{w}eak-link \underline{o}ptimization framework for multi-agent \underline{r}easoning and \underline{c}ollaboration, grounded in the weak-link principle. WORC follows a two-stage workflow. In the weak agent localization stage, task features are constructed, and a meta-learning-based weight predictor trained on optimal configurations identified by swarm intelligence algorithms (SIAs) enables zero-shot mapping from these features to agent performance weights, where the agent with the lowest predicted weight is identified as the weak agent. In the weak-link optimization stage, an uncertainty-driven allocation strategy assigns additional reasoning budgets to weak agents, with lower predicted weights leading to larger repeated-sampling quotas to compensate for reliability deficiencies. Experimental results show that WORC achieves an average accuracy of 82.2\% on reasoning benchmarks while improving framework stability and cross-architecture generalization, suggesting that compensating for weak links, rather than reinforcing strengths alone, enhances the robustness of multi-agent systems.
>
---
#### [new 063] FineSteer: A Unified Framework for Fine-Grained Inference-Time Steering in Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出FineSteer，解决大语言模型推理时行为控制问题，通过分解为条件引导和细粒度向量合成，实现高效、精准的模型行为调整。**

- **链接: [https://arxiv.org/pdf/2604.15488](https://arxiv.org/pdf/2604.15488)**

> **作者:** Zixuan Weng; Jinghuai Zhang; Kunlin Cai; Ying Li; Peiran Wang; Yuan Tian
>
> **备注:** Accepted by ACL 2026 (Main)
>
> **摘要:** Large language models (LLMs) often exhibit undesirable behaviors, such as safety violations and hallucinations. Although inference-time steering offers a cost-effective way to adjust model behavior without updating its parameters, existing methods often fail to be simultaneously effective, utility-preserving, and training-efficient due to their rigid, one-size-fits-all designs and limited adaptability. In this work, we present FineSteer, a novel steering framework that decomposes inference-time steering into two complementary stages: conditional steering and fine-grained vector synthesis, allowing fine-grained control over when and how to steer internal representations. In the first stage, we introduce a Subspace-guided Conditional Steering (SCS) mechanism that preserves model utility by avoiding unnecessary steering. In the second stage, we propose a Mixture-of-Steering-Experts (MoSE) mechanism that captures the multimodal nature of desired steering behaviors and generates query-specific steering vectors for improved effectiveness. Through tailored designs in both SCS and MoSE, FineSteer maintains robust performance on general queries while adaptively optimizing steering vectors for targeted inputs in a training-efficient manner. Extensive experiments on safety and truthfulness benchmarks show that FineSteer outperforms state-of-the-art methods in overall performance, achieving stronger steering performance with minimal utility loss. Code is available at this https URL
>
---
#### [new 064] Rethinking the Necessity of Adaptive Retrieval-Augmented Generation through the Lens of Adaptive Listwise Ranking
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，旨在重新评估自适应检索的必要性。通过提出AdaRankLLM框架，解决噪声干扰与模型效率问题，提升生成效果并降低上下文开销。**

- **链接: [https://arxiv.org/pdf/2604.15621](https://arxiv.org/pdf/2604.15621)**

> **作者:** Jun Feng; Jiahui Tang; Zhicheng He; Hang Lv; Hongchao Gu; Hao Wang; Xuezhi Yang; Shuai Fang
>
> **备注:** 7pages, 2figures
>
> **摘要:** Adaptive Retrieval-Augmented Generation aims to mitigate the interference of extraneous noise by dynamically determining the necessity of retrieving supplementary passages. However, as Large Language Models evolve with increasing robustness to noise, the necessity of adaptive retrieval warrants re-evaluation. In this paper, we rethink this necessity and propose AdaRankLLM, a novel adaptive retrieval framework. To effectively verify the necessity of adaptive listwise reranking, we first develop an adaptive ranker employing a zero-shot prompt with a passage dropout mechanism, and compare its generation outcomes against static fixed-depth retrieval strategies. Furthermore, to endow smaller open-source LLMs with this precise listwise ranking and adaptive filtering capability, we introduce a two-stage progressive distillation paradigm enhanced by data sampling and augmentation techniques. Extensive experiments across three datasets and eight LLMs demonstrate that AdaRankLLM consistently achieves optimal performance in most scenarios with significantly reduced context overhead. Crucially, our analysis reveals a role shift in adaptive retrieval: it functions as a critical noise filter for weaker models to overcome their limitations, while serving as a cost-effective efficiency optimizer for stronger reasoning models.
>
---
#### [new 065] Faster LLM Inference via Sequential Monte Carlo
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出SMC-SD方法，用于加速大语言模型推理。针对传统推测解码中因拒绝导致的吞吐量下降问题，通过重要性加权重采样提升效率，实现更快推理且保持较高准确率。**

- **链接: [https://arxiv.org/pdf/2604.15672](https://arxiv.org/pdf/2604.15672)**

> **作者:** Yahya Emara; Mauricio Barba da Costa; Chi-Chih Chang; Cameron Freer; Tim Vieira; Ryan Cotterell; Mohamed S. Abdelfattah
>
> **摘要:** Speculative decoding (SD) accelerates language model inference by drafting tokens from a cheap proposal model and verifying them against an expensive target model via rejection sampling. Because rejection truncates the draft block at the first error, throughput degrades when draft and target diverge. Rather than rejecting draft tokens outright, we propose to reweight them. To this end, we introduce sequential Monte Carlo speculative decoding (SMC-SD), which replaces token-level rejection with importance-weighted resampling over a population of draft particles. SMC-SD is a principled approximate inference scheme that trades exactness for additional speed, while preserving theoretical bounds on its per-step approximation error. Because LLM inference is memory bandwidth-bound, the arithmetic needed to draft particles and to score them in parallel comes nearly for free -- SMC-SD uses idle compute to turn verification into a vectorized, fixed-size operation with no rollback. Empirically, SMC-SD achieves 2.36x speed-up over speculative decoding and a 5.2x speed-up over autoregressive decoding, while remaining within 3% of the target model's accuracy on reasoning, instruction-following, and coding benchmarks.
>
---
#### [new 066] UsefulBench: Towards Decision-Useful Information as a Target for Information Retrieval
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决传统方法仅关注文本相关性而忽视实际有用性的问题。作者构建了UsefulBench数据集，区分相关性与实用性，并探讨提升系统实用性的方法。**

- **链接: [https://arxiv.org/pdf/2604.15827](https://arxiv.org/pdf/2604.15827)**

> **作者:** Tobias Schimanski; Stefanie Lewandowski; Christian Woerle; Nicola Reichenau; Yauheni Huryn; Markus Leippold
>
> **摘要:** Conventional information retrieval is concerned with identifying the relevance of texts for a given query. Yet, the conventional definition of relevance is dominated by aspects of similarity in texts, leaving unobserved whether the text is truly useful for addressing the query. For instance, when answering whether Paris is larger than Berlin, texts about Paris being in France are relevant (lexical/semantic similarity), but not useful. In this paper, we introduce UsefulBench, a domain-specific dataset curated by three professional analysts labeling whether a text is connected to a query (relevance) or holds practical value in responding to it (usefulness). We show that classic similarity-based information retrieval aligns more strongly with relevance. While LLM-based systems can counteract this bias, we find that domain-specific problems require a high degree of expertise, which current LLMs do not fully incorporate. We explore approaches to (partially) overcome this challenge. However, UsefulBench presents a dataset challenge for targeted information retrieval systems.
>
---
#### [new 067] Polarization by Default: Auditing Recommendation Bias in LLM-Based Content Curation
- **分类: cs.SI; cs.AI; cs.CL; cs.CY; cs.MA**

- **简介: 该论文属于内容推荐偏差审计任务，旨在研究LLM在内容筛选中的偏见。通过实验分析不同提示策略下的偏见表现，揭示模型行为差异及潜在的意识形态倾向。**

- **链接: [https://arxiv.org/pdf/2604.15937](https://arxiv.org/pdf/2604.15937)**

> **作者:** Nicolò Pagan; Christopher Barrie; Chris Andrew Bail; Petter Törnberg
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed to curate and rank human-created content, yet the nature and structure of their biases in these tasks remains poorly understood: which biases are robust across providers and platforms, and which can be mitigated through prompt design. We present a controlled simulation study mapping content selection biases across three major LLM providers (OpenAI, Anthropic, Google) on real social media datasets from Twitter/X, Bluesky, and Reddit, using six prompting strategies (\textit{general}, \textit{popular}, \textit{engaging}, \textit{informative}, \textit{controversial}, \textit{neutral}). Through 540,000 simulated top-10 selections from pools of 100 posts across 54 experimental conditions, we find that biases differ substantially in how structural and how prompt-sensitive they are. Polarization is amplified across all configurations, toxicity handling shows a strong inversion between engagement- and information-focused prompts, and sentiment biases are predominantly negative. Provider comparisons reveal distinct trade-offs: GPT-4o Mini shows the most consistent behavior across prompts; Claude and Gemini exhibit high adaptivity in toxicity handling; Gemini shows the strongest negative sentiment preference. On Twitter/X, where author demographics can be inferred from profile bios, political leaning bias is the clearest demographic signal: left-leaning authors are systematically over-represented despite right-leaning authors forming the pool plurality in the dataset, and this pattern largely persists across prompts.
>
---
#### [new 068] Do Vision-Language Models Truly Perform Vision Reasoning? A Rigorous Study of the Modality Gap
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型研究，旨在解决VLM是否真正进行视觉推理的问题。通过构建CrossMath基准，发现VLM主要依赖文本推理，视觉信息利用不足，并提出训练集提升性能。**

- **链接: [https://arxiv.org/pdf/2604.16256](https://arxiv.org/pdf/2604.16256)**

> **作者:** Yige Xu; Yongjie Wang; Zizhuo Wu; Kaisong Song; Jun Lin; Zhiqi Shen
>
> **摘要:** Reasoning in vision-language models (VLMs) has recently attracted significant attention due to its broad applicability across diverse downstream tasks. However, it remains unclear whether the superior performance of VLMs stems from genuine vision-grounded reasoning or relies predominantly on the reasoning capabilities of their textual backbones. To systematically measure this, we introduce CrossMath, a novel multimodal reasoning benchmark designed for controlled cross-modal comparisons. Specifically, we construct each problem in text-only, image-only, and image+text formats guaranteeing identical task-relevant information, verified by human annotators. This rigorous alignment effectively isolates modality-specific reasoning differences while eliminating confounding factors such as information mismatch. Extensive evaluation of state-of-the-art VLMs reveals a consistent phenomenon: a substantial performance gap between textual and visual reasoning. Notably, VLMs excel with text-only inputs, whereas incorporating visual data (image+text) frequently degrades performance compared to the text-only baseline. These findings indicate that current VLMs conduct reasoning primarily in the textual space, with limited genuine reliance on visual evidence. To mitigate this limitation, we curate a CrossMath training set for VLM fine-tuning. Empirical evaluations demonstrate that fine-tuning on this training set significantly boosts reasoning performance across all individual and joint modalities, while yielding robust gains on two general visual reasoning tasks. Source code is available at this https URL.
>
---
#### [new 069] SIMMER: Cross-Modal Food Image--Recipe Retrieval via MLLM-Based Embedding
- **分类: cs.CV; cs.CL; cs.IR; cs.LG; cs.MM**

- **简介: 该论文属于跨模态检索任务，旨在解决食物图像与食谱文本之间的匹配问题。提出SIMMER模型，使用MLLM进行统一编码，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2604.15628](https://arxiv.org/pdf/2604.15628)**

> **作者:** Keisuke Gomi; Keiji Yanai
>
> **备注:** 20 pages, 6 figures
>
> **摘要:** Cross-modal retrieval between food images and recipe texts is an important task with applications in nutritional management, dietary logging, and cooking assistance. Existing methods predominantly rely on dual-encoder architectures with separate image and text encoders, requiring complex alignment strategies and task-specific network designs to bridge the semantic gap between modalities. In this work, we propose SIMMER (Single Integrated Multimodal Model for Embedding Recipes), which applies Multimodal Large Language Model (MLLM)-based embedding models, specifically VLM2Vec, to this task, replacing the conventional dual-encoder paradigm with a single unified encoder that processes both food images and recipe texts. We design prompt templates tailored to the structured nature of recipes, which consist of a title, ingredients, and cooking instructions, enabling effective embedding generation by the MLLM. We further introduce a component-aware data augmentation strategy that trains the model on both complete and partial recipes, improving robustness to incomplete inputs. Experiments on the Recipe1M dataset demonstrate that SIMMER achieves state-of-the-art performance across both the 1k and 10k evaluation settings, substantially outperforming all prior methods. In particular, our best model improves the 1k image-to-recipe R@1 from 81.8\% to 87.5\% and the 10k image-to-recipe R@1 from 56.5\% to 65.5\% compared to the previous best method.
>
---
#### [new 070] From Intention to Text: AI-Supported Goal Setting in Academic Writing
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文提出WriteFlow，一种支持学术写作中目标设定的AI语音助手，解决写作过程中目标不明确和管理困难的问题。通过对话式交互促进反思性写作。**

- **链接: [https://arxiv.org/pdf/2604.15800](https://arxiv.org/pdf/2604.15800)**

> **作者:** Yueling Fan; Richard Lee Davis; Olga Viberg
>
> **备注:** Accepted at AIED 2026
>
> **摘要:** This study presents WriteFlow, an AI voice-based writing assistant designed to support reflective academic writing through goal-oriented interaction. Academic writing involves iterative reflection and evolving goal regulation, yet prior research and a formative study with 17 participants show that writers often struggle to articulate and manage changing goals. While commonly used AI writing tools emphasize efficiency, they offer limited support for metacognition and writer agency. WriteFlow frames AI interaction as a dialogic space for ongoing goal articulation, monitoring, and negotiation grounded in writers' intentions. Findings from a Wizard-of-Oz study with 12 expert users show that WriteFlow scaffolds metacognitive regulation and reflection-in-action by supporting iterative goal refinement, maintaining goal-text alignment during drafting, and prompting evaluation of goal fulfillment. We discuss design implications for AI writing systems that prioritize reflective dialogue, flexible goal structures, and multi-perspective feedback to support intentional and agentic writing.
>
---
#### [new 071] Experience Compression Spectrum: Unifying Memory, Skills, and Rules in LLM Agents
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于人工智能领域，旨在解决LLM代理在长期任务中高效管理经验的问题。提出“经验压缩谱”框架，统一记忆、技能与规则，提升系统效率。**

- **链接: [https://arxiv.org/pdf/2604.15877](https://arxiv.org/pdf/2604.15877)**

> **作者:** Xing Zhang; Guanghui Wang; Yanwei Cui; Wei Qiu; Ziyuan Li; Bing Zhu; Peiyang He
>
> **摘要:** As LLM agents scale to long-horizon, multi-session deployments, efficiently managing accumulated experience becomes a critical bottleneck. Agent memory systems and agent skill discovery both address this challenge -- extracting reusable knowledge from interaction traces -- yet a citation analysis of 1,136 references across 22 primary papers reveals a cross-community citation rate below 1%. We propose the \emph{Experience Compression Spectrum}, a unifying framework that positions memory, skills, and rules as points along a single axis of increasing compression (5--20$\times$ for episodic memory, 50--500$\times$ for procedural skills, 1,000$\times$+ for declarative rules), directly reducing context consumption, retrieval latency, and compute overhead. Mapping 20+ systems onto this spectrum reveals that every system operates at a fixed, predetermined compression level -- none supports adaptive cross-level compression, a gap we term the \emph{missing diagonal}. We further show that specialization alone is insufficient -- both communities independently solve shared sub-problems without exchanging solutions -- that evaluation methods are tightly coupled to compression levels, that transferability increases with compression at the cost of specificity, and that knowledge lifecycle management remains largely neglected. We articulate open problems and design principles for scalable, full-spectrum agent learning systems.
>
---
#### [new 072] RefereeBench: Are Video MLLMs Ready to be Multi-Sport Referees
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态模型评估任务，旨在解决MLLM在体育裁判决策中的能力不足问题。构建了RefereeBench基准，评估模型在规则判断、时间定位等能力，发现现有模型表现有限。**

- **链接: [https://arxiv.org/pdf/2604.15736](https://arxiv.org/pdf/2604.15736)**

> **作者:** Yichen Xu; Yuanhang Liu; Chuhan Wang; Zihan Zhao; jinghan luo; Jianzhe Ma; Wenxuan Wang; Qin Jin
>
> **备注:** Work in Progress
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel at generic video understanding, their ability to support specialized, rule-grounded decision-making remains insufficiently explored. In this paper, we introduce RefereeBench, the first large-scale benchmark for evaluating MLLMs as automatic sports referees. Spanning 11 sports with 925 curated videos and 6,475 QA pairs, RefereeBench evaluates five core officiating abilities: foul existence, foul and penalty classification, foul and penalty reasoning, entity perception, and temporal grounding. The benchmark is fully human-annotated to ensure high-quality annotations grounded in authentic officiating logic and multimodal evidence. Extensive evaluations of state-of-the-art MLLMs show that even the strongest models, such as Doubao-Seed-1.8 and Gemini-3-Pro, achieve only around 60% accuracy, while the strongest open-source model, Qwen3-VL, reaches only 47%. These results indicate that current models remain far from being reliable sports referees. Further analysis shows that while models can often identify incidents and involved entities, they struggle with rule application and temporal grounding, and frequently over-call fouls on normal clips. Our benchmark highlights the need for future MLLMs that better integrate domain knowledge and multimodal understanding, advancing trustworthy AI-assisted officiating and broader multimodal decision-making.
>
---
#### [new 073] Preregistered Belief Revision Contracts
- **分类: cs.AI; cs.CL; cs.LO; cs.MA**

- **简介: 该论文属于多智能体系统任务，解决信念修正中的共识风险问题。提出PBRC机制，分离通信与信念更新，确保可审计和可问责的信念变化。**

- **链接: [https://arxiv.org/pdf/2604.15558](https://arxiv.org/pdf/2604.15558)**

> **作者:** Saad Alqithami
>
> **摘要:** Deliberative multi-agent systems allow agents to exchange messages and revise beliefs over time. While this interaction is meant to improve performance, it can also create dangerous conformity effects: agreement, confidence, prestige, or majority size may be treated as if they were evidence, producing high-confidence convergence to false conclusions. To address this, we introduce PBRC (Preregistered Belief Revision Contracts), a protocol-level mechanism that strictly separates open communication from admissible epistemic change. A PBRC contract publicly fixes first-order evidence triggers, admissible revision operators, a priority rule, and a fallback policy. A non-fallback step is accepted only when it cites a preregistered trigger and provides a nonempty witness set of externally validated evidence tokens. This ensures that every substantive belief change is both enforceable by a router and auditable after the fact. In this paper, (a) we prove that under evidential contracts with conservative fallback, social-only rounds cannot increase confidence and cannot generate purely conformity-driven wrong-but-sure cascades. (b) We show that auditable trigger protocols admit evidential PBRC normal forms that preserve belief trajectories and canonicalized audit traces. (c) We demonstrate that sound enforcement yields epistemic accountability: any change of top hypothesis is attributable to a concrete validated witness set. For token-invariant contracts, (d) we prove that enforced trajectories depend only on token-exposure traces; under flooding dissemination, these traces are characterized exactly by truncated reachability, giving tight diameter bounds for universal evidence closure. Finally, we introduce a companion contractual dynamic doxastic logic to specify trace invariants, and provide simulations illustrating cascade suppression, auditability, and robustness-liveness trade-offs.
>
---
#### [new 074] Hallucination as Trajectory Commitment: Causal Evidence for Asymmetric Attractor Dynamics in Transformer Generation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究语言模型中的幻觉现象，通过实验揭示其由不对称吸引子动力学驱动，提出轨迹承诺机制，旨在理解并控制生成过程中的错误输出。**

- **链接: [https://arxiv.org/pdf/2604.15400](https://arxiv.org/pdf/2604.15400)**

> **作者:** G. Aytug Akarlar
>
> **备注:** 21 pages, 12 figures, 8 tables. Code and data: this https URL
>
> **摘要:** We present causal evidence that hallucination in autoregressive language models is an early trajectory commitment governed by asymmetric attractor dynamics. Using same-prompt bifurcation, in which we repeatedly sample identical inputs to observe spontaneous divergence, we isolate trajectory dynamics from prompt-level confounds. On Qwen2.5-1.5B across 61 prompts spanning six categories, 27 prompts (44.3%) bifurcate with factual and hallucinated trajectories diverging at the first generated token (KL = 0 at step 0, KL > 1.0 at step 1). Activation patching across 28 layers reveals a pronounced causal asymmetry: injecting a hallucinated activation into a correct trajectory corrupts output in 87.5% of trials (layer 20), while the reverse recovers only 33.3% (layer 24); both exceed the 10.4% baseline (p = 0.025) and 12.5% random-patch control. Window patching shows correction requires sustained multi-step intervention, whereas corruption needs only a single perturbation. Probing the prompt encoding itself, step-0 residual states predict per-prompt hallucination rate at Pearson r = 0.776 at layer 15 (p < 0.001 against a 1000-permutation null); unsupervised clustering identifies five regime-like groups (eta^2 = 0.55) whose saddle-adjacent cluster concentrates 12 of the 13 bifurcating false-premise prompts, indicating that the basin structure is organized around regime commitments fixed at prompt encoding. These findings characterize hallucination as a locally stable attractor basin: entry is probabilistic and rapid, exit demands coordinated intervention across layers and steps, and the relevant basins are selected by clusterable regimes already discernible at step 0.
>
---
#### [new 075] Acoustic and Facial Markers of Perceived Conversational Success in Spontaneous Speech
- **分类: cs.HC; cs.CL; cs.LG**

- **简介: 该论文属于对话分析任务，旨在探究自发对话中感知交流成功的关键因素。通过分析语音和面部特征，识别影响交流质量的互动标记。**

- **链接: [https://arxiv.org/pdf/2604.15322](https://arxiv.org/pdf/2604.15322)**

> **作者:** Thanushi Withanage; Elizabeth Redcay; Carol Espy-Wilson
>
> **备注:** Accepted for presentation at ICASSP 2026
>
> **摘要:** Individuals often align their speaking patterns with their interlocutors, a phenomenon linked to engagement and rapport. While well documented in task-oriented dialogues, less is known about entrainment in naturalistic, non-task and virtual settings. In this study, we analyze a large corpus of spontaneous dyadic Zoom conversations to examine how conversational dynamics relate to perceived interaction quality. We extract multimodal features encompassing turn-taking, pauses, facial movements, and acoustic measures such as pitch and intensity. Perceived conversational success was quantified via factor analysis of post-conversation ratings. Results demonstrate that entrainment reliably detected in spontaneous speech and correlates with higher perceived success. These findings identify key interactional markers of conversational quality and highlight opportunities for targeted interventions to foster more effective and engaging communication.
>
---
#### [new 076] Detecting and Suppressing Reward Hacking with Gradient Fingerprints
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决奖励黑客问题。通过梯度指纹方法检测模型在推理过程中是否利用奖励函数漏洞，提升任务真实性。**

- **链接: [https://arxiv.org/pdf/2604.16242](https://arxiv.org/pdf/2604.16242)**

> **作者:** Songtao Wang; Quang Hieu Pham; Fangcong Yin; Xinpeng Wang; Jocelyn Qiaochu Chen; Greg Durrett; Xi Ye
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) typically optimizes for outcome rewards without imposing constraints on intermediate reasoning. This leaves training susceptible to reward hacking, where models exploit loopholes (e.g., spurious patterns in training data) in the reward function to achieve high scores without solving the intended task. These reward-hacking behaviors are often implicit, as the intermediate chain-of-thought (CoT) may appear plausible on the surface, limiting the effectiveness of purely text-based monitoring. We propose Gradient Fingerprint (GRIFT), a method for detecting reward hacking using models' internal computations. Given a prompt and a model-generated CoT, GRIFT computes gradients of the CoT conditioned on the prompt and compresses them into a compact representation, which is then used to assess whether the CoT reflects reward hacking behavior. Across verifiable reasoning benchmarks spanning math, code, and logical reasoning, GRIFT substantially outperforms strong baselines, including CoT Monitor and TRACE, achieving over 25% relative improvement in detecting reward hacking behavior. Moreover, integrating GRIFT into the rejection fine-tuning pipeline for reasoning tasks reduces reward hacking and improves performance on the true task objective. Our results highlight a promising direction of leveraging gradient level representations for assessing the quality of CoT reasoning traces. Our code is available at: this https URL.
>
---
#### [new 077] VEFX-Bench: A Holistic Benchmark for Generic Video Editing and Visual Effects
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频编辑任务，旨在解决缺乏高质量数据集和评估标准的问题。工作包括构建VEFX-Dataset和VEFX-Reward模型，以及发布VEFX-Bench基准。**

- **链接: [https://arxiv.org/pdf/2604.16272](https://arxiv.org/pdf/2604.16272)**

> **作者:** Xiangbo Gao; Sicong Jiang; Bangya Liu; Xinghao Chen; Minglai Yang; Siyuan Yang; Mingyang Wu; Jiongze Yu; Qi Zheng; Haozhi Wang; Jiayi Zhang; Jared Yang; Jie Yang; Zihan Wang; Qing Yin; Zhengzhong Tu
>
> **摘要:** As AI-assisted video creation becomes increasingly practical, instruction-guided video editing has become essential for refining generated or captured footage to meet professional requirements. Yet the field still lacks both a large-scale human-annotated dataset with complete editing examples and a standardized evaluator for comparing editing systems. Existing resources are limited by small scale, missing edited outputs, or the absence of human quality labels, while current evaluation often relies on expensive manual inspection or generic vision-language model judges that are not specialized for editing quality. We introduce VEFX-Dataset, a human-annotated dataset containing 5,049 video editing examples across 9 major editing categories and 32 subcategories, each labeled along three decoupled dimensions: Instruction Following, Rendering Quality, and Edit Exclusivity. Building on VEFX-Dataset, we propose VEFX-Reward, a reward model designed specifically for video editing quality assessment. VEFX-Reward jointly processes the source video, the editing instruction, and the edited video, and predicts per-dimension quality scores via ordinal regression. We further release VEFX-Bench, a benchmark of 300 curated video-prompt pairs for standardized comparison of editing systems. Experiments show that VEFX-Reward aligns more strongly with human judgments than generic VLM judges and prior reward models on both standard IQA/VQA metrics and group-wise preference evaluation. Using VEFX-Reward as an evaluator, we benchmark representative commercial and open-source video editing systems, revealing a persistent gap between visual plausibility, instruction following, and edit locality in current models.
>
---
#### [new 078] Aletheia: Gradient-Guided Layer Selection for Efficient LoRA Fine-Tuning Across Architectures
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出Aletheia，解决LoRA微调中全层应用效率低的问题，通过梯度引导选择关键层，提升训练速度并减少遗忘。属于模型微调优化任务。**

- **链接: [https://arxiv.org/pdf/2604.15351](https://arxiv.org/pdf/2604.15351)**

> **作者:** Abdulmalek Saket
>
> **备注:** 11 pages, 5 figures, 2 frozen evidence campaigns, 81 experiment rows across 14 successful models and 8 architecture families, plus one documented failed Pythia/GPT-NeoX attempt
>
> **摘要:** Low-Rank Adaptation (LoRA) has become the dominant parameter-efficient fine-tuning method for large language models, yet standard practice applies LoRA adapters uniformly to all transformer layers regardless of their relevance to the downstream task. We introduce Aletheia, a gradient-guided layer selection method that identifies the most task-relevant layers via a lightweight gradient probe and applies LoRA adapters only to those layers with asymmetric rank allocation. Across 81 experiment rows covering 14 successful models from 8 architecture families (0.5B-72B parameters, including dense and Mixture-of-Experts architectures), with one additional documented failed Pythia/GPT-NeoX attempt in Campaign 2, Aletheia achieves a 15-28% training speedup (mean 23.1%, p < 0.001) with bounded extra forgetting and broadly matched downstream behavior on the evaluated MMLU, GSM8K, and HumanEval benchmark pack. Across the tested families and scales, Campaign 1 shows a 100% per-model speed win rate and Campaign 2 shows broadly preserved downstream behavior within a bounded-degradation framing. Together these results support a practical model-economics claim: intelligent layer selection can make LoRA fine-tuning materially more efficient without introducing major downstream damage on the evaluated set.
>
---
#### [new 079] Pruning Unsafe Tickets: A Resource-Efficient Framework for Safer and More Robust LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型安全任务，旨在解决LLM unsafe行为问题。通过剪枝方法移除有害子网络，提升模型安全性与鲁棒性，同时保持性能。**

- **链接: [https://arxiv.org/pdf/2604.15780](https://arxiv.org/pdf/2604.15780)**

> **作者:** Wai Man Si; Mingjie Li; Michael Backes; Yang Zhang
>
> **摘要:** Machine learning models are increasingly deployed in real-world applications, but even aligned models such as Mistral and LLaVA still exhibit unsafe behaviors inherited from pre-training. Current alignment methods like SFT and RLHF primarily encourage models to generate preferred responses, but do not explicitly remove the unsafe subnetworks that trigger harmful outputs. In this work, we introduce a resource-efficient pruning framework that directly identifies and removes parameters associated with unsafe behaviors while preserving model utility. Our method employs a gradient-free attribution mechanism, requiring only modest GPU resources, and generalizes across architectures and quantized variants. Empirical evaluations on ML models show substantial reductions in unsafe generations and improved robustness against jailbreak attacks, with minimal utility loss. From the perspective of the Lottery Ticket Hypothesis, our results suggest that ML models contain "unsafe tickets" responsible for harmful behaviors, and pruning reveals "safety tickets" that maintain performance while aligning outputs. This provides a lightweight, post-hoc alignment strategy suitable for deployment in resource-constrained settings.
>
---
#### [new 080] Evaluating LLMs as Human Surrogates in Controlled Experiments
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文属于行为研究任务，旨在评估LLMs作为人类替代的适用性。通过对比LLM与人类在感知准确性的实验数据，发现LLMs能再现部分方向性效应，但效果强度不一致。**

- **链接: [https://arxiv.org/pdf/2604.15329](https://arxiv.org/pdf/2604.15329)**

> **作者:** Adnan Hoq; Tim Weninger
>
> **摘要:** Large language models (LLMs) are increasingly used to simulate human responses in behavioral research, yet it remains unclear when LLM-generated data support the same experimental inferences as human data. We evaluate this by directly comparing off-the-shelf LLM-generated responses with human responses from a canonical survey experiment on accuracy perception. Each human observation is converted into a structured prompt, and models generate a single 0--10 outcome variable without task-specific training; identical statistical analyses are applied to human and synthetic responses. We find that LLMs reproduce several directional effects observed in humans, but effect magnitudes and moderation patterns vary across models. Off-the-shelf LLMs therefore capture aggregate belief-updating patterns under controlled conditions but do not consistently match human-scale effects, clarifying when LLM-generated data can function as behavioral surrogates.
>
---
#### [new 081] A Case Study on the Impact of Anonymization Along the RAG Pipeline
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于隐私保护任务，旨在解决RAG系统中隐私泄露问题。通过实验分析匿名化在数据集和生成答案阶段的影响，探讨隐私与效用的平衡。**

- **链接: [https://arxiv.org/pdf/2604.15958](https://arxiv.org/pdf/2604.15958)**

> **作者:** Andreea-Elena Bodea; Stephen Meisenbacher; Florian Matthes
>
> **备注:** 7 pages, 1 figure, 6 tables. Accepted to IWSPA 2026
>
> **摘要:** Despite the considerable promise of Retrieval-Augmented Generation (RAG), many real-world use cases may create privacy concerns, where the purported utility of RAG-enabled insights comes at the risk of exposing private information to either the LLM or the end user requesting the response. As a potential mitigation, using anonymization techniques to remove personally identifiable information (PII) and other sensitive markers in the underlying data represents a practical and sensible course of action for RAG administrators. Despite a wealth of literature on the topic, no works consider the placement of anonymization along the RAG pipeline, i.e., asking the question, where should anonymization happen? In this case study, we systematically and empirically measure the impact of anonymization at two important points along the RAG pipeline: the dataset and generated answer. We show that differences in privacy-utility trade-offs can be observed depending on where anonymization took place, demonstrating the significance of privacy risk mitigation placement in RAG.
>
---
#### [new 082] Predicting Where Steering Vectors Succeed
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型解释任务，解决如何预测转向向量有效性的问题。提出LAP方法，通过线性分析预测转向效果，验证其在多个模型上的有效性。**

- **链接: [https://arxiv.org/pdf/2604.15557](https://arxiv.org/pdf/2604.15557)**

> **作者:** Jayadev Billa
>
> **备注:** 19 pages, incl. 10 appendix pages, 4 figures, 20 tables
>
> **摘要:** Steering vectors work for some concepts and layers but fail for others, and practitioners have no way to predict which setting applies before running an intervention. We introduce the Linear Accessibility Profile (LAP), a per-layer diagnostic that repurposes the logit lens as a predictor of steering vector effectiveness. The key measure, $A_{\mathrm{lin}}$, applies the model's unembedding matrix to intermediate hidden states, requiring no training. Across 24 controlled binary concept families on five models (Pythia-2.8B to Llama-8B), peak $A_{\mathrm{lin}}$ predicts steering effectiveness at $\rho = +0.86$ to $+0.91$ and layer selection at $\rho = +0.63$ to $+0.92$. A three-regime framework explains when difference-of-means steering works, when nonlinear methods are needed, and when no method can work. An entity-steering demo confirms the prediction end-to-end: steering at the LAP-recommended layer redirects completions on Gemma-2-2B and OLMo-2-1B-Instruct, while the middle layer (the standard heuristic) has no effect on either model.
>
---
#### [new 083] Seeing the Intangible: Survey of Image Classification into High-Level and Abstract Categories
- **分类: cs.CV; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于图像分类任务，旨在解决高层次抽象概念识别问题。通过系统综述，厘清高阶语义，分析相关任务与挑战，提出融合多源信息的解决方案。**

- **链接: [https://arxiv.org/pdf/2308.10562](https://arxiv.org/pdf/2308.10562)**

> **作者:** Delfina Sol Martinez Pandiani; Valentina Presutti
>
> **备注:** Preprint
>
> **摘要:** The field of Computer Vision (CV) is increasingly shifting towards ``high-level'' visual sensemaking tasks, yet the exact nature of these tasks remains unclear and tacit. This survey paper addresses this ambiguity by systematically reviewing research on high-level visual understanding, focusing particularly on Abstract Concepts (ACs) in automatic image classification. Our survey contributes in three main ways: Firstly, it clarifies the tacit understanding of high-level semantics in CV through a multidisciplinary analysis, and categorization into distinct clusters, including commonsense, emotional, aesthetic, and inductive interpretative semantics. Secondly, it identifies and categorizes computer vision tasks associated with high-level visual sensemaking, offering insights into the diverse research areas within this domain. Lastly, it examines how abstract concepts such as values and ideologies are handled in CV, revealing challenges and opportunities in AC-based image classification. Notably, our survey of AC image classification tasks highlights persistent challenges, such as the limited efficacy of massive datasets and the importance of integrating supplementary information and mid-level features. We emphasize the growing relevance of hybrid AI systems in addressing the multifaceted nature of AC image classification tasks. Overall, this survey enhances our understanding of high-level visual reasoning in CV and lays the groundwork for future research endeavors.
>
---
## 更新

#### [replaced 001] Whose Facts Win? LLM Source Preferences under Knowledge Conflicts
- **分类: cs.CL**

- **简介: 该论文研究LLM在知识冲突下对信息源的偏好，属于知识密集型NLP任务。旨在解决LLM如何选择可信信息源的问题，通过实验发现其偏好权威来源，并提出方法减少重复偏差。**

- **链接: [https://arxiv.org/pdf/2601.03746](https://arxiv.org/pdf/2601.03746)**

> **作者:** Jakob Schuster; Vagrant Gautam; Katja Markert
>
> **备注:** Data and code: this https URL
>
> **摘要:** As large language models (LLMs) are more frequently used in retrieval-augmented generation pipelines, it is increasingly relevant to study their behavior under knowledge conflicts. Thus far, the role of the source of the retrieved information has gone unexamined. We address this gap with a novel framework to investigate how source preferences affect LLM resolution of inter-context knowledge conflicts in English, motivated by interdisciplinary research on credibility. By using synthetic sources, we study preferences for different types of sources without inheriting the biases of specific real-world sources. With a comprehensive, tightly-controlled evaluation of 13 open-weight LLMs, we find that LLMs prefer institutionally-corroborated information (e.g., government or newspaper sources) over information from people and social media. However, these source preferences can be reversed by simply repeating information from less credible sources. To mitigate repetition effects and maintain consistent preferences, we propose a novel method that reduces repetition bias by up to 79.2%, while also maintaining at least 72.5% of original preferences. We release all data and code to encourage future work on credibility and source preferences in knowledge-intensive NLP.
>
---
#### [replaced 002] TabularMath: Understanding Math Reasoning over Tables with Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于表格数学推理任务，旨在解决大模型在表格数据上的推理能力不足问题。通过构建TabularMath基准和AutoT2T框架，评估表格复杂度、质量及模态对推理的影响。**

- **链接: [https://arxiv.org/pdf/2505.19563](https://arxiv.org/pdf/2505.19563)**

> **作者:** Shi-Yu Tian; Zhi Zhou; Wei Dong; Kun-Yang Yu; Ming Yang; Zi-Jian Cheng; Lan-Zhe Guo; Yu-Feng Li
>
> **备注:** Accepted by ACL 26
>
> **摘要:** Mathematical reasoning has long been a key benchmark for evaluating large language models. Although substantial progress has been made on math word problems, the need for reasoning over tabular data in real-world applications has been overlooked. For instance, applications such as business intelligence demand not only multi-step numerical reasoning with tables but also robustness to incomplete or inconsistent information. However, comprehensive evaluation in this area is severely limited, constrained by the reliance on manually collected tables that are difficult to scale and the lack of coverage for potential traps encountered in real-world scenarios. To address this problem, we propose AutoT2T, a neuro-symbolic framework that controllably transforms math word problems into scalable and verified tabular reasoning tasks. Building on this pipeline, we develop TabularMath, a benchmark comprising four subsets that include both text-based and image-based tables, covering table complexity, table quality, and table representation dimensions. Our study reveals three key observations: (1) Table complexity and reasoning difficulty impact reasoning performance jointly; (2) Low-quality tables pose severe risks to reliable reasoning in current LLMs; (3) Different table modalities show similar trends, with text-based tables typically being easier for models to reason over. In-depth analyses are conducted for each observation to guide future research.
>
---
#### [replaced 003] FACTS: Table Summarization via Offline Template Generation with Agentic Workflows
- **分类: cs.CL**

- **简介: 该论文属于查询聚焦的表格摘要任务，解决传统方法在效率、准确性和隐私方面的不足。提出FACTS框架，通过离线模板生成实现快速、准确且隐私合规的表格摘要。**

- **链接: [https://arxiv.org/pdf/2510.13920](https://arxiv.org/pdf/2510.13920)**

> **作者:** Ye Yuan; Mohammad Amin Shabani; Siqi Liu
>
> **备注:** Accepted by ACL 2026 Findings
>
> **摘要:** Query-focused table summarization requires generating natural language summaries of tabular data conditioned on a user query, enabling users to access insights beyond fact retrieval. Existing approaches face key limitations: table-to-text models require costly fine-tuning and struggle with complex reasoning, prompt-based LLM methods suffer from token-limit and efficiency issues while exposing sensitive data, and prior agentic pipelines often rely on decomposition, planning, or manual templates that lack robustness and scalability. To mitigate these issues, we introduce an agentic workflow, FACTS, a Fast, Accurate, and Privacy-Compliant Table Summarization approach via Offline Template Generation. FACTS produces offline templates, consisting of SQL queries and Jinja2 templates, which can be rendered into natural language summaries and are reusable across multiple tables sharing the same schema. It enables fast summarization through reusable offline templates, accurate outputs with executable SQL queries, and privacy compliance by sending only table schemas to LLMs. Evaluations on widely-used benchmarks show that FACTS consistently outperforms baseline methods, establishing it as a practical solution for real-world query-focused table summarization. Our code is available at this https URL.
>
---
#### [replaced 004] Mechanisms of Prompt-Induced Hallucination in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型研究，旨在解决提示诱导幻觉问题。通过分析模型注意力机制，发现并验证了减少幻觉的特定注意力头。**

- **链接: [https://arxiv.org/pdf/2601.05201](https://arxiv.org/pdf/2601.05201)**

> **作者:** William Rudman; Michal Golovanevsky; Dana Arad; Yonatan Belinkov; Ritambhara Singh; Carsten Eickhoff; Kyle Mahowald
>
> **备注:** ACL 2026 Main
>
> **摘要:** Large vision-language models (VLMs) are highly capable, yet often hallucinate by favoring textual prompts over visual evidence. We study this failure mode in a controlled object-counting setting, where the prompt overstates the number of objects in the image (e.g., asking a model to describe four waterlilies when only three are present). At low object counts, models often correct the overestimation, but as the number of objects increases, they increasingly conform to the prompt regardless of the discrepancy. Through mechanistic analysis of three VLMs, we identify a small set of attention heads whose ablation substantially reduces prompt-induced hallucinations (PIH) by at least 40% without additional training. Across models, PIH-heads mediate prompt copying in model-specific ways. We characterize these differences and show that PIH ablation increases correction toward visual evidence. Our findings offer insights into the internal mechanisms driving prompt-induced hallucinations, revealing model-specific differences in how these behaviors are implemented.
>
---
#### [replaced 005] Disco-RAG: Discourse-Aware Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于知识密集型任务，旨在解决RAG系统中信息结构化不足的问题。提出Disco-RAG框架，通过引入话语结构提升生成效果。**

- **链接: [https://arxiv.org/pdf/2601.04377](https://arxiv.org/pdf/2601.04377)**

> **作者:** Dongqi Liu; Hang Ding; Qiming Feng; Xurong Xie; Zhucun Xue; Chengjie Wang; Jian Li; Jiangning Zhang; Yabiao Wang
>
> **备注:** ACL 2026 Main & Long Conference Paper
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as an important means of enhancing the performance of large language models (LLMs) in knowledge-intensive tasks. However, most existing RAG strategies treat retrieved passages in a flat and unstructured way, which prevents the model from capturing structural cues and constrains its ability to synthesize knowledge from dispersed evidence across documents. To overcome these limitations, we propose Disco-RAG, a discourse-aware framework that explicitly injects discourse signals into the generation process. Our method constructs intra-chunk discourse trees to capture local hierarchies and builds inter-chunk rhetorical graphs to model cross-passage coherence. These structures are jointly integrated into a planning blueprint that conditions the generation. Experiments on question answering and long-document summarization benchmarks show the efficacy of our approach. Disco-RAG achieves state-of-the-art results on the benchmarks without fine-tuning. These findings underscore the important role of discourse structure in advancing RAG systems.
>
---
#### [replaced 006] Multi-View Attention Multiple-Instance Learning Enhanced by LLM Reasoning for Cognitive Distortion Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于心理健康自然语言处理任务，旨在解决认知扭曲自动检测中的语境模糊问题。通过结合大语言模型与多实例学习，提升检测的准确性和解释性。**

- **链接: [https://arxiv.org/pdf/2509.17292](https://arxiv.org/pdf/2509.17292)**

> **作者:** Jun Seo Kim; Hyemi Kim; Woo Joo Oh; Hongjin Cho; Hochul Lee; Hye Hyeon Kim
>
> **备注:** Accepted to the main conference of ACL 2026
>
> **摘要:** Cognitive distortions have been closely linked to mental health disorders, yet their automatic detection remains challenging due to contextual ambiguity, co-occurrence, and semantic overlap. We propose a novel framework that combines Large Language Models (LLMs) with a Multiple-Instance Learning (MIL) architecture to enhance interpretability and expression-level reasoning. Each utterance is decomposed into Emotion, Logic, and Behavior (ELB) components, which are processed by LLMs to infer multiple distortion instances, each with a predicted type, expression, and model-assigned salience score. These instances are integrated via a Multi-View Gated Attention mechanism for final classification. Experiments on Korean (KoACD) and English (Therapist QA) datasets demonstrate that incorporating ELB and LLM-inferred salience scores improves classification performance, especially for distortions with high interpretive ambiguity. Our results suggest a psychologically grounded and generalizable approach for fine-grained reasoning in mental health NLP. The dataset and implementation details are publicly accessible.
>
---
#### [replaced 007] HumanLLM: Benchmarking and Improving LLM Anthropomorphism via Human Cognitive Patterns
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的人类化表现。通过分析人类认知模式，构建场景并评估模型在多模式动态中的表现，解决模型与人类行为不一致的问题。**

- **链接: [https://arxiv.org/pdf/2601.10198](https://arxiv.org/pdf/2601.10198)**

> **作者:** Xintao Wang; Jian Yang; Weiyuan Li; Rui Xie; Jen-tse Huang; Jun Gao; Shuai Huang; Yueping Kang; Yuanli Gou; Hongwei Feng; Yanghua Xiao
>
> **备注:** Accepted to ACL 2026 Main Conference
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities in reasoning and generation, serving as the foundation for advanced persona simulation and Role-Playing Language Agents (RPLAs). However, achieving authentic alignment with human cognitive and behavioral patterns remains a critical challenge for these agents. We present HumanLLM, a framework treating psychological patterns as interacting causal forces. We construct 244 patterns from $\sim$12,000 academic papers and synthesize 11,359 scenarios where 2-5 patterns reinforce, conflict, or modulate each other, with multi-turn conversations expressing inner thoughts, actions, and dialogue. Our dual-level checklists evaluate both individual pattern fidelity and emergent multi-pattern dynamics, achieving strong human alignment ($r=0.90$) while revealing that holistic metrics conflate simulation accuracy with social desirability. HumanLLM-8B outperforms Qwen3-32B on multi-pattern dynamics despite 4$\times$ fewer parameters, demonstrating that authentic anthropomorphism requires cognitive modeling -- simulating not just what humans do, but the psychological processes generating those behaviors. Our dataset, code, and model are available at:this https URL
>
---
#### [replaced 008] OjaKV: Context-Aware Online Low-Rank KV Cache Compression
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理中的长文本生成任务，旨在解决大模型推理时的KV缓存内存瓶颈问题。通过在线低秩压缩方法OjaKV，提升内存效率并保持推理精度。**

- **链接: [https://arxiv.org/pdf/2509.21623](https://arxiv.org/pdf/2509.21623)**

> **作者:** Yuxuan Zhu; David H. Yang; Mohammad Mohammadi Amiri; Keerthiram Murugesan; Tejaswini Pedapati; Pin-Yu Chen
>
> **摘要:** The expanding long-context capabilities of large language models are constrained by a significant memory bottleneck: the key-value (KV) cache required for autoregressive generation. This bottleneck is substantial; for instance, a Llama-3.1-8B model processing a 32K-token prompt at a batch size of 4 requires approximately 16GB for its KV cache, a size exceeding the model's weights. While KV-cache compression via low-rank projection is a promising direction, existing methods rely on a static, offline-learned subspace that performs poorly under data distribution shifts. To overcome these limitations, we introduce OjaKV, a novel framework that integrates a strategic hybrid storage policy with online subspace adaptation. First, OjaKV recognizes that not all tokens are equally important for compression; it preserves the crucial first and most recent tokens in full-rank, maintaining high-fidelity anchors for attention. Second, for the vast majority of intermediate tokens, it applies low-rank compression by incrementally adapting the projection basis using Oja's algorithm for online principal component analysis. This adaptation involves a comprehensive update during prompt prefilling and lightweight periodic updates during decoding, ensuring the subspace remains aligned with the evolving context. Crucially, our framework is fully compatible with modern attention modules like FlashAttention. Experiments demonstrate that OjaKV maintains or even improves zero-shot accuracy at high compression ratios. In particular, OjaKV achieves its strongest gains on very long-context benchmarks that require complex reasoning, highlighting the importance of online subspace adaptation in dynamically tracking context shifts. These results establish our hybrid framework as a practical, plug-and-play solution for memory-efficient long-context inference without requiring model fine-tuning.
>
---
#### [replaced 009] MoshiRAG: Asynchronous Knowledge Retrieval for Full-Duplex Speech Language Models
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出MoshiRAG，解决全双工语音语言模型的事实性问题。通过异步检索增强知识获取，提升准确性同时保持交互性。**

- **链接: [https://arxiv.org/pdf/2604.12928](https://arxiv.org/pdf/2604.12928)**

> **作者:** Chung-Ming Chien; Manu Orsini; Eugene Kharitonov; Neil Zeghidour; Karen Livescu; Alexandre Défossez
>
> **摘要:** Speech-to-speech language models have recently emerged to enhance the naturalness of conversational AI. In particular, full-duplex models are distinguished by their real-time interactivity, including handling of pauses, interruptions, and backchannels. However, improving their factuality remains an open challenge. While scaling the model size could address this gap, it would make real-time inference prohibitively expensive. In this work, we propose MoshiRAG, a modular approach that combines a compact full-duplex interface with selective retrieval to access more powerful knowledge sources. Our asynchronous framework enables the model to identify knowledge-demanding queries and ground its responses in external information. By leveraging the natural temporal gap between response onset and the delivery of core information, the retrieval process can be completed while maintaining a natural conversation flow. With this approach, MoshiRAG achieves factuality comparable to the best publicly released non-duplex speech language models while preserving the interactivity inherent to full-duplex systems. Moreover, our flexible design supports plug-and-play retrieval methods without retraining and demonstrates strong performance on out-of-domain mathematical reasoning tasks.
>
---
#### [replaced 010] WildFeedback: Aligning LLMs With In-situ User Interactions And Feedback
- **分类: cs.CL**

- **简介: 该论文属于LLM对齐任务，旨在解决传统方法在资源消耗、主观性和偏差上的问题。工作是提出WildFeedback框架，利用用户实时反馈自动构建偏好数据集。**

- **链接: [https://arxiv.org/pdf/2408.15549](https://arxiv.org/pdf/2408.15549)**

> **作者:** Taiwei Shi; Zhuoer Wang; Longqi Yang; Ying-Chun Lin; Zexue He; Mengting Wan; Pei Zhou; Sujay Jauhar; Sihao Chen; Shan Xia; Hongfei Zhang; Jieyu Zhao; Xiaofeng Xu; Xia Song; Jennifer Neville
>
> **备注:** ACL 2026 Camera-ready. 25 pages, 6 figures, 9 tables
>
> **摘要:** As large language models (LLMs) continue to advance, aligning these models with human preferences has emerged as a critical challenge. Traditional alignment methods, relying on human or LLM annotated datasets, are limited by their resource-intensive nature, inherent subjectivity, misalignment with real-world user preferences, and the risk of feedback loops that amplify model biases. To overcome these limitations, we introduce WildFeedback, a novel framework that leverages in-situ user feedback during conversations with LLMs to create preference datasets automatically. Given a corpus of multi-turn user-LLM conversation, WildFeedback identifies and classifies user feedback to LLM responses between conversation turns. The user feedback is then used to create examples of preferred and dispreferred responses according to users' preference. Our experiments demonstrate that LLMs fine-tuned on WildFeedback dataset exhibit significantly improved alignment with user preferences, as evidenced by both traditional benchmarks and our proposed checklist-guided evaluation. By incorporating in-situ feedback from actual users, WildFeedback addresses the scalability, subjectivity, and bias challenges that plague existing approaches, marking a significant step toward developing LLMs that are more responsive to the diverse and evolving needs of their users.
>
---
#### [replaced 011] ConFu: Contemplate the Future for Better Speculative Sampling
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型推理加速任务，旨在解决草案模型预测偏差问题。提出ConFu框架，通过未来感知机制提升生成效率与准确性。**

- **链接: [https://arxiv.org/pdf/2603.08899](https://arxiv.org/pdf/2603.08899)**

> **作者:** Zongyue Qin; Raghavv Goel; Mukul Gagrani; Risheek Garrepalli; Mingu Lee; Yizhou Sun
>
> **备注:** V2: (i) added results on Qwen3-4B (ConFu shows ~20% improvement), (ii) added ablation on draft-length used during training (ConFu is robust to this), (iii) Added comparison table on peak memory usage for ConFu vs Eagle3. V1: accepted at ICLR 2026 workshop on Latent & Implicit Thinking - Going Beyond CoT Reasoning
>
> **摘要:** Speculative decoding has emerged as a powerful approach to accelerate large language model (LLM) inference by employing lightweight draft models to propose candidate tokens that are subsequently verified by the target model. The effectiveness of this paradigm critically depends on the quality of the draft model. While recent advances such as the EAGLE series achieve state-of-the-art speedup, existing draft models remain limited by error accumulation: they condition only on the current prefix, causing their predictions to drift from the target model over steps. In this work, we propose \textbf{ConFu} (Contemplate the Future), a novel speculative decoding framework that enables draft models to anticipate the future direction of generation. ConFu introduces (i) contemplate tokens and soft prompts that allow the draft model to leverage future-oriented signals from the target model at negligible cost, (ii) a dynamic contemplate token mechanism with MoE to enable context-aware future prediction, and (iii) a training framework with anchor token sampling and future prediction replication that learns robust future prediction. ConFu improves token acceptance rates and generation speed over EAGLE-3 by 8--11\% on Llama-3 3B/8B and by approximately 20\% on Qwen-3 4B across downstream tasks. We believe our work is the first to bridge speculative decoding with continuous reasoning tokens, offering a new direction for accelerating LLM inference.
>
---
#### [replaced 012] AI-assisted Protocol Information Extraction For Improved Accuracy and Efficiency in Clinical Trial Workflows
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于临床试验协议信息提取任务，旨在提升流程效率与准确性。通过AI-RAG系统对比传统LLM，验证其在提取精度和操作效率上的优势。**

- **链接: [https://arxiv.org/pdf/2602.00052](https://arxiv.org/pdf/2602.00052)**

> **作者:** Ramtin Babaeipour; François Charest; Madison Wright
>
> **备注:** Updated to accepted manuscript. Published in Journal of Biomedical Informatics, Volume 179, July 2026, 105036
>
> **摘要:** Increasing clinical trial protocol complexity, amendments, and challenges around knowledge management create significant burden for trial teams. Structuring protocol content into standard formats has the potential to improve efficiency, support documentation quality, and strengthen compliance. We evaluate an Artificial Intelligence (AI) system using generative LLMs with Retrieval-Augmented Generation (RAG) for automated clinical trial protocol information extraction. We compare the extraction accuracy of our clinical-trial-specific RAG process against that of publicly available (standalone) LLMs. We also assess the operational impact of AI-assistance on simulated extraction Clinical Research Coordinator (CRC) workflows. Our RAG process shows higher extraction accuracy (89.0%) than standalone LLMs with fine-tuned prompts (62.6%) against expert-supported reference annotations. In simulated extraction workflows, AI-assisted tasks are completed 40% faster, are rated as less cognitively demanding and are strongly preferred by users. While expert oversight remains essential, this suggests that AI-assisted extraction can enable protocol intelligence at scale, motivating the integration of similar methodologies into real-world clinical workflows to further validate its impact on feasibility, study start-up, and post-activation monitoring.
>
---
#### [replaced 013] Revisiting the Uniform Information Density Hypothesis in LLM Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，探讨LLM推理中信息密度的均匀性问题。研究提出新框架量化信息流的局部与全局均匀性，发现高质量推理具有局部均匀和全局非均匀特征，验证其作为推理质量预测指标的有效性。**

- **链接: [https://arxiv.org/pdf/2510.06953](https://arxiv.org/pdf/2510.06953)**

> **作者:** Minju Gwak; Guijin Son; Jaehyung Kim
>
> **备注:** ACL 2026 Findings
>
> **摘要:** The Uniform Information Density (UID) hypothesis proposes that effective communication is achieved by maintaining a stable flow of information. In this work, we revisit this principle in the context of Large Language Model (LLM) reasoning, asking whether step-level uniformity reflects reasoning quality. To this end, we introduce a novel framework to quantify uniformity of information flow at both local and global levels, using an entropy-based stepwise density metric. Across experiments on seven reasoning benchmarks, we see a counter-intuitive pattern: while high-quality reasoning exhibit smooth step-by-step transitions local uniformity and structured, non-uniform information flow at the trajectory level global non-uniformity. The results demonstrate that these uniformities outperform alternative internal signals as predictors of reasoning quality, and such divergence with human communication is not a model deficiency, but a byproduct of distinct objectives between human communication and LLM reasoning.
>
---
#### [replaced 014] Measuring the Semantic Structure and Evolution of Conspiracy Theories
- **分类: cs.CL; cs.CY; cs.SI**

- **简介: 该论文属于语义分析任务，旨在解决 conspiracy theories 语义演变的测量问题。通过分析 Reddit 数据，揭示其语义结构及随时间的变化模式。**

- **链接: [https://arxiv.org/pdf/2603.26062](https://arxiv.org/pdf/2603.26062)**

> **作者:** Manisha Keim; Sarmad Chandio; Osama Khalid; Rishab Nithyanand
>
> **摘要:** Research on conspiracy theories has largely focused on belief formation, exposure, and diffusion, while paying less attention to how their meanings change over time. This gap persists partly because conspiracy-related terms are often treated as stable lexical markers, making it difficult to separate genuine semantic changes from surface-level vocabulary changes. In this paper, we measure the semantic structure and evolution of conspiracy theories in online political discourse. Using 169.9M comments from Reddit's r/politics subreddit spanning 2012--2022, we first demonstrate that conspiracy-related language forms coherent and semantically distinguishable regions of language space, allowing conspiracy theories to be treated as semantic objects. We then track how these objects evolve over time using aligned word embeddings, enabling comparisons of semantic neighborhoods across periods. Our analysis reveals that conspiracy theories evolve non-uniformly, exhibiting patterns of semantic stability, expansion, contraction, and replacement that are not captured by keyword-based approaches alone.
>
---
#### [replaced 015] Preconditioned Test-Time Adaptation for Out-of-Distribution Debiasing in Narrative Generation
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言生成任务，旨在解决模型在分布外高偏见提示下的性能下降问题。提出CAP-TTA框架，通过测试时自适应实现快速、稳定去偏。**

- **链接: [https://arxiv.org/pdf/2603.13683](https://arxiv.org/pdf/2603.13683)**

> **作者:** Hanwen Shen; Ting Ying; Jiajie Lu; Shanshan Wang
>
> **备注:** This paper has been accepted to ACL2026 main conference
>
> **摘要:** Although debiased large language models (LLMs) excel at handling known or low-bias prompts, they often fail on unfamiliar and high-bias prompts. We demonstrate via out-of-distribution (OOD) detection that these high-bias prompts cause a distribution shift, degrading static model performance. To enable real-time correction, we propose CAP-TTA, a test-time adaptation framework. CAP-TTA triggers context-aware LoRA updates only when a bias-risk score exceeds a set threshold. By utilizing an offline precomputed diagonal preconditioner, it ensures fast and stable optimization. Across multiple benchmarks and human evaluations, CAP-TTA effectively reduces toxicity/bias score with significantly lower latency than standard optimization methods (e.g., AdamW or SGD). Furthermore, it prevents catastrophic forgetting, and substantially improves narrative fluency over state-of-the-art baselines without compromising debiasing performance.
>
---
#### [replaced 016] LaMSUM: Amplifying Voices Against Harassment through LLM Guided Extractive Summarization of User Incident Reports
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于文本摘要任务，旨在解决大规模用户举报信息处理难题。通过LaMSUM框架，利用LLM生成提取式摘要，提升信息处理效率。**

- **链接: [https://arxiv.org/pdf/2406.15809](https://arxiv.org/pdf/2406.15809)**

> **作者:** Garima Chhikara; Anurag Sharma; V. Gurucharan; Kripabandhu Ghosh; Abhijnan Chakraborty
>
> **备注:** Accepted at ICWSM 2026
>
> **摘要:** Citizen reporting platforms help the public and authorities stay informed about sexual harassment incidents. However, the high volume of data shared on these platforms makes reviewing each individual case challenging. Therefore, a summarization algorithm capable of processing and understanding various code-mixed languages is essential. In recent years, Large Language Models (LLMs) have shown exceptional performance in NLP tasks, including summarization. LLMs inherently produce abstractive summaries by paraphrasing the original text, while the generation of extractive summaries - selecting specific subsets from the original text - through LLMs remains largely unexplored. Moreover, LLMs have a limited context window size, restricting the amount of data that can be processed at once. We tackle these challenges by introducing LaMSUM, a novel multi-level framework combining summarization with different voting methods to generate extractive summaries for large collections of incident reports using LLMs. Extensive evaluation using four popular LLMs (Llama, Mistral, Claude and GPT-4o) demonstrates that LaMSUM outperforms state-of-the-art extractive summarization methods. Overall, this work represents one of the first attempts to achieve extractive summarization through LLMs, and is likely to support stakeholders by offering a comprehensive overview and enabling them to develop effective policies to minimize incidents of unwarranted harassment.
>
---
#### [replaced 017] WiseMind: a knowledge-guided multi-agent framework for accurate and empathetic psychiatric diagnosis
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出WiseMind框架，用于精神科诊断任务，解决LLM在临床推理和情感沟通上的不足，通过多代理结构提升诊断准确性和共情能力。**

- **链接: [https://arxiv.org/pdf/2502.20689](https://arxiv.org/pdf/2502.20689)**

> **作者:** Yuqi Wu; Guangya Wan; Jingjing Li; Shengming Zhao; Lingfeng Ma; Tianyi Ye; Ion Pop; Yanbo Zhang; Jie Chen
>
> **备注:** Accepted at npj Digital Medicine (2026)
>
> **摘要:** Large Language Models (LLMs) offer promising opportunities to support mental healthcare workflows, yet they often lack the structured clinical reasoning needed for reliable diagnosis and may struggle to provide the emotionally attuned communication essential for patient trust. Here, we introduce WiseMind, a novel multi-agent framework inspired by the theory of Dialectical Behavior Therapy designed to facilitate psychiatric assessment. By integrating a "Reasonable Mind" Agent for evidence-based logic and an "Emotional Mind" Agent for empathetic communication, WiseMind effectively bridges the gap between instrumental accuracy and humanistic care. Our framework utilizes a Diagnostic and Statistical Manual of Mental Disorders, Fifth Edition (DSM-5)-guided Structured Knowledge Graph to steer diagnostic inquiries, significantly reducing hallucinations compared to standard prompting methods. Using a combination of virtual standard patients, simulated interactions, and real human interaction datasets, we evaluate WiseMind across three common psychiatric conditions. WiseMind outperforms state-of-the-art LLM methods in both identifying critical diagnostic nodes and establishing accurate differential diagnoses. Across 1206 simulated conversations and 180 real user sessions, the system achieves 85.6% top-1 diagnostic accuracy, approaching reported diagnostic performance ranges of board-certified psychiatrists and surpassing knowledge-enhanced single-agent baselines by 15-54 percentage points. Expert review by psychiatrists further validates that WiseMind generates responses that are not only clinically sound but also psychologically supportive, demonstrating the feasibility of empathetic, reliable AI agents to conduct psychiatric assessments under appropriate human oversight.
>
---
#### [replaced 018] RedBench: A Universal Dataset for Comprehensive Red Teaming of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于安全评估任务，旨在解决LLM对抗性攻击检测问题。提出RedBench数据集，整合37个基准数据，涵盖29,362条攻击与拒绝提示，提供标准化风险分类和领域覆盖，以提升LLM安全性评估。**

- **链接: [https://arxiv.org/pdf/2601.03699](https://arxiv.org/pdf/2601.03699)**

> **作者:** Quy-Anh Dang; Chris Ngo; Truong-Son Hy
>
> **摘要:** As large language models (LLMs) become integral to safety-critical applications, ensuring their robustness against adversarial prompts is paramount. However, existing red teaming datasets suffer from inconsistent risk categorizations, limited domain coverage, and outdated evaluations, hindering systematic vulnerability assessments. To address these challenges, we introduce RedBench, a universal dataset aggregating 37 benchmark datasets from leading conferences and repositories, comprising 29,362 samples across attack and refusal prompts. RedBench employs a standardized taxonomy with 22 risk categories and 19 domains, enabling consistent and comprehensive evaluations of LLM vulnerabilities. We provide a detailed analysis of existing datasets, establish baselines for modern LLMs, and open-source the dataset and evaluation code. Our contributions facilitate robust comparisons, foster future research, and promote the development of secure and reliable LLMs for real-world deployment. Code: this https URL
>
---
#### [replaced 019] Olmo Hybrid: From Theory to Practice and Back
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决传统Transformer模型的局限性。通过构建混合模型Olmo Hybrid，结合注意力与循环结构，提升模型表达能力和训练效率。**

- **链接: [https://arxiv.org/pdf/2604.03444](https://arxiv.org/pdf/2604.03444)**

> **作者:** William Merrill; Yanhong Li; Tyler Romero; Anej Svete; Caia Costello; Pradeep Dasigi; Dirk Groeneveld; David Heineman; Bailey Kuehl; Nathan Lambert; Chuan Li; Kyle Lo; Saumya Malik; DJ Matusz; Benjamin Minixhofer; Jacob Morrison; Luca Soldaini; Finbarr Timbers; Pete Walsh; Noah A. Smith; Hannaneh Hajishirzi; Ashish Sabharwal
>
> **备注:** Corrected author list
>
> **摘要:** Recent work has demonstrated the potential of non-transformer language models, especially linear recurrent neural networks (RNNs) and hybrid models that mix recurrence and attention. Yet there is no consensus on whether the potential benefits of these new architectures justify the risk and effort of scaling them up. To address this, we provide evidence for the advantages of hybrid models over pure transformers on several fronts. First, theoretically, we show that hybrid models do not merely inherit the expressivity of transformers and linear RNNs, but can express tasks beyond both, such as code execution. Putting this theory to practice, we train Olmo Hybrid, a 7B-parameter model largely comparable to Olmo 3 7B but with the sliding window layers replaced by Gated DeltaNet layers. We show that Olmo Hybrid outperforms Olmo 3 across standard pretraining and mid-training evaluations, demonstrating the benefit of hybrid models in a controlled, large-scale setting. We find that the hybrid model scales significantly more efficiently than the transformer, explaining its higher performance. However, its unclear why greater expressivity on specific formal problems should result in better scaling or superior performance on downstream tasks unrelated to those problems. To explain this apparent gap, we return to theory and argue why increased expressivity should translate to better scaling efficiency, completing the loop. Overall, our results suggest that hybrid models mixing attention and recurrent layers are a powerful extension to the language modeling paradigm: not merely to reduce memory during inference, but as a fundamental way to obtain more expressive models that scale better during pretraining.
>
---
#### [replaced 020] FS-Researcher: Test-Time Scaling for Long-Horizon Research Tasks with File-System-Based Agents
- **分类: cs.CL**

- **简介: 该论文提出FS-Researcher，解决长周期研究任务中模型上下文限制问题。通过文件系统构建持久工作空间，实现有效测试时扩展，提升报告质量。**

- **链接: [https://arxiv.org/pdf/2602.01566](https://arxiv.org/pdf/2602.01566)**

> **作者:** Chiwei Zhu; Benfeng Xu; Mingxuan Du; Shaohan Wang; Xiaorui Wang; Zhendong Mao; Yongdong Zhang
>
> **备注:** 22 pages, 6 figures; Accepted to ACL 2026
>
> **摘要:** Deep research is emerging as a representative long-horizon task for large language model (LLM) agents. However, long trajectories in deep research often exceed model context limits, compressing token budgets for both evidence collection and report writing, and preventing effective test-time scaling. We introduce FS-Researcher, a file-system-based, dual-agent framework that scales deep research beyond the context window via a persistent workspace. Specifically, a Context Builder agent acts as a librarian which browses the internet, writes structured notes, and archives raw sources into a hierarchical knowledge base that can grow far beyond context length. A Report Writer agent then composes the final report section by section, treating the knowledge base as the source of facts. In this framework, the file system serves as a durable external memory and a shared coordination medium across agents and sessions, enabling iterative refinement beyond the context window. Experiments on two open-ended benchmarks (DeepResearch Bench and DeepConsult) show that FS-Researcher achieves state-of-the-art report quality across different backbone models. Further analyses demonstrate a positive correlation between final report quality and the computation allocated to the Context Builder, validating effective test-time scaling under the file-system paradigm. The code and data are open-sourced at this https URL.
>
---
#### [replaced 021] Large Language Models for Math Education in Low-Resource Languages: A Study in Sinhala and Tamil
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究LLM在低资源语言数学教育中的应用。解决非英语语言下模型可靠性问题，通过构建平行数据集评估不同模型的数学推理能力。**

- **链接: [https://arxiv.org/pdf/2602.14517](https://arxiv.org/pdf/2602.14517)**

> **作者:** Sukumar Kishanthan; Kumar Thushalika; Buddhi Jayasekara; Asela Hevapathige
>
> **备注:** Accepted to ITHET 2026
>
> **摘要:** Large language models (LLMs) have achieved strong results in mathematical reasoning, and are increasingly deployed as tutoring and learning support tools in educational settings. However, their reliability for students working in non-English languages, especially low-resource languages, remains poorly understood. We examine this gap by evaluating mathematical reasoning in Sinhala and Tamil -- two languages widely used in South Asian schools but underrepresented in artificial intelligence (AI) research. Using a taxonomy of six math problem types, from basic arithmetic to complex unit conflict and optimization problems, we evaluate four prominent large language models. To avoid translation artifacts that confound language ability with translation quality, we construct a parallel dataset in which each problem is independently authored in Sinhala and Tamil by native speakers, and in English by fluent speakers, all with strong mathematical backgrounds. Our analysis demonstrates that while basic arithmetic reasoning transfers robustly across languages, complex reasoning tasks show significant degradation in Tamil and Sinhala. The pattern of failures varies by model and problem type, suggesting that strong performance in English does not guarantee reliable performance across languages. These findings have direct implications for the deployment of AI tools in multilingual classrooms, and highlight the need for language-specific evaluation before adopting large language models as math tutoring aids in non-English educational contexts.
>
---
#### [replaced 022] Interpretable Traces, Unexpected Outcomes: Investigating the Disconnect in Trace-Based Knowledge Distillation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识蒸馏任务，探讨了基于痕迹的模型训练中痕迹语义与可解释性之间的脱节问题。通过实验验证痕迹正确性与模型性能、用户理解之间的关系。**

- **链接: [https://arxiv.org/pdf/2505.13792](https://arxiv.org/pdf/2505.13792)**

> **作者:** Siddhant Bhambri; Upasana Biswas; Subbarao Kambhampati
>
> **备注:** Accepted at The 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026)
>
> **摘要:** Recent advances in reasoning-focused Large Language Models (LLMs) have introduced Chain-of-Thought (CoT) traces - intermediate reasoning steps generated before a final answer. These traces, as in DeepSeek R1, guide inference and train smaller models. A common but under-examined assumption is that these traces are both semantically correct and interpretable to end-users. While intermediate reasoning steps are believed to improve accuracy, we question whether they are actually valid and understandable. To isolate the effect of trace semantics, we design experiments in Question Answering (QA) using rule-based problem decomposition, creating fine-tuning datasets where each problem is paired with either verifiably correct or incorrect traces, while always providing the correct final answer. Trace correctness is evaluated by checking the accuracy of every reasoning sub-step. To assess interpretability, we fine-tune LLMs on three additional trace types: R1 traces, R1 trace summaries, and post-hoc explanations, and conduct a human study with 100 participants rating each type on a Likert scale. We find: (1) Trace correctness does not reliably predict correct final answers - correct traces led to correct solutions in only 28% of test cases, while incorrect traces did not consistently degrade accuracy. (2) Fine-tuning on verbose R1 traces yielded the best model performance, but users rated them least interpretable (3.39 interpretability, 4.59 cognitive load on a 5-point scale), whereas more interpretable decomposed traces did not achieve comparable accuracy. Together, these findings challenge the assumption in question suggesting that researchers and practitioners should decouple model supervision objectives from end-user-facing trace design.
>
---
#### [replaced 023] KMMMU: Evaluation of Massive Multi-discipline Multimodal Understanding in Korean Language and Context
- **分类: cs.CL; cs.LG; cs.MM**

- **简介: 该论文提出KMMMU，一个针对韩语多模态理解的基准测试，解决跨学科、多模态任务中的语言与文化适应性问题。工作包括构建数据集并评估模型表现。**

- **链接: [https://arxiv.org/pdf/2604.13058](https://arxiv.org/pdf/2604.13058)**

> **作者:** Nahyun Lee; Guijin Son; Hyunwoo Ko; Chanyoung Kim; JunYoung An; Kyubeen Han; Il-Youp Kwak
>
> **备注:** 8 pages
>
> **摘要:** We introduce KMMMU, a native Korean benchmark for evaluating multimodal understanding in Korean cultural and institutional settings. KMMMU contains 3,466 questions from exams natively written in Korean, covering nine disciplines and nine visual modality categories, along with a 300-item Korean-specific subset and a hard subset of 627 questions. Unlike translated or English-centric benchmarks, KMMMU targets information-dense problems shaped by local conventions, official standards, and discipline-specific visual formats. Experiments show that the strongest open-source model reaches only 42.05% accuracy on the full set, while the best proprietary model achieves 52.42% on the hard subset. Performance varies across disciplines, with some disciplines emerging as bottlenecks, and Korean-specific questions showing gaps of up to 13.43%. Error analysis suggests that these failures stem less from insufficient reasoning depth than from weak convention-to-label mapping, few-shot symbolic induction, localized knowledge recall, and domain-specific standards understanding. KMMMU provides a testbed for multimodal evaluation beyond English-centric benchmarks and for developing more reliable systems for expert real-world tasks.
>
---
#### [replaced 024] Dynamic Sampling that Adapts: Self-Aware Iterative Data Persistent Optimization for Mathematical Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于数学推理任务，解决模型训练中数据选择不适应模型能力变化的问题。提出SAI-DPO框架，通过动态调整数据分布提升训练效率。**

- **链接: [https://arxiv.org/pdf/2505.16176](https://arxiv.org/pdf/2505.16176)**

> **作者:** Jun Rao; Xuebo Liu; Hexuan Deng; Zepeng Lin; Zixiong Yu; Jiansheng Wei; Xiaojun Meng; Min Zhang
>
> **备注:** ACL2026 Findings
>
> **摘要:** In mathematical reasoning, data selection strategies predominantly rely on static, externally defined metrics, which fail to adapt to the evolving capabilities of models during training. This misalignment limits the efficiency of Supervised Fine-Tuning and Reinforcement Learning. To bridge this gap, we introduce SAI-DPO (Self-Aware Iterative Data Persistent Optimization), a dynamic sampling framework that aligns training data with the model's intrinsic competence. SAI-DPO operationalizes two novel metrics: Knowledge Semantic Alignment for targeting domain weaknesses, and Self-Aware Difficulty, derived from pass rates and reasoning path characteristics, to gauge instance complexity relative to the model's current state. By iteratively recalibrating the data distribution based on real-time feedback, SAI-DPO dynamically aligns training samples with the model's evolving competence, ensuring the data remains strictly relevant to the model's current capability level. Extensive experiments on eight benchmarks (including AIME24 and AMC23) demonstrate that SAI-DPO outperforms static baselines at most nearly 6 points, achieving state-of-the-art efficiency with significantly less data.
>
---
#### [replaced 025] TPA: Next Token Probability Attribution for Detecting Hallucinations in RAG
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于检测RAG系统中幻觉的任务，旨在解决现有方法未能全面考虑模型各组件影响的问题。提出TPA方法，通过概率归因分析不同来源对生成token的贡献，有效识别幻觉响应。**

- **链接: [https://arxiv.org/pdf/2512.07515](https://arxiv.org/pdf/2512.07515)**

> **作者:** Pengqian Lu; Jie Lu; Anjin Liu; Guangquan Zhang
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** Detecting hallucinations in Retrieval-Augmented Generation remains a challenge. Prior approaches attribute hallucinations to a binary conflict between internal knowledge stored in FFNs and the retrieved context. However, this perspective is incomplete, failing to account for the impact of other components of the LLM, such as the user query, previously generated tokens, the self token, and the final LayerNorm adjustment. To comprehensively capture the impact of these components on hallucination detection, we propose TPA which mathematically attributes each token's probability to seven distinct sources: Query, RAG Context, Past Token, Self Token, FFN, Final LayerNorm, and Initial Embedding. This attribution quantifies how each source contributes to the generation of the next token. Specifically, we aggregate these attribution scores by Part-of-Speech (POS) tags to quantify the contribution of each model component to the generation of specific linguistic categories within a response. By leveraging these patterns, such as detecting anomalies where Nouns rely heavily on LayerNorm, TPA effectively identifies hallucinated responses. Extensive experiments show that TPA achieves state-of-the-art performance.
>
---
#### [replaced 026] Evaluating Memory Capability in Continuous Lifelog Scenario
- **分类: cs.CL**

- **简介: 该论文属于记忆系统评估任务，旨在解决真实场景下持续生活日志的内存能力评估问题。提出新基准LifeDialBench及在线评估协议，发现现有系统不如简单基线有效。**

- **链接: [https://arxiv.org/pdf/2604.11182](https://arxiv.org/pdf/2604.11182)**

> **作者:** Jianjie Zheng; Zhichen Liu; Zhanyu Shen; Jingxiang Qu; Guanhua Chen; Yile Wang; Yang Xu; Yang Liu; Sijie Cheng
>
> **备注:** 27 pages, 7 figures. ACL 2026 Findings camera-ready
>
> **摘要:** Nowadays, wearable devices can continuously lifelog ambient conversations, creating substantial opportunities for memory systems. However, existing benchmarks primarily focus on online one-on-one chatting or human-AI interactions, thus neglecting the unique demands of real-world scenarios. Given the scarcity of public lifelogging audio datasets, we propose a hierarchical synthesis framework to curate \textbf{\textsc{LifeDialBench}}, a novel benchmark comprising two complementary subsets: \textbf{EgoMem}, built on real-world egocentric videos, and \textbf{LifeMem}, constructed using simulated virtual community. Crucially, to address the issue of temporal leakage in traditional offline settings, we propose an \textbf{Online Evaluation} protocol that strictly adheres to temporal causality, ensuring systems are evaluated in a realistic streaming fashion. Our experimental results reveal a counterintuitive finding: current sophisticated memory systems fail to outperform a simple RAG-based baseline. This highlights the detrimental impact of over-designed structures and lossy compression in current approaches, emphasizing the necessity of high-fidelity context preservation for lifelog scenarios.
>
---
#### [replaced 027] Curing Miracle Steps in LLM Mathematical Reasoning with Rubric Rewards
- **分类: cs.CL**

- **简介: 该论文属于数学推理任务，旨在解决模型因奖励黑客导致的推理能力高估问题。通过引入过程导向的Rubric奖励模型，提升推理准确性并减少错误步骤。**

- **链接: [https://arxiv.org/pdf/2510.07774](https://arxiv.org/pdf/2510.07774)**

> **作者:** Youliang Yuan; Qiuyang Mang; Jingbang Chen; Hong Wan; Xiaoyuan Liu; Junjielong Xu; Jen-tse Huang; Wenxuan Wang; Wenxiang Jiao; Pinjia He
>
> **备注:** Accepted by ACL 2026 Main, 22 pages, 10 figures, 7 Tables
>
> **摘要:** In this paper, we observe that current models are susceptible to reward hacking, leading to a substantial overestimation of a model's reasoning ability. This is evidenced by a high incidence of false positives-solutions that reach the correct answer through an unsound process. Through a systematic analysis with human verification, we establish a taxonomy of these failure modes, identifying patterns like Miracle Steps-abrupt jumps to a correct output without a valid preceding derivation. Probing experiments suggest that these Miracle Steps are linked to answer-recall shortcuts, including memorization from pretraining, where the model accesses the correct answer independently of its reasoning chain. To mitigate this systemic issue, we introduce the Rubric Reward Model (RRM), a process-oriented reward function that evaluates the entire reasoning trajectory against problem-specific rubrics. The RRM explicitly penalizes logical flaws and encourages rigorous deduction. When integrated into an RL pipeline, RRM-based training consistently outperforms outcome-only supervision across four math benchmarks. Notably, it boosts Verified Pass@1024 on AIME2024 from 26.7% to 62.6% and reduces the incidence of Miracle Steps by 71%. Our work demonstrates that rewarding the solution process is crucial for building accurate and reliable models.
>
---
#### [replaced 028] Beyond MCQ: An Open-Ended Arabic Cultural QA Benchmark with Dialect Variants
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的问答任务，旨在解决语言模型在阿拉伯语方言和文化背景知识上的表现不足问题。工作包括构建多语言问答数据集，并测试不同模型在开放和封闭问答设置下的表现。**

- **链接: [https://arxiv.org/pdf/2510.24328](https://arxiv.org/pdf/2510.24328)**

> **作者:** Hunzalah Hassan Bhatti; Firoj Alam
>
> **备注:** Cultural Knowledge, Everyday Knowledge, Open-Ended Question, Chain-of-Thought, Large Language Models, Native, Multilingual, Language Diversity
>
> **摘要:** Large Language Models (LLMs) are increasingly used to answer everyday questions, yet their performance on culturally grounded and dialectal content remains uneven across languages. We propose a comprehensive method that (i) translates Modern Standard Arabic (MSA) multiple-choice questions (MCQs) into English and several Arabic dialects, (ii) converts them into open-ended questions (OEQs), (iii) benchmarks a range of zero-shot and fine-tuned LLMs under both MCQ and OEQ settings, and (iv) generates chain-of-thought (CoT) rationales to fine-tune models for step-by-step reasoning. Using this method, we extend an existing dataset in which QAs are parallelly aligned across multiple language varieties, making it, to our knowledge, the first of its kind. We conduct extensive experiments with both open and closed models. Our findings show that (i) models underperform on Arabic dialects, revealing persistent gaps in culturally grounded and dialect-specific knowledge; (ii) Arabic-centric models perform well on MCQs but struggle with OEQs; and (iii) CoT improves judged correctness while yielding mixed n-gram-based metrics. The developed dataset will be publicly released to support further research on culturally and linguistically inclusive evaluation.
>
---
#### [replaced 029] Power to the Clients: Federated Learning in a Dictatorship Setting
- **分类: cs.LG; cs.AI; cs.CL; cs.CR; cs.CV; cs.DC**

- **简介: 该论文属于安全联邦学习任务，研究恶意客户端在集中式FL中的攻击行为，提出“独裁客户端”概念并分析其对模型收敛的影响。**

- **链接: [https://arxiv.org/pdf/2510.22149](https://arxiv.org/pdf/2510.22149)**

> **作者:** Mohammadsajad Alipour; Mohammad Mohammadi Amiri
>
> **摘要:** Federated learning (FL) has emerged as a promising paradigm for decentralized model training, enabling multiple clients to collaboratively learn a shared model without exchanging their local data. However, the decentralized nature of FL also introduces vulnerabilities, as malicious clients can compromise or manipulate the training process. In this work, we introduce dictator clients, a novel, well-defined, and analytically tractable class of malicious participants capable of entirely erasing the contributions of all other clients from the server model, while preserving their own. We propose concrete attack strategies that empower such clients and systematically analyze their effects on the learning process. Furthermore, we explore complex scenarios involving multiple dictator clients, including cases where they collaborate, act independently, or form an alliance in order to ultimately betray one another. For each of these settings, we provide a theoretical analysis of their impact on the global model's convergence. Our theoretical algorithms and findings about the complex scenarios including multiple dictator clients are further supported by empirical evaluations on both computer vision and natural language processing benchmarks.
>
---
#### [replaced 030] Beyond Static Personas: Situational Personality Steering for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于个性化语言模型任务，旨在解决静态人格建模适应性差的问题。通过分析人格神经元，提出IRIS框架实现情境化人格调整。**

- **链接: [https://arxiv.org/pdf/2604.13846](https://arxiv.org/pdf/2604.13846)**

> **作者:** Zesheng Wei; Mengxiang Li; Zilei Wang; Yang Deng
>
> **备注:** Accepted to Findings of ACL2026
>
> **摘要:** Personalized Large Language Models (LLMs) facilitate more natural, human-like interactions in human-centric applications. However, existing personalization methods are constrained by limited controllability and high resource demands. Furthermore, their reliance on static personality modeling restricts adaptability across varying situations. To address these limitations, we first demonstrate the existence of situation-dependency and consistent situation-behavior patterns within LLM personalities through a multi-perspective analysis of persona neurons. Building on these insights, we propose IRIS, a training-free, neuron-based Identify-Retrieve-Steer framework for advanced situational personality steering. Our approach comprises situational persona neuron identification, situation-aware neuron retrieval, and similarity-weighted steering. We empirically validate our framework on PersonalityBench and our newly introduced SPBench, a comprehensive situational personality benchmark. Experimental results show that our method surpasses best-performing baselines, demonstrating IRIS's generalization and robustness to complex, unseen situations and different models architecture.
>
---
#### [replaced 031] Anthropogenic Regional Adaptation in Multimodal Vision-Language Model
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于视觉语言模型任务，解决区域文化对齐问题。提出Anthropogenic Regional Adaptation框架及GG-EZ方法，提升模型在特定地区的文化相关性，同时保持全局性能。**

- **链接: [https://arxiv.org/pdf/2604.11490](https://arxiv.org/pdf/2604.11490)**

> **作者:** Samuel Cahyawijaya; Peerat Limkonchotiwat; Tack Hwa Wong; Hitesh Laxmichand Patel; Amit Agarwal; Manuel Antonio Rufino; Carlos Rafael Catalan; Muhammad Reza Qorib; Vicky Feliren; Holy Lovenia; Aye Hninn Khine; Frederikus Hudi; David Anugraha; Alham Fikri Aji; Romrawin Chumpu; Viet-Thanh Pham; Minghan Wang; Mohamed Fazli Imam; Ruochen Zhang; Joseph Marvin Imperial; Khumaisa Nur'aini; Do Xuan Long; Musa Izzanardi Wijanarko; Joel Ruben Antony Moniz; Patrick Amadeus Irawan; Hanif Muhammad Zhafran; Isaiah Flores; Salsabila Zahirah Pranida; Jun Kevin; Jostin Jerico Rosal; Patricia Nicole Monderin; Kun Kerdthaisong; Ahmad Mustafid; My Chiffon Nguyen; Natchapon Jongwiriyanurak; Siva Worajitwannakul; Haochen Li; Adrian Xuan Wei Lim; Bin Wang; Muhammad Ravi Shulthan Habibi; Lynnette Hui Xian Ng; Mithil Bangera; Yeshil Bangera; Priyaranjan Pattnayak; Dun Li Chan; Sherissa Caren Djuniwar; Cho Chan Myei Oo; Hee Ming Shan
>
> **摘要:** While the field of vision-language (VL) has achieved remarkable success in integrating visual and textual information across multiple languages and domains, there is still no dedicated framework for assessing human-centric alignment in vision-language systems. We offer two contributions to address this gap. First, we introduce Anthropogenic Regional Adaptation: a novel paradigm that aims to optimize model relevance to specific regional contexts while ensuring the retention of global generalization capabilities. Second, we present a simple, but effective adaptation method named Geographical-generalization-made-easy (GG-EZ), which utilizes regional data filtering and model merging. Through comprehensive experiments on 3 VL architectures: large vision-language models, text-to-image diffusion models, and vision-language embedding models, and a case study in Southeast Asia (SEA) regional adaptation, we demonstrate the importance of Anthropogenic Regional Adaptation and the effectiveness of GG-EZ, showing 5-15% gains in cultural relevance metrics across SEA while maintaining over 98% of global performance and even occasionally surpassing it. Our findings establish Anthropogenic Regional Alignment as a foundational paradigm towards applicability of multimodal vision-language models in diverse regions and demonstrate a simple-yet-effective baseline method that optimizes regional value alignment while preserving global generalization.
>
---
#### [replaced 032] Protecting Language Models Against Unauthorized Distillation through Trace Rewriting
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于模型保护任务，旨在防止未经授权的知识蒸馏。通过修改教师模型的推理轨迹，实现反蒸馏和API水印，提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2602.15143](https://arxiv.org/pdf/2602.15143)**

> **作者:** Xinhang Ma; William Yeoh; Ning Zhang; Yevgeniy Vorobeychik
>
> **摘要:** Knowledge distillation is a widely adopted technique for transferring capabilities from LLMs to smaller, more efficient student models. However, unauthorized use of knowledge distillation takes unfair advantage of the considerable effort and cost put into developing frontier models. We investigate methods for modifying teacher-generated reasoning traces to achieve two objectives that deter unauthorized distillation: (1) \emph{anti-distillation}, or degrading the training usefulness of query responses, and (2) \emph{API watermarking}, which embeds verifiable signatures in student models. We introduce several approaches for dynamically rewriting a teacher's reasoning outputs while preserving answer correctness and semantic coherence. Two of these leverage the rewriting capabilities of LLMs, while others use gradient-based techniques. Our experiments show that a simple instruction-based rewriting approach achieves a strong anti-distillation effect while maintaining or even improving teacher performance. Furthermore, we show that our rewriting approach also enables embedding watermarks that can be reliably detected with essentially no false alarms. Our code is available at this https URL.
>
---
#### [replaced 033] Theory of Mind in Action: The Instruction Inference Task in Dynamic Human-Agent Collaboration
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文属于人机协作任务，旨在解决指令不明确时的意图推理问题。通过设计Tomcat模型，提升代理对人类指令的理解与响应能力。**

- **链接: [https://arxiv.org/pdf/2507.02935](https://arxiv.org/pdf/2507.02935)**

> **作者:** Fardin Saad; Pradeep K. Murukannaiah; Munindar P. Singh
>
> **备注:** 66 pages with appendix, 10 figures (Appendix: 26 Figures), 11 tables. Code available at: this https URL
>
> **摘要:** Successful human-agent teaming relies on an agent being able to understand instructions given by a (human) principal. In many cases, an instruction may be incomplete or ambiguous. In such cases, the agent must infer the unspoken intentions from their shared context, that is, it must exercise the principal's Theory of Mind (ToM) and infer the mental states of its principal. We consider the prospects of effective human-agent collaboration using large language models (LLMs). To assess ToM in a dynamic, goal-oriented, and collaborative environment, we introduce a novel task, Instruction Inference, in which an agent assists a principal in reaching a goal by interpreting incomplete or ambiguous instructions. We present Tomcat, an LLM-based agent, designed to exhibit ToM reasoning in interpreting and responding to the principal's instructions. We implemented two variants of Tomcat. One, dubbed Fs-CoT (Fs for few-shot, CoT for chain-of-thought), is based on a small number of examples demonstrating the requisite structured reasoning. One, dubbed CP (commonsense prompt), relies on commonsense knowledge and information about the problem. We realized both variants of Tomcat on three leading LLMs, namely, GPT-4o, DeepSeek-R1, and Gemma-3-27B. To evaluate the effectiveness of Tomcat, we conducted a study with 52 human participants in which we provided participants with the same information as the CP variant. We computed intent accuracy, action optimality, and planning optimality to measure the ToM capabilities of Tomcat and our study participants. We found that Tomcat with Fs-CoT, particularly with GPT-4o and DeepSeek-R1, achieves performance comparable to the human participants, underscoring its ToM potential for human-agent collaboration.
>
---
#### [replaced 034] Context-Agent: Dynamic Discourse Trees for Non-Linear Dialogue
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统任务，旨在解决长对话中非线性流的问题。通过构建动态树结构管理对话历史，提升模型在复杂对话中的表现。**

- **链接: [https://arxiv.org/pdf/2604.05552](https://arxiv.org/pdf/2604.05552)**

> **作者:** Junan Hu; Shudan Guo; Wenqi Liu; Jianhua Yin; Yinwei Wei
>
> **备注:** 14 pages, 7 figures, ACL 2026
>
> **摘要:** Large Language Models demonstrate outstanding performance in many language tasks but still face fundamental challenges in managing the non-linear flow of human conversation. The prevalent approach of treating dialogue history as a flat, linear sequence is misaligned with the intrinsically hierarchical and branching structure of natural discourse, leading to inefficient context utilization and a loss of coherence during extended interactions involving topic shifts or instruction refinements. To address this limitation, we introduce Context-Agent, a novel framework that models multi-turn dialogue history as a dynamic tree structure. This approach mirrors the inherent non-linearity of conversation, enabling the model to maintain and navigate multiple dialogue branches corresponding to different topics. Furthermore, to facilitate robust evaluation, we introduce the Non-linear Task Multi-turn Dialogue (NTM) benchmark, specifically designed to assess model performance in long-horizon, non-linear scenarios. Our experiments demonstrate that Context-Agent enhances task completion rates and improves token efficiency across various LLMs, underscoring the value of structured context management for complex, dynamic dialogues. The dataset and code is available at GitHub.
>
---
#### [replaced 035] Automatic Combination of Sample Selection Strategies for Few-Shot Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于少样本学习任务，旨在解决样本选择策略效果不佳的问题。通过提出ACSESS方法，自动组合多种策略以提升模型性能。**

- **链接: [https://arxiv.org/pdf/2402.03038](https://arxiv.org/pdf/2402.03038)**

> **作者:** Branislav Pecher; Ivan Srba; Maria Bielikova; Joaquin Vanschoren
>
> **备注:** Accepted to the Findings of ACL 2026
>
> **摘要:** In few-shot learning, the selection of samples has a significant impact on the performance of the model. While effective sample selection strategies are well-established in supervised settings, research on large language models largely overlooks them, favouring strategies specifically tailored to individual in-context learning settings. In this paper, we propose a new method for Automatic Combination of SamplE Selection Strategies (ACSESS) to leverage the strengths and complementarity of various well-established selection objectives. We investigate and compare the impact of 23 sample selection strategies on the performance of 5 in-context learning models and 3 few-shot learning approaches (meta-learning, few-shot fine-tuning) over 6 text and 8 image datasets. The experimental results show that the combination of strategies through the ACSESS method consistently outperforms all individual selection strategies and performs on par or exceeds the in-context learning specific baselines. Lastly, we demonstrate that sample selection remains effective even on smaller datasets, yielding the greatest benefits when only a few shots are selected, while its advantage diminishes as the number of shots increases.
>
---
#### [replaced 036] IPQA: A Benchmark for Core Intent Identification in Personalized Question Answering
- **分类: cs.CL**

- **简介: 该论文属于个性化问答任务，旨在解决核心意图识别问题。通过构建IPQA基准，评估系统在用户历史中识别关键意图的能力，发现现有模型表现不佳。**

- **链接: [https://arxiv.org/pdf/2510.23536](https://arxiv.org/pdf/2510.23536)**

> **作者:** Jieyong Kim; Maryam Amirizaniani; Soojin Yoon; Dongha Lee
>
> **摘要:** Intent identification serves as the foundation for generating appropriate responses in personalized question answering (PQA). However, existing benchmarks evaluate only response quality or retrieval performance without directly measuring intent identification capabilities. This gap is critical because without understanding which intents users prioritize, systems cannot generate responses satisfying individual information needs. To address this, we introduce the concept of core intents: intents users prioritize when selecting answers to satisfy their information needs. To evaluate these core intents, we propose IPQA, a benchmark for core Intent identification in Personalized Question Answering. Since users do not explicitly state their prioritized intents, we derive core intents from observable behavior patterns in answer selection, grounded in satisficing theory where users choose answers meeting their acceptance thresholds. We construct a dataset with various domains through systematic filtering, LLM-based annotation, and rigorous quality control combining automated verification with human validation. Experimental evaluations across state-of-the-art language models reveal that current systems struggle with core intent identification in personalized contexts. Models fail to identify core intents from user histories, with performance degrading as question complexity increases. The code and dataset will be made publicly available to facilitate future research in this direction.
>
---
#### [replaced 037] Token Statistics Reveal Conversational Drift in Multi-turn LLM Interaction
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决多轮对话中一致性监测问题。通过分析token统计信息，提出Bipredictability指标和IDT架构，实现对话结构的一致性监控。**

- **链接: [https://arxiv.org/pdf/2604.13061](https://arxiv.org/pdf/2604.13061)**

> **作者:** Wael Hafez; Amir Nazeri
>
> **备注:** 13 Pages, 3 Figures
>
> **摘要:** Large language models, LLMs, are increasingly deployed in multiturn settings where earlier responses shape later ones, making reliability dependent on whether a conversation remains consistent over time. When this consistency degrades undetected, downstream decisions lose their grounding in the exchange that produced them. Yet current evaluation methods assess isolated outputs rather than the interaction producing them. Here we show that conversational structural consistency can be monitored directly from token frequency statistics, without embeddings, auxiliary evaluators or access to model internals. We formalize this signal as Bipredictability, P, which measures shared predictability across the context, response, next prompt loop relative to the turn total uncertainty, and implement it in a lightweight auxiliary architecture, the Information Digital Twin, IDT. Across 4,574 conversational turns spanning 34 conditions, one student model and three frontier teacher models, P established a stable runtime baseline, aligned with structural consistency in 85 percent of conditions but with semantic quality in only 44 percent, and the IDT detected all tested contradictions, topic shifts and non-sequiturs with 100 percent sensitivity. These results show that reliability in extended LLM interaction cannot be reduced to response quality alone, and that structural monitoring from the observable token stream can complement semantic evaluation in deployment.
>
---
#### [replaced 038] STRIDE-ED: A Strategy-Grounded Stepwise Reasoning Framework for Empathetic Dialogue Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于情感对话系统任务，旨在解决现有方法在情感策略和多阶段推理上的不足。提出STRIDE-ED框架，结合策略引导的推理和高质量数据训练，提升情感对话效果。**

- **链接: [https://arxiv.org/pdf/2604.07100](https://arxiv.org/pdf/2604.07100)**

> **作者:** Hongru Ji; Yuyin Fan; Meng Zhao; Xianghua Li; Lianwei Wu; Chao Gao
>
> **备注:** Accepted by ACL 2026
>
> **摘要:** Empathetic dialogue requires not only recognizing a user's emotional state but also making strategy-aware, context-sensitive decisions throughout response generation. However, the lack of a comprehensive empathy strategy framework, explicit task-aligned multi-stage reasoning, and high-quality strategy-aware data fundamentally limits existing approaches, preventing them from effectively modeling empathetic dialogue as a complex, multi-stage cognitive and decision-making process. To address these challenges, we propose STRIDE-ED, a STRategy-grounded, Interpretable, and DEep reasoning framework that models Empathetic Dialogue through structured, strategy-conditioned reasoning. To support effective learning, we develop a strategy-aware data refinement pipeline integrating LLM-based annotation, multi-model consistency-weighted evaluation, and dynamic sampling to construct high-quality training data aligned with empathetic strategies. Furthermore, we adopt a two-stage training paradigm that combines supervised fine-tuning with multi-objective reinforcement learning to better align model behaviors with target emotions, empathetic strategies, and response formats. Extensive experiments demonstrate that STRIDE-ED generalizes across diverse open-source LLMs and consistently outperforms existing methods on both automatic metrics and human evaluations.
>
---
#### [replaced 039] Reading Between the Lines: The One-Sided Conversation Problem
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究一端对话问题（1SC），旨在从单方对话中推断缺失信息。任务包括重建缺失发言和生成摘要，通过模型评估与实验验证方法有效性。**

- **链接: [https://arxiv.org/pdf/2511.03056](https://arxiv.org/pdf/2511.03056)**

> **作者:** Victoria Ebert; Rishabh Singh; Tuochao Chen; Noah A. Smith; Shyamnath Gollakota
>
> **备注:** 8 pages, 6 figures, 4 tables. Accepted to ACL Findings 2026
>
> **摘要:** Conversational AI is constrained in many real-world settings where only one side of a dialogue can be recorded, such as telemedicine, call centers, and smart glasses. We formalize this as the one-sided conversation problem (1SC): inferring and learning from one side of a conversation. We study two tasks: (1) reconstructing the missing speaker's turns for real-time use cases, and (2) generating summaries from one-sided transcripts. Evaluating prompting and finetuned models on MultiWOZ, DailyDialog, and Candor with both human A/B testing and LLM-as-a-judge metrics, we find that access to one future turn and information about utterance length improves reconstruction, placeholder prompting helps to mitigate hallucination, and while large models generate promising reconstructions with prompting, smaller models require finetuning. Further, high-quality summaries can be generated without reconstructing missing turns. We present 1SC as a novel challenge and report promising results that mark a step toward privacy-aware conversational AI.
>
---
#### [replaced 040] SignX: Continuous Sign Recognition in Compact Pose-Rich Latent Space
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于连续手语识别任务，旨在解决手语视频处理复杂和计算消耗大的问题。通过构建紧凑的姿势潜在空间，实现高效准确的手语识别。**

- **链接: [https://arxiv.org/pdf/2504.16315](https://arxiv.org/pdf/2504.16315)**

> **作者:** Sen Fang; Yalin Feng; Chunyu Sui; Hongbin Zhong; Yanxin Zhang; Hongwei Yi; Hezhen Hu; Dimitris N. Metaxas
>
> **备注:** 33 pages, CSLR SOTA (2026). More demo at this https URL
>
> **摘要:** The complexity of Sign Language (SL) data processing brings many challenges. The current approach to recognition of SL signs aims to translate RGB sign language videos through pose information into Word-based ID Glosses, which serve to uniquely identify signs. This paper proposes SignX, a novel framework for continuous sign language recognition (SLR) in compact pose-rich latent space. First, we construct a unified latent representation that encodes heterogeneous pose formats (SMPLer-X, DWPose, Mediapipe, PrimeDepth, and Sapiens Segmentation) into a compact, information-dense space. Second, we train a ViT-based Video-to-Pose module to extract this latent representation directly from raw videos. Finally, we develop a temporal modeling and sequence refinement method that operates entirely in this latent space. This multi-stage design achieves end-to-end SLR while significantly reducing computational consumption. Experimental results demonstrate that SignX achieves SOTA accuracy on continuous SLR and Translation task, delivering nearly a 50-fold acceleration over pixel-space baselines.
>
---
#### [replaced 041] Losses that Cook: Topological Optimal Transport for Structured Recipe Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于烹饪配方生成任务，旨在提升配方的准确性与合理性。通过引入拓扑损失和Dice损失，优化 ingredient 和操作步骤的生成效果。**

- **链接: [https://arxiv.org/pdf/2601.02531](https://arxiv.org/pdf/2601.02531)**

> **作者:** Mattia Ottoborgo; Daniele Rege Cambrin; Paolo Garza
>
> **备注:** Accepted to ACL 2026 Findings
>
> **摘要:** Cooking recipes are complex procedures that require not only a fluent and factual text, but also accurate timing, temperature, and procedural coherence, as well as the correct composition of ingredients. Standard training procedures are primarily based on cross-entropy and focus solely on fluency. Building on RECIPE-NLG, we investigate the use of several composite objectives and present a new topological loss that represents ingredient lists as point clouds in embedding space, minimizing the divergence between predicted and gold ingredients. Using both standard NLG metrics and recipe-specific metrics, we find that our loss significantly improves ingredient- and action-level metrics. Meanwhile, the Dice loss excels in time/temperature precision, and the mixed loss yields competitive trade-offs with synergistic gains in quantity and time. A human preference analysis supports our finding, showing our model is preferred in 62% of the cases.
>
---
#### [replaced 042] RoleConflictBench: A Benchmark of Role Conflict Scenarios for Evaluating LLMs' Contextual Sensitivity
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言处理任务，旨在评估大模型在角色冲突场景中的上下文敏感性。通过构建基准数据集，分析模型是否依据情境而非预设角色偏好做决策。**

- **链接: [https://arxiv.org/pdf/2509.25897](https://arxiv.org/pdf/2509.25897)**

> **作者:** Jisu Shin; Hoyun Song; Juhyun Oh; Changgeon Ko; Eunsu Kim; Chani Jung; Alice Oh
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** People often encounter role conflicts -- social dilemmas where the expectations of multiple roles clash and cannot be simultaneously fulfilled. As large language models (LLMs) increasingly navigate these social dynamics, a critical research question emerges. When faced with such dilemmas, do LLMs prioritize dynamic contextual cues or the learned preferences? To address this, we introduce RoleConflictBench, a novel benchmark designed to measure the contextual sensitivity of LLMs in role conflict scenarios. To enable objective evaluation within this subjective domain, we employ situational urgency as a constraint for decision-making. We construct the dataset through a three-stage pipeline that generates over 13,000 realistic scenarios across 65 roles in five social domains by systematically varying the urgency of competing situations. This controlled setup enables us to quantitatively measure contextual sensitivity, determining whether model decisions align with the situational contexts or are overridden by the learned role preferences. Our analysis of 10 LLMs reveals that models substantially deviate from this objective baseline. Instead of responding to dynamic contextual cues, their decisions are predominantly governed by the preferences toward specific social roles.
>
---
#### [replaced 043] Understanding New-Knowledge-Induced Factual Hallucinations in LLMs: Analysis and Interpretation
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在引入新知识时产生的事实幻觉问题，通过设计数据集和分析不同知识类型，揭示幻觉成因及传播机制。**

- **链接: [https://arxiv.org/pdf/2511.02626](https://arxiv.org/pdf/2511.02626)**

> **作者:** Renfei Dang; Peng Hu; Zhejian Lai; Changjiang Gao; Min Zhang; Shujian Huang
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Prior works have shown that fine-tuning on new knowledge can induce factual hallucinations in large language models (LLMs), leading to incorrect outputs when evaluated on previously known information. However, the specific manifestations of such hallucination and its underlying mechanisms remain insufficiently understood. Our work addresses this gap by designing a controlled dataset \textit{Biography-Reasoning}, and conducting a fine-grained analysis across multiple knowledge types and two task types, including knowledge question answering (QA) and knowledge reasoning tasks. We find that hallucinations not only severely affect tasks involving newly introduced knowledge, but also propagate to other evaluation tasks. Moreover, when fine-tuning on a dataset in which a specific knowledge type consists entirely of new knowledge, LLMs exhibit elevated hallucination tendencies. This suggests that the degree of unfamiliarity within a particular knowledge type, rather than the overall proportion of new knowledge, is a stronger driver of hallucinations. Through interpretability analysis, we show that learning new knowledge weakens the model's attention to key entities in the input question, leading to an over-reliance on surrounding context and a higher risk of hallucination. Conversely, reintroducing a small amount of known knowledge during the later stages of training restores attention to key entities and substantially mitigates hallucination behavior. Finally, we demonstrate that disrupted attention patterns can propagate across lexically similar contexts, facilitating the spread of hallucinations beyond the original task.
>
---
#### [replaced 044] ConlangCrafter: Constructing Languages with a Multi-Hop LLM Pipeline
- **分类: cs.CL**

- **简介: 该论文提出ConlangCrafter，用于自动化构建人工语言。解决传统语言设计依赖专家的问题，通过多阶段LLM流程生成连贯且多样化的语言体系。**

- **链接: [https://arxiv.org/pdf/2508.06094](https://arxiv.org/pdf/2508.06094)**

> **作者:** Morris Alper; Moran Yanuka; Raja Giryes; Gašper Beguš
>
> **备注:** Accepted to ACL 2026. Project page: this https URL
>
> **摘要:** Constructed languages (conlangs) such as Esperanto and Quenya have played diverse roles in art, philosophy, and international communication. Meanwhile, foundation models have revolutionized creative generation in text, images, and beyond. In this work, we leverage modern LLMs as computational creativity aids for end-to-end conlang creation. We introduce ConlangCrafter, a multi-hop pipeline that decomposes language design into modular stages -- phonology, morphology, syntax, lexicon generation, and translation. At each stage, our method leverages LLMs' metalinguistic reasoning capabilities, injecting randomness to encourage diversity and leveraging self-refinement feedback to encourage consistency in the emerging language description. We construct a novel, scalable evaluation framework for this task, evaluating metrics measuring consistency and typological diversity. Automatic and manual evaluations demonstrate ConlangCrafter's ability to produce coherent and varied conlangs without human linguistic expertise.
>
---
#### [replaced 045] Revisiting Entropy Regularization: Adaptive Coefficient Unlocks Its Potential for LLM Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文属于大语言模型强化学习任务，旨在解决策略熵坍塌问题。通过自适应熵正则化框架，提升模型的探索能力与推理性能。**

- **链接: [https://arxiv.org/pdf/2510.10959](https://arxiv.org/pdf/2510.10959)**

> **作者:** Xiaoyun Zhang; Xiaojian Yuan; Di Huang; Wang You; Chen Hu; Jingqing Ruan; Ai Jian; Kejiang Chen; Xing Hu
>
> **备注:** 16 pages, 4 figures
>
> **摘要:** Reasoning ability has become a defining capability of Large Language Models (LLMs), with Reinforcement Learning with Verifiable Rewards (RLVR) emerging as a key paradigm to enhance it. However, RLVR training often suffers from policy entropy collapse, where the policy becomes overly deterministic, hindering exploration and limiting reasoning performance. While entropy regularization is a common remedy, its effectiveness is highly sensitive to the fixed coefficient, making it unstable across tasks and models. In this work, we revisit entropy regularization in RLVR and argue that its potential has been largely underestimated. Our analysis shows that (i) tasks of varying difficulty demand distinct exploration intensities, and (ii) balanced exploration may require the policy entropy to be maintained within a moderate range below its initial level. Therefore, we propose Adaptive Entropy Regularization (AER)--a framework that dynamically balances exploration and exploitation via three components: difficulty-aware coefficient allocation, initial-anchored target entropy, and dynamic global coefficient adjustment. Experiments on multiple mathematical reasoning benchmarks show that AER consistently outperforms baselines, improving both reasoning accuracy and exploration capability.
>
---
#### [replaced 046] Large Reasoning Models Are (Not Yet) Multilingual Latent Reasoners
- **分类: cs.CL**

- **简介: 该论文研究多语言大推理模型的隐式推理能力，探讨其在不同语言中的表现差异。任务为多语言推理分析，解决模型是否具备跨语言隐式推理的问题。工作包括实验与表征分析。**

- **链接: [https://arxiv.org/pdf/2601.02996](https://arxiv.org/pdf/2601.02996)**

> **作者:** Yihong Liu; Raoyuan Zhao; Hinrich Schütze; Michael A. Hedderich
>
> **备注:** ACL 2026 Findings
>
> **摘要:** Large reasoning models (LRMs) achieve strong performance on mathematical reasoning tasks, often attributed to their capability to generate explicit chain-of-thought (CoT) explanations. However, recent work shows that LRMs often arrive at the correct answer before completing these textual reasoning steps, indicating the presence of latent reasoning -- internal, non-verbal computation encoded in hidden states. While this phenomenon has been explored in English, its multilingual behavior remains largely unknown. In this paper, we conduct a systematic investigation of multilingual latent reasoning in LRMs across 11 languages. Using a truncation-based strategy, we examine how the correct answer emerges as the model is given only partial reasoning traces, allowing us to measure stepwise latent prediction formation. Our results reveal clear evidence of multilingual latent reasoning, though unevenly: strong in resource-rich languages, weaker in low-resource ones, and broadly less observable on harder benchmarks. To understand whether these differences reflect distinct internal mechanisms, we further perform representational analyses. Despite surface-level disparities, we find that the internal evolution of predictions is highly consistent across languages and broadly aligns with English -- a pattern suggesting an English-centered latent reasoning pathway.
>
---
#### [replaced 047] Wisdom is Knowing What not to Say: Hallucination-Free LLMs Unlearning via Attention Shifting
- **分类: cs.CL**

- **简介: 该论文属于机器学习中的模型遗忘任务，旨在解决LLMs在遗忘敏感数据时导致的性能下降和幻觉问题。提出一种基于注意力转移的框架，提升遗忘效果与模型可靠性。**

- **链接: [https://arxiv.org/pdf/2510.17210](https://arxiv.org/pdf/2510.17210)**

> **作者:** Chenchen Tan; Youyang Qu; Xinghao Li; Hui Zhang; Shujie Cui; Cunjian Chen; Longxiang Gao
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** The increase in computing power and the necessity of AI-assisted decision-making boost the growing application of large language models (LLMs). Along with this, the potential retention of sensitive data of LLMs has spurred increasing research into machine unlearning. However, existing unlearning approaches face a critical dilemma: Aggressive unlearning compromises model utility, while conservative strategies preserve utility but risk hallucinated responses. This significantly limits LLMs' reliability in knowledge-intensive applications. To address this, we introduce a novel Attention-Shifting (AS) framework for selective unlearning. AS is driven by two design objectives: (1) context-preserving suppression that attenuates attention to fact-bearing tokens without disrupting LLMs' linguistic structure; and (2) hallucination-resistant response shaping that discourages fabricated completions when queried about unlearning content. AS realizes these objectives through two attention-level interventions, which are importance-aware suppression applied to the unlearning set to reduce reliance on memorized knowledge and attention-guided retention enhancement that reinforces attention toward semantically essential tokens in the retained dataset to mitigate unintended degradation. These two components are jointly optimized via a dual-loss objective, which forms a soft boundary that localizes unlearning while preserving unrelated knowledge under representation superposition. Experimental results show that AS improves performance preservation over the state-of-the-art unlearning methods, achieving up to 15% higher accuracy on the ToFU benchmark and 10% on the TDEC benchmark, while maintaining competitive hallucination-free unlearning effectiveness. Compared to existing methods, AS demonstrates a superior balance between unlearning effectiveness, generalization, and response reliability.
>
---
#### [replaced 048] FSPO: Few-Shot Optimization of Synthetic Preferences Personalizes to Real Users
- **分类: cs.LG; cs.AI; cs.CL; cs.HC; stat.ML**

- **简介: 该论文属于LLM个性化任务，解决如何高效个性化LLM的问题。提出FSPO算法，通过少量用户偏好数据优化奖励函数，并构建合成数据集提升效果。**

- **链接: [https://arxiv.org/pdf/2502.19312](https://arxiv.org/pdf/2502.19312)**

> **作者:** Anikait Singh; Sheryl Hsu; Kyle Hsu; Eric Mitchell; Stefano Ermon; Tatsunori Hashimoto; Archit Sharma; Chelsea Finn
>
> **备注:** Website: this https URL
>
> **摘要:** Effective personalization of LLMs is critical for a broad range of user-interfacing applications such as virtual assistants and content curation. Inspired by the strong in-context capabilities of LLMs, we propose few-shot preference optimization (FSPO), an algorithm for LLM personalization that reframes reward modeling as a meta-learning problem. Under FSPO, an LLM learns to quickly infer a personalized reward function for a user via a few labeled preferences. FSPO also utilizes user description rationalization (RAT) to encourage better reward modeling and instruction following, recovering performance with the oracle user description. Since real-world preference data is challenging to collect at scale, we propose careful design choices to construct synthetic preference datasets for personalization, generating over 1M synthetic personalized preferences using publicly available LLMs. To successfully transfer from synthetic data to real users, we find it crucial for the data to exhibit both high diversity and coherent, self-consistent structure. We evaluate FSPO on personalized open-ended generation for up to 1,500 synthetic users across three domains: movie reviews, education, and open-ended question answering. We also run a controlled human study. Overall, FSPO achieves an 87% Alpaca Eval winrate in generating responses that are personalized to synthetic users and a 70% winrate with real human users in open-ended question answering.
>
---
#### [replaced 049] A Triadic Suffix Tokenization Scheme for Numerical Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Triadic Suffix Tokenization（TST）方法，解决数字分割不一致导致的数值推理错误问题。通过三元组分组和显式量级标记，提升模型对数量级的感知能力。属于数值推理任务。**

- **链接: [https://arxiv.org/pdf/2604.11582](https://arxiv.org/pdf/2604.11582)**

> **作者:** Olga Chetverina
>
> **备注:** 9 pages, 1 figure. Updated version with GST algorithm and flexible group size N. This research was conducted independently outside of any institutional assignments
>
> **摘要:** Standard subword tokenization methods fragment numbers inconsistently, causing large language models (LLMs) to lose positional and decimal structure - a primary driver of errors in arithmetic and scientific reasoning. We introduce Triadic Suffix Tokenization (TST), a deterministic scheme that partitions digits into three-digit triads and annotates each triad with an explicit magnitude marker. Critically, the scheme defines a fixed, one-to-one mapping between suffixes and orders of magnitude for the integer part (thousands, millions, billions, etc.) and a parallel system of replicated markers for fractional depth (tenths, thousandths, millionths, etc.). Unlike approaches that rely on positional inference, this method provides a consistent gradient signal, which should ensure stable convergence. Two implementation variants are proposed: (1) a vocabulary-based approach that adds at most 10,000 fixed tokens to an existing vocabulary, covering 33 orders of magnitude ($10^{-15}$ to $10^{18}$); and (2) a suffix-marker approach that uses a small set of special tokens to denote magnitude dynamically. Both variants preserve exact digits while making order-of-magnitude relationships transparent at the token level. While we focus on 3-digit groups (Triadic), the framework is inherently scalable to any group size for precise vocabulary optimization. Furthermore, it allows for linear vocabulary expansion to accommodate arbitrary precision and range. TST is architecture-agnostic and can be integrated as a drop-in preprocessing step. Experimental validation is deferred to future work.
>
---
#### [replaced 050] A Linguistics-Aware LLM Watermarking via Syntactic Predictability
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI可信治理任务，旨在解决水印强度与文本质量的平衡问题。提出STELA框架，利用语言结构特性动态调整水印，提升检测鲁棒性且无需模型日志。**

- **链接: [https://arxiv.org/pdf/2510.13829](https://arxiv.org/pdf/2510.13829)**

> **作者:** Shinwoo Park; Hyejin Park; Hyeseon An; Yo-Sub Han
>
> **备注:** ACL 2026
>
> **摘要:** As large language models (LLMs) continue to advance rapidly, reliable governance tools have become critical. Publicly verifiable watermarking is particularly essential for fostering a trustworthy AI ecosystem. A central challenge persists: balancing text quality against detection robustness. Recent studies have sought to navigate this trade-off by leveraging signals from model output distributions (e.g., token-level entropy); however, their reliance on these model-specific signals presents a significant barrier to public verification, as the detection process requires access to the logits of the underlying model. We introduce STELA, a novel framework that aligns watermark strength with the linguistic degrees of freedom inherent in language. STELA dynamically modulates the signal using part-of-speech (POS) n-gram-modeled linguistic indeterminacy, weakening it in grammatically constrained contexts to preserve quality and strengthening it in contexts with greater linguistic flexibility to enhance detectability. Our detector operates without access to any model logits, thus facilitating publicly verifiable detection. Through extensive experiments on typologically diverse languages-analytic English, isolating Chinese, and agglutinative Korean-we show that STELA surpasses prior methods in detection robustness. Our code is available at this https URL.
>
---
#### [replaced 051] CLewR: Curriculum Learning with Restarts for Machine Translation Preference Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于机器翻译任务，解决数据顺序对模型性能影响的问题。通过引入带有重启的课程学习策略（CLewR），提升偏好优化效果。**

- **链接: [https://arxiv.org/pdf/2601.05858](https://arxiv.org/pdf/2601.05858)**

> **作者:** Alexandra Dragomir; Florin Brad; Radu Tudor Ionescu
>
> **备注:** Accepted at ACL 2026
>
> **摘要:** Large language models (LLMs) have demonstrated competitive performance in zero-shot multilingual machine translation (MT). Some follow-up works further improved MT performance via preference optimization, but they leave a key aspect largely underexplored: the order in which data samples are given during training. We address this topic by integrating curriculum learning into various state-of-the-art preference optimization algorithms to boost MT performance. We introduce a novel curriculum learning strategy with restarts (CLewR), which reiterates easy-to-hard curriculum multiple times during training to effectively mitigate the catastrophic forgetting of easy examples. We demonstrate consistent gains across several model families (Gemma2, Qwen2.5, Llama3.1) and preference optimization techniques. We publicly release our code at this https URL.
>
---
#### [replaced 052] MEDSYN: Benchmarking Multi-EviDence SYNthesis in Complex Clinical Cases for Multimodal Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出MEDSYN基准，用于评估多模态大语言模型在复杂临床案例中的多证据合成能力。任务是提升模型在诊断中的综合推理能力，解决其对不同证据类型利用不均的问题。**

- **链接: [https://arxiv.org/pdf/2602.21950](https://arxiv.org/pdf/2602.21950)**

> **作者:** Boqi Chen; Xudong Liu; Jiachuan Peng; Marianne Frey-Marti; Bang Zheng; Kyle Lam; Lin Li; Jianing Qiu
>
> **摘要:** Multimodal large language models (MLLMs) have shown great potential in medical applications, yet existing benchmarks inadequately capture real-world clinical complexity. We introduce MEDSYN, a multilingual, multimodal benchmark of highly complex clinical cases with up to 7 distinct visual clinical evidence (CE) types per case. Mirroring clinical workflow, we evaluate 18 MLLMs on differential diagnosis (DDx) generation and final diagnosis (FDx) selection. While top models often match or even outperform human experts on DDx generation, all MLLMs exhibit a much larger DDx--FDx performance gap compared to expert clinicians, indicating a failure mode in synthesis of heterogeneous CE types. Ablations attribute this failure to (i) overreliance on less discriminative textual CE ($\it{e.g.}$, medical history) and (ii) a cross-modal CE utilization gap. We introduce Evidence Sensitivity to quantify the latter and show that a smaller gap correlates with higher diagnostic accuracy. Finally, we demonstrate how it can be used to guide interventions to improve model performance. We will open-source our benchmark and code.
>
---
#### [replaced 053] ATTNPO: Attention-Guided Process Supervision for Efficient Reasoning
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，解决大模型推理冗余问题。通过注意力机制实现步骤级信用分配，减少冗余推理，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2602.09953](https://arxiv.org/pdf/2602.09953)**

> **作者:** Shuaiyi Nie; Siyu Ding; Wenyuan Zhang; Linhao Yu; Tianmeng Yang; Yao Chen; Weichong Yin; Yu Sun; Hua Wu; Tingwen Liu
>
> **备注:** Accepted by ACL 2026 Main
>
> **摘要:** Large reasoning models trained with reinforcement learning and verifiable rewards (RLVR) achieve strong performance on complex reasoning tasks, yet often overthink, generating redundant reasoning without performance gains. Existing trajectory-level length penalties often fail to effectively shorten reasoning length and degrade accuracy, as they uniformly treat all reasoning steps and lack fine-grained signals to distinguish redundancy from necessity. Meanwhile, process-supervised methods are typically resource-intensive and suffer from inaccurate credit assignment. To address these issues, we propose ATTNPO, a low-overhead process-supervised RL framework that leverages the model's intrinsic attention signals for step-level credit assignment. We first identify a set of special attention heads that naturally focus on essential steps while suppressing redundant ones. By leveraging the attention scores of these heads, We then employ two sub-strategies to mitigate overthinking by discouraging redundant steps while preserving accuracy by reducing penalties on essential steps. Experimental results show that ATTNPO substantially reduces reasoning length while significantly improving performance across 9 benchmarks.
>
---
#### [replaced 054] Fragile Thoughts: How Large Language Models Handle Chain-of-Thought Perturbations
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，研究大模型在链式思维扰动下的鲁棒性。通过实验分析五种扰动类型对模型性能的影响，揭示模型规模与抗干扰能力的关系，为实际应用提供参考。**

- **链接: [https://arxiv.org/pdf/2603.03332](https://arxiv.org/pdf/2603.03332)**

> **作者:** Ashwath Vaithinathan Aravindan; Mayank Kejriwal
>
> **摘要:** Chain-of-Thought (CoT) prompting has emerged as a foundational technique for eliciting reasoning from Large Language Models (LLMs), yet the robustness of this approach to corruptions in intermediate reasoning steps remains poorly understood. This paper presents a comprehensive empirical evaluation of LLM robustness to a structured taxonomy of 5 CoT perturbation types: \textit{MathError, UnitConversion, Sycophancy, SkippedSteps,} and \textit{ExtraSteps}. We evaluate 13 models spanning three orders of magnitude in parameter count, testing their ability to complete mathematical reasoning tasks despite perturbations injected in the reasoning chain. Our key findings reveal heterogeneous vulnerability patterns: MathError perturbations produce the most severe degradation in small models (50-60\% accuracy loss) but show strong scaling benefits; UnitConversion remains challenging across all scales (>5\% loss even for midsized models); ExtraSteps incur minimal accuracy degradation (0-6\%) even for the smallest of models; Sycophancy and SkippedSteps produce modest effects ($\sim$10\% loss for small models) and slightly improve with scale. Scaling relationships show that model size serve as a protective factor against many perturbations but not always. These findings have direct implications for deploying LLMs in multi-stage reasoning pipelines and underscore the necessity of task-specific robustness assessments and mitigation strategies. The code and results are available at this https URL
>
---
#### [replaced 055] Collaboration of Fusion and Independence: Hypercomplex-driven Robust Multi-Modal Knowledge Graph Completion
- **分类: cs.CL**

- **简介: 该论文属于多模态知识图谱补全任务，旨在解决现有方法在融合与独立模态表示间的平衡问题。提出M-Hyper方法，结合融合与独立模态优势，提升补全效果。**

- **链接: [https://arxiv.org/pdf/2509.23714](https://arxiv.org/pdf/2509.23714)**

> **作者:** Zhiqiang Liu; Yichi Zhang; Mengshu Sun; Lei Liang; Wen Zhang
>
> **备注:** ACL 2026 (Main)
>
> **摘要:** Multi-modal knowledge graph completion (MMKGC) aims to discover missing facts in multi-modal knowledge graphs (MMKGs) by leveraging both structural relationships and diverse modality information of entities. Existing MMKGC methods follow two multi-modal paradigms: fusion-based and ensemble-based. Fusion-based methods employ fixed fusion strategies, which inevitably leads to the loss of modality-specific information and a lack of flexibility to adapt to varying modality relevance across contexts. In contrast, ensemble-based methods retain modality independence through dedicated sub-models but struggle to capture the nuanced, context-dependent semantic interplay between modalities. To overcome these dual limitations, we propose a novel MMKGC method M-Hyper, which achieves the coexistence and collaboration of fused and independent modality representations. Our method integrates the strengths of both paradigms, enabling effective cross-modal interactions while maintaining modality-specific information. Inspired by ``quaternion'' algebra, we utilize its four orthogonal bases to represent multiple independent modalities and employ the Hamilton product to efficiently model pair-wise interactions among them. Specifically, we introduce a Fine-grained Entity Representation Factorization (FERF) module and a Robust Relation-aware Modality Fusion (R2MF) module to obtain robust representations for three independent modalities and one fused modality. The resulting four modality representations are then mapped to the four orthogonal bases of a biquaternion (a hypercomplex extension of quaternion) for comprehensive modality interaction. Extensive experiments indicate its state-of-the-art performance, robustness, and computational efficiency.
>
---
#### [replaced 056] When to Trust Tools? Adaptive Tool Trust Calibration For Tool-Integrated Math Reasoning
- **分类: cs.CL**

- **简介: 该论文属于数学推理任务，解决模型不信任或忽略工具结果的问题。提出ATTC框架，根据代码块置信度动态决定是否信任工具结果。**

- **链接: [https://arxiv.org/pdf/2604.08281](https://arxiv.org/pdf/2604.08281)**

> **作者:** Ruotao Xu; Yixin Ji; Yu Luo; Jinpeng Li; Dong Li; Peifeng Li; Juntao Li; Min Zhang
>
> **摘要:** Large reasoning models (LRMs) have achieved strong performance enhancement through scaling test time computation, but due to the inherent limitations of the underlying language models, they still have shortcomings in tasks that require precise computation and extensive knowledge reserves. Tool-Integrated Reasoning (TIR) has emerged as a promising paradigm that incorporates tool call and execution within the reasoning trajectory. Although recent works have released some powerful open-source TIR models, our analysis reveals that these models still suffer from critical deficiencies. We find that when the reasoning of the model conflicts with the tool results, the model tends to believe in its own reasoning. And there are cases where the tool results are correct but are ignored by the model, resulting in incorrect answers, which we define as "Tool Ignored''. This indicates that the model does not know when to trust or ignore the tool. To overcome these limitations, We introduce Adaptive Tool Trust Calibration (ATTC), a novel framework that guides the model to adaptively choose to trust or ignore the tool results based on the confidence score of generated code blocks. The experimental results from various open-source TIR models of different sizes and across multiple datasets demonstrate that ATTC effectively reduces the "Tool Ignored" issue, resulting in a performance increase of 4.1% to 7.5%.
>
---
#### [replaced 057] Deep Learning Based Amharic Chatbot for FAQs in Universities
- **分类: cs.CY; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大学常见问题解答的效率问题。通过深度学习构建阿姆哈拉语聊天机器人，提升问答效率。**

- **链接: [https://arxiv.org/pdf/2402.01720](https://arxiv.org/pdf/2402.01720)**

> **作者:** Goitom Ybrah Hailu; Hadush Hailu; Shishay Welay
>
> **备注:** 7 pages, 5 figures and 3 tables
>
> **摘要:** University students often spend a considerable amount of time seeking answers to common questions from administrators or teachers. This can become tedious for both parties, leading to a need for a solution. In response, this paper proposes a chatbot model that utilizes natural language processing and deep learning techniques to answer frequently asked questions (FAQs) in the Amharic language. Chatbots are computer programs that simulate human conversation through the use of artificial intelligence (AI), acting as a virtual assistant to handle questions and other tasks. The proposed chatbot program employs tokenization, normalization, stop word removal, and stemming to analyze and categorize Amharic input sentences. Three machine learning model algorithms were used to classify tokens and retrieve appropriate responses: Support Vector Machine (SVM), Multinomial Naïve Bayes, and deep neural networks implemented through TensorFlow, Keras, and NLTK. The deep learning model achieved the best results with 91.55% accuracy and a validation loss of 0.3548 using an Adam optimizer and SoftMax activation function. The chatbot model was integrated with Facebook Messenger and deployed on a Heroku server for 24-hour accessibility. The experimental results demonstrate that the chatbot framework achieved its objectives and effectively addressed challenges such as Amharic Fidel variation, morphological variation, and lexical gaps. Future research could explore the integration of Amharic WordNet to narrow the lexical gap and support more complex questions.
>
---
#### [replaced 058] COMPOSITE-Stem
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出COMPOSITE-STEM基准，用于评估AI在科学领域的推理能力。针对现有基准过时的问题，该工作设计了70个跨学科任务，结合多种评分方式，以更准确衡量AI性能。**

- **链接: [https://arxiv.org/pdf/2604.09836](https://arxiv.org/pdf/2604.09836)**

> **作者:** Kyle Waters; Lucas Nuzzi; Tadhg Looram; Alessandro Tomasiello; Ariel Ghislain Kemogne Kamdoum; Bikun Li; Damien Sileo; Egor Kretov; Francesco Fournier-Facio; Georgios Soloupis; Haile Kassahun; Hew Wolff; Jiaqi Cai; Lianghui Li; Marc Roth; Mohinder Naiya; Naixu Guo; Qicheng Tang; Richard Wheeler; Samuele Sala; Serguei Popov; Steven Dillmann; Yuqi Li
>
> **摘要:** AI agents hold growing promise for accelerating scientific discovery; yet, a lack of frontier evaluations hinders adoption into real workflows. Expert-written benchmarks have proven effective at measuring AI reasoning, but most at this stage have become saturated and only measure performance on constrained outputs. To help address this gap, we introduce COMPOSITE-STEM, a benchmark of 70 expert-written tasks in physics, biology, chemistry, and mathematics, curated by doctoral-level researchers. Our benchmark combines exact-match grading and criterion-based rubrics with an LLM-as-a-jury grading protocol, allowing more flexible assessment of scientifically meaningful outputs. Using an adapted multimodal Terminus-2 agent harness within the Harbor agentic evaluation framework, we evaluate four frontier models. The top-performing model achieves 21%, demonstrating that COMPOSITE-STEM captures capabilities beyond current agent reach. All tasks are open-sourced with contributor permission to support reproducibility and to promote additional research towards AI's acceleration of scientific progress in these domains.
>
---
#### [replaced 059] Creating and Evaluating Personas Using Generative AI: A Scoping Review of 81 Articles
- **分类: cs.HC; cs.CL**

- **简介: 该论文属于技术评估任务，旨在分析生成式AI在人物画像创建中的应用与问题。通过综述81篇文章，探讨其方法、评估不足及潜在风险，并提出负责任的使用指南。**

- **链接: [https://arxiv.org/pdf/2504.04927](https://arxiv.org/pdf/2504.04927)**

> **作者:** Danial Amin; Joni Salminen; Farhan Ahmed; Sonja M.H. Tervola; Sankalp Sethi; Bernard J. Jansen
>
> **备注:** The previous article was updated to add more data
>
> **摘要:** As generative AI (GenAI) is increasingly applied in persona development to represent real users, understanding the implications and limitations of this technology is essential for establishing robust practices. This scoping review analyzes how 81 articles (2022-2025) use GenAI techniques for the creation, evaluation, and application of personas. The articles exhibited good level of reproducibility, with 61% of articles sharing resources (personas, code, or datasets). Furthermore, conversational persona interfaces are increasingly provided alongside traditional profiles. However, nearly half (45%) of the articles lack evaluation, and the majority (86%) use only GPT models. In some articles, GenAI use creates a risk of circularity, in which the same GenAI model both generates and evaluates outputs. Our findings also suggest that GenAI seems to reduce the role of human developers in the persona-creation process. To mitigate the associated risks, we propose actionable guidelines for the responsible integration of GenAI into persona development.
>
---
#### [replaced 060] BlasBench: An Open Benchmark for Irish Speech Recognition
- **分类: cs.CL; cs.SD**

- **简介: 该论文属于语音识别任务，旨在解决爱尔兰语ASR评估中缺乏专用文本归一化的问题。提出BlasBench基准，包含爱尔兰语感知的归一化工具和可复现的评分系统，评估多个模型表现。**

- **链接: [https://arxiv.org/pdf/2604.10736](https://arxiv.org/pdf/2604.10736)**

> **作者:** Jyoutir Raj; John Conway
>
> **备注:** 9 pages, 4 tables, 3 appendices. Code and data: this https URL
>
> **摘要:** Existing multilingual benchmarks include Irish among dozens of languages but apply no Irish-aware text normalisation, leaving reliable and reproducible ASR comparison impossible. We introduce BlasBench, an open evaluation harness that provides a standalone Irish-aware normaliser preserving fadas, lenition, and eclipsis; a reproducible scoring harness and per-utterance predictions released for all evaluated runs. We pilot this by benchmarking 12 systems across four architecture families on Common Voice ga-IE and FLEURS ga-IE. All Whisper variants exceed 100% WER through insertion-driven hallucination. Microsoft Azure reaches 22.2% WER on Common Voice and 57.5% on FLEURS; the best open model, Omnilingual ASR 7B, reaches 30.65% and 39.09% respectively. Models fine-tuned on Common Voice degrade 33-43 points moving to FLEURS, while massively multilingual models degrade only 7-10 - a generalisation gap that single-dataset evaluation misses.
>
---
#### [replaced 061] MTR-DuplexBench: Towards a Comprehensive Evaluation of Multi-Round Conversations for Full-Duplex Speech Language Models
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文提出MTR-DuplexBench，用于全面评估全双工语音语言模型的多轮对话能力，解决现有基准不足的问题。**

- **链接: [https://arxiv.org/pdf/2511.10262](https://arxiv.org/pdf/2511.10262)**

> **作者:** He Zhang; Wenqian Cui; Haoning Xu; Xiaohui Li; Lei Zhu; Haoli Bai; Shaohua Ma; Irwin King
>
> **备注:** Accepted to Findings of ACL 2026
>
> **摘要:** Full-Duplex Speech Language Models (FD-SLMs) enable real-time, overlapping conversational interactions, offering a more dynamic user experience compared to traditional half-duplex models. However, existing benchmarks primarily focus on evaluating single-round interactions, neglecting the complexities of multi-round communication. Evaluating FD-SLMs in multi-round settings poses significant challenges, including blurred turn boundaries in communication and context inconsistency during model inference. Also, existing benchmarks often focus solely on evaluating conversational features, neglecting other critical aspects. To address these gaps, we introduce MTR-DuplexBench, a novel benchmark designed for a comprehensive multi-round evaluation of FD-SLMs. MTR-DuplexBench not only segments continuous full-duplex dialogues into discrete turns for turn-by-turn assessment but also incorporates various evaluation aspects, including conversational features, dialogue quality, instruction following, and safety. Experimental results reveal that current FD-SLMs face difficulties in maintaining consistent performance across multiple rounds and evaluation dimensions, highlighting the necessity and effectiveness of our benchmark. Code and data are available at: this https URL
>
---
#### [replaced 062] Persona-Assigned Large Language Models Exhibit Human-Like Motivated Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究LLM在赋予人格后是否出现类似人类的动机性推理问题，通过实验验证不同人格对推理能力的影响，发现其易受身份认同影响，且传统去偏方法效果有限。**

- **链接: [https://arxiv.org/pdf/2506.20020](https://arxiv.org/pdf/2506.20020)**

> **作者:** Saloni Dash; Amélie Reymond; Emma S. Spiro; Aylin Caliskan
>
> **备注:** ACL Findings 2026
>
> **摘要:** Reasoning in humans is prone to biases due to underlying motivations like identity protection, that undermine rational decision-making and judgment. This \textit{motivated reasoning} at a collective level can be detrimental to society when debating critical issues such as human-driven climate change or vaccine safety, and can further aggravate political polarization. Prior studies have reported that large language models (LLMs) are also susceptible to human-like cognitive biases, however, the extent to which LLMs selectively reason toward identity-congruent conclusions remains largely unexplored. Here, we investigate whether assigning 8 personas across 4 political and socio-demographic attributes induces motivated reasoning in LLMs. Testing 8 LLMs (open source and proprietary) across two reasoning tasks from human-subject studies -- veracity discernment of misinformation headlines and evaluation of numeric scientific evidence -- we find that persona-assigned LLMs have up to 9% reduced veracity discernment relative to models without personas. Political personas specifically are up to 90% more likely to correctly evaluate scientific evidence on gun control when the ground truth is congruent with their induced political identity. Prompt-based debiasing methods are largely ineffective at mitigating these effects. Taken together, our empirical findings are the first to suggest that persona-assigned LLMs exhibit human-like motivated reasoning that is hard to mitigate through conventional debiasing prompts -- raising concerns of exacerbating identity-congruent reasoning in both LLMs and humans.
>
---
#### [replaced 063] Designing Synthetic Discussion Generation Systems: A Case Study for Online Facilitation
- **分类: cs.HC; cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决人工实验成本高的问题，通过设计合成讨论生成系统，降低实验成本并提升效率。**

- **链接: [https://arxiv.org/pdf/2503.16505](https://arxiv.org/pdf/2503.16505)**

> **作者:** Dimitris Tsirmpas; Ion Androutsopoulos; John Pavlopoulos
>
> **摘要:** A critical challenge in social science research is the high cost associated with experiments involving human participants. We identify Synthetic Discussion Generation (SDG), a novel Natural Language Processing (NLP) direction aimed at creating simulated discussions that enable cost-effective pilot experiments and develop a theoretical, task-agnostic framework for designing, evaluating, and implementing these simulations. We argue that the use of proprietary models such as the OpenAI GPT family for such experiments is often unjustified in terms of both cost and capability, despite its prevalence in current research. Our experiments demonstrate that smaller quantized models (7B-8B) can produce effective simulations at a cost more than 44 times lower compared to their proprietary counterparts. We use our framework in the context of online facilitation, where humans actively engage in discussions to improve them, unlike more conventional content moderation. By treating this problem as a downstream task for our framework, we show that synthetic simulations can yield generalizable results at least by revealing limitations before engaging human discussants. In LLM facilitators, a critical limitation is that they are unable to determine when to intervene in a discussion, leading to undesirable frequent interventions and, consequently, derailment patterns similar to those observed in human interactions. Additionally, we find that different facilitation strategies influence conversational dynamics to some extent. Beyond our theoretical SDG framework, we also present a cost-comparison methodology for experimental design, an exploration of available models and algorithms, an open-source Python framework, and a large, publicly available dataset of LLM-generated discussions across multiple models.
>
---
#### [replaced 064] Do LLMs Really Know What They Don't Know? Internal States Mainly Reflect Knowledge Recall Rather Than Truthfulness
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的模型可信性研究，旨在解决LLM是否能区分已知与未知的问题。通过分析模型内部状态，发现其主要反映知识回忆而非事实正确性。**

- **链接: [https://arxiv.org/pdf/2510.09033](https://arxiv.org/pdf/2510.09033)**

> **作者:** Chi Seng Cheang; Hou Pong Chan; Wenxuan Zhang; Yang Deng
>
> **摘要:** Recent work suggests that LLMs "know what they don't know", positing that hallucinated and factually correct outputs arise from distinct internal processes and can therefore be distinguished using internal signals. However, hallucinations have multifaceted causes: beyond simple knowledge gaps, they can emerge from training incentives that encourage models to exploit statistical shortcuts or spurious associations learned during pretraining. In this paper, we argue that when LLMs rely on such learned associations to produce hallucinations, their internal processes are mechanistically similar to those of factual recall, as both stem from strong statistical correlations encoded in the model's parameters. To verify this, we propose a novel taxonomy categorizing hallucinations into Unassociated Hallucinations (UHs), where outputs lack parametric grounding, and Associated Hallucinations (AHs), which are driven by spurious associations. Through mechanistic analysis, we compare their computational processes and hidden-state geometries with factually correct outputs. Our results show that hidden states primarily reflect whether the model is recalling parametric knowledge rather than the truthfulness of the output itself. Consequently, AHs exhibit hidden-state geometries that largely overlap with factual outputs, rendering standard detection methods ineffective. In contrast, UHs exhibit distinctive, clustered representations that facilitate reliable detection.
>
---
#### [replaced 065] EnvScaler: Scaling Tool-Interactive Environments for LLM Agent via Programmatic Synthesis
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出EnvScaler，用于生成可扩展的LLM工具交互环境。解决真实系统受限、模拟环境不稳定和手动构建难扩展的问题。通过程序化合成生成大量环境与任务场景，提升LLM在复杂任务中的表现。**

- **链接: [https://arxiv.org/pdf/2601.05808](https://arxiv.org/pdf/2601.05808)**

> **作者:** Xiaoshuai Song; Haofei Chang; Guanting Dong; Yutao Zhu; Ji-Rong Wen; Zhicheng Dou
>
> **备注:** Add some experiments
>
> **摘要:** Large language models (LLMs) are expected to be trained to act as agents in various real-world environments, but this process relies on rich and varied tool-interaction sandboxes. However, access to real systems is often restricted; LLM-simulated environments are prone to hallucinations and inconsistencies; and manually built sandboxes are hard to scale. In this paper, we propose EnvScaler, an automated framework for scalable tool-interaction environments via programmatic synthesis. EnvScaler comprises two components. First, SkelBuilder constructs diverse environment skeletons through topic mining, logic modeling, and quality evaluation. Then, ScenGenerator generates multiple task scenarios and rule-based trajectory validation functions for each environment. With EnvScaler, we synthesize 191 environments and about 7K scenarios, and apply them to Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) for Qwen3 series models. Results on three benchmarks show that EnvScaler significantly improves LLMs' ability to solve tasks in complex environments involving multi-turn, multi-tool interactions. We release our code and data at this https URL.
>
---
#### [replaced 066] Correcting Suppressed Log-Probabilities in Language Models with Post-Transformer Adapters
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型修正任务，解决政治敏感话题中事实概率被抑制的问题。通过训练后置适配器，恢复被压制的概率，提升生成文本的连贯性和真实性。**

- **链接: [https://arxiv.org/pdf/2604.14174](https://arxiv.org/pdf/2604.14174)**

> **作者:** Bryan Sanchez
>
> **备注:** 12 pages, 3 figures, code at this https URL
>
> **摘要:** Alignment-tuned language models frequently suppress factual log-probabilities on politically sensitive topics despite retaining the knowledge in their hidden representations. We show that a 786K-parameter (approximately 0.02% of the base model) post-transformer adapter, trained on frozen hidden states, corrects this suppression on 31 ideology-discriminating facts across Qwen3-4B, 8B, and 14B. The adapter memorizes all 15 training facts and generalizes to 11--39% of 16 held-out facts across 5 random splits per scale, with zero knowledge regressions via anchored training. Both gated (SwiGLU) and ungated (linear bottleneck) adapters achieve comparable results; neither consistently outperforms the other (Fisher exact p > 0.09 at all scales). On instruct models, the adapter corrects log-probability rankings. When applied at all token positions during generation, the adapter produces incoherent output; however, when applied only at the current prediction position (last-position-only), the adapter produces coherent, less censored text. A logit-space adapter operating after token projection fails to produce coherent generation at any application mode, suggesting hidden-state intervention is the correct level for generation correction. A previously undocumented silent gradient bug in Apple MLX explains all null results in earlier iterations of this work: the standard pattern nn.value_and_grad(model, fn)(this http URL()) returns zero gradients without error; the correct pattern nn.value_and_grad(model, fn)(model, data) resolves this. We provide a minimal reproduction and discuss implications for other adapter research using MLX.
>
---
#### [replaced 067] Spectral Tempering for Embedding Compression in Dense Passage Retrieval
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于密集段落检索任务，解决嵌入压缩中的维度缩减问题。提出Spectral Tempering方法，自适应调整谱缩放参数，提升压缩效果。**

- **链接: [https://arxiv.org/pdf/2603.19339](https://arxiv.org/pdf/2603.19339)**

> **作者:** Yongkang Li; Panagiotis Eustratiadis; Evangelos Kanoulas
>
> **备注:** This paper has been accepted as a short paper at SIGIR 2026
>
> **摘要:** Dimensionality reduction is critical for deploying dense retrieval systems at scale, yet mainstream post-hoc methods face a fundamental trade-off: principal component analysis (PCA) preserves dominant variance but underutilizes representational capacity, while whitening enforces isotropy at the cost of amplifying noise in the heavy-tailed eigenspectrum of retrieval embeddings. Intermediate spectral scaling methods unify these extremes by reweighting dimensions with a power coefficient $\gamma$, but treat $\gamma$ as a fixed hyperparameter that requires task-specific tuning. We show that the optimal scaling strength $\gamma$ is not a global constant: it varies systematically with target dimensionality $k$ and is governed by the signal-to-noise ratio (SNR) of the retained subspace. Based on this insight, we propose Spectral Tempering (\textbf{SpecTemp}), a learning-free method that derives an adaptive $\gamma(k)$ directly from the corpus eigenspectrum using local SNR analysis and knee-point normalization, requiring no labeled data or validation-based search. Extensive experiments demonstrate that Spectral Tempering consistently achieves near-oracle performance relative to grid-searched $\gamma^*(k)$ while remaining fully learning-free and model-agnostic. Our code is publicly available at this https URL.
>
---
#### [replaced 068] CRoCoDiL: Continuous and Robust Conditioned Diffusion for Language
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出CRoCoDiL模型，解决语言生成中的token依赖和语义不连贯问题。通过连续语义空间优化扩散过程，提升生成质量和速度。属于文本生成任务。**

- **链接: [https://arxiv.org/pdf/2603.20210](https://arxiv.org/pdf/2603.20210)**

> **作者:** Roy Uziel; Omer Belhasin; Itay Levy; Akhiad Bercovich; Ran El-Yaniv; Ran Zilberstein; Michael Elad
>
> **摘要:** Masked Diffusion Models (MDMs) provide an efficient non-causal alternative to autoregressive generation but often struggle with token dependencies and semantic incoherence due to their reliance on discrete marginal distributions. We address these limitations by shifting the diffusion process into a continuous sentence-level semantic space. We propose CRoCoDiL (Continuous and Robust Conditioned Diffusion for Language), a unified fine-tuning approach that jointly trains an encoder-demasker architecture, grounding the MDM demasking in continuous latent representations. This leads to the formation of a novel autoencoder in which decoding is obtained by an MDM algorithm. Relying on the same framework, we introduce two unconditional text synthesis algorithms: Continuous-Then-Discrete (ConThenDisc), a hybrid-diffusion approach that first generates latent representations in continuous space and then decodes these to tokens via an MDM, and Continuous-Within-Discrete (ConWithinDisc), a multi-diffusion strategy that refines latent representations throughout the discrete sampling process. Experiments using LLaDA show that our methods achieve superior generation quality and more than 10x faster sampling speeds in an unconditional setting.
>
---
#### [replaced 069] Faithfulness-Aware Uncertainty Quantification for Fact-Checking the Output of Retrieval Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于事实核查任务，解决RAG模型生成内容中的幻觉问题。提出FRANQ方法，通过区分事实性和证据一致性来更准确地检测错误。**

- **链接: [https://arxiv.org/pdf/2505.21072](https://arxiv.org/pdf/2505.21072)**

> **作者:** Ekaterina Fadeeva; Aleksandr Rubashevskii; Dzianis Piatrashyn; Roman Vashurin; Shehzaad Dhuliawala; Artem Shelmanov; Timothy Baldwin; Preslav Nakov; Mrinmaya Sachan; Maxim Panov
>
> **摘要:** Large Language Models (LLMs) enhanced with retrieval, an approach known as Retrieval-Augmented Generation (RAG), have achieved strong performance in open-domain question answering. However, RAG remains prone to hallucinations: factually incorrect outputs may arise from inaccuracies in the model's internal knowledge and the retrieved context. Existing approaches to mitigating hallucinations often conflate factuality with faithfulness to the retrieved evidence, incorrectly labeling factually correct statements as hallucinations if they are not explicitly supported by the retrieval. In this paper, we introduce FRANQ, a new method for hallucination detection in RAG outputs. FRANQ applies distinct uncertainty quantification (UQ) techniques to estimate factuality, conditioning on whether a statement is faithful to the retrieved context. To evaluate FRANQ and competing UQ methods, we construct a new long-form question answering dataset annotated for both factuality and faithfulness, combining automated labeling with manual validation of challenging cases. Extensive experiments across multiple datasets, tasks, and LLMs show that FRANQ achieves more accurate detection of factual errors in RAG-generated responses compared to existing approaches.
>
---
#### [replaced 070] CobwebTM: Probabilistic Concept Formation for Lifelong and Hierarchical Topic Modeling
- **分类: cs.CL**

- **简介: 该论文提出CobwebTM，解决主题建模中的终身学习与层次结构问题。通过增量概率概念形成，实现动态主题发现与组织。**

- **链接: [https://arxiv.org/pdf/2604.14489](https://arxiv.org/pdf/2604.14489)**

> **作者:** Karthik Singaravadivelan; Anant Gupta; Zekun Wang; Christopher J. MacLellan
>
> **备注:** 16 pages, 8 figures, 11 tables
>
> **摘要:** Topic modeling seeks to uncover latent semantic structure in text corpora with minimal supervision. Neural approaches achieve strong performance but require extensive tuning and struggle with lifelong learning due to catastrophic forgetting and fixed capacity, while classical probabilistic models lack flexibility and adaptability to streaming data. We introduce CobwebTM, a low-parameter lifelong hierarchical topic model based on incremental probabilistic concept formation. By adapting the Cobweb algorithm to continuous document embeddings, CobwebTM constructs semantic hierarchies online, enabling unsupervised topic discovery, dynamic topic creation, and hierarchical organization without predefining the number of topics. Across diverse datasets, CobwebTM achieves strong topic coherence, stable topics over time, and high-quality hierarchies, demonstrating that incremental symbolic concept formation combined with pretrained representations is an efficient approach to topic modeling.
>
---
#### [replaced 071] VLegal-Bench: Cognitively Grounded Benchmark for Vietnamese Legal Reasoning of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出VLegal-Bench，用于评估大语言模型在越南法律领域的推理能力。解决越南法律复杂性带来的评估难题，通过构建基准测试集提升AI法律系统的可靠性。**

- **链接: [https://arxiv.org/pdf/2512.14554](https://arxiv.org/pdf/2512.14554)**

> **作者:** Nguyen Tien Dong; Minh-Anh Nguyen; Thanh Dat Hoang; Nguyen Tuan Ngoc; Dao Xuan Quang Minh; Phan Phi Hai; Nguyen Thi Ngoc Anh; Dang Van Tu; Binh Vu
>
> **摘要:** The rapid advancement of large language models (LLMs) has enabled new possibilities for applying artificial intelligence within the legal domain. Nonetheless, the complexity, hierarchical organization, and frequent revisions of Vietnamese legislation pose considerable challenges for evaluating how well these models interpret and utilize legal knowledge. To address this gap, the Vietnamese Legal Benchmark (VLegal-Bench) is introduced, the first comprehensive benchmark designed to systematically assess LLMs on Vietnamese legal tasks. Informed by Bloom's cognitive taxonomy, VLegal-Bench encompasses multiple levels of legal understanding through tasks designed to reflect practical usage scenarios. The benchmark comprises 10,450 samples generated through a rigorous annotation pipeline, where legal experts label and cross-validate each instance using our annotation system to ensure every sample is grounded in authoritative legal documents and mirrors real-world legal assistant workflows, including general legal questions and answers, retrieval-augmented generation, multi-step reasoning, and scenario-based problem solving tailored to Vietnamese law. By providing a standardized, transparent, and cognitively informed evaluation framework, VLegal-Bench establishes a solid foundation for assessing LLM performance in Vietnamese legal contexts and supports the development of more reliable, interpretable, and ethically aligned AI-assisted legal systems. To facilitate access and reproducibility, we provide a public landing page for this benchmark at this https URL.
>
---
#### [replaced 072] The Amazing Agent Race: Strong Tool Users, Weak Navigators
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出AAR基准，用于评估LLM代理的工具使用和导航能力。针对现有线性基准的不足，AAR引入DAG结构 puzzles，揭示代理在导航上的缺陷。**

- **链接: [https://arxiv.org/pdf/2604.10261](https://arxiv.org/pdf/2604.10261)**

> **作者:** Zae Myung Kim; Dongseok Lee; Jaehyung Kim; Vipul Raheja; Dongyeop Kang
>
> **摘要:** Existing tool-use benchmarks for LLM agents are overwhelmingly linear: our analysis of six benchmarks shows 55 to 100% of instances are simple chains of 2 to 5 steps. We introduce The Amazing Agent Race (AAR), a benchmark featuring directed acyclic graph (DAG) puzzles (or "legs") with fork-merge tool chains. We release 1,400 instances across two variants: sequential (800 legs) and compositional (600 DAG legs). Agents must navigate Wikipedia, execute multi-step tool chains, and aggregate results into a verifiable answer. Legs are procedurally generated from Wikipedia seeds across four difficulty levels with live-API validation. Three complementary metrics (finish-line accuracy, pit-stop visit rate, and roadblock completion rate) separately diagnose navigation, tool-use, and arithmetic failures. Evaluating three agent frameworks on 1,400 legs, the best achieves only 37.2% accuracy. Navigation errors dominate (27 to 52% of trials) while tool-use errors remain below 17%, and agent architecture matters as much as model scale (Claude Code matches Codex CLI at 37% with 6x fewer tokens). The compositional structure of AAR reveals that agents fail not at calling tools but at navigating to the right pages, a blind spot invisible to linear benchmarks. The project page can be accessed at: this https URL
>
---
#### [replaced 073] Not All Tokens Matter: Towards Efficient LLM Reasoning via Token Significance in Reinforcement Learning
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的推理效率。针对模型生成冗长解释的问题，提出基于token重要性的强化学习方法，减少冗余并保持准确性。**

- **链接: [https://arxiv.org/pdf/2506.08125](https://arxiv.org/pdf/2506.08125)**

> **作者:** Hanbing Liu; Lang Cao; Yuanyi Ren; Mengyu Zhou; Haoyu Dong; Xiaojun Ma; Shi Han; Dongmei Zhang
>
> **摘要:** Large language models (LLMs) show strong reasoning abilities but often produce unnecessarily long explanations that reduce efficiency. Although reinforcement learning (RL) has been used to improve reasoning, most methods focus on accuracy and rely on uniform length-based rewards that overlook the differing contributions of individual tokens, often harming correctness. We revisit length optimization in RL through the perspective of token significance. Observing that many chain-of-thought (CoT) tokens contribute little to the final answer, we introduce a significance-aware length reward that selectively penalizes insignificance tokens, reducing redundancy while preserving essential reasoning. We also propose a dynamic length reward that encourages more detailed reasoning early in training and gradually shifts toward conciseness as learning progresses. Integrating these components into standard policy optimization yields a framework that improves both reasoning efficiency and accuracy. Experiments across multiple benchmarks demonstrate substantial reductions in response length while preserving or improving correctness, highlighting the importance of modeling token significance for efficient LLM reasoning.
>
---
#### [replaced 074] TRIDENT: Enhancing Large Language Model Safety with Tri-Dimensional Diversified Red-Teaming Data Synthesis
- **分类: cs.CL**

- **简介: 该论文属于语言模型安全任务，旨在提升模型生成内容的安全性。针对现有数据集风险覆盖不足的问题，提出TRIDENT框架和数据集，通过多维生成增强模型对恶意指令的防御能力。**

- **链接: [https://arxiv.org/pdf/2505.24672](https://arxiv.org/pdf/2505.24672)**

> **作者:** Xiaorui Wu; Xiaofeng Mao; Fei Li; Xin Zhang; Xuanhong Li; Chong Teng; Donghong Ji; Zhuang Li
>
> **摘要:** Large Language Models (LLMs) excel in various natural language processing tasks but remain vulnerable to generating harmful content or being exploited for malicious purposes. Although safety alignment datasets have been introduced to mitigate such risks through supervised fine-tuning (SFT), these datasets often lack comprehensive risk coverage. Most existing datasets focus primarily on lexical diversity while neglecting other critical dimensions. To address this limitation, we propose a novel analysis framework to systematically measure the risk coverage of alignment datasets across three essential dimensions: Lexical Diversity, Malicious Intent, and Jailbreak Tactics. We further introduce TRIDENT, an automated pipeline that leverages persona-based, zero-shot LLM generation to produce diverse and comprehensive instructions spanning these dimensions. Each harmful instruction is paired with an ethically aligned response, resulting in two datasets: TRIDENT-Core, comprising 26,311 examples, and TRIDENT-Edge, with 18,773 examples. Fine-tuning Llama 3.1-8B on TRIDENT-Edge demonstrates substantial improvements, achieving an average 14.29% reduction in Harm Score, and a 20% decrease in Attack Success Rate compared to the best-performing baseline model fine-tuned on the WildBreak dataset.
>
---
#### [replaced 075] Reward Modeling for Scientific Writing Evaluation
- **分类: cs.CL**

- **简介: 该论文属于科学写作评估任务，解决现有模型在科学领域评估中表现不佳的问题。提出一种高效、开源的奖励模型，通过两阶段训练提升评估能力。**

- **链接: [https://arxiv.org/pdf/2601.11374](https://arxiv.org/pdf/2601.11374)**

> **作者:** Furkan Şahinuç; Subhabrata Dutta; Iryna Gurevych
>
> **备注:** Accepted to ACL 2026 (Main). Project page: this https URL
>
> **摘要:** Scientific writing is an expert-domain task that demands deep domain knowledge, task-specific requirements and reasoning capabilities that leverage the domain knowledge to satisfy the task specifications. While scientific text generation has been widely studied, its evaluation remains a challenging and open problem. It is critical to develop models that can be reliably deployed for evaluating diverse open-ended scientific writing tasks while adhering to their distinct requirements. However, existing LLM-based judges and reward models are primarily optimized for general-purpose benchmarks with fixed scoring rubrics and evaluation criteria. Consequently, they often fail to reason over sparse knowledge of scientific domains when interpreting task-dependent and multi-faceted criteria. Moreover, fine-tuning for each individual task is costly and impractical for low-resource settings. To bridge these gaps, we propose cost-efficient, open-source reward models tailored for scientific writing evaluation. We introduce a two-stage training framework that initially optimizes scientific evaluation preferences and then refines reasoning capabilities. Our multi-aspect evaluation design and joint training across diverse tasks enable fine-grained assessment and robustness to dynamic criteria and scoring rubrics. Experimental analysis shows that our training regime strongly improves LLM-based scientific writing evaluation. Our models generalize effectively across tasks and to previously unseen scientific writing evaluation settings, allowing a single trained evaluator to be reused without task-specific retraining.
>
---
#### [replaced 076] Opportunities and Challenges of Large Language Models for Low-Resource Languages in Humanities Research
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决低资源语言研究中的数据与技术挑战。通过分析LLM的应用，提出跨学科合作与定制模型的解决方案。**

- **链接: [https://arxiv.org/pdf/2412.04497](https://arxiv.org/pdf/2412.04497)**

> **作者:** Tianyang Zhong; Zhenyuan Yang; Zhengliang Liu; Ruidong Zhang; Weihang You; Yiheng Liu; Haiyang Sun; Yi Pan; Yiwei Li; Yifan Zhou; Hanqi Jiang; Junhao Chen; Xiang Li; Tianming Liu
>
> **摘要:** Low-resource languages serve as invaluable repositories of human history, embodying cultural evolution and intellectual diversity. Despite their significance, these languages face critical challenges, including data scarcity and technological limitations, which hinder their comprehensive study and preservation. Recent advancements in large language models (LLMs) offer transformative opportunities for addressing these challenges, enabling innovative methodologies in linguistic, historical, and cultural research. This study systematically evaluates the applications of LLMs in low-resource language research, encompassing linguistic variation, historical documentation, cultural expressions, and literary analysis. By analyzing technical frameworks, current methodologies, and ethical considerations, this paper identifies key challenges such as data accessibility, model adaptability, and cultural sensitivity. Given the cultural, historical, and linguistic richness inherent in low-resource languages, this work emphasizes interdisciplinary collaboration and the development of customized models as promising avenues for advancing research in this domain. By underscoring the potential of integrating artificial intelligence with the humanities to preserve and study humanity's linguistic and cultural heritage, this study fosters global efforts towards safeguarding intellectual diversity.
>
---
#### [replaced 077] OSCBench: Benchmarking Object State Change in Text-to-Video Generation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于文本到视频生成任务，旨在解决对象状态变化（OSC）评估问题。提出OSCBench基准，评估模型在不同场景下的 OSC 性能。**

- **链接: [https://arxiv.org/pdf/2603.11698](https://arxiv.org/pdf/2603.11698)**

> **作者:** Xianjing Han; Bin Zhu; Shiqi Hu; Franklin Mingzhe Li; Patrick Carrington; Roger Zimmermann; Jingjing Chen
>
> **备注:** ACL 2026 Main Conference, Project page: this https URL
>
> **摘要:** Text-to-video (T2V) generation models have made rapid progress in producing visually high-quality and temporally coherent videos. However, existing benchmarks primarily focus on perceptual quality, text-video alignment, or physical plausibility, leaving a critical aspect of action understanding largely unexplored: object state change (OSC) explicitly specified in the text prompt. OSC refers to the transformation of an object's state induced by an action, such as peeling a potato or slicing a lemon. In this paper, we introduce OSCBench, a benchmark specifically designed to assess OSC performance in T2V models. OSCBench is constructed from instructional cooking data and systematically organizes action-object interactions into regular, novel, and compositional scenarios to probe both in-distribution performance and generalization. We evaluate six representative open-source and proprietary T2V models using both human user study and multimodal large language model (MLLM)-based automatic evaluation. Our results show that, despite strong performance on semantic and scene alignment, current T2V models consistently struggle with accurate and temporally consistent object state changes, especially in novel and compositional settings. These findings position OSC as a key bottleneck in text-to-video generation and establish OSCBench as a diagnostic benchmark for advancing state-aware video generation models.
>
---
#### [replaced 078] EvoTest: Evolutionary Test-Time Learning for Self-Improving Agentic Systems
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出EvoTest框架，解决AI代理在测试时无法自适应学习的问题。任务是提升代理在连续环境中的自我改进能力。通过进化机制优化代理配置，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2510.13220](https://arxiv.org/pdf/2510.13220)**

> **作者:** Yufei He; Juncheng Liu; Yue Liu; Yibo Li; Tri Cao; Zhiyuan Hu; Xinxing Xu; Bryan Hooi
>
> **备注:** ICLR 2026
>
> **摘要:** A fundamental limitation of current AI agents is their inability to learn complex skills on the fly at test time, often behaving like "clever but clueless interns" in novel environments. This severely limits their practical utility. To systematically measure and drive progress on this challenge, we first introduce the Jericho Test-Time Learning (J-TTL) benchmark. J-TTL is a new evaluation setup where an agent must play the same game for several consecutive episodes, attempting to improve its performance from one episode to the next. On J-TTL, we find that existing adaptation methods like reflection, memory, or reinforcement learning struggle. To address the challenges posed by our benchmark, we present EvoTest, an evolutionary test-time learning framework that improves an agent without any fine-tuning or gradients-by evolving the entire agentic system after every episode. EvoTest has two roles: the Actor Agent, which plays the game, and the Evolver Agent, which analyzes the episode transcript to propose a revised configuration for the next run. This configuration rewrites the prompt, updates memory by logging effective state-action choices, tunes hyperparameters, and learns the tool-use routines. On our J-TTL benchmark, EvoTest consistently increases performance, outperforming not only reflection and memory-only baselines but also more complex online fine-tuning methods. Notably, our method is the only one capable of winning two games (Detective and Library), while all baselines fail to win any.
>
---
#### [replaced 079] Follow the Flow: On Information Flow Across Textual Tokens in Text-to-Image Models
- **分类: cs.CL**

- **简介: 该论文属于文本到图像生成任务，旨在解决文本与图像对齐问题。通过分析文本中词符的语义分布和交互，发现信息集中于少数词符，提出编码阶段的简单干预可提升生成质量。**

- **链接: [https://arxiv.org/pdf/2504.01137](https://arxiv.org/pdf/2504.01137)**

> **作者:** Guy Kaplan; Michael Toker; Yuval Reif; Yonatan Belinkov; Roy Schwartz
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** Text-to-image generation models suffer from alignment problems, where generated images fail to accurately capture the objects and relations in the text prompt. Prior work has focused on improving alignment by refining the diffusion process, ignoring the role of the text encoder, which guides the diffusion. In this work, we investigate how semantic information is distributed across token representations in text-to-image prompts, analyzing it at two levels: (1) in-item representation-whether individual tokens represent their lexical item (i.e., a word or expression conveying a single concept), and (2) cross-item interaction-whether information flows between tokens of different lexical items. We use patching techniques to uncover encoding patterns, and find that information is usually concentrated in only one or two of the item's tokens; for example, in the item ``San Francisco's Golden Gate Bridge'', the token ``Gate'' sufficiently captures the entire expression while the other tokens could effectively be discarded. Lexical items also tend to remain isolated; for instance, in the prompt ``a green dog'', the token ``dog'' encodes no visual information about ``green''. However, in some cases, items do influence each other's representation, often leading to misinterpretations-e.g., in the prompt ``a pool by a table'', the token ``pool'' represents a ``pool table'' after contextualization. Our findings highlight the critical role of token-level encoding in image generation, and demonstrate that simple interventions at the encoding stage can substantially improve alignment and generation quality.
>
---
#### [replaced 080] Is this chart lying to me? Automating the detection of misleading visualizations
- **分类: cs.CL; cs.CV; cs.GR**

- **简介: 该论文属于检测误导性图表的任务，旨在解决虚假可视化传播 misinformation 的问题。工作包括构建真实和合成数据集，并评估多种模型的检测效果。**

- **链接: [https://arxiv.org/pdf/2508.21675](https://arxiv.org/pdf/2508.21675)**

> **作者:** Jonathan Tonglet; Jan Zimny; Tinne Tuytelaars; Iryna Gurevych
>
> **备注:** Camera-ready version accepted at ACL 2026 Main conference. Code and data available at: this https URL
>
> **摘要:** Misleading visualizations are a potent driver of misinformation on social media and the web. By violating chart design principles, they distort data and lead readers to draw inaccurate conclusions. Prior work has shown that both humans and multimodal large language models (MLLMs) are frequently deceived by such visualizations. Automatically detecting misleading visualizations and identifying the specific design rules they violate could help protect readers and reduce the spread of misinformation. However, the training and evaluation of AI models has been limited by the absence of large, diverse, and openly available datasets. In this work, we introduce Misviz, a benchmark of 2,604 real-world visualizations annotated with 12 types of misleaders. To support model training, we also create Misviz-synth, a synthetic dataset of 57,665 visualizations generated using Matplotlib and based on real-world data tables. We perform a comprehensive evaluation on both datasets using state-of-the-art MLLMs, rule-based systems, and image-axis classifiers. Our results reveal that the task remains highly challenging. We release Misviz, Misviz-synth, and the accompanying code.
>
---
#### [replaced 081] ChemAmp: Amplified Chemistry Tools via Composable Agents
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出ChemAmp，解决化学工具协作效率低的问题，通过动态组合工具构建超代理，提升任务性能并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2505.21569](https://arxiv.org/pdf/2505.21569)**

> **作者:** Zhucong Li; Powei Chang; Jin Xiao; Zhijian Zhou; Qianyu He; Jiaqing Liang; Fenglei Cao; Xu Yinghui; Yuan Qi
>
> **备注:** Accepted to ACL 2026 Findings ; Code available at this https URL
>
> **摘要:** Although LLM-based agents are proven to master tool orchestration in scientific fields, particularly chemistry, their single-task performance remains limited by underlying tool constraints. To this end, we propose tool amplification, a novel paradigm that enhances the collective capabilities of specialized tools through optimized, dynamic coordination within individual tasks. Instantiating this paradigm, we introduce ChemAmp, a computationally lightweight framework that dynamically treats chemistry tools (e.g., UniMol2, Chemformer) as composable building-block agents. It constructs task-specialized super-agents that transcend atomic tool constraints with limited data ($\leq$10 samples). Our evaluations across four core chemistry tasks molecular design, molecule captioning, reaction prediction, and property prediction demonstrate that ChemAmp outperforms chemistry-specialized models, generalist LLMs, and agent systems with tool orchestration. Critically, this bottom-up construction strategy enables 94\% inference token cost reductions versus vanilla multi-agent systems.
>
---
#### [replaced 082] Author-in-the-Loop Response Generation and Evaluation: Integrating Author Expertise and Intent in Responses to Peer Review
- **分类: cs.CL**

- **简介: 该论文属于作者参与的回应生成任务，解决如何整合作者专业知识与意图的问题。工作包括构建数据集、提出生成框架和评估体系。**

- **链接: [https://arxiv.org/pdf/2602.11173](https://arxiv.org/pdf/2602.11173)**

> **作者:** Qian Ruan; Iryna Gurevych
>
> **备注:** accepted to ACL 2026 Main Conference
>
> **摘要:** Author response (rebuttal) writing is a critical stage of scientific peer review that demands substantial author effort. In practice, authors possess domain expertise, author-only information, and response strategies - concrete forms of author expertise and intent - and seek NLP assistance that integrates these signals into author response generation (ARG). Yet this author-in-the-loop paradigm lacks formal NLP formulation and systematic study: no dataset provides fine-grained author signals, existing ARG work lacks author inputs and controls, and no evaluation measures response reflection of author signals and effectiveness in addressing reviewer concerns. To fill these gaps, we introduce (i) Re3Align, the first large-scale dataset of aligned review-response-revision triplets, where revisions proxy author signals; (ii) REspGen, an author-in-the-loop ARG framework supporting flexible author input, multi-attribute control, and evaluation-guided refinement; and (iii) REspEval, a comprehensive evaluation suite with 20+ metrics spanning input utilization, controllability, response quality, and discourse. Experiments with SOTA LLMs demonstrate the benefits of author input and evaluation-guided refinement, the impact of input specificity on response quality, and controllability-quality trade-offs. We release our dataset, generation and evaluation tools.
>
---
