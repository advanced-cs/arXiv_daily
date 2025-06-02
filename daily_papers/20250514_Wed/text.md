# 自然语言处理 cs.CL

- **最新发布 82 篇**

- **更新 42 篇**

## 最新发布

#### [new 001] Putting It All into Context: Simplifying Agents with LCLMs
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究语言模型代理在复杂任务（如SWE-bench）中的架构简化问题，属于自动化任务处理领域。针对现有方法过度依赖复杂框架（多工具、多代理）的现象，作者提出仅通过长上下文语言模型（LCLMs）整合环境信息并优化提示，证明无需复杂架构即可达到竞争性效果：Gemini-1.5-Pro获得38%解决率，Gemini-2.5-Pro达50.8%，两阶段混合方案达48.6%。**

- **链接: [http://arxiv.org/pdf/2505.08120v1](http://arxiv.org/pdf/2505.08120v1)**

> **作者:** Mingjian Jiang; Yangjun Ruan; Luis Lastras; Pavan Kapanipathi; Tatsunori Hashimoto
>
> **摘要:** Recent advances in language model (LM) agents have demonstrated significant potential for automating complex real-world tasks. To make progress on these difficult tasks, LM agent architectures have become increasingly complex, often incorporating multi-step retrieval tools, multiple agents, and scaffolding adapted to the underlying LM. In this work, we investigate whether all of this complexity is necessary, or if parts of these scaffolds can be removed on challenging tasks like SWE-bench. We show that in the case of SWE-bench, simply putting the entire environment into the context of a long context language model (LCLM) and properly prompting the model makes it competitive with carefully tuned, complex agent scaffolds. We show that a Gemini-1.5-Pro model without any scaffolding or tools achieves 38% on SWE-Bench-Verified, comparable with approaches using carefully tuned agent scaffolds (32%). While the unscaffolded approach with Gemini-1.5-Pro falls short of the strongest agentic architectures, we demonstrate that the more capable Gemini-2.5-Pro using the same unscaffolded approach directly attains a 50.8% solve rate. Additionally, a two-stage approach combining Gemini-1.5-Pro with Claude-3.7 achieves a competitive 48.6% solve rate.
>
---
#### [new 002] Automatic Task Detection and Heterogeneous LLM Speculative Decoding
- **分类: cs.CL; I.2.7**

- **简介: 该论文针对大模型推理加速，解决传统推测解码在多样化下游任务中效率与速度难以兼顾的问题。提出自动任务划分方法，将任务分配至异构草稿模型，结合任务数据对齐目标模型，并设计轻量级提示分类器动态路由。实验显示准确率提升6-50%，推理加速1.10-2.64倍。**

- **链接: [http://arxiv.org/pdf/2505.08600v1](http://arxiv.org/pdf/2505.08600v1)**

> **作者:** Danying Ge; Jianhua Gao; Qizhi Jiang; Yifei Feng; Weixing Ji
>
> **备注:** 10 pages, 10 figures, 2 tables
>
> **摘要:** Speculative decoding, which combines a draft model with a target model, has emerged as an effective approach to accelerate large language model (LLM) inference. However, existing methods often face a trade-off between the acceptance rate and decoding speed in downstream tasks due to the limited capacity of the draft model, making it difficult to ensure efficiency across diverse tasks. To address this problem, we propose a speculative decoding algorithm tailored for downstream task optimization. It includes an automatic task partitioning and assigning method, which automatically categorizes downstream tasks into different sub-tasks and assigns them to a set of heterogeneous draft models. Each draft model is aligned with the target model using task-specific data, thereby enhancing the consistency of inference results. In addition, our proposed method incorporates an online lightweight prompt classifier to dynamically route prompts to the appropriate draft model. Experimental results demonstrate that the proposed method improves draft accuracy by 6% to 50% over vanilla speculative decoding, while achieving a speedup of 1.10x to 2.64x in LLM inference.
>
---
#### [new 003] Joint Detection of Fraud and Concept Drift inOnline Conversations with LLM-Assisted Judgment
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究在线对话中的欺诈检测与概念漂移联合识别任务，旨在解决传统静态方法误判良性话题转换（概念漂移）为欺诈的问题。提出两阶段框架：集成分类模型初筛可疑对话，结合单类漂移检测器(OCDD)分析语义偏移，通过LLM判断漂移性质（欺诈/正常），并在社交工程数据集验证了检测精度与可解释性的提升。**

- **链接: [http://arxiv.org/pdf/2505.07852v1](http://arxiv.org/pdf/2505.07852v1)**

> **作者:** Ali Senol; Garima Agrawal; Huan Liu
>
> **摘要:** Detecting fake interactions in digital communication platforms remains a challenging and insufficiently addressed problem. These interactions may appear as harmless spam or escalate into sophisticated scam attempts, making it difficult to flag malicious intent early. Traditional detection methods often rely on static anomaly detection techniques that fail to adapt to dynamic conversational shifts. One key limitation is the misinterpretation of benign topic transitions referred to as concept drift as fraudulent behavior, leading to either false alarms or missed threats. We propose a two stage detection framework that first identifies suspicious conversations using a tailored ensemble classification model. To improve the reliability of detection, we incorporate a concept drift analysis step using a One Class Drift Detector (OCDD) to isolate conversational shifts within flagged dialogues. When drift is detected, a large language model (LLM) assesses whether the shift indicates fraudulent manipulation or a legitimate topic change. In cases where no drift is found, the behavior is inferred to be spam like. We validate our framework using a dataset of social engineering chat scenarios and demonstrate its practical advantages in improving both accuracy and interpretability for real time fraud detection. To contextualize the trade offs, we compare our modular approach against a Dual LLM baseline that performs detection and judgment using different language models.
>
---
#### [new 004] TiSpell: A Semi-Masked Methodology for Tibetan Spelling Correction covering Multi-Level Error with Data Augmentation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于藏语多级拼写纠错任务，旨在解决现有方法仅关注字符或音节单级纠错、缺乏整合及数据不足的问题。提出基于无标注文本的数据增强方法生成多级错误语料，并设计半掩码模型TiSpell，通过局部与全局上下文协同处理字符/音节错误，实验验证其效果优于基线并接近最优模型。**

- **链接: [http://arxiv.org/pdf/2505.08037v1](http://arxiv.org/pdf/2505.08037v1)**

> **作者:** Yutong Liu; Feng Xiao; Ziyue Zhang; Yongbin Yu; Cheng Huang; Fan Gao; Xiangxiang Wang; Ma-bao Ban; Manping Fan; Thupten Tsering; Cheng Huang; Gadeng Luosang; Renzeng Duojie; Nyima Tashi
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Multi-level Tibetan spelling correction addresses errors at both the character and syllable levels within a unified model. Existing methods focus mainly on single-level correction and lack effective integration of both levels. Moreover, there are no open-source datasets or augmentation methods tailored for this task in Tibetan. To tackle this, we propose a data augmentation approach using unlabeled text to generate multi-level corruptions, and introduce TiSpell, a semi-masked model capable of correcting both character- and syllable-level errors. Although syllable-level correction is more challenging due to its reliance on global context, our semi-masked strategy simplifies this process. We synthesize nine types of corruptions on clean sentences to create a robust training set. Experiments on both simulated and real-world data demonstrate that TiSpell, trained on our dataset, outperforms baseline models and matches the performance of state-of-the-art approaches, confirming its effectiveness.
>
---
#### [new 005] Implementing Long Text Style Transfer with LLMs through Dual-Layered Sentence and Paragraph Structure Extraction and Mapping
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究长文本风格迁移任务，解决零样本LLMs在保持段落结构连贯和语义一致性的问题。提出分层框架ZeroStylus，通过句子级风格适配和段落级结构提取，动态构建模板库实现上下文感知转换。实验证明其优于基线方法，无需微调或平行语料。**

- **链接: [http://arxiv.org/pdf/2505.07888v1](http://arxiv.org/pdf/2505.07888v1)**

> **作者:** Yusen Wu; Xiaotie Deng
>
> **摘要:** This paper addresses the challenge in long-text style transfer using zero-shot learning of large language models (LLMs), proposing a hierarchical framework that combines sentence-level stylistic adaptation with paragraph-level structural coherence. We argue that in the process of effective paragraph-style transfer, to preserve the consistency of original syntactic and semantic information, it is essential to perform style transfer not only at the sentence level but also to incorporate paragraph-level semantic considerations, while ensuring structural coherence across inter-sentential relationships. Our proposed framework, ZeroStylus, operates through two systematic phases: hierarchical template acquisition from reference texts and template-guided generation with multi-granular matching. The framework dynamically constructs sentence and paragraph template repositories, enabling context-aware transformations while preserving inter-sentence logical relationships. Experimental evaluations demonstrate significant improvements over baseline methods, with structured rewriting achieving 6.90 average score compared to 6.70 for direct prompting approaches in tri-axial metrics assessing style consistency, content preservation, and expression quality. Ablation studies validate the necessity of both template hierarchies during style transfer, showing higher content preservation win rate against sentence-only approaches through paragraph-level structural encoding, as well as direct prompting method through sentence-level pattern extraction and matching. The results establish new capabilities for coherent long-text style transfer without requiring parallel corpora or LLM fine-tuning.
>
---
#### [new 006] Adaptive Schema-aware Event Extraction with Retrieval-Augmented Generation
- **分类: cs.CL; I.2.7**

- **简介: 该论文研究事件抽取任务，解决现有系统模式僵化及评估基准缺失问题。提出ASEE方法，结合模式检索与生成技术动态适配事件模式，并构建多维度基准MD-SEE验证效果。新方法显著提升了跨领域事件抽取的准确性。**

- **链接: [http://arxiv.org/pdf/2505.08690v1](http://arxiv.org/pdf/2505.08690v1)**

> **作者:** Sheng Liang; Hang Lv; Zhihao Wen; Yaxiong Wu; Yongyue Zhang; Hao Wang; Yong Liu
>
> **备注:** 15 pages, 3 figures
>
> **摘要:** Event extraction (EE) is a fundamental task in natural language processing (NLP) that involves identifying and extracting event information from unstructured text. Effective EE in real-world scenarios requires two key steps: selecting appropriate schemas from hundreds of candidates and executing the extraction process. Existing research exhibits two critical gaps: (1) the rigid schema fixation in existing pipeline systems, and (2) the absence of benchmarks for evaluating joint schema matching and extraction. Although large language models (LLMs) offer potential solutions, their schema hallucination tendencies and context window limitations pose challenges for practical deployment. In response, we propose Adaptive Schema-aware Event Extraction (ASEE), a novel paradigm combining schema paraphrasing with schema retrieval-augmented generation. ASEE adeptly retrieves paraphrased schemas and accurately generates targeted structures. To facilitate rigorous evaluation, we construct the Multi-Dimensional Schema-aware Event Extraction (MD-SEE) benchmark, which systematically consolidates 12 datasets across diverse domains, complexity levels, and language settings. Extensive evaluations on MD-SEE show that our proposed ASEE demonstrates strong adaptability across various scenarios, significantly improving the accuracy of event extraction.
>
---
#### [new 007] Exploiting Text Semantics for Few and Zero Shot Node Classification on Text-attributed Graph
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究文本属性图（TAG）的少样本/零样本节点分类任务，旨在解决现有方法过度依赖图增强而忽视文本语义的问题。提出文本语义增强方法TSA，通过正语义匹配（检索相似文本）和负语义对比（构造对立语义文本）强化监督信号，在5个数据集上超越13个基线模型，准确率提升超5%。**

- **链接: [http://arxiv.org/pdf/2505.08168v1](http://arxiv.org/pdf/2505.08168v1)**

> **作者:** Yuxiang Wang; Xiao Yan; Shiyu Jin; Quanqing Xu; Chuang Hu; Yuanyuan Zhu; Bo Du; Jia Wu; Jiawei Jiang
>
> **摘要:** Text-attributed graph (TAG) provides a text description for each graph node, and few- and zero-shot node classification on TAGs have many applications in fields such as academia and social networks. Existing work utilizes various graph-based augmentation techniques to train the node and text embeddings, while text-based augmentations are largely unexplored. In this paper, we propose Text Semantics Augmentation (TSA) to improve accuracy by introducing more text semantic supervision signals. Specifically, we design two augmentation techniques, i.e., positive semantics matching and negative semantics contrast, to provide more reference texts for each graph node or text description. Positive semantic matching retrieves texts with similar embeddings to match with a graph node. Negative semantic contrast adds a negative prompt to construct a text description with the opposite semantics, which is contrasted with the original node and text. We evaluate TSA on 5 datasets and compare with 13 state-of-the-art baselines. The results show that TSA consistently outperforms all baselines, and its accuracy improvements over the best-performing baseline are usually over 5%.
>
---
#### [new 008] HYPERNYM MERCURY: Token Optimization through Semantic Field Constriction and Reconstruction from Hypernyms. A New Text Compression Method
- **分类: cs.CL**

- **简介: 该论文提出一种新型文本压缩方法（HYPERNYM MERCURY），通过语义场收缩与上位词重构实现词级语义压缩，属于NLP领域的计算优化任务。解决大语言模型提示中标记冗余问题，能在保持90%以上语义相似度的前提下减少标记量，支持无损压缩与粒度控制，并通过多体裁文本实验验证有效性。**

- **链接: [http://arxiv.org/pdf/2505.08058v1](http://arxiv.org/pdf/2505.08058v1)**

> **作者:** Chris Forrester; Octavia Sulea
>
> **摘要:** Compute optimization using token reduction of LLM prompts is an emerging task in the fields of NLP and next generation, agentic AI. In this white paper, we introduce a novel (patent pending) text representation scheme and a first-of-its-kind word-level semantic compression of paragraphs that can lead to over 90\% token reduction, while retaining high semantic similarity to the source text. We explain how this novel compression technique can be lossless and how the detail granularity is controllable. We discuss benchmark results over open source data (i.e. Bram Stoker's Dracula available through Project Gutenberg) and show how our results hold at the paragraph level, across multiple genres and models.
>
---
#### [new 009] Are We Paying Attention to Her? Investigating Gender Disambiguation and Attention in Machine Translation
- **分类: cs.CL**

- **简介: 该论文研究机器翻译中的性别消歧任务，解决传统指标无法评估模型对上下文性别线索利用的问题。提出最小对准确率(MPA)指标，验证主流模型倾向统计性别刻板印象，并分析编码器注意力机制对性别线索的差异化响应（男性分散、女性集中）。**

- **链接: [http://arxiv.org/pdf/2505.08546v1](http://arxiv.org/pdf/2505.08546v1)**

> **作者:** Chiara Manna; Afra Alishahi; Frédéric Blain; Eva Vanmassenhove
>
> **摘要:** While gender bias in modern Neural Machine Translation (NMT) systems has received much attention, traditional evaluation metrics do not to fully capture the extent to which these systems integrate contextual gender cues. We propose a novel evaluation metric called Minimal Pair Accuracy (MPA), which measures the reliance of models on gender cues for gender disambiguation. MPA is designed to go beyond surface-level gender accuracy metrics by focusing on whether models adapt to gender cues in minimal pairs -- sentence pairs that differ solely in the gendered pronoun, namely the explicit indicator of the target's entity gender in the source language (EN). We evaluate a number of NMT models on the English-Italian (EN--IT) language pair using this metric, we show that they ignore available gender cues in most cases in favor of (statistical) stereotypical gender interpretation. We further show that in anti-stereotypical cases, these models tend to more consistently take masculine gender cues into account while ignoring the feminine cues. Furthermore, we analyze the attention head weights in the encoder component and show that while all models encode gender information to some extent, masculine cues elicit a more diffused response compared to the more concentrated and specialized responses to feminine gender cues.
>
---
#### [new 010] NurValues: Real-World Nursing Values Evaluation for Large Language Models in Clinical Context
- **分类: cs.CL; 68T50; I.2.7**

- **简介: 该论文属于价值观对齐评估任务，旨在解决大语言模型在临床护理场景中与专业伦理脱节的问题。研究者构建首个护理价值观基准NurValues，通过真实案例与对抗样本构建双难度数据集，评估23个主流模型，发现模型在正义维度表现最弱，上下文学习可显著提升伦理对齐，为临床AI开发提供价值敏感基准。**

- **链接: [http://arxiv.org/pdf/2505.08734v1](http://arxiv.org/pdf/2505.08734v1)**

> **作者:** Ben Yao; Qiuchi Li; Yazhou Zhang; Siyu Yang; Bohan Zhang; Prayag Tiwari; Jing Qin
>
> **备注:** 25 pages, 10 figures, 16 tables
>
> **摘要:** This work introduces the first benchmark for nursing value alignment, consisting of five core value dimensions distilled from international nursing codes: Altruism, Human Dignity, Integrity, Justice, and Professionalism. The benchmark comprises 1,100 real-world nursing behavior instances collected through a five-month longitudinal field study across three hospitals of varying tiers. These instances are annotated by five clinical nurses and then augmented with LLM-generated counterfactuals with reversed ethic polarity. Each original case is paired with a value-aligned and a value-violating version, resulting in 2,200 labeled instances that constitute the Easy-Level dataset. To increase adversarial complexity, each instance is further transformed into a dialogue-based format that embeds contextual cues and subtle misleading signals, yielding a Hard-Level dataset. We evaluate 23 state-of-the-art (SoTA) LLMs on their alignment with nursing values. Our findings reveal three key insights: (1) DeepSeek-V3 achieves the highest performance on the Easy-Level dataset (94.55), where Claude 3.5 Sonnet outperforms other models on the Hard-Level dataset (89.43), significantly surpassing the medical LLMs; (2) Justice is consistently the most difficult nursing value dimension to evaluate; and (3) in-context learning significantly improves alignment. This work aims to provide a foundation for value-sensitive LLMs development in clinical settings. The dataset and the code are available at https://huggingface.co/datasets/Ben012345/NurValues.
>
---
#### [new 011] A Head to Predict and a Head to Question: Pre-trained Uncertainty Quantification Heads for Hallucination Detection in LLM Outputs
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型（LLMs）生成虚假信息（幻觉）的检测问题，提出预训练不确定性量化（UQ）头部模块。通过监督训练结合注意力特征，增强模型对输出可靠性的评估能力，实现在跨领域、跨语言的幻觉检测任务中达到最优性能，并公开了适配主流模型的预训练UQ模块。**

- **链接: [http://arxiv.org/pdf/2505.08200v1](http://arxiv.org/pdf/2505.08200v1)**

> **作者:** Artem Shelmanov; Ekaterina Fadeeva; Akim Tsvigun; Ivan Tsvigun; Zhuohan Xie; Igor Kiselev; Nico Daheim; Caiqi Zhang; Artem Vazhentsev; Mrinmaya Sachan; Preslav Nakov; Timothy Baldwin
>
> **摘要:** Large Language Models (LLMs) have the tendency to hallucinate, i.e., to sporadically generate false or fabricated information. This presents a major challenge, as hallucinations often appear highly convincing and users generally lack the tools to detect them. Uncertainty quantification (UQ) provides a framework for assessing the reliability of model outputs, aiding in the identification of potential hallucinations. In this work, we introduce pre-trained UQ heads: supervised auxiliary modules for LLMs that substantially enhance their ability to capture uncertainty compared to unsupervised UQ methods. Their strong performance stems from the powerful Transformer architecture in their design and informative features derived from LLM attention maps. Experimental evaluation shows that these heads are highly robust and achieve state-of-the-art performance in claim-level hallucination detection across both in-domain and out-of-domain prompts. Moreover, these modules demonstrate strong generalization to languages they were not explicitly trained on. We pre-train a collection of UQ heads for popular LLM series, including Mistral, Llama, and Gemma 2. We publicly release both the code and the pre-trained heads.
>
---
#### [new 012] AC-Reason: Towards Theory-Guided Actual Causality Reasoning with Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对现有大模型在实际因果推理中缺乏理论支撑、可解释性差的问题，提出AC-Reason半形式化推理框架。通过分析因果要素（充分性/必要性/常态性）并设计理论驱动算法，结合新构建的AC-Bench测试集（含千条标注样本），显著提升LLM在因果归因任务中的准确率（如GPT-4达75.04%），同时揭示了主流模型推理的忠实性差异。**

- **链接: [http://arxiv.org/pdf/2505.08750v1](http://arxiv.org/pdf/2505.08750v1)**

> **作者:** Yanxi Zhang; Xin Cong; Zhong Zhang; Xiao Liu; Dongyan Zhao; Yesai Wu
>
> **摘要:** Actual causality (AC), a fundamental aspect of causal reasoning (CR), is responsible for attribution and responsibility assignment in real-world scenarios. However, existing LLM-based methods lack grounding in formal AC theory, resulting in limited interpretability. Therefore, we propose AC-Reason, a semi-formal reasoning framework that identifies causally relevant events within an AC scenario, infers the values of their formal causal factors (e.g., sufficiency, necessity, and normality), and answers AC queries via a theory-guided algorithm with explanations. While AC-Reason does not explicitly construct a causal graph, it operates over variables in the underlying causal structure to support principled reasoning. To enable comprehensive evaluation, we introduce AC-Bench, a new benchmark built upon and substantially extending Big-Bench Hard Causal Judgment (BBH-CJ). AC-Bench comprises ~1K carefully annotated samples, each with detailed reasoning steps and focuses solely on actual causation. The case study shows that synthesized samples in AC-Bench present greater challenges for LLMs. Extensive experiments on BBH-CJ and AC-Bench show that AC-Reason consistently improves LLM performance over baselines. On BBH-CJ, all tested LLMs surpass the average human rater accuracy of 69.60%, with GPT-4 + AC-Reason achieving 75.04%. On AC-Bench, GPT-4 + AC-Reason again achieves the highest accuracy of 71.82%. AC-Bench further enables fine-grained analysis of reasoning faithfulness, revealing that only Qwen-2.5-72B-Instruct, Claude-3.5-Sonnet, and GPT-4o exhibit faithful reasoning, whereas GPT-4 tends to exploit shortcuts. Finally, our ablation study proves that integrating AC theory into LLMs is highly effective, with the proposed algorithm contributing the most significant performance gains.
>
---
#### [new 013] FalseReject: A Resource for Improving Contextual Safety and Mitigating Over-Refusals in LLMs via Structured Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大语言模型（LLMs）因安全对齐导致的过度拒绝良性查询问题，提出资源FalseReject，包含16k安全相关结构化问答数据。通过图驱动的多智能体对抗框架生成多样化提示，并利用显式推理增强模型对安全语境的判别力。研究构建了训练集和测试基准，实验表明其能有效减少误拒且不降低安全性，属于LLM安全优化任务。**

- **链接: [http://arxiv.org/pdf/2505.08054v1](http://arxiv.org/pdf/2505.08054v1)**

> **作者:** Zhehao Zhang; Weijie Xu; Fanyou Wu; Chandan K. Reddy
>
> **摘要:** Safety alignment approaches in large language models (LLMs) often lead to the over-refusal of benign queries, significantly diminishing their utility in sensitive scenarios. To address this challenge, we introduce FalseReject, a comprehensive resource containing 16k seemingly toxic queries accompanied by structured responses across 44 safety-related categories. We propose a graph-informed adversarial multi-agent interaction framework to generate diverse and complex prompts, while structuring responses with explicit reasoning to aid models in accurately distinguishing safe from unsafe contexts. FalseReject includes training datasets tailored for both standard instruction-tuned models and reasoning-oriented models, as well as a human-annotated benchmark test set. Our extensive benchmarking on 29 state-of-the-art (SOTA) LLMs reveals persistent over-refusal challenges. Empirical results demonstrate that supervised finetuning with FalseReject substantially reduces unnecessary refusals without compromising overall safety or general language capabilities.
>
---
#### [new 014] Development of a WAZOBIA-Named Entity Recognition System
- **分类: cs.CL; cs.HC; cs.IR; cs.LG**

- **简介: 该论文属于命名实体识别任务，旨在解决非洲语言（豪萨、约鲁巴、伊博语）缺乏专用NER系统的问题。通过构建标注数据集，结合CRF、BiLSTM、BERT-RNN等模型及OCR技术，开发多语言NER工具，验证其在人名、机构、地点识别中的有效性，证明利用现有框架可为资源稀缺语言构建高效系统。**

- **链接: [http://arxiv.org/pdf/2505.07884v1](http://arxiv.org/pdf/2505.07884v1)**

> **作者:** S. E Emedem; I. E Onyenwe; E. G Onyedinma
>
> **备注:** 6 pages, 3 figures, 1 table
>
> **摘要:** Named Entity Recognition NER is very crucial for various natural language processing applications, including information extraction, machine translation, and sentiment analysis. Despite the ever-increasing interest in African languages within computational linguistics, existing NER systems focus mainly on English, European, and a few other global languages, leaving a significant gap for under-resourced languages. This research presents the development of a WAZOBIA-NER system tailored for the three most prominent Nigerian languages: Hausa, Yoruba, and Igbo. This research begins with a comprehensive compilation of annotated datasets for each language, addressing data scarcity and linguistic diversity challenges. Exploring the state-of-the-art machine learning technique, Conditional Random Fields (CRF) and deep learning models such as Bidirectional Long Short-Term Memory (BiLSTM), Bidirectional Encoder Representation from Transformers (Bert) and fine-tune with a Recurrent Neural Network (RNN), the study evaluates the effectiveness of these approaches in recognizing three entities: persons, organizations, and locations. The system utilizes optical character recognition (OCR) technology to convert textual images into machine-readable text, thereby enabling the Wazobia system to accept both input text and textual images for extraction purposes. The system achieved a performance of 0.9511 in precision, 0.9400 in recall, 0.9564 in F1-score, and 0.9301 in accuracy. The model's evaluation was conducted across three languages, with precision, recall, F1-score, and accuracy as key assessment metrics. The Wazobia-NER system demonstrates that it is feasible to build robust NER tools for under-resourced African languages using current NLP frameworks and transfer learning.
>
---
#### [new 015] Scalable LLM Math Reasoning Acceleration with Low-rank Distillation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型加速任务，旨在解决高效推理方法导致的大语言模型数学推理能力下降问题。提出Caprese方法，通过低秩蒸馏在保留原参数的基础上增加少量参数（约1%），用2万合成样本恢复数学能力，同时减少激活参数（如Gemma 2B削减）和生成延迟（Qwen延迟降11%），且不影响语言任务性能。**

- **链接: [http://arxiv.org/pdf/2505.07861v1](http://arxiv.org/pdf/2505.07861v1)**

> **作者:** Harry Dong; Bilge Acun; Beidi Chen; Yuejie Chi
>
> **摘要:** Due to long generations, large language model (LLM) math reasoning demands significant computational resources and time. While many existing efficient inference methods have been developed with excellent performance preservation on language tasks, they often severely degrade math performance. In this paper, we propose Caprese, a low-cost distillation method to recover lost capabilities from deploying efficient inference methods, focused primarily in feedforward blocks. With original weights unperturbed, roughly 1% of additional parameters, and only 20K synthetic training samples, we are able to recover much if not all of the math capabilities lost from efficient inference for thinking LLMs and without harm to language tasks for instruct LLMs. Moreover, Caprese slashes the number of active parameters (~2B cut for Gemma 2 9B and Llama 3.1 8B) and integrates cleanly into existing model layers to reduce latency (>11% reduction to generate 2048 tokens with Qwen 2.5 14B) while encouraging response brevity.
>
---
#### [new 016] CrashSage: A Large Language Model-Centered Framework for Contextual and Interpretable Traffic Crash Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于交通事故分析的文本建模任务，旨在解决传统方法忽略语义关联、信息损失大的问题。提出CrashSage框架：通过表格转文本构建结构化叙事，数据增强提升连贯性，微调LLaMA3-8B实现事故严重性推理（优于基线模型），并运用可解释技术揭示关键风险因素。**

- **链接: [http://arxiv.org/pdf/2505.07853v1](http://arxiv.org/pdf/2505.07853v1)**

> **作者:** Hao Zhen; Jidong J. Yang
>
> **备注:** 20 pages, 7 figures
>
> **摘要:** Road crashes claim over 1.3 million lives annually worldwide and incur global economic losses exceeding \$1.8 trillion. Such profound societal and financial impacts underscore the urgent need for road safety research that uncovers crash mechanisms and delivers actionable insights. Conventional statistical models and tree ensemble approaches typically rely on structured crash data, overlooking contextual nuances and struggling to capture complex relationships and underlying semantics. Moreover, these approaches tend to incur significant information loss, particularly in narrative elements related to multi-vehicle interactions, crash progression, and rare event characteristics. This study presents CrashSage, a novel Large Language Model (LLM)-centered framework designed to advance crash analysis and modeling through four key innovations. First, we introduce a tabular-to-text transformation strategy paired with relational data integration schema, enabling the conversion of raw, heterogeneous crash data into enriched, structured textual narratives that retain essential structural and relational context. Second, we apply context-aware data augmentation using a base LLM model to improve narrative coherence while preserving factual integrity. Third, we fine-tune the LLaMA3-8B model for crash severity inference, demonstrating superior performance over baseline approaches, including zero-shot, zero-shot with chain-of-thought prompting, and few-shot learning, with multiple models (GPT-4o, GPT-4o-mini, LLaMA3-70B). Finally, we employ a gradient-based explainability technique to elucidate model decisions at both the individual crash level and across broader risk factor dimensions. This interpretability mechanism enhances transparency and enables targeted road safety interventions by providing deeper insights into the most influential factors.
>
---
#### [new 017] ALOHA: Empowering Multilingual Agent for University Orientation with Hierarchical Retrieval
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言信息检索任务，旨在解决现有LLMs和搜索引擎在校园信息查询中缺乏领域知识、多语言支持不足的问题。作者提出ALOHA系统，通过分层检索增强响应准确性，整合外部API提供多语言交互服务，经评估证实其优于商业产品，已服务超1.2万人。**

- **链接: [http://arxiv.org/pdf/2505.08130v1](http://arxiv.org/pdf/2505.08130v1)**

> **作者:** Mingxu Tao; Bowen Tang; Mingxuan Ma; Yining Zhang; Hourun Li; Feifan Wen; Hao Ma; Jia Yang
>
> **备注:** To appear in NAACL 2025 Demo Track
>
> **摘要:** The rise of Large Language Models~(LLMs) revolutionizes information retrieval, allowing users to obtain required answers through complex instructions within conversations. However, publicly available services remain inadequate in addressing the needs of faculty and students to search campus-specific information. It is primarily due to the LLM's lack of domain-specific knowledge and the limitation of search engines in supporting multilingual and timely scenarios. To tackle these challenges, we introduce ALOHA, a multilingual agent enhanced by hierarchical retrieval for university orientation. We also integrate external APIs into the front-end interface to provide interactive service. The human evaluation and case study show our proposed system has strong capabilities to yield correct, timely, and user-friendly responses to the queries in multiple languages, surpassing commercial chatbots and search engines. The system has been deployed and has provided service for more than 12,000 people.
>
---
#### [new 018] Hakim: Farsi Text Embedding Model
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理中的文本嵌入任务，针对波斯语嵌入模型研究不足的问题，提出新型模型Hakim。通过构建三个新数据集和基于BERT/RetroMAE的架构，实现了FaMTEB基准8.5%的性能提升，优化了聊天机器人、检索增强生成系统的历史上下文处理能力，建立了波斯语理解的新基准。**

- **链接: [http://arxiv.org/pdf/2505.08435v1](http://arxiv.org/pdf/2505.08435v1)**

> **作者:** Mehran Sarmadi; Morteza Alikhani; Erfan Zinvandi; Zahra Pourbahman
>
> **摘要:** Recent advancements in text embedding have significantly improved natural language understanding across many languages, yet Persian remains notably underrepresented in large-scale embedding research. In this paper, we present Hakim, a novel state-of-the-art Persian text embedding model that achieves a 8.5% performance improvement over existing approaches on the FaMTEB benchmark, outperforming all previously developed Persian language models. As part of this work, we introduce three new datasets - Corpesia, Pairsia-sup, and Pairsia-unsup - to support supervised and unsupervised training scenarios. Additionally, Hakim is designed for applications in chatbots and retrieval-augmented generation (RAG) systems, particularly addressing retrieval tasks that require incorporating message history within these systems. We also propose a new baseline model built on the BERT architecture. Our language model consistently achieves higher accuracy across various Persian NLP tasks, while the RetroMAE-based model proves particularly effective for textual information retrieval applications. Together, these contributions establish a new foundation for advancing Persian language understanding.
>
---
#### [new 019] Recovering Event Probabilities from Large Language Model Embeddings via Axiomatic Constraints
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究从大语言模型嵌入中恢复符合概率论公理的连贯事件概率，解决LLM生成概率违反概率公理的问题。通过扩展变分自编码器在潜在空间施加概率可加性等约束，使嵌入重构时自然生成合理概率。实验表明该方法在互补事件中恢复的概率比LLM直接输出的更准确、更接近真实分布。**

- **链接: [http://arxiv.org/pdf/2505.07883v1](http://arxiv.org/pdf/2505.07883v1)**

> **作者:** Jian-Qiao Zhu; Haijiang Yan; Thomas L. Griffiths
>
> **摘要:** Rational decision-making under uncertainty requires coherent degrees of belief in events. However, event probabilities generated by Large Language Models (LLMs) have been shown to exhibit incoherence, violating the axioms of probability theory. This raises the question of whether coherent event probabilities can be recovered from the embeddings used by the models. If so, those derived probabilities could be used as more accurate estimates in events involving uncertainty. To explore this question, we propose enforcing axiomatic constraints, such as the additive rule of probability theory, in the latent space learned by an extended variational autoencoder (VAE) applied to LLM embeddings. This approach enables event probabilities to naturally emerge in the latent space as the VAE learns to both reconstruct the original embeddings and predict the embeddings of semantically related events. We evaluate our method on complementary events (i.e., event A and its complement, event not-A), where the true probabilities of the two events must sum to 1. Experiment results on open-weight language models demonstrate that probabilities recovered from embeddings exhibit greater coherence than those directly reported by the corresponding models and align closely with the true probabilities.
>
---
#### [new 020] Large Language Models and Arabic Content: A Review
- **分类: cs.CL; cs.AI**

- **简介: 该论文为综述研究，探讨大语言模型（LLMs）在阿拉伯语自然语言处理中的应用。针对阿拉伯语资源稀缺、语言结构复杂等问题，梳理了预训练模型在多种任务与方言中的成果，分析了微调、提示工程等技术对性能的优化，并汇总了常用数据集与持续增长的应用趋势。**

- **链接: [http://arxiv.org/pdf/2505.08004v1](http://arxiv.org/pdf/2505.08004v1)**

> **作者:** Haneh Rhel; Dmitri Roussinov
>
> **备注:** Original language: English This paper has been submitted to the First International Conference on Artificial Intelligence and Generative AI (FICAILY 2025), and it has been accepted for presentation at FICAILY on 9-10/July 2025 and for publication in the Springer Nature. Number of pages: 16 Publication status Accepted/In press - 7 Apr 2025 https://www.gena-ai-libya2025.com/
>
> **摘要:** Over the past three years, the rapid advancement of Large Language Models (LLMs) has had a profound impact on multiple areas of Artificial Intelligence (AI), particularly in Natural Language Processing (NLP) across diverse languages, including Arabic. Although Arabic is considered one of the most widely spoken languages across 27 countries in the Arabic world and used as a second language in some other non-Arabic countries as well, there is still a scarcity of Arabic resources, datasets, and tools. Arabic NLP tasks face various challenges due to the complexities of the Arabic language, including its rich morphology, intricate structure, and diverse writing standards, among other factors. Researchers have been actively addressing these challenges, demonstrating that pre-trained Large Language Models (LLMs) trained on multilingual corpora achieve significant success in various Arabic NLP tasks. This study provides an overview of using large language models (LLMs) for the Arabic language, highlighting early pre-trained Arabic Language models across various NLP applications and their ability to handle diverse Arabic content tasks and dialects. It also provides an overview of how techniques like finetuning and prompt engineering can enhance the performance of these models. Additionally, the study summarizes common Arabic benchmarks and datasets while presenting our observations on the persistent upward trend in the adoption of LLMs.
>
---
#### [new 021] SEM: Reinforcement Learning for Search-Efficient Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型（LLMs）搜索决策优化任务，旨在解决模型冗余调用搜索引擎导致的效率低下问题。提出SEM强化学习框架，通过平衡数据集训练模型区分自主回答与检索场景，结合结构化推理模板和GRPO策略优化搜索行为，在保证精度的同时显著降低无效检索。**

- **链接: [http://arxiv.org/pdf/2505.07903v1](http://arxiv.org/pdf/2505.07903v1)**

> **作者:** Zeyang Sha; Shiwen Cui; Weiqiang Wang
>
> **摘要:** Recent advancements in Large Language Models(LLMs) have demonstrated their capabilities not only in reasoning but also in invoking external tools, particularly search engines. However, teaching models to discern when to invoke search and when to rely on their internal knowledge remains a significant challenge. Existing reinforcement learning approaches often lead to redundant search behaviors, resulting in inefficiencies and over-cost. In this paper, we propose SEM, a novel post-training reinforcement learning framework that explicitly trains LLMs to optimize search usage. By constructing a balanced dataset combining MuSiQue and MMLU, we create scenarios where the model must learn to distinguish between questions it can answer directly and those requiring external retrieval. We design a structured reasoning template and employ Group Relative Policy Optimization(GRPO) to post-train the model's search behaviors. Our reward function encourages accurate answering without unnecessary search while promoting effective retrieval when needed. Experimental results demonstrate that our method significantly reduces redundant search operations while maintaining or improving answer accuracy across multiple challenging benchmarks. This framework advances the model's reasoning efficiency and extends its capability to judiciously leverage external knowledge.
>
---
#### [new 022] Scaling Context, Not Parameters: Training a Compact 7B Language Model for Efficient Long-Context Processing
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于高效长上下文语言模型训练任务，旨在解决传统模型处理长文本时的参数膨胀和效率问题。通过优化上下文扩展而非增加参数，提出了7B参数的MegaBeam-Mistral-7B模型，支持512K token上下文，在合规监控等现实任务中验证有效性，并在三大基准测试中展现领先性能，实现无需增强技术的长程推理能力。**

- **链接: [http://arxiv.org/pdf/2505.08651v1](http://arxiv.org/pdf/2505.08651v1)**

> **作者:** Chen Wu; Yin Song
>
> **备注:** 8 pages, 6 figures, ACL 2025 (Industry Track)
>
> **摘要:** We present MegaBeam-Mistral-7B, a language model that supports 512K-token context length. Our work addresses practical limitations in long-context training, supporting real-world tasks such as compliance monitoring and verification. Evaluated on three long-context benchmarks, our 7B-parameter model demonstrates superior in-context learning performance on HELMET and robust retrieval and tracing capability on RULER. It is currently the only open model to achieve competitive long-range reasoning on BABILong at 512K context length without RAG or targeted fine-tuning. Released as fully open source under the Apache 2.0 license, the model has been downloaded over 100,000 times on Hugging Face. Model available at: https://huggingface.co/aws-prototyping/MegaBeam-Mistral-7B-512k
>
---
#### [new 023] PLHF: Prompt Optimization with Few-Shot Human Feedback
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大模型提示优化任务，解决输出质量难以量化评估时的优化难题。提出PLHF框架，借鉴RLHF思想，通过少量人类反馈训练评估模块替代传统指标，实现单轮优化。实验表明其效果优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.07886v1](http://arxiv.org/pdf/2505.07886v1)**

> **作者:** Chun-Pai Yang; Kan Zheng; Shou-De Lin
>
> **摘要:** Automatic prompt optimization frameworks are developed to obtain suitable prompts for large language models (LLMs) with respect to desired output quality metrics. Although existing approaches can handle conventional tasks such as fixed-solution question answering, defining the metric becomes complicated when the output quality cannot be easily assessed by comparisons with standard golden samples. Consequently, optimizing the prompts effectively and efficiently without a clear metric becomes a critical challenge. To address the issue, we present PLHF (which stands for "P"rompt "L"earning with "H"uman "F"eedback), a few-shot prompt optimization framework inspired by the well-known RLHF technique. Different from naive strategies, PLHF employs a specific evaluator module acting as the metric to estimate the output quality. PLHF requires only a single round of human feedback to complete the entire prompt optimization process. Empirical results on both public and industrial datasets show that PLHF outperforms prior output grading strategies for LLM prompt optimizations.
>
---
#### [new 024] A Tale of Two Identities: An Ethical Audit of Human and AI-Crafted Personas
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于AI伦理评估任务，研究LLM生成合成人格时对少数族裔的表述偏见问题。通过对比3个模型生成的1512个合成人格与人类文本，发现LLM过度强调种族标签、使用文化刻板语言，导致"算法他者化"现象——少数身份被强化但失真，引发刻板印象等伦理风险。研究提出叙事感知评估指标和社区验证方案，旨在优化合成身份生成技术。**

- **链接: [http://arxiv.org/pdf/2505.07850v1](http://arxiv.org/pdf/2505.07850v1)**

> **作者:** Pranav Narayanan Venkit; Jiayi Li; Yingfan Zhou; Sarah Rajtmajer; Shomir Wilson
>
> **摘要:** As LLMs (large language models) are increasingly used to generate synthetic personas particularly in data-limited domains such as health, privacy, and HCI, it becomes necessary to understand how these narratives represent identity, especially that of minority communities. In this paper, we audit synthetic personas generated by 3 LLMs (GPT4o, Gemini 1.5 Pro, Deepseek 2.5) through the lens of representational harm, focusing specifically on racial identity. Using a mixed methods approach combining close reading, lexical analysis, and a parameterized creativity framework, we compare 1512 LLM generated personas to human-authored responses. Our findings reveal that LLMs disproportionately foreground racial markers, overproduce culturally coded language, and construct personas that are syntactically elaborate yet narratively reductive. These patterns result in a range of sociotechnical harms, including stereotyping, exoticism, erasure, and benevolent bias, that are often obfuscated by superficially positive narrations. We formalize this phenomenon as algorithmic othering, where minoritized identities are rendered hypervisible but less authentic. Based on these findings, we offer design recommendations for narrative-aware evaluation metrics and community-centered validation protocols for synthetic identity generation.
>
---
#### [new 025] Towards Contamination Resistant Benchmarks
- **分类: cs.CL**

- **简介: 该论文属于大语言模型（LLM）评估任务，旨在解决评估过程中数据污染导致结果不可靠的问题。研究者提出基于凯撒密码的污染抵抗基准，通过控制污染条件测试发现现有LLMs表现显著下降，揭示了模型真实能力与评估偏差间的矛盾，为构建抗污染评估体系提供了新方法。**

- **链接: [http://arxiv.org/pdf/2505.08389v1](http://arxiv.org/pdf/2505.08389v1)**

> **作者:** Rahmatullah Musawi; Sheng Lu
>
> **摘要:** The rapid development of large language models (LLMs) has transformed the landscape of natural language processing. Evaluating LLMs properly is crucial for understanding their potential and addressing concerns such as safety. However, LLM evaluation is confronted by various factors, among which contamination stands out as a key issue that undermines the reliability of evaluations. In this work, we introduce the concept of contamination resistance to address this challenge. We propose a benchmark based on Caesar ciphers (e.g., "ab" to "bc" when the shift is 1), which, despite its simplicity, is an excellent example of a contamination resistant benchmark. We test this benchmark on widely used LLMs under various settings, and we find that these models struggle with this benchmark when contamination is controlled. Our findings reveal issues in current LLMs and raise important questions regarding their true capabilities. Our work contributes to the development of contamination resistant benchmarks, enabling more rigorous LLM evaluation and offering insights into the true capabilities and limitations of LLMs.
>
---
#### [new 026] Small but Significant: On the Promise of Small Language Models for Accessible AIED
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 该论文属AI教育（AIED）领域，探讨小型语言模型（SLMs）在教育技术中的潜力。针对当前领域过度依赖资源密集型大模型（如GPT）导致资源受限机构难以获取AI工具的问题，研究通过知识组件发现实验证明Phi-2等SLMs无需复杂提示即可有效解决关键教育挑战，主张发展低成本SLM方案以促进教育公平。**

- **链接: [http://arxiv.org/pdf/2505.08588v1](http://arxiv.org/pdf/2505.08588v1)**

> **作者:** Yumou Wei; Paulo Carvalho; John Stamper
>
> **备注:** This vision paper advocates using small language models (e.g., Phi-2) in AI for education (AIED)
>
> **摘要:** GPT has become nearly synonymous with large language models (LLMs), an increasingly popular term in AIED proceedings. A simple keyword-based search reveals that 61% of the 76 long and short papers presented at AIED 2024 describe novel solutions using LLMs to address some of the long-standing challenges in education, and 43% specifically mention GPT. Although LLMs pioneered by GPT create exciting opportunities to strengthen the impact of AI on education, we argue that the field's predominant focus on GPT and other resource-intensive LLMs (with more than 10B parameters) risks neglecting the potential impact that small language models (SLMs) can make in providing resource-constrained institutions with equitable and affordable access to high-quality AI tools. Supported by positive results on knowledge component (KC) discovery, a critical challenge in AIED, we demonstrate that SLMs such as Phi-2 can produce an effective solution without elaborate prompting strategies. Hence, we call for more attention to developing SLM-based AIED approaches.
>
---
#### [new 027] A document processing pipeline for the construction of a dataset for topic modeling based on the judgments of the Italian Supreme Court
- **分类: cs.CL**

- **简介: 该论文属于法律文本处理任务，旨在解决意大利最高法院判决缺乏公开主题建模数据集的问题。研究者开发了集成文档布局分析、OCR和文本匿名化的处理流程，构建优化数据集，并应用BERTopic和LLMs生成主题标签与摘要，验证了方法在准确性和模型性能上的提升。**

- **链接: [http://arxiv.org/pdf/2505.08439v1](http://arxiv.org/pdf/2505.08439v1)**

> **作者:** Matteo Marulli; Glauco Panattoni; Marco Bertini
>
> **备注:** 51 pages
>
> **摘要:** Topic modeling in Italian legal research is hindered by the lack of public datasets, limiting the analysis of legal themes in Supreme Court judgments. To address this, we developed a document processing pipeline that produces an anonymized dataset optimized for topic modeling. The pipeline integrates document layout analysis (YOLOv8x), optical character recognition, and text anonymization. The DLA module achieved a mAP@50 of 0.964 and a mAP@50-95 of 0.800. The OCR detector reached a mAP@50-95 of 0.9022, and the text recognizer (TrOCR) obtained a character error rate of 0.0047 and a word error rate of 0.0248. Compared to OCR-only methods, our dataset improved topic modeling with a diversity score of 0.6198 and a coherence score of 0.6638. We applied BERTopic to extract topics and used large language models to generate labels and summaries. Outputs were evaluated against domain expert interpretations. Claude Sonnet 3.7 achieved a BERTScore F1 of 0.8119 for labeling and 0.9130 for summarization.
>
---
#### [new 028] Reassessing Graph Linearization for Sequence-to-sequence AMR Parsing: On the Advantages and Limitations of Triple-Based Encoding
- **分类: cs.CL**

- **简介: 该论文研究AMR解析任务，探索将图结构线性化为序列的方法。针对传统Penman编码在深层次图中节点距离远、需逆角色处理重入节点的问题，提出三元组编码方案。实验表明三元组虽能直接表示图结构，但在嵌套表达上仍逊于Penman的简洁性，需进一步优化。**

- **链接: [http://arxiv.org/pdf/2505.08504v1](http://arxiv.org/pdf/2505.08504v1)**

> **作者:** Jeongwoo Kang; Maximin Coavoux; Cédric Lopez; Didier Schwab
>
> **备注:** published at Insights from Negative Results in NLP (workshop EMNLP 2025)
>
> **摘要:** Sequence-to-sequence models are widely used to train Abstract Meaning Representation (Banarescu et al., 2013, AMR) parsers. To train such models, AMR graphs have to be linearized into a one-line text format. While Penman encoding is typically used for this purpose, we argue that it has limitations: (1) for deep graphs, some closely related nodes are located far apart in the linearized text (2) Penman's tree-based encoding necessitates inverse roles to handle node re-entrancy, doubling the number of relation types to predict. To address these issues, we propose a triple-based linearization method and compare its efficiency with Penman linearization. Although triples are well suited to represent a graph, our results suggest room for improvement in triple encoding to better compete with Penman's concise and explicit representation of a nested graph structure.
>
---
#### [new 029] IterKey: Iterative Keyword Generation with LLMs for Enhanced Retrieval Augmented Generation
- **分类: cs.CL**

- **简介: 该论文针对检索增强生成（RAG）中稀疏检索方法可解释性强但准确性不足的问题，提出IterKey框架。通过大语言模型分阶段迭代生成检索关键词、验证答案并优化关键词，在保持可解释性的同时提升准确性。实验表明其性能超越传统BM25方法，接近密集检索效果，实现了准确性与透明度的平衡。**

- **链接: [http://arxiv.org/pdf/2505.08450v1](http://arxiv.org/pdf/2505.08450v1)**

> **作者:** Kazuki Hayashi; Hidetaka Kamigaito; Shinya Kouda; Taro Watanabe
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a way to complement the in-context knowledge of Large Language Models (LLMs) by integrating external documents. However, real-world applications demand not only accuracy but also interpretability. While dense retrieval methods provide high accuracy, they lack interpretability; conversely, sparse retrieval methods offer transparency but often fail to capture the full intent of queries due to their reliance on keyword matching. To address these issues, we introduce IterKey, an LLM-driven iterative keyword generation framework that enhances RAG via sparse retrieval. IterKey consists of three LLM-driven stages: generating keywords for retrieval, generating answers based on retrieved documents, and validating the answers. If validation fails, the process iteratively repeats with refined keywords. Across four QA tasks, experimental results show that IterKey achieves 5% to 20% accuracy improvements over BM25-based RAG and simple baselines. Its performance is comparable to dense retrieval-based RAG and prior iterative query refinement methods using dense models. In summary, IterKey is a novel BM25-based approach leveraging LLMs to iteratively refine RAG, effectively balancing accuracy with interpretability.
>
---
#### [new 030] Re$^2$: A Consistency-ensured Dataset for Full-stage Peer Review and Multi-turn Rebuttal Discussions
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文构建了Re²数据集，解决现有同行评审数据多样性不足、数据不一致及缺乏多轮反驳支持的问题。通过整合大量初始投稿、评审和反驳数据，并设计多轮对话框架，支持开发动态LLM助手，帮助作者改进稿件并减轻评审负担。**

- **链接: [http://arxiv.org/pdf/2505.07920v1](http://arxiv.org/pdf/2505.07920v1)**

> **作者:** Daoze Zhang; Zhijian Bao; Sihang Du; Zhiyi Zhao; Kuangling Zhang; Dezheng Bao; Yang Yang
>
> **备注:** 2 figures, 5 tables
>
> **摘要:** Peer review is a critical component of scientific progress in the fields like AI, but the rapid increase in submission volume has strained the reviewing system, which inevitably leads to reviewer shortages and declines review quality. Besides the growing research popularity, another key factor in this overload is the repeated resubmission of substandard manuscripts, largely due to the lack of effective tools for authors to self-evaluate their work before submission. Large Language Models (LLMs) show great promise in assisting both authors and reviewers, and their performance is fundamentally limited by the quality of the peer review data. However, existing peer review datasets face three major limitations: (1) limited data diversity, (2) inconsistent and low-quality data due to the use of revised rather than initial submissions, and (3) insufficient support for tasks involving rebuttal and reviewer-author interactions. To address these challenges, we introduce the largest consistency-ensured peer review and rebuttal dataset named Re^2, which comprises 19,926 initial submissions, 70,668 review comments, and 53,818 rebuttals from 24 conferences and 21 workshops on OpenReview. Moreover, the rebuttal and discussion stage is framed as a multi-turn conversation paradigm to support both traditional static review tasks and dynamic interactive LLM assistants, providing more practical guidance for authors to refine their manuscripts and helping alleviate the growing review burden. Our data and code are available in https://anonymous.4open.science/r/ReviewBench_anon/.
>
---
#### [new 031] LCES: Zero-shot Automated Essay Scoring via Pairwise Comparisons Using Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对零样本自动化作文评分任务，解决传统方法因模型偏差直接生成绝对分数不准确的问题。提出LCES方法，利用大模型对作文进行成对比较，结合RankNet将比较结果转为连续分数，提升评分准确性及效率。实验表明其优于现有零样本方法且模型适应性更强。**

- **链接: [http://arxiv.org/pdf/2505.08498v1](http://arxiv.org/pdf/2505.08498v1)**

> **作者:** Takumi Shibata; Yuichi Miyamura
>
> **备注:** 14 pages, 4 figures
>
> **摘要:** Recent advances in large language models (LLMs) have enabled zero-shot automated essay scoring (AES), providing a promising way to reduce the cost and effort of essay scoring in comparison with manual grading. However, most existing zero-shot approaches rely on LLMs to directly generate absolute scores, which often diverge from human evaluations owing to model biases and inconsistent scoring. To address these limitations, we propose LLM-based Comparative Essay Scoring (LCES), a method that formulates AES as a pairwise comparison task. Specifically, we instruct LLMs to judge which of two essays is better, collect many such comparisons, and convert them into continuous scores. Considering that the number of possible comparisons grows quadratically with the number of essays, we improve scalability by employing RankNet to efficiently transform LLM preferences into scalar scores. Experiments using AES benchmark datasets show that LCES outperforms conventional zero-shot methods in accuracy while maintaining computational efficiency. Moreover, LCES is robust across different LLM backbones, highlighting its applicability to real-world zero-shot AES.
>
---
#### [new 032] RepCali: High Efficient Fine-tuning Via Representation Calibration in Latent Space for Pre-trained Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的预训练语言模型（PLM）微调任务，旨在解决编码器输出与解码器输入不匹配的问题。通过提出RepCali方法，在编码器潜在空间嵌入可校准模块，优化特征表示以提升下游任务性能。该方法适配主流PLM架构，实验验证其高效性与普适性。**

- **链接: [http://arxiv.org/pdf/2505.08463v1](http://arxiv.org/pdf/2505.08463v1)**

> **作者:** Fujun Zhang; XiangDong Su
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** Fine-tuning pre-trained language models (PLMs) has become a dominant paradigm in applying PLMs to downstream tasks. However, with limited fine-tuning, PLMs still struggle with the discrepancies between the representation obtained from the PLMs' encoder and the optimal input to the PLMs' decoder. This paper tackles this challenge by learning to calibrate the representation of PLMs in the latent space. In the proposed representation calibration method (RepCali), we integrate a specific calibration block to the latent space after the encoder and use the calibrated output as the decoder input. The merits of the proposed RepCali include its universality to all PLMs with encoder-decoder architectures, its plug-and-play nature, and ease of implementation. Extensive experiments on 25 PLM-based models across 8 tasks (including both English and Chinese datasets) demonstrate that the proposed RepCali offers desirable enhancements to PLMs (including LLMs) and significantly improves the performance of downstream tasks. Comparison experiments across 4 benchmark tasks indicate that RepCali is superior to the representative fine-tuning baselines.
>
---
#### [new 033] On the Geometry of Semantics in Next-token Prediction
- **分类: cs.CL**

- **简介: 该论文属于语言模型训练机制分析，探究仅通过下一词预测(NTP)任务训练的模型如何捕获语义语法结构。研究发现NTP优化隐式引导模型通过奇异值分解(SVD)编码潜在概念，词嵌入自动分解捕捉词共现模式。提出基于谱聚类的新方法识别语义，并连接分布语义理论与神经网络训练动态，揭示NTP隐含偏差对语义表征形成的作用。**

- **链接: [http://arxiv.org/pdf/2505.08348v1](http://arxiv.org/pdf/2505.08348v1)**

> **作者:** Yize Zhao; Christos Thrampoulidis
>
> **摘要:** Modern language models demonstrate a remarkable ability to capture linguistic meaning despite being trained solely through next-token prediction (NTP). We investigate how this conceptually simple training objective leads models to extract and encode latent semantic and grammatical concepts. Our analysis reveals that NTP optimization implicitly guides models to encode concepts via singular value decomposition (SVD) factors of a centered data-sparsity matrix that captures next-word co-occurrence patterns. While the model never explicitly constructs this matrix, learned word and context embeddings effectively factor it to capture linguistic structure. We find that the most important SVD factors are learned first during training, motivating the use of spectral clustering of embeddings to identify human-interpretable semantics, including both classical k-means and a new orthant-based method directly motivated by our interpretation of concepts. Overall, our work bridges distributional semantics, neural collapse geometry, and neural network training dynamics, providing insights into how NTP's implicit biases shape the emergence of meaning representations in language models.
>
---
#### [new 034] Graph Laplacian Wavelet Transformer via Learnable Spectral Decomposition
- **分类: cs.CL**

- **简介: 该论文针对结构化语言任务中序列模型二次复杂度问题，提出图小波变换器(GWT)。通过可学习的多尺度谱分解替代传统自注意力机制，基于句法/语义图构建拉普拉斯算子，实现高效、可解释的图结构序列建模，降低计算与内存开销。**

- **链接: [http://arxiv.org/pdf/2505.07862v1](http://arxiv.org/pdf/2505.07862v1)**

> **作者:** Andrew Kiruluta; Eric Lundy; Priscilla Burity
>
> **摘要:** Existing sequence to sequence models for structured language tasks rely heavily on the dot product self attention mechanism, which incurs quadratic complexity in both computation and memory for input length N. We introduce the Graph Wavelet Transformer (GWT), a novel architecture that replaces this bottleneck with a learnable, multi scale wavelet transform defined over an explicit graph Laplacian derived from syntactic or semantic parses. Our analysis shows that multi scale spectral decomposition offers an interpretable, efficient, and expressive alternative to quadratic self attention for graph structured sequence modeling.
>
---
#### [new 035] Boosting Performance on ARC is a Matter of Perspective
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对抽象推理任务ARC-AGI，解决大语言模型抽象推理能力不足的问题。通过数据增强、深度优先搜索生成多候选解，并利用模型自身概率评分筛选最优方案，实现公开方法中71.6%的最高得分，兼具高透明度与极低推理成本（单任务约2分钱）。**

- **链接: [http://arxiv.org/pdf/2505.07859v1](http://arxiv.org/pdf/2505.07859v1)**

> **作者:** Daniel Franzen; Jan Disselhoff; David Hartmann
>
> **备注:** 14 pages, 5 figures, 5 tables
>
> **摘要:** The Abstraction and Reasoning Corpus (ARC-AGI) poses a significant challenge for large language models (LLMs), exposing limitations in their abstract reasoning abilities. In this work, we leverage task-specific data augmentations throughout the training, generation, and scoring phases, and employ a depth-first search algorithm to generate diverse, high-probability candidate solutions. Furthermore, we utilize the LLM not only as a generator but also as a scorer, using its output probabilities to select the most promising solutions. Our method achieves a score of 71.6% (286.5/400 solved tasks) on the public ARC-AGI evaluation set, demonstrating state-of-the-art performance among publicly available approaches. While concurrent closed-source work has reported higher scores, our method distinguishes itself through its transparency, reproducibility, and remarkably low inference cost, averaging only around 2ct per task on readily available hardware (we assume a price of 36ct/hour for a Nvidia 4090 GPU).
>
---
#### [new 036] Accelerating Chain-of-Thought Reasoning: When Goal-Gradient Importance Meets Dynamic Skipping
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的推理效率优化任务，旨在解决思维链（CoT）提示推理冗余导致的算力消耗问题。通过提出Adaptive GoGI-Skip框架，结合目标梯度重要性指标（GoGI）和动态跳过机制（ADS），动态压缩冗余推理步骤，在保持精度的同时平均减少45%的token并提升1.6-2倍推理速度。**

- **链接: [http://arxiv.org/pdf/2505.08392v1](http://arxiv.org/pdf/2505.08392v1)**

> **作者:** Ren Zhuang; Ben Wang; Shuifa Sun
>
> **摘要:** Large Language Models leverage Chain-of-Thought (CoT) prompting for complex tasks, but their reasoning traces are often excessively verbose and inefficient, leading to significant computational costs and latency. Current CoT compression techniques typically rely on generic importance metrics and static compression rates, which may inadvertently remove functionally critical tokens or fail to adapt to varying reasoning complexity. To overcome these limitations, we propose Adaptive GoGI-Skip, a novel framework learning dynamic CoT compression via supervised fine-tuning. This approach introduces two synergistic innovations: (1) Goal-Gradient Importance (GoGI), a novel metric accurately identifying functionally relevant tokens by measuring the gradient influence of their intermediate representations on the final answer loss, and (2) Adaptive Dynamic Skipping (ADS), a mechanism dynamically regulating the compression rate based on runtime model uncertainty while ensuring local coherence through an adaptive N-token constraint. To our knowledge, this is the first work unifying a goal-oriented, gradient-based importance metric with dynamic, uncertainty-aware skipping for CoT compression. Trained on compressed MATH data, Adaptive GoGI-Skip demonstrates strong cross-domain generalization across diverse reasoning benchmarks including AIME, GPQA, and GSM8K. It achieves substantial efficiency gains - reducing CoT token counts by over 45% on average and delivering 1.6-2.0 times inference speedups - while maintaining high reasoning accuracy. Notably, it significantly outperforms existing baselines by preserving accuracy even at high effective compression rates, advancing the state of the art in the CoT reasoning efficiency-accuracy trade-off.
>
---
#### [new 037] Are LLMs complicated ethical dilemma analyzers?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型（LLMs）的伦理推理能力，属于伦理决策评估任务。通过构建含196个现实伦理困境的基准数据集（含专家标注与非专家对比），使用复合指标评估前沿模型。结果表明：LLMs在文本结构上优于非专家，但缺乏历史背景理解和策略创新，人类虽结构松散但语义接近，揭示了模型在抽象伦理推理中的局限性。**

- **链接: [http://arxiv.org/pdf/2505.08106v1](http://arxiv.org/pdf/2505.08106v1)**

> **作者:** Jiashen; Du; Jesse Yao; Allen Liu; Zhekai Zhang
>
> **备注:** CS194-280 Advanced LLM Agents project. Project page: https://github.com/ALT-JS/ethicaLLM
>
> **摘要:** One open question in the study of Large Language Models (LLMs) is whether they can emulate human ethical reasoning and act as believable proxies for human judgment. To investigate this, we introduce a benchmark dataset comprising 196 real-world ethical dilemmas and expert opinions, each segmented into five structured components: Introduction, Key Factors, Historical Theoretical Perspectives, Resolution Strategies, and Key Takeaways. We also collect non-expert human responses for comparison, limited to the Key Factors section due to their brevity. We evaluate multiple frontier LLMs (GPT-4o-mini, Claude-3.5-Sonnet, Deepseek-V3, Gemini-1.5-Flash) using a composite metric framework based on BLEU, Damerau-Levenshtein distance, TF-IDF cosine similarity, and Universal Sentence Encoder similarity. Metric weights are computed through an inversion-based ranking alignment and pairwise AHP analysis, enabling fine-grained comparison of model outputs to expert responses. Our results show that LLMs generally outperform non-expert humans in lexical and structural alignment, with GPT-4o-mini performing most consistently across all sections. However, all models struggle with historical grounding and proposing nuanced resolution strategies, which require contextual abstraction. Human responses, while less structured, occasionally achieve comparable semantic similarity, suggesting intuitive moral reasoning. These findings highlight both the strengths and current limitations of LLMs in ethical decision-making.
>
---
#### [new 038] The Sound of Populism: Distinct Linguistic Features Across Populist Variants
- **分类: cs.CL**

- **简介: 该论文属于政治文本分析任务，旨在探究不同民粹主义变体（左/右翼、反精英、人民中心）的语言特征差异。通过结合传统LIWC工具与RoBERTa模型分析美国总统演讲，揭示民粹修辞的共性与差异：右翼和人民中心倾向情绪化表达，左翼与反精英则更克制，整体呈现战略性的直接语调以塑造亲民领导形象。**

- **链接: [http://arxiv.org/pdf/2505.07874v1](http://arxiv.org/pdf/2505.07874v1)**

> **作者:** Yu Wang; Runxi Yu; Zhongyuan Wang; Jing He
>
> **摘要:** This study explores the sound of populism by integrating the classic Linguistic Inquiry and Word Count (LIWC) features, which capture the emotional and stylistic tones of language, with a fine-tuned RoBERTa model, a state-of-the-art context-aware language model trained to detect nuanced expressions of populism. This approach allows us to uncover the auditory dimensions of political rhetoric in U.S. presidential inaugural and State of the Union addresses. We examine how four key populist dimensions (i.e., left-wing, right-wing, anti-elitism, and people-centrism) manifest in the linguistic markers of speech, drawing attention to both commonalities and distinct tonal shifts across these variants. Our findings reveal that populist rhetoric consistently features a direct, assertive ``sound" that forges a connection with ``the people'' and constructs a charismatic leadership persona. However, this sound is not simply informal but strategically calibrated. Notably, right-wing populism and people-centrism exhibit a more emotionally charged discourse, resonating with themes of identity, grievance, and crisis, in contrast to the relatively restrained emotional tones of left-wing and anti-elitist expressions.
>
---
#### [new 039] TrumorGPT: Graph-Based Retrieval-Augmented Large Language Model for Fact-Checking
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于健康领域事实核查任务，旨在区分真实健康传言（trumors）与虚假信息。为解决LLM幻觉和静态数据局限，提出TrumorGPT框架，结合图检索增强生成（GraphRAG）和动态更新的语义知识图谱，提升核查准确性。实验表明其在公共卫生声明验证中表现优异。**

- **链接: [http://arxiv.org/pdf/2505.07891v1](http://arxiv.org/pdf/2505.07891v1)**

> **作者:** Ching Nam Hang; Pei-Duo Yu; Chee Wei Tan
>
> **摘要:** In the age of social media, the rapid spread of misinformation and rumors has led to the emergence of infodemics, where false information poses a significant threat to society. To combat this issue, we introduce TrumorGPT , a novel generative artificial intelligence solution designed for fact-checking in the health domain. TrumorGPT aims to distinguish "trumors", which are health-related rumors that turn out to be true, providing a crucial tool in differentiating between mere speculation and verified facts. This framework leverages a large language model (LLM) with few-shot learning for semantic health knowledge graph construction and semantic reasoning. TrumorGPT incorporates graph-based retrieval-augmented generation (GraphRAG) to address the hallucination issue common in LLMs and the limitations of static training data. GraphRAG involves accessing and utilizing information from regularly updated semantic health knowledge graphs that consist of the latest medical news and health information, ensuring that fact-checking by TrumorGPT is based on the most recent data. Evaluating with extensive healthcare datasets, TrumorGPT demonstrates superior performance in fact-checking for public health claims. Its ability to effectively conduct fact-checking across various platforms marks a critical step forward in the fight against health-related misinformation, enhancing trust and accuracy in the digital information age.
>
---
#### [new 040] DeltaEdit: Enhancing Sequential Editing in Large Language Models by Controlling Superimposed Noise
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大语言模型的顺序知识编辑任务，旨在解决长期编辑后叠加噪声累积导致的成功率下降问题。提出DeltaEdit方法，通过动态正交约束优化参数，减少编辑间干扰，提升持续更新效果，并在实验中验证了其编辑成功率和泛化能力的优势。**

- **链接: [http://arxiv.org/pdf/2505.07899v1](http://arxiv.org/pdf/2505.07899v1)**

> **作者:** Ding Cao; Yuchen Cai; Rongxi Guo; Xuesong He; Guiquan Liu
>
> **摘要:** Sequential knowledge editing techniques aim to continuously update the knowledge in large language models at a low cost, preventing the models from generating outdated or incorrect information. However, existing sequential editing methods suffer from a significant decline in editing success rates after long-term editing. Through theoretical analysis and experiments, we identify that as the number of edits increases, the model's output increasingly deviates from the desired target, leading to a drop in editing success rates. We refer to this issue as the accumulation of superimposed noise problem. To address this, we identify the factors contributing to this deviation and propose DeltaEdit, a novel method that optimizes update parameters through a dynamic orthogonal constraints strategy, effectively reducing interference between edits to mitigate deviation. Experimental results demonstrate that DeltaEdit significantly outperforms existing methods in edit success rates and the retention of generalization capabilities, ensuring stable and reliable model performance even under extensive sequential editing.
>
---
#### [new 041] Probability Consistency in Large Language Models: Theoretical Foundations Meet Empirical Discrepancies
- **分类: cs.CL**

- **简介: 该论文研究自回归大模型（LLMs）在不同分词顺序下学习概率分布的一致性任务。理论证明序列困惑度对因子分解顺序不变，但实验发现训练顺序（前向、后向、随机排列）导致系统偏差，随机排列表现差异显著，揭示自注意力机制的位置/局部性偏差，为检测概率不一致性提供新方法。**

- **链接: [http://arxiv.org/pdf/2505.08739v1](http://arxiv.org/pdf/2505.08739v1)**

> **作者:** Xiaoliang Luo; Xinyi Xu; Michael Ramscar; Bradley C. Love
>
> **摘要:** Can autoregressive large language models (LLMs) learn consistent probability distributions when trained on sequences in different token orders? We prove formally that for any well-defined probability distribution, sequence perplexity is invariant under any factorization, including forward, backward, or arbitrary permutations. This result establishes a rigorous theoretical foundation for studying how LLMs learn from data and defines principled protocols for empirical evaluation. Applying these protocols, we show that prior studies examining ordering effects suffer from critical methodological flaws. We retrain GPT-2 models across forward, backward, and arbitrary permuted orders on scientific text. We find systematic deviations from theoretical invariance across all orderings with arbitrary permutations strongly deviating from both forward and backward models, which largely (but not completely) agreed with one another. Deviations were traceable to differences in self-attention, reflecting positional and locality biases in processing. Our theoretical and empirical results provide novel avenues for understanding positional biases in LLMs and suggest methods for detecting when LLMs' probability distributions are inconsistent and therefore untrustworthy.
>
---
#### [new 042] TUMS: Enhancing Tool-use Abilities of LLMs with Multi-structure Handlers
- **分类: cs.CL**

- **简介: 该论文属于工具增强型大语言模型（LLMs）领域，旨在解决LLMs调用工具时因参数生成错误导致的执行失效问题。提出TUMS框架，通过参数级处理（意图识别、任务分解、多结构参数生成器）提升工具使用精度，在ToolQA基准上效果显著提升。**

- **链接: [http://arxiv.org/pdf/2505.08402v1](http://arxiv.org/pdf/2505.08402v1)**

> **作者:** Aiyao He; Sijia Cui; Shuai Xu; Yanna Wang; Bo Xu
>
> **备注:** Accepted to ICONIP 2024
>
> **摘要:** Recently, large language models(LLMs) have played an increasingly important role in solving a wide range of NLP tasks, leveraging their capabilities of natural language understanding and generating. Integration with external tools further enhances LLMs' effectiveness, providing more precise, timely, and specialized responses. However, LLMs still encounter difficulties with non-executable actions and improper actions, which are primarily attributed to incorrect parameters. The process of generating parameters by LLMs is confined to the tool level, employing the coarse-grained strategy without considering the different difficulties of various tools. To address this issue, we propose TUMS, a novel framework designed to enhance the tool-use capabilities of LLMs by transforming tool-level processing into parameter-level processing. Specifically, our framework consists of four key components: (1) an intent recognizer that identifies the user's intent to help LLMs better understand the task; (2) a task decomposer that breaks down complex tasks into simpler subtasks, each involving a tool call; (3) a subtask processor equipped with multi-structure handlers to generate accurate parameters; and (4) an executor. Our empirical studies have evidenced the effectiveness and efficiency of the TUMS framework with an average of 19.6\% and 50.6\% improvement separately on easy and hard benchmarks of ToolQA, meanwhile, we demonstrated the key contribution of each part with ablation experiments, offering more insights and stimulating future research on Tool-augmented LLMs.
>
---
#### [new 043] Assessing and Mitigating Medical Knowledge Drift and Conflicts in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于医疗AI可靠性研究，旨在解决大型语言模型(LLMs)因医学知识更新导致的过时/矛盾建议问题。研究创建了DriftMedQA基准评估模型时效性，测试发现主流模型存在知识漂移和自相矛盾现象，并提出检索增强生成与偏好微调结合的优化策略，显著提升临床建议可靠性。**

- **链接: [http://arxiv.org/pdf/2505.07968v1](http://arxiv.org/pdf/2505.07968v1)**

> **作者:** Weiyi Wu; Xinwen Xu; Chongyang Gao; Xingjian Diao; Siting Li; Lucas A. Salas; Jiang Gui
>
> **摘要:** Large Language Models (LLMs) have great potential in the field of health care, yet they face great challenges in adapting to rapidly evolving medical knowledge. This can lead to outdated or contradictory treatment suggestions. This study investigated how LLMs respond to evolving clinical guidelines, focusing on concept drift and internal inconsistencies. We developed the DriftMedQA benchmark to simulate guideline evolution and assessed the temporal reliability of various LLMs. Our evaluation of seven state-of-the-art models across 4,290 scenarios demonstrated difficulties in rejecting outdated recommendations and frequently endorsing conflicting guidance. Additionally, we explored two mitigation strategies: Retrieval-Augmented Generation and preference fine-tuning via Direct Preference Optimization. While each method improved model performance, their combination led to the most consistent and reliable results. These findings underscore the need to improve LLM robustness to temporal shifts to ensure more dependable applications in clinical practice.
>
---
#### [new 044] Aya Vision: Advancing the Frontier of Multilingual Multimodality
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 该论文针对多语言多模态模型构建中的数据稀缺、翻译失真及灾难性遗忘问题，提出合成数据框架和跨模态模型融合技术，开发了Aya Vision系列模型，在保持文本能力的同时提升多模态生成性能，显著优于更大规模的基准模型。属于多模态自然语言处理任务。**

- **链接: [http://arxiv.org/pdf/2505.08751v1](http://arxiv.org/pdf/2505.08751v1)**

> **作者:** Saurabh Dash; Yiyang Nan; John Dang; Arash Ahmadian; Shivalika Singh; Madeline Smith; Bharat Venkitesh; Vlad Shmyhlo; Viraat Aryabumi; Walter Beller-Morales; Jeremy Pekmez; Jason Ozuzu; Pierre Richemond; Acyr Locatelli; Nick Frosst; Phil Blunsom; Aidan Gomez; Ivan Zhang; Marzieh Fadaee; Manoj Govindassamy; Sudip Roy; Matthias Gallé; Beyza Ermis; Ahmet Üstün; Sara Hooker
>
> **摘要:** Building multimodal language models is fundamentally challenging: it requires aligning vision and language modalities, curating high-quality instruction data, and avoiding the degradation of existing text-only capabilities once vision is introduced. These difficulties are further magnified in the multilingual setting, where the need for multimodal data in different languages exacerbates existing data scarcity, machine translation often distorts meaning, and catastrophic forgetting is more pronounced. To address the aforementioned challenges, we introduce novel techniques spanning both data and modeling. First, we develop a synthetic annotation framework that curates high-quality, diverse multilingual multimodal instruction data, enabling Aya Vision models to produce natural, human-preferred responses to multimodal inputs across many languages. Complementing this, we propose a cross-modal model merging technique that mitigates catastrophic forgetting, effectively preserving text-only capabilities while simultaneously enhancing multimodal generative performance. Aya-Vision-8B achieves best-in-class performance compared to strong multimodal models such as Qwen-2.5-VL-7B, Pixtral-12B, and even much larger Llama-3.2-90B-Vision. We further scale this approach with Aya-Vision-32B, which outperforms models more than twice its size, such as Molmo-72B and LLaMA-3.2-90B-Vision. Our work advances multilingual progress on the multi-modal frontier, and provides insights into techniques that effectively bend the need for compute while delivering extremely high performance.
>
---
#### [new 045] Judging the Judges: Can Large Vision-Language Models Fairly Evaluate Chart Comprehension and Reasoning?
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究大型视觉语言模型(LVLM)作为自动评估工具在图表理解任务中的可行性，属于模型评估任务。针对现有评估方法成本高、封闭性强的问题，系统测试了13个开源LVLM的评判能力，设计多维度评估标准并分析偏差，发现部分模型能达到GPT-4水平但存在位置偏好等局限性。**

- **链接: [http://arxiv.org/pdf/2505.08468v1](http://arxiv.org/pdf/2505.08468v1)**

> **作者:** Md Tahmid Rahman Laskar; Mohammed Saidul Islam; Ridwan Mahbub; Ahmed Masry; Mizanur Rahman; Amran Bhuiyan; Mir Tafseer Nayeem; Shafiq Joty; Enamul Hoque; Jimmy Huang
>
> **备注:** Accepted at ACL 2025 Industry Track
>
> **摘要:** Charts are ubiquitous as they help people understand and reason with data. Recently, various downstream tasks, such as chart question answering, chart2text, and fact-checking, have emerged. Large Vision-Language Models (LVLMs) show promise in tackling these tasks, but their evaluation is costly and time-consuming, limiting real-world deployment. While using LVLMs as judges to assess the chart comprehension capabilities of other LVLMs could streamline evaluation processes, challenges like proprietary datasets, restricted access to powerful models, and evaluation costs hinder their adoption in industrial settings. To this end, we present a comprehensive evaluation of 13 open-source LVLMs as judges for diverse chart comprehension and reasoning tasks. We design both pairwise and pointwise evaluation tasks covering criteria like factual correctness, informativeness, and relevancy. Additionally, we analyze LVLM judges based on format adherence, positional consistency, length bias, and instruction-following. We focus on cost-effective LVLMs (<10B parameters) suitable for both research and commercial use, following a standardized evaluation protocol and rubric to measure the LVLM judge's accuracy. Experimental results reveal notable variability: while some open LVLM judges achieve GPT-4-level evaluation performance (about 80% agreement with GPT-4 judgments), others struggle (below ~10% agreement). Our findings highlight that state-of-the-art open-source LVLMs can serve as cost-effective automatic evaluators for chart-related tasks, though biases such as positional preference and length bias persist.
>
---
#### [new 046] Efficient Fairness Testing in Large Language Models: Prioritizing Metamorphic Relations for Bias Detection
- **分类: cs.CL; cs.AI; cs.SE**

- **简介: 该论文针对大语言模型（LLM）的公平性测试任务，提出基于蜕变关系（MR）优先级排序的高效偏见检测方法。为解决传统测试用例爆炸问题，通过句子多样性评估优化MR排序，提升缺陷发现效率。实验表明，该方法较随机/距离排序的故障检测率提升22%/12%，并降低计算成本，验证了其在LLM公平性测试中的有效性。**

- **链接: [http://arxiv.org/pdf/2505.07870v1](http://arxiv.org/pdf/2505.07870v1)**

> **作者:** Suavis Giramata; Madhusudan Srinivasan; Venkat Naidu Gudivada; Upulee Kanewala
>
> **摘要:** Large Language Models (LLMs) are increasingly deployed in various applications, raising critical concerns about fairness and potential biases in their outputs. This paper explores the prioritization of metamorphic relations (MRs) in metamorphic testing as a strategy to efficiently detect fairness issues within LLMs. Given the exponential growth of possible test cases, exhaustive testing is impractical; therefore, prioritizing MRs based on their effectiveness in detecting fairness violations is crucial. We apply a sentence diversity-based approach to compute and rank MRs to optimize fault detection. Experimental results demonstrate that our proposed prioritization approach improves fault detection rates by 22% compared to random prioritization and 12% compared to distance-based prioritization, while reducing the time to the first failure by 15% and 8%, respectively. Furthermore, our approach performs within 5% of fault-based prioritization in effectiveness, while significantly reducing the computational cost associated with fault labeling. These results validate the effectiveness of diversity-based MR prioritization in enhancing fairness testing for LLMs.
>
---
#### [new 047] Enhancing Cache-Augmented Generation (CAG) with Adaptive Contextual Compression for Scalable Knowledge Integration
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对大模型知识集成任务，解决缓存增强生成（CAG）在扩展动态知识库时的效率瓶颈。通过自适应上下文压缩技术动态管理预载知识，并设计混合CAG-RAG框架融合选择性检索，提升多跳推理能力与系统可扩展性，优化知识密集型任务性能。**

- **链接: [http://arxiv.org/pdf/2505.08261v1](http://arxiv.org/pdf/2505.08261v1)**

> **作者:** Rishabh Agrawal; Himanshu Kumar
>
> **摘要:** The rapid progress in large language models (LLMs) has paved the way for novel approaches in knowledge-intensive tasks. Among these, Cache-Augmented Generation (CAG) has emerged as a promising alternative to Retrieval-Augmented Generation (RAG). CAG minimizes retrieval latency and simplifies system design by preloading knowledge into the model's context. However, challenges persist in scaling CAG to accommodate large and dynamic knowledge bases effectively. This paper introduces Adaptive Contextual Compression (ACC), an innovative technique designed to dynamically compress and manage context inputs, enabling efficient utilization of the extended memory capabilities of modern LLMs. To further address the limitations of standalone CAG, we propose a Hybrid CAG-RAG Framework, which integrates selective retrieval to augment preloaded contexts in scenarios requiring additional information. Comprehensive evaluations on diverse datasets highlight the proposed methods' ability to enhance scalability, optimize efficiency, and improve multi-hop reasoning performance, offering practical solutions for real-world knowledge integration challenges.
>
---
#### [new 048] Polysemy of Synthetic Neurons Towards a New Type of Explanatory Categorical Vector Spaces
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于AI模型可解释性研究，针对语言模型中合成神经元的多义性问题。现有理论认为多义性源于潜在空间特征叠加，作者提出新几何框架：将n层神经元定义为基于n-1层子维度构建的非正交分类向量空间，通过激活空间结构和内部注意力机制识别关键分类区域，提升模型效率。**

- **链接: [http://arxiv.org/pdf/2505.07831v1](http://arxiv.org/pdf/2505.07831v1)**

> **作者:** Michael Pichat; William Pogrund; Paloma Pichat; Judicael Poumay; Armanouche Gasparian; Samuel Demarchi; Martin Corbet; Alois Georgeon; Michael Veillet-Guillem
>
> **摘要:** The polysemantic nature of synthetic neurons in artificial intelligence language models is currently understood as the result of a necessary superposition of distributed features within the latent space. We propose an alternative approach, geometrically defining a neuron in layer n as a categorical vector space with a non-orthogonal basis, composed of categorical sub-dimensions extracted from preceding neurons in layer n-1. This categorical vector space is structured by the activation space of each neuron and enables, via an intra-neuronal attention process, the identification and utilization of a critical categorical zone for the efficiency of the language model - more homogeneous and located at the intersection of these different categorical sub-dimensions.
>
---
#### [new 049] LongCodeBench: Evaluating Coding LLMs at 1M Context Windows
- **分类: cs.CL; cs.AI**

- **简介: 论文提出LongCodeBench（LCB），针对长上下文编码任务（如代码理解与修复）构建评估基准，解决现有模型在百万级上下文窗口下性能评估缺失的问题。通过整合真实GitHub问题设计QA和缺陷修复任务，分层测试模型能力，发现主流模型（如Claude 3.5、Qwen2.5）在长上下文场景下性能显著下降。**

- **链接: [http://arxiv.org/pdf/2505.07897v1](http://arxiv.org/pdf/2505.07897v1)**

> **作者:** Stefano Rando; Luca Romani; Alessio Sampieri; Yuta Kyuragi; Luca Franco; Fabio Galasso; Tatsunori Hashimoto; John Yang
>
> **摘要:** Context lengths for models have grown rapidly, from thousands to millions of tokens in just a few years. The extreme context sizes of modern long-context models have made it difficult to construct realistic long-context benchmarks -- not only due to the cost of collecting million-context tasks but also in identifying realistic scenarios that require significant contexts. We identify code comprehension and repair as a natural testbed and challenge task for long-context models and introduce LongCodeBench (LCB), a benchmark to test LLM coding abilities in long-context scenarios. Our benchmark tests both the comprehension and repair capabilities of LCLMs in realistic and important settings by drawing from real-world GitHub issues and constructing QA (LongCodeQA) and bug fixing (LongSWE-Bench) tasks. We carefully stratify the complexity of our benchmark, enabling us to evaluate models across different scales -- ranging from Qwen2.5 14B Instruct to Google's flagship Gemini model. We find that long-context remains a weakness for all models, with performance drops such as from 29% to 3% for Claude 3.5 Sonnet, or from 70.2% to 40% for Qwen2.5.
>
---
#### [new 050] Large Language Models Meet Stance Detection: A Survey of Tasks, Methods, Applications, Challenges and Future Directions
- **分类: cs.CL; cs.LG; cs.SI**

- **简介: 该论文为综述研究，针对现有LLM在立场检测中的系统性总结不足，系统分析了LLM驱动的立场检测方法、数据集与应用，提出基于学习方式、数据模态和目标关系的分类法，并讨论技术挑战及未来方向，旨在指导下一代系统开发。**

- **链接: [http://arxiv.org/pdf/2505.08464v1](http://arxiv.org/pdf/2505.08464v1)**

> **作者:** Lata Pangtey; Anukriti Bhatnagar; Shubhi Bansal; Shahid Shafi Dar; Nagendra Kumar
>
> **摘要:** Stance detection is essential for understanding subjective content across various platforms such as social media, news articles, and online reviews. Recent advances in Large Language Models (LLMs) have revolutionized stance detection by introducing novel capabilities in contextual understanding, cross-domain generalization, and multimodal analysis. Despite these progressions, existing surveys often lack comprehensive coverage of approaches that specifically leverage LLMs for stance detection. To bridge this critical gap, our review article conducts a systematic analysis of stance detection, comprehensively examining recent advancements of LLMs transforming the field, including foundational concepts, methodologies, datasets, applications, and emerging challenges. We present a novel taxonomy for LLM-based stance detection approaches, structured along three key dimensions: 1) learning methods, including supervised, unsupervised, few-shot, and zero-shot; 2) data modalities, such as unimodal, multimodal, and hybrid; and 3) target relationships, encompassing in-target, cross-target, and multi-target scenarios. Furthermore, we discuss the evaluation techniques and analyze benchmark datasets and performance trends, highlighting the strengths and limitations of different architectures. Key applications in misinformation detection, political analysis, public health monitoring, and social media moderation are discussed. Finally, we identify critical challenges such as implicit stance expression, cultural biases, and computational constraints, while outlining promising future directions, including explainable stance reasoning, low-resource adaptation, and real-time deployment frameworks. Our survey highlights emerging trends, open challenges, and future directions to guide researchers and practitioners in developing next-generation stance detection systems powered by large language models.
>
---
#### [new 051] BioProBench: Comprehensive Dataset and Benchmark in Biological Protocol Understanding and Reasoning
- **分类: cs.CL**

- **简介: 该论文属于生物协议理解与推理的基准构建任务，旨在解决大模型（LLMs）在专业流程文本上评估不足的问题。研究者创建了BioProBench，包含5个核心任务的大规模数据集（27K协议生成556K实例），测试了12个主流模型，发现模型在深层推理、结构化生成任务中表现薄弱，且生物专用模型落后于通用LLMs，为改进AI系统提供诊断框架。**

- **链接: [http://arxiv.org/pdf/2505.07889v1](http://arxiv.org/pdf/2505.07889v1)**

> **作者:** Yuyang Liu; Liuzhenghao Lv; Xiancheng Zhang; Li Yuan; Yonghong Tian
>
> **摘要:** Biological protocols are fundamental to reproducible and safe life science research. While LLMs excel on general tasks, their systematic evaluation on these highly specialized, accuracy-critical, and inherently procedural texts remains limited. In this work, we present BioProBench, the first large-scale, integrated multi-task benchmark for biological protocol understanding and reasoning. While limited benchmarks have touched upon specific aspects like protocol QA, BioProBench provides a comprehensive suite of five core tasks: Protocol Question Answering, Step Ordering, Error Correction, Protocol Generation, and Protocol Reasoning, enabling a holistic evaluation of LLMs on procedural biological texts. Built upon 27K original protocols, it yields nearly 556K high-quality structured instances. We evaluate 12 mainstream open/closed-source LLMs on BioProBench. Experimental results reveal that while top models preform well on surface understanding tasks, struggle significantly with deep reasoning and structured generation tasks like ordering and generation. Furthermore, model comparisons reveal diverse performance: certain open-source models approach closed-source levels on some tasks, yet bio-specific small models lag behind general LLMs, indicating limitations on complex procedural content. Overall, our findings underscore that procedural reasoning within biological protocols represents a significant challenge for current LLMs. BioProBench serves as a standardized framework to diagnose these specific limitations and guide the development of AI systems better equipped for safely automating complex scientific procedures. The code and data are available at: https://github.com/YuyangSunshine/bioprotocolbench and https://huggingface.co/datasets/GreatCaptainNemo/BioProBench.
>
---
#### [new 052] Enhancing Thyroid Cytology Diagnosis with RAG-Optimized LLMs and Pa-thology Foundation Models
- **分类: cs.CL; q-bio.QM**

- **简介: 该论文属于医疗AI辅助诊断任务，旨在提升甲状腺细胞学诊断的准确性与标准化。研究通过融合检索增强生成（RAG）优化的大语言模型与病理基础模型，整合动态知识库检索和高分辨率图像特征分析，解决细胞学判读差异大、诊断标准不统一的问题。实验表明模型能有效区分甲状腺病变良恶性，UNI基础模型的AUC达0.73-0.93。**

- **链接: [http://arxiv.org/pdf/2505.08590v1](http://arxiv.org/pdf/2505.08590v1)**

> **作者:** Hussien Al-Asi; Jordan P Reynolds; Shweta Agarwal; Bryan J Dangott; Aziza Nassar; Zeynettin Akkus
>
> **摘要:** Advancements in artificial intelligence (AI) are transforming pathology by integrat-ing large language models (LLMs) with retrieval-augmented generation (RAG) and domain-specific foundation models. This study explores the application of RAG-enhanced LLMs coupled with pathology foundation models for thyroid cytology diagnosis, addressing challenges in cytological interpretation, standardization, and diagnostic accuracy. By leveraging a curated knowledge base, RAG facilitates dy-namic retrieval of relevant case studies, diagnostic criteria, and expert interpreta-tion, improving the contextual understanding of LLMs. Meanwhile, pathology foun-dation models, trained on high-resolution pathology images, refine feature extrac-tion and classification capabilities. The fusion of these AI-driven approaches en-hances diagnostic consistency, reduces variability, and supports pathologists in dis-tinguishing benign from malignant thyroid lesions. Our results demonstrate that integrating RAG with pathology-specific LLMs significantly improves diagnostic efficiency and interpretability, paving the way for AI-assisted thyroid cytopathology, with foundation model UNI achieving AUC 0.73-0.93 for correct prediction of surgi-cal pathology diagnosis from thyroid cytology samples.
>
---
#### [new 053] Fusing Bidirectional Chains of Thought and Reward Mechanisms A Method for Enhancing Question-Answering Capabilities of Large Language Models for Chinese Intangible Cultural Heritage
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于领域特定问答任务，旨在解决大模型在非遗知识微调中的偏差、错误继承和遗忘问题。提出结合双向思维链（正反向推理）与奖励机制的方法，基于ICH-Qwen模型，通过反向激活知识提升答案准确性，并引入加权评估优化训练。实验表明其在多指标和跨领域数据集上优于基线方法。**

- **链接: [http://arxiv.org/pdf/2505.08167v1](http://arxiv.org/pdf/2505.08167v1)**

> **作者:** Ruilin Liu; Zhixiao Zhao; Jieqiong Li; Chang Liu; Dongbo Wang
>
> **备注:** 22 pages, 5 figures
>
> **摘要:** The rapid development of large language models (LLMs) has provided significant support and opportunities for the advancement of domain-specific LLMs. However, fine-tuning these large models using Intangible Cultural Heritage (ICH) data inevitably faces challenges such as bias, incorrect knowledge inheritance, and catastrophic forgetting. To address these issues, we propose a novel training method that integrates a bidirectional chains of thought and a reward mechanism. This method is built upon ICH-Qwen, a large language model specifically designed for the field of intangible cultural heritage. The proposed method enables the model to not only perform forward reasoning but also enhances the accuracy of the generated answers by utilizing reverse questioning and reverse reasoning to activate the model's latent knowledge. Additionally, a reward mechanism is introduced during training to optimize the decision-making process. This mechanism improves the quality of the model's outputs through structural and content evaluations with different weighting schemes. We conduct comparative experiments on ICH-Qwen, with results demonstrating that our method outperforms 0-shot, step-by-step reasoning, knowledge distillation, and question augmentation methods in terms of accuracy, Bleu-4, and Rouge-L scores on the question-answering task. Furthermore, the paper highlights the effectiveness of combining the bidirectional chains of thought and reward mechanism through ablation experiments. In addition, a series of generalizability experiments are conducted, with results showing that the proposed method yields improvements on various domain-specific datasets and advanced models in areas such as Finance, Wikidata, and StrategyQA. This demonstrates that the method is adaptable to multiple domains and provides a valuable approach for model training in future applications across diverse fields.
>
---
#### [new 054] TSLFormer: A Lightweight Transformer Model for Turkish Sign Language Recognition Using Skeletal Landmarks
- **分类: cs.CL; eess.IV**

- **简介: 该论文研究土耳其手语识别任务，提出轻量级Transformer模型TSLFormer，通过骨骼关键点降低输入维度，将手势视为序列进行翻译。解决了传统视频处理方法计算量大、难以实时应用的问题，在36,000样本数据集上验证了高效性与准确性，适用于听障人士辅助系统。**

- **链接: [http://arxiv.org/pdf/2505.07890v1](http://arxiv.org/pdf/2505.07890v1)**

> **作者:** Kutay Ertürk; Furkan Altınışık; İrem Sarıaltın; Ömer Nezih Gerek
>
> **摘要:** This study presents TSLFormer, a light and robust word-level Turkish Sign Language (TSL) recognition model that treats sign gestures as ordered, string-like language. Instead of using raw RGB or depth videos, our method only works with 3D joint positions - articulation points - extracted using Google's Mediapipe library, which focuses on the hand and torso skeletal locations. This creates efficient input dimensionality reduction while preserving important semantic gesture information. Our approach revisits sign language recognition as sequence-to-sequence translation, inspired by the linguistic nature of sign languages and the success of transformers in natural language processing. Since TSLFormer uses the self-attention mechanism, it effectively captures temporal co-occurrence within gesture sequences and highlights meaningful motion patterns as words unfold. Evaluated on the AUTSL dataset with over 36,000 samples and 227 different words, TSLFormer achieves competitive performance with minimal computational cost. These results show that joint-based input is sufficient for enabling real-time, mobile, and assistive communication systems for hearing-impaired individuals.
>
---
#### [new 055] Evaluating Financial Sentiment Analysis with Annotators Instruction Assisted Prompting: Enhancing Contextual Interpretation and Stock Prediction Accuracy
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究金融情感分析任务，旨在解决传统数据集因标注主观性导致大模型（LLMs）评估偏差的问题。通过设计标注者指令辅助提示（AIAP），将人工标注规则整合到模型提示中，标准化情感理解，并在WallStreetBets数据集上验证有效性。方法提升LLMs性能达9.08%，并提出基于置信度的情感索引，优化股票预测模型。**

- **链接: [http://arxiv.org/pdf/2505.07871v1](http://arxiv.org/pdf/2505.07871v1)**

> **作者:** A M Muntasir Rahman; Ajim Uddin; Guiling "Grace" Wang
>
> **摘要:** Financial sentiment analysis (FSA) presents unique challenges to LLMs that surpass those in typical sentiment analysis due to the nuanced language used in financial contexts. The prowess of these models is often undermined by the inherent subjectivity of sentiment classifications in existing benchmark datasets like Financial Phrasebank. These datasets typically feature undefined sentiment classes that reflect the highly individualized perspectives of annotators, leading to significant variability in annotations. This variability results in an unfair expectation for LLMs during benchmarking, where they are tasked to conjecture the subjective viewpoints of human annotators without sufficient context. In this paper, we introduce the Annotators' Instruction Assisted Prompt, a novel evaluation prompt designed to redefine the task definition of FSA for LLMs. By integrating detailed task instructions originally intended for human annotators into the LLMs' prompt framework, AIAP aims to standardize the understanding of sentiment across both human and machine interpretations, providing a fair and context-rich foundation for sentiment analysis. We utilize a new dataset, WSBS, derived from the WallStreetBets subreddit to demonstrate how AIAP significantly enhances LLM performance by aligning machine operations with the refined task definitions. Experimental results demonstrate that AIAP enhances LLM performance significantly, with improvements up to 9.08. This context-aware approach not only yields incremental gains in performance but also introduces an innovative sentiment-indexing method utilizing model confidence scores. This method enhances stock price prediction models and extracts more value from the financial sentiment analysis, underscoring the significance of WSB as a critical source of financial text. Our research offers insights into both improving FSA through better evaluation methods.
>
---
#### [new 056] Unpacking Robustness in Inflectional Languages: Adversarial Evaluation and Mechanistic Insights
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究屈折语（如波兰语）对抗样本攻击下的模型鲁棒性，属于对抗鲁棒性评估任务。针对现有方法主要面向英语等非屈折语的局限，提出基于边归因修补（EAP）的机制解释方法，构建波兰语/英语平行语料库和新基准MultiEmo，分析词形变化对模型抗攻击能力的影响机制。**

- **链接: [http://arxiv.org/pdf/2505.07856v1](http://arxiv.org/pdf/2505.07856v1)**

> **作者:** Paweł Walkowiak; Marek Klonowski; Marcin Oleksy; Arkadiusz Janz
>
> **摘要:** Various techniques are used in the generation of adversarial examples, including methods such as TextBugger which introduce minor, hardly visible perturbations to words leading to changes in model behaviour. Another class of techniques involves substituting words with their synonyms in a way that preserves the text's meaning but alters its predicted class, with TextFooler being a prominent example of such attacks. Most adversarial example generation methods are developed and evaluated primarily on non-inflectional languages, typically English. In this work, we evaluate and explain how adversarial attacks perform in inflectional languages. To explain the impact of inflection on model behaviour and its robustness under attack, we designed a novel protocol inspired by mechanistic interpretability, based on Edge Attribution Patching (EAP) method. The proposed evaluation protocol relies on parallel task-specific corpora that include both inflected and syncretic variants of texts in two languages -- Polish and English. To analyse the models and explain the relationship between inflection and adversarial robustness, we create a new benchmark based on task-oriented dataset MultiEmo, enabling the identification of mechanistic inflection-related elements of circuits within the model and analyse their behaviour under attack.
>
---
#### [new 057] Revealing economic facts: LLMs know more than they say
- **分类: cs.CL; cs.LG; econ.GN; q-fin.EC; I.2.7**

- **简介: 该论文属于经济数据估计与填补任务，研究如何利用大语言模型（LLMs）隐藏状态挖掘其未直接表达的经济信息。通过训练隐藏状态的线性模型，在县级（失业率）和企业级（总资产）数据预测上超越模型文本输出，证明隐藏状态蕴含更丰富信息。提出小样本训练、无目标标签迁移学习方法，并验证其在数据超分辨率和缺失填补中的实用性。**

- **链接: [http://arxiv.org/pdf/2505.08662v1](http://arxiv.org/pdf/2505.08662v1)**

> **作者:** Marcus Buckmann; Quynh Anh Nguyen; Edward Hill
>
> **备注:** 34 pages, 17 figures
>
> **摘要:** We investigate whether the hidden states of large language models (LLMs) can be used to estimate and impute economic and financial statistics. Focusing on county-level (e.g. unemployment) and firm-level (e.g. total assets) variables, we show that a simple linear model trained on the hidden states of open-source LLMs outperforms the models' text outputs. This suggests that hidden states capture richer economic information than the responses of the LLMs reveal directly. A learning curve analysis indicates that only a few dozen labelled examples are sufficient for training. We also propose a transfer learning method that improves estimation accuracy without requiring any labelled data for the target variable. Finally, we demonstrate the practical utility of hidden-state representations in super-resolution and data imputation tasks.
>
---
#### [new 058] Alignment Drift in CEFR-prompted LLMs for Interactive Spanish Tutoring
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLMs）作为西班牙语自适应教学工具的可行性，属于教育技术任务。旨在解决系统提示能否稳定控制LLM生成符合学生CEFR语言等级（A1-C1）的文本难度问题。通过让LLM交替扮演师生角色模拟对话，发现仅靠提示无法维持长期交互稳定性（对齐漂移），并提出低成本模型评估方法。**

- **链接: [http://arxiv.org/pdf/2505.08351v1](http://arxiv.org/pdf/2505.08351v1)**

> **作者:** Mina Almasi; Ross Deans Kristensen-McLachlan
>
> **摘要:** This paper investigates the potentials of Large Language Models (LLMs) as adaptive tutors in the context of second-language learning. In particular, we evaluate whether system prompting can reliably constrain LLMs to generate only text appropriate to the student's competence level. We simulate full teacher-student dialogues in Spanish using instruction-tuned, open-source LLMs ranging in size from 7B to 12B parameters. Dialogues are generated by having an LLM alternate between tutor and student roles with separate chat histories. The output from the tutor model is then used to evaluate the effectiveness of CEFR-based prompting to control text difficulty across three proficiency levels (A1, B1, C1). Our findings suggest that while system prompting can be used to constrain model outputs, prompting alone is too brittle for sustained, long-term interactional contexts - a phenomenon we term alignment drift. Our results provide insights into the feasibility of LLMs for personalized, proficiency-aligned adaptive tutors and provide a scalable method for low-cost evaluation of model performance without human participants.
>
---
#### [new 059] Task-Adaptive Semantic Communications with Controllable Diffusion-based Data Regeneration
- **分类: cs.CL; C.2.1; I.4.8**

- **简介: 该论文研究任务自适应语义通信，旨在动态优化语义信息传输以适应不同下游任务。基于扩散模型，提出发送压缩语义表征实现初步重建，接收端反馈任务需求提示，结合注意力机制增强关键细节传输，解决传统通信带宽效率不足问题，提升任务相关信息的保留与压缩性能。**

- **链接: [http://arxiv.org/pdf/2505.07980v1](http://arxiv.org/pdf/2505.07980v1)**

> **作者:** Fupei Guo; Achintha Wijesinghe; Songyang Zhang; Zhi Ding
>
> **摘要:** Semantic communications represent a new paradigm of next-generation networking that shifts bit-wise data delivery to conveying the semantic meanings for bandwidth efficiency. To effectively accommodate various potential downstream tasks at the receiver side, one should adaptively convey the most critical semantic information. This work presents a novel task-adaptive semantic communication framework based on diffusion models that is capable of dynamically adjusting the semantic message delivery according to various downstream tasks. Specifically, we initialize the transmission of a deep-compressed general semantic representation from the transmitter to enable diffusion-based coarse data reconstruction at the receiver. The receiver identifies the task-specific demands and generates textual prompts as feedback. Integrated with the attention mechanism, the transmitter updates the semantic transmission with more details to better align with the objectives of the intended receivers. Our test results demonstrate the efficacy of the proposed method in adaptively preserving critical task-relevant information for semantic communications while preserving high compression efficiency.
>
---
#### [new 060] HealthBench: Evaluating Large Language Models Towards Improved Human Health
- **分类: cs.CL**

- **简介: 该论文提出HealthBench，用于评估大语言模型在医疗领域的性能与安全性，属于AI模型评估任务。针对现有医疗评测缺乏开放性对话的问题，构建了含5000条多轮对话的基准，由医生制定4.8万项评分标准。研究显示模型性能持续提升（如GPT-4o达32%），并发布两个优化版本支撑医疗应用发展。**

- **链接: [http://arxiv.org/pdf/2505.08775v1](http://arxiv.org/pdf/2505.08775v1)**

> **作者:** Rahul K. Arora; Jason Wei; Rebecca Soskin Hicks; Preston Bowman; Joaquin Quiñonero-Candela; Foivos Tsimpourlas; Michael Sharman; Meghan Shah; Andrea Vallone; Alex Beutel; Johannes Heidecke; Karan Singhal
>
> **备注:** Blog: https://openai.com/index/healthbench/ Code: https://github.com/openai/simple-evals
>
> **摘要:** We present HealthBench, an open-source benchmark measuring the performance and safety of large language models in healthcare. HealthBench consists of 5,000 multi-turn conversations between a model and an individual user or healthcare professional. Responses are evaluated using conversation-specific rubrics created by 262 physicians. Unlike previous multiple-choice or short-answer benchmarks, HealthBench enables realistic, open-ended evaluation through 48,562 unique rubric criteria spanning several health contexts (e.g., emergencies, transforming clinical data, global health) and behavioral dimensions (e.g., accuracy, instruction following, communication). HealthBench performance over the last two years reflects steady initial progress (compare GPT-3.5 Turbo's 16% to GPT-4o's 32%) and more rapid recent improvements (o3 scores 60%). Smaller models have especially improved: GPT-4.1 nano outperforms GPT-4o and is 25 times cheaper. We additionally release two HealthBench variations: HealthBench Consensus, which includes 34 particularly important dimensions of model behavior validated via physician consensus, and HealthBench Hard, where the current top score is 32%. We hope that HealthBench grounds progress towards model development and applications that benefit human health.
>
---
#### [new 061] AM-Thinking-v1: Advancing the Frontier of Reasoning at 32B Scale
- **分类: cs.CL**

- **简介: 该论文提出AM-Thinking-v1（32B参数语言模型），聚焦提升推理能力，解决中等规模开源模型性能与实用性平衡问题。通过监督微调和强化学习优化Qwen2.5-32B基础模型，在数学（AIME 85.3）和编码（LiveCodeBench 70.3）任务上超越同类模型，验证32B规模开源方案的高效部署潜力，并开源模型推动协作创新。**

- **链接: [http://arxiv.org/pdf/2505.08311v1](http://arxiv.org/pdf/2505.08311v1)**

> **作者:** Yunjie Ji; Xiaoyu Tian; Sitong Zhao; Haotian Wang; Shuaiting Chen; Yiping Peng; Han Zhao; Xiangang Li
>
> **摘要:** We present AM-Thinking-v1, a 32B dense language model that advances the frontier of reasoning, embodying the collaborative spirit of open-source innovation. Outperforming DeepSeek-R1 and rivaling leading Mixture-of-Experts (MoE) models like Qwen3-235B-A22B and Seed1.5-Thinking, AM-Thinking-v1 achieves impressive scores of 85.3 on AIME 2024, 74.4 on AIME 2025, and 70.3 on LiveCodeBench, showcasing state-of-the-art mathematical and coding capabilities among open-source models of similar scale. Built entirely from the open-source Qwen2.5-32B base model and publicly available queries, AM-Thinking-v1 leverages a meticulously crafted post-training pipeline - combining supervised fine-tuning and reinforcement learning - to deliver exceptional reasoning capabilities. This work demonstrates that the open-source community can achieve high performance at the 32B scale, a practical sweet spot for deployment and fine-tuning. By striking a balance between top-tier performance and real-world usability, we hope AM-Thinking-v1 inspires further collaborative efforts to harness mid-scale models, pushing reasoning boundaries while keeping accessibility at the core of innovation. We have open-sourced our model on \href{https://huggingface.co/a-m-team/AM-Thinking-v1}{Hugging Face}.
>
---
#### [new 062] Large Language Model Psychometrics: A Systematic Review of Evaluation, Validation, and Enhancement
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文为系统综述，属于跨学科研究，针对大语言模型（LLMs）传统评估方法的局限性，提出整合心理测量学理论，解决评估类人心理特质、动态基准构建及人本化验证等问题。通过系统梳理LLM心理测量学框架，优化评测原则、方法及验证体系，推动人本AI系统发展，并提供开源资源库。**

- **链接: [http://arxiv.org/pdf/2505.08245v1](http://arxiv.org/pdf/2505.08245v1)**

> **作者:** Haoran Ye; Jing Jin; Yuhang Xie; Xin Zhang; Guojie Song
>
> **备注:** 63 pages, 482 references
>
> **摘要:** The rapid advancement of large language models (LLMs) has outpaced traditional evaluation methodologies. It presents novel challenges, such as measuring human-like psychological constructs, navigating beyond static and task-specific benchmarks, and establishing human-centered evaluation. These challenges intersect with Psychometrics, the science of quantifying the intangible aspects of human psychology, such as personality, values, and intelligence. This survey introduces and synthesizes an emerging interdisciplinary field of LLM Psychometrics, which leverages psychometric instruments, theories, and principles to evaluate, understand, and enhance LLMs. We systematically explore the role of Psychometrics in shaping benchmarking principles, broadening evaluation scopes, refining methodologies, validating results, and advancing LLM capabilities. This paper integrates diverse perspectives to provide a structured framework for researchers across disciplines, enabling a more comprehensive understanding of this nascent field. Ultimately, we aim to provide actionable insights for developing future evaluation paradigms that align with human-level AI and promote the advancement of human-centered AI systems for societal benefit. A curated repository of LLM psychometric resources is available at https://github.com/valuebyte-ai/Awesome-LLM-Psychometrics.
>
---
#### [new 063] Scaling Laws for Speculative Decoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究大模型推测解码的扩展规律，解决推理效率优化问题。通过分析预训练数据量、模型容量和批处理规模对解码速度的影响，提出对数线性扩展法则，并开发Scylla系统实现多维优化，在主流模型上验证了吞吐量翻倍的效果，提升摘要和问答任务性能。**

- **链接: [http://arxiv.org/pdf/2505.07858v1](http://arxiv.org/pdf/2505.07858v1)**

> **作者:** Siyuan Yan; Mo Zhu; Guo-qing Jiang; Jianfei Wang; Jiaxing Chen; Wentai Zhang; Xiang Liao; Xiao Cui; Chen Zhang; Zhuoran Song; Ran Zhu
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** The escalating demand for efficient decoding in large language models (LLMs) is particularly critical for reasoning-intensive architectures like OpenAI-o3 and DeepSeek-R1, which depend on extended chain-of-thought reasoning. This study investigates speculative decoding techniques through dense LLM architectures to establish foundational insights for accelerating reasoning tasks. While speculative decoding methods leveraging parallel draft-verification cycles have emerged as promising acceleration techniques, the scaling laws governing decoding efficiency remain under-explored compared to conventional backbone LLMs developed through Pretraining->SFT->RLHF training paradigms. In this work, we discover Log-linear Scaling Laws (Theorem 1.1, 1.2 and 1.3) governing draft model acceptance rate (or decoding speed) across three dimensions: pretraining token volume, draft model capacity, and decoding batch size. Building on these laws, we achieve Scylla, which coordinates multi-dimensional scaling for popular LLMs (Llama2/3, Qwen2.5). Empirical validation shows Scylla achieves 1.5-2.2 higher acceptance rate than EAGLE2 and 0.3 higher than EAGLE3 at temperature T = 0, with peak performance gains on summarization and QA tasks (Figure 2). Industrial inference engine deployments demonstrate 2X decoding throughput improvements over EAGLE2 (Table 5), validating the transformative potential of systematic scaling for efficient LLM inference. Code will be released later.
>
---
#### [new 064] Enhanced Urdu Intent Detection with Large Language Models and Prototype-Informed Predictive Pipelines
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对乌尔都语意图检测任务，解决其缺乏少样本学习策略的问题。提出结合对比学习与预训练大语言模型的方法，利用未标注数据优化模型表征，并构建包含原型注意力机制的LLMPIA管道。实验在ATIS和Web Queries数据集验证有效性，少样本场景下F1显著超越现有方法。**

- **链接: [http://arxiv.org/pdf/2505.07857v1](http://arxiv.org/pdf/2505.07857v1)**

> **作者:** Faiza Hassan; Summra Saleem; Kashif Javed; Muhammad Nabeel Asim; Abdur Rehman; Andreas Dengel
>
> **备注:** 42 pages, 10 figures(including 6 graphs)
>
> **摘要:** Multifarious intent detection predictors are developed for different languages, including English, Chinese and French, however, the field remains underdeveloped for Urdu, the 10th most spoken language. In the realm of well-known languages, intent detection predictors utilize the strategy of few-shot learning and prediction of unseen classes based on the model training on seen classes. However, Urdu language lacks few-shot strategy based intent detection predictors and traditional predictors are focused on prediction of the same classes which models have seen in the train set. To empower Urdu language specific intent detection, this introduces a unique contrastive learning approach that leverages unlabeled Urdu data to re-train pre-trained language models. This re-training empowers LLMs representation learning for the downstream intent detection task. Finally, it reaps the combined potential of pre-trained LLMs and the prototype-informed attention mechanism to create a comprehensive end-to-end LLMPIA intent detection pipeline. Under the paradigm of proposed predictive pipeline, it explores the potential of 6 distinct language models and 13 distinct similarity computation methods. The proposed framework is evaluated on 2 public benchmark datasets, namely ATIS encompassing 5836 samples and Web Queries having 8519 samples. Across ATIS dataset under 4-way 1 shot and 4-way 5 shot experimental settings LLMPIA achieved 83.28% and 98.25% F1-Score and on Web Queries dataset produced 76.23% and 84.42% F1-Score, respectively. In an additional case study on the Web Queries dataset under same classes train and test set settings, LLMPIA outperformed state-of-the-art predictor by 53.55% F1-Score.
>
---
#### [new 065] Evaluating the Effectiveness of Black-Box Prompt Optimization as the Scale of LLMs Continues to Grow
- **分类: cs.CL**

- **简介: 该论文研究黑盒提示优化在大型语言模型（LLM）规模增长时的有效性，属于自然语言处理优化任务。针对现有方法在小规模模型（如7B）有效但大规模模型（如671B）效果未知的问题，实验评估了三种优化方法在DeepSeek V3等大模型上的表现，发现改进有限，并通过不同规模的Qwen系列验证了优化效果随模型增大而衰减的逆缩放规律。**

- **链接: [http://arxiv.org/pdf/2505.08303v1](http://arxiv.org/pdf/2505.08303v1)**

> **作者:** Ziyu Zhou; Yihang Wu; Jingyuan Yang; Zhan Xiao; Rongjun Li
>
> **摘要:** Black-Box prompt optimization methods have emerged as a promising strategy for refining input prompts to better align large language models (LLMs), thereby enhancing their task performance. Although these methods have demonstrated encouraging results, most studies and experiments have primarily focused on smaller-scale models (e.g., 7B, 14B) or earlier versions (e.g., GPT-3.5) of LLMs. As the scale of LLMs continues to increase, such as with DeepSeek V3 (671B), it remains an open question whether these black-box optimization techniques will continue to yield significant performance improvements for models of such scale. In response to this, we select three well-known black-box optimization methods and evaluate them on large-scale LLMs (DeepSeek V3 and Gemini 2.0 Flash) across four NLU and NLG datasets. The results show that these black-box prompt optimization methods offer only limited improvements on these large-scale LLMs. Furthermore, we hypothesize that the scale of the model is the primary factor contributing to the limited benefits observed. To explore this hypothesis, we conducted experiments on LLMs of varying sizes (Qwen 2.5 series, ranging from 7B to 72B) and observed an inverse scaling law, wherein the effectiveness of black-box optimization methods diminished as the model size increased.
>
---
#### [new 066] QoSBERT: An Uncertainty-Aware Approach based on Pre-trained Language Models for Service Quality Prediction
- **分类: cs.CL**

- **简介: 该论文提出QoSBERT框架，解决云服务质量(QoS)预测中传统方法依赖人工特征、缺乏置信度评估的问题。通过预训练语言模型将服务元数据转为语义描述，结合蒙特卡洛Dropout进行不确定性量化，在基准测试中显著降低预测误差（MAE降11.7%），同时提供可靠置信区间，实现可信的服务质量预测与优化。**

- **链接: [http://arxiv.org/pdf/2505.07863v1](http://arxiv.org/pdf/2505.07863v1)**

> **作者:** Ziliang Wang; Xiaohong Zhang; Ze Shi Li; Meng Yan
>
> **摘要:** Accurate prediction of Quality of Service (QoS) metrics is fundamental for selecting and managing cloud based services. Traditional QoS models rely on manual feature engineering and yield only point estimates, offering no insight into the confidence of their predictions. In this paper, we propose QoSBERT, the first framework that reformulates QoS prediction as a semantic regression task based on pre trained language models. Unlike previous approaches relying on sparse numerical features, QoSBERT automatically encodes user service metadata into natural language descriptions, enabling deep semantic understanding. Furthermore, we integrate a Monte Carlo Dropout based uncertainty estimation module, allowing for trustworthy and risk-aware service quality prediction, which is crucial yet underexplored in existing QoS models. QoSBERT applies attentive pooling over contextualized embeddings and a lightweight multilayer perceptron regressor, fine tuned jointly to minimize absolute error. We further exploit the resulting uncertainty estimates to select high quality training samples, improving robustness in low resource settings. On standard QoS benchmark datasets, QoSBERT achieves an average reduction of 11.7% in MAE and 6.7% in RMSE for response time prediction, and 6.9% in MAE for throughput prediction compared to the strongest baselines, while providing well calibrated confidence intervals for robust and trustworthy service quality estimation. Our approach not only advances the accuracy of service quality prediction but also delivers reliable uncertainty quantification, paving the way for more trustworthy, data driven service selection and optimization.
>
---
#### [new 067] Multimodal Assessment of Classroom Discourse Quality: A Text-Centered Attention-Based Multi-Task Learning Approach
- **分类: cs.CY; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态教育评估任务，旨在解决传统课堂话语质量人工评估效率低、现有AI方法难以全面分析整节课段的问题。提出基于注意力机制的多模态融合模型（文本、音频、视频），采用多任务学习和有序分类方法，自动评估三类教学话语质量。实验表明文本模态主导，融合音频特征后模型性能接近人类评分水平。**

- **链接: [http://arxiv.org/pdf/2505.07902v1](http://arxiv.org/pdf/2505.07902v1)**

> **作者:** Ruikun Hou; Babette Bühler; Tim Fütterer; Efe Bozkir; Peter Gerjets; Ulrich Trautwein; Enkelejda Kasneci
>
> **备注:** The 18th International Conference on Educational Data Mining (EDM 2025)
>
> **摘要:** Classroom discourse is an essential vehicle through which teaching and learning take place. Assessing different characteristics of discursive practices and linking them to student learning achievement enhances the understanding of teaching quality. Traditional assessments rely on manual coding of classroom observation protocols, which is time-consuming and costly. Despite many studies utilizing AI techniques to analyze classroom discourse at the utterance level, investigations into the evaluation of discursive practices throughout an entire lesson segment remain limited. To address this gap, our study proposes a novel text-centered multimodal fusion architecture to assess the quality of three discourse components grounded in the Global Teaching InSights (GTI) observation protocol: Nature of Discourse, Questioning, and Explanations. First, we employ attention mechanisms to capture inter- and intra-modal interactions from transcript, audio, and video streams. Second, a multi-task learning approach is adopted to jointly predict the quality scores of the three components. Third, we formulate the task as an ordinal classification problem to account for rating level order. The effectiveness of these designed elements is demonstrated through an ablation study on the GTI Germany dataset containing 92 videotaped math lessons. Our results highlight the dominant role of text modality in approaching this task. Integrating acoustic features enhances the model's consistency with human ratings, achieving an overall Quadratic Weighted Kappa score of 0.384, comparable to human inter-rater reliability (0.326). Our study lays the groundwork for the future development of automated discourse quality assessment to support teacher professional development through timely feedback on multidimensional discourse practices.
>
---
#### [new 068] Memorization-Compression Cycles Improve Generalization
- **分类: cs.LG; cs.AI; cs.CL; cs.IT; math.IT**

- **简介: 该论文属于语言模型预训练任务，旨在提升模型泛化能力并减少灾难遗忘。通过理论证明表示压缩可增强泛化，提出信息瓶颈建模目标IBLM和动态切换记忆/压缩阶段的GAPT算法，在GPT-2训练中验证了表征熵降低与泛化提升的协同效应。**

- **链接: [http://arxiv.org/pdf/2505.08727v1](http://arxiv.org/pdf/2505.08727v1)**

> **作者:** Fangyuan Yu
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** We prove theoretically that generalization improves not only through data scaling but also by compressing internal representations. To operationalize this insight, we introduce the Information Bottleneck Language Modeling (IBLM) objective, which reframes language modeling as a constrained optimization problem: minimizing representation entropy subject to optimal prediction performance. Empirically, we observe an emergent memorization-compression cycle during LLM pretraining, evidenced by oscillation positive/negative gradient alignment between cross-entropy and Matrix-Based Entropy (MBE), a measure of representation entropy. This pattern closely mirrors the predictive-compressive trade-off prescribed by IBLM and also parallels the biological alternation between awake learning and sleep consolidation. Motivated by this observation, we propose Gated Phase Transition (GAPT), a training algorithm that adaptively switches between memorization and compression phases. When applied to GPT-2 pretraining on FineWeb dataset, GAPT reduces MBE by 50% and improves cross-entropy by 4.8%. GAPT improves OOD generalizatino by 35% in a pretraining task on arithmetic multiplication. In a setting designed to simulate catastrophic forgetting, GAPT reduces interference by compressing and separating representations, achieving a 97% improvement in separation - paralleling the functional role of sleep consolidation.
>
---
#### [new 069] CellVerse: Do Large Language Models Really Understand Cell Biology?
- **分类: q-bio.QM; cs.AI; cs.CL; q-bio.CB**

- **简介: 论文评估大语言模型（LLMs）在语言驱动单细胞分析任务中的生物学理解能力，属生物信息学任务。为解决现有LLMs在细胞生物学任务中缺乏系统评估的问题，提出统一基准CellVerse，整合多组学数据并设计三层分析任务（细胞、药物、基因）。测试14个模型发现通用模型具备初步理解力，但整体性能不足，药物预测接近随机水平，揭示LLMs应用挑战，为下一代单细胞分析奠定基础。**

- **链接: [http://arxiv.org/pdf/2505.07865v1](http://arxiv.org/pdf/2505.07865v1)**

> **作者:** Fan Zhang; Tianyu Liu; Zhihong Zhu; Hao Wu; Haixin Wang; Donghao Zhou; Yefeng Zheng; Kun Wang; Xian Wu; Pheng-Ann Heng
>
> **摘要:** Recent studies have demonstrated the feasibility of modeling single-cell data as natural languages and the potential of leveraging powerful large language models (LLMs) for understanding cell biology. However, a comprehensive evaluation of LLMs' performance on language-driven single-cell analysis tasks still remains unexplored. Motivated by this challenge, we introduce CellVerse, a unified language-centric question-answering benchmark that integrates four types of single-cell multi-omics data and encompasses three hierarchical levels of single-cell analysis tasks: cell type annotation (cell-level), drug response prediction (drug-level), and perturbation analysis (gene-level). Going beyond this, we systematically evaluate the performance across 14 open-source and closed-source LLMs ranging from 160M to 671B on CellVerse. Remarkably, the experimental results reveal: (1) Existing specialist models (C2S-Pythia) fail to make reasonable decisions across all sub-tasks within CellVerse, while generalist models such as Qwen, Llama, GPT, and DeepSeek family models exhibit preliminary understanding capabilities within the realm of cell biology. (2) The performance of current LLMs falls short of expectations and has substantial room for improvement. Notably, in the widely studied drug response prediction task, none of the evaluated LLMs demonstrate significant performance improvement over random guessing. CellVerse offers the first large-scale empirical demonstration that significant challenges still remain in applying LLMs to cell biology. By introducing CellVerse, we lay the foundation for advancing cell biology through natural languages and hope this paradigm could facilitate next-generation single-cell analysis.
>
---
#### [new 070] Visually Guided Decoding: Gradient-Free Hard Prompt Inversion with Language Models
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究文本到图像模型的提示生成任务，解决现有方法生成不连贯、可解释性差的问题。提出VGD方法，通过无梯度优化结合大语言模型生成能力与CLIP视觉对齐，无需训练即可生成语义一致且易理解的提示词，提升交互可控性。**

- **链接: [http://arxiv.org/pdf/2505.08622v1](http://arxiv.org/pdf/2505.08622v1)**

> **作者:** Donghoon Kim; Minji Bae; Kyuhong Shim; Byonghyo Shim
>
> **备注:** ICLR 2025
>
> **摘要:** Text-to-image generative models like DALL-E and Stable Diffusion have revolutionized visual content creation across various applications, including advertising, personalized media, and design prototyping. However, crafting effective textual prompts to guide these models remains challenging, often requiring extensive trial and error. Existing prompt inversion approaches, such as soft and hard prompt techniques, are not so effective due to the limited interpretability and incoherent prompt generation. To address these issues, we propose Visually Guided Decoding (VGD), a gradient-free approach that leverages large language models (LLMs) and CLIP-based guidance to generate coherent and semantically aligned prompts. In essence, VGD utilizes the robust text generation capabilities of LLMs to produce human-readable prompts. Further, by employing CLIP scores to ensure alignment with user-specified visual concepts, VGD enhances the interpretability, generalization, and flexibility of prompt generation without the need for additional training. Our experiments demonstrate that VGD outperforms existing prompt inversion techniques in generating understandable and contextually relevant prompts, facilitating more intuitive and controllable interactions with text-to-image models.
>
---
#### [new 071] TRAIL: Trace Reasoning and Agentic Issue Localization
- **分类: cs.AI; cs.CL**

- **简介: 该论文针对智能体工作流中复杂轨迹评估难题，提出自动化评估方法。任务属于AI系统分析与调试，解决传统人工评估低效、错误溯源困难问题。工作包括建立错误分类体系、发布标注数据集TRAIL（含148条多场景轨迹），验证大模型调试能力差（Gemini仅11%准确），推动智能工作流评估研究。**

- **链接: [http://arxiv.org/pdf/2505.08638v1](http://arxiv.org/pdf/2505.08638v1)**

> **作者:** Darshan Deshpande; Varun Gangal; Hersh Mehta; Jitin Krishnan; Anand Kannappan; Rebecca Qian
>
> **备注:** Dataset link: https://huggingface.co/datasets/PatronusAI/TRAIL
>
> **摘要:** The increasing adoption of agentic workflows across diverse domains brings a critical need to scalably and systematically evaluate the complex traces these systems generate. Current evaluation methods depend on manual, domain-specific human analysis of lengthy workflow traces - an approach that does not scale with the growing complexity and volume of agentic outputs. Error analysis in these settings is further complicated by the interplay of external tool outputs and language model reasoning, making it more challenging than traditional software debugging. In this work, we (1) articulate the need for robust and dynamic evaluation methods for agentic workflow traces, (2) introduce a formal taxonomy of error types encountered in agentic systems, and (3) present a set of 148 large human-annotated traces (TRAIL) constructed using this taxonomy and grounded in established agentic benchmarks. To ensure ecological validity, we curate traces from both single and multi-agent systems, focusing on real-world applications such as software engineering and open-world information retrieval. Our evaluations reveal that modern long context LLMs perform poorly at trace debugging, with the best Gemini-2.5-pro model scoring a mere 11% on TRAIL. Our dataset and code are made publicly available to support and accelerate future research in scalable evaluation for agentic workflows.
>
---
#### [new 072] Not that Groove: Zero-Shot Symbolic Music Editing
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文研究符号音乐编辑任务，解决AI音频生成灵活性不足及标注数据缺乏问题。提出使用零样本提示的大语言模型编辑鼓点节奏，设计交互格式连接模型与音乐，并提供与音乐家判断一致的数据集进行评估。**

- **链接: [http://arxiv.org/pdf/2505.08203v1](http://arxiv.org/pdf/2505.08203v1)**

> **作者:** Li Zhang
>
> **摘要:** Most work in AI music generation focused on audio, which has seen limited use in the music production industry due to its rigidity. To maximize flexibility while assuming only textual instructions from producers, we are among the first to tackle symbolic music editing. We circumvent the known challenge of lack of labeled data by proving that LLMs with zero-shot prompting can effectively edit drum grooves. The recipe of success is a creatively designed format that interfaces LLMs and music, while we facilitate evaluation by providing an evaluation dataset with annotated unit tests that highly aligns with musicians' judgment.
>
---
#### [new 073] Optimizing Retrieval-Augmented Generation: Analysis of Hyperparameter Impact on Performance and Efficiency
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究检索增强生成（RAG）系统的超参数优化，属于自然语言处理任务，旨在解决大模型幻觉和知识滞后问题。通过对比Chroma/Faiss向量库、分块策略、重排序等参数对6项指标的影响，揭示速度-精度权衡，证明合理参数组合可使检索精度达99%，为医疗等关键领域提供高效优化方案。**

- **链接: [http://arxiv.org/pdf/2505.08445v1](http://arxiv.org/pdf/2505.08445v1)**

> **作者:** Adel Ammar; Anis Koubaa; Omer Nacar; Wadii Boulila
>
> **摘要:** Large language models achieve high task performance yet often hallucinate or rely on outdated knowledge. Retrieval-augmented generation (RAG) addresses these gaps by coupling generation with external search. We analyse how hyperparameters influence speed and quality in RAG systems, covering Chroma and Faiss vector stores, chunking policies, cross-encoder re-ranking, and temperature, and we evaluate six metrics: faithfulness, answer correctness, answer relevancy, context precision, context recall, and answer similarity. Chroma processes queries 13% faster, whereas Faiss yields higher retrieval precision, revealing a clear speed-accuracy trade-off. Naive fixed-length chunking with small windows and minimal overlap outperforms semantic segmentation while remaining the quickest option. Re-ranking provides modest gains in retrieval quality yet increases runtime by roughly a factor of 5, so its usefulness depends on latency constraints. These results help practitioners balance computational cost and accuracy when tuning RAG systems for transparent, up-to-date responses. Finally, we re-evaluate the top configurations with a corrective RAG workflow and show that their advantages persist when the model can iteratively request additional evidence. We obtain a near-perfect context precision (99%), which demonstrates that RAG systems can achieve extremely high retrieval accuracy with the right combination of hyperparameters, with significant implications for applications where retrieval quality directly impacts downstream task performance, such as clinical decision support in healthcare.
>
---
#### [new 074] LLM-based Prompt Ensemble for Reliable Medical Entity Recognition from EHRs
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究基于LLM的提示集成方法，解决电子健康记录（EHR）中非结构化文本的医疗实体识别问题。通过对比GPT-4o和DeepSeek-R1模型在零/少样本及集成策略下的表现，发现GPT-4o结合嵌入相似度与多数投票的集成方法效果最优（F1=0.95），提升了识别可靠性。**

- **链接: [http://arxiv.org/pdf/2505.08704v1](http://arxiv.org/pdf/2505.08704v1)**

> **作者:** K M Sajjadul Islam; Ayesha Siddika Nipu; Jiawei Wu; Praveen Madiraju
>
> **备注:** IEEE 26th International Conference on Information Reuse and Integration for Data Science (IRI 2025), San Jose, CA, USA
>
> **摘要:** Electronic Health Records (EHRs) are digital records of patient information, often containing unstructured clinical text. Named Entity Recognition (NER) is essential in EHRs for extracting key medical entities like problems, tests, and treatments to support downstream clinical applications. This paper explores prompt-based medical entity recognition using large language models (LLMs), specifically GPT-4o and DeepSeek-R1, guided by various prompt engineering techniques, including zero-shot, few-shot, and an ensemble approach. Among all strategies, GPT-4o with prompt ensemble achieved the highest classification performance with an F1-score of 0.95 and recall of 0.98, outperforming DeepSeek-R1 on the task. The ensemble method improved reliability by aggregating outputs through embedding-based similarity and majority voting.
>
---
#### [new 075] Beyond Input Activations: Identifying Influential Latents by Gradient Sparse Autoencoders
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型可解释性任务，针对稀疏自编码器(SAE)分析中忽略潜在特征对输出的因果影响问题，提出梯度稀疏自编码器(GradSAE)，通过融合输出端梯度信息识别关键潜在变量，实现更精准的模型控制和解释。**

- **链接: [http://arxiv.org/pdf/2505.08080v1](http://arxiv.org/pdf/2505.08080v1)**

> **作者:** Dong Shu; Xuansheng Wu; Haiyan Zhao; Mengnan Du; Ninghao Liu
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Sparse Autoencoders (SAEs) have recently emerged as powerful tools for interpreting and steering the internal representations of large language models (LLMs). However, conventional approaches to analyzing SAEs typically rely solely on input-side activations, without considering the causal influence between each latent feature and the model's output. This work is built on two key hypotheses: (1) activated latents do not contribute equally to the construction of the model's output, and (2) only latents with high causal influence are effective for model steering. To validate these hypotheses, we propose Gradient Sparse Autoencoder (GradSAE), a simple yet effective method that identifies the most influential latents by incorporating output-side gradient information.
>
---
#### [new 076] NAZM: Network Analysis of Zonal Metrics in Persian Poetic Tradition
- **分类: cs.SI; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于计算语言学与数字人文领域，旨在通过构建多维诗人相似网络，量化分析波斯古典诗歌传统中的影响力动态。基于Ganjoor语料提取多维度特征建立网络模型，运用中心性指标和社区检测识别核心诗人、风格枢纽及文学流派，揭示结构重要性与传统认知的差异，提供数据驱动的文学研究新范式。**

- **链接: [http://arxiv.org/pdf/2505.08052v1](http://arxiv.org/pdf/2505.08052v1)**

> **作者:** Kourosh Shahnazari; Seyed Moein Ayyoubzadeh
>
> **摘要:** This study formalizes a computational model to simulate classical Persian poets' dynamics of influence through constructing a multi-dimensional similarity network. Using a rigorously curated dataset based on Ganjoor's corpus, we draw upon semantic, lexical, stylistic, thematic, and metrical features to demarcate each poet's corpus. Each is contained within weighted similarity matrices, which are then appended to generate an aggregate graph showing poet-to-poet influence. Further network investigation is carried out to identify key poets, style hubs, and bridging poets by calculating degree, closeness, betweenness, eigenvector, and Katz centrality measures. Further, for typological insight, we use the Louvain community detection algorithm to demarcate clusters of poets sharing both style and theme coherence, which correspond closely to acknowledged schools of literature like Sabk-e Hindi, Sabk-e Khorasani, and the Bazgasht-e Adabi phenomenon. Our findings provide a new data-driven view of Persian literature distinguished between canonical significance and interextual influence, thus highlighting relatively lesser-known figures who hold great structural significance. Combining computational linguistics with literary study, this paper produces an interpretable and scalable model for poetic tradition, enabling retrospective reflection as well as forward-looking research within digital humanities.
>
---
#### [new 077] CodePDE: An Inference Framework for LLM-driven PDE Solver Generation
- **分类: cs.LG; cs.AI; cs.CL; cs.NA; math.NA**

- **简介: 该论文提出CodePDE框架，将偏微分方程(PDE)求解转化为代码生成任务，利用大语言模型自动生成求解器。解决传统数值方法依赖专家知识、计算成本高，以及神经网络求解器需大量数据、缺乏可解释性的问题。通过推理算法和扩展策略实现模型自优化与测试扩展，无需针对性训练，在多个PDE问题上达到超人类性能，并系统分析了生成求解器的特性与局限。**

- **链接: [http://arxiv.org/pdf/2505.08783v1](http://arxiv.org/pdf/2505.08783v1)**

> **作者:** Shanda Li; Tanya Marwah; Junhong Shen; Weiwei Sun; Andrej Risteski; Yiming Yang; Ameet Talwalkar
>
> **摘要:** Partial differential equations (PDEs) are fundamental to modeling physical systems, yet solving them remains a complex challenge. Traditional numerical solvers rely on expert knowledge to implement and are computationally expensive, while neural-network-based solvers require large training datasets and often lack interpretability. In this work, we frame PDE solving as a code generation task and introduce CodePDE, the first inference framework for generating PDE solvers using large language models (LLMs). Leveraging advanced inference-time algorithms and scaling strategies, CodePDE unlocks critical capacities of LLM for PDE solving: reasoning, debugging, selfrefinement, and test-time scaling -- all without task-specific tuning. CodePDE achieves superhuman performance across a range of representative PDE problems. We also present a systematic empirical analysis of LLM generated solvers, analyzing their accuracy, efficiency, and numerical scheme choices. Our findings highlight the promise and the current limitations of LLMs in PDE solving, offering a new perspective on solver design and opportunities for future model development. Our code is available at https://github.com/LithiumDA/CodePDE.
>
---
#### [new 078] A Large-Scale Empirical Analysis of Custom GPTs' Vulnerabilities in the OpenAI Ecosystem
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于安全漏洞实证分析，针对OpenAI生态中自定义GPTs的安全风险，解决现有研究缺乏大规模实证评估的问题。通过分析14,904个案例，量化七类威胁（如钓鱼、代码攻击）的普遍性，揭示95%以上模型防护不足，并发现基础模型缺陷会加剧定制化产品的漏洞，强调强化安全措施的必要性。**

- **链接: [http://arxiv.org/pdf/2505.08148v1](http://arxiv.org/pdf/2505.08148v1)**

> **作者:** Sunday Oyinlola Ogundoyin; Muhammad Ikram; Hassan Jameel Asghar; Benjamin Zi Hao Zhao; Dali Kaafar
>
> **摘要:** Millions of users leverage generative pretrained transformer (GPT)-based language models developed by leading model providers for a wide range of tasks. To support enhanced user interaction and customization, many platforms-such as OpenAI-now enable developers to create and publish tailored model instances, known as custom GPTs, via dedicated repositories or application stores. These custom GPTs empower users to browse and interact with specialized applications designed to meet specific needs. However, as custom GPTs see growing adoption, concerns regarding their security vulnerabilities have intensified. Existing research on these vulnerabilities remains largely theoretical, often lacking empirical, large-scale, and statistically rigorous assessments of associated risks. In this study, we analyze 14,904 custom GPTs to assess their susceptibility to seven exploitable threats, such as roleplay-based attacks, system prompt leakage, phishing content generation, and malicious code synthesis, across various categories and popularity tiers within the OpenAI marketplace. We introduce a multi-metric ranking system to examine the relationship between a custom GPT's popularity and its associated security risks. Our findings reveal that over 95% of custom GPTs lack adequate security protections. The most prevalent vulnerabilities include roleplay-based vulnerabilities (96.51%), system prompt leakage (92.20%), and phishing (91.22%). Furthermore, we demonstrate that OpenAI's foundational models exhibit inherent security weaknesses, which are often inherited or amplified in custom GPTs. These results highlight the urgent need for enhanced security measures and stricter content moderation to ensure the safe deployment of GPT-based applications.
>
---
#### [new 079] SciCom Wiki: Fact-Checking and FAIR Knowledge Distribution for Scientific Videos and Podcasts
- **分类: cs.DL; cs.CL; cs.MM**

- **简介: 该论文提出SciCom Wiki平台，解决科学视频/播客中的错误信息及知识基础设施碎片化问题。任务为构建FAIR（可寻、可访问、互操作、可重用）科学传播知识库，开发神经符号计算事实核查工具，将媒体转为知识图谱以对比验证内容。通过调研需求、开发开源系统及多轮评估验证可行性，强调协作应对信息洪流。**

- **链接: [http://arxiv.org/pdf/2505.07912v1](http://arxiv.org/pdf/2505.07912v1)**

> **作者:** Tim Wittenborg; Constantin Sebastian Tremel; Niklas Stehr; Oliver Karras; Markus Stocker; Sören Auer
>
> **备注:** 18 pages, 10 figures, submitted to TPDL 2025
>
> **摘要:** Democratic societies need accessible, reliable information. Videos and Podcasts have established themselves as the medium of choice for civic dissemination, but also as carriers of misinformation. The emerging Science Communication Knowledge Infrastructure (SciCom KI) curating non-textual media is still fragmented and not adequately equipped to scale against the content flood. Our work sets out to support the SciCom KI with a central, collaborative platform, the SciCom Wiki, to facilitate FAIR (findable, accessible, interoperable, reusable) media representation and the fact-checking of their content, particularly for videos and podcasts. Building an open-source service system centered around Wikibase, we survey requirements from 53 stakeholders, refine these in 11 interviews, and evaluate our prototype based on these requirements with another 14 participants. To address the most requested feature, fact-checking, we developed a neurosymbolic computational fact-checking approach, converting heterogenous media into knowledge graphs. This increases machine-readability and allows comparing statements against equally represented ground-truth. Our computational fact-checking tool was iteratively evaluated through 10 expert interviews, a public user survey with 43 participants verified the necessity and usability of our tool. Overall, our findings identified several needs to systematically support the SciCom KI. The SciCom Wiki, as a FAIR digital library complementing our neurosymbolic computational fact-checking framework, was found suitable to address the raised requirements. Further, we identified that the SciCom KI is severely underdeveloped regarding FAIR knowledge and related systems facilitating its collaborative creation and curation. Our system can provide a central knowledge node, yet a collaborative effort is required to scale against the imminent (mis-)information flood.
>
---
#### [new 080] Large Language Models for Computer-Aided Design: A Survey
- **分类: cs.LG; cs.CL; cs.GR; cs.MM**

- **简介: 该论文是一篇系统综述，探讨大语言模型（LLMs）与计算机辅助设计（CAD）的交叉领域。任务为整合LLMs在CAD中的应用现状，解决该领域缺乏全面研究的问题。工作包括分析LLMs对CAD流程的优化潜力，分类六类关键应用场景，并展望未来研究方向。**

- **链接: [http://arxiv.org/pdf/2505.08137v1](http://arxiv.org/pdf/2505.08137v1)**

> **作者:** Licheng Zhang; Bach Le; Naveed Akhtar; Siew-Kei Lam; Tuan Ngo
>
> **摘要:** Large Language Models (LLMs) have seen rapid advancements in recent years, with models like ChatGPT and DeepSeek, showcasing their remarkable capabilities across diverse domains. While substantial research has been conducted on LLMs in various fields, a comprehensive review focusing on their integration with Computer-Aided Design (CAD) remains notably absent. CAD is the industry standard for 3D modeling and plays a vital role in the design and development of products across different industries. As the complexity of modern designs increases, the potential for LLMs to enhance and streamline CAD workflows presents an exciting frontier. This article presents the first systematic survey exploring the intersection of LLMs and CAD. We begin by outlining the industrial significance of CAD, highlighting the need for AI-driven innovation. Next, we provide a detailed overview of the foundation of LLMs. We also examine both closed-source LLMs as well as publicly available models. The core of this review focuses on the various applications of LLMs in CAD, providing a taxonomy of six key areas where these models are making considerable impact. Finally, we propose several promising future directions for further advancements, which offer vast opportunities for innovation and are poised to shape the future of CAD technology. Github: https://github.com/lichengzhanguom/LLMs-CAD-Survey-Taxonomy
>
---
#### [new 081] Arrow-Guided VLM: Enhancing Flowchart Understanding via Arrow Direction Encoding
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究流程图理解的视觉语言模型任务，解决现有模型误判箭头方向及拓扑结构的问题。提出七阶段流程，整合箭头检测、OCR文本提取和结构化提示生成，无需微调将流程图QA准确率从80%提升至89%，显著改善下一步推理（100%准确率）。方法依赖显式箭头编码，但多入边节点仍存挑战，未来将扩展数据集并验证BPMN/UML适用性。（100字）**

- **链接: [http://arxiv.org/pdf/2505.07864v1](http://arxiv.org/pdf/2505.07864v1)**

> **作者:** Takamitsu Omasa; Ryo Koshihara; Masumi Morishige
>
> **备注:** 11 pages, 1 figures,
>
> **摘要:** Flowcharts are indispensable tools in software design and business-process analysis, yet current vision-language models (VLMs) frequently misinterpret the directional arrows and graph topology that set these diagrams apart from natural images. We introduce a seven-stage pipeline grouped into three broader processes: (1) arrow-aware detection of nodes and arrow endpoints; (2) optical character recognition (OCR) to extract node text; and (3) construction of a structured prompt that guides the VLMs. Tested on a 90-question benchmark distilled from 30 annotated flowcharts, the method raises overall accuracy from 80 % to 89 % (+9 percentage points) without any task-specific fine-tuning. The gain is most pronounced for next-step queries (25/30 -> 30/30; 100 %, +17 pp); branch-result questions improve more modestly, and before-step questions remain difficult. A parallel evaluation with an LLM-as-a-Judge protocol shows the same trends, reinforcing the advantage of explicit arrow encoding. Limitations include dependence on detector and OCR precision, the small evaluation set, and residual errors at nodes with multiple incoming edges. Future work will enlarge the benchmark with synthetic and handwritten flowcharts and assess the approach on Business Process Model and Notation (BPMN) and Unified Modeling Language (UML).
>
---
#### [new 082] A Reproduction Study: The Kernel PCA Interpretation of Self-Attention Fails Under Scrutiny
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于验证性研究，针对自注意力机制是否实现核主成分分析(KPCA)的论点进行复现检验。通过分析10种Transformer架构，发现原主张的三个核心论据（值向量与Gram矩阵特征向量对齐、投影误差优化、Gram矩阵特征值统计）均存在矛盾，表明自注意力与KPCA缺乏实证关联。**

- **链接: [http://arxiv.org/pdf/2505.07908v1](http://arxiv.org/pdf/2505.07908v1)**

> **作者:** Karahan Sarıtaş; Çağatay Yıldız
>
> **摘要:** In this reproduction study, we revisit recent claims that self-attention implements kernel principal component analysis (KPCA) (Teo et al., 2024), positing that (i) value vectors $V$ capture the eigenvectors of the Gram matrix of the keys, and (ii) that self-attention projects queries onto the principal component axes of the key matrix $K$ in a feature space. Our analysis reveals three critical inconsistencies: (1) No alignment exists between learned self-attention value vectors and what is proposed in the KPCA perspective, with average similarity metrics (optimal cosine similarity $\leq 0.32$, linear CKA (Centered Kernel Alignment) $\leq 0.11$, kernel CKA $\leq 0.32$) indicating negligible correspondence; (2) Reported decreases in reconstruction loss $J_\text{proj}$, arguably justifying the claim that the self-attention minimizes the projection error of KPCA, are misinterpreted, as the quantities involved differ by orders of magnitude ($\sim\!10^3$); (3) Gram matrix eigenvalue statistics, introduced to justify that $V$ captures the eigenvector of the gram matrix, are irreproducible without undocumented implementation-specific adjustments. Across 10 transformer architectures, we conclude that the KPCA interpretation of self-attention lacks empirical support.
>
---
## 更新

#### [replaced 001] Studying the Effects of Collaboration in Interactive Theme Discovery Systems
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2408.09030v2](http://arxiv.org/pdf/2408.09030v2)**

> **作者:** Alvin Po-Chun Chen; Dananjay Srinivas; Alexandra Barry; Maksim Seniw; Maria Leonor Pacheco
>
> **摘要:** NLP-assisted solutions have gained considerable traction to support qualitative data analysis. However, there does not exist a unified evaluation framework that can account for the many different settings in which qualitative researchers may employ them. In this paper, we take a first step in this direction by proposing an evaluation framework to study the way in which different tools may result in different outcomes depending on the collaboration strategy employed. Specifically, we study the impact of synchronous vs. asynchronous collaboration using two different NLP-assisted qualitative research tools and present a comprehensive analysis of significant differences in the consistency, cohesiveness, and correctness of their outputs.
>
---
#### [replaced 002] Red Teaming the Mind of the Machine: A Systematic Evaluation of Prompt Injection and Jailbreak Vulnerabilities in LLMs
- **分类: cs.CR; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.04806v2](http://arxiv.org/pdf/2505.04806v2)**

> **作者:** Chetan Pathade
>
> **备注:** 7 Pages, 6 Figures
>
> **摘要:** Large Language Models (LLMs) are increasingly integrated into consumer and enterprise applications. Despite their capabilities, they remain susceptible to adversarial attacks such as prompt injection and jailbreaks that override alignment safeguards. This paper provides a systematic investigation of jailbreak strategies against various state-of-the-art LLMs. We categorize over 1,400 adversarial prompts, analyze their success against GPT-4, Claude 2, Mistral 7B, and Vicuna, and examine their generalizability and construction logic. We further propose layered mitigation strategies and recommend a hybrid red-teaming and sandboxing approach for robust LLM security.
>
---
#### [replaced 003] Adaptive Integrated Layered Attention (AILA)
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.IR; cs.NE**

- **链接: [http://arxiv.org/pdf/2503.22742v2](http://arxiv.org/pdf/2503.22742v2)**

> **作者:** William Claster; Suhas KM; Dhairya Gundechia
>
> **摘要:** We propose Adaptive Integrated Layered Attention (AILA), a neural network architecture that combines dense skip connections with different mechanisms for adaptive feature reuse across network layers. We evaluate AILA on three challenging tasks: price forecasting for various commodities and indices (S&P 500, Gold, US dollar Futures, Coffee, Wheat), image recognition using the CIFAR-10 dataset, and sentiment analysis on the IMDB movie review dataset. In all cases, AILA matches strong deep learning baselines (LSTMs, Transformers, and ResNets), achieving it at a fraction of the training and inference time. Notably, we implement and test two versions of the model - AILA-Architecture 1, which uses simple linear layers as the connection mechanism between layers, and AILA-Architecture 2, which implements an attention mechanism to selectively focus on outputs from previous layers. Both architectures are applied in a single-task learning setting, with each model trained separately for individual tasks. Results confirm that AILA's adaptive inter-layer connections yield robust gains by flexibly reusing pertinent features at multiple network depths. The AILA approach thus presents an extension to existing architectures, improving long-range sequence modeling, image recognition with optimised computational speed, and SOTA classification performance in practice.
>
---
#### [replaced 004] Round and Round We Go! What makes Rotary Positional Encodings useful?
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.06205v3](http://arxiv.org/pdf/2410.06205v3)**

> **作者:** Federico Barbero; Alex Vitvitskyi; Christos Perivolaropoulos; Razvan Pascanu; Petar Veličković
>
> **摘要:** Positional Encodings (PEs) are a critical component of Transformer-based Large Language Models (LLMs), providing the attention mechanism with important sequence-position information. One of the most popular types of encoding used today in LLMs are Rotary Positional Encodings (RoPE), that rotate the queries and keys based on their relative distance. A common belief is that RoPE is useful because it helps to decay token dependency as relative distance increases. In this work, we argue that this is unlikely to be the core reason. We study the internals of a trained Gemma 7B model to understand how RoPE is being used at a mechanical level. We find that Gemma learns to use RoPE to construct robust "positional" attention patterns by exploiting the highest frequencies. We also find that, in general, Gemma greatly prefers to use the lowest frequencies of RoPE, which we suspect are used to carry semantic information. We mathematically prove interesting behaviours of RoPE and conduct experiments to verify our findings, proposing a modification of RoPE that fixes some highlighted issues and improves performance. We believe that this work represents an interesting step in better understanding PEs in LLMs, which we believe holds crucial value for scaling LLMs to large sizes and context lengths.
>
---
#### [replaced 005] LLMSR@XLLM25: Less is More: Enhancing Structured Multi-Agent Reasoning via Quality-Guided Distillation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.16408v2](http://arxiv.org/pdf/2504.16408v2)**

> **作者:** Jiahao Yuan; Xingzhe Sun; Xing Yu; Jingwen Wang; Dehui Du; Zhiqing Cui; Zixiang Di
>
> **备注:** XLLM @ ACL 2025 Shared Task-III: LLM for Structural Reasoning (LLM-SR)
>
> **摘要:** The LLMSR@XLLM25 formulates a low-resource structural reasoning task that challenges LLMs to generate interpretable, step-by-step rationales with minimal labeled data. We present Less is More, the third-place winning approach in the LLMSR@XLLM25, which focuses on structured reasoning from only 24 labeled examples. Our approach leverages a multi-agent framework with reverse-prompt induction, retrieval-augmented reasoning synthesis via GPT-4o, and dual-stage reward-guided filtering to distill high-quality supervision across three subtasks: question parsing, CoT parsing, and step-level verification. All modules are fine-tuned from Meta-Llama-3-8B-Instruct under a unified LoRA+ setup. By combining structure validation with reward filtering across few-shot and zero-shot prompts, our pipeline consistently improves structure reasoning quality. These results underscore the value of controllable data distillation in enhancing structured inference under low-resource constraints. Our code is available at https://github.com/JhCircle/Less-is-More.
>
---
#### [replaced 006] SMI: An Information-Theoretic Metric for Predicting Model Knowledge Solely from Pre-Training Signals
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.04066v2](http://arxiv.org/pdf/2502.04066v2)**

> **作者:** Changhao Jiang; Ming Zhang; Junjie Ye; Xiaoran Fan; Yifei Cao; Jiajun Sun; Zhiheng Xi; Shihan Dou; Yi Dong; Yujiong Shen; Jingqi Tong; Zhen Wang; Tao Liang; Zhihui Fei; Mingyang Wan; Guojun Ma; Qi Zhang; Tao Gui; Xuanjing Huang
>
> **摘要:** The GPT-4 technical report highlights the possibility of predicting model performance on downstream tasks using only pre-training signals, though detailed methodologies are absent. Such predictive capabilities are essential for resource-efficient pre-training and the construction of task-aligned datasets. In this paper, we aim to predict performance in closed-book question answering (QA), a vital downstream task indicative of a model's internal knowledge. We address three primary challenges: (1) limited access to and understanding of pre-training corpora, (2) limitations of current evaluation methods for pre-trained models, and (3) limitations of frequency-based metrics in predicting model performance. In response to these challenges, we conduct large-scale retrieval and semantic analysis across the pre-training corpora of 21 publicly available and 3 custom-trained large language models. Subsequently, we develop a multi-template QA evaluation framework incorporating paraphrased question variants. Building on these foundations, we propose Size-dependent Mutual Information (SMI), an information-theoretic metric that linearly correlates pre-training data characteristics, model size, and QA accuracy, without requiring any additional training. The experimental results demonstrate that SMI outperforms co-occurrence-based baselines, achieving $R^2$ > 0.75 on models with over one billion parameters. Theoretical analysis further reveals the marginal benefits of scaling model size and optimizing data, indicating that the upper limit of specific QA task accuracy is approximately 80%. Our project is available at https://github.com/yuhui1038/SMI.
>
---
#### [replaced 007] No Preference Left Behind: Group Distributional Preference Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.20299v2](http://arxiv.org/pdf/2412.20299v2)**

> **作者:** Binwei Yao; Zefan Cai; Yun-Shiuan Chuang; Shanglin Yang; Ming Jiang; Diyi Yang; Junjie Hu
>
> **摘要:** Preferences within a group of people are not uniform but follow a distribution. While existing alignment methods like Direct Preference Optimization (DPO) attempt to steer models to reflect human preferences, they struggle to capture the distributional pluralistic preferences within a group. These methods often skew toward dominant preferences, overlooking the diversity of opinions, especially when conflicting preferences arise. To address this issue, we propose Group Distributional Preference Optimization (GDPO), a novel framework that aligns language models with the distribution of preferences within a group by incorporating the concept of beliefs that shape individual preferences. GDPO calibrates a language model using statistical estimation of the group's belief distribution and aligns the model with belief-conditioned preferences, offering a more inclusive alignment framework than traditional methods. In experiments using both synthetic controllable opinion generation and real-world movie review datasets, we show that DPO fails to align with the targeted belief distributions, while GDPO consistently reduces this alignment gap during training. Moreover, our evaluation metrics demonstrate that GDPO outperforms existing approaches in aligning with group distributional preferences, marking a significant advance in pluralistic alignment.
>
---
#### [replaced 008] Crossing Boundaries: Leveraging Semantic Divergences to Explore Cultural Novelty in Cooking Recipes
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.24027v2](http://arxiv.org/pdf/2503.24027v2)**

> **作者:** Florian Carichon; Romain Rampa; Golnoosh Farnadi
>
> **备注:** Updated to match the version accepted at ACM FAccT 2025. Includes revised text and results
>
> **摘要:** Novelty modeling and detection is a core topic in Natural Language Processing (NLP), central to numerous tasks such as recommender systems and automatic summarization. It involves identifying pieces of text that deviate in some way from previously known information. However, novelty is also a crucial determinant of the unique perception of relevance and quality of an experience, as it rests upon each individual's understanding of the world. Social factors, particularly cultural background, profoundly influence perceptions of novelty and innovation. Cultural novelty arises from differences in salience and novelty as shaped by the distance between distinct communities. While cultural diversity has garnered increasing attention in artificial intelligence (AI), the lack of robust metrics for quantifying cultural novelty hinders a deeper understanding of these divergences. This gap limits quantifying and understanding cultural differences within computational frameworks. To address this, we propose an interdisciplinary framework that integrates knowledge from sociology and management. Central to our approach is GlobalFusion, a novel dataset comprising 500 dishes and approximately 100,000 cooking recipes capturing cultural adaptation from over 150 countries. By introducing a set of Jensen-Shannon Divergence metrics for novelty, we leverage this dataset to analyze textual divergences when recipes from one community are modified by another with a different cultural background. The results reveal significant correlations between our cultural novelty metrics and established cultural measures based on linguistic, religious, and geographical distances. Our findings highlight the potential of our framework to advance the understanding and measurement of cultural diversity in AI.
>
---
#### [replaced 009] CURIE: Evaluating LLMs On Multitask Scientific Long Context Understanding and Reasoning
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.13517v2](http://arxiv.org/pdf/2503.13517v2)**

> **作者:** Hao Cui; Zahra Shamsi; Gowoon Cheon; Xuejian Ma; Shutong Li; Maria Tikhanovskaya; Peter Norgaard; Nayantara Mudur; Martyna Plomecka; Paul Raccuglia; Yasaman Bahri; Victor V. Albert; Pranesh Srinivasan; Haining Pan; Philippe Faist; Brian Rohr; Ekin Dogus Cubuk; Muratahan Aykol; Amil Merchant; Michael J. Statt; Dan Morris; Drew Purves; Elise Kleeman; Ruth Alcantara; Matthew Abraham; Muqthar Mohammad; Ean Phing VanLee; Chenfei Jiang; Elizabeth Dorfman; Eun-Ah Kim; Michael P Brenner; Viren Jain; Sameera Ponda; Subhashini Venugopalan
>
> **备注:** Accepted at ICLR 2025 main conference
>
> **摘要:** Scientific problem-solving involves synthesizing information while applying expert knowledge. We introduce CURIE, a scientific long-Context Understanding,Reasoning and Information Extraction benchmark to measure the potential of Large Language Models (LLMs) in scientific problem-solving and assisting scientists in realistic workflows. This benchmark introduces ten challenging tasks with a total of 580 problems and solution pairs curated by experts in six disciplines - materials science, condensed matter physics, quantum computing, geospatial analysis, biodiversity, and proteins - covering both experimental and theoretical work-flows in science. We evaluate a range of closed and open LLMs on tasks in CURIE which requires domain expertise, comprehension of long in-context information,and multi-step reasoning. While Gemini Flash 2.0 and Claude-3 show consistent high comprehension across domains, the popular GPT-4o and command-R+ fail dramatically on protein sequencing tasks. With the best performance at 32% there is much room for improvement for all models. We hope that insights gained from CURIE can guide the future development of LLMs in sciences. Evaluation code and data are in https://github.com/google/curie
>
---
#### [replaced 010] AI Hiring with LLMs: A Context-Aware and Explainable Multi-Agent Framework for Resume Screening
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.02870v2](http://arxiv.org/pdf/2504.02870v2)**

> **作者:** Frank P. -W. Lo; Jianing Qiu; Zeyu Wang; Haibao Yu; Yeming Chen; Gao Zhang; Benny Lo
>
> **备注:** Accepted by CVPR 2025 Workshop
>
> **摘要:** Resume screening is a critical yet time-intensive process in talent acquisition, requiring recruiters to analyze vast volume of job applications while remaining objective, accurate, and fair. With the advancements in Large Language Models (LLMs), their reasoning capabilities and extensive knowledge bases demonstrate new opportunities to streamline and automate recruitment workflows. In this work, we propose a multi-agent framework for resume screening using LLMs to systematically process and evaluate resumes. The framework consists of four core agents, including a resume extractor, an evaluator, a summarizer, and a score formatter. To enhance the contextual relevance of candidate assessments, we integrate Retrieval-Augmented Generation (RAG) within the resume evaluator, allowing incorporation of external knowledge sources, such as industry-specific expertise, professional certifications, university rankings, and company-specific hiring criteria. This dynamic adaptation enables personalized recruitment, bridging the gap between AI automation and talent acquisition. We assess the effectiveness of our approach by comparing AI-generated scores with ratings provided by HR professionals on a dataset of anonymized online resumes. The findings highlight the potential of multi-agent RAG-LLM systems in automating resume screening, enabling more efficient and scalable hiring workflows.
>
---
#### [replaced 011] Can (A)I Change Your Mind?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.01844v3](http://arxiv.org/pdf/2503.01844v3)**

> **作者:** Miriam Havin; Timna Wharton Kleinman; Moran Koren; Yaniv Dover; Ariel Goldstein
>
> **备注:** Accetped to CogSci 2025
>
> **摘要:** The increasing integration of large language models (LLMs) based conversational agents into everyday life raises critical cognitive and social questions about their potential to influence human opinions. Although previous studies have shown that LLM-based agents can generate persuasive content, these typically involve controlled English-language settings. Addressing this, our preregistered study explored LLMs' persuasive capabilities in more ecological, unconstrained scenarios, examining both static (written paragraphs) and dynamic (conversations via Telegram) interaction types. Conducted entirely in Hebrew with 200 participants, the study assessed the persuasive effects of both LLM and human interlocutors on controversial civil policy topics. Results indicated that participants adopted LLM and human perspectives similarly, with significant opinion changes evident across all conditions, regardless of interlocutor type or interaction mode. Confidence levels increased significantly in most scenarios. These findings demonstrate LLM-based agents' robust persuasive capabilities across diverse sources and settings, highlighting their potential impact on shaping public opinions.
>
---
#### [replaced 012] Graph RAG for Legal Norms: A Hierarchical and Temporal Approach
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2505.00039v2](http://arxiv.org/pdf/2505.00039v2)**

> **作者:** Hudson de Martim
>
> **摘要:** This article proposes an adaptation of Graph Retrieval Augmented Generation (Graph RAG) specifically designed for the analysis and comprehension of legal norms, which are characterized by their predefined hierarchical structure, extensive network of internal and external references and multiple temporal versions. By combining structured knowledge graphs with contextually enriched text segments, Graph RAG offers a promising solution to address the inherent complexity and vast volume of legal data. The integration of hierarchical structure and temporal evolution into knowledge graphs - along with the concept of comprehensive Text Units - facilitates the construction of richer, interconnected representations of legal knowledge. Through a detailed analysis of Graph RAG and its application to legal norm datasets, this article aims to advance the field of Artificial Intelligence applied to Law, creating opportunities for more effective systems in legal research, legislative analysis, and decision support.
>
---
#### [replaced 013] Query-driven Document-level Scientific Evidence Extraction from Biomedical Studies
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.06186v2](http://arxiv.org/pdf/2505.06186v2)**

> **作者:** Massimiliano Pronesti; Joao Bettencourt-Silva; Paul Flanagan; Alessandra Pascale; Oisin Redmond; Anya Belz; Yufang Hou
>
> **摘要:** Extracting scientific evidence from biomedical studies for clinical research questions (e.g., Does stem cell transplantation improve quality of life in patients with medically refractory Crohn's disease compared to placebo?) is a crucial step in synthesising biomedical evidence. In this paper, we focus on the task of document-level scientific evidence extraction for clinical questions with conflicting evidence. To support this task, we create a dataset called CochraneForest, leveraging forest plots from Cochrane systematic reviews. It comprises 202 annotated forest plots, associated clinical research questions, full texts of studies, and study-specific conclusions. Building on CochraneForest, we propose URCA (Uniform Retrieval Clustered Augmentation), a retrieval-augmented generation framework designed to tackle the unique challenges of evidence extraction. Our experiments show that URCA outperforms the best existing methods by up to 10.3% in F1 score on this task. However, the results also underscore the complexity of CochraneForest, establishing it as a challenging testbed for advancing automated evidence synthesis systems.
>
---
#### [replaced 014] BLAB: Brutally Long Audio Bench
- **分类: cs.AI; cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.03054v2](http://arxiv.org/pdf/2505.03054v2)**

> **作者:** Orevaoghene Ahia; Martijn Bartelds; Kabir Ahuja; Hila Gonen; Valentin Hofmann; Siddhant Arora; Shuyue Stella Li; Vishal Puttagunta; Mofetoluwa Adeyemi; Charishma Buchireddy; Ben Walls; Noah Bennett; Shinji Watanabe; Noah A. Smith; Yulia Tsvetkov; Sachin Kumar
>
> **摘要:** Developing large audio language models (LMs) capable of understanding diverse spoken interactions is essential for accommodating the multimodal nature of human communication and can increase the accessibility of language technologies across different user populations. Recent work on audio LMs has primarily evaluated their performance on short audio segments, typically under 30 seconds, with limited exploration of long-form conversational speech segments that more closely reflect natural user interactions with these models. We introduce Brutally Long Audio Bench (BLAB), a challenging long-form audio benchmark that evaluates audio LMs on localization, duration estimation, emotion, and counting tasks using audio segments averaging 51 minutes in length. BLAB consists of 833+ hours of diverse, full-length audio clips, each paired with human-annotated, text-based natural language questions and answers. Our audio data were collected from permissively licensed sources and underwent a human-assisted filtering process to ensure task compliance. We evaluate six open-source and proprietary audio LMs on BLAB and find that all of them, including advanced models such as Gemini 2.0 Pro and GPT-4o, struggle with the tasks in BLAB. Our comprehensive analysis reveals key insights into the trade-offs between task difficulty and audio duration. In general, we find that audio LMs struggle with long-form speech, with performance declining as duration increases. They perform poorly on localization, temporal reasoning, counting, and struggle to understand non-phonemic information, relying more on prompts than audio content. BLAB serves as a challenging evaluation framework to develop audio LMs with robust long-form audio understanding capabilities.
>
---
#### [replaced 015] Beyond Single-Turn: A Survey on Multi-Turn Interactions with Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.04717v3](http://arxiv.org/pdf/2504.04717v3)**

> **作者:** Yubo Li; Xiaobin Shen; Xinyu Yao; Xueying Ding; Yidi Miao; Ramayya Krishnan; Rema Padman
>
> **摘要:** Recent advancements in large language models (LLMs) have revolutionized their ability to handle single-turn tasks, yet real-world applications demand sophisticated multi-turn interactions. This survey provides a comprehensive review of recent advancements in evaluating and enhancing multi-turn interactions in LLMs. Focusing on task-specific scenarios, from instruction following in diverse domains such as math and coding to complex conversational engagements in roleplay, healthcare, education, and even adversarial jailbreak settings, we systematically examine the challenges of maintaining context, coherence, fairness, and responsiveness over prolonged dialogues. The paper organizes current benchmarks and datasets into coherent categories that reflect the evolving landscape of multi-turn dialogue evaluation. In addition, we review a range of enhancement methodologies under multi-turn settings, including model-centric strategies (contextual learning, supervised fine-tuning, reinforcement learning, and new architectures), external integration approaches (memory-augmented, retrieval-based methods, and knowledge graph), and agent-based techniques for collaborative interactions. Finally, we discuss open challenges and propose future directions for research to further advance the robustness and effectiveness of multi-turn interactions in LLMs. Related resources and papers are available at https://github.com/yubol-cmu/Awesome-Multi-Turn-LLMs.
>
---
#### [replaced 016] DeepDistill: Enhancing LLM Reasoning Capabilities via Large-Scale Difficulty-Graded Data Training
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.17565v3](http://arxiv.org/pdf/2504.17565v3)**

> **作者:** Xiaoyu Tian; Sitong Zhao; Haotian Wang; Shuaiting Chen; Yiping Peng; Yunjie Ji; Han Zhao; Xiangang Li
>
> **摘要:** Although large language models (LLMs) have recently achieved remarkable performance on various complex reasoning benchmarks, the academic community still lacks an in-depth understanding of base model training processes and data quality. To address this, we construct a large-scale, difficulty-graded reasoning dataset containing approximately 3.34 million unique queries of varying difficulty levels and about 40 million distilled responses generated by multiple models over several passes. Leveraging pass rate and Coefficient of Variation (CV), we precisely select the most valuable training data to enhance reasoning capability. Notably, we observe a training pattern shift, indicating that reasoning-focused training based on base models requires higher learning rates for effective training. Using this carefully selected data, we significantly improve the reasoning capabilities of the base model, achieving a pass rate of 79.2\% on the AIME2024 mathematical reasoning benchmark. This result surpasses most current distilled models and closely approaches state-of-the-art performance. We provide detailed descriptions of our data processing, difficulty assessment, and training methodology, and have publicly released all datasets and methods to promote rapid progress in open-source long-reasoning LLMs. The dataset is available at: \href{https://huggingface.co/datasets/a-m-team/AM-DeepSeek-Distilled-40M}{https://huggingface.co/datasets/a-m-team/AM-DeepSeek-Distilled-40M}
>
---
#### [replaced 017] Efficient Adaptation For Remote Sensing Visual Grounding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.23083v2](http://arxiv.org/pdf/2503.23083v2)**

> **作者:** Hasan Moughnieh; Mohamad Chalhoub; Hasan Nasrallah; Cristiano Nattero; Paolo Campanella; Giovanni Nico; Ali J. Ghandour
>
> **摘要:** Adapting pre-trained models has become an effective strategy in artificial intelligence, offering a scalable and efficient alternative to training models from scratch. In the context of remote sensing (RS), where visual grounding(VG) remains underexplored, this approach enables the deployment of powerful vision-language models to achieve robust cross-modal understanding while significantly reducing computational overhead. To address this, we applied Parameter Efficient Fine Tuning (PEFT) techniques to adapt these models for RS-specific VG tasks. Specifically, we evaluated LoRA placement across different modules in Grounding DINO and used BitFit and adapters to fine-tune the OFA foundation model pre-trained on general-purpose VG datasets. This approach achieved performance comparable to or surpassing current State Of The Art (SOTA) models while significantly reducing computational costs. This study highlights the potential of PEFT techniques to advance efficient and precise multi-modal analysis in RS, offering a practical and cost-effective alternative to full model training.
>
---
#### [replaced 018] 2.5 Years in Class: A Multimodal Textbook for Vision-Language Pretraining
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.00958v4](http://arxiv.org/pdf/2501.00958v4)**

> **作者:** Wenqi Zhang; Hang Zhang; Xin Li; Jiashuo Sun; Yongliang Shen; Weiming Lu; Deli Zhao; Yueting Zhuang; Lidong Bing
>
> **备注:** Under review
>
> **摘要:** Compared to image-text pair data, interleaved corpora enable Vision-Language Models (VLMs) to understand the world more naturally like humans. However, such existing datasets are crawled from webpage, facing challenges like low knowledge density, loose image-text relations, and poor logical coherence between images. On the other hand, the internet hosts vast instructional videos (e.g., online geometry courses) that are widely used by humans to learn foundational subjects, yet these valuable resources remain underexplored in VLM training. In this paper, we introduce a high-quality \textbf{multimodal textbook} corpus with richer foundational knowledge for VLM pretraining. It collects over 2.5 years of instructional videos, totaling 22,000 class hours. We first use an LLM-proposed taxonomy to systematically gather instructional videos. Then we progressively extract and refine visual (keyframes), audio (ASR), and textual knowledge (OCR) from the videos, and organize as an image-text interleaved corpus based on temporal order. Compared to its counterparts, our video-centric textbook offers more coherent context, richer knowledge, and better image-text alignment. Experiments demonstrate its superb pretraining performance, particularly in knowledge- and reasoning-intensive tasks like ScienceQA and MathVista. Moreover, VLMs pre-trained on our textbook exhibit outstanding interleaved context awareness, leveraging visual and textual cues in their few-shot context for task solving. Our code are available at https://github.com/DAMO-NLP-SG/multimodal_textbook.
>
---
#### [replaced 019] Ask, Fail, Repeat: Meeseeks, an Iterative Feedback Benchmark for LLMs' Multi-turn Instruction-following Ability
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.21625v2](http://arxiv.org/pdf/2504.21625v2)**

> **作者:** Jiaming Wang; Yunke Zhao; Peng Ding; Jun Kuang; Zongyu Wang; Xuezhi Cao; Xunliang Cai
>
> **摘要:** The ability to follow instructions accurately is fundamental for Large Language Models (LLMs) to serve as reliable agents in real-world applications. For complex instructions, LLMs often struggle to fulfill all requirements in a single attempt. In practice, users typically provide iterative feedback until the LLM generates a response that meets all requirements. However, existing instruction-following benchmarks are either single-turn or introduce new requirements in each turn without allowing self-correction. To address this gap, we propose \textbf{Meeseeks} (named after Mr. Meeseeks from \textit{Rick and Morty}\footnote{Rick and Morty is an American adult animated science fiction sitcom created by Justin Roiland and Dan Harmon for Cartoon Network's nighttime programming block Adult Swim.}.) Meeseeks simulates realistic human-LLM interactions through an iterative feedback framework, which enables models to self-correct based on specific requirement failures in each turn, better reflecting real-world user-end usage patterns. Meanwhile, the benchmark implements a comprehensive evaluation system with 38 capability tags organized across three dimensions: Intent Recognition, Granular Content Validation, and Output Structure Validation. Through rigorous evaluation across LLMs, Meeseeks provides valuable insights into LLMs' instruction-following capabilities in multi-turn scenarios.
>
---
#### [replaced 020] From Calculation to Adjudication: Examining LLM judges on Mathematical Reasoning Tasks
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2409.04168v2](http://arxiv.org/pdf/2409.04168v2)**

> **作者:** Andreas Stephan; Dawei Zhu; Matthias Aßenmacher; Xiaoyu Shen; Benjamin Roth
>
> **摘要:** To reduce the need for human annotations, large language models (LLMs) have been proposed as judges of the quality of other candidate models. The performance of LLM judges is typically evaluated by measuring the correlation with human judgments on generative tasks such as summarization or machine translation. In contrast, we study LLM judges on mathematical reasoning tasks. These tasks require multi-step reasoning, and the correctness of their solutions is verifiable, enabling a more objective evaluation. We perform a detailed performance analysis and find that easy samples are easy to judge, and difficult samples are difficult to judge. Our analysis uncovers a strong correlation between judgment performance and the candidate model task performance, indicating that judges tend to favor higher-quality models even if their answer is incorrect. As a consequence, we test whether we can predict the behavior of LLM judges using simple features such as part-of-speech tags and find that we can correctly predict 70%-75% of judgments. We conclude this study by analyzing practical use cases, showing that LLM judges consistently detect the on-average better model but largely fail if we use them to improve task performance.
>
---
#### [replaced 021] Are Transformers Able to Reason by Connecting Separated Knowledge in Training Data?
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.15857v5](http://arxiv.org/pdf/2501.15857v5)**

> **作者:** Yutong Yin; Zhaoran Wang
>
> **备注:** Accepted by ICLR 2025
>
> **摘要:** Humans exhibit remarkable compositional reasoning by integrating knowledge from various sources. For example, if someone learns ( B = f(A) ) from one source and ( C = g(B) ) from another, they can deduce ( C=g(B)=g(f(A)) ) even without encountering ( ABC ) together, showcasing the generalization ability of human intelligence. In this paper, we introduce a synthetic learning task, "FTCT" (Fragmented at Training, Chained at Testing), to validate the potential of Transformers in replicating this skill and interpret its inner mechanism. In the training phase, data consist of separated knowledge fragments from an overall causal graph. During testing, Transformers must infer complete causal graph traces by integrating these fragments. Our findings demonstrate that few-shot Chain-of-Thought prompting enables Transformers to perform compositional reasoning on FTCT by revealing correct combinations of fragments, even if such combinations were absent in the training data. Furthermore, the emergence of compositional reasoning ability is strongly correlated with the model complexity and training-testing data similarity. We propose, both theoretically and empirically, that Transformers learn an underlying generalizable program from training, enabling effective compositional reasoning during testing.
>
---
#### [replaced 022] FutureVision: A methodology for the investigation of future cognition
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.01597v2](http://arxiv.org/pdf/2502.01597v2)**

> **作者:** Tiago Timponi Torrent; Mark Turner; Nicolás Hinrichs; Frederico Belcavello; Igor Lourenço; Arthur Lorenzi Almeida; Marcelo Viridiano; Ely Edison Matos
>
> **备注:** Paper accepted at CogSci 2025
>
> **摘要:** This paper presents a methodology combining multimodal semantic analysis with an eye-tracking experimental protocol to investigate the cognitive effort involved in understanding the communication of future scenarios. To demonstrate the methodology, we conduct a pilot study examining how visual fixation patterns vary during the evaluation of valence and counterfactuality in fictional ad pieces describing futuristic scenarios, using a portable eye tracker. Participants eye movements are recorded while evaluating the stimuli and describing them to a conversation partner. Gaze patterns are analyzed alongside semantic representations of the stimuli and participants descriptions, constructed from a frame semantic annotation of both linguistic and visual modalities. Preliminary results show that far-future and pessimistic scenarios are associated with longer fixations and more erratic saccades, supporting the hypothesis that fractures in the base spaces underlying the interpretation of future scenarios increase cognitive load for comprehenders.
>
---
#### [replaced 023] Vision-Language Models Do Not Understand Negation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.09425v2](http://arxiv.org/pdf/2501.09425v2)**

> **作者:** Kumail Alhamoud; Shaden Alshammari; Yonglong Tian; Guohao Li; Philip Torr; Yoon Kim; Marzyeh Ghassemi
>
> **备注:** CVPR 2025; project page: https://negbench.github.io
>
> **摘要:** Many practical vision-language applications require models that understand negation, e.g., when using natural language to retrieve images which contain certain objects but not others. Despite advancements in vision-language models (VLMs) through large-scale training, their ability to comprehend negation remains underexplored. This study addresses the question: how well do current VLMs understand negation? We introduce NegBench, a new benchmark designed to evaluate negation understanding across 18 task variations and $79$k examples spanning image, video, and medical datasets. The benchmark consists of two core tasks designed to evaluate negation understanding in diverse multimodal settings: Retrieval with Negation and Multiple Choice Questions with Negated Captions. Our evaluation reveals that modern VLMs struggle significantly with negation, often performing at chance level. To address these shortcomings, we explore a data-centric approach wherein we finetune CLIP models on large-scale synthetic datasets containing millions of negated captions. We show that this approach can result in a 10% increase in recall on negated queries and a 28% boost in accuracy on multiple-choice questions with negated captions.
>
---
#### [replaced 024] CursorCore: Assist Programming through Aligning Anything
- **分类: cs.CL; cs.AI; cs.SE**

- **链接: [http://arxiv.org/pdf/2410.07002v3](http://arxiv.org/pdf/2410.07002v3)**

> **作者:** Hao Jiang; Qi Liu; Rui Li; Shengyu Ye; Shijin Wang
>
> **摘要:** Large language models have been successfully applied to programming assistance tasks, such as code completion, code insertion, and instructional code editing. However, these applications remain insufficiently automated and struggle to effectively integrate various types of information during the programming process, including coding history, current code, and user instructions. In this work, we propose a new conversational framework that comprehensively integrates these information sources, collect data to train our models and evaluate their performance. Firstly, to thoroughly evaluate how well models align with different types of information and the quality of their outputs, we introduce a new benchmark, APEval (Assist Programming Eval), to comprehensively assess the performance of models in programming assistance tasks. Then, for data collection, we develop a data generation pipeline, Programming-Instruct, which synthesizes training data from diverse sources, such as GitHub and online judge platforms. This pipeline can automatically generate various types of messages throughout the programming process. Finally, using this pipeline, we generate 219K samples, fine-tune multiple models, and develop the CursorCore series. We show that CursorCore outperforms other models of comparable size. This framework unifies applications such as inline chat and automated editing, contributes to the advancement of coding assistants. Code, models and data are freely available at https://github.com/TechxGenus/CursorCore.
>
---
#### [replaced 025] OnPrem.LLM: A Privacy-Conscious Document Intelligence Toolkit
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.07672v2](http://arxiv.org/pdf/2505.07672v2)**

> **作者:** Arun S. Maiya
>
> **备注:** 6 pages
>
> **摘要:** We present OnPrem$.$LLM, a Python-based toolkit for applying large language models (LLMs) to sensitive, non-public data in offline or restricted environments. The system is designed for privacy-preserving use cases and provides prebuilt pipelines for document processing and storage, retrieval-augmented generation (RAG), information extraction, summarization, classification, and prompt/output processing with minimal configuration. OnPrem$.$LLM supports multiple LLM backends -- including llama$.$cpp, Ollama, vLLM, and Hugging Face Transformers -- with quantized model support, GPU acceleration, and seamless backend switching. Although designed for fully local execution, OnPrem$.$LLM also supports integration with a wide range of cloud LLM providers when permitted, enabling hybrid deployments that balance performance with data control. A no-code web interface extends accessibility to non-technical users.
>
---
#### [replaced 026] Cite Before You Speak: Enhancing Context-Response Grounding in E-commerce Conversational LLM-Agents
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.04830v3](http://arxiv.org/pdf/2503.04830v3)**

> **作者:** Jingying Zeng; Hui Liu; Zhenwei Dai; Xianfeng Tang; Chen Luo; Samarth Varshney; Zhen Li; Qi He
>
> **摘要:** With the advancement of conversational large language models (LLMs), several LLM-based Conversational Shopping Agents (CSA) have been developed to help customers smooth their online shopping. The primary objective in building an engaging and trustworthy CSA is to ensure the agent's responses about product factoids are accurate and factually grounded. However, two challenges remain. First, LLMs produce hallucinated or unsupported claims. Such inaccuracies risk spreading misinformation and diminishing customer trust. Second, without providing knowledge source attribution in CSA response, customers struggle to verify LLM-generated information. To address both challenges, we present an easily productionized solution that enables a ''citation experience'' to our customers. We build auto-evaluation metrics to holistically evaluate LLM's grounding and attribution capabilities, suggesting that citation generation paradigm substantially improves grounding performance by 13.83%. To deploy this capability at scale, we introduce Multi-UX-Inference system, which appends source citations to LLM outputs while preserving existing user experience features and supporting scalable inference. Large-scale online A/B tests show that grounded CSA responses improves customer engagement by 3% - 10%, depending on UX variations.
>
---
#### [replaced 027] Gradual Binary Search and Dimension Expansion : A general method for activation quantization in LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.13989v2](http://arxiv.org/pdf/2504.13989v2)**

> **作者:** Lucas Maisonnave; Cyril Moineau; Olivier Bichler; Fabrice Rastello
>
> **摘要:** Large language models (LLMs) have become pivotal in artificial intelligence, demonstrating strong capabilities in reasoning, understanding, and generating data. However, their deployment on edge devices is hindered by their substantial size, often reaching several billion parameters. Quantization is a widely used method to reduce memory usage and inference time, however LLMs present unique challenges due to the prevalence of outliers in their activations. In this work, we leverage the theoretical advantages of Hadamard matrices over random rotation matrices to push the boundaries of quantization in LLMs. We demonstrate that Hadamard matrices are more effective in reducing outliers, which are a significant obstacle in achieving low-bit quantization. Our method based on a gradual binary search enables 3-bit quantization for weights, activations, and key-value (KV) caches, resulting in a 40% increase in accuracy on common benchmarks compared to SoTA methods. We extend the use of rotation matrices to support non-power-of-2 embedding dimensions, similar to the Qwen architecture, by employing the Paley algorithm. We theoretically demonstrates the superiority of Hadamard matrices in reducing outliers.We achieved 3-bit quantization for weights, activations, and KV cache, significantly enhancing model performance. Our experimental results on multiple models family like Mistral, LLaMA, and Qwen demonstrate the effectiveness of our approach, outperforming existing methods and enabling practical 3-bit quantization.
>
---
#### [replaced 028] Bridging LLMs and KGs without Fine-Tuning: Intermediate Probing Meets Subgraph-Aware Entity Descriptions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.06787v3](http://arxiv.org/pdf/2408.06787v3)**

> **作者:** Bo Xue; Yi Xu; Yunchong Song; Yiming Pang; Yuyang Ren; Jiaxin Ding; Luoyi Fu; Xinbing Wang
>
> **摘要:** Traditional knowledge graph completion (KGC) methods rely solely on structural information, struggling with the inherent sparsity of knowledge graphs (KGs). Large Language Models (LLMs) learn extensive knowledge from large corpora with powerful context modeling, making them promising for mitigating the limitations of previous methods. Directly fine-tuning LLMs offers great capability but comes at the cost of huge time and memory consumption, while utilizing frozen LLMs yields suboptimal results.In this work, we aim to leverage LLMs for KGC effectively and efficiently. We capture the context-aware hidden states of knowledge triples by employing prompts to stimulate the intermediate layers of LLMs. We then train a data-efficient classifier on these hidden states to harness the inherent capabilities of frozen LLMs in KGC. Additionally, to reduce ambiguity and enrich knowledge representation, we generate detailed entity descriptions through subgraph sampling on KGs. Extensive experiments on standard benchmarks demonstrate the efficiency and effectiveness of our approach. We outperform traditional KGC methods across most datasets and, notably, achieve classification performance comparable to fine-tuned LLMs while enhancing GPU memory efficiency by $188\times$ and accelerating training and inference by $13.48\times$.
>
---
#### [replaced 029] Self-reflecting Large Language Models: A Hegelian Dialectical Approach
- **分类: cs.CL; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.14917v5](http://arxiv.org/pdf/2501.14917v5)**

> **作者:** Sara Abdali; Can Goksen; Saeed Amizadeh; Julie E. Maybee; Kazuhito Koishida
>
> **摘要:** Investigating NLP through a philosophical lens has recently caught researcher's eyes as it connects computational methods with classical schools of philosophy. This paper introduces a philosophical approach inspired by the \textit{Hegelian Dialectic} for LLMs' \textit{self-reflection}, utilizing a self-dialectical approach to emulate internal critiques and then synthesize new ideas by resolving the opposing points of view. Moreover, this paper investigates the effect of LLMs' temperature for generation by establishing a dynamic annealing approach, which promotes the creativity in the early stages and gradually refines it by focusing on the nuances, as well as a fixed-temperature strategy for generation. We assess the effectiveness of our proposed method in generating novel ideas and in improving the reasoning abilities of LLMs during problem-solving. Moreover, we implement a Multi-Agent Majority Voting (MAMV) strategy to assess the validity and novelty of the generated ideas, which proves useful in the absence of domain experts. Our experiments demonstrate promising results in generating ideas and enhancing problem-solving performance.
>
---
#### [replaced 030] IndicSQuAD: A Comprehensive Multilingual Question Answering Dataset for Indic Languages
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.03688v2](http://arxiv.org/pdf/2505.03688v2)**

> **作者:** Sharvi Endait; Ruturaj Ghatage; Aditya Kulkarni; Rajlaxmi Patil; Raviraj Joshi
>
> **摘要:** The rapid progress in question-answering (QA) systems has predominantly benefited high-resource languages, leaving Indic languages largely underrepresented despite their vast native speaker base. In this paper, we present IndicSQuAD, a comprehensive multi-lingual extractive QA dataset covering nine major Indic languages, systematically derived from the SQuAD dataset. Building on previous work with MahaSQuAD for Marathi, our approach adapts and extends translation techniques to maintain high linguistic fidelity and accurate answer-span alignment across diverse languages. IndicSQuAD comprises extensive training, validation, and test sets for each language, providing a robust foundation for model development. We evaluate baseline performances using language-specific monolingual BERT models and the multilingual MuRIL-BERT. The results indicate some challenges inherent in low-resource settings. Moreover, our experiments suggest potential directions for future work, including expanding to additional languages, developing domain-specific datasets, and incorporating multimodal data. The dataset and models are publicly shared at https://github.com/l3cube-pune/indic-nlp
>
---
#### [replaced 031] Scaling Laws for Floating Point Quantization Training
- **分类: cs.LG; cs.AR; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.02423v2](http://arxiv.org/pdf/2501.02423v2)**

> **作者:** Xingwu Sun; Shuaipeng Li; Ruobing Xie; Weidong Han; Kan Wu; Zhen Yang; Yixing Li; An Wang; Shuai Li; Jinbao Xue; Yu Cheng; Yangyu Tao; Zhanhui Kang; Chengzhong Xu; Di Wang; Jie Jiang
>
> **摘要:** Low-precision training is considered an effective strategy for reducing both training and downstream inference costs. Previous scaling laws for precision mainly focus on integer quantization, which pay less attention to the constituents in floating-point (FP) quantization, and thus cannot well fit the LLM losses in this scenario. In contrast, while FP quantization training is more commonly implemented in production, it's research has been relatively superficial. In this paper, we thoroughly explore the effects of FP quantization targets, exponent bits, mantissa bits, and the calculation granularity of the scaling factor in FP quantization training performance of LLM models. In addition to an accurate FP quantization unified scaling law, we also provide valuable suggestions for the community: (1) Exponent bits contribute slightly more to the model performance than mantissa bits. We provide the optimal exponent-mantissa bit ratio for different bit numbers, which is available for future reference by hardware manufacturers; (2) We discover the formation of the critical data size in low-precision LLM training. Too much training data exceeding the critical data size will inversely bring in degradation of LLM performance; (3) The optimal FP quantization precision is directly proportional to the computational power, but within a wide computational power range. We estimate that the best cost-performance precision should lie between 4-8 bits.
>
---
#### [replaced 032] Integrating Single-Cell Foundation Models with Graph Neural Networks for Drug Response Prediction
- **分类: cs.LG; cs.CL; q-bio.QM**

- **链接: [http://arxiv.org/pdf/2504.14361v2](http://arxiv.org/pdf/2504.14361v2)**

> **作者:** Till Rossner; Ziteng Li; Jonas Balke; Nikoo Salehfard; Tom Seifert; Ming Tang
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** AI-driven drug response prediction holds great promise for advancing personalized cancer treatment. However, the inherent heterogenity of cancer and high cost of data generation make accurate prediction challenging. In this study, we investigate whether incorporating the pretrained foundation model scGPT can enhance the performance of existing drug response prediction frameworks. Our approach builds on the DeepCDR framework, which encodes drug representations from graph structures and cell representations from multi-omics profiles. We adapt this framework by leveraging scGPT to generate enriched cell representations using its pretrained knowledge to compensate for limited amount of data. We evaluate our modified framework using IC$_{50}$ values on Pearson correlation coefficient (PCC) and a leave-one-drug out validation strategy, comparing it against the original DeepCDR framework and a prior scFoundation-based approach. scGPT not only outperforms previous approaches but also exhibits greater training stability, highlighting the value of leveraging scGPT-derived knowledge in this domain.
>
---
#### [replaced 033] Multi-Party Supervised Fine-tuning of Language Models for Multi-Party Dialogue Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.05342v4](http://arxiv.org/pdf/2412.05342v4)**

> **作者:** Xiaoyu Wang; Ningyuan Xi; Teng Chen; Qingqing Gu; Yue Zhao; Xiaokai Chen; Zhonglin Jiang; Yong Chen; Luo Ji
>
> **备注:** Accepted by IJCNN 2025
>
> **摘要:** Large Language Models (LLM) are usually fine-tuned to participate in dyadic or two-party dialogues, which can not adapt well to multi-party dialogues (MPD), which hinders their applications in such scenarios including multi-personal meetings, discussions and daily communication. Previous LLM-based researches mainly focus on the multi-agent framework, while their base LLMs are still pairwisely fine-tuned. In this work, we design a multi-party fine-tuning framework (MuPaS) for LLMs on the multi-party dialogue datasets, and prove such a straightforward framework can let the LLM align with the multi-party conversation style efficiently and effectively. We also design two training strategies which can convert MuPaS into the MPD simulator. Substantial experiments show that MuPaS can achieve state-of-the-art multi-party response, higher accuracy of the-next-speaker prediction, higher human and automatic evaluated utterance qualities, and can even generate reasonably with out-of-distribution scene, topic and role descriptions. The MuPaS framework bridges the LLM training with more complicated multi-party applications, such as conversation generation, virtual rehearsal or meta-universe.
>
---
#### [replaced 034] SeriesBench: A Benchmark for Narrative-Driven Drama Series Understanding
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.21435v3](http://arxiv.org/pdf/2504.21435v3)**

> **作者:** Chenkai Zhang; Yiming Lei; Zeming Liu; Haitao Leng; Shaoguo Liu; Tingting Gao; Qingjie Liu; Yunhong Wang
>
> **备注:** 29 pages, 15 figures, CVPR 2025
>
> **摘要:** With the rapid development of Multi-modal Large Language Models (MLLMs), an increasing number of benchmarks have been established to evaluate the video understanding capabilities of these models. However, these benchmarks focus on standalone videos and mainly assess "visual elements" like human actions and object states. In reality, contemporary videos often encompass complex and continuous narratives, typically presented as a series. To address this challenge, we propose SeriesBench, a benchmark consisting of 105 carefully curated narrative-driven series, covering 28 specialized tasks that require deep narrative understanding. Specifically, we first select a diverse set of drama series spanning various genres. Then, we introduce a novel long-span narrative annotation method, combined with a full-information transformation approach to convert manual annotations into diverse task formats. To further enhance model capacity for detailed analysis of plot structures and character relationships within series, we propose a novel narrative reasoning framework, PC-DCoT. Extensive results on SeriesBench indicate that existing MLLMs still face significant challenges in understanding narrative-driven series, while PC-DCoT enables these MLLMs to achieve performance improvements. Overall, our SeriesBench and PC-DCoT highlight the critical necessity of advancing model capabilities to understand narrative-driven series, guiding the future development of MLLMs. SeriesBench is publicly available at https://github.com/zackhxn/SeriesBench-CVPR2025.
>
---
#### [replaced 035] Multi-Modal Language Models as Text-to-Image Model Evaluators
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.00759v2](http://arxiv.org/pdf/2505.00759v2)**

> **作者:** Jiahui Chen; Candace Ross; Reyhane Askari-Hemmat; Koustuv Sinha; Melissa Hall; Michal Drozdzal; Adriana Romero-Soriano
>
> **摘要:** The steady improvements of text-to-image (T2I) generative models lead to slow deprecation of automatic evaluation benchmarks that rely on static datasets, motivating researchers to seek alternative ways to evaluate the T2I progress. In this paper, we explore the potential of multi-modal large language models (MLLMs) as evaluator agents that interact with a T2I model, with the objective of assessing prompt-generation consistency and image aesthetics. We present Multimodal Text-to-Image Eval (MT2IE), an evaluation framework that iteratively generates prompts for evaluation, scores generated images and matches T2I evaluation of existing benchmarks with a fraction of the prompts used in existing static benchmarks. Moreover, we show that MT2IE's prompt-generation consistency scores have higher correlation with human judgment than scores previously introduced in the literature. MT2IE generates prompts that are efficient at probing T2I model performance, producing the same relative T2I model rankings as existing benchmarks while using only 1/80th the number of prompts for evaluation.
>
---
#### [replaced 036] Discriminative Finetuning of Generative Large Language Models without Reward Models and Human Preference Data
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.18679v2](http://arxiv.org/pdf/2502.18679v2)**

> **作者:** Siqi Guo; Ilgee Hong; Vicente Balmaseda; Changlong Yu; Liang Qiu; Xin Liu; Haoming Jiang; Tuo Zhao; Tianbao Yang
>
> **备注:** 18 pages, 7 figures
>
> **摘要:** Supervised fine-tuning (SFT) has become a crucial step for aligning pretrained large language models (LLMs) using supervised datasets of input-output pairs. However, despite being supervised, SFT is inherently limited by its generative training objective. To address its limitations, the existing common strategy is to follow SFT with a separate phase of preference optimization (PO), which relies on either human-labeled preference data or a strong reward model to guide the learning process. In this paper, we address the limitations of SFT by exploring one of the most successful techniques in conventional supervised learning: discriminative learning. We introduce Discriminative Fine-Tuning (DFT), an improved variant of SFT, which mitigates the burden of collecting human-labeled preference data or training strong reward models. Unlike SFT that employs a generative approach and overlooks negative data, DFT adopts a discriminative paradigm that increases the probability of positive answers while suppressing potentially negative ones, aiming for data prediction instead of token prediction. Our contributions include: (i) a discriminative probabilistic framework for fine-tuning LLMs by explicitly modeling the discriminative likelihood of an answer among all possible outputs given an input; (ii) efficient algorithms to optimize this discriminative likelihood; and (iii) extensive experiments demonstrating DFT's effectiveness, achieving performance better than SFT and comparable to if not better than SFT$\rightarrow$PO. The code can be found at https://github.com/Optimization-AI/DFT.
>
---
#### [replaced 037] Efficient Shapley Value-based Non-Uniform Pruning of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.01731v2](http://arxiv.org/pdf/2505.01731v2)**

> **作者:** Chuan Sun; Han Yu; Lizhen Cui; Xiaoxiao Li
>
> **摘要:** Pruning large language models (LLMs) is a promising solution for reducing model sizes and computational complexity while preserving performance. Traditional layer-wise pruning methods often adopt a uniform sparsity approach across all layers, which leads to suboptimal performance due to the varying significance of individual transformer layers within the model not being accounted for. To this end, we propose the Shapley Value-based Non-Uniform Pruning (SV-NUP) method for LLMs. This approach quantifies the contribution of each transformer layer to the overall model performance, enabling the assignment of tailored pruning budgets to different layers to retain critical parameters. To further improve efficiency, we design the Sliding Window-based Shapley Value approximation method. It substantially reduces computational overhead compared to exact SV calculation methods. Extensive experiments on various LLMs including LLaMA-v1, LLaMA-v2 and OPT demonstrate the effectiveness of the proposed approach. The results reveal that non-uniform pruning significantly enhances the performance of pruned models. Notably, SV-NUP achieves a reduction in perplexity (PPL) of 18.01% and 19.55% on LLaMA-7B and LLaMA-13B, respectively, compared to SparseGPT at 70% sparsity.
>
---
#### [replaced 038] Codifying Character Logic in Role-Playing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.07705v2](http://arxiv.org/pdf/2505.07705v2)**

> **作者:** Letian Peng; Jingbo Shang
>
> **摘要:** This paper introduces Codified Profiles for role-playing, a novel approach that represents character logic as structured, executable functions for behavioral decision-making. Each profile defines a set of functions parse_by_scene(scene) that outputs a list of logic-grounded assertions triggered_statements, using both explicit control structures (e.g., if-then-else) and condition checks like check_condition(scene, question), where each question is a semantically meaningful prompt about the scene (e.g., "Is the character in danger?") discriminated by the role-playing LLM as true, false, or unknown. This explicit representation offers three key advantages over traditional prompt-based profiles, which append character descriptions directly into text prompts: (1) Persistence, by enforcing complete and consistent execution of character logic, rather than relying on the model's implicit reasoning; (2) Updatability, through systematic inspection and revision of behavioral logic, which is difficult to track or debug in prompt-only approaches; (3) Controllable Randomness, by supporting stochastic behavior directly within the logic, enabling fine-grained variability that prompting alone struggles to achieve. To validate these advantages, we introduce a new benchmark constructed from 83 characters and 5,141 scenes curated from Fandom, using NLI-based scoring to compare character responses against ground-truth actions. Our experiments demonstrate the significant benefits of codified profiles in improving persistence, updatability, and behavioral diversity. Notably, by offloading a significant portion of reasoning to preprocessing, codified profiles enable even 1B-parameter models to perform high-quality role-playing, providing a scalable and efficient foundation for local deployment of role-play agents.
>
---
#### [replaced 039] DeepSeek-R1 Thoughtology: Let's think about LLM Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.07128v2](http://arxiv.org/pdf/2504.07128v2)**

> **作者:** Sara Vera Marjanović; Arkil Patel; Vaibhav Adlakha; Milad Aghajohari; Parishad BehnamGhader; Mehar Bhatia; Aditi Khandelwal; Austin Kraft; Benno Krojer; Xing Han Lù; Nicholas Meade; Dongchan Shin; Amirhossein Kazemnejad; Gaurav Kamath; Marius Mosbach; Karolina Stańczak; Siva Reddy
>
> **备注:** 142 pages, pre-print
>
> **摘要:** Large Reasoning Models like DeepSeek-R1 mark a fundamental shift in how LLMs approach complex problems. Instead of directly producing an answer for a given input, DeepSeek-R1 creates detailed multi-step reasoning chains, seemingly "thinking" about a problem before providing an answer. This reasoning process is publicly available to the user, creating endless opportunities for studying the reasoning behaviour of the model and opening up the field of Thoughtology. Starting from a taxonomy of DeepSeek-R1's basic building blocks of reasoning, our analyses on DeepSeek-R1 investigate the impact and controllability of thought length, management of long or confusing contexts, cultural and safety concerns, and the status of DeepSeek-R1 vis-\`a-vis cognitive phenomena, such as human-like language processing and world modelling. Our findings paint a nuanced picture. Notably, we show DeepSeek-R1 has a 'sweet spot' of reasoning, where extra inference time can impair model performance. Furthermore, we find a tendency for DeepSeek-R1 to persistently ruminate on previously explored problem formulations, obstructing further exploration. We also note strong safety vulnerabilities of DeepSeek-R1 compared to its non-reasoning counterpart, which can also compromise safety-aligned LLMs.
>
---
#### [replaced 040] Why do LLMs attend to the first token?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.02732v3](http://arxiv.org/pdf/2504.02732v3)**

> **作者:** Federico Barbero; Álvaro Arroyo; Xiangming Gu; Christos Perivolaropoulos; Michael Bronstein; Petar Veličković; Razvan Pascanu
>
> **摘要:** Large Language Models (LLMs) tend to attend heavily to the first token in the sequence -- creating a so-called attention sink. Many works have studied this phenomenon in detail, proposing various ways to either leverage or alleviate it. Attention sinks have been connected to quantisation difficulties, security issues, and streaming attention. Yet, while many works have provided conditions in which they occur or not, a critical question remains shallowly answered: Why do LLMs learn such patterns and how are they being used? In this work, we argue theoretically and empirically that this mechanism provides a method for LLMs to avoid over-mixing, connecting this to existing lines of work that study mathematically how information propagates in Transformers. We conduct experiments to validate our theoretical intuitions and show how choices such as context length, depth, and data packing influence the sink behaviour. We hope that this study provides a new practical perspective on why attention sinks are useful in LLMs, leading to a better understanding of the attention patterns that form during training.
>
---
#### [replaced 041] Evaluating the Symbol Binding Ability of Large Language Models for Multiple-Choice Questions in Vietnamese General Education
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2310.12059v5](http://arxiv.org/pdf/2310.12059v5)**

> **作者:** Duc-Vu Nguyen; Quoc-Nam Nguyen
>
> **备注:** Accepted at SoICT 2023
>
> **摘要:** In this paper, we evaluate the ability of large language models (LLMs) to perform multiple choice symbol binding (MCSB) for multiple choice question answering (MCQA) tasks in zero-shot, one-shot, and few-shot settings. We focus on Vietnamese, with fewer challenging MCQA datasets than in English. The two existing datasets, ViMMRC 1.0 and ViMMRC 2.0, focus on literature. Recent research in Vietnamese natural language processing (NLP) has focused on the Vietnamese National High School Graduation Examination (VNHSGE) from 2019 to 2023 to evaluate ChatGPT. However, these studies have mainly focused on how ChatGPT solves the VNHSGE step by step. We aim to create a novel and high-quality dataset by providing structured guidelines for typing LaTeX formulas for mathematics, physics, chemistry, and biology. This dataset can be used to evaluate the MCSB ability of LLMs and smaller language models (LMs) because it is typed in a strict LaTeX style. We focus on predicting the character (A, B, C, or D) that is the most likely answer to a question, given the context of the question. Our evaluation of six well-known LLMs, namely BLOOMZ-7.1B-MT, LLaMA-2-7B, LLaMA-2-70B, GPT-3, GPT-3.5, and GPT-4.0, on the ViMMRC 1.0 and ViMMRC 2.0 benchmarks and our proposed dataset shows promising results on the MCSB ability of LLMs for Vietnamese. The dataset is available for research purposes only.
>
---
#### [replaced 042] MobA: Multifaceted Memory-Enhanced Adaptive Planning for Efficient Mobile Task Automation
- **分类: cs.MA; cs.AI; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2410.13757v3](http://arxiv.org/pdf/2410.13757v3)**

> **作者:** Zichen Zhu; Hao Tang; Yansi Li; Dingye Liu; Hongshen Xu; Kunyao Lan; Danyang Zhang; Yixuan Jiang; Hao Zhou; Chenrun Wang; Situo Zhang; Liangtai Sun; Yixiao Wang; Yuheng Sun; Lu Chen; Kai Yu
>
> **备注:** NAACL 2025 Demo Track [code] https://github.com/OpenDFM/MobA [dataset] https://huggingface.co/datasets/OpenDFM/MobA-MobBench
>
> **摘要:** Existing Multimodal Large Language Model (MLLM)-based agents face significant challenges in handling complex GUI (Graphical User Interface) interactions on devices. These challenges arise from the dynamic and structured nature of GUI environments, which integrate text, images, and spatial relationships, as well as the variability in action spaces across different pages and tasks. To address these limitations, we propose MobA, a novel MLLM-based mobile assistant system. MobA introduces an adaptive planning module that incorporates a reflection mechanism for error recovery and dynamically adjusts plans to align with the real environment contexts and action module's execution capacity. Additionally, a multifaceted memory module provides comprehensive memory support to enhance adaptability and efficiency. We also present MobBench, a dataset designed for complex mobile interactions. Experimental results on MobBench and AndroidArena demonstrate MobA's ability to handle dynamic GUI environments and perform complex mobile tasks.
>
---
