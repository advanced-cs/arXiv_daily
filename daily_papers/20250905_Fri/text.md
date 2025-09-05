# 自然语言处理 cs.CL

- **最新发布 71 篇**

- **更新 51 篇**

## 最新发布

#### [new 001] MobileRAG: Enhancing Mobile Agent with Retrieval-Augmented Generation
- **分类: cs.CL; cs.CV**

- **简介: 论文提出MobileRAG框架，通过RAG技术解决移动代理依赖LLM、缺乏环境交互和记忆能力的问题，提升复杂任务处理能力，并引入MobileRAG-Eval评估基准。**

- **链接: [http://arxiv.org/pdf/2509.03891v1](http://arxiv.org/pdf/2509.03891v1)**

> **作者:** Gowen Loo; Chang Liu; Qinghong Yin; Xiang Chen; Jiawei Chen; Jingyuan Zhang; Yu Tian
>
> **摘要:** Smartphones have become indispensable in people's daily lives, permeating nearly every aspect of modern society. With the continuous advancement of large language models (LLMs), numerous LLM-based mobile agents have emerged. These agents are capable of accurately parsing diverse user queries and automatically assisting users in completing complex or repetitive operations. However, current agents 1) heavily rely on the comprehension ability of LLMs, which can lead to errors caused by misoperations or omitted steps during tasks, 2) lack interaction with the external environment, often terminating tasks when an app cannot fulfill user queries, and 3) lack memory capabilities, requiring each instruction to reconstruct the interface and being unable to learn from and correct previous mistakes. To alleviate the above issues, we propose MobileRAG, a mobile agents framework enhanced by Retrieval-Augmented Generation (RAG), which includes InterRAG, LocalRAG, and MemRAG. It leverages RAG to more quickly and accurately identify user queries and accomplish complex and long-sequence mobile tasks. Additionally, to more comprehensively assess the performance of MobileRAG, we introduce MobileRAG-Eval, a more challenging benchmark characterized by numerous complex, real-world mobile tasks that require external knowledge assistance. Extensive experimental results on MobileRAG-Eval demonstrate that MobileRAG can easily handle real-world mobile tasks, achieving 10.3\% improvement over state-of-the-art methods with fewer operational steps. Our code is publicly available at: https://github.com/liuxiaojieOutOfWorld/MobileRAG_arxiv
>
---
#### [new 002] Expanding Foundational Language Capabilities in Open-Source LLMs through a Korean Case Study
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文旨在提升开源大语言模型的韩语能力，解决其在韩语任务中的性能不足问题。通过设计Llama-3-Motif，采用LlamaPro与Masked Structure Growth技术，优化韩英平衡数据集，实现韩语基准测试超越现有模型，接近GPT-4水平。**

- **链接: [http://arxiv.org/pdf/2509.03972v1](http://arxiv.org/pdf/2509.03972v1)**

> **作者:** Junghwan Lim; Gangwon Jo; Sungmin Lee; Jiyoung Park; Dongseok Kim; Jihwan Kim; Junhyeok Lee; Wai Ting Cheung; Dahye Choi; Kibong Choi; Jaeyeon Huh; Beomgyu Kim; Jangwoong Kim; Taehyun Kim; Haesol Lee; Jeesoo Lee; Dongpin Oh; Changseok Song; Daewon Suh
>
> **摘要:** We introduce Llama-3-Motif, a language model consisting of 102 billion parameters, specifically designed to enhance Korean capabilities while retaining strong performance in English. Developed on the Llama 3 architecture, Llama-3-Motif employs advanced training techniques, including LlamaPro and Masked Structure Growth, to effectively scale the model without altering its core Transformer architecture. Using the MoAI platform for efficient training across hyperscale GPU clusters, we optimized Llama-3-Motif using a carefully curated dataset that maintains a balanced ratio of Korean and English data. Llama-3-Motif shows decent performance on Korean-specific benchmarks, outperforming existing models and achieving results comparable to GPT-4.
>
---
#### [new 003] RTQA : Recursive Thinking for Complex Temporal Knowledge Graph Question Answering with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文提出RTQA框架，用于复杂时间知识图谱问答，解决现有方法处理隐式约束不足、推理能力弱和错误传播问题，通过递归分解和多路径聚合提升性能。**

- **链接: [http://arxiv.org/pdf/2509.03995v1](http://arxiv.org/pdf/2509.03995v1)**

> **作者:** Zhaoyan Gong; Juan Li; Zhiqiang Liu; Lei Liang; Huajun Chen; Wen Zhang
>
> **备注:** EMNLP 2025
>
> **摘要:** Current temporal knowledge graph question answering (TKGQA) methods primarily focus on implicit temporal constraints, lacking the capability of handling more complex temporal queries, and struggle with limited reasoning abilities and error propagation in decomposition frameworks. We propose RTQA, a novel framework to address these challenges by enhancing reasoning over TKGs without requiring training. Following recursive thinking, RTQA recursively decomposes questions into sub-problems, solves them bottom-up using LLMs and TKG knowledge, and employs multi-path answer aggregation to improve fault tolerance. RTQA consists of three core components: the Temporal Question Decomposer, the Recursive Solver, and the Answer Aggregator. Experiments on MultiTQ and TimelineKGQA benchmarks demonstrate significant Hits@1 improvements in "Multiple" and "Complex" categories, outperforming state-of-the-art methods. Our code and data are available at https://github.com/zjukg/RTQA.
>
---
#### [new 004] Topic Identification in LLM Input-Output Pairs through the Lens of Information Bottleneck
- **分类: cs.CL; cs.LG; q-fin.GN**

- **简介: 该论文提出UDIB方法，基于信息瓶颈理论改进LLM输入输出对的主题识别，提升SDM框架对内在幻觉的检测能力。**

- **链接: [http://arxiv.org/pdf/2509.03533v1](http://arxiv.org/pdf/2509.03533v1)**

> **作者:** Igor Halperin
>
> **备注:** 26 pages, 4 figures
>
> **摘要:** Large Language Models (LLMs) are prone to critical failure modes, including \textit{intrinsic faithfulness hallucinations} (also known as confabulations), where a response deviates semantically from the provided context. Frameworks designed to detect this, such as Semantic Divergence Metrics (SDM), rely on identifying latent topics shared between prompts and responses, typically by applying geometric clustering to their sentence embeddings. This creates a disconnect, as the topics are optimized for spatial proximity, not for the downstream information-theoretic analysis. In this paper, we bridge this gap by developing a principled topic identification method grounded in the Deterministic Information Bottleneck (DIB) for geometric clustering. Our key contribution is to transform the DIB method into a practical algorithm for high-dimensional data by substituting its intractable KL divergence term with a computationally efficient upper bound. The resulting method, which we dub UDIB, can be interpreted as an entropy-regularized and robustified version of K-means that inherently favors a parsimonious number of informative clusters. By applying UDIB to the joint clustering of LLM prompt and response embeddings, we generate a shared topic representation that is not merely spatially coherent but is fundamentally structured to be maximally informative about the prompt-response relationship. This provides a superior foundation for the SDM framework and offers a novel, more sensitive tool for detecting confabulations.
>
---
#### [new 005] On Robustness and Reliability of Benchmark-Based Evaluation of LLMs
- **分类: cs.CL; cs.AI**

- **简介: 论文研究LLM在基准测试中的鲁棒性与可靠性，解决基准评估是否反映真实场景的问题。通过生成改写问题测试34个模型，发现分数下降，强调需更实际的基准。**

- **链接: [http://arxiv.org/pdf/2509.04013v1](http://arxiv.org/pdf/2509.04013v1)**

> **作者:** Riccardo Lunardi; Vincenzo Della Mea; Stefano Mizzaro; Kevin Roitero
>
> **备注:** Accepted at ECAI 2025
>
> **摘要:** Large Language Models (LLMs) effectiveness is usually evaluated by means of benchmarks such as MMLU, ARC-C, or HellaSwag, where questions are presented in their original wording, thus in a fixed, standardized format. However, real-world applications involve linguistic variability, requiring models to maintain their effectiveness across diverse rewordings of the same question or query. In this study, we systematically assess the robustness of LLMs to paraphrased benchmark questions and investigate whether benchmark-based evaluations provide a reliable measure of model capabilities. We systematically generate various paraphrases of all the questions across six different common benchmarks, and measure the resulting variations in effectiveness of 34 state-of-the-art LLMs, of different size and effectiveness. Our findings reveal that while LLM rankings remain relatively stable across paraphrased inputs, absolute effectiveness scores change, and decline significantly. This suggests that LLMs struggle with linguistic variability, raising concerns about their generalization abilities and evaluation methodologies. Furthermore, the observed performance drop challenges the reliability of benchmark-based evaluations, indicating that high benchmark scores may not fully capture a model's robustness to real-world input variations. We discuss the implications of these findings for LLM evaluation methodologies, emphasizing the need for robustness-aware benchmarks that better reflect practical deployment scenarios.
>
---
#### [new 006] NoteBar: An AI-Assisted Note-Taking System for Personal Knowledge Management
- **分类: cs.CL**

- **简介: 该论文提出NoteBar系统，解决AI笔记工具效率低问题，通过人格信息与高效模型自动分类笔记，构建包含MBTI人格的注释数据集，支持个人知识管理。**

- **链接: [http://arxiv.org/pdf/2509.03610v1](http://arxiv.org/pdf/2509.03610v1)**

> **作者:** Josh Wisoff; Yao Tang; Zhengyu Fang; Jordan Guzman; YuTang Wang; Alex Yu
>
> **摘要:** Note-taking is a critical practice for capturing, organizing, and reflecting on information in both academic and professional settings. The recent success of large language models has accelerated the development of AI-assisted tools, yet existing solutions often struggle with efficiency. We present NoteBar, an AI-assisted note-taking tool that leverages persona information and efficient language models to automatically organize notes into multiple categories and better support user workflows. To support research and evaluation in this space, we further introduce a novel persona-conditioned dataset of 3,173 notes and 8,494 annotated concepts across 16 MBTI personas, offering both diversity and semantic richness for downstream tasks. Finally, we demonstrate that NoteBar can be deployed in a practical and cost-effective manner, enabling interactive use without reliance on heavy infrastructure. Together, NoteBar and its accompanying dataset provide a scalable and extensible foundation for advancing AI-assisted personal knowledge management.
>
---
#### [new 007] Semantic Analysis of SNOMED CT Concept Co-occurrences in Clinical Documentation using MIMIC-IV
- **分类: cs.CL**

- **简介: 该论文通过分析MIMIC-IV临床文档中SNOMED CT概念的共现模式与语义相似性，探索嵌入模型对临床关联的补充作用。任务为临床文本挖掘，旨在提升文档分析准确性，揭示潜在临床关系并辅助决策支持。**

- **链接: [http://arxiv.org/pdf/2509.03662v1](http://arxiv.org/pdf/2509.03662v1)**

> **作者:** Ali Noori; Somya Mohanty; Prashanti Manda
>
> **摘要:** Clinical notes contain rich clinical narratives but their unstructured format poses challenges for large-scale analysis. Standardized terminologies such as SNOMED CT improve interoperability, yet understanding how concepts relate through co-occurrence and semantic similarity remains underexplored. In this study, we leverage the MIMIC-IV database to investigate the relationship between SNOMED CT concept co-occurrence patterns and embedding-based semantic similarity. Using Normalized Pointwise Mutual Information (NPMI) and pretrained embeddings (e.g., ClinicalBERT, BioBERT), we examine whether frequently co-occurring concepts are also semantically close, whether embeddings can suggest missing concepts, and how these relationships evolve temporally and across specialties. Our analyses reveal that while co-occurrence and semantic similarity are weakly correlated, embeddings capture clinically meaningful associations not always reflected in documentation frequency. Embedding-based suggestions frequently matched concepts later documented, supporting their utility for augmenting clinical annotations. Clustering of concept embeddings yielded coherent clinical themes (symptoms, labs, diagnoses, cardiovascular conditions) that map to patient phenotypes and care patterns. Finally, co-occurrence patterns linked to outcomes such as mortality and readmission demonstrate the practical utility of this approach. Collectively, our findings highlight the complementary value of co-occurrence statistics and semantic embeddings in improving documentation completeness, uncovering latent clinical relationships, and informing decision support and phenotyping applications.
>
---
#### [new 008] CANDY: Benchmarking LLMs' Limitations and Assistive Potential in Chinese Misinformation Fact-Checking
- **分类: cs.CL; cs.AI**

- **简介: 论文提出CANDY基准，评估LLMs在中文虚假信息事实核查中的局限性与辅助潜力。通过构建2万实例数据集，分析模型生成结论的误差类型，揭示事实伪造为主要问题，并验证LLMs作为辅助工具提升人类核查效率的可行性。**

- **链接: [http://arxiv.org/pdf/2509.03957v1](http://arxiv.org/pdf/2509.03957v1)**

> **作者:** Ruiling Guo; Xinwei Yang; Chen Huang; Tong Zhang; Yong Hu
>
> **备注:** Findings of EMNLP 2025
>
> **摘要:** The effectiveness of large language models (LLMs) to fact-check misinformation remains uncertain, despite their growing use. To this end, we present CANDY, a benchmark designed to systematically evaluate the capabilities and limitations of LLMs in fact-checking Chinese misinformation. Specifically, we curate a carefully annotated dataset of ~20k instances. Our analysis shows that current LLMs exhibit limitations in generating accurate fact-checking conclusions, even when enhanced with chain-of-thought reasoning and few-shot prompting. To understand these limitations, we develop a taxonomy to categorize flawed LLM-generated explanations for their conclusions and identify factual fabrication as the most common failure mode. Although LLMs alone are unreliable for fact-checking, our findings indicate their considerable potential to augment human performance when deployed as assistive tools in scenarios. Our dataset and code can be accessed at https://github.com/SCUNLP/CANDY
>
---
#### [new 009] MTQA:Matrix of Thought for Enhanced Reasoning in Complex Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 论文针对复杂问答任务中大模型推理能力不足的问题，提出MTQA框架，通过Matrix of Thought结构和事实校正机制提升效率与准确性，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.03918v1](http://arxiv.org/pdf/2509.03918v1)**

> **作者:** Fengxiao Tang; Yufeng Li; Zongzong Wu; Ming Zhao
>
> **摘要:** Complex Question Answering (QA) is a fundamental and challenging task in NLP. While large language models (LLMs) exhibit impressive performance in QA, they suffer from significant performance degradation when facing complex and abstract QA tasks due to insufficient reasoning capabilities. Works such as Chain-of-Thought (CoT) and Tree-of-Thought (ToT) aim to enhance LLMs' reasoning abilities, but they face issues such as in-layer redundancy in tree structures and single paths in chain structures. Although some studies utilize Retrieval-Augmented Generation (RAG) methods to assist LLMs in reasoning, the challenge of effectively utilizing large amounts of information involving multiple entities and hops remains critical. To address this, we propose the Matrix of Thought (MoT), a novel and efficient LLM thought structure. MoT explores the problem in both horizontal and vertical dimensions through the "column-cell communication" mechanism, enabling LLMs to actively engage in multi-strategy and deep-level thinking, reducing redundancy within the column cells and enhancing reasoning capabilities. Furthermore, we develop a fact-correction mechanism by constructing knowledge units from retrieved knowledge graph triples and raw text to enhance the initial knowledge for LLM reasoning and correct erroneous answers. This leads to the development of an efficient and accurate QA framework (MTQA). Experimental results show that our framework outperforms state-of-the-art methods on four widely-used datasets in terms of F1 and EM scores, with reasoning time only 14.4\% of the baseline methods, demonstrating both its efficiency and accuracy. The code for this framework is available at https://github.com/lyfiter/mtqa.
>
---
#### [new 010] PARCO: Phoneme-Augmented Robust Contextual ASR via Contrastive Entity Disambiguation
- **分类: cs.CL; cs.AI; cs.LG; cs.SD**

- **简介: 该论文针对ASR中命名实体识别与同音词歧义问题，提出PARCO方法。通过音素感知编码、对比实体消歧等技术，提升多token实体识别准确率，减少误判，显著优于基线模型。**

- **链接: [http://arxiv.org/pdf/2509.04357v1](http://arxiv.org/pdf/2509.04357v1)**

> **作者:** Jiajun He; Naoki Sawada; Koichi Miyazaki; Tomoki Toda
>
> **备注:** Accepted by ASRU 2025
>
> **摘要:** Automatic speech recognition (ASR) systems struggle with domain-specific named entities, especially homophones. Contextual ASR improves recognition but often fails to capture fine-grained phoneme variations due to limited entity diversity. Moreover, prior methods treat entities as independent tokens, leading to incomplete multi-token biasing. To address these issues, we propose Phoneme-Augmented Robust Contextual ASR via COntrastive entity disambiguation (PARCO), which integrates phoneme-aware encoding, contrastive entity disambiguation, entity-level supervision, and hierarchical entity filtering. These components enhance phonetic discrimination, ensure complete entity retrieval, and reduce false positives under uncertainty. Experiments show that PARCO achieves CER of 4.22% on Chinese AISHELL-1 and WER of 11.14% on English DATA2 under 1,000 distractors, significantly outperforming baselines. PARCO also demonstrates robust gains on out-of-domain datasets like THCHS-30 and LibriSpeech.
>
---
#### [new 011] Towards Stable and Personalised Profiles for Lexical Alignment in Spoken Human-Agent Dialogue
- **分类: cs.CL; cs.HC**

- **简介: 该论文旨在构建稳定、个性化的词汇资料以实现人机对话中的词汇对齐。通过调整数据量和词性类别项目数，评估性能指标，发现小规模资料（10分钟语音，每类5-10词）在性能与数据效率间取得最佳平衡，为对话代理的词汇对齐提供基础。**

- **链接: [http://arxiv.org/pdf/2509.04104v1](http://arxiv.org/pdf/2509.04104v1)**

> **作者:** Keara Schaaij; Roel Boumans; Tibor Bosse; Iris Hendrickx
>
> **备注:** Accepted for TSD 2025
>
> **摘要:** Lexical alignment, where speakers start to use similar words across conversation, is known to contribute to successful communication. However, its implementation in conversational agents remains underexplored, particularly considering the recent advancements in large language models (LLMs). As a first step towards enabling lexical alignment in human-agent dialogue, this study draws on strategies for personalising conversational agents and investigates the construction of stable, personalised lexical profiles as a basis for lexical alignment. Specifically, we varied the amounts of transcribed spoken data used for construction as well as the number of items included in the profiles per part-of-speech (POS) category and evaluated profile performance across time using recall, coverage, and cosine similarity metrics. It was shown that smaller and more compact profiles, created after 10 min of transcribed speech containing 5 items for adjectives, 5 items for conjunctions, and 10 items for adverbs, nouns, pronouns, and verbs each, offered the best balance in both performance and data efficiency. In conclusion, this study offers practical insights into constructing stable, personalised lexical profiles, taking into account minimal data requirements, serving as a foundational step toward lexical alignment strategies in conversational agents.
>
---
#### [new 012] What if I ask in \textit{alia lingua}? Measuring Functional Similarity Across Languages
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究多语言模型跨语言输出一致性，应用κ_p指标分析20种语言和47个受试者，发现模型规模越大跨语言一致性越高，且模型内一致性优于模型间，凸显κ_p作为评估工具的价值。**

- **链接: [http://arxiv.org/pdf/2509.04032v1](http://arxiv.org/pdf/2509.04032v1)**

> **作者:** Debangan Mishra; Arihant Rastogi; Agyeya Negi; Shashwat Goel; Ponnurangam Kumaraguru
>
> **备注:** Preprint, 11 Pages
>
> **摘要:** How similar are model outputs across languages? In this work, we study this question using a recently proposed model similarity metric $\kappa_p$ applied to 20 languages and 47 subjects in GlobalMMLU. Our analysis reveals that a model's responses become increasingly consistent across languages as its size and capability grow. Interestingly, models exhibit greater cross-lingual consistency within themselves than agreement with other models prompted in the same language. These results highlight not only the value of $\kappa_p$ as a practical tool for evaluating multilingual reliability, but also its potential to guide the development of more consistent multilingual systems.
>
---
#### [new 013] MLSD: A Novel Few-Shot Learning Approach to Enhance Cross-Target and Cross-Domain Stance Detection
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出MLSD方法，通过度量学习与三元组损失构建判别性嵌入空间，解决跨目标和跨域立场检测中数据不足的问题，提升模型在新领域中的表现。**

- **链接: [http://arxiv.org/pdf/2509.03725v1](http://arxiv.org/pdf/2509.03725v1)**

> **作者:** Parush Gera; Tempestt Neal
>
> **摘要:** We present the novel approach for stance detection across domains and targets, Metric Learning-Based Few-Shot Learning for Cross-Target and Cross-Domain Stance Detection (MLSD). MLSD utilizes metric learning with triplet loss to capture semantic similarities and differences between stance targets, enhancing domain adaptation. By constructing a discriminative embedding space, MLSD allows a cross-target or cross-domain stance detection model to acquire useful examples from new target domains. We evaluate MLSD in multiple cross-target and cross-domain scenarios across two datasets, showing statistically significant improvement in stance detection performance across six widely used stance detection models.
>
---
#### [new 014] Breaking the Mirror: Activation-Based Mitigation of Self-Preference in LLM Evaluators
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对LLM评估者中的自我偏好偏差问题，提出通过构建数据集和对比激活添加等方法生成steering vectors，在推理时减少不当自我偏好，但发现其在合法案例中不稳定，揭示自我偏好多方向性，推动更稳健干预。**

- **链接: [http://arxiv.org/pdf/2509.03647v1](http://arxiv.org/pdf/2509.03647v1)**

> **作者:** Dani Roytburg; Matthew Bozoukov; Matthew Nguyen; Jou Barzdukas; Simon Fu; Narmeen Oozeer
>
> **摘要:** Large language models (LLMs) increasingly serve as automated evaluators, yet they suffer from "self-preference bias": a tendency to favor their own outputs over those of other models. This bias undermines fairness and reliability in evaluation pipelines, particularly for tasks like preference tuning and model routing. We investigate whether lightweight steering vectors can mitigate this problem at inference time without retraining. We introduce a curated dataset that distinguishes self-preference bias into justified examples of self-preference and unjustified examples of self-preference, and we construct steering vectors using two methods: Contrastive Activation Addition (CAA) and an optimization-based approach. Our results show that steering vectors can reduce unjustified self-preference bias by up to 97\%, substantially outperforming prompting and direct preference optimization baselines. Yet steering vectors are unstable on legitimate self-preference and unbiased agreement, implying self-preference spans multiple or nonlinear directions. This underscores both their promise and limits as safeguards for LLM-as-judges and motivates more robust interventions.
>
---
#### [new 015] E-ARMOR: Edge case Assessment and Review of Multilingual Optical Character Recognition
- **分类: cs.CL; cs.AI**

- **简介: 该论文评估多语言OCR系统在边缘设备的性能，对比传统OCR与LVLMs在资源受限环境下的效率与准确性，提出优化的Sprinklr-Edge-OCR模型，证明传统系统在边缘部署中更具优势。**

- **链接: [http://arxiv.org/pdf/2509.03615v1](http://arxiv.org/pdf/2509.03615v1)**

> **作者:** Aryan Gupta; Anupam Purwar
>
> **备注:** Sprinklr OCR provides a fast and compute light way of performing OCR
>
> **摘要:** Optical Character Recognition (OCR) in multilingual, noisy, and diverse real-world images remains a significant challenge for optical character recognition systems. With the rise of Large Vision-Language Models (LVLMs), there is growing interest in their ability to generalize and reason beyond fixed OCR pipelines. In this work, we introduce Sprinklr-Edge-OCR, a novel OCR system built specifically optimized for edge deployment in resource-constrained environments. We present a large-scale comparative evaluation of five state-of-the-art LVLMs (InternVL, Qwen, GOT OCR, LLaMA, MiniCPM) and two traditional OCR systems (Sprinklr-Edge-OCR, SuryaOCR) on a proprietary, doubly hand annotated dataset of multilingual (54 languages) images. Our benchmark covers a broad range of metrics including accuracy, semantic consistency, language coverage, computational efficiency (latency, memory, GPU usage), and deployment cost. To better reflect real-world applicability, we also conducted edge case deployment analysis, evaluating model performance on CPU only environments. Among the results, Qwen achieved the highest precision (0.54), while Sprinklr-Edge-OCR delivered the best overall F1 score (0.46) and outperformed others in efficiency, processing images 35 faster (0.17 seconds per image on average) and at less than 0.01 of the cost (0.006 USD per 1,000 images) compared to LVLM. Our findings demonstrate that the most optimal OCR systems for edge deployment are the traditional ones even in the era of LLMs due to their low compute requirements, low latency, and very high affordability.
>
---
#### [new 016] Drivel-ology: Challenging LLMs with Interpreting Nonsense with Depth
- **分类: cs.CL**

- **简介: 论文提出Drivelology任务，评估LLM处理有深度无意义文本的能力，发现其在语用理解上的不足，构建多语言数据集并分析模型表现。**

- **链接: [http://arxiv.org/pdf/2509.03867v1](http://arxiv.org/pdf/2509.03867v1)**

> **作者:** Yang Wang; Chenghao Xiao; Chia-Yi Hsiao; Zi Yan Chang; Chi-Li Chen; Tyler Loakman; Chenghua Lin
>
> **备注:** Accepted for oral presentation at the EMNLP 2025 Main Conference
>
> **摘要:** We introduce Drivelology, a unique linguistic phenomenon characterised as "nonsense with depth", utterances that are syntactically coherent yet pragmatically paradoxical, emotionally loaded, or rhetorically subversive. While such expressions may resemble surface-level nonsense, they encode implicit meaning requiring contextual inference, moral reasoning, or emotional interpretation. We find that current large language models (LLMs), despite excelling at many natural language processing (NLP) tasks, consistently fail to grasp the layered semantics of Drivelological text. To investigate this, we construct a small but diverse benchmark dataset of over 1,200 meticulously curated examples, with select instances in English, Mandarin, Spanish, French, Japanese, and Korean. Annotation was especially challenging: each of the examples required careful expert review to verify that it truly reflected Drivelological characteristics. The process involved multiple rounds of discussion and adjudication to address disagreements, highlighting the subtle and subjective nature of the Drivelology. We evaluate a range of LLMs on classification, generation, and reasoning tasks. Our results reveal clear limitations of LLMs: models often confuse Drivelology with shallow nonsense, produce incoherent justifications, or miss the implied rhetorical function altogether. These findings highlight a deeper representational gap in LLMs' pragmatic understanding and challenge the assumption that statistical fluency implies cognitive comprehension. We release our dataset and code to facilitate further research in modelling linguistic depth beyond surface-level coherence.
>
---
#### [new 017] Decoding the Poetic Language of Emotion in Korean Modern Poetry: Insights from a Human-Labeled Dataset and AI Modeling
- **分类: cs.CL; cs.CY; cs.LG**

- **简介: 本研究构建KPoEM数据集，解决韩语现代诗歌隐喻与文化差异导致的情感分析难题。通过微调模型提升情感分类性能，并结合计算与文学分析，推动诗歌情感量化研究。**

- **链接: [http://arxiv.org/pdf/2509.03932v1](http://arxiv.org/pdf/2509.03932v1)**

> **作者:** Iro Lim; Haein Ji; Byungjun Kim
>
> **备注:** 30 pages, 13 tables, 2 figures, Digital Humanities and Social Sciences Korea Conference, James Joo-Jin Kim Center for Korean Studies, University of Pennsylvania, Philadelphia, USA
>
> **摘要:** This study introduces KPoEM (Korean Poetry Emotion Mapping) , a novel dataset for computational emotion analysis in modern Korean poetry. Despite remarkable progress in text-based emotion classification using large language models, poetry-particularly Korean poetry-remains underexplored due to its figurative language and cultural specificity. We built a multi-label emotion dataset of 7,662 entries, including 7,007 line-level entries from 483 poems and 615 work-level entries, annotated with 44 fine-grained emotion categories from five influential Korean poets. A state-of-the-art Korean language model fine-tuned on this dataset significantly outperformed previous models, achieving 0.60 F1-micro compared to 0.34 from models trained on general corpora. The KPoEM model, trained through sequential fine-tuning-first on general corpora and then on the KPoEM dataset-demonstrates not only an enhanced ability to identify temporally and culturally specific emotional expressions, but also a strong capacity to preserve the core sentiments of modern Korean poetry. This study bridges computational methods and literary analysis, presenting new possibilities for the quantitative exploration of poetic emotions through structured data that faithfully retains the emotional and cultural nuances of Korean literature.
>
---
#### [new 018] Arabic Chatbot Technologies in Education: An Overview
- **分类: cs.CL**

- **简介: 该论文综述阿拉伯语教育聊天机器人技术，分析其应用现状与技术特点，揭示现代技术应用不足的研究缺口，并提出未来研究方向。**

- **链接: [http://arxiv.org/pdf/2509.04066v1](http://arxiv.org/pdf/2509.04066v1)**

> **作者:** Hicham Bourhil; Yacine El Younoussi
>
> **备注:** Published as a book chapter in: Transformaci\'on Digital en la Educaci\'on: Innovaciones y Desaf\'ios desde los Campus Virtuales (UA Journals, 2024), pp. 11-14
>
> **摘要:** The recent advancements in Artificial Intelligence (AI) in general, and in Natural Language Processing (NLP) in particular, and some of its applications such as chatbots, have led to their implementation in different domains like education, healthcare, tourism, and customer service. Since the COVID-19 pandemic, there has been an increasing interest in these digital technologies to allow and enhance remote access. In education, e-learning systems have been massively adopted worldwide. The emergence of Large Language Models (LLM) such as BERT (Bidirectional Encoder Representations from Transformers) and GPT (Generative Pre-trained Transformers) made chatbots even more popular. In this study, we present a survey on existing Arabic chatbots in education and their different characteristics such as the adopted approaches, language variety, and metrics used to measure their performance. We were able to identified some research gaps when we discovered that, despite the success of chatbots in other languages such as English, only a few educational Arabic chatbots used modern techniques. Finally, we discuss future directions of research in this field.
>
---
#### [new 019] Exploring NLP Benchmarks in an Extremely Low-Resource Setting
- **分类: cs.CL**

- **简介: 论文针对低资源语言Ladin，通过翻译意大利语数据生成合成数据集，解决标注数据不足问题。采用过滤与回译确保质量，提升机器翻译效果，并发布首个Ladin情感分析与MCQA数据集。**

- **链接: [http://arxiv.org/pdf/2509.03962v1](http://arxiv.org/pdf/2509.03962v1)**

> **作者:** Ulin Nuha; Adam Jatowt
>
> **摘要:** The effectiveness of Large Language Models (LLMs) diminishes for extremely low-resource languages, such as indigenous languages, primarily due to the lack of labeled data. Despite growing interest, the availability of high-quality natural language processing (NLP) datasets for these languages remains limited, making it difficult to develop robust language technologies. This paper addresses such gap by focusing on Ladin, an endangered Romance language, specifically targeting the Val Badia variant. Leveraging a small set of parallel Ladin-Italian sentence pairs, we create synthetic datasets for sentiment analysis and multiple-choice question answering (MCQA) by translating monolingual Italian data. To ensure linguistic quality and reliability, we apply rigorous filtering and back-translation procedures in our method. We further demonstrate that incorporating these synthetic datasets into machine translation training leads to substantial improvements over existing Italian-Ladin translation baselines. Our contributions include the first publicly available sentiment analysis and MCQA datasets for Ladin, establishing foundational resources that can support broader NLP research and downstream applications for this underrepresented language.
>
---
#### [new 020] Explicit and Implicit Data Augmentation for Social Event Detection
- **分类: cs.CL; cs.SI**

- **简介: 该论文提出SED-Aug框架，通过显式文本生成与隐式特征扰动结合，解决社会事件检测中标注成本高的问题，提升模型鲁棒性与数据多样性，在Twitter数据集上F1得分提升超15%。**

- **链接: [http://arxiv.org/pdf/2509.04202v1](http://arxiv.org/pdf/2509.04202v1)**

> **作者:** Congbo Ma; Yuxia Wang; Jia Wu; Jian Yang; Jing Du; Zitai Qiu; Qing Li; Hu Wang; Preslav Nakov
>
> **摘要:** Social event detection involves identifying and categorizing important events from social media, which relies on labeled data, but annotation is costly and labor-intensive. To address this problem, we propose Augmentation framework for Social Event Detection (SED-Aug), a plug-and-play dual augmentation framework, which combines explicit text-based and implicit feature-space augmentation to enhance data diversity and model robustness. The explicit augmentation utilizes large language models to enhance textual information through five diverse generation strategies. For implicit augmentation, we design five novel perturbation techniques that operate in the feature space on structural fused embeddings. These perturbations are crafted to keep the semantic and relational properties of the embeddings and make them more diverse. Specifically, SED-Aug outperforms the best baseline model by approximately 17.67% on the Twitter2012 dataset and by about 15.57% on the Twitter2018 dataset in terms of the average F1 score. The code is available at GitHub: https://github.com/congboma/SED-Aug.
>
---
#### [new 021] AR$^2$: Adversarial Reinforcement Learning for Abstract Reasoning in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出AR²框架，通过对抗强化学习提升大语言模型的抽象推理能力，解决现有方法缺乏显式抽象训练的问题。教师模型生成复杂描述，学生模型提取计算内核，提升未见任务的准确性。**

- **链接: [http://arxiv.org/pdf/2509.03537v1](http://arxiv.org/pdf/2509.03537v1)**

> **作者:** Cheng-Kai Yeh; Hsing-Wang Lee; Chung-Hung Kuo; Hen-Hsen Huang
>
> **备注:** 7 pages, accepted by CIKM 2025 as a short paper
>
> **摘要:** Abstraction--the ability to recognize and distill essential computational patterns from complex problem statements--is a foundational skill in computer science, critical both for human problem-solvers and coding-oriented large language models (LLMs). Despite recent advances in training LLMs for code generation using reinforcement learning (RL), most existing approaches focus primarily on superficial pattern recognition, overlooking explicit training for abstraction. In this study, we propose AR$^2$ (Adversarial Reinforcement Learning for Abstract Reasoning), a novel framework explicitly designed to enhance the abstraction abilities of LLMs. AR$^2$ employs a teacher model to transform kernel problems into narrative-rich, challenging descriptions without changing their fundamental logic. Simultaneously, a student coding model is trained to solve these complex narrative problems by extracting their underlying computational kernels. Experimental results demonstrate that AR$^2$ substantially improves the student model's accuracy on previously unseen, challenging programming tasks, underscoring abstraction as a key skill for enhancing LLM generalization.
>
---
#### [new 022] SelfAug: Mitigating Catastrophic Forgetting in Retrieval-Augmented Generation via Distribution Self-Alignment
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对RAG任务中微调导致的灾难性遗忘问题，提出SelfAug方法通过分布自对齐保留模型语义分布，平衡下游学习与通用能力保持。**

- **链接: [http://arxiv.org/pdf/2509.03934v1](http://arxiv.org/pdf/2509.03934v1)**

> **作者:** Yuqing Huang; Rongyang Zhang; Qimeng Wang; Chengqiang Lu; Yan Gao; Yi Wu; Yao Hu; Xuyang Zhi; Guiquan Liu; Xin Li; Hao Wang; Enhong Chen
>
> **摘要:** Recent advancements in large language models (LLMs) have revolutionized natural language processing through their remarkable capabilities in understanding and executing diverse tasks. While supervised fine-tuning, particularly in Retrieval-Augmented Generation (RAG) scenarios, effectively enhances task-specific performance, it often leads to catastrophic forgetting, where models lose their previously acquired knowledge and general capabilities. Existing solutions either require access to general instruction data or face limitations in preserving the model's original distribution. To overcome these limitations, we propose SelfAug, a self-distribution alignment method that aligns input sequence logits to preserve the model's semantic distribution, thereby mitigating catastrophic forgetting and improving downstream performance. Extensive experiments demonstrate that SelfAug achieves a superior balance between downstream learning and general capability retention. Our comprehensive empirical analysis reveals a direct correlation between distribution shifts and the severity of catastrophic forgetting in RAG scenarios, highlighting how the absence of RAG capabilities in general instruction tuning leads to significant distribution shifts during fine-tuning. Our findings not only advance the understanding of catastrophic forgetting in RAG contexts but also provide a practical solution applicable across diverse fine-tuning scenarios. Our code is publicly available at https://github.com/USTC-StarTeam/SelfAug.
>
---
#### [new 023] Improving Factuality in LLMs via Inference-Time Knowledge Graph Construction
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出在推理时动态构建知识图谱，结合内部与外部知识，解决LLMs事实不一致问题，提升问答准确性与可解释性。**

- **链接: [http://arxiv.org/pdf/2509.03540v1](http://arxiv.org/pdf/2509.03540v1)**

> **作者:** Shanglin Wu; Lihui Liu; Jinho D. Choi; Kai Shu
>
> **摘要:** Large Language Models (LLMs) often struggle with producing factually consistent answers due to limitations in their parametric memory. Retrieval-Augmented Generation (RAG) methods address this issue by incorporating external knowledge from trusted sources at inference time. However, such methods typically treat knowledge as unstructured text, which limits their ability to support compositional reasoning and identify factual inconsistencies. To overcome these limitations, we propose a novel framework that dynamically constructs and expands knowledge graphs (KGs) during inference, integrating both internal knowledge extracted from LLMs and external information retrieved from external sources. Our method begins by extracting a seed KG from the question via prompting, followed by iterative expansion using the LLM's latent knowledge. The graph is then selectively refined through external retrieval, enhancing factual coverage and correcting inaccuracies. We evaluate our approach on three diverse factual QA benchmarks, demonstrating consistent improvements in factual accuracy, answer precision, and interpretability over baseline prompting and static KG-augmented methods. Our findings suggest that inference-time KG construction is a promising direction for enhancing LLM factuality in a structured, interpretable, and scalable manner.
>
---
#### [new 024] MAGneT: Coordinated Multi-Agent Generation of Synthetic Multi-Turn Mental Health Counseling Sessions
- **分类: cs.CL; cs.AI**

- **简介: 论文提出MAGneT框架，通过多智能体协作生成高质量合成心理咨询会话，解决数据稀缺问题。分解生成任务为子任务，整合自动与专家评估，提升生成质量与治疗对齐度，实验显示优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.04183v1](http://arxiv.org/pdf/2509.04183v1)**

> **作者:** Aishik Mandal; Tanmoy Chakraborty; Iryna Gurevych
>
> **备注:** 25 pages, 29 figures
>
> **摘要:** The growing demand for scalable psychological counseling highlights the need for fine-tuning open-source Large Language Models (LLMs) with high-quality, privacy-compliant data, yet such data remains scarce. Here we introduce MAGneT, a novel multi-agent framework for synthetic psychological counseling session generation that decomposes counselor response generation into coordinated sub-tasks handled by specialized LLM agents, each modeling a key psychological technique. Unlike prior single-agent approaches, MAGneT better captures the structure and nuance of real counseling. In addition, we address inconsistencies in prior evaluation protocols by proposing a unified evaluation framework integrating diverse automatic and expert metrics. Furthermore, we expand the expert evaluations from four aspects of counseling in previous works to nine aspects, enabling a more thorough and robust assessment of data quality. Empirical results show that MAGneT significantly outperforms existing methods in quality, diversity, and therapeutic alignment of the generated counseling sessions, improving general counseling skills by 3.2% and CBT-specific skills by 4.3% on average on cognitive therapy rating scale (CTRS). Crucially, experts prefer MAGneT-generated sessions in 77.2% of cases on average across all aspects. Moreover, fine-tuning an open-source model on MAGneT-generated sessions shows better performance, with improvements of 6.3% on general counseling skills and 7.3% on CBT-specific skills on average on CTRS over those fine-tuned with sessions generated by baseline methods. We also make our code and data public.
>
---
#### [new 025] Measuring Bias or Measuring the Task: Understanding the Brittle Nature of LLM Gender Biases
- **分类: cs.CL**

- **简介: 该论文研究任务提示对LLM性别偏见测量的影响，发现微小提示变化显著改变偏见结果，离散选择指标放大偏见，揭示评估脆弱性，并引发测试设计与生态效度的新问题。**

- **链接: [http://arxiv.org/pdf/2509.04373v1](http://arxiv.org/pdf/2509.04373v1)**

> **作者:** Bufan Gao; Elisa Kreiss
>
> **摘要:** As LLMs are increasingly applied in socially impactful settings, concerns about gender bias have prompted growing efforts both to measure and mitigate such bias. These efforts often rely on evaluation tasks that differ from natural language distributions, as they typically involve carefully constructed task prompts that overtly or covertly signal the presence of gender bias-related content. In this paper, we examine how signaling the evaluative purpose of a task impacts measured gender bias in LLMs. Concretely, we test models under prompt conditions that (1) make the testing context salient, and (2) make gender-focused content salient. We then assess prompt sensitivity across four task formats with both token-probability and discrete-choice metrics. We find that even minor prompt changes can substantially alter bias outcomes, sometimes reversing their direction entirely. Discrete-choice metrics further tend to amplify bias relative to probabilistic measures. These findings do not only highlight the brittleness of LLM gender bias evaluations but open a new puzzle for the NLP benchmarking and development community: To what extent can well-controlled testing designs trigger LLM ``testing mode'' performance, and what does this mean for the ecological validity of future benchmarks.
>
---
#### [new 026] False Sense of Security: Why Probing-based Malicious Input Detection Fails to Generalize
- **分类: cs.CL**

- **简介: 论文针对基于探测的恶意输入检测方法，发现其依赖表面模式而非语义有害性，通过实验验证并提出改进方向，指出当前方法存在安全假象。**

- **链接: [http://arxiv.org/pdf/2509.03888v1](http://arxiv.org/pdf/2509.03888v1)**

> **作者:** Cheng Wang; Zeming Wei; Qin Liu; Muhao Chen
>
> **摘要:** Large Language Models (LLMs) can comply with harmful instructions, raising serious safety concerns despite their impressive capabilities. Recent work has leveraged probing-based approaches to study the separability of malicious and benign inputs in LLMs' internal representations, and researchers have proposed using such probing methods for safety detection. We systematically re-examine this paradigm. Motivated by poor out-of-distribution performance, we hypothesize that probes learn superficial patterns rather than semantic harmfulness. Through controlled experiments, we confirm this hypothesis and identify the specific patterns learned: instructional patterns and trigger words. Our investigation follows a systematic approach, progressing from demonstrating comparable performance of simple n-gram methods, to controlled experiments with semantically cleaned datasets, to detailed analysis of pattern dependencies. These results reveal a false sense of security around current probing-based approaches and highlight the need to redesign both models and evaluation protocols, for which we provide further discussions in the hope of suggesting responsible further research in this direction. We have open-sourced the project at https://github.com/WangCheng0116/Why-Probe-Fails.
>
---
#### [new 027] Multilevel Analysis of Cryptocurrency News using RAG Approach with Fine-Tuned Mistral Large Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出基于RAG与微调Mistral模型的多级多任务分析框架，解决加密货币新闻分析中的幻觉与信息整合问题。通过生成图/文本摘要、层次化整合及知识图谱表示，实现定量与定性分析，提供全面洞察。**

- **链接: [http://arxiv.org/pdf/2509.03527v1](http://arxiv.org/pdf/2509.03527v1)**

> **作者:** Bohdan M. Pavlyshenko
>
> **摘要:** In the paper, we consider multilevel multitask analysis of cryptocurrency news using a fine-tuned Mistral 7B large language model with retrieval-augmented generation (RAG). On the first level of analytics, the fine-tuned model generates graph and text summaries with sentiment scores as well as JSON representations of summaries. Higher levels perform hierarchical stacking that consolidates sets of graph-based and text-based summaries as well as summaries of summaries into comprehensive reports. The combination of graph and text summaries provides complementary views of cryptocurrency news. The model is fine-tuned with 4-bit quantization using the PEFT/LoRA approach. The representation of cryptocurrency news as knowledge graph can essentially eliminate problems with large language model hallucinations. The obtained results demonstrate that the use of fine-tuned Mistral 7B LLM models for multilevel cryptocurrency news analysis can conduct informative qualitative and quantitative analytics, providing important insights.
>
---
#### [new 028] QuesGenie: Intelligent Multimodal Question Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出QuesGenie系统，解决教育资源丰富但缺乏定制化练习材料的问题。通过多模态输入处理、强化学习与交互界面，实现自动化生成多样化问题，提升资源利用效率与用户体验。**

- **链接: [http://arxiv.org/pdf/2509.03535v1](http://arxiv.org/pdf/2509.03535v1)**

> **作者:** Ahmed Mubarak; Amna Ahmed; Amira Nasser; Aya Mohamed; Fares El-Sadek; Mohammed Ahmed; Ahmed Salah; Youssef Sobhy
>
> **备注:** 7 pages, 8 figures, 12 tables. Supervised by Dr. Ahmed Salah and TA Youssef Sobhy
>
> **摘要:** In today's information-rich era, learners have access to abundant educational resources, but the lack of practice materials tailored to these resources presents a significant challenge. This project addresses that gap by developing a multi-modal question generation system that can automatically generate diverse question types from various content formats. The system features four major components: multi-modal input handling, question generation, reinforcement learning from human feedback (RLHF), and an end-to-end interactive interface. This project lays the foundation for automated, scalable, and intelligent question generation, carefully balancing resource efficiency, robust functionality and a smooth user experience.
>
---
#### [new 029] Facts Fade Fast: Evaluating Memorization of Outdated Medical Knowledge in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文评估LLMs对过时医疗知识的依赖，创建MedRevQA和MedChangeQA数据集，发现所有模型均依赖旧知识，分析原因并提出改进方向，旨在提升医疗AI的准确性和时效性。**

- **链接: [http://arxiv.org/pdf/2509.04304v1](http://arxiv.org/pdf/2509.04304v1)**

> **作者:** Juraj Vladika; Mahdi Dhaini; Florian Matthes
>
> **备注:** Accepted to Findings of EMNLP 2025
>
> **摘要:** The growing capabilities of Large Language Models (LLMs) show significant potential to enhance healthcare by assisting medical researchers and physicians. However, their reliance on static training data is a major risk when medical recommendations evolve with new research and developments. When LLMs memorize outdated medical knowledge, they can provide harmful advice or fail at clinical reasoning tasks. To investigate this problem, we introduce two novel question-answering (QA) datasets derived from systematic reviews: MedRevQA (16,501 QA pairs covering general biomedical knowledge) and MedChangeQA (a subset of 512 QA pairs where medical consensus has changed over time). Our evaluation of eight prominent LLMs on the datasets reveals consistent reliance on outdated knowledge across all models. We additionally analyze the influence of obsolete pre-training data and training strategies to explain this phenomenon and propose future directions for mitigation, laying the groundwork for developing more current and reliable medical AI systems.
>
---
#### [new 030] Enhancing Speech Large Language Models through Reinforced Behavior Alignment
- **分类: cs.CL; eess.AS**

- **简介: 论文提出RBA框架，通过自我合成数据和强化学习提升语音大模型的指令遵循能力，解决跨模态差异问题，优于传统方法并扩展至多任务。**

- **链接: [http://arxiv.org/pdf/2509.03526v1](http://arxiv.org/pdf/2509.03526v1)**

> **作者:** Yansong Liu; Jiateng Li; Yuan Liu
>
> **摘要:** The recent advancements of Large Language Models (LLMs) have spurred considerable research interest in extending their linguistic capabilities beyond text to other modalities, which leads to emergence of speech-based LLMs (SpeechLMs) with capability of processing user request in either speech or textual formats. However, owing to inter-modal discrepancies, these SpeechLMs still exhibit a significant performance gap compared to their text-based LLM counterparts in instruction-following, particularly when confronted with the dynamic and variable nature of user speech. To address this challenge, this paper introduces a framework termed Reinforced Behavior Alignment (RBA), designed to bolster the language generation proficiency of SpeechLMs. Instead of relying on supervised fine-tuning from human annotations, RBA employs a self-synthesis methodology to generate extensive, high-fidelity alignment data by a powerful teacher LLM. Then SpeechLMs is aligned its behavior with that of a teacher using a reinforcement learning-based approach. Experimental results demonstrate that this method effectively enhances the instruction-following capabilities of SpeechLMs that outperform conventional distillation baselines. Crucially, we demonstrate that RBA can be seamlessly extended to tasks such including spoken question answering and speech-to-text translation, attaining state-of-the-art performance on open benchmarks with only self-generated data.
>
---
#### [new 031] NE-PADD: Leveraging Named Entity Knowledge for Robust Partial Audio Deepfake Detection via Attention Aggregation
- **分类: cs.CL**

- **简介: 论文提出NE-PADD方法，针对部分音频深度伪造检测（PADD）任务，解决帧级定位与语义信息利用问题。通过SpeechNER和PADD双分支，结合注意力融合与转移机制，实验表明优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.03829v1](http://arxiv.org/pdf/2509.03829v1)**

> **作者:** Huhong Xian; Rui Liu; Berrak Sisman; Haizhou Li
>
> **摘要:** Different from traditional sentence-level audio deepfake detection (ADD), partial audio deepfake detection (PADD) requires frame-level positioning of the location of fake speech. While some progress has been made in this area, leveraging semantic information from audio, especially named entities, remains an underexplored aspect. To this end, we propose NE-PADD, a novel method for Partial Audio Deepfake Detection (PADD) that leverages named entity knowledge through two parallel branches: Speech Name Entity Recognition (SpeechNER) and PADD. The approach incorporates two attention aggregation mechanisms: Attention Fusion (AF) for combining attention weights and Attention Transfer (AT) for guiding PADD with named entity semantics using an auxiliary loss. Built on the PartialSpoof-NER dataset, experiments show our method outperforms existing baselines, proving the effectiveness of integrating named entity knowledge in PADD. The code is available at https://github.com/AI-S2-Lab/NE-PADD.
>
---
#### [new 032] MultiWikiQA: A Reading Comprehension Benchmark in 300+ Languages
- **分类: cs.CL**

- **简介: 该论文提出MultiWikiQA数据集，用于多语言阅读理解评估。通过LLM生成跨306种语言的问答对，结合人类评估验证质量，对比不同模型性能，解决多语言NLP基准缺失问题。**

- **链接: [http://arxiv.org/pdf/2509.04111v1](http://arxiv.org/pdf/2509.04111v1)**

> **作者:** Dan Saattrup Smart
>
> **摘要:** We introduce a new reading comprehension dataset, dubbed MultiWikiQA, which covers 306 languages. The context data comes from Wikipedia articles, with questions generated by an LLM and the answers appearing verbatim in the Wikipedia articles. We conduct a crowdsourced human evaluation of the fluency of the generated questions across 30 of the languages, providing evidence that the questions are of good quality. We evaluate 6 different language models, both decoder and encoder models of varying sizes, showing that the benchmark is sufficiently difficult and that there is a large performance discrepancy amongst the languages. The dataset and survey evaluations are freely available.
>
---
#### [new 033] Inverse IFEval: Can LLMs Unlearn Stubborn Training Conventions to Follow Real Instructions?
- **分类: cs.CL**

- **简介: 论文提出Inverse IFEval基准，评估LLMs克服训练偏见、遵循对抗性指令的能力，解决其认知惯性问题，通过八类挑战测试，构建跨领域数据集，强调提升指令适应性的重要性。**

- **链接: [http://arxiv.org/pdf/2509.04292v1](http://arxiv.org/pdf/2509.04292v1)**

> **作者:** Qinyan Zhang; Xinping Lei; Ruijie Miao; Yu Fu; Haojie Fan; Le Chang; Jiafan Hou; Dingling Zhang; Zhongfei Hou; Ziqiang Yang; Changxin Pu; Fei Hu; Jingkai Liu; Mengyun Liu; Yang Liu; Xiang Gao; Jiaheng Liu; Tong Yang; Zaiyuan Wang; Ge Zhang; Wenhao Huang
>
> **摘要:** Large Language Models (LLMs) achieve strong performance on diverse tasks but often exhibit cognitive inertia, struggling to follow instructions that conflict with the standardized patterns learned during supervised fine-tuning (SFT). To evaluate this limitation, we propose Inverse IFEval, a benchmark that measures models Counter-intuitive Abilitytheir capacity to override training-induced biases and comply with adversarial instructions. Inverse IFEval introduces eight types of such challenges, including Question Correction, Intentional Textual Flaws, Code without Comments, and Counterfactual Answering. Using a human-in-the-loop pipeline, we construct a dataset of 1012 high-quality Chinese and English questions across 23 domains, evaluated under an optimized LLM-as-a-Judge framework. Experiments on existing leading LLMs demonstrate the necessity of our proposed Inverse IFEval benchmark. Our findings emphasize that future alignment efforts should not only pursue fluency and factual correctness but also account for adaptability under unconventional contexts. We hope that Inverse IFEval serves as both a diagnostic tool and a foundation for developing methods that mitigate cognitive inertia, reduce overfitting to narrow patterns, and ultimately enhance the instruction-following reliability of LLMs in diverse and unpredictable real-world scenarios.
>
---
#### [new 034] Joint Modeling of Entities and Discourse Relations for Coherence Assessment
- **分类: cs.CL**

- **简介: 该论文针对文本连贯性评估任务，解决现有方法仅单独建模实体或话语关系的问题，提出两种联合建模方法，整合实体与话语关系特征以提升连贯性评估效果。**

- **链接: [http://arxiv.org/pdf/2509.04182v1](http://arxiv.org/pdf/2509.04182v1)**

> **作者:** Wei Liu; Michael Strube
>
> **备注:** EMNLP 2025
>
> **摘要:** In linguistics, coherence can be achieved by different means, such as by maintaining reference to the same set of entities across sentences and by establishing discourse relations between them. However, most existing work on coherence modeling focuses exclusively on either entity features or discourse relation features, with little attention given to combining the two. In this study, we explore two methods for jointly modeling entities and discourse relations for coherence assessment. Experiments on three benchmark datasets show that integrating both types of features significantly enhances the performance of coherence models, highlighting the benefits of modeling both simultaneously for coherence evaluation.
>
---
#### [new 035] Multimodal Proposal for an AI-Based Tool to Increase Cross-Assessment of Messages
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文提出多模态框架，解决盈利电话会议分层结构建模问题，通过分层语篇树与双阶段Transformer生成语义嵌入，应用于财务分析及跨领域高风险沟通。**

- **链接: [http://arxiv.org/pdf/2509.03529v1](http://arxiv.org/pdf/2509.03529v1)**

> **作者:** Alejandro Álvarez Castro; Joaquín Ordieres-Meré
>
> **备注:** Presented at NLMLT2025 (https://airccse.org/csit/V15N16.html), 15 pages, 5 figures
>
> **摘要:** Earnings calls represent a uniquely rich and semi-structured source of financial communication, blending scripted managerial commentary with unscripted analyst dialogue. Although recent advances in financial sentiment analysis have integrated multi-modal signals, such as textual content and vocal tone, most systems rely on flat document-level or sentence-level models, failing to capture the layered discourse structure of these interactions. This paper introduces a novel multi-modal framework designed to generate semantically rich and structurally aware embeddings of earnings calls, by encoding them as hierarchical discourse trees. Each node, comprising either a monologue or a question-answer pair, is enriched with emotional signals derived from text, audio, and video, as well as structured metadata including coherence scores, topic labels, and answer coverage assessments. A two-stage transformer architecture is proposed: the first encodes multi-modal content and discourse metadata at the node level using contrastive learning, while the second synthesizes a global embedding for the entire conference. Experimental results reveal that the resulting embeddings form stable, semantically meaningful representations that reflect affective tone, structural logic, and thematic alignment. Beyond financial reporting, the proposed system generalizes to other high-stakes unscripted communicative domains such as tele-medicine, education, and political discourse, offering a robust and explainable approach to multi-modal discourse representation. This approach offers practical utility for downstream tasks such as financial forecasting and discourse evaluation, while also providing a generalizable method applicable to other domains involving high-stakes communication.
>
---
#### [new 036] Speech-Based Cognitive Screening: A Systematic Evaluation of LLM Adaptation Strategies
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文系统评估了LLM适应策略在语音筛查中的应用，旨在提升阿尔茨海默病等痴呆症的检测准确性。通过对比文本与多模态模型，分析不同微调方法（如类中心演示、推理增强、参数高效微调）的效果，发现适配策略显著影响检测性能，开源模型可匹敌商业系统。**

- **链接: [http://arxiv.org/pdf/2509.03525v1](http://arxiv.org/pdf/2509.03525v1)**

> **作者:** Fatemeh Taherinezhad; Mohamad Javad Momeni Nezhad; Sepehr Karimi; Sina Rashidi; Ali Zolnour; Maryam Dadkhah; Yasaman Haghbin; Hossein AzadMaleki; Maryam Zolnoori
>
> **摘要:** Over half of US adults with Alzheimer disease and related dementias remain undiagnosed, and speech-based screening offers a scalable detection approach. We compared large language model adaptation strategies for dementia detection using the DementiaBank speech corpus, evaluating nine text-only models and three multimodal audio-text models on recordings from DementiaBank speech corpus. Adaptations included in-context learning with different demonstration selection policies, reasoning-augmented prompting, parameter-efficient fine-tuning, and multimodal integration. Results showed that class-centroid demonstrations achieved the highest in-context learning performance, reasoning improved smaller models, and token-level fine-tuning generally produced the best scores. Adding a classification head substantially improved underperforming models. Among multimodal models, fine-tuned audio-text systems performed well but did not surpass the top text-only models. These findings highlight that model adaptation strategies, including demonstration selection, reasoning design, and tuning method, critically influence speech-based dementia detection, and that properly adapted open-weight models can match or exceed commercial systems.
>
---
#### [new 037] VoxRole: A Comprehensive Benchmark for Evaluating Speech-Based Role-Playing Agents
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文提出VoxRole基准，解决语音角色扮演代理评估中忽视语音特征与缺乏标准基准的问题，构建包含65.6小时电影对话数据的资源，通过自动化流程生成角色档案并进行多维评估。**

- **链接: [http://arxiv.org/pdf/2509.03940v1](http://arxiv.org/pdf/2509.03940v1)**

> **作者:** Weihao Wu; Liang Cao; Xinyu Wu; Zhiwei Lin; Rui Niu; Jingbei Li; Zhiyong Wu
>
> **摘要:** Recent significant advancements in Large Language Models (LLMs) have greatly propelled the development of Role-Playing Conversational Agents (RPCAs). These systems aim to create immersive user experiences through consistent persona adoption. However, current RPCA research faces dual limitations. First, existing work predominantly focuses on the textual modality, entirely overlooking critical paralinguistic features including intonation, prosody, and rhythm in speech, which are essential for conveying character emotions and shaping vivid identities. Second, the speech-based role-playing domain suffers from a long-standing lack of standardized evaluation benchmarks. Most current spoken dialogue datasets target only fundamental capability assessments, featuring thinly sketched or ill-defined character profiles. Consequently, they fail to effectively quantify model performance on core competencies like long-term persona consistency. To address this critical gap, we introduce VoxRole, the first comprehensive benchmark specifically designed for the evaluation of speech-based RPCAs. The benchmark comprises 13335 multi-turn dialogues, totaling 65.6 hours of speech from 1228 unique characters across 261 movies. To construct this resource, we propose a novel two-stage automated pipeline that first aligns movie audio with scripts and subsequently employs an LLM to systematically build multi-dimensional profiles for each character. Leveraging VoxRole, we conduct a multi-dimensional evaluation of contemporary spoken dialogue models, revealing crucial insights into their respective strengths and limitations in maintaining persona consistency.
>
---
#### [new 038] ResearchPulse: Building Method-Experiment Chains through Multi-Document Scientific Inference
- **分类: cs.CL; cs.MM**

- **简介: 论文提出多文档科学推理任务，解决跨论文结构化对齐问题，设计ResearchPulse框架及基准数据集，实验表明其优于GPT-4o。**

- **链接: [http://arxiv.org/pdf/2509.03565v1](http://arxiv.org/pdf/2509.03565v1)**

> **作者:** Qi Chen; Jingxuan Wei; Zhuoya Yao; Haiguang Wang; Gaowei Wu; Bihui Yu; Siyuan Li; Cheng Tan
>
> **备注:** Accepted to ACM MM 2025
>
> **摘要:** Understanding how scientific ideas evolve requires more than summarizing individual papers-it demands structured, cross-document reasoning over thematically related research. In this work, we formalize multi-document scientific inference, a new task that extracts and aligns motivation, methodology, and experimental results across related papers to reconstruct research development chains. This task introduces key challenges, including temporally aligning loosely structured methods and standardizing heterogeneous experimental tables. We present ResearchPulse, an agent-based framework that integrates instruction planning, scientific content extraction, and structured visualization. It consists of three coordinated agents: a Plan Agent for task decomposition, a Mmap-Agent that constructs motivation-method mind maps, and a Lchart-Agent that synthesizes experimental line charts. To support this task, we introduce ResearchPulse-Bench, a citation-aware benchmark of annotated paper clusters. Experiments show that our system, despite using 7B-scale agents, consistently outperforms strong baselines like GPT-4o in semantic alignment, structural consistency, and visual fidelity. The dataset are available in https://huggingface.co/datasets/ResearchPulse/ResearchPulse-Bench.
>
---
#### [new 039] Can Language Models Handle a Non-Gregorian Calendar?
- **分类: cs.CL**

- **简介: 该论文评估语言模型处理非公历（如日本历）的能力，发现模型在历法转换和跨历法一致性上表现不足，强调需提升文化特定历法理解。**

- **链接: [http://arxiv.org/pdf/2509.04432v1](http://arxiv.org/pdf/2509.04432v1)**

> **作者:** Mutsumi Sasaki; Go Kamoda; Ryosuke Takahashi; Kosuke Sato; Kentaro Inui; Keisuke Sakaguchi; Benjamin Heinzerling
>
> **摘要:** Temporal reasoning and knowledge are essential capabilities for language models (LMs). While much prior work has analyzed and improved temporal reasoning in LMs, most studies have focused solely on the Gregorian calendar. However, many non-Gregorian systems, such as the Japanese, Hijri, and Hebrew calendars, are in active use and reflect culturally grounded conceptions of time. If and how well current LMs can accurately handle such non-Gregorian calendars has not been evaluated so far. Here, we present a systematic evaluation of how well open-source LMs handle one such non-Gregorian system: the Japanese calendar. For our evaluation, we create datasets for four tasks that require both temporal knowledge and temporal reasoning. Evaluating a range of English-centric and Japanese-centric LMs, we find that some models can perform calendar conversions, but even Japanese-centric models struggle with Japanese-calendar arithmetic and with maintaining consistency across calendars. Our results highlight the importance of developing LMs that are better equipped for culture-specific calendar understanding.
>
---
#### [new 040] Improving Narrative Classification and Explanation via Fine Tuned Language Models
- **分类: cs.CL**

- **简介: 该论文针对新闻中隐含叙事的分类与解释任务，解决传统NLP方法在检测隐含信息和多标签分类上的不足。通过微调BERT结合GPT-4o及ReACT框架，提升叙事识别与解释的准确性，并引入结构化知识库减少幻觉，应用于媒体分析等场景。**

- **链接: [http://arxiv.org/pdf/2509.04077v1](http://arxiv.org/pdf/2509.04077v1)**

> **作者:** Rishit Tyagi; Rahul Bouri; Mohit Gupta
>
> **摘要:** Understanding covert narratives and implicit messaging is essential for analyzing bias and sentiment. Traditional NLP methods struggle with detecting subtle phrasing and hidden agendas. This study tackles two key challenges: (1) multi-label classification of narratives and sub-narratives in news articles, and (2) generating concise, evidence-based explanations for dominant narratives. We fine-tune a BERT model with a recall-oriented approach for comprehensive narrative detection, refining predictions using a GPT-4o pipeline for consistency. For narrative explanation, we propose a ReACT (Reasoning + Acting) framework with semantic retrieval-based few-shot prompting, ensuring grounded and relevant justifications. To enhance factual accuracy and reduce hallucinations, we incorporate a structured taxonomy table as an auxiliary knowledge base. Our results show that integrating auxiliary knowledge in prompts improves classification accuracy and justification reliability, with applications in media analysis, education, and intelligence gathering.
>
---
#### [new 041] Real-Time Detection of Hallucinated Entities in Long-Form Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出一种实时检测长文本生成中虚构实体的方法，解决现有技术成本高、适用性差的问题。通过网络搜索标注数据，训练高效分类器，实现70B参数模型的准确检测，并公开数据集促进复用。**

- **链接: [http://arxiv.org/pdf/2509.03531v1](http://arxiv.org/pdf/2509.03531v1)**

> **作者:** Oscar Obeso; Andy Arditi; Javier Ferrando; Joshua Freeman; Cameron Holmes; Neel Nanda
>
> **摘要:** Large language models are now routinely used in high-stakes applications where hallucinations can cause serious harm, such as medical consultations or legal advice. Existing hallucination detection methods, however, are impractical for real-world use, as they are either limited to short factual queries or require costly external verification. We present a cheap, scalable method for real-time identification of hallucinated tokens in long-form generations, and scale it effectively to 70B parameter models. Our approach targets \emph{entity-level hallucinations} -- e.g., fabricated names, dates, citations -- rather than claim-level, thereby naturally mapping to token-level labels and enabling streaming detection. We develop an annotation methodology that leverages web search to annotate model responses with grounded labels indicating which tokens correspond to fabricated entities. This dataset enables us to train effective hallucination classifiers with simple and efficient methods such as linear probes. Evaluating across four model families, our classifiers consistently outperform baselines on long-form responses, including more expensive methods such as semantic entropy (e.g., AUC 0.90 vs 0.71 for Llama-3.3-70B), and are also an improvement in short-form question-answering settings. Moreover, despite being trained only with entity-level labels, our probes effectively detect incorrect answers in mathematical reasoning tasks, indicating generalization beyond entities. While our annotation methodology is expensive, we find that annotated responses from one model can be used to train effective classifiers on other models; accordingly, we publicly release our datasets to facilitate reuse. Overall, our work suggests a promising new approach for scalable, real-world hallucination detection.
>
---
#### [new 042] The ProLiFIC dataset: Leveraging LLMs to Unveil the Italian Lawmaking Process
- **分类: cs.CL; cs.CY; cs.LG**

- **简介: 该论文构建ProLiFIC数据集，解决法律过程挖掘数据不足问题，通过LLMs结构化非结构化数据，为法律PM提供基准。**

- **链接: [http://arxiv.org/pdf/2509.03528v1](http://arxiv.org/pdf/2509.03528v1)**

> **作者:** Matilde Contestabile; Chiara Ferrara; Alberto Giovannetti; Giovanni Parrillo; Andrea Vandin
>
> **摘要:** Process Mining (PM), initially developed for industrial and business contexts, has recently been applied to social systems, including legal ones. However, PM's efficacy in the legal domain is limited by the accessibility and quality of datasets. We introduce ProLiFIC (Procedural Lawmaking Flow in Italian Chambers), a comprehensive event log of the Italian lawmaking process from 1987 to 2022. Created from unstructured data from the Normattiva portal and structured using large language models (LLMs), ProLiFIC aligns with recent efforts in integrating PM with LLMs. We exemplify preliminary analyses and propose ProLiFIC as a benchmark for legal PM, fostering new developments.
>
---
#### [new 043] SPFT-SQL: Enhancing Large Language Model for Text-to-SQL Parsing by Self-Play Fine-Tuning
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对文本到SQL解析任务，提出SPFT-SQL方法，解决现有自玩微调在生成准确SQL上的不足。通过验证迭代数据合成与错误驱动损失机制，提升模型区分正确与错误SQL的能力，实验表明效果优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.03937v1](http://arxiv.org/pdf/2509.03937v1)**

> **作者:** Yuhao Zhang; Shaoming Duan; Jinhang Su; Chuanyi Liu; Peiyi Han
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Despite the significant advancements of self-play fine-tuning (SPIN), which can transform a weak large language model (LLM) into a strong one through competitive interactions between models of varying capabilities, it still faces challenges in the Text-to-SQL task. SPIN does not generate new information, and the large number of correct SQL queries produced by the opponent model during self-play reduces the main model's ability to generate accurate SQL queries. To address this challenge, we propose a new self-play fine-tuning method tailored for the Text-to-SQL task, called SPFT-SQL. Prior to self-play, we introduce a verification-based iterative fine-tuning approach, which synthesizes high-quality fine-tuning data iteratively based on the database schema and validation feedback to enhance model performance, while building a model base with varying capabilities. During the self-play fine-tuning phase, we propose an error-driven loss method that incentivizes incorrect outputs from the opponent model, enabling the main model to distinguish between correct SQL and erroneous SQL generated by the opponent model, thereby improving its ability to generate correct SQL. Extensive experiments and in-depth analyses on six open-source LLMs and five widely used benchmarks demonstrate that our approach outperforms existing state-of-the-art (SOTA) methods.
>
---
#### [new 044] A RoBERTa-Based Functional Syntax Annotation Model for Chinese Texts
- **分类: cs.CL; I.2.7**

- **简介: 论文提出基于RoBERTa的中文功能句法标注模型，构建标注数据集并微调模型，实现高精度命名实体识别，填补自动标注系统空白。**

- **链接: [http://arxiv.org/pdf/2509.04046v1](http://arxiv.org/pdf/2509.04046v1)**

> **作者:** Han Xiaohui; Zhang Yunlong; Guo Yuxi
>
> **备注:** The paper includes 10 pages, 6 tables, and 4 figures. This project is completed with the assistance of National Center for Language Technology and Digital Economy Research (No. GJLX20250002), and is funded by Heilongjiang Language Research Committee Project Construction of an Adaptive Intelligent Chinese Learning Platform for International Students in China (No. G2025Y003)
>
> **摘要:** Systemic Functional Grammar and its branch, Cardiff Grammar, have been widely applied to discourse analysis, semantic function research, and other tasks across various languages and texts. However, an automatic annotation system based on this theory for Chinese texts has not yet been developed, which significantly constrains the application and promotion of relevant theories. To fill this gap, this research introduces a functional syntax annotation model for Chinese based on RoBERTa (Robustly Optimized BERT Pretraining Approach). The study randomly selected 4,100 sentences from the People's Daily 2014 corpus and annotated them according to functional syntax theory to establish a dataset for training. The study then fine-tuned the RoBERTa-Chinese wwm-ext model based on the dataset to implement the named entity recognition task, achieving an F1 score of 0.852 on the test set that significantly outperforms other comparative models. The model demonstrated excellent performance in identifying core syntactic elements such as Subject (S), Main Verb (M), and Complement (C). Nevertheless, there remains room for improvement in recognizing entities with imbalanced label samples. As the first integration of functional syntax with attention-based NLP models, this research provides a new method for automated Chinese functional syntax analysis and lays a solid foundation for subsequent studies.
>
---
#### [new 045] Measuring How (Not Just Whether) VLMs Build Common Ground
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出四维度指标评估VLMs的互动共同理解构建，解决现有基准仅关注任务成功与否的问题，通过对比模型与人类表现，揭示任务成功与grounding质量的脱节。**

- **链接: [http://arxiv.org/pdf/2509.03805v1](http://arxiv.org/pdf/2509.03805v1)**

> **作者:** Saki Imai; Mert İnan; Anthony Sicilia; Malihe Alikhani
>
> **摘要:** Large vision language models (VLMs) increasingly claim reasoning skills, yet current benchmarks evaluate them in single-turn or question answering settings. However, grounding is an interactive process in which people gradually develop shared understanding through ongoing communication. We introduce a four-metric suite (grounding efficiency, content alignment, lexical adaptation, and human-likeness) to systematically evaluate VLM performance in interactive grounding contexts. We deploy the suite on 150 self-play sessions of interactive referential games between three proprietary VLMs and compare them with human dyads. All three models diverge from human patterns on at least three metrics, while GPT4o-mini is the closest overall. We find that (i) task success scores do not indicate successful grounding and (ii) high image-utterance alignment does not necessarily predict task success. Our metric suite and findings offer a framework for future research on VLM grounding.
>
---
#### [new 046] Reading Between the Signs: Predicting Future Suicidal Ideation from Adolescent Social Media Texts
- **分类: cs.CL**

- **简介: 该论文提出Early-SIB模型，通过分析青少年社交媒体历史帖子序列，预测其未来自杀意念与行为，解决传统方法无法提前预警的问题，实现0.73平衡准确率。**

- **链接: [http://arxiv.org/pdf/2509.03530v1](http://arxiv.org/pdf/2509.03530v1)**

> **作者:** Paul Blum; Enrico Liscio; Ruixuan Zhang; Caroline Figueroa; Pradeep K. Murukannaiah
>
> **摘要:** Suicide is a leading cause of death among adolescents (12-18), yet predicting it remains a significant challenge. Many cases go undetected due to a lack of contact with mental health services. Social media, however, offers a unique opportunity, as young people often share their thoughts and struggles online in real time. In this work, we propose a novel task and method to approach it: predicting suicidal ideation and behavior (SIB) from forum posts before an adolescent explicitly expresses suicidal ideation on an online forum. This predictive framing, where no self-disclosure is used as input at any stage, remains largely unexplored in the suicide prediction literature. To this end, we introduce Early-SIB, a transformer-based model that sequentially processes the posts a user writes and engages with to predict whether they will write a SIB post. Our model achieves a balanced accuracy of 0.73 for predicting future SIB on a Dutch youth forum, demonstrating that such tools can offer a meaningful addition to traditional methods.
>
---
#### [new 047] Synthesizing Sheet Music Problems for Evaluation and Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出基于音乐理论的乐谱问题合成框架，解决LLMs/MLLMs缺乏乐谱理解评估基准与训练数据的问题。通过生成文本/视觉模态的可验证问题，构建SSMR-Bench基准与训练集，提升模型乐谱推理与创作能力。**

- **链接: [http://arxiv.org/pdf/2509.04059v1](http://arxiv.org/pdf/2509.04059v1)**

> **作者:** Zhilin Wang; Zhe Yang; Yun Luo; Yafu Li; Haoran Zhang; Runzhe Zhan; Derek F. Wong; Jizhe Zhou; Yu Cheng
>
> **备注:** 11 pages
>
> **摘要:** Enhancing the ability of Large Language Models (LLMs) and Multimodal Large Language Models (MLLMs) to interpret sheet music is a crucial step toward building AI musicians. However, current research lacks both evaluation benchmarks and training data for sheet music reasoning. To address this, we propose the idea of synthesizing sheet music problems grounded in music theory, which can serve both as evaluation benchmarks and as training data for reinforcement learning with verifiable rewards (RLVR). We introduce a data synthesis framework that generates verifiable sheet music questions in both textual and visual modalities, leading to the Synthetic Sheet Music Reasoning Benchmark (SSMR-Bench) and a complementary training set. Evaluation results on SSMR-Bench show the importance of models' reasoning abilities in interpreting sheet music. At the same time, the poor performance of Gemini 2.5-Pro highlights the challenges that MLLMs still face in interpreting sheet music in a visual format. By leveraging synthetic data for RLVR, Qwen3-8B-Base and Qwen2.5-VL-Instruct achieve improvements on the SSMR-Bench. Besides, the trained Qwen3-8B-Base surpasses GPT-4 in overall performance on MusicTheoryBench and achieves reasoning performance comparable to GPT-4 with the strategies of Role play and Chain-of-Thought. Notably, its performance on math problems also improves relative to the original Qwen3-8B-Base. Furthermore, our results show that the enhanced reasoning ability can also facilitate music composition. In conclusion, we are the first to propose the idea of synthesizing sheet music problems based on music theory rules, and demonstrate its effectiveness not only in advancing model reasoning for sheet music understanding but also in unlocking new possibilities for AI-assisted music creation.
>
---
#### [new 048] A Comprehensive Survey on Trustworthiness in Reasoning with Large Language Models
- **分类: cs.CL; cs.AI; cs.CR**

- **简介: 该论文综述了大语言模型中基于链式推理（CoT）的可信性研究，聚焦真实性、安全性等五大维度，分析现有方法、局限及未来方向，旨在提升模型可信度。**

- **链接: [http://arxiv.org/pdf/2509.03871v1](http://arxiv.org/pdf/2509.03871v1)**

> **作者:** Yanbo Wang; Yongcan Yu; Jian Liang; Ran He
>
> **备注:** 38 pages. This survey considers papers published up to June 30, 2025. Work in progress
>
> **摘要:** The development of Long-CoT reasoning has advanced LLM performance across various tasks, including language understanding, complex problem solving, and code generation. This paradigm enables models to generate intermediate reasoning steps, thereby improving both accuracy and interpretability. However, despite these advancements, a comprehensive understanding of how CoT-based reasoning affects the trustworthiness of language models remains underdeveloped. In this paper, we survey recent work on reasoning models and CoT techniques, focusing on five core dimensions of trustworthy reasoning: truthfulness, safety, robustness, fairness, and privacy. For each aspect, we provide a clear and structured overview of recent studies in chronological order, along with detailed analyses of their methodologies, findings, and limitations. Future research directions are also appended at the end for reference and discussion. Overall, while reasoning techniques hold promise for enhancing model trustworthiness through hallucination mitigation, harmful content detection, and robustness improvement, cutting-edge reasoning models themselves often suffer from comparable or even greater vulnerabilities in safety, robustness, and privacy. By synthesizing these insights, we hope this work serves as a valuable and timely resource for the AI safety community to stay informed on the latest progress in reasoning trustworthiness. A full list of related papers can be found at \href{https://github.com/ybwang119/Awesome-reasoning-safety}{https://github.com/ybwang119/Awesome-reasoning-safety}.
>
---
#### [new 049] Align-then-Slide: A complete evaluation framework for Ultra-Long Document-Level Machine Translation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Align-then-Slide框架，解决超长文档级机器翻译中现有评估方法无法处理全文档输出和句子对齐问题。通过自动对齐和多粒度分块评估，验证其与人类判断一致，提升翻译质量。**

- **链接: [http://arxiv.org/pdf/2509.03809v1](http://arxiv.org/pdf/2509.03809v1)**

> **作者:** Jiaxin Guo; Daimeng Wei; Yuanchang Luo; Xiaoyu Chen; Zhanglin Wu; Huan Yang; Hengchao Shang; Zongyao Li; Zhiqiang Rao; Jinlong Yang; Hao Yang
>
> **备注:** under preview
>
> **摘要:** Large language models (LLMs) have ushered in a new era for document-level machine translation (\textit{doc}-mt), yet their whole-document outputs challenge existing evaluation methods that assume sentence-by-sentence alignment. We introduce \textit{\textbf{Align-then-Slide}}, a complete evaluation framework for ultra-long doc-mt. In the Align stage, we automatically infer sentence-level source-target correspondences and rebuild the target to match the source sentence number, resolving omissions and many-to-one/one-to-many mappings. In the n-Chunk Sliding Evaluate stage, we calculate averaged metric scores under 1-, 2-, 3- and 4-chunk for multi-granularity assessment. Experiments on the WMT benchmark show a Pearson correlation of 0.929 between our method with expert MQM rankings. On a newly curated real-world test set, our method again aligns closely with human judgments. Furthermore, preference data produced by Align-then-Slide enables effective CPO training and its direct use as a reward model for GRPO, both yielding translations preferred over a vanilla SFT baseline. The results validate our framework as an accurate, robust, and actionable evaluation tool for doc-mt systems.
>
---
#### [new 050] SiLVERScore: Semantically-Aware Embeddings for Sign Language Generation Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出SiLVERScore，用于手语生成评估，解决现有回译方法无法捕捉多模态特征及错误来源混淆的问题，通过语义嵌入实现联合空间评估，显著提升判别性能。**

- **链接: [http://arxiv.org/pdf/2509.03791v1](http://arxiv.org/pdf/2509.03791v1)**

> **作者:** Saki Imai; Mert İnan; Anthony Sicilia; Malihe Alikhani
>
> **摘要:** Evaluating sign language generation is often done through back-translation, where generated signs are first recognized back to text and then compared to a reference using text-based metrics. However, this two-step evaluation pipeline introduces ambiguity: it not only fails to capture the multimodal nature of sign language-such as facial expressions, spatial grammar, and prosody-but also makes it hard to pinpoint whether evaluation errors come from sign generation model or the translation system used to assess it. In this work, we propose SiLVERScore, a novel semantically-aware embedding-based evaluation metric that assesses sign language generation in a joint embedding space. Our contributions include: (1) identifying limitations of existing metrics, (2) introducing SiLVERScore for semantically-aware evaluation, (3) demonstrating its robustness to semantic and prosodic variations, and (4) exploring generalization challenges across datasets. On PHOENIX-14T and CSL-Daily datasets, SiLVERScore achieves near-perfect discrimination between correct and random pairs (ROC AUC = 0.99, overlap < 7%), substantially outperforming traditional metrics.
>
---
#### [new 051] The Telephone Game: Evaluating Semantic Drift in Unified Models
- **分类: cs.CV; cs.CL**

- **简介: 论文提出UCF-UM框架，通过循环I2T/T2I评估统一模型的跨模态一致性，解决现有单次评估无法检测语义漂移的问题，引入MCD、SDR等指标及ND400基准，揭示模型在跨模态转换中的稳定性差异。**

- **链接: [http://arxiv.org/pdf/2509.04438v1](http://arxiv.org/pdf/2509.04438v1)**

> **作者:** Sabbir Mollah; Rohit Gupta; Sirnam Swetha; Qingyang Liu; Ahnaf Munir; Mubarak Shah
>
> **摘要:** Employing a single, unified model (UM) for both visual understanding (image-to-text: I2T) and and visual generation (text-to-image: T2I) has opened a new direction in Visual Language Model (VLM) research. While UMs can also support broader unimodal tasks (e.g., text-to-text, image-to-image), we focus on the core cross-modal pair T2I and I2T, as consistency between understanding and generation is critical for downstream use. Existing evaluations consider these capabilities in isolation: FID and GenEval for T2I, and benchmarks such as MME, MMBench for I2T. These single-pass metrics do not reveal whether a model that understands a concept can also render it, nor whether meaning is preserved when cycling between image and text modalities. To address this, we introduce the Unified Consistency Framework for Unified Models (UCF-UM), a cyclic evaluation protocol that alternates I2T and T2I over multiple generations to quantify semantic drift. UCF formulates 3 metrics: (i) Mean Cumulative Drift (MCD), an embedding-based measure of overall semantic loss; (ii) Semantic Drift Rate (SDR), that summarizes semantic decay rate; and (iii) Multi-Generation GenEval (MGG), an object-level compliance score extending GenEval. To assess generalization beyond COCO, which is widely used in training; we create a new benchmark ND400, sampled from NoCaps and DOCCI and evaluate on seven recent models. UCF-UM reveals substantial variation in cross-modal stability: some models like BAGEL maintain semantics over many alternations, whereas others like Vila-u drift quickly despite strong single-pass scores. Our results highlight cyclic consistency as a necessary complement to standard I2T and T2I evaluations, and provide practical metrics to consistently assess unified model's cross-modal stability and strength of their shared representations. Code: https://github.com/mollahsabbir/Semantic-Drift-in-Unified-Models
>
---
#### [new 052] CausalARC: Abstract Reasoning with Causal World Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出CausalARC框架，用于低数据和分布外场景的抽象推理任务。通过因果世界模型与数据增强，实现少样本学习，评估语言模型在抽象推理、反事实推理、程序合成和因果发现等四类任务中的表现。**

- **链接: [http://arxiv.org/pdf/2509.03636v1](http://arxiv.org/pdf/2509.03636v1)**

> **作者:** Jacqueline Maasch; John Kalantari; Kia Khezeli
>
> **摘要:** Reasoning requires adaptation to novel problem settings under limited data and distribution shift. This work introduces CausalARC: an experimental testbed for AI reasoning in low-data and out-of-distribution regimes, modeled after the Abstraction and Reasoning Corpus (ARC). Each CausalARC reasoning task is sampled from a fully specified causal world model, formally expressed as a structural causal model. Principled data augmentations provide observational, interventional, and counterfactual feedback about the world model in the form of few-shot, in-context learning demonstrations. As a proof-of-concept, we illustrate the use of CausalARC for four language model evaluation settings: (1) abstract reasoning with test-time training, (2) counterfactual reasoning with in-context learning, (3) program synthesis, and (4) causal discovery with logical reasoning.
>
---
#### [new 053] Promptception: How Sensitive Are Large Multimodal Models to Prompts?
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究大模型对提示的敏感性，提出Promptception框架，评估10个模型在MCQA任务中的表现，分析提示变化对准确率的影响，并提出针对性的提示原则以提升评估公平性。**

- **链接: [http://arxiv.org/pdf/2509.03986v1](http://arxiv.org/pdf/2509.03986v1)**

> **作者:** Mohamed Insaf Ismithdeen; Muhammad Uzair Khattak; Salman Khan
>
> **备注:** Accepted to EMNLP 2025
>
> **摘要:** Despite the success of Large Multimodal Models (LMMs) in recent years, prompt design for LMMs in Multiple-Choice Question Answering (MCQA) remains poorly understood. We show that even minor variations in prompt phrasing and structure can lead to accuracy deviations of up to 15% for certain prompts and models. This variability poses a challenge for transparent and fair LMM evaluation, as models often report their best-case performance using carefully selected prompts. To address this, we introduce Promptception, a systematic framework for evaluating prompt sensitivity in LMMs. It consists of 61 prompt types, spanning 15 categories and 6 supercategories, each targeting specific aspects of prompt formulation, and is used to evaluate 10 LMMs ranging from lightweight open-source models to GPT-4o and Gemini 1.5 Pro, across 3 MCQA benchmarks: MMStar, MMMU-Pro, MVBench. Our findings reveal that proprietary models exhibit greater sensitivity to prompt phrasing, reflecting tighter alignment with instruction semantics, while open-source models are steadier but struggle with nuanced and complex phrasing. Based on this analysis, we propose Prompting Principles tailored to proprietary and open-source LMMs, enabling more robust and fair model evaluation.
>
---
#### [new 054] Singular Value Few-shot Adaptation of Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 论文提出CLIP-SVD，通过SVD优化CLIP参数，实现高效少样本领域适应，提升分类性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.03740v1](http://arxiv.org/pdf/2509.03740v1)**

> **作者:** Taha Koleilat; Hassan Rivaz; Yiming Xiao
>
> **备注:** 10 pages, 2 figures, 8 tables
>
> **摘要:** Vision-language models (VLMs) like CLIP have shown impressive zero-shot and few-shot learning capabilities across diverse applications. However, adapting these models to new fine-grained domains remains difficult due to reliance on prompt engineering and the high cost of full model fine-tuning. Existing adaptation approaches rely on augmented components, such as prompt tokens and adapter modules, which could limit adaptation quality, destabilize the model, and compromise the rich knowledge learned during pretraining. In this work, we present \textbf{CLIP-SVD}, a novel \textit{multi-modal} and \textit{parameter-efficient} adaptation technique that leverages Singular Value Decomposition (SVD) to modify the internal parameter space of CLIP without injecting additional modules. Specifically, we fine-tune only the singular values of the CLIP parameter matrices to rescale the basis vectors for domain adaptation while retaining the pretrained model. This design enables enhanced adaptation performance using only \textbf{0.04\%} of the model's total parameters and better preservation of its generalization ability. CLIP-SVD achieves state-of-the-art classification results on 11 natural and 10 biomedical datasets, outperforming previous methods in both accuracy and generalization under few-shot settings. Additionally, we leverage a natural language-based approach to analyze the effectiveness and dynamics of the CLIP adaptation to allow interpretability of CLIP-SVD. The code is publicly available at https://github.com/HealthX-Lab/CLIP-SVD.
>
---
#### [new 055] Emergent Hierarchical Reasoning in LLMs through Reinforcement Learning
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究如何通过强化学习提升大语言模型的推理能力。针对现有RL算法效率低的问题，提出HICRA算法，聚焦高层战略规划优化，验证语义熵作为探索指标的有效性，显著提升模型推理性能。**

- **链接: [http://arxiv.org/pdf/2509.03646v1](http://arxiv.org/pdf/2509.03646v1)**

> **作者:** Haozhe Wang; Qixin Xu; Che Liu; Junhong Wu; Fangzhen Lin; Wenhu Chen
>
> **备注:** Preprint
>
> **摘要:** Reinforcement Learning (RL) has proven highly effective at enhancing the complex reasoning abilities of Large Language Models (LLMs), yet underlying mechanisms driving this success remain largely opaque. Our analysis reveals that puzzling phenomena like ``aha moments", ``length-scaling'' and entropy dynamics are not disparate occurrences but hallmarks of an emergent reasoning hierarchy, akin to the separation of high-level strategic planning from low-level procedural execution in human cognition. We uncover a compelling two-phase dynamic: initially, a model is constrained by procedural correctness and must improve its low-level skills. The learning bottleneck then decisively shifts, with performance gains being driven by the exploration and mastery of high-level strategic planning. This insight exposes a core inefficiency in prevailing RL algorithms like GRPO, which apply optimization pressure agnostically and dilute the learning signal across all tokens. To address this, we propose HIerarchy-Aware Credit Assignment (HICRA), an algorithm that concentrates optimization efforts on high-impact planning tokens. HICRA significantly outperforms strong baselines, demonstrating that focusing on this strategic bottleneck is key to unlocking advanced reasoning. Furthermore, we validate semantic entropy as a superior compass for measuring strategic exploration over misleading metrics such as token-level entropy.
>
---
#### [new 056] Delta Activations: A Representation for Finetuned Large Language Models
- **分类: cs.LG; cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出Delta Activations方法，通过测量微调模型与基础模型的激活差异，生成向量嵌入以解决模型管理难题。旨在实现跨任务/领域的有效聚类与任务嵌入，促进模型复用。**

- **链接: [http://arxiv.org/pdf/2509.04442v1](http://arxiv.org/pdf/2509.04442v1)**

> **作者:** Zhiqiu Xu; Amish Sethi; Mayur Naik; Ser-Nam Lim
>
> **摘要:** The success of powerful open source Large Language Models (LLMs) has enabled the community to create a vast collection of post-trained models adapted to specific tasks and domains. However, navigating and understanding these models remains challenging due to inconsistent metadata and unstructured repositories. We introduce Delta Activations, a method to represent finetuned models as vector embeddings by measuring shifts in their internal activations relative to a base model. This representation allows for effective clustering by domain and task, revealing structure in the model landscape. Delta Activations also demonstrate desirable properties: it is robust across finetuning settings and exhibits an additive property when finetuning datasets are mixed. In addition, we show that Delta Activations can embed tasks via few-shot finetuning, and further explore its use for model selection and merging. We hope Delta Activations can facilitate the practice of reusing publicly available models. Code is available at https://github.com/OscarXZQ/delta_activations.
>
---
#### [new 057] CoT-Space: A Theoretical Framework for Internal Slow-Thinking via Reinforcement Learning
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出CoT-Space框架，解决传统RL框架无法处理多步推理的问题，通过连续语义空间优化分析噪声与风险，证明收敛到最优CoT长度的理论，并实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.04027v1](http://arxiv.org/pdf/2509.04027v1)**

> **作者:** Zeyu Gan; Hao Yi; Yong Liu
>
> **备注:** Preprint Edition
>
> **摘要:** Reinforcement Learning (RL) has become a pivotal approach for enhancing the reasoning capabilities of Large Language Models (LLMs). However, a significant theoretical gap persists, as traditional token-level RL frameworks fail to align with the reasoning-level nature of complex, multi-step thought processes like Chain-of-Thought (CoT). To address this challenge, we introduce CoT-Space, a novel theoretical framework that recasts LLM reasoning from a discrete token-prediction task to an optimization process within a continuous, reasoning-level semantic space. By analyzing this process from both a noise perspective and a risk perspective, we demonstrate that the convergence to an optimal CoT length is a natural consequence of the fundamental trade-off between underfitting and overfitting. Furthermore, extensive experiments provide strong empirical validation for our theoretical findings. Our framework not only provides a coherent explanation for empirical phenomena such as overthinking but also offers a solid theoretical foundation to guide the future development of more effective and generalizable reasoning agents.
>
---
#### [new 058] Contextualized Token Discrimination for Speech Search Query Correction
- **分类: cs.SD; cs.CL**

- **简介: 该论文提出CTD方法，用于语音搜索中的查询拼写纠正。通过BERT生成上下文表示，结合组合层增强语义，对比原始与上下文表示纠正错误，并构建新基准数据集。**

- **链接: [http://arxiv.org/pdf/2509.04393v1](http://arxiv.org/pdf/2509.04393v1)**

> **作者:** Junyu Lu; Di Jiang; Mengze Hong; Victor Junqiu Wei; Qintian Guo; Zhiyang Su
>
> **摘要:** Query spelling correction is an important function of modern search engines since it effectively helps users express their intentions clearly. With the growing popularity of speech search driven by Automated Speech Recognition (ASR) systems, this paper introduces a novel method named Contextualized Token Discrimination (CTD) to conduct effective speech query correction. In CTD, we first employ BERT to generate token-level contextualized representations and then construct a composition layer to enhance semantic information. Finally, we produce the correct query according to the aggregated token representation, correcting the incorrect tokens by comparing the original token representations and the contextualized representations. Extensive experiments demonstrate the superior performance of our proposed method across all metrics, and we further present a new benchmark dataset with erroneous ASR transcriptions to offer comprehensive evaluations for audio query correction.
>
---
#### [new 059] Self-adaptive Dataset Construction for Real-World Multimodal Safety Scenarios
- **分类: cs.CV; cs.CL; cs.CR**

- **简介: 论文提出图像导向的自适应数据集构建方法，解决现有方法无法覆盖复杂多模态安全场景及缺乏评估指标的问题，生成35k图像-文本对数据集并设计标准化评估指标，验证其有效性。**

- **链接: [http://arxiv.org/pdf/2509.04403v1](http://arxiv.org/pdf/2509.04403v1)**

> **作者:** Jingen Qu; Lijun Li; Bo Zhang; Yichen Yan; Jing Shao
>
> **备注:** Accepted at EMNLP 2025 Findings
>
> **摘要:** Multimodal large language models (MLLMs) are rapidly evolving, presenting increasingly complex safety challenges. However, current dataset construction methods, which are risk-oriented, fail to cover the growing complexity of real-world multimodal safety scenarios (RMS). And due to the lack of a unified evaluation metric, their overall effectiveness remains unproven. This paper introduces a novel image-oriented self-adaptive dataset construction method for RMS, which starts with images and end constructing paired text and guidance responses. Using the image-oriented method, we automatically generate an RMS dataset comprising 35k image-text pairs with guidance responses. Additionally, we introduce a standardized safety dataset evaluation metric: fine-tuning a safety judge model and evaluating its capabilities on other safety datasets.Extensive experiments on various tasks demonstrate the effectiveness of the proposed image-oriented pipeline. The results confirm the scalability and effectiveness of the image-oriented approach, offering a new perspective for the construction of real-world multimodal safety datasets.
>
---
#### [new 060] The Personality Illusion: Revealing Dissociation Between Self-Reports & Behavior in LLMs
- **分类: cs.AI; cs.CL; cs.CY; cs.LG; stat.ML**

- **简介: 该论文研究LLM性格特征，揭示自我报告与行为的脱节。通过分析训练阶段特质演变、自我报告预测有效性及干预措施影响，发现指令对齐稳定特质表达，但自我报告无法可靠预测行为，挑战LLM人格假设。**

- **链接: [http://arxiv.org/pdf/2509.03730v1](http://arxiv.org/pdf/2509.03730v1)**

> **作者:** Pengrui Han; Rafal Kocielnik; Peiyang Song; Ramit Debnath; Dean Mobbs; Anima Anandkumar; R. Michael Alvarez
>
> **备注:** We make public all code and source data at https://github.com/psychology-of-AI/Personality-Illusion
>
> **摘要:** Personality traits have long been studied as predictors of human behavior.Recent advances in Large Language Models (LLMs) suggest similar patterns may emerge in artificial systems, with advanced LLMs displaying consistent behavioral tendencies resembling human traits like agreeableness and self-regulation. Understanding these patterns is crucial, yet prior work primarily relied on simplified self-reports and heuristic prompting, with little behavioral validation. In this study, we systematically characterize LLM personality across three dimensions: (1) the dynamic emergence and evolution of trait profiles throughout training stages; (2) the predictive validity of self-reported traits in behavioral tasks; and (3) the impact of targeted interventions, such as persona injection, on both self-reports and behavior. Our findings reveal that instructional alignment (e.g., RLHF, instruction tuning) significantly stabilizes trait expression and strengthens trait correlations in ways that mirror human data. However, these self-reported traits do not reliably predict behavior, and observed associations often diverge from human patterns. While persona injection successfully steers self-reports in the intended direction, it exerts little or inconsistent effect on actual behavior. By distinguishing surface-level trait expression from behavioral consistency, our findings challenge assumptions about LLM personality and underscore the need for deeper evaluation in alignment and interpretability.
>
---
#### [new 061] Crossing the Species Divide: Transfer Learning from Speech to Animal Sounds
- **分类: cs.LG; cs.AI; cs.CL; cs.SD; 68T07; I.5.4; I.2.6; H.5.5**

- **简介: 该论文研究自监督语音模型（如HuBERT、WavLM）在生物声学检测分类中的迁移学习效果，分析时间信息及噪声影响，结果与微调模型相当，证明其在生物声学研究中的潜力。（99字）**

- **链接: [http://arxiv.org/pdf/2509.04166v1](http://arxiv.org/pdf/2509.04166v1)**

> **作者:** Jules Cauzinille; Marius Miron; Olivier Pietquin; Masato Hagiwara; Ricard Marxer; Arnaud Rey; Benoit Favre
>
> **备注:** 5 pages, 3 figures, uses dcase2025.sty, submitted to DCASE 2025
>
> **摘要:** Self-supervised speech models have demonstrated impressive performance in speech processing, but their effectiveness on non-speech data remains underexplored. We study the transfer learning capabilities of such models on bioacoustic detection and classification tasks. We show that models such as HuBERT, WavLM, and XEUS can generate rich latent representations of animal sounds across taxa. We analyze the models properties with linear probing on time-averaged representations. We then extend the approach to account for the effect of time-wise information with other downstream architectures. Finally, we study the implication of frequency range and noise on performance. Notably, our results are competitive with fine-tuned bioacoustic pre-trained models and show the impact of noise-robust pre-training setups. These findings highlight the potential of speech-based self-supervised learning as an efficient framework for advancing bioacoustic research.
>
---
#### [new 062] NER Retriever: Zero-Shot Named Entity Retrieval with Type-Aware Embeddings
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文提出NER Retriever，解决零样本命名实体检索问题。通过利用LLM中间层表示和对比学习，构建类型感知的实体嵌入，实现无需预定义类型的高效实体检索，在多个基准上超越传统方法。**

- **链接: [http://arxiv.org/pdf/2509.04011v1](http://arxiv.org/pdf/2509.04011v1)**

> **作者:** Or Shachar; Uri Katz; Yoav Goldberg; Oren Glickman
>
> **备注:** Findings of EMNLP 2025
>
> **摘要:** We present NER Retriever, a zero-shot retrieval framework for ad-hoc Named Entity Retrieval, a variant of Named Entity Recognition (NER), where the types of interest are not provided in advance, and a user-defined type description is used to retrieve documents mentioning entities of that type. Instead of relying on fixed schemas or fine-tuned models, our method builds on internal representations of large language models (LLMs) to embed both entity mentions and user-provided open-ended type descriptions into a shared semantic space. We show that internal representations, specifically the value vectors from mid-layer transformer blocks, encode fine-grained type information more effectively than commonly used top-layer embeddings. To refine these representations, we train a lightweight contrastive projection network that aligns type-compatible entities while separating unrelated types. The resulting entity embeddings are compact, type-aware, and well-suited for nearest-neighbor search. Evaluated on three benchmarks, NER Retriever significantly outperforms both lexical and dense sentence-level retrieval baselines. Our findings provide empirical support for representation selection within LLMs and demonstrate a practical solution for scalable, schema-free entity retrieval. The NER Retriever Codebase is publicly available at https://github.com/ShacharOr100/ner_retriever
>
---
#### [new 063] LibriQuote: A Speech Dataset of Fictional Character Utterances for Expressive Zero-Shot Speech Synthesis
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文提出LibriQuote数据集，用于零样本语音合成任务，解决现有数据集规模小、表达性语音比例不明的问题。数据集包含12.7K小时非表达性语音及5.3K小时角色引用表达性语音，设计测试集评估系统生成表达性语音能力，并验证微调后系统效果提升。**

- **链接: [http://arxiv.org/pdf/2509.04072v1](http://arxiv.org/pdf/2509.04072v1)**

> **作者:** Gaspard Michel; Elena V. Epure; Christophe Cerisara
>
> **摘要:** Text-to-speech (TTS) systems have recently achieved more expressive and natural speech synthesis by scaling to large speech datasets. However, the proportion of expressive speech in such large-scale corpora is often unclear. Besides, existing expressive speech corpora are typically smaller in scale and primarily used for benchmarking TTS systems. In this paper, we introduce the LibriQuote dataset, an English corpus derived from read audiobooks, designed for both fine-tuning and benchmarking expressive zero-shot TTS system. The training dataset includes 12.7K hours of read, non-expressive speech and 5.3K hours of mostly expressive speech drawn from character quotations. Each utterance in the expressive subset is supplemented with the context in which it was written, along with pseudo-labels of speech verbs and adverbs used to describe the quotation (\textit{e.g. ``he whispered softly''}). Additionally, we provide a challenging 7.5 hour test set intended for benchmarking TTS systems: given a neutral reference speech as input, we evaluate system's ability to synthesize an expressive utterance while preserving reference timbre. We validate qualitatively the test set by showing that it covers a wide range of emotions compared to non-expressive speech, along with various accents. Extensive subjective and objective evaluations show that fine-tuning a baseline TTS system on LibriQuote significantly improves its synthesized speech intelligibility, and that recent systems fail to synthesize speech as expressive and natural as the ground-truth utterances. The dataset and evaluation code are freely available. Audio samples can be found at https://libriquote.github.io/.
>
---
#### [new 064] Psychologically Enhanced AI Agents
- **分类: cs.AI; cs.CL; cs.CY; cs.HC; cs.MA**

- **简介: 论文提出MBTI-in-Thoughts框架，通过心理人格调用增强AI代理行为，提升任务表现。解决AI代理心理适配性问题，支持多智能体通信，验证方法可扩展至其他心理模型。**

- **链接: [http://arxiv.org/pdf/2509.04343v1](http://arxiv.org/pdf/2509.04343v1)**

> **作者:** Maciej Besta; Shriram Chandran; Robert Gerstenberger; Mathis Lindner; Marcin Chrapek; Sebastian Hermann Martschat; Taraneh Ghandi; Patrick Iff; Hubert Niewiadomski; Piotr Nyczyk; Jürgen Müller; Torsten Hoefler
>
> **摘要:** We introduce MBTI-in-Thoughts, a framework for enhancing the effectiveness of Large Language Model (LLM) agents through psychologically grounded personality conditioning. Drawing on the Myers-Briggs Type Indicator (MBTI), our method primes agents with distinct personality archetypes via prompt engineering, enabling control over behavior along two foundational axes of human psychology, cognition and affect. We show that such personality priming yields consistent, interpretable behavioral biases across diverse tasks: emotionally expressive agents excel in narrative generation, while analytically primed agents adopt more stable strategies in game-theoretic settings. Our framework supports experimenting with structured multi-agent communication protocols and reveals that self-reflection prior to interaction improves cooperation and reasoning quality. To ensure trait persistence, we integrate the official 16Personalities test for automated verification. While our focus is on MBTI, we show that our approach generalizes seamlessly to other psychological frameworks such as Big Five, HEXACO, or Enneagram. By bridging psychological theory and LLM behavior design, we establish a foundation for psychologically enhanced AI agents without any fine-tuning.
>
---
#### [new 065] No Thoughts Just AI: Biased LLM Recommendations Limit Human Agency in Resume Screening
- **分类: cs.CY; cs.AI; cs.CL; cs.HC; K.4.2**

- **简介: 该研究通过简历筛选实验，探讨AI种族偏见对人类决策的影响。发现人类在AI偏见下倾向跟随其推荐，隐性偏见测试显示决策受种族刻板印象影响。结果强调AI-HITL场景中人类自主性的局限，呼吁政策制定者关注系统设计与监管。**

- **链接: [http://arxiv.org/pdf/2509.04404v1](http://arxiv.org/pdf/2509.04404v1)**

> **作者:** Kyra Wilson; Mattea Sim; Anna-Maria Gueorguieva; Aylin Caliskan
>
> **备注:** Published in Proceedings of the 2025 AAAI/ACM Conference on AI, Ethics, and Society; code available at https://github.com/kyrawilson/No-Thoughts-Just-AI
>
> **摘要:** In this study, we conduct a resume-screening experiment (N=528) where people collaborate with simulated AI models exhibiting race-based preferences (bias) to evaluate candidates for 16 high and low status occupations. Simulated AI bias approximates factual and counterfactual estimates of racial bias in real-world AI systems. We investigate people's preferences for White, Black, Hispanic, and Asian candidates (represented through names and affinity groups on quality-controlled resumes) across 1,526 scenarios and measure their unconscious associations between race and status using implicit association tests (IATs), which predict discriminatory hiring decisions but have not been investigated in human-AI collaboration. When making decisions without AI or with AI that exhibits no race-based preferences, people select all candidates at equal rates. However, when interacting with AI favoring a particular group, people also favor those candidates up to 90% of the time, indicating a significant behavioral shift. The likelihood of selecting candidates whose identities do not align with common race-status stereotypes can increase by 13% if people complete an IAT before conducting resume screening. Finally, even if people think AI recommendations are low quality or not important, their decisions are still vulnerable to AI bias under certain circumstances. This work has implications for people's autonomy in AI-HITL scenarios, AI and work, design and evaluation of AI hiring systems, and strategies for mitigating bias in collaborative decision-making tasks. In particular, organizational and regulatory policy should acknowledge the complex nature of AI-HITL decision making when implementing these systems, educating people who use them, and determining which are subject to oversight.
>
---
#### [new 066] Towards an Action-Centric Ontology for Cooking Procedures Using Temporal Graphs
- **分类: cs.AI; cs.CL**

- **简介: 论文提出一种基于时间图的领域特定语言，用于形式化烹饪流程，解决其复杂性和模糊性，通过动作图建模并评估其在自动化烹饪中的潜力。**

- **链接: [http://arxiv.org/pdf/2509.04159v1](http://arxiv.org/pdf/2509.04159v1)**

> **作者:** Aarush Kumbhakern; Saransh Kumar Gupta; Lipika Dey; Partha Pratim Das
>
> **备注:** 6 pages, 3 figures, 1 table, 11 references, ACM International Conference on Multimedia 2025 - Multi-modal Food Computing Workshop
>
> **摘要:** Formalizing cooking procedures remains a challenging task due to their inherent complexity and ambiguity. We introduce an extensible domain-specific language for representing recipes as directed action graphs, capturing processes, transfers, environments, concurrency, and compositional structure. Our approach enables precise, modular modeling of complex culinary workflows. Initial manual evaluation on a full English breakfast recipe demonstrates the DSL's expressiveness and suitability for future automated recipe analysis and execution. This work represents initial steps towards an action-centric ontology for cooking, using temporal graphs to enable structured machine understanding, precise interpretation, and scalable automation of culinary processes - both in home kitchens and professional culinary settings.
>
---
#### [new 067] ArcMemo: Abstract Reasoning Composition with Lifelong LLM Memory
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 论文提出ArcMemo，通过概念级长期记忆解决LLM推理中上下文重置导致的信息丢失问题，实现持续学习。方法包括抽象总结和动态检索，提升ARC-AGI基准性能7.5%。**

- **链接: [http://arxiv.org/pdf/2509.04439v1](http://arxiv.org/pdf/2509.04439v1)**

> **作者:** Matthew Ho; Chen Si; Zhaoxiang Feng; Fangxu Yu; Zhijian Liu; Zhiting Hu; Lianhui Qin
>
> **摘要:** While inference-time scaling enables LLMs to carry out increasingly long and capable reasoning traces, the patterns and insights uncovered during these traces are immediately discarded once the context window is reset for a new query. External memory is a natural way to persist these discoveries, and recent work has shown clear benefits for reasoning-intensive tasks. We see an opportunity to make such memories more broadly reusable and scalable by moving beyond instance-based memory entries (e.g. exact query/response pairs, or summaries tightly coupled with the original problem context) toward concept-level memory: reusable, modular abstractions distilled from solution traces and stored in natural language. For future queries, relevant concepts are selectively retrieved and integrated into the prompt, enabling test-time continual learning without weight updates. Our design introduces new strategies for abstracting takeaways from rollouts and retrieving entries for new queries, promoting reuse and allowing memory to expand with additional experiences. On the challenging ARC-AGI benchmark, our method yields a 7.5% relative gain over a strong no-memory baseline with performance continuing to scale with inference compute. We find abstract concepts to be the most consistent memory design, outscoring the baseline at all tested inference compute scales. Moreover, we confirm that dynamically updating memory during test-time outperforms an otherwise identical fixed memory setting with additional attempts, supporting the hypothesis that solving more problems and abstracting more patterns to memory enables further solutions in a form of self-improvement. Code available at https://github.com/matt-seb-ho/arc_memo.
>
---
#### [new 068] SPECS: Specificity-Enhanced CLIP-Score for Long Image Caption Evaluation
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出SPECS，用于长图像描述生成评估，解决现有指标效率低、语义捕捉不足问题。通过改进CLIP增强具体性，实现高效且与人类判断高度相关。**

- **链接: [http://arxiv.org/pdf/2509.03897v1](http://arxiv.org/pdf/2509.03897v1)**

> **作者:** Xiaofu Chen; Israfel Salazar; Yova Kementchedjhieva
>
> **摘要:** As interest grows in generating long, detailed image captions, standard evaluation metrics become increasingly unreliable. N-gram-based metrics though efficient, fail to capture semantic correctness. Representational Similarity (RS) metrics, designed to address this, initially saw limited use due to high computational costs, while today, despite advances in hardware, they remain unpopular due to low correlation to human judgments. Meanwhile, metrics based on large language models (LLMs) show strong correlation with human judgments, but remain too expensive for iterative use during model development. We introduce SPECS (Specificity-Enhanced CLIPScore), a reference-free RS metric tailored to long image captioning. SPECS modifies CLIP with a new objective that emphasizes specificity: rewarding correct details and penalizing incorrect ones. We show that SPECS matches the performance of open-source LLM-based metrics in correlation to human judgments, while being far more efficient. This makes it a practical alternative for iterative checkpoint evaluation during image captioning model development.Our code can be found at https://github.com/mbzuai-nlp/SPECS.
>
---
#### [new 069] Towards a Unified View of Large Language Model Post-Training
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出统一框架，将强化学习与监督微调视为同一优化过程，推导通用梯度估计器，设计HPT算法动态融合在线/离线数据，提升模型推理稳定性与性能。**

- **链接: [http://arxiv.org/pdf/2509.04419v1](http://arxiv.org/pdf/2509.04419v1)**

> **作者:** Xingtai Lv; Yuxin Zuo; Youbang Sun; Hongyi Liu; Yuntian Wei; Zhekai Chen; Lixuan He; Xuekai Zhu; Kaiyan Zhang; Bingning Wang; Ning Ding; Bowen Zhou
>
> **摘要:** Two major sources of training data exist for post-training modern language models: online (model-generated rollouts) data, and offline (human or other-model demonstrations) data. These two types of data are typically used by approaches like Reinforcement Learning (RL) and Supervised Fine-Tuning (SFT), respectively. In this paper, we show that these approaches are not in contradiction, but are instances of a single optimization process. We derive a Unified Policy Gradient Estimator, and present the calculations of a wide spectrum of post-training approaches as the gradient of a common objective under different data distribution assumptions and various bias-variance tradeoffs. The gradient estimator is constructed with four interchangeable parts: stabilization mask, reference policy denominator, advantage estimate, and likelihood gradient. Motivated by our theoretical findings, we propose Hybrid Post-Training (HPT), an algorithm that dynamically selects different training signals. HPT is designed to yield both effective exploitation of demonstration and stable exploration without sacrificing learned reasoning patterns. We provide extensive experiments and ablation studies to verify the effectiveness of our unified theoretical framework and HPT. Across six mathematical reasoning benchmarks and two out-of-distribution suites, HPT consistently surpasses strong baselines across models of varying scales and families.
>
---
#### [new 070] Towards a Neurosymbolic Reasoning System Grounded in Schematic Representations
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Embodied-LM系统，结合神经网络与符号推理，利用图像图式和空间推理提升LLM的逻辑推理能力，解决其缺乏稳健认知结构的问题。**

- **链接: [http://arxiv.org/pdf/2509.03644v1](http://arxiv.org/pdf/2509.03644v1)**

> **作者:** François Olivier; Zied Bouraoui
>
> **备注:** To appear in Proceedings of Machine Learning Research, 19th Conference on Neurosymbolic Learning and Reasoning, 2025
>
> **摘要:** Despite significant progress in natural language understanding, Large Language Models (LLMs) remain error-prone when performing logical reasoning, often lacking the robust mental representations that enable human-like comprehension. We introduce a prototype neurosymbolic system, Embodied-LM, that grounds understanding and logical reasoning in schematic representations based on image schemas-recurring patterns derived from sensorimotor experience that structure human cognition. Our system operationalizes the spatial foundations of these cognitive structures using declarative spatial reasoning within Answer Set Programming. Through evaluation on logical deduction problems, we demonstrate that LLMs can be guided to interpret scenarios through embodied cognitive structures, that these structures can be formalized as executable programs, and that the resulting representations support effective logical reasoning with enhanced interpretability. While our current implementation focuses on spatial primitives, it establishes the computational foundation for incorporating more complex and dynamic representations.
>
---
#### [new 071] Evaluating the Robustness of Retrieval-Augmented Generation to Adversarial Evidence in the Health Domain
- **分类: cs.IR; cs.CL**

- **简介: 该论文评估RAG系统在健康领域对抗性证据下的鲁棒性，研究不同文档类型和问题框架对模型输出的影响，发现对抗性文档降低准确性，但有益证据可提升鲁棒性，为设计安全RAG系统提供指导。**

- **链接: [http://arxiv.org/pdf/2509.03787v1](http://arxiv.org/pdf/2509.03787v1)**

> **作者:** Shakiba Amirshahi; Amin Bigdeli; Charles L. A. Clarke; Amira Ghenai
>
> **摘要:** Retrieval augmented generation (RAG) systems provide a method for factually grounding the responses of a Large Language Model (LLM) by providing retrieved evidence, or context, as support. Guided by this context, RAG systems can reduce hallucinations and expand the ability of LLMs to accurately answer questions outside the scope of their training data. Unfortunately, this design introduces a critical vulnerability: LLMs may absorb and reproduce misinformation present in retrieved evidence. This problem is magnified if retrieved evidence contains adversarial material explicitly intended to promulgate misinformation. This paper presents a systematic evaluation of RAG robustness in the health domain and examines alignment between model outputs and ground-truth answers. We focus on the health domain due to the potential for harm caused by incorrect responses, as well as the availability of evidence-based ground truth for many common health-related questions. We conduct controlled experiments using common health questions, varying both the type and composition of the retrieved documents (helpful, harmful, and adversarial) as well as the framing of the question by the user (consistent, neutral, and inconsistent). Our findings reveal that adversarial documents substantially degrade alignment, but robustness can be preserved when helpful evidence is also present in the retrieval pool. These findings offer actionable insights for designing safer RAG systems in high-stakes domains by highlighting the need for retrieval safeguards. To enable reproducibility and facilitate future research, all experimental results are publicly available in our github repository. https://github.com/shakibaam/RAG_ROBUSTNESS_EVAL
>
---
## 更新

#### [replaced 001] ACING: Actor-Critic for Instruction Learning in Black-Box LLMs
- **分类: cs.CL; cs.AI; cs.LG; cs.SY; eess.SY; math.OC**

- **链接: [http://arxiv.org/pdf/2411.12736v2](http://arxiv.org/pdf/2411.12736v2)**

> **作者:** Salma Kharrat; Fares Fourati; Marco Canini
>
> **备注:** Accepted at EMNLP 2025
>
> **摘要:** The effectiveness of Large Language Models (LLMs) in solving tasks depends significantly on the quality of their instructions, which often require substantial human effort to craft. This underscores the need for automated instruction optimization. However, optimizing instructions is particularly challenging when working with black-box LLMs, where model parameters and gradients are inaccessible. We introduce ACING, an actor-critic reinforcement learning framework that formulates instruction optimization as a stateless, continuous-action problem, enabling exploration of infinite instruction spaces using only black-box feedback. ACING automatically discovers prompts that outperform human-written prompts in 76% of instruction-induction tasks, with gains of up to 33 points and a 10-point median improvement over the best automatic baseline in 33 tasks spanning instruction-induction, summarization, and chain-of-thought reasoning. Extensive ablations highlight its robustness and efficiency. An implementation of ACING is available at https://github.com/salmakh1/ACING.
>
---
#### [replaced 002] Oyster-I: Beyond Refusal -- Constructive Safety Alignment for Responsible Language Models
- **分类: cs.AI; cs.CL; cs.CY; cs.HC; cs.SC**

- **链接: [http://arxiv.org/pdf/2509.01909v2](http://arxiv.org/pdf/2509.01909v2)**

> **作者:** Ranjie Duan; Jiexi Liu; Xiaojun Jia; Shiji Zhao; Ruoxi Cheng; Fengxiang Wang; Cheng Wei; Yong Xie; Chang Liu; Defeng Li; Yinpeng Dong; Yichi Zhang; Yuefeng Chen; Chongwen Wang; Xingjun Ma; Xingxing Wei; Yang Liu; Hang Su; Jun Zhu; Xinfeng Li; Yitong Sun; Jie Zhang; Jinzhao Hu; Sha Xu; Yitong Yang; Jialing Tao; Hui Xue
>
> **备注:** Technical Report Code & Model weights available: https://github.com/Alibaba-AAIG/Oyster
>
> **摘要:** Large language models (LLMs) typically deploy safety mechanisms to prevent harmful content generation. Most current approaches focus narrowly on risks posed by malicious actors, often framing risks as adversarial events and relying on defensive refusals. However, in real-world settings, risks also come from non-malicious users seeking help while under psychological distress (e.g., self-harm intentions). In such cases, the model's response can strongly influence the user's next actions. Simple refusals may lead them to repeat, escalate, or move to unsafe platforms, creating worse outcomes. We introduce Constructive Safety Alignment (CSA), a human-centric paradigm that protects against malicious misuse while actively guiding vulnerable users toward safe and helpful results. Implemented in Oyster-I (Oy1), CSA combines game-theoretic anticipation of user reactions, fine-grained risk boundary discovery, and interpretable reasoning control, turning safety into a trust-building process. Oy1 achieves state-of-the-art safety among open models while retaining high general capabilities. On our Constructive Benchmark, it shows strong constructive engagement, close to GPT-5, and unmatched robustness on the Strata-Sword jailbreak dataset, nearing GPT-o1 levels. By shifting from refusal-first to guidance-first safety, CSA redefines the model-user relationship, aiming for systems that are not just safe, but meaningfully helpful. We release Oy1, code, and the benchmark to support responsible, user-centered AI.
>
---
#### [replaced 003] PIN: A Knowledge-Intensive Dataset for Paired and Interleaved Multimodal Documents
- **分类: cs.AI; cs.CL; cs.CV; cs.MM**

- **链接: [http://arxiv.org/pdf/2406.13923v2](http://arxiv.org/pdf/2406.13923v2)**

> **作者:** Junjie Wang; Yuxiang Zhang; Minghao Liu; Yin Zhang; Yatai Ji; Weihao Xuan; Nie Lin; Kang Zhu; Zhiqiang Lin; Yiming Ren; Chunyang Jiang; Yiyao Yu; Zekun Wang; Tiezhen Wang; Wenhao Huang; Jie Fu; Qunshu Liu; Yujiu Yang; Ge Zhang; Ruibin Yuan; Bei Chen; Wenhu Chen
>
> **备注:** Technical report v1.0
>
> **摘要:** Recent advancements in large multimodal models (LMMs) have leveraged extensive multimodal datasets to enhance capabilities in complex knowledge-driven tasks. However, persistent challenges in perceptual and reasoning errors limit their efficacy, particularly in interpreting intricate visual data and deducing multimodal relationships. To address these issues, we introduce PIN (Paired and INterleaved multimodal documents), a novel data format designed to foster a deeper integration of visual and textual knowledge. The PIN format uniquely combines semantically rich Markdown files, which preserve fine-grained textual structures, with holistic overall images that capture the complete document layout. Following this format, we construct and release two large-scale, open-source datasets: PIN-200M (~200 million documents) and PIN-14M (~14 million), compiled from diverse web and scientific sources in both English and Chinese. To maximize usability, we provide detailed statistical analyses and equip the datasets with quality signals, enabling researchers to easily filter and select data for specific tasks. Our work provides the community with a versatile data format and substantial resources, offering a foundation for new research in pre-training strategies and the development of more powerful knowledge-intensive LMMs.
>
---
#### [replaced 004] Mitigating Bias in Text Classification via Prompt-Based Text Transformation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2305.06166v3](http://arxiv.org/pdf/2305.06166v3)**

> **作者:** Charmaine Barker; Dimitar Kazakov
>
> **备注:** This version corrects an error in the model specification
>
> **摘要:** The presence of specific linguistic signals particular to a certain sub-group can become highly salient to language models during training. In automated decision-making settings, this may lead to biased outcomes when models rely on cues that correlate with protected characteristics. We investigate whether prompting ChatGPT to rewrite text using simplification, neutralisation, localisation, and formalisation can reduce demographic signals while preserving meaning. Experimental results show a statistically significant drop in location classification accuracy across multiple models after transformation, suggesting reduced reliance on group-specific language. At the same time, sentiment analysis and rating prediction tasks confirm that the core meaning of the reviews remains greatly intact. These results suggest that prompt-based rewriting offers a practical and generalisable approach for mitigating bias in text classification.
>
---
#### [replaced 005] How Can I Publish My LLM Benchmark Without Giving the True Answers Away?
- **分类: cs.LG; cs.AI; cs.CL; stat.ME**

- **链接: [http://arxiv.org/pdf/2505.18102v5](http://arxiv.org/pdf/2505.18102v5)**

> **作者:** Takashi Ishida; Thanawat Lodkaew; Ikko Yamane
>
> **备注:** Extended version of the paper presented as an Oral at the ICML 2025 Workshop on the Impact of Memorization on Trustworthy Foundation Models
>
> **摘要:** Publishing a large language model (LLM) benchmark on the Internet risks contaminating future LLMs: the benchmark may be unintentionally (or intentionally) used to train or select a model. A common mitigation is to keep the benchmark private and let participants submit their models or predictions to the organizers. However, this strategy will require trust in a single organization and still permits test-set overfitting through repeated queries. To overcome this issue, we propose a way to publish benchmarks without completely disclosing the ground-truth answers to the questions, while still maintaining the ability to openly evaluate LLMs. Our main idea is to inject randomness to the answers by preparing several logically correct answers, and only include one of them as the solution in the benchmark. This reduces the best possible accuracy, i.e., Bayes accuracy, of the benchmark. Not only is this helpful to keep us from disclosing the ground truth, but this approach also offers a test for detecting data contamination. In principle, even fully capable models should not surpass the Bayes accuracy. If a model surpasses this ceiling despite this expectation, this is a strong signal of data contamination. We present experimental evidence that our method can detect data contamination accurately on a wide range of benchmarks, models, and training methodologies.
>
---
#### [replaced 006] Small Changes, Large Consequences: Analyzing the Allocational Fairness of LLMs in Hiring Contexts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.04316v2](http://arxiv.org/pdf/2501.04316v2)**

> **作者:** Preethi Seshadri; Hongyu Chen; Sameer Singh; Seraphina Goldfarb-Tarrant
>
> **摘要:** Large language models (LLMs) are increasingly being deployed in high-stakes applications like hiring, yet their potential for unfair decision-making remains understudied in generative and retrieval settings. In this work, we examine the allocational fairness of LLM-based hiring systems through two tasks that reflect actual HR usage: resume summarization and applicant ranking. By constructing a synthetic resume dataset with controlled perturbations and curating job postings, we investigate whether model behavior differs across demographic groups. Our findings reveal that generated summaries exhibit meaningful differences more frequently for race than for gender perturbations. Models also display non-uniform retrieval selection patterns across demographic groups and exhibit high ranking sensitivity to both gender and race perturbations. Surprisingly, retrieval models can show comparable sensitivity to both demographic and non-demographic changes, suggesting that fairness issues may stem from broader model brittleness. Overall, our results indicate that LLM-based hiring systems, especially in the retrieval stage, can exhibit notable biases that lead to discriminatory outcomes in real-world contexts.
>
---
#### [replaced 007] MEDUSA: A Multimodal Deep Fusion Multi-Stage Training Framework for Speech Emotion Recognition in Naturalistic Conditions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.09556v2](http://arxiv.org/pdf/2506.09556v2)**

> **作者:** Georgios Chatzichristodoulou; Despoina Kosmopoulou; Antonios Kritikos; Anastasia Poulopoulou; Efthymios Georgiou; Athanasios Katsamanis; Vassilis Katsouros; Alexandros Potamianos
>
> **备注:** Interspeech 2025
>
> **摘要:** SER is a challenging task due to the subjective nature of human emotions and their uneven representation under naturalistic conditions. We propose MEDUSA, a multimodal framework with a four-stage training pipeline, which effectively handles class imbalance and emotion ambiguity. The first two stages train an ensemble of classifiers that utilize DeepSER, a novel extension of a deep cross-modal transformer fusion mechanism from pretrained self-supervised acoustic and linguistic representations. Manifold MixUp is employed for further regularization. The last two stages optimize a trainable meta-classifier that combines the ensemble predictions. Our training approach incorporates human annotation scores as soft targets, coupled with balanced data sampling and multitask learning. MEDUSA ranked 1st in Task 1: Categorical Emotion Recognition in the Interspeech 2025: Speech Emotion Recognition in Naturalistic Conditions Challenge.
>
---
#### [replaced 008] UI-Bench: A Benchmark for Evaluating Design Capabilities of AI Text-to-App Tools
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.20410v3](http://arxiv.org/pdf/2508.20410v3)**

> **作者:** Sam Jung; Agustin Garcinuno; Spencer Mateega
>
> **摘要:** AI text-to-app tools promise high quality applications and websites in minutes, yet no public benchmark rigorously verifies those claims. We introduce UI-Bench, the first large-scale benchmark that evaluates visual excellence across competing AI text-to-app tools through expert pairwise comparison. Spanning 10 tools, 30 prompts, 300 generated sites, and 4,000+ expert judgments, UI-Bench ranks systems with a TrueSkill-derived model that yields calibrated confidence intervals. UI-Bench establishes a reproducible standard for advancing AI-driven web design. We release (i) the complete prompt set, (ii) an open-source evaluation framework, and (iii) a public leaderboard. The generated sites rated by participants will be released soon. View the UI-Bench leaderboard at https://uibench.ai/leaderboard.
>
---
#### [replaced 009] Enhancing FKG.in: automating Indian food composition analysis
- **分类: cs.AI; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2412.05248v3](http://arxiv.org/pdf/2412.05248v3)**

> **作者:** Saransh Kumar Gupta; Lipika Dey; Partha Pratim Das; Geeta Trilok-Kumar; Ramesh Jain
>
> **备注:** 15 pages, 5 figures, 30 references, International Conference on Pattern Recognition 2024 - Multimedia Assisted Dietary Management Workshop
>
> **摘要:** This paper presents a novel approach to compute food composition data for Indian recipes using a knowledge graph for Indian food (FKG[.]in) and LLMs. The primary focus is to provide a broad overview of an automated food composition analysis workflow and describe its core functionalities: nutrition data aggregation, food composition analysis, and LLM-augmented information resolution. This workflow aims to complement FKG[.]in and iteratively supplement food composition data from verified knowledge bases. Additionally, this paper highlights the challenges of representing Indian food and accessing food composition data digitally. It also reviews three key sources of food composition data: the Indian Food Composition Tables, the Indian Nutrient Databank, and the Nutritionix API. Furthermore, it briefly outlines how users can interact with the workflow to obtain diet-based health recommendations and detailed food composition information for numerous recipes. We then explore the complex challenges of analyzing Indian recipe information across dimensions such as structure, multilingualism, and uncertainty as well as present our ongoing work on LLM-based solutions to address these issues. The methods proposed in this workshop paper for AI-driven knowledge curation and information resolution are application-agnostic, generalizable, and replicable for any domain.
>
---
#### [replaced 010] AgenTracer: Who Is Inducing Failure in the LLM Agentic Systems?
- **分类: cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2509.03312v2](http://arxiv.org/pdf/2509.03312v2)**

> **作者:** Guibin Zhang; Junhao Wang; Junjie Chen; Wangchunshu Zhou; Kun Wang; Shuicheng Yan
>
> **摘要:** Large Language Model (LLM)-based agentic systems, often comprising multiple models, complex tool invocations, and orchestration protocols, substantially outperform monolithic agents. Yet this very sophistication amplifies their fragility, making them more prone to system failure. Pinpointing the specific agent or step responsible for an error within long execution traces defines the task of agentic system failure attribution. Current state-of-the-art reasoning LLMs, however, remain strikingly inadequate for this challenge, with accuracy generally below 10%. To address this gap, we propose AgenTracer, the first automated framework for annotating failed multi-agent trajectories via counterfactual replay and programmed fault injection, producing the curated dataset TracerTraj. Leveraging this resource, we develop AgenTracer-8B, a lightweight failure tracer trained with multi-granular reinforcement learning, capable of efficiently diagnosing errors in verbose multi-agent interactions. On the Who&When benchmark, AgenTracer-8B outperforms giant proprietary LLMs like Gemini-2.5-Pro and Claude-4-Sonnet by up to 18.18%, setting a new standard in LLM agentic failure attribution. More importantly, AgenTracer-8B delivers actionable feedback to off-the-shelf multi-agent systems like MetaGPT and MaAS with 4.8-14.2% performance gains, empowering self-correcting and self-evolving agentic AI.
>
---
#### [replaced 011] Training LLMs to be Better Text Embedders through Bidirectional Reconstruction
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2509.03020v2](http://arxiv.org/pdf/2509.03020v2)**

> **作者:** Chang Su; Dengliang Shi; Siyuan Huang; Jintao Du; Changhua Meng; Yu Cheng; Weiqiang Wang; Zhouhan Lin
>
> **备注:** accepted by EMNLP 2025 Main Conference
>
> **摘要:** Large language models (LLMs) have increasingly been explored as powerful text embedders. Existing LLM-based text embedding approaches often leverage the embedding of the final token, typically a reserved special token such as [EOS]. However, these tokens have not been intentionally trained to capture the semantics of the whole context, limiting their capacity as text embeddings, especially for retrieval and re-ranking tasks. We propose to add a new training stage before contrastive learning to enrich the semantics of the final token embedding. This stage employs bidirectional generative reconstruction tasks, namely EBQ2D (Embedding-Based Query-to-Document) and EBD2Q (Embedding-Based Document-to-Query), which interleave to anchor the [EOS] embedding and reconstruct either side of Query-Document pairs. Experimental results demonstrate that our additional training stage significantly improves LLM performance on the Massive Text Embedding Benchmark (MTEB), achieving new state-of-the-art results across different LLM base models and scales.
>
---
#### [replaced 012] Science Across Languages: Assessing LLM Multilingual Translation of Scientific Papers
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17882v2](http://arxiv.org/pdf/2502.17882v2)**

> **作者:** Hannah Calzi Kleidermacher; James Zou
>
> **摘要:** Scientific research is inherently global. However, the vast majority of academic journals are published exclusively in English, creating barriers for non-native-English-speaking researchers. In this study, we leverage large language models (LLMs) to translate published scientific articles while preserving their native JATS XML formatting, thereby developing a practical, automated approach for implementation by academic journals. Using our approach, we translate articles across multiple scientific disciplines into 28 languages. To evaluate translation accuracy, we introduce a novel question-and-answer (QA) benchmarking method, in which an LLM generates comprehension-based questions from the original text and then answers them based on the translated text. Our benchmark results show an average performance of 95.9%, showing that the key scientific details are accurately conveyed. In a user study, we translate the scientific papers of 15 researchers into their native languages, finding that the authors consistently found the translations to accurately capture the original information in their articles. Interestingly, a third of the authors found many technical terms "overtranslated," expressing a preference to keep terminology more familiar in English untranslated. Finally, we demonstrate how in-context learning techniques can be used to align translations with domain-specific preferences such as mitigating overtranslation, highlighting the adaptability and utility of LLM-driven scientific translation. The code and translated articles are available at https://hankleid.github.io/ProjectMundo.
>
---
#### [replaced 013] Forewarned is Forearmed: Pre-Synthesizing Jailbreak-like Instructions to Enhance LLM Safety Guardrail to Potential Attacks
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.20038v3](http://arxiv.org/pdf/2508.20038v3)**

> **作者:** Sheng Liu; Qiang Sheng; Danding Wang; Yang Li; Guang Yang; Juan Cao
>
> **备注:** EMNLP 2025 findings
>
> **摘要:** Despite advances in improving large language model (LLM) to refuse to answer malicious instructions, widely used LLMs remain vulnerable to jailbreak attacks where attackers generate instructions with distributions differing from safety alignment corpora. New attacks expose LLMs' inability to recognize unseen malicious instructions, highlighting a critical distributional mismatch between training data and real-world attacks that forces developers into reactive patching cycles. To tackle this challenge, we propose IMAGINE, a synthesis framework that leverages embedding space distribution analysis to generate jailbreak-like instructions. This approach effectively fills the distributional gap between authentic jailbreak patterns and safety alignment corpora. IMAGINE follows an iterative optimization process that dynamically evolves text generation distributions across iterations, thereby augmenting the coverage of safety alignment data distributions through synthesized data examples. Based on the safety-aligned corpus enhanced through IMAGINE, our framework demonstrates significant decreases in attack success rate on Qwen2.5, Llama3.1, and Llama3.2 without compromising their utility.
>
---
#### [replaced 014] Exploring Linguistic Features for Turkish Text Readability
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2306.03774v4](http://arxiv.org/pdf/2306.03774v4)**

> **作者:** Ahmet Yavuz Uluslu; Gerold Schneider
>
> **摘要:** This paper presents the first comprehensive study on automatic readability assessment of Turkish texts. We combine state-of-the-art neural network models with linguistic features at lexical, morphological, syntactic and discourse levels to develop an advanced readability tool. We evaluate the effectiveness of traditional readability formulas compared to modern automated methods and identify key linguistic features that determine the readability of Turkish texts.
>
---
#### [replaced 015] FRIDA to the Rescue! Analyzing Synthetic Data Effectiveness in Object-Based Common Sense Reasoning for Disaster Response
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.18452v3](http://arxiv.org/pdf/2502.18452v3)**

> **作者:** Mollie Shichman; Claire Bonial; Austin Blodgett; Taylor Hudson; Francis Ferraro; Rachel Rudinger
>
> **备注:** 8 pages, 3 figures, 5 tables
>
> **摘要:** During Human Robot Interactions in disaster relief scenarios, Large Language Models (LLMs) have the potential for substantial physical reasoning to assist in mission objectives. However, these reasoning capabilities are often found only in larger models, which are not currently reasonable to deploy on robotic systems due to size constraints. To meet our problem space requirements, we introduce a dataset and pipeline to create Field Reasoning and Instruction Decoding Agent (FRIDA) models. In our pipeline, domain experts and linguists combine their knowledge to make high-quality, few-shot prompts used to generate synthetic data for fine-tuning. We hand-curate datasets for this few-shot prompting and for evaluation to improve LLM reasoning on both general and disaster-specific objects. We concurrently run an ablation study to understand which kinds of synthetic data most affect performance. We fine-tune several small instruction-tuned models and find that ablated FRIDA models only trained on objects' physical state and function data outperformed both the FRIDA models trained on all synthetic data and the base models in our evaluation. We demonstrate that the FRIDA pipeline is capable of instilling physical common sense with minimal data.
>
---
#### [replaced 016] DynaSaur: Large Language Agents Beyond Predefined Actions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.01747v3](http://arxiv.org/pdf/2411.01747v3)**

> **作者:** Dang Nguyen; Viet Dac Lai; Seunghyun Yoon; Ryan A. Rossi; Handong Zhao; Ruiyi Zhang; Puneet Mathur; Nedim Lipka; Yu Wang; Trung Bui; Franck Dernoncourt; Tianyi Zhou
>
> **备注:** Published as a conference paper at COLM 2025
>
> **摘要:** Existing LLM agent systems typically select actions from a fixed and predefined set at every step. While this approach is effective in closed, narrowly scoped environments, it presents two major challenges for real-world, open-ended scenarios: (1) it significantly restricts the planning and acting capabilities of LLM agents, and (2) it requires substantial human effort to enumerate and implement all possible actions, which is impractical in complex environments with a vast number of potential actions. To address these limitations, we propose an LLM agent framework that can dynamically create and compose actions as needed. In this framework, the agent interacts with its environment by generating and executing programs written in a general-purpose programming language. Moreover, generated actions are accumulated over time for future reuse. Our extensive experiments across multiple benchmarks show that this framework significantly improves flexibility and outperforms prior methods that rely on a fixed action set. Notably, it enables LLM agents to adapt and recover in scenarios where predefined actions are insufficient or fail due to unforeseen edge cases. Our code can be found in https://github.com/adobe-research/dynasaur.
>
---
#### [replaced 017] Learning Optimal Prompt Ensemble for Multi-source Visual Prompt Transfer
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.12311v4](http://arxiv.org/pdf/2504.12311v4)**

> **作者:** Jianhua Liu; Liwen Cao; Yanru Wu; Zijie Zhao; Yang Li
>
> **摘要:** Prompt tuning has emerged as a lightweight strategy for adapting foundation models to downstream tasks, particularly for resource-constrained systems. As pre-trained prompts become valuable assets, combining multiple source prompts offers a promising approach to enhance generalization for new tasks by leveraging complementary knowledge. However, naive aggregation often overlooks different source prompts have different contribution potential to the target task. To address this, we propose HGPrompt, a dynamic framework that learns optimal ensemble weights. These weights are optimized by jointly maximizing an information-theoretic metric for transferability and minimizing gradient conflicts via a novel regularization strategy. Specifically, we propose a differentiable prompt transferability metric to captures the discriminability of prompt-induced features on the target task. Meanwhile, HGPrompt match the gradient variances with respect to different source prompts based on Hessian and Fisher Information, ensuring stable and coherent knowledge transfer while suppressing gradient conflicts among them. Extensive experiments on the large-scale VTAB benchmark demonstrate the state-of-the-art performance of HGPrompt, validating its effectiveness in learning an optimal ensemble for effective multi-source prompt transfer.
>
---
#### [replaced 018] MiniCPM4: Ultra-Efficient LLMs on End Devices
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.07900v2](http://arxiv.org/pdf/2506.07900v2)**

> **作者:** MiniCPM Team; Chaojun Xiao; Yuxuan Li; Xu Han; Yuzhuo Bai; Jie Cai; Haotian Chen; Wentong Chen; Xin Cong; Ganqu Cui; Ning Ding; Shengda Fan; Yewei Fang; Zixuan Fu; Wenyu Guan; Yitong Guan; Junshao Guo; Yufeng Han; Bingxiang He; Yuxiang Huang; Baoxi Ji; Cunliang Kong; Qiuzuo Li; Siyuan Li; Wenhao Li; Xin Li; Yanghao Li; Yishan Li; Zhen Li; Dan Liu; Biyuan Lin; Yankai Lin; Xiang Long; Quanyu Lu; Yaxi Lu; Peiyan Luo; Hongya Lyu; Litu Ou; Yinxu Pan; Lushi Pu; Zekai Qu; Qundong Shi; Zijun Song; Jiayuan Su; Zhou Su; Ao Sun; Xianghui Sun; Peijun Tang; Fangzheng Wang; Feng Wang; Shuo Wang; Yudong Wang; Zheng Wang; Yesai Wu; Zhenyu Xiao; Jie Xie; Zihao Xie; Xiaoyue Xu; Yukun Yan; Jiarui Yuan; Jinqian Zhang; Kaihuo Zhang; Lei Zhang; Linyue Zhang; Xueren Zhang; Yudi Zhang; Hengyu Zhao; Weilin Zhao; Weilun Zhao; Yuanqian Zhao; Zhi Zheng; Chuyue Zhou; Ge Zhou; Jie Zhou; Wei Zhou; Yanghao Zhou; Zihan Zhou; Zixuan Zhou; Zhiyuan Liu; Guoyang Zeng; Chao Jia; Dahai Li; Maosong Sun
>
> **备注:** MiniCPM4 Technical Report
>
> **摘要:** This paper introduces MiniCPM4, a highly efficient large language model (LLM) designed explicitly for end-side devices. We achieve this efficiency through systematic innovation in four key dimensions: model architecture, training data, training algorithms, and inference systems. Specifically, in terms of model architecture, we propose InfLLM v2, a trainable sparse attention mechanism that accelerates both prefilling and decoding phases for long-context processing. Regarding training data, we propose UltraClean, an efficient and accurate pre-training data filtering and generation strategy, and UltraChat v2, a comprehensive supervised fine-tuning dataset. These datasets enable satisfactory model performance to be achieved using just 8 trillion training tokens. Regarding training algorithms, we propose ModelTunnel v2 for efficient pre-training strategy search, and improve existing post-training methods by introducing chunk-wise rollout for load-balanced reinforcement learning and data-efficient tenary LLM, BitCPM. Regarding inference systems, we propose CPM.cu that integrates sparse attention, model quantization, and speculative sampling to achieve efficient prefilling and decoding. To meet diverse on-device requirements, MiniCPM4 is available in two versions, with 0.5B and 8B parameters, respectively. Furthermore, we construct a hybrid reasoning model, MiniCPM4.1, which can be used in both deep reasoning mode and non-reasoning mode. Evaluation results demonstrate that MiniCPM4 and MiniCPM4.1 outperform similar-sized open-source models across benchmarks, with the 8B variants showing significant speed improvements on long sequence understanding and generation.
>
---
#### [replaced 019] SLM-Bench: A Comprehensive Benchmark of Small Language Models on Environmental Impacts--Extended Version
- **分类: cs.CL; cs.CY; cs.PF**

- **链接: [http://arxiv.org/pdf/2508.15478v2](http://arxiv.org/pdf/2508.15478v2)**

> **作者:** Nghiem Thanh Pham; Tung Kieu; Duc-Manh Nguyen; Son Ha Xuan; Nghia Duong-Trung; Danh Le-Phuoc
>
> **备注:** 24 pages. An extended version of "SLM-Bench: A Comprehensive Benchmark of Small Language Models on Environmental Impacts" accepted at EMNLP 2025
>
> **摘要:** Small Language Models (SLMs) offer computational efficiency and accessibility, yet a systematic evaluation of their performance and environmental impact remains lacking. We introduce SLM-Bench, the first benchmark specifically designed to assess SLMs across multiple dimensions, including accuracy, computational efficiency, and sustainability metrics. SLM-Bench evaluates 15 SLMs on 9 NLP tasks using 23 datasets spanning 14 domains. The evaluation is conducted on 4 hardware configurations, providing a rigorous comparison of their effectiveness. Unlike prior benchmarks, SLM-Bench quantifies 11 metrics across correctness, computation, and consumption, enabling a holistic assessment of efficiency trade-offs. Our evaluation considers controlled hardware conditions, ensuring fair comparisons across models. We develop an open-source benchmarking pipeline with standardized evaluation protocols to facilitate reproducibility and further research. Our findings highlight the diverse trade-offs among SLMs, where some models excel in accuracy while others achieve superior energy efficiency. SLM-Bench sets a new standard for SLM evaluation, bridging the gap between resource efficiency and real-world applicability.
>
---
#### [replaced 020] From Attack Descriptions to Vulnerabilities: A Sentence Transformer-Based Approach
- **分类: cs.CR; cs.CL; cs.LG; 68T50 Natural language processing; D.4.6; I.2.7**

- **链接: [http://arxiv.org/pdf/2509.02077v2](http://arxiv.org/pdf/2509.02077v2)**

> **作者:** Refat Othman; Diaeddin Rimawi; Bruno Rossi; Barbara Russo
>
> **备注:** Accepted in The Journal of Systems and Software (2025)
>
> **摘要:** In the domain of security, vulnerabilities frequently remain undetected even after their exploitation. In this work, vulnerabilities refer to publicly disclosed flaws documented in Common Vulnerabilities and Exposures (CVE) reports. Establishing a connection between attacks and vulnerabilities is essential for enabling timely incident response, as it provides defenders with immediate, actionable insights. However, manually mapping attacks to CVEs is infeasible, thereby motivating the need for automation. This paper evaluates 14 state-of-the-art (SOTA) sentence transformers for automatically identifying vulnerabilities from textual descriptions of attacks. Our results demonstrate that the multi-qa-mpnet-base-dot-v1 (MMPNet) model achieves superior classification performance when using attack Technique descriptions, with an F1-score of 89.0, precision of 84.0, and recall of 94.7. Furthermore, it was observed that, on average, 56% of the vulnerabilities identified by the MMPNet model are also represented within the CVE repository in conjunction with an attack, while 61% of the vulnerabilities detected by the model correspond to those cataloged in the CVE repository. A manual inspection of the results revealed the existence of 275 predicted links that were not documented in the MITRE repositories. Consequently, the automation of linking attack techniques to vulnerabilities not only enhances the detection and response capabilities related to software security incidents but also diminishes the duration during which vulnerabilities remain exploitable, thereby contributing to the development of more secure systems.
>
---
#### [replaced 021] Two-Stage Quranic QA via Ensemble Retrieval and Instruction-Tuned Answer Extraction
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2508.06971v2](http://arxiv.org/pdf/2508.06971v2)**

> **作者:** Mohamed Basem; Islam Oshallah; Ali Hamdi; Khaled Shaban; Hozaifa Kassab
>
> **备注:** 8 pages , 4 figures , Accepted in Aiccsa 2025 , https://conferences.sigappfr.org/aiccsa2025/
>
> **摘要:** Quranic Question Answering presents unique challenges due to the linguistic complexity of Classical Arabic and the semantic richness of religious texts. In this paper, we propose a novel two-stage framework that addresses both passage retrieval and answer extraction. For passage retrieval, we ensemble fine-tuned Arabic language models to achieve superior ranking performance. For answer extraction, we employ instruction-tuned large language models with few-shot prompting to overcome the limitations of fine-tuning on small datasets. Our approach achieves state-of-the-art results on the Quran QA 2023 Shared Task, with a MAP@10 of 0.3128 and MRR@10 of 0.5763 for retrieval, and a pAP@10 of 0.669 for extraction, substantially outperforming previous methods. These results demonstrate that combining model ensembling and instruction-tuned language models effectively addresses the challenges of low-resource question answering in specialized domains.
>
---
#### [replaced 022] Explaining Length Bias in LLM-Based Preference Evaluations
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2407.01085v5](http://arxiv.org/pdf/2407.01085v5)**

> **作者:** Zhengyu Hu; Linxin Song; Jieyu Zhang; Zheyuan Xiao; Tianfu Wang; Zhengyu Chen; Nicholas Jing Yuan; Jianxun Lian; Kaize Ding; Hui Xiong
>
> **摘要:** The use of large language models (LLMs) as judges, particularly in preference comparisons, has become widespread, but this reveals a notable bias towards longer responses, undermining the reliability of such evaluations. To better understand such bias, we propose to decompose the preference evaluation metric, specifically the win rate, into two key components: desirability and information mass, where the former is length-independent and related to trustworthiness such as correctness, toxicity, and consistency, and the latter is length-dependent and represents the amount of information in the response. We empirically demonstrated the decomposition through controlled experiments and found that response length impacts evaluations by influencing information mass. To derive a reliable evaluation metric that assesses content quality without being confounded by response length, we propose AdapAlpaca, a simple yet effective adjustment to win rate measurement. Specifically, AdapAlpaca ensures a fair comparison of response quality by aligning the lengths of reference and test model responses under equivalent length intervals.
>
---
#### [replaced 023] Transplant Then Regenerate: A New Paradigm for Text Data Augmentation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.14723v2](http://arxiv.org/pdf/2508.14723v2)**

> **作者:** Guangzhan Wang; Hongyu Zhang; Beijun Shen; Xiaodong Gu
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Data augmentation is a critical technique in deep learning. Traditional methods like Back-translation typically focus on lexical-level rephrasing, which primarily produces variations with the same semantics. While large language models (LLMs) have enhanced text augmentation by their "knowledge emergence" capability, controlling the style and structure of these outputs remains challenging and requires meticulous prompt engineering. In this paper, we propose LMTransplant, a novel text augmentation paradigm leveraging LLMs. The core idea of LMTransplant is transplant-then-regenerate: incorporating seed text into a context expanded by LLM, and asking the LLM to regenerate a variant based on the expanded context. This strategy allows the model to create more diverse and creative content-level variants by fully leveraging the knowledge embedded in LLMs, while preserving the core attributes of the original text. We evaluate LMTransplant across various text-related tasks, demonstrating its superior performance over existing text augmentation methods. Moreover, LMTransplant demonstrates exceptional scalability as the size of augmented data grows.
>
---
#### [replaced 024] That is Unacceptable: the Moral Foundations of Canceling
- **分类: cs.CY; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.05720v3](http://arxiv.org/pdf/2503.05720v3)**

> **作者:** Soda Marem Lo; Oscar Araque; Rajesh Sharma; Marco Antonio Stranisci
>
> **摘要:** Canceling is a morally-driven phenomenon that hinders the development of safe social media platforms and contributes to ideological polarization. To address this issue we present the Canceling Attitudes Detection (CADE) dataset, an annotated corpus of canceling incidents aimed at exploring the factors of disagreements in evaluating people canceling attitudes on social media. Specifically, we study the impact of annotators' morality in their perception of canceling, showing that morality is an independent axis for the explanation of disagreement on this phenomenon. Annotator's judgments heavily depend on the type of controversial events and involved celebrities. This shows the need to develop more event-centric datasets to better understand how harms are perpetrated in social media and to develop more aware technologies for their detection.
>
---
#### [replaced 025] Short-video Propagation Influence Rating: A New Real-world Dataset and A New Large Graph Model
- **分类: cs.CV; cs.CL; cs.LG; cs.MM; cs.SI**

- **链接: [http://arxiv.org/pdf/2503.23746v2](http://arxiv.org/pdf/2503.23746v2)**

> **作者:** Dizhan Xue; Shengsheng Qian; Chuanrui Hu; Changsheng Xu
>
> **摘要:** Short-video platforms have gained immense popularity, captivating the interest of millions, if not billions, of users globally. Recently, researchers have highlighted the significance of analyzing the propagation of short-videos, which typically involves discovering commercial values, public opinions, user behaviors, etc. This paper proposes a new Short-video Propagation Influence Rating (SPIR) task and aims to promote SPIR from both the dataset and method perspectives. First, we propose a new Cross-platform Short-Video (XS-Video) dataset, which aims to provide a large-scale and real-world short-video propagation network across various platforms to facilitate the research on short-video propagation. Our XS-Video dataset includes 117,720 videos, 381,926 samples, and 535 topics across 5 biggest Chinese platforms, annotated with the propagation influence from level 0 to 9. To the best of our knowledge, this is the first large-scale short-video dataset that contains cross-platform data or provides all of the views, likes, shares, collects, fans, comments, and comment content. Second, we propose a Large Graph Model (LGM) named NetGPT, based on a novel three-stage training mechanism, to bridge heterogeneous graph-structured data with the powerful reasoning ability and knowledge of Large Language Models (LLMs). Our NetGPT can comprehend and analyze the short-video propagation graph, enabling it to predict the long-term propagation influence of short-videos. Comprehensive experimental results evaluated by both classification and regression metrics on our XS-Video dataset indicate the superiority of our method for SPIR.
>
---
#### [replaced 026] MultiGen: Child-Friendly Multilingual Speech Generator with LLMs
- **分类: eess.AS; cs.AI; cs.CL; eess.SP**

- **链接: [http://arxiv.org/pdf/2508.08715v3](http://arxiv.org/pdf/2508.08715v3)**

> **作者:** Xiaoxue Gao; Huayun Zhang; Nancy F. Chen
>
> **备注:** 5 pages
>
> **摘要:** Generative speech models have demonstrated significant potential in improving human-machine interactions, offering valuable real-world applications such as language learning for children. However, achieving high-quality, child-friendly speech generation remains challenging, particularly for low-resource languages across diverse languages and cultural contexts. In this paper, we propose MultiGen, a multilingual speech generation model with child-friendly interaction, leveraging LLM architecture for speech generation tailored for low-resource languages. We propose to integrate age-appropriate multilingual speech generation using LLM architectures, which can be used to facilitate young children's communication with AI systems through culturally relevant context in three low-resource languages: Singaporean accent Mandarin, Malay, and Tamil. Experimental results from both objective metrics and subjective evaluations demonstrate the superior performance of the proposed MultiGen compared to baseline methods.
>
---
#### [replaced 027] Autoformalization in the Wild: Assessing LLMs on Real-World Mathematical Definitions
- **分类: cs.CL; cs.FL**

- **链接: [http://arxiv.org/pdf/2502.12065v3](http://arxiv.org/pdf/2502.12065v3)**

> **作者:** Lan Zhang; Marco Valentino; Andre Freitas
>
> **备注:** EMNLP 2025 Camera-Ready Version
>
> **摘要:** Thanks to their linguistic capabilities, LLMs offer an opportunity to bridge the gap between informal mathematics and formal languages through autoformalization. However, it is still unclear how well LLMs generalize to sophisticated and naturally occurring mathematical statements. To address this gap, we investigate the task of autoformalizing real-world mathematical definitions: a critical component of mathematical discourse. Specifically, we introduce two novel resources for autoformalization, collecting definitions from Wikipedia (Def_Wiki) and arXiv papers (Def_ArXiv). We then systematically evaluate a range of LLMs, analyzing their ability to formalize definitions into Isabelle/HOL. Furthermore, we investigate strategies to enhance LLMs' performance including refinement through external feedback from Proof Assistants, and formal definition grounding, where we augment LLMs' formalizations through relevant contextual elements from formal mathematical libraries. Our findings reveal that definitions present a greater challenge compared to existing benchmarks, such as miniF2F. In particular, we found that LLMs still struggle with self-correction, and aligning with relevant mathematical libraries. At the same time, structured refinement methods and definition grounding strategies yield notable improvements of up to 16% on self-correction capabilities and 43% on the reduction of undefined errors, highlighting promising directions for enhancing LLM-based autoformalization in real-world scenarios.
>
---
#### [replaced 028] Context Reasoner: Incentivizing Reasoning Capability for Contextualized Privacy and Safety Compliance via Reinforcement Learning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.14585v2](http://arxiv.org/pdf/2505.14585v2)**

> **作者:** Wenbin Hu; Haoran Li; Huihao Jing; Qi Hu; Ziqian Zeng; Sirui Han; Heli Xu; Tianshu Chu; Peizhao Hu; Yangqiu Song
>
> **备注:** Accepted to EMNLP 2025 Main
>
> **摘要:** While Large Language Models (LLMs) exhibit remarkable capabilities, they also introduce significant safety and privacy risks. Current mitigation strategies often fail to preserve contextual reasoning capabilities in risky scenarios. Instead, they rely heavily on sensitive pattern matching to protect LLMs, which limits the scope. Furthermore, they overlook established safety and privacy standards, leading to systemic risks for legal compliance. To address these gaps, we formulate safety and privacy issues into contextualized compliance problems following the Contextual Integrity (CI) theory. Under the CI framework, we align our model with three critical regulatory standards: GDPR, EU AI Act, and HIPAA. Specifically, we employ reinforcement learning (RL) with a rule-based reward to incentivize contextual reasoning capabilities while enhancing compliance with safety and privacy norms. Through extensive experiments, we demonstrate that our method not only significantly enhances legal compliance (achieving a +8.58% accuracy improvement in safety/privacy benchmarks) but also further improves general reasoning capability. For OpenThinker-7B, a strong reasoning model that significantly outperforms its base model Qwen2.5-7B-Instruct across diverse subjects, our method enhances its general reasoning capabilities, with +2.05% and +8.98% accuracy improvement on the MMLU and LegalBench benchmark, respectively.
>
---
#### [replaced 029] R2C2-Coder: Enhancing and Benchmarking Real-world Repository-level Code Completion Abilities of Code Large Language Models
- **分类: cs.CL; cs.SE**

- **链接: [http://arxiv.org/pdf/2406.01359v3](http://arxiv.org/pdf/2406.01359v3)**

> **作者:** Ken Deng; Jiaheng Liu; He Zhu; Congnan Liu; Jingxin Li; Jiakai Wang; Peng Zhao; Chenchen Zhang; Yanan Wu; Xueqiao Yin; Yuanxing Zhang; Zizheng Zhan; Wenbo Su; Bangyu Xiang; Tiezheng Ge; Bo Zheng
>
> **摘要:** Code completion models have made significant progress in recent years. Recently, repository-level code completion has drawn more attention in modern software development, and several baseline methods and benchmarks have been proposed. However, existing repository-level code completion methods often fall short of fully using the extensive context of a project repository, such as the intricacies of relevant files and class hierarchies. Besides, the existing benchmarks usually focus on limited code completion scenarios, which cannot reflect the repository-level code completion abilities well of existing methods. To address these limitations, we propose the R2C2-Coder to enhance and benchmark the real-world repository-level code completion abilities of code Large Language Models, where the R2C2-Coder includes a code prompt construction method R2C2-Enhance and a well-designed benchmark R2C2-Bench. Specifically, first, in R2C2-Enhance, we first construct the candidate retrieval pool and then assemble the completion prompt by retrieving from the retrieval pool for each completion cursor position. Second, based on R2C2 -Enhance, we can construct a more challenging and diverse R2C2-Bench with training, validation and test splits, where a context perturbation strategy is proposed to simulate the real-world repository-level code completion well. Extensive results on multiple benchmarks demonstrate the effectiveness of our R2C2-Coder.
>
---
#### [replaced 030] Learning an Efficient Multi-Turn Dialogue Evaluator from Multiple Judges
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.00454v2](http://arxiv.org/pdf/2508.00454v2)**

> **作者:** Yuqi Tang; Kehua Feng; Yunfeng Wang; Zhiwen Chen; Chengfei Lv; Gang Yu; Qiang Zhang; Keyan Ding
>
> **备注:** 15 pages, 2 pages, under review
>
> **摘要:** Evaluating the conversational abilities of large language models (LLMs) remains a challenging task. Current mainstream approaches primarily rely on the "LLM-as-a-judge" paradigm, where an LLM is prompted to serve as an evaluator to assess dialogue quality. However, such methods often suffer from various biases, which undermine the reliability and consistency of the evaluation results. To mitigate these biases, recent methods employ multiple LLMs as judges and aggregate their judgments to select the optimal assessment. Although effective, this multi-judge approach incurs significant computational overhead during inference. In this paper, we propose an efficient multi-turn dialogue evaluator that captures the collective wisdom of multiple LLM judges by aggregating their preference knowledge into a single model. Our approach preserves the advantages of diverse multi-judge feedback while drastically reducing the evaluation cost, enabling fast and flexible dialogue quality assessment. Extensive experiments on seven single rating and pairwise comparison dialogue evaluation benchmarks demonstrate that our method outperforms existing baselines across diverse scenarios, showcasing its efficiency and robustness.
>
---
#### [replaced 031] FutureGen: A RAG-based Approach to Generate the Future Work of Scientific Article
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.16561v3](http://arxiv.org/pdf/2503.16561v3)**

> **作者:** Ibrahim Al Azher; Miftahul Jannat Mokarrama; Zhishuai Guo; Sagnik Ray Choudhury; Hamed Alhoori
>
> **备注:** 12 pages, 6 figures, Accepted for publication at the Workshop on AI Principles in Science Communication (Ai4SC'25), held in conjunction with the IEEE eScience Conference 2025
>
> **摘要:** The Future Work section of a scientific article outlines potential research directions by identifying gaps and limitations of a current study. This section serves as a valuable resource for early-career researchers seeking unexplored areas and experienced researchers looking for new projects or collaborations. In this study, we generate future work suggestions from a scientific article. To enrich the generation process with broader insights and reduce the chance of missing important research directions, we use context from related papers using RAG. We experimented with various Large Language Models (LLMs) integrated into Retrieval-Augmented Generation (RAG). We incorporate an LLM feedback mechanism to enhance the quality of the generated content and introduce an LLM-as-a-judge framework for robust evaluation, assessing key aspects such as novelty, hallucination, and feasibility. Our results demonstrate that the RAG-based approach using GPT-4o mini, combined with an LLM feedback mechanism, outperforms other methods based on both qualitative and quantitative evaluations. Moreover, we conduct a human evaluation to assess the LLM as an extractor, generator, and feedback provider.
>
---
#### [replaced 032] Can Compact Language Models Search Like Agents? Distillation-Guided Policy Optimization for Preserving Agentic RAG Capabilities
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.20324v2](http://arxiv.org/pdf/2508.20324v2)**

> **作者:** Rikuto Kotoge; Mai Nishimura; Jiaxin Ma
>
> **摘要:** Reinforcement Learning has emerged as a post-training approach to elicit agentic RAG behaviors such as search and planning from language models. However, compact language models (e.g., 0.5B parameters) struggle due to poor reasoning ability, resulting in sparse rewards and unstable training. To overcome these difficulties, we propose Distillation-Guided Policy Optimization (DGPO), which addresses the challenges through cold-start initialization from teacher demonstrations and continuous teacher guidance during policy optimization. To systematically evaluate our approach, we introduce Agentic RAG Capabilities (ARC), a fine-grained metric analyzing reasoning, search coordination, and response synthesis. Comprehensive experiments demonstrate that DGPO enables compact models to achieve sophisticated agentic search behaviors, even outperforming the larger teacher model in some cases. DGPO makes agentic RAG feasible in computing resource-constrained environments.
>
---
#### [replaced 033] HalluEntity: Benchmarking and Understanding Entity-Level Hallucination Detection
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11948v3](http://arxiv.org/pdf/2502.11948v3)**

> **作者:** Min-Hsuan Yeh; Max Kamachee; Seongheon Park; Yixuan Li
>
> **备注:** TMLR 2025
>
> **摘要:** To mitigate the impact of hallucination nature of LLMs, many studies propose detecting hallucinated generation through uncertainty estimation. However, these approaches predominantly operate at the sentence or paragraph level, failing to pinpoint specific spans or entities responsible for hallucinated content. This lack of granularity is especially problematic for long-form outputs that mix accurate and fabricated information. To address this limitation, we explore entity-level hallucination detection. We propose a new data set, HalluEntity, which annotates hallucination at the entity level. Based on the dataset, we comprehensively evaluate uncertainty-based hallucination detection approaches across 17 modern LLMs. Our experimental results show that uncertainty estimation approaches focusing on individual token probabilities tend to over-predict hallucinations, while context-aware methods show better but still suboptimal performance. Through an in-depth qualitative study, we identify relationships between hallucination tendencies and linguistic properties and highlight important directions for future research. HalluEntity: https://huggingface.co/datasets/samuelyeh/HalluEntity
>
---
#### [replaced 034] DAPFAM: A Domain-Aware Family-level Dataset to benchmark cross domain patent retrieval
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2506.22141v2](http://arxiv.org/pdf/2506.22141v2)**

> **作者:** Iliass Ayaou; Denis Cavallucci; Hicham Chibane
>
> **摘要:** Patent prior-art retrieval becomes especially challenging when relevant disclosures cross technological boundaries. Existing benchmarks lack explicit domain partitions, making it difficult to assess how retrieval systems cope with such shifts. We introduce DAPFAM, a family-level benchmark with explicit IN-domain and OUT-domain partitions defined by a new IPC3 overlap scheme. The dataset contains 1,247 query families and 45,336 target families aggregated at the family level to reduce international redundancy, with citation based relevance judgments. We conduct 249 controlled experiments spanning lexical (BM25) and dense (transformer) backends, document and passage level retrieval, multiple query and document representations, aggregation strategies, and hybrid fusion via Reciprocal Rank Fusion (RRF). Results reveal a pronounced domain gap: OUT-domain performance remains roughly five times lower than IN-domain across all configurations. Passage-level retrieval consistently outperforms document-level, and dense methods provide modest gains over BM25, but none close the OUT-domain gap. Document-level RRF yields strong effectiveness efficiency trade-offs with minimal overhead. By exposing the persistent challenge of cross-domain retrieval, DAPFAM provides a reproducible, compute-aware testbed for developing more robust patent IR systems. The dataset is publicly available on huggingface at https://huggingface.co/datasets/datalyes/DAPFAM_patent.
>
---
#### [replaced 035] Is Random Attention Sufficient for Sequence Modeling? Disentangling Trainable Components in the Transformer
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01115v3](http://arxiv.org/pdf/2506.01115v3)**

> **作者:** Yihe Dong; Lorenzo Noci; Mikhail Khodak; Mufan Li
>
> **摘要:** The transformer architecture is central to the success of modern Large Language Models (LLMs), in part due to its surprising ability to perform a wide range of tasks - including mathematical reasoning, memorization, and retrieval - using only gradient-based learning on next-token prediction. While the core component of a transformer is the self-attention mechanism, we question how much, and which aspects, of the performance gains can be attributed to it. To this end, we compare standard transformers to variants in which either the MLP layers or the attention weights are frozen at initialization. Surprisingly, we find that attention with frozen key and query weights is not only able to form induction heads, but can also perform competitively on language modeling. We formalize this by proving a new expressivity result for transformer models with frozen key and query weights. To further isolate the contribution of attention, we design MixiT, an architecture with entirely random attention scores, with provably stable signal propagation that overcomes prior depth-wise scaling challenges in random transformers. We use the successes and failures of MixiT to understand the role each transformer component plays, such as attention being largely responsible for in-context reasoning, and MLPs being responsible for, but collaborates with attention, on knowledge storage. Our results suggest that the transformer architecture has a built-in inductive bias towards forming specialized circuits, as it does even without learnable attention weights.
>
---
#### [replaced 036] HamRaz: A Culture-Based Persian Conversation Dataset for Person-Centered Therapy Using LLM Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.05982v2](http://arxiv.org/pdf/2502.05982v2)**

> **作者:** Mohammad Amin Abbasi; Farnaz Sadat Mirnezami; Ali Neshati; Hassan Naderi
>
> **摘要:** We present HamRaz, a culturally adapted Persian-language dataset for AI-assisted mental health support, grounded in Person-Centered Therapy (PCT). To reflect real-world therapeutic challenges, we combine script-based dialogue with adaptive large language models (LLM) role-playing, capturing the ambiguity and emotional nuance of Persian-speaking clients. We introduce HamRazEval, a dual-framework for assessing conversational and therapeutic quality using General Metrics and specialized psychological relationship measures. Human evaluations show HamRaz outperforms existing baselines in empathy, coherence, and realism. This resource contributes to the Digital Humanities by bridging language, culture, and mental health in underrepresented communities.
>
---
#### [replaced 037] DaMoC: Efficiently Selecting the Optimal Large Language Model for Fine-tuning Domain Tasks Based on Data and Model Compression
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.01221v2](http://arxiv.org/pdf/2509.01221v2)**

> **作者:** Wei Huang; Huang Wei; Yinggui Wang
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Large language models (LLMs) excel in general tasks but struggle with domain-specific ones, requiring fine-tuning with specific data. With many open-source LLMs available, selecting the best model for fine-tuning downstream tasks is challenging, primarily focusing on how to quickly identify the optimal LLM. We introduce a Data and Model Compression Framework (DaMoC) that addresses this challenge by: 1) Data Level: A systematic categorization of data filtering methodologies for LLMs is first established, classifying them into three distinct paradigms: (1) distribution-aware methods, (2) quality-aware methods, and (3) hybrid approaches considering both dimensions. Further, we enhance the density of key tokens in the text achieving token compression. Subsequently, we use an LLM to iterative rewrite the text to optimize its expression. 2) Model Level: We use layer similarity scores to assess each layer's importance and remove those with lower importance. Then, we introduce a sparse merging paradigm to preserve as much of the original model's capability as possible. Extensive experiments on four datasets, medical Q&A, financial Q&A, general Q&A, and reading comprehension, show that we can select the optimal LLM while saving approximately 20-fold in training time.
>
---
#### [replaced 038] NADI 2025: The First Multidialectal Arabic Speech Processing Shared Task
- **分类: cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2509.02038v2](http://arxiv.org/pdf/2509.02038v2)**

> **作者:** Bashar Talafha; Hawau Olamide Toyin; Peter Sullivan; AbdelRahim Elmadany; Abdurrahman Juma; Amirbek Djanibekov; Chiyu Zhang; Hamad Alshehhi; Hanan Aldarmaki; Mustafa Jarrar; Nizar Habash; Muhammad Abdul-Mageed
>
> **摘要:** We present the findings of the sixth Nuanced Arabic Dialect Identification (NADI 2025) Shared Task, which focused on Arabic speech dialect processing across three subtasks: spoken dialect identification (Subtask 1), speech recognition (Subtask 2), and diacritic restoration for spoken dialects (Subtask 3). A total of 44 teams registered, and during the testing phase, 100 valid submissions were received from eight unique teams. The distribution was as follows: 34 submissions for Subtask 1 "five teams{\ae}, 47 submissions for Subtask 2 "six teams", and 19 submissions for Subtask 3 "two teams". The best-performing systems achieved 79.8% accuracy on Subtask 1, 35.68/12.20 WER/CER (overall average) on Subtask 2, and 55/13 WER/CER on Subtask 3. These results highlight the ongoing challenges of Arabic dialect speech processing, particularly in dialect identification, recognition, and diacritic restoration. We also summarize the methods adopted by participating teams and briefly outline directions for future editions of NADI.
>
---
#### [replaced 039] EQ-Knight: A Memory-Augmented LLM Agent for Strategic Affective Gaming in Debt Recovery
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.21080v4](http://arxiv.org/pdf/2503.21080v4)**

> **作者:** Yunbo Long; Yuhan Liu; Liming Xu; Alexandra Brintrup
>
> **摘要:** Large language model-based chatbots have enhanced engagement in financial negotiations, but their overreliance on passive empathy introduces critical risks in credit collection. While empathy-driven approaches preserve client satisfaction in benign cases, they fail catastrophically against dishonest debtors--individuals who exploit conciliatory tactics to manipulate terms or evade repayment. Blindly prioritizing "customer experience" in such scenarios leads to creditor vulnerabilities: revenue leakage, moral hazard, and systemic exploitation. To address this, we propose EQ-Knight, an LLM agent that dynamically optimizes emotional strategy to defend creditor interests. Unlike naive empathy-centric bots, EQ-Knight integrates emotion memory and game-theoretic reasoning, powered by a Hidden Markov Model (HMM) to track and predict debtor emotional states. By analyzing both real-time and historical emotional cues, EQ-Knight strategically counters negative emotions (e.g., aggression, feigned distress) while preserving productive debtor relationships. Experiments demonstrate EQ-Knight's superiority over conventional LLM negotiators: it achieves a 32\% reduction in concession losses without compromising recovery rates, particularly in adversarial cases where debtors weaponize negative emotions (e.g., intimidation, guilt-tripping) to coerce concessions. For credit agencies, EQ-Knight transforms LLMs from high-risk "people-pleasers" into strategic emotion-defenders--balancing emotional intelligence with tactical rigor to enforce accountability and deter exploitation.
>
---
#### [replaced 040] Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.11110v4](http://arxiv.org/pdf/2501.11110v4)**

> **作者:** Yiyao Yu; Yuxiang Zhang; Dongdong Zhang; Xiao Liang; Hengyuan Zhang; Xingxing Zhang; Ziyi Yang; Mahmoud Khademi; Hany Awadalla; Junjie Wang; Yujiu Yang; Furu Wei
>
> **备注:** Accepted to ACL 2025 (Main)
>
> **摘要:** Large Language Models (LLMs) have made notable progress in mathematical reasoning, yet often rely on single-paradigm reasoning, limiting their effectiveness across diverse tasks. We introduce Chain-of-Reasoning (CoR), a novel unified framework integrating multiple reasoning paradigms--Natural Language Reasoning (NLR), Algorithmic Reasoning (AR), and Symbolic Reasoning (SR)--to enable synergistic collaboration. CoR generates multiple potential answers via different reasoning paradigms and synthesizes them into a coherent final solution. We propose a Progressive Paradigm Training (PPT) strategy for models to progressively master these paradigms, leading to CoR-Math-7B. Experimental results demonstrate that CoR-Math-7B significantly outperforms current SOTA models, achieving up to a 41.0% absolute improvement over GPT-4o in theorem proving and a 15.0% improvement over RL-based methods on the MATH benchmark in arithmetic tasks. These results show the enhanced mathematical comprehension ability of our model, enabling zero-shot generalization across tasks.
>
---
#### [replaced 041] An Unsupervised Natural Language Processing Pipeline for Assessing Referral Appropriateness
- **分类: cs.CL; cs.LG; 68T50; I.2.7; J.1; J.3**

- **链接: [http://arxiv.org/pdf/2501.14701v2](http://arxiv.org/pdf/2501.14701v2)**

> **作者:** Vittorio Torri; Annamaria Bottelli; Michele Ercolanoni; Olivia Leoni; Francesca Ieva
>
> **备注:** 49 pages, 10 figures
>
> **摘要:** Objective: Assessing the appropriateness of diagnostic referrals is critical for improving healthcare efficiency and reducing unnecessary procedures. However, this task becomes challenging when referral reasons are recorded only as free text rather than structured codes, like in the Italian NHS. To address this gap, we propose a fully unsupervised Natural Language Processing (NLP) pipeline capable of extracting and evaluating referral reasons without relying on labelled datasets. Methods: Our pipeline leverages Transformer-based embeddings pre-trained on Italian medical texts to cluster referral reasons and assess their alignment with appropriateness guidelines. It operates in an unsupervised setting and is designed to generalize across different examination types. We analyzed two complete regional datasets from the Lombardy Region (Italy), covering all referrals between 2019 and 2021 for venous echocolordoppler of the lower limbs (ECD;n=496,971; development) and flexible endoscope colonoscopy (FEC; n=407,949; testing only). For both, a random sample of 1,000 referrals was manually annotated to measure performance. Results: The pipeline achieved high performance in identifying referral reasons (Prec=92.43% (ECD), 93.59% (FEC); Rec=83.28% (ECD), 92.70% (FEC)) and appropriateness (Prec=93.58% (ECD), 94.66% (FEC); Rec=91.52% (ECD), 93.96% (FEC)). At the regional level, the analysis identified relevant inappropriate referral groups and variation across contexts, findings that informed a new Lombardy Region resolution to reinforce guideline adherence. Conclusions: This study presents a robust, scalable, unsupervised NLP pipeline for assessing referral appropriateness in large, real-world datasets. It demonstrates how such data can be effectively leveraged, providing public health authorities with a deployable AI tool to monitor practices and support evidence-based policy.
>
---
#### [replaced 042] MyProfessors: Mining Turkish Student Reviews
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2109.02325v5](http://arxiv.org/pdf/2109.02325v5)**

> **作者:** Ibrahim Faruk Ceylan; Necmettin Bera Calik
>
> **备注:** The paper is withdrawn due to the scraping errors in the dataset collection process and affected results
>
> **摘要:** We introduce Hocalarim (MyProfessors), the largest student review dataset available for the Turkish language. It consists of over 5000 professor reviews left online by students, with different aspects of education rated on a scale of 1 to 5 stars. We investigate the properties of the dataset and present its statistics. We examine the impact of students' institution type on their ratings and the correlation of students' bias to give positive or negative feedback.
>
---
#### [replaced 043] Understanding Space Is Rocket Science -- Only Top Reasoning Models Can Solve Spatial Understanding Tasks
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.02175v2](http://arxiv.org/pdf/2509.02175v2)**

> **作者:** Nils Hoehing; Mayug Maniparambil; Ellen Rushe; Noel E. O'Connor; Anthony Ventresque
>
> **摘要:** We propose RocketScience, an open-source contrastive VLM benchmark that tests for spatial relation understanding. It is comprised of entirely new real-world image-text pairs covering mostly relative spatial understanding and the order of objects. The benchmark is designed to be very easy for humans and hard for the current generation of VLMs, and this is empirically verified. Our results show a striking lack of spatial relation understanding in open source and frontier commercial VLMs and a surprisingly high performance of reasoning models. Additionally, we perform a disentanglement analysis to separate the contributions of object localization and spatial reasoning in chain-of-thought-based models and find that the performance on the benchmark is bottlenecked by spatial reasoning and not object localization capabilities. We release the dataset with a CC-BY-4.0 license and make the evaluation code available at: https://github.com/nilshoehing/rocketscience
>
---
#### [replaced 044] Explicit Learning and the LLM in Machine Translation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.09454v4](http://arxiv.org/pdf/2503.09454v4)**

> **作者:** Malik Marmonier; Rachel Bawden; Benoît Sagot
>
> **摘要:** This study explores an LLM's ability to learn new languages using explanations found in a grammar book, a process we term "explicit learning." To rigorously assess this ability, we design controlled translation experiments between English and constructed languages generated, through specific cryptographic means, from Latin or French. Contrary to previous studies, our results demonstrate that LLMs do possess a measurable capacity for explicit learning. This ability, however, diminishes as the complexity of the linguistic phenomena to be learned increases. Supervised fine-tuning on ad hoc chains of thought significantly enhances LLM performance but struggles to generalize to typologically novel or more complex linguistic features. These findings point to the need for more diverse training sets and alternative fine-tuning strategies to further improve explicit learning by LLMs, benefiting low-resource languages typically described in grammar books but lacking extensive corpora.
>
---
#### [replaced 045] Rapid Word Learning Through Meta In-Context Learning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.14791v4](http://arxiv.org/pdf/2502.14791v4)**

> **作者:** Wentao Wang; Guangyuan Jiang; Tal Linzen; Brenden M. Lake
>
> **备注:** EMNLP 2025
>
> **摘要:** Humans can quickly learn a new word from a few illustrative examples, and then systematically and flexibly use it in novel contexts. Yet the abilities of current language models for few-shot word learning, and methods for improving these abilities, are underexplored. In this study, we introduce a novel method, Meta-training for IN-context learNing Of Words (Minnow). This method trains language models to generate new examples of a word's usage given a few in-context examples, using a special placeholder token to represent the new word. This training is repeated on many new words to develop a general word-learning ability. We find that training models from scratch with Minnow on human-scale child-directed language enables strong few-shot word learning, comparable to a large language model (LLM) pre-trained on orders of magnitude more data. Furthermore, through discriminative and generative evaluations, we demonstrate that finetuning pre-trained LLMs with Minnow improves their ability to discriminate between new words, identify syntactic categories of new words, and generate reasonable new usages and definitions for new words, based on one or a few in-context examples. These findings highlight the data efficiency of Minnow and its potential to improve language model performance in word learning tasks.
>
---
#### [replaced 046] A Survey of Graph Retrieval-Augmented Generation for Customized Large Language Models
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2501.13958v2](http://arxiv.org/pdf/2501.13958v2)**

> **作者:** Qinggang Zhang; Shengyuan Chen; Yuanchen Bei; Zheng Yuan; Huachi Zhou; Zijin Hong; Hao Chen; Yilin Xiao; Chuang Zhou; Yi Chang; Xiao Huang
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities in a wide range of tasks, yet their application to specialized domains remains challenging due to the need for deep expertise. Retrieval-Augmented generation (RAG) has emerged as a promising solution to customize LLMs for professional fields by seamlessly integrating external knowledge bases, enabling real-time access to domain-specific expertise during inference. Despite its potential, traditional RAG systems, based on flat text retrieval, face three critical challenges: (i) complex query understanding in professional contexts, (ii) difficulties in knowledge integration across distributed sources, and (iii) system efficiency bottlenecks at scale. This survey presents a systematic analysis of Graph-based Retrieval-Augmented Generation (GraphRAG), a new paradigm that revolutionizes domain-specific LLM applications. GraphRAG addresses traditional RAG limitations through three key innovations: (i) graph-structured knowledge representation that explicitly captures entity relationships and domain hierarchies, (ii) efficient graph-based retrieval techniques that enable context-preserving knowledge retrieval with multihop reasoning ability, and (iii) structure-aware knowledge integration algorithms that leverage retrieved knowledge for accurate and logical coherent generation of LLMs. In this survey, we systematically analyze the technical foundations of GraphRAG and examine current implementations across various professional domains, identifying key technical challenges and promising research directions. All the related resources of GraphRAG, including research papers, open-source data, and projects, are collected for the community in https://github.com/DEEP-PolyU/Awesome-GraphRAG.
>
---
#### [replaced 047] Improving Chain-of-Thought Reasoning via Quasi-Symbolic Abstractions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12616v2](http://arxiv.org/pdf/2502.12616v2)**

> **作者:** Leonardo Ranaldi; Marco Valentino; Andrè Freitas
>
> **摘要:** Chain-of-Though (CoT) represents a common strategy for reasoning in Large Language Models (LLMs) by decomposing complex tasks into intermediate inference steps. However, explanations generated via CoT are susceptible to content biases that negatively affect their robustness and faithfulness. To mitigate existing limitations, recent work has proposed using logical formalisms coupled with external symbolic solvers. However, fully symbolic approaches possess the bottleneck of requiring a complete translation from natural language to formal languages, a process that affects efficiency and flexibility. To achieve a trade-off, this paper investigates methods to disentangle content from logical reasoning without a complete formalisation. In particular, we present QuaSAR (for Quasi-Symbolic Abstract Reasoning), a variation of CoT that guides LLMs to operate at a higher level of abstraction via quasi-symbolic explanations. Our framework leverages the capability of LLMs to formalise only relevant variables and predicates, enabling the coexistence of symbolic elements with natural language. We show the impact of QuaSAR for in-context learning and for constructing demonstrations to improve the reasoning capabilities of smaller models. Our experiments show that quasi-symbolic abstractions can improve CoT-based methods by up to 8% accuracy, enhancing robustness and consistency on challenging adversarial variations on both natural language (i.e. MMLU-Redux) and symbolic reasoning tasks (i.e., GSM-Symbolic).
>
---
#### [replaced 048] Spotlight Attention: Towards Efficient LLM Generation via Non-linear Hashing-based KV Cache Retrieval
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.19740v2](http://arxiv.org/pdf/2508.19740v2)**

> **作者:** Wenhao Li; Yuxin Zhang; Gen Luo; Haiyuan Wan; Ziyang Gong; Fei Chao; Rongrong Ji
>
> **摘要:** Reducing the key-value (KV) cache burden in Large Language Models (LLMs) significantly accelerates inference. Dynamically selecting critical KV caches during decoding helps maintain performance. Existing methods use random linear hashing to identify important tokens, but this approach is inefficient due to the orthogonal distribution of queries and keys within two narrow cones in LLMs. We introduce Spotlight Attention, a novel method that employs non-linear hashing functions to optimize the embedding distribution of queries and keys, enhancing coding efficiency and robustness. We also developed a lightweight, stable training framework using a Bradley-Terry ranking-based loss, enabling optimization of the non-linear hashing module on GPUs with 16GB memory in 8 hours. Experimental results show that Spotlight Attention drastically improves retrieval precision while shortening the length of the hash code at least 5$\times$ compared to traditional linear hashing. Finally, we exploit the computational advantages of bitwise operations by implementing specialized CUDA kernels, achieving hashing retrieval for 512K tokens in under 100$\mu$s on a single A100 GPU, with end-to-end throughput up to 3$\times$ higher than vanilla decoding.
>
---
#### [replaced 049] EigenBench: A Comparative Behavioral Measure of Value Alignment
- **分类: cs.AI; cs.CL; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.01938v2](http://arxiv.org/pdf/2509.01938v2)**

> **作者:** Jonathn Chang; Leonhard Piff; Suvadip Sana; Jasmine X. Li; Lionel Levine
>
> **摘要:** Aligning AI with human values is a pressing unsolved problem. To address the lack of quantitative metrics for value alignment, we propose EigenBench: a black-box method for comparatively benchmarking language models' values. Given an ensemble of models, a constitution describing a value system, and a dataset of scenarios, our method returns a vector of scores quantifying each model's alignment to the given constitution. To produce these scores, each model judges the outputs of other models across many scenarios, and these judgments are aggregated with EigenTrust (Kamvar et al, 2003), yielding scores that reflect a weighted-average judgment of the whole ensemble. EigenBench uses no ground truth labels, as it is designed to quantify traits for which reasonable judges may disagree on the correct label. Using prompted personas, we test whether EigenBench scores are more sensitive to the model or the prompt: we find that most of the variance is explained by the prompt, but a small residual quantifies the disposition of the model itself.
>
---
#### [replaced 050] Extending FKG.in: Towards a Food Claim Traceability Network
- **分类: cs.AI; cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2508.16117v2](http://arxiv.org/pdf/2508.16117v2)**

> **作者:** Saransh Kumar Gupta; Rizwan Gulzar Mir; Lipika Dey; Partha Pratim Das; Anirban Sen; Ramesh Jain
>
> **备注:** 10 pages, 3 figures, 1 table, 45 references, ACM International Conference on Multimedia 2025 - Multi-modal Food Computing Workshop
>
> **摘要:** The global food landscape is rife with scientific, cultural, and commercial claims about what foods are, what they do, what they should not do, or should not do. These range from rigorously studied health benefits (probiotics improve gut health) and misrepresentations (soaked almonds make one smarter) to vague promises (superfoods boost immunity) and culturally rooted beliefs (cold foods cause coughs). Despite their widespread influence, the infrastructure for tracing, verifying, and contextualizing these claims remains fragmented and underdeveloped. In this paper, we propose a Food Claim-Traceability Network (FCN) as an extension of FKG[.]in, a knowledge graph of Indian food that we have been incrementally building. We also present the ontology design and the semi-automated knowledge curation workflow that we used to develop a proof of concept of FKG[.]in-FCN using Reddit data and Large Language Models. FCN integrates curated data inputs, structured schemas, and provenance-aware pipelines for food-related claim extraction and validation. While directly linked to the Indian food knowledge graph as an application, our methodology remains application-agnostic and adaptable to other geographic, culinary, or regulatory settings. By modeling food claims and their traceability in a structured, verifiable, and explainable way, we aim to contribute to more transparent and accountable food knowledge ecosystems, supporting researchers, policymakers, and most importantly, everyday consumers in navigating a world saturated with dietary assertions.
>
---
#### [replaced 051] Modular Techniques for Synthetic Long-Context Data Generation in Language Model Training and Evaluation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.01185v2](http://arxiv.org/pdf/2509.01185v2)**

> **作者:** Seganrasan Subramanian; Abhigya Verma
>
> **备注:** 26 pages, 4 figures
>
> **摘要:** The ability of large language models (LLMs) to process and reason over long textual inputs is critical for a wide range of real-world applications. However, progress in this area is significantly constrained by the absence of high-quality, diverse, and verifiable long-context datasets suitable for both training and evaluation. This work introduces a modular, extensible framework for synthetic long-context data generation via prompt-based interaction with LLMs. The framework supports multiple training and alignment objectives, including Supervised Fine-Tuning (SFT), Direct Preference Optimization (DPO), and Group Relative Policy Optimization (GRPO). It encompasses four core generation paradigms: multi-turn conversational dialogues, document-grounded input-output pairs, verifiable instruction-response tasks, and long-context reasoning examples. Through templated prompting, a model-agnostic architecture, and metadata-enriched outputs, the proposed approach facilitates scalable, controllable, and purpose-aligned dataset creation for advancing long-context capabilities in LLMs.
>
---
