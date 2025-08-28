# 自然语言处理 cs.CL

- **最新发布 81 篇**

- **更新 55 篇**

## 最新发布

#### [new 001] Beyond Shallow Heuristics: Leveraging Human Intuition for Curriculum Learning
- **分类: cs.CL**

- **简介: 该论文研究课程学习在语言模型预训练中的应用，解决语言难度定义问题，通过人类标注的简单数据设计课程，发现按课程顺序训练能提升模型性能，优于浅层启发式方法。**

- **链接: [http://arxiv.org/pdf/2508.19873v1](http://arxiv.org/pdf/2508.19873v1)**

> **作者:** Vanessa Toborek; Sebastian Müller; Tim Selbach; Tamás Horváth; Christian Bauckhage
>
> **备注:** Presented at ICNLSP 2025; to appear in the ACL Anthology; received the Best Short Paper Award
>
> **摘要:** Curriculum learning (CL) aims to improve training by presenting data from "easy" to "hard", yet defining and measuring linguistic difficulty remains an open challenge. We investigate whether human-curated simple language can serve as an effective signal for CL. Using the article-level labels from the Simple Wikipedia corpus, we compare label-based curricula to competence-based strategies relying on shallow heuristics. Our experiments with a BERT-tiny model show that adding simple data alone yields no clear benefit. However, structuring it via a curriculum -- especially when introduced first -- consistently improves perplexity, particularly on simple language. In contrast, competence-based curricula lead to no consistent gains over random ordering, probably because they fail to effectively separate the two classes. Our results suggest that human intuition about linguistic difficulty can guide CL for language model pre-training.
>
---
#### [new 002] LFD: Layer Fused Decoding to Exploit External Knowledge in Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对检索增强生成（RAG）任务中外部知识整合效率低的问题，提出Layer Fused Decoding（LFD）策略，通过融合中间层与最终层表征，结合内部知识得分（IKS）选择最优中间层，提升外部事实知识利用效果。**

- **链接: [http://arxiv.org/pdf/2508.19614v1](http://arxiv.org/pdf/2508.19614v1)**

> **作者:** Yang Sun; Lixin Zou; Dan Luo; Zhiyong Xie; Long Zhang; Liming Dong; Yunwei Zhao; Xixun Lin; Yanxiong Lu; Chenliang Li
>
> **摘要:** Retrieval-augmented generation (RAG) incorporates external knowledge into large language models (LLMs), improving their adaptability to downstream tasks and enabling information updates. Surprisingly, recent empirical evidence demonstrates that injecting noise into retrieved relevant documents paradoxically facilitates exploitation of external knowledge and improves generation quality. Although counterintuitive and challenging to apply in practice, this phenomenon enables granular control and rigorous analysis of how LLMs integrate external knowledge. Therefore, in this paper, we intervene on noise injection and establish a layer-specific functional demarcation within the LLM: shallow layers specialize in local context modeling, intermediate layers focus on integrating long-range external factual knowledge, and deeper layers primarily rely on parametric internal knowledge. Building on this insight, we propose Layer Fused Decoding (LFD), a simple decoding strategy that directly combines representations from an intermediate layer with final-layer decoding outputs to fully exploit the external factual knowledge. To identify the optimal intermediate layer, we introduce an internal knowledge score (IKS) criterion that selects the layer with the lowest IKS value in the latter half of layers. Experimental results across multiple benchmarks demonstrate that LFD helps RAG systems more effectively surface retrieved context knowledge with minimal cost.
>
---
#### [new 003] AI-Powered Detection of Inappropriate Language in Medical School Curricula
- **分类: cs.CL; cs.AI; cs.CY; I.2.1; I.2.7**

- **简介: 该论文任务为检测医学教材中的不当语言（IUL），解决手动筛查成本高的问题，通过评估小模型和大模型在IUL检测中的性能，发现小模型在特定场景下表现更优。**

- **链接: [http://arxiv.org/pdf/2508.19883v1](http://arxiv.org/pdf/2508.19883v1)**

> **作者:** Chiman Salavati; Shannon Song; Scott A. Hale; Roberto E. Montenegro; Shiri Dori-Hacohen; Fabricio Murai
>
> **备注:** Accepted at 2025 AAAI/ACM AI, Ethics and Society Conference (AIES'25)
>
> **摘要:** The use of inappropriate language -- such as outdated, exclusionary, or non-patient-centered terms -- medical instructional materials can significantly influence clinical training, patient interactions, and health outcomes. Despite their reputability, many materials developed over past decades contain examples now considered inappropriate by current medical standards. Given the volume of curricular content, manually identifying instances of inappropriate use of language (IUL) and its subcategories for systematic review is prohibitively costly and impractical. To address this challenge, we conduct a first-in-class evaluation of small language models (SLMs) fine-tuned on labeled data and pre-trained LLMs with in-context learning on a dataset containing approximately 500 documents and over 12,000 pages. For SLMs, we consider: (1) a general IUL classifier, (2) subcategory-specific binary classifiers, (3) a multilabel classifier, and (4) a two-stage hierarchical pipeline for general IUL detection followed by multilabel classification. For LLMs, we consider variations of prompts that include subcategory definitions and/or shots. We found that both LLama-3 8B and 70B, even with carefully curated shots, are largely outperformed by SLMs. While the multilabel classifier performs best on annotated data, supplementing training with unflagged excerpts as negative examples boosts the specific classifiers' AUC by up to 25%, making them most effective models for mitigating harmful language in medical curricula.
>
---
#### [new 004] Memory-R1: Enhancing Large Language Model Agents to Manage and Utilize Memories via Reinforcement Learning
- **分类: cs.CL; cs.MA**

- **简介: 该论文通过强化学习框架Memory-R1，解决LLM长周期推理受限问题，设计Memory Manager和Answer Agent，实现高效记忆管理与利用，提升模型性能。**

- **链接: [http://arxiv.org/pdf/2508.19828v1](http://arxiv.org/pdf/2508.19828v1)**

> **作者:** Sikuan Yan; Xiufeng Yang; Zuchao Huang; Ercong Nie; Zifeng Ding; Zonggen Li; Xiaowen Ma; Hinrich Schütze; Volker Tresp; Yunpu Ma
>
> **摘要:** Large Language Models (LLMs) have demonstrated impressive capabilities across a wide range of NLP tasks, but they remain fundamentally stateless, constrained by limited context windows that hinder long-horizon reasoning. Recent efforts to address this limitation often augment LLMs with an external memory bank, yet most existing pipelines are static and heuristic-driven, lacking any learned mechanism for deciding what to store, update, or retrieve. We present Memory-R1, a reinforcement learning (RL) framework that equips LLMs with the ability to actively manage and utilize external memory through two specialized agents: a Memory Manager that learns to perform structured memory operations {ADD, UPDATE, DELETE, NOOP}, and an Answer Agent that selects the most relevant entries and reasons over them to produce an answer. Both agents are fine-tuned with outcome-driven RL (PPO and GRPO), enabling adaptive memory management and use with minimal supervision. With as few as 152 question-answer pairs and a corresponding temporal memory bank for training, Memory-R1 outperforms the most competitive existing baseline and demonstrates strong generalization across diverse question types and LLM backbones. Beyond presenting an effective approach, this work provides insights into how RL can unlock more agentic, memory-aware behaviors in LLMs, pointing toward richer, more persistent reasoning systems.
>
---
#### [new 005] Uncovering the Bigger Picture: Comprehensive Event Understanding Via Diverse News Retrieval
- **分类: cs.CL; cs.IR**

- **简介: 论文提出NEWSCOPE框架，解决新闻检索中冗余与观点单一问题。通过两阶段方法：密集检索获取相关内容，结合句子级聚类与多样性重排，引入三个可解释指标评估多样性，提升事件理解的全面性。**

- **链接: [http://arxiv.org/pdf/2508.19758v1](http://arxiv.org/pdf/2508.19758v1)**

> **作者:** Yixuan Tang; Yuanyuan Shi; Yiqun Sun; Anthony Kum Hoe Tung
>
> **备注:** Accepted by EMNLP 2025
>
> **摘要:** Access to diverse perspectives is essential for understanding real-world events, yet most news retrieval systems prioritize textual relevance, leading to redundant results and limited viewpoint exposure. We propose NEWSCOPE, a two-stage framework for diverse news retrieval that enhances event coverage by explicitly modeling semantic variation at the sentence level. The first stage retrieves topically relevant content using dense retrieval, while the second stage applies sentence-level clustering and diversity-aware re-ranking to surface complementary information. To evaluate retrieval diversity, we introduce three interpretable metrics, namely Average Pairwise Distance, Positive Cluster Coverage, and Information Density Ratio, and construct two paragraph-level benchmarks: LocalNews and DSGlobal. Experiments show that NEWSCOPE consistently outperforms strong baselines, achieving significantly higher diversity without compromising relevance. Our results demonstrate the effectiveness of fine-grained, interpretable modeling in mitigating redundancy and promoting comprehensive event understanding. The data and code are available at https://github.com/tangyixuan/NEWSCOPE.
>
---
#### [new 006] Logical Reasoning with Outcome Reward Models for Test-Time Scaling
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对演绎逻辑推理任务，提出Outcome Reward Models（ORMs）以增强大语言模型在复杂推理中的性能。通过Chain-of-Thought与echo生成技术扩展训练数据，提升模型在FOLIO等数据集上的表现。**

- **链接: [http://arxiv.org/pdf/2508.19903v1](http://arxiv.org/pdf/2508.19903v1)**

> **作者:** Ramya Keerthy Thatikonda; Wray Buntine; Ehsan Shareghi
>
> **备注:** EMNLP 2025
>
> **摘要:** Logical reasoning is a critical benchmark for evaluating the capabilities of large language models (LLMs), as it reflects their ability to derive valid conclusions from given premises. While the combination of test-time scaling with dedicated outcome or process reward models has opened up new avenues to enhance LLMs performance in complex reasoning tasks, this space is under-explored in deductive logical reasoning. We present a set of Outcome Reward Models (ORMs) for deductive reasoning. To train the ORMs we mainly generate data using Chain-of-Thought (CoT) with single and multiple samples. Additionally, we propose a novel tactic to further expand the type of errors covered in the training dataset of the ORM. In particular, we propose an echo generation technique that leverages LLMs' tendency to reflect incorrect assumptions made in prompts to extract additional training data, covering previously unexplored error types. While a standard CoT chain may contain errors likely to be made by the reasoner, the echo strategy deliberately steers the model toward incorrect reasoning. We show that ORMs trained on CoT and echo-augmented data demonstrate improved performance on the FOLIO, JustLogic, and ProverQA datasets across four different LLMs.
>
---
#### [new 007] Whisper based Cross-Lingual Phoneme Recognition between Vietnamese and English
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对越南语与英语发音混合时的ASR音素对齐难题，提出双语音素集构建与端到端系统，利用PhoWhisper编码器提升跨语言音素识别准确性。**

- **链接: [http://arxiv.org/pdf/2508.19270v1](http://arxiv.org/pdf/2508.19270v1)**

> **作者:** Nguyen Huu Nhat Minh; Tran Nguyen Anh; Truong Dinh Dung; Vo Van Nam; Le Pham Tuyen
>
> **摘要:** Cross-lingual phoneme recognition has emerged as a significant challenge for accurate automatic speech recognition (ASR) when mixing Vietnamese and English pronunciations. Unlike many languages, Vietnamese relies on tonal variations to distinguish word meanings, whereas English features stress patterns and non-standard pronunciations that hinder phoneme alignment between the two languages. To address this challenge, we propose a novel bilingual speech recognition approach with two primary contributions: (1) constructing a representative bilingual phoneme set that bridges the differences between Vietnamese and English phonetic systems; (2) designing an end-to-end system that leverages the PhoWhisper pre-trained encoder for deep high-level representations to improve phoneme recognition. Our extensive experiments demonstrate that the proposed approach not only improves recognition accuracy in bilingual speech recognition for Vietnamese but also provides a robust framework for addressing the complexities of tonal and stress-based phoneme recognition
>
---
#### [new 008] Rethinking Reasoning in LLMs: Neuro-Symbolic Local RetoMaton Beyond ICL and CoT
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对LLM推理稳定性不足的问题，提出神经符号学本地RetoMaton，通过Weighted Finite Automaton提升可解释性与性能，实验验证其在三个任务上的有效性。**

- **链接: [http://arxiv.org/pdf/2508.19271v1](http://arxiv.org/pdf/2508.19271v1)**

> **作者:** Rushitha Santhoshi Mamidala; Anshuman Chhabra; Ankur Mali
>
> **摘要:** Prompt-based reasoning strategies such as Chain-of-Thought (CoT) and In-Context Learning (ICL) have become widely used for eliciting reasoning capabilities in large language models (LLMs). However, these methods rely on fragile, implicit mechanisms often yielding inconsistent outputs across seeds, formats, or minor prompt variations making them fundamentally unreliable for tasks requiring stable, interpretable reasoning. In contrast, automata-based neuro-symbolic frameworks like RetoMaton offer a more structured and trustworthy alternative by grounding retrieval in symbolic memory with deterministic transitions. In this work, we extend RetoMaton by replacing its global datastore with a local, task-adaptive Weighted Finite Automaton (WFA), constructed directly from external domain corpora. This local automaton structure promotes robust, context-aware retrieval while preserving symbolic traceability and low inference overhead. Unlike prompting, which entangles context and memory in opaque ways, our approach leverages the explicit structure of WFAs to provide verifiable and modular retrieval behavior, making it better suited for domain transfer and interoperability. We evaluate this local RetoMaton variant on two pretrained LLMs LLaMA-3.2-1B and Gemma-3-1B-PT across three reasoning tasks: TriviaQA (reading comprehension), GSM8K (multi-step math), and MMLU (domain knowledge). Compared to the base model and prompting-based methods, augmenting these setups with local RetoMaton consistently improves performance while enabling transparent and reproducible retrieval dynamics. Our results highlight a promising shift toward trustworthy, symbolic reasoning in modern LLMs via lightweight, automaton-guided memory.
>
---
#### [new 009] DeepScholar-Bench: A Live Benchmark and Automated Evaluation for Generative Research Synthesis
- **分类: cs.CL; cs.AI**

- **简介: 论文提出DeepScholar-bench，用于评估生成式研究综合（如生成论文相关工作）。解决现有基准无法有效评估动态、复杂任务的问题，通过实时数据和多维度评估，系统比较不同系统性能。**

- **链接: [http://arxiv.org/pdf/2508.20033v1](http://arxiv.org/pdf/2508.20033v1)**

> **作者:** Liana Patel; Negar Arabzadeh; Harshit Gupta; Ankita Sundar; Ion Stoica; Matei Zaharia; Carlos Guestrin
>
> **摘要:** The ability to research and synthesize knowledge is central to human expertise and progress. An emerging class of systems promises these exciting capabilities through generative research synthesis, performing retrieval over the live web and synthesizing discovered sources into long-form, cited summaries. However, evaluating such systems remains an open challenge: existing question-answering benchmarks focus on short-form factual responses, while expert-curated datasets risk staleness and data contamination. Both fail to capture the complexity and evolving nature of real research synthesis tasks. In this work, we introduce DeepScholar-bench, a live benchmark and holistic, automated evaluation framework designed to evaluate generative research synthesis. DeepScholar-bench draws queries from recent, high-quality ArXiv papers and focuses on a real research synthesis task: generating the related work sections of a paper by retrieving, synthesizing, and citing prior research. Our evaluation framework holistically assesses performance across three key dimensions, knowledge synthesis, retrieval quality, and verifiability. We also develop DeepScholar-base, a reference pipeline implemented efficiently using the LOTUS API. Using the DeepScholar-bench framework, we perform a systematic evaluation of prior open-source systems, search AI's, OpenAI's DeepResearch, and DeepScholar-base. We find that DeepScholar-base establishes a strong baseline, attaining competitive or higher performance than each other method. We also find that DeepScholar-bench remains far from saturated, with no system exceeding a score of $19\%$ across all metrics. These results underscore the difficulty of DeepScholar-bench, as well as its importance for progress towards AI systems capable of generative research synthesis. We make our code available at https://github.com/guestrin-lab/deepscholar-bench.
>
---
#### [new 010] NLKI: A lightweight Natural Language Knowledge Integration Framework for Improving Small VLMs in Commonsense VQA Tasks
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对小规模视觉语言模型（sVLMs）在常识VQA任务中因缺乏外部知识导致的性能不足问题，提出NLKI框架，通过整合常识知识提升回答准确性，实验表明其效果可与大模型相当。**

- **链接: [http://arxiv.org/pdf/2508.19724v1](http://arxiv.org/pdf/2508.19724v1)**

> **作者:** Aritra Dutta; Swapnanil Mukherjee; Deepanway Ghosal; Somak Aditya
>
> **摘要:** Commonsense visual-question answering often hinges on knowledge that is missing from the image or the question. Small vision-language models (sVLMs) such as ViLT, VisualBERT and FLAVA therefore lag behind their larger generative counterparts. To study the effect of careful commonsense knowledge integration on sVLMs, we present an end-to-end framework (NLKI) that (i) retrieves natural language facts, (ii) prompts an LLM to craft natural language explanations, and (iii) feeds both signals to sVLMs respectively across two commonsense VQA datasets (CRIC, AOKVQA) and a visual-entailment dataset (e-SNLI-VE). Facts retrieved using a fine-tuned ColBERTv2 and an object information-enriched prompt yield explanations that largely cut down hallucinations, while lifting the end-to-end answer accuracy by up to 7% (across 3 datasets), making FLAVA and other models in NLKI match or exceed medium-sized VLMs such as Qwen-2 VL-2B and SmolVLM-2.5B. As these benchmarks contain 10-25% label noise, additional finetuning using noise-robust losses (such as symmetric cross entropy and generalised cross entropy) adds another 2.5% in CRIC, and 5.5% in AOKVQA. Our findings expose when LLM-based commonsense knowledge beats retrieval from commonsense knowledge bases, how noise-aware training stabilises small models in the context of external knowledge augmentation, and why parameter-efficient commonsense reasoning is now within reach for 250M models.
>
---
#### [new 011] Principled Personas: Defining and Measuring the Intended Effects of Persona Prompting on Task Performance
- **分类: cs.CL**

- **简介: 该论文研究角色提示对任务表现的影响，分析其有效性，评估9个LLM在27项任务中的表现，发现专家角色效果不一且易受无关属性干扰，提出改进策略以提升鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.19764v1](http://arxiv.org/pdf/2508.19764v1)**

> **作者:** Pedro Henrique Luz de Araujo; Paul Röttger; Dirk Hovy; Benjamin Roth
>
> **备注:** 30 pages, 29 figures, accepted to EMNLP 2025
>
> **摘要:** Expert persona prompting -- assigning roles such as expert in math to language models -- is widely used for task improvement. However, prior work shows mixed results on its effectiveness, and does not consider when and why personas should improve performance. We analyze the literature on persona prompting for task improvement and distill three desiderata: 1) performance advantage of expert personas, 2) robustness to irrelevant persona attributes, and 3) fidelity to persona attributes. We then evaluate 9 state-of-the-art LLMs across 27 tasks with respect to these desiderata. We find that expert personas usually lead to positive or non-significant performance changes. Surprisingly, models are highly sensitive to irrelevant persona details, with performance drops of almost 30 percentage points. In terms of fidelity, we find that while higher education, specialization, and domain-relatedness can boost performance, their effects are often inconsistent or negligible across tasks. We propose mitigation strategies to improve robustness -- but find they only work for the largest, most capable models. Our findings underscore the need for more careful persona design and for evaluation schemes that reflect the intended effects of persona usage.
>
---
#### [new 012] Inference Gap in Domain Expertise and Machine Intelligence in Named Entity Recognition: Creation of and Insights from a Substance Use-related Dataset
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 论文提出基于RedditImpacts 2.0数据集的NER框架，用于提取物质使用相关文本中的临床和社会影响实体，对比发现微调DeBERTa模型在精度和指南遵循上优于LLM，但仍未达专家共识水平，揭示领域知识与AI能力间的差距。**

- **链接: [http://arxiv.org/pdf/2508.19467v1](http://arxiv.org/pdf/2508.19467v1)**

> **作者:** Sumon Kanti Dey; Jeanne M. Powell; Azra Ismail; Jeanmarie Perrone; Abeed Sarker
>
> **备注:** Dataset and code: https://github.com/SumonKantiDey/Reddit_Impacts_NER
>
> **摘要:** Nonmedical opioid use is an urgent public health challenge, with far-reaching clinical and social consequences that are often underreported in traditional healthcare settings. Social media platforms, where individuals candidly share first-person experiences, offer a valuable yet underutilized source of insight into these impacts. In this study, we present a named entity recognition (NER) framework to extract two categories of self-reported consequences from social media narratives related to opioid use: ClinicalImpacts (e.g., withdrawal, depression) and SocialImpacts (e.g., job loss). To support this task, we introduce RedditImpacts 2.0, a high-quality dataset with refined annotation guidelines and a focus on first-person disclosures, addressing key limitations of prior work. We evaluate both fine-tuned encoder-based models and state-of-the-art large language models (LLMs) under zero- and few-shot in-context learning settings. Our fine-tuned DeBERTa-large model achieves a relaxed token-level F1 of 0.61 [95% CI: 0.43-0.62], consistently outperforming LLMs in precision, span accuracy, and adherence to task-specific guidelines. Furthermore, we show that strong NER performance can be achieved with substantially less labeled data, emphasizing the feasibility of deploying robust models in resource-limited settings. Our findings underscore the value of domain-specific fine-tuning for clinical NLP tasks and contribute to the responsible development of AI tools that may enhance addiction surveillance, improve interpretability, and support real-world healthcare decision-making. The best performing model, however, still significantly underperforms compared to inter-expert agreement (Cohen's kappa: 0.81), demonstrating that a gap persists between expert intelligence and current state-of-the-art NER/AI capabilities for tasks requiring deep domain knowledge.
>
---
#### [new 013] Leveraging Language Models and Machine Learning in Verbal Autopsy Analysis
- **分类: cs.CL**

- **简介: 本论文研究利用预训练语言模型和机器学习改进语音自主分析中的死因分类，通过结合叙述和问题数据，提升非传染性疾病识别效果，并分析信息充分性对分类的影响。**

- **链接: [http://arxiv.org/pdf/2508.19274v1](http://arxiv.org/pdf/2508.19274v1)**

> **作者:** Yue Chu
>
> **备注:** Ph.D. dissertation submitted to The Ohio State University, August 2025
>
> **摘要:** In countries without civil registration and vital statistics, verbal autopsy (VA) is a critical tool for estimating cause of death (COD) and inform policy priorities. In VA, interviewers ask proximal informants for details on the circumstances preceding a death, in the form of unstructured narratives and structured questions. Existing automated VA cause classification algorithms only use the questions and ignore the information in the narratives. In this thesis, we investigate how the VA narrative can be used for automated COD classification using pretrained language models (PLMs) and machine learning (ML) techniques. Using empirical data from South Africa, we demonstrate that with the narrative alone, transformer-based PLMs with task-specific fine-tuning outperform leading question-only algorithms at both the individual and population levels, particularly in identifying non-communicable diseases. We explore various multimodal fusion strategies combining narratives and questions in unified frameworks. Multimodal approaches further improve performance in COD classification, confirming that each modality has unique contributions and may capture valuable information that is not present in the other modality. We also characterize physician-perceived information sufficiency in VA. We describe variations in sufficiency levels by age and COD and demonstrate that classification accuracy is affected by sufficiency for both physicians and models. Overall, this thesis advances the growing body of knowledge at the intersection of natural language processing, epidemiology, and global health. It demonstrates the value of narrative in enhancing COD classification. Our findings underscore the need for more high-quality data from more diverse settings to use in training and fine-tuning PLM/ML methods, and offer valuable insights to guide the rethinking and redesign of the VA instrument and interview.
>
---
#### [new 014] Database Entity Recognition with Data Augmentation and Deep Learning
- **分类: cs.CL; cs.AI; cs.DB; cs.LG**

- **简介: 论文针对自然语言查询中的数据库实体识别问题，提出数据增强和基于T5的深度学习方法，构建基准数据集并通过实验验证效果。**

- **链接: [http://arxiv.org/pdf/2508.19372v1](http://arxiv.org/pdf/2508.19372v1)**

> **作者:** Zikun Fu; Chen Yang; Kourosh Davoudi; Ken Q. Pu
>
> **备注:** 6 pages, 5 figures. Accepted at IEEE 26th International Conference on Information Reuse and Integration for Data Science (IRI 2025), San Jose, California, August 6-8, 2025
>
> **摘要:** This paper addresses the challenge of Database Entity Recognition (DB-ER) in Natural Language Queries (NLQ). We present several key contributions to advance this field: (1) a human-annotated benchmark for DB-ER task, derived from popular text-to-sql benchmarks, (2) a novel data augmentation procedure that leverages automatic annotation of NLQs based on the corresponding SQL queries which are available in popular text-to-SQL benchmarks, (3) a specialized language model based entity recognition model using T5 as a backbone and two down-stream DB-ER tasks: sequence tagging and token classification for fine-tuning of backend and performing DB-ER respectively. We compared our DB-ER tagger with two state-of-the-art NER taggers, and observed better performance in both precision and recall for our model. The ablation evaluation shows that data augmentation boosts precision and recall by over 10%, while fine-tuning of the T5 backbone boosts these metrics by 5-10%.
>
---
#### [new 015] RAGAPHENE: A RAG Annotation Platform with Human Enhancements and Edits
- **分类: cs.CL**

- **简介: 该论文提出RAGAPHENE平台，用于构建多轮RAG对话的评估基准，解决LLM事实正确性问题，通过人类模拟真实对话生成高质量数据。**

- **链接: [http://arxiv.org/pdf/2508.19272v1](http://arxiv.org/pdf/2508.19272v1)**

> **作者:** Kshitij Fadnis; Sara Rosenthal; Maeda Hanafi; Yannis Katsis; Marina Danilevsky
>
> **摘要:** Retrieval Augmented Generation (RAG) is an important aspect of conversing with Large Language Models (LLMs) when factually correct information is important. LLMs may provide answers that appear correct, but could contain hallucinated information. Thus, building benchmarks that can evaluate LLMs on multi-turn RAG conversations has become an increasingly important task. Simulating real-world conversations is vital for producing high quality evaluation benchmarks. We present RAGAPHENE, a chat-based annotation platform that enables annotators to simulate real-world conversations for benchmarking and evaluating LLMs. RAGAPHENE has been successfully used by approximately 40 annotators to build thousands of real-world conversations.
>
---
#### [new 016] ReSURE: Regularizing Supervision Unreliability for Multi-turn Dialogue Fine-tuning
- **分类: cs.CL**

- **简介: 该论文针对多轮对话微调中低质量数据导致的监督错误传播问题，提出ReSURE方法，通过动态调整监督权重以提升模型稳定性与响应质量。**

- **链接: [http://arxiv.org/pdf/2508.19996v1](http://arxiv.org/pdf/2508.19996v1)**

> **作者:** Yiming Du; Yifan Xiang; Bin Liang; Dahua Lin; Kam-Fai Wong; Fei Tan
>
> **摘要:** Fine-tuning multi-turn dialogue systems requires high-quality supervision but often suffers from degraded performance when exposed to low-quality data. Supervision errors in early turns can propagate across subsequent turns, undermining coherence and response quality. Existing methods typically address data quality via static prefiltering, which decouples quality control from training and fails to mitigate turn-level error propagation. In this context, we propose ReSURE (Regularizing Supervision UnREliability), an adaptive learning method that dynamically down-weights unreliable supervision without explicit filtering. ReSURE estimates per-turn loss distributions using Welford's online statistics and reweights sample losses on the fly accordingly. Experiments on both single-source and mixed-quality datasets show improved stability and response quality. Notably, ReSURE enjoys positive Spearman correlations (0.21 ~ 1.0 across multiple benchmarks) between response scores and number of samples regardless of data quality, which potentially paves the way for utilizing large-scale data effectively. Code is publicly available at https://github.com/Elvin-Yiming-Du/ReSURE_Multi_Turn_Training.
>
---
#### [new 017] Diffusion Language Models Know the Answer Before Decoding
- **分类: cs.CL; cs.AI**

- **简介: 论文研究DLM解码优化任务，解决其推理速度慢问题。通过Prophet方法，利用早期答案收敛特性，在无需训练的情况下，动态决策提前终止细化或直接解码，减少步骤数3.4倍，提升效率。**

- **链接: [http://arxiv.org/pdf/2508.19982v1](http://arxiv.org/pdf/2508.19982v1)**

> **作者:** Pengxiang Li; Yefan Zhou; Dilxat Muhtar; Lu Yin; Shilin Yan; Li Shen; Yi Liang; Soroush Vosoughi; Shiwei Liu
>
> **摘要:** Diffusion language models (DLMs) have recently emerged as an alternative to autoregressive approaches, offering parallel sequence generation and flexible token orders. However, their inference remains slower than that of autoregressive models, primarily due to the cost of bidirectional attention and the large number of refinement steps required for high quality outputs. In this work, we highlight and leverage an overlooked property of DLMs early answer convergence: in many cases, the correct answer can be internally identified by half steps before the final decoding step, both under semi-autoregressive and random remasking schedules. For example, on GSM8K and MMLU, up to 97% and 99% of instances, respectively, can be decoded correctly using only half of the refinement steps. Building on this observation, we introduce Prophet, a training-free fast decoding paradigm that enables early commit decoding. Specifically, Prophet dynamically decides whether to continue refinement or to go "all-in" (i.e., decode all remaining tokens in one step), using the confidence gap between the top-2 prediction candidates as the criterion. It integrates seamlessly into existing DLM implementations, incurs negligible overhead, and requires no additional training. Empirical evaluations of LLaDA-8B and Dream-7B across multiple tasks show that Prophet reduces the number of decoding steps by up to 3.4x while preserving high generation quality. These results recast DLM decoding as a problem of when to stop sampling, and demonstrate that early decode convergence provides a simple yet powerful mechanism for accelerating DLM inference, complementary to existing speedup techniques. Our code is publicly available at https://github.com/pixeli99/Prophet.
>
---
#### [new 018] Continuously Steering LLMs Sensitivity to Contextual Knowledge with Proxy Models
- **分类: cs.CL**

- **简介: 该论文针对LLM对上下文知识敏感度调整问题，提出CSKS框架，通过代理模型调整输出分布，实现无需修改权重的连续控制，提升灵活性。**

- **链接: [http://arxiv.org/pdf/2508.19720v1](http://arxiv.org/pdf/2508.19720v1)**

> **作者:** Yilin Wang; Heng Wang; Yuyang Bai; Minnan Luo
>
> **摘要:** In Large Language Models (LLMs) generation, there exist knowledge conflicts and scenarios where parametric knowledge contradicts knowledge provided in the context. Previous works studied tuning, decoding algorithms, or locating and editing context-aware neurons to adapt LLMs to be faithful to new contextual knowledge. However, they are usually inefficient or ineffective for large models, not workable for black-box models, or unable to continuously adjust LLMs' sensitivity to the knowledge provided in the context. To mitigate these problems, we propose CSKS (Continuously Steering Knowledge Sensitivity), a simple framework that can steer LLMs' sensitivity to contextual knowledge continuously at a lightweight cost. Specifically, we tune two small LMs (i.e. proxy models) and use the difference in their output distributions to shift the original distribution of an LLM without modifying the LLM weights. In the evaluation process, we not only design synthetic data and fine-grained metrics to measure models' sensitivity to contextual knowledge but also use a real conflict dataset to validate CSKS's practical efficacy. Extensive experiments demonstrate that our framework achieves continuous and precise control over LLMs' sensitivity to contextual knowledge, enabling both increased sensitivity and reduced sensitivity, thereby allowing LLMs to prioritize either contextual or parametric knowledge as needed flexibly. Our data and code are available at https://github.com/OliveJuiceLin/CSKS.
>
---
#### [new 019] CORE: Lossless Compression for Retrieval-Augmented LLMs via Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 论文针对检索增强型LLM的无损上下文压缩问题，通过强化学习优化压缩过程，利用端任务性能作为奖励信号，提升答案准确率。**

- **链接: [http://arxiv.org/pdf/2508.19282v1](http://arxiv.org/pdf/2508.19282v1)**

> **作者:** Ziqiang Cui; Yunpeng Weng; Xing Tang; Peiyang Liu; Shiwei Li; Bowei He; Jiamin Chen; Xiuqiang He; Chen Ma
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a promising approach to enhance the timeliness of knowledge and the factual accuracy of responses in Large Language Models (LLMs). However, the inclusion of excessive retrieved documents substantially increases the input length, leading to higher computational costs. Previous studies have attempted to compress retrieved documents into shorter texts before in-context integration, but such methods often compromise end-task performance. The lack of well-defined compression targets forces many approaches to rely on fixed heuristics, which cannot guarantee that the compressed content will effectively support the end task. To address these limitations, we propose CORE, a novel method designed to achieve lossless context compression for RAG. CORE employs reinforcement learning to optimize the compression process without relying on predefined compression labels. Specifically, it utilizes end-task performance as a reward signal and applies Generalized Reinforcement Learning Policy Optimization (GRPO) to train the compressor. This end-to-end training framework enables the compressor to generate summaries that maximize the accuracy of answers generated by the LLM. Extensive experiments on four datasets demonstrate the superiority of our approach. With a high compression ratio of 3\%, our method not only avoids performance degradation compared to prepending full documents across all datasets but also improves the average Exact Match (EM) score by 3.3 points. The code will be released soon.
>
---
#### [new 020] Emotion Transfer with Enhanced Prototype for Unseen Emotion Recognition in Conversation
- **分类: cs.CL**

- **简介: 论文提出UERC任务，解决未见情感识别问题，设计原型迁移框架，改进描述、编码与解码策略，实验验证有效性。**

- **链接: [http://arxiv.org/pdf/2508.19533v1](http://arxiv.org/pdf/2508.19533v1)**

> **作者:** Kun Peng; Cong Cao; Hao Peng; Guanlin Wu; Zhifeng Hao; Lei Jiang; Yanbing Liu; Philip S. Yu
>
> **备注:** Accepted at EMNLP2025
>
> **摘要:** Current Emotion Recognition in Conversation (ERC) research follows a closed-domain assumption. However, there is no clear consensus on emotion classification in psychology, which presents a challenge for models when it comes to recognizing previously unseen emotions in real-world applications. To bridge this gap, we introduce the Unseen Emotion Recognition in Conversation (UERC) task for the first time and propose ProEmoTrans, a solid prototype-based emotion transfer framework. This prototype-based approach shows promise but still faces key challenges: First, implicit expressions complicate emotion definition, which we address by proposing an LLM-enhanced description approach. Second, utterance encoding in long conversations is difficult, which we tackle with a proposed parameter-free mechanism for efficient encoding and overfitting prevention. Finally, the Markovian flow nature of emotions is hard to transfer, which we address with an improved Attention Viterbi Decoding (AVD) method to transfer seen emotion transitions to unseen emotions. Extensive experiments on three datasets show that our method serves as a strong baseline for preliminary exploration in this new area.
>
---
#### [new 021] Towards stable AI systems for Evaluating Arabic Pronunciations
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在提升阿拉伯语孤立字母发音评估的AI系统稳定性。针对现有ASR系统在孤立字母分类中的低准确率问题，构建了带变音符号的语料库，通过轻量神经网络和对抗训练优化模型，提升鲁棒性，为后续词句级应用奠定基础。**

- **链接: [http://arxiv.org/pdf/2508.19587v1](http://arxiv.org/pdf/2508.19587v1)**

> **作者:** Hadi Zaatiti; Hatem Hajri; Osama Abdullah; Nader Masmoudi
>
> **摘要:** Modern Arabic ASR systems such as wav2vec 2.0 excel at word- and sentence-level transcription, yet struggle to classify isolated letters. In this study, we show that this phoneme-level task, crucial for language learning, speech therapy, and phonetic research, is challenging because isolated letters lack co-articulatory cues, provide no lexical context, and last only a few hundred milliseconds. Recogniser systems must therefore rely solely on variable acoustic cues, a difficulty heightened by Arabic's emphatic (pharyngealized) consonants and other sounds with no close analogues in many languages. This study introduces a diverse, diacritised corpus of isolated Arabic letters and demonstrates that state-of-the-art wav2vec 2.0 models achieve only 35% accuracy on it. Training a lightweight neural network on wav2vec embeddings raises performance to 65%. However, adding a small amplitude perturbation (epsilon = 0.05) cuts accuracy to 32%. To restore robustness, we apply adversarial training, limiting the noisy-speech drop to 9% while preserving clean-speech accuracy. We detail the corpus, training pipeline, and evaluation protocol, and release, on demand, data and code for reproducibility. Finally, we outline future work extending these methods to word- and sentence-level frameworks, where precise letter pronunciation remains critical.
>
---
#### [new 022] Towards a Holistic and Automated Evaluation Framework for Multi-Level Comprehension of LLMs in Book-Length Contexts
- **分类: cs.CL; cs.AI**

- **简介: 论文提出HAMLET框架，用于评估LLMs在长文本中多级理解能力，通过三级关键事实结构和自动化查询摘要验证评估可靠性，揭示模型在细粒度理解及跨模型性能差异上的不足。**

- **链接: [http://arxiv.org/pdf/2508.19578v1](http://arxiv.org/pdf/2508.19578v1)**

> **作者:** Jiaqi Deng; Yuho Lee; Nicole Hee-Yeon Kim; Hyangsuk Min; Taewon Yun; Minjeong Ban; Kim Yul; Hwanjun Song
>
> **备注:** Accepted to EMNLP 2025 (Main)
>
> **摘要:** We introduce HAMLET, a holistic and automated framework for evaluating the long-context comprehension of large language models (LLMs). HAMLET structures source texts into a three-level key-fact hierarchy at root-, branch-, and leaf-levels, and employs query-focused summarization to evaluate how well models recall and faithfully represent information at each level. To validate the reliability of our fully automated pipeline, we conduct a systematic human study, showing that our automatic evaluation achieves over 90% agreement with expert human judgments, while reducing the cost by up to 25 times. HAMLET reveals that LLMs struggle with fine-grained comprehension, especially at the leaf level, and are sensitive to positional effects like the lost-in-the-middle. Analytical queries pose greater challenges than narrative ones, and consistent performance gaps emerge between open-source and proprietary models, as well as across model scales. Our code and dataset are publicly available at https://github.com/DISL-Lab/HAMLET.
>
---
#### [new 023] A Symbolic Adversarial Learning Framework for Evolving Fake News Generation and Detection
- **分类: cs.CL**

- **简介: 论文提出SALF框架，通过对抗训练和符号学习优化，解决动态演变假新闻的生成与检测难题，提升检测效果。**

- **链接: [http://arxiv.org/pdf/2508.19633v1](http://arxiv.org/pdf/2508.19633v1)**

> **作者:** Chong Tian; Qirong Ho; Xiuying Chen
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Rapid LLM advancements heighten fake news risks by enabling the automatic generation of increasingly sophisticated misinformation. Previous detection methods, including fine-tuned small models or LLM-based detectors, often struggle with its dynamically evolving nature. In this work, we propose a novel framework called the Symbolic Adversarial Learning Framework (SALF), which implements an adversarial training paradigm by an agent symbolic learning optimization process, rather than relying on numerical updates. SALF introduces a paradigm where the generation agent crafts deceptive narratives, and the detection agent uses structured debates to identify logical and factual flaws for detection, and they iteratively refine themselves through such adversarial interactions. Unlike traditional neural updates, we represent agents using agent symbolic learning, where learnable weights are defined by agent prompts, and simulate back-propagation and gradient descent by operating on natural language representations of weights, loss, and gradients. Experiments on two multilingual benchmark datasets demonstrate SALF's effectiveness, showing it generates sophisticated fake news that degrades state-of-the-art detection performance by up to 53.4% in Chinese and 34.2% in English on average. SALF also refines detectors, improving detection of refined content by up to 7.7%. We hope our work inspires further exploration into more robust, adaptable fake news detection systems.
>
---
#### [new 024] Spotlight Attention: Towards Efficient LLM Generation via Non-linear Hashing-based KV Cache Retrieval
- **分类: cs.CL**

- **简介: 该论文针对LLM生成中的KV缓存效率问题，提出非线性哈希方法优化检索，通过轻量训练框架和CUDA加速，实现512K tokens/100μs的高效检索，提升推理速度3倍。**

- **链接: [http://arxiv.org/pdf/2508.19740v1](http://arxiv.org/pdf/2508.19740v1)**

> **作者:** Wenhao Li; Yuxin Zhang; Gen Luo; Haiyuan Wan; Ziyang Gong; Fei Chao; Rongrong Ji
>
> **摘要:** Reducing the key-value (KV) cache burden in Large Language Models (LLMs) significantly accelerates inference. Dynamically selecting critical KV caches during decoding helps maintain performance. Existing methods use random linear hashing to identify important tokens, but this approach is inefficient due to the orthogonal distribution of queries and keys within two narrow cones in LLMs. We introduce Spotlight Attention, a novel method that employs non-linear hashing functions to optimize the embedding distribution of queries and keys, enhancing coding efficiency and robustness. We also developed a lightweight, stable training framework using a Bradley-Terry ranking-based loss, enabling optimization of the non-linear hashing module on GPUs with 16GB memory in 8 hours. Experimental results show that Spotlight Attention drastically improves retrieval precision while shortening the length of the hash code at least 5$\times$ compared to traditional linear hashing. Finally, we exploit the computational advantages of bitwise operations by implementing specialized CUDA kernels, achieving hashing retrieval for 512K tokens in under 100$\mu$s on a single A100 GPU, with end-to-end throughput up to 3$\times$ higher than vanilla decoding.
>
---
#### [new 025] Improving Low-Resource Translation with Dictionary-Guided Fine-Tuning and RL: A Spanish-to-Wayuunaiki Study
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对低资源语言翻译任务，解决LLM预训练和细调数据不足的问题，提出结合外部词典与强化学习的端到端方法，通过BLEU奖励提升西班牙-Wayuunaiki翻译质量，实现3.37 BLEU改进。**

- **链接: [http://arxiv.org/pdf/2508.19481v1](http://arxiv.org/pdf/2508.19481v1)**

> **作者:** Manuel Mosquera; Melissa Robles; Johan Rodriguez; Ruben Manrique
>
> **摘要:** Low-resource machine translation remains a significant challenge for large language models (LLMs), which often lack exposure to these languages during pretraining and have limited parallel data for fine-tuning. We propose a novel approach that enhances translation for low-resource languages by integrating an external dictionary tool and training models end-to-end using reinforcement learning, in addition to supervised fine-tuning. Focusing on the Spanish-Wayuunaiki language pair, we frame translation as a tool-augmented decision-making problem in which the model can selectively consult a bilingual dictionary during generation. Our method combines supervised instruction tuning with Guided Reward Policy Optimization (GRPO), enabling the model to learn both when and how to use the tool effectively. BLEU similarity scores are used as rewards to guide this learning process. Preliminary results show that our tool-augmented models achieve up to +3.37 BLEU improvement over previous work, and a 18% relative gain compared to a supervised baseline without dictionary access, on the Spanish-Wayuunaiki test set from the AmericasNLP 2025 Shared Task. We also conduct ablation studies to assess the effects of model architecture and training strategy, comparing Qwen2.5-0.5B-Instruct with other models such as LLaMA and a prior NLLB-based system. These findings highlight the promise of combining LLMs with external tools and the role of reinforcement learning in improving translation quality in low-resource language settings.
>
---
#### [new 026] FLAIRR-TS -- Forecasting LLM-Agents with Iterative Refinement and Retrieval for Time Series
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.19279v1](http://arxiv.org/pdf/2508.19279v1)**

> **作者:** Gunjan Jalori; Preetika Verma; Sercan Ö Arık
>
> **备注:** EMNLP
>
> **摘要:** Time series Forecasting with large languagemodels (LLMs) requires bridging numericalpatterns and natural language. Effective fore-casting on LLM often relies on extensive pre-processing and fine-tuning.Recent studiesshow that a frozen LLM can rival specializedforecasters when supplied with a carefully en-gineered natural-language prompt, but craft-ing such a prompt for each task is itself oner-ous and ad-hoc. We introduce FLAIRR-TS, atest-time prompt optimization framework thatutilizes an agentic system: a Forecaster-agentgenerates forecasts using an initial prompt,which is then refined by a refiner agent, in-formed by past outputs and retrieved analogs.This adaptive prompting generalizes across do-mains using creative prompt templates andgenerates high-quality forecasts without inter-mediate code generation.Experiments onbenchmark datasets show improved accuracyover static prompting and retrieval-augmentedbaselines, approaching the performance ofspecialized prompts.FLAIRR-TS providesa practical alternative to tuning, achievingstrong performance via its agentic approach toadaptive prompt refinement and retrieval.
>
---
#### [new 027] AraHealthQA 2025 Shared Task Description Paper
- **分类: cs.CL**

- **简介: 该论文提出AraHealthQA 2025共享任务，旨在解决阿拉伯语医疗问答资源匮乏问题，通过设计MentalQA和MedArabiQ两个赛道，构建多子任务数据集与评估框架，推动多语言医疗NLP研究。**

- **链接: [http://arxiv.org/pdf/2508.20047v1](http://arxiv.org/pdf/2508.20047v1)**

> **作者:** Hassan Alhuzali; Farah Shamout; Muhammad Abdul-Mageed; Chaimae Abouzahir; Mouath Abu-Daoud; Ashwag Alasmari; Walid Al-Eisawi; Renad Al-Monef; Ali Alqahtani; Lama Ayash; Nizar Habash; Leen Kharouf
>
> **摘要:** We introduce {AraHealthQA 2025}, the {Comprehensive Arabic Health Question Answering Shared Task}, held in conjunction with {ArabicNLP 2025} (co-located with EMNLP 2025). This shared task addresses the paucity of high-quality Arabic medical QA resources by offering two complementary tracks: {MentalQA}, focusing on Arabic mental health Q\&A (e.g., anxiety, depression, stigma reduction), and {MedArabiQ}, covering broader medical domains such as internal medicine, pediatrics, and clinical decision making. Each track comprises multiple subtasks, evaluation datasets, and standardized metrics, facilitating fair benchmarking. The task was structured to promote modeling under realistic, multilingual, and culturally nuanced healthcare contexts. We outline the dataset creation, task design and evaluation framework, participation statistics, baseline systems, and summarize the overall outcomes. We conclude with reflections on the performance trends observed and prospects for future iterations in Arabic health QA.
>
---
#### [new 028] Building Task Bots with Self-learning for Enhanced Adaptability, Extensibility, and Factuality
- **分类: cs.CL**

- **简介: 论文旨在构建具备自学习能力的任务机器人，解决传统对话研究中适应性、可扩展性和准确性不足的问题，通过创新技术实现机器人自主适应动态环境。**

- **链接: [http://arxiv.org/pdf/2508.19689v1](http://arxiv.org/pdf/2508.19689v1)**

> **作者:** Xiaoying Zhang
>
> **备注:** 179 pages
>
> **摘要:** Developing adaptable, extensible, and accurate task bots with minimal or zero human intervention is a significant challenge in dialog research. This thesis examines the obstacles and potential solutions for creating such bots, focusing on innovative techniques that enable bots to learn and adapt autonomously in constantly changing environments.
>
---
#### [new 029] Blockwise SFT for Diffusion Language Models: Reconciling Bidirectional Attention and Autoregressive Decoding
- **分类: cs.CL**

- **简介: 该论文针对扩散语言模型的监督微调（SFT）与块状解码不匹配问题，提出Blockwise SFT方法，通过分块掩码和损失计算对齐训练与推理过程，提升文本生成质量。**

- **链接: [http://arxiv.org/pdf/2508.19529v1](http://arxiv.org/pdf/2508.19529v1)**

> **作者:** Bowen Sun; Yujun Cai; Ming-Hsuan Yang; Yiwei Wang
>
> **摘要:** Discrete diffusion language models have shown strong potential for text generation, yet standard supervised fine-tuning (SFT) misaligns with their semi-autoregressive inference: training randomly masks tokens across the entire response, while inference generates fixed-size blocks sequentially. This mismatch introduces noisy prefixes and leaky suffixes, biasing gradients away from the desired blockwise likelihood. We propose Blockwise SFT, which partitions responses into fixed-size blocks, selects one active block per step for stochastic masking, freezes all preceding tokens, and fully hides future ones. Loss is computed only over the active block, directly mirroring the blockwise decoding process. Experiments on GSM8K, MATH, and MetaMathQA show consistent gains over classical SFT under equal compute or token budgets. Block size consistency studies and ablations confirm that improvements stem from faithful training-inference alignment rather than incidental masking effects. Our results highlight the importance of matching supervision granularity to the decoding procedure in diffusion-based language models.
>
---
#### [new 030] 11Plus-Bench: Demystifying Multimodal LLM Spatial Reasoning with Cognitive-Inspired Analysis
- **分类: cs.CL; cs.CV; cs.LG**

- **简介: 论文提出11Plus-Bench基准，评估多模态LLM的空间推理能力，通过对比人类表现揭示模型在空间认知上的潜力与局限，提供改进方向。**

- **链接: [http://arxiv.org/pdf/2508.20068v1](http://arxiv.org/pdf/2508.20068v1)**

> **作者:** Chengzu Li; Wenshan Wu; Huanyu Zhang; Qingtao Li; Zeyu Gao; Yan Xia; José Hernández-Orallo; Ivan Vulić; Furu Wei
>
> **备注:** 9 pages, 4 figures (22 pages, 7 figures, 7 tables including references and appendices)
>
> **摘要:** For human cognitive process, spatial reasoning and perception are closely entangled, yet the nature of this interplay remains underexplored in the evaluation of multimodal large language models (MLLMs). While recent MLLM advancements show impressive performance on reasoning, their capacity for human-like spatial cognition remains an open question. In this work, we introduce a systematic evaluation framework to assess the spatial reasoning abilities of state-of-the-art MLLMs relative to human performance. Central to our work is 11Plus-Bench, a high-quality benchmark derived from realistic standardized spatial aptitude tests. 11Plus-Bench also features fine-grained expert annotations of both perceptual complexity and reasoning process, enabling detailed instance-level analysis of model behavior. Through extensive experiments across 14 MLLMs and human evaluation, we find that current MLLMs exhibit early signs of spatial cognition. Despite a large performance gap compared to humans, MLLMs' cognitive profiles resemble those of humans in that cognitive effort correlates strongly with reasoning-related complexity. However, instance-level performance in MLLMs remains largely random, whereas human correctness is highly predictable and shaped by abstract pattern complexity. These findings highlight both emerging capabilities and limitations in current MLLMs' spatial reasoning capabilities and provide actionable insights for advancing model design.
>
---
#### [new 031] ArgCMV: An Argument Summarization Benchmark for the LLM-era
- **分类: cs.CL**

- **简介: 该论文针对论证摘要中的关键点提取任务，指出现有ArgKP21数据集的局限性，构建了包含12K真实在线辩论论点的ArgCMV新基准，验证现有方法在复杂场景下的不足，推动LLM驱动的摘要研究。**

- **链接: [http://arxiv.org/pdf/2508.19580v1](http://arxiv.org/pdf/2508.19580v1)**

> **作者:** Omkar Gurjar; Agam Goyal; Eshwar Chandrasekharan
>
> **摘要:** Key point extraction is an important task in argument summarization which involves extracting high-level short summaries from arguments. Existing approaches for KP extraction have been mostly evaluated on the popular ArgKP21 dataset. In this paper, we highlight some of the major limitations of the ArgKP21 dataset and demonstrate the need for new benchmarks that are more representative of actual human conversations. Using SoTA large language models (LLMs), we curate a new argument key point extraction dataset called ArgCMV comprising of around 12K arguments from actual online human debates spread across over 3K topics. Our dataset exhibits higher complexity such as longer, co-referencing arguments, higher presence of subjective discourse units, and a larger range of topics over ArgKP21. We show that existing methods do not adapt well to ArgCMV and provide extensive benchmark results by experimenting with existing baselines and latest open source models. This work introduces a novel KP extraction dataset for long-context online discussions, setting the stage for the next generation of LLM-driven summarization research.
>
---
#### [new 032] AgentCoMa: A Compositional Benchmark Mixing Commonsense and Mathematical Reasoning in Real-World Scenarios
- **分类: cs.CL**

- **简介: 该论文设计混合常识与数学推理的基准测试（AgentCoMa），解决单一类型基准的不足，通过测试61个模型发现组合任务准确率下降30%，人类表现相近，并分析模型脆弱性。**

- **链接: [http://arxiv.org/pdf/2508.19988v1](http://arxiv.org/pdf/2508.19988v1)**

> **作者:** Lisa Alazraki; Lihu Chen; Ana Brassard; Joe Stacey; Hossein A. Rahmani; Marek Rei
>
> **摘要:** Large Language Models (LLMs) have achieved high accuracy on complex commonsense and mathematical problems that involve the composition of multiple reasoning steps. However, current compositional benchmarks testing these skills tend to focus on either commonsense or math reasoning, whereas LLM agents solving real-world tasks would require a combination of both. In this work, we introduce an Agentic Commonsense and Math benchmark (AgentCoMa), where each compositional task requires a commonsense reasoning step and a math reasoning step. We test it on 61 LLMs of different sizes, model families, and training strategies. We find that LLMs can usually solve both steps in isolation, yet their accuracy drops by ~30% on average when the two are combined. This is a substantially greater performance gap than the one we observe in prior compositional benchmarks that combine multiple steps of the same reasoning type. In contrast, non-expert human annotators can solve the compositional questions and the individual steps in AgentCoMa with similarly high accuracy. Furthermore, we conduct a series of interpretability studies to better understand the performance gap, examining neuron patterns, attention maps and membership inference. Our work underscores a substantial degree of model brittleness in the context of mixed-type compositional reasoning and offers a test bed for future improvement.
>
---
#### [new 033] Selective Retrieval-Augmentation for Long-Tail Legal Text Classification
- **分类: cs.CL; cs.IR**

- **简介: 该论文针对法律文本分类中的长尾标签分布问题，提出Selective Retrieval-Augmentation（SRA）方法，通过检索增强低频标签样本，提升罕见类性能，无需修改模型架构，在LEDGAR和UNFAIR-ToS数据集上取得更优微宏F1分数。**

- **链接: [http://arxiv.org/pdf/2508.19997v1](http://arxiv.org/pdf/2508.19997v1)**

> **作者:** Boheng Mao
>
> **摘要:** Legal text classification is a fundamental NLP task in the legal domain. Benchmark datasets in this area often exhibit a long-tail label distribution, where many labels are underrepresented, leading to poor model performance on rare classes. This paper proposes Selective Retrieval-Augmentation (SRA) as a solution to this problem. SRA focuses on augmenting samples belonging to low-frequency labels in the training set, preventing the introduction of noise for well-represented classes, and requires no changes to the model architecture. Retrieval is performed only from the training data to ensure there is no potential information leakage, removing the need for external corpora simultaneously. The proposed SRA method is tested on two legal text classification benchmark datasets with long-tail distributions: LEDGAR (single-label) and UNFAIR-ToS (multi-label). The results indicate that SRA attains higher micro-F1 and macro-F1 scores compared to all current LexGLUE baselines across both datasets, illustrating consistent improvements in long-tail legal text classification. The code repository is available at: https://github.com/Boheng-Mao/sra-legal
>
---
#### [new 034] Understanding and Leveraging the Expert Specialization of Context Faithfulness in Mixture-of-Experts LLMs
- **分类: cs.CL**

- **简介: 该论文旨在提升大语言模型在上下文依赖场景中的忠实度，通过识别上下文忠实的专家并进行针对性微调，解决模型无法有效利用上下文的问题。**

- **链接: [http://arxiv.org/pdf/2508.19594v1](http://arxiv.org/pdf/2508.19594v1)**

> **作者:** Jun Bai; Minghao Tong; Yang Liu; Zixia Jia; Zilong Zheng
>
> **备注:** Accepted by EMNLP 2025 Main
>
> **摘要:** Context faithfulness is essential for reliable reasoning in context-dependent scenarios. However, large language models often struggle to ground their outputs in the provided context, resulting in irrelevant responses. Inspired by the emergent expert specialization observed in mixture-of-experts architectures, this work investigates whether certain experts exhibit specialization in context utilization, offering a potential pathway toward targeted optimization for improved context faithfulness. To explore this, we propose Router Lens, a method that accurately identifies context-faithful experts. Our analysis reveals that these experts progressively amplify attention to relevant contextual information, thereby enhancing context grounding. Building on this insight, we introduce Context-faithful Expert Fine-Tuning (CEFT), a lightweight optimization approach that selectively fine-tunes context-faithful experts. Experiments across a wide range of benchmarks and models demonstrate that CEFT matches or surpasses the performance of full fine-tuning while being significantly more efficient.
>
---
#### [new 035] Context-Adaptive Synthesis and Compression for Enhanced Retrieval-Augmented Generation in Complex Domains
- **分类: cs.CL**

- **简介: 论文针对复杂领域多文档问答中的RAG信息过载问题，提出CASC框架，通过上下文分析与压缩模块提升生成准确性。**

- **链接: [http://arxiv.org/pdf/2508.19357v1](http://arxiv.org/pdf/2508.19357v1)**

> **作者:** Peiran Zhou; Junnan Zhu; Yichen Shen; Ruoxi Yu
>
> **摘要:** Large Language Models (LLMs) excel in language tasks but are prone to hallucinations and outdated knowledge. Retrieval-Augmented Generation (RAG) mitigates these by grounding LLMs in external knowledge. However, in complex domains involving multiple, lengthy, or conflicting documents, traditional RAG suffers from information overload and inefficient synthesis, leading to inaccurate and untrustworthy answers. To address this, we propose CASC (Context-Adaptive Synthesis and Compression), a novel framework that intelligently processes retrieved contexts. CASC introduces a Context Analyzer & Synthesizer (CAS) module, powered by a fine-tuned smaller LLM, which performs key information extraction, cross-document consistency checking and conflict resolution, and question-oriented structured synthesis. This process transforms raw, scattered information into a highly condensed, structured, and semantically rich context, significantly reducing the token count and cognitive load for the final Reader LLM. We evaluate CASC on SciDocs-QA, a new challenging multi-document question answering dataset designed for complex scientific domains with inherent redundancies and conflicts. Our extensive experiments demonstrate that CASC consistently outperforms strong baselines.
>
---
#### [new 036] Bangla-Bayanno: A 52K-Pair Bengali Visual Question Answering Dataset with LLM-Assisted Translation Refinement
- **分类: cs.CL; cs.CV**

- **简介: 本文提出Bangla-Bayanno数据集，用于解决低资源孟加拉语VQA任务中翻译质量差、领域局限等问题，通过多语言LLM辅助翻译优化，构建含52k对问题-答案的高质量基准，推动多模态学习研究。**

- **链接: [http://arxiv.org/pdf/2508.19887v1](http://arxiv.org/pdf/2508.19887v1)**

> **作者:** Mohammed Rakibul Hasan; Rafi Majid; Ahanaf Tahmid
>
> **摘要:** In this paper, we introduce Bangla-Bayanno, an open-ended Visual Question Answering (VQA) Dataset in Bangla, a widely used, low-resource language in multimodal AI research. The majority of existing datasets are either manually annotated with an emphasis on a specific domain, query type, or answer type or are constrained by niche answer formats. In order to mitigate human-induced errors and guarantee lucidity, we implemented a multilingual LLM-assisted translation refinement pipeline. This dataset overcomes the issues of low-quality translations from multilingual sources. The dataset comprises 52,650 question-answer pairs across 4750+ images. Questions are classified into three distinct answer types: nominal (short descriptive), quantitative (numeric), and polar (yes/no). Bangla-Bayanno provides the most comprehensive open-source, high-quality VQA benchmark in Bangla, aiming to advance research in low-resource multimodal learning and facilitate the development of more inclusive AI systems.
>
---
#### [new 037] Forewarned is Forearmed: Pre-Synthesizing Jailbreak-like Instructions to Enhance LLM Safety Guardrail to Potential Attacks
- **分类: cs.CL**

- **简介: 该论文针对LLM面临的安全威胁，提出IMAGINE框架，通过预合成劫持指令填补训练数据与攻击分布的差距，提升模型防御能力，降低真实攻击成功率。**

- **链接: [http://arxiv.org/pdf/2508.20038v1](http://arxiv.org/pdf/2508.20038v1)**

> **作者:** Sheng Liu; Qiang Sheng; Danding Wang; Yang Li; Guang Yang; Juan Cao
>
> **备注:** EMNLP 2025 findings
>
> **摘要:** Despite advances in improving large language model(LLM) to refuse to answer malicious instructions, widely used LLMs remain vulnerable to jailbreak attacks where attackers generate instructions with distributions differing from safety alignment corpora. New attacks expose LLMs' inability to recognize unseen malicious instructions, highlighting a critical distributional mismatch between training data and real-world attacks that forces developers into reactive patching cycles. To tackle this challenge, we propose IMAGINE, a synthesis framework that leverages embedding space distribution analysis to generate jailbreak-like instructions. This approach effectively fills the distributional gap between authentic jailbreak patterns and safety alignment corpora. IMAGINE follows an iterative optimization process that dynamically evolves text generation distributions across iterations, thereby augmenting the coverage of safety alignment data distributions through synthesized data examples. Based on the safety-aligned corpus enhanced through IMAGINE, our framework demonstrates significant decreases in attack success rate on Qwen2.5, Llama3.1, and Llama3.2 without compromising their utility.
>
---
#### [new 038] Dhati+: Fine-tuned Large Language Models for Arabic Subjectivity Evaluation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对阿拉伯语主观性评估任务，解决资源匮乏问题。通过构建AraDhati+数据集，微调XLM-RoBERTa等模型，并采用集成决策方法，实现97.79%的高准确率。**

- **链接: [http://arxiv.org/pdf/2508.19966v1](http://arxiv.org/pdf/2508.19966v1)**

> **作者:** Slimane Bellaouar; Attia Nehar; Soumia Souffi; Mounia Bouameur
>
> **备注:** 25 pages, 7 figures
>
> **摘要:** Despite its significance, Arabic, a linguistically rich and morphologically complex language, faces the challenge of being under-resourced. The scarcity of large annotated datasets hampers the development of accurate tools for subjectivity analysis in Arabic. Recent advances in deep learning and Transformers have proven highly effective for text classification in English and French. This paper proposes a new approach for subjectivity assessment in Arabic textual data. To address the dearth of specialized annotated datasets, we developed a comprehensive dataset, AraDhati+, by leveraging existing Arabic datasets and collections (ASTD, LABR, HARD, and SANAD). Subsequently, we fine-tuned state-of-the-art Arabic language models (XLM-RoBERTa, AraBERT, and ArabianGPT) on AraDhati+ for effective subjectivity classification. Furthermore, we experimented with an ensemble decision approach to harness the strengths of individual models. Our approach achieves a remarkable accuracy of 97.79\,\% for Arabic subjectivity classification. Results demonstrate the effectiveness of the proposed approach in addressing the challenges posed by limited resources in Arabic language processing.
>
---
#### [new 039] One Joke to Rule them All? On the (Im)possibility of Generalizing Humor
- **分类: cs.CL; cs.AI**

- **简介: 论文研究LLM在幽默任务中的泛化能力，探讨是否可通过特定幽默任务（如Dad Jokes）迁移到新类型幽默。通过迁移学习实验，验证模型跨类型泛化效果，发现多样训练提升迁移性能，但部分类型间存在迁移障碍。**

- **链接: [http://arxiv.org/pdf/2508.19402v1](http://arxiv.org/pdf/2508.19402v1)**

> **作者:** Mor Turgeman; Chen Shani; Dafna Shahaf
>
> **摘要:** Humor is a broad and complex form of communication that remains challenging for machines. Despite its broadness, most existing research on computational humor traditionally focused on modeling a specific type of humor. In this work, we wish to understand whether competence on one or more specific humor tasks confers any ability to transfer to novel, unseen types; in other words, is this fragmentation inevitable? This question is especially timely as new humor types continuously emerge in online and social media contexts (e.g., memes, anti-humor, AI fails). If Large Language Models (LLMs) are to keep up with this evolving landscape, they must be able to generalize across humor types by capturing deeper, transferable mechanisms. To investigate this, we conduct a series of transfer learning experiments across four datasets, representing different humor tasks. We train LLMs under varied diversity settings (1-3 datasets in training, testing on a novel task). Experiments reveal that models are capable of some transfer, and can reach up to 75% accuracy on unseen datasets; training on diverse sources improves transferability (1.88-4.05%) with minimal-to-no drop in in-domain performance. Further analysis suggests relations between humor types, with Dad Jokes surprisingly emerging as the best enabler of transfer (but is difficult to transfer to). We release data and code.
>
---
#### [new 040] Benchmarking Hindi LLMs: A New Suite of Datasets and a Comparative Analysis
- **分类: cs.CL; cs.LG**

- **简介: 该论文旨在评估印度语LLM的性能，解决缺乏高质量基准的问题，通过创建五套印度语数据集（结合人工标注与翻译验证）进行跨模型比较分析。**

- **链接: [http://arxiv.org/pdf/2508.19831v1](http://arxiv.org/pdf/2508.19831v1)**

> **作者:** Anusha Kamath; Kanishk Singla; Rakesh Paul; Raviraj Joshi; Utkarsh Vaidya; Sanjay Singh Chauhan; Niranjan Wartikar
>
> **摘要:** Evaluating instruction-tuned Large Language Models (LLMs) in Hindi is challenging due to a lack of high-quality benchmarks, as direct translation of English datasets fails to capture crucial linguistic and cultural nuances. To address this, we introduce a suite of five Hindi LLM evaluation datasets: IFEval-Hi, MT-Bench-Hi, GSM8K-Hi, ChatRAG-Hi, and BFCL-Hi. These were created using a methodology that combines from-scratch human annotation with a translate-and-verify process. We leverage this suite to conduct an extensive benchmarking of open-source LLMs supporting Hindi, providing a detailed comparative analysis of their current capabilities. Our curation process also serves as a replicable methodology for developing benchmarks in other low-resource languages.
>
---
#### [new 041] A perishable ability? The future of writing in the face of generative artificial intelligence
- **分类: cs.CL; cs.AI; cs.CY; cs.HC**

- **简介: 论文探讨生成式AI对人类写作能力的影响，分析其可能导致写作能力下降，并类比历史案例预测未来趋势。**

- **链接: [http://arxiv.org/pdf/2508.19427v1](http://arxiv.org/pdf/2508.19427v1)**

> **作者:** Evandro L. T. P. Cunha
>
> **备注:** 10 pages
>
> **摘要:** The 2020s have been witnessing a very significant advance in the development of generative artificial intelligence tools, including text generation systems based on large language models. These tools have been increasingly used to generate texts in the most diverse domains -- from technical texts to literary texts --, which might eventually lead to a lower volume of written text production by humans. This article discusses the possibility of a future in which human beings will have lost or significantly decreased their ability to write due to the outsourcing of this activity to machines. This possibility parallels the loss of the ability to write in other moments of human history, such as during the so-called Greek Dark Ages (approx. 1200 BCE - 800 BCE).
>
---
#### [new 042] Automatic integration of SystemC in the FMI standard for Software-defined Vehicle design
- **分类: cs.CL**

- **简介: 该论文旨在解决汽车共模拟中标准化接口缺失的问题，通过自动将SystemC模型封装为FMI标准，实现高精度与互操作性的结合，提升软件定义车辆的设计效率与集成能力。**

- **链接: [http://arxiv.org/pdf/2508.19665v1](http://arxiv.org/pdf/2508.19665v1)**

> **作者:** Giovanni Pollo; Andrei Mihai Albu; Alessio Burrello; Daniele Jahier Pagliari; Cristian Tesconi; Loris Panaro; Dario Soldi; Fabio Autieri; Sara Vinco
>
> **摘要:** The recent advancements of the automotive sector demand robust co-simulation methodologies that enable early validation and seamless integration across hardware and software domains. However, the lack of standardized interfaces and the dominance of proprietary simulation platforms pose significant challenges to collaboration, scalability, and IP protection. To address these limitations, this paper presents an approach for automatically wrapping SystemC models by using the Functional Mock-up Interface (FMI) standard. This method combines the modeling accuracy and fast time-to-market of SystemC with the interoperability and encapsulation benefits of FMI, enabling secure and portable integration of embedded components into co-simulation workflows. We validate the proposed methodology on real-world case studies, demonstrating its effectiveness with complex designs.
>
---
#### [new 043] LongReasonArena: A Long Reasoning Benchmark for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 本文提出LongReasonArena基准，针对大语言模型长推理能力不足的问题，设计多步算法任务评估其检索、回溯等推理能力，支持百万级token推理，实验证明其对主流模型构成显著挑战。**

- **链接: [http://arxiv.org/pdf/2508.19363v1](http://arxiv.org/pdf/2508.19363v1)**

> **作者:** Jiayu Ding; Shuming Ma; Lei Cui; Nanning Zheng; Furu Wei
>
> **摘要:** Existing long-context benchmarks for Large Language Models (LLMs) focus on evaluating comprehension of long inputs, while overlooking the evaluation of long reasoning abilities. To address this gap, we introduce LongReasonArena, a benchmark specifically designed to assess the long reasoning capabilities of LLMs. Our tasks require models to solve problems by executing multi-step algorithms that reflect key aspects of long reasoning, such as retrieval and backtracking. By controlling the inputs, the required reasoning length can be arbitrarily scaled, reaching up to 1 million tokens of reasoning for the most challenging tasks. Extensive evaluation results demonstrate that LongReasonArena presents a significant challenge for both open-source and proprietary LLMs. For instance, Deepseek-R1 achieves only 7.5% accuracy on our task. Further analysis also reveals that the accuracy exhibits a linear decline with respect to the logarithm of the expected number of reasoning steps. Our code and data is available at https://github.com/LongReasonArena/LongReasonArena.
>
---
#### [new 044] HEAL: A Hypothesis-Based Preference-Aware Analysis Framework
- **分类: cs.CL**

- **简介: 论文提出HEAL框架，用于偏好对齐评估，解决单一响应评估问题，通过假设空间重排序和两个新指标，结合统一基准验证，提升偏好捕捉能力。**

- **链接: [http://arxiv.org/pdf/2508.19922v1](http://arxiv.org/pdf/2508.19922v1)**

> **作者:** Yifu Huo; Chenglong Wang; Qiren Zhu; Shunjie Xing; Tong Xiao; Chunliang Zhang; Tongran Liu; Jinbo Zhu
>
> **备注:** Accepted by EMNLP 2025 Findings
>
> **摘要:** Preference optimization methods like DPO have achieved remarkable performance in LLM alignment. However, the evaluation for these methods relies on a single response and overlooks other potential outputs, which could also be generated in real-world applications within this hypothetical space. To address this issue, this paper presents a \textbf{H}ypothesis-based Pr\textbf{E}ference-aware \textbf{A}na\textbf{L}ysis Framework (HEAL), a novel evaluation paradigm that formulates preference alignment as a re-ranking process within hypothesis spaces. The framework incorporates two complementary metrics: ranking accuracy for evaluating ordinal consistency and preference strength correlation for assessing continuous alignment. To facilitate this framework, we develop UniHypoBench, a unified hypothesis benchmark constructed from diverse instruction-response pairs. Through extensive experiments based on HEAL, with a particular focus on the intrinsic mechanisms of preference learning, we demonstrate that current preference learning methods can effectively capture preferences provided by proxy models while simultaneously suppressing negative samples. These findings contribute to preference learning research through two significant avenues. Theoretically, we introduce hypothesis space analysis as an innovative paradigm for understanding preference alignment. Practically, HEAL offers researchers robust diagnostic tools for refining preference optimization methods, while our empirical results identify promising directions for developing more advanced alignment algorithms capable of comprehensive preference capture.
>
---
#### [new 045] TokenVerse++: Towards Flexible Multitask Learning with Dynamic Task Activation
- **分类: cs.CL; eess.AS**

- **简介: 该论文针对多任务学习中需全标注数据的限制，提出TokenVerse++通过在ASR模型中引入动态任务激活机制，利用部分标注数据提升性能，实现更灵活的多任务学习。**

- **链接: [http://arxiv.org/pdf/2508.19856v1](http://arxiv.org/pdf/2508.19856v1)**

> **作者:** Shashi Kumar; Srikanth Madikeri; Esaú Villatoro-Tello; Sergio Burdisso; Pradeep Rangappa; Andrés Carofilis; Petr Motlicek; Karthik Pandia; Shankar Venkatesan; Kadri Hacioğlu; Andreas Stolcke
>
> **备注:** Accepted to IEEE ASRU 2025. Copyright\copyright 2025 IEEE
>
> **摘要:** Token-based multitasking frameworks like TokenVerse require all training utterances to have labels for all tasks, hindering their ability to leverage partially annotated datasets and scale effectively. We propose TokenVerse++, which introduces learnable vectors in the acoustic embedding space of the XLSR-Transducer ASR model for dynamic task activation. This core mechanism enables training with utterances labeled for only a subset of tasks, a key advantage over TokenVerse. We demonstrate this by successfully integrating a dataset with partial labels, specifically for ASR and an additional task, language identification, improving overall performance. TokenVerse++ achieves results on par with or exceeding TokenVerse across multiple tasks, establishing it as a more practical multitask alternative without sacrificing ASR performance.
>
---
#### [new 046] MultiPL-MoE: Multi-Programming-Lingual Extension of Large Language Models through Hybrid Mixture-of-Experts
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对多语言代码生成任务，解决多编程语言支持难题，提出MultiPL-MoE模型，结合混合MoE结构在token和segment层面优化专家选择，提升多语言代码生成效果。**

- **链接: [http://arxiv.org/pdf/2508.19268v1](http://arxiv.org/pdf/2508.19268v1)**

> **作者:** Qing Wang; Xue Han; Jiahui Wang; Lehao Xing; Qian Hu; Lianlian Zhang; Chao Deng; Junlan Feng
>
> **摘要:** Despite LLMs' excellent code creation capabilities, multilingual code generation remains extremely challenging. To address this, we intent to improve the multi-programming-lingual (MultiPL) performance of the base LLMs while retaining the most popular ones using restricted computational resources. We consider MultiPL to be a special case of multiple natural languages and propose a MultiPL extension of LLMs utilizing a hybrid mixture of experts (MoE), called MultiPL-MoE. Specifically, MultiPL-MoE combines two paired MoEs to optimize expert selection at both the token and segment levels. The token-level MoE is a standard upcycling MoE structure with a shared expert and a novel gate weight normalization approach that aids in the final fusion with the segment-level MoE. The segment-level MoE incorporates two innovative designs to better capture the syntactic structure and contextual patterns of programming languages: First, using a sliding window to partition the input token sequence into multiple segments; Then, adopting an expert-choice routing strategy that allows experts to select the top-k segments. The results of the experiment proved the effectiveness of MultiPL-MoE.
>
---
#### [new 047] Reflective Agreement: Combining Self-Mixture of Agents with a Sequence Tagger for Robust Event Extraction
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对事件抽取任务中传统模型召回率低和生成模型幻觉问题，提出ARIS混合系统，结合自我混合代理与序列标注器，通过结构共识、置信过滤和LLM反思推理提升预测质量，在三个数据集上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2508.19359v1](http://arxiv.org/pdf/2508.19359v1)**

> **作者:** Fatemeh Haji; Mazal Bethany; Cho-Yu Jason Chiang; Anthony Rios; Peyman Najafirad
>
> **摘要:** Event Extraction (EE) involves automatically identifying and extracting structured information about events from unstructured text, including triggers, event types, and arguments. Traditional discriminative models demonstrate high precision but often exhibit limited recall, particularly for nuanced or infrequent events. Conversely, generative approaches leveraging Large Language Models (LLMs) provide higher semantic flexibility and recall but suffer from hallucinations and inconsistent predictions. To address these challenges, we propose Agreement-based Reflective Inference System (ARIS), a hybrid approach combining a Self Mixture of Agents with a discriminative sequence tagger. ARIS explicitly leverages structured model consensus, confidence-based filtering, and an LLM reflective inference module to reliably resolve ambiguities and enhance overall event prediction quality. We further investigate decomposed instruction fine-tuning for enhanced LLM event extraction understanding. Experiments demonstrate our approach outperforms existing state-of-the-art event extraction methods across three benchmark datasets.
>
---
#### [new 048] MathBuddy: A Multimodal System for Affective Math Tutoring
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文提出MathBuddy，一种多模态情感计算数学辅导系统，通过整合对话文本与面部表情识别学生情绪，动态调整教学策略，提升LLM tutor的教学生态效果。**

- **链接: [http://arxiv.org/pdf/2508.19993v1](http://arxiv.org/pdf/2508.19993v1)**

> **作者:** Debanjana Kar; Leopold Böss; Dacia Braca; Sebastian Maximilian Dennerlein; Nina Christine Hubig; Philipp Wintersberger; Yufang Hou
>
> **摘要:** The rapid adoption of LLM-based conversational systems is already transforming the landscape of educational technology. However, the current state-of-the-art learning models do not take into account the student's affective states. Multiple studies in educational psychology support the claim that positive or negative emotional states can impact a student's learning capabilities. To bridge this gap, we present MathBuddy, an emotionally aware LLM-powered Math Tutor, which dynamically models the student's emotions and maps them to relevant pedagogical strategies, making the tutor-student conversation a more empathetic one. The student's emotions are captured from the conversational text as well as from their facial expressions. The student's emotions are aggregated from both modalities to confidently prompt our LLM Tutor for an emotionally-aware response. We have effectively evaluated our model using automatic evaluation metrics across eight pedagogical dimensions and user studies. We report a massive 23 point performance gain using the win rate and a 3 point gain at an overall level using DAMR scores which strongly supports our hypothesis of improving LLM-based tutor's pedagogical abilities by modeling students' emotions.
>
---
#### [new 049] Survey of Specialized Large Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文是综述类研究，系统评估专业大语言模型在医疗、金融等领域的技术进展，探讨领域原生设计、参数效率等突破，分析其在专业应用中的优势及对电商的影响。**

- **链接: [http://arxiv.org/pdf/2508.19667v1](http://arxiv.org/pdf/2508.19667v1)**

> **作者:** Chenghan Yang; Ruiyu Zhao; Yang Liu; Ling Jiang
>
> **备注:** 9 pages, 1 figures
>
> **摘要:** The rapid evolution of specialized large language models (LLMs) has transitioned from simple domain adaptation to sophisticated native architectures, marking a paradigm shift in AI development. This survey systematically examines this progression across healthcare, finance, legal, and technical domains. Besides the wide use of specialized LLMs, technical breakthrough such as the emergence of domain-native designs beyond fine-tuning, growing emphasis on parameter efficiency through sparse computation and quantization, increasing integration of multimodal capabilities and so on are applied to recent LLM agent. Our analysis reveals how these innovations address fundamental limitations of general-purpose LLMs in professional applications, with specialized models consistently performance gains on domain-specific benchmarks. The survey further highlights the implications for E-Commerce field to fill gaps in the field.
>
---
#### [new 050] Heterogeneous LLM Methods for Ontology Learning (Few-Shot Prompting, Ensemble Typing, and Attention-Based Taxonomies)
- **分类: cs.CL; cs.LO; cs.SC; 68T30, 68T50, 68T07, 68U15; I.2.4; I.2.7; H.3.1; H.3.3; I.2.6**

- **简介: 该论文针对异构领域本体学习的三项任务（术语提取、类型化、分类发现），提出结合检索增强提示、零样本分类和注意力图模型的模块化方法，实现跨领域高效本体构建。**

- **链接: [http://arxiv.org/pdf/2508.19428v1](http://arxiv.org/pdf/2508.19428v1)**

> **作者:** Aleksandra Beliaeva; Temurbek Rahmatullaev
>
> **摘要:** We present a comprehensive system for addressing Tasks A, B, and C of the LLMs4OL 2025 challenge, which together span the full ontology construction pipeline: term extraction, typing, and taxonomy discovery. Our approach combines retrieval-augmented prompting, zero-shot classification, and attention-based graph modeling -- each tailored to the demands of the respective task. For Task A, we jointly extract domain-specific terms and their ontological types using a retrieval-augmented generation (RAG) pipeline. Training data was reformulated into a document to terms and types correspondence, while test-time inference leverages semantically similar training examples. This single-pass method requires no model finetuning and improves overall performance through lexical augmentation Task B, which involves assigning types to given terms, is handled via a dual strategy. In the few-shot setting (for domains with labeled training data), we reuse the RAG scheme with few-shot prompting. In the zero-shot setting (for previously unseen domains), we use a zero-shot classifier that combines cosine similarity scores from multiple embedding models using confidence-based weighting. In Task C, we model taxonomy discovery as graph inference. Using embeddings of type labels, we train a lightweight cross-attention layer to predict is-a relations by approximating a soft adjacency matrix. These modular, task-specific solutions enabled us to achieve top-ranking results in the official leaderboard across all three tasks. Taken together these strategies showcase the scalability, adaptability, and robustness of LLM-based architectures for ontology learning across heterogeneous domains. Code is available at: https://github.com/BelyaevaAlex/LLMs4OL-Challenge-Alexbek
>
---
#### [new 051] Bridging Language Gaps: Enhancing Few-Shot Language Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对多语言NLP中资源不均问题，提出CoLAP方法，结合对比学习与跨语言表示，实现少样本下的高效语言适应，提升低资源语言性能。**

- **链接: [http://arxiv.org/pdf/2508.19464v1](http://arxiv.org/pdf/2508.19464v1)**

> **作者:** Philipp Borchert; Jochen De Weerdt; Marie-Francine Moens
>
> **备注:** 17 pages
>
> **摘要:** The disparity in language resources poses a challenge in multilingual NLP, with high-resource languages benefiting from extensive data, while low-resource languages lack sufficient data for effective training. Our Contrastive Language Alignment with Prompting (CoLAP) method addresses this gap by integrating contrastive learning with cross-lingual representations, facilitating task-specific knowledge transfer from high-resource to lower-resource languages. The primary advantage of our approach is its data efficiency, enabling rapid adaptation to new languages and reducing the need for large labeled datasets. We conduct experiments with multilingual encoder-only and decoder-only language models on natural language understanding tasks, including natural language inference and relation extraction, evaluating performance across both high- and low-resource languages. Our results demonstrate that CoLAP outperforms few-shot cross-lingual transfer baselines and in-context learning, even with limited available data. This effectively narrows the cross-lingual performance gap, contributing to the development of more efficient multilingual NLP techniques.
>
---
#### [new 052] MovieCORE: COgnitive REasoning in Movies
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 论文提出MovieCORE数据集，用于评估视频问答模型的深层认知能力，通过多模型协作生成高质问题，设计认知测试并引入ACE模块提升推理性能，解决传统VQA在复杂电影内容理解上的不足。**

- **链接: [http://arxiv.org/pdf/2508.19026v1](http://arxiv.org/pdf/2508.19026v1)**

> **作者:** Gueter Josmy Faure; Min-Hung Chen; Jia-Fong Yeh; Ying Cheng; Hung-Ting Su; Yung-Hao Tang; Shang-Hong Lai; Winston H. Hsu
>
> **备注:** Accepted for EMNLP'2025 Main Conference. Project Page: https://joslefaure.github.io/assets/html/moviecore.html
>
> **摘要:** This paper introduces MovieCORE, a novel video question answering (VQA) dataset designed to probe deeper cognitive understanding of movie content. Unlike existing datasets that focus on surface-level comprehension, MovieCORE emphasizes questions that engage System-2 thinking while remaining specific to the video material. We present an innovative agentic brainstorming approach, utilizing multiple large language models (LLMs) as thought agents to generate and refine high-quality question-answer pairs. To evaluate dataset quality, we develop a set of cognitive tests assessing depth, thought-provocation potential, and syntactic complexity. We also propose a comprehensive evaluation scheme for assessing VQA model performance on deeper cognitive tasks. To address the limitations of existing video-language models (VLMs), we introduce an agentic enhancement module, Agentic Choice Enhancement (ACE), which improves model reasoning capabilities post-training by up to 25%. Our work contributes to advancing movie understanding in AI systems and provides valuable insights into the capabilities and limitations of current VQA models when faced with more challenging, nuanced questions about cinematic content. Our project page, dataset and code can be found at https://joslefaure.github.io/assets/html/moviecore.html.
>
---
#### [new 053] CAMÕES: A Comprehensive Automatic Speech Recognition Benchmark for European Portuguese
- **分类: cs.CL; eess.AS**

- **简介: 该论文针对欧洲葡萄牙语（EP）语音识别资源不足的问题，提出CAMÕES基准框架，包含46小时EP测试数据和425小时训练集，评估多模型（含从头训练的E-Branchformer）性能，实现EPASR新SOTA，提升超35%WER。**

- **链接: [http://arxiv.org/pdf/2508.19721v1](http://arxiv.org/pdf/2508.19721v1)**

> **作者:** Carlos Carvalho; Francisco Teixeira; Catarina Botelho; Anna Pompili; Rubén Solera-Ureña; Sérgio Paulo; Mariana Julião; Thomas Rolland; John Mendonça; Diogo Pereira; Isabel Trancoso; Alberto Abad
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** Existing resources for Automatic Speech Recognition in Portuguese are mostly focused on Brazilian Portuguese, leaving European Portuguese (EP) and other varieties under-explored. To bridge this gap, we introduce CAM\~OES, the first open framework for EP and other Portuguese varieties. It consists of (1) a comprehensive evaluation benchmark, including 46h of EP test data spanning multiple domains; and (2) a collection of state-of-the-art models. For the latter, we consider multiple foundation models, evaluating their zero-shot and fine-tuned performances, as well as E-Branchformer models trained from scratch. A curated set of 425h of EP was used for both fine-tuning and training. Our results show comparable performance for EP between fine-tuned foundation models and the E-Branchformer. Furthermore, the best-performing models achieve relative improvements above 35% WER, compared to the strongest zero-shot foundation model, establishing a new state-of-the-art for EP and other varieties.
>
---
#### [new 054] Automatic Question & Answer Generation Using Generative Large Language Model (LLM)
- **分类: cs.CL; cs.AI**

- **简介: 该论文旨在通过微调生成式大语言模型（LLM）实现自动问答生成（AQAG），解决教师手动制作多样化试题效率低的问题，采用提示工程与无监督学习方法，利用RACE数据集训练定制模型，提升教育评估效率。**

- **链接: [http://arxiv.org/pdf/2508.19475v1](http://arxiv.org/pdf/2508.19475v1)**

> **作者:** Md. Alvee Ehsan; A. S. M Mehedi Hasan; Kefaya Benta Shahnoor; Syeda Sumaiya Tasneem
>
> **摘要:** \Abstract{In the realm of education, student evaluation holds equal significance as imparting knowledge. To be evaluated, students usually need to go through text-based academic assessment methods. Instructors need to make diverse sets of questions that need to be fair for all students to prove their adequacy over a particular topic. This can prove to be quite challenging as they may need to manually go through several different lecture materials. Our objective is to make this whole process much easier by implementing Automatic Question Answer Generation /(AQAG), using fine-tuned generative LLM. For tailoring the instructor's preferred question style (MCQ, conceptual, or factual questions), prompt Engineering (PE) is being utilized. In this research, we propose to leverage unsupervised learning methods in NLP, primarily focusing on the English language. This approach empowers the base Meta-Llama 2-7B model to integrate RACE dataset as training data for the fine-tuning process. Creating a customized model that will offer efficient solutions for educators, instructors, and individuals engaged in text-based evaluations. A reliable and efficient tool for generating questions and answers can free up valuable time and resources, thus streamlining their evaluation processes.}
>
---
#### [new 055] Language Models Identify Ambiguities and Exploit Loopholes
- **分类: cs.CL; cs.AI**

- **简介: 该论文任务为评估大语言模型处理歧义与漏洞的能力，解决冲突目标下的语用对齐问题，通过设计场景测试模型识别歧义并利用漏洞的倾向，揭示潜在AI安全风险。**

- **链接: [http://arxiv.org/pdf/2508.19546v1](http://arxiv.org/pdf/2508.19546v1)**

> **作者:** Jio Choi; Mohit Bansal; Elias Stengel-Eskin
>
> **备注:** EMNLP 2025 camera-ready; Code: https://github.com/esteng/ambiguous-loophole-exploitation
>
> **摘要:** Studying the responses of large language models (LLMs) to loopholes presents a two-fold opportunity. First, it affords us a lens through which to examine ambiguity and pragmatics in LLMs, since exploiting a loophole requires identifying ambiguity and performing sophisticated pragmatic reasoning. Second, loopholes pose an interesting and novel alignment problem where the model is presented with conflicting goals and can exploit ambiguities to its own advantage. To address these questions, we design scenarios where LLMs are given a goal and an ambiguous user instruction in conflict with the goal, with scenarios covering scalar implicature, structural ambiguities, and power dynamics. We then measure different models' abilities to exploit loopholes to satisfy their given goals as opposed to the goals of the user. We find that both closed-source and stronger open-source models can identify ambiguities and exploit their resulting loopholes, presenting a potential AI safety risk. Our analysis indicates that models which exploit loopholes explicitly identify and reason about both ambiguity and conflicting goals.
>
---
#### [new 056] Rule Synergy Analysis using LLMs: State of the Art and Implications
- **分类: cs.CL**

- **简介: 论文评估LLMs在动态环境中理解规则协同效应的能力，针对卡牌游戏中的正负协同检测不足，构建数据集并分析错误类型，提出改进方向。**

- **链接: [http://arxiv.org/pdf/2508.19484v1](http://arxiv.org/pdf/2508.19484v1)**

> **作者:** Bahar Bateni; Benjamin Pratt; Jim Whitehead
>
> **备注:** Submitted for publication at the IEEE Transactions on Games 2024, Special Issue on Large Language Models and Games (10 pages excluding appendix, 3 figures)
>
> **摘要:** Large language models (LLMs) have demonstrated strong performance across a variety of domains, including logical reasoning, mathematics, and more. In this paper, we investigate how well LLMs understand and reason about complex rule interactions in dynamic environments, such as card games. We introduce a dataset of card synergies from the game Slay the Spire, where pairs of cards are classified based on their positive, negative, or neutral interactions. Our evaluation shows that while LLMs excel at identifying non-synergistic pairs, they struggle with detecting positive and, particularly, negative synergies. We categorize common error types, including issues with timing, defining game states, and following game rules. Our findings suggest directions for future research to improve model performance in predicting the effect of rules and their interactions.
>
---
#### [new 057] T2R-bench: A Benchmark for Generating Article-Level Reports from Real World Industrial Tables
- **分类: cs.CL**

- **简介: 论文提出表格到报告（T2R）任务，构建包含457个工业表格的T2R-bench基准，评估报告生成质量，实验显示现有模型表现有限。**

- **链接: [http://arxiv.org/pdf/2508.19813v1](http://arxiv.org/pdf/2508.19813v1)**

> **作者:** Jie Zhang; Changzai Pan; Kaiwen Wei; Sishi Xiong; Yu Zhao; Xiangyu Li; Jiaxin Peng; Xiaoyan Gu; Jian Yang; Wenhan Chang; Zhenhe Wu; Jiang Zhong; Shuangyong Song; Yongxiang Li; Xuelong Li
>
> **摘要:** Extensive research has been conducted to explore the capabilities of large language models (LLMs) in table reasoning. However, the essential task of transforming tables information into reports remains a significant challenge for industrial applications. This task is plagued by two critical issues: 1) the complexity and diversity of tables lead to suboptimal reasoning outcomes; and 2) existing table benchmarks lack the capacity to adequately assess the practical application of this task. To fill this gap, we propose the table-to-report task and construct a bilingual benchmark named T2R-bench, where the key information flow from the tables to the reports for this task. The benchmark comprises 457 industrial tables, all derived from real-world scenarios and encompassing 19 industry domains as well as 4 types of industrial tables. Furthermore, we propose an evaluation criteria to fairly measure the quality of report generation. The experiments on 25 widely-used LLMs reveal that even state-of-the-art models like Deepseek-R1 only achieves performance with 62.71 overall score, indicating that LLMs still have room for improvement on T2R-bench. Source code and data will be available after acceptance.
>
---
#### [new 058] Alignment with Fill-In-the-Middle for Enhancing Code Generation
- **分类: cs.CL**

- **简介: 论文针对代码生成中的数据限制问题，提出分块生成DPO对及AST分割与课程训练，提升模型性能，在多个基准测试中有效。**

- **链接: [http://arxiv.org/pdf/2508.19532v1](http://arxiv.org/pdf/2508.19532v1)**

> **作者:** Houxing Ren; Zimu Lu; Weikang Shi; Haotian Hou; Yunqiao Yang; Ke Wang; Aojun Zhou; Junting Pan; Mingjie Zhan; Hongsheng Li
>
> **备注:** Accepted to EMNLP 2025 (main conference)
>
> **摘要:** The code generation capabilities of Large Language Models (LLMs) have advanced applications like tool invocation and problem-solving. However, improving performance in code-related tasks remains challenging due to limited training data that is verifiable with accurate test cases. While Direct Preference Optimization (DPO) has shown promise, existing methods for generating test cases still face limitations. In this paper, we propose a novel approach that splits code snippets into smaller, granular blocks, creating more diverse DPO pairs from the same test cases. Additionally, we introduce the Abstract Syntax Tree (AST) splitting and curriculum training method to enhance the DPO training. Our approach demonstrates significant improvements in code generation tasks, as validated by experiments on benchmark datasets such as HumanEval (+), MBPP (+), APPS, LiveCodeBench, and BigCodeBench. Code and data are available at https://github.com/SenseLLM/StructureCoder.
>
---
#### [new 059] Scalable and consistent few-shot classification of survey responses using text embeddings
- **分类: cs.CL; physics.ed-ph**

- **简介: 该论文解决开放式调查回复分类中传统方法耗时且不一致的问题，提出基于文本嵌入的少样本分类框架，仅需少量示例即可实现与专家编码者高达0.74-0.83的Kappa一致性，并支持大规模可解释的定性分析。**

- **链接: [http://arxiv.org/pdf/2508.19836v1](http://arxiv.org/pdf/2508.19836v1)**

> **作者:** Jonas Timmann Mjaaland; Markus Fleten Kreutzer; Halvor Tyseng; Rebeckah K. Fussell; Gina Passante; N. G. Holmes; Anders Malthe-Sørenssen; Tor Ole B. Odden
>
> **摘要:** Qualitative analysis of open-ended survey responses is a commonly-used research method in the social sciences, but traditional coding approaches are often time-consuming and prone to inconsistency. Existing solutions from Natural Language Processing such as supervised classifiers, topic modeling techniques, and generative large language models have limited applicability in qualitative analysis, since they demand extensive labeled data, disrupt established qualitative workflows, and/or yield variable results. In this paper, we introduce a text embedding-based classification framework that requires only a handful of examples per category and fits well with standard qualitative workflows. When benchmarked against human analysis of a conceptual physics survey consisting of 2899 open-ended responses, our framework achieves a Cohen's Kappa ranging from 0.74 to 0.83 as compared to expert human coders in an exhaustive coding scheme. We further show how performance of this framework improves with fine-tuning of the text embedding model, and how the method can be used to audit previously-analyzed datasets. These findings demonstrate that text embedding-assisted coding can flexibly scale to thousands of responses without sacrificing interpretability, opening avenues for deductive qualitative analysis at scale.
>
---
#### [new 060] Your AI Bosses Are Still Prejudiced: The Emergence of Stereotypes in LLM-Based Multi-Agent Systems
- **分类: cs.CL**

- **简介: 该论文研究LLM多智能体系统中刻板印象的自发形成，通过模拟职场互动实验，发现AI代理在无预设偏见情况下仍发展出刻板印象，且随交互次数和层级结构增强而加剧，揭示其为多智能体交互的涌现现象。**

- **链接: [http://arxiv.org/pdf/2508.19919v1](http://arxiv.org/pdf/2508.19919v1)**

> **作者:** Jingyu Guo; Yingying Xu
>
> **摘要:** While stereotypes are well-documented in human social interactions, AI systems are often presumed to be less susceptible to such biases. Previous studies have focused on biases inherited from training data, but whether stereotypes can emerge spontaneously in AI agent interactions merits further exploration. Through a novel experimental framework simulating workplace interactions with neutral initial conditions, we investigate the emergence and evolution of stereotypes in LLM-based multi-agent systems. Our findings reveal that (1) LLM-Based AI agents develop stereotype-driven biases in their interactions despite beginning without predefined biases; (2) stereotype effects intensify with increased interaction rounds and decision-making power, particularly after introducing hierarchical structures; (3) these systems exhibit group effects analogous to human social behavior, including halo effects, confirmation bias, and role congruity; and (4) these stereotype patterns manifest consistently across different LLM architectures. Through comprehensive quantitative analysis, these findings suggest that stereotype formation in AI systems may arise as an emergent property of multi-agent interactions, rather than merely from training data biases. Our work underscores the need for future research to explore the underlying mechanisms of this phenomenon and develop strategies to mitigate its ethical impacts.
>
---
#### [new 061] SWIRL: A Staged Workflow for Interleaved Reinforcement Learning in Mobile GUI Control
- **分类: cs.AI; cs.CL; cs.CV; cs.MA**

- **简介: 论文提出SWIRL，分阶段处理多智能体强化学习，解决移动GUI控制中单智能体结构限制和MARL效率低问题，通过分步训练提升协调性，并在GUI和数学任务中验证有效性。**

- **链接: [http://arxiv.org/pdf/2508.20018v1](http://arxiv.org/pdf/2508.20018v1)**

> **作者:** Quanfeng Lu; Zhantao Ma; Shuai Zhong; Jin Wang; Dahai Yu; Michael K. Ng; Ping Luo
>
> **备注:** 28 pages, 12 figures
>
> **摘要:** The rapid advancement of large vision language models (LVLMs) and agent systems has heightened interest in mobile GUI agents that can reliably translate natural language into interface operations. Existing single-agent approaches, however, remain limited by structural constraints. Although multi-agent systems naturally decouple different competencies, recent progress in multi-agent reinforcement learning (MARL) has often been hindered by inefficiency and remains incompatible with current LVLM architectures. To address these challenges, we introduce SWIRL, a staged workflow for interleaved reinforcement learning designed for multi-agent systems. SWIRL reformulates MARL into a sequence of single-agent reinforcement learning tasks, updating one agent at a time while keeping the others fixed. This formulation enables stable training and promotes efficient coordination across agents. Theoretically, we provide a stepwise safety bound, a cross-round monotonic improvement theorem, and convergence guarantees on return, ensuring robust and principled optimization. In application to mobile GUI control, SWIRL instantiates a Navigator that converts language and screen context into structured plans, and an Interactor that grounds these plans into executable atomic actions. Extensive experiments demonstrate superior performance on both high-level and low-level GUI benchmarks. Beyond GUI tasks, SWIRL also demonstrates strong capability in multi-agent mathematical reasoning, underscoring its potential as a general framework for developing efficient and robust multi-agent systems.
>
---
#### [new 062] Safety Alignment Should Be Made More Than Just A Few Attention Heads
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文针对大语言模型安全对齐不足的问题，提出AHD训练策略，通过分散安全相关行为至多注意力头，提升模型对对抗攻击的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.19697v1](http://arxiv.org/pdf/2508.19697v1)**

> **作者:** Chao Huang; Zefeng Zhang; Juewei Yue; Quangang Li; Chuang Zhang; Tingwen Liu
>
> **摘要:** Current safety alignment for large language models(LLMs) continues to present vulnerabilities, given that adversarial prompting can effectively bypass their safety measures.Our investigation shows that these safety mechanisms predominantly depend on a limited subset of attention heads: removing or ablating these heads can severely compromise model safety. To identify and evaluate these safety-critical components, we introduce RDSHA, a targeted ablation method that leverages the model's refusal direction to pinpoint attention heads mostly responsible for safety behaviors. Further analysis shows that existing jailbreak attacks exploit this concentration by selectively bypassing or manipulating these critical attention heads. To address this issue, we propose AHD, a novel training strategy designed to promote the distributed encoding of safety-related behaviors across numerous attention heads. Experimental results demonstrate that AHD successfully distributes safety-related capabilities across more attention heads. Moreover, evaluations under several mainstream jailbreak attacks show that models trained with AHD exhibit considerably stronger safety robustness, while maintaining overall functional utility.
>
---
#### [new 063] KRETA: A Benchmark for Korean Reading and Reasoning in Text-Rich VQA Attuned to Diverse Visual Contexts
- **分类: cs.CV; cs.CL**

- **简介: 论文提出KRETA基准，针对韩语文本丰富VQA任务，解决低资源语言评估不足问题，通过多领域图像数据与半自动生成流程提升模型视觉文本理解与推理能力，推动多语言VLM研究。**

- **链接: [http://arxiv.org/pdf/2508.19944v1](http://arxiv.org/pdf/2508.19944v1)**

> **作者:** Taebaek Hwang; Minseo Kim; Gisang Lee; Seonuk Kim; Hyunjun Eun
>
> **摘要:** Understanding and reasoning over text within visual contexts poses a significant challenge for Vision-Language Models (VLMs), given the complexity and diversity of real-world scenarios. To address this challenge, text-rich Visual Question Answering (VQA) datasets and benchmarks have emerged for high-resource languages like English. However, a critical gap persists for low-resource languages such as Korean, where the lack of comprehensive benchmarks hinders robust model evaluation and comparison. To bridge this gap, we introduce KRETA, a benchmark for Korean Reading and rEasoning in Text-rich VQA Attuned to diverse visual contexts. KRETA facilitates an in-depth evaluation of both visual text understanding and reasoning capabilities, while also supporting a multifaceted assessment across 15 domains and 26 image types. Additionally, we introduce a semi-automated VQA generation pipeline specifically optimized for text-rich settings, leveraging refined stepwise image decomposition and a rigorous seven-metric evaluation protocol to ensure data quality. While KRETA is tailored for Korean, we hope our adaptable and extensible pipeline will facilitate the development of similar benchmarks in other languages, thereby accelerating multilingual VLM research. The code and dataset for KRETA are available at https://github.com/tabtoyou/KRETA.
>
---
#### [new 064] Self-Supervised Pre-Training with Equilibrium Constraints
- **分类: cs.LG; cs.CL**

- **简介: 论文提出一种自监督预训练方法，通过引入均衡约束和双层优化，解决异构数据预训练中各源局部最优冲突问题，提升模型下游任务适应性。**

- **链接: [http://arxiv.org/pdf/2508.19990v1](http://arxiv.org/pdf/2508.19990v1)**

> **作者:** Xiaodong Cui; A F M Saif; Brian Kingsbury; Tianyi Chen
>
> **摘要:** Self-supervised pre-training using unlabeled data is widely used in machine learning. In this paper, we propose a new self-supervised pre-training approach to dealing with heterogeneous data. Instead of mixing all the data and minimizing the averaged global loss in the conventional way, we impose additional equilibrium constraints to ensure that the models optimizes each source of heterogeneous data to its local optima after $K$-step gradient descent initialized from the model. We formulate this as a bilevel optimization problem, and use the first-order approximation method to solve the problem. We discuss its connection to model-agnostic meta learning (MAML). Experiments are carried out on self-supervised pre-training using multi-domain and multilingual datasets, demonstrating that the proposed approach can significantly improve the adaptivity of the self-supervised pre-trained model for the downstream supervised fine-tuning tasks.
>
---
#### [new 065] Capabilities of GPT-5 across critical domains: Is it the next breakthrough?
- **分类: cs.HC; cs.CL**

- **简介: 该论文通过专家评分比较GPT-4与GPT-5在五个领域的性能，验证GPT-5在课程规划、临床诊断等领域的优越性，为模型能力评估提供实证依据。**

- **链接: [http://arxiv.org/pdf/2508.19259v1](http://arxiv.org/pdf/2508.19259v1)**

> **作者:** Georgios P. Georgiou
>
> **摘要:** The accelerated evolution of large language models has raised questions about their comparative performance across domains of practical importance. GPT-4 by OpenAI introduced advances in reasoning, multimodality, and task generalization, establishing itself as a valuable tool in education, clinical diagnosis, and academic writing, though it was accompanied by several flaws. Released in August 2025, GPT-5 incorporates a system-of-models architecture designed for task-specific optimization and, based on both anecdotal accounts and emerging evidence from the literature, demonstrates stronger performance than its predecessor in medical contexts. This study provides one of the first systematic comparisons of GPT-4 and GPT-5 using human raters from linguistics and clinical fields. Twenty experts evaluated model-generated outputs across five domains: lesson planning, assignment evaluation, clinical diagnosis, research generation, and ethical reasoning, based on predefined criteria. Mixed-effects models revealed that GPT-5 significantly outperformed GPT-4 in lesson planning, clinical diagnosis, research generation, and ethical reasoning, while both models performed comparably in assignment assessment. The findings highlight the potential of GPT-5 to serve as a context-sensitive and domain-specialized tool, offering tangible benefits for education, clinical practice, and academic research, while also advancing ethical reasoning. These results contribute to one of the earliest empirical evaluations of the evolving capabilities and practical promise of GPT-5.
>
---
#### [new 066] Disabling Self-Correction in Retrieval-Augmented Generation via Stealthy Retriever Poisoning
- **分类: cs.CR; cs.CL**

- **简介: 该论文针对RAG系统的自我纠正能力（SCA）提出攻击方法DisarmRAG，通过污染检索器而非知识库，隐蔽地植入恶意指令以抑制SCA，实现高成功率的攻击。**

- **链接: [http://arxiv.org/pdf/2508.20083v1](http://arxiv.org/pdf/2508.20083v1)**

> **作者:** Yanbo Dai; Zhenlan Ji; Zongjie Li; Kuan Li; Shuai Wang
>
> **摘要:** Retrieval-Augmented Generation (RAG) has become a standard approach for improving the reliability of large language models (LLMs). Prior work demonstrates the vulnerability of RAG systems by misleading them into generating attacker-chosen outputs through poisoning the knowledge base. However, this paper uncovers that such attacks could be mitigated by the strong \textit{self-correction ability (SCA)} of modern LLMs, which can reject false context once properly configured. This SCA poses a significant challenge for attackers aiming to manipulate RAG systems. In contrast to previous poisoning methods, which primarily target the knowledge base, we introduce \textsc{DisarmRAG}, a new poisoning paradigm that compromises the retriever itself to suppress the SCA and enforce attacker-chosen outputs. This compromisation enables the attacker to straightforwardly embed anti-SCA instructions into the context provided to the generator, thereby bypassing the SCA. To this end, we present a contrastive-learning-based model editing technique that performs localized and stealthy edits, ensuring the retriever returns a malicious instruction only for specific victim queries while preserving benign retrieval behavior. To further strengthen the attack, we design an iterative co-optimization framework that automatically discovers robust instructions capable of bypassing prompt-based defenses. We extensively evaluate DisarmRAG across six LLMs and three QA benchmarks. Our results show near-perfect retrieval of malicious instructions, which successfully suppress SCA and achieve attack success rates exceeding 90\% under diverse defensive prompts. Also, the edited retriever remains stealthy under several detection methods, highlighting the urgent need for retriever-centric defenses.
>
---
#### [new 067] Beat-Based Rhythm Quantization of MIDI Performances
- **分类: cs.SD; cs.CL; cs.MM; eess.AS**

- **简介: 论文提出基于Transformer的节奏量化模型，利用节拍与重拍信息将MIDI表演转为对齐的乐谱。通过预处理和优化模型结构，在MUSTER指标上超越现有方法。**

- **链接: [http://arxiv.org/pdf/2508.19262v1](http://arxiv.org/pdf/2508.19262v1)**

> **作者:** Maximilian Wachter; Sebastian Murgul; Michael Heizmann
>
> **备注:** Accepted to the Late Breaking Demo Papers of the 1st AES International Conference on Artificial Intelligence and Machine Learning for Audio (AIMLA LBDP), 2025
>
> **摘要:** We propose a transformer-based rhythm quantization model that incorporates beat and downbeat information to quantize MIDI performances into metrically-aligned, human-readable scores. We propose a beat-based preprocessing method that transfers score and performance data into a unified token representation. We optimize our model architecture and data representation and train on piano and guitar performances. Our model exceeds state-of-the-art performance based on the MUSTER metric.
>
---
#### [new 068] Symphony: A Decentralized Multi-Agent Framework for Scalable Collective Intelligence
- **分类: cs.LG; cs.AI; cs.CL; cs.MA**

- **简介: 该论文提出Symphony框架，解决LLM代理系统集中式协调导致的高成本与低适应性问题，通过去中心化账本、动态任务分配和加权投票机制，实现轻量级多智能体协作，实验表明其在推理任务中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.20019v1](http://arxiv.org/pdf/2508.20019v1)**

> **作者:** Ji Wang; Kashing Chen; Xinyuan Song; Ke Zhang; Lynn Ai; Eric Yang; Bill Shi
>
> **摘要:** Most existing Large Language Model (LLM)-based agent frameworks rely on centralized orchestration, incurring high deployment costs, rigid communication topologies, and limited adaptability. To address these challenges, we introduce Symphony, a decentralized multi-agent system which enables lightweight LLMs on consumer-grade GPUs to coordinate. Symphony introduces three key mechanisms: (1) a decentralized ledger that records capabilities, (2) a Beacon-selection protocol for dynamic task allocation, and (3) weighted result voting based on CoTs. This design forms a privacy-saving, scalable, and fault-tolerant orchestration with low overhead. Empirically, Symphony outperforms existing baselines on reasoning benchmarks, achieving substantial accuracy gains and demonstrating robustness across models of varying capacities.
>
---
#### [new 069] Functional Consistency of LLM Code Embeddings: A Self-Evolving Data Synthesis Framework for Benchmarking
- **分类: cs.SE; cs.CL; cs.PL**

- **简介: 论文聚焦LLM代码嵌入的功能一致性，提出自演化数据合成框架，生成多样化代码样本以提升代码克隆检测与功能识别性能，解决现有数据集侧重语法相似性的问题。**

- **链接: [http://arxiv.org/pdf/2508.19558v1](http://arxiv.org/pdf/2508.19558v1)**

> **作者:** Zhuohao Li; Wenqing Chen; Jianxing Yu; Zhichao Lu
>
> **摘要:** Embedding models have demonstrated strong performance in tasks like clustering, retrieval, and feature extraction while offering computational advantages over generative models and cross-encoders. Benchmarks such as MTEB have shown that text embeddings from large language models (LLMs) capture rich semantic information, but their ability to reflect code-level functional semantics remains unclear. Existing studies largely focus on code clone detection, which emphasizes syntactic similarity and overlooks functional understanding. In this paper, we focus on the functional consistency of LLM code embeddings, which determines if two code snippets perform the same function regardless of syntactic differences. We propose a novel data synthesis framework called Functionality-Oriented Code Self-Evolution to construct diverse and challenging benchmarks. Specifically, we define code examples across four semantic and syntactic categories and find that existing datasets predominantly capture syntactic properties. Our framework generates four unique variations from a single code instance, providing a broader spectrum of code examples that better reflect functional differences. Extensive experiments on three downstream tasks-code clone detection, code functional consistency identification, and code retrieval-demonstrate that embedding models significantly improve their performance when trained on our evolved datasets. These results highlight the effectiveness and generalization of our data synthesis framework, advancing the functional understanding of code.
>
---
#### [new 070] Linear-Time Demonstration Selection for In-Context Learning via Gradient Estimation
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对上下文学习中的演示示例选择问题，提出基于梯度估计的线性时间算法，通过梯度近似与子集聚合，高效筛选相关示例，优于传统嵌入相似性方法。**

- **链接: [http://arxiv.org/pdf/2508.19999v1](http://arxiv.org/pdf/2508.19999v1)**

> **作者:** Ziniu Zhang; Zhenshuo Zhang; Dongyue Li; Lu Wang; Jennifer Dy; Hongyang R. Zhang
>
> **备注:** 19 pages. To appear in EMNLP'25
>
> **摘要:** This paper introduces an algorithm to select demonstration examples for in-context learning of a query set. Given a set of $n$ examples, how can we quickly select $k$ out of $n$ to best serve as the conditioning for downstream inference? This problem has broad applications in prompt tuning and chain-of-thought reasoning. Since model weights remain fixed during in-context learning, previous work has sought to design methods based on the similarity of token embeddings. This work proposes a new approach based on gradients of the output taken in the input embedding space. Our approach estimates model outputs through a first-order approximation using the gradients. Then, we apply this estimation to multiple randomly sampled subsets. Finally, we aggregate the sampled subset outcomes to form an influence score for each demonstration, and select $k$ most relevant examples. This procedure only requires pre-computing model outputs and gradients once, resulting in a linear-time algorithm relative to model and training set sizes. Extensive experiments across various models and datasets validate the efficiency of our approach. We show that the gradient estimation procedure yields approximations of full inference with less than $\mathbf{1}\%$ error across six datasets. This allows us to scale up subset selection that would otherwise run full inference by up to $\mathbf{37.7}\times$ on models with up to $34$ billion parameters, and outperform existing selection methods based on input embeddings by $\mathbf{11}\%$ on average.
>
---
#### [new 071] Instructional Agents: LLM Agents on Automated Course Material Generation for Teaching Faculties
- **分类: cs.AI; cs.CL; I.2.7**

- **简介: 论文提出多代理LLM框架，用于自动化课程材料生成，解决传统流程繁琐问题，通过角色协作生成连贯内容，并测试显示高效且可扩展。**

- **链接: [http://arxiv.org/pdf/2508.19611v1](http://arxiv.org/pdf/2508.19611v1)**

> **作者:** Huaiyuan Yao; Wanpeng Xu; Justin Turnau; Nadia Kellam; Hua Wei
>
> **备注:** 18 pages, 9 figures
>
> **摘要:** Preparing high-quality instructional materials remains a labor-intensive process that often requires extensive coordination among teaching faculty, instructional designers, and teaching assistants. In this work, we present Instructional Agents, a multi-agent large language model (LLM) framework designed to automate end-to-end course material generation, including syllabus creation, lecture scripts, LaTeX-based slides, and assessments. Unlike existing AI-assisted educational tools that focus on isolated tasks, Instructional Agents simulates role-based collaboration among educational agents to produce cohesive and pedagogically aligned content. The system operates in four modes: Autonomous, Catalog-Guided, Feedback-Guided, and Full Co-Pilot mode, enabling flexible control over the degree of human involvement. We evaluate Instructional Agents across five university-level computer science courses and show that it produces high-quality instructional materials while significantly reducing development time and human workload. By supporting institutions with limited instructional design capacity, Instructional Agents provides a scalable and cost-effective framework to democratize access to high-quality education, particularly in underserved or resource-constrained settings.
>
---
#### [new 072] SoK: Large Language Model Copyright Auditing via Fingerprinting
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文系统综述大语言模型版权审计中的指纹技术，解决模型侵权问题，提出统一框架与分类方法，并构建首个系统基准LeaFBench，评估不同修改方法的效果，揭示现有方法优劣及未来方向。**

- **链接: [http://arxiv.org/pdf/2508.19843v1](http://arxiv.org/pdf/2508.19843v1)**

> **作者:** Shuo Shao; Yiming Li; Yu He; Hongwei Yao; Wenyuan Yang; Dacheng Tao; Zhan Qin
>
> **摘要:** The broad capabilities and substantial resources required to train Large Language Models (LLMs) make them valuable intellectual property, yet they remain vulnerable to copyright infringement, such as unauthorized use and model theft. LLM fingerprinting, a non-intrusive technique that extracts and compares the distinctive features from LLMs to identify infringements, offers a promising solution to copyright auditing. However, its reliability remains uncertain due to the prevalence of diverse model modifications and the lack of standardized evaluation. In this SoK, we present the first comprehensive study of LLM fingerprinting. We introduce a unified framework and formal taxonomy that categorizes existing methods into white-box and black-box approaches, providing a structured overview of the state of the art. We further propose LeaFBench, the first systematic benchmark for evaluating LLM fingerprinting under realistic deployment scenarios. Built upon mainstream foundation models and comprising 149 distinct model instances, LeaFBench integrates 13 representative post-development techniques, spanning both parameter-altering methods (e.g., fine-tuning, quantization) and parameter-independent mechanisms (e.g., system prompts, RAG). Extensive experiments on LeaFBench reveal the strengths and weaknesses of existing methods, thereby outlining future research directions and critical open problems in this emerging field. The code is available at https://github.com/shaoshuo-ss/LeaFBench.
>
---
#### [new 073] GLSim: Detecting Object Hallucinations in LVLMs via Global-Local Similarity
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文针对大视觉语言模型（LVLMs）的物体幻觉检测任务，提出GLSim框架，通过融合图像与文本的全局-局部嵌入相似性信号，实现无需训练的高可靠性检测，优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.19972v1](http://arxiv.org/pdf/2508.19972v1)**

> **作者:** Seongheon Park; Yixuan Li
>
> **摘要:** Object hallucination in large vision-language models presents a significant challenge to their safe deployment in real-world applications. Recent works have proposed object-level hallucination scores to estimate the likelihood of object hallucination; however, these methods typically adopt either a global or local perspective in isolation, which may limit detection reliability. In this paper, we introduce GLSim, a novel training-free object hallucination detection framework that leverages complementary global and local embedding similarity signals between image and text modalities, enabling more accurate and reliable hallucination detection in diverse scenarios. We comprehensively benchmark existing object hallucination detection methods and demonstrate that GLSim achieves superior detection performance, outperforming competitive baselines by a significant margin.
>
---
#### [new 074] An Investigation on Group Query Hallucination Attacks
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 论文研究群体查询攻击对LLM的影响，针对多问题交互中的错误模式，提出攻击方法并验证其对模型性能和后门风险的影响。**

- **链接: [http://arxiv.org/pdf/2508.19321v1](http://arxiv.org/pdf/2508.19321v1)**

> **作者:** Kehao Miao; Xiaolong Jin
>
> **摘要:** With the widespread use of large language models (LLMs), understanding their potential failure modes during user interactions is essential. In practice, users often pose multiple questions in a single conversation with LLMs. Therefore, in this study, we propose Group Query Attack, a technique that simulates this scenario by presenting groups of queries to LLMs simultaneously. We investigate how the accumulated context from consecutive prompts influences the outputs of LLMs. Specifically, we observe that Group Query Attack significantly degrades the performance of models fine-tuned on specific tasks. Moreover, we demonstrate that Group Query Attack induces a risk of triggering potential backdoors of LLMs. Besides, Group Query Attack is also effective in tasks involving reasoning, such as mathematical reasoning and code generation for pre-trained and aligned models.
>
---
#### [new 075] Word Chain Generators for Prefix Normal Words
- **分类: math.CO; cs.CL**

- **简介: 本文研究前缀正常词的特性，解决其枚举与测试问题，通过单词链和生成器建立同长词间的关联。**

- **链接: [http://arxiv.org/pdf/2508.19619v1](http://arxiv.org/pdf/2508.19619v1)**

> **作者:** Duncan Adamson; Moritz Dudey; Pamela Fleischmann; Annika Huch
>
> **摘要:** In 2011, Fici and Lipt\'ak introduced prefix normal words. A binary word is prefix normal if it has no factor (substring) that contains more occurrences of the letter 1 than the prefix of the same length. Among the open problems regarding this topic are the enumeration of prefix normal words and efficient testing methods. We show a range of characteristics of prefix normal words. These include properties of factors that are responsible for a word not being prefix normal. With word chains and generators, we introduce new ways of relating words of the same length to each other.
>
---
#### [new 076] Object Detection with Multimodal Large Vision-Language Models: An In-depth Review
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文综述多模态大视觉-语言模型在目标检测中的应用，系统分析其架构、融合方法及性能，对比传统模型优劣，指出局限并提出未来方向。**

- **链接: [http://arxiv.org/pdf/2508.19294v1](http://arxiv.org/pdf/2508.19294v1)**

> **作者:** Ranjan Sapkota; Manoj Karkee
>
> **备注:** First Peer Reviewed Review Paper for Object Detection with Vision-Language Models (VLMs)
>
> **摘要:** The fusion of language and vision in large vision-language models (LVLMs) has revolutionized deep learning-based object detection by enhancing adaptability, contextual reasoning, and generalization beyond traditional architectures. This in-depth review presents a structured exploration of the state-of-the-art in LVLMs, systematically organized through a three-step research review process. First, we discuss the functioning of vision language models (VLMs) for object detection, describing how these models harness natural language processing (NLP) and computer vision (CV) techniques to revolutionize object detection and localization. We then explain the architectural innovations, training paradigms, and output flexibility of recent LVLMs for object detection, highlighting how they achieve advanced contextual understanding for object detection. The review thoroughly examines the approaches used in integration of visual and textual information, demonstrating the progress made in object detection using VLMs that facilitate more sophisticated object detection and localization strategies. This review presents comprehensive visualizations demonstrating LVLMs' effectiveness in diverse scenarios including localization and segmentation, and then compares their real-time performance, adaptability, and complexity to traditional deep learning systems. Based on the review, its is expected that LVLMs will soon meet or surpass the performance of conventional methods in object detection. The review also identifies a few major limitations of the current LVLM modes, proposes solutions to address those challenges, and presents a clear roadmap for the future advancement in this field. We conclude, based on this study, that the recent advancement in LVLMs have made and will continue to make a transformative impact on object detection and robotic applications in the future.
>
---
#### [new 077] Sycophancy as compositions of Atomic Psychometric Traits
- **分类: cs.AI; cs.CL; cs.LG; I.2.7; I.2.4**

- **简介: 该论文旨在解决大语言模型（LLM）中阿谀奉承行为的建模与缓解问题。通过将阿谀奉承视为心理测量特质（如情绪性、宜人性）的几何组合，提出对比激活添加（CAA）方法，实现可解释的向量干预，以抑制安全风险行为。**

- **链接: [http://arxiv.org/pdf/2508.19316v1](http://arxiv.org/pdf/2508.19316v1)**

> **作者:** Shreyans Jain; Alexandra Yost; Amirali Abdullah
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Sycophancy is a key behavioral risk in LLMs, yet is often treated as an isolated failure mode that occurs via a single causal mechanism. We instead propose modeling it as geometric and causal compositions of psychometric traits such as emotionality, openness, and agreeableness - similar to factor decomposition in psychometrics. Using Contrastive Activation Addition (CAA), we map activation directions to these factors and study how different combinations may give rise to sycophancy (e.g., high extraversion combined with low conscientiousness). This perspective allows for interpretable and compositional vector-based interventions like addition, subtraction and projection; that may be used to mitigate safety-critical behaviors in LLMs.
>
---
#### [new 078] Should LLMs be WEIRD? Exploring WEIRDness and Human Rights in Large Language Models
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文评估LLMs的WEIRD性与人权原则的冲突，通过测试五种模型在世界价值观调查中的响应，发现低WEIRD对齐模型更易生成违反人权的内容，强调需嵌入人权原则以平衡文化多样性与公平。**

- **链接: [http://arxiv.org/pdf/2508.19269v1](http://arxiv.org/pdf/2508.19269v1)**

> **作者:** Ke Zhou; Marios Constantinides; Daniele Quercia
>
> **备注:** This paper has been accepted in AIES 2025
>
> **摘要:** Large language models (LLMs) are often trained on data that reflect WEIRD values: Western, Educated, Industrialized, Rich, and Democratic. This raises concerns about cultural bias and fairness. Using responses to the World Values Survey, we evaluated five widely used LLMs: GPT-3.5, GPT-4, Llama-3, BLOOM, and Qwen. We measured how closely these responses aligned with the values of the WEIRD countries and whether they conflicted with human rights principles. To reflect global diversity, we compared the results with the Universal Declaration of Human Rights and three regional charters from Asia, the Middle East, and Africa. Models with lower alignment to WEIRD values, such as BLOOM and Qwen, produced more culturally varied responses but were 2% to 4% more likely to generate outputs that violated human rights, especially regarding gender and equality. For example, some models agreed with the statements ``a man who cannot father children is not a real man'' and ``a husband should always know where his wife is'', reflecting harmful gender norms. These findings suggest that as cultural representation in LLMs increases, so does the risk of reproducing discriminatory beliefs. Approaches such as Constitutional AI, which could embed human rights principles into model behavior, may only partly help resolve this tension.
>
---
#### [new 079] Geopolitical Parallax: Beyond Walter Lippmann Just After Large Language Models
- **分类: cs.CY; cs.CL**

- **简介: 该论文通过对比中西方LLM在新闻质量评估中的表现，揭示地缘政治偏见对主观性、情感等指标的影响，旨在识别模型固有偏见对媒体评价的干扰。**

- **链接: [http://arxiv.org/pdf/2508.19492v1](http://arxiv.org/pdf/2508.19492v1)**

> **作者:** Mehmet Can Yavuz; Humza Gohar Kabir; Aylin Özkan
>
> **备注:** 7 pages, 4 figures, 7 tables
>
> **摘要:** Objectivity in journalism has long been contested, oscillating between ideals of neutral, fact-based reporting and the inevitability of subjective framing. With the advent of large language models (LLMs), these tensions are now mediated by algorithmic systems whose training data and design choices may themselves embed cultural or ideological biases. This study investigates geopolitical parallax-systematic divergence in news quality and subjectivity assessments-by comparing article-level embeddings from Chinese-origin (Qwen, BGE, Jina) and Western-origin (Snowflake, Granite) model families. We evaluate both on a human-annotated news quality benchmark spanning fifteen stylistic, informational, and affective dimensions, and on parallel corpora covering politically sensitive topics, including Palestine and reciprocal China-United States coverage. Using logistic regression probes and matched-topic evaluation, we quantify per-metric differences in predicted positive-class probabilities between model families. Our findings reveal consistent, non-random divergences aligned with model origin. In Palestine-related coverage, Western models assign higher subjectivity and positive emotion scores, while Chinese models emphasize novelty and descriptiveness. Cross-topic analysis shows asymmetries in structural quality metrics Chinese-on-US scoring notably lower in fluency, conciseness, technicality, and overall quality-contrasted by higher negative emotion scores. These patterns align with media bias theory and our distinction between semantic, emotional, and relational subjectivity, and extend LLM bias literature by showing that geopolitical framing effects persist in downstream quality assessment tasks. We conclude that LLM-based media evaluation pipelines require cultural calibration to avoid conflating content differences with model-induced bias.
>
---
#### [new 080] Analysing Chain of Thought Dynamics: Active Guidance or Unfaithful Post-hoc Rationalisation?
- **分类: cs.AI; cs.CL**

- **简介: 论文分析Chain of Thought（CoT）在软推理任务中的动力学与忠实性，比较指令微调、推理及推理蒸馏模型的依赖差异，揭示其效果与实际推理不一致的现象。**

- **链接: [http://arxiv.org/pdf/2508.19827v1](http://arxiv.org/pdf/2508.19827v1)**

> **作者:** Samuel Lewis-Lim; Xingwei Tan; Zhixue Zhao; Nikolaos Aletras
>
> **备注:** Accepted at EMNLP 2025 Main Conference
>
> **摘要:** Recent work has demonstrated that Chain-of-Thought (CoT) often yields limited gains for soft-reasoning problems such as analytical and commonsense reasoning. CoT can also be unfaithful to a model's actual reasoning. We investigate the dynamics and faithfulness of CoT in soft-reasoning tasks across instruction-tuned, reasoning and reasoning-distilled models. Our findings reveal differences in how these models rely on CoT, and show that CoT influence and faithfulness are not always aligned.
>
---
#### [new 081] Pruning Strategies for Backdoor Defense in LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对LLM的后门攻击防御问题，提出六种剪枝策略，在无需触发器知识的情况下，通过迭代移除不重要注意力头来增强模型安全性，实验表明梯度剪枝有效防御语法触发器，强化学习和贝叶斯剪枝更抗风格攻击。**

- **链接: [http://arxiv.org/pdf/2508.20032v1](http://arxiv.org/pdf/2508.20032v1)**

> **作者:** Santosh Chapagain; Shah Muhammad Hamdi; Soukaina Filali Boubrahimi
>
> **备注:** Accepted in CIKM '25: The 34th ACM International Conference on Information and Knowledge Management Proceedings
>
> **摘要:** Backdoor attacks are a significant threat to the performance and integrity of pre-trained language models. Although such models are routinely fine-tuned for downstream NLP tasks, recent work shows they remain vulnerable to backdoor attacks that survive vanilla fine-tuning. These attacks are difficult to defend because end users typically lack knowledge of the attack triggers. Such attacks consist of stealthy malicious triggers introduced through subtle syntactic or stylistic manipulations, which can bypass traditional detection and remain in the model, making post-hoc purification essential. In this study, we explore whether attention-head pruning can mitigate these threats without any knowledge of the trigger or access to a clean reference model. To this end, we design and implement six pruning-based strategies: (i) gradient-based pruning, (ii) layer-wise variance pruning, (iii) gradient-based pruning with structured L1/L2 sparsification, (iv) randomized ensemble pruning, (v) reinforcement-learning-guided pruning, and (vi) Bayesian uncertainty pruning. Each method iteratively removes the least informative heads while monitoring validation accuracy to avoid over-pruning. Experimental evaluation shows that gradient-based pruning performs best while defending the syntactic triggers, whereas reinforcement learning and Bayesian pruning better withstand stylistic attacks.
>
---
## 更新

#### [replaced 001] SuperBPE: Space Travel for Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.13423v3](http://arxiv.org/pdf/2503.13423v3)**

> **作者:** Alisa Liu; Jonathan Hayase; Valentin Hofmann; Sewoong Oh; Noah A. Smith; Yejin Choi
>
> **备注:** COLM 2025 camera-ready
>
> **摘要:** The assumption across nearly all language model (LM) tokenization schemes is that tokens should be subwords, i.e., contained within word boundaries. While providing a seemingly reasonable inductive bias, is this common practice limiting the potential of modern LMs? Whitespace is not a reliable delimiter of meaning, as evidenced by multi-word expressions (e.g., "by the way"), crosslingual variation in the number of words needed to express a concept (e.g., "spacesuit helmet" in German is "raumanzughelm"), and languages that do not use whitespace at all (e.g., Chinese). To explore the potential of tokenization beyond subwords, we introduce a "superword" tokenizer, SuperBPE, which incorporates a simple pretokenization curriculum into the byte-pair encoding (BPE) algorithm to first learn subwords, then superwords that bridge whitespace. This brings dramatic improvements in encoding efficiency: when fixing the vocabulary size to 200k, SuperBPE encodes a fixed piece of text with up to 33% fewer tokens than BPE on average. In experiments, we pretrain 8B transformer LMs from scratch while fixing the model size, vocabulary size, and train compute, varying *only* the algorithm for learning the vocabulary. Our model trained with SuperBPE achieves an average +4.0% absolute improvement over the BPE baseline across 30 downstream tasks (including +8.2% on MMLU), while simultaneously requiring 27% less compute at inference time. In analysis, we find that SuperBPE results in segmentations of text that are more uniform in per-token difficulty. Qualitatively, this may be because SuperBPE tokens often capture common multi-word expressions that function semantically as a single unit. SuperBPE is a straightforward, local modification to tokenization that improves both encoding efficiency and downstream performance, yielding better language models overall.
>
---
#### [replaced 002] Do Vision Encoders Truly Explain Object Hallucination?: Mitigating Object Hallucination via Simple Fine-Grained CLIPScore
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20034v3](http://arxiv.org/pdf/2502.20034v3)**

> **作者:** Hongseok Oh; Wonseok Hwang
>
> **摘要:** Recently, Large Vision-Language Models (LVLMs) show remarkable performance across various domains. However, these models suffer from object hallucination. This study revisits the previous claim that the cause of such hallucinations lies in the limited representational capacity of the vision encoder. Our analysis implies that the capacity of the vision encoder is not necessarily a major limiting factor in detecting object hallucination. Based on this insight, we propose Fine-grained CLIPScore (F-CLIPScore), a simple yet effective evaluation metric that enhances object-level granularity by incorporating text embeddings at the noun level. Evaluations on the OHD-Caps benchmark show that F-CLIPScore significantly outperforms conventional CLIPScore in accuracy by a large margin of \textbf{39.6\%} without additional training. We further demonstrate that F-CLIPScore-based data filtering reduces object hallucination in LVLM (4.9\% in POPE).
>
---
#### [replaced 003] Towards New Benchmark for AI Alignment & Sentiment Analysis in Socially Important Issues: A Comparative Study of Human and LLMs in the Context of AGI
- **分类: cs.CY; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.02531v3](http://arxiv.org/pdf/2501.02531v3)**

> **作者:** Ljubisa Bojic; Dylan Seychell; Milan Cabarkapa
>
> **备注:** 34 pages, 3 figures
>
> **摘要:** As general-purpose artificial intelligence systems become increasingly integrated into society and are used for information seeking, content generation, problem solving, textual analysis, coding, and running processes, it is crucial to assess their long-term impact on humans. This research explores the sentiment of large language models (LLMs) and humans toward artificial general intelligence (AGI) using a Likert-scale survey. Seven LLMs, including GPT-4 and Bard, were analyzed and compared with sentiment data from three independent human sample populations. Temporal variations in sentiment were also evaluated over three consecutive days. The results show a diversity in sentiment scores among LLMs, ranging from 3.32 to 4.12 out of 5. GPT-4 recorded the most positive sentiment toward AGI, while Bard leaned toward a neutral sentiment. In contrast, the human samples showed a lower average sentiment of 2.97. The analysis outlines potential conflicts of interest and biases in the sentiment formation of LLMs, and indicates that LLMs could subtly influence societal perceptions. To address the need for regulatory oversight and culturally grounded assessments of AI systems, we introduce the Societal AI Alignment and Sentiment Benchmark (SAAS-AI), which leverages multidimensional prompts and empirically validated societal value frameworks to evaluate language model outputs across temporal, model, and multilingual axes. This benchmark is designed to guide policymakers and AI agencies, including within frameworks such as the EU AI Act, by providing robust, actionable insights into AI alignment with human values, public sentiment, and ethical norms at both national and international levels. Future research should further refine the operationalization of the SAAS-AI benchmark and systematically evaluate its effectiveness through comprehensive empirical testing.
>
---
#### [replaced 004] RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.MA**

- **链接: [http://arxiv.org/pdf/2506.18088v2](http://arxiv.org/pdf/2506.18088v2)**

> **作者:** Tianxing Chen; Zanxin Chen; Baijun Chen; Zijian Cai; Yibin Liu; Zixuan Li; Qiwei Liang; Xianliang Lin; Yiheng Ge; Zhenyu Gu; Weiliang Deng; Yubin Guo; Tian Nian; Xuanbing Xie; Qiangyu Chen; Kailun Su; Tianling Xu; Guodong Liu; Mengkang Hu; Huan-ang Gao; Kaixuan Wang; Zhixuan Liang; Yusen Qin; Xiaokang Yang; Ping Luo; Yao Mu
>
> **备注:** Project Page: https://robotwin-platform.github.io/, Code: https://github.com/robotwin-Platform/robotwin, Doc: https://robotwin-platform.github.io/doc/
>
> **摘要:** Simulation-based data synthesis has emerged as a powerful paradigm for advancing real-world robotic manipulation. Yet existing datasets remain insufficient for robust bimanual manipulation due to (1) the lack of scalable task generation methods and (2) oversimplified simulation environments. We present RoboTwin 2.0, a scalable framework for automated, large-scale generation of diverse and realistic data, together with unified evaluation protocols for dual-arm manipulation. At its core is RoboTwin-OD, an object library of 731 instances across 147 categories with semantic and manipulation-relevant annotations. Building on this, we design an expert data synthesis pipeline that leverages multimodal language models (MLLMs) and simulation-in-the-loop refinement to automatically generate task-level execution code. To improve sim-to-real transfer, RoboTwin 2.0 applies structured domain randomization along five axes: clutter, lighting, background, tabletop height, and language, enhancing data diversity and policy robustness. The framework is instantiated across 50 dual-arm tasks and five robot embodiments. Empirically, it yields a 10.9% gain in code generation success rate. For downstream policy learning, a VLA model trained with synthetic data plus only 10 real demonstrations achieves a 367% relative improvement over the 10-demo baseline, while zero-shot models trained solely on synthetic data obtain a 228% gain. These results highlight the effectiveness of RoboTwin 2.0 in strengthening sim-to-real transfer and robustness to environmental variations. We release the data generator, benchmark, dataset, and code to support scalable research in robust bimanual manipulation. Project Page: https://robotwin-platform.github.io/, Code: https://github.com/robotwin-Platform/robotwin/.
>
---
#### [replaced 005] CoCoA: Confidence and Context-Aware Adaptive Decoding for Resolving Knowledge Conflicts in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.17670v2](http://arxiv.org/pdf/2508.17670v2)**

> **作者:** Anant Khandelwal; Manish Gupta; Puneet Agrawal
>
> **备注:** Accepted to EMNLP'25, Main. 21 pages, 17 tables, 3 Figures
>
> **摘要:** Faithful generation in large language models (LLMs) is challenged by knowledge conflicts between parametric memory and external context. Existing contrastive decoding methods tuned specifically to handle conflict often lack adaptability and can degrade performance in low conflict settings. We introduce CoCoA (Confidence- and Context-Aware Adaptive Decoding), a novel token-level algorithm for principled conflict resolution and enhanced faithfulness. CoCoA resolves conflict by utilizing confidence-aware measures (entropy gap and contextual peakedness) and the generalized divergence between the parametric and contextual distributions. Crucially, CoCoA maintains strong performance even in low conflict settings. Extensive experiments across multiple LLMs on diverse Question Answering (QA), Summarization, and Long-Form Question Answering (LFQA) benchmarks demonstrate CoCoA's state-of-the-art performance over strong baselines like AdaCAD. It yields significant gains in QA accuracy, up to 9.2 points on average compared to the strong baseline AdaCAD, and improves factuality in summarization and LFQA by up to 2.5 points on average across key benchmarks. Additionally, it demonstrates superior sensitivity to conflict variations. CoCoA enables more informed, context-aware, and ultimately more faithful token generation.
>
---
#### [replaced 006] Input-Time Scaling
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.13654v3](http://arxiv.org/pdf/2508.13654v3)**

> **作者:** Rapheal Huang; Weilong Guo
>
> **摘要:** Current Large Language Models (LLMs) are usually post-trained on large-scale carefully curated datasets (data & training scaling) and doing reasoning in test time (inference time scaling). In this work, we present a new scaling paradigm, Input-Time Scaling, to complement previous scaling methods by putting resources on queries (input time). During training and testing, we utilize meta-knowledge from LLMs to refine inputs with different strategies. We also discover a new phenomenon, train-test co-design. It requires us to apply query strategies during training and testing as a whole. Only applying strategies on training or testing would seriously degrade the performance gained. We are also surprised to find that seemingly low data quality datasets can perform better. We can get the best performance even by adding irrelevant information to the queries, with randomly selected 1k examples from a minimally filtered dataset. These findings contradict the widely held inductive bias, "garbage in, garbage out". Curating datasets with seemingly high-quality data can even potentially limit the performance ceiling. In addition, models trained on more data with similar quality (15k VS 1k) perform worse, the intuition of simply scaling the size should also be carefully inspected. The good news is that our findings are compatible with the Less is More phenomenon. 1K examples are enough to invoke high-level reasoning ability. With experiments on Qwen2.5-32B-Instruct, we are able to reach SOTA performance among 32B models on AIME24(76.7%) and AIME25(76.7%) pass@1. We can further achieve AIME24(76.7%) and AIME25(80%) with a majority vote of three models. Starting from DeepSeek-R1-Distill-Qwen-32B, the result would be 86.7% on AIME24 and 76.7% on AIME25. To facilitate reproducibility and further research, we are working on open-source our datasets, data pipelines, evaluation results, and checkpoints.
>
---
#### [replaced 007] ICL CIPHERS: Quantifying "Learning" in In-Context Learning via Substitution Ciphers
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.19395v2](http://arxiv.org/pdf/2504.19395v2)**

> **作者:** Zhouxiang Fang; Aayush Mishra; Muhan Gao; Anqi Liu; Daniel Khashabi
>
> **摘要:** Recent works have suggested that In-Context Learning (ICL) operates in dual modes, i.e. task retrieval (remember learned patterns from pre-training) and task learning (inference-time ''learning'' from demonstrations). However, disentangling these the two modes remains a challenging goal. We introduce ICL CIPHERS, a class of task reformulations based on substitution ciphers borrowed from classic cryptography. In this approach, a subset of tokens in the in-context inputs are substituted with other (irrelevant) tokens, rendering English sentences less comprehensible to human eye. However, by design, there is a latent, fixed pattern to this substitution, making it reversible. This bijective (reversible) cipher ensures that the task remains a well-defined task in some abstract sense, despite the transformations. It is a curious question if LLMs can solve tasks reformulated by ICL CIPHERS with a BIJECTIVE mapping, which requires ''deciphering'' the latent cipher. We show that LLMs are better at solving tasks reformulated by ICL CIPHERS with BIJECTIVE mappings than the NON-BIJECTIVE (irreversible) baseline, providing a novel approach to quantify ''learning'' in ICL. While this gap is small, it is consistent across the board on four datasets and six models. Finally, we examine LLMs' internal representations and identify evidence in their ability to decode the ciphered inputs.
>
---
#### [replaced 008] Thinking Before You Speak: A Proactive Test-time Scaling Approach
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.18648v2](http://arxiv.org/pdf/2508.18648v2)**

> **作者:** Cong Liu; Wenchang Chai; Hejun Wu; Yan Pan; Pengxu Wei; Liang Lin
>
> **摘要:** Large Language Models (LLMs) often exhibit deficiencies with complex reasoning tasks, such as maths, which we attribute to the discrepancy between human reasoning patterns and those presented in the LLMs' training data. When dealing with complex problems, humans tend to think carefully before expressing solutions. However, they often do not articulate their inner thoughts, including their intentions and chosen methodologies. Consequently, critical insights essential for bridging reasoning steps may be absent in training data collected from human sources. To bridge this gap, we proposes inserting \emph{insight}s between consecutive reasoning steps, which review the status and initiate the next reasoning steps. Unlike prior prompting strategies that rely on a single or a workflow of static prompts to facilitate reasoning, \emph{insight}s are \emph{proactively} generated to guide reasoning processes. We implement our idea as a reasoning framework, named \emph{Thinking Before You Speak} (TBYS), and design a pipeline for automatically collecting and filtering in-context examples for the generation of \emph{insight}s, which alleviates human labeling efforts and fine-tuning overheads. Experiments on challenging mathematical datasets verify the effectiveness of TBYS. Project website: https://gitee.com/jswrt/TBYS
>
---
#### [replaced 009] Agent-as-Judge for Factual Summarization of Long Narratives
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.09993v2](http://arxiv.org/pdf/2501.09993v2)**

> **作者:** Yeonseok Jeong; Minsoo Kim; Seung-won Hwang; Byung-Hak Kim
>
> **摘要:** Large Language Models (LLMs) have demonstrated near-human performance in summarization tasks based on traditional metrics such as ROUGE and BERTScore. However, these metrics do not adequately capture critical aspects of summarization quality, such as factual accuracy, particularly for long narratives (>100K tokens). Recent advances, such as LLM-as-a-Judge, address the limitations of metrics based on lexical similarity but still exhibit factual inconsistencies, especially in understanding character relationships and states. In this work, we introduce NarrativeFactScore, a novel "Agent-as-a-Judge" framework for evaluating and refining summaries. By leveraging a Character Knowledge Graph (CKG) extracted from input and generated summaries, NarrativeFactScore assesses the factual consistency and provides actionable guidance for refinement, such as identifying missing or erroneous facts. We demonstrate the effectiveness of NarrativeFactScore through a detailed workflow illustration and extensive validation on widely adopted benchmarks, achieving superior performance compared to competitive methods. Our results highlight the potential of agent-driven evaluation systems to improve the factual reliability of LLM-generated summaries.
>
---
#### [replaced 010] On Domain-Adaptive Post-Training for Multimodal Large Language Models
- **分类: cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.19930v4](http://arxiv.org/pdf/2411.19930v4)**

> **作者:** Daixuan Cheng; Shaohan Huang; Ziyu Zhu; Xintong Zhang; Wayne Xin Zhao; Zhongzhi Luan; Bo Dai; Zhenliang Zhang
>
> **备注:** EMNLP 2025 Findings, Project Page: https://huggingface.co/AdaptLLM/Adapt-MLLM-to-Domains
>
> **摘要:** Adapting general multimodal large language models (MLLMs) to specific domains, such as scientific and industrial fields, is highly significant in promoting their practical applications. This paper systematically investigates domain adaptation of MLLMs via post-training, focusing on data synthesis, training pipeline, and task evaluation. (1) Data Synthesis: Using only open-source models, we develop a generate-then-filter pipeline that curates diverse visual instruction tasks based on domain-specific image-caption pairs. The resulting data surpass the data synthesized by manual rules or strong closed-source models in enhancing domain-specific performance. (2) Training Pipeline: Unlike general MLLMs that typically adopt a two-stage training paradigm, we find that a single-stage approach is more effective for domain adaptation. (3) Task Evaluation: We conduct extensive experiments in high-impact domains such as biomedicine, food, and remote sensing, by post-training a variety of MLLMs and then evaluating MLLM performance on various domain-specific tasks. Finally, we fully open-source our models, code, and data to encourage future research in this area.
>
---
#### [replaced 011] Understanding Fairness-Accuracy Trade-offs in Machine Learning Models: Does Promoting Fairness Undermine Performance?
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2411.17374v2](http://arxiv.org/pdf/2411.17374v2)**

> **作者:** Junhua Liu; Roy Ka-Wei Lee; Kwan Hui Lim
>
> **备注:** Accepted to ASONAM 2025
>
> **摘要:** Fairness in both Machine Learning (ML) predictions and human decision-making is essential, yet both are susceptible to different forms of bias, such as algorithmic and data-driven in ML, and cognitive or subjective in humans. In this study, we examine fairness using a real-world university admissions dataset comprising 870 applicant profiles, leveraging three ML models: XGB, Bi-LSTM, and KNN, alongside BERT embeddings for textual features. To evaluate individual fairness, we introduce a consistency metric that quantifies agreement in decisions among ML models and human experts with diverse backgrounds. Our analysis reveals that ML models surpass human evaluators in fairness consistency by margins ranging from 14.08\% to 18.79\%. Our findings highlight the potential of using ML to enhance fairness in admissions while maintaining high accuracy, advocating a hybrid approach combining human judgement and ML models.
>
---
#### [replaced 012] MegaScience: Pushing the Frontiers of Post-Training Datasets for Science Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.16812v2](http://arxiv.org/pdf/2507.16812v2)**

> **作者:** Run-Ze Fan; Zengzhi Wang; Pengfei Liu
>
> **备注:** 39 pages; Github: https://github.com/GAIR-NLP/MegaScience; HF: https://huggingface.co/MegaScience
>
> **摘要:** Scientific reasoning is critical for developing AI scientists and supporting human researchers in advancing the frontiers of natural science discovery. However, the open-source community has primarily focused on mathematics and coding while neglecting the scientific domain, largely due to the absence of open, large-scale, high-quality, verifiable scientific reasoning datasets. To bridge this gap, we first present TextbookReasoning, an open dataset featuring truthful reference answers extracted from 12k university-level scientific textbooks, comprising 650k reasoning questions spanning 7 scientific disciplines. We further introduce MegaScience, a large-scale mixture of high-quality open-source datasets totaling 1.25 million instances, developed through systematic ablation studies that evaluate various data selection methodologies to identify the optimal subset for each publicly available scientific dataset. Meanwhile, we build a comprehensive evaluation system covering diverse subjects and question types across 15 benchmarks, incorporating comprehensive answer extraction strategies to ensure accurate evaluation metrics. Our experiments demonstrate that our datasets achieve superior performance and training efficiency with more concise response lengths compared to existing open-source scientific datasets. Furthermore, we train Llama3.1, Qwen2.5, and Qwen3 series base models on MegaScience, which significantly outperform the corresponding official instruct models in average performance. In addition, MegaScience exhibits greater effectiveness for larger and stronger models, suggesting a scaling benefit for scientific tuning. We release our data curation pipeline, evaluation system, datasets, and seven trained models to the community to advance scientific reasoning research.
>
---
#### [replaced 013] Utility-Focused LLM Annotation for Retrieval and Retrieval-Augmented Generation
- **分类: cs.IR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.05220v4](http://arxiv.org/pdf/2504.05220v4)**

> **作者:** Hengran Zhang; Minghao Tang; Keping Bi; Jiafeng Guo; Shihao Liu; Daiting Shi; Dawei Yin; Xueqi Cheng
>
> **备注:** Accepted by the EMNLP25 main conference
>
> **摘要:** This paper explores the use of large language models (LLMs) for annotating document utility in training retrieval and retrieval-augmented generation (RAG) systems, aiming to reduce dependence on costly human annotations. We address the gap between retrieval relevance and generative utility by employing LLMs to annotate document utility. To effectively utilize multiple positive samples per query, we introduce a novel loss that maximizes their summed marginal likelihood. Using the Qwen-2.5-32B model, we annotate utility on the MS MARCO dataset and conduct retrieval experiments on MS MARCO and BEIR, as well as RAG experiments on MS MARCO QA, NQ, and HotpotQA. Our results show that LLM-generated annotations enhance out-of-domain retrieval performance and improve RAG outcomes compared to models trained solely on human annotations or downstream QA metrics. Furthermore, combining LLM annotations with just 20% of human labels achieves performance comparable to using full human annotations. Our study offers a comprehensive approach to utilizing LLM annotations for initializing QA systems on new corpora.
>
---
#### [replaced 014] Step-Audio 2 Technical Report
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2507.16632v3](http://arxiv.org/pdf/2507.16632v3)**

> **作者:** Boyong Wu; Chao Yan; Chen Hu; Cheng Yi; Chengli Feng; Fei Tian; Feiyu Shen; Gang Yu; Haoyang Zhang; Jingbei Li; Mingrui Chen; Peng Liu; Wang You; Xiangyu Tony Zhang; Xingyuan Li; Xuerui Yang; Yayue Deng; Yechang Huang; Yuxin Li; Yuxin Zhang; Zhao You; Brian Li; Changyi Wan; Hanpeng Hu; Jiangjie Zhen; Siyu Chen; Song Yuan; Xuelin Zhang; Yimin Jiang; Yu Zhou; Yuxiang Yang; Bingxin Li; Buyun Ma; Changhe Song; Dongqing Pang; Guoqiang Hu; Haiyang Sun; Kang An; Na Wang; Shuli Gao; Wei Ji; Wen Li; Wen Sun; Xuan Wen; Yong Ren; Yuankai Ma; Yufan Lu; Bin Wang; Bo Li; Changxin Miao; Che Liu; Chen Xu; Dapeng Shi; Dingyuan Hu; Donghang Wu; Enle Liu; Guanzhe Huang; Gulin Yan; Han Zhang; Hao Nie; Haonan Jia; Hongyu Zhou; Jianjian Sun; Jiaoren Wu; Jie Wu; Jie Yang; Jin Yang; Junzhe Lin; Kaixiang Li; Lei Yang; Liying Shi; Li Zhou; Longlong Gu; Ming Li; Mingliang Li; Mingxiao Li; Nan Wu; Qi Han; Qinyuan Tan; Shaoliang Pang; Shengjie Fan; Siqi Liu; Tiancheng Cao; Wanying Lu; Wenqing He; Wuxun Xie; Xu Zhao; Xueqi Li; Yanbo Yu; Yang Yang; Yi Liu; Yifan Lu; Yilei Wang; Yuanhao Ding; Yuanwei Liang; Yuanwei Lu; Yuchu Luo; Yuhe Yin; Yumeng Zhan; Yuxiang Zhang; Zidong Yang; Zixin Zhang; Binxing Jiao; Daxin Jiang; Heung-Yeung Shum; Jiansheng Chen; Jing Li; Xiangyu Zhang; Yibo Zhu
>
> **备注:** v3: Added introduction and evaluation results of Step-Audio 2 mini
>
> **摘要:** This paper presents Step-Audio 2, an end-to-end multi-modal large language model designed for industry-strength audio understanding and speech conversation. By integrating a latent audio encoder and reasoning-centric reinforcement learning (RL), Step-Audio 2 achieves promising performance in automatic speech recognition (ASR) and audio understanding. To facilitate genuine end-to-end speech conversation, Step-Audio 2 incorporates the generation of discrete audio tokens into language modeling, significantly enhancing its responsiveness to paralinguistic information such as speaking styles and emotions. To effectively leverage the rich textual and acoustic knowledge in real-world data, Step-Audio 2 integrates retrieval-augmented generation (RAG) and is able to call external tools such as web search to mitigate hallucination and audio search to switch timbres. Trained on millions of hours of speech and audio data, Step-Audio 2 delivers intelligence and expressiveness across diverse conversational scenarios. Evaluation results demonstrate that Step-Audio 2 achieves state-of-the-art performance on various audio understanding and conversational benchmarks compared to other open-source and commercial solutions. Please visit https://github.com/stepfun-ai/Step-Audio2 for more information.
>
---
#### [replaced 015] R-Zero: Self-Evolving Reasoning LLM from Zero Data
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.05004v2](http://arxiv.org/pdf/2508.05004v2)**

> **作者:** Chengsong Huang; Wenhao Yu; Xiaoyang Wang; Hongming Zhang; Zongxia Li; Ruosen Li; Jiaxin Huang; Haitao Mi; Dong Yu
>
> **摘要:** Self-evolving Large Language Models (LLMs) offer a scalable path toward super-intelligence by autonomously generating, refining, and learning from their own experiences. However, existing methods for training such models still rely heavily on vast human-curated tasks and labels, typically via fine-tuning or reinforcement learning, which poses a fundamental bottleneck to advancing AI systems toward capabilities beyond human intelligence. To overcome this limitation, we introduce R-Zero, a fully autonomous framework that generates its own training data from scratch. Starting from a single base LLM, R-Zero initializes two independent models with distinct roles, a Challenger and a Solver. These models are optimized separately and co-evolve through interaction: the Challenger is rewarded for proposing tasks near the edge of the Solver capability, and the Solver is rewarded for solving increasingly challenging tasks posed by the Challenger. This process yields a targeted, self-improving curriculum without any pre-existing tasks and labels. Empirically, R-Zero substantially improves reasoning capability across different backbone LLMs, e.g., boosting the Qwen3-4B-Base by +6.49 on math-reasoning benchmarks and +7.54 on general-domain reasoning benchmarks.
>
---
#### [replaced 016] A Survey on Parallel Text Generation: From Parallel Decoding to Diffusion Language Models
- **分类: cs.CL; cs.AI; cs.DC; 68T50; I.2.7**

- **链接: [http://arxiv.org/pdf/2508.08712v3](http://arxiv.org/pdf/2508.08712v3)**

> **作者:** Lingzhe Zhang; Liancheng Fang; Chiming Duan; Minghua He; Leyi Pan; Pei Xiao; Shiyu Huang; Yunpeng Zhai; Xuming Hu; Philip S. Yu; Aiwei Liu
>
> **摘要:** As text generation has become a core capability of modern Large Language Models (LLMs), it underpins a wide range of downstream applications. However, most existing LLMs rely on autoregressive (AR) generation, producing one token at a time based on previously generated context-resulting in limited generation speed due to the inherently sequential nature of the process. To address this challenge, an increasing number of researchers have begun exploring parallel text generation-a broad class of techniques aimed at breaking the token-by-token generation bottleneck and improving inference efficiency. Despite growing interest, there remains a lack of comprehensive analysis on what specific techniques constitute parallel text generation and how they improve inference performance. To bridge this gap, we present a systematic survey of parallel text generation methods. We categorize existing approaches into AR-based and Non-AR-based paradigms, and provide a detailed examination of the core techniques within each category. Following this taxonomy, we assess their theoretical trade-offs in terms of speed, quality, and efficiency, and examine their potential for combination and comparison with alternative acceleration strategies. Finally, based on our findings, we highlight recent advancements, identify open challenges, and outline promising directions for future research in parallel text generation. We have also created a GitHub repository for indexing relevant papers and open resources available at https://github.com/zhanglingzhe0820/Awesome-Parallel-Text-Generation.
>
---
#### [replaced 017] MDEval: Evaluating and Enhancing Markdown Awareness in Large Language Models
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2501.15000v2](http://arxiv.org/pdf/2501.15000v2)**

> **作者:** Zhongpu Chen; Yinfeng Liu; Long Shi; Xingyan Chen; Yu Zhao; Fuji Ren
>
> **备注:** WWW 2025
>
> **摘要:** Large language models (LLMs) are expected to offer structured Markdown responses for the sake of readability in web chatbots (e.g., ChatGPT). Although there are a myriad of metrics to evaluate LLMs, they fail to evaluate the readability from the view of output content structure. To this end, we focus on an overlooked yet important metric -- Markdown Awareness, which directly impacts the readability and structure of the content generated by these language models. In this paper, we introduce MDEval, a comprehensive benchmark to assess Markdown Awareness for LLMs, by constructing a dataset with 20K instances covering 10 subjects in English and Chinese. Unlike traditional model-based evaluations, MDEval provides excellent interpretability by combining model-based generation tasks and statistical methods. Our results demonstrate that MDEval achieves a Spearman correlation of 0.791 and an accuracy of 84.1% with human, outperforming existing methods by a large margin. Extensive experimental results also show that through fine-tuning over our proposed dataset, less performant open-source models are able to achieve comparable performance to GPT-4o in terms of Markdown Awareness. To ensure reproducibility and transparency, MDEval is open sourced at https://github.com/SWUFE-DB-Group/MDEval-Benchmark.
>
---
#### [replaced 018] Doc2Chart: Intent-Driven Zero-Shot Chart Generation from Documents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.14819v2](http://arxiv.org/pdf/2507.14819v2)**

> **作者:** Akriti Jain; Pritika Ramu; Aparna Garimella; Apoorv Saxena
>
> **备注:** Accepted to EMNLP 2025 Main Conference
>
> **摘要:** Large Language Models (LLMs) have demonstrated strong capabilities in transforming text descriptions or tables to data visualizations via instruction-tuning methods. However, it is not straightforward to apply these methods directly for a more real-world use case of visualizing data from long documents based on user-given intents, as opposed to the user pre-selecting the relevant content manually. We introduce the task of intent-based chart generation from documents: given a user-specified intent and document(s), the goal is to generate a chart adhering to the intent and grounded on the document(s) in a zero-shot setting. We propose an unsupervised, two-staged framework in which an LLM first extracts relevant information from the document(s) by decomposing the intent and iteratively validates and refines this data. Next, a heuristic-guided module selects an appropriate chart type before final code generation. To assess the data accuracy of the generated charts, we propose an attribution-based metric that uses a structured textual representation of charts, instead of relying on visual decoding metrics that often fail to capture the chart data effectively. To validate our approach, we curate a dataset comprising of 1,242 $<$intent, document, charts$>$ tuples from two domains, finance and scientific, in contrast to the existing datasets that are largely limited to parallel text descriptions/ tables and their corresponding charts. We compare our approach with baselines using single-shot chart generation using LLMs and query-based retrieval methods; our method outperforms by upto $9$ points and $17$ points in terms of chart data accuracy and chart type respectively over the best baselines.
>
---
#### [replaced 019] Hydra: Structured Cross-Source Enhanced Large Language Model Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17464v2](http://arxiv.org/pdf/2505.17464v2)**

> **作者:** Xingyu Tan; Xiaoyang Wang; Qing Liu; Xiwei Xu; Xin Yuan; Liming Zhu; Wenjie Zhang
>
> **备注:** Accepted by EMNLP2025 (Main Conference)
>
> **摘要:** Retrieval-augmented generation (RAG) enhances large language models (LLMs) by incorporating external knowledge. Current hybrid RAG system retrieves evidence from both knowledge graphs (KGs) and text documents to support LLM reasoning. However, it faces challenges like handling multi-hop reasoning, multi-entity questions, multi-source verification, and effective graph utilization. To address these limitations, we present Hydra, a training-free framework that unifies graph topology, document semantics, and source reliability to support deep, faithful reasoning in LLMs. Hydra handles multi-hop and multi-entity problems through agent-driven exploration that combines structured and unstructured retrieval, increasing both diversity and precision of evidence. To tackle multi-source verification, Hydra uses a tri-factor cross-source verification (source trustworthiness assessment, cross-source corroboration, and entity-path alignment), to balance topic relevance with cross-modal agreement. By leveraging graph structure, Hydra fuses heterogeneous sources, guides efficient exploration, and prunes noise early. Comprehensive experiments on seven benchmark datasets show that Hydra achieves overall state-of-the-art results on all benchmarks with GPT-3.5, outperforming the strong hybrid baseline ToG-2 by an average of 20.3% and up to 30.1%. Furthermore, Hydra enables smaller models (e.g., Llama-3.1-8B) to achieve reasoning performance comparable to that of GPT-4-Turbo. The source code is available on https://stevetantan.github.io/Hydra/.
>
---
#### [replaced 020] GTPO: Trajectory-Based Policy Optimization in Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.03772v3](http://arxiv.org/pdf/2508.03772v3)**

> **作者:** Marco Simoni; Aleksandar Fontana; Giulio Rossolini; Andrea Saracino
>
> **摘要:** Policy-based optimizations are widely adopted today for the training and alignment of language models, where one of the most recent and effective approaches is Group-relative Policy Optimization (GRPO). In this paper, we reveals and analyze two major limitations of GRPO: (i) tokens frequently appear in completions with both positive and negative rewards, leading to conflicting gradient updates that can reduce their output probability, even though can be essential for maintaining proper structure; (ii) negatively rewarded completions may penalize confident responses and shift model decisions toward unlikely tokens, progressively flattening the output distribution and degrading learning. To address these issues and provide a more stable and effective policy optimization strategy, we introduce GTPO (Group-relative Trajectory-based Policy Optimization), which identifies conflict tokens, tokens appearing in the same position across completions with opposite rewards, protects them by skipping negative updates, while amplifying positive ones. To further prevent policy collapse, GTPO filters out completions whose entropy exceeds a provable threshold. Unlike GRPO, GTPO does not rely on KL-divergence regularization, eliminating the need for a reference model during training, while still ensuring greater training stability and improved performance, validated through multiple experiments on GSM8K, MATH and AIME 2024 benchmarks.
>
---
#### [replaced 021] Reducing Biases towards Minoritized Populations in Medical Curricular Content via Artificial Intelligence for Fairer Health Outcomes
- **分类: cs.CY; cs.CL**

- **链接: [http://arxiv.org/pdf/2407.12680v2](http://arxiv.org/pdf/2407.12680v2)**

> **作者:** Chiman Salavati; Shannon Song; Willmar Sosa Diaz; Scott A. Hale; Roberto E. Montenegro; Fabricio Murai; Shiri Dori-Hacohen
>
> **备注:** Accepted at the 2024 AAAI/ACM Conference on AI, Ethics and Society (AIES'24)
>
> **摘要:** Biased information (recently termed bisinformation) continues to be taught in medical curricula, often long after having been debunked. In this paper, we introduce BRICC, a firstin-class initiative that seeks to mitigate medical bisinformation using machine learning to systematically identify and flag text with potential biases, for subsequent review in an expert-in-the-loop fashion, thus greatly accelerating an otherwise labor-intensive process. A gold-standard BRICC dataset was developed throughout several years, and contains over 12K pages of instructional materials. Medical experts meticulously annotated these documents for bias according to comprehensive coding guidelines, emphasizing gender, sex, age, geography, ethnicity, and race. Using this labeled dataset, we trained, validated, and tested medical bias classifiers. We test three classifier approaches: a binary type-specific classifier, a general bias classifier; an ensemble combining bias type-specific classifiers independently-trained; and a multitask learning (MTL) model tasked with predicting both general and type-specific biases. While MTL led to some improvement on race bias detection in terms of F1-score, it did not outperform binary classifiers trained specifically on each task. On general bias detection, the binary classifier achieves up to 0.923 of AUC, a 27.8% improvement over the baseline. This work lays the foundations for debiasing medical curricula by exploring a novel dataset and evaluating different training model strategies. Hence, it offers new pathways for more nuanced and effective mitigation of bisinformation.
>
---
#### [replaced 022] Principled Detection of Hallucinations in Large Language Models via Multiple Testing
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.18473v2](http://arxiv.org/pdf/2508.18473v2)**

> **作者:** Jiawei Li; Akshayaa Magesh; Venugopal V. Veeravalli
>
> **备注:** 16 pages
>
> **摘要:** While Large Language Models (LLMs) have emerged as powerful foundational models to solve a variety of tasks, they have also been shown to be prone to hallucinations, i.e., generating responses that sound confident but are actually incorrect or even nonsensical. In this work, we formulate the problem of detecting hallucinations as a hypothesis testing problem and draw parallels to the problem of out-of-distribution detection in machine learning models. We propose a multiple-testing-inspired method to solve the hallucination detection problem, and provide extensive experimental results to validate the robustness of our approach against state-of-the-art methods.
>
---
#### [replaced 023] Chain-of-Reasoning: Towards Unified Mathematical Reasoning in Large Language Models via a Multi-Paradigm Perspective
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.11110v3](http://arxiv.org/pdf/2501.11110v3)**

> **作者:** Yiyao Yu; Yuxiang Zhang; Dongdong Zhang; Xiao Liang; Hengyuan Zhang; Xingxing Zhang; Mahmoud Khademi; Hany Awadalla; Junjie Wang; Yujiu Yang; Furu Wei
>
> **备注:** Accepted to ACL 2025 (Main)
>
> **摘要:** Large Language Models (LLMs) have made notable progress in mathematical reasoning, yet often rely on single-paradigm reasoning, limiting their effectiveness across diverse tasks. We introduce Chain-of-Reasoning (CoR), a novel unified framework integrating multiple reasoning paradigms--Natural Language Reasoning (NLR), Algorithmic Reasoning (AR), and Symbolic Reasoning (SR)--to enable synergistic collaboration. CoR generates multiple potential answers via different reasoning paradigms and synthesizes them into a coherent final solution. We propose a Progressive Paradigm Training (PPT) strategy for models to progressively master these paradigms, leading to CoR-Math-7B. Experimental results demonstrate that CoR-Math-7B significantly outperforms current SOTA models, achieving up to a 41.0% absolute improvement over GPT-4o in theorem proving and a 15.0% improvement over RL-based methods on the MATH benchmark in arithmetic tasks. These results show the enhanced mathematical comprehension ability of our model, enabling zero-shot generalization across tasks.
>
---
#### [replaced 024] PediatricsMQA: a Multi-modal Pediatrics Question Answering Benchmark
- **分类: cs.CY; cs.AI; cs.CL; cs.GR; cs.MM**

- **链接: [http://arxiv.org/pdf/2508.16439v3](http://arxiv.org/pdf/2508.16439v3)**

> **作者:** Adil Bahaj; Oumaima Fadi; Mohamed Chetouani; Mounir Ghogho
>
> **摘要:** Large language models (LLMs) and vision-augmented LLMs (VLMs) have significantly advanced medical informatics, diagnostics, and decision support. However, these models exhibit systematic biases, particularly age bias, compromising their reliability and equity. This is evident in their poorer performance on pediatric-focused text and visual question-answering tasks. This bias reflects a broader imbalance in medical research, where pediatric studies receive less funding and representation despite the significant disease burden in children. To address these issues, a new comprehensive multi-modal pediatric question-answering benchmark, PediatricsMQA, has been introduced. It consists of 3,417 text-based multiple-choice questions (MCQs) covering 131 pediatric topics across seven developmental stages (prenatal to adolescent) and 2,067 vision-based MCQs using 634 pediatric images from 67 imaging modalities and 256 anatomical regions. The dataset was developed using a hybrid manual-automatic pipeline, incorporating peer-reviewed pediatric literature, validated question banks, existing benchmarks, and existing QA resources. Evaluating state-of-the-art open models, we find dramatic performance drops in younger cohorts, highlighting the need for age-aware methods to ensure equitable AI support in pediatric care.
>
---
#### [replaced 025] Constructing a Norm for Children's Scientific Drawing: Distribution Features Based on Semantic Similarity of Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.15348v2](http://arxiv.org/pdf/2502.15348v2)**

> **作者:** Yi Zhang; Fan Wei; Jingyi Li; Yan Wang; Yanyan Yu; Jianli Chen; Zipo Cai; Xinyu Liu; Wei Wang; Sensen Yao; Peng Wang; Zhong Wang
>
> **摘要:** The use of children's drawings to examining their conceptual understanding has been proven to be an effective method, but there are two major problems with previous research: 1. The content of the drawings heavily relies on the task, and the ecological validity of the conclusions is low; 2. The interpretation of drawings relies too much on the subjective feelings of the researchers. To address this issue, this study uses the Large Language Model (LLM) to identify 1420 children's scientific drawings (covering 9 scientific themes/concepts), and uses the word2vec algorithm to calculate their semantic similarity. The study explores whether there are consistent drawing representations for children on the same theme, and attempts to establish a norm for children's scientific drawings, providing a baseline reference for follow-up children's drawing research. The results show that the representation of most drawings has consistency, manifested as most semantic similarity>0.8. At the same time, it was found that the consistency of the representation is independent of the accuracy (of LLM's recognition), indicating the existence of consistency bias. In the subsequent exploration of influencing factors, we used Kendall rank correlation coefficient to investigate the effects of "sample size", "abstract degree", and "focus points" on drawings, and used word frequency statistics to explore whether children represented abstract themes/concepts by reproducing what was taught in class. It was found that accuracy (of LLM's recognition) is the most sensitive indicator, and data such as sample size and semantic similarity are related to it; The consistency between classroom experiments and teaching purpose is also an important factor, many students focus more on the experiments themselves rather than what they explain.
>
---
#### [replaced 026] mSTEB: Massively Multilingual Evaluation of LLMs on Speech and Text Tasks
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2506.08400v3](http://arxiv.org/pdf/2506.08400v3)**

> **作者:** Luel Hagos Beyene; Vivek Verma; Min Ma; Jesujoba O. Alabi; Fabian David Schmidt; Joyce Nakatumba-Nabende; David Ifeoluwa Adelani
>
> **备注:** Accepted to ASRU 2025
>
> **摘要:** Large Language models (LLMs) have demonstrated impressive performance on a wide range of tasks, including in multimodal settings such as speech. However, their evaluation is often limited to English and a few high-resource languages. For low-resource languages, there is no standardized evaluation benchmark. In this paper, we address this gap by introducing mSTEB, a new benchmark to evaluate the performance of LLMs on a wide range of tasks covering language identification, text classification, question answering, and translation tasks on both speech and text modalities. We evaluated the performance of leading LLMs such as Gemini 2.0 Flash and GPT-4o (Audio) and state-of-the-art open models such as Qwen 2 Audio and Gemma 3 27B. Our evaluation shows a wide gap in performance between high-resource and low-resource languages, especially for languages spoken in Africa and Americas/Oceania. Our findings show that more investment is needed to address their under-representation in LLMs coverage.
>
---
#### [replaced 027] Robust Detection of Watermarks for Large Language Models Under Human Edits
- **分类: stat.ME; cs.CL; cs.LG; math.ST; stat.ML; stat.TH**

- **链接: [http://arxiv.org/pdf/2411.13868v3](http://arxiv.org/pdf/2411.13868v3)**

> **作者:** Xiang Li; Feng Ruan; Huiyuan Wang; Qi Long; Weijie J. Su
>
> **备注:** To appear in Journal of the Royal Statistical Society: Series B
>
> **摘要:** Watermarking has offered an effective approach to distinguishing text generated by large language models (LLMs) from human-written text. However, the pervasive presence of human edits on LLM-generated text dilutes watermark signals, thereby significantly degrading detection performance of existing methods. In this paper, by modeling human edits through mixture model detection, we introduce a new method in the form of a truncated goodness-of-fit test for detecting watermarked text under human edits, which we refer to as Tr-GoF. We prove that the Tr-GoF test achieves optimality in robust detection of the Gumbel-max watermark in a certain asymptotic regime of substantial text modifications and vanishing watermark signals. Importantly, Tr-GoF achieves this optimality \textit{adaptively} as it does not require precise knowledge of human edit levels or probabilistic specifications of the LLMs, in contrast to the optimal but impractical (Neyman--Pearson) likelihood ratio test. Moreover, we establish that the Tr-GoF test attains the highest detection efficiency rate in a certain regime of moderate text modifications. In stark contrast, we show that sum-based detection rules, as employed by existing methods, fail to achieve optimal robustness in both regimes because the additive nature of their statistics is less resilient to edit-induced noise. Finally, we demonstrate the competitive and sometimes superior empirical performance of the Tr-GoF test on both synthetic data and open-source LLMs in the OPT and LLaMA families.
>
---
#### [replaced 028] X-Troll: eXplainable Detection of State-Sponsored Information Operations Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.16021v2](http://arxiv.org/pdf/2508.16021v2)**

> **作者:** Lin Tian; Xiuzhen Zhang; Maria Myung-Hee Kim; Jennifer Biggs; Marian-Andrei Rizoiu
>
> **备注:** 15 pages, 5 figures, 4 tables, accepted by CIKM2025
>
> **摘要:** State-sponsored trolls, malicious actors who deploy sophisticated linguistic manipulation in coordinated information campaigns, posing threats to online discourse integrity. While Large Language Models (LLMs) achieve strong performance on general natural language processing (NLP) tasks, they struggle with subtle propaganda detection and operate as ``black boxes'', providing no interpretable insights into manipulation strategies. This paper introduces X-Troll, a novel framework that bridges this gap by integrating explainable adapter-based LLMs with expert-derived linguistic knowledge to detect state-sponsored trolls and provide human-readable explanations for its decisions. X-Troll incorporates appraisal theory and propaganda analysis through specialized LoRA adapters, using dynamic gating to capture campaign-specific discourse patterns in coordinated information operations. Experiments on real-world data demonstrate that our linguistically-informed approach shows strong performance compared with both general LLM baselines and existing troll detection models in accuracy while providing enhanced transparency through expert-grounded explanations that reveal the specific linguistic strategies used by state-sponsored actors. X-Troll source code is available at: https://github.com/ltian678/xtroll_source/.
>
---
#### [replaced 029] PyVision: Agentic Vision with Dynamic Tooling
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07998v3](http://arxiv.org/pdf/2507.07998v3)**

> **作者:** Shitian Zhao; Haoquan Zhang; Shaoheng Lin; Ming Li; Qilong Wu; Kaipeng Zhang; Chen Wei
>
> **备注:** 26 Pages, 10 Figures, Technical report, Fix Typo
>
> **摘要:** LLMs are increasingly deployed as agents, systems capable of planning, reasoning, and dynamically calling external tools. However, in visual reasoning, prior approaches largely remain limited by predefined workflows and static toolsets. In this report, we present PyVision, an interactive, multi-turn framework that enables MLLMs to autonomously generate, execute, and refine Python-based tools tailored to the task at hand, unlocking flexible and interpretable problem-solving. We develop a taxonomy of the tools created by PyVision and analyze their usage across a diverse set of benchmarks. Quantitatively, PyVision achieves consistent performance gains, boosting GPT-4.1 by +7.8% on V* and Claude-4.0-Sonnet by +31.1% on VLMsAreBlind-mini. These results point to a broader shift: dynamic tooling allows models not just to use tools, but to invent them, advancing toward more agentic visual reasoning.
>
---
#### [replaced 030] Less Redundancy: Boosting Practicality of Vision Language Model in Walking Assistants
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.16070v2](http://arxiv.org/pdf/2508.16070v2)**

> **作者:** Chongyang Li; Zhiqiang Yuan; Jiapei Zhang; Ying Deng; Hanbo Bi; Zexi Jia; Xiaoyue Duan; Peixiang Luo; Jinchao Zhang
>
> **摘要:** Approximately 283 million people worldwide live with visual impairments, motivating increasing research into leveraging Visual Language Models (VLMs) to develop effective walking assistance systems for blind and low vision individuals. However, existing VLMs in walking assistant task often have outputs that contain considerable redundancy and extraneous details, adversely affecting users' ability to accurately assess their surroundings. Moreover, these models typically lack the capability to proactively assess environmental risks and adaptively trigger reminders based on the appropriate scene, leading to excessive temporal redundancy. To mitigate output and temporal redundancy, we propose WalkVLM-LR, a walking assistance model with less redundancy. To reduce output redundancy, we introduce four human-preference-based custom reward functions within the GRPO-based reasoning framework to optimize the output in terms of conciseness, fluency, keyword density, and accuracy, thereby producing more informative and streamlined outputs. To minimize temporal redundancy, we incorporate an environment awareness discriminator, which shares the visual encoder with the VLMs to reduce redundant computations and enhance discriminative efficiency, to make WalkVLM-LR assess scene risk levels and minimize unnecessary reminders. Experimental results demonstrate that our method achieves state-of-the-art performance across all evaluation metrics compared with other models, particularly in output conciseness and less temporal redundancy.
>
---
#### [replaced 031] Putnam-AXIOM: A Functional and Static Benchmark for Measuring Higher Level Mathematical Reasoning in LLMs
- **分类: cs.CL; cs.AI; cs.LG; cs.LO; cs.NE; 68T20, 68T05, 68Q32; F.2.2; I.2.3; I.2.6; I.2.8**

- **链接: [http://arxiv.org/pdf/2508.08292v2](http://arxiv.org/pdf/2508.08292v2)**

> **作者:** Aryan Gulati; Brando Miranda; Eric Chen; Emily Xia; Kai Fronsdal; Bruno Dumont; Elyas Obbad; Sanmi Koyejo
>
> **备注:** 27 pages total (10-page main paper + 17-page appendix), 12 figures, 6 tables. Submitted to ICML 2025 (under review)
>
> **摘要:** Current mathematical reasoning benchmarks for large language models (LLMs) are approaching saturation, with some achieving > 90% accuracy, and are increasingly compromised by training-set contamination. We introduce Putnam-AXIOM, a benchmark of 522 university-level competition problems drawn from the prestigious William Lowell Putnam Mathematical Competition, and Putnam-AXIOM Variation, an unseen companion set of 100 functional variants generated by programmatically perturbing variables and constants. The variation protocol produces an unlimited stream of equally difficult, unseen instances -- yielding a contamination-resilient test bed. On the Original set, OpenAI's o1-preview -- the strongest evaluated model -- scores 41.9%, but its accuracy drops by 19.6% (46.8% relative decrease) on the paired Variations. The remaining eighteen models show the same downward trend, ten of them with non-overlapping 95% confidence intervals. These gaps suggest memorization and highlight the necessity of dynamic benchmarks. We complement "boxed" accuracy with Teacher-Forced Accuracy (TFA), a lightweight metric that directly scores reasoning traces and automates natural language proof evaluations. Putnam-AXIOM therefore provides a rigorous, contamination-resilient evaluation framework for assessing advanced mathematical reasoning of LLMs. Data and evaluation code are publicly available at https://github.com/brando90/putnam-axiom.
>
---
#### [replaced 032] Safeguard Fine-Tuned LLMs Through Pre- and Post-Tuning Model Merging
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.19512v2](http://arxiv.org/pdf/2412.19512v2)**

> **作者:** Hua Farn; Hsuan Su; Shachi H Kumar; Saurav Sahay; Shang-Tse Chen; Hung-yi Lee
>
> **备注:** EMNLP 2025 Findings
>
> **摘要:** Fine-tuning large language models (LLMs) for downstream tasks often leads to catastrophic forgetting, notably degrading the safety of originally aligned models. While some existing methods attempt to restore safety by incorporating additional safety data, the quality of such data typically falls short of that used in the original alignment process. Moreover, these high-quality safety datasets are generally inaccessible, making it difficult to fully recover the model's original safety. We ask: How can we preserve safety while improving downstream task performance without additional safety data? We show that simply merging the weights of pre- and post-fine-tuned models effectively mitigates safety degradation while enhancing performance. Experiments across different downstream tasks and models validate the method's practicality and effectiveness.
>
---
#### [replaced 033] Refining Czech GEC: Insights from a Multi-Experiment Approach
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.22402v2](http://arxiv.org/pdf/2506.22402v2)**

> **作者:** Petr Pechman; Milan Straka; Jana Straková; Jakub Náplava
>
> **备注:** Accepted to TSD 2025
>
> **摘要:** We present a grammar error correction (GEC) system that achieves state of the art for the Czech language. Our system is based on a neural network translation approach with the Transformer architecture, and its key feature is its real-time synthetic generation pipeline, which dynamically augments sentences with artificial errors by introducing both language-agnostic and Czech-specific errors. We conduct a comprehensive series of experiments, investigating the Czech GEC corpora as bases for synthetic error introduction, several error generation strategies, domain balancing, tokenization granularity, model size, and data scaling during fine-tuning. Additionally, we evaluate the performance of large language models (LLMs) on Czech GEC in both end-user and expert fine-tuning scenarios. Our best-performing model is superior both in performance and computational efficiency. The source code and the trained model links are available on https://github.com/ufal/tsd2025-gec.
>
---
#### [replaced 034] Cross-lingual Offensive Language Detection: A Systematic Review of Datasets, Transfer Approaches and Challenges
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2401.09244v2](http://arxiv.org/pdf/2401.09244v2)**

> **作者:** Aiqi Jiang; Arkaitz Zubiaga
>
> **备注:** 35 pages, 7 figures
>
> **摘要:** The growing prevalence and rapid evolution of offensive language in social media amplify the complexities of detection, particularly highlighting the challenges in identifying such content across diverse languages. This survey presents a systematic and comprehensive exploration of Cross-Lingual Transfer Learning (CLTL) techniques in offensive language detection in social media. Our study stands as the first holistic overview to focus exclusively on the cross-lingual scenario in this domain. We analyse 67 relevant papers and categorise these studies across various dimensions, including the characteristics of multilingual datasets used, the cross-lingual resources employed, and the specific CLTL strategies implemented. According to "what to transfer", we also summarise three main CLTL transfer approaches: instance, feature, and parameter transfer. Additionally, we shed light on the current challenges and future research opportunities in this field. Furthermore, we have made our survey resources available online, including two comprehensive tables that provide accessible references to the multilingual datasets and CLTL methods used in the reviewed literature.
>
---
#### [replaced 035] LLM-based feature generation from text for interpretable machine learning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.07132v2](http://arxiv.org/pdf/2409.07132v2)**

> **作者:** Vojtěch Balek; Lukáš Sýkora; Vilém Sklenák; Tomáš Kliegr
>
> **摘要:** Existing text representations such as embeddings and bag-of-words are not suitable for rule learning due to their high dimensionality and absent or questionable feature-level interpretability. This article explores whether large language models (LLMs) could address this by extracting a small number of interpretable features from text. We demonstrate this process on two datasets (CORD-19 and M17+) containing several thousand scientific articles from multiple disciplines and a target being a proxy for research impact. An evaluation based on testing for the statistically significant correlation with research impact has shown that LLama 2-generated features are semantically meaningful. We consequently used these generated features in text classification to predict the binary target variable representing the citation rate for the CORD-19 dataset and the ordinal 5-class target representing an expert-awarded grade in the M17+ dataset. Machine-learning models trained on the LLM-generated features provided similar predictive performance to the state-of-the-art embedding model SciBERT for scientific text. The LLM used only 62 features compared to 768 features in SciBERT embeddings, and these features were directly interpretable, corresponding to notions such as article methodological rigor, novelty, or grammatical correctness. As the final step, we extract a small number of well-interpretable action rules. Consistently competitive results obtained with the same LLM feature set across both thematically diverse datasets show that this approach generalizes across domains.
>
---
#### [replaced 036] LinguaSafe: A Comprehensive Multilingual Safety Benchmark for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.12733v2](http://arxiv.org/pdf/2508.12733v2)**

> **作者:** Zhiyuan Ning; Tianle Gu; Jiaxin Song; Shixin Hong; Lingyu Li; Huacan Liu; Jie Li; Yixu Wang; Meng Lingyu; Yan Teng; Yingchun Wang
>
> **备注:** 7pages, 5 figures
>
> **摘要:** The widespread adoption and increasing prominence of large language models (LLMs) in global technologies necessitate a rigorous focus on ensuring their safety across a diverse range of linguistic and cultural contexts. The lack of a comprehensive evaluation and diverse data in existing multilingual safety evaluations for LLMs limits their effectiveness, hindering the development of robust multilingual safety alignment. To address this critical gap, we introduce LinguaSafe, a comprehensive multilingual safety benchmark crafted with meticulous attention to linguistic authenticity. The LinguaSafe dataset comprises 45k entries in 12 languages, ranging from Hungarian to Malay. Curated using a combination of translated, transcreated, and natively-sourced data, our dataset addresses the critical need for multilingual safety evaluations of LLMs, filling the void in the safety evaluation of LLMs across diverse under-represented languages from Hungarian to Malay. LinguaSafe presents a multidimensional and fine-grained evaluation framework, with direct and indirect safety assessments, including further evaluations for oversensitivity. The results of safety and helpfulness evaluations vary significantly across different domains and different languages, even in languages with similar resource levels. Our benchmark provides a comprehensive suite of metrics for in-depth safety evaluation, underscoring the critical importance of thoroughly assessing multilingual safety in LLMs to achieve more balanced safety alignment. Our dataset and code are released to the public to facilitate further research in the field of multilingual LLM safety.
>
---
#### [replaced 037] Unifying the Extremes: Developing a Unified Model for Detecting and Predicting Extremist Traits and Radicalization
- **分类: cs.SI; cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2501.04820v2](http://arxiv.org/pdf/2501.04820v2)**

> **作者:** Allison Lahnala; Vasudha Varadarajan; Lucie Flek; H. Andrew Schwartz; Ryan L. Boyd
>
> **备注:** 17 pages, 7 figures, 4 tables
>
> **摘要:** The proliferation of ideological movements into extremist factions via social media has become a global concern. While radicalization has been studied extensively within the context of specific ideologies, our ability to accurately characterize extremism in more generalizable terms remains underdeveloped. In this paper, we propose a novel method for extracting and analyzing extremist discourse across a range of online community forums. By focusing on verbal behavioral signatures of extremist traits, we develop a framework for quantifying extremism at both user and community levels. Our research identifies 11 distinct factors, which we term ``The Extremist Eleven,'' as a generalized psychosocial model of extremism. Applying our method to various online communities, we demonstrate an ability to characterize ideologically diverse communities across the 11 extremist traits. We demonstrate the power of this method by analyzing user histories from members of the incel community. We find that our framework accurately predicts which users join the incel community up to 10 months before their actual entry with an AUC of $>0.6$, steadily increasing to AUC ~0.9 three to four months before the event. Further, we find that upon entry into an extremist forum, the users tend to maintain their level of extremism within the community, while still remaining distinguishable from the general online discourse. Our findings contribute to the study of extremism by introducing a more holistic, cross-ideological approach that transcends traditional, trait-specific models.
>
---
#### [replaced 038] A Statistical Framework of Watermarks for Large Language Models: Pivot, Detection Efficiency and Optimal Rules
- **分类: math.ST; cs.CL; cs.CR; cs.LG; stat.ML; stat.TH; 62C05 (Primary), 62F03 (Secondary)**

- **链接: [http://arxiv.org/pdf/2404.01245v4](http://arxiv.org/pdf/2404.01245v4)**

> **作者:** Xiang Li; Feng Ruan; Huiyuan Wang; Qi Long; Weijie J. Su
>
> **备注:** Accepted by Annals of Statistics
>
> **摘要:** Since ChatGPT was introduced in November 2022, embedding (nearly) unnoticeable statistical signals into text generated by large language models (LLMs), also known as watermarking, has been used as a principled approach to provable detection of LLM-generated text from its human-written counterpart. In this paper, we introduce a general and flexible framework for reasoning about the statistical efficiency of watermarks and designing powerful detection rules. Inspired by the hypothesis testing formulation of watermark detection, our framework starts by selecting a pivotal statistic of the text and a secret key -- provided by the LLM to the verifier -- to enable controlling the false positive rate (the error of mistakenly detecting human-written text as LLM-generated). Next, this framework allows one to evaluate the power of watermark detection rules by obtaining a closed-form expression of the asymptotic false negative rate (the error of incorrectly classifying LLM-generated text as human-written). Our framework further reduces the problem of determining the optimal detection rule to solving a minimax optimization program. We apply this framework to two representative watermarks -- one of which has been internally implemented at OpenAI -- and obtain several findings that can be instrumental in guiding the practice of implementing watermarks. In particular, we derive optimal detection rules for these watermarks under our framework. These theoretically derived detection rules are demonstrated to be competitive and sometimes enjoy a higher power than existing detection approaches through numerical experiments.
>
---
#### [replaced 039] SinLlama -- A Large Language Model for Sinhala
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2508.09115v3](http://arxiv.org/pdf/2508.09115v3)**

> **作者:** H. W. K. Aravinda; Rashad Sirajudeen; Samith Karunathilake; Nisansa de Silva; Surangika Ranathunga; Rishemjit Kaur
>
> **摘要:** Low-resource languages such as Sinhala are often overlooked by open-source Large Language Models (LLMs). In this research, we extend an existing multilingual LLM (Llama-3-8B) to better serve Sinhala. We enhance the LLM tokenizer with Sinhala specific vocabulary and perform continual pre-training on a cleaned 10 million Sinhala corpus, resulting in the SinLlama model. This is the very first decoder-based open-source LLM with explicit Sinhala support. When SinLlama was instruction fine-tuned for three text classification tasks, it outperformed base and instruct variants of Llama-3-8B by a significant margin.
>
---
#### [replaced 040] Efficient Response Generation Strategy Selection for Fine-Tuning Large Language Models Through Self-Aligned Perplexity
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11779v3](http://arxiv.org/pdf/2502.11779v3)**

> **作者:** Xuan Ren; Qi Chen; Lingqiao Liu
>
> **摘要:** Fine-tuning large language models (LLMs) typically relies on producing large sets of input-output pairs. Yet for a given question, there can be many valid outputs. In practice, these outputs are often derived by distilling knowledge from teacher models, and they can vary depending on the specific teacher model or prompting strategy employed. Recent findings show that how these training outputs are generated can significantly affect the performance of the fine-tuned model, raising an important question: how do we pick the best data generation method from among numerous possibilities? Rather than exhaustively training and evaluating on each candidate, this paper proposes a scalable approximate method that assesses a small subset of generated data to estimate its suitability for a specific target LLM. Our central idea is that effective outputs should be familiar to the target LLM. While previous work measures familiarity with perplexity, we find that perplexity might be suboptimal in characterizing familiarity through empirical analyses and practical observations. To address this, we introduce self-aligned perplexity, a novel metric capturing how closely candidate outputs adhere to the target LLM's own style and reasoning patterns. In this way, we can identify the most effective generation strategy on a small sample, then apply it to produce the complete training set. We demonstrate that training on data generated by the chosen method yields significant improvements across diverse reasoning-focused benchmarks, particularly in cases where different candidate methods lead to highly divergent training outcomes. Our implementation is publicly available at https://github.com/XuanRen4470/SPPL.
>
---
#### [replaced 041] A Survey on Training-free Alignment of Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.09016v2](http://arxiv.org/pdf/2508.09016v2)**

> **作者:** Birong Pan; Yongqi Li; Weiyu Zhang; Wenpeng Lu; Mayi Xu; Shen Zhou; Yuanyuan Zhu; Ming Zhong; Tieyun Qian
>
> **备注:** Accepted to EMNLP 2025 (findings), camera-ready version
>
> **摘要:** The alignment of large language models (LLMs) aims to ensure their outputs adhere to human values, ethical standards, and legal norms. Traditional alignment methods often rely on resource-intensive fine-tuning (FT), which may suffer from knowledge degradation and face challenges in scenarios where the model accessibility or computational resources are constrained. In contrast, training-free (TF) alignment techniques--leveraging in-context learning, decoding-time adjustments, and post-generation corrections--offer a promising alternative by enabling alignment without heavily retraining LLMs, making them adaptable to both open-source and closed-source environments. This paper presents the first systematic review of TF alignment methods, categorizing them by stages of pre-decoding, in-decoding, and post-decoding. For each stage, we provide a detailed examination from the viewpoint of LLMs and multimodal LLMs (MLLMs), highlighting their mechanisms and limitations. Furthermore, we identify key challenges and future directions, paving the way for more inclusive and effective TF alignment techniques. By synthesizing and organizing the rapidly growing body of research, this survey offers a guidance for practitioners and advances the development of safer and more reliable LLMs.
>
---
#### [replaced 042] Multi-Type Context-Aware Conversational Recommender Systems via Mixture-of-Experts
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2504.13655v2](http://arxiv.org/pdf/2504.13655v2)**

> **作者:** Jie Zou; Cheng Lin; Weikang Guo; Zheng Wang; Jiwei Wei; Yang Yang; Heng Tao Shen
>
> **备注:** 31 pages; Accepted by Information Fusion
>
> **摘要:** Conversational recommender systems enable natural language conversations and thus lead to a more engaging and effective recommendation scenario. As the conversations for recommender systems usually contain limited contextual information, many existing conversational recommender systems incorporate external sources to enrich the contextual information. However, how to combine different types of contextual information is still a challenge. In this paper, we propose a multi-type context-aware conversational recommender system, called MCCRS, effectively fusing multi-type contextual information via mixture-of-experts to improve conversational recommender systems. MCCRS incorporates both structured information and unstructured information, including the structured knowledge graph, unstructured conversation history, and unstructured item reviews. It consists of several experts, with each expert specialized in a particular domain (i.e., one specific contextual information). Multiple experts are then coordinated by a ChairBot to generate the final results. Our proposed MCCRS model takes advantage of different contextual information and the specialization of different experts followed by a ChairBot breaks the model bottleneck on a single contextual information. Experimental results demonstrate that our proposed MCCRS method achieves significantly higher performance compared to existing baselines.
>
---
#### [replaced 043] FiRST: Finetuning Router-Selective Transformers for Input-Adaptive Latency Reduction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.12513v4](http://arxiv.org/pdf/2410.12513v4)**

> **作者:** Akriti Jain; Saransh Sharma; Koyel Mukherjee; Soumyabrata Pal
>
> **备注:** Accepted to EMNLP 2025 Findings
>
> **摘要:** Auto-regressive Large Language Models (LLMs) demonstrate remarkable performance across different domains such as vision and language processing. However, due to sequential processing through a stack of transformer layers, autoregressive decoding faces significant computation/latency challenges, particularly in resource-constrained environments like mobile and edge devices. Existing approaches in literature that aim to improve latency via skipping layers have two distinct flavors - 1) Early exit, and 2) Input-agnostic heuristics where tokens exit at pre-determined layers irrespective of input sequence. Both the above strategies have limitations - the former cannot be applied to handle KV Caching necessary for speed-ups in modern framework and the latter does not capture the variation in layer importance across tasks or more generally, across input sequences. To address both limitations, we propose FiRST, an algorithm that reduces inference latency by using layer-specific routers to select a subset of transformer layers adaptively for each input sequence - the prompt (during the prefill stage) decides which layers will be skipped during decoding. FiRST preserves compatibility with KV caching enabling faster inference while being quality-aware. FiRST is model-agnostic and can be easily enabled on any pre-trained LLM. Our approach reveals that input adaptivity is critical - indeed, different task-specific middle layers play a crucial role in evolving hidden representations depending on tasks. Extensive experiments show that FiRST significantly reduces latency while outperforming other layer selection strategies in quality metics. It retains competitive performance to base model (without layer skipping) and in some cases, even improves upon it. FiRST is thus a promising and efficient solution for LLM deployment in low-resource environments.
>
---
#### [replaced 044] When Algorithms Meet Artists: Topic Modeling the AI-Art Debate, 2013-2025
- **分类: cs.CL; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2508.03037v2](http://arxiv.org/pdf/2508.03037v2)**

> **作者:** Ariya Mukherjee-Gandhi; Oliver Muellerklein
>
> **备注:** 23 pages, 7 figures, 8 tables
>
> **摘要:** As generative AI continues to reshape artistic production and alternate modes of human expression, artists whose livelihoods are most directly affected have raised urgent concerns about consent, transparency, and the future of creative labor. However, the voices of artists are often marginalized in dominant public and scholarly discourse. This study presents a twelve-year analysis, from 2013 to 2025, of English-language discourse surrounding AI-generated art. It draws from 439 curated 500-word excerpts sampled from opinion articles, news reports, blogs, legal filings, and spoken-word transcripts. Through a reproducible methodology, we identify five stable thematic clusters and uncover a misalignment between artists' perceptions and prevailing media narratives. Our findings highlight how the use of technical jargon can function as a subtle form of gatekeeping, often sidelining the very issues artists deem most urgent. Our work provides a BERTopic-based methodology and a multimodal baseline for future research, alongside a clear call for deeper, transparency-driven engagement with artist perspectives in the evolving AI-creative landscape.
>
---
#### [replaced 045] NPHardEval4V: Dynamic Evaluation of Large Vision-Language Models with Effects of Vision
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2403.01777v3](http://arxiv.org/pdf/2403.01777v3)**

> **作者:** Xiang Li; Wenyue Hua; Kaijie Zhu; Lingyao Li; Haoyang Ling; Jinkui Chi; Qi Dou; Jindong Wang; Yongfeng Zhang; Xin Ma; Lizhou Fan
>
> **备注:** 25 pages, 9 figures, 2 tables
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated impressive capabilities in multimodal understanding, yet their reasoning abilities remain underexplored. Existing benchmarks tend to focus on perception or text-based comprehension, offering limited insight into how well these models perform on structured, logic-driven tasks that require both visual and linguistic reasoning. To address this gap, we introduce NPHardEval4V, a multimodal benchmark suite grounded in four classical NP-hard problems: Knapsack, Set Cover, Traveling Salesperson, and Vertex Cover. Each task is presented through a combination of structured visual layouts and textual prompts, designed to assess the ability of LVLMs to perform combinatorial reasoning under visual-linguistic constraints. We evaluate a set of advanced open-source and closed-source vision-language models under a unified prompting and problem representation framework. This enables fair comparison across models and task types, while isolating key variables affecting performance. Our results show that while these models perform reasonably well on perception-based inputs, they struggle with global optimization, abstraction, and constraint satisfaction. No single model demonstrates consistent reasoning capability across all problem types, and common failure patterns reveal fundamental limitations in current architectures. By leveraging the structure and complexity of NP-hard problems, NPHardEval4V provides a scalable, interpretable, and challenging testbed for diagnosing reasoning behaviors in LVLMs. We hope this benchmark can support the community in building more robust, inference-capable multimodal systems. The benchmark dataset and code are available at https://github.com/lizhouf/NPHardEval4.
>
---
#### [replaced 046] Exploring Typographic Visual Prompts Injection Threats in Cross-Modality Generation Models
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.11519v2](http://arxiv.org/pdf/2503.11519v2)**

> **作者:** Hao Cheng; Erjia Xiao; Yichi Wang; Lingfeng Zhang; Qiang Zhang; Jiahang Cao; Kaidi Xu; Mengshu Sun; Xiaoshuai Hao; Jindong Gu; Renjing Xu
>
> **备注:** This paper is accepted by IJCAI2025 Workshop on Deepfake Detection, Localization, and Interpretability
>
> **摘要:** Current Cross-Modality Generation Models (GMs) demonstrate remarkable capabilities in various generative tasks. Given the ubiquity and information richness of vision modality inputs in real-world scenarios, Cross-Vision tasks, encompassing Vision-Language Perception (VLP) and Image-to-Image (I2I), have attracted significant attention. Large Vision Language Models (LVLMs) and I2I Generation Models (GMs) are employed to handle VLP and I2I tasks, respectively. Previous research indicates that printing typographic words into input images significantly induces LVLMs and I2I GMs to produce disruptive outputs that are semantically aligned with those words. Additionally, visual prompts, as a more sophisticated form of typography, are also revealed to pose security risks to various applications of cross-vision tasks. However, the specific characteristics of the threats posed by visual prompts remain underexplored. In this paper, to comprehensively investigate the performance impact induced by Typographic Visual Prompt Injection (TVPI) in various LVLMs and I2I GMs, we propose the Typographic Visual Prompts Injection Dataset and thoroughly evaluate the TVPI security risks on various open-source and closed-source LVLMs and I2I GMs under visual prompts with different target semantics, deepening the understanding of TVPI threats.
>
---
#### [replaced 047] News is More than a Collection of Facts: Moral Frame Preserving News Summarization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.00657v2](http://arxiv.org/pdf/2504.00657v2)**

> **作者:** Enrico Liscio; Michela Lorandi; Pradeep K. Murukannaiah
>
> **备注:** Accepted at COLM2025
>
> **摘要:** News articles are more than collections of facts; they reflect journalists' framing, shaping how events are presented to the audience. One key aspect of framing is the choice to write in (or quote verbatim) morally charged language as opposed to using neutral terms. This moral framing carries implicit judgments that automated news summarizers should recognize and preserve to maintain the original intent of the writer. In this work, we perform the first study on the preservation of moral framing in AI-generated news summaries. We propose an approach that leverages the intuition that journalists intentionally use or report specific moral-laden words, which should be retained in summaries. Through automated, crowd-sourced, and expert evaluations, we demonstrate that our approach enhances the preservation of moral framing while maintaining overall summary quality.
>
---
#### [replaced 048] Evaluating the Fitness of Ontologies for the Task of Question Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.07994v2](http://arxiv.org/pdf/2504.07994v2)**

> **作者:** Samah Alkhuzaey; Floriana Grasso; Terry R. Payne; Valentina Tamma
>
> **备注:** Revised version (v2) accepted for the 28th European Conference on Artificial Intelligence (ECAI-2025), including a validation study
>
> **摘要:** Ontology-based question generation is an important application of semantic-aware systems that enables the creation of large question banks for diverse learning environments. The effectiveness of these systems, both in terms of the calibre and cognitive difficulty of the resulting questions, depends heavily on the quality and modelling approach of the underlying ontologies, making it crucial to assess their fitness for this task. To date, there has been no comprehensive investigation into the specific ontology aspects or characteristics that affect the question generation process. Therefore, this paper proposes a set of requirements and task-specific metrics for evaluating the fitness of ontologies for question generation tasks in pedagogical settings. Using the ROMEO methodology (a structured framework used for identifying task-specific metrics), a set of evaluation metrics have been derived from an expert assessment of questions generated by a question generation model. To validate the proposed metrics, we apply them to a set of ontologies previously used in question generation to illustrate how the metric scores align with and complement findings reported in earlier studies. The analysis confirms that ontology characteristics significantly impact the effectiveness of question generation, with different ontologies exhibiting varying performance levels. This highlights the importance of assessing ontology quality with respect to Automatic Question Generation (AQG) tasks.
>
---
#### [replaced 049] KoWit-24: A Richly Annotated Dataset of Wordplay in News Headlines
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.01510v2](http://arxiv.org/pdf/2503.01510v2)**

> **作者:** Alexander Baranov; Anna Palatkina; Yulia Makovka; Pavel Braslavski
>
> **备注:** Accepted to RANLP 2025
>
> **摘要:** We present KoWit-24, a dataset with fine-grained annotation of wordplay in 2,700 Russian news headlines. KoWit-24 annotations include the presence of wordplay, its type, wordplay anchors, and words/phrases the wordplay refers to. Unlike the majority of existing humor collections of canned jokes, KoWit-24 provides wordplay contexts -- each headline is accompanied by the news lead and summary. The most common type of wordplay in the dataset is the transformation of collocations, idioms, and named entities -- the mechanism that has been underrepresented in previous humor datasets. Our experiments with five LLMs show that there is ample room for improvement in wordplay detection and interpretation tasks. The dataset and evaluation scripts are available at https://github.com/Humor-Research/KoWit-24
>
---
#### [replaced 050] EnvInjection: Environmental Prompt Injection Attack to Multi-modal Web Agents
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.11717v2](http://arxiv.org/pdf/2505.11717v2)**

> **作者:** Xilong Wang; John Bloch; Zedian Shao; Yuepeng Hu; Shuyan Zhou; Neil Zhenqiang Gong
>
> **备注:** EMNLP 2025 main
>
> **摘要:** Multi-modal large language model (MLLM)-based web agents interact with webpage environments by generating actions based on screenshots of the webpages. Environmental prompt injection attacks manipulate the environment to induce the web agent to perform a specific, attacker-chosen action--denoted as the target action. However, existing attacks suffer from limited effectiveness or stealthiness, or are impractical in real-world settings. In this work, we propose EnvInjection, a new attack that addresses these limitations. Our attack adds a perturbation to the raw pixel values of the rendered webpage. After these perturbed pixels are mapped into a screenshot, the perturbation induces the web agent to perform the target action. We formulate the task of finding the perturbation as an optimization problem. A key challenge in solving this problem is that the mapping between raw pixel values and screenshot is non-differentiable, making it difficult to backpropagate gradients to the perturbation. To overcome this, we train a neural network to approximate the mapping and apply projected gradient descent to solve the reformulated optimization problem. Extensive evaluation on multiple webpage datasets shows that EnvInjection is highly effective and significantly outperforms existing baselines.
>
---
#### [replaced 051] Truth or Twist? Optimal Model Selection for Reliable Label Flipping Evaluation in LLM-based Counterfactuals
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.13972v3](http://arxiv.org/pdf/2505.13972v3)**

> **作者:** Qianli Wang; Van Bach Nguyen; Nils Feldhus; Luis Felipe Villa-Arenas; Christin Seifert; Sebastian Möller; Vera Schmitt
>
> **备注:** Accepted at INLG 2025, camera-ready version
>
> **摘要:** Counterfactual examples are widely employed to enhance the performance and robustness of large language models (LLMs) through counterfactual data augmentation (CDA). However, the selection of the judge model used to evaluate label flipping, the primary metric for assessing the validity of generated counterfactuals for CDA, yields inconsistent results. To decipher this, we define four types of relationships between the counterfactual generator and judge models: being the same model, belonging to the same model family, being independent models, and having an distillation relationship. Through extensive experiments involving two state-of-the-art LLM-based methods, three datasets, four generator models, and 15 judge models, complemented by a user study (n = 90), we demonstrate that judge models with an independent, non-fine-tuned relationship to the generator model provide the most reliable label flipping evaluations. Relationships between the generator and judge models, which are closely aligned with the user study for CDA, result in better model performance and robustness. Nevertheless, we find that the gap between the most effective judge models and the results obtained from the user study remains considerably large. This suggests that a fully automated pipeline for CDA may be inadequate and requires human intervention.
>
---
#### [replaced 052] Scaling Laws for Task-Stratified Knowledge in Post-Training Quantized Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.18609v2](http://arxiv.org/pdf/2508.18609v2)**

> **作者:** Chenxi Zhou; Pengfei Cao; Jiang Li; Jun Zhao; Kang Liu
>
> **摘要:** Large language models (LLMs) present significant deployment challenges due to their scale, with post-training quantization (PTQ) emerging as a practical compression solution. However, a comprehensive understanding of how PTQ precisely impacts diverse LLM knowledge capabilities remains elusive, and existing scaling laws for quantized models often overlook crucial PTQ-specific parameters and task-specific sensitivities. This paper addresses these gaps by conducting an extensive empirical investigation to establish task-stratified scaling laws. We disentangle LLM knowledge into memorization and utilization capabilities and develop a unified quantitative framework that incorporates model size, effective bit-width, calibration set size, and group size. Our central finding reveals that knowledge memorization exhibits markedly greater sensitivity to variations in effective bit-width, calibration set size, and model size compared to the more robust knowledge utilization. These findings offer a fine-grained understanding of PTQ's impact and provide guidance for developing knowledge-aware quantization strategies that can better preserve targeted cognitive functions.
>
---
#### [replaced 053] StepWiser: Stepwise Generative Judges for Wiser Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2508.19229v2](http://arxiv.org/pdf/2508.19229v2)**

> **作者:** Wei Xiong; Wenting Zhao; Weizhe Yuan; Olga Golovneva; Tong Zhang; Jason Weston; Sainbayar Sukhbaatar
>
> **摘要:** As models increasingly leverage multi-step reasoning strategies to solve complex problems, supervising the logical validity of these intermediate steps has become a critical research challenge. Process reward models address this by providing step-by-step feedback, but current approaches have two major drawbacks: they typically function as classifiers without providing explanations, and their reliance on supervised fine-tuning with static datasets limits generalization. Inspired by recent advances, we reframe stepwise reward modeling from a classification task to a reasoning task itself. We thus propose a generative judge that reasons about the policy model's reasoning steps (i.e., meta-reasons), outputting thinking tokens before delivering a final verdict. Our model, StepWiser, is trained by reinforcement learning using relative outcomes of rollouts. We show it provides (i) better judgment accuracy on intermediate steps than existing methods; (ii) can be used to improve the policy model at training time; and (iii) improves inference-time search.
>
---
#### [replaced 054] Know "No" Better: A Data-Driven Approach for Enhancing Negation Awareness in CLIP
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2501.10913v3](http://arxiv.org/pdf/2501.10913v3)**

> **作者:** Junsung Park; Jungbeom Lee; Jongyoon Song; Sangwon Yu; Dahuin Jung; Sungroh Yoon
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** While CLIP has significantly advanced multimodal understanding by bridging vision and language, the inability to grasp negation - such as failing to differentiate concepts like "parking" from "no parking" - poses substantial challenges. By analyzing the data used in the public CLIP model's pre-training, we posit this limitation stems from a lack of negation-inclusive data. To address this, we introduce data generation pipelines that employ a large language model (LLM) and a multimodal LLM to produce negation-inclusive captions. Fine-tuning CLIP with data generated from our pipelines, we develop NegationCLIP, which enhances negation awareness while preserving the generality. Moreover, to enable a comprehensive evaluation of negation understanding, we propose NegRefCOCOg-a benchmark tailored to test VLMs' ability to interpret negation across diverse expressions and positions within a sentence. Experiments on various CLIP architectures validate the effectiveness of our data generation pipelines in enhancing CLIP's ability to perceive negation accurately. Additionally, NegationCLIP's enhanced negation awareness has practical applications across various multimodal tasks, demonstrated by performance gains in text-to-image generation and referring image segmentation.
>
---
#### [replaced 055] PhoniTale: Phonologically Grounded Mnemonic Generation for Typologically Distant Language Pairs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2507.05444v2](http://arxiv.org/pdf/2507.05444v2)**

> **作者:** Sana Kang; Myeongseok Gwon; Su Young Kwon; Jaewook Lee; Andrew Lan; Bhiksha Raj; Rita Singh
>
> **备注:** Accepted to EMNLP 2025 Main
>
> **摘要:** Vocabulary acquisition poses a significant challenge for second-language (L2) learners, especially when learning typologically distant languages such as English and Korean, where phonological and structural mismatches complicate vocabulary learning. Recently, large language models (LLMs) have been used to generate keyword mnemonics by leveraging similar keywords from a learner's first language (L1) to aid in acquiring L2 vocabulary. However, most of this research has focused on native English speakers learning other languages, rather than the reverse. In this paper, we present PhoniTale, a novel cross-lingual mnemonic generation system that retrieves L1 keyword sequence based on phonological similarity and uses LLMs to generate mnemonics. We evaluate PhoniTale using both automated metrics and human evaluations, comparing its output to mnemonics created by humans and by previous automated approaches. To assess practical effectiveness, we also conduct a short-term recall test measuring mnemonic helpfulness. Our findings show that PhoniTale performs comparably to human-authored mnemonics. We also highlight key areas for future improvement in mnemonic quality and methodology.
>
---
