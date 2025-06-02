# 自然语言处理 cs.CL

- **最新发布 144 篇**

- **更新 130 篇**

## 最新发布

#### [new 001] MCTSr-Zero: Self-Reflective Psychological Counseling Dialogues Generation via Principles and Adaptive Exploration
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文提出MCTSr-Zero框架，解决心理辅导等开放对话中LLM难以遵循主观原则（如共情、伦理）的问题。通过"领域对齐"调整MCTS搜索目标，结合再生与元提示机制拓宽策略探索，生成符合心理标准的对话数据。提出PsyEval评估基准，实验显示其显著提升对话质量。**

- **链接: [http://arxiv.org/pdf/2505.23229v1](http://arxiv.org/pdf/2505.23229v1)**

> **作者:** Hao Lu; Yanchi Gu; Haoyuan Huang; Yulin Zhou; Ningxin Zhu; Chen Li
>
> **备注:** 50 pages, 3 figures
>
> **摘要:** The integration of Monte Carlo Tree Search (MCTS) with Large Language Models (LLMs) has demonstrated significant success in structured, problem-oriented tasks. However, applying these methods to open-ended dialogues, such as those in psychological counseling, presents unique challenges. Unlike tasks with objective correctness, success in therapeutic conversations depends on subjective factors like empathetic engagement, ethical adherence, and alignment with human preferences, for which strict "correctness" criteria are ill-defined. Existing result-oriented MCTS approaches can therefore produce misaligned responses. To address this, we introduce MCTSr-Zero, an MCTS framework designed for open-ended, human-centric dialogues. Its core innovation is "domain alignment", which shifts the MCTS search objective from predefined end-states towards conversational trajectories that conform to target domain principles (e.g., empathy in counseling). Furthermore, MCTSr-Zero incorporates "Regeneration" and "Meta-Prompt Adaptation" mechanisms to substantially broaden exploration by allowing the MCTS to consider fundamentally different initial dialogue strategies. We evaluate MCTSr-Zero in psychological counseling by generating multi-turn dialogue data, which is used to fine-tune an LLM, PsyLLM. We also introduce PsyEval, a benchmark for assessing multi-turn psychological counseling dialogues. Experiments demonstrate that PsyLLM achieves state-of-the-art performance on PsyEval and other relevant metrics, validating MCTSr-Zero's effectiveness in generating high-quality, principle-aligned conversational data for human-centric domains and addressing the LLM challenge of consistently adhering to complex psychological standards.
>
---
#### [new 002] StressTest: Can YOUR Speech LM Handle the Stress?
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于语音语言模型(SLM)评估与改进任务，旨在解决模型对句子重音（语义强调）推理与检测能力不足的问题。提出StressTest基准测试模型对重音敏感度，构建合成数据集Stress17k并微调出StresSLM模型，在重音相关任务中显著优于现有模型。**

- **链接: [http://arxiv.org/pdf/2505.22765v1](http://arxiv.org/pdf/2505.22765v1)**

> **作者:** Iddo Yosha; Gallil Maimon; Yossi Adi
>
> **摘要:** Sentence stress refers to emphasis, placed on specific words within a spoken utterance to highlight or contrast an idea, or to introduce new information. It is often used to imply an underlying intention that is not explicitly stated. Recent advances in speech-aware language models (SLMs) have enabled direct processing of audio, allowing models to bypass transcription and access the full richness of the speech signal and perform audio reasoning tasks such as spoken question answering. Despite the crucial role of sentence stress in shaping meaning and speaker intent, it remains largely overlooked in evaluation and development of such models. In this work, we address this gap by introducing StressTest, a benchmark specifically designed to evaluate a model's ability to distinguish between interpretations of spoken sentences based on the stress pattern. We assess the performance of several leading SLMs and find that, despite their overall capabilities, they perform poorly on such tasks. To overcome this limitation, we propose a novel synthetic data generation pipeline, and create Stress17k, a training set that simulates change of meaning implied by stress variation. Then, we empirically show that optimizing models with this synthetic dataset aligns well with real-world recordings and enables effective finetuning of SLMs. Results suggest, that our finetuned model, StresSLM, significantly outperforms existing models on both sentence stress reasoning and detection tasks. Code, models, data, and audio samples - pages.cs.huji.ac.il/adiyoss-lab/stresstest.
>
---
#### [new 003] Detecting Stealthy Backdoor Samples based on Intra-class Distance for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大型语言模型安全检测任务，旨在解决现有方法无法有效检测生成任务中隐秘后门样本的问题。提出RFTC方法，通过参考模型输出差异筛选可疑样本，结合TF-IDF聚类分析类内距离差异，有效识别中毒样本，实验显示优于基线方法。**

- **链接: [http://arxiv.org/pdf/2505.23015v1](http://arxiv.org/pdf/2505.23015v1)**

> **作者:** Jinwen Chen; Hainan Zhang; Fei Sun; Qinnan Zhang; Sijia Wen; Ziwei Wang; Zhiming Zheng
>
> **摘要:** Fine-tuning LLMs with datasets containing stealthy backdoors from publishers poses security risks to downstream applications. Mainstream detection methods either identify poisoned samples by analyzing the prediction probability of poisoned classification models or rely on the rewriting model to eliminate the stealthy triggers. However, the former cannot be applied to generation tasks, while the latter may degrade generation performance and introduce new triggers. Therefore, efficiently eliminating stealthy poisoned samples for LLMs remains an urgent problem. We observe that after applying TF-IDF clustering to the sample response, there are notable differences in the intra-class distances between clean and poisoned samples. Poisoned samples tend to cluster closely because of their specific malicious outputs, whereas clean samples are more scattered due to their more varied responses. Thus, in this paper, we propose a stealthy backdoor sample detection method based on Reference-Filtration and Tfidf-Clustering mechanisms (RFTC). Specifically, we first compare the sample response with the reference model's outputs and consider the sample suspicious if there's a significant discrepancy. And then we perform TF-IDF clustering on these suspicious samples to identify the true poisoned samples based on the intra-class distance. Experiments on two machine translation datasets and one QA dataset demonstrate that RFTC outperforms baselines in backdoor detection and model performance. Further analysis of different reference models also confirms the effectiveness of our Reference-Filtration.
>
---
#### [new 004] SenWiCh: Sense-Annotation of Low-Resource Languages for WiC using Hybrid Methods
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于词义消歧任务，旨在解决低资源语言缺乏高质量评测数据的问题。团队通过混合方法创建了覆盖九种低资源语言的词感标注数据集，并提出半自动标注技术，通过Word-in-Context实验验证了数据集对跨语言迁移和消歧研究的实用性。**

- **链接: [http://arxiv.org/pdf/2505.23714v1](http://arxiv.org/pdf/2505.23714v1)**

> **作者:** Roksana Goworek; Harpal Karlcut; Muhammad Shezad; Nijaguna Darshana; Abhishek Mane; Syam Bondada; Raghav Sikka; Ulvi Mammadov; Rauf Allahverdiyev; Sriram Purighella; Paridhi Gupta; Muhinyia Ndegwa; Haim Dubossarsky
>
> **备注:** 8 pages, 22 figures, submitted to SIGTYP 2025 workshop in ACL
>
> **摘要:** This paper addresses the critical need for high-quality evaluation datasets in low-resource languages to advance cross-lingual transfer. While cross-lingual transfer offers a key strategy for leveraging multilingual pretraining to expand language technologies to understudied and typologically diverse languages, its effectiveness is dependent on quality and suitable benchmarks. We release new sense-annotated datasets of sentences containing polysemous words, spanning nine low-resource languages across diverse language families and scripts. To facilitate dataset creation, the paper presents a demonstrably beneficial semi-automatic annotation method. The utility of the datasets is demonstrated through Word-in-Context (WiC) formatted experiments that evaluate transfer on these low-resource languages. Results highlight the importance of targeted dataset creation and evaluation for effective polysemy disambiguation in low-resource settings and transfer studies. The released datasets and code aim to support further research into fair, robust, and truly multilingual NLP.
>
---
#### [new 005] Self-Critique and Refinement for Faithful Natural Language Explanations
- **分类: cs.CL**

- **简介: 该论文属于提升模型解释可信赖性的任务。针对自然语言解释（NLEs）不忠实于模型实际推理的问题，提出SR-NLE框架，通过迭代的自我批评与两种反馈机制（含特征归因）改进解释，无需额外训练，实验显示其将不忠实率从54.81%降至36.02%。**

- **链接: [http://arxiv.org/pdf/2505.22823v1](http://arxiv.org/pdf/2505.22823v1)**

> **作者:** Yingming Wang; Pepa Atanasova
>
> **备注:** 21 pages, 10 figures, 14 tables
>
> **摘要:** With the rapid development of large language models (LLMs), natural language explanations (NLEs) have become increasingly important for understanding model predictions. However, these explanations often fail to faithfully represent the model's actual reasoning process. While existing work has demonstrated that LLMs can self-critique and refine their initial outputs for various tasks, this capability remains unexplored for improving explanation faithfulness. To address this gap, we introduce Self-critique and Refinement for Natural Language Explanations (SR-NLE), a framework that enables models to improve the faithfulness of their own explanations -- specifically, post-hoc NLEs -- through an iterative critique and refinement process without external supervision. Our framework leverages different feedback mechanisms to guide the refinement process, including natural language self-feedback and, notably, a novel feedback approach based on feature attribution that highlights important input words. Our experiments across three datasets and four state-of-the-art LLMs demonstrate that SR-NLE significantly reduces unfaithfulness rates, with our best method achieving an average unfaithfulness rate of 36.02%, compared to 54.81% for baseline -- an absolute reduction of 18.79%. These findings reveal that the investigated LLMs can indeed refine their explanations to better reflect their actual reasoning process, requiring only appropriate guidance through feedback without additional training or fine-tuning.
>
---
#### [new 006] Cross-Task Experiential Learning on LLM-based Multi-Agent Collaboration
- **分类: cs.CL; cs.AI; cs.MA**

- **简介: 该论文属于LLM驱动的多智能体协作优化任务。针对现有方法孤立处理任务导致冗余计算和泛化不足的问题，提出MAEL框架：构建图结构协作网络，通过存储高价值任务经验并在推理时调用，提升跨任务学习效率与解决方案质量。**

- **链接: [http://arxiv.org/pdf/2505.23187v1](http://arxiv.org/pdf/2505.23187v1)**

> **作者:** Yilong Li; Chen Qian; Yu Xia; Ruijie Shi; Yufan Dang; Zihao Xie; Ziming You; Weize Chen; Cheng Yang; Weichuan Liu; Ye Tian; Xuantang Xiong; Lei Han; Zhiyuan Liu; Maosong Sun
>
> **备注:** Work in Progress
>
> **摘要:** Large Language Model-based multi-agent systems (MAS) have shown remarkable progress in solving complex tasks through collaborative reasoning and inter-agent critique. However, existing approaches typically treat each task in isolation, resulting in redundant computations and limited generalization across structurally similar tasks. To address this, we introduce multi-agent cross-task experiential learning (MAEL), a novel framework that endows LLM-driven agents with explicit cross-task learning and experience accumulation. We model the task-solving workflow on a graph-structured multi-agent collaboration network, where agents propagate information and coordinate via explicit connectivity. During the experiential learning phase, we quantify the quality for each step in the task-solving workflow and store the resulting rewards along with the corresponding inputs and outputs into each agent's individual experience pool. During inference, agents retrieve high-reward, task-relevant experiences as few-shot examples to enhance the effectiveness of each reasoning step, thereby enabling more accurate and efficient multi-agent collaboration. Experimental results on diverse datasets demonstrate that MAEL empowers agents to learn from prior task experiences effectively-achieving faster convergence and producing higher-quality solutions on current tasks.
>
---
#### [new 007] Bounded Rationality for LLMs: Satisficing Alignment at Inference-Time
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型对齐任务，解决多目标优化中忽视人类决策机制的问题。提出SITAlign框架，推理时通过最大化主要目标并满足次要目标的阈值约束实现满意策略对齐，结合理论分析与实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2505.23729v1](http://arxiv.org/pdf/2505.23729v1)**

> **作者:** Mohamad Chehade; Soumya Suvra Ghosal; Souradip Chakraborty; Avinash Reddy; Dinesh Manocha; Hao Zhu; Amrit Singh Bedi
>
> **备注:** Accepted at ICML 2025
>
> **摘要:** Aligning large language models with humans is challenging due to the inherently multifaceted nature of preference feedback. While existing approaches typically frame this as a multi-objective optimization problem, they often overlook how humans actually make decisions. Research on bounded rationality suggests that human decision making follows satisficing strategies-optimizing primary objectives while ensuring others meet acceptable thresholds. To bridge this gap and operationalize the notion of satisficing alignment, we propose SITAlign: an inference time framework that addresses the multifaceted nature of alignment by maximizing a primary objective while satisfying threshold-based constraints on secondary criteria. We provide theoretical insights by deriving sub-optimality bounds of our satisficing based inference alignment approach. We empirically validate SITAlign's performance through extensive experimentation on multiple benchmarks. For instance, on the PKU-SafeRLHF dataset with the primary objective of maximizing helpfulness while ensuring a threshold on harmlessness, SITAlign outperforms the state-of-the-art multi objective decoding strategy by a margin of 22.3% in terms of GPT-4 win-tie rate for helpfulness reward while adhering to the threshold on harmlessness.
>
---
#### [new 008] ML-Agent: Reinforcing LLM Agents for Autonomous Machine Learning Engineering
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自主机器学习任务，旨在解决现有LLM代理依赖人工提示、无法自适应优化的问题。提出ML-Agent框架，包含探索增强微调、分步RL及奖励模块，通过强化学习训练LLM代理。7B模型仅用9任务训练即超越671B DeepSeek-R1，具跨任务泛化能力。**

- **链接: [http://arxiv.org/pdf/2505.23723v1](http://arxiv.org/pdf/2505.23723v1)**

> **作者:** Zexi Liu; Jingyi Chai; Xinyu Zhu; Shuo Tang; Rui Ye; Bo Zhang; Lei Bai; Siheng Chen
>
> **摘要:** The emergence of large language model (LLM)-based agents has significantly advanced the development of autonomous machine learning (ML) engineering. However, most existing approaches rely heavily on manual prompt engineering, failing to adapt and optimize based on diverse experimental experiences. Focusing on this, for the first time, we explore the paradigm of learning-based agentic ML, where an LLM agent learns through interactive experimentation on ML tasks using online reinforcement learning (RL). To realize this, we propose a novel agentic ML training framework with three key components: (1) exploration-enriched fine-tuning, which enables LLM agents to generate diverse actions for enhanced RL exploration; (2) step-wise RL, which enables training on a single action step, accelerating experience collection and improving training efficiency; (3) an agentic ML-specific reward module, which unifies varied ML feedback signals into consistent rewards for RL optimization. Leveraging this framework, we train ML-Agent, driven by a 7B-sized Qwen-2.5 LLM for autonomous ML. Remarkably, despite being trained on merely 9 ML tasks, our 7B-sized ML-Agent outperforms the 671B-sized DeepSeek-R1 agent. Furthermore, it achieves continuous performance improvements and demonstrates exceptional cross-task generalization capabilities.
>
---
#### [new 009] LoLA: Low-Rank Linear Attention With Sparse Caching
- **分类: cs.CL; cs.LG**

- **简介: 论文提出LoLA方法，优化长序列Transformer推理效率与精度。针对线性注意力在长上下文中的记忆冲突问题，其通过分三类存储键值对（局部窗口、稀疏缓存、隐藏状态），减少缓存需求，提升准确率（4K上下文达97.4%），支持8K上下文推理。**

- **链接: [http://arxiv.org/pdf/2505.23666v1](http://arxiv.org/pdf/2505.23666v1)**

> **作者:** Luke McDermott; Robert W. Heath Jr.; Rahul Parhi
>
> **摘要:** Transformer-based large language models suffer from quadratic complexity at inference on long sequences. Linear attention methods are efficient alternatives, however, they fail to provide an accurate approximation of softmax attention. By additionally incorporating sliding window attention into each linear attention head, this gap can be closed for short context-length tasks. Unfortunately, these approaches cannot recall important information from long contexts due to "memory collisions". In this paper , we propose LoLA: Low-rank Linear Attention with sparse caching. LoLA separately stores additional key-value pairs that would otherwise interfere with past associative memories. Moreover, LoLA further closes the gap between linear attention models and transformers by distributing past key-value pairs into three forms of memory: (i) recent pairs in a local sliding window; (ii) difficult-to-memorize pairs in a sparse, global cache; and (iii) generic pairs in the recurrent hidden state of linear attention. As an inference-only strategy, LoLA enables pass-key retrieval on up to 8K context lengths on needle-in-a-haystack tasks from RULER. It boosts the accuracy of the base subquadratic model from 0.6% to 97.4% at 4K context lengths, with a 4.6x smaller cache than that of Llama-3.1 8B. LoLA demonstrates strong performance on zero-shot commonsense reasoning tasks among 1B and 8B parameter subquadratic models. Finally, LoLA is an extremely lightweight approach: Nearly all of our results can be reproduced on a single consumer GPU.
>
---
#### [new 010] ZIPA: A family of efficient models for multilingual phone recognition
- **分类: cs.CL; cs.SD**

- **简介: 该论文提出ZIPA模型，用于多语言音素识别任务，解决高效建模与跨语言泛化问题。构建IPAPack++语料库（17,132小时），开发Zipformer基线的变体（ZIPA-T和ZIPA-CR），参数更少但性能更优；通过噪声学生训练进一步提升。但社会语音多样性建模仍存挑战。**

- **链接: [http://arxiv.org/pdf/2505.23170v1](http://arxiv.org/pdf/2505.23170v1)**

> **作者:** Jian Zhu; Farhan Samir; Eleanor Chodroff; David R. Mortensen
>
> **备注:** ACL 2025 Main
>
> **摘要:** We present ZIPA, a family of efficient speech models that advances the state-of-the-art performance of crosslinguistic phone recognition. We first curated IPAPack++, a large-scale multilingual speech corpus with 17,132 hours of normalized phone transcriptions and a novel evaluation set capturing unseen languages and sociophonetic variation. With the large-scale training data, ZIPA, including transducer (ZIPA-T) and CTC-based (ZIPA-CR) variants, leverage the efficient Zipformer backbones and outperform existing phone recognition systems with much fewer parameters. Further scaling via noisy student training on 11,000 hours of pseudo-labeled multilingual data yields further improvement. While ZIPA achieves strong performance on benchmarks, error analysis reveals persistent limitations in modeling sociophonetic diversity, underscoring challenges for future research.
>
---
#### [new 011] LLM-based HSE Compliance Assessment: Benchmark, Performance, and Advancements
- **分类: cs.CL**

- **简介: 该论文提出HSE-Bench基准，评估LLMs在HSE合规评估中的表现，解决其领域知识不足和结构化法律推理缺陷。工作包括构建包含1000+问题的基准，测试多种模型，发现其依赖语义匹配而非系统推理，进而提出RoE方法引导专家式推理提升决策准确性。**

- **链接: [http://arxiv.org/pdf/2505.22959v1](http://arxiv.org/pdf/2505.22959v1)**

> **作者:** Jianwei Wang; Mengqi Wang; Yinsi Zhou; Zhenchang Xing; Qing Liu; Xiwei Xu; Wenjie Zhang; Liming Zhu
>
> **摘要:** Health, Safety, and Environment (HSE) compliance assessment demands dynamic real-time decision-making under complicated regulations and complex human-machine-environment interactions. While large language models (LLMs) hold significant potential for decision intelligence and contextual dialogue, their capacity for domain-specific knowledge in HSE and structured legal reasoning remains underexplored. We introduce HSE-Bench, the first benchmark dataset designed to evaluate the HSE compliance assessment capabilities of LLM. HSE-Bench comprises over 1,000 manually curated questions drawn from regulations, court cases, safety exams, and fieldwork videos, and integrates a reasoning flow based on Issue spotting, rule Recall, rule Application, and rule Conclusion (IRAC) to assess the holistic reasoning pipeline. We conduct extensive evaluations on different prompting strategies and more than 10 LLMs, including foundation models, reasoning models and multimodal vision models. The results show that, although current LLMs achieve good performance, their capabilities largely rely on semantic matching rather than principled reasoning grounded in the underlying HSE compliance context. Moreover, their native reasoning trace lacks the systematic legal reasoning required for rigorous HSE compliance assessment. To alleviate these, we propose a new prompting technique, Reasoning of Expert (RoE), which guides LLMs to simulate the reasoning process of different experts for compliance assessment and reach a more accurate unified decision. We hope our study highlights reasoning gaps in LLMs for HSE compliance and inspires further research on related tasks.
>
---
#### [new 012] ToolHaystack: Stress-Testing Tool-Augmented Language Models in Realistic Long-Term Interactions
- **分类: cs.CL**

- **简介: 该论文提出ToolHaystack基准，评估LLMs在长期复杂交互中的工具使用能力。针对现有评测忽视长期上下文和干扰的问题，设计含多任务及噪声的连续对话测试，发现14个先进模型在此场景下表现显著下滑，揭示其长期稳健性不足。**

- **链接: [http://arxiv.org/pdf/2505.23662v1](http://arxiv.org/pdf/2505.23662v1)**

> **作者:** Beong-woo Kwak; Minju Kim; Dongha Lim; Hyungjoo Chae; Dongjin Kang; Sunghwan Kim; Dongil Yang; Jinyoung Yeo
>
> **备注:** Our code and data are available at https://github.com/bwookwak/ToolHaystack
>
> **摘要:** Large language models (LLMs) have demonstrated strong capabilities in using external tools to address user inquiries. However, most existing evaluations assume tool use in short contexts, offering limited insight into model behavior during realistic long-term interactions. To fill this gap, we introduce ToolHaystack, a benchmark for testing the tool use capabilities in long-term interactions. Each test instance in ToolHaystack includes multiple tasks execution contexts and realistic noise within a continuous conversation, enabling assessment of how well models maintain context and handle various disruptions. By applying this benchmark to 14 state-of-the-art LLMs, we find that while current models perform well in standard multi-turn settings, they often significantly struggle in ToolHaystack, highlighting critical gaps in their long-term robustness not revealed by previous tool benchmarks.
>
---
#### [new 013] Can Modern NLP Systems Reliably Annotate Chest Radiography Exams? A Pre-Purchase Evaluation and Comparative Study of Solutions from AWS, Google, Azure, John Snow Labs, and Open-Source Models on an Independent Pediatric Dataset
- **分类: cs.CL**

- **简介: 该论文评估AWS、Google等四款商业NLP系统及开源模型在儿科胸片报告标注的性能，解决临床NLP工具缺乏针对性评测的问题。通过分析9.5万份报告，比较实体提取与断言检测准确性，发现系统间表现差异显著（最高76% vs 最低50%），强调部署前需严格验证。**

- **链接: [http://arxiv.org/pdf/2505.23030v1](http://arxiv.org/pdf/2505.23030v1)**

> **作者:** Shruti Hegde; Mabon Manoj Ninan; Jonathan R. Dillman; Shireen Hayatghaibi; Lynn Babcock; Elanchezhian Somasundaram
>
> **摘要:** General-purpose clinical natural language processing (NLP) tools are increasingly used for the automatic labeling of clinical reports. However, independent evaluations for specific tasks, such as pediatric chest radiograph (CXR) report labeling, are limited. This study compares four commercial clinical NLP systems - Amazon Comprehend Medical (AWS), Google Healthcare NLP (GC), Azure Clinical NLP (AZ), and SparkNLP (SP) - for entity extraction and assertion detection in pediatric CXR reports. Additionally, CheXpert and CheXbert, two dedicated chest radiograph report labelers, were evaluated on the same task using CheXpert-defined labels. We analyzed 95,008 pediatric CXR reports from a large academic pediatric hospital. Entities and assertion statuses (positive, negative, uncertain) from the findings and impression sections were extracted by the NLP systems, with impression section entities mapped to 12 disease categories and a No Findings category. CheXpert and CheXbert extracted the same 13 categories. Outputs were compared using Fleiss Kappa and accuracy against a consensus pseudo-ground truth. Significant differences were found in the number of extracted entities and assertion distributions across NLP systems. SP extracted 49,688 unique entities, GC 16,477, AZ 31,543, and AWS 27,216. Assertion accuracy across models averaged around 62%, with SP highest (76%) and AWS lowest (50%). CheXpert and CheXbert achieved 56% accuracy. Considerable variability in performance highlights the need for careful validation and review before deploying NLP tools for clinical report labeling.
>
---
#### [new 014] Table-R1: Inference-Time Scaling for Table Reasoning
- **分类: cs.CL**

- **简介: 该论文研究表格推理任务的推理时间缩放问题，提出蒸馏与RLVR（强化学习+可验证奖励）两种后训练策略。通过DeepSeek-R1推理轨迹蒸馏优化LLM得Table-R1-SFT模型，结合任务特定奖励函数与GRPO算法得Table-R1-Zero模型。7B参数的Zero模型在多类表格任务中性能超GPT-4.1及DeepSeek-R1，且跨领域泛化性强。**

- **链接: [http://arxiv.org/pdf/2505.23621v1](http://arxiv.org/pdf/2505.23621v1)**

> **作者:** Zheyuan Yang; Lyuhao Chen; Arman Cohan; Yilun Zhao
>
> **摘要:** In this work, we present the first study to explore inference-time scaling on table reasoning tasks. We develop and evaluate two post-training strategies to enable inference-time scaling: distillation from frontier model reasoning traces and reinforcement learning with verifiable rewards (RLVR). For distillation, we introduce a large-scale dataset of reasoning traces generated by DeepSeek-R1, which we use to fine-tune LLMs into the Table-R1-SFT model. For RLVR, we propose task-specific verifiable reward functions and apply the GRPO algorithm to obtain the Table-R1-Zero model. We evaluate our Table-R1-series models across diverse table reasoning tasks, including short-form QA, fact verification, and free-form QA. Notably, the Table-R1-Zero model matches or exceeds the performance of GPT-4.1 and DeepSeek-R1, while using only a 7B-parameter LLM. It also demonstrates strong generalization to out-of-domain datasets. Extensive ablation and qualitative analyses reveal the benefits of instruction tuning, model architecture choices, and cross-task generalization, as well as emergence of essential table reasoning skills during RL training.
>
---
#### [new 015] From Chat Logs to Collective Insights: Aggregative Question Answering
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出聚合式问答（AQA）任务，解决现有方法无法有效从大规模聊天记录中提取集体见解的问题。通过构建含6,027个问题的WildChat-AQA基准数据集，验证现有模型在处理聚合性查询（如特定群体的新兴问题）时存在推理不足或计算成本过高的缺陷，强调需开发新方法挖掘对话数据的集体洞察。**

- **链接: [http://arxiv.org/pdf/2505.23765v1](http://arxiv.org/pdf/2505.23765v1)**

> **作者:** Wentao Zhang; Woojeong Kim; Yuntian Deng
>
> **摘要:** Conversational agents powered by large language models (LLMs) are rapidly becoming integral to our daily interactions, generating unprecedented amounts of conversational data. Such datasets offer a powerful lens into societal interests, trending topics, and collective concerns. Yet, existing approaches typically treat these interactions as independent and miss critical insights that could emerge from aggregating and reasoning across large-scale conversation logs. In this paper, we introduce Aggregative Question Answering, a novel task requiring models to reason explicitly over thousands of user-chatbot interactions to answer aggregative queries, such as identifying emerging concerns among specific demographics. To enable research in this direction, we construct a benchmark, WildChat-AQA, comprising 6,027 aggregative questions derived from 182,330 real-world chatbot conversations. Experiments show that existing methods either struggle to reason effectively or incur prohibitive computational costs, underscoring the need for new approaches capable of extracting collective insights from large-scale conversational data.
>
---
#### [new 016] Adaptive Jailbreaking Strategies Based on the Semantic Understanding Capabilities of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于AI安全领域对抗攻击任务，针对大语言模型（LLMs）的安全约束绕过问题，提出基于语义理解能力的自适应越狱策略。通过将LLMs分为Type I/II两类，设计针对性攻击方法，实验显示对GPT-4o等模型成功率高达98.9%。**

- **链接: [http://arxiv.org/pdf/2505.23404v1](http://arxiv.org/pdf/2505.23404v1)**

> **作者:** Mingyu Yu; Wei Wang; Yanjie Wei; Sujuan Qin
>
> **摘要:** Adversarial attacks on Large Language Models (LLMs) via jailbreaking techniques-methods that circumvent their built-in safety and ethical constraints-have emerged as a critical challenge in AI security. These attacks compromise the reliability of LLMs by exploiting inherent weaknesses in their comprehension capabilities. This paper investigates the efficacy of jailbreaking strategies that are specifically adapted to the diverse levels of understanding exhibited by different LLMs. We propose the Adaptive Jailbreaking Strategies Based on the Semantic Understanding Capabilities of Large Language Models, a novel framework that classifies LLMs into Type I and Type II categories according to their semantic comprehension abilities. For each category, we design tailored jailbreaking strategies aimed at leveraging their vulnerabilities to facilitate successful attacks. Extensive experiments conducted on multiple LLMs demonstrate that our adaptive strategy markedly improves the success rate of jailbreaking. Notably, our approach achieves an exceptional 98.9% success rate in jailbreaking GPT-4o(29 May 2025 release)
>
---
#### [new 017] Generating Diverse Training Samples for Relation Extraction with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于关系抽取任务，旨在解决LLMs生成训练样本结构相似、措辞单一的问题。提出通过ICL提示指令和DPO微调两种方法提升样本多样性并保持正确性，实验表明生成数据能更好训练非LLM模型。**

- **链接: [http://arxiv.org/pdf/2505.23108v1](http://arxiv.org/pdf/2505.23108v1)**

> **作者:** Zexuan Li; Hongliang Dai; Piji Li
>
> **备注:** ACL2025 Main
>
> **摘要:** Using Large Language Models (LLMs) to generate training data can potentially be a preferable way to improve zero or few-shot NLP tasks. However, many problems remain to be investigated for this direction. For the task of Relation Extraction (RE), we find that samples generated by directly prompting LLMs may easily have high structural similarities with each other. They tend to use a limited variety of phrasing while expressing the relation between a pair of entities. Therefore, in this paper, we study how to effectively improve the diversity of the training samples generated with LLMs for RE, while also maintaining their correctness. We first try to make the LLMs produce dissimilar samples by directly giving instructions in In-Context Learning (ICL) prompts. Then, we propose an approach to fine-tune LLMs for diversity training sample generation through Direct Preference Optimization (DPO). Our experiments on commonly used RE datasets show that both attempts can improve the quality of the generated training data. We also find that comparing with directly performing RE with an LLM, training a non-LLM RE model with its generated samples may lead to better performance.
>
---
#### [new 018] MMBoundary: Advancing MLLM Knowledge Boundary Awareness through Reasoning Step Confidence Calibration
- **分类: cs.CL**

- **简介: 该论文属于多模态推理任务，解决MLLM多步骤推理中因未评估每步置信度导致的幻觉累积问题。提出MMBoundary框架，通过文本与跨模态自奖励信号估计每步置信度，结合监督微调和多奖励强化学习校准，提升推理链自修正能力，降低误差并提升性能。**

- **链接: [http://arxiv.org/pdf/2505.23224v1](http://arxiv.org/pdf/2505.23224v1)**

> **作者:** Zhitao He; Sandeep Polisetty; Zhiyuan Fan; Yuchen Huang; Shujin Wu; Yi R.; Fung
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** In recent years, multimodal large language models (MLLMs) have made significant progress but continue to face inherent challenges in multimodal reasoning, which requires multi-level (e.g., perception, reasoning) and multi-granular (e.g., multi-step reasoning chain) advanced inferencing. Prior work on estimating model confidence tends to focus on the overall response for training and calibration, but fails to assess confidence in each reasoning step, leading to undesirable hallucination snowballing. In this work, we present MMBoundary, a novel framework that advances the knowledge boundary awareness of MLLMs through reasoning step confidence calibration. To achieve this, we propose to incorporate complementary textual and cross-modal self-rewarding signals to estimate confidence at each step of the MLLM reasoning process. In addition to supervised fine-tuning MLLM on this set of self-rewarded confidence estimation signal for initial confidence expression warm-up, we introduce a reinforcement learning stage with multiple reward functions for further aligning model knowledge and calibrating confidence at each reasoning step, enhancing reasoning chain self-correction. Empirical results show that MMBoundary significantly outperforms existing methods across diverse domain datasets and metrics, achieving an average of 7.5% reduction in multimodal confidence calibration errors and up to 8.3% improvement in task performance.
>
---
#### [new 019] Unraveling LoRA Interference: Orthogonal Subspaces for Robust Model Merging
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型合并任务，解决LoRA微调模型合并时性能下降问题。提出OSRM方法，通过正交子空间约束参数，减少任务间干扰，提升合并效果并保持单任务准确率，兼容现有算法。**

- **链接: [http://arxiv.org/pdf/2505.22934v1](http://arxiv.org/pdf/2505.22934v1)**

> **作者:** Haobo Zhang; Jiayu Zhou
>
> **备注:** 14 pages, 5 figures, 16 tables, accepted by ACL 2025
>
> **摘要:** Fine-tuning large language models (LMs) for individual tasks yields strong performance but is expensive for deployment and storage. Recent works explore model merging to combine multiple task-specific models into a single multi-task model without additional training. However, existing merging methods often fail for models fine-tuned with low-rank adaptation (LoRA), due to significant performance degradation. In this paper, we show that this issue arises from a previously overlooked interplay between model parameters and data distributions. We propose Orthogonal Subspaces for Robust model Merging (OSRM) to constrain the LoRA subspace *prior* to fine-tuning, ensuring that updates relevant to one task do not adversely shift outputs for others. Our approach can seamlessly integrate with most existing merging algorithms, reducing the unintended interference among tasks. Extensive experiments on eight datasets, tested with three widely used LMs and two large LMs, demonstrate that our method not only boosts merging performance but also preserves single-task accuracy. Furthermore, our approach exhibits greater robustness to the hyperparameters of merging. These results highlight the importance of data-parameter interaction in model merging and offer a plug-and-play solution for merging LoRA models.
>
---
#### [new 020] SocialMaze: A Benchmark for Evaluating Social Reasoning in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出SocialMaze基准，评估大语言模型的社交推理能力，解决现有评测 oversimplified且无法全面考察动态交互与信息不确定性的不足。通过设计涵盖社交游戏、日常互动和数字社区的6项任务，结合自动与人工验证，分析模型在复杂社交场景中的推理表现，并验证针对性微调的有效性。**

- **链接: [http://arxiv.org/pdf/2505.23713v1](http://arxiv.org/pdf/2505.23713v1)**

> **作者:** Zixiang Xu; Yanbo Wang; Yue Huang; Jiayi Ye; Haomin Zhuang; Zirui Song; Lang Gao; Chenxi Wang; Zhaorun Chen; Yujun Zhou; Sixian Li; Wang Pan; Yue Zhao; Jieyu Zhao; Xiangliang Zhang; Xiuying Chen
>
> **备注:** Code available at https://github.com/xzx34/SocialMaze
>
> **摘要:** Large language models (LLMs) are increasingly applied to socially grounded tasks, such as online community moderation, media content analysis, and social reasoning games. Success in these contexts depends on a model's social reasoning ability - the capacity to interpret social contexts, infer others' mental states, and assess the truthfulness of presented information. However, there is currently no systematic evaluation framework that comprehensively assesses the social reasoning capabilities of LLMs. Existing efforts often oversimplify real-world scenarios and consist of tasks that are too basic to challenge advanced models. To address this gap, we introduce SocialMaze, a new benchmark specifically designed to evaluate social reasoning. SocialMaze systematically incorporates three core challenges: deep reasoning, dynamic interaction, and information uncertainty. It provides six diverse tasks across three key settings: social reasoning games, daily-life interactions, and digital community platforms. Both automated and human validation are used to ensure data quality. Our evaluation reveals several key insights: models vary substantially in their ability to handle dynamic interactions and integrate temporally evolving information; models with strong chain-of-thought reasoning perform better on tasks requiring deeper inference beyond surface-level cues; and model reasoning degrades significantly under uncertainty. Furthermore, we show that targeted fine-tuning on curated reasoning examples can greatly improve model performance in complex social scenarios. The dataset is publicly available at: https://huggingface.co/datasets/MBZUAI/SocialMaze
>
---
#### [new 021] NegVQA: Can Vision Language Models Understand Negation?
- **分类: cs.CL; cs.AI; cs.CV; cs.CY; cs.LG**

- **简介: 该论文属于视觉问答（VQA）任务，旨在评估视觉语言模型（VLMs）对否定的理解能力。针对VLMs在否定场景下的表现缺陷，构建了含7,379个否定问题的NegVQA基准数据集，并发现模型性能显著下降且存在U型规模效应，揭示了VLMs的关键不足。**

- **链接: [http://arxiv.org/pdf/2505.22946v1](http://arxiv.org/pdf/2505.22946v1)**

> **作者:** Yuhui Zhang; Yuchang Su; Yiming Liu; Serena Yeung-Levy
>
> **备注:** Published at ACL 2025 Findings
>
> **摘要:** Negation is a fundamental linguistic phenomenon that can entirely reverse the meaning of a sentence. As vision language models (VLMs) continue to advance and are deployed in high-stakes applications, assessing their ability to comprehend negation becomes essential. To address this, we introduce NegVQA, a visual question answering (VQA) benchmark consisting of 7,379 two-choice questions covering diverse negation scenarios and image-question distributions. We construct NegVQA by leveraging large language models to generate negated versions of questions from existing VQA datasets. Evaluating 20 state-of-the-art VLMs across seven model families, we find that these models struggle significantly with negation, exhibiting a substantial performance drop compared to their responses to the original questions. Furthermore, we uncover a U-shaped scaling trend, where increasing model size initially degrades performance on NegVQA before leading to improvements. Our benchmark reveals critical gaps in VLMs' negation understanding and offers insights into future VLM development. Project page available at https://yuhui-zh15.github.io/NegVQA/.
>
---
#### [new 022] Query Routing for Retrieval-Augmented Language Models
- **分类: cs.CL**

- **简介: 该论文属于查询路由任务，旨在解决检索增强语言模型（RAG）中因静态路由方法无法适应动态文档影响导致的响应质量不均问题。提出RAGRouter，通过结合文档嵌入与对比学习，动态捕捉知识变化以优化模型选择，实验显示其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.23052v1](http://arxiv.org/pdf/2505.23052v1)**

> **作者:** Jiarui Zhang; Xiangyu Liu; Yong Hu; Chaoyue Niu; Fan Wu; Guihai Chen
>
> **摘要:** Retrieval-Augmented Generation (RAG) significantly improves the performance of Large Language Models (LLMs) on knowledge-intensive tasks. However, varying response quality across LLMs under RAG necessitates intelligent routing mechanisms, which select the most suitable model for each query from multiple retrieval-augmented LLMs via a dedicated router model. We observe that external documents dynamically affect LLMs' ability to answer queries, while existing routing methods, which rely on static parametric knowledge representations, exhibit suboptimal performance in RAG scenarios. To address this, we formally define the new retrieval-augmented LLM routing problem, incorporating the influence of retrieved documents into the routing framework. We propose RAGRouter, a RAG-aware routing design, which leverages document embeddings and RAG capability embeddings with contrastive learning to capture knowledge representation shifts and enable informed routing decisions. Extensive experiments on diverse knowledge-intensive tasks and retrieval settings show that RAGRouter outperforms the best individual LLM by 3.61% on average and existing routing methods by 3.29%-9.33%. With an extended score-threshold-based mechanism, it also achieves strong performance-efficiency trade-offs under low-latency constraints.
>
---
#### [new 023] Bayesian Attention Mechanism: A Probabilistic Framework for Positional Encoding and Context Length Extrapolation
- **分类: cs.CL; cs.LG; I.2.6; I.2.7**

- **简介: 该论文属于Transformer模型优化任务，旨在解决传统位置编码（PE）方法在理论解释和长上下文外推能力上的不足。提出贝叶斯注意力机制（BAM），将PE建模为概率先验，统一现有方法并引入广义高斯先验，提升长序列信息检索能力，在500倍训练上下文长度时仍保持高准确率，参数增加极少。**

- **链接: [http://arxiv.org/pdf/2505.22842v1](http://arxiv.org/pdf/2505.22842v1)**

> **作者:** Arthur S. Bianchessi; Rodrigo C. Barros; Lucas S. Kupssinskü
>
> **摘要:** Transformer-based language models rely on positional encoding (PE) to handle token order and support context length extrapolation. However, existing PE methods lack theoretical clarity and rely on limited evaluation metrics to substantiate their extrapolation claims. We propose the Bayesian Attention Mechanism (BAM), a theoretical framework that formulates positional encoding as a prior within a probabilistic model. BAM unifies existing methods (e.g., NoPE and ALiBi) and motivates a new Generalized Gaussian positional prior that substantially improves long-context generalization. Empirically, BAM enables accurate information retrieval at $500\times$ the training context length, outperforming previous state-of-the-art context length generalization in long context retrieval accuracy while maintaining comparable perplexity and introducing minimal additional parameters.
>
---
#### [new 024] Satori-SWE: Evolutionary Test-Time Scaling for Sample-Efficient Software Engineering
- **分类: cs.CL; cs.AI; cs.SE**

- **简介: 该论文提出EvoScale方法，解决小规模语言模型在软件工程任务（如GitHub问题处理）中性能不足的问题。通过进化算法与强化学习结合，迭代优化生成结果，减少采样需求，提升32B模型性能至超百亿参数模型水平，实现高效样本利用。**

- **链接: [http://arxiv.org/pdf/2505.23604v1](http://arxiv.org/pdf/2505.23604v1)**

> **作者:** Guangtao Zeng; Maohao Shen; Delin Chen; Zhenting Qi; Subhro Das; Dan Gutfreund; David Cox; Gregory Wornell; Wei Lu; Zhang-Wei Hong; Chuang Gan
>
> **摘要:** Language models (LMs) perform well on standardized coding benchmarks but struggle with real-world software engineering tasks such as resolving GitHub issues in SWE-Bench, especially when model parameters are less than 100B. While smaller models are preferable in practice due to their lower computational cost, improving their performance remains challenging. Existing approaches primarily rely on supervised fine-tuning (SFT) with high-quality data, which is expensive to curate at scale. An alternative is test-time scaling: generating multiple outputs, scoring them using a verifier, and selecting the best one. Although effective, this strategy often requires excessive sampling and costly scoring, limiting its practical application. We propose Evolutionary Test-Time Scaling (EvoScale), a sample-efficient method that treats generation as an evolutionary process. By iteratively refining outputs via selection and mutation, EvoScale shifts the output distribution toward higher-scoring regions, reducing the number of samples needed to find correct solutions. To reduce the overhead from repeatedly sampling and selection, we train the model to self-evolve using reinforcement learning (RL). Rather than relying on external verifiers at inference time, the model learns to self-improve the scores of its own generations across iterations. Evaluated on SWE-Bench-Verified, EvoScale enables our 32B model, Satori-SWE-32B, to match or exceed the performance of models with over 100B parameters while using a few samples. Code, data, and models will be fully open-sourced.
>
---
#### [new 025] Generalized Category Discovery in Event-Centric Contexts: Latent Pattern Mining with LLMs
- **分类: cs.CL**

- **简介: 该论文提出PaMA框架，针对事件中心场景下的广义类别发现（EC-GCD）任务，解决长文本中主观聚类分歧和少数类不平衡问题。通过LLM挖掘事件模式并优化分类-聚类对齐，结合平衡原型挖掘 pipeline，在新 scam 数据集上提升12.58% H-score，验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.23304v1](http://arxiv.org/pdf/2505.23304v1)**

> **作者:** Yi Luo; Qiwen Wang; Junqi Yang; Luyao Tang; Zhenghao Lin; Zhenzhe Ying; Weiqiang Wang; Chen Lin
>
> **摘要:** Generalized Category Discovery (GCD) aims to classify both known and novel categories using partially labeled data that contains only known classes. Despite achieving strong performance on existing benchmarks, current textual GCD methods lack sufficient validation in realistic settings. We introduce Event-Centric GCD (EC-GCD), characterized by long, complex narratives and highly imbalanced class distributions, posing two main challenges: (1) divergent clustering versus classification groupings caused by subjective criteria, and (2) Unfair alignment for minority classes. To tackle these, we propose PaMA, a framework leveraging LLMs to extract and refine event patterns for improved cluster-class alignment. Additionally, a ranking-filtering-mining pipeline ensures balanced representation of prototypes across imbalanced categories. Evaluations on two EC-GCD benchmarks, including a newly constructed Scam Report dataset, demonstrate that PaMA outperforms prior methods with up to 12.58% H-score gains, while maintaining strong generalization on base GCD datasets.
>
---
#### [new 026] MEDAL: A Framework for Benchmarking LLMs as Multilingual Open-Domain Chatbots and Dialogue Evaluators
- **分类: cs.CL**

- **简介: 该论文提出MEDAL框架，用于评估多语言开放领域聊天机器人及对话评估模型。针对现有基准数据静态、过时、缺乏多语言覆盖的问题，通过多LLM生成多样化对话、GPT-4.1分析性能差异，并构建新多语种基准测试集，揭示当前LLMs在共情和推理等细微问题上的局限性。**

- **链接: [http://arxiv.org/pdf/2505.22777v1](http://arxiv.org/pdf/2505.22777v1)**

> **作者:** John Mendonça; Alon Lavie; Isabel Trancoso
>
> **备注:** May ARR
>
> **摘要:** As the capabilities of chatbots and their underlying LLMs continue to dramatically improve, evaluating their performance has increasingly become a major blocker to their further development. A major challenge is the available benchmarking datasets, which are largely static, outdated, and lacking in multilingual coverage, limiting their ability to capture subtle linguistic and cultural variations. This paper introduces MEDAL, an automated multi-agent framework for generating, evaluating, and curating more representative and diverse open-domain dialogue evaluation benchmarks. Our approach leverages several state-of-the-art LLMs to generate user-chatbot multilingual dialogues, conditioned on varied seed contexts. A strong LLM (GPT-4.1) is then used for a multidimensional analysis of the performance of the chatbots, uncovering noticeable cross-lingual performance differences. Guided by this large-scale evaluation, we curate a new meta-evaluation multilingual benchmark and human-annotate samples with nuanced quality judgments. This benchmark is then used to assess the ability of several reasoning and non-reasoning LLMs to act as evaluators of open-domain dialogues. We find that current LLMs struggle to detect nuanced issues, particularly those involving empathy and reasoning.
>
---
#### [new 027] WorkForceAgent-R1: Incentivizing Reasoning Capability in LLM-based Web Agents via Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出WorkForceAgent-R1，针对基于LLM的Web代理在动态企业网页任务中推理能力不足的问题，通过规则驱动的R1强化学习框架优化单步推理与规划，采用结构化奖励函数提升动作正确性和格式规范性，实验显示其较监督基线提升10.26-16.59%，性能接近GPT-4o。**

- **链接: [http://arxiv.org/pdf/2505.22942v1](http://arxiv.org/pdf/2505.22942v1)**

> **作者:** Yuchen Zhuang; Di Jin; Jiaao Chen; Wenqi Shi; Hanrui Wang; Chao Zhang
>
> **备注:** Work in Progress
>
> **摘要:** Large language models (LLMs)-empowered web agents enables automating complex, real-time web navigation tasks in enterprise environments. However, existing web agents relying on supervised fine-tuning (SFT) often struggle with generalization and robustness due to insufficient reasoning capabilities when handling the inherently dynamic nature of web interactions. In this study, we introduce WorkForceAgent-R1, an LLM-based web agent trained using a rule-based R1-style reinforcement learning framework designed explicitly to enhance single-step reasoning and planning for business-oriented web navigation tasks. We employ a structured reward function that evaluates both adherence to output formats and correctness of actions, enabling WorkForceAgent-R1 to implicitly learn robust intermediate reasoning without explicit annotations or extensive expert demonstrations. Extensive experiments on the WorkArena benchmark demonstrate that WorkForceAgent-R1 substantially outperforms SFT baselines by 10.26-16.59%, achieving competitive performance relative to proprietary LLM-based agents (gpt-4o) in workplace-oriented web navigation tasks.
>
---
#### [new 028] Training Language Models to Generate Quality Code with Program Analysis Feedback
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出REAL框架，通过强化学习结合程序分析和单元测试，解决大语言模型生成代码质量不足（如安全漏洞、可维护性差）的问题。无需人工标注，优于现有方法，提升代码质量和功能性。**

- **链接: [http://arxiv.org/pdf/2505.22704v1](http://arxiv.org/pdf/2505.22704v1)**

> **作者:** Feng Yao; Zilong Wang; Liyuan Liu; Junxia Cui; Li Zhong; Xiaohan Fu; Haohui Mai; Vish Krishnan; Jianfeng Gao; Jingbo Shang
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Code generation with large language models (LLMs), often termed vibe coding, is increasingly adopted in production but fails to ensure code quality, particularly in security (e.g., SQL injection vulnerabilities) and maintainability (e.g., missing type annotations). Existing methods, such as supervised fine-tuning and rule-based post-processing, rely on labor-intensive annotations or brittle heuristics, limiting their scalability and effectiveness. We propose REAL, a reinforcement learning framework that incentivizes LLMs to generate production-quality code using program analysis-guided feedback. Specifically, REAL integrates two automated signals: (1) program analysis detecting security or maintainability defects and (2) unit tests ensuring functional correctness. Unlike prior work, our framework is prompt-agnostic and reference-free, enabling scalable supervision without manual intervention. Experiments across multiple datasets and model scales demonstrate that REAL outperforms state-of-the-art methods in simultaneous assessments of functionality and code quality. Our work bridges the gap between rapid prototyping and production-ready code, enabling LLMs to deliver both speed and quality.
>
---
#### [new 029] Label-Guided In-Context Learning for Named Entity Recognition
- **分类: cs.CL**

- **简介: 该论文属于命名实体识别（NER）任务，针对现有In-Context Learning（ICL）方法因忽略训练标签导致性能不足的问题，提出DEER方法：通过标签引导的token级统计优化示例选择，并定位易错token进行修正。实验显示其性能接近监督微调，且在低资源场景稳健。**

- **链接: [http://arxiv.org/pdf/2505.23722v1](http://arxiv.org/pdf/2505.23722v1)**

> **作者:** Fan Bai; Hamid Hassanzadeh; Ardavan Saeedi; Mark Dredze
>
> **备注:** Preprint
>
> **摘要:** In-context learning (ICL) enables large language models (LLMs) to perform new tasks using only a few demonstrations. In Named Entity Recognition (NER), demonstrations are typically selected based on semantic similarity to the test instance, ignoring training labels and resulting in suboptimal performance. We introduce DEER, a new method that leverages training labels through token-level statistics to improve ICL performance. DEER first enhances example selection with a label-guided, token-based retriever that prioritizes tokens most informative for entity recognition. It then prompts the LLM to revisit error-prone tokens, which are also identified using label statistics, and make targeted corrections. Evaluated on five NER datasets using four different LLMs, DEER consistently outperforms existing ICL methods and approaches the performance of supervised fine-tuning. Further analysis shows its effectiveness on both seen and unseen entities and its robustness in low-resource settings.
>
---
#### [new 030] Threading the Needle: Reweaving Chain-of-Thought Reasoning to Explain Human Label Variation
- **分类: cs.CL**

- **简介: 该论文针对现有LLM解释方法逆向生成无法有效捕捉人类标注差异的问题，提出基于思维链的正向推理路径提取支持/反对陈述，并开发HLV评估框架通过排名比较提升与人类标注分布的对齐，实验显示优于基线方法。**

- **链接: [http://arxiv.org/pdf/2505.23368v1](http://arxiv.org/pdf/2505.23368v1)**

> **作者:** Beiduo Chen; Yang Janet Liu; Anna Korhonen; Barbara Plank
>
> **备注:** 22 pages, 7 figures
>
> **摘要:** The recent rise of reasoning-tuned Large Language Models (LLMs)--which generate chains of thought (CoTs) before giving the final answer--has attracted significant attention and offers new opportunities for gaining insights into human label variation, which refers to plausible differences in how multiple annotators label the same data instance. Prior work has shown that LLM-generated explanations can help align model predictions with human label distributions, but typically adopt a reverse paradigm: producing explanations based on given answers. In contrast, CoTs provide a forward reasoning path that may implicitly embed rationales for each answer option, before generating the answers. We thus propose a novel LLM-based pipeline enriched with linguistically-grounded discourse segmenters to extract supporting and opposing statements for each answer option from CoTs with improved accuracy. We also propose a rank-based HLV evaluation framework that prioritizes the ranking of answers over exact scores, which instead favor direct comparison of label distributions. Our method outperforms a direct generation method as well as baselines on three datasets, and shows better alignment of ranking methods with humans, highlighting the effectiveness of our approach.
>
---
#### [new 031] DeepTheorem: Advancing LLM Reasoning for Theorem Proving Through Natural Language and Reinforcement Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DeepTheorem框架，针对传统形式化定理证明与LLM的自然语言优势不匹配问题，构建121K规模的IMO级非形式化数学定理数据集，设计RL-Zero强化学习策略及推理评估指标，提升LLM的数学定理证明能力。**

- **链接: [http://arxiv.org/pdf/2505.23754v1](http://arxiv.org/pdf/2505.23754v1)**

> **作者:** Ziyin Zhang; Jiahao Xu; Zhiwei He; Tian Liang; Qiuzhi Liu; Yansi Li; Linfeng Song; Zhengwen Liang; Zhuosheng Zhang; Rui Wang; Zhaopeng Tu; Haitao Mi; Dong Yu
>
> **摘要:** Theorem proving serves as a major testbed for evaluating complex reasoning abilities in large language models (LLMs). However, traditional automated theorem proving (ATP) approaches rely heavily on formal proof systems that poorly align with LLMs' strength derived from informal, natural language knowledge acquired during pre-training. In this work, we propose DeepTheorem, a comprehensive informal theorem-proving framework exploiting natural language to enhance LLM mathematical reasoning. DeepTheorem includes a large-scale benchmark dataset consisting of 121K high-quality IMO-level informal theorems and proofs spanning diverse mathematical domains, rigorously annotated for correctness, difficulty, and topic categories, accompanied by systematically constructed verifiable theorem variants. We devise a novel reinforcement learning strategy (RL-Zero) explicitly tailored to informal theorem proving, leveraging the verified theorem variants to incentivize robust mathematical inference. Additionally, we propose comprehensive outcome and process evaluation metrics examining proof correctness and the quality of reasoning steps. Extensive experimental analyses demonstrate DeepTheorem significantly improves LLM theorem-proving performance compared to existing datasets and supervised fine-tuning protocols, achieving state-of-the-art accuracy and reasoning quality. Our findings highlight DeepTheorem's potential to fundamentally advance automated informal theorem proving and mathematical exploration.
>
---
#### [new 032] VIGNETTE: Socially Grounded Bias Evaluation for Vision-Language Models
- **分类: cs.CL**

- **简介: 该论文提出VIGNETTE，一个含30M+图像的VQA基准，评估视觉语言模型(VLM)的社会偏见。针对现有研究局限于性别职业关联的问题，覆盖事实性、感知、刻板印象和决策四方向，分析模型如何通过视觉线索推断社会层级，揭示隐含歧视模式，填补VLM复杂社会偏见评估的空白。**

- **链接: [http://arxiv.org/pdf/2505.22897v1](http://arxiv.org/pdf/2505.22897v1)**

> **作者:** Chahat Raj; Bowen Wei; Aylin Caliskan; Antonios Anastasopoulos; Ziwei Zhu
>
> **备注:** 17 pages
>
> **摘要:** While bias in large language models (LLMs) is well-studied, similar concerns in vision-language models (VLMs) have received comparatively less attention. Existing VLM bias studies often focus on portrait-style images and gender-occupation associations, overlooking broader and more complex social stereotypes and their implied harm. This work introduces VIGNETTE, a large-scale VQA benchmark with 30M+ images for evaluating bias in VLMs through a question-answering framework spanning four directions: factuality, perception, stereotyping, and decision making. Beyond narrowly-centered studies, we assess how VLMs interpret identities in contextualized settings, revealing how models make trait and capability assumptions and exhibit patterns of discrimination. Drawing from social psychology, we examine how VLMs connect visual identity cues to trait and role-based inferences, encoding social hierarchies, through biased selections. Our findings uncover subtle, multifaceted, and surprising stereotypical patterns, offering insights into how VLMs construct social meaning from inputs.
>
---
#### [new 033] CLaC at SemEval-2025 Task 6: A Multi-Architecture Approach for Corporate Environmental Promise Verification
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对企业ESG报告中的环境承诺验证任务（SemEval-2025任务6），解决承诺识别、证据评估、清晰度评价及验证时间判断四个子问题。提出三种模型：ESG-BERT分类器、融合语言特征的改进模型及结合注意力机制、文档元数据和多目标学习的集成模型。通过实验显示，集成方法以0.5268的得分超越基线，验证了语言特征、注意力机制等方法的有效性。**

- **链接: [http://arxiv.org/pdf/2505.23538v1](http://arxiv.org/pdf/2505.23538v1)**

> **作者:** Nawar Turk; Eeham Khan; Leila Kosseim
>
> **备注:** Accepted to SemEval-2025 Task 6 (ACL 2025)
>
> **摘要:** This paper presents our approach to the SemEval-2025 Task~6 (PromiseEval), which focuses on verifying promises in corporate ESG (Environmental, Social, and Governance) reports. We explore three model architectures to address the four subtasks of promise identification, supporting evidence assessment, clarity evaluation, and verification timing. Our first model utilizes ESG-BERT with task-specific classifier heads, while our second model enhances this architecture with linguistic features tailored for each subtask. Our third approach implements a combined subtask model with attention-based sequence pooling, transformer representations augmented with document metadata, and multi-objective learning. Experiments on the English portion of the ML-Promise dataset demonstrate progressive improvement across our models, with our combined subtask approach achieving a leaderboard score of 0.5268, outperforming the provided baseline of 0.5227. Our work highlights the effectiveness of linguistic feature extraction, attention pooling, and multi-objective learning in promise verification tasks, despite challenges posed by class imbalance and limited training data.
>
---
#### [new 034] From Parameters to Prompts: Understanding and Mitigating the Factuality Gap between Fine-Tuned LLMs
- **分类: cs.CL**

- **简介: 该论文属事实知识提取任务，旨在解决微调LLMs的事实性差距问题。通过实验与理论分析，发现测试阶段的提示（如ICL）可有效缓解已知与未知知识间的差距，证明提示作用超越微调数据，ICL能弥补数据不足，强调需重新评估数据选择方法。**

- **链接: [http://arxiv.org/pdf/2505.23410v1](http://arxiv.org/pdf/2505.23410v1)**

> **作者:** Xuan Gong; Hanbo Huang; Shiyu Liang
>
> **备注:** The code of this paper will be released soon
>
> **摘要:** Factual knowledge extraction aims to explicitly extract knowledge parameterized in pre-trained language models for application in downstream tasks. While prior work has been investigating the impact of supervised fine-tuning data on the factuality of large language models (LLMs), its mechanism remains poorly understood. We revisit this impact through systematic experiments, with a particular focus on the factuality gap that arises when fine-tuning on known versus unknown knowledge. Our findings show that this gap can be mitigated at the inference stage, either under out-of-distribution (OOD) settings or by using appropriate in-context learning (ICL) prompts (i.e., few-shot learning and Chain of Thought (CoT)). We prove this phenomenon theoretically from the perspective of knowledge graphs, showing that the test-time prompt may diminish or even overshadow the impact of fine-tuning data and play a dominant role in knowledge extraction. Ultimately, our results shed light on the interaction between finetuning data and test-time prompt, demonstrating that ICL can effectively compensate for shortcomings in fine-tuning data, and highlighting the need to reconsider the use of ICL prompting as a means to evaluate the effectiveness of fine-tuning data selection methods.
>
---
#### [new 035] Enhancing Large Language Models'Machine Translation via Dynamic Focus Anchoring
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，针对多义词等上下文敏感单元导致的翻译误差，提出动态焦点锚定方法：动态识别难点并引导LLM聚焦，减少信息扁平化错误，无需额外训练，提升多语言对翻译准确性，资源消耗低。**

- **链接: [http://arxiv.org/pdf/2505.23140v1](http://arxiv.org/pdf/2505.23140v1)**

> **作者:** Qiuyu Ding; Zhiqiang Cao; Hailong Cao; Tiejun Zhao
>
> **摘要:** Large language models have demonstrated exceptional performance across multiple crosslingual NLP tasks, including machine translation (MT). However, persistent challenges remain in addressing context-sensitive units (CSUs), such as polysemous words. These CSUs not only affect the local translation accuracy of LLMs, but also affect LLMs' understanding capability for sentences and tasks, and even lead to translation failure. To address this problem, we propose a simple but effective method to enhance LLMs' MT capabilities by acquiring CSUs and applying semantic focus. Specifically, we dynamically analyze and identify translation challenges, then incorporate them into LLMs in a structured manner to mitigate mistranslations or misunderstandings of CSUs caused by information flattening. Efficiently activate LLMs to identify and apply relevant knowledge from its vast data pool in this way, ensuring more accurate translations for translating difficult terms. On a benchmark dataset of MT, our proposed method achieved competitive performance compared to multiple existing open-sourced MT baseline models. It demonstrates effectiveness and robustness across multiple language pairs, including both similar language pairs and distant language pairs. Notably, the proposed method requires no additional model training and enhances LLMs' performance across multiple NLP tasks with minimal resource consumption.
>
---
#### [new 036] Exploring Scaling Laws for EHR Foundation Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究电子健康记录（EHR）基础模型的扩展规律，旨在填补其与语言模型（LLMs）在扩展性研究上的空白。通过在MIMIC-IV数据上训练不同规模的Transformer模型，分析计算资源、参数量与临床效用间的幂律关系等模式，揭示EHR模型与LLMs相似的扩展行为，为高效训练及临床预测提供指导。**

- **链接: [http://arxiv.org/pdf/2505.22964v1](http://arxiv.org/pdf/2505.22964v1)**

> **作者:** Sheng Zhang; Qin Liu; Naoto Usuyama; Cliff Wong; Tristan Naumann; Hoifung Poon
>
> **摘要:** The emergence of scaling laws has profoundly shaped the development of large language models (LLMs), enabling predictable performance gains through systematic increases in model size, dataset volume, and compute. Yet, these principles remain largely unexplored in the context of electronic health records (EHRs) -- a rich, sequential, and globally abundant data source that differs structurally from natural language. In this work, we present the first empirical investigation of scaling laws for EHR foundation models. By training transformer architectures on patient timeline data from the MIMIC-IV database across varying model sizes and compute budgets, we identify consistent scaling patterns, including parabolic IsoFLOPs curves and power-law relationships between compute, model parameters, data size, and clinical utility. These findings demonstrate that EHR models exhibit scaling behavior analogous to LLMs, offering predictive insights into resource-efficient training strategies. Our results lay the groundwork for developing powerful EHR foundation models capable of transforming clinical prediction tasks and advancing personalized healthcare.
>
---
#### [new 037] The Warmup Dilemma: How Learning Rate Strategies Impact Speech-to-Text Model Convergence
- **分类: cs.CL**

- **简介: 该论文聚焦语音到文本模型训练中的学习率策略优化问题，针对大规模Transformer变体（如Conformer）收敛难题，对比分析不同学习率预热方案，发现次指数预热更优，且高初始学习率可加速初期收敛但不提升最终性能。**

- **链接: [http://arxiv.org/pdf/2505.23420v1](http://arxiv.org/pdf/2505.23420v1)**

> **作者:** Marco Gaido; Sara Papi; Luisa Bentivogli; Alessio Brutti; Mauro Cettolo; Roberto Gretter; Marco Matassoni; Mohamed Nabih; Matteo Negri
>
> **备注:** Accepted to IWSLT 2025
>
> **摘要:** Training large-scale models presents challenges not only in terms of resource requirements but also in terms of their convergence. For this reason, the learning rate (LR) is often decreased when the size of a model is increased. Such a simple solution is not enough in the case of speech-to-text (S2T) trainings, where evolved and more complex variants of the Transformer architecture -- e.g., Conformer or Branchformer -- are used in light of their better performance. As a workaround, OWSM designed a double linear warmup of the LR, increasing it to a very small value in the first phase before updating it to a higher value in the second phase. While this solution worked well in practice, it was not compared with alternative solutions, nor was the impact on the final performance of different LR warmup schedules studied. This paper fills this gap, revealing that i) large-scale S2T trainings demand a sub-exponential LR warmup, and ii) a higher LR in the warmup phase accelerates initial convergence, but it does not boost final performance.
>
---
#### [new 038] Don't Take the Premise for Granted: Evaluating the Premise Critique Ability of Large Language Models
- **分类: cs.CL**

- **简介: 该论文评估大型语言模型（LLMs）对错误前提的批判能力，解决其过度信任缺陷问题。提出PCBench基准，包含四种错误类型及三难度等级，测试15个模型，发现模型依赖显式提示，能力受难度和错误类型影响，推理能力不相关，强调需加强主动输入验证。**

- **链接: [http://arxiv.org/pdf/2505.23715v1](http://arxiv.org/pdf/2505.23715v1)**

> **作者:** Jinzhe Li; Gengxu Li; Yi Chang; Yuan Wu
>
> **备注:** 31 pages,13 figures,15 tables
>
> **摘要:** Large language models (LLMs) have witnessed rapid advancements, demonstrating remarkable capabilities. However, a notable vulnerability persists: LLMs often uncritically accept flawed or contradictory premises, leading to inefficient reasoning and unreliable outputs. This emphasizes the significance of possessing the \textbf{Premise Critique Ability} for LLMs, defined as the capacity to proactively identify and articulate errors in input premises. Most existing studies assess LLMs' reasoning ability in ideal settings, largely ignoring their vulnerabilities when faced with flawed premises. Thus, we introduce the \textbf{Premise Critique Bench (PCBench)}, designed by incorporating four error types across three difficulty levels, paired with multi-faceted evaluation metrics. We conducted systematic evaluations of 15 representative LLMs. Our findings reveal: (1) Most models rely heavily on explicit prompts to detect errors, with limited autonomous critique; (2) Premise critique ability depends on question difficulty and error type, with direct contradictions being easier to detect than complex or procedural errors; (3) Reasoning ability does not consistently correlate with the premise critique ability; (4) Flawed premises trigger overthinking in reasoning models, markedly lengthening responses due to repeated attempts at resolving conflicts. These insights underscore the urgent need to enhance LLMs' proactive evaluation of input validity, positioning premise critique as a foundational capability for developing reliable, human-centric systems. The code is available at https://github.com/MLGroupJLU/Premise_Critique.
>
---
#### [new 039] Cross-Domain Bilingual Lexicon Induction via Pretrained Language Models
- **分类: cs.CL**

- **简介: 该论文提出跨领域双语词典诱导任务，针对专业领域数据稀缺及静态词向量偏差问题，采用预训练模型结合Code Switch策略优化词嵌入，提升领域适配性，在医疗等三领域实验中平均提升0.78分。**

- **链接: [http://arxiv.org/pdf/2505.23146v1](http://arxiv.org/pdf/2505.23146v1)**

> **作者:** Qiuyu Ding; Zhiqiang Cao; Hailong Cao; Tiejun Zhao
>
> **摘要:** Bilingual Lexicon Induction (BLI) is generally based on common domain data to obtain monolingual word embedding, and by aligning the monolingual word embeddings to obtain the cross-lingual embeddings which are used to get the word translation pairs. In this paper, we propose a new task of BLI, which is to use the monolingual corpus of the general domain and target domain to extract domain-specific bilingual dictionaries. Motivated by the ability of Pre-trained models, we propose a method to get better word embeddings that build on the recent work on BLI. This way, we introduce the Code Switch(Qin et al., 2020) firstly in the cross-domain BLI task, which can match differit is yet to be seen whether these methods are suitable for bilingual lexicon extraction in professional fields. As we can see in table 1, the classic and efficient BLI approach, Muse and Vecmap, perform much worse on the Medical dataset than on the Wiki dataset. On one hand, the specialized domain data set is relatively smaller compared to the generic domain data set generally, and specialized words have a lower frequency, which will directly affect the translation quality of bilingual dictionaries. On the other hand, static word embeddings are widely used for BLI, however, in some specific fields, the meaning of words is greatly influenced by context, in this case, using only static word embeddings may lead to greater bias. ent strategies in different contexts, making the model more suitable for this task. Experimental results show that our method can improve performances over robust BLI baselines on three specific domains by averagely improving 0.78 points.
>
---
#### [new 040] Can LLMs Deceive CLIP? Benchmarking Adversarial Compositionality of Pre-trained Multimodal Representation via Text Updates
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.SD**

- **简介: 该论文属于多模态模型鲁棒性评估任务，旨在通过文本攻击揭示CLIP等预训练模型的组合性漏洞。提出MAC基准，利用LLM生成欺骗性文本样本评估模型漏洞，并通过自训练方法（含拒绝采样微调和多样性过滤）提升攻击效果与样本多样性，验证了在图像、视频等模态中的有效性。**

- **链接: [http://arxiv.org/pdf/2505.22943v1](http://arxiv.org/pdf/2505.22943v1)**

> **作者:** Jaewoo Ahn; Heeseung Yun; Dayoon Ko; Gunhee Kim
>
> **备注:** ACL 2025 Main. Code is released at https://vision.snu.ac.kr/projects/mac
>
> **摘要:** While pre-trained multimodal representations (e.g., CLIP) have shown impressive capabilities, they exhibit significant compositional vulnerabilities leading to counterintuitive judgments. We introduce Multimodal Adversarial Compositionality (MAC), a benchmark that leverages large language models (LLMs) to generate deceptive text samples to exploit these vulnerabilities across different modalities and evaluates them through both sample-wise attack success rate and group-wise entropy-based diversity. To improve zero-shot methods, we propose a self-training approach that leverages rejection-sampling fine-tuning with diversity-promoting filtering, which enhances both attack success rate and sample diversity. Using smaller language models like Llama-3.1-8B, our approach demonstrates superior performance in revealing compositional vulnerabilities across various multimodal representations, including images, videos, and audios.
>
---
#### [new 041] Unsupervised Word-level Quality Estimation for Machine Translation Through the Lens of Annotators (Dis)agreement
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文提出无监督词级机器翻译质量评估方法，通过分析模型可解释性及不确定性量化识别错误，解决现有方法依赖昂贵标注或大模型的问题。研究评估14项指标在12种翻译任务中的表现，揭示无监督方法潜力及监督方法在标注分歧下的局限性。**

- **链接: [http://arxiv.org/pdf/2505.23183v1](http://arxiv.org/pdf/2505.23183v1)**

> **作者:** Gabriele Sarti; Vilém Zouhar; Malvina Nissim; Arianna Bisazza
>
> **备注:** Under review. Code: https://github.com/gsarti/labl/tree/main/examples/unsup_wqe Metrics: https://huggingface.co/datasets/gsarti/unsup_wqe_metrics
>
> **摘要:** Word-level quality estimation (WQE) aims to automatically identify fine-grained error spans in machine-translated outputs and has found many uses, including assisting translators during post-editing. Modern WQE techniques are often expensive, involving prompting of large language models or ad-hoc training on large amounts of human-labeled data. In this work, we investigate efficient alternatives exploiting recent advances in language model interpretability and uncertainty quantification to identify translation errors from the inner workings of translation models. In our evaluation spanning 14 metrics across 12 translation directions, we quantify the impact of human label variation on metric performance by using multiple sets of human labels. Our results highlight the untapped potential of unsupervised metrics, the shortcomings of supervised methods when faced with label uncertainty, and the brittleness of single-annotator evaluation practices.
>
---
#### [new 042] Self-Correcting Code Generation Using Small Language Models
- **分类: cs.CL**

- **简介: 该论文属于代码生成任务，旨在解决小模型自我修正能力不足的问题。提出CoCoS方法，通过在线强化学习优化多轮修正过程，设计累积奖励和细粒度奖励函数，提升小模型代码生成质量，较基线提升35.8%（MBPP）和27.7%（HumanEval）。**

- **链接: [http://arxiv.org/pdf/2505.23060v1](http://arxiv.org/pdf/2505.23060v1)**

> **作者:** Jeonghun Cho; Deokhyung Kang; Hyounghun Kim; Gary Geunbae Lee
>
> **摘要:** Self-correction has demonstrated potential in code generation by allowing language models to revise and improve their outputs through successive refinement. Recent studies have explored prompting-based strategies that incorporate verification or feedback loops using proprietary models, as well as training-based methods that leverage their strong reasoning capabilities. However, whether smaller models possess the capacity to effectively guide their outputs through self-reflection remains unexplored. Our findings reveal that smaller models struggle to exhibit reflective revision behavior across both self-correction paradigms. In response, we introduce CoCoS, an approach designed to enhance the ability of small language models for multi-turn code correction. Specifically, we propose an online reinforcement learning objective that trains the model to confidently maintain correct outputs while progressively correcting incorrect outputs as turns proceed. Our approach features an accumulated reward function that aggregates rewards across the entire trajectory and a fine-grained reward better suited to multi-turn correction scenarios. This facilitates the model in enhancing initial response quality while achieving substantial improvements through self-correction. With 1B-scale models, CoCoS achieves improvements of 35.8% on the MBPP and 27.7% on HumanEval compared to the baselines.
>
---
#### [new 043] Improving Multilingual Social Media Insights: Aspect-based Comment Analysis
- **分类: cs.CL**

- **简介: 该论文属于多语言社交媒体评论分析任务，旨在解决语言自由与观点分散导致的NLP任务挑战。提出CAT-G方法，通过多语言LLM监督微调生成评论方面术语，并用DPO优化模型预测，同时构建英、中、马来、印尼语测试集，实现跨语言性能对比。**

- **链接: [http://arxiv.org/pdf/2505.23037v1](http://arxiv.org/pdf/2505.23037v1)**

> **作者:** Longyin Zhang; Bowei Zou; Ai Ti Aw
>
> **备注:** The paper was peer-reviewed
>
> **摘要:** The inherent nature of social media posts, characterized by the freedom of language use with a disjointed array of diverse opinions and topics, poses significant challenges to downstream NLP tasks such as comment clustering, comment summarization, and social media opinion analysis. To address this, we propose a granular level of identifying and generating aspect terms from individual comments to guide model attention. Specifically, we leverage multilingual large language models with supervised fine-tuning for comment aspect term generation (CAT-G), further aligning the model's predictions with human expectations through DPO. We demonstrate the effectiveness of our method in enhancing the comprehension of social media discourse on two NLP tasks. Moreover, this paper contributes the first multilingual CAT-G test set on English, Chinese, Malay, and Bahasa Indonesian. As LLM capabilities vary among languages, this test set allows for a comparative analysis of performance across languages with varying levels of LLM proficiency.
>
---
#### [new 044] Automatic classification of stop realisation with wav2vec2.0
- **分类: cs.CL**

- **简介: 该论文使用wav2vec2.0模型实现停止音爆破音的自动分类，解决语音数据中可变语音现象标注工具不足的问题。通过在英语和日语语料库上训练模型，实现高准确率分类，验证了预训练模型在语音自动标注中的潜力，助力 Phonetic研究规模化。**

- **链接: [http://arxiv.org/pdf/2505.23688v1](http://arxiv.org/pdf/2505.23688v1)**

> **作者:** James Tanner; Morgan Sonderegger; Jane Stuart-Smith; Jeff Mielke; Tyler Kendall
>
> **备注:** Accepted for Interspeech 2025. 5 pages, 3 figures
>
> **摘要:** Modern phonetic research regularly makes use of automatic tools for the annotation of speech data, however few tools exist for the annotation of many variable phonetic phenomena. At the same time, pre-trained self-supervised models, such as wav2vec2.0, have been shown to perform well at speech classification tasks and latently encode fine-grained phonetic information. We demonstrate that wav2vec2.0 models can be trained to automatically classify stop burst presence with high accuracy in both English and Japanese, robust across both finely-curated and unprepared speech corpora. Patterns of variability in stop realisation are replicated with the automatic annotations, and closely follow those of manual annotations. These results demonstrate the potential of pre-trained speech models as tools for the automatic annotation and processing of speech corpus data, enabling researchers to `scale-up' the scope of phonetic research with relative ease.
>
---
#### [new 045] Map&Make: Schema Guided Text to Table Generation
- **分类: cs.CL**

- **简介: 该论文属于文本到表格生成任务，旨在解决现有方法在复杂信息提取和数据推理上的不足。提出Map&Make框架，通过分解文本为原子命题提取潜在schema，生成结构化表格，提升信息总结的准确性和可靠性。在Rotowire和Livesum数据集上验证，改进效果并通过消融研究分析关键因素。**

- **链接: [http://arxiv.org/pdf/2505.23174v1](http://arxiv.org/pdf/2505.23174v1)**

> **作者:** Naman Ahuja; Fenil Bardoliya; Chitta Baral; Vivek Gupta
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** Transforming dense, detailed, unstructured text into an interpretable and summarised table, also colloquially known as Text-to-Table generation, is an essential task for information retrieval. Current methods, however, miss out on how and what complex information to extract; they also lack the ability to infer data from the text. In this paper, we introduce a versatile approach, Map&Make, which "dissects" text into propositional atomic statements. This facilitates granular decomposition to extract the latent schema. The schema is then used to populate the tables that capture the qualitative nuances and the quantitative facts in the original text. Our approach is tested against two challenging datasets, Rotowire, renowned for its complex and multi-table schema, and Livesum, which demands numerical aggregation. By carefully identifying and correcting hallucination errors in Rotowire, we aim to achieve a cleaner and more reliable benchmark. We evaluate our method rigorously on a comprehensive suite of comparative and referenceless metrics. Our findings demonstrate significant improvement results across both datasets with better interpretability in Text-to-Table generation. Moreover, through detailed ablation studies and analyses, we investigate the factors contributing to superior performance and validate the practicality of our framework in structured summarization tasks.
>
---
#### [new 046] DyePack: Provably Flagging Test Set Contamination in LLMs Using Backdoors
- **分类: cs.CL**

- **简介: 论文提出DyePack框架，通过注入含随机目标的后门样本到测试数据，检测LLMs是否训练时泄露测试集。任务为识别模型测试污染，解决无需模型内部信息的检测问题，方法设计多后门并精确控制低假阳性率（如MMLU-Pro达0.000073%），实验覆盖多选与生成任务，成功检测污染模型。**

- **链接: [http://arxiv.org/pdf/2505.23001v1](http://arxiv.org/pdf/2505.23001v1)**

> **作者:** Yize Cheng; Wenxiao Wang; Mazda Moayeri; Soheil Feizi
>
> **摘要:** Open benchmarks are essential for evaluating and advancing large language models, offering reproducibility and transparency. However, their accessibility makes them likely targets of test set contamination. In this work, we introduce DyePack, a framework that leverages backdoor attacks to identify models that used benchmark test sets during training, without requiring access to the loss, logits, or any internal details of the model. Like how banks mix dye packs with their money to mark robbers, DyePack mixes backdoor samples with the test data to flag models that trained on it. We propose a principled design incorporating multiple backdoors with stochastic targets, enabling exact false positive rate (FPR) computation when flagging every model. This provably prevents false accusations while providing strong evidence for every detected case of contamination. We evaluate DyePack on five models across three datasets, covering both multiple-choice and open-ended generation tasks. For multiple-choice questions, it successfully detects all contaminated models with guaranteed FPRs as low as 0.000073% on MMLU-Pro and 0.000017% on Big-Bench-Hard using eight backdoors. For open-ended generation tasks, it generalizes well and identifies all contaminated models on Alpaca with a guaranteed false positive rate of just 0.127% using six backdoors.
>
---
#### [new 047] Data-efficient Meta-models for Evaluation of Context-based Questions and Answers in LLMs
- **分类: cs.CL**

- **简介: 该论文属于大语言模型（LLM）幻觉检测任务，针对现有监督方法依赖大量标注数据的问题，提出结合高效分类与降维技术的数据高效框架，减少训练样本需求（如250样本），在RAG基准测试中达到与强基线相当性能，助力工业部署。**

- **链接: [http://arxiv.org/pdf/2505.23299v1](http://arxiv.org/pdf/2505.23299v1)**

> **作者:** Julia Belikova; Konstantin Polev; Rauf Parchiev; Dmitry Simakov
>
> **摘要:** Large Language Models (LLMs) and Retrieval-Augmented Generation (RAG) systems are increasingly deployed in industry applications, yet their reliability remains hampered by challenges in detecting hallucinations. While supervised state-of-the-art (SOTA) methods that leverage LLM hidden states -- such as activation tracing and representation analysis -- show promise, their dependence on extensively annotated datasets limits scalability in real-world applications. This paper addresses the critical bottleneck of data annotation by investigating the feasibility of reducing training data requirements for two SOTA hallucination detection frameworks: Lookback Lens, which analyzes attention head dynamics, and probing-based approaches, which decode internal model representations. We propose a methodology combining efficient classification algorithms with dimensionality reduction techniques to minimize sample size demands while maintaining competitive performance. Evaluations on standardized question-answering RAG benchmarks show that our approach achieves performance comparable to strong proprietary LLM-based baselines with only 250 training samples. These results highlight the potential of lightweight, data-efficient paradigms for industrial deployment, particularly in annotation-constrained scenarios.
>
---
#### [new 048] ExpeTrans: LLMs Are Experiential Transfer Learners
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出ExpeTrans框架，属于迁移学习任务。针对传统方法依赖大量人工收集任务经验的问题，设计LLMs自主迁移源任务经验至新任务的方案，降低成本并提升泛化能力，实验在13个数据集验证有效性。**

- **链接: [http://arxiv.org/pdf/2505.23191v1](http://arxiv.org/pdf/2505.23191v1)**

> **作者:** Jinglong Gao; Xiao Ding; Lingxiao Zou; Bibo Cai; Bing Qin; Ting Liu
>
> **备注:** 9 pages, 12 figs/tables
>
> **摘要:** Recent studies provide large language models (LLMs) with textual task-solving experiences via prompts to improve their performance. However, previous methods rely on substantial human labor or time to gather such experiences for each task, which is impractical given the growing variety of task types in user queries to LLMs. To address this issue, we design an autonomous experience transfer framework to explore whether LLMs can mimic human cognitive intelligence to autonomously transfer experience from existing source tasks to newly encountered target tasks. This not only allows the acquisition of experience without extensive costs of previous methods, but also offers a novel path for the generalization of LLMs. Experimental results on 13 datasets demonstrate that our framework effectively improves the performance of LLMs. Furthermore, we provide a detailed analysis of each module in the framework.
>
---
#### [new 049] ER-REASON: A Benchmark Dataset for LLM-Based Clinical Reasoning in the Emergency Room
- **分类: cs.CL**

- **简介: 该论文提出ER-REASON基准数据集，旨在评估LLM在急诊室临床推理中的表现。针对现有医疗问答任务孤立、依赖人工标注的问题，构建包含3984例患者多模态病历的数据集，覆盖急诊全流程（分诊、评估、治疗等），并收录医生推理过程作为参考。实验显示LLM与临床推理存在差距，强调需改进模型的临床决策能力。**

- **链接: [http://arxiv.org/pdf/2505.22919v1](http://arxiv.org/pdf/2505.22919v1)**

> **作者:** Nikita Mehandru; Niloufar Golchini; David Bamman; Travis Zack; Melanie F. Molina; Ahmed Alaa
>
> **摘要:** Large language models (LLMs) have been extensively evaluated on medical question answering tasks based on licensing exams. However, real-world evaluations often depend on costly human annotators, and existing benchmarks tend to focus on isolated tasks that rarely capture the clinical reasoning or full workflow underlying medical decisions. In this paper, we introduce ER-Reason, a benchmark designed to evaluate LLM-based clinical reasoning and decision-making in the emergency room (ER)--a high-stakes setting where clinicians make rapid, consequential decisions across diverse patient presentations and medical specialties under time pressure. ER-Reason includes data from 3,984 patients, encompassing 25,174 de-identified longitudinal clinical notes spanning discharge summaries, progress notes, history and physical exams, consults, echocardiography reports, imaging notes, and ER provider documentation. The benchmark includes evaluation tasks that span key stages of the ER workflow: triage intake, initial assessment, treatment selection, disposition planning, and final diagnosis--each structured to reflect core clinical reasoning processes such as differential diagnosis via rule-out reasoning. We also collected 72 full physician-authored rationales explaining reasoning processes that mimic the teaching process used in residency training, and are typically absent from ER documentation. Evaluations of state-of-the-art LLMs on ER-Reason reveal a gap between LLM-generated and clinician-authored clinical reasoning for ER decisions, highlighting the need for future research to bridge this divide.
>
---
#### [new 050] Improving QA Efficiency with DistilBERT: Fine-Tuning and Inference on mobile Intel CPUs
- **分类: cs.CL**

- **简介: 该论文研究问答系统效率优化任务，针对移动Intel CPU实现实时推理与高精度需求，通过DistilBERT微调、数据增强及超参数优化，实现F1值0.6536与0.12秒/问推理速度，对比规则模型和BERT模型，为资源受限场景提供高效解决方案。**

- **链接: [http://arxiv.org/pdf/2505.22937v1](http://arxiv.org/pdf/2505.22937v1)**

> **作者:** Ngeyen Yinkfu
>
> **备注:** This paper presents an efficient transformer-based question-answering model optimized for inference on a 13th Gen Intel i7 CPU. The proposed approach balances performance and computational efficiency, making it suitable for real-time applications on resource-constrained devices. Code for this paper is available upon request via email at nyinkfu@andrew.cmu.edu
>
> **摘要:** This study presents an efficient transformer-based question-answering (QA) model optimized for deployment on a 13th Gen Intel i7-1355U CPU, using the Stanford Question Answering Dataset (SQuAD) v1.1. Leveraging exploratory data analysis, data augmentation, and fine-tuning of a DistilBERT architecture, the model achieves a validation F1 score of 0.6536 with an average inference time of 0.1208 seconds per question. Compared to a rule-based baseline (F1: 0.3124) and full BERT-based models, our approach offers a favorable trade-off between accuracy and computational efficiency. This makes it well-suited for real-time applications on resource-constrained systems. The study includes systematic evaluation of data augmentation strategies and hyperparameter configurations, providing practical insights into optimizing transformer models for CPU-based inference.
>
---
#### [new 051] Infinite-Instruct: Synthesizing Scaling Code instruction Data with Bidirectional Synthesis and Static Verification
- **分类: cs.CL**

- **简介: 该论文提出Infinite-Instruct框架，解决传统代码指令数据合成中多样性不足和逻辑薄弱的问题。通过反向构造代码转问题、知识图重构增强逻辑、静态分析过滤无效样本三阶段方法，提升LLM代码生成性能。实验显示其在主流基准上显著优于基线模型，且开源数据集。任务属代码指令数据合成与LLM训练优化。**

- **链接: [http://arxiv.org/pdf/2505.23177v1](http://arxiv.org/pdf/2505.23177v1)**

> **作者:** Wenjing Xing; Wenke Lu; Yeheng Duan; Bing Zhao; Zhenghui kang; Yaolong Wang; Kai Gao; Lei Qiao
>
> **摘要:** Traditional code instruction data synthesis methods suffer from limited diversity and poor logic. We introduce Infinite-Instruct, an automated framework for synthesizing high-quality question-answer pairs, designed to enhance the code generation capabilities of large language models (LLMs). The framework focuses on improving the internal logic of synthesized problems and the quality of synthesized code. First, "Reverse Construction" transforms code snippets into diverse programming problems. Then, through "Backfeeding Construction," keywords in programming problems are structured into a knowledge graph to reconstruct them into programming problems with stronger internal logic. Finally, a cross-lingual static code analysis pipeline filters invalid samples to ensure data quality. Experiments show that on mainstream code generation benchmarks, our fine-tuned models achieve an average performance improvement of 21.70% on 7B-parameter models and 36.95% on 32B-parameter models. Using less than one-tenth of the instruction fine-tuning data, we achieved performance comparable to the Qwen-2.5-Coder-Instruct. Infinite-Instruct provides a scalable solution for LLM training in programming. We open-source the datasets used in the experiments, including both unfiltered versions and filtered versions via static analysis. The data are available at https://github.com/xingwenjing417/Infinite-Instruct-dataset
>
---
#### [new 052] Automatic Construction of Multiple Classification Dimensions for Managing Approaches in Scientific Papers
- **分类: cs.CL**

- **简介: 该论文提出自动构建科学论文方法的多维度分类体系，解决传统方法查询耗时、组织混乱的问题。通过四层次模式识别提取方法，设计树结构相似度及聚类算法，形成五维度分类空间，提升查询相关性与效率。**

- **链接: [http://arxiv.org/pdf/2505.23252v1](http://arxiv.org/pdf/2505.23252v1)**

> **作者:** Bing Ma; Hai Zhuge
>
> **备注:** 26 pages, 9 figures
>
> **摘要:** Approaches form the foundation for conducting scientific research. Querying approaches from a vast body of scientific papers is extremely time-consuming, and without a well-organized management framework, researchers may face significant challenges in querying and utilizing relevant approaches. Constructing multiple dimensions on approaches and managing them from these dimensions can provide an efficient solution. Firstly, this paper identifies approach patterns using a top-down way, refining the patterns through four distinct linguistic levels: semantic level, discourse level, syntactic level, and lexical level. Approaches in scientific papers are extracted based on approach patterns. Additionally, five dimensions for categorizing approaches are identified using these patterns. This paper proposes using tree structure to represent step and measuring the similarity between different steps with a tree-structure-based similarity measure that focuses on syntactic-level similarities. A collection similarity measure is proposed to compute the similarity between approaches. A bottom-up clustering algorithm is proposed to construct class trees for approach components within each dimension by merging each approach component or class with its most similar approach component or class in each iteration. The class labels generated during the clustering process indicate the common semantics of the step components within the approach components in each class and are used to manage the approaches within the class. The class trees of the five dimensions collectively form a multi-dimensional approach space. The application of approach queries on the multi-dimensional approach space demonstrates that querying within this space ensures strong relevance between user queries and results and rapidly reduces search space through a class-based query mechanism.
>
---
#### [new 053] Can LLMs Reason Abstractly Over Math Word Problems Without CoT? Disentangling Abstract Formulation From Arithmetic Computation
- **分类: cs.CL**

- **简介: 论文研究数学应用题评估方法，指出传统指标混淆抽象建模与计算，通过分离评估发现LLMs准确率受限于计算而非建模，分析CoT主要提升计算，揭示模型先抽象后计算的机制，强调需分离评估以改进推理能力。**

- **链接: [http://arxiv.org/pdf/2505.23701v1](http://arxiv.org/pdf/2505.23701v1)**

> **作者:** Ziling Cheng; Meng Cao; Leila Pishdad; Yanshuai Cao; Jackie Chi Kit Cheung
>
> **摘要:** Final-answer-based metrics are commonly used for evaluating large language models (LLMs) on math word problems, often taken as proxies for reasoning ability. However, such metrics conflate two distinct sub-skills: abstract formulation (capturing mathematical relationships using expressions) and arithmetic computation (executing the calculations). Through a disentangled evaluation on GSM8K and SVAMP, we find that the final-answer accuracy of Llama-3 and Qwen2.5 (1B-32B) without CoT is overwhelmingly bottlenecked by the arithmetic computation step and not by the abstract formulation step. Contrary to the common belief, we show that CoT primarily aids in computation, with limited impact on abstract formulation. Mechanistically, we show that these two skills are composed conjunctively even in a single forward pass without any reasoning steps via an abstract-then-compute mechanism: models first capture problem abstractions, then handle computation. Causal patching confirms these abstractions are present, transferable, composable, and precede computation. These behavioural and mechanistic findings highlight the need for disentangled evaluation to accurately assess LLM reasoning and to guide future improvements.
>
---
#### [new 054] Towards a More Generalized Approach in Open Relation Extraction
- **分类: cs.CL**

- **简介: 该论文属于开放关系抽取（OpenRE）任务，旨在解决传统方法无法处理未标注数据中已知与新型关系混合分布的问题。提出MixORE框架，通过两阶段结合关系分类与聚类，联合学习两类关系。实验显示其优于基线模型，推动通用OpenRE发展。**

- **链接: [http://arxiv.org/pdf/2505.22801v1](http://arxiv.org/pdf/2505.22801v1)**

> **作者:** Qing Wang; Yuepei Li; Qiao Qiao; Kang Zhou; Qi Li
>
> **备注:** Accepted by ACL 2025 Main Conference
>
> **摘要:** Open Relation Extraction (OpenRE) seeks to identify and extract novel relational facts between named entities from unlabeled data without pre-defined relation schemas. Traditional OpenRE methods typically assume that the unlabeled data consists solely of novel relations or is pre-divided into known and novel instances. However, in real-world scenarios, novel relations are arbitrarily distributed. In this paper, we propose a generalized OpenRE setting that considers unlabeled data as a mixture of both known and novel instances. To address this, we propose MixORE, a two-phase framework that integrates relation classification and clustering to jointly learn known and novel relations. Experiments on three benchmark datasets demonstrate that MixORE consistently outperforms competitive baselines in known relation classification and novel relation clustering. Our findings contribute to the advancement of generalized OpenRE research and real-world applications.
>
---
#### [new 055] Uncovering Visual-Semantic Psycholinguistic Properties from the Distributional Structure of Text Embedding Spac
- **分类: cs.CL**

- **简介: 该论文属于心理语言学属性估计任务，旨在通过文本数据而非多模态信息，准确评估词语的图像性和具体性。提出无监督的邻域稳定性指标（NSM），通过量化语义嵌入空间中词邻域的尖锐度实现预测，实验显示其效果优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.23029v1](http://arxiv.org/pdf/2505.23029v1)**

> **作者:** Si Wu; Sebastian Bruch
>
> **备注:** Accepted for ACL 2025. This is the camera-ready version. Will be presenting in July 2025 in Vienna
>
> **摘要:** Imageability (potential of text to evoke a mental image) and concreteness (perceptibility of text) are two psycholinguistic properties that link visual and semantic spaces. It is little surprise that computational methods that estimate them do so using parallel visual and semantic spaces, such as collections of image-caption pairs or multi-modal models. In this paper, we work on the supposition that text itself in an image-caption dataset offers sufficient signals to accurately estimate these properties. We hypothesize, in particular, that the peakedness of the neighborhood of a word in the semantic embedding space reflects its degree of imageability and concreteness. We then propose an unsupervised, distribution-free measure, which we call Neighborhood Stability Measure (NSM), that quantifies the sharpness of peaks. Extensive experiments show that NSM correlates more strongly with ground-truth ratings than existing unsupervised methods, and is a strong predictor of these properties for classification. Our code and data are available on GitHub (https://github.com/Artificial-Memory-Lab/imageability).
>
---
#### [new 056] EmoBench-UA: A Benchmark Dataset for Emotion Detection in Ukrainian
- **分类: cs.CL**

- **简介: 该论文聚焦乌克兰语情感检测任务，填补其基准数据集缺失的空白。作者构建首个标注数据集EmoBench-UA，通过Toloka.ai众包确保标注质量，并测试基线模型、翻译合成数据及LLMs。研究揭示非主流语言情感分类的挑战，强调需开发乌克兰语专属模型与资源。**

- **链接: [http://arxiv.org/pdf/2505.23297v1](http://arxiv.org/pdf/2505.23297v1)**

> **作者:** Daryna Dementieva; Nikolay Babakov; Alexander Fraser
>
> **摘要:** While Ukrainian NLP has seen progress in many texts processing tasks, emotion classification remains an underexplored area with no publicly available benchmark to date. In this work, we introduce EmoBench-UA, the first annotated dataset for emotion detection in Ukrainian texts. Our annotation schema is adapted from the previous English-centric works on emotion detection (Mohammad et al., 2018; Mohammad, 2022) guidelines. The dataset was created through crowdsourcing using the Toloka.ai platform ensuring high-quality of the annotation process. Then, we evaluate a range of approaches on the collected dataset, starting from linguistic-based baselines, synthetic data translated from English, to large language models (LLMs). Our findings highlight the challenges of emotion classification in non-mainstream languages like Ukrainian and emphasize the need for further development of Ukrainian-specific models and training resources.
>
---
#### [new 057] Active Layer-Contrastive Decoding Reduces Hallucination in Large Language Model Generation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大语言模型生成中的幻觉问题，提出Active Layer-Contrastive Decoding（ActLCD）。通过强化学习策略，动态决定对比层应用时机，将解码视为序列决策，利用奖励感知分类器提升事实性。实验显示其优于现有方法，在五个基准测试中有效减少幻觉。**

- **链接: [http://arxiv.org/pdf/2505.23657v1](http://arxiv.org/pdf/2505.23657v1)**

> **作者:** Hongxiang Zhang; Hao Chen; Tianyi Zhang; Muhao Chen
>
> **摘要:** Recent decoding methods improve the factuality of large language models~(LLMs) by refining how the next token is selected during generation. These methods typically operate at the token level, leveraging internal representations to suppress superficial patterns. Nevertheless, LLMs remain prone to hallucinations, especially over longer contexts. In this paper, we propose Active Layer-Contrastive Decoding (ActLCD), a novel decoding strategy that actively decides when to apply contrasting layers during generation. By casting decoding as a sequential decision-making problem, ActLCD employs a reinforcement learning policy guided by a reward-aware classifier to optimize factuality beyond the token level. Our experiments demonstrate that ActLCD surpasses state-of-the-art methods across five benchmarks, showcasing its effectiveness in mitigating hallucinations in diverse generation scenarios.
>
---
#### [new 058] Talent or Luck? Evaluating Attribution Bias in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大语言模型（LLM）归因偏差评估任务，旨在解决模型在事件结果归因时对不同人口群体的系统性偏见问题。提出基于认知理论的框架，分析LLM将成功/失败归因于内部（如能力）或外部（如运气）因素时的群体差异，以揭示潜在公平性风险。**

- **链接: [http://arxiv.org/pdf/2505.22910v1](http://arxiv.org/pdf/2505.22910v1)**

> **作者:** Chahat Raj; Mahika Banerjee; Aylin Caliskan; Antonios Anastasopoulos; Ziwei Zhu
>
> **备注:** 18 pages
>
> **摘要:** When a student fails an exam, do we tend to blame their effort or the test's difficulty? Attribution, defined as how reasons are assigned to event outcomes, shapes perceptions, reinforces stereotypes, and influences decisions. Attribution Theory in social psychology explains how humans assign responsibility for events using implicit cognition, attributing causes to internal (e.g., effort, ability) or external (e.g., task difficulty, luck) factors. LLMs' attribution of event outcomes based on demographics carries important fairness implications. Most works exploring social biases in LLMs focus on surface-level associations or isolated stereotypes. This work proposes a cognitively grounded bias evaluation framework to identify how models' reasoning disparities channelize biases toward demographic groups.
>
---
#### [new 059] ATLAS: Learning to Optimally Memorize the Context at Test Time
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于长序列建模任务，旨在解决传统Transformer及现代循环网络在长上下文理解与外推中的局限。针对内存容量不足、在线更新片面性及固定内存表达力弱的问题，提出ATLAS模块与DeepTransformers架构，通过全局优化记忆存储，提升长序列处理能力，在语言模型、常识推理等任务中显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2505.23735v1](http://arxiv.org/pdf/2505.23735v1)**

> **作者:** Ali Behrouz; Zeman Li; Praneeth Kacham; Majid Daliri; Yuan Deng; Peilin Zhong; Meisam Razaviyayn; Vahab Mirrokni
>
> **摘要:** Transformers have been established as the most popular backbones in sequence modeling, mainly due to their effectiveness in in-context retrieval tasks and the ability to learn at scale. Their quadratic memory and time complexity, however, bound their applicability in longer sequences and so has motivated researchers to explore effective alternative architectures such as modern recurrent neural networks (a.k.a long-term recurrent memory module). Despite their recent success in diverse downstream tasks, they struggle in tasks that requires long context understanding and extrapolation to longer sequences. We observe that these shortcomings come from three disjoint aspects in their design: (1) limited memory capacity that is bounded by the architecture of memory and feature mapping of the input; (2) online nature of update, i.e., optimizing the memory only with respect to the last input; and (3) less expressive management of their fixed-size memory. To enhance all these three aspects, we present ATLAS, a long-term memory module with high capacity that learns to memorize the context by optimizing the memory based on the current and past tokens, overcoming the online nature of long-term memory models. Building on this insight, we present a new family of Transformer-like architectures, called DeepTransformers, that are strict generalizations of the original Transformer architecture. Our experimental results on language modeling, common-sense reasoning, recall-intensive, and long-context understanding tasks show that ATLAS surpasses the performance of Transformers and recent linear recurrent models. ATLAS further improves the long context performance of Titans, achieving +80\% accuracy in 10M context length of BABILong benchmark.
>
---
#### [new 060] Counting trees: A treebank-driven exploration of syntactic variation in speech and writing across languages
- **分类: cs.CL**

- **简介: 论文提出基于依存树库的句法结构比较方法，通过提取去词汇化子树分析英语和斯洛文尼亚语口语与书面语差异。任务为量化跨模态句法变异，解决系统研究语法使用变异的难题。工作包括统计结构数量、多样性及重叠，发现口语结构更少且独特，支持互动性表达需求，提出通用分析框架。**

- **链接: [http://arxiv.org/pdf/2505.22774v1](http://arxiv.org/pdf/2505.22774v1)**

> **作者:** Kaja Dobrovoljc
>
> **摘要:** This paper presents a novel treebank-driven approach to comparing syntactic structures in speech and writing using dependency-parsed corpora. Adopting a fully inductive, bottom-up method, we define syntactic structures as delexicalized dependency (sub)trees and extract them from spoken and written Universal Dependencies (UD) treebanks in two syntactically distinct languages, English and Slovenian. For each corpus, we analyze the size, diversity, and distribution of syntactic inventories, their overlap across modalities, and the structures most characteristic of speech. Results show that, across both languages, spoken corpora contain fewer and less diverse syntactic structures than their written counterparts, with consistent cross-linguistic preferences for certain structural types across modalities. Strikingly, the overlap between spoken and written syntactic inventories is very limited: most structures attested in speech do not occur in writing, pointing to modality-specific preferences in syntactic organization that reflect the distinct demands of real-time interaction and elaborated writing. This contrast is further supported by a keyness analysis of the most frequent speech-specific structures, which highlights patterns associated with interactivity, context-grounding, and economy of expression. We argue that this scalable, language-independent framework offers a useful general method for systematically studying syntactic variation across corpora, laying the groundwork for more comprehensive data-driven theories of grammar in use.
>
---
#### [new 061] Context Robust Knowledge Editing for Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型知识编辑任务，解决现有方法在存在前置上下文时编辑效果差的问题。提出CHED基准评估上下文鲁棒性，并开发CoRE方法通过最小化隐藏状态的上下文差异提升编辑成功率，同时分析用户输入与模型回复作为上下文的差异化影响。**

- **链接: [http://arxiv.org/pdf/2505.23026v1](http://arxiv.org/pdf/2505.23026v1)**

> **作者:** Haewon Park; Gyubin Choi; Minjun Kim; Yohan Jo
>
> **备注:** ACL 2025 Findings. Our code and datasets are available at (https://github.com/holi-lab/CoRE)
>
> **摘要:** Knowledge editing (KE) methods offer an efficient way to modify knowledge in large language models. Current KE evaluations typically assess editing success by considering only the edited knowledge without any preceding contexts. In real-world applications, however, preceding contexts often trigger the retrieval of the original knowledge and undermine the intended edit. To address this issue, we develop CHED -- a benchmark designed to evaluate the context robustness of KE methods. Evaluations on CHED show that they often fail when preceding contexts are present. To mitigate this shortcoming, we introduce CoRE, a KE method designed to strengthen context robustness by minimizing context-sensitive variance in hidden states of the model for edited knowledge. This method not only improves the editing success rate in situations where a preceding context is present but also preserves the overall capabilities of the model. We provide an in-depth analysis of the differing impacts of preceding contexts when introduced as user utterances versus assistant responses, and we dissect attention-score patterns to assess how specific tokens influence editing success.
>
---
#### [new 062] What Has Been Lost with Synthetic Evaluation?
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究用LLM生成评估基准的局限性。针对数据生成成本高、易走捷径的问题，通过生成CondaQA和DROP的变体数据，对比人工数据集，发现LLM生成数据虽符合规范且成本低，但难度不足。揭示合成数据在挑战性上的缺陷，呼吁重新评估此方法。**

- **链接: [http://arxiv.org/pdf/2505.22830v1](http://arxiv.org/pdf/2505.22830v1)**

> **作者:** Alexander Gill; Abhilasha Ravichander; Ana Marasović
>
> **备注:** 9 pages main, 5 pages reference, 24 pages appendix
>
> **摘要:** Large language models (LLMs) are increasingly used for data generation. However, creating evaluation benchmarks raises the bar for this emerging paradigm. Benchmarks must target specific phenomena, penalize exploiting shortcuts, and be challenging. Through two case studies, we investigate whether LLMs can meet these demands by generating reasoning over-text benchmarks and comparing them to those created through careful crowdsourcing. Specifically, we evaluate both the validity and difficulty of LLM-generated versions of two high-quality reading comprehension datasets: CondaQA, which evaluates reasoning about negation, and DROP, which targets reasoning about quantities. We find that prompting LLMs can produce variants of these datasets that are often valid according to the annotation guidelines, at a fraction of the cost of the original crowdsourcing effort. However, we show that they are less challenging for LLMs than their human-authored counterparts. This finding sheds light on what may have been lost by generating evaluation data with LLMs, and calls for critically reassessing the immediate use of this increasingly prevalent approach to benchmark creation.
>
---
#### [new 063] Document-Level Text Generation with Minimum Bayes Risk Decoding using Optimal Transport
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文档级文本生成任务，旨在解决传统MBR解码因依赖句子级utility函数导致文档生成效果不佳的问题。提出MBR-OT方法，利用Wasserstein距离改进文档utility计算，实验显示其在机器翻译、文本简化和图像描述任务中优于标准MBR。**

- **链接: [http://arxiv.org/pdf/2505.23078v1](http://arxiv.org/pdf/2505.23078v1)**

> **作者:** Yuu Jinnai
>
> **备注:** ACL 2025
>
> **摘要:** Document-level text generation tasks are known to be more difficult than sentence-level text generation tasks as they require the understanding of longer context to generate high-quality texts. In this paper, we investigate the adaption of Minimum Bayes Risk (MBR) decoding for document-level text generation tasks. MBR decoding makes use of a utility function to estimate the output with the highest expected utility from a set of candidate outputs. Although MBR decoding is shown to be effective in a wide range of sentence-level text generation tasks, its performance on document-level text generation tasks is limited as many of the utility functions are designed for evaluating the utility of sentences. To this end, we propose MBR-OT, a variant of MBR decoding using Wasserstein distance to compute the utility of a document using a sentence-level utility function. The experimental result shows that the performance of MBR-OT outperforms that of the standard MBR in document-level machine translation, text simplification, and dense image captioning tasks. Our code is available at https://github.com/jinnaiyuu/mbr-optimal-transport
>
---
#### [new 064] Spoken Language Modeling with Duration-Penalized Self-Supervised Units
- **分类: cs.CL; eess.AS**

- **简介: 该论文属于语音语言模型优化任务，旨在探索自监督语音单元的码本大小与时长粗细对SLM性能的影响。通过DPDP方法分析不同语言层级，发现粗粒度单元在句子重生成及低码率词汇/句法任务中更优，但电话与单词层级无显著优势，证明DPDP是获取有效粗单元的高效手段。**

- **链接: [http://arxiv.org/pdf/2505.23494v1](http://arxiv.org/pdf/2505.23494v1)**

> **作者:** Nicol Visser; Herman Kamper
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Spoken language models (SLMs) operate on acoustic units obtained by discretizing self-supervised speech representations. Although the characteristics of these units directly affect performance, the interaction between codebook size and unit coarseness (i.e., duration) remains unexplored. We investigate SLM performance as we vary codebook size and unit coarseness using the simple duration-penalized dynamic programming (DPDP) method. New analyses are performed across different linguistic levels. At the phone and word levels, coarseness provides little benefit, as long as the codebook size is chosen appropriately. However, when producing whole sentences in a resynthesis task, SLMs perform better with coarser units. In lexical and syntactic language modeling tasks, coarser units also give higher accuracies at lower bitrates. We therefore show that coarser units aren't always better, but that DPDP is a simple and efficient way to obtain coarser units for the tasks where they are beneficial.
>
---
#### [new 065] How Does Response Length Affect Long-Form Factuality
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文研究响应长度对长文本事实性的影响，针对LLMs生成中事实错误问题，提出自动双层评估框架，通过实验发现长响应事实精度更低，主要因知识耗尽而非错误传播或上下文长度。**

- **链接: [http://arxiv.org/pdf/2505.23295v1](http://arxiv.org/pdf/2505.23295v1)**

> **作者:** James Xu Zhao; Jimmy Z. J. Liu; Bryan Hooi; See-Kiong Ng
>
> **备注:** ACL 2025 Findings. 24 pages, 10 figures, 18 tables. Code available at https://github.com/XuZhao0/length-bias-factuality
>
> **摘要:** Large language models (LLMs) are widely used for long-form text generation. However, factual errors in the responses would undermine their reliability. Despite growing attention to LLM factuality, the effect of response length on factuality remains underexplored. In this work, we systematically investigate this relationship by first introducing an automatic and bi-level long-form factuality evaluation framework, which achieves high agreement with human annotations while being cost-effective. Using this framework, we conduct controlled experiments and find that longer responses exhibit lower factual precision, confirming the presence of length bias. To explain this phenomenon, we empirically examine three hypotheses: error propagation, long context, and facts exhaustion. Our results reveal that facts exhaustion, where the model gradually exhausts more reliable knowledge, is the primary cause of factual degradation, rather than the other two hypotheses.
>
---
#### [new 066] Neither Stochastic Parroting nor AGI: LLMs Solve Tasks through Context-Directed Extrapolation from Training Data Priors
- **分类: cs.CL**

- **简介: 该立场论文反驳LLMs仅为"随机鹦鹉"或具威胁性AGI的极端观点，提出其通过"上下文引导外推"机制利用训练数据先验解决问题。认为其能力可预测可控，非无限扩展，指明研究应聚焦该机制与训练数据的交互。**

- **链接: [http://arxiv.org/pdf/2505.23323v1](http://arxiv.org/pdf/2505.23323v1)**

> **作者:** Harish Tayyar Madabushi; Melissa Torgbi; Claire Bonial
>
> **摘要:** In this position paper we raise critical awareness of a realistic view of LLM capabilities that eschews extreme alternative views that LLMs are either "stochastic parrots" or in possession of "emergent" advanced reasoning capabilities, which, due to their unpredictable emergence, constitute an existential threat. Our middle-ground view is that LLMs extrapolate from priors from their training data, and that a mechanism akin to in-context learning enables the targeting of the appropriate information from which to extrapolate. We call this "context-directed extrapolation." Under this view, substantiated though existing literature, while reasoning capabilities go well beyond stochastic parroting, such capabilities are predictable, controllable, not indicative of advanced reasoning akin to high-level cognitive capabilities in humans, and not infinitely scalable with additional training. As a result, fears of uncontrollable emergence of agency are allayed, while research advances are appropriately refocused on the processes of context-directed extrapolation and how this interacts with training data to produce valuable capabilities in LLMs. Future work can therefore explore alternative augmenting techniques that do not rely on inherent advanced reasoning in LLMs.
>
---
#### [new 067] Can Large Language Models Match the Conclusions of Systematic Reviews?
- **分类: cs.CL**

- **简介: 该论文评估大型语言模型（LLMs）生成系统综述的能力，探究其能否匹配临床专家结论。通过构建MedEvidence基准（100项系统综述及对应研究），测试24种LLM，发现推理、模型规模未显著提升性能，知识微调降低准确率，模型存在过度自信、缺乏科学怀疑等缺陷，表明LLMs尚无法可靠替代专家综述。**

- **链接: [http://arxiv.org/pdf/2505.22787v1](http://arxiv.org/pdf/2505.22787v1)**

> **作者:** Christopher Polzak; Alejandro Lozano; Min Woo Sun; James Burgess; Yuhui Zhang; Kevin Wu; Serena Yeung-Levy
>
> **摘要:** Systematic reviews (SR), in which experts summarize and analyze evidence across individual studies to provide insights on a specialized topic, are a cornerstone for evidence-based clinical decision-making, research, and policy. Given the exponential growth of scientific articles, there is growing interest in using large language models (LLMs) to automate SR generation. However, the ability of LLMs to critically assess evidence and reason across multiple documents to provide recommendations at the same proficiency as domain experts remains poorly characterized. We therefore ask: Can LLMs match the conclusions of systematic reviews written by clinical experts when given access to the same studies? To explore this question, we present MedEvidence, a benchmark pairing findings from 100 SRs with the studies they are based on. We benchmark 24 LLMs on MedEvidence, including reasoning, non-reasoning, medical specialist, and models across varying sizes (from 7B-700B). Through our systematic evaluation, we find that reasoning does not necessarily improve performance, larger models do not consistently yield greater gains, and knowledge-based fine-tuning degrades accuracy on MedEvidence. Instead, most models exhibit similar behavior: performance tends to degrade as token length increases, their responses show overconfidence, and, contrary to human experts, all models show a lack of scientific skepticism toward low-quality findings. These results suggest that more work is still required before LLMs can reliably match the observations from expert-conducted SRs, even though these systems are already deployed and being used by clinicians. We release our codebase and benchmark to the broader research community to further investigate LLM-based SR systems.
>
---
#### [new 068] Probability-Consistent Preference Optimization for Enhanced LLM Reasoning
- **分类: cs.CL**

- **简介: 该论文提出概率一致偏好优化（PCPO）框架，改进大语言模型的数学推理能力。针对现有方法仅关注答案正确性而忽视逻辑连贯性的问题，PCPO通过表面正确性和 token 级概率一致性双指标优化偏好选择，实验显示其优于传统方法。**

- **链接: [http://arxiv.org/pdf/2505.23540v1](http://arxiv.org/pdf/2505.23540v1)**

> **作者:** Yunqiao Yang; Houxing Ren; Zimu Lu; Ke Wang; Weikang Shi; Aojun Zhou; Junting Pan; Mingjie Zhan; Hongsheng Li
>
> **备注:** 14 pages, to be published in ACL 2025 findings
>
> **摘要:** Recent advances in preference optimization have demonstrated significant potential for improving mathematical reasoning capabilities in large language models (LLMs). While current approaches leverage high-quality pairwise preference data through outcome-based criteria like answer correctness or consistency, they fundamentally neglect the internal logical coherence of responses. To overcome this, we propose Probability-Consistent Preference Optimization (PCPO), a novel framework that establishes dual quantitative metrics for preference selection: (1) surface-level answer correctness and (2) intrinsic token-level probability consistency across responses. Extensive experiments show that our PCPO consistently outperforms existing outcome-only criterion approaches across a diverse range of LLMs and benchmarks. Our code is publicly available at https://github.com/YunqiaoYang/PCPO.
>
---
#### [new 069] Pre-Training Curriculum for Multi-Token Prediction in Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型预训练任务，针对小模型难以有效利用多标记预测（MTP）的问题，提出两种课程学习策略：前向课程（从NTP逐步过渡到MTP）和反向课程。实验表明前向课程提升小模型下游任务表现及生成质量并保留自推测解码优势，反向课程虽提升性能但无此优势。**

- **链接: [http://arxiv.org/pdf/2505.22757v1](http://arxiv.org/pdf/2505.22757v1)**

> **作者:** Ansar Aynetdinov; Alan Akbik
>
> **备注:** Accepted to ACL 2025 (Main)
>
> **摘要:** Multi-token prediction (MTP) is a recently proposed pre-training objective for language models. Rather than predicting only the next token (NTP), MTP predicts the next $k$ tokens at each prediction step, using multiple prediction heads. MTP has shown promise in improving downstream performance, inference speed, and training efficiency, particularly for large models. However, prior work has shown that smaller language models (SLMs) struggle with the MTP objective. To address this, we propose a curriculum learning strategy for MTP training, exploring two variants: a forward curriculum, which gradually increases the complexity of the pre-training objective from NTP to MTP, and a reverse curriculum, which does the opposite. Our experiments show that the forward curriculum enables SLMs to better leverage the MTP objective during pre-training, improving downstream NTP performance and generative output quality, while retaining the benefits of self-speculative decoding. The reverse curriculum achieves stronger NTP performance and output quality, but fails to provide any self-speculative decoding benefits.
>
---
#### [new 070] First Steps Towards Overhearing LLM Agents: A Case Study With Dungeons & Dragons Gameplay
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文提出"监听代理"新范式，探索LLM通过被动监听人类对话提供辅助。以D&D游戏为案例，用多模态模型分析DM语音，完成背景任务并提供建议，经人类评估验证其有效性，开源工具支持后续研究。**

- **链接: [http://arxiv.org/pdf/2505.22809v1](http://arxiv.org/pdf/2505.22809v1)**

> **作者:** Andrew Zhu; Evan Osgood; Chris Callison-Burch
>
> **备注:** 8 pages, 5 figures. In submission at EMNLP 2025
>
> **摘要:** Much work has been done on conversational LLM agents which directly assist human users with tasks. We present an alternative paradigm for interacting with LLM agents, which we call "overhearing agents". These overhearing agents do not actively participate in conversation -- instead, they "listen in" on human-to-human conversations and perform background tasks or provide suggestions to assist the user. In this work, we explore the overhearing agents paradigm through the lens of Dungeons & Dragons gameplay. We present an in-depth study using large multimodal audio-language models as overhearing agents to assist a Dungeon Master. We perform a human evaluation to examine the helpfulness of such agents and find that some large audio-language models have the emergent ability to perform overhearing agent tasks using implicit audio cues. Finally, we release Python libraries and our project code to support further research into the overhearing agents paradigm at https://github.com/zhudotexe/overhearing_agents.
>
---
#### [new 071] LiTEx: A Linguistic Taxonomy of Explanations for Understanding Within-Label Variation in Natural Language Inference
- **分类: cs.CL**

- **简介: 该论文属于自然语言推理（NLI）领域，旨在解决标注者对同一前提-假设对赋予相同标签但推理逻辑差异的问题。提出LiTEx分类系统，通过标注e-SNLI数据集验证其可靠性，分析解释与标签/高亮的关联，并证明其指导的解释生成更贴近人类推理。任务：分析标签内变异；问题：同标签不同理由；工作：构建分类、验证及生成应用。**

- **链接: [http://arxiv.org/pdf/2505.22848v1](http://arxiv.org/pdf/2505.22848v1)**

> **作者:** Pingjun Hong; Beiduo Chen; Siyao Peng; Marie-Catherine de Marneffe; Barbara Plank
>
> **备注:** 21 pages, 6 figures
>
> **摘要:** There is increasing evidence of Human Label Variation (HLV) in Natural Language Inference (NLI), where annotators assign different labels to the same premise-hypothesis pair. However, within-label variation--cases where annotators agree on the same label but provide divergent reasoning--poses an additional and mostly overlooked challenge. Several NLI datasets contain highlighted words in the NLI item as explanations, but the same spans on the NLI item can be highlighted for different reasons, as evidenced by free-text explanations, which offer a window into annotators' reasoning. To systematically understand this problem and gain insight into the rationales behind NLI labels, we introduce LITEX, a linguistically-informed taxonomy for categorizing free-text explanations. Using this taxonomy, we annotate a subset of the e-SNLI dataset, validate the taxonomy's reliability, and analyze how it aligns with NLI labels, highlights, and explanations. We further assess the taxonomy's usefulness in explanation generation, demonstrating that conditioning generation on LITEX yields explanations that are linguistically closer to human explanations than those generated using only labels or highlights. Our approach thus not only captures within-label variation but also shows how taxonomy-guided generation for reasoning can bridge the gap between human and model explanations more effectively than existing strategies.
>
---
#### [new 072] Enhancing Marker Scoring Accuracy through Ordinal Confidence Modelling in Educational Assessments
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属自动化作文评分（AES）中的置信度建模任务，旨在解决仅在高可靠性时发布分数的伦理问题。通过将置信度估计转化为n元分类（分箱法）并提出Kernel加权顺序交叉熵损失函数，利用CEFR等级的顺序性提升准确性。最佳模型使47%分数达100%CEFR一致，优于基线AES的92%。**

- **链接: [http://arxiv.org/pdf/2505.23315v1](http://arxiv.org/pdf/2505.23315v1)**

> **作者:** Abhirup Chakravarty; Mark Brenchley; Trevor Breakspear; Ian Lewin; Yan Huang
>
> **备注:** This is the preprint version of our paper accepted to ACL 2025 (Industry Track). The DOI will be added once available
>
> **摘要:** A key ethical challenge in Automated Essay Scoring (AES) is ensuring that scores are only released when they meet high reliability standards. Confidence modelling addresses this by assigning a reliability estimate measure, in the form of a confidence score, to each automated score. In this study, we frame confidence estimation as a classification task: predicting whether an AES-generated score correctly places a candidate in the appropriate CEFR level. While this is a binary decision, we leverage the inherent granularity of the scoring domain in two ways. First, we reformulate the task as an n-ary classification problem using score binning. Second, we introduce a set of novel Kernel Weighted Ordinal Categorical Cross Entropy (KWOCCE) loss functions that incorporate the ordinal structure of CEFR labels. Our best-performing model achieves an F1 score of 0.97, and enables the system to release 47% of scores with 100% CEFR agreement and 99% with at least 95% CEFR agreement -compared to approximately 92% (approx.) CEFR agreement from the standalone AES model where we release all AM predicted scores.
>
---
#### [new 073] ChartMind: A Comprehensive Benchmark for Complex Real-world Multimodal Chart Question Answering
- **分类: cs.CL**

- **简介: 该论文属于复杂多模态图表问答（CQA）任务，针对现有评估方法忽视真实场景需求的问题，提出ChartMind基准（覆盖七类任务、多语言、多种图表格式）及ChartLLM框架，通过优化上下文理解提升模型推理精度，显著超越传统方法。**

- **链接: [http://arxiv.org/pdf/2505.23242v1](http://arxiv.org/pdf/2505.23242v1)**

> **作者:** Jingxuan Wei; Nan Xu; Junnan Zhu; Yanni Hao; Gaowei Wu; Bihui Yu; Lei Wang
>
> **摘要:** Chart question answering (CQA) has become a critical multimodal task for evaluating the reasoning capabilities of vision-language models. While early approaches have shown promising performance by focusing on visual features or leveraging large-scale pre-training, most existing evaluations rely on rigid output formats and objective metrics, thus ignoring the complex, real-world demands of practical chart analysis. In this paper, we introduce ChartMind, a new benchmark designed for complex CQA tasks in real-world settings. ChartMind covers seven task categories, incorporates multilingual contexts, supports open-domain textual outputs, and accommodates diverse chart formats, bridging the gap between real-world applications and traditional academic benchmarks. Furthermore, we propose a context-aware yet model-agnostic framework, ChartLLM, that focuses on extracting key contextual elements, reducing noise, and enhancing the reasoning accuracy of multimodal large language models. Extensive evaluations on ChartMind and three representative public benchmarks with 14 mainstream multimodal models show our framework significantly outperforms the previous three common CQA paradigms: instruction-following, OCR-enhanced, and chain-of-thought, highlighting the importance of flexible chart understanding for real-world CQA. These findings suggest new directions for developing more robust chart reasoning in future research.
>
---
#### [new 074] Characterizing the Expressivity of Transformer Language Models
- **分类: cs.CL**

- **简介: 该论文分析Transformer表达能力，解决现有理论与实际模型差异问题。通过精确刻画固定精度、软注意力和掩码的Transformer，证明其等价于含单一过去时态算子的线性时序逻辑，并建立跨理论的统一框架，实验验证模型在能力范围内可完美泛化。**

- **链接: [http://arxiv.org/pdf/2505.23623v1](http://arxiv.org/pdf/2505.23623v1)**

> **作者:** Jiaoda Li; Ryan Cotterell
>
> **摘要:** Transformer-based language models (LMs) have achieved widespread empirical success, but their theoretical expressive power remains only partially understood. Prior work often relies on idealized models with assumptions -- such as arbitrary numerical precision and hard attention -- that diverge from real-world transformers. In this work, we provide an exact characterization of fixed-precision transformers with strict future masking and soft attention, an idealization that more closely mirrors practical implementations. We show that these models are precisely as expressive as a specific fragment of linear temporal logic that includes only a single temporal operator: the past operator. We further relate this logic to established classes in formal language theory, automata theory, and algebra, yielding a rich and unified theoretical framework for understanding transformer expressivity. Finally, we present empirical results that align closely with our theory: transformers trained on languages within their theoretical capacity generalize perfectly over lengths, while they consistently fail to generalize on languages beyond it.
>
---
#### [new 075] SNS-Bench-VL: Benchmarking Multimodal Large Language Models in Social Networking Services
- **分类: cs.CL**

- **简介: 该论文提出SNS-Bench-VL基准，评估多模态大语言模型在社交平台的性能。针对现有评测缺乏多模态支持的问题，设计涵盖8类任务（如内容理解、推荐等）的4001个图文问题，测试25个模型，揭示社交情境理解的挑战，推动多模态智能研究。**

- **链接: [http://arxiv.org/pdf/2505.23065v1](http://arxiv.org/pdf/2505.23065v1)**

> **作者:** Hongcheng Guo; Zheyong Xie; Shaosheng Cao; Boyang Wang; Weiting Liu; Anjie Le; Lei Li; Zhoujun Li
>
> **摘要:** With the increasing integration of visual and textual content in Social Networking Services (SNS), evaluating the multimodal capabilities of Large Language Models (LLMs) is crucial for enhancing user experience, content understanding, and platform intelligence. Existing benchmarks primarily focus on text-centric tasks, lacking coverage of the multimodal contexts prevalent in modern SNS ecosystems. In this paper, we introduce SNS-Bench-VL, a comprehensive multimodal benchmark designed to assess the performance of Vision-Language LLMs in real-world social media scenarios. SNS-Bench-VL incorporates images and text across 8 multimodal tasks, including note comprehension, user engagement analysis, information retrieval, and personalized recommendation. It comprises 4,001 carefully curated multimodal question-answer pairs, covering single-choice, multiple-choice, and open-ended tasks. We evaluate over 25 state-of-the-art multimodal LLMs, analyzing their performance across tasks. Our findings highlight persistent challenges in multimodal social context comprehension. We hope SNS-Bench-VL will inspire future research towards robust, context-aware, and human-aligned multimodal intelligence for next-generation social networking services.
>
---
#### [new 076] AutoSchemaKG: Autonomous Knowledge Graph Construction through Dynamic Schema Induction from Web-Scale Corpora
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AutoSchemaKG框架，实现无需预定义模式的自主知识图构建。针对传统方法依赖人工schema的局限，其利用大语言模型同步抽取知识三元组并动态诱导schema，处理5000万文档生成超大规模知识图（90亿节点/59亿边），提升多跳问答准确率，且自动schema与人工设计达95%语义一致，验证动态schema对LLM事实性的增强效果。**

- **链接: [http://arxiv.org/pdf/2505.23628v1](http://arxiv.org/pdf/2505.23628v1)**

> **作者:** Jiaxin Bai; Wei Fan; Qi Hu; Qing Zong; Chunyang Li; Hong Ting Tsang; Hongyu Luo; Yauwai Yim; Haoyu Huang; Xiao Zhou; Feng Qin; Tianshi Zheng; Xi Peng; Xin Yao; Huiwen Yang; Leijie Wu; Yi Ji; Gong Zhang; Renhai Chen; Yangqiu Song
>
> **备注:** 9 pages, preprint, code: https://github.com/HKUST-KnowComp/AutoSchemaKG
>
> **摘要:** We present AutoSchemaKG, a framework for fully autonomous knowledge graph construction that eliminates the need for predefined schemas. Our system leverages large language models to simultaneously extract knowledge triples and induce comprehensive schemas directly from text, modeling both entities and events while employing conceptualization to organize instances into semantic categories. Processing over 50 million documents, we construct ATLAS (Automated Triple Linking And Schema induction), a family of knowledge graphs with 900+ million nodes and 5.9 billion edges. This approach outperforms state-of-the-art baselines on multi-hop QA tasks and enhances LLM factuality. Notably, our schema induction achieves 95\% semantic alignment with human-crafted schemas with zero manual intervention, demonstrating that billion-scale knowledge graphs with dynamically induced schemas can effectively complement parametric knowledge in large language models.
>
---
#### [new 077] ScEdit: Script-based Assessment of Knowledge Editing
- **分类: cs.CL**

- **简介: 该论文提出ScEdit基准，用于评估知识编辑技术在真实场景中的表现。针对现有评估方法任务简单、脱离应用的问题，其构建包含反事实和时序编辑的脚本数据集，结合词符与文本级评估，扩展至"如何操作"型任务，揭示现有方法性能下降，凸显挑战。**

- **链接: [http://arxiv.org/pdf/2505.23291v1](http://arxiv.org/pdf/2505.23291v1)**

> **作者:** Xinye Li; Zunwen Zheng; Qian Zhang; Dekai Zhuang; Jiabao Kang; Liyan Xu; Qingbin Liu; Xi Chen; Zhiying Tu; Dianhui Chu; Dianbo Sui
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Knowledge Editing (KE) has gained increasing attention, yet current KE tasks remain relatively simple. Under current evaluation frameworks, many editing methods achieve exceptionally high scores, sometimes nearing perfection. However, few studies integrate KE into real-world application scenarios (e.g., recent interest in LLM-as-agent). To support our analysis, we introduce a novel script-based benchmark -- ScEdit (Script-based Knowledge Editing Benchmark) -- which encompasses both counterfactual and temporal edits. We integrate token-level and text-level evaluation methods, comprehensively analyzing existing KE techniques. The benchmark extends traditional fact-based ("What"-type question) evaluation to action-based ("How"-type question) evaluation. We observe that all KE methods exhibit a drop in performance on established metrics and face challenges on text-level metrics, indicating a challenging task. Our benchmark is available at https://github.com/asdfo123/ScEdit.
>
---
#### [new 078] Translation in the Wild
- **分类: cs.CL**

- **简介: 该论文探讨大型语言模型（LLMs）在未专门训练翻译任务下仍具备跨语言翻译能力的机制。任务为解析其翻译能力来源，解决LLMs如何通过预训练数据（如"意外双语"内容）及模型结构实现翻译的问题。提出翻译能力源于两种不同预训练数据类型的假设，并讨论验证此假设及重新定义深度学习时代翻译的可能。**

- **链接: [http://arxiv.org/pdf/2505.23548v1](http://arxiv.org/pdf/2505.23548v1)**

> **作者:** Yuri Balashov
>
> **备注:** 4 figures
>
> **摘要:** Large Language Models (LLMs) excel in translation among other things, demonstrating competitive performance for many language pairs in zero- and few-shot settings. But unlike dedicated neural machine translation models, LLMs are not trained on any translation-related objective. What explains their remarkable translation abilities? Are these abilities grounded in "incidental bilingualism" (Briakou et al. 2023) in training data? Does instruction tuning contribute to it? Are LLMs capable of aligning and leveraging semantically identical or similar monolingual contents from different corners of the internet that are unlikely to fit in a single context window? I offer some reflections on this topic, informed by recent studies and growing user experience. My working hypothesis is that LLMs' translation abilities originate in two different types of pre-training data that may be internalized by the models in different ways. I discuss the prospects for testing the "duality" hypothesis empirically and its implications for reconceptualizing translation, human and machine, in the age of deep learning.
>
---
#### [new 079] GeNRe: A French Gender-Neutral Rewriting System Using Collective Nouns
- **分类: cs.CL**

- **简介: 该论文提出GeNRe系统，解决法语中阳性泛指导致的性别偏见问题。作为首个使用集体名词的法语性别中立改写系统，其结合规则引擎、微调语言模型及Claude 3 Opus指令模型，通过生成中性表达减少文本偏见，推动法语NLP的公平性研究。**

- **链接: [http://arxiv.org/pdf/2505.23630v1](http://arxiv.org/pdf/2505.23630v1)**

> **作者:** Enzo Doyen; Amalia Todirascu
>
> **备注:** Accepted to ACL 2025 Findings; 9 pages, 2 figures
>
> **摘要:** A significant portion of the textual data used in the field of Natural Language Processing (NLP) exhibits gender biases, particularly due to the use of masculine generics (masculine words that are supposed to refer to mixed groups of men and women), which can perpetuate and amplify stereotypes. Gender rewriting, an NLP task that involves automatically detecting and replacing gendered forms with neutral or opposite forms (e.g., from masculine to feminine), can be employed to mitigate these biases. While such systems have been developed in a number of languages (English, Arabic, Portuguese, German, French), automatic use of gender neutralization techniques (as opposed to inclusive or gender-switching techniques) has only been studied for English. This paper presents GeNRe, the very first French gender-neutral rewriting system using collective nouns, which are gender-fixed in French. We introduce a rule-based system (RBS) tailored for the French language alongside two fine-tuned language models trained on data generated by our RBS. We also explore the use of instruct-based models to enhance the performance of our other systems and find that Claude 3 Opus combined with our dictionary achieves results close to our RBS. Through this contribution, we hope to promote the advancement of gender bias mitigation techniques in NLP for French.
>
---
#### [new 080] StrucSum: Graph-Structured Reasoning for Long Document Extractive Summarization with LLMs
- **分类: cs.CL**

- **简介: 该论文提出StrucSum，一种无训练的图结构提示框架，解决LLMs在长文档抽取式摘要中结构建模与关键信息识别不足的问题。通过邻近感知提示、中心性评估及遮罩策略注入结构信号，提升零样本摘要的准确性和事实一致性，实验显示显著优于基线方法。**

- **链接: [http://arxiv.org/pdf/2505.22950v1](http://arxiv.org/pdf/2505.22950v1)**

> **作者:** Haohan Yuan; Sukhwa Hong; Haopeng Zhang
>
> **摘要:** Large language models (LLMs) have shown strong performance in zero-shot summarization, but often struggle to model document structure and identify salient information in long texts. In this work, we introduce StrucSum, a training-free prompting framework that enhances LLM reasoning through sentence-level graph structures. StrucSum injects structural signals into prompts via three targeted strategies: Neighbor-Aware Prompting (NAP) for local context, Centrality-Aware Prompting (CAP) for importance estimation, and Centrality-Guided Masking (CGM) for efficient input reduction. Experiments on ArXiv, PubMed, and Multi-News demonstrate that StrucSum consistently improves both summary quality and factual consistency over unsupervised baselines and vanilla prompting. Notably, on ArXiv, it boosts FactCC and SummaC by 19.2 and 9.7 points, indicating stronger alignment between summaries and source content. These findings suggest that structure-aware prompting is a simple yet effective approach for zero-shot extractive summarization with LLMs, without any training or task-specific tuning.
>
---
#### [new 081] Evaluating AI capabilities in detecting conspiracy theories on YouTube
- **分类: cs.CL; cs.CY; cs.SI**

- **简介: 该论文评估AI检测YouTube阴谋论的能力，任务为内容识别。通过比较文本LLM、多模态模型与RoBERTa基线，在零样本设置下测试其检测效果，发现文本模型召回率高但精度低，多模态无显著优势，RoBERTa表现接近LLM。强调需提升检测精准度。**

- **链接: [http://arxiv.org/pdf/2505.23570v1](http://arxiv.org/pdf/2505.23570v1)**

> **作者:** Leonardo La Rocca; Francesco Corso; Francesco Pierri
>
> **备注:** Submitted for review to OSNEM Special Issue of April 2025
>
> **摘要:** As a leading online platform with a vast global audience, YouTube's extensive reach also makes it susceptible to hosting harmful content, including disinformation and conspiracy theories. This study explores the use of open-weight Large Language Models (LLMs), both text-only and multimodal, for identifying conspiracy theory videos shared on YouTube. Leveraging a labeled dataset of thousands of videos, we evaluate a variety of LLMs in a zero-shot setting and compare their performance to a fine-tuned RoBERTa baseline. Results show that text-based LLMs achieve high recall but lower precision, leading to increased false positives. Multimodal models lag behind their text-only counterparts, indicating limited benefits from visual data integration. To assess real-world applicability, we evaluate the most accurate models on an unlabeled dataset, finding that RoBERTa achieves performance close to LLMs with a larger number of parameters. Our work highlights the strengths and limitations of current LLM-based approaches for online harmful content detection, emphasizing the need for more precise and robust systems.
>
---
#### [new 082] Revisiting Overthinking in Long Chain-of-Thought from the Perspective of Self-Doubt
- **分类: cs.CL**

- **简介: 该论文针对长链推理中模型过度思考问题，提出从"自我怀疑"角度分析其成因。发现模型因过度验证已正确答案导致冗余推理，进而提出通过提示方法让模型先质疑问题有效性再简洁回答。实验表明该方法有效减少了4种RLLMs的推理步骤和输出长度。**

- **链接: [http://arxiv.org/pdf/2505.23480v1](http://arxiv.org/pdf/2505.23480v1)**

> **作者:** Keqin Peng; Liang Ding; Yuanxin Ouyang; Meng Fang; Dacheng Tao
>
> **摘要:** Reasoning Large Language Models (RLLMs) have demonstrated impressive performance on complex tasks, largely due to the adoption of Long Chain-of-Thought (Long CoT) reasoning. However, they often exhibit overthinking -- performing unnecessary reasoning steps even after arriving at the correct answer. Prior work has largely focused on qualitative analyses of overthinking through sample-based observations of long CoTs. In contrast, we present a quantitative analysis of overthinking from the perspective of self-doubt, characterized by excessive token usage devoted to re-verifying already-correct answer. We find that self-doubt significantly contributes to overthinking. In response, we introduce a simple and effective prompting method to reduce the model's over-reliance on input questions, thereby avoiding self-doubt. Specifically, we first prompt the model to question the validity of the input question, and then respond concisely based on the outcome of that evaluation. Experiments on three mathematical reasoning tasks and four datasets with missing premises demonstrate that our method substantially reduces answer length and yields significant improvements across nearly all datasets upon 4 widely-used RLLMs. Further analysis demonstrates that our method effectively minimizes the number of reasoning steps and reduces self-doubt.
>
---
#### [new 083] Automated Essay Scoring Incorporating Annotations from Automated Feedback Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文属自动作文评分（AES）任务，旨在通过整合拼写/语法纠错与论点元素反馈注释提升评分准确性。研究将两种LLM生成的注释融入评分流程，利用微调编码器模型验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2505.22771v1](http://arxiv.org/pdf/2505.22771v1)**

> **作者:** Christopher Ormerod
>
> **备注:** 10 pages, AIME-Con Conference Submission
>
> **摘要:** This study illustrates how incorporating feedback-oriented annotations into the scoring pipeline can enhance the accuracy of automated essay scoring (AES). This approach is demonstrated with the Persuasive Essays for Rating, Selecting, and Understanding Argumentative and Discourse Elements (PERSUADE) corpus. We integrate two types of feedback-driven annotations: those that identify spelling and grammatical errors, and those that highlight argumentative components. To illustrate how this method could be applied in real-world scenarios, we employ two LLMs to generate annotations -- a generative language model used for spell-correction and an encoder-based token classifier trained to identify and mark argumentative elements. By incorporating annotations into the scoring process, we demonstrate improvements in performance using encoder-based large language models fine-tuned as classifiers.
>
---
#### [new 084] UAQFact: Evaluating Factual Knowledge Utilization of LLMs on Unanswerable Questions
- **分类: cs.CL**

- **简介: 该论文属于LLMs评估任务，旨在解决现有数据集无法评测模型利用事实知识处理不可回答问题（UAQ）的不足。工作包括构建基于知识图谱的双语UAQ数据集UAQFact，设计评估内部/外部知识利用的两项任务，实验显示LLMs即使具备知识仍表现不稳定，外部知识可提升但未被充分使用。**

- **链接: [http://arxiv.org/pdf/2505.23461v1](http://arxiv.org/pdf/2505.23461v1)**

> **作者:** Chuanyuan Tan; Wenbiao Shao; Hao Xiong; Tong Zhu; Zhenhua Liu; Kai Shi; Wenliang Chen
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Handling unanswerable questions (UAQ) is crucial for LLMs, as it helps prevent misleading responses in complex situations. While previous studies have built several datasets to assess LLMs' performance on UAQ, these datasets lack factual knowledge support, which limits the evaluation of LLMs' ability to utilize their factual knowledge when handling UAQ. To address the limitation, we introduce a new unanswerable question dataset UAQFact, a bilingual dataset with auxiliary factual knowledge created from a Knowledge Graph. Based on UAQFact, we further define two new tasks to measure LLMs' ability to utilize internal and external factual knowledge, respectively. Our experimental results across multiple LLM series show that UAQFact presents significant challenges, as LLMs do not consistently perform well even when they have factual knowledge stored. Additionally, we find that incorporating external knowledge may enhance performance, but LLMs still cannot make full use of the knowledge which may result in incorrect responses.
>
---
#### [new 085] ToMAP: Training Opponent-Aware LLM Persuaders with Theory of Mind
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出ToMAP方法，通过整合心智理论模块训练LLM说服剂。针对现有模型缺乏对手意识的问题，其设计文本编码器-分类器预测对手立场，并用强化学习优化动态说服策略。实验显示，3B参数的ToMAP超越GPT-4o等大模型，生成多样化且逻辑性强的论点，适合长期对话场景。**

- **链接: [http://arxiv.org/pdf/2505.22961v1](http://arxiv.org/pdf/2505.22961v1)**

> **作者:** Peixuan Han; Zijia Liu; Jiaxuan You
>
> **摘要:** Large language models (LLMs) have shown promising potential in persuasion, but existing works on training LLM persuaders are still preliminary. Notably, while humans are skilled in modeling their opponent's thoughts and opinions proactively and dynamically, current LLMs struggle with such Theory of Mind (ToM) reasoning, resulting in limited diversity and opponent awareness. To address this limitation, we introduce Theory of Mind Augmented Persuader (ToMAP), a novel approach for building more flexible persuader agents by incorporating two theory of mind modules that enhance the persuader's awareness and analysis of the opponent's mental state. Specifically, we begin by prompting the persuader to consider possible objections to the target central claim, and then use a text encoder paired with a trained MLP classifier to predict the opponent's current stance on these counterclaims. Our carefully designed reinforcement learning schema enables the persuader learns how to analyze opponent-related information and utilize it to generate more effective arguments. Experiments show that the ToMAP persuader, while containing only 3B parameters, outperforms much larger baselines, like GPT-4o, with a relative gain of 39.4% across multiple persuadee models and diverse corpora. Notably, ToMAP exhibits complex reasoning chains and reduced repetition during training, which leads to more diverse and effective arguments. The opponent-aware feature of ToMAP also makes it suitable for long conversations and enables it to employ more logical and opponent-aware strategies. These results underscore our method's effectiveness and highlight its potential for developing more persuasive language agents. Code is available at: https://github.com/ulab-uiuc/ToMAP.
>
---
#### [new 086] EL4NER: Ensemble Learning for Named Entity Recognition via Multiple Small-Parameter Large Language Models
- **分类: cs.CL**

- **简介: 该论文针对命名实体识别（NER）任务，解决依赖大参数语言模型成本高、资源消耗大及隐私问题。提出EL4NER方法，通过集成多个小型开源LLM的ICL输出，结合任务分解流水线、跨度级相似度检索及自验证机制，实现在更低参数成本下超越大模型性能，达SOTA效果。**

- **链接: [http://arxiv.org/pdf/2505.23038v1](http://arxiv.org/pdf/2505.23038v1)**

> **作者:** Yuzhen Xiao; Jiahe Song; Yongxin Xu; Ruizhe Zhang; Yiqi Xiao; Xin Lu; Runchuan Zhu; Bowen Jiang; Junfeng Zhao
>
> **摘要:** In-Context Learning (ICL) technique based on Large Language Models (LLMs) has gained prominence in Named Entity Recognition (NER) tasks for its lower computing resource consumption, less manual labeling overhead, and stronger generalizability. Nevertheless, most ICL-based NER methods depend on large-parameter LLMs: the open-source models demand substantial computational resources for deployment and inference, while the closed-source ones incur high API costs, raise data-privacy concerns, and hinder community collaboration. To address this question, we propose an Ensemble Learning Method for Named Entity Recognition (EL4NER), which aims at aggregating the ICL outputs of multiple open-source, small-parameter LLMs to enhance overall performance in NER tasks at less deployment and inference cost. Specifically, our method comprises three key components. First, we design a task decomposition-based pipeline that facilitates deep, multi-stage ensemble learning. Second, we introduce a novel span-level sentence similarity algorithm to establish an ICL demonstration retrieval mechanism better suited for NER tasks. Third, we incorporate a self-validation mechanism to mitigate the noise introduced during the ensemble process. We evaluated EL4NER on multiple widely adopted NER datasets from diverse domains. Our experimental results indicate that EL4NER surpasses most closed-source, large-parameter LLM-based methods at a lower parameter cost and even attains state-of-the-art (SOTA) performance among ICL-based methods on certain datasets. These results show the parameter efficiency of EL4NER and underscore the feasibility of employing open-source, small-parameter LLMs within the ICL paradigm for NER tasks.
>
---
#### [new 087] Evaluating the performance and fragility of large language models on the self-assessment for neurological surgeons
- **分类: cs.CL**

- **简介: 该论文评估大型语言模型在神经外科考试题中的表现及抗干扰能力，解决其对临床相关干扰信息的脆弱性问题。研究测试28个模型在2904道试题上的表现，引入含多义词的干扰框架，发现仅6个模型通过考试，干扰导致准确率最高降20.4%，凸显需提升模型鲁棒性以保障临床应用安全。**

- **链接: [http://arxiv.org/pdf/2505.23477v1](http://arxiv.org/pdf/2505.23477v1)**

> **作者:** Krithik Vishwanath; Anton Alyakin; Mrigayu Ghosh; Jin Vivian Lee; Daniel Alexander Alber; Karl L. Sangwon; Douglas Kondziolka; Eric Karl Oermann
>
> **备注:** 22 pages, 3 main figures, 3 supplemental figures
>
> **摘要:** The Congress of Neurological Surgeons Self-Assessment for Neurological Surgeons (CNS-SANS) questions are widely used by neurosurgical residents to prepare for written board examinations. Recently, these questions have also served as benchmarks for evaluating large language models' (LLMs) neurosurgical knowledge. This study aims to assess the performance of state-of-the-art LLMs on neurosurgery board-like questions and to evaluate their robustness to the inclusion of distractor statements. A comprehensive evaluation was conducted using 28 large language models. These models were tested on 2,904 neurosurgery board examination questions derived from the CNS-SANS. Additionally, the study introduced a distraction framework to assess the fragility of these models. The framework incorporated simple, irrelevant distractor statements containing polysemous words with clinical meanings used in non-clinical contexts to determine the extent to which such distractions degrade model performance on standard medical benchmarks. 6 of the 28 tested LLMs achieved board-passing outcomes, with the top-performing models scoring over 15.7% above the passing threshold. When exposed to distractions, accuracy across various model architectures was significantly reduced-by as much as 20.4%-with one model failing that had previously passed. Both general-purpose and medical open-source models experienced greater performance declines compared to proprietary variants when subjected to the added distractors. While current LLMs demonstrate an impressive ability to answer neurosurgery board-like exam questions, their performance is markedly vulnerable to extraneous, distracting information. These findings underscore the critical need for developing novel mitigation strategies aimed at bolstering LLM resilience against in-text distractions, particularly for safe and effective clinical deployment.
>
---
#### [new 088] Climate Finance Bench
- **分类: cs.CL**

- **简介: 该论文提出Climate Finance Bench，一个针对企业气候披露文档问答任务的基准测试，包含跨11个行业的33份报告及330个专家验证的问答对。任务对比RAG方法，揭示检索模块的准确性是主要瓶颈，并倡导AI气候应用中采用透明碳报告技术（如权重量化）。**

- **链接: [http://arxiv.org/pdf/2505.22752v1](http://arxiv.org/pdf/2505.22752v1)**

> **作者:** Rafik Mankour; Yassine Chafai; Hamada Saleh; Ghassen Ben Hassine; Thibaud Barreau; Peter Tankov
>
> **备注:** Dataset is available at https://github.com/Pladifes/climate_finance_bench
>
> **摘要:** Climate Finance Bench introduces an open benchmark that targets question-answering over corporate climate disclosures using Large Language Models. We curate 33 recent sustainability reports in English drawn from companies across all 11 GICS sectors and annotate 330 expert-validated question-answer pairs that span pure extraction, numerical reasoning, and logical reasoning. Building on this dataset, we propose a comparison of RAG (retrieval-augmented generation) approaches. We show that the retriever's ability to locate passages that actually contain the answer is the chief performance bottleneck. We further argue for transparent carbon reporting in AI-for-climate applications, highlighting advantages of techniques such as Weight Quantization.
>
---
#### [new 089] OWL: Probing Cross-Lingual Recall of Memorized Texts via World Literature
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于跨语言记忆探测任务，探究大型语言模型（LLMs）在非英语及跨语言文本中的记忆与召回能力。通过构建含31.5K多语言对齐文本的OWL数据集，设计直接探测、名字填空和前缀生成任务，评估模型对不同语言书籍内容的识别与生成能力，发现LLMs可跨语言召回记忆内容，即使无直接翻译训练数据。**

- **链接: [http://arxiv.org/pdf/2505.22945v1](http://arxiv.org/pdf/2505.22945v1)**

> **作者:** Alisha Srivastava; Emir Korukluoglu; Minh Nhat Le; Duyen Tran; Chau Minh Pham; Marzena Karpinska; Mohit Iyyer
>
> **备注:** preprint, 25 pages
>
> **摘要:** Large language models (LLMs) are known to memorize and recall English text from their pretraining data. However, the extent to which this ability generalizes to non-English languages or transfers across languages remains unclear. This paper investigates multilingual and cross-lingual memorization in LLMs, probing if memorized content in one language (e.g., English) can be recalled when presented in translation. To do so, we introduce OWL, a dataset of 31.5K aligned excerpts from 20 books in ten languages, including English originals, official translations (Vietnamese, Spanish, Turkish), and new translations in six low-resource languages (Sesotho, Yoruba, Maithili, Malagasy, Setswana, Tahitian). We evaluate memorization across model families and sizes through three tasks: (1) direct probing, which asks the model to identify a book's title and author; (2) name cloze, which requires predicting masked character names; and (3) prefix probing, which involves generating continuations. We find that LLMs consistently recall content across languages, even for texts without direct translation in pretraining data. GPT-4o, for example, identifies authors and titles 69% of the time and masked entities 6% of the time in newly translated excerpts. Perturbations (e.g., masking characters, shuffling words) modestly reduce direct probing accuracy (7% drop for shuffled official translations). Our results highlight the extent of cross-lingual memorization and provide insights on the differences between the models.
>
---
#### [new 090] GateNLP at SemEval-2025 Task 10: Hierarchical Three-Step Prompting for Multilingual Narrative Classification
- **分类: cs.CL**

- **简介: 该论文提出H3Prompt方法，针对SemEval-2025任务10子任务2的多语言新闻叙事分类问题。通过三步LLM提示策略（先分领域，再主叙事，最后子叙事），解决跨语言自动识别新闻叙事层级分类的难题，在英语测试中获全球第一。**

- **链接: [http://arxiv.org/pdf/2505.22867v1](http://arxiv.org/pdf/2505.22867v1)**

> **作者:** Iknoor Singh; Carolina Scarton; Kalina Bontcheva
>
> **摘要:** The proliferation of online news and the increasing spread of misinformation necessitate robust methods for automatic data analysis. Narrative classification is emerging as a important task, since identifying what is being said online is critical for fact-checkers, policy markers and other professionals working on information studies. This paper presents our approach to SemEval 2025 Task 10 Subtask 2, which aims to classify news articles into a pre-defined two-level taxonomy of main narratives and sub-narratives across multiple languages. We propose Hierarchical Three-Step Prompting (H3Prompt) for multilingual narrative classification. Our methodology follows a three-step Large Language Model (LLM) prompting strategy, where the model first categorises an article into one of two domains (Ukraine-Russia War or Climate Change), then identifies the most relevant main narratives, and finally assigns sub-narratives. Our approach secured the top position on the English test set among 28 competing teams worldwide. The code is available at https://github.com/GateNLP/H3Prompt.
>
---
#### [new 091] PBEBench: A Multi-Step Programming by Examples Reasoning Benchmark inspired by Historical Linguistics
- **分类: cs.CL**

- **简介: 该论文提出PBEBench，一个多步骤编程通过示例（PbE）推理基准，受历史语言学启发，评估LLMs在抽象推理中的泛化能力。通过自动化生成可控难度的测试集（近1000例），发现当前最优模型Claude-3.7-Sonnet仅54%通过率，揭示LLMs在该类问题上的不足。**

- **链接: [http://arxiv.org/pdf/2505.23126v1](http://arxiv.org/pdf/2505.23126v1)**

> **作者:** Atharva Naik; Darsh Agrawal; Manav Kapadnis; Yuwei An; Yash Mathur; Carolyn Rose; David Mortensen
>
> **摘要:** Recently, long chain of thought (LCoT), Large Language Models (LLMs), have taken the machine learning world by storm with their breathtaking reasoning capabilities. However, are the abstract reasoning abilities of these models general enough for problems of practical importance? Unlike past work, which has focused mainly on math, coding, and data wrangling, we focus on a historical linguistics-inspired inductive reasoning problem, formulated as Programming by Examples. We develop a fully automated pipeline for dynamically generating a benchmark for this task with controllable difficulty in order to tackle scalability and contamination issues to which many reasoning benchmarks are subject. Using our pipeline, we generate a test set with nearly 1k instances that is challenging for all state-of-the-art reasoning LLMs, with the best model (Claude-3.7-Sonnet) achieving a mere 54% pass rate, demonstrating that LCoT LLMs still struggle with a class or reasoning that is ubiquitous in historical linguistics as well as many other domains.
>
---
#### [new 092] When Models Reason in Your Language: Controlling Thinking Trace Language Comes at the Cost of Accuracy
- **分类: cs.CL**

- **简介: 该论文研究多语言推理任务，评估大型推理模型（LRMs）在非英语语言中的推理能力。发现模型常退化为英语或输出碎片化推理，存在多语言推理差距。通过提示干预可提升可读性但降低准确率，少量数据微调部分缓解问题。指出当前模型多语言推理局限及优化方向。**

- **链接: [http://arxiv.org/pdf/2505.22888v1](http://arxiv.org/pdf/2505.22888v1)**

> **作者:** Jirui Qi; Shan Chen; Zidi Xiong; Raquel Fernández; Danielle S. Bitterman; Arianna Bisazza
>
> **摘要:** Recent Large Reasoning Models (LRMs) with thinking traces have shown strong performance on English reasoning tasks. However, their ability to think in other languages is less studied. This capability is as important as answer accuracy for real world applications because users may find the reasoning trace useful for oversight only when it is expressed in their own language. We comprehensively evaluate two leading families of LRMs on our XReasoning benchmark and find that even the most advanced models often revert to English or produce fragmented reasoning in other languages, revealing a substantial gap in multilingual reasoning. Prompt based interventions that force models to reason in the users language improve readability and oversight but reduce answer accuracy, exposing an important trade off. We further show that targeted post training on just 100 examples mitigates this mismatch, though some accuracy loss remains. Our results highlight the limited multilingual reasoning capabilities of current LRMs and outline directions for future work. Code and data are available at https://github.com/Betswish/mCoT-XReasoning.
>
---
#### [new 093] Elicit and Enhance: Advancing Multimodal Reasoning in Medical Scenarios
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医学多模态推理任务，旨在解决现有模型在医疗场景中推理能力不足的问题。提出MedE²两阶段训练方法：首阶段用2000个文本示例激发推理行为，次阶段通过1500个严格筛选的多模态病例增强推理能力，实验显示其显著提升医学推理性能并具 robustness。**

- **链接: [http://arxiv.org/pdf/2505.23118v1](http://arxiv.org/pdf/2505.23118v1)**

> **作者:** Linjie Mu; Zhongzhen Huang; Yakun Zhu; Xiangyu Zhao; Shaoting Zhang; Xiaofan Zhang
>
> **摘要:** Effective clinical decision-making depends on iterative, multimodal reasoning across diverse sources of evidence. The recent emergence of multimodal reasoning models has significantly transformed the landscape of solving complex tasks. Although such models have achieved notable success in mathematics and science, their application to medical domains remains underexplored. In this work, we propose \textit{MedE$^2$}, a two-stage post-training pipeline that elicits and then enhances multimodal reasoning for medical domains. In Stage-I, we fine-tune models using 2,000 text-only data samples containing precisely orchestrated reasoning demonstrations to elicit reasoning behaviors. In Stage-II, we further enhance the model's reasoning capabilities using 1,500 rigorously curated multimodal medical cases, aligning model reasoning outputs with our proposed multimodal medical reasoning preference. Extensive experiments demonstrate the efficacy and reliability of \textit{MedE$^2$} in improving the reasoning performance of medical multimodal models. Notably, models trained with \textit{MedE$^2$} consistently outperform baselines across multiple medical multimodal benchmarks. Additional validation on larger models and under inference-time scaling further confirms the robustness and practical utility of our approach.
>
---
#### [new 094] A Practical Approach for Building Production-Grade Conversational Agents with Workflow Graphs
- **分类: cs.CL; cs.AI; I.2.7**

- **简介: 该论文聚焦构建生产级对话代理任务，解决工业场景中平衡LLMs灵活性与服务约束的冲突问题。提出基于工作流图的框架，通过电商案例研究，设计实施流程与优化策略，实现可控、可靠的AI代理开发。**

- **链接: [http://arxiv.org/pdf/2505.23006v1](http://arxiv.org/pdf/2505.23006v1)**

> **作者:** Chiwan Park; Wonjun Jang; Daeryong Kim; Aelim Ahn; Kichang Yang; Woosung Hwang; Jihyeon Roh; Hyerin Park; Hyosun Wang; Min Seok Kim; Jihoon Kang
>
> **备注:** Accepted to ACL 2025 Industry Track. 12 pages, 5 figures
>
> **摘要:** The advancement of Large Language Models (LLMs) has led to significant improvements in various service domains, including search, recommendation, and chatbot applications. However, applying state-of-the-art (SOTA) research to industrial settings presents challenges, as it requires maintaining flexible conversational abilities while also strictly complying with service-specific constraints. This can be seen as two conflicting requirements due to the probabilistic nature of LLMs. In this paper, we propose our approach to addressing this challenge and detail the strategies we employed to overcome their inherent limitations in real-world applications. We conduct a practical case study of a conversational agent designed for the e-commerce domain, detailing our implementation workflow and optimizations. Our findings provide insights into bridging the gap between academic research and real-world application, introducing a framework for developing scalable, controllable, and reliable AI-driven agents.
>
---
#### [new 095] Tell, Don't Show: Leveraging Language Models' Abstractive Retellings to Model Literary Themes
- **分类: cs.CL**

- **简介: 该论文属于文学主题建模任务，解决传统词袋方法（如LDA）在分析文学文本时因依赖感官细节而效果不佳的问题。提出Retell方法：通过语言模型将文学段落转为抽象概括，再用LDA提取主题，优于单独使用LDA或直接询问LM。通过高中教材案例验证其文化分析潜力。**

- **链接: [http://arxiv.org/pdf/2505.23166v1](http://arxiv.org/pdf/2505.23166v1)**

> **作者:** Li Lucy; Camilla Griffiths; Sarah Levine; Jennifer L. Eberhardt; Dorottya Demszky; David Bamman
>
> **备注:** 26 pages, 7 figures, Findings of ACL 2025
>
> **摘要:** Conventional bag-of-words approaches for topic modeling, like latent Dirichlet allocation (LDA), struggle with literary text. Literature challenges lexical methods because narrative language focuses on immersive sensory details instead of abstractive description or exposition: writers are advised to "show, don't tell." We propose Retell, a simple, accessible topic modeling approach for literature. Here, we prompt resource-efficient, generative language models (LMs) to tell what passages show, thereby translating narratives' surface forms into higher-level concepts and themes. By running LDA on LMs' retellings of passages, we can obtain more precise and informative topics than by running LDA alone or by directly asking LMs to list topics. To investigate the potential of our method for cultural analytics, we compare our method's outputs to expert-guided annotations in a case study on racial/cultural identity in high school English language arts books.
>
---
#### [new 096] The Arabic AI Fingerprint: Stylometric Analysis and Detection of Large Language Models Text
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于阿拉伯语AI文本检测任务，解决低资源语言中机器生成文本难以识别的问题。通过分析多生成策略及模型（如GPT-4、Llama）在学术/社交媒体中的阿拉伯语输出，发现机器独特语言特征，开发BERT检测模型，正式场景达99.9%精度，揭示跨领域检测挑战，为信息完整性保护提供基础。**

- **链接: [http://arxiv.org/pdf/2505.23276v1](http://arxiv.org/pdf/2505.23276v1)**

> **作者:** Maged S. Al-Shaibani; Moataz Ahmed
>
> **摘要:** Large Language Models (LLMs) have achieved unprecedented capabilities in generating human-like text, posing subtle yet significant challenges for information integrity across critical domains, including education, social media, and academia, enabling sophisticated misinformation campaigns, compromising healthcare guidance, and facilitating targeted propaganda. This challenge becomes severe, particularly in under-explored and low-resource languages like Arabic. This paper presents a comprehensive investigation of Arabic machine-generated text, examining multiple generation strategies (generation from the title only, content-aware generation, and text refinement) across diverse model architectures (ALLaM, Jais, Llama, and GPT-4) in academic, and social media domains. Our stylometric analysis reveals distinctive linguistic patterns differentiating human-written from machine-generated Arabic text across these varied contexts. Despite their human-like qualities, we demonstrate that LLMs produce detectable signatures in their Arabic outputs, with domain-specific characteristics that vary significantly between different contexts. Based on these insights, we developed BERT-based detection models that achieved exceptional performance in formal contexts (up to 99.9\% F1-score) with strong precision across model architectures. Our cross-domain analysis confirms generalization challenges previously reported in the literature. To the best of our knowledge, this work represents the most comprehensive investigation of Arabic machine-generated text to date, uniquely combining multiple prompt generation methods, diverse model architectures, and in-depth stylometric analysis across varied textual domains, establishing a foundation for developing robust, linguistically-informed detection systems essential for preserving information integrity in Arabic-language contexts.
>
---
#### [new 097] ContextQFormer: A New Context Modeling Method for Multi-Turn Multi-Modal Conversations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多轮多模态对话任务，旨在解决现有模型在长上下文交互中的能力不足问题。提出ContextQFormer模块通过记忆块增强上下文表示，并构建新数据集TMDialog，实验显示较基线提升2%-4%可用率。**

- **链接: [http://arxiv.org/pdf/2505.23121v1](http://arxiv.org/pdf/2505.23121v1)**

> **作者:** Yiming Lei; Zhizheng Yang; Zeming Liu; Haitao Leng; Shaoguo Liu; Tingting Gao; Qingjie Liu; Yunhong Wang
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Multi-modal large language models have demonstrated remarkable zero-shot abilities and powerful image-understanding capabilities. However, the existing open-source multi-modal models suffer from the weak capability of multi-turn interaction, especially for long contexts. To address the issue, we first introduce a context modeling module, termed ContextQFormer, which utilizes a memory block to enhance the presentation of contextual information. Furthermore, to facilitate further research, we carefully build a new multi-turn multi-modal dialogue dataset (TMDialog) for pre-training, instruction-tuning, and evaluation, which will be open-sourced lately. Compared with other multi-modal dialogue datasets, TMDialog contains longer conversations, which supports the research of multi-turn multi-modal dialogue. In addition, ContextQFormer is compared with three baselines on TMDialog and experimental results illustrate that ContextQFormer achieves an improvement of 2%-4% in available rate over baselines.
>
---
#### [new 098] Understanding Refusal in Language Models with Sparse Autoencoders
- **分类: cs.CL**

- **简介: 该论文研究语言模型拒绝行为的内部机制，通过稀疏自编码器识别关键潜特征，干预其影响生成，分析潜在关系及对抗技术，提升分类泛化，开源代码。**

- **链接: [http://arxiv.org/pdf/2505.23556v1](http://arxiv.org/pdf/2505.23556v1)**

> **作者:** Wei Jie Yeo; Nirmalendu Prakash; Clement Neo; Roy Ka-Wei Lee; Erik Cambria; Ranjan Satapathy
>
> **摘要:** Refusal is a key safety behavior in aligned language models, yet the internal mechanisms driving refusals remain opaque. In this work, we conduct a mechanistic study of refusal in instruction-tuned LLMs using sparse autoencoders to identify latent features that causally mediate refusal behaviors. We apply our method to two open-source chat models and intervene on refusal-related features to assess their influence on generation, validating their behavioral impact across multiple harmful datasets. This enables a fine-grained inspection of how refusal manifests at the activation level and addresses key research questions such as investigating upstream-downstream latent relationship and understanding the mechanisms of adversarial jailbreaking techniques. We also establish the usefulness of refusal features in enhancing generalization for linear probes to out-of-distribution adversarial samples in classification tasks. We open source our code in https://github.com/wj210/refusal_sae.
>
---
#### [new 099] Machine-Facing English: Defining a Hybrid Register Shaped by Human-AI Discourse
- **分类: cs.CL**

- **简介: 该论文研究"面向机器的英语"（MFE）这一新兴语言变体，分析人机交互如何催生语法僵化、语用简化等特征，通过双语测试数据识别其五项核心特征，探讨语言效率与表达丰富性的矛盾，为AI交互设计和多语种教育提供理论依据。**

- **链接: [http://arxiv.org/pdf/2505.23035v1](http://arxiv.org/pdf/2505.23035v1)**

> **作者:** Hyunwoo Kim; Hanau Yi
>
> **摘要:** Machine-Facing English (MFE) is an emergent register shaped by the adaptation of everyday language to the expanding presence of AI interlocutors. Drawing on register theory (Halliday 1985, 2006), enregisterment (Agha 2003), audience design (Bell 1984), and interactional pragmatics (Giles & Ogay 2007), this study traces how sustained human-AI interaction normalizes syntactic rigidity, pragmatic simplification, and hyper-explicit phrasing - features that enhance machine parseability at the expense of natural fluency. Our analysis is grounded in qualitative observations from bilingual (Korean/English) voice- and text-based product testing sessions, with reflexive drafting conducted using Natural Language Declarative Prompting (NLD-P) under human curation. Thematic analysis identifies five recurrent traits - redundant clarity, directive syntax, controlled vocabulary, flattened prosody, and single-intent structuring - that improve execution accuracy but compress expressive range. MFE's evolution highlights a persistent tension between communicative efficiency and linguistic richness, raising design challenges for conversational interfaces and pedagogical considerations for multilingual users. We conclude by underscoring the need for comprehensive methodological exposition and future empirical validation.
>
---
#### [new 100] Puzzled by Puzzles: When Vision-Language Models Can't Take a Hint
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **简介: 该论文评估视觉语言模型（VLM）解决rebus谜题的能力。针对VLM在抽象推理、符号推理及文化语言双关理解上的不足，构建了包含多样英语rebus的基准数据集，分析不同模型表现，发现其虽能处理简单线索，但复杂任务表现欠佳。**

- **链接: [http://arxiv.org/pdf/2505.23759v1](http://arxiv.org/pdf/2505.23759v1)**

> **作者:** Heekyung Lee; Jiaxin Ge; Tsung-Han Wu; Minwoo Kang; Trevor Darrell; David M. Chan
>
> **摘要:** Rebus puzzles, visual riddles that encode language through imagery, spatial arrangement, and symbolic substitution, pose a unique challenge to current vision-language models (VLMs). Unlike traditional image captioning or question answering tasks, rebus solving requires multi-modal abstraction, symbolic reasoning, and a grasp of cultural, phonetic and linguistic puns. In this paper, we investigate the capacity of contemporary VLMs to interpret and solve rebus puzzles by constructing a hand-generated and annotated benchmark of diverse English-language rebus puzzles, ranging from simple pictographic substitutions to spatially-dependent cues ("head" over "heels"). We analyze how different VLMs perform, and our findings reveal that while VLMs exhibit some surprising capabilities in decoding simple visual clues, they struggle significantly with tasks requiring abstract reasoning, lateral thinking, and understanding visual metaphors.
>
---
#### [new 101] Child-Directed Language Does Not Consistently Boost Syntax Learning in Language Models
- **分类: cs.CL**

- **简介: 该论文探讨儿童导向语言（CDL）是否提升语言模型的句法学习，测试其跨语言（英、法、德）、模型类型（掩码/因果）及评估设置的普适性。结果显示CDL多数表现不及维基数据，提出新方法FIT-CLAMS，指出需控制频率效应以准确评估句法能力。（99字）**

- **链接: [http://arxiv.org/pdf/2505.23689v1](http://arxiv.org/pdf/2505.23689v1)**

> **作者:** Francesca Padovani; Jaap Jumelet; Yevgen Matusevych; Arianna Bisazza
>
> **备注:** 21 pages, 4 figures, 4 tables
>
> **摘要:** Seminal work by Huebner et al. (2021) showed that language models (LMs) trained on English Child-Directed Language (CDL) can reach similar syntactic abilities as LMs trained on much larger amounts of adult-directed written text, suggesting that CDL could provide more effective LM training material than the commonly used internet-crawled data. However, the generalizability of these results across languages, model types, and evaluation settings remains unclear. We test this by comparing models trained on CDL vs. Wikipedia across two LM objectives (masked and causal), three languages (English, French, German), and three syntactic minimal-pair benchmarks. Our results on these benchmarks show inconsistent benefits of CDL, which in most cases is outperformed by Wikipedia models. We then identify various shortcomings in previous benchmarks, and introduce a novel testing methodology, FIT-CLAMS, which uses a frequency-controlled design to enable balanced comparisons across training corpora. Through minimal pair evaluations and regression analysis we show that training on CDL does not yield stronger generalizations for acquiring syntax and highlight the importance of controlling for frequency effects when evaluating syntactic ability.
>
---
#### [new 102] Are Reasoning Models More Prone to Hallucination?
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大型推理模型（LRMs）是否更易产生幻觉，分析不同后训练方法对事实性任务的影响。发现冷启动微调与RL可减少幻觉，而蒸馏或无微调方法加剧问题；揭示"错误重复"和"思考-答案不匹配"行为，及模型不确定性与事实准确性的不匹配机制，以优化训练策略减少幻觉。**

- **链接: [http://arxiv.org/pdf/2505.23646v1](http://arxiv.org/pdf/2505.23646v1)**

> **作者:** Zijun Yao; Yantao Liu; Yanxu Chen; Jianhui Chen; Junfeng Fang; Lei Hou; Juanzi Li; Tat-Seng Chua
>
> **摘要:** Recently evolved large reasoning models (LRMs) show powerful performance in solving complex tasks with long chain-of-thought (CoT) reasoning capability. As these LRMs are mostly developed by post-training on formal reasoning tasks, whether they generalize the reasoning capability to help reduce hallucination in fact-seeking tasks remains unclear and debated. For instance, DeepSeek-R1 reports increased performance on SimpleQA, a fact-seeking benchmark, while OpenAI-o3 observes even severer hallucination. This discrepancy naturally raises the following research question: Are reasoning models more prone to hallucination? This paper addresses the question from three perspectives. (1) We first conduct a holistic evaluation for the hallucination in LRMs. Our analysis reveals that LRMs undergo a full post-training pipeline with cold start supervised fine-tuning (SFT) and verifiable reward RL generally alleviate their hallucination. In contrast, both distillation alone and RL training without cold start fine-tuning introduce more nuanced hallucinations. (2) To explore why different post-training pipelines alters the impact on hallucination in LRMs, we conduct behavior analysis. We characterize two critical cognitive behaviors that directly affect the factuality of a LRM: Flaw Repetition, where the surface-level reasoning attempts repeatedly follow the same underlying flawed logic, and Think-Answer Mismatch, where the final answer fails to faithfully match the previous CoT process. (3) Further, we investigate the mechanism behind the hallucination of LRMs from the perspective of model uncertainty. We find that increased hallucination of LRMs is usually associated with the misalignment between model uncertainty and factual accuracy. Our work provides an initial understanding of the hallucination in LRMs.
>
---
#### [new 103] Diagnosing and Addressing Pitfalls in KG-RAG Datasets: Toward More Reliable Benchmarking
- **分类: cs.CL; cs.LG**

- **简介: 该论文聚焦KGQA基准改进任务，针对现有数据集标注不准确、问题设计缺陷及知识过时等问题，提出KGQAGen框架，结合知识 grounding、LLM生成与符号验证生成高质量QA数据，构建了10K规模的Wikidata基准，揭示现有模型局限，推动严谨评估。**

- **链接: [http://arxiv.org/pdf/2505.23495v1](http://arxiv.org/pdf/2505.23495v1)**

> **作者:** Liangliang Zhang; Zhuorui Jiang; Hongliang Chi; Haoyang Chen; Mohammed Elkoumy; Fali Wang; Qiong Wu; Zhengyi Zhou; Shirui Pan; Suhang Wang; Yao Ma
>
> **备注:** 9 pages
>
> **摘要:** Knowledge Graph Question Answering (KGQA) systems rely on high-quality benchmarks to evaluate complex multi-hop reasoning. However, despite their widespread use, popular datasets such as WebQSP and CWQ suffer from critical quality issues, including inaccurate or incomplete ground-truth annotations, poorly constructed questions that are ambiguous, trivial, or unanswerable, and outdated or inconsistent knowledge. Through a manual audit of 16 popular KGQA datasets, including WebQSP and CWQ, we find that the average factual correctness rate is only 57 %. To address these issues, we introduce KGQAGen, an LLM-in-the-loop framework that systematically resolves these pitfalls. KGQAGen combines structured knowledge grounding, LLM-guided generation, and symbolic verification to produce challenging and verifiable QA instances. Using KGQAGen, we construct KGQAGen-10k, a ten-thousand scale benchmark grounded in Wikidata, and evaluate a diverse set of KG-RAG models. Experimental results demonstrate that even state-of-the-art systems struggle on this benchmark, highlighting its ability to expose limitations of existing models. Our findings advocate for more rigorous benchmark construction and position KGQAGen as a scalable framework for advancing KGQA evaluation.
>
---
#### [new 104] Proximalized Preference Optimization for Diverse Feedback Types: A Decomposed Perspective on DPO
- **分类: cs.CL**

- **简介: 该论文属于大型语言模型对齐任务，旨在解决对比对齐方法导致的"似然不足"问题（输出偏离预期），提出PRO方法。通过分解DPO损失函数，发现其正则项简化引发问题，完整重构正则项并提出近似方案，实现对多种反馈类型的统一优化，实验验证其优势。**

- **链接: [http://arxiv.org/pdf/2505.23316v1](http://arxiv.org/pdf/2505.23316v1)**

> **作者:** Kaiyang Guo; Yinchuan Li; Zhitang Chen
>
> **摘要:** Direct alignment methods typically optimize large language models (LLMs) by contrasting the likelihoods of preferred versus dispreferred responses. While effective in steering LLMs to match relative preference, these methods are frequently noted for decreasing the absolute likelihoods of example responses. As a result, aligned models tend to generate outputs that deviate from the expected patterns, exhibiting reward-hacking effect even without a reward model. This undesired consequence exposes a fundamental limitation in contrastive alignment, which we characterize as likelihood underdetermination. In this work, we revisit direct preference optimization (DPO) -- the seminal direct alignment method -- and demonstrate that its loss theoretically admits a decomposed reformulation. The reformulated loss not only broadens applicability to a wider range of feedback types, but also provides novel insights into the underlying cause of likelihood underdetermination. Specifically, the standard DPO implementation implicitly oversimplifies a regularizer in the reformulated loss, and reinstating its complete version effectively resolves the underdetermination issue. Leveraging these findings, we introduce PRoximalized PReference Optimization (PRO), a unified method to align with diverse feeback types, eliminating likelihood underdetermination through an efficient approximation of the complete regularizer. Comprehensive experiments show the superiority of PRO over existing methods in scenarios involving pairwise, binary and scalar feedback.
>
---
#### [new 105] Structured Memory Mechanisms for Stable Context Representation in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于大语言模型长期上下文理解任务，旨在解决传统模型在处理长依赖时的上下文丢失和语义漂移问题。提出结构化记忆机制，包含显式记忆单元、门控写入、注意力读取及遗忘函数，并设计联合训练目标优化记忆策略，实验验证了其在文本生成、多轮问答和跨上下文推理中的有效性。**

- **链接: [http://arxiv.org/pdf/2505.22921v1](http://arxiv.org/pdf/2505.22921v1)**

> **作者:** Yue Xing; Tao Yang; Yijiashun Qi; Minggu Wei; Yu Cheng; Honghui Xin
>
> **摘要:** This paper addresses the limitations of large language models in understanding long-term context. It proposes a model architecture equipped with a long-term memory mechanism to improve the retention and retrieval of semantic information across paragraphs and dialogue turns. The model integrates explicit memory units, gated writing mechanisms, and attention-based reading modules. A forgetting function is introduced to enable dynamic updates of memory content, enhancing the model's ability to manage historical information. To further improve the effectiveness of memory operations, the study designs a joint training objective. This combines the main task loss with constraints on memory writing and forgetting. It guides the model to learn better memory strategies during task execution. Systematic evaluation across multiple subtasks shows that the model achieves clear advantages in text generation consistency, stability in multi-turn question answering, and accuracy in cross-context reasoning. In particular, the model demonstrates strong semantic retention and contextual coherence in long-text tasks and complex question answering scenarios. It effectively mitigates the context loss and semantic drift problems commonly faced by traditional language models when handling long-term dependencies. The experiments also include analysis of different memory structures, capacity sizes, and control strategies. These results further confirm the critical role of memory mechanisms in language understanding. They demonstrate the feasibility and effectiveness of the proposed approach in both architectural design and performance outcomes.
>
---
#### [new 106] LLMs for Argument Mining: Detection, Extraction, and Relationship Classification of pre-defined Arguments in Online Comments
- **分类: cs.CL**

- **简介: 该论文研究LLMs在在线评论中预定义论点挖掘的任务，解决其在争议话题中的检测、提取及关系分类性能问题。通过评估四类先进LLM在2000+条评论上的表现，发现其整体效果良好但存在处理长文本和情绪化语言的不足，同时指出环境成本高的局限。**

- **链接: [http://arxiv.org/pdf/2505.22956v1](http://arxiv.org/pdf/2505.22956v1)**

> **作者:** Matteo Guida; Yulia Otmakhova; Eduard Hovy; Lea Frermann
>
> **摘要:** Automated large-scale analysis of public discussions around contested issues like abortion requires detecting and understanding the use of arguments. While Large Language Models (LLMs) have shown promise in language processing tasks, their performance in mining topic-specific, pre-defined arguments in online comments remains underexplored. We evaluate four state-of-the-art LLMs on three argument mining tasks using datasets comprising over 2,000 opinion comments across six polarizing topics. Quantitative evaluation suggests an overall strong performance across the three tasks, especially for large and fine-tuned LLMs, albeit at a significant environmental cost. However, a detailed error analysis revealed systematic shortcomings on long and nuanced comments and emotionally charged language, raising concerns for downstream applications like content moderation or opinion analysis. Our results highlight both the promise and current limitations of LLMs for automated argument analysis in online comments.
>
---
#### [new 107] Dataset Cartography for Large Language Model Alignment: Mapping and Diagnosing Preference Data
- **分类: cs.CL**

- **简介: 该论文提出Alignment Data Map工具，利用GPT-4o分析人类偏好数据，解决LLM对齐中数据收集效率低的问题。通过计算对齐分数构建数据地图，筛选高价值样本（仅需33%数据即可达相近性能），并诊断低质量或误标样本，提升数据利用效率。**

- **链接: [http://arxiv.org/pdf/2505.23114v1](http://arxiv.org/pdf/2505.23114v1)**

> **作者:** Seohyeong Lee; Eunwon Kim; Hwaran Lee; Buru Chang
>
> **摘要:** Human preference data plays a critical role in aligning large language models (LLMs) with human values. However, collecting such data is often expensive and inefficient, posing a significant scalability challenge. To address this, we introduce Alignment Data Map, a GPT-4o-assisted tool for analyzing and diagnosing preference data. Using GPT-4o as a proxy for LLM alignment, we compute alignment scores for LLM-generated responses to instructions from existing preference datasets. These scores are then used to construct an Alignment Data Map based on their mean and variance. Our experiments show that using only 33 percent of the data, specifically samples in the high-mean, low-variance region, achieves performance comparable to or better than using the entire dataset. This finding suggests that the Alignment Data Map can significantly improve data collection efficiency by identifying high-quality samples for LLM alignment without requiring explicit annotations. Moreover, the Alignment Data Map can diagnose existing preference datasets. Our analysis shows that it effectively detects low-impact or potentially misannotated samples. Source code is available online.
>
---
#### [new 108] Discriminative Policy Optimization for Token-Level Reward Models
- **分类: cs.CL**

- **简介: 论文提出Q-RM方法，解决生成模型与奖励模型冲突导致的训练不稳定和信用分配问题。通过分离奖励模型与生成过程，基于Q函数优化学习token级奖励，提升LLM复杂推理能力。实验显示其在数学任务上显著优于基线模型，且收敛速度加快11-12倍。**

- **链接: [http://arxiv.org/pdf/2505.23363v1](http://arxiv.org/pdf/2505.23363v1)**

> **作者:** Hongzhan Chen; Tao Yang; Shiping Gao; Ruijun Chen; Xiaojun Quan; Hongtao Tian; Ting Yao
>
> **备注:** ICML 2025
>
> **摘要:** Process reward models (PRMs) provide more nuanced supervision compared to outcome reward models (ORMs) for optimizing policy models, positioning them as a promising approach to enhancing the capabilities of LLMs in complex reasoning tasks. Recent efforts have advanced PRMs from step-level to token-level granularity by integrating reward modeling into the training of generative models, with reward scores derived from token generation probabilities. However, the conflict between generative language modeling and reward modeling may introduce instability and lead to inaccurate credit assignments. To address this challenge, we revisit token-level reward assignment by decoupling reward modeling from language generation and derive a token-level reward model through the optimization of a discriminative policy, termed the Q-function Reward Model (Q-RM). We theoretically demonstrate that Q-RM explicitly learns token-level Q-functions from preference data without relying on fine-grained annotations. In our experiments, Q-RM consistently outperforms all baseline methods across various benchmarks. For example, when integrated into PPO/REINFORCE algorithms, Q-RM enhances the average Pass@1 score by 5.85/4.70 points on mathematical reasoning tasks compared to the ORM baseline, and by 4.56/5.73 points compared to the token-level PRM counterpart. Moreover, reinforcement learning with Q-RM significantly enhances training efficiency, achieving convergence 12 times faster than ORM on GSM8K and 11 times faster than step-level PRM on MATH. Code and data are available at https://github.com/homzer/Q-RM.
>
---
#### [new 109] FAMA: The First Large-Scale Open-Science Speech Foundation Model for English and Italian
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文提出FAMA——首个开源英语和意大利语语音基础模型，解决现有模型封闭性导致的可复现性差问题。工作包括：基于15万+小时开源语音数据训练模型，构建1.6万小时清洗伪标记数据集，实现性能与速度优势（快8倍），并完全开放代码、数据及模型。**

- **链接: [http://arxiv.org/pdf/2505.22759v1](http://arxiv.org/pdf/2505.22759v1)**

> **作者:** Sara Papi; Marco Gaido; Luisa Bentivogli; Alessio Brutti; Mauro Cettolo; Roberto Gretter; Marco Matassoni; Mohamed Nabih; Matteo Negri
>
> **摘要:** The development of speech foundation models (SFMs) like Whisper and SeamlessM4T has significantly advanced the field of speech processing. However, their closed nature--with inaccessible training data and code--poses major reproducibility and fair evaluation challenges. While other domains have made substantial progress toward open science by developing fully transparent models trained on open-source (OS) code and data, similar efforts in speech remain limited. To fill this gap, we introduce FAMA, the first family of open science SFMs for English and Italian, trained on 150k+ hours of OS speech data. Moreover, we present a new dataset containing 16k hours of cleaned and pseudo-labeled speech for both languages. Results show that FAMA achieves competitive performance compared to existing SFMs while being up to 8 times faster. All artifacts, including code, datasets, and models, are released under OS-compliant licenses, promoting openness in speech technology research.
>
---
#### [new 110] ARC: Argument Representation and Coverage Analysis for Zero-Shot Long Document Summarization with Instruction Following LLMs
- **分类: cs.CL**

- **简介: 该论文属于零样本长文档摘要任务，旨在解决LLM在保留关键论点信息上的不足。提出ARC框架评估LLM生成摘要对法律和科学领域中关键论点的覆盖，发现信息遗漏问题，并分析位置偏见和角色偏好影响，强调需改进策略。**

- **链接: [http://arxiv.org/pdf/2505.23654v1](http://arxiv.org/pdf/2505.23654v1)**

> **作者:** Mohamed Elaraby; Diane Litman
>
> **摘要:** Integrating structured information has long improved the quality of abstractive summarization, particularly in retaining salient content. In this work, we focus on a specific form of structure: argument roles, which are crucial for summarizing documents in high-stakes domains such as law. We investigate whether instruction-tuned large language models (LLMs) adequately preserve this information. To this end, we introduce Argument Representation Coverage (ARC), a framework for measuring how well LLM-generated summaries capture salient arguments. Using ARC, we analyze summaries produced by three open-weight LLMs in two domains where argument roles are central: long legal opinions and scientific articles. Our results show that while LLMs cover salient argument roles to some extent, critical information is often omitted in generated summaries, particularly when arguments are sparsely distributed throughout the input. Further, we use ARC to uncover behavioral patterns -- specifically, how the positional bias of LLM context windows and role-specific preferences impact the coverage of key arguments in generated summaries, emphasizing the need for more argument-aware summarization strategies.
>
---
#### [new 111] Sentinel: Attention Probing of Proxy Models for LLM Context Compression with an Understanding Perspective
- **分类: cs.CL; cs.AI**

- **简介: 论文提出Sentinel框架，通过探测代理模型的注意力信号实现LLM上下文压缩。针对现有压缩方法依赖监督训练成本高的问题，其利用轻量分类器解析0.5B代理模型的注意力权重筛选相关句子。实验显示其压缩率达5倍，QA性能媲美7B模型，证明无需专用压缩模型即可高效压缩。**

- **链接: [http://arxiv.org/pdf/2505.23277v1](http://arxiv.org/pdf/2505.23277v1)**

> **作者:** Yong Zhang; Yanwen Huang; Ning Cheng; Yang Guo; Yun Zhu; Yanmeng Wang; Shaojun Wang; Jing Xiao
>
> **备注:** Preprint. 17 pages including appendix
>
> **摘要:** Retrieval-augmented generation (RAG) enhances large language models (LLMs) with external context, but retrieved passages are often lengthy, noisy, or exceed input limits. Existing compression methods typically require supervised training of dedicated compression models, increasing cost and reducing portability. We propose Sentinel, a lightweight sentence-level compression framework that reframes context filtering as an attention-based understanding task. Rather than training a compression model, Sentinel probes decoder attention from an off-the-shelf 0.5B proxy LLM using a lightweight classifier to identify sentence relevance. Empirically, we find that query-context relevance estimation is consistent across model scales, with 0.5B proxies closely matching the behaviors of larger models. On the LongBench benchmark, Sentinel achieves up to 5$\times$ compression while matching the QA performance of 7B-scale compression systems. Our results suggest that probing native attention signals enables fast, effective, and question-aware context compression. Code available at: https://github.com/yzhangchuck/Sentinel.
>
---
#### [new 112] Verify-in-the-Graph: Entity Disambiguation Enhancement for Complex Claim Verification with Interactive Graph Representation
- **分类: cs.CL; cs.AI; cs.DB; cs.IR**

- **简介: 该论文属于声明验证任务，针对复杂声明中实体消歧不足的问题，提出VeGraph框架：通过图表示分解声明为三元组，迭代交互知识库消解模糊实体，最后验证剩余三元组。实验显示其在HoVer和FEVEROUS数据集上优于基线模型。**

- **链接: [http://arxiv.org/pdf/2505.22993v1](http://arxiv.org/pdf/2505.22993v1)**

> **作者:** Hoang Pham; Thanh-Do Nguyen; Khac-Hoai Nam Bui
>
> **备注:** Published at NAACL 2025 Main Conference
>
> **摘要:** Claim verification is a long-standing and challenging task that demands not only high accuracy but also explainability of the verification process. This task becomes an emerging research issue in the era of large language models (LLMs) since real-world claims are often complex, featuring intricate semantic structures or obfuscated entities. Traditional approaches typically address this by decomposing claims into sub-claims and querying a knowledge base to resolve hidden or ambiguous entities. However, the absence of effective disambiguation strategies for these entities can compromise the entire verification process. To address these challenges, we propose Verify-in-the-Graph (VeGraph), a novel framework leveraging the reasoning and comprehension abilities of LLM agents. VeGraph operates in three phases: (1) Graph Representation - an input claim is decomposed into structured triplets, forming a graph-based representation that integrates both structured and unstructured information; (2) Entity Disambiguation -VeGraph iteratively interacts with the knowledge base to resolve ambiguous entities within the graph for deeper sub-claim verification; and (3) Verification - remaining triplets are verified to complete the fact-checking process. Experiments using Meta-Llama-3-70B (instruct version) show that VeGraph achieves competitive performance compared to baselines on two benchmarks HoVer and FEVEROUS, effectively addressing claim verification challenges. Our source code and data are available for further exploitation.
>
---
#### [new 113] Infi-MMR: Curriculum-based Unlocking Multimodal Reasoning via Phased Reinforcement Learning in Multimodal Small Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Infi-MMR框架，针对多模态小模型推理能力弱、数据稀缺及视觉干扰等问题，设计三阶段课程式强化学习：先用文本数据激活基础推理，再用图文数据迁移推理技能，最后用无标注多模态数据增强跨模态推理。模型Infi-MMR-3B在数学推理等任务中达SOTA。**

- **链接: [http://arxiv.org/pdf/2505.23091v1](http://arxiv.org/pdf/2505.23091v1)**

> **作者:** Zeyu Liu; Yuhang Liu; Guanghao Zhu; Congkai Xie; Zhen Li; Jianbo Yuan; Xinyao Wang; Qing Li; Shing-Chi Cheung; Shengyu Zhang; Fei Wu; Hongxia Yang
>
> **摘要:** Recent advancements in large language models (LLMs) have demonstrated substantial progress in reasoning capabilities, such as DeepSeek-R1, which leverages rule-based reinforcement learning to enhance logical reasoning significantly. However, extending these achievements to multimodal large language models (MLLMs) presents critical challenges, which are frequently more pronounced for Multimodal Small Language Models (MSLMs) given their typically weaker foundational reasoning abilities: (1) the scarcity of high-quality multimodal reasoning datasets, (2) the degradation of reasoning capabilities due to the integration of visual processing, and (3) the risk that direct application of reinforcement learning may produce complex yet incorrect reasoning processes. To address these challenges, we design a novel framework Infi-MMR to systematically unlock the reasoning potential of MSLMs through a curriculum of three carefully structured phases and propose our multimodal reasoning model Infi-MMR-3B. The first phase, Foundational Reasoning Activation, leverages high-quality textual reasoning datasets to activate and strengthen the model's logical reasoning capabilities. The second phase, Cross-Modal Reasoning Adaptation, utilizes caption-augmented multimodal data to facilitate the progressive transfer of reasoning skills to multimodal contexts. The third phase, Multimodal Reasoning Enhancement, employs curated, caption-free multimodal data to mitigate linguistic biases and promote robust cross-modal reasoning. Infi-MMR-3B achieves both state-of-the-art multimodal math reasoning ability (43.68% on MathVerse testmini, 27.04% on MathVision test, and 21.33% on OlympiadBench) and general reasoning ability (67.2% on MathVista testmini).
>
---
#### [new 114] Be.FM: Open Foundation Models for Human Behavior
- **分类: cs.AI; cs.CE; cs.CL**

- **简介: 该论文属于人类行为建模任务，旨在探索基础模型在行为分析中的潜力。提出开源模型Be.FM，基于大语言模型微调行为数据，构建基准任务评估其预测行为、推断个体/群体特征及生成情境见解的能力，验证行为科学应用。**

- **链接: [http://arxiv.org/pdf/2505.23058v1](http://arxiv.org/pdf/2505.23058v1)**

> **作者:** Yutong Xie; Zhuoheng Li; Xiyuan Wang; Yijun Pan; Qijia Liu; Xingzhi Cui; Kuang-Yu Lo; Ruoyi Gao; Xingjian Zhang; Jin Huang; Walter Yuan; Matthew O. Jackson; Qiaozhu Mei
>
> **摘要:** Despite their success in numerous fields, the potential of foundation models for modeling and understanding human behavior remains largely unexplored. We introduce Be.FM, one of the first open foundation models designed for human behavior modeling. Built upon open-source large language models and fine-tuned on a diverse range of behavioral data, Be.FM can be used to understand and predict human decision-making. We construct a comprehensive set of benchmark tasks for testing the capabilities of behavioral foundation models. Our results demonstrate that Be.FM can predict behaviors, infer characteristics of individuals and populations, generate insights about contexts, and apply behavioral science knowledge.
>
---
#### [new 115] Enhancing Study-Level Inference from Clinical Trial Papers via RL-based Numeric Reasoning
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于临床试验系统综述自动化任务，旨在解决传统文本推理方法无法有效提取数值证据并推导研究结论的问题。提出基于强化学习（RL）的数值推理系统，通过提取结构化数据（如事件计数、标准差）并结合领域逻辑进行推断，在CochraneForest基准测试中F1值提升21%，优于大型语言模型。**

- **链接: [http://arxiv.org/pdf/2505.22928v1](http://arxiv.org/pdf/2505.22928v1)**

> **作者:** Massimiliano Pronesti; Michela Lorandi; Paul Flanagan; Oisin Redmon; Anya Belz; Yufang Hou
>
> **摘要:** Systematic reviews in medicine play a critical role in evidence-based decision-making by aggregating findings from multiple studies. A central bottleneck in automating this process is extracting numeric evidence and determining study-level conclusions for specific outcomes and comparisons. Prior work has framed this problem as a textual inference task by retrieving relevant content fragments and inferring conclusions from them. However, such approaches often rely on shallow textual cues and fail to capture the underlying numeric reasoning behind expert assessments. In this work, we conceptualise the problem as one of quantitative reasoning. Rather than inferring conclusions from surface text, we extract structured numerical evidence (e.g., event counts or standard deviations) and apply domain knowledge informed logic to derive outcome-specific conclusions. We develop a numeric reasoning system composed of a numeric data extraction model and an effect estimate component, enabling more accurate and interpretable inference aligned with the domain expert principles. We train the numeric data extraction model using different strategies, including supervised fine-tuning (SFT) and reinforcement learning (RL) with a new value reward model. When evaluated on the CochraneForest benchmark, our best-performing approach -- using RL to train a small-scale number extraction model -- yields up to a 21% absolute improvement in F1 score over retrieval-based systems and outperforms general-purpose LLMs of over 400B parameters by up to 9%. Our results demonstrate the promise of reasoning-driven approaches for automating systematic evidence synthesis.
>
---
#### [new 116] Segment Policy Optimization: Effective Segment-Level Credit Assignment in RL for Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出Segment Policy Optimization（SPO），优化强化学习在大型语言模型推理中的信用分配。针对现有方法在token级（估计不准）或轨迹级（信号粗糙）的缺陷，SPO通过分段级优势估计平衡两者，采用蒙特卡洛估计无需critic模型，并设计分段策略、概率掩码等方法。其SPO-chain和SPO-tree分别提升短/长推理任务在GSM8K和MATH500数据集6-12%及7-11%的准确率。**

- **链接: [http://arxiv.org/pdf/2505.23564v1](http://arxiv.org/pdf/2505.23564v1)**

> **作者:** Yiran Guo; Lijie Xu; Jie Liu; Dan Ye; Shuang Qiu
>
> **摘要:** Enhancing the reasoning capabilities of large language models effectively using reinforcement learning (RL) remains a crucial challenge. Existing approaches primarily adopt two contrasting advantage estimation granularities: Token-level methods (e.g., PPO) aim to provide the fine-grained advantage signals but suffer from inaccurate estimation due to difficulties in training an accurate critic model. On the other extreme, trajectory-level methods (e.g., GRPO) solely rely on a coarse-grained advantage signal from the final reward, leading to imprecise credit assignment. To address these limitations, we propose Segment Policy Optimization (SPO), a novel RL framework that leverages segment-level advantage estimation at an intermediate granularity, achieving a better balance by offering more precise credit assignment than trajectory-level methods and requiring fewer estimation points than token-level methods, enabling accurate advantage estimation based on Monte Carlo (MC) without a critic model. SPO features three components with novel strategies: (1) flexible segment partition; (2) accurate segment advantage estimation; and (3) policy optimization using segment advantages, including a novel probability-mask strategy. We further instantiate SPO for two specific scenarios: (1) SPO-chain for short chain-of-thought (CoT), featuring novel cutpoint-based partition and chain-based advantage estimation, achieving $6$-$12$ percentage point improvements in accuracy over PPO and GRPO on GSM8K. (2) SPO-tree for long CoT, featuring novel tree-based advantage estimation, which significantly reduces the cost of MC estimation, achieving $7$-$11$ percentage point improvements over GRPO on MATH500 under 2K and 4K context evaluation. We make our code publicly available at https://github.com/AIFrameResearch/SPO.
>
---
#### [new 117] Decomposing Elements of Problem Solving: What "Math" Does RL Teach?
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文分析强化学习（RL）对大语言模型（LLMs）数学推理能力的影响，提出将问题解决分解为计划、执行、验证三个要素。研究发现RL主要提升模型的执行鲁棒性（温度蒸馏），但其规划能力不足导致无法解决新问题（覆盖墙）。通过构建合成任务验证，提出改进探索和泛化可突破限制。**

- **链接: [http://arxiv.org/pdf/2505.22756v1](http://arxiv.org/pdf/2505.22756v1)**

> **作者:** Tian Qin; Core Francisco Park; Mujin Kwun; Aaron Walsman; Eran Malach; Nikhil Anand; Hidenori Tanaka; David Alvarez-Melis
>
> **摘要:** Mathematical reasoning tasks have become prominent benchmarks for assessing the reasoning capabilities of LLMs, especially with reinforcement learning (RL) methods such as GRPO showing significant performance gains. However, accuracy metrics alone do not support fine-grained assessment of capabilities and fail to reveal which problem-solving skills have been internalized. To better understand these capabilities, we propose to decompose problem solving into fundamental capabilities: Plan (mapping questions to sequences of steps), Execute (correctly performing solution steps), and Verify (identifying the correctness of a solution). Empirically, we find that GRPO mainly enhances the execution skill-improving execution robustness on problems the model already knows how to solve-a phenomenon we call temperature distillation. More importantly, we show that RL-trained models struggle with fundamentally new problems, hitting a 'coverage wall' due to insufficient planning skills. To explore RL's impact more deeply, we construct a minimal, synthetic solution-tree navigation task as an analogy for mathematical problem-solving. This controlled setup replicates our empirical findings, confirming RL primarily boosts execution robustness. Importantly, in this setting, we identify conditions under which RL can potentially overcome the coverage wall through improved exploration and generalization to new solution paths. Our findings provide insights into the role of RL in enhancing LLM reasoning, expose key limitations, and suggest a path toward overcoming these barriers. Code is available at https://github.com/cfpark00/RL-Wall.
>
---
#### [new 118] GSO: Challenging Software Optimization Tasks for Evaluating SWE-Agents
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出GSO基准，评估AI代理（SWE-Agents）的软件优化能力。针对现有模型在优化任务中的不足，通过分析10个代码库的102个优化案例，构建评测体系。发现顶尖模型成功率不足5%，主要失败原因包括低级语言处理困难、优化策略不足及瓶颈定位不准。研究公开了基准代码与数据，推动相关研究。**

- **链接: [http://arxiv.org/pdf/2505.23671v1](http://arxiv.org/pdf/2505.23671v1)**

> **作者:** Manish Shetty; Naman Jain; Jinjian Liu; Vijay Kethanaboyina; Koushik Sen; Ion Stoica
>
> **备注:** Website: https://gso-bench.github.io/
>
> **摘要:** Developing high-performance software is a complex task that requires specialized expertise. We introduce GSO, a benchmark for evaluating language models' capabilities in developing high-performance software. We develop an automated pipeline that generates and executes performance tests to analyze repository commit histories to identify 102 challenging optimization tasks across 10 codebases, spanning diverse domains and programming languages. An agent is provided with a codebase and performance test as a precise specification, and tasked to improve the runtime efficiency, which is measured against the expert developer optimization. Our quantitative evaluation reveals that leading SWE-Agents struggle significantly, achieving less than 5% success rate, with limited improvements even with inference-time scaling. Our qualitative analysis identifies key failure modes, including difficulties with low-level languages, practicing lazy optimization strategies, and challenges in accurately localizing bottlenecks. We release the code and artifacts of our benchmark along with agent trajectories to enable future research.
>
---
#### [new 119] On-Policy RL with Optimal Reward Baseline
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，针对训练不稳定和计算低效问题，提出OPO算法。通过精确on-policy训练和最优奖励基线降低梯度方差，提升大语言模型对齐与推理的稳定性，减少策略偏移并增强响应多样性。**

- **链接: [http://arxiv.org/pdf/2505.23585v1](http://arxiv.org/pdf/2505.23585v1)**

> **作者:** Yaru Hao; Li Dong; Xun Wu; Shaohan Huang; Zewen Chi; Furu Wei
>
> **摘要:** Reinforcement learning algorithms are fundamental to align large language models with human preferences and to enhance their reasoning capabilities. However, current reinforcement learning algorithms often suffer from training instability due to loose on-policy constraints and computational inefficiency due to auxiliary models. In this work, we propose On-Policy RL with Optimal reward baseline (OPO), a novel and simplified reinforcement learning algorithm designed to address these challenges. OPO emphasizes the importance of exact on-policy training, which empirically stabilizes the training process and enhances exploration. Moreover, OPO introduces the optimal reward baseline that theoretically minimizes gradient variance. We evaluate OPO on mathematical reasoning benchmarks. The results demonstrate its superior performance and training stability without additional models or regularization terms. Furthermore, OPO achieves lower policy shifts and higher output entropy, encouraging more diverse and less repetitive responses. These results highlight OPO as a promising direction for stable and effective reinforcement learning in large language model alignment and reasoning tasks. The implementation is provided at https://github.com/microsoft/LMOps/tree/main/opo.
>
---
#### [new 120] VF-Eval: Evaluating Multimodal LLMs for Generating Feedback on AIGC Videos
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出VF-Eval基准，评估多模态LLMs在AIGC视频反馈中的能力，填补现有评估忽视合成视频的空白。通过四个任务（连贯性验证、错误识别等）测试13个模型，发现GPT-4.1表现仍不均衡，证明基准挑战性，并通过RePrompt实验验证其改进视频生成的潜力。**

- **链接: [http://arxiv.org/pdf/2505.23693v1](http://arxiv.org/pdf/2505.23693v1)**

> **作者:** Tingyu Song; Tongyan Hu; Guo Gan; Yilun Zhao
>
> **备注:** ACL 2025 Main
>
> **摘要:** MLLMs have been widely studied for video question answering recently. However, most existing assessments focus on natural videos, overlooking synthetic videos, such as AI-generated content (AIGC). Meanwhile, some works in video generation rely on MLLMs to evaluate the quality of generated videos, but the capabilities of MLLMs on interpreting AIGC videos remain largely underexplored. To address this, we propose a new benchmark, VF-Eval, which introduces four tasks-coherence validation, error awareness, error type detection, and reasoning evaluation-to comprehensively evaluate the abilities of MLLMs on AIGC videos. We evaluate 13 frontier MLLMs on VF-Eval and find that even the best-performing model, GPT-4.1, struggles to achieve consistently good performance across all tasks. This highlights the challenging nature of our benchmark. Additionally, to investigate the practical applications of VF-Eval in improving video generation, we conduct an experiment, RePrompt, demonstrating that aligning MLLMs more closely with human feedback can benefit video generation.
>
---
#### [new 121] R2I-Bench: Benchmarking Reasoning-Driven Text-to-Image Generation
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出R2I-Bench基准，评估文本到图像生成模型的推理能力。针对现有模型推理不足问题，构建覆盖常识、数学等推理类别的评测数据集，并设计R2IScore指标，通过问答形式评估文本-图像对齐、推理准确性和图像质量。实验显示当前模型推理性能有限，强调需开发更强大的推理感知架构。**

- **链接: [http://arxiv.org/pdf/2505.23493v1](http://arxiv.org/pdf/2505.23493v1)**

> **作者:** Kaijie Chen; Zihao Lin; Zhiyang Xu; Ying Shen; Yuguang Yao; Joy Rimchala; Jiaxin Zhang; Lifu Huang
>
> **备注:** Project Page: https://r2i-bench.github.io
>
> **摘要:** Reasoning is a fundamental capability often required in real-world text-to-image (T2I) generation, e.g., generating ``a bitten apple that has been left in the air for more than a week`` necessitates understanding temporal decay and commonsense concepts. While recent T2I models have made impressive progress in producing photorealistic images, their reasoning capability remains underdeveloped and insufficiently evaluated. To bridge this gap, we introduce R2I-Bench, a comprehensive benchmark specifically designed to rigorously assess reasoning-driven T2I generation. R2I-Bench comprises meticulously curated data instances, spanning core reasoning categories, including commonsense, mathematical, logical, compositional, numerical, causal, and concept mixing. To facilitate fine-grained evaluation, we design R2IScore, a QA-style metric based on instance-specific, reasoning-oriented evaluation questions that assess three critical dimensions: text-image alignment, reasoning accuracy, and image quality. Extensive experiments with 16 representative T2I models, including a strong pipeline-based framework that decouples reasoning and generation using the state-of-the-art language and image generation models, demonstrate consistently limited reasoning performance, highlighting the need for more robust, reasoning-aware architectures in the next generation of T2I systems. Project Page: https://r2i-bench.github.io
>
---
#### [new 122] Jigsaw-R1: A Study of Rule-based Visual Reinforcement Learning with Jigsaw Puzzles
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究多模态大语言模型（MLLMs）的规则化视觉强化学习（RL），以拼图任务为实验框架，解决其在感知任务中的表现与泛化问题。通过分析模型在拼图任务中的微调、任务迁移、推理模式及训练策略，揭示RL优于监督微调（SFT）的泛化能力，并探讨模型推理机制与预存复杂模式的影响。**

- **链接: [http://arxiv.org/pdf/2505.23590v1](http://arxiv.org/pdf/2505.23590v1)**

> **作者:** Zifu Wang; Junyi Zhu; Bo Tang; Zhiyu Li; Feiyu Xiong; Jiaqian Yu; Matthew B. Blaschko
>
> **摘要:** The application of rule-based reinforcement learning (RL) to multimodal large language models (MLLMs) introduces unique challenges and potential deviations from findings in text-only domains, particularly for perception-heavy tasks. This paper provides a comprehensive study of rule-based visual RL using jigsaw puzzles as a structured experimental framework, revealing several key findings. \textit{Firstly,} we find that MLLMs, initially performing near to random guessing on simple puzzles, achieve near-perfect accuracy and generalize to complex, unseen configurations through fine-tuning. \textit{Secondly,} training on jigsaw puzzles can induce generalization to other visual tasks, with effectiveness tied to specific task configurations. \textit{Thirdly,} MLLMs can learn and generalize with or without explicit reasoning, though open-source models often favor direct answering. Consequently, even when trained for step-by-step reasoning, they can ignore the thinking process in deriving the final answer. \textit{Fourthly,} we observe that complex reasoning patterns appear to be pre-existing rather than emergent, with their frequency increasing alongside training and task difficulty. \textit{Finally,} our results demonstrate that RL exhibits more effective generalization than Supervised Fine-Tuning (SFT), and an initial SFT cold start phase can hinder subsequent RL optimization. Although these observations are based on jigsaw puzzles and may vary across other visual tasks, this research contributes a valuable piece of jigsaw to the larger puzzle of collective understanding rule-based visual RL and its potential in multimodal learning. The code is available at: \href{https://github.com/zifuwanggg/Jigsaw-R1}{https://github.com/zifuwanggg/Jigsaw-R1}.
>
---
#### [new 123] Identity resolution of software metadata using Large Language Models
- **分类: cs.SE; cs.CL; cs.DL**

- **简介: 该论文属于软件元数据身份解析任务，旨在解决跨平台软件元数据异构性和整合难题。通过评估指令调优的大型语言模型在该任务中的表现，测试其处理模糊案例的能力，并提出基于共识的自动化决策代理，以提升FAIR原则下生命科学领域研究软件元数据的整合精度和效率。**

- **链接: [http://arxiv.org/pdf/2505.23500v1](http://arxiv.org/pdf/2505.23500v1)**

> **作者:** Eva Martín del Pico; Josep Lluís Gelpí; Salvador Capella-Gutiérrez
>
> **摘要:** Software is an essential component of research. However, little attention has been paid to it compared with that paid to research data. Recently, there has been an increase in efforts to acknowledge and highlight the importance of software in research activities. Structured metadata from platforms like bio.tools, Bioconductor, and Galaxy ToolShed offers valuable insights into research software in the Life Sciences. Although originally intended to support discovery and integration, this metadata can be repurposed for large-scale analysis of software practices. However, its quality and completeness vary across platforms, reflecting diverse documentation practices. To gain a comprehensive view of software development and sustainability, consolidating this metadata is necessary, but requires robust mechanisms to address its heterogeneity and scale. This article presents an evaluation of instruction-tuned large language models for the task of software metadata identity resolution, a critical step in assembling a cohesive collection of research software. Such a collection is the reference component for the Software Observatory at OpenEBench, a platform that aggregates metadata to monitor the FAIRness of research software in the Life Sciences. We benchmarked multiple models against a human-annotated gold standard, examined their behavior on ambiguous cases, and introduced an agreement-based proxy for high-confidence automated decisions. The proxy achieved high precision and statistical robustness, while also highlighting the limitations of current models and the broader challenges of automating semantic judgment in FAIR-aligned software metadata across registries and repositories.
>
---
#### [new 124] Human Empathy as Encoder: AI-Assisted Depression Assessment in Special Education
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文提出HEAE框架，解决特殊教育中AI抑郁评估不足的问题。通过融合学生叙述文本与教师生成的9维同理心向量（基于PHQ-9），结合人机协同提升评估准确性（7级分类达82.74%），推动伦理导向的情感计算。**

- **链接: [http://arxiv.org/pdf/2505.23631v1](http://arxiv.org/pdf/2505.23631v1)**

> **作者:** Boning Zhao
>
> **备注:** 7 pages, 6 figures. Under review
>
> **摘要:** Assessing student depression in sensitive environments like special education is challenging. Standardized questionnaires may not fully reflect students' true situations. Furthermore, automated methods often falter with rich student narratives, lacking the crucial, individualized insights stemming from teachers' empathetic connections with students. Existing methods often fail to address this ambiguity or effectively integrate educator understanding. To address these limitations by fostering a synergistic human-AI collaboration, this paper introduces Human Empathy as Encoder (HEAE), a novel, human-centered AI framework for transparent and socially responsible depression severity assessment. Our approach uniquely integrates student narrative text with a teacher-derived, 9-dimensional "Empathy Vector" (EV), its dimensions guided by the PHQ-9 framework,to explicitly translate tacit empathetic insight into a structured AI input enhancing rather than replacing human judgment. Rigorous experiments optimized the multimodal fusion, text representation, and classification architecture, achieving 82.74% accuracy for 7-level severity classification. This work demonstrates a path toward more responsible and ethical affective computing by structurally embedding human empathy
>
---
#### [new 125] Domain-Aware Tensor Network Structure Search
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于张量网络结构搜索任务，旨在解决现有方法计算成本高、忽略领域信息及缺乏可解释性的问题。提出tnLLM框架，结合领域知识与大语言模型推理，通过领域感知提示管道直接预测张量结构，并生成解释。实验显示其以更少计算达最优效果，还可加速现有方法收敛。**

- **链接: [http://arxiv.org/pdf/2505.23537v1](http://arxiv.org/pdf/2505.23537v1)**

> **作者:** Giorgos Iacovides; Wuyang Zhou; Chao Li; Qibin Zhao; Danilo Mandic
>
> **摘要:** Tensor networks (TNs) provide efficient representations of high-dimensional data, yet identification of the optimal TN structures, the so called tensor network structure search (TN-SS) problem, remains a challenge. Current state-of-the-art (SOTA) algorithms are computationally expensive as they require extensive function evaluations, which is prohibitive for real-world applications. In addition, existing methods ignore valuable domain information inherent in real-world tensor data and lack transparency in their identified TN structures. To this end, we propose a novel TN-SS framework, termed the tnLLM, which incorporates domain information about the data and harnesses the reasoning capabilities of large language models (LLMs) to directly predict suitable TN structures. The proposed framework involves a domain-aware prompting pipeline which instructs the LLM to infer suitable TN structures based on the real-world relationships between tensor modes. In this way, our approach is capable of not only iteratively optimizing the objective function, but also generating domain-aware explanations for the identified structures. Experimental results demonstrate that tnLLM achieves comparable TN-SS objective function values with much fewer function evaluations compared to SOTA algorithms. Furthermore, we demonstrate that the LLM-enabled domain information can be used to find good initializations in the search space for sampling-based SOTA methods to accelerate their convergence while preserving theoretical performance guarantees.
>
---
#### [new 126] Nosey: Open-source hardware for acoustic nasalance
- **分类: cs.SD; cs.CL**

- **简介: 该论文提出开源低成本硬件Nosey，解决商用鼻腔共鸣检测设备昂贵且不可定制的问题。通过对比实验验证其与商用设备的一致性，探讨麦克风及材料定制方法，证明其为灵活经济的替代方案。（99字）**

- **链接: [http://arxiv.org/pdf/2505.23339v1](http://arxiv.org/pdf/2505.23339v1)**

> **作者:** Maya Dewhurst; Jack Collins; Justin J. H. Lo; Roy Alderton; Sam Kirkham
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** We introduce Nosey (Nasalance Open Source Estimation sYstem), a low-cost, customizable, 3D-printed system for recording acoustic nasalance data that we have made available as open-source hardware (http://github.com/phoneticslab/nosey). We first outline the motivations and design principles behind our hardware nasalance system, and then present a comparison between Nosey and a commercial nasalance device. Nosey shows consistently higher nasalance scores than the commercial device, but the magnitude of contrast between phonological environments is comparable between systems. We also review ways of customizing the hardware to facilitate testing, such as comparison of microphones and different construction materials. We conclude that Nosey is a flexible and cost-effective alternative to commercial nasometry devices and propose some methodological considerations for its use in data collection.
>
---
#### [new 127] Does Machine Unlearning Truly Remove Model Knowledge? A Framework for Auditing Unlearning in LLMs
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于机器unlearning（模型遗忘）审计任务，解决LLMs删除特定知识效果评估难题。提出含3个数据集、6种算法及5种提示方法的审计框架，并引入基于中间激活扰动的新技术，提升评估鲁棒性。**

- **链接: [http://arxiv.org/pdf/2505.23270v1](http://arxiv.org/pdf/2505.23270v1)**

> **作者:** Haokun Chen; Yueqi Zhang; Yuan Bi; Yao Zhang; Tong Liu; Jinhe Bi; Jian Lan; Jindong Gu; Claudia Grosser; Denis Krompass; Nassir Navab; Volker Tresp
>
> **摘要:** In recent years, Large Language Models (LLMs) have achieved remarkable advancements, drawing significant attention from the research community. Their capabilities are largely attributed to large-scale architectures, which require extensive training on massive datasets. However, such datasets often contain sensitive or copyrighted content sourced from the public internet, raising concerns about data privacy and ownership. Regulatory frameworks, such as the General Data Protection Regulation (GDPR), grant individuals the right to request the removal of such sensitive information. This has motivated the development of machine unlearning algorithms that aim to remove specific knowledge from models without the need for costly retraining. Despite these advancements, evaluating the efficacy of unlearning algorithms remains a challenge due to the inherent complexity and generative nature of LLMs. In this work, we introduce a comprehensive auditing framework for unlearning evaluation, comprising three benchmark datasets, six unlearning algorithms, and five prompt-based auditing methods. By using various auditing algorithms, we evaluate the effectiveness and robustness of different unlearning strategies. To explore alternatives beyond prompt-based auditing, we propose a novel technique that leverages intermediate activation perturbations, addressing the limitations of auditing methods that rely solely on model inputs and outputs.
>
---
#### [new 128] VScan: Rethinking Visual Token Reduction for Efficient Large Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型（LVLMs）效率优化任务，旨在解决细粒度视觉编码导致计算成本过高问题。提出VScan框架，在视觉编码阶段通过全局/局部扫描合并冗余token，并在语言模型中间层进行剪枝，实现加速推理（如LLaVA模型预填速提升2.91倍，计算量降90%）同时保持性能。**

- **链接: [http://arxiv.org/pdf/2505.22654v1](http://arxiv.org/pdf/2505.22654v1)**

> **作者:** Ce Zhang; Kaixin Ma; Tianqing Fang; Wenhao Yu; Hongming Zhang; Zhisong Zhang; Yaqi Xie; Katia Sycara; Haitao Mi; Dong Yu
>
> **摘要:** Recent Large Vision-Language Models (LVLMs) have advanced multi-modal understanding by incorporating finer-grained visual perception and encoding. However, such methods incur significant computational costs due to longer visual token sequences, posing challenges for real-time deployment. To mitigate this, prior studies have explored pruning unimportant visual tokens either at the output layer of the visual encoder or at the early layers of the language model. In this work, we revisit these design choices and reassess their effectiveness through comprehensive empirical studies of how visual tokens are processed throughout the visual encoding and language decoding stages. Guided by these insights, we propose VScan, a two-stage visual token reduction framework that addresses token redundancy by: (1) integrating complementary global and local scans with token merging during visual encoding, and (2) introducing pruning at intermediate layers of the language model. Extensive experimental results across four LVLMs validate the effectiveness of VScan in accelerating inference and demonstrate its superior performance over current state-of-the-arts on sixteen benchmarks. Notably, when applied to LLaVA-NeXT-7B, VScan achieves a 2.91$\times$ speedup in prefilling and a 10$\times$ reduction in FLOPs, while retaining 95.4% of the original performance.
>
---
#### [new 129] Cultural Evaluations of Vision-Language Models Have a Lot to Learn from Cultural Theory
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型（VLM）文化分析任务，针对其在文化评估中表现不足的问题，提出基于文化理论的五个框架，系统识别图像中的文化维度，以完善VLM的文化能力评估。**

- **链接: [http://arxiv.org/pdf/2505.22793v1](http://arxiv.org/pdf/2505.22793v1)**

> **作者:** Srishti Yadav; Lauren Tilton; Maria Antoniak; Taylor Arnold; Jiaang Li; Siddhesh Milind Pawar; Antonia Karamolegkou; Stella Frank; Zhaochong An; Negar Rostamzadeh; Daniel Hershcovich; Serge Belongie; Ekaterina Shutova
>
> **摘要:** Modern vision-language models (VLMs) often fail at cultural competency evaluations and benchmarks. Given the diversity of applications built upon VLMs, there is renewed interest in understanding how they encode cultural nuances. While individual aspects of this problem have been studied, we still lack a comprehensive framework for systematically identifying and annotating the nuanced cultural dimensions present in images for VLMs. This position paper argues that foundational methodologies from visual culture studies (cultural studies, semiotics, and visual studies) are necessary for cultural analysis of images. Building upon this review, we propose a set of five frameworks, corresponding to cultural dimensions, that must be considered for a more complete analysis of the cultural competencies of VLMs.
>
---
#### [new 130] Large Language Models for Depression Recognition in Spoken Language Integrating Psychological Knowledge
- **分类: cs.HC; cs.CL**

- **简介: 该论文属于多模态抑郁检测任务，旨在解决大语言模型（LLM）在非文本语音线索处理及心理知识整合上的不足。研究结合Wav2Vec提取音频特征与LLM文本分析，创新性地通过心理知识问答集增强模型诊断能力，在DAIC-WOZ数据集上显著降低了MAE和RMSE误差。**

- **链接: [http://arxiv.org/pdf/2505.22863v1](http://arxiv.org/pdf/2505.22863v1)**

> **作者:** Yupei Li; Shuaijie Shao; Manuel Milling; Björn W. Schuller
>
> **摘要:** Depression is a growing concern gaining attention in both public discourse and AI research. While deep neural networks (DNNs) have been used for recognition, they still lack real-world effectiveness. Large language models (LLMs) show strong potential but require domain-specific fine-tuning and struggle with non-textual cues. Since depression is often expressed through vocal tone and behaviour rather than explicit text, relying on language alone is insufficient. Diagnostic accuracy also suffers without incorporating psychological expertise. To address these limitations, we present, to the best of our knowledge, the first application of LLMs to multimodal depression detection using the DAIC-WOZ dataset. We extract the audio features using the pre-trained model Wav2Vec, and mapped it to text-based LLMs for further processing. We also propose a novel strategy for incorporating psychological knowledge into LLMs to enhance diagnostic performance, specifically using a question and answer set to grant authorised knowledge to LLMs. Our approach yields a notable improvement in both Mean Absolute Error (MAE) and Root Mean Square Error (RMSE) compared to a base score proposed by the related original paper. The codes are available at https://github.com/myxp-lyp/Depression-detection.git
>
---
#### [new 131] TailorSQL: An NL2SQL System Tailored to Your Query Workload
- **分类: cs.DB; cs.CL**

- **简介: 论文提出TailorSQL，属NL2SQL任务。针对现有方法忽略历史查询数据的问题，其利用过往SQL工作负载中的隐含信息（如常用连接路径、表列语义），优化SQL生成的准确率与速度，在基准测试中提升达2倍执行精度。**

- **链接: [http://arxiv.org/pdf/2505.23039v1](http://arxiv.org/pdf/2505.23039v1)**

> **作者:** Kapil Vaidya; Jialin Ding; Sebastian Kosak; David Kernert; Chuan Lei; Xiao Qin; Abhinav Tripathy; Ramesh Balan; Balakrishnan Narayanaswamy; Tim Kraska
>
> **摘要:** NL2SQL (natural language to SQL) translates natural language questions into SQL queries, thereby making structured data accessible to non-technical users, serving as the foundation for intelligent data applications. State-of-the-art NL2SQL techniques typically perform translation by retrieving database-specific information, such as the database schema, and invoking a pre-trained large language model (LLM) using the question and retrieved information to generate the SQL query. However, existing NL2SQL techniques miss a key opportunity which is present in real-world settings: NL2SQL is typically applied on existing databases which have already served many SQL queries in the past. The past query workload implicitly contains information which is helpful for accurate NL2SQL translation and is not apparent from the database schema alone, such as common join paths and the semantics of obscurely-named tables and columns. We introduce TailorSQL, a NL2SQL system that takes advantage of information in the past query workload to improve both the accuracy and latency of translating natural language questions into SQL. By specializing to a given workload, TailorSQL achieves up to 2$\times$ improvement in execution accuracy on standardized benchmarks.
>
---
#### [new 132] Synthetic Document Question Answering in Hungarian
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文聚焦匈牙利语文档视觉问答任务，针对其数据稀缺问题，构建了合成数据集HuDocVQA、人工精筛数据集HuDocVQA-manual及OCR训练数据集HuCCPDF，通过质量过滤提升数据质量，实验表明微调后模型性能显著提升，推动多语言文档问答研究。**

- **链接: [http://arxiv.org/pdf/2505.23008v1](http://arxiv.org/pdf/2505.23008v1)**

> **作者:** Jonathan Li; Zoltan Csaki; Nidhi Hiremath; Etash Guha; Fenglu Hong; Edward Ma; Urmish Thakker
>
> **摘要:** Modern VLMs have achieved near-saturation accuracy in English document visual question-answering (VQA). However, this task remains challenging in lower resource languages due to a dearth of suitable training and evaluation data. In this paper we present scalable methods for curating such datasets by focusing on Hungarian, approximately the 17th highest resource language on the internet. Specifically, we present HuDocVQA and HuDocVQA-manual, document VQA datasets that modern VLMs significantly underperform on compared to English DocVQA. HuDocVQA-manual is a small manually curated dataset based on Hungarian documents from Common Crawl, while HuDocVQA is a larger synthetically generated VQA data set from the same source. We apply multiple rounds of quality filtering and deduplication to HuDocVQA in order to match human-level quality in this dataset. We also present HuCCPDF, a dataset of 117k pages from Hungarian Common Crawl PDFs along with their transcriptions, which can be used for training a model for Hungarian OCR. To validate the quality of our datasets, we show how finetuning on a mixture of these datasets can improve accuracy on HuDocVQA for Llama 3.2 11B Instruct by +7.2%. Our datasets and code will be released to the public to foster further research in multilingual DocVQA.
>
---
#### [new 133] Conversational Alignment with Artificial Intelligence in Context
- **分类: cs.CY; cs.CL**

- **简介: 该论文探讨AI对话系统与人类交流规范的对齐问题，提出CONTEXT-ALIGN框架评估设计选择，分析LLM架构对实现对话对齐的限制，属AI伦理与对话系统优化任务。**

- **链接: [http://arxiv.org/pdf/2505.22907v1](http://arxiv.org/pdf/2505.22907v1)**

> **作者:** Rachel Katharine Sterken; James Ravi Kirkpatrick
>
> **备注:** 20 pages, to be published in Philosophical Perspectives
>
> **摘要:** The development of sophisticated artificial intelligence (AI) conversational agents based on large language models raises important questions about the relationship between human norms, values, and practices and AI design and performance. This article explores what it means for AI agents to be conversationally aligned to human communicative norms and practices for handling context and common ground and proposes a new framework for evaluating developers' design choices. We begin by drawing on the philosophical and linguistic literature on conversational pragmatics to motivate a set of desiderata, which we call the CONTEXT-ALIGN framework, for conversational alignment with human communicative practices. We then suggest that current large language model (LLM) architectures, constraints, and affordances may impose fundamental limitations on achieving full conversational alignment.
>
---
#### [new 134] Socratic-PRMBench: Benchmarking Process Reward Models with Systematic Reasoning Patterns
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Socratic-PRMBench基准，用于系统评估过程奖励模型（PRMs）在六类推理模式（分解、转化等）中的表现。针对现有评估忽视多模式推理缺陷的问题，构建含2995条缺陷推理路径的测试集，实验显示当前PRMs在多模式推理评估中存在显著不足。**

- **链接: [http://arxiv.org/pdf/2505.23474v1](http://arxiv.org/pdf/2505.23474v1)**

> **作者:** Xiang Li; Haiyang Yu; Xinghua Zhang; Ziyang Huang; Shizhu He; Kang Liu; Jun Zhao; Fei Huang; Yongbin Li
>
> **摘要:** Process Reward Models (PRMs) are crucial in complex reasoning and problem-solving tasks (e.g., LLM agents with long-horizon decision-making) by verifying the correctness of each intermediate reasoning step. In real-world scenarios, LLMs may apply various reasoning patterns (e.g., decomposition) to solve a problem, potentially suffering from errors under various reasoning patterns. Therefore, PRMs are required to identify errors under various reasoning patterns during the reasoning process. However, existing benchmarks mainly focus on evaluating PRMs with stepwise correctness, ignoring a systematic evaluation of PRMs under various reasoning patterns. To mitigate this gap, we introduce Socratic-PRMBench, a new benchmark to evaluate PRMs systematically under six reasoning patterns, including Transformation, Decomposition, Regather, Deduction, Verification, and Integration. Socratic-PRMBench}comprises 2995 reasoning paths with flaws within the aforementioned six reasoning patterns. Through our experiments on both PRMs and LLMs prompted as critic models, we identify notable deficiencies in existing PRMs. These observations underscore the significant weakness of current PRMs in conducting evaluations on reasoning steps under various reasoning patterns. We hope Socratic-PRMBench can serve as a comprehensive testbed for systematic evaluation of PRMs under diverse reasoning patterns and pave the way for future development of PRMs.
>
---
#### [new 135] NGPU-LM: GPU-Accelerated N-Gram Language Model for Context-Biasing in Greedy ASR Decoding
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **简介: 该论文针对ASR上下文偏置中n-gram模型并行化不足的问题，提出GPU优化的NGPU-LM。通过重新设计数据结构，支持多种ASR模型的高效贪心解码，计算开销<7%，减少域外场景准确率差距超50%，开源实现。**

- **链接: [http://arxiv.org/pdf/2505.22857v1](http://arxiv.org/pdf/2505.22857v1)**

> **作者:** Vladimir Bataev; Andrei Andrusenko; Lilit Grigoryan; Aleksandr Laptev; Vitaly Lavrukhin; Boris Ginsburg
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Statistical n-gram language models are widely used for context-biasing tasks in Automatic Speech Recognition (ASR). However, existing implementations lack computational efficiency due to poor parallelization, making context-biasing less appealing for industrial use. This work rethinks data structures for statistical n-gram language models to enable fast and parallel operations for GPU-optimized inference. Our approach, named NGPU-LM, introduces customizable greedy decoding for all major ASR model types - including transducers, attention encoder-decoder models, and CTC - with less than 7% computational overhead. The proposed approach can eliminate more than 50% of the accuracy gap between greedy and beam search for out-of-domain scenarios while avoiding significant slowdown caused by beam search. The implementation of the proposed NGPU-LM is open-sourced.
>
---
#### [new 136] ZeroGUI: Automating Online GUI Learning at Zero Human Cost
- **分类: cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出ZeroGUI框架，解决现有GUI代理依赖人工标注及适应性差的问题，通过VLM自动生成任务、评估奖励及两阶段在线强化学习，在零人工成本下提升OSWorld和AndroidLab环境性能。**

- **链接: [http://arxiv.org/pdf/2505.23762v1](http://arxiv.org/pdf/2505.23762v1)**

> **作者:** Chenyu Yang; Shiqian Su; Shi Liu; Xuan Dong; Yue Yu; Weijie Su; Xuehui Wang; Zhaoyang Liu; Jinguo Zhu; Hao Li; Wenhai Wang; Yu Qiao; Xizhou Zhu; Jifeng Dai
>
> **摘要:** The rapid advancement of large Vision-Language Models (VLMs) has propelled the development of pure-vision-based GUI Agents, capable of perceiving and operating Graphical User Interfaces (GUI) to autonomously fulfill user instructions. However, existing approaches usually adopt an offline learning framework, which faces two core limitations: (1) heavy reliance on high-quality manual annotations for element grounding and action supervision, and (2) limited adaptability to dynamic and interactive environments. To address these limitations, we propose ZeroGUI, a scalable, online learning framework for automating GUI Agent training at Zero human cost. Specifically, ZeroGUI integrates (i) VLM-based automatic task generation to produce diverse training goals from the current environment state, (ii) VLM-based automatic reward estimation to assess task success without hand-crafted evaluation functions, and (iii) two-stage online reinforcement learning to continuously interact with and learn from GUI environments. Experiments on two advanced GUI Agents (UI-TARS and Aguvis) demonstrate that ZeroGUI significantly boosts performance across OSWorld and AndroidLab environments. The code is available at https://github.com/OpenGVLab/ZeroGUI.
>
---
#### [new 137] Rethinking Regularization Methods for Knowledge Graph Completion
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于知识图谱补全（KGC）任务，旨在解决传统正则化方法未深度挖掘潜力的问题。通过重新设计正则化策略，提出基于选择性稀疏的SPR方法，针对性惩罚嵌入向量的关键特征，抑制噪声。实验表明该方法有效缓解过拟合并突破原有性能上限。**

- **链接: [http://arxiv.org/pdf/2505.23442v1](http://arxiv.org/pdf/2505.23442v1)**

> **作者:** Linyu Li; Zhi Jin; Yuanpeng He; Dongming Jin; Haoran Duan; Zhengwei Tao; Xuan Zhang; Jiandong Li
>
> **摘要:** Knowledge graph completion (KGC) has attracted considerable attention in recent years because it is critical to improving the quality of knowledge graphs. Researchers have continuously explored various models. However, most previous efforts have neglected to take advantage of regularization from a deeper perspective and therefore have not been used to their full potential. This paper rethinks the application of regularization methods in KGC. Through extensive empirical studies on various KGC models, we find that carefully designed regularization not only alleviates overfitting and reduces variance but also enables these models to break through the upper bounds of their original performance. Furthermore, we introduce a novel sparse-regularization method that embeds the concept of rank-based selective sparsity into the KGC regularizer. The core idea is to selectively penalize those components with significant features in the embedding vector, thus effectively ignoring many components that contribute little and may only represent noise. Various comparative experiments on multiple datasets and multiple models show that the SPR regularization method is better than other regularization methods and can enable the KGC model to further break through the performance margin.
>
---
#### [new 138] AgentAlign: Navigating Safety Alignment in the Shift from Informative to Agentic Large Language Models
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文聚焦大语言模型安全对齐任务，针对代理型模型易执行恶意任务的问题，提出AgentAlign框架。通过抽象行为链在模拟环境中生成安全指令数据，平衡安全与实用性。实验表明其显著提升模型安全性（35.8%-79.5%），同时保持或增强有用性，优于提示方法。**

- **链接: [http://arxiv.org/pdf/2505.23020v1](http://arxiv.org/pdf/2505.23020v1)**

> **作者:** Jinchuan Zhang; Lu Yin; Yan Zhou; Songlin Hu
>
> **备注:** Submitted to ACL 2025
>
> **摘要:** The acquisition of agentic capabilities has transformed LLMs from "knowledge providers" to "action executors", a trend that while expanding LLMs' capability boundaries, significantly increases their susceptibility to malicious use. Previous work has shown that current LLM-based agents execute numerous malicious tasks even without being attacked, indicating a deficiency in agentic use safety alignment during the post-training phase. To address this gap, we propose AgentAlign, a novel framework that leverages abstract behavior chains as a medium for safety alignment data synthesis. By instantiating these behavior chains in simulated environments with diverse tool instances, our framework enables the generation of highly authentic and executable instructions while capturing complex multi-step dynamics. The framework further ensures model utility by proportionally synthesizing benign instructions through non-malicious interpretations of behavior chains, precisely calibrating the boundary between helpfulness and harmlessness. Evaluation results on AgentHarm demonstrate that fine-tuning three families of open-source models using our method substantially improves their safety (35.8% to 79.5% improvement) while minimally impacting or even positively enhancing their helpfulness, outperforming various prompting methods. The dataset and code have both been open-sourced.
>
---
#### [new 139] SWE-bench Goes Live!
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文提出动态代码修复基准SWE-bench-Live，解决传统基准静态化、覆盖窄、依赖人工的缺陷。通过自动化流水线持续收集2024年后真实GitHub issue（1319任务/93仓库），配套Docker环境确保可复现，揭示模型在实时任务中的性能差距，助力LLM在动态开发场景的评估。**

- **链接: [http://arxiv.org/pdf/2505.23419v1](http://arxiv.org/pdf/2505.23419v1)**

> **作者:** Linghao Zhang; Shilin He; Chaoyun Zhang; Yu Kang; Bowen Li; Chengxing Xie; Junhao Wang; Maoquan Wang; Yufan Huang; Shengyu Fu; Elsie Nallipogu; Qingwei Lin; Yingnong Dang; Saravan Rajmohan; Dongmei Zhang
>
> **备注:** Homepage: \url{https://swe-bench-live.github.io/}, Code: \url{https://github.com/SWE-bench-Live}, Dataset: \url{https://huggingface.co/SWE-bench-Live}
>
> **摘要:** The issue-resolving task, where a model generates patches to fix real-world bugs, has emerged as a critical benchmark for evaluating the capabilities of large language models (LLMs). While SWE-bench and its variants have become standard in this domain, they suffer from key limitations: they have not been updated since their initial releases, cover a narrow set of repositories, and depend heavily on manual effort for instance construction and environment setup. These factors hinder scalability and introduce risks of overfitting and data contamination. In this work, we present \textbf{SWE-bench-Live}, a \textit{live-updatable} benchmark designed to overcome these challenges. Our initial release consists of 1,319 tasks derived from real GitHub issues created since 2024, spanning 93 repositories. Each task is accompanied by a dedicated Docker image to ensure reproducible execution. Central to our benchmark is \method, an automated curation pipeline that streamlines the entire process from instance creation to environment setup, removing manual bottlenecks and enabling scalability and continuous updates. We evaluate a range of state-of-the-art agent frameworks and LLMs on SWE-bench-Live, revealing a substantial performance gap compared to static benchmarks like SWE-bench, even under controlled evaluation conditions. To better understand this discrepancy, we perform detailed analyses across repository origin, issue recency, and task difficulty. By providing a fresh, diverse, and executable benchmark grounded in live repository activity, SWE-bench-Live facilitates rigorous, contamination-resistant evaluation of LLMs and agents in dynamic, real-world software development settings.
>
---
#### [new 140] MMSI-Bench: A Benchmark for Multi-Image Spatial Intelligence
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出MMSI-Bench，一个多图像空间智能基准，解决现有评估仅测试单图像关系的问题。通过构建1000道多选题（含干扰项与推理步骤），评估34个模型，揭示其与人类（97%）的显著差距，并分析四大错误模式，推动多图像空间推理研究。**

- **链接: [http://arxiv.org/pdf/2505.23764v1](http://arxiv.org/pdf/2505.23764v1)**

> **作者:** Sihan Yang; Runsen Xu; Yiman Xie; Sizhe Yang; Mo Li; Jingli Lin; Chenming Zhu; Xiaochen Chen; Haodong Duan; Xiangyu Yue; Dahua Lin; Tai Wang; Jiangmiao Pang
>
> **备注:** 34 pages. A comprehensive, fully human-curated, multi-image-based spatial intelligence benchmark with reasoning annotation for MLLMs. Project page: https://runsenxu.com/projects/MMSI_Bench
>
> **摘要:** Spatial intelligence is essential for multimodal large language models (MLLMs) operating in the complex physical world. Existing benchmarks, however, probe only single-image relations and thus fail to assess the multi-image spatial reasoning that real-world deployments demand. We introduce MMSI-Bench, a VQA benchmark dedicated to multi-image spatial intelligence. Six 3D-vision researchers spent more than 300 hours meticulously crafting 1,000 challenging, unambiguous multiple-choice questions from over 120,000 images, each paired with carefully designed distractors and a step-by-step reasoning process. We conduct extensive experiments and thoroughly evaluate 34 open-source and proprietary MLLMs, observing a wide gap: the strongest open-source model attains roughly 30% accuracy and OpenAI's o3 reasoning model reaches 40%, while humans score 97%. These results underscore the challenging nature of MMSI-Bench and the substantial headroom for future research. Leveraging the annotated reasoning processes, we also provide an automated error analysis pipeline that diagnoses four dominant failure modes, including (1) grounding errors, (2) overlap-matching and scene-reconstruction errors, (3) situation-transformation reasoning errors, and (4) spatial-logic errors, offering valuable insights for advancing multi-image spatial intelligence. Project page: https://runsenxu.com/projects/MMSI_Bench .
>
---
#### [new 141] Differential Information: An Information-Theoretic Perspective on Preference Optimization
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文从信息论角度分析直接偏好优化（DPO），提出差异信息分布（DID）理论，证明其log-ratio奖励的最优性，揭示偏好需满足的策略条件，并分析熵对策略的影响。通过实验验证理论，指出高熵差异信息适合通用指令跟随，低熵适合知识问答，完善DPO的理论基础。**

- **链接: [http://arxiv.org/pdf/2505.23761v1](http://arxiv.org/pdf/2505.23761v1)**

> **作者:** Yunjae Won; Hyunji Lee; Hyeonbin Hwang; Minjoon Seo
>
> **备注:** 41 pages, 13 figures; due to the 1,920-character limitation imposed on the abstract field by arXiv, the abstract included on the arXiv page is slightly abbreviated compared to the version presented in the PDF
>
> **摘要:** Direct Preference Optimization (DPO) has become a standard technique for aligning language models with human preferences in a supervised manner. Despite its empirical success, the theoretical justification behind its log-ratio reward parameterization remains incomplete. In this work, we address this gap by utilizing the Differential Information Distribution (DID): a distribution over token sequences that captures the information gained during policy updates. First, we show that when preference labels encode the differential information required to transform a reference policy into a target policy, the log-ratio reward in DPO emerges as the uniquely optimal form for learning the target policy via preference optimization. This result naturally yields a closed-form expression for the optimal sampling distribution over rejected responses. Second, we find that the condition for preferences to encode differential information is fundamentally linked to an implicit assumption regarding log-margin ordered policies-an inductive bias widely used in preference optimization yet previously unrecognized. Finally, by analyzing the entropy of the DID, we characterize how learning low-entropy differential information reinforces the policy distribution, while high-entropy differential information induces a smoothing effect, which explains the log-likelihood displacement phenomenon. We validate our theoretical findings in synthetic experiments and extend them to real-world instruction-following datasets. Our results suggest that learning high-entropy differential information is crucial for general instruction-following, while learning low-entropy differential information benefits knowledge-intensive question answering. Overall, our work presents a unifying perspective on the DPO objective, the structure of preference data, and resulting policy behaviors through the lens of differential information.
>
---
#### [new 142] FlashFormer: Whole-Model Kernels for Efficient Low-Batch Inference
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于低批次推理优化任务，针对边缘部署和延迟敏感场景，解决内存带宽及核启动开销导致的效率问题，提出FlashFormer内核，优化单批次Transformer模型推理，实验证明在不同模型规模和量化设置下显著加速。**

- **链接: [http://arxiv.org/pdf/2505.22758v1](http://arxiv.org/pdf/2505.22758v1)**

> **作者:** Aniruddha Nrusimha; William Brandon; Mayank Mishra; Yikang Shen; Rameswar Panda; Jonathan Ragan-Kelley; Yoon Kim
>
> **摘要:** The size and compute characteristics of modern large language models have led to an increased interest in developing specialized kernels tailored for training and inference. Existing kernels primarily optimize for compute utilization, targeting the large-batch training and inference settings. However, low-batch inference, where memory bandwidth and kernel launch overheads contribute are significant factors, remains important for many applications of interest such as in edge deployment and latency-sensitive applications. This paper describes FlashFormer, a proof-of-concept kernel for accelerating single-batch inference for transformer-based large language models. Across various model sizes and quantizations settings, we observe nontrivial speedups compared to existing state-of-the-art inference kernels.
>
---
#### [new 143] DenoiseRotator: Enhance Pruning Robustness for LLMs via Importance Concentration
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大型语言模型（LLM）剪枝优化任务，旨在解决现有剪枝方法在半结构稀疏约束下性能显著下降的问题。提出DenoiseRotator方法，通过可学习的正交变换重新分配权重重要性，集中关键参数的重要性，增强剪枝鲁棒性，可与主流剪枝技术（如SparseGPT）结合，实验显示其显著缩小了压缩模型与原模型的性能差距。**

- **链接: [http://arxiv.org/pdf/2505.23049v1](http://arxiv.org/pdf/2505.23049v1)**

> **作者:** Tianteng Gu; Bei Liu; Bo Xiao; Ke Zeng; Jiacheng Liu; Yanmin Qian
>
> **摘要:** Pruning is a widely used technique to compress large language models (LLMs) by removing unimportant weights, but it often suffers from significant performance degradation - especially under semi-structured sparsity constraints. Existing pruning methods primarily focus on estimating the importance of individual weights, which limits their ability to preserve critical capabilities of the model. In this work, we propose a new perspective: rather than merely selecting which weights to prune, we first redistribute parameter importance to make the model inherently more amenable to pruning. By minimizing the information entropy of normalized importance scores, our approach concentrates importance onto a smaller subset of weights, thereby enhancing pruning robustness. We instantiate this idea through DenoiseRotator, which applies learnable orthogonal transformations to the model's weight matrices. Our method is model-agnostic and can be seamlessly integrated with existing pruning techniques such as Magnitude, SparseGPT, and Wanda. Evaluated on LLaMA3, Qwen2.5, and Mistral models under 50% unstructured and 2:4 semi-structured sparsity, DenoiseRotator consistently improves perplexity and zero-shot accuracy. For instance, on LLaMA3-70B pruned with SparseGPT at 2:4 semi-structured sparsity, DenoiseRotator reduces the perplexity gap to the dense model by 58%, narrowing the degradation from 8.1 to 3.4 points. Codes are available at https://github.com/Axel-gu/DenoiseRotator.
>
---
#### [new 144] MAP: Revisiting Weight Decomposition for Low-Rank Adaptation
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于参数高效微调（PEFT）任务，解决现有方法中权重分解方向定义缺乏几何理论基础的问题。提出MAP框架，将权重矩阵转为高维向量，通过归一化预训练权重、学习方向更新及标量系数独立缩放基向量与更新向量的幅度，实现严谨灵活的适应，提升性能并兼容现有PEFT方法。**

- **链接: [http://arxiv.org/pdf/2505.23094v1](http://arxiv.org/pdf/2505.23094v1)**

> **作者:** Chongjie Si; Zhiyi Shi; Yadao Wang; Xiaokang Yang; Susanto Rahardja; Wei Shen
>
> **摘要:** The rapid development of large language models has revolutionized natural language processing, but their fine-tuning remains computationally expensive, hindering broad deployment. Parameter-efficient fine-tuning (PEFT) methods, such as LoRA, have emerged as solutions. Recent work like DoRA attempts to further decompose weight adaptation into direction and magnitude components. However, existing formulations often define direction heuristically at the column level, lacking a principled geometric foundation. In this paper, we propose MAP, a novel framework that reformulates weight matrices as high-dimensional vectors and decouples their adaptation into direction and magnitude in a rigorous manner. MAP normalizes the pre-trained weights, learns a directional update, and introduces two scalar coefficients to independently scale the magnitude of the base and update vectors. This design enables more interpretable and flexible adaptation, and can be seamlessly integrated into existing PEFT methods. Extensive experiments show that MAP significantly improves performance when coupling with existing methods, offering a simple yet powerful enhancement to existing PEFT methods. Given the universality and simplicity of MAP, we hope it can serve as a default setting for designing future PEFT methods.
>
---
## 更新

#### [replaced 001] Shortcut-connected Expert Parallelism for Accelerating Mixture-of-Experts
- **分类: cs.LG; cs.CL; cs.DC**

- **链接: [http://arxiv.org/pdf/2404.05019v3](http://arxiv.org/pdf/2404.05019v3)**

> **作者:** Weilin Cai; Juyong Jiang; Le Qin; Junwei Cui; Sunghun Kim; Jiayi Huang
>
> **摘要:** Expert parallelism has emerged as a key strategy for distributing the computational workload of sparsely-gated mixture-of-experts (MoE) models across multiple devices, enabling the processing of increasingly large-scale models. However, the All-to-All communication inherent to expert parallelism poses a significant bottleneck, limiting the efficiency of MoE models. Although existing optimization methods partially mitigate this issue, they remain constrained by the sequential dependency between communication and computation operations. To address this challenge, we propose ScMoE, a novel shortcut-connected MoE architecture integrated with an overlapping parallelization strategy. ScMoE decouples communication from its conventional sequential ordering, enabling up to 100% overlap with computation. Compared to the prevalent top-2 MoE baseline, ScMoE achieves speedups of 1.49 times in training and 1.82 times in inference. Moreover, our experiments and analyses indicate that ScMoE not only achieves comparable but in some instances surpasses the model quality of existing approaches.
>
---
#### [replaced 002] Fast Large Language Model Collaborative Decoding via Speculation
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.01662v2](http://arxiv.org/pdf/2502.01662v2)**

> **作者:** Jiale Fu; Yuchu Jiang; Junkai Chen; Jiaming Fan; Xin Geng; Xu Yang
>
> **摘要:** Large Language Model (LLM) collaborative decoding techniques improve output quality by combining the outputs of multiple models at each generation step, but they incur high computational costs. In this paper, we introduce Collaborative decoding via Speculation (CoS), a novel framework that accelerates collaborative decoding without compromising performance. Inspired by Speculative Decoding--where a small proposal model generates tokens sequentially, and a larger target model verifies them in parallel, our approach builds on two key insights: (1) the verification distribution can be the combined distribution of both the proposal and target models, and (2) alternating each model as the proposer and verifier can further enhance efficiency. We generalize this method to collaboration among n models and theoretically prove that CoS is never slower than standard collaborative decoding, typically achieving faster speed. Extensive experiments demonstrate CoS is 1.11x-2.23x faster than standard collaborative decoding without compromising generation quality. Our code is available at https://github.com/Kamichanw/CoS/.
>
---
#### [replaced 003] Pangu Embedded: An Efficient Dual-system LLM Reasoner with Metacognition
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22375v2](http://arxiv.org/pdf/2505.22375v2)**

> **作者:** Hanting Chen; Yasheng Wang; Kai Han; Dong Li; Lin Li; Zhenni Bi; Jinpeng Li; Haoyu Wang; Fei Mi; Mingjian Zhu; Bin Wang; Kaikai Song; Yifei Fu; Xu He; Yu Luo; Chong Zhu; Quan He; Xueyu Wu; Wei He; Hailin Hu; Yehui Tang; Dacheng Tao; Xinghao Chen; Yunhe Wang
>
> **摘要:** This work presents Pangu Embedded, an efficient Large Language Model (LLM) reasoner developed on Ascend Neural Processing Units (NPUs), featuring flexible fast and slow thinking capabilities. Pangu Embedded addresses the significant computational costs and inference latency challenges prevalent in existing reasoning-optimized LLMs. We propose a two-stage training framework for its construction. In Stage 1, the model is finetuned via an iterative distillation process, incorporating inter-iteration model merging to effectively aggregate complementary knowledge. This is followed by reinforcement learning on Ascend clusters, optimized by a latency-tolerant scheduler that combines stale synchronous parallelism with prioritized data queues. The RL process is guided by a Multi-source Adaptive Reward System (MARS), which generates dynamic, task-specific reward signals using deterministic metrics and lightweight LLM evaluators for mathematics, coding, and general problem-solving tasks. Stage 2 introduces a dual-system framework, endowing Pangu Embedded with a "fast" mode for routine queries and a deeper "slow" mode for complex inference. This framework offers both manual mode switching for user control and an automatic, complexity-aware mode selection mechanism that dynamically allocates computational resources to balance latency and reasoning depth. Experimental results on benchmarks including AIME 2024, GPQA, and LiveCodeBench demonstrate that Pangu Embedded with 7B parameters, outperforms similar-size models like Qwen3-8B and GLM4-9B. It delivers rapid responses and state-of-the-art reasoning quality within a single, unified model architecture, highlighting a promising direction for developing powerful yet practically deployable LLM reasoners.
>
---
#### [replaced 004] Fusing Bidirectional Chains of Thought and Reward Mechanisms A Method for Enhancing Question-Answering Capabilities of Large Language Models for Chinese Intangible Cultural Heritage
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.08167v3](http://arxiv.org/pdf/2505.08167v3)**

> **作者:** Ruilin Liu; Zhixiao Zhao; Jieqiong Li; Chang Liu; Dongbo Wang
>
> **备注:** 22 pages, 5 figures
>
> **摘要:** The rapid development of large language models (LLMs) has provided significant support and opportunities for the advancement of domain-specific LLMs. However, fine-tuning these large models using Intangible Cultural Heritage (ICH) data inevitably faces challenges such as bias, incorrect knowledge inheritance, and catastrophic forgetting. To address these issues, we propose a novel training method that integrates a bidirectional chains of thought and a reward mechanism. This method is built upon ICH-Qwen, a large language model specifically designed for the field of intangible cultural heritage. The proposed method enables the model to not only perform forward reasoning but also enhances the accuracy of the generated answers by utilizing reverse questioning and reverse reasoning to activate the model's latent knowledge. Additionally, a reward mechanism is introduced during training to optimize the decision-making process. This mechanism improves the quality of the model's outputs through structural and content evaluations with different weighting schemes. We conduct comparative experiments on ICH-Qwen, with results demonstrating that our method outperforms 0-shot, step-by-step reasoning, knowledge distillation, and question augmentation methods in terms of accuracy, Bleu-4, and Rouge-L scores on the question-answering task. Furthermore, the paper highlights the effectiveness of combining the bidirectional chains of thought and reward mechanism through ablation experiments. In addition, a series of generalizability experiments are conducted, with results showing that the proposed method yields improvements on various domain-specific datasets and advanced models in areas such as Finance, Wikidata, and StrategyQA. This demonstrates that the method is adaptable to multiple domains and provides a valuable approach for model training in future applications across diverse fields.
>
---
#### [replaced 005] CLEME2.0: Towards Interpretable Evaluation by Disentangling Edits for Grammatical Error Correction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.00934v2](http://arxiv.org/pdf/2407.00934v2)**

> **作者:** Jingheng Ye; Zishan Xu; Yinghui Li; Linlin Song; Qingyu Zhou; Hai-Tao Zheng; Ying Shen; Wenhao Jiang; Hong-Gee Kim; Ruitong Liu; Xin Su; Zifei Shan
>
> **备注:** 19 pages, 12 tables, 3 figures. Accepted to ACL 2025 Main
>
> **摘要:** The paper focuses on the interpretability of Grammatical Error Correction (GEC) evaluation metrics, which received little attention in previous studies. To bridge the gap, we introduce **CLEME2.0**, a reference-based metric describing four fundamental aspects of GEC systems: hit-correction, wrong-correction, under-correction, and over-correction. They collectively contribute to exposing critical qualities and locating drawbacks of GEC systems. Evaluating systems by combining these aspects also leads to superior human consistency over other reference-based and reference-less metrics. Extensive experiments on two human judgment datasets and six reference datasets demonstrate the effectiveness and robustness of our method, achieving a new state-of-the-art result. Our codes are released at https://github.com/THUKElab/CLEME.
>
---
#### [replaced 006] LLM Meets Scene Graph: Can Large Language Models Understand and Generate Scene Graphs? A Benchmark and Empirical Study
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19510v2](http://arxiv.org/pdf/2505.19510v2)**

> **作者:** Dongil Yang; Minjin Kim; Sunghwan Kim; Beong-woo Kwak; Minjun Park; Jinseok Hong; Woontack Woo; Jinyoung Yeo
>
> **备注:** ACL 2025
>
> **摘要:** The remarkable reasoning and generalization capabilities of Large Language Models (LLMs) have paved the way for their expanding applications in embodied AI, robotics, and other real-world tasks. To effectively support these applications, grounding in spatial and temporal understanding in multimodal environments is essential. To this end, recent works have leveraged scene graphs, a structured representation that encodes entities, attributes, and their relationships in a scene. However, a comprehensive evaluation of LLMs' ability to utilize scene graphs remains limited. In this work, we introduce Text-Scene Graph (TSG) Bench, a benchmark designed to systematically assess LLMs' ability to (1) understand scene graphs and (2) generate them from textual narratives. With TSG Bench we evaluate 11 LLMs and reveal that, while models perform well on scene graph understanding, they struggle with scene graph generation, particularly for complex narratives. Our analysis indicates that these models fail to effectively decompose discrete scenes from a complex narrative, leading to a bottleneck when generating scene graphs. These findings underscore the need for improved methodologies in scene graph generation and provide valuable insights for future research. The demonstration of our benchmark is available at https://tsg-bench.netlify.app. Additionally, our code and evaluation data are publicly available at https://github.com/docworlds/tsg-bench.
>
---
#### [replaced 007] GIVE: Structured Reasoning of Large Language Models with Knowledge Graph Inspired Veracity Extrapolation
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.08475v3](http://arxiv.org/pdf/2410.08475v3)**

> **作者:** Jiashu He; Mingyu Derek Ma; Jinxuan Fan; Dan Roth; Wei Wang; Alejandro Ribeiro
>
> **摘要:** Existing approaches based on context prompting or reinforcement learning (RL) to improve the reasoning capacities of large language models (LLMs) depend on the LLMs' internal knowledge to produce reliable Chain-Of-Thought (CoT). However, no matter the size of LLMs, certain problems cannot be resolved in a single forward pass. Meanwhile, agent-based reasoning systems require access to a comprehensive nonparametric knowledge base, which is often costly or not feasible for use in scientific and niche domains. We present Graph Inspired Veracity Extrapolation (GIVE), a novel reasoning method that merges parametric and non-parametric memories to improve accurate reasoning with minimal external input. GIVE guides the LLM agent to select the most pertinent expert data (observe), engage in query-specific divergent thinking (reflect), and then synthesize this information to produce the final output (speak). Extensive experiments demonstrated the following benefits of our framework: (1) GIVE boosts the performance of LLMs across various sizes. (2) In some scenarios, GIVE allows smaller LLMs to surpass larger, more sophisticated ones in scientific tasks (GPT3.5T + GIVE > GPT4). (3) GIVE is effective on scientific and open-domain assessments. (4) GIVE is a training-free method that enables LLMs to tackle new problems that extend beyond their training data (up to 43.5% -> 88.2%} accuracy improvement). (5) GIVE allows LLM agents to reason using both restricted (very small) and noisy (very large) knowledge sources, accommodating knowledge graphs (KG) ranging from 135 to more than 840k nodes. (6) The reasoning process involved in GIVE is fully interpretable.
>
---
#### [replaced 008] ParamMute: Suppressing Knowledge-Critical FFNs for Faithful Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.15543v2](http://arxiv.org/pdf/2502.15543v2)**

> **作者:** Pengcheng Huang; Zhenghao Liu; Yukun Yan; Haiyan Zhao; Xiaoyuan Yi; Hao Chen; Zhiyuan Liu; Maosong Sun; Tong Xiao; Ge Yu; Chenyan Xiong
>
> **备注:** 22 pages, 7 figures, 7 tables
>
> **摘要:** Large language models (LLMs) integrated with retrieval-augmented generation (RAG) have improved factuality by grounding outputs in external evidence. However, they remain susceptible to unfaithful generation, where outputs contradict retrieved context despite its relevance and accuracy. Existing approaches aiming to improve faithfulness primarily focus on enhancing the utilization of external context, but often overlook the persistent influence of internal parametric knowledge during generation. In this work, we investigate the internal mechanisms behind unfaithful generation and identify a subset of mid-to-deep feed-forward networks (FFNs) that are disproportionately activated in such cases. Building on this insight, we propose Parametric Knowledge Muting through FFN Suppression (ParamMute), a framework that improves contextual faithfulness by suppressing the activation of unfaithfulness-associated FFNs and calibrating the model toward retrieved knowledge. To evaluate our approach, we introduce CoFaithfulQA, a benchmark specifically designed to evaluate faithfulness in scenarios where internal knowledge conflicts with accurate external evidence. Experimental results show that ParamMute significantly enhances faithfulness across both CoFaithfulQA and the established ConFiQA benchmark, achieving substantial reductions in reliance on parametric memory. These findings underscore the importance of mitigating internal knowledge dominance and provide a new direction for improving LLM trustworthiness in RAG. All code will be released via GitHub.
>
---
#### [replaced 009] LoRA-MGPO: Mitigating Double Descent in Low-Rank Adaptation via Momentum-Guided Perturbation Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14538v2](http://arxiv.org/pdf/2502.14538v2)**

> **作者:** Yupeng Chang; Chenlu Guo; Yi Chang; Yuan Wu
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) methods, such as Low-Rank Adaptation (LoRA), enable efficient adaptation of large language models (LLMs) via low-rank matrix optimization with frozen weights. However, LoRA typically exhibits "double descent" in training loss as rank increases, characterized by a three-phase dynamics: initial convergence, transient divergence, and eventual stabilization. This non-monotonic behavior delays convergence and impairs generalization through unstable gradients and attraction to sharp minima. To address these challenges, we propose LoRA-MGPO, a novel LoRA-based framework incorporating Momentum-Guided Perturbation Optimization (MGPO). First, MGPO eliminates Sharpness-Aware Minimization (SAM)'s dual gradient computations by reusing momentum vectors from optimizer states to guide perturbation directions. This retains SAM's training stability and flat minima preference with maintained efficiency. Second, MGPO incorporates adaptive perturbation normalization, scaling perturbation intensity via exponential moving average (EMA)-smoothed gradient magnitudes. Experiments on natural language understanding and generation benchmarks demonstrate that LoRA-MGPO outperforms LoRA and state-of-the-art PEFT methods. Further analysis confirms its ability to stabilize training and reduce sharp minima attraction, with smoother loss curves and improved convergence behavior. The code is available at https://github.com/llm172/LoRA-MGPO
>
---
#### [replaced 010] SciHorizon: Benchmarking AI-for-Science Readiness from Scientific Data to Large Language Models
- **分类: cs.LG; cs.CL; cs.DL; cs.IR**

- **链接: [http://arxiv.org/pdf/2503.13503v3](http://arxiv.org/pdf/2503.13503v3)**

> **作者:** Chuan Qin; Xin Chen; Chengrui Wang; Pengmin Wu; Xi Chen; Yihang Cheng; Jingyi Zhao; Meng Xiao; Xiangchao Dong; Qingqing Long; Boya Pan; Han Wu; Chengzan Li; Yuanchun Zhou; Hui Xiong; Hengshu Zhu
>
> **摘要:** In recent years, the rapid advancement of Artificial Intelligence (AI) technologies, particularly Large Language Models (LLMs), has revolutionized the paradigm of scientific discovery, establishing AI-for-Science (AI4Science) as a dynamic and evolving field. However, there is still a lack of an effective framework for the overall assessment of AI4Science, particularly from a holistic perspective on data quality and model capability. Therefore, in this study, we propose SciHorizon, a comprehensive assessment framework designed to benchmark the readiness of AI4Science from both scientific data and LLM perspectives. First, we introduce a generalizable framework for assessing AI-ready scientific data, encompassing four key dimensions: Quality, FAIRness, Explainability, and Compliance-which are subdivided into 15 sub-dimensions. Drawing on data resource papers published between 2018 and 2023 in peer-reviewed journals, we present recommendation lists of AI-ready datasets for Earth, Life, and Materials Sciences, making a novel and original contribution to the field. Concurrently, to assess the capabilities of LLMs across multiple scientific disciplines, we establish 16 assessment dimensions based on five core indicators Knowledge, Understanding, Reasoning, Multimodality, and Values spanning Mathematics, Physics, Chemistry, Life Sciences, and Earth and Space Sciences. Using the developed benchmark datasets, we have conducted a comprehensive evaluation of over 50 representative open-source and closed source LLMs. All the results are publicly available and can be accessed online at www.scihorizon.cn/en.
>
---
#### [replaced 011] Can We Predict Performance of Large Models across Vision-Language Tasks?
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.10112v2](http://arxiv.org/pdf/2410.10112v2)**

> **作者:** Qinyu Zhao; Ming Xu; Kartik Gupta; Akshay Asthana; Liang Zheng; Stephen Gould
>
> **备注:** ICML2025. Project page: https://github.com/Qinyu-Allen-Zhao/CrossPred-LVLM
>
> **摘要:** Evaluating large vision-language models (LVLMs) is very expensive, due to high computational cost and the wide variety of tasks. The good news is that if we already have some observed performance scores, we may be able to infer unknown ones. In this study, we propose a new framework for predicting unknown performance scores based on observed ones from other LVLMs or tasks. We first formulate the performance prediction as a matrix completion task. Specifically, we construct a sparse performance matrix $\boldsymbol{R}$, where each entry $R_{mn}$ represents the performance score of the $m$-th model on the $n$-th dataset. By applying probabilistic matrix factorization (PMF) with Markov chain Monte Carlo (MCMC), we can complete the performance matrix, i.e., predict unknown scores. Additionally, we estimate the uncertainty of performance prediction based on MCMC. Practitioners can evaluate their models on untested tasks with higher uncertainty first, which quickly reduces the prediction errors. We further introduce several improvements to enhance PMF for scenarios with sparse observed performance scores. Our experiments demonstrate the accuracy of PMF in predicting unknown scores, the reliability of uncertainty estimates in ordering evaluations, and the effectiveness of our enhancements for handling sparse data.
>
---
#### [replaced 012] SimGRAG: Leveraging Similar Subgraphs for Knowledge Graphs Driven Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2412.15272v2](http://arxiv.org/pdf/2412.15272v2)**

> **作者:** Yuzheng Cai; Zhenyue Guo; Yiwen Pei; Wanrui Bian; Weiguo Zheng
>
> **备注:** accepted by ACL 2025 (Findings)
>
> **摘要:** Recent advancements in large language models (LLMs) have shown impressive versatility across various tasks. To eliminate their hallucinations, retrieval-augmented generation (RAG) has emerged as a powerful approach, leveraging external knowledge sources like knowledge graphs (KGs). In this paper, we study the task of KG-driven RAG and propose a novel Similar Graph Enhanced Retrieval-Augmented Generation (SimGRAG) method. It effectively addresses the challenge of aligning query texts and KG structures through a two-stage process: (1) query-to-pattern, which uses an LLM to transform queries into a desired graph pattern, and (2) pattern-to-subgraph, which quantifies the alignment between the pattern and candidate subgraphs using a graph semantic distance (GSD) metric. We also develop an optimized retrieval algorithm that efficiently identifies the top-k subgraphs within 1-second on a 10-million-scale KG. Extensive experiments show that SimGRAG outperforms state-of-the-art KG-driven RAG methods in both question answering and fact verification. Our code is available at https://github.com/YZ-Cai/SimGRAG.
>
---
#### [replaced 013] CASS: Nvidia to AMD Transpilation with Data, Models, and Benchmark
- **分类: cs.AR; cs.AI; cs.CL; cs.LG; cs.PL**

- **链接: [http://arxiv.org/pdf/2505.16968v3](http://arxiv.org/pdf/2505.16968v3)**

> **作者:** Ahmed Heakl; Sarim Hashmi; Gustavo Bertolo Stahl; Seung Hun Eddie Han; Salman Khan; Abdulrahman Mahmoud
>
> **备注:** 20 pages, 11 figures, 5 tables
>
> **摘要:** We introduce CASS, the first large-scale dataset and model suite for cross-architecture GPU code transpilation, targeting both source-level (CUDA <--> HIP) and assembly-level (Nvidia SASS <--> AMD RDNA3) translation. The dataset comprises 70k verified code pairs across host and device, addressing a critical gap in low-level GPU code portability. Leveraging this resource, we train the CASS family of domain-specific language models, achieving 95% source translation accuracy and 37.5% assembly translation accuracy, substantially outperforming commercial baselines such as GPT-4o, Claude, and Hipify. Our generated code matches native performance in over 85% of test cases, preserving runtime and memory behavior. To support rigorous evaluation, we introduce CASS-Bench, a curated benchmark spanning 16 GPU domains with ground-truth execution. All data, models, and evaluation tools are released as open source to foster progress in GPU compiler tooling, binary compatibility, and LLM-guided hardware translation.
>
---
#### [replaced 014] SPRI: Aligning Large Language Models with Context-Situated Principles
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.03397v2](http://arxiv.org/pdf/2502.03397v2)**

> **作者:** Hongli Zhan; Muneeza Azmat; Raya Horesh; Junyi Jessy Li; Mikhail Yurochkin
>
> **备注:** Forty-Second International Conference on Machine Learning (ICML 2025) Camera-Ready Version
>
> **摘要:** Aligning Large Language Models to integrate and reflect human values, especially for tasks that demand intricate human oversight, is arduous since it is resource-intensive and time-consuming to depend on human expertise for context-specific guidance. Prior work has utilized predefined sets of rules or principles to steer the behavior of models (Bai et al., 2022; Sun et al., 2023). However, these principles tend to be generic, making it challenging to adapt them to each individual input query or context. In this work, we present Situated-PRInciples (SPRI), a framework requiring minimal or no human effort that is designed to automatically generate guiding principles in real-time for each input query and utilize them to align each response. We evaluate SPRI on three tasks, and show that 1) SPRI can derive principles in a complex domain-specific task that leads to on-par performance as expert-crafted ones; 2) SPRI-generated principles lead to instance-specific rubrics that outperform prior LLM-as-a-judge frameworks; 3) using SPRI to generate synthetic SFT data leads to substantial improvement on truthfulness. We release our code and model generations at https://github.com/honglizhan/SPRI-public.
>
---
#### [replaced 015] Towards Logically Sound Natural Language Reasoning with Logic-Enhanced Language Model Agents
- **分类: cs.AI; cs.CL; cs.GT; cs.LO**

- **链接: [http://arxiv.org/pdf/2408.16081v2](http://arxiv.org/pdf/2408.16081v2)**

> **作者:** Agnieszka Mensfelt; Kostas Stathis; Vince Trencsenyi
>
> **备注:** Source code: https://github.com/dicelab-rhul/LELMA
>
> **摘要:** Large language models (LLMs) are increasingly explored as general-purpose reasoners, particularly in agentic contexts. However, their outputs remain prone to mathematical and logical errors. This is especially challenging in open-ended tasks, where unstructured outputs lack explicit ground truth and may contain subtle inconsistencies. To address this issue, we propose Logic-Enhanced Language Model Agents (LELMA), a framework that integrates LLMs with formal logic to enable validation and refinement of natural language reasoning. LELMA comprises three components: an LLM-Reasoner, an LLM-Translator, and a Solver, and employs autoformalization to translate reasoning into logic representations, which are then used to assess logical validity. Using game-theoretic scenarios such as the Prisoner's Dilemma as testbeds, we highlight the limitations of both less capable (Gemini 1.0 Pro) and advanced (GPT-4o) models in generating logically sound reasoning. LELMA achieves high accuracy in error detection and improves reasoning correctness via self-refinement, particularly in GPT-4o. The study also highlights challenges in autoformalization accuracy and in evaluation of inherently ambiguous open-ended reasoning tasks.
>
---
#### [replaced 016] Token Pruning in Multimodal Large Language Models: Are We Solving the Right Problem?
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.11501v2](http://arxiv.org/pdf/2502.11501v2)**

> **作者:** Zichen Wen; Yifeng Gao; Weijia Li; Conghui He; Linfeng Zhang
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Multimodal large language models (MLLMs) have shown remarkable performance for cross-modal understanding and generation, yet still suffer from severe inference costs. Recently, abundant works have been proposed to solve this problem with token pruning, which identifies the redundant tokens in MLLMs and then prunes them to reduce the computation and KV storage costs, leading to significant acceleration without training. While these methods claim efficiency gains, critical questions about their fundamental design and evaluation remain unanswered: Why do many existing approaches underperform even compared to naive random token selection? Are attention-based scoring sufficient for reliably identifying redundant tokens? Is language information really helpful during token pruning? What makes a good trade-off between token importance and duplication? Are current evaluation protocols comprehensive and unbiased? The ignorance of previous research on these problems hinders the long-term development of token pruning. In this paper, we answer these questions one by one, providing insights into the design of future token pruning methods.
>
---
#### [replaced 017] mOSCAR: A Large-scale Multilingual and Multimodal Document-level Corpus
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2406.08707v2](http://arxiv.org/pdf/2406.08707v2)**

> **作者:** Matthieu Futeral; Armel Zebaze; Pedro Ortiz Suarez; Julien Abadji; Rémi Lacroix; Cordelia Schmid; Rachel Bawden; Benoît Sagot
>
> **备注:** ACL 2025 (Findings)
>
> **摘要:** Multimodal Large Language Models (mLLMs) are trained on a large amount of text-image data. While most mLLMs are trained on caption-like data only, Alayrac et al. (2022) showed that additionally training them on interleaved sequences of text and images can lead to the emergence of in-context learning capabilities. However, the dataset they used, M3W, is not public and is only in English. There have been attempts to reproduce their results but the released datasets are English-only. In contrast, current multilingual and multimodal datasets are either composed of caption-like only or medium-scale or fully private data. This limits mLLM research for the 7,000 other languages spoken in the world. We therefore introduce mOSCAR, to the best of our knowledge the first large-scale multilingual and multimodal document corpus crawled from the web. It covers 163 languages, 303M documents, 200B tokens and 1.15B images. We carefully conduct a set of filtering and evaluation steps to make sure mOSCAR is sufficiently safe, diverse and of good quality. We additionally train two types of multilingual model to prove the benefits of mOSCAR: (1) a model trained on a subset of mOSCAR and captioning data and (2) a model trained on captioning data only. The model additionally trained on mOSCAR shows a strong boost in few-shot learning performance across various multilingual image-text tasks and benchmarks, confirming previous findings for English-only mLLMs. The dataset is released under the Creative Commons CC BY 4.0 license and can be accessed here: https://huggingface.co/datasets/oscar-corpus/mOSCAR
>
---
#### [replaced 018] Multilingual Encoder Knows more than You Realize: Shared Weights Pretraining for Extremely Low-Resource Languages
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.10852v2](http://arxiv.org/pdf/2502.10852v2)**

> **作者:** Zeli Su; Ziyin Zhang; Guixian Xu; Jianing Liu; XU Han; Ting Zhang; Yushuang Dong
>
> **备注:** ACL 2025 camera-ready
>
> **摘要:** While multilingual language models like XLM-R have advanced multilingualism in NLP, they still perform poorly in extremely low-resource languages. This situation is exacerbated by the fact that modern LLMs such as LLaMA and Qwen support far fewer languages than XLM-R, making text generation models non-existent for many languages in the world. To tackle this challenge, we propose a novel framework for adapting multilingual encoders to text generation in extremely low-resource languages. By reusing the weights between the encoder and the decoder, our framework allows the model to leverage the learned semantic space of the encoder, enabling efficient learning and effective generalization in low-resource languages. Applying this framework to four Chinese minority languages, we present XLM-SWCM, and demonstrate its superior performance on various downstream tasks even when compared with much larger models.
>
---
#### [replaced 019] Decomposed Opinion Summarization with Verified Aspect-Aware Modules
- **分类: cs.CL; cs.IR**

- **链接: [http://arxiv.org/pdf/2501.17191v3](http://arxiv.org/pdf/2501.17191v3)**

> **作者:** Miao Li; Jey Han Lau; Eduard Hovy; Mirella Lapata
>
> **备注:** 37 pages, long paper, present at ACL 2025
>
> **摘要:** Opinion summarization plays a key role in deriving meaningful insights from large-scale online reviews. To make the process more explainable and grounded, we propose a domain-agnostic modular approach guided by review aspects (e.g., cleanliness for hotel reviews) which separates the tasks of aspect identification, opinion consolidation, and meta-review synthesis to enable greater transparency and ease of inspection. We conduct extensive experiments across datasets representing scientific research, business, and product domains. Results show that our approach generates more grounded summaries compared to strong baseline models, as verified through automated and human evaluations. Additionally, our modular approach, which incorporates reasoning based on review aspects, produces more informative intermediate outputs than other knowledge-agnostic decomposition approaches. Lastly, we provide empirical results to show that these intermediate outputs can support humans in summarizing opinions from large volumes of reviews.
>
---
#### [replaced 020] X-TURING: Towards an Enhanced and Efficient Turing Test for Long-Term Dialogue Agents
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2408.09853v2](http://arxiv.org/pdf/2408.09853v2)**

> **作者:** Weiqi Wu; Hongqiu Wu; Hai Zhao
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** The Turing test examines whether AIs exhibit human-like behaviour in natural language conversations. The traditional setting limits each participant to one message at a time and requires constant human participation. This fails to reflect a natural conversational style and hinders the evaluation of dialogue agents based on Large Language Models (LLMs) in complex and prolonged interactions. This paper proposes \textbf{\textsc{X-Turing}}, which enhances the original test with a \textit{burst dialogue} pattern, allowing more dynamic exchanges using consecutive messages. It further reduces human workload by iteratively generating dialogues that simulate the long-term interaction between the agent and a human to compose the majority of the test process. With the \textit{pseudo-dialogue} history, the agent then engages in a shorter dialogue with a real human, which is paired with a human-human conversation on the same topic to be judged using questionnaires. We introduce the \textit{X-Turn Pass-Rate} metric to assess the human likeness of LLMs across varying durations. While LLMs like GPT-4 initially perform well, achieving pass rates of 51.9\% and 38.9\% during 3 turns and 10 turns of dialogues respectively, their performance drops as the dialogue progresses, which underscores the difficulty in maintaining consistency in the long term.
>
---
#### [replaced 021] LEXam: Benchmarking Legal Reasoning on 340 Law Exams
- **分类: cs.CL; cs.AI; cs.LG; 68T50; I.2**

- **链接: [http://arxiv.org/pdf/2505.12864v2](http://arxiv.org/pdf/2505.12864v2)**

> **作者:** Yu Fan; Jingwei Ni; Jakob Merane; Etienne Salimbeni; Yang Tian; Yoan Hermstrüwer; Yinya Huang; Mubashara Akhtar; Florian Geering; Oliver Dreyer; Daniel Brunner; Markus Leippold; Mrinmaya Sachan; Alexander Stremitzer; Christoph Engel; Elliott Ash; Joel Niklaus
>
> **摘要:** Long-form legal reasoning remains a key challenge for large language models (LLMs) in spite of recent advances in test-time scaling. We introduce LEXam, a novel benchmark derived from 340 law exams spanning 116 law school courses across a range of subjects and degree levels. The dataset comprises 4,886 law exam questions in English and German, including 2,841 long-form, open-ended questions and 2,045 multiple-choice questions. Besides reference answers, the open questions are also accompanied by explicit guidance outlining the expected legal reasoning approach such as issue spotting, rule recall, or rule application. Our evaluation on both open-ended and multiple-choice questions present significant challenges for current LLMs; in particular, they notably struggle with open questions that require structured, multi-step legal reasoning. Moreover, our results underscore the effectiveness of the dataset in differentiating between models with varying capabilities. Adopting an LLM-as-a-Judge paradigm with rigorous human expert validation, we demonstrate how model-generated reasoning steps can be evaluated consistently and accurately. Our evaluation setup provides a scalable method to assess legal reasoning quality beyond simple accuracy metrics. Project page: https://lexam-benchmark.github.io/
>
---
#### [replaced 022] AntiLeakBench: Preventing Data Contamination by Automatically Constructing Benchmarks with Updated Real-World Knowledge
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.13670v2](http://arxiv.org/pdf/2412.13670v2)**

> **作者:** Xiaobao Wu; Liangming Pan; Yuxi Xie; Ruiwen Zhou; Shuai Zhao; Yubo Ma; Mingzhe Du; Rui Mao; Anh Tuan Luu; William Yang Wang
>
> **备注:** Accepted to ACL 2025 main conference. Code and data are at https://github.com/bobxwu/AntiLeakBench
>
> **摘要:** Data contamination hinders fair LLM evaluation by introducing test data into newer models' training sets. Existing studies solve this challenge by updating benchmarks with newly collected data. However, they fail to guarantee contamination-free evaluation as the newly collected data may contain pre-existing knowledge, and their benchmark updates rely on intensive human labor. To address these issues, we in this paper propose AntiLeak-Bench, an automated anti-leakage benchmarking framework. Instead of simply using newly collected data, we construct samples with explicitly new knowledge absent from LLMs' training sets, which thus ensures strictly contamination-free evaluation. We further design a fully automated workflow to build and update our benchmark without human labor. This significantly reduces the cost of benchmark maintenance to accommodate emerging LLMs. Through extensive experiments, we highlight that data contamination likely exists before LLMs' cutoff time and demonstrate AntiLeak-Bench effectively overcomes this challenge.
>
---
#### [replaced 023] Too Consistent to Detect: A Study of Self-Consistent Errors in LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17656v2](http://arxiv.org/pdf/2505.17656v2)**

> **作者:** Hexiang Tan; Fei Sun; Sha Liu; Du Su; Qi Cao; Xin Chen; Jingang Wang; Xunliang Cai; Yuanzhuo Wang; Huawei Shen; Xueqi Cheng
>
> **摘要:** As large language models (LLMs) often generate plausible but incorrect content, error detection has become increasingly critical to ensure truthfulness. However, existing detection methods often overlook a critical problem we term as self-consistent error, where LLMs repeatly generate the same incorrect response across multiple stochastic samples. This work formally defines self-consistent errors and evaluates mainstream detection methods on them. Our investigation reveals two key findings: (1) Unlike inconsistent errors, whose frequency diminishes significantly as LLM scale increases, the frequency of self-consistent errors remains stable or even increases. (2) All four types of detection methshods significantly struggle to detect self-consistent errors. These findings reveal critical limitations in current detection methods and underscore the need for improved methods. Motivated by the observation that self-consistent errors often differ across LLMs, we propose a simple but effective cross-model probe method that fuses hidden state evidence from an external verifier LLM. Our method significantly enhances performance on self-consistent errors across three LLM families.
>
---
#### [replaced 024] OrionBench: A Benchmark for Chart and Human-Recognizable Object Detection in Infographics
- **分类: cs.CV; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.17473v3](http://arxiv.org/pdf/2505.17473v3)**

> **作者:** Jiangning Zhu; Yuxing Zhou; Zheng Wang; Juntao Yao; Yima Gu; Yuhui Yuan; Shixia Liu
>
> **摘要:** Given the central role of charts in scientific, business, and communication contexts, enhancing the chart understanding capabilities of vision-language models (VLMs) has become increasingly critical. A key limitation of existing VLMs lies in their inaccurate visual grounding of infographic elements, including charts and human-recognizable objects (HROs) such as icons and images. However, chart understanding often requires identifying relevant elements and reasoning over them. To address this limitation, we introduce OrionBench, a benchmark designed to support the development of accurate object detection models for charts and HROs in infographics. It contains 26,250 real and 78,750 synthetic infographics, with over 6.9 million bounding box annotations. These annotations are created by combining the model-in-the-loop and programmatic methods. We demonstrate the usefulness of OrionBench through three applications: 1) constructing a Thinking-with-Boxes scheme to boost the chart understanding performance of VLMs, 2) comparing existing object detection models, and 3) applying the developed detection model to document layout and UI element detection.
>
---
#### [replaced 025] BRIGHTER: BRIdging the Gap in Human-Annotated Textual Emotion Recognition Datasets for 28 Languages
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11926v4](http://arxiv.org/pdf/2502.11926v4)**

> **作者:** Shamsuddeen Hassan Muhammad; Nedjma Ousidhoum; Idris Abdulmumin; Jan Philip Wahle; Terry Ruas; Meriem Beloucif; Christine de Kock; Nirmal Surange; Daniela Teodorescu; Ibrahim Said Ahmad; David Ifeoluwa Adelani; Alham Fikri Aji; Felermino D. M. A. Ali; Ilseyar Alimova; Vladimir Araujo; Nikolay Babakov; Naomi Baes; Ana-Maria Bucur; Andiswa Bukula; Guanqun Cao; Rodrigo Tufino Cardenas; Rendi Chevi; Chiamaka Ijeoma Chukwuneke; Alexandra Ciobotaru; Daryna Dementieva; Murja Sani Gadanya; Robert Geislinger; Bela Gipp; Oumaima Hourrane; Oana Ignat; Falalu Ibrahim Lawan; Rooweither Mabuya; Rahmad Mahendra; Vukosi Marivate; Alexander Panchenko; Andrew Piper; Charles Henrique Porto Ferreira; Vitaly Protasov; Samuel Rutunda; Manish Shrivastava; Aura Cristina Udrea; Lilian Diana Awuor Wanzare; Sophie Wu; Florian Valentin Wunderlich; Hanif Muhammad Zhafran; Tianhui Zhang; Yi Zhou; Saif M. Mohammad
>
> **备注:** Accepted at ACL2025 (Main)
>
> **摘要:** People worldwide use language in subtle and complex ways to express emotions. Although emotion recognition--an umbrella term for several NLP tasks--impacts various applications within NLP and beyond, most work in this area has focused on high-resource languages. This has led to significant disparities in research efforts and proposed solutions, particularly for under-resourced languages, which often lack high-quality annotated datasets. In this paper, we present BRIGHTER--a collection of multi-labeled, emotion-annotated datasets in 28 different languages and across several domains. BRIGHTER primarily covers low-resource languages from Africa, Asia, Eastern Europe, and Latin America, with instances labeled by fluent speakers. We highlight the challenges related to the data collection and annotation processes, and then report experimental results for monolingual and crosslingual multi-label emotion identification, as well as emotion intensity recognition. We analyse the variability in performance across languages and text domains, both with and without the use of LLMs, and show that the BRIGHTER datasets represent a meaningful step towards addressing the gap in text-based emotion recognition.
>
---
#### [replaced 026] BA-LoRA: Bias-Alleviating Low-Rank Adaptation to Mitigate Catastrophic Inheritance in Large Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2408.04556v5](http://arxiv.org/pdf/2408.04556v5)**

> **作者:** Yupeng Chang; Yi Chang; Yuan Wu
>
> **备注:** 25 pages
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable proficiency across various natural language processing (NLP) tasks. However, adapting LLMs to downstream applications requires computationally intensive and memory-demanding fine-tuning procedures. To alleviate these burdens, parameter-efficient fine-tuning (PEFT) techniques have emerged as a promising approach to tailor LLMs with minimal computational overhead. While PEFT methods offer substantial advantages, they do not fully address the pervasive issue of bias propagation from pre-training data. This work introduces Bias-Alleviating Low-Rank Adaptation (BA-LoRA), a novel PEFT method designed to counteract bias inheritance. BA-LoRA incorporates three distinct regularization terms: (1) a consistency regularizer, (2) a diversity regularizer, and (3) a singular value decomposition regularizer. These regularizers aim to enhance the models' consistency, diversity, and generalization capabilities during fine-tuning. We conduct extensive experiments on natural language understanding (NLU) and natural language generation (NLG) tasks using prominent LLMs such as LLaMA, Mistral, and Gemma. The results demonstrate that BA-LoRA outperforms LoRA and its state-of-the-art variants. Moreover, the extended experiments demonstrate that our method effectively mitigates the adverse effects of pre-training bias, leading to more reliable and robust model outputs. The code is available at https://github.com/cyp-jlu-ai/BA-LoRA.
>
---
#### [replaced 027] DiagnosisArena: Benchmarking Diagnostic Reasoning for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.14107v4](http://arxiv.org/pdf/2505.14107v4)**

> **作者:** Yakun Zhu; Zhongzhen Huang; Linjie Mu; Yutong Huang; Wei Nie; Jiaji Liu; Shaoting Zhang; Pengfei Liu; Xiaofan Zhang
>
> **摘要:** The emergence of groundbreaking large language models capable of performing complex reasoning tasks holds significant promise for addressing various scientific challenges, including those arising in complex clinical scenarios. To enable their safe and effective deployment in real-world healthcare settings, it is urgently necessary to benchmark the diagnostic capabilities of current models systematically. Given the limitations of existing medical benchmarks in evaluating advanced diagnostic reasoning, we present DiagnosisArena, a comprehensive and challenging benchmark designed to rigorously assess professional-level diagnostic competence. DiagnosisArena consists of 1,113 pairs of segmented patient cases and corresponding diagnoses, spanning 28 medical specialties, deriving from clinical case reports published in 10 top-tier medical journals. The benchmark is developed through a meticulous construction pipeline, involving multiple rounds of screening and review by both AI systems and human experts, with thorough checks conducted to prevent data leakage. Our study reveals that even the most advanced reasoning models, o3, o1, and DeepSeek-R1, achieve only 51.12%, 31.09%, and 17.79% accuracy, respectively. This finding highlights a significant generalization bottleneck in current large language models when faced with clinical diagnostic reasoning challenges. Through DiagnosisArena, we aim to drive further advancements in AI's diagnostic reasoning capabilities, enabling more effective solutions for real-world clinical diagnostic challenges. We provide the benchmark and evaluation tools for further research and development https://github.com/SPIRAL-MED/DiagnosisArena.
>
---
#### [replaced 028] Business as Rulesual: A Benchmark and Framework for Business Rule Flow Modeling with LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18542v2](http://arxiv.org/pdf/2505.18542v2)**

> **作者:** Chen Yang; Ruping Xu; Ruizhe Li; Bin Cao; Jing Fan
>
> **摘要:** Process mining aims to discover, monitor and optimize the actual behaviors of real processes. While prior work has mainly focused on extracting procedural action flows from instructional texts, rule flows embedded in business documents remain underexplored. To this end, we introduce a novel annotated Chinese dataset, BPRF, which contains 50 business process documents with 326 explicitly labeled business rules across multiple domains. Each rule is represented as a <Condition, Action> pair, and we annotate logical dependencies between rules (sequential, conditional, or parallel). We also propose ExIde, a framework for automatic business rule extraction and dependency relationship identification using large language models (LLMs). We evaluate ExIde using 12 state-of-the-art (SOTA) LLMs on the BPRF dataset, benchmarking performance on both rule extraction and dependency classification tasks of current LLMs. Our results demonstrate the effectiveness of ExIde in extracting structured business rules and analyzing their interdependencies for current SOTA LLMs, paving the way for more automated and interpretable business process automation.
>
---
#### [replaced 029] Jailbreaking to Jailbreak
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.09638v2](http://arxiv.org/pdf/2502.09638v2)**

> **作者:** Jeremy Kritz; Vaughn Robinson; Robert Vacareanu; Bijan Varjavand; Michael Choi; Bobby Gogov; Scale Red Team; Summer Yue; Willow E. Primack; Zifan Wang
>
> **摘要:** Large Language Models (LLMs) can be used to red team other models (e.g. jailbreaking) to elicit harmful contents. While prior works commonly employ open-weight models or private uncensored models for doing jailbreaking, as the refusal-training of strong LLMs (e.g. OpenAI o3) refuse to help jailbreaking, our work turn (almost) any black-box LLMs into attackers. The resulting $J_2$ (jailbreaking-to-jailbreak) attackers can effectively jailbreak the safeguard of target models using various strategies, both created by themselves or from expert human red teamers. In doing so, we show their strong but under-researched jailbreaking capabilities. Our experiments demonstrate that 1) prompts used to create $J_2$ attackers transfer across almost all black-box models; 2) an $J_2$ attacker can jailbreak a copy of itself, and this vulnerability develops rapidly over the past 12 months; 3) reasong models, such as Sonnet-3.7, are strong $J_2$ attackers compared to others. For example, when used against the safeguard of GPT-4o, $J_2$ (Sonnet-3.7) achieves 0.975 attack success rate (ASR), which matches expert human red teamers and surpasses the state-of-the-art algorithm-based attacks. Among $J_2$ attackers, $J_2$ (o3) achieves highest ASR (0.605) against Sonnet-3.5, one of the most robust models.
>
---
#### [replaced 030] A Survey of Uncertainty Estimation Methods on Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.00172v2](http://arxiv.org/pdf/2503.00172v2)**

> **作者:** Zhiqiu Xia; Jinxuan Xu; Yuqian Zhang; Hang Liu
>
> **备注:** ACL Findings 2025
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities across various tasks. However, these models could offer biased, hallucinated, or non-factual responses camouflaged by their fluency and realistic appearance. Uncertainty estimation is the key method to address this challenge. While research efforts in uncertainty estimation are ramping up, there is a lack of comprehensive and dedicated surveys on LLM uncertainty estimation. This survey presents four major avenues of LLM uncertainty estimation. Furthermore, we perform extensive experimental evaluations across multiple methods and datasets. At last, we provide critical and promising future directions for LLM uncertainty estimation.
>
---
#### [replaced 031] Small Language Models: Architectures, Techniques, Evaluation, Problems and Future Adaptation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19529v2](http://arxiv.org/pdf/2505.19529v2)**

> **作者:** Tanjil Hasan Sakib; Md. Tanzib Hosain; Md. Kishor Morol
>
> **备注:** 9 pages
>
> **摘要:** Small Language Models (SLMs) have gained substantial attention due to their ability to execute diverse language tasks successfully while using fewer computer resources. These models are particularly ideal for deployment in limited environments, such as mobile devices, on-device processing, and edge systems. In this study, we present a complete assessment of SLMs, focussing on their design frameworks, training approaches, and techniques for lowering model size and complexity. We offer a novel classification system to organize the optimization approaches applied for SLMs, encompassing strategies like pruning, quantization, and model compression. Furthermore, we assemble SLM's studies of evaluation suite with some existing datasets, establishing a rigorous platform for measuring SLM capabilities. Alongside this, we discuss the important difficulties that remain unresolved in this sector, including trade-offs between efficiency and performance, and we suggest directions for future study. We anticipate this study to serve as a beneficial guide for researchers and practitioners who aim to construct compact, efficient, and high-performing language models.
>
---
#### [replaced 032] How Transformers Learn Regular Language Recognition: A Theoretical Study on Training Dynamics and Implicit Bias
- **分类: cs.LG; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2505.00926v3](http://arxiv.org/pdf/2505.00926v3)**

> **作者:** Ruiquan Huang; Yingbin Liang; Jing Yang
>
> **备注:** accepted by ICML 2025
>
> **摘要:** Language recognition tasks are fundamental in natural language processing (NLP) and have been widely used to benchmark the performance of large language models (LLMs). These tasks also play a crucial role in explaining the working mechanisms of transformers. In this work, we focus on two representative tasks in the category of regular language recognition, known as `even pairs' and `parity check', the aim of which is to determine whether the occurrences of certain subsequences in a given sequence are even. Our goal is to explore how a one-layer transformer, consisting of an attention layer followed by a linear layer, learns to solve these tasks by theoretically analyzing its training dynamics under gradient descent. While even pairs can be solved directly by a one-layer transformer, parity check need to be solved by integrating Chain-of-Thought (CoT), either into the inference stage of a transformer well-trained for the even pairs task, or into the training of a one-layer transformer. For both problems, our analysis shows that the joint training of attention and linear layers exhibits two distinct phases. In the first phase, the attention layer grows rapidly, mapping data sequences into separable vectors. In the second phase, the attention layer becomes stable, while the linear layer grows logarithmically and approaches in direction to a max-margin hyperplane that correctly separates the attention layer outputs into positive and negative samples, and the loss decreases at a rate of $O(1/t)$. Our experiments validate those theoretical results.
>
---
#### [replaced 033] GWQ: Gradient-Aware Weight Quantization for Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2411.00850v4](http://arxiv.org/pdf/2411.00850v4)**

> **作者:** Yihua Shao; Yan Gu; Siyu Chen; Haiyang Liu; Zixian Zhu; Zijian Ling; Minxi Yan; Ziyang Yan; Chenyu Zhang; Michele Magno; Haotong Qin; Yan Wang; Jingcai Guo; Ling Shao; Hao Tang
>
> **摘要:** Large language models (LLMs) show impressive performance in solving complex language tasks. However, its large number of parameters presents significant challenges for the deployment. So, compressing LLMs to low bits can enable to deploy on resource-constrained devices. To address this problem, we propose gradient-aware weight quantization (GWQ), the first quantization approach for low-bit weight quantization that leverages gradients to localize outliers, requiring only a minimal amount of calibration data for outlier detection. GWQ retains the top 1\% outliers preferentially at FP16 precision, while the remaining non-outlier weights are stored in a low-bit. We widely evaluate GWQ on different task include language modeling, grounding detection, massive multitask language understanding and vision-language question and answering. Results show that models quantified by GWQ performs better than other quantization method. During quantization process, GWQ only need one calibration set to realize effective quant. Also, GWQ achieves 1.2x inference speedup in comparison to the original model and effectively reduces the inference memory.
>
---
#### [replaced 034] Chain of Grounded Objectives: Bridging Process and Goal-oriented Prompting for Code Generation
- **分类: cs.CL; cs.AI; cs.SE**

- **链接: [http://arxiv.org/pdf/2501.13978v2](http://arxiv.org/pdf/2501.13978v2)**

> **作者:** Sangyeop Yeo; Seung-won Hwang; Yu-Seung Ma
>
> **备注:** Accepted by ECOOP 2025 main conference
>
> **摘要:** The use of Large Language Models (LLMs) for code generation has gained significant attention in recent years. Existing methods often aim to improve the quality of generated code by incorporating additional contextual information or guidance into input prompts. Many of these approaches adopt sequential reasoning strategies, mimicking human-like step-by-step thinking. However, such strategies may constrain flexibility, as they do not always align with the structured characteristics of programming languages. This paper introduces the Chain of Grounded Objectives (CGO), a method that embeds functional objectives into input prompts to enhance code generation. By leveraging appropriately structured objectives as input and avoiding explicit sequential procedures, CGO adapts effectively to the structured nature of programming tasks. Empirical evaluations demonstrate that CGO effectively enhances code generation, addressing limitations of existing approaches.
>
---
#### [replaced 035] Divide and Conquer: A Hybrid Strategy Defeats Multimodal Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.16555v3](http://arxiv.org/pdf/2412.16555v3)**

> **作者:** Yanxu Mao; Peipei Liu; Tiehan Cui; Zhaoteng Yan; Congying Liu; Datao You
>
> **摘要:** Large language models (LLMs) are widely applied in various fields of society due to their powerful reasoning, understanding, and generation capabilities. However, the security issues associated with these models are becoming increasingly severe. Jailbreaking attacks, as an important method for detecting vulnerabilities in LLMs, have been explored by researchers who attempt to induce these models to generate harmful content through various attack methods. Nevertheless, existing jailbreaking methods face numerous limitations, such as excessive query counts, limited coverage of jailbreak modalities, low attack success rates, and simplistic evaluation methods. To overcome these constraints, this paper proposes a multimodal jailbreaking method: JMLLM. This method integrates multiple strategies to perform comprehensive jailbreak attacks across text, visual, and auditory modalities. Additionally, we contribute a new and comprehensive dataset for multimodal jailbreaking research: TriJail, which includes jailbreak prompts for all three modalities. Experiments on the TriJail dataset and the benchmark dataset AdvBench, conducted on 13 popular LLMs, demonstrate advanced attack success rates and significant reduction in time overhead.
>
---
#### [replaced 036] Distill CLIP (DCLIP): Enhancing Image-Text Retrieval via Cross-Modal Transformer Distillation
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21549v2](http://arxiv.org/pdf/2505.21549v2)**

> **作者:** Daniel Csizmadia; Andrei Codreanu; Victor Sim; Vighnesh Prabhu; Michael Lu; Kevin Zhu; Sean O'Brien; Vasu Sharma
>
> **摘要:** We present Distill CLIP (DCLIP), a fine-tuned variant of the CLIP model that enhances multimodal image-text retrieval while preserving the original model's strong zero-shot classification capabilities. CLIP models are typically constrained by fixed image resolutions and limited context, which can hinder their effectiveness in retrieval tasks that require fine-grained cross-modal understanding. DCLIP addresses these challenges through a meta teacher-student distillation framework, where a cross-modal transformer teacher is fine-tuned to produce enriched embeddings via bidirectional cross-attention between YOLO-extracted image regions and corresponding textual spans. These semantically and spatially aligned global representations guide the training of a lightweight student model using a hybrid loss that combines contrastive learning and cosine similarity objectives. Despite being trained on only ~67,500 samples curated from MSCOCO, Flickr30k, and Conceptual Captions-just a fraction of CLIP's original dataset-DCLIP significantly improves image-text retrieval metrics (Recall@K, MAP), while retaining approximately 94% of CLIP's zero-shot classification performance. These results demonstrate that DCLIP effectively mitigates the trade-off between task specialization and generalization, offering a resource-efficient, domain-adaptive, and detail-sensitive solution for advanced vision-language tasks. Code available at https://anonymous.4open.science/r/DCLIP-B772/README.md.
>
---
#### [replaced 037] LLMs Can Achieve High-quality Simultaneous Machine Translation as Efficiently as Offline
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.09570v2](http://arxiv.org/pdf/2504.09570v2)**

> **作者:** Biao Fu; Minpeng Liao; Kai Fan; Chengxi Li; Liang Zhang; Yidong Chen; Xiaodong Shi
>
> **备注:** Camera ready version for ACL 2025 Findings
>
> **摘要:** When the complete source sentence is provided, Large Language Models (LLMs) perform excellently in offline machine translation even with a simple prompt "Translate the following sentence from [src lang] into [tgt lang]:". However, in many real scenarios, the source tokens arrive in a streaming manner and simultaneous machine translation (SiMT) is required, then the efficiency and performance of decoder-only LLMs are significantly limited by their auto-regressive nature. To enable LLMs to achieve high-quality SiMT as efficiently as offline translation, we propose a novel paradigm that includes constructing supervised fine-tuning (SFT) data for SiMT, along with new training and inference strategies. To replicate the token input/output stream in SiMT, the source and target tokens are rearranged into an interleaved sequence, separated by special tokens according to varying latency requirements. This enables powerful LLMs to learn read and write operations adaptively, based on varying latency prompts, while still maintaining efficient auto-regressive decoding. Experimental results show that, even with limited SFT data, our approach achieves state-of-the-art performance across various SiMT benchmarks, and preserves the original abilities of offline translation. Moreover, our approach generalizes well to document-level SiMT setting without requiring specific fine-tuning, even beyond the offline translation model.
>
---
#### [replaced 038] Reducing Tool Hallucination via Reliability Alignment
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.04141v3](http://arxiv.org/pdf/2412.04141v3)**

> **作者:** Hongshen Xu; Zichen Zhu; Lei Pan; Zihan Wang; Su Zhu; Da Ma; Ruisheng Cao; Lu Chen; Kai Yu
>
> **摘要:** Large Language Models (LLMs) have expanded their capabilities beyond language generation to interact with external tools, enabling automation and real-world applications. However, tool hallucinations, where models either select inappropriate tools or misuse them, pose significant challenges, leading to erroneous task execution, increased computational costs, and reduced system reliability. To systematically address this issue, we define and categorize tool hallucinations into two main types, tool selection hallucination and tool usage hallucination. To evaluate and mitigate these issues, we introduce RelyToolBench, which integrates specialized test cases and novel metrics to assess hallucination-aware task success and efficiency. Finally, we propose Relign, a reliability alignment framework that expands the tool-use action space to include indecisive actions, allowing LLMs to defer tool use, seek clarification, or adjust tool selection dynamically. Through extensive experiments, we demonstrate that Relign significantly reduces tool hallucinations, improves task reliability, and enhances the efficiency of LLM tool interactions.
>
---
#### [replaced 039] DREAM: Drafting with Refined Target Features and Entropy-Adaptive Cross-Attention Fusion for Multimodal Speculative Decoding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.19201v2](http://arxiv.org/pdf/2505.19201v2)**

> **作者:** Yunhai Hu; Tianhua Xia; Zining Liu; Rahul Raman; Xingyu Liu; Bo Bao; Eric Sather; Vithursan Thangarasa; Sai Qian Zhang
>
> **摘要:** Speculative decoding (SD) has emerged as a powerful method for accelerating autoregressive generation in large language models (LLMs), yet its integration into vision-language models (VLMs) remains underexplored. We introduce DREAM, a novel speculative decoding framework tailored for VLMs that combines three key innovations: (1) a cross-attention-based mechanism to inject intermediate features from the target model into the draft model for improved alignment, (2) adaptive intermediate feature selection based on attention entropy to guide efficient draft model training, and (3) visual token compression to reduce draft model latency. DREAM enables efficient, accurate, and parallel multimodal decoding with significant throughput improvement. Experiments across a diverse set of recent popular VLMs, including LLaVA, Pixtral, SmolVLM and Gemma3, demonstrate up to 3.6x speedup over conventional decoding and significantly outperform prior SD baselines in both inference throughput and speculative draft acceptance length across a broad range of multimodal benchmarks. The code is publicly available at: https://github.com/SAI-Lab-NYU/DREAM.git
>
---
#### [replaced 040] Comparing Human and AI Rater Effects Using the Many-Facet Rasch Model
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.18486v2](http://arxiv.org/pdf/2505.18486v2)**

> **作者:** Hong Jiao; Dan Song; Won-Chan Lee
>
> **摘要:** Large language models (LLMs) have been widely explored for automated scoring in low-stakes assessment to facilitate learning and instruction. Empirical evidence related to which LLM produces the most reliable scores and induces least rater effects needs to be collected before the use of LLMs for automated scoring in practice. This study compared ten LLMs (ChatGPT 3.5, ChatGPT 4, ChatGPT 4o, OpenAI o1, Claude 3.5 Sonnet, Gemini 1.5, Gemini 1.5 Pro, Gemini 2.0, as well as DeepSeek V3, and DeepSeek R1) with human expert raters in scoring two types of writing tasks. The accuracy of the holistic and analytic scores from LLMs compared with human raters was evaluated in terms of Quadratic Weighted Kappa. Intra-rater consistency across prompts was compared in terms of Cronbach Alpha. Rater effects of LLMs were evaluated and compared with human raters using the Many-Facet Rasch model. The results in general supported the use of ChatGPT 4o, Gemini 1.5 Pro, and Claude 3.5 Sonnet with high scoring accuracy, better rater reliability, and less rater effects.
>
---
#### [replaced 041] CodeSteer: Symbolic-Augmented Language Models via Code/Text Guidance
- **分类: cs.CL; cs.AI; cs.LG; cs.SC; cs.SE**

- **链接: [http://arxiv.org/pdf/2502.04350v2](http://arxiv.org/pdf/2502.04350v2)**

> **作者:** Yongchao Chen; Yilun Hao; Yueying Liu; Yang Zhang; Chuchu Fan
>
> **备注:** 28 pages, 12 figures
>
> **摘要:** Existing methods fail to effectively steer Large Language Models (LLMs) between textual reasoning and code generation, leaving symbolic computing capabilities underutilized. We introduce CodeSteer, an effective method for guiding LLM code/text generation. We construct a comprehensive benchmark SymBench comprising 37 symbolic tasks with adjustable complexity and also synthesize datasets of 12k multi-turn guidance/generation trajectories and 5.5k guidance comparison pairs. We fine-tune the Llama-3-8B model with a newly designed multi-turn supervised fine-tuning (SFT) and direct preference optimization (DPO). The resulting model, CodeSteerLLM, augmented with the proposed symbolic and self-answer checkers, effectively guides the code/text generation of larger models. Augmenting GPT-4o with CodeSteer raises its average performance score from 53.3 to 86.4, even outperforming the existing best LLM OpenAI o1 (82.7), o1-preview (74.8), and DeepSeek R1 (76.8) across all 37 tasks (28 seen, 9 unseen). Trained for GPT-4o, CodeSteer demonstrates superior generalizability, providing an average 41.8 performance boost on Claude, Mistral, and GPT-3.5. CodeSteer-guided LLMs fully harness symbolic computing to maintain strong performance on highly complex tasks. Models, Datasets, and Codes are available at https://github.com/yongchao98/CodeSteer-v1.0 and https://huggingface.co/yongchao98.
>
---
#### [replaced 042] SOTOPIA-$Ω$: Dynamic Strategy Injection Learning and Social Instruction Following Evaluation for Social Agents
- **分类: cs.CL; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2502.15538v3](http://arxiv.org/pdf/2502.15538v3)**

> **作者:** Wenyuan Zhang; Tianyun Liu; Mengxiao Song; Xiaodong Li; Tingwen Liu
>
> **备注:** Accepted by ACL 2025 (Main Conference)
>
> **摘要:** Despite the abundance of prior social strategies possessed by humans, there remains a paucity of research dedicated to their transfer and integration into social agents. Our proposed SOTOPIA-$\Omega$ framework aims to address and bridge this gap, with a particular focus on enhancing the social capabilities of language agents. This framework dynamically injects multi-step reasoning strategies inspired by negotiation theory and two simple direct strategies into expert agents, thereby automating the construction of a high-quality social dialogue training corpus. Additionally, we introduce the concept of Social Instruction Following (S-IF) and propose two new S-IF evaluation metrics that complement social capability. We demonstrate that several 7B models trained on high-quality corpus not only significantly surpass the expert agent (GPT-4) in achieving social goals but also enhance S-IF performance. Analysis and variant experiments validate the advantages of dynamic construction, which can especially break the agent's prolonged deadlock.
>
---
#### [replaced 043] BioProBench: Comprehensive Dataset and Benchmark in Biological Protocol Understanding and Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.07889v2](http://arxiv.org/pdf/2505.07889v2)**

> **作者:** Yuyang Liu; Liuzhenghao Lv; Xiancheng Zhang; Li Yuan; Yonghong Tian
>
> **摘要:** Biological protocols are fundamental to reproducibility and safety in life science research. While large language models (LLMs) perform well on general tasks, their systematic evaluation on these highly specialized, accuracy-critical, and inherently procedural texts remains limited. In this work, we present BioProBench, the first large-scale, multi-task benchmark for biological protocol understanding and reasoning. While there are several benchmark tasks involving protocol question answering, BioProBench provides a comprehensive suite of five core tasks: Protocol Question Answering, Step Ordering, Error Correction, Protocol Generation, and Protocol Reasoning, enabling a holistic evaluation of LLMs on procedural biological texts. Built upon 27K original protocols, it yields nearly 556K high-quality structured instances. We evaluate 12 mainstream open/closed-source LLMs. Experimental results reveal that some models perform well on basic understanding tasks (e.g., \sim70% PQA-Acc., >64% ERR F1), but struggle significantly with deep reasoning and structured generation tasks like ordering and generation. Furthermore, model comparisons show diverse performance: certain open-source models approach closed-source levels on some tasks, yet bio-specific small models lag behind general LLMs, indicating limitations on complex procedural content. Overall, BioProBench, through its task design and experimental findings, systematically reveals the fundamental challenges for current LLMs in procedural knowledge understanding, deep adaptability to specific domains, reliability of structured reasoning, and handling of sophisticated precision and safety constraints, providing key directions for future AI in the field of scientific experiment automation. The code and data are available at: https://github.com/YuyangSunshine/bioprotocolbench and https://huggingface.co/datasets/BioProBench/BioProBench.
>
---
#### [replaced 044] Multi-Modal Framing Analysis of News
- **分类: cs.CL; cs.CY; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.20960v3](http://arxiv.org/pdf/2503.20960v3)**

> **作者:** Arnav Arora; Srishti Yadav; Maria Antoniak; Serge Belongie; Isabelle Augenstein
>
> **摘要:** Automated frame analysis of political communication is a popular task in computational social science that is used to study how authors select aspects of a topic to frame its reception. So far, such studies have been narrow, in that they use a fixed set of pre-defined frames and focus only on the text, ignoring the visual contexts in which those texts appear. Especially for framing in the news, this leaves out valuable information about editorial choices, which include not just the written article but also accompanying photographs. To overcome such limitations, we present a method for conducting multi-modal, multi-label framing analysis at scale using large (vision-) language models. Grounding our work in framing theory, we extract latent meaning embedded in images used to convey a certain point and contrast that to the text by comparing the respective frames used. We also identify highly partisan framing of topics with issue-specific frame analysis found in prior qualitative work. We demonstrate a method for doing scalable integrative framing analysis of both text and image in news, providing a more complete picture for understanding media bias.
>
---
#### [replaced 045] Agent-UniRAG: A Trainable Open-Source LLM Agent Framework for Unified Retrieval-Augmented Generation Systems
- **分类: cs.CL; cs.AI; cs.DB; cs.IR**

- **链接: [http://arxiv.org/pdf/2505.22571v2](http://arxiv.org/pdf/2505.22571v2)**

> **作者:** Hoang Pham; Thuy-Duong Nguyen; Khac-Hoai Nam Bui
>
> **摘要:** This paper presents a novel approach for unified retrieval-augmented generation (RAG) systems using the recent emerging large language model (LLM) agent concept. Specifically, Agent LLM, which utilizes LLM as fundamental controllers, has become a promising approach to enable the interpretability of RAG tasks, especially for complex reasoning question-answering systems (e.g., multi-hop queries). Nonetheless, previous works mainly focus on solving RAG systems with either single-hop or multi-hop approaches separately, which limits the application of those approaches to real-world applications. In this study, we propose a trainable agent framework called Agent-UniRAG for unified retrieval-augmented LLM systems, which enhances the effectiveness and interpretability of RAG systems. The main idea is to design an LLM agent framework to solve RAG tasks step-by-step based on the complexity of the inputs, simultaneously including single-hop and multi-hop queries in an end-to-end manner. Furthermore, we introduce SynAgent-RAG, a synthetic dataset to enable the proposed agent framework for small open-source LLMs (e.g., Llama-3-8B). The results show comparable performances with closed-source and larger open-source LLMs across various RAG benchmarks. Our source code and dataset are publicly available for further exploitation.
>
---
#### [replaced 046] Instruction-Tuning LLMs for Event Extraction with Annotation Guidelines
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.16377v2](http://arxiv.org/pdf/2502.16377v2)**

> **作者:** Saurabh Srivastava; Sweta Pati; Ziyu Yao
>
> **备注:** Accepted at ACL Findings 2025
>
> **摘要:** In this work, we study the effect of annotation guidelines -- textual descriptions of event types and arguments, when instruction-tuning large language models for event extraction. We conducted a series of experiments with both human-provided and machine-generated guidelines in both full- and low-data settings. Our results demonstrate the promise of annotation guidelines when there is a decent amount of training data and highlight its effectiveness in improving cross-schema generalization and low-frequency event-type performance.
>
---
#### [replaced 047] Improving Brain-to-Image Reconstruction via Fine-Grained Text Bridging
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22150v2](http://arxiv.org/pdf/2505.22150v2)**

> **作者:** Runze Xia; Shuo Feng; Renzhi Wang; Congchi Yin; Xuyun Wen; Piji Li
>
> **备注:** CogSci2025
>
> **摘要:** Brain-to-Image reconstruction aims to recover visual stimuli perceived by humans from brain activity. However, the reconstructed visual stimuli often missing details and semantic inconsistencies, which may be attributed to insufficient semantic information. To address this issue, we propose an approach named Fine-grained Brain-to-Image reconstruction (FgB2I), which employs fine-grained text as bridge to improve image reconstruction. FgB2I comprises three key stages: detail enhancement, decoding fine-grained text descriptions, and text-bridged brain-to-image reconstruction. In the detail-enhancement stage, we leverage large vision-language models to generate fine-grained captions for visual stimuli and experimentally validate its importance. We propose three reward metrics (object accuracy, text-image semantic similarity, and image-image semantic similarity) to guide the language model in decoding fine-grained text descriptions from fMRI signals. The fine-grained text descriptions can be integrated into existing reconstruction methods to achieve fine-grained Brain-to-Image reconstruction.
>
---
#### [replaced 048] Resolving Lexical Bias in Model Editing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.10411v3](http://arxiv.org/pdf/2408.10411v3)**

> **作者:** Hammad Rizwan; Domenic Rosati; Ga Wu; Hassan Sajjad
>
> **摘要:** Model editing aims to modify the outputs of large language models after they are trained. Previous approaches have often involved direct alterations to model weights, which can result in model degradation. Recent techniques avoid making modifications to the model's weights by using an adapter that applies edits to the model when triggered by semantic similarity in the representation space. We demonstrate that current adapter methods are critically vulnerable to strong lexical biases, leading to issues such as applying edits to irrelevant prompts with overlapping words. This paper presents a principled approach to learning a disentangled representation space that facilitates precise localization of edits by maintaining distance between irrelevant prompts while preserving proximity among paraphrases. In our empirical study, we show that our method (Projector Editor Networks for Model Editing - PENME) achieves state-of-the-art model editing results while being more computationally efficient during inference than previous methods and adaptable across different architectures.
>
---
#### [replaced 049] LongFaith: Enhancing Long-Context Reasoning in LLMs with Faithful Synthetic Data
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12583v2](http://arxiv.org/pdf/2502.12583v2)**

> **作者:** Cehao Yang; Xueyuan Lin; Chengjin Xu; Xuhui Jiang; Shengjie Ma; Aofan Liu; Hui Xiong; Jian Guo
>
> **摘要:** Despite the growing development of long-context large language models (LLMs), data-centric approaches relying on synthetic data have been hindered by issues related to faithfulness, which limit their effectiveness in enhancing model performance on tasks such as long-context reasoning and question answering (QA). These challenges are often exacerbated by misinformation caused by lack of verification, reasoning without attribution, and potential knowledge conflicts. We propose LongFaith, a novel pipeline for synthesizing faithful long-context reasoning instruction datasets. By integrating ground truth and citation-based reasoning prompts, we eliminate distractions and improve the accuracy of reasoning chains, thus mitigating the need for costly verification processes. We open-source two synthesized datasets, LongFaith-SFT and LongFaith-PO, which systematically address multiple dimensions of faithfulness, including verified reasoning, attribution, and contextual grounding. Extensive experiments on multi-hop reasoning datasets and LongBench demonstrate that models fine-tuned on these datasets significantly improve performance. Our ablation studies highlight the scalability and adaptability of the LongFaith pipeline, showcasing its broad applicability in developing long-context LLMs.
>
---
#### [replaced 050] Multilingual Question Answering in Low-Resource Settings: A Dzongkha-English Benchmark for Foundation Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18638v2](http://arxiv.org/pdf/2505.18638v2)**

> **作者:** Md. Tanzib Hosain; Rajan Das Gupta; Md. Kishor Morol
>
> **备注:** 24 pages, 20 figures
>
> **摘要:** In this work, we provide DZEN, a dataset of parallel Dzongkha and English test questions for Bhutanese middle and high school students. The over 5K questions in our collection span a variety of scientific topics and include factual, application, and reasoning-based questions. We use our parallel dataset to test a number of Large Language Models (LLMs) and find a significant performance difference between the models in English and Dzongkha. We also look at different prompting strategies and discover that Chain-of-Thought (CoT) prompting works well for reasoning questions but less well for factual ones. We also find that adding English translations enhances the precision of Dzongkha question responses. Our results point to exciting avenues for further study to improve LLM performance in Dzongkha and, more generally, in low-resource languages. We release the dataset at: https://github.com/kraritt/llm_dzongkha_evaluation.
>
---
#### [replaced 051] $T^5Score$: A Methodology for Automatically Assessing the Quality of LLM Generated Multi-Document Topic Sets
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.17390v3](http://arxiv.org/pdf/2407.17390v3)**

> **作者:** Itamar Trainin; Omri Abend
>
> **备注:** Published in the Findings of ACL 2025
>
> **摘要:** Using LLMs for Multi-Document Topic Extraction has recently gained popularity due to their apparent high-quality outputs, expressiveness, and ease of use. However, most existing evaluation practices are not designed for LLM-generated topics and result in low inter-annotator agreement scores, hindering the reliable use of LLMs for the task. To address this, we introduce $T^5Score$, an evaluation methodology that decomposes the quality of a topic set into quantifiable aspects, measurable through easy-to-perform annotation tasks. This framing enables a convenient, manual or automatic, evaluation procedure resulting in a strong inter-annotator agreement score. To substantiate our methodology and claims, we perform extensive experimentation on multiple datasets and report the results.
>
---
#### [replaced 052] Firm or Fickle? Evaluating Large Language Models Consistency in Sequential Interactions
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.22353v2](http://arxiv.org/pdf/2503.22353v2)**

> **作者:** Yubo Li; Yidi Miao; Xueying Ding; Ramayya Krishnan; Rema Padman
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Large Language Models (LLMs) have shown remarkable capabilities across various tasks, but their deployment in high-stake domains requires consistent performance across multiple interaction rounds. This paper introduces a comprehensive framework for evaluating and improving LLM response consistency, making three key contributions. First, we propose a novel Position-Weighted Consistency (PWC) score that captures both the importance of early-stage stability and recovery patterns in multi-turn interactions. Second, we present a carefully curated benchmark dataset spanning diverse domains and difficulty levels, specifically designed to evaluate LLM consistency under various challenging follow-up scenarios. Third, we introduce Confidence-Aware Response Generation (CARG), a framework that significantly improves response stability by incorporating model confidence signals into the generation process. Empirical results demonstrate that CARG significantly improves response stability without sacrificing accuracy, underscoring its potential for reliable LLM deployment in critical applications.
>
---
#### [replaced 053] GraphNarrator: Generating Textual Explanations for Graph Neural Networks
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.15268v2](http://arxiv.org/pdf/2410.15268v2)**

> **作者:** Bo Pan; Zhen Xiong; Guanchen Wu; Zheng Zhang; Yifei Zhang; Liang Zhao
>
> **备注:** ACL 2025 (Main)
>
> **摘要:** Graph representation learning has garnered significant attention due to its broad applications in various domains, such as recommendation systems and social network analysis. Despite advancements in graph learning methods, challenges still remain in explainability when graphs are associated with semantic features. In this paper, we present GraphNarrator, the first method designed to generate natural language explanations for Graph Neural Networks. GraphNarrator employs a generative language model that maps input-output pairs to explanations reflecting the model's decision-making process. To address the lack of ground truth explanations to train the model, we propose first generating pseudo-labels that capture the model's decisions from saliency-based explanations, then using Expert Iteration to iteratively train the pseudo-label generator based on training objectives on explanation quality. The high-quality pseudo-labels are finally utilized to train an end-to-end explanation generator model. Extensive experiments are conducted to demonstrate the effectiveness of GraphNarrator in producing faithful, concise, and human-preferred natural language explanations.
>
---
#### [replaced 054] YESciEval: Robust LLM-as-a-Judge for Scientific Question Answering
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.14279v2](http://arxiv.org/pdf/2505.14279v2)**

> **作者:** Jennifer D'Souza; Hamed Babaei Giglou; Quentin Münch
>
> **备注:** 9 pages, 4 figures, Accepted as a Long Paper at the 63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)
>
> **摘要:** Large Language Models (LLMs) drive scientific question-answering on modern search engines, yet their evaluation robustness remains underexplored. We introduce YESciEval, an open-source framework that combines fine-grained rubric-based assessment with reinforcement learning to mitigate optimism bias in LLM evaluators. We release multidisciplinary scienceQ&A datasets, including adversarial variants, with evaluation scores from multiple LLMs. Independent of proprietary models and human feedback, our approach enables scalable, cost-free evaluation. By advancing reliable LLM-as-a-judge models, this work supports AI alignment and fosters robust, transparent evaluation essential for scientific inquiry.
>
---
#### [replaced 055] Multimodal Inverse Attention Network with Intrinsic Discriminant Feature Exploitation for Fake News Detection
- **分类: cs.LG; cs.CL; cs.CV; cs.IR; cs.MM**

- **链接: [http://arxiv.org/pdf/2502.01699v2](http://arxiv.org/pdf/2502.01699v2)**

> **作者:** Tianlin Zhang; En Yu; Yi Shao; Jiande Sun
>
> **备注:** 13 pages, 6 figures
>
> **摘要:** Multimodal fake news detection has garnered significant attention due to its profound implications for social security. While existing approaches have contributed to understanding cross-modal consistency, they often fail to leverage modal-specific representations and explicit discrepant features. To address these limitations, we propose a Multimodal Inverse Attention Network (MIAN), a novel framework that explores intrinsic discriminative features based on news content to advance fake news detection. Specifically, MIAN introduces a hierarchical learning module that captures diverse intra-modal relationships through local-to-global and local-to-local interactions, thereby generating enhanced unimodal representations to improve the identification of fake news at the intra-modal level. Additionally, a cross-modal interaction module employs a co-attention mechanism to establish and model dependencies between the refined unimodal representations, facilitating seamless semantic integration across modalities. To explicitly extract inconsistency features, we propose an inverse attention mechanism that effectively highlights the conflicting patterns and semantic deviations introduced by fake news in both intra- and inter-modality. Extensive experiments on benchmark datasets demonstrate that MIAN significantly outperforms state-of-the-art methods, underscoring its pivotal contribution to advancing social security through enhanced multimodal fake news detection.
>
---
#### [replaced 056] Multi-Domain Explainability of Preferences
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.20088v2](http://arxiv.org/pdf/2505.20088v2)**

> **作者:** Nitay Calderon; Liat Ein-Dor; Roi Reichart
>
> **摘要:** Preference mechanisms, such as human preference, LLM-as-a-Judge (LaaJ), and reward models, are central to aligning and evaluating large language models (LLMs). Yet, the underlying concepts that drive these preferences remain poorly understood. In this work, we propose a fully automated method for generating local and global concept-based explanations of preferences across multiple domains. Our method utilizes an LLM to identify concepts that distinguish between chosen and rejected responses, and to represent them with concept-based vectors. To model the relationships between concepts and preferences, we propose a white-box Hierarchical Multi-Domain Regression model that captures both domain-general and domain-specific effects. To evaluate our method, we curate a dataset spanning eight challenging and diverse domains and explain twelve mechanisms. Our method achieves strong preference prediction performance, outperforming baselines while also being explainable. Additionally, we assess explanations in two application-driven settings. First, guiding LLM outputs with concepts from LaaJ explanations yields responses that those judges consistently prefer. Second, prompting LaaJs with concepts explaining humans improves their preference predictions. Together, our work establishes a new paradigm for explainability in the era of LLMs.
>
---
#### [replaced 057] FutureGen: LLM-RAG Approach to Generate the Future Work of Scientific Article
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.16561v2](http://arxiv.org/pdf/2503.16561v2)**

> **作者:** Ibrahim Al Azher; Miftahul Jannat Mokarrama; Zhishuai Guo; Sagnik Ray Choudhury; Hamed Alhoori
>
> **备注:** 19 pages, 5 figures
>
> **摘要:** The future work section of a scientific article outlines potential research directions by identifying gaps and limitations of a current study. This section serves as a valuable resource for early-career researchers seeking unexplored areas and experienced researchers looking for new projects or collaborations. In this study, we generate future work suggestions from key sections of a scientific article alongside related papers and analyze how the trends have evolved. We experimented with various Large Language Models (LLMs) and integrated Retrieval-Augmented Generation (RAG) to enhance the generation process. We incorporate a LLM feedback mechanism to improve the quality of the generated content and propose an LLM-as-a-judge approach for evaluation. Our results demonstrated that the RAG-based approach with LLM feedback outperforms other methods evaluated through qualitative and quantitative metrics. Moreover, we conduct a human evaluation to assess the LLM as an extractor and judge. The code and dataset for this project are here, code: HuggingFace
>
---
#### [replaced 058] Unveiling Environmental Impacts of Large Language Model Serving: A Functional Unit View
- **分类: cs.LG; cs.AR; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11256v2](http://arxiv.org/pdf/2502.11256v2)**

> **作者:** Yanran Wu; Inez Hua; Yi Ding
>
> **备注:** 17 pages, 38 figures, Proceedings of the The 63rd Annual Meeting of the Association for Computational Linguistics, Vienna, Austria, July 27-August 1st, 2025
>
> **摘要:** Large language models (LLMs) offer powerful capabilities but come with significant environmental impact, particularly in carbon emissions. Existing studies benchmark carbon emissions but lack a standardized basis for comparison across different model configurations. To address this, we introduce the concept of functional unit (FU) as a standardized basis and develop FUEL, the first FU-based framework for evaluating LLM serving's environmental impact. Through three case studies, we uncover key insights and trade-offs in reducing carbon emissions by optimizing model size, quantization strategy, and hardware choice, paving the way for more sustainable LLM serving. The code is available at https://github.com/jojacola/FUEL.
>
---
#### [replaced 059] KBQA-o1: Agentic Knowledge Base Question Answering with Monte Carlo Tree Search
- **分类: cs.CL; cs.AI; cs.DB**

- **链接: [http://arxiv.org/pdf/2501.18922v3](http://arxiv.org/pdf/2501.18922v3)**

> **作者:** Haoran Luo; Haihong E; Yikai Guo; Qika Lin; Xiaobao Wu; Xinyu Mu; Wenhao Liu; Meina Song; Yifan Zhu; Luu Anh Tuan
>
> **备注:** Accepted by ICML 2025 main conference
>
> **摘要:** Knowledge Base Question Answering (KBQA) aims to answer natural language questions with a large-scale structured knowledge base (KB). Despite advancements with large language models (LLMs), KBQA still faces challenges in weak KB awareness, imbalance between effectiveness and efficiency, and high reliance on annotated data. To address these challenges, we propose KBQA-o1, a novel agentic KBQA method with Monte Carlo Tree Search (MCTS). It introduces a ReAct-based agent process for stepwise logical form generation with KB environment exploration. Moreover, it employs MCTS, a heuristic search method driven by policy and reward models, to balance agentic exploration's performance and search space. With heuristic exploration, KBQA-o1 generates high-quality annotations for further improvement by incremental fine-tuning. Experimental results show that KBQA-o1 outperforms previous low-resource KBQA methods with limited annotated data, boosting Llama-3.1-8B model's GrailQA F1 performance to 78.5% compared to 48.5% of the previous sota method with GPT-3.5-turbo. Our code is publicly available.
>
---
#### [replaced 060] A Reality Check on Context Utilisation for Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.17031v2](http://arxiv.org/pdf/2412.17031v2)**

> **作者:** Lovisa Hagström; Sara Vera Marjanović; Haeun Yu; Arnav Arora; Christina Lioma; Maria Maistro; Pepa Atanasova; Isabelle Augenstein
>
> **备注:** Accepted at ACL 2025
>
> **摘要:** Retrieval-augmented generation (RAG) helps address the limitations of parametric knowledge embedded within a language model (LM). In real world settings, retrieved information can vary in complexity, yet most investigations of LM utilisation of context has been limited to synthetic text. We introduce DRUID (Dataset of Retrieved Unreliable, Insufficient and Difficult-to-understand contexts) with real-world queries and contexts manually annotated for stance. The dataset is based on the prototypical task of automated claim verification, for which automated retrieval of real-world evidence is crucial. We compare DRUID to synthetic datasets (CounterFact, ConflictQA) and find that artificial datasets often fail to represent the complexity and diversity of realistically retrieved context. We show that synthetic datasets exaggerate context characteristics rare in real retrieved data, which leads to inflated context utilisation results, as measured by our novel ACU score. Moreover, while previous work has mainly focused on singleton context characteristics to explain context utilisation, correlations between singleton context properties and ACU on DRUID are surprisingly small compared to other properties related to context source. Overall, our work underscores the need for real-world aligned context utilisation studies to represent and improve performance in real-world RAG settings.
>
---
#### [replaced 061] PolyMath: Evaluating Mathematical Reasoning in Multilingual Contexts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.18428v3](http://arxiv.org/pdf/2504.18428v3)**

> **作者:** Yiming Wang; Pei Zhang; Jialong Tang; Haoran Wei; Baosong Yang; Rui Wang; Chenshu Sun; Feitong Sun; Jiran Zhang; Junxuan Wu; Qiqian Cang; Yichang Zhang; Fei Huang; Junyang Lin; Fei Huang; Jingren Zhou
>
> **备注:** 50 pages, 19 tables, 9 figures
>
> **摘要:** In this paper, we introduce PolyMath, a multilingual mathematical reasoning benchmark covering 18 languages and 4 easy-to-hard difficulty levels. Our benchmark ensures difficulty comprehensiveness, language diversity, and high-quality translation, making it a highly discriminative multilingual mathematical benchmark in the era of reasoning LLMs. We conduct a comprehensive evaluation for advanced LLMs and find that even Qwen-3-235B-A22B-Thinking and Gemini-2.5-pro, achieve only 54.6 and 52.2 benchmark scores, with about 40% accuracy under the highest level From a language perspective, our benchmark reveals several key challenges of LLMs in multilingual reasoning: (1) Reasoning performance varies widely across languages for current LLMs; (2) Input-output language consistency is low in reasoning LLMs and may be correlated with performance; (3) The thinking length differs significantly by language for current LLMs. Additionally, we demonstrate that controlling the output language in the instructions has the potential to affect reasoning performance, especially for some low-resource languages, suggesting a promising direction for improving multilingual capabilities in LLMs.
>
---
#### [replaced 062] Speculative Decoding Meets Quantization: Compatibility Evaluation and Hierarchical Framework Design
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.22179v2](http://arxiv.org/pdf/2505.22179v2)**

> **作者:** Yudi Zhang; Weilin Zhao; Xu Han; Tiejun Zhao; Wang Xu; Hailong Cao; Conghui Zhu
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Speculative decoding and quantization effectively accelerate memory-bound inference of large language models. Speculative decoding mitigates the memory bandwidth bottleneck by verifying multiple tokens within a single forward pass, which increases computational effort. Quantization achieves this optimization by compressing weights and activations into lower bit-widths and also reduces computations via low-bit matrix multiplications. To further leverage their strengths, we investigate the integration of these two techniques. Surprisingly, experiments applying the advanced speculative decoding method EAGLE-2 to various quantized models reveal that the memory benefits from 4-bit weight quantization are diminished by the computational load from speculative decoding. Specifically, verifying a tree-style draft incurs significantly more time overhead than a single-token forward pass on 4-bit weight quantized models. This finding led to our new speculative decoding design: a hierarchical framework that employs a small model as an intermediate stage to turn tree-style drafts into sequence drafts, leveraging the memory access benefits of the target quantized model. Experimental results show that our hierarchical approach achieves a 2.78$\times$ speedup across various tasks for the 4-bit weight Llama-3-70B model on an A100 GPU, outperforming EAGLE-2 by 1.31$\times$. Code available at https://github.com/AI9Stars/SpecMQuant.
>
---
#### [replaced 063] VietASR: Achieving Industry-level Vietnamese ASR with 50-hour labeled data and Large-Scale Speech Pretraining
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **链接: [http://arxiv.org/pdf/2505.21527v2](http://arxiv.org/pdf/2505.21527v2)**

> **作者:** Jianheng Zhuo; Yifan Yang; Yiwen Shao; Yong Xu; Dong Yu; Kai Yu; Xie Chen
>
> **摘要:** Automatic speech recognition (ASR) has made remarkable progress but heavily relies on large-scale labeled data, which is scarce for low-resource languages like Vietnamese. While existing systems such as Whisper, USM, and MMS achieve promising performance, their efficacy remains inadequate in terms of training costs, latency, and accessibility. To address these issues, we propose VietASR, a novel ASR training pipeline that leverages vast amounts of unlabeled data and a small set of labeled data. Through multi-iteration ASR-biased self-supervised learning on a large-scale unlabeled dataset, VietASR offers a cost-effective and practical solution for enhancing ASR performance. Experiments demonstrate that pre-training on 70,000-hour unlabeled data and fine-tuning on merely 50-hour labeled data yield a lightweight but powerful ASR model. It outperforms Whisper Large-v3 and commercial ASR systems on real-world data. Our code and models will be open-sourced to facilitate research in low-resource ASR.
>
---
#### [replaced 064] Automatic Transmission for LLM Tiers: Optimizing Cost and Accuracy in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.20921v2](http://arxiv.org/pdf/2505.20921v2)**

> **作者:** Injae Na; Keonwoong Noh; Woohwan Jung
>
> **备注:** ACL 2025 (Findings)
>
> **摘要:** LLM providers typically offer multiple LLM tiers, varying in performance and price. As NLP tasks become more complex and modularized, selecting the suitable LLM tier for each subtask is a key challenge to balance between cost and performance. To address the problem, we introduce LLM Automatic Transmission (LLM-AT) framework that automatically selects LLM tiers without training. LLM-AT consists of Starter, Generator, and Judge. The starter selects the initial LLM tier expected to solve the given question, the generator produces a response using the LLM of the selected tier, and the judge evaluates the validity of the response. If the response is invalid, LLM-AT iteratively upgrades to a higher-tier model, generates a new response, and re-evaluates until a valid response is obtained. Additionally, we propose accuracy estimator, which enables the suitable initial LLM tier selection without training. Given an input question, accuracy estimator estimates the expected accuracy of each LLM tier by computing the valid response rate across top-k similar queries from past inference records. Experiments demonstrate that LLM-AT achieves superior performance while reducing costs, making it a practical solution for real-world applications.
>
---
#### [replaced 065] SequentialBreak: Large Language Models Can be Fooled by Embedding Jailbreak Prompts into Sequential Prompt Chains
- **分类: cs.CR; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2411.06426v3](http://arxiv.org/pdf/2411.06426v3)**

> **作者:** Bijoy Ahmed Saiem; MD Sadik Hossain Shanto; Rakib Ahsan; Md Rafi ur Rashid
>
> **摘要:** As the integration of the Large Language Models (LLMs) into various applications increases, so does their susceptibility to misuse, raising significant security concerns. Numerous jailbreak attacks have been proposed to assess the security defense of LLMs. Current jailbreak attacks mainly rely on scenario camouflage, prompt obfuscation, prompt optimization, and prompt iterative optimization to conceal malicious prompts. In particular, sequential prompt chains in a single query can lead LLMs to focus on certain prompts while ignoring others, facilitating context manipulation. This paper introduces SequentialBreak, a novel jailbreak attack that exploits this vulnerability. We discuss several scenarios, not limited to examples like Question Bank, Dialog Completion, and Game Environment, where the harmful prompt is embedded within benign ones that can fool LLMs into generating harmful responses. The distinct narrative structures of these scenarios show that SequentialBreak is flexible enough to adapt to various prompt formats beyond those discussed. Extensive experiments demonstrate that SequentialBreak uses only a single query to achieve a substantial gain of attack success rate over existing baselines against both open-source and closed-source models. Through our research, we highlight the urgent need for more robust and resilient safeguards to enhance LLM security and prevent potential misuse. All the result files and website associated with this research are available in this GitHub repository: https://anonymous.4open.science/r/JailBreakAttack-4F3B/.
>
---
#### [replaced 066] Exploring the Limitations of Mamba in COPY and CoT Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.03810v3](http://arxiv.org/pdf/2410.03810v3)**

> **作者:** Ruifeng Ren; Zhicong Li; Yong Liu
>
> **备注:** Mamba, Chain of Thought
>
> **摘要:** Transformers have become the backbone of modern Large Language Models (LLMs); however, their inference overhead grows linearly with the sequence length, posing challenges for modeling long sequences. In light of this, Mamba has attracted attention for maintaining a constant inference size, with empirical evidence demonstrating that it can match Transformer performance in sequence modeling while significantly reducing computational costs. However, an open question remains: can Mamba always bring savings while achieving performance comparable to Transformers? In this paper, we focus on analyzing the expressive ability of Mamba to perform our defined COPY operation and Chain of Thought (CoT) reasoning. First, inspired by the connection between Mamba and linear attention, we show that constant-sized Mamba may struggle to perform COPY operations while Transformers can handle them more easily. However, when the size of Mamba grows linearly with the input sequence length, it can accurately perform COPY, but in this case, Mamba no longer provides overhead savings. Based on this observation, we further analyze Mamba's ability to tackle CoT tasks, which can be described by the Dynamic Programming (DP) problems. Our findings suggest that to solve arbitrary DP problems, the total cost of Mamba is still comparable to standard Transformers. However, similar to efficient Transformers, when facing DP problems with favorable properties such as locality, Mamba can provide savings in overhead. Our experiments on the copy and CoT tasks further demonstrate Mamba's limitations compared to Transformers in learning these tasks.
>
---
#### [replaced 067] Improving Parallel Program Performance with LLM Optimizers via Agent-System Interfaces
- **分类: cs.LG; cs.AI; cs.CL; cs.DC**

- **链接: [http://arxiv.org/pdf/2410.15625v3](http://arxiv.org/pdf/2410.15625v3)**

> **作者:** Anjiang Wei; Allen Nie; Thiago S. F. X. Teixeira; Rohan Yadav; Wonchan Lee; Ke Wang; Alex Aiken
>
> **摘要:** Modern scientific discovery increasingly relies on high-performance computing for complex modeling and simulation. A key challenge in improving parallel program performance is efficiently mapping tasks to processors and data to memory, a process dictated by intricate, low-level system code known as mappers. Developing high-performance mappers demands days of manual tuning, posing a significant barrier for domain scientists without systems expertise. We introduce a framework that automates mapper development with generative optimization, leveraging richer feedback beyond scalar performance metrics. Our approach features the Agent-System Interface, which includes a Domain-Specific Language (DSL) to abstract away the low-level complexity of system code and define a structured search space, as well as AutoGuide, a mechanism that interprets raw execution output into actionable feedback. Unlike traditional reinforcement learning methods such as OpenTuner, which rely solely on scalar feedback, our method finds superior mappers in far fewer iterations. With just 10 iterations, it outperforms OpenTuner even after 1000 iterations, achieving 3.8X faster performance. Our approach finds mappers that surpass expert-written mappers by up to 1.34X speedup across nine benchmarks while reducing tuning time from days to minutes.
>
---
#### [replaced 068] Enhancing Automated Interpretability with Output-Centric Feature Descriptions
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.08319v2](http://arxiv.org/pdf/2501.08319v2)**

> **作者:** Yoav Gur-Arieh; Roy Mayan; Chen Agassy; Atticus Geiger; Mor Geva
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Automated interpretability pipelines generate natural language descriptions for the concepts represented by features in large language models (LLMs), such as plants or the first word in a sentence. These descriptions are derived using inputs that activate the feature, which may be a dimension or a direction in the model's representation space. However, identifying activating inputs is costly, and the mechanistic role of a feature in model behavior is determined both by how inputs cause a feature to activate and by how feature activation affects outputs. Using steering evaluations, we reveal that current pipelines provide descriptions that fail to capture the causal effect of the feature on outputs. To fix this, we propose efficient, output-centric methods for automatically generating feature descriptions. These methods use the tokens weighted higher after feature stimulation or the highest weight tokens after applying the vocabulary "unembedding" head directly to the feature. Our output-centric descriptions better capture the causal effect of a feature on model outputs than input-centric descriptions, but combining the two leads to the best performance on both input and output evaluations. Lastly, we show that output-centric descriptions can be used to find inputs that activate features previously thought to be "dead".
>
---
#### [replaced 069] Frankentext: Stitching random text fragments into long-form narratives
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18128v2](http://arxiv.org/pdf/2505.18128v2)**

> **作者:** Chau Minh Pham; Jenna Russell; Dzung Pham; Mohit Iyyer
>
> **摘要:** We introduce Frankentexts, a new type of long-form narratives produced by LLMs under the extreme constraint that most tokens (e.g., 90%) must be copied verbatim from human writings. This task presents a challenging test of controllable generation, requiring models to satisfy a writing prompt, integrate disparate text fragments, and still produce a coherent narrative. To generate Frankentexts, we instruct the model to produce a draft by selecting and combining human-written passages, then iteratively revise the draft while maintaining a user-specified copy ratio. We evaluate the resulting Frankentexts along three axes: writing quality, instruction adherence, and detectability. Gemini-2.5-Pro performs surprisingly well on this task: 81% of its Frankentexts are coherent and 100% relevant to the prompt. Notably, up to 59% of these outputs are misclassified as human-written by detectors like Pangram, revealing limitations in AI text detectors. Human annotators can sometimes identify Frankentexts through their abrupt tone shifts and inconsistent grammar between segments, especially in longer generations. Beyond presenting a challenging generation task, Frankentexts invite discussion on building effective detectors for this new grey zone of authorship, provide training data for mixed authorship detection, and serve as a sandbox for studying human-AI co-writing processes.
>
---
#### [replaced 070] Pandora's Box or Aladdin's Lamp: A Comprehensive Analysis Revealing the Role of RAG Noise in Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2408.13533v3](http://arxiv.org/pdf/2408.13533v3)**

> **作者:** Jinyang Wu; Shuai Zhang; Feihu Che; Mingkuan Feng; Pengpeng Shao; Jianhua Tao
>
> **备注:** ACL 2025 Main
>
> **摘要:** Retrieval-Augmented Generation (RAG) has emerged as a crucial method for addressing hallucinations in large language models (LLMs). While recent research has extended RAG models to complex noisy scenarios, these explorations often confine themselves to limited noise types and presuppose that noise is inherently detrimental to LLMs, potentially deviating from real-world retrieval environments and restricting practical applicability. In this paper, we define seven distinct noise types from a linguistic perspective and establish a Noise RAG Benchmark (NoiserBench), a comprehensive evaluation framework encompassing multiple datasets and reasoning tasks. Through empirical evaluation of eight representative LLMs with diverse architectures and scales, we reveal that these noises can be further categorized into two practical groups: noise that is beneficial to LLMs (aka beneficial noise) and noise that is harmful to LLMs (aka harmful noise). While harmful noise generally impairs performance, beneficial noise may enhance several aspects of model capabilities and overall performance. Our analysis offers insights for developing more robust, adaptable RAG solutions and mitigating hallucinations across diverse retrieval scenarios. Code is available at https://github.com/jinyangwu/NoiserBench.
>
---
#### [replaced 071] On the Risk of Evidence Pollution for Malicious Social Text Detection in the Era of LLMs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.12600v2](http://arxiv.org/pdf/2410.12600v2)**

> **作者:** Herun Wan; Minnan Luo; Zhixiong Su; Guang Dai; Xiang Zhao
>
> **摘要:** Evidence-enhanced detectors present remarkable abilities in identifying malicious social text. However, the rise of large language models (LLMs) brings potential risks of evidence pollution to confuse detectors. This paper explores potential manipulation scenarios including basic pollution, and rephrasing or generating evidence by LLMs. To mitigate the negative impact, we propose three defense strategies from the data and model sides, including machine-generated text detection, a mixture of experts, and parameter updating. Extensive experiments on four malicious social text detection tasks with ten datasets illustrate that evidence pollution significantly compromises detectors, where the generating strategy causes up to a 14.4% performance drop. Meanwhile, the defense strategies could mitigate evidence pollution, but they faced limitations for practical employment. Further analysis illustrates that polluted evidence (i) is of high quality, evaluated by metrics and humans; (ii) would compromise the model calibration, increasing expected calibration error up to 21.6%; and (iii) could be integrated to amplify the negative impact, especially for encoder-based LMs, where the accuracy drops by 21.8%.
>
---
#### [replaced 072] DReSD: Dense Retrieval for Speculative Decoding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.15572v2](http://arxiv.org/pdf/2502.15572v2)**

> **作者:** Milan Gritta; Huiyin Xue; Gerasimos Lampouras
>
> **备注:** ACL (Findings) 2025
>
> **摘要:** Speculative decoding (SD) accelerates Large Language Model (LLM) generation by using an efficient draft model to propose the next few tokens, which are verified by the LLM in a single forward call, reducing latency while preserving its outputs. We focus on retrieval-based SD where the draft model retrieves the next tokens from a non-parametric datastore. Sparse retrieval (REST), which operates on the surface form of strings, is currently the dominant paradigm due to its simplicity and scalability. However, its effectiveness is limited due to the usage of short contexts and exact string matching. Instead, we introduce Dense Retrieval for Speculative Decoding (DReSD), a novel framework that uses approximate nearest neighbour search with contextualised token embeddings to retrieve the most semantically relevant token sequences for SD. Extensive experiments show that DReSD achieves (on average) 87% higher acceptance rates, 65% longer accepted tokens and 19% faster generation speeds compared to sparse retrieval (REST).
>
---
#### [replaced 073] Human-Readable Adversarial Prompts: An Investigation into LLM Vulnerabilities Using Situational Context
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.16359v3](http://arxiv.org/pdf/2412.16359v3)**

> **作者:** Nilanjana Das; Edward Raff; Aman Chadha; Manas Gaur
>
> **备注:** arXiv admin note: text overlap with arXiv:2407.14644
>
> **摘要:** As the AI systems become deeply embedded in social media platforms, we've uncovered a concerning security vulnerability that goes beyond traditional adversarial attacks. It becomes important to assess the risks of LLMs before the general public use them on social media platforms to avoid any adverse impacts. Unlike obvious nonsensical text strings that safety systems can easily catch, our work reveals that human-readable situation-driven adversarial full-prompts that leverage situational context are effective but much harder to detect. We found that skilled attackers can exploit the vulnerabilities in open-source and proprietary LLMs to make a malicious user query safe for LLMs, resulting in generating a harmful response. This raises an important question about the vulnerabilities of LLMs. To measure the robustness against human-readable attacks, which now present a potent threat, our research makes three major contributions. First, we developed attacks that use movie scripts as situational contextual frameworks, creating natural-looking full-prompts that trick LLMs into generating harmful content. Second, we developed a method to transform gibberish adversarial text into readable, innocuous content that still exploits vulnerabilities when used within the full-prompts. Finally, we enhanced the AdvPrompter framework with p-nucleus sampling to generate diverse human-readable adversarial texts that significantly improve attack effectiveness against models like GPT-3.5-Turbo-0125 and Gemma-7b. Our findings show that these systems can be manipulated to operate beyond their intended ethical boundaries when presented with seemingly normal prompts that contain hidden adversarial elements. By identifying these vulnerabilities, we aim to drive the development of more robust safety mechanisms that can withstand sophisticated attacks in real-world applications.
>
---
#### [replaced 074] Learning to Reason from Feedback at Test-Time
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.15771v2](http://arxiv.org/pdf/2502.15771v2)**

> **作者:** Yanyang Li; Michael Lyu; Liwei Wang
>
> **备注:** ACL 2025 Main; Project Page: https://github.com/LaVi-Lab/FTTT
>
> **摘要:** Solving complex tasks in a single attempt is challenging for large language models (LLMs). Iterative interaction with the environment and feedback is often required to achieve success, making effective feedback utilization a critical topic. Existing approaches either struggle with length generalization or rely on naive retries without leveraging prior information. In this paper, we introduce FTTT, a novel paradigm that formulates feedback utilization as an optimization problem at test time. Additionally, we propose a learnable test-time optimizer, OpTune, to effectively exploit feedback. Experiments on two LLMs across four reasoning datasets demonstrate that FTTT and OpTune achieve superior scalability and performance.
>
---
#### [replaced 075] Instruct-SkillMix: A Powerful Pipeline for LLM Instruction Tuning
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2408.14774v4](http://arxiv.org/pdf/2408.14774v4)**

> **作者:** Simran Kaur; Simon Park; Anirudh Goyal; Sanjeev Arora
>
> **摘要:** We introduce Instruct-SkillMix, an automated approach for creating diverse, high quality SFT data for instruction-following. The pipeline involves two stages, each leveraging an existing powerful LLM: (1) Skill extraction: uses the LLM to extract core "skills" for instruction-following by directly prompting the model. This is inspired by ``LLM metacognition'' of Didolkar et al. (2024); (2) Data generation: uses the powerful LLM to generate (instruction, response) data that exhibit a randomly chosen pair of these skills. Here, the use of random skill combinations promotes diversity and difficulty. The estimated cost of creating the dataset is under $600. Vanilla SFT (i.e., no PPO, DPO, or RL methods) on data generated from Instruct-SkillMix leads to strong gains on instruction following benchmarks such as AlpacaEval 2.0, MT-Bench, and WildBench. With just 4K examples, LLaMA-3-8B-Base achieves 42.76% length-controlled win rate on AlpacaEval 2.0, a level similar to frontier models like Claude 3 Opus and LLaMA-3.1-405B-Instruct. Ablation studies also suggest plausible reasons for why creating open instruction-tuning datasets via naive crowd-sourcing has proved difficult. In our dataset, adding 20% low quality answers (``shirkers'') causes a noticeable degradation in performance. The Instruct-SkillMix pipeline seems flexible and adaptable to other settings.
>
---
#### [replaced 076] Are Generative Models Underconfident? Better Quality Estimation with Boosted Model Probability
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2502.11115v2](http://arxiv.org/pdf/2502.11115v2)**

> **作者:** Tu Anh Dinh; Jan Niehues
>
> **摘要:** Quality Estimation (QE) is estimating quality of the model output during inference when the ground truth is not available. Deriving output quality from the models' output probability is the most trivial and low-effort way. However, we show that the output probability of text-generation models can appear underconfident. At each output step, there can be multiple correct options, making the probability distribution spread out more. Thus, lower probability does not necessarily mean lower output quality. Due to this observation, we propose a QE approach called BoostedProb, which boosts the model's confidence in cases where there are multiple viable output options. With no increase in complexity, BoostedProb is notably better than raw model probability in different settings, achieving on average +0.194 improvement in Pearson correlation to ground-truth quality. It also comes close to or outperforms more costly approaches like supervised or ensemble-based QE in certain settings.
>
---
#### [replaced 077] Dataset Featurization: Uncovering Natural Language Features through Unsupervised Data Reconstruction
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.17541v2](http://arxiv.org/pdf/2502.17541v2)**

> **作者:** Michal Bravansky; Vaclav Kubon; Suhas Hariharan; Robert Kirk
>
> **摘要:** Interpreting data is central to modern research. Large language models (LLMs) show promise in providing such natural language interpretations of data, yet simple feature extraction methods such as prompting often fail to produce accurate and versatile descriptions for diverse datasets and lack control over granularity and scale. To address these limitations, we propose a domain-agnostic method for dataset featurization that provides precise control over the number of features extracted while maintaining compact and descriptive representations comparable to human labeling. Our method optimizes the selection of informative binary features by evaluating the ability of an LLM to reconstruct the original data using those features. We demonstrate its effectiveness in dataset modeling tasks and through two case studies: (1) Constructing a feature representation of jailbreak tactics that compactly captures both the effectiveness and diversity of a larger set of human-crafted attacks; and (2) automating the discovery of features that align with human preferences, achieving accuracy and robustness comparable to human-crafted features. Moreover, we show that the pipeline scales effectively, improving as additional features are sampled, making it suitable for large and diverse datasets.
>
---
#### [replaced 078] STeCa: Step-level Trajectory Calibration for LLM Agent Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14276v2](http://arxiv.org/pdf/2502.14276v2)**

> **作者:** Hanlin Wang; Jian Wang; Chak Tou Leong; Wenjie Li
>
> **备注:** Accepted by ACL2025 Findings
>
> **摘要:** Large language model (LLM)-based agents have shown promise in tackling complex tasks by interacting dynamically with the environment. Existing work primarily focuses on behavior cloning from expert demonstrations or preference learning through exploratory trajectory sampling. However, these methods often struggle to address long-horizon tasks, where suboptimal actions accumulate step by step, causing agents to deviate from correct task trajectories. To address this, we highlight the importance of timely calibration and the need to automatically construct calibration trajectories for training agents. We propose Step-Level Trajectory Calibration (STeCa), a novel framework for LLM agent learning. Specifically, STeCa identifies suboptimal actions through a step-level reward comparison during exploration. It constructs calibrated trajectories using LLM-driven reflection, enabling agents to learn from improved decision-making processes. We finally leverage these calibrated trajectories with successful trajectories for reinforced training. Extensive experiments demonstrate that STeCa significantly outperforms existing methods. Further analysis highlights that timely calibration enables agents to complete tasks with greater robustness. Our code and data are available at https://github.com/WangHanLinHenry/STeCa.
>
---
#### [replaced 079] HiDe-LLaVA: Hierarchical Decoupling for Continual Instruction Tuning of Multimodal Large Language Model
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.12941v2](http://arxiv.org/pdf/2503.12941v2)**

> **作者:** Haiyang Guo; Fanhu Zeng; Ziwei Xiang; Fei Zhu; Da-Han Wang; Xu-Yao Zhang; Cheng-Lin Liu
>
> **备注:** ACL 2025 (Main)
>
> **摘要:** Instruction tuning is widely used to improve a pre-trained Multimodal Large Language Model (MLLM) by training it on curated task-specific datasets, enabling better comprehension of human instructions. However, it is infeasible to collect all possible instruction datasets simultaneously in real-world scenarios. Thus, enabling MLLM with continual instruction tuning is essential for maintaining their adaptability. However, existing methods often trade off memory efficiency for performance gains, significantly compromising overall efficiency. In this paper, we propose a task-specific expansion and task-general fusion framework based on the variations in Centered Kernel Alignment (CKA) similarity across different model layers when trained on diverse datasets. Furthermore, we analyze the information leakage present in the existing benchmark and propose a new and more challenging benchmark to rationally evaluate the performance of different methods. Comprehensive experiments showcase a significant performance improvement of our method compared to existing state-of-the-art methods. Code and dataset are released at https://github.com/Ghy0501/HiDe-LLaVA.
>
---
#### [replaced 080] C$^2$LEVA: Toward Comprehensive and Contamination-Free Language Model Evaluation
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.04947v3](http://arxiv.org/pdf/2412.04947v3)**

> **作者:** Yanyang Li; Tin Long Wong; Cheung To Hung; Jianqiao Zhao; Duo Zheng; Ka Wai Liu; Michael R. Lyu; Liwei Wang
>
> **备注:** Findings of ACL 2025; Project Page: https://github.com/LaVi-Lab/C2LEVA
>
> **摘要:** Recent advances in large language models (LLMs) have shown significant promise, yet their evaluation raises concerns, particularly regarding data contamination due to the lack of access to proprietary training data. To address this issue, we present C$^2$LEVA, a comprehensive bilingual benchmark featuring systematic contamination prevention. C$^2$LEVA firstly offers a holistic evaluation encompassing 22 tasks, each targeting a specific application or ability of LLMs, and secondly a trustworthy assessment due to our contamination-free tasks, ensured by a systematic contamination prevention strategy that fully automates test data renewal and enforces data protection during benchmark data release. Our large-scale evaluation of 15 open-source and proprietary models demonstrates the effectiveness of C$^2$LEVA.
>
---
#### [replaced 081] Cross-modal RAG: Sub-dimensional Retrieval-Augmented Text-to-Image Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.21956v2](http://arxiv.org/pdf/2505.21956v2)**

> **作者:** Mengdan Zhu; Senhao Cheng; Guangji Bai; Yifei Zhang; Liang Zhao
>
> **摘要:** Text-to-image generation increasingly demands access to domain-specific, fine-grained, and rapidly evolving knowledge that pretrained models cannot fully capture. Existing Retrieval-Augmented Generation (RAG) methods attempt to address this by retrieving globally relevant images, but they fail when no single image contains all desired elements from a complex user query. We propose Cross-modal RAG, a novel framework that decomposes both queries and images into sub-dimensional components, enabling subquery-aware retrieval and generation. Our method introduces a hybrid retrieval strategy - combining a sub-dimensional sparse retriever with a dense retriever - to identify a Pareto-optimal set of images, each contributing complementary aspects of the query. During generation, a multimodal large language model is guided to selectively condition on relevant visual features aligned to specific subqueries, ensuring subquery-aware image synthesis. Extensive experiments on MS-COCO, Flickr30K, WikiArt, CUB, and ImageNet-LT demonstrate that Cross-modal RAG significantly outperforms existing baselines in both retrieval and generation quality, while maintaining high efficiency.
>
---
#### [replaced 082] FlexDuo: A Pluggable System for Enabling Full-Duplex Capabilities in Speech Dialogue Systems
- **分类: cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2502.13472v2](http://arxiv.org/pdf/2502.13472v2)**

> **作者:** Borui Liao; Yulong Xu; Jiao Ou; Kaiyuan Yang; Weihua Jian; Pengfei Wan; Di Zhang
>
> **摘要:** Full-Duplex Speech Dialogue Systems (Full-Duplex SDS) have significantly enhanced the naturalness of human-machine interaction by enabling real-time bidirectional communication. However, existing approaches face challenges such as difficulties in independent module optimization and contextual noise interference due to highly coupled architectural designs and oversimplified binary state modeling. This paper proposes FlexDuo, a flexible full-duplex control module that decouples duplex control from spoken dialogue systems through a plug-and-play architectural design. Furthermore, inspired by human information-filtering mechanisms in conversations, we introduce an explicit Idle state. On one hand, the Idle state filters redundant noise and irrelevant audio to enhance dialogue quality. On the other hand, it establishes a semantic integrity-based buffering mechanism, reducing the risk of mutual interruptions while ensuring accurate response transitions. Experimental results on the Fisher corpus demonstrate that FlexDuo reduces the false interruption rate by 24.9% and improves response accuracy by 7.6% compared to integrated full-duplex dialogue system baselines. It also outperforms voice activity detection (VAD) controlled baseline systems in both Chinese and English dialogue quality. The proposed modular architecture and state-based dialogue model provide a novel technical pathway for building flexible and efficient duplex dialogue systems.
>
---
#### [replaced 083] GETReason: Enhancing Image Context Extraction through Hierarchical Multi-Agent Reasoning
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21863v2](http://arxiv.org/pdf/2505.21863v2)**

> **作者:** Shikhhar Siingh; Abhinav Rawat; Chitta Baral; Vivek Gupta
>
> **摘要:** Publicly significant images from events hold valuable contextual information, crucial for journalism and education. However, existing methods often struggle to extract this relevance accurately. To address this, we introduce GETReason (Geospatial Event Temporal Reasoning), a framework that moves beyond surface-level image descriptions to infer deeper contextual meaning. We propose that extracting global event, temporal, and geospatial information enhances understanding of an image's significance. Additionally, we introduce GREAT (Geospatial Reasoning and Event Accuracy with Temporal Alignment), a new metric for evaluating reasoning-based image understanding. Our layered multi-agent approach, assessed using a reasoning-weighted metric, demonstrates that meaningful insights can be inferred, effectively linking images to their broader event context.
>
---
#### [replaced 084] NeedleInATable: Exploring Long-Context Capability of Large Language Models towards Long-Structured Tables
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.06560v2](http://arxiv.org/pdf/2504.06560v2)**

> **作者:** Lanrui Wang; Mingyu Zheng; Hongyin Tang; Zheng Lin; Yanan Cao; Jingang Wang; Xunliang Cai; Weiping Wang
>
> **备注:** Work in Progress
>
> **摘要:** Processing structured tabular data, particularly large and lengthy tables, constitutes a fundamental yet challenging task for large language models (LLMs). However, existing long-context benchmarks like Needle-in-a-Haystack primarily focus on unstructured text, neglecting the challenge of diverse structured tables. Meanwhile, previous tabular benchmarks mainly consider downstream tasks that require high-level reasoning abilities, and overlook models' underlying fine-grained perception of individual table cells, which is crucial for practical and robust LLM-based table applications. To address this gap, we introduce \textsc{NeedleInATable} (NIAT), a new long-context tabular benchmark that treats each table cell as a ``needle'' and requires models to extract the target cell based on cell locations or lookup questions. Our comprehensive evaluation of various LLMs and multimodal LLMs reveals a substantial performance gap between popular downstream tabular tasks and the simpler NIAT task, suggesting that they may rely on dataset-specific correlations or shortcuts to obtain better benchmark results but lack truly robust long-context understanding towards structured tables. Furthermore, we demonstrate that using synthesized NIAT training data can effectively improve performance on both NIAT task and downstream tabular tasks, which validates the importance of NIAT capability for LLMs' genuine table understanding ability. Our data, code and models will be released to facilitate future research.
>
---
#### [replaced 085] Neuro-symbolic Training for Reasoning over Spatial Language
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.13828v3](http://arxiv.org/pdf/2406.13828v3)**

> **作者:** Tanawan Premsri; Parisa Kordjamshidi
>
> **备注:** 9 pages, 4 figures, NAACL 2025 findings
>
> **摘要:** Spatial reasoning based on natural language expressions is essential for everyday human tasks. This reasoning ability is also crucial for machines to interact with their environment in a human-like manner. However, recent research shows that even state-of-the-art language models struggle with spatial reasoning over text, especially when facing nesting spatial expressions. This is attributed to not achieving the right level of abstraction required for generalizability. To alleviate this issue, we propose training language models with neuro-symbolic techniques that exploit the spatial logical rules as constraints, providing additional supervision to improve spatial reasoning and question answering. Training language models to adhere to spatial reasoning rules guides them in making more effective and general abstractions for transferring spatial knowledge to various domains. We evaluate our approach on existing spatial question-answering benchmarks. Our results indicate the effectiveness of our proposed technique in improving language models in complex multi-hop spatial reasoning over text.
>
---
#### [replaced 086] ASTPrompter: Preference-Aligned Automated Language Model Red-Teaming to Generate Low-Perplexity Unsafe Prompts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.09447v4](http://arxiv.org/pdf/2407.09447v4)**

> **作者:** Amelia F. Hardy; Houjun Liu; Bernard Lange; Duncan Eddy; Mykel J. Kochenderfer
>
> **备注:** 8 pages, 7 pages of appendix, 3 tables, 4 figures
>
> **摘要:** Existing LLM red-teaming approaches prioritize high attack success rate, often resulting in high-perplexity prompts. This focus overlooks low-perplexity attacks that are more difficult to filter, more likely to arise during benign usage, and more impactful as negative downstream training examples. In response, we introduce ASTPrompter, a single-step optimization method that uses contrastive preference learning to train an attacker to maintain low perplexity while achieving a high attack success rate (ASR). ASTPrompter achieves an attack success rate 5.1 times higher on Llama-8.1B while using inputs that are 2.1 times more likely to occur according to the frozen LLM. Furthermore, our attack transfers to Mistral-7B, Qwen-7B, and TinyLlama in both black- and white-box settings. Lastly, by tuning a single hyperparameter in our method, we discover successful attack prefixes along an efficient frontier between ASR and perplexity, highlighting perplexity as a previously under-considered factor in red-teaming.
>
---
#### [replaced 087] BenchmarkCards: Large Language Model and Risk Reporting
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.12974v2](http://arxiv.org/pdf/2410.12974v2)**

> **作者:** Anna Sokol; Elizabeth Daly; Michael Hind; David Piorkowski; Xiangliang Zhang; Nuno Moniz; Nitesh Chawla
>
> **摘要:** Large language models (LLMs) are powerful tools capable of handling diverse tasks. Comparing and selecting appropriate LLMs for specific tasks requires systematic evaluation methods, as models exhibit varying capabilities across different domains. However, finding suitable benchmarks is difficult given the many available options. This complexity not only increases the risk of benchmark misuse and misinterpretation but also demands substantial effort from LLM users, seeking the most suitable benchmarks for their specific needs. To address these issues, we introduce \texttt{BenchmarkCards}, an intuitive and validated documentation framework that standardizes critical benchmark attributes such as objectives, methodologies, data sources, and limitations. Through user studies involving benchmark creators and users, we show that \texttt{BenchmarkCards} can simplify benchmark selection and enhance transparency, facilitating informed decision-making in evaluating LLMs. Data & Code: https://github.com/SokolAnn/BenchmarkCards
>
---
#### [replaced 088] CodePMP: Scalable Preference Model Pretraining for Large Language Model Reasoning
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.02229v2](http://arxiv.org/pdf/2410.02229v2)**

> **作者:** Huimu Yu; Xing Wu; Haotian Xu; Debing Zhang; Songlin Hu
>
> **备注:** work in progress
>
> **摘要:** Large language models (LLMs) have made significant progress in natural language understanding and generation, driven by scalable pretraining and advanced finetuning. However, enhancing reasoning abilities in LLMs, particularly via reinforcement learning from human feedback (RLHF), remains challenging due to the scarcity of high-quality preference data, which is labor-intensive to annotate and crucial for reward model (RM) finetuning. To alleviate this issue, we introduce CodePMP, a scalable preference model pretraining (PMP) pipeline that utilizes a large corpus of synthesized code-preference pairs from publicly available high-quality source code. CodePMP improves RM finetuning efficiency by pretraining preference models on large-scale synthesized code-preference pairs. We evaluate CodePMP on mathematical reasoning tasks (GSM8K, MATH) and logical reasoning tasks (ReClor, LogiQA2.0), consistently showing significant improvements in reasoning performance of LLMs and highlighting the importance of scalable preference model pretraining for efficient reward modeling.
>
---
#### [replaced 089] Learning to Poison Large Language Models for Downstream Manipulation
- **分类: cs.LG; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2402.13459v3](http://arxiv.org/pdf/2402.13459v3)**

> **作者:** Xiangyu Zhou; Yao Qiang; Saleh Zare Zade; Mohammad Amin Roshani; Prashant Khanduri; Douglas Zytko; Dongxiao Zhu
>
> **摘要:** The advent of Large Language Models (LLMs) has marked significant achievements in language processing and reasoning capabilities. Despite their advancements, LLMs face vulnerabilities to data poisoning attacks, where the adversary inserts backdoor triggers into training data to manipulate outputs. This work further identifies additional security risks in LLMs by designing a new data poisoning attack tailored to exploit the supervised fine-tuning (SFT) process. We propose a novel gradient-guided backdoor trigger learning (GBTL) algorithm to identify adversarial triggers efficiently, ensuring an evasion of detection by conventional defenses while maintaining content integrity. Through experimental validation across various language model tasks, including sentiment analysis, domain generation, and question answering, our poisoning strategy demonstrates a high success rate in compromising various LLMs' outputs. We further propose two defense strategies against data poisoning attacks, including in-context learning (ICL) and continuous learning (CL), which effectively rectify the behavior of LLMs and significantly reduce the decline in performance. Our work highlights the significant security risks present during SFT of LLMs and the necessity of safeguarding LLMs against data poisoning attacks.
>
---
#### [replaced 090] Temporal Relation Extraction in Clinical Texts: A Span-based Graph Transformer Approach
- **分类: cs.CL; cs.AI; cs.LG; I.2.7; I.2.1; J.3**

- **链接: [http://arxiv.org/pdf/2503.18085v2](http://arxiv.org/pdf/2503.18085v2)**

> **作者:** Rochana Chaturvedi; Peyman Baghershahi; Sourav Medya; Barbara Di Eugenio
>
> **备注:** Introducing a novel method for joint extraction of medical events and temporal relations from free-text, leveraging clinical LPLMs and Heterogeneous Graph Transformers, achieving a 5.5% improvement over the previous state-of-the-art and up to 8.9% on long-range relations
>
> **摘要:** Temporal information extraction from unstructured text is essential for contextualizing events and deriving actionable insights, particularly in the medical domain. We address the task of extracting clinical events and their temporal relations using the well-studied I2B2 2012 Temporal Relations Challenge corpus. This task is inherently challenging due to complex clinical language, long documents, and sparse annotations. We introduce GRAPHTREX, a novel method integrating span-based entity-relation extraction, clinical large pre-trained language models (LPLMs), and Heterogeneous Graph Transformers (HGT) to capture local and global dependencies. Our HGT component facilitates information propagation across the document through innovative global landmarks that bridge distant entities. Our method improves the state-of-the-art with 5.5% improvement in the tempeval $F_1$ score over the previous best and up to 8.9% improvement on long-range relations, which presents a formidable challenge. We further demonstrate generalizability by establishing a strong baseline on the E3C corpus. This work not only advances temporal information extraction but also lays the groundwork for improved diagnostic and prognostic models through enhanced temporal reasoning.
>
---
#### [replaced 091] Skywork Open Reasoner 1 Technical Report
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22312v2](http://arxiv.org/pdf/2505.22312v2)**

> **作者:** Jujie He; Jiacai Liu; Chris Yuhao Liu; Rui Yan; Chaojie Wang; Peng Cheng; Xiaoyu Zhang; Fuxiang Zhang; Jiacheng Xu; Wei Shen; Siyuan Li; Liang Zeng; Tianwen Wei; Cheng Cheng; Bo An; Yang Liu; Yahui Zhou
>
> **摘要:** The success of DeepSeek-R1 underscores the significant role of reinforcement learning (RL) in enhancing the reasoning capabilities of large language models (LLMs). In this work, we present Skywork-OR1, an effective and scalable RL implementation for long Chain-of-Thought (CoT) models. Building on the DeepSeek-R1-Distill model series, our RL approach achieves notable performance gains, increasing average accuracy across AIME24, AIME25, and LiveCodeBench from 57.8% to 72.8% (+15.0%) for the 32B model and from 43.6% to 57.5% (+13.9%) for the 7B model. Our Skywork-OR1-32B model surpasses both DeepSeek-R1 and Qwen3-32B on the AIME24 and AIME25 benchmarks, while achieving comparable results on LiveCodeBench. The Skywork-OR1-7B and Skywork-OR1-Math-7B models demonstrate competitive reasoning capabilities among models of similar size. We perform comprehensive ablation studies on the core components of our training pipeline to validate their effectiveness. Additionally, we thoroughly investigate the phenomenon of entropy collapse, identify key factors affecting entropy dynamics, and demonstrate that mitigating premature entropy collapse is critical for improved test performance. To support community research, we fully open-source our model weights, training code, and training datasets.
>
---
#### [replaced 092] Understanding Bias Reinforcement in LLM Agents Debate
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.16814v2](http://arxiv.org/pdf/2503.16814v2)**

> **作者:** Jihwan Oh; Minchan Jeong; Jongwoo Ko; Se-Young Yun
>
> **备注:** ICML 2025
>
> **摘要:** Large Language Models $($LLMs$)$ solve complex problems using training-free methods like prompt engineering and in-context learning, yet ensuring reasoning correctness remains challenging. While self-correction methods such as self-consistency and self-refinement aim to improve reliability, they often reinforce biases due to the lack of effective feedback mechanisms. Multi-Agent Debate $($MAD$)$ has emerged as an alternative, but we identify two key limitations: bias reinforcement, where debate amplifies model biases instead of correcting them, and lack of perspective diversity, as all agents share the same model and reasoning patterns, limiting true debate effectiveness. To systematically evaluate these issues, we introduce $\textit{MetaNIM Arena}$, a benchmark designed to assess LLMs in adversarial strategic decision-making, where dynamic interactions influence optimal decisions. To overcome MAD's limitations, we propose $\textbf{DReaMAD}$ $($$\textbf{D}$iverse $\textbf{Rea}$soning via $\textbf{M}$ulti-$\textbf{A}$gent $\textbf{D}$ebate with Refined Prompt$)$, a novel framework that $(1)$ refines LLM's strategic prior knowledge to improve reasoning quality and $(2)$ promotes diverse viewpoints within a single model by systematically modifying prompts, reducing bias. Empirical results show that $\textbf{DReaMAD}$ significantly improves decision accuracy, reasoning diversity, and bias mitigation across multiple strategic tasks, establishing it as a more effective approach for LLM-based decision-making.
>
---
#### [replaced 093] DeepSeek vs. o3-mini: How Well can Reasoning LLMs Evaluate MT and Summarization?
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.08120v2](http://arxiv.org/pdf/2504.08120v2)**

> **作者:** Daniil Larionov; Sotaro Takeshita; Ran Zhang; Yanran Chen; Christoph Leiter; Zhipin Wang; Christian Greisinger; Steffen Eger
>
> **摘要:** Reasoning-enabled large language models (LLMs) excel in logical tasks, yet their utility for evaluating natural language generation remains unexplored. This study systematically compares reasoning LLMs with non-reasoning counterparts across machine translation and text summarization evaluation tasks. We evaluate eight models spanning state-of-the-art reasoning models (DeepSeek-R1, OpenAI o3), their distilled variants (8B-70B parameters), and equivalent non-reasoning LLMs. Experiments on WMT23 and SummEval benchmarks reveal architecture and task-dependent benefits: OpenAI o3-mini models show improved performance with increased reasoning on MT, while DeepSeek-R1 and generally underperforms compared to its non-reasoning variant except in summarization consistency evaluation. Correlation analysis demonstrates that reasoning token usage correlates with evaluation quality only in specific models, while almost all models generally allocate more reasoning tokens when identifying more quality issues. Distillation maintains reasonable performance up to 32B parameter models but degrades substantially at 8B scale. This work provides the first assessment of reasoning LLMs for NLG evaluation and comparison to non-reasoning models. We share our code to facilitate further research: https://github.com/NL2G/reasoning-eval.
>
---
#### [replaced 094] GSQ-Tuning: Group-Shared Exponents Integer in Fully Quantized Training for LLMs On-Device Fine-tuning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12913v3](http://arxiv.org/pdf/2502.12913v3)**

> **作者:** Sifan Zhou; Shuo Wang; Zhihang Yuan; Mingjia Shi; Yuzhang Shang; Dawei Yang
>
> **备注:** Accepted by Findings of ACL 2025
>
> **摘要:** Large Language Models (LLMs) fine-tuning technologies have achieved remarkable results. However, traditional LLM fine-tuning approaches face significant challenges: they require large Floating Point (FP) computation, raising privacy concerns when handling sensitive data, and are impractical for resource-constrained edge devices. While Parameter-Efficient Fine-Tuning (PEFT) techniques reduce trainable parameters, their reliance on floating-point arithmetic creates fundamental incompatibilities with edge hardware. In this work, we introduce a novel framework for on-device LLM fine-tuning that eliminates the need for floating-point operations in both inference and training, named GSQ-Tuning. At its core is the Group-Shared Exponents Integer format, which efficiently represents model parameters in integer format using shared exponents among parameter groups. When combined with LoRA-like adapters, this enables fully integer-based fine-tuning that is both memory and compute efficient. We demonstrate that our approach achieves accuracy comparable to BF16-based fine-tuning while significantly reducing 1.85x memory usage. Moreover, compared to FP8, our method can reduce 5x power consumption and 11x chip area with same performance, making large-scale model adaptation feasible on edge devices.
>
---
#### [replaced 095] Uncertainty Quantification for LLMs through Minimum Bayes Risk: Bridging Confidence and Consistency
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.04964v4](http://arxiv.org/pdf/2502.04964v4)**

> **作者:** Roman Vashurin; Maiya Goloburda; Albina Ilina; Aleksandr Rubashevskii; Preslav Nakov; Artem Shelmanov; Maxim Panov
>
> **摘要:** Uncertainty quantification (UQ) methods for Large Language Models (LLMs) encompass a variety of approaches, with two major types being particularly prominent: information-based, which focus on model confidence expressed as token probabilities, and consistency-based, which assess the semantic relationship between multiple outputs generated using repeated sampling. Several recent methods have combined these two approaches to boost UQ performance. However, they sometimes fail to outperform much simpler baseline methods. Our work discusses the fundamental approach to constructing uncertainty measures that directly links uncertainty with the minimum Bayes risks achieved by LLM decoding. Building on these findings, we propose a novel approach to integrating model confidence with output consistency, resulting in a family of efficient and robust UQ methods. Our investigation reveals distinctive characteristics of LLMs as probabilistic models, which help to explain why these UQ methods underperform in certain tasks. Based on these findings, we propose a new way of synthesizing model confidence and output consistency, leading to a family of efficient and robust UQ methods. We evaluate our approach across various tasks such as question answering, abstractive summarization, and machine translation, demonstrating sizable improvements over state-of-the-art UQ approaches.
>
---
#### [replaced 096] Do we still need Human Annotators? Prompting Large Language Models for Aspect Sentiment Quad Prediction
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.13044v3](http://arxiv.org/pdf/2502.13044v3)**

> **作者:** Nils Constantin Hellwig; Jakob Fehle; Udo Kruschwitz; Christian Wolff
>
> **摘要:** Aspect sentiment quad prediction (ASQP) facilitates a detailed understanding of opinions expressed in a text by identifying the opinion term, aspect term, aspect category and sentiment polarity for each opinion. However, annotating a full set of training examples to fine-tune models for ASQP is a resource-intensive process. In this study, we explore the capabilities of large language models (LLMs) for zero- and few-shot learning on the ASQP task across five diverse datasets. We report F1 scores almost up to par with those obtained with state-of-the-art fine-tuned models and exceeding previously reported zero- and few-shot performance. In the 20-shot setting on the Rest16 restaurant domain dataset, LLMs achieved an F1 score of 51.54, compared to 60.39 by the best-performing fine-tuned method MVP. Additionally, we report the performance of LLMs in target aspect sentiment detection (TASD), where the F1 scores were close to fine-tuned models, achieving 68.93 on Rest16 in the 30-shot setting, compared to 72.76 with MVP. While human annotators remain essential for achieving optimal performance, LLMs can reduce the need for extensive manual annotation in ASQP tasks.
>
---
#### [replaced 097] Structure-Enhanced Protein Instruction Tuning: Towards General-Purpose Protein Understanding with LLMs
- **分类: cs.CL; q-bio.BM**

- **链接: [http://arxiv.org/pdf/2410.03553v3](http://arxiv.org/pdf/2410.03553v3)**

> **作者:** Wei Wu; Chao Wang; Liyi Chen; Mingze Yin; Yiheng Zhu; Kun Fu; Jieping Ye; Hui Xiong; Zheng Wang
>
> **备注:** Accepted by KDD2025
>
> **摘要:** Proteins, as essential biomolecules, play a central role in biological processes, including metabolic reactions and DNA replication. Accurate prediction of their properties and functions is crucial in biological applications. Recent development of protein language models (pLMs) with supervised fine tuning provides a promising solution to this problem. However, the fine-tuned model is tailored for particular downstream prediction task, and achieving general-purpose protein understanding remains a challenge. In this paper, we introduce Structure-Enhanced Protein Instruction Tuning (SEPIT) framework to bridge this gap. Our approach incorporates a novel structure-aware module into pLMs to enrich their structural knowledge, and subsequently integrates these enhanced pLMs with large language models (LLMs) to advance protein understanding. In this framework, we propose a novel instruction tuning pipeline. First, we warm up the enhanced pLMs using contrastive learning and structure denoising. Then, caption-based instructions are used to establish a basic understanding of proteins. Finally, we refine this understanding by employing a mixture of experts (MoEs) to capture more complex properties and functional information with the same number of activated parameters. Moreover, we construct the largest and most comprehensive protein instruction dataset to date, which allows us to train and evaluate the general-purpose protein understanding model. Extensive experiments on both open-ended generation and closed-set answer tasks demonstrate the superior performance of SEPIT over both closed-source general LLMs and open-source LLMs trained with protein knowledge.
>
---
#### [replaced 098] Length-Controlled Margin-Based Preference Optimization without Reference Model
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.14643v2](http://arxiv.org/pdf/2502.14643v2)**

> **作者:** Gengxu Li; Tingyu Xia; Yi Chang; Yuan Wu
>
> **备注:** 18 pages, 3 figures, 6 tables
>
> **摘要:** Direct Preference Optimization (DPO) is a widely adopted offline algorithm for preference-based reinforcement learning from human feedback (RLHF), designed to improve training simplicity and stability by redefining reward functions. However, DPO is hindered by several limitations, including length bias, memory inefficiency, and probability degradation. To address these challenges, we propose Length-Controlled Margin-Based Preference Optimization (LMPO), a more efficient and robust alternative. LMPO introduces a uniform reference model as an upper bound for the DPO loss, enabling a more accurate approximation of the original optimization objective. Additionally, an average log-probability optimization strategy is employed to minimize discrepancies between training and inference phases. A key innovation of LMPO lies in its Length-Controlled Margin-Based loss function, integrated within the Bradley-Terry framework. This loss function regulates response length while simultaneously widening the margin between preferred and rejected outputs. By doing so, it mitigates probability degradation for both accepted and discarded responses, addressing a significant limitation of existing methods. We evaluate LMPO against state-of-the-art preference optimization techniques on two open-ended large language models, Mistral and LLaMA3, across six conditional benchmarks. Our experimental results demonstrate that LMPO effectively controls response length, reduces probability degradation, and outperforms existing approaches. The code is available at https://github.com/gengxuli/LMPO.
>
---
#### [replaced 099] Reasoning-to-Defend: Safety-Aware Reasoning Can Defend Large Language Models from Jailbreaking
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12970v2](http://arxiv.org/pdf/2502.12970v2)**

> **作者:** Junda Zhu; Lingyong Yan; Shuaiqiang Wang; Dawei Yin; Lei Sha
>
> **备注:** 18 pages
>
> **摘要:** Large Reasoning Models (LRMs) have demonstrated impressive performances across diverse domains. However, how safety of Large Language Models (LLMs) benefits from enhanced reasoning capabilities against jailbreak queries remains unexplored. To bridge this gap, in this paper, we propose Reasoning-to-Defend (R2D), a novel training paradigm that integrates a safety-aware reasoning mechanism into LLMs' generation. This enables self-evaluation at each step of the reasoning process, forming safety pivot tokens as indicators of the safety status of responses. Furthermore, in order to improve the accuracy of predicting pivot tokens, we propose Contrastive Pivot Optimization (CPO), which enhances the model's perception of the safety status of given dialogues. LLMs dynamically adjust their response strategies during reasoning, significantly enhancing their safety capabilities defending jailbreak attacks. Extensive experiments demonstrate that R2D effectively mitigates various attacks and improves overall safety, while maintaining the original performances. This highlights the substantial potential of safety-aware reasoning in improving robustness of LRMs and LLMs against various jailbreaks.
>
---
#### [replaced 100] Tensor Product Attention Is All You Need
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.06425v4](http://arxiv.org/pdf/2501.06425v4)**

> **作者:** Yifan Zhang; Yifeng Liu; Huizhuo Yuan; Zhen Qin; Yang Yuan; Quanquan Gu; Andrew C Yao
>
> **备注:** 52 pages, 11 figures
>
> **摘要:** Scaling language models to handle longer input sequences typically necessitates large key-value (KV) caches, resulting in substantial memory overhead during inference. In this paper, we propose Tensor Product Attention (TPA), a novel attention mechanism that uses tensor decompositions to represent queries, keys, and values compactly, substantially shrinking the KV cache size at inference time. By factorizing these representations into contextual low-rank components and seamlessly integrating with Rotary Position Embedding (RoPE), TPA achieves improved model quality alongside memory efficiency. Based on TPA, we introduce the Tensor Product Attention Transformer,(T6), a new model architecture for sequence modeling. Through extensive empirical evaluation on language modeling tasks, we demonstrate that T6 surpasses or matches the performance of standard Transformer baselines, including Multi-Head Attention (MHA), Multi-Query Attention (MQA), Grouped-Query Attention (GQA), and Multi-Head Latent Attention (MLA) across various metrics, including perplexity and a range of established evaluation benchmarks. Notably, TPA's memory efficiency and computational efficiency at the decoding stage enable processing longer sequences under fixed resource constraints, addressing a critical scalability challenge in modern language models. The code is available at https://github.com/tensorgi/T6.
>
---
#### [replaced 101] Improving Continual Pre-training Through Seamless Data Packing
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22018v2](http://arxiv.org/pdf/2505.22018v2)**

> **作者:** Ruicheng Yin; Xuan Gao; Changze Lv; Xiaohua Wang; Xiaoqing Zheng; Xuanjing Huang
>
> **备注:** Accepted to ACL 2025 Findings
>
> **摘要:** Continual pre-training has demonstrated significant potential in enhancing model performance, particularly in domain-specific scenarios. The most common approach for packing data before continual pre-training involves concatenating input texts and splitting them into fixed-length sequences. While straightforward and efficient, this method often leads to excessive truncation and context discontinuity, which can hinder model performance. To address these issues, we explore the potential of data engineering to enhance continual pre-training, particularly its impact on model performance and efficiency. We propose Seamless Packing (SP), a novel data packing strategy aimed at preserving contextual information more effectively and enhancing model performance. Our approach employs a sliding window technique in the first stage that synchronizes overlapping tokens across consecutive sequences, ensuring better continuity and contextual coherence. In the second stage, we adopt a First-Fit-Decreasing algorithm to pack shorter texts into bins slightly larger than the target sequence length, thereby minimizing padding and truncation. Empirical evaluations across various model architectures and corpus domains demonstrate the effectiveness of our method, outperforming baseline method in 99% of all settings. Code is available at https://github.com/Infernus-WIND/Seamless-Packing.
>
---
#### [replaced 102] EXIT: Context-Aware Extractive Compression for Enhancing Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2412.12559v3](http://arxiv.org/pdf/2412.12559v3)**

> **作者:** Taeho Hwang; Sukmin Cho; Soyeong Jeong; Hoyun Song; SeungYoon Han; Jong C. Park
>
> **备注:** Findings of ACL 2025
>
> **摘要:** We introduce EXIT, an extractive context compression framework that enhances both the effectiveness and efficiency of retrieval-augmented generation (RAG) in question answering (QA). Current RAG systems often struggle when retrieval models fail to rank the most relevant documents, leading to the inclusion of more context at the expense of latency and accuracy. While abstractive compression methods can drastically reduce token counts, their token-by-token generation process significantly increases end-to-end latency. Conversely, existing extractive methods reduce latency but rely on independent, non-adaptive sentence selection, failing to fully utilize contextual information. EXIT addresses these limitations by classifying sentences from retrieved documents - while preserving their contextual dependencies - enabling parallelizable, context-aware extraction that adapts to query complexity and retrieval quality. Our evaluations on both single-hop and multi-hop QA tasks show that EXIT consistently surpasses existing compression methods and even uncompressed baselines in QA accuracy, while also delivering substantial reductions in inference time and token count. By improving both effectiveness and efficiency, EXIT provides a promising direction for developing scalable, high-quality QA solutions in RAG pipelines. Our code is available at https://github.com/ThisIsHwang/EXIT
>
---
#### [replaced 103] Enhancing Retrieval for ESGLLM via ESG-CID -- A Disclosure Content Index Finetuning Dataset for Mapping GRI and ESRS
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.10674v2](http://arxiv.org/pdf/2503.10674v2)**

> **作者:** Shafiuddin Rehan Ahmed; Ankit Parag Shah; Quan Hung Tran; Vivek Khetan; Sukryool Kang; Ankit Mehta; Yujia Bao; Wei Wei
>
> **备注:** Long paper
>
> **摘要:** Climate change has intensified the need for transparency and accountability in organizational practices, making Environmental, Social, and Governance (ESG) reporting increasingly crucial. Frameworks like the Global Reporting Initiative (GRI) and the new European Sustainability Reporting Standards (ESRS) aim to standardize ESG reporting, yet generating comprehensive reports remains challenging due to the considerable length of ESG documents and variability in company reporting styles. To facilitate ESG report automation, Retrieval-Augmented Generation (RAG) systems can be employed, but their development is hindered by a lack of labeled data suitable for training retrieval models. In this paper, we leverage an underutilized source of weak supervision -- the disclosure content index found in past ESG reports -- to create a comprehensive dataset, ESG-CID, for both GRI and ESRS standards. By extracting mappings between specific disclosure requirements and corresponding report sections, and refining them using a Large Language Model as a judge, we generate a robust training and evaluation set. We benchmark popular embedding models on this dataset and show that fine-tuning BERT-based models can outperform commercial embeddings and leading public models, even under temporal data splits for cross-report style transfer from GRI to ESRS. Data: https://huggingface.co/datasets/airefinery/esg_cid_retrieval
>
---
#### [replaced 104] Graph of Records: Boosting Retrieval Augmented Generation for Long-context Summarization with Graphs
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.11001v2](http://arxiv.org/pdf/2410.11001v2)**

> **作者:** Haozhen Zhang; Tao Feng; Jiaxuan You
>
> **备注:** Accepted by ACL 2025 Main. The code is available at https://github.com/ulab-uiuc/GoR
>
> **摘要:** Retrieval-augmented generation (RAG) has revitalized Large Language Models (LLMs) by injecting non-parametric factual knowledge. Compared with long-context LLMs, RAG is considered an effective summarization tool in a more concise and lightweight manner, which can interact with LLMs multiple times using diverse queries to get comprehensive responses. However, the LLM-generated historical responses, which contain potentially insightful information, are largely neglected and discarded by existing approaches, leading to suboptimal results. In this paper, we propose $\textit{graph of records}$ ($\textbf{GoR}$), which leverages historical responses generated by LLMs to enhance RAG for long-context global summarization. Inspired by the $\textit{retrieve-then-generate}$ paradigm of RAG, we construct a graph by establishing an edge between the retrieved text chunks and the corresponding LLM-generated response. To further uncover the intricate correlations between them, GoR features a $\textit{graph neural network}$ and an elaborately designed $\textit{BERTScore}$-based objective for self-supervised model training, enabling seamless supervision signal backpropagation between reference summaries and node embeddings. We comprehensively compare GoR with 12 baselines across four long-context summarization datasets, and the results indicate that our proposed method reaches the best performance ($\textit{e.g.}$, 15%, 8%, and 19% improvement over retrievers w.r.t. Rouge-L, Rouge-1, and Rouge-2 on the WCEP dataset). Extensive experiments further demonstrate the effectiveness of GoR.
>
---
#### [replaced 105] The Aloe Family Recipe for Open and Specialized Healthcare LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.04388v2](http://arxiv.org/pdf/2505.04388v2)**

> **作者:** Dario Garcia-Gasulla; Jordi Bayarri-Planas; Ashwin Kumar Gururajan; Enrique Lopez-Cuena; Adrian Tormos; Daniel Hinjos; Pablo Bernabeu-Perez; Anna Arias-Duart; Pablo Agustin Martin-Torres; Marta Gonzalez-Mallo; Sergio Alvarez-Napagao; Eduard Ayguadé-Parra; Ulises Cortés
>
> **备注:** Follow-up work from arXiv:2405.01886
>
> **摘要:** Purpose: With advancements in Large Language Models (LLMs) for healthcare, the need arises for competitive open-source models to protect the public interest. This work contributes to the field of open medical LLMs by optimizing key stages of data preprocessing and training, while showing how to improve model safety (through DPO) and efficacy (through RAG). The evaluation methodology used, which includes four different types of tests, defines a new standard for the field. The resultant models, shown to be competitive with the best private alternatives, are released with a permisive license. Methods: Building on top of strong base models like Llama 3.1 and Qwen 2.5, Aloe Beta uses a custom dataset to enhance public data with synthetic Chain of Thought examples. The models undergo alignment with Direct Preference Optimization, emphasizing ethical and policy-aligned performance in the presence of jailbreaking attacks. Evaluation includes close-ended, open-ended, safety and human assessments, to maximize the reliability of results. Results: Recommendations are made across the entire pipeline, backed by the solid performance of the Aloe Family. These models deliver competitive performance across healthcare benchmarks and medical fields, and are often preferred by healthcare professionals. On bias and toxicity, the Aloe Beta models significantly improve safety, showing resilience to unseen jailbreaking attacks. For a responsible release, a detailed risk assessment specific to healthcare is attached to the Aloe Family models. Conclusion: The Aloe Beta models, and the recipe that leads to them, are a significant contribution to the open-source medical LLM field, offering top-of-the-line performance while maintaining high ethical requirements. This work sets a new standard for developing and reporting aligned LLMs in healthcare.
>
---
#### [replaced 106] System-1.5 Reasoning: Traversal in Language and Latent Spaces with Dynamic Shortcuts
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.18962v2](http://arxiv.org/pdf/2505.18962v2)**

> **作者:** Xiaoqiang Wang; Suyuchen Wang; Yun Zhu; Bang Liu
>
> **备注:** Work in progress
>
> **摘要:** Chain-of-thought (CoT) reasoning enables large language models (LLMs) to move beyond fast System-1 responses and engage in deliberative System-2 reasoning. However, this comes at the cost of significant inefficiency due to verbose intermediate output. Recent latent-space reasoning methods improve efficiency by operating on hidden states without decoding into language, yet they treat all steps uniformly, failing to distinguish critical deductions from auxiliary steps and resulting in suboptimal use of computational resources. In this paper, we propose System-1.5 Reasoning, an adaptive reasoning framework that dynamically allocates computation across reasoning steps through shortcut paths in latent space. Specifically, System-1.5 Reasoning introduces two types of dynamic shortcuts. The model depth shortcut (DS) adaptively reasons along the vertical depth by early exiting non-critical tokens through lightweight adapter branches, while allowing critical tokens to continue through deeper Transformer layers. The step shortcut (SS) reuses hidden states across the decoding steps to skip trivial steps and reason horizontally in latent space. Training System-1.5 Reasoning involves a two-stage self-distillation process: first distilling natural language CoT into latent-space continuous thought, and then distilling full-path System-2 latent reasoning into adaptive shortcut paths (System-1.5 Reasoning). Experiments on reasoning tasks demonstrate the superior performance of our method. For example, on GSM8K, System-1.5 Reasoning achieves reasoning performance comparable to traditional CoT fine-tuning methods while accelerating inference by over 20x and reducing token generation by 92.31% on average.
>
---
#### [replaced 107] Understanding In-Context Machine Translation for Low-Resource Languages: A Case Study on Manchu
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.11862v2](http://arxiv.org/pdf/2502.11862v2)**

> **作者:** Renhao Pei; Yihong Liu; Peiqin Lin; François Yvon; Hinrich Schütze
>
> **备注:** ACL 2025
>
> **摘要:** In-context machine translation (MT) with large language models (LLMs) is a promising approach for low-resource MT, as it can readily take advantage of linguistic resources such as grammar books and dictionaries. Such resources are usually selectively integrated into the prompt so that LLMs can directly perform translation without any specific training, via their in-context learning capability (ICL). However, the relative importance of each type of resource, e.g., dictionary, grammar book, and retrieved parallel examples, is not entirely clear. To address this gap, this study systematically investigates how each resource and its quality affect the translation performance, with the Manchu language as our case study. To remove any prior knowledge of Manchu encoded in the LLM parameters and single out the effect of ICL, we also experiment with an enciphered version of Manchu texts. Our results indicate that high-quality dictionaries and good parallel examples are very helpful, while grammars hardly help. In a follow-up study, we showcase a promising application of in-context MT: parallel data augmentation as a way to bootstrap a conventional MT model. When monolingual data abound, generating synthetic parallel data through in-context MT offers a pathway to mitigate data scarcity and build effective and efficient low-resource neural MT systems.
>
---
#### [replaced 108] Agentic Knowledgeable Self-awareness
- **分类: cs.CL; cs.AI; cs.CV; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2504.03553v2](http://arxiv.org/pdf/2504.03553v2)**

> **作者:** Shuofei Qiao; Zhisong Qiu; Baochang Ren; Xiaobin Wang; Xiangyuan Ru; Ningyu Zhang; Xiang Chen; Yong Jiang; Pengjun Xie; Fei Huang; Huajun Chen
>
> **备注:** ACL 2025
>
> **摘要:** Large Language Models (LLMs) have achieved considerable performance across various agentic planning tasks. However, traditional agent planning approaches adopt a "flood irrigation" methodology that indiscriminately injects gold trajectories, external feedback, and domain knowledge into agent models. This practice overlooks the fundamental human cognitive principle of situational self-awareness during decision-making-the ability to dynamically assess situational demands and strategically employ resources during decision-making. We propose agentic knowledgeable self-awareness to address this gap, a novel paradigm enabling LLM-based agents to autonomously regulate knowledge utilization. Specifically, we propose KnowSelf, a data-centric approach that applies agents with knowledgeable self-awareness like humans. Concretely, we devise a heuristic situation judgement criterion to mark special tokens on the agent's self-explored trajectories for collecting training data. Through a two-stage training process, the agent model can switch between different situations by generating specific special tokens, achieving optimal planning effects with minimal costs. Our experiments demonstrate that KnowSelf can outperform various strong baselines on different tasks and models with minimal use of external knowledge. Code is available at https://github.com/zjunlp/KnowSelf.
>
---
#### [replaced 109] BioVL-QR: Egocentric Biochemical Vision-and-Language Dataset Using Micro QR Codes
- **分类: cs.CV; cs.CL; cs.MM**

- **链接: [http://arxiv.org/pdf/2404.03161v3](http://arxiv.org/pdf/2404.03161v3)**

> **作者:** Tomohiro Nishimoto; Taichi Nishimura; Koki Yamamoto; Keisuke Shirai; Hirotaka Kameko; Yuto Haneji; Tomoya Yoshida; Keiya Kajimura; Taiyu Cui; Chihiro Nishiwaki; Eriko Daikoku; Natsuko Okuda; Fumihito Ono; Shinsuke Mori
>
> **备注:** ICIP2025
>
> **摘要:** This paper introduces BioVL-QR, a biochemical vision-and-language dataset comprising 23 egocentric experiment videos, corresponding protocols, and vision-and-language alignments. A major challenge in understanding biochemical videos is detecting equipment, reagents, and containers because of the cluttered environment and indistinguishable objects. Previous studies assumed manual object annotation, which is costly and time-consuming. To address the issue, we focus on Micro QR Codes. However, detecting objects using only Micro QR Codes is still difficult due to blur and occlusion caused by object manipulation. To overcome this, we propose an object labeling method combining a Micro QR Code detector with an off-the-shelf hand object detector. As an application of the method and BioVL-QR, we tackled the task of localizing the procedural steps in an instructional video. The experimental results show that using Micro QR Codes and our method improves biochemical video understanding. Data and code are available through https://nishi10mo.github.io/BioVL-QR/
>
---
#### [replaced 110] Toward universal steering and monitoring of AI models
- **分类: cs.CL; cs.AI; stat.ML**

- **链接: [http://arxiv.org/pdf/2502.03708v2](http://arxiv.org/pdf/2502.03708v2)**

> **作者:** Daniel Beaglehole; Adityanarayanan Radhakrishnan; Enric Boix-Adserà; Mikhail Belkin
>
> **摘要:** Modern AI models contain much of human knowledge, yet understanding of their internal representation of this knowledge remains elusive. Characterizing the structure and properties of this representation will lead to improvements in model capabilities and development of effective safeguards. Building on recent advances in feature learning, we develop an effective, scalable approach for extracting linear representations of general concepts in large-scale AI models (language models, vision-language models, and reasoning models). We show how these representations enable model steering, through which we expose vulnerabilities, mitigate misaligned behaviors, and improve model capabilities. Additionally, we demonstrate that concept representations are remarkably transferable across human languages and combinable to enable multi-concept steering. Through quantitative analysis across hundreds of concepts, we find that newer, larger models are more steerable and steering can improve model capabilities beyond standard prompting. We show how concept representations are effective for monitoring misaligned content (hallucinations, toxic content). We demonstrate that predictive models built using concept representations are more accurate for monitoring misaligned content than using models that judge outputs directly. Together, our results illustrate the power of using internal representations to map the knowledge in AI models, advance AI safety, and improve model capabilities.
>
---
#### [replaced 111] EarthSE: A Benchmark Evaluating Earth Scientific Exploration Capability for Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.17139v2](http://arxiv.org/pdf/2505.17139v2)**

> **作者:** Wanghan Xu; Xiangyu Zhao; Yuhao Zhou; Xiaoyu Yue; Ben Fei; Fenghua Ling; Wenlong Zhang; Lei Bai
>
> **摘要:** Advancements in Large Language Models (LLMs) drive interest in scientific applications, necessitating specialized benchmarks such as Earth science. Existing benchmarks either present a general science focus devoid of Earth science specificity or cover isolated subdomains, lacking holistic evaluation. Furthermore, current benchmarks typically neglect the assessment of LLMs' capabilities in open-ended scientific exploration. In this paper, we present a comprehensive and professional benchmark for the Earth sciences, designed to evaluate the capabilities of LLMs in scientific exploration within this domain, spanning from fundamental to advanced levels. Leveraging a corpus of 100,000 research papers, we first construct two Question Answering (QA) datasets: Earth-Iron, which offers extensive question coverage for broad assessment, and Earth-Silver, which features a higher level of difficulty to evaluate professional depth. These datasets encompass five Earth spheres, 114 disciplines, and 11 task categories, assessing foundational knowledge crucial for scientific exploration. Most notably, we introduce Earth-Gold with new metrics, a dataset comprising open-ended multi-turn dialogues specifically designed to evaluate the advanced capabilities of LLMs in scientific exploration, including methodology induction, limitation analysis, and concept proposal. Extensive experiments reveal limitations in 11 leading LLMs across different domains and tasks, highlighting considerable room for improvement in their scientific exploration capabilities. The benchmark is available on https://huggingface.co/ai-earth .
>
---
#### [replaced 112] DynaCode: A Dynamic Complexity-Aware Code Benchmark for Evaluating Large Language Models in Code Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.10452v2](http://arxiv.org/pdf/2503.10452v2)**

> **作者:** Wenhao Hu; Jinhao Duan; Chunchen Wei; Li Zhang; Yue Zhang; Kaidi Xu
>
> **备注:** 18 pages, 13 figures. Accepted to the ACL 2025 Findings
>
> **摘要:** The rapid advancement of large language models (LLMs) has significantly improved their performance in code generation tasks. However, existing code benchmarks remain static, consisting of fixed datasets with predefined problems. This makes them vulnerable to memorization during training, where LLMs recall specific test cases instead of generalizing to new problems, leading to data contamination and unreliable evaluation results. To address these issues, we introduce DynaCode, a dynamic, complexity-aware benchmark that overcomes the limitations of static datasets. DynaCode evaluates LLMs systematically using a complexity-aware metric, incorporating both code complexity and call-graph structures. DynaCode achieves large-scale diversity, generating up to 189 million unique nested code problems across four distinct levels of code complexity, referred to as units, and 16 types of call graphs. Results on 12 latest LLMs show an average performance drop of 16.8% to 45.7% compared to MBPP+, a static code generation benchmark, with performance progressively decreasing as complexity increases. This demonstrates DynaCode's ability to effectively differentiate LLMs. Additionally, by leveraging call graphs, we gain insights into LLM behavior, particularly their preference for handling subfunction interactions within nested code. Our benchmark and evaluation code are available at https://github.com/HWH-2000/DynaCode.
>
---
#### [replaced 113] Learning to Reason under Off-Policy Guidance
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.14945v4](http://arxiv.org/pdf/2504.14945v4)**

> **作者:** Jianhao Yan; Yafu Li; Zican Hu; Zhi Wang; Ganqu Cui; Xiaoye Qu; Yu Cheng; Yue Zhang
>
> **备注:** Work in progress
>
> **摘要:** Recent advances in large reasoning models (LRMs) demonstrate that sophisticated behaviors such as multi-step reasoning and self-reflection can emerge via reinforcement learning with verifiable rewards~(\textit{RLVR}). However, existing \textit{RLVR} approaches are inherently ``on-policy'', limiting learning to a model's own outputs and failing to acquire reasoning abilities beyond its initial capabilities. To address this issue, we introduce \textbf{LUFFY} (\textbf{L}earning to reason \textbf{U}nder o\textbf{FF}-polic\textbf{Y} guidance), a framework that augments \textit{RLVR} with off-policy reasoning traces. LUFFY dynamically balances imitation and exploration by combining off-policy demonstrations with on-policy rollouts during training. Specifically, LUFFY combines the Mixed-Policy GRPO framework, which has a theoretically guaranteed convergence rate, alongside policy shaping via regularized importance sampling to avoid superficial and rigid imitation during mixed-policy training. Compared with previous RLVR methods, LUFFY achieves an over \textbf{+6.4} average gain across six math benchmarks and an advantage of over \textbf{+6.2} points in out-of-distribution tasks. Most significantly, we show that LUFFY successfully trains weak models in scenarios where on-policy RLVR completely fails. These results provide compelling evidence that LUFFY transcends the fundamental limitations of on-policy RLVR and demonstrates the great potential of utilizing off-policy guidance in RLVR.
>
---
#### [replaced 114] Re-ranking Using Large Language Models for Mitigating Exposure to Harmful Content on Social Media Platforms
- **分类: cs.CL; cs.AI; cs.CY; cs.SI**

- **链接: [http://arxiv.org/pdf/2501.13977v3](http://arxiv.org/pdf/2501.13977v3)**

> **作者:** Rajvardhan Oak; Muhammad Haroon; Claire Jo; Magdalena Wojcieszak; Anshuman Chhabra
>
> **备注:** Accepted to ACL 2025 Main Conference
>
> **摘要:** Social media platforms utilize Machine Learning (ML) and Artificial Intelligence (AI) powered recommendation algorithms to maximize user engagement, which can result in inadvertent exposure to harmful content. Current moderation efforts, reliant on classifiers trained with extensive human-annotated data, struggle with scalability and adapting to new forms of harm. To address these challenges, we propose a novel re-ranking approach using Large Language Models (LLMs) in zero-shot and few-shot settings. Our method dynamically assesses and re-ranks content sequences, effectively mitigating harmful content exposure without requiring extensive labeled data. Alongside traditional ranking metrics, we also introduce two new metrics to evaluate the effectiveness of re-ranking in reducing exposure to harmful content. Through experiments on three datasets, three models and across three configurations, we demonstrate that our LLM-based approach significantly outperforms existing proprietary moderation approaches, offering a scalable and adaptable solution for harm mitigation.
>
---
#### [replaced 115] EFIM: Efficient Serving of LLMs for Infilling Tasks with Improved KV Cache Reuse
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.21889v2](http://arxiv.org/pdf/2505.21889v2)**

> **作者:** Tianyu Guo; Hande Dong; Yichong Leng; Feng Liu; Cheater Lin; Nong Xiao; Xianwei Zhang
>
> **备注:** 31st International European Conference on Parallel and Distributed Computing (Euro-Par 2025 Oral)
>
> **摘要:** Large language models (LLMs) are often used for infilling tasks, which involve predicting or generating missing information in a given text. These tasks typically require multiple interactions with similar context. To reduce the computation of repeated historical tokens, cross-request key-value (KV) cache reuse, a technique that stores and reuses intermediate computations, has become a crucial method in multi-round interactive services. However, in infilling tasks, the KV cache reuse is often hindered by the structure of the prompt format, which typically consists of a prefix and suffix relative to the insertion point. Specifically, the KV cache of the prefix or suffix part is frequently invalidated as the other part (suffix or prefix) is incrementally generated. To address the issue, we propose EFIM, a transformed prompt format of FIM to unleash the performance potential of KV cache reuse. Although the transformed prompt can solve the inefficiency, it exposes subtoken generation problems in current LLMs, where they have difficulty generating partial words accurately. Therefore, we introduce a fragment tokenization training method which splits text into multiple fragments before tokenization during data processing. Experiments on two representative LLMs show that LLM serving with EFIM can lower the latency by 52% and improve the throughput by 98% while maintaining the original infilling capability. EFIM's source code is publicly available at https://github.com/gty111/EFIM.
>
---
#### [replaced 116] Joint Localization and Activation Editing for Low-Resource Fine-Tuning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.01179v4](http://arxiv.org/pdf/2502.01179v4)**

> **作者:** Wen Lai; Alexander Fraser; Ivan Titov
>
> **备注:** Accepted by ICML 2025 (camera-ready version). The code is released at https://github.com/wenlai-lavine/jola
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) methods, such as LoRA, are commonly used to adapt LLMs. However, the effectiveness of standard PEFT methods is limited in low-resource scenarios with only a few hundred examples. Recent advances in interpretability research have inspired the emergence of activation editing (or steering) techniques, which modify the activations of specific model components. Due to their extremely small parameter counts, these methods show promise for small datasets. However, their performance is highly dependent on identifying the correct modules to edit and often lacks stability across different datasets. In this paper, we propose Joint Localization and Activation Editing (JoLA), a method that jointly learns (1) which heads in the Transformer to edit (2) whether the intervention should be additive, multiplicative, or both and (3) the intervention parameters themselves - the vectors applied as additive offsets or multiplicative scalings to the head output. Through evaluations on three benchmarks spanning commonsense reasoning, natural language understanding, and natural language generation, we demonstrate that JoLA consistently outperforms existing methods. The code for the method is released at https://github.com/wenlai-lavine/jola.
>
---
#### [replaced 117] Theoretical guarantees on the best-of-n alignment policy
- **分类: cs.LG; cs.CL; cs.IT; math.IT**

- **链接: [http://arxiv.org/pdf/2401.01879v3](http://arxiv.org/pdf/2401.01879v3)**

> **作者:** Ahmad Beirami; Alekh Agarwal; Jonathan Berant; Alexander D'Amour; Jacob Eisenstein; Chirag Nagpal; Ananda Theertha Suresh
>
> **备注:** ICML 2025
>
> **摘要:** A simple and effective method for the inference-time alignment and scaling test-time compute of generative models is best-of-$n$ sampling, where $n$ samples are drawn from a reference policy, ranked based on a reward function, and the highest ranking one is selected. A commonly used analytical expression in the literature claims that the KL divergence between the best-of-$n$ policy and the reference policy is equal to $\log (n) - (n-1)/n.$ We disprove the validity of this claim, and show that it is an upper bound on the actual KL divergence. We also explore the tightness of this upper bound in different regimes, and propose a new estimator for the KL divergence and empirically show that it provides a tight approximation. We also show that the win rate of the best-of-$n$ policy against the reference policy is upper bounded by $n/(n+1)$ and derive bounds on the tightness of this characterization. We conclude with analyzing the tradeoffs between win rate and KL divergence of the best-of-$n$ alignment policy, which demonstrate that very good tradeoffs are achievable with $n < 1000$.
>
---
#### [replaced 118] LLM as Effective Streaming Processor: Bridging Streaming-Batch Mismatches with Group Position Encoding
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.16983v2](http://arxiv.org/pdf/2505.16983v2)**

> **作者:** Junlong Tong; Jinlan Fu; Zixuan Lin; Yingqi Fan; Anhao Zhao; Hui Su; Xiaoyu Shen
>
> **备注:** ACL 2025 Findings
>
> **摘要:** Large Language Models (LLMs) are primarily designed for batch processing. Existing methods for adapting LLMs to streaming rely either on expensive re-encoding or specialized architectures with limited scalability. This work identifies three key mismatches in adapting batch-oriented LLMs to streaming: (1) input-attention, (2) output-attention, and (3) position-ID mismatches. While it is commonly assumed that the latter two mismatches require frequent re-encoding, our analysis reveals that only the input-attention mismatch significantly impacts performance, indicating re-encoding outputs is largely unnecessary. To better understand this discrepancy with the common assumption, we provide the first comprehensive analysis of the impact of position encoding on LLMs in streaming, showing that preserving relative positions within source and target contexts is more critical than maintaining absolute order. Motivated by the above analysis, we introduce a group position encoding paradigm built on batch architectures to enhance consistency between streaming and batch modes. Extensive experiments on cross-lingual and cross-modal tasks demonstrate that our method outperforms existing approaches. Our method requires no architectural modifications, exhibits strong generalization in both streaming and batch modes. The code is available at repository https://github.com/EIT-NLP/StreamingLLM.
>
---
#### [replaced 119] RepCali: High Efficient Fine-tuning Via Representation Calibration in Latent Space for Pre-trained Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.08463v2](http://arxiv.org/pdf/2505.08463v2)**

> **作者:** Fujun Zhang; Xiaoying Fan; XiangDong Su; Guanglai Gao
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** Fine-tuning pre-trained language models (PLMs) has become a dominant paradigm in applying PLMs to downstream tasks. However, with limited fine-tuning, PLMs still struggle with the discrepancies between the representation obtained from the PLMs' encoder and the optimal input to the PLMs' decoder. This paper tackles this challenge by learning to calibrate the representation of PLMs in the latent space. In the proposed representation calibration method (RepCali), we integrate a specific calibration block to the latent space after the encoder and use the calibrated output as the decoder input. The merits of the proposed RepCali include its universality to all PLMs with encoder-decoder architectures, its plug-and-play nature, and ease of implementation. Extensive experiments on 25 PLM-based models across 8 tasks (including both English and Chinese datasets) demonstrate that the proposed RepCali offers desirable enhancements to PLMs (including LLMs) and significantly improves the performance of downstream tasks. Comparison experiments across 4 benchmark tasks indicate that RepCali is superior to the representative fine-tuning baselines.
>
---
#### [replaced 120] What's In Your Field? Mapping Scientific Research with Knowledge Graphs and Large Language Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.09894v2](http://arxiv.org/pdf/2503.09894v2)**

> **作者:** Abhipsha Das; Nicholas Lourie; Siavash Golkar; Mariel Pettee
>
> **备注:** 9 pages, 5 pdf figures
>
> **摘要:** The scientific literature's exponential growth makes it increasingly challenging to navigate and synthesize knowledge across disciplines. Large language models (LLMs) are powerful tools for understanding scientific text, but they fail to capture detailed relationships across large bodies of work. Unstructured approaches, like retrieval augmented generation, can sift through such corpora to recall relevant facts; however, when millions of facts influence the answer, unstructured approaches become cost prohibitive. Structured representations offer a natural complement -- enabling systematic analysis across the whole corpus. Recent work enhances LLMs with unstructured or semistructured representations of scientific concepts; to complement this, we try extracting structured representations using LLMs. By combining LLMs' semantic understanding with a schema of scientific concepts, we prototype a system that answers precise questions about the literature as a whole. Our schema applies across scientific fields and we extract concepts from it using only 20 manually annotated abstracts. To demonstrate the system, we extract concepts from 30,000 papers on arXiv spanning astrophysics, fluid dynamics, and evolutionary biology. The resulting database highlights emerging trends and, by visualizing the knowledge graph, offers new ways to explore the ever-growing landscape of scientific knowledge. Demo: abby101/surveyor-0 on HF Spaces. Code: https://github.com/chiral-carbon/kg-for-science.
>
---
#### [replaced 121] Sample-Efficient Human Evaluation of Large Language Models via Maximum Discrepancy Competition
- **分类: cs.LG; cs.CL; cs.HC**

- **链接: [http://arxiv.org/pdf/2404.08008v2](http://arxiv.org/pdf/2404.08008v2)**

> **作者:** Kehua Feng; Keyan Ding; Hongzhi Tan; Kede Ma; Zhihua Wang; Shuangquan Guo; Yuzhou Cheng; Ge Sun; Guozhou Zheng; Qiang Zhang; Huajun Chen
>
> **备注:** 35 pages, 6 figures, Accepted by ACL 2025
>
> **摘要:** Reliable evaluation of large language models (LLMs) is impeded by two key challenges: objective metrics often fail to reflect human perception of natural language, and exhaustive human labeling is prohibitively expensive. Here, we propose a sample-efficient human evaluation method for LLMs based on the principle of MAximum Discrepancy (MAD) Competition. Our method automatically and adaptively selects a compact set of input instructions that maximize semantic discrepancy between pairs of LLM responses. Human evaluators then perform three-alternative forced choices on these paired responses, which are aggregated into a global ranking using Elo rating. We apply our approach to compare eight widely used LLMs across four tasks: scientific knowledge understanding, mathematical reasoning, creative and functional writing, and code generation and explanation. Experimental results show that our sample-efficient evaluation method recovers "gold-standard" model rankings with a handful of MAD-selected instructions, reveals respective strengths and weaknesses of each LLM, and offers nuanced insights to guide future LLM development. Code is available at https://github.com/weiji-Feng/MAD-Eval .
>
---
#### [replaced 122] RULEBREAKERS: Challenging LLMs at the Crossroads between Formal Logic and Human-like Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.16502v2](http://arxiv.org/pdf/2410.16502v2)**

> **作者:** Jason Chan; Robert Gaizauskas; Zhixue Zhao
>
> **备注:** Preprint. Accepted by ICML 2025
>
> **摘要:** Formal logic enables computers to reason in natural language by representing sentences in symbolic forms and applying rules to derive conclusions. However, in what our study characterizes as "rulebreaker" scenarios, this method can lead to conclusions that are typically not inferred or accepted by humans given their common sense and factual knowledge. Inspired by works in cognitive science, we create RULEBREAKERS, the first dataset for rigorously evaluating the ability of large language models (LLMs) to recognize and respond to rulebreakers (versus non-rulebreakers) in a human-like manner. Evaluating seven LLMs, we find that most models, including GPT-4o, achieve mediocre accuracy on RULEBREAKERS and exhibit some tendency to over-rigidly apply logical rules unlike what is expected from typical human reasoners. Further analysis suggests that this apparent failure is potentially associated with the models' poor utilization of their world knowledge and their attention distribution patterns. Whilst revealing a limitation of current LLMs, our study also provides a timely counterbalance to a growing body of recent works that propose methods relying on formal logic to improve LLMs' general reasoning capabilities, highlighting their risk of further increasing divergence between LLMs and human-like reasoning.
>
---
#### [replaced 123] REVS: Unlearning Sensitive Information in Language Models via Rank Editing in the Vocabulary Space
- **分类: cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2406.09325v5](http://arxiv.org/pdf/2406.09325v5)**

> **作者:** Tomer Ashuach; Martin Tutek; Yonatan Belinkov
>
> **备注:** ACL 2025 Findings, 24 pages, 4 figures
>
> **摘要:** Language models (LMs) risk inadvertently memorizing and divulging sensitive or personally identifiable information (PII) seen in training data, causing privacy concerns. Current approaches to address this issue involve costly dataset scrubbing, or model filtering through unlearning and model editing, which can be bypassed through extraction attacks. We propose REVS, a novel non-gradient-based method for unlearning sensitive information from LMs. REVS identifies and modifies a small subset of neurons relevant for constituent tokens that form sensitive information. To adequately evaluate our method on truly sensitive information, we curate three datasets: email and URL datasets naturally memorized by the models, and a synthetic social security number dataset that we tune the models to memorize. Compared to other methods, REVS demonstrates superior performance in unlearning sensitive information and robustness to extraction attacks, while retaining underlying model integrity.
>
---
#### [replaced 124] On-Device Collaborative Language Modeling via a Mixture of Generalists and Specialists
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.13931v4](http://arxiv.org/pdf/2409.13931v4)**

> **作者:** Dongyang Fan; Bettina Messmer; Nikita Doikov; Martin Jaggi
>
> **备注:** Camera-ready version
>
> **摘要:** On-device LLMs have gained increasing attention for their ability to enhance privacy and provide a personalized user experience. To facilitate private learning with scarce data, Federated Learning has become a standard approach. However, it faces challenges such as computational resource heterogeneity and data heterogeneity among end users. We propose CoMiGS ($\textbf{Co}$llaborative learning with a $\textbf{Mi}$xture of $\textbf{G}$eneralists and $\textbf{S}$pecialists), the first approach to address both challenges. A key innovation of our method is the bi-level optimization formulation of the Mixture-of-Experts learning objective, where the router is optimized using a separate validation set to ensure alignment with the target distribution. We solve our objective with alternating minimization, for which we provide a theoretical analysis. Our method shares generalist experts across users while localizing a varying number of specialist experts, thereby adapting to users' computational resources and preserving privacy. Through extensive experiments, we show CoMiGS effectively balances general and personalized knowledge for each token generation. We demonstrate that CoMiGS remains robust against overfitting-due to the generalists' regularizing effect-while adapting to local data through specialist expertise. We open source our codebase for collaborative LLMs.
>
---
#### [replaced 125] K-Paths: Reasoning over Graph Paths for Drug Repurposing and Drug Interaction Prediction
- **分类: cs.LG; cs.CL; q-bio.BM**

- **链接: [http://arxiv.org/pdf/2502.13344v3](http://arxiv.org/pdf/2502.13344v3)**

> **作者:** Tassallah Abdullahi; Ioanna Gemou; Nihal V. Nayak; Ghulam Murtaza; Stephen H. Bach; Carsten Eickhoff; Ritambhara Singh
>
> **摘要:** Biomedical knowledge graphs (KGs) encode rich, structured information critical for drug discovery tasks, but extracting meaningful insights from large-scale KGs remains challenging due to their complex structure. Existing biomedical subgraph retrieval methods are tailored for graph neural networks (GNNs), limiting compatibility with other paradigms, including large language models (LLMs). We introduce K-Paths, a model-agnostic retrieval framework that extracts structured, diverse, and biologically meaningful multi-hop paths from dense biomedical KGs. These paths enable the prediction of unobserved drug-drug and drug-disease interactions, including those involving entities not seen during training, thus supporting inductive reasoning. K-Paths is training-free and employs a diversity-aware adaptation of Yen's algorithm to extract the K shortest loopless paths between entities in a query, prioritizing biologically relevant and relationally diverse connections. These paths serve as concise, interpretable reasoning chains that can be directly integrated with LLMs or GNNs to improve generalization, accuracy, and enable explainable inference. Experiments on benchmark datasets show that K-Paths improves zero-shot reasoning across state-of-the-art LLMs. For instance, Tx-Gemma 27B improves by 19.8 and 4.0 F1 points on interaction severity prediction and drug repurposing tasks, respectively. Llama 70B achieves gains of 8.5 and 6.2 points on the same tasks. K-Paths also boosts the training efficiency of EmerGNN, a state-of-the-art GNN, by reducing the KG size by 90% while maintaining predictive performance. Beyond efficiency, K-Paths bridges the gap between KGs and LLMs, enabling scalable and explainable LLM-augmented scientific discovery. We release our code and the retrieved paths as a benchmark for inductive reasoning.
>
---
#### [replaced 126] FCMR: Robust Evaluation of Financial Cross-Modal Multi-Hop Reasoning
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2412.12567v3](http://arxiv.org/pdf/2412.12567v3)**

> **作者:** Seunghee Kim; Changhyeon Kim; Taeuk Kim
>
> **备注:** ACL 2025
>
> **摘要:** Real-world decision-making often requires integrating and reasoning over information from multiple modalities. While recent multimodal large language models (MLLMs) have shown promise in such tasks, their ability to perform multi-hop reasoning across diverse sources remains insufficiently evaluated. Existing benchmarks, such as MMQA, face challenges due to (1) data contamination and (2) a lack of complex queries that necessitate operations across more than two modalities, hindering accurate performance assessment. To address this, we present Financial Cross-Modal Multi-Hop Reasoning (FCMR), a benchmark created to analyze the reasoning capabilities of MLLMs by urging them to combine information from textual reports, tables, and charts within the financial domain. FCMR is categorized into three difficulty levels-Easy, Medium, and Hard-facilitating a step-by-step evaluation. In particular, problems at the Hard level require precise cross-modal three-hop reasoning and are designed to prevent the disregard of any modality. Experiments on this new benchmark reveal that even state-of-the-art MLLMs struggle, with the best-performing model (Claude 3.5 Sonnet) achieving only 30.4% accuracy on the most challenging tier. We also conduct analysis to provide insights into the inner workings of the models, including the discovery of a critical bottleneck in the information retrieval phase.
>
---
#### [replaced 127] VideoRAG: Retrieval-Augmented Generation over Video Corpus
- **分类: cs.CV; cs.AI; cs.CL; cs.IR; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.05874v3](http://arxiv.org/pdf/2501.05874v3)**

> **作者:** Soyeong Jeong; Kangsan Kim; Jinheon Baek; Sung Ju Hwang
>
> **备注:** ACL Findings 2025
>
> **摘要:** Retrieval-Augmented Generation (RAG) is a powerful strategy for improving the factual accuracy of models by retrieving external knowledge relevant to queries and incorporating it into the generation process. However, existing approaches primarily focus on text, with some recent advancements considering images, and they largely overlook videos, a rich source of multimodal knowledge capable of representing contextual details more effectively than any other modality. While very recent studies explore the use of videos in response generation, they either predefine query-associated videos without retrieval or convert videos into textual descriptions losing multimodal richness. To tackle these, we introduce VideoRAG, a framework that not only dynamically retrieves videos based on their relevance with queries but also utilizes both visual and textual information. The operation of VideoRAG is powered by recent Large Video Language Models (LVLMs), which enable the direct processing of video content to represent it for retrieval and the seamless integration of retrieved videos jointly with queries for response generation. Also, inspired by that the context size of LVLMs may not be sufficient to process all frames in extremely long videos and not all frames are equally important, we introduce a video frame selection mechanism to extract the most informative subset of frames, along with a strategy to extract textual information from videos (as it can aid the understanding of video content) when their subtitles are not available. We experimentally validate the effectiveness of VideoRAG, showcasing that it is superior to relevant baselines. Code is available at https://github.com/starsuzi/VideoRAG.
>
---
#### [replaced 128] ReflectionCoder: Learning from Reflection Sequence for Enhanced One-off Code Generation
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2405.17057v2](http://arxiv.org/pdf/2405.17057v2)**

> **作者:** Houxing Ren; Mingjie Zhan; Zhongyuan Wu; Aojun Zhou; Junting Pan; Hongsheng Li
>
> **备注:** Accepted to ACL 2025 (main conference)
>
> **摘要:** Code generation plays a crucial role in various tasks, such as code auto-completion and mathematical reasoning. Previous work has proposed numerous methods to enhance code generation performance, including integrating feedback from the compiler. Inspired by this, we present ReflectionCoder, a novel approach that effectively leverages reflection sequences constructed by integrating compiler feedback to improve one-off code generation performance. Furthermore, we propose reflection self-distillation and dynamically masked distillation to effectively utilize these reflection sequences. Extensive experiments on three benchmarks, i.e., HumanEval (+), MBPP (+), and MultiPL-E, demonstrate that models fine-tuned with our method achieve state-of-the-art performance. Beyond the code domain, we believe this approach can benefit other domains that focus on final results and require long reasoning paths. Code and data are available at https://github.com/SenseLLM/ReflectionCoder.
>
---
#### [replaced 129] Position: Scaling LLM Agents Requires Asymptotic Analysis with LLM Primitives
- **分类: cs.CL; cs.AI; cs.CC; cs.LG; cs.NE**

- **链接: [http://arxiv.org/pdf/2502.04358v2](http://arxiv.org/pdf/2502.04358v2)**

> **作者:** Elliot Meyerson; Xin Qiu
>
> **备注:** In Proceedings of the 42nd International Conference on Machine Learning (ICML 2025); 13 pages including references
>
> **摘要:** Decomposing hard problems into subproblems often makes them easier and more efficient to solve. With large language models (LLMs) crossing critical reliability thresholds for a growing slate of capabilities, there is an increasing effort to decompose systems into sets of LLM-based agents, each of whom can be delegated sub-tasks. However, this decomposition (even when automated) is often intuitive, e.g., based on how a human might assign roles to members of a human team. How close are these role decompositions to optimal? This position paper argues that asymptotic analysis with LLM primitives is needed to reason about the efficiency of such decomposed systems, and that insights from such analysis will unlock opportunities for scaling them. By treating the LLM forward pass as the atomic unit of computational cost, one can separate out the (often opaque) inner workings of a particular LLM from the inherent efficiency of how a set of LLMs are orchestrated to solve hard problems. In other words, if we want to scale the deployment of LLMs to the limit, instead of anthropomorphizing LLMs, asymptotic analysis with LLM primitives should be used to reason about and develop more powerful decompositions of large problems into LLM agents.
>
---
#### [replaced 130] Hijacking Large Language Models via Adversarial In-Context Learning
- **分类: cs.LG; cs.CL; cs.CR**

- **链接: [http://arxiv.org/pdf/2311.09948v3](http://arxiv.org/pdf/2311.09948v3)**

> **作者:** Xiangyu Zhou; Yao Qiang; Saleh Zare Zade; Prashant Khanduri; Dongxiao Zhu
>
> **摘要:** In-context learning (ICL) has emerged as a powerful paradigm leveraging LLMs for specific downstream tasks by utilizing labeled examples as demonstrations (demos) in the preconditioned prompts. Despite its promising performance, crafted adversarial attacks pose a notable threat to the robustness of LLMs. Existing attacks are either easy to detect, require a trigger in user input, or lack specificity towards ICL. To address these issues, this work introduces a novel transferable prompt injection attack against ICL, aiming to hijack LLMs to generate the target output or elicit harmful responses. In our threat model, the hacker acts as a model publisher who leverages a gradient-based prompt search method to learn and append imperceptible adversarial suffixes to the in-context demos via prompt injection. We also propose effective defense strategies using a few shots of clean demos, enhancing the robustness of LLMs during ICL. Extensive experimental results across various classification and jailbreak tasks demonstrate the effectiveness of the proposed attack and defense strategies. This work highlights the significant security vulnerabilities of LLMs during ICL and underscores the need for further in-depth studies.
>
---
