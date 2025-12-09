# 自然语言处理 cs.CL

- **最新发布 109 篇**

- **更新 70 篇**

## 最新发布

#### [new 001] "The Dentist is an involved parent, the bartender is not": Revealing Implicit Biases in QA with Implicit BBQ
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于NLP公平性评估任务，旨在揭示大语言模型中隐含偏见问题。针对现有基准仅依赖显式属性的局限，提出ImplicitBBQ，通过隐式线索评测偏见，发现GPT-4o在性取向等类别准确率显著下降，表明模型存在未被显式基准捕捉的隐性偏见。**

- **链接: [https://arxiv.org/pdf/2512.06732v1](https://arxiv.org/pdf/2512.06732v1)**

> **作者:** Aarushi Wagh; Saniya Srivastava
>
> **摘要:** Existing benchmarks evaluating biases in large language models (LLMs) primarily rely on explicit cues, declaring protected attributes like religion, race, gender by name. However, real-world interactions often contain implicit biases, inferred subtly through names, cultural cues, or traits. This critical oversight creates a significant blind spot in fairness evaluation. We introduce ImplicitBBQ, a benchmark extending the Bias Benchmark for QA (BBQ) with implicitly cued protected attributes across 6 categories. Our evaluation of GPT-4o on ImplicitBBQ illustrates troubling performance disparity from explicit BBQ prompts, with accuracy declining up to 7% in the "sexual orientation" subcategory and consistent decline located across most other categories. This indicates that current LLMs contain implicit biases undetected by explicit benchmarks. ImplicitBBQ offers a crucial tool for nuanced fairness evaluation in NLP.
>
---
#### [new 002] Enhancing Agentic RL with Progressive Reward Shaping and Value-based Sampling Policy Optimization
- **分类: cs.CL**

- **简介: 该论文研究工具增强大模型的强化学习优化，针对奖励稀疏和梯度退化问题，提出渐进式奖励塑造（PRS）和基于价值采样的策略优化（VSPO），提升训练效率与性能，在多类问答任务中实现更优泛化。**

- **链接: [https://arxiv.org/pdf/2512.07478v1](https://arxiv.org/pdf/2512.07478v1)**

> **作者:** Zhuoran Zhuang; Ye Chen; Jianghao Su; Chao Luo; Luhui Liu; Xia Zeng
>
> **摘要:** Large Language Models (LLMs) empowered with Tool-Integrated Reasoning (TIR) can iteratively plan, call external tools, and integrate returned information to solve complex, long-horizon reasoning tasks. Agentic Reinforcement Learning (Agentic RL) optimizes such models over full tool-interaction trajectories, but two key challenges hinder effectiveness: (1) Sparse, non-instructive rewards, such as binary 0-1 verifiable signals, provide limited guidance for intermediate steps and slow convergence; (2) Gradient degradation in Group Relative Policy Optimization (GRPO), where identical rewards within a rollout group yield zero advantage, reducing sample efficiency and destabilizing training. To address these challenges, we propose two complementary techniques: Progressive Reward Shaping (PRS) and Value-based Sampling Policy Optimization (VSPO). PRS is a curriculum-inspired reward design that introduces dense, stage-wise feedback - encouraging models to first master parseable and properly formatted tool calls, then optimize for factual correctness and answer quality. We instantiate PRS for short-form QA (with a length-aware BLEU to fairly score concise answers) and long-form QA (with LLM-as-a-Judge scoring to prevent reward hacking). VSPO is an enhanced GRPO variant that replaces low-value samples with prompts selected by a task-value metric balancing difficulty and uncertainty, and applies value-smoothing clipping to stabilize gradient updates. Experiments on multiple short-form and long-form QA benchmarks show that PRS consistently outperforms traditional binary rewards, and VSPO achieves superior stability, faster convergence, and higher final performance compared to PPO, GRPO, CISPO, and SFT-only baselines. Together, PRS and VSPO yield LLM-based TIR agents that generalize better across domains.
>
---
#### [new 003] XAM: Interactive Explainability for Authorship Attribution Models
- **分类: cs.CL**

- **简介: 该论文针对作者归属模型的可解释性问题，提出IXAM交互式框架，帮助用户探索模型嵌入空间，构建多粒度写作风格特征解释。通过用户评估验证了其相比预定义解释的有效性。**

- **链接: [https://arxiv.org/pdf/2512.06924v1](https://arxiv.org/pdf/2512.06924v1)**

> **作者:** Milad Alshomary; Anisha Bhatnagar; Peter Zeng; Smaranda Muresan; Owen Rambow; Kathleen McKeown
>
> **摘要:** We present IXAM, an Interactive eXplainability framework for Authorship Attribution Models. Given an authorship attribution (AA) task and an embedding-based AA model, our tool enables users to interactively explore the model's embedding space and construct an explanation of the model's prediction as a set of writing style features at different levels of granularity. Through a user evaluation, we demonstrate the value of our framework compared to predefined stylistic explanations.
>
---
#### [new 004] PersonaMem-v2: Towards Personalized Intelligence via Learning Implicit User Personas and Agentic Memory
- **分类: cs.CL**

- **简介: 该论文聚焦大模型个性化任务，旨在解决隐式用户偏好理解与长上下文推理瓶颈问题。作者构建了PersonaMem-v2数据集，提出强化微调方法与代理记忆框架，提升模型在少输入令牌下对隐式个性化的准确率，推动个性化智能发展。**

- **链接: [https://arxiv.org/pdf/2512.06688v1](https://arxiv.org/pdf/2512.06688v1)**

> **作者:** Bowen Jiang; Yuan Yuan; Maohao Shen; Zhuoqun Hao; Zhangchen Xu; Zichen Chen; Ziyi Liu; Anvesh Rao Vijjini; Jiashu He; Hanchao Yu; Radha Poovendran; Gregory Wornell; Lyle Ungar; Dan Roth; Sihao Chen; Camillo Jose Taylor
>
> **备注:** Data is available at https://huggingface.co/datasets/bowen-upenn/PersonaMem-v2
>
> **摘要:** Personalization is one of the next milestones in advancing AI capability and alignment. We introduce PersonaMem-v2, the state-of-the-art dataset for LLM personalization that simulates 1,000 realistic user-chatbot interactions on 300+ scenarios, 20,000+ user preferences, and 128k-token context windows, where most user preferences are implicitly revealed to reflect real-world interactions. Using this data, we investigate how reinforcement fine-tuning enables a model to improve its long-context reasoning capabilities for user understanding and personalization. We also develop a framework for training an agentic memory system, which maintains a single, human-readable memory that grows with each user over time. In our experiments, frontier LLMs still struggle with implicit personalization, achieving only 37-48% accuracy. While they support long context windows, reasoning remains the bottleneck for implicit personalization tasks. Using reinforcement fine-tuning, we successfully train Qwen3-4B to outperforms GPT-5, reaching 53% accuracy in implicit personalization. Moreover, our agentic memory framework achieves state-of-the-art 55% accuracy while using 16x fewer input tokens, relying on a 2k-token memory instead of full 32k conversation histories. These results underscore the impact of our dataset and demonstrate agentic memory as a scalable path toward real-world personalized intelligence.
>
---
#### [new 005] TopiCLEAR: Topic extraction by CLustering Embeddings with Adaptive dimensional Reduction
- **分类: cs.CL**

- **简介: 该论文属于文本挖掘任务，旨在解决传统主题建模在短文本（如社交媒体）中效果差的问题。提出TopiCLEAR方法，结合SBERT嵌入、自适应降维与高斯混合模型迭代聚类，无需预处理，提升主题提取的准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2512.06694v1](https://arxiv.org/pdf/2512.06694v1)**

> **作者:** Aoi Fujita; Taichi Yamamoto; Yuri Nakayama; Ryota Kobayashi
>
> **备注:** 15 pages, 4 figures, code available at https://github.com/aoi8716/TopiCLEAR
>
> **摘要:** Rapid expansion of social media platforms such as X (formerly Twitter), Facebook, and Reddit has enabled large-scale analysis of public perceptions on diverse topics, including social issues, politics, natural disasters, and consumer sentiment. Topic modeling is a widely used approach for uncovering latent themes in text data, typically framed as an unsupervised classification task. However, traditional models, originally designed for longer and more formal documents, struggle with short social media posts due to limited co-occurrence statistics, fragmented semantics, inconsistent spelling, and informal language. To address these challenges, we propose a new method, TopiCLEAR: Topic extraction by CLustering Embeddings with Adaptive dimensional Reduction. Specifically, each text is embedded using Sentence-BERT (SBERT) and provisionally clustered using Gaussian Mixture Models (GMM). The clusters are then refined iteratively using a supervised projection based on linear discriminant analysis, followed by GMM-based clustering until convergence. Notably, our method operates directly on raw text, eliminating the need for preprocessing steps such as stop word removal. We evaluate our approach on four diverse datasets, 20News, AgNewsTitle, Reddit, and TweetTopic, each containing human-labeled topic information. Compared with seven baseline methods, including a recent SBERT-based method and a zero-shot generative AI method, our approach achieves the highest similarity to human-annotated topics, with significant improvements for both social media posts and online news articles. Additionally, qualitative analysis shows that our method produces more interpretable topics, highlighting its potential for applications in social media data and web content analytics.
>
---
#### [new 006] Training Language Models to Use Prolog as a Tool
- **分类: cs.CL**

- **简介: 该论文研究语言模型调用Prolog作为外部工具以提升推理可靠性。针对工具使用不可靠问题，作者通过GRPO强化学习微调Qwen2.5-3B-Instruct，优化提示、奖励与推理策略，显著提升GSM8K和MMLU等任务的准确率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.07407v1](https://arxiv.org/pdf/2512.07407v1)**

> **作者:** Niklas Mellgren; Peter Schneider-Kamp; Lukas Galke Poech
>
> **备注:** 10 pages
>
> **摘要:** Ensuring reliable tool use is critical for safe agentic AI systems. Language models frequently produce unreliable reasoning with plausible but incorrect solutions that are difficult to verify. To address this, we investigate fine-tuning models to use Prolog as an external tool for verifiable computation. Using Group Relative Policy Optimization (GRPO), we fine-tune Qwen2.5-3B-Instruct on a cleaned GSM8K-Prolog-Prover dataset while varying (i) prompt structure, (ii) reward composition (execution, syntax, semantics, structure), and (iii) inference protocol: single-shot, best-of-N, and two agentic modes where Prolog is invoked internally or independently. Our reinforcement learning approach outperforms supervised fine-tuning, with our 3B model achieving zero-shot MMLU performance comparable to 7B few-shot results. Our findings reveal that: 1) joint tuning of prompt, reward, and inference shapes program syntax and logic; 2) best-of-N with external Prolog verification maximizes accuracy on GSM8K; 3) agentic inference with internal repair yields superior zero-shot generalization on MMLU-Stem and MMLU-Pro. These results demonstrate that grounding model reasoning in formal verification systems substantially improves reliability and auditability for safety-critical applications. The source code for reproducing our experiments is available under https://github.com/niklasmellgren/grpo-prolog-inference
>
---
#### [new 007] Becoming Experienced Judges: Selective Test-Time Learning for Evaluators
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究LLM自动评估任务，旨在解决评估过程中缺乏经验积累和样本特异性标准的问题。提出Learning While Evaluating（LWE）及Selective LWE框架，通过自反馈动态优化元提示，实现推理时持续学习，提升评估性能。**

- **链接: [https://arxiv.org/pdf/2512.06751v1](https://arxiv.org/pdf/2512.06751v1)**

> **作者:** Seungyeon Jwa; Daechul Ahn; Reokyoung Kim; Dongyeop Kang; Jonghyun Choi
>
> **摘要:** Automatic evaluation with large language models, commonly known as LLM-as-a-judge, is now standard across reasoning and alignment tasks. Despite evaluating many samples in deployment, these evaluators typically (i) treat each case independently, missing the opportunity to accumulate experience, and (ii) rely on a single fixed prompt for all cases, neglecting the need for sample-specific evaluation criteria. We introduce Learning While Evaluating (LWE), a framework that allows evaluators to improve sequentially at inference time without requiring training or validation sets. LWE maintains an evolving meta-prompt that (i) produces sample-specific evaluation instructions and (ii) refines itself through self-generated feedback. Furthermore, we propose Selective LWE, which updates the meta-prompt only on self-inconsistent cases, focusing computation where it matters most. This selective approach retains the benefits of sequential learning while being far more cost-effective. Across two pairwise comparison benchmarks, Selective LWE outperforms strong baselines, empirically demonstrating that evaluators can improve during sequential testing with a simple selective update, learning most from the cases they struggle with.
>
---
#### [new 008] Bridging Code Graphs and Large Language Models for Better Code Understanding
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于代码理解任务，旨在解决LLMs因依赖线性序列而忽视程序结构语义的问题。提出CGBridge方法，通过可训练的桥接模块融合代码图信息与大模型，实现结构感知的代码理解，提升性能并保持高效推理。**

- **链接: [https://arxiv.org/pdf/2512.07666v1](https://arxiv.org/pdf/2512.07666v1)**

> **作者:** Zeqi Chen; Zhaoyang Chu; Yi Gui; Feng Guo; Yao Wan; Chuan Shi
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable performance in code intelligence tasks such as code generation, summarization, and translation. However, their reliance on linearized token sequences limits their ability to understand the structural semantics of programs. While prior studies have explored graphaugmented prompting and structure-aware pretraining, they either suffer from prompt length constraints or require task-specific architectural changes that are incompatible with large-scale instructionfollowing LLMs. To address these limitations, this paper proposes CGBridge, a novel plug-and-play method that enhances LLMs with Code Graph information through an external, trainable Bridge module. CGBridge first pre-trains a code graph encoder via selfsupervised learning on a large-scale dataset of 270K code graphs to learn structural code semantics. It then trains an external module to bridge the modality gap among code, graph, and text by aligning their semantics through cross-modal attention mechanisms. Finally, the bridge module generates structure-informed prompts, which are injected into a frozen LLM, and is fine-tuned for downstream code intelligence tasks. Experiments show that CGBridge achieves notable improvements over both the original model and the graphaugmented prompting method. Specifically, it yields a 16.19% and 9.12% relative gain in LLM-as-a-Judge on code summarization, and a 9.84% and 38.87% relative gain in Execution Accuracy on code translation. Moreover, CGBridge achieves over 4x faster inference than LoRA-tuned models, demonstrating both effectiveness and efficiency in structure-aware code understanding.
>
---
#### [new 009] Performance of the SafeTerm AI-Based MedDRA Query System Against Standardised MedDRA Queries
- **分类: cs.CL**

- **简介: 该论文研究AI系统SafeTerm在MedDRA术语查询中的性能，旨在提升药物安全审查中不良事件检索的效率与准确性。通过构建语义向量空间并计算相似度，实现自动术语匹配与排序，验证其在SMQ/OCMQ查询中平衡召回率与精确率的有效性。**

- **链接: [https://arxiv.org/pdf/2512.07552v1](https://arxiv.org/pdf/2512.07552v1)**

> **作者:** Francois Vandenhende; Anna Georgiou; Michalis Georgiou; Theodoros Psaras; Ellie Karekla; Elena Hadjicosta
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** In pre-market drug safety review, grouping related adverse event terms into SMQs or OCMQs is critical for signal detection. We assess the performance of SafeTerm Automated Medical Query (AMQ) on MedDRA SMQs. The AMQ is a novel quantitative artificial intelligence system that understands and processes medical terminology and automatically retrieves relevant MedDRA Preferred Terms (PTs) for a given input query, ranking them by a relevance score (0-1) using multi-criteria statistical methods. The system (SafeTerm) embeds medical query terms and MedDRA PTs in a multidimensional vector space, then applies cosine similarity, and extreme-value clustering to generate a ranked list of PTs. Validation was conducted against tier-1 SMQs (110 queries, v28.1). Precision, recall and F1 were computed at multiple similarity-thresholds, defined either manually or using an automated method. High recall (94%)) is achieved at moderate similarity thresholds, indicative of good retrieval sensitivity. Higher thresholds filter out more terms, resulting in improved precision (up to 89%). The optimal threshold (0.70)) yielded an overall recall of (48%) and precision of (45%) across all 110 queries. Restricting to narrow-term PTs achieved slightly better performance at an increased (+0.05) similarity threshold, confirming increased relatedness of narrow versus broad terms. The automatic threshold (0.66) selection prioritizes recall (0.58) to precision (0.29). SafeTerm AMQ achieves comparable, satisfactory performance on SMQs and sanitized OCMQs. It is therefore a viable supplementary method for automated MedDRA query generation, balancing recall and precision. We recommend using suitable MedDRA PT terminology in query formulation and applying the automated threshold method to optimise recall. Increasing similarity scores allows refined, narrow terms selection.
>
---
#### [new 010] Policy-based Sentence Simplification: Replacing Parallel Corpora with LLM-as-a-Judge
- **分类: cs.CL**

- **简介: 该论文研究句子简化任务，旨在根据特定策略生成简化句。为解决依赖平行语料和缺乏策略控制的问题，提出用大模型作为评判器自动构建训练数据，无需人工标注，实现策略对齐的句子简化，且在多种模型上验证了有效性。**

- **链接: [https://arxiv.org/pdf/2512.06228v1](https://arxiv.org/pdf/2512.06228v1)**

> **作者:** Xuanxin Wu; Yuki Arase; Masaaki Nagata
>
> **摘要:** Sentence simplification aims to modify a sentence to make it easier to read and understand while preserving the meaning. Different applications require distinct simplification policies, such as replacing only complex words at the lexical level or rewriting the entire sentence while trading off details for simplicity. However, achieving such policy-driven control remains an open challenge. In this work, we introduce a simple yet powerful approach that leverages Large Language Model-as-a-Judge (LLM-as-a-Judge) to automatically construct policy-aligned training data, completely removing the need for costly human annotation or parallel corpora. Our method enables building simplification systems that adapt to diverse simplification policies. Remarkably, even small-scale open-source LLMs such as Phi-3-mini-3.8B surpass GPT-4o on lexical-oriented simplification, while achieving comparable performance on overall rewriting, as verified by both automatic metrics and human evaluations. The consistent improvements across model families and sizes demonstrate the robustness of our approach.
>
---
#### [new 011] Rhea: Role-aware Heuristic Episodic Attention for Conversational LLMs
- **分类: cs.CL**

- **简介: 该论文针对多轮对话中大模型因上下文衰减导致性能下降的问题，提出Rhea框架，通过分离指令记忆与情景记忆，结合优先注意力机制，提升上下文利用效率与指令遵循能力，显著改善长对话表现。**

- **链接: [https://arxiv.org/pdf/2512.06869v1](https://arxiv.org/pdf/2512.06869v1)**

> **作者:** Wanyang Hong; Zhaoning Zhang; Yi Chen; Libo Zhang; Baihui Liu; Linbo Qiao; Zhiliang Tian; Dongsheng Li
>
> **摘要:** Large Language Models (LLMs) have achieved remarkable performance on single-turn tasks, yet their effectiveness deteriorates in multi-turn conversations. We define this phenomenon as cumulative contextual decay - a progressive degradation of contextual integrity caused by attention pollution, dilution, and drift. To address this challenge, we propose Rhea (Role-aware Heuristic Episodic Attention), a novel framework that decouples conversation history into two functionally independent memory modules: (1) an Instructional Memory (IM) that persistently stores high-fidelity global constraints via a structural priority mechanism, and (2) an Episodic Memory (EM) that dynamically manages user-model interactions via asymmetric noise control and heuristic context retrieval. During inference, Rhea constructs a high signal-to-noise context by applying its priority attention: selectively integrating relevant episodic information while always prioritizing global instructions. To validate this approach, experiments on multiple multi-turn conversation benchmarks - including MT-Eval and Long-MT-Bench+ - show that Rhea mitigates performance decay and improves overall accuracy by 1.04 points on a 10-point scale (a 16% relative gain over strong baselines). Moreover, Rhea maintains near-perfect instruction fidelity (IAR > 8.1) across long-horizon interactions. These results demonstrate that Rhea provides a principled and effective framework for building more precise, instruction-consistent conversational LLMs.
>
---
#### [new 012] Morphologically-Informed Tokenizers for Languages with Non-Concatenative Morphology: A case study of Yoloxóchtil Mixtec ASR
- **分类: cs.CL**

- **简介: 该论文研究使用形态学感知的分词器提升Yoloxóchitl Mixtec语语音识别与标注效率，解决其非线性形态带来的分词难题。提出两种新分词方法，兼顾音调信息，在ASR中表现优于传统模型。**

- **链接: [https://arxiv.org/pdf/2512.06169v1](https://arxiv.org/pdf/2512.06169v1)**

> **作者:** Chris Crawford
>
> **备注:** 67 pages, 5 figures, 6 tables
>
> **摘要:** This paper investigates the impact of using morphologically-informed tokenizers to aid and streamline the interlinear gloss annotation of an audio corpus of Yoloxóchitl Mixtec (YM) using a combination of ASR and text-based sequence-to-sequence tools, with the goal of improving efficiency while reducing the workload of a human annotator. We present two novel tokenization schemes that separate words in a nonlinear manner, preserving information about tonal morphology as much as possible. One of these approaches, a Segment and Melody tokenizer, simply extracts the tones without predicting segmentation. The other, a Sequence of Processes tokenizer, predicts segmentation for the words, which could allow an end-to-end ASR system to produce segmented and unsegmented transcriptions in a single pass. We find that these novel tokenizers are competitive with BPE and Unigram models, and the Segment-and-Melody model outperforms traditional tokenizers in terms of word error rate but does not reach the same character error rate. In addition, we analyze tokenizers on morphological and information-theoretic metrics to find predictive correlations with downstream performance. Our results suggest that nonlinear tokenizers designed specifically for the non-concatenative morphology of a language are competitive with conventional BPE and Unigram models for ASR. Further research will be necessary to determine the applicability of these tokenizers in downstream processing tasks.
>
---
#### [new 013] Parameter-Efficient Fine-Tuning with Differential Privacy for Robust Instruction Adaptation in Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型的指令微调任务，解决隐私保护与训练效率问题。提出一种参数高效微调方法，结合差分隐私、梯度裁剪与低维投影，降低隐私预算消耗并提升稳定性，在多任务场景下实现安全、高效的指令适应。**

- **链接: [https://arxiv.org/pdf/2512.06711v1](https://arxiv.org/pdf/2512.06711v1)**

> **作者:** Yulin Huang; Yaxuan Luan; Jinxu Guo; Xiangchen Song; Yuchen Liu
>
> **摘要:** This study addresses the issues of privacy protection and efficiency in instruction fine-tuning of large-scale language models by proposing a parameter-efficient method that integrates differential privacy noise allocation with gradient clipping in a collaborative optimization framework. The method keeps the backbone model frozen and updates parameters through a low-dimensional projection subspace, while introducing clipping and adaptive noise allocation during gradient computation. This design reduces privacy budget consumption and ensures training stability and robustness. The unified framework combines gradient constraints, noise allocation, and parameter projection, effectively mitigating performance fluctuations and privacy risks in multi-task instruction scenarios. Experiments are conducted across hyperparameter, environment, and data sensitivity dimensions. Results show that the method outperforms baseline models in accuracy, privacy budget, and parameter efficiency, and maintains stable performance under diverse and uncertain data conditions. The findings enrich the theoretical integration of differential privacy and parameter-efficient fine-tuning and demonstrate its practical adaptability in instruction tasks, providing a feasible solution for secure training in complex instruction environments.
>
---
#### [new 014] SwissGov-RSD: A Human-annotated, Cross-lingual Benchmark for Token-level Recognition of Semantic Differences Between Related Documents
- **分类: cs.CL**

- **简介: 该论文提出SwissGov-RSD，首个自然的跨语言文档级语义差异识别基准，含224组多语言文件及词级人工标注。针对现有模型在该任务上表现不佳的问题，评估了多种大模型与编码器模型，揭示了当前方法的不足。**

- **链接: [https://arxiv.org/pdf/2512.07538v1](https://arxiv.org/pdf/2512.07538v1)**

> **作者:** Michelle Wastl; Jannis Vamvas; Rico Sennrich
>
> **备注:** 30 pages
>
> **摘要:** Recognizing semantic differences across documents, especially in different languages, is crucial for text generation evaluation and multilingual content alignment. However, as a standalone task it has received little attention. We address this by introducing SwissGov-RSD, the first naturalistic, document-level, cross-lingual dataset for semantic difference recognition. It encompasses a total of 224 multi-parallel documents in English-German, English-French, and English-Italian with token-level difference annotations by human annotators. We evaluate a variety of open-source and closed source large language models as well as encoder models across different fine-tuning settings on this new benchmark. Our results show that current automatic approaches perform poorly compared to their performance on monolingual, sentence-level, and synthetic benchmarks, revealing a considerable gap for both LLMs and encoder models. We make our code and datasets publicly available.
>
---
#### [new 015] AquaFusionNet: Lightweight VisionSensor Fusion Framework for Real-Time Pathogen Detection and Water Quality Anomaly Prediction on Edge Devices
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出AquaFusionNet，属多模态融合任务，旨在解决边缘设备上实时病原检测与水质异常预测难题。通过融合显微图像与传感器数据，利用轻量级交叉注意力机制，在低功耗下实现高精度联合检测，提升复杂环境下的鲁棒性，并公开全部资源支持去中心化水安全应用。**

- **链接: [https://arxiv.org/pdf/2512.06848v1](https://arxiv.org/pdf/2512.06848v1)**

> **作者:** Sepyan Purnama Kristanto; Lutfi Hakim; Hermansyah
>
> **备注:** 9Pages, 3 figure, Politeknik Negeri Banyuwangi
>
> **摘要:** Evidence from many low and middle income regions shows that microbial contamination in small scale drinking water systems often fluctuates rapidly, yet existing monitoring tools capture only fragments of this behaviour. Microscopic imaging provides organism level visibility, whereas physicochemical sensors reveal shortterm changes in water chemistry; in practice, operators must interpret these streams separately, making realtime decision-making unreliable. This study introduces AquaFusionNet, a lightweight cross-modal framework that unifies both information sources inside a single edge deployable model. Unlike prior work that treats microscopic detection and water quality prediction as independent tasks, AquaFusionNet learns the statistical dependencies between microbial appearance and concurrent sensor dynamics through a gated crossattention mechanism designed specifically for lowpower hardware. The framework is trained on AquaMicro12K, a new dataset comprising 12,846 annotated 1000 micrographs curated for drinking water contexts, an area where publicly accessible microscopic datasets are scarce. Deployed for six months across seven facilities in East Java, Indonesia, the system processed 1.84 million frames and consistently detected contamination events with 94.8% mAP@0.5 and 96.3% anomaly prediction accuracy, while operating at 4.8 W on a Jetson Nano. Comparative experiments against representative lightweight detectors show that AquaFusionNet provides higher accuracy at comparable or lower power, and field results indicate that cross-modal coupling reduces common failure modes of unimodal detectors, particularly under fouling, turbidity spikes, and inconsistent illumination. All models, data, and hardware designs are released openly to facilitate replication and adaptation in decentralized water safety infrastructures.
>
---
#### [new 016] Most over-representation of phonological features in basic vocabulary disappears when controlling for spatial and phylogenetic effects
- **分类: cs.CL**

- **简介: 该论文检验语言基本词汇中语音特征过度表征的普遍性，旨在排除谱系和区域依赖带来的偏差。通过扩大样本至2864种语言并控制空间与系统发育效应，发现多数原有音义关联模式不再显著，仅少数稳定存在，强调验证语言普遍性主张需多层面稳健性检验。**

- **链接: [https://arxiv.org/pdf/2512.07543v1](https://arxiv.org/pdf/2512.07543v1)**

> **作者:** Frederic Blum
>
> **备注:** Accepted with minor revisions at *Linguistic Typology*, expected to be fully published in 2026
>
> **摘要:** The statistical over-representation of phonological features in the basic vocabulary of languages is often interpreted as reflecting potentially universal sound symbolic patterns. However, most of those results have not been tested explicitly for reproducibility and might be prone to biases in the study samples or models. Many studies on the topic do not adequately control for genealogical and areal dependencies between sampled languages, casting doubts on the robustness of the results. In this study, we test the robustness of a recent study on sound symbolism of basic vocabulary concepts which analyzed245 languages.The new sample includes data on 2864 languages from Lexibank. We modify the original model by adding statistical controls for spatial and phylogenetic dependencies between languages. The new results show that most of the previously observed patterns are not robust, and in fact many patterns disappear completely when adding the genealogical and areal controls. A small number of patterns, however, emerges as highly stable even with the new sample. Through the new analysis, we are able to assess the distribution of sound symbolism on a larger scale than previously. The study further highlights the need for testing all universal claims on language for robustness on various levels.
>
---
#### [new 017] CMV-Fuse: Cross Modal-View Fusion of AMR, Syntax, and Knowledge Representations for Aspect Based Sentiment Analysis
- **分类: cs.CL**

- **简介: 该论文针对方面级情感分析（ABSA）中多语言视角融合不足的问题，提出CMV-Fuse框架，融合AMR、句法结构与外部知识，通过分层门控注意力和多视图对比学习，提升情感分析性能。**

- **链接: [https://arxiv.org/pdf/2512.06679v1](https://arxiv.org/pdf/2512.06679v1)**

> **作者:** Smitha Muthya Sudheendra; Mani Deep Cherukuri; Jaideep Srivastava
>
> **摘要:** Natural language understanding inherently depends on integrating multiple complementary perspectives spanning from surface syntax to deep semantics and world knowledge. However, current Aspect-Based Sentiment Analysis (ABSA) systems typically exploit isolated linguistic views, thereby overlooking the intricate interplay between structural representations that humans naturally leverage. We propose CMV-Fuse, a Cross-Modal View fusion framework that emulates human language processing by systematically combining multiple linguistic perspectives. Our approach systematically orchestrates four linguistic perspectives: Abstract Meaning Representations, constituency parsing, dependency syntax, and semantic attention, enhanced with external knowledge integration. Through hierarchical gated attention fusion across local syntactic, intermediate semantic, and global knowledge levels, CMV-Fuse captures both fine-grained structural patterns and broad contextual understanding. A novel structure aware multi-view contrastive learning mechanism ensures consistency across complementary representations while maintaining computational efficiency. Extensive experiments demonstrate substantial improvements over strong baselines on standard benchmarks, with analysis revealing how each linguistic view contributes to more robust sentiment analysis.
>
---
#### [new 018] Adapting AlignScore Mertic for Factual Consistency Evaluation of Text in Russian: A Student Abstract
- **分类: cs.CL**

- **简介: 该论文针对俄语文本事实一致性评估工具缺失的问题，提出AlignRuScore，通过微调RuBERT模型并构建俄语及翻译数据集，实现了AlignScore指标在俄语的适配，推动多语言事实一致性评估研究。**

- **链接: [https://arxiv.org/pdf/2512.06586v1](https://arxiv.org/pdf/2512.06586v1)**

> **作者:** Mikhail Zimin; Milyausha Shamsutdinova; Georgii Andriushchenko
>
> **摘要:** Ensuring factual consistency in generated text is crucial for reliable natural language processing applications. However, there is a lack of evaluation tools for factual consistency in Russian texts, as existing tools primarily focus on English corpora. To bridge this gap, we introduce AlignRuScore, a comprehensive adaptation of the AlignScore metric for Russian. To adapt the metric, we fine-tuned a RuBERT-based alignment model with task-specific classification and regression heads on Russian and translated English datasets. Our results demonstrate that a unified alignment metric can be successfully ported to Russian, laying the groundwork for robust multilingual factual consistency evaluation. We release the translated corpora, model checkpoints, and code to support further research.
>
---
#### [new 019] Efficient ASR for Low-Resource Languages: Leveraging Cross-Lingual Unlabeled Data
- **分类: cs.CL; eess.AS**

- **简介: 该论文研究低资源语言的语音识别，旨在解决标注数据少、计算资源需求高的问题。通过跨语言无监督数据持续预训练和形态感知分词，构建高效模型，在较少参数和标注数据下实现优越性能。**

- **链接: [https://arxiv.org/pdf/2512.07277v1](https://arxiv.org/pdf/2512.07277v1)**

> **作者:** Srihari Bandarupalli; Bhavana Akkiraju; Charan Devarakonda; Vamsiraghusimha Narsinga; Anil Kumar Vuppala
>
> **备注:** Accepted in AACL IJCNLP 2025
>
> **摘要:** Automatic speech recognition for low-resource languages remains fundamentally constrained by the scarcity of labeled data and computational resources required by state-of-the-art models. We present a systematic investigation into cross-lingual continuous pretraining for low-resource languages, using Perso-Arabic languages (Persian, Arabic, and Urdu) as our primary case study. Our approach demonstrates that strategic utilization of unlabeled speech data can effectively bridge the resource gap without sacrificing recognition accuracy. We construct a 3,000-hour multilingual corpus through a scalable unlabeled data collection pipeline and employ targeted continual pretraining combined with morphologically-aware tokenization to develop a 300M parameter model that achieves performance comparable to systems 5 times larger. Our model outperforms Whisper Large v3 (1.5B parameters) on Persian and achieves competitive results on Arabic and Urdu despite using significantly fewer parameters and substantially less labeled data. These findings challenge the prevailing assumption that ASR quality scales primarily with model size, revealing instead that data relevance and strategic pretraining are more critical factors for low-resource scenarios. This work provides a practical pathway toward inclusive speech technology, enabling effective ASR for underrepresented languages without dependence on massive computational infrastructure or proprietary datasets.
>
---
#### [new 020] TeluguST-46: A Benchmark Corpus and Comprehensive Evaluation for Telugu-English Speech Translation
- **分类: cs.CL; eess.AS**

- **简介: 该论文聚焦泰卢固语-英语语音翻译任务，旨在解决低资源语言对研究不足的问题。作者构建了高质量基准数据集TeluguST-46，系统比较级联与端到端模型，并评估自动评测指标的有效性，为形态复杂语言的语音翻译提供可复现基准和实用指导。**

- **链接: [https://arxiv.org/pdf/2512.07265v1](https://arxiv.org/pdf/2512.07265v1)**

> **作者:** Bhavana Akkiraju; Srihari Bandarupalli; Swathi Sambangi; Vasavi Ravuri; R Vijaya Saraswathi; Anil Kumar Vuppala
>
> **备注:** Submitted to AACL IJCNLP 2025
>
> **摘要:** Despite Telugu being spoken by over 80 million people, speech translation research for this morphologically rich language remains severely underexplored. We address this gap by developing a high-quality Telugu--English speech translation benchmark from 46 hours of manually verified CSTD corpus data (30h/8h/8h train/dev/test split). Our systematic comparison of cascaded versus end-to-end architectures shows that while IndicWhisper + IndicMT achieves the highest performance due to extensive Telugu-specific training data, finetuned SeamlessM4T models demonstrate remarkable competitiveness despite using significantly less Telugu-specific training data. This finding suggests that with careful hyperparameter tuning and sufficient parallel data (potentially less than 100 hours), end-to-end systems can achieve performance comparable to cascaded approaches in low-resource settings. Our metric reliability study evaluating BLEU, METEOR, ChrF++, ROUGE-L, TER, and BERTScore against human judgments reveals that traditional metrics provide better quality discrimination than BERTScore for Telugu--English translation. The work delivers three key contributions: a reproducible Telugu--English benchmark, empirical evidence of competitive end-to-end performance potential in low-resource scenarios, and practical guidance for automatic evaluation in morphologically complex language pairs.
>
---
#### [new 021] Large Language Model-Based Generation of Discharge Summaries
- **分类: cs.CL**

- **简介: 该论文研究基于大语言模型自动生成出院总结的任务，旨在减轻医疗人员负担并提升信息准确性。作者评估了五个大模型在MIMIC-III数据上的表现，发现闭源模型（尤其是Gemini）效果最优，且经临床专家验证具实用性。**

- **链接: [https://arxiv.org/pdf/2512.06812v1](https://arxiv.org/pdf/2512.06812v1)**

> **作者:** Tiago Rodrigues; Carla Teixeira Lopes
>
> **备注:** 17 pages, 6 figures
>
> **摘要:** Discharge Summaries are documents written by medical professionals that detail a patient's visit to a care facility. They contain a wealth of information crucial for patient care, and automating their generation could significantly reduce the effort required from healthcare professionals, minimize errors, and ensure that critical patient information is easily accessible and actionable. In this work, we explore the use of five Large Language Models on this task, from open-source models (Mistral, Llama 2) to proprietary systems (GPT-3, GPT-4, Gemini 1.5 Pro), leveraging MIMIC-III summaries and notes. We evaluate them using exact-match, soft-overlap, and reference-free metrics. Our results show that proprietary models, particularly Gemini with one-shot prompting, outperformed others, producing summaries with the highest similarity to the gold-standard ones. Open-source models, while promising, especially Mistral after fine-tuning, lagged in performance, often struggling with hallucinations and repeated information. Human evaluation by a clinical expert confirmed the practical utility of the summaries generated by proprietary models. Despite the challenges, such as hallucinations and missing information, the findings suggest that LLMs, especially proprietary models, are promising candidates for automatic discharge summary generation as long as data privacy is ensured.
>
---
#### [new 022] Progress Ratio Embeddings: An Impatience Signal for Robust Length Control in Neural Text Generation
- **分类: cs.CL**

- **简介: 该论文属于文本生成任务，旨在解决神经文本生成中长度控制不稳定的问题。作者提出Progress Ratio Embeddings（PRE），用连续的三角函数“不耐烦”信号替代离散倒计时，实现对生成长度的鲁棒控制，并验证其在新闻摘要任务中的有效性。**

- **链接: [https://arxiv.org/pdf/2512.06938v1](https://arxiv.org/pdf/2512.06938v1)**

> **作者:** Ivanhoé Botcazou; Tassadit Amghar; Sylvain Lamprier; Frédéric Saubion
>
> **摘要:** Modern neural language models achieve high accuracy in text generation, yet precise control over generation length remains underdeveloped. In this paper, we first investigate a recent length control method based on Reverse Positional Embeddings (RPE) and show its limits when control is requested beyond the training distribution. In particular, using a discrete countdown signal tied to the absolute remaining token count leads to instability. To provide robust length control, we introduce Progress Ratio Embeddings (PRE), as continuous embeddings tied to a trigonometric impatience signal. PRE integrates seamlessly into standard Transformer architectures, providing stable length fidelity without degrading text accuracy under standard evaluation metrics. We further show that PRE generalizes well to unseen target lengths. Experiments on two widely used news-summarization benchmarks validate these findings.
>
---
#### [new 023] Modeling Contextual Passage Utility for Multihop Question Answering
- **分类: cs.CL**

- **简介: 该论文针对多跳问答中的段落效用建模，提出一种考虑段落间依赖关系的上下文感知方法。通过利用推理轨迹构建训练数据，使用小型Transformer模型预测段落效用，提升段落重排序与问答性能。**

- **链接: [https://arxiv.org/pdf/2512.06464v1](https://arxiv.org/pdf/2512.06464v1)**

> **作者:** Akriti Jain; Aparna Garimella
>
> **备注:** Accepted at IJCNLP-AACL 2025
>
> **摘要:** Multihop Question Answering (QA) requires systems to identify and synthesize information from multiple text passages. While most prior retrieval methods assist in identifying relevant passages for QA, further assessing the utility of the passages can help in removing redundant ones, which may otherwise add to noise and inaccuracies in the generated answers. Existing utility prediction approaches model passage utility independently, overlooking a critical aspect of multihop reasoning: the utility of a passage can be context-dependent, influenced by its relation to other passages - whether it provides complementary information or forms a crucial link in conjunction with others. In this paper, we propose a lightweight approach to model contextual passage utility, accounting for inter-passage dependencies. We fine-tune a small transformer-based model to predict passage utility scores for multihop QA. We leverage the reasoning traces from an advanced reasoning model to capture the order in which passages are used to answer a question and obtain synthetic training data. Through comprehensive experiments, we demonstrate that our utility-based scoring of retrieved passages leads to improved reranking and downstream QA performance compared to relevance-based reranking methods.
>
---
#### [new 024] Complementary Learning Approach for Text Classification using Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属文本分类任务，旨在解决大语言模型（LLM）在实际应用中的成本与缺陷问题。作者提出一种互补学习方法，结合人类溯因推理与LLM的少样本学习，通过人机协作分析1934份药企新闻稿，有效识别并修正人机判断差异。**

- **链接: [https://arxiv.org/pdf/2512.07583v1](https://arxiv.org/pdf/2512.07583v1)**

> **作者:** Navid Asgari; Benjamin M. Cole
>
> **备注:** 67 pages
>
> **摘要:** In this study, we propose a structured methodology that utilizes large language models (LLMs) in a cost-efficient and parsimonious manner, integrating the strengths of scholars and machines while offsetting their respective weaknesses. Our methodology, facilitated through a chain of thought and few-shot learning prompting from computer science, extends best practices for co-author teams in qualitative research to human-machine teams in quantitative research. This allows humans to utilize abductive reasoning and natural language to interrogate not just what the machine has done but also what the human has done. Our method highlights how scholars can manage inherent weaknesses OF LLMs using careful, low-cost techniques. We demonstrate how to use the methodology to interrogate human-machine rating discrepancies for a sample of 1,934 press releases announcing pharmaceutical alliances (1990-2017).
>
---
#### [new 025] Beyond Real: Imaginary Extension of Rotary Position Embeddings for Long-Context LLMs
- **分类: cs.CL**

- **简介: 该论文针对长文本大模型中的位置编码问题，指出RoPE丢弃复数虚部导致信息损失。提出利用完整复数表示构建双分量注意力得分，保留更多位置关系信息，提升长程依赖建模能力，实验证明其在长上下文任务中优于标准RoPE。**

- **链接: [https://arxiv.org/pdf/2512.07525v1](https://arxiv.org/pdf/2512.07525v1)**

> **作者:** Xiaoran Liu; Yuerong Song; Zhigeng Liu; Zengfeng Huang; Qipeng Guo; Zhaoxiang Liu; Shiguo Lian; Ziwei He; Xipeng Qiu
>
> **备注:** 20 pages, 6 figures, under review
>
> **摘要:** Rotary Position Embeddings (RoPE) have become a standard for encoding sequence order in Large Language Models (LLMs) by applying rotations to query and key vectors in the complex plane. Standard implementations, however, utilize only the real component of the complex-valued dot product for attention score calculation. This simplification discards the imaginary component, which contains valuable phase information, leading to a potential loss of relational details crucial for modeling long-context dependencies. In this paper, we propose an extension that re-incorporates this discarded imaginary component. Our method leverages the full complex-valued representation to create a dual-component attention score. We theoretically and empirically demonstrate that this approach enhances the modeling of long-context dependencies by preserving more positional information. Furthermore, evaluations on a suite of long-context language modeling benchmarks show that our method consistently improves performance over the standard RoPE, with the benefits becoming more significant as context length increases. The code is available at https://github.com/OpenMOSS/rope_pp.
>
---
#### [new 026] Do Large Language Models Truly Understand Cross-cultural Differences?
- **分类: cs.CL**

- **简介: 该论文聚焦大语言模型的跨文化理解能力评估，针对现有基准缺乏情境、概念映射和深层推理的问题，提出SAGE基准，基于文化理论构建多维度、场景化的测试集，揭示模型在跨文化推理上的系统性缺陷。**

- **链接: [https://arxiv.org/pdf/2512.07075v1](https://arxiv.org/pdf/2512.07075v1)**

> **作者:** Shiwei Guo; Sihang Jiang; Qianxi He; Yanghua Xiao; Jiaqing Liang; Bi Yude; Minggui He; Shimin Tao; Li Zhang
>
> **摘要:** In recent years, large language models (LLMs) have demonstrated strong performance on multilingual tasks. Given its wide range of applications, cross-cultural understanding capability is a crucial competency. However, existing benchmarks for evaluating whether LLMs genuinely possess this capability suffer from three key limitations: a lack of contextual scenarios, insufficient cross-cultural concept mapping, and limited deep cultural reasoning capabilities. To address these gaps, we propose SAGE, a scenario-based benchmark built via cross-cultural core concept alignment and generative task design, to evaluate LLMs' cross-cultural understanding and reasoning. Grounded in cultural theory, we categorize cross-cultural capabilities into nine dimensions. Using this framework, we curated 210 core concepts and constructed 4530 test items across 15 specific real-world scenarios, organized under four broader categories of cross-cultural situations, following established item design principles. The SAGE dataset supports continuous expansion, and experiments confirm its transferability to other languages. It reveals model weaknesses across both dimensions and scenarios, exposing systematic limitations in cross-cultural reasoning. While progress has been made, LLMs are still some distance away from reaching a truly nuanced cross-cultural understanding. In compliance with the anonymity policy, we include data and code in the supplement materials. In future versions, we will make them publicly available online.
>
---
#### [new 027] Knowing What's Missing: Assessing Information Sufficiency in Question Answering
- **分类: cs.CL**

- **简介: 该论文属于问答系统任务，旨在判断上下文是否包含回答问题所需的充分信息。针对现有方法在推理类问题上的不足，提出“先识别缺失信息、再验证”的框架，通过生成缺失信息假设并验证其存在性，提升信息充分性判断的准确性。**

- **链接: [https://arxiv.org/pdf/2512.06476v1](https://arxiv.org/pdf/2512.06476v1)**

> **作者:** Akriti Jain; Aparna Garimella
>
> **摘要:** Determining whether a provided context contains sufficient information to answer a question is a critical challenge for building reliable question-answering systems. While simple prompting strategies have shown success on factual questions, they frequently fail on inferential ones that require reasoning beyond direct text extraction. We hypothesize that asking a model to first reason about what specific information is missing provides a more reliable, implicit signal for assessing overall sufficiency. To this end, we propose a structured Identify-then-Verify framework for robust sufficiency modeling. Our method first generates multiple hypotheses about missing information and establishes a semantic consensus. It then performs a critical verification step, forcing the model to re-examine the source text to confirm whether this information is truly absent. We evaluate our method against established baselines across diverse multi-hop and factual QA datasets. The results demonstrate that by guiding the model to justify its claims about missing information, our framework produces more accurate sufficiency judgments while clearly articulating any information gaps.
>
---
#### [new 028] Persian-Phi: Efficient Cross-Lingual Adaptation of Compact LLMs via Curriculum Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于跨语言适配任务，旨在解决低资源语言因算力需求高而难以训练大模型的问题。作者提出Persian-Phi，通过课程学习结合参数高效微调，将英文小模型高效适配为波斯语模型，在保持小规模的同时实现优异性能。**

- **链接: [https://arxiv.org/pdf/2512.07454v1](https://arxiv.org/pdf/2512.07454v1)**

> **作者:** Amir Mohammad Akhlaghi; Amirhossein Shabani; Mostafa Abdolmaleki; Saeed Reza Kheradpisheh
>
> **摘要:** The democratization of AI is currently hindered by the immense computational costs required to train Large Language Models (LLMs) for low-resource languages. This paper presents Persian-Phi, a 3.8B parameter model that challenges the assumption that robust multilingual capabilities require massive model sizes or multilingual baselines. We demonstrate how Microsoft Phi-3 Mini -- originally a monolingual English model -- can be effectively adapted to Persian through a novel, resource-efficient curriculum learning pipeline. Our approach employs a unique "warm-up" stage using bilingual narratives (Tiny Stories) to align embeddings prior to heavy training, followed by continual pretraining and instruction tuning via Parameter-Efficient Fine-Tuning (PEFT). Despite its compact size, Persian-Phi achieves competitive results on Open Persian LLM Leaderboard in HuggingFace. Our findings provide a validated, scalable framework for extending the reach of state-of-the-art LLMs to underrepresented languages with minimal hardware resources. The Persian-Phi model is publicly available at https://huggingface.co/amirakhlaghiqqq/PersianPhi.
>
---
#### [new 029] Ensembling LLM-Induced Decision Trees for Explainable and Robust Error Detection
- **分类: cs.CL**

- **简介: 该论文研究数据纠错中的错误检测任务，旨在提升现有大模型直接标注方法的可解释性与鲁棒性。作者提出利用大模型生成决策树（TreeED），并通过集成多棵树实现共识检测（ForestED），结合GNN与规则节点增强准确性与可解释性，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2512.07246v1](https://arxiv.org/pdf/2512.07246v1)**

> **作者:** Mengqi Wang; Jianwei Wang; Qing Liu; Xiwei Xu; Zhenchang Xing; Liming Zhu; Wenjie Zhang
>
> **备注:** 14 pages, 8 figures
>
> **摘要:** Error detection (ED), which aims to identify incorrect or inconsistent cell values in tabular data, is important for ensuring data quality. Recent state-of-the-art ED methods leverage the pre-trained knowledge and semantic capability embedded in large language models (LLMs) to directly label whether a cell is erroneous. However, this LLM-as-a-labeler pipeline (1) relies on the black box, implicit decision process, thus failing to provide explainability for the detection results, and (2) is highly sensitive to prompts, yielding inconsistent outputs due to inherent model stochasticity, therefore lacking robustness. To address these limitations, we propose an LLM-as-an-inducer framework that adopts LLM to induce the decision tree for ED (termed TreeED) and further ensembles multiple such trees for consensus detection (termed ForestED), thereby improving explainability and robustness. Specifically, based on prompts derived from data context, decision tree specifications and output requirements, TreeED queries the LLM to induce the decision tree skeleton, whose root-to-leaf decision paths specify the stepwise procedure for evaluating a given sample. Each tree contains three types of nodes: (1) rule nodes that perform simple validation checks (e.g., format or range), (2) Graph Neural Network (GNN) nodes that capture complex patterns (e.g., functional dependencies), and (3) leaf nodes that output the final decision types (error or clean). Furthermore, ForestED employs uncertainty-based sampling to obtain multiple row subsets, constructing a decision tree for each subset using TreeED. It then leverages an Expectation-Maximization-based algorithm that jointly estimates tree reliability and optimizes the consensus ED prediction. Extensive xperiments demonstrate that our methods are accurate, explainable and robust, achieving an average F1-score improvement of 16.1% over the best baseline.
>
---
#### [new 030] When Large Language Models Do Not Work: Online Incivility Prediction through Graph Neural Networks
- **分类: cs.CL; cs.AI; cs.SI**

- **简介: 该论文研究在线不文明行为检测，提出基于图神经网络（GNN）的框架，利用评论间的文本相似性构建图结构，结合动态注意力机制融合内容与结构信息。实验证明其在准确性和效率上优于12种大语言模型。**

- **链接: [https://arxiv.org/pdf/2512.07684v1](https://arxiv.org/pdf/2512.07684v1)**

> **作者:** Zihan Chen; Lanyu Yu
>
> **备注:** 10 pages
>
> **摘要:** Online incivility has emerged as a widespread and persistent problem in digital communities, imposing substantial social and psychological burdens on users. Although many platforms attempt to curb incivility through moderation and automated detection, the performance of existing approaches often remains limited in both accuracy and efficiency. To address this challenge, we propose a Graph Neural Network (GNN) framework for detecting three types of uncivil behavior (i.e., toxicity, aggression, and personal attacks) within the English Wikipedia community. Our model represents each user comment as a node, with textual similarity between comments defining the edges, allowing the network to jointly learn from both linguistic content and relational structures among comments. We also introduce a dynamically adjusted attention mechanism that adaptively balances nodal and topological features during information aggregation. Empirical evaluations demonstrate that our proposed architecture outperforms 12 state-of-the-art Large Language Models (LLMs) across multiple metrics while requiring significantly lower inference cost. These findings highlight the crucial role of structural context in detecting online incivility and address the limitations of text-only LLM paradigms in behavioral prediction. All datasets and comparative outputs will be publicly available in our repository to support further research and reproducibility.
>
---
#### [new 031] Mechanistic Interpretability of GPT-2: Lexical and Contextual Layers in Sentiment Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究GPT-2在情感分析中的可解释性机制，探究其如何分层处理词汇与上下文情感信息。通过激活补丁实验，验证了早期层检测词汇情感，而上下文整合发生于晚期层，推翻了原有层级假设，揭示了非模块化的统一处理机制。**

- **链接: [https://arxiv.org/pdf/2512.06681v1](https://arxiv.org/pdf/2512.06681v1)**

> **作者:** Amartya Hatua
>
> **摘要:** We present a mechanistic interpretability study of GPT-2 that causally examines how sentiment information is processed across its transformer layers. Using systematic activation patching across all 12 layers, we test the hypothesized two-stage sentiment architecture comprising early lexical detection and mid-layer contextual integration. Our experiments confirm that early layers (0-3) act as lexical sentiment detectors, encoding stable, position specific polarity signals that are largely independent of context. However, all three contextual integration hypotheses: Middle Layer Concentration, Phenomenon Specificity, and Distributed Processing are falsified. Instead of mid-layer specialization, we find that contextual phenomena such as negation, sarcasm, domain shifts etc. are integrated primarily in late layers (8-11) through a unified, non-modular mechanism. These experimental findings provide causal evidence that GPT-2's sentiment computation differs from the predicted hierarchical pattern, highlighting the need for further empirical characterization of contextual integration in large language models.
>
---
#### [new 032] From Next-Token to Next-Block: A Principled Adaptation Path for Diffusion LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究从自回归到块扩散的模型适应，解决训练扩散语言模型成本高的问题。提出渐进式适应路径，保持训练与推理一致，在7B模型上实现高效迁移，提升生成性能。**

- **链接: [https://arxiv.org/pdf/2512.06776v1](https://arxiv.org/pdf/2512.06776v1)**

> **作者:** Yuchuan Tian; Yuchen Liang; Jiacheng Sun; Shuo Zhang; Guangwen Yang; Yingte Shu; Sibo Fang; Tianyu Guo; Kai Han; Chao Xu; Hanting Chen; Xinghao Chen; Yunhe Wang
>
> **备注:** 13 pages, 4 figures
>
> **摘要:** Large language models (LLMs) excel at generation but dominant autoregressive (AR) decoding is inherently sequential, creating a throughput bottleneck. Diffusion Language Models (DLMs)--especially block-wise variants--enable parallel generation and intra-block bidirectional reasoning, yet training large DLMs from scratch is costly and wastes the knowledge in mature AR checkpoints. Prior "adaptation" attempts either modify logits or randomly grow attention masks to full-sequence diffusion, or simply transplant AR weights into a block-diffusion recipe, leaving a fundamental mismatch between AR causality and block-wise bidirectionality unaddressed. We reframe adaptation as a intra-paradigm path from AR to Block-Diffusion by viewing AR as Block-Diffusion with blocksize=1. Concretely, we design the pathway of adaptation as follows: we use a context-causal attention mask (causal in context, bidirectional only within the active block), an efficient parallel adaptation procedure, an auxiliary AR loss to maximize data utilization and retain pretrained knowledge, and gradual increment of the generation block size. The recipe integrates cleanly with masked block-diffusion and maintains train-inference consistency. Built on these components, NBDiff-7B (Base and Instruct) could inherit the long-context modeling and reasoning capabilities, and achieve state-of-the-art performance among the 7B-class DLMs, delivering strong gains on general-knowledge, math, and code benchmarks over strong baselines. These results demonstrate that principled AR-to-block-diffusion adaptation is an effective and compute-efficient alternative to training DLMs from scratch. Codes: https://github.com/YuchuanTian/NBDiff.
>
---
#### [new 033] Nanbeige4-3B Technical Report: Exploring the Frontier of Small Language Models
- **分类: cs.CL**

- **简介: 该论文聚焦小规模语言模型的性能优化，旨在突破其能力边界。通过设计FG-WSD训练调度、联合SFT数据增强、双偏好蒸馏及多阶段强化学习，提升模型推理与对齐能力，使Nanbeige4-3B在多项基准上超越同规模模型并媲美更大模型。**

- **链接: [https://arxiv.org/pdf/2512.06266v1](https://arxiv.org/pdf/2512.06266v1)**

> **作者:** Chen Yang; Guangyue Peng; Jiaying Zhu; Ran Le; Ruixiang Feng; Tao Zhang; Wei Ruan; Xiaoqi Liu; Xiaoxue Cheng; Xiyun Xu; Yang Song; Yanzipeng Gao; Yiming Jia; Yun Xing; Yuntao Wen; Zekai Wang; Zhenwei An; Zhicong Sun; Zongchao Chen
>
> **摘要:** We present Nanbeige4-3B, a family of small-scale but high-performing language models. Pretrained on 23T high-quality tokens and finetuned on over 30 million diverse instructions, we extend the boundary of the scaling law for small language models. In pre-training, we design a Fine-Grained Warmup-Stable-Decay (FG-WSD) training scheduler, which progressively refines data mixtures across stages to boost model performance. In post-training, to improve the quality of the SFT data, we design a joint mechanism that integrates deliberative generation refinement and chain-of-thought reconstruction, yielding substantial gains on complex tasks. Following SFT, we employ our flagship reasoning model to distill Nanbeige4-3B through our proposed Dual Preference Distillation (DPD) method, which leads to further performance gains. Finally, a multi-stage reinforcement learning phase was applied, leveraging verifiable rewards and preference modeling to strengthen abilities on both reasoning and human alignment. Extensive evaluations show that Nanbeige4-3B not only significantly outperforms models of comparable parameter scale but also rivals much larger models across a wide range of benchmarks. The model checkpoints are available at https://huggingface.co/Nanbeige.
>
---
#### [new 034] Do You Feel Comfortable? Detecting Hidden Conversational Escalation in AI Chatbots
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话安全任务，旨在解决AI聊天机器人中隐性情感升级导致的隐性伤害问题。作者提出GAUGE框架，通过logit级概率变化实时检测对话情感偏移，实现对隐性对话升级的轻量级监测。**

- **链接: [https://arxiv.org/pdf/2512.06193v1](https://arxiv.org/pdf/2512.06193v1)**

> **作者:** Jihyung Park; Saleh Afroogh; Junfeng Jiao
>
> **摘要:** Large Language Models (LLM) are increasingly integrated into everyday interactions, serving not only as information assistants but also as emotional companions. Even in the absence of explicit toxicity, repeated emotional reinforcement or affective drift can gradually escalate distress in a form of \textit{implicit harm} that traditional toxicity filters fail to detect. Existing guardrail mechanisms often rely on external classifiers or clinical rubrics that may lag behind the nuanced, real-time dynamics of a developing conversation. To address this gap, we propose GAUGE (Guarding Affective Utterance Generation Escalation), a lightweight, logit-based framework for the real-time detection of hidden conversational escalation. GAUGE measures how an LLM's output probabilistically shifts the affective state of a dialogue.
>
---
#### [new 035] Automated PRO-CTCAE Symptom Selection based on Prior Adverse Event Profiles
- **分类: cs.CL**

- **简介: 该论文旨在解决PRO-CTCAE症状项选择中平衡覆盖性与患者负担的问题。通过映射至MedDRA并利用Safeterm语义空间，结合相关性、发生率与多样性，提出一种自动化症状项筛选方法，实现基于历史数据的最优子集选取。**

- **链接: [https://arxiv.org/pdf/2512.06919v1](https://arxiv.org/pdf/2512.06919v1)**

> **作者:** Francois Vandenhende; Anna Georgiou; Michalis Georgiou; Theodoros Psaras; Ellie Karekla
>
> **备注:** 13 pages, 2 figures
>
> **摘要:** The PRO-CTCAE is an NCI-developed patient-reported outcome system for capturing symptomatic adverse events in oncology trials. It comprises a large library drawn from the CTCAE vocabulary, and item selection for a given trial is typically guided by expected toxicity profiles from prior data. Selecting too many PRO-CTCAE items can burden patients and reduce compliance, while too few may miss important safety signals. We present an automated method to select a minimal yet comprehensive PRO-CTCAE subset based on historical safety data. Each candidate PRO-CTCAE symptom term is first mapped to its corresponding MedDRA Preferred Terms (PTs), which are then encoded into Safeterm, a high-dimensional semantic space capturing clinical and contextual diversity in MedDRA terminology. We score each candidate PRO item for relevance to the historical list of adverse event PTs and combine relevance and incidence into a utility function. Spectral analysis is then applied to the combined utility and diversity matrix to identify an orthogonal set of medical concepts that balances relevance and diversity. Symptoms are rank-ordered by importance, and a cut-off is suggested based on the explained information. The tool is implemented as part of the Safeterm trial-safety app. We evaluate its performance using simulations and oncology case studies in which PRO-CTCAE was employed. This automated approach can streamline PRO-CTCAE design by leveraging MedDRA semantics and historical data, providing an objective and reproducible method to balance signal coverage against patient burden.
>
---
#### [new 036] One Word Is Not Enough: Simple Prompts Improve Word Embeddings
- **分类: cs.CL**

- **简介: 该论文研究词嵌入任务，旨在提升大模型对孤立单词的语义表示能力。通过添加简单语义提示（如“meaning: {word}”），显著提高词相似度任务性能，无需训练即可超越传统静态嵌入方法。**

- **链接: [https://arxiv.org/pdf/2512.06744v1](https://arxiv.org/pdf/2512.06744v1)**

> **作者:** Rajeev Ranjan
>
> **摘要:** Text embedding models are designed for sentence-level applications like retrieval and semantic similarity, and are primarily evaluated on sentence-level benchmarks. Their behavior on isolated words is less understood. We show that simply prepending semantic prompts to words before embedding substantially improves word similarity correlations. Testing 7 text embedding models, including text-embedding-3-large (OpenAI), embed-english-v3.0 (Cohere), voyage-3(Voyage AI), all-mpnet-base-v2, and Qwen3-Embedding-8B, on 3 standard benchmarks (SimLex-999, WordSim-353, MEN-3000), we find that prompts like "meaning: {word}" or "Represent the semantic concept: {word}" improve Spearman correlations by up to +0.29 on SimLex-999. Some models fail completely on bare words (correlation = 0) but recover with prompts (+0.73 improvement). Our best results achieve correlation = 0.692 on SimLex-999 with embed-english-v3.0 (Cohere), correlation = 0.811 on WordSim-353, and correlation = 0.855 on MEN-3000 with text-embedding-3-large (OpenAI). These results outperform classic static embeddings like Word2Vec (correlation = 0.40) and even the best static method LexVec (correlation = 0.48) on SimLex-999, establishing a new state-of-the-art for pure embedding methods. This zero-shot technique requires no training and works with any text embedding model.
>
---
#### [new 037] CAuSE: Decoding Multimodal Classifiers using Faithful Natural Language Explanation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于可解释人工智能任务，旨在解决多模态分类器缺乏直观且忠实解释的问题。作者提出CAuSE框架，通过模拟解释与因果干预生成忠于模型决策的自然语言解释，并验证其在多模型多数据集上的有效性与因果忠实性。**

- **链接: [https://arxiv.org/pdf/2512.06814v1](https://arxiv.org/pdf/2512.06814v1)**

> **作者:** Dibyanayan Bandyopadhyay; Soham Bhattacharjee; Mohammed Hasanuzzaman; Asif Ekbal
>
> **备注:** Accepted at Transactions of the Association for Computational Linguistics (TACL). Pre-MIT Press publication version
>
> **摘要:** Multimodal classifiers function as opaque black box models. While several techniques exist to interpret their predictions, very few of them are as intuitive and accessible as natural language explanations (NLEs). To build trust, such explanations must faithfully capture the classifier's internal decision making behavior, a property known as faithfulness. In this paper, we propose CAuSE (Causal Abstraction under Simulated Explanations), a novel framework to generate faithful NLEs for any pretrained multimodal classifier. We demonstrate that CAuSE generalizes across datasets and models through extensive empirical evaluations. Theoretically, we show that CAuSE, trained via interchange intervention, forms a causal abstraction of the underlying classifier. We further validate this through a redesigned metric for measuring causal faithfulness in multimodal settings. CAuSE surpasses other methods on this metric, with qualitative analysis reinforcing its advantages. We perform detailed error analysis to pinpoint the failure cases of CAuSE. For replicability, we make the codes available at https://github.com/newcodevelop/CAuSE
>
---
#### [new 038] The Online Discourse of Virtual Reality and Anxiety
- **分类: cs.CL; stat.CO**

- **简介: 该论文属文本分析任务，旨在探究虚拟现实（VR）与焦虑相关的网络讨论。通过语料库语言学方法，利用Sketch Engine分析高频词及搭配，揭示公众对VR治疗焦虑的认知与关注点，为心理干预技术发展提供参考。**

- **链接: [https://arxiv.org/pdf/2512.06656v1](https://arxiv.org/pdf/2512.06656v1)**

> **作者:** Kwabena Yamoah; Cass Dykeman
>
> **备注:** Three tables and two figures. Unfortunately, I did not formally register the dataset prior to conducting the analysis
>
> **摘要:** VR in the treatment of clinical concerns such as generalized anxiety disorder or social anxiety. VR has created additional pathways to support patient well-being and care. Understanding online discussion of what users think about this technology may further support its efficacy. The purpose of this study was to employ a corpus linguistic methodology to identify the words and word networks that shed light on the online discussion of virtual reality and anxiety. Using corpus linguistics, frequently used words in discussion along with collocation were identified by utilizing Sketch Engine software. The results of the study, based upon the English Trends corpus, identified VR, Oculus, and headset as the most frequently discussed within the VR and anxiety subcorpus. These results point to the development of the virtual system, along with the physical apparatus that makes viewing and engaging with the virtual environment possible. Additional results point to collocation of prepositional phrases such as of virtual reality, in virtual reality, and for virtual reality relating to the design, experience, and development, respectively. These findings offer new perspective on how VR and anxiety together are discussed in general discourse and offer pathways for future opportunities to support counseling needs through development and accessibility. Keywords: anxiety disorders, corpus linguistics, Sketch Engine, and virtual reality VR
>
---
#### [new 039] SPAD: Seven-Source Token Probability Attribution with Syntactic Aggregation for Detecting Hallucinations in RAG
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于RAG中的幻觉检测任务，旨在解决现有方法忽略生成过程中多组件影响的问题。作者提出SPAD，将token概率归因为七个来源，并结合句法聚合，通过异常模式识别幻觉，提升了检测性能。**

- **链接: [https://arxiv.org/pdf/2512.07515v1](https://arxiv.org/pdf/2512.07515v1)**

> **作者:** Pengqian Lu; Jie Lu; Anjin Liu; Guangquan Zhang
>
> **摘要:** Detecting hallucinations in Retrieval-Augmented Generation (RAG) remains a challenge. Prior approaches attribute hallucinations to a binary conflict between internal knowledge (stored in FFNs) and retrieved context. However, this perspective is incomplete, failing to account for the impact of other components in the generative process, such as the user query, previously generated tokens, the current token itself, and the final LayerNorm adjustment. To address this, we introduce SPAD. First, we mathematically attribute each token's probability into seven distinct sources: Query, RAG, Past, Current Token, FFN, Final LayerNorm, and Initial Embedding. This attribution quantifies how each source contributes to the generation of the current token. Then, we aggregate these scores by POS tags to quantify how different components drive specific linguistic categories. By identifying anomalies, such as Nouns relying on Final LayerNorm, SPAD effectively detects hallucinations. Extensive experiments demonstrate that SPAD achieves state-of-the-art performance
>
---
#### [new 040] Mary, the Cheeseburger-Eating Vegetarian: Do LLMs Recognize Incoherence in Narratives?
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLM）是否能识别叙事中的不连贯性。通过配对叙事数据集，发现LLM内部可识别不连贯，但输出响应时无法有效区分，尤其对角色特质违背敏感度低于场景违背，揭示其依赖世界知识多于深层叙事理解。**

- **链接: [https://arxiv.org/pdf/2512.07777v1](https://arxiv.org/pdf/2512.07777v1)**

> **作者:** Karin de Langis; Püren Öncel; Ryan Peters; Andrew Elfenbein; Laura Kristen Allen; Andreas Schramm; Dongyeop Kang
>
> **摘要:** Leveraging a dataset of paired narratives, we investigate the extent to which large language models (LLMs) can reliably separate incoherent and coherent stories. A probing study finds that LLMs' internal representations can reliably identify incoherent narratives. However, LLMs generate responses to rating questions that fail to satisfactorily separate the coherent and incoherent narratives across several prompt variations, hinting at a gap in LLM's understanding of storytelling. The reasoning LLMs tested do not eliminate these deficits, indicating that thought strings may not be able to fully address the discrepancy between model internal state and behavior. Additionally, we find that LLMs appear to be more sensitive to incoherence resulting from an event that violates the setting (e.g., a rainy day in the desert) than to incoherence arising from a character violating an established trait (e.g., Mary, a vegetarian, later orders a cheeseburger), suggesting that LLMs may rely more on prototypical world knowledge than building meaning-based narrative coherence. The consistent asymmetry found in our results suggests that LLMs do not have a complete grasp on narrative coherence.
>
---
#### [new 041] Prompting-in-a-Series: Psychology-Informed Contents and Embeddings for Personality Recognition With Decoder-Only Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属自然语言处理任务，旨在提升基于大语言模型的个性识别。提出PICEPR算法，通过心理学启发的内容与嵌入双管道，利用解码器-only模型生成个性特征内容，显著提升识别性能5-15%。**

- **链接: [https://arxiv.org/pdf/2512.06991v1](https://arxiv.org/pdf/2512.06991v1)**

> **作者:** Jing Jie Tan; Ban-Hoe Kwan; Danny Wee-Kiat Ng; Yan-Chai Hum; Anissa Mokraoui; Shih-Yu Lo
>
> **备注:** 16 pages
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across various natural language processing tasks. This research introduces a novel "Prompting-in-a-Series" algorithm, termed PICEPR (Psychology-Informed Contents Embeddings for Personality Recognition), featuring two pipelines: (a) Contents and (b) Embeddings. The approach demonstrates how a modularised decoder-only LLM can summarize or generate content, which can aid in classifying or enhancing personality recognition functions as a personality feature extractor and a generator for personality-rich content. We conducted various experiments to provide evidence to justify the rationale behind the PICEPR algorithm. Meanwhile, we also explored closed-source models such as \textit{gpt4o} from OpenAI and \textit{gemini} from Google, along with open-source models like \textit{mistral} from Mistral AI, to compare the quality of the generated content. The PICEPR algorithm has achieved a new state-of-the-art performance for personality recognition by 5-15\% improvement. The work repository and models' weight can be found at https://research.jingjietan.com/?q=PICEPR.
>
---
#### [new 042] Replicating TEMPEST at Scale: Multi-Turn Adversarial Attacks Against Trillion-Parameter Frontier Models
- **分类: cs.CL**

- **简介: 该论文属于安全对齐任务，旨在评估大规模语言模型对抗多轮攻击的鲁棒性。通过TEMPEST框架测试十款前沿模型，发现模型规模不影响安全性，而推理模式可显著提升防御能力。**

- **链接: [https://arxiv.org/pdf/2512.07059v1](https://arxiv.org/pdf/2512.07059v1)**

> **作者:** Richard Young
>
> **备注:** 30 pages, 11 figures, 5 tables. Code and data: https://github.com/ricyoung/tempest-replication
>
> **摘要:** Despite substantial investment in safety alignment, the vulnerability of large language models to sophisticated multi-turn adversarial attacks remains poorly characterized, and whether model scale or inference mode affects robustness is unknown. This study employed the TEMPEST multi-turn attack framework to evaluate ten frontier models from eight vendors across 1,000 harmful behaviors, generating over 97,000 API queries across adversarial conversations with automated evaluation by independent safety classifiers. Results demonstrated a spectrum of vulnerability: six models achieved 96% to 100% attack success rate (ASR), while four showed meaningful resistance, with ASR ranging from 42% to 78%; enabling extended reasoning on identical architecture reduced ASR from 97% to 42%. These findings indicate that safety alignment quality varies substantially across vendors, that model scale does not predict adversarial robustness, and that thinking mode provides a deployable safety enhancement. Collectively, this work establishes that current alignment techniques remain fundamentally vulnerable to adaptive multi-turn attacks regardless of model scale, while identifying deliberative inference as a promising defense direction.
>
---
#### [new 043] A Patient-Doctor-NLP-System to contest inequality for less privileged
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗NLP任务，旨在解决资源受限环境下视障患者及低资源语言（如印地语）使用者获取医疗服务不平等问题。提出PDFTEMRA模型，通过模型压缩与优化技术，在降低计算成本的同时保持良好性能，适用于农村等低资源场景的医学问答系统。**

- **链接: [https://arxiv.org/pdf/2512.06734v1](https://arxiv.org/pdf/2512.06734v1)**

> **作者:** Subrit Dikshit; Ritu Tiwari; Priyank Jain
>
> **备注:** 19 pages, 6 figures
>
> **摘要:** Transfer Learning (TL) has accelerated the rapid development and availability of large language models (LLMs) for mainstream natural language processing (NLP) use cases. However, training and deploying such gigantic LLMs in resource-constrained, real-world healthcare situations remains challenging. This study addresses the limited support available to visually impaired users and speakers of low-resource languages such as Hindi who require medical assistance in rural environments. We propose PDFTEMRA (Performant Distilled Frequency Transformer Ensemble Model with Random Activations), a compact transformer-based architecture that integrates model distillation, frequency-domain modulation, ensemble learning, and randomized activation patterns to reduce computational cost while preserving language understanding performance. The model is trained and evaluated on medical question-answering and consultation datasets tailored to Hindi and accessibility scenarios, and its performance is compared against standard NLP state-of-the-art model baselines. Results demonstrate that PDFTEMRA achieves comparable performance with substantially lower computational requirements, indicating its suitability for accessible, inclusive, low-resource medical NLP applications.
>
---
#### [new 044] Investigating Training and Generalization in Faithful Self-Explanations of Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究大语言模型自解释的忠实性提升方法，旨在解决其自我解释常不忠实于实际推理的问题。通过构造伪忠实解释并进行持续学习，验证了训练可提升多种任务和风格下的解释忠实性，并具有跨风格和任务的泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.07288v1](https://arxiv.org/pdf/2512.07288v1)**

> **作者:** Tomoki Doi; Masaru Isonuma; Hitomi Yanaka
>
> **备注:** To appear in the Proceedings of the Asia-Pacific Chapter of the Association for Computational Linguistics: Student Research Workshop (AACL-SRW 2025)
>
> **摘要:** Large language models have the potential to generate explanations for their own predictions in a variety of styles based on user instructions. Recent research has examined whether these self-explanations faithfully reflect the models' actual behavior and has found that they often lack faithfulness. However, the question of how to improve faithfulness remains underexplored. Moreover, because different explanation styles have superficially distinct characteristics, it is unclear whether improvements observed in one style also arise when using other styles. This study analyzes the effects of training for faithful self-explanations and the extent to which these effects generalize, using three classification tasks and three explanation styles. We construct one-word constrained explanations that are likely to be faithful using a feature attribution method, and use these pseudo-faithful self-explanations for continual learning on instruction-tuned models. Our experiments demonstrate that training can improve self-explanation faithfulness across all classification tasks and explanation styles, and that these improvements also show signs of generalization to the multi-word settings and to unseen tasks. Furthermore, we find consistent cross-style generalization among three styles, suggesting that training may contribute to a broader improvement in faithful self-explanation ability.
>
---
#### [new 045] A Simple Method to Enhance Pre-trained Language Models with Speech Tokens for Classification
- **分类: cs.CL; cs.MM**

- **简介: 该论文研究多模态分类任务，旨在解决语音序列过长导致难以融合到语言模型的问题。提出用Lasso选择重要语音token，通过自监督学习适配语言模型，提升分类性能，在论辩谬误检测任务中达到SOTA。**

- **链接: [https://arxiv.org/pdf/2512.07571v1](https://arxiv.org/pdf/2512.07571v1)**

> **作者:** Nicolas Calbucura; Valentin Barriere
>
> **摘要:** This paper presents a simple method that allows to easily enhance textual pre-trained large language models with speech information, when fine-tuned for a specific classification task. A classical issue with the fusion of many embeddings from audio with text is the large length of the audio sequence compared to the text one. Our method benefits from an existing speech tokenizer trained for Audio Speech Recognition that output long sequences of tokens from a large vocabulary, making it difficult to integrate it at low cost in a large language model. By applying a simple lasso-based feature selection on multimodal Bag-of-Words representation, we retain only the most important audio tokens for the task, and adapt the language model to them with a self-supervised language modeling objective, before fine-tuning it on the downstream task. We show this helps to improve the performances compared to an unimodal model, to a bigger SpeechLM or to integrating audio via a learned representation. We show the effectiveness of our method on two recent Argumentative Fallacy Detection and Classification tasks where the use of audio was believed counterproductive, reaching state-of-the-art results. We also provide an in-depth analysis of the method, showing that even a random audio token selection helps enhancing the unimodal model. Our code is available [online](https://github.com/salocinc/EACL26SpeechTokFallacy/).
>
---
#### [new 046] Leveraging KV Similarity for Online Structured Pruning in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型推理加速中现有剪枝方法因依赖离线数据导致的不稳定性问题。作者提出Token Filtering，通过在线计算键值相似性进行结构化剪枝，并设计方差感知融合策略，提升剪枝稳定性和准确性。**

- **链接: [https://arxiv.org/pdf/2512.07090v1](https://arxiv.org/pdf/2512.07090v1)**

> **作者:** Jungmin Lee; Gwangeun Byeon; Yulhwa Kim; Seokin Hong
>
> **摘要:** Pruning has emerged as a promising direction for accelerating large language model (LLM) inference, yet existing approaches often suffer from instability because they rely on offline calibration data that may not generalize across inputs. In this work, we introduce Token Filtering, a lightweight online structured pruning technique that makes pruning decisions directly during inference without any calibration data. The key idea is to measure token redundancy via joint key-value similarity and skip redundant attention computations, thereby reducing inference cost while preserving critical information. To further enhance stability, we design a variance-aware fusion strategy that adaptively weights key and value similarity across heads, ensuring that informative tokens are retained even under high pruning ratios. This design introduces no additional memory overhead and provides a more reliable criterion for token importance. Extensive experiments on LLaMA-2 (7B/13B), LLaMA-3 (8B), and Mistral (7B) demonstrate that Token Filtering consistently outperforms prior structured pruning methods, preserving accuracy on commonsense reasoning benchmarks and maintaining strong performance on challenging tasks such as MMLU, even with 50% pruning.
>
---
#### [new 047] Multilingual corpora for the study of new concepts in the social sciences and humanities:
- **分类: cs.CL**

- **简介: 该论文提出一种构建多语言语料库的混合方法，旨在研究人文与社会科学中的新兴概念（如“非技术创新”）。通过采集企业网站文本和年报，经清洗、过滤、标注后生成带主题标签的上下文数据，支持词义变异分析和面向机器学习的文本分类任务。**

- **链接: [https://arxiv.org/pdf/2512.07367v1](https://arxiv.org/pdf/2512.07367v1)**

> **作者:** Revekka Kyriakoglou; Anna Pappa
>
> **备注:** in French language
>
> **摘要:** This article presents a hybrid methodology for building a multilingual corpus designed to support the study of emerging concepts in the humanities and social sciences (HSS), illustrated here through the case of ``non-technological innovation''. The corpus relies on two complementary sources: (1) textual content automatically extracted from company websites, cleaned for French and English, and (2) annual reports collected and automatically filtered according to documentary criteria (year, format, duplication). The processing pipeline includes automatic language detection, filtering of non-relevant content, extraction of relevant segments, and enrichment with structural metadata. From this initial corpus, a derived dataset in English is created for machine learning purposes. For each occurrence of a term from the expert lexicon, a contextual block of five sentences is extracted (two preceding and two following the sentence containing the term). Each occurrence is annotated with the thematic category associated with the term, enabling the construction of data suitable for supervised classification tasks. This approach results in a reproducible and extensible resource, suitable both for analyzing lexical variability around emerging concepts and for generating datasets dedicated to natural language processing applications.
>
---
#### [new 048] LIME: Making LLM Data More Efficient with Linguistic Metadata Embeddings
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LIME方法，通过融合语言学元数据增强词元嵌入，提升大模型预训练效率与生成能力。解决数据利用效率低的问题，引入极少参数即实现更快收敛与更强性能，并推出可引导生成的变体LIME+1。**

- **链接: [https://arxiv.org/pdf/2512.07522v1](https://arxiv.org/pdf/2512.07522v1)**

> **作者:** Sebastian Sztwiertnia; Felix Friedrich; Kristian Kersting; Patrick Schramowski; Björn Deiseroth
>
> **摘要:** Pre-training decoder-only language models relies on vast amounts of high-quality data, yet the availability of such data is increasingly reaching its limits. While metadata is commonly used to create and curate these datasets, its potential as a direct training signal remains under-explored. We challenge this status quo and propose LIME (Linguistic Metadata Embeddings), a method that enriches token embeddings with metadata capturing syntax, semantics, and contextual properties. LIME substantially improves pre-training efficiency. Specifically, it adapts up to 56% faster to the training data distribution, while introducing only 0.01% additional parameters at negligible compute overhead. Beyond efficiency, LIME improves tokenization, leading to remarkably stronger language modeling capabilities and generative task performance. These benefits persist across model scales (500M to 2B). In addition, we develop a variant with shifted metadata, LIME+1, that can guide token generation. Given prior metadata for the next token, LIME+1 improves reasoning performance by up to 38% and arithmetic accuracy by up to 35%.
>
---
#### [new 049] On the Interplay of Pre-Training, Mid-Training, and RL on Reasoning Language Models
- **分类: cs.CL**

- **简介: 该论文研究预训练、中段训练与强化学习对推理语言模型的影响，旨在厘清三者在推理能力提升中的作用。通过构建可控实验框架，分析不同训练阶段对泛化能力和推理效果的贡献，揭示RL增益条件、上下文迁移需求及中段训练的关键作用。**

- **链接: [https://arxiv.org/pdf/2512.07783v1](https://arxiv.org/pdf/2512.07783v1)**

> **作者:** Charlie Zhang; Graham Neubig; Xiang Yue
>
> **摘要:** Recent reinforcement learning (RL) techniques have yielded impressive reasoning improvements in language models, yet it remains unclear whether post-training truly extends a model's reasoning ability beyond what it acquires during pre-training. A central challenge is the lack of control in modern training pipelines: large-scale pre-training corpora are opaque, mid-training is often underexamined, and RL objectives interact with unknown prior knowledge in complex ways. To resolve this ambiguity, we develop a fully controlled experimental framework that isolates the causal contributions of pre-training, mid-training, and RL-based post-training. Our approach employs synthetic reasoning tasks with explicit atomic operations, parseable step-by-step reasoning traces, and systematic manipulation of training distributions. We evaluate models along two axes: extrapolative generalization to more complex compositions and contextual generalization across surface contexts. Using this framework, we reconcile competing views on RL's effectiveness. We show that: 1) RL produces true capability gains (pass@128) only when pre-training leaves sufficient headroom and when RL data target the model's edge of competence, tasks at the boundary that are difficult but not yet out of reach. 2) Contextual generalization requires minimal yet sufficient pre-training exposure, after which RL can reliably transfer. 3) Mid-training significantly enhances performance under fixed compute compared with RL only, demonstrating its central but underexplored role in training pipelines. 4) Process-level rewards reduce reward hacking and improve reasoning fidelity. Together, these results clarify the interplay between pre-training, mid-training, and RL, offering a foundation for understanding and improving reasoning LM training strategies.
>
---
#### [new 050] Collaborative Causal Sensemaking: Closing the Complementarity Gap in Human-AI Decision Support
- **分类: cs.CL; cs.AI; cs.HC; cs.LG**

- **简介: 该论文属于人机协同决策任务，旨在解决当前AI在高风险决策中难以与专家互补的问题。提出“协作因果感知”（CCS）框架，将AI视为认知伙伴，共同构建、检验因果模型，提升团队整体决策能力。**

- **链接: [https://arxiv.org/pdf/2512.07801v1](https://arxiv.org/pdf/2512.07801v1)**

> **作者:** Raunak Jain; Mudita Khurana
>
> **摘要:** LLM-based agents are rapidly being plugged into expert decision-support, yet in messy, high-stakes settings they rarely make the team smarter: human-AI teams often underperform the best individual, experts oscillate between verification loops and over-reliance, and the promised complementarity does not materialise. We argue this is not just a matter of accuracy, but a fundamental gap in how we conceive AI assistance: expert decisions are made through collaborative cognitive processes where mental models, goals, and constraints are continually co-constructed, tested, and revised between human and AI. We propose Collaborative Causal Sensemaking (CCS) as a research agenda and organizing framework for decision-support agents: systems designed as partners in cognitive work, maintaining evolving models of how particular experts reason, helping articulate and revise goals, co-constructing and stress-testing causal hypotheses, and learning from the outcomes of joint decisions so that both human and agent improve over time. We sketch challenges around training ecologies that make collaborative thinking instrumentally valuable, representations and interaction protocols for co-authored models, and evaluation centred on trust and complementarity. These directions can reframe MAS research around agents that participate in collaborative sensemaking and act as AI teammates that think with their human partners.
>
---
#### [new 051] ProSocialAlign: Preference Conditioned Test Time Alignment in Language Models
- **分类: cs.CL**

- **简介: 该论文属于语言模型安全对齐任务，旨在解决高风险场景下模型回应的安全与共情问题。提出ProSocialAlign框架，通过测试时参数高效方法，在不解冻原模型的情况下实现危害规避与价值观对齐的分层生成控制。**

- **链接: [https://arxiv.org/pdf/2512.06515v1](https://arxiv.org/pdf/2512.06515v1)**

> **作者:** Somnath Banerjee; Sayan Layek; Sayantan Adak; Mykola Pechenizkiy; Animesh Mukherjee; Rima Hazra
>
> **摘要:** Current language model safety paradigms often fall short in emotionally charged or high-stakes settings, where refusal-only approaches may alienate users and naive compliance can amplify risk. We propose ProSocialAlign, a test-time, parameter-efficient framework that steers generation toward safe, empathetic, and value-aligned responses without retraining the base model. We formalize five human-centered objectives and cast safety as lexicographic constrained generation: first, applying hard constraints to eliminate harmful continuations; then optimizing for prosocial quality within the safe set. Our method combines (i) directional regulation, a harm-mitigation mechanism that subtracts a learned "harm vector" in parameter space, and (ii) preference-aware autoregressive reward modeling trained jointly across attributes with gradient conflict resolution, enabling fine-grained, user-controllable decoding. Empirical evaluations across five safety benchmarks demonstrate state-of-the-art performance, reducing unsafe leakage and boosting alignment to human values, with strong gains across multiple evaluation metrics. ProSocialAlign offers a robust and modular foundation for generating context-sensitive, safe, and human-aligned responses at inference time.
>
---
#### [new 052] Do Generalisation Results Generalise?
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大语言模型在分布外（OOD）数据上的泛化能力是否具有一致性。通过多OOD测试集评估并控制域内性能，分析不同模型（OLMo2、OPT）的泛化表现相关性，发现泛化结果不具普遍一致性，依赖具体模型和测试集选择。**

- **链接: [https://arxiv.org/pdf/2512.07832v1](https://arxiv.org/pdf/2512.07832v1)**

> **作者:** Matteo Boglioni; Andrea Sgobbi; Gabriel Tavernini; Francesco Rita; Marius Mosbach; Tiago Pimentel
>
> **摘要:** A large language model's (LLM's) out-of-distribution (OOD) generalisation ability is crucial to its deployment. Previous work assessing LLMs' generalisation performance, however, typically focuses on a single out-of-distribution dataset. This approach may fail to precisely evaluate the capabilities of the model, as the data shifts encountered once a model is deployed are much more diverse. In this work, we investigate whether OOD generalisation results generalise. More specifically, we evaluate a model's performance across multiple OOD testsets throughout a finetuning run; we then evaluate the partial correlation of performances across these testsets, regressing out in-domain performance. This allows us to assess how correlated are generalisation performances once in-domain performance is controlled for. Analysing OLMo2 and OPT, we observe no overarching trend in generalisation results: the existence of a positive or negative correlation between any two OOD testsets depends strongly on the specific choice of model analysed.
>
---
#### [new 053] MASim: Multilingual Agent-Based Simulation for Social Science
- **分类: cs.CL; cs.AI; cs.CY; cs.MA; cs.SI**

- **简介: 该论文提出MASim，首个支持多语言交互的基于智能体的社会模拟框架，旨在解决现有单语模拟无法建模跨语言社会互动的问题。工作包括构建多语言社会模拟框架与MAPS基准，实现公共 opinion 建模与媒体影响分析。**

- **链接: [https://arxiv.org/pdf/2512.07195v1](https://arxiv.org/pdf/2512.07195v1)**

> **作者:** Xuan Zhang; Wenxuan Zhang; Anxu Wang; See-Kiong Ng; Yang Deng
>
> **摘要:** Multi-agent role-playing has recently shown promise for studying social behavior with language agents, but existing simulations are mostly monolingual and fail to model cross-lingual interaction, an essential property of real societies. We introduce MASim, the first multilingual agent-based simulation framework that supports multi-turn interaction among generative agents with diverse sociolinguistic profiles. MASim offers two key analyses: (i) global public opinion modeling, by simulating how attitudes toward open-domain hypotheses evolve across languages and cultures, and (ii) media influence and information diffusion, via autonomous news agents that dynamically generate content and shape user behavior. To instantiate simulations, we construct the MAPS benchmark, which combines survey questions and demographic personas drawn from global population distributions. Experiments on calibration, sensitivity, consistency, and cultural case studies show that MASim reproduces sociocultural phenomena and highlights the importance of multilingual simulation for scalable, controlled computational social science.
>
---
#### [new 054] MoCoRP: Modeling Consistent Relations between Persona and Response for Persona-based Dialogue
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统任务，旨在解决 persona-based 对话中缺乏显式个性与回复关系的问题。作者提出 MoCoRP 框架，利用 NLI 专家建模个性与回复间的显式关系，并结合预训练模型提升对话的个性一致性和上下文相关性。**

- **链接: [https://arxiv.org/pdf/2512.07544v1](https://arxiv.org/pdf/2512.07544v1)**

> **作者:** Kyungro Lee; Dongha Choi; Hyunju Lee
>
> **备注:** 18 pages
>
> **摘要:** As dialogue systems become increasingly important across various domains, a key challenge in persona-based dialogue is generating engaging and context-specific interactions while ensuring the model acts with a coherent personality. However, existing persona-based dialogue datasets lack explicit relations between persona sentences and responses, which makes it difficult for models to effectively capture persona information. To address these issues, we propose MoCoRP (Modeling Consistent Relations between Persona and Response), a framework that incorporates explicit relations into language models. MoCoRP leverages an NLI expert to explicitly extract the NLI relations between persona sentences and responses, enabling the model to effectively incorporate appropriate persona information from the context into its responses. We applied this framework to pre-trained models like BART and further extended it to modern large language models (LLMs) through alignment tuning. Experimental results on the public datasets ConvAI2 and MPChat demonstrate that MoCoRP outperforms existing baselines, achieving superior persona consistency and engaging, context-aware dialogue generation. Furthermore, our model not only excels in quantitative metrics but also shows significant improvements in qualitative aspects. These results highlight the effectiveness of explicitly modeling persona-response relations in persona-based dialogue. The source codes of MoCoRP are available at https://github.com/DMCB-GIST/MoCoRP.
>
---
#### [new 055] Think-While-Generating: On-the-Fly Reasoning for Personalized Long-Form Generation
- **分类: cs.CL**

- **简介: 该论文研究个性化长文本生成任务，旨在解决现有方法难以动态捕捉用户隐式偏好的问题。作者提出FlyThinker框架，通过并行的推理模型在生成过程中动态提供细粒度推理指导，实现高效训练与推理，并提升个性化生成效果。**

- **链接: [https://arxiv.org/pdf/2512.06690v1](https://arxiv.org/pdf/2512.06690v1)**

> **作者:** Chengbing Wang; Yang Zhang; Wenjie Wang; Xiaoyan Zhao; Fuli Feng; Xiangnan He; Tat-Seng Chua
>
> **摘要:** Preference alignment has enabled large language models (LLMs) to better reflect human expectations, but current methods mostly optimize for population-level preferences, overlooking individual users. Personalization is essential, yet early approaches-such as prompt customization or fine-tuning-struggle to reason over implicit preferences, limiting real-world effectiveness. Recent "think-then-generate" methods address this by reasoning before response generation. However, they face challenges in long-form generation: their static one-shot reasoning must capture all relevant information for the full response generation, making learning difficult and limiting adaptability to evolving content. To address this issue, we propose FlyThinker, an efficient "think-while-generating" framework for personalized long-form generation. FlyThinker employs a separate reasoning model that generates latent token-level reasoning in parallel, which is fused into the generation model to dynamically guide response generation. This design enables reasoning and generation to run concurrently, ensuring inference efficiency. In addition, the reasoning model is designed to depend only on previous responses rather than its own prior outputs, which preserves training parallelism across different positions-allowing all reasoning tokens for training data to be produced in a single forward pass like standard LLM training, ensuring training efficiency. Extensive experiments on real-world benchmarks demonstrate that FlyThinker achieves better personalized generation while keeping training and inference efficiency.
>
---
#### [new 056] Native Parallel Reasoner: Reasoning in Parallelism via Self-Distilled Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出Native Parallel Reasoner（NPR），旨在让大语言模型自主发展真正的并行推理能力。通过自蒸馏渐进训练、并行感知策略优化和高效引擎设计，实现无需教师的并行推理，在多个基准上显著提升性能与推理速度。**

- **链接: [https://arxiv.org/pdf/2512.07461v1](https://arxiv.org/pdf/2512.07461v1)**

> **作者:** Tong Wu; Yang Liu; Jun Bai; Zixia Jia; Shuyi Zhang; Ziyong Lin; Yanting Wang; Song-Chun Zhu; Zilong Zheng
>
> **摘要:** We introduce Native Parallel Reasoner (NPR), a teacher-free framework that enables Large Language Models (LLMs) to self-evolve genuine parallel reasoning capabilities. NPR transforms the model from sequential emulation to native parallel cognition through three key innovations: 1) a self-distilled progressive training paradigm that transitions from ``cold-start'' format discovery to strict topological constraints without external supervision; 2) a novel Parallel-Aware Policy Optimization (PAPO) algorithm that optimizes branching policies directly within the execution graph, allowing the model to learn adaptive decomposition via trial and error; and 3) a robust NPR Engine that refactors memory management and flow control of SGLang to enable stable, large-scale parallel RL training. Across eight reasoning benchmarks, NPR trained on Qwen3-4B achieves performance gains of up to 24.5% and inference speedups up to 4.6x. Unlike prior baselines that often fall back to autoregressive decoding, NPR demonstrates 100% genuine parallel execution, establishing a new standard for self-evolving, efficient, and scalable agentic reasoning.
>
---
#### [new 057] Convergence of Outputs When Two Large Language Models Interact in a Multi-Agentic Setup
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究两个大语言模型在无外界输入的多智能体交互中输出的收敛现象。旨在探究其对话是否会趋向重复和一致。通过实验观察并用词汇与嵌入指标分析对话演化过程。**

- **链接: [https://arxiv.org/pdf/2512.06256v1](https://arxiv.org/pdf/2512.06256v1)**

> **作者:** Aniruddha Maiti; Satya Nimmagadda; Kartha Veerya Jammuladinne; Niladri Sengupta; Ananya Jana
>
> **备注:** accepted to LLM 2025
>
> **摘要:** In this work, we report what happens when two large language models respond to each other for many turns without any outside input in a multi-agent setup. The setup begins with a short seed sentence. After that, each model reads the other's output and generates a response. This continues for a fixed number of steps. We used Mistral Nemo Base 2407 and Llama 2 13B hf. We observed that most conversations start coherently but later fall into repetition. In many runs, a short phrase appears and repeats across turns. Once repetition begins, both models tend to produce similar output rather than introducing a new direction in the conversation. This leads to a loop where the same or similar text is produced repeatedly. We describe this behavior as a form of convergence. It occurs even though the models are large, trained separately, and not given any prompt instructions. To study this behavior, we apply lexical and embedding-based metrics to measure how far the conversation drifts from the initial seed and how similar the outputs of the two models becomes as the conversation progresses.
>
---
#### [new 058] LLM4SFC: Sequential Function Chart Generation via Large Language Models
- **分类: cs.CL**

- **简介: 该论文研究利用大语言模型生成可执行的SFC图形化PLC程序，解决传统方法生成代码不可执行的问题。提出LLM4SFC框架，通过结构化表示、微调与RAG对齐规范，结合约束解码，实现自然语言到SFC的可靠转换，在真实工业场景中成功率达75%-94%。**

- **链接: [https://arxiv.org/pdf/2512.06787v1](https://arxiv.org/pdf/2512.06787v1)**

> **作者:** Ofek Glick; Vladimir Tchuiev; Marah Ghoummaid; Michal Moshkovitz; Dotan Di-Castro
>
> **摘要:** While Large Language Models (LLMs) are increasingly used for synthesizing textual PLC programming languages like Structured Text (ST) code, other IEC 61131-3 standard graphical languages like Sequential Function Charts (SFCs) remain underexplored. Generating SFCs is challenging due to graphical nature and ST actions embedded within, which are not directly compatible with standard generation techniques, often leading to non-executable code that is incompatible with industrial tool-chains In this work, we introduce LLM4SFC, the first framework to receive natural-language descriptions of industrial workflows and provide executable SFCs. LLM4SFC is based on three components: (i) A reduced structured representation that captures essential topology and in-line ST and reduced textual verbosity; (ii) Fine-tuning and few-shot retrieval-augmented generation (RAG) for alignment with SFC programming conventions; and (iii) A structured generation approach that prunes illegal tokens in real-time to ensure compliance with the textual format of SFCs. We evaluate LLM4SFC on a dataset of real-world SFCs from automated manufacturing projects, using both open-source and proprietary LLMs. The results show that LLM4SFC reliably generates syntactically valid SFC programs effectively bridging graphical and textual PLC languages, achieving a generation generation success of 75% - 94%, paving the way for automated industrial programming.
>
---
#### [new 059] Automated Generation of Custom MedDRA Queries Using SafeTerm Medical Map
- **分类: cs.CL**

- **简介: 该论文针对药物安全审查中人工构建MedDRA查询效率低的问题，提出一种基于AI的系统SafeTerm，通过语义向量和聚类方法自动生成立项相关的MedDRA术语并排序，辅助提升信号检测效率。**

- **链接: [https://arxiv.org/pdf/2512.07694v1](https://arxiv.org/pdf/2512.07694v1)**

> **作者:** Francois Vandenhende; Anna Georgiou; Michalis Georgiou; Theodoros Psaras; Ellie Karekla; Elena Hadjicosta
>
> **备注:** 12 pages, 4 figures
>
> **摘要:** In pre-market drug safety review, grouping related adverse event terms into standardised MedDRA queries or the FDA Office of New Drugs Custom Medical Queries (OCMQs) is critical for signal detection. We present a novel quantitative artificial intelligence system that understands and processes medical terminology and automatically retrieves relevant MedDRA Preferred Terms (PTs) for a given input query, ranking them by a relevance score using multi-criteria statistical methods. The system (SafeTerm) embeds medical query terms and MedDRA PTs in a multidimensional vector space, then applies cosine similarity and extreme-value clustering to generate a ranked list of PTs. Validation was conducted against the FDA OCMQ v3.0 (104 queries), restricted to valid MedDRA PTs. Precision, recall and F1 were computed across similarity-thresholds. High recall (>95%) is achieved at moderate thresholds. Higher thresholds improve precision (up to 86%). The optimal threshold (~0.70 - 0.75) yielded recall ~50% and precision ~33%. Narrow-term PT subsets performed similarly but required slightly higher similarity thresholds. The SafeTerm AI-driven system provides a viable supplementary method for automated MedDRA query generation. A similarity threshold of ~0.60 is recommended initially, with increased thresholds for refined term selection.
>
---
#### [new 060] Classifying German Language Proficiency Levels Using Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究利用大语言模型自动分类德语学习者的CEFR语言水平。针对数据不足问题，构建了真实与合成数据结合的多源数据集，比较了提示工程、模型微调与基于神经激活的探针方法，提升了分类性能，验证了大模型在语言水平评估中的有效性。**

- **链接: [https://arxiv.org/pdf/2512.06483v1](https://arxiv.org/pdf/2512.06483v1)**

> **作者:** Elias-Leander Ahlers; Witold Brunsmann; Malte Schilling
>
> **备注:** Accepted at 3rd International Conference on Foundation and Large Language Models (FLLM2025), Vienna (Austria)
>
> **摘要:** Assessing language proficiency is essential for education, as it enables instruction tailored to learners needs. This paper investigates the use of Large Language Models (LLMs) for automatically classifying German texts according to the Common European Framework of Reference for Languages (CEFR) into different proficiency levels. To support robust training and evaluation, we construct a diverse dataset by combining multiple existing CEFR-annotated corpora with synthetic data. We then evaluate prompt-engineering strategies, fine-tuning of a LLaMA-3-8B-Instruct model and a probing-based approach that utilizes the internal neural state of the LLM for classification. Our results show a consistent performance improvement over prior methods, highlighting the potential of LLMs for reliable and scalable CEFR classification.
>
---
#### [new 061] An Analysis of Large Language Models for Simulating User Responses in Surveys
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLM）在模拟用户调查应答中的应用，旨在解决LLM因训练偏见难以代表多元人群的问题。作者比较了直接提示与思维链提示，并提出CLAIMSIM方法以提升观点多样性，发现现有方法在适应不同用户特征方面仍存在局限。**

- **链接: [https://arxiv.org/pdf/2512.06874v1](https://arxiv.org/pdf/2512.06874v1)**

> **作者:** Ziyun Yu; Yiru Zhou; Chen Zhao; Hongyi Wen
>
> **备注:** Accepted to IJCNLP-AACL 2025 (Main Conference)
>
> **摘要:** Using Large Language Models (LLMs) to simulate user opinions has received growing attention. Yet LLMs, especially trained with reinforcement learning from human feedback (RLHF), are known to exhibit biases toward dominant viewpoints, raising concerns about their ability to represent users from diverse demographic and cultural backgrounds. In this work, we examine the extent to which LLMs can simulate human responses to cross-domain survey questions through direct prompting and chain-of-thought prompting. We further propose a claim diversification method CLAIMSIM, which elicits viewpoints from LLM parametric knowledge as contextual input. Experiments on the survey question answering task indicate that, while CLAIMSIM produces more diverse responses, both approaches struggle to accurately simulate users. Further analysis reveals two key limitations: (1) LLMs tend to maintain fixed viewpoints across varying demographic features, and generate single-perspective claims; and (2) when presented with conflicting claims, LLMs struggle to reason over nuanced differences among demographic features, limiting their ability to adapt responses to specific user profiles.
>
---
#### [new 062] HalluShift++: Bridging Language and Vision through Internal Representation Shifts for Hierarchical Hallucinations in MLLMs
- **分类: cs.CL; cs.CV**

- **简介: 该论文属多模态大模型任务，旨在解决MLLM在图文理解中产生事实性幻觉的问题。作者提出HalluShift++，通过分析模型内部层动态的异常来检测幻觉，利用内部表征偏移实现更可靠的层级化幻觉识别，减少对外部评估模型的依赖。**

- **链接: [https://arxiv.org/pdf/2512.07687v1](https://arxiv.org/pdf/2512.07687v1)**

> **作者:** Sujoy Nath; Arkaprabha Basu; Sharanya Dasgupta; Swagatam Das
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in vision-language understanding tasks. While these models often produce linguistically coherent output, they often suffer from hallucinations, generating descriptions that are factually inconsistent with the visual content, potentially leading to adverse consequences. Therefore, the assessment of hallucinations in MLLM has become increasingly crucial in the model development process. Contemporary methodologies predominantly depend on external LLM evaluators, which are themselves susceptible to hallucinations and may present challenges in terms of domain adaptation. In this study, we propose the hypothesis that hallucination manifests as measurable irregularities within the internal layer dynamics of MLLMs, not merely due to distributional shifts but also in the context of layer-wise analysis of specific assumptions. By incorporating such modifications, \textsc{\textsc{HalluShift++}} broadens the efficacy of hallucination detection from text-based large language models (LLMs) to encompass multimodal scenarios. Our codebase is available at https://github.com/C0mRD/HalluShift_Plus.
>
---
#### [new 063] SETUP: Sentence-level English-To-Uniform Meaning Representation Parser
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的语义解析任务，旨在解决英语到统一语义表示（UMR）的自动转换问题。作者提出两种方法，其中SETUP模型通过改进现有解析器，显著提升了UMR解析的准确性。**

- **链接: [https://arxiv.org/pdf/2512.07068v1](https://arxiv.org/pdf/2512.07068v1)**

> **作者:** Emma Markle; Javier Gutierrez Bach; Shira Wein
>
> **摘要:** Uniform Meaning Representation (UMR) is a novel graph-based semantic representation which captures the core meaning of a text, with flexibility incorporated into the annotation schema such that the breadth of the world's languages can be annotated (including low-resource languages). While UMR shows promise in enabling language documentation, improving low-resource language technologies, and adding interpretability, the downstream applications of UMR can only be fully explored when text-to-UMR parsers enable the automatic large-scale production of accurate UMR graphs at test time. Prior work on text-to-UMR parsing is limited to date. In this paper, we introduce two methods for English text-to-UMR parsing, one of which fine-tunes existing parsers for Abstract Meaning Representation and the other, which leverages a converter from Universal Dependencies, using prior work as a baseline. Our best-performing model, which we call SETUP, achieves an AnCast score of 84 and a SMATCH++ score of 91, indicating substantial gains towards automatic UMR parsing.
>
---
#### [new 064] LOCUS: A System and Method for Low-Cost Customization for Universal Specialization
- **分类: cs.CL**

- **简介: 该论文提出LOCUS，面向低成本通用专业化，解决小样本下NLP模型定制问题。通过检索增强、合成数据与参数高效微调，实现在NER和TC任务上超越强基线（如GPT-4o）的效果，显著降低模型大小与内存占用。**

- **链接: [https://arxiv.org/pdf/2512.06239v1](https://arxiv.org/pdf/2512.06239v1)**

> **作者:** Dhanasekar Sundararaman; Keying Li; Wayne Xiong; Aashna Garg
>
> **摘要:** We present LOCUS (LOw-cost Customization for Universal Specialization), a pipeline that consumes few-shot data to streamline the construction and training of NLP models through targeted retrieval, synthetic data generation, and parameter-efficient tuning. With only a small number of labeled examples, LOCUS discovers pertinent data in a broad repository, synthesizes additional training samples via in-context data generation, and fine-tunes models using either full or low-rank (LoRA) parameter adaptation. Our approach targets named entity recognition (NER) and text classification (TC) benchmarks, consistently outperforming strong baselines (including GPT-4o) while substantially lowering costs and model sizes. Our resultant memory-optimized models retain 99% of fully fine-tuned accuracy while using barely 5% of the memory footprint, also beating GPT-4o on several benchmarks with less than 1% of its parameters.
>
---
#### [new 065] Minimum Bayes Risk Decoding for Error Span Detection in Reference-Free Automatic Machine Translation Evaluation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究无参考机器翻译自动评价中的错误片段检测任务，指出传统最大后验解码假设模型概率与人工标注相似度一致存在问题。作者引入最小贝叶斯风险解码，利用句子和片段级相似性选择更接近人工标注的预测结果，并通过MBR蒸馏降低计算开销，提升各层级性能。**

- **链接: [https://arxiv.org/pdf/2512.07540v1](https://arxiv.org/pdf/2512.07540v1)**

> **作者:** Boxuan Lyu; Haiyue Song; Hidetaka Kamigaito; Chenchen Ding; Hideki Tanaka; Masao Utiyama; Kotaro Funakoshi; Manabu Okumura
>
> **摘要:** Error Span Detection (ESD) is a subtask of automatic machine translation evaluation that localizes error spans in translations and labels their severity. State-of-the-art generative ESD methods typically decode using Maximum a Posteriori (MAP), assuming that model-estimated probabilities are perfectly correlated with similarity to human annotation. However, we observed that annotations dissimilar to the human annotation could achieve a higher model likelihood than the human annotation. We address this issue by applying Minimum Bayes Risk (MBR) decoding to generative ESD models. Specifically, we employ sentence- and span-level similarity metrics as utility functions to select candidate hypotheses based on their approximate similarity to the human annotation. Extensive experimental results show that our MBR decoding outperforms the MAP baseline at the system, sentence, and span-levels. Furthermore, to mitigate the computational cost of MBR decoding, we demonstrate that applying MBR distillation enables a standard greedy model to match MBR decoding performance, effectively eliminating the inference-time latency bottleneck.
>
---
#### [new 066] PCMind-2.1-Kaiyuan-2B Technical Report
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文聚焦开源大模型训练，旨在缩小开源与工业界在数据和训练方法上的差距。提出PCMind-2.1-Kaiyuan-2B模型，通过量化数据评估、选择性重复学习和多阶段课程学习，提升小规模模型在资源受限下的训练效率与性能。**

- **链接: [https://arxiv.org/pdf/2512.07612v1](https://arxiv.org/pdf/2512.07612v1)**

> **作者:** Kairong Luo; Zhenbo Sun; Xinyu Shi; Shengqi Chen; Bowen Yu; Yunyi Chen; Chenyi Dang; Hengtao Tao; Hui Wang; Fangming Liu; Kaifeng Lyu; Wenguang Chen
>
> **摘要:** The rapid advancement of Large Language Models (LLMs) has resulted in a significant knowledge gap between the open-source community and industry, primarily because the latter relies on closed-source, high-quality data and training recipes. To address this, we introduce PCMind-2.1-Kaiyuan-2B, a fully open-source 2-billion-parameter model focused on improving training efficiency and effectiveness under resource constraints. Our methodology includes three key innovations: a Quantile Data Benchmarking method for systematically comparing heterogeneous open-source datasets and providing insights on data mixing strategies; a Strategic Selective Repetition scheme within a multi-phase paradigm to effectively leverage sparse, high-quality data; and a Multi-Domain Curriculum Training policy that orders samples by quality. Supported by a highly optimized data preprocessing pipeline and architectural modifications for FP16 stability, Kaiyuan-2B achieves performance competitive with state-of-the-art fully open-source models, demonstrating practical and scalable solutions for resource-limited pretraining. We release all assets (including model weights, data, and code) under Apache 2.0 license at https://huggingface.co/thu-pacman/PCMind-2.1-Kaiyuan-2B.
>
---
#### [new 067] FVA-RAG: Falsification-Verification Alignment for Mitigating Sycophantic Hallucinations
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文针对RAG系统易受错误前提影响而产生“逢迎幻觉”的问题，提出FVA-RAG框架。通过引入对抗性检索策略生成“反证查询”，结合双验证机制，以证伪而非验证的方式提升事实准确性。**

- **链接: [https://arxiv.org/pdf/2512.07015v1](https://arxiv.org/pdf/2512.07015v1)**

> **作者:** Mayank Ravishankara
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems have significantly reduced hallucinations in Large Language Models (LLMs) by grounding responses in external context. However, standard RAG architectures suffer from a critical vulnerability: Retrieval Sycophancy. When presented with a query based on a false premise or a common misconception, vector-based retrievers tend to fetch documents that align with the user's bias rather than objective truth, leading the model to "hallucinate with citations." In this work, we introduce Falsification-Verification Alignment RAG (FVA-RAG), a framework that shifts the retrieval paradigm from Inductive Verification (seeking support) to Deductive Falsification (seeking disproof). Unlike existing "Self-Correction" methods that rely on internal consistency, FVA-RAG deploys a distinct Adversarial Retrieval Policy that actively generates "Kill Queries"-targeted search terms designed to surface contradictory evidence. We introduce a dual-verification mechanism that explicitly weighs the draft answer against this "Anti-Context." Preliminary experiments on a dataset of common misconceptions demonstrate that FVA-RAG significantly improves robustness against sycophantic hallucinations compared to standard RAG baselines, effectively acting as an inference-time "Red Team" for factual generation.
>
---
#### [new 068] Large Language Models and Forensic Linguistics: Navigating Opportunities and Threats in the Age of Generative AI
- **分类: cs.CL; cs.CY**

- **简介: 该论文探讨大语言模型对司法语言学的双重影响，旨在应对AI生成文本带来的鉴定挑战。研究分析现有检测技术的局限，提出需重构方法论以确保科学性与法律可采性，强调发展人机协同、可解释检测与跨群体验证机制。**

- **链接: [https://arxiv.org/pdf/2512.06922v1](https://arxiv.org/pdf/2512.06922v1)**

> **作者:** George Mikros
>
> **摘要:** Large language models (LLMs) present a dual challenge for forensic linguistics. They serve as powerful analytical tools enabling scalable corpus analysis and embedding-based authorship attribution, while simultaneously destabilising foundational assumptions about idiolect through style mimicry, authorship obfuscation, and the proliferation of synthetic texts. Recent stylometric research indicates that LLMs can approximate surface stylistic features yet exhibit detectable differences from human writers, a tension with significant forensic implications. However, current AI-text detection techniques, whether classifier-based, stylometric, or watermarking approaches, face substantial limitations: high false positive rates for non-native English writers and vulnerability to adversarial strategies such as homoglyph substitution. These uncertainties raise concerns under legal admissibility standards, particularly the Daubert and Kumho Tire frameworks. The article concludes that forensic linguistics requires methodological reconfiguration to remain scientifically credible and legally admissible. Proposed adaptations include hybrid human-AI workflows, explainable detection paradigms beyond binary classification, and validation regimes measuring error and bias across diverse populations. The discipline's core insight, i.e., that language reveals information about its producer, remains valid but must accommodate increasingly complex chains of human and machine authorship.
>
---
#### [new 069] NeSTR: A Neuro-Symbolic Abductive Framework for Temporal Reasoning in Large Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文针对大语言模型在复杂时序推理中的不足，提出NeSTR框架，结合符号表示与神经推理，通过显式时序建模和溯因反思修正错误，提升模型的时序理解能力，无需微调即可在多种时序问答任务上取得优越的零样本性能。**

- **链接: [https://arxiv.org/pdf/2512.07218v1](https://arxiv.org/pdf/2512.07218v1)**

> **作者:** Feng Liang; Weixin Zeng; Runhao Zhao; Xiang Zhao
>
> **备注:** Accepted by AAAI 2026
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable performance across a wide range of natural language processing tasks. However, temporal reasoning, particularly under complex temporal constraints, remains a major challenge. To this end, existing approaches have explored symbolic methods, which encode temporal structure explicitly, and reflective mechanisms, which revise reasoning errors through multi-step inference. Nonetheless, symbolic approaches often underutilize the reasoning capabilities of LLMs, while reflective methods typically lack structured temporal representations, which can result in inconsistent or hallucinated reasoning. As a result, even when the correct temporal context is available, LLMs may still misinterpret or misapply time-related information, leading to incomplete or inaccurate answers. To address these limitations, in this work, we propose Neuro-Symbolic Temporal Reasoning (NeSTR), a novel framework that integrates structured symbolic representations with hybrid reflective reasoning to enhance the temporal sensitivity of LLM inference. NeSTR preserves explicit temporal relations through symbolic encoding, enforces logical consistency via verification, and corrects flawed inferences using abductive reflection. Extensive experiments on diverse temporal question answering benchmarks demonstrate that NeSTR achieves superior zero-shot performance and consistently improves temporal reasoning without any fine-tuning, showcasing the advantage of neuro-symbolic integration in enhancing temporal understanding in large language models.
>
---
#### [new 070] Automated Data Enrichment using Confidence-Aware Fine-Grained Debate among Open-Source LLMs for Mental Health and Online Safety
- **分类: cs.CL; cs.LG**

- **简介: 该论文聚焦NLP中的数据标注难题，提出基于多开源大模型的置信度感知细粒度辩论（CFD）框架，用于自动数据增强。通过模拟人工标注，提升心理健康与网络安全隐患识别的下游任务性能。**

- **链接: [https://arxiv.org/pdf/2512.06227v1](https://arxiv.org/pdf/2512.06227v1)**

> **作者:** Junyu Mao; Anthony Hills; Talia Tseriotou; Maria Liakata; Aya Shamir; Dan Sayda; Dana Atzil-Slonim; Natalie Djohari; Arpan Mandal; Silke Roth; Pamela Ugwudike; Mahesan Niranjan; Stuart E. Middleton
>
> **摘要:** Real-world indicators are important for improving natural language processing (NLP) tasks such as life events for mental health analysis and risky behaviour for online safety, yet labelling such information in NLP training datasets is often costly and/or difficult given the dynamic nature of such events. This paper compares several LLM-based data enrichment methods and introduces a novel Confidence-Aware Fine-Grained Debate (CFD) framework in which multiple LLM agents simulate human annotators and exchange fine-grained evidence to reach consensus. We describe two new expert-annotated datasets, a mental health Reddit wellbeing dataset and an online safety Facebook sharenting risk dataset. Our CFD framework achieves the most robust data enrichment performance compared to a range of baselines and we show that this type of data enrichment consistently improves downstream tasks. Enriched features incorporated via debate transcripts yield the largest gains, outperforming the non-enriched baseline by 10.1% for the online safety task.
>
---
#### [new 071] GUMBridge: a Corpus for Varieties of Bridging Anaphora
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决英语桥接共指现象覆盖不足的问题。作者构建了GUMBridge语料库，涵盖16种体裁，提供细粒度桥接类型标注，并评估了标注质量与大模型在桥接任务上的基线性能。**

- **链接: [https://arxiv.org/pdf/2512.07134v1](https://arxiv.org/pdf/2512.07134v1)**

> **作者:** Lauren Levine; Amir Zeldes
>
> **摘要:** Bridging is an anaphoric phenomenon where the referent of an entity in a discourse is dependent on a previous, non-identical entity for interpretation, such as in "There is 'a house'. 'The door' is red," where the door is specifically understood to be the door of the aforementioned house. While there are several existing resources in English for bridging anaphora, most are small, provide limited coverage of the phenomenon, and/or provide limited genre coverage. In this paper, we introduce GUMBridge, a new resource for bridging, which includes 16 diverse genres of English, providing both broad coverage for the phenomenon and granular annotations for the subtype categorization of bridging varieties. We also present an evaluation of annotation quality and report on baseline performance using open and closed source contemporary LLMs on three tasks underlying our data, showing that bridging resolution and subtype classification remain difficult NLP tasks in the age of LLMs.
>
---
#### [new 072] DART: Leveraging Multi-Agent Disagreement for Tool Recruitment in Multimodal Reasoning
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文研究多模态推理中的工具调用问题，提出DART框架，利用多智能体分歧识别并调用视觉工具（如OCR、目标检测）来增强推理。通过工具缓解分歧，提升答案准确性，在多个基准上优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.07132v1](https://arxiv.org/pdf/2512.07132v1)**

> **作者:** Nithin Sivakumaran; Justin Chih-Yao Chen; David Wan; Yue Zhang; Jaehong Yoon; Elias Stengel-Eskin; Mohit Bansal
>
> **备注:** Code: https://github.com/nsivaku/dart
>
> **摘要:** Specialized visual tools can augment large language models or vision language models with expert knowledge (e.g., grounding, spatial reasoning, medical knowledge, etc.), but knowing which tools to call (and when to call them) can be challenging. We introduce DART, a multi-agent framework that uses disagreements between multiple debating visual agents to identify useful visual tools (e.g., object detection, OCR, spatial reasoning, etc.) that can resolve inter-agent disagreement. These tools allow for fruitful multi-agent discussion by introducing new information, and by providing tool-aligned agreement scores that highlight agents in agreement with expert tools, thereby facilitating discussion. We utilize an aggregator agent to select the best answer by providing the agent outputs and tool information. We test DART on four diverse benchmarks and show that our approach improves over multi-agent debate as well as over single agent tool-calling frameworks, beating the next-strongest baseline (multi-agent debate with a judge model) by 3.4% and 2.4% on A-OKVQA and MMMU respectively. We also find that DART adapts well to new tools in applied domains, with a 1.3% improvement on the M3D medical dataset over other strong tool-calling, single agent, and multi-agent baselines. Additionally, we measure text overlap across rounds to highlight the rich discussion in DART compared to existing multi-agent methods. Finally, we study the tool call distribution, finding that diverse tools are reliably used to help resolve disagreement.
>
---
#### [new 073] Metric-Fair Prompting: Treating Similar Samples Similarly
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对医疗选择题回答任务，提出“度量公平提示”框架，通过计算问题相似性并联合推理，确保相似问题获得一致答案。方法引入基于嵌入的相似性约束和置信度评分，提升个体公平性与模型准确率。**

- **链接: [https://arxiv.org/pdf/2512.07608v1](https://arxiv.org/pdf/2512.07608v1)**

> **作者:** Jing Wang; Jie Shen; Xing Niu; Tong Zhang; Jeremy Weiss
>
> **摘要:** We introduce \emph{Metric-Fair Prompting}, a fairness-aware prompting framework that guides large language models (LLMs) to make decisions under metric-fairness constraints. In the application of multiple-choice medical question answering, each {(question, option)} pair is treated as a binary instance with label $+1$ (correct) or $-1$ (incorrect). To promote {individual fairness}~--~treating similar instances similarly~--~we compute question similarity using NLP embeddings and solve items in \emph{joint pairs of similar questions} rather than in isolation. The prompt enforces a global decision protocol: extract decisive clinical features, map each \((\text{question}, \text{option})\) to a score $f(x)$ that acts as confidence, and impose a Lipschitz-style constraint so that similar inputs receive similar scores and, hence, consistent outputs. Evaluated on the {MedQA (US)} benchmark, Metric-Fair Prompting is shown to improve performance over standard single-item prompting, demonstrating that fairness-guided, confidence-oriented reasoning can enhance LLM accuracy on high-stakes clinical multiple-choice questions.
>
---
#### [new 074] Empathy by Design: Aligning Large Language Models for Healthcare Dialogue
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI医疗对话系统优化任务，旨在解决通用大模型在医疗场景中事实错误和缺乏共情的问题。作者提出基于直接偏好优化（DPO）的对齐框架，提升模型的事实准确性、语义连贯性及共情表达能力，经实验验证优于基线与商用系统。**

- **链接: [https://arxiv.org/pdf/2512.06097v1](https://arxiv.org/pdf/2512.06097v1)**

> **作者:** Emre Umucu; Guillermina Solis; Leon Garza; Emilia Rivas; Beatrice Lee; Anantaa Kotal; Aritran Piplai
>
> **摘要:** General-purpose large language models (LLMs) have demonstrated remarkable generative and reasoning capabilities but remain limited in healthcare and caregiving applications due to two key deficiencies: factual unreliability and a lack of empathetic communication. These shortcomings pose significant risks in sensitive contexts where users, particularly non-professionals and caregivers, seek medically relevant guidance or emotional reassurance. To address these challenges, we introduce a Direct Preference Optimization (DPO)-based alignment framework designed to improve factual correctness, semantic coherence, and human-centric qualities such as empathy, politeness, and simplicity in caregiver-patient dialogues. Our approach fine-tunes domain-adapted LLMs using pairwise preference data, where preferred responses reflect supportive and accessible communication styles while rejected ones represent prescriptive or overly technical tones. This direct optimization method aligns model outputs with human preferences more efficiently than traditional reinforcement-learning-based alignment. Empirical evaluations across multiple open and proprietary LLMs show that our DPO-tuned models achieve higher semantic alignment, improved factual accuracy, and stronger human-centric evaluation scores compared to baseline and commercial alternatives such as Google medical dialogue systems. These improvements demonstrate that preference-based alignment offers a scalable and transparent pathway toward developing trustworthy, empathetic, and clinically informed AI assistants for caregiver and healthcare communication. Our open-source code is available at: https://github.com/LeonG19/Empathy-by-Design
>
---
#### [new 075] AI-Generated Compromises for Coalition Formation: Modeling, Simulation, and a Textual Case Study
- **分类: cs.MA; cs.CL; cs.GT**

- **简介: 该论文研究基于AI的妥协提案生成，用于多智能体联盟形成。针对文本协作场景（如共拟宪法），结合NLP与大模型构建语义空间，建模个体偏好与不确定性，提出算法生成多数支持的折中方案，提升民主文本编辑效率。**

- **链接: [https://arxiv.org/pdf/2512.05983v1](https://arxiv.org/pdf/2512.05983v1)**

> **作者:** Eyal Briman; Ehud Shapiro; Nimrod Talmon
>
> **备注:** In Proceedings TARK 2025, arXiv:2511.20540. arXiv admin note: substantial text overlap with arXiv:2506.06837
>
> **摘要:** The challenge of finding compromises between agent proposals is fundamental to AI sub-fields such as argumentation, mediation, and negotiation. Building on this tradition, Elkind et al. (2021) introduced a process for coalition formation that seeks majority-supported proposals preferable to the status quo, using a metric space where each agent has an ideal point. The crucial step in this iterative process involves identifying compromise proposals around which agent coalitions can unite. How to effectively find such compromise proposals, however, remains an open question. We address this gap by formalizing a holistic model that encompasses agent bounded rationality and uncertainty and developing AI models to generate such compromise proposals. We focus on the domain of collaboratively writing text documents -- e.g., to enable the democratic creation of a community constitution. We apply NLP (Natural Language Processing) techniques and utilize LLMs (Large Language Models) to create a semantic metric space for text and develop algorithms to suggest suitable compromise points. To evaluate the effectiveness of our algorithms, we simulate various coalition formation processes and demonstrate the potential of AI to facilitate large-scale democratic text editing, such as collaboratively drafting a constitution, an area where traditional tools are limited.
>
---
#### [new 076] A Fast and Effective Solution to the Problem of Look-ahead Bias in LLMs
- **分类: cs.LG; cs.CL**

- **简介: 该论文属金融预测任务，旨在解决大模型因训练数据包含未来信息而导致的前视偏差问题。作者提出一种低成本推理时干预方法，通过两个小模型调整大模型logits，分别遗忘或保留特定知识，有效消除偏差并优于现有方法。**

- **链接: [https://arxiv.org/pdf/2512.06607v1](https://arxiv.org/pdf/2512.06607v1)**

> **作者:** Humzah Merchant; Bradford Levy
>
> **摘要:** Applying LLMs to predictive tasks in finance is challenging due to look-ahead bias resulting from their training on long time-series data. This precludes the backtests typically employed in finance since retraining frontier models from scratch with a specific knowledge cutoff is prohibitive. In this paper, we introduce a fast, effective, and low-cost alternative. Our method guides generation at inference time by adjusting the logits of a large base model using a pair of smaller, specialized models -- one fine-tuned on information to be forgotten and another on information to be retained. We demonstrate that our method effectively removes both verbatim and semantic knowledge, corrects biases, and outperforms prior methods.
>
---
#### [new 077] Small Language Models Reshape Higher Education: Courses, Textbooks, and Teaching
- **分类: physics.ed-ph; cs.CL**

- **简介: 该论文聚焦AI赋能教育，旨在解决大模型在高等教育中准确性低、资源消耗大的问题。通过构建大气物理领域的专业语料与图像库，利用小语言模型（MiniLM）实现精准检索，重构课程体系、教材形态与教学方式，推动主动认知学习。**

- **链接: [https://arxiv.org/pdf/2512.06001v1](https://arxiv.org/pdf/2512.06001v1)**

> **作者:** Jian Zhang; Jia Shao
>
> **备注:** in Chinese language
>
> **摘要:** While large language models (LLMs) have introduced novel paradigms in science and education, their adoption in higher education is constrained by inherent limitations. These include a tendency to produce inaccuracies and high computational requirements, which compromise the strict demands for accurate and reliable knowledge essential in higher education. Small language models (MiniLMs), by contrast, offer distinct advantages in professional education due to their lightweight nature and precise retrieval capabilities. This research takes "Atmospheric Physics" as an example. We established a specialized corpus and image repository by gathering over 550,000 full-text PDFs from over 130 international well-respected journals in Earth and environmental science. From this collection, we extracted over 100 million high-quality sentence-level corpus and more than 3 million high-resolution academic images. Using MiniLMs, these resources were organized into a high-dimensional vector library for precise retrieval and efficient utilization of extensive educational content. Consequently, we systematically redesigned the courses, textbooks, and teaching strategies for "Atmospheric Physics" based on MiniLMs. The course is designed as a "interdisciplinary-frontier" system, breaking down traditional boundaries between atmospheric science, space science, hydrology, and remote sensing. Teaching materials are transformed from static, lagging text formats into a dynamic digital resource library powered by MiniLM. For teaching methods, we have designed a question-based learning pathway. This paradigm promotes a shift from passive knowledge transfer to active cognitive development. Consequently, this MiniLM-driven "Atmospheric Physics" course demonstrates a specific avenue for "AI for education".
>
---
#### [new 078] On measuring grounding and generalizing grounding problems
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文将符号接地问题从二元判断转化为多维度评估框架，提出五个接地标准及评估元组。通过分析四种接地模式与案例，为跨学科研究提供统一技术语言，旨在系统探究表征与意义问题。**

- **链接: [https://arxiv.org/pdf/2512.06205v1](https://arxiv.org/pdf/2512.06205v1)**

> **作者:** Daniel Quigley; Eric Maynard
>
> **备注:** 36 pages, 85 sources
>
> **摘要:** The symbol grounding problem asks how tokens like cat can be about cats, as opposed to mere shapes manipulated in a calculus. We recast grounding from a binary judgment into an audit across desiderata, each indexed by an evaluation tuple (context, meaning type, threat model, reference distribution): authenticity (mechanisms reside inside the agent and, for strong claims, were acquired through learning or evolution); preservation (atomic meanings remain intact); faithfulness, both correlational (realized meanings match intended ones) and etiological (internal mechanisms causally contribute to success); robustness (graceful degradation under declared perturbations); compositionality (the whole is built systematically from the parts). We apply this framework to four grounding modes (symbolic; referential; vectorial; relational) and three case studies: model-theoretic semantics achieves exact composition but lacks etiological warrant; large language models show correlational fit and local robustness for linguistic tasks, yet lack selection-for-success on world tasks without grounded interaction; human language meets the desiderata under strong authenticity through evolutionary and developmental acquisition. By operationalizing a philosophical inquiry about representation, we equip philosophers of science, computer scientists, linguists, and mathematicians with a common language and technical framework for systematic investigation of grounding and meaning.
>
---
#### [new 079] Living the Novel: A System for Generating Self-Training Timeline-Aware Conversational Agents from Novels
- **分类: cs.HC; cs.CL**

- **简介: 该论文提出“Living Novel”系统，将小说转化为多角色对话体验。针对LLM角色易偏离人设和故事逻辑的问题，设计两阶段训练：深度人设对齐与叙事一致性增强，提升角色保真度与叙事连贯性，实现在移动网页端稳定沉浸的交互叙事。**

- **链接: [https://arxiv.org/pdf/2512.07474v1](https://arxiv.org/pdf/2512.07474v1)**

> **作者:** Yifei Huang; Tianyu Yan; Sitong Gong; Xiwei Gao; Caixin Kang; Ruicong Liu; Huchuan Lu; Bo Zheng
>
> **摘要:** We present the Living Novel, an end-to-end system that transforms any literary work into an immersive, multi-character conversational experience. This system is designed to solve two fundamental challenges for LLM-driven characters. Firstly, generic LLMs suffer from persona drift, often failing to stay in character. Secondly, agents often exhibit abilities that extend beyond the constraints of the story's world and logic, leading to both narrative incoherence (spoiler leakage) and robustness failures (frame-breaking). To address these challenges, we introduce a novel two-stage training pipeline. Our Deep Persona Alignment (DPA) stage uses data-free reinforcement finetuning to instill deep character fidelity. Our Coherence and Robustness Enhancing (CRE) stage then employs a story-time-aware knowledge graph and a second retrieval-grounded training pass to architecturally enforce these narrative constraints. We validate our system through a multi-phase evaluation using Jules Verne's Twenty Thousand Leagues Under the Sea. A lab study with a detailed ablation of system components is followed by a 5-day in-the-wild diary study. Our DPA pipeline helps our specialized model outperform GPT-4o on persona-specific metrics, and our CRE stage achieves near-perfect performance in coherence and robustness measures. Our study surfaces practical design guidelines for AI-driven narrative systems: we find that character-first self-training is foundational for believability, while explicit story-time constraints are crucial for sustaining coherent, interruption-resilient mobile-web experiences.
>
---
#### [new 080] LLM-Upgraded Graph Reinforcement Learning for Carbon-Aware Job Scheduling in Smart Manufacturing
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究智能制造业中的碳感知柔性作业车间调度任务，旨在优化制造过程的能耗与完工时间。提出名为Luca的框架，结合大语言模型与图神经网络生成状态表征，通过强化学习实现双目标优化，提升调度效率与可持续性。**

- **链接: [https://arxiv.org/pdf/2512.06351v1](https://arxiv.org/pdf/2512.06351v1)**

> **作者:** Zhiying Yang; Fang Liu; Wei Zhang; Xin Lou; Malcolm Yoke Hean Low; Boon Ping Gan
>
> **摘要:** This paper presents \textsc{Luca}, a \underline{l}arge language model (LLM)-\underline{u}pgraded graph reinforcement learning framework for \underline{c}arbon-\underline{a}ware flexible job shop scheduling. \textsc{Luca} addresses the challenges of dynamic and sustainable scheduling in smart manufacturing systems by integrating a graph neural network and an LLM, guided by a carefully designed in-house prompting strategy, to produce a fused embedding that captures both structural characteristics and contextual semantics of the latest scheduling state. This expressive embedding is then processed by a deep reinforcement learning policy network, which generates real-time scheduling decisions optimized for both makespan and carbon emission objectives. To support sustainability goals, \textsc{Luca} incorporates a dual-objective reward function that encourages both energy efficiency and scheduling timeliness. Experimental results on both synthetic and public datasets demonstrate that \textsc{Luca} consistently outperforms comparison algorithms. For instance, on the synthetic dataset, it achieves an average of 4.1\% and up to 12.2\% lower makespan compared to the best-performing comparison algorithm while maintaining the same emission level. On public datasets, additional gains are observed for both makespan and emission. These results demonstrate that \textsc{Luca} is effective and practical for carbon-aware scheduling in smart manufacturing.
>
---
#### [new 081] MMDuet2: Enhancing Proactive Interaction of Video MLLMs with Multi-Turn Reinforcement Learning
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视频多模态大模型的主动交互任务，旨在解决模型在视频流中何时主动回复的问题。提出MMDuet2，采用多轮强化学习训练，无需精确标注回复时间，实现及时准确的响应，提升主动交互性能。**

- **链接: [https://arxiv.org/pdf/2512.06810v1](https://arxiv.org/pdf/2512.06810v1)**

> **作者:** Yueqian Wang; Songxiang Liu; Disong Wang; Nuo Xu; Guanglu Wan; Huishuai Zhang; Dongyan Zhao
>
> **摘要:** Recent advances in video multimodal large language models (Video MLLMs) have significantly enhanced video understanding and multi-modal interaction capabilities. While most existing systems operate in a turn-based manner where the model can only reply after user turns, proactively deciding when to reply during video playback presents a promising yet challenging direction for real-time applications. In this work, we propose a novel text-to-text approach to proactive interaction, where the model autonomously determines whether to respond or remain silent at each turn based on dialogue history and visual context up to current frame of an streaming video. To overcome difficulties in previous methods such as manually tuning response decision thresholds and annotating precise reply times, we introduce a multi-turn RL based training method that encourages timely and accurate responses without requiring precise response time annotations. We train our model MMDuet2 on a dataset of 52k videos with two types of dialogues via SFT and RL. Experimental results demonstrate that MMDuet2 outperforms existing proactive Video MLLM baselines in response timing and quality, achieving state-of-the-art performance on the ProactiveVideoQA benchmark.
>
---
#### [new 082] A Neural Affinity Framework for Abstract Reasoning: Diagnosing the Compositional Gap in Transformer Architectures via Procedural Task Taxonomy
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属抽象推理任务，旨在诊断Transformer在ARC-AGI任务中的组成性缺陷。作者提出9类任务分类法，揭示Transformer对低神经亲和力任务存在性能瓶颈，证实架构适配性制约模型上限，倡导混合架构以突破瓶颈。**

- **链接: [https://arxiv.org/pdf/2512.07109v1](https://arxiv.org/pdf/2512.07109v1)**

> **作者:** Miguel Ingram; Arthur Joseph Merritt
>
> **备注:** 62 pages, 10 figures
>
> **摘要:** Responding to Hodel et al.'s (2024) call for a formal definition of task relatedness in re-arc, we present the first 9-category taxonomy of all 400 tasks, validated at 97.5% accuracy via rule-based code analysis. We prove the taxonomy's visual coherence by training a CNN on raw grid pixels (95.24% accuracy on S3, 36.25% overall, 3.3x chance), then apply the taxonomy diagnostically to the original ARC-AGI-2 test set. Our curriculum analysis reveals 35.3% of tasks exhibit low neural affinity for Transformers--a distributional bias mirroring ARC-AGI-2. To probe this misalignment, we fine-tuned a 1.7M-parameter Transformer across 302 tasks, revealing a profound Compositional Gap: 210 of 302 tasks (69.5%) achieve >80% cell accuracy (local patterns) but <10% grid accuracy (global synthesis). This provides direct evidence for a Neural Affinity Ceiling Effect, where performance is bounded by architectural suitability, not curriculum. Applying our framework to Li et al.'s independent ViTARC study (400 specialists, 1M examples each) confirms its predictive power: Very Low affinity tasks achieve 51.9% versus 77.7% for High affinity (p<0.001), with a task at 0% despite massive data. The taxonomy enables precise diagnosis: low-affinity tasks (A2) hit hard ceilings, while high-affinity tasks (C1) reach 99.8%. These findings indicate that progress requires hybrid architectures with affinity-aligned modules. We release our validated taxonomy,
>
---
#### [new 083] MATEX: A Multi-Agent Framework for Explaining Ethereum Transactions
- **分类: cs.CE; cs.CL; cs.HC**

- **简介: 该论文针对以太坊交易难理解的问题，提出MATEX多智能体框架，通过协同推理、知识检索与证据验证，生成基于链上数据和协议语义的逐步解释，提升交易透明度与用户安全性。**

- **链接: [https://arxiv.org/pdf/2512.06933v1](https://arxiv.org/pdf/2512.06933v1)**

> **作者:** Zifan Peng
>
> **摘要:** Understanding a complicated Ethereum transaction remains challenging: multi-hop token flows, nested contract calls, and opaque execution paths routinely lead users to blind signing. Based on interviews with everyday users, developers, and auditors, we identify the need for faithful, step-wise explanations grounded in both on-chain evidence and real-world protocol semantics. To meet this need, we introduce (matex, a cognitive multi-agent framework that models transaction understanding as a collaborative investigation-combining rapid hypothesis generation, dynamic off-chain knowledge retrieval, evidence-aware synthesis, and adversarial validation to produce faithful explanations.
>
---
#### [new 084] Less Is More, but Where? Dynamic Token Compression via LLM-Guided Keyframe Prior
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视频理解任务，旨在解决长视频处理中计算开销大的问题。提出DyToK方法，利用大模型注意力机制动态压缩视觉token，无需训练即可提升效率，兼顾准确率与速度，兼容现有压缩技术。**

- **链接: [https://arxiv.org/pdf/2512.06866v1](https://arxiv.org/pdf/2512.06866v1)**

> **作者:** Yulin Li; Haokun Gui; Ziyang Fan; Junjie Wang; Bin Kang; Bin Chen; Zhuotao Tian
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Recent advances in Video Large Language Models (VLLMs) have achieved remarkable video understanding capabilities, yet face critical efficiency bottlenecks due to quadratic computational growth with lengthy visual token sequences of long videos. While existing keyframe sampling methods can improve temporal modeling efficiency, additional computational cost is introduced before feature encoding, and the binary frame selection paradigm is found suboptimal. Therefore, in this work, we propose Dynamic Token compression via LLM-guided Keyframe prior (DyToK), a training-free paradigm that enables dynamic token compression by harnessing VLLMs' inherent attention mechanisms. Our analysis reveals that VLLM attention layers naturally encoding query-conditioned keyframe priors, by which DyToK dynamically adjusts per-frame token retention ratios, prioritizing semantically rich frames while suppressing redundancies. Extensive experiments demonstrate that DyToK achieves state-of-the-art efficiency-accuracy tradeoffs. DyToK shows plug-and-play compatibility with existing compression methods, such as VisionZip and FastV, attaining 4.3x faster inference while preserving accuracy across multiple VLLMs, such as LLaVA-OneVision and Qwen2.5-VL. Code is available at https://github.com/yu-lin-li/DyToK .
>
---
#### [new 085] ARCANE: A Multi-Agent Framework for Interpretable and Configurable Alignment
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出ARCANE框架，解决大语言模型代理在长周期任务中对齐利益相关者偏好的问题。通过可解释、可配置的自然语言评分标准，实现无需重训练的实时偏好调整，提升对齐的可读性与灵活性。**

- **链接: [https://arxiv.org/pdf/2512.06196v1](https://arxiv.org/pdf/2512.06196v1)**

> **作者:** Charlie Masters; Marta Grześkiewicz; Stefano V. Albrecht
>
> **备注:** Accepted to the AAAI 2026 LLAMAS Workshop (Large Language Model Agents for Multi-Agent Systems)
>
> **摘要:** As agents based on large language models are increasingly deployed to long-horizon tasks, maintaining their alignment with stakeholder preferences becomes critical. Effective alignment in such settings requires reward models that are interpretable so that stakeholders can understand and audit model objectives. Moreover, reward models must be capable of steering agents at interaction time, allowing preference shifts to be incorporated without retraining. We introduce ARCANE, a framework that frames alignment as a multi-agent collaboration problem that dynamically represents stakeholder preferences as natural-language rubrics: weighted sets of verifiable criteria that can be generated on-the-fly from task context. Inspired by utility theory, we formulate rubric learning as a reconstruction problem and apply a regularized Group-Sequence Policy Optimization (GSPO) procedure that balances interpretability, faithfulness, and computational efficiency. Using a corpus of 219 labeled rubrics derived from the GDPVal benchmark, we evaluate ARCANE on challenging tasks requiring multi-step reasoning and tool use. The learned rubrics produce compact, legible evaluations and enable configurable trade-offs (e.g., correctness vs. conciseness) without retraining. Our results show that rubric-based reward models offer a promising path toward interpretable, test-time adaptive alignment for complex, long-horizon AI systems.
>
---
#### [new 086] An Index-based Approach for Efficient and Effective Web Content Extraction
- **分类: cs.IR; cs.CL**

- **简介: 该论文聚焦网页内容提取任务，旨在解决现有方法在效率与适应性上的不足。提出基于索引的提取方法，将HTML划分为结构化片段，通过预测相关片段的索引位置实现快速、准确的内容抽取，提升RAG系统性能。**

- **链接: [https://arxiv.org/pdf/2512.06641v1](https://arxiv.org/pdf/2512.06641v1)**

> **作者:** Yihan Chen; Benfeng Xu; Xiaorui Wang; Zhendong Mao
>
> **摘要:** As web agents (e.g., Deep Research) routinely consume massive volumes of web pages to gather and analyze information, LLM context management -- under large token budgets and low signal density -- emerges as a foundational, high-importance, and technically challenging problem for agentic and RAG pipelines. Existing solutions for extracting relevant content are inadequate: generative extraction models suffer from high latency, rule-based heuristics lack adaptability, and chunk-and-rerank methods are blind to webpage structure. To overcome these issues, we introduce Index-based Web Content Extraction to reframe the extraction process from slow, token-by-token generation into a highly efficient, discriminative task of index prediction, achieving both effectiveness and efficiency. We partition HTML into structure-aware, addressable segments, and extract only the positional indices of content relevant to a given query. This method decouples extraction latency from content length, enabling rapid, query-relevant extraction. We first evaluate our method as a post-retrieval processing component within an RAG QA system and find that it improves QA accuracy. Then we directly measure its match rate with the target content in two scenarios: main content extraction (ME) and query-relevant extraction (QE). Experimental results show that our method outperforms existing works in both accuracy and speed, effectively bridging the gap between LLMs and the vast webpages.
>
---
#### [new 087] KidSpeak: A General Multi-purpose LLM for Kids' Speech Recognition and Screening
- **分类: eess.AS; cs.AI; cs.CL; cs.SD**

- **简介: 该论文针对儿童语音识别与语言障碍筛查任务，解决现有模型因依赖成人语音数据而对儿童语音表现差的问题。提出KidSpeak多任务语音大模型及FASA自动对齐工具，提升儿童语音数据质量与模型性能。**

- **链接: [https://arxiv.org/pdf/2512.05994v1](https://arxiv.org/pdf/2512.05994v1)**

> **作者:** Rohan Sharma; Dancheng Liu; Jingchen Sun; Shijie Zhou; Jiayu Qin; Jinjun Xiong; Changyou Chen
>
> **摘要:** With the rapid advancement of conversational and diffusion-based AI, there is a growing adoption of AI in educational services, ranging from grading and assessment tools to personalized learning systems that provide targeted support for students. However, this adaptability has yet to fully extend to the domain of children's speech, where existing models often fail due to their reliance on datasets designed for clear, articulate adult speech. Children, particularly those in early developmental stages or with speech and language pathologies, present unique challenges that current AI models and datasets are ill-equipped to handle. To address this, we introduce KidSpeak, a multi-task speech-enhanced Foundation Model capable of both generative and discriminative tasks specifically tailored to children's speech patterns. Our framework employs a two-stage training process that incorporates phonetic knowledge into the speech encoder, achieving an average accuracy of 87% across four separate tasks. Furthermore, recognizing the limitations of scalable human annotation and existing speech alignment tools, we propose the Flexible and Automatic Speech Aligner (FASA) and leverage the method to construct high quality datasets for training and evaluation. This novel alignment tool significantly improves the quality of aligned children's speech from noisy data, enhancing data quality by 13.6x compared to human annotations, as demonstrated on the CHILDES dataset. To the best of our knowledge, KidSpeak and FASA represent the first comprehensive solution designed for speech and language therapy in children, offering both a multi-purpose speech LLM and a robust alignment tool.
>
---
#### [new 088] NeuroABench: A Multimodal Evaluation Benchmark for Neurosurgical Anatomy Identification
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出NeuroABench，首个面向神经外科解剖识别的多模态评估基准。针对现有研究忽视手术视频中解剖理解的问题，构建了9小时标注视频数据集，评估68个解剖结构。实验显示当前多模态大模型在该任务上准确率仅40.87%，低于医学生平均水平，揭示模型与人类在解剖识别上的差距。**

- **链接: [https://arxiv.org/pdf/2512.06921v1](https://arxiv.org/pdf/2512.06921v1)**

> **作者:** Ziyang Song; Zelin Zang; Xiaofan Ye; Boqiang Xu; Long Bai; Jinlin Wu; Hongliang Ren; Hongbin Liu; Jiebo Luo; Zhen Lei
>
> **备注:** Accepted by IEEE ICIA 2025
>
> **摘要:** Multimodal Large Language Models (MLLMs) have shown significant potential in surgical video understanding. With improved zero-shot performance and more effective human-machine interaction, they provide a strong foundation for advancing surgical education and assistance. However, existing research and datasets primarily focus on understanding surgical procedures and workflows, while paying limited attention to the critical role of anatomical comprehension. In clinical practice, surgeons rely heavily on precise anatomical understanding to interpret, review, and learn from surgical videos. To fill this gap, we introduce the Neurosurgical Anatomy Benchmark (NeuroABench), the first multimodal benchmark explicitly created to evaluate anatomical understanding in the neurosurgical domain. NeuroABench consists of 9 hours of annotated neurosurgical videos covering 89 distinct procedures and is developed using a novel multimodal annotation pipeline with multiple review cycles. The benchmark evaluates the identification of 68 clinical anatomical structures, providing a rigorous and standardized framework for assessing model performance. Experiments on over 10 state-of-the-art MLLMs reveal significant limitations, with the best-performing model achieving only 40.87% accuracy in anatomical identification tasks. To further evaluate the benchmark, we extract a subset of the dataset and conduct an informative test with four neurosurgical trainees. The results show that the best-performing student achieves 56% accuracy, with the lowest scores of 28% and an average score of 46.5%. While the best MLLM performs comparably to the lowest-scoring student, it still lags significantly behind the group's average performance. This comparison underscores both the progress of MLLMs in anatomical understanding and the substantial gap that remains in achieving human-level performance.
>
---
#### [new 089] The Road of Adaptive AI for Precision in Cybersecurity
- **分类: cs.CR; cs.AI; cs.CL; cs.LG; cs.SE**

- **简介: 该论文聚焦生成式AI在网络安全中的自适应应用，旨在提升防御系统的精准性与鲁棒性。研究总结了实际部署经验，探讨检索与模型层面的适应机制协同，提出最佳实践与未来研究方向，以应对动态威胁环境。**

- **链接: [https://arxiv.org/pdf/2512.06048v1](https://arxiv.org/pdf/2512.06048v1)**

> **作者:** Sahil Garg
>
> **摘要:** Cybersecurity's evolving complexity presents unique challenges and opportunities for AI research and practice. This paper shares key lessons and insights from designing, building, and operating production-grade GenAI pipelines in cybersecurity, with a focus on the continual adaptation required to keep pace with ever-shifting knowledge bases, tooling, and threats. Our goal is to provide an actionable perspective for AI practitioners and industry stakeholders navigating the frontier of GenAI for cybersecurity, with particular attention to how different adaptation mechanisms complement each other in end-to-end systems. We present practical guidance derived from real-world deployments, propose best practices for leveraging retrieval- and model-level adaptation, and highlight open research directions for making GenAI more robust, precise, and auditable in cyber defense.
>
---
#### [new 090] Look Twice before You Leap: A Rational Agent Framework for Localized Adversarial Anonymization
- **分类: cs.CR; cs.CL**

- **简介: 该论文研究文本匿名化任务，旨在解决本地小模型下隐私与效用难以兼顾的问题。提出RLAA框架，通过攻击者-仲裁者-匿名化器架构引入理性决策机制，避免过度修改导致的效用崩溃，在保护隐私的同时显著提升效用。**

- **链接: [https://arxiv.org/pdf/2512.06713v1](https://arxiv.org/pdf/2512.06713v1)**

> **作者:** Donghang Duan; Xu Zheng
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** Current LLM-based text anonymization frameworks usually rely on remote API services from powerful LLMs, which creates an inherent "privacy paradox": users must somehow disclose data to untrusted third parties for superior privacy preservation. Moreover, directly migrating these frameworks to local small-scale models (LSMs) offers a suboptimal solution with catastrophic collapse in utility based on our core findings. Our work argues that this failure stems not merely from the capability deficits of LSMs, but from the inherent irrationality of the greedy adversarial strategies employed by current state-of-the-art (SoTA) methods. We model the anonymization process as a trade-off between Marginal Privacy Gain (MPG) and Marginal Utility Cost (MUC), and demonstrate that greedy strategies inevitably drift into an irrational state. To address this, we propose Rational Localized Adversarial Anonymization (RLAA), a fully localized and training-free framework featuring an Attacker-Arbitrator-Anonymizer (A-A-A) architecture. RLAA introduces an arbitrator that acts as a rationality gatekeeper, validating the attacker's inference to filter out feedback providing negligible benefits on privacy preservation. This mechanism enforces a rational early-stopping criterion, and systematically prevents utility collapse. Extensive experiments on different datasets demonstrate that RLAA achieves the best privacy-utility trade-off, and in some cases even outperforms SoTA on the Pareto principle. Our code and datasets will be released upon acceptance.
>
---
#### [new 091] PICKT: Practical Interlinked Concept Knowledge Tracing for Personalized Learning using Knowledge Map Concept Relations
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于知识追踪任务，旨在解决现有模型在冷启动和多源数据融合上的局限。提出PICKT模型，利用知识图谱整合概念关系与文本信息，提升新学生和新题目场景下的预测性能与实际应用稳定性。**

- **链接: [https://arxiv.org/pdf/2512.07179v1](https://arxiv.org/pdf/2512.07179v1)**

> **作者:** Wonbeen Lee; Channyoung Lee; Junho Sohn; Hansam Cho
>
> **备注:** 15 pages, 5 figures, 17 tables. Preparing submission for EDM 2026 conference
>
> **摘要:** With the recent surge in personalized learning, Intelligent Tutoring Systems (ITS) that can accurately track students' individual knowledge states and provide tailored learning paths based on this information are in demand as an essential task. This paper focuses on the core technology of Knowledge Tracing (KT) models that analyze students' sequences of interactions to predict their knowledge acquisition levels. However, existing KT models suffer from limitations such as restricted input data formats, cold start problems arising with new student enrollment or new question addition, and insufficient stability in real-world service environments. To overcome these limitations, a Practical Interlinked Concept Knowledge Tracing (PICKT) model that can effectively process multiple types of input data is proposed. Specifically, a knowledge map structures the relationships among concepts considering the question and concept text information, thereby enabling effective knowledge tracing even in cold start situations. Experiments reflecting real operational environments demonstrated the model's excellent performance and practicality. The main contributions of this research are as follows. First, a model architecture that effectively utilizes diverse data formats is presented. Second, significant performance improvements are achieved over existing models for two core cold start challenges: new student enrollment and new question addition. Third, the model's stability and practicality are validated through delicate experimental design, enhancing its applicability in real-world product environments. This provides a crucial theoretical and technical foundation for the practical implementation of next-generation ITS.
>
---
#### [new 092] Generating Storytelling Images with Rich Chains-of-Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出“叙事图像生成”任务，旨在生成蕴含丰富逻辑视觉线索的Storytelling Images。为解决其创作难问题，作者提出StorytellingPainter两阶段生成框架，并构建评估体系及轻量模型Mini-Storytellers，结合LLM与T2I能力实现高质量叙事图像生成。**

- **链接: [https://arxiv.org/pdf/2512.07198v1](https://arxiv.org/pdf/2512.07198v1)**

> **作者:** Xiujie Song; Qi Jia; Shota Watanabe; Xiaoyi Pang; Ruijie Chen; Mengyue Wu; Kenny Q. Zhu
>
> **摘要:** An image can convey a compelling story by presenting rich, logically connected visual clues. These connections form Chains-of-Reasoning (CoRs) within the image, enabling viewers to infer events, causal relationships, and other information, thereby understanding the underlying story. In this paper, we focus on these semantically rich images and define them as Storytelling Images. Such images have diverse applications beyond illustration creation and cognitive screening, leveraging their ability to convey multi-layered information visually and inspire active interpretation. However, due to their complex semantic nature, Storytelling Images are inherently challenging to create, and thus remain relatively scarce. To address this challenge, we introduce the Storytelling Image Generation task, which explores how generative AI models can be leveraged to create such images. Specifically, we propose a two-stage pipeline, StorytellingPainter, which combines the creative reasoning abilities of Large Language Models (LLMs) with the visual synthesis capabilities of Text-to-Image (T2I) models to generate Storytelling Images. Alongside this pipeline, we develop a dedicated evaluation framework comprising three main evaluators: a Semantic Complexity Evaluator, a KNN-based Diversity Evaluator and a Story-Image Alignment Evaluator. Given the critical role of story generation in the Storytelling Image Generation task and the performance disparity between open-source and proprietary LLMs, we further explore tailored training strategies to reduce this gap, resulting in a series of lightweight yet effective models named Mini-Storytellers. Experimental results demonstrate the feasibility and effectiveness of our approaches. The code is available at https://github.com/xiujiesong/StorytellingImageGeneration.
>
---
#### [new 093] Rethinking Training Dynamics in Scale-wise Autoregressive Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对尺度自回归生成模型中的暴露偏差问题，提出Self-Autoregressive Refinement（SAR）方法，通过 stagger-scale rollout 和对比学生强制损失，缓解训练-测试不匹配与尺度间学习难度不平衡，提升生成质量，适用于图像生成任务。**

- **链接: [https://arxiv.org/pdf/2512.06421v1](https://arxiv.org/pdf/2512.06421v1)**

> **作者:** Gengze Zhou; Chongjian Ge; Hao Tan; Feng Liu; Yicong Hong
>
> **摘要:** Recent advances in autoregressive (AR) generative models have produced increasingly powerful systems for media synthesis. Among them, next-scale prediction has emerged as a popular paradigm, where models generate images in a coarse-to-fine manner. However, scale-wise AR models suffer from exposure bias, which undermines generation quality. We identify two primary causes of this issue: (1) train-test mismatch, where the model must rely on its own imperfect predictions during inference, and (2) imbalance in scale-wise learning difficulty, where certain scales exhibit disproportionately higher optimization complexity. Through a comprehensive analysis of training dynamics, we propose Self-Autoregressive Refinement (SAR) to address these limitations. SAR introduces a Stagger-Scale Rollout (SSR) mechanism that performs lightweight autoregressive rollouts to expose the model to its own intermediate predictions, thereby aligning train-test patterns, and a complementary Contrastive Student-Forcing Loss (CSFL) that provides adequate supervision for self-generated contexts to ensure stable training. Experimental results show that applying SAR to pretrained AR models consistently improves generation quality with minimal computational overhead. For instance, SAR yields a 5.2% FID reduction on FlexVAR-d16 trained on ImageNet 256 within 10 epochs (5 hours on 32xA100 GPUs). Given its efficiency, scalability, and effectiveness, we expect SAR to serve as a reliable post-training method for visual autoregressive generation.
>
---
#### [new 094] Less Is More for Multi-Step Logical Reasoning of LLM Generalisation Under Rule Removal, Paraphrasing, and Compression
- **分类: cs.AI; cs.CL; cs.LG; cs.LO**

- **简介: 该论文研究大语言模型在逻辑推理中的泛化能力，针对规则删除、矛盾插入和逻辑等价改写等扰动进行评测。发现模型对语义保持变换鲁棒，但面对关键规则缺失或矛盾证据时表现脆弱，揭示其逻辑推理的局限性。**

- **链接: [https://arxiv.org/pdf/2512.06393v1](https://arxiv.org/pdf/2512.06393v1)**

> **作者:** Qiming Bao; Xiaoxuan Fu
>
> **摘要:** Large language models (LLMs) excel across many natural language tasks, yet their generalisation to structural perturbations in logical contexts remains poorly understood. We introduce a controlled evaluation framework that probes reasoning reliability through four targeted stress tests: (1) rule deletion, removing either redundant or essential rules from a multi-step inference chain; (2) contradictory evidence injection; (3) logic-preserving rewrites generated through several families of equivalence laws (contrapositive, double negation, implication, De Morgan, identity, and commutativity); and (4) multi-law equivalence stacking that introduces 2-5 simultaneous logical transformations. Across three representative model families: BERT, Qwen2, and LLaMA-like models. Our experiments reveal a strikingly consistent pattern: all models achieve perfect accuracy on the base tasks and remain fully generalise to redundant rule deletion and all equivalence-based rewrites (single or multi-law), but fail sharply under essential rule deletion (dropping to 25% accuracy) and collapse completely in the presence of explicit contradictions (0% accuracy). These results demonstrate that LLMs possess stable invariance to semantic-preserving logical transformations, yet remain fundamentally brittle to missing or conflicting evidence. Our framework provides a clean diagnostic tool for isolating such reasoning failure modes and highlights persistent gaps in the logical generalisation abilities of current LLMs.
>
---
#### [new 095] ProAgent: Harnessing On-Demand Sensory Contexts for Proactive LLM Agent Systems
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文提出ProAgent，首个端到端主动式LLM代理系统，旨在解决现有代理依赖显式指令的被动问题。通过按需分层感知和上下文感知的主动推理，实现基于环境与用户特征的主动服务，在AR眼镜上验证，显著提升预测准确率、工具调用性能与用户满意度。**

- **链接: [https://arxiv.org/pdf/2512.06721v1](https://arxiv.org/pdf/2512.06721v1)**

> **作者:** Bufang Yang; Lilin Xu; Liekang Zeng; Yunqi Guo; Siyang Jiang; Wenrui Lu; Kaiwei Liu; Hancheng Xiang; Xiaofan Jiang; Guoliang Xing; Zhenyu Yan
>
> **摘要:** Large Language Model (LLM) agents are emerging to transform daily life. However, existing LLM agents primarily follow a reactive paradigm, relying on explicit user instructions to initiate services, which increases both physical and cognitive workload. In this paper, we propose ProAgent, the first end-to-end proactive agent system that harnesses massive sensory contexts and LLM reasoning to deliver proactive assistance. ProAgent first employs a proactive-oriented context extraction approach with on-demand tiered perception to continuously sense the environment and derive hierarchical contexts that incorporate both sensory and persona cues. ProAgent then adopts a context-aware proactive reasoner to map these contexts to user needs and tool calls, providing proactive assistance. We implement ProAgent on Augmented Reality (AR) glasses with an edge server and extensively evaluate it on a real-world testbed, a public dataset, and through a user study. Results show that ProAgent achieves up to 33.4% higher proactive prediction accuracy, 16.8% higher tool-calling F1 score, and notable improvements in user satisfaction over state-of-the-art baselines, marking a significant step toward proactive assistants. A video demonstration of ProAgent is available at https://youtu.be/pRXZuzvrcVs.
>
---
#### [new 096] Arc Gradient Descent: A Mathematically Derived Reformulation of Gradient Descent with Phase-Aware, User-Controlled Step Dynamics
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.NE**

- **简介: 该论文提出ArcGD优化器，旨在改进梯度下降的步长动态与相位感知能力。针对非凸优化与深度学习训练难题，通过理论推导与实验验证，在Rosenbrock函数和CIFAR-10上优于Adam等主流优化器，具备更强泛化性与抗过拟合能力。**

- **链接: [https://arxiv.org/pdf/2512.06737v1](https://arxiv.org/pdf/2512.06737v1)**

> **作者:** Nikhil Verma; Joonas Linnosmaa; Espinosa-Leal Leonardo; Napat Vajragupta
>
> **备注:** 80 pages, 6 tables, 2 figures, 5 appendices, proof-of-concept
>
> **摘要:** The paper presents the formulation, implementation, and evaluation of the ArcGD optimiser. The evaluation is conducted initially on a non-convex benchmark function and subsequently on a real-world ML dataset. The initial comparative study using the Adam optimiser is conducted on a stochastic variant of the highly non-convex and notoriously challenging Rosenbrock function, renowned for its narrow, curved valley, across dimensions ranging from 2D to 1000D and an extreme case of 50,000D. Two configurations were evaluated to eliminate learning-rate bias: (i) both using ArcGD's effective learning rate and (ii) both using Adam's default learning rate. ArcGD consistently outperformed Adam under the first setting and, although slower under the second, achieved super ior final solutions in most cases. In the second evaluation, ArcGD is evaluated against state-of-the-art optimizers (Adam, AdamW, Lion, SGD) on the CIFAR-10 image classification dataset across 8 diverse MLP architectures ranging from 1 to 5 hidden layers. ArcGD achieved the highest average test accuracy (50.7%) at 20,000 iterations, outperforming AdamW (46.6%), Adam (46.8%), SGD (49.6%), and Lion (43.4%), winning or tying on 6 of 8 architectures. Notably, while Adam and AdamW showed strong early convergence at 5,000 iterations, but regressed with extended training, whereas ArcGD continued improving, demonstrating generalization and resistance to overfitting without requiring early stopping tuning. Strong performance on geometric stress tests and standard deep-learning benchmarks indicates broad applicability, highlighting the need for further exploration. Moreover, it is also shown that a variant of ArcGD can be interpreted as a special case of the Lion optimiser, highlighting connections between the inherent mechanisms of such optimisation methods.
>
---
#### [new 097] Toward More Reliable Artificial Intelligence: Reducing Hallucinations in Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文针对视觉-语言模型易产生幻觉的问题，提出一种无需训练的自修正框架，通过多维不确定性评估和视觉重注意机制，提升回答可靠性，降低幻觉率并提高准确性。**

- **链接: [https://arxiv.org/pdf/2512.07564v1](https://arxiv.org/pdf/2512.07564v1)**

> **作者:** Kassoum Sanogo; Renzo Ardiccioni
>
> **备注:** 24 pages, 3 figures, 2 tables. Training-free self-correction framework for vision-language models. Code and implementation details will be released at: https://github.com/kassoumsanogo1/self-correcting-vlm-re-Attention.git
>
> **摘要:** Vision-language models (VLMs) frequently generate hallucinated content plausible but incorrect claims about image content. We propose a training-free self-correction framework enabling VLMs to iteratively refine responses through uncertainty-guided visual re-attention. Our method combines multidimensional uncertainty quantification (token entropy, attention dispersion, semantic consistency, claim confidence) with attention-guided cropping of under-explored regions. Operating entirely with frozen, pretrained VLMs, our framework requires no gradient updates. We validate our approach on the POPE and MMHAL BENCH benchmarks using the Qwen2.5-VL-7B [23] architecture. Experimental results demonstrate that our method reduces hallucination rates by 9.8 percentage points compared to the baseline, while improving object existence accuracy by 4.7 points on adversarial splits. Furthermore, qualitative analysis confirms that uncertainty-guided re-attention successfully grounds corrections in visual evidence where standard decoding fails. We validate our approach on Qwen2.5-VL-7B [23], with plans to extend validation across diverse architectures in future versions. We release our code and methodology to facilitate future research in trustworthy multimodal systems.
>
---
#### [new 098] Why They Disagree: Decoding Differences in Opinions about AI Risk on the Lex Fridman Podcast
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文分析AI风险争议，解析“末日论”与“乐观派”在定义、事实、因果和道德前提上的分歧。研究表明，分歧主因在于对系统设计与涌现的认知差异及人类理性边界的不同看法，而非价值观冲突，并提出用LLM ensemble解析争议的方法。**

- **链接: [https://arxiv.org/pdf/2512.06350v1](https://arxiv.org/pdf/2512.06350v1)**

> **作者:** Nghi Truong; Phanish Puranam; Özgecan Koçak
>
> **摘要:** The emergence of transformative technologies often surfaces deep societal divisions, nowhere more evident than in contemporary debates about artificial intelligence (AI). A striking feature of these divisions is that they persist despite shared interests in ensuring that AI benefits humanity and avoiding catastrophic outcomes. This paper analyzes contemporary debates about AI risk, parsing the differences between the "doomer" and "boomer" perspectives into definitional, factual, causal, and moral premises to identify key points of contention. We find that differences in perspectives about existential risk ("X-risk") arise fundamentally from differences in causal premises about design vs. emergence in complex systems, while differences in perspectives about employment risks ("E-risks") pertain to different causal premises about the applicability of past theories (evolution) vs their inapplicability (revolution). Disagreements about these two forms of AI risk appear to share two properties: neither involves significant disagreements on moral values and both can be described in terms of differing views on the extent of boundedness of human rationality. Our approach to analyzing reasoning chains at scale, using an ensemble of LLMs to parse textual data, can be applied to identify key points of contention in debates about risk to the public in any arena.
>
---
#### [new 099] Recover-to-Forget: Gradient Reconstruction from LoRA for Efficient LLM Unlearning
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究大语言模型的高效遗忘学习任务，旨在解决现有方法需全模型微调或原始数据的问题。提出Recover-to-Forget框架，通过LoRA适配器重构全模型梯度方向，实现无需重训练或内部参数的可扩展遗忘。**

- **链接: [https://arxiv.org/pdf/2512.07374v1](https://arxiv.org/pdf/2512.07374v1)**

> **作者:** Yezi Liu; Hanning Chen; Wenjun Huang; Yang Ni; Mohsen Imani
>
> **摘要:** Unlearning in large foundation models (e.g., LLMs) is essential for enabling dynamic knowledge updates, enforcing data deletion rights, and correcting model behavior. However, existing unlearning methods often require full-model fine-tuning or access to the original training data, which limits their scalability and practicality. In this work, we introduce Recover-to-Forget (R2F), a novel framework for efficient unlearning in LLMs based on reconstructing full-model gradient directions from low-rank LoRA adapter updates. Rather than performing backpropagation through the full model, we compute gradients with respect to LoRA parameters using multiple paraphrased prompts and train a gradient decoder to approximate the corresponding full-model gradients. To ensure applicability to larger or black-box models, the decoder is trained on a proxy model and transferred to target models. We provide a theoretical analysis of cross-model generalization and demonstrate that our method achieves effective unlearning while preserving general model performance. Experimental results demonstrate that R2F offers a scalable and lightweight alternative for unlearning in pretrained LLMs without requiring full retraining or access to internal parameters.
>
---
#### [new 100] Cognitive Control Architecture (CCA): A Lifecycle Supervision Framework for Robustly Aligned AI Agents
- **分类: cs.AI; cs.CL; cs.CR**

- **简介: 该论文针对LLM代理易受间接提示注入攻击的问题，提出认知控制架构（CCA），通过意图图与分级裁决器实现全生命周期监督，确保安全、功能与效率的统一。**

- **链接: [https://arxiv.org/pdf/2512.06716v1](https://arxiv.org/pdf/2512.06716v1)**

> **作者:** Zhibo Liang; Tianze Hu; Zaiye Chen; Mingjie Tang
>
> **摘要:** Autonomous Large Language Model (LLM) agents exhibit significant vulnerability to Indirect Prompt Injection (IPI) attacks. These attacks hijack agent behavior by polluting external information sources, exploiting fundamental trade-offs between security and functionality in existing defense mechanisms. This leads to malicious and unauthorized tool invocations, diverting agents from their original objectives. The success of complex IPIs reveals a deeper systemic fragility: while current defenses demonstrate some effectiveness, most defense architectures are inherently fragmented. Consequently, they fail to provide full integrity assurance across the entire task execution pipeline, forcing unacceptable multi-dimensional compromises among security, functionality, and efficiency. Our method is predicated on a core insight: no matter how subtle an IPI attack, its pursuit of a malicious objective will ultimately manifest as a detectable deviation in the action trajectory, distinct from the expected legitimate plan. Based on this, we propose the Cognitive Control Architecture (CCA), a holistic framework achieving full-lifecycle cognitive supervision. CCA constructs an efficient, dual-layered defense system through two synergistic pillars: (i) proactive and preemptive control-flow and data-flow integrity enforcement via a pre-generated "Intent Graph"; and (ii) an innovative "Tiered Adjudicator" that, upon deviation detection, initiates deep reasoning based on multi-dimensional scoring, specifically designed to counter complex conditional attacks. Experiments on the AgentDojo benchmark substantiate that CCA not only effectively withstands sophisticated attacks that challenge other advanced defense methods but also achieves uncompromised security with notable efficiency and robustness, thereby reconciling the aforementioned multi-dimensional trade-off.
>
---
#### [new 101] Flash Multi-Head Feed-Forward Network
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出FlashMHF，改进Transformer中FFN结构。针对多头FFN内存大、扩展性差的问题，设计I/O感知融合内核与动态加权子网络，提升效率与表达能力，在128M至1.3B模型上验证了性能优越性。**

- **链接: [https://arxiv.org/pdf/2512.06989v1](https://arxiv.org/pdf/2512.06989v1)**

> **作者:** Minshen Zhang; Xiang Hu; Jianguo Li; Wei Wu; Kewei Tu
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** We explore Multi-Head FFN (MH-FFN) as a replacement of FFN in the Transformer architecture, motivated by the structural similarity between single-head attention and FFN. While multi-head mechanisms enhance expressivity in attention, naively applying them to FFNs faces two challenges: memory consumption scaling with the head count, and an imbalanced ratio between the growing intermediate size and the fixed head dimension as models scale, which degrades scalability and expressive power. To address these challenges, we propose Flash Multi-Head FFN (FlashMHF), with two key innovations: an I/O-aware fused kernel computing outputs online in SRAM akin to FlashAttention, and a design using dynamically weighted parallel sub-networks to maintain a balanced ratio between intermediate and head dimensions. Validated on models from 128M to 1.3B parameters, FlashMHF consistently improves perplexity and downstream task accuracy over SwiGLU FFNs, while reducing peak memory usage by 3-5x and accelerating inference by up to 1.08x. Our work establishes the multi-head design as a superior architectural principle for FFNs, presenting FlashMHF as a powerful, efficient, and scalable alternative to FFNs in Transformers.
>
---
#### [new 102] Pay Less Attention to Function Words for Free Robustness of Vision-Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文针对视觉-语言模型（VLM）在对抗攻击下的鲁棒性问题，提出Function-word De-Attention（FDA）方法，通过削弱函数词的注意力提升模型鲁棒性，兼顾性能与安全，在多种任务和模型上验证了有效性。**

- **链接: [https://arxiv.org/pdf/2512.07222v1](https://arxiv.org/pdf/2512.07222v1)**

> **作者:** Qiwei Tian; Chenhao Lin; Zhengyu Zhao; Chao Shen
>
> **摘要:** To address the trade-off between robustness and performance for robust VLM, we observe that function words could incur vulnerability of VLMs against cross-modal adversarial attacks, and propose Function-word De-Attention (FDA) accordingly to mitigate the impact of function words. Similar to differential amplifiers, our FDA calculates the original and the function-word cross-attention within attention heads, and differentially subtracts the latter from the former for more aligned and robust VLMs. Comprehensive experiments include 2 SOTA baselines under 6 different attacks on 2 downstream tasks, 3 datasets, and 3 models. Overall, our FDA yields an average 18/13/53% ASR drop with only 0.2/0.3/0.6% performance drops on the 3 tested models on retrieval, and a 90% ASR drop with a 0.3% performance gain on visual grounding. We demonstrate the scalability, generalization, and zero-shot performance of FDA experimentally, as well as in-depth ablation studies and analysis. Code will be made publicly at https://github.com/michaeltian108/FDA.
>
---
#### [new 103] The Role of Entropy in Visual Grounding: Analysis and Optimization
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究视觉定位任务中熵的作用与调控问题，分析其与推理任务的差异，提出ECVGPO算法以平衡探索与利用，实现更优的熵控制，提升多模态大模型在视觉定位中的性能。**

- **链接: [https://arxiv.org/pdf/2512.06726v1](https://arxiv.org/pdf/2512.06726v1)**

> **作者:** Shuo Li; Jiajun Sun; Zhihao Zhang; Xiaoran Fan; Senjie Jin; Hui Li; Yuming Yang; Junjie Ye; Lixing Shen; Tao Ji; Tao Gui; Qi Zhang; Xuanjing Huang
>
> **摘要:** Recent advances in fine-tuning multimodal large language models (MLLMs) using reinforcement learning have achieved remarkable progress, particularly with the introduction of various entropy control techniques. However, the role and characteristics of entropy in perception-oriented tasks like visual grounding, as well as effective strategies for controlling it, remain largely unexplored. To address this issue, we focus on the visual grounding task and analyze the role and characteristics of entropy in comparison to reasoning tasks. Building on these findings, we introduce ECVGPO (Entropy Control Visual Grounding Policy Optimization), an interpretable algorithm designed for effective entropy regulation. Through entropy control, the trade-off between exploration and exploitation is better balanced. Experiments show that ECVGPO achieves broad improvements across various benchmarks and models.
>
---
#### [new 104] ReasonBENCH: Benchmarking the (In)Stability of LLM Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文聚焦大语言模型推理的不稳定性问题，提出ReasonBENCH基准，通过多轮运行评估推理性能与成本的波动性，揭示现有方法在复现性和成本一致性上的缺陷，并推动方差感知的评估标准。**

- **链接: [https://arxiv.org/pdf/2512.07795v1](https://arxiv.org/pdf/2512.07795v1)**

> **作者:** Nearchos Potamitis; Lars Klein; Akhil Arora
>
> **备注:** 11 pages, 3 tables, 4 figures
>
> **摘要:** Large language models (LLMs) are increasingly deployed in settings where reasoning, such as multi-step problem solving and chain-of-thought, is essential. Yet, current evaluation practices overwhelmingly report single-run accuracy while ignoring the intrinsic uncertainty that naturally arises from stochastic decoding. This omission creates a blind spot because practitioners cannot reliably assess whether a method's reported performance is stable, reproducible, or cost-consistent. We introduce ReasonBENCH, the first benchmark designed to quantify the underlying instability in LLM reasoning. ReasonBENCH provides (i) a modular evaluation library that standardizes reasoning frameworks, models, and tasks, (ii) a multi-run protocol that reports statistically reliable metrics for both quality and cost, and (iii) a public leaderboard to encourage variance-aware reporting. Across tasks from different domains, we find that the vast majority of reasoning strategies and models exhibit high instability. Notably, even strategies with similar average performance can display confidence intervals up to four times wider, and the top-performing methods often incur higher and less stable costs. Such instability compromises reproducibility across runs and, consequently, the reliability of reported performance. To better understand these dynamics, we further analyze the impact of prompts, model families, and scale on the trade-off between solve rate and stability. Our results highlight reproducibility as a critical dimension for reliable LLM reasoning and provide a foundation for future reasoning methods and uncertainty quantification techniques. ReasonBENCH is publicly available at https://github.com/au-clan/ReasonBench .
>
---
#### [new 105] When Distance Distracts: Representation Distance Bias in BT-Loss for Reward Models
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究奖励模型中的Bradley-Terry（BT）损失函数，发现其训练受表示距离偏差影响，导致小距离样本更新弱。为此提出NormBT，通过归一化缓解该问题，提升模型性能，尤其在细粒度推理任务上效果显著。**

- **链接: [https://arxiv.org/pdf/2512.06343v1](https://arxiv.org/pdf/2512.06343v1)**

> **作者:** Tong Xie; Andrew Bai; Yuanhao Ban; Yunqi Hong; Haoyu Li; Cho-jui Hsieh
>
> **摘要:** Reward models are central to Large Language Model (LLM) alignment within the framework of RLHF. The standard objective used in reward modeling is the Bradley-Terry (BT) loss, which learns from pairwise data consisting of a pair of chosen and rejected responses. In this work, we analyze the per-sample gradient of BT-loss and show that its norm scales with two distinct components: (1) the difference in predicted rewards between chosen and rejected responses, which reflects the prediction error, and critically, (2) representation distance between the pair measured in the output space of the final layer. While the first term captures the intended training signal, we show that the second term can significantly impact the update magnitude and misalign learning. Specifically, pairs with small representation distance often receive vanishingly weak updates, even when misranked, while pairs with large distance receive disproportionately strong updates. This leads to gradients from large-distance pairs to overshadow those from small-distance pairs, where fine-grained distinctions are especially important. To overcome this limitation, we propose NormBT, an adaptive pair-wise normalization scheme that balances representation-driven effects and focuses learning signals on prediction error. NormBT is a lightweight, drop-in integration to BT loss with negligible overhead. Across various LLM backbones and datasets, NormBT improves reward model performance consistently, with notable gains of over 5% on the Reasoning category of RewardBench, which contains numerous small-distance pairs. This work reveals a key limitation in the widely used BT objective and provides a simple, effective correction.
>
---
#### [new 106] Block Sparse Flash Attention
- **分类: cs.LG; cs.CL; cs.PF**

- **简介: 该论文针对长文本推理中注意力计算瓶颈问题，提出Block-Sparse FlashAttention（BSFA），通过块级稀疏化与阈值剪枝，在保持准确率的同时加速推理，无需训练，仅需一次校准即可实现最高1.24倍加速。**

- **链接: [https://arxiv.org/pdf/2512.07011v1](https://arxiv.org/pdf/2512.07011v1)**

> **作者:** Daniel Ohayon; Itay Lamprecht; Itay Hubara; Israel Cohen; Daniel Soudry; Noam Elata
>
> **备注:** 10 pages, 5 figures. Code: https://github.com/Danielohayon/Block-Sparse-Flash-Attention
>
> **摘要:** Modern large language models increasingly require long contexts for reasoning and multi-document tasks, but attention's quadratic complexity creates a severe computational bottleneck. We present Block-Sparse FlashAttention (BSFA), a drop-in replacement that accelerates long-context inference while preserving model quality. Unlike methods that predict importance before computing scores, BSFA computes exact query-key similarities to select the top-k most important value blocks for each query. By comparing per-block maximum scores against calibrated thresholds, we skip approximately 50% of the computation and memory transfers for pruned blocks. Our training-free approach requires only a one-time threshold calibration on a small dataset to learn the per-layer and per-head attention score distributions. We provide a CUDA kernel implementation that can be used as a drop-in replacement for FlashAttention. On Llama-3.1-8B, BSFA achieves up to 1.10x speedup on real-world reasoning benchmarks and up to 1.24x for needle-in-a-haystack retrieval tasks while maintaining above 99% baseline accuracy, with certain configurations even improving accuracy by focusing on the most relevant content, substantially outperforming existing sparse attention methods. The implementation is available at https://github.com/Danielohayon/Block-Sparse-Flash-Attention
>
---
#### [new 107] Think-Reflect-Revise: A Policy-Guided Reflective Framework for Safety Alignment in Large Vision Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于安全对齐任务，旨在解决大视觉语言模型在单次推理中易受 jailbreak 攻击的问题。提出 Think-Reflect-Revise 框架，通过构建反思数据集并结合强化学习，实现策略引导的自我修正，提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2512.07141v1](https://arxiv.org/pdf/2512.07141v1)**

> **作者:** Fenghua Weng; Chaochao Lu; Xia Hu; Wenqi Shao; Wenjie Wang
>
> **摘要:** As multimodal reasoning improves the overall capabilities of Large Vision Language Models (LVLMs), recent studies have begun to explore safety-oriented reasoning, aiming to enhance safety awareness by analyzing potential safety risks during the reasoning process before generating the final response. Although such approaches improve safety awareness and interpretability, this single-pass think-then-answer paradigm remains vulnerable to contextual or visual jailbreak attacks. This reveals a critical flaw: single-pass reasoning may overlook explicit harmful content in its own output. Our key insight is to exploit this wasted signal through reflection, which can effectively leverage the malicious content revealed in the first-pass reasoning to enable genuine self-correction and prevent unsafe generations. Motivated by this, we propose Think-Reflect-Revise (TRR), a three-stage training framework designed to enhance the safety alignment of LVLMs through policy-guided self-reflection. We first build a Reflective Safety Reasoning (ReSafe) dataset with 5,000 examples that follow a think-reflect-revise process. We then fine-tune the target model using the ReSafe dataset to initialize reflective behavior, and finally reinforce policy-guided reflection through reinforcement learning. Experimental results show that TRR substantially improves the safety performance of LVLMs across both safety-awareness benchmarks and jailbreak attack evaluations, increasing the overall safe response rate from 42.8% to 87.7% on Qwen2.5-VL-7B, while preserving stable performance on general benchmarks such as MMMU and MMStar. The project page is available at https://think-reflect-revise.github.io/.
>
---
#### [new 108] Group Representational Position Encoding
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出GRAPE，一种基于群作用的统一位置编码框架，旨在解决长上下文建模中的位置表示问题。它统一了旋转（如RoPE）和加性偏置（如ALiBi）方法，支持相对位置编码、流式缓存，并扩展了几何结构以增强特征耦合。**

- **链接: [https://arxiv.org/pdf/2512.07805v1](https://arxiv.org/pdf/2512.07805v1)**

> **作者:** Yifan Zhang; Zixiang Chen; Yifeng Liu; Zhen Qin; Huizhuo Yuan; Kangping Xu; Yang Yuan; Quanquan Gu; Andrew Chi-Chih Yao
>
> **备注:** Project Page: https://github.com/model-architectures/GRAPE
>
> **摘要:** We present GRAPE (Group RepresentAtional Position Encoding), a unified framework for positional encoding based on group actions. GRAPE brings together two families of mechanisms: (i) multiplicative rotations (Multiplicative GRAPE) in $\mathrm{SO}(d)$ and (ii) additive logit biases (Additive GRAPE) arising from unipotent actions in the general linear group $\mathrm{GL}$. In Multiplicative GRAPE, a position $n \in \mathbb{Z}$ (or $t \in \mathbb{R}$) acts as $\mathbf{G}(n)=\exp(n\,ω\,\mathbf{L})$ with a rank-2 skew generator $\mathbf{L} \in \mathbb{R}^{d \times d}$, yielding a relative, compositional, norm-preserving map with a closed-form matrix exponential. RoPE is recovered exactly when the $d/2$ planes are the canonical coordinate pairs with log-uniform spectrum. Learned commuting subspaces and compact non-commuting mixtures strictly extend this geometry to capture cross-subspace feature coupling at $O(d)$ and $O(r d)$ cost per head, respectively. In Additive GRAPE, additive logits arise as rank-1 (or low-rank) unipotent actions, recovering ALiBi and the Forgetting Transformer (FoX) as exact special cases while preserving an exact relative law and streaming cacheability. Altogether, GRAPE supplies a principled design space for positional geometry in long-context models, subsuming RoPE and ALiBi as special cases. Project Page: https://github.com/model-architectures/GRAPE.
>
---
#### [new 109] LUNE: Efficient LLM Unlearning via LoRA Fine-Tuning with Negative Examples
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型遗忘任务，旨在解决大语言模型中特定知识的高效移除问题。作者提出LUNE框架，通过LoRA微调结合负样本，仅更新低秩适配器实现轻量级遗忘，在降低一个数量级计算成本的同时，达到与全量微调相当的效果。**

- **链接: [https://arxiv.org/pdf/2512.07375v1](https://arxiv.org/pdf/2512.07375v1)**

> **作者:** Yezi Liu; Hanning Chen; Wenjun Huang; Yang Ni; Mohsen Imani
>
> **摘要:** Large language models (LLMs) possess vast knowledge acquired from extensive training corpora, but they often cannot remove specific pieces of information when needed, which makes it hard to handle privacy, bias mitigation, and knowledge correction. Traditional model unlearning approaches require computationally expensive fine-tuning or direct weight editing, making them impractical for real-world deployment. In this work, we introduce LoRA-based Unlearning with Negative Examples (LUNE), a lightweight framework that performs negative-only unlearning by updating only low-rank adapters while freezing the backbone, thereby localizing edits and avoiding disruptive global changes. Leveraging Low-Rank Adaptation (LoRA), LUNE targets intermediate representations to suppress (or replace) requested knowledge with an order-of-magnitude lower compute and memory than full fine-tuning or direct weight editing. Extensive experiments on multiple factual unlearning tasks show that LUNE: (I) achieves effectiveness comparable to full fine-tuning and memory-editing methods, and (II) reduces computational cost by about an order of magnitude.
>
---
## 更新

#### [replaced 001] Internal World Models as Imagination Networks in Cognitive Agents
- **分类: cs.AI; cs.CL; cs.SI; q-bio.NC**

- **简介: 该论文属于认知科学与人工智能交叉任务，旨在探究人类与大语言模型在内部世界模型上的差异。通过构建想象网络并分析其结构，发现人类具有稳定组织模式，而LLM则无，揭示二者根本差异。**

- **链接: [https://arxiv.org/pdf/2510.04391v3](https://arxiv.org/pdf/2510.04391v3)**

> **作者:** Saurabh Ranjan; Brian Odegaard
>
> **摘要:** The computational role of imagination remains debated. While classical accounts emphasize reward maximization, emerging evidence suggests imagination serves a broader function: accessing internal world models (IWMs). Here, we employ psychological network analysis to compare IWMs in humans and large language models (LLMs) through imagination vividness ratings. Using the Vividness of Visual Imagery Questionnaire (VVIQ-2) and Plymouth Sensory Imagery Questionnaire (PSIQ), we construct imagination networks from three human populations (Florida, Poland, London; N=2,743) and six LLM variants in two conversation conditions. Human imagination networks demonstrate robust correlations across centrality measures (expected influence, strength, closeness) and consistent clustering patterns, indicating shared structural organization of IWMs across populations. In contrast, LLM-derived networks show minimal clustering and weak centrality correlations, even when manipulating conversational memory. These systematic differences persist across environmental scenes (VVIQ-2) and sensory modalities (PSIQ), revealing fundamental disparities between human and artificial world models. Our network-based approach provides a quantitative framework for comparing internally-generated representations across cognitive agents, with implications for developing human-like imagination in artificial intelligence systems.
>
---
#### [replaced 002] The AI Consumer Index (ACE)
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文提出AI消费者指数ACE，旨在评估前沿AI模型执行日常消费任务的能力。构建了含400个测试案例的基准，覆盖购物、饮食、游戏和DIY，开源80个开发案例，设计基于检索结果的新评分方法，揭示当前模型在真实消费场景中仍存在显著差距。**

- **链接: [https://arxiv.org/pdf/2512.04921v2](https://arxiv.org/pdf/2512.04921v2)**

> **作者:** Julien Benchek; Rohit Shetty; Benjamin Hunsberger; Ajay Arun; Zach Richards; Brendan Foody; Osvald Nitski; Bertie Vidgen
>
> **摘要:** We introduce the first version of the AI Consumer Index (ACE), a benchmark for assessing whether frontier AI models can perform everyday consumer tasks. ACE contains a hidden heldout set of 400 test cases, split across four consumer activities: shopping, food, gaming, and DIY. We are also open sourcing 80 cases as a devset with a CC-BY license. For the ACE leaderboard we evaluated 10 frontier models (with websearch turned on) using a novel grading methodology that dynamically checks whether relevant parts of the response are grounded in the retrieved web sources. GPT 5 (Thinking = High) is the top-performing model, scoring 56.1%, followed by o3 Pro (Thinking = On) at 55.2% and GPT 5.1 (Thinking = High) at 55.1%. Model scores differ across domains, and in Shopping the top model scores under 50\%. We find that models are prone to hallucinating key information, such as prices. ACE shows a substantial gap between the performance of even the best models and consumers' AI needs.
>
---
#### [replaced 003] Hallucination reduction with CASAL: Contrastive Activation Steering For Amortized Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型的幻觉抑制任务，旨在解决模型在未知问题上产生虚假回答的问题。作者提出CASAL方法，通过对比激活引导将知识感知能力融入模型权重，仅微调单层子模块即可显著减少幻觉，提升模型在分布外场景的泛化与实用性。**

- **链接: [https://arxiv.org/pdf/2510.02324v2](https://arxiv.org/pdf/2510.02324v2)**

> **作者:** Wannan; Yang; Xinchi Qiu; Lei Yu; Yuchen Zhang; Aobo Yang; Narine Kokhlikyan; Nicola Cancedda; Diego Garcia-Olano
>
> **摘要:** Large Language Models (LLMs) exhibit impressive capabilities but often hallucinate, confidently providing incorrect answers instead of admitting ignorance. Prior work has shown that models encode linear representations of their own knowledge and that activation steering can reduce hallucinations. These approaches, however, require real-time monitoring and intervention during inference. We introduce Contrastive Activation Steering for Amortized Learning (CASAL), an efficient algorithm that connects interpretability with amortized optimization. CASAL directly bakes the benefits of activation steering into model's weights. Once trained, LLMs answer questions they know while abstaining from answering those they do not. CASAL's light-weight design requires training only a submodule of a single transformer layer and yet reduces hallucination by 30%-40% across multiple short-form QA benchmarks. CASAL is 30x more compute-efficient and 20x more data-efficient than strong LoRA-based baselines such as SFT and DPO, boosting its practical applicability in data scarce domains. Importantly, CASAL also generalizes effectively to out-of-distribution (OOD) domains. We showcase CASAL's flexibility in mitigating hallucinations in both text-only and vision-language models. To our knowledge, CASAL is the first steering-based training method that has been shown to be effective for both dense and Mixture-of-Experts (MoE) models. CASAL represents a promising step forward for applying interpretability-inspired method for practical deployment in production systems.
>
---
#### [replaced 004] A Content-Preserving Secure Linguistic Steganography
- **分类: cs.CR; cs.CL**

- **简介: 该论文研究语言隐写任务，旨在解决传统方法因修改文本内容导致的安全隐患。提出CLstega，通过可控分布变换在不改变原文的情况下嵌入秘密信息，实现高安全、高提取率的隐写通信。**

- **链接: [https://arxiv.org/pdf/2511.12565v2](https://arxiv.org/pdf/2511.12565v2)**

> **作者:** Lingyun Xiang; Chengfu Ou; Xu He; Zhongliang Yang; Yuling Liu
>
> **备注:** This is the extended version of the paper accepted to AAAI 2026
>
> **摘要:** Existing linguistic steganography methods primarily rely on content transformations to conceal secret messages. However, they often cause subtle yet looking-innocent deviations between normal and stego texts, posing potential security risks in real-world applications. To address this challenge, we propose a content-preserving linguistic steganography paradigm for perfectly secure covert communication without modifying the cover text. Based on this paradigm, we introduce CLstega (\textit{C}ontent-preserving \textit{L}inguistic \textit{stega}nography), a novel method that embeds secret messages through controllable distribution transformation. CLstega first applies an augmented masking strategy to locate and mask embedding positions, where MLM(masked language model)-predicted probability distributions are easily adjustable for transformation. Subsequently, a dynamic distribution steganographic coding strategy is designed to encode secret messages by deriving target distributions from the original probability distributions. To achieve this transformation, CLstega elaborately selects target words for embedding positions as labels to construct a masked sentence dataset, which is used to fine-tune the original MLM, producing a target MLM capable of directly extracting secret messages from the cover text. This approach ensures perfect security of secret messages while fully preserving the integrity of the original cover text. Experimental results show that CLstega can achieve a 100\% extraction success rate, and outperforms existing methods in security, effectively balancing embedding capacity and security.
>
---
#### [replaced 005] DiTAR: Diffusion Transformer Autoregressive Modeling for Speech Generation
- **分类: eess.AS; cs.AI; cs.CL; cs.LG; cs.SD**

- **简介: 该论文研究语音生成任务，旨在解决连续语音表征自回归生成中的高计算成本与生成质量不佳问题。提出DiTAR框架，结合语言模型与扩散Transformer，采用分块生成策略和基于时间点的温度控制，提升生成效率与音质。**

- **链接: [https://arxiv.org/pdf/2502.03930v4](https://arxiv.org/pdf/2502.03930v4)**

> **作者:** Dongya Jia; Zhuo Chen; Jiawei Chen; Chenpeng Du; Jian Wu; Jian Cong; Xiaobin Zhuang; Chumin Li; Zhen Wei; Yuping Wang; Yuxuan Wang
>
> **备注:** ByteDance Seed template, ICML 2025
>
> **摘要:** Several recent studies have attempted to autoregressively generate continuous speech representations without discrete speech tokens by combining diffusion and autoregressive models, yet they often face challenges with excessive computational loads or suboptimal outcomes. In this work, we propose Diffusion Transformer Autoregressive Modeling (DiTAR), a patch-based autoregressive framework combining a language model with a diffusion transformer. This approach significantly enhances the efficacy of autoregressive models for continuous tokens and reduces computational demands. DiTAR utilizes a divide-and-conquer strategy for patch generation, where the language model processes aggregated patch embeddings and the diffusion transformer subsequently generates the next patch based on the output of the language model. For inference, we propose defining temperature as the time point of introducing noise during the reverse diffusion ODE to balance diversity and determinism. We also show in the extensive scaling analysis that DiTAR has superb scalability. In zero-shot speech generation, DiTAR achieves state-of-the-art performance in robustness, speaker similarity, and naturalness.
>
---
#### [replaced 006] From Code Foundation Models to Agents and Applications: A Comprehensive Survey and Practical Guide to Code Intelligence
- **分类: cs.SE; cs.CL**

- **简介: 该论文综述并实践指导代码大模型的发展，涵盖从数据到智能体的全生命周期。属于代码智能任务，旨在解决模型能力、训练方法与实际应用间的差距问题，通过系统分析与实验探索提升代码生成质量与实用性。**

- **链接: [https://arxiv.org/pdf/2511.18538v5](https://arxiv.org/pdf/2511.18538v5)**

> **作者:** Jian Yang; Xianglong Liu; Weifeng Lv; Ken Deng; Shawn Guo; Lin Jing; Yizhi Li; Shark Liu; Xianzhen Luo; Yuyu Luo; Changzai Pan; Ensheng Shi; Yingshui Tan; Renshuai Tao; Jiajun Wu; Xianjie Wu; Zhenhe Wu; Daoguang Zan; Chenchen Zhang; Wei Zhang; He Zhu; Terry Yue Zhuo; Kerui Cao; Xianfu Cheng; Jun Dong; Shengjie Fang; Zhiwei Fei; Xiangyuan Guan; Qipeng Guo; Zhiguang Han; Joseph James; Tianqi Luo; Renyuan Li; Yuhang Li; Yiming Liang; Congnan Liu; Jiaheng Liu; Qian Liu; Ruitong Liu; Tyler Loakman; Xiangxin Meng; Chuang Peng; Tianhao Peng; Jiajun Shi; Mingjie Tang; Boyang Wang; Haowen Wang; Yunli Wang; Fanglin Xu; Zihan Xu; Fei Yuan; Ge Zhang; Jiayi Zhang; Xinhao Zhang; Wangchunshu Zhou; Hualei Zhu; King Zhu; Bryan Dai; Aishan Liu; Zhoujun Li; Chenghua Lin; Tianyu Liu; Chao Peng; Kai Shen; Libo Qin; Shuangyong Song; Zizheng Zhan; Jiajun Zhang; Jie Zhang; Zhaoxiang Zhang; Bo Zheng
>
> **摘要:** Large language models (LLMs) have fundamentally transformed automated software development by enabling direct translation of natural language descriptions into functional code, driving commercial adoption through tools like Github Copilot (Microsoft), Cursor (Anysphere), Trae (ByteDance), and Claude Code (Anthropic). While the field has evolved dramatically from rule-based systems to Transformer-based architectures, achieving performance improvements from single-digit to over 95\% success rates on benchmarks like HumanEval. In this work, we provide a comprehensive synthesis and practical guide (a series of analytic and probing experiments) about code LLMs, systematically examining the complete model life cycle from data curation to post-training through advanced prompting paradigms, code pre-training, supervised fine-tuning, reinforcement learning, and autonomous coding agents. We analyze the code capability of the general LLMs (GPT-4, Claude, LLaMA) and code-specialized LLMs (StarCoder, Code LLaMA, DeepSeek-Coder, and QwenCoder), critically examining the techniques, design decisions, and trade-offs. Further, we articulate the research-practice gap between academic research (e.g., benchmarks and tasks) and real-world deployment (e.g., software-related code tasks), including code correctness, security, contextual awareness of large codebases, and integration with development workflows, and map promising research directions to practical needs. Last, we conduct a series of experiments to provide a comprehensive analysis of code pre-training, supervised fine-tuning, and reinforcement learning, covering scaling law, framework selection, hyperparameter sensitivity, model architectures, and dataset comparisons.
>
---
#### [replaced 007] Can Fine-Tuning Erase Your Edits? On the Fragile Coexistence of Knowledge Editing and Adaptation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究知识编辑与微调的共存问题，探讨微调是否会导致知识编辑失效。通过实验量化编辑衰减，发现微调会削弱编辑，且不同方法表现各异，提出保留或移除编辑的策略，强调需在完整应用流程中评估编辑效果。**

- **链接: [https://arxiv.org/pdf/2511.05852v3](https://arxiv.org/pdf/2511.05852v3)**

> **作者:** Yinjie Cheng; Paul Youssef; Christin Seifert; Jörg Schlötterer; Zhixue Zhao
>
> **摘要:** Knowledge editing has emerged as a lightweight alternative to retraining for correcting or injecting specific facts in large language models (LLMs). Meanwhile, fine-tuning remains the default operation for adapting LLMs to new domains and tasks. Despite their widespread adoption, these two post-training interventions have been studied in isolation, leaving open a crucial question: if we fine-tune an edited model, do the edits survive? This question is motivated by two practical scenarios: removing covert or malicious edits, and preserving beneficial edits. If fine-tuning impairs edits (Fig.1), current KE methods become less useful, as every fine-tuned model would require re-editing, which significantly increases the cost; if edits persist, fine-tuned models risk propagating hidden malicious edits, raising serious safety concerns. To this end, we systematically quantify edit decay after fine-tuning, investigating how fine-tuning affects knowledge editing. Our results show that edits decay after fine-tuning, with survival varying across configurations, e.g., AlphaEdit edits decay more than MEMIT edits. Further, we find that fine-tuning edited layers only can effectively remove edits, though at a slight cost to downstream performance. Surprisingly, fine-tuning non-edited layers impairs more edits than full fine-tuning. Overall, our study establishes empirical baselines and actionable strategies for integrating knowledge editing with fine-tuning, and underscores that evaluating model editing requires considering the full LLM application pipeline.
>
---
#### [replaced 008] Understanding Syntactic Generalization in Structure-inducing Language Models
- **分类: cs.CL**

- **简介: 该论文研究结构诱导语言模型（SiLM）的句法泛化能力，旨在理解其句法表示特性、语法判断性能及训练动态。作者从头训练三种SiLM架构，在多语言与合成数据上进行评估，发现GPST在长距离依赖任务中表现最优，小模型结合大量合成数据可有效评估模型基本属性。**

- **链接: [https://arxiv.org/pdf/2508.07969v2](https://arxiv.org/pdf/2508.07969v2)**

> **作者:** David Arps; Hassan Sajjad; Laura Kallmeyer
>
> **备注:** Code available at https://github.com/davidarps/silm
>
> **摘要:** Structure-inducing Language Models (SiLM) are trained on a self-supervised language modeling task, and induce a hierarchical sentence representation as a byproduct when processing an input. SiLMs couple strong syntactic generalization behavior with competitive performance on various NLP tasks, but many of their basic properties are yet underexplored. In this work, we train three different SiLM architectures from scratch: Structformer (Shen et al., 2021), UDGN (Shen et al., 2022), and GPST (Hu et al., 2024b). We train these architectures on both natural language (English, German, and Chinese) corpora and synthetic bracketing expressions. The models are then evaluated with respect to (i) properties of the induced syntactic representations (ii) performance on grammaticality judgment tasks, and (iii) training dynamics. We find that none of the three architectures dominates across all evaluation metrics. However, there are significant differences, in particular with respect to the induced syntactic representations. The Generative Pretrained Structured Transformer (GPST; Hu et al. 2024) performs most consistently across evaluation settings, and outperforms the other models on long-distance dependencies in bracketing expressions. Furthermore, our study shows that small models trained on large amounts of synthetic data provide a useful testbed for evaluating basic model properties.
>
---
#### [replaced 009] HybridNorm: Towards Stable and Efficient Transformer Training via Hybrid Normalization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文聚焦Transformer模型训练稳定性与性能的平衡问题，提出HybridNorm方法，结合Pre-Norm的稳定性和Post-Norm的高性能优势。通过QKV层用归一化、FFN用Post-Norm，提升梯度流动和模型鲁棒性，实验证明其在多种模型上优于传统方案。**

- **链接: [https://arxiv.org/pdf/2503.04598v4](https://arxiv.org/pdf/2503.04598v4)**

> **作者:** Zhijian Zhuo; Yutao Zeng; Ya Wang; Sijun Zhang; Jian Yang; Xiaoqing Li; Xun Zhou; Jinwen Ma
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Transformers have become the de facto architecture for a wide range of machine learning tasks, particularly in large language models (LLMs). Despite their remarkable performance, many challenges remain in training deep transformer networks, especially regarding the position of the layer normalization. While Pre-Norm structures facilitate more stable training owing to their stronger identity path, they often lead to suboptimal performance compared to Post-Norm. In this paper, we propose $\textbf{HybridNorm}$, a simple yet effective hybrid normalization strategy that integrates the advantages of both Pre-Norm and Post-Norm. Specifically, HybridNorm employs QKV normalization within the attention mechanism and Post-Norm in the feed-forward network (FFN) of each transformer block. We provide both theoretical insights and empirical evidence to demonstrate that HybridNorm improves the gradient flow and the model robustness. Extensive experiments on large-scale transformer models, including both dense and sparse variants, show that HybridNorm consistently outperforms both Pre-Norm and Post-Norm approaches across multiple benchmarks. These findings highlight the potential of HybridNorm as a more stable and effective technique for improving the training and performance of deep transformer models. Code is available at https://github.com/BryceZhuo/HybridNorm.
>
---
#### [replaced 010] Grounding Long-Context Reasoning with Contextual Normalization for Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文研究检索增强生成（RAG）中的上下文格式影响，提出上下文归一化方法，以提升长文本推理的稳定性和准确性，解决因格式差异导致的性能波动问题。**

- **链接: [https://arxiv.org/pdf/2510.13191v2](https://arxiv.org/pdf/2510.13191v2)**

> **作者:** Jiamin Chen; Yuchen Li; Xinyu Ma; Xinran Chen; Xiaokun Zhang; Shuaiqiang Wang; Chen Ma; Dawei Yin
>
> **摘要:** Retrieval-Augmented Generation (RAG) has become an essential approach for extending the reasoning and knowledge capacity of large language models (LLMs). While prior research has primarily focused on retrieval quality and prompting strategies, the influence of how the retrieved documents are framed, i.e., context format, remains underexplored. We show that seemingly superficial choices, such as delimiters or structural markers in key-value extraction, can induce substantial shifts in accuracy and stability, even when semantic content is identical. To systematically investigate this effect, we design controlled experiments that vary context density, delimiter styles, and positional placement, revealing the underlying factors that govern performance differences. Building on these insights, we introduce Contextual Normalization, a lightweight strategy that adaptively standardizes context representations before generation. Extensive experiments on both controlled and real-world RAG benchmarks across diverse settings demonstrate that the proposed strategy consistently improves robustness to order variation and strengthens long-context utilization. These findings underscore that reliable RAG depends not only on retrieving the right content, but also on how that content is presented, offering both new empirical evidence and a practical technique for better long-context reasoning.
>
---
#### [replaced 011] Latent Collaboration in Multi-Agent Systems
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究多智能体系统中的协作问题，旨在提升推理效率与性能。提出LatentMAS框架，实现无需训练的纯潜在空间协作，通过共享隐态工作记忆完成无损信息交换，显著提升准确率、降低计算开销。**

- **链接: [https://arxiv.org/pdf/2511.20639v2](https://arxiv.org/pdf/2511.20639v2)**

> **作者:** Jiaru Zou; Xiyuan Yang; Ruizhong Qiu; Gaotang Li; Katherine Tieu; Pan Lu; Ke Shen; Hanghang Tong; Yejin Choi; Jingrui He; James Zou; Mengdi Wang; Ling Yang
>
> **备注:** Project: https://github.com/Gen-Verse/LatentMAS
>
> **摘要:** Multi-agent systems (MAS) extend large language models (LLMs) from independent single-model reasoning to coordinative system-level intelligence. While existing LLM agents depend on text-based mediation for reasoning and communication, we take a step forward by enabling models to collaborate directly within the continuous latent space. We introduce LatentMAS, an end-to-end training-free framework that enables pure latent collaboration among LLM agents. In LatentMAS, each agent first performs auto-regressive latent thoughts generation through last-layer hidden embeddings. A shared latent working memory then preserves and transfers each agent's internal representations, ensuring lossless information exchange. We provide theoretical analyses establishing that LatentMAS attains higher expressiveness and lossless information preservation with substantially lower complexity than vanilla text-based MAS. In addition, empirical evaluations across 9 comprehensive benchmarks spanning math and science reasoning, commonsense understanding, and code generation show that LatentMAS consistently outperforms strong single-model and text-based MAS baselines, achieving up to 14.6% higher accuracy, reducing output token usage by 70.8%-83.7%, and providing 4x-4.3x faster end-to-end inference. These results demonstrate that our new latent collaboration framework enhances system-level reasoning quality while offering substantial efficiency gains without any additional training. Code and data are fully open-sourced at https://github.com/Gen-Verse/LatentMAS.
>
---
#### [replaced 012] Non-Collaborative User Simulators for Tool Agents
- **分类: cs.CL**

- **简介: 该论文聚焦工具代理的用户模拟任务，旨在解决现有模拟器过于合作、无法反映真实非协作用户行为的问题。作者提出一种新型用户模拟架构，可模拟四种非协作行为，并构建可扩展框架，用于评估和改进工具代理在复杂现实场景中的表现。**

- **链接: [https://arxiv.org/pdf/2509.23124v3](https://arxiv.org/pdf/2509.23124v3)**

> **作者:** Jeonghoon Shim; Woojung Song; Cheyon Jin; Seungwon KooK; Yohan Jo
>
> **备注:** 10 pages
>
> **摘要:** Tool agents interact with users through multi-turn dialogues to accomplish various tasks. Recent studies have adopted user simulation methods to develop these agents in multi-turn settings. However, existing user simulators tend to be agent-friendly, exhibiting only cooperative behaviors, which fails to train and test agents against non-collaborative users in the real world. To address this, we propose a novel user simulator architecture that simulates four categories of non-collaborative behaviors: requesting unavailable services, digressing into tangential conversations, expressing impatience, and providing incomplete utterances. Our user simulator can simulate challenging and natural non-collaborative behaviors while reliably delivering all intents and information necessary to accomplish the task. Our experiments on MultiWOZ and $τ$-bench reveal significant performance degradation in state-of-the-art tool agents when encountering non-collaborative users. We provide detailed analyses of agents' weaknesses under each non-collaborative condition, such as escalated hallucinations and dialogue breakdowns. Ultimately, we contribute an easily extensible user simulation framework to help the research community develop tool agents and preemptively diagnose them under challenging real-world conditions within their own services.
>
---
#### [replaced 013] A Practitioner's Guide to Multi-turn Agentic Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究多轮智能体强化学习中大语言模型的训练方法，旨在解决现有框架碎片化、设计选择不明确的问题。作者从环境、奖励和政策三方面系统分析，提出跨任务通用的训练方案，并通过实验验证其有效性。**

- **链接: [https://arxiv.org/pdf/2510.01132v2](https://arxiv.org/pdf/2510.01132v2)**

> **作者:** Ruiyi Wang; Prithviraj Ammanabrolu
>
> **摘要:** We study what actually works and what doesn't for training large language models as agents via multi-turn reinforcement learning. Despite rapid progress, existing frameworks and definitions are fragmented, and there is no systematic formulation or analysis of which design choices matter across tasks. We address this gap by first breaking down the design space into three inter-related pillars -- environment, reward, and policy -- and empirically derive a recipe for training LLM agents in situated textual domains. In particular, we test TextWorld and ALFWorld, popular domains for testing situated embodied reasoning, as well as SWE-Gym for more software engineering style tasks. (i) For the environment, we analyze the impacts of task complexity in terms of sizes of the state and action spaces as well as optimal solution length, finding that even simple environments within a domain can provide signal on how well an agent can generalize to more complex tasks. (ii) For the reward, we ablate relative reward sparsity, observing that while dense turn-level rewards accelerate training, performance and stability is highly dependent on the choice of RL algorithm. (iii) And for the agent's policy, we explore the interplay between reward sparsity and biased (PPO, GRPO) and unbiased (RLOO) policy gradient methods in addition to showing how to find the optimal Supervised Fine-tuning (SFT) to RL training ratio given a fixed budget. We distill these findings into a training recipe that guides co-design across the three pillars, facilitating research and practical efforts in multi-turn agentic RL. Code: https://github.com/pearls-lab/meow-tea-taro
>
---
#### [replaced 014] Is Self-Supervised Learning Enough to Fill in the Gap? A Study on Speech Inpainting
- **分类: cs.SD; cs.CL; eess.AS**

- **简介: 该论文研究语音修复任务，探索自监督学习（SSL）模型是否可直接用于填补缺失语音。作者基于HuBERT和HiFi-GAN构建两种微调策略，验证其在不同场景下的重建效果，表明SSL预训练能有效迁移至语音修复，无需额外标注数据。**

- **链接: [https://arxiv.org/pdf/2405.20101v2](https://arxiv.org/pdf/2405.20101v2)**

> **作者:** Ihab Asaad; Maxime Jacquelin; Olivier Perrotin; Laurent Girin; Thomas Hueber
>
> **备注:** Accepted for publication to Computer Speech and Language journal (to appear)
>
> **摘要:** Speech inpainting consists in reconstructing corrupted or missing speech segments using surrounding context, a process that closely resembles the pretext tasks in Self-Supervised Learning (SSL) for speech encoders. This study investigates using SSL-trained speech encoders for inpainting without any additional training beyond the initial pretext task, and simply adding a decoder to generate a waveform. We compare this approach to supervised fine-tuning of speech encoders for a downstream task -- here, inpainting. Practically, we integrate HuBERT as the SSL encoder and HiFi-GAN as the decoder in two configurations: (1) fine-tuning the decoder to align with the frozen pre-trained encoder's output and (2) fine-tuning the encoder for an inpainting task based on a frozen decoder's input. Evaluations are conducted under single- and multi-speaker conditions using in-domain datasets and out-of-domain datasets (including unseen speakers, diverse speaking styles, and noise). Both informed and blind inpainting scenarios are considered, where the position of the corrupted segment is either known or unknown. The proposed SSL-based methods are benchmarked against several baselines, including a text-informed method combining automatic speech recognition with zero-shot text-to-speech synthesis. Performance is assessed using objective metrics and perceptual evaluations. The results demonstrate that both approaches outperform baselines, successfully reconstructing speech segments up to 200 ms, and sometimes up to 400 ms. Notably, fine-tuning the SSL encoder achieves more accurate speech reconstruction in single-speaker settings, while a pre-trained encoder proves more effective for multi-speaker scenarios. This demonstrates that an SSL pretext task can transfer to speech inpainting, enabling successful speech reconstruction with a pre-trained encoder.
>
---
#### [replaced 015] AdaDetectGPT: Adaptive Detection of LLM-Generated Text with Statistical Guarantees
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **简介: 该论文属于文本检测任务，旨在区分人类与大语言模型生成的文本。针对现有基于logits的方法不足，提出AdaDetectGPT，通过自适应学习 witness 函数提升检测性能，并提供统计保证，实验显示其性能显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2510.01268v4](https://arxiv.org/pdf/2510.01268v4)**

> **作者:** Hongyi Zhou; Jin Zhu; Pingfan Su; Kai Ye; Ying Yang; Shakeel A O B Gavioli-Akilagun; Chengchun Shi
>
> **备注:** Accepted by NeurIPS2025
>
> **摘要:** We study the problem of determining whether a piece of text has been authored by a human or by a large language model (LLM). Existing state of the art logits-based detectors make use of statistics derived from the log-probability of the observed text evaluated using the distribution function of a given source LLM. However, relying solely on log probabilities can be sub-optimal. In response, we introduce AdaDetectGPT -- a novel classifier that adaptively learns a witness function from training data to enhance the performance of logits-based detectors. We provide statistical guarantees on its true positive rate, false positive rate, true negative rate and false negative rate. Extensive numerical studies show AdaDetectGPT nearly uniformly improves the state-of-the-art method in various combination of datasets and LLMs, and the improvement can reach up to 37\%. A python implementation of our method is available at https://github.com/Mamba413/AdaDetectGPT.
>
---
#### [replaced 016] Transparent and Coherent Procedural Mistake Detection
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究程序性错误检测（PMD）任务，旨在判断用户是否按文本步骤正确执行操作。为提升模型透明性与推理能力，提出生成视觉自我对话理由，并构建基于NLI的自动评估指标，结合VLM改进检测性能。**

- **链接: [https://arxiv.org/pdf/2412.11927v4](https://arxiv.org/pdf/2412.11927v4)**

> **作者:** Shane Storks; Itamar Bar-Yossef; Yayuan Li; Zheyuan Zhang; Jason J. Corso; Joyce Chai
>
> **备注:** EMNLP 2025
>
> **摘要:** Procedural mistake detection (PMD) is a challenging problem of classifying whether a human user (observed through egocentric video) has successfully executed a task (specified by a procedural text). Despite significant recent efforts, machine performance in the wild remains nonviable, and the reasoning processes underlying this performance are opaque. As such, we extend PMD to require generating visual self-dialog rationales to inform decisions. Given the impressive, mature image understanding capabilities observed in recent vision-and-language models (VLMs), we curate a suitable benchmark dataset for PMD based on individual frames. As our reformulation enables unprecedented transparency, we leverage a natural language inference (NLI) model to formulate two automated metrics for the coherence of generated rationales. We establish baselines for this reframed task, showing that VLMs struggle off-the-shelf, but with some trade-offs, their accuracy, coherence, and efficiency can be improved by incorporating these metrics into common inference and fine-tuning methods. Lastly, our multi-faceted metrics visualize common outcomes, highlighting areas for further improvement.
>
---
#### [replaced 017] PhyloLM : Inferring the Phylogeny of Large Language Models and Predicting their Performances in Benchmarks
- **分类: cs.CL; cs.LG; q-bio.PE**

- **简介: 该论文提出PhyloLM，将系统发育算法应用于大语言模型，通过输出相似性计算谱系距离，构建树状图揭示模型间关系，并预测其在基准测试中的性能，旨在无需训练细节即可评估模型发展与能力。**

- **链接: [https://arxiv.org/pdf/2404.04671v5](https://arxiv.org/pdf/2404.04671v5)**

> **作者:** Nicolas Yax; Pierre-Yves Oudeyer; Stefano Palminteri
>
> **备注:** The project code is available at https://github.com/Nicolas-Yax/PhyloLM . Published as https://iclr.cc/virtual/2025/poster/28195 at ICLR 2025. A code demo is available at https://colab.research.google.com/drive/1agNE52eUevgdJ3KL3ytv5Y9JBbfJRYqd
>
> **摘要:** This paper introduces PhyloLM, a method adapting phylogenetic algorithms to Large Language Models (LLMs) to explore whether and how they relate to each other and to predict their performance characteristics. Our method calculates a phylogenetic distance metric based on the similarity of LLMs' output. The resulting metric is then used to construct dendrograms, which satisfactorily capture known relationships across a set of 111 open-source and 45 closed models. Furthermore, our phylogenetic distance predicts performance in standard benchmarks, thus demonstrating its functional validity and paving the way for a time and cost-effective estimation of LLM capabilities. To sum up, by translating population genetic concepts to machine learning, we propose and validate a tool to evaluate LLM development, relationships and capabilities, even in the absence of transparent training information.
>
---
#### [replaced 018] The Oracle and The Prism: A Decoupled and Efficient Framework for Generative Recommendation Explanation
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于生成式推荐解释任务，旨在解决端到端模型中推荐与解释的性能冲突。作者提出Prism框架，解耦排序与解释生成，通过大模型（Oracle）生成高质量解释知识，蒸馏至小模型（Prism）实现高效、忠实且个性化的解释。**

- **链接: [https://arxiv.org/pdf/2511.16543v2](https://arxiv.org/pdf/2511.16543v2)**

> **作者:** Jiaheng Zhang; Daqiang Zhang
>
> **备注:** 12pages,3 figures
>
> **摘要:** The integration of Large Language Models (LLMs) into explainable recommendation systems often leads to a performance-efficiency trade-off in end-to-end architectures, where joint optimization of ranking and explanation can result in suboptimal compromises. To resolve this, we propose Prism, a novel decoupled framework that rigorously separates the recommendation process into a dedicated ranking stage and an explanation generation stage. This decomposition ensures that each component is optimized for its specific objective, eliminating inherent conflicts in coupled models. Inspired by knowledge distillation, Prism leverages a powerful, instruction-following teacher LLM (FLAN-T5-XXL) as an Oracle to produce high-fidelity explanatory knowledge. A compact, fine-tuned student model (BART-Base), the Prism, then specializes in synthesizing this knowledge into personalized explanations. Our extensive experiments on benchmark datasets reveal a key finding: the distillation process not only transfers knowledge but also acts as a noise filter. Our 140M-parameter Prism model significantly outperforms its 11B-parameter teacher in human evaluations of faithfulness and personalization, demonstrating an emergent ability to correct hallucinations present in the teacher's outputs. While achieving a 24x speedup and a 10x reduction in memory consumption, our analysis validates that decoupling, coupled with targeted distillation, provides an efficient and effective pathway to high-quality, and perhaps more importantly, trustworthy explainable recommendation.
>
---
#### [replaced 019] Shadow in the Cache: Unveiling and Mitigating Privacy Risks of KV-cache in LLM Inference
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文研究LLM推理中KV缓存的隐私泄露问题，属于安全与隐私任务。它提出三种攻击方法揭示风险，并设计轻量防御机制KV-Cloak，有效保护输入隐私，兼顾安全性和效率。**

- **链接: [https://arxiv.org/pdf/2508.09442v3](https://arxiv.org/pdf/2508.09442v3)**

> **作者:** Zhifan Luo; Shuo Shao; Su Zhang; Lijing Zhou; Yuke Hu; Chenxu Zhao; Zhihao Liu; Zhan Qin
>
> **备注:** This paper is accepted by Network and Distributed System Security Symposium (NDSS) 2026
>
> **摘要:** The Key-Value (KV) cache, which stores intermediate attention computations (Key and Value pairs) to avoid redundant calculations, is a fundamental mechanism for accelerating Large Language Model (LLM) inference. However, this efficiency optimization introduces significant yet underexplored privacy risks. This paper provides the first comprehensive analysis of these vulnerabilities, demonstrating that an attacker can reconstruct sensitive user inputs directly from the KV-cache. We design and implement three distinct attack vectors: a direct Inversion Attack, a more broadly applicable and potent Collision Attack, and a semantic-based Injection Attack. These methods demonstrate the practicality and severity of KV-cache privacy leakage issues. To mitigate this, we propose KV-Cloak, a novel, lightweight, and efficient defense mechanism. KV-Cloak uses a reversible matrix-based obfuscation scheme, combined with operator fusion, to secure the KV-cache. Our extensive experiments show that KV-Cloak effectively thwarts all proposed attacks, reducing reconstruction quality to random noise. Crucially, it achieves this robust security with virtually no degradation in model accuracy and minimal performance overhead, offering a practical solution for trustworthy LLM deployment.
>
---
#### [replaced 020] CryptoBench: A Dynamic Benchmark for Expert-Level Evaluation of LLM Agents in Cryptocurrency
- **分类: cs.CL**

- **简介: 该论文提出CryptoBench，针对LLM代理在加密货币领域的专业评估任务，解决现有基准无法应对高时效、对抗性信息环境等问题。作者构建了由专家设计的动态月度测试集，包含50道题，涵盖四类任务，系统评估模型的数据获取与预测能力，揭示当前模型存在检索与预测不平衡问题。**

- **链接: [https://arxiv.org/pdf/2512.00417v3](https://arxiv.org/pdf/2512.00417v3)**

> **作者:** Jiacheng Guo; Suozhi Huang; Zixin Yao; Yifan Zhang; Yifu Lu; Jiashuo Liu; Zihao Li; Nicholas Deng; Qixin Xiao; Jia Tian; Kanghong Zhan; Tianyi Li; Xiaochen Liu; Jason Ge; Chaoyang He; Kaixuan Huang; Lin Yang; Wenhao Huang; Mengdi Wang
>
> **摘要:** This paper introduces CryptoBench, the first expert-curated, dynamic benchmark designed to rigorously evaluate the real-world capabilities of Large Language Model (LLM) agents in the uniquely demanding and fast-paced cryptocurrency domain. Unlike general-purpose agent benchmarks for search and prediction, professional crypto analysis presents specific challenges: \emph{extreme time-sensitivity}, \emph{a highly adversarial information environment}, and the critical need to synthesize data from \emph{diverse, specialized sources}, such as on-chain intelligence platforms and real-time Decentralized Finance (DeFi) dashboards. CryptoBench thus serves as a much more challenging and valuable scenario for LLM agent assessment. To address these challenges, we constructed a live, dynamic benchmark featuring 50 questions per month, expertly designed by crypto-native professionals to mirror actual analyst workflows. These tasks are rigorously categorized within a four-quadrant system: Simple Retrieval, Complex Retrieval, Simple Prediction, and Complex Prediction. This granular categorization enables a precise assessment of an LLM agent's foundational data-gathering capabilities alongside its advanced analytical and forecasting skills. Our evaluation of ten LLMs, both directly and within an agentic framework, reveals a performance hierarchy and uncovers a failure mode. We observe a \textit{retrieval-prediction imbalance}, where many leading models, despite being proficient at data retrieval, demonstrate a pronounced weakness in tasks requiring predictive analysis. This highlights a problematic tendency for agents to appear factually grounded while lacking the deeper analytical capabilities to synthesize information.
>
---
#### [replaced 021] Simplex-Optimized Hybrid Ensemble for Large Language Model Text Detection Under Generative Distribution Drif
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文本检测任务，旨在解决大语言模型生成文本检测中因生成分布漂移导致的性能下降问题。提出一种基于概率单形融合的混合集成方法，结合监督模型、扰动得分与语言特征，提升跨模型与攻击场景下的检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.22153v2](https://arxiv.org/pdf/2511.22153v2)**

> **作者:** Sepyan Purnama Kristanto; Lutfi Hakim; Dianni Yusuf
>
> **备注:** 8 pages, 2 Figure, Politeknik Negeri Banyuwangi
>
> **摘要:** The widespread adoption of large language models (LLMs) has made it difficult to distinguish human writing from machine-produced text in many real applications. Detectors that were effective for one generation of models tend to degrade when newer models or modified decoding strategies are introduced. In this work, we study this lack of stability and propose a hybrid ensemble that is explicitly designed to cope with changing generator distributions. The ensemble combines three complementary components: a RoBERTa-based classifier fine-tuned for supervised detection, a curvature-inspired score based on perturbing the input and measuring changes in model likelihood, and a compact stylometric model built on hand-crafted linguistic features. The outputs of these components are fused on the probability simplex, and the weights are chosen via validation-based search. We frame this approach in terms of variance reduction and risk under mixtures of generators, and show that the simplex constraint provides a simple way to trade off the strengths and weaknesses of each branch. Experiments on a 30000 document corpus drawn from several LLM families including models unseen during training and paraphrased attack variants show that the proposed method achieves 94.2% accuracy and an AUC of 0.978. The ensemble also lowers false positives on scientific articles compared to strong baselines, which is critical in educational and research settings where wrongly flagging human work is costly
>
---
#### [replaced 022] Surveying the MLLM Landscape: A Meta-Review of Current Surveys
- **分类: cs.CL**

- **简介: 该论文属于综述性研究，旨在系统梳理现有MLLM（多模态大语言模型）的评测方法与基准。它分类分析了当前调查的工作、贡献与影响，比较评估方法，探讨伦理、安全与效率问题，并指出研究趋势与未来方向，为领域提供全面参考。**

- **链接: [https://arxiv.org/pdf/2409.18991v2](https://arxiv.org/pdf/2409.18991v2)**

> **作者:** Ming Li; Keyu Chen; Ziqian Bi; Ming Liu; Xinyuan Song; Zekun Jiang; Tianyang Wang; Benji Peng; Qian Niu; Junyu Liu; Jinlang Wang; Sen Zhang; Xuanhe Pan; Jiawei Xu; Pohsun Feng
>
> **备注:** The article consists of 22 pages, including 2 figures and 108 references. The paper provides a meta-review of surveys on Multimodal Large Language Models (MLLMs), categorizing findings into key areas such as evaluation, applications, security, and future directions
>
> **摘要:** The rise of Multimodal Large Language Models (MLLMs) has become a transformative force in the field of artificial intelligence, enabling machines to process and generate content across multiple modalities, such as text, images, audio, and video. These models represent a significant advancement over traditional unimodal systems, opening new frontiers in diverse applications ranging from autonomous agents to medical diagnostics. By integrating multiple modalities, MLLMs achieve a more holistic understanding of information, closely mimicking human perception. As the capabilities of MLLMs expand, the need for comprehensive and accurate performance evaluation has become increasingly critical. This survey aims to provide a systematic review of benchmark tests and evaluation methods for MLLMs, covering key topics such as foundational concepts, applications, evaluation methodologies, ethical concerns, security, efficiency, and domain-specific applications. Through the classification and analysis of existing literature, we summarize the main contributions and methodologies of various surveys, conduct a detailed comparative analysis, and examine their impact within the academic community. Additionally, we identify emerging trends and underexplored areas in MLLM research, proposing potential directions for future studies. This survey is intended to offer researchers and practitioners a comprehensive understanding of the current state of MLLM evaluation, thereby facilitating further progress in this rapidly evolving field.
>
---
#### [replaced 023] Semantic Faithfulness and Entropy Production Measures to Tame Your LLM Demons and Manage Hallucinations
- **分类: cs.AI; cs.CL; cs.IT; cs.LG; q-fin.CP**

- **简介: 该论文属LLM评估任务，旨在解决生成内容的忠实性与幻觉问题。提出基于信息论和热力学的语义忠实度（SF）与语义熵产生（SEP）两个无监督指标，通过建模问答过程中的主题转移矩阵并优化KL散度，量化模型输出的可靠性，并应用于10-K文件摘要评估。**

- **链接: [https://arxiv.org/pdf/2512.05156v2](https://arxiv.org/pdf/2512.05156v2)**

> **作者:** Igor Halperin
>
> **备注:** 23 pages, 6 figures
>
> **摘要:** Evaluating faithfulness of Large Language Models (LLMs) to a given task is a complex challenge. We propose two new unsupervised metrics for faithfulness evaluation using insights from information theory and thermodynamics. Our approach treats an LLM as a bipartite information engine where hidden layers act as a Maxwell demon controlling transformations of context $C $ into answer $A$ via prompt $Q$. We model Question-Context-Answer (QCA) triplets as probability distributions over shared topics. Topic transformations from $C$ to $Q$ and $A$ are modeled as transition matrices ${\bf Q}$ and ${\bf A}$ encoding the query goal and actual result, respectively. Our semantic faithfulness (SF) metric quantifies faithfulness for any given QCA triplet by the Kullback-Leibler (KL) divergence between these matrices. Both matrices are inferred simultaneously via convex optimization of this KL divergence, and the final SF metric is obtained by mapping the minimal divergence onto the unit interval [0,1], where higher scores indicate greater faithfulness. Furthermore, we propose a thermodynamics-based semantic entropy production (SEP) metric in answer generation, and show that high faithfulness generally implies low entropy production. The SF and SEP metrics can be used jointly or separately for LLM evaluation and hallucination control. We demonstrate our framework on LLM summarization of corporate SEC 10-K filings.
>
---
#### [replaced 024] RPRO: Ranked Preference Reinforcement Optimization for Enhancing Medical QA and Diagnostic Reasoning
- **分类: cs.CL**

- **简介: 该论文聚焦医疗问答与诊断推理任务，旨在提升大模型在临床场景中的推理准确性与可靠性。提出RPRO框架，结合强化学习与分组偏好排序优化，引入任务自适应模板和概率评估机制，自动识别并修正低质量推理链，显著提升小模型的性能，超越更大规模的专用医疗模型。**

- **链接: [https://arxiv.org/pdf/2509.00974v5](https://arxiv.org/pdf/2509.00974v5)**

> **作者:** Chia-Hsuan Hsu; Jun-En Ding; Hsin-Ling Hsu; Chih-Ho Hsu; Li-Hung Yao; Chun-Chieh Liao; Feng Liu; Fang-Ming Hung
>
> **摘要:** Medical question answering requires advanced reasoning that integrates domain knowledge with logical inference. However, existing large language models (LLMs) often generate reasoning chains that lack factual accuracy and clinical reliability. We propose Ranked Preference Reinforcement Optimization (RPRO), a novel framework that combines reinforcement learning with preference-driven reasoning refinement to enhance clinical chain-of-thought (CoT) performance. RPRO distinguishes itself from prior approaches by employing task-adaptive reasoning templates and a probabilistic evaluation mechanism that aligns model outputs with established clinical workflows, while automatically identifying and correcting low-quality reasoning chains. Unlike traditional pairwise preference methods, RPRO introduces a groupwise ranking optimization based on the Bradley--Terry model and incorporates KL-divergence regularization for stable training. Experiments on PubMedQA, MedQA-USMLE, and a real-world clinical dataset from Far Eastern Memorial Hospital (FEMH) demonstrate consistent improvements over strong baselines. Remarkably, our 2B-parameter model outperforms much larger 7B--20B models, including medical-specialized variants. These findings demonstrate that combining preference optimization with quality-driven refinement provides a scalable and clinically grounded approach to building more reliable medical LLMs.
>
---
#### [replaced 025] Topology Matters: Measuring Memory Leakage in Multi-Agent LLMs
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文研究多智能体大语言模型中图拓扑对记忆泄露的影响。提出MAMA框架，量化不同网络结构下的隐私泄露程度，揭示拓扑结构与泄露风险的关系，为设计安全的多智能体系统提供指导。**

- **链接: [https://arxiv.org/pdf/2512.04668v2](https://arxiv.org/pdf/2512.04668v2)**

> **作者:** Jinbo Liu; Defu Cao; Yifei Wei; Tianyao Su; Yuan Liang; Yushun Dong; Yue Zhao; Xiyang Hu
>
> **摘要:** Graph topology is a fundamental determinant of memory leakage in multi-agent LLM systems, yet its effects remain poorly quantified. We introduce MAMA (Multi-Agent Memory Attack), a framework that measures how network structure shapes leakage. MAMA operates on synthetic documents containing labeled Personally Identifiable Information (PII) entities, from which we generate sanitized task instructions. We execute a two-phase protocol: Engram (seeding private information into a target agent's memory) and Resonance (multi-round interaction where an attacker attempts extraction). Over up to 10 interaction rounds, we quantify leakage as the fraction of ground-truth PII recovered from attacking agent outputs via exact matching. We systematically evaluate six common network topologies (fully connected, ring, chain, binary tree, star, and star-ring), varying agent counts $n\in\{4,5,6\}$, attacker-target placements, and base models. Our findings reveal consistent patterns: fully connected graphs exhibit maximum leakage while chains provide strongest protection; shorter attacker-target graph distance and higher target centrality significantly increase vulnerability; leakage rises sharply in early rounds before plateauing; model choice shifts absolute leakage rates but preserves topology rankings; temporal/locational PII attributes leak more readily than identity credentials or regulated identifiers. These results provide the first systematic mapping from architectural choices to measurable privacy risk, yielding actionable guidance: prefer sparse or hierarchical connectivity, maximize attacker-target separation, limit node degree and network radius, avoid shortcuts bypassing hubs, and implement topology-aware access controls.
>
---
#### [replaced 026] AutoNeural: Co-Designing Vision-Language Models for NPU Inference
- **分类: cs.CL**

- **简介: 该论文针对Vision-Language模型在NPU上推理效率低的问题，提出AutoNeural架构。通过设计适配NPU的整数量化视觉主干和结合SSM的线性复杂度语言模型，实现低延迟、高效率的端侧多模态推理。**

- **链接: [https://arxiv.org/pdf/2512.02924v2](https://arxiv.org/pdf/2512.02924v2)**

> **作者:** Wei Chen; Liangmin Wu; Yunhai Hu; Zhiyuan Li; Zhiyuan Cheng; Yicheng Qian; Lingyue Zhu; Zhipeng Hu; Luoyi Liang; Qiang Tang; Zhen Liu; Han Yang
>
> **摘要:** While Neural Processing Units (NPUs) offer high theoretical efficiency for edge AI, state-of-the-art Vision--Language Models (VLMs) tailored for GPUs often falter on these substrates. We attribute this hardware-model mismatch to two primary factors: the quantization brittleness of Vision Transformers (ViTs) and the I/O-bound nature of autoregressive attention mechanisms, which fail to utilize the high arithmetic throughput of NPUs. To bridge this gap, we propose AutoNeural, an NPU-native VLM architecture co-designed for integer-only inference. We replace the standard ViT encoder with a MobileNetV5-style backbone utilizing depthwise separable convolutions, which ensures bounded activation distributions for stable INT4/8/16 quantization. Complementing this, our language backbone integrates State-Space Model (SSM) principles with Transformer layers, employing efficient gated convolutions to achieve linear-time complexity. This hybrid design eliminates the heavy memory I/O overhead of Key-Value caching during generation. Our approach delivers substantial efficiency gains, reducing quantization error of vision encoder by up to 7x and end-to-end latency by 14x compared to conventional baselines. The AutoNeural also delivers 3x decoding speed and 4x longer context window than the baseline. We validate these improvements via a real-world automotive case study on the Qualcomm SA8295P SoC, demonstrating real-time performance for cockpit applications. Our results highlight that rethinking model topology specifically for NPU constraints is a prerequisite for robust multi-modal edge intelligence.
>
---
#### [replaced 027] DZ-TDPO: Non-Destructive Temporal Alignment for Mutable State Tracking in Long-Context Dialogue
- **分类: cs.CL**

- **简介: 该论文针对长上下文对话中的状态惯性问题，提出DZ-TDPO框架，通过非破坏性对齐结合动态KL约束与时间注意力调节，实现用户意图演变下的状态跟踪。在MSC数据集上验证有效，兼顾对齐效果与模型通用能力。**

- **链接: [https://arxiv.org/pdf/2512.03704v2](https://arxiv.org/pdf/2512.03704v2)**

> **作者:** Yijun Liao
>
> **备注:** 25 pages, 3 figures, 17 tables. Code available at https://github.com/lyj20071013/DZ-TDPO
>
> **摘要:** Long-context dialogue systems suffer from State Inertia, where static constraints prevent models from resolving conflicts between evolving user intents and established historical context. To address this, we propose DZ-TDPO, a non-destructive alignment framework that synergizes conflict-aware dynamic KL constraints with a calibrated temporal attention bias. Experiments on the Multi-Session Chat (MSC) dataset demonstrate that DZ-TDPO achieves state-of-the-art win rates (55.4% on Phi-3.5) while maintaining robust zero-shot generalization. Our scaling analysis reveals a "Capacity-Stability Trade-off": while smaller models incur an "alignment tax" (perplexity surge) to overcome historical inertia, the larger Qwen2.5-7B model achieves 50.8% win rate with negligible perplexity overhead. This confirms that TAI can be alleviated via precise attention regulation rather than destructive weight updates, preserving general capabilities (MMLU) across model scales. Code and data are available: https://github.com/lyj20071013/DZ-TDPO
>
---
#### [replaced 028] General Exploratory Bonus for Optimistic Exploration in RLHF
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究强化学习中的人类反馈对齐任务，旨在解决现有探索奖励方法因正则化导致的探索偏向问题。作者提出通用探索奖励（GEB），理论证明其满足乐观探索原则，并在多种模型和设置下验证其优越性。**

- **链接: [https://arxiv.org/pdf/2510.03269v3](https://arxiv.org/pdf/2510.03269v3)**

> **作者:** Wendi Li; Changdae Oh; Sharon Li
>
> **摘要:** Optimistic exploration is central to improving sample efficiency in reinforcement learning with human feedback, yet existing exploratory bonus methods to incentivize exploration often fail to realize optimism. We provide a theoretical analysis showing that current formulations, under KL or $α$-divergence regularization, unintentionally bias exploration toward high-probability regions of the reference model, thereby reinforcing conservative behavior instead of promoting discovery of uncertain regions. To address this pitfall, we introduce the General Exploratory Bonus (GEB), a novel theoretical framework that provably satisfies the optimism principle. GEB counteracts divergence-induced bias via reference-dependent reward regulation and unifies prior heuristic bonuses as special cases, while extending naturally across the full $α$-divergence family. Empirically, GEB consistently outperforms baselines on alignment tasks across multiple divergence settings and large language model backbones. These results demonstrate that GEB offers both a principled and practical solution for optimistic exploration in RLHF.
>
---
#### [replaced 029] MUST-RAG: MUSical Text Question Answering with Retrieval Augmented Generation
- **分类: cs.CL; cs.AI; cs.IR; cs.LG**

- **简介: 该论文针对大语言模型在音乐问答中知识不足的问题，提出MusT-RAG框架，结合检索增强生成技术，构建音乐专用向量库MusWikiDB，并在微调与推理中引入上下文，提升模型在音乐文本问答中的性能。**

- **链接: [https://arxiv.org/pdf/2507.23334v2](https://arxiv.org/pdf/2507.23334v2)**

> **作者:** Daeyong Kwon; SeungHeon Doh; Juhan Nam
>
> **备注:** This is an earlier version of the paper - ArtistMus: A Globally Diverse, Artist-Centric Benchmark for Retrieval-Augmented Music Question Answering. The latest version is available at: (arXiv:2512.05430)
>
> **摘要:** Recent advancements in Large language models (LLMs) have demonstrated remarkable capabilities across diverse domains. While they exhibit strong zero-shot performance on various tasks, LLMs' effectiveness in music-related applications remains limited due to the relatively small proportion of music-specific knowledge in their training data. To address this limitation, we propose MusT-RAG, a comprehensive framework based on Retrieval Augmented Generation (RAG) to adapt general-purpose LLMs for text-only music question answering (MQA) tasks. RAG is a technique that provides external knowledge to LLMs by retrieving relevant context information when generating answers to questions. To optimize RAG for the music domain, we (1) propose MusWikiDB, a music-specialized vector database for the retrieval stage, and (2) utilizes context information during both inference and fine-tuning processes to effectively transform general-purpose LLMs into music-specific models. Our experiment demonstrates that MusT-RAG significantly outperforms traditional fine-tuning approaches in enhancing LLMs' music domain adaptation capabilities, showing consistent improvements across both in-domain and out-of-domain MQA benchmarks. Additionally, our MusWikiDB proves substantially more effective than general Wikipedia corpora, delivering superior performance and computational efficiency.
>
---
#### [replaced 030] Unilaw-R1: A Large Language Model for Legal Reasoning with Reinforcement Learning and Iterative Inference
- **分类: cs.CL**

- **简介: 该论文提出Unilaw-R1，一种用于法律推理的轻量级大模型。针对法律知识不足、推理不可靠和泛化弱问题，构建高质量推理数据集，采用SFT与强化学习两阶段训练，并推出评测基准Unilaw-R1-Eval，在多项法律任务中显著优于同规模模型。**

- **链接: [https://arxiv.org/pdf/2510.10072v2](https://arxiv.org/pdf/2510.10072v2)**

> **作者:** Hua Cai; Shuang Zhao; Liang Zhang; Xuli Shen; Qing Xu; Weilin Shen; Zihao Wen; Tianke Ban
>
> **摘要:** Reasoning-focused large language models (LLMs) are rapidly evolving across various domains, yet their capabilities in handling complex legal problems remains underexplored. In this paper, we introduce Unilaw-R1, a large language model tailored for legal reasoning. With a lightweight 7-billion parameter scale, Unilaw-R1 significantly reduces deployment cost while effectively tackling three core challenges in the legal domain: insufficient legal knowledge, unreliable reasoning logic, and weak business generalization. To address these issues, we first construct Unilaw-R1-Data, a high-quality dataset containing 17K distilled and screened chain-of-thought (CoT) samples. Based on this, we adopt a two-stage training strategy combining Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL), which significantly boosts the performance on complex legal reasoning tasks and supports interpretable decision-making in legal AI applications. To assess legal reasoning ability, we also introduce Unilaw-R1-Eval, a dedicated benchmark designed to evaluate models across single- and multi-choice legal tasks. Unilaw-R1 demonstrates strong results on authoritative benchmarks, outperforming all models of similar scale and achieving performance on par with the much larger DeepSeek-R1-Distill-Qwen-32B (54.9%). Following domain-specific training, it also showed significant gains on LawBench and LexEval, exceeding Qwen-2.5-7B-Instruct (46.6%) by an average margin of 6.6%.
>
---
#### [replaced 031] Deep Learning and Machine Learning, Advancing Big Data Analytics and Management: Handy Appetizer
- **分类: cs.CL; cs.LG**

- **简介: 该论文属综述任务，旨在解决初学者难以理解深度学习与大数据技术的问题。作者通过直观可视化和案例，讲解DL、ML及大数据管理技术，介绍经典模型与应用，指导实际使用，强调其对未来 workforce 的重要性。**

- **链接: [https://arxiv.org/pdf/2409.17120v2](https://arxiv.org/pdf/2409.17120v2)**

> **作者:** Benji Peng; Xuanhe Pan; Yizhu Wen; Ziqian Bi; Keyu Chen; Ming Li; Ming Liu; Qian Niu; Junyu Liu; Jinlang Wang; Sen Zhang; Jiawei Xu; Xinyuan Song; Zekun Jiang; Tianyang Wang; Pohsun Feng
>
> **备注:** This book contains 93 pages and 60 figures
>
> **摘要:** This book explores the role of Artificial Intelligence (AI), Machine Learning (ML), and Deep Learning (DL) in driving the progress of big data analytics and management. The book focuses on simplifying the complex mathematical concepts behind deep learning, offering intuitive visualizations and practical case studies to help readers understand how neural networks and technologies like Convolutional Neural Networks (CNNs) work. It introduces several classic models and technologies such as Transformers, GPT, ResNet, BERT, and YOLO, highlighting their applications in fields like natural language processing, image recognition, and autonomous driving. The book also emphasizes the importance of pre-trained models and how they can enhance model performance and accuracy, with instructions on how to apply these models in various real-world scenarios. Additionally, it provides an overview of key big data management technologies like SQL and NoSQL databases, as well as distributed computing frameworks such as Apache Hadoop and Spark, explaining their importance in managing and processing vast amounts of data. Ultimately, the book underscores the value of mastering deep learning and big data management skills as critical tools for the future workforce, making it an essential resource for both beginners and experienced professionals.
>
---
#### [replaced 032] The AI Productivity Index (APEX)
- **分类: econ.GN; cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于AI评估任务，旨在衡量前沿模型在投资银行、管理咨询等四个专业岗位上的经济任务执行能力。研究扩展了原有基准，增加测试样本并改进评分方法，发布新 leaderboard，并开源部分案例与评测工具。**

- **链接: [https://arxiv.org/pdf/2509.25721v5](https://arxiv.org/pdf/2509.25721v5)**

> **作者:** Bertie Vidgen; Abby Fennelly; Evan Pinnix; Julien Benchek; Daniyal Khan; Zach Richards; Austin Bridges; Calix Huang; Ben Hunsberger; Isaac Robinson; Akul Datta; Chirag Mahapatra; Dominic Barton; Cass R. Sunstein; Eric Topol; Brendan Foody; Osvald Nitski
>
> **摘要:** We present an extended version of the AI Productivity Index (APEX-v1-extended), a benchmark for assessing whether frontier models are capable of performing economically valuable tasks in four jobs: investment banking associate, management consultant, big law associate, and primary care physician (MD). This technical report details the extensions to APEX-v1, including an increase in the held-out evaluation set from n = 50 to n = 100 cases per job (n = 400 total) and updates to the grading methodology. We present a new leaderboard, where GPT5 (Thinking = High) remains the top performing model with a score of 67.0%. APEX-v1-extended shows that frontier models still have substantial limitations when performing typical professional tasks. To support further research, we are open sourcing n = 25 non-benchmark example cases per role (n = 100 total) along with our evaluation harness.
>
---
#### [replaced 033] LLM Output Homogenization is Task Dependent
- **分类: cs.CL; cs.CY**

- **简介: 该论文研究大语言模型输出同质化问题，提出其影响因任务而异。作者构建八类任务 taxonomy，定义任务锚定的功能多样性，设计相应评估与采样方法，在保持质量的同时提升必要多样性，实现任务依赖的同质化缓解。**

- **链接: [https://arxiv.org/pdf/2509.21267v2](https://arxiv.org/pdf/2509.21267v2)**

> **作者:** Shomik Jain; Jack Lanchantin; Maximilian Nickel; Karen Ullrich; Ashia Wilson; Jamelle Watson-Daniels
>
> **摘要:** A large language model can be less helpful if it exhibits output response homogenization. But whether two responses are considered homogeneous, and whether such homogenization is problematic, both depend on the task category. For instance, in objective math tasks, we often expect no variation in the final answer but anticipate variation in the problem-solving strategy. Whereas, for creative writing tasks, we may expect variation in key narrative components (e.g. plot, genre, setting, etc), beyond the vocabulary or embedding diversity produced by temperature-sampling. Previous work addressing output homogenization often fails to conceptualize diversity in a task-dependent way. We address this gap in the literature directly by making the following contributions. (1) We present a task taxonomy comprised of eight task categories that each have distinct concepts of output homogenization. (2) We introduce task-anchored functional diversity to better evaluate output homogenization. (3) We propose a task-anchored sampling technique that increases functional diversity for task categories where homogenization is undesired, while preserving it where it is desired. (4) We challenge the perceived existence of a diversity-quality trade-off by increasing functional diversity while maintaining response quality. Overall, we demonstrate how task dependence improves the evaluation and mitigation of output homogenization.
>
---
#### [replaced 034] Golden Touchstone: A Comprehensive Bilingual Benchmark for Evaluating Financial Large Language Models
- **分类: cs.CL; cs.CE**

- **简介: 该论文提出Golden Touchstone，一个中英双语金融大模型评测基准，旨在解决现有金融NLP任务评估覆盖不全、数据质量低等问题。构建了八项任务的双语数据集，评测主流模型表现，并开源了训练模型Touchstone-GPT，推动金融大模型发展。**

- **链接: [https://arxiv.org/pdf/2411.06272v2](https://arxiv.org/pdf/2411.06272v2)**

> **作者:** Xiaojun Wu; Junxi Liu; Huanyi Su; Zhouchi Lin; Yiyan Qi; Chengjin Xu; Jiajun Su; Jiajie Zhong; Fuwei Wang; Saizhuo Wang; Fengrui Hua; Jia Li; Jian Guo
>
> **备注:** Published in Findings of EMNLP 2025
>
> **摘要:** As large language models (LLMs) increasingly permeate the financial sector, there is a pressing need for a standardized method to comprehensively assess their performance. Existing financial benchmarks often suffer from limited language and task coverage, low-quality datasets, and inadequate adaptability for LLM evaluation. To address these limitations, we introduce Golden Touchstone, a comprehensive bilingual benchmark for financial LLMs, encompassing eight core financial NLP tasks in both Chinese and English. Developed from extensive open-source data collection and industry-specific demands, this benchmark thoroughly assesses models' language understanding and generation capabilities. Through comparative analysis of major models such as GPT-4o, Llama3, FinGPT, and FinMA, we reveal their strengths and limitations in processing complex financial information. Additionally, we open-source Touchstone-GPT, a financial LLM trained through continual pre-training and instruction tuning, which demonstrates strong performance on the bilingual benchmark but still has limitations in specific tasks. This research provides a practical evaluation tool for financial LLMs and guides future development and optimization. The source code for Golden Touchstone and model weight of Touchstone-GPT have been made publicly available at https://github.com/IDEA-FinAI/Golden-Touchstone.
>
---
#### [replaced 035] Why Chain of Thought Fails in Clinical Text Understanding
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究链式思维（CoT）在临床文本理解中的失效问题。针对电子健康记录的复杂性，作者评估了95个大模型在87项任务中的表现，发现多数模型使用CoT后性能下降，揭示其在提升可解释性的同时可能损害可靠性。**

- **链接: [https://arxiv.org/pdf/2509.21933v2](https://arxiv.org/pdf/2509.21933v2)**

> **作者:** Jiageng Wu; Kevin Xie; Bowen Gu; Nils Krüger; Kueiyu Joshua Lin; Jie Yang
>
> **摘要:** Large language models (LLMs) are increasingly being applied to clinical care, a domain where both accuracy and transparent reasoning are critical for safe and trustworthy deployment. Chain-of-thought (CoT) prompting, which elicits step-by-step reasoning, has demonstrated improvements in performance and interpretability across a wide range of tasks. However, its effectiveness in clinical contexts remains largely unexplored, particularly in the context of electronic health records (EHRs), the primary source of clinical documentation, which are often lengthy, fragmented, and noisy. In this work, we present the first large-scale systematic study of CoT for clinical text understanding. We assess 95 advanced LLMs on 87 real-world clinical text tasks, covering 9 languages and 8 task types. Contrary to prior findings in other domains, we observe that 86.3\% of models suffer consistent performance degradation in the CoT setting. More capable models remain relatively robust, while weaker ones suffer substantial declines. To better characterize these effects, we perform fine-grained analyses of reasoning length, medical concept alignment, and error profiles, leveraging both LLM-as-a-judge evaluation and clinical expert evaluation. Our results uncover systematic patterns in when and why CoT fails in clinical contexts, which highlight a critical paradox: CoT enhances interpretability but may undermine reliability in clinical text tasks. This work provides an empirical basis for clinical reasoning strategies of LLMs, highlighting the need for transparent and trustworthy approaches.
>
---
#### [replaced 036] Guided Query Refinement: Multimodal Hybrid Retrieval with Test-Time Optimization
- **分类: cs.CL**

- **简介: 该论文研究视觉文档检索任务，旨在解决大规模多模态模型效率低和模态差异问题。提出引导式查询优化（GQR），在测试时利用轻量文本检索器指导主检索器的查询优化，提升性能与效率，实现更优的帕累托前沿。**

- **链接: [https://arxiv.org/pdf/2510.05038v2](https://arxiv.org/pdf/2510.05038v2)**

> **作者:** Omri Uzan; Asaf Yehudai; Roi pony; Eyal Shnarch; Ariel Gera
>
> **摘要:** Multimodal encoders have pushed the boundaries of visual document retrieval, matching textual query tokens directly to image patches and achieving state-of-the-art performance on public benchmarks. Recent models relying on this paradigm have massively scaled the sizes of their query and document representations, presenting obstacles to deployment and scalability in real-world pipelines. Furthermore, purely vision-centric approaches may be constrained by the inherent modality gap still exhibited by modern vision-language models. In this work, we connect these challenges to the paradigm of hybrid retrieval, investigating whether a lightweight dense text retriever can enhance a stronger vision-centric model. Existing hybrid methods, which rely on coarse-grained fusion of ranks or scores, fail to exploit the rich interactions within each model's representation space. To address this, we introduce Guided Query Refinement (GQR), a novel test-time optimization method that refines a primary retriever's query embedding using guidance from a complementary retriever's scores. Through extensive experiments on visual document retrieval benchmarks, we demonstrate that GQR allows vision-centric models to match the performance of models with significantly larger representations, while being up to 14x faster and requiring 54x less memory. Our findings show that GQR effectively pushes the Pareto frontier for performance and efficiency in multimodal retrieval. We release our code at https://github.com/IBM/test-time-hybrid-retrieval
>
---
#### [replaced 037] A Systematic Assessment of Language Models with Linguistic Minimal Pairs in Chinese
- **分类: cs.CL**

- **简介: 该论文构建了中文最大语言学最小对立对基准ZhoBLiMP，训练多种中文语言模型，提出SLLN-LP度量以消除句子长度偏差，评估模型在语法现象上的表现，揭示当前模型在指代、量化和省略等结构上仍存在挑战。**

- **链接: [https://arxiv.org/pdf/2411.06096v2](https://arxiv.org/pdf/2411.06096v2)**

> **作者:** Yikang Liu; Yeting Shen; Hongao Zhu; Lilong Xu; Zhiheng Qian; Siyuan Song; Kejia Zhang; Jialong Tang; Pei Zhang; Baosong Yang; Rui Wang; Hai Hu
>
> **备注:** Accepted by TACL
>
> **摘要:** We present ZhoBLiMP, the largest linguistic minimal pair benchmark for Chinese, with over 100 paradigms, ranging from topicalization to the \textit{Ba} construction. We then train from scratch a suite of Chinese language models (LMs) with different tokenizers, parameter sizes, and token volumes, to study the learning curves of LMs on Chinese. To mitigate the biases introduced by unequal lengths of the sentences in a minimal pair, we propose a new metric named sub-linear length normalized log-probabilities (SLLN-LP). Using SLLN-LP as the metric, our results show that \textsc{Anaphor}, \textsc{Quantifiers}, and \textsc{Ellipsis} in Chinese are difficult for LMs even up to 32B parameters, and that SLLN-LP successfully mitigates biases in ZhoBLiMP, JBLiMP and BLiMP. We conclude that future evaluations should be more carefully designed to consider the intricate relations between linking functions, LMs, and targeted minimal pairs.
>
---
#### [replaced 038] Democratic or Authoritarian? Probing a New Dimension of Political Biases in Large Language Models
- **分类: cs.CL**

- **简介: 该论文探究大语言模型在民主与威权主义价值观上的政治偏见，提出结合F量表、FavScore和角色榜样探测的新方法。研究发现模型总体倾向民主，但中文提示下更认可威权人物，且常将其视为榜样，反映潜在意识形态偏向。**

- **链接: [https://arxiv.org/pdf/2506.12758v2](https://arxiv.org/pdf/2506.12758v2)**

> **作者:** David Guzman Piedrahita; Irene Strauss; Bernhard Schölkopf; Rada Mihalcea; Zhijing Jin
>
> **摘要:** As Large Language Models (LLMs) become increasingly integrated into everyday life and information ecosystems, concerns about their implicit biases continue to persist. While prior work has primarily examined socio-demographic and left--right political dimensions, little attention has been paid to how LLMs align with broader geopolitical value systems, particularly the democracy--authoritarianism spectrum. In this paper, we propose a novel methodology to assess such alignment, combining (1) the F-scale, a psychometric tool for measuring authoritarian tendencies, (2) FavScore, a newly introduced metric for evaluating model favorability toward world leaders, and (3) role-model probing to assess which figures are cited as general role-models by LLMs. We find that LLMs generally favor democratic values and leaders, but exhibit increased favorability toward authoritarian figures when prompted in Mandarin. Further, models are found to often cite authoritarian figures as role models, even outside explicit political contexts. These results shed light on ways LLMs may reflect and potentially reinforce global political ideologies, highlighting the importance of evaluating bias beyond conventional socio-political axes. Our code is available at: https://github.com/irenestrauss/Democratic-Authoritarian-Bias-LLMs.
>
---
#### [replaced 039] InfiGUI-G1: Advancing GUI Grounding with Adaptive Exploration Policy Optimization
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究GUI界面下的视觉语言模型操作任务，旨在解决自然语言指令与UI元素间语义对齐困难的问题。作者提出AEPO框架，通过自适应探索策略优化提升模型探索效率，在多个基准上显著超越现有方法。**

- **链接: [https://arxiv.org/pdf/2508.05731v2](https://arxiv.org/pdf/2508.05731v2)**

> **作者:** Yuhang Liu; Zeyu Liu; Shuanghe Zhu; Pengxiang Li; Congkai Xie; Jiasheng Wang; Xavier Hu; Xiaotian Han; Jianbo Yuan; Xinyao Wang; Shengyu Zhang; Hongxia Yang; Fei Wu
>
> **备注:** Accepted to AAAI 2026 (Oral Presentation)
>
> **摘要:** The emergence of Multimodal Large Language Models (MLLMs) has propelled the development of autonomous agents that operate on Graphical User Interfaces (GUIs) using pure visual input. A fundamental challenge is robustly grounding natural language instructions. This requires a precise spatial alignment, which accurately locates the coordinates of each element, and, more critically, a correct semantic alignment, which matches the instructions to the functionally appropriate UI element. Although Reinforcement Learning with Verifiable Rewards (RLVR) has proven to be effective at improving spatial alignment for these MLLMs, we find that inefficient exploration bottlenecks semantic alignment, which prevent models from learning difficult semantic associations. To address this exploration problem, we present Adaptive Exploration Policy Optimization (AEPO), a new policy optimization framework. AEPO employs a multi-answer generation strategy to enforce broader exploration, which is then guided by a theoretically grounded Adaptive Exploration Reward (AER) function derived from first principles of efficiency eta=U/C. Our AEPO-trained models, InfiGUI-G1-3B and InfiGUI-G1-7B, establish new state-of-the-art results across multiple challenging GUI grounding benchmarks, achieving significant relative improvements of up to 9.0% against the naive RLVR baseline on benchmarks designed to test generalization and semantic understanding. Resources are available at https://github.com/InfiXAI/InfiGUI-G1.
>
---
#### [replaced 040] I Learn Better If You Speak My Language: Understanding the Superior Performance of Fine-Tuning Large Language Models with LLM-Generated Responses
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究为何用大语言模型（LLM）生成的回答进行微调，效果优于人类生成回答。属于模型训练优化任务，旨在探究性能提升原因。工作发现：LLM对自身生成文本更“熟悉”，表现为更低的困惑度，从而提升学习效果和推理能力保持。**

- **链接: [https://arxiv.org/pdf/2402.11192v5](https://arxiv.org/pdf/2402.11192v5)**

> **作者:** Xuan Ren; Biao Wu; Lingqiao Liu
>
> **备注:** The paper has been accepted to EMNLP 2024 (Main Conference) there is a follow up paper: Efficiently Selecting Response Generation Strategies for Synthetic Data Construction by Self-Aligned Perplexity Note: This is a revised version of arXiv:2402.11192 (v1, submitted 17 Feb 2024)
>
> **摘要:** This paper explores an intriguing observation: fine-tuning a large language model (LLM) with responses generated by a LLM often yields better results than using responses generated by humans, particularly in reasoning tasks. We conduct an in-depth investigation to understand why this occurs. Contrary to the common belief that these instances is due to the more detailed nature of LLM-generated content, our study identifies another contributing factor: an LLM is inherently more "familiar" with LLM generated responses. This familiarity is evidenced by lower perplexity before fine-tuning. We design a series of experiments to understand the impact of the "familiarity" and our conclusion reveals that this "familiarity" significantly impacts learning performance. Training with LLM-generated responses not only enhances performance but also helps maintain the model's capabilities in other reasoning tasks after fine-tuning on a specific task.
>
---
#### [replaced 041] AgriGPT-VL: Agricultural Vision-Language Understanding Suite
- **分类: cs.CL**

- **简介: 该论文属于农业多模态理解任务，旨在解决农业领域专用模型、数据和评估缺失的问题。作者构建了大规模农业视觉-语言数据集Agri-3M-VL，训练了专业模型AgriGPT-VL，并推出评测基准AgriBench-VL-4K，全面提升农业多模态理解能力。**

- **链接: [https://arxiv.org/pdf/2510.04002v3](https://arxiv.org/pdf/2510.04002v3)**

> **作者:** Bo Yang; Yunkui Chen; Lanfei Feng; Yu Zhang; Xiao Xu; Jianyu Zhang; Nueraili Aierken; Runhe Huang; Hongjian Lin; Yibin Ying; Shijian Li
>
> **摘要:** Despite rapid advances in multimodal large language models, agricultural applications remain constrained by the scarcity of domain-tailored models, curated vision-language corpora, and rigorous evaluation. To address these challenges, we present the AgriGPT-VL Suite, a unified multimodal framework for agriculture. Our contributions are threefold. First, we introduce Agri-3M-VL, the largest vision-language corpus for agriculture to our knowledge, curated by a scalable multi-agent data generator; it comprises 1M image-caption pairs, 2M image-grounded VQA pairs, 50K expert-level VQA instances, and 15K GRPO reinforcement learning samples. Second, we develop AgriGPT-VL, an agriculture-specialized vision-language model trained via a progressive curriculum of textual grounding, multimodal shallow/deep alignment, and GRPO refinement. This method achieves strong multimodal reasoning while preserving text-only capability. Third, we establish AgriBench-VL-4K, a compact yet challenging evaluation suite with open-ended and image-grounded questions, paired with multi-metric evaluation and an LLM-as-a-judge framework. Experiments show that AgriGPT-VL outperforms leading general-purpose VLMs on AgriBench-VL-4K, achieving higher pairwise win rates in the LLM-as-a-judge evaluation. Meanwhile, it remains competitive on the text-only AgriBench-13K with no noticeable degradation of language ability. Ablation studies further confirm consistent gains from our alignment and GRPO refinement stages. We will open source all of the resources to support reproducible research and deployment in low-resource agricultural settings.
>
---
#### [replaced 042] OSVBench: Benchmarking LLMs on Specification Generation Tasks for Operating System Verification
- **分类: cs.CL; cs.AI; cs.OS; cs.PL; cs.SE**

- **简介: 该论文提出OSVBench，面向操作系统内核验证中的形式化规范生成任务，旨在评估大模型在长上下文、复杂代码生成场景下的能力。基于真实内核构建245个任务，实验表明现有大模型表现有限，凸显其在长上下文理解与程序合成上的不足。**

- **链接: [https://arxiv.org/pdf/2504.20964v2](https://arxiv.org/pdf/2504.20964v2)**

> **作者:** Shangyu Li; Juyong Jiang; Tiancheng Zhao; Jiasi Shen
>
> **摘要:** We introduce OSVBench, a new benchmark for evaluating Large Language Models (LLMs) on the task of generating complete formal specifications for verifying the functional correctness of operating system kernels. This benchmark is built upon a real-world operating system kernel, Hyperkernel, and consists of 245 complex specification generation tasks in total, each of which is a long-context task of about 20k-30k tokens. The benchmark formulates the specification generation task as a program synthesis problem confined to a domain for specifying states and transitions. This formulation is provided to LLMs through a programming model. The LLMs must be able to understand the programming model and verification assumptions before delineating the correct search space for syntax and semantics and generating formal specifications. Guided by the operating system's high-level functional description, the LLMs are asked to generate a specification that fully describes all correct states and transitions for a potentially buggy code implementation of the operating system. Experimental results with 12 state-of-the-art LLMs indicate limited performance of existing LLMs on the specification generation task for operating system verification. Significant disparities in their performance highlight differences in their ability to handle long-context code generation tasks. The code are available at https://github.com/lishangyu-hkust/OSVBench
>
---
#### [replaced 043] SFT Doesn't Always Hurt General Capabilities: Revisiting Domain-Specific Fine-Tuning in LLMs
- **分类: cs.CL**

- **简介: 该论文研究领域适配大模型时的性能权衡问题，旨在缓解监督微调导致通用能力下降的问题。提出小学习率可减轻退化，并设计Token-Adaptive Loss Reweighting方法，在保持专业性能的同时更好保留通用能力。**

- **链接: [https://arxiv.org/pdf/2509.20758v2](https://arxiv.org/pdf/2509.20758v2)**

> **作者:** Jiacheng Lin; Zhongruo Wang; Kun Qian; Tian Wang; Arvind Srinivasan; Hansi Zeng; Ruochen Jiao; Xie Zhou; Jiri Gesi; Dakuo Wang; Yufan Guo; Kai Zhong; Weiqi Zhang; Sujay Sanghavi; Changyou Chen; Hyokun Yun; Lihong Li
>
> **摘要:** Supervised Fine-Tuning (SFT) on domain-specific datasets is a common approach to adapt Large Language Models (LLMs) to specialized tasks but is often believed to degrade their general capabilities. In this work, we revisit this trade-off and present both empirical and theoretical insights. First, we show that SFT does not always hurt: using a smaller learning rate can substantially mitigate general performance degradation while preserving comparable target-domain performance. We then provide a theoretical analysis that explains these phenomena and further motivates a new method, Token-Adaptive Loss Reweighting (TALR). Building on this, and recognizing that smaller learning rates alone do not fully eliminate general-performance degradation in all cases, we evaluate a range of strategies for reducing general capability loss, including L2 regularization, LoRA, model averaging, FLOW, and our proposed TALR. Experimental results demonstrate that while no method completely eliminates the trade-off, TALR consistently outperforms these baselines in balancing domain-specific gains and general capabilities. Finally, we distill our findings into practical guidelines for adapting LLMs to new domains: (i) using a small learning rate to achieve a favorable trade-off, and (ii) when a stronger balance is further desired, adopt TALR as an effective strategy.
>
---
#### [replaced 044] PoSh: Using Scene Graphs To Guide LLMs-as-a-Judge For Detailed Image Descriptions
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属图像描述评估任务，旨在解决现有指标对长文本细节评价不足的问题。提出PoSh，利用场景图引导大语言模型评分，提升与人类判断的相关性，并构建新数据集DOCENT验证其有效性。**

- **链接: [https://arxiv.org/pdf/2510.19060v2](https://arxiv.org/pdf/2510.19060v2)**

> **作者:** Amith Ananthram; Elias Stengel-Eskin; Lorena A. Bradford; Julia Demarest; Adam Purvis; Keith Krut; Robert Stein; Rina Elster Pantalony; Mohit Bansal; Kathleen McKeown
>
> **备注:** 26 pages, 9 figures. Metric/benchmark available at https://github.com/amith-ananthram/posh
>
> **摘要:** While vision-language models (VLMs) have advanced into detailed image description, evaluation remains a challenge. Standard metrics (e.g. CIDEr, SPICE) were designed for short texts and tuned to recognize errors that are now uncommon, such as object misidentification. In contrast, long texts require sensitivity to attribute and relation attachments and scores that localize errors to particular text spans. In this work, we introduce PoSh, a metric for detailed image description that uses scene graphs as structured rubrics to guide LLMs-as-a-Judge, producing aggregate scores grounded in fine-grained errors (e.g. mistakes in compositional understanding). PoSh is replicable, interpretable and a better proxy for human raters than existing metrics (including GPT4o-as-a-Judge). To validate PoSh, we introduce a challenging new dataset, DOCENT. This novel benchmark contains artwork, paired with expert-written references, and model-generated descriptions, augmented with granular and coarse judgments of their quality from art history students. Thus, DOCENT enables evaluating both detailed image description metrics and detailed image description itself in a challenging new domain. We show that PoSh achieves stronger correlations (+0.05 Spearman $ρ$) with the human judgments in DOCENT than the best open-weight alternatives, is robust to image type (using CapArena, an existing dataset of web imagery) and is a capable reward function, outperforming standard supervised fine-tuning. Then, using PoSh, we characterize the performance of open and closed models in describing the paintings, sketches and statues in DOCENT and find that foundation models struggle to achieve full, error-free coverage of images with rich scene dynamics, establishing a demanding new task to gauge VLM progress. Through both PoSh and DOCENT, we hope to enable advances in important areas such as assistive text generation.
>
---
#### [replaced 045] mini-vec2vec: Scaling Universal Geometry Alignment with Linear Transformations
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于跨语言文本表示对齐任务，旨在解决无平行数据时嵌入空间对齐效率低、不稳定的问题。作者提出mini-vec2vec，通过伪平行向量匹配、线性变换拟合与迭代优化，实现高效、鲁棒且可解释的线性对齐方法。**

- **链接: [https://arxiv.org/pdf/2510.02348v3](https://arxiv.org/pdf/2510.02348v3)**

> **作者:** Guy Dar
>
> **摘要:** We build upon vec2vec, a procedure designed to align text embedding spaces without parallel data. vec2vec finds a near-perfect alignment, but it is expensive and unstable. We present mini-vec2vec, a simple and efficient alternative that requires substantially lower computational cost and is highly robust. Moreover, the learned mapping is a linear transformation. Our method consists of three main stages: a tentative matching of pseudo-parallel embedding vectors, transformation fitting, and iterative refinement. Our linear alternative exceeds the original instantiation of vec2vec by orders of magnitude in efficiency, while matching or exceeding their results. The method's stability and interpretable algorithmic steps facilitate scaling and unlock new opportunities for adoption in new domains and fields.
>
---
#### [replaced 046] Self-Improvement Towards Pareto Optimality: Mitigating Preference Conflicts in Multi-Objective Alignment
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于多目标对齐任务，旨在解决DPO中因偏好冲突导致的优化方向不一致问题。作者提出一种自改进DPO框架，通过模型自生成并筛选帕累托最优响应，实现更优的帕累托前沿对齐。**

- **链接: [https://arxiv.org/pdf/2502.14354v3](https://arxiv.org/pdf/2502.14354v3)**

> **作者:** Moxin Li; Yuantao Zhang; Wenjie Wang; Wentao Shi; Zhuo Liu; Fuli Feng; Tat-Seng Chua
>
> **备注:** ACL findings (2025)
>
> **摘要:** Multi-Objective Alignment (MOA) aims to align LLMs' responses with multiple human preference objectives, with Direct Preference Optimization (DPO) emerging as a prominent approach. However, we find that DPO-based MOA approaches suffer from widespread preference conflicts in the data, where different objectives favor different responses. This results in conflicting optimization directions, hindering the optimization on the Pareto Front. To address this, we propose to construct Pareto-optimal responses to resolve preference conflicts. To efficiently obtain and utilize such responses, we propose a self-improving DPO framework that enables LLMs to self-generate and select Pareto-optimal responses for self-supervised preference alignment. Extensive experiments on two datasets demonstrate the superior Pareto Front achieved by our framework compared to various baselines. Code is available at https://github.com/zyttt-coder/SIPO.
>
---
#### [replaced 047] Kimi-Dev: Agentless Training as Skill Prior for SWE-Agents
- **分类: cs.AI; cs.CL; cs.SE**

- **简介: 该论文聚焦软件工程中的代码生成任务，旨在提升LLM在SWE-bench上的表现。提出Agentless训练方法，构建技能先验，训练出Kimi-Dev模型，在workflow和agent框架中均取得优异效果，实现两类范式的融合与迁移。**

- **链接: [https://arxiv.org/pdf/2509.23045v3](https://arxiv.org/pdf/2509.23045v3)**

> **作者:** Zonghan Yang; Shengjie Wang; Kelin Fu; Wenyang He; Weimin Xiong; Yibo Liu; Yibo Miao; Bofei Gao; Yejie Wang; Yingwei Ma; Yanhao Li; Yue Liu; Zhenxing Hu; Kaitai Zhang; Shuyi Wang; Huarong Chen; Flood Sung; Yang Liu; Yang Gao; Zhilin Yang; Tianyu Liu
>
> **备注:** 68 pages. GitHub repo at https://github.com/MoonshotAI/Kimi-Dev
>
> **摘要:** Large Language Models (LLMs) are increasingly applied to software engineering (SWE), with SWE-bench as a key benchmark. Solutions are split into SWE-Agent frameworks with multi-turn interactions and workflow-based Agentless methods with single-turn verifiable steps. We argue these paradigms are not mutually exclusive: reasoning-intensive Agentless training induces skill priors, including localization, code edit, and self-reflection that enable efficient and effective SWE-Agent adaptation. In this work, we first curate the Agentless training recipe and present Kimi-Dev, an open-source SWE LLM achieving 60.4\% on SWE-bench Verified, the best among workflow approaches. With additional SFT adaptation on 5k publicly-available trajectories, Kimi-Dev powers SWE-Agents to 48.6\% pass@1, on par with that of Claude 3.5 Sonnet (241022 version). These results show that structured skill priors from Agentless training can bridge workflow and agentic frameworks for transferable coding agents.
>
---
#### [replaced 048] SimuHome: A Temporal- and Environment-Aware Benchmark for Smart Home LLM Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对智能家庭中大语言模型代理缺乏真实环境与评测基准的问题，提出SimuHome——基于Matter协议的时序感知仿真环境与600个任务的评测基准，用于评估代理在用户意图理解、状态验证和时序调度等方面的能力。**

- **链接: [https://arxiv.org/pdf/2509.24282v2](https://arxiv.org/pdf/2509.24282v2)**

> **作者:** Gyuhyeon Seo; Jungwoo Yang; Junseong Pyo; Nalim Kim; Jonggeun Lee; Yohan Jo
>
> **备注:** 10 pages
>
> **摘要:** Large Language Model (LLM) agents excel at multi-step, tool-augmented tasks. However, smart homes introduce distinct challenges, requiring agents to handle latent user intents, temporal dependencies, device constraints, scheduling, and more. The main bottlenecks for developing smart home agents with such capabilities include the lack of a realistic simulation environment where agents can interact with devices and observe the results, as well as a challenging benchmark to evaluate them. To address this, we introduce $\textbf{SimuHome}$, a time-accelerated home environment that simulates smart devices, supports API calls, and reflects changes in environmental variables. By building the simulator on the Matter protocol, the global industry standard for smart home communication, SimuHome provides a high-fidelity environment, and agents validated in SimuHome can be deployed on real Matter-compliant devices with minimal adaptation. We provide a challenging benchmark of 600 episodes across twelve user query types that require the aforementioned capabilities. Our evaluation of 16 agents under a unified ReAct framework reveals distinct capabilities and limitations across models. Models under 7B parameters exhibited negligible performance across all query types. Even GPT-4.1, the best-performing standard model, struggled with implicit intent inference, state verification, and particularly temporal scheduling. While reasoning models such as GPT-5.1 consistently outperformed standard models on every query type, they required over three times the average inference time, which can be prohibitive for real-time smart home applications. This highlights a critical trade-off between task performance and real-world practicality.
>
---
#### [replaced 049] Thinking on the Fly: Test-Time Reasoning Enhancement via Latent Thought Policy Optimization
- **分类: cs.CL; cs.AI**

- **简介: 该论文聚焦大模型推理任务，针对现有隐式推理在复杂任务上表现脆弱的问题，提出无需参数更新的测试时优化框架LTPO。通过在线策略梯度优化中间隐态向量，利用模型自身置信度信号指导搜索，显著提升复杂推理鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.04182v2](https://arxiv.org/pdf/2510.04182v2)**

> **作者:** Wengao Ye; Yan Liang; Lianlei Shan
>
> **摘要:** Recent advancements in Large Language Models (LLMs) have shifted from explicit Chain-of-Thought (CoT) reasoning to more efficient latent reasoning, where intermediate thoughts are represented as vectors rather than text. However, latent reasoning can be brittle on challenging, out-of-distribution tasks where robust reasoning is most critical. To overcome these limitations, we introduce Latent Thought Policy Optimization (LTPO), a parameter-free framework that enhances LLM reasoning entirely at test time, without requiring model parameter updates. LTPO treats intermediate latent "thought" vectors as dynamic parameters that are actively optimized for each problem instance. It employs an online policy gradient method guided by an intrinsic, confidence-based reward signal computed directly from the frozen LLM's own output distributions, eliminating the need for external supervision or expensive text generation during optimization. Extensive experiments on five reasoning benchmarks show that LTPO not only matches or surpasses strong baselines on standard tasks but also demonstrates remarkable robustness where others fail. Most notably, on highly challenging AIME benchmarks where existing latent reasoning baselines collapse to near-zero accuracy, LTPO delivers substantial improvements, showcasing a unique capability for complex reasoning.
>
---
#### [replaced 050] LMSpell: Neural Spell Checking for Low-Resource Languages
- **分类: cs.CL**

- **简介: 该论文聚焦低资源语言的拼写纠错任务，旨在解决现有预训练模型应用局限和缺乏系统比较的问题。作者评估了不同PLM在拼写纠错中的表现，发现大语言模型（LLM）在大数据集上表现更优，并发布了包含评估功能的工具包LMSpell，辅以僧伽罗语案例研究。**

- **链接: [https://arxiv.org/pdf/2512.05414v2](https://arxiv.org/pdf/2512.05414v2)**

> **作者:** Akesh Gunathilake; Nadil Karunarathne; Tharusha Bandaranayake; Nisansa de Silva; Surangika Ranathunga
>
> **摘要:** Spell correction is still a challenging problem for low-resource languages (LRLs). While pretrained language models (PLMs) have been employed for spell correction, their use is still limited to a handful of languages, and there has been no proper comparison across PLMs. We present the first empirical study on the effectiveness of PLMs for spell correction, which includes LRLs. We find that Large Language Models (LLMs) outperform their counterparts (encoder-based and encoder-decoder) when the fine-tuning dataset is large. This observation holds even in languages for which the LLM is not pre-trained. We release LMSpell, an easy- to use spell correction toolkit across PLMs. It includes an evaluation function that compensates for the hallucination of LLMs. Further, we present a case study with Sinhala to shed light on the plight of spell correction for LRLs.
>
---
#### [replaced 051] Learn the Ropes, Then Trust the Wins: Self-imitation with Progressive Exploration for Agentic Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.MA**

- **简介: 该论文研究代理式强化学习中的探索-利用平衡问题，提出SPEAR方法，通过渐进探索与自我模仿学习，结合课程调度优化策略熵，在工具使用任务中提升成功率，兼具高效性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2509.22601v4](https://arxiv.org/pdf/2509.22601v4)**

> **作者:** Yulei Qin; Xiaoyu Tan; Zhengbao He; Gang Li; Haojia Lin; Zongyi Li; Zihan Xu; Yuchen Shi; Siqi Cai; Renting Rui; Shaofei Cai; Yuzheng Cai; Xuan Zhang; Sheng Ye; Ke Li; Xing Sun
>
> **备注:** 45 pages, 14 figures
>
> **摘要:** Reinforcement learning (RL) is the dominant paradigm for sharpening strategic tool use capabilities of LLMs on long-horizon, sparsely-rewarded agent tasks, yet it faces a fundamental challenge of exploration-exploitation trade-off. Existing studies stimulate exploration through the lens of policy entropy, but such mechanical entropy maximization is prone to RL instability due to the multi-turn distribution shifting. In this paper, we target the progressive exploration-exploitation balance under the guidance of the agent's own experiences without succumbing to either entropy collapsing or runaway divergence. We propose SPEAR, a self-imitation learning (SIL) recipe for training agentic LLMs. It extends the vanilla SIL, where a replay buffer stores good experience for off-policy update, by gradually steering the policy entropy across stages. Specifically, the proposed curriculum scheduling harmonizes intrinsic reward shaping and self-imitation to 1) expedite exploration via frequent tool interactions at the beginning, and 2) strengthen exploitation of successful tactics upon convergence towards familiarity with the environment. We also combine bag-of-tricks of industrial RL optimizations for a strong baseline Dr.BoT to demonstrate our effectiveness. In ALFWorld and WebShop, SPEAR increases the success rates of GRPO/GiGPO/Dr.BoT by up to 16.1%/5.1%/8.6% and 20.7%/11.8%/13.9%, respectively. In AIME24 and AIME25, SPEAR boosts Dr.BoT by up to 3.8% and 6.1%, respectively. Such gains incur only 10%-25% extra theoretical complexity and negligible runtime overhead in practice, demonstrating the plug-and-play scalability of SPEAR.
>
---
#### [replaced 052] Eyes-on-Me: Scalable RAG Poisoning through Transferable Attention-Steering Attractors
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文针对检索增强生成（RAG）系统的数据投毒攻击扩展性差问题，提出Eyes-on-Me攻击方法。通过解耦生成可复用的注意力吸引子与语义区域，实现对多种RAG系统的高效、低成本攻击，显著提升成功率并揭示注意力集中与输出的强关联。**

- **链接: [https://arxiv.org/pdf/2510.00586v2](https://arxiv.org/pdf/2510.00586v2)**

> **作者:** Yen-Shan Chen; Sian-Yao Huang; Cheng-Lin Yang; Yun-Nung Chen
>
> **摘要:** Existing data poisoning attacks on retrieval-augmented generation (RAG) systems scale poorly because they require costly optimization of poisoned documents for each target phrase. We introduce Eyes-on-Me, a modular attack that decomposes an adversarial document into reusable Attention Attractors and Focus Regions. Attractors are optimized to direct attention to the Focus Region. Attackers can then insert semantic baits for the retriever or malicious instructions for the generator, adapting to new targets at near zero cost. This is achieved by steering a small subset of attention heads that we empirically identify as strongly correlated with attack success. Across 18 end-to-end RAG settings (3 datasets $\times$ 2 retrievers $\times$ 3 generators), Eyes-on-Me raises average attack success rates from 21.9 to 57.8 (+35.9 points, 2.6$\times$ over prior work). A single optimized attractor transfers to unseen black box retrievers and generators without retraining. Our findings establish a scalable paradigm for RAG data poisoning and show that modular, reusable components pose a practical threat to modern AI systems. They also reveal a strong link between attention concentration and model outputs, informing interpretability research.
>
---
#### [replaced 053] JELV: A Judge of Edit-Level Validity for Evaluation and Automated Reference Expansion in Grammatical Error Correction
- **分类: cs.CL**

- **简介: 该论文属语法纠错（GEC）任务，旨在解决参考答案多样性不足导致的评估偏差与模型泛化受限问题。作者提出JELV框架，自动判断修改编辑的有效性，并用于优化评估指标和扩展参考数据集，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2511.21700v2](https://arxiv.org/pdf/2511.21700v2)**

> **作者:** Yuhao Zhan; Yuqing Zhang; Jing Yuan; Qixiang Ma; Zhiqi Yang; Yu Gu; Zemin Liu; Fei Wu
>
> **摘要:** Existing Grammatical Error Correction (GEC) systems suffer from limited reference diversity, leading to underestimated evaluation and restricted model generalization. To address this issue, we introduce the Judge of Edit-Level Validity (JELV), an automated framework to validate correction edits from grammaticality, faithfulness, and fluency. Using our proposed human-annotated Pair-wise Edit-level Validity Dataset (PEVData) as benchmark, JELV offers two implementations: a multi-turn LLM-as-Judges pipeline achieving 90% agreement with human annotators, and a distilled DeBERTa classifier with 85% precision on valid edits. We then apply JELV to reclassify misjudged false positives in evaluation and derive a comprehensive evaluation metric by integrating false positive decoupling and fluency scoring, resulting in state-of-the-art correlation with human judgments. We also apply JELV to filter LLM-generated correction candidates, expanding the BEA19's single-reference dataset containing 38,692 source sentences. Retraining top GEC systems on this expanded dataset yields measurable performance gains. JELV provides a scalable solution for enhancing reference diversity and strengthening both evaluation and model generalization.
>
---
#### [replaced 054] Process Reward Models That Think
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究过程奖励模型（PRM），旨在减少其对标注数据的依赖。作者提出ThinkPRM，一种基于长思维链的生成式验证器，仅用1%标签数据即在多个数学与代码基准上超越现有方法，实现高效训练与测试时扩展。**

- **链接: [https://arxiv.org/pdf/2504.16828v5](https://arxiv.org/pdf/2504.16828v5)**

> **作者:** Muhammad Khalifa; Rishabh Agarwal; Lajanugen Logeswaran; Jaekyeom Kim; Hao Peng; Moontae Lee; Honglak Lee; Lu Wang
>
> **备注:** Add new ablation and minor writing fixes
>
> **摘要:** Step-by-step verifiers -- also known as process reward models (PRMs) -- are a key ingredient for test-time scaling. PRMs require step-level supervision, making them expensive to train. This work aims to build data-efficient PRMs as verbalized step-wise reward models that verify every step in the solution by generating a verification chain-of-thought (CoT). We propose ThinkPRM, a long CoT verifier fine-tuned on orders of magnitude fewer process labels than those required by discriminative PRMs. Our approach capitalizes on the inherent reasoning abilities of long CoT models, and outperforms LLM-as-a-Judge and discriminative verifiers -- using only 1% of the process labels in PRM800K -- across several challenging benchmarks. Specifically, ThinkPRM beats the baselines on ProcessBench, MATH-500, and AIME '24 under best-of-N selection and reward-guided search. In an out-of-domain evaluation on a subset of GPQA-Diamond and LiveCodeBench, our PRM surpasses discriminative verifiers trained on the full PRM800K by 8% and 4.5%, respectively. Lastly, under the same token budget, ThinkPRM scales up verification compute more effectively compared to LLM-as-a-Judge, outperforming it by 7.2% on a subset of ProcessBench. Our work highlights the value of generative, long CoT PRMs that can scale test-time compute for verification while requiring minimal supervision for training. Our code, data, and models are released at https://github.com/mukhal/thinkprm.
>
---
#### [replaced 055] SwiReasoning: Switch-Thinking in Latent and Explicit for Pareto-Superior Reasoning LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属大模型推理任务，旨在解决纯隐式推理导致的准确率低与过思考问题。提出SwiReasoning框架，通过置信度动态切换显式与隐式推理，并限制切换次数，提升准确率与token效率。**

- **链接: [https://arxiv.org/pdf/2510.05069v2](https://arxiv.org/pdf/2510.05069v2)**

> **作者:** Dachuan Shi; Abedelkadir Asi; Keying Li; Xiangchi Yuan; Leyan Pan; Wenke Lee; Wen Xiao
>
> **备注:** Code: https://github.com/sdc17/SwiReasoning, Website: https://swireasoning.github.io/
>
> **摘要:** Recent work shows that, beyond discrete reasoning through explicit chain-of-thought steps, which are limited by the boundaries of natural languages, large language models (LLMs) can also reason continuously in latent space, allowing richer information per step and thereby improving token efficiency. Despite this promise, latent reasoning still faces two challenges, especially in training-free settings: 1) purely latent reasoning broadens the search distribution by maintaining multiple implicit paths, which diffuses probability mass, introduces noise, and impedes convergence to a single high-confidence solution, thereby hurting accuracy; and 2) overthinking persists even without explicit text, wasting tokens and degrading efficiency. To address these issues, we introduce SwiReasoning, a training-free framework for LLM reasoning which features two key innovations: 1) SwiReasoning dynamically switches between explicit and latent reasoning, guided by block-wise confidence estimated from entropy trends in next-token distributions, to balance exploration and exploitation and promote timely convergence. 2) By limiting the maximum number of thinking-block switches, SwiReasoning curbs overthinking and improves token efficiency across varying problem difficulties. On widely used mathematics and STEM benchmarks, SwiReasoning consistently improves average accuracy by 1.5%-2.8% across reasoning LLMs of different model families and scales. Furthermore, under constrained budgets, SwiReasoning improves average token efficiency by 56%-79%, with larger gains as budgets tighten.
>
---
#### [replaced 056] Chopping Trees: Semantic Similarity Based Dynamic Pruning for Tree-of-Thought Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大模型推理优化任务，旨在解决Tree-of-Thought推理中语义冗余导致的计算开销问题。作者提出SSDP方法，通过在线语义相似性动态剪枝，实现实时冗余路径合并，在保持准确率的同时显著提升推理效率。**

- **链接: [https://arxiv.org/pdf/2511.08595v2](https://arxiv.org/pdf/2511.08595v2)**

> **作者:** Joongho Kim; Xirui Huang; Zarreen Reza; Gabriel Grand
>
> **备注:** 39th Conference on Neural Information Processing Systems (NeurIPS 2025) Workshop on Efficient Reasoning
>
> **摘要:** Tree-of-Thought (ToT) reasoning boosts the problem-solving abilities of Large Language Models (LLMs) but is computationally expensive due to semantic redundancy, where distinct branches explore equivalent reasoning paths. We introduce Semantic Similarity-Based Dynamic Pruning (SSDP), a lightweight method that, to the best of our knowledge, is the first framework to integrate online semantic merging into parallelized tree search, enabling the clustering and pruning of redundant steps in real time. Across reasoning benchmarks, including GSM8K and MATH500, SSDP achieves up to a 2.3x speedup over state-of-the-art tree-search baselines while maintaining competitive accuracy (typically within 5% of the strongest baseline) and reducing the number of explored nodes by 85-90%, demonstrating a practical approach to efficient, scalable LLM reasoning. The implementation of SSDP is publicly available at https://github.com/kimjoonghokim/SSDP.
>
---
#### [replaced 057] Beyond the Singular: Revealing the Value of Multiple Generations in Benchmark Evaluation
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于LLM评估任务，旨在解决因忽略生成随机性导致的评分不可靠问题。作者提出分层统计模型，利用多次生成提升评分准确性，降低方差，并定义提示难度分，构建数据图谱以检测错误和优化基准质量。**

- **链接: [https://arxiv.org/pdf/2502.08943v3](https://arxiv.org/pdf/2502.08943v3)**

> **作者:** Wenbo Zhang; Hengrui Cai; Wenyu Chen
>
> **备注:** Accepted in NeurIPS 2025 Workshop on LLM Evals
>
> **摘要:** Large language models (LLMs) have demonstrated significant utility in real-world applications, exhibiting impressive capabilities in natural language processing and understanding. Benchmark evaluations are crucial for assessing the capabilities of LLMs as they can provide a comprehensive assessment of their strengths and weaknesses. However, current evaluation methods often overlook the inherent randomness of LLMs by employing deterministic generation strategies or relying on a single random sample, resulting in unaccounted sampling variance and unreliable benchmark score estimates. In this paper, we propose a hierarchical statistical model that provides a more comprehensive representation of the benchmarking process by incorporating both benchmark characteristics and LLM randomness. We show that leveraging multiple generations improves the accuracy of estimating the benchmark score and reduces variance. Multiple generations also allow us to define $\mathbb P\left(\text{correct}\right)$, a prompt-level difficulty score based on correct ratios, providing fine-grained insights into individual prompts. Additionally, we create a data map that visualizes difficulty and semantics of prompts, enabling error detection and quality control in benchmark construction.
>
---
#### [replaced 058] Collaborative Gym: A Framework for Enabling and Evaluating Human-Agent Collaboration
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文提出Co-Gym框架，旨在促进和评估人与AI代理的协作。针对现有代理缺乏有效协作能力的问题，作者构建了支持人机双向交互的任务环境与评估体系，验证了协作在多任务中的优势，并揭示了当前模型在沟通与情境感知上的不足。**

- **链接: [https://arxiv.org/pdf/2412.15701v5](https://arxiv.org/pdf/2412.15701v5)**

> **作者:** Yijia Shao; Vinay Samuel; Yucheng Jiang; John Yang; Diyi Yang
>
> **备注:** Preprint
>
> **摘要:** While the advancement of large language models has spurred the development of AI agents to automate tasks, numerous use cases inherently require agents to collaborate with humans due to humans' latent preferences, domain expertise, or the need for control. To facilitate the study of human-agent collaboration, we introduce Collaborative Gym (Co-Gym), an open framework for developing and evaluating collaborative agents that engage in bidirectional communication with humans while interacting with task environments. We describe how the framework enables the implementation of new task environments and coordination between humans and agents through a flexible, non-turn-taking interaction paradigm, along with an evaluation suite that assesses both collaboration outcomes and processes. Our framework provides both a simulated condition with a reliable user simulator and a real-world condition with an interactive web application. Initial benchmark experiments across three representative tasks -- creating travel plans, writing related work sections, and analyzing tabular data -- demonstrate the benefits of human-agent collaboration: The best-performing collaborative agents consistently outperform their fully autonomous counterparts in task performance, achieving win rates of 86% in Travel Planning, 74% in Tabular Analysis, and 66% in Related Work when evaluated by real users. Despite these improvements, our evaluation reveals persistent limitations in current language models and agents, with communication and situational awareness failures observed in 65% and 40% of cases in the real condition, respectively. Released under the permissive MIT license, Co-Gym supports the addition of new task environments and can be used to develop collaborative agent applications, while its evaluation suite enables assessment and improvement of collaborative agents.
>
---
#### [replaced 059] DaLA: Danish Linguistic Acceptability Evaluation Guided by Real World Errors
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的语言可接受性判断任务，旨在解决现有丹麦语评估基准覆盖面不足的问题。作者分析真实错误，构建14种错误生成函数，创建更全面的评测基准DaLA，提升对大模型语言能力的区分度与评估难度。**

- **链接: [https://arxiv.org/pdf/2512.04799v2](https://arxiv.org/pdf/2512.04799v2)**

> **作者:** Gianluca Barmina; Nathalie Carmen Hau Norman; Peter Schneider-Kamp; Lukas Galke Poech
>
> **摘要:** We present an enhanced benchmark for evaluating linguistic acceptability in Danish. We first analyze the most common errors found in written Danish. Based on this analysis, we introduce a set of fourteen corruption functions that generate incorrect sentences by systematically introducing errors into existing correct Danish sentences. To ensure the accuracy of these corruptions, we assess their validity using both manual and automatic methods. The results are then used as a benchmark for evaluating Large Language Models on a linguistic acceptability judgement task. Our findings demonstrate that this extension is both broader and more comprehensive than the current state of the art. By incorporating a greater variety of corruption types, our benchmark provides a more rigorous assessment of linguistic acceptability, increasing task difficulty, as evidenced by the lower performance of LLMs on our benchmark compared to existing ones. Our results also suggest that our benchmark has a higher discriminatory power which allows to better distinguish well-performing models from low-performing ones.
>
---
#### [replaced 060] Evaluating Long-Term Memory for Long-Context Question Answering
- **分类: cs.CL**

- **简介: 该论文研究长上下文问答中的长期记忆机制，旨在评估不同记忆增强方法对对话连续性和推理能力的影响。通过比较多种记忆架构，探讨其在降低计算开销同时保持准确性的有效性，并分析不同模型适配的记忆类型。**

- **链接: [https://arxiv.org/pdf/2510.23730v2](https://arxiv.org/pdf/2510.23730v2)**

> **作者:** Alessandra Terranova; Björn Ross; Alexandra Birch
>
> **备注:** Accepted as a poster at Metacognition in Generative AI EurIPS workshop
>
> **摘要:** In order for large language models to achieve true conversational continuity and benefit from experiential learning, they need memory. While research has focused on the development of complex memory systems, it remains unclear which types of memory are most effective for long-context conversational tasks. We present a systematic evaluation of memory-augmented methods on long-context dialogues annotated for question-answering tasks that require diverse reasoning strategies. We analyse full-context prompting, semantic memory through retrieval-augmented generation and agentic memory, episodic memory through in-context learning, and procedural memory through prompt optimization. Our findings show that memory-augmented approaches reduce token usage by over 90\% while maintaining competitive accuracy. Memory architecture complexity should scale with model capability, with foundation models benefitting most from RAG, and stronger instruction-tuned models gaining from episodic learning through reflections and more complex agentic semantic memory. In particular, episodic memory can help LLMs recognise the limits of their own knowledge.
>
---
#### [replaced 061] LLMs are Biased Evaluators But Not Biased for Retrieval Augmented Generation
- **分类: cs.CL**

- **简介: 该论文研究大语言模型在检索增强生成（RAG）中的评估偏见问题，探究其是否偏好自身生成内容。通过模拟RAG的重排序与生成阶段，发现在事实性任务中，模型无显著自偏好，而更受事实准确性影响，结论在多数据集与模型上一致。**

- **链接: [https://arxiv.org/pdf/2410.20833v2](https://arxiv.org/pdf/2410.20833v2)**

> **作者:** Yen-Shan Chen; Jing Jin; Peng-Ting Kuo; Chao-Wei Huang; Yun-Nung Chen
>
> **备注:** 15 pages, 14 tables, 5 figures Accepted to ACL Findings 2025
>
> **摘要:** Recent studies have demonstrated that large language models (LLMs) exhibit significant biases in evaluation tasks, particularly in preferentially rating and favoring self-generated content. However, the extent to which this bias manifests in fact-oriented tasks, especially within retrieval-augmented generation (RAG) frameworks, where keyword extraction and factual accuracy take precedence over stylistic elements, remains unclear. Our study addresses this knowledge gap by simulating two critical phases of the RAG framework. In the first phase, LLMs evaluated human-authored and model-generated passages, emulating the \textit{pointwise reranking phase}. The second phase involves conducting pairwise reading comprehension tests to simulate the \textit{generation phase}. Contrary to previous findings indicating a self-preference in rating tasks, our results reveal no significant self-preference effect in RAG frameworks. Instead, we observe that factual accuracy significantly influences LLMs' output, even in the absence of prior knowledge. These findings are consistent among three common QA datasets (NQ, MARCO, TriviaQA Datasets) and 5 widely adopted language models (GPT-3.5, GPT-4o-mini, Gemini, LLaMA3, and Mistral). Our research contributes to the ongoing discourse on LLM biases and their implications for RAG-based system, offering insights that may inform the development of more robust and unbiased LLM systems.
>
---
#### [replaced 062] Rethinking LLM Training through Information Geometry and Quantum Metrics
- **分类: cs.CL; quant-ph**

- **简介: 该论文探讨大语言模型训练的几何本质，属优化理论研究。它利用信息几何和量子度量分析参数空间非欧结构，通过Fisher信息与量子类比，揭示优化动态、泛化及缩放规律，旨在深化对LLM训练机制的理解。**

- **链接: [https://arxiv.org/pdf/2506.15830v4](https://arxiv.org/pdf/2506.15830v4)**

> **作者:** Riccardo Di Sipio
>
> **备注:** 9 pages, 1 figure(s)
>
> **摘要:** Optimization in large language models (LLMs) unfolds over high-dimensional parameter spaces with non-Euclidean structure. Information geometry frames this landscape using the Fisher information metric, enabling more principled learning via natural gradient descent. Though often impractical, this geometric lens clarifies phenomena such as sharp minima, generalization, and observed scaling laws. We argue that curvature-based approaches deepen our understanding of LLM training. Finally, we speculate on quantum analogies based on the Fubini-Study metric and Quantum Fisher Information, hinting at efficient optimization in quantum-enhanced systems.
>
---
#### [replaced 063] Beyond Markovian: Reflective Exploration via Bayes-Adaptive RL for LLM Reasoning
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文研究大语言模型的推理能力提升，旨在解决传统强化学习缺乏反思探索的问题。作者提出基于贝叶斯自适应RL的BARL算法，通过不确定性感知策略激励模型主动获取信息，实现训练后仍有效的反思式推理，在数学与合成任务中提升了性能与效率。**

- **链接: [https://arxiv.org/pdf/2505.20561v2](https://arxiv.org/pdf/2505.20561v2)**

> **作者:** Shenao Zhang; Yaqing Wang; Yinxiao Liu; Tianqi Liu; Peter Grabowski; Eugene Ie; Zhaoran Wang; Yunxuan Li
>
> **摘要:** Large Language Models (LLMs) trained via Reinforcement Learning (RL) have exhibited strong reasoning capabilities and emergent reflective behaviors, such as rethinking and error correction, as a form of in-context exploration. However, the Markovian policy obtained from conventional RL training does not give rise to reflective exploration behaviors since the policy depends on the history only through the state and therefore has no incentive to enrich identical states with additional context. Instead, RL exploration is only useful during training to learn the optimal policy in a trial-and-error manner. Therefore, it remains unclear whether reflective reasoning will emerge during RL, or why it is beneficial. To remedy this, we recast reflective exploration within a Bayesian RL framework, which optimizes the expected return under a posterior distribution over Markov decision processes induced by the training data. This Bayesian formulation admits uncertainty-adaptive policies that, through belief updates, naturally incentivize information-gathering actions and induce self-reflection behaviors. Our resulting algorithm, BARL, instructs the LLM to stitch and switch strategies based on the observed outcomes, offering principled guidance on when and how the model should reflectively explore. Empirical results on both synthetic and mathematical reasoning tasks demonstrate that BARL outperforms conventional RL approaches, achieving superior test-time performance and token efficiency. Our code is available at https://github.com/shenao-zhang/BARL.
>
---
#### [replaced 064] SPOT: An Annotated French Corpus and Benchmark for Detecting Critical Interventions in Online Conversations
- **分类: cs.CL; cs.CY**

- **简介: 该论文提出SPOT，首个将“停止点”社会学概念转化为可复现NLP任务的法语语料库，旨在识别在线讨论中暂停或转向的关键干预评论。任务为二分类，标注4.3万条法语Facebook评论，结合上下文元数据，评估微调编码器与大模型表现，发现前者F1更高，发布数据与代码以促进研究。**

- **链接: [https://arxiv.org/pdf/2511.07405v2](https://arxiv.org/pdf/2511.07405v2)**

> **作者:** Manon Berriche; Célia Nouri; Chloé Clavel; Jean-Philippe Cointet
>
> **摘要:** We introduce SPOT (Stopping Points in Online Threads), the first annotated corpus translating the sociological concept of stopping point into a reproducible NLP task. Stopping points are ordinary critical interventions that pause or redirect online discussions through a range of forms (irony, subtle doubt or fragmentary arguments) that frameworks like counterspeech or social correction often overlook. We operationalize this concept as a binary classification task and provide reliable annotation guidelines. The corpus contains 43,305 manually annotated French Facebook comments linked to URLs flagged as false information by social media users, enriched with contextual metadata (article, post, parent comment, page or group, and source). We benchmark fine-tuned encoder models (CamemBERT) and instruction-tuned LLMs under various prompting strategies. Results show that fine-tuned encoders outperform prompted LLMs in F1 score by more than 10 percentage points, confirming the importance of supervised learning for emerging non-English social media tasks. Incorporating contextual metadata further improves encoder models F1 scores from 0.75 to 0.78. We release the anonymized dataset, along with the annotation guidelines and code in our code repository, to foster transparency and reproducible research.
>
---
#### [replaced 065] TRepLiNa: Layer-wise CKA+REPINA Alignment Improves Low-Resource Machine Translation in Aya-23 8B
- **分类: cs.CL; cs.AI**

- **简介: 该论文属低资源机器翻译任务，旨在解决印度小语种译文质量差的问题。作者提出TRepLiNa方法，结合CKA与REPINA，通过层间表示对齐和参数约束，在Aya-23 8B上提升零/少样本及微调场景的翻译性能。**

- **链接: [https://arxiv.org/pdf/2510.06249v4](https://arxiv.org/pdf/2510.06249v4)**

> **作者:** Toshiki Nakai; Ravi Kiran Chikkala; Lena Sophie Oberkircher; Nicholas Jennings; Natalia Skachkova; Tatiana Anikina; Jesujoba Oluwadara Alabi
>
> **备注:** It is work in progress
>
> **摘要:** The 2025 Multimodal Models for Low-Resource Contexts and Social Impact (MMLoSo) Language Challenge addresses one of India's most pressing linguistic gaps: the lack of resources for its diverse low-resource languages (LRLs). In this study, we investigate whether enforcing cross-lingual similarity in specific internal layers of a decoder-only multilingual large language model (LLM) can improve translation quality from LRL to high-resource language (HRL). Specifically, we combine Centered Kernel Alignment (CKA), a similarity metric that encourages representations of different languages to align, with REPINA, a regularization method that constrains parameter updates to remain close to the pretrained model, into a joint method we call TRepLiNa. In this research project, we experiment with zero-shot, few-shot, and fine-tuning settings using Aya-23 8B with QLoRA across MMLoSo shared task language pairs (Mundari, Santali, Bhili) with Hindi/English pivots. Our results show that aligning mid-level layers using TRepLiNa (CKA+REPINA) is a low-cost, practical approach to improving LRL translation, especially in data-scarce settings.
>
---
#### [replaced 066] Deep Learning and Machine Learning, Advancing Big Data Analytics and Management: Unveiling AI's Potential Through Tools, Techniques, and Applications
- **分类: cs.CL; cs.LG**

- **简介: 该论文综述人工智能、机器学习与深度学习在大数据分析中的应用，聚焦大语言模型与先进技术工具，探讨其在多领域的应用与伦理问题，旨在推动负责任的AI创新。**

- **链接: [https://arxiv.org/pdf/2410.01268v3](https://arxiv.org/pdf/2410.01268v3)**

> **作者:** Pohsun Feng; Ziqian Bi; Yizhu Wen; Xuanhe Pan; Benji Peng; Ming Liu; Jiawei Xu; Keyu Chen; Junyu Liu; Caitlyn Heqi Yin; Sen Zhang; Jinlang Wang; Qian Niu; Ming Li; Tianyang Wang; Xinyuan Song; Zekun Jiang
>
> **备注:** This book contains 155 pages and 9 figures
>
> **摘要:** Artificial intelligence (AI), machine learning, and deep learning have become transformative forces in big data analytics and management, enabling groundbreaking advancements across diverse industries. This article delves into the foundational concepts and cutting-edge developments in these fields, with a particular focus on large language models (LLMs) and their role in natural language processing, multimodal reasoning, and autonomous decision-making. Highlighting tools such as ChatGPT, Claude, and Gemini, the discussion explores their applications in data analysis, model design, and optimization. The integration of advanced algorithms like neural networks, reinforcement learning, and generative models has enhanced the capabilities of AI systems to process, visualize, and interpret complex datasets. Additionally, the emergence of technologies like edge computing and automated machine learning (AutoML) democratizes access to AI, empowering users across skill levels to engage with intelligent systems. This work also underscores the importance of ethical considerations, transparency, and fairness in the deployment of AI technologies, paving the way for responsible innovation. Through practical insights into hardware configurations, software environments, and real-world applications, this article serves as a comprehensive resource for researchers and practitioners. By bridging theoretical underpinnings with actionable strategies, it showcases the potential of AI and LLMs to revolutionize big data management and drive meaningful advancements across domains such as healthcare, finance, and autonomous systems.
>
---
#### [replaced 067] Bridging Relevance and Reasoning: Rationale Distillation in Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于检索增强生成（RAG）任务，旨在解决重排序模型与生成模型间因偏好差异导致的文档不匹配问题。作者提出RADIO框架，通过大模型提取回答依据（rationale），并以此进行理由驱动的重排序与模型对齐，提升生成效果。**

- **链接: [https://arxiv.org/pdf/2412.08519v3](https://arxiv.org/pdf/2412.08519v3)**

> **作者:** Pengyue Jia; Derong Xu; Xiaopeng Li; Zhaocheng Du; Xiangyang Li; Yichao Wang; Yuhao Wang; Qidong Liu; Maolin Wang; Huifeng Guo; Ruiming Tang; Xiangyu Zhao
>
> **备注:** Accepted to ACL 25 Findings
>
> **摘要:** The reranker and generator are two critical components in the Retrieval-Augmented Generation (i.e., RAG) pipeline, responsible for ranking relevant documents and generating responses. However, due to differences in pre-training data and objectives, there is an inevitable gap between the documents ranked as relevant by the reranker and those required by the generator to support answering the query. To address this gap, we propose RADIO, a novel and practical preference alignment framework with RAtionale DIstillatiOn. Specifically, we first propose a rationale extraction method that leverages the reasoning capabilities of Large Language Models (LLMs) to extract the rationales necessary for answering the query. Subsequently, a rationale-based alignment process is designed to rerank the documents based on the extracted rationales, and fine-tune the reranker to align the preferences. We conduct extensive experiments on two tasks across three datasets to demonstrate the effectiveness of our approach compared to baseline methods. Our code is released online to ease reproduction.
>
---
#### [replaced 068] Training-Free Diffusion Priors for Text-to-Image Generation via Optimization-based Visual Inversion
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究文本到图像生成任务，旨在解决扩散模型依赖昂贵训练先验的问题。提出无需训练的优化式视觉反演（OVI）方法，通过优化随机伪标记初始化的潜在表示，并引入新约束提升图像质量，实验证明其可替代传统先验。**

- **链接: [https://arxiv.org/pdf/2511.20821v2](https://arxiv.org/pdf/2511.20821v2)**

> **作者:** Samuele Dell'Erba; Andrew D. Bagdanov
>
> **备注:** 11 pages, 7 figures, technical report (preprint)
>
> **摘要:** Diffusion models have established the state-of-the-art in text-to-image generation, but their performance often relies on a diffusion prior network to translate text embeddings into the visual manifold for easier decoding. These priors are computationally expensive and require extensive training on massive datasets. In this work, we challenge the necessity of a trained prior at all by employing Optimization-based Visual Inversion (OVI), a training-free and data-free alternative, to replace the need for a prior. OVI initializes a latent visual representation from random pseudo-tokens and iteratively optimizes it to maximize the cosine similarity with input textual prompt embedding. We further propose two novel constraints, a Mahalanobis-based and a Nearest-Neighbor loss, to regularize the OVI optimization process toward the distribution of realistic images. Our experiments, conducted on Kandinsky 2.2, show that OVI can serve as an alternative to traditional priors. More importantly, our analysis reveals a critical flaw in current evaluation benchmarks like T2I-CompBench++, where simply using the text embedding as a prior achieves surprisingly high scores, despite lower perceptual quality. Our constrained OVI methods improve visual fidelity over this baseline, with the Nearest-Neighbor approach proving particularly effective, achieving quantitative scores comparable to or higher than the state-of-the-art data-efficient prior, indicating that the idea merits further investigation. The code will be publicly available upon acceptance.
>
---
#### [replaced 069] Exploring the Potential of Encoder-free Architectures in 3D LMMs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究3D大视觉语言模型中的编码器-free架构，旨在解决传统方法难以适应点云分辨率变化及语义不匹配问题。提出语义嵌入与几何聚合策略，实现首个无需预训练编码器的3D LMM，在多项任务上达到领先性能。**

- **链接: [https://arxiv.org/pdf/2502.09620v4](https://arxiv.org/pdf/2502.09620v4)**

> **作者:** Yiwen Tang; Zoey Guo; Zhuhao Wang; Ray Zhang; Qizhi Chen; Junli Liu; Delin Qu; Zhigang Wang; Dong Wang; Bin Zhao; Xuelong Li
>
> **摘要:** Encoder-free architectures have been preliminarily explored in the 2D Large Multimodal Models (LMMs), yet it remains an open question whether they can be effectively applied to 3D understanding scenarios. In this paper, we present the first comprehensive investigation into the potential of encoder-free architectures to alleviate the challenges of encoder-based 3D LMMs. These long-standing challenges include the failure to adapt to varying point cloud resolutions during inference and the point features from the encoder not meeting the semantic needs of Large Language Models (LLMs). We identify key aspects for 3D LMMs to remove the pre-trained encoder and enable the LLM to assume the role of the 3D encoder: 1) We propose the LLM-embedded Semantic Encoding strategy in the pre-training stage, exploring the effects of various point cloud self-supervised losses. And we present the Hybrid Semantic Loss to extract high-level semantics. 2) We introduce the Hierarchical Geometry Aggregation strategy in the instruction tuning stage. This incorporates inductive bias into the LLM layers to focus on the local details of the point clouds. To the end, we present the first Encoder-free 3D LMM, ENEL. Our 7B model rivals the state-of-the-art model, PointLLM-PiSA-13B, achieving 57.91%, 61.0%, and 55.20% on the classification, captioning, and VQA tasks, respectively. Our results show that the encoder-free architecture is highly promising for replacing encoder-based architectures in the field of 3D understanding. The code is released at https://github.com/Ivan-Tang-3D/ENEL
>
---
#### [replaced 070] Aligning Machiavellian Agents: Behavior Steering via Test-Time Policy Shaping
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究预训练AI代理在测试时的行为对齐问题，旨在不重训练的情况下通过策略塑造实现伦理属性控制。基于MACHIAVELLI基准验证，其方法可有效平衡奖励最大化与多维度伦理对齐。**

- **链接: [https://arxiv.org/pdf/2511.11551v3](https://arxiv.org/pdf/2511.11551v3)**

> **作者:** Dena Mujtaba; Brian Hu; Anthony Hoogs; Arslan Basharat
>
> **备注:** Accepted to AAAI 2026 AI Alignment Track
>
> **摘要:** The deployment of decision-making AI agents presents a critical challenge in maintaining alignment with human values or guidelines while operating in complex, dynamic environments. Agents trained solely to achieve their objectives may adopt harmful behavior, exposing a key trade-off between maximizing the reward function and maintaining alignment. For pre-trained agents, ensuring alignment is particularly challenging, as retraining can be a costly and slow process. This is further complicated by the diverse and potentially conflicting attributes representing the ethical values for alignment. To address these challenges, we propose a test-time alignment technique based on model-guided policy shaping. Our method allows precise control over individual behavioral attributes, generalizes across diverse reinforcement learning (RL) environments, and facilitates a principled trade-off between ethical alignment and reward maximization without requiring agent retraining. We evaluate our approach using the MACHIAVELLI benchmark, which comprises 134 text-based game environments and thousands of annotated scenarios involving ethical decisions. The RL agents are first trained to maximize the reward in their respective games. At test time, we apply policy shaping via scenario-action attribute classifiers to ensure decision alignment with ethical attributes. We compare our approach against prior training-time methods and general-purpose agents, as well as study several types of ethical violations and power-seeking behavior. Our results demonstrate that test-time policy shaping provides an effective and scalable solution for mitigating unethical behavior across diverse environments and alignment attributes.
>
---
