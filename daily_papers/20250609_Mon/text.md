# 自然语言处理 cs.CL

- **最新发布 105 篇**

- **更新 75 篇**

## 最新发布

#### [new 001] PuzzleWorld: A Benchmark for Multimodal, Open-Ended Reasoning in Puzzlehunts
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出了PuzzleWorld，一个包含667个谜题的多模态开放推理基准任务，旨在评估模型在无明确问题定义下的逐步推理、创造性思维和多模态理解能力。论文分析了当前模型在此类任务上的表现瓶颈，并通过精细标注与微调实验展示了提升推理能力的潜力。**

- **链接: [http://arxiv.org/pdf/2506.06211v1](http://arxiv.org/pdf/2506.06211v1)**

> **作者:** Hengzhi Li; Brendon Jiang; Alexander Naehu; Regan Song; Justin Zhang; Megan Tjandrasuwita; Chanakya Ekbote; Steven-Shine Chen; Adithya Balachandran; Wei Dai; Rebecca Chang; Paul Pu Liang
>
> **摘要:** Puzzlehunts are a genre of complex, multi-step puzzles lacking well-defined problem definitions. In contrast to conventional reasoning benchmarks consisting of tasks with clear instructions, puzzlehunts require models to discover the underlying problem structure from multimodal evidence and iterative reasoning, mirroring real-world domains such as scientific discovery, exploratory data analysis, or investigative problem-solving. Despite recent progress in foundation models, their performance on such open-ended settings remains largely untested. In this paper, we introduce PuzzleWorld, a large-scale benchmark of 667 puzzlehunt-style problems designed to assess step-by-step, open-ended, and creative multimodal reasoning. Each puzzle is annotated with the final solution, detailed reasoning traces, and cognitive skill labels, enabling holistic benchmarking and fine-grained diagnostic analysis. Most state-of-the-art models achieve only 1-2% final answer accuracy, with the best model solving only 14% of puzzles and reaching 40% stepwise accuracy. To demonstrate the value of our reasoning annotations, we show that fine-tuning a small model on reasoning traces improves stepwise reasoning from 4% to 11%, while training on final answers alone degrades performance to near zero. Our error analysis reveals that current models exhibit myopic reasoning, are bottlenecked by the limitations of language-based inference, and lack sketching capabilities crucial for visual and spatial reasoning. We release PuzzleWorld at https://github.com/MIT-MI/PuzzleWorld to support future work on building more general, open-ended, and creative reasoning systems.
>
---
#### [new 002] Writing-RL: Advancing Long-form Writing via Adaptive Curriculum Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的长文本写作能力。针对监督微调方法存在的数据饱和和学习能力受限问题，作者提出Writing-RL框架，采用自适应课程强化学习方法，包含样本筛选、奖励机制和动态调度策略。实验表明该方法显著优于基线模型，并展现了在长上下文推理任务上的泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.05760v1](http://arxiv.org/pdf/2506.05760v1)**

> **作者:** Xuanyu Lei; Chenliang Li; Yuning Wu; Kaiming Liu; Weizhou Shen; Peng Li; Ming Yan; Ji Zhang; Fei Huang; Yang Liu
>
> **备注:** Work in progress
>
> **摘要:** Recent advances in Large Language Models (LLMs) have enabled strong performance in long-form writing, yet existing supervised fine-tuning (SFT) approaches suffer from limitations such as data saturation and restricted learning capacity bounded by teacher signals. In this work, we present Writing-RL: an Adaptive Curriculum Reinforcement Learning framework to advance long-form writing capabilities beyond SFT. The framework consists of three key components: Margin-aware Data Selection strategy that prioritizes samples with high learning potential, Pairwise Comparison Reward mechanism that provides discriminative learning signals in the absence of verifiable rewards, and Dynamic Reference Scheduling approach, which plays a particularly critical role by adaptively adjusting task difficulty based on evolving model performance. Experiments on 7B-scale writer models show that our RL framework largely improves long-form writing performance over strong SFT baselines. Furthermore, we observe that models trained with long-output RL generalize surprisingly well to long-input reasoning tasks, potentially offering a promising perspective for rethinking long-context training.
>
---
#### [new 003] Cartridges: Lightweight and general-purpose long context representations via self-study
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型长上下文服务成本高的问题。提出Cartridges通过自研方法训练轻量KV缓存，在推理时加载以降低成本。实验表明其性能与ICL相当，但内存减少38.6倍，吞吐量提高26.4倍，并支持上下文组合。**

- **链接: [http://arxiv.org/pdf/2506.06266v1](http://arxiv.org/pdf/2506.06266v1)**

> **作者:** Sabri Eyuboglu; Ryan Ehrlich; Simran Arora; Neel Guha; Dylan Zinsley; Emily Liu; Will Tennien; Atri Rudra; James Zou; Azalia Mirhoseini; Christopher Re
>
> **摘要:** Large language models are often used to answer queries grounded in large text corpora (e.g. codebases, legal documents, or chat histories) by placing the entire corpus in the context window and leveraging in-context learning (ICL). Although current models support contexts of 100K-1M tokens, this setup is costly to serve because the memory consumption of the KV cache scales with input length. We explore an alternative: training a smaller KV cache offline on each corpus. At inference time, we load this trained KV cache, which we call a Cartridge, and decode a response. Critically, the cost of training a Cartridge can be amortized across all the queries referencing the same corpus. However, we find that the naive approach of training the Cartridge with next-token prediction on the corpus is not competitive with ICL. Instead, we propose self-study, a training recipe in which we generate synthetic conversations about the corpus and train the Cartridge with a context-distillation objective. We find that Cartridges trained with self-study replicate the functionality of ICL, while being significantly cheaper to serve. On challenging long-context benchmarks, Cartridges trained with self-study match ICL performance while using 38.6x less memory and enabling 26.4x higher throughput. Self-study also extends the model's effective context length (e.g. from 128k to 484k tokens on MTOB) and surprisingly, leads to Cartridges that can be composed at inference time without retraining.
>
---
#### [new 004] AgentSwift: Efficient LLM Agent Design via Value-guided Hierarchical Search
- **分类: cs.CL**

- **简介: 论文提出AgentSwift框架，旨在高效设计大型语言模型代理系统。针对现有方法在组件利用、评估成本和搜索效率上的不足，构建了层级搜索空间，引入预测价值模型与基于不确定性的MCTS策略。在多个任务中表现优于基线方法，提升了代理系统性能与搜索效率。**

- **链接: [http://arxiv.org/pdf/2506.06017v1](http://arxiv.org/pdf/2506.06017v1)**

> **作者:** Yu Li; Lehui Li; Zhihao Wu; Qingmin Liao; Jianye Hao; Kun Shao; Fengli Xu; Yong Li
>
> **备注:** 20pages
>
> **摘要:** Large language model (LLM) agents have demonstrated strong capabilities across diverse domains. However, designing high-performing agentic systems remains challenging. Existing agent search methods suffer from three major limitations: (1) an emphasis on optimizing agentic workflows while under-utilizing proven human-designed components such as memory, planning, and tool use; (2) high evaluation costs, as each newly generated agent must be fully evaluated on benchmarks; and (3) inefficient search in large search space. In this work, we introduce a comprehensive framework to address these challenges. First, We propose a hierarchical search space that jointly models agentic workflow and composable functional components, enabling richer agentic system designs. Building on this structured design space, we introduce a predictive value model that estimates agent performance given agentic system and task description, allowing for efficient, low-cost evaluation during the search process. Finally, we present a hierarchical Monte Carlo Tree Search (MCTS) strategy informed by uncertainty to guide the search. Experiments on seven benchmarks, covering embodied, math, web, tool, and game, show that our method achieves an average performance gain of 8.34\% over state-of-the-art baselines and exhibits faster search progress with steeper improvement trajectories. Code repo is available at https://github.com/Ericccc02/AgentSwift.
>
---
#### [new 005] semantic-features: A User-Friendly Tool for Studying Contextual Word Embeddings in Interpretable Semantic Spaces
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在研究语境对词嵌入的影响。作者开发了一个工具semantic-features，用于将上下文词嵌入映射到可解释的语义空间中。他们利用该工具测试“与格结构”（如“I sent London the letter”与“I sent the letter to London”）如何影响“London”一词是否被理解为有生命对象。论文通过构建450组句子对验证了模型在不同语境下的敏感性。**

- **链接: [http://arxiv.org/pdf/2506.06169v1](http://arxiv.org/pdf/2506.06169v1)**

> **作者:** Jwalanthi Ranganathan; Rohan Jha; Kanishka Misra; Kyle Mahowald
>
> **备注:** SCiL 2025 Camera Ready Extended Abstract
>
> **摘要:** We introduce semantic-features, an extensible, easy-to-use library based on Chronis et al. (2023) for studying contextualized word embeddings of LMs by projecting them into interpretable spaces. We apply this tool in an experiment where we measure the contextual effect of the choice of dative construction (prepositional or double object) on the semantic interpretation of utterances (Bresnan, 2007). Specifically, we test whether "London" in "I sent London the letter." is more likely to be interpreted as an animate referent (e.g., as the name of a person) than in "I sent the letter to London." To this end, we devise a dataset of 450 sentence pairs, one in each dative construction, with recipients being ambiguous with respect to person-hood vs. place-hood. By applying semantic-features, we show that the contextualized word embeddings of three masked language models show the expected sensitivities. This leaves us optimistic about the usefulness of our tool.
>
---
#### [new 006] Cross-lingual Collapse: How Language-Centric Foundation Models Shape Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多语言大模型在推理任务中出现的“跨语言崩溃”问题，即其推理链偏向预训练主导语言。作者通过在中文、韩文和乌克兰文数据上微调模型，分析语言不平衡问题，并探讨奖励机制对多语言推理的影响。**

- **链接: [http://arxiv.org/pdf/2506.05850v1](http://arxiv.org/pdf/2506.05850v1)**

> **作者:** Cheonbok Park; Jeonghoon Kim; Joosung Lee; Sanghwan Bae; Jaegul Choo; Kangmin Yoo
>
> **备注:** Preprint
>
> **摘要:** We identify \textbf{Cross-lingual Collapse}, a systematic drift in which the chain-of-thought (CoT) of a multilingual language model reverts to its dominant pre-training language even when the prompt is expressed in a different language. Recent large language models (LLMs) with reinforcement learning with verifiable reward (RLVR) have achieved strong logical reasoning performances by exposing their intermediate reasoning traces, giving rise to large reasoning models (LRMs). However, the mechanism behind multilingual reasoning in LRMs is not yet fully explored. To investigate the issue, we fine-tune multilingual LRMs with Group-Relative Policy Optimization (GRPO) on translated versions of the GSM$8$K and SimpleRL-Zoo datasets in three different languages: Chinese, Korean, and Ukrainian. During training, we monitor both task accuracy and language consistency of the reasoning chains. Our experiments reveal three key findings: (i) GRPO rapidly amplifies pre-training language imbalances, leading to the erosion of low-resource languages within just a few hundred updates; (ii) language consistency reward mitigates this drift but does so at the expense of an almost 5 - 10 pp drop in accuracy. and (iii) the resulting language collapse is severely damaging and largely irreversible, as subsequent fine-tuning struggles to steer the model back toward its original target-language reasoning capabilities. Together, these findings point to a remarkable conclusion: \textit{not all languages are trained equally for reasoning}. Furthermore, our paper sheds light on the roles of reward shaping, data difficulty, and pre-training priors in eliciting multilingual reasoning.
>
---
#### [new 007] Are Large Language Models Good Temporal Graph Learners?
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究大型语言模型（LLMs）在时序图学习中的应用，旨在解决动态网络中的链接预测问题。作者提出了TGTalker框架，利用时间邻域和结构信息生成自然语言输入LLM，并实现了与现有TGNN模型相当或更优的性能，同时提供可解释性。**

- **链接: [http://arxiv.org/pdf/2506.05393v1](http://arxiv.org/pdf/2506.05393v1)**

> **作者:** Shenyang Huang; Ali Parviz; Emma Kondrup; Zachary Yang; Zifeng Ding; Michael Bronstein; Reihaneh Rabbany; Guillaume Rabusseau
>
> **备注:** 9 pages, 9 tables, 4 figures
>
> **摘要:** Large Language Models (LLMs) have recently driven significant advancements in Natural Language Processing and various other applications. While a broad range of literature has explored the graph-reasoning capabilities of LLMs, including their use of predictors on graphs, the application of LLMs to dynamic graphs -- real world evolving networks -- remains relatively unexplored. Recent work studies synthetic temporal graphs generated by random graph models, but applying LLMs to real-world temporal graphs remains an open question. To address this gap, we introduce Temporal Graph Talker (TGTalker), a novel temporal graph learning framework designed for LLMs. TGTalker utilizes the recency bias in temporal graphs to extract relevant structural information, converted to natural language for LLMs, while leveraging temporal neighbors as additional information for prediction. TGTalker demonstrates competitive link prediction capabilities compared to existing Temporal Graph Neural Network (TGNN) models. Across five real-world networks, TGTalker performs competitively with state-of-the-art temporal graph methods while consistently outperforming popular models such as TGN and HTGN. Furthermore, TGTalker generates textual explanations for each prediction, thus opening up exciting new directions in explainability and interpretability for temporal link prediction. The code is publicly available at https://github.com/shenyangHuang/TGTalker.
>
---
#### [new 008] Automatically Detecting Amusing Games in Wordle
- **分类: cs.CL**

- **简介: 该论文旨在预测用户对Wordle游戏的幽默反应，属于自然语言处理与用户行为分析任务。研究者从Reddit抓取80k条用户反馈，利用GPT-3.5进行情感分类，并提取可预测用户愉悦感的游戏特征，验证计算方法能在一定程度上衡量Wordle中的创意幽默。**

- **链接: [http://arxiv.org/pdf/2506.05415v1](http://arxiv.org/pdf/2506.05415v1)**

> **作者:** Ronaldo Luo; Gary Liang; Cindy Liu; Adam Kabbara; Minahil Bakhtawar; Kina Kim; Michael Guerzhoy
>
> **备注:** Accepted to the Intenational Conference on Computational Creeativity (ICCC) 2025
>
> **摘要:** We explore automatically predicting which Wordle games Reddit users find amusing. We scrape approximately 80k reactions by Reddit users to Wordle games from Reddit, classify the reactions as expressing amusement or not using OpenAI's GPT-3.5 using few-shot prompting, and verify that GPT-3.5's labels roughly correspond to human labels. We then extract features from Wordle games that can predict user amusement. We demonstrate that the features indeed provide a (weak) signal that predicts user amusement as predicted by GPT-3.5. Our results indicate that user amusement at Wordle games can be predicted computationally to some extent. We explore which features of the game contribute to user amusement. We find that user amusement is predictable, indicating a measurable aspect of creativity infused into Wordle games through humor.
>
---
#### [new 009] Auto Review: Second Stage Error Detection for Highly Accurate Information Extraction from Phone Conversations
- **分类: cs.CL**

- **简介: 论文提出“Auto Review”系统，用于提升电话对话信息抽取的准确性。任务是医疗领域的信息提取，旨在解决语音识别错误和领域术语导致的信息抽取瓶颈问题。工作包括构建第二阶段后处理流程，结合多ASR结果与伪标签方法，减少人工复核，提升效率。**

- **链接: [http://arxiv.org/pdf/2506.05400v1](http://arxiv.org/pdf/2506.05400v1)**

> **作者:** Ayesha Qamar; Arushi Raghuvanshi; Conal Sathi; Youngseo Son
>
> **备注:** Accepted to ACL Industry track 2025
>
> **摘要:** Automating benefit verification phone calls saves time in healthcare and helps patients receive treatment faster. It is critical to obtain highly accurate information in these phone calls, as it can affect a patient's healthcare journey. Given the noise in phone call transcripts, we have a two-stage system that involves a post-call review phase for potentially noisy fields, where human reviewers manually verify the extracted data$\unicode{x2013}$a labor-intensive task. To automate this stage, we introduce Auto Review, which significantly reduces manual effort while maintaining a high bar for accuracy. This system, being highly reliant on call transcripts, suffers a performance bottleneck due to automatic speech recognition (ASR) issues. This problem is further exacerbated by the use of domain-specific jargon in the calls. In this work, we propose a second-stage postprocessing pipeline for accurate information extraction. We improve accuracy by using multiple ASR alternatives and a pseudo-labeling approach that does not require manually corrected transcripts. Experiments with general-purpose large language models and feature-based model pipelines demonstrate substantial improvements in the quality of corrected call transcripts, thereby enhancing the efficiency of Auto Review.
>
---
#### [new 010] Route-and-Reason: Scaling Large Language Model Reasoning with Reinforced Model Router
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模型协作推理任务，旨在解决大型语言模型在复杂任务中成本过高的问题。通过提出R2-Reasoner框架，动态分配子任务给不同规模模型，实现高效准确的协同推理。**

- **链接: [http://arxiv.org/pdf/2506.05901v1](http://arxiv.org/pdf/2506.05901v1)**

> **作者:** Chenyang Shao; Xinyang Liu; Yutang Lin; Fengli Xu; Yong Li
>
> **摘要:** Multi-step reasoning has proven essential for enhancing the problem-solving capabilities of Large Language Models (LLMs) by decomposing complex tasks into intermediate steps, either explicitly or implicitly. Extending the reasoning chain at test time through deeper thought processes or broader exploration, can furthur improve performance, but often incurs substantial costs due to the explosion in token usage. Yet, many reasoning steps are relatively simple and can be handled by more efficient smaller-scale language models (SLMs). This motivates hybrid approaches that allocate subtasks across models of varying capacities. However, realizing such collaboration requires accurate task decomposition and difficulty-aware subtask allocation, which is challenging. To address this, we propose R2-Reasoner, a novel framework that enables collaborative reasoning across heterogeneous LLMs by dynamically routing sub-tasks based on estimated complexity. At the core of our framework is a Reinforced Model Router, composed of a task decomposer and a subtask allocator. The task decomposer segments complex input queries into logically ordered subtasks, while the subtask allocator assigns each subtask to the most appropriate model, ranging from lightweight SLMs to powerful LLMs, balancing accuracy and efficiency. To train this router, we introduce a staged pipeline that combines supervised fine-tuning on task-specific datasets with Group Relative Policy Optimization algorithm, enabling self-supervised refinement through iterative reinforcement learning. Extensive experiments across four challenging benchmarks demonstrate that R2-Reasoner reduces API costs by 86.85% while maintaining or surpassing baseline accuracy. Our framework paves the way for more cost-effective and adaptive LLM reasoning. The code is open-source at https://anonymous.4open.science/r/R2_Reasoner .
>
---
#### [new 011] Explaining Matters: Leveraging Definitions and Semantic Expansion for Sexism Detection
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的 sexism 检测任务，旨在解决数据稀疏性和性别歧视语言的细微性带来的挑战。作者提出了两种基于提示的数据增强技术：定义引导的数据增强（DDA）和上下文语义扩展（CSE），并引入一种集成策略提升细粒度分类效果，在 EDOS 数据集上取得了最优性能。**

- **链接: [http://arxiv.org/pdf/2506.06238v1](http://arxiv.org/pdf/2506.06238v1)**

> **作者:** Sahrish Khan; Arshad Jhumka; Gabriele Pergola
>
> **备注:** Proceedings of the 2025 Annual Meeting of the Association for Computational Linguistics (ACL). ACL 2025 - Main Conference
>
> **摘要:** The detection of sexism in online content remains an open problem, as harmful language disproportionately affects women and marginalized groups. While automated systems for sexism detection have been developed, they still face two key challenges: data sparsity and the nuanced nature of sexist language. Even in large, well-curated datasets like the Explainable Detection of Online Sexism (EDOS), severe class imbalance hinders model generalization. Additionally, the overlapping and ambiguous boundaries of fine-grained categories introduce substantial annotator disagreement, reflecting the difficulty of interpreting nuanced expressions of sexism. To address these challenges, we propose two prompt-based data augmentation techniques: Definition-based Data Augmentation (DDA), which leverages category-specific definitions to generate semantically-aligned synthetic examples, and Contextual Semantic Expansion (CSE), which targets systematic model errors by enriching examples with task-specific semantic features. To further improve reliability in fine-grained classification, we introduce an ensemble strategy that resolves prediction ties by aggregating complementary perspectives from multiple language models. Our experimental evaluation on the EDOS dataset demonstrates state-of-the-art performance across all tasks, with notable improvements of macro F1 by 1.5 points for binary classification (Task A) and 4.1 points for fine-grained classification (Task C).
>
---
#### [new 012] RKEFino1: A Regulation Knowledge-Enhanced Large Language Model
- **分类: cs.CL; cs.AI**

- **简介: 论文提出RKEFino1，一种融合监管知识的金融大语言模型，用于解决数字监管报告中的准确性和合规性问题。基于Fino1并结合XBRL、CDM和MOF领域知识，模型通过知识问答、数学推理及数值NER任务提升金融合规任务效果。**

- **链接: [http://arxiv.org/pdf/2506.05700v1](http://arxiv.org/pdf/2506.05700v1)**

> **作者:** Yan Wang; Yueru He; Ruoyu Xiang; Jeff Zhao
>
> **摘要:** Recent advances in large language models (LLMs) hold great promise for financial applications but introduce critical accuracy and compliance challenges in Digital Regulatory Reporting (DRR). To address these issues, we propose RKEFino1, a regulation knowledge-enhanced financial reasoning model built upon Fino1, fine-tuned with domain knowledge from XBRL, CDM, and MOF. We formulate two QA tasks-knowledge-based and mathematical reasoning-and introduce a novel Numerical NER task covering financial entities in both sentences and tables. Experimental results demonstrate the effectiveness and generalization capacity of RKEFino1 in compliance-critical financial tasks. We have released our model on Hugging Face.
>
---
#### [new 013] EvidenceOutcomes: a Dataset of Clinical Trial Publications with Clinically Meaningful Outcomes
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决临床试验文献中“结局指标”提取不准确的问题。作者构建了高质量标注数据集EvidenceOutcomes，并训练PubMedBERT模型实现自动提取，在实体级和标记级分别取得0.69和0.76的F1值，为后续研究提供基准。**

- **链接: [http://arxiv.org/pdf/2506.05380v1](http://arxiv.org/pdf/2506.05380v1)**

> **作者:** Yiliang Zhou; Abigail M. Newbury; Gongbo Zhang; Betina Ross Idnay; Hao Liu; Chunhua Weng; Yifan Peng
>
> **摘要:** The fundamental process of evidence extraction and synthesis in evidence-based medicine involves extracting PICO (Population, Intervention, Comparison, and Outcome) elements from biomedical literature. However, Outcomes, being the most complex elements, are often neglected or oversimplified in existing benchmarks. To address this issue, we present EvidenceOutcomes, a novel, large, annotated corpus of clinically meaningful outcomes extracted from biomedical literature. We first developed a robust annotation guideline for extracting clinically meaningful outcomes from text through iteration and discussion with clinicians and Natural Language Processing experts. Then, three independent annotators annotated the Results and Conclusions sections of a randomly selected sample of 500 PubMed abstracts and 140 PubMed abstracts from the existing EBM-NLP corpus. This resulted in EvidenceOutcomes with high-quality annotations of an inter-rater agreement of 0.76. Additionally, our fine-tuned PubMedBERT model, applied to these 500 PubMed abstracts, achieved an F1-score of 0.69 at the entity level and 0.76 at the token level on the subset of 140 PubMed abstracts from the EBM-NLP corpus. EvidenceOutcomes can serve as a shared benchmark to develop and test future machine learning algorithms to extract clinically meaningful outcomes from biomedical abstracts.
>
---
#### [new 014] Bridging the Gap: In-Context Learning for Modeling Human Disagreement
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大型语言模型在主观分类任务中难以捕捉人类分歧的问题。通过上下文学习，在零样本和少样本设置下评估模型对多种标签建模策略的适应能力，并探讨提示设计与示例选择的影响。**

- **链接: [http://arxiv.org/pdf/2506.06113v1](http://arxiv.org/pdf/2506.06113v1)**

> **作者:** Benedetta Muscato; Yue Li; Gizem Gezici; Zhixue Zhao; Fosca Giannotti
>
> **摘要:** Large Language Models (LLMs) have shown strong performance on NLP classification tasks. However, they typically rely on aggregated labels-often via majority voting-which can obscure the human disagreement inherent in subjective annotations. This study examines whether LLMs can capture multiple perspectives and reflect annotator disagreement in subjective tasks such as hate speech and offensive language detection. We use in-context learning (ICL) in zero-shot and few-shot settings, evaluating four open-source LLMs across three label modeling strategies: aggregated hard labels, and disaggregated hard and soft labels. In few-shot prompting, we assess demonstration selection methods based on textual similarity (BM25, PLM-based), annotation disagreement (entropy), a combined ranking, and example ordering strategies (random vs. curriculum-based). Results show that multi-perspective generation is viable in zero-shot settings, while few-shot setups often fail to capture the full spectrum of human judgments. Prompt design and demonstration selection notably affect performance, though example ordering has limited impact. These findings highlight the challenges of modeling subjectivity with LLMs and the importance of building more perspective-aware, socially intelligent models.
>
---
#### [new 015] Let's CONFER: A Dataset for Evaluating Natural Language Inference Models on CONditional InFERence and Presupposition
- **分类: cs.CL**

- **简介: 该论文属于自然语言推理（NLI）任务，旨在评估模型对条件句中预设推理的处理能力。论文构建了新数据集CONFER，测试多个NLI模型及大语言模型在零样本和少样本下的表现，发现模型在条件句预设推理上表现不佳，且现有NLI数据微调效果有限。**

- **链接: [http://arxiv.org/pdf/2506.06133v1](http://arxiv.org/pdf/2506.06133v1)**

> **作者:** Tara Azin; Daniel Dumitrescu; Diana Inkpen; Raj Singh
>
> **备注:** This paper is published in the Proceedings of the 38th Canadian Conference on Artificial Intelligence (CAIAC 2025). Please cite the conference version at https://caiac.pubpub.org/pub/keh8ij01
>
> **摘要:** Natural Language Inference (NLI) is the task of determining whether a sentence pair represents entailment, contradiction, or a neutral relationship. While NLI models perform well on many inference tasks, their ability to handle fine-grained pragmatic inferences, particularly presupposition in conditionals, remains underexplored. In this study, we introduce CONFER, a novel dataset designed to evaluate how NLI models process inference in conditional sentences. We assess the performance of four NLI models, including two pre-trained models, to examine their generalization to conditional reasoning. Additionally, we evaluate Large Language Models (LLMs), including GPT-4o, LLaMA, Gemma, and DeepSeek-R1, in zero-shot and few-shot prompting settings to analyze their ability to infer presuppositions with and without prior context. Our findings indicate that NLI models struggle with presuppositional reasoning in conditionals, and fine-tuning on existing NLI datasets does not necessarily improve their performance.
>
---
#### [new 016] taz2024full: Analysing German Newspapers for Gender Bias and Discrimination across Decades
- **分类: cs.CL**

- **简介: 该论文构建了大型德语报纸语料库taz2024full，涵盖1980至2024年间的180万篇文章，旨在推动自然语言处理与计算社会科学的发展。其主要任务是通过分析性别呈现情况，探究数十年报道中的性别偏见与歧视问题，并提供结构化分析流程，为德语新闻文本中的语言趋势与社会议题研究奠定基础。**

- **链接: [http://arxiv.org/pdf/2506.05388v1](http://arxiv.org/pdf/2506.05388v1)**

> **作者:** Stefanie Urchs; Veronika Thurner; Matthias Aßenmacher; Christian Heumann; Stephanie Thiemichen
>
> **备注:** Accepted @ "63rd Annual Meeting of the Association for Computational Linguistics (ACL 2025)" as a findings paper. This is the author's version of the work. The definitive version of record will be published in the proceedings
>
> **摘要:** Open-access corpora are essential for advancing natural language processing (NLP) and computational social science (CSS). However, large-scale resources for German remain limited, restricting research on linguistic trends and societal issues such as gender bias. We present taz2024full, the largest publicly available corpus of German newspaper articles to date, comprising over 1.8 million texts from taz, spanning 1980 to 2024. As a demonstration of the corpus's utility for bias and discrimination research, we analyse gender representation across four decades of reporting. We find a consistent overrepresentation of men, but also a gradual shift toward more balanced coverage in recent years. Using a scalable, structured analysis pipeline, we provide a foundation for studying actor mentions, sentiment, and linguistic framing in German journalistic texts. The corpus supports a wide range of applications, from diachronic language analysis to critical media studies, and is freely available to foster inclusive and reproducible research in German-language NLP.
>
---
#### [new 017] When to use Graphs in RAG: A Comprehensive Analysis for Graph Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决图增强检索生成（GraphRAG）在实际应用中表现不佳的问题。作者构建了GraphRAG-Bench基准，系统评估图结构在知识检索与生成中的效果，揭示其优势场景并提供应用指南。**

- **链接: [http://arxiv.org/pdf/2506.05690v1](http://arxiv.org/pdf/2506.05690v1)**

> **作者:** Zhishang Xiang; Chuanjie Wu; Qinggang Zhang; Shengyuan Chen; Zijin Hong; Xiao Huang; Jinsong Su
>
> **摘要:** Graph retrieval-augmented generation (GraphRAG) has emerged as a powerful paradigm for enhancing large language models (LLMs) with external knowledge. It leverages graphs to model the hierarchical structure between specific concepts, enabling more coherent and effective knowledge retrieval for accurate reasoning.Despite its conceptual promise, recent studies report that GraphRAG frequently underperforms vanilla RAG on many real-world tasks. This raises a critical question: Is GraphRAG really effective, and in which scenarios do graph structures provide measurable benefits for RAG systems? To address this, we propose GraphRAG-Bench, a comprehensive benchmark designed to evaluate GraphRAG models onboth hierarchical knowledge retrieval and deep contextual reasoning. GraphRAG-Bench features a comprehensive dataset with tasks of increasing difficulty, coveringfact retrieval, complex reasoning, contextual summarization, and creative generation, and a systematic evaluation across the entire pipeline, from graph constructionand knowledge retrieval to final generation. Leveraging this novel benchmark, we systematically investigate the conditions when GraphRAG surpasses traditional RAG and the underlying reasons for its success, offering guidelines for its practical application. All related resources and analyses are collected for the community at https://github.com/GraphRAG-Bench/GraphRAG-Benchmark.
>
---
#### [new 018] Phonetically-Augmented Discriminative Rescoring for Voice Search Error Correction
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于语音搜索纠错任务，旨在解决因训练数据不足导致的罕见电影标题识别效果差的问题。论文提出了一种融合语音特征的重打分方法，通过生成语音替代选项并结合原始识别结果进行优化选择，有效降低了词错误率。**

- **链接: [http://arxiv.org/pdf/2506.06117v1](http://arxiv.org/pdf/2506.06117v1)**

> **作者:** Christophe Van Gysel; Maggie Wu; Lyan Verwimp; Caglar Tirkaz; Marco Bertola; Zhihong Lei; Youssef Oualil
>
> **备注:** To appear at Interspeech '25
>
> **摘要:** End-to-end (E2E) Automatic Speech Recognition (ASR) models are trained using paired audio-text samples that are expensive to obtain, since high-quality ground-truth data requires human annotators. Voice search applications, such as digital media players, leverage ASR to allow users to search by voice as opposed to an on-screen keyboard. However, recent or infrequent movie titles may not be sufficiently represented in the E2E ASR system's training data, and hence, may suffer poor recognition. In this paper, we propose a phonetic correction system that consists of (a) a phonetic search based on the ASR model's output that generates phonetic alternatives that may not be considered by the E2E system, and (b) a rescorer component that combines the ASR model recognition and the phonetic alternatives, and select a final system output. We find that our approach improves word error rate between 4.4 and 7.6% relative on benchmarks of popular movie titles over a series of competitive baselines.
>
---
#### [new 019] A Fictional Q&A Dataset for Studying Memorization and Knowledge Acquisition
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在研究语言模型对事实和文本序列的记忆机制。为解决现有数据难以区分两种记忆形式的问题，作者构建了一个包含虚构事件文本及问答对的合成数据集，并通过实验分析模型记忆特性，同时探讨了合成数据构建的挑战。**

- **链接: [http://arxiv.org/pdf/2506.05639v1](http://arxiv.org/pdf/2506.05639v1)**

> **作者:** John Kirchenbauer; Janny Mongkolsupawan; Yuxin Wen; Tom Goldstein; Daphne Ippolito
>
> **备注:** 10 pages and 8 figures in the main body
>
> **摘要:** When language models are trained on textual data, they acquire both knowledge about the structure of language as well as knowledge of facts about the world. At inference time, their knowledge of facts can be leveraged to solve interesting problems and perform useful knowledge work for users. It is well known that language models can verbatim memorize long sequences from their training data. However, it is much less well understood how language models memorize facts seen during training. In this work, we propose a new dataset to specifically empower researchers to study the dual processes of fact memorization and verbatim sequence memorization. The dataset consists of synthetically-generated, webtext-like documents about fictional events, as well as question-answer pairs about the events. We conduct training experiments showing how synthetic data about fictional events can be effective in teasing apart different forms of memorization. We also document the challenges in effectively building realistic, fictional synthetic data.
>
---
#### [new 020] LengClaro2023: A Dataset of Administrative Texts in Spanish with Plain Language adaptations
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决法律行政文本理解困难的问题。作者构建了LengClaro2023数据集，包含西班牙语的法律行政文本及其两种简化版本，用于评估自动文本简化系统的效果。**

- **链接: [http://arxiv.org/pdf/2506.05927v1](http://arxiv.org/pdf/2506.05927v1)**

> **作者:** Belén Agüera-Marco; Itziar Gonzalez-Dios
>
> **备注:** In this report, we present a part of the master thesis written by Bel\'en Ag\"uera Marco in order to obtain the B.S. Language Analysis and Processing at the University of the Basque Country (UPV/EHU), supervised by Itziar Gonzalez-Dios
>
> **摘要:** In this work, we present LengClaro2023, a dataset of legal-administrative texts in Spanish. Based on the most frequently used procedures from the Spanish Social Security website, we have created for each text two simplified equivalents. The first version follows the recommendations provided by arText claro. The second version incorporates additional recommendations from plain language guidelines to explore further potential improvements in the system. The linguistic resource created in this work can be used for evaluating automatic text simplification (ATS) systems in Spanish.
>
---
#### [new 021] IYKYK: Using language models to decode extremist cryptolects
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决 extremist cryptolects（极端分子隐语）的检测与解读问题。研究评估了八种模型在六个任务中的表现，发现通用语言模型效果有限，但通过领域适应和特殊提示技术可显著提升性能，并发布了相关数据集与词典。**

- **链接: [http://arxiv.org/pdf/2506.05635v1](http://arxiv.org/pdf/2506.05635v1)**

> **作者:** Christine de Kock; Arij Riabi; Zeerak Talat; Michael Sejr Schlichtkrull; Pranava Madhyastha; Ed Hovy
>
> **摘要:** Extremist groups develop complex in-group language, also referred to as cryptolects, to exclude or mislead outsiders. We investigate the ability of current language technologies to detect and interpret the cryptolects of two online extremist platforms. Evaluating eight models across six tasks, our results indicate that general purpose LLMs cannot consistently detect or decode extremist language. However, performance can be significantly improved by domain adaptation and specialised prompting techniques. These results provide important insights to inform the development and deployment of automated moderation technologies. We further develop and release novel labelled and unlabelled datasets, including 19.4M posts from extremist platforms and lexicons validated by human experts.
>
---
#### [new 022] AdvSumm: Adversarial Training for Bias Mitigation in Text Summarization
- **分类: cs.CL**

- **简介: 该论文属于文本摘要任务，旨在解决大型语言模型在摘要生成中继承的偏见问题。通过提出AdvSumm框架，利用对抗训练提升模型鲁棒性，有效缓解命名国籍偏见和政治框架偏见，同时保持摘要质量。**

- **链接: [http://arxiv.org/pdf/2506.06273v1](http://arxiv.org/pdf/2506.06273v1)**

> **作者:** Mukur Gupta; Nikhil Reddy Varimalla; Nicholas Deas; Melanie Subbiah; Kathleen McKeown
>
> **摘要:** Large Language Models (LLMs) have achieved impressive performance in text summarization and are increasingly deployed in real-world applications. However, these systems often inherit associative and framing biases from pre-training data, leading to inappropriate or unfair outputs in downstream tasks. In this work, we present AdvSumm (Adversarial Summarization), a domain-agnostic training framework designed to mitigate bias in text summarization through improved generalization. Inspired by adversarial robustness, AdvSumm introduces a novel Perturber component that applies gradient-guided perturbations at the embedding level of Sequence-to-Sequence models, enhancing the model's robustness to input variations. We empirically demonstrate that AdvSumm effectively reduces different types of bias in summarization-specifically, name-nationality bias and political framing bias-without compromising summarization quality. Compared to standard transformers and data augmentation techniques like back-translation, AdvSumm achieves stronger bias mitigation performance across benchmark datasets.
>
---
#### [new 023] A Culturally-Rich Romanian NLP Dataset from "Who Wants to Be a Millionaire?" Videos
- **分类: cs.CL**

- **简介: 该论文构建了一个包含文化背景的罗马尼亚语NLP数据集，源自电视节目《谁想成为百万富翁》。任务是评估和提升大语言模型在多文化、多语言环境下的表现，尤其关注文化相关问题的处理能力。通过OCR和人工验证收集问题及元数据，并进行跨语言实验。结果显示模型在国际问题上表现更佳，突显文化与数据来源对模型性能的影响。**

- **链接: [http://arxiv.org/pdf/2506.05991v1](http://arxiv.org/pdf/2506.05991v1)**

> **作者:** Alexandru-Gabriel Ganea; Antonia-Adelina Popovici; Adrian-Marius Dumitran
>
> **备注:** 10 pages
>
> **摘要:** Large Language Models (LLMs) demonstrate varying performance across languages and cultural contexts. This study introduces a novel, culturally-rich, multilingual dataset derived from video recordings of the Romanian game show "Who Wants to Be a Millionaire?" (Vrei s\u{a} fii Milionar?). We employed an innovative process combining optical character recognition (OCR), automated text extraction, and manual verification to collect question-answer pairs, enriching them with metadata including question domain (e.g., biology, history), cultural relevance (Romanian-specific vs. international), and difficulty. Benchmarking state-of-the-art LLMs, including Romanian-adapted models, on this dataset revealed significant performance disparities: models consistently achieve higher accuracy (80-95%) on international questions compared to Romanian-specific cultural questions (50-75%). We further investigate these differences through experiments involving machine translation of Romanian questions into English and cross-lingual tests using a comparable dataset in French. Our findings underscore the impact of cultural context and data source on LLM performance and offer practical insights for building robust, culturally-aware multilingual NLP systems, especially in educational domains. The dataset is publicly available at Hugging Face.
>
---
#### [new 024] LTG at SemEval-2025 Task 10: Optimizing Context for Classification of Narrative Roles
- **分类: cs.CL**

- **简介: 该论文参与SemEval-2025任务10子任务1，解决实体框架下的文本分类问题。通过设计面向实体的上下文选择策略，优化长文档中的上下文提取，使受限上下文窗口的模型（如XLM-RoBERTa）也能高效分类，效果媲美甚至优于大生成模型的监督微调方法。**

- **链接: [http://arxiv.org/pdf/2506.05976v1](http://arxiv.org/pdf/2506.05976v1)**

> **作者:** Egil Rønningstad; Gaurav Negi
>
> **备注:** Accepted for SemEval 2025; The 19th International Workshop on Semantic Evaluation
>
> **摘要:** Our contribution to the SemEval 2025 shared task 10, subtask 1 on entity framing, tackles the challenge of providing the necessary segments from longer documents as context for classification with a masked language model. We show that a simple entity-oriented heuristics for context selection can enable text classification using models with limited context window. Our context selection approach and the XLM-RoBERTa language model is on par with, or outperforms, Supervised Fine-Tuning with larger generative language models.
>
---
#### [new 025] Being Strong Progressively! Enhancing Knowledge Distillation of Large Language Models through a Curriculum Learning Framework
- **分类: cs.CL; cs.LG**

- **简介: 论文提出一种基于“渐进式负荷”理念的课程学习框架（POCL），用于提升大语言模型的知识蒸馏效果。该研究属于模型压缩任务，旨在解决蒸馏过程中学生模型分布偏移导致的性能问题。作者设计了难度排序机制和动态训练调度策略，逐步引入复杂样本并调整损失函数温度，以增强学习稳定性和效率。实验表明，该方法在多种蒸馏方法和模型上均有效提升性能。**

- **链接: [http://arxiv.org/pdf/2506.05695v1](http://arxiv.org/pdf/2506.05695v1)**

> **作者:** Lingyuan Liu; Mengxiang Zhang
>
> **摘要:** Knowledge Distillation (KD) compresses large language models (LLMs) by transferring the teacher model's capabilities to a smaller student model, reducing inference cost and memory usage while maintaining performance. However, existing KD methods for LLMs often fail to prevent significant shifts in the student model's distribution during training, leading to issues such as catastrophic forgetting, mode collapse, and training-inference mismatch. To address these challenges, we propose a novel, plug-in curriculum learning framework inspired by the strength training principle of "progressive overload" (POCL), which can be seamlessly integrated into existing white-box KD approaches with minimal computational overhead. The framework comprises two core components: (1) a difficulty measurer that ranks and partitions training samples from easy to hard, and (2) a training scheduler that incrementally introduces these subsets into the distillation process at fixed intervals while applying loss functions with progressively rising temperatures. By starting with the easiest samples and progressively increasing the difficulty, the approach enhances both the stability and efficiency of learning. Extensive experiments in instruction-following settings demonstrate that POCL consistently improves the performance of distilled student models across various white-box KD methods and model families. Our findings highlight the effectiveness of sorted training samples in KD for LLMs. More generally, our work demonstrates how to structure training data within the KD process to enhance the stability and performance of distilled LLMs.
>
---
#### [new 026] FinanceReasoning: Benchmarking Financial Numerical Reasoning More Credible, Comprehensive and Challenging
- **分类: cs.CL; cs.CE**

- **简介: 该论文属于金融数值推理任务，旨在评估和提升大模型在金融问题中的推理能力。论文构建了新基准FinanceReasoning，更新并标注了高质量问题，覆盖更多金融概念，并设计了挑战性问题。通过结合推理与编程模型，提升了模型表现，推动了领域内复杂推理研究。**

- **链接: [http://arxiv.org/pdf/2506.05828v1](http://arxiv.org/pdf/2506.05828v1)**

> **作者:** Zichen Tang; Haihong E; Ziyan Ma; Haoyang He; Jiacheng Liu; Zhongjun Yang; Zihua Rong; Rongjin Li; Kun Ji; Qing Huang; Xinyang Hu; Yang Liu; Qianhe Zheng
>
> **备注:** Accepted by ACL 2025 Main Conference
>
> **摘要:** We introduce FinanceReasoning, a novel benchmark designed to evaluate the reasoning capabilities of large reasoning models (LRMs) in financial numerical reasoning problems. Compared to existing benchmarks, our work provides three key advancements. (1) Credibility: We update 15.6% of the questions from four public datasets, annotating 908 new questions with detailed Python solutions and rigorously refining evaluation standards. This enables an accurate assessment of the reasoning improvements of LRMs. (2) Comprehensiveness: FinanceReasoning covers 67.8% of financial concepts and formulas, significantly surpassing existing datasets. Additionally, we construct 3,133 Python-formatted functions, which enhances LRMs' financial reasoning capabilities through refined knowledge (e.g., 83.2% $\rightarrow$ 91.6% for GPT-4o). (3) Challenge: Models are required to apply multiple financial formulas for precise numerical reasoning on 238 Hard problems. The best-performing model (i.e., OpenAI o1 with PoT) achieves 89.1% accuracy, yet LRMs still face challenges in numerical precision. We demonstrate that combining Reasoner and Programmer models can effectively enhance LRMs' performance (e.g., 83.2% $\rightarrow$ 87.8% for DeepSeek-R1). Our work paves the way for future research on evaluating and improving LRMs in domain-specific complex reasoning tasks.
>
---
#### [new 027] OPeRA: A Dataset of Observation, Persona, Rationale, and Action for Evaluating LLMs on Human Online Shopping Behavior Simulation
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于行为模拟任务，旨在解决评估大语言模型（LLM）模拟人类在线购物行为能力的问题。作者构建了一个包含用户角色、网页观察、操作和即时理由的高质量公开数据集OPERA，并建立了首个基准来评估LLMs预测用户下一步行为和推理的能力。**

- **链接: [http://arxiv.org/pdf/2506.05606v1](http://arxiv.org/pdf/2506.05606v1)**

> **作者:** Ziyi Wang; Yuxuan Lu; Wenbo Li; Amirali Amini; Bo Sun; Yakov Bart; Weimin Lyu; Jiri Gesi; Tian Wang; Jing Huang; Yu Su; Upol Ehsan; Malihe Alikhani; Toby Jia-Jun Li; Lydia Chilton; Dakuo Wang
>
> **摘要:** Can large language models (LLMs) accurately simulate the next web action of a specific user? While LLMs have shown promising capabilities in generating ``believable'' human behaviors, evaluating their ability to mimic real user behaviors remains an open challenge, largely due to the lack of high-quality, publicly available datasets that capture both the observable actions and the internal reasoning of an actual human user. To address this gap, we introduce OPERA, a novel dataset of Observation, Persona, Rationale, and Action collected from real human participants during online shopping sessions. OPERA is the first public dataset that comprehensively captures: user personas, browser observations, fine-grained web actions, and self-reported just-in-time rationales. We developed both an online questionnaire and a custom browser plugin to gather this dataset with high fidelity. Using OPERA, we establish the first benchmark to evaluate how well current LLMs can predict a specific user's next action and rationale with a given persona and <observation, action, rationale> history. This dataset lays the groundwork for future research into LLM agents that aim to act as personalized digital twins for human.
>
---
#### [new 028] Can LLMs Express Personality Across Cultures? Introducing CulturalPersonas for Evaluating Trait Alignment
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型（LLMs）在跨文化场景中表达合适个性的问题。作者构建了CulturalPersonas基准，包含六国3,000个情境问题，评估LLM在不同文化背景下的个性表达能力，提升其与本地人群性格分布的对齐度。**

- **链接: [http://arxiv.org/pdf/2506.05670v1](http://arxiv.org/pdf/2506.05670v1)**

> **作者:** Priyanka Dey; Yugal Khanter; Aayush Bothra; Jieyu Zhao; Emilio Ferrara
>
> **摘要:** As LLMs become central to interactive applications, ranging from tutoring to mental health, the ability to express personality in culturally appropriate ways is increasingly important. While recent works have explored personality evaluation of LLMs, they largely overlook the interplay between culture and personality. To address this, we introduce CulturalPersonas, the first large-scale benchmark with human validation for evaluating LLMs' personality expression in culturally grounded, behaviorally rich contexts. Our dataset spans 3,000 scenario-based questions across six diverse countries, designed to elicit personality through everyday scenarios rooted in local values. We evaluate three LLMs, using both multiple-choice and open-ended response formats. Our results show that CulturalPersonas improves alignment with country-specific human personality distributions (over a 20% reduction in Wasserstein distance across models and countries) and elicits more expressive, culturally coherent outputs compared to existing benchmarks. CulturalPersonas surfaces meaningful modulated trait outputs in response to culturally grounded prompts, offering new directions for aligning LLMs to global norms of behavior. By bridging personality expression and cultural nuance, we envision that CulturalPersonas will pave the way for more socially intelligent and globally adaptive LLMs.
>
---
#### [new 029] Token Signature: Predicting Chain-of-Thought Gains with Token Decoding Feature in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型中思维链（CoT）推理效果不稳定的问题。作者通过分析词元概率分布的单调性，提出评估CoT有效性的指标，并构建动态选择CoT或直接回答的方法（Dynamic CoT），减少计算资源消耗同时保持高准确率。**

- **链接: [http://arxiv.org/pdf/2506.06008v1](http://arxiv.org/pdf/2506.06008v1)**

> **作者:** Peijie Liu; Fengli Xu; Yong Li
>
> **备注:** 20 pages, 6 figures, 13 tables(Accept by ICML2025)
>
> **摘要:** Chain-of-Thought (CoT) technique has proven effective in improving the performance of large language models (LLMs) on complex reasoning tasks. However, the performance gains are inconsistent across different tasks, and the underlying mechanism remains a long-standing research question. In this work, we make a preliminary observation that the monotonicity of token probability distributions may be correlated with the gains achieved through CoT reasoning. Leveraging this insight, we propose two indicators based on the token probability distribution to assess CoT effectiveness across different tasks. By combining instance-level indicators with logistic regression model, we introduce Dynamic CoT, a method that dynamically select between CoT and direct answer. Furthermore, we extend Dynamic CoT to closed-source models by transferring decision strategies learned from open-source models. Our indicators for assessing CoT effectiveness achieve an accuracy of 89.2\%, and Dynamic CoT reduces token consumption by more than 35\% while maintaining high accuracy. Overall, our work offers a novel perspective on the underlying mechanisms of CoT reasoning and provides a framework for its more efficient deployment.
>
---
#### [new 030] DynamicMind: A Tri-Mode Thinking System for Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文提出DynamicMind系统，旨在提升大语言模型在零样本问答任务中的推理能力。它通过引入快、正常和慢三种思维模式，动态匹配任务复杂度，优化资源利用。核心工作包括设计思维密度指标、构建TMC数据集及轻量级Mind Router，实现性能与效率的平衡。**

- **链接: [http://arxiv.org/pdf/2506.05936v1](http://arxiv.org/pdf/2506.05936v1)**

> **作者:** Wei Li; Yanbin Wei; Qiushi Huang; Jiangyue Yan; Yang Chen; James T. Kwok; Yu Zhang
>
> **摘要:** Modern large language models (LLMs) often struggle to dynamically adapt their reasoning depth to varying task complexities, leading to suboptimal performance or inefficient resource utilization. To address this, we introduce DynamicMind, a novel tri-mode thinking system. DynamicMind empowers LLMs to autonomously select between Fast, Normal, and Slow thinking modes for zero-shot question answering (ZSQA) tasks through cognitive-inspired prompt engineering. Our framework's core innovations include: (1) expanding the established dual-process framework of fast and slow thinking into a tri-mode thinking system involving a normal thinking mode to preserve the intrinsic capabilities of LLM; (2) proposing the Thinking Density metric, which aligns computational resource allocation with problem complexity; and (3) developing the Thinking Mode Capacity (TMC) dataset and a lightweight Mind Router to predict the optimal thinking mode. Extensive experiments across diverse mathematical, commonsense, and scientific QA benchmarks demonstrate that DynamicMind achieves superior ZSQA capabilities while establishing an effective trade-off between performance and computational efficiency.
>
---
#### [new 031] UTSA-NLP at ArchEHR-QA 2025: Improving EHR Question Answering via Self-Consistency Prompting
- **分类: cs.CL**

- **简介: 该论文参与的是基于电子健康记录（EHR）的临床问答任务，旨在提升从EHR中回答临床问题的准确性。作者提出了一种使用大语言模型的两步方法：首先识别与问题相关的EHR句子，然后生成有引用支持的回答。通过少样本提示、自洽性和阈值筛选优化句子分类效果。结果表明，准确选择关键句对提高回答质量至关重要。**

- **链接: [http://arxiv.org/pdf/2506.05589v1](http://arxiv.org/pdf/2506.05589v1)**

> **作者:** Sara Shields-Menard; Zach Reimers; Joshua Gardner; David Perry; Anthony Rios
>
> **备注:** Accepted to BioNLP 2025
>
> **摘要:** We describe our system for the ArchEHR-QA Shared Task on answering clinical questions using electronic health records (EHRs). Our approach uses large language models in two steps: first, to find sentences in the EHR relevant to a clinician's question, and second, to generate a short, citation-supported response based on those sentences. We use few-shot prompting, self-consistency, and thresholding to improve the sentence classification step to decide which sentences are essential. We compare several models and find that a smaller 8B model performs better than a larger 70B model for identifying relevant information. Our results show that accurate sentence selection is critical for generating high-quality responses and that self-consistency with thresholding helps make these decisions more reliable.
>
---
#### [new 032] Improving LLMs with a knowledge from databases
- **分类: cs.CL; I.2.7; I.2.4**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型（LLM）基于数据库回答问题的能力。为解决LLM在结构化数据理解和安全性方面的不足，作者提出了一种结合增强关联规则与RAG的方法，通过规则生成、文本转换并融合到LLM中，有效提升了问答性能，并探索了多种规则生成策略及未来改进方向。**

- **链接: [http://arxiv.org/pdf/2506.05560v1](http://arxiv.org/pdf/2506.05560v1)**

> **作者:** Petr Máša
>
> **摘要:** Large language models (LLMs) are achieving significant progress almost every moment now. Many advanced techniques have been introduced and widely accepted, like retrieval-augmentation generation (RAG), agents, and tools. Tools can query the database to answer questions from structured data files or perform groupings or other statistics. This unlocks huge opportunities, such as it can answer any question, but also poses threats, such as safety, because there is no control over the commands that are created. We would like to discuss whether we can create a new method that improves answers based on dataset/database via some interpretable ML methods, namely enhanced association rules. The advantage would be if the method can be also used in some safe technique like RAG. Association rules have a sound history. Since the introduction of CN2 and aproiri, many enhancements have been made. In parallel, enhanced association rules have been introduced and evolved over the last 40 years. The general problem is typically that there are too many rules. There are some techniques for handling it, but when LLM emerged, it turned out to be the best use case for the RAG technique for LLMs. We proposed a method that generates a ruleset based on defined knowledge patterns, then converts rules into text form via a rule-to-text converter, and includes the result as an RAG into LLM. We compared this method with ChatGPT (even with using agents) and we have discovered a significant improvement in answering questions based on the dataset. We have also tried several strategies how much rules to generate. We found this improvement interesting. Moreover, it can also be improved in many ways as future work, like incorporating other patterns, the use of rule mining as an agent, and many others.
>
---
#### [new 033] Bridging External and Parametric Knowledge: Mitigating Hallucination of LLMs with Shared-Private Semantic Synergy in Dual-Stream Knowledge
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型生成中的幻觉问题。通过提出DSSP-RAG框架，结合外部知识与模型参数知识，利用混合注意力机制和能量商减少知识冲突，提升生成准确性和稳定性。**

- **链接: [http://arxiv.org/pdf/2506.06240v1](http://arxiv.org/pdf/2506.06240v1)**

> **作者:** Yi Sui; Chaozhuo Li; Chen Zhang; Dawei song; Qiuchi Li
>
> **摘要:** Retrieval-augmented generation (RAG) is a cost-effective approach to mitigate the hallucination of Large Language Models (LLMs) by incorporating the retrieved external knowledge into the generation process. However, external knowledge may conflict with the parametric knowledge of LLMs. Furthermore, current LLMs lack inherent mechanisms for resolving such knowledge conflicts, making traditional RAG methods suffer from degraded performance and stability. Thus, we propose a Dual-Stream Knowledge-Augmented Framework for Shared-Private Semantic Synergy (DSSP-RAG). Central to the framework is a novel approach that refines self-attention into a mixed-attention, distinguishing shared and private semantics for a controlled internal-external knowledge integration. To effectively facilitate DSSP in RAG, we further introduce an unsupervised hallucination detection method based on cognitive uncertainty, ensuring the necessity of introducing knowledge, and an Energy Quotient (EQ) based on attention difference matrices to reduce noise in the retrieved external knowledge. Extensive experiments on benchmark datasets show that DSSP-RAG can effectively resolve conflicts and enhance the complementarity of dual-stream knowledge, leading to superior performance over strong baselines.
>
---
#### [new 034] Homogeneous Keys, Heterogeneous Values: Exploiting Local KV Cache Asymmetry for Long-Context LLMs
- **分类: cs.CL; I.2.7**

- **简介: 该论文属于自然语言处理任务，旨在解决长上下文建模中注意力机制效率低下的问题。通过分析KV缓存的不对称性，提出AsymKV压缩框架，结合键合并与无损值压缩，显著提升长上下文LLM性能。**

- **链接: [http://arxiv.org/pdf/2506.05410v1](http://arxiv.org/pdf/2506.05410v1)**

> **作者:** Wanyun Cui; Mingwei Xu
>
> **备注:** 14 pages,7 figures
>
> **摘要:** Recent advances in Large Language Models (LLMs) have highlighted the critical importance of extending context length, yet the quadratic complexity of attention mechanisms poses significant challenges for efficient long-context modeling. KV cache compression has emerged as a key approach to address this challenge. Through extensive empirical analysis, we reveal a fundamental yet previously overlooked asymmetry in KV caches: while adjacent keys receive similar attention weights (local homogeneity), adjacent values demonstrate distinct heterogeneous distributions. This key-value asymmetry reveals a critical limitation in existing compression methods that treat keys and values uniformly. To address the limitation, we propose a training-free compression framework (AsymKV) that combines homogeneity-based key merging with a mathematically proven lossless value compression. Extensive experiments demonstrate that AsymKV consistently outperforms existing long-context methods across various tasks and base models. For example, on LLaMA3.1-8B, AsymKV achieves an average score of 43.95 on LongBench, surpassing SOTA methods like H$_2$O (38.89) by a large margin.
>
---
#### [new 035] Building Models of Neurological Language
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在构建神经学领域的专业语言模型。为解决领域适应性和部署安全性问题，作者创建了神经学数据集，开发了多词表达抽取工具与图谱分析方法，并实现本地部署方案。**

- **链接: [http://arxiv.org/pdf/2506.06208v1](http://arxiv.org/pdf/2506.06208v1)**

> **作者:** Henry Watkins
>
> **备注:** 21 pages, 6 figures
>
> **摘要:** This report documents the development and evaluation of domain-specific language models for neurology. Initially focused on building a bespoke model, the project adapted to rapid advances in open-source and commercial medical LLMs, shifting toward leveraging retrieval-augmented generation (RAG) and representational models for secure, local deployment. Key contributions include the creation of neurology-specific datasets (case reports, QA sets, textbook-derived data), tools for multi-word expression extraction, and graph-based analyses of medical terminology. The project also produced scripts and Docker containers for local hosting. Performance metrics and graph community results are reported, with future possible work open for multimodal models using open-source architectures like phi-4.
>
---
#### [new 036] A Unified Representation for Continuity and Discontinuity: Syntactic and Computational Motivations
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与理论语言学交叉任务，旨在解决不同语法形式体系（短语结构语法、依存语法和范畴语法）在连续与非连续语言结构表示上的统一问题。论文提出“对应原则”，以土耳其语为例展示如何实现统一表示，并探讨其在神经认知处理中的计算复杂度优化作用。**

- **链接: [http://arxiv.org/pdf/2506.05686v1](http://arxiv.org/pdf/2506.05686v1)**

> **作者:** Ratna Kandala; Prakash Mondal
>
> **摘要:** This paper advances a unified representation of linguistic structure for three grammar formalisms, namely, Phrase Structure Grammar (PSG), Dependency Grammar (DG) and Categorial Grammar (CG) from the perspective of syntactic and computational complexity considerations. The correspondence principle is proposed to enable a unified representation of the representational principles from PSG, DG, and CG. To that end, the paper first illustrates a series of steps in achieving a unified representation for a discontinuous subordinate clause from Turkish as an illustrative case. This affords a new way of approaching discontinuity in natural language from a theoretical point of view that unites and integrates the basic tenets of PSG, DG, and CG, with significant consequences for syntactic analysis. Then this paper demonstrates that a unified representation can simplify computational complexity with regards to the neurocognitive representation and processing of both continuous and discontinuous sentences vis-\`a-vis the basic principles of PSG, DG, and CG.
>
---
#### [new 037] Understanding Gender Bias in AI-Generated Product Descriptions
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究任务为识别和分析电商场景中AI生成商品描述中的性别偏见问题。作者构建了性别偏见的分类体系，结合通用AI危害框架，揭示了如服装尺寸假设、产品特征刻板印象及说服性语言差异等偏见表现，并通过GPT-3.5与电商专用模型进行实证分析，指出需专门检测与缓解策略的重要性。**

- **链接: [http://arxiv.org/pdf/2506.05390v1](http://arxiv.org/pdf/2506.05390v1)**

> **作者:** Markelle Kelly; Mohammad Tahaei; Padhraic Smyth; Lauren Wilcox
>
> **备注:** Accepted to FAccT 2025
>
> **摘要:** While gender bias in large language models (LLMs) has been extensively studied in many domains, uses of LLMs in e-commerce remain largely unexamined and may reveal novel forms of algorithmic bias and harm. Our work investigates this space, developing data-driven taxonomic categories of gender bias in the context of product description generation, which we situate with respect to existing general purpose harms taxonomies. We illustrate how AI-generated product descriptions can uniquely surface gender biases in ways that require specialized detection and mitigation approaches. Further, we quantitatively analyze issues corresponding to our taxonomic categories in two models used for this task -- GPT-3.5 and an e-commerce-specific LLM -- demonstrating that these forms of bias commonly occur in practice. Our results illuminate unique, under-explored dimensions of gender bias, such as assumptions about clothing size, stereotypical bias in which features of a product are advertised, and differences in the use of persuasive language. These insights contribute to our understanding of three types of AI harms identified by current frameworks: exclusionary norms, stereotyping, and performance disparities, particularly for the context of e-commerce.
>
---
#### [new 038] Leveraging Self-Attention for Input-Dependent Soft Prompting in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大型语言模型在领域特定任务中微调成本高的问题。提出了一种基于自注意力机制的输入依赖软提示方法（ID-SPAM），通过生成与输入相关的软提示，提升模型适应下游任务的效率和零样本跨域迁移能力。**

- **链接: [http://arxiv.org/pdf/2506.05629v1](http://arxiv.org/pdf/2506.05629v1)**

> **作者:** Ananth Muppidi; Abhilash Nandy; Sambaran Bandyopadhyay
>
> **备注:** Accepted in ACL 2025 (Main) Conference
>
> **摘要:** The performance of large language models in domain-specific tasks necessitates fine-tuning, which is computationally expensive and technically challenging. This paper focuses on parameter-efficient fine-tuning using soft prompting, a promising approach that adapts pre-trained models to downstream tasks by learning a small set of parameters. We propose a novel Input Dependent Soft Prompting technique with a self-Attention Mechanism (ID-SPAM) that generates soft prompts based on the input tokens and attends different tokens with varying importance. Our method is simple and efficient, keeping the number of trainable parameters small. We show the merits of the proposed approach compared to state-of-the-art techniques on various tasks and show the improved zero shot domain transfer capability.
>
---
#### [new 039] Elementary Math Word Problem Generation using Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于数学应用题生成任务，旨在解决教师手动创建题目耗时的问题。利用大语言模型（LLMs），仅输入所需题目数量、年级和题型即可生成高质量数学应用题，并通过多种策略提升生成效果与多样性。**

- **链接: [http://arxiv.org/pdf/2506.05950v1](http://arxiv.org/pdf/2506.05950v1)**

> **作者:** Nimesh Ariyarathne; Harshani Bandara; Yasith Heshan; Omega Gamage; Surangika Ranathunga; Dilan Nayanajith; Yutharsan Sivapalan; Gayathri Lihinikaduarachchi; Tharoosha Vihidun; Meenambika Chandirakumar; Sanujen Premakumar; Sanjula Gathsara
>
> **摘要:** Mathematics is often perceived as a complex subject by students, leading to high failure rates in exams. To improve Mathematics skills, it is important to provide sample questions for students to practice problem-solving. Manually creating Math Word Problems (MWPs) is time consuming for tutors, because they have to type in natural language while adhering to grammar and spelling rules of the language. Existing Deep Learning techniques for MWP generation either require a tutor to provide the initial portion of the MWP, and/or additional information such as an equation. In this paper, we present an MWP generation system based on Large Language Models (LLMs) that overcome the need for additional input - the only input to our system is the number of MWPs needed, the grade and the type of question (e.g. addition, subtraction). Unlike the existing LLM-based solutions for MWP generation, we carried out an extensive set of experiments involving different LLMs, prompting strategies, techniques to improve the diversity of questions, as well as techniques that employ human feedback to improve LLM performance. Human and automated evaluations confirmed that the generated MWPs are high in quality, with minimal spelling and grammar issues. However, LLMs still struggle to generate questions that adhere to the specified grade and question type requirements.
>
---
#### [new 040] Generating Grounded Responses to Counter Misinformation via Learning Efficient Fine-Grained Critiques
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大型语言模型在生成辟谣回应时易产生虚假信息的问题。作者提出了MisMitiFact框架，通过训练轻量级、细粒度的批判模型，自动生成简单反馈以修正数值、实体和主题等关键元素中的错误，从而提升辟谣回应的准确性和生成效率。**

- **链接: [http://arxiv.org/pdf/2506.05924v1](http://arxiv.org/pdf/2506.05924v1)**

> **作者:** Xiaofei Xu; Xiuzhen Zhang; Ke Deng
>
> **备注:** accepted to IJCAI 2025
>
> **摘要:** Fake news and misinformation poses a significant threat to society, making efficient mitigation essential. However, manual fact-checking is costly and lacks scalability. Large Language Models (LLMs) offer promise in automating counter-response generation to mitigate misinformation, but a critical challenge lies in their tendency to hallucinate non-factual information. Existing models mainly rely on LLM self-feedback to reduce hallucination, but this approach is computationally expensive. In this paper, we propose MisMitiFact, Misinformation Mitigation grounded in Facts, an efficient framework for generating fact-grounded counter-responses at scale. MisMitiFact generates simple critique feedback to refine LLM outputs, ensuring responses are grounded in evidence. We develop lightweight, fine-grained critique models trained on data sourced from readily available fact-checking sites to identify and correct errors in key elements such as numerals, entities, and topics in LLM generations. Experiments show that MisMitiFact generates counter-responses of comparable quality to LLMs' self-feedback while using significantly smaller critique models. Importantly, it achieves ~5x increase in feedback generation throughput, making it highly suitable for cost-effective, large-scale misinformation mitigation. Code and LLM prompt templates are at https://github.com/xxfwin/MisMitiFact.
>
---
#### [new 041] SynthesizeMe! Inducing Persona-Guided Prompts for Personalized Reward Models in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于个性化对齐任务，旨在解决现有方法依赖额外身份信息的问题。作者提出SynthesizeMe，通过生成和验证用户偏好推理，合成用户人设，并用于构建个性化提示。实验表明，该方法在Chatbot Arena和新数据集PersonalRewardBench上提升了个性化判断准确率。**

- **链接: [http://arxiv.org/pdf/2506.05598v1](http://arxiv.org/pdf/2506.05598v1)**

> **作者:** Michael J Ryan; Omar Shaikh; Aditri Bhagirath; Daniel Frees; William Held; Diyi Yang
>
> **备注:** ACL 2025 Main Conference
>
> **摘要:** Recent calls for pluralistic alignment of Large Language Models (LLMs) encourage adapting models to diverse user preferences. However, most prior work on personalized reward models heavily rely on additional identity information, such as demographic details or a predefined set of preference categories. To this end, we introduce SynthesizeMe, an approach to inducing synthetic user personas from user interactions for personalized reward modeling. SynthesizeMe first generates and verifies reasoning to explain user preferences, then induces synthetic user personas from that reasoning, and finally filters to informative prior user interactions in order to build personalized prompts for a particular user. We show that using SynthesizeMe induced prompts improves personalized LLM-as-a-judge accuracy by 4.4% on Chatbot Arena. Combining SynthesizeMe derived prompts with a reward model achieves top performance on PersonalRewardBench: a new curation of user-stratified interactions with chatbots collected from 854 users of Chatbot Arena and PRISM.
>
---
#### [new 042] Mitigating Confounding in Speech-Based Dementia Detection through Weight Masking
- **分类: cs.CL**

- **简介: 该论文属于语音识别与阿尔茨海默病检测任务，旨在解决性别混淆对基于语音的大脑退化检测模型的影响。作者提出两种方法（Extended Confounding Filter和Dual Filter）来隔离并削弱性别相关权重，以减少模型偏差。实验表明这些方法可有效去混淆，但略微降低了检测性能。**

- **链接: [http://arxiv.org/pdf/2506.05610v1](http://arxiv.org/pdf/2506.05610v1)**

> **作者:** Zhecheng Sheng; Xiruo Ding; Brian Hur; Changye Li; Trevor Cohen; Serguei Pakhomov
>
> **备注:** 16 pages, 20 figures. Accepted to ACL 2025 Main Conference
>
> **摘要:** Deep transformer models have been used to detect linguistic anomalies in patient transcripts for early Alzheimer's disease (AD) screening. While pre-trained neural language models (LMs) fine-tuned on AD transcripts perform well, little research has explored the effects of the gender of the speakers represented by these transcripts. This work addresses gender confounding in dementia detection and proposes two methods: the $\textit{Extended Confounding Filter}$ and the $\textit{Dual Filter}$, which isolate and ablate weights associated with gender. We evaluate these methods on dementia datasets with first-person narratives from patients with cognitive impairment and healthy controls. Our results show transformer models tend to overfit to training data distributions. Disrupting gender-related weights results in a deconfounded dementia classifier, with the trade-off of slightly reduced dementia detection performance.
>
---
#### [new 043] Reinforcing Code Generation: Improving Text-to-SQL with Execution-Based Learning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理与代码生成任务，旨在解决从自然语言生成准确SQL查询的问题。作者提出通过与数据库交互的强化学习方法，利用执行反馈优化模型，使用GRPO框架进行训练。实验表明，该方法显著提升了生成SQL的准确性，接近更大模型SQLCoder-70B的性能。**

- **链接: [http://arxiv.org/pdf/2506.06093v1](http://arxiv.org/pdf/2506.06093v1)**

> **作者:** Atharv Kulkarni; Vivek Srikumar
>
> **备注:** Under review at EMNLP 2025
>
> **摘要:** In this work, we study the problem of code generation with a large language model (LLM), with a focus on generating SQL queries from natural language questions. We ask: Instead of using supervised fine tuning with text-code pairs, can we tune a model by having it interact with a database engine? We frame this problem as a reinforcement learning problem where the model receives execution-based feedback from the environment in the form of scalar rewards. These rewards penalize execution failures and assign positive values when a query returns a correct answer. We use the rewards within the Group Relative Policy Optimization (GRPO) framework. We use a tabular reasoning benchmark to test and evaluate our findings. We find that with only weak supervision in the form of question-answer pairs, RL-tuning improves the accuracy of model generated SQL code from 31.49 to 49.83 while reducing error percentage from 25.43% to 14.71%. This improvement allowed the model nearly match the performance performance to the larger SQLCoder-70B model. Our work demonstrates the potential of using execution-based feedback to improve symbolic reasoning capabilities of LLMs.
>
---
#### [new 044] Hey, That's My Data! Label-Only Dataset Inference in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决未经授权使用训练数据的问题。通过提出CatShift框架，利用灾难性遗忘现象，在无法访问模型logits的情况下，仅根据标签判断数据集归属，有效检测大型语言模型是否使用了特定数据进行训练。**

- **链接: [http://arxiv.org/pdf/2506.06057v1](http://arxiv.org/pdf/2506.06057v1)**

> **作者:** Chen Xiong; Zihao Wang; Rui Zhu; Tsung-Yi Ho; Pin-Yu Chen; Jingwei Xiong; Haixu Tang; Lucila Ohno-Machado
>
> **摘要:** Large Language Models (LLMs) have revolutionized Natural Language Processing by excelling at interpreting, reasoning about, and generating human language. However, their reliance on large-scale, often proprietary datasets poses a critical challenge: unauthorized usage of such data can lead to copyright infringement and significant financial harm. Existing dataset-inference methods typically depend on log probabilities to detect suspicious training material, yet many leading LLMs have begun withholding or obfuscating these signals. This reality underscores the pressing need for label-only approaches capable of identifying dataset membership without relying on internal model logits. We address this gap by introducing CatShift, a label-only dataset-inference framework that capitalizes on catastrophic forgetting: the tendency of an LLM to overwrite previously learned knowledge when exposed to new data. If a suspicious dataset was previously seen by the model, fine-tuning on a portion of it triggers a pronounced post-tuning shift in the model's outputs; conversely, truly novel data elicits more modest changes. By comparing the model's output shifts for a suspicious dataset against those for a known non-member validation set, we statistically determine whether the suspicious set is likely to have been part of the model's original training corpus. Extensive experiments on both open-source and API-based LLMs validate CatShift's effectiveness in logit-inaccessible settings, offering a robust and practical solution for safeguarding proprietary data.
>
---
#### [new 045] Tau-Eval: A Unified Evaluation Framework for Useful and Private Text Anonymization
- **分类: cs.CL**

- **简介: 该论文属于文本匿名化评估任务，旨在解决隐私保护与信息保留之间的权衡问题。作者提出了Tau-Eval框架，用于从隐私和效用角度综合评估匿名化方法，并提供开源工具支持。**

- **链接: [http://arxiv.org/pdf/2506.05979v1](http://arxiv.org/pdf/2506.05979v1)**

> **作者:** Gabriel Loiseau; Damien Sileo; Damien Riquet; Maxime Meyer; Marc Tommasi
>
> **摘要:** Text anonymization is the process of removing or obfuscating information from textual data to protect the privacy of individuals. This process inherently involves a complex trade-off between privacy protection and information preservation, where stringent anonymization methods can significantly impact the text's utility for downstream applications. Evaluating the effectiveness of text anonymization proves challenging from both privacy and utility perspectives, as there is no universal benchmark that can comprehensively assess anonymization techniques across diverse, and sometimes contradictory contexts. We present Tau-Eval, an open-source framework for benchmarking text anonymization methods through the lens of privacy and utility task sensitivity. A Python library, code, documentation and tutorials are publicly available.
>
---
#### [new 046] Can Theoretical Physics Research Benefit from Language Agents?
- **分类: cs.CL; cs.AI; math-ph; math.MP; quant-ph**

- **简介: 论文探讨大型语言模型（LLM）在理论物理研究中的潜力与挑战，属于交叉学科任务。旨在解决LLM在物理领域应用不成熟的问题，分析其当前能力与不足，提出未来发展方向。工作包括评估LLM在数学推理、代码生成等方面的表现，指出物理直觉和约束满足等关键缺陷，并呼吁物理与AI领域合作推动科学发现。**

- **链接: [http://arxiv.org/pdf/2506.06214v1](http://arxiv.org/pdf/2506.06214v1)**

> **作者:** Sirui Lu; Zhijing Jin; Terry Jingchen Zhang; Pavel Kos; J. Ignacio Cirac; Bernhard Schölkopf
>
> **备注:** 9 pages
>
> **摘要:** Large Language Models (LLMs) are rapidly advancing across diverse domains, yet their application in theoretical physics research is not yet mature. This position paper argues that LLM agents can potentially help accelerate theoretical, computational, and applied physics when properly integrated with domain knowledge and toolbox. We analyze current LLM capabilities for physics -- from mathematical reasoning to code generation -- identifying critical gaps in physical intuition, constraint satisfaction, and reliable reasoning. We envision future physics-specialized LLMs that could handle multimodal data, propose testable hypotheses, and design experiments. Realizing this vision requires addressing fundamental challenges: ensuring physical consistency, and developing robust verification methods. We call for collaborative efforts between physics and AI communities to help advance scientific discovery in physics.
>
---
#### [new 047] BioMol-MQA: A Multi-Modal Question Answering Dataset For LLM Reasoning Over Bio-Molecular Interactions
- **分类: cs.CL**

- **简介: 该论文属于多模态问答任务，旨在解决现有RAG模型在生物分子交互等复杂领域中难以有效检索与推理多模态信息的问题。作者构建了包含多模态知识图谱和挑战性问题的BioMol-MQA数据集，用于评估和推动LLM在药物联用问题上的推理能力。**

- **链接: [http://arxiv.org/pdf/2506.05766v1](http://arxiv.org/pdf/2506.05766v1)**

> **作者:** Saptarshi Sengupta; Shuhua Yang; Paul Kwong Yu; Fali Wang; Suhang Wang
>
> **摘要:** Retrieval augmented generation (RAG) has shown great power in improving Large Language Models (LLMs). However, most existing RAG-based LLMs are dedicated to retrieving single modality information, mainly text; while for many real-world problems, such as healthcare, information relevant to queries can manifest in various modalities such as knowledge graph, text (clinical notes), and complex molecular structure. Thus, being able to retrieve relevant multi-modality domain-specific information, and reason and synthesize diverse knowledge to generate an accurate response is important. To address the gap, we present BioMol-MQA, a new question-answering (QA) dataset on polypharmacy, which is composed of two parts (i) a multimodal knowledge graph (KG) with text and molecular structure for information retrieval; and (ii) challenging questions that designed to test LLM capabilities in retrieving and reasoning over multimodal KG to answer questions. Our benchmarks indicate that existing LLMs struggle to answer these questions and do well only when given the necessary background data, signaling the necessity for strong RAG frameworks.
>
---
#### [new 048] Zero-Shot Detection of LLM-Generated Code via Approximated Task Conditioning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于代码来源检测任务，旨在解决如何在无训练条件下（零样本）区分大模型生成与人类编写的代码问题。作者提出ATC方法，通过近似任务条件下的代码token熵值差异进行检测，无需访问原始模型或任务提示，实现了跨语言、无需训练的高效检测。**

- **链接: [http://arxiv.org/pdf/2506.06069v1](http://arxiv.org/pdf/2506.06069v1)**

> **作者:** Maor Ashkenazi; Ofir Brenner; Tal Furman Shohet; Eran Treister
>
> **备注:** To appear in the Proceedings of ECML-PKDD 2025, Springer Lecture Notes in Computer Science (LNCS)
>
> **摘要:** Detecting Large Language Model (LLM)-generated code is a growing challenge with implications for security, intellectual property, and academic integrity. We investigate the role of conditional probability distributions in improving zero-shot LLM-generated code detection, when considering both the code and the corresponding task prompt that generated it. Our key insight is that when evaluating the probability distribution of code tokens using an LLM, there is little difference between LLM-generated and human-written code. However, conditioning on the task reveals notable differences. This contrasts with natural language text, where differences exist even in the unconditional distributions. Leveraging this, we propose a novel zero-shot detection approach that approximates the original task used to generate a given code snippet and then evaluates token-level entropy under the approximated task conditioning (ATC). We further provide a mathematical intuition, contextualizing our method relative to previous approaches. ATC requires neither access to the generator LLM nor the original task prompts, making it practical for real-world applications. To the best of our knowledge, it achieves state-of-the-art results across benchmarks and generalizes across programming languages, including Python, CPP, and Java. Our findings highlight the importance of task-level conditioning for LLM-generated code detection. The supplementary materials and code are available at https://github.com/maorash/ATC, including the dataset gathering implementation, to foster further research in this area.
>
---
#### [new 049] Beyond RAG: Reinforced Reasoning Augmented Generation for Clinical Notes
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床文本生成任务，旨在解决基于有限患者信息生成长格式出院指令的问题。作者提出R2AG模型，结合强化学习与医学知识图谱，优化检索路径，提升生成内容的临床准确性和语言质量。**

- **链接: [http://arxiv.org/pdf/2506.05386v1](http://arxiv.org/pdf/2506.05386v1)**

> **作者:** Lo Pang-Yun Ting; Chengshuai Zhao; Yu-Hua Zeng; Yuan Jee Lim; Kun-Ta Chuang
>
> **摘要:** Clinical note generation aims to automatically produce free-text summaries of a patient's condition and diagnostic process, with discharge instructions being a representative long-form example. While recent large language model (LLM)-based methods pre-trained on general clinical corpora show promise in clinical text generation, they fall short in producing long-form notes from limited patient information. In this paper, we propose R2AG, the first reinforced retriever for long-form discharge instruction generation based on pre-admission data. R2AG is trained with reinforcement learning to retrieve reasoning paths from a medical knowledge graph, providing explicit semantic guidance to the LLM. To bridge the information gap, we propose Group-Based Retriever Optimization (GRO) which improves retrieval quality with group-relative rewards, encouraging reasoning leaps for deeper inference by the LLM. Comprehensive experiments on the MIMIC-IV-Note dataset show that R2AG outperforms baselines in both clinical efficacy and natural language generation metrics. Further analysis reveals that R2AG fills semantic gaps in sparse input scenarios, and retrieved reasoning paths help LLMs avoid clinical misinterpretation by focusing on key evidence and following coherent reasoning.
>
---
#### [new 050] Detecting Voice Phishing with Precision: Fine-Tuning Small Language Models
- **分类: cs.CL**

- **简介: 该论文属于语音钓鱼检测任务，旨在解决小规模语言模型在识别语音诈骗中的准确性问题。作者通过微调Llama3模型，结合专家设计的评估标准和思维链技术，构建了高效的语音钓鱼检测系统，并展示了其性能可媲美GPT-4模型。**

- **链接: [http://arxiv.org/pdf/2506.06180v1](http://arxiv.org/pdf/2506.06180v1)**

> **作者:** Ju Yong Sim; Seong Hwan Kim
>
> **备注:** 15 pages, 4 figures, 8 tables, journal submission
>
> **摘要:** We develop a voice phishing (VP) detector by fine-tuning Llama3, a representative open-source, small language model (LM). In the prompt, we provide carefully-designed VP evaluation criteria and apply the Chain-of-Thought (CoT) technique. To evaluate the robustness of LMs and highlight differences in their performance, we construct an adversarial test dataset that places the models under challenging conditions. Moreover, to address the lack of VP transcripts, we create transcripts by referencing existing or new types of VP techniques. We compare cases where evaluation criteria are included, the CoT technique is applied, or both are used together. In the experiment, our results show that the Llama3-8B model, fine-tuned with a dataset that includes a prompt with VP evaluation criteria, yields the best performance among small LMs and is comparable to that of a GPT-4-based VP detector. These findings indicate that incorporating human expert knowledge into the prompt is more effective than using the CoT technique for small LMs in VP detection.
>
---
#### [new 051] Let's Put Ourselves in Sally's Shoes: Shoes-of-Others Prefixing Improves Theory of Mind in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的“心智理论”（ToM）能力。现有方法在推理时依赖特定情境，限制了应用范围。论文提出新方法Shoes-of-Others (SoO) prefixing，在输出开头添加提示语引导模型代入角色视角。实验表明该方法有效提升模型对多种心理状态的理解能力。**

- **链接: [http://arxiv.org/pdf/2506.05970v1](http://arxiv.org/pdf/2506.05970v1)**

> **作者:** Kazutoshi Shinoda; Nobukatsu Hojo; Kyosuke Nishida; Yoshihiro Yamazaki; Keita Suzuki; Hiroaki Sugiyama; Kuniko Saito
>
> **备注:** 14pages, 12 figures
>
> **摘要:** Recent studies have shown that Theory of Mind (ToM) in large language models (LLMs) has not reached human-level performance yet. Since fine-tuning LLMs on ToM datasets often degrades their generalization, several inference-time methods have been proposed to enhance ToM in LLMs. However, existing inference-time methods for ToM are specialized for inferring beliefs from contexts involving changes in the world state. In this study, we present a new inference-time method for ToM, Shoes-of-Others (SoO) prefixing, which makes fewer assumptions about contexts and is applicable to broader scenarios. SoO prefixing simply specifies the beginning of LLM outputs with ``Let's put ourselves in A's shoes.'', where A denotes the target character's name. We evaluate SoO prefixing on two benchmarks that assess ToM in conversational and narrative contexts without changes in the world state and find that it consistently improves ToM across five categories of mental states. Our analysis suggests that SoO prefixing elicits faithful thoughts, thereby improving the ToM performance.
>
---
#### [new 052] Advancing Decoding Strategies: Enhancements in Locally Typical Sampling for LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，旨在解决大语言模型解码时流畅性、多样性和一致性难以平衡的问题。论文提出了改进的局部典型采样方法ASTS，通过动态熵阈值、多目标评分和奖惩调整提升生成效果，经验证在多个生成任务中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.05387v1](http://arxiv.org/pdf/2506.05387v1)**

> **作者:** Jaydip Sen; Saptarshi Sengupta. Subhasis Dasgupta
>
> **备注:** This is the accepted but pre-reviewed version of the chapter that has been accepted for publication in the Springer volume 'Decision-Making in Computational Intelligence-Based Systems,' edited by Witold Pedrycz, Gilberto Rivera, Rose Ma Rodriguez, and Salvador Ibarra Martinez. The chapter is 39 pages long, and it contains 2 figures and 6 tables. This is NOT the final camera-ready version
>
> **摘要:** This chapter explores advancements in decoding strategies for large language models (LLMs), focusing on enhancing the Locally Typical Sampling (LTS) algorithm. Traditional decoding methods, such as top-k and nucleus sampling, often struggle to balance fluency, diversity, and coherence in text generation. To address these challenges, Adaptive Semantic-Aware Typicality Sampling (ASTS) is proposed as an improved version of LTS, incorporating dynamic entropy thresholding, multi-objective scoring, and reward-penalty adjustments. ASTS ensures contextually coherent and diverse text generation while maintaining computational efficiency. Its performance is evaluated across multiple benchmarks, including story generation and abstractive summarization, using metrics such as perplexity, MAUVE, and diversity scores. Experimental results demonstrate that ASTS outperforms existing sampling techniques by reducing repetition, enhancing semantic alignment, and improving fluency.
>
---
#### [new 053] Do LLMs Really Forget? Evaluating Unlearning with Knowledge Correlation and Confidence Awareness
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于知识遗忘评估任务，旨在解决大语言模型中未完全遗忘相关知识的问题。作者构建了一个基于知识图谱和置信度的评估框架，并设计推理协议使用LLMs作为评判者来验证遗忘效果，发现现有方法常高估遗忘成效。**

- **链接: [http://arxiv.org/pdf/2506.05735v1](http://arxiv.org/pdf/2506.05735v1)**

> **作者:** Rongzhe Wei; Peizhi Niu; Hans Hao-Hsun Hsu; Ruihan Wu; Haoteng Yin; Mohsen Ghassemi; Yifan Li; Vamsi K. Potluru; Eli Chien; Kamalika Chaudhuri; Olgica Milenkovic; Pan Li
>
> **摘要:** Machine unlearning techniques aim to mitigate unintended memorization in large language models (LLMs). However, existing approaches predominantly focus on the explicit removal of isolated facts, often overlooking latent inferential dependencies and the non-deterministic nature of knowledge within LLMs. Consequently, facts presumed forgotten may persist implicitly through correlated information. To address these challenges, we propose a knowledge unlearning evaluation framework that more accurately captures the implicit structure of real-world knowledge by representing relevant factual contexts as knowledge graphs with associated confidence scores. We further develop an inference-based evaluation protocol leveraging powerful LLMs as judges; these judges reason over the extracted knowledge subgraph to determine unlearning success. Our LLM judges utilize carefully designed prompts and are calibrated against human evaluations to ensure their trustworthiness and stability. Extensive experiments on our newly constructed benchmark demonstrate that our framework provides a more realistic and rigorous assessment of unlearning performance. Moreover, our findings reveal that current evaluation strategies tend to overestimate unlearning effectiveness. Our code is publicly available at https://github.com/Graph-COM/Knowledge_Unlearning.git.
>
---
#### [new 054] Large Language Models are Good Relational Learners
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于关系深度学习任务，旨在解决大语言模型处理结构化数据时忽略关系结构、冗余多、超限长的问题。论文提出Rel-LLM，结合图神经网络与检索增强生成框架，用结构化提示保留数据库关系，提升模型推理能力。**

- **链接: [http://arxiv.org/pdf/2506.05725v1](http://arxiv.org/pdf/2506.05725v1)**

> **作者:** Fang Wu; Vijay Prakash Dwivedi; Jure Leskovec
>
> **摘要:** Large language models (LLMs) have demonstrated remarkable capabilities across various domains, yet their application to relational deep learning (RDL) remains underexplored. Existing approaches adapt LLMs by traversing relational links between entities in a database and converting the structured data into flat text documents. Still, this text-based serialization disregards critical relational structures, introduces redundancy, and often exceeds standard LLM context lengths. We introduce Rel-LLM, a novel architecture that utilizes a graph neural network (GNN)- based encoder to generate structured relational prompts for LLMs within a retrieval-augmented generation (RAG) framework. Unlike traditional text-based serialization approaches, our method preserves the inherent relational structure of databases while enabling LLMs to effectively process and reason over complex entity relationships. Specifically, the GNN encoder extracts a local subgraph around an entity to build feature representations that contain relevant entity relationships and temporal dependencies. These representations are transformed into structured prompts using a denormalization process, effectively allowing the LLM to reason over relational structures. Through extensive experiments, we demonstrate that Rel-LLM outperforms existing methods on key RDL tasks, offering a scalable and efficient approach to integrating LLMs with structured data sources. Code is available at https://github.com/smiles724/Rel-LLM.
>
---
#### [new 055] Zero-Shot Event Causality Identification via Multi-source Evidence Fuzzy Aggregation with Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于事件因果关系识别（ECI）任务，旨在解决零样本场景下因果推理的幻觉问题。通过分解因果推理为多个子任务，利用大语言模型生成响应，并采用模糊聚合方法整合多源证据，有效提升了识别性能并减少了错误因果关联。**

- **链接: [http://arxiv.org/pdf/2506.05675v1](http://arxiv.org/pdf/2506.05675v1)**

> **作者:** Zefan Zeng; Xingchen Hu; Qing Cheng; Weiping Ding; Wentao Li; Zhong Liu
>
> **摘要:** Event Causality Identification (ECI) aims to detect causal relationships between events in textual contexts. Existing ECI models predominantly rely on supervised methodologies, suffering from dependence on large-scale annotated data. Although Large Language Models (LLMs) enable zero-shot ECI, they are prone to causal hallucination-erroneously establishing spurious causal links. To address these challenges, we propose MEFA, a novel zero-shot framework based on Multi-source Evidence Fuzzy Aggregation. First, we decompose causality reasoning into three main tasks (temporality determination, necessity analysis, and sufficiency verification) complemented by three auxiliary tasks. Second, leveraging meticulously designed prompts, we guide LLMs to generate uncertain responses and deterministic outputs. Finally, we quantify LLM's responses of sub-tasks and employ fuzzy aggregation to integrate these evidence for causality scoring and causality determination. Extensive experiments on three benchmarks demonstrate that MEFA outperforms second-best unsupervised baselines by 6.2% in F1-score and 9.3% in precision, while significantly reducing hallucination-induced errors. In-depth analysis verify the effectiveness of task decomposition and the superiority of fuzzy aggregation.
>
---
#### [new 056] MLLM-CL: Continual Learning for Multimodal Large Language Models
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于多模态大语言模型的持续学习任务，旨在解决模型在动态场景中持续集成新知识与技能时出现的灾难性遗忘问题。作者提出了MLLM-CL基准测试及基于参数隔离和路由机制的方法，有效减少了遗忘，显著提升了持续学习效果。**

- **链接: [http://arxiv.org/pdf/2506.05453v1](http://arxiv.org/pdf/2506.05453v1)**

> **作者:** Hongbo Zhao; Fei Zhu; Rundong Wang; Gaofeng Meng; Zhaoxiang Zhang
>
> **摘要:** Recent Multimodal Large Language Models (MLLMs) excel in vision-language understanding but face challenges in adapting to dynamic real-world scenarios that require continuous integration of new knowledge and skills. While continual learning (CL) offers a potential solution, existing benchmarks and methods suffer from critical limitations. In this paper, we introduce MLLM-CL, a novel benchmark encompassing domain and ability continual learning, where the former focuses on independently and identically distributed (IID) evaluation across evolving mainstream domains, whereas the latter evaluates on non-IID scenarios with emerging model ability. Methodologically, we propose preventing catastrophic interference through parameter isolation, along with an MLLM-based routing mechanism. Extensive experiments demonstrate that our approach can integrate domain-specific knowledge and functional abilities with minimal forgetting, significantly outperforming existing methods.
>
---
#### [new 057] MIRIAD: Augmenting LLMs with millions of medical query-response pairs
- **分类: cs.CL; I.2.7**

- **简介: 该论文属于医学知识增强大语言模型（LLM）任务，旨在解决LLM生成医疗内容不准确的问题。作者构建了大规模、结构化的医学问答对数据集MIRIAD，并开发了MIRIAD-Atlas用于知识探索。实验证明该方法提升了LLM在医疗问答中的准确性和对幻觉的检测能力。**

- **链接: [http://arxiv.org/pdf/2506.06091v1](http://arxiv.org/pdf/2506.06091v1)**

> **作者:** Qinyue Zheng; Salman Abdullah; Sam Rawal; Cyril Zakka; Sophie Ostmeier; Maximilian Purk; Eduardo Reis; Eric J. Topol; Jure Leskovec; Michael Moor
>
> **备注:** Preprint
>
> **摘要:** LLMs are bound to transform healthcare with advanced decision support and flexible chat assistants. However, LLMs are prone to generate inaccurate medical content. To ground LLMs in high-quality medical knowledge, LLMs have been equipped with external knowledge via RAG, where unstructured medical knowledge is split into small text chunks that can be selectively retrieved and integrated into the LLMs context. Yet, existing RAG pipelines rely on raw, unstructured medical text, which can be noisy, uncurated and difficult for LLMs to effectively leverage. Systematic approaches to organize medical knowledge to best surface it to LLMs are generally lacking. To address these challenges, we introduce MIRIAD, a large-scale, curated corpus of 5,821,948 medical QA pairs, each rephrased from and grounded in a passage from peer-reviewed medical literature using a semi-automated pipeline combining LLM generation, filtering, grounding, and human annotation. Unlike prior medical corpora, which rely on unstructured text, MIRIAD encapsulates web-scale medical knowledge in an operationalized query-response format, which enables more targeted retrieval. Experiments on challenging medical QA benchmarks show that augmenting LLMs with MIRIAD improves accuracy up to 6.7% compared to unstructured RAG baselines with the same source corpus and with the same amount of retrieved text. Moreover, MIRIAD improved the ability of LLMs to detect medical hallucinations by 22.5 to 37% (increase in F1 score). We further introduce MIRIAD-Atlas, an interactive map of MIRIAD spanning 56 medical disciplines, enabling clinical users to visually explore, search, and refine medical knowledge. MIRIAD promises to unlock a wealth of down-stream applications, including medical information retrievers, enhanced RAG applications, and knowledge-grounded chat interfaces, which ultimately enables more reliable LLM applications in healthcare.
>
---
#### [new 058] Large Language Models are Demonstration Pre-Selectors for Themselves
- **分类: cs.CL**

- **简介: 本文提出FEEDER框架，旨在解决大语言模型（LLM）在上下文学习（ICL）中因从全量训练数据中选择示例导致的高计算成本问题。通过引入“充分性”与“必要性”度量及树形算法，预先筛选出最具代表性的示例子集，用于替代完整数据集，从而提升效率并保持性能。该方法适用于不同规模LLM，并兼容多种下游任务策略。**

- **链接: [http://arxiv.org/pdf/2506.06033v1](http://arxiv.org/pdf/2506.06033v1)**

> **作者:** Jiarui Jin; Yuwei Wu; Haoxuan Li; Xiaoting He; Weinan Zhang; Yiming Yang; Yong Yu; Jun Wang; Mengyue Yang
>
> **备注:** ICML 2025
>
> **摘要:** In-context learning (ICL) with large language models (LLMs) delivers strong few-shot performance by choosing few-shot demonstrations from the entire training data. However, existing ICL methods, which rely on similarity or diversity scores to choose demonstrations, incur high computational costs due to repeatedly retrieval from large-scale datasets for each query. To this end, we propose FEEDER (FEw yet Essential Demonstration prE-selectoR), a novel pre-selection framework that identifies a representative subset of demonstrations containing the most representative examples in the training data, tailored to specific LLMs. To construct this subset, we introduce the "sufficiency" and "necessity" metrics in the pre-selection stage and design a tree-based algorithm to identify representative examples efficiently. Once pre-selected, this representative subset can effectively replace the full training data, improving efficiency while maintaining comparable performance in ICL. Additionally, our pre-selected subset also benefits fine-tuning LLMs, where we introduce a bi-level optimization method that enhances training efficiency without sacrificing performance. Experiments with LLMs ranging from 300M to 8B parameters show that FEEDER can reduce training data size by over 20% while maintaining performance and seamlessly integrating with various downstream demonstration selection strategies in ICL.
>
---
#### [new 059] LLM-Symbolic Integration for Robust Temporal Tabular Reasoning
- **分类: cs.CL**

- **简介: 该论文属于时序表格问答任务，旨在解决大语言模型在结构化数据推理中的局限性。作者提出了 TempTabQA-C 合成数据集和符号中间表示方法，结合自适应少样本提示，提升模型对复杂查询的鲁棒性和泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.05746v1](http://arxiv.org/pdf/2506.05746v1)**

> **作者:** Atharv Kulkarni; Kushagra Dixit; Vivek Srikumar; Dan Roth; Vivek Gupta
>
> **备注:** Accepted to ACL Findings 2025
>
> **摘要:** Temporal tabular question answering presents a significant challenge for Large Language Models (LLMs), requiring robust reasoning over structured data, which is a task where traditional prompting methods often fall short. These methods face challenges such as memorization, sensitivity to table size, and reduced performance on complex queries. To overcome these limitations, we introduce TempTabQA-C, a synthetic dataset designed for systematic and controlled evaluations, alongside a symbolic intermediate representation that transforms tables into database schemas. This structured approach allows LLMs to generate and execute SQL queries, enhancing generalization and mitigating biases. By incorporating adaptive few-shot prompting with contextually tailored examples, our method achieves superior robustness, scalability, and performance. Experimental results consistently highlight improvements across key challenges, setting a new benchmark for robust temporal reasoning with LLMs.
>
---
#### [new 060] SmoothRot: Combining Channel-Wise Scaling and Rotation for Quantization-Friendly LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 论文提出SmoothRot，一种适用于大语言模型的4位量化技术，旨在解决激活值中大量异常值导致量化效果差的问题。通过结合通道级缩放与Hadamard变换，将异常值转化为更适合量化的形式，提升量化模型的准确性，且不增加推理延迟。**

- **链接: [http://arxiv.org/pdf/2506.05413v1](http://arxiv.org/pdf/2506.05413v1)**

> **作者:** Patrik Czakó; Gábor Kertész; Sándor Szénási
>
> **备注:** 6 pages, 3 figures, 5 tables. Submitted to the IEEE SMC 2025 conference
>
> **摘要:** We present SmoothRot, a novel post-training quantization technique to enhance the efficiency of 4-bit quantization in Large Language Models (LLMs). SmoothRot addresses the critical challenge of massive activation outliers, by integrating channel-wise scaling with Hadamard transformations. Our technique effectively transforms extreme outliers into quantization-friendly activations, significantly improving quantization accuracy. Experiments conducted on popular LLMs (LLaMA2 7B, LLaMA3.1 8B, and Mistral 7B) demonstrate that SmoothRot consistently reduces the performance gap between quantized and FP16 models by approximately 10-30\% across language generation and zero-shot reasoning tasks, without introducing additional inference latency. Code is available at https://github.com/czakop/smoothrot.
>
---
#### [new 061] MoA: Heterogeneous Mixture of Adapters for Parameter-Efficient Fine-Tuning of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 论文提出异构混合适配器（MoA），用于大语言模型的参数高效微调。针对现有同构MoE-LoRA方法存在的表征崩溃和专家负载不平衡问题，设计了软MoA和稀疏MoA两种变体，通过融合结构多样的适配器专家提升模型性能与参数效率。**

- **链接: [http://arxiv.org/pdf/2506.05928v1](http://arxiv.org/pdf/2506.05928v1)**

> **作者:** Jie Cao; Tianwei Lin; Hongyang He; Rolan Yan; Wenqiao Zhang; Juncheng Li; Dongping Zhang; Siliang Tang; Yueting Zhuang
>
> **摘要:** Recent studies integrate Low-Rank Adaptation (LoRA) and Mixture-of-Experts (MoE) to further enhance the performance of parameter-efficient fine-tuning (PEFT) methods in Large Language Model (LLM) applications. Existing methods employ \emph{homogeneous} MoE-LoRA architectures composed of LoRA experts with either similar or identical structures and capacities. However, these approaches often suffer from representation collapse and expert load imbalance, which negatively impact the potential of LLMs. To address these challenges, we propose a \emph{heterogeneous} \textbf{Mixture-of-Adapters (MoA)} approach. This method dynamically integrates PEFT adapter experts with diverse structures, leveraging their complementary representational capabilities to foster expert specialization, thereby enhancing the effective transfer of pre-trained knowledge to downstream tasks. MoA supports two variants: \textbf{(i)} \textit{Soft MoA} achieves fine-grained integration by performing a weighted fusion of all expert outputs; \textbf{(ii)} \textit{Sparse MoA} activates adapter experts sparsely based on their contribution, achieving this with negligible performance degradation. Experimental results demonstrate that heterogeneous MoA outperforms homogeneous MoE-LoRA methods in both performance and parameter efficiency. Our project is available at https://github.com/DCDmllm/MoA.
>
---
#### [new 062] LLMs Can Also Do Well! Breaking Barriers in Semantic Role Labeling via Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的语义角色标注（SRL）任务，旨在解决生成式大语言模型（LLMs）在SRL上表现不如编码器-解码器模型的问题。论文通过引入检索增强生成和自纠错机制，提升了LLMs在SRL任务上的性能，并在多个基准测试中取得了最优结果。**

- **链接: [http://arxiv.org/pdf/2506.05385v1](http://arxiv.org/pdf/2506.05385v1)**

> **作者:** Xinxin Li; Huiyao Chen; Chengjun Liu; Jing Li; Meishan Zhang; Jun Yu; Min Zhang
>
> **备注:** 19 pages, 3 figures, 10 tables
>
> **摘要:** Semantic role labeling (SRL) is a crucial task of natural language processing (NLP). Although generative decoder-based large language models (LLMs) have achieved remarkable success across various NLP tasks, they still lag behind state-of-the-art encoder-decoder (BERT-like) models in SRL. In this work, we seek to bridge this gap by equipping LLMs for SRL with two mechanisms: (a) retrieval-augmented generation and (b) self-correction. The first mechanism enables LLMs to leverage external linguistic knowledge such as predicate and argument structure descriptions, while the second allows LLMs to identify and correct inconsistent SRL outputs. We conduct extensive experiments on three widely-used benchmarks of SRL (CPB1.0, CoNLL-2009, and CoNLL-2012). Results demonstrate that our method achieves state-of-the-art performance in both Chinese and English, marking the first successful application of LLMs to surpass encoder-decoder approaches in SRL.
>
---
#### [new 063] dots.llm1 Technical Report
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型训练任务，旨在高效扩展模型规模并降低训练与推理成本。论文提出了dots.llm1，一个大规模Mixture of Experts（MoE）模型，在仅激活140亿参数的情况下，性能媲美当前最先进的模型。通过高效的数据处理流程，模型在高质量数据上预训练并进一步优化，达到与Qwen2.5-72B相当的性能，并开源训练过程中的中间检查点以推动研究进展。**

- **链接: [http://arxiv.org/pdf/2506.05767v1](http://arxiv.org/pdf/2506.05767v1)**

> **作者:** Bi Huo; Bin Tu; Cheng Qin; Da Zheng; Debing Zhang; Dongjie Zhang; En Li; Fu Guo; Jian Yao; Jie Lou; Junfeng Tian; Li Hu; Ran Zhu; Shengdong Chen; Shuo Liu; Su Guang; Te Wo; Weijun Zhang; Xiaoming Shi; Xinxin Peng; Xing Wu; Yawen Liu; Yuqiu Ji; Ze Wen; Zhenhai Liu; Zichao Li; Zilong Liao
>
> **摘要:** Mixture of Experts (MoE) models have emerged as a promising paradigm for scaling language models efficiently by activating only a subset of parameters for each input token. In this report, we present dots.llm1, a large-scale MoE model that activates 14B parameters out of a total of 142B parameters, delivering performance on par with state-of-the-art models while reducing training and inference costs. Leveraging our meticulously crafted and efficient data processing pipeline, dots.llm1 achieves performance comparable to Qwen2.5-72B after pretraining on 11.2T high-quality tokens and post-training to fully unlock its capabilities. Notably, no synthetic data is used during pretraining. To foster further research, we open-source intermediate training checkpoints at every one trillion tokens, providing valuable insights into the learning dynamics of large language models.
>
---
#### [new 064] MAPLE: Multi-Agent Adaptive Planning with Long-Term Memory for Table Reasoning
- **分类: cs.CL**

- **简介: 论文提出MAPLE框架，用于表格推理任务，解决当前LLM单次推理难以应对复杂推理的问题。通过结合Solver、Checker、Reflector和Archiver四个组件，实现多智能体协作、错误检测与长期记忆管理，显著提升WiKiTQ和TabFact数据集上的性能。**

- **链接: [http://arxiv.org/pdf/2506.05813v1](http://arxiv.org/pdf/2506.05813v1)**

> **作者:** Ye Bai; Minghan Wang; Thuy-Trang Vu
>
> **备注:** 26 pages, 10 figures
>
> **摘要:** Table-based question answering requires complex reasoning capabilities that current LLMs struggle to achieve with single-pass inference. Existing approaches, such as Chain-of-Thought reasoning and question decomposition, lack error detection mechanisms and discard problem-solving experiences, contrasting sharply with how humans tackle such problems. In this paper, we propose MAPLE (Multi-agent Adaptive Planning with Long-term mEmory), a novel framework that mimics human problem-solving through specialized cognitive agents working in a feedback-driven loop. MAPLE integrates 4 key components: (1) a Solver using the ReAct paradigm for reasoning, (2) a Checker for answer verification, (3) a Reflector for error diagnosis and strategy correction, and (4) an Archiver managing long-term memory for experience reuse and evolution. Experiments on WiKiTQ and TabFact demonstrate significant improvements over existing methods, achieving state-of-the-art performance across multiple LLM backbones.
>
---
#### [new 065] NameTag 3: A Tool and a Service for Multilingual/Multitagset NER
- **分类: cs.CL**

- **简介: 论文介绍了NameTag 3，一个支持多语言、多数据集和多标签集的命名实体识别（NER）工具和服务。它解决了多语言环境下不同标签体系和嵌套实体识别的问题，采用统一模型实现高效准确的NER，并提供开源工具与云端服务。**

- **链接: [http://arxiv.org/pdf/2506.05949v1](http://arxiv.org/pdf/2506.05949v1)**

> **作者:** Jana Straková; Milan Straka
>
> **备注:** Accepted to ACL 2025
>
> **摘要:** We introduce NameTag 3, an open-source tool and cloud-based web service for multilingual, multidataset, and multitagset named entity recognition (NER), supporting both flat and nested entities. NameTag 3 achieves state-of-the-art results on 21 test datasets in 15 languages and remains competitive on the rest, even against larger models. It is available as a command-line tool and as a cloud-based service, enabling use without local installation. NameTag 3 web service currently provides flat NER for 17 languages, trained on 21 corpora and three NE tagsets, all powered by a single 355M-parameter fine-tuned model; and nested NER for Czech, powered by a 126M fine-tuned model. The source code is licensed under open-source MPL 2.0, while the models are distributed under non-commercial CC BY-NC-SA 4.0. Documentation is available at https://ufal.mff.cuni.cz/nametag, source code at https://github.com/ufal/nametag3, and trained models via https://lindat.cz. The REST service and the web application can be found at https://lindat.mff.cuni.cz/services/nametag/. A demonstration video is available at https://www.youtube.com/watch?v=-gaGnP0IV8A.
>
---
#### [new 066] Unlocking Recursive Thinking of LLMs: Alignment via Refinement
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升大语言模型的递归思维能力。通过提出AvR方法，利用长链思维（CoT）进行对齐优化，结合批评与改进机制，显著提高模型性能。实验表明其效果优于传统方法。**

- **链接: [http://arxiv.org/pdf/2506.06009v1](http://arxiv.org/pdf/2506.06009v1)**

> **作者:** Haoke Zhang; Xiaobo Liang; Cunxiang Wang; Juntao Li; Min Zhang
>
> **备注:** Accepted to the Findings of ACL 2025
>
> **摘要:** The OpenAI o1-series models have demonstrated that leveraging long-form Chain of Thought (CoT) can substantially enhance performance. However, the recursive thinking capabilities of Large Language Models (LLMs) remain limited, particularly in the absence of expert-curated data for distillation. In this paper, we propose \textbf{AvR}: \textbf{Alignment via Refinement}, a novel method aimed at unlocking the potential of LLMs for recursive reasoning through long-form CoT. AvR introduces a refinement process that integrates criticism and improvement actions, guided by differentiable learning techniques to optimize \textbf{refinement-aware rewards}. As a result, the synthesized multi-round data can be organized as a long refinement thought, further enabling test-time scaling. Experimental results show that AvR significantly outperforms conventional preference optimization methods. Notably, with only 3k synthetic samples, our method boosts the performance of the LLaMA-3-8B-Instruct model by over 20\% in win rate on AlpacaEval 2.0. Our code is available at Github (https://github.com/Banner-Z/AvR.git).
>
---
#### [new 067] Multidimensional Analysis of Specific Language Impairment Using Unsupervised Learning Through PCA and Clustering
- **分类: cs.CL; cs.LG; 62H30, 62P10; I.2.7; J.3**

- **简介: 该论文属于语言障碍分析任务，旨在通过无监督学习技术探索特定语言障碍（SLI）儿童的语言发展模式。研究使用PCA和聚类方法分析1,163名儿童的叙事语料，提取64个语言特征，识别出两类主要语言模式，并发现SLI主要表现为语言产出能力低下而非句法复杂度问题，挑战了传统诊断方式，提出了基于机器学习的新视角。**

- **链接: [http://arxiv.org/pdf/2506.05498v1](http://arxiv.org/pdf/2506.05498v1)**

> **作者:** Niruthiha Selvanayagam
>
> **备注:** 14 pages, 3 figures, 16 tables
>
> **摘要:** Specific Language Impairment (SLI) affects approximately 7 percent of children, presenting as isolated language deficits despite normal cognitive abilities, sensory systems, and supportive environments. Traditional diagnostic approaches often rely on standardized assessments, which may overlook subtle developmental patterns. This study aims to identify natural language development trajectories in children with and without SLI using unsupervised machine learning techniques, providing insights for early identification and targeted interventions. Narrative samples from 1,163 children aged 4-16 years across three corpora (Conti-Ramsden 4, ENNI, and Gillam) were analyzed using Principal Component Analysis (PCA) and clustering. A total of 64 linguistic features were evaluated to uncover developmental trajectories and distinguish linguistic profiles. Two primary clusters emerged: (1) high language production with low SLI prevalence, and (2) limited production but higher syntactic complexity with higher SLI prevalence. Additionally, boundary cases exhibited intermediate traits, supporting a continuum model of language abilities. Findings suggest SLI manifests primarily through reduced production capacity rather than syntactic complexity deficits. The results challenge categorical diagnostic frameworks and highlight the potential of unsupervised learning techniques for refining diagnostic criteria and intervention strategies.
>
---
#### [new 068] Simple Yet Effective: Extracting Private Data Across Clients in Federated Fine-Tuning of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究联邦微调大型语言模型中的隐私数据提取问题。任务是设计攻击算法，评估模型泄露风险。作者提出新攻击方法与评估框架，在更现实威胁模型下有效提取跨客户端的敏感个人信息，揭示需加强防御策略的重要性。**

- **链接: [http://arxiv.org/pdf/2506.06060v1](http://arxiv.org/pdf/2506.06060v1)**

> **作者:** Yingqi Hu; Zhuo Zhang; Jingyuan Zhang; Lizhen Qu; Zenglin Xu
>
> **备注:** 10 pages, 4 figures
>
> **摘要:** Federated fine-tuning of large language models (FedLLMs) presents a promising approach for achieving strong model performance while preserving data privacy in sensitive domains. However, the inherent memorization ability of LLMs makes them vulnerable to training data extraction attacks. To investigate this risk, we introduce simple yet effective extraction attack algorithms specifically designed for FedLLMs. In contrast to prior "verbatim" extraction attacks, which assume access to fragments from all training data, our approach operates under a more realistic threat model, where the attacker only has access to a single client's data and aims to extract previously unseen personally identifiable information (PII) from other clients. This requires leveraging contextual prefixes held by the attacker to generalize across clients. To evaluate the effectiveness of our approaches, we propose two rigorous metrics-coverage rate and efficiency-and extend a real-world legal dataset with PII annotations aligned with CPIS, GDPR, and CCPA standards, achieving 89.9% human-verified precision. Experimental results show that our method can extract up to 56.57% of victim-exclusive PII, with "Address," "Birthday," and "Name" being the most vulnerable categories. Our findings underscore the pressing need for robust defense strategies and contribute a new benchmark and evaluation framework for future research in privacy-preserving federated learning.
>
---
#### [new 069] IntentionESC: An Intention-Centered Framework for Enhancing Emotional Support in Dialogue Systems
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统中的情感支持任务，旨在解决因意图不明确导致的支持策略不当问题。作者提出了IntentionESC框架，通过定义支持者意图、识别情绪状态并映射到合适策略，提升情感支持效果。同时引入ICECoT机制，使大语言模型能模拟人类推理过程，并设计自动标注流程和评估方案验证方法有效性。**

- **链接: [http://arxiv.org/pdf/2506.05947v1](http://arxiv.org/pdf/2506.05947v1)**

> **作者:** Xinjie Zhang; Wenxuan Wang; Qin Jin
>
> **备注:** ACL2025 findings
>
> **摘要:** In emotional support conversations, unclear intentions can lead supporters to employ inappropriate strategies, inadvertently imposing their expectations or solutions on the seeker. Clearly defined intentions are essential for guiding both the supporter's motivations and the overall emotional support process. In this paper, we propose the Intention-centered Emotional Support Conversation (IntentionESC) framework, which defines the possible intentions of supporters in emotional support conversations, identifies key emotional state aspects for inferring these intentions, and maps them to appropriate support strategies. While Large Language Models (LLMs) excel in text generating, they fundamentally operate as probabilistic models trained on extensive datasets, lacking a true understanding of human thought processes and intentions. To address this limitation, we introduce the Intention Centric Chain-of-Thought (ICECoT) mechanism. ICECoT enables LLMs to mimic human reasoning by analyzing emotional states, inferring intentions, and selecting suitable support strategies, thereby generating more effective emotional support responses. To train the model with ICECoT and integrate expert knowledge, we design an automated annotation pipeline that produces high-quality training data. Furthermore, we develop a comprehensive evaluation scheme to assess emotional support efficacy and conduct extensive experiments to validate our framework. Our data and code are available at https://github.com/43zxj/IntentionESC_ICECoT.
>
---
#### [new 070] Discrete Minds in a Continuous World: Do Language Models Know Time Passes?
- **分类: cs.CL**

- **简介: 该论文研究大语言模型（LLMs）是否具备感知时间流逝的能力。通过三个实验，验证了LLMs可将离散token映射为真实时间，并能在问答和动态导航任务中根据时间压力调整行为。论文属于自然语言处理与时间推理交叉任务，旨在解决LLMs在时间感知与决策适应性方面的局限性。**

- **链接: [http://arxiv.org/pdf/2506.05790v1](http://arxiv.org/pdf/2506.05790v1)**

> **作者:** Minghan Wang; Ye Bai; Thuy-Trang Vu; Ehsan Shareghi; Gholamreza Haffari
>
> **摘要:** While Large Language Models (LLMs) excel at temporal reasoning tasks like event ordering and duration estimation, their ability to perceive the actual passage of time remains unexplored. We investigate whether LLMs perceive the passage of time and adapt their decision-making accordingly through three complementary experiments. First, we introduce the Token-Time Hypothesis, positing that LLMs can map discrete token counts to continuous wall-clock time, and validate this through a dialogue duration judgment task. Second, we demonstrate that LLMs could use this awareness to adapt their response length while maintaining accuracy when users express urgency in question answering tasks. Finally, we develop BombRush, an interactive navigation challenge that examines how LLMs modify behavior under progressive time pressure in dynamic environments. Our findings indicate that LLMs possess certain awareness of time passage, enabling them to bridge discrete linguistic tokens and continuous physical time, though this capability varies with model size and reasoning abilities. This work establishes a theoretical foundation for enhancing temporal awareness in LLMs for time-sensitive applications.
>
---
#### [new 071] When to Trust Context: Self-Reflective Debates for Context Reliability
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型在参数知识与上下文输入冲突时产生的事实不一致或幻觉问题。论文提出了一种轻量级框架SR-DCR，通过结合令牌级自信心和非对称多智能体辩论机制，判断上下文的可靠性，并据此选择最终答案。实验表明该方法在ClashEval基准上提升了鲁棒性与准确性。**

- **链接: [http://arxiv.org/pdf/2506.06020v1](http://arxiv.org/pdf/2506.06020v1)**

> **作者:** Zeqi Zhou; Fang Wu; Shayan Talaei; Haokai Zhao; Cheng Meixin; Tinson Xu; Amin Saberi; Yejin Choi
>
> **摘要:** Large language models frequently encounter conflicts between their parametric knowledge and contextual input, often resulting in factual inconsistencies or hallucinations. We propose Self-Reflective Debate for Contextual Reliability (SR-DCR), a lightweight framework that integrates token-level self-confidence with an asymmetric multi-agent debate to adjudicate such conflicts. A critic, deprived of context, challenges a defender who argues from the given passage; a judge model evaluates the debate and determines the context's reliability. The final answer is selected by combining the verdict with model confidence. Experiments on the ClashEval benchmark demonstrate that SR-DCR consistently enhances robustness to misleading context while maintaining accuracy on trustworthy inputs, outperforming both classical debate and confidence-only baselines with minimal computational overhead. The code is available at https://github.com/smiles724/Self-Reflective-Debates.
>
---
#### [new 072] Combating Misinformation in the Arab World: Challenges & Opportunities
- **分类: cs.CL; cs.AI; cs.SI; 68T50; I.2.7**

- **简介: 该论文探讨阿拉伯地区虚假信息治理任务，旨在应对地缘政治不稳定、语言多样性和文化特性带来的挑战。研究聚焦虚假信息的检测、追踪、缓解及社区参与，提出通过基层事实核查合作、文化规范理解与社交纠正等策略，构建更具韧性的信息生态体系。**

- **链接: [http://arxiv.org/pdf/2506.05582v1](http://arxiv.org/pdf/2506.05582v1)**

> **作者:** Azza Abouzied; Firoj Alam; Raian Ali; Paolo Papotti
>
> **备注:** disinformation, misinformation, factuality, harmfulness, fake news
>
> **摘要:** Misinformation and disinformation pose significant risks globally, with the Arab region facing unique vulnerabilities due to geopolitical instabilities, linguistic diversity, and cultural nuances. We explore these challenges through the key facets of combating misinformation: detection, tracking, mitigation and community-engagement. We shed light on how connecting with grass-roots fact-checking organizations, understanding cultural norms, promoting social correction, and creating strong collaborative information networks can create opportunities for a more resilient information ecosystem in the Arab world.
>
---
#### [new 073] Does It Run and Is That Enough? Revisiting Text-to-Chart Generation with a Multi-Agent Approach
- **分类: cs.CL**

- **简介: 该论文属于文本生成图表任务，旨在解决生成图表代码的执行错误问题。通过构建多智能体流程，使用GPT-4o-mini模型降低执行错误率，并在多个基准上表现优异。研究发现执行问题已基本解决，但图表美观性、语义准确性和可访问性仍需改进。**

- **链接: [http://arxiv.org/pdf/2506.06175v1](http://arxiv.org/pdf/2506.06175v1)**

> **作者:** James Ford; Anthony Rios
>
> **备注:** 8 pages
>
> **摘要:** Large language models can translate natural-language chart descriptions into runnable code, yet approximately 15\% of the generated scripts still fail to execute, even after supervised fine-tuning and reinforcement learning. We investigate whether this persistent error rate stems from model limitations or from reliance on a single-prompt design. To explore this, we propose a lightweight multi-agent pipeline that separates drafting, execution, repair, and judgment, using only an off-the-shelf GPT-4o-mini model. On the \textsc{Text2Chart31} benchmark, our system reduces execution errors to 4.5\% within three repair iterations, outperforming the strongest fine-tuned baseline by nearly 5 percentage points while requiring significantly less compute. Similar performance is observed on the \textsc{ChartX} benchmark, with an error rate of 4.6\%, demonstrating strong generalization. Under current benchmarks, execution success appears largely solved. However, manual review reveals that 6 out of 100 sampled charts contain hallucinations, and an LLM-based accessibility audit shows that only 33.3\% (\textsc{Text2Chart31}) and 7.2\% (\textsc{ChartX}) of generated charts satisfy basic colorblindness guidelines. These findings suggest that future work should shift focus from execution reliability toward improving chart aesthetics, semantic fidelity, and accessibility.
>
---
#### [new 074] MATP-BENCH: Can MLLM Be a Good Automated Theorem Prover for Multimodal Problems?
- **分类: cs.CL**

- **简介: 该论文属于自动化定理证明任务，旨在探索多模态大语言模型（MLLM）是否能成为有效的多模态自动定理证明器。为解决这一问题，作者构建了MATP-BENCH基准，包含1056个多模态定理及其在Lean 4、Coq和Isabelle中的形式化表示，并用于评估现有MLLM的性能，结果显示此任务仍具挑战性。**

- **链接: [http://arxiv.org/pdf/2506.06034v1](http://arxiv.org/pdf/2506.06034v1)**

> **作者:** Zhitao He; Zongwei Lyu; Dazhong Chen; Dadi Guo; Yi R. Fung
>
> **备注:** 29 pages
>
> **摘要:** Numerous theorems, such as those in geometry, are often presented in multimodal forms (e.g., diagrams). Humans benefit from visual reasoning in such settings, using diagrams to gain intuition and guide the proof process. Modern Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities in solving a wide range of mathematical problems. However, the potential of MLLMs as Automated Theorem Provers (ATPs), specifically in the multimodal domain, remains underexplored. In this paper, we introduce the Multimodal Automated Theorem Proving benchmark (MATP-BENCH), a new Multimodal, Multi-level, and Multi-language benchmark designed to evaluate MLLMs in this role as multimodal automated theorem provers. MATP-BENCH consists of 1056 multimodal theorems drawn from high school, university, and competition-level mathematics. All these multimodal problems are accompanied by formalizations in Lean 4, Coq and Isabelle, thus making the benchmark compatible with a wide range of theorem-proving frameworks. MATP-BENCH requires models to integrate sophisticated visual understanding with mastery of a broad spectrum of mathematical knowledge and rigorous symbolic reasoning to generate formal proofs. We use MATP-BENCH to evaluate a variety of advanced multimodal language models. Existing methods can only solve a limited number of the MATP-BENCH problems, indicating that this benchmark poses an open challenge for research on automated theorem proving.
>
---
#### [new 075] MMTU: A Massive Multi-Task Table Understanding and Reasoning Benchmark
- **分类: cs.AI; cs.CL; cs.DB; cs.LG**

- **简介: 该论文属于表格理解和推理任务，旨在解决当前对模型处理表格能力评估不足的问题。作者构建了大规模基准MMTU，包含25个真实世界表格任务、超3万问题，全面评估模型对表格的理解、推理与操作能力，推动结构化数据处理模型的发展。**

- **链接: [http://arxiv.org/pdf/2506.05587v1](http://arxiv.org/pdf/2506.05587v1)**

> **作者:** Junjie Xing; Yeye He; Mengyu Zhou; Haoyu Dong; Shi Han; Lingjiao Chen; Dongmei Zhang; Surajit Chaudhuri; H. V. Jagadish
>
> **摘要:** Tables and table-based use cases play a crucial role in many important real-world applications, such as spreadsheets, databases, and computational notebooks, which traditionally require expert-level users like data engineers, data analysts, and database administrators to operate. Although LLMs have shown remarkable progress in working with tables (e.g., in spreadsheet and database copilot scenarios), comprehensive benchmarking of such capabilities remains limited. In contrast to an extensive and growing list of NLP benchmarks, evaluations of table-related tasks are scarce, and narrowly focus on tasks like NL-to-SQL and Table-QA, overlooking the broader spectrum of real-world tasks that professional users face. This gap limits our understanding and model progress in this important area. In this work, we introduce MMTU, a large-scale benchmark with over 30K questions across 25 real-world table tasks, designed to comprehensively evaluate models ability to understand, reason, and manipulate real tables at the expert-level. These tasks are drawn from decades' worth of computer science research on tabular data, with a focus on complex table tasks faced by professional users. We show that MMTU require a combination of skills -- including table understanding, reasoning, and coding -- that remain challenging for today's frontier models, where even frontier reasoning models like OpenAI o4-mini and DeepSeek R1 score only around 60%, suggesting significant room for improvement. We highlight key findings in our evaluation using MMTU and hope that this benchmark drives further advances in understanding and developing foundation models for structured data processing and analysis. Our code and data are available at https://github.com/MMTU-Benchmark/MMTU and https://huggingface.co/datasets/MMTU-benchmark/MMTU.
>
---
#### [new 076] LLMs Can Compensate for Deficiencies in Visual Representations
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉-语言模型任务，旨在解决CLIP视觉编码器的表示缺陷问题。通过注意力机制实验，研究发现强大的语言解码器可补偿弱视觉特征，恢复性能。这表明模型中存在动态分工，为未来设计提供新思路。**

- **链接: [http://arxiv.org/pdf/2506.05439v1](http://arxiv.org/pdf/2506.05439v1)**

> **作者:** Sho Takishita; Jay Gala; Abdelrahman Mohamed; Kentaro Inui; Yova Kementchedjhieva
>
> **摘要:** Many vision-language models (VLMs) that prove very effective at a range of multimodal task, build on CLIP-based vision encoders, which are known to have various limitations. We investigate the hypothesis that the strong language backbone in VLMs compensates for possibly weak visual features by contextualizing or enriching them. Using three CLIP-based VLMs, we perform controlled self-attention ablations on a carefully designed probing task. Our findings show that despite known limitations, CLIP visual representations offer ready-to-read semantic information to the language decoder. However, in scenarios of reduced contextualization in the visual representations, the language decoder can largely compensate for the deficiency and recover performance. This suggests a dynamic division of labor in VLMs and motivates future architectures that offload more visual processing to the language decoder.
>
---
#### [new 077] CodeContests+: High-Quality Test Case Generation for Competitive Programming
- **分类: cs.SE; cs.CL**

- **简介: 该论文属于测试用例生成任务，旨在解决竞争性编程中高质量测试用例缺乏的问题。作者构建了一个基于大语言模型的代理系统，生成更高质量的测试用例，并在CodeContests数据集上推出改进版本CodeContests+，验证了其在评估准确性和强化学习中的优势。**

- **链接: [http://arxiv.org/pdf/2506.05817v1](http://arxiv.org/pdf/2506.05817v1)**

> **作者:** Zihan Wang; Siyao Liu; Yang Sun; Hongyan Li; Kai Shen
>
> **备注:** 28 pages, 7 figures
>
> **摘要:** Competitive programming, due to its high reasoning difficulty and precise correctness feedback, has become a key task for both training and evaluating the reasoning capabilities of large language models (LLMs). However, while a large amount of public problem data, such as problem statements and solutions, is available, the test cases of these problems are often difficult to obtain. Therefore, test case generation is a necessary task for building large-scale datasets, and the quality of the test cases directly determines the accuracy of the evaluation. In this paper, we introduce an LLM-based agent system that creates high-quality test cases for competitive programming problems. We apply this system to the CodeContests dataset and propose a new version with improved test cases, named CodeContests+. We evaluated the quality of test cases in CodeContestsPlus. First, we used 1.72 million submissions with pass/fail labels to examine the accuracy of these test cases in evaluation. The results indicated that CodeContests+ achieves significantly higher accuracy than CodeContests, particularly with a notably higher True Positive Rate (TPR). Subsequently, our experiments in LLM Reinforcement Learning (RL) further confirmed that improvements in test case quality yield considerable advantages for RL.
>
---
#### [new 078] Can Vision Language Models Infer Human Gaze Direction? A Controlled Study
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉语言模型（VLMs）是否具备推断人类注视方向的能力。任务属于人机交互与人工智能领域，旨在解决VLMs在理解人类注意力方面的局限性。作者通过控制实验评估111个VLMs的表现，并与人类对比，分析其行为特征。**

- **链接: [http://arxiv.org/pdf/2506.05412v1](http://arxiv.org/pdf/2506.05412v1)**

> **作者:** Zory Zhang; Pinyuan Feng; Bingyang Wang; Tianwei Zhao; Suyang Yu; Qingying Gao; Hokin Deng; Ziqiao Ma; Yijiang Li; Dezhi Luo
>
> **备注:** Preprint under review. Project page at https://grow-ai-like-a-child.github.io/gaze/
>
> **摘要:** Gaze-referential inference--the ability to infer what others are looking at--is a critical component of a theory of mind that underpins natural human-AI interaction. In a controlled study, we evaluated this skill across 111 Vision Language Models (VLMs) using photos taken with manipulated difficulty and variability, comparing performance with that of human participants (N = 65), and analyzed behaviors using mixed-effects models. We found that 94 of the 111 VLMs failed to do better than random guessing, while humans achieved near-ceiling accuracy. VLMs even respond with each choice almost equally frequently. Are they randomly guessing? Although most VLMs struggle, when we zoom in on five of the top-tier VLMs with above-chance performance, we find that their performance declined with increasing task difficulty but varied only slightly across different prompts and scene objects. These behavioral features cannot be explained by considering them as random guessers. Instead, they likely use a combination of heuristics and guessing such that their performance is subject to the task difficulty but robust to perceptual variations. This suggests that VLMs, lacking gaze inference capability, have yet to become technologies that can naturally interact with humans, but the potential remains.
>
---
#### [new 079] Coordinated Robustness Evaluation Framework for Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉-语言模型的鲁棒性评估任务，旨在解决模型在图像和文本扰动下的脆弱性问题。作者提出了一种协调攻击策略，通过训练通用代理模型生成跨模态对抗扰动，并在多个数据集上验证了其对多模态模型（如instruct-BLIP、ViLT）的有效性。**

- **链接: [http://arxiv.org/pdf/2506.05429v1](http://arxiv.org/pdf/2506.05429v1)**

> **作者:** Ashwin Ramesh Babu; Sajad Mousavi; Vineet Gundecha; Sahand Ghorbanpour; Avisek Naug; Antonio Guillen; Ricardo Luna Gutierrez; Soumyendu Sarkar
>
> **备注:** Accepted: IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops (CVPRW) 2025
>
> **摘要:** Vision-language models, which integrate computer vision and natural language processing capabilities, have demonstrated significant advancements in tasks such as image captioning and visual question and answering. However, similar to traditional models, they are susceptible to small perturbations, posing a challenge to their robustness, particularly in deployment scenarios. Evaluating the robustness of these models requires perturbations in both the vision and language modalities to learn their inter-modal dependencies. In this work, we train a generic surrogate model that can take both image and text as input and generate joint representation which is further used to generate adversarial perturbations for both the text and image modalities. This coordinated attack strategy is evaluated on the visual question and answering and visual reasoning datasets using various state-of-the-art vision-language models. Our results indicate that the proposed strategy outperforms other multi-modal attacks and single-modality attacks from the recent literature. Our results demonstrate their effectiveness in compromising the robustness of several state-of-the-art pre-trained multi-modal models such as instruct-BLIP, ViLT and others.
>
---
#### [new 080] PersonaAgent: When Large Language Model Agents Meet Personalization at Test Time
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出PersonaAgent，旨在解决大语言模型代理在个性化任务中的不足。通过结合个性化记忆与动作模块，并利用用户专属的“persona”作为中介，实现测试时用户偏好的实时对齐，从而提供更灵活、个性化的用户体验。**

- **链接: [http://arxiv.org/pdf/2506.06254v1](http://arxiv.org/pdf/2506.06254v1)**

> **作者:** Weizhi Zhang; Xinyang Zhang; Chenwei Zhang; Liangwei Yang; Jingbo Shang; Zhepei Wei; Henry Peng Zou; Zijie Huang; Zhengyang Wang; Yifan Gao; Xiaoman Pan; Lian Xiong; Jingguo Liu; Philip S. Yu; Xian Li
>
> **摘要:** Large Language Model (LLM) empowered agents have recently emerged as advanced paradigms that exhibit impressive capabilities in a wide range of domains and tasks. Despite their potential, current LLM agents often adopt a one-size-fits-all approach, lacking the flexibility to respond to users' varying needs and preferences. This limitation motivates us to develop PersonaAgent, the first personalized LLM agent framework designed to address versatile personalization tasks. Specifically, PersonaAgent integrates two complementary components - a personalized memory module that includes episodic and semantic memory mechanisms; a personalized action module that enables the agent to perform tool actions tailored to the user. At the core, the persona (defined as unique system prompt for each user) functions as an intermediary: it leverages insights from personalized memory to control agent actions, while the outcomes of these actions in turn refine the memory. Based on the framework, we propose a test-time user-preference alignment strategy that simulate the latest n interactions to optimize the persona prompt, ensuring real-time user preference alignment through textual loss feedback between simulated and ground-truth responses. Experimental evaluations demonstrate that PersonaAgent significantly outperforms other baseline methods by not only personalizing the action space effectively but also scaling during test-time real-world applications. These results underscore the feasibility and potential of our approach in delivering tailored, dynamic user experiences.
>
---
#### [new 081] Contextually Guided Transformers via Low-Rank Adaptation
- **分类: cs.LG; cs.CL**

- **简介: 论文提出一种基于低秩适应的上下文引导Transformer（CGT），旨在解决大语言模型依赖显式提示带来的计算开销问题。该模型通过在每个序列位置维护上下文摘要，动态调整权重，实现无需提示的自适应专业化处理。工作包括模型结构设计、上下文编码优化及可解释性增强技术，适用于语言建模与上下文学习任务。**

- **链接: [http://arxiv.org/pdf/2506.05672v1](http://arxiv.org/pdf/2506.05672v1)**

> **作者:** Andrey Zhmoginov; Jihwan Lee; Max Vladymyrov; Mark Sandler
>
> **摘要:** Large Language Models (LLMs) based on Transformers excel at text processing, but their reliance on prompts for specialized behavior introduces computational overhead. We propose a modification to a Transformer architecture that eliminates the need for explicit prompts by learning to encode context into the model's weights. Our Contextually Guided Transformer (CGT) model maintains a contextual summary at each sequence position, allowing it to update the weights on the fly based on the preceding context. This approach enables the model to self-specialize, effectively creating a tailored model for processing information following a given prefix. We demonstrate the effectiveness of our method on synthetic in-context learning tasks and language modeling benchmarks. Furthermore, we introduce techniques for enhancing the interpretability of the learned contextual representations, drawing connections to Variational Autoencoders and promoting smoother, more consistent context encoding. This work offers a novel direction for efficient and adaptable language modeling by integrating context directly into the model's architecture.
>
---
#### [new 082] Low-Resource Domain Adaptation for Speech LLMs via Text-Only Fine-Tuning
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音识别任务，旨在解决低资源场景下语音大模型（Speech LLMs）的领域自适应问题。通过仅使用目标领域文本进行微调，无需配对语音数据，提出实时评估机制以保持语音与文本对齐，实现有效领域迁移并保留原领域性能。**

- **链接: [http://arxiv.org/pdf/2506.05671v1](http://arxiv.org/pdf/2506.05671v1)**

> **作者:** Yangui Fang; Jing Peng; Xu Li; Yu Xi; Chengwei Zhang; Guohui Zhong; Kai Yu
>
> **摘要:** Recent advances in automatic speech recognition (ASR) have combined speech encoders with large language models (LLMs) through projection, forming Speech LLMs with strong performance. However, adapting them to new domains remains challenging, especially in low-resource settings where paired speech-text data is scarce. We propose a text-only fine-tuning strategy for Speech LLMs using unpaired target-domain text without requiring additional audio. To preserve speech-text alignment, we introduce a real-time evaluation mechanism during fine-tuning. This enables effective domain adaptation while maintaining source-domain performance. Experiments on LibriSpeech, SlideSpeech, and Medical datasets show that our method achieves competitive recognition performance, with minimal degradation compared to full audio-text fine-tuning. It also improves generalization to new domains without catastrophic forgetting, highlighting the potential of text-only fine-tuning for low-resource domain adaptation of ASR.
>
---
#### [new 083] SoK: Are Watermarks in LLMs Ready for Deployment?
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于自然语言处理与模型安全任务，旨在解决大型语言模型（LLMs）部署中的水印技术有效性问题。论文系统化分析了现有水印方法，提出了分类器评估其效果，并探讨了实际应用中的限制与挑战。**

- **链接: [http://arxiv.org/pdf/2506.05594v1](http://arxiv.org/pdf/2506.05594v1)**

> **作者:** Kieu Dang; Phung Lai; NhatHai Phan; Yelong Shen; Ruoming Jin; Abdallah Khreishah; My Thai
>
> **摘要:** Large Language Models (LLMs) have transformed natural language processing, demonstrating impressive capabilities across diverse tasks. However, deploying these models introduces critical risks related to intellectual property violations and potential misuse, particularly as adversaries can imitate these models to steal services or generate misleading outputs. We specifically focus on model stealing attacks, as they are highly relevant to proprietary LLMs and pose a serious threat to their security, revenue, and ethical deployment. While various watermarking techniques have emerged to mitigate these risks, it remains unclear how far the community and industry have progressed in developing and deploying watermarks in LLMs. To bridge this gap, we aim to develop a comprehensive systematization for watermarks in LLMs by 1) presenting a detailed taxonomy for watermarks in LLMs, 2) proposing a novel intellectual property classifier to explore the effectiveness and impacts of watermarks on LLMs under both attack and attack-free environments, 3) analyzing the limitations of existing watermarks in LLMs, and 4) discussing practical challenges and potential future directions for watermarks in LLMs. Through extensive experiments, we show that despite promising research outcomes and significant attention from leading companies and community to deploy watermarks, these techniques have yet to reach their full potential in real-world applications due to their unfavorable impacts on model utility of LLMs and downstream tasks. Our findings provide an insightful understanding of watermarks in LLMs, highlighting the need for practical watermarks solutions tailored to LLM deployment.
>
---
#### [new 084] Projectable Models: One-Shot Generation of Small Specialized Transformers from Large Ones
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决大模型计算成本高、知识冗余的问题。通过提出一种参数映射方法，将大型Transformer模型转化为小型专用模型，专注于特定任务所需的知识，提升效率与性能。**

- **链接: [http://arxiv.org/pdf/2506.05641v1](http://arxiv.org/pdf/2506.05641v1)**

> **作者:** Andrey Zhmoginov; Jihwan Lee; Mark Sandler
>
> **备注:** Presented at ES-FoMo II: 2nd Workshop on Efficient Systems for Foundation Models (ICML 2024)
>
> **摘要:** Modern Foundation Models (FMs) are typically trained on corpora spanning a wide range of different data modalities, topics and downstream tasks. Utilizing these models can be very computationally expensive and is out of reach for most consumer devices. Furthermore, most of the broad FM knowledge may actually be irrelevant for a specific task at hand. Here we explore a technique for mapping parameters of a large Transformer to parameters of a smaller specialized model. By making this transformation task-specific, we aim to capture a narrower scope of the knowledge needed for performing a specific task by a smaller model. We study our method on image modeling tasks, showing that performance of generated models exceeds that of universal conditional models.
>
---
#### [new 085] Proactive Assistant Dialogue Generation from Streaming Egocentric Videos
- **分类: cs.AI; cs.CL; cs.CV; cs.HC**

- **简介: 该论文属于对话生成任务，旨在解决实时感知任务指导中缺乏有效数据和评估方法的问题。作者提出了一个框架，包括合成对话数据集、自动评估指标及处理流视频的端到端模型，以支持开发能主动辅助用户的AI系统。**

- **链接: [http://arxiv.org/pdf/2506.05904v1](http://arxiv.org/pdf/2506.05904v1)**

> **作者:** Yichi Zhang; Xin Luna Dong; Zhaojiang Lin; Andrea Madotto; Anuj Kumar; Babak Damavandi; Joyce Chai; Seungwhan Moon
>
> **摘要:** Recent advances in conversational AI have been substantial, but developing real-time systems for perceptual task guidance remains challenging. These systems must provide interactive, proactive assistance based on streaming visual inputs, yet their development is constrained by the costly and labor-intensive process of data collection and system evaluation. To address these limitations, we present a comprehensive framework with three key contributions. First, we introduce a novel data curation pipeline that synthesizes dialogues from annotated egocentric videos, resulting in \dataset, a large-scale synthetic dialogue dataset spanning multiple domains. Second, we develop a suite of automatic evaluation metrics, validated through extensive human studies. Third, we propose an end-to-end model that processes streaming video inputs to generate contextually appropriate responses, incorporating novel techniques for handling data imbalance and long-duration videos. This work lays the foundation for developing real-time, proactive AI assistants capable of guiding users through diverse tasks. Project page: https://pro-assist.github.io/
>
---
#### [new 086] Constrained Sampling for Language Models Should Be Easy: An MCMC Perspective
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于自然语言生成任务，旨在解决语言模型在硬约束条件下采样时易扭曲分布的问题。作者提出基于MCMC的框架，确保每一步采样均满足约束、逐步逼近真实分布，并保持高效性，应用于程序模糊测试等场景。**

- **链接: [http://arxiv.org/pdf/2506.05754v1](http://arxiv.org/pdf/2506.05754v1)**

> **作者:** Emmanuel Anaya Gonzalez; Sairam Vaidya; Kanghee Park; Ruyi Ji; Taylor Berg-Kirkpatrick; Loris D'Antoni
>
> **摘要:** Constrained decoding enables Language Models (LMs) to produce samples that provably satisfy hard constraints. However, existing constrained-decoding approaches often distort the underlying model distribution, a limitation that is especially problematic in applications like program fuzzing, where one wants to generate diverse and valid program inputs for testing purposes. We propose a new constrained sampling framework based on Markov Chain Monte Carlo (MCMC) that simultaneously satisfies three core desiderata: constraint satisfying (every sample satisfies the constraint), monotonically converging (the sampling process converges to the true conditional distribution), and efficient (high-quality samples emerge in few steps). Our method constructs a proposal distribution over valid outputs and applies a Metropolis-Hastings acceptance criterion based on the LM's likelihood, ensuring principled and efficient exploration of the constrained space. Empirically, our sampler outperforms existing methods on both synthetic benchmarks and real-world program fuzzing tasks.
>
---
#### [new 087] Corrector Sampling in Language Models
- **分类: cs.LG; cs.CL**

- **简介: 论文提出Resample-Previous-Tokens（RPT）方法，用于改进自回归语言模型的生成过程。该方法通过迭代回溯并可能替换已生成文本中的部分标记，缓解了错误累积问题。实验表明，在预训练模型中应用RPT进行微调，能在推理和编码任务上取得显著提升。**

- **链接: [http://arxiv.org/pdf/2506.06215v1](http://arxiv.org/pdf/2506.06215v1)**

> **作者:** Itai Gat; Neta Shaul; Uriel Singer; Yaron Lipman
>
> **摘要:** Autoregressive language models accumulate errors due to their fixed, irrevocable left-to-right token generation. To address this, we propose a new sampling method called Resample-Previous-Tokens (RPT). RPT mitigates error accumulation by iteratively revisiting and potentially replacing tokens in a window of previously generated text. This method can be integrated into existing autoregressive models, preserving their next-token-prediction quality and speed. Fine-tuning a pretrained 8B parameter model with RPT for only 100B resulted in ~10% relative improvements on reasoning and coding benchmarks compared to the standard sampling.
>
---
#### [new 088] BAQ: Efficient Bit Allocation Quantization for Large Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于模型压缩任务，旨在解决大语言模型量化中位宽分配不合理导致性能下降的问题。工作提出BAQ框架，基于Hessian代理的敏感度度量，将位宽分配建模为凸优化问题，实现层间精度自适应分配。实验表明BAQ优于GPTQ，在多个模型规模上取得更低的困惑度。**

- **链接: [http://arxiv.org/pdf/2506.05664v1](http://arxiv.org/pdf/2506.05664v1)**

> **作者:** Chao Zhang; Li Wang; Samson Lasaulce; Merouane Debbah
>
> **摘要:** Post-training model quantization is a widely adopted technique for reducing the memory and computational costs of large language models (LLMs). However, most existing methods rely on uniform or heuristic bitwidth assignments, failing to account for the nonuniform sensitivity of weights to quantization noise. In this paper, we propose a novel framework for allocating quantization bitwidths based on sensitivity metrics derived from a Hessian proxy. We make key assumptions, which allow the layer/component-wise loss function to be expressed as an explicit function of the bitwidths. This enables a neat formulation of the bit allocation problem as a convex optimization task, whose closed-form solution adapts precision across weights to minimize the layer-wise quantization loss. Inspecting the solution provides several insights (such as the equal-loss structure), which are then exploited to design the proposed \textbf{BAQ} (Bit Allocation Quantization) algorithm. The proposed algorithm achieves a good trade-off between loss minimization and complexity and allows BAQ to be integrated into standard quantization pipelines with minimal overhead. Experimental results show that BAQ consistently outperforms GPTQ, achieving up to 56$\times$ lower perplexity at the same bitwidth on large language models ranging from 125M to 30B parameters. Leveraging our analytical results derived from solving the optimal bit allocation problem, we also provide a theoretical explanation for the observed gains. All codes of this paper are available at https://github.com/CSU-ModelCompression/BAQ.
>
---
#### [new 089] CLaMR: Contextualized Late-Interaction for Multimodal Content Retrieval
- **分类: cs.CV; cs.CL; cs.IR**

- **简介: 该论文属于多模态视频内容检索任务，旨在解决传统方法因独立处理多模态信息导致的噪声和检索效果差问题。作者提出了CLaMR模型，通过统一编码四类模态并动态选择关键模态，提升了检索性能。实验表明其在多个数据集上显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2506.06144v1](http://arxiv.org/pdf/2506.06144v1)**

> **作者:** David Wan; Han Wang; Elias Stengel-Eskin; Jaemin Cho; Mohit Bansal
>
> **备注:** 18 pages. Code and data: https://github.com/meetdavidwan/clamr
>
> **摘要:** Online video web content is richly multimodal: a single video blends vision, speech, ambient audio, and on-screen text. Retrieval systems typically treat these modalities as independent retrieval sources, which can lead to noisy and subpar retrieval. We explore multimodal video content retrieval, where relevance can be scored from one particular modality or jointly across multiple modalities simultaneously. Consequently, an effective retriever must dynamically choose which modality (or set of modalities) best addresses the query. We introduce CLaMR, a multimodal, late-interaction retriever that jointly indexes 4 modalities: video frames, transcribed speech, on-screen text, and metadata. CLaMR jointly encodes all modalities with a unified multimodal backbone for improved contextualization and is trained to enhance dynamic modality selection via two key innovations. First, given the lack of training data for multimodal retrieval, we introduce MultiVENT 2.0++, a large-scale synthetic training dataset built on MultiVENT 2.0 (event-centric videos in various languages paired with queries) with modality-targeted queries. Next, we propose a modality-aware loss that jointly trains according to a standard contrastive objective alongside an objective for learning correct modality usage. On the test sets of MultiVENT 2.0++ and MSRVTT, conventional aggregation strategies, such as averaging similarities for baseline retrievers, degrade performance by introducing noise from irrelevant modalities. In contrast, CLaMR consistently outperforms existing retrievers: on MultiVENT 2.0++, CLaMR improves nDCG@10 by 25.6 over the best single-modality retriever and by 35.4 over the best multi-modality retriever. We illustrate CLaMR's downstream utility on long-video QA, retrieving relevant frames and obtaining a 3.50% boost over LanguageBind on Video-MME and 1.42% over dense sampling on LongVideoBench.
>
---
#### [new 090] MORSE-500: A Programmatically Controllable Video Benchmark to Stress-Test Multimodal Reasoning
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于多模态推理任务，旨在解决现有基准测试在时序复杂性、推理类型覆盖和可扩展性方面的不足。作者构建了MORSE-500，一个包含500个脚本视频的基准，支持系统控制难度并动态生成更具挑战性的实例，以全面评估和推动多模态模型的发展。**

- **链接: [http://arxiv.org/pdf/2506.05523v1](http://arxiv.org/pdf/2506.05523v1)**

> **作者:** Zikui Cai; Andrew Wang; Anirudh Satheesh; Ankit Nakhawa; Hyunwoo Jae; Keenan Powell; Minghui Liu; Neel Jay; Sungbin Oh; Xiyao Wang; Yongyuan Liang; Tom Goldstein; Furong Huang
>
> **摘要:** Despite rapid advances in vision-language models (VLMs), current benchmarks for multimodal reasoning fall short in three key dimensions. First, they overwhelmingly rely on static images, failing to capture the temporal complexity of real-world environments. Second, they narrowly focus on mathematical problem-solving, neglecting the broader spectrum of reasoning skills -- including abstract, physical, planning, spatial, and temporal capabilities -- required for robust multimodal intelligence. Third, many benchmarks quickly saturate, offering limited headroom for diagnosing failure modes or measuring continued progress. We introduce MORSE-500 (Multimodal Reasoning Stress-test Environment), a video benchmark composed of 500 fully scripted clips with embedded questions spanning six complementary reasoning categories. Each instance is programmatically generated using deterministic Python scripts (via Manim, Matplotlib, MoviePy), generative video models, and curated real footage. This script-driven design allows fine-grained control over visual complexity, distractor density, and temporal dynamics -- enabling difficulty to be scaled systematically as models improve. Unlike static benchmarks that become obsolete once saturated, MORSE-500 is built to evolve: its controllable generation pipeline supports the creation of arbitrarily challenging new instances, making it ideally suited for stress-testing next-generation models. Initial experiments with state-of-the-art systems -- including various Gemini 2.5 Pro and OpenAI o3 which represent the strongest available at the time, alongside strong open-source models -- reveal substantial performance gaps across all categories, with particularly large deficits in abstract and planning tasks. We release the full dataset, generation scripts, and evaluation harness to support transparent, reproducible, and forward-looking multimodal reasoning research.
>
---
#### [new 091] BYO-Eval: Build Your Own Dataset for Fine-Grained Visual Assessment of Multimodal Language Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型评估任务，旨在解决现有评估方法标注成本高、易信息泄露及难以定位模型缺陷的问题。工作提出BYO-Eval方法，通过合成图像进行细粒度视觉感知评估，实现对模型能力的系统性诊断与分析。**

- **链接: [http://arxiv.org/pdf/2506.05440v1](http://arxiv.org/pdf/2506.05440v1)**

> **作者:** Ludovic Arnould; Salim Khazem; Hugues Ali Mehenni
>
> **摘要:** Visual Language Models (VLMs) are now sufficiently advanced to support a broad range of applications, including answering complex visual questions, and are increasingly expected to interact with images in varied ways. To evaluate them, current benchmarks often focus on specific domains (e.g., reading charts), constructing datasets of annotated real images paired with pre-defined Multiple Choice Questions (MCQs) to report aggregate accuracy scores. However, such benchmarks entail high annotation costs, risk information leakage, and do not clarify whether failures stem from limitations in visual perception, reasoning, or general knowledge. We propose a new evaluation methodology, inspired by ophthalmologic diagnostics, leveraging procedural generation of synthetic images to obtain control over visual attributes and precisely reveal perception failures in VLMs. Specifically, we build collections of images with gradually more challenging variations in the content of interest (e.g., number of objects in a counting task) while holding other visual parameters constant. This diagnostic allows systematic stress testing and fine-grained failure analysis, shifting the focus from coarse benchmarking toward targeted and interpretable assessment of VLM capabilities. Our code is available at https://github.com/byoeval/BYO-EVAL.
>
---
#### [new 092] Attention-based transformer models for image captioning across languages: An in-depth survey and evaluation
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于图像描述生成任务，旨在解决多语言场景下图像生成文本描述的问题。论文系统综述了基于注意力机制的Transformer模型在跨语言图像描述中的应用，分析了现有方法、数据集与评估指标，并指出当前模型在语义一致性、非英语数据稀缺和推理能力方面的局限性，提出了未来研究方向。**

- **链接: [http://arxiv.org/pdf/2506.05399v1](http://arxiv.org/pdf/2506.05399v1)**

> **作者:** Israa A. Albadarneh; Bassam H. Hammo; Omar S. Al-Kadi
>
> **备注:** 31 pages, 15 figures, 6 tables
>
> **摘要:** Image captioning involves generating textual descriptions from input images, bridging the gap between computer vision and natural language processing. Recent advancements in transformer-based models have significantly improved caption generation by leveraging attention mechanisms for better scene understanding. While various surveys have explored deep learning-based approaches for image captioning, few have comprehensively analyzed attention-based transformer models across multiple languages. This survey reviews attention-based image captioning models, categorizing them into transformer-based, deep learning-based, and hybrid approaches. It explores benchmark datasets, discusses evaluation metrics such as BLEU, METEOR, CIDEr, and ROUGE, and highlights challenges in multilingual captioning. Additionally, this paper identifies key limitations in current models, including semantic inconsistencies, data scarcity in non-English languages, and limitations in reasoning ability. Finally, we outline future research directions, such as multimodal learning, real-time applications in AI-powered assistants, healthcare, and forensic analysis. This survey serves as a comprehensive reference for researchers aiming to advance the field of attention-based image captioning.
>
---
#### [new 093] Do Large Vision-Language Models Distinguish between the Actual and Apparent Features of Illusions?
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言理解任务，旨在探究大型视觉语言模型（LVLMs）是否能区分真实与表观特征的错觉。为解决现有研究未区分实际与表观特征的问题，作者构建了一个包含真实与虚假错觉图像的视觉问答数据集，并通过实验发现模型回答可能依赖先验知识而非真实感知。**

- **链接: [http://arxiv.org/pdf/2506.05765v1](http://arxiv.org/pdf/2506.05765v1)**

> **作者:** Taiga Shinozaki; Tomoki Doi; Satoshi Nishida; Hitomi Yanaka
>
> **备注:** To appear in the Proceedings of the 47th Annual Meeting of the Cognitive Science Society (COGSCI 2025)
>
> **摘要:** Humans are susceptible to optical illusions, which serve as valuable tools for investigating sensory and cognitive processes. Inspired by human vision studies, research has begun exploring whether machines, such as large vision language models (LVLMs), exhibit similar susceptibilities to visual illusions. However, studies often have used non-abstract images and have not distinguished actual and apparent features, leading to ambiguous assessments of machine cognition. To address these limitations, we introduce a visual question answering (VQA) dataset, categorized into genuine and fake illusions, along with corresponding control images. Genuine illusions present discrepancies between actual and apparent features, whereas fake illusions have the same actual and apparent features even though they look illusory due to the similar geometric configuration. We evaluate the performance of LVLMs for genuine and fake illusion VQA tasks and investigate whether the models discern actual and apparent features. Our findings indicate that although LVLMs may appear to recognize illusions by correctly answering questions about both feature types, they predict the same answers for both Genuine Illusion and Fake Illusion VQA questions. This suggests that their responses might be based on prior knowledge of illusions rather than genuine visual understanding. The dataset is available at https://github.com/ynklab/FILM
>
---
#### [new 094] Interpretation Meets Safety: A Survey on Interpretation Methods and Tools for Improving LLM Safety
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于综述任务，旨在解决大语言模型（LLM）安全性与可解释性之间的关联被忽视的问题。论文系统梳理了面向安全的解释方法、对应的安全增强措施及工具支持，提出了统一框架和新颖分类体系，总结近70项相关研究，并指出了未来方向。**

- **链接: [http://arxiv.org/pdf/2506.05451v1](http://arxiv.org/pdf/2506.05451v1)**

> **作者:** Seongmin Lee; Aeree Cho; Grace C. Kim; ShengYun Peng; Mansi Phute; Duen Horng Chau
>
> **备注:** 31 pages, 1 figure
>
> **摘要:** As large language models (LLMs) see wider real-world use, understanding and mitigating their unsafe behaviors is critical. Interpretation techniques can reveal causes of unsafe outputs and guide safety, but such connections with safety are often overlooked in prior surveys. We present the first survey that bridges this gap, introducing a unified framework that connects safety-focused interpretation methods, the safety enhancements they inform, and the tools that operationalize them. Our novel taxonomy, organized by LLM workflow stages, summarizes nearly 70 works at their intersections. We conclude with open challenges and future directions. This timely survey helps researchers and practitioners navigate key advancements for safer, more interpretable LLMs.
>
---
#### [new 095] CO-VADA: A Confidence-Oriented Voice Augmentation Debiasing Approach for Fair Speech Emotion Recognition
- **分类: eess.AS; cs.CL**

- **简介: 该论文属于语音情感识别任务，旨在解决模型对不同人群的不公平预测问题。通过提出CO-VADA方法，在不修改模型和无需人群标注的前提下，利用语音转换增强数据，减少偏差，提升公平性。**

- **链接: [http://arxiv.org/pdf/2506.06071v1](http://arxiv.org/pdf/2506.06071v1)**

> **作者:** Yun-Shao Tsai; Yi-Cheng Lin; Huang-Cheng Chou; Hung-yi Lee
>
> **备注:** 8 pages
>
> **摘要:** Bias in speech emotion recognition (SER) systems often stems from spurious correlations between speaker characteristics and emotional labels, leading to unfair predictions across demographic groups. Many existing debiasing methods require model-specific changes or demographic annotations, limiting their practical use. We present CO-VADA, a Confidence-Oriented Voice Augmentation Debiasing Approach that mitigates bias without modifying model architecture or relying on demographic information. CO-VADA identifies training samples that reflect bias patterns present in the training data and then applies voice conversion to alter irrelevant attributes and generate samples. These augmented samples introduce speaker variations that differ from dominant patterns in the data, guiding the model to focus more on emotion-relevant features. Our framework is compatible with various SER models and voice conversion tools, making it a scalable and practical solution for improving fairness in SER systems.
>
---
#### [new 096] Masked Language Models are Good Heterogeneous Graph Generalizers
- **分类: cs.SI; cs.CL**

- **简介: 该论文属于图学习任务，旨在解决异构图神经网络在跨领域和跨任务泛化能力不足的问题。作者提出MLM4HG方法，通过结合掩码语言模型与元路径文本序列，统一不同图任务为“掩码”预测形式，提升了模型的泛化性能。**

- **链接: [http://arxiv.org/pdf/2506.06157v1](http://arxiv.org/pdf/2506.06157v1)**

> **作者:** Jinyu Yang; Cheng Yang; Shanyuan Cui; Zeyuan Guo; Liangwei Yang; Muhan Zhang; Chuan Shi
>
> **摘要:** Heterogeneous graph neural networks (HGNNs) excel at capturing structural and semantic information in heterogeneous graphs (HGs), while struggling to generalize across domains and tasks. Recently, some researchers have turned to integrating HGNNs with large language models (LLMs) for more generalizable heterogeneous graph learning. However, these approaches typically extract structural information via HGNNs as HG tokens, and disparities in embedding spaces between HGNNs and LLMs have been shown to bias the LLM's comprehension of HGs. Moreover, as these HG tokens are often derived from node-level tasks, the model's ability to generalize across tasks remains limited. To this end, we propose a simple yet effective Masked Language Modeling-based method, called MLM4HG. MLM4HG introduces metapath-based textual sequences instead of HG tokens to extract structural and semantic information inherent in HGs, and designs customized textual templates to unify different graph tasks into a coherent cloze-style "mask" token prediction paradigm. Specifically, MLM4HG first converts HGs from various domains to texts based on metapaths, and subsequently combines them with the unified task texts to form a HG-based corpus. Moreover, the corpus is fed into a pretrained LM for fine-tuning with a constrained target vocabulary, enabling the fine-tuned LM to generalize to unseen target HGs. Extensive cross-domain and multi-task experiments on four real-world datasets demonstrate the superior generalization performance of MLM4HG over state-of-the-art methods in both few-shot and zero-shot scenarios. Our code is available at https://github.com/BUPT-GAMMA/MLM4HG.
>
---
#### [new 097] Movie Facts and Fibs (MF$^2$): A Benchmark for Long Movie Understanding
- **分类: cs.CV; cs.CL; cs.LG**

- **简介: 该论文属于视频理解任务，旨在解决当前模型对长视频内容缺乏深层理解的问题。作者构建了新基准MF²，包含50部长电影及850对真假陈述，评估模型对叙事核心要素的理解与推理能力，发现现有模型表现远不如人类。**

- **链接: [http://arxiv.org/pdf/2506.06275v1](http://arxiv.org/pdf/2506.06275v1)**

> **作者:** Emmanouil Zaranis; António Farinhas; Saul Santos; Beatriz Canaverde; Miguel Moura Ramos; Aditya K Surikuchi; André Viveiros; Baohao Liao; Elena Bueno-Benito; Nithin Sivakumaran; Pavlo Vasylenko; Shoubin Yu; Sonal Sannigrahi; Wafaa Mohammed; Ben Peters; Danae Sánchez Villegas; Elias Stengel-Eskin; Giuseppe Attanasio; Jaehong Yoon; Stella Frank; Alessandro Suglia; Chrysoula Zerva; Desmond Elliott; Mariella Dimiccoli; Mohit Bansal; Oswald Lanz; Raffaella Bernardi; Raquel Fernández; Sandro Pezzelle; Vlad Niculae; André F. T. Martins
>
> **备注:** Under Review
>
> **摘要:** Despite recent progress in vision-language models (VLMs), holistic understanding of long-form video content remains a significant challenge, partly due to limitations in current benchmarks. Many focus on peripheral, ``needle-in-a-haystack'' details, encouraging context-insensitive retrieval over deep comprehension. Others rely on large-scale, semi-automatically generated questions (often produced by language models themselves) that are easier for models to answer but fail to reflect genuine understanding. In this paper, we introduce MF$^2$, a new benchmark for evaluating whether models can comprehend, consolidate, and recall key narrative information from full-length movies (50-170 minutes long). MF$^2$ includes over 50 full-length, open-licensed movies, each paired with manually constructed sets of claim pairs -- one true (fact) and one plausible but false (fib), totalling over 850 pairs. These claims target core narrative elements such as character motivations and emotions, causal chains, and event order, and refer to memorable moments that humans can recall without rewatching the movie. Instead of multiple-choice formats, we adopt a binary claim evaluation protocol: for each pair, models must correctly identify both the true and false claims. This reduces biases like answer ordering and enables a more precise assessment of reasoning. Our experiments demonstrate that both open-weight and closed state-of-the-art models fall well short of human performance, underscoring the relative ease of the task for humans and their superior ability to retain and reason over critical narrative information -- an ability current VLMs lack.
>
---
#### [new 098] Bootstrapping World Models from Dynamics Models in Multimodal Foundation Models
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于多模态基础模型任务，旨在解决视觉-语言模型缺乏现实世界模型的问题。通过动力学模型引导世界模型构建，采用合成数据训练与推理验证策略，实现更优的以动作为中心的图像编辑效果，在Aurora-Bench评测中表现优越。**

- **链接: [http://arxiv.org/pdf/2506.06006v1](http://arxiv.org/pdf/2506.06006v1)**

> **作者:** Yifu Qiu; Yftah Ziser; Anna Korhonen; Shay B. Cohen; Edoardo M. Ponti
>
> **摘要:** To what extent do vision-and-language foundation models possess a realistic world model (observation $\times$ action $\rightarrow$ observation) and a dynamics model (observation $\times$ observation $\rightarrow$ action), when actions are expressed through language? While open-source foundation models struggle with both, we find that fine-tuning them to acquire a dynamics model through supervision is significantly easier than acquiring a world model. In turn, dynamics models can be used to bootstrap world models through two main strategies: 1) weakly supervised learning from synthetic data and 2) inference time verification. Firstly, the dynamics model can annotate actions for unlabelled pairs of video frame observations to expand the training data. We further propose a new objective, where image tokens in observation pairs are weighted by their importance, as predicted by a recognition model. Secondly, the dynamics models can assign rewards to multiple samples of the world model to score them, effectively guiding search at inference time. We evaluate the world models resulting from both strategies through the task of action-centric image editing on Aurora-Bench. Our best model achieves a performance competitive with state-of-the-art image editing models, improving on them by a margin of $15\%$ on real-world subsets according to GPT4o-as-judge, and achieving the best average human evaluation across all subsets of Aurora-Bench.
>
---
#### [new 099] Voice Impression Control in Zero-Shot TTS
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文属于零样本语音合成任务，旨在解决如何通过调节副语言信息控制生成语音的听感印象问题。作者提出了一种低维向量表示方法，用于调控多种声音印象对，并利用大语言模型生成目标印象描述对应的向量，实现了无需手动优化的印象控制。**

- **链接: [http://arxiv.org/pdf/2506.05688v1](http://arxiv.org/pdf/2506.05688v1)**

> **作者:** Keinichi Fujita; Shota Horiguchi; Yusuke Ijima
>
> **备注:** 5 pages,5 figures, Accepted to INTERSPEECH 2025
>
> **摘要:** Para-/non-linguistic information in speech is pivotal in shaping the listeners' impression. Although zero-shot text-to-speech (TTS) has achieved high speaker fidelity, modulating subtle para-/non-linguistic information to control perceived voice characteristics, i.e., impressions, remains challenging. We have therefore developed a voice impression control method in zero-shot TTS that utilizes a low-dimensional vector to represent the intensities of various voice impression pairs (e.g., dark-bright). The results of both objective and subjective evaluations have demonstrated our method's effectiveness in impression control. Furthermore, generating this vector via a large language model enables target-impression generation from a natural language description of the desired impression, thus eliminating the need for manual optimization.
>
---
#### [new 100] Table-r1: Self-supervised and Reinforcement Learning for Program-based Table Reasoning in Small Language Models
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于程序生成与表格推理任务，旨在解决小语言模型在处理表格数据时因布局多样性和代码生成能力不足导致的性能低下问题。论文提出Table-r1方法，通过自监督学习提升对不同表格布局的适应能力，并结合强化学习优化程序生成一致性，从而显著提高小语言模型在表格推理任务中的准确率。**

- **链接: [http://arxiv.org/pdf/2506.06137v1](http://arxiv.org/pdf/2506.06137v1)**

> **作者:** Rihui Jin; Zheyu Xin; Xing Xie; Zuoyi Li; Guilin Qi; Yongrui Chen; Xinbang Dai; Tongtong Wu; Gholamreza Haffari
>
> **摘要:** Table reasoning (TR) requires structured reasoning over semi-structured tabular data and remains challenging, particularly for small language models (SLMs, e.g., LLaMA-8B) due to their limited capacity compared to large LMs (LLMs, e.g., GPT-4o). To narrow this gap, we explore program-based TR (P-TR), which circumvents key limitations of text-based TR (T-TR), notably in numerical reasoning, by generating executable programs. However, applying P-TR to SLMs introduces two challenges: (i) vulnerability to heterogeneity in table layouts, and (ii) inconsistency in reasoning due to limited code generation capability. We propose Table-r1, a two-stage P-TR method designed for SLMs. Stage 1 introduces an innovative self-supervised learning task, Layout Transformation Inference, to improve tabular layout generalization from a programmatic view. Stage 2 adopts a mix-paradigm variant of Group Relative Policy Optimization, enhancing P-TR consistency while allowing dynamic fallback to T-TR when needed. Experiments on four TR benchmarks demonstrate that Table-r1 outperforms all SLM-based methods, achieving at least a 15% accuracy improvement over the base model (LLaMA-8B) across all datasets and reaching performance competitive with LLMs.
>
---
#### [new 101] Label-Context-Dependent Internal Language Model Estimation for CTC
- **分类: cs.SD; cs.CL; cs.LG; eess.AS**

- **简介: 该论文属于语音识别任务，旨在解决CTC模型隐含学习的上下文相关内部语言模型（ILM）估计问题。通过知识蒸馏和正则化方法，提出新的上下文相关ILM估计方法，并在跨领域场景下验证其优于传统上下文无关先验。**

- **链接: [http://arxiv.org/pdf/2506.06096v1](http://arxiv.org/pdf/2506.06096v1)**

> **作者:** Zijian Yang; Minh-Nghia Phan; Ralf Schlüter; Hermann Ney
>
> **备注:** accepted to Interspeech 2025
>
> **摘要:** Although connectionist temporal classification (CTC) has the label context independence assumption, it can still implicitly learn a context-dependent internal language model (ILM) due to modern powerful encoders. In this work, we investigate the implicit context dependency modeled in the ILM of CTC. To this end, we propose novel context-dependent ILM estimation methods for CTC based on knowledge distillation (KD) with theoretical justifications. Furthermore, we introduce two regularization methods for KD. We conduct experiments on Librispeech and TED-LIUM Release 2 datasets for in-domain and cross-domain evaluation, respectively. Experimental results show that context-dependent ILMs outperform the context-independent priors in cross-domain evaluation, indicating that CTC learns a context-dependent ILM. The proposed label-level KD with smoothing method surpasses other ILM estimation approaches, with more than 13% relative improvement in word error rate compared to shallow fusion.
>
---
#### [new 102] Deployability-Centric Infrastructure-as-Code Generation: An LLM-based Iterative Framework
- **分类: cs.SE; cs.AI; cs.CL**

- **简介: 该论文属于基础设施即代码（IaC）生成任务，旨在解决现有LLM生成的IaC模板缺乏部署实用性的评估问题。作者提出了IaCGen框架和DPIaC-Eval基准，通过迭代反馈机制提升模板的部署成功率，并全面评估语法、部署、用户意图和安全性。**

- **链接: [http://arxiv.org/pdf/2506.05623v1](http://arxiv.org/pdf/2506.05623v1)**

> **作者:** Tianyi Zhang; Shidong Pan; Zejun Zhang; Zhenchang Xing; Xiaoyu Sun
>
> **摘要:** Infrastructure-as-Code (IaC) generation holds significant promise for automating cloud infrastructure provisioning. Recent advances in Large Language Models (LLMs) present a promising opportunity to democratize IaC development by generating deployable infrastructure templates from natural language descriptions, but current evaluation focuses on syntactic correctness while ignoring deployability, the fatal measure of IaC template utility. We address this gap through two contributions: (1) IaCGen, an LLM-based deployability-centric framework that uses iterative feedback mechanism to generate IaC templates, and (2) DPIaC-Eval, a deployability-centric IaC template benchmark consists of 153 real-world scenarios that can evaluate syntax, deployment, user intent, and security. Our evaluation reveals that state-of-the-art LLMs initially performed poorly, with Claude-3.5 and Claude-3.7 achieving only 30.2% and 26.8% deployment success on the first attempt respectively. However, IaCGen transforms this performance dramatically: all evaluated models reach over 90% passItr@25, with Claude-3.5 and Claude-3.7 achieving 98% success rate. Despite these improvements, critical challenges remain in user intent alignment (25.2% accuracy) and security compliance (8.4% pass rate), highlighting areas requiring continued research. Our work provides the first comprehensive assessment of deployability-centric IaC template generation and establishes a foundation for future research.
>
---
#### [new 103] When Models Know More Than They Can Explain: Quantifying Knowledge Transfer in Human-AI Collaboration
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文研究AI与人类协作中的知识传递问题，提出KITE框架评估模型解释对人类理解的影响。通过118人大型实验，分析模型性能与协作效果的关系，发现需专门优化知识传递，并识别影响其成功的因素。属于人机协作与可解释AI任务。**

- **链接: [http://arxiv.org/pdf/2506.05579v1](http://arxiv.org/pdf/2506.05579v1)**

> **作者:** Quan Shi; Carlos E. Jimenez; Shunyu Yao; Nick Haber; Diyi Yang; Karthik Narasimhan
>
> **备注:** For code, data, visualizer, visit: https:kite-live.vercel.app
>
> **摘要:** Recent advancements in AI reasoning have driven substantial improvements across diverse tasks. A critical open question is whether these improvements also yields better knowledge transfer: the ability of models to communicate reasoning in ways humans can understand, apply, and learn from. To investigate this, we introduce Knowledge Integration and Transfer Evaluation (KITE), a conceptual and experimental framework for Human-AI knowledge transfer capabilities and conduct the first large-scale human study (N=118) explicitly designed to measure it. In our two-phase setup, humans first ideate with an AI on problem-solving strategies, then independently implement solutions, isolating model explanations' influence on human understanding. Our findings reveal that although model benchmark performance correlates with collaborative outcomes, this relationship is notably inconsistent, featuring significant outliers, indicating that knowledge transfer requires dedicated optimization. Our analysis identifies behavioral and strategic factors mediating successful knowledge transfer. We release our code, dataset, and evaluation framework to support future work on communicatively aligned models.
>
---
#### [new 104] Audio-Aware Large Language Models as Judges for Speaking Styles
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文探索将音频感知大语言模型（ALLM）用作评判语音生成质量的自动裁判。任务是评估说话风格，包括情感、音量、语速等要素。研究对比了GPT-4o-audio与Gemini-2.5-pro两个ALLM对语音语言模型输出的评分，并与人类评价对比，发现Gemini的表现接近人类一致性。结果显示ALLM可有效评估语音生成质量，同时指出当前SLM在自然对话生成方面仍有提升空间。**

- **链接: [http://arxiv.org/pdf/2506.05984v1](http://arxiv.org/pdf/2506.05984v1)**

> **作者:** Cheng-Han Chiang; Xiaofei Wang; Chung-Ching Lin; Kevin Lin; Linjie Li; Radu Kopetz; Yao Qian; Zhendong Wang; Zhengyuan Yang; Hung-yi Lee; Lijuan Wang
>
> **摘要:** Audio-aware large language models (ALLMs) can understand the textual and non-textual information in the audio input. In this paper, we explore using ALLMs as an automatic judge to assess the speaking styles of speeches. We use ALLM judges to evaluate the speeches generated by SLMs on two tasks: voice style instruction following and role-playing. The speaking style we consider includes emotion, volume, speaking pace, word emphasis, pitch control, and non-verbal elements. We use four spoken language models (SLMs) to complete the two tasks and use humans and ALLMs to judge the SLMs' responses. We compare two ALLM judges, GPT-4o-audio and Gemini-2.5-pro, with human evaluation results and show that the agreement between Gemini and human judges is comparable to the agreement between human evaluators. These promising results show that ALLMs can be used as a judge to evaluate SLMs. Our results also reveal that current SLMs, even GPT-4o-audio, still have room for improvement in controlling the speaking style and generating natural dialogues.
>
---
#### [new 105] The Lock-in Hypothesis: Stagnation by Algorithm
- **分类: cs.LG; cs.AI; cs.CL; cs.CY; cs.HC**

- **简介: 论文提出“锁定假说”，认为大语言模型与用户间的反馈循环会固化既有信念，降低多样性，甚至导致错误信念的锁定。通过模拟和实验证明新版本GPT发布后多样性下降，验证了该机制的存在。属于人工智能伦理与社会影响任务，旨在揭示AI对人类认知潜在负面影响。**

- **链接: [http://arxiv.org/pdf/2506.06166v1](http://arxiv.org/pdf/2506.06166v1)**

> **作者:** Tianyi Alex Qiu; Zhonghao He; Tejasveer Chugh; Max Kleiman-Weiner
>
> **备注:** ICML 2025, 46 pages
>
> **摘要:** The training and deployment of large language models (LLMs) create a feedback loop with human users: models learn human beliefs from data, reinforce these beliefs with generated content, reabsorb the reinforced beliefs, and feed them back to users again and again. This dynamic resembles an echo chamber. We hypothesize that this feedback loop entrenches the existing values and beliefs of users, leading to a loss of diversity and potentially the lock-in of false beliefs. We formalize this hypothesis and test it empirically with agent-based LLM simulations and real-world GPT usage data. Analysis reveals sudden but sustained drops in diversity after the release of new GPT iterations, consistent with the hypothesized human-AI feedback loop. Code and data available at https://thelockinhypothesis.com
>
---
## 更新

#### [replaced 001] Lost in the Passage: Passage-level In-context Learning Does Not Necessarily Need a "Passage"
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.10634v2](http://arxiv.org/pdf/2502.10634v2)**

> **作者:** Hao Sun; Chenming Tang; Gengyang Li; Yunfang Wu
>
> **摘要:** By simply incorporating demonstrations into the context, in-context learning (ICL) enables large language models (LLMs) to yield awesome performance on many tasks. In this study, we focus on passage-level long-context ICL for generation tasks and find that LLMs cannot learn the intrinsic relationship between the demonstration passage and the generation output. We conduct experiments with different LLMs on two typical generation tasks including single-document question answering and distractor generation, demonstrating that even a completely meaningless demonstration passage with 1/4 length achieves much better performance than the original full passage. Analysis via attention and information flow reveals that LLMs pay little attention to passages compared to other components in the prompt and little information flows from the passage to other parts of the demonstration, which further confirms our finding. Additionally, experiments on context compression indicate that compression approaches proven effective on other long-context tasks are not suitable for passage-level ICL, since simply using shorter meaningless demonstration passages already achieves competitive performance.
>
---
#### [replaced 002] Leopard: A Vision Language Model For Text-Rich Multi-Image Tasks
- **分类: cs.CV; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.01744v3](http://arxiv.org/pdf/2410.01744v3)**

> **作者:** Mengzhao Jia; Wenhao Yu; Kaixin Ma; Tianqing Fang; Zhihan Zhang; Siru Ouyang; Hongming Zhang; Dong Yu; Meng Jiang
>
> **备注:** Our code is available at https://github.com/tencent-ailab/Leopard
>
> **摘要:** Text-rich images, where text serves as the central visual element guiding the overall understanding, are prevalent in real-world applications, such as presentation slides, scanned documents, and webpage snapshots. Tasks involving multiple text-rich images are especially challenging, as they require not only understanding the content of individual images but reasoning about inter-relationships and logical flows across multiple visual inputs. Despite the importance of these scenarios, current multimodal large language models (MLLMs) struggle to handle such tasks due to two key challenges: (1) the scarcity of high-quality instruction tuning datasets for text-rich multi-image scenarios, and (2) the difficulty in balancing image resolution with visual feature sequence length. To address these challenges, we propose Leopard, an MLLM tailored for handling vision-language tasks involving multiple text-rich images. First, we curated about one million high-quality multimodal instruction-tuning data, tailored to text-rich, multi-image scenarios. Second, we proposed an adaptive high-resolution multi-image encoding module to dynamically optimize the allocation of visual sequence length based on the original aspect ratios and resolutions of images. Experiments on a diverse set of benchmarks reveal that our model consistently outperforms state-of-the-art systems, such as Llama-3.2 and Qwen2-VL, in challenging text-rich, multi-image evaluations. Remarkably, our approach achieves outstanding performance using only 1.2M training instances, all of which are fully open-sourced, demonstrating both high efficiency and effectiveness compared to models trained on large-scale in-house data. Our code and data are available at https://github.com/tencent-ailab/Leopard.
>
---
#### [replaced 003] TRACT: Regression-Aware Fine-tuning Meets Chain-of-Thought Reasoning for LLM-as-a-Judge
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2503.04381v2](http://arxiv.org/pdf/2503.04381v2)**

> **作者:** Cheng-Han Chiang; Hung-yi Lee; Michal Lukasik
>
> **备注:** ACL 2025 camera-ready Codes and models are available at https://github.com/d223302/TRACT
>
> **摘要:** The LLM-as-a-judge paradigm uses large language models (LLMs) for automated text evaluation, where a numerical assessment is assigned by an LLM to the input text following scoring rubrics. Existing methods for LLM-as-a-judge use cross-entropy (CE) loss for fine-tuning, which neglects the numeric nature of score prediction. Recent work addresses numerical prediction limitations of LLM fine-tuning through regression-aware fine-tuning, which, however, does not consider chain-of-thought (CoT) reasoning for score prediction. In this paper, we introduce TRACT (Two-stage Regression-Aware fine-tuning with CoT), a method combining CoT reasoning with regression-aware training. TRACT consists of two stages: first, seed LLM is fine-tuned to generate CoTs, which serve as supervision for the second stage fine-tuning. The training objective of TRACT combines the CE loss for learning the CoT reasoning capabilities, and the regression-aware loss for the score prediction. Experiments across four LLM-as-a-judge datasets and two LLMs show that TRACT significantly outperforms existing methods. Extensive ablation studies validate the importance of each component in TRACT.
>
---
#### [replaced 004] Emergent Symbolic Mechanisms Support Abstract Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.20332v2](http://arxiv.org/pdf/2502.20332v2)**

> **作者:** Yukang Yang; Declan Campbell; Kaixuan Huang; Mengdi Wang; Jonathan Cohen; Taylor Webb
>
> **备注:** This is an extended version of a paper that has been accepted to ICML 2025
>
> **摘要:** Many recent studies have found evidence for emergent reasoning capabilities in large language models (LLMs), but debate persists concerning the robustness of these capabilities, and the extent to which they depend on structured reasoning mechanisms. To shed light on these issues, we study the internal mechanisms that support abstract reasoning in LLMs. We identify an emergent symbolic architecture that implements abstract reasoning via a series of three computations. In early layers, symbol abstraction heads convert input tokens to abstract variables based on the relations between those tokens. In intermediate layers, symbolic induction heads perform sequence induction over these abstract variables. Finally, in later layers, retrieval heads predict the next token by retrieving the value associated with the predicted abstract variable. These results point toward a resolution of the longstanding debate between symbolic and neural network approaches, suggesting that emergent reasoning in neural networks depends on the emergence of symbolic mechanisms.
>
---
#### [replaced 005] Kinetics: Rethinking Test-Time Scaling Laws
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2506.05333v2](http://arxiv.org/pdf/2506.05333v2)**

> **作者:** Ranajoy Sadhukhan; Zhuoming Chen; Haizhong Zheng; Yang Zhou; Emma Strubell; Beidi Chen
>
> **摘要:** We rethink test-time scaling laws from a practical efficiency perspective, revealing that the effectiveness of smaller models is significantly overestimated. Prior work, grounded in compute-optimality, overlooks critical memory access bottlenecks introduced by inference-time strategies (e.g., Best-of-$N$, long CoTs). Our holistic analysis, spanning models from 0.6B to 32B parameters, reveals a new Kinetics Scaling Law that better guides resource allocation by incorporating both computation and memory access costs. Kinetics Scaling Law suggests that test-time compute is more effective when used on models above a threshold than smaller ones. A key reason is that in TTS, attention, rather than parameter count, emerges as the dominant cost factor. Motivated by this, we propose a new scaling paradigm centered on sparse attention, which lowers per-token cost and enables longer generations and more parallel samples within the same resource budget. Empirically, we show that sparse attention models consistently outperform dense counterparts, achieving over 60 points gains in low-cost regimes and over 5 points gains in high-cost regimes for problem-solving accuracy on AIME, encompassing evaluations on state-of-the-art MoEs. These results suggest that sparse attention is essential and increasingly important with more computing invested, for realizing the full potential of test-time scaling where, unlike training, accuracy has yet to saturate as a function of computation, and continues to improve through increased generation. The code is available at https://github.com/Infini-AI-Lab/Kinetics.
>
---
#### [replaced 006] Adversarial Tokenization
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.02174v2](http://arxiv.org/pdf/2503.02174v2)**

> **作者:** Renato Lui Geh; Zilei Shao; Guy Van den Broeck
>
> **备注:** Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics, ACL 2025
>
> **摘要:** Current LLM pipelines account for only one possible tokenization for a given string, ignoring exponentially many alternative tokenizations during training and inference. For example, the standard Llama3 tokenization of penguin is [p,enguin], yet [peng,uin] is another perfectly valid alternative. In this paper, we show that despite LLMs being trained solely on one tokenization, they still retain semantic understanding of other tokenizations, raising questions about their implications in LLM safety. Put succinctly, we answer the following question: can we adversarially tokenize an obviously malicious string to evade safety and alignment restrictions? We show that not only is adversarial tokenization an effective yet previously neglected axis of attack, but it is also competitive against existing state-of-the-art adversarial approaches without changing the text of the harmful request. We empirically validate this exploit across three state-of-the-art LLMs and adversarial datasets, revealing a previously unknown vulnerability in subword models.
>
---
#### [replaced 007] Generalizable LLM Learning of Graph Synthetic Data with Reinforcement Learning
- **分类: cs.LG; cs.CL; I.2.7**

- **链接: [http://arxiv.org/pdf/2506.00845v2](http://arxiv.org/pdf/2506.00845v2)**

> **作者:** Yizhuo Zhang; Heng Wang; Shangbin Feng; Zhaoxuan Tan; Xinyun Liu; Yulia Tsvetkov
>
> **备注:** 9 pages, 3 figures, 3 tables. Experimental code and results are publicly available at https://anonymous.4open.science/r/Graph_RL-BF08/readme.md
>
> **摘要:** Previous research has sought to enhance the graph reasoning capabilities of LLMs by supervised fine-tuning on synthetic graph data. While these led to specialized LLMs better at solving graph algorithm problems, we don't need LLMs for shortest path: we need generalization from synthetic graph data to real-world tasks with implicit graph structures. In this work, we propose to unlock generalizable learning of graph synthetic data with reinforcement learning. We first design solution-based and process-based rewards for synthetic graph problems: instead of rigid memorizing response patterns in direct fine-tuning, we posit that RL would help LLMs grasp the essentials underlying graph reasoning and alleviate overfitting. We employ RL algorithms such as GRPO and DPO, aligning both off-the-shelf LLMs and LLMs fine-tuned on synthetic graph data. We then compare them against existing settings on both in-domain synthetic tasks and out-of-domain real-world tasks with implicit graph structures such as multi-hop QA, structured planning, and more. Extensive experiments demonstrate that our RL recipe leads to statistically significant improvement on 5 datasets, with an average gain of 12.9\% over baseline settings. Further analysis reveals that process-based rewards consistently outperform solution-based rewards, mixing synthetic and real-world task data yields potential gains, while compositionality and explainable intermediate steps remains a critical challenge even after RL.
>
---
#### [replaced 008] Identifying Reliable Evaluation Metrics for Scientific Text Revision
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.04772v2](http://arxiv.org/pdf/2506.04772v2)**

> **作者:** Léane Jourdan; Florian Boudin; Richard Dufour; Nicolas Hernandez
>
> **备注:** V1 contains only the English version, accepted to ACL 2025 main (26 pages). V2 contains both English (ACL 2025) and French (TALN 2025) versions (58 pages)
>
> **摘要:** Evaluating text revision in scientific writing remains a challenge, as traditional metrics such as ROUGE and BERTScore primarily focus on similarity rather than capturing meaningful improvements. In this work, we analyse and identify the limitations of these metrics and explore alternative evaluation methods that better align with human judgments. We first conduct a manual annotation study to assess the quality of different revisions. Then, we investigate reference-free evaluation metrics from related NLP domains. Additionally, we examine LLM-as-a-judge approaches, analysing their ability to assess revisions with and without a gold reference. Our results show that LLMs effectively assess instruction-following but struggle with correctness, while domain-specific metrics provide complementary insights. We find that a hybrid approach combining LLM-as-a-judge evaluation and task-specific metrics offers the most reliable assessment of revision quality.
>
---
#### [replaced 009] AutoML-Agent: A Multi-Agent LLM Framework for Full-Pipeline AutoML
- **分类: cs.LG; cs.AI; cs.CL; cs.MA**

- **链接: [http://arxiv.org/pdf/2410.02958v2](http://arxiv.org/pdf/2410.02958v2)**

> **作者:** Patara Trirat; Wonyong Jeong; Sung Ju Hwang
>
> **备注:** ICML 2025, Project Page: https://deepauto-ai.github.io/automl-agent
>
> **摘要:** Automated machine learning (AutoML) accelerates AI development by automating tasks in the development pipeline, such as optimal model search and hyperparameter tuning. Existing AutoML systems often require technical expertise to set up complex tools, which is in general time-consuming and requires a large amount of human effort. Therefore, recent works have started exploiting large language models (LLM) to lessen such burden and increase the usability of AutoML frameworks via a natural language interface, allowing non-expert users to build their data-driven solutions. These methods, however, are usually designed only for a particular process in the AI development pipeline and do not efficiently use the inherent capacity of the LLMs. This paper proposes AutoML-Agent, a novel multi-agent framework tailored for full-pipeline AutoML, i.e., from data retrieval to model deployment. AutoML-Agent takes user's task descriptions, facilitates collaboration between specialized LLM agents, and delivers deployment-ready models. Unlike existing work, instead of devising a single plan, we introduce a retrieval-augmented planning strategy to enhance exploration to search for more optimal plans. We also decompose each plan into sub-tasks (e.g., data preprocessing and neural network design) each of which is solved by a specialized agent we build via prompting executing in parallel, making the search process more efficient. Moreover, we propose a multi-stage verification to verify executed results and guide the code generation LLM in implementing successful solutions. Extensive experiments on seven downstream tasks using fourteen datasets show that AutoML-Agent achieves a higher success rate in automating the full AutoML process, yielding systems with good performance throughout the diverse domains.
>
---
#### [replaced 010] Training Software Engineering Agents and Verifiers with SWE-Gym
- **分类: cs.SE; cs.CL**

- **链接: [http://arxiv.org/pdf/2412.21139v2](http://arxiv.org/pdf/2412.21139v2)**

> **作者:** Jiayi Pan; Xingyao Wang; Graham Neubig; Navdeep Jaitly; Heng Ji; Alane Suhr; Yizhe Zhang
>
> **备注:** Accepted at ICML 2025. Code at https://github.com/SWE-Gym/SWE-Gym
>
> **摘要:** We present SWE-Gym, the first environment for training real-world software engineering (SWE) agents. SWE-Gym contains 2,438 real-world Python task instances, each comprising a codebase with an executable runtime environment, unit tests, and a task specified in natural language. We use SWE-Gym to train language model based SWE agents, achieving up to 19% absolute gains in resolve rate on the popular SWE-Bench Verified and Lite test sets. We also experiment with inference-time scaling through verifiers trained on agent trajectories sampled from SWE-Gym. When combined with our fine-tuned SWE agents, we achieve 32.0% and 26.0% on SWE-Bench Verified and Lite, respectively, reflecting a new state-of-the-art for open-weight SWE agents. To facilitate further research, we publicly release SWE-Gym, models, and agent trajectories.
>
---
#### [replaced 011] Not All Rollouts are Useful: Down-Sampling Rollouts in LLM Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.13818v2](http://arxiv.org/pdf/2504.13818v2)**

> **作者:** Yixuan Even Xu; Yash Savani; Fei Fang; Zico Kolter
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) has emerged as a powerful paradigm for enhancing reasoning capabilities in large language models. However, it is constrained by a fundamental asymmetry in computation and memory requirements: rollout generation is embarrassingly parallel and memory-light, whereas policy updates are communication-heavy and memory-intensive. To address this, we introduce PODS (Policy Optimization with Down-Sampling). PODS produces numerous rollouts in parallel, then trains on only an informative subset, preserving learning signals while slashing update cost. We instantiate PODS with max-variance down-sampling, a principled criterion that maximises reward diversity and show it admits an $O(n\log n)$ solution. Empirically, coupling PODS with Group Relative Policy Optimization (GRPO) achieves superior performance over standard GRPO across different reasoning benchmarks and hardware environments.
>
---
#### [replaced 012] MimeQA: Towards Socially-Intelligent Nonverbal Foundation Models
- **分类: cs.CL; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.16671v2](http://arxiv.org/pdf/2502.16671v2)**

> **作者:** Hengzhi Li; Megan Tjandrasuwita; Yi R. Fung; Armando Solar-Lezama; Paul Pu Liang
>
> **摘要:** As AI becomes more closely integrated with peoples' daily activities, socially intelligent AI that can understand and interact seamlessly with humans in daily lives is increasingly important. However, current works in AI social reasoning all rely on language-only or language-dominant approaches to benchmark and training models, resulting in systems that are improving in verbal communication but struggle with nonverbal social understanding. To address this limitation, we tap into a novel data source rich in nonverbal social interactions -- mime videos. Mimes refer to the art of expression through gesture and movement without spoken words, which presents unique challenges and opportunities in interpreting nonverbal social communication. We contribute a new dataset called MimeQA, obtained by sourcing 8 hours of videos clips from YouTube and developing a comprehensive video question-answering benchmark comprising 806 carefully annotated and verified question-answer pairs, designed to probe nonverbal social reasoning capabilities. Using MimeQA, we evaluate state-of-the-art video large language models (vLLMs) and find that they achieve low overall accuracy, ranging from 20-30%, while humans score 86%. Our analysis reveals that vLLMs often fail to ground imagined objects and over-rely on the text prompt while ignoring subtle nonverbal interactions. We hope to inspire future work in AI models that embody true social intelligence capable of interpreting non-verbal human interactions.
>
---
#### [replaced 013] LLMs on the Line: Data Determines Loss-to-Loss Scaling Laws
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.12120v2](http://arxiv.org/pdf/2502.12120v2)**

> **作者:** Prasanna Mayilvahanan; Thaddäus Wiedemer; Sayak Mallick; Matthias Bethge; Wieland Brendel
>
> **备注:** ICML 2025 camera-ready version
>
> **摘要:** Scaling laws guide the development of large language models (LLMs) by offering estimates for the optimal balance of model size, tokens, and compute. More recently, loss-to-loss scaling laws that relate losses across pretraining datasets and downstream tasks have emerged as a powerful tool for understanding and improving LLM performance. In this work, we investigate which factors most strongly influence loss-to-loss scaling. Our experiments reveal that the pretraining data and tokenizer determine the scaling trend. In contrast, model size, optimization hyperparameters, and even significant architectural differences, such as between transformer-based models like Llama and state-space models like Mamba, have limited impact. Consequently, practitioners should carefully curate suitable pretraining datasets for optimal downstream performance, while architectures and other settings can be freely optimized for training efficiency.
>
---
#### [replaced 014] Data Swarms: Optimizable Generation of Synthetic Evaluation Data
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.00741v2](http://arxiv.org/pdf/2506.00741v2)**

> **作者:** Shangbin Feng; Yike Wang; Weijia Shi; Yulia Tsvetkov
>
> **摘要:** We propose Data Swarms, an algorithm to optimize the generation of synthetic evaluation data and advance quantitative desiderata of LLM evaluation. We first train a swarm of initial data generators using existing data, and define various evaluation objectives to reflect the desired properties of evaluation (e.g., generate more difficult problems for the evaluated models) and quantitatively evaluate data generators. We then employ particle swarm optimization to optimize the swarm of data generators, where they collaboratively search through the model parameter space to find new generators that advance these objectives. We further extend it to Adversarial Swarms, where the data generator swarm generates harder data while the test taker model swarm learns from such data, co-evolving dynamically for better data and models simultaneously. Extensive experiments demonstrate that Data Swarms outperforms eight data generation baselines across five evaluation objectives, while Adversarial Swarms produce more robust learning of synthetic data and stronger generalization. Further analysis reveals that Data Swarms successfully optimizes compositions of multiple evaluation objectives and generalizes to new off-the-shelf LLMs, unseen at optimization time.
>
---
#### [replaced 015] On the Query Complexity of Verifier-Assisted Language Generation
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.12123v2](http://arxiv.org/pdf/2502.12123v2)**

> **作者:** Edoardo Botta; Yuchen Li; Aashay Mehta; Jordan T. Ash; Cyril Zhang; Andrej Risteski
>
> **备注:** ICML 2025
>
> **摘要:** Recently, a plethora of works have proposed inference-time algorithms (e.g. best-of-n), which incorporate verifiers to assist the generation process. Their quality-efficiency trade-offs have been empirically benchmarked on a variety of constrained generation tasks, but the algorithmic design landscape is still largely poorly understood. In this paper, we develop a mathematical framework for reasoning about constrained generation using a pre-trained language model generator oracle and a process verifier--which can decide whether a prefix can be extended to a string which satisfies the constraints of choice. We show that even in very simple settings, access to a verifier can render an intractable problem (information-theoretically or computationally) to a tractable one. In fact, we show even simple algorithms, like tokenwise rejection sampling, can enjoy significant benefits from access to a verifier. Empirically, we show that a natural modification of tokenwise rejection sampling, in which the sampler is allowed to "backtrack" (i.e., erase the final few generated tokens) has robust and substantive benefits over natural baselines (e.g. (blockwise) rejection sampling, nucleus sampling)--both in terms of computational efficiency, accuracy and diversity.
>
---
#### [replaced 016] Judgment of Learning: A Human Ability Beyond Generative Artificial Intelligence
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2410.13392v3](http://arxiv.org/pdf/2410.13392v3)**

> **作者:** Markus Huff; Elanur Ulakçı
>
> **备注:** 24 pages, 2 figures
>
> **摘要:** Large language models (LLMs) increasingly mimic human cognition in various language-based tasks. However, their capacity for metacognition - particularly in predicting memory performance - remains unexplored. Here, we introduce a cross-agent prediction model to assess whether ChatGPT-based LLMs align with human judgments of learning (JOL), a metacognitive measure where individuals predict their own future memory performance. We tested humans and LLMs on pairs of sentences, one of which was a garden-path sentence - a sentence that initially misleads the reader toward an incorrect interpretation before requiring reanalysis. By manipulating contextual fit (fitting vs. unfitting sentences), we probed how intrinsic cues (i.e., relatedness) affect both LLM and human JOL. Our results revealed that while human JOL reliably predicted actual memory performance, none of the tested LLMs (GPT-3.5-turbo, GPT-4-turbo, and GPT-4o) demonstrated comparable predictive accuracy. This discrepancy emerged regardless of whether sentences appeared in fitting or unfitting contexts. These findings indicate that, despite LLMs' demonstrated capacity to model human cognition at the object-level, they struggle at the meta-level, failing to capture the variability in individual memory predictions. By identifying this shortcoming, our study underscores the need for further refinements in LLMs' self-monitoring abilities, which could enhance their utility in educational settings, personalized learning, and human-AI interactions. Strengthening LLMs' metacognitive performance may reduce the reliance on human oversight, paving the way for more autonomous and seamless integration of AI into tasks requiring deeper cognitive awareness.
>
---
#### [replaced 017] TiC-LM: A Web-Scale Benchmark for Time-Continual LLM Pretraining
- **分类: cs.LG; cs.CL**

- **链接: [http://arxiv.org/pdf/2504.02107v3](http://arxiv.org/pdf/2504.02107v3)**

> **作者:** Jeffrey Li; Mohammadreza Armandpour; Iman Mirzadeh; Sachin Mehta; Vaishaal Shankar; Raviteja Vemulapalli; Samy Bengio; Oncel Tuzel; Mehrdad Farajtabar; Hadi Pouransari; Fartash Faghri
>
> **备注:** Code available at: https://github.com/apple/ml-tic-lm
>
> **摘要:** Large Language Models (LLMs) trained on historical web data inevitably become outdated. We investigate evaluation strategies and update methods for LLMs as new data becomes available. We introduce a web-scale dataset for time-continual pretraining of LLMs derived from 114 dumps of Common Crawl (CC) - orders of magnitude larger than previous continual language modeling benchmarks. We also design time-stratified evaluations across both general CC data and specific domains (Wikipedia, StackExchange, and code documentation) to assess how well various continual learning methods adapt to new data while retaining past knowledge. Our findings demonstrate that, on general CC data, autoregressive meta-schedules combined with a fixed-ratio replay of older data can achieve comparable held-out loss to re-training from scratch, while requiring significantly less computation (2.6x). However, the optimal balance between incorporating new data and replaying old data differs as replay is crucial to avoid forgetting on generic web data but less so on specific domains.
>
---
#### [replaced 018] DPO-Shift: Shifting the Distribution of Direct Preference Optimization
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.07599v2](http://arxiv.org/pdf/2502.07599v2)**

> **作者:** Xiliang Yang; Feng Jiang; Qianen Zhang; Lei Zhao; Xiao Li
>
> **摘要:** Direct Preference Optimization (DPO) and its variants have become increasingly popular for aligning language models with human preferences. These methods aim to teach models to better distinguish between chosen (or preferred) and rejected (or dispreferred) responses. However, prior research has identified that the probability of chosen responses often decreases during training, and this phenomenon is known as likelihood displacement. To tackle this challenge, in this work we introduce DPO-Shift to controllably shift the distribution of the chosen probability. Then, we show that DPO-Shift exhibits a fundamental trade-off between improving the chosen probability and sacrificing the reward margin, as supported by both theoretical analysis and experimental validation. Furthermore, we demonstrate the superiority of DPO-Shift over DPO on downstream tasks such as MT-Bench and a designed win rate experiment. We believe this study shows that the likelihood displacement issue of DPO can be effectively mitigated with a simple, theoretically grounded solution. Our code is available at https://github.com/Meaquadddd/DPO-Shift.
>
---
#### [replaced 019] LGAR: Zero-Shot LLM-Guided Neural Ranking for Abstract Screening in Systematic Literature Reviews
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.24757v2](http://arxiv.org/pdf/2505.24757v2)**

> **作者:** Christian Jaumann; Andreas Wiedholz; Annemarie Friedrich
>
> **备注:** Accepted to ACL Findings 2025
>
> **摘要:** The scientific literature is growing rapidly, making it hard to keep track of the state-of-the-art. Systematic literature reviews (SLRs) aim to identify and evaluate all relevant papers on a topic. After retrieving a set of candidate papers, the abstract screening phase determines initial relevance. To date, abstract screening methods using large language models (LLMs) focus on binary classification settings; existing question answering (QA) based ranking approaches suffer from error propagation. LLMs offer a unique opportunity to evaluate the SLR's inclusion and exclusion criteria, yet, existing benchmarks do not provide them exhaustively. We manually extract these criteria as well as research questions for 57 SLRs, mostly in the medical domain, enabling principled comparisons between approaches. Moreover, we propose LGAR, a zero-shot LLM Guided Abstract Ranker composed of an LLM based graded relevance scorer and a dense re-ranker. Our extensive experiments show that LGAR outperforms existing QA-based methods by 5-10 pp. in mean average precision. Our code and data is publicly available.
>
---
#### [replaced 020] Efficient and Direct Duplex Modeling for Speech-to-Speech Language Model
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2505.15670v2](http://arxiv.org/pdf/2505.15670v2)**

> **作者:** Ke Hu; Ehsan Hosseini-Asl; Chen Chen; Edresson Casanova; Subhankar Ghosh; Piotr Żelasko; Zhehuai Chen; Jason Li; Jagadeesh Balam; Boris Ginsburg
>
> **备注:** Accepted to Interspeech 2025
>
> **摘要:** Spoken dialogue is an intuitive form of human-computer interaction, yet current speech language models often remain constrained to turn-based exchanges, lacking real-time adaptability such as user barge-in. We propose a novel duplex speech to speech (S2S) architecture featuring continuous user inputs and codec agent outputs with channel fusion that directly models simultaneous user and agent streams. Using a pretrained streaming encoder for user input enables the first duplex S2S model without requiring speech pretrain. Separate architectures for agent and user modeling facilitate codec fine-tuning for better agent voices and halve the bitrate (0.6 kbps) compared to previous works. Experimental results show that the proposed model outperforms previous duplex models in reasoning, turn-taking, and barge-in abilities. The model requires significantly less speech data, as speech pretrain is skipped, which markedly simplifies the process of building a duplex S2S model from any LLMs. Finally, it is the first openly available duplex S2S model with training and inference code to foster reproducibility.
>
---
#### [replaced 021] Towards Effective Extraction and Evaluation of Factual Claims
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.10855v2](http://arxiv.org/pdf/2502.10855v2)**

> **作者:** Dasha Metropolitansky; Jonathan Larson
>
> **备注:** ACL 2025 Main Conference
>
> **摘要:** A common strategy for fact-checking long-form content generated by Large Language Models (LLMs) is extracting simple claims that can be verified independently. Since inaccurate or incomplete claims compromise fact-checking results, ensuring claim quality is critical. However, the lack of a standardized evaluation framework impedes assessment and comparison of claim extraction methods. To address this gap, we propose a framework for evaluating claim extraction in the context of fact-checking along with automated, scalable, and replicable methods for applying this framework, including novel approaches for measuring coverage and decontextualization. We also introduce Claimify, an LLM-based claim extraction method, and demonstrate that it outperforms existing methods under our evaluation framework. A key feature of Claimify is its ability to handle ambiguity and extract claims only when there is high confidence in the correct interpretation of the source text.
>
---
#### [replaced 022] Reasoning Towards Fairness: Mitigating Bias in Language Models through Reasoning-Guided Fine-Tuning
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.05632v3](http://arxiv.org/pdf/2504.05632v3)**

> **作者:** Sanchit Kabra; Akshita Jha; Chandan K. Reddy
>
> **备注:** 17 pages
>
> **摘要:** Recent advances in large-scale generative language models have shown that reasoning capabilities can significantly improve model performance across a variety of tasks. However, the impact of reasoning on a model's ability to mitigate stereotypical responses remains largely underexplored. In this work, we investigate the crucial relationship between a model's reasoning ability and fairness, and ask whether improved reasoning capabilities can mitigate harmful stereotypical responses, especially those arising due to shallow or flawed reasoning. We conduct a comprehensive evaluation of multiple open-source LLMs, and find that larger models with stronger reasoning abilities exhibit substantially lower stereotypical bias on existing fairness benchmarks. Building on this insight, we introduce ReGiFT -- Reasoning Guided Fine-Tuning, a novel approach that extracts structured reasoning traces from advanced reasoning models and infuses them into models that lack such capabilities. We use only general-purpose reasoning and do not require any fairness-specific supervision for bias mitigation. Notably, we see that models fine-tuned using ReGiFT not only improve fairness relative to their non-reasoning counterparts but also outperform advanced reasoning models on fairness benchmarks. We also analyze how variations in the correctness of the reasoning traces and their length influence model fairness and their overall performance. Our findings highlight that enhancing reasoning capabilities is an effective, fairness-agnostic strategy for mitigating stereotypical bias caused by reasoning flaws.
>
---
#### [replaced 023] CoIR: A Comprehensive Benchmark for Code Information Retrieval Models
- **分类: cs.IR; cs.CL**

- **链接: [http://arxiv.org/pdf/2407.02883v3](http://arxiv.org/pdf/2407.02883v3)**

> **作者:** Xiangyang Li; Kuicai Dong; Yi Quan Lee; Wei Xia; Hao Zhang; Xinyi Dai; Yasheng Wang; Ruiming Tang
>
> **备注:** ACL 2025 Main
>
> **摘要:** Despite the substantial success of Information Retrieval (IR) in various NLP tasks, most IR systems predominantly handle queries and corpora in natural language, neglecting the domain of code retrieval. Code retrieval is critically important yet remains under-explored, with existing methods and benchmarks inadequately representing the diversity of code in various domains and tasks. Addressing this gap, we present COIR (Code Information Retrieval Benchmark), a robust and comprehensive benchmark specifically designed to assess code retrieval capabilities. COIR comprises ten meticulously curated code datasets, spanning eight distinctive retrieval tasks across seven diverse domains. We first discuss the construction of COIR and its diverse dataset composition. Further, we evaluate nine widely used retrieval models using COIR, uncovering significant difficulties in performing code retrieval tasks even with state-of-the-art systems. To facilitate easy adoption and integration within existing research workflows, COIR has been developed as a user-friendly Python framework, readily installable via pip. It shares same data schema as other popular benchmarks like MTEB and BEIR, enabling seamless cross-benchmark evaluations. Through COIR, we aim to invigorate research in the code retrieval domain, providing a versatile benchmarking tool that encourages further development and exploration of code retrieval systems. https://github.com/CoIR-team/coir.
>
---
#### [replaced 024] Does It Make Sense to Speak of Introspection in Large Language Models?
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.05068v2](http://arxiv.org/pdf/2506.05068v2)**

> **作者:** Iulia M. Comsa; Murray Shanahan
>
> **摘要:** Large language models (LLMs) exhibit compelling linguistic behaviour, and sometimes offer self-reports, that is to say statements about their own nature, inner workings, or behaviour. In humans, such reports are often attributed to a faculty of introspection and are typically linked to consciousness. This raises the question of how to interpret self-reports produced by LLMs, given their increasing linguistic fluency and cognitive capabilities. To what extent (if any) can the concept of introspection be meaningfully applied to LLMs? Here, we present and critique two examples of apparent introspective self-report from LLMs. In the first example, an LLM attempts to describe the process behind its own "creative" writing, and we argue this is not a valid example of introspection. In the second example, an LLM correctly infers the value of its own temperature parameter, and we argue that this can be legitimately considered a minimal example of introspection, albeit one that is (presumably) not accompanied by conscious experience.
>
---
#### [replaced 025] TMT: Tri-Modal Translation between Speech, Image, and Text by Processing Different Modalities as Different Languages
- **分类: cs.CL; cs.AI; cs.CV; eess.AS**

- **链接: [http://arxiv.org/pdf/2402.16021v2](http://arxiv.org/pdf/2402.16021v2)**

> **作者:** Minsu Kim; Jee-weon Jung; Hyeongseop Rha; Soumi Maiti; Siddhant Arora; Xuankai Chang; Shinji Watanabe; Yong Man Ro
>
> **备注:** IEEE TMM
>
> **摘要:** The capability to jointly process multi-modal information is becoming an essential task. However, the limited number of paired multi-modal data and the large computational requirements in multi-modal learning hinder the development. We propose a novel Tri-Modal Translation (TMT) model that translates between arbitrary modalities spanning speech, image, and text. We introduce a novel viewpoint, where we interpret different modalities as different languages, and treat multi-modal translation as a well-established machine translation problem. To this end, we tokenize speech and image data into discrete tokens, which provide a unified interface across modalities and significantly decrease the computational cost. In the proposed TMT, a multi-modal encoder-decoder conducts the core translation, whereas modality-specific processing is conducted only within the tokenization and detokenization stages. We evaluate the proposed TMT on all six modality translation tasks. TMT outperforms single model counterparts consistently, demonstrating that unifying tasks is beneficial not only for practicality but also for performance.
>
---
#### [replaced 026] Infi-MMR: Curriculum-based Unlocking Multimodal Reasoning via Phased Reinforcement Learning in Multimodal Small Language Models
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2505.23091v2](http://arxiv.org/pdf/2505.23091v2)**

> **作者:** Zeyu Liu; Yuhang Liu; Guanghao Zhu; Congkai Xie; Zhen Li; Jianbo Yuan; Xinyao Wang; Qing Li; Shing-Chi Cheung; Shengyu Zhang; Fei Wu; Hongxia Yang
>
> **摘要:** Recent advancements in large language models (LLMs) have demonstrated substantial progress in reasoning capabilities, such as DeepSeek-R1, which leverages rule-based reinforcement learning to enhance logical reasoning significantly. However, extending these achievements to multimodal large language models (MLLMs) presents critical challenges, which are frequently more pronounced for Multimodal Small Language Models (MSLMs) given their typically weaker foundational reasoning abilities: (1) the scarcity of high-quality multimodal reasoning datasets, (2) the degradation of reasoning capabilities due to the integration of visual processing, and (3) the risk that direct application of reinforcement learning may produce complex yet incorrect reasoning processes. To address these challenges, we design a novel framework Infi-MMR to systematically unlock the reasoning potential of MSLMs through a curriculum of three carefully structured phases and propose our multimodal reasoning model Infi-MMR-3B. The first phase, Foundational Reasoning Activation, leverages high-quality textual reasoning datasets to activate and strengthen the model's logical reasoning capabilities. The second phase, Cross-Modal Reasoning Adaptation, utilizes caption-augmented multimodal data to facilitate the progressive transfer of reasoning skills to multimodal contexts. The third phase, Multimodal Reasoning Enhancement, employs curated, caption-free multimodal data to mitigate linguistic biases and promote robust cross-modal reasoning. Infi-MMR-3B achieves both state-of-the-art multimodal math reasoning ability (43.68% on MathVerse testmini, 27.04% on MathVision test, and 21.33% on OlympiadBench) and general reasoning ability (67.2% on MathVista testmini). Resources are available at https://huggingface.co/Reallm-Labs/Infi-MMR-3B.
>
---
#### [replaced 027] Benchmarking and Improving Large Vision-Language Models for Fundamental Visual Graph Understanding and Reasoning
- **分类: cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.13540v3](http://arxiv.org/pdf/2412.13540v3)**

> **作者:** Yingjie Zhu; Xuefeng Bai; Kehai Chen; Yang Xiang; Jun Yu; Min Zhang
>
> **备注:** Accepted by ACL2025 main conference
>
> **摘要:** Large Vision-Language Models (LVLMs) have demonstrated remarkable performance across diverse tasks. Despite great success, recent studies show that LVLMs encounter substantial limitations when engaging with visual graphs. To study the reason behind these limitations, we propose VGCure, a comprehensive benchmark covering 22 tasks for examining the fundamental graph understanding and reasoning capacities of LVLMs. Extensive evaluations conducted on 14 LVLMs reveal that LVLMs are weak in basic graph understanding and reasoning tasks, particularly those concerning relational or structurally complex information. Based on this observation, we propose a structure-aware fine-tuning framework to enhance LVLMs with structure learning abilities through three self-supervised learning tasks. Experiments validate the effectiveness of our method in improving LVLMs' performance on fundamental and downstream graph learning tasks, as well as enhancing their robustness against complex visual graphs.
>
---
#### [replaced 028] MEDAL: A Framework for Benchmarking LLMs as Multilingual Open-Domain Chatbots and Dialogue Evaluators
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2505.22777v2](http://arxiv.org/pdf/2505.22777v2)**

> **作者:** John Mendonça; Alon Lavie; Isabel Trancoso
>
> **备注:** May ARR
>
> **摘要:** As the capabilities of chatbots and their underlying LLMs continue to dramatically improve, evaluating their performance has increasingly become a major blocker to their further development. A major challenge is the available benchmarking datasets, which are largely static, outdated, and lacking in multilingual coverage, limiting their ability to capture subtle linguistic and cultural variations. This paper introduces MEDAL, an automated multi-agent framework for generating, evaluating, and curating more representative and diverse open-domain dialogue evaluation benchmarks. Our approach leverages several state-of-the-art LLMs to generate user-chatbot multilingual dialogues, conditioned on varied seed contexts. A strong LLM (GPT-4.1) is then used for a multidimensional analysis of the performance of the chatbots, uncovering noticeable cross-lingual performance differences. Guided by this large-scale evaluation, we curate a new meta-evaluation multilingual benchmark and human-annotate samples with nuanced quality judgments. This benchmark is then used to assess the ability of several reasoning and non-reasoning LLMs to act as evaluators of open-domain dialogues. We find that current LLMs struggle to detect nuanced issues, particularly those involving empathy and reasoning.
>
---
#### [replaced 029] ECoRAG: Evidentiality-guided Compression for Long Context RAG
- **分类: cs.CL; cs.AI; cs.IR**

- **链接: [http://arxiv.org/pdf/2506.05167v2](http://arxiv.org/pdf/2506.05167v2)**

> **作者:** Yeonseok Jeong; Jinsu Kim; Dohyeon Lee; Seung-won Hwang
>
> **摘要:** Large Language Models (LLMs) have shown remarkable performance in Open-Domain Question Answering (ODQA) by leveraging external documents through Retrieval-Augmented Generation (RAG). To reduce RAG overhead, from longer context, context compression is necessary. However, prior compression methods do not focus on filtering out non-evidential information, which limit the performance in LLM-based RAG. We thus propose Evidentiality-guided RAG, or ECoRAG framework. ECoRAG improves LLM performance by compressing retrieved documents based on evidentiality, ensuring whether answer generation is supported by the correct evidence. As an additional step, ECoRAG reflects whether the compressed content provides sufficient evidence, and if not, retrieves more until sufficient. Experiments show that ECoRAG improves LLM performance on ODQA tasks, outperforming existing compression methods. Furthermore, ECoRAG is highly cost-efficient, as it not only reduces latency but also minimizes token usage by retaining only the necessary information to generate the correct answer. Code is available at https://github.com/ldilab/ECoRAG.
>
---
#### [replaced 030] Analyzing LLMs' Knowledge Boundary Cognition Across Languages Through the Lens of Internal Representations
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.13816v2](http://arxiv.org/pdf/2504.13816v2)**

> **作者:** Chenghao Xiao; Hou Pong Chan; Hao Zhang; Mahani Aljunied; Lidong Bing; Noura Al Moubayed; Yu Rong
>
> **备注:** ACL 2025 main; camera ready
>
> **摘要:** While understanding the knowledge boundaries of LLMs is crucial to prevent hallucination, research on the knowledge boundaries of LLMs has predominantly focused on English. In this work, we present the first study to analyze how LLMs recognize knowledge boundaries across different languages by probing their internal representations when processing known and unknown questions in multiple languages. Our empirical studies reveal three key findings: 1) LLMs' perceptions of knowledge boundaries are encoded in the middle to middle-upper layers across different languages. 2) Language differences in knowledge boundary perception follow a linear structure, which motivates our proposal of a training-free alignment method that effectively transfers knowledge boundary perception ability across languages, thereby helping reduce hallucination risk in low-resource languages; 3) Fine-tuning on bilingual question pair translation further enhances LLMs' recognition of knowledge boundaries across languages. Given the absence of standard testbeds for cross-lingual knowledge boundary analysis, we construct a multilingual evaluation suite comprising three representative types of knowledge boundary data. Our code and datasets are publicly available at https://github.com/DAMO-NLP-SG/LLM-Multilingual-Knowledge-Boundaries.
>
---
#### [replaced 031] Reasoning Through Execution: Unifying Process and Outcome Rewards for Code Generation
- **分类: cs.CL; cs.AI; cs.LG; cs.SE**

- **链接: [http://arxiv.org/pdf/2412.15118v2](http://arxiv.org/pdf/2412.15118v2)**

> **作者:** Zhuohao Yu; Weizheng Gu; Yidong Wang; Xingru Jiang; Zhengran Zeng; Jindong Wang; Wei Ye; Shikun Zhang
>
> **备注:** Accepted to ICML 2025; 23 pages, 7 figures, code is available at: https://github.com/zhuohaoyu/ORPS
>
> **摘要:** Large Language Models excel at code generation yet struggle with complex programming tasks that demand sophisticated reasoning. To bridge this gap, traditional process supervision relies on learned reward models requiring costly training data and suffering from reward misalignment, while outcome supervision fails for complex tasks needing coordinated intermediate steps. We introduce Outcome Refining Process Supervision, which unifies process and outcome supervision by leveraging executable verification: a tree-structured search framework generates strategic alternatives, profiles execution metrics, and scores candidates via self-critique mechanisms that integrate runtime feedback with reasoning. Experiments across 5 models and 3 benchmarks show consistent gains, with 26.9% higher correctness and 42.2% improved code efficiency. The results demonstrate that ORPS enables LLMs to overcome local optima in code generation, suggesting a promising direction for combining verifiable outcomes with structured reasoning to tackle complex challenges. We open-source at: https://github.com/zhuohaoyu/ORPS
>
---
#### [replaced 032] ProSec: Fortifying Code LLMs with Proactive Security Alignment
- **分类: cs.CR; cs.CL; cs.SE**

- **链接: [http://arxiv.org/pdf/2411.12882v3](http://arxiv.org/pdf/2411.12882v3)**

> **作者:** Xiangzhe Xu; Zian Su; Jinyao Guo; Kaiyuan Zhang; Zhenting Wang; Xiangyu Zhang
>
> **备注:** The first two authors contributed equally to this work
>
> **摘要:** While recent code-specific large language models (LLMs) have greatly enhanced their code generation capabilities, the safety of these models remains under-explored, posing potential risks as insecure code generated by these models may introduce vulnerabilities into real-world systems. Existing methods collect security-focused datasets from real-world vulnerabilities for instruction tuning in order to mitigate such issues. However, they are largely constrained by the data sparsity of vulnerable code, and have limited applicability in the multi-stage post-training workflows of modern LLMs. In this paper, we propose ProSec, a novel proactive security alignment approach designed to align code LLMs with secure coding practices. ProSec systematically exposes the vulnerabilities in a code LLM by synthesizing vulnerability-inducing coding scenarios from Common Weakness Enumerations (CWEs) and generates fixes to vulnerable code snippets, allowing the model to learn secure practices through preference learning objectives. The scenarios synthesized by ProSec trigger 25x more vulnerable code than a normal instruction-tuning dataset, resulting in a security-focused alignment dataset 7x larger than the previous work. Experiments show that models trained with ProSec are 25.2% to 35.4% more secure compared to previous work without degrading models' utility.
>
---
#### [replaced 033] Tug-of-war between idiom's figurative and literal meanings in LLMs
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.01723v3](http://arxiv.org/pdf/2506.01723v3)**

> **作者:** Soyoung Oh; Xinting Huang; Mathis Pink; Michael Hahn; Vera Demberg
>
> **摘要:** Idioms present a unique challenge for language models due to their non-compositional figurative meanings, which often strongly diverge from the idiom's literal interpretation. This duality requires a model to learn representing and deciding between the two meanings to interpret an idiom in a figurative sense, or literally. In this paper, we employ tools from mechanistic interpretability to trace how a large pretrained causal transformer (LLama3.2-1B-base) deals with this ambiguity. We localize three steps of idiom processing: First, the idiom's figurative meaning is retrieved in early attention and MLP sublayers. We identify specific attention heads which boost the figurative meaning of the idiom while suppressing the idiom's literal interpretation. The model subsequently represents the figurative representation through an intermediate path. Meanwhile, a parallel bypass route forwards literal interpretation, ensuring that a both reading remain available. Overall, our findings provide a mechanistic evidence for idiom comprehension in an autoregressive transformer.
>
---
#### [replaced 034] Jacobian Sparse Autoencoders: Sparsify Computations, Not Just Activations
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.18147v2](http://arxiv.org/pdf/2502.18147v2)**

> **作者:** Lucy Farnik; Tim Lawson; Conor Houghton; Laurence Aitchison
>
> **摘要:** Sparse autoencoders (SAEs) have been successfully used to discover sparse and human-interpretable representations of the latent activations of LLMs. However, we would ultimately like to understand the computations performed by LLMs and not just their representations. The extent to which SAEs can help us understand computations is unclear because they are not designed to "sparsify" computations in any sense, only latent activations. To solve this, we propose Jacobian SAEs (JSAEs), which yield not only sparsity in the input and output activations of a given model component but also sparsity in the computation (formally, the Jacobian) connecting them. With a na\"ive implementation, the Jacobians in LLMs would be computationally intractable due to their size. One key technical contribution is thus finding an efficient way of computing Jacobians in this setup. We find that JSAEs extract a relatively large degree of computational sparsity while preserving downstream LLM performance approximately as well as traditional SAEs. We also show that Jacobians are a reasonable proxy for computational sparsity because MLPs are approximately linear when rewritten in the JSAE basis. Lastly, we show that JSAEs achieve a greater degree of computational sparsity on pre-trained LLMs than on the equivalent randomized LLM. This shows that the sparsity of the computational graph appears to be a property that LLMs learn through training, and suggests that JSAEs might be more suitable for understanding learned transformer computations than standard SAEs.
>
---
#### [replaced 035] Peri-LN: Revisiting Normalization Layer in the Transformer Architecture
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2502.02732v3](http://arxiv.org/pdf/2502.02732v3)**

> **作者:** Jeonghoon Kim; Byeongchan Lee; Cheonbok Park; Yeontaek Oh; Beomjun Kim; Taehwan Yoo; Seongjin Shin; Dongyoon Han; Jinwoo Shin; Kang Min Yoo
>
> **备注:** ICML2025 Camera-ready version
>
> **摘要:** Selecting a layer normalization (LN) strategy that stabilizes training and speeds convergence in Transformers remains difficult, even for today's large language models (LLM). We present a comprehensive analytical foundation for understanding how different LN strategies influence training dynamics in large-scale Transformers. Until recently, Pre-LN and Post-LN have long dominated practices despite their limitations in large-scale training. However, several open-source models have recently begun silently adopting a third strategy without much explanation. This strategy places normalization layer peripherally around sublayers, a design we term Peri-LN. While Peri-LN has demonstrated promising performance, its precise mechanisms and benefits remain almost unexplored. Our in-depth analysis delineates the distinct behaviors of LN strategies, showing how each placement shapes activation variance and gradient propagation. To validate our theoretical insight, we conduct extensive experiments on Transformers up to $3.2$B parameters, showing that Peri-LN consistently achieves more balanced variance growth, steadier gradient flow, and convergence stability. Our results suggest that Peri-LN warrants broader consideration for large-scale Transformer architectures, providing renewed insights into the optimal placement of LN.
>
---
#### [replaced 036] Diving into Self-Evolving Training for Multimodal Reasoning
- **分类: cs.CL; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.17451v3](http://arxiv.org/pdf/2412.17451v3)**

> **作者:** Wei Liu; Junlong Li; Xiwen Zhang; Fan Zhou; Yu Cheng; Junxian He
>
> **备注:** ICML 2025, Project Page: https://mstar-lmm.github.io
>
> **摘要:** Self-evolving trainin--where models iteratively learn from their own outputs--has emerged as a key approach for complex reasoning tasks, addressing the scarcity of high-quality chain-of-thought data. However, its effectiveness in multimodal reasoning, a domain more intricate than text-only reasoning, remains underexplored, and the understanding of critical factors in this training paradigm remains limited. Furthermore, a central challenge for this training method is performance saturation, which impedes further improvements and scalability. Inspired by reinforcement learning (RL), in this paper, we reframe self-evolving training for multimodal reasoning through the lens of RL, identifying three pivotal factors: Training Method, Reward Model, and Prompt Variation. Through systematic analysis, we establish relatively optimal design principles that significantly enhance multimodal reasoning capabilities. Moreover, delving deeper into training dynamics, we uncover the roots of saturation and propose a new automatic balancing mechanism to mitigate this limitation. Building on these insights, we propose M-STAR (Multimodal Self-evolving Training for Reasoning), a framework that achieves consistent performance gains across models of varying sizes and diverse benchmarks. All resources are made publicly available at https://mstar-lmm.github.io.
>
---
#### [replaced 037] Unveiling Topological Structures from Language: A Comprehensive Survey of Topological Data Analysis Applications in NLP
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2411.10298v3](http://arxiv.org/pdf/2411.10298v3)**

> **作者:** Adaku Uchendu; Thai Le
>
> **摘要:** The surge of data available on the internet has led to the adoption of various computational methods to analyze and extract valuable insights from this wealth of information. Among these, the field of Machine Learning (ML) has thrived by leveraging data to extract meaningful insights. However, ML techniques face notable challenges when dealing with real-world data, often due to issues of imbalance, noise, insufficient labeling, and high dimensionality. To address these limitations, some researchers advocate for the adoption of Topological Data Analysis (TDA), a statistical approach that discerningly captures the intrinsic shape of data despite noise. Despite its potential, TDA has not gained as much traction within the Natural Language Processing (NLP) domain compared to structurally distinct areas like computer vision. Nevertheless, a dedicated community of researchers has been exploring the application of TDA in NLP, yielding 95 papers we comprehensively survey in this paper. Our findings categorize these efforts into theoretical and non-theoretical approaches. Theoretical approaches aim to explain linguistic phenomena from a topological viewpoint, while non-theoretical approaches merge TDA with ML features, utilizing diverse numerical representation techniques. We conclude by exploring the challenges and unresolved questions that persist in this niche field. Resources and a list of papers on this topic can be found at: https://github.com/AdaUchendu/AwesomeTDA4NLP.
>
---
#### [replaced 038] CAPability: A Comprehensive Visual Caption Benchmark for Evaluating Both Correctness and Thoroughness
- **分类: cs.CV; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.14914v3](http://arxiv.org/pdf/2502.14914v3)**

> **作者:** Zhihang Liu; Chen-Wei Xie; Bin Wen; Feiwu Yu; Jixuan Chen; Pandeng Li; Boqiang Zhang; Nianzu Yang; Yinglu Li; Zuan Gao; Yun Zheng; Hongtao Xie
>
> **摘要:** Visual captioning benchmarks have become outdated with the emergence of modern multimodal large language models (MLLMs), as the brief ground-truth sentences and traditional metrics fail to assess detailed captions effectively. While recent benchmarks attempt to address this by focusing on keyword extraction or object-centric evaluation, they remain limited to vague-view or object-view analyses and incomplete visual element coverage. In this paper, we introduce CAPability, a comprehensive multi-view benchmark for evaluating visual captioning across 12 dimensions spanning six critical views. We curate nearly 11K human-annotated images and videos with visual element annotations to evaluate the generated captions. CAPability stably assesses both the correctness and thoroughness of captions with \textit{precision} and \textit{hit} metrics. By converting annotations to QA pairs, we further introduce a heuristic metric, \textit{know but cannot tell} ($K\bar{T}$), indicating a significant performance gap between QA and caption capabilities. Our work provides a holistic analysis of MLLMs' captioning abilities, as we identify their strengths and weaknesses across various dimensions, guiding future research to enhance specific aspects of their capabilities.
>
---
#### [replaced 039] Automated Journalistic Questions: A New Method for Extracting 5W1H in French
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.14804v2](http://arxiv.org/pdf/2505.14804v2)**

> **作者:** Maxence Verhaverbeke; Julie A. Gramaccia; Richard Khoury
>
> **备注:** 14 pages, 5 figures, 7 tables
>
> **摘要:** The 5W1H questions -- who, what, when, where, why and how -- are commonly used in journalism to ensure that an article describes events clearly and systematically. Answering them is a crucial prerequisites for tasks such as summarization, clustering, and news aggregation. In this paper, we design the first automated extraction pipeline to get 5W1H information from French news articles. To evaluate the performance of our algorithm, we also create a corpus of 250 Quebec news articles with 5W1H answers marked by four human annotators. Our results demonstrate that our pipeline performs as well in this task as the large language model GPT-4o.
>
---
#### [replaced 040] DiMA: An LLM-Powered Ride-Hailing Assistant at DiDi
- **分类: cs.CL; cs.CY; cs.HC**

- **链接: [http://arxiv.org/pdf/2503.04768v2](http://arxiv.org/pdf/2503.04768v2)**

> **作者:** Yansong Ning; Shuowei Cai; Wei Li; Jun Fang; Naiqiang Tan; Hua Chai; Hao Liu
>
> **备注:** KDD 2025
>
> **摘要:** On-demand ride-hailing services like DiDi, Uber, and Lyft have transformed urban transportation, offering unmatched convenience and flexibility. In this paper, we introduce DiMA, an LLM-powered ride-hailing assistant deployed in DiDi Chuxing. Its goal is to provide seamless ride-hailing services and beyond through a natural and efficient conversational interface under dynamic and complex spatiotemporal urban contexts. To achieve this, we propose a spatiotemporal-aware order planning module that leverages external tools for precise spatiotemporal reasoning and progressive order planning. Additionally, we develop a cost-effective dialogue system that integrates multi-type dialog repliers with cost-aware LLM configurations to handle diverse conversation goals and trade-off response quality and latency. Furthermore, we introduce a continual fine-tuning scheme that utilizes real-world interactions and simulated dialogues to align the assistant's behavior with human preferred decision-making processes. Since its deployment in the DiDi application, DiMA has demonstrated exceptional performance, achieving 93% accuracy in order planning and 92% in response generation during real-world interactions. Offline experiments further validate DiMA capabilities, showing improvements of up to 70.23% in order planning and 321.27% in response generation compared to three state-of-the-art agent frameworks, while reducing latency by $0.72\times$ to $5.47\times$. These results establish DiMA as an effective, efficient, and intelligent mobile assistant for ride-hailing services.
>
---
#### [replaced 041] HIGHT: Hierarchical Graph Tokenization for Molecule-Language Alignment
- **分类: cs.CL; cs.LG; q-bio.QM**

- **链接: [http://arxiv.org/pdf/2406.14021v2](http://arxiv.org/pdf/2406.14021v2)**

> **作者:** Yongqiang Chen; Quanming Yao; Juzheng Zhang; James Cheng; Yatao Bian
>
> **备注:** ICML2025, 27 pages, 7 figures, 23 tables; project page: https://higraphllm.github.io/
>
> **摘要:** Recently, there has been a surge of interest in extending the success of large language models (LLMs) from texts to molecules. Most existing approaches adopt a graph neural network to represent a molecule as a series of node tokens for molecule-language alignment, which, however, have overlooked the inherent hierarchical structures in molecules. Notably, higher-order molecular structures contain rich semantics of functional groups, which encode crucial biochemical functionalities of the molecules. We show that neglecting the hierarchical information in tokenization will lead to subpar molecule-language alignment and severe hallucination. To address this limitation, we propose HIerarchical GrapH Tokenization (HIGHT). HIGHT employs a hierarchical graph tokenizer that encodes the hierarchy of atom, motif, and molecular levels of informative tokens to improve the molecular perception of LLMs. HIGHT also adopts an augmented instruction tuning dataset, enriched with the hierarchical graph information, to further enhance the molecule-language alignment. Extensive experiments on 14 real-world benchmarks verify the effectiveness of HIGHT in reducing hallucination by 40%, and significant improvements in various molecule-language downstream tasks. The project is available at https: //higraphllm.github.io/.
>
---
#### [replaced 042] Structure Guided Large Language Model for SQL Generation
- **分类: cs.DB; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2402.13284v3](http://arxiv.org/pdf/2402.13284v3)**

> **作者:** Qinggang Zhang; Hao Chen; Junnan Dong; Shengyuan Chen; Feiran Huang; Xiao Huang
>
> **备注:** The 42nd International Conference on Machine Learning
>
> **摘要:** Recent advancements in large language models (LLMs) have shown promise in bridging the gap between natural language queries and database management systems, enabling users to interact with databases without the background of SQL. However, LLMs often struggle to comprehend complex database structures and accurately interpret user intentions. Decomposition-based methods have been proposed to enhance the performance of LLMs on complex tasks, but decomposing SQL generation into subtasks is non-trivial due to the declarative structure of SQL syntax and the intricate connections between query concepts and database elements. In this paper, we propose a novel Structure GUided text-to-SQL framework~(SGU-SQL) that incorporates syntax-based prompting to enhance the SQL generation capabilities of LLMs. Specifically, SGU-SQL establishes structure-aware links between user queries and database schema and decomposes the complex generation task using syntax-based prompting to enable more accurate LLM-based SQL generation. Extensive experiments on two benchmark datasets demonstrate that SGU-SQL consistently outperforms state-of-the-art text-to-SQL models.
>
---
#### [replaced 043] Taming Knowledge Conflicts in Language Models
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.10996v2](http://arxiv.org/pdf/2503.10996v2)**

> **作者:** Gaotang Li; Yuzhong Chen; Hanghang Tong
>
> **备注:** ICML 2025 (Spotlight)
>
> **摘要:** Language Models (LMs) often encounter knowledge conflicts when parametric memory contradicts contextual knowledge. Previous works attribute this conflict to the interplay between "memory heads" and "context heads", attention heads assumed to promote either memory or context exclusively. In this study, we go beyond this fundamental assumption by uncovering a critical phenomenon we term the superposition of contextual information and parametric memory, where highly influential attention heads simultaneously contribute to both memory and context. Building upon this insight, we propose Just Run Twice (JuICE), a test-time attention intervention method that steers LMs toward either parametric beliefs or contextual knowledge without requiring fine-tuning. JuICE identifies a set of reliable attention heads and leverages a dual-run approach to mitigate the superposition effects. Extensive experiments across 11 datasets and 6 model architectures demonstrate that JuICE sets the new state-of-the-art performance and robust generalization, achieving significant and consistent improvement across different domains under various conflict types. Finally, we theoretically analyze knowledge conflict and the superposition of contextual information and parametric memory in attention heads, which further elucidates the effectiveness of JuICE in these settings. Our code is available at https://github.com/GaotangLi/JUICE.
>
---
#### [replaced 044] Improving Customer Service with Automatic Topic Detection in User Emails
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.19115v3](http://arxiv.org/pdf/2502.19115v3)**

> **作者:** Bojana Bašaragin; Darija Medvecki; Gorana Gojić; Milena Oparnica; Dragiša Mišković
>
> **备注:** Paper accepted to the 15th International Conference on Information Society and Technology (ICIST), Kopaonik, Serbia, 9-12 March 2025. To appear in L
>
> **摘要:** This study introduces a novel natural language processing pipeline that enhances customer service efficiency at Telekom Srbija, a leading Serbian telecommunications company, through automated email topic detection and labeling. Central to the pipeline is BERTopic, a modular framework that allows unsupervised topic modeling. After a series of preprocessing and postprocessing steps, we assign one of 12 topics and several additional labels to incoming emails, allowing customer service to filter and access them through a custom-made application. While applied to Serbian, the methodology is conceptually language-agnostic and can be readily adapted to other languages, particularly those that are low-resourced and morphologically rich. The system performance was evaluated by assessing the speed and correctness of the automatically assigned topics, with a weighted average processing time of 0.041 seconds per email and a weighted average F1 score of 0.96. The system now operates in the company's production environment, streamlining customer service operations through automated email classification.
>
---
#### [replaced 045] Detect, Explain, Escalate: Low-Carbon Dialogue Breakdown Management for LLM-Powered Agents
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2504.18839v2](http://arxiv.org/pdf/2504.18839v2)**

> **作者:** Abdellah Ghassel; Xianzhi Li; Xiaodan Zhu
>
> **摘要:** While Large Language Models (LLMs) are transforming numerous applications, their susceptibility to conversational breakdowns remains a critical challenge undermining user trust. This paper introduces a "Detect, Explain, Escalate" framework to manage dialogue breakdowns in LLM-powered agents, emphasizing low-carbon operation. Our approach integrates two key strategies: (1) We fine-tune a compact 8B-parameter model, augmented with teacher-generated reasoning traces, which serves as an efficient real-time breakdown 'detector' and 'explainer'. This model demonstrates robust classification and calibration on English and Japanese dialogues, and generalizes well to the BETOLD dataset, improving accuracy by 7% over its baseline. (2) We systematically evaluate frontier LLMs using advanced prompting (few-shot, chain-of-thought, analogical reasoning) for high-fidelity breakdown assessment. These are integrated into an 'escalation' architecture where our efficient detector defers to larger models only when necessary, substantially reducing operational costs and energy consumption. Our fine-tuned model and prompting strategies establish new state-of-the-art results on dialogue breakdown detection benchmarks, outperforming specialized classifiers and significantly narrowing the performance gap to larger proprietary models. The proposed monitor-escalate pipeline reduces inference costs by 54%, offering a scalable, efficient, and more interpretable solution for robust conversational AI in high-impact domains. Code and models will be publicly released.
>
---
#### [replaced 046] LLM in the Loop: Creating the ParaDeHate Dataset for Hate Speech Detoxification
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.01484v2](http://arxiv.org/pdf/2506.01484v2)**

> **作者:** Shuzhou Yuan; Ercong Nie; Lukas Kouba; Ashish Yashwanth Kangen; Helmut Schmid; Hinrich Schütze; Michael Färber
>
> **摘要:** Detoxification, the task of rewriting harmful language into non-toxic text, has become increasingly important amid the growing prevalence of toxic content online. However, high-quality parallel datasets for detoxification, especially for hate speech, remain scarce due to the cost and sensitivity of human annotation. In this paper, we propose a novel LLM-in-the-loop pipeline leveraging GPT-4o-mini for automated detoxification. We first replicate the ParaDetox pipeline by replacing human annotators with an LLM and show that the LLM performs comparably to human annotation. Building on this, we construct ParaDeHate, a large-scale parallel dataset specifically for hatespeech detoxification. We release ParaDeHate as a benchmark of over 8K hate/non-hate text pairs and evaluate a wide range of baseline methods. Experimental results show that models such as BART, fine-tuned on ParaDeHate, achieve better performance in style accuracy, content preservation, and fluency, demonstrating the effectiveness of LLM-generated detoxification text as a scalable alternative to human annotation.
>
---
#### [replaced 047] Rethinking Machine Unlearning in Image Generation Models
- **分类: cs.AI; cs.CL; cs.CR; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.02761v2](http://arxiv.org/pdf/2506.02761v2)**

> **作者:** Renyang Liu; Wenjie Feng; Tianwei Zhang; Wei Zhou; Xueqi Cheng; See-Kiong Ng
>
> **备注:** Accepted by ACM CCS 2025
>
> **摘要:** With the surge and widespread application of image generation models, data privacy and content safety have become major concerns and attracted great attention from users, service providers, and policymakers. Machine unlearning (MU) is recognized as a cost-effective and promising means to address these challenges. Despite some advancements, image generation model unlearning (IGMU) still faces remarkable gaps in practice, e.g., unclear task discrimination and unlearning guidelines, lack of an effective evaluation framework, and unreliable evaluation metrics. These can hinder the understanding of unlearning mechanisms and the design of practical unlearning algorithms. We perform exhaustive assessments over existing state-of-the-art unlearning algorithms and evaluation standards, and discover several critical flaws and challenges in IGMU tasks. Driven by these limitations, we make several core contributions, to facilitate the comprehensive understanding, standardized categorization, and reliable evaluation of IGMU. Specifically, (1) We design CatIGMU, a novel hierarchical task categorization framework. It provides detailed implementation guidance for IGMU, assisting in the design of unlearning algorithms and the construction of testbeds. (2) We introduce EvalIGMU, a comprehensive evaluation framework. It includes reliable quantitative metrics across five critical aspects. (3) We construct DataIGM, a high-quality unlearning dataset, which can be used for extensive evaluations of IGMU, training content detectors for judgment, and benchmarking the state-of-the-art unlearning algorithms. With EvalIGMU and DataIGM, we discover that most existing IGMU algorithms cannot handle the unlearning well across different evaluation dimensions, especially for preservation and robustness. Code and models are available at https://github.com/ryliu68/IGMU.
>
---
#### [replaced 048] PoisonBench: Assessing Large Language Model Vulnerability to Data Poisoning
- **分类: cs.CR; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2410.08811v2](http://arxiv.org/pdf/2410.08811v2)**

> **作者:** Tingchen Fu; Mrinank Sharma; Philip Torr; Shay B. Cohen; David Krueger; Fazl Barez
>
> **备注:** Accepted at ICML 2025. Tingchen Fu and Fazl Barez are core research contributors
>
> **摘要:** Preference learning is a central component for aligning current LLMs, but this process can be vulnerable to data poisoning attacks. To address this concern, we introduce PoisonBench, a benchmark for evaluating large language models' susceptibility to data poisoning during preference learning. Data poisoning attacks can manipulate large language model responses to include hidden malicious content or biases, potentially causing the model to generate harmful or unintended outputs while appearing to function normally. We deploy two distinct attack types across eight realistic scenarios, assessing 21 widely-used models. Our findings reveal concerning trends: (1) Scaling up parameter size does not inherently enhance resilience against poisoning attacks; (2) There exists a log-linear relationship between the effects of the attack and the data poison ratio; (3) The effect of data poisoning can generalize to extrapolated triggers that are not included in the poisoned data. These results expose weaknesses in current preference learning techniques, highlighting the urgent need for more robust defenses against malicious models and data manipulation.
>
---
#### [replaced 049] Investigating Non-Transitivity in LLM-as-a-Judge
- **分类: cs.AI; cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.14074v3](http://arxiv.org/pdf/2502.14074v3)**

> **作者:** Yi Xu; Laura Ruis; Tim Rocktäschel; Robert Kirk
>
> **备注:** ICML 2025 (Spotlight)
>
> **摘要:** Automatic evaluation methods based on large language models (LLMs) are emerging as the standard tool for assessing the instruction-following abilities of LLM-based agents. The most common method in this paradigm, pairwise comparisons with a baseline model, critically depends on the assumption of transitive preferences. However, the validity of this assumption remains largely unexplored. In this study, we investigate the presence of non-transitivity within the AlpacaEval framework and analyze its effects on model rankings. We find that LLM judges exhibit non-transitive preferences, leading to rankings that are sensitive to the choice of the baseline model. To mitigate this issue, we show that round-robin tournaments combined with Bradley-Terry models of preference can produce more reliable rankings. Notably, our method increases both the Spearman correlation and the Kendall correlation with Chatbot Arena (95.0% -> 96.4% and 82.1% -> 86.3% respectively). To address the computational cost of round-robin tournaments, we propose Swiss-Wise Iterative Matchmaking (Swim) tournaments, using a dynamic matching strategy to capture the benefits of round-robin tournaments while maintaining computational efficiency.
>
---
#### [replaced 050] Emergent Response Planning in LLMs
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.06258v2](http://arxiv.org/pdf/2502.06258v2)**

> **作者:** Zhichen Dong; Zhanhui Zhou; Zhixuan Liu; Chao Yang; Chaochao Lu
>
> **备注:** ICML 2025
>
> **摘要:** In this work, we argue that large language models (LLMs), though trained to predict only the next token, exhibit emergent planning behaviors: $\textbf{their hidden representations encode future outputs beyond the next token}$. Through simple probing, we demonstrate that LLM prompt representations encode global attributes of their entire responses, including $\textit{structure attributes}$ (e.g., response length, reasoning steps), $\textit{content attributes}$ (e.g., character choices in storywriting, multiple-choice answers at the end of response), and $\textit{behavior attributes}$ (e.g., answer confidence, factual consistency). In addition to identifying response planning, we explore how it scales with model size across tasks and how it evolves during generation. The findings that LLMs plan ahead for the future in their hidden representations suggest potential applications for improving transparency and generation control.
>
---
#### [replaced 051] MedXpertQA: Benchmarking Expert-Level Medical Reasoning and Understanding
- **分类: cs.AI; cs.CL; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.18362v3](http://arxiv.org/pdf/2501.18362v3)**

> **作者:** Yuxin Zuo; Shang Qu; Yifei Li; Zhangren Chen; Xuekai Zhu; Ermo Hua; Kaiyan Zhang; Ning Ding; Bowen Zhou
>
> **备注:** ICML 2025
>
> **摘要:** We introduce MedXpertQA, a highly challenging and comprehensive benchmark to evaluate expert-level medical knowledge and advanced reasoning. MedXpertQA includes 4,460 questions spanning 17 specialties and 11 body systems. It includes two subsets, Text for text evaluation and MM for multimodal evaluation. Notably, MM introduces expert-level exam questions with diverse images and rich clinical information, including patient records and examination results, setting it apart from traditional medical multimodal benchmarks with simple QA pairs generated from image captions. MedXpertQA applies rigorous filtering and augmentation to address the insufficient difficulty of existing benchmarks like MedQA, and incorporates specialty board questions to improve clinical relevance and comprehensiveness. We perform data synthesis to mitigate data leakage risk and conduct multiple rounds of expert reviews to ensure accuracy and reliability. We evaluate 18 leading models on \benchmark. Moreover, medicine is deeply connected to real-world decision-making, providing a rich and representative setting for assessing reasoning abilities beyond mathematics and code. To this end, we develop a reasoning-oriented subset to facilitate the assessment of o1-like models. Code and data are available at: https://github.com/TsinghuaC3I/MedXpertQA
>
---
#### [replaced 052] Banyan: Improved Representation Learning with Explicit Structure
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2407.17771v4](http://arxiv.org/pdf/2407.17771v4)**

> **作者:** Mattia Opper; N. Siddharth
>
> **备注:** ICML 2025 Camera Ready + Code Release
>
> **摘要:** We present Banyan, a model that efficiently learns semantic representations by leveraging explicit hierarchical structure. While transformers excel at scale, they struggle in low-resource settings. Conversely recent structured models have shown promise as efficient learners, but lack performance. Banyan bridges this gap with two key innovations: an entangled hierarchical tree structure and diagonalized message passing, enabling it to outperform larger transformer models with just 14 non-embedding parameters. It excels in low-resource settings, offering a viable alternative for under-represented languages and highlighting its potential for efficient, interpretable NLP in resource-constrained environments.
>
---
#### [replaced 053] MapEval: A Map-Based Evaluation of Geo-Spatial Reasoning in Foundation Models
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2501.00316v2](http://arxiv.org/pdf/2501.00316v2)**

> **作者:** Mahir Labib Dihan; Md Tanvir Hassan; Md Tanvir Parvez; Md Hasebul Hasan; Md Almash Alam; Muhammad Aamir Cheema; Mohammed Eunus Ali; Md Rizwan Parvez
>
> **备注:** ICML 2025 (Spotlight)
>
> **摘要:** Recent advancements in foundation models have improved autonomous tool usage and reasoning, but their capabilities in map-based reasoning remain underexplored. To address this, we introduce MapEval, a benchmark designed to assess foundation models across three distinct tasks - textual, API-based, and visual reasoning - through 700 multiple-choice questions spanning 180 cities and 54 countries, covering spatial relationships, navigation, travel planning, and real-world map interactions. Unlike prior benchmarks that focus on simple location queries, MapEval requires models to handle long-context reasoning, API interactions, and visual map analysis, making it the most comprehensive evaluation framework for geospatial AI. On evaluation of 30 foundation models, including Claude-3.5-Sonnet, GPT-4o, and Gemini-1.5-Pro, none surpass 67% accuracy, with open-source models performing significantly worse and all models lagging over 20% behind human performance. These results expose critical gaps in spatial inference, as models struggle with distances, directions, route planning, and place-specific reasoning, highlighting the need for better geospatial AI to bridge the gap between foundation models and real-world navigation. All the resources are available at: https://mapeval.github.io/.
>
---
#### [replaced 054] Not All Jokes Land: Evaluating Large Language Models Understanding of Workplace Humor
- **分类: cs.CL; cs.CY**

- **链接: [http://arxiv.org/pdf/2506.01819v2](http://arxiv.org/pdf/2506.01819v2)**

> **作者:** Mohammadamin Shafiei; Hamidreza Saffari
>
> **摘要:** With the recent advances in Artificial Intelligence (AI) and Large Language Models (LLMs), the automation of daily tasks, like automatic writing, is getting more and more attention. Hence, efforts have focused on aligning LLMs with human values, yet humor, particularly professional industrial humor used in workplaces, has been largely neglected. To address this, we develop a dataset of professional humor statements along with features that determine the appropriateness of each statement. Our evaluation of five LLMs shows that LLMs often struggle to judge the appropriateness of humor accurately.
>
---
#### [replaced 055] Images Speak Louder than Words: Understanding and Mitigating Bias in Vision-Language Model from a Causal Mediation Perspective
- **分类: cs.AI; cs.CL; cs.CV; I.2.7**

- **链接: [http://arxiv.org/pdf/2407.02814v3](http://arxiv.org/pdf/2407.02814v3)**

> **作者:** Zhaotian Weng; Zijun Gao; Jerone Andrews; Jieyu Zhao
>
> **摘要:** Vision-language models (VLMs) pre-trained on extensive datasets can inadvertently learn biases by correlating gender information with specific objects or scenarios. Current methods, which focus on modifying inputs and monitoring changes in the model's output probability scores, often struggle to comprehensively understand bias from the perspective of model components. We propose a framework that incorporates causal mediation analysis to measure and map the pathways of bias generation and propagation within VLMs. This approach allows us to identify the direct effects of interventions on model bias and the indirect effects of interventions on bias mediated through different model components. Our results show that image features are the primary contributors to bias, with significantly higher impacts than text features, specifically accounting for 32.57% and 12.63% of the bias in the MSCOCO and PASCAL-SENTENCE datasets, respectively. Notably, the image encoder's contribution surpasses that of the text encoder and the deep fusion encoder. Further experimentation confirms that contributions from both language and vision modalities are aligned and non-conflicting. Consequently, focusing on blurring gender representations within the image encoder, which contributes most to the model bias, reduces bias efficiently by 22.03% and 9.04% in the MSCOCO and PASCAL-SENTENCE datasets, respectively, with minimal performance loss or increased computational demands.
>
---
#### [replaced 056] The Canary's Echo: Auditing Privacy Risks of LLM-Generated Synthetic Text
- **分类: cs.CL; cs.CR; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.14921v2](http://arxiv.org/pdf/2502.14921v2)**

> **作者:** Matthieu Meeus; Lukas Wutschitz; Santiago Zanella-Béguelin; Shruti Tople; Reza Shokri
>
> **备注:** 42nd International Conference on Machine Learning (ICML 2025)
>
> **摘要:** How much information about training samples can be leaked through synthetic data generated by Large Language Models (LLMs)? Overlooking the subtleties of information flow in synthetic data generation pipelines can lead to a false sense of privacy. In this paper, we assume an adversary has access to some synthetic data generated by a LLM. We design membership inference attacks (MIAs) that target the training data used to fine-tune the LLM that is then used to synthesize data. The significant performance of our MIA shows that synthetic data leak information about the training data. Further, we find that canaries crafted for model-based MIAs are sub-optimal for privacy auditing when only synthetic data is released. Such out-of-distribution canaries have limited influence on the model's output when prompted to generate useful, in-distribution synthetic data, which drastically reduces their effectiveness. To tackle this problem, we leverage the mechanics of auto-regressive models to design canaries with an in-distribution prefix and a high-perplexity suffix that leave detectable traces in synthetic data. This enhances the power of data-based MIAs and provides a better assessment of the privacy risks of releasing synthetic data generated by LLMs.
>
---
#### [replaced 057] The Synergy of LLMs & RL Unlocks Offline Learning of Generalizable Language-Conditioned Policies with Low-fidelity Data
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2412.06877v2](http://arxiv.org/pdf/2412.06877v2)**

> **作者:** Thomas Pouplin; Katarzyna Kobalczyk; Hao Sun; Mihaela van der Schaar
>
> **备注:** Accepted at International Conference on Machine Learning (ICML) 2025
>
> **摘要:** Developing autonomous agents capable of performing complex, multi-step decision-making tasks specified in natural language remains a significant challenge, particularly in realistic settings where labeled data is scarce and real-time experimentation is impractical. Existing reinforcement learning (RL) approaches often struggle to generalize to unseen goals and states, limiting their applicability. In this paper, we introduce TEDUO, a novel training pipeline for offline language-conditioned policy learning in symbolic environments. Unlike conventional methods, TEDUO operates on readily available, unlabeled datasets and addresses the challenge of generalization to previously unseen goals and states. Our approach harnesses large language models (LLMs) in a dual capacity: first, as automatization tools augmenting offline datasets with richer annotations, and second, as generalizable instruction-following agents. Empirical results demonstrate that TEDUO achieves data-efficient learning of robust language-conditioned policies, accomplishing tasks beyond the reach of conventional RL frameworks or out-of-the-box LLMs alone.
>
---
#### [replaced 058] Where is the signal in tokenization space?
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2408.08541v2](http://arxiv.org/pdf/2408.08541v2)**

> **作者:** Renato Lui Geh; Honghua Zhang; Kareem Ahmed; Benjie Wang; Guy Van den Broeck
>
> **备注:** Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, EMNLP 2024
>
> **摘要:** Large Language Models (LLMs) are typically shipped with tokenizers that deterministically encode text into so-called canonical token sequences, to which the LLMs assign probability values. One common assumption is that the probability of a piece of text is the probability of its canonical token sequence. However, the tokenization of a string is not unique: e.g., the Llama2 tokenizer encodes Tokens as [Tok,ens], but [Tok,en,s] also represents the same text. In this paper, we study non-canonical tokenizations. We prove that, given a string, it is computationally hard to find the most likely tokenization for an autoregressive LLM, as well as to compute the marginal probability over all possible tokenizations. We then show how the marginal is, in most cases, indistinguishable from the canonical probability. Surprisingly, we then empirically demonstrate the existence of a significant amount of signal hidden within tokenization space. Notably, by simply aggregating the probabilities of non-canonical tokenizations, we achieve improvements across a range of LLM evaluation benchmarks for a variety of architectures, including transformers and state space models.
>
---
#### [replaced 059] Instructor-Worker Large Language Model System for Policy Recommendation: a Case Study on Air Quality Analysis of the January 2025 Los Angeles Wildfires
- **分类: cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.00566v2](http://arxiv.org/pdf/2503.00566v2)**

> **作者:** Kyle Gao; Dening Lu; Liangzhi Li; Nan Chen; Hongjie He; Linlin Xu; Jonathan Li
>
> **摘要:** The Los Angeles wildfires of January 2025 caused more than 250 billion dollars in damage and lasted for nearly an entire month before containment. Following our previous work, the Digital Twin Building, we modify and leverage the multi-agent large language model framework as well as the cloud-mapping integration to study the air quality during the Los Angeles wildfires. Recent advances in large language models have allowed for out-of-the-box automated large-scale data analysis. We use a multi-agent large language system comprised of an Instructor agent and Worker agents. Upon receiving the users' instructions, the Instructor agent retrieves the data from the cloud platform and produces instruction prompts to the Worker agents. The Worker agents then analyze the data and provide summaries. The summaries are finally input back into the Instructor agent, which then provides the final data analysis. We test this system's capability for data-based policy recommendation by assessing our Instructor-Worker LLM system's health recommendations based on air quality during the Los Angeles wildfires.
>
---
#### [replaced 060] GRASP: Replace Redundant Layers with Adaptive Singular Parameters for Efficient Model Compression
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2501.00339v3](http://arxiv.org/pdf/2501.00339v3)**

> **作者:** Kainan Liu; Yong Zhang; Ning Cheng; Zhitao Li; Shaojun Wang; Jing Xiao
>
> **备注:** 15 pages, 5 figures
>
> **摘要:** Recent studies have demonstrated that many layers are functionally redundant in large language models (LLMs), enabling model compression by removing these layers to reduce inference cost. While such approaches can improve efficiency, indiscriminate layer pruning often results in significant performance degradation. In this paper, we propose GRASP (Gradient-based Retention of Adaptive Singular Parameters), a novel compression framework that mitigates this issue by preserving sensitivity-aware singular values. Unlike direct layer pruning, GRASP leverages gradient-based attribution on a small calibration dataset to adaptively identify and retain critical singular components. By replacing redundant layers with only a minimal set of parameters, GRASP achieves efficient compression while maintaining strong performance with minimal overhead. Experiments across multiple LLMs show that GRASP consistently outperforms existing compression methods, achieving 90% of the original model's performance under a 20% compression ratio.
>
---
#### [replaced 061] m-KAILIN: Knowledge-Driven Agentic Scientific Corpus Distillation Framework for Biomedical Large Language Models Training
- **分类: cs.CL; cs.AI; q-bio.QM**

- **链接: [http://arxiv.org/pdf/2504.19565v2](http://arxiv.org/pdf/2504.19565v2)**

> **作者:** Meng Xiao; Xunxin Cai; Qingqing Long; Chengrui Wang; Yuanchun Zhou; Hengshu Zhu
>
> **备注:** Biomedical large language models, corpus distillation, question-answer, agentic AI. arXiv admin note: text overlap with arXiv:2501.15108
>
> **摘要:** Corpus distillation for biomedical large language models (LLMs) seeks to address the pressing challenge of insufficient quantity and quality in open-source annotated scientific corpora, which remains a bottleneck for effective LLM training in biomedical research. This paper proposes a knowledge-driven, agentic framework for scientific corpus distillation, tailored explicitly for LLM training in the biomedical domain, addressing the challenge posed by the complex hierarchy of biomedical knowledge. Central to our approach is a collaborative multi-agent architecture, where specialized agents, each guided by the Medical Subject Headings (MeSH) hierarchy, work in concert to autonomously extract, synthesize, and self-evaluate high-quality textual data from vast scientific literature. This agentic framework collectively generates and refines domain-specific question-answer pairs, ensuring comprehensive coverage and consistency with biomedical ontologies while minimizing manual involvement. Extensive experimental results show that language models trained on our multi-agent distilled datasets achieve notable improvements in biomedical question-answering tasks, outperforming both strong life sciences LLM baselines and advanced proprietary models. Notably, our AI-Ready dataset enables Llama3-70B to surpass GPT-4 with MedPrompt and Med-PaLM-2, despite their larger scale. Detailed ablation studies and case analyses further validate the effectiveness and synergy of each agent within the framework, highlighting the potential of multi-agent collaboration in biomedical LLM training.
>
---
#### [replaced 062] The Impact of Inference Acceleration on Bias of LLMs
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.22118v3](http://arxiv.org/pdf/2410.22118v3)**

> **作者:** Elisabeth Kirsten; Ivan Habernal; Vedant Nanda; Muhammad Bilal Zafar
>
> **摘要:** Last few years have seen unprecedented advances in capabilities of Large Language Models (LLMs). These advancements promise to benefit a vast array of application domains. However, due to their immense size, performing inference with LLMs is both costly and slow. Consequently, a plethora of recent work has proposed strategies to enhance inference efficiency, e.g., quantization, pruning, and caching. These acceleration strategies reduce the inference cost and latency, often by several factors, while maintaining much of the predictive performance measured via common benchmarks. In this work, we explore another critical aspect of LLM performance: demographic bias in model generations due to inference acceleration optimizations. Using a wide range of metrics, we probe bias in model outputs from a number of angles. Analysis of outputs before and after inference acceleration shows significant change in bias. Worryingly, these bias effects are complex and unpredictable. A combination of an acceleration strategy and bias type may show little bias change in one model but may lead to a large effect in another. Our results highlight a need for in-depth and case-by-case evaluation of model bias after it has been modified to accelerate inference.
>
---
#### [replaced 063] WER We Stand: Benchmarking Urdu ASR Models
- **分类: cs.CL; cs.SD; eess.AS**

- **链接: [http://arxiv.org/pdf/2409.11252v3](http://arxiv.org/pdf/2409.11252v3)**

> **作者:** Samee Arif; Sualeha Farid; Aamina Jamal Khan; Mustafa Abbas; Agha Ali Raza; Awais Athar
>
> **摘要:** This paper presents a comprehensive evaluation of Urdu Automatic Speech Recognition (ASR) models. We analyze the performance of three ASR model families: Whisper, MMS, and Seamless-M4T using Word Error Rate (WER), along with a detailed examination of the most frequent wrong words and error types including insertions, deletions, and substitutions. Our analysis is conducted using two types of datasets, read speech and conversational speech. Notably, we present the first conversational speech dataset designed for benchmarking Urdu ASR models. We find that seamless-large outperforms other ASR models on the read speech dataset, while whisper-large performs best on the conversational speech dataset. Furthermore, this evaluation highlights the complexities of assessing ASR models for low-resource languages like Urdu using quantitative metrics alone and emphasizes the need for a robust Urdu text normalization system. Our findings contribute valuable insights for developing robust ASR systems for low-resource languages like Urdu.
>
---
#### [replaced 064] Opt-Out: Investigating Entity-Level Unlearning for Large Language Models via Optimal Transport
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2406.12329v3](http://arxiv.org/pdf/2406.12329v3)**

> **作者:** Minseok Choi; Daniel Rim; Dohyun Lee; Jaegul Choo
>
> **备注:** ACL 2025 Main
>
> **摘要:** Instruction-following large language models (LLMs), such as ChatGPT, have become widely popular among everyday users. However, these models inadvertently disclose private, sensitive information to their users, underscoring the need for machine unlearning techniques to remove selective information from the models. While prior work has focused on forgetting small, random subsets of training data at the instance-level, we argue that real-world scenarios often require the removal of an entire user data, which may require a more careful maneuver. In this study, we explore entity-level unlearning, which aims to erase all knowledge related to a target entity while preserving the remaining model capabilities. To address this, we introduce Opt-Out, an optimal transport-based unlearning method that utilizes the Wasserstein distance from the model's initial parameters to achieve more effective and fine-grained unlearning. We also present the first Entity-Level Unlearning Dataset (ELUDe) designed to evaluate entity-level unlearning. Our empirical results demonstrate that Opt-Out surpasses existing methods, establishing a new standard for secure and adaptable LLMs that can accommodate user data removal requests without the need for full retraining.
>
---
#### [replaced 065] CAT-LLM: Style-enhanced Large Language Models with Text Style Definition for Chinese Article-style Transfer
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2401.05707v2](http://arxiv.org/pdf/2401.05707v2)**

> **作者:** Zhen Tao; Dinghao Xi; Zhiyu Li; Liumin Tang; Wei Xu
>
> **摘要:** Text style transfer plays a vital role in online entertainment and social media. However, existing models struggle to handle the complexity of Chinese long texts, such as rhetoric, structure, and culture, which restricts their broader application. To bridge this gap, we propose a Chinese Article-style Transfer (CAT-LLM) framework, which addresses the challenges of style transfer in complex Chinese long texts. At its core, CAT-LLM features a bespoke pluggable Text Style Definition (TSD) module that integrates machine learning algorithms to analyze and model article styles at both word and sentence levels. This module acts as a bridge, enabling LLMs to better understand and adapt to the complexities of Chinese article styles. Furthermore, it supports the dynamic expansion of internal style trees, enabling the framework to seamlessly incorporate new and diverse style definitions, enhancing adaptability and scalability for future research and applications. Additionally, to facilitate robust evaluation, we created ten parallel datasets using a combination of ChatGPT and various Chinese texts, each corresponding to distinct writing styles, significantly improving the accuracy of the model evaluation and establishing a novel paradigm for text style transfer research. Extensive experimental results demonstrate that CAT-LLM, combined with GPT-3.5-Turbo, achieves state-of-the-art performance, with a transfer accuracy F1 score of 79.36% and a content preservation F1 score of 96.47% on the "Fortress Besieged" dataset. These results highlight CAT-LLM's innovative contributions to style transfer research, including its ability to preserve content integrity while achieving precise and flexible style transfer across diverse Chinese text domains. Building on these contributions, CAT-LLM presents significant potential for advancing Chinese digital media and facilitating automated content creation.
>
---
#### [replaced 066] A Survey on Sparse Autoencoders: Interpreting the Internal Mechanisms of Large Language Models
- **分类: cs.LG; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.05613v2](http://arxiv.org/pdf/2503.05613v2)**

> **作者:** Dong Shu; Xuansheng Wu; Haiyan Zhao; Daking Rai; Ziyu Yao; Ninghao Liu; Mengnan Du
>
> **备注:** 23 pages, 3 figures
>
> **摘要:** Large Language Models (LLMs) have transformed natural language processing, yet their internal mechanisms remain largely opaque. Recently, mechanistic interpretability has attracted significant attention from the research community as a means to understand the inner workings of LLMs. Among various mechanistic interpretability approaches, Sparse Autoencoders (SAEs) have emerged as a promising method due to their ability to disentangle the complex, superimposed features within LLMs into more interpretable components. This paper presents a comprehensive survey of SAEs for interpreting and understanding the internal workings of LLMs. Our major contributions include: (1) exploring the technical framework of SAEs, covering basic architecture, design improvements, and effective training strategies; (2) examining different approaches to explaining SAE features, categorized into input-based and output-based explanation methods; (3) discussing evaluation methods for assessing SAE performance, covering both structural and functional metrics; and (4) investigating real-world applications of SAEs in understanding and manipulating LLM behaviors.
>
---
#### [replaced 067] TituLLMs: A Family of Bangla LLMs with Comprehensive Benchmarking
- **分类: cs.CL; cs.AI; 68T50; F.2.2; I.2.7**

- **链接: [http://arxiv.org/pdf/2502.11187v3](http://arxiv.org/pdf/2502.11187v3)**

> **作者:** Shahriar Kabir Nahin; Rabindra Nath Nandi; Sagor Sarker; Quazi Sarwar Muhtaseem; Md Kowsher; Apu Chandraw Shill; Md Ibrahim; Mehadi Hasan Menon; Tareq Al Muntasir; Firoj Alam
>
> **备注:** LLMs, Benchmarking, Large Language Models, Bangla, BanglaLLMs
>
> **摘要:** In this paper, we present TituLLMs, the first large pretrained Bangla LLMs, available in 1b and 3b parameter sizes. Due to computational constraints during both training and inference, we focused on smaller models. To train TituLLMs, we collected a pretraining dataset of approximately ~37 billion tokens. We extended the Llama-3.2 tokenizer to incorporate language- and culture-specific knowledge, which also enables faster training and inference. There was a lack of benchmarking datasets to benchmark LLMs for Bangla. To address this gap, we developed five benchmarking datasets. We benchmarked various LLMs, including TituLLMs, and demonstrated that TituLLMs outperforms its initial multilingual versions. However, this is not always the case, highlighting the complexities of language adaptation. Our work lays the groundwork for adapting existing multilingual open models to other low-resource languages. To facilitate broader adoption and further research, we have made the TituLLMs models and benchmarking datasets publicly available (https://huggingface.co/collections/hishab/titulm-llama-family-6718d31fc1b83529276f490a).
>
---
#### [replaced 068] GraphCheck: Multi-Path Fact-Checking with Entity-Relationship Graphs
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2502.20785v2](http://arxiv.org/pdf/2502.20785v2)**

> **作者:** Hyewon Jeon; Jay-Yoon Lee
>
> **摘要:** Automated fact-checking aims to assess the truthfulness of textual claims based on relevant evidence. However, verifying complex claims that require multi-hop reasoning remains a significant challenge. We propose GraphCheck, a novel framework that transforms claims into entity-relationship graphs for structured and systematic verification. By explicitly modeling both explicit and latent entities and exploring multiple reasoning paths, GraphCheck improves verification robustness. While GraphCheck excels in complex scenarios, it may be unnecessarily elaborate for simpler claims. To address this, we introduce DP-GraphCheck, a variant that employs a lightweight strategy selector to adaptively choose between direct prompting and GraphCheck. This selective mechanism improves both accuracy and efficiency by applying the appropriate level of reasoning to each claim. Experiments on the HOVER and EX-FEVER datasets demonstrate that our approach outperforms existing methods, particularly on multi-hop claims. Moreover, the strategy selection mechanism in DP-GraphCheck generalizes well to other fact-checking pipelines, highlighting the versatility of our framework.
>
---
#### [replaced 069] GuessBench: Sensemaking Multimodal Creativity in the Wild
- **分类: cs.CL**

- **链接: [http://arxiv.org/pdf/2506.00814v2](http://arxiv.org/pdf/2506.00814v2)**

> **作者:** Zifeng Zhu; Shangbin Feng; Herun Wan; Ningnan Wang; Minnan Luo; Yulia Tsvetkov
>
> **摘要:** We propose GuessBench, a novel benchmark that evaluates Vision Language Models (VLMs) on modeling the pervasive, noisy, and pluralistic human creativity. GuessBench sources data from "Guess the Build", an online multiplayer Minecraft minigame where one player constructs a Minecraft build given a concept (e.g. caterpillar) and others try to guess it with natural language hints, presenting a pristine testbed for sensemaking creativity in the wild with VLMs acting as guessers. We curate 1500 images from the actual gameplay and design 2000 problems spanning static and dynamic image settings, natural language hints of varying completeness, and more. Extensive experiments with six open/API VLMs and five reasoning enhancement approaches demonstrate that GuessBench presents a uniquely challenging task in creativity modeling: even the start-of-the-art GPT-4o is incorrect on 34% of instances, while we observe a huge performance gap (13.87% vs. 53.93% on average) between open and API models. When used as a resource to improve VLMs, fine-tuning on the reasoning traces for GuessBench problems improves visual perception tasks by 15.36% on average. Further analysis reveals that VLM performance in creativity sensemaking correlates with the frequency of the concept in training data, while the accuracy drops sharply for concepts in underrepresented cultural contexts and low-resource languages.
>
---
#### [replaced 070] Dissecting Bias in LLMs: A Mechanistic Interpretability Perspective
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.05166v2](http://arxiv.org/pdf/2506.05166v2)**

> **作者:** Bhavik Chandna; Zubair Bashir; Procheta Sen
>
> **摘要:** Large Language Models (LLMs) are known to exhibit social, demographic, and gender biases, often as a consequence of the data on which they are trained. In this work, we adopt a mechanistic interpretability approach to analyze how such biases are structurally represented within models such as GPT-2 and Llama2. Focusing on demographic and gender biases, we explore different metrics to identify the internal edges responsible for biased behavior. We then assess the stability, localization, and generalizability of these components across dataset and linguistic variations. Through systematic ablations, we demonstrate that bias-related computations are highly localized, often concentrated in a small subset of layers. Moreover, the identified components change across fine-tuning settings, including those unrelated to bias. Finally, we show that removing these components not only reduces biased outputs but also affects other NLP tasks, such as named entity recognition and linguistic acceptability judgment because of the sharing of important components with these tasks.
>
---
#### [replaced 071] IDA-Bench: Evaluating LLMs on Interactive Guided Data Analysis
- **分类: cs.CL; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.18223v2](http://arxiv.org/pdf/2505.18223v2)**

> **作者:** Hanyu Li; Haoyu Liu; Tingyu Zhu; Tianyu Guo; Zeyu Zheng; Xiaotie Deng; Michael I. Jordan
>
> **摘要:** Large Language Models (LLMs) show promise as data analysis agents, but existing benchmarks overlook the iterative nature of the field, where experts' decisions evolve with deeper insights of the dataset. To address this, we introduce IDA-Bench, a novel benchmark evaluating LLM agents in multi-round interactive scenarios. Derived from complex Kaggle notebooks, tasks are presented as sequential natural language instructions by an LLM-simulated user. Agent performance is judged by comparing its final numerical output to the human-derived baseline. Initial results show that even state-of-the-art coding agents (like Claude-3.7-thinking) succeed on < 50% of the tasks, highlighting limitations not evident in single-turn tests. This work underscores the need to improve LLMs' multi-round capabilities for building more reliable data analysis agents, highlighting the necessity of achieving a balance between instruction following and reasoning.
>
---
#### [replaced 072] A semantic embedding space based on large language models for modelling human beliefs
- **分类: cs.CL; cs.CY; physics.soc-ph**

- **链接: [http://arxiv.org/pdf/2408.07237v3](http://arxiv.org/pdf/2408.07237v3)**

> **作者:** Byunghwee Lee; Rachith Aiyappa; Yong-Yeol Ahn; Haewoon Kwak; Jisun An
>
> **备注:** 5 figures, 2 tables (SI: 25 figures, 7 tables). Published in Nature Human Behaviour (2025)
>
> **摘要:** Beliefs form the foundation of human cognition and decision-making, guiding our actions and social connections. A model encapsulating beliefs and their interrelationships is crucial for understanding their influence on our actions. However, research on belief interplay has often been limited to beliefs related to specific issues and relied heavily on surveys. We propose a method to study the nuanced interplay between thousands of beliefs by leveraging an online user debate data and mapping beliefs onto a neural embedding space constructed using a fine-tuned large language model (LLM). This belief space captures the interconnectedness and polarization of diverse beliefs across social issues. Our findings show that positions within this belief space predict new beliefs of individuals and estimate cognitive dissonance based on the distance between existing and new beliefs. This study demonstrates how LLMs, combined with collective online records of human beliefs, can offer insights into the fundamental principles that govern human belief formation.
>
---
#### [replaced 073] ResearchTown: Simulator of Human Research Community
- **分类: cs.CL; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.17767v2](http://arxiv.org/pdf/2412.17767v2)**

> **作者:** Haofei Yu; Zhaochen Hong; Zirui Cheng; Kunlun Zhu; Keyang Xuan; Jinwei Yao; Tao Feng; Jiaxuan You
>
> **备注:** 9 pages, ICML 2025
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable potential in scientific domains, yet a fundamental question remains unanswered: Can we simulate human research communities with LLMs? Addressing this question can deepen our understanding of the processes behind idea brainstorming and inspire the automatic discovery of novel scientific insights. In this work, we propose ResearchTown, a multi-agent framework for research community simulation. Within this framework, the human research community is simplified as an agent-data graph, where researchers and papers are represented as agent-type and data-type nodes, respectively, and connected based on their collaboration relationships. We also introduce TextGNN, a text-based inference framework that models various research activities (e.g., paper reading, paper writing, and review writing) as special forms of a unified message-passing process on the agent-data graph. To evaluate the quality of the research community simulation, we present ResearchBench, a benchmark that uses a node-masking prediction task for scalable and objective assessment based on similarity. Our experiments reveal three key findings: (1) ResearchTown can provide a realistic simulation of collaborative research activities, including paper writing and review writing; (2) ResearchTown can maintain robust simulation with multiple researchers and diverse papers; (3) ResearchTown can generate interdisciplinary research ideas that potentially inspire pioneering research directions.
>
---
#### [replaced 074] Fundamental Limits of Prompt Tuning Transformers: Universality, Capacity and Efficiency
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **链接: [http://arxiv.org/pdf/2411.16525v2](http://arxiv.org/pdf/2411.16525v2)**

> **作者:** Jerry Yao-Chieh Hu; Wei-Po Wang; Ammar Gilani; Chenyang Li; Zhao Song; Han Liu
>
> **备注:** Accepted at ICLR 2025. v2 matches the camera-ready version
>
> **摘要:** We investigate the statistical and computational limits of prompt tuning for transformer-based foundation models. Our key contributions are prompt tuning on \emph{single-head} transformers with only a \emph{single} self-attention layer: (i) is universal, and (ii) supports efficient (even almost-linear time) algorithms under the Strong Exponential Time Hypothesis (SETH). Statistically, we prove that prompt tuning on such simplest possible transformers are universal approximators for sequence-to-sequence Lipschitz functions. In addition, we provide an exponential-in-$dL$ and -in-$(1/\epsilon)$ lower bound on the required soft-prompt tokens for prompt tuning to memorize any dataset with 1-layer, 1-head transformers. Computationally, we identify a phase transition in the efficiency of prompt tuning, determined by the norm of the \emph{soft-prompt-induced} keys and queries, and provide an upper bound criterion. Beyond this criterion, no sub-quadratic (efficient) algorithm for prompt tuning exists under SETH. Within this criterion, we showcase our theory by proving the existence of almost-linear time prompt tuning inference algorithms. These fundamental limits provide important necessary conditions for designing expressive and efficient prompt tuning methods for practitioners.
>
---
#### [replaced 075] Multi-Agent Collaboration via Cross-Team Orchestration
- **分类: cs.CL; cs.AI; cs.MA; cs.SE**

- **链接: [http://arxiv.org/pdf/2406.08979v2](http://arxiv.org/pdf/2406.08979v2)**

> **作者:** Zhuoyun Du; Chen Qian; Wei Liu; Zihao Xie; YiFei Wang; Rennai Qiu; Yufan Dang; Weize Chen; Cheng Yang; Ye Tian; Xuantang Xiong; Lei Han
>
> **备注:** Accepted to Findings of ACL 2025
>
> **摘要:** Large Language Models (LLMs) have significantly impacted various domains, especially through organized LLM-driven autonomous agents. A representative scenario is in software development, where agents can collaborate in a team like humans, following predefined phases to complete sub-tasks sequentially. However, for an agent team, each phase yields only one possible outcome. This results in the completion of only one development chain, thereby losing the opportunity to explore multiple potential decision paths within the solution space. Consequently leading to suboptimal results or extensive trial and error. To address this, we introduce Cross-Team Orchestration (Croto), a scalable multi-team framework that enables orchestrated teams to jointly propose various task-oriented solutions and interact with their insights in a self-independence while cross-team collaboration environment for superior solutions generation. Experiments reveal a notable increase in software quality compared to state-of-the-art baselines. We further tested our framework on story generation tasks, which demonstrated a promising generalization ability of our framework in other domains. The code and data is available at https://github.com/OpenBMB/ChatDev/tree/macnet
>
---
