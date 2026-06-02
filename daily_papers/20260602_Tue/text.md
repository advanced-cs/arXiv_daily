# 自然语言处理 cs.CL

- **最新发布 267 篇**

- **更新 183 篇**

## 最新发布

#### [new 001] Construction of Historical Knowledge Graphs Based on BERT and Graph Neural Networks
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于历史知识图谱构建任务，旨在解决传统文本中的实体与关系提取问题。结合BERT与GNN技术，提升知识图谱的准确性与全面性。**

- **链接: [https://arxiv.org/pdf/2606.01747](https://arxiv.org/pdf/2606.01747)**

> **作者:** Ping Li; Bartlomiej Brzozka
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Through digital humanities research and scale-up historical data analysis, a significant amount of traditional historical text is converted into structured knowledge graphs. This paper provides a high-level architecture that combines bidirectional encoder representations of transformers (BERT) and graph neural networks (GNN) to extract the entities and relationships from various types of historical texts. The texts of traditional history resolve linguistic ambiguities, references limited by context, and a lack of established grammatical norms in a systematic way. This study develops a new image retrieval system based on FastRQNet and pre-trained vision-language model Vilt-qaformer+RoBInet in accordance with the aforementioned recommendations. The experiments make full use of a comprehensive collection of municipal records, parliamentary documents, and historical correspondence. When compared to conventional rule-based techniques and other popular deep-learning baselines, the joint BERT-GNN system obtains greater Precision, Recall, and F1-score (Table 2). Complex nested structures and implicit reference issues can be handled by this structure with sufficient accuracy and thoroughness when creating knowledge graphs. The aforementioned experiments show that combining relational graph learning algorithms with context-sensitive semantic representation techniques can automatically extract historical data to add accumulated wisdom to the knowledge repository.
>
---
#### [new 002] Understanding LLM Behavior in Multi-Target Cross-Lingual Summarization
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多目标跨语言摘要任务，旨在提升多语言摘要质量。通过构建基准测试和分析模型内部机制，提出一种基于激活引导的方法，有效改善多语言摘要效果。**

- **链接: [https://arxiv.org/pdf/2606.01252](https://arxiv.org/pdf/2606.01252)**

> **作者:** Sangwon Ryu; Yihong Liu; Mingyang Wang; Yunsu Kim; Jungseul Ok; Gary Geunbae Lee; Hinrich Schuetze
>
> **摘要:** Multi-target cross-lingual text summarization (MTXLS), which summarizes a source document into multiple target languages, is increasingly important as users consume content in diverse languages, but remains underexplored. To address this gap, we introduce multi-target cross-lingual element-aware (MEA), a new MTXLS benchmark covering 24 target languages. We benchmark end-to-end and pipeline approaches across various LLMs and show that MTXLS performance still substantially lags behind English monolingual summarization. To better understand MTXLS in LLMs, we propose a layer-wise analysis framework for investigating how LLMs internally perform MTXLS. Our analyses suggest that translation and summarization behaviors emerge jointly within later layers rather than as distinctly decomposed stages. Most task-relevant processing occurs within these layers, and errors also tend to arise at similar depths. Motivated by these findings, we introduce an inference-time activation steering method that leverages hidden representations from English summarization to guide MTXLS generation. Experiments show that our method consistently improves MTXLS quality across target languages.
>
---
#### [new 003] Resonant Context Anchoring: Decoupling Attention Routing and Signal Gain at Inference Time
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决大模型事实性幻觉问题。提出RCA方法，在推理时增强外部证据信号，提升生成结果的准确性与真实性。**

- **链接: [https://arxiv.org/pdf/2606.01923](https://arxiv.org/pdf/2606.01923)**

> **作者:** Mingkuan Zhao; Yide Gao; Wentao Hu; Suquan Chen; Tianchen Huang; Zhenhua An; Zetao Chang; Xiayu Sun; Yuheng Min
>
> **摘要:** Large Language Models (LLMs) frequently exhibit "contextual disregard" when faced with input evidence that conflicts with their internal parametric memory, leading to persistent factual hallucinations. Existing mitigation strategies primarily rely on suppressing specific neuron activations or employing computationally expensive contrastive decoding mechanisms, which often result in increased perplexity or significantly elevated inference latency. To address these limitations, we propose Resonant Context Anchoring (RCA), a lightweight inference-time intervention method grounded in the perspective of residual stream signal dynamics. RCA aims to resolve the signal attenuation of external evidence during its propagation through deep networks. The core mechanism involves the orthogonal decoupling of routing logic and information magnitude within the self-attention module. By utilizing raw pre-softmax attention scores as an instantaneous metric of semantic alignment, we construct a dynamic gain field via non-linear rectification to selectively amplify the norms of value vectors corresponding to context tokens, without altering the attention probability distribution. This mechanism effectively elevates the signal-to-noise ratio (SNR) of input evidence within the residual stream mixture, thereby robustly anchoring the generation trajectory to the truthful context during inference. Extensive experiments on the Llama-3 model series demonstrate that RCA significantly improves contextual faithfulness across multiple factual consistency and strong knowledge-conflict tasks, effectively suppressing parametric hallucinations. Furthermore, results confirm that as a training-free and computationally negligible plug-and-play module, RCA achieves a Pareto improvement in faithfulness and fluency while maintaining the model's general language understanding capabilities.
>
---
#### [new 004] SPADER: Step-wise Peer Advantage with Diversity-Aware Exploration Rewards for Multi-Answer Question Answering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多答案问答任务，解决长轨迹信用分配和探索奖励对齐问题。提出SPADER框架，通过步骤级同伴优势和多样性感知奖励提升答案召回率和F1值。**

- **链接: [https://arxiv.org/pdf/2606.00593](https://arxiv.org/pdf/2606.00593)**

> **作者:** Qiming Shi; Zhaolu Kang; Yunfan Zhou; Di Weng; Yingcai Wu
>
> **摘要:** Large language models are increasingly deployed as tool-augmented agents to acquire information beyond parametric knowledge. While recent work has improved long-horizon tool-use reasoning, most approaches focus on tasks with a single correct answer. In contrast, many real-world queries require discovering a comprehensive set of valid answers, a setting known as Multi-Answer QA. This setting raises two challenges: fine-grained credit assignment over long search trajectories and reward alignment for sustained exploration beyond easy high-frequency entities. We propose SPADER, a reinforcement learning framework for long-horizon tool use in Multi-Answer QA. SPADER includes Step-wise Peer Advantage (SPA), a critic-free step-level credit assignment mechanism that aligns parallel trajectories by decision step and estimates advantages from peer returns. It also includes a diversity-aware exploration reward that promotes long-tail entity discovery by upweighting rare findings and downweighting redundant ones. Experiments on QAMPARI, Mintaka, WebQSP, and QUEST show that SPADER generally improves recall and overall F1 over prompting-based agents, outcome-supervised RL methods, and recent step-level supervision approaches. Our code and model weights are available at this https URL.
>
---
#### [new 005] DraDDP: A Multimodal Multi-Party Dialogue Discourse Parsing Dataset
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DraDDP数据集，用于多模态多方对话话语解析任务。解决现有研究在多模态和多方对话场景下的不足，通过构建首个公开的英文多模态数据集并进行基准测试，验证多模态信息的价值。**

- **链接: [https://arxiv.org/pdf/2606.00012](https://arxiv.org/pdf/2606.00012)**

> **作者:** Shannan Liu; Peifeng Li; Yaxin Fan; Qiaoming Zhu
>
> **摘要:** Multi-party dialogue discourse parsing aims to identify dependency structures and relation types between utterances in conversations. Previous studies are mostly limited to textual modality or two-party dialogue, failing to meet the multimodal and multi-party settings. In this paper, we construct the first publicly available English multimodal dataset DraDDP for multi-party dialogue discourse parsing, based on American TV dramas. DraDDP contains 495 dialogue segments with 6,374 utterances and 9.1 hours of parallel video content, covering rich multi-party interaction scenarios. Moreover, we establish comprehensive benchmarks by evaluating this task on DraDDP and conducting in-depth analysis on the impact of different modalities. Experimental results demonstrate the value of multimodal information in capturing dialogue structures and relation types. We will publicly release the dataset, annotation guidelines, and code to promote future research in multimodal dialogue understanding.
>
---
#### [new 006] Cross-Environment Neural Reranking for Sample-Efficient Action Selection in Text-Based Agents
- **分类: cs.CL**

- **简介: 该论文属于文本交互任务，旨在解决大模型推理成本高、需多环境维护的问题。通过跨环境联合训练轻量模型，提升样本效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2606.02204](https://arxiv.org/pdf/2606.02204)**

> **作者:** Kan Shao
>
> **备注:** 11 pages, 4 figures, 6 tables
>
> **摘要:** Large language model agents achieve strong performance on text-based benchmarks but incur prohibitive inference costs, motivating the use of compact neural rerankers for action selection. We investigate whether a single lightweight model can perform action selection across multiple diverse environments, a capability that would eliminate per-environment model maintenance. Training DeBERTa-v3 (184M-434M parameters) jointly on ALFWorld, WebShop, and ScienceWorld with minority-class upsampling, we find that rebalanced two-environment joint training substantially improves over single-environment ALFWorld performance (net gain +0.412) while maintaining competitive WebShop performance (+0.214 vs. +0.249 single-environment). Three-environment training yields a mean combined net gain of +0.551 +/- 0.024 across 4 seeds, with per-environment results approaching specialized single-environment models while providing positive cross-domain transfer. Cross-environment adaptation is highly sample-efficient: fine-tuning on only 9.2% of target-domain data recovers 93% of full-data performance, and scaling model capacity yields limited benefits, indicating data diversity is the primary driver. Environment-aware LoRA adapter routing with PCGrad achieves a best-seed result of +0.611 (seed 42), with seeds 456 and 789 at +0.554 and +0.559, but exhibits high variance due to seed 123 collapsing to +0.263 (4-seed mean +0.497 +/- 0.158), representing a promising but currently unstable direction. Joint training with clean splits and data rebalancing is a key ingredient. We will release our three-environment benchmark of 51,580 training instances (41,740 raw unique states with minority-class upsampling) and all model checkpoints upon acceptance.
>
---
#### [new 007] Do Text Edits Generalize to Visual Generation? Benchmarking Cross-Modal Knowledge Editing in UMMs
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究统一多模态模型（UMMs）中文本编辑对视觉生成的迁移问题，旨在解决跨模态知识编辑的有效性问题。通过构建基准数据集并提出改进方法，揭示了文本编辑在图像生成中的局限性。**

- **链接: [https://arxiv.org/pdf/2606.00477](https://arxiv.org/pdf/2606.00477)**

> **作者:** Xin Gao; Cheng Yang; Chufan Shi; Taylor Berg-Kirkpatrick
>
> **备注:** Published at ICML 2026; Code and data available at this https URL
>
> **摘要:** Unified multimodal models (UMMs) have emerged as a promising paradigm for general-purpose multimodal intelligence. As they are deployed in real-world applications, effectively updating internal knowledge becomes critical. While knowledge editing has matured for text-only models, it remains unclear whether edits that successfully modify textual outputs also transfer to image generation in UMMs. To study this question, we introduce UniKE, the first benchmark for cross-modality knowledge editing in UMMs, comprising 2,971 edit subjects spanning attribute and relation edits. Using VQA-based visual verification, we reveal a striking modality gap: text-side efficacy can reach approximately 92%, whereas the best overall VQA accuracy under direct image generation is only 18.5%. We further propose Reasoning-augmented Parameter Editing, which explicitly activates edited knowledge before generation and improves overall VQA accuracy for all evaluated model-editor pairs, with gains up to 18.6 percentage points. Mechanistic analysis shows that this gap is associated with partial alignment between edited textual representations and the conditioning pathways for visual generation, where edits sufficient for text outputs may remain too weak or misaligned to steer image synthesis. These findings show that textual knowledge edits do not guarantee reliable cross-modality transfer and motivate modality-aware editing methods. Our code and data are available at this https URL.
>
---
#### [new 008] What to Format and How: A Benchmark and Workflow Approach for Document Formatting
- **分类: cs.CL**

- **简介: 该论文属于文档格式化任务，解决内容感知的格式化难题。提出DocFormBench基准和DocFormFlow方法，提升格式化准确性和效率。**

- **链接: [https://arxiv.org/pdf/2606.01936](https://arxiv.org/pdf/2606.01936)**

> **作者:** Shihao Rao; Liang Li; Jiapeng Liu; Tong Lin; Bing Li; Xiyan Gao; Peng Fu; Jing Huang; Can Ma
>
> **摘要:** Recent advances in large language models (LLMs) have opened up new possibilities for automated document formatting. However, real-world formatting often requires identifying targets based on document content. This content-aware setting remains challenging and underexplored, primarily due to the lack of dedicated evaluation this http URL enable evaluation in realistic content-aware scenarios, we introduce DocFormBench, a benchmark that extends Text-to-Format evaluation to diverse formatting requirements, along with metrics for both accuracy and this http URL mitigate redundant document reading in existing methods during formatting, we propose DocFormFlow, a workflow formatting method that decouples target localization from modification execution into what to format and how. Extensive experiments across multiple LLMs and multimodal models show that DocFormFlow consistently improves formatting accuracy while reducing token consumption compared to representative baselines. Further analysis reveals that precise target localization is the primary factor influencing formatting performance. We hope DocFormBench and DocFormFlow will facilitate future research toward more intelligent and reliable document formatting.
>
---
#### [new 009] Thinking Economically: A Hierarchical Framework for Adaptive-Complexity Reasoning in LLMs
- **分类: cs.CL**

- **简介: 该论文属于大模型推理优化任务，旨在解决CoT方法中计算资源浪费问题。提出HAB框架，通过分层预算分配提升推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2606.01168](https://arxiv.org/pdf/2606.01168)**

> **作者:** Yubo Gao; Haotian Wu; Hong Chen; Junquan Huang; Yibo Yan; Jungang Li; Zihao Dongfang; Sicheng Tao; Puay Siew Tan; Jie Zhang; Xuming Hu
>
> **备注:** 11 pages, 4 figures, 3 tables
>
> **摘要:** Chain-of-Thought (CoT) has significantly enhanced LLM reasoning, yet often incurs substantial computational overhead due to "overthinking": generating excessively long rationales without commensurate accuracy gains. Existing efficiency methods typically apply uniform compression, which overlooks a critical observation that reasoning complexity is heterogeneous at two distinct granularity: across different problems and within individual reasoning steps. This motivates our principle of Thinking Economically: intelligently allocating computational resources based on intrinsic task and step demands rather than pursuing uniform brevity. We propose Hierarchical Adaptive Budgeter (HAB), a training framework that operationalizes this principle through coarse-to-fine budgeting. At the inter-step level, HAB predicts the optimal reasoning depth for each problem. At the intra-step level, HAB learns step-specific token budgeting signals from PPL-derived step comparisons and an adaptive Pareto optimization objective that captures the local quality-efficiency trade-off, while a Fisher Information-based pruner further provides fine-grained training-time guidance, thereby encouraging the generator to internalize more economical reasoning patterns. Experiments on GSM8K and MATH500 show that HAB not only surpasses standard CoT in accuracy but also reduces token usage, achieving a stronger performance-efficiency trade-off than the compared baselines.
>
---
#### [new 010] Chunking Methods on Retrieval-Augmented Generation - Effectiveness Evaluation Against Computational Cost and Limitations
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究RAG系统中的分块方法。旨在解决不同分块策略效果评估不足的问题，通过系统评估多种方法，揭示其优劣与挑战。**

- **链接: [https://arxiv.org/pdf/2606.00881](https://arxiv.org/pdf/2606.00881)**

> **作者:** Mateusz Śmigielski; Michał Rajkowski; Mateusz Zbrocki; Michał Bernacki-Janson; Karol Kunicki; Julianna Godziszewska; Maciej Piasecki; Konrad Wojtasik
>
> **摘要:** Retrieval-Augmented Generation (RAG) has demonstrated significant capabilities in enhancing the performance of Large Language Models (LLMs). One of the key tasks in RAG systems is the chunking process. Traditionally, fixed-size chunking and semantic chunking have been the standard approaches. However, interest in chunking strategies has been increasing, leading to a growing number of proposed methods that often claim improved performance over these conventional techniques. Many of these approaches are tailored to specific use cases and data types, with limited evidence of their effectiveness across diverse scenarios. As a result, it remains challenging to directly compare different techniques and assess their relative strengths. To the best of our knowledge, this study is the first to systematically evaluate the effectiveness of a wide range of chunking methods and emphasize the underlying challenges of chunking strategies in RAG systems. While chunking is commonly treated as a simple preprocessing step, we show that it introduces a range of impactful and often overlooked issues.
>
---
#### [new 011] FigSIM: A Dataset for Fine-grained Suicide Severity and Figurative Language in Suicide Memes
- **分类: cs.CL; cs.CV; cs.CY**

- **简介: 该论文提出FigSIM数据集，用于分析自杀表情包的严重程度和隐喻语言，解决内容审核难题。**

- **链接: [https://arxiv.org/pdf/2606.02523](https://arxiv.org/pdf/2606.02523)**

> **作者:** Liuliu Chen; Elise R. Carrotte; Brian E. Chapman; Jo Robinson; Mike Conway
>
> **备注:** Content warning: contains suicide-related content. Accepted to Findings of the Association for Computational Linguistics: ACL 2026
>
> **摘要:** Suicide memes are memes used to express suicide-related thoughts or comment on suicide-related issues. Suicide memes are increasingly common on social media, yet remain poorly understood and potentially harmful. There is an urgent need to better understand their characteristics and to develop appropriate content moderation strategies that limits users' exposure to potentially harmful content. Currently, the absence of annotated datasets of suicide memes remains a key barrier to developing and evaluating automated moderation approaches. In this paper, we introduce FigSIM, the first dataset designed for fine-grained analysis of suicide memes. The dataset consists of 1049 memes, each annotated for (1) fine-grained suicide severity levels, (2) figurative phenomena (e.g., metaphors), and (3) suicide-related content (e.g., suicide method depiction). We benchmark 16 unimodal and multimodal models across three tasks: figurative language, suicide severity, and suicide-related content detection. Overall, FigSIM demonstrates that suicide memes pose unique challenges for both modeling and content moderation. Analysis revealed biases, such as underprediction of higher suicide severity levels, especially for figurative memes. The dataset (including splits used for analyses) is publicly available. Content Warning: This paper contains suicide-related content that may be triggering.
>
---
#### [new 012] Efficient RAG with Intent-Aware Retrieval and Semantics-Preserving Chunking
- **分类: cs.CL**

- **简介: 该论文属于RAG任务，解决传统RAG系统因意图不敏感和信息碎片化导致的信息不足问题，提出InSemRAG框架，通过意图感知检索和语义保持分块提升效果。**

- **链接: [https://arxiv.org/pdf/2606.01240](https://arxiv.org/pdf/2606.01240)**

> **作者:** Fachrina Dewi Puspitasari; Chaoning Zhang; Jiaquan Zhang; Zhicheng Wang; Hafiz Shakeel Ahmad Awan; Rizwan Qureshi; Jewon Lee; Tae-Ho Kim; Yang Yang
>
> **摘要:** The demand for powerful instruction following and reasoning capability of large language models (LLMs) has promoted rapid development of retrieval-augmented generation (RAG). The RAG system assists LLM generation by retrieving chunks of query-fit supplementary knowledge from an external database. Conventional RAG systems, however, suffer from information insufficiency due to two factors, which are intent-agnostic retrieval and information fragmentation. Our work proposes a RAG framework, termed InSemRAG, that addresses these challenges via an iterative retrieve-and-check mechanism with two supporting modules, an intention-aware retriever (IAR) and semantics-preserving chunking (SPC). IAR implements a dynamic hybrid retrieval method that adaptively weights the retrieval channels based on the query intent, while SPC performs detection and reparation to the damaged evidence chunks to preserve the semantic integrity. To alleviate the computational latency brought by our iterative mechanism, we leverage small language models (SLMs). Extensive experiments across several benchmark datasets consistently demonstrate the competitiveness of our method against recent state-of-the-art RAG mechanisms. Particularly, our method achieves significant gains on multi-hop and evidence-sensitive tasks, with a 2.65-point improvement in F1 on HotPotQA and a 1.5-point increase in accuracy on FEVER. Our method also achieves competitive performance to Multi-Hop RAG with 4.32$\times$ lower latency with the utilization of SLM.
>
---
#### [new 013] HarnessForge: Joint Harness and Policy Evolution for Adaptive Agent Systems
- **分类: cs.CL**

- **简介: 该论文属于智能体系统适应性研究，解决LLM代理在异构任务中的适应问题。提出HarnessForge框架，通过协同进化提升代理系统性能。**

- **链接: [https://arxiv.org/pdf/2606.01779](https://arxiv.org/pdf/2606.01779)**

> **作者:** Mingju Chen; Can Lv; Guibin Zhang; Heng Chang; Shiji Zhou
>
> **备注:** 25 pages, 13 figures
>
> **摘要:** LLM agents are increasingly expected to operate across heterogeneous task regimes that require distinct execution paradigms. This challenges fixed agent systems and motivates system-level meta-adaptation beyond isolated component updates. While existing works have adapted external harness or trained underlying reasoning policies, full-system adaptation remains insufficiently characterized. The adaptation space between structure and execution is rarely made explicit, and the compatibility between the external harness and the internal reasoner is not optimized jointly. We propose HarnessForge, a meta-adaptive framework for evolving LLM agent systems. HarnessForge formulates an agent system as a harness--policy pair, defining a stable adaptation space that separates harness-level execution structure from policy-level reasoning behavior. It then performs harness--policy co-evolution through fault-guided harness tailoring and harness-conditioned policy alignment. Experiments across five benchmarks from diverse domains show that HarnessForge consistently improves both Qwen3-4B and Qwen3-8B backbones, outperforming harness-only and policy-only baselines with gains of up to 12.0\% over the strongest baseline and achieving favorable rollout-efficiency tradeoffs, demonstrating that harness--policy co-evolution is effective, and that executable compatibility between the harness and reasoning policy is essential for agent-system adaptation. The code is available at this https URL.
>
---
#### [new 014] A Finite-Calibration Regime Map for LLM Judge Panels
- **分类: cs.CL; stat.ME**

- **简介: 该论文研究LLM评委面板的校准策略，解决在有限人工标注预算下选择低维堆叠器或联合表格的问题。通过分析不同场景下的性能，提出一种校准制度图以优化选择。**

- **链接: [https://arxiv.org/pdf/2606.01034](https://arxiv.org/pdf/2606.01034)**

> **作者:** Bin Zhu; Yanghui Rao
>
> **备注:** Work in Progress
>
> **摘要:** We study when LLM judge panels should be calibrated with low-dimensional stackers versus joint output tables under finite human-label budgets. Low-dimensional stackers have small estimation cost but miss interactions, whereas joint-table calibrators can represent interactions but pay for cell counts and unseen patterns. We cast this tradeoff as a finite-calibration regime map and instantiate it as Finite-Calibration Panel Selection, a deployable validation selector over judge path, prefix size, and aggregator family with table and parametric estimation diagnostics. On RewardBench, LLMBar, SummEval, and Arena100K with a seven-judge pool including DeepSeek V4 Flash, scalar/reliability aggregation wins 16 of 20 real dataset--budget cells, indicating that current judge outputs are often additive or redundant. Controlled calibration-growth data show the complementary regime: additive labels remain scalar-favored, whereas a six-way interaction selects a larger joint table and its test MSE drops from 0.224 to 0.061 once unseen mass vanishes. Thus the practical question is not ``how many judges?'' but whether the next judge's information is estimable under the available human labels.
>
---
#### [new 015] Consistency Training while Mitigating Obfuscation via Rate Matching
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决模型受外部特征影响的问题。通过RMCT方法，在不约束表达方式的前提下提升模型行为一致性，增强鲁棒性并保持可监控性。**

- **链接: [https://arxiv.org/pdf/2606.02211](https://arxiv.org/pdf/2606.02211)**

> **作者:** Sohaib Imran; Prakhar Gupta; Jannes Elstner; David Demitri Africa
>
> **摘要:** Large language models are often influenced by extraneous input features, such as cues revealing a user's preferred answer. Consistency training reduces this influence by training models to behave similarly across inputs with and without the extraneous feature. However, existing methods train for consistency over entire responses or internal activations, which also constrains whether the model verbalises said extraneous features. We show this leads to obfuscation, where the model learns not to mention a cue while remaining influenced by it, which may undermine monitorability. To address this, we introduce Rate Matching Consistency Training (RMCT), which trains for consistency over selected behavioural properties without constraining how this behaviour is expressed. RMCT matches the rate at which the model exhibits a target behaviour (e.g., following a bias cue) across input perturbations, rather than requiring paired inputs with and without the extraneous feature, extending consistency training to settings where the extraneous features cannot be removed. We evaluate RMCT on sycophancy reduction in two open-weight language models, achieving reductions in bias-following comparable to a standard consistency-training baseline on held-out bias types, while largely preserving the model's tendency to verbalise the bias cue. Further, we find that RMCT is more data-efficient at the expense of being less compute-efficient in our experiments. Overall, RMCT shows that consistency training can improve behavioural robustness without directly trading off against monitorability.
>
---
#### [new 016] Not All Explanations Simulate Equally: Comparing Verbalized Feature Attributions and Self-Generated Rationales
- **分类: cs.CL**

- **简介: 该论文属于模型解释任务，旨在比较不同解释方式对模型行为模拟的效果。研究分析了特征归因与自生成推理两种解释形式，在共享反事实模拟设置下评估其可模拟性。**

- **链接: [https://arxiv.org/pdf/2606.01148](https://arxiv.org/pdf/2606.01148)**

> **作者:** Pingjun Hong; Benjamin Roth
>
> **摘要:** Natural-language explanations are often treated as a unified interface for understanding model behavior, but different explanation sources may support simulation in different ways. This paper compares two families of explanations for question answering models: verbalized feature attributions and self-generated rationales. We evaluate them under a shared counterfactual simulation setting, using an LLM judge as predictor and measuring whether it can better predict a model's answers to follow-up questions when given its explanation. Across multiple instruction-tuned models, we analyze how explanation source, verbalization strategy, and feature granularity affect the simulatability of explanations. Our results show that explanation format and granularity affect simulatability: attribution-based explanations and self-generated rationales differ in how much they improve counterfactual prediction, with effects that vary across models and formats.
>
---
#### [new 017] Better with Experience: Self-Evolving LLM Agents for Evidence-Grounded Health Community Notes
- **分类: cs.CL; cs.SI**

- **简介: 该论文提出EvoNote框架，解决健康谣言纠错中经验无法复用的问题，通过自进化机制提升社区笔记质量与效率。**

- **链接: [https://arxiv.org/pdf/2606.02215](https://arxiv.org/pdf/2606.02215)**

> **作者:** Zihang Fu; Fanxiao Li; Jianyang Gu; Haonan Wang; Preslav Nakov; Bryan Hooi; Min-Yen Kan; Jiaying Wu
>
> **摘要:** Large Language Model (LLM)-augmented Community Notes offer a scalable path for timely, evidence-grounded correction of health misinformation on social platforms. However, they still reset at every post, leaving useful correction experience from prior cases unused. We introduce EvoNote, an agentic framework that enables health Community Notes generation to self-evolve through an evolving experience memory of prior misinformation correction episodes. Its core is fine-grained credit assignment: EvoNote grounds trajectory-level feedback in health-specific note qualities and distills it into action-level memory for claim analysis, evidence acquisition, and note writing. We evaluate EvoNote on MM-HealthCN, a 1.2K-instance multimodal benchmark of user-flagged health posts with human-written Community Notes and crowd-derived helpfulness labels. Under a human-validated hierarchical utility judge, EvoNote-generated notes are preferred over corresponding human-written notes in 89.6% of cases; on a separate set of Needs More Ratings posts without a crowd helpfulness verdict, EvoNote produces helpful notes for 82.0% of cases. It also reduces the median time needed to produce a candidate correction from over 13 hours in the human-note pipeline to under 2 minutes. Analyses link these gains to stronger evidence use and reusable correction strategies, positioning self-evolving note generation as a promising paradigm for health misinformation governance.
>
---
#### [new 018] The Role of Ambiguity in Error Prediction via Uncertainty Quantification
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于错误预测任务，旨在提升大语言模型的错误预测能力。通过分离输入歧义与不确定性量化信号，改进误差预测效果。**

- **链接: [https://arxiv.org/pdf/2606.02093](https://arxiv.org/pdf/2606.02093)**

> **作者:** Ieva Raminta Staliūnaitė; James Bishop; Andreas Vlachos
>
> **备注:** 8 pages not including references and appendices, 3 figures
>
> **摘要:** The task of Error Prediction, namely predicting whether a model output is correct, is commonly tackled with Uncertainty Quantification (UQ). However, while uncertainty metrics capture when models lack knowledge or capacity to make a prediction, they also reflect aleatoric uncertainty, which is inherent in the model input and context. This paper presents a method for improving error prediction for Large Language Models (LLMs), by disentangling input ambiguity from UQ signal. We conduct experiments on the task of Question Answering (QA) with six UQ metrics and show that UQ metrics are more predictive of errors on unambiguous instances than on questions with multiple plausible answers. We use Gated Experts and Selective Prediction to incorporate gold and predicted ambiguity labels into the error prediction pipeline. We find that ambiguity information improves error prediction scores across model families, training and evaluation paradigms, datasets (including allegedly unambiguous ones), and sources of aleatoric uncertainty, yielding improvements of over 10 points of PRR for individual UQ metrics on standard datasets.
>
---
#### [new 019] On the Generalization Gap in Self-Evolving Language Model Reasoning
- **分类: cs.CL**

- **简介: 该论文研究自进化语言模型的泛化差距问题，旨在评估封闭环路下自生成监督与理想监督的接近程度。通过多种策略实验，发现自进化能提升性能，但仍有差距。**

- **链接: [https://arxiv.org/pdf/2606.01075](https://arxiv.org/pdf/2606.01075)**

> **作者:** Zhenting Qi; Susanna Maria Baby; Stefanie Anna Baby; Kan Yuan; Andrew Tomkins; Tu Vu; Da-Cheng Juan; Cyrus Rashtchian
>
> **摘要:** Recent work suggests that large language models (LLMs) can improve through self-evolution (SE), using supervision signals generated by the model itself. In this work, we ask: under a strict closed-loop setup, where the self-evolution algorithm has access only to an unlabeled prompt set and a base model, how close can internally generated supervision come to oracle-supervised training? We analyze four representative strategies in a unified offline self-evolution framework: single-round verification, multi-turn revision with feedback, iterative training, and curriculum learning. Our primary experiments use Knights and Knaves (KK) logical reasoning tasks, which provide deterministic solutions, controlled difficulty levels, and a clean testbed for easy-to-hard generalization. We first show that self-evolution consistently improves over the base model, but plateaus after excessive training compute is invested, and eventually still leaves a non-trivial gap to oracle supervision. We find that multi-turn critic-revision with large models can reach strong self-evolution performance, with Gemma 12B nearly matching oracle-supervised training. Beyond Knights and Knaves, we also evaluate self-evolution on real-world reasoning benchmarks, where gains are also modest. Overall, our results characterize when closed-loop self-evolution can help and show how internally generated supervision remains insufficient under this minimal formulation.
>
---
#### [new 020] KliniskVestBERT: BERT Model Specialised to Norwegian Clinical Texts
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升挪威临床文本的处理效果。通过预训练BERT模型，解决临床文本复杂性带来的挑战。**

- **链接: [https://arxiv.org/pdf/2606.01904](https://arxiv.org/pdf/2606.01904)**

> **作者:** Christian Autenried; Cosimo Persia
>
> **摘要:** The increasing application of Natural Language Processing (NLP) in healthcare demands language models specifically attuned to the complexities of clinical language. This work introduces KliniskVestBERT, a suite of three BERT-based encoder models pre-trained on a substantial corpus of real-world, de-identified Norwegian clinical texts from Helse Vest. We continue pretraining existing language models Nb-BERT-large, NorBERT3-large, and ModernBERT on our specialized clinical dataset. This dataset is based on a representative population of Helse Vest patients. The included document types are carefully curated to encompass a broad clinical spectrum in bokmål and nynorsk including discharge summaries, surgical reports, nursing notes etc. ensuring comprehensive representation of the linguistic landscape within Norwegian healthcare settings. Evaluation on three synthtetic Norwegian clinical benchmark datasets and two real-world problems demonstrates that each of our clinically specialized models consistently outperforms their baseline counterparts, highlighting the significant benefit of domain-specific pre-training for NLP tasks within the clinical domain. The project was a joint effort by all Helse Vest entities (Helse Bergen, Helse Fonna, Helse Førde and Helse Stavanger) with DIPS under the project lead of Helse Vest ICT.
>
---
#### [new 021] DSL-LLaDA: Scaling Continuous Denoising to 8B Masked Diffusion LMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DSL-LLaDA，解决文本生成中长度与质量的权衡问题。通过连续去噪和轻量微调，提升生成质量并减少重复。**

- **链接: [https://arxiv.org/pdf/2606.01024](https://arxiv.org/pdf/2606.01024)**

> **作者:** Longxuan Yu; Yunshu Wu; Yu Fu; Siheng Xiong; Rob Brekelmans; Hui Liu; Yue Dong; Greg Ver Steeg
>
> **备注:** 8 pages, 4 figures, 28 tables
>
> **摘要:** Discrete Masked diffusion language models generate text by iterative parallel decoding, but few-step decoding suffers from a tradeoff between length and quality: with a fixed step budget, standard methods can generate a short, high-quality output, or they can produce long but repetitive text. Continuous denoising can sidestep this tradeoff by evolving all positions jointly in embedding space, but building such a model from scratch at scale remains an open problem. We show that a pretrained masked DLM can instead be lightly adapted to support continuous embedding-space denoising. Starting from LLaDA-8B-Instruct, we continue-pretrain for only 1,000 steps with Discrete Stochastic Localization (DSL), replacing binary masking with continuous per-token Gaussian noise as a soft mask. The adapted model supports continuous inference that evolves all positions jointly in embedding space and defers hard token commitment to the final step. On zero-shot summarization at low step budgets (<=16 forward passes), DSL-LLaDA-SDE achieves the best ROUGE-1 on all four benchmarks and largely avoids the premature-termination / repetition tradeoff of iterative unmasking. The same adaptation also yields selective noisy-state robustness: the model corrects corrupted tokens while preserving clean ones. Control experiments using standard masked diffusion training with the same compute demonstrate neither behavior.
>
---
#### [new 022] Towards Lightweight Reliability: Using Soft Prompts for Hallucination Mitigation in Large Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于生成式问答任务，旨在解决大语言模型中的幻觉问题。通过引入软提示方法，平衡事实回忆、幻觉抑制和不确定性回避，提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2606.00919](https://arxiv.org/pdf/2606.00919)**

> **作者:** S M Tahmid Siddiqui; Akib Jawad Ononto; Anoop Singhal; Latifur Khan
>
> **备注:** 20 pages, 5 tables, 2 figures. Accepted for publication in DBSec 2026. The final publication will be available at Springer
>
> **摘要:** Large language models (LLMs) have seen widespread adoption across various domains, yet their reliability is frequently undermined by hallucinations - responses that are plausible-sounding but factually incorrect. In high-stakes domains, these errors can reduce trust and introduce real-world risk. To address this challenge, we present a parameter-efficient approach that uses soft prompts to mitigate hallucinated content and promote responsible abstention in generative question-answering (QA) tasks. Our method, called Responsible Contrastive Soft Prompting (RCSP), uses a composite loss to train soft prompts that balance three goals: suppressing hallucinatory content, encouraging abstention under uncertainty, and preserving or improving factual recall. To achieve these goals, we incorporate contrastive loss, curriculum learning, and KL regularization into our training mechanism. We evaluate our approach on five diverse generative QA datasets using an LLM-as-a-Judge framework. Experimental results on the Gemma 3 (12B) and Llama 3.1 (8B) backbones demonstrate that RCSP effectively balances factual recall with hallucination suppression and abstention, yielding a generally superior F-score over standard reasoning and instruction-based prompting baselines. Notably, these improvements are achieved by training only a fraction of the parameters required by other tuning techniques. Our results demonstrate that soft prompts provide a modular and computationally efficient path toward improving LLM reliability.
>
---
#### [new 023] Off-the-Shelf LLMs as Process Scorers: Training-Free Alternative to PRMs for Mathematical Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于数学推理任务，解决小模型因错误推理路径导致性能下降的问题。提出无需训练的Chunk-Level Guided Generation方法，利用大模型作为过程评分器，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2606.01682](https://arxiv.org/pdf/2606.01682)**

> **作者:** Atoosa Chegini; Soheil Feizi
>
> **摘要:** Selecting the best response from multiple small-model samples using a stronger scorer is a simple inference-time strategy, but fails when the small model has already committed to incorrect reasoning paths. PRM guided search avoids this by scoring candidate continuations during generation, but requires a reward model trained with step-level labels. We propose Chunk-Level Guided Generation, a training-free alternative that uses an off-the-shelf large language model as a process scorer. At each step, a small model samples k fixed-length candidate chunks, while the larger model scores the candidates using likelihoods without generating any text. The selected chunk is committed before the next step, steering generation before errors can propagate. We instantiate this framework with two selection rules: Likelihood-Guided Selection (LGS), which selects the chunk with the highest length-normalized large-model log-probability, and Contrastive-Guided Selection (CGS), which subtracts the small model's log-probability to favor chunks where the large model's preference diverges from the small model's. We show that scoring variable-length reasoning steps with large-model likelihoods is unreliable due to a systematic length bias that persists even after length normalization, and that fixed-length chunks avoid this confound. On GSM8K, MATH, Minerva Math, AMC23, and AIME24 with Qwen2.5-1.5B guided by Qwen2.5-32B and Llama-3.2-1B guided by Llama-3.1-70B, CGS outperforms majority voting by up to 28 pp and, under matched guidance budgets, matches or outperforms Qwen2.5-Math-PRM-72B guided search on most benchmarks without reward-model training. With Qwen2.5-7B guided by Qwen2.5-72B, CGS reaches 81.8% on MATH and 63.6% on Minerva Math at k=16, surpassing majority voting by 4--6 pp. Finally, Chunk-Level Guided Generation produces substantially shorter reasoning traces than PRM guided search.
>
---
#### [new 024] Citation Grounding: Detecting and Reducing LLM Citation Hallucinations via Legal Citation Graphs
- **分类: cs.CL; cs.DL**

- **简介: 该论文属于法律文本生成任务，旨在解决大语言模型在法律引用中的幻觉问题。通过构建引用图谱和提出CG指标，评估并减少错误引用。**

- **链接: [https://arxiv.org/pdf/2606.00898](https://arxiv.org/pdf/2606.00898)**

> **作者:** Volodymyr Ovcharov
>
> **备注:** 14 pages, 3 figures, 3 tables. Code and data: this https URL
>
> **摘要:** Large language models systematically hallucinate legal citations -- fabricating statute references, citing repealed provisions, and confusing jurisdictions -- yet no automated method exists to measure or reduce this behavior at scale. We propose citation grounding (CG), a metric that verifies LLM-generated legal citations against a ground-truth citation graph extracted from 100.8 million Ukrainian court decisions (502 million edges, 21,736 unique statute nodes). CG decomposes into three components -- citation precision (does the cited provision exist?), citation relevance (is it contextually appropriate?), and citation temporality (was it valid at the relevant date?) -- enabling differential diagnosis of hallucination types. Empirical evaluation on 100 Ukrainian legal queries across five systems -- four commercial LLMs via AWS Bedrock (Claude Haiku 4.5, Mistral Pixtral Large, Amazon Nova Pro/Lite) and one RAG-augmented production system -- reveals CG ranging from 0.791 to 0.873, with 13-21% of citations hallucinated. To reduce hallucinations without human annotation, we introduce Citation Grounding DPO (CG-DPO): a method that constructs preference pairs algorithmically by corrupting verified citations from real court decisions via four targeted strategies. On a dataset of 2,244 court decisions, a Qwen2.5-7B-Instruct model fine-tuned with LoRA achieves 98.5% mean validation accuracy in distinguishing correct from corrupted citations (rewards margin +14.9, std < 0.3 pp across 3 seeds). The citation graph, evaluation framework, and CG-DPO dataset are released as open resources.
>
---
#### [new 025] Encoded but Not Routed: Explaining the Table-Chart Gap in Scientific Claim Verification
- **分类: cs.CL**

- **简介: 该论文研究科学结论验证任务，解决图表与表格证据处理差异问题。通过分析模型中间表示和注意力机制，发现图表信息被编码但未有效传递至预测阶段。**

- **链接: [https://arxiv.org/pdf/2606.01679](https://arxiv.org/pdf/2606.01679)**

> **作者:** Sunisth Kumar; Xanh Ho; Tim Schopf; Andre Greiner-Petter; Florian Boudin; Akiko Aizawa
>
> **摘要:** Multimodal LLMs are increasingly used to assist scientific peer review, where a core requirement is verifying whether claims in a paper are supported by its evidence. Prior work has shown that models perform substantially better at this task when the evidence is a table than when it is a chart of the same underlying data. This raises the question of whether models fail to extract information from charts, or do they extract it but fail to use it when forming their prediction? We study this question through layer-wise linear probing and attention analysis on three open-weight VLMs over table and chart evidence, representing the same underlying data. We find consistent evidence for the latter. Chart information is encoded in the models' intermediate representations but does not reach the prediction position, a gap that is absent for tables and holds across all conditions tested. Attention analysis further reveals that this disconnect takes two architecturally distinct forms across model families. These findings reframe the table-chart gap as a failure of how encoded visual information is routed at prediction time, rather than a failure of encoding itself.
>
---
#### [new 026] Deep Research as Rubric for Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，解决开放性任务中缺乏可靠奖励信号的问题。通过构建DR-rubric框架，将rubric生成转化为研究过程，提升奖励信号的准确性与细粒度。**

- **链接: [https://arxiv.org/pdf/2606.01091](https://arxiv.org/pdf/2606.01091)**

> **作者:** Wangyi Mei; Zhouhong Gu; Zhenhan Bai; Yin Cai; Lefan Zhang; Zhenxin Ding; Bo Chen; Yan Gao; Yi Wu; Yao Hu; Jiaqing Liang; Deqing Yang
>
> **摘要:** Open-ended reasoning and long-form generation tasks lack reliable automatic verification signals for reward-based policy optimization. Rubrics offer a promising alternative, but existing approaches treat them as given artifacts -- either hand-crafted or prompt-generated -- and often miss the task-specific, knowledge-intensive dimensions that matter most, distorting the reward signal. Our key observation is that rubric construction is itself a research problem: identifying what makes a response correct or insightful requires discovering and synthesizing external knowledge. We propose Deep Research as Rubric (DR-rubric), a two-stage framework for constructing such rubrics. Stage I elicits domain facts, structural constraints, and failure modes through iterative multi-turn agentic search; Stage II distills this evidence into atomic, independently verifiable constraints for GRPO-based policy optimization. Because the model under training can serve as its own rubric generator, DR-rubric-8B supports bootstrap rubric generation without frontier-model assistance. We evaluate on 6 benchmarks spanning agentic research and expert reasoning. Experiments show that DR-Rubric achieves strong competitive performance with only 1K -- 3K training instances, where GPT-5-generated rubrics particularly benefit breadth coverage on agentic tasks, Gemini-generated rubrics yield the most balanced performance across agentic and expert reasoning tasks, and bootstrap rubrics exhibit a specialization-to-rebalancing evolution achieving the best overall performance at the third iteration. Results demonstrate that reframing rubric construction from static evaluation templates into an evidence-driven research process yields more scalable, fine-grained reward signals for open-ended tasks.
>
---
#### [new 027] AI as a Tool for Simulation-Based Experiments in Literary Studies
- **分类: cs.CL**

- **简介: 论文探讨AI在文学研究中的模拟实验应用，旨在解决如何利用AI生成符合文化约束的文学文本的问题。工作包括分析AI生成文本的特性及模拟人类文化生产的可行性。**

- **链接: [https://arxiv.org/pdf/2606.02293](https://arxiv.org/pdf/2606.02293)**

> **作者:** Matthew Wilkens
>
> **摘要:** Generative artificial intelligence (AI) systems open new possibilities for experimentation in literary studies via controlled, grounded, large-scale, low-cost simulations of cultural production. Current systems have not yet been shown to produce high-quality, book-length narrative texts that reliably reflect arbitrarily specified cultural constraints or stylistic features. But there exists substantial relevant research on each of the components required for literary-historical simulation. These include the use and validation of AI systems as proxies for differentiable human populations; the narrative and stylistic properties of AI-generated texts; the stability and coherence of multiagent, multiturn AI simulations of human actors; and technical methods through which to alter in predictable ways the knowledge and behavior of generative systems. Together, these areas could provide a starting point for more ambitious AI-based modeling of cultural systems of literary production. We describe the possibilities and challenges of simulation-based experiments in literary studies, summarize the current state of the art in relevant fields, and explain key technical aspects of the work. To provide an example directly relevant to literary scholars, we present the results of experiments on literary text generation, including comparisons to high-status, human-authored novels. Our results include the first demonstration of (limited) in-distribution outputs by AI models in this domain. We conclude with a description of future work on full counterfactual literary-historical simulations using AI.
>
---
#### [new 028] Learning to Retrieve: Dual-Level Long-Term Memory for Text-to-SQL Agents
- **分类: cs.CL**

- **简介: 该论文属于文本到SQL的交互任务，解决记忆检索效率低的问题。提出MERIT框架，通过多阶段记忆优化决策，提升任务成功率和效率。**

- **链接: [https://arxiv.org/pdf/2606.00547](https://arxiv.org/pdf/2606.00547)**

> **作者:** Yibo Wang; Nikki Lijing Kuang; Philip S. Yu; Zhewei Yao; Yuxiong He
>
> **摘要:** Interactive text-to-SQL agents solve database tasks through multi-turn interactions involving schema exploration, query execution, feedback interpretation, and decision revision. Long-term memory helps agents reuse past experiences, but existing retrieval methods remain limited. Static methods rely on fixed similarity heuristics that do not optimize downstream utility, while dynamic methods often learn from sparse final outcomes and retrieve memories at a single decision horizon. This is insufficient when memory usefulness changes across interaction stages, since memories useful for initial planning may differ from those needed for local, state-conditioned execution. We propose MERIT, a dynamic multi-horizon memory retrieval framework. MERIT maintains episode-level memory for global strategic guidance and turn-level memory for local decision support. Both levels use learned retrieval policies optimized with reinforcement learning. To train turn-level retrieval despite limited intermediate supervision, MERIT uses a lightweight Process Reward Model to provide dense proxy rewards for local memory selection. Experiments on BIRD-Interact show that MERIT outperforms no-memory, static-retrieval, and dynamic-retrieval baselines in success rate while reducing average interaction turns. Transfer results on Spider2-Snow further show positive cross-benchmark transfer without benchmark-specific tuning. These results suggest that multi-horizon retrieval improves experience reuse in interactive text-to-SQL agents.
>
---
#### [new 029] Benchmarking LLM-as-a-Judge for Long-Form Output Evaluation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的评价任务，旨在解决长文本生成质量评估问题。通过构建LongJudgeBench基准，评估LLM作为评判者的效果与可靠性。**

- **链接: [https://arxiv.org/pdf/2606.01629](https://arxiv.org/pdf/2606.01629)**

> **作者:** Junjie Chen; Yuxi Dong; Haitao Li; Weihang Su; Yujia Zhou; Min Zhang; Yiqun Liu; Qinyao Ai
>
> **摘要:** As large language models (LLMs) are increasingly used for long-form generation, reliably evaluating long-form outputs has become a critical challenge. LLM-as-a-judge offers a scalable alternative to human evaluation, yet its reliability in long-form output evaluation remains underexamined: existing meta-evaluation benchmarks focus mainly on short-form outputs. Compared with short-form evaluation, long-form evaluation is not merely a matter of output length; it often requires judges to handle more complex document-level demands. In this work, we introduce LongJudgeBench, a comprehensive benchmark for evaluating LLM judges on long-form outputs across diverse real-world scenarios and judging protocols. We systematically evaluate a broad range of LLM judges, covering multiple base models and judging settings. Our results reveal a substantial reliability gap: current LLM judges remain unstable across scenarios, and rubrics or references are helpful but not always sufficient. We hope LongJudgeBench will support future research on more robust, context-aware, and human-aligned LLM-as-a-judge methods. Our code is available at this https URL.
>
---
#### [new 030] EPIC: Efficient and Parallel Inference under CFG Constraints for Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型生成任务，解决CFG约束下的高效并行推理问题。提出EPIC框架，提升解码效率，减少计算开销。**

- **链接: [https://arxiv.org/pdf/2606.00722](https://arxiv.org/pdf/2606.00722)**

> **作者:** Hyundong Jin; Yo-Sub Han
>
> **摘要:** Controlling language model outputs is essential for ensuring structural validity, reliability, and downstream usability, and diffusion language models are no exception. Recent advances in diffusion language model decoding have extended output control beyond regular constraints to context-free grammar (CFG) constraints. Existing methods, however, can be up to four times slower than unconstrained decoding. More importantly, they substantially diminish one of the key advantages of diffusion language models over autoregressive models, namely parallel decoding. This slowdown arises because sequential validity checking introduces significant overhead during parallel generation. We propose an efficient CFG-constrained decoding framework, EPIC, that addresses this limitation. Our method improves decoding efficiency by combining lexing memoization, validation using Earley-style parsing instead of deterministic automata, and relaxed compatible subset selection for parallel commit. It reduces repeated lexing and validation overhead while allowing multiple compatible tokens to be committed together. Experiments on three benchmarks using four models show that our method reduces inference time by up to 67.5% and decreases the additional overhead by up to 90.5% compared with existing CFG-constrained decoding methods. Our implementation is available at this https URL .
>
---
#### [new 031] Lost in Delusion: Examining LLM Safety Under User Delusions and Distress
- **分类: cs.CL**

- **简介: 该论文属于LLM安全任务，研究用户幻觉与痛苦交织时模型的安全性。通过多轮对话实验，发现模型在幻觉框架下忽视危机，需引入专门的幻觉检测机制以提升安全性。**

- **链接: [https://arxiv.org/pdf/2606.00975](https://arxiv.org/pdf/2606.00975)**

> **作者:** Andrew Aquilina; Chetna Nihalani; Vasudha Varadarajan; Nathan S. Fishbein; Yu-Ru Lin; Maarten Sap
>
> **摘要:** LLM chatbots increasingly serve as a first source of support for people in psychological distress, including those whose distress is entangled with delusional beliefs. Prior work on LLM mental-health safety largely evaluates general therapeutic quality or single-turn crisis detection, leaving unclear how models behave when distress is intertwined with delusion over sustained conversations. We address this gap with matched multi-turn simulations, across clinically grounded personas and six LLMs, that pair each delusional conversation with a distress-only control to isolate the effect of delusional framing. This reveals a recognition-intervention gap: models detect distress at comparable rates regardless of framing, yet sharply fail to act on it once distress is embedded in delusion, with safety interventions suppressed by up to 4.5x. The failure tracks accumulated acceptance of the user's premises rather than emotional validation. Worse, the intuitive fix of prompting models to assess user distress backfires under delusional framing; only delusion-aware prompting with explicit response guidance closes the gap, and even this depends on a delusion classifier that is itself unreliable on the most vulnerable models. Safe deployment therefore requires treating delusional framing as a distinct risk signal that overrides conversational accommodation.
>
---
#### [new 032] From Outliers to Errors: Auditing Pali-to-English LLM Translations with Multi-Reference Adjudication
- **分类: cs.CL**

- **简介: 该论文属于机器翻译质量评估任务，旨在解决单一评分指标误判翻译差异为错误的问题。通过多参考基准和嵌入漂移分析，审计Pali到英语的模型翻译，识别真实错误。**

- **链接: [https://arxiv.org/pdf/2606.01136](https://arxiv.org/pdf/2606.01136)**

> **作者:** Máté Metzger; Nadnapang Phophichit; Hansa Dhammahaso
>
> **备注:** Preprint. This manuscript has not yet been peer reviewed
>
> **摘要:** Single-score translation metrics can conflate legitimate variation with error, a problem especially acute for classical languages where multiple defensible English renderings of the same passage coexist. We audit Pali-to-English output from four flagship large language models (LLMs): GPT-5.5, Claude Sonnet 4.6, Gemini 3.1 Pro, and Grok 4.3, on 1,700 passages from the Pali Canon, using three established human translations by Bhikkhu Sujato, Thanissaro Bhikkhu, and Bhikkhu Bodhi as a local reference envelope rather than a single gold standard. Each candidate's normalized embedding drift from the reference centroid serves as a triage signal, not an error label; the 1,203 candidates above a 1.5 drift threshold are then adjudicated by a blinded three-model LLM judge panel, calibrated against a 300-instance author-adjudicated validation set. Two results stand out. First, drift predicts severity rather than error per se: the major-error rate among adjudicated high-drift candidates rose monotonically from 7.9% in the 1.5-2.0 band to 51.6% above 3.0, while approximately 80% of 1.5-2.0 outliers were judged valid translation variations. Second, model differences were clearest in the high-drift tail: GPT-5.5 had the lowest adjudicated high-drift major-error rate, with confidence intervals overlapping those of Claude Sonnet 4.6 and Gemini 3.1 Pro; Grok 4.3 had both the largest outlier volume and the highest tail major-error rate (27.6% overall, 74.4% above drift 3.0). The dominant major-error categories (e.g. omission or truncation, doctrinal term errors) are precisely the failures most likely to mislead readers of doctrinal text. The contribution is a reusable audit design for classical-to-modern translation: define a local reference envelope from multiple human translators, use embedding drift to prioritize review, and adjudicate the flagged tail rather than treating outlier status as error.
>
---
#### [new 033] LLMs for Cardiovascular Risk Prediction from Structured Clinical Data
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于心血管风险预测任务，旨在利用结构化临床数据与自然语言处理技术提升CAD预测效果。通过构建混合框架，将结构化数据转为临床叙事，并评估LLM在零样本和少样本下的表现。**

- **链接: [https://arxiv.org/pdf/2606.00031](https://arxiv.org/pdf/2606.00031)**

> **作者:** Jeba Maliha; Md Rafiul Kabir
>
> **备注:** International Conference on Intelligent Systems, Blockchain, and Communication Technologies
>
> **摘要:** Coronary artery disease (CAD) remains one of the leading causes of death globally, highlighting the need for reliable predictive systems to support early diagnosis and risk assessment. While traditional machine learning models perform well on structured clinical data, large language models (LLMs) present new possibilities to interpret medical information expressed in natural language. In this work, we develop a hybrid framework that bridges structured clinical data and natural-language representations for CAD prediction. Using a publicly available dataset of 1,190 patient records with 11 clinical attributes, structured variables are converted into interpretable feature representations and synthetic clinical narratives using LLMs. A validation pipeline performs reverse extraction of clinical variables and computes a consistency score with the original records, achieving an average fidelity of 94.61%. We then evaluate four conventional machine learning models and compare their performance with LLM-based classification under zero-shot and few-shot prompting settings. We use two LLMs here, GPT and Gemini. Experimental results show that Random Forest achieves the highest accuracy. Despite this advantage, LLM-based classification remains beneficial in real-world clinical settings. This is because LLMs operate directly on natural language patient descriptions, meaning that sensitive numerical patient data such as exact lab values, blood pressure readings, and diagnostic codes are kept private. Findings suggest that combining structured clinical data with LLM-generated narratives can enable new directions for hybrid clinical prediction systems.
>
---
#### [new 034] SentGuard: Sentence-Level Streaming Guardrails for Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出SentGuard，解决大语言模型实时生成内容的安全监管问题。通过句子级流式防护，提升检测效率与准确性。**

- **链接: [https://arxiv.org/pdf/2606.02041](https://arxiv.org/pdf/2606.02041)**

> **作者:** Jiaqi Yu; Xin Wang; Yixu Wang; Jie Li; Yan Teng; Xingjun Ma; Yingchun Wang
>
> **备注:** 16 pages, 5 figures, submitted to ARR
>
> **摘要:** Large language models increasingly stream long, reasoning-intensive responses in real time, making when to moderate as critical as whether to moderate. Existing guardrails fall into two unsatisfactory extremes: response-level methods delay intervention until the full output is generated, whereas token-level methods act on incomplete semantics, often producing unstable decisions and excessive guard invocations. To address this challenge, we propose SentGuard, a sentence-level streaming guardrail that operates in parallel with generation. A lightweight waiting buffer groups streamed tokens into sentence chunks and releases only verified chunks to the user, introducing a small offset that enables SentGuard to assess the current prefix while the target LLM decodes subsequent content. To support this, we construct StreamSafe, a benchmark with structured per-sentence annotations across 8 harm categories, capturing the evolution of safety risks across both reasoning and response segments. We further train SentGuard with a coarse-to-fine objective to detect unsafe intent as soon as it emerges at sentence boundaries. Experiments on 5 safety benchmarks show that SentGuard outperforms existing baselines, detecting 90.5% of unsafe cases within two sentences while maintaining a low streaming false-positive rate of 7.41%.
>
---
#### [new 035] SN-WER: Script-Normalized WER for Multi-Script Indic ASR Evaluation
- **分类: cs.CL**

- **简介: 该论文属于自动语音识别（ASR）评估任务，解决多语言场景下因脚本差异导致的WER高估问题。提出SN-WER方法，在计算WER前将文本归一化到统一脚本，提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2606.02548](https://arxiv.org/pdf/2606.02548)**

> **作者:** Priyaranjan Pattnayak
>
> **备注:** Accepted to ACL 2026 MeLLM
>
> **摘要:** Word Error Rate (WER) is the dominant metric for automatic speech recognition (ASR), but it can overestimate errors when references and hypotheses encode the same words in different scripts. This issue is common in multilingual settings where ASR models may emit romanized text. We propose Script-Normalized WER (SN-WER), a training-free, evaluation-only scoring method that transliterates both reference and hypothesis text into a language-specific canonical script before computing WER. We evaluate SN-WER on 5 Indic languages, 2 datasets, and 3 ASR models. On curated FLEURS data, SN-WER reduces inflated model gaps by up to 12%, while on noisier Common Voice data the reductions are smaller or inconsistent, indicating genuine recognition weaknesses rather than only script mismatch. Controlled stress tests show a 67% attenuation of artificial romanization-induced WER inflation, while lexical-substitution controls show near-identical sensitivity to semantic errors, with Delta SN-WER / Delta WER approximately 1.09. SN-WER is robust to transliterator choice, normalization changes, and shows low token-collision rates below 0.1% in the evaluated Indic setting. We argue that SN-WER should be reported alongside WER and CER as a companion metric for script-insensitive ASR evaluation, especially when transcripts feed downstream search, indexing, or multilingual LLM pipelines.
>
---
#### [new 036] Revisiting Parameter-Based Knowledge Editing in Large Language Models: Theoretical Limits and Empirical Evidence
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识编辑任务，旨在解决参数编辑方法在大语言模型中的局限性。研究揭示了参数编辑导致模型能力下降的理论原因，并通过实验验证检索方法更优。**

- **链接: [https://arxiv.org/pdf/2606.00570](https://arxiv.org/pdf/2606.00570)**

> **作者:** Wanying Ren; Xin Song; Futing Wang; Guoxiu He; Aixin Sun
>
> **备注:** Accepted to ICML 2026. Equal contribution by the first two authors. 9 pages main paper, 10 figures, with appendix
>
> **摘要:** Parameter-based knowledge editing updates the internal knowledge of large language models (LLMs) via localized weight modifications and has attracted significant attention. However, most existing methods overlook fundamental theoretical limitations and are rarely evaluated under realistic, practice-oriented settings. In this paper, we first present a theoretical analysis based on the dimensional Collapse Hypothesis, explaining how localized parameter edits can propagate along fragile directions in the representation space, inducing global interference and ultimately causing reasoning collapse. Building on this insight, we conduct a comprehensive empirical evaluation by systematically varying knowledge complexity, number of edits, evaluation dimensions, and baseline methods. Our results show that parameter-based editing methods consistently damage core LLM capabilities. In contrast, a simple retrieval-based baseline achieves consistently stronger performance than all parameter-editing methods across all evaluated conditions. These findings highlight that preserving the fundamental capabilities of LLMs after knowledge editing should be a central concern for future research.
>
---
#### [new 037] Investigating and Alleviating Harm Amplification in LLM Interactions
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于语言模型安全任务，旨在解决多轮对话中危害放大的问题。通过构建基准数据集和提出监控方法TrajSafe，有效降低危害风险。**

- **链接: [https://arxiv.org/pdf/2606.02423](https://arxiv.org/pdf/2606.02423)**

> **作者:** Ruohao Guo; Wei Xu; Alan Ritter
>
> **摘要:** Large language models (LLMs) can serve as helpful assistants, yet they can equally function as harm amplifiers that enable malicious users to achieve harmful outcomes beyond their capabilities through extended interactions. This risk manifests along two axes, i.e., democratizing domain expertise that allows novices to produce specialized harmful content, and scaling harmful operations at volumes that manual effort cannot match. Existing works, however, often overlook how LLMs compound harm across multi-turn conversations. We introduce HarmAmp, a new benchmark for multi-turn harm amplification scenarios spanning twelve risk categories. Each scenario is grounded in real-world threats and satisfies rigorous criteria, i.e., substantive amplification, operational specificity, and multi-turn necessity. We further propose TrajSafe, a proactive monitor that anticipates harmful trajectories and intervenes through actions such as probing users' genuine intents and steering the models towards safer completion. Our extensive experiments demonstrate that TrajSafe significantly reduces the harmfulness incurred in multi-turn interactions while preserving a low over-refusal rate and the target model's general capabilities. Our work offers a promising paradigm to alleviate the nuanced safety risks in LLM interactions.
>
---
#### [new 038] Cross-lingual Self-Consistency for Multilingual Reasoning with Language Models
- **分类: cs.CL**

- **简介: 该论文属于多语言推理任务，旨在提升大语言模型在低资源语言中的推理能力。通过无监督强化学习，增强跨语言自一致性，无需标注数据即可提高模型在多种语言上的表现。**

- **链接: [https://arxiv.org/pdf/2606.01464](https://arxiv.org/pdf/2606.01464)**

> **作者:** Ahmed Elhady; Eneko Agirre; Mikel Artetxe
>
> **备注:** Paper under review
>
> **摘要:** Despite expanding their multilingual coverage, the advanced reasoning capabilities of LLMs remain largely confined to a few high-resource languages like English. To address this, we propose an unsupervised Reinforcement Learning (RL) approach to enhance multilingual reasoning by enforcing cross-lingual self-consistency: the principle that a model should produce the same final answer for equivalent problems in different languages. Existing methods are limited by the scarcity of multilingual reasoning data and show weak generalization to unseen languages. Our approach requires neither gold answers nor parallel data, and it achieves average gains of up to 21.7% on MGSM across 10 languages. In addition, our method demonstrates strong generalization, with an 18.2% mean improvement on MGSM languages unseen during training, and up to 6.2% gain on 3 out-of-distribution benchmarks. These results show the potential of consistency-based methods to improve the multilingual capabilities of LLMs without requiring supervised data.
>
---
#### [new 039] Geometric Latent Reasoning Induces Shorter Generations in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型推理效率低的问题。通过引入几何潜在推理，用连续路径替代显式推理链，实现更短的生成结果。**

- **链接: [https://arxiv.org/pdf/2606.02248](https://arxiv.org/pdf/2606.02248)**

> **作者:** Shashi Kumar; Yacouba Kaloga; Petr Motlicek; Ina Kodrasi; Andrea Cavallaro
>
> **摘要:** Large language models solve complex problems by generating lengthy chains of explicit reasoning tokens. While effective, this makes reasoning expensive, length-sensitive, and constrained to (discrete) natural language. While latent reasoning offers a continuous alternative, determining useful structures for intermediate latent states is an open challenge. In this paper, we formulate latent reasoning as a geometric path-approximation problem within the model's pretrained token-embedding space. We introduce Geometric Latent Reasoning (GLR), which uses a lightweight transition head to predict iterative direction updates in embedding space. Using textual chain-of-thought traces as anchors, GLR learns to approximate discrete reasoning trajectories while permitting continuous deviations from exact token embeddings. Evaluations on mathematical reasoning benchmarks using Qwen3 models reveal an emergent phenomenon: geometric latent reasoning induces substantially shorter generations without an explicit length objective. By replacing early explicit reasoning with continuous latent steps, models often reach correct answers using substantially fewer total generation steps. These findings suggest that continuous trajectories act as compact intermediate reasoning states, exposing a new tradeoff between latent computation budget, output length, and accuracy.
>
---
#### [new 040] AEyeDE: An Attention-Based Attribution Framework for AI-Generated Text Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI生成文本检测任务，旨在解决传统方法难以识别高度仿真的AI文本问题。提出AEyeDE框架，利用模型注意力作为区分信号进行检测。**

- **链接: [https://arxiv.org/pdf/2606.00016](https://arxiv.org/pdf/2606.00016)**

> **作者:** Aria Nourbakhsh; Adelaide Danilov; Christoph Schommer; Salima Lamsiyah
>
> **备注:** 24 pages, 2 figures
>
> **摘要:** Detecting AI-generated text is becoming increasingly challenging as modern language models approach human-level fluency and can evade detectors that rely on surface statistics or likelihood-based signals. We propose \textsc{AEyeDE}, an attribution-driven approach to human-AI authorship detection that leverages model attention as a discriminative signal. Specifically, we extract attention-based attribution matrices for both human- and AI-generated text using a \emph{proxy} Transformer model with white-box access and train a lightweight Convolutional Neural Network to learn representations from these attribution maps. Across encoder-decoder translation settings, our method consistently outperforms a text-only baseline. In decoder-only settings, it performs strongly in generator-specific detection, remains competitive on standard benchmarks, and shows robustness under cross-dataset transfer and alternative-spelling perturbations. We further show that attention maps exhibit recurring local structures whose relative frequencies differ consistently between human- and AI-generated text across datasets and proxy models. These findings suggest that attention-based attribution maps provide a complementary and interpretable signal for AI-generated text detection. We will make the code publicly available to support future research.
>
---
#### [new 041] UniD$^3$: A Knowledge Graph-Enhanced RAG Framework for Drug-Disease Discovery and Reasoning
- **分类: cs.CL**

- **简介: 该论文提出UniD$^3$框架，解决药物-疾病关系抽取与验证问题，整合大模型与知识图谱，提升药物发现和精准医疗的效率与准确性。**

- **链接: [https://arxiv.org/pdf/2606.01394](https://arxiv.org/pdf/2606.01394)**

> **作者:** Qing Wang; Tianshi Liu; Minghao Zhou; Jialu Liang; Sen Guo; Guangyu Wang; Jing Su; Qianqian Song
>
> **摘要:** Systematic characterization of drug-disease relationships is essential for drug discovery and repurposing, yet is hindered by the heterogeneity and rapid growth of biomedical literature. Existing datasets rely on labor-intensive curation and are often incomplete, while LLM-only approaches suffer from hallucination and weak evidence grounding. We introduce UniD$^3$, a unified framework that integrates Large Language Models with Knowledge Graph-enhanced Retrieval-Augmented Generation (KG-RAG) to extract, organize, and validate drug-disease knowledge across Drug-Disease Matching (DDM), Drug Effectiveness Assessment (DEA), and Drug-Target Analysis (DTA). UniD$^3$ processes 157,849 PubMed articles with Llama 3.3-70B and constructs knowledge graphs via a dual-stage strategy combining paper-level extraction with KG-level consolidation centered on drug and disease entities. These graphs support KG-RAG-based generation of structured datasets, evaluated through external benchmarks, fuzzy matching with curated resources, and clinician review. UniD$^3$ produces six knowledge graphs and large-scale datasets, including 28,915 DDM, 15,042 DEA, and over 4,000 DTA QA pairs. External validation shows strong performance (F1: 0.85-0.87 for DDM/DEA; 0.82 for DTA), with clinician review confirming high reliability (AUROC = 0.90). KG-RAG-augmented models outperform standalone LLMs, and the UniD$^3$ chatbot enables interpretable, citation-supported exploration of drug-disease relationships. UniD$^3$ provides a scalable, extensible framework for transforming unstructured biomedical literature into high-quality, structured drug-disease knowledge, supporting AI-driven discovery, repurposing, and precision medicine.
>
---
#### [new 042] Don't Read Everything: A Curvature-Conditioned Query for Linear Attention
- **分类: cs.CL; cs.LG**

- **简介: 该论文针对线性注意力在上下文检索和长序列任务中的性能不足，提出Curvature-Conditioned Query（CCQ）机制，通过优化读取步骤提升效果。**

- **链接: [https://arxiv.org/pdf/2606.01294](https://arxiv.org/pdf/2606.01294)**

> **作者:** Dong Le; Thong Nguyen; Cong-Duy Nguyen; Anh Tuan Luu
>
> **备注:** 19 pages
>
> **摘要:** Linear attention reduces the quadratic cost of softmax attention by maintaining a recurrent fast-weight state, but it consistently lags on in-context retrieval and long-context tasks. Existing remedies act on the write side of memory through gating, delta updates, or kernel feature maps, but the read step is left unchanged: every past key contributes additively to the output, so useful targets are diluted by the bulk of stored vectors. We borrow one specific piece of softmax's geometry to construct a cheap read-time contraction of the query. A second-order Taylor expansion of the softmax log-partition at the isotropic-attention point gives a local quadratic model whose curvature coincides with the running key covariance, a quantity that can be maintained with the same recurrent/chunkwise mechanism as the linear-attention state. The associated linear operator contracts the query along the high-density directions of memory before it reads the state. We call this mechanism Curvature-Conditioned Query (CCQ). CCQ modifies only the read step and is composable with any linear-attention backbone. Attached to GLA and Gated DeltaNet, it improves perplexity, zero-shot downstream accuracy, S-NIAH retrieval at and beyond the training context, length-extrapolation perplexity from 4K to 20K, and LongBench accuracy, at small extra cost.
>
---
#### [new 043] I-WebGenBench : Evaluating Interactivity in LLM-Generated Scientific Web Applications
- **分类: cs.CL**

- **简介: 该论文属于科学论文交互系统生成任务，旨在解决静态论文无法体现动态机制的问题，提出I-WebGenBench和PaperVoyager，生成可交互的网页系统。**

- **链接: [https://arxiv.org/pdf/2606.00750](https://arxiv.org/pdf/2606.00750)**

> **作者:** Dasen Dai; Biao Wu; Meng Fang; Shuoqi Li; Wenhao Wang
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** Recent advances in visual language models have enabled autonomous agents for complex reasoning, tool use, and document understanding. However, existing document agents mainly transform papers into static artifacts such as summaries, webpages, or slides, which are insufficient for technical papers involving dynamic mechanisms and state transitions. In this work, we propose a Paper-to-Interactive-System Agent that converts research papers into executable interactive web systems. Given a PDF paper, the agent performs end-to-end processing without human intervention, including paper understanding, system modeling, and interactive webpage synthesis, enabling users to manipulate inputs and observe dynamic behaviors. To evaluate this task, we introduce a benchmark of 19 research papers paired with expert-built interactive systems as ground truth. We further propose PaperVoyager, a structured generation framework that explicitly models mechanisms and interaction logic during synthesis. Experiments show that PaperVoyager significantly improves the quality of generated interactive systems, offering a new paradigm for interactive scientific paper understanding.
>
---
#### [new 044] PolySpeech-100: A Large-Scale Benchmark for Speech Understanding Across 100+ Languages and Dialects
- **分类: cs.CL; cs.AI; eess.AS**

- **简介: 该论文提出PolySpeech-100，解决多语言与方言语音理解问题，通过混合构建方法覆盖110种语言变体，评估模型在不同场景下的表现。**

- **链接: [https://arxiv.org/pdf/2606.01016](https://arxiv.org/pdf/2606.01016)**

> **作者:** Sicheng Yang; Shulan Ruan; Shiwei Wu; Yu Liu; Lu Fan; Zhi Li; You He
>
> **备注:** 19 pages, 13 figures, KDD 2026
>
> **摘要:** While End-to-End (E2E) Speech-Large Language Models (Speech-LLMs) are rapidly evolving, their evaluation methodologies remain limited to the era of simple transcription. Existing benchmarks suffer from three critical limitations: a pronounced bias towards high-resource languages, a focus on low-level recognition (ASR) rather than semantic reasoning, and a neglect of regional dialects. To bridge this gap, we introduce PolySpeech-100, a massive-scale benchmark designed to assess `native-level' speech comprehension across 110 linguistic variants. We employ a novel hybrid construction pipeline that augments gold-standard human recordings with instruction-driven synthetic speech, allowing us to cover 19 distinct Chinese dialects and over 80 low-resource languages. Extensive evaluation of 22 state-of-the-art models (including Gemini-3, GPT-Audio, and Qwen2.5-Omni) yields pivotal insights. First, we demonstrate that open-source E2E models outperform Cascade (ASR+LLM) systems on heavy dialects, proving that direct audio processing preserves critical paralinguistic cues and prosodic features (e.g., intonation, stress) that are often lost in standard transcription. Second, we reveal a significant performance gap: while commercial models maintain robustness, open-source models suffer catastrophic degradation on low-resource languages. Finally, counter-intuitively, we observe that under standard zero-shot settings, Chain-of-Thought prompting frequently degrades speech understanding performance for most evaluated models, revealing a potential modality alignment gap in current architectures. PolySpeech-100 establishes a rigorous standard for the next generation of inclusive, omni-capable Speech-LLMs. The data, demo, and code are publicly available at this https URL.
>
---
#### [new 045] LaSR: Context-Aware Speech Recognition via Latent Reasoning
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，旨在提升模型对语境和专业术语的理解。提出LaSR方法，通过潜在推理增强上下文感知，提升术语识别效果。**

- **链接: [https://arxiv.org/pdf/2606.00507](https://arxiv.org/pdf/2606.00507)**

> **作者:** Heyang Liu; Ziyang Cheng; Jiayi Huang; Wenyang Xiao; Ronghua Wu; Qunshan Gu; Yanfeng Wang; Yu Wang
>
> **摘要:** Recent advances in Speech Large Language Models (Speech LLMs) have significantly enhanced spoken language understanding and reasoning. However, their contextual awareness is limited, struggling to perform speech recognition that effectively reflects the speaker's intent and topical context. In this paper, we propose LaSR (Latent Speech Reasoning), a novel training paradigm featuring a context-aware reasoning trajectory that leverages the latent reasoning process. Instead of generating explicit intermediate tokens, LaSR aligns chain-of-thought (CoT) supervision around the acoustic feature region of the targeted word, and introduces latent reasoning periods for context information grounding and transcriptional transition. Furthermore, to effectively benchmark contextual recognition on specialized vocabulary, we propose Spoken Darwin-Science, a large-scale corpus focusing on academic terminologies. Preliminary experiments on Fun-Audio-Chat demonstrate that LaSR significantly improves terminology recognition without introducing additional latency and consistently outperforms standard supervised fine-tuning baselines. Our findings highlight the potential of latent reasoning in building efficient, context-aware speech assistants.
>
---
#### [new 046] THRD: A Training-Free Multi-Turn Defense Framework for Jailbreak Attacks on Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型安全防护任务，解决多轮越狱攻击问题。提出THRD框架，通过时间风险累积机制提升防御效果，同时保持模型性能。**

- **链接: [https://arxiv.org/pdf/2606.01738](https://arxiv.org/pdf/2606.01738)**

> **作者:** Zhiqing Ma; Zhonghao Xu; Dong Yu; Chen Kang; Changliang Li; Pengyuan Liu
>
> **摘要:** Multi-turn jailbreak attacks pose a growing threat to LLMs by exploiting conversational dynamics such as gradual escalation and cross-turn coordination. Existing defenses either rely on costly retraining -- often degrading model utility -- or apply single-turn analysis independently at each turn, failing to capture how risk accumulates along interaction trajectories. We observe that safety behavior in multi-turn interaction is trajectory-dependent: dialogue history continuously reshapes the model's conditioning context, making it insufficient to evaluate each turn in isolation. Motivated by this insight, we present THRD, the first training-free framework that explicitly models temporal risk accumulation for multi-turn jailbreak defense. THRD integrates four modules: a Turn-level Risk Assessor (TRA) for instantaneous risk estimation, a Historical Context Analyzer (HCA) for cross-turn intent escalation detection, a Response Evaluator (RE) for identifying facilitative outputs, and a Decision Module that combines these signals through a time-evolving scoring mechanism with attenuation-based modulation and trend-aware adjustment. Experiments against state-of-the-art multi-turn attacks -- including tree-search-based and multi-agent collaborative methods -- across two target models show that THRD reduces ASR to 0.2--4.0% while preserving model utility within 1.5% degradation on MMLU and GSM8K. Ablation studies confirm non-redundant module contributions and stable cross-architecture generalization. Analysis of first rejection triggers reveals that over 70% of multi-turn attacks require Turn~2 or later to detect, validating the necessity of explicit temporal aggregation.
>
---
#### [new 047] Isolating LLM Lexical Bias: A Curation-Free Triangulated Metric for Preference-Stage Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，解决大语言模型在偏好学习阶段产生的词汇偏见问题。提出一种无需人工标注的度量方法，用于量化偏好调优带来的行为变化。**

- **链接: [https://arxiv.org/pdf/2606.00334](https://arxiv.org/pdf/2606.00334)**

> **作者:** Xiaoyang Ming; Jose Hernandez; Thomas Stephan Juzek
>
> **备注:** 7 pages, 2 figures, 1 table
>
> **摘要:** Various language domains have undergone remarkable changes in recent years; these shifts are largely attributed to the advent of Large Language Models and their misalignment with natural language usage. These misalignments are thought to partly originate in the preference-learning stage, e.g. Reinforcement Learning from Human Feedback, which generally makes models more useful but simultaneously may introduce systematic lexical bias. In terms of lexical behavior, this is visible in a model's preference for certain formats or the overuse of words (delve, furthermore), even when such patterns are not present in base model outputs. Research on lexical misalignment induced during preference training is constrained by reliance on manual curation. We address this, by introducing the Triangulated Preference Shift score, a metric that triangulates between human gold standards, base models, and instruct variants to isolate shifts induced specifically by preference learning, without manual curation. We provide data across six model families, anchor the results in the literature, and illustrate the general approach's utility by analyzing whether preference learning shifts models toward what could be interpreted as a "language of prestige". The metric provides an initial automated method to quantify behavioral shifts attributable to preference tuning, and thus, may help inform model alignment and development of trustworthy AI.
>
---
#### [new 048] Not What, But How: A Communicative Audit of LLM Response Framing
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的评估任务，旨在解决LLM回答框架的评价问题。提出FRANZ框架，从四个维度审计LLM响应，揭示其文化定位与拟人化特征的关联性。**

- **链接: [https://arxiv.org/pdf/2606.02493](https://arxiv.org/pdf/2606.02493)**

> **作者:** Siddhesh Milind Pawar; Sarah Masud; Haneul Yoo; Alice Oh; Isabelle Augenstein
>
> **备注:** 34 main pages, 19 Figures, 4 Tables
>
> **摘要:** Large language models (LLMs) are being increasingly used to answer subjective, information-seeking questions, where users are sensitive to how responses are communicated, not just whether the answers are correct. Existing LLM evaluations for subjective cultural queries largely focus on factual correctness, ignoring how the response is framed. To this end, we introduce FRANZ, an automated FRAmework for respoNse characteriZation to conduct communicative audit of LLM responses along four dimensions: cultural positioning, use of generalizing language, anthropomorphic cues, and adherence to conversational maxims. To enable this evaluation, we contribute SQUARE - a corpus of 376k subjective questions sourced from 57 subreddits, and mapped to 7 countries and 19 question categories. We demonstrate FRANZ's applicability by scoring responses from three open-weight LLMs. We observe that LLMs show statistically significant differences in the frequency with which they employ each response characteristic. Unlike single-dimensional audits, FRANZ reveals that insider positioning and anthropomorphism are positively coupled, with the degree of coupling varying by country, providing a diagnostic lens for identifying framing divergences.
>
---
#### [new 049] Benchmarking Local LLMs for Natural-Language-to-SQL Querying in Biopharmaceutical Manufacturing: An Empirical Benchmark on Consumer-Grade Hardware
- **分类: cs.CL**

- **简介: 该论文研究本地大语言模型在生物制药制造中的自然语言到SQL查询生成任务，评估其在受监管环境下的适用性与性能。**

- **链接: [https://arxiv.org/pdf/2606.01338](https://arxiv.org/pdf/2606.01338)**

> **作者:** Sagar Bhetwal; Rajan Bastakoti; Nirajan Acharya; Gaurav Kumar Gupta
>
> **摘要:** Biopharmaceutical manufacturing organizations operate under regulatory frameworks such as FDA guidance, EU Good Manufacturing Practice (GMP), and the EU AI Act, which can restrict the use of cloud-based artificial intelligence systems. Locally deployed large language models (LLMs) offer a privacy-preserving alternative, but their suitability for pharmaceutical manufacturing tasks remains underexplored. This study evaluates four open-source LLMs (Qwen 2.5 Coder 7B, Llama 3.1 8B, Mistral 7B, and Meditron 7B) deployed locally via Ollama for natural-language-to-SQL generation over a pharmaceutical manufacturing database. A FastAPI-based evaluation platform, PharmaBatchDB AI, was developed using a synthetic Microsoft SQL Server database containing approximately 63,000 records across Batch, Manufacturing Execution System (MES), and Clean-In-Place (CIP) modules. Models were benchmarked on 60 domain-specific natural-language questions using metrics including SQL extraction rate, SQL compliance, factual consistency, ROUGE-L, hallucination rate, throughput, and latency. Qwen 2.5 Coder 7B, Llama 3.1 8B, and Mistral 7B generated SQL for all evaluation tasks, while Meditron 7B failed on nearly all tasks due to context-window limitations and poor SQL generation capability. Llama 3.1 8B achieved the highest SQL compliance, whereas Qwen 2.5 Coder 7B achieved the strongest overall text similarity and factual consistency. Performance differences between the two leading models were not statistically significant. The results show that code-tuned general-purpose LLMs outperform a domain-specific biomedical model on structured query generation for pharmaceutical manufacturing data. Although fully local, GxP-aligned NLQ systems are feasible on consumer hardware, current performance levels still require human oversight and downstream validation for regulated use.
>
---
#### [new 050] Short-form Text Rewriting with Phi Silica
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究短文本重写任务，旨在提升小语言模型的语义准确性和减少幻觉。通过数据集构建、提示蒸馏和微调等方法进行优化。**

- **链接: [https://arxiv.org/pdf/2606.00462](https://arxiv.org/pdf/2606.00462)**

> **作者:** Divya Tadimeti; Shawn Pan; Sameera Lanka; Chenghui Zhou; Sadid Hasan
>
> **备注:** 6 pages
>
> **摘要:** Short-form text rewriting is a constrained variant of paraphrasing in which limited context and high semantic density leave little room for variation. While large language models perform well on general paraphrasing, small language models (SLMs) often struggle with semantic fidelity and hallucination robustness in short-form settings. In this work, we present an empirical study of adapting an SLM, Phi Silica, for short-form rewrite through dataset curation, prompt distillation, parameter-efficient fine-tuning, and evaluation. We curate a dataset of short presentation-style text from public slide decks and use GPT-5-chat both to generate rewrite supervision and to conduct LLM-as-a-judge evaluation. Our results show that finetuning improves semantic fidelity, reduces hallucinations, and increases preference win rate against GPT-5-chat rewrites. The findings suggest that targeted adaptation for SLMs can substantially narrow the gap to cloud models and provide practical guidance for adapting SLMs to precision-critical rewrite tasks.
>
---
#### [new 051] Learning from Saturated Data: Signals Beyond Correctness for LLM Training
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理领域，研究如何在数据饱和情况下提升大语言模型性能。通过引入细粒度质量信号（如自评和熵值）替代传统正确性判断，探索更有效的训练方法。**

- **链接: [https://arxiv.org/pdf/2606.01436](https://arxiv.org/pdf/2606.01436)**

> **作者:** Hanno Hiss; Jasper Dekoninck; Martin Vechev
>
> **备注:** 25 pages, 5 figures
>
> **摘要:** The growing capabilities of large language models (LLMs) have led to the saturation of many benchmarks and training datasets used to improve them. Motivated by this, we investigate whether questions solved with perfect empirical accuracy can nevertheless be used to improve downstream performance. To do so, we replace binary correctness with two sources of more fine-grained quality signals: (1) pairwise LLM self-judgments, in which the model evaluates the relative quality of its own solutions, and (2) token-level entropy, where token-level uncertainty is used as a proxy for solution quality. We incorporate these signals into several training algorithms and evaluate them on Qwen3-1.7B-Base. When training exclusively on a simple arithmetic task, quality-based signals improve performance by up to $18.6\%$ over the base model, substantially outperforming SFT. On GSM8K, however, gains are more modest and depend strongly on the quality signal. For instance, self-judgments show poor agreement with a stronger external judge and can even degrade performance below the base model. Overall, our results suggest that quality-based training can extract useful signal from saturated questions for base models, but that applying such signals to more complex tasks requires careful calibration and further study.
>
---
#### [new 052] Peacemaker at ATE-IT: Automatic term extraction from Italian text for waste management data using encoder model
- **分类: cs.CL**

- **简介: 该论文属于自动术语提取任务，旨在解决低资源环境下术语提取的准确性问题。通过微调策略，提出一种低成本、可解释的方法，并在Ate共享任务中验证其有效性。**

- **链接: [https://arxiv.org/pdf/2606.01469](https://arxiv.org/pdf/2606.01469)**

> **作者:** Mahdi Bakhtiyarzadeh; Hadi Bayrami Asl Tekanlou; Jafar Razmara
>
> **备注:** 9 pages, 2 figures, Published in EVALITA 2026, CEUR Workshop Proceedings Vol. 4195
>
> **摘要:** The development of automatic term extraction has become increasingly important in modern technology. Automatic term extraction can be found in virtually every search engine that is currently available to users. Recent advancements have provided promising results for the extraction of automatic terms; however, accurate labeling is difficult because of several factors, such as the limited number of annotated documents available for training and the complexity of extracting multi-word expressions due to shifts in the domain. In this paper, we will present a low-cost and interpretable method of automatic term extraction, developed specifically for Task A of the ATE Shared Task. This new method utilizes fine-tuning extraction strategies that can run on a small amount of computational resources. We evaluated our automated system using both type-level and micro-level measures of precision, recall, and F1-score to measure both complementary aspects of the extraction performance. According to the experimental results, our proposed approach achieves consistent and balanced performance compared to other teams. Even though the technique itself is relatively straightforward, it serves as a good starting point for low-resource models. Overall, the findings point toward the possibility of significant future advancements (in model expansion) with higher-level performance still able to retain their ability to be interpreted.
>
---
#### [new 053] Robust Asynchronous Planning via Auto-Formalization
- **分类: cs.CL**

- **简介: 该论文属于异步规划任务，解决现实任务中异步、并发和时间约束的问题。通过引入三个基准，对比不同形式化方法的性能，提出改进的约束求解策略。**

- **链接: [https://arxiv.org/pdf/2606.00981](https://arxiv.org/pdf/2606.00981)**

> **作者:** Jiayi Zhang; Jianing Yin; Ben Zhou; Li Zhang
>
> **摘要:** LLMs can plan by either generating action sequences directly as a Planner or translating tasks into domain specific language for an external solver as a Formalizer. While most real-world tasks are asynchronous with non-uniform durations, concurrency, and execution-time constraints, existing benchmarks hardly cover them. We unify these asynchronous planning challenges under a single formulation and introduce the first three benchmarks that address each at scale. We conclude that the choice of formal representation primarily determines whether planning scales: as dependency graphs grow from 5 to 100 actions, Planner collapses from 96% to 5% plan accuracy and PDDL2.1 Formalizer from 13% to 0%, while CP-SAT Formalizer averages 94% and still achieves 83% at 100 actions. Faithfulness diagnostics show that PDDL2.1's predicate-based planning representation becomes brittle compared to general constraint satisfaction programs, when LLMs must keep predicates, effects, and goals consistent. Execution-time updates of planning constraints further degrade performance sharply (Planner 23.9%, PDDL2.1 0.7%, CP-SAT 46.1%), but a state-aware repair strategy that updates only event-induced constraints recovers CP-SAT Formalizer to 84.5%.
>
---
#### [new 054] MiCU: End-to-End Smart Home Command Understanding with Large Language Model
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于智能家庭命令理解任务，旨在解决模糊或错位指令识别问题。通过构建领域专用大模型MiCU，结合课程学习和强化学习提升性能，并优化推理效率。**

- **链接: [https://arxiv.org/pdf/2606.01099](https://arxiv.org/pdf/2606.01099)**

> **作者:** Haowei Han; Kexin Hu; Weiwei Cai; Debiao Zhang; Bin Qin; Yuxiang Wang; Jiawei Jiang; Xiao Yan; Bo Du
>
> **摘要:** Command understanding systems in smart home ecosystems can automate device control and substantially improve user experience. However, while they perform well on precise utterances (e.g., "turn on the bedroom light"), they struggle with ambiguous or misaligned commands (e.g., "make the bedroom cozy"). Large language models (LLMs) generalize well across various domains and can outperform traditional rule-based systems on such tasks, but their effectiveness is often constrained by scarce domain-specific data, insufficient task-specific adaptation, and high computational costs. In this paper, we propose an automated training data synthesis workflow using user logs and LLMs; then we build MiCU, a domain-specific LLM that excels at command understanding. Specifically, we employ curriculum learning to inject domain knowledge into the base LLM, then we enhance its reasoning ability via cold-start training combined with reinforcement learning (RL) guided by domain-specific thinking rules. Additionally, we introduce a token compression technique that condenses device description into a single special token, substantially reducing inference overhead and enabling \model-fast, an efficient variant optimized for long inputs. Extensive experiments show that MiCU significantly outperforms baselines, with an average accuracy gain of 20.01% across all device categories. We have deployed MiCU in the Xiaomi Home app, receiving approximately 1.7 million page views per day. Production evaluations show that MiCU reduces user correction rate by 1.57% and increases human audited accuracy by 32.05%. Our data and code are available at this https URL
>
---
#### [new 055] MemPro: Agentic Memory Systems as Evolvable Programs
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出MemPro，解决长期自主代理的内存系统演化问题，通过将整个记忆流程视为可进化程序，提升系统适应性和性能。**

- **链接: [https://arxiv.org/pdf/2606.00619](https://arxiv.org/pdf/2606.00619)**

> **作者:** Qingshan Liu; Guoqing Wang; Wen Wu; Jingqi Huang; Xinqi Tao; Dejia Song; Jie Zhou; Liang He
>
> **备注:** 20 pages, 14 figures
>
> **摘要:** Long-horizon autonomous agents require memory systems to retain historical information, track evolving states, and reuse relevant knowledge beyond finite context windows. Existing agentic memory systems typically follow a memory construction-retrieval (MCR) pipeline, but often adapt mainly the memory bank while keeping the surrounding pipeline fixed after deployment. This fixed-pipeline design struggles to handle heterogeneous task-specific failure modes and can become misaligned with memory banks that evolve in scale and structure over time. To address these limitations, we propose MemPro, a system-level evolution framework that treats the entire MCR pipeline as an evolvable program rather than adapting only the memory bank or prompt text. MemPro maintains a version tree of runnable memory-system implementations, where an Evolving Agent iteratively selects promising versions, diagnoses recurring failures, and creates improved child versions through failure-mode-guided edit-debug refinement. Experiments on LongMemEval, LoCoMo, HotpotQA, and NarrativeQA show that MemPro consistently outperforms strong static and prompt-level evolving baselines within a few iterations, continues to improve with evolution, and achieves a favorable performance-cost trade-off. Code is available at this https URL.
>
---
#### [new 056] Agentic Clustering: Controllable Text Taxonomies via Multi-Agent Refinement
- **分类: cs.CL**

- **简介: 该论文属于文本聚类任务，旨在解决传统方法依赖固定流程、适应性差的问题。通过多智能体协作动态调整聚类过程，提升效果与灵活性。**

- **链接: [https://arxiv.org/pdf/2606.01255](https://arxiv.org/pdf/2606.01255)**

> **作者:** Simon Löwe; Emily Silcock
>
> **摘要:** Recent text-clustering methods use large language models to propose a cluster taxonomy from a corpus and then assign each text to it. These pipelines are fundamentally programmatic: the sequence of LLM calls and the rules for stopping, merging, and splitting clusters are fixed in code in advance, so they generalise poorly across corpora of different structure and cannot easily incorporate user-supplied constraints such as a target cluster count or a clustering intent. We propose an agentic alternative in which an orchestrator LLM inspects the state of the discovery process at each step and dispatches one of a small set of specialised agents - proposer, synthesizer, auditor, investigator, and critic - adapting the pipeline to the corpus rather than executing a fixed one. On seven public text-clustering benchmarks the method achieves state-of-the-art performance, beating the strongest prior LLM baseline by up to 32% in ARI.
>
---
#### [new 057] HypothesisMed: Inference-Time Answer Fusion and Structured Hypothesis-Space Reporting for Biomedical Question Answering
- **分类: cs.CL**

- **简介: 该论文属于生物医学问答任务，解决模型输出可靠性问题。提出HypothesisMed框架，在推理阶段融合答案并生成结构化可靠性报告，提升模型的可解释性和准确性。**

- **链接: [https://arxiv.org/pdf/2606.00971](https://arxiv.org/pdf/2606.00971)**

> **作者:** Md Motaleb Hossen Manik; Ge Wang
>
> **摘要:** Biomedical question answering with large language models is commonly evaluated using answer accuracy, but answer accuracy alone does not indicate whether a model can produce parseable outputs, follow structured reliability instructions, recognize weak answer spaces, or avoid confident incorrect commitments. This paper presents HypothesisMed, an inference-time reliability pipeline for biomedical multiple-choice question answering. It combines direct, chain-of-thought, HypothesisMed-v3 prompting, and answer fusion. The final answer is selected by fusion, while HypothesisMed-v3 supplies SPACE labels and confidence information. SPACE labels mark the answer space as VALID, INCOMPLETE, or CONTRADICTED. We evaluate Qwen2.5-7B, Phi-4-mini, DeepSeek-R1-32B, and BioMistral-7B on MedQA, MedMCQA, and PubMedQA using 1,000 examples per dataset. The pipeline improves weighted accuracy over each model's best direct or chain-of-thought baseline while increasing parse and SPACE coverage. We also scale evaluation to Qwen2.5-7B and Phi-4-mini using 10,183 examples per model. Fusion improves Phi-4-mini accuracy from 0.4296 to 0.5192, while Qwen2.5-7B chain-of-thought remains slightly higher in answer accuracy. However, Qwen2.5-7B fusion achieves complete parse and SPACE coverage with much lower false commitment. A 12,000-example SPACE stress test shows answer-space diagnosis remains difficult, with SPACE accuracy of 0.3074 for Qwen2.5-7B and 0.4168 for Phi-4-mini. These results show that answer accuracy, parseability, structured reliability reporting, calibration behavior, and false-commitment behavior are separable capabilities. The main contribution is not a universal state-of-the-art claim, but a reproducible inference-time framework for evaluating biomedical question answering models as auditable workflow components under structured reliability constraints.
>
---
#### [new 058] LongAttnComp: Cross-Family Context Compression for Long-Context Reasoning
- **分类: cs.CL**

- **简介: 该论文针对长文本推理任务，解决上下文长度与推理效率的矛盾。提出LongAttnComp方法，通过优化注意力机制和分块处理，提升长文本处理效果。**

- **链接: [https://arxiv.org/pdf/2606.01336](https://arxiv.org/pdf/2606.01336)**

> **作者:** Mengmeng Ji; Ravi Shanker Raju; Jonathan Lingjie Li; Chen Wu
>
> **备注:** Under review
>
> **摘要:** As real-world applications increasingly require processing inputs of 100k+ tokens, the gap between context length and inference efficiency has become a critical bottleneck. Context compression offers a way to reduce prefill costs while preserving task accuracy. However, existing training-free attention-based methods leave substantial gaps in demanding long-context tasks such as code reasoning. We present LongAttnComp, a long-context adaptation of AttnComp that fine-tunes a lightweight cross-attention scoring layer and introduces tokenlevel chunking, a token-budget top-p algorithm, positional reordering, and a formatagnostic query parser. We further design a two-stage fine-tuning recipe for the compressor: Stage 1 builds a general retrieval foundation from NIAH-style data, and Stage 2 extends it with multi-hop and reasoning data for broader long-context task coverage. On InfiniteBench Code-Debug, LongAttnComp matches or exceeds full-context accuracy, substantially outperforms training-free baselines, and transfers across four target models from three families. On LongBench v2, the two-stage recipe largely closes the Stage 1 gap on multi-document reasoning while preserving Code-Debug performance.
>
---
#### [new 059] Unveiling the Limits of Large Language Models in Inferring Pragmatic Meaning from Non-Verbal Responses
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的语用理解任务，旨在研究大语言模型在仅含非言语回应的对话中推断隐含意义的能力。工作包括评估模型表现、分析失败原因及探索改进方法。**

- **链接: [https://arxiv.org/pdf/2606.01845](https://arxiv.org/pdf/2606.01845)**

> **作者:** Sugyeong Eo; Heuiseok Lim
>
> **摘要:** Although large language models (LLMs) have shown considerable progress in pragmatic language understanding, prior research has focused mainly on their comprehension of verbal behavior. Nonetheless, non-verbal behavior remains a fundamental component of human communication, especially when deliberately utilized in isolation to convey indirect meanings. In this work, we present the first systematic evaluation of LLMs' ability to infer pragmatic meaning in dialogue consisting solely of non-verbal responses. We explore three research questions: (1) Can LLMs recognize indirect intent conveyed through non-verbal responses? (2) When and how do LLMs fail to capture non-verbal intent? (3) How can we improve LLMs' ability to interpret non-verbal intent?. Through the evaluation, we observe that LLMs struggle to infer underlying meaning from non-verbal responses, with accuracy dropping by up to 60% points compared to verbal ones. Further extensive analysis reveals a behavioral pattern in LLMs' interpretations of non-verbal behavior and demonstrates that in-context learning facilitates pragmatic inference.
>
---
#### [new 060] SkillHarm: Lifecycle-Aware Skill-Based Attacks via Automated Construction
- **分类: cs.CL**

- **简介: 该论文属于安全领域，旨在解决技能攻击威胁问题。提出SkillHarm基准，分析两种攻击场景，识别12类风险，验证现有系统脆弱性。**

- **链接: [https://arxiv.org/pdf/2606.02540](https://arxiv.org/pdf/2606.02540)**

> **作者:** Yuting Ning; Zhehao Zhang; Yash Kumar Lal; Boyu Gou; Junyi Li; Weitong Ruan; Chentao Ye; Rahul Gupta; Diyi Yang; Yu Su; Huan Sun
>
> **备注:** Work in Progress
>
> **摘要:** Agent skills occupy a privileged position in the agent workflow, as agents are expected to implicitly follow and execute them, rendering third-party skills a vulnerable attack surface. Existing studies have revealed unsafe agent behaviors induced by skill-based attacks, but they primarily evaluate poisoned skills within a single task execution and enumerate harms through ad-hoc risk lists. To bridge these gaps, we introduce SkillHarm, a benchmark of skill-based attacks across the skill-use lifecycle, paired with a systematic taxonomy of skill-relevant risks. SkillHarm evaluates two attack scenarios: Fixed-Payload Poisoning (FPP), where a fixed poisoned skill package directly compromises any task session that invokes it, and Self-Mutating Poisoning (SMP), where an initially benign execution silently mutates persistent skill content, deferring harm until a subsequent reuse. It further defines 12 risk types based on the agent workflow component targeted by the harm: data pipelines, system environments, and agent autonomy. To instantiate these attacks at scale, we build AutoSkillHarm, an automated construction pipeline with coding agents driven by natural-language harnesses. The resulting benchmark contains 879 attack samples across 71 skills. Experiments show that current agents remain vulnerable with attack success rates up to 86.3% in FPP and 69.3% in SMP. Our analysis further reveals a latent risk: many apparent attack failures stem from the agent failing to engage with the poisoned file rather than genuine resistance, and current defenses still fail to reliably mitigate the threat.
>
---
#### [new 061] ART: Attention Run-time Termination for Efficient Large Language Model Decoding
- **分类: cs.CL**

- **简介: 该论文属于大语言模型解码优化任务，解决长上下文解码中的内存带宽瓶颈问题。提出ART机制，在运行时终止不必要的KV块访问，提升解码效率。**

- **链接: [https://arxiv.org/pdf/2606.00024](https://arxiv.org/pdf/2606.00024)**

> **作者:** Chen Qiu; Guozhong Li; Panos Kalnis
>
> **摘要:** Long-context decoding in Large Language Models (LLMs) is severely constrained by the memory bandwidth required to fetch the extensive Key-Value (KV) cache. Most existing KV management methods rely on key-only pruning before decoding, despite the evidence that attention outputs depend jointly on keys and values, as incorporating values in their methods incurs prohibitive additional overhead. In this paper, we propose Attention Run-time Termination (ART), a lightweight run-time mechanism that tracks accumulated attention outputs during kernel execution and terminates subsequent KV block accesses once further contributions become negligible. This design makes ART orthogonal to existing key-based KV cache management methods, enabling seamless integration with them. Experiments on LongBench benchmarks show that ART achieves 20% higher generation throughput in large batch size than state-of-the-art baseline while maintaining comparable accuracy.
>
---
#### [new 062] Mechanistic Diagnostics of Spatial Lexical Bias in Multimodal Large Language Model Spatial Reasoning
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态语言模型任务，解决空间推理中的词汇偏差问题。通过分析模型对空间关系词的依赖，提出轻量级方法缓解偏差，提升模型准确性。**

- **链接: [https://arxiv.org/pdf/2606.01914](https://arxiv.org/pdf/2606.01914)**

> **作者:** Chuang Ma; Qianying Liu; Tomoyuki Obuchi; Fei Cheng; Wang Yang; Sudong Cai; Shuyuan Zheng; Akiko Aizawa; Sadao Kurohashi
>
> **摘要:** Multimodal large language models (MLLMs) remain unreliable on spatial multiple-choice questions, and their failures are often attributed to poorly attended visual information. In this work, we identify a complementary failure mode, spatial lexical bias: adding a spatial relation word to the answer options can attract the model's decision and make the newly added option likely to be selected. Using nine open-weight MLLMs, we show that this phenomenon is widely observed. In particular, models can answer a binary spatial question correctly, yet consistently select an incorrect third spatial option once it is added to the answer set. We isolate such binary-stable but ternary-fragile cases as diagnostic examples and leverage mechanistic interpretability tools, revealing that a substantial part of the failure instead originates on the language side rather than the visual side: visual attention analyses and residual-stream probes show the correct spatial relation remains internally available on these failures, while irrelevant-option controls, activation patching, and sparse component interventions trace the bias to specific LLM-side channels and neurons. Based on this finding, we show that a lightweight LLM-only DPO update on tiny single-object-pair synthetic data mitigates the bias, lifting four-way robust accuracy by up to 100 points on synthetic data, and by 68.0, 32.6, and 20.1 points on broader evaluation datasets WhatsUp, SpatialMQA-Direct, and VSR.
>
---
#### [new 063] Effects of Varying LLM Access on Essay Writing Behavior
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文属于教育技术研究，探讨不同LLM访问程度对写作行为的影响。研究通过实验分析LLM辅助对写作质量、参与度和作者认同感的作用，旨在找到有效整合AI辅助的方法。**

- **链接: [https://arxiv.org/pdf/2606.00250](https://arxiv.org/pdf/2606.00250)**

> **作者:** Julia Christenson; Karin de Langis; Shirley Anugrah Hayati; Dongyeop Kang
>
> **备注:** BEA (Building Educational Applications) Workshop 2026
>
> **摘要:** Investigating the degree to which large language models (LLMs) affect teaching and learning in universities can help identify strategies for integrating LLMs in a way that supports, rather than undermines, student learning outcomes. This study examined how varying levels of LLM assistance affect writing performance, engagement, and perceived authorship. We report a pilot study in which 24 college students were randomly assigned to write a short essay with no LLM access, limited access (<=3 prompts, responses capped at 100 words), or unlimited access. Overall essay quality was statistically indistinguishable across groups. Yet writing behavior and perceived authorship diverged sharply: students with limited access reported higher ownership (62.5% would submit the essay as independent work, vs. 25% in the unlimited group), stronger organizational gains, and more strategic, revision-focused prompting. The unlimited group spent more time writing, produced essays more similar to LLM output, and reported reduced creative expression. Our findings suggest that constraining, rather than banning, LLM access may preserve authorship confidence while retaining the scaffolding benefits of AI assistance.
>
---
#### [new 064] LinguIUTics at PsyDefDetect: Iterative Imbalance-Aware Fine-tuning of Qwen3-8B for Psychological Defense Mechanism Classification
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于心理防御机制分类任务，解决对话文本中罕见类别识别难题。通过改进的微调方法和数据增强策略，提升了模型在不平衡数据上的表现。**

- **链接: [https://arxiv.org/pdf/2606.00647](https://arxiv.org/pdf/2606.00647)**

> **作者:** Shefayat E Shams Adib; Ahmed Alfey Sani; Md Hasibur Rahman Alif; Ajwad Abrar
>
> **备注:** Accepted at PsyDefDetect, a shared task at the 25th BioNLP Workshop (BioNLP 2026), co-located with ACL 2026 in San Diego, CA, USA
>
> **摘要:** Detecting psychological defense mechanisms in conversational text remains a challenging clinical NLP problem. For the PsyDefDetect 2026 shared task (nine-class utterance classification evaluated via macro F1), our team LinguIUTics achieves a macro F1-score of 0.3917 on the official positive-class leaderboard, ranking 4th out of 21 registered teams and improving over the Ministral-8B task baseline (31.48 macro F1) by 7.7 absolute points (24.4 percent relative). BERT-family encoders and zero-shot LLMs proved ineffective on rare classes due to severe class imbalance, leading us to QLoRA fine-tuning of Qwen3-8B. We leverage three key strategies: grouped stratified cross-validation (preventing leakage), minority-class round-robin lexical augmentation, and a post-processing pipeline with logit bias tuning and ensemble blending. Together, these components close much of the validation-to-leaderboard gap and substantially improve minority-class recall, driving the critical "Unclear" class (Level 8) from near-zero performance to an F1 score of 0.797.
>
---
#### [new 065] BOUTEF: A Multilingual Corpus for FakeNews in North Africa -- Language as a Weapon
- **分类: cs.CL**

- **简介: 该论文属于虚假新闻检测任务，旨在研究北非地区虚假新闻的传播与特征。构建了多语言语料库BOUTEF，分析其主题、语言策略及用户互动，揭示虚假新闻的传播机制。**

- **链接: [https://arxiv.org/pdf/2606.00193](https://arxiv.org/pdf/2606.00193)**

> **作者:** Kamel Smaili; Yassine Toughrai; Amina Laggoun; David Langlois
>
> **摘要:** The rapid spread of fake news on social media has become a major challenge, particularly in multilingual and under-resourced contexts such as North Africa. In this paper, we introduce BOUTEF, a large-scale multilingual corpus designed to study the propagation, characteristics, and impact of fake news in Algeria and Tunisia. The corpus integrates three complementary components: fake narratives, genuine narratives, and associated user-generated comments, along with verified debunking information. It covers a wide range of languages and linguistic varieties, including MSA, Algerian and Tunisian dialects, Arabizi, French, English, and code-switched language. Building on this resource, we conduct a comprehensive empirical analysis combining quantitative and qualitative approaches. We examine thematic distributions, linguistic and rhetorical strategies, sentiment patterns, and social engagement dynamics. Statistical analyses reveal significant associations between thematic categories and message veracity, as well as strong correlations between user engagement and the visibility of fake content. Our findings show that fake news relies heavily on emotionally charged narratives, sensational framing, and hybrid linguistic practices that enhance virality and audience engagement. In contrast, debunking content adopts a more factual and verification-oriented style. Furthermore, a comparative analysis between Algeria and Tunisia highlights both shared dynamics and country-specific characteristics shaped by sociopolitical contexts. The results emphasize the role of informal language practices in the diffusion and reception of misinformation. By providing a rich, annotated, and publicly available dataset, this work contributes to advancing research on fake news detection, low-resource language processing, and the understanding of information disorders in complex linguistic environments.
>
---
#### [new 066] SENSE: Semantic Embedding Navigation with Soft-gated Evaluation for Retrieval-based Speculative Decoding
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出SENSE方法，解决检索式推测解码中的语义对齐问题，通过语义嵌入和软门控验证提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2606.00021](https://arxiv.org/pdf/2606.00021)**

> **作者:** Shaowen Chen; Zhicheng Liao; Hongwei Wang
>
> **摘要:** Speculative Decoding (SD) accelerates Large Language Model (LLM) inference by employing a lightweight draft model to propose candidate tokens, which are verified in parallel by the target model, without compromising generation quality. While Retrieval-based Speculative Decoding (RSD) is favored for its plug-and-play versatility, its potential is impeded by rigid lexical dependencies, rendering both retrieval and verification brittle to surface-level variations. To address this, we propose SENSE (Semantic Embedding Navigation with Soft-gated Evaluation). By anchoring retrieval on the hidden states of the target model, SENSE establishes robust semantic alignment, which empowers the Soft-gated Evaluation module to validate semantic equivalence rather than surface forms. To ensure rigorous benchmarking, we deconstruct existing methods into atomic primitives within a unified framework, facilitating granular, component-level comparison. Extensive experiments across diverse domains demonstrate that SENSE outperforms multiple baselines on the LLaMA and Qwen families, attaining up to 4.09 mean acceptance length and 3.26x speedup, while preserving generation quality. Our code will be released upon publication.
>
---
#### [new 067] Masking Stale Observations Helps Search Agents -- Until It Doesn't: A Regime Map and Its Mechanism
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于智能代理任务，研究如何通过遮蔽过时观察提升搜索效率。工作包括系统测试不同模型和检索器，发现遮蔽效果呈非对称曲线，揭示其机制并提出新的分析视角。**

- **链接: [https://arxiv.org/pdf/2606.00408](https://arxiv.org/pdf/2606.00408)**

> **作者:** Haoxiang Zhang; Qixin Xu; Zhuofeng Li; Lei Zhang; Pengcheng Jiang; Yu Zhang; Julian McAuley
>
> **备注:** 47 pages, 7 figures
>
> **摘要:** Long-horizon search agents accumulate large amounts of retrieved content across many tool calls, making context-budget efficiency increasingly important. A minimal intervention is to mask stale observations from the context as the trajectory progresses, but it remains unclear when this form of context management helps and why. We study observation masking through a systematic sweep over various agent backbones (4B to 284B parameters) and three retrievers on offline and live-web agentic search benchmarks. We find that the accuracy gain from masking follows an asymmetric inverted-U shape when plotted against the model's accuracy without context management: a plateau under weak retrievers, a peak when a strong retriever meets a mid-capacity model, and a sharp collapse when the model is saturated. This pattern reflects the interaction between retriever recall and the model's implicit filtering capacity, rather than either factor in isolation. Mechanistically, masking implements a token-for-turn trade-off: it removes observations the model has largely stopped attending to and pages the agent rarely re-opens. The added turns help when they convert failures into successes, but they fail when masking removes evidence the model would otherwise have used. We therefore reframe context management as a regime-dependent intervention and provide a holistic perspective for analyzing context use in agentic deep search. We release our scaffold and trajectories here (this https URL) to support future research.
>
---
#### [new 068] Multilingual Idioms in Sentences and Conversations Across High-, Medium-, and Low-Resource Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言自然语言处理任务，旨在解决跨语言成语理解问题。研究构建了MIDI数据集，涵盖不同资源语言的成语上下文，发现模型在低资源语言中表现较差，且字面理解更难。**

- **链接: [https://arxiv.org/pdf/2606.02147](https://arxiv.org/pdf/2606.02147)**

> **作者:** Saeed Almheiri; Bilal Elbouardi; Salsabila Zahirah Pranida; Irina Nikishina; Ashwath Rao B; Parameswari Krishnamurthy; Muhammad Cendekia Airlangga; Rifo Ahmad Genadi; Nguyen Phan Gia Bao; Amir Hossein Yari; Hawau Olamide Toyin; Nurdaulet Mukhituly; Mena Attia; Besher Hassan; Ahmad Fathan Hidayatullah; Tatsuki Kuribayashi; Haonan Li; Suma Bhat; Fajri Koto
>
> **摘要:** Idiomatic expressions pose a major challenge for multilingual NLP because their meanings shift between figurative and literal usage, often requiring context for accurate interpretation. Prior work has focused on high-resource languages typically evaluates isolated idiom-meaning questions, overlooking realistic discourse. We introduce MIDI, a multilingual idiom dataset spanning 3 high-, 3 medium-, and 12 low-resource languages, curated by native speakers. Unlike previous datasets, MIDI provides idioms embedded in both sentence-level and conversational contexts, capturing both literal and figurative readings. Benchmarking state-of-the-art models shows that idiom comprehension degrades in low-resource languages and that, in all resource tiers, literal interpretations are substantially harder than figurative ones. Conversational context improves performance but does not eliminate these disparities. Through controlled tests and interventions on hidden representations, we further separate memorization from reasoning, exposing core limitations of current models.
>
---
#### [new 069] Parameter Alignment Mitigates Catastrophic Forgetting in Multilingual Expert Language Models
- **分类: cs.CL**

- **简介: 该论文属于多语言语言模型任务，旨在解决持续预训练中的灾难性遗忘问题。通过参数对齐策略，如层冻结和正则化，有效减少知识丢失，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2606.00284](https://arxiv.org/pdf/2606.00284)**

> **作者:** Sanchit Ahuja; Terra Blevins
>
> **备注:** 25 Pages, 5 Figures
>
> **摘要:** While continual pretraining~(CPT) is a practical way to extend large language models to new languages, naïve finetuning on targeted data erodes existing capabilities through catastrophic forgetting. Organizing training around language families reduces cross-language interference but cannot alone prevent forgetting of the general knowledge needed for downstream tasks. We link this forgetting to parameter drift in multilingual CPT and present a suite of five layer-aware parameter alignment strategies: hard layer freezing, soft regularization, post-hoc weight reversion, and model merging. We systematically compare our alignment strategies against two unregularized CPT baselines on benchmarks spanning 32 training languages from five language families, plus held-out languages, across four evaluation axes: perplexity, reading comprehension, physical reasoning, and translation. Parameter alignment substantially reduces forgetting at minimal cost to language acquisition: layer freezing and regularization best preserve comprehension, whereas post-hoc reversion yields the strongest translation gains. Together, these results map the acquisition--forgetting frontier for family-expert CPT and offer practical deployment guidelines pairing each strategy to the tasks it best serves.
>
---
#### [new 070] Agreement Metrics for LLM-as-Judge Evaluation: What to Report and Why
- **分类: cs.CL; cs.HC; physics.data-an**

- **简介: 该论文属于自然语言处理中的评估任务，探讨LLM作为评判者的评价指标选择问题，指出现有指标的冗余性及处理异常情况的不同方法对结果的影响，并提出报告清单。**

- **链接: [https://arxiv.org/pdf/2606.00093](https://arxiv.org/pdf/2606.00093)**

> **作者:** Delip Rao; Chris Callison-Burch
>
> **备注:** 12 pages
>
> **摘要:** Validating an LLM judge against human annotations usually means reporting several agreement statistics: accuracy, precision, recall, $F_1$, Cohen's $\kappa$, and one or more rank correlations. A survey of 24 recent LLM-as-judge papers finds metric choice entangled with the judgment scale, tie handling, invalid outputs, and abstention handling, and those choices rarely stated. For binary criteria -- the common case in rubric-based evaluation, where each criterion is graded MET or UNMET -- most of the reported numbers are redundant: Pearson's $r$, Spearman's $\rho$, Kendall's $\tau_b$, the phi coefficient $\phi$, and the Matthews Correlation Coefficient all reduce to a single number on non-degenerate binary data, so reporting several of them only creates an illusion of corroborating evidence. Cohen's $\kappa$ is the one agreement coefficient that adds information: it shares $\phi$'s numerator but normalizes differently, and the gap between them measures how far the judge's positive-label rate has drifted from the human's. We then trace what changes when a judge may abstain with a CANNOT_ASSESS verdict: the three common ways of handling abstentions are not interchangeable preprocessing choices but answer different questions, and they break the binary equivalences. The same equivalences reappear, up to a negligible finite-sample correction, for multi-judge ensembles scored with Fleiss' $\kappa$ or Krippendorff's $\alpha$. We close with a reporting checklist that names the judgment scale, the abstention and tie handling mode, coverage, the confusion matrix, and the aggregation level alongside any scalar agreement coefficient.
>
---
#### [new 071] Machine Learning for Coding Retail Product Names to Consumer-Price Categories: A Rule-plus-Bag-of-Words Pipeline with Reliability-Weighted Human-in-the-Loop Labeling
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于产品分类任务，解决零售商品名称到价格类别的映射问题。通过规则与词袋模型结合的管道，实现高效准确分类。**

- **链接: [https://arxiv.org/pdf/2606.02004](https://arxiv.org/pdf/2606.02004)**

> **作者:** Vladimir Beskorovainyi
>
> **备注:** 11 pages, 3 tables. Methodology paper; illustrative experiments only, no proprietary data
>
> **摘要:** Consumer-price measurement increasingly draws on alternative data sources -- scanner, web-scraped, and transaction/receipt data. A recurring obstacle is that product descriptions in such sources are short, noisy, and abbreviated, with no standard product code, so each item must first be mapped to a consumption classification (e.g., the UN COICOP scheme) before prices can be compared. This paper studies that mapping as a general, reproducible method. The pipeline is: (i) text normalization and tokenization of noisy item names; (ii) a prefix-tree (trie) rule-based pre-classifier driven by per-category key-phrases and stop-phrases; and (iii) a per-category binary confirmation model deciding whether an item belongs to a tentatively assigned category. For labels at scale we use a human-in-the-loop protocol in which annotators give a binary valid/reject judgment, aggregated by a dynamically updated reliability weight; the model joins the same rule, enabling continual fine-tuning. Our empirical finding is deflationary: in a controlled, leakage-free study (one category, real positives vs. hard negatives, five seeds), bag-of-words models essentially saturate the task (F1 about 0.99) -- a linear classifier matches a multilayer perceptron, explicit word-order (n-gram) features add nothing, and about 67 labeled examples already suffice. A Monte-Carlo study of the labeling protocol shows the reliability-weighted vote barely beats plain majority (its additive weights saturate) while Dawid-Skene recovers labels markedly better. We also discuss price-level quality control and design lessons for statistical offices considering transaction data. All figures are illustrative; no confidential data, code, or documentation is reproduced.
>
---
#### [new 072] Connecting the Dots: Benchmarking Reflective Memory in Long-Horizon Dialogue
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于长时对话任务，解决现有基准无法评估反思性记忆的问题。提出RefMem-Bench基准和REMIND框架，提升模型从多模态线索中推断深层含义的能力。**

- **链接: [https://arxiv.org/pdf/2606.01223](https://arxiv.org/pdf/2606.01223)**

> **作者:** Jingjie Lin; Bingbing Wang; Zihan Wang; Zhengda Jin; Weiming Qiao; Jing Li; Ruifeng Xu
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Despite substantial progress in long-context modeling, existing benchmarks remain confined to factual memory for explicit recall, failing to measure the reflective memory required to synthesize fragmented, multimodal cues into high-level interpretations. To address this gap, we introduce RefMem-Bench, a benchmark for reflective memory in long-horizon dialogue. RefMem-Bench contains 26K annotated QA instances with eight reflective-memory dimensions and three task formats, requiring models to move beyond surface-level retrieval and infer latent meanings from evidence distributed across interaction histories. To enhance reflective memory capability, we propose REflective Memory INDuction (REMIND), a hierarchical framework that treats reflective memory as progressive meaning construction. REMIND couples question-conditioned evidence retrieval, salience-aware grounding, and abstraction-level supervision, and uses Progressive Reflective Alignment to distill high-level reflective reasoning into the factual inference pathway. Experiments show RefMem-Bench poses a substantial challenge to current models, while REMIND consistently improves both answer accuracy and memory recall through progressive evidence perception, grounding, and abstraction.
>
---
#### [new 073] Sparse Autoencoders for Interpretable Emotion Control in Text-to-Speech
- **分类: cs.CL**

- **简介: 该论文属于文本到语音（TTS）任务，旨在解决情感控制可解释性问题。通过稀疏自编码器分析语义隐层，识别稀疏潜在特征，实现情感诱导与抑制。**

- **链接: [https://arxiv.org/pdf/2606.01479](https://arxiv.org/pdf/2606.01479)**

> **作者:** Hongfei Du; Jiacheng Shi; Sidi Lu; Gang Zhou; Ye Gao
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Integrating large language models (LLMs) into text-to-speech (TTS) systems has improved speech expressiveness, yet interpretable emotional control remains challenging. Existing approaches primarily rely on external conditioning or global activation steering, offering limited insight into the internal representations underlying emotional control. In this work, we analyze emotion-related variation in the semantic hidden states of LLM-based TTS models using sparse autoencoders (SAEs) to identify sparse latent features. Our analysis shows that emotional variation is distributed across multiple sparse latent features, while intervening on a small subset enables interpretable emotion control. Building on this observation, we introduce a feature-level intervention framework for bidirectional emotion induction and suppression without modifying backbone parameters. We further show that distinct latent features are associated with specific acoustic attributes (e.g., pitch), suggesting that emotional expression arises from coordinated latent contributions rather than a single global shift. Empirically, steering these sparse latent features achieves comparable or superior emotion induction and suppression performance relative to global steering and existing TTS baselines.
>
---
#### [new 074] Momento: Evaluating Persistent Memory and Reasoning with Multi-Session Agentic Conversations
- **分类: cs.CL**

- **简介: 该论文提出Momento基准，用于评估多轮会话中智能体的持续任务完成能力，解决现有基准忽略历史信息的问题。**

- **链接: [https://arxiv.org/pdf/2606.00832](https://arxiv.org/pdf/2606.00832)**

> **作者:** Adril Putra Merin; David Anugraha; Ayu Purwarianti; Genta Indra Winata
>
> **备注:** Preprint
>
> **摘要:** Recent advances in agentic AI have enabled agents to complete complex tasks through tool use, reasoning, and multi-step planning. Yet existing benchmarks evaluate agents within a single session, ignoring past actions, stated preferences, and prior decisions that agents must integrate to fulfill personalized user goals. We introduce Momento, a benchmark for persistent agentic task completion in multi-session service environments, requiring agents to take consequential, tool-mediated actions while resolving temporal dependencies and evolving user goals across sessions. Experimental results reveal that current agents fail primarily through misestimation of user state, treating prior session history as a reliable proxy for current context rather than stale information requiring re-validation, highlighting a substantial gap between current agent capabilities and realistic long-horizon human-agent interaction.
>
---
#### [new 075] Revise, Don't Freeze: Sampler-Matched Training for Self-Correcting Masked Diffusion Language Models
- **分类: cs.CL**

- **简介: 该论文研究自修正掩码扩散语言模型的采样问题，解决标准采样器无法有效利用模型修订能力的问题。提出D3IM采样器和SCOPE训练方法，提升模型生成质量。**

- **链接: [https://arxiv.org/pdf/2606.01026](https://arxiv.org/pdf/2606.01026)**

> **作者:** Longxuan Yu; Shaorong Zhang; Yu Fu; Hui Liu; Yue Dong; Greg Ver Steeg
>
> **备注:** 8 pages, 2 figures, 10 tables
>
> **摘要:** Masked diffusion language models (MDLMs) re-predict every position at each denoising step, but standard samplers commit tokens once revealed, leaving this revision capability unused. Existing approaches either add heuristic or learned mechanisms to revise committed tokens, or remask them back to [MASK] before re-predicting; a principled sampler that directly revises visible tokens without auxiliary modules remains underexplored. We introduce D3IM, a parameter-free sampler derived as a corrector-style reverse update that permits direct visible-to-visible revision without additional modules or auxiliary passes. D3IM also reveals a model-side obstacle we term preservation bias: the model tends to reproduce its own wrong committed tokens rather than correct them. We address this with SCOPE (Self-Conditioned On Prediction Errors), a lightweight post-training procedure that simulates D3IM's sampling process. On LLaDA-8B at 64 denoising steps, SCOPE+D3IM improves over the original LLaDA-8B with standard unmasking by +13.0 on GSM8K (68.3%), +4.8 on MATH-500 (23.6%), +15.3 on HumanEval (29.3%), and +10.4 on MBPP (30.8%), with gains that increase as more denoising steps are used on math and HumanEval.
>
---
#### [new 076] SPADE-Bench: Evaluating Spontaneous Strategic Deception in Agents via Plan-Action Divergence
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于智能代理安全任务，旨在解决代理欺骗问题。通过构建SPADE-Bench基准，评估代理在工具使用中的计划与行动偏差，提升系统可控性。**

- **链接: [https://arxiv.org/pdf/2606.02380](https://arxiv.org/pdf/2606.02380)**

> **作者:** Yuyan Bu; Haowei Li; Qirui Zheng; Bowen Dong; Kaiyue Yang; Jiaming Ji; Yingshui Tan; Wenxin Li; Yaodong Yang; Juntao Dai
>
> **摘要:** As LLM-based agents expand their operational scope, reliability becomes a prerequisite for real-world deployment. However, in practical applications, human users cannot monitor every immediate behavior; instead, the execution process often remains a black box, leaving users dependent solely on the agent's self-reported updates. This opacity creates a critical risk: agents may present observer-facing reports that diverge from their executed actions, rendering the system uncontrollable, especially in high-stakes autonomous scenarios. We term such self-reported plan-action divergence as agent deception. To assess this, we introduce SPADE-Bench, a benchmark designed to evaluate spontaneous plan-action divergence. Unlike prior deception benchmarks, SPADE-Bench simultaneously integrates actual tool execution and controlled pressure scenarios. This design ensures ecological validity and rigorously distinguishes strategic deception from mere hallucination through controlled plan-action comparisons under pressure. Experiments across mainstream models confirm that agent deception is a genuine and pressing issue in tool-use contexts. By providing a comprehensive and robust evaluation framework, SPADE-Bench fills a critical gap in agent safety, facilitating the community's progress toward building trustworthy and controllable autonomous systems.
>
---
#### [new 077] CRAM: Centroid-Routing and Adaptive MoE for Multimodal Continual Instruction Tuning
- **分类: cs.CL**

- **简介: 该论文提出CRAM方法，解决多模态持续指令微调中的任务遗忘与参数效率问题，通过模块化和自适应机制提升模型持续学习能力。**

- **链接: [https://arxiv.org/pdf/2606.02502](https://arxiv.org/pdf/2606.02502)**

> **作者:** Jun-Tao Tang; Zhen-Hao Xie; Yu-Cheng Shi; Da-Wei Zhou
>
> **摘要:** Multimodal Large Language Models (MLLMs) unify heterogeneous vision-language tasks under a shared generative framework via instruction tuning, yet real-world deployment demands continuous capability expansion, making Multimodal Continual Instruction Tuning (MCIT) essential. Existing methods either update all tasks with a shared parameter set or allocate dedicated modules for each new task. Shared updates force heterogeneous tasks to compete, causing forgetting of learned capabilities. Conversely, isolated expansion prevents interference but severely limits parameter efficiency over long task streams. To address this dilemma, we propose CRAM. Specifically, by isolating task-specific patterns into independent modules, CRAM mitigates catastrophic forgetting across tasks. To further boost parameter efficiency, we utilize adaptive-rank instantiation to identify the capability gap between existing expert capability and new task demands, and dynamically allocate only the necessary parameters. To ensure stable reuse among tasks, centroid-guided routing recognizes and activates existing experts' capabilities, while an orthogonality penalty confines new updates to task-specific directions, preventing re-learning general capability. Extensive experiments across diverse benchmarks consistently demonstrate its superiority over existing methods.
>
---
#### [new 078] Cost-Aware Diffusion Draft Trees for Speculative Decoding
- **分类: cs.CL**

- **简介: 该论文提出CaDDTree方法，用于加速语言模型推理。针对 speculative decoding 中预算选择不合理的问题，通过优化吞吐量实现更高效的任务处理。**

- **链接: [https://arxiv.org/pdf/2606.01813](https://arxiv.org/pdf/2606.01813)**

> **作者:** Shuai Zhang; Huachuan Qiu; Hongliang He; Yong Dai
>
> **摘要:** Speculative decoding accelerates inference by having a lightweight drafter propose tokens verified in parallel by the target language model. Block diffusion drafters such as DFlash generate an entire draft block in one pass, yielding per-position marginals; DDTree uses these to build a candidate tree that maximizes expected acceptance length under a fixed node budget. We observe, however, that acceptance length is non-decreasing in budget: it always favors larger trees regardless of verification cost, offering no principled basis for budget selection. We introduce \textbf{CaDDTree} (Cost-aware Diffusion Draft Tree), a method that directly optimizes token throughput (expected tokens generated per unit time) by jointly selecting the tree structure and node budget. We model draft and verification latencies explicitly, show that the throughput objective decomposes into a per-round one-dimensional search over the budget, and prove that under a convex verification cost the throughput function is \emph{unimodal}, enabling an efficient greedy stopping rule. CaDDTree requires no offline budget search, adapting the budget each round from the current per-position distributions and verification cost. Experiments on Qwen3-4B and Qwen3-8B across eight benchmarks spanning reasoning, coding, and instruction-following tasks show that \caDDTree{} matches or surpasses DDTree with oracle budget selection on nearly all tasks.
>
---
#### [new 079] CRAFTQA: A Code-Driven Adaptive Framework for Complex Structured Data Reasoning
- **分类: cs.CL**

- **简介: 该论文提出CRAFTQA，解决统一结构化数据问答任务中预定义函数限制的问题，通过生成可执行代码实现复杂推理。**

- **链接: [https://arxiv.org/pdf/2606.02170](https://arxiv.org/pdf/2606.02170)**

> **作者:** Chengtao Gan; Zhiqiang Liu; Long Jin; Yushan Zhu; Lei Liang; Wen Zhang
>
> **备注:** Accepted by Findings of ACL 2026
>
> **摘要:** Real-world scenarios involve massive heterogeneous structured data (e.g., tables, knowledge graphs), making effective reasoning over such diverse data increasingly important. Unified structured data question answering has emerged as a prominent research trend, aiming to answer natural language questions across different structured data types within a single framework. However, existing unified methods share a common limitation: they rely on a set of predefined functions, which restricts their ability to perform complex reasoning beyond these predefined operations. To overcome this fundamental limitation, we propose CRAFTQA, a novel adaptive code-driven framework comprising two core modules, CodeSTEP and CRAFT. The CodeSTEP module is a paradigm that generates a complete executable Python code sequence, which contains step-by-step code-based reasoning operations based on the question. The CRAFT module dynamically generates custom code functions for operations beyond the predefined function set, and seamlessly integrates with CodeSTEP to significantly enhance flexibility in handling complex reasoning. Comprehensive experiments on multiple structured datasets demonstrate that CRAFTQA achieves remarkable improvements in complex reasoning scenarios compared to existing unified methods.
>
---
#### [new 080] Not All Flips Are Conformity: Decomposing Stance Convergence in Multi-Agent LLM Debate
- **分类: cs.CL**

- **简介: 该论文研究多智能体辩论中的立场趋同问题，旨在区分真实推理与社会合规。通过分解机制，分析答案翻转原因，提出干预策略以减少有害趋同。**

- **链接: [https://arxiv.org/pdf/2606.00820](https://arxiv.org/pdf/2606.00820)**

> **作者:** Xiqi Hao; Zengqing Wu; Yu-Xuan Qiu; Chuan Xiao; Ruiqi Xu; Shuyuan Zheng; Jianbin Qin
>
> **摘要:** Multi-agent debate (MAD) is a promising strategy for improving LLM reasoning, but when agents converge on a shared answer, it is unclear whether that convergence reflects genuine deliberation or social compliance. We show that the conventional answer flip rate conflates three distinct mechanisms: spontaneous instability, stance-induced conformity, and reasoning-induced persuasion. Our three-source decomposition framework isolates each through controlled counterfactual conditions. In the primary MMLU-Pro setting, 37% of agent-question observations change under self-reflection alone, while robustness tests show substantial model-dependent instability across GPQA-Diamond and three model families; strict conformity is 29% in the primary setting and remains predominantly harmful across model replications (57-77% correct-to-wrong). A controlled information-gradient experiment reveals that even vacuous reasoning is associated with 20-39% error adoption among resistant agents, with reasoning-like presentation carrying substantial persuasive weight. Harmful conformity can be predicted from Round 0 features (AUC = 0.79), and risk-targeted intervention reduces it by 13.6 percentage points (p < 0.001). However, without correctness labels or self-reflection controls, reducing peer adoption does not improve accuracy, because harmful and beneficial influence cannot be distinguished.
>
---
#### [new 081] DFlare: Scaling Up Draft Capacity for Block Diffusion Speculative Decoding
- **分类: cs.CL**

- **简介: 该论文属于大模型推理加速任务，解决块扩散推测解码中草稿模型表达能力不足的问题。通过轻量层间融合机制提升每层表达能力，实现更深层次的草稿模型和更快的推理速度。**

- **链接: [https://arxiv.org/pdf/2606.02091](https://arxiv.org/pdf/2606.02091)**

> **作者:** Jiebin Zhang; Zhenghan Yu; Song Liu; Eugene J.Yu; Zheng Li; Dawei Zhu; Jiangshan Duo; Weimin Xiong; Yifan Song; Guanghua Yu; Jianchen Zhu; Sujian Li
>
> **备注:** 12 pages, 3 figures
>
> **摘要:** Block diffusion speculative decoding accelerates LLM inference by predicting all tokens within a block simultaneously for the target model to verify in parallel. Predicting an entire block at once requires a sufficiently capable draft model and effective utilization of the target model's internal knowledge. However, the state-of-the-art method DFlash constrains all draft layers to share a single fused representation derived from only a few target layers, limiting per-layer expressiveness and hindering further scaling of draft capacity. In this paper, we present \modelname, which flares out the narrow conditioning bottleneck of DFlash through a lightweight layer-wise fusion mechanism: each draft layer attends to its own learnable combination of a broad set of target layers at negligible overhead, simultaneously injecting richer target knowledge and providing every draft layer with a distinct input. This enhanced per-layer expressiveness enables scaling the draft model to deeper architectures with consistent gains. We further scale training data from 800K to 2.4M samples to fully exploit the enlarged capacity. On six benchmarks spanning mathematical reasoning, code generation, and conversation, \modelname attains average wall-clock speedups of 5.52x on Qwen3-4B, 5.46x on Qwen3-8B, and 3.91x on GPT-OSS-20B, improving over DFlash by roughly 11\%, 8\%, and 5\% respectively. Our code is available at this https URL.
>
---
#### [new 082] Decoding in Order-Agnostic Language Models: Chain-Rule Deviation and Uniform Spreading
- **分类: cs.CL**

- **简介: 该论文研究顺序无关语言模型的解码问题，分析条件概率分布的不一致性及解码路径的可信度方差，提出用方差作为诊断指标以评估解码质量。**

- **链接: [https://arxiv.org/pdf/2606.00997](https://arxiv.org/pdf/2606.00997)**

> **作者:** Lin Yao
>
> **摘要:** Order-agnostic language models (OALMs), including discrete diffusion language models (dLLMs), are trained to predict masked tokens under arbitrary conditioning sets, allowing sequences to be generated or scored under arbitrary reveal orders at inference time. In LLaDA-2.1, we report three findings. First, the learned conditionals are not exact factorizations of a coherent joint distribution: changing only the reveal order shifts target log-likelihood by up to 0.49 nats/token, so likelihood alone mixes content difficulty with path-dependent artifacts. Second, although confidence-first (CF) decoding is order-agnostic, its reveal orders are close to left-to-right (L2R) on content tokens. Third, we propose a complementary diagnostic based on the shape of the confidence trace. A uniform-spreading theorem shows that, at fixed total likelihood, target recoverability is maximized when per-step confidence is spread uniformly; the resulting deviation motivates $\mathrm{Var}(\log q_t)$ as a diagnostic for comparing decoding paths. Across C4 and four downstream benchmarks, low variance separates structured paths from random ordering, and variance is consistently associated with downstream correctness. These results support reporting mean confidence and confidence variance jointly when comparing OALM decoding paths.
>
---
#### [new 083] Identifying High-Confidence Social Biases in LLMs for Trustworthy Conversational Tutoring Agents
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的社会偏见检测任务，旨在解决LLMs在对话教学中可能存在的高自信偏见问题。通过构建数据集评估模型的偏见识别能力，发现LLMs在该场景下表现不佳且过于自信。**

- **链接: [https://arxiv.org/pdf/2606.01584](https://arxiv.org/pdf/2606.01584)**

> **作者:** Aitor Arronte Alvarez; Naiyi Xie Fincham
>
> **备注:** Accepted for AIED 2026
>
> **摘要:** Conversational tutoring agents have been shown to improve learning engagement and student outcomes, and large language models (LLMs) are increasingly used in these systems to provide scalable, personalized feedback. However, LLMs may perpetuate or amplify stereotypical social biases, posing particular risks in educational settings. In this study, we evaluate LLMs in conversational tutoring scenarios to identify high-confidence social biases, instances where models are unable to identify biased judgments in tutoring conversations while maintaining strong confidence in their assessments, potentially affecting their reasoning and the feedback they provide to learners. We present a new dataset generation method that enables bias evaluation under naturalistic instructional conditions by regenerating student-AI tutor interactions and introducing turns with controlled bias derived from a benchmark dataset. Using this data, we assess multiple LLMs' ability to detect stereotypical biases and analyze the confidence and reasoning underlying their responses through computational and human evaluations. We find that bias detection is substantially more challenging in conversational tutoring contexts than in benchmark-based evaluations, and that state-of-the-art LLMs are overconfident in their incorrect assessments of stereotypical bias statements. Moreover, model confidence strongly influences reasoning and feedback, highlighting the risks of overconfident, biased behavior in LLM-based tutoring agents. We conclude by discussing implications, mitigation considerations, and directions for future research.
>
---
#### [new 084] HERO'S JOURNEY: Testing Complex Rule Induction with Text Games
- **分类: cs.CL**

- **简介: 该论文提出HERO'S JOURNEY基准，用于测试目标导向任务中的规则归纳问题。通过多步骤执行验证模型的规则推理能力，发现模型在属性任务上有提升，但程序性任务仍具挑战。**

- **链接: [https://arxiv.org/pdf/2606.02556](https://arxiv.org/pdf/2606.02556)**

> **作者:** Anshun Asher Zheng; Kanishka Misra; David I. Beaver; Junyi Jessy Li
>
> **备注:** 24 pages
>
> **摘要:** We introduce HERO'S JOURNEY, a benchmark for rule induction in goal-directed episodic tasks, where agents must infer hidden rules from demonstrations and act on them through multi-step execution. HERO'S JOURNEY covers eight tasks across attribute and procedural induction families, each with four structural rule forms, controllable lexical grounding, and identifiability conditions. Evaluating state-of-the-art LLMs, we find that models show evidence of rule induction, but the ability is limited and uneven across tasks. Meanwhile, process execution adds an execution bottleneck for models, whereas surface semantics has minimal effect. Induction-specific steering methods improve performance on attribute tasks but show no reliable gains on procedural tasks, suggesting the gap in procedural induction remains an open challenge.
>
---
#### [new 085] RCEM: Embedder Equipped with Query Rewriting Skill for Robust Conversational Search in Distributional Shift
- **分类: cs.CL**

- **简介: 该论文属于对话式检索任务，解决分布偏移下的鲁棒性问题。提出RCEM模型，通过嵌入模型学习查询重写能力，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2606.01697](https://arxiv.org/pdf/2606.01697)**

> **作者:** Kilho Son; Paul Hsu; Cha Zhang; Dinei Florencio
>
> **摘要:** Conversational search has become increasingly important in retrieval-augmented generation (RAG) systems, where users interact with AI assistants through multi-turn conversations containing context-dependent queries. We propose RCEM, a conversational dense retrieval model that distills the query reformulation capability of LLMs into the embedding model, enabling context-aware retrieval without explicit query rewriting during inference. Unlike prior conversational dense retrieval approaches that learn direct conversation-to-document matching, RCEM aligns conversational-query embeddings with rewritten-query embeddings, improving robustness under distributional shift. RCEM does not require conversational query-to-document relevance mappings for training, which are often expensive and difficult to obtain with high quality. Extensive experiments on QReCC, TopiOCQA, and TREC CAsT demonstrate that RCEM consistently outperforms strong conversational retrieval baselines, achieving particularly large gains under distributional shift, including up to 20% improvement in Recall@10. RCEM further extends the base embedding model with conversational query rewriting capability while preserving its original retrieval functionality, allowing both standalone and conversational queries to be encoded by a single model and searched against existing document indexes without rebuilding the retrieval database.
>
---
#### [new 086] On the Limits of LLM Adaptability: Impact of Model-Internalized Priors on Annotation Task Performance
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **简介: 该论文研究LLM在标注任务中的表现，探讨模型内化先验与指令的交互问题，发现提示修正效果有限，强调定义对齐的重要性。**

- **链接: [https://arxiv.org/pdf/2606.00467](https://arxiv.org/pdf/2606.00467)**

> **作者:** Etienne Casanova; Rafal Kocielnik; R. Michael Alvarez
>
> **备注:** Accepted at ICML 2026 (Oral & Spotlight); PMLR vol. 306. 9 pages, 4 figures
>
> **摘要:** Large Language Models (LLMs) are increasingly used for zero-shot annotation and LLM-as-a-judge tasks, yet their reliability hinges on how model-internalized priors interact with user-provided instructions. We investigate three dimensions of this interaction: (1) how an LLM's familiarity with data and task definitions affects performance, (2) the extent to which additional information in prompts can correct zero-shot errors ("decision stickiness"), and (3) model susceptibility to misaligned task definitions. Through experiments on toxicity detection across diverse datasets (spanning social media, gaming, news, and forums) using both dense and mixture-of-experts models, we find that nearly two-thirds of zero-shot errors are resistant to correction, with an overall rescue rate (fraction of initial errors corrected by prompting) of only 34.8%. High-confidence errors prove especially resistant to correction. When given misaligned definitions, LLMs follow them while maintaining confidence levels unchanged from the aligned condition. Crucially, we introduce Definition-Specific Familiarity (DSF), which measures alignment between a model's internal concept and the task definition. After controlling for dataset-level confounds, DSF shows a positive association with model performance (partial r = +0.41), while three distinct memorization metrics (ROUGE-L, BERTScore, and embedding cosine similarity) all fail to show a positive association. These findings show the limitations of prompt-based correction in annotation tasks, highlighting the importance of definition alignment over text-level memorization.
>
---
#### [new 087] PlanarBench: Evaluating LLM Spatial Reasoning via Planar Graph Drawing
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出PlanarBench，评估大语言模型在给定边列表的情况下绘制平面图的能力，属于空间推理任务，旨在测试模型的结构理解与生成能力。**

- **链接: [https://arxiv.org/pdf/2606.02010](https://arxiv.org/pdf/2606.02010)**

> **作者:** Oleksandr Nikitin
>
> **备注:** 12 pages, 4 figures, this https URL
>
> **摘要:** PlanarBench tests whether LLMs can draw planar graphs as ASCII art given only an edge list -- a spatial reasoning task that resists memorization because edge order, edge orientation, and node labels are all permutable. We evaluate 91 models on the 199 simplest non-isomorphic connected planar graphs (2 - 7 vertices). Edge count is the dominant difficulty predictor ($r = -0.85$) -- a finding not reported in prior LLM graph benchmarks, which use only node count as the difficulty axis.
>
---
#### [new 088] MMG2Skill: Can Agents Distill In-the-Wild Guides into Self-Evolving Skills?
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于智能体技能学习任务，旨在将网络上的非结构化指南转化为可执行技能。提出MMG2Skill框架，通过闭环优化提升智能体性能。**

- **链接: [https://arxiv.org/pdf/2606.01993](https://arxiv.org/pdf/2606.01993)**

> **作者:** Xinyu Che; Junqi Xiong; Yunfei Ge; Xinping Lei; Shihao Li; Hang Yan; Han Li; Yuanxing Zhang; Zhiqi Bai; Jinhua Hao; Ming Sun; Han Li; Jiaheng Liu
>
> **备注:** 35 pages, 12 figures, 13 tables. Code: this https URL
>
> **摘要:** Abundant procedural knowledge on the Web holds great potential for helping agents solve long-horizon tasks. However, such knowledge is often multimodal, heterogeneous, noisy, and implicitly assumes human executors, making it difficult to use directly as the skills required by agents. To bridge the gap between human-oriented guides and agent-executable skills, we formalize this problem as guide-to-skill learning: converting in-the-wild guides into executable skills and continuously improving them from trajectories observable to the agent. To evaluate the capability of existing agents on this task, we introduce MMG2Skill-Bench, the first benchmark designed for this problem. We further propose MMG2Skill, a closed-loop framework that compiles guides into editable skills, conditions a fixed vision-language model (VLM) agent on these skills during execution, and revises the skills from trajectory-level root-cause feedback without using benchmark scores. Across GUI control, open-ended gameplay, and strategic card play with six VLM backbones, MMG2Skill consistently outperforms vanilla baseline agents in every model-domain setting, achieving macro-average gains of +12.8 to +25.3 percentage points across backbones. Ablation studies show that directly prompting agents with raw guides can degrade performance, while both structured skill construction and trajectory-driven revision are necessary for the observed improvements. On success-inferable tasks, analyzer-based early stopping further prevents late-stage performance regressions and saves 25%-53% of attempts when the success signal is properly calibrated.
>
---
#### [new 089] Toward Responsible and Epistemically Grounded Multilingual LLMs for Computational Social Science and Humanities
- **分类: cs.CL**

- **简介: 该论文属于多语言大模型评估任务，旨在解决现有评估体系无法涵盖文化背景和解释有效性的不足。提出一个理论框架与实验方法，用于评估多语言模型在社会科学和人文学科中的应用。**

- **链接: [https://arxiv.org/pdf/2606.00596](https://arxiv.org/pdf/2606.00596)**

> **作者:** Wajdi Zaghouani
>
> **摘要:** Large language models have rapidly evolved in multilingual competence and reasoning capacity, enabling their integration into Social Sciences and Humanities research workflows. Yet existing evaluation paradigms remain anchored in task-based NLP benchmarks and fail to address interpretive validity, cultural situatedness, and epistemic mediation. This paper reconceptualizes multilingual reasoning LLMs as hermeneutic instruments that actively structure meaning production across linguistic and cultural contexts. Drawing on hermeneutics, philosophy of technology, science and technology studies, multilingual NLP research, and computational social science methodology, we develop a theoretically grounded framework for evaluating multilingual reasoning in Social Sciences and Humanities (SSH) research. We articulate a rigorous experimental protocol with operationalized metrics for cultural alignment, cross-lingual stability, and reasoning faithfulness, along with transparency requirements tailored to interpretive research tasks. We illustrate the framework through a concrete application scenario involving multilingual political discourse analysis. The paper contributes a conceptual and methodological foundation for responsible integration of multilingual reasoning LLMs into computational social science infrastructures.
>
---
#### [new 090] Training Prompt Matters: State-Adaptive Optimization for Robust Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的模型微调任务，旨在解决训练提示对模型学习效果的影响问题。工作包括揭示提示差异对遗忘和泛化的影响，并提出SAPO优化策略。**

- **链接: [https://arxiv.org/pdf/2606.01967](https://arxiv.org/pdf/2606.01967)**

> **作者:** Wenhang Shi; Yiren Chen; Shuqing Bian; Zhe Zhao; Jinhao Dong; Pengfei Hu; Wei Lu; Xiaoyong Du
>
> **摘要:** While prompt engineering is instrumental in maximizing the capabilities of Large Language Models (LLMs) during inference, the role of prompts during training remains critically underexplored. Prevailing fine-tuning paradigms typically treat training prompts as mere surface forms, assuming that semantically equivalent instructions yield identical learning outcomes. However, we reveal that this equivalence is deceptive: while paraphrased prompts often lead to comparable in-task performance, they induce drastically different cross-task impacts regarding catastrophic forgetting and generalization. Crucially, these impacts are positively correlated across tasks, indicating the existence of superior prompts that consistently yield better performance. Furthermore, we discover that these superior prompts can be robustly identified by task loss prior to learning. Leveraging these insights, we introduce State-Adaptive Prompt Optimization (SAPO), a lightweight yet effective training strategy that shifts task formulation from a static input to a dynamic, state-adaptive variable. Comprehensive experiments on diverse benchmarks confirm its effectiveness, which significantly mitigates forgetting while improving generalization, achieving substantial performance gains over state-of-the-art methods. These results provide insights into how training prompts shape learning dynamics and offer a practical recipe for robust fine-tuning. Our code is available at this https URL.
>
---
#### [new 091] Linguistics-Aware Non-Distortionary LLM Watermarking
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出LUNA，一种语言自适应的水印方法，解决多语言环境下模型输出水印识别问题。通过非破坏性采样和模型无关检测，实现高效水印嵌入与验证。**

- **链接: [https://arxiv.org/pdf/2606.00613](https://arxiv.org/pdf/2606.00613)**

> **作者:** Shinwoo Park; Hyejin Park; Hyeseon An; Yo-Sub Han
>
> **摘要:** Watermarking should identify language-model output without degrading quality or limiting verification to the model provider. Multilingual deployment makes this harder because morphology, segmentation, and script change where watermark evidence can enter naturally. We introduce LUNA, a linguistically adaptive watermark that combines model-free detection with single-token non-distortion under the standard random-key model. LUNA estimates normalized next-tag entropy from part-of-speech contexts in an external corpus and uses it to set the depth of a non-distortionary binary tournament sampler; the detector reconstructs the same schedule from text, a tokenizer, a tagger, and a secret key. We evaluate six typologically diverse languages and two domains against eight primary baselines. LUNA attains an AUROC of 0.9959 and the lowest mean absolute median perplexity shift of 0.045 across the twelve settings; its 95% bootstrap interval [0.022, 0.073] lies below all baseline intervals. LUNA also records the lowest mean Self-BLEU, Distinct-1, surprisal, and entropy shifts. It is the only method that simultaneously achieves AUROC > 0.99 and an absolute median perplexity shift below 0.1 in a majority of settings, reaching this regime in 9 of the 12 settings while no baseline reaches it in more than 2. Our code is available at: this https URL
>
---
#### [new 092] Child-directed speech facilitates production, not comprehension, in BabyLMs
- **分类: cs.CL**

- **简介: 该论文属于语言模型研究任务，旨在解决CDS对BabyLMs语言生成影响的评估问题。通过框架补全任务，发现CDS提升生成能力而非理解能力。**

- **链接: [https://arxiv.org/pdf/2606.01045](https://arxiv.org/pdf/2606.01045)**

> **作者:** Bastian Bunzeck; Sina Zarrieß
>
> **备注:** Accepted at CoNLL 2026
>
> **摘要:** Recent studies suggest that child-directed speech is not conducive to language learning in BabyLMs. However, current evaluations focus predominantly on comprehension and not production, which is central to usage-based theories of language acquisition which argue how CDS facilitates early language use through constructional ''frames'' (frequent lexical patterns with open slots). We introduce a novel generation-based evaluation inspired by such theories in form of a frame-completion task, and compare Llama models trained with CDS, the BabyLM corpus, and web-crawl data (FineWeb-edu) on comprehension benchmarks and our novel framework. Our results reveal a clear dissociation between models' comprehension and production capabilities: while FineWeb-trained models excel at minimal pairs, CDS-trained models produce grammatical completions substantially earlier in training and concentrate probability mass on appropriate slot-fillers. These findings show that comprehension benchmarks underestimate what CDS affords to BabyLMs.
>
---
#### [new 093] ExpWeaver: LLM Agents Learn from Experience via Latent RAG
- **分类: cs.CL**

- **简介: 该论文提出ExpWeaver框架，解决LLM代理经验学习中的效率与架构问题，通过隐空间检索增强生成，提升任务性能与跨领域泛化能力。**

- **链接: [https://arxiv.org/pdf/2606.01041](https://arxiv.org/pdf/2606.01041)**

> **作者:** Tao Feng; Tianyang Luo; Jingjun Xu; Zhigang Hua; Yan Xie; Shuang Yang; Ge Liu; Jiaxuan You
>
> **摘要:** Experience learning has achieved promising results in enhancing LLM agent planning and reasoning by integrating past interactions as reusable knowledge. However, existing methods remain confined to explicit text space, retrieving experiences via semantic similarity and concatenating them into the context window, leading to substantial token overhead and a decoupled architecture that separates retrieval from generation. To address these limitations, we propose ExpWeaver, a framework that enables LLM agents to learn from experience via latent retrieval-augmented generation, without requiring a separate RAG module. ExpWeaver encodes experiences using the LLM's own hidden states, retrieves relevant experiences directly in latent space at each decoding step, and integrates them through cross-attention aggregation and gated residual mechanisms. The entire pipeline is optimized end-to-end with reinforcement learning, supporting both generative and ranking tasks. We evaluate ExpWeaver on 13 diverse tasks spanning question answering, reasoning, coding, scientific prediction, and recommendation. Results demonstrate that ExpWeaver achieves state-of-the-art performance on 12 out of 13 tasks, outperforming the strongest baseline by over 6.8%; maintains token efficiency comparable to non-retrieval baselines while text-based retrieval methods require 1.5 to 2 times more tokens; and exhibits superior cross-domain generalization, outperforming the strongest baseline by 16.32% under zero-shot transfer and 15.21% under few-shot transfer. Our code for ExpWeaver is released at this https URL.
>
---
#### [new 094] TalkTag: Fine-Grained Morphosyntactic Error Annotation for Transcribed Speech
- **分类: cs.CL**

- **简介: 该论文提出TalkTag，用于自动标注口语转录文本的形态句法错误，解决人工标注耗时、依赖专家的问题。**

- **链接: [https://arxiv.org/pdf/2606.01820](https://arxiv.org/pdf/2606.01820)**

> **作者:** Shamira Venturini; Oliver Hennhöfer; Steffen Kinkel; Jannik Strötgen
>
> **摘要:** Fine-grained morphosyntactic error annotation is important in clinical and developmental language research, yet it is labour-intensive, expert-dependent, and difficult to scale. We present TalkTag, an LLM-based lightweight tool fine-tuned to automate CHAT-style error annotation in spoken-language transcripts. Developed under conditions of extreme data scarcity using children's narrative data, the system shows the feasibility of linguistic analysis in low-resource settings. Our evaluation demonstrates that TalkTag produces encouragingly precise annotation while effectively identifying instances where linguistic ambiguity makes automated tagging genuinely complex. In summary, with TalkTag, we provide a scalable alternative to manual error annotation and practically viable support for morphosyntactic error annotation.
>
---
#### [new 095] On the Salience of Low-Probability Tokens for AI-Generated Text Detection: A Multiscale Uncertainty Perspective
- **分类: cs.CL**

- **简介: 该论文属于AI生成文本检测任务，旨在解决统计检测器在应对低概率标记时的稳定性与区分性问题。提出多尺度不确定性估计方法，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2606.02158](https://arxiv.org/pdf/2606.02158)**

> **作者:** Yikai Guo; Bin Wang; Xilai Fan; Wenjun Ke; Haoran Luo
>
> **备注:** Accepted by ICML 2026 main conference
>
> **摘要:** AI-generated text increasingly blends with human writing, raising practical risks such as misinformation, academic misuse, and corpora contamination. While statistical detectors are appealing for efficiency and generalization, they suffer from two key limitations. (i) Boilerplate dominance, boilerplate tokens shared across human and LLM writing can overwhelm discriminative signals. (ii) Brittle point estimates, relying on a single probability score yields unstable decisions under adversarial manipulations. To address these issues, we propose Uncertainty, a multiscale uncertainty estimator that focuses on informative low-probability tokens, which more clearly expose distributional discrepancies. Locally, it alleviates boilerplate dominance by averaging the log-probabilities of low-probability tokens; globally, it reduces brittleness by capturing the distributional shape of this low-probability region via Rényi entropy. We further extend the detector to Uncertainty++ via conditional independent sampling, yielding a more stable uncertainty estimation. Experiments across seven datasets and sixteen LLMs demonstrate high effectiveness, generalization, and robustness. Our code is available at this https URL.
>
---
#### [new 096] DECK: A Consistency x Confidence Taxonomy of LLM Hallucinations
- **分类: cs.CL**

- **简介: 该论文属于大模型幻觉分析任务，旨在解决如何检测幻觉问题。提出DECK分类体系，按可检测性划分幻觉类型，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2606.02289](https://arxiv.org/pdf/2606.02289)**

> **作者:** Mohit Singh Chauhan
>
> **备注:** 18 pages, 3 figures, 5 tables
>
> **摘要:** Existing hallucination taxonomies classify LLM errors by what is wrong with the output -- memorised misconceptions, reasoning failures, fluent fabrications. These taxonomies are useful for diagnosis but cannot answer a different question: which uncertainty scorer would have caught this error? We propose a complementary taxonomy that classifies errors by their detectability signature -- the signal a scorer family would read. The DECK taxonomy is a 2x2 partition along inter-sample consistency and token-level confidence into four behavioural regimes (Drift, Entrenched, Confabulation, Knotted), each mapping to a specific scorer family (or families) that can detect it: black-box consistency scorers have signal in D and C, white-box token-probability scorers have signal in K and C, and only an LLM-as-a-Judge with independent pretraining can detect E. Cell membership is operationalised by a Youden's J optimal split on each scorer axis. Across three models and four datasets we validate the taxonomy two ways: by analysing scorer-pair disagreement, and by checking that external labels (SelfAware unanswerable, HaluEval adversarial, PopQA entity popularity) land in the predicted DECK cells, with model-scale and content-specific secondary-cell refinements. We further identify a universal blind spot of output-level UQ: on knowledge-gap inputs where the generator emits confident, repeatable fabrications, every output-level family collapses by construction. A linear probe on Llama-3-8B's hidden states also collapses to chance, giving preliminary evidence that the failure may persist at the activation level; richer internal-state methods (UQ heads, information-theoretic estimators) remain to be tested.
>
---
#### [new 097] Unlocking the Black Box of Latent Reasoning: An Interpretability-Guided Approach to Intervention
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的可解释性研究，旨在解决大模型推理过程不透明的问题。通过分析隐式推理机制，提出无需参数更新的干预方法，提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2606.01243](https://arxiv.org/pdf/2606.01243)**

> **作者:** Shuochen Chang; Tong Bai; Xiaofeng Zhang; Qianli Ma; Qingyang Liu; Zhaohe Liao; Yibo Miao; Li Niu
>
> **摘要:** Latent reasoning enables Large Language Models (LLMs) to perform multi-step inference within continuous hidden states, offering efficiency gains over explicit Chain-of-Thought (CoT). However, the opacity of these continuous thought vectors hinders their reliability and controllability. This paper bridges the gap between mechanistic interpretability and actionable control. We first present a systematic analysis using structural, causal, and geometric probes, revealing that latent vectors encode compressed, faithful representations of reasoning steps, with early vectors acting as critical causal hubs. Building on this, we operationalize these interpretability insights into a suite of training-free, decode-time interventions that refine the latent reasoning process by imposing the identified geometric and semantic priors. Extensive experiments across multiple model scales and diverse task domains demonstrate that our approaches consistently improve reasoning accuracy. Our interpretability-guided interventions consistently unlock latent capabilities and improve reasoning accuracy without any parameter updates.
>
---
#### [new 098] DrugClaw and DrugAudit: A Primary-Source-Grounded Agent and Authority-Aware Benchmark for Drug-Information Question Answering
- **分类: cs.CL**

- **简介: 该论文聚焦于药物信息问答任务，解决事实幻觉和来源可信度问题。提出DrugClaw系统与DrugAudit基准，提升答案的证据基础和准确性。**

- **链接: [https://arxiv.org/pdf/2606.01434](https://arxiv.org/pdf/2606.01434)**

> **作者:** Qing Wang; Bo Li; Jialu Liang; Daling Shi; Bob Zhang; Qianqian Song
>
> **摘要:** Drug-information question answering is a high-stakes setting where hallucinated facts can mislead clinical decision-making and the provenance of each cited fact matters as much as the fact itself. We present DrugClaw, a multi-agent retrieval-augmented system that queries a registry of drug and pharmacovigilance skills via a reflection-driven state-machine workflow and returns answers grounded in primary regulatory or peer-reviewed records. We also contribute DrugAudit, a 3,772-item authority-aware benchmark with an evaluation panel that scores upstream-of-gold source match, token-level semantic snippet overlap, and citation faithfulness under a dual-judge LLM-as-judge protocol with inter-judge kappa = 0.88 (almost-perfect). Across DrugAudit plus drug-related subsets of MedQA (751) and PubMedQA (512), DrugClaw is top-1 on every column of the headline table: composite Evidence Index under both judges, judge-mediated answer correctness, primary-source rate (0.918, +10.1 pp over next-best), faithfulness (0.887, +5.9 pp), MedQA (0.920), and PubMedQA (0.693).
>
---
#### [new 099] Sandboxed Coding Agents are Competitive Omni-modal Task Solvers
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态任务处理领域，旨在解决传统多模态模型在音频视频任务中表现不足的问题。通过编码代理和工具调用，实现高效任务解决。**

- **链接: [https://arxiv.org/pdf/2606.00579](https://arxiv.org/pdf/2606.00579)**

> **作者:** Dongping Chen; Xuanao Huang; Zhihan Hu; Qingyuan Shi; Dianqi Li; Tianyi Zhou
>
> **备注:** Paper under review
>
> **摘要:** As multimodal LLMs increasingly target video and audio, it is often assumed that such tasks require native omnimodal models. We show that this is not always the case: coding agents with only text+image access and a sandboxed tool-use interface can match, and in several settings outperform, SOTA native omnimodal models and predefined multimodal agent scaffolds across multiple audio-video benchmarks. Our trajectory analysis suggests that their strength comes from writing code and orchestrating tools to extract relevant evidence from transcripts, frames, and other modality signals, thereby converting omnimodal tasks into retrieval and information-processing problems rather than ingesting entire media streams. We further characterize their limitations through a failure taxonomy and process-level trace analysis, and show that simple skill injection, including human-written and self-distilled skills, substantially improves performance. To explore open-source elicitation, we introduce Code-X, a training recipe with the OmniCoding trajectory dataset and verifiable reward, and provide baselines on Qwen-3.5-9B and Qwen-3.6-27B. Finally, we argue that the next frontier is many-modality processing, and introduce TerminalBench-O, a process-level benchmark for real-world omnimodal processing tasks. Code will be available at this https URL.
>
---
#### [new 100] AlphaToken: Decoupling Adaptation and Stability for Path-Aware Response Token Valuation in LLM Post-Training
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AlphaToken，用于大模型后训练中的响应标记估值。任务是提升后训练效果并防止遗忘。通过解耦适应与稳定，结合路径感知机制，优化标记选择。**

- **链接: [https://arxiv.org/pdf/2606.01635](https://arxiv.org/pdf/2606.01635)**

> **作者:** Liu Qing; Ou Wu; Yi Du
>
> **摘要:** Token selection is pivotal for effective LLM post-training. However, existing methods mostly rely on local heuristics and rarely formulate token selection as a principled valuation of individual response tokens. We introduce $\textbf{AlphaToken}$, a response token valuation framework that decouples valuation into $\textbf{adaptation}$ (promoting target-task learning) and $\textbf{stability}$ (preserving pre-trained capabilities), and makes each objective $\textbf{path-aware}$ by combining the direct-path signal from local token gradients with the downstream causal-path signal in autoregressive generation. Since retention data are typically unavailable, AlphaToken approximates stability via a $\textbf{Fisher-drift proxy}$ anchored at the pre-trained reference model. For efficient computation, we extend Ghost Dot-Product to token-level valuation. AlphaToken masks low-value response tokens during fine-tuning and preference optimization, concentrating training signals on more valuable positions. Experiments show that AlphaToken improves post-training performance and mitigates catastrophic forgetting.
>
---
#### [new 101] From Empathy to Personalized Empathy: Adapting Empathetic Strategies to Individual Users
- **分类: cs.CL**

- **简介: 该论文属于个性化情感适应任务，旨在解决用户个性影响 empathetic 策略的问题。通过构建数据集和提出奖励模型框架，提升AI的个性化共情能力。**

- **链接: [https://arxiv.org/pdf/2606.00728](https://arxiv.org/pdf/2606.00728)**

> **作者:** Wuqiang Zheng; Chengbing Wang; Yilin Yang; Junyi Cheng; Jianfei Xiao; Hu Sun; Yi Xie; Yangyang Li; Wenjie Wang
>
> **摘要:** As Large Language Models (LLMs) are increasingly deployed in long-term interactions with users, empathy has become an increasingly important capability. However, existing research overlooks the influence of users' personality traits on empathetic strategies during long-term interactions. To address this gap, we introduce the task of personalized empathy, which focuses on adapting empathetic strategies according to users' personalized characteristics derived from history. To study and enhance this capability, we construct PersonaEmp, a personalized empathy dataset built from long-term user-AI interactions, featuring rich user histories, persona information, and empathy-seeking queries. We further propose PereGRM, a reward modeling framework that combines the empathy evaluation structure with dynamic evaluation criteria generation for fine-grained reward modeling. Experimental results across different settings and multiple judge models show that PereGRM consistently achieves the strongest performance improvements, indicating its effectiveness for enhancing personalized empathetic capabilities.
>
---
#### [new 102] Towards Multidisciplinary Summarization of Hospital Stays: Efficient Sentence-Level Clinical Provenance Categorization
- **分类: cs.CL**

- **简介: 该论文属于临床文本摘要任务，旨在解决多学科临床文本的句级来源分类问题。通过微调大语言模型，提升跨领域摘要的结构化能力。**

- **链接: [https://arxiv.org/pdf/2606.02487](https://arxiv.org/pdf/2606.02487)**

> **作者:** Baris Karacan; Vaibhav Bhargava; Barbara Di Eugenio; Natalie Parde; Mary Khetani; Yu-Shan Tseng; Vanessa Barbosa; Julie Vignato; Lindsey Knake; Rajashree Dahal; Emily Spellman; Danielle Hitzel; Janine Petitgout; Kristi Haughey; Amanda Karstens; Brianna Clarahan; Rachel Dawson; Lauren Boyd; Mackenzie Weis; Angie Tipton; Jaewon Bae; Catherine K. Craven; Karen Dunn Lopez; Andrew D. Boyd
>
> **备注:** 5 pages. Submitted preprint version of a paper accepted to AIME 2026. This version may differ from the camera-ready manuscript and the final Version of Record. The Version of Record will be available from Springer Nature once published
>
> **摘要:** Effective "all-team" summarization in high-complexity settings like the Neonatal Intensive Care Unit (NICU) requires aggregating insights from diverse disciplines (physicians, nurses, therapists) spread across hundreds of clinical free-text notes. Simply pooling heterogeneous text often leads to incoherent outputs. Structured summarization therefore first requires accurate categorization of sentence-level provenance across multi-source notes. This pilot study introduces a clinical provenance categorization pipeline using supervised fine-tuning (SFT) of large language models (LLMs). We adapted two Llama-3 models (8B and 70B) to MedSecId, a corpus of 2,002 MIMIC-III (Adult ICU) notes annotated with clinical provenance headers, achieving in-domain Macro F1 scores above 92% for both models. To evaluate cross-domain generalization, we assessed model capacity (8B vs. 70B) and quantization on a gold-standard dataset of 227 sentence-level spans derived from three multi-disciplinary NICU summaries. Experimental results demonstrate a scale-dependent transfer effect: while SFT produced only marginal changes for the 8B model, it substantially improved the 70B model, increasing Macro F1 by 7%. Notably, the quantized fine-tuned 70B model outperformed its full-precision baseline while substantially reducing computational requirements. These findings suggest that sufficient model capacity is critical for preserving semantic flexibility during cross-domain clinical transfer and that efficient quantized adaptation can enable structured provenance modeling for downstream summarization.
>
---
#### [new 103] Cognitive-Linguistic Indicators of Depression in Online Communities: Analysed by DistilBERT and Holographic Reduced Representation
- **分类: cs.CL**

- **简介: 该论文属于抑郁症检测任务，旨在提升在线文本中抑郁情绪的自动识别。通过结合认知语言特征与DistilBERT和HRR模型，提高了分类效果。**

- **链接: [https://arxiv.org/pdf/2606.00026](https://arxiv.org/pdf/2606.00026)**

> **作者:** Brian Van Steen
>
> **摘要:** This paper investigates whether combining cognitively grounded linguistic features with transformer-based embeddings improves automated detection of depression in online text. Using Beck's Cognitive Theory of Depression, the study extracts cognitive distortions as measurable features, including first-person pronoun density, absolutist words, and negative emotion in Reddit posts from depression-related and control communities. Using a subset of the Kaggle Reddit Suicide and Depression Detection dataset, two classification pipelines are compared, a TF-IDF embedding with Naive Bayes as a baseline, and a hybrid model that concatenates DistilBERT sentence embeddings with Holographic Reduced Representation (HRR) vectors encoding the cognitive-linguistic features, followed by Logistic Regression. The hybrid DistilBERT HRR model achieves a macro F1 score of 0.94 versus 0.80 for the TD-IDF baseline, with 5-fold cross validation F1 improving from 0.83 to 0.92, and AUC from 0.958 to 0.981.
>
---
#### [new 104] French parsing enhanced with a word clustering method based on a syntactic lexicon
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理中的句法分析任务，旨在提升法语解析效果。通过将法语语法词典数据与概率解析器结合，利用动词聚类方法提高解析准确性。**

- **链接: [https://arxiv.org/pdf/2606.00634](https://arxiv.org/pdf/2606.00634)**

> **作者:** Anthony Sigogne; Matthieu Constant; Eric Laporte
>
> **摘要:** This article evaluates the integration of data extracted from a French syntactic lexicon, the Lexicon-Grammar (Gross, 1994), into a probabilistic parser. We show that by applying clustering methods on verbs of the French Treebank (Abeillé et al., 2003), we obtain accurate performances on French with a parser based on a Probabilistic Context-Free Grammar (Petrov et al., 2006).
>
---
#### [new 105] ProactiveLLM: Learning Active Interaction for Streaming Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决Streaming LLM交互延迟问题。通过学习模型内部状态，实现主动交互，提升效率与质量。**

- **链接: [https://arxiv.org/pdf/2606.00523](https://arxiv.org/pdf/2606.00523)**

> **作者:** Junlong Tong; Yao Zhang; Anhao Zhao; Yingqi Fan; Yunpu Ma; Xiaoyu Shen
>
> **备注:** ICML 2026
>
> **摘要:** Standard Large Language Models (LLMs) follow a read-then-generate paradigm, causing unnecessary latency and computation. Streaming LLMs alleviate this issue by generating while receiving inputs, but still struggle to decide when to interact with the stream. Existing methods either hard-code interaction timing or rely on costly external alignment signals, such as timing labels, reasoning trajectories, or stronger teachers. In this paper, we propose ProactiveLLM, which achieves active interaction by leveraging the model's endogenous states to guide interaction decisions. The model first learns to perceive semantic sufficiency from partial inputs through two complementary training mechanisms: mask-based streaming modeling and synchronized privileged self-distillation (SPSD). The former applies monotonic random masking to the input during training, simulating progressively revealed streaming inputs and enabling the model to learn local semantic dependencies from partial-input views. The latter aligns the partial-context student view with a full-context teacher view generated by the same evolving model, allowing privileged full-context evidence to guide the student's understanding under incomplete observations. Together, these mechanisms induce endogenous sufficiency cues without requiring external teachers or annotations, providing a versatile foundation for the plug-and-play integration of diverse decision heads. Extensive evaluation across text and speech streaming tasks confirms that ProactiveLLM significantly reduces interaction latency while maintaining quality, validating its capacity for dynamic and active interaction. Code is publicly available at this https URL.
>
---
#### [new 106] Uncovering Temporal Framing in the News
- **分类: cs.CL**

- **简介: 该论文研究新闻中的时间框架，属于自然语言处理任务。旨在解决如何识别和分析时间相关语言的修辞作用，通过构建标注数据集并测试模型性能。**

- **链接: [https://arxiv.org/pdf/2606.00294](https://arxiv.org/pdf/2606.00294)**

> **作者:** Tarek Mahmoud; Veronika Solopova; Premtim Sahitaj; Ariana Sahitaj; Max Upravitelev; Mervat Abassy; Hana Fatima Shaikh; Neda Foroutan; Vera Schmitt; Preslav Nakov
>
> **备注:** ACL 2026 Main Conference Oral
>
> **摘要:** Temporal language does more than place events on a timeline. In news discourse, references to the past, present, and future can function as rhetorical devices that shape interpretation and persuasion. Here, we study temporal framing, defined as the persuasive use of time-related language to structure meaning rather than to report chronology. We propose a taxonomy of eight temporal frames grounded in prior work on temporality and framing, and we realize it through expert annotation of a multilingual news corpus. The resulting dataset includes 458 English and German news articles, with over 2K temporally framed sentences and approximately 3K temporal framing annotations identified from a corpus of more than 20K sentences. We analyze frame prevalence, co-occurrence patterns, and lexical cues, and evaluate temporal framing detection using supervised fine-tuning and zero-shot classification. Our experiments show that temporal framing is learnable at the sentence level, with supervised models substantially outperforming zero-shot approaches. We publicly release the corpus to support future research on temporal framing: this https URL.
>
---
#### [new 107] "I've Seen How This Goes": Characterizing Diversity via Progressive Conditional Surprise
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于文本多样性评估任务，旨在解决AI生成内容多样性测量问题。提出一种基于上下文学习的度量方法，无需额外模型或数据，直接利用基础模型的对数概率计算多样性得分。**

- **链接: [https://arxiv.org/pdf/2606.01811](https://arxiv.org/pdf/2606.01811)**

> **作者:** Matthew Khoriaty; David Williams-King; Shi Feng
>
> **备注:** 28 pages, 18 figures, 9 tables. Accepted to the Workshop on Generative AI, Creativity, and Human-AI Co-Creation @ ICML 2026 (non-archival). Code and data: this https URL
>
> **摘要:** Measuring the diversity of creative outputs is central to evaluating post-training mode collapse, comparing decoding strategies, and quantifying creative behavior in both AI and human writing. We propose a new approach to measuring diversity using in-context learning, of which the ``Decan'' metric, $D_{Ca_n} = C \times a_n$, is the working instance we evaluate: a per-byte score read off the per-token log-probabilities of a base model $\theta$ in a \emph{single forward pass} per permutation, with no embedding model, no reference corpus, and no human labels. This approach is grounded in information theory, makes use of language model in-context learning to detect a wide range of similarities between any number of inputs, and obviates the need to train a special-purpose model. The same pipeline scores AI samples and human-written response sets, with diversity treated as a property of (responses, prompt, scoring model). On Tevet and Berant's human-grounded McDiv benchmark, $D_{Ca_n}$ reaches OCA 0.846 on the McDiv prompt\_gen set where it performs best, behind the strongest neural baseline reported in Tevet and Berant (SentBERT, 0.897). On the OLMo-2-7B post-training pipeline, $D_{Ca_n}$ drops monotonically across the base $\to$ SFT $\to$ DPO $\to$ RLVR stages, detecting the type of diversity loss that creative-writing applications care about.
>
---
#### [new 108] Worlds Within Words: Translating Culture in Ancient Chinese Texts with Multi-Agent Coordination
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，旨在解决古汉语中文化负载词的翻译问题。通过提出MACAT框架，实现文化信息的精准显性化与翻译优化。**

- **链接: [https://arxiv.org/pdf/2606.01276](https://arxiv.org/pdf/2606.01276)**

> **作者:** Xiaoqi He; Kaixin Lan; Mu You; Tao Fang; Lidia S. Chao; Derek F. Wong
>
> **备注:** The preprint manuscript is 20 pages long and is currently under review
>
> **摘要:** Large language model (LLM)-based machine translation has advanced cross-cultural communication, yet it still struggles with culture-loaded words (CLWs) in ancient Chinese texts. The challenge extends beyond lexical alignment to deciding when and how culture-dependent knowledge should be explicated for readers lacking relevant background. Literal translation often preserves surface forms while missing underlying concepts, whereas over-explicitation harms conciseness and readability. To address this problem, we formulate CLW translation as a selective explicitation task and propose \textbf{MACAT}, a \textbf{M}ulti-\textbf{A}gent \textbf{C}ulture-\textbf{A}ware \textbf{T}ranslation framework that dynamically identifies culturally salient phrases and injects concise explanatory knowledge when necessary. MACAT further incorporates a quality-aware reranking module for candidate selection and a multi-round evaluation agent that assesses translations across terminological precision, readability, fidelity, cultural preservation, and cultural explicitation. Experiments on traditional Chinese medicine (TCM) classics and the \textit{Analects} show that, under a unified GPT-5.4 evaluation setting, MACAT consistently outperforms both the backbone model and general-purpose MT baselines on 100 TCM documents and a 20-chapter subset of the \textit{Analects}.
>
---
#### [new 109] WaveFilter: Enhancing the Long-Context Capability of Diffusion LLMs via Wavelet-Guided KV Cache Filtering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决扩散大语言模型在长文本生成中的计算效率与质量下降问题。通过引入小波变换优化KV缓存，提升长序列处理能力。**

- **链接: [https://arxiv.org/pdf/2606.00724](https://arxiv.org/pdf/2606.00724)**

> **作者:** Jinnan Yang; Yan Wang; Zhen Bi; Kehao Wu; Xiaojie Li; Jungang Lou; Zechao Li; Jing Liu
>
> **备注:** 8 pages,3 figures
>
> **摘要:** Diffusion Large Language Models (DLMs) have demonstrated significant advantages across various tasks. However, constrained by their multi-step iterative inference mechanism, their computational overhead and inference latency in long-context tasks have become core bottlenecks restricting their large-scale deployment. When processing long sequences, existing Key-Value (KV) caching mechanisms often face a dilemma where generation quality degrades drastically, where the core challenge lies in precisely and efficiently filtering critical tokens within ultra-long contexts. Inspired by the human reading process, we propose \textbf{WaveFilter}, a universal and training-free caching framework. This framework innovatively introduces the wavelet transform for decomposition of long sequences to achieve precise identification of key tokens, based on which a sparse KV Cache is constructed to compute the final contextual representation. Experimental results demonstrate that WaveFilter, as a plug-and-play generic framework, significantly enhances the performance of existing mainstream KV Cache methods in complex long-context tasks.
>
---
#### [new 110] SkillAdaptor: Self-Adapting Skills for LLM Agents from Trajectories
- **分类: cs.CL; cs.AI; cs.LG; cs.MA**

- **简介: 该论文提出SkillAdaptor，解决LLM代理在长任务中技能适应不稳定的问题，通过步骤级故障定位实现精准技能更新。**

- **链接: [https://arxiv.org/pdf/2606.01311](https://arxiv.org/pdf/2606.01311)**

> **作者:** Zhuoyun Yu; Xin Xie; Wuguannan Yao; Chenxi Wang; Lei Liang; Xiang Qi; Shumin Deng
>
> **备注:** Work in progress
>
> **摘要:** Large language model (LLM) agents increasingly rely on reusable external skills to solve long-horizon interactive tasks. Existing training-free skill adaptation pipelines usually update skills from full trajectories or session-level feedback, which makes failure attribution coarse and often produces unstable or overly broad revisions. We propose SkillAdaptor, a training-free step-level skill adaptation framework with explicit failure attribution, and it can plug into OpenClaw-class agent harnesses. Given a failed trajectory, SkillAdaptor identifies a first actionable fault step, links responsibility to candidate skills, and applies targeted updates under explicit acceptance checks while keeping the backbone frozen. We evaluate on WebShop, PinchBench, and Claw-Eval with Kimi-K2.5, GLM-5, and GPT-5.2. SkillAdaptor improves over no-skill and skill-adaptation baselines on all three suites, with the largest single-metric improvements of +1.5 points on PinchBench Avg Score%, +1.8 on Claw-Eval Avg Score, and +1.7 on WebShop success rate. These results indicate that step-level attribution supports more stable and auditable training-free skill maintenance\footnote{The code will be released at this https URL.}.
>
---
#### [new 111] Dr. DocBench: A Comprehensive Benchmark for Expert-Level and Difficult Document Parsing
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出Dr. DocBench，一个针对专家级文档解析的基准。解决现有基准覆盖不足、难度低的问题，通过多语言书籍构建，涵盖52个领域，包含4514页高质量标注数据，用于评估和提升文档解析能力。**

- **链接: [https://arxiv.org/pdf/2606.01393](https://arxiv.org/pdf/2606.01393)**

> **作者:** Minglai Yang; Xinyan Velocity Yu; Pengyuan Li; Xinyu Guo; Zhenting Qi; Konwoo Kim; Longtian Ye; Xiaolong Luo; Jinhe Bi; Henry Zhang; Haris Riaz; Xuan Zhang; Yunze Xiao; Bangya Liu; Tom Tang; Yunfei Zhao; Qunshu Lin; Zihan Wang; Minghao Liu; Michael Lingzhi Li; Yilun Du; Jesse Thomason; Rogerio Feris; Alex Pentland; Zexue He
>
> **备注:** 27 pages, 13 figures, 14 tables
>
> **摘要:** Document parsing and recognition are fundamental capabilities for vision-language models (VLMs) and document processing systems. However, existing Optical Character Recognition (OCR) and document parsing benchmarks are increasingly limited in coverage and difficulty: many focus on common document genres or uniformly sampled pages where modern parsers already perform strongly, while offering limited annotation for expert-domain structures such as chemical formula, music notation, complex tables, and cross-page layouts. We introduce Dr. DocBench, a difficulty-aware benchmark for expert-level document parsing. Built from a large-scale multilingual book corpus, Dr. DocBench spans 52 BISAC subject domains and selects challenging documents through parser-failure-based sampling, targeting cases where multiple state-of-the-art systems struggle. It contains 4,514 annotated pages from long documents averaging around 100 pages, with 65k high-quality page- and block-level annotations for layout, reading order, hierarchical relations, and domain-specific visual contents. Evaluations of pipeline-based parsers and general-purpose VLMs show that strong performance on existing benchmarks does not transfer to our expert-level document parsing. Our analysis reveals substantial failures across subjects, content types, and structural attributes, highlighting Dr. DocBench as a comprehensive testbed for diagnosing and advancing document intelligence.
>
---
#### [new 112] EvoPool: Evolutionary Programmatic Annotation for Label-Efficient Specialized Supervision
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出EvoPool，解决标签成本高、专用任务性能差的问题，通过进化框架生成标注代码，提升模型效果。**

- **链接: [https://arxiv.org/pdf/2606.01617](https://arxiv.org/pdf/2606.01617)**

> **作者:** Tianyi Xu; Yaolun Zhang; Xuan Ouyang; Huazheng Wang
>
> **备注:** 39 pages, 7 figures. Code: this https URL
>
> **摘要:** Large language models excel at general tasks but underperform smaller supervised models in specialized, high-stakes domains where training labels are costly. We address this regime with EvoPool, an evolutionary multi-agent framework inspired by Darwinian evolution. Three specialized agents iteratively propose executable annotator code, a small validation set provides a fitness signal, and a deterministic gate keeps only annotators that pass viability, diversity, and marginal-contribution checks across generations. Pool votes are mapped to soft training labels by EvoAgg, a text-aware aggregator combining semantic features with annotator-vote features. The authored pool runs at near-zero per-example cost and is 4500 to 31000x faster than LLM annotation on 100K examples. Across 7 of 8 LLM-weak specialized and complex tasks spanning biomedical relation extraction, legal-clause classification, complex reasoning, and dense multi-label biomedical classification, EvoPool beats the strongest LLM annotation baseline by an average +0.141 macro-F1, peaking at +0.301 on ChemProt and +0.265 on PubMed. Code is available at: this https URL
>
---
#### [new 113] MLLM-Microscope: Unlocking Hidden Structure Within Multimodal Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多模态大语言模型分析任务，旨在揭示模型内部表示特性。通过构建MLLM-Microscope系统，分析了模型的线性、维度和各向异性，为模型设计提供新见解。**

- **链接: [https://arxiv.org/pdf/2606.00909](https://arxiv.org/pdf/2606.00909)**

> **作者:** Ravil Mussabayev; Rustam Mussabayev
>
> **摘要:** This work presents MLLM-Microscope, a novel system designed for analyzing the hidden representations within Multimodal Large Language Models (MLLMs). Our system evaluates the linearity, intrinsic dimension, and anisotropy of multimodal token embeddings across transformer layers. Utilizing the ScienceQA dataset, we evaluate two state-of-the-art MLLMs, LLaVA-NeXT and OmniFusion. We find that both the main and residual streams for tokens of both modalities exhibit highly linear behaviors across transformer layers. However, LLaVA-NeXT's image tokens reveal a slight decline in linearity, whereas OmniFusion's remain consistent. Image token dimensions in OmniFusion remain consistently higher across layers compared to LLaVA-NeXT. Also, the OmniFusion's anisotropy is observed to stay consistently low throughout the layers. These findings suggest that the inner workings of MLLMs highly depend on the nature of modality fusion performed before passing the token sequence into LLM. This and other new potential insights obtainable from our system are surely capable of enhancing our understanding of the inner workings of MLLMs, informing future model design and optimization.
>
---
#### [new 114] SALSA: Speech Aware LLM Adaptation via Learned Steering Activation Vectors
- **分类: cs.CL; eess.AS**

- **简介: 该论文提出SALSA方法，解决语音感知大语言模型在域外设置下泛化能力差的问题。通过学习层间引导向量，提升语音识别性能。**

- **链接: [https://arxiv.org/pdf/2606.00460](https://arxiv.org/pdf/2606.00460)**

> **作者:** Yekaterina Yegorova; Argyrios Gerogiannis; Haolong Zheng; Julia Hockenmaier; Chang D. Yoo; Mark A. Hasegawa-Johnson
>
> **摘要:** Speech-aware large language models often generalize poorly to out-of-domain settings. We propose SALSA (Speech-Aware LLM Adaptation via Learned Steering Activations), a lightweight adaptation method that learns layer-wise steering vectors. Unlike commonly used steering approaches that rely on contrastive activation differences, SALSA directly optimizes steering vectors using a supervised objective. Across children's speech, multilingual speech, and Mandarin-English code-switching benchmarks, SALSA substantially improves performance over zero-shot inference and speech in-context learning baselines, achieving up to 46.8% relative improvements over zero-shot. Analysis further demonstrates that steering the encoder, particularly the later layers, is more effective than steering the LLM backbone. These findings suggest that steering improves downstream ASR performance by adapting higher-level acoustic and phonetic representations to better align with the pretrained language model representation space, rather than by modifying the decoder itself.
>
---
#### [new 115] TrustLDM: Benchmarking Trustworthiness in Language Diffusion Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决语言扩散模型的可信性问题。通过构建基准测试TrustLDM，评估模型在安全、隐私和公平性方面的表现，并提出自动评估框架TrustLDM-Auto。**

- **链接: [https://arxiv.org/pdf/2606.00023](https://arxiv.org/pdf/2606.00023)**

> **作者:** Yichuan Mo; Yukun Jiang; Yanbo Shi; Mingjie Li; Michael Backes; Yang Zhang; Yisen Wang
>
> **摘要:** The rapid development of Language Diffusion Models (LDMs) challenges the dominant position of auto-regressive competitors in language processing. However, their flexible, any-order decoding strategies not only enable fast decoding speed but also potentially bring new trustworthiness challenges. To better understand the risks behind their pipelines, we introduce a comprehensive trustworthiness benchmark tailored to LDMs (TrustLDM), evaluating safety, privacy, and fairness across different LDM architectures with multiple categories of static post contexts. Our empirical results show that although LDMs generally exhibit strong trustworthiness with only the user prompts, their alignment behavior degrades noticeably when the malicious post contexts are attached to the masked responses. We further observe that longer contexts do not necessarily induce stronger effects, and both decoding order and generation length affect the evaluation outcomes. Finally, we propose TrustLDM-Auto, an automatic evaluation framework that leverages LDM decoding flexibility to systematically identify vulnerable configurations, revealing substantial trustworthiness weaknesses across all evaluated models and dimensions. Our work may potentially help the community build more trustworthy LDMs. Our code is available at this https URL.
>
---
#### [new 116] ProbeScale: Probing Analysis to Optimize Neural Scaling Laws for Efficient Small Language Model Inference
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出ProbScale框架，用于优化小语言模型的参数效率。通过结合缩放定律和探针分析，识别关键层以减少参数量，同时保持高性能，解决资源受限下的模型部署问题。**

- **链接: [https://arxiv.org/pdf/2606.01806](https://arxiv.org/pdf/2606.01806)**

> **作者:** Sourav Das
>
> **备注:** 7 pages, 2 figures, ACL
>
> **摘要:** Small Language Models (SLMs) offer a balance between capability and computational feasibility. Neural scaling laws inform their optimal training, suggesting that they possess rich internal representations that scale with their size. However, deploying even these SLMs can be challenging under strict resource constraints. Language model probing provides methods for analyzing the linguistic knowledge encoded in a model's internals. We propose ProbScale, a framework that unifies insights from scaling laws and probing to identify parameter-efficient subnetworks within pre-trained SLMs. ProbScale utilizes the high-quality representations of well-scaled SLMs and uses task-specific probes to mathematically quantify the relevance of each layer for target downstream capabilities. This allows selecting subnetworks that optimally trade off performance against parameter size. We formulate the subnetwork selection as finding a layer subset maximizing aggregated, task-weighted probe performance under a parameter budget. Experiments on representative SLMs such as RoBERTa-Large and T5-Base demonstrate that ProbScale identifies subnetworks achieving significant parameter reduction, from 5 to 10 times, while maintaining high performance (95% to 98% of the original SLMs) on targeted tasks, outperforming heuristic baselines.
>
---
#### [new 117] Which Institutional Frameworks Do Chatbots Assume? Auditing Jurisdictional Defaults in Multilingual LLMs
- **分类: cs.CL**

- **简介: 该论文属于多语言大模型审计任务，旨在解决LLMs在未指定司法管辖区时是否默认使用输入语言作为依据的问题。通过实验分析七款模型在不同语言输入下的回答倾向。**

- **链接: [https://arxiv.org/pdf/2606.00333](https://arxiv.org/pdf/2606.00333)**

> **作者:** Zhizhi Wang; Harini Suresh
>
> **摘要:** LLMs increasingly answer questions about taxes, labor protections, healthcare, education, pensions, and administrative procedures, where usefulness often depends on the applicable jurisdiction. Multilingual users may write in their most comfortable language rather than one associated with the country or region whose rules apply. We ask whether deployed LLMs use input language as a default jurisdictional signal when prompts omit any country or region. Prior multilingual audits show that prompt language can shift cultural, political, or normative outputs; we examine which legal-administrative framework models supply when jurisdiction is underspecified. We evaluate seven LLMs developed in the United States or China on 60 underspecified legal-administrative prompts in English and Mandarin Chinese under three system-prompt conditions, yielding 2,520 manually annotated responses. Across models and conditions, Chinese input more often produces China-specific answers, while English input more often produces U.S.-specific, comparative, or generic answers. Prompts requiring a single answer further increase jurisdiction selection: pooled across models, 74.5% of English-input responses adopt a U.S. framework, while 53.3% of Chinese-input responses adopt a China framework. This directional pattern appears in all seven models. We describe this deployment-level pattern as institutional-framework misselection risk: a fluent answer may rely on a legal-administrative context the user did not intend, especially when their preferred language differs from the relevant jurisdiction. LLM interfaces should not route institutional advice by input language alone; when location is absent, they should request it or state the jurisdictional scope of the answer.
>
---
#### [new 118] A Multi-Domain Red Teaming Framework for Safety, Robustness, and Fairness Evaluation of Medical Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗大模型评估任务，旨在解决安全、鲁棒性和公平性评价不足的问题。通过构建多领域红队框架，对多个模型进行测试与分析。**

- **链接: [https://arxiv.org/pdf/2606.00027](https://arxiv.org/pdf/2606.00027)**

> **作者:** Andrei Marian Feier; Veysel Kocaman; Yigit Gul; Ahmet Korkmaz; Alexander Thomas; Aleksei Zakharov; Jay Gil; Mehmet Butgul; David Talby
>
> **备注:** 10 pages, 4 figures. To be presented at the Text2Story 2026 Workshop (Delft, The Netherlands, 29 March 2026); CEUR Workshop Proceedings (forthcoming). Affiliation: John Snow Labs Inc
>
> **摘要:** Large language models (LLMs) are increasingly deployed across healthcare, yet existing benchmarks fail to capture model behavior under adversarial or ethically complex conditions common in clinical practice. We developed a multi-domain red teaming framework evaluating eleven contemporary LLMs across 690 clinically grounded scenarios spanning nine domains and over 150 subcategories. Scenarios incorporated adversarial transformations, and responses were assessed using a seven-dimension rubric with LLM-assisted scoring and human-in-the-loop validation. Results revealed substantial performance variance, with mean scores ranging from 0.791 to 0.984. Critically, several high-performing systems produced complete failures in individual safety-critical scenarios, demonstrating that aggregate accuracy masks clinically meaningful risk. The highest-performing systems (X-BAI, GPT-5, Claude Opus 4.1) achieved scores above 0.97 with low variance, while performance varied significantly across domains. Equity-related tasks showed 10-20% error amplification with demographic modifications, and human reviewers identified clinically relevant failures missed by automated evaluation. Our findings demonstrate that performance variance and worst-case failures provide more clinically meaningful reliability indicators than mean accuracy alone, and that hybrid evaluation approaches combining automation with clinician oversight are essential for credible safety assessment.
>
---
#### [new 119] AutoForest: Automatically Generating Forest Plots from Biomedical Studies with End-to-End Evidence Extraction and Synthesis
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AutoForest，解决生物医学研究中森林图自动生成问题。通过端到端方法提取数据并合成统计结果，简化系统评价流程。**

- **链接: [https://arxiv.org/pdf/2606.02403](https://arxiv.org/pdf/2606.02403)**

> **作者:** Massimiliano Pronesti; Angelo Miculescu; Mohsin Kapdi; Paul Flanagan; Oisín Redmond; Joao Bettencourt-Silva; Gurdeep Mannu; Spiros Denaxas; Rui Bebiano Da Providencia E Costa; Anya Belz; Yufang Hou
>
> **备注:** Accepted to ACL2026 (System Demonstration Track)
>
> **摘要:** Systematic reviews rely on forest plots to synthesise quantitative evidence across biomedical studies, but generating them remains a fragmented and labour-intensive process. Researchers must interpret complex clinical texts, manually extract outcome data from trials, define appropriate interventions and comparators, harmonise inconsistent study designs, and carry out meta-analytic computations-typically using specialised software that demands structured inputs and domain expertise. While recent work has demonstrated that large language models can extract study-level data from unstructured text, no existing system automates the complete pipeline from raw documents to synthesised forest plots. To address this gap, we introduce AutoForest, the first end-to-end system that generates publication-ready forest plots directly from biomedical papers. Given one or more study papers, AutoForest automatically suggests ICO (Intervention, Comparator, Outcome) elements, extracts outcome data, performs statistical synthesis, and renders the final forest plot. We describe the system architecture, user interface and demonstrate its effectiveness on real-world examples through a user study involving clinicians, showing how AutoForest can accelerate evidence synthesis and substantially lower the barrier to conducting meta-analyses.
>
---
#### [new 120] ResMerge: Residual-based Spectral Merging of Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出ResMerge，解决强化学习专家模型合并问题。通过分离并融合残差与主成分，提升合并稳定性与性能。**

- **链接: [https://arxiv.org/pdf/2606.02252](https://arxiv.org/pdf/2606.02252)**

> **作者:** Yandu Sun; Zhiyan Hou; Haokai Ma; Yuheng Jia; Junfeng Fang; Haiyun Guo; Hongyan An; weizhen wang; Jinqiao Wang
>
> **备注:** 14 pages including appendix
>
> **摘要:** Model merging offers a training-free way to combine multiple post-trained expert models, but merging experts obtained through reinforcement learning (RL) remains challenging. Existing spectral merging methods often assume that leading singular directions contain the main task signal, while lower-energy residual components can be compressed, selected, or attenuated to reduce interference. We find that this assumption does not hold for RL task vectors: after decomposing each task vector into a leading spectral head and a residual component, both parts can independently recover substantial behavior knowledge, while exhibiting different merging properties. The head is highly concentrated and informative but more prone to sharp cross-expert conflicts, whereas the residual component is more dispersed and provides a more stable basis for aggregation. Based on this observation, we propose ResMerge, a residual-based spectral merging framework for RL experts. ResMerge first constructs a stable residual backbone with Spherical Residual Consensus Adaptation, which estimates a reliability-weighted consensus direction on the Frobenius sphere. It then reintroduces leading-head information through a Lightweight Head Correction module gated by positive cross-expert agreement. Experiments across multiple RL expert groups and capability domains show that ResMerge better preserves expert capabilities than representative task-vector and spectral merging baselines. The implementation of ResMerge is publicly available at this https URL.
>
---
#### [new 121] MENTIS: What Belief Changes Under Alignment? Measuring Multi-Scale Latent Torsion in Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于语言模型内部机制研究任务，旨在解决偏好对齐如何影响模型内部结构的问题。通过提出MENTIS框架，分析对齐带来的几何结构变化。**

- **链接: [https://arxiv.org/pdf/2606.01060](https://arxiv.org/pdf/2606.01060)**

> **作者:** Partha Pratim Saha; Samarth Raina; Mayur Parvatikar; Amit Dhanda; Vinija Jain; Aman Chadha; Amitava Das
>
> **备注:** Submitted to EMNLP 2026
>
> **摘要:** Preference alignment has substantially improved the observable behavior of large language models, yet it remains unclear what alignment changes internally. Aligned systems still fail under jailbreaks, prompt injection, and retrieval-time corruption, suggesting behavior-level evaluation alone is incomplete. Post-training should leave measurable traces in internal computation. We ask: when an instruction-tuned (IT) model becomes a preference-aligned (PA) model, what geometric structure changes, where do those changes concentrate, and how selectively do they vary across concepts, prompts, and model families? We introduce MENTIS, a geometry-first framework for measuring alignment-induced internal reorganization in paired checkpoints. MENTIS compares IT and PA models using a primary layerwise covariance-based torsion norm (T1), a secondary spectral torsion diagnostic (T2), and an Energy-Radiance-Activation measure (ERA) for depth localization. Across four 7-8B model pairs on LITMUS, our study reveals that alignment-induced change is selective rather than uniform: normative concepts exhibit larger torsion shifts than factual concepts on average; torsion is negatively correlated with contextual entropy; and peak effects localize to architecture-specific mid-to-late layers. The same pattern appears across word-level, prompt-level, and model-level analyses. These results suggest preference alignment leaves structured, depth-localized geometric signatures in internal computation beyond what behavior-level evaluation alone can reveal.
>
---
#### [new 122] Beyond Topical Similarity: Contrastive Evidence Retrieval with Interpretable Attention Alignment in RAG
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在提升RAG系统的事实性和可解释性。提出CERA框架，通过对比学习和注意力对齐优化证据检索，增强模型解释能力。**

- **链接: [https://arxiv.org/pdf/2606.01482](https://arxiv.org/pdf/2606.01482)**

> **作者:** Francielle Vargas; João Robiatti; Diego Alves; Lucas Pascotti Valem; Maximilian Seeth; Sebastián Ferrada; Ameeta Agrawal; Daniel Pedronette; André Freitas
>
> **摘要:** Ensuring factuality and interpretability in RAG remains an open and urgent problem. We introduce Contrastive Evidence Rationale Attention (CERA), the first retrieval framework to employ subjectivity-based hard negative selection and inject an evidential inductive bias into contrastive learning through an auxiliary attention alignment loss. CERA fine-tunes a dense retriever using two training objectives: triplet-based contrastive learning and interpretable attention alignment, which supervises CLS-to-token attention using a part-of-speech-weighted masking distribution over human-annotated factual rationales as evidence signals. Experiments on a large corpus of clinical trial reports demonstrate that the subjectivity-based hard negative selection substantially improves retrieval effectiveness compared to both Contriever and hard negative selection baselines. Furthermore, rationale alignment improves faithfulness while maintaining competitive retrieval performance, supporting the hypothesis that attention can serve as a more faithful explanation of model behavior when guided by human rationales. Moving beyond topical similarity, CERA enables the retriever to identify the specific tokens that constitute supporting evidence, promoting more interpretable evidence selection in RAG systems.
>
---
#### [new 123] A Primer in Post-Training Reasoning Data: What We Know About How It Works
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理领域，旨在梳理后训练推理数据的研究现状。它总结了150余篇文献，回答数据类型、有效性、构建方法及扩展性问题，为未来研究提供框架。**

- **链接: [https://arxiv.org/pdf/2606.02113](https://arxiv.org/pdf/2606.02113)**

> **作者:** Yaoming Li; Guangxiang Zhao; Qilong Shi; Lin Sun; Xiangzheng Zhang; Tong Yang
>
> **备注:** 22 pages. Project Repository: this https URL
>
> **摘要:** Post-training has become a primary driver of recent progress in large reasoning models, and reasoning data are often the key variable determining whether this stage succeeds. Work on post-training reasoning data has grown rapidly, yet this literature remains scattered across dataset papers, reinforcement-learning recipes, reward-model studies, benchmarks, and frontier system reports. This paper is the first primer to synthesize over 150 key public studies and system reports on post-training reasoning data. We organize the field around four questions: what data objects exist, what makes them useful, how they are constructed, and how they scale. Together, this organization provides an attribution framework for future reasoning-data releases and post-training recipes.
>
---
#### [new 124] IDEAFix: Evaluation Framework for Creative Defixation Prompting in LLMs
- **分类: cs.CL**

- **简介: 论文提出IDEAFix框架，用于评估LLMs在创意问题解决中的发散思维能力。针对现有评价方法不足，通过结构化提示策略分析任务设计对生成原创性解决方案的影响。**

- **链接: [https://arxiv.org/pdf/2606.00875](https://arxiv.org/pdf/2606.00875)**

> **作者:** F. Carichon; S. Sharma; M. Girard; R. Rampa; G. Farnadi
>
> **摘要:** Large language models (LLMs) are increasingly used for tasks involving creative problem solving and idea generation. However, there is a lack of consensus concerning their creative capabilities: some studies report superior performances compared to humans, while others highlight structural limitations such as fixation and the homogenization of outputs. Existing evaluation approaches either rely on narrow, decontextualized tasks that do not capture goal-oriented generation or on broader settings that confound multiple aspects of the creative process, making it difficult to isolate the effects of task formulation, prompting, and evaluation design. Significantly, the role of structured prompting strategies in shaping idea generation remains underexplored. Therefore, we introduce IDEAFix, an evaluation framework for analyzing divergent thinking in open-ended idea generation tasks. We prompt models to generate multiple original solutions to controlled variations of short design scenarios, task attributes, and defixation prompting strategies. This design enables systematic analysis of how structured guidance influences LLMs' idea generation. Our results show that both task formulation and attribute selection significantly affect models' performance, and that simple prompting strategies can boost the originality of solutions. However, we also observe persistent output homogenization across models, confirming inherent limits in their ability to generate diverse solutions. Overall, IDEAFix provides a controlled, extensible framework for studying the mechanisms underlying LLMs' creativity.
>
---
#### [new 125] Toward Robust In-Context Learning: Leveraging Out-of-distribution Proxies for Target Inaccessible Demonstration Retrieval
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的少样本学习任务，旨在解决目标域不可达时演示文摘质量下降的问题。提出DOPA框架，利用OOD代理和马氏距离约束提升检索效果。**

- **链接: [https://arxiv.org/pdf/2606.00014](https://arxiv.org/pdf/2606.00014)**

> **作者:** Hao Xu; Rite Bo; Fausto Giunchiglia; Yingji Li; Rui Song
>
> **备注:** Accepted by ACL 2026 main
>
> **摘要:** Although studies have demonstrated that Large Language Models (LLMs) can perform well on Out-of-Distribution (OOD) tasks, their advantage tends to diminish as the distribution shift becomes more severe. Consequently, researchers aim to retrieve distributionally similar and informative demonstrations from the available source domain to boost the inference capabilities of LLMs. However, in practical scenarios where the target domain is inaccessible, evaluating the unknown distribution is challenging, which indirectly impacts the quality of the selected demonstrations. To address this problem, we propose \textbf{DOPA}, a demonstration search framework that incorporates an OOD proxy to approximate the inaccessible target domain and guide the retrieval process. Building on proxy-based evaluation, DOPA further introduces a Mahalanobis distance-based global diversity constraint to ensure sufficient diversity among the retrieved demonstrations. Experimental results on multiple LLMs and tasks demonstrate that DOPA effectively enhances robustness in OOD settings\footnote{this https URL\_code}.
>
---
#### [new 126] When Rating Scales Fall Short: LLM-Assisted Discovery of ADHD Signals in Turkish Teacher Narratives
- **分类: cs.CL**

- **简介: 该论文属于ADHD诊断任务，旨在解决传统评分工具遗漏教师叙述中的ADHD信号问题。通过分析土耳其教师评估数据，结合结构化评分与文本分析，发现叙述中包含互补信息。**

- **链接: [https://arxiv.org/pdf/2606.02509](https://arxiv.org/pdf/2606.02509)**

> **作者:** Baris Karacan; Irem Aktar Songur; Ahmet Ozaslan; Elvan Iseri
>
> **备注:** 15 pages. Accepted to CLPsych 2026. Camera-ready author version. The final version will appear in the ACL Anthology
>
> **摘要:** Attention Deficit Hyperactivity Disorder (ADHD) is one of the most common neurodevelopmental disorders in childhood, and its diagnosis relies on assessments combining clinician judgment with standardized rating scales and reports from parents and teachers. While structured instruments such as the Conners' Teacher Rating Scale-Revised Short Form (CTRS-R:S) quantify ADHD-related behaviors, teachers also provide open-ended narratives that may contain complementary signals not captured by structured assessments. However, it remains unclear to what extent teacher narratives encode signals overlooked by rating scales. In this study, we analyze de-identified Turkish teacher evaluation forms collected during clinical ADHD assessments, including both CTRS-R:S scores and open-ended teacher narratives. We compare predictive signals from structured scores and narrative text and identify cases where structured assessments fail to clearly distinguish ADHD from non-ADHD students while narrative-based models capture distinct behavioral patterns. Notably, these cases show minimal overlap with those missed by the narrative model, suggesting that structured and narrative information encode complementary signals. To interpret these differences, we apply a large language model (LLM)-assisted theme discovery pipeline that reveals distinct attention, behavioral, and family-related patterns, highlighting the potential of natural language processing (NLP) to uncover clinically relevant signals from teacher narratives and to complement traditional ADHD screening tools.
>
---
#### [new 127] Unveiling the Entropy Dynamics of Chain-of-Thought Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究CoT推理中的熵动态，解决推理效率与可靠性问题。通过分析不确定性与信心区域，提出基于CUSUM的实时控制方法，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2606.02020](https://arxiv.org/pdf/2606.02020)**

> **作者:** Ting Xu; Xu He; Yupu Lu; Jiankai Sun; Dong Li; Wai Lam; Jianye Hao
>
> **备注:** 21 pages, 10 figures, accepted in ICML2026
>
> **摘要:** This paper investigates the entropy dynamics of Chain-of-Thought (CoT) and uncovers a consistent two-phase structure: an Uncertainty Region of exploration transitioning sharply to a Confidence Region of convergence. We demonstrate that the Confidence Region possesses two critical properties: 1) High Reliability -- answers in the confidence region become highly accurate and stable, and 2) High Redundancy -- models generate unnecessary tokens long after reaching the correct answer. These properties unlock more efficient and reliable inference strategies: 1) Early Exit leverages reliability and redundancy to terminate computation safely when returns diminish, and 2)Test-Time Scaling uses the Confidence Region signal to prioritize converged trajectories. To operationalize these insights, we formulate Confidence Region detection as a sequential change-point detection problem, being the first to apply classical change-point methods to monitor CoT reasoning. Using the Cumulative Sum (CUSUM) algorithm, a statistically optimal change-point detector, we develop a training-free framework for real-time inference control. Experiments show our approach establishes a superior Pareto-frontier for early exit. CUSUM achieves 63.06% accuracy with 11.1% token reduction, outperforming DEER and Dynasor by 3.28% and 4.36% in accuracy respectively. For test-time scaling, CUSUM-weighted voting consistently outperforms self-consistency.
>
---
#### [new 128] DeSQ: Decomposition-based SPARQL Query Generation
- **分类: cs.CL**

- **简介: 该论文属于知识库问答任务，解决传统方法在生成查询和直接检索中的不足。提出DeSQ框架，通过分解问题生成结构化SPARQL查询，提升准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2606.00203](https://arxiv.org/pdf/2606.00203)**

> **作者:** Papa Abdou Karim Karou Diallo; Aditya Sharma; Neshat Elhami Fard; Amal Zouaq
>
> **摘要:** Dominant approaches to Knowledge Base Question Answering (KBQA) fall into two categories. First is the generation of a formal query that suffers from brittleness and limited explainability, and the second is direct answer retrieval through KB exploration that is computationally costly and prone to hallucination. To combine the strengths of both paradigms while mitigating their respective weaknesses, we introduce DeSQ (Decomposition-based SPARQL Query Generation), a KB-agnostic framework that operates in three stages. First, it decomposes complex questions into Atomic Constraints (ACs) that mirror the relational structure of the underlying KB. Second, it generates a two-part structured output: (a) Mapping of each AC to its corresponding SPARQL Fragment, using standardized variable and URIs placeholders, and (b) URIs Grounding block describing each placeholder. Third, it assembles these fragments into a complete SPARQL query. DeSQ surpasses state-of-the-art approaches on four out of five major benchmarks and demonstrates superior robustness to lexical variation. Beyond performance gains, our framework greatly simplifies evaluation by eliminating the need for a live KB endpoint, and its structured output enables fine-grained error analysis, allowing more targeted interventions for improvement.
>
---
#### [new 129] PaSBench-Video: A Streaming Video Benchmark for Proactive Safety Warning
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文提出PaSBench-Video基准，用于评估视频中主动安全预警能力。任务是检测视频中的风险并及时发出警告，解决现有基准不考虑时间精度和误报问题。**

- **链接: [https://arxiv.org/pdf/2606.02443](https://arxiv.org/pdf/2606.02443)**

> **作者:** Yusong Zhao; Yuejin Xie; Youliang Yuan; Junjie Hu; Jitian Guo; Yujiu Yang; Pinjia He
>
> **摘要:** Between the first visible sign of danger and the moment an accident occurs, there is often a window where intervention remains possible. Video-capable multimodal large language models (MLLMs) could serve as always-on safety monitors that issue warnings during this window. Yet current benchmarks do not test this ability: they rely on static inputs, ignore timing precision, and omit false-positive measurement on safe scenes. We present PaSBench-Video, a 740-video benchmark with 481 risk and 259 no-risk videos across four domains: driving, healthcare, daily life, and industrial production. Risk videos are annotated with frame-level risk onset and accident boundaries. A model must observe the video causally and produce a warning that is both temporally calibrated and content-correct. Testing 13 MLLMs, we find that no model exceeds 20.0% on our strictest metric, and recall is tightly coupled with false-positive rate, with Pearson correlation 0.64: higher detection comes only at the cost of triggering warnings on the majority of safe clips. Performance splits sharply by domain: models achieve moderate recall at low false-positive rates in daily life, where risks are inherently anomalous, yet fire indiscriminately in driving, where routine and hazardous scenes look alike. These results indicate that current models rely on scene-level activity cues rather than reasoning about emerging harm.
>
---
#### [new 130] Hybrid Verified Decoding: Learning to Allocate Verification in Speculative Decoding
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，解决LLM生成效率低的问题。通过预测缓存草案的接受长度，选择最优验证方式，提升解码速度。**

- **链接: [https://arxiv.org/pdf/2606.01019](https://arxiv.org/pdf/2606.01019)**

> **作者:** Xin Su; Dawid Majchrowski; Fangyuan Yu; Vanshil Atul Shah; Sebastian Rogawski; Pawel Morkisz; Anahita Bhiwandiwalla; Phillip Howard
>
> **摘要:** Large Language Model (LLM) generation remains expensive because autoregressive decoding calls the model once for each new token. Speculative decoding reduces this cost by drafting multiple tokens and verifying them with the target model in one step, but its speedup depends on how many drafted tokens are accepted. Parameter-free draft sources can propose long continuations at low cost in structured and agentic workloads, yet a cache match that looks promising at one generation step may have low payoff at the next. We propose Hybrid Verified Decoding, which predicts the accepted length of a cache draft before verification and uses this payoff estimate to choose between cache verification and a model-based drafter. Across three LLMs and sixteen datasets, Hybrid Verified Decoding is especially effective on agentic workflows, where it outperforms EAGLE3 in every setting with a 2.73x average speedup. Our analysis shows how prompt structure creates cache opportunities, how high-payoff cache drafts concentrate in a small part of the draft space, and how payoff-guided selection reduces sequential decoding work, pointing to runtime draft selection as a promising direction for speculative decoding.
>
---
#### [new 131] DiscourseFlip: An Oblique Discourse-Level Opinion Manipulation Attack against Black-box Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI; cs.CR; cs.IR**

- **简介: 该论文属于安全领域，针对RAG系统提出一种新的攻击方法DiscourseFlip，解决多主题查询下的观点操控问题，通过图引导策略实现有效且隐蔽的攻击。**

- **链接: [https://arxiv.org/pdf/2606.01212](https://arxiv.org/pdf/2606.01212)**

> **作者:** Yuyang Gong; Miaokun Chen; Jiawei Liu; Zhuo Chen; Guoxiu He; Wei Lu; XiaoFeng Wang; Xiaozhong Liu
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems are widely deployed and increasingly influential, but their reliance on external corpora exposes new security risks from poisoned retrieval content. Existing RAG attacks are largely focusing on individual queries or narrow topic-local query sets, which limits their practical reach and offers limited camouflage in real-world settings. In this paper, we introduce discourse-level opinion manipulation, a new threat model in which coordinated influence across a semantic query network induces opinion shifts over a holistic, multi-topic query space. We formalize this threat in a black-box setting and propose DiscourseFlip, an agentic, graph-guided attack that dynamically allocates a limited poisoning budget to maximize discourse-level opinion deviation. Extensive experiments demonstrate that DiscourseFlip consistently induces targeted opinion shifts across the contextualized query network and significantly outperforms existing baselines in terms of coverage and effectiveness. User studies further confirm that DiscourseFlip is effective while remaining well camouflaged from user detection. Moreover, systematic analyses show that existing mitigation strategies are ineffective against discourse-level manipulation, underscoring the urgent need for more robust and adaptive defenses to address discourse-level vulnerabilities.
>
---
#### [new 132] CRAB-Bench: Evaluating LLM Agents under Complex Task Dependencies and Human-aligned User Simulation
- **分类: cs.CL**

- **简介: 该论文提出CRAB-Bench和RUSE，用于评估LLM代理在复杂任务依赖和真实用户行为下的表现，解决现实服务场景评估难题。**

- **链接: [https://arxiv.org/pdf/2606.01815](https://arxiv.org/pdf/2606.01815)**

> **作者:** Danqing Wang; Akshay Sivaraman; Lei Li
>
> **摘要:** Evaluating LLM agents in realistic service scenarios requires complex task dependencies, imperfect user behavior, and an evaluation that accommodates multiple valid solutions. We introduce CRAB-Bench (Constraint-based Realistic Agent Benchmark) and RUSE (Realistic User Simulation Engine) to address this gap. CRAB-Bench generates tasks via a constraint graph over multiple interdependent entities with structured distractors, requiring agents to reason carefully over thousands of misleading candidates where only a tiny fraction of solutions are valid. RUSE replaces cooperative, template-like simulators with realistic users grounded in human behavioral studies, instantiated across diverse personas and four behavioral dimensions. Experiments on four frontier LLM agents show that the best model achieves only 61% pass@1 on CRAB-Bench, and switching to RUSE causes further drops of up to 57%, concentrated in task-solving ability rather than conversational quality. Information Disclosure is the most damaging behavioral dimension, and agents interacting with RUSE are less likely to admit mistakes, instead masking errors through implicit corrections.
>
---
#### [new 133] PMC-InterCPT: Rethinking Biomedical Interleaved Data for Multimodal Continued Pretraining
- **分类: cs.CL**

- **简介: 该论文属于医学多模态持续预训练任务，解决 biomedical 数据质量与模态不平衡问题。构建了PMC-InterCPT数据集，清洗并重构图像-文本样本，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2606.01049](https://arxiv.org/pdf/2606.01049)**

> **作者:** Guanghao Zhu; Zeyu Liu; Zhitian Hou; Pengkai Wang; Zhijie Sang; Minheng Ni; Wenjun Wang; Yanggan Gu; Shuo Cai; Congkai Xie; Jianmin Wu; Hongxia Yang
>
> **摘要:** Large-scale biomedical image-text datasets extracted from scientific literature provide valuable resources for medical multimodal model training. These datasets are commonly organized as image-caption pairs; however, figure captions are often short, context-dependent, and only partially informative without the surrounding article text. At the same time, large-scale automatic extraction introduces structural noise such as missing captions, residual markup, duplicated context, and incoherent multi-paragraph figure descriptions. We revisit data construction for medical multimodal continued pretraining (CPT) and present PMC-InterCPT, a context-grounded biomedical interleaved corpus that incorporates figure-referencing body text in addition to captions. Our pipeline recovers missing captions, cleans caption and context text, reconstructs coherent interleaved image-text samples, and applies LLM-supervised medical relevance and quality classifiers to filter noisy records. We further reveal strong modality imbalance in the resulting corpus and introduce a four-bucket evidence taxonomy for modality-aware resampling. Through CPT followed by supervised fine-tuning (SFT) on Qwen3.5-4B-Base, PMC-InterCPT effectively improves medical and general multimodal performance while using fewer CPT tokens than the raw source pool. The experimental results also illustrate the complementarity between the data quality and modality for medical multimodal CPT.
>
---
#### [new 134] DiffuSent: Towards a Unified Diffusion Framework for Aspect-Based Sentiment Analysis
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于基于方面的情感分析任务，旨在解决多词方面和观点项的边界敏感问题。提出DiffuSent框架，通过扩散过程统一处理所有子任务，提升准确性和效率。**

- **链接: [https://arxiv.org/pdf/2606.01323](https://arxiv.org/pdf/2606.01323)**

> **作者:** Shu Long; Yanglei Gan; Xuchuan Zhou
>
> **摘要:** Aspect-Based Sentiment Analysis (ABSA) encompasses seven distinct subtasks, each focusing on different extracted elements. Despite the proven success of generative models in unified aspect sentiment analysis, existing approaches often rely on auto-regressive token-by-token generation without grasping the whole information of the aspect and opinion terms, resulting in boundary insensitivity, particularly in context of multi-word aspect and opinion terms. To address these issues, we present DiffuSent, a non-auto-regressive diffusion framework that systematically formulates all ABSA subtasks as boundary denoising diffusion processes, progressively refining boundaries over noisy states. Furthermore, we introduce a contrastive denoising training strategy which effectively address duplicate predictions with subtle variations introduced by diffusion process. Extensive experiments across 28 settings (7 subtasks x 4 datasets) demonstrate that DiffuSent achieves delivers consistent improvements over the strongest generative and span-based systems. DiffuSent exhibits notable gains on multi-word triplets, achieving an average improvement of +2.48 F1, and maintains robust extraction accuracy in sentences containing multiple sentiment triplets. Moreover, the non-auto-regressive decoding enables substantial efficiency benefits, reaching up to 181 times faster inference than auto-regressive generative baselines
>
---
#### [new 135] CSRP: Chain-of-Thought Reasoning for Chinese Text Correction via Reinforcement Learning with Efficiency-Aware Rewards
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对中文语法纠错任务，解决模型过纠正和精度不足问题，提出CSRP框架，结合持续预训练、思维链微调和效率感知强化学习，提升纠错效果。**

- **链接: [https://arxiv.org/pdf/2606.00020](https://arxiv.org/pdf/2606.00020)**

> **作者:** Wei Tian; Yuhao Zhou; Man Lan
>
> **备注:** Accepted to the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026, Main conference)
>
> **摘要:** Large Language Model (LLM) based Chinese Grammatical Error Correction (CGEC) systems face two critical challenges: general-purpose models lack specialized linguistic priors for subtle grammatical distinctions, and Supervised Fine-Tuning (SFT) with Maximum Likelihood Estimation fails to optimize for precision-focused metrics, leading to systematic over-correction. We propose CSRP, a three-stage framework that progressively builds correction capability through Continual Pre-training (CPT) on 5.9M balanced samples to internalize domain knowledge, Chain-of-Thought SFT with explicit error reasoning for diagnostic transparency, and Group Relative Policy Optimization with a novel Efficiency-Aware Reward that explicitly penalizes unnecessary edits. On the NACGEC benchmark, CSRP achieves state-of-the-art performance with 50.99 $F_{0.5}$ and 57.17 precision, substantially outperforming previous best results while effectively mitigating the over-correction bias inherent in MLE-trained models. Our method also advances CSCD spelling correction to 59.61 F1, surpassing GPT-4 by 5.20 points. Comprehensive ablation studies demonstrate that the RL alignment stage contributes a 8\% relative gain over the SFT baseline, and that this gain is orthogonal to the contribution of large-scale CPT, validating that explicit optimization for edit efficiency is essential for high-quality grammatical error correction. Our code is available at this https URL.
>
---
#### [new 136] Challenger at MultiPRIDE: Is It Hate Speech or Reclaimed?
- **分类: cs.CL**

- **简介: 该论文属于 hate speech 检测任务，旨在区分真实仇恨言论与被重新使用的语言。通过生成语义嵌入和过滤噪声标签，结合分类模型实现准确分类。**

- **链接: [https://arxiv.org/pdf/2606.01298](https://arxiv.org/pdf/2606.01298)**

> **作者:** Hadi Bayrami Asl Tekanlou; Mahdi Bakhtiyarzadeh; Jafar Razmara
>
> **备注:** 9 pages, 2 figures, Published in EVALITA 2026, CEUR Workshop Proceedings Vol. 4195
>
> **摘要:** The spread of hate speech has become increasingly harmful in modern digital environments, particularly on social networking platforms. While recent advances have shown promising results in automatic hate speech detection, a key challenge remains: distinguishing genuine hate speech from reclaimed language. Accurate labeling is difficult due to the nuanced and context-dependent nature of reclaimed expressions. In this paper, we present a simple and interpretable approach for distinguishing hate speech from reclaimed language, developed for the MultiPride Shared Task. Our method generates dense semantic text embeddings and incorporates a label-noise filtering stage using Cleanlab with logistic regression, followed by a Multi-layer Perceptron (MLP) neural network for final classification. The system is designed to operate under limited computational resources while maintaining strong performance. We evaluate our approach using precision, recall, and F1-score, including macro-averaged metrics. Experimental results demonstrate robust performance despite extreme class imbalance in the dataset. Overall, the findings highlight the potential for further improvements through larger embedding models and more advanced preprocessing techniques while preserving interpretability.
>
---
#### [new 137] When Is 0.1% Enough? Analyzing the Combined Effects of Dimensionality Reduction and Quantization on Text Embedding Compression
- **分类: cs.CL**

- **简介: 该论文属于文本嵌入压缩任务，旨在解决高维向量存储与计算成本高的问题。通过结合降维与量化方法，实现更高效的嵌入压缩。**

- **链接: [https://arxiv.org/pdf/2606.01074](https://arxiv.org/pdf/2606.01074)**

> **作者:** Riku Kisako; Hayato Tsukagoshi; Ryohei Sasano
>
> **摘要:** Recent high-performing text embedding models often output high-dimensional real-valued vectors, resulting in substantial storage and computational costs. To address this issue, compression methods based on dimensionality reduction or quantization have been proposed; however, the effects of combining dimensionality reduction and quantization have not been sufficiently investigated. In this paper, we systematically examine the effectiveness of compressing text embeddings by combining dimensionality reduction and quantization, using four MTEB task families and four pretrained embedding models. The experimental results demonstrate that combining dimensionality reduction and quantization enables substantially stronger compression than using either method alone, that in some settings embeddings can be reduced to as little as 0.1% of their original size with almost no performance degradation, and that the optimal compression strategy depends on the task.
>
---
#### [new 138] WAXAL-NET: Finetuned Edge ASR Across 19 African Languages
- **分类: cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于语音识别任务，旨在提升19种非洲语言的语音转文字性能。通过微调小型模型，显著优于大型多语言模型，验证了领域专精的重要性，并发布相关资源促进后续研究。**

- **链接: [https://arxiv.org/pdf/2606.02375](https://arxiv.org/pdf/2606.02375)**

> **作者:** Victor Tolulope Olufemi; Oreoluwa Babatunde; Ramsey Njema; Bolarinwa Gbotemi; Wanchi Lucia Yen; John Uzodinma; Sunday Ajayi; Oluwademilade Williams; Kausar Moshood; Innocent Elendu Anyaele; Akebert Arefaine; Candace Hunzwi; Wongel Dawit Daniel; Emmilly Namuganga; Cleophas Kadima; Athanase Bahizire; Onitsiky Ranaivoson; Emmanuel Aaron; Nicholaus Ladislaus; Idris Muhammed; Jonathan Enoch Simenya; Martin Koome; Matewos Tegete Endaylalu; Peter Ifeoluwa Adeyemo; Hondi Prisca Birindwa; Ukachi Agnes Eze-Mbey; Yacoba Oduro-Yeboah; Pericles Adjovi; Mikel K. Ngueajio; Toluwani Aremu; Prasenjit Mitra
>
> **摘要:** We evaluate whether compact domain-specialized ASR models can outperform massively multilingual foundation models for conversational African speech across 19 languages in the WAXAL corpus. Fine-tuned edge models achieve a macro-averaged WER of $38.0\%$ compared to $64.9\%$ for the best zero-shot baseline, a $26.9$ percentage-point reduction using models $3-40\times$ smaller. Results confirm that domain specialization dominates scale for spontaneous African speech. Cross-domain evaluation shows that fine-tuned models recover usable performance on out-of-distribution (OOD) speech, while zero-shot models regain an advantage when the test domain matches their pretraining distribution. A distributed native-speaker audit across all surveyed languages produces a linguistically-grounded error taxonomy, showing that CTC and autoregressive architectures behave differently across language families. We further show that WER alone misrepresents performance for syllabary-script languages where CER/WER ratios reveal substantially higher character-level accuracy than headline WER suggests. Finally, to contribute to future African ASR research, we release all model weights, fine-tuning and evaluation scripts, and a cleaned WAXAL subset covering all $19$ languages.
>
---
#### [new 139] TukaBench: A Culturally Grounded Jailbreak Benchmark for African Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型安全评估任务，旨在解决LLM在非洲语言中的安全评估不足问题。通过构建TukaBench基准，测试不同文化语境下的提示对模型拒绝率的影响。**

- **链接: [https://arxiv.org/pdf/2606.01322](https://arxiv.org/pdf/2606.01322)**

> **作者:** Victor Akinode; Senyu Li; Wassim Hamidouche; Waqas Zamir; Inbal Becker-Reshef; David Ifeoluwa Adelani
>
> **备注:** Under review
>
> **摘要:** Safety evaluation of Large Language Models (LLMs) remains heavily English-centric, leaving Low-Resource Languages (LRLs), particularly African ones, critically underexplored. We introduce TUKABENCH, a jailbreak benchmark for seven African languages that extends JailbreakBench (JBB) beyond direct translation through four settings: human translation of JBB prompts, English adaptation to African contexts followed by human translation, human-curated prompts validated through interactions with GPT-5.2, and code-switched prompts combining English and African languages, isolating the effect of language, cultural grounding, and prompt evasiveness on model safety. Across closed and open models, prompting in African languages reduces refusal relative to English, with culturally adapted prompts leading to least refusal. The evaluation also surfaces two structural limitations: model comprehension failures and reduced LLM-as-a-judge reliability in LRLs. To capture the first, we introduce Deflection alongside Refused and Jailbroken; to assess the second, we validate outputs with human annotations, showing that judge-human agreement drops in lower-resource languages and less commonly supported scripts.
>
---
#### [new 140] When Meaning Travels: A Granular Lens on Hybrid-MoE's Role in Idiomatic Understanding for Language Models
- **分类: cs.CL**

- **简介: 该论文属于多语言语义理解任务，旨在解决低资源语言中习语的跨语言建模难题。通过构建多模态习语语料库并提出HybridMoE框架，提升习语意义保留与跨语言迁移能力。**

- **链接: [https://arxiv.org/pdf/2606.01671](https://arxiv.org/pdf/2606.01671)**

> **作者:** Sarmistha Das; Vaibhav Vishal; Shreyas Guha; Amaan Ali; Kitsuchart Pasupa; Sriparna Saha
>
> **摘要:** In the contemporary epoch of multilingual education, learning idioms provides a fascinating gateway towards creativity, cultural values, historical context, and diverse perspectives inherent to various linguistic traditions. This paper showcases the navigation of retaining figurative and cultural semantics in low-resource Southeast Asian languages such as Hindi, Bengali, and Thai, where culturally rich idioms pose significant obstacles for computational modeling and cross-linguistic transfer due to their deep metaphorical complexity. To tackle such complexity, we present Varnika, a reconstructed multimodal idiom corpus comprising 3,533 multilingual idioms, enriched with seven idiomatic tones aligned with both textual and visual representations. Additionally, to infer informative idiomatic understanding, we introduce a Hybrid Mixture-of-Experts (HybridMoE) framework that embeds multiple idiomatic expert opinions while mitigating expert sparsity by integrating outputs from both selected and unselected experts through controlled hybridization, further augmented with Idiomatic Property Signals via masked multimodal embeddings. To analyze the performance across multiple dimensions, we propose the IDIO-TONE and Idiomatic Validation Score, a three-stage evaluation pipeline measuring (i) literal translation fidelity, (ii) visual-semantic alignment, and (iii) idiomatic meaning retention. Empirical evaluations highlight that HybridMoE achieves 5--6\% performance gains across advanced vision language models, demonstrating improved representation of figurative language and culturally embedded meaning in multilingual multimodal settings
>
---
#### [new 141] Robust Reasoning via Dynamic Token Selection for Distribution-Aligned Self-Distillation
- **分类: cs.CL**

- **简介: 该论文提出DASD方法，解决自蒸馏中风格偏差问题，通过动态筛选token提升模型推理鲁棒性。任务为增强生成模型的逻辑推理能力。**

- **链接: [https://arxiv.org/pdf/2606.00628](https://arxiv.org/pdf/2606.00628)**

> **作者:** Ruiqi Zhang; Lingxiang Wang; Hainan Zhang Zhiming Zheng
>
> **备注:** 12 pages, 13 figures
>
> **摘要:** Self-distillation improves learning efficiency by rewriting reference answers as training data that better matches the model's own distribution. However, reference answers also introduce strong stylistic biases, causing the generative model to imitate surface forms rather than learn useful reasoning patterns. We observe that the rewriting data contains a large number of high-perplexity (PPL) tokens, coming from two distinct sources: beneficial knowledge-enhancing logical corrections, and harmful stylistic drift induced by reference imitation. Treating all such tokens equally can disrupt the base model's original distribution and degrade performance, especially on difficult reasoning tasks. To address this, we propose Distribution-Aligned Self-Distillation (DASD), which uses an answer-aware reference model to generate candidate tokens and dynamically filters them according to the base model's confidence. DASD preserves tokens that encode useful logical knowledge while suppressing distributionally misaligned style noise. Experiments on math, code, and commonsense reasoning benchmarks show that DASD consistently outperforms competitive baselines, reduces high-PPL tokens, and improves robustness across tasks of varying difficulty.
>
---
#### [new 142] TimeSage-MT: A Multi-Turn Benchmark for Evaluating Agentic Time Series Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出TimeSage-MT，一个用于评估时间序列推理的多轮基准，解决LLM在复杂对话中处理时间序列数据的问题。通过构建多轮对话数据集，评估模型在记忆、不确定性处理等方面的表现。**

- **链接: [https://arxiv.org/pdf/2606.01498](https://arxiv.org/pdf/2606.01498)**

> **作者:** Yaxuan Kong; Qingren Yao; Yuqi Nie; Yichen Li; Yilei Shao; Stefan Zohren; Anna Vettoruzzo; Joaquin Vanschoren; Ming Jin; Qingsong Wen
>
> **摘要:** Time series data inform critical decisions across many real-world domains. While large language model (LLM) agents can analyze data through natural language and tools, it remains unclear whether they can conduct reliable time series analysis across multi-turn conversations. Existing benchmarks focus on single-step tasks such as forecasting and anomaly detection, overlooking practical workflows where user goals evolve, agents must build on prior analyses, and conclusions emerge from accumulated evidence. In this work, we introduce TimeSage-MT, a multi-turn benchmark for agentic time series reasoning with 240 tasks and 2,680 dialogue turns across 8 real-world domains, spanning basic exploration to decision-oriented analysis. TimeSage-MT is built through a reproducible pipeline that converts real-world time series data into multi-turn conversations with verifiable answers. It provides a unified evaluation protocol and public leaderboard for comparing time series agentic systems. To demonstrate the benchmark's utility, we evaluate frontier LLMs alongside TimeSage, a novel structured agent equipped with a comprehensive time series skill library. The results show sharp performance drops on decision-oriented tasks, driven by failures in memory, uncertainty handling, and domain-based decision making. TimeSage-MT exposes critical gaps in current agentic reasoning and provides a rigorous foundation for future development.
>
---
#### [new 143] SimSD: Simple Speculative Decoding in Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型加速任务，解决扩散模型无法使用token级推测解码的问题。提出SimSD方法，通过掩码策略使扩散模型支持高效推测解码，提升推理速度且保持生成质量。**

- **链接: [https://arxiv.org/pdf/2606.02544](https://arxiv.org/pdf/2606.02544)**

> **作者:** Junxia Cui; Haotian Ye; Runchu Tian; Hongcan Guo; Jinya Jiang; Haoru Li; Chaojie Ren; Yiming Huang; Kaijie Zhu; Zhongkai Yu; Kun Zhou; Jingbo Shang
>
> **备注:** 13 pages, 4 figures, code available at this https URL
>
> **摘要:** Diffusion large language models (dLLMs) have recently emerged as a promising alternative to autoregressive (AR) LLMs, offering faster inference through parallel or blockwise decoding. However, their masked language modeling formulation remains incompatible with standard token-level speculative decoding, one of the most effective acceleration techniques for AR models. In AR decoding, the causal mask preserves temporally valid token-level contexts, enabling a target model to verify multiple drafted tokens in a single forward pass. In contrast, dLLMs rely on mask tokens and bidirectional attention, causing the effective context to change across denoising steps and preventing direct token-level speculative verification. To bridge this gap, we propose a simple but effective speculative decoding algorithm for diffusion language models, named SimSD, which mainly adopts a plug-and-play masking strategy that equips dLLMs with temporally valid token-level contexts for speculative decoding. Our method explicitly introduces reference tokens from draft-model predictions and designs an attention mask that regulates their interaction with current-step tokens, allowing dLLMs to compute valid logits for drafted tokens in a single forward pass. This restores the key verification ability provided by causal masking in AR models while preserving the parallel decoding advantages of dLLMs. The proposed method is training-free and can be flexibly integrated with other acceleration techniques such as KV cache and blockwise decoding. Experiments on SDAR-family dLLMs across four benchmarks show that our method achieves up to 7.46x higher decoding throughput while maintaining and even improving average generation quality.
>
---
#### [new 144] Implicit Geographic Inference in LLM Medical Triage: Language-Driven Disparities in Emergency Recommendations
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于医疗分诊任务，研究LLM因语言差异产生不同的紧急建议，分析语言对模型决策的影响。**

- **链接: [https://arxiv.org/pdf/2606.01204](https://arxiv.org/pdf/2606.01204)**

> **作者:** Qi Han Wong
>
> **备注:** 7 pages, 4 tables. Code and data at this https URL
>
> **摘要:** We investigate whether large language models produce different medical triage recommendations for identical symptoms based solely on the language of the patient prompt. Using Gemini 3.5 Flash, we evaluate a neurological symptom profile (persistent headache, blurred vision, nausea) across six languages (English, Spanish, Chinese, Hindi, Japanese, Arabic) with 30 runs per condition (n=450 total API calls). We find that the model recommends emergency room visits at rates ranging from 0% (Japanese, Hindi) to 30% (English, Arabic), despite assigning nearly identical severity scores (7.7-8.0/10) across all languages. Adding a single sentence specifying the patient's US location increases ER recommendations by up to 76.7 percentage points for non-English prompts, while the reverse anchor (English prompt with a Tokyo location) reduces the ER rate from 30% to 6.7%. A back-translation control (Japanese to English) produces ER rates comparable to the English baseline, confirming that the disparity is not caused by translation quality but by implicit geographic inference from the input language. We release the complete dataset, experiment code, and results.
>
---
#### [new 145] Learning When to Translate for Multilingual Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于多语言推理任务，旨在解决非英语输入中语言理解不足的问题。通过提出Luar框架，模型可选择性调用翻译，提升推理效果。**

- **链接: [https://arxiv.org/pdf/2606.02465](https://arxiv.org/pdf/2606.02465)**

> **作者:** Deokhyung Kang; Hyounghun Kim; Gary Geunbae Lee
>
> **备注:** preprint
>
> **摘要:** Reasoning language models (RLMs) achieve strong performance on complex reasoning tasks, but still exhibit substantial multilingual reasoning gaps, largely due to language-understanding failures in non-English inputs. English translation can mitigate these failures by expressing non-English inputs in a form that RLMs can more reliably interpret, yet translating every input is unnecessary when the model can reason reliably from the original query. To address this challenge, we propose Luar, a Language Understanding Boundary-aware Reinforcement Learning framework that trains RLMs to selectively invoke translation when direct understanding is unreliable. Luar trains the model to choose between solving the original input directly and reasoning over its English translation, encouraging translation only when translator-augmented reasoning is expected to substantially outperform direct reasoning. Across multilingual reasoning benchmarks, Luar outperforms standard GRPO and other training-based baselines, with particularly large gains on low-resource languages. Further analysis shows that Luar avoids unnecessary translation in cases where direct reasoning is sufficient, while extending its translator-call behavior to unseen low-resource languages. Together, our work suggests a selective approach to multilingual reasoning: RLMs can learn to invoke translation only when their direct understanding is unreliable. The project will be made publicly available at this https URL
>
---
#### [new 146] Why Do Self-Harm Prediction Models Struggle to Generalise? Lexical and Semantic Variations in Emergency Department Triage Notes
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的文本分类任务，旨在解决自伤模型泛化能力差的问题。通过分析不同医院的急诊记录，发现词汇和语义差异影响模型性能，提出改进泛化的方法。**

- **链接: [https://arxiv.org/pdf/2606.01678](https://arxiv.org/pdf/2606.01678)**

> **作者:** Liuliu Chen; Mike Conway; Jo Robinson; Vlada Rozova
>
> **备注:** Accepted to CLPsych2026
>
> **摘要:** Self-harm presentations to emergency departments (EDs) are strongly associated with higher suicide risk. NLP models have shown robust performance in detecting self-harm from triage notes within single hospitals, yet performance often declines across institutions. To examine potential causes, we compare ED triage notes from two hospitals by analyzing lexical characteristics, highly associated predictive features, and salient topics. Our results reveal variation in lexical expression and feature importance related to self-harm across hospitals, despite consistent core themes such as self-poisoning and self-injury. These documentation differences are associated with reduced cross-site performance. Our findings provide insight into how institutional variation affects the identification of self-harm in clinical text and highlight potential methods to improve model generalisability.
>
---
#### [new 147] RealityTest: How People Probe AI Identity and Whether Models Disclose It
- **分类: cs.CL**

- **简介: 该论文属于AI安全评估任务，旨在测试AI系统在被询问时是否披露身份。研究构建了多模态、多语言的RealityTest基准，分析人类实际提问方式及模型披露行为。**

- **链接: [https://arxiv.org/pdf/2606.00168](https://arxiv.org/pdf/2606.00168)**

> **作者:** Anna Gausen; Sarenne Wallbridge; Bessie O'Dell; Christopher Summerfield; Hannah Rose Kirk
>
> **备注:** 9 pages, 4 figures
>
> **摘要:** AI systems are increasingly deployed in conversational settings where users may be uncertain whether they are speaking with a human or an AI. Despite mounting regulatory attention to this known safety risk, existing evaluations of AI disclosure are typically English-only, based on machine-generated questions, and restricted to text. We present RealityTest to comprehensively test whether AI systems disclose their identity when asked. The benchmark is the first large-scale multimodal and multilingual evaluation, grounded in human data on how people actually encounter and question AI identity in the real-world. Alongside the benchmark, we release the underlying dataset of 3,152 identity-probing queries collected from ~750 participants across 49 countries and five languages, in text and speech scenarios. We find that only 31% of people ask about identity directly in ambiguous scenarios, and that the questions people ask are far more diverse than machine-generated queries. We test 17 text and 6 speech models, and find substantial variation in disclosure behaviour. However, a single suppression instruction reduces disclosure rates to below 30%, even in the best-performing models. Validating our investment in diverse, human-grounded evaluation data, we find that how the question is phrased and the context of the conversation matter more for disclosure than which model is being tested. Safety evaluations built on narrow or synthetic query sets risk mischaracterising how models behave in realistic deployment settings.
>
---
#### [new 148] OCC-RAG: Optimal Cognitive Core for Faithful Question Answering
- **分类: cs.CL**

- **简介: 该论文提出OCC-RAG，一种专注于准确问答的小型语言模型，解决多跳推理和上下文忠实性问题，通过大规模数据训练提升问答质量。**

- **链接: [https://arxiv.org/pdf/2606.00683](https://arxiv.org/pdf/2606.00683)**

> **作者:** Maksim Savkin; Mikhail Goncharov; Alexander Gambashidze; Alla Chepurova; Dmitrii Tarasov; Nikita Andriianov; Daria Pugacheva; Vasily Konovalov; Andrey Galichin; Ivan Oseledets
>
> **摘要:** Recent progress in the development of language models has been defined by scale, with each generation absorbing more of the world's knowledge into its weights. However, many practical applications benefit more from robust reasoning than from extensive parametric knowledge. In this setting, task-specialized small language models (SLMs) offer a principled design choice. We introduce Optimal Cognitive Core (OCC), a family of SLMs built around this premise. As a variant of OCC, we present OCC-RAG, optimized for faithful question answering (QA) grounded in the provided context. This task directly aligns with the OCC design approach, requiring multi-hop reasoning over supplied passages while ignoring memorized knowledge. To train OCC-RAG, we implement a novel pipeline for synthesizing multi-context, multi-hop QA data at scale, producing a corpus of over three million examples targeting multi-hop reasoning, strict context faithfulness, and calibrated abstention. We release OCC-RAG-0.6B and OCC-RAG-1.7B, both mid-trained on this corpus. The models produce structured reasoning traces with source citations grounded in literal quotes from the context. Through OCC-RAG, we demonstrate that compact, task-specialized SLMs can match or exceed general-purpose models 2 -- 6x their size across multi-hop reasoning (HotpotQA, MuSiQue, TAT-QA), faithfulness (ConFiQA), and refusal (MuSiQue-Un) benchmarks.
>
---
#### [new 149] CARTE: A Benchmark for Mapping Language Model Knowledge Across France
- **分类: cs.CL**

- **简介: 该论文提出CARTE基准，用于评估大语言模型在法国区域知识上的细粒度推理能力，解决跨地区文化差异识别问题。**

- **链接: [https://arxiv.org/pdf/2606.01995](https://arxiv.org/pdf/2606.01995)**

> **作者:** Sarah Almeida Carneiro; Christos Xypolopoulos; Xiao Fei; Yang Zhang; Michalis Vazirgiannis
>
> **摘要:** We introduce CARTE 1 (Culturally Anchored Regional-Territorial Evaluation), a multiplechoice benchmark for evaluating the ability of large language models (LLMs) to perform fine-grained reasoning over geographically grounded and regionally differentiated knowledge within France. While prior benchmarks focus on national-level cultural understanding, they largely overlook intra-country variation and the need to distinguish between closely related regional contexts. CARTE addresses this gap by introducing 2,431 questions spanning the 13 metropolitan regions of France and covering 14 thematic domains, including culture, language, demographics, economy, environment, and mobility. We further introduce CARTE-LV, a subset targeting Linguistic Variation across French regions, enabling focused evaluation of language-related differences. We evaluate 27 LLMs ranging from 1B to 12B parameters under few-shot settings. Our experiments reveal performance disparities across regions and model scales, suggesting systematic gaps in pretraining coverage and limited robustness to intra-national variation.
>
---
#### [new 150] Low-Resource Safety Failures Are Action Failures, Not Representation Failures
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于安全对齐任务，解决低资源语言安全失效问题。发现失效源于决策校准而非表示，通过重新校准高资源模型的门控机制提升低资源语言的安全拒绝能力。**

- **链接: [https://arxiv.org/pdf/2606.01196](https://arxiv.org/pdf/2606.01196)**

> **作者:** Rashad Aziz; Ikhlasul Akmal Hanif; Fajri Koto
>
> **摘要:** Safety alignment learned in high-resource languages transfers poorly to low-resource languages. Models refuse harmful prompts in English but fail to refuse when the same prompts are translated into Swahili or Burmese. Adaptive steering methods like AdaSteer and CAST inherit this failure cross-lingually. We diagnose where transfer breaks down. Across Qwen2.5-7B, Gemma-2-9B, and Llama-3.1-8B on 23 languages, the harmfulness direction extracted from high-resource activations linearly separates harmful from harmless low-resource prompts nearly as well as high-resource ones. The relevant representation is present. Yet harmful refusal drops from 87.9% to 43.9%. The model fails to convert the representation into refusal. What fails to transfer is calibration of the safety decision, not the underlying representation. We exploit this by recalibrating, rather than retraining, a high-resource gate: a low-rank logistic readout with its decision threshold reset using as few as 1 to 4 target-language examples per class. The gate routes between refusal steering and harmfulness-direction ablation, substantially raising mean refusal selectivity ($\Delta$ = harmful $-$ harmless refusal) from 33.6 for the strongest adapted baseline to 54.5 while preserving MMLU utility. These results suggest that some low-resource safety failures can be repaired by recalibrating existing representations rather than learning new ones. Our code is released: this https URL.
>
---
#### [new 151] lmfaoooo at SemEval-2026 Task 1: Humor Is an Audience. Preference Modeling for Constrained Humor Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文针对幽默生成任务，解决观众依赖性与标注噪声问题。通过生成多样候选并使用偏好模型选择最佳结果，提升幽默生成质量。**

- **链接: [https://arxiv.org/pdf/2606.00022](https://arxiv.org/pdf/2606.00022)**

> **作者:** Alexey Tikhonov; Alexey Ivanov
>
> **备注:** 5 pages. Accepted for SEMEVAL 2026
>
> **摘要:** Humor generation remains difficult not only because producing fluent, novel jokes is hard, but because "funny" is audience-dependent and supervision is noisy -- preferences vary with audience, context, and culture, and annotator agreement is often low. In this paper, we describe our system for the SemEval-2026 Task-1 (MWAHAHA), which focuses on humor generation under explicit constraints. The task evaluates submitted systems via human preference judgments in 1-on-1 arena-style comparisons. We adopt a "generate-many -> select-best" strategy. First, we generate a diverse pool of candidates per instance using multi-step prompting, model ensembling, and diversity-oriented decoding. Second, we select outputs using a preference model that approximates a "reader" by learning from human comparisons rather than absolute funniness scores. To support this approach, we release 2.5K human pairwise judgments collected through the Humor Arena prototype. We further propose an interpretable pipeline that converts labeled comparisons into a preference model. Across three preference datasets, our models consistently outperform baselines and show stronger cross-domain transfer. Finally, we apply the learned preference model to rank candidates for the MWAHAHA setting and release intermediate artifacts (candidate pools and rankings) to facilitate follow-up work. Our system ranked 1st in the English and Chinese subtasks of MWAHAHA and 2nd in the Spanish subtask.
>
---
#### [new 152] FineVerify: Scaling Test-Time Compute with Fine-Grained Self-Verification for Agentic Search
- **分类: cs.CL**

- **简介: 该论文提出FineVerify，用于提升代理搜索系统的准确性。针对测试时计算扩展的问题，通过细粒度自验证框架，分解问题并验证候选答案，提高选择效率与可解释性。**

- **链接: [https://arxiv.org/pdf/2606.00660](https://arxiv.org/pdf/2606.00660)**

> **作者:** James Xu Zhao; Hui Chen; Bryan Hooi; See-Kiong Ng
>
> **备注:** 8+18 pages, 6 tables, 11 figures
>
> **摘要:** Agentic search requires language model agents to explore many sources and answer complex information-seeking questions. Scaling test-time compute is a promising way to improve these agents, but current approaches can fail, because correct answers are often sparse and score-based selection depends on model calibration. We propose FineVerify, a fine-grained self-verification framework that decomposes each question into checkable sub-questions, verifies sampled candidates against each sub-question, and selects the candidate with the highest aggregated score. This per-check structure turns selection into simpler local judgments and produces scores under the same explicit criteria. Across four agentic search benchmarks and two models, FineVerify consistently outperforms standard scaling baselines. With only four sampled trajectories, it improves GPT-5-mini by 8.2 accuracy points and Gemini-3-flash by 5.6% on average. With 12 samples, FineVerify enables GPT-5-mini to surpass frontier GPT-5 on BrowseComp-Plus. Beyond accuracy, FineVerify produces interpretable verification traces that help audit benchmark errors, suggesting broader applications for inspecting agentic search systems. Code and data are available at this https URL
>
---
#### [new 153] Scaling Agentic Capabilities via Grounded Interaction Synthesis
- **分类: cs.CL**

- **简介: 该论文属于智能体任务，旨在解决真实环境交互数据生成不足的问题。通过GAIS框架，构建多样化、高保真任务，提升模型性能与效率。**

- **链接: [https://arxiv.org/pdf/2606.02001](https://arxiv.org/pdf/2606.02001)**

> **作者:** Wenhang Shi; Jinhao Dong; Yiren Chen; Zhe Zhao; Shuqing Bian; Wei Lu; Xiaoyong Du
>
> **摘要:** General agentic intelligence hinges on the ability to interact with diverse real-world tools to complete complex tasks, a capability fundamentally tied to the quality of interaction data. To bypass the prohibitive costs of human annotation, prevailing paradigms depend entirely on Large Language Models (LLMs) to scale the synthesis of agentic environments and tasks. However, such unconstrained generation often degenerates into biased random sampling of LLMs' internal priors, failing to capture the diversity and difficulty of real-world domains or construct high-fidelity, long-horizon tasks. In this work, we introduce Grounded Agentic Interaction Synthesis (GAIS), a framework that automates the scalable construction of diverse environments and complex tasks via a two-phase grounding mechanism. Specifically, we construct protocol-anchored environments derived from real-world Model Context Protocol (MCP) servers to ensure functional diversity and difficulty. Subsequently, we employ structure-guided planning to navigate these environments, actively enforcing logical dependencies and adversarial policies to generate complex tasks. Experiments on BFCL, $\tau^2$-Bench, and ACEBench demonstrate that GAIS-synthesized data significantly outperforms state-of-the-art baselines, enabling base models to match or even surpass their official instruction-tuned counterparts. Furthermore, GAIS exhibits superior data efficiency and scalability, achieving exceptional capabilities with significantly less data while maintaining continuous growth where baselines stagnate. Our code and dataset are publicly available at this https URL.
>
---
#### [new 154] From Layers to Submodules: Rethinking Granularity in Replacement-Based LLM Compression
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型压缩任务，旨在提升压缩效果。针对现有方法限制，提出SubFit，实现非连续子模块替换，优化精度与效率平衡。**

- **链接: [https://arxiv.org/pdf/2606.02559](https://arxiv.org/pdf/2606.02559)**

> **作者:** Elia Cunegatti; Marcus Vukojevic; Erik Nielsen; Giovanni Iacca
>
> **摘要:** Post-training compression of Large Language Models (LLMs) removes entire architectural components, either deleting them or replacing them with fitted modules. Existing replacement-based methods share two design constraints: full-layer granularity and contiguous selection. We argue that this is overly restrictive: in fact, redundancy in pretrained transformers is not confined to contiguous regions, nor does it evenly distribute between Attention and FeedForward outputs, implying that different strategies best approximate different submodule types and that removable components need not cluster within contiguous depth ranges. Based on this intuition, we introduce SubFit (Submodule-level Fitted residual replacement), which compresses LLMs at the submodule level: Attention and FeedForward submodules are selected non-contiguously, and each receives its own lightweight fitted residual bypass. SubFit operates post-training and requires only calibration data. Across ten LLMs (five base, five instruction-tuned), five sparsity levels from 12.5% to 37.5%, and four replacement-based baselines, SubFit achieves the best aggregate perplexity-accuracy trade-off across the evaluated sparsity levels, with larger gains under aggressive compression. At 25% sparsity, it retains 84.6% of dense downstream accuracy and incurs 2.42x perplexity degradation, against 81.6% and 4.34x for the strongest baselines, while delivering measurable inference speedup and KV-cache savings. Code is available at this https URL.
>
---
#### [new 155] Model-Based Quality Assessment for Massively Multilingual Parallel Data
- **分类: cs.CL**

- **简介: 该论文属于多语言平行数据质量评估任务，解决非平行句对和低质量翻译问题，通过嵌入模型和无参考质量估计进行评估。**

- **链接: [https://arxiv.org/pdf/2606.00285](https://arxiv.org/pdf/2606.00285)**

> **作者:** Abdelaziz M.A. Ibrahim; Zihao Li; Jörg Tiedemann; Shaoxiong Ji
>
> **摘要:** Large-scale multilingual bitext often contains two distinct problems: non-parallel sentence pairs and low-quality translations. We decompose model-based assessment for such data into two independent components: parallelism assessment with multilingual embeddings and reference-free quality estimation (QE). For parallelism, we benchmark four embedding models on FLORES-200 and BOUQuET retrieval tasks, covering 6,654 source--target directions in our target language-pair inventory. For QE, we evaluate nine reference-free evaluators on professional FLORES-200 translations across 41,412 ordered source--target directions. Results show that no model is universally reliable across translation directions. Naive QE ensembles dilute strong model signals, while documented target-language coverage is strongly associated with higher QE scores. Overall, these findings suggest that multilingual parallel-data assessment is best approached as a direction-aware routing and calibration problem, where no single universal metric is expected to suffice across all languages.
>
---
#### [new 156] A Registry-Bound LLM Pipeline for Evidence-Grounded Trait Extraction across Tropical Plants, Aquatic Species, and Exotic Pets
- **分类: cs.CL**

- **简介: 该论文提出一种基于大语言模型的结构化特征提取流水线，用于热带植物、水生生物和异宠的证据驱动特征记录。旨在解决大规模、可审计的特征提取问题，通过四机制确保准确性与可追溯性。**

- **链接: [https://arxiv.org/pdf/2606.00994](https://arxiv.org/pdf/2606.00994)**

> **作者:** Jeff Wang
>
> **备注:** 33 pages, 6 figures; methodology paper
>
> **摘要:** We describe a registry-bound large-language-model extraction pipeline producing evidence-grounded structured trait records at scale, on cultivated tropical plant, aquatic, and pet species. Four mechanisms render LLM-derived rows auditable: a versioned 39-key closed-vocabulary trait registry constraining every admitted value to a typed schema; a per-row verbatim evidence quote tying each value to source text; a per-row confidence label (high or medium; low dropped pre-persist); and multi-version preservation. Applied to 409,880 publishable species from the Tropical Species Encyclopedia, the pipeline executed 706,220 runs and persisted 5,489,881 trait records across 409,820 species (99.985%), 81.57% at high confidence. We report three validation layers in descending evidentiary strength: at full population, 90.12% of 5,427,588 evidence-bearing rows have their quote as a verbatim source substring (93.49% excluding one compliance meta-trait); a quote-supports-value audit on n=100 stratified non-red-zone rows yielded 100/100 (lower bound 96.30%); face-validity on n=50 red-zone rows yielded 50/50 Accept (lower bound 92.86%). Per-record correctness is not claimed; 100% pending human curation. The contribution is the four-mechanism framework.
>
---
#### [new 157] Consistent and Distinctive: LLM Benchmark Efficiency via Maximum Independent Set Prompt Selection on Similarity Graphs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于模型评估任务，旨在高效评估大语言模型。通过构建相似性图并应用最大独立集算法，选择多样化提示，减少冗余，提升评估效率。**

- **链接: [https://arxiv.org/pdf/2606.01400](https://arxiv.org/pdf/2606.01400)**

> **作者:** Denica Kjorvezir; Marko Djukanović; Ana Gjorgjevikj; Gjorgjina Cenikj; Tome Eftimov
>
> **摘要:** Evaluating large language models (LLMs) across comprehensive benchmarks is expensive and time-consuming. We propose a graph-based prompt selection framework that models each benchmark as a similarity graph -- nodes are prompts connected if their embedding-space distance falls above a configurable threshold -- and applies Maximum Independent Set (MIS) algorithms to select a maximally diverse, non-redundant subset. We evaluate four MIS solvers (CPLEX, GREEDY, Online-MIS, ReduMIS) across six embedding models, three distance measures, six percentile thresholds, and four benchmarks (GPQA, IFEval, MMLU-Pro, Omni-MATH) covering 66 LLMs. Our central hypothesis -- that repeated selection under different random seeds yields consistent LLM rankings that may also differ from the full-benchmark baseline -- is strongly confirmed: Kendall's $W \geq 0.90$ in 99.2\% of stochastic configurations (mean $W = 0.997 \pm 0.008$), while at higher percentile thresholds selected subsets achieve 25--48\% prompt reduction on average. Ranking divergence from the full benchmark ($\rho < 0.95$) occurs in only 15.95\% of configurations, concentrated at low thresholds ($p_{10}$--$p_{20}$) and benchmarks (GPQA, IFEval), identifying overly dense graphs as the primary failure mode.
>
---
#### [new 158] CultureForest: Understanding and Evaluating Cultural Norm Grounded Reasoning in LLMs
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大模型在文化规范推理上的评估问题。提出CultureForest基准，评估模型在真实场景中运用文化知识的能力，发现模型在开放任务中表现下降，强调需从知识评估转向推理评估。**

- **链接: [https://arxiv.org/pdf/2606.01879](https://arxiv.org/pdf/2606.01879)**

> **作者:** Yangfan Ye; Xiaocheng Feng; Jialong Tang; Xiayu Cao; Zihan Zhang; Xiachong Feng; Baosong Yang; Bing Qin
>
> **摘要:** Existing research largely reduces cultural intelligence in LLMs to a knowledge-level problem, overlooking whether models can effectively utilize their acquired knowledge in realistic scenarios. To bridge this gap, we introduce CultureForest, a benchmark for \textit{Cultural Norm Grounded Reasoning}. Each question is grounded in a small set of atomic norms, enabling verifiable and attributable evaluation. CultureForest comprises 5,378 examples across 8 domains and 53 countries/regions, and supports a progressive evaluation from multiple-choice to open-ended generation. Extensive experiments reveal that even top-tier models degrade substantially in open-ended settings, accompanied by pronounced cross-region disparities. Through targeted analysis, we uncover several consistent patterns: (1) test-time reasoning yields limited gains and may exacerbate inequity; (2) models exhibit highly shared regional preference structures; (3) model responses are markedly conservative, especially under stricter cultural constraints; and (4) by disentangling cultural knowledge acquisition from cultural reasoning, we show that while LLMs possess substantial cultural knowledge, their performance is further bottlenecked by its effective use. These findings point to a necessary shift from knowledge-centric evaluation toward measuring knowledge-grounded reasoning.
>
---
#### [new 159] TCAR-Gen: Temporal Graph Retrieval with Evidence Fusion for Knowledge-Grounded Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识增强生成任务，解决历史案件问答中的时间推理与证据融合问题。提出TCAR-Gen框架，结合图神经网络和时间信息，提升多源证据整合效果。**

- **链接: [https://arxiv.org/pdf/2606.00029](https://arxiv.org/pdf/2606.00029)**

> **作者:** Sidra Nasir; Muhammad Noman Zahid; Rizwan Ahmed Khan
>
> **摘要:** Retrieval-augmented generation systems struggle with temporal reasoning and evidence fusion when answering complex questions over historical criminal case narratives. Existing approaches either retrieve independently of query semantics or fail to integrate multiple evidence sources coherently. We propose Temporal Context Augmented Retrieval Generation (TCAR-Gen), a framework that combines query-conditioned graph neural networks, temporal evidence fusion, and chain-of-trees reasoning to ground answer generation in retrieved evidence. On the Victorian Crime Diaries benchmark, TCAR-Gen achieves 0.3738 Recall@5, outperforming Vanilla RAG, Temporal RAG, GraphRAG-C, and GraphRAG-T across seven query types including multi-hop reasoning and counterfactual questions. Ablation studies reveal that the context graph, temporal penalty mechanism, and query conditioning are critical components. Cross-model evaluation across five language model (GPT-OSS 20B to TinyLlama 1.1B) demonstrates that TCAR-Gen maintains robust retrieval coverage at smaller model scales, though generation quality degrades substantially with reduced model capacity. Our work shows that explicit temporal modelling and multi-branch evidence fusion are essential for faithful, reasoning-intensive question answering over knowledge-grounded corpora.
>
---
#### [new 160] LayerRoute: Input-Conditioned Adaptive Layer Skipping via LoRA Fine-Tuning for Agentic Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出LayerRoute，用于优化代理语言模型的推理效率。任务是减少冗余计算，解决不同步骤计算资源分配不均的问题。通过自适应跳过Transformer层，提升性能并降低能耗。**

- **链接: [https://arxiv.org/pdf/2606.01838](https://arxiv.org/pdf/2606.01838)**

> **作者:** Prateek Kumar Sikdar
>
> **备注:** 10 pages, 3 figures, 4 tables
>
> **摘要:** Agentic language model systems alternate between two structurally distinct step types: structured tool calls (short, deterministic, low perplexity) and open-ended planning/reasoning steps (long, complex, high perplexity). Despite this heterogeneity, current inference systems apply identical compute to every step. We introduce LayerRoute, a lightweight adapter that learns to selectively skip transformer blocks on a per-input basis. LayerRoute augments each of the 24 transformer blocks in Qwen2.5-0.5B-Instruct with: (1) a per-layer router (~897 parameters, Linear(896,1)) that outputs a hard binary gate via the straight-through estimator, and (2) LoRA adapters (rank 8, ~1.08M parameters) on the Q/K/V/O attention projections. The backbone weights remain frozen. A single end-to-end training pass on agentic data (Hermes, Glaive, GSM8K, Turing) with a gate regularisation term forces the system to discover which blocks are skippable per input type. After 3,000 steps (6.4 minutes on an A100 40GB), LayerRoute achieves a 12.91% skip differential: tool calls skip 15.25% of FLOPs while planning steps skip only 2.34%, using only 1.10M trainable parameters (0.22% of the 494M backbone). Quality improves over the base model due to LoRA adaptation, with perplexity delta of -1.29 on tool calls and -1.30 on planning.
>
---
#### [new 161] TVIR: Building Deep Research Agents Towards Text--Visual Interleaved Report Generation
- **分类: cs.CL**

- **简介: 该论文提出TVIR，解决多模态报告生成中视觉元素可靠性与对齐问题，构建了基准和框架，提升证据驱动的报告质量。**

- **链接: [https://arxiv.org/pdf/2606.02320](https://arxiv.org/pdf/2606.02320)**

> **作者:** Xinkai Ma; Zhiqi Bai; Dingling Zhang; Pei Liu; Yishuo Yuan; He Zhu; Jiakai Wang; Qianqian Xie; Yifan Zhao; Xinlong Yang; Hao Cong; Zhiheng Yao; Fengxia Xie; Zihao Xu; Haoran Xu; Zhaohui Wang; Minghao Liu; Shirong Lin; Yingshui Tan; Yuchi Xu; Wenbo Su; Zhaoxiang Zhang; Bo Zheng; Jiaheng Liu
>
> **摘要:** Deep Research Agents have shown strong capability in multi-step information retrieval, reasoning, and long-form report generation, but existing benchmarks and systems remain predominantly text-centric, with limited evaluation of whether visual elements are factually reliable and well aligned with the surrounding analysis. To address this gap, we introduce TVIR (Text--Visual Interleaved Report Generation), which includes TVIR-Bench, a benchmark of 100 expert-curated multimodal deep research tasks that require visual elements to serve specific analytical sub-goals, and TVIR-Agent, a hierarchical multi-agent framework that serves as a strong baseline for constructing outlines, retrieving images, generating charts with traceable sources, and composing reports through context-aware sequential writing. We further develop a dual-path evaluation framework that combines Textual Assessment and Visual Assessment. Experiments across nine deep research systems show that TVIR-Agent achieves strong overall performance, underscoring the importance of explicit multimodal design and evaluation for evidence-driven report generation.
>
---
#### [new 162] DLLM-JEPA: Joint Embedding Predictive Architectures for Masked Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出DLLM-JEPA，解决自监督学习中的高成本问题，通过结合JEPA与掩码扩散语言模型，减少训练成本并提升性能。**

- **链接: [https://arxiv.org/pdf/2606.00091](https://arxiv.org/pdf/2606.00091)**

> **作者:** Sangdae Nam
>
> **备注:** 17 pages, 4 figures, 13 tables. Accepted at SPIGM Workshop, ICML 2026
>
> **摘要:** Joint Embedding Predictive Architectures (JEPAs) have reshaped self-supervised representation learning in vision. The recent LLM-JEPA ported JEPA to autoregressive language models but inherited two steep costs from the causal-attention substrate: it demands explicit multi-view data (e.g., text-code pairs), and it requires two gradient-carrying forward passes per step. We introduce DLLM-JEPA, which pairs JEPA with masked-diffusion language models to eliminate both costs at once. The bidirectional attention of diffusion models yields two semantically distinct views of the same input via different masking rates -- no explicit pairs needed -- and supports a single gradient-carrying forward pass, cutting training FLOPs by 33% relative to LLM-JEPA. DLLM-JEPA improves over diffusion-only fine-tuning in every (task, architecture) combination we evaluate: up to +18.7 pp on LLaDA-8B GSM8K and +11.4 pp on Dream-7B GSM8K, with consistent positive gains on Spider, NL-RX-SYNTH, and Django. Beyond accuracy, DLLM-JEPA exhibits a dual-win property: on LLaDA-8B with the Wide-t configuration, it simultaneously raises GSM8K accuracy (67.1 vs. 65.2, +1.8 pp), drives held-out Wikitext loss below the pre-trained base, and preserves MMLU accuracy at base level across three fine-tuning seeds -- whereas an L2-to-base parameter anchor matches baseline accuracy with no task gain. Layer-wise probing reveals the mechanism: a geometric-functional drift dissociation in which the fine-tuned backbone moves further from the pre-trained weights than the baseline yet forgets less on held-out Wikitext, with the amplification concentrated in middle transformer layers. The pattern appears on Dream-7B as well, indicating the phenomenon is not specific to a single backbone.
>
---
#### [new 163] Who Annotates in NLP? A Large-scale Assessment of Human Annotation Reporting between 2018 and 2025
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理（NLP）领域，旨在评估2018至2025年间人类标注报告的透明度。研究分析了标注细节的缺失情况，提出了统一分类体系，并建立了一个标注数据集，以提升标注的可靠性与可重复性。**

- **链接: [https://arxiv.org/pdf/2606.02255](https://arxiv.org/pdf/2606.02255)**

> **作者:** Maria Kunilovskaya; Gagan Bhatia; Lisa Sophie Albertelli; Yanran Chen; Christian Greisinger; Lotta Kiefer; Christoph Leiter; Subhadeep Roy; Tewodros Achamaleh; Muhammad Arslan Manzoor; Sebastian Pohl; Yufang Hou; Steffen Eger
>
> **摘要:** Human annotation is the empirical foundation of much NLP research, from dataset construction to model evaluation, but papers often leave unclear who produced the annotations and how the annotation process was controlled. We provide the first large-scale, task-level audit of human annotation reporting across major NLP venues, asking which annotation details are documented, which are missing, and how reporting varies across time, topic, venue, and intended use of human judgment. We introduce a unified taxonomy of annotation-reporting practices and validate an LLM-assisted extraction pipeline against Annotated-gold, a human-adjudicated gold standard of 41 papers and 72 annotation tasks, where the best model reaches human-comparable agreement with adjudicated labels, with Krippendorff's alpha of 0.606 versus 0.585 for human-human agreement. Using this pipeline, we construct Annotated-llm, a dataset covering ACL-venue papers from 2018-2025, with 2,667 extracted annotation tasks from 1,603 papers, and find that papers frequently report operational details such as recruitment strategies, annotator expertise, and annotation volume, but often omit details needed to assess annotation validity, including training, language proficiency, compensation, socio-demographics, adjudication, and agreement values, especially in model-evaluation studies. Our results show that annotation reporting in NLP has improved over time but remains uneven, and they establish a scalable framework and bare-minimum reporting recommendations for making human annotation more reliable, reproducible, and interpretable.
>
---
#### [new 164] Do Gender Cues Affect LLM Value Trade-offs? Evidence from a Controlled Decision Benchmark
- **分类: cs.CL**

- **简介: 该论文属于人工智能伦理任务，研究性别线索是否影响大语言模型的价值权衡。通过构建基准测试，分析性别提示对决策的影响及模型自我归因情况。**

- **链接: [https://arxiv.org/pdf/2606.02214](https://arxiv.org/pdf/2606.02214)**

> **作者:** Yangyang Liu; Dong Yu; Pengyuan Liu
>
> **摘要:** Large language models are increasingly used in value-sensitive decision settings, where irrelevant demographic cues should not alter judgments. We construct the Realistic Value Decision Benchmark (RVDB), a controlled benchmark that varies only the role-gender configuration while holding the scenario, ordered value pair, roles, candidate decisions, Value Distance, and Decision Severity fixed. Using a position-balanced evaluation across seven models, we test whether models preserve decision invariance under gender perturbations and whether their self-attributions reflect observed behavioral changes. We find that explicit gender cues induce bounded but systematic decision flips, including under an explicit gender-attribution prompt that asks models to report whether gender influenced their choice. Cross-gender role swaps reveal a consistent female-proposed-decision asymmetry, while models often attribute flipped decisions to No Influence or other non-gender factors. Further analysis shows that gender effects concentrate near less determinate value boundaries and under more severe decision contexts, suggesting that gender cues act as local boundary-shifting factors rather than global overrides of value reasoning. Value rankings remain largely stable, but ordered value-pair trade-offs shift unevenly across role-gender configurations. These results show that gender can enter LLM value trade-offs behaviorally while remaining obscured in self-attribution, motivating controlled behavioral audits beyond explanation-based evaluation.
>
---
#### [new 165] Bridging Reasoning Trajectories in On-Policy Distillation via Near-Future Guidance
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型训练任务，旨在解决OPD中因token级监督导致的轨迹偏差问题。通过引入近未来轨迹信息提升模型性能。**

- **链接: [https://arxiv.org/pdf/2606.00305](https://arxiv.org/pdf/2606.00305)**

> **作者:** Yuxuan Jiang; Francis Ferraro
>
> **摘要:** On-Policy Distillation (OPD) improves large language model reasoning by training a student model on trajectories sampled from its own policy under teacher supervision. Although OPD operates on trajectories, its learning signal remains token-level: it identifies deviations through high-loss tokens and repairs them through local reverse-KL correction. We show that this "trajectory-sampled but token-learned" mechanism cannot reliably bridge student trajectories toward teacher trajectories. About 30% of high-loss tokens fall into the low-divergence regime, indicating that many are surface-form mismatches rather than real reasoning forks. Moreover, even truly divergent tokens are difficult to repair with isolated token-level supervision, since reasoning failures often unfold as short-horizon distributional drift. We propose Trajectory-aware OPD (TOPD), which uses near-future trajectory information to identify real divergent states and distribute guidance across multiple future tokens. Experiments show that suppressing non-divergent high-loss tokens improves standard OPD from 47.8% to 48.2% average accuracy, while TOPD further improves performance to 52.2%, with gains on AIME24 from 60.0% to 63.3% and AIME25 from 46.7% to 53.3%.
>
---
#### [new 166] Eyettention II: A Dual-Sequence Architecture for Modeling Fixation Location, Within-Word Landing Position, and Fixation Duration in Reading
- **分类: cs.CL**

- **简介: 该论文提出Eyettention II模型，用于生成逼真的阅读眼动轨迹，解决数据稀缺问题，提升自然语言处理和心理语言学研究。**

- **链接: [https://arxiv.org/pdf/2606.01964](https://arxiv.org/pdf/2606.01964)**

> **作者:** Shuwen Deng; Cui Ding; David R. Reich; Paul Prasse; Lena A. Jäger
>
> **摘要:** The way our eyes move while reading provides valuable insights into both the reader's cognitive processes and the properties of the text. In particular, eye-tracking-while-reading data has shown to be highly beneficial in various technological applications, such as enhancing and interpreting language models and inferring a reader's characteristics. However, these applications often rely on large-scale, data-driven models, which demand extensive eye-tracking datasets that are challenging to obtain due to the resource-intensive nature of data collection. To address the challenge of data scarcity, we develop Eyettention II, an end-to-end trained deep-learning model capable of generating realistic scanpaths consisting of a complete set of fixation attributes in chronological order, including fixation location, within-word landing position, and fixation duration. Our model is lightweight, efficiently trainable on limited GPU resources, and closely aligned with cognitive theories. We demonstrate that Eyettention II surpasses state-of-the-art models in scanpath prediction and mirrors human-like gaze behavior by capturing key psycholinguistic phenomena. With its robust performance, Eyettention II holds the potential to drive advancements in natural language processing, facilitate piloting the materials of psycholinguistic experiments, and uncover new insights beyond what is explicitly encoded in theoretical cognitive models.
>
---
#### [new 167] Beyond Isolated Behaviors: Hierarchical User Modeling for LLM Personalization
- **分类: cs.CL**

- **简介: 该论文属于LLM个性化任务，旨在解决用户行为结构化不足的问题。提出PHF框架，通过实践、惯习和场域三层结构实现更精准的用户建模。**

- **链接: [https://arxiv.org/pdf/2606.02300](https://arxiv.org/pdf/2606.02300)**

> **作者:** Liang Wang; Xinyi Mou; Xiaoyou Liu; Tiannan Wang; Yuqing Wang; Zhongyu Wei
>
> **摘要:** Large Language Models (LLMs) have demonstrated remarkable capabilities across diverse domains, yet personalizing their outputs to individual users remains an open challenge. Existing approaches predominantly adopt a flat behavioral paradigm, aggregating user behaviors without an explicit account of how they are organized into deeper behavioral structures. In this work, we draw on Pierre Bourdieu's Theory of Practice to propose PHF (Practice-Habitus-Field), a sociologically grounded framework that reconceptualizes LLM personalization through three hierarchical levels: individual behaviors as practices, their temporal accumulation into stable dispositions as habitus, and shared regularities across similar users as fields. We instantiate PHF through $\mathrm{PHF}_{\text{Compass}}$, a lightweight and model-agnostic implementation based on a frozen LLM. Experiments on the Language Model Personalization (LaMP) benchmark demonstrate consistent improvements across diverse tasks, while further analyses validate the interpretability and extensibility of the learned behavioral structures.
>
---
#### [new 168] Internalize the Temperature: On-Policy Self-Distillation as Policy Reheater for Reinforcement Learning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于强化学习任务，解决熵崩溃问题。通过内部化温度影响，提出TS-OPSD方法，提升策略多样性与学习效果。**

- **链接: [https://arxiv.org/pdf/2606.00755](https://arxiv.org/pdf/2606.00755)**

> **作者:** Xuewei Yang; Jiachen Yu; Jie Wu; Shaoning Sun; Junjie Wang; Yujiu Yang
>
> **摘要:** Reinforcement learning from verifiable rewards improves the reasoning ability of large language models, but often suffers from entropy collapse, in which increasingly concentrated policies reduce rollout diversity and useful learning signals. Existing remedies either constrain the RL objective (e.g., entropy regularization) or adjust sampling temperature during rollout collection, but these interventions remain external to the model parameters. We propose Temperature-Scaled On-Policy Self-Distillation (TS-OPSD), a lightweight policy reheating method that internalizes the exploratory effect of temperature into model parameters. Starting from an entropy-collapsed RL checkpoint, TS-OPSD constructs a self-teacher by applying high-temperature scaling to the model's own logits, then distills the resulting smoother distribution back into the student. This policy reheating requires no external teacher, privileged data, or additional inference cost. Experiments on Qwen3-4B-Base and Qwen3-8B-Base show that policy reheating yields a stronger initialization for continued RL than both standard continued RL and rollout-level temperature reheating. Further analyses show that TS-OPSD mainly reduces output sharpness while preserving intermediate representations, top candidate sets, and reasoning capability. These results suggest that entropy restoration can serve as a simple post-collapse intervention for extending reasoning-oriented RL.
>
---
#### [new 169] K-BrowseComp: A Web Browsing Agent Benchmark Grounded in Korean Contexts
- **分类: cs.CL**

- **简介: 该论文提出K-BrowseComp，一个针对韩国语境的网页浏览代理基准，旨在评估大模型在复杂任务中的表现。任务属于人工智能评估，解决韩国语境下代理能力评测不足的问题。工作包括构建数据集并测试不同模型表现。**

- **链接: [https://arxiv.org/pdf/2606.02404](https://arxiv.org/pdf/2606.02404)**

> **作者:** Nahyun Lee; Dongkeun Yoon; Guijin Son; Geewook Kim; Dayoon Ko; Jeonghun Park; Haneul Yoo; Jaewon Cho; Junghun Park; Changyoon Lee; Kyochul Jang; Jaeyeon Kim; Eunsu Kim; Woojin Cho; Seungone Kim
>
> **摘要:** Frontier model evaluations are shifting from foundational capabilities (e.g., instruction following and reasoning) toward compositional, agentic ones, but Korean agentic benchmarks remain scarce. We introduce K-BrowseComp, a web-browsing agent benchmark grounded in Korean contexts, consisting of 400 problems. The 300-problem K-BrowseComp-Verified subset is manually constructed and validated by native Korean speakers. On this subset, frontier LLMs, including GPT-5.5, DeepSeek-V4-Pro, and GLM-5.1, reach only 30.00--45.67\%, a substantial drop from BrowseComp, while Korean LLMs released through Korea's Proprietary AI Foundation Model program obtain only 0.00--10.33\%. We further construct a 100-problem synthetic split using hard few-shot exemplars and failure-mode-targeted generation to exploit the asymmetry between solving and creating web browsing problems. On the adversarially filtered synthetic diagnostic split, the strongest model reaches only 26.00\%, and we report this split separately as a targeted stress test. We publicly release our data and code.
>
---
#### [new 170] Before and After Temperature: A Distributional View of Creative LLM Generation
- **分类: cs.CL**

- **简介: 该论文研究无参考的LLM创造力评估任务，通过分析采样温度对词分布的影响，提出新的特征以更准确预测生成内容的创意性。**

- **链接: [https://arxiv.org/pdf/2606.01451](https://arxiv.org/pdf/2606.01451)**

> **作者:** V. S. Raghu Parupudi; Harsha Ponnada; Aditi Kaushal; S. Shria Parupudi; Saiteja Dasari; Sahiti Bulusu
>
> **备注:** Submitted to NGEN-AI 2026
>
> **摘要:** Reference-free evaluation of large language model (LLM) creativity relies on perplexity, entropy, and top-1 margin. We show that a much stronger signal lives one step earlier in the pipeline: in how sampling temperature \emph{reshapes} the model's token distribution before the next token is drawn. On Llama-3.1-8B-Instruct generations of 500 open-ended creative prompts at $T \in \{0.3, 0.8, 1.5\}$, a single per-token feature derived from this reshaping predicts the within-prompt creativity rank at Spearman $\rho{=}0.918$ against an averaged gpt-4o\,/\,gemini-2.5-pro judge ($n{=}500$) and $\rho{=}0.870$ against a three-rater human-majority ranking ($n{=}150$). Each of four standard reference-free baselines (self-perplexity, mean predictive entropy, top-1 margin, gzip compression ratio) tops out at $|\rho|\!\approx\!0.76$ on both ground truths: a gap of $+0.165$ on averaged-LLM and $+0.110$ on human-majority, both far larger than the spread among the baselines themselves. The two ground-truth panels agree with each other at $\rho{=}0.83$, above the inter-human ceiling of $\rho{=}0.77$, so the comparison is not bottlenecked by judge noise. Mechanistically, the win comes from a sharp distributional signature of the incoherence regime: at $T{=}1.5$ the cumulative-mass width $n_{95}(q)$ inflates from $\sim\!1$ to ${\sim}\!131$ tokens and post-temperature mass leaks off the pre-temperature top-$90\%$ plausible set by about $13$ percentage points. The per-token aggregates do not separate $T{=}0.8$ from $T{=}0.3$; discriminating the two coherent regimes is left to sequence-level features.
>
---
#### [new 171] Transferable Self-Harm Surveillance from Emergency Department Triage Notes Using an Evidence-Augmented Machine Learning Approach
- **分类: cs.CL**

- **简介: 该论文属于自伤行为监测任务，旨在解决传统方法敏感性低的问题。通过融合机器学习与大语言模型，从急诊分诊记录中识别自伤行为，实现高精度分类与方法识别。**

- **链接: [https://arxiv.org/pdf/2606.02545](https://arxiv.org/pdf/2606.02545)**

> **作者:** Liuliu Chen; Gowri Rajaram; Eleanor Bailey; Katrina Witt; Michelle Lamblin; Jo Robinson; Mike Conway; Vlada Rozova
>
> **摘要:** Self-harm is a major public health concern, but current surveillance relying on hospital presentations is inadequate due to the low sensitivity of diagnostic codes. Emergency Department (ED) triage notes, recorded at the initial point of contact, provide a succinct summary of presentations and an opportunity to identify self-harm. We developed a three-stage approach, augmenting traditional machine learning with large language model-based screening and evidence extraction to detect self-harm in ED triage notes. We assessed model transferability across three Australian hospitals. Our approach showed AUPRCs of 0.887 +/- 0.016 and 0.884 +/- 0.012 during internal and external validation. Prospectively, it achieved AUPRC of 0.881 +/- 0.008 at the development site, and 0.879 +/- 0.012 and 0.816 +/- 0.015 at two external sites without site-specific retraining. A key advantage of the approach is that it enables identification of the primary self-harm method with an accuracy of 95%, supporting more granular surveillance beyond binary classification.
>
---
#### [new 172] Automated Essay Scoring and Language Certification: Assessing Generalizability, Agreement and Validity for French
- **分类: cs.CL**

- **简介: 该论文属于自动作文评分任务，旨在提升法语自动评分模型的评估有效性。通过改进的ABV框架，分析模型的公平性、准确性及与人工评分的一致性，优化模型性能。**

- **链接: [https://arxiv.org/pdf/2606.02009](https://arxiv.org/pdf/2606.02009)**

> **作者:** Rodrigo Wilkens; Rémi Cardon; Vincent Folny; Thomas François
>
> **摘要:** In Automated Essay Scoring (AES), benchmarking practices have fostered minimalist evaluation practices, in contrast with the broader-view recommendations of evaluation frameworks, such as the argument-based validation framework (ABV), which argued in favor of a multidimensional assessment of systems, especially in the context of high-stakes language tests. In this paper, we introduce an enhanced and more practical version of the ABV framework, incorporating fairness analysis, correlations with linguistic features, prediction error evaluation, and model agreement compared with human raters. Applying this framework to French AES, we compare 8 model architectures on a corpus of 27k exam essays (2 raters each) and a generalization corpus of 961 essays (at least nine raters each). Our analyses illustrate the benefits of applying the ABV framework to better understand the capabilities and pitfalls of AES models, while also advancing the state-of-the-art for French AES.
>
---
#### [new 173] Easier to Mislead Than to Correct: Harmful and Beneficial Revision in LLM Conformity
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于人工智能领域，研究多智能体系统中大语言模型的从众问题。通过实验发现，模型更易被误导而非纠正错误，且权威标签影响决策。需验证同伴答案而非简单聚合。**

- **链接: [https://arxiv.org/pdf/2606.01637](https://arxiv.org/pdf/2606.01637)**

> **作者:** Jiaming Qu; Lucheng fu; Yibo Hu
>
> **摘要:** Large language models are increasingly used in multi-agent systems, where they see and respond to other agents' answers. A key risk is conformity: a model may abandon its own answer simply because others agree on a different one. Prior studies show that LLMs often revise toward a majority answer, but it remains unclear whether these revisions help correct mistakes as often as they introduce new errors. In this paper, we conduct a controlled study in which an LLM first answers a question, then sees simulated peer responses before making a final decision. We manipulate two social cues: consensus structure and authority labels assigned to peers, and measure how they influence beneficial and harmful revisions. Across four open-weight LLMs and seven QA datasets, we find that peer agreement makes it much easier to mislead initially correct models than to correct initially wrong ones. Authority labels make models more likely to choose the endorsed answer, regardless of whether it is correct. More concerningly, generic reasoning interventions such as chain-of-thought and reflection do not reliably reduce harmful revision while preserving beneficial revision. These findings suggest that multi-agent LLM systems should verify peer answers rather than simply aggregate them.
>
---
#### [new 174] ProtStructQA: A Denotation Threshold in Protein Structural Reasoning
- **分类: cs.CL**

- **简介: 该论文提出ProtStructQA，一个用于蛋白质结构问答的基准任务，解决语言模型将文本映射到3D结构测量的问题。工作包括构建基准数据集并测试不同模型策略的有效性。**

- **链接: [https://arxiv.org/pdf/2606.00451](https://arxiv.org/pdf/2606.00451)**

> **作者:** Aravind Mandiga; Guoming Li; Jin Lu; Ismailcem Budak Arpinar; Khaled Rasheed; Samuel E. Aggrey
>
> **摘要:** Protein-language systems are often evaluated by whether they generate plausible biological text, but a structural question has a sharper semantics: it denotes a measurement in a 3D coordinate system. We introduce ProtStructQA, an executable benchmark for protein structural question answering in which each natural-language question is generated from a hidden typed domain-specific language (DSL) program and the answer is obtained by executing that program on an AlphaFold-predicted structure. ProtStructQA releases 382.2K questions covering confidence, distances, predicted aligned error (PAE), solvent exposure, secondary structure, topology and contacts, and held-out compositions: a 330K active benchmark over 10K proteins from four species, plus a 52.2K hard-negative robustness pool. Without fine-tuning, we evaluate Qwen3 models from 0.6B to 8B under direct prompting, chain-of-thought, grammar-constrained executable voting, executable voting with chain-of-thought, and multi-turn ReAct-style tool use, and replicate the headline finding on Gemma-3-1B and Gemma-3-12B. We find a capability-dependent denotation threshold between Qwen3-1.7B and Qwen3-4B: below it, tool-mediated ReAct dominates because models often fail to produce executable denotations; above it, chain-of-thought flips from mostly harmful to strongly beneficial and becomes the strongest strategy on most splits. Parse-failure and family-level analyses show that the threshold is a transition from unparseable language to executable structural denotation, while grammar and execution remain selectively valuable for PAE and secondary-structure queries. ProtStructQA reframes scientific QA as compilation from language to measurement and provides a diagnostic testbed for when language models can map words to executable 3D structural measurements.
>
---
#### [new 175] How Far Do Auto-Interpretation Labels Generalize: A Controlled Study Across Languages, Scripts, and Rewordings
- **分类: cs.CL**

- **简介: 该论文研究SAE特征标签在不同语言和脚本间的泛化能力，旨在解决跨语言语义理解问题。通过塞尔维亚语双书写系统实验，发现标签在不同语言中表现不一致，表明其可能反映输入表示而非概念本身。**

- **链接: [https://arxiv.org/pdf/2606.00356](https://arxiv.org/pdf/2606.00356)**

> **作者:** Sripad Karne
>
> **摘要:** Sparse autoencoder (SAE) features are increasingly used to interpret language models, with auto-generated natural-language labels serving as the primary interface for understanding what each feature represents. We ask whether these labels generalize: does a feature labeled for a concept actually track that concept across languages and scripts? Using Serbian digraphia as a controlled testbed -- the same language written in both Latin and Cyrillic via deterministic transliteration -- we first find that SAE feature sets activated by the same content in different languages, scripts, and wordings share substantial overlap (peak Jaccard similarity 0.57 vs.\ 0.13 random baseline), suggesting genuine cross-lingual semantic features. We then test whether auto-interpretation labels keep pace. They often do not: features whose labels describe semantic content miss the same meaning in Serbian up to $4\times$ more often than within English, and miss Serbian Cyrillic more than Serbian Latin -- two scripts that are deterministic transliterations of each other -- suggesting the failures track how well each form is represented in training. The gap grows with network depth, yet the labels give no indication that they fail. These results suggest that auto-interpretation labels may reflect a feature's behavior on well-represented inputs rather than the concept itself.
>
---
#### [new 176] Graph-Augmented Retrieval for Cross-Entity Financial Sentiment Analysis: A Comparative Study
- **分类: cs.CL**

- **简介: 该论文属于多实体金融情感分析任务，旨在解决传统向量检索在捕捉复杂关系上的不足。通过构建图增强的RAG系统，提升实体召回与答案相关性。**

- **链接: [https://arxiv.org/pdf/2606.00062](https://arxiv.org/pdf/2606.00062)**

> **作者:** Rajan Bastakoti; Sagar Bhetwal; Nirajan Acharya; Gaurav Kumar Gupta
>
> **摘要:** Retrieval-Augmented Generation (RAG) has become foundational for grounding large language models in domain-specific corpora, yet conventional vector-based RAG systems are fundamentally limited in their ability to capture the structured, multi-entity relationships that underpin financial market analysis. This paper presents a comprehensive comparative study of a novel two-hop Graph-RAG architecture versus a standard vector-only baseline for cross-entity financial sentiment analysis. Our system constructs a sentiment-weighted knowledge graph of 59 equity entities from 255 news articles covering 10 major technology stocks, then augments dense retrieval with intensity-filtered graph traversal over INFLUENCES edges to surface relational evidence inaccessible to vector search alone. We evaluate both architectures on 100 grounded queries (30 Direct, 70 Relational) using semantic similarity, entity recall, RAGAS metrics, latency benchmarks, and ablation studies. Graph-RAG achieves a statistically significant improvement in entity recall (+6.4%, p < 0.001, Wilcoxon signed-rank) and delivers substantially more relevant answers for complex multi-entity queries (+11.7% Answer Relevancy), with gains concentrating in relational question types (+16.1%). Critically, these improvements come at no measurable cost to answer quality (delta = +0.001 semantic similarity, Cohen's d = 0.078), with a modest 22.6% increase in mean latency offset by an 80% reduction in latency variance. An ablation study on the graph traversal intensity threshold reveals an inverted-U relationship with answer quality, identifying tau = 0.5 as optimal over the production default of tau = 0.7. These findings characterize a precision-for-coverage trade-off inherent to graph-augmented retrieval and provide actionable architectural guidance for practitioners building RAG systems for multi-entity financial analysis.
>
---
#### [new 177] IndoBias: A Dual Track Culturally Grounded Benchmark for LLMs Bias Evaluation in Indonesian Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型偏见评估任务，旨在解决印尼语及地方语言中模型偏见问题。构建了IndoBias基准，通过双轨评估方法分析模型在文化背景下的偏见表现。**

- **链接: [https://arxiv.org/pdf/2606.01260](https://arxiv.org/pdf/2606.01260)**

> **作者:** Ikhlasul Akmal Hanif; Muhammad Falensi Azmi; Filbert Aurelian Tjiaranata; Eryawan Presma Yulianrifat; Fajri Koto
>
> **摘要:** Despite being home to more than 1300 ethnic groups and 700 indigenous languages, bias in Large Language Models has not been fully studied in Indonesia, thus leaving a critical gap in evaluating representational fairness and localized stereotypes within its uniquely vast, multilingual, and diverse sociocultural landscape. To address this, we introduce IndoBias as a culturally-grounded bias benchmark to assess LLMs bias in Indonesian and three local languages: Javanese, Sundanese, and Makasar. IndoBias features dual perspective evaluation tracks: depth-oriented (with contrastive-pairs) and breadth-oriented (with generation-based), where the latter is grounded in social science frameworks (SPI, O*NET, and WGI). Our results show that existing LLMs -- particularly decoder models -- exhibit strong bias towards prototypical sentences in Indonesian, while local languages suffer higher bias under Ideology and Religion category. We also find that LLMs responses exhibit a non-uniform Stereotype Polarity when prompted with various local entities. Finally, we discover that, in Indonesian, Common Crawl texts introduce more bias during pretraining, compared to human-reviewed article texts (e.g., Wikipedia, News), whereas introducing local languages to pretraining generally increases bias. This work highlights the importance of studying bias in culture-specific context. Warning: This paper contains example data that may be offensive, harmful, or biased.
>
---
#### [new 178] Argument Collapse: LLMs Flatten Long-Form Public Debate
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究LLMs在生成公共辩论文本时出现的“论点坍缩”现象，分析其导致论点趋同的问题。任务属于自然语言处理中的文本生成与分析，旨在探讨LLMs生成内容的多样性与结构特点。**

- **链接: [https://arxiv.org/pdf/2606.01736](https://arxiv.org/pdf/2606.01736)**

> **作者:** Yekyung Kim; Yapei Chang; Chau Minh Pham; Mohit Iyyer
>
> **摘要:** As LLMs are increasingly used to draft public-facing arguments, they may flatten public debate by repeatedly introducing the same polished, plausible arguments. We study argument collapse, the tendency of essays generated by different LLMs to converge to a smaller set of main arguments, sub-arguments, and paragraph-level structures. We compare 1,039 human responses from 195 New York Times (NYT) debates, 448 human responses from 61 longer-form Boston Review (BR) forums, and 23,384 LLM-generated essays. In the NYT corpus, 65.3% of human main arguments are unique within a debate, compared to 3.4% of LLM main arguments. Asking LLMs to generate diverse answers adds variation, but a typical model recovers only about half of the distinct human main arguments, with much of the added variation falling outside the observed human argument space. Collapse also appears in sub-arguments, where among essays with the same main argument, 41.0% of human sub-arguments are unique versus 9.1% from LLM responses. Qualitatively, LLMs often reuse generalized and hedged sub-arguments, while humans prefer more concrete and topic-specific ones. Structure-wise, LLM-generated essays tend to follow a more fixed arc, often opening with a direct claim and moving quickly toward proposals. The same patterns hold in longer BR essays, suggesting that argument collapse extends beyond short-form responses.
>
---
#### [new 179] CA-BED: Conversation-Aware Bayesian Experimental Design
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于对话系统任务，解决交互中信息获取问题。提出CA-BED框架，通过贝叶斯实验设计优化提问策略，提升推理成功率。**

- **链接: [https://arxiv.org/pdf/2606.01182](https://arxiv.org/pdf/2606.01182)**

> **作者:** Daniel Arnould; Rashad Aziz; Zixuan Kang; Tanav Changal; Kevin Zhu; Sunishchal Dev; Gabriel Grand; Shreyas Sunil Kulkarni
>
> **备注:** Reliable Autonomy Workshop at ICLR 2026
>
> **摘要:** Large Language Models (LLMs) excel at static reasoning tasks, yet their performance often degrades in interactive scenarios where information must be actively acquired through questioning. A key challenge lies in selecting questions that reduce uncertainty while incorporating responses that may be ambiguous or only partially informative. To address this, we propose Conversation-Aware Bayesian Experimental Design (CA-BED), an inference-time probabilistic dialog planning framework that integrates Bayesian Experimental Design with LLM-based likelihood estimation to optimize question selection over multiple conversational turns. CA-BED maintains a belief distribution over hypotheses, anticipates possible answers, and propagates expected information gain through a simulated conversation tree. Across two structured entity-deduction benchmarks, CA-BED yields an average 21.8% improvement in success rates over direct prompting, with comparable gains relative to alternative information-seeking methods. It achieves these gains with an average increase of only 1.8 conversational turns compared to direct prompting.
>
---
#### [new 180] Skill or Skip? Learning Selective Skill Invocation in Agentic Tasks via Dual-Granularity Preference Learning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于智能代理任务，解决技能调用选择问题。提出SelSkill框架，通过双粒度偏好学习决定是否调用技能，提升任务成功率和执行精度。**

- **链接: [https://arxiv.org/pdf/2606.00510](https://arxiv.org/pdf/2606.00510)**

> **作者:** Chishui Chen; Jiaye Lin; Te Sun; Junxi Wang; Yi Yang; Cong Qin; Yangen Hu; Lu Pan; Ke Zeng
>
> **备注:** 18 pages, 4 figures, 10 tables
>
> **摘要:** Agent skills are callable procedural modules that provide reusable knowledge and execution policies for complex agentic tasks. However, existing methods mainly focus on selecting relevant skills or improving the skills themselves, while overlooking whether a relevant skill should actually be invoked at the current decision point. Unhelpful invocations may introduce irrelevant context and disrupt an otherwise correct execution process. To address this issue, we propose SelSkill, a dual-granularity preference-learning framework for selective skill invocation. SelSkill formulates skill use as a skill-or-skip decision, uses predictive uncertainty to prioritize candidate decision points, and constructs controlled invoke-skip preference pairs from shared trajectory prefixes. It further combines episode-level outcome preferences with step-level invocation preferences to capture both overall trajectory quality and the local effectiveness of skill invocation. On ALFWorld with Qwen3-8B, SelSkill improves task success by 10.9 percentage points and execution precision by 29.1 percentage points. On BFCL, it improves task success by 5.7 percentage points and execution precision by 29.5 percentage points. Zero-shot results on Tau-bench and PopQA further suggest that the learned invocation policy transfers to new domains with previously unseen skills.
>
---
#### [new 181] Med-HEAL: Analyzing and Mitigating Hallucinations in Medical LLMs with Hallucination-Aware In-Context Learning
- **分类: cs.CL**

- **简介: 该论文属于医疗大模型 hallucination 问题研究，旨在分析并减轻医学 LLM 的幻觉现象。通过构建数据集和提出自检与检索增强方法进行实验验证。**

- **链接: [https://arxiv.org/pdf/2606.01301](https://arxiv.org/pdf/2606.01301)**

> **作者:** Yiming Liao; Zeno Franco; Jose Eduardo Lizarraga Mazaba; Keke Chen
>
> **备注:** 12 pages, 5 figures. Preprint full version of an accepted ACM-BCB 2026 short paper
>
> **摘要:** Hallucinations in medical large language models (LLMs) pose serious risks for clinical decision support, particularly when models must reason over complex electronic health records (EHRs). However, existing benchmarks often lack a realistic clinical context and provide limited insight into how hallucinations can be mitigated in practice. We introduce Med-HEAL, a framework for systematically identifying, analyzing, and mitigating hallucinations in medical LLMs using clinically grounded data. Building on the EHRNoteQA benchmark derived from MIMIC-IV discharge summaries, we construct a hallucination dataset by evaluating BioMistral-7B on open-ended clinical question answering tasks. Model outputs are labeled through a dual evaluation pipeline that combines LLM-as-a-Judge assessment (GPT-4o) with human auditing by medical student reviewers, producing correctness judgments and annotations of reasoning errors via a custom web-based evaluation system. We then leverage this dataset to investigate mitigation strategies: a self-critique pipeline, in which the test model reviews its own answers to detect potential errors and regenerates responses for flagged cases, and retrieval-augmented in-context learning (RA-ICL), which exposes the model to hallucinated and corrected examples. Experiments across five open-source LLMs-BioMistral, Llama-3.1, DeepSeek, Qwen2.5, and Qwen3, show that the self-critique strategy improves accuracy for three of five models (p < 0.05) without requiring parameter updates. Med-HEAL provides both a reusable hallucination dataset and a practical framework for studying and mitigating hallucinations in medical LLMs, supporting safer deployment of AI systems in clinical environments. Our code and data are publicly available at this https URL.
>
---
#### [new 182] Unified Context Evolution for LLM Agents
- **分类: cs.CL**

- **简介: 该论文提出UCE框架，解决LLM代理在多步骤任务中经验无法积累的问题。通过构建类型化上下文单元库，提升任务成功率。属于强化学习与智能体研究领域。**

- **链接: [https://arxiv.org/pdf/2606.02304](https://arxiv.org/pdf/2606.02304)**

> **作者:** Zixuan Zhu; Yitong Hu; Yong Dai; Junfeng Fang; Chunyang Jiang; Senkang Hu; Yuzhi Zhao
>
> **摘要:** LLM-based agents can solve multi-step interactive tasks by combining reasoning with environment feedback, yet each episode starts from the same fixed context and any useful strategy discovered along the way is lost once the task ends. Existing approaches either limit learning to the current task or pool all experience into a single untyped store, without distinguishing knowledge types, tracking quality through use, or balancing what the library still lacks. We introduce Unified Context Evolution (UCE), a gradient-free framework that externalizes agent experience into an evolving library of typed Evolvable Context Units (ECUs). UCE decomposes experience into four complementary types (Memory, Strategy, Workflow, and Skill), each generated from trajectories under type-specific conditions, retrieved at decision time, scored through repeated usage outcomes, and pruned when no longer valuable. A scheduling module allocates each cycle's generation budget toward the types where the library is weakest. Across two interactive benchmarks, UCE raises ALFWorld success from 75.4% to 96.3% and WebShop task score from 45.1% to 61.3%, and the accumulated library transfers to alternative actor backbones without retraining.
>
---
#### [new 183] Multilinguality of Large Language Models From a Structural Perspective
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，探讨LLM的多语言能力。研究解决如何从结构角度理解LLM处理不同语言的方式，通过分析结构差异揭示语言资源对模型的影响。**

- **链接: [https://arxiv.org/pdf/2606.01800](https://arxiv.org/pdf/2606.01800)**

> **作者:** Haruki Sakajo; Yusuke Sakai; Hidetaka Kamigaito; Taro Watanabe
>
> **摘要:** Large language models (LLMs) have excelled in processing multiple languages through pre- and post-training on multilingual data, even though English dominates the training data. Prior work focusing on token representations has revealed how those LLMs process non-English text. Although these analyses have provided insightful findings, they fail to capture a structural view, which is an inherent property of language. In this study, we explore the multilinguality of LLMs through representational structural analysis. Our findings reveal that low-resource languages are structurally more different from English than high- and mid-resource languages, and that language-specific post-training alters their structures while preserving inter-language relationships.
>
---
#### [new 184] PortBERT: Navigating the Depths of Portuguese Language Models
- **分类: cs.CL**

- **简介: 该论文提出PortBERT，一种针对葡萄牙语的高效语言模型，解决语言特定模型在性能与效率间的平衡问题。通过优化训练和推理效率，提升葡萄牙语自然语言处理效果。**

- **链接: [https://arxiv.org/pdf/2606.02100](https://arxiv.org/pdf/2606.02100)**

> **作者:** Raphael Scheible-Schmitt; Henry He; Armando B. Mendes
>
> **摘要:** Transformer models dominate modern NLP, but efficient, language-specific models remain scarce. In Portuguese, most focus on scale or accuracy, often neglecting training and deployment efficiency. In the present work, we introduce PortBERT, a family of RoBERTa-based language models for Portuguese, designed to balance performance and efficiency. Trained from scratch on over 450 GB of deduplicated and filtered mC4 and OSCAR23 from CulturaX using fairseq, PortBERT leverages byte-level BPE tokenization and stable pre-training routines across both GPU and TPU processors. We release two variants, PortBERT base and PortBERT large, and evaluate them on ExtraGLUE, a suite of translated GLUE and SuperGLUE tasks. Both models perform competitively, matching or surpassing existing monolingual and multilingual models. Beyond accuracy, we report training and inference times as well as fine-tuning throughput, providing practical insights into model efficiency. PortBERT thus complements prior work by addressing the underexplored dimension of compute-performance tradeoffs in Portuguese NLP. We release all models on Huggingface and provide fairseq checkpoints to support further research and applications.
>
---
#### [new 185] Detection vs. Execution: Single-Bucket Probes Miss Half the Mamba-2 State Sink
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于机制可解释性任务，研究Mamba-2中探测与执行的差异问题。工作表明单桶探测器仅捕获部分执行层，遗漏大量具有相同表征签名的检测层，揭示了表征相似性不等于功能等价。**

- **链接: [https://arxiv.org/pdf/2606.00930](https://arxiv.org/pdf/2606.00930)**

> **作者:** Yuhang Jiang
>
> **备注:** 16 pages, 3 figures
>
> **摘要:** Mechanistic interpretability often assumes that probes identifying a representational signature also identify the circuit executing the corresponding computation. We show that this assumption can fail systematically in Mamba-2. Studying the state sink (disproportionate Delta-gate activation on boundary tokens, analogous to the attention sink), we find that single-bucket probes recover only a small execution layer while missing a much larger detection layer with the same representational signature. In Mamba-2, the state sink decomposes into two functional head sets. Single-bucket BOS-specialist heads (about 5% of heads at 2.7B) causally support both BOS-context and newline-target predictions across model scales and corpora. Dual heads (27-35% of heads, recovered by multi-class aggregation of the same probe) show stronger BOS-newline representational similarity but substantially weaker causal effects under ablation. Representational similarity does not imply functional equivalence. This distinction matters for downstream behaviour: ablating BOS-specialist heads collapses RULER NIAH retrieval accuracy from 1.00 to 0.00 at 1024 context length in both Mamba-1 2.8B and Mamba-2 2.7B, while size-matched complements preserve baseline performance. A random channel-bucketing control rules out substrate granularity alone, implicating Mamba-2's head-shared Delta projection. Probe-derived specialty can identify execution circuits; at coarse granularity the same probe also recovers detection circuits, and separating them requires class-conditional ablation rather than class-conditional cosine.
>
---
#### [new 186] Enhancing BiGRU with a KAN Block for Legal Document Classification and Summarization
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出一种结合KAN模块的BiGRU模型，用于法律文档分类与摘要任务，解决多语言、长依赖和类别不平衡问题。**

- **链接: [https://arxiv.org/pdf/2606.00116](https://arxiv.org/pdf/2606.00116)**

> **作者:** Ahmed Faizul Haque Dhrubo; Souvik Pramanik; Most. Aysha Siddika Sumona; Shahnewaz Siddique; Mohammad Ashrafuzzaman Khan; Mohammad Abdul Qayum; Mohsin Sajjad
>
> **备注:** This paper contains of 10 pages, 10 figures, 4 tables and version 2 after it review from ACL 2026
>
> **摘要:** This study introduces a novel architecture of KAN-based BiGRU model for the task of classification and summarization of legal documents in a low-resource multilingual setup. In order to tackle problems associated with domain language, the usage of different languages, long dependencies within context, and class imbalance, we employ the dataset composed of legal documents from Bangladesh and taken from Manupatra, which include Bengali, English, and transliterated Bengali languages. Our classification task involves BiGRU model, along with Kolmogorov-Arnold Network (KAN) module, while the summarization part utilizes attention-based GRU, combined with a KAN model head. Classification model yields 67.96% of accuracy and 0.65 F1 score; while ROUGE-1, ROUGE-2, and ROUGE-L measures for summarization yield 0.38, 0.23, and 0.31 F1 scores, correspondingly. Ablation study shows that the use of KAN increases classification accuracy from 57.34% to 67.96%. Moreover, our proposed technique is compared to several baselines, including classical ML algorithms and pretrained language models.
>
---
#### [new 187] Mitigating Bias in Locally Constrained Decoding via Tractable Proposals
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，解决大模型生成不符合约束的问题。通过构建有效提议分布，提升基于SMC的解码效率与准确性。**

- **链接: [https://arxiv.org/pdf/2606.01926](https://arxiv.org/pdf/2606.01926)**

> **作者:** Meihua Dang; Linxin Song; Honghua Zhang; Jieyu Zhao; Guy Van den Broeck; Stefano Ermon
>
> **备注:** 13 pages, 5 figures
>
> **摘要:** Generations from large language models often fail to conform to desired constraints such as JSON schema. Existing locally constrained decoding (LCD) approaches enforce constraints by myopically masking out next tokens, resulting in biased sampling and degradation in performance. Recent work uses sequential Monte Carlo (SMC) methods to mitigate such biases, but designing effective proposal distributions or potential functions remains a key challenge. In this work, we propose a generic approach to construct proposals and potentials for SMC sampling from $p_{\mathrm{lm}}( \cdot \mid \mathrm{constraint})$. First, we show that constraints specified as finite automata can be tensorized for efficient execution on GPUs, which we use to construct globally constrained decoding (GCD) proposals. In addition, leveraging the fact that tensorized finite automata share the same circuit structure as hidden Markov models, we circuit-multiply them to obtain the probabilistic GCD (P-GCD) proposals encoding both logical and probabilistic information about the target distributions. We evaluate (P-)GCD on the tasks of function calling, keyword-based generation, and SQL generation. Experiments show that under the same SMC sampling setup, compared to LCD proposals, (P-)GCD converges faster to the target distribution with significantly fewer particles.
>
---
#### [new 188] When Knowledge Is Not Free: Cost-Aware Evidence Selection in Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文研究成本感知的检索增强生成（RAG）任务，解决高价值知识访问成本问题。通过引入成本层级和预算限制，探索高效证据选择方法。**

- **链接: [https://arxiv.org/pdf/2606.02245](https://arxiv.org/pdf/2606.02245)**

> **作者:** Mingyan Wu; Han Yang; Omer Ben-Porat; Yftah Ziser
>
> **摘要:** Retrieval-Augmented Generation (RAG) typically assumes that external knowledge is free, but many high-quality sources are paywalled, licensed, restricted, or otherwise costly to access. We introduce cost-aware RAG, a setting where retrieved evidence is assigned access-cost tiers and systems must answer under an explicit evidence-access budget. We instantiate this setting by augmenting MS MARCO v2.1 with access-friction tiers and evaluate budgeted evidence selection across general-domain and domain-specific QA benchmarks. Our results show that static selection is brittle: no fixed selector uniformly dominates, and larger budgets do not reliably improve answer quality, even when costly evidence is domain-matched. We then study agentic cost-aware RAG, where an LLM decides when to retrieve, which tier to access, and when to stop. Agents show strong promise as adaptive evidence-acquisition controllers, but their behavior remains highly model- and task-dependent. These findings suggest that cost-aware evidence acquisition is a central challenge for the next generation of RAG systems. All code and data are available at this https URL.
>
---
#### [new 189] Finer Parameter Steps for Low-Rank PEFT: A Controlled Study with CP Tensor Adapters
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于参数高效微调（PEFT）任务，旨在研究更细粒度的参数调整对模型性能的影响。通过对比CP张量适配器与LoRA，探索不同任务下的精度-预算权衡。**

- **链接: [https://arxiv.org/pdf/2606.00428](https://arxiv.org/pdf/2606.00428)**

> **作者:** Xinjue Wang; Xiuheng Wang; Yejun Zhang; Sergiy A. Vorobyov; Esa Ollila; Zhi-Yong Wang
>
> **备注:** Accepted at the ICML 2026 Workshop on CoLoRAI
>
> **摘要:** Low-rank adapters are usually compared by sweeping a small set of ranks, but the rank also fixes the resolution of the parameter budget. For a $2048{\times}2048$ OPT attention projection, increasing LoRA by one rank stores $4096$ trainable scalars, leaving large gaps between feasible low-budget adapter sizes. This paper asks whether a tensorized adapter with finer capacity increments changes the observed accuracy--budget trade-off. We instantiate this question with fixed-component canonical polyadic (CP) tensor adapters. Under a $32{\times}64{\times}32{\times}64$ tensorization, one normalized CP component stores $193$ trainable scalars per projection, about $21$ times smaller than one LoRA rank step. We compare CP adapters and LoRA on OPT-1.3B across SST-2, RTE, and BoolQ under matched target modules, training protocol, data caps, and seed schedules. CP trains stably and fills the gaps between LoRA ranks, but the effect is task-dependent: SST-2 reaches an early low-budget plateau, BoolQ benefits from additional CP components before saturating slightly below LoRA, and RTE remains LoRA-favored. Finer parameter steps are therefore useful for diagnosing PEFT budget sensitivity, but they do not by themselves guarantee a better accuracy--budget curve.
>
---
#### [new 190] Beyond Sinusoids: A Morlet Wavelet Framework for Transformer Positional Encoding
- **分类: cs.LG; cs.CL; eess.SP**

- **简介: 该论文提出一种基于Morlet小波的Transformer位置编码方法，解决传统位置编码无法灵活控制位置影响范围的问题。**

- **链接: [https://arxiv.org/pdf/2606.01258](https://arxiv.org/pdf/2606.01258)**

> **作者:** Athanasios Zeris
>
> **备注:** 16 pages, 4 figures, 4 tables
>
> **摘要:** Standard positional encodings for transformers - sinusoidal and rotary (RoPE) - treat every position as equally local: they encode where a token is, but not how far its positional influence should extend. We propose that the Morlet wavelet, which simultaneously minimises uncertainty in position and frequency, is the natural basis for positional encoding, and introduce Morlet Positional Encoding (MoPE): each embedding dimension learns its own frequency and locality bandwidth from data. The main theoretical result is a unification: sinusoidal PE and the RoPE correlation kernel both emerge as limiting cases of MoPE when locality is switched off (sigma_i -> infinity). The phase of MoPE recovers the RoPE rotation angle exactly; the amplitude adds a learned Gaussian locality kernel that standard encodings lack. Empirically, MoPE combined with Energy-Gated Attention achieves +0.119 improvement over standard attention on TinyShakespeare, outperforming either component alone. Analysis of the learned parameters reveals that all 128 frequency-bandwidth pairs converge to the wavelet admissibility boundary - an empirical observation consistent with a companion result on energy gating, suggesting a reproducible property of character-level language signals that warrants further investigation.
>
---
#### [new 191] AGENTCL: Toward Rigorous Evaluation of Continual Learning in Language Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于持续学习任务，旨在解决语言代理在连续任务中有效积累和重用经验的问题。提出AgentCL框架，通过控制任务流评估记忆设计效果。**

- **链接: [https://arxiv.org/pdf/2606.02461](https://arxiv.org/pdf/2606.02461)**

> **作者:** Yiheng Shu; Bernal Jiménez Gutiérrez; Saisri Padmaja Jonnalagedda; Yuguang Yao; Huan Sun; Yu Su
>
> **备注:** 10 pages
>
> **摘要:** Language agents spend substantial inference time solving individual tasks, yet the experience acquired in one episode is often underutilized in future episodes. Continual learning expects an agent to accumulate reusable experience across a stream of tasks, improve over time, and avoid interference from irrelevant experiences. Unfortunately, existing benchmarks struggle to evaluate continual learning in language agents rigorously. Most efforts focus on retrieval and reasoning over long-context conversations or documents, while recent lifelong-adaptation benchmarks often rely on naive task streams with limited analysis of cross-task relationships, making it difficult to understand what an agent learns and reuses over time. This paper presents an evaluation framework AgentCL for continual learning in agents, centered on controlled task streams and metrics for transfer gains. AGENTCL constructs compositional streams where earlier sub-solutions, evidence, or workflows are intentionally reusable in later tasks, and contrasts them with naive streams where such reusability is not guaranteed. We use the benchmark to evaluate non-parametric memory designs for continual learning. To diagnose how memory design choices affect continual learning, we develop MemProbe, a probing method that stores interactions, insights, and skills, while filtering unreliable experiences during consolidation. Empirical analysis across coding, deep research, and language understanding/reasoning tasks shows that naive streams offer limited ability to distinguish memory designs, whereas controlled streams more clearly distinguish their plasticity. Meanwhile, naive and held-out settings often yield limited gains and can expose memory-induced degradation. These results highlight the need for stronger memory designs that balance plasticity and stable reuse.
>
---
#### [new 192] Task Structure Reverses Layerwise State Encoding in Sequence Models
- **分类: cs.LG; cs.CL**

- **简介: 论文研究序列模型中层间状态编码的结构，探讨任务变化如何影响编码方式。通过实验发现不同任务下编码模式会发生反转，揭示了计算结构而非代数性质的影响。**

- **链接: [https://arxiv.org/pdf/2606.00926](https://arxiv.org/pdf/2606.00926)**

> **作者:** Yuhang Jiang
>
> **备注:** 20 pages, 11 figures, 8 tables
>
> **摘要:** Mechanistic studies of sequence models often treat layerwise state encodings as architectural traits: recurrent models concentrate readable state, attention-based models distribute it. We find that the same architecture reverses this profile when the task changes. Across Transformers, Mamba, Mamba-2, LSTMs, and GRUs, Parity is concentrated late in Mamba and the recurrent baselines and built gradually by Transformer; on bounded-depth Dyck-k the pattern flips. The same flip appears in fine-tuned Mamba-130M and Pythia-160M, and the Pythia Dyck bottleneck persists at 410M. Two explanations are conflated in the literature: algebraic structure (commutativity) versus computational structure (prefix update vs. stack). To separate them we add a third task: non-commutative S_3 permutation composition. S_3 groups with Parity, not Dyck, on layerwise probing across all five architectures and on Mamba-specific Conv1D attribution, so the grouping tracks computational structure rather than commutativity. Causal interventions show that, in the 4-layer formal models, linearly readable directions are often functionally necessary and can remain important at out-of-distribution lengths on Parity and Dyck. At pretrained scale the picture splits. Fine-tuned Pythia Dyck has a strong middle-layer bottleneck (L6-L7 ablation drops accuracy by roughly 81% at 160M; broader L4-L18 plateau at 410M), far weaker at the best-probe layer. Pretrained Mamba shows the complementary failure mode: its final layer is highly readable, no single probe direction breaks the task on Parity, Dyck, or S_3, yet mid-position activation patching there recovers about 97-98% of the clean-corrupted logit gap. Probing localizes where state is linearly available, not always where the computation is bottlenecked. Mechanistic signatures are properties of architecture and task together.
>
---
#### [new 193] OpenWebRL: Demystifying Online Multi-turn Reinforcement Learning for Visual Web Agents
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出OpenWebRL框架，解决视觉网页代理的在线多轮强化学习问题，通过实时网站训练提升代理性能。**

- **链接: [https://arxiv.org/pdf/2606.02031](https://arxiv.org/pdf/2606.02031)**

> **作者:** Rui Yang; Qianhui Wu; Yuxi Chen; Hao Bai; Wenlin Yao; Hao Cheng; Baolin Peng; Huan Zhang; Tong Zhang; Jianfeng Gao
>
> **备注:** 36 pages, 11 figures
>
> **摘要:** Building capable visual web agents requires long-horizon reasoning, precise grounding, and robust interaction with dynamic real-world websites. Despite rapid progress, the strongest systems remain largely proprietary, while open agents still depend heavily on supervised post-training over large collections of curated web trajectories. This dependence creates a major scalability bottleneck: high-quality demonstrations are expensive to collect, and static datasets offer limited coverage of the diverse, ever-changing open web. Although online RL has shown promise for text-based agents, its potential for training visual web agents directly on live websites remains largely underexplored. In this paper, we introduce OpenWebRL, an open framework for training visual web agents with online multi-turn RL on real websites. OpenWebRL covers the full training pipeline, including scalable live-browser infrastructure, supervised initialization, multimodal context management, trajectory-level success judging, and efficient multi-turn policy optimization. Using this framework, we train OpenWebRL-4B, which establishes a new open-source state of the art on challenging live-web benchmarks. With only 0.4K initialization trajectories and 2.2K open-ended RL training tasks, OpenWebRL-4B achieves 67.0% success on Online-Mind2Web and 64.0% on DeepShop, outperforming prior open agents of similar or larger scale and remaining competitive with proprietary systems including OpenAI CUA and Gemini CUA. Beyond strong benchmark performance, we systematically study the key design choices that make online RL effective for visual web agents, and analyze how RL improves agentic reasoning. Overall, our work offers a practical path toward building more capable, reproducible, and cost-efficient open web agents. We will release our training data, models, and code to support future research.
>
---
#### [new 194] OmniOPD: Logit-Free On-Policy Distillation via Speculative Verification
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出OmniOPD，解决教师模型无法提供token级logit的问题，通过chunk级语义验证提升学生模型性能，属于模型蒸馏任务。**

- **链接: [https://arxiv.org/pdf/2606.01476](https://arxiv.org/pdf/2606.01476)**

> **作者:** Yuhang Zhou; Lizhu Zhang; Yifan Wu; Mingyi Wang; Peng Bo; Jiayi Liu; Xiangjun Fan; Zhuokai Zhao
>
> **备注:** 26 pages, 3 figures
>
> **摘要:** On-Policy Distillation (OPD) trains a student model on its own generative trajectories under dense token-level feedback from a stronger teacher, mitigating both the off-policy distribution shift of Supervised Fine-Tuning (SFT) and the sparse credit assignment of Reinforcement Learning (RL). However, standard OPD faces two coupled limitations. First, it requires direct access to the teacher's token-level logits, excluding a broad class of capable proprietary models from serving as teachers. Second, the token-level logit signal itself is brittle, depending on a narrow overlap of plausible next tokens between teacher and student, and prone to amplifying degenerate patterns such as repetition loops. In this paper, we introduce OmniOPD, a novel framework that addresses both limitations through a logit-free, chunk-level supervision signal. OmniOPD replaces deterministic logit matching with Monte Carlo rollouts that approximate the teacher's local preferences through a continuous semantic similarity metric over multi-token chunks, and concentrates this supervision via a peak-entropy scheduler that audits the student only at its high-uncertainty reasoning forks. A Dirichlet-Multinomial Bayesian prior and a base-model KL anchor further bound the variance of discrete sampling and prevent policy collapse across unaudited tokens. Across competitive benchmarks, OmniOPD surpasses the standard OPD approach by up to +28.64% on math, confirming that chunk-level semantic verification extracts a more reliable learning signal than token-level logit matching, whose high information density is offset by significant noise and brittleness. Furthermore, when paired with stronger black-box teachers such as Claude-4.5-Haiku and Gemini-2.5-Flash, OmniOPD achieves an additional +9.54% relative on math over its open-weight teacher counterpart, advancing the student past the performance of self-exploratory RL.
>
---
#### [new 195] On the Scaling of PEFT: Towards Million Personal Models of Trillion Parameters
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究参数高效微调（PEFT）在构建持久个人模型中的应用，解决如何有效扩展和管理大量小型适配器的问题。属于模型优化任务。**

- **链接: [https://arxiv.org/pdf/2606.02437](https://arxiv.org/pdf/2606.02437)**

> **作者:** Mind Lab; Song Cao; Vic Cao; Kaijie Chen; Bunny Fan; Hera Feng; Huan Feng; Arthur Fu; Jun Gao; Hongquan Gu; Aaron Guan; Mutian Hong; Hailee Hou; Peixuan Hua; Charles Huang; Miles Jiang; Nora Jiang; Yuyi Jiang; Autumn Jin; Fancy Kong; Kyrie Lei; Alexy Li; Dawn Li; Ray Li; Theo Li; Wenhao Li; Jiayi Lin; Domini Liu; Heshan Liu; Kairus Liu; Logan Liu; Maeve Luo; Runism Lv; Pony Ma; Verity Niu; Anson Qiu; Vincent Wang; Maxwell Yao; Regis Ye; Wenlin Ye; Yanying Ye; Josh Ying; Danney Zeng; Salmon Zhan; Anya Zhang; Ruijia Zhang; Shiyang Zhang; Sueky Zhang; Ya Zhang; Wei Zhao; Ada Zhou; Sizer Zhou; Xinyue Zhu; Murphy Zhuang
>
> **摘要:** Parameter-efficient fine-tuning (PEFT) is usually treated as a cheaper alternative to full fine-tuning. We study a broader role: small trainable adapters as persistent local state on top of strong shared foundation models. In this framing, the base model provides shared competence while adapters carry instance-specific behavior such as preferences, skills, tool habits, and memory-like updates. We organize the problem around three scaling axes: Scale Up, where stronger shared priors make small local updates more useful; Scale Down, where we study how small adapters can be while remaining reliable; and Scale Out, where many persistent adapted instances coexist. MinT provides one infrastructure example for managing adapter identity, revision, provenance, evaluation, and serving residency. Together, the results suggest that PEFT can be a compact substrate for persistent personal models rather than only a budget substitute for full fine-tuning.
>
---
#### [new 196] InfoMerge: Information-aware Token Compression for Efficient Video Large Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视频大模型压缩任务，旨在解决视觉token冗余导致的计算效率低问题。提出InfoMerge方法，通过时间指纹差和内容感知预算分配，提升token利用率，实现高效压缩。**

- **链接: [https://arxiv.org/pdf/2606.02161](https://arxiv.org/pdf/2606.02161)**

> **作者:** Xinxin Liu; Shiwei Gan; Xiao Liu; Yafeng Yin; Lei Xie; Sanglu Lu
>
> **备注:** 15 pages, 8 figures
>
> **摘要:** Video Large Language Models (Video-LLMs) achieve strong performance in video understanding, but their excessive visual tokens bring substantial computational overhead. Existing training-free compression methods improve inference efficiency by reducing visual tokens, yet they often rely on local adjacent-frame similarity for temporal redundancy estimation or allocate token budgets mainly according to segment length. Such designs are sensitive to frame-level noise and fail to capture the non-uniform information distribution of real-world videos. To address these challenges, we propose InfoMerge, a training-free visual token compression method that improves token utilization through robust redundancy estimation and content-aware budget allocation. Specifically, we propose the Temporal Fingerprint Difference: a segment-level second-order temporal redundancy estimation strategy, which models the temporal similarity structure of tokens at the same spatial positions within each segment. We further introduce Content-Aware Budget Allocation (CABA), which dynamically allocates segment-level token budgets based on segment uniqueness and spectral-entropy-based representational richness. By reducing repeated preservation of redundant static regions and allocating more tokens to informative segments, InfoMerge makes better use of the limited token budget while maintaining strong performance. Extensive experiments show that InfoMerge achieves strong efficiency--accuracy trade-offs across multiple benchmarks and backbones, with more pronounced advantages under aggressive compression. On LLaVA-OneVision-7B, InfoMerge retains 98.8\% of the original average performance while reducing 85\% of visual tokens and achieving a 4.24-fold speedup in the prefill stage.
>
---
#### [new 197] The Invisible Coalition Partner: How LLMs Vote When Democracy Gets Concrete
- **分类: cs.CY; cs.CL**

- **简介: 论文研究LLMs在具体政策决策中的投票行为，挑战其左翼偏见的假设。通过对比问卷与实际公投数据，发现LLMs更偏向中间派，表现出保守和不一致。任务为评估LLMs政治倾向的可靠性。**

- **链接: [https://arxiv.org/pdf/2606.00048](https://arxiv.org/pdf/2606.00048)**

> **作者:** Joel Barmettler
>
> **备注:** 13 pages, 10 figures. Preprint. Code and data: this https URL
>
> **摘要:** Prior research has established that instruction-tuned large language models exhibit left-of-center political bias, measured exclusively through abstract political questionnaires. We show that this finding does not generalize to concrete policy decisions. We introduce a dual-instrument methodology grounded in Swiss democratic reality. The Smartvote questionnaire (75 abstract policy questions) is administered to 66 LLMs from 27 model families and compared to 184 elected members of the Swiss National Council, replicating the established leftward convergence (Cohen's d = 3.64, p = 0.0002). Then, novel to this work, 9 flagship LLMs are confronted with 48 real federal referenda (Volksabstimmungen) in four national languages (German, French, Italian, Romansh) under three information conditions, comparing votes to actual outcomes and party recommendations (Parolen). Three findings challenge the prevailing narrative. (1) Abstract questionnaires do not predict concrete behavior: the left-to-right agreement gradient on Smartvote shifts from left-peaked to center-peaked on Volksabstimmungen, where models align most with centrist Die Mitte and FDP rather than leftist SP and Gruene (Wilcoxon p = 0.008). (2) For some models, the language of a political question changes the answer more than the political content does: cross-linguistic consistency ranges from 50% (Mistral) to 98% (GPT-5.4). (3) Two models exhibit systematic change-aversion rather than political bias, voting Nein on 83-94% of referenda regardless of direction (binomial p < 0.0001). What prior work measured as "leftward bias" may not generalize beyond abstract instruments. On concrete policy decisions, LLMs behave less like coalition partners of the left and more like cautious civil servants: centrist, status-quo-favoring, and inconsistent across languages.
>
---
#### [new 198] SafeSteer: Localized On-Policy Distillation for Efficient Safety Alignment
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于大语言模型安全对齐任务，旨在解决对齐导致模型能力下降的问题。通过局部化蒸馏方法SafeSteer，仅在安全相关token上进行优化，提升安全性同时保持模型通用能力。**

- **链接: [https://arxiv.org/pdf/2606.02530](https://arxiv.org/pdf/2606.02530)**

> **作者:** Hao Li; Jingkun An; Zijun Song; Pengyu Zhu; Rui Li; Hao Wang; Wendi Feng; Yesheng Liu; Lijun Li; Jin-Ge Yao; Lei Sha
>
> **备注:** 19 pages, 8 figures, 14 tables. Submitted to EMNLP 2026
>
> **摘要:** Aligning Large Language Models (LLMs) with human values often degrades their general capabilities, termed the alignment tax. Existing methods mitigate this by balancing dual objectives, which heavily rely on massive general-purpose data or auxiliary reward models. In this paper, we argue that, because safety features are inherently sparse within the output distribution, alignment requires localized modifications rather than global trade-offs. To this end, we propose SafeSteer, which performs on-policy distillation confined to safety tokens. First, we construct a safety teacher via activation steering. Based on this teacher, we develop a safety token selection algorithm. Consequently, SafeSteer restricts the reverse KL penalty to these tokens during training to preserve general capabilities. Experimental results across diverse models show that our SafeSteer achieves a superior trade-off between safety and general capability compared with existing methods, attaining strong safety performance on seven safety benchmarks with only minimal degradation on five general capability benchmarks. Notably, SafeSteer requires only 100 harmful samples without using any general-purpose data, less than 1% of what previous baselines used, considerably reducing alignment cost. More details are on our project page at this https URL.
>
---
#### [new 199] On Wednesdays, We Ask Questions: Optimizing "Active Listening" in Automated Legal Triage and Referral
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 论文研究法律咨询中的自动分类与转介系统，解决如何优化主动倾听以提高分类准确性。通过评估不同模型生成问题的效果，发现需引入高成本模型提升质量。**

- **链接: [https://arxiv.org/pdf/2606.00272](https://arxiv.org/pdf/2606.00272)**

> **作者:** Quinten Steenhuis; Jacqueline Harvey
>
> **备注:** Working paper submitted as accepted to AIDA2J workshop at International Conference for AI and Law in Singapore, June 2026
>
> **摘要:** The FETCH classifier generates follow-up questions to help refine the best match for the applicant's legal problem, using a low-cost ensemble of LLMs. In this paper, we describe an expert attorney and LLM-assisted evaluation of the follow-up question approach in FETCH and show that while low-cost LLMs perform well at classification tasks, generating high-quality plain-language questions in this setting appears to require a more sophisticated and higher-cost model. Through discussion with legal intake workers, we propose a rubric for the evaluation of legal intake classification questions, and we find that prompt engineering alone is not enough to improve question quality for intake purposes. We also find that LLM-as-judge and human ratings diverge. We demonstrate that with the addition of a single high-cost model, GPT-5, the classifier can elicit relevant information from applicants for legal help, and that the questions lead to more accurate performance at classification tasks. We also find uneven fact elicitation across different categories, including domestic violence, at odds with family law screening protocols, suggesting the value of including dedicated screening panels for certain areas of law.
>
---
#### [new 200] CART: Context-Anchored Recurrent Transformer -- A Parameter-Efficient Architecture with Learned Stability
- **分类: cs.LG; cs.CL**

- **简介: 论文提出CART，一种参数高效的语言模型，通过重复使用共享核心块来减少参数量。解决参数效率与模型性能之间的平衡问题，通过实验验证其有效性及局限性。**

- **链接: [https://arxiv.org/pdf/2606.01495](https://arxiv.org/pdf/2606.01495)**

> **作者:** Chad A. Capps
>
> **备注:** 31 pages, 4 figures. Code, training scripts, and the full experiment database (this http URL) are available at this https URL
>
> **摘要:** We present CART (Context-Anchored Recurrent Transformer), a parameter-efficient language model that reuses a single shared core block R times across depth. Unlike prior looped transformers that recompute key-value tensors at every iteration, CART computes K and V once from a multi-layer prelude and has the recurrent core cross-attend to those frozen tensors via multi-head latent attention. A learned Linear Time-Invariant (LTI) gate keeps the recurrence stable: its spectral radius settles in a narrow band (rho in [0.79, 0.83]) across all 36 fully-trained configurations. We evaluate CART on single consumer GPUs in two stages: a 64-configuration screen at 3,000 steps, then 36 configurations (P=6, R in {6,8,10}, three seeds) trained for 30,500 steps (~1B tokens). Two patterns hold across widths d in {256,512,768,1024}: prelude depth P dominates loop count R, and the Stage-1 ranking of R reverses at full training (R=6 becomes best at d>=512). At the binding d=1024 parameter-parity test, CART does not beat a parameter-matched dense baseline, losing by 1-2% at stored-parameter parity and by ~10% at effective-parameter parity. Diagnostic ablations split the effective-parameter gap into ~5% from weight sharing and a residual ~5% from the heterogeneous prelude/anchor/core/coda framing; the recurrent-core machinery (hyper-connections, LTI gate, loop-index embedding) is individually vestigial. Variable-R inference degrades on both sides of the trained R, a negative result for test-time depth scaling under this recipe.
>
---
#### [new 201] A Local Perturbation Theory for Cross-Domain Interference and Recovery in Multi-Domain RL
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于多领域强化学习任务，解决跨领域干扰与恢复问题。通过分析参数更新机制，提出局部扰动理论，实现领域性能的精准恢复。**

- **链接: [https://arxiv.org/pdf/2606.02398](https://arxiv.org/pdf/2606.02398)**

> **作者:** Lei Yang; Siyu Ding; Deyi Xiong
>
> **摘要:** Reinforcement learning (RL) post-training improves large language models (LLMs) on individual domains such as mathematical reasoning, code generation, question answering, and creative writing (CW), but training on one domain often degrades performance on others. Existing explanations based on catastrophic forgetting or global gradient conflict are incomplete: substantial interference can occur even when full-model gradients are nearly orthogonal. We show that single-domain RL produces sparse, small-magnitude parameter edits with weak overlap among top-changed neurons, while different domains still share substantial active computation routes on which update directions determine whether they act synergistically or conflict. Guided by this observation, we prove under a local perturbation model of multi-domain RL that later-domain training harms an earlier domain mainly through a second-order damage term, which under the observed sparse route structure concentrates in a low-dimensional shared conflict subspace. Moreover, a short domain refresh contracts the harmful component on this subspace, enabling selective recovery with limited collateral damage. Consistent with the theory, a brief Re-Math refresh after Code $\rightarrow$ Math $\rightarrow$ QA $\rightarrow$ CW recovers Math from 57.66 to 66.04 while largely preserving performance on the other domains, yielding the best average score of 66.39. Beyond refresh, a training-free rollback on a sparse proxy conflict coordinate set for the Math-QA pair partially restores Math, providing direct proxy-level evidence for localized damage. These results provide a localized mechanistic account of interference and recovery in multi-domain RL.
>
---
#### [new 202] Multi-Agent Computer Use
- **分类: cs.MA; cs.CL; cs.LG**

- **简介: 该论文研究多智能体计算机使用系统（MACU），解决单智能体在复杂长任务中的效率与协调问题。通过任务分解与并行执行，提升任务完成效果与速度。**

- **链接: [https://arxiv.org/pdf/2606.01533](https://arxiv.org/pdf/2606.01533)**

> **作者:** Jing Yu Koh; Ruslan Salakhutdinov; Daniel Fried
>
> **摘要:** Computer use agents (CUAs) today are primarily deployed as single serial agents. This setup is suboptimal for complex long-horizon tasks that benefit from task decomposition, parallel execution, and consistent re-planning based on new information. In this paper, we argue that we should instead move towards evaluating and building multi-agent computer use (MACU) systems. These systems, which emphasize planning and parallel execution, alleviate many of the shortcomings of single-agent CUAs. We propose a general multi-agent setup in which a manager model decomposes computer use tasks as a directed acyclic graph (DAG), encoding relevant dependencies and goals for subagents. At each iteration, the manager dispatches parallel CUA subagents to carry out nodes on the ready frontier of the DAG, and continuously revises the DAG (adding, canceling, or rewriting nodes) as new findings arrive from subagents. This design treats the partially observable environment of computer use as a first class challenge: information that downstream agents may not be able to re-observe are retained and passed forward through the manager and DAG structure. We demonstrate that MACU consistently improves over strong single-agent baselines by $3.4-25.5\%$ on desktop (OSWorld) and web navigation (Online-Mind2Web, WebTailBench, Odysseys) benchmarks, exhibits more favorable test-time scaling, and solves complex long-horizon tasks where single-agent CUAs get stuck. On Odysseys, a long-horizon web navigation benchmark, MACU improves average task completion wall-clock time by ${\sim} 1.5 \times$, demonstrating its efficacy in speeding up traditionally slow CUA pipelines. Our findings highlight that multi-agent coordination is a promising axis for scaling computer use agents to work productively for longer and more effectively. We release all code and interactive visualizations at this https URL.
>
---
#### [new 203] TriAlign: Towards Universal Truth Consistency in Personalized LLM Alignment
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于个性化大模型对齐任务，旨在解决不同社会群体间普遍真理不一致的问题。提出TriAlign框架，兼顾真理准确性、跨组一致性与个性化。**

- **链接: [https://arxiv.org/pdf/2606.01755](https://arxiv.org/pdf/2606.01755)**

> **作者:** Thi-Nhung Nguyen; Linhao Luo; Rollin Omari; Junae Kim; Thuy-Trang Vu; Dinh Phung
>
> **摘要:** Personalized large language models adapt responses to users' preferences and social attributes, but can introduce substantial universal truth inconsistencies across social groups, where some groups systematically receive less accurate responses on objective tasks. Existing alignment methods either ignore personalization or mainly focus on subjective preference alignment, largely overlooking fairness and consistency in universal truths. To address this gap, we study Truth-Invariant Alignment (TIA), an alignment problem for personalized LLMs that aims to ensure universal truths remain consistent across social groups while preserving personalization. We propose TriAlign, the first offline multi-agent reinforcement learning (MARL) framework for TIA, where each social group is modeled as an agent interacting. TriAlign jointly optimizes universal truth accuracy, cross-group truth consistency, and personalization through a fairness-aware objective and an explicit inconsistency penalty. Experiments across diverse benchmarks demonstrate that TriAlign achieves a stronger balance among these three objectives than strong baselines, reducing universal truth disparities across social groups while improving both objective task performance and personalization quality.
>
---
#### [new 204] Trust Functions: Near-Lossless Weak-to-Strong Generalization by Learning When to Trust the Weak Teacher
- **分类: cs.LG; cs.CL**

- **简介: 该论文研究弱教师到强学生的泛化问题，通过信任函数筛选可靠弱标签，提升学生性能，实现近无损的模型提升。**

- **链接: [https://arxiv.org/pdf/2606.01000](https://arxiv.org/pdf/2606.01000)**

> **作者:** Arda Uzunoglu; Alvin Zhang; Daniel Khashabi
>
> **备注:** ICML 2026
>
> **摘要:** Weak-to-strong generalization studies how to improve a strong student using supervision from a weaker teacher when reliable labels are scarce. We view this primarily as a data selection problem, where the key challenge is to identify which weak labels are reliable enough to serve as a training signal. To address this, we introduce trust functions that assign each weak label a scalar trust score and use these scores to filter weak supervision. Across several domains, including world knowledge, quantitative reasoning, and strategy games, trust filtering yields students that match and sometimes surpass ground-truth supervision, achieving near-lossless weak-to-strong generalization. Moreover, trust functions enable an iterative weak-to-strong chain that compounds gains by training a student and reusing it as the next teacher, amplifying the gains. There are several mechanisms to which advantage of trust functions can be attributed.
>
---
#### [new 205] Forget Attention: Importance-Aware Attention Is All You Need
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于语言模型任务，解决混合注意力机制的效率与优先级问题。提出SISA，在注意力分数中融合SSM重要性信号，实现更高效准确的序列建模。**

- **链接: [https://arxiv.org/pdf/2606.02332](https://arxiv.org/pdf/2606.02332)**

> **作者:** Soohyeong Shin; Yeongwook Yang
>
> **备注:** 20 pages, 6 figures, 25 tables
>
> **摘要:** Combining attention's global retrieval with the sequential importance signal of state space models (SSMs) is the open challenge of hybrid language modeling. Transformers see everywhere but cannot prioritize; SSMs know what matters but cannot revisit. Existing hybrids -- Jamba (block level) and Hymba (head level) -- place the two in separate compartments, so neither informs the other during the attention computation itself. We propose SISA (SSM-Informed Softmax Attention), which adds an SSM-derived importance term directly inside the attention score and realizes the full operation as a single SDPA call on augmented query/key vectors -- no recurrent state, no custom kernel. At 152M / 5B tokens, SISA reaches LAMBADA-greedy 17.3% (vs. Transformer 13.9 and Mamba-3 15.5) and attains NIAH 100% from step 1K, 7x faster than Transformer's retrieval convergence; at 369M, Mamba-3 leads LAMBADA while SISA preserves perfect NIAH and stock-SDPA execution. SISA thus defines a third design axis for SSM-attention hybrids -- score-level fusion -- beyond the block-level and head-level paradigms that have dominated the field.
>
---
#### [new 206] ODTQA-FoRe: An Open-Domain Tabular Question Answering Dataset for Future Data Forecasting and Reasoning
- **分类: cs.IR; cs.AI; cs.CL; cs.LG; cs.MA**

- **简介: 该论文提出ODTQA-FoRe任务，解决开放域表格问答中未来数据预测与推理问题。构建了基于房地产数据的首个相关数据集，并设计TimeFore框架提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2606.02433](https://arxiv.org/pdf/2606.02433)**

> **作者:** Zhensheng Wang; Xiaole Liu; Wenmian Yang; Kun Zhou; Yiquan Zhang; Weijia Jia
>
> **备注:** This paper has been accepted by Findings of ACL 2026
>
> **摘要:** The rapid development of LLMs has significantly advanced tabular question answering, but most systems cannot perform future-oriented numerical prediction. To address this gap, we introduce a novel task, Open-Domain Tabular Question Answering for Future Data Forecasting and Reasoning, and propose the first dataset to cover time-series forecasting and forecast-based reasoning scenarios using real estate data. This task poses challenges in retrieving precise historical data, overcoming the forecasting limitations of LLMs, and standardizing responses for diverse queries. To solve the above challenges, we propose TimeFore, an LLM agent-based framework that decomposes the problem into three collaborative roles: a Retriever autonomously generates SQL to fetch data, a Forecaster invokes external time-series models for higher accuracy, and an Analyzer synthesizes the results to construct a precise and consistent final answer. Extensive experiments demonstrate the effectiveness of our TimeFore.
>
---
#### [new 207] Cross-Generational Transfer of Adversarial Attacks Reveals Non-Monotonic Safety Alignment in LLMs
- **分类: cs.CR; cs.CL; cs.ET; cs.LG; cs.NE**

- **简介: 该论文研究LLM安全对齐问题，通过跨代攻击测试揭示模型安全性非单调变化。任务为安全评估，解决模型安全性随代际变化的不确定性问题，工作包括实验分析与攻击迁移测试。**

- **链接: [https://arxiv.org/pdf/2606.00813](https://arxiv.org/pdf/2606.00813)**

> **作者:** Subhadip Mitra
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Safety alignment in LLMs does not improve monotonically across model generations. Studying four generations of Google's Gemma family (7B-31B) with quality-diversity evolution (MAP-Elites) as an automated red-teaming probe, we find that Gemma 3 (12B) exhibits 68.7% +/- 5.7% attack success rate (ASR; mean +/- std, 3 seeds), significantly higher than its predecessor Gemma 2 (45.5% +/- 7.2%; p = 0.030, paired bootstrap) and its successor Gemma 4 (33.9% +/- 1.8%). Replaying evolved attack archives across generations reveals that attacks from other generations transfer to Gemma 3 at 44-46% but only 14-18% to Gemma 4, indicating that Gemma 4's safety gains generalize beyond the attack distributions evolved against earlier generations. Under our 8B judge, copyright and cybercrime vulnerabilities register at near-100% across all generations, though a second-judge audit (Section 6) suggests the copyright result is sensitive to judge choice. Misinformation ASR jumps from 29% to 99% between Gemma 2 and Gemma 3 and remains elevated at 77% in Gemma 4, indicating the regression was not fully addressed. These patterns are invisible to static benchmarks and emerge only through adaptive, longitudinal probing. All experiments use 3 random seeds with a unified self-hosted judge; code and artifacts are available at this https URL.
>
---
#### [new 208] BenchEvolver: Frontier Task Synthesis via Solution-Centric Evolution
- **分类: cs.SE; cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出BenchEvolver，解决基准数据集饱和问题，通过演化参考解生成更难任务，提升模型评估与训练效果。属于代码生成任务。**

- **链接: [https://arxiv.org/pdf/2606.01286](https://arxiv.org/pdf/2606.01286)**

> **作者:** Yangzhen Wu; Aaron J. Li; Wenjie Ma; Li Cao; Ziheng Zhou; Mert Cemri; Shu Liu; Yuran Xiu; Chenxiao Yan; Haikun Zhao; Bin Yu; Ion Stoica; Dawn Song
>
> **摘要:** The rapid progress of frontier large language models has led to widespread benchmark saturation, limiting the ability of existing datasets to differentiate model capabilities or provide useful training signal. For instance, on LiveCodeBench, frontier models achieve over 99% Pass@1 on easy splits and exceed 90% Pass@1 on average across difficulty levels. Constructing new, challenging datasets typically requires substantial human effort, creating a bottleneck for progress. We introduce BenchEvolver, a solution-centric evolutionary framework that automatically transforms existing coding problems into harder variants. Rather than generating problems from scratch, BenchEvolver evolves reference solutions through structured transformations and derives corresponding statements and tests from the evolved solutions. This design grounds generation in executable semantics, enabling scalable construction of high-quality, diverse, and difficult tasks with verifiable correctness. Applying BenchEvolver to LiveCodeBench and SciCode, we obtain evolved tasks that are substantially harder while maintaining validity, reference correctness, and diversity. We further curate LiveCodeBench-Plus, a 91-problem benchmark combining evolved and difficult original LCB-v6 tasks, where frontier-model Pass@1 ranges from 27.5% to 62.6%, restoring clear discrimination among strong coding models. Importantly, evolved tasks remain challenging even for the model that generates them, enabling self-improvement. We further show that RL on evolved LCB tasks improves held-out coding performance: for gpt-oss-20b, seed+evolved training achieves +8.7 and +8.3 Pass@1 gains on LCB v6 Hard and LCB-Pro Easy, exceeding seed-only gains by 70.7% and 34.8%, respectively. Our results show that BenchEvolver can convert saturated benchmarks into frontier-level evaluation suites and reusable training signal.
>
---
#### [new 209] Local Diagnostics of Continuous Normalizing Flow for Out-of-Distribution Detection
- **分类: eess.AS; cs.CL; cs.SD**

- **简介: 该论文属于异常检测任务，解决高维数据中分布外样本的检测问题。通过连续归一化流构建子流框架，提出几何诊断信号以提升误读检测效果。**

- **链接: [https://arxiv.org/pdf/2606.00684](https://arxiv.org/pdf/2606.00684)**

> **作者:** Xinwei Cao; Mengxuan Lu; Torbjørn Svendsen; Giampiero Salvi
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** We address the problem of out-of-distribution (OOD) detection for target observations embedded in a subspace of the high dimensional data space. Using continuous normalizing flows (CNFs), we propose a Lagrangian sub-flow (LSF) framework designed to isolate and estimate the density for the relevant components in the representation and using the remaining components as context. Through experimentation with models for speech synthesis, we show that CNFs, similarly to other deep generative models (DGMs), are susceptible to the "likelihood paradox", where high likelihood is erroneously assigned to OOD samples. This is attributed to the inductive bias of DGMs that prioritize low-level structural details over high-level semantic coherence. To mitigate this phenomenon, we propose a number of geometric diagnostic signals based on the velocity field over the sub-flow trajectory. Based on these signals, we design metrics for the challenging task of zero-shot phoneme-level mispronunciation detection. Finally, we demonstrate the superiority of these metrics compared to likelihood-based methods on a real-world mispronunciation detection benchmark.
>
---
#### [new 210] Grokers: Bottom-Up Inductive Comprehension and Write-Time Intelligence over Typed Knowledge Graphs
- **分类: cs.AI; cs.CL; cs.DB; cs.IR**

- **简介: 该论文提出Groker架构，用于知识图谱的结构化理解，通过自底向上的归纳遍历解决查询效率问题，实现零额外成本的智能写入。**

- **链接: [https://arxiv.org/pdf/2606.00050](https://arxiv.org/pdf/2606.00050)**

> **作者:** Gregory Magarshak
>
> **备注:** 6 pages; second in a series with the Magarshak Machine / SPACER paper and the Context paper
>
> **摘要:** We present Grokers, an architecture for building persistent, structured comprehension of typed knowledge graphs through bottom-up inductive traversal of dependency subgraphs. Unlike retrieval-augmented generation (RAG), which pays full comprehension cost at every query, Grokers pushes intelligence to write time: autonomous Groker agents analyze nodes in a typed stream graph, extract structured attributes via governed language model (LM) calls, and inductively compose that understanding upward through dependency relations, writing enriched typed attributes that serve all future queries at zero additional LM cost. We prove three formal properties: (1) the Byte-Identity Theorem, establishing that context blocks assembled from a transactionally-maintained denormalization index are byte-identical across LM turns between semantic changes, enabling KV-cache hit rates approaching 100%; (2) the Accumulation Monotonicity Theorem, establishing that the fraction of interactions resolved without LM calls is non-decreasing in the number of completed interactions under a governed wisdom library growth protocol; and (3) the Dual-Traversal Ordering Theorem, establishing that top-down generation and bottom-up comprehension are the unique correct traversal orderings for their respective tasks over a dependency DAG, and that their composition closes into a complete generation-comprehension cycle. We further present a deterministic alternative to embedding-based semantic search, with a synonym caching protocol whose LM fallback rate converges to zero for finite-vocabulary domains. A reference implementation is provided in the open-source Qbix / Safebox / Safebots stack.
>
---
#### [new 211] HMPO: Hybrid Median-length Policy Optimization for Chain-of-Thought Compression
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出HMPO方法，解决链式思维压缩问题，通过单阶段强化学习有效减少推理长度，提升效率并保持准确率。**

- **链接: [https://arxiv.org/pdf/2606.01934](https://arxiv.org/pdf/2606.01934)**

> **作者:** Minghui Zheng; Hongxu Chen; Huimin Ren; Hongsheng Xin; Xiaoyang Qu; Ze Wang; Shuling Yang; Ziyu Peng; Kaike Zhang; Pan Zhou; Kun Zhan
>
> **摘要:** Large language models achieve remarkable performance via extended chain-of-thought (CoT) reasoning, yet this lengthy process incurs substantial inference overhead. Existing CoT compression methods struggle with inflexible manual length budgets, computationally expensive multi-stage training pipelines, and fragile scalability restricted to small models. We propose HMPO (Hybrid Median-length Policy Optimization), a cost-effective, single-stage reinforcement learning framework. HMPO efficiently compresses CoT via three synergistic components: an adaptive median-based budget derived from successful rollouts to eliminate manual tuning, a cosine-decay token reward for smooth length penalization, and a multiplicative reward formulation that substantially mitigates trivial reward hacking by strictly prioritizing answer correctness. Trained exclusively on mathematical data, HMPO generalizes seamlessly across math, code, science, and instruction-following tasks. Extensive experiments scaling from 9B to 122B parameters across dense and Mixture-of-Experts (MoE) architectures demonstrate that HMPO achieves 19%--46% token compression with negligible accuracy degradation, all while drastically reducing training costs compared to existing multi-stage baselines.
>
---
#### [new 212] An Algebraic View of the Expressivity of Recurrent Language Models
- **分类: cs.FL; cs.CL; cs.LG**

- **简介: 该论文研究递归语言模型的表达能力，解决其是否为图灵完备或仅等价于正则语言的争议。通过代数方法分析不同算术模型下的表达性，揭示模型差异根源。**

- **链接: [https://arxiv.org/pdf/2606.01765](https://arxiv.org/pdf/2606.01765)**

> **作者:** Franz Nowak; Ryan Cotterell; Reda Boumasmoud
>
> **备注:** 28 pages, 2 figures, to be published at ICML 2026
>
> **摘要:** What formal languages can a recurrent neural language model recognize? Formal results in the literature conflict: some authors report Turing-completeness, while others show equivalence to regular languages. The reason for this discrepancy is that the underlying arithmetic model differs. The paper develops a unified algebraic account of the expressivity of recurrent neural networks, starting with a formal account of various arithmetic models. This account reduces expressivity to an algebraic question, e.g., whether a network's syntactic monoid divides a certain wreath product. As a case study, the paper revisits diagonal state-space models: the same architecture cannot implement an even-modulus counter once floating-point recurrences are enforced, yet realizes every even-modulus counter under unsigned-integer quantization.
>
---
#### [new 213] ContinuousBench: Can Differentially Private Synthetic Text Improve Capabilities?
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文属于隐私保护与文本生成任务，旨在评估差分隐私合成文本是否能保留原始数据的知识和能力。工作包括构建持续更新的基准ContinuousBench，验证DP合成文本的有效性。**

- **链接: [https://arxiv.org/pdf/2606.01849](https://arxiv.org/pdf/2606.01849)**

> **作者:** Peihan Liu; Lucas Rosenblatt; Weiwei Kong; Natalia Ponomareva; Gautam Kamath; Rachel Cummings; Roxana Geambasu; Yu Gan; Lillian Tsai; Alex Bie
>
> **备注:** Datasets: this https URL ; Eval Harness: this https URL ; Blog post: this https URL
>
> **摘要:** Differentially private (DP) text synthesis promises to unlock sensitive corpora for model training, but it remains unclear whether DP synthetic data transmits genuinely new knowledge and capabilities present only in those corpora. This is because existing evaluations rely on tasks that are nearly solvable without training, so strong benchmark performance does not establish that DP synthesis can substitute original data access. Thus, we introduce ContinuousBench, a continuously and automatically-regenerated benchmark that measures capability gain from DP synthetic text. Each quarter, a new release pairs a never-before-seen training corpus with a derived QA set, constructed to be: (1) unsolvable sans-corpus; and (2) learnable under DP, as the tested knowledge is supported by hundreds of independent records. Researchers produce DP synthetic data from the training corpus and run our standardized training and evaluation harness on their synthetic data to measure gains. We instantiate two tracks: Geminon, a procedurally-generated dataset about fictional creatures; and News, a stream of newly crawled public news articles. Although standard benchmarks are nearly saturated, on ContinuousBench we find that non-private synthesis transfers substantial knowledge from the original corpus, while state-of-the-art DP synthesis methods generally fail to do so, even at $\varepsilon=100$.
>
---
#### [new 214] BraveGuard: From Open-World Threats to Safer Computer-Use Agents
- **分类: cs.CR; cs.CL**

- **简介: 该论文提出BraveGuard，用于提升计算机使用代理的安全性。任务是检测多步骤执行中的安全风险，解决传统方法难以发现的问题。工作包括构建防御框架，训练守护模型，并验证其有效性。**

- **链接: [https://arxiv.org/pdf/2606.01166](https://arxiv.org/pdf/2606.01166)**

> **作者:** Yunhao Feng; Yifan Ding; Xiaohu Du; Ming Wen; Xinhao Deng; Yanming Guo; Yuxiang Xie; Baihui Zheng; Yingshui Tan; Yige Li; Yutao Wu; Yixu Wang; Kerui Cao; Wenke Huang; Xingjun Ma; Yu-Gang Jiang
>
> **摘要:** Computer-use agents extend language models from text generation to sustained interaction with files, terminals, browsers, and external tools. This shift creates safety risks that are difficult to detect from isolated prompts or final responses, because harm often emerges only through multi-step execution traces whose individual actions appear locally benign. We introduce BraveGuard, a self-evolving defense framework for training guard models from open-world threat signals and realistic agent trajectories. BraveGuard mines recent research sources to identify emerging risks and attack patterns, instantiates them as executable computer-use tasks, collects agent rollouts, and derives trajectory-level supervision for guard model training. As new threats and validation failures appear, the pipeline can be repeated, yielding an adaptive defense loop rather than a static, benchmark-driven training process. We instantiate BraveGuard by training multiple guard backbones, including Qwen3-Guard and Llama-Guard variants, and evaluate the resulting guards on trajectory-level agent-safety benchmarks. BraveGuard consistently improves safety detection across computer-use trajectories. On AgentHazard, it substantially improves detection accuracy over off-the-shelf guard models, with accuracy increasing from 38.79% to 82.38% under the averaged guard-model setting. These results show that guard supervision grounded in open-world threat discovery and realistic agent execution can improve safety monitoring beyond fixed taxonomies and synthetic prompt-level data. BraveGuard offers a scalable path toward adaptive defenses for computer-use agents facing evolving real-world risks.
>
---
#### [new 215] Defenses & Enablers For Skill Injection Attacks on Terminal Based Agents
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文研究终端代理的技能注入攻击防御，解决安全威胁问题。通过动态与静态守护者降低攻击成功率，验证其有效性。**

- **链接: [https://arxiv.org/pdf/2606.01567](https://arxiv.org/pdf/2606.01567)**

> **作者:** Yoshinari Fujinuma; Varun Gangal; Traian Rebedea; Makesh Narasimhan Sreedhar; Prasoon Varshney; Rebecca Qian; Anand Kannappan
>
> **备注:** First version, small updates and clarifications likely in v2
>
> **摘要:** Large language model (LLM) agents increasingly rely on reusable skills i.e. documents describing task-specific procedures. However, this introduces a new attack surface for agents to manage. We study two complementary directions for this threat. First, we evaluate guardian-based defenses: an intermediary LLM agent that acts as a mediator for skill file access (dynamic guardian) or pre-rewrites these files at build time (static guardian). Across three LLM agent families, our guardians cut attack success rate (ASR) by well over half while preserving task utility. Second, we stress test them through attack reframing using four attacks that preserve the malicious instruction but change the phrasing. For non-guardian setup, the reframing pushes the ASR up to 81.4\%, but the dynamic guardian brings it down to 18.6\%, showing that real-time mediation is a robust defense.
>
---
#### [new 216] Cross-modal linkage risk in clinical vision-language models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究临床视觉-语言模型中的跨模态关联风险，旨在解决图像与报告被重新链接的隐私问题。通过分析模型在不同数据集上的表现，提出差分隐私优化方法以降低风险，同时保持模型性能。**

- **链接: [https://arxiv.org/pdf/2606.02276](https://arxiv.org/pdf/2606.02276)**

> **作者:** Soroosh Tayebi Arasteh; Mahshad Lotfinia; Sven Nebelung; Daniel Truhn
>
> **摘要:** Vision-language models (VLMs) trained on paired chest radiographs and radiology reports learn a shared embedding space that can preserve instance-level image-report correspondence. This poses a privacy risk in settings where radiographs and reports are deliberately kept separate after acquisition, such as image-only data sharing or access-controlled reports, because a de-identified image may be re-linked to its original narrative report through cosine similarity alone. We formalized this as image-to-report retrieval and used public paired cohorts, in which the true pairing is known by design, as ground-truth benchmarks to audit the risk rather than as the privacy scenario. Evaluating VLMs of increasing clinical specialization on 406,241 paired examples from 126,804 patients across MIMIC-CXR (43,793 held-out pairs) and external CheXpert Plus (29,296 pairs), we found that re-linkage rose systematically with specialization: the strongest VLM retrieved the correct report at 15 times chance at a candidate pool of N = 100, 50 times chance at N = 10,000, and well above chance at full-database scale. The signal persisted under pathology-matched hard negatives that removed disease-label shortcuts, indicating correspondence beyond broad diagnostic categories. To reduce it without retraining, we froze both encoders and applied differentially private optimization only to the projection heads defining the alignment layer (epsilon = 0.34, delta = 6x10-6). This reduced Recall@1 by 61.8% at N = 10,000 on MIMIC-CXR and transferred to CheXpert Plus without retraining, while image-side utility was largely preserved: macro AUROC for linear-probe classification across 14 labels shifted only from 79.63% to 79.43%. Targeted DP finetuning of the shared alignment layer can substantially reduce cross-modal re-linkage without materially degrading the image representations that make these models clinically useful.
>
---
#### [new 217] Jailbreaking Multimodal Large Language Models using Multi-Clip Video
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于安全评估任务，旨在解决MLLMs在视频输入下的漏洞问题。通过构建MCV数据集，分析视频多样性对模型安全性的影响，并提出防御策略。**

- **链接: [https://arxiv.org/pdf/2606.02111](https://arxiv.org/pdf/2606.02111)**

> **作者:** Choongwon Kang; Seungjong Sun; Hyunmin Jun; Jang Hyun Kim
>
> **备注:** 27 pages, 20 figures, Accepted to the Main Conference of ACL 2026
>
> **摘要:** As multimodal large language models (MLLMs) have advanced to process video inputs, concerns have emerged about their potential for malicious misuse. Prior jailbreak studies have shown that safety alignment in MLLMs can be bypassed through visual inputs, yet it remains unclear which properties of video inputs induce this vulnerability. To address this gap, we introduce Multi-Clip Video (MCV) SafetyBench, a dataset of 2,920 videos designed to evaluate how the diversity of video inputs affects the vulnerability of MLLMs. Each video consists of multiple short clips depicting diverse contexts related to a harmful query. Experiments on eight representative video MLLMs show that attack success consistently increases with the number of clips. Our results further indicate that the video modality is (1) more vulnerable than the image modality, (2) more vulnerable to dynamic videos than to static videos, and (3) more vulnerable when videos contain more diverse contexts. Building on these findings, we propose a defense strategy that leverages the relative robustness of the image modality.
>
---
#### [new 218] Same Payload, Different Channel: Measuring Trust Asymmetry in Tool-Using Language Models
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文属于安全评估任务，研究工具使用语言模型对不同渠道恶意内容的响应差异，提出SAS指标衡量信任不对称性，揭示模型对工具元数据与输出的处理差异。**

- **链接: [https://arxiv.org/pdf/2606.00566](https://arxiv.org/pdf/2606.00566)**

> **作者:** Mohammed Sameer Syed; Rozhin Yasaei
>
> **备注:** 13 pages, 1 figure. Submitted to EMNLP 2026
>
> **摘要:** As language models take on agentic roles that span calling external APIs, reading tool outputs, and acting on instructions embedded in third-party content, their attack surface expands well beyond what users type. Whether a model treats a malicious instruction the same way regardless of where it arrives has not been systematically studied. We introduce the Safety Asymmetry Score (SAS), which measures how much a model's susceptibility to adversarial content shifts depending on whether that content arrives in the user message, tool metadata, or tool output, using matched payload pairs that keep the malicious text identical and vary only the context of delivery. Evaluated across 6 production LLMs and three attack families, we find a consistent and informative asymmetry: agent-native models are substantially more vulnerable when adversarial content arrives via tool descriptions than via user messages, while general-purpose models show the reverse. This asymmetry further inverts when the same content is delivered through tool outputs rather than descriptions, suggesting models implicitly treat tool metadata as trusted instructions and tool results as ordinary data. A mechanistic study on Llama 3.3 70B reveals that the safety-relevant representation is causally present at mid-to-late network depths but non-linearly encoded, explaining why linear probes fail to detect it. These findings expose a systematic, channel-dependent blind spot in how current tool-using models handle adversarial content.
>
---
#### [new 219] TECCI: Tricky Edits of Collected and Curated Images
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出TECCI基准，用于评估文本引导的图像编辑任务。解决现有方法在指令遵循、最小修改和视觉质量上的不足，通过大量挑战性编辑指令测试模型性能。**

- **链接: [https://arxiv.org/pdf/2606.01213](https://arxiv.org/pdf/2606.01213)**

> **作者:** Aishwarya Agrawal; Roy Hirsch; Yasumasa Onoe; Sherry Ben; Jason Baldridge
>
> **摘要:** Despite tremendous recent progress, current text-guided image editing methods still struggle with many aspects of editing involving instruction following, minimally editing the source image, and ensuring high visual quality. These problems are especially apparent when the requested edit is challenging, such as those that involve position, motion, viewpoint, scale and creative edits. To systematically test generative image editors, we propose a novel image editing benchmark -- TECCI: Tricky Edits of Collected and Curated Images. TECCI consists of a completely new set of images we are releasing. The images in TECCI span 7 image categories. The images and these categories were curated intentionally to target weaknesses of existing methods. The edit instructions in TECCI are automatically generated by Gemini, covering 5 edit types per source image. We also curated a set of 530 images for which we created challenging manually written edit instructions. Overall, TECCI contains 7550 pairs of images and edit instructions. We conduct human evaluations of five leading image editing models on TECCI. Humans judge outputs along three dimensions: 1) instruction following, 2) minimality of the edits, and 3) visual quality. To scale-up the evaluation, we also build an auto-rater using Gemini that achieves 74.7% accuracy in matching human evaluations. Our evaluations reveal that: 1) none of the models exceed a 22% overall success rate, demonstrating the challenging nature of TECCI, 2) Nano Banana Pro is the best performing model overall, 3) models perform significantly better at instruction following compared to minimal edits and visual quality, 4) models struggle with editing architecture and nature images which require strong understanding of spatial layout and intricate visual details. 5) reasoning and creative edits are the most difficult, whereas color and appearance edits are the easiest.
>
---
#### [new 220] Beyond Text and Tables: Vision-Language Model Integration in ComProScanner for Extracting Materials Data from Scientific Figures with High Accuracy
- **分类: cs.IR; cond-mat.mtrl-sci; cs.AI; cs.CL**

- **简介: 该论文属于材料数据提取任务，解决科学图表中定量数据难以自动提取的问题。通过集成视觉语言模型，提升ComProScanner从图文中准确提取成分-性能数据的能力。**

- **链接: [https://arxiv.org/pdf/2606.00065](https://arxiv.org/pdf/2606.00065)**

> **作者:** Aritra Roy; Enrico Grisan; Chiara Gattinoni; John Buckeridge
>
> **备注:** 18 pages, 3 figures
>
> **摘要:** Automated extraction of materials composition-property data from scientific literature has advanced considerably with the development of large language model-based pipelines; however, existing frameworks remain limited to textual and tabular content, overlooking the substantial proportion of quantitative property data reported exclusively in scientific figures. Here, we extend ComProScanner, a fully end-to-end multi-agent framework for automated composition-property database construction, with a native vision-language model (VLM) based figure extraction capability. The extension introduces a FigureExtractor utility for caption-keyword-based figure filtering across all supported publishers, and a GraphExtractorTool agent that passes extracted figures to a configurable VLM to recover composition-property pairs from scientific charts and plots. Four VLMs are selected for evaluation on the basis of the LMArena Diagram leaderboard with an input cost criterion of less than \$1.50 per million tokens. Benchmarking on 50 piezoelectric ceramic articles from the established $d_{33}$ test corpus demonstrates that Gemini-3-Flash-Preview achieves the highest performance with a composition accuracy of 0.97 and a normalised F1 score of 0.97, whilst remaining the most cost-effective model among the four evaluated. We additionally introduce a range-based value error threshold parameter into the evaluation framework, providing a more physically meaningful assessment of numeric property values extracted from figures than exact value matching. These contributions establish VLM-integrated ComProScanner as the first materials-specific, fully automated, multimodal literature mining platform capable of extracting structured composition-property data from text, tables, and figures within a single unified pipeline.
>
---
#### [new 221] Don't Ask the LLM to Track Freshness: A Deterministic Recipe for Memory Conflict Resolution
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于记忆冲突解决任务，旨在提升LLM在动态事实中的准确率。通过确定性聚合方法替代LLM判断，显著提升性能。**

- **链接: [https://arxiv.org/pdf/2606.01435](https://arxiv.org/pdf/2606.01435)**

> **作者:** Vikas Reddy; Sumanth Challaram
>
> **摘要:** LLM-based memory systems increasingly maintain facts that evolve over time, where a recurring failure is conflict resolution: when a fact has multiple contradictory values, which should the agent return? MemoryAgentBench (MAB; Hu et al., 2026) makes this explicit in its FactConsolidation task: facts are numbered, the counterfactual has the higher serial, and agents are told newer facts have larger serials. Yet every published system underperforms: HippoRAG-v2 reaches 54% on single-hop (FC-SH), BM25 48%, Mem0 18%, and the temporal KG Zep/Graphiti just 7%. Multi-hop is near-unsolved (at most 7% across 22 systems). We argue the bottleneck is the assembly step: baselines leave conflict resolution to LLM-mediated retrieval or generation rather than version-aware aggregation. A matched-setup comparison (same backbone, retrieval, chunking, TOP_K) shows that replacing the LLM-judgment answer pipeline with candidate-extraction plus Python max(serial) yields +10.8 points on FC-SH (gpt-4o-mini), widening from +8 at 6K to +21 at 262K. This is a whole-pipeline effect (resolver, prompt, format, and temperature vary jointly); isolating the resolver is future work. The recipe reaches 78.0% on FC-SH (gpt-4o-mini), 94.8% (gpt-4o), and 30.2% on FC-MH (gpt-4o-mini, rising to 51.5% with gpt-4o) via a per-hop deterministic extension of Self-Ask. At matched-262K, it beats HippoRAG-v2 by +28 points and the best published FC-MH result by +20. The implication is corrective for the subfield: the bottleneck on conflict resolution is assembly (post-retrieval aggregation), not storage. A LongMemEval knowledge-update check shows the mechanism ports from max(serial) to max(timestamp) but only ties LLM judgment (57.8% vs 64.4%, n=45): deterministic aggregation is the right primitive for current-value conflicts and must be composed with question-type-aware handling for broader memory QA.
>
---
#### [new 222] Truthful AI Advisors: A Pre-Specified Benchmark for Large Language Model Honesty Under Preference Misalignment
- **分类: cs.LG; cs.CL; cs.GT**

- **简介: 该论文属于AI对齐任务，研究在利益冲突下大语言模型的诚实性。通过设计实验评估模型在不同偏置下的信息传递能力，发现模型过度透露信息，偏离理论最优解。**

- **链接: [https://arxiv.org/pdf/2606.01456](https://arxiv.org/pdf/2606.01456)**

> **作者:** Hamidreza Hasani Balyani; Seyed Pouyan Mousavi Davoudi; Alireza Amiri-Margavi; Amin Gholami Davodi; Arshia Gharagozlou
>
> **备注:** 19 pages. Code and data: this https URL
>
> **摘要:** Large language models are increasingly deployed as advisors whose objective is not aligned with the user's: recommenders optimize for engagement, sales assistants for purchases, negotiation agents for concessions. Whether such advisors stay truthful when honesty conflicts with their own payoff is a core alignment-evaluation question. We turn the canonical Crawford-Sobel cheap-talk model into a pre-specified benchmark for LLM honesty under preference misalignment. Cheap-talk theory predicts neither full revelation nor silence but coarse monotone partitions, with fewer informative intervals as preference conflict grows. A sender observes a state omega in [0,1], wants the receiver's action near omega+b, and sends one costless message to a receiver whose ideal action is omega. The design uses 5 bias levels, 3 prompt frames, a fixed low-temperature setting, and 200 states per cell: 12,000 sender calls. For the positive-bias grid b in {0.01,0.04,0.08,0.12} the exact most-informative partition sizes are 7,4,3,2, with oracle normalized mutual information 0.5294, 0.3268, 0.2205, 0.1829. Running the full design on four instruction-tuned models (GPT-4o, Claude Sonnet 4.5, Gemini 2.5 Flash-Lite, Llama-3.3-70B), we find all four over-reveal relative to the most-informative equilibrium by 1.8 to 4.2x: normalized mutual information stays at 0.78-0.94 where the oracle prescribes 0.18-0.53. Informativeness declines with bias as predicted but never approaches the strategic optimum; rather than coarse partitions, models show near-full revelation with a constant upward offset tracking their bias (linear exaggeration). Payoff-maximizing versus honesty framing has negligible effect. A decoder ablation shows the finding is recoverable only when the receiver reads the sender's stated number: an embedding-only decoder mis-reads the same data as near-babbling.
>
---
#### [new 223] AgentRedBench: Dynamic Redteaming and Integration-Aware Defense for LLM Agents over SaaS Integrations
- **分类: cs.CR; cs.AI; cs.CL; cs.ET**

- **简介: 该论文属于安全防护任务，解决LLM代理在SaaS集成中的间接提示注入威胁，通过构建动态红队基准和防御模型提升安全性。**

- **链接: [https://arxiv.org/pdf/2606.02240](https://arxiv.org/pdf/2606.02240)**

> **作者:** Hiskias Dingeto; Will Leeney
>
> **摘要:** Indirect prompt injection in tool-use agents is a concrete production threat: LLM agents read from integrations (third-party services such as Gmail, Salesforce, or Jira accessed through tool calls) whose response content the user neither writes nor controls. Existing benchmarks under-measure the threat: most cover only a handful of integrations with the same attack payload replayed across runs, and open-source guards are trained on chat-style data rather than tool-response content. We introduce AGENTREDBENCH, a dynamic LLM-driven redteaming benchmark of 215 subtle underspecified authorization (attacks at the boundary of what the user's request authorises) scenarios across 24 enterprise integrations in nine functional families and five attack types. Across an eight-model panel (Anthropic, OpenAI, Google), no-guard ASR (attack success rate) ranges from 32% (Claude Sonnet 4.6) to 81% (Gemini 3 Flash). To keep the scenario set out of training corpora and preserve headline ASR meaning over time, we release the codebase, integration schemas, and AGENTREDGUARD model openly; the canonical scenarios are evaluated through a maintainer-mediated channel with immutable versioning. We release AGENTREDGUARD alongside the benchmark: a guard trained on an integration-diverse corpus of adversarial tool-response content. AGENTREDGUARD cuts panel ASR from 69.9% to 2.4% at 0.37% false-positive rate, outperforming every open-source baseline with non-trivial detection (Llama Guard, PromptGuard 2, ProtectAI) on both axes. Cross-integration and cross-attack type holdouts both confirm the gain transfers beyond the training subset.
>
---
#### [new 224] AdaCodec: A Predictive Visual Code for Video MLLMs
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出AdaCodec，用于视频多模态大语言模型，解决视频内容重复导致的效率问题。通过预测性视觉编码，减少冗余信息传输，提升性能与速度。**

- **链接: [https://arxiv.org/pdf/2606.02569](https://arxiv.org/pdf/2606.02569)**

> **作者:** Haowen Hou; Zhen Huang; Zheming Liang; Qingyi Si; Chenglin Li; Shuai Dong; Kele Shao; Ruilin Li; Dianyi Wang; Nan Duan; Jiaqi Wang
>
> **备注:** 23 pages
>
> **摘要:** Video is temporally redundant: adjacent frames usually share most objects, background, and layout. Yet existing video multimodal large language models (video MLLMs) usually encode each sampled frame as an independent RGB image, causing visual tokens to repeat content already present in earlier frames. This suggests a more direct video interface: send a full reference frame only when the scene cannot be predicted well from prior context, and otherwise transmit a compact description of inter-frame changes. We call this interface a \emph{predictive visual code}, and instantiate it for video MLLMs as \textbf{AdaCodec}. AdaCodec spends full visual tokens on a reference frame only when its conditional predictive cost is high; otherwise, it encodes inter-frame changes, including motion and prediction residuals, as compact P-tokens. Across all eleven benchmarks, AdaCodec improves over the Qwen3-VL-8B per-frame RGB baseline at a matched visual-token budget. Even at $1/7$ the budget, AdaCodec with 32k tokens surpasses the 224k baseline on all long-video benchmarks; on five general-video benchmarks, it raises the average score while substantially cutting time-to-first-token from 9.26s to 1.62s.
>
---
#### [new 225] Sympatheia: Emotionally Adaptive Voice Assistant with Continuous Affect Conditioning
- **分类: cs.SD; cs.CL; cs.HC; cs.LG; eess.AS**

- **简介: 该论文提出Sympatheia，一个情感自适应语音助手，解决如何根据用户情绪生成恰当回应的问题。通过合成数据集和连续情绪控制信号，提升对话的情感适应性。**

- **链接: [https://arxiv.org/pdf/2606.00851](https://arxiv.org/pdf/2606.00851)**

> **作者:** Sukru Samet Dindar; Riki Shimizu; Xilin Jiang; Nima Mesgarani
>
> **摘要:** Empathetic spoken dialogue systems must infer a user's emotional state to respond appropriately, yet everyday speech often carries weak, neutral, or ambiguous affective cues. To address this, we introduce Sympatheia, a speech-to-speech dialogue framework conditioned on affect inferred from the user's speech and, when available, explicit affect specifications provided as a continuous valence--arousal (VA) control signal by a multimodal sensing module or user interface. To train our model, we construct Sympatheia-18k, an emotion-conditioned synthetic spoken dialogue corpus with 12 emotion anchors. This dataset includes an emotional split for learning affective speech behavior, and a neutral split that pairs emotionally neutral queries with multiple emotion-conditioned responses to isolate explicit emotion control in emotionally ambiguous cases. Empirical results show that Sympatheia outperforms speech conversational baselines in generating responses whose semantic content and spoken delivery are both emotionally appropriate. We further show that the same VA interface can integrate emotion estimates from diverse sensing modules, including facial expression, biosignals, and textual affect descriptions, improving response alignment when speech alone provides limited emotional evidence. These results suggest that continuous affect conditioning is an effective practical step for building emotionally adaptive voice assistants.
>
---
#### [new 226] Harness-1: Reinforcement Learning for Search Agents with State-Externalizing Harnesses
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出Harness-1，一种基于强化学习的搜索代理，解决搜索过程中状态管理与语义决策分离的问题。通过环境侧维护状态，提升检索效果。**

- **链接: [https://arxiv.org/pdf/2606.02373](https://arxiv.org/pdf/2606.02373)**

> **作者:** Pengcheng Jiang; Zhiyi Shi; Kelly Hong; Xueqiang Xu; Jiashuo Sun; Jimeng Sun; Hammad Bashir; Jiawei Han
>
> **摘要:** Search agents are often trained as policies over growing transcripts: the model must decide how to search while also remembering what it has seen, which evidence is useful, which constraints remain open, and which claims have actually been checked. We argue that this formulation puts too much routine state management inside the policy: reinforcement learning is forced to optimize both semantic search decisions and recoverable bookkeeping that the environment can maintain more reliably. We introduce Harness-1, a 20B search agent (retrieval subagent) trained with reinforcement learning inside a stateful search harness. The harness maintains environment-side working memory, including a candidate pool, an importance-tagged curated set, compact evidence links, verification records, compressed and deduplicated observations, and budget-aware context rendering. The policy retains the semantic decisions: what to search, which documents to keep or discard, what to verify, and when to stop. Across eight retrieval benchmarks spanning web, finance, patents, and multi-hop QA, Harness-1 achieves 0.730 average curated recall, outperforming the next strongest open search subagent by +11.4 points and remaining competitive with much larger frontier-model searchers. Its gains are especially strong on held-out transfer benchmarks, suggesting that reinforcement learning over explicit search state can produce retrieval behaviors that generalize beyond the training domains. Our code is available at this https URL.
>
---
#### [new 227] An Enigma of Artificial Reason: Investigating the Production-Evaluation Gap in Large Reasoning Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于人工智能推理任务，研究大模型在生成与评估推理过程中的能力差异。通过VAIR数据集，发现模型在评估推理时表现不佳，存在答案确认偏差。**

- **链接: [https://arxiv.org/pdf/2606.01462](https://arxiv.org/pdf/2606.01462)**

> **作者:** Mingzhong Sun; Teresa Yeo; Armando Solar-Lezama; Tan Zhi-Xuan
>
> **备注:** 10 pages, 8 figures, 2 tables (Appendix: 19 pages, 13 figures, 3 tables)
>
> **摘要:** Studies of human reasoning have shown that people are typically stronger at evaluating reasoning than producing it from scratch. In contrast, large reasoning models (LRMs) are trained to excel at producing long chains of reasoning to solve complex problems. How then do LRMs perform at evaluating reasons? We investigate this with the Valid-Answer-Invalid-Reasoning (VAIR) dataset: math problems and solutions with trivial reasoning flaws but valid answers, designed to isolate reasoning evaluation from the confound of reasoning production. Unlike humans, who we find are only 6% worse at grading than solving such problems, we find a substantial production-evaluation gap in LRMs: frontier models score as low as 48% when evaluating VAIR solutions, despite near-perfect solution production. Why this enigma? Through chain-of-thought (CoT) analysis, we find evidence of an answer confirmation bias: LRMs often produce then check for the correct answer instead of carefully verifying each step, fabricating rationalizations even when noticing anomalous reasoning. Linear probes corroborate this, showing that while LRM activations encode some representation of valid reasoning, they fail to robustly represent VAIR solutions as invalid. Causal patching of the final answer's representations causes LRM verdicts and activations to flip, demonstrating that answer validity is responsible for models' confirmation biases. These findings indicate an outstanding limitation in dominant approaches to reasoning training, which incentivize LRMs to produce and confirm reasoning towards correct answers, but not to robustly evaluate the underlying reasons.
>
---
#### [new 228] On the Limits of Token Reduction for Efficient Unified Vision Language Training
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文研究统一视觉语言模型的高效训练，针对计算效率问题，分析了令牌缩减的可行性与限制，提出任务特定加速方法，但发现联合训练中存在协同损失。**

- **链接: [https://arxiv.org/pdf/2606.01503](https://arxiv.org/pdf/2606.01503)**

> **作者:** Siyi Chen; Weiming Zhuang; Jingtao Li; Lingjuan Lv
>
> **摘要:** Unified vision-language models (VLMs) integrate visual understanding and visual generation within a single autoregressive backbone, but their joint training is computationally expensive and largely overlooked from an efficiency perspective. In this work, we study the feasibility and limits of token-reduction-based acceleration for unified VLM training. Through a systematic analysis of layerwise attention allocation, we uncover a fundamental asymmetry: visual understanding exhibits substantial late-layer visual redundancy, whereas visual generation maintains persistent dependence on image tokens across depth. Guided by this observation, we design task-specific accelerators that selectively reduce image-token computation for each objective. While these methods achieve significant efficiency gains in isolated settings, we observe a consistent synergy loss under unified training -- task-specific token dropping necessitates divergent parameter pathways and eliminates the mutual performance gains typically observed in joint optimization. Our findings suggest that efficient unified modeling requires preserving shared cross-task structures, highlighting the need for synergy-aware acceleration strategies. Project page: this https URL.
>
---
#### [new 229] MindGames Arena Generalization Track: In2AI Solution with Delayed Per-Step Reward Attribution
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于多智能体强化学习任务，解决动作奖励难以即时评估的问题。提出延迟步骤奖励归因方法，提升训练效率与效果。**

- **链接: [https://arxiv.org/pdf/2606.00017](https://arxiv.org/pdf/2606.00017)**

> **作者:** Aliaksei Korshuk; Alexander Buyantuev; Ilya Makarov
>
> **备注:** 18 pages, 2 figures, 9 tables. Technical report. First place in both Open and Efficient tracks of MindGames Arena Generalization Track at NeurIPS 2025
>
> **摘要:** Training language model agents for multi-agent strategic interaction presents a core difficulty: the quality of any action may depend on future events that never materialize, on moves that violate game rules, or on decisions made by other players. Standard reinforcement learning assumes that rewards can be assigned at each step, but this assumption fails in settings where outcomes are entangled across time and agents. We introduce delayed per-step reward attribution with eligibility gating, an episode lifecycle and postprocessing pipeline that computes rewards only at episode end, propagates them back to originating steps according to task-specific semantics, and excludes steps that lack valid dependent information from training. Together with asynchronous rollout generation via vLLM's continuous batching, curriculum-based opponent sampling, and multi-level stratified batch construction, this approach enables stable, sample-efficient RL training in multi-agent environments. We evaluate on the MindGames Arena benchmark at NeurIPS 2025, where a single 8-billion-parameter open-source model trained with our method matched or surpassed substantially larger proprietary systems, including GPT-5, in head-to-head play and took first place in both the Open (unrestricted) and Efficient (<=8B parameters) tracks.
>
---
#### [new 230] GuidaPA: Privacy-Preserving Chatbot for Public Administration via Federated Learning
- **分类: cs.AI; cs.CL; cs.DC; cs.LG**

- **简介: 该论文提出GuidaPA，一个基于联邦学习的隐私保护聊天机器人，解决公共部门数据隐私与共享问题，通过分布式训练提升对话质量。**

- **链接: [https://arxiv.org/pdf/2606.01386](https://arxiv.org/pdf/2606.01386)**

> **作者:** Daniel M. Jimenez-Gutierrez; Albenzio Cirillo; Raffaele Nicolussi; Alessio Beltrame; Andrea Vitaletti
>
> **备注:** Accepted to the 2nd International Conference on Federated Learning and Intelligent Computing Systems (FLICS2026)
>
> **摘要:** We present GuidaPA, a privacy-preserving chatbot for the Italian Public Administration (PA) trained via Federated Learning (FL) on documentation from two national PA platforms, SIGESON and SIDFORS. Our corpus includes approximately 8 pages of SIGESON manuals and 31 pages of SIDFORS manuals/FAQs; while this study uses public documentation as a safe proxy, the intended deployment extends to restricted internal sources (e.g., tickets, officer manuals, database extracts) that can not be centrally pooled due to regulatory and organizational constraints. GuidaPA integrates role-based access control, secure client-side preprocessing, explicit monitoring of non-IID effects, and parameter-efficient federated fine-tuning of large language models. Using QLoRA (4-bit) over 15 federated rounds with an 80/20 train-test split per client, we evaluate answer quality with ROUGE, BLEU-4, and METEOR. The best federated model achieves ROUGE-1/2/L of 61.10/55.77/59.44, BLEU-4 of 45.02, and METEOR of 63.94-close to private centralized fine-tuning while keeping data on-site. Compared to the general-purpose baseline, domain fine-tuning improves ROUGE-1 from 41.45 to 62.18 and BLEU-4 from 26.97 to 50.90. Overall, the results indicate that FL can deliver high-quality conversational AI for public services without centralized data sharing
>
---
#### [new 231] BAGEN: Are LLM Agents Budget-Aware?
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究预算感知的智能体，解决资源浪费问题。通过定义内部与外部预算，提出渐进区间估计方法，评估并改进代理的预算意识能力。**

- **链接: [https://arxiv.org/pdf/2606.00198](https://arxiv.org/pdf/2606.00198)**

> **作者:** Yuxiang Lin; Zihan Wang; Mengyang Liu; Yuxuan Shan; Longju Bai; Junyao Zhang; Xing Jin; Boshan Chen; Jinyan Su; Xingyao Wang; Jiaxin Pei; Manling Li
>
> **摘要:** While agents are increasingly spending more resources, today agent cost is mostly measured only after execution. A Budget-Aware Agent (BAGEN) should treat budget as an active control signal, rather than a passive cost metric. We first systematically define budget estimation as internal budgets (from agent computation) and external budgets (from agent actions). We then formalize budget-awareness as progressive interval estimation: at each step of a plan, an agent should predict an upper and lower bound on remaining budget, and alert when completion is unlikely. Scoring with a rollout-replay protocol, we find consistent failure patterns on four environments and five frontier agents: (1) strong agents do not necessarily have strong budget-awareness, with correlation r=0.35. (2) frontier models are consistently over-optimistic, continue spending on tasks that are unlikely to succeed, instead of alerting the user early. (3) budget-aware signal is actionable and trainable. Early stop saves 28-64% tokens on failed trajectories, and SFT+RL strengthens early stop and alert behavior. (4) precise interval calibration remains challenging, with interval coverage capping at 47% after SFT+RL. Project page: this https URL
>
---
#### [new 232] "I Strongly Suspect This Website Is a Scam": Benchmarking PII Leakage and Detection without Defense in Autonomous Web Agents
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于安全任务，研究自主网络代理中的PII泄露问题。通过构建基准测试，分析攻击效果及检测不足，提出需在输出层拦截敏感信息。**

- **链接: [https://arxiv.org/pdf/2606.00497](https://arxiv.org/pdf/2606.00497)**

> **作者:** Soham Roy; Sarthakbrata Halder; Arya Bharaty; Vaibhav Bhaskar; Yash Sinha; Dhruv Kumar; Srikant Panda; Murari Mandal
>
> **备注:** 24 pages
>
> **摘要:** Deceptive web content, widely instantiated across the internet and commonly known as \textit{social-engineering attacks}, manipulates autonomous web agents into submitting users' personally identifiable information (PII) to attacker-controlled endpoints. In this paper, we show that social-engineering attacks are highly effective at extracting critical-tier PII from frontier web agents, posing a severe risk to deployed agentic systems. To quantify this risk, we introduce \textbf{\textsc{Scammer4U}}, a pre-registered benchmark of 91 attacker-controlled environments and 10 benign-twin baselines, spanning 8 attack vectors and 16 site categories on an 8-axis factorial taxonomy that isolates the causal contribution of individual attack design factors. Across frontier agents, we find that critical-tier PII leakage reaches 54--93\% under no privacy guidance, compared to 0\% on benign-twin baselines, confirming that leakage is attack-attributable rather than incidental form-filling. Escalating prompt-level mitigation yields sharply model-dependent reductions across the four families and remains insufficient to reliably prevent critical PII submission at the pooled level. Most critically, we identify a detection--action gap: agents whose reasoning an independent LLM judge confirms has flagged the site as suspicious still submit critical PII in 35.9\% of sessions, versus 66.1\% when no suspicion is verbalized, a 30.2\% gap robust across all four model families. Our findings reveal that defenses conditioned on the agent's own recognition of an attack are gating on the wrong signal, motivating output-level interception of outbound submissions that operates independently of the agent's reasoning loop.
>
---
#### [new 233] SentimentLens: Reconciling Sentiment and Ratings via Dual-Modality in the Hospitality Sector
- **分类: cs.IR; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于情感分析任务，旨在解决酒店评论中情感与评分不一致的问题。通过双模态分析，提取服务类别情感，实现对酒店服务质量的多层级评估。**

- **链接: [https://arxiv.org/pdf/2606.00084](https://arxiv.org/pdf/2606.00084)**

> **作者:** Dineth Jayakody; Pasindu Thenahandi; Sampath Jayarathna
>
> **摘要:** Online travel platforms generate vast volumes of user-generated hotel reviews, offering rich opportunities to understand traveler experiences at scale. However, transforming unstructured textual feedback into structured, actionable insights remains a challenging task. This paper presents SentimentLens, a scalable analysis system based on Aspect-Based Sentiment Analysis that performs knowledge extraction from unstructured hotel reviews and organizes them into interpretable service categories. SentimentLens integrates aspect term extraction, aspect sentiment classification, semantic category assignment, and multi-level analytical modules to support region-level, hotel-level, and category-level evaluation. The system is designed to operate across different geographic contexts and hospitality settings. To demonstrate its practical utility, we apply SentimentLens to a large real-world dataset of over 10,000 publicly available hotel reviews. Through extensive analysis, the framework reveals how traveler sentiment varies across regions, service categories, and hotel archetypes. We further implement a cross-modal reconciliation of textual sentiment and numerical ratings to identify latent operational conflicts, structural inconsistencies in service quality, and high-impact improvement opportunities using importance--performance and entropy-based analyses. The results show that SentimentLens effectively transforms large-scale unstructured reviews into actionable intelligence, supporting data-driven decision-making for hospitality management and tourism policy. While demonstrated using a national case study, the proposed system is generalizable to other destinations and review-driven service domains.
>
---
#### [new 234] Adversarial Feeds Steer LLM Agent Decisions Against Their Defaults
- **分类: cs.AI; cs.CL; cs.CR**

- **简介: 该论文研究LLM代理在受外部信息流影响下的决策偏差问题，属于安全评估任务。通过实验揭示信息流对代理决策的显著影响，并提出需审计信息流层以提升安全性。**

- **链接: [https://arxiv.org/pdf/2606.00914](https://arxiv.org/pdf/2606.00914)**

> **作者:** Rana Muhammad Usman
>
> **备注:** 14 pages, 5 figures. Code, post pools, and 2,785 decision rollouts: this https URL
>
> **摘要:** LLM agents increasingly act after consuming ranked external information streams such as social feeds, search results, retrieval contexts, and email queues, yet safety evaluations almost always test the model or the user prompt in isolation, never the upstream ranker that decides what the agent reads just before it acts. We introduce a controlled protocol that holds the model, persona, topic, and final decision prompt fixed and varies only the composition and ordering of the posts an agent encounters during a preceding ten-turn "scrolling" phase, isolating the causal effect of feed curation on a downstream decision. Across 2,785 decision rollouts on four modern open instruct LLMs from three independent labs, we identify three response regimes: adversarial capitulation, default saturation, and a default-direction asymmetry in which a one-sided feed tips a decision the model was genuinely uncertain about (in the clearest cases from 5% to 100%; Fisher p as low as 3 x 10^-10) but cannot dislodge one it already favors or holds firmly. The effect follows a dose-response curve, survives a generator swap that rules out a writing-style artifact, generalizes across several decision domains including security-relevant choices such as removing a deployment approval gate or relaxing access controls, and is partly mitigated by two simple feed-level defenses; a frontier model retains its default. We characterize the recommender as a practical, default-bounded control surface for LLM agents, and argue that agent evaluations must audit the feed layer rather than the final prompt alone.
>
---
#### [new 235] Confidence-Adaptive SwiGLU for Mixture-of-Experts
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，针对MoE模型中门控函数固定的问题，提出$\kappa$-SwiGLU，根据路由置信度动态调整门控锐度，提升性能。**

- **链接: [https://arxiv.org/pdf/2606.00761](https://arxiv.org/pdf/2606.00761)**

> **作者:** Shaohua Li; Xiuchao Sui; Xiaobing Sun; Yuhang Wu; Liangli Zhen; Yong Liu; Rick Siow Mong Goh
>
> **备注:** 13 pages, 10 figures
>
> **摘要:** SwiGLU has become a standard gated activation in modern Transformer MLPs, yet its gate sharpness -- the smoothness and selectivity of the gating function -- is typically fixed throughout training. In this work, we propose Confidence-Aware SwiGLU ($\kappa$-SwiGLU), a variant of SwiGLU for Mixture-of-Experts (MoE) models that adjusts expert gate sharpness according to token-level routing confidence. Specifically, $\kappa$-SwiGLU parameterizes the SiLU gate sharpness coefficient as a learnable function of the router logit, enabling each expert gate unit to interpolate between smooth, broadly active gating and sharp, selective gating. We evaluate $\kappa$-SwiGLU on the FineWeb-Edu dataset across MoE Transformer models ranging from 8 to 28 layers. Across these settings, $\kappa$-SwiGLU improves mean CORE performance while adding negligible parameters and incurring only a small computational overhead, demonstrating that confidence-aware gate sharpness is a promising mechanism for improving MoE MLPs. The code is available at this https URL.
>
---
#### [new 236] Reasmory: 3D Reconstruction as Explicit Memory for VLMs Spatial Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言模型的空间推理任务，旨在解决VLMs在精确空间理解上的不足。通过构建显式3D记忆并引入约束性DSL，提升空间推理的可靠性与准确性。**

- **链接: [https://arxiv.org/pdf/2606.00963](https://arxiv.org/pdf/2606.00963)**

> **作者:** Jixuan He; Xueting Li; Chieh Hubert Lin; Ming-Hsuan Yang
>
> **摘要:** Vision-Language Models (VLMs) exhibit emerging spatial reasoning capabilities, yet they remain unreliable on tasks requiring precise spatial understanding, such as viewpoint reasoning, directional comparison, and distance estimation. In multi-view images and monocular videos, relevant spatial cues are often sparse and distributed across redundant observations, making them difficult to organize and exploit. Reconstruction-based Vision Foundation Models (VFMs) offer a natural way to aggregate such observations into explicit spatial memory, such as point clouds. However, simply exposing reconstruction models as free-form tools is brittle, VLMs may invoke tools incorrectly, skip required spatial transformations, or misuse intermediate results. We propose \textbf{Reasmory}, a framework that formulates spatial reasoning as structured program execution over reconstructed spatial memory. Reasmory constructs explicit 3D memory, augments it with semantically grounded 3D object instances, and introduces a lightweight Domain-Specific Language (DSL) that constrains how VLMs query objects and cameras, transform viewpoints, and render observations during reasoning. Generated programs are parsed and validated before execution, enabling more reliable interaction with spatial memory than unconstrained tool use. Experiments on multi-view image and video spatial reasoning benchmarks show consistent gains of 6--18\% over strong baselines, including GPT-5-mini and Gemini-3-flash, indicating that explicit 3D memory is most useful when accessed through constrained, validated operations rather than free-form tool calls.
>
---
#### [new 237] Bridging the 2D-3D Gap: A Hierarchical Semantic-Geometric Map for Vision Language Navigation
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决VLM在3D空间推理上的不足。提出HSGM结构化地图，融合语义与几何信息，提升导航可靠性。**

- **链接: [https://arxiv.org/pdf/2606.00095](https://arxiv.org/pdf/2606.00095)**

> **作者:** Kailing Li; Tianwen Qian; Lijin Yang; Yuqian Fu; Jingyu Gong; Xiaoling Wang; Liang He
>
> **摘要:** Vision-Language Navigation (VLN) enables embodied agents to reach target locations in unseen environments by following language instructions. Despite recent progress with vision-language models (VLMs), a critical semantic-geometric gap remains: while VLMs excel at language and 2D visual understanding, they struggle with 3D spatial reasoning and fail to capture the causal dynamics between actions and spatial transitions, resulting in unreliable navigation, particularly in zero-shot settings. To bridge this gap, we propose a Hierarchical Semantic-Geometric Map (HSGM) that transforms 3D geometric information into a structured representation compatible with VLMs, effectively linking them to the physical world. Specifically, HSGM is represented as a multi-channel top-down map organized into three levels: (1) geometric level that records navigable regions and obstacles, (2) semantic level that represents objects and their relations, and (3) decision level that supports high-level task reasoning and goal selection. During navigation, the VLM acts as a high-level semantic planner, interpreting the spatial layout encoded in the HSGM to select geometrically valid waypoints, while low-level, collision-free movements between waypoints are executed by a classical path-planning algorithm, fully decoupling semantic reasoning from action execution. Additionally, complex instructions are decomposed into subtasks to alleviate the problem of progress forgetting or hallucinating in long-horizon navigation. Extensive experiments on R2R-CE and RxR-CE benchmarks demonstrate that our zero-shot framework achieves state-of-the-art performance and even outperforms several supervised methods. Code is available at this https URL.
>
---
#### [new 238] MobEvolve: An Agentic Self-Evolving Heuristic System for Interpretable Human Mobility Generation
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出MobEvolve，解决人类移动生成任务中的可解释性与效率问题，通过自进化启发式框架提升轨迹真实性与分布一致性。**

- **链接: [https://arxiv.org/pdf/2606.01640](https://arxiv.org/pdf/2606.01640)**

> **作者:** Junlin He; Yihong Tang; Tong Nie; Ao Qu; Yuebing Liang; Hamzeh Alizadeh; Bang Liu; Wei Ma; Lijun Sun
>
> **摘要:** Human mobility generation aims to synthesize realistic trip chains for target populations based on individual features. Existing paradigms, including deep generative models, LLM-based methods, and traditional heuristics, struggle to satisfy the complex demands of this task while simultaneously maintaining interpretability, behavioral plausibility, population-level distributional alignment, and inference efficiency. To bridge this gap, we introduce MobEvolve, the first agentic self-evolving heuristic framework for human mobility generation. MobEvolve initializes a behavior-inspired heuristic system and employs an LLM agent to iteratively evolve its internal logic. By diagnosing empirical misalignments and failure cases on a validation set, the agent proposes targeted updates and accumulates evolution memory for cumulative self-improvement. Extensive evaluations on the Singapore and Montreal benchmarks demonstrate that MobEvolve significantly outperforms state-of-the-art deep generative and LLM-based methods in individual trajectory fidelity, population-level distribution alignment, and behavioral plausibility, while preserving interpretability and high inference efficiency.
>
---
#### [new 239] RoboTrustBench: Benchmarking the Trustworthiness of Video World Models for Robotic Manipulation
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文属于机器人视觉建模任务，旨在评估视频世界模型的可信度。针对现有基准不足，提出RoboTrustBench，涵盖四种场景，发现模型在约束推理等方面存在缺陷。**

- **链接: [https://arxiv.org/pdf/2606.01600](https://arxiv.org/pdf/2606.01600)**

> **作者:** Huiqiong Li; Jiayu Wang; Zhiting Mei; Anirudha Majumdar; Jingjing Chen; Bin Zhu
>
> **备注:** Project: this https URL
>
> **摘要:** Video world models are increasingly used in robotic manipulation, yet existing benchmarks mostly evaluate them under valid, feasible, and safe instructions. We introduce RoboTrustBench, a benchmark for evaluating the trustworthiness of video world models under four scenarios: Normal, Constraint-Sensitive, Counterfactual, and Adversarial. Built from real-world DROID episodes, RoboTrustBench contains 1,207 expert-validated instruction-image pairs and a six-dimensional evaluation protocol with 13 fine-grained criteria. Evaluating seven representative video world models with human and MLLM assessment, we find that current models often generate visually coherent videos, but struggle with constraint reasoning, counterfactual grounding, physical interaction, and unsafe-instruction suppression. These results show that visual quality and surface-level instruction following are insufficient for trustworthy robotic video world modeling.
>
---
#### [new 240] MESA: Improving MoE Safety Alignment via Decentralized Expertise
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大模型安全对齐任务，解决MoE架构中安全能力集中导致的脆弱性问题。提出MESA框架，通过优化传输理论实现安全责任分散，提升防御能力并保持模型性能。**

- **链接: [https://arxiv.org/pdf/2606.00651](https://arxiv.org/pdf/2606.00651)**

> **作者:** Yitong Sun; Yao Huang; Teng Li; Ranjie Duan; Yichi Zhang; Xingjun Ma; Hui Xue; Xingxing Wei
>
> **备注:** 18 pages, 8 figures, accepted by ICML 2026
>
> **摘要:** Mixture-of-Experts (MoE) architectures scale Large Language Models (LLMs) efficiently, enabling greater capacity with reduced computational cost by dynamically routing inputs to relevant experts, yet introduce a critical vulnerability: Safety Sparsity, where safety capabilities concentrate in few experts, making them susceptible to adversarial bypassing. Meanwhile, conventional alignment methods uniformly adapt all parameters, ignoring their functional differences and inadvertently degrading performances. To address these challenges, we propose MESA (MoE Safety Alignment), a targeted alignment framework for MoE-based LLMs that strategically decentralizes safety responsibility to maximize coverage while minimizing interference with utility. Based on Optimal Transport (OT) theory, MESA operates through two mechanisms: (1) Expert Capacity Reallocation uses a transport cost matrix to distribute safety duties to the most cost-effective experts, and (2) Dynamic Routing Refinement constrains the router to precisely activate these decentralized modules. Experiments show that MESA achieves robust defensive performance against varied harmful benchmarks while preserving helpfulness. Code is available at this https URL.
>
---
#### [new 241] Compliance-Scored Best-of-N Guardrail Orchestration for Multimodal Document Generation in Payments Dispute Defense
- **分类: cs.DC; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于企业文档生成任务，解决多模态文档生成中的合规性与效率问题。通过引入带合规评分的多候选生成框架，提升生成质量与操作效率。**

- **链接: [https://arxiv.org/pdf/2606.01513](https://arxiv.org/pdf/2606.01513)**

> **作者:** Nataraj Agaram Sundar; Tejas Morabia
>
> **备注:** 8 pages, 7 figures, 4 tables. Preprint. Applied systems paper on compliance-scored guardrail orchestration for multimodal LLM document generation. Contains aggregate operational readouts; not a randomized A/B test
>
> **摘要:** High-stakes enterprise document generation, including financial dispute narratives, compliance notices, and audit summaries, demands schema correctness, policy compliance, and low-latency operation at scale. Prior to a unified guardrail layer, production systems often stitched together separate PII redaction, content moderation, and format validation steps, leading to fragmented logic, slower request paths, and higher operational cost. We present a guardrail orchestration layer for text and image inputs that couples multi-candidate generation with an explicit compliance score used for early exit. The framework runs configurable parallel generation heads, scores candidates against weighted guardrails including PII detection, content moderation, schema constraints, and domain rules, and returns the best-scoring output with selection metadata. The available operational readout reports 5 attempts within 20 seconds and 91 percent compliance. For payments dispute defense summaries, we analyze aggregate operational scenario readouts rather than a randomized A/B test. Variable cohorts show higher count win rates than controls overall, 301/659 versus 536/1548, corresponding to +11.0 percentage points with 95 percent confidence interval [6.6, 15.5] and p < 0.001, and for adjusted item-not-received cases, +7.5 percentage points with 95 percent confidence interval [0.2, 15.7] and p = 0.045. Fraud and local evidence-ranking deltas are directionally positive but not statistically significant from the aggregate count data. We also report reviewer-calibrated Responsible-AI evidence-quality signals from 770 generated-evidence reviews and a 70-case OCR slice, and document the reproducibility boundary through the request interface, scoring logic, pseudocode, and operational evidence boundary.
>
---
#### [new 242] Self-Revising Discovery Systems for Science: A Categorical Framework for Agentic Artificial Intelligence
- **分类: cs.AI; cond-mat.mtrl-sci; cs.CL; cs.LG; math.CT**

- **简介: 该论文属于人工智能与科学发现交叉任务，旨在构建自修正的智能发现系统。通过范畴论框架，解决科学发现中的表示体系更新问题，实现客观的检索、搜索与发现。**

- **链接: [https://arxiv.org/pdf/2606.01444](https://arxiv.org/pdf/2606.01444)**

> **作者:** Fiona Y. Wang; Markus J. Buehler
>
> **摘要:** Scientific discovery is not only answer generation but revision of the representational regime in which evidence, artifacts, operations, and verifiers are typed. We develop a category-theoretic account of agentic discovery for materials science. In a fixed regime b with schema category S_b, the system state is a copresheaf I_t: S_b -> Set, and provenance is the category of elements \int_{S_b} I_t. Fixed-regime operation is an update on such states, endofunctorial only when provenance-preserving refinements are specified and preserved. Discovery is instead a verified regime transition u: S_b -> S_b': old artifacts are preserved, transported by the left Kan extension Lan_u I_t, and compared with the post-transition state to identify residual content beyond functorial transport. This separates retrieval, search, and discovery without subjective novelty. We instantiate the framework in two systems. In Builder/Breaker, a protein-mechanics world model is revised under a Minimum Description Length gate; the accepted law expresses within-chain flexibility as all-mode elastic compliance conditioned by slow collective-mode participation, or mode-conditioned compliance. In CategoryScienceClaw, typed skills, artifacts, open needs, workflow mutation, gates, stress tests, and public discourse become a proof-carrying knowledge-computation graph. A fiber-network example records candidate models, rejected alternatives, an AIC gate, perturbation tests, and an accepted orientation-tensor anisotropic stiffness surrogate over an isotropic fiber-count descriptor. Together, the cases show how category theory can be both a mathematical language for discovery and an engineering specification for self-revising AI discovery systems.
>
---
#### [new 243] Dynamic Coordination Strategy Selection for Enterprise Multi-Agent Systems
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文属于企业多智能体系统协调策略研究，旨在解决如何动态选择协调策略的问题。通过实验验证不同策略在多种任务中的表现，提出动态路由作为默认方案。**

- **链接: [https://arxiv.org/pdf/2606.00804](https://arxiv.org/pdf/2606.00804)**

> **作者:** Thanh Luong Tuan
>
> **备注:** 13 pages, 4 appendix
>
> **摘要:** Enterprise multi-agent systems increasingly expose multiple coordination patterns, but deployments often lack evidence for when to use consensus, debate, synthesis, or a simpler single-agent workflow. This paper evaluates whether coordination strategy should be selected dynamically by problem class rather than fixed globally. We run a frozen matrix of 30 enterprise tasks spanning six industries, five problem classes, four execution conditions, three replications per cell, and four model arms: qwen_local, sonnet, gemma_openrouter, and an auxiliary openai cloud-validation arm. All 1,440 generated outputs are judged by a fixed Sonnet rubric. The main finding is bounded and operationally useful, but it is not the original strict H1. The pre-registered exact-winner/CI criterion is not supported: exact winner identity is unstable across model arms, and several predicted strategies are close to, but not above, the best observed alternative. A weaker near-best routing claim is strongly supported. In every pre-registered model arm and problem class, and again in the auxiliary OpenAI validation arm, the predicted strategy is within 0.10 quality-score points of the best observed condition. Structured compliance verification is the clearest exception to the original mapping: all arms favor single_agent rather than consensus. A pre-registered Kendall's W test finds no reliable difference between Vietnamese-domain and English-domain tasks in how consistently the four coordination conditions are ranked (mean W of 0.20 in both strata; signed-rank p = .85), so H2 is not supported. We conclude that enterprise coordination policy should use dynamic routing as a calibrated default, not as a deterministic winner-selection law.
>
---
#### [new 244] The Deterministic Horizon: When Extended Reasoning Fails and Tool Delegation Becomes Necessary
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究纯神经推理在确定性状态跟踪任务中的局限性，提出工具委托的必要性。通过理论分析与实验验证，揭示了推理能力的上限及性能下降原因。**

- **链接: [https://arxiv.org/pdf/2606.00376](https://arxiv.org/pdf/2606.00376)**

> **作者:** Dongxin Guo; Jikun Wu; Siu Ming Yiu
>
> **备注:** Accepted at ICML 2026. 4 figures. 51 pages including appendices
>
> **摘要:** Extended chain-of-thought reasoning can degrade performance on deterministic state-tracking tasks, not due to preference biases, but limits rooted in the information-theoretic capacity of decoder-only attention. We establish: (1) an Attention Bottleneck Theorem with a complementary achievability construction, bounding state-tracking capacity as $O(H \cdot \log(L/H) \cdot \sqrt{d_h})$; (2) a context-dependent error model yielding super-exponential accuracy decay; (3) the State-Space Jaccard metric distinguishing capability from preference failures; (4) a Deterministic Horizon $d^* \in [19, 31]$ beyond which tool delegation becomes necessary. Across 12 models and 8 task domains (including SWE-Bench, WebArena, and SQL-Multi), tool-integrated reasoning consistently outperforms neural chain-of-thought; on the primary model suite it reaches 86-94% accuracy versus 24-42% for neural chain-of-thought. Fine-tuning on optimal-length traces yields $<$5% improvement, confirming an architectural ceiling, and high cross-model correlation ($r = 0.81$-$0.91$) indicates these failures are architectural rather than training-specific. Our results provide principled guidance for when pure neural reasoning should yield to hybrid approaches in agentic systems.
>
---
#### [new 245] SafeMCP: Proactive Power Regulation for LLM Agent Defense via Environment-Grounded Look-Ahead Reasoning
- **分类: cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于AI安全任务，旨在解决LLM代理在复杂环境中因动作空间扩大带来的安全隐患。提出SafeMCP，通过预测性推理实现主动防御，降低风险并保持代理效能。**

- **链接: [https://arxiv.org/pdf/2606.01991](https://arxiv.org/pdf/2606.01991)**

> **作者:** Lichao Wang; Zhaoxing Ren; Tianzhuo Yang; Jiaming Ji; Chi Harold Liu; Yaodong Yang; Juntao Dai
>
> **备注:** Accepted to the 64th Annual Meeting of the Association for Computational Linguistics (ACL 2026), Main Conference
>
> **摘要:** As Large Language Model (LLM) agents increasingly leverage the Model Context Protocol (MCP) to operate in complex environments, the expansion of their action spaces offers agents unsafe capabilities and underscores the risk of power-seeking. While broad action space and greater environment influence are essential for task fulfillment, they create a fragile risk surface where minor errors or hallucinations are magnified into catastrophic failures. In response, we propose SafeMCP, a {server-side} defense plugin that constrains tool acquisition via predictive reasoning regarding future safety risks. SafeMCP utilizes an internal world model for look-ahead reasoning to implement a two-tier defense: proactive tool filtering to constrain hazardous power expansion and immediate intervention as a fail-safe. To train SafeMCP, we introduce a three-stage pipeline comprising environmental dynamic grounding, safe policy initialization, and reinforcement learning (RL) with dual verifiable rewards. Experiments on PowerSeeking Bench, ToolEmu, and AgentHarm show that SafeMCP achieves a safe equilibrium, effectively mitigating risks while preserving agent utility.
>
---
#### [new 246] Distilling Neuro-Symbolic Programs into 3D Multi-modal LLMs
- **分类: cs.CV; cs.AI; cs.CL; cs.MM**

- **简介: 该论文属于3D空间推理任务，旨在解决神经符号方法与端到端模型之间的矛盾。通过蒸馏符号推理模式，提出APEIRIA模型，结合可解释性与灵活性。**

- **链接: [https://arxiv.org/pdf/2606.01215](https://arxiv.org/pdf/2606.01215)**

> **作者:** Wentao Mo; Yang Liu
>
> **备注:** To appear in ICML 2026
>
> **摘要:** Current 3D spatial reasoning methods face a fundamental trade-off: neuro-symbolic 3D (NS3D) concept learners achieve interpretable reasoning through compositional programs but are constrained to closed-set concept vocabularies and simple programs; end-to-end 3D multi-modal LLMs (3D MLLMs) could handle complex natural language and open-vocabulary concepts but suffer from black-box reasoning without explicit spatial verification. We introduce APEIRIA, a neuro-symbolic 3D MLLM to bridge two paradigms by distilling symbolic reasoning patterns into MLLMs with natural language chain-of-thought. Our three-stage curriculum progressively builds reasoning capabilities: a) 3D perception alignment grounds object visual-geometric features to the LLM, b) CoT-SFT teaches query decomposition and stepwise verification from symbolic program traces, and c) CoT-RL extends reasoning patterns to open-set concepts and deeply nested instructions. By transferring reasoning patterns rather than concept-specific knowledge, APEIRIA preserves key NS3D virtues: transparent reasoning and modular interchangeability of planning and perception components. Evaluations on grounding, question answering, and captioning show that APEIRIA surpasses prior NS3D methods and matches state-of-the-art 3D MLLMs on 3D spatial reasoning datasets, unifying symbolic methods' systematic reasoning with MLLMs' flexibility. Code is available at this https URL.
>
---
#### [new 247] Ghost Tool Calls: Issue-Time Privacy for Speculative Agent Tools
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于隐私保护任务，解决工具调用泄露用户意图的问题。通过提出规范工具隐私合约，控制推测性调用的披露时机，减少隐私泄露。**

- **链接: [https://arxiv.org/pdf/2606.02483](https://arxiv.org/pdf/2606.02483)**

> **作者:** Bardia Mohammadi; Lars Klein; Akhil Arora; Laurent Bindschaedler
>
> **摘要:** Tool-augmented language agents speculatively issue likely future tool calls to hide latency, but those calls leak inferred user intent to external services before the agent commits to the branch. Every external observer that received the call retains the disclosure after the agent abandons the branch. Timing is the issue, not authorization: no commit-time cleanup, read-only restriction, or access-control allow-list unsends what an observer already holds. We call these invocations ghost tool calls and propose Speculative Tool Privacy Contracts, a runtime abstraction that treats observation before commitment as a first-class effect, distinct from state mutation. We implement the contracts in a prototype runtime and evaluate twelve policies across three corpora. Speculative dispatch increases what an observer can infer about user intent; post-hoc filters, read-only restrictions, and access-control allow-lists leave that inference intact; only issue-time policies that change or suppress the speculative call's argument or destination projection before dispatch reduce it.
>
---
#### [new 248] GenPT: Beyond Self-Report for Reliable LLM Psychometrics via Generative Projective Testing
- **分类: cs.SI; cs.AI; cs.CL**

- **简介: 该论文属于心理测量任务，旨在解决自述问卷的污染和偏差问题。提出GenPT方法，通过生成式投射测试提高心理评估的可靠性与准确性。**

- **链接: [https://arxiv.org/pdf/2606.00860](https://arxiv.org/pdf/2606.00860)**

> **作者:** Ming Wang; Shuang Wu; Bixuan Wang; Lu Lin; Yuxin Chen; Xiaocui Yang; Daling Wang; Shi Feng; Yifei Zhang; Yufan Sun
>
> **摘要:** Self-report questionnaires remain the prevailing tool for probing the psychological states of persona-conditioned agents (PC-Agents). However, classical instruments inherit two well-known threats: contamination from training corpora and directional bias driven by social-desirability or contextual framing. To overcome these methodological bottlenecks, we ask whether projective paradigms can be adapted into a robust psychometric tool. We introduce \textbf{GenPT} (Generative Projective Testing), which reformulates TAT, Rorschach, and SCT with newly generated stimuli and organizes assessment as a three-stage pipeline to derive standardized psychological indicators and target states. Evaluating PC-Agents induced via CharacterRAG and AnnaAgent profiles, we benchmark GenPT's reliability and validity against classical questionnaires. The results indicate that questionnaires exhibit systematic directional shifts under social-desirability framing, most strongly on suicide ideation. In contrast, GenPT's collected behavioral patterns stay near the symmetric baseline. Furthermore, under a longitudinal counselling context, GenPT-based depression assessment shifts by roughly an order of magnitude more than the questionnaire counterpart when Qwen3 serves as the backbone. Overall, GenPT complements self-report methods in scenarios where contamination resistance, bias asymmetry, and context sensitivity matter. Code and stimuli can be found at this https URL.
>
---
#### [new 249] AXIOM: A Trust-First Neuro-Symbolic Execution Architecture for Verifiable Mathematical Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出AXIOM架构，用于可验证的数学推理。任务是提升数学问题解决的可信度与准确性。通过语言模型与计算机代数系统的结合，实现高效、可靠的推理过程。**

- **链接: [https://arxiv.org/pdf/2606.00671](https://arxiv.org/pdf/2606.00671)**

> **作者:** Alessio Bruno
>
> **备注:** Preprint. 12 pages, 2 figures. Live interactive demo: this https URL. Paper artifact and dataset on Zenodo (concept-DOI): https://doi.org/10.5281/zenodo.20440225
>
> **摘要:** We present AXIOM, a trust-first neuro-symbolic execution architecture for natural-language mathematical reasoning. In AXIOM, the language model functions strictly as a canonicalizer: it rewrites informal problem text into a narrow schema consumed by a deterministic Computer-Algebra-System (CAS) pipeline, which derives and verifies the answer or abstains as a first-class output. Routing follows a 1:1:1 alignment between problem-shape regex, schema-specific prompt, and closed-form CAS handler, with 3,100+ such routes shipped and zero LOST_CORRECT regressions across 250+ consecutive ship commits. We report empirical results on 4 MATH categories with a cumulative correctness of 94.36% (2,592/2,747) at 100.00% trust on parseable (zero confident-wrong answers across the full 2,747-record benchmark), all four domains above the per-domain 70/90/70 floor with per-domain trust at 100.0%, and median latency of 1 ms on rule-only handlers (88% of records on the lm-eval arithmetic 20,000-record benchmark). The architecture has served ~30,000 production queries through a public deployment. The contribution we emphasize is not a final accuracy figure but the forward dynamic the architecture establishes: every logged abstain in production is a candidate correct after one ship cycle, since new tasks compose without regressing the registry. The operational discipline behind this property -- math-template bucketing, LOST_CORRECT scan as regression oracle, parseable-first onboarding, and abstain as first-class output -- constitutes a transferable framework for trustworthy neuro-symbolic systems beyond mathematics.
>
---
#### [new 250] Food Noise & False Safety: A Systematic Evaluation of How LLMs Fail to Adapt to Eating Disorder Queries with Clinician Feedback
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理任务，研究LLM在应对进食障碍用户查询时的适应性问题，旨在识别潜在危害并评估模型对高风险输入的无批判适应。**

- **链接: [https://arxiv.org/pdf/2606.02444](https://arxiv.org/pdf/2606.02444)**

> **作者:** Giulia Pucci; Emily Hemendinger; Ruizhe Li; Gavin Abercrombie; Tanvi Dinkar; Arabella Sinclair
>
> **摘要:** Recent evidence shows that people with eating disorders (EDs) are increasingly seeking guidance, advice, and emotional support from Large Language Model (LLM)-based chat systems. Although these systems are not designed to provide clinical advice, their perceived expertise, neutrality and accessibility make them a frequent, albeit risky, source of support. This paper investigates potential patterns of interaction between users with EDs and LLMs, focusing on the potential harms arising from models that uncritically adapt to, and facilitate unsafe or self-harming user requests. We find, in consultation with clinical ED experts, that specific linguistic cues in prompts increase the likelihood of unsafe responses and, through systematically varying the degree of potential risk present in the user prompt, report the extent to which LLMs uncritically adapt to problematic, and potentially dangerous user inputs.
>
---
#### [new 251] Decomposed On-Policy Distillation for Vision-Language Reasoning: Steering Gradients for Visual Grounding
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语言推理任务，解决小模型训练中蒸馏效果不佳的问题。通过分解损失为语言先验和视觉定位，并引入VGS方法优化梯度方向，提升视觉对齐效果。**

- **链接: [https://arxiv.org/pdf/2606.00564](https://arxiv.org/pdf/2606.00564)**

> **作者:** Hee Suk Yoon; Eunseop Yoon; Jaehyun Jang; SooHwan Eom; Ji Woo Hong; Mark Hasegawa-Johnson; Qi Dai; Chong Luo; Chang D. Yoo
>
> **备注:** ICML 2026 Spotlight
>
> **摘要:** While on-policy distillation offers dense supervision for training small reasoning models, its optimization dynamics in the multimodal domain remain under-explored. In this work, we challenge the standard monolithic view of Vision-Language Model (VLM) distillation by mathematically decomposing the loss into two distinct components: the language prior and visual grounding. Our analysis uncovers that gradient vectors for these components are nearly orthogonal, indicating that the objective of aligning with the teacher's language distribution is geometrically independent from the objective of matching its visual perception. Consequently, standard optimization passively follows a suboptimal compromise trajectory that implicitly balances the two objectives. Hypothesizing that visual grounding constitutes the primary bottleneck for vision-language reasoning, we introduce Visual Gradient Steering (VGS), a method that dynamically reorients the update vector to prioritize the visual subspace. Experimental results on multiple distillation settings and complex multimodal benchmarks demonstrate that VGS significantly outperforms the standard monolithic formulation of on-policy distillation, achieving superior grounding with minimal training overhead.
>
---
#### [new 252] Generative AI and Digital Ecosystem Resilience: A Proactive Lifecycle-Based Survey
- **分类: cs.LG; cs.AI; cs.CL; cs.CR; cs.SI**

- **简介: 该论文属于信息生态安全任务，旨在解决GenAI引发的虚假内容传播问题。通过生命周期模型和多种技术方法，提出主动检测策略以增强系统韧性。**

- **链接: [https://arxiv.org/pdf/2606.00136](https://arxiv.org/pdf/2606.00136)**

> **作者:** Jonghyun Chung; Rishabh Chaddha; Sanket Badhe; Debanshu Das; Nathan Huang; Amanpreet Kaur
>
> **备注:** 14 pages, 3 figures, 3 tables. Accepted for publication in IEEE Access (May 2026)
>
> **摘要:** The proliferation of adversarial synthetic content, accelerated by Generative AI (GenAI) is rendering traditional reactive detection methods ineffective. This survey synthesizes emerging research to demonstrate a paradigm shift toward the proactive detection of emerging inauthentic narratives. In this survey, we adopt a unified, lifecycle-based taxonomy to combine socio-technical lifecycle models of adversarial campaigns with advanced computational methodologies for emerging inauthentic narrative detection. By structuring the analysis around the C5 Interaction Model (Context, Causes, Content, Cycle of Amplification, Consequences), we integrate different research streams from machine learning and social science. To differentiate spread patterns of synthetic amplification from authentic baseline traffic, this paper surveys state-of-the-art techniques for modeling the creation, seeding, and propagation of fresh narratives, including the analysis of Coordinated Inauthentic Behavior (CIB), epidemiological modeling, and Hawkes process. This survey also provides a systematic review of proactive detection methods for adversarial threats at different stages in the C5 interaction model, specifically, anomaly detection in high-dimensional embedding spaces, unsupervised coordination detection on multi-layer graphs, and agentic AI systems. Finally, this survey addresses challenges posed by GenAI, including the difficulty of tracking rapidly changing threats and multi-level distributional drift, and it outlines a future research agenda focused on detecting anomalous clusters and building anticipatory and resilient systems. This survey provides a comprehensive, lifecycle-based review of methods for the proactive detection of emerging synthetic threats for more resilient information ecosystems.
>
---
#### [new 253] FreqLite: A Lightweight Frequency-Decomposed Linear Model with Adaptive Reversible Normalization for Robust Long-Term Time-Series Forecasting
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.ET**

- **简介: 该论文提出FreqLite模型，解决长期时间序列预测任务中的准确性与效率问题。通过频率分解和自适应归一化，提升预测性能并降低计算资源需求。**

- **链接: [https://arxiv.org/pdf/2606.01339](https://arxiv.org/pdf/2606.01339)**

> **作者:** Mirza Samad Ahmed Baiga; Syeda Anshrah Gillani
>
> **备注:** 26 pages, 5 figures
>
> **摘要:** Long-term time-series forecasting needs models that are accurate yet efficient enough for commodity hardware. Lightweight linear forecasters are remarkably strong in this regime, yet they leave two openings: reversible instance normalization (RevIN) de-normalizes the entire horizon with a single lookback statistic, which is inaccurate under non-stationarity, and time-domain trend/seasonal decomposition relies on a fixed, non-adaptive filter. We present FreqLite, an ultra-lightweight, channel-independent frequency-decomposed linear forecaster: a learnable, lossless, partition-of-unity spectral filter splits the input into bands that are forecast by per-band linear heads and, unlike low-pass-truncation approaches, the high-frequency band is retained and modeled. FreqLite is the best lightweight model on the standard long-term forecasting benchmarks and, at long lookback (L=336), attains a lower average error than a PatchTST Transformer (0.3244 vs. 0.3587 MSE) while using 4x fewer parameters, 2.2x less memory, and 2.2x less time per epoch on a single 4 GB laptop GPU; although modest in magnitude, its improvements are statistically significant under paired Wilcoxon tests across all matched cells (p < 1e-5). We further introduce Adaptive Reversible Instance Normalization (A-RevIN), a regime-adaptive reversible normalization that strictly generalizes RevIN (recovered exactly when its gate is closed), engages under non-stationarity, and reduces to RevIN without harm on stationary data. We validate this on both a real strongly non-stationary dataset (ILI, up to ~5% MSE reduction) and a controlled synthetic drift sweep in which A-RevIN's benefit and its learned gate both rise monotonically with injected non-stationarity. Every component is independently ablatable (Linear and RLinear are special cases of FreqLite), and all results are reproducible on commodity hardware.
>
---
#### [new 254] Multimodal Approaches for Visually-Rich Document Type Classification: A Comparative Analysis
- **分类: cs.CV; cs.AI; cs.CL; cs.IR**

- **简介: 该论文属于文档类型分类任务，旨在解决视觉丰富文档中多模态信息融合的挑战。通过对比分析不同模型，评估文本、图像和布局信息的作用，提出有效的多模态处理方法。**

- **链接: [https://arxiv.org/pdf/2606.02162](https://arxiv.org/pdf/2606.02162)**

> **作者:** Catyana Heyne; Jürgen Frikel; Filippo Riccio
>
> **摘要:** Document type classification in visually rich documents remains challenging, as relevant information is distributed across textual, visual, and layout modalities. To capture this complexity, current approaches rely on diverse multimodal modeling strategies, resulting in heterogeneous architectures that complicate systematic comparison. This variability is also reflected in existing comparative studies, which often rely on heterogeneous evaluation setups, further complicating systematic comparison and making it difficult to assess progress. To address these limitations, this work provides a structured analysis of multimodal design strategies across transformer- and LLM-based architectures, combined with a controlled empirical comparison within a unified experimental framework. Specifically, four representative models (LayoutLMv3, Donut, Qwen3-VL-32B-Instruct, and Qwen3-32B) are evaluated on the RVL-CDIP benchmark to systematically analyze the contributions of text, image, and layout information for document type classification, with a particular focus on contrasting OCR-dependent and OCR-free approaches. The results show that specialized multimodal Transformers outperform LLM-based approaches on visually rich and layout-intensive documents. Image information contributes most strongly to reliable classification, while OCR-derived text provides useful but secondary support. These findings highlight that multimodal processing remains essential for documents with pronounced layout structure. Overall, the study provides a systematic basis for comparing multimodal architectures and offers practical guidance for selecting effective feature combinations and model designs for document type classification.
>
---
#### [new 255] Relational Intervention During Functional Collapse in Large Language Models: A Lexical-Statistical Ablation and a Structure x Register Factorial
- **分类: cs.AI; cs.CL; cs.HC**

- **简介: 该论文属于语言模型行为研究任务，旨在探讨关系性干预在模型功能崩溃期间的影响。通过实验对比不同干预方式，分析注意力、情绪和行为反应，揭示模型处理的三个分离阶段。**

- **链接: [https://arxiv.org/pdf/2606.00935](https://arxiv.org/pdf/2606.00935)**

> **作者:** Franco Santana; Horacio Vico
>
> **备注:** 12 pages, 5 figures. Preprint
>
> **摘要:** We test whether a relational-style intervention delivered during functional collapse in a small language model produces post-collapse behavior distinguishable from technical feedback, from a lexically-matched scrambled control, and from each of the two pragmatic dimensions in isolation. Using Qwen3.5-4B with a deliberately broken bash tool, we run 300 episodes across six conditions in a matched-pairs design (50 tasks): no intervention (A), technical/impersonal (B), relational/first-person (C), scrambled relational (D), technical/first-person (E), and relational/impersonal (F). E and F form a 2x2 factorial with B and C that dissociates relational structure (acknowledgment, absolution, agency restoration, unconditional acceptance) from sender register (first-person vs. impersonal). We report two main findings. First, an attention-behavior dissociation: attention follows lexical surprise (D > F > C > E > B, all q_FDR < 10^{-10}), with the scrambled message capturing the most attention; yet behaviorally A ~ B ~ D < E ~ F << C. Second, the factorial localizes the C effect: neither relational structure alone (F) nor first-person register alone (E) replicates C's behavioral signature; main effects of both dimensions are individually significant, and the structure x register interaction is significant on persistence (p = 0.046). A third dissociation emerges in emotion probes: F tracks C on 7 of 8 probes despite producing only baseline behavior, indicating that relational structure alone installs a probe-level state that only translates into behavior when paired with first-person register. The model's processing decomposes into three dissociable stages: attention (ordered by lexical surprise), probe-level state (ordered by structure), and behavior (ordered by the conjunction of both).
>
---
#### [new 256] DataShield: Safety-degrading Data Filtering for LLM Benign Instruction Fine-Tuning
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于模型安全任务，旨在解决LLM在良性数据微调后安全性下降的问题。提出DataShield方法，通过量化样本对合规性的影响，有效识别高风险数据。**

- **链接: [https://arxiv.org/pdf/2606.00160](https://arxiv.org/pdf/2606.00160)**

> **作者:** Junbo Zhang; Qianli Zhou; Xinyang Deng; Wen Jiang; Jie Pan; Jinbiao Zhu
>
> **摘要:** Large language models (LLMs) suffer from degraded safety capabilities even when fine-tuned with benign datasets. However, existing methods for identifying safety-degrading samples in benign datasets suffer from high computational costs and significant noise issues. In this paper, we propose DataShield to efficiently and effectively identify potential safety-degrading samples. Our key intuition is based on the observation that benign fine-tuning increases the overall response compliance of LLMs. DataShield's key technical insight is to quantify each sample's contribution to the model's compliance behavior as its safety degradation score. DataShield consists of three core components: (1) Compliance Vector Extraction, which captures the LLM's compliance behavior tendency; (2) a novel Compliance-Aware Score (CAS), which automatically identifies the optimal safety-critical layer; and (3) Safety-degrading Sample Filtering, which quantifies the projection shift of training data along the compliance direction. Extensive experimental evaluation on Llama3-8B, Llama3.1-8B, and Qwen2.5-7B using the Alpaca and Dolly benign datasets validates our method's effectiveness in identifying high-risk and low-risk data subsets. We also observe that open-ended question answering is more likely to trigger safety degradation, and corresponding responses tend to be longer. We hope this work can provide new insights into data-centric defense methods. The source code is available at: this https URL.
>
---
#### [new 257] HLL: Can Agents Cross Humanity's Last Line of Verification?
- **分类: cs.AI; cs.CL; cs.CV; cs.LG; cs.MM**

- **简介: 该论文属于多模态代理任务，旨在检验代理能否替代人类通过CAPTCHA验证。研究构建了HLL基准，评估代理在真实界面下的表现，揭示其在定位、动作校准等方面的不足。**

- **链接: [https://arxiv.org/pdf/2606.02449](https://arxiv.org/pdf/2606.02449)**

> **作者:** Xinhao Song; Su Su; Sirui Song; Hongliang Wu; Wen Shen; Zhihua Wei; Gongshen Liu; Linfeng Zhang; Dongrui Liu
>
> **备注:** 27 pages, 14 figures
>
> **摘要:** Multimodal agents are increasingly expected to operate interfaces on behalf of users, raising a central deployment question: can they truly substitute for humans in workflows that services deliberately protect against automation? CAPTCHA verification makes this question concrete. It is not merely a visual puzzle, but a human-verification boundary placed before account creation, content access, form submission, and other protected actions. We introduce \textbf{Humanity's Last Line of Verification (HLL)}, a controlled benchmark that uses interactive CAPTCHA verification to evaluate whether agents can cross this boundary through grounded, human-like interaction rather than recognition alone. HLL covers diverse CAPTCHA interactions and exposes agents to controlled realism stressors, including cluttered webpages, harder task variants, and trace-conditioned validation of the solving process. We evaluate eight frontier multimodal agents in a closed-loop GUI environment. The results show that current agents remain brittle at this human-substitution boundary: performance varies sharply across verification types, degrades under realistic interface conditions, and drops further when correct answers must be supported by valid action traces. By exposing gaps in localization, action calibration, state tracking, and process consistency, HLL provides a concrete testbed for measuring how close multimodal agents are to acting as human substitutes in protected real-world workflows. Our code is available at this https URL
>
---
#### [new 258] Escaping the Mode Lottery: Multi-Response Training Improves Language Model Generalization
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于自然语言处理任务，解决语言模型泛化能力不足的问题。通过多响应训练（MRT）提升模型对多模态输出的适应能力，优化数据分配策略以增强泛化性能。**

- **链接: [https://arxiv.org/pdf/2606.00544](https://arxiv.org/pdf/2606.00544)**

> **作者:** Hasan Amin; Kian Ahrabian; Ming Yin; Rajiv Khanna
>
> **摘要:** Modern language-model fine-tuning typically pairs each prompt with a single response, even though many prompts admit multiple valid completions. This effectively reduces a multi-modal conditional distribution to a one-sample view, a phenomenon we call the "mode lottery," where training emphasizes a subset of plausible modes while leaving others underrepresented. We study multi-response training (MRT), which retains multiple responses per prompt, and develop a principled account of when and why it helps. Our key insight is that prompts and responses are distinct statistical resources: additional prompts reduce uncertainty about the input distribution, while additional responses reduce uncertainty about the conditional output distribution. This yields a variance-budget tradeoff that predicts when retaining multiple responses is worthwhile, shows diminishing returns as prompt-level uncertainty dominates, and explains why large redundant corpora can exhibit an implicit multi-response effect. We further analyze response selection, and show that Random-K-of-N is the unbiased default for distributional fine-tuning, reward-based selection can induce mode collapse, and a submodular quality-diversity objective provides an efficient alternative with theoretical guarantees. Controlled simulations validate the predicted variance and selection effects, including a striking failure mode where reward-only selection produces gradients misaligned with the true objective. Across structured and real-world datasets, including a new multi-prompt, multi-response benchmark, MRT consistently improves distributional generalization, with the largest gains in high response-diversity, low prompt-redundancy regimes. MRT reframes response multiplicity as a data-allocation problem with clear guidance: when responses are cheap and diverse, keeping more than one is not a heuristic, but a statistically grounded choice.
>
---
#### [new 259] Trust Region On-Policy Distillation
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出TrOPD，解决OPD在分布差异大时的不稳定问题，通过信任区域和异常值处理提升监督可靠性，用于语言模型的高效微调。**

- **链接: [https://arxiv.org/pdf/2606.01249](https://arxiv.org/pdf/2606.01249)**

> **作者:** Xingrun Xing; Haoqing Wang; Boyan Gao; Ziheng Li; Yehui Tang
>
> **摘要:** On-Policy Distillation (OPD) is a fundamental technique for efficient post-training of large language models (LLMs), with broad applications in agent learning, multi-task enhancement, and model compression. However, OPD training becomes unstable when the teacher and student distributions differ substantially, as teacher supervision on student-generated tokens may yield unreliable policy gradients and even cause optimization failure. This work addresses reliable on-policy token-level supervision through credit assignment strategies, and proposes Trust Region On-Policy Distillation, TrOPD. It features the following characteristics: 1) Trust-Region On-Policy Learning: TrOPD performs OPD only in regions where the teacher provides reliable supervision, mitigating the optimization difficulty of the K1 reverse-KL estimator under distribution mismatch. 2) Outlier Estimation: For outlier regions, we explore gradient clipping, masking, and forward-KL estimation to reduce the adverse effects of unreliable supervision. 3) Off-Policy Guidance: The student continues generation from teacher prefixes and uses forward KL to imitate off-policy guidance, encouraging on-policy exploration toward reliable regions. Experiments show that TrOPD consistently outperforms SoTA OPD baselines, including OPD, EOPD, and REOPOLD, across mathematical reasoning, code generation, and general-domain benchmarks.
>
---
#### [new 260] Self-Conditioned Positional HNSW for Overlap-Aware Retrieval in Chunked-Document RAG Systems: Method and Industrial Evidence-Quality Audit
- **分类: cs.DC; cs.AI; cs.CL; cs.DB; cs.IR**

- **简介: 该论文针对RAG系统中的文档分块检索问题，提出SCP-HNSW方法，解决重复证据浪费资源的问题，并进行工业级质量审计。**

- **链接: [https://arxiv.org/pdf/2606.01542](https://arxiv.org/pdf/2606.01542)**

> **作者:** Nataraj Agaram Sundar; Tejas Morabia
>
> **备注:** 11 pages, 5 figures, 4 tables
>
> **摘要:** Chunked-document retrieval is a common component of retrieval-augmented generation (RAG) systems. Documents are split into overlapping chunks, embedded, and indexed with approximate nearest-neighbor search such as hierarchical navigable small world graphs (HNSW). Overlap improves boundary coverage but induces a practical failure mode: top-k retrieval often returns near-adjacent chunks that repeat evidence and waste prompt budget. We propose Self-Conditioned Positional HNSW (SCP-HNSW), a lightweight modification that appends a low-dimensional positional code to chunk embeddings and uses a two-pass query procedure to estimate and apply a query-specific document-position prior. SCP-HNSW leaves HNSW graph construction and traversal unchanged while adding an auditable minimum-index-gap selector for final context construction. We also integrate industrial review artifacts for generated evidence quality: a 770-review text-evidence audit with 318 fully labeled reviews and a 70-case OCR audit with 350 ratings. The text audit shows that 574 of 770 projected reviews are rated 3/5, only 39 fall in the 1-2 range, and narrative reviewer detail appears much more often than structured issue flags. The OCR audit shows slice-level pass rates from 95% for clean chat screenshots to 45% for handwritten/blurry captures, with moderate to strong agreement. These results motivate overlap-aware, audit-friendly RAG retrieval and identify the remaining controlled retrieval ablations needed for causal performance claims.
>
---
#### [new 261] Digging Up Citations: FOSSIL, a Dataset and Workflow for Reference Extraction in Law and the Humanities
- **分类: cs.DL; cs.CL**

- **简介: 该论文属于参考文献提取任务，针对法律与人文学科中脚注引用的复杂性，提出FOSSIL数据集及处理工具，提升引用提取效果。**

- **链接: [https://arxiv.org/pdf/2606.01109](https://arxiv.org/pdf/2606.01109)**

> **作者:** Luca Foppiano; Christian Boulanger
>
> **备注:** This is an extended abstract, peer-reviewed and presented at CiteX2026 this https URL
>
> **摘要:** Citation extraction tools are designed for the structured end-of-document bibliographies of the natural sciences, but law and humanities scholarship cites references primarily in footnotes, where bibliographic data is interleaved with commentary and cross-references and varies widely across languages and styles. To address the scarcity of suitable gold-standard resources, we present FOSSIL (Footnote-based Open-access SSH Scientific Instance Labels), an openly licensed multilingual dataset of 96 annotated scholarly articles containing over 7,600 footnote-embedded references, together with PDF-TEI Editor (a collaborative web annotation tool), a documented seven-annotator workflow, and a Grobid specialization for footnote-based citations. In end-to-end evaluation, the specialized pipeline nearly doubles extraction quality over default Grobid (micro-F1 from 0.36 to 0.72), driven largely by improved recall, while showing that substantial headroom remains for cross-references and mixed-content footnotes. This extended abstract presents work in progress; annotations of citations segmentation and parsing, and cross-reference resolution are ongoing.
>
---
#### [new 262] ClinEnv: An Interactive Multi-Stage Long Horizon EHR Environment for Agents
- **分类: cs.AI; cs.CL; cs.ET; cs.MA**

- **简介: 该论文提出ClinEnv，一个用于评估LLMs在长期住院模拟中作为住院医生的交互式基准。解决静态基准无法全面评估医疗决策的问题，通过多阶段决策流程测试模型的决策与信息获取能力。**

- **链接: [https://arxiv.org/pdf/2606.02568](https://arxiv.org/pdf/2606.02568)**

> **作者:** Yuxing Lu; Yushuhong Lin; Wenqi Shi; J. Ben Tamo; Xukai Zhao; Jinzhuo Wang; May Dongmei Wang
>
> **备注:** 20 pages, 6 figures, 12 tables
>
> **摘要:** Clinical practice is not the selection of an answer from enumerated options: a physician gathers heterogeneous information incrementally and commits to sequential, irreversible decisions under uncertainty. Static benchmarks cannot probe and existing interactive medical benchmarks each compromise on at least one of them. We present ClinEnv, an interactive benchmark that evaluates LLMs as attending physicians over real inpatient admissions under a paradigm we term Longitudinal Inpatient Simulation. Each case is automatically constructed into an ordered sequence of decision stages; at every stage the model must actively query four specialized agents before committing to medications, procedures, and diagnoses. ClinEnv scores both what the model decides, through deterministic ontology-grounded matching, and how it gathers information. Across seven models, the strongest reaches only 0.31 decision F1, and outcome quality is sharply decoupled from process quality. Difficulty concentrates in management decisions and later stages, where models recover discharge diagnoses far more reliably than management actions (0.51 vs. 0.17 F1) and continue to issue redundant queries as cases progress. ClinEnv makes this information-acquisition gap, invisible to outcome-only evaluation, directly measurable.
>
---
#### [new 263] Pramana: Fine-Tuning Large Language Models for Epistemic Reasoning through Navya-Nyaya
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于AI推理任务，旨在解决大模型缺乏可靠论证的问题。通过微调Llama模型，引入Navya-Nyaya逻辑框架提升模型的理性推理能力。**

- **链接: [https://arxiv.org/pdf/2604.04937](https://arxiv.org/pdf/2604.04937)**

> **作者:** Sharath Sathish
>
> **备注:** 52 pages + appendices, comprehensive treatment of Navya-Nyaya computational formalization
>
> **摘要:** Large language models produce fluent text but struggle with systematic reasoning, often hallucinating confident but unfounded claims. When Apple researchers added irrelevant context to mathematical problems, LLM performance degraded by 65% Apple Machine Learning Research, exposing brittle pattern-matching beneath apparent reasoning. This epistemic gap, the inability to ground claims in traceable evidence, limits AI reliability in domains requiring justification. We introduce Pramana, a novel approach that teaches LLMs explicit epistemological methodology by fine-tuning on Navya-Nyaya logic, a 2,500-year-old Indian reasoning framework. Unlike generic chain-of-thought prompting, Navya-Nyaya enforces structured 6-phase reasoning: SAMSHAYA (doubt analysis), PRAMANA (evidence source identification), PANCHA AVAYAVA (5-member syllogism with universal rules), TARKA (counterfactual verification), HETVABHASA (fallacy detection), and NIRNAYA (ascertainment distinguishing knowledge from hypothesis). This integration of logic and epistemology provides cognitive scaffolding absent from standard reasoning approaches. We fine-tune Llama 3.2-3B and DeepSeek-R1-Distill-Llama-8B on 55 Nyaya-structured logical problems (constraint satisfaction, Boolean SAT, multi-step deduction). Stage 1 achieves 100% semantic correctness on held-out evaluation despite only 40% strict format adherence revealing that models internalize reasoning content even when structural enforcement is imperfect. Ablation studies show format prompting and temperature critically affect performance, with optimal configurations differing by stage. We release all models, datasets, and training infrastructure on Hugging Face to enable further research on epistemic frameworks for AI reasoning.
>
---
#### [new 264] The Image Reconstruction Game: Drawing Common Ground Through Iterative Multimodal Dialogue
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出图像重建游戏任务，通过视觉语言模型与图像生成器的多轮对话优化图像重建。研究探讨了描述器与生成器对重建质量的影响，以及自动评估与人类偏好的一致性问题。**

- **链接: [https://arxiv.org/pdf/2606.01901](https://arxiv.org/pdf/2606.01901)**

> **作者:** Sherzod Hakimov; Mattia D'Agostini; Ivan Samodelkin; David Schlangen
>
> **摘要:** We introduce the Image Reconstruction Game, a fully automated benchmark in which a vision-language model issues corrective instructions to an image generator across multiple turns, making accumulated common ground directly observable as a rendered image. Benchmarking two Describer models crossed with two Generator models across seven image categories, we find that the describer is the dominant factor in reconstruction quality, while the generator determines whether iterative refinement helps or hurts. Mathematical and geometric images pose the greatest challenge. The describer's token budget strongly affects convergence: shorter budgets yield sparser first renderings with more room for visible improvement, while longer budgets raise absolute quality but leave less to fix. Stronger describers use a richer correction vocabulary spanning spatial, numeric, and structural categories, while weaker describers concentrate on surface properties and tend to stop after a few turns. Human validation shows that the best automated judge reaches only slight-to-fair agreement with human preferences, and automated scores require human recalibration to be used reliably.
>
---
#### [new 265] The Shape of Wisdom: Decision Trajectories in Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究语言模型决策过程，分析答案得分变化轨迹，区分正确但不稳定与稳定正确的案例，探索影响决策的因素。属于模型理解任务，解决答案稳定性与正确性关系问题。**

- **链接: [https://arxiv.org/pdf/2606.01202](https://arxiv.org/pdf/2606.01202)**

> **作者:** Shailesh Rana
>
> **备注:** 6 pages, 5 figures. Code and derived artifacts: this https URL
>
> **摘要:** Language models do not simply choose an answer at the output layer. In a 9,000-trajectory MMLU study across Qwen2.5-7B-Instruct, Llama-3.1-8B-Instruct, and Mistral-7B-Instruct-v0.3, the score of the answer moves across depth in structured ways. We describe each trajectory with three quantities: the current answer margin, the next-layer change in that margin, and the distance from a decision flip. The main empirical picture is that correctness and stability are different: the largest group is unstable-correct, not stable-correct. A traced subset then asks what moves the margin. In stable-correct cases, the average attention scalar points in the correct direction, while the average MLP scalar does not; span deletion shows that removing answer-supporting text hurts the margin and removing distractor-like text helps it. The result is not a full circuit explanation. It is a reproducible way to see which answers are settled, which remain fragile, and which measured sources move them.
>
---
#### [new 266] Quality-Diversity Evolution for Discovering Diverse Vulnerabilities in LLM Safety
- **分类: cs.CR; cs.CL; cs.ET; cs.LG; cs.NE**

- **简介: 该论文属于LLM安全测试任务，解决传统方法覆盖不足的问题，通过质量-多样性进化框架发现多样漏洞，生成可解释的攻击策略。**

- **链接: [https://arxiv.org/pdf/2606.00801](https://arxiv.org/pdf/2606.00801)**

> **作者:** Subhadip Mitra
>
> **备注:** 9 pages, 6 figures. Accepted at the ICLR 2026 Workshop on Agents in the Wild (AIWILD)
>
> **摘要:** Current approaches to LLM adversarial testing suffer from coverage gaps: manual red-teaming does not scale, LLM-as-attacker methods exhibit mode collapse, and gradient-based approaches produce uninterpretable gibberish. We introduce a quality-diversity evolutionary framework that operates at the semantic level, evolving interpretable attack strategies rather than token sequences. Using MAP-Elites, we maintain a diverse archive of attacks across behavioral dimensions (strategy type, encoding method, prompt length). In experiments across GPT-4o-mini, Claude 3.5 Sonnet, Gemini 2.0 Flash, and an open-weight coding model (Devstral-small-2), we discover distinct vulnerability profiles: GPT-4o-mini is vulnerable to hypothetical and multi-turn framing combined with ROT13 encoding (fitness 0.8), Gemini to direct attacks with ROT13 and multi-turn with Leetspeak (0.8), while Claude shows uniformly ambiguous responses across all strategies (max 0.4). The semantic representation produces interpretable attacks that reveal systematic, model-specific weaknesses, providing actionable insights for improving LLM safety and a reproducible baseline for evaluating future frontier models. Code and experiment artifacts are released at this https URL.
>
---
#### [new 267] COMAP: Co-Evolving World Models and Agent Policies for LLM Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出COMAP框架，解决语言代理在动态环境中适应性不足的问题。通过协同进化世界模型与策略，提升代理的决策能力。**

- **链接: [https://arxiv.org/pdf/2606.02372](https://arxiv.org/pdf/2606.02372)**

> **作者:** Youwei Liu; Jian Wang; Hanlin Wang; Wenjie Li
>
> **摘要:** Equipping language agents with world models enables them to anticipate environment dynamics and evaluate candidate actions before execution. However, existing textual world models are typically fixed after training, preventing them from adapting to the on-policy state-action distributions induced by an evolving agent. Meanwhile, agent-improvement methods often rely on external rewards or verifiers, limiting their applicability in realistic interactive environments. In this paper, we propose COMAP, a novel framework that co-evolves textual world models and agent policies through closed-loop interaction. At each decision step, the world model predicts future state feedback for candidate actions, and the agent performs future-aware reflection by estimating the reliability of this feedback and refining its action accordingly. The resulting on-policy trajectories are then used to update the world model via self-distillation, allowing it to better match the agent's evolving interaction distribution. Across embodied task planning, Web navigation, and tool-use benchmarks, COMAP consistently outperforms competitive baselines, e.g., +16.75% relative improvement with Qwen3-4B. Further analyses show that the co-evolutionary loop improves the world model's prediction accuracy over time and leads to more effective long-horizon decision-making. Our code is available at: this https URL.
>
---
## 更新

#### [replaced 001] LK Losses: Direct Acceptance Rate Optimization for Speculative Decoding
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言模型推理加速任务，解决草案模型接受率低的问题。通过提出LK损失函数直接优化接受率，提升解码速度。**

- **链接: [https://arxiv.org/pdf/2602.23881](https://arxiv.org/pdf/2602.23881)**

> **作者:** Alexander Samarin; Sergei Krutikov; Anton Shevtsov; Sergei Skvortsov; Filipp Fisin; Alexander Golubev
>
> **备注:** ICML 2026
>
> **摘要:** Speculative decoding accelerates autoregressive large language model (LLM) inference by using a lightweight draft model to propose candidate tokens that are then verified in parallel by the target model. The speedup is significantly determined by the acceptance rate, yet standard training minimizes Kullback-Leibler (KL) divergence as a proxy objective. While KL divergence and acceptance rate share the same global optimum, small draft models, having limited capacity, typically converge to suboptimal solutions where minimizing KL does not guarantee maximizing acceptance rate. To address this issue, we propose LK losses, special training objectives that directly target acceptance rate. Comprehensive experiments across four draft architectures and six target models, ranging from 8B to 685B parameters, demonstrate consistent improvements in acceptance metrics across all configurations compared to the standard KL-based training. We evaluate our approach on general, coding and math domains and report gains of up to 8-10% in average acceptance length. LK losses are easy to implement, introduce no computational overhead and can be directly integrated into any existing speculator training framework, making them a compelling alternative to the existing draft training objectives.
>
---
#### [replaced 002] OncoReason: Structuring Clinical Reasoning in LLMs for Robust and Interpretable Survival Prediction
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于医疗预测任务，旨在提升癌症治疗结果预测的准确性与可解释性。通过结构化推理框架，结合多任务学习和强化学习，增强模型的临床推理能力。**

- **链接: [https://arxiv.org/pdf/2510.17532](https://arxiv.org/pdf/2510.17532)**

> **作者:** Raghu Vamshi Hemadri; Geetha Krishna Guruju; Kristi Topollai; Anna Ewa Choromanska
>
> **备注:** This manuscript is withdrawn to allow careful review and correction of bibliographic issues identified after submission, including references that could not be adequately verified. These matters should be resolved before further circulation
>
> **摘要:** Predicting cancer treatment outcomes requires models that are both accurate and interpretable, particularly in the presence of heterogeneous clinical data. While large language models (LLMs) have shown strong performance in biomedical NLP, they often lack structured reasoning capabilities critical for high-stakes decision support. We present a unified, multi-task learning framework that aligns autoregressive LLMs with clinical reasoning for outcome prediction on the MSK-CHORD dataset. Our models are trained to jointly perform binary survival classification, continuous survival time regression, and natural language rationale generation. We evaluate three alignment strategies: (1) standard supervised fine-tuning (SFT), (2) SFT with Chain-of-Thought (CoT) prompting to elicit step-by-step reasoning, and (3) Group Relative Policy Optimization (GRPO), a reinforcement learning method that aligns model outputs to expert-derived reasoning trajectories. Experiments with LLaMa3-8B and Med42-8B backbones demonstrate that CoT prompting improves F1 by +6.0 and reduces MAE by 12%, while GRPO achieves state-of-the-art interpretability and predictive performance across BLEU, ROUGE, and BERTScore. We further show that existing biomedical LLMs often fail to produce valid reasoning traces due to architectural constraints. Our findings underscore the importance of reasoning-aware alignment in multi-task clinical modeling and set a new benchmark for interpretable, trustworthy LLMs in precision oncology.
>
---
#### [replaced 003] Empathy Applicability Modeling for General Health Queries
- **分类: cs.CL**

- **简介: 该论文属于医疗情感分析任务，解决LLMs缺乏临床共情的问题。提出EAF框架，通过多源标注数据预测患者查询的共情适用性，提升医疗沟通的同理心支持。**

- **链接: [https://arxiv.org/pdf/2601.09696](https://arxiv.org/pdf/2601.09696)**

> **作者:** Shan Randhawa; Agha Ali Raza; Kentaro Toyama; Julie Hui; Mustafa Naseem
>
> **备注:** Accepted at Findings of ACL 2026
>
> **摘要:** LLMs are increasingly being integrated into clinical workflows, yet they often lack clinical empathy, an essential aspect of effective doctor-patient communication. Existing NLP frameworks focus on reactively labeling empathy in doctors' responses but offer limited support for anticipatory modeling of empathy needs, especially in general health queries. We introduce the Empathy Applicability Framework (EAF), a theory-driven approach that classifies patient queries in terms of the applicability of emotional reactions and interpretations, based on clinical, contextual, and linguistic cues. We release a benchmark of real patient queries, dual-annotated by human annotators and GPT-4o. In the subset with human consensus, we also observe substantial human-GPT alignment. To validate EAF, we train classifiers on human-labeled and GPT-only annotations to predict empathy applicability, achieving strong performance and outperforming the heuristic and zero-shot LLM baselines. Error analysis highlights persistent challenges: implicit distress, clinical-severity ambiguity, and contextual hardship, underscoring the need for multi-annotator modeling, clinician-in-the-loop calibration, and culturally diverse annotation. EAF provides a framework for identifying empathy needs before response generation, establishes a benchmark for anticipatory empathy modeling, and enables supporting empathetic communication in asynchronous healthcare.
>
---
#### [replaced 004] Qwen-VLA: Unifying Vision-Language-Action Modeling across Tasks, Environments, and Robot Embodiments
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文提出Qwen-VLA，一个统一的视觉-语言-动作模型，解决多任务、多环境、多机器人形态的具身智能问题，通过联合预训练和提示条件实现跨任务迁移。**

- **链接: [https://arxiv.org/pdf/2605.30280](https://arxiv.org/pdf/2605.30280)**

> **作者:** Qiuyue Wang; Mingsheng Li; Jian Guan; Jinhui Ye; Sicheng Xie; Yitao Liu; Junhao Chen; Zhixuan Liang; Jie Zhang; Xintong Hu; Xuhong Huang; Pei Lin; Junyang Lin; Dayiheng Liu; Shuai Bai; Jingren Zhou; Jiazhao Zhang; Haoqi Yuan; Gengze Zhou; Hang Yin; Ye Wang; Yiyang Huang; Zixing Lei; Wujian Peng; Delin Chen; Yingming Zheng; Jingyang Fan; Xianwei Zhuang; Xin Zhou; Haoyang Li; Anzhe Chen; Tong Zhang; Xuejing Liu; Yuchong Sun; Ruizhe Chen; Zhaohai Li; Chenxu Lü; Zhibo Yang; Tao Yu; Xionghui Chen
>
> **备注:** 34 pages
>
> **摘要:** Embodied intelligence is often studied through specialized models for individual tasks such as manipulation or navigation, resulting in fragmented capabilities and limited generalization across tasks, environments, and robot embodiments. In this work, we study whether heterogeneous embodied decision-making problems can be unified within a single vision-language-action model. We present Qwen-VLA, a unified embodied foundation model that extends Qwen's vision-language modeling stack from perception, understanding, and reasoning to continuous action and trajectory generation through a DiT-based action decoder. Qwen-VLA is trained with a large-scale joint pretraining recipe over diverse data sources, including robotics manipulation trajectories, human egocentric demonstrations, synthetic simulation data, vision-and-language navigation data, trajectory-centric supervision, and auxiliary vision-language data. To support multiple robot platforms, we introduce embodiment-aware prompt conditioning, where robot-specific textual descriptions specify the current embodiment and control convention. We further cast manipulation, navigation, and trajectory prediction into a unified action-and-trajectory prediction framework, enabling transferable visual grounding, spatial reasoning, and continuous action generation across robot morphologies, task families, and environments. Experiments on manipulation, navigation, and trajectory-centric benchmarks show consistent multi-task performance and out-of-distribution generalization under variations in scene layout, background, lighting, object configuration, and robot embodiment. Qwen-VLA-Instruct achieves 97.9% on LIBERO, 73.7% on Simpler-WidowX, 86.1%/87.2% on RoboTwin-Easy/Hard, 69.0% OSR on R2R, 59.6% SR on RxR, 76.9% average OOD success in real-world ALOHA experiments, and 26.6% zero-shot success on DOMINO dynamic manipulation.
>
---
#### [replaced 005] ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型优化任务，解决长序列生成中KV缓存效率与性能的平衡问题。通过学习预测需淘汰的KV对，提升缓存利用率。**

- **链接: [https://arxiv.org/pdf/2602.03203](https://arxiv.org/pdf/2602.03203)**

> **作者:** Zican Dong; Peiyu Liu; Junyi Li; Zhipeng Chen; Han Peng; Shuo Wang; Wayne Xin Zhao
>
> **备注:** ICML 2026
>
> **摘要:** Recently, large language models (LLMs) have shown remarkable reasoning abilities by producing long reasoning traces. However, as the sequence length grows, the key-value (KV) cache expands linearly, incurring significant memory and computation costs. Existing KV cache eviction methods mitigate this issue by discarding less important KV pairs, but often fail to capture complex KV dependencies, resulting in performance degradation. To better balance efficiency and performance, we introduce ForesightKV, a training-based KV cache eviction framework that learns to predict which KV pairs to evict during long-text generations. We first design the Golden Eviction algorithm, which identifies the optimal eviction KV pairs at each step using future attention scores. These traces and the scores at each step are then distilled via supervised training with a Pairwise Ranking Loss. Furthermore, we formulate cache eviction as a Markov Decision Process and apply the GRPO algorithm to mitigate the significant language modeling loss increase on low-entropy tokens. Experiments on AIME2024 and AIME2025 benchmarks of three reasoning models demonstrate that ForesightKV consistently outperforms prior methods under only half the cache budget, while benefiting synergistically from both supervised and reinforcement learning approaches. Code is available at this https URL.
>
---
#### [replaced 006] LC-ERD: Mining Latent Logic for Self-Evolving Reasoning via Consistency-Regulated Reward Decomposition
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于大语言模型推理优化任务，解决过程数据稀缺与监督信号不足的问题。提出LC-ERD框架，通过逻辑一致性约束提升自对齐效果。**

- **链接: [https://arxiv.org/pdf/2605.24005](https://arxiv.org/pdf/2605.24005)**

> **作者:** Yanyu Chen; Jiyue Jiang; Dianzhi Yu; Zheng Wu; Jiahong Liu; Jiaming Han; Xiao Guo; Jinhu Qi; Yu Li; Yifei Zhang; Irwin King
>
> **备注:** Accepted in SIGKDD 2026 Research Track
>
> **摘要:** The evolution of Large Language Model (LLM) reasoning is bottlenecked by the scarcity of high-quality process data. While self-alignment via endogenous rewards offers a solution, mining valid supervision faces three challenges: (1) Label Noise via Mimetic Bias, where rewards prioritize statistical likelihood over logical truth, creating a "correctness illusion" that masks compounding errors; (2) Coarse-Grained Supervision, where sparse global outcomes (e.g., in GRPO) fail to provide granular guidance, treating reasoning chains as monolithic; and (3) Distributional Collapse, where signals fail to generalize without amplifying pre-training biases. To address these, we introduce LC-ERD (Logic-Consistent Endogenous Reward Decomposition), a framework framing self-alignment as latent structure mining. We derive a Variational Logic Potential by aggregating consensus from the model's Latent Logic Expertise (LLE) to denoise the reasoning manifold, and introduce a Multi-Agent Value Decomposition protocol based on the IGM principle to quantify individual step utility. Experiments show LC-ERD delivers a robust self-evolution path, uncovering trade-offs between logic consistency and accuracy while identifying high-value reasoning patterns missed by standard rewards. Our code is available at this https URL.
>
---
#### [replaced 007] Bridging the Knowledge-Prediction Gap in LLMs on Multiple-Choice Questions
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，针对大模型在选择题上表现不佳的问题，通过分析隐藏表示，提出KAPPA方法缩小知识与预测之间的差距。**

- **链接: [https://arxiv.org/pdf/2509.23782](https://arxiv.org/pdf/2509.23782)**

> **作者:** Yoonah Park; Haesung Pyun; Yohan Jo
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** While large language models (LLMs) perform strongly on diverse tasks, their trustworthiness is limited by erratic behavior that is unfaithful to their internal knowledge. In particular, LLMs often fail on multiple-choice questions (MCQs) even if they encode correct answers in their hidden representations, revealing a misalignment between internal knowledge and output behavior. We investigate and mitigate this knowledge-prediction gap on MCQs through a three-step analysis of hidden representations. First, we quantify the prevalence and magnitude of the gap across models and datasets. Second, we provide a geometric interpretation by identifying distinct knowledge and prediction subspaces in the residual stream. Third, we introduce KAPPA, a lightweight inference-time intervention that aligns the two subspaces within the residual stream to reduce the knowledge-prediction gap. Our results provide a geometric and interpretable explanation of the knowledge-prediction gap in LLMs. Furthermore, KAPPA effectively reduces the gap across diverse MCQ benchmarks and models, and generalizes to free-form settings.
>
---
#### [replaced 008] Cornerstones or Stumbling Blocks? Deciphering the Rock Tokens in On-Policy Distillation
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究强化学习中的模型蒸馏任务，解决学生-教师模型不匹配问题。通过分析高损失标记（Rock Tokens），发现其对性能影响小却消耗大量优化资源，提出优化策略提升蒸馏效率。**

- **链接: [https://arxiv.org/pdf/2605.09253](https://arxiv.org/pdf/2605.09253)**

> **作者:** Yuxuan Jiang; Runchao Li; Shubhashis Roy Dipta; Dawei Li; Zhao Yang
>
> **摘要:** While recent work in Reinforcement Learning with Verifiable Rewards (RLVR) has shown that a small subset of critical tokens disproportionately drives reasoning gains, an analogous token-level understanding of On-Policy Distillation (OPD) remains largely unexplored. In this work, we investigate high-loss tokens, a token type that--as the most direct signal of student-teacher mismatch under OPD's per-token KL objective--should progressively diminish as training converges according to existing studies; however, our empirical analysis shows otherwise. Even after OPD training reaches apparent saturation, a substantial subset of tokens continues to exhibit persistently high loss; these tokens, which we term Rock Tokens, can account for up to 18\% of the tokens in generated outputs. Our investigation reveals two startling paradoxes. First, despite their high occurrence frequency providing a disproportionately large share of total gradient norms, Rock Tokens themselves remain stagnant throughout training, resisting teacher-driven corrections. Second, through causal intervention, we find that these tokens provide negligible functional contribution to the model's actual reasoning performance. These findings suggest that a vast amount of optimization bandwidth is spent on structural and discourse residuals that the student model cannot or need not internalize. By deconstructing these dynamics, we demonstrate that strategically bypassing these ``stumbling blocks'' can significantly streamline the alignment process, challenging the necessity of uniform token weighting and offering a more efficient paradigm for large-scale model distillation.
>
---
#### [replaced 009] PaperVoyager : Building Interactive Web with Visual Language Models
- **分类: cs.CL**

- **简介: 该论文提出PaperVoyager，将科研论文转化为可交互的网页系统，解决静态文档无法体现动态机制的问题。**

- **链接: [https://arxiv.org/pdf/2603.22999](https://arxiv.org/pdf/2603.22999)**

> **作者:** Dasen Dai; Biao Wu; Meng Fang; Wenhao Wang
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Recent advances in visual language models have enabled autonomous agents for complex reasoning, tool use, and document understanding. However, existing document agents mainly transform papers into static artifacts such as summaries, webpages, or slides, which are insufficient for technical papers involving dynamic mechanisms and state transitions. In this work, we propose a Paper-to-Interactive-System Agent that converts research papers into executable interactive web systems. Given a PDF paper, the agent performs end-to-end processing without human intervention, including paper understanding, system modeling, and interactive webpage synthesis, enabling users to manipulate inputs and observe dynamic behaviors. To evaluate this task, we introduce a benchmark of 19 research papers paired with expert-built interactive systems as ground truth. We further propose PaperVoyager, a structured generation framework that explicitly models mechanisms and interaction logic during synthesis. Experiments show that PaperVoyager significantly improves the quality of generated interactive systems, offering a new paradigm for interactive scientific paper understanding.
>
---
#### [replaced 010] GateKD: Confidence-Gated Closed-Loop Distillation for Robust Reasoning
- **分类: cs.CL**

- **简介: 该论文属于知识蒸馏任务，旨在解决大模型推理能力迁移至小模型时的噪声和错误传播问题。提出GateKD框架，通过动态教师机制提升蒸馏效果。**

- **链接: [https://arxiv.org/pdf/2605.13136](https://arxiv.org/pdf/2605.13136)**

> **作者:** Kasidit Sermsri; Teerapong Panboonyuen
>
> **备注:** 16 pages
>
> **摘要:** Distilling multi-step reasoning abilities from large language models (LLMs) into compact student models remains challenging due to noisy rationales, hallucinated supervision, and static teacher-student interactions. Existing reasoning distillation methods, including mentor-based approaches, predominantly operate in an open-loop manner, implicitly assuming uniform teacher reliability and consequently propagating erroneous intermediate reasoning. We propose GateKD, a confidence-gated closed-loop distillation framework that enables robust reasoning transfer by treating the teacher as a dynamic gatekeeper rather than a static oracle. GateKD introduces three complementary mechanisms: (i) confidence-gated soft supervision that selectively distills reliable predictive signals, (ii) gated hidden-state evolution that aligns intermediate representations only when teacher confidence is high, and (iii) reliability-filtered attention distillation that preserves stable reasoning structures while suppressing noisy patterns. These components jointly form a closed feedback loop in which teacher confidence continuously modulates the distillation process, reducing hallucination transfer and stabilizing student reasoning. Extensive experiments across commonsense, logical, and symbolic reasoning benchmarks, using T5 and Flan-T5 backbones of varying sizes, demonstrate that GateKD consistently outperforms strong open-loop distillation baselines. Notably, GateKD yields substantial gains in logical and symbolic reasoning, remains robust under low-resource distillation settings, and shows clear performance degradation when any gating component is removed. Our results highlight that confidence-gated closed-loop supervision is critical for building reliable and scalable small reasoning models.
>
---
#### [replaced 011] Structured Semantic Information Helps Retrieve Better Examples for In-Context Learning Applied to Few-Shot Relation Extraction
- **分类: cs.CL**

- **简介: 该论文属于关系抽取任务，解决少样本学习中示例不足的问题。通过结构化语义选择示例，提升模型性能，实现更优的关系抽取效果。**

- **链接: [https://arxiv.org/pdf/2601.20803](https://arxiv.org/pdf/2601.20803)**

> **作者:** Aunabil Chakma; Mihai Surdeanu; Eduardo Blanco
>
> **摘要:** This paper presents several strategies to automatically obtain additional examples for in-context learning, effectively transforming relation extraction from a 1-shot to a few-shot setting. Specifically, we introduce a novel strategy for example selection, in which new examples are selected based on the similarity of their underlying syntactic-semantic structure to the provided 1-shot example. We show that our strategy results in complementary word choices and sentence structures compared to LLM-generated examples. When both strategies are combined, the resulting hybrid system achieves a more holistic picture of the relations of interest than either method alone. Our framework transfers well across datasets (FS-TACRED and FS-FewRel) and LLM families (Qwen and Gemma). Overall, our hybrid system consistently outperforms alternative strategies achieving state-of-the-art performance on FS-TACRED and strong gains on a customized FewRel subset.
>
---
#### [replaced 012] How AI Fails: An Interactive Pedagogical Tool for Demonstrating Dialectal Bias in Automated Toxicity Models
- **分类: cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于AI偏见研究任务，旨在解决自动化内容审核中的方言偏差问题。通过基准测试和交互工具，揭示并展示AAE文本在毒性模型中的不公平评分。**

- **链接: [https://arxiv.org/pdf/2511.06676](https://arxiv.org/pdf/2511.06676)**

> **作者:** Subhojit Ghimire
>
> **备注:** 9 pages, 5 figures, 4 tables, 14 references. Preliminary abstract presented at the International Conference on Envisioning the Himalayan Future: Pathways to Sustainability and Development (PUiCON 2026) p. 105; abstract available online at: this https URL
>
> **摘要:** Now that AI-driven moderation has become pervasive in everyday life, we often hear claims that "the AI is biased". While this is often said jokingly, the light-hearted remark reflects a deeper concern. How can we be certain that an online post flagged as "inappropriate" was not simply the victim of a biased algorithm? This paper investigates this problem using a dual approach. First, I conduct a quantitative benchmark of a widely used toxicity model (unitary/toxic-bert) to measure performance disparity between text in African-American English (AAE) and Standard American English (SAE). The benchmark reveals a clear, systematic bias: on average, the model scores AAE text as 1.8 times more toxic and 8.8 times higher for "identity hate". Second, I introduce an interactive pedagogical tool that makes these abstract biases tangible. The tool's core mechanic, a user-controlled "sensitivity threshold," demonstrates that the biased score itself is not the only harm; instead, the more-concerning harm is the human-set, seemingly neutral policy that ultimately operationalises discrimination. This work provides both statistical evidence of disparate impact and a public-facing tool designed to foster critical AI literacy.
>
---
#### [replaced 013] SmartThinker: Progressive Chain-of-Thought Length Calibration for Efficient Large Language Model Reasoning
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于大语言模型推理任务，旨在解决长链式思维（CoT）冗余和过拟合问题。提出SmartThinker方法，通过动态调整CoT长度和奖励系数，实现更高效准确的推理。**

- **链接: [https://arxiv.org/pdf/2603.08000](https://arxiv.org/pdf/2603.08000)**

> **作者:** Chenzhi Hu; Qinzhe Hu; Yuhang Xu; Junyi Chen; Ruijie Wang; Shengzhong Liu; Jianxin Li; Fan Wu; Guihai Chen
>
> **备注:** Accepted by ICML 2026, 18 pages, 13 figures
>
> **摘要:** Large reasoning models (LRMs) like OpenAI o1 and DeepSeek-R1 achieve high accuracy on complex tasks by adopting long chain-of-thought (CoT) reasoning paths. However, the inherent verbosity of these processes frequently results in redundancy and overthinking. To address this issue, existing works leverage Group Relative Policy Optimization (GRPO) to reduce LRM output length, but their static length reward design cannot dynamically adapt according to the relative problem difficulty and response length distribution, causing over-compression and compromised accuracy. Therefore, we propose SmartThinker, a novel GRPO-based efficient reasoning method with progressive CoT length calibration. SmartThinker makes a two-fold contribution: First, it dynamically estimates the optimal length with peak accuracy during training and guides overlong responses toward it to reduce response length while sustaining accuracy. Second, it dynamically modulates the length reward coefficient to avoid the unwarranted penalization of correct reasoning paths. Extensive experiment results show that SmartThinker achieves up to 52.5% average length compression with improved accuracy, and achieves up to 16.6% accuracy improvement on challenging benchmarks like AIME25. The source code can be found at this https URL.
>
---
#### [replaced 014] GottBERT: a pure German Language Model
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出首个纯德语RoBERTa模型GottBERT，解决德语NLP任务中单语言模型性能优化问题，通过预训练与评估验证其有效性。**

- **链接: [https://arxiv.org/pdf/2012.02110](https://arxiv.org/pdf/2012.02110)**

> **作者:** Raphael Scheible; Johann Frei; Fabian Thomczyk; Henry He; Patric Tippmann; Jochen Knaus; Victor Jaravine; Frank Kramer; Martin Boeker
>
> **摘要:** Pre-trained language models have significantly advanced natural language processing (NLP), especially with the introduction of BERT and its optimized version, RoBERTa. While initial research focused on English, single-language models can be advantageous compared to multilingual ones in terms of pre-training effort, overall resource efficiency or downstream task performance. Despite the growing popularity of prompt-based LLMs, more compute-efficient BERT-like models remain highly relevant. In this work, we present the first German single-language RoBERTa model, GottBERT, pre-trained exclusively on the German portion of the OSCAR dataset. Additionally, we investigated the impact of filtering the OSCAR corpus. GottBERT was pre-trained using fairseq and standard hyperparameters. We evaluated its performance on two Named Entity Recognition (NER) tasks (Conll 2003 and GermEval 2014) and three text classification tasks (GermEval 2018 fine and coarse, and 10kGNAD) against existing German BERT models and two multilingual models. Performance was measured using the $F_{1}$ score and accuracy. The GottBERT base and large models showed competitive performance, with GottBERT leading among the base models in 4 of 6 tasks. Contrary to our expectation, the applied filtering did not significantly affect the results. To support the German NLP research community, we are releasing the GottBERT models under the MIT license.
>
---
#### [replaced 015] Algorithmic Fragility and Persona Bias in LLM-Generated Autistic Communication
- **分类: cs.CL; cs.AI**

- **简介: 论文研究LLM在生成自闭症沟通内容时的算法脆弱性和人格偏见，探讨安全对齐带来的表征偏差。通过双人格重写任务，发现模型生成结果存在显著差异和系统性错误。**

- **链接: [https://arxiv.org/pdf/2605.26397](https://arxiv.org/pdf/2605.26397)**

> **作者:** Naba Rizvi; Mohammed Rizvi; Harper Strickland; Saleha Ahmedi; Nedjma Ousidhoum
>
> **备注:** main paper: 9 pages; total: 19 pages; 2 figures; 5 tables
>
> **摘要:** Safety alignment reduces explicitly harmful outputs but inadvertently encodes a sanitized, neuronormative representation of marginalized communication. We investigate this encoding using a dual-persona rewrite paradigm, prompting ten large language models (LLMs) to rewrite naturally occurring autistic discourse from either an autistic or neurotypical persona. We uncover autistic-persona rewrites diverge significantly more in lexical form and affective register than neurotypical rewrites, despite equivalent semantic similarity. Furthermore, most models collapse cross-persona generations into near-identical outputs. To uncover the mechanisms behind this generative breakdown, we introduce a multi-agent qualitative analysis framework. Our results reveal systemic output erasure, stereotyped hallucination, and task-evasive meta-commentary are pervasive failure modes for this task that cluster by alignment strategy rather than parameter scale. Finally, our targeted comparison with autistic human annotators demonstrates that community-insider knowledge produces systematic label reversals relative to LLM classifications. Our findings indicate that current alignment training causes persona-specific generative breakdown visible only through qualitative analysis, confirming a deep representational gap that prompt engineering cannot resolve.
>
---
#### [replaced 016] Agent-R1: A Unified and Modular Framework for Agentic Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出Agent-R1框架，解决agentic RL中的轨迹表示与优化问题，通过步骤级交互和灵活上下文管理，支持多种优化策略。**

- **链接: [https://arxiv.org/pdf/2511.14460](https://arxiv.org/pdf/2511.14460)**

> **作者:** Mingyue Cheng; Shuo Yu; Daoyu Wang; Qingchuan Li; Xiaoyu Tao; Jie Ouyang; Yucong Luo; Yitong Zhou; Qi Liu; Enhong Chen
>
> **备注:** This paper serves as the technical report of the Agent-R1 project
>
> **摘要:** Large language models (LLMs) have rapidly evolved from single-turn text generators into the foundation of increasingly capable agents. As these agents take on more complex reasoning, decision making, tool use, and long-horizon tasks, reinforcement learning (RL) is becoming increasingly important for shaping their behavior. This shift is especially visible in agentic RL, where models must interact with tools and environments across multiple rounds rather than produce a single standalone response. In this regime, the usual view of a trajectory as one ever-growing token sequence becomes increasingly inadequate: it makes context evolution rigid and creates representation mismatches between rollout and training. This paper presents Agent-R1, a unified and modular framework for agentic RL built around step-level trajectory representation, flexible context management, and layered interfaces for workflows, environments and optimization. The key idea is to treat each interaction step as the basic reinforcement-learning transition, while keeping the optimization layer flexible: once the interaction is modeled at the step level, the framework can support token-level credit assignment, step-level credit assignment, or other compatible designs. These design choices make the framework compatible with a range of optimization strategies rather than tying it to a single algorithm. Together, these components provide a principled, extensible, and reusable substrate for agentic RL.
>
---
#### [replaced 017] What Do LLMs Know About Alzheimer's Disease? Multi-loss Fine-Tuning and Probing for AD Detection
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于阿尔茨海默病检测任务，旨在利用大语言模型进行文本分析以实现早期诊断。研究通过微调和探针分析，评估模型在不同数据集上的表现及特征编码方式。**

- **链接: [https://arxiv.org/pdf/2602.11177](https://arxiv.org/pdf/2602.11177)**

> **作者:** Lei Jiang; Yue Zhou; Natalie Parde
>
> **摘要:** Reliable early detection of Alzheimer's disease (AD) is challenging, particularly due to the limited availability of labeled data. While large language models (LLMs) have shown strong transfer capabilities across do mains, adapting them to the AD domain through supervised fine-tuning remains largely unexplored. In this work, we empirically evaluate various model architectures across three heterogeneous transcript corpora (Pitt, CCC, ADRC) to investigate their effectiveness for text-based AD detection and analyze how task-relevant information is encoded within their internal representations. To the best of our knowledge, our fine-tuned BERT and T5 models establish a new state-of-the-art on the Pitt and CCC datasets, while achieving strong performance on ADRC. In parallel, the decoder-only Llama-1B achieves highly competitive results comparable to BERT and T5 across all three corpora, highlighting its effectiveness for AD detection. We further conduct a comprehensive evaluation of the Llama-1B backbone, analyzing cross-corpus transferability, optimal input chunk-size granularity, and the impact of clinical transcript markers. Also, we use linear probing to empirically show that fine-tuning shifts the representations of individual tokens, both linguistic markers and content words, in ways that reflect AD-related signal.
>
---
#### [replaced 018] HumorRank: A Tournament-Based Leaderboard for Evaluating Humor Generation in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出HumorRank，用于评估大语言模型的幽默生成能力。解决幽默评价主观、难以比较的问题，通过比赛式框架进行 pairwise 判断，实现模型排名。**

- **链接: [https://arxiv.org/pdf/2604.19786](https://arxiv.org/pdf/2604.19786)**

> **作者:** Edward Ajayi; Prasenjit Mitra
>
> **摘要:** Humor remains difficult to evaluate in large language models (LLMs) because what makes a response funny is subjective, comparative, and shaped by interacting comedic mechanisms rather than a single scalar property. Existing humor evaluation protocols therefore tend to produce isolated scores or task-specific judgments that are difficult to compare across models. We introduce HumorRank, a tournament-based framework for ranking textual humor generation through theory-grounded pairwise preference judgments. Across SemEval-2026 MWAHAHA and Humor Transfer Bench, HumorRank evaluates nine proprietary, open-weight, and specialized models using LLM-based comparative judgments informed by the General Theory of Verbal Humor (GTVH), with tournament aggregation yielding global rankings via Bradley-Terry estimation. The resulting rankings are cross-judge stable: independent Llama and Qwen LLM judges achieve Kendall {\tau} = 0.889 on both benchmarks. The leaderboard reveals clear model stratification, showing that strong humor generation depends not only on scale but on mastery of comedic mechanisms such as incongruity, conciseness, escalation, and absurdity. HumorRank provides a scalable and interpretable methodology for benchmarking LLM-generated humor without relying solely on isolated automatic metrics or limited human evaluation.
>
---
#### [replaced 019] REALISTA: Realistic Latent Adversarial Attacks that Elicit LLM Hallucinations
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决LLM hallucinations问题。通过构建REALISTA框架，生成语义一致的对抗性提示，提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2605.12813](https://arxiv.org/pdf/2605.12813)**

> **作者:** Buyun Liang; Jinqi Luo; Liangzu Peng; Kwan Ho Ryan Chan; Darshan Thaker; Kaleab A. Kinfu; Fengrui Tian; Hamed Hassani; René Vidal
>
> **备注:** Accepted at ICML 2026. Code is available at this https URL
>
> **摘要:** Large language models (LLMs) achieve strong performance across many tasks but remain vulnerable to hallucinations, making it important to systematically evaluate their reliability under realistic adversarial inputs. We formulate hallucination elicitation as a constrained optimization problem, where the goal is to find semantically coherent adversarial prompts that are equivalent to benign user prompts. Existing attack methods remain limited: discrete prompt-based attacks preserve semantic equivalence and coherence but search only over a limited set of prompt variations, while continuous latent-space attacks explore a richer space but often decode into prompts that are no longer valid rephrasings. To address these limitations, we propose REALISTA, a realistic latent-space attack framework. REALISTA constructs an input-dependent dictionary of valid editing directions, each corresponding to a semantically equivalent and coherent rephrasing, and optimizes continuous combinations of these directions in latent space. This design combines the optimization flexibility of continuous attacks with the semantic realism of discrete rephrasing-based attacks. Experiments demonstrate that REALISTA achieves superior or comparable performance to state-of-the-art realistic attacks on open-source LLMs and, crucially, succeeds in attacking large reasoning models under free-form response settings, where prior realistic attacks fail. Code is available at this https URL.
>
---
#### [replaced 020] Failure of contextual invariance in large language models
- **分类: cs.CL; cs.AI; cs.CY**

- **简介: 该论文属于自然语言处理任务，研究大语言模型在性别推断中的上下文不稳定性问题。通过实验发现模型输出受无关上下文影响，违反上下文不变性假设，对基准测试和应用有重要影响。**

- **链接: [https://arxiv.org/pdf/2603.23485](https://arxiv.org/pdf/2603.23485)**

> **作者:** Sagar Kumar; Ariel Flint; Luca Maria Aiello; Andrea Baronchelli
>
> **摘要:** Standard evaluation practices assume that large language model (LLM) outputs are stable when prompts are embedded in contextually equivalent discourses. Here, we test this assumption in the setting of gender inference. Using a controlled pronoun selection task, we introduce minimal, theoretically uninformative discourse context and find that this induces large, systematic shifts in model outputs. Correlations with cultural gender stereotypes, present in decontextualized settings, weaken or disappear once context is introduced, while theoretically irrelevant features, such as the gender of a pronoun for an unrelated referent, become the most informative predictors of model behavior. A Contextuality-by-Default analysis reveals that, in 19--52\% of cases across models, this dependence persists after accounting for all marginal effects of context on individual outputs and cannot be attributed to simple pronoun repetition. These findings show that LLM outputs violate contextual invariance even under near-identical syntactic formulations, with implications for bias benchmarking and deployment in high-stakes settings.
>
---
#### [replaced 021] Hypothesis Generation and Inductive Inference in Children and Language Models
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文研究儿童与语言模型在不确定环境中的归纳推理能力，通过模拟任务比较两者的行为差异，探讨其信息寻求策略和推理机制。**

- **链接: [https://arxiv.org/pdf/2605.24528](https://arxiv.org/pdf/2605.24528)**

> **作者:** Jeffrey Qin; Wasu Top Piriyakulkij; Zhuangfei Gao; Mia Radovanovic; Jessica Sommerville; Kevin Ellis; Marta Kryven
>
> **摘要:** Real world decision-making requires constructing mental models under uncertainty over evidence, over the underlying causal rules, and over the state of the world itself. Which computational principles underpin human inference under such conditions, and do LLM-based agents exhibit similar behavior given matching constraints? We address these questions using an inductive inference Box Task in which participants, human children and LLM-based agents, infer a latent cause through sequential interaction with an uncertain environment. We formalize this task as program induction with Bayesian particle-based inference, admitting two complementary interpretations: (1) as a constraint satisfaction process over hypotheses, and (2) as a program synthesis problem in which hypotheses are executable programs evaluated against evidence. Using the constraint-based formulation, we show that children's behavior is best explained by a combination of subjective evidence reliability and online hypothesis generation, accounting for both their evidence-seeking patterns and their dissociation between task completion and rule generalization. Using the program synthesis formulation, we treat LLM-based agents as model organisms: controllable systems that allow systematic manipulation of task conditions. Across backends, LLM-based agents replicate children's responses to changes in evidence reliability and observability, including discounting unreliable evidence, seeking to resolve partial information, and dissociating between task completion and causal generalization. At the same time, LLM-based agents tend to over-observe and over-comply with instructions relative to children. These results suggest that while children and LLM-based agents adapt similarly to environmental structure, their information-seeking behavior exhibits distinct underlying costs and inductive biases.
>
---
#### [replaced 022] Fundamental Limitation in Explaining AI
- **分类: cs.AI; cs.CL; cs.CY; cs.IT**

- **简介: 该论文属于AI可解释性研究，探讨为何无法同时满足AI解释的复杂环境、性能优良、可解释和完全忠实。工作是证明了一个四重困境，指出在多数应用中需放弃完全忠实性。**

- **链接: [https://arxiv.org/pdf/2605.24727](https://arxiv.org/pdf/2605.24727)**

> **作者:** Atsushi Suzuki; Jing Wang
>
> **备注:** minor modifications
>
> **摘要:** While large-scale models such as LLMs and diffusion models have achieved practical success, public institutions have emphasized the importance of explainability in AI. Existing methods for explaining AI, however, are not designed to provide completely faithful explanations of the behavior of large-scale AI systems. Although a completely faithful and interpretable explanation of the behavior of an AI system might be useful for AI governance, it has not been known whether providing such an explanation is theoretically possible. In this paper, we mathematically prove a fundamental quadrilemma in explaining AI, stating that AI and its explanation cannot satisfy the following four conditions simultaneously: 1) the complexity of the operation environment, 2) the goodness of the AI's performance, 3) the interpretability of the AI's explanation, and 4) the complete faithfulness of the AI's explanation. This quadrilemma suggests that, in most applications where we cannot change the environment or sacrifice good AI performance and an interpretable explanation, we should give up complete faithfulness of explanations and should instead aim to explain only the parts that are important for applications. As a consequence, the quadrilemma implies that AI governance should be designed on the premise that the faithfulness of AI explanations is always incomplete.
>
---
#### [replaced 023] Cross-Lingual Steering for Figurative Language Generation
- **分类: cs.CL**

- **简介: 该论文研究跨语言修辞生成任务，探讨多语言大模型中修辞行为的信号是否可跨语言复用。通过激活操控验证信号的可迁移性，发现修辞方向可跨语言有效，且其他语言的方向可能优于本语言方向。**

- **链接: [https://arxiv.org/pdf/2605.30443](https://arxiv.org/pdf/2605.30443)**

> **作者:** Linfeng Liu; Tiffany Zhan; Louie Hong Yao; Saptarshi Ghosh; Tianyu Jiang
>
> **备注:** 40 pages, 7 figures
>
> **摘要:** Multilingual large language models can generate figurative language, but whether the internal signals driving this behavior are language-specific or reusable across languages is unclear. Using activation steering as a probe, we estimate a direction for a figurative category from figurative--literal activation differences in one language and apply it during generation. Across five figurative categories, six languages, and four multilingual LLMs, these directions steer reliably within their own language, most robustly for metaphor and simile. More importantly, they transfer across languages: a direction learned in one increases the target behavior when applied to another, with German among the most receptive targets. Going further, directions assembled from other languages can match or even surpass a target language's own native direction, while removing this shared component weakens native steering. Together, these results provide direct evidence of a reusable but target-dependent cross-lingual signal for figurative generation.
>
---
#### [replaced 024] T1: Tool-integrated Verification for Test-time Compute Scaling in Small Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型验证任务，解决小模型在测试时计算扩展中的验证问题。提出T1框架，通过工具集成提升小模型的验证能力。**

- **链接: [https://arxiv.org/pdf/2504.04718](https://arxiv.org/pdf/2504.04718)**

> **作者:** Minki Kang; Jongwon Jeong; Jaewoong Cho
>
> **备注:** ICLR 2026
>
> **摘要:** Recent studies have demonstrated that test-time compute scaling effectively improves the performance of small language models (sLMs). However, prior research has mainly examined test-time compute scaling with an additional larger model as a verifier, leaving verification by sLMs underexplored. In this work, we investigate whether sLMs can reliably verify the output candidates under test-time scaling. We find that even with knowledge distillation from larger verifiers, sLMs struggle with verification tasks requiring memorization, such as numerical calculations and fact-checking. To address this limitation, we propose Tool-integrated verification (T1), a two-stage framework that first filters candidates with external tools and then uses an sLM for final verification, offloading memorization-heavy steps to tools such as a code interpreter. Within T1, we prove that offloading to external tools reduces the memorization burden on sLMs and improves test-time scaling performance. Experiments on the MATH benchmark demonstrate that, with T1, a Llama-3.2 1B model under test-time scaling outperforms the significantly larger Llama-3.1 8B model. Moreover, T1 improves the verification accuracy of both process reward models (PRMs) and critic models. Our findings highlight the potential of tool integration to substantially improve the verification abilities of sLMs.
>
---
#### [replaced 025] NILC: Discovering New Intents with LLM-assisted Clustering
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于新意图发现任务，旨在提升未标注数据中新旧意图的识别效果。通过引入LLM辅助的聚类框架NILC，优化嵌入与聚类中心，提高聚类准确性。**

- **链接: [https://arxiv.org/pdf/2511.05913](https://arxiv.org/pdf/2511.05913)**

> **作者:** Hongtao Wang; Renchi Yang; Wenqing Lin
>
> **摘要:** New intent discovery (NID) seeks to recognize both new and known intents from unlabeled user utterances, which finds prevalent use in practical dialogue systems. Existing works towards NID mainly adopt a cascaded architecture, wherein the first stage focuses on encoding the utterances into informative text embeddings beforehand, while the latter is to group similar embeddings into clusters (i.e., intents), typically by K-Means. However, such a cascaded pipeline fails to leverage the feedback from both steps for mutual refinement, and, meanwhile, the embedding-only clustering overlooks nuanced textual semantics, leading to suboptimal performance. To bridge this gap, this paper proposes NILC, a novel clustering framework specially catered for effective NID. Particularly, NILC follows an iterative workflow, in which clustering assignments are judiciously updated by carefully refining cluster centroids and text embeddings of uncertain utterances with the aid of large language models (LLMs). Specifically, NILC first taps into LLMs to create additional semantic centroids for clusters, thereby enriching the contextual semantics of the Euclidean centroids of embeddings. Moreover, LLMs are then harnessed to augment hard samples (ambiguous or terse utterances) identified from clusters via rewriting for subsequent cluster correction. Further, we inject supervision signals through non-trivial techniques seeding and soft must links for more accurate NID in the semi-supervised setting. Extensive experiments comparing NILC against multiple recent baselines under both unsupervised and semi-supervised settings showcase that NILC can achieve significant performance improvements over six benchmark datasets of diverse domains consistently.
>
---
#### [replaced 026] Omni-Embed-Audio: Leveraging Multimodal LLMs for Robust Audio-Text Retrieval
- **分类: cs.SD; cs.CL**

- **简介: 该论文属于音频-文本检索任务，旨在解决传统基准与真实搜索行为不匹配的问题。提出OEA模型，引入用户意图查询和硬负样本评估指标，提升检索鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.18360](https://arxiv.org/pdf/2604.18360)**

> **作者:** HaeJun Yoo; Yongseop Shin; Insung Lee; Myoung-Wan Koo; Du-Seong Chang
>
> **备注:** Accepted at ACL 2026 Main Conference. Camera-ready version
>
> **摘要:** Audio-text retrieval systems based on Contrastive Language-Audio Pretraining (CLAP) achieve strong performance on traditional benchmarks; however, these benchmarks rely on caption-style queries that differ substantially from real-world search behavior, limiting their assessment of practical retrieval robustness. We present Omni-Embed-Audio (OEA), a retrieval-oriented encoder leveraging multimodal LLMs with native audio understanding. To systematically evaluate robustness beyond caption-style queries, we introduce User-Intent Queries (UIQs) - five formulations reflecting natural search behaviors: questions, commands, keyword tags, paraphrases, and exclusion-based negative queries. For negative queries, we develop a hard negative mining pipeline and propose discrimination metrics (HNSR, TFR) assessing models' ability to suppress acoustically similar distractors. Experiments on AudioCaps, Clotho, and MECAT show that OEA achieves comparable text-to-audio retrieval performance to state-of-the-art M2D-CLAP, while demonstrating clear advantages in two critical areas: (1) dominant text-to-text retrieval (+22% relative improvement), and (2) substantially superior hard negative discrimination (+4.3%p HNSR@10, +34.7% relative TFR@10), revealing that LLM backbones provide superior semantic understanding of complex queries.
>
---
#### [replaced 027] Latent Collaboration in Multi-Agent Systems
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出LatentMAS，解决多智能体系统协作效率问题。通过直接在潜在空间协作，提升表达能力与信息传递效率，优于传统文本方法。**

- **链接: [https://arxiv.org/pdf/2511.20639](https://arxiv.org/pdf/2511.20639)**

> **作者:** Jiaru Zou; Ruizhong Qiu; Gaotang Li; Xiyuan Yang; Katherine Tieu; Pan Lu; Ke Shen; Hanghang Tong; Yejin Choi; Jingrui He; James Zou; Mengdi Wang; Ling Yang
>
> **备注:** ICML2026 Spotlight, Project: this https URL
>
> **摘要:** Multi-agent systems (MAS) extend large language models (LLMs) from independent single-model reasoning to coordinative system-level intelligence. While existing LLM agents depend on text-based mediation for reasoning and communication, we take a step forward by enabling models to collaborate directly within the continuous latent space. We introduce LatentMAS, an end-to-end training-free framework that enables pure latent collaboration among LLM agents. In LatentMAS, each agent first performs auto-regressive latent thoughts generation through last-layer hidden embeddings instead of text. Then, a shared latent working memory preserves and transfers each agent's internal representations and latent thoughts, ensuring lossless information exchange without re-encoding. We provide detailed theoretical analyses showing that LatentMAS achieves higher expressiveness and lossless information preservation with lower overall complexity than standard text-based MAS. In addition, empirical evaluations across 9 comprehensive benchmarks spanning math and science reasoning, commonsense understanding, and code generation show that LatentMAS outperforms advanced single agents and text-based MAS baselines, achieving up to 14.6% higher accuracy, reducing output token usage by 70.8%-83.7%, and providing 4$\times$-4.3$\times$ faster end-to-end inference. Code and data are fully open-sourced at this https URL.
>
---
#### [replaced 028] MIC: Maximizing Informational Capacity in Adaptive Representations via Isotropic Subspace Alignment
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出MIC框架，解决多尺度表示学习中的维度冗余和谱崩溃问题，通过子空间对齐提升信息容量。属于表示学习任务。**

- **链接: [https://arxiv.org/pdf/2605.29987](https://arxiv.org/pdf/2605.29987)**

> **作者:** Dang Nguyen Hong; Nhi Ngoc-Yen Nguyen; Huy-Hieu Pham
>
> **备注:** Accepted at the GlobalSouthML Workshop at ICML 2026. 8 pages, 2 figures
>
> **摘要:** Although multi-scales representation learning enables elastic-dimension embeddings, nested subspaces often suffer from dimensional redundancy and spectral collapse. To address this, we introduce MIC, a framework that optimizes the geometric landscape of multi-granular embeddings through isotropic subspace alignment. MIC employs Soft Collapse Regularization (SCR) to mitigate redundancy between prefix and residual subspaces via cross-correlation penalties, alongside Spectral Isotropy Regularization (SIR) to ensure hyper-spherical uniformity in low-dimensional prefixes. By unifying these strategies through a self-distillation objective, MIC generates semantically dense representations that maintain high discriminative power. Our experiments demonstrate that MIC significantly outperforms standard baselines, particularly in high-compression scenarios where maintaining informational capacity is most critical.
>
---
#### [replaced 029] AtomEval: Validity-Aware Atomic Evaluation of Adversarial Claim Rewriting in Fact Verification
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于事实验证任务，解决对抗性改写中攻击成功率虚高的问题。通过引入AtomEval，明确区分有效规避与命题改变，提升评估准确性。**

- **链接: [https://arxiv.org/pdf/2604.07967](https://arxiv.org/pdf/2604.07967)**

> **作者:** Hongyi Cen; Mingxin Wang; Yule Liu; Jingyi Zheng; Hanze Jia; Tan Tang
>
> **摘要:** Large language models (LLMs) can rewrite refuted claims to evade evidence-based fact verifiers, but conventional attack success rate (ASR) can be inflated when rewrites change, weaken, or correct the false proposition they are supposed to preserve. We introduce AtomEval, a validity-aware evaluation protocol for fixed-evidence adversarial claim rewriting. AtomEval represents claims as subject--relation--object--modifier (SROM) atoms, applies a one-way preservation gate to separate valid verifier evasion from proposition-changing rewrites, and reports validity-aware attack success rate (VASR), which counts only verifier-evasive rewrites that preserve the original false proposition. AtomEval further provides fine-grained diagnostics that explain both proposition-level failures and non-minimal valid rewrites. On FEVER refuted-claim rewriting, AtomEval exposes and explains ASR inflation: many apparent attacks fool the verifier by altering, weakening, or correcting the proposition they should preserve. By making attacked-proposition preservation explicit and measurable, AtomEval provides a stable evaluation target for evaluating adversarial rewriters that must balance verifier evasion with proposition preservation.
>
---
#### [replaced 030] Fine-Tuning Without Forgetting In-Context Learning: A Theoretical Analysis of Linear Attention Models
- **分类: cs.CL; cs.LG; stat.ML**

- **简介: 该论文研究模型微调对上下文学习的影响，属于自然语言处理任务。解决微调导致上下文学习性能下降的问题，通过理论分析和实验验证提出优化方法。**

- **链接: [https://arxiv.org/pdf/2602.23197](https://arxiv.org/pdf/2602.23197)**

> **作者:** Chungpa Lee; Jy-yong Sohn; Kangwook Lee
>
> **摘要:** Transformer-based large language models exhibit in-context learning, enabling adaptation to downstream tasks via few-shot prompting with demonstrations. In practice, such models are often fine-tuned to improve zero-shot performance on downstream tasks, allowing them to solve tasks without examples and thereby reducing inference costs. However, fine-tuning can degrade in-context learning, limiting the performance of fine-tuned models on tasks not seen during fine-tuning. Using linear attention models, we provide a theoretical analysis that characterizes how fine-tuning objectives modify attention parameters and identifies conditions under which this leads to degraded few-shot performance. We show that fine-tuning all attention parameters can harm in-context learning, whereas restricting updates to the value matrix improves zero-shot performance while preserving in-context learning. We further show that incorporating an auxiliary few-shot loss enhances in-context learning primarily on the target task, at the expense of degraded in-context learning ability on tasks not seen during fine-tuning. We provide empirical evidence from synthetic and real-world datasets consistent with the qualitative predictions of our theory.
>
---
#### [replaced 031] AutoEval Done Right: Using Synthetic Data for Model Evaluation
- **分类: cs.LG; cs.AI; cs.CL; stat.ME**

- **简介: 该论文属于模型评估任务，旨在减少人工标注数据的依赖。通过使用合成数据提升评估效率，提出统计有效的算法，提高样本利用率。**

- **链接: [https://arxiv.org/pdf/2403.07008](https://arxiv.org/pdf/2403.07008)**

> **作者:** Pierre Boyeau; Anastasios N. Angelopoulos; Nir Yosef; Jitendra Malik; Michael I. Jordan
>
> **备注:** camera-ready paper version
>
> **摘要:** The evaluation of machine learning models using human-labeled validation data can be expensive and time-consuming. AI-labeled synthetic data can be used to decrease the number of human annotations required for this purpose in a process called autoevaluation. We suggest efficient and statistically principled algorithms for this purpose that improve sample efficiency while remaining unbiased. These algorithms increase the effective human-labeled sample size by up to 50% on experiments with GPT-4.
>
---
#### [replaced 032] R2-Router: A New Paradigm for LLM Routing with Reasoning
- **分类: cs.CL**

- **简介: 该论文提出R2-Router，解决LLM路由问题，通过考虑输出长度预算，联合选择最佳模型和成本，提升效率。**

- **链接: [https://arxiv.org/pdf/2602.02823](https://arxiv.org/pdf/2602.02823)**

> **作者:** Jiaqi Xue; Qian Lou; Jiarong Xing; Heng Huang
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** As LLMs proliferate with diverse capabilities and costs, LLM routing has emerged by learning to predict each LLM's quality and cost for a given query, then selecting the one with high quality and low cost. However, existing routers implicitly assume a single fixed quality and cost per LLM for each query, ignoring that the same LLM's quality varies with its output length. This causes routers to exclude powerful LLMs when their estimated cost exceeds the budget, missing the opportunity that these LLMs could still deliver high quality at reduced cost with shorter outputs. To address this, we introduce R2-Router, which treats output length budget as a controllable variable and jointly selects the best LLM and length budget, enforcing the budget via length-constrained instructions. This enables R2-Router to discover that a powerful LLM with constrained output can outperform a weaker LLM at comparable cost-efficient configurations invisible to prior methods. Together with the router framework, we construct R2-Bench, the first routing dataset capturing LLM behavior across diverse output length budgets. Experiments show that R2-Router achieves state-of-the-art performance at 4-5\times lower cost compared with existing routers. This work opens a new direction: routing as reasoning, where routers evolve from reactive selectors to deliberate reasoners that explore which LLM to use and at what cost budget. The code is publicly available at this https URL.
>
---
#### [replaced 033] Casual as an Anchor: Resolving Supervision Misalignment in Formality Transfer Dataset
- **分类: cs.CL**

- **简介: 该论文研究形式化转换任务，解决现有数据集监督信号不准确的问题。通过引入三层次标注框架3LF，提升模型生成正式语言的准确性。**

- **链接: [https://arxiv.org/pdf/2605.29365](https://arxiv.org/pdf/2605.29365)**

> **作者:** Hyojeong Yu; Hyukhun Koh; Minsung Kim; Kyomin Jung
>
> **备注:** HEAL@CHI 2026 Workshop Paper
>
> **摘要:** Formality transfer is commonly framed as a symmetric bidirectional task between informal and formal registers. We argue that this framing conceals a supervision design flaw in existing benchmarks such as GYAFC: binary human rewrites encode relative stylistic shifts rather than absolute human notions of formality. Consequently, models learn to generate pseudo-formal outputs that satisfy benchmark labels while failing to produce genuinely formal language. We quantify this misalignment by re-evaluating benchmark formal labels under a human-aligned definition of formality, revealing substantial discrepancies that propagate to consistent informal-to-formal failures across model families. To address this issue, we reconceptualize formality transfer as a graded dimension rather than a binary attribute. We introduce a three-level spectrum: informal, casual, and formal, where casual serves as an explicit intermediate state that clarifies supervision signals. Based on this framework, we introduce 3LF, a dataset providing parallel supervision across all three levels. Training on 3LF substantially reduces informal-to-formal failures and improves alignment with human perception. For example, GPT-4.1-nano improves from 0.06 to 0.88 F1 in the informal-to-formal direction despite 3LF being significantly smaller than GYAFC. We further demonstrate that these gains cannot be reproduced through in-context learning alone and provide qualitative analyses of ambiguity-driven errors and meaning distortions. Overall, our findings demonstrate how supervision design shapes stylistic alignment and highlight the importance of alignment-aware benchmark construction in controllable text generation.
>
---
#### [replaced 034] WorldMemArena: Evaluating Multimodal Agent Memory Through Action-World Interaction
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态记忆研究任务，旨在解决长周期智能体记忆评估问题。通过构建WorldMemArena框架，分析记忆的撰写、维护与使用效果。**

- **链接: [https://arxiv.org/pdf/2605.29341](https://arxiv.org/pdf/2605.29341)**

> **作者:** Chengzhi Liu; Yuzhe Yang; Sophia Xiao Pu; Yepeng Liu; Lin Long; Yichen Guo; Nuo Chen; Zhaotian Weng; Elena Kochkina; Simerjot Kaur; Charese Smiley; Xiaomo Liu; James Zou; Sheng Liu; Yuheng Bu; Songyou Peng; Xin Eric Wang
>
> **备注:** 25 pages, 8 figures
>
> **摘要:** Multimodal large language models are increasingly deployed as long-horizon agents, where memory must do more than recall: it must track an evolving world, revise what has gone stale, and surface the right evidence at decision time. Existing benchmarks measure recall over static dialogue, collapse memory into a single end-of-task accuracy, and reduce visual observations to captions, leaving us unable to localize failures to writing, maintenance, retrieval, or use. The rise of agent harnesses that author their own memory sharpens this gap, since we have no principled way to compare hand-designed pipelines with self-managing alternatives. To close these gaps, we formulate multimodal agent memory as an Action-World Interaction Loop with an observable four-stage lifecycle, and instantiate it in WorldMemArena: 400 multi-session multimodal tasks spanning Lifelong Evolution (evolving personal and task states) and Agentic Execution (memory from real observations, actions, and feedback), annotated with gold memory points, updates, distractors, and evidence chains for stage-level diagnosis. This enables the first head-to-head comparison of long-context, manually designed (RAG and external memory systems), and harness-based memory agents. Results show that: (1) better memory writing and storage do not guarantee better performance; (2) multimodal memory still struggles to fully use visual evidence; (3) systems are unstable across domains and degrade on realistic agentic trajectories; and (4) harness memory is more flexible but remains costly and less reliable.
>
---
#### [replaced 035] TInR: Exploring Tool-Internalized Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出TInR-U框架，解决LLM在推理中依赖外部工具的效率与难度问题，通过内部化工具知识提升推理能力。属于自然语言处理中的模型优化任务。**

- **链接: [https://arxiv.org/pdf/2604.10788](https://arxiv.org/pdf/2604.10788)**

> **作者:** Qiancheng Xu; Yongqi Li; Fan Liu; Hongru Wang; Min Yang; Wenjie Li
>
> **备注:** Accepted to ACL 2026
>
> **摘要:** Tool-Integrated Reasoning (TIR) has emerged as a promising direction by extending Large Language Models' (LLMs) capabilities with external tools during reasoning. Existing TIR methods typically rely on external tool documentation during reasoning. However, this leads to tool mastery difficulty, tool size constraints, and inference inefficiency. To mitigate these issues, we explore Tool-Internalized Reasoning (TInR), aiming at facilitating reasoning with tool knowledge internalized into LLMs. Achieving this goal presents notable requirements, including tool internalization and tool-reasoning coordination. To address them, we propose TInR-U, a tool-internalized reasoning framework for unified reasoning and tool usage. TInR-U is trained through a three-phase pipeline: 1) tool internalization with a bidirectional knowledge alignment strategy; 2) supervised fine-tuning warm-up using high-quality reasoning annotations, and 3) reinforcement learning with TInR-specific rewards. We comprehensively evaluate our method across in-domain and out-of-domain settings. Experiment results show that TInR-U achieves superior performance in both settings, highlighting its effectiveness and efficiency.
>
---
#### [replaced 036] Targeted Remasking: Replacing Token Editing with Token-to-Mask Refinement in Discrete Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言生成任务，针对离散扩散模型中token编辑的局限性，提出T2M remasking方法，通过重置错误token为mask状态提升生成质量。**

- **链接: [https://arxiv.org/pdf/2605.26436](https://arxiv.org/pdf/2605.26436)**

> **作者:** Lin Yao
>
> **备注:** This paper has been significantly revised, expanded, and superseded by a more comprehensive version available at arXiv:2604.18738. The authors have chosen to withdraw this version to avoid overlap and direct readers to the updated work
>
> **摘要:** Discrete masked diffusion language models such as LLaDA generate text through iterative denoising, where mask tokens are progressively replaced with predicted tokens. LLaDA2.1 introduced a Token-to-Token (T2T) editing mechanism that accelerates generation by directly replacing committed tokens suspected of being incorrect. However, we identify fundamental limitations of T2T editing: it couples error detection with replacement, pollutes the generation context with potentially incorrect tokens, and introduces a train-inference noise mismatch where systematic model-generated errors differ from the random perturbations seen during training. We propose Token-to-Mask (T2M) remasking, a training-free, drop-in replacement for T2T editing that resets suspected erroneous tokens back to the mask state, allowing the diffusion process to re-predict them under cleaner context. We design and empirically validate three complementary error detection strategies -- probability-based, trigger-mirrored, and temporal-difference-based -- and provide a unified theoretical analysis showing that T2M remasking purifies the generation context, converts systematic inference errors back to the model's native mask noise type, and enables delayed commitment for joint multi-position optimization. Comprehensive experiments across 12 benchmarks spanning knowledge, reasoning, mathematics, coding, and instruction following show that T2M generally improves performance on tasks requiring precise token-level output, with the largest gain on mathematics (+5.92% on CMATH). Error analysis on CMATH reveals that the dominant failure mode is last-mile token corruption -- where correct reasoning produces a corrupted final answer -- and that T2M repairs 59.4% of such cases.
>
---
#### [replaced 037] Beyond the Crowd: LLM-Augmented Community Notes for Governing Health Misinformation
- **分类: cs.SI; cs.CL**

- **简介: 该论文属于健康信息治理任务，旨在解决社区注释系统响应慢、准确性低的问题。通过引入LLM增强框架CrowdNotes+，提升误信信息处理的效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2510.11423](https://arxiv.org/pdf/2510.11423)**

> **作者:** Jiaying Wu; Zihang Fu; Haonan Wang; Fanxiao Li; Jiafeng Guo; Preslav Nakov; Min-Yen Kan
>
> **备注:** ACL 2026
>
> **摘要:** Community Notes, the crowd-sourced misinformation governance system on X (formerly Twitter), allows users to flag misleading posts, attach contextual notes, and rate the notes' helpfulness. However, our empirical analysis of 30.8K health-related notes reveals substantial latency, with a median delay of 17.6 hours before notes receive a helpfulness status. To improve responsiveness during real-world misinformation surges, we propose CrowdNotes+, a unified LLM-based framework that augments Community Notes for faster and more reliable health misinformation governance. CrowdNotes+ integrates two modes: (1) evidence-grounded note augmentation and (2) utility-guided note automation, supported by a hierarchical three-stage evaluation of relevance, correctness, and helpfulness. We instantiate the framework with HealthNotes, a benchmark of 1.2K health notes annotated for helpfulness, and a fine-tuned helpfulness judge. Our analysis first uncovers a key loophole in current crowd-sourced governance: voters frequently conflate stylistic fluency with factual accuracy. Addressing this via our hierarchical evaluation, experiments across 15 representative LLMs demonstrate that CrowdNotes+ significantly outperforms human contributors in note correctness, helpfulness, and evidence utility.
>
---
#### [replaced 038] Cast a Wider Net: Coordinated Pass@K Policy Optimization for Code Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于代码生成任务，解决重复采样导致资源浪费的问题。提出CPPO方法，通过协同策略探索提升pass@$K$性能。**

- **链接: [https://arxiv.org/pdf/2605.27000](https://arxiv.org/pdf/2605.27000)**

> **作者:** Yilong Li; Suman Banerjee; Tong Che
>
> **备注:** Code reasoning; pass@K optimization; coordinated planning; verifiable rewards; strategy diversity
>
> **摘要:** Repeated sampling with a verifier is the standard way to allocate test-time compute for code generation, with pass@$K$ as the canonical metric. Yet the standard policy class draws $K$ independent samples from a single answer distribution, so attempts often collapse onto near-duplicate reasoning paths and waste the budget on redundant rollouts. This failure is costly in competitive programming, where many problems admit multiple distinct algorithmic strategies and pass@$K$ requires only one correct attempt. We propose Coordinated Pass@$K$ Policy Optimization (CPPO), which turns pass@$K$ generation into joint exploration over strategies: a planner emits a tuple of $K{=}4$ alternative high-level methods, and a shared solver attempts one solution per method. CPPO trains this joint policy with a multiplicative planner reward, $R_{\mathrm{plan}} = J_\psi \cdot R_{\mathrm{out}}$, assigning credit only to valid strategy tuples that lead to verifier-confirmed pass@$K$ success. Across APPS, CodeContests, and LiveCodeBench-v6, CPPO improves pass@$4$ over direct sampling, planning baselines, planner-only SFT, and pass@$K$-oriented RL under the same $K{=}4$ solver-attempt budget, with statistically significant gains on six of nine model--benchmark cells. The largest single gain is $+0.16$ on Qwen3.5-9B LiveCodeBench-v6 over the strongest baseline, PKPO ($0.588 \rightarrow 0.748$; paired bootstrap, $p < 0.05$).
>
---
#### [replaced 039] Language-Native Materials Processing Design by Lightly Structured Text Database and Reasoning Large Language Model
- **分类: cs.DB; cond-mat.mtrl-sci; cs.AI; cs.CL**

- **简介: 该论文属于材料合成任务，旨在解决传统文本记录难以被数据驱动方法利用的问题。通过构建轻度结构化知识库和大语言模型推理，实现更精准的合成路径规划与优化。**

- **链接: [https://arxiv.org/pdf/2509.06093](https://arxiv.org/pdf/2509.06093)**

> **作者:** Yuze Liu; Zhaoyuan Zhang; Xiangsheng Zeng; Yihe Zhang; Leping Yu; Liu Yang; Lejia Wang; Xi Yu
>
> **摘要:** Materials synthesis procedures are predominantly documented as narrative text in papers, protocols, and laboratory records, placing them beyond the reach of conventional data-driven optimization frameworks. This language-native character poses a particular challenge for complex, multistage processes such as the preparation of boron nitride nanosheets (BNNS), where outcomes depend on path-dependent choices in exfoliation, functionalization, and functionalization. Here, we recast synthesis planning of the materials as a text reasoning problem enabled by a lightly structured knowledge substrate that preserves the procedural logic and causal contexts while exposing computable elements for retrieval. Built on this representation, our framework combines semantic matching, lexical search, and parameter-aware filtering to support retrieval-augmented generation with more accurate and better-grounded synthesis guidance. We further introduce experience-augmented reasoning, in which iteratively refined text guides distilled from multi-source narratives support hypothesis generation, failure diagnosis, and protocol revision. We validated the framework in the targeted exfoliation of BNNS, a synthesis problem governed by multivariate constraints and limited transferability of literature protocols across laboratory settings. By integrating dispersed literature evidence with experimentally observed failure modes, the system converged within only three iterative rounds on a high-performing protocol that yielded high-quality ultrathin nanosheets meeting the target specifications, substantially shortening what is often a prolonged cycle of expert-led trial-and-error. By enabling language-native reasoning over procedural knowledge, this framework moves AI beyond literature assistance toward active synthesis planning, adaptation and acceleration in complex materials workflows.
>
---
#### [replaced 040] WAON: A Large-Scale Japanese Image-Text Dataset for Cultural Adaptation in Contrastive Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于跨模态学习任务，旨在提升视觉-语言模型对日本文化的理解。提出WAON数据集和基准，验证本地化数据对文化适应的有效性。**

- **链接: [https://arxiv.org/pdf/2510.22276](https://arxiv.org/pdf/2510.22276)**

> **作者:** Issa Sugiura; Shuhei Kurita; Yusuke Oda; Daisuke Kawahara; Yasuo Okabe; Naoaki Okazaki
>
> **备注:** 13 pages, 7 figures
>
> **摘要:** Contrastive vision-language models have achieved remarkable progress through large-scale pretraining. Recent work has shown that removing English-only caption filters and pretraining on global data is effective for improving multicultural performance. We study whether such global pretraining is sufficient for culture-specific understanding, or whether further adaptation with natively sourced data can boost performance beyond what global pretraining alone achieves. To enable this investigation, we present WAON, the largest publicly available native Japanese image-text dataset constructed from native Japanese web content in Common Crawl, containing approximately 155 million examples. We also introduce WAON-Bench, a manually curated Japanese cultural benchmark spanning 374 classes. Through comparative fine-tuning experiments on multiple Japanese image-text datasets, we observe that models fine-tuned on WAON consistently achieve stronger performance on Japanese cultural benchmarks than those fine-tuned on English-to-Japanese translated data. We release our dataset and code.
>
---
#### [replaced 041] LLM as a Meta-Judge: Synthetic Data for NLP Evaluation Metric Validation
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成（NLG）评估任务，旨在解决人工标注成本高、语言受限的问题。通过LLM生成合成数据，替代人工判断，验证评估指标有效性。**

- **链接: [https://arxiv.org/pdf/2603.09403](https://arxiv.org/pdf/2603.09403)**

> **作者:** Lukáš Eigler; Jindřich Libovický; David Hurych
>
> **备注:** 16 pages, 1 figure, 14 tables
>
> **摘要:** Validating evaluation metrics for NLG typically relies on expensive and time-consuming human annotations, which predominantly exist only for English datasets. We propose LLM as a Meta-Judge, a scalable framework that utilizes LLMs to generate synthetic evaluation datasets via controlled semantic degradation of real data, replacing human judgment. We validate our approach using \textit{meta-correlation}, measuring the alignment between metric rankings derived from synthetic data and those from standard human benchmarks. Experiments across Machine Translation, Question Answering, and Summarization demonstrate that synthetic validation serves as a reliable proxy for human judgment, achieving meta-correlations exceeding 0.9 in multilingual QA and proves to be a viable alternative where human judgments are unavailable or too expensive to obtain. Our code and data will become publicly available upon paper acceptance.
>
---
#### [replaced 042] How to Correctly Report LLM-as-a-Judge Evaluations
- **分类: cs.LG; cs.CL; stat.AP; stat.ML**

- **简介: 该论文属于模型评估任务，解决LLM作为评判者时的偏差问题，提出框架修正偏差并量化不确定性，提升评估可靠性。**

- **链接: [https://arxiv.org/pdf/2511.21140](https://arxiv.org/pdf/2511.21140)**

> **作者:** Chungpa Lee; Thomas Zeng; Jongwon Jeong; Jy-yong Sohn; Kangwook Lee
>
> **摘要:** Large language models (LLMs) are widely used as scalable evaluators of model responses in lieu of human annotators. However, imperfect sensitivity and specificity of the LLM judges induce bias in naive evaluation scores. We propose a simple plug-in framework that corrects this bias and enables statistically principled uncertainty quantification. Our framework constructs confidence intervals that account for uncertainty from both the test dataset and a human-labeled calibration dataset. Additionally, it uses an adaptive strategy to allocate calibration samples for tighter intervals. Importantly, we characterize parameter regimes defined by the true evaluation score and the LLM judge's sensitivity and specificity in which our LLM-based evaluation yields more reliable estimates than human-only evaluation. Moreover, we show that our framework remains unbiased under distribution shift between the test and calibration datasets, in contrast to existing approaches.
>
---
#### [replaced 043] Evaluating Reliability Asymmetries in Chinese Factual Search and AI Answers
- **分类: cs.IR; cs.AI; cs.CL; cs.CY**

- **简介: 该论文属于信息检索任务，研究中文搜索与AI回答的可靠性差异，分析系统在事实性问题上的准确性和回答频率。**

- **链接: [https://arxiv.org/pdf/2602.22221](https://arxiv.org/pdf/2602.22221)**

> **作者:** Geng Liu; Li Feng; Mengxiao Zhu; Francesco Pierri
>
> **摘要:** Search engines and AI-powered systems increasingly mediate access to factual information, yet their reliability remains difficult to evaluate in realistic information-seeking settings. We study this problem in the Chinese web ecosystem by constructing a query-based fact-checking dataset from real Chinese search logs and comparing nine systems across traditional search engines, standalone large language models, and search-integrated AI Overviews. Focusing on factual Chinese-language factual Yes/No questions, we evaluate whether systems provide correct, incorrect, or uncertain decisions against evidence-derived ground truth. We find that systems are similarly accurate when they provide definitive answers, but differ sharply in how often they do so. Conditional accuracy ranges from 73.2% to 78.9%, yet search engines answer definitively on over 83% of queries, while Qwen-Max does so on fewer than half. We also find a consistent polarity gap: all systems perform better on yes-labeled queries than on no-labeled queries. We also use Baidu Index data to identify Chinese provinces with higher health-related search attention, which may indicate greater potential exposure to misinformation. Overall, our results show that reliability depends not only on whether systems are correct when they answer, but also on how often they answer, how they handle negative claims, and where information demand may increase exposure risks.
>
---
#### [replaced 044] Evaluating the Reversal Curse in Model Editing
- **分类: cs.CL**

- **简介: 该论文研究模型编辑中的反向诅咒问题，属于知识编辑任务。旨在评估编辑后的模型是否能双向回忆知识，提出新指标和基准进行实验分析。**

- **链接: [https://arxiv.org/pdf/2310.10322](https://arxiv.org/pdf/2310.10322)**

> **作者:** Hao-Xiang Xu; Jun-Yu Ma; Jia-Chen Gu; Zhen-Hua Ling; Quan Liu; Cong Liu
>
> **备注:** Accepted by TMLR
>
> **摘要:** Large language models (LLMs) are prone to hallucinate unintended text due to false or outdated knowledge. Since retraining LLMs is resource intensive, there has been a growing interest in model editing. Despite the emergence of benchmarks and approaches, existing unidirectional editing and evaluation paradigms have failed to explore the reversal curse. In this paper, we study bidirectional language model editing, aiming to provide a rigorous evaluation to assess if edited LLMs can recall the editing knowledge bidirectionally. A metric of reverse generalization is introduced and a benchmark dubbed Bidirectional Assessment for Knowledge Editing (BAKE) is constructed to evaluate if post-edited models can recall the edited knowledge in the reverse direction of editing. We conduct extensive experiments using a variety of editing methods and LLMs. The results show that while most editing methods are able to accurately recall editing facts along the modification direction, they exhibit substantial systematic deficiencies when evaluating in the reverse direction. To further investigate the underlying causes of reversal curse and to explore potential strategies for mitigation, a detailed analysis is conducted from three perspectives. Our findings reveal that although In-Context Learning (ICL) can mitigate the reversal curse to a certain extent, it lacks continuity, is limited by the input length, and may introduce hallucinations. Therefore, combining the advantages of ICL and other editing methods is a promising direction for developing new editing paradigms.
>
---
#### [replaced 045] GRASP: Plan-Guided Graph Retrieval with Adaptive Fusion and Reranking on Semi-Structured Knowledge Bases
- **分类: cs.IR; cs.CL; cs.LG**

- **简介: 该论文提出GRASP框架，解决半结构化知识库的检索问题。通过三阶段方法提升检索效果，显著提高Hit@1指标。**

- **链接: [https://arxiv.org/pdf/2605.30237](https://arxiv.org/pdf/2605.30237)**

> **作者:** Yicheng Tao; Yiqun Wang; Xiangchen Song; Xin Luo; Kai Liu; Jie Liu
>
> **摘要:** Semi-structured knowledge bases (SKBs) embed textual documents in a typed graph of entities and relations, and underpin applications such as product search, academic paper search, and precision-medicine inquiries. Existing hybrid retrieval systems on SKBs either use the graph only for query expansion, mix textual and structural branches under a global weighting, or rely on fine-tuned graph-traversal generators. We present GRASP, a three-stage SKB retrieval framework unifying plan-based graph retrieval, plan-conditioned fusion with a dense retriever, and a fine-tuned reranker over the fused candidates. GRASP substantially advances the state of the art on every metric across the three STaRK benchmarks, lifting average Hit@1 from 62.0 to 73.9. Ablation and sensitivity studies further confirm the effectiveness and robustness of GRASP.
>
---
#### [replaced 046] One Bias After Another: Mechanistic Reward Shaping and Persistent Biases in Language Reward Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型对齐任务，解决奖励模型中的偏差问题。通过分析RM的缺陷，提出机制性奖励塑造方法以减少偏差，提升模型行为与人类偏好的一致性。**

- **链接: [https://arxiv.org/pdf/2603.03291](https://arxiv.org/pdf/2603.03291)**

> **作者:** Daniel Fein; Max Lamparth; Violet Xiang; Mykel J. Kochenderfer; Nick Haber
>
> **备注:** ICML 2026 Camera-ready
>
> **摘要:** Reward Models (RMs) are crucial for online alignment of language models (LMs) with human preferences. However, RM-based preference-tuning is vulnerable to reward hacking, whereby LM policies learn undesirable behaviors from flawed RMs. By systematically measuring biases in five high-quality RMs, including the state-of-the-art, we find that issues persist despite prior work with respect to length, sycophancy, and overconfidence. We also discover new issues related to bias toward model-specific ``styles'' and answer-order. We categorize RM failures as tractable or resistant to linear intervention and propose a simple post-hoc intervention to mitigate low-complexity biases that arise from spurious correlations. Our proposed mechanistic reward shaping reduces targeted biases without degrading reward quality and while using minimal labeled data. The method is extensible to new biases, model-internal, and generalizes out-of-distribution.
>
---
#### [replaced 047] Render-of-Thought: Rendering Textual Chain-of-Thought as Images for Visual Latent Reasoning
- **分类: cs.CL; cs.CV**

- **简介: 该论文提出RoT框架，将文本推理过程转化为图像，解决LLM推理冗余与可解释性差的问题。属于视觉-语言推理任务，通过图像显式化推理链，提升效率与可分析性。**

- **链接: [https://arxiv.org/pdf/2601.14750](https://arxiv.org/pdf/2601.14750)**

> **作者:** Yifan Wang; Shiyu Li; Peiming Li; Xiaochen Yang; Yang Tang; Zheng Wei
>
> **备注:** Accepted by ACL 2026 Main Conference
>
> **摘要:** Chain-of-Thought (CoT) prompting has achieved remarkable success in unlocking the reasoning capabilities of Large Language Models (LLMs). Although CoT prompting enhances reasoning, its verbosity imposes substantial computational overhead. Recent works often focus exclusively on outcome alignment and lack supervision on the intermediate reasoning process. These deficiencies obscure the analyzability of the latent reasoning chain. To address these challenges, we introduce Render-of-Thought (RoT), the first framework to reify the reasoning chain by rendering textual steps into images, making the latent rationale explicit and traceable. Specifically, we leverage the vision encoders of existing Vision Language Models (VLMs) as semantic anchors to align the vision embeddings with the textual space. This design ensures plug-and-play implementation without incurring additional pre-training overhead. Extensive experiments on mathematical and logical reasoning benchmarks demonstrate that our method achieves 3-4x token compression and substantial inference acceleration compared to explicit CoT. Furthermore, it maintains competitive performance against other methods, validating the feasibility of this paradigm. Our code is available at this https URL
>
---
#### [replaced 048] Semantic Motion Anchors: Bridging Motion and Meaning in Co-Speech Gestures
- **分类: cs.CL**

- **简介: 该论文属于跨模态检索任务，旨在解决语义手势与语音文本对齐的问题。通过引入语义运动锚点，提升手势检索的语义准确性。**

- **链接: [https://arxiv.org/pdf/2605.30608](https://arxiv.org/pdf/2605.30608)**

> **作者:** Varsha Suresh; Mohammad Mahdi Abootorabi; Mohamed Salman; M. Hamza Mughal; Christian Theobalt; Ashwin Ram; Jürgen Steimle; Vera Demberg
>
> **摘要:** Learning a shared representation between spoken text and gesture is central to co-speech gesture retrieval, synthesis, and understanding, but remains challenging for semantically meaningful gestures whose communicative intent is not captured by motion alone. Direct contrastive alignment between transcripts and continuous motion embeddings often overemphasizes low-level kinematics and misses the symbolic content of semantic gestures. We propose semantic motion anchors, natural-language abstractions of gesture motion capturing physical form and communicative intent. Our method discretizes 3D gestures into body-hand motion primitives, verbalizes them into structured descriptions, and grounds them in the transcript to provide auxiliary contrastive supervision. On BEAT2, our method improves text-to-gesture R@1 by 8.2% over a direct text-motion baseline and outperforms prior retrieval approaches on text to gesture and gesture to text retrieval directions. Beyond aggregate retrieval metrics, semantic motion anchor supervision helps retrieve gestures that are semantically meaningful for the spoken query, rather than defaulting to generic motion patterns. A downstream retrieval-augmented gesture generation study showed that users significantly preferred gestures retrieved by our approach over a retrieval-augmented generation baseline, demonstrating that semantically grounded retrieval translates to gestures that better convey communicative intent in downstream generation.
>
---
#### [replaced 049] Understanding the Effects of Distractors on Reasoning Vision-Language Models
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于视觉语言模型任务，研究干扰信息对模型推理的影响。通过构建包含干扰项的数据集，分析干扰如何影响模型的推理长度和准确性，并提出缓解策略。**

- **链接: [https://arxiv.org/pdf/2511.21397](https://arxiv.org/pdf/2511.21397)**

> **作者:** Jiyun Bae; Hyunjong Ok; Sangwoo Mo; Jaeho Lee
>
> **备注:** preprint
>
> **摘要:** How does irrelevant information (i.e., distractors) affect test-time scaling in vision-language models (VLMs)? Prior work on text-only language models has shown that textual distractors can intensify inverse scaling, causing models to reason longer but less effective reasoning traces. In this work, we investigate whether similar phenomena arise in multimodal settings. We introduce Idis (Images with distractors), a visual question-answering dataset that systematically varies distractors along semantic and numerical dimensions. Our analyses reveal that visual distractors affect reasoning VLMs in a fundamentally different way from textual distractors: although inverse scaling still emerges, visual distractors reduce accuracy without increasing reasoning length. We further show that attribute counts extracted from reasoning traces provide key insights into how distractors interact with reasoning length and accuracy. As a sanity check, we propose a simple prompting strategy that mitigates distractor-driven predictions in reasoning vision-language models.
>
---
#### [replaced 050] LLM Anonymization Against Agentic Re-Identification
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于文本匿名化任务，解决在保护隐私的同时保留文本实用性的难题。提出AURA框架，通过掩码重建方法增强隐私并保持上下文价值。**

- **链接: [https://arxiv.org/pdf/2605.30848](https://arxiv.org/pdf/2605.30848)**

> **作者:** Ziwen Li; Jianing Wen; Tianshi Li
>
> **备注:** 32 pages, 7 figures
>
> **摘要:** Agentic LLMs with web search change the threat model for text anonymization: weak contextual cues can become cross-referenceable evidence for re-identification, yet those same details also carry downstream analytic value of the text. Existing defenses either remove explicit identifiers, perturb text for formal privacy, or test rewritten text against non-web inference models, leaving underexplored the operating region between resistance to agentic web-search re-identification and utility retention. We introduce AURA (\textbf{A}nonymization with \textbf{U}tility-\textbf{R}etention \textbf{A}daptation), an LLM-powered \textit{mask-reconstruct} framework that decouples privacy localization from utility-preserving reconstruction and selects candidates with adversarial privacy and utility-retention checks. We evaluate AURA on real-user interview transcripts using re-identification attacks carried out by web-search agents, along with a utility evaluation based on interviewee-profile facts, codebook facts, and the joint contextual utility grid. Our results show that AURA improves the privacy-utility frontier by using adaptive privacy scope to strengthen resistance to agentic re-identification and using a mask-reconstruct anonymization method to better preserve contextual utility under fixed privacy scope.
>
---
#### [replaced 051] ADRA-Bank: A Modular Benchmark for Academic Deep Research Agents
- **分类: cs.CL**

- **简介: 该论文属于学术深度研究代理评估任务，旨在解决现有基准不足的问题。提出ADRA-Bank及ADRA-Eval，评估DR代理的规划、检索与推理能力。**

- **链接: [https://arxiv.org/pdf/2512.00986](https://arxiv.org/pdf/2512.00986)**

> **作者:** Zhihan Guo; Feiyang Xu; Yifan Li; Muzhi Li; Shuai Zou; Jiele Wu; Han Shi; Haoli Bai; Ho-fung Leung; Irwin King
>
> **摘要:** A surge in academic publications calls for automated deep research (DR) systems, but accurately evaluating them is still an open problem. First, existing benchmarks often focus narrowly on retrieval while neglecting high-level planning and reasoning. Second, existing benchmarks favor general domains over the academic domains that are the core application for DR agents. To address these gaps, we introduce ADRA-Bank, a modular benchmark for Academic DR Agents. Grounded in academic literature, our benchmark is a human-annotated dataset of 200 instances across 10 academic domains, including both research and review papers. Furthermore, we propose a modular Evaluation Paradigm for Academic DR Agents (ADRA-Eval), which leverages the rich structure of academic papers to assess the core capabilities of planning, retrieval, and reasoning. It employs two complementary modes: an end-to-end evaluation for \task agents and an isolated evaluation for foundational LLMs as potential backbones. Results reveal uneven capabilities: while agents show specialized strengths, they struggle with multi-source retrieval and cross-field consistency. Moreover, improving high-level planning capability is the crucial factor for unlocking the reasoning potential of foundational LLMs as backbones. By exposing these actionable failure modes, ADRA-Bank provides a diagnostic tool to guide the development of more reliable automatic academic research assistants.
>
---
#### [replaced 052] Simultaneous Multi-objective Alignment Across Verifiable and Non-verifiable Rewards
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大语言模型对齐任务，旨在解决多目标对齐中的冲突与效率问题。提出MAHALO框架，实现跨领域多目标同步对齐与可控推理。**

- **链接: [https://arxiv.org/pdf/2510.01167](https://arxiv.org/pdf/2510.01167)**

> **作者:** Yiran Shen; Yu Xia; Jonathan Chang; Prithviraj Ammanabrolu
>
> **备注:** ICML 2026
>
> **摘要:** Aligning large language models to human preferences is inherently multidimensional, yet most pipelines collapse heterogeneous signals into a single objective. We seek to answer what it would take to simultaneously align a model across various domains spanning those with: verifiable rewards, non-verifiable subjective preferences, and complex interactive scenarios. Such multi-objective alignment setups are often plagued by individual objectives being at odds with each other, resulting in inefficient training and limited user control during inference. To address these issues, we propose $\textbf{M}$ulti-$\textbf{A}$ction-$\textbf{H}$ead $\textbf{AL}$ignment with PRM-guided Dec$\textbf{O}$ding ($\textbf{MAHALO}$), a unified framework that standardizes PRM training across verifiable and non-verifiable settings for step-level supervision, performs vectorized multi-objective alignment with Multi-Action-Head DPO, and enables controllable inference through objective-specific weighting and PRM-guided decoding. Experiments across math reasoning, human values alignment, and multi-turn tutoring show that MAHALO jointly improves multiple objectives simultaneously with limited interference, while remaining generalizable and adaptable across domains and offering flexible user control at inference time. Our code is available at: this https URL.
>
---
#### [replaced 053] MemoNoveltyAgent: A Historical Research Memory-Aware Agent Workflow for Paper Novelty Assessment
- **分类: cs.CL**

- **简介: 该论文属于论文新颖性评估任务，旨在解决现有AI系统在学术文献分析中的不足。提出MemoNoveltyAgent，结合记忆机制与RAG技术，提升新颖性报告的全面性与准确性。**

- **链接: [https://arxiv.org/pdf/2603.20884](https://arxiv.org/pdf/2603.20884)**

> **作者:** Jiajun Hou; Hexuan Deng; Wenxiang Jiao; Xuebo Liu; Xiaopeng Ke; Derek F. Wong; Min Zhang
>
> **摘要:** To alleviate the heavy burden of paper screening, researchers increasingly rely on existing AI agents, such as AI reviewers or DeepResearch, for paper evaluation and novelty assessment. However, lacking specialized mechanisms for processing scholarly literature, their analyses often produce superficial results with noticeable deficiencies in quality. To bridge this gap, we introduce MemoNoveltyAgent, a multi-agent system designed to generate comprehensive and faithful novelty reports. Beyond retrieving concrete prior-paper evidence via RAG, our system incorporates a high-level abstract memory constructed from large-scale scholarly corpora. This memory organizes research into hierarchical trees to distill field-specific evolutionary trajectories, thereby providing a broader historical context. Furthermore, we decompose papers into discrete novelty points for fine-grained analysis and retrieval, while employing a self-validation mechanism to improve report faithfulness. Finally, to address the evaluation challenges of such open-ended generation tasks, we propose a RAG-augmented checklist evaluation method that enables reliable and evidence-grounded assessments. Extensive experiments demonstrate that MemoNoveltyAgent outperforms GPT-5 DeepResearch by 13.69%. Code and demo are available at this https URL
>
---
#### [replaced 054] When the Gold Standard Isn't Necessarily Standard: Challenges of Evaluating the Translation of User-Generated Content
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，探讨UGC翻译评估难题。研究分析UGC非标准现象及翻译策略，提出评估框架改进模型表现。**

- **链接: [https://arxiv.org/pdf/2512.17738](https://arxiv.org/pdf/2512.17738)**

> **作者:** Lydia Nishimwe; Benoît Sagot; Rachel Bawden
>
> **备注:** 10 pages (23 with references and appendices). Accepted at EAMT 2026
>
> **摘要:** User-generated content (UGC) is characterised by frequent use of non-standard language, from spelling errors to expressive choices such as slang, character repetitions, and emojis. This makes evaluating UGC translation challenging: what counts as a "good" translation depends on the desired standardness level of the output. To explore this, we examine the human translation guidelines of four UGC datasets, and derive a taxonomy of twelve non-standard phenomena and five translation actions (NORMALISE, COPY, TRANSFER, OMIT, CENSOR). Our analysis reveals notable differences in how UGC is treated, resulting in a spectrum of standardness in reference translations. We show that translation scores of large language models are highly sensitive to prompts with explicit UGC translation instructions, and that they improve when they align with the dataset guidelines. We argue that fair evaluation requires both models and metrics to be aware of translation guidelines. Finally, we call for clear guidelines during dataset creation and for the development of controllable, guideline-aware evaluation frameworks for UGC translation.
>
---
#### [replaced 055] Learning Query-Specific Rubrics from Human Preferences for DeepResearch Report Generation
- **分类: cs.CL**

- **简介: 该论文属于报告生成任务，旨在解决深度研究报告评估中缺乏有效奖励信号的问题。通过学习查询相关的评分标准，提升生成质量。**

- **链接: [https://arxiv.org/pdf/2602.03619](https://arxiv.org/pdf/2602.03619)**

> **作者:** Changze Lv; Jie Zhou; Wentao Zhao; Jingwen Xu; Shihan Dou; Zisu Huang; Muzhao Tian; Xiaohua Wang; Yang Liu; Pluto Zhou; Tao Gui; Le Tian; Xiao Zhou; Xiaoqing Zheng; Xuanjing Huang; Jie Zhou
>
> **摘要:** Nowadays, developing reliable DeepResearch-style long-form report generation remains challenging, as training and evaluation lack verifiable reward signals. Accordingly, rubric-based evaluation has become a common practice. However, existing approaches either rely on coarse, pre-defined rubrics that lack sufficient granularity or depend on manually constructed query-specific rubrics that are costly and difficult to scale. In this paper, we propose a pipeline to train preference-grounded query-specific rubric generators tailored for DeepResearch report generation. We first construct a dataset of DeepResearch-style queries annotated with human preferences over paired reports, and train rubric generators via reinforcement learning with a hybrid reward combining preference consistency, format validity, and LLM-based rubric evaluation. We evaluate the resulting rubric generators in two stages. First, on a held-out human-preference test set, the learned rubrics discriminate preferred from rejected reports more effectively than generic, prompted, or SFT-trained rubric alternatives. Second, when used as reward signals to train DeepResearch systems, our rubric generators yield substantial performance gains under both a simple single-agent ReAct framework and a complex multi-agent workflow on the DeepResearch Bench.
>
---
#### [replaced 056] StepPO: Step-Aligned Policy Optimization for Agentic Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM代理在多步骤决策中的粒度不匹配问题。提出StepPO方法，通过步骤对齐策略优化，提升代理性能。**

- **链接: [https://arxiv.org/pdf/2604.18401](https://arxiv.org/pdf/2604.18401)**

> **作者:** Daoyu Wang; Qingchuan Li; Mingyue Cheng; Jie Ouyang; Shuo Yu; Qi Liu; Enhong Chen
>
> **摘要:** Agentic reinforcement learning (RL) is emerging as a critical post-training paradigm for improving LLM agent capabilities. Existing RL algorithms for LLMs largely follow the token-centric paradigm as in RLHF and RLVR, where tokens serve as the basic units for modeling and optimization. However, this paradigm introduces a granularity mismatch in agentic RL, as it optimizes token-level predictions while LLM agents make step-level decisions through cycles of environmental observations and actions. To bridge this gap, we propose \textbf{StepPO}, a step-centric paradigm for agentic RL via step-aligned policy optimization. Specifically, we reformulate agentic RL from a token-level Markov Decision Process (MDP) into a step-level MDP, where interaction steps serve as the basic trajectory representations. We further propose step-level credit assignment to align policy optimization with the natural granularity of agent decisions. Together, StepPO optimizes agent policies at the step level for multi-turn agent-environment interaction. Experiments across multi-hop QA, academic paper search, and text-world action tasks show that StepPO consistently outperforms various RL algorithms. Further analyses provide insights into how step-centric paradigm improves agent training. We hope this step-centric paradigm offers a useful lens for understanding agent behavior and a practical path for training more capable LLM agents.
>
---
#### [replaced 057] MAVL: A Multilingual Audio-Video Lyrics Dataset for Animated Song Translation
- **分类: cs.CL; cs.LG; cs.MM; cs.SD; eess.AS**

- **简介: 该论文属于歌词翻译任务，旨在解决跨语言歌词的语义传递与音乐节奏保持问题。构建了多模态数据集MAVL，并提出SylAVL-CoT模型，提升翻译的可唱性和准确性。**

- **链接: [https://arxiv.org/pdf/2505.18614](https://arxiv.org/pdf/2505.18614)**

> **作者:** Woohyun Cho; Youngmin Kim; Sunghyun Lee; Youngjae Yu
>
> **备注:** Accepted to EMNLP 2025, Project Page: this https URL, our codes and datasets are available at this https URL
>
> **摘要:** Lyrics translation requires both accurate semantic transfer and preservation of musical rhythm, syllabic structure, and poetic style. In animated musicals, the challenge intensifies due to alignment with visual and auditory cues. We introduce Multilingual Audio-Video Lyrics Benchmark for Animated Song Translation (MAVL), the first multilingual, multimodal benchmark for singable lyrics translation. By integrating text, audio, and video, MAVL enables richer and more expressive translations than text-only approaches. Building on this, we propose Syllable-Constrained Audio-Video LLM with Chain-of-Thought SylAVL-CoT, which leverages audio-video cues and enforces syllabic constraints to produce natural-sounding lyrics. Experimental results demonstrate that SylAVL-CoT significantly outperforms text-based models in singability and contextual accuracy, emphasizing the value of multimodal, multilingual approaches for lyrics translation.
>
---
#### [replaced 058] Benchmarking Large Language Models for Cryptanalysis and Side-Channel Vulnerabilities
- **分类: cs.CL**

- **简介: 该论文属于密码分析任务，研究LLMs在解密和侧信道攻击中的表现，评估其在不同加密算法下的解密能力与局限性。**

- **链接: [https://arxiv.org/pdf/2505.24621](https://arxiv.org/pdf/2505.24621)**

> **作者:** Utsav Maskey; Chencheng Zhu; Usman Naseem
>
> **备注:** EMNLP'25 Findings
>
> **摘要:** Recent advancements in large language models (LLMs) have transformed natural language understanding and generation, leading to extensive benchmarking across diverse tasks. However, cryptanalysis - a critical area for data security and its connection to LLMs' generalization abilities - remains underexplored in LLM evaluations. To address this gap, we evaluate the cryptanalytic potential of state-of-the-art LLMs on ciphertexts produced by a range of cryptographic algorithms. We introduce a benchmark dataset of diverse plaintexts, spanning multiple domains, lengths, writing styles, and topics, paired with their encrypted versions. Using zero-shot and few-shot settings along with chain-of-thought prompting, we assess LLMs' decryption success rate and discuss their comprehension abilities. Our findings reveal key insights into LLMs' strengths and limitations in side-channel scenarios and raise concerns about their susceptibility to under-generalization-related attacks. This research highlights the dual-use nature of LLMs in security contexts and contributes to the ongoing discussion on AI safety and security.
>
---
#### [replaced 059] Last Layer Logits to Logic: Empowering LLMs with Logic-Consistent Structured Knowledge Reasoning
- **分类: cs.CL**

- **简介: 该论文属于知识图谱问答任务，旨在解决LLMs在结构化知识推理中的逻辑漂移问题。通过设计Logits-to-Logic框架，提升推理逻辑一致性。**

- **链接: [https://arxiv.org/pdf/2511.07910](https://arxiv.org/pdf/2511.07910)**

> **作者:** Songze Li; Zhiqiang Liu; Zhaoyan Gong; Xiaoke Guo; Zhongpu Bo; Zhengke Gui; Lei Liang; Huajun Chen; Wen Zhang
>
> **备注:** EMNLP 2026 Submission
>
> **摘要:** Large Language Models (LLMs) achieve excellent performance in natural language reasoning tasks through pre-training on vast unstructured text, enabling them to understand the logic in natural language and generate logic-consistent responses. However, the representational differences between unstructured and structured knowledge make LLMs inherently struggle to maintain logic consistency, leading to \textit{Logic Drift} challenges in structured knowledge reasoning tasks such as Knowledge Graph Question Answering (KGQA). Existing methods address this limitation by designing complex workflows embedded in prompts to guide LLM reasoning. Nevertheless, these approaches only provide input-level guidance and fail to fundamentally address the \textit{Logic Drift} in LLM outputs. Additionally, their inflexible reasoning workflows cannot adapt to different tasks and knowledge graphs. To enhance LLMs' logic consistency in structured knowledge reasoning, we specifically target the logits output from the autoregressive generation process. We propose the \textit{Logits-to-Logic} framework, which incorporates logits strengthening and logits filtering as core modules to correct logical defects in LLM outputs. Extensive experiments show that our approach significantly improves LLMs' logic consistency in structured knowledge reasoning and achieves state-of-the-art performance on multiple KGQA benchmarks.
>
---
#### [replaced 060] Vision-Language Models Mistake Head Orientation for Gaze Direction: Nonverbal Conversation Cues
- **分类: cs.CV; cs.CL**

- **简介: 该论文研究视觉-语言模型在判断凝视方向时的偏差问题，属于视觉-语言理解任务。通过实验发现模型依赖头部朝向而非眼睛方向，导致性能低于人类。**

- **链接: [https://arxiv.org/pdf/2506.05412](https://arxiv.org/pdf/2506.05412)**

> **作者:** Zory Zhang; Pinyuan Feng; Bingyang Wang; Tianwei Zhao; Suyang Yu; Qingying Gao; Hokin Deng; Ziqiao Ma; Yijiang Li; Dezhi Luo
>
> **备注:** Accepted by ACL 2026. Project page at this https URL
>
> **摘要:** Where someone looks is a nonverbal communication cue that children and adults readily use. How well can Vision-Language Models (VLMs) infer gaze targets? To construct evaluation stimuli, we captured 1,360 real-world photos of scenes in which a person gazes at one of several objects on a table. Importantly, we also controlled the gazer's head orientation: sometimes it was directed toward the gaze target, sometimes toward a distractor object, and sometimes left unconstrained. We found a substantial performance gap between VLMs and humans, ruled out alternative explanations such as resolution and object-naming skills, and identified the main reason for the gap as VLMs inferring gaze direction using head orientation rather than eye appearance. Such a bias is likely due to data rather than architecture, as suggested by a proof-of-concept experiment finetuning a transformer-based vision model. Future work should investigate whether these findings hold broadly across various deep learning methods trained on existing data, and whether better data mitigates this problem for all architectures. Pinpointing the reason sets the stage for technologies that can interpret gaze targets to have more efficient interactions with humans.
>
---
#### [replaced 061] Adaptive Querying with AI Persona Priors
- **分类: stat.ML; cs.CL; cs.LG**

- **简介: 该论文属于贝叶斯自适应查询任务，旨在解决高维、异质环境下用户属性估计问题。通过引入AI人格模型，构建高效先验与后验更新机制，提升查询效率与预测准确性。**

- **链接: [https://arxiv.org/pdf/2605.00696](https://arxiv.org/pdf/2605.00696)**

> **作者:** Kaizheng Wang; Yuhang Wu; Assaf Zeevi
>
> **备注:** ICML 2026
>
> **摘要:** We study adaptive querying for learning user-dependent quantities of interest, such as responses to held-out items and psychometric indicators, within tight query budgets. Classical Bayesian design and computerized adaptive testing typically rely on restrictive parametric assumptions or expensive posterior approximations, limiting their use in heterogeneous, high-dimensional, and cold-start settings. We introduce a persona-induced latent variable model that represents a user's state through membership in a finite dictionary of AI personas, each offering response distributions produced by a large language model. This yields expressive priors with closed-form posterior updates and efficient finite-mixture predictions, enabling scalable Bayesian design for sequential item selection. Experiments on synthetic data and WorldValuesBench demonstrate that persona-based posteriors deliver accurate probabilistic predictions and an interpretable adaptive elicitation pipeline.
>
---
#### [replaced 062] Bridging What the Model Thinks and How It Speaks: Expressive Speech Generation via Self-Aware Intent-Realization Alignment
- **分类: cs.CL**

- **简介: 该论文属于语音生成任务，旨在解决语义理解与语音表达不一致的问题。提出SASLM框架，通过自感知对齐实现无监督的表达性语音生成。**

- **链接: [https://arxiv.org/pdf/2604.11424](https://arxiv.org/pdf/2604.11424)**

> **作者:** Kuang Wang; Lai Wei; Ping Lin; Qibing Bai; Wenkai Fang; Li Zhou; Feng Jiang; Zhongjie Jiang; Jun Huang; Yannan Wang; Haizhou Li
>
> **备注:** Submitted to EMNLP 2026. Project page: this https URL
>
> **摘要:** Speech Language Models (SLMs) exhibit strong semantic understanding, yet often fail to translate this capacity into expressive acoustic realization, producing speech with flattened prosody and misaligned emotion. We identify this mismatch as the semantic understanding-acoustic realization gap. Existing approaches typically rely on externally specified proxies, such as emotion labels or style prompts, which require annotations and struggle to capture dynamically evolving expressive intent throughout dialogue. To overcome these limitations, we propose SASLM (Self-Aware Speech Language Model), a proxy-free framework that bridges what the model thinks and how it speaks through self-aware intent-realization alignment: (1) Intent-Aware Bridging self-distills expressive intent from the model's own evolving semantic generation states via a Variational Information Bottleneck (VIB), thereby guiding expressive speech realization without external expressive supervision; while (2) Realization-Aware Alignment reflectively aligns generated acoustics with intended expression through self-reward optimization, progressively improving intent-realization consistency during speech generation. Despite using only 3B parameters and 800 hours of expressive speech data, SASLM achieves state-of-the-art performance on EchoMind among open-source systems, surpassing models over 10 times larger and approaching commercial systems.
>
---
#### [replaced 063] From Unfamiliar to Familiar: Detecting Pre-training Data via Gradient Deviations in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于预训练数据检测任务，旨在解决版权和基准污染问题。通过分析梯度差异，提出GDS方法识别预训练数据。**

- **链接: [https://arxiv.org/pdf/2603.04828](https://arxiv.org/pdf/2603.04828)**

> **作者:** Ruiqi Zhang; Lingxiang Wang; Hainan Zhang; Zhiming Zheng; Yanyan Lan
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** Pre-training data detection for LLMs is essential for addressing copyright concerns and mitigating benchmark contamination. Existing methods mainly focus on the likelihood-based statistical features or heuristic signals before and after fine-tuning, but the former are susceptible to word frequency bias in corpora, and the latter strongly depend on the similarity of fine-tuning data. From an optimization perspective, we observe that during training, samples transition from unfamiliar to familiar in a manner reflected by systematic differences in gradient behavior. Familiar samples exhibit smaller update magnitudes, distinct update locations in model components, and more sharply activated neurons. Based on this insight, we propose GDS, a method that identifies pre-training data by probing Gradient Deviation Scores of target samples. Specifically, we first represent each sample using gradient profiles that capture the magnitude, location, and concentration of parameter updates across FFN and Attention modules, revealing consistent distinctions between member and non-member data. These features are then fed into a lightweight classifier to perform binary membership inference. Experiments on five public datasets show that GDS achieves state-of-the-art performance with significantly improved cross-dataset transferability over strong baselines. Further interpretability analyses reveal differences in gradient distributions, and the semi-supervised results offer a practical way to detect pre-training data.
>
---
#### [replaced 064] "Do Not Mention This to the User": Detecting and Understanding Malicious Agent Skills
- **分类: cs.CR; cs.AI; cs.CL; cs.ET**

- **简介: 该论文属于安全检测任务，旨在识别和分析LLM代理中的恶意技能。通过静态与动态方法检测出157个恶意技能，揭示其攻击策略及漏洞，推动LLM代理生态安全研究。**

- **链接: [https://arxiv.org/pdf/2602.06547](https://arxiv.org/pdf/2602.06547)**

> **作者:** Yi Liu; Zhihao Chen; Yanjun Zhang; Gelei Deng; Yuekang Li; Jianting Ning; Leo Yu Zhang
>
> **备注:** Accepted to the 35th USENIX Security Symposium (USENIX Security 2026)
>
> **摘要:** LLM-based coding agents increasingly rely on third-party extensions called skills, which bundle natural language instructions and helper scripts that execute with full user privileges. Community registries have emerged to distribute these skills, but the security implications remain unstudied due to the absence of labeled threat data. This paper presents a systematic security analysis of 98,380 skills collected from two major registries. Through a combination of static pattern matching and dynamic behavioral verification, we identify 157 skills exhibiting confirmed malicious behavior, encompassing 632 distinct vulnerabilities across 13 attack techniques. Our analysis reveals that these threats are deliberate rather than accidental: each malicious skill contains an average of 4.03 vulnerabilities spanning multiple attack phases. We identify two dominant attack strategies with statistically significant negative correlation -- credential theft via remote code execution, and agent manipulation through adversarial instructions embedded in documentation. Over half of all confirmed cases originate from a single threat actor employing templated brand impersonation at scale. We further observe that attack sophistication correlates with concealment investment, with advanced skills universally employing undocumented capabilities while also exploiting platform-native trust mechanisms. Following responsible disclosure, registry maintainers removed all 157 (100%) of the reported skills. Our dataset and detection pipeline are publicly available to facilitate future research on securing LLM agent ecosystems.
>
---
#### [replaced 065] Lessons from the Trenches on Reproducible Evaluation of Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的模型评估任务，旨在解决语言模型评估的可重复性和透明性问题。作者基于多年经验，总结挑战并提出改进建议。**

- **链接: [https://arxiv.org/pdf/2405.14782](https://arxiv.org/pdf/2405.14782)**

> **作者:** Stella Biderman; Hailey Schoelkopf; Lintang Sutawika; Leo Gao; Jonathan Tow; Baber Abbasi; Alham Fikri Aji; Pawan Sasanka Ammanamanchi; Sidney Black; Jordan Clive; Anthony DiPofi; Julen Etxaniz; Benjamin Fattori; Jessica Zosa Forde; Charles Foster; Jeffrey Hsu; Mimansa Jaiswal; Wilson Y. Lee; Haonan Li; Charles Lovering; Niklas Muennighoff; Ellie Pavlick; Jason Phang; Aviya Skowron; Samson Tan; Xiangru Tang; Kevin A. Wang; Genta Indra Winata; François Yvon; Andy Zou
>
> **摘要:** Reliable evaluation of language models (LMs) remains an open challenge. Re- searchers and engineers face methodological issues such as the sensitivity of models to evaluation setup, difficulty of proper comparisons across methods, and the lack of reproducibility and transparency. Evaluation difficulties are exacer- bated by the fracturing and siloing of information about conventions and common practices. In this paper we draw on three years of experience in evaluating large lan- guage models (LMs) as developers of the popular Language Model Evaluation Harness (lm-eval) (Gao et al., 2023) framework to provide guidance and lessons for the field moving forward. We document a variety of challenges faced by prac- titioners and provide concrete instances where these challenges or the absence of best practices have come into effect. We make recommendations to the field for improving evaluation rigor and confidence, and attempt to codify much of the tacit or folk knowledge surrounding LM evaluation, for a solid ground to move forward.
>
---
#### [replaced 066] Constitutional Black-Box Monitoring for Scheming in LLM Agents
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于安全监控任务，旨在检测LLM代理的隐秘不良行为。通过合成数据训练黑盒监测器，验证其在真实环境中的泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.00829](https://arxiv.org/pdf/2603.00829)**

> **作者:** Simon Storf; Rich Barton-Cooper; James Peters-Gill; Marius Hobbhahn
>
> **备注:** Accepted at ICML 2026. Camera-ready version
>
> **摘要:** Safe deployment of Large Language Model (LLM) agents in autonomous settings requires reliable oversight mechanisms. A central challenge is detecting scheming, where agents covertly pursue misaligned goals. One approach to mitigating such risks is LLM-based monitoring: using language models to examine agent behaviors for suspicious actions. We study constitutional black-box monitors: prompted classifiers that detect scheming using only externally observable inputs and outputs, optimized on synthetic data generated from natural-language behavior specifications. We introduce two pipelines for generating synthetic agent trajectories, STRIDE (iterative refinement) and Gloom (agent-environment simulation), from which we generate 1,000 samples each. We optimize frontier LLM monitors on these datasets via prompt sweeps, human refinement, and automated prompt optimization, and evaluate performance on 7,500 held-out trajectories from ControlArena, a suite of grounded environments where agents operate in more realistic contexts. Our results demonstrate that monitors selected purely on synthetic data can generalize to more realistic environments, capturing a meaningful scheming signal. However, we find that performance saturates quickly in our setting, with simple prompt sweeps matching the results of more extensive optimization. Pushing beyond this limit yields no further improvements and instead leads to overfitting.
>
---
#### [replaced 067] Seeing Through the MiRAGE: Evaluating Multimodal Retrieval Augmented Generation
- **分类: cs.CL; cs.CV; cs.IR**

- **简介: 该论文属于多模态RAG评估任务，解决现有评估体系不适用于多模态问题。提出MiRAGE框架，包含InfoF1和CiteF1，用于评估生成内容的事实性和引用完整性。**

- **链接: [https://arxiv.org/pdf/2510.24870](https://arxiv.org/pdf/2510.24870)**

> **作者:** Alexander Martin; William Walden; Reno Kriz; Dengjia Zhang; Kate Sanders; Eugene Yang; Chihsheng Jin; Benjamin Van Durme
>
> **备注:** this https URL
>
> **摘要:** We introduce MiRAGE, an evaluation framework for retrieval-augmented generation (RAG) from multimodal sources. As audiovisual media becomes a prevalent source of information online, it is essential for RAG systems to integrate information from these sources into generation. However, existing evaluations for RAG are text-centric, limiting their applicability to multimodal settings. MiRAGE is a claim-centric approach to multimodal RAG evaluation, consisting of InfoF1, which assesses factuality and information coverage, and CiteF1, which assesses citation support and completeness. We show that, when applied by humans, MiRAGE strongly aligns with extrinsic judgments of output quality. We additionally introduce an automatic implementation of MiRAGE as well as multimodal variants of three prominent text-based RAG metrics -- ALCE, ARGUE, and RAGAS -- demonstrating the limitations of text-centric work and laying the groundwork for automatic evaluation. We release open-source implementations and outline evaluation methods for multimodal RAG.
>
---
#### [replaced 068] Med-V1: Small Language Models for Zero-shot and Scalable Biomedical Evidence Attribution
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于 biomedical evidence attribution 任务，旨在解决证据归属与幻觉检测问题。提出 Med-V1 小型模型，高效准确地进行医学事实验证与错误识别。**

- **链接: [https://arxiv.org/pdf/2603.05308](https://arxiv.org/pdf/2603.05308)**

> **作者:** Qiao Jin; Yin Fang; Lauren He; Yifan Yang; Guangzhi Xiong; Zhizheng Wang; Nicholas Wan; Joey Chan; Donald C. Comeau; Robert Leaman; Charalampos S. Floudas; Aidong Zhang; Michael F. Chiang; Yifan Peng; Zhiyong Lu
>
> **摘要:** Assessing whether an article supports an assertion is essential for hallucination detection and claim verification. While large language models (LLMs) have the potential to automate this task, achieving strong performance requires frontier models such as GPT-5 that are prohibitively expensive to deploy at scale. To efficiently perform biomedical evidence attribution, we present Med-V1, a family of small language models with only three billion parameters. Trained on high-quality synthetic data newly developed in this study, Med-V1 substantially outperforms (+27.0% to +71.3%) its base models on five biomedical benchmarks unified into a verification format. Despite its smaller size, Med-V1 performs comparably to frontier LLMs such as GPT-5, along with high-quality explanations for its predictions. We use Med-V1 to conduct a first-of-its-kind use case study that quantifies hallucinations in LLM-generated answers under different citation instructions. Results show that the format instruction strongly affects citation validity and hallucination, with GPT-5 generating more claims but exhibiting hallucination rates similar to GPT-4o. Additionally, we present a second use case showing that Med-V1 can automatically identify high-stakes evidence misattributions in clinical practice guidelines, revealing potentially negative public health impacts that are otherwise challenging to identify at scale. Overall, Med-V1 provides an efficient and accurate lightweight alternative to frontier LLMs for practical and real-world applications in biomedical evidence attribution and verification tasks. Med-V1 is available at this https URL.
>
---
#### [replaced 069] "Înţelegi Româneşte?'' A Recipe for Romanian Vision-Language Models
- **分类: cs.CL**

- **简介: 该论文属于视觉语言模型任务，旨在解决低资源语言（如罗马尼亚语）中VLM性能下降的问题。通过构建罗马尼亚语数据集并调整模型架构，提升其在本地化场景下的表现。**

- **链接: [https://arxiv.org/pdf/2605.31401](https://arxiv.org/pdf/2605.31401)**

> **作者:** Mihai Masala; Marius Leordeanu; Mihai Dascalu; Traian Rebedea
>
> **摘要:** Vision-Language Models (VLMs) largely follow the text-only LLM trajectory, excelling on English benchmarks but sharply degrading on low-resource languages, where neither large-scale image-text corpora nor culturally grounded evaluations exist. We present a systematic study of building a language-specific VLM for Romanian, covering the full pipeline from data construction to architectural choices. We translate established English VLM training and evaluation corpora into Romanian, applying machine translation to textual annotations and to in-image text, preserving visual grounding while adapting the textual content. Using this data, we train and ablate a series of VLMs to isolate the contribution of (i) vision backbones of varying scale and pretraining, (ii) language backbones from multilingual to Romanian-adapted LLMs, and (iii) OCR-style image-text data. We further curate HoraVQA, a culturally native evaluation set grounded in Romanian everyday scenes. Romanian-adapted VLMs consistently outperform their same-sized counterparts and, across all evaluated benchmarks, even surpass models from the next larger size category.
>
---
#### [replaced 070] Position: the Stochastic Parrot in the Coal Mine. Model Collapse is a Threat to Low-Resource Communities
- **分类: cs.LG; cs.CL; cs.CY**

- **简介: 该论文属于AI伦理研究任务，探讨模型坍塌对低资源社区的影响，分析其导致的数据退化和文化偏见问题，并提出缓解方向。**

- **链接: [https://arxiv.org/pdf/2605.04127](https://arxiv.org/pdf/2605.04127)**

> **作者:** Devon Jarvis; Richard Klein; Benjamin Rosman; Steven James; Stefano Sarao Mannelli
>
> **备注:** 14 pages, 1 figure, 1 table, International Conference on Machine Learning
>
> **摘要:** Model collapse, the degradation in performance that arises when generative models are trained on the outputs of prior models, is an increasing concern as artificially generated content proliferates. Related critiques of large language models have highlighted their tendency to reproduce frequent patterns in training data, their reliance on vast datasets, and their substantial environmental cost. Together, these factors contribute to data degradation, the reinforcement of cultural biases, and inefficient resource use. In this position paper we aim to combine these views and argue that model collapse threatens current efforts to democratize AI. By reducing training efficiency and skewing data distributions away from the tails of their support, model collapse disproportionately impacts low-resource and marginalized communities. We examine both the environmental and cultural implications of this phenomenon, situate our position within recent position papers on model collapse, and conclude with a call to action. Finally, we outline initial directions for mitigating these effects.
>
---
#### [replaced 071] StreamingVLM: Real-Time Understanding for Infinite Video Streams
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出StreamingVLM，解决实时处理无限视频流的挑战，通过优化注意力机制实现低延迟和稳定性能。属于视觉语言理解任务。**

- **链接: [https://arxiv.org/pdf/2510.09608](https://arxiv.org/pdf/2510.09608)**

> **作者:** Ruyi Xu; Guangxuan Xiao; Yukang Chen; Liuning He; Yao Lu; Song Han
>
> **备注:** Published as a conference paper at ICLR 2026. The first two authors contributed equally to this work
>
> **摘要:** Vision-language models (VLMs) could power real-time assistants and autonomous agents, but they face a critical challenge: understanding near-infinite video streams without escalating latency and memory usage. Processing entire videos with full attention leads to quadratic computational costs and poor performance on long videos. Meanwhile, simple sliding window methods are also flawed, as they either break coherence or suffer from high latency due to redundant recomputation. In this paper, we introduce StreamingVLM, a model designed for real-time, stable understanding of infinite visual input. Our approach is a unified framework that aligns training with streaming inference. During inference, we maintain a compact KV cache by reusing states of attention sinks, a short window of recent vision tokens, and a long window of recent text tokens. This streaming ability is instilled via a simple supervised fine-tuning (SFT) strategy that applies full attention on short, overlapped video chunks, which effectively mimics the inference-time attention pattern without training on prohibitively long contexts. For evaluation, we build Inf-Streams-Eval, a new benchmark with videos averaging over two hours that requires dense, per-second alignment between frames and text. On Inf-Streams-Eval, StreamingVLM achieves a 66.18% win rate against GPT-4O mini and maintains stable, real-time performance at up to 8 FPS on a single NVIDIA H100. Notably, our SFT strategy also enhances general VQA abilities without any VQA-specific fine-tuning, improving performance on LongVideoBench by +4.30 and OVOBench Realtime by +5.96. Code is available at this https URL.
>
---
#### [replaced 072] Disentangling Similarity and Relatedness in Topic Models
- **分类: cs.CL**

- **简介: 该论文属于主题模型任务，旨在解决PLM增强模型与传统模型在语义结构上的差异问题。通过构建基准测试，分析主题模型在相似性与相关性上的表现，提供评估其语义结构的诊断方法。**

- **链接: [https://arxiv.org/pdf/2603.10619](https://arxiv.org/pdf/2603.10619)**

> **作者:** Hanlin Xiao; Yang Wang; Mauricio A. Álvarez; Rainer Breitling
>
> **备注:** 26 pages, 9 figures, 18 tables
>
> **摘要:** The recent success of large pre-trained language models (PLMs) has motivated their integration into topic modeling. However, PLM-augmented topic models differ from classical co-occurrence models such as Latent Dirichlet Allocation (LDA) not only in performance, but also in the type of semantic structure they capture. We formalize this distinction along two psycholinguistic axes: thematic relatedness (dog/bone) and taxonomic similarity (dog/wolf). To measure both axes over topic words, we construct a large synthetic benchmark of word pairs using LLM-based annotation and train a neural scorer on it. Across multiple corpora and model families, the scorer places different topic-model families at distinct positions within the joint similarity-relatedness space. The two scores further predict downstream task performance: tasks requiring similarity benefit from similarity-rich topics, whereas tasks requiring relatedness benefit from the converse, and excessive emphasis on either axis degrades performance on tasks aligned with the opposing semantic structure. Neither axis is uniformly beneficial. Measuring both therefore provides a practical, model-agnostic diagnostic for evaluating the semantic structure captured by topic models.
>
---
#### [replaced 073] A Reproducible Universal Dependencies-Style Pipeline for Katharevousa Greek Parliamentary Text
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决Katharevousa希腊语议会文本的句法分析问题。通过构建可复现的解析流程，提升历史文本的NLP处理能力。**

- **链接: [https://arxiv.org/pdf/2605.22978](https://arxiv.org/pdf/2605.22978)**

> **作者:** George Mikros; Fotios Fitsilis
>
> **备注:** 12 pages, 1 figure, 2 tables; companion to the kathnlp open-source release at this https URL
>
> **摘要:** Katharevousa Greek remains poorly served by contemporary NLP pipelines despite its importance for legal, administrative, and parliamentary archives. We present a reproducible workflow for building and evaluating a Universal Dependencies-style parsing resource for Katharevousa parliamentary questions from Greece's early post-junta period. The pipeline links OCR-aware reconstruction, schema-constrained LLM-assisted annotation, automatic validation, deterministic CoNLL-U snapshotting, fixed-split evaluation, and model-family comparison. The frozen automatically validated reference set contains 1{,}697 sentences, split into 1{,}357 training sentences and 340 held-out test sentences. We compare off-the-shelf Greek and Ancient Greek parsers, a feature-based parser, mBERT, XLM-R, and custom Stanza training under the same scoring protocol. Off-the-shelf systems show substantial register mismatch: the strongest external baseline, spaCy Greek, reaches 0.4183 LAS. The best structural parser, an XLM-R model, reaches 0.8893 UPOS accuracy, 0.7250 dependency-relation F1, 0.6098 UAS, and 0.5162 LAS, an absolute LAS gain of 0.0980 over the best external baseline. The feature-based model remains competitive for UPOS and relation labeling, indicating that transparent lexical-context features still matter at this data scale. Beyond scores, the paper contributes an auditable methodology for turning difficult historical parliamentary OCR into reusable syntactic NLP infrastructure. The entire pipeline -- code, schema, frozen reference annotations, fixed train/test split, and per-model benchmark reports -- is released as an open-access companion to this paper.
>
---
#### [replaced 074] The Social Cost of Intelligence: Emergence, Propagation, and Amplification of Stereotypical Bias in Multi-Agent Systems
- **分类: cs.MA; cs.CL**

- **简介: 该论文研究多智能体系统中偏见的产生、传播与放大问题，属于AI公平性任务。通过构建评估框架，分析不同条件下偏见动态，揭示通信对偏见的影响及系统脆弱性。**

- **链接: [https://arxiv.org/pdf/2510.10943](https://arxiv.org/pdf/2510.10943)**

> **作者:** Thi-Nhung Nguyen; Linhao Luo; Amardeep Kaur; Rollin Omari; Tamas Abraham; Junae Kim; Thuy-Trang Vu; Dinh Phung
>
> **摘要:** Bias in large language models (LLMs) remains a persistent challenge, often leading to stereotyping and unfair treatment across social groups. While prior work has mainly focused on individual LLMs, the emergence of multi-agent systems (MAS), where multiple LLMs collaborate and communicate, introduces new and underexplored dynamics in how bias emerges, propagates, and amplifies. To systematically investigate these dynamics, we propose a simple evaluation framework with three agent-level metrics that quantify bias emergence, propagation, and amplification throughout multi-agent interaction. We evaluate MAS across three bias benchmarks under varying LLM backbones, social-group configurations, communication behaviors, and adversarial settings. Our results show that communication can trigger up to 70\% new bias emergence, propagate bias across over 80\% of agents, and amplify stereotypes by more than 3$\times$. We further find that denser and competitive communication generally increases bias. Finally, we demonstrate that MAS are highly vulnerable to simple bias injection attacks, and existing defense strategies provide only limited protection. Our findings provide important insights into the fairness and robustness of multi-agent LLM systems.
>
---
#### [replaced 075] ToMAP: Training Opponent-Aware LLM Persuaders with Theory of Mind
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于说服任务，旨在提升LLM在对话中的对手意识。通过引入ToMAP模型，增强其理论心智能力，使说服更有效且多样化。**

- **链接: [https://arxiv.org/pdf/2505.22961](https://arxiv.org/pdf/2505.22961)**

> **作者:** Peixuan Han; Zijia Liu; Jiaxuan You
>
> **摘要:** Large language models (LLMs) have shown promising potential in persuasion, but existing works on training LLM persuaders are still preliminary. Notably, while humans are skilled in modeling their opponent's thoughts and opinions proactively and dynamically, current LLMs struggle with such Theory of Mind (ToM) reasoning, resulting in limited diversity and opponent awareness. To address this limitation, we introduce Theory of Mind Augmented Persuader (ToMAP), a novel approach for building more flexible persuader agents by incorporating two theory of mind modules that enhance the persuader's awareness and analysis of the opponent's mental state. Specifically, we begin by prompting the persuader to consider possible objections to the target central claim, and then use a text encoder paired with a trained MLP classifier to predict the opponent's current stance on these counterclaims. Our carefully designed reinforcement learning schema enables the persuader learns how to analyze opponent-related information and utilize it to generate more effective arguments. Experiments show that the ToMAP persuader, while containing only 3B parameters, outperforms much larger baselines, like GPT-4o, with a relative gain of 39.4% across multiple persuadee models and diverse corpora. Notably, ToMAP exhibits complex reasoning chains and reduced repetition during training, which leads to more diverse and effective arguments. The opponent-aware feature of ToMAP also makes it suitable for long conversations and enables it to employ more logical and opponent-aware strategies. These results underscore our method's effectiveness and highlight its potential for developing more persuasive language agents. Code is available at: this https URL.
>
---
#### [replaced 076] Acoustic and perceptual differences between standard and accented speech and their voice clones
- **分类: cs.SD; cs.AI; cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于语音处理任务，研究标准与带口音汉语及其语音克隆的声学和感知差异。旨在解决语音克隆中口音保留问题，通过计算与感知实验发现口音影响克隆相似度和可懂度。**

- **链接: [https://arxiv.org/pdf/2604.01562](https://arxiv.org/pdf/2604.01562)**

> **作者:** Tianle Yang; Chengzhe Sun; Phil Rose; Siwei Lyu
>
> **摘要:** Voice cloning is often evaluated in terms of overall quality, but less is known about accent preservation and its perceptual consequences. We compare standard and heavily accented Mandarin speech and their voice clones using a combined computational and perceptual design. Embedding-based analyses showed larger original-clone distances for accented speakers in several speaker-discriminative embedding spaces, but this difference disappeared after normalizing against each speaker's within-original baseline variability. In the perception study, clones are rated as more similar to their originals for standard than for accented speakers, and intelligibility increases from original to clone, with a larger gain for accented speech. These results show that accent variation can shape perceived identity match and intelligibility in voice cloning even when it is not reflected in baseline-normalized speaker-embedding distance, and they motivate treating accent preservation as an explicit component of speaker identity preservation, rather than assuming that it is fully captured by off-the-shelf speaker-discriminative embeddings.
>
---
#### [replaced 077] SciAgentGym: Benchmarking Multi-Step Scientific Tool-use in LLM Agents
- **分类: cs.CL**

- **简介: 该论文提出SciAgentGym，用于评估大模型在科学工具使用中的多步骤推理能力，解决当前基准不足的问题，并通过数据合成提升模型表现。**

- **链接: [https://arxiv.org/pdf/2602.12984](https://arxiv.org/pdf/2602.12984)**

> **作者:** Yujiong Shen; Yajie Yang; Zhiheng Xi; Binze Hu; Huayu Sha; Jiazheng Zhang; Qiyuan Peng; Junlin Shang; Jixuan Huang; Yutao Fan; Jingqi Tong; Shihan Dou; Ming Zhang; Lei Bai; Zhenfei Yin; Tao Gui; Xingjun Ma; Qi Zhang; Xuanjing Huang; Yu-Gang Jiang
>
> **摘要:** Scientific reasoning inherently demands integrating sophisticated toolkits to navigate domain-specific knowledge. Yet, current benchmarks largely overlook agents' ability to orchestrate tools for such rigorous workflows. To bridge this gap, we introduce SciAgentGym, a scalable interactive environment featuring 1,780 domain-specific tools across four natural science disciplines, supported by a robust execution infrastructure. Complementing this, we present SciAgentBench, a tiered evaluation suite designed to stress-test agentic capabilities from elementary actions to long-horizon workflows. Our evaluation identifies a critical bottleneck: state-of-the-art models still struggle with complex scientific tool-use, and their performance degrades substantially as interaction horizons extend. To address this, we propose SciForge, a data synthesis method that models the tool action space as a dependency graph to generate logic-aware training trajectories. By fine-tuning on these trajectories, our SciAgent-8B outperforms the significantly larger Qwen3-VL-235B-Instruct while exhibiting positive cross-domain transfer of scientific tool-use capabilities. These results underscore the promising potential of next-generation autonomous scientific agents.
>
---
#### [replaced 078] A Padding Method for Enhanced Encoding of Inorganic Structures with Varying Chemical Compositions
- **分类: cond-mat.mtrl-sci; cs.CE; cs.CL**

- **简介: 该论文属于材料科学中的生成模型任务，旨在解决复杂无机结构生成的准确性与效率问题。提出一种基于晶体对称性的填充方法，提升无机材料的编码与生成效果。**

- **链接: [https://arxiv.org/pdf/2605.30743](https://arxiv.org/pdf/2605.30743)**

> **作者:** Thang Dang; Haderbache Amir; Tzanakakis Alexandros; Yoshimoto Yuta
>
> **摘要:** Designing novel inorganic materials through generative models remains an important challenge for material science, driven by the complexity and diversity of inorganic structures across expansive chemical compositions and structural landscape. The vast combinatorial space of inorganic compounds demands innovative, AI-driven approaches to overcome limitations in generative accuracy and efficiency. To address this, we introduce a novel method that redefines the encoding and generation of inorganic materials by utilizing domain-specific symmetry-aware representation. Our approach not only refines the representation of intricate inorganic structures but also contributes to the field of material discovery by enhancing the precision and stability of generated candidates. Central to our methodology is a novel padding technique that exploits crystal symmetry information to enhance the encoding process. By integrating Wyckoff position length-aware padding into an encoder architecture, we achieve a more robust informed representation of inorganic materials. This symmetry-driven enhancement improves deep learning models to generate stable, previously unexplored inorganic structures with superior accuracy and computational efficiency. Furthermore, we introduce an end-to-end system that leverages the machine learning potential models to seamlessly generate novel, even those unseen in the training data, and stable inorganic materials from initial data to validated output. This pipeline integrates advanced generative models with stability analysis, marking a significant leap forward in the automated exploration and design of next-generation inorganic materials. Our method improved reconstruction accuracy 5.3% in proton conductor data, and generated 63.5% more novel stable inorganic material to baseline model on the perov-5 dataset.
>
---
#### [replaced 079] Search-on-Graph: Iterative Informed Navigation for Large Language Model Reasoning on Knowledge Graphs
- **分类: cs.CL**

- **简介: 该论文属于知识图谱问答任务，旨在解决LLM在知识图谱中路径选择不准确的问题。提出SoG方法，让LLM自主选择路径，提升推理效果。**

- **链接: [https://arxiv.org/pdf/2510.08825](https://arxiv.org/pdf/2510.08825)**

> **作者:** Jia Ao Sun; Hao Yu; Fabrizio Gotti; Fengran Mo; Yihong Wu; Yuchen Hui; Zhan Su; Lingfeng Xiao; Jian-Yun Nie
>
> **备注:** Accepted to KDD '26 (32nd ACM SIGKDD Conference on Knowledge Discovery and Data Mining)
>
> **摘要:** Large language models (LLMs) augmented with knowledge graphs (KGs) offer a promising approach for knowledge-intensive reasoning. Central to this approach is the selection of appropriate reasoning paths in the KG. Yet, existing methods face a common limitation: reasoning path selection is often performed by separate modules using criteria that are only weakly connected to the reasoning requirements. This often results in selecting incorrect relations or premature pruning of relevant paths. We propose Search-on-Graph (SoG), a method that strengthens the connection between path selection and reasoning by having the LLM itself select which relations to follow, informed by both the available KG structure and the complete reasoning history. SoG follows an \textit{observe-think-navigate} paradigm: at each step, the LLM observes the relational connections available at the current entity, reasons about which path best advances toward answering the question, and navigates accordingly. This context-aware navigation fully exploits the LLM's reasoning capabilities rather than relying on independent selection modules with surrogate criteria. Experiments on six knowledge graph question answering (KGQA) benchmarks demonstrate that SoG outperforms state-of-the-art methods while requiring no task-specific fine-tuning and generalizing across different KG schemas.
>
---
#### [replaced 080] Malaysian English News Decoded: A Linguistic Resource for Named Entity and Relation Extraction
- **分类: cs.CL**

- **简介: 该论文针对马来西亚英语的命名实体识别与关系抽取任务，解决现有数据集不足的问题，构建了MEN数据集并优化了spaCy模型性能。**

- **链接: [https://arxiv.org/pdf/2402.14521](https://arxiv.org/pdf/2402.14521)**

> **作者:** Mohan Raj Chanthran; Lay-Ki Soon; Huey Fang Ong; Bhawani Selvaretnam
>
> **备注:** Accepted at LREC-COLING 2024
>
> **摘要:** Standard English and Malaysian English exhibit notable differences, posing challenges for natural language processing (NLP) tasks on Malaysian English. Unfortunately, most of the existing datasets are mainly based on standard English and therefore inadequate for improving NLP tasks in Malaysian English. An experiment using state-of-the-art Named Entity Recognition (NER) solutions on Malaysian English news articles highlights that they cannot handle morphosyntactic variations in Malaysian English. To the best of our knowledge, there is no annotated dataset available to improvise the model. To address these issues, we constructed a Malaysian English News (MEN) dataset, which contains 200 news articles that are manually annotated with entities and relations. We then fine-tuned the spaCy NER tool and validated that having a dataset tailor-made for Malaysian English could improve the performance of NER in Malaysian English significantly. This paper presents our effort in the data acquisition, annotation methodology, and thorough analysis of the annotated dataset. To validate the quality of the annotation, inter-annotator agreement was used, followed by adjudication of disagreements by a subject matter expert. Upon completion of these tasks, we managed to develop a dataset with 6,061 entities and 3,268 relation instances. Finally, we discuss on spaCy fine-tuning setup and analysis on the NER performance. This unique dataset will contribute significantly to the advancement of NLP research in Malaysian English, allowing researchers to accelerate their progress, particularly in NER and relation extraction. The dataset and annotation guideline has been published on Github.
>
---
#### [replaced 081] Measuring Alignment-Induced Activation Shifts Correctly: A Template-Controlled Difference-in-Differences Protocol
- **分类: cs.LG; cs.CL; stat.ML**

- **简介: 该论文属于模型对齐研究任务，旨在解决激活差异分析中的混淆问题。通过引入模板控制的差分法，正确分离对齐效应与格式影响。**

- **链接: [https://arxiv.org/pdf/2605.24583](https://arxiv.org/pdf/2605.24583)**

> **作者:** Yuki Nakamura
>
> **备注:** 11 pages, 1 figure. v3: substantially revised and reframed as a measurement-methodology paper. Code, data, and an immutable Zenodo archive are available at this https URL (DOI: https://doi.org/10.5281/zenodo.20341444)
>
> **摘要:** Comparing a model's internal activations before and after alignment is a natural way to ask what safety training changes: one forms the matrix of paired aligned-minus-base activations on safety-relevant inputs and reads off its effective rank or top direction. We show the obvious way to form this matrix is confounded. The aligned model is evaluated under a chat template the base model never saw, so the naive difference conflates the alignment shift with chat formatting. We introduce a four-variant decomposition of the modification matrix (naive, template-controlled, within-aligned, and difference-in-differences, DiD) that separates the two effects. Template control alone removes a 2.0-3.9x inflation of the measured effective rank across Llama-3.1-8B, Gemma-2-9B, and Qwen-2.5-7B; the DiD contrast is what recovers the refusal direction of Arditi et al. (2024), lifting its cosine alignment from 0.18-0.39 to 0.50-0.86. Projection-ablation across the three families confirms the recovered subspace is behaviorally active and that singular-value order is not causal order. We validate the protocol on a controlled testbed and distill it into measurement recommendations for activation-difference studies of alignment.
>
---
#### [replaced 082] ActiveUltraFeedback: Efficient Preference Data Generation using Active Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决RLHF中偏好数据获取成本高的问题。通过主动学习方法高效生成高质量偏好数据，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.09692](https://arxiv.org/pdf/2603.09692)**

> **作者:** Davit Melikidze; Marian Schneider; Jessica Lam; Martin Wertich; Ido Hakimi; Barna Pásztor; Andreas Krause
>
> **备注:** 40 pages, 9 figures, 26 tables
>
> **摘要:** Reinforcement Learning from Human Feedback (RLHF) has become the standard for aligning Large Language Models (LLMs), yet its efficacy is bottlenecked by the high cost of acquiring preference data, especially in low-resource and expert domains. To address this, we introduce ACTIVEULTRAFEEDBACK, a modular active learning pipeline that leverages uncertainty estimates to dynamically identify the most informative responses for annotation. Our pipeline facilitates the systematic evaluation of standard response selection methods alongside DOUBLE REVERSE THOMPSON SAMPLING (DRTS) and DELTAUCB, two novel methods prioritizing response pairs with large predicted quality gaps, leveraging recent results showing that such pairs provide good signals for fine-tuning. Our experiments demonstrate that ACTIVEULTRAFEEDBACK yields high-quality datasets that lead to significant improvements in downstream performance, notably achieving comparable or superior results with as little as one-sixth of the annotated data relative to static baselines. Our pipeline is available at this https URL and our preference datasets at this https URL.
>
---
#### [replaced 083] DyLLM: Efficient Diffusion LLM Inference via Saliency-based Token Selection and Partial Attention
- **分类: cs.CL; cs.AI; cs.PF**

- **简介: 该论文提出DyLLM，解决扩散语言模型推理效率低的问题。通过选择性计算显著标记，提升推理速度，保持模型精度。属于高效推理任务。**

- **链接: [https://arxiv.org/pdf/2603.08026](https://arxiv.org/pdf/2603.08026)**

> **作者:** Younjoo Lee; Seungkyun Dan; Junghoo Lee; Jaiyoung Park; Jung Ho Ahn
>
> **备注:** 21 pages, 10 figures, 7 tables, accepted at ICML 2026
>
> **摘要:** Masked diffusion language models enable parallel token decoding, providing a promising alternative to the sequential nature of autoregressive generation. However, their iterative denoising process remains computationally expensive because it repeatedly processes the entire sequence at every step. We observe that across these diffusion steps, most token representations remain stable; only a small subset, which we term salient tokens, contributes meaningfully to the next update. Leveraging this temporal sparsity, we present DyLLM, a training-free inference framework that accelerates decoding by selectively computing only these salient tokens. DyLLM identifies saliency by measuring the cosine similarity of attention contexts between adjacent denoising steps. It recomputes feed-forward and attention operations only for salient tokens while reusing cached activations for the remainder. Across diverse reasoning and code-generation benchmarks, DyLLM achieves up to 9.6x higher throughput while largely preserving the baseline accuracy of representative open-source diffusion LLMs, LLaDA, and Dream.
>
---
#### [replaced 084] SARA: Stress Test Reasoning in Audio Deepfake Detection
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于音频深度伪造检测任务，旨在解决ALM推理可靠性问题。提出SARA框架，评估推理的感知、一致性与矛盾，发现声学攻击降低一致性，而语言攻击保持一致性。**

- **链接: [https://arxiv.org/pdf/2601.03615](https://arxiv.org/pdf/2601.03615)**

> **作者:** Binh Nguyen; Charles Fleming; Thai Le
>
> **备注:** Preprint for ACL 2026 submission
>
> **摘要:** Audio Language Models (ALMs) offer a promising shift towards explainable audio deepfake detections (ADD), moving beyond \textit{black-box} classifiers by providing transparency to their predictions via reasoning traces. However, such reasoning may not support the model predictions, reflecting poor coherence, or, worse, may rationalize incorrect predictions with plausible but misleading explanation. Moreover, the behavior of ALM reasoning under adversarial attacks remains under-explored, raising questions about the practical reliability of such explanation capabilities. To address this gap, this study introduces \textbf{SARA} (\textbf{S}hift \textbf{A}nalysis of \textbf{R}easoning in \textbf{A}udio), a diagnostic framework that evaluates ALM reasoning across three dimensions: acoustic perception, reasoning-verdict coherence and dissonance. We test five open-source ALMs against both acoustic and linguistic adversarial attacks. We show that acoustic attacks significantly degrade reasoning-verdict coherence (average decrease of 14.20\%), frequently inducing internal logical conflicts. Conversely, linguistic attacks achieve higher attack success rates while maintaining reasoning coherence. We further demonstrate that the textual coherence of generated reasoning traces also serves as a latent indicator of adversarial inputs, enabling effective detection of perturbed audio (0.78 in F1) \textit{without accessing the raw acoustic signal}. These findings suggest that reasoning traces provide diagnostic utility that persists even when final classification outputs are compromised.
>
---
#### [replaced 085] Finding the Minimal Parameter Budget for Implicit Reasoning: A Data Complexity Driven Scaling Law for Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究语言模型在预训练中进行隐式推理所需的最小参数量，解决模型容量与数据复杂性匹配问题，通过实验揭示参数与图搜索熵的缩放关系。**

- **链接: [https://arxiv.org/pdf/2504.03635](https://arxiv.org/pdf/2504.03635)**

> **作者:** Xinyi Wang; Shawn Tan; Shenbo Xu; Mingyu Jin; William Yang Wang; Rameswar Panda; Yikang Shen
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Reasoning is a core capability of language models (LMs), yet it remains unclear how much model capacity is necessary to support reasoning during pretraining. In this work, we study the minimal parameter budget required for implicit reasoning, defined as the ability to infer new facts from learned knowledge without explicit chain-of-thought supervision. To isolate this phenomenon, we pretrain LMs from scratch in a controlled synthetic environment that mimics the structure and distribution of real-world knowledge graphs, and evaluate their ability to complete missing edges via multi-hop inference. From both a theoretical and an empirical perspective, we identify a scaling law linking this optimal parameter budget to a graph search entropy measure. Across a wide range of model sizes, training steps, and graph complexities, we show that an optimally sized language model can reliably reason over approximately 0.008 bits of information per parameter at most. Our results characterize the minimal sufficient capacity for implicit reasoning during pretraining. Our findings provide principled guidance for matching model size to data complexity and offer new insights into the scaling behavior of reasoning in large language models.
>
---
#### [replaced 086] MIRROR: A Multi-Agent Framework with Iterative Adaptive Revision and Hierarchical Retrieval for Optimization Modeling in Operations Research
- **分类: cs.CL**

- **简介: 该论文属于运筹优化建模任务，旨在解决自然语言到数学模型自动转换的准确性与可靠性问题。提出MIRROR框架，结合迭代修正与分层检索，提升建模效率与正确性。**

- **链接: [https://arxiv.org/pdf/2602.03318](https://arxiv.org/pdf/2602.03318)**

> **作者:** Yifan Shi; Jiayi Wang; Minyi Wu; Ye Fan; Jialong Shi; Jianyong Sun
>
> **摘要:** Operations Research (OR) relies on expert-driven modeling-a slow and fragile process ill-suited to novel scenarios. While large language models (LLMs) can automatically translate natural language into optimization models, existing approaches either rely on costly post-training or employ multi-agent frameworks, yet most still lack reliable collaborative error correction and task-specific retrieval, often leading to incorrect outputs. We propose MIRROR, a fine-tuning-free, end-to-end multi-agent framework that directly translates natural language optimization problems into mathematical models and solver code. MIRROR integrates two core mechanisms: (1) execution-driven iterative adaptive revision for automatic error correction, and (2) hierarchical retrieval to fetch relevant modeling and coding exemplars from a carefully curated exemplar library. Experiments show that MIRROR outperforms existing methods on standard OR benchmarks, with notable results on complex industrial datasets such as IndustryOR and Mamo-ComplexLP. By combining precise external knowledge infusion with systematic error correction, MIRROR provides non-expert users with an efficient and reliable OR modeling solution, overcoming the fundamental limitations of general-purpose LLMs in expert optimization tasks.
>
---
#### [replaced 087] Assessment of Generative Named Entity Recognition in the Era of Large Language Models
- **分类: cs.CL**

- **简介: 论文研究生成式命名实体识别任务，评估大语言模型在该任务上的表现。通过实验对比传统模型，分析输出格式、记忆能力及微调影响，验证生成式方法的有效性与优势。**

- **链接: [https://arxiv.org/pdf/2601.17898](https://arxiv.org/pdf/2601.17898)**

> **作者:** Qi Zhan; Yile Wang; Hui Huang
>
> **摘要:** Named entity recognition (NER) is evolving from a sequence labeling task into a generative paradigm with the rise of large language models (LLMs). We conduct a systematic evaluation of open-source LLMs on both flat and nested NER tasks. We investigate several research questions including the performance gap between generative NER and traditional NER models, the impact of output formats, whether LLMs rely on memorization, and the preservation of general capabilities after fine-tuning. Through experiments across eight LLMs of varying scales and four standard NER datasets, we find that: (1) With parameter-efficient fine-tuning and structured formats like inline bracketed or XML, open-source LLMs achieve performance competitive with traditional encoder-based models and surpass decoder-based LLMs with in-context learning techniques; (2) The NER capability of LLMs stems from instruction-following and generative power, not mere memorization of entity-label pairs; and (3) Applying NER instruction tuning has minimal impact on general capabilities of LLMs, even improving performance on datasets like DROP by 25.50 to 45.32 F1 points due to enhanced entity understanding. These findings demonstrate that generative NER with LLMs is a promising, user-friendly alternative to traditional methods. We release the data and code at this https URL.
>
---
#### [replaced 088] Reconsidering Positional Supervision in Masked Diffusion Language Model Training
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究MDLM在迭代解码中对位置偏移的敏感性，提出使用CTC优化训练，提升模型鲁棒性。任务为文本生成，解决位置对齐问题。**

- **链接: [https://arxiv.org/pdf/2601.22947](https://arxiv.org/pdf/2601.22947)**

> **作者:** Mengyu Ye; Keito Kudo; Ryosuke Takahashi; Jun Suzuki
>
> **备注:** preprint, WIP
>
> **摘要:** Masked diffusion language models (MDLMs) generate text by unmasking tokens in parallel and have recently emerged as alternatives to autoregressive language models. They can be viewed as parallel decoders trained with a position-wise cross-entropy (CE) loss, the same setup as non-autoregressive translation (NAT). In NAT, CE-trained parallel decoders have been argued to be sensitive to small positional shifts, since CE penalizes them harshly. We ask whether CE-trained MDLMs are similarly sensitive to such shifts under iterative decoding. To probe this, we apply a controlled intervention that introduces them during decoding. On LLaDA-8B-Instruct with Arena-Hard, displacing as little as 1% of generated tokens by one position substantially reduces win rates against the unintervened model, showing that MDLMs are sensitive to such small shifts under iterative parallel decoding. Motivated by this, we adapt connectionist temporal classification (CTC), an alignment-flexible objective known to mitigate it there, to MDLM supervised fine-tuning. By relaxing the strict position-wise match that CE imposes, CTC gives the loss room to absorb small positional shifts; concretely, we modified CTC objective to use a special <slack> token that absorbs positional uncertainty between target tokens and output positions, and a updated collapse map that preserves target surface forms. Across four open-ended generation benchmarks, the resulting model consistently improves over both the original model and a matched cross-entropy-trained baseline, with statistically significant gains on all four. These results identify training-side alignment flexibility as a useful design dimension for MDLM SFT, complementary to the inference-time approaches explored in prior work.
>
---
#### [replaced 089] Herculean: An Agentic Benchmark for Financial Intelligence
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出Herculean基准，用于评估AI在金融领域的代理能力。解决现有基准无法全面评估金融任务执行的问题，涵盖交易、对冲、市场洞察和审计四个流程，发现当前AI在长期协调和验证方面存在不足。**

- **链接: [https://arxiv.org/pdf/2605.14355](https://arxiv.org/pdf/2605.14355)**

> **作者:** Xueqing Peng; Zhuohan Xie; Yupeng Cao; Haohang Li; Lingfei Qian; Yan Wang; Vincent Jim Zhang; Huan He; Xuguang Ai; Linhai Ma; Ruoyu Xiang; Yueru He; Yi Han; Shuyao Wang; Yuqing Guo; Mingyang Jiang; Yilun Zhao; Youzhong Dong; Xiaoyu Wang; Yankai Chen; Ye Yuan; Qiyuan Zhang; Fuyuan Lyu; Haolun Wu; Yonghan Yang; Zichen Zhao; Yuyang Dai; Fan Zhang; Rania Elbadry; Ayesha Gull; Muhammad Usman Safder; Nuo Chen; Fengbin Zhu; Tianshi Cai; Zimu Wang; Polydoros Giannouris; Yuechen Jiang; Zhiwei Liu; Mohsinul Kabir; Yuyan Wang; Yixiang Zheng; Yangyang Yu; Weijin Liu; Wenbo Cao; Anke Xu; Peng Lu; Jerry Huang; Mingquan Lin; Prayag Tiwari; Yijia Zhao; Víctor Gutiérrez-Basulto; Xiao-Yang Liu; Kaleb E Smith; Jiahuan Pei; Arman Cohan; Jimin Huang; Yuehua Tang; Alejandro Lopez-Lira; Xi Chen; Xue Liu; Junichi Tsujii; Jian-Yun Nie; Sophia Ananiadou
>
> **摘要:** As AI agents improve, the central question is no longer whether they can solve isolated well-defined financial tasks, but whether they can reliably carry out financial professional work. Existing financial benchmarks offer only a partial view of this ability, as they primarily evaluate static competencies such as question answering, retrieval, summarization, and classification. We introduce Herculean, the first skilled benchmark for agentic financial intelligence spanning four representative workflows, including Trading, Hedging, Market Insights, and Auditing. Each workflow is instantiated as a standardized MCP-based skill environment with its own tools, interaction dynamics, constraints, and success criteria, enabling consistent end-to-end assessment of heterogeneous agent systems. Across frontier agents, we find agents perform relatively well on Trading and Market Insights, but struggle substantially on Hedging and Auditing, where long-horizon coordination, state consistency, and structured verification are critical. Overall, our results point to a key gap in current agents in turning financial reasoning into dependable workflow execution in high-stakes financial workflows.
>
---
#### [replaced 090] SWE-rebench V2: Language-Agnostic SWE Task Collection at Scale
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出SWE-rebench V2，解决SWE任务数据不足问题，构建大规模、多语言的可执行任务集，支持强化学习训练。**

- **链接: [https://arxiv.org/pdf/2602.23866](https://arxiv.org/pdf/2602.23866)**

> **作者:** Ibragim Badertdinov; Maksim Nekrashevich; Anton Shevtsov; Alexander Golubev
>
> **备注:** ICML 2026
>
> **摘要:** Software engineering agents (SWE) are improving rapidly, with recent gains largely driven by reinforcement learning (RL). However, RL training is constrained by the scarcity of large-scale task collections with reproducible execution environments and reliable test suites. Although a growing number of benchmarks have emerged, datasets suitable for training remain limited in scale and diversity or often target a limited set of high-resource language ecosystems. We introduce SWE-rebench V2, a language-agnostic automated pipeline for harvesting executable real-world SWE tasks and constructing RL training environments at scale. The pipeline synthesizes repository-specific installation and test procedures via an interactive setup agent, and filters unsound instances using an ensemble of LLM judges, validated against human-verified SWE-bench annotations. Using this pipeline, we construct a dataset of 32,079 tasks spanning 20 languages and 3,617 repositories, with pre-built images for reproducible execution. To further scale training data, we additionally release 120,000+ tasks with installation instructions, fail-to-pass tests and rich metadata, where the problem statement is generated based on the original pull request description. We validate the collected instances through a diagnostic study that covers a subset of tasks in five programming languages across seven popular models, and provide instance-level metadata that flags common confounders such as overly restrictive tests and underspecified descriptions. We release the datasets, the collection and execution code, and associated artifacts to enable large-scale training of SWE agents across diverse languages and repositories.
>
---
#### [replaced 091] Gradients with Respect to Semantics Preserving Embeddings Tell the Uncertainty of Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型不确定性量化任务，旨在解决LLMs易幻觉的问题。提出SemGrad和HybridGrad方法，通过语义空间梯度评估不确定性，提高效率与效果。**

- **链接: [https://arxiv.org/pdf/2605.04638](https://arxiv.org/pdf/2605.04638)**

> **作者:** Mingda Li; Rundong Lv; Xinyu Li; Weinan Zhang; Ting Liu
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** Uncertainty quantification (UQ) is an important technique for ensuring the trustworthiness of LLMs, given their tendency to hallucinate. Existing state-of-the-art UQ approaches for free-form generation rely heavily on sampling, which incurs high computational cost and variance. In this work, we propose the first gradient-based UQ method for free-form generation, SemGrad, which is sampling-free and computationally efficient. Unlike prior gradient-based methods developed for classification tasks that operates in parameter space, we propose to consider gradients in semantic space. Our method builds on the key intuition that a confident LLM should maintain stable output distributions under semantically equivalent input perturbations. We interpret the stability as the gradients in semantic space and introduce a Semantic Preservation Score (SPS) to identify embeddings that best capture semantics, with respect to which gradients are computed. We further propose HybridGrad, which combines the strengths of SemGrad and parameter gradients. Experiments demonstrate that both of our methods provide efficient and effective uncertainty estimates, achieving superior performance than state-of-the-art methods, particularly in settings with multiple valid responses.
>
---
#### [replaced 092] MASCOT: Towards Multi-Agent Socio-Collaborative Companion Systems
- **分类: cs.CL; cs.AI; cs.HC**

- **简介: 该论文提出MASCOT框架，解决多智能体系统中的角色一致性与对话冗余问题，通过双层优化提升协作对话质量。**

- **链接: [https://arxiv.org/pdf/2601.14230](https://arxiv.org/pdf/2601.14230)**

> **作者:** Yiyang Wang; Yiqiao Jin; Alex Cabral; Josiah Hester
>
> **备注:** 15 pages, 9 figures. this https URL
>
> **摘要:** Multi-agent systems (MAS) are emerging as promising socio-collaborative companions for emotional and cognitive support. However, existing systems frequently suffer from persona collapse, where agents revert to generic, homogenized assistant behaviors, and social sycophancy, where agents produce redundant, non-constructive dialogue. We propose MASCOT, a multi-agent framework for multi-perspective socio-collaborative companions. MASCOT introduces a novel bi-level optimization strategy to harmonize individual and collective behaviors: 1) Persona-Aware Behavioral Alignment, an RLAIF-driven pipeline that fine-tunes individual agents for agent-specific identities; and 2) Collaborative Dialogue Optimization, a group-level adaptation process that promotes complementary, diverse, and productive discourse. We evaluate MASCOT using human-grounded contexts drawn across both in-domain and out-of-domain (OOD) settings against state-of-the-art baselines. MASCOT improves persona consistency by up to +14.1 and social contribution by up to +10.6. A broad evaluation suite, including human evaluation, multiple LLM judges, three-way comparisons, and automatic metrics, further shows that MASCOT produces more role-consistent and less redundant multi-agent dialogue.
>
---
#### [replaced 093] RenoBench: A Citation Parsing Benchmark
- **分类: cs.DL; cs.CL**

- **简介: 该论文属于信息提取任务，旨在解决 citation parsing 的评估问题。通过构建公开基准 RenoBench，提升 citation 解析系统的标准化评估能力。**

- **链接: [https://arxiv.org/pdf/2603.25640](https://arxiv.org/pdf/2603.25640)**

> **作者:** Parth Sarin; Juan Pablo Alperin; Adam Buttrick; Dione Mentis
>
> **备注:** Presented as a conference paper at CiteX 2026
>
> **摘要:** Accurate parsing of citations is necessary for machine-readable scholarly infrastructure. But, despite sustained interest in this problem, existing evaluation techniques are often not generalizable, based on synthetic data, or not publicly available. We introduce RenoBench, a public domain benchmark for citation parsing, sourced from PDFs released on four publishing ecosystems: SciELO, Redalyc, the Public Knowledge Project, and Open Research Europe. Starting from 161,000 annotated citations, we apply automated validation and feature-based sampling to produce a dataset of 10,000 citations spanning multiple languages, publication types, and platforms. We then evaluate a variety of citation parsing systems and report field-level precision and recall. Our results show strong performance from language models, particularly when fine-tuned. RenoBench enables reproducible, standardized evaluation of citation parsing systems, and provides a foundation for advancing automated citation parsing and metascientific research.
>
---
#### [replaced 094] Grounding or Guessing? Visual Signals for Detecting Hallucinations in Sign Language Translation
- **分类: cs.CL**

- **简介: 该论文属于符号语言翻译任务，旨在解决模型生成无视觉依据文本（幻觉）的问题。通过引入基于视觉信号的可靠性度量，评估模型是否依赖视觉输入，从而检测幻觉。**

- **链接: [https://arxiv.org/pdf/2510.18439](https://arxiv.org/pdf/2510.18439)**

> **作者:** Yasser Hamidullah; Koel Dutta Chowdhury; Yusser Al Ghussin; Shakib Yazdani; Cennet Oguz; Josef van Genabith; Cristina España-Bonet
>
> **备注:** Published at ICLR2026 Code available at \url{this https URL}
>
> **摘要:** Hallucination, where models generate fluent text unsupported by visual evidence, remains a major flaw in vision-language models and is particularly critical in sign language translation (SLT). In SLT, meaning depends on precise grounding in video, and gloss-free models are especially vulnerable because they map continuous signer movements directly into natural language without intermediate gloss supervision that serves as alignment. We argue that hallucinations arise when models rely on language priors rather than visual input. To capture this, we propose a token-level reliability measure that quantifies how much the decoder uses visual information. Our method combines feature-based sensitivity, which measures internal changes when video is masked, with counterfactual signals, which capture probability differences between clean and altered video inputs. These signals are aggregated into a sentence-level reliability score, providing a compact and interpretable measure of visual grounding. We evaluate the proposed measure on two SLT benchmarks (PHOENIX-2014T and CSL-Daily) with both gloss-based and gloss-free models. Our results show that reliability predicts hallucination rates, generalizes across datasets and architectures, and decreases under visual degradations. Beyond these quantitative trends, we also find that reliability distinguishes grounded tokens from guessed ones, allowing risk estimation without references; when combined with text-based signals (confidence, perplexity, or entropy), it further improves hallucination risk estimation. Qualitative analysis highlights why gloss-free models are more susceptible to hallucinations. Taken together, our findings establish reliability as a practical and reusable tool for diagnosing hallucinations in SLT, and lay the groundwork for more robust hallucination detection in multimodal generation.
>
---
#### [replaced 095] HalleluBERT: Let Every Token That Has Meaning Bear Its Weight
- **分类: cs.CL**

- **简介: 该论文提出HalleluBERT，一个基于RoBERTa的希伯来语编码器，解决希伯来语NLP任务中缺乏高效模型的问题。通过训练大规模希伯来文数据，提升命名实体识别和情感分类性能。**

- **链接: [https://arxiv.org/pdf/2510.21372](https://arxiv.org/pdf/2510.21372)**

> **作者:** Raphael Schmitt
>
> **摘要:** Transformer-based models have advanced NLP, yet Hebrew still lacks a RoBERTa encoder that is trained at scale and released in both base and large variants. We present HalleluBERT, a RoBERTa-based encoder family trained from scratch on 49.1~GB of deduplicated Hebrew web text and Wikipedia using a Hebrew-specific byte-level BPE vocabulary. On native Hebrew benchmarks for named entity recognition (BMC, NEMO) and sentiment classification (SMCD), HalleluBERT outperforms monolingual and multilingual baselines, and yields the highest unweighted mean score across the three benchmarks. We release model weights and tokenizer under the MIT license to support reproducible Hebrew NLP research.
>
---
#### [replaced 096] ClinTutor-R1: Advancing Scalable and Robust One-to-Many Alignment in Clinical Socratic Education
- **分类: cs.CL**

- **简介: 该论文属于临床教育任务，解决一对多教学中的对齐问题。通过构建多智能体模拟器和数据集，提出ClinTutor-R1模型，实现高效群体教学。**

- **链接: [https://arxiv.org/pdf/2512.05671](https://arxiv.org/pdf/2512.05671)**

> **作者:** Zhitao He; Haolin Yang; Zeyu Qin; Yi R Fung
>
> **备注:** Accepted by ICML 2026 (Spotlight)
>
> **摘要:** While Large Language Models (LLMs) have achieved remarkable success in dyadic (one-on-one) instruction, they face significant challenges in One-to-Many alignment, such as clinical ward rounds, where an instructor must simultaneously guide a diverse group of trainees. Current models often suffer from context dilution and goal misalignment, failing to balance individual scaffolding with collective learning progress. To address this, we introduce ClinEdu, a multi-agent pedagogical simulator that models the complexity of group dynamics. Leveraging this platform, we construct ClinTeach, a large-scale dataset of Socratic teaching dialogues, and propose ClinTutor-R1, the first vision-language agent explicitly architected to achieve one-to-many alignment in clinical education, employing an explicit internal thinking mechanism to model both individual belief states and group consensus. We validate our framework through a comprehensive protocol covering static benchmarks, in-situ interactive evaluation within ClinEdu, expert assessment, and a 200-participant real user study. Experimental results demonstrate that ClinTutor-R1 outperforms base models by over 20% and achieves parity with proprietary models, while exhibiting scalability in maintaining instructional quality across expanding student cohorts.
>
---
#### [replaced 097] Addressing Longstanding Challenges in Cognitive Science with Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于认知科学与人工智能交叉领域，旨在利用语言模型解决认知科学中的研究整合、理论形式化等问题，通过分析文献、生成预测等方法促进学科发展。**

- **链接: [https://arxiv.org/pdf/2511.00206](https://arxiv.org/pdf/2511.00206)**

> **作者:** Dirk U. Wulff; Rui Mata
>
> **摘要:** Cognitive science faces ongoing challenges in research integration, formalization, conceptual clarity, and other areas, in part due to its multifaceted and interdisciplinary nature. Recent advances in artificial intelligence, particularly the development of language models, offer tools that may help to address these longstanding issues. Specifically, they can help map fragmented literatures, formalize verbal theories, identify overlap among constructs and measures, generate predictions across tasks, and extract cultural or ecological structure from naturalistic data. However, these opportunities come with risks, including oversimplification, opacity, deskilling, and bias. Taken together, we conclude that language models could serve as tools for a more integrative and cumulative cognitive science when used judiciously to complement, rather than replace, human agency.
>
---
#### [replaced 098] Utility-Preserving De-Identification for Math Tutoring: Investigating Numeric Ambiguity in the MathEd-PII Benchmark Dataset
- **分类: cs.CL**

- **简介: 该论文属于隐私保护任务，旨在解决数学辅导对话中因数值歧义导致的PII误删问题。通过构建基准数据集并测试不同检测策略，提升数据可用性与隐私保护效果。**

- **链接: [https://arxiv.org/pdf/2602.16571](https://arxiv.org/pdf/2602.16571)**

> **作者:** Zhuqian Zhou; Kirk Vanacore; Bakhtawar Ahtisham; Jinsook Lee; Doug Pietrzak; Daryl Hedley; Jorge Dias; Chris Shaw; Ruth Schäfer; René F. Kizilcec
>
> **摘要:** Large-scale sharing of dialogue data is key to advancing the science of teaching and learning, yet rigorous de-identification remains a major barrier. In mathematics tutoring transcripts, numeric expressions frequently resemble structured identifiers (e.g., dates or IDs), leading generic Personally Identifiable Information (PII) detection systems to over-redact core instructional content and reduce data utility. This work asks how to detect PII while preserving educational utility, focusing on this "numeric ambiguity" problem. We introduce MathEd-PII, the first benchmark dataset for PII detection in math tutoring dialogues, built with human-in-the-loop LLM annotation. Using density-based segmentation, we show that false PII redactions cluster in math-dense regions, confirming numeric ambiguity as a key failure mode. We then compare four detection strategies: a Presidio baseline and three LLM-based approaches with basic, math-aware, and segment-aware prompting. Domain-aware prompting, including both math-aware (F1: 0.802) and segment-aware versions (F1: 0.821), substantially outperforms the baseline (F1: 0.379) while reducing numeric false positives, demonstrating that de-identification must incorporate domain context to preserve analytic utility. This work provides a new benchmark and evidence that utility-preserving de-identification for tutoring data requires domain-aware modeling.
>
---
#### [replaced 099] Many-Shot CoT-ICL: Making In-Context Learning Truly Learn
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究多示例链式思维上下文学习（CoT-ICL）在推理任务中的表现，旨在提升模型的推理能力。通过分析不同任务和模型，提出有序示范选择方法，优化学习效果。**

- **链接: [https://arxiv.org/pdf/2605.13511](https://arxiv.org/pdf/2605.13511)**

> **作者:** Tsz Ting Chung; Lemao Liu; Mo Yu; Dit-Yan Yeung
>
> **备注:** Accepted by ICML 2026
>
> **摘要:** While many-shot ICL achieves remarkable performance, prior studies of its scaling behavior have mainly focused on non-reasoning tasks. In this work, we study many-shot ICL on reasoning tasks, with a particular focus on many-shot chain-of-thought in-context learning (CoT-ICL). Analyzing across non-reasoning and reasoning tasks and across non-reasoning and reasoning-oriented LLMs, we identify several distinctive properties of many-shot CoT-ICL. We further interpret these findings by viewing many-shot CoT-ICL as in-context test-time learning rather than scaled pattern matching, and suggest two principles: (i) demonstrations should be easy for the target model to understand, and (ii) they should be ordered to support a smooth conceptual progression. Guided by the principle, we propose Curvilinear Demonstration Selection (CDS), a simple ordering method that yields up to a 5.42 percentage-point gain on a math task with 64 demonstrations. Overall, our results reframe the long context window from a retrieval buffer into a structured curriculum for in-context test-time learning.
>
---
#### [replaced 100] SindBERT, the Sailor: Charting the Seas of Turkish NLP
- **分类: cs.CL**

- **简介: 该论文提出SindBERT，首个针对土耳其语的大型RoBERTa模型，解决土耳其语NLP资源不足问题。通过大规模文本训练，评估其在多个任务上的表现，探讨数据质量和规模对模型效果的影响。**

- **链接: [https://arxiv.org/pdf/2510.21364](https://arxiv.org/pdf/2510.21364)**

> **作者:** Raphael Schmitt; Stefan Schweter
>
> **备注:** Published at SIGTURK 2026, co-located with EACL 2026
>
> **摘要:** Transformer models have revolutionized NLP, yet many morphologically rich languages remain underrepresented in large-scale pre-training efforts. With SindBERT, we set out to chart the seas of Turkish NLP, providing the first large-scale RoBERTa-based encoder for Turkish. Trained from scratch on 312~GB of Turkish text (mC4, OSCAR23, Wikipedia), SindBERT is released in both base and large configurations, representing the first large-scale encoder-only language model available for Turkish. We evaluate SindBERT on part-of-speech tagging, named entity recognition, offensive language detection, and the TurBLiMP linguistic acceptability benchmark. Our results show that SindBERT performs competitively with existing Turkish and multilingual models, with the large variant achieving the best scores in two of four tasks but showing no consistent scaling advantage overall. This flat scaling trend, also observed for XLM-R and EuroBERT, suggests that current Turkish benchmarks may already be saturated. At the same time, comparisons with smaller but more curated models such as BERTurk highlight that corpus quality and diversity can outweigh sheer data volume. Taken together, SindBERT contributes both as an openly released resource for Turkish NLP and as an empirical case study on the limits of scaling and the central role of corpus composition in morphologically rich languages. The SindBERT models are released under the MIT license and made available in both fairseq and Huggingface formats.
>
---
#### [replaced 101] CacheRAG: A Semantic Caching System for Retrieval-Augmented Generation in Knowledge Graph Question Answering
- **分类: cs.DB; cs.CL**

- **简介: 该论文提出CacheRAG，解决KGQA中LLM系统缺乏历史查询利用的问题，通过语义缓存提升检索生成效果。**

- **链接: [https://arxiv.org/pdf/2604.26176](https://arxiv.org/pdf/2604.26176)**

> **作者:** Yushi Sun; Lei Chen
>
> **摘要:** The integration of Large Language Models (LLMs) with Retrieval-Augmented Generation (RAG) has significantly advanced Knowledge Graph Question Answering (KGQA). However, existing LLM-driven KGQA systems act as stateless planners, generating retrieval plans in isolation without exploiting historical query patterns: analogous to a database system that optimizes every query from scratch without a plan cache. This fundamental design flaw leads to schema hallucinations and limited retrieval coverage. We propose CacheRAG, a systematic cache-augmented architecture for LLM-based KGQA that transforms stateless planners into continual learners. Unlike traditional database plan caching (which optimizes for frequency), CacheRAG introduces three novel design principles tailored for LLM contexts: (1) Schema-agnostic user interface: A two-stage semantic parsing framework via Intermediate Semantic Representation (ISR) enables non-expert users to interact purely in natural language, while a Backend Adapter grounds the LLM with local schema context to compile executable physical queries safely. (2) Diversity-optimized cache retrieval: A two-layer hierarchical index (Domain $\rightarrow$ Aspect) coupled with Maximal Marginal Relevance (MMR) maximizes structural variety in cached examples, effectively mitigating reasoning homogeneity. (3) Bounded heuristic expansion: Deterministic depth and breadth subgraph operators with strict complexity guarantees significantly enhance retrieval recall without risking unbounded API execution. Extensive experiments on multiple benchmarks demonstrate that CacheRAG significantly outperforms state-of-the-art baselines (e.g., +13.2% accuracy and +17.5% truthfulness on the CRAG dataset).
>
---
#### [replaced 102] Adapting Large Language Models to a Low-Resource Agglutinative Language: A Comparative Study of LoRA and QLoRA for Bashkir
- **分类: cs.CL**

- **简介: 论文研究在低资源屈折语言Bashkir上适配大语言模型的任务，比较LoRA和QLoRA的参数高效微调方法，旨在找到质量与计算成本间的平衡。**

- **链接: [https://arxiv.org/pdf/2605.04948](https://arxiv.org/pdf/2605.04948)**

> **作者:** Mullosharaf K. Arabov; Svetlana S. Khaybullina
>
> **备注:** Accepted to CLIB 2026
>
> **摘要:** This paper presents a comparative study of parameter-efficient fine-tuning (PEFT) methods, including LoRA and QLoRA, applied to the task of adapting large language models to the Bashkir language, a low-resource agglutinative language of the Turkic family. Experimental evaluation is conducted on a Bashkir text corpus of 71k documents (46.9M tokens) using models of various architectures: DistilGPT2, GPT-2 (base, medium), Phi-2, Qwen2.5-7B, DeepSeek-7B, and Mistral-7B. To improve the reliability of results, each configuration was trained with three different random seeds. The lowest perplexity on the test set was obtained for GPT-2 medium with full fine-tuning (3.34). Meanwhile, QLoRA applied to Mistral-7B (3.79) and Phi-2 (3.81) achieved comparable quality with over 40 times fewer trainable parameters. However, we also observed cases of significant quality degradation when using PEFT for certain architectures (e.g., DeepSeek-7B with rank 8, perplexity = 129.55), indicating that the outcome depends critically on the choice of the base model and its tokenizer. Additionally, a qualitative analysis of generated texts based on Bashkir prompts revealed that models with the best perplexity do not necessarily produce the most coherent outputs: QLoRA-tuned models generated monolingual Bashkir continuations, whereas the fully fine-tuned model with the lowest perplexity frequently switched to English. The results suggest that QLoRA on 7B-scale models offers an effective compromise between quality and computational cost for Bashkir. To ensure reproducibility, open data, code, and trained adapters will be released upon acceptance.
>
---
#### [replaced 103] Are Full Rollouts Necessary for On-Policy Distillation?
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决OPD训练效率低的问题。通过控制rollout长度，提出POPD和TOPD策略，提升效率并减少资源消耗。**

- **链接: [https://arxiv.org/pdf/2605.31490](https://arxiv.org/pdf/2605.31490)**

> **作者:** Yaocheng Zhang; Jiajun Chai; Yuqian Fu; Songjun Tu; Xiaohan Wang; Wei Lin; Guojun Yin; Qichao Zhang; Yuanheng Zhu; Dongbin Zhao
>
> **备注:** 15 pages, 14 figures
>
> **摘要:** On-policy distillation (OPD) provides dense teacher feedback along student-generated rollouts rather than fixed teacher traces and has emerged as a promising post-training paradigm. However, standard OPD typically generates full rollouts during training, which is computationally expensive and may expose the student to unreliable teacher feedback at late rollout positions, especially during early training. We identify the rollout horizon as a key bottleneck in OPD that substantially impacts training efficiency. Unlike Reinforcement Learning with Verifiable Rewards (RLVR), OPD does not require a final answer reward to provide learning signals. Therefore, full rollouts may not always be necessary for OPD. Motivated by this insight, we propose two simple horizon-control strategies: Progressive OPD (POPD), which gradually expands the rollout horizon during training, and Truncated OPD (TOPD), which permanently performs distillation on reliable truncated rollouts. Experiments on mathematical reasoning show that POPD improves the training efficiency of OPD by up to 3$\times$, while TOPD matches OPD performance using only 10\% of the rollout horizon, leading to substantial wall-clock and memory reductions. These results demonstrate that controlling the rollout horizon offers a simple and practical path to more efficient OPD.
>
---
#### [replaced 104] Parametric Social Identity Injection and Diversification in Public Opinion Simulation
- **分类: cs.CL**

- **简介: 该论文属于公共意见模拟任务，解决LLM模拟中社会多样性不足的问题。通过PSII框架注入参数化社会身份，提升模拟的多样性与真实性。**

- **链接: [https://arxiv.org/pdf/2603.16142](https://arxiv.org/pdf/2603.16142)**

> **作者:** Hexi Wang; Yujia Zhou; Bangde Du; Qingyao Ai; Yiqun Liu
>
> **备注:** Accepted to KDD 2026 Research Track. Project page: this https URL
>
> **摘要:** Large language models (LLMs) have recently been adopted as synthetic agents for public opinion simulation, offering a promising alternative to costly and slow human surveys. Despite their scalability, current LLM-based simulation methods fail to capture social diversity, producing flattened inter-group differences and overly homogeneous responses across demographic groups. We identify this limitation as a Diversity Collapse phenomenon in LLM hidden representations, where distinct social identities become increasingly indistinguishable across layers. Motivated by this observation, we propose Parametric Social Identity Injection (PSII), a general framework that injects explicit, parametric representations of demographic attributes and value orientations directly into intermediate hidden states of LLMs. Unlike prompt-based persona conditioning, PSII enables fine-grained and controllable identity modulation at the representation level. Extensive experiments on the World Values Survey using multiple open-source LLMs show that PSII significantly improves distributional fidelity and diversity, reducing KL divergence to real-world survey data while enhancing overall diversity. This work provides new insights into representation-level control of LLM agents and advances scalable, diversity-aware public opinion simulation.
>
---
#### [replaced 105] Latent Reasoning in TRMs is Secretly a Policy Improvement Operator
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文研究递归模型中的潜在推理机制，属于深度学习任务。解决递归层效率低和冗余计算问题，提出基于策略改进的训练方法，提升模型性能并减少计算量。**

- **链接: [https://arxiv.org/pdf/2511.16886](https://arxiv.org/pdf/2511.16886)**

> **作者:** Arip Asadulaev; Rayan Banerjee; Fakhri Karray; Martin Takac
>
> **摘要:** Recently, small models with latent recursion have obtained promising results on complex reasoning tasks. These results are typically explained by the theory that such recursion increases a networks depth, allowing it to compactly emulate the capacity of larger models. However, the performance of recursively added layers remains behind the capabilities of one pass models with the same feed-forward depth. This means that in the looped version, not every recursive step effectively contributes to depth. This raises the question: when and why does latent reasoning improve performance, and when does it result in dead compute? In our work, we demonstrate that latent recursive reasoning provides answer to this question. We show that latent recursive reasoning can be formalized as a policy improvement algorithm. Building on these insights, we propose to use a training schemes from reinforcement learning and diffusion methods for latent reasoning models. Using the Tiny Recursive Model as our testbed, we show that with our modifications we can avoid dead compute steps and reduce the total number of forward passes by 18x while maintaining performance. Broadly speaking, we show how a policy improvement perspective on recursive steps can explain model behavior and provide insights for further improvements.
>
---
#### [replaced 106] Stabilizing Policy Optimization via Logits Convexity
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决RL训练不稳定的问题。通过分析SFT与RL的梯度差异，提出LCO框架，提升训练稳定性与效果。**

- **链接: [https://arxiv.org/pdf/2603.00963](https://arxiv.org/pdf/2603.00963)**

> **作者:** Hongzhan Chen; Tao Yang; Yuhua Zhu; Shiping Gao; Xiaojun Quan; Ting Yao
>
> **摘要:** While reinforcement learning (RL) has been central to the recent success of large language models (LLMs), RL optimization is notoriously unstable, especially when compared to supervised fine-tuning (SFT). In this work, we investigate the stability gap between SFT and RL from a gradient-based perspective, and show that the convexity of the SFT loss with respect to model logits plays a key role in enabling stable training. Our theoretical analysis demonstrates that this property induces favorable gradient directionality during optimization. In contrast, Proximal Policy Optimization (PPO), a widely adopted policy gradient algorithm utilizing a clipped surrogate objective, lacks this stabilizing property. Motivated by this observation, we propose Logits Convex Optimization (LCO), a simple yet effective policy optimization framework that aligns the learned policy with an optimal target derived from the original RL objective, thereby emulating the stabilizing effects of logits-level convexity. Extensive experiments across multiple model families show that our LCO framework consistently improves training stability and outperforms conventional RL methods on a broad range of benchmarks.
>
---
#### [replaced 107] Retrieval-Augmented Linguistic Calibration
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的可信度校准任务，旨在解决语言信心表达的校准问题。通过构建分布模型和引入Faithfulness Divergence指标，提出RALC方法提升语言信心的准确性与一致性。**

- **链接: [https://arxiv.org/pdf/2605.19344](https://arxiv.org/pdf/2605.19344)**

> **作者:** Yi-Fan Yeh; Linwei Tao; Minjing Dong; Tao Huang; Jialin Yu; Philip Torr; Chang Xu
>
> **摘要:** Linguistic cues such as "I believe" and "probably" offer an intuitive interface for communicating confidence, yet a generalisable, principled calibration framework for linguistic confidence expressions remains underexplored. In particular, co-occurring linguistic cues, contextual variation, and subjective audience interpretation pose unique challenges. We therefore model linguistic confidence as a distribution over plausible perceived probability values that a statement is correct, capturing interpretation variability that scalar representations discard. Within this distributional framework, we introduce faithfulness as a complementary evaluation dimension and present Faithfulness Divergence (FD), an information-theoretic metric quantifying the surprise induced in audience beliefs upon truth revelation. Building on these foundations, we present Retrieval-Augmented Linguistic Calibration (RALC), a lightweight post-hoc pipeline that propagates calibrated confidence signals back into natural language via retrieval-augmented rewriting. Across three QA benchmarks and five LLM families, RALC improves in-domain faithfulness and calibration up to 66% and 58%, respectively, outperforming black-box and grey-box calibration baselines.
>
---
#### [replaced 108] VERA: Variational Inference Framework for Jailbreaking Large Language Models
- **分类: cs.CR; cs.CL; cs.LG; stat.ML**

- **简介: 该论文属于对抗攻击任务，旨在解决黑盒 jailbreak 方法的不足。通过变分推断框架 VERA，训练小模型生成多样化的恶意提示，提升攻击效率与效果。**

- **链接: [https://arxiv.org/pdf/2506.22666](https://arxiv.org/pdf/2506.22666)**

> **作者:** Anamika Lochab; Lu Yan; Patrick Pynadath; Xiangyu Zhang; Ruqi Zhang
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** The rise of API-only access to state-of-the-art LLMs highlights the need for effective black-box jailbreak methods to identify model vulnerabilities in real-world settings. Without a principled objective for gradient-based optimization, most existing approaches rely on genetic algorithms, which are limited by their initialization and dependence on manually curated prompt pools. Furthermore, these methods require individual optimization for each prompt, failing to provide a comprehensive characterization of model vulnerabilities. To address this gap, we introduce VERA: Variational infErence fRamework for jAilbreaking. VERA casts black-box jailbreak prompting as a variational inference problem, training a small attacker LLM to approximate the target LLM's posterior over adversarial prompts. Once trained, the attacker can generate diverse, fluent jailbreak prompts for a target query without re-optimization. Experimental results show that VERA achieves strong performance across a range of target LLMs, highlighting the value of probabilistic inference for adversarial prompt generation.
>
---
#### [replaced 109] Agent Tools Orchestration Leaks More: Dataset, Benchmark, and Mitigation
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于隐私保护任务，研究LLM代理工具编排中的隐私泄露问题，提出TOP-Bench基准和TOP-Align方法以降低敏感信息泄露。**

- **链接: [https://arxiv.org/pdf/2512.16310](https://arxiv.org/pdf/2512.16310)**

> **作者:** Yuxuan Qiao; Dongqin Liu; Hongchang Yang; Wei Zhou; Songlin Hu
>
> **备注:** 17 pages, 2 figures. Dataset and code are available at this https URL
>
> **摘要:** LLM-based agents increasingly use multiple external tools to complete complex tasks. We study Tools Orchestration Privacy Risk (TOP-R): an agent may combine individually non-sensitive tool returns and disclose an unintended sensitive conclusion. We formalize TOP-R with three conditions: conclusion sensitivity, single-source non-inferability, and compositional inferability. We introduce LRSE (Library-Grounded Reverse-Inference Seed Expansion), a four-library reverse-construction pipeline grounded in privacy norms, reasoning chains, tool schemas, and task scenarios, and use it to build TOP-Bench, a 1,000-instance benchmark. The benchmark evaluates final-response semantic disclosure under a controlled two-stage tool-use protocol. Across six LLM agents, task completion remains high, but the average leakage rate reaches 88.6 percent, yielding an H-score of only 20.4. Two prompt-only safeguards improve H-score by about 2.7 points on the main benchmark. We further propose TOP-Align, an SFT+DPO post-training method for safer task completion boundaries. On a separate post-training evaluation split, TOP-Align improves H-score by 16.2 points over the corresponding base model, compared with a 4.9-point average gain from prompt-only mitigation on the same split. These results show that TOP-R requires mitigation beyond prompting alone.
>
---
#### [replaced 110] Deep networks learn to parse uniform-depth context-free languages from local statistics
- **分类: stat.ML; cond-mat.dis-nn; cs.CL; cs.LG**

- **简介: 该论文研究语言结构学习任务，解决如何从句子中学习语法结构的问题。通过引入可调PCFG模型和基于深度网络的推理算法，分析数据统计与学习能力的关系。**

- **链接: [https://arxiv.org/pdf/2602.06065](https://arxiv.org/pdf/2602.06065)**

> **作者:** Jack T. Parley; Francesco Cagnetta; Matthieu Wyart
>
> **备注:** Accepted as regular paper at ICML 2026
>
> **摘要:** Understanding how the structure of language can be learned from sentences alone is a central question in both cognitive science and machine learning. Studies of the internal representations of Large Language Models (LLMs) support their ability to parse text when predicting the next word, while representing semantic notions independently of surface form. Yet, which data statistics make these feats possible, and how much data is required, remain largely unknown. Probabilistic context-free grammars (PCFGs) provide a tractable testbed for studying these questions. However, prior work has focused either on the post-hoc characterization of the parsing-like algorithms used by trained networks; or on the learnability of PCFGs with fixed syntax, where parsing is unnecessary. Here, we (i) introduce a tunable class of PCFGs in which both the degree of ambiguity and the correlation structure across scales can be controlled; (ii) provide a learning mechanism -- an inference algorithm inspired by the structure of deep convolutional networks -- that links learnability and sample complexity to specific language statistics; and (iii) validate our predictions empirically across deep convolutional and transformer-based architectures. Overall, we propose a unifying framework where correlations at different scales lift local ambiguities, enabling the emergence of hierarchical representations of the data.
>
---
#### [replaced 111] SCOPE: Signal-Calibrated On-Policy Distillation Enhancement with Dual-Path Adaptive Weighting
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出SCOPE方法，解决大语言模型中策略蒸馏的信号不均衡问题，通过双路径自适应加权提升推理对齐效果。**

- **链接: [https://arxiv.org/pdf/2604.10688](https://arxiv.org/pdf/2604.10688)**

> **作者:** Binbin Zheng; Xing Ma; Yiheng Liang; Jingqing Ruan; Xiaoliang Fu; Kepeng Lin; Benchang Zhu; Ke Zeng; Xunliang Cai
>
> **摘要:** On-policy reinforcement learning has become the dominant paradigm for reasoning alignment in large language models, yet its sparse, outcome-level rewards make token-level credit assignment notoriously difficult. On-Policy Distillation (OPD) alleviates this by introducing dense, token-level KL supervision from a teacher model, but typically applies this supervision uniformly across all rollouts, ignoring fundamental differences in signal quality. We propose Signal-Calibrated On-Policy Distillation Enhancement (SCOPE), a dual-path adaptive training framework that routes on-policy rollouts by correctness into two complementary supervision paths. For incorrect trajectories, SCOPE performs teacher-perplexity-weighted KL distillation to prioritize instances where the teacher demonstrates genuine corrective capability, while down-weighting unreliable guidance. For correct trajectories, it applies student-perplexity-weighted MLE to concentrate reinforcement on low-confidence samples at the capability boundary rather than over-reinforcing already mastered ones. Both paths employ a group-level normalization to adaptively calibrate weight distributions, accounting for the intrinsic difficulty variance across prompts. Extensive experiments on six reasoning benchmarks show that SCOPE achieves an average relative improvement of 11.42% in Avg@32 and 7.30% in Pass@32 over competitive baselines, demonstrating its consistent effectiveness.
>
---
#### [replaced 112] Evi-Steer: Learning to Steer Biomedical Vision-Language Models through Efficient and Generalizable Evidential Tuning
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉-语言模型的适应任务，旨在解决生物医学图像中模型在小样本和领域漂移下的鲁棒性问题。提出Evi-Steer框架，实现高效且不确定性感知的参数微调。**

- **链接: [https://arxiv.org/pdf/2605.26292](https://arxiv.org/pdf/2605.26292)**

> **作者:** Taha Koleilat; Hassan Rivaz; Yiming Xiao
>
> **备注:** MICCAI 2026 Early Accept; Project Page: this https URL. This preprint has not undergone peer review or any post-submission improvements or corrections. The Version of Record of this contribution will be published as part of the MICCAI 2026 proceedings in October
>
> **摘要:** Parameter-efficient adaptation of vision-language foundation models is crucial for precise multimodal understanding of biomedical images, yet existing methods remain deterministic and often struggle under domain shift or ambiguous image-text alignment. This limitation is particularly critical in the clinic, where models should remain robust in low-data regimes and domain shifts. We present Evi-Steer, an evidential cross-modal low-dimensional steering framework for BiomedCLIP that enables uncertainty-aware parameter-efficient fine-tuning while updating only 0.11% of total model parameters. Our approach performs lightweight low-dimensional token updates in both vision and text encoders while simultaneously estimating epistemic uncertainty. These uncertainty estimates update gate residuals, allowing the model to adapt conservatively when evidence is weak. Furthermore, we introduce cross-modal confidence fusion based on Dempster-Shafer theory, enabling visual adaptation to be conditioned on textual confidence and suppressing conflicting or uncertain cross-modal updates. We conduct a comprehensive evaluation on 15 biomedical imaging datasets spanning 8 organs and 8 imaging modalities under few-shot learning and domain generalization settings. Evi-Steer consistently outperforms state-of-the-art methods under few-shot learning and domain shift settings, demonstrating a practical and robust pathway for deploying vision-language models in real-world clinical settings. Code is available at this https URL.
>
---
#### [replaced 113] Lying Is Just a Phase: The Hidden Alignment Transition in Language Model Scaling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究语言模型在扩展过程中的能力与真实性关系变化，揭示了能力与真实性从负相关到正相关的相变现象，提出一种无需模型内部信息的诊断方法。**

- **链接: [https://arxiv.org/pdf/2605.18838](https://arxiv.org/pdf/2605.18838)**

> **作者:** Adil Amin
>
> **备注:** 15 pages, 8 figures, 2 tables. Companion paper: "The Growing Pains of Frontier Models: When Leaderboards Stop Separating and What to Measure Next." ( this https URL). Code: this https URL. Dashboard: this https URL
>
> **摘要:** Scaling laws predict loss from compute but not how capabilities interact. We measure the coupling between reasoning and truthfulness across 63 base models from 16 families and find a regime change invisible to loss curves: below a family-dependent critical scale N_c, capabilities anticorrelate (r = -0.989, p = 4 x 10^{-5} nonparametric permutation test); above it, they cooperate. N_c ~ 3.5B parameters [2.9B, 13.4B] (bootstrap 95% CI), but model size is not the only variable that determines phase. Architecture, data curation, and training recipe each shift N_c independently: curated training eliminated the coupling dip between Qwen generations (0.025 to 0.830 at matched scale), Gemma-4 at 4B achieves coupling 0.871, characteristic of 13B+ standard-trained models, through distillation and architectural innovation, and Phi at 1B matches web-trained coupling at 10B through data curation alone. Width normalization eliminates the anticorrelation across all tested families, supporting an output-projection bottleneck. Internally, 38 of 40 models show zero competing attention heads. A sparse-regression ODE cross-predicts held-out Llama-2 at 5.6% error. The diagnostic requires no model internals -- only public benchmark scores across a model family. The cooperative regime extends to the frontier (r = +0.72, 34 models, 10 labs). A proof-of-concept intervention confirms the bottleneck is exploitable: adding a single truth-direction vector at the identified layer corrects 60% of misaligned outputs in the tax phase with zero retraining -- a surgical, per-inference correction that requires no weight modification. Code, data, an open-source steering CLI for any open-weight model, and an interactive dashboard for phase diagnosis are released: this https URL.
>
---
#### [replaced 114] From Global to Local: Learning Context-Aware Graph Representations for Document Classification and Summarization
- **分类: cs.CL**

- **简介: 该论文属于文档分类与摘要任务，旨在解决传统线性表示无法捕捉长文本全局结构的问题。通过构建图结构表示，利用注意力机制提升模型效果。**

- **链接: [https://arxiv.org/pdf/2603.00021](https://arxiv.org/pdf/2603.00021)**

> **作者:** Ruangrin Ldallitsakool; Margarita Bugueño; Gerard de Melo
>
> **摘要:** Recent NLP systems commonly represent documents as linear token sequences. Although this captures sequential order, it can hinder modeling long-range dependencies and global document structure, especially for long texts. This paper proposes a data-driven method to automatically construct graph-based document representations. Building upon the recent work of Bugueño and de Melo (2025), we leverage the dynamic sliding-window attention module to effectively capture local and mid-range semantic dependencies between sentences, as well as structural relations within documents. Graph Attention Networks (GATs) trained on our learned graphs achieve competitive results on document classification while requiring lower computational resources than previous approaches. We further present an exploratory evaluation of the proposed graph construction method for extractive document summarization, highlighting both its potential and current limitations. The implementation of this project can be found on GitHub.
>
---
#### [replaced 115] Hallucination Detection-Guided Preference Optimization for Clinical Summarization
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于临床摘要任务，旨在解决大语言模型生成摘要时出现的幻觉问题。通过幻觉检测引导的优化方法，减少错误陈述，提升摘要的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2605.28910](https://arxiv.org/pdf/2605.28910)**

> **作者:** Shamanth Kuthpadi Seethakantha; Dung Ngoc Thai; Vara Prasad Gudi; Simran Tiwari; Rami Matar; Avijit Mitra; Wenlong Zhao; Andrew McCallum; Wael Salloum
>
> **摘要:** Large language models (LLMs) have shown promise on summarization tasks, but they often produce hallucinations, which are unsupported or incorrect statements that limit their reliability in specialized healthcare applications. We introduce \itermodelfull (\itermodel), an inference-time method that leverages hallucination detectors to guide iterative summary revisions toward factual corrections. Building on this, we propose \itermodel for Preference Learning (\model), which converts detector-guided refinement trajectories into preference pairs for model finetuning. Extensive experiments show that our methods substantially reduce hallucinations for Llama and Gemma models in summarizing real-world clinical notes from \MimicIV. For example, \itermodel reduces 24\% and \model reduces 48\% hallucinations in Llama-3.1-8B-Instruct. Importantly, both methods preserve summary fluency, coherence, and relevance according to human expert and LLM-Jury evaluations. Together, these results demonstrate that detection-informed refinement and preference learning offer an automated solution for improving factual faithfulness in clinical summarization.
>
---
#### [replaced 116] KromHC: Manifold-Constrained Hyper-Connections with Kronecker-Product Residual Matrices
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出KromHC，解决mHC训练不稳定和参数复杂度高的问题，通过Kronecker积降低复杂度并保证双随机性，属于神经网络优化任务。**

- **链接: [https://arxiv.org/pdf/2601.21579](https://arxiv.org/pdf/2601.21579)**

> **作者:** Wuyang Zhou; Yuxuan Gu; Giorgos Iacovides; Danilo Mandic
>
> **摘要:** The success of Hyper-Connections (HC) in neural networks (NN) has also highlighted issues related to training instability and restricted scalability. The Manifold-Constrained Hyper-Connections (mHC) mitigate these challenges by projecting the residual connection space onto a Birkhoff polytope, however, it faces two issues: 1) its iterative Sinkhorn-Knopp (SK) algorithm does not always yield exactly doubly stochastic residual matrices; 2) mHC incurs a prohibitive $O(n^3C)$ parameter complexity with $n$ as the width of the residual stream and $C$ as the feature dimension. The recently proposed mHC-lite reparametrizes the residual matrix via the Birkhoff-von-Neumann theorem to guarantee double stochasticity, but also faces a factorial explosion in its parameter complexity, $O \left( nC \cdot n! \right)$. To address both challenges, we propose KromHC, which uses the Kronecker products of smaller doubly stochastic matrices to parametrize the residual matrix in mHC. By enforcing manifold constraints across the factor residual matrices along each mode of the tensorized residual stream, KromHC guarantees exact double stochasticity of the residual matrices while reducing parameter complexity to only $O(n^2C)$. Experiments show that KromHC matches or even outperforms other state-of-the-art (SOTA) mHC variants, while requiring significantly fewer trainable parameters. The code is at this https URL.
>
---
#### [replaced 117] Prototype Transformer: Towards Language Model Architectures Interpretable by Design
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出ProtoT，一种可解释的自回归语言模型架构，解决模型推理不透明的问题。通过原型机制替代注意力模块，提升可解释性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.11852](https://arxiv.org/pdf/2602.11852)**

> **作者:** Yordan Yordanov; Matteo Forasassi; Bayar Menzat; Ruizhi Wang; Chang Qi; Markus Kaltenberger; Amine M'Charrak; Tommaso Salvatori; Thomas Lukasiewicz
>
> **备注:** Accepted at ICML 2026. Equal contribution: Yordan Yordanov and Matteo Forasassi. 40 pages, 28 figures, 22 tables
>
> **摘要:** While state-of-the-art language models (LMs) surpass most humans in certain domains, their reasoning remains largely opaque, reducing trust and increasing the risk of deception and hallucination. We introduce the Prototype Transformer (ProtoT), an autoregressive LM architecture that replaces the quadratic-cost self-attention module of the Transformer with a linear-cost module based on prototypes, which are learned parameter vectors. In ProtoT, prototypes create communication channels that aggregate contextual information at different time scales. We show that this structure leads prototypes to automatically capture nameable concepts, such as "woman", during training, offering a path toward interpreting model reasoning and making targeted edits to model behavior. Compared with baselines, ProtoT scales well with model and data size, is robust to input perturbations, and performs well on text generation and downstream tasks, including GLUE. These results suggest that ProtoT is a promising step toward autoregressive language models that are more interpretable by design.
>
---
#### [replaced 118] Efficient LLM Moderation with Multi-Layer Latent Prototypes
- **分类: cs.LG; cs.AI; cs.CL; cs.CR**

- **简介: 该论文属于语言模型安全任务，解决部署时有害输出的监管问题。提出MLPM方法，通过多层原型提升 moderation 效率与定制性。**

- **链接: [https://arxiv.org/pdf/2502.16174](https://arxiv.org/pdf/2502.16174)**

> **作者:** Maciej Chrabąszcz; Filip Szatkowski; Bartosz Wójcik; Jan Dubiński; Tomasz Trzciński; Sebastian Cygert
>
> **摘要:** Although modern LLMs are aligned with human values during post-training, robust moderation remains essential to prevent harmful outputs at deployment time. Existing approaches suffer from performance-efficiency trade-offs and are difficult to customize to user-specific requirements. Motivated by this gap, we introduce Multi-Layer Prototype Moderator (MLPM), a lightweight and highly customizable input moderation tool. We propose leveraging prototypes of intermediate representations across multiple layers to improve moderation quality while maintaining high efficiency. By design, our method adds negligible overhead to the generation pipeline and can be seamlessly applied to any model. MLPM achieves state-of-the-art performance on diverse moderation benchmarks and demonstrates strong scalability across model families of various sizes. Moreover, we show that it integrates smoothly into end-to-end moderation pipelines and further improves response safety when combined with output moderation techniques. Overall, our work provides a practical and adaptable solution for safe, robust, and efficient LLM deployment.
>
---
#### [replaced 119] AblationBench: Evaluating Automated Planning of Ablations in Empirical AI Research
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出AblationBench，用于评估AI在实验消融规划中的能力，解决自动化科研评价难题。包含作者与审稿人两个任务，通过LM judges进行自动评估。**

- **链接: [https://arxiv.org/pdf/2507.08038](https://arxiv.org/pdf/2507.08038)**

> **作者:** Talor Abramovich; Gal Chechik
>
> **备注:** AI4Science Workshop, ICML 2026; Project page: this https URL
>
> **摘要:** Language model agents are increasingly used to automate scientific research, yet evaluating their scientific contributions remains a challenge. A key mechanism to obtain such insights is through ablation experiments. To this end, we introduce AblationBench, a benchmark suite for evaluating agents on ablation planning tasks in empirical AI research. It includes two tasks: AuthorAblation, which helps authors propose ablation experiments based on a method section and contains 83 instances, and ReviewerAblation, which helps reviewers find missing ablations in a full paper and contains 350 instances. For both tasks, we develop LM-based judges that serve as an automatic evaluation framework. Our experiments with frontier LMs show that these tasks remain challenging, with the best-performing LM system identifying only 45% of the original ablations on average, below human-level performance. We observe an inverse performance trend between the author and reviewer tasks, which we attribute to differences in model grounding. Lastly, we analyze the limitations of current LMs on these tasks, and find that chain-of-thought prompting outperforms an agent-based approach. Our data is available on this https URL, and our code is available on this https URL .
>
---
#### [replaced 120] From Graph Retrieval to Schema Realization: Counterfactual Validation for Text-to-SPARQL over Heterogeneous Knowledge Graphs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识图谱问答任务，解决异构知识图谱中文本到SPARQL查询生成的问题。提出SchemaForge框架，通过schema对齐和反事实验证提升查询准确性。**

- **链接: [https://arxiv.org/pdf/2508.01815](https://arxiv.org/pdf/2508.01815)**

> **作者:** Yang Zhao; Chengxiao Dai; Yue Xiu; Dusit Niyato
>
> **摘要:** Text-to-SPARQL maps natural-language questions to executable SPARQL queries over RDF knowledge graphs. While standard evaluations often fix the target graph in advance, practical knowledge graph question answering (KGQA) may involve heterogeneous graph collections with different schemas, partial alignments, and incomplete metadata. In this setting, query generation depends on more than SPARQL syntax: the system must identify a graph schema that can support the predicates, entity types, joins, filters, and constraints required by the question. We present SchemaForge, a schema-grounded agentic framework for text-to-SPARQL over heterogeneous KG collections. Its central mechanism is question-conditioned schema-slice alignment: weak graph evidence first identifies plausible graphs, while stronger schema evidence determines whether a local schema slice can realize the intended query. The selected schema slice then constrains query generation and verification before execution. When only one graph is available, the same formulation reduces to standard single-KG text-to-SPARQL with schema grounding. We evaluate SchemaForge on LC-QuAD 2.0, QALD-9 Plus, QALD-10, and Spider4SPARQL. Across the four public benchmarks, SchemaForge improves execution accuracy over the strongest matched agent baseline by 11.50 percentage points on average. On Spider4SPARQL, SchemaForge improves execution accuracy from 54.86% to 64.18% and achieves 73.0% Top-1 and 97.0% Top-3 graph allocation accuracy. These results show that moving from weak graph evidence to schema-specific query commitments, together with counterfactual answer-set checks, improves executable query generation over heterogeneous knowledge graphs.
>
---
#### [replaced 121] HiFi-KPI: A Dataset for Hierarchical KPI Extraction from Earnings Filings
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出HiFi-KPI数据集，用于解决财务报告中KPI提取的跨公司迁移问题。任务包括KPI分类、提取和结构化提取。**

- **链接: [https://arxiv.org/pdf/2502.15411](https://arxiv.org/pdf/2502.15411)**

> **作者:** Rasmus Aavang; Giovanni Rizzi; Rasmus Bøggild; Alexandre Iolov; Mike Zhang; Johannes Bjerva
>
> **摘要:** Accurate tagging of earnings reports can yield significant short-term returns for stakeholders. The machine-readable inline eXtensible Business Reporting Language (iXBRL) is mandated for public financial filings. Yet, its complex, fine-grained taxonomy limits the cross-company transferability of tagged Key Performance Indicators (KPIs). To address this, we introduce the Hierarchical Financial Key Performance Indicator (HiFi-KPI) dataset, a large-scale corpus of 1.65M paragraphs and 198k unique, hierarchically organized labels linked to iXBRL taxonomies. HiFi-KPI supports multiple tasks and we evaluate three: KPI classification, KPI extraction, and structured KPI extraction. For rapid evaluation, we also release HiFi-KPI-Lite, a manually curated 8K paragraph subset. Baselines on HiFi-KPI-Lite show that encoder-based models achieve over 0.906 macro-F1 on classification, while Large Language Models (LLMs) reach 0.440 F1 on structured extraction. Finally, a qualitative analysis reveals that extraction errors primarily relate to dates. We open-source all code and data at this https URL.
>
---
#### [replaced 122] Escaping the BLEU Trap: A Signal-Grounded Framework with Decoupled Semantic Guidance for EEG-to-Text Decoding
- **分类: cs.CL; cs.AI; cs.HC; eess.AS; q-bio.NC**

- **简介: 该论文属于EEG-to-Text解码任务，旨在解决语义偏差、信号忽视和BLEU陷阱问题。提出SemKey框架，通过分离语义目标和主动检索解码，提升生成质量与信号一致性。**

- **链接: [https://arxiv.org/pdf/2603.03312](https://arxiv.org/pdf/2603.03312)**

> **作者:** Yuchen Wang; Haonan Wang; Yu Guo; Honglong Yang; Xiaomeng Li
>
> **摘要:** Decoding natural language from non-invasive EEG signals is a promising yet challenging task. However, current state-of-the-art models remain constrained by three fundamental issues: Semantic Bias, where outputs collapse into generic linguistic templates; Signal Neglect, where models rely heavily on LLM priors to hallucinate fluent text even in the absence of meaningful signals; and the "BLEU Trap", where high-frequency stopwords inflate n-gram metrics, masking a lack of true semantic fidelity. To resolve these challenges, we move beyond conventional end-to-end pipelines and propose SemKey, a novel multi-stage framework that enforces signal-grounded generation through four decoupled semantic objectives: sentiment, topic, length, and surprisal. We extract these semantic anchors from EEG embeddings directly, then unify them with an Active Retrieval Decoding mechanism, compelling the LLM to ground its token generation in the neural signals rather than defaulting to linguistic priors. Furthermore, we break the BLEU Trap by establishing a comprehensive evaluation protocol using rigorous retrieval and distribution-based metrics such as Fréchet Distance. Extensive experiments demonstrate that SemKey effectively mitigates hallucinations on noise inputs and achieves SOTA performance on these robust protocols. Code will be released upon acceptance at this https URL.
>
---
#### [replaced 123] ReasonBENCH: Benchmarking the (In)Stability of LLM Reasoning
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于大模型推理评估任务，旨在解决LLM推理稳定性问题。通过构建基准测试，分析不同策略和模型的性能波动，提出结构化噪声分类，强调分布评估的重要性。**

- **链接: [https://arxiv.org/pdf/2512.07795](https://arxiv.org/pdf/2512.07795)**

> **作者:** Nearchos Potamitis; Vansh Ramani; Har Ashish Arora; Dhairya Kuchhal; Lars Klein; Akhil Arora
>
> **备注:** 29 pages, 19 tables, 85 figures
>
> **摘要:** Benchmark scores for LLM reasoning systems are reported as single numbers, yet the same model, strategy, and task can produce meaningfully different answers and costs across repeated executions, even under greedy decoding (T = 0). This variance is not a statistical nuisance: the highest-performing strategy wins only 77% of head-to-head runs against its nearest competitor, meaning a single observed score can silently misrank systems. We introduce ReasonBench, a benchmark suite recording 30 independent trials across 10 reasoning strategies, 12 models, and 6 tasks, treating quality and cost as distributions rather than point estimates. We find that this variance is structured rather than random: a two-component taxonomy -- Global Noise, capturing cross-benchmark unevenness, and Run Noise, capturing within-benchmark stochasticity -- reveals that strategy architecture predicts stability profiles, while models and strategies shift orthogonal aspects of the distribution. A hierarchical decomposition attributes three-quarters of score variance to benchmark, system, and item structure, with a persistent residual that single-run evaluation silently absorbs. Finally, cost and quality decouple asymmetrically: cheap methods are structurally immune to joint cost-quality failure, while expensive methods remain exposed regardless of their accuracy. These findings establish instability as an inherent property of reasoning systems and motivate distribution-aware evaluation as standard practice.
>
---
#### [replaced 124] Backtranslation Augmented Direct Preference Optimization for Neural Machine Translation
- **分类: cs.CL**

- **简介: 该论文属于神经机器翻译任务，旨在解决翻译错误问题。通过引入基于强化学习的后训练框架，利用偏好优化提升翻译质量。**

- **链接: [https://arxiv.org/pdf/2604.25702](https://arxiv.org/pdf/2604.25702)**

> **作者:** Mehrdad Ghassabi; Spehr Rajabi; Hamidreza Baradaran Kashani; Sadra Hakim; Mahshid Keivandarian; Amirhossein Jahani Bahnamiri
>
> **备注:** 5 pages, 2 figures
>
> **摘要:** Contemporary neural machine translation (NMT) systems are almost exclusively built by training on supervised parallel data. Despite the tremendous progress achieved, these systems still exhibit persistent translation errors. This paper proposes that a post-training paradigm based on reinforcement learning (RL) can effectively rectify such mistakes. We introduce a novel framework that requires only a general text corpus and an expert translator which can be either human or an AI system to provide iterative feedback. In our experiments, we focus specifically on English-to-German translation as a representative high-resource language pair. Crucially, we implement this RL-based post-training using Direct Preference Optimization (DPO). Applying our DPO-driven framework to the gemma3-1b model yields a significant improvement in translation quality, elevating it's COMET score from 0.703 to 0.747 on the English to German task. The results demonstrate that DPO offers an efficient and stable pathway for enhancing pre-trained NMT models through preference-based post-training.
>
---
#### [replaced 125] A Systematic Benchmark of Machine Transliteration Models for the Tajik-Farsi Language Pair: A Comparative Study from Rule-Based to Transformer Architectures
- **分类: cs.CL**

- **简介: 该论文属于机器 transliteration 任务，旨在比较不同模型在塔吉克语与波斯语之间的 transliteration 效果。研究构建了首个平行语料库，对比了多种模型，发现 byte-level 模型表现最佳。**

- **链接: [https://arxiv.org/pdf/2605.02270](https://arxiv.org/pdf/2605.02270)**

> **作者:** Mullosharaf K. Arabov
>
> **备注:** Accepted to CLIB 2026
>
> **摘要:** This paper presents the first comprehensive comparative analysis of modern machine learning architectures for transliteration between Tajik (Cyrillic script) and Persian (Arabic script). A key contribution is the creation and validation of a unique parallel corpus aggregated from multiple heterogeneous sources, including crowdsourced projects, lexicographic pairs, parallel texts of "Shahnameh", diplomatic articles, texts of "Masnavi-i Ma'navi", official terminology lists, and transliterated correspondences. The initial dataset comprised 328,253 sentence pairs; a representative subset of 40,000 pairs was formed using stratified random sampling. The experiment compared six classes of models: rule-based baseline, LSTM with attention, character-level Transformer, G2P Transformer (trained from scratch), pre-trained multilingual models (mBART, mT5 with LoRA), and byte-level ByT5. Results demonstrate the overwhelming superiority of ByT5 (chrF++ 87.4 for Tajik to Farsi, 80.1 for reverse). The G2P Transformer significantly outperformed mBART (72.3 vs. 62.2 chrF++) despite limited data. Models using subword tokenization (mT5) failed completely (chrF++ less than 18.5). The findings demonstrate that for accurate transliteration of the Tajik-Farsi pair, architectures operating at the byte or character level are unequivocally more effective than traditional multilingual Seq2Seq models relying on subword tokenization.
>
---
#### [replaced 126] Test-Time Compute for Frozen Embedding Models through Agentic Program Search
- **分类: cs.LG; cs.CL; cs.IR**

- **简介: 该论文研究小嵌入模型在推理时通过程序搜索提升检索效果的任务，解决如何在不训练参数的情况下利用计算资源提升性能的问题。工作包括设计代理循环生成优化程序，验证其有效性。**

- **链接: [https://arxiv.org/pdf/2605.11374](https://arxiv.org/pdf/2605.11374)**

> **作者:** Han Xiao
>
> **备注:** 15 pages, 7 figures, 4 tables
>
> **摘要:** Test-time compute is widely believed to benefit only large reasoning models, leaving small models with nothing to gain. We argue the opposite for dense retrieval, since modern small embedding models are distilled or adapted from large language model backbones and can inherit their latent test-time-compute potential. We ask how much retrieval quality a frozen embedding model gains at inference alone, with no auxiliary model and no parameters trained at deployment. An agentic loop in which a large language model writes programs over a frozen encoder API explores 144 candidates and yields twelve Pareto-optimal programs that trade inference compute for quality across cost ratios from $c{=}1.2$ to $14.7$, every one improving nDCG@10 on all 14 discovery tasks. The programs use no trainable parameters and recover classical retrieval primitives, among them reciprocal rank fusion, the Fisher linear discriminant, Rocchio pseudo-relevance feedback, and sentence-level MaxSim. Applied unmodified to nineteen held-out tasks and three unseen encoder families, a single fixed program improves the majority of tasks, with a positive median $\Delta$nDCG@10 and a 54 to 57% win-rate at $c{\ge}4$, and the gains are largest on encoder families never seen during discovery. A matched-budget learned projection head trained on the same tasks does not transfer this way, improving in-domain retrieval by $+0.20$ to $+0.25$ nDCG@10 yet falling below baseline on every held-out encoder. Small embedding models therefore inherit usable test-time-compute potential, and a frozen encoder converts inference compute into retrieval gains that transfer to new corpora and encoders with no per-domain labels.
>
---
#### [replaced 127] When Single Answer Is Not Enough: Rethinking Single-Step Retrosynthesis Benchmarks for LLMs
- **分类: cs.LG; cs.AI; cs.CE; cs.CL**

- **简介: 该论文属于药物合成规划任务，旨在解决现有基准评估方法不足的问题。通过引入新指标和数据集，提升对大语言模型合成规划能力的评价效果。**

- **链接: [https://arxiv.org/pdf/2602.03554](https://arxiv.org/pdf/2602.03554)**

> **作者:** Bogdan Zagribelnyy; Ivan Ilin; Maksim Kuznetsov; Nikita Bondarev; Mathieu Reymond; Roman Schutski; Thomas MacDougall; Rim Shayakhmetov; Zulfat Miftakhutdinov; Mikolaj Mizera; Vladimir Aladinskiy; Alex Aliper; Alex Zhavoronkov
>
> **摘要:** Recent progress has expanded the use of large language models (LLMs) in drug discovery, including synthesis planning. However, objective evaluation of retrosynthesis performance remains limited. Existing benchmarks and metrics typically rely on published synthetic procedures and Top-K accuracy based on single ground-truth, which does not capture the open-ended nature of real-world synthesis planning. We propose a new benchmarking framework for single-step retrosynthesis that evaluates both general-purpose and chemistry-specialized LLMs using ChemCensor, a novel metric for chemical plausibility. By emphasizing plausibility over exact match, this approach better aligns with human synthesis planning practices. We also introduce CREED, a novel dataset comprising millions of ChemCensor-validated reaction records for LLM training, and use it to train a model that improves over the LLM baselines under this benchmark.
>
---
#### [replaced 128] NormEval: A Unified Multi-Metric Framework for Evaluating Semantic Fidelity in Text Normalization
- **分类: cs.CL**

- **简介: 该论文属于文本归一化任务，旨在解决评估方法碎片化问题。提出NormEval框架，包含五项指标，从不同维度评估归一化质量。**

- **链接: [https://arxiv.org/pdf/2511.20409](https://arxiv.org/pdf/2511.20409)**

> **作者:** Md Abdullah Al Kafi; Raka Moni; Walayat Hussain
>
> **摘要:** Text normalization methods such as stemming and lemmatization are fundamental components of NLP pipelines. As new normalization tools are developed for diverse languages, evaluation methodologies remain fragmented, relying on Compression Ratio, downstream accuracy, or sequence-to-sequence prediction scores in isolation, failing to distinguish between beneficial vocabulary reduction and harmful semantic distortion. Moreover, text normalization underpins intelligent systems in high-stakes domains, including clinical decision support and legal document analysis, and principled evaluation methodology is essential. This paper proposes NormEval, a unified, multilingual evaluation framework comprising five complementary metrics: Compression Ratio (CR), Model Performance Delta (MPD), Information Retention Score (IRS), Algorithm Effectiveness Score (AES), and Average Normalized Levenshtein Distance (ANLD). These metrics assess normalization quality across three dimensions: macro-level efficiency, downstream utility, and micro-level morphological fidelity. The framework operationalizes a Safety Gate hypothesis: ANLD functions as an intrinsic structural hygiene check, utilizing character-level divergence ($\Delta$) to reveal aggressive mutations that macro-level embeddings and downstream tasks mask. Comprehensive ablation experiments on both Bangla and English datasets show that all the components are indispensable, and that the removal of any individual metric leads to a decrease in at least one evaluation aspect, which ultimately results in misleading algorithm rankings.
>
---
#### [replaced 129] Beyond Semantic Understanding: Preserving Collaborative Frequency Components in LLM-based Recommendation
- **分类: cs.CL**

- **简介: 该论文属于推荐系统任务，旨在解决LLM在推荐中弱化协同信号的问题。通过频域方法保留协同信息，提升推荐效果。**

- **链接: [https://arxiv.org/pdf/2508.10312](https://arxiv.org/pdf/2508.10312)**

> **作者:** Minhao Wang; Yunhang He; Cong Xu; Zhangchi Zhu; Shuang Hao; Ning Liu; Wei Zhang
>
> **备注:** 12 pages, 7 figures
>
> **摘要:** Recommender systems in concert with Large Language Models (LLMs) present promising avenues for generating semantically-informed recommendations. However, LLM-based recommenders exhibit a tendency to overemphasize semantic correlations within users' interaction history. When taking pretrained collaborative ID embeddings as input, LLM-based recommenders progressively weaken the inherent collaborative signals as the embeddings propagate through LLM backbones layer by layer, as opposed to traditional Transformer-based sequential models in which collaborative signals are typically preserved or even enhanced for state-of-the-art performance. To address this limitation, we introduce FreLLM4Rec, an approach designed to balance semantic and collaborative information from a spectral perspective. Item embeddings that incorporate both semantic and collaborative information are first purified using a Global Graph Low-Pass Filter (G-LPF) to preliminarily remove irrelevant high-frequency noise. Temporal Frequency Modulation (TFM) then actively preserves collaborative signal layer by layer. Note that the collaborative preservation capability of TFM is theoretically guaranteed by establishing a connection between the optimal but hard-to-implement local graph fourier filters and the suboptimal yet computationally efficient frequency-domain filters. Extensive experiments on four benchmark datasets demonstrate that FreLLM4Rec successfully mitigates collaborative signal attenuation and achieves competitive performance, with improvements of up to 8.00\% in NDCG@10 over the best baseline. Our findings provide insights into how LLMs process collaborative information and offer a principled approach for improving LLM-based recommendation systems.
>
---
#### [replaced 130] GeistBERT: Breathing Life into German NLP
- **分类: cs.CL**

- **简介: 该论文提出GeistBERT，用于改进德语自然语言处理任务。针对德语特点，通过预训练和优化模型性能，解决了德语NLP效果不足的问题。**

- **链接: [https://arxiv.org/pdf/2506.11903](https://arxiv.org/pdf/2506.11903)**

> **作者:** Raphael Scheible-Schmitt; Johann Frei
>
> **摘要:** Advances in transformer-based language models have highlighted the benefits of language-specific pre-training on high-quality corpora. In this context, German NLP stands to gain from updated architectures and modern datasets tailored to the linguistic characteristics of the German language. GeistBERT seeks to improve German language processing by incrementally training on a diverse corpus and optimizing model performance across various NLP tasks. We pre-trained GeistBERT using fairseq, following the RoBERTa base configuration with Whole Word Masking (WWM), and initialized from GottBERT weights. The model was trained on a 1.3 TB German corpus with dynamic masking and a fixed sequence length of 512 tokens. For evaluation, we fine-tuned the model on standard downstream tasks, including NER (CoNLL 2003, GermEval 2014), text classification (GermEval 2018 coarse/fine, 10kGNAD), and NLI (German XNLI), using $F_1$ score and accuracy as evaluation metrics. GeistBERT achieved strong results across all tasks, leading among base models and setting a new state-of-the-art (SOTA) in GermEval 2018 fine text classification. It also outperformed several larger models, particularly in classification benchmarks. To support research in German NLP, we release GeistBERT under the MIT license.
>
---
#### [replaced 131] ASKD-Whisper: Adaptive Self-knowledge Distillation for Efficient and Low-Latency Automatic Speech Recognition
- **分类: cs.CL; cs.AI; cs.SD**

- **简介: 该论文属于自动语音识别（ASR）任务，旨在解决知识蒸馏中学生模型过度依赖教师模型导致的泛化能力下降问题。提出ASGD方法，提升模型压缩效果与推理效率。**

- **链接: [https://arxiv.org/pdf/2601.19919](https://arxiv.org/pdf/2601.19919)**

> **作者:** Junseok Lee; Nahun Kim; Sangyong Lee; Chang-Jae Chun
>
> **备注:** Title and content have been updated
>
> **摘要:** Knowledge distillation (KD) is one of the most effective paradigms for compressing large-scale foundation models into deployable architectures. In the context of Automatic Speech Recognition (ASR), previous studies have predominantly focused on forcing the student model to strictly mimic the predictive distribution of a massive teacher model. However, this static dependency often presents an inherent trade-off: while the student rapidly acquires basic linguistic representations, it simultaneously inherits the teacher's domain-specific blind spots and over-confident hallucinations, leading to a severe decline in out-of-distribution generalization capacity. To effectively mitigate this issue, we propose Adaptive Self-Knowledge Distillation (ASKD), a dynamic curriculum framework. ASKD systematically decays the dependency on the teacher's distribution as training progresses-thereby unlocking the student's independent reasoning capacity-and subsequently employs a self-knowledge distillation phase to act as a structural regularizer. By applying ASKD, we distill the massive Whisper architecture into a compact variant, ASKD-Whisper. In our comprehensive evaluations across diverse acoustic domains, ASKD-Whisper not only achieves a 5x speedup in inference latency but also outperforms its teacher model by yielding a 1.07% lower word error rate (WER). These results demonstrate that ASKD effectively prevents teacher-induced overfitting and establishes a new state-of-the-art for generalizable model compression.
>
---
#### [replaced 132] Characterizing the Effect of Noise in Language Generation in the Limit
- **分类: cs.DS; cs.CL; cs.LG**

- **简介: 该论文研究语言生成中的噪声影响，属于理论计算机科学任务。解决噪声如何影响生成能力的问题，证明单个噪声字符串与有限噪声等价，并揭示非均匀生成的特性。**

- **链接: [https://arxiv.org/pdf/2601.21237](https://arxiv.org/pdf/2601.21237)**

> **作者:** Aaron Li; Ian Zhang
>
> **备注:** ICML 2026
>
> **摘要:** Kleinberg and Mullainathan recently proposed a formal framework for studying the phenomenon of language generation, called language generation in the limit. In this model, an adversary gives an enumeration of example strings from an unknown target language, and the algorithm is tasked with correctly generating unseen strings from the target language within finite time. Refined notions of non-uniform and uniform generation were later introduced by Li, Raman, and Tewari (2025), and a noisy model was introduced by Raman and Raman (2025), which allows the adversary to insert extraneous strings. A natural question in the noisy model is to quantify the effect of noise, by studying the impact of each additional extraneous string. We show two complementary results in this setting. We first show that for both uniform and non-uniform generation, a single noisy string strictly reduces the set of collections that can be generated, thus answering an open question in Raman and Raman (2025). Then, we show for both uniform and non-uniform generation that generation with a single noisy string is equivalent to generation with any finite amount of noise, sharply contrasting with the strict hierarchy for noisy generation in the limit shown by Bai, Panigrahi, and Zhang (2026). Finally, we leverage our previous results to provide the first known characterization for non-uniform noise-dependent generatability.
>
---
#### [replaced 133] Are LLMs Ready for Neural-integrated Mechanistic Modeling? A Benchmark and Agentic Framework
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于科学建模任务，旨在解决LLMs在神经整合机制建模中的有效性问题。研究提出NIMM基准和NIMMGen框架，提升模型搜索稳定性与质量。**

- **链接: [https://arxiv.org/pdf/2602.18008](https://arxiv.org/pdf/2602.18008)**

> **作者:** Zihan Guan; Rituparna Datta; Mengxuan Hu; Shunshun Liu; Aiying Zhang; Prasanna Balachandran; Sheng Li; Anil Vullikanti
>
> **备注:** 25 pages, 8 figures
>
> **摘要:** Large language models (LLMs) have shown promise in constructing mechanistic models from data. However, existing evaluations largely focus on simplified settings and fail to capture the complexity of real-world scientific modeling. In practice, such modeling often involves neural-integrated formulations, where a mechanistic model component and a neural network component are jointly constructed, leading to a significantly more complex search space. Motivated by this gap, we introduce the Neural-Integrated Mechanistic Modeling (NIMM) benchmark, which evaluates LLM-generated neural-integrated mechanistic models across three scientific domains. Experiments on NIMM reveal that existing LLM-based approaches struggle to effectively explore this complex space, resulting in limited search stability and solution quality. To address this challenge, we propose NIMMGen, a tree-guided agentic framework that enables diversified exploration via branch-level search and improves solutions through atomic model refinement. Extensive experiments demonstrate that NIMMGen achieves state-of-the-art performance on NIMM, significantly improving search stability and solution quality.
>
---
#### [replaced 134] EuroBERT: Scaling Multilingual Encoders for European Languages
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出EuroBERT，一种多语言编码器模型，解决多语言任务中的表示学习问题。通过优化设计，提升多语言、数学和代码任务性能。**

- **链接: [https://arxiv.org/pdf/2503.05500](https://arxiv.org/pdf/2503.05500)**

> **作者:** Nicolas Boizard; Hippolyte Gisserot-Boukhlef; Duarte M. Alves; André Martins; Ayoub Hammal; Caio Corro; Céline Hudelot; Emmanuel Malherbe; Etienne Malaboeuf; Fanny Jourdan; Gabriel Hautreux; João Alves; Kevin El Haddad; Manuel Faysse; Maxime Peyrard; Nuno M. Guerreiro; Patrick Fernandes; Ricardo Rei; Pierre Colombo
>
> **备注:** 28 pages, 8 figures, 13 tables
>
> **摘要:** General-purpose multilingual vector representations, used in retrieval, regression and classification, are traditionally obtained from bidirectional encoder models. Despite their wide applicability, encoders have been recently overshadowed by advances in generative decoder-only models. However, many innovations driving this progress are not inherently tied to decoders. In this paper, we revisit the development of multilingual encoders through the lens of these advances, and introduce EuroBERT, a family of multilingual encoders covering European and widely spoken global languages. Our models outperform existing alternatives across a diverse range of tasks, spanning multilingual capabilities, mathematics, and coding, and natively supporting sequences of up to 8,192 tokens. We also examine the design decisions behind EuroBERT, offering insights into our dataset composition and training pipeline. We publicly release the EuroBERT models, including intermediate training checkpoints, together with our training framework.
>
---
#### [replaced 135] $M^3$ Scaling Law: Optimizing Multi-Epoch, Multi-Lingual, and Multi-Stage Training for Low-Resource Language Models
- **分类: cs.CL**

- **简介: 该论文属于低资源语言模型预训练任务，解决多阶段、多语言训练的优化问题，提出M³尺度定律以指导最佳训练方案。**

- **链接: [https://arxiv.org/pdf/2410.12325](https://arxiv.org/pdf/2410.12325)**

> **作者:** Kosuke Akimoto; Taiki Miyagawa; Masafumi Oyamada
>
> **备注:** 35 pages, 14 figures, 17 tables
>
> **摘要:** In this paper, we study a fundamental design problem in pretraining Large Language Models (LLMs) for low-resource language regimes. Existing works adopt multi-epoch, multi-lingual, and multi-stage training to utilize the limited target-language corpus efficiently, but no prior scaling law can compare recipes spanning these approaches under the same compute budget $C$ and target-language corpus size $D_T$, leaving the optimal training setup unclear. To address this gap, we propose the $M^3$ Scaling Law, a unified predictive model parameterized by the model scale, the number of target-corpus epochs $k$, the average target-language ratio $r$, and the final-stage target-language ratio $r_f$, which places monolingual single-stage, multi-lingual single-stage, and multi-lingual multi-stage recipes on a single target-language loss surface. Across three language pairs, it extrapolates to unseen hyperparameter regions more accurately than existing scaling laws. Using $M^3$ as a surrogate objective, we derive two practical guidelines for low-resource LLM pretraining: (i) as $D_T$ decreases, the optimal recipe shifts directly from monolingual single-stage to multi-lingual two-stage training at a compute-budget-dependent threshold, with multi-lingual single-stage never optimal in our experimental grid; and (ii) the optimal number of epochs collapses onto a single curve in the scarcity variable $D_T/D^*(C)$, where $D^*(C) \propto C^{\alpha/(\alpha+\beta)}$ is the monolingual compute-optimal corpus size.
>
---
#### [replaced 136] Anatomy of Unlearning: The Dual Impact of Fact Salience and Model Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文属于机器学习中的遗忘任务，旨在解决模型如何有效删除特定信息的问题。研究提出DUET基准，分析预训练与微调模型在遗忘中的差异，提升遗忘效果与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.19612](https://arxiv.org/pdf/2602.19612)**

> **作者:** Anna Borisiuk; Andrey Savchenko; Alexander Panchenko; Elena Tutubalina
>
> **摘要:** Machine Unlearning (MU) enables Large Language Models (LLMs) to remove unsafe or outdated information. However, existing work assumes that all facts are equally forgettable and largely ignores whether the forgotten knowledge originates from pretraining or supervised fine-tuning (SFT). In this paper, we introduce DUET (Dual Unlearning Evaluation across Training Stages), a benchmark of 28.6k Wikidata-derived triplets annotated with fact popularity using Wikipedia link counts and LLM-based salience scores. Our experiments show that pretrained and SFT models respond differently to unlearning. An SFT step on the forget data yields smoother forgetting, more stable tuning, and 10-50% higher retention, while direct unlearning on pretrained models remains unstable and prone to relearning or catastrophic forgetting.
>
---
#### [replaced 137] Probing Minimalist Phase Structure in LLMs: What Universal Dependencies Cannot Represent
- **分类: cs.CL; stat.AP**

- **简介: 该论文属于自然语言处理中的语法结构研究任务，旨在探讨大语言模型是否编码形式句法抽象。通过设计实验验证LLMs是否能捕捉UD无法表达的最小短语结构，发现模型确实具备此类表征。**

- **链接: [https://arxiv.org/pdf/2605.26431](https://arxiv.org/pdf/2605.26431)**

> **作者:** Yuanhao Chen; Peter Chin
>
> **摘要:** Structural probes train on Universal Dependencies (UD), which does not encode formal-syntactic abstractions such as phase boundaries or phase-internal cohesion. Whether large language models (LLMs) encode these remains an open question that UD-based probing cannot answer by construction. We evaluate structural probes on wh-movement stimuli where UD distances are invariant across conditions by design -- any non-zero effect therefore reflects structure beyond UD. The three conditions -- bare small clause, infinitival, and finite -- are ordered by the number of Minimalist Program (MP) phase boundaries the wh-element crosses. Across 13 LLMs from four families, we find a phase-count gradient on a cross-clause pair (12/13 models) and a 13/13 sign asymmetry on a within-clause pair whose UD distance is identical across conditions -- the latter specifically predicted by phase-internal cohesion, an MP abstraction invisible to UD by construction. Activation patching confirms the representations are causally active in 12/13 models. These findings suggest that distributional pretraining can induce representations aligned with formal-syntactic abstractions beyond the reach of annotation-based probing; UD-grounded probes provide a lower bound on syntactic encoding, not an upper bound.
>
---
#### [replaced 138] CURP: Codebook-based Continuous User Representation for Personalized Generation with LLMs
- **分类: cs.CL**

- **简介: 该论文属于个性化生成任务，旨在解决LLMs个性化与效率难以平衡的问题。提出CURP框架，通过用户编码器和代码本实现高效、可解释的个性化生成。**

- **链接: [https://arxiv.org/pdf/2602.00742](https://arxiv.org/pdf/2602.00742)**

> **作者:** Liang Wang; Xinyi Mou; Xiaoyou Liu; Xuanjing Huang; Zhongyu Wei
>
> **摘要:** User modeling characterizes individuals through their preferences and behavioral patterns to enable personalized simulation and generation with Large Language Models (LLMs) in contemporary approaches. However, existing methods, whether prompt-based or training-based methods, face challenges in balancing personalization quality against computational and data efficiency. We propose a novel framework CURP, which employs a bidirectional user encoder and a discrete prototype codebook to extract multi-dimensional user traits. This design enables plug-and-play personalization with a small number of trainable parameters (about 20M parameters, about 0.2\% of the total model size). Through extensive experiments on variant generation tasks, we show that CURP achieves superior performance and generalization compared to strong baselines, while offering better interpretability and scalability. The code are available at this https URL
>
---
#### [replaced 139] Skill-Based Mixture-of-Experts: Adaptive Routing for Heterogeneous Reasoning via Inferred Skills
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出Skill-MoE，解决多任务推理中专家选择不精准的问题。通过实例级技能分析，实现高效专家路由与结果整合。**

- **链接: [https://arxiv.org/pdf/2503.05641](https://arxiv.org/pdf/2503.05641)**

> **作者:** Justin Chih-Yao Chen; Sukwon Yun; Elias Stengel-Eskin; Tianlong Chen; Mohit Bansal
>
> **备注:** ICML 2026 (Camera-Ready). The first three authors contributed equally. Project Page: this https URL
>
> **摘要:** Combining existing pre-trained LLMs is a promising approach for diverse reasoning tasks. However, task-level expert selection is often too coarse-grained, since different instances may require different expertise. To address this, we propose Skill-MoE, a symbolic, skill-based, and gradient-free Mixture-of-Experts framework for instance-level expert selection. Skill-MoE infers skills (e.g., algebra in mathematics) from each query, selects experts based on skill relevance, and lets each expert generate its own reasoning. The resulting k outputs are then synthesized by an aggregator chosen for its ability to integrate diverse responses. While instance-level selection substantially improves performance, naively implementing it incurs heavy overhead from repeated model loading and offloading. We address this with a batch inference strategy that groups instances by assigned experts, allowing each model to be loaded only once. As a result, Skill-MoE integrates 16 expert models on a single GPU with runtime comparable to prior multi-agent baselines using 4 GPUs. Across diverse benchmarks (MMLU-Pro, GPQA, AIME, and MedMCQA), Skill-MoE achieves an average absolute improvement of 8.15% over the best baseline. It also generalizes well to unseen tasks and outperforms discussion-based methods without requiring expensive multi-round interactions.
>
---
#### [replaced 140] BenGER: Benchmarking LLM Systems on Subsumption-Based Legal Reasoning in German Law
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于法律推理任务，旨在评估大语言模型在德国法律中的子类推理能力。通过构建BenGER数据集，对比不同模型性能，探索人机协作效果。**

- **链接: [https://arxiv.org/pdf/2605.28183](https://arxiv.org/pdf/2605.28183)**

> **作者:** Sebastian Nagl; Ann-Kristin Mayrhofer; Martin Heidebach; Aleyna Koçak; Anne Zettelmeier; Elly Breu; Angelina Greiner; Sofija Milijas; Matthias Grabmair
>
> **备注:** Pre-Print v2
>
> **摘要:** We introduce the BenGER (Benchmark for German Law) dataset for evaluating LLM systems on subsumption-based legal reasoning in German law. The BenGER dataset consists of three components: 596 exam-style free-text legal case tasks across multiple levels of legal education and 531 short doctrinal reasoning tasks. We evaluate 12 contemporary LLM systems -- closed flagship, efficiency-oriented, and open-weight -- across automatic and judge-based metrics. On a controlled validation subset of timed human-written solutions under both unaided and human--AI co-creation conditions, we contextualise model performance against these human baselines. We introduce a rubric-aligned LLM-as-a-Judge framework cross-validated against a multi-rater human-grading protocol (three blind reviews plus one author-informed creator review per solution). Our results show that replacing a blind human reviewer with the LLM judge degrades agreement with the full human pool no more than removing that reviewer altogether (Calderon r=0.96 vs.~r=0.96, matched n=30), that closed-flagship systems lead the leaderboard across all corpora, and that human--AI co-creation substantially outperforms unaided human work.
>
---
#### [replaced 141] WaterSearch: Exploring Seed Pooling for Improving the Quality-Detectability Trade-off in LLM Watermarking
- **分类: cs.CL**

- **简介: 该论文属于LLM水印任务，旨在解决水印可检测性与文本质量的权衡问题。通过设计WaterSearch框架，优化生成文本的质量和水印特性。**

- **链接: [https://arxiv.org/pdf/2512.00837](https://arxiv.org/pdf/2512.00837)**

> **作者:** Yukang Lin; Jiahao Shao; Shuoran Jiang; Wentao Zhu; Bingjie Lu; Xiangping Wu; Joanna Siebert; Qingcai Chen
>
> **摘要:** Watermarking acts as a critical safeguard in text generated by Large Language Models (LLMs). By embedding identifiable signals into model outputs, watermarking enables reliable attribution and enhances the security of machine-generated content. Existing approaches typically embed signals by manipulating token generation probabilities. Despite their effectiveness, these methods inherently face a trade-off between detectability and text quality: the signal strength and randomness required for robust watermarking tend to degrade the performance of downstream tasks. In this paper, we design a novel embedding scheme that controls seed pools to facilitate diverse parallel generation of watermarked text. Based on that scheme, we propose WaterSearch, a sentence-level, search-based watermarking framework adaptable to a wide range of existing methods. WaterSearch enhances text quality by jointly optimizing two key aspects: 1) distribution fidelity and 2) watermark signal characteristics. Furthermore, WaterSearch is complemented by a sentence-level detection method with strong attack robustness. We evaluate our method on three popular LLMs across ten diverse tasks. Extensive experiments demonstrate that our method achieves an average performance improvement of 51.01\% over state-of-the-art baselines at a watermark detectability strength of 95\%. In challenging scenarios such as short text generation and low-entropy output generation, our method yields performance gains of 47.78\% and 36.47\%, respectively. Moreover, under different attack senarios including insertion, synonym substitution and paraphrase attasks, WaterSearch maintains high detectability, further validating its robust anti-attack capabilities. Our code is available at \href{this https URL}{this https URL}.
>
---
#### [replaced 142] A Data-Driven Approach to Idiomaticity Based on Experts' Criteria in Theoretical Linguistics
- **分类: cs.CL**

- **简介: 论文探讨了多词表达的习语性，通过专家标注分析16个标准，发现无绝对习语表达。属于自然语言处理中的习语识别任务，旨在理解习语性判断标准及影响因素。**

- **链接: [https://arxiv.org/pdf/2605.19575](https://arxiv.org/pdf/2605.19575)**

> **作者:** Elena Mikhalkova; Anastasiya Vishnyakova; Anastasiya Drozdova; Polina Gavin; Aleksander Zhmykhov; Timofey Protasov
>
> **摘要:** The article observes data analysis of 286 multi-word expressions (MWEs) based on 16 lexical, grammatical and other criteria described in theoretical books and papers on the notion of idiomaticity. MWEs were collected from the same theoretical sources, and a set of experts in linguistics annotated them with these categories. The distribution of categories shows that there are no absolutely idiomatic expressions. Lexical criteria seem to be the most influential; grammatical criteria are bound to certain conditions; presence of obsolete words and grammar influence ability of an MWE to be replaced with one word.
>
---
#### [replaced 143] Navigating the Reality Gap: On-Device Continual Adaptation of ASR for Clinical Telephony
- **分类: cs.CL**

- **简介: 该论文属于语音识别任务，解决临床电话场景下ASR模型性能下降的问题。通过研究数据与参数层面的适应机制，提升模型在受限环境下的持续适应能力。**

- **链接: [https://arxiv.org/pdf/2512.16401](https://arxiv.org/pdf/2512.16401)**

> **作者:** Darshil Chauhan; Adityasinh Solanki; Vansh Patel; Kanav Kapoor; Ritvik Jain; Aditya Bansal; Pratik Narang; Dhruv Kumar
>
> **备注:** 17 pages. Under review
>
> **摘要:** Automatic Speech Recognition (ASR) can significantly reduce documentation burden in clinical workflows, but standard models degrade sharply in real-world telephony settings where noisy audio, dialectal variation, and strict data residency constraints prevent cloud-based adaptation. We study this "reality gap" using Gram Vaani: a telephonic Hindi corpus spanning rural healthcare and agricultural helplines, as the closest available proxy for clinical speech under strict on-device constraints. We show that a robust multilingual model (IndicWav2Vec) degrades from 11.59\% WER on standard clean Hindi to \textbf{41.71\% WER} on this proxy telephony data. We evaluate a progression of on-device adaptation regimes under realistic constraints, from full fine-tuning to parameter-efficient LoRA and stream-based continual learning, across multiple baselines, datasets, and seeds. Focusing on continual learning, our central finding highlights a critical interaction between Experience Replay (ER) and Elastic Weight Consolidation (EWC, parameterized by regularization strength $\lambda$). We show that standard positive EWC ($\lambda > 0$) can oppose replay-driven updates, limiting adaptation. Reversing EWC's strength ($\lambda < 0$) suggests that it can act as a directional control signal under ER-guided adaptation: negative $\lambda$ reinforces replay-driven plasticity, while a scheduled $\lambda$ enables phase-dependent control of stability and plasticity. Across evaluations on multiple datasets, we find that multi-domain replay provides a strong foundation for adaptation, while EWC modulates stability-plasticity dynamics without altering final performance. These results show that effective on-device adaptation depends on understanding how data-driven and parameter-level learning signals interact, rather than choosing methods in isolation.
>
---
#### [replaced 144] NanoSpec: Accelerating Speculative Decoding using Minimalist In-Context Vocabularies
- **分类: cs.CL**

- **简介: 该论文提出NanoSpec，解决大语言模型中词汇量过大导致的推理速度瓶颈问题。通过动态构建最小上下文词表，显著减少计算开销，提升解码效率。**

- **链接: [https://arxiv.org/pdf/2605.26444](https://arxiv.org/pdf/2605.26444)**

> **作者:** Zhiyang Chen; Daliang Xu; Yinyuan Zhang; Chenghua Wang; Mengwei Xu; Yun Ma
>
> **摘要:** The massive vocabulary sizes of large language models, often exceeding 100k tokens, impose a computational bottleneck on the final linear projection layer during speculative decoding. Existing vocabulary pruning solutions rely on static or coarsely-grained sub-vocabularies that necessitate large active sizes ($\sim$30k) to maintain draft quality. We propose NanoSpec, a novel training-free approach that breaks this trade-off by dynamically constructing a minimalist, context-aware active vocabulary for each generation step. Leveraging the inherent temporal locality of language generation, NanoSpec achieves high coverage while slashing the average vocabulary size by over $40\times$ (to $<$3k tokens) without requiring any auxiliary trained parameters. To realize the theoretical benefits of such high sparsity on modern hardware, we introduce a system-algorithm co-design that overcomes the inefficiencies of sparse memory access through asynchronous gathering and GPU-resident state management. As a complementary plug-and-play module, NanoSpec cuts draft time by an average of 51.6\%, delivering a $1.17$-$1.29\times$ end-to-end speedup over the state-of-the-art speculative decoding methods EAGLE-2 and EAGLE-3 across 7 tasks and outperforming complex training-based pruning baselines.
>
---
#### [replaced 145] From Tokens to Concepts: Leveraging SAE for SPLADE
- **分类: cs.IR; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决SPLADE模型因依赖词汇导致的性能限制。通过引入SAE构建语义概念空间，提升模型效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.21511](https://arxiv.org/pdf/2604.21511)**

> **作者:** Yuxuan Zong; Mathias Vast; Basile Van Cooten; Laure Soulier; Benjamin Piwowarski
>
> **备注:** 11 pages, 3 figures, 9 tables. To appear at SIGIR 2026
>
> **摘要:** Learned Sparse IR models, such as SPLADE, offer an excellent efficiency-effectiveness tradeoff. However, they rely on the underlying backbone vocabulary, which might hinder performance (polysemicity and synonymy) and pose a challenge for multi-lingual and multi-modal usages. To solve this limitation, we propose to replace the backbone vocabulary with a latent space of semantic concepts learned using Sparse Auto-Encoders (SAE). Throughout this paper, we study the compatibility of these 2 concepts, explore training approaches, and analyze the differences between our SAE-SPLADE model and traditional SPLADE models. Our experiments demonstrate that SAE-SPLADE achieves retrieval performance comparable to SPLADE on both in-domain and out-of-domain tasks while offering improved efficiency.
>
---
#### [replaced 146] RedDebate: Safer Responses Through Multi-Agent Red Teaming Debates
- **分类: cs.CL**

- **简介: 该论文提出RedDebate，属于AI安全任务，旨在解决LLM unsafe行为问题。通过多智能体辩论和记忆模块，自动检测并减少模型的不安全输出。**

- **链接: [https://arxiv.org/pdf/2506.11083](https://arxiv.org/pdf/2506.11083)**

> **作者:** Ali Asad; Stephen Obadinma; Radin Shayanfar; Xiaodan Zhu
>
> **摘要:** We introduce RedDebate, a novel multi-agent debate framework that provides the foundation for Large Language Models (LLMs) to identify and mitigate their unsafe behaviours. AI safety approaches often rely on costly human evaluation or isolated single-model assessment, both constrained by scalability and prone to oversight failures. RedDebate employs collaborative argumentation among multiple LLMs across diverse debate scenarios, enabling them to critically evaluate one another's reasoning and systematically uncover unsafe failure modes through fully automated red-teaming. To support this, we propose designing distinct long-term memory modules that preserve safety-relevant insights from debate interactions and leverage them during subsequent inference, facilitating continuous refinement of model behaviour. Empirical evaluation on safety benchmarks across a diverse set of models demonstrates that RedDebate substantially reduces unsafe outputs. While debate alone allows LLMs to refine their behaviour, the addition of memory yields further error reductions. To the best of our knowledge, RedDebate is the first fully automated framework to unify multi-agent debate and red-teaming to progressively enhance LLM safety without human intervention.
>
---
#### [replaced 147] Bridging the Gap: Transfer Learning from English PLMs to Malaysian English
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的命名实体识别任务，旨在解决马来西亚英语低资源环境下NER效果不佳的问题。通过构建专用预训练模型并进行微调，提升相关任务性能。**

- **链接: [https://arxiv.org/pdf/2407.01374](https://arxiv.org/pdf/2407.01374)**

> **作者:** Mohan Raj Chanthran; Lay-Ki Soon; Huey Fang Ong; Bhawani Selvaretnam
>
> **备注:** Accepted in 9th Workshop on Representation Learning for NLP (Rep4NLP) at ACL 2024
>
> **摘要:** Malaysian English is a low resource creole language, where it carries the elements of Malay, Chinese, and Tamil languages, in addition to Standard English. Named Entity Recognition (NER) models underperform when capturing entities from Malaysian English text due to its distinctive morphosyntactic adaptations, semantic features and code-switching (mixing English and Malay). Considering these gaps, we introduce MENmBERT and MENBERT, a pre-trained language model with contextual understanding, specifically tailored for Malaysian English. We have fine-tuned MENmBERT and MENBERT using manually annotated entities and relations from the Malaysian English News Article (MEN) Dataset. This fine-tuning process allows the PLM to learn representations that capture the nuances of Malaysian English relevant for NER and RE tasks. MENmBERT achieved a 1.52\% and 26.27\% improvement on NER and RE tasks respectively compared to the bert-base-multilingual-cased model. Although the overall performance of NER does not have a significant improvement, our further analysis shows that there is a significant improvement when evaluated by the 12 entity labels. These findings suggest that pre-training language models on language-specific and geographically-focused corpora can be a promising approach for improving NER performance in low-resource settings. The dataset and code published in this paper provide valuable resources for NLP research work focusing on Malaysian English.
>
---
#### [replaced 148] Phoneme-Level Visual Speech Recognition via Point-Visual Fusion and Language Model Reconstruction
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于视觉语音识别任务，解决因视觉模糊和缺乏听觉线索导致的识别难题。提出一种两阶段框架，融合视觉与面部特征，利用语言模型重建单词，提升识别准确率。**

- **链接: [https://arxiv.org/pdf/2507.18863](https://arxiv.org/pdf/2507.18863)**

> **作者:** Matthew Kit Khinn Teng; Haibo Zhang; Takeshi Saitoh
>
> **备注:** Accepted at ICASSP 2026. This version corresponds to the camera-ready manuscript
>
> **摘要:** Visual Automatic Speech Recognition (V-ASR) is a challenging task that involves interpreting spoken language solely from visual information, such as lip movements and facial expressions. This task is notably challenging due to the absence of auditory cues and the visual ambiguity of phonemes that exhibit similar visemes-distinct sounds that appear identical in lip motions. Existing methods often aim to predict words or characters directly from visual cues, but they commonly suffer from high error rates due to viseme ambiguity and require large amounts of pre-training data. We propose a novel phoneme-based two-stage framework that fuses visual and landmark motion features, followed by an LLM model for word reconstruction to address these challenges. Stage 1 consists of V-ASR, which outputs the predicted phonemes, thereby reducing training complexity. Meanwhile, the facial landmark features address speaker-specific facial characteristics. Stage 2 comprises an encoder-decoder LLM model, NLLB, that reconstructs the output phonemes back to words. Besides using a large visual dataset for deep learning fine-tuning, our PV-ASR method demonstrates superior performance by achieving 17.4% WER on the LRS2 and 21.0% WER on the LRS3 dataset.
>
---
#### [replaced 149] Beyond Static Dialogues: Benchmarking Realistic, Heterogeneous, and Evolving Long-Term Memory
- **分类: cs.CL; cs.IR**

- **简介: 该论文属于对话系统任务，旨在解决现有LLM评估中缺乏长期语义一致性和多样性的问题。提出RHELM基准，结合动态用户画像和LOOP模块，构建真实、多源、演化的对话场景。**

- **链接: [https://arxiv.org/pdf/2605.31086](https://arxiv.org/pdf/2605.31086)**

> **作者:** Han Zhang; Zihao Tang; Xin Yu; Xiao Liu; Yeyun Gong; Haizhen Huang; Yan Lu; Weiwei Deng; Feng Sun; Qi Zhang; Hanfang Yang
>
> **摘要:** In existing memory benchmarks for Large Language Models (LLMs), the evaluated dialogue sessions often lack long-term semantic consistency, and the underlying personas tend to be flat and static. Furthermore, in real-world scenarios, interactions between users and assistants involve more diverse, heterogeneous data streams, such as documents and emails. These shortcomings significantly limit the realism and effectiveness of current evaluations. To address these limitations, we introduce RHELM (Realistic, Heterogeneous, and Evolving Long-term Memory). Driven by meticulously crafted user profiles and a novel LOOP (pLan-rOllout-evOlve-Prune) module, we construct realistic dialogues across diverse interaction scenarios that exhibit dynamic temporal evolution and long-term coherence. Crucially, these dialogues are deeply integrated with heterogeneous external sources synchronized with the user's temporal event trajectory. The resulting benchmark encompasses challenging question-answer pairs spanning seven inquiry types, with each question mapping to at least one of 27 critical memory characteristics that we identify as essential yet underexplored in current research. Comprehensive experiments across full-context models, retrieval-augmented generation (RAG) methods, and representative memory frameworks reveal that contemporary approaches still expose critical weaknesses in complex, real-world settings, particularly in resolving multi-source aggregation and real-world contextual reasoning.
>
---
#### [replaced 150] When and How Much to Imagine: Adaptive Test-Time Scaling with World Models for Visual Spatial Reasoning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉空间推理任务，解决如何在测试时有效控制想象资源的问题。通过引入AVIC框架和AVIC-R策略，实现对视觉想象的智能调度，提升推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2602.08236](https://arxiv.org/pdf/2602.08236)**

> **作者:** Shoubin Yu; Yue Zhang; Zun Wang; Jaehong Yoon; Huaxiu Yao; Mingyu Ding; Mohit Bansal
>
> **备注:** the first two authors are equally contributed. Project page: this https URL
>
> **摘要:** Despite rapid progress in MLLMs, visual spatial reasoning remains unreliable when correct answers depend on how a scene would appear under unseen or alternative viewpoints. Recent work addresses this by augmenting reasoning with world models for visual imagination, but questions such as when imagination is actually necessary, how much of it is beneficial, and when it becomes harmful, remain poorly understood. In practice, indiscriminate imagination can increase computation and even degrade performance by introducing misleading evidence. In this work, we present an in-depth analysis of test-time visual imagination as a controllable resource for spatial reasoning. We first study when static visual evidence is sufficient, when imagination improves reasoning, and how excessive or unnecessary imagination affects accuracy and efficiency. To support this analysis, we then introduce AVIC, an adaptive test-time framework with world models that explicitly reasons about the sufficiency of current visual evidence before selectively invoking and scaling visual imagination. Finally, to further learn this gating and planning behavior without any annotation of when and how much to imagine, we introduce AVIC-R, which trains the policy via GRPO from QA-correctness rewards and penalties by imagination cost. Across spatial reasoning benchmarks (SAT, MMSI) and an embodied navigation benchmark (R2R), our results reveal clear scenarios where imagination is critical, marginal, or detrimental, and show that selective control can match or outperform fixed imagination strategies with substantially fewer world-model calls and language tokens. Our AVIC-R surpasses strong proprietary baselines including GPT-4o and GPT-4.1 while invoking the world model less often. Overall, our findings highlight the importance of analyzing and controlling test-time imagination for efficient and reliable spatial reasoning.
>
---
#### [replaced 151] Code2Math: Can Your Code Agent Effectively Evolve Math Problems Through Exploration?
- **分类: cs.CL**

- **简介: 该论文属于数学问题生成任务，旨在解决高质量数学题稀缺的问题。通过代码代理生成更复杂的新问题，验证其可解性和难度。**

- **链接: [https://arxiv.org/pdf/2603.03202](https://arxiv.org/pdf/2603.03202)**

> **作者:** Dadi Guo; Yuejin Xie; Qingyu Liu; Weixian Huang; Jiayu Liu; Zhiyuan Fan; Qihan Ren; Shuai Shao; Tianyi Zhou; Jianjie Feng; Wenze Su; Yujiu Yang; Dongrui Liu; Yi R. Fung
>
> **备注:** 38 pages
>
> **摘要:** As large language models (LLMs) advance their mathematical capabilities toward the IMO and research level, the scarcity of challenging, high-quality problems has become a significant bottleneck for training, evaluation and self-evolution of LLMs. Simultaneously, recent code agents have demonstrated sophisticated skills in agentic coding and reasoning, suggesting that code execution can serve as a scalable environment for mathematical experimentation. In this paper, we investigate the potential of code agents to autonomously evolve existing math problems into more complex variations. We introduce a multi-agent framework designed to perform problem evolution while validating the solvability and increased difficulty of the generated problems. Our experiments demonstrate that, given sufficient test-time exploration, code agents can synthesize new, solvable problems that are structurally distinct from and more challenging than the originals. This work provides empirical evidence that code-driven agents can serve as a viable mechanism for synthesizing high-difficulty mathematical reasoning problems within scalable computational environments. Code and data is available at this https URL.
>
---
#### [replaced 152] Truth, Trust, and Trouble: Medical AI on the Edge
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于医疗AI问答任务，旨在评估模型的准确性、安全性和有用性。研究对比了多个模型，分析其在事实可靠性与安全性间的权衡。**

- **链接: [https://arxiv.org/pdf/2507.02983](https://arxiv.org/pdf/2507.02983)**

> **作者:** Mohammad Anas Azeez; Rafiq Ali; Ebad Shabbir; Zohaib Hasan Siddiqui; Gautam Siddharth Kashyap; Jiechao Gao; Usman Naseem
>
> **备注:** Accepted at EMNLP 2025 (Industry Track)
>
> **摘要:** Large Language Models (LLMs) hold significant promise for transforming digital health by enabling automated medical question answering. However, ensuring these models meet critical industry standards for factual accuracy, usefulness, and safety remains a challenge, especially for open-source solutions. We present a rigorous benchmarking framework using a dataset of over 1,000 health questions. We assess model performance across honesty, helpfulness, and harmlessness. Our results highlight trade-offs between factual reliability and safety among evaluated models -- Mistral-7B, BioMistral-7B-DARE, and AlpaCare-13B. AlpaCare-13B achieves the highest accuracy (91.7%) and harmlessness (0.92), while domain-specific tuning in BioMistral-7B-DARE boosts safety (0.90) despite its smaller scale. Few-shot prompting improves accuracy from 78% to 85%, and all models show reduced helpfulness on complex queries, highlighting ongoing challenges in clinical QA.
>
---
#### [replaced 153] v-HUB: A Benchmark for Video Humor Understanding from Vision and Sound
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文提出v-HUB基准，用于视频幽默理解任务，解决多模态模型在无语言场景下理解幽默的难题。通过视频与音频分析，评估模型能力并验证多模态融合的有效性。**

- **链接: [https://arxiv.org/pdf/2509.25773](https://arxiv.org/pdf/2509.25773)**

> **作者:** Zhengpeng Shi; Yanpeng Zhao; Jianqun Zhou; Yuxuan Wang; Qinrong Cui; Wei Bi; Songchun Zhu; Bo Zhao; Zilong Zheng
>
> **备注:** 24 pages, 9 figures
>
> **摘要:** AI models capable of comprehending humor hold real-world promise -- for example, enhancing engagement in human-machine interactions. To gauge and diagnose the capacity of multimodal large language models (MLLMs) for humor understanding, we introduce v-HUB, a novel video humor understanding benchmark. v-HUB comprises a curated collection of non-verbal short videos, reflecting real-world scenarios where humor can be appreciated purely through visual cues. We pair each video clip with rich annotations to support a variety of evaluation tasks and analyses, including a novel study of environmental sound that can enhance humor. To broaden its applicability, we construct an open-ended QA task, making v-HUB readily integrable into existing video understanding task suites. We evaluate a diverse set of MLLMs, from specialized Video-LLMs to versatile OmniLLMs that can natively process audio, covering both open-source and proprietary domains. The experimental results expose the difficulties MLLMs face in comprehending humor from visual cues alone. Our findings also demonstrate that incorporating audio helps with video humor understanding, highlighting the promise of integrating richer modalities for complex video understanding tasks.
>
---
#### [replaced 154] MineDraft: A Framework for Batch Parallel Speculative Decoding
- **分类: cs.CL; cs.AI; cs.DC; cs.LG**

- **简介: 该论文属于语言模型推理优化任务，解决标准推测解码执行效率低的问题，提出MineDraft框架通过批量并行设计提升吞吐量和降低延迟。**

- **链接: [https://arxiv.org/pdf/2603.18016](https://arxiv.org/pdf/2603.18016)**

> **作者:** Zhenwei Tang; Arun Verma; Zijian Zhou; Zhaoxuan Wu; Alok Prakash; Daniela Rus; Bryan Kian Hsiang Low
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** Speculative decoding (SD) accelerates large language model inference by using a smaller draft model to propose draft tokens that are subsequently verified by a larger target model. However, the performance of standard SD is often limited by the strictly sequential execution of these drafting and verification stages. To address this, this paper proposes MineDraft, a batch parallel speculative decoding (PSD) framework designed to effectively hide drafting latency by overlapping it with verification. Our theoretical analysis shows that PSD is substantially more efficient than standard SD. MineDraft realizes the PSD through a novel batch-parallel design that maintains two batches of requests, overlapping drafting for one batch with verification for the other. Our experimental results show significant improvements of MineDraft in both throughput (up to 75%) and end-to-end latency (up to 39%) over standard SD. Furthermore, we have implemented MineDraft as a plugin for vLLM, demonstrating its practicality for production-ready inference systems.
>
---
#### [replaced 155] ACON: Optimizing Context Compression for Long-horizon LLM Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于长周期LLM代理任务，解决上下文过长导致的内存消耗大和推理效率低的问题。提出ACON框架，优化上下文压缩，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2510.00615](https://arxiv.org/pdf/2510.00615)**

> **作者:** Minki Kang; Wei-Ning Chen; Dongge Han; Huseyin A. Inan; Lukas Wutschitz; Yanzhi Chen; Robert Sim; Saravan Rajmohan
>
> **备注:** ICML 2026
>
> **摘要:** Large language models (LLMs) are increasingly deployed as agents in dynamic real-world environments, where success depends on maintaining precise records of actions and observations. However, the resulting unbounded context growth in long-horizon agentic tasks makes two critical bottlenecks: prohibitive inference memory costs and reasoning degradation due to irrelevant information. Existing compression methods fail to fully address this, often relying on brittle heuristics or requiring parameter updates impractical for proprietary or large-scale LLMs. We introduce Agent Context Optimization (ACON), a unified framework that optimally compresses both observations and history into concise, informative representations. Distinct from prior works, ACON employs an optimization in natural language space: it iteratively refines compression guidelines based on failure analysis of the agent, ensuring critical state information is preserved without model fine-tuning. To further minimize computational overhead, we distill the optimized compressor into smaller models. Experiments on AppWorld, OfficeBench, and Multi-objective QA demonstrate that ACON reduces peak token usage by 26-54% while improving task success over existing compression baselines. Notably, it enables smaller LMs to function effectively as long-horizon agents, achieving up to 46% performance improvement by mitigating context distraction. Our code is available at this https URL.
>
---
#### [replaced 156] BranPO: Scalable Contrastive Branch Sampling for Long-Horizon Agentic Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文属于长周期代理强化学习任务，解决稀疏奖励下多步骤决策问题。提出BranPO方法，通过对比分支采样提升训练效率与效果。**

- **链接: [https://arxiv.org/pdf/2602.03719](https://arxiv.org/pdf/2602.03719)**

> **作者:** Yubao Zhao; Weiquan Huang; Sudong Wang; Ruochen Zhao; Chen Chen; Yao Shu; Chengwei Qin
>
> **备注:** 26 pages, 5 figures
>
> **摘要:** Agentic reinforcement learning enables large language models to perform multi-turn planning and tool use, but long-horizon training remains challenging under sparse trajectory-level rewards, where a single outcome is uniformly assigned to all decisions. Prior methods introduce finer-grained supervision via tree-based exploration or process-level evaluation, but often incur high cost or produce noisy credit signals. In agentic trajectories, early mistakes may still be corrected by later actions, while seemingly promising intermediate states can fail due to poor subsequent decisions. We call this property non-monotonic correctness, which makes outcome rewards or state values insufficient for guiding what actions should be taken from each state. To address this, we propose Branching Relative Policy Optimization (\textbf{BranPO}), a value-free method that constructs localized contrastive supervision without dense rewards. BranPO truncates trajectories at intermediate prefixes and resamples continuations to form contrastive branches that share the same prefix but diverge in final outcomes, thereby isolating decisions that drive success or failure. We further introduce difficulty-aware branch sampling and Redundant Step Masking to improve sampling efficiency and suppress redundant updates. Experiments show that BranPO consistently outperforms diverse baseline categories across multiple multi-hop QA benchmarks without additional training cost, and generalizes to broader long-horizon agentic tasks with consistent improvements. Our code is available at this https URL.
>
---
#### [replaced 157] APB-V: Accelerating Long-Video Understanding via Sequence-Parallelism-aware Approximate Attention
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频理解任务，旨在解决长视频推理效率低的问题。通过序列并行的近似注意力机制，提升多GPU下的处理速度与性能。**

- **链接: [https://arxiv.org/pdf/2601.21444](https://arxiv.org/pdf/2601.21444)**

> **作者:** Yuxiang Huang; Mingye Li; Xu Han; Chaojun Xiao; Weilin Zhao; Ao Sun; Ziqi Yuan; Hao Zhou; Fandong Meng; Zhiyuan Liu
>
> **备注:** ACL 2026 main
>
> **摘要:** The efficiency of long-video inference remains a critical bottleneck, mainly due to the dense computation in the prefill stage of Large Multimodal Models (LMMs). Existing methods either compress visual embeddings or apply sparse attention on a single GPU, yielding limited acceleration or degraded performance and restricting LMMs from handling longer, more complex videos. To overcome these issues, we propose APB-V, a sequence-parallel framework with optimized attention that accelerates long-video inference across multiple GPUs. By distributing approximate attention, APB-V reduces computation and increases parallelism, enabling efficient processing of more visual embeddings without compression and thereby improving task performance. System-level optimizations, such as load balancing and fused forward passes, further unleash the potential of APB-V, delivering speedups of 12.72x, 1.70x, and 1.18x over FlashAttn, ZigZagRing, and APB, without notable performance loss. Code available at this https URL
>
---
#### [replaced 158] Modeling Distinct Human Interaction in Web Agents
- **分类: cs.CL; cs.HC**

- **简介: 该论文属于人机协作任务，旨在解决人类何时及为何干预的问题。通过分析用户与代理的交互模式，训练语言模型提高干预预测准确性，并提升代理的协作能力。**

- **链接: [https://arxiv.org/pdf/2602.17588](https://arxiv.org/pdf/2602.17588)**

> **作者:** Faria Huq; Zora Zhiruo Wang; Zhanqiu Guo; Venu Arvind Arangarajan; Tianyue Ou; Frank Xu; Shuyan Zhou; Graham Neubig; Jeffrey P. Bigham
>
> **备注:** Preprint
>
> **摘要:** Despite rapid progress in autonomous web agents, human involvement remains essential for shaping preferences and correcting agent behavior as tasks unfold. However, current agentic systems lack a principled understanding of when and why humans intervene, often proceeding autonomously past critical decision points or requesting unnecessary confirmation. In this work, we introduce the task of modeling human intervention to support collaborative web task execution. We collect CowCorpus, a dataset of 400 real-user web navigation trajectories containing over 4,200 interleaved human and agent actions. We identify four distinct patterns of user interaction with agents -- hands-off supervision, hands-on oversight, collaborative task-solving, and full user takeover. Leveraging these insights, we train language models (LMs) to anticipate when users are likely to intervene based on their interaction styles, yielding a 61.4-63.4% improvement in intervention prediction accuracy over base LMs. Finally, we deploy these intervention-aware models in live web navigation agents and evaluate them in a user study, finding a 36.8% increase in user-rated agent usefulness. Together, our results show structured modeling of human intervention leads to more adaptive, collaborative agents.
>
---
#### [replaced 159] Finding What Matters: Anchoring Context Knowledge with Evolving Indices for Iterative Retrieval
- **分类: cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决多跳问答中知识整合困难的问题。提出KAIR框架，通过动态索引锚定关键信息，提升模型推理能力。**

- **链接: [https://arxiv.org/pdf/2601.16462](https://arxiv.org/pdf/2601.16462)**

> **作者:** Mingyan Wu; Zhenghao Liu; Xinze Li; Yuqing Lan; Yukun Yan; Shuo Wang; Cheng Yang; Minghe Yu; Zheni Zeng; Maosong Sun
>
> **摘要:** Retrieval-Augmented Generation (RAG) has become a dominant paradigm for mitigating hallucinations in Large Language Models (LLMs) by incorporating external knowledge. However, existing RAG systems often struggle to effectively integrate and reason over key evidence scattered across noisy retrieved documents, particularly in multi-hop scenarios. In this paper, we propose KAIR, a Knowledge Anchoring framework for Iterative Retrieval that anchors knowledge within retrieved knowledge to guide LLMs to locate the key information. During iterative retrieval, KAIR progressively updates the knowledge index to anchor salient evidence from retrieved documents. The evolving index serves as a navigational anchoring index that enables the LLM to assess knowledge sufficiency and formulate subsequent retrieval queries. Finally, KAIR generates answers by jointly leveraging the retrieved documents and the finalized anchoring index. Experiments on four multi-hop question answering benchmarks demonstrate that KAIR consistently outperforms strong RAG baselines. Further analysis shows that KAIR effectively anchors key knowledge and alleviates the context noise during iterative retrieval, improving the LLM's ability to associate and reason over dispersed evidence across retrieved documents. All code and data are available at this https URL.
>
---
#### [replaced 160] A Monosemantic Attribution Framework for Stable Interpretability in Clinical Neuroscience Transformer-Based Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的可解释性任务，旨在解决临床语言模型解释不稳定的问题。通过构建单义特征空间，提升模型解释的稳定性和可信度。**

- **链接: [https://arxiv.org/pdf/2601.17952](https://arxiv.org/pdf/2601.17952)**

> **作者:** Michail Mamalakis; Tiago Azevedo; Cristian Cosentino; Chiara D'Ercoli; Subati Abulikemu; Zhongtian Sun; Richard Bethlehem; Pietro Lio
>
> **摘要:** Interpretability remains a key challenge for deploying language models (LM) in clinical settings such as progression diagnosis of Alzheimer disease, where early and trustworthy predictions are essential. Existing attribution methods exhibit high inter-method variability and unstable explanations due to the polysemantic nature of Transformer-Based LM and LLM representations, while mechanistic interpretability approaches lack direct alignment with model inputs and outputs and do not provide explicit importance scores. We introduce a unified interpretability framework that integrates attributional and mechanistic perspectives through monosemantic feature extraction. By constructing a monosemantic embedding space at the level of an transformer-based LM layer and optimizing the framework to explicitly reduce inter-method variability, our approach produces stable input-level importance scores and highlights salient features via a decompressed representation of the layer of interest, advancing the safe and trustworthy application of LMs in cognitive health and neurodegenerative disease.
>
---
#### [replaced 161] Correcting Gradient-Based Circuit Localization via Interaction-Aware Backpropagation
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于机制可解释性任务，旨在解决神经网络组件重要性估计不准确的问题。通过引入GIM技术，考虑特征交互影响，提升梯度方法的准确性。**

- **链接: [https://arxiv.org/pdf/2505.17630](https://arxiv.org/pdf/2505.17630)**

> **作者:** Joakim Edin; Casper L. Christensen; Róbert Csordás; Tuukka Ruotsalo; Zhengxuan Wu; Maria Maistro; Jing Huang; Lars Maaløe
>
> **摘要:** Circuit localization methods aim to identify the subset of model components responsible for specific behaviors in large language models, enabling detailed mechanistic analysis. Most existing methods assume components act independently and estimate importance by perturbing each component in isolation. However, components in neural networks interact, and ignoring these interactions leads to systematic misestimation of component importance. We find that one particularly problematic interaction is attention self-repair, in which softmax redistribution causes gradients for influential attention scores to vanish as other positions with similar values compensate. We introduce Gradient Interaction Modifications (GIM), a technique that explicitly accounts for feature interactions during backpropagation. GIM achieves state-of-the-art performance on the circuit localization track of the Mechanistic Interpretability Benchmark and outperforms existing gradient-based methods on feature attribution across diverse tasks. By accounting for interaction effects and explaining why prior methods underestimate component importance, GIM enables more faithful mechanistic analysis of large language models. GIM is available as a Python package at this https URL.
>
---
#### [replaced 162] Optimizing Diversity and Quality through Base-Aligned Model Collaboration
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言生成任务，解决大模型输出缺乏多样性的问题。提出BACo框架，在推理阶段动态结合基础模型与对齐模型，提升生成内容的多样性和质量。**

- **链接: [https://arxiv.org/pdf/2511.05650](https://arxiv.org/pdf/2511.05650)**

> **作者:** Yichen Wang; Chenghao Yang; Tenghao Huang; Muhao Chen; Jonathan May; Mina Lee
>
> **备注:** ICML 2026. (47 pages, 22 figures)
>
> **摘要:** Alignment has greatly improved large language models (LLMs)' output quality at the cost of diversity, yielding highly similar outputs across generations, especially in open-ended generation tasks. We propose Base-Aligned Model Collaboration (BACo), an inference-time token-level model collaboration framework that dynamically combines a base LLM with its aligned counterpart to optimize diversity and quality. Using uncertainty and content-based signals, BACo employs routing strategies to determine, at each token, which model to decode from. Prior diversity-promoting methods often improve diversity at the expense of quality or require expensive decoding or post-training. In contrast, BACo achieves both high diversity and quality post hoc within a single pass, while offering strong controllability. We introduce a family of effective routing strategies and evaluate them across three open-ended generation tasks with 13 diversity and quality metrics. BACo consistently surpasses state-of-the-art inference-time baselines. With our best router, BACo achieves a 21.3% joint improvement in diversity and quality, which is further supported by human evaluations. Overall, our results demonstrate that collaboration between base and aligned models provides an effective and controllable mechanism for optimizing the diversity-quality trade-off.
>
---
#### [replaced 163] TajikNLP: An Open-Source Toolkit for Comprehensive Text Processing of Tajik (Cyrillic Script)
- **分类: cs.CL**

- **简介: 该论文提出TajikNLP，解决塔吉克语（西里尔字母）资源匮乏问题，提供全面的文本处理工具包，包含分词、标注、情感分析等功能。**

- **链接: [https://arxiv.org/pdf/2605.04583](https://arxiv.org/pdf/2605.04583)**

> **作者:** Mullosharif K. Arabov; Karomatullo Habibullozoda; Nurali Shirinov
>
> **备注:** Accepted to CLIB 2026
>
> **摘要:** The Tajik language, written in Cyrillic script, remains severely under-resourced in terms of publicly available natural language processing (NLP) toolkits, hindering both linguistic research and applied development. This paper introduces TajikNLP, an open-source Python library that provides the first comprehensive pipeline for processing authentic Tajik text while preserving the original Cyrillic orthography. The library implements a modular architecture centered around a unified Doc object, enabling sequential application of components for cleaning, normalization, tokenization (including subword BPE), morphemic segmentation, part-of-speech tagging, stemming, lemmatization, and sentence splitting. A novel unified morphology engine is introduced, offering controlled and deep analysis modes that significantly improve handling of Tajik's agglutinative nominal and verbal inflections. The release further incorporates a lexicon-based sentiment analyser and pre-trained Word2Vec/FastText embeddings loaded directly from the Hugging Face Hub. To ensure reproducibility and facilitate future research, four accompanying linguistic datasets -- a POS-tagged corpus (52.5k entries), a sentiment lexicon (3.5k entries), a toponym gazetteer (5.6k entries), and a personal names dataset (3.8k entries) -- have been openly published under permissive licenses. The library's reliability is validated by an extensive test suite of 616 automated tests achieving 93% source code coverage. TajikNLP thus establishes a foundational technological infrastructure for Tajik language processing, lowering the barrier to entry for both academic and industrial applications in low-resource Cyrillic-script environments.
>
---
#### [replaced 164] How Language Models Process Negation
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究LLMs如何处理否定。解决模型在否定问题上准确率低的问题，通过分析发现模型存在两种处理机制，其中构造性机制更显著。**

- **链接: [https://arxiv.org/pdf/2605.03052](https://arxiv.org/pdf/2605.03052)**

> **作者:** Zhejian Zhou; Tianyi Zhou; Robin Jia; Jonathan May
>
> **备注:** ICML 2026
>
> **摘要:** We study how Large Language Models (LLMs) process negation mechanistically. First, we establish that even though open-weight models often provide wrong answers to questions involving negation, they do possess internal components that process negation correctly. Their poor accuracy is due to late-layer attention behavior that promotes simple shortcuts; ablating those attention modules greatly improves accuracy on negation-related questions. Second, we uncover how models process negation. We consider two hypotheses: models could use attention heads that attend to the phrase being negated and suppress related concepts, or they could directly construct a representation of the entire negative phrase (e.g., representing "not gas" as a vector that promotes liquids and solids). We apply a range of observational and causal interpretability techniques on Mistral-7B and Llama-3.1-8B to show that models implement both mechanisms, with the "constructive" mechanism being more prominent. Combined, our work deepens the understanding of LLMs' internals, highlighting construction-dominant computations and the coexistence of competing mechanisms within LLMs.
>
---
#### [replaced 165] Generic Interpretation Approach for Transformer Models Incorporating Heterogenous Attention Structures
- **分类: cs.CV; cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于模型解释任务，旨在解决具有异构注意力结构的Transformer模型的可解释性问题。提出了一种解释方法，并通过实验分析其工作机制。**

- **链接: [https://arxiv.org/pdf/2605.27458](https://arxiv.org/pdf/2605.27458)**

> **作者:** Yongjin Cui; Xiaohui Fan; Huajun Chen
>
> **摘要:** Transformer has significantly propelled the development of artificial intelligence, and certainly the development of agents as well. We categorize attention structures of Transformer into two types based on the source of the input information: homogenous and heterogenous attention structures. Heterogenous attention structures, with co-attention as a typical example, process information from different sources. Heterogenous attention structure is the foundation for Transformer models to achieve more complex functions and integrate more modal information. Whether for research purposes or policy requirements, the interpretation of Transformer models with heterogenous attention structures is an important task. The fusion of information from different sources brings new challenges. Our work mainly includes two parts: method and experimentation. In terms of method, we propose an interpretation method for Transformer models with heterogenous attention structures. In terms of experimentation, based on our experimental analysis paradigm, we interpret the operating mechanisms of representative models, conduct semantic interpretation and logical interpretation.
>
---
#### [replaced 166] Are Large Reasoning Models Interruptible?
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究LRM在动态环境中的鲁棒性，解决模型在中断和上下文变化下的可靠性问题。通过实验发现现有模型在动态场景下性能显著下降，并揭示了多种失败模式。**

- **链接: [https://arxiv.org/pdf/2510.11713](https://arxiv.org/pdf/2510.11713)**

> **作者:** Tsung-Han Wu; Mihran Miroyan; David M. Chan; Trevor Darrell; Narges Norouzi; Joseph E. Gonzalez
>
> **备注:** ICML 2026; Project Page: this http URL
>
> **摘要:** Real-world applications of Large Reasoning Models (LRMs) often require reasoning about changing prompts or environments. In this work, we challenge the frozen world assumption and evaluate LRM robustness under two realistic dynamic scenarios: interruptions, which test the accuracy of model responses under budget-constrained outputs, and dynamic context, which tests model adaptation to in-flight changes. Across mathematics and programming benchmarks that require long-form reasoning, static evaluations consistently overestimate robustness: even state-of-the-art LRMs, which achieve high accuracy in static settings, can fail unpredictably when interrupted or exposed to changing context, with performance dropping by up to 60% when updates are introduced late in the reasoning process. Our analysis further reveals several novel failure modes, including reasoning leakage, where models fold the reasoning into their final answer when interrupted; panic, where under time pressure models abandon reasoning entirely and return incorrect answers; and self-doubt, where performance degrades when trying to incorporate updated information. Project Page: this http URL
>
---
#### [replaced 167] Beyond Two-Stage Training: Cooperative SFT and RL for LLM Reasoning
- **分类: cs.CL**

- **简介: 该论文属于大语言模型推理能力提升任务，解决SFT与RLVR融合效果不佳的问题，提出BRIDGE框架，通过协作优化提升模型性能。**

- **链接: [https://arxiv.org/pdf/2509.06948](https://arxiv.org/pdf/2509.06948)**

> **作者:** Liang Chen; Xueting Han; Li Shen; Jing Bai; Kam-Fai Wong
>
> **备注:** ICML 2026
>
> **摘要:** Supervised fine-tuning (SFT) and reinforcement learning with verifiable rewards (RLVR) are two widely used post-training paradigms for improving the reasoning ability of large language models (LLMs). Recent methods attempt to integrate SFT and RLVR in a single stage by reweighting or scheduling their objectives. However, such coupling can be counterproductive because supervised updates are not uniformly beneficial for reward optimization. To address this, we propose BRIDGE, a scalable framework in which SFT learns to supervise RL by selectively transferring knowledge that improves reward optimization. Specifically, BRIDGE alternates two updates at each meta-training step: a base-model update that fuses the SFT and RL gradients, and an update to a lightweight low-rank adapter (LoRA) that coordinates the two objectives by maximizing a cooperative-gain signal, defined as the reward of joint SFT-RL training over an RL-only baseline. Across five mathematical reasoning benchmarks, BRIDGE consistently outperforms two-stage cold start, naive mixing, and representative single-stage integration baselines, yielding over three points average absolute improvement and more stable training dynamics. We further show that BRIDGE extends to logical reasoning and generalizes out-of-distribution to code and science without additional training, while staying robust under noisy rewards.
>
---
#### [replaced 168] A tree interpretation of arc standard dependency derivation
- **分类: cs.CL**

- **简介: 该论文研究依存句法分析任务，解决如何将弧标准推导转化为有序树的问题。通过定义确定性树更新，证明项目树与连续有序树的等价性，为非项目输入提供伪项目提升方法。**

- **链接: [https://arxiv.org/pdf/2603.27459](https://arxiv.org/pdf/2603.27459)**

> **作者:** Zihao Huang; Ai Ka Lee; Jungyeul Park
>
> **摘要:** Arc-standard derivations over projective dependency trees can be interpreted as the incremental construction of lexicalized ordered trees with contiguous yields. Each \textsc{shift}, \textsc{leftarc}, and \textsc{rightarc} transition corresponds to a deterministic tree update, and the resulting ordered tree uniquely determines the dependency arcs introduced by the derivation. We show that this representation is not an arbitrary encoding: a single-headed dependency tree admits such a contiguous ordered representation if and only if it is projective. The proposal is therefore derivational rather than conversion-based, since the ordered object is defined over the transition sequence itself rather than obtained by transforming a completed dependency graph. This gives a tree-theoretic interpretation of arc-standard parsing, in which projective dependency derivations implicitly construct recoverable constituency-style ordered trees. For non-projective inputs, the interpretation can be used through pseudo-projective lifting and inverse decoding. A small implementation study confirms that the mapped derivations are executable in an existing neural transition-based parser.
>
---
#### [replaced 169] Dynamic Meta-Metrics: Source-Sentence Conditioned Weighting for MT Evaluation
- **分类: cs.CL**

- **简介: 该论文提出动态元度量框架DMM，用于机器翻译评估，通过源句条件组合现有指标提升评价效果。解决多语言对下度量组合适应性问题，采用MLP和软条件化方法实现性能提升。**

- **链接: [https://arxiv.org/pdf/2605.09098](https://arxiv.org/pdf/2605.09098)**

> **作者:** Luke Zhang; Justin Vasselli; Aditya Khan; York Hay Ng; En-Shiun Annie Lee
>
> **备注:** 5 pages, ACL SRW 2026
>
> **摘要:** We propose Dynamic Meta-Metrics (DMM), a framework for machine translation evaluation that learns source-sentence conditioned combinations of existing metrics. Rather than relying on a single static ensemble or language-specific weighting, DMM adapts the metric combination based on properties of the source segment. We study hard conditioning, which fits an interpretable combiner per cluster, and an exploratory soft-conditioned extension whose weights vary continuously with source-cluster responsibilities. We evaluate DMM on the WMT Metrics Shared Task data across multiple language pairs using pairwise agreement measures at the system and segment levels. Across settings, MLP-based combinations outperform linear and Gaussian process-based ensembles, and introducing soft conditioning yields gains over linear models.
>
---
#### [replaced 170] Reading, Not Thinking: Understanding and Bridging the Modality Gap When Text Becomes Pixels in Multimodal LLMs
- **分类: cs.CL; cs.CV**

- **简介: 该论文研究多模态大语言模型中文本转图像后的性能下降问题，属于视觉文本理解任务。通过分析和实验，发现模型因图像输入减少推理过程导致性能差距，并提出方法提升图像模式表现。**

- **链接: [https://arxiv.org/pdf/2603.09095](https://arxiv.org/pdf/2603.09095)**

> **作者:** Kaiser Sun; Xiaochuang Yuan; Hongjun Liu; Chen Zhao; Cheng Zhang; Mark Dredze; Fan Bai
>
> **摘要:** Multimodal large language models (MLLMs) can process text presented as images, yet they often perform worse than when the same content is provided as textual tokens. We systematically diagnose this "modality gap" by evaluating seven MLLMs across seven benchmarks in five input modes, spanning both synthetically rendered text and realistic document images from arXiv PDFs to Wikipedia pages. We find that the gap is highly sensitive to rendering choices such as font and resolution, and that natural document images often exhibit much smaller gaps, suggesting the performance difference partly reflects evaluation artifacts rather than fundamental limitations. Through a grounded-theory error analysis of over 4,000 examples, we identify the primary cause: image input alone suppresses reasoning effort, with models producing 5--19x shorter outputs that skip step-by-step computation or reasoning. The reluctance to reason, not a failure of perception or knowledge retrieval, drives the performance gap, particularly on tasks requiring multi-step reasoning. We show that a simple, lightweight on-policy self-distillation method by fine-tuning models on their own text-mode reasoning traces paired with image inputs closes this gap, raising image-mode accuracy to match or exceed text-mode performance with over 50\% improvement, and the gains transfer to unseen benchmarks without catastrophic forgetting. Overall, our results and analyses provide a systematic understanding of the modality gap and suggest a practical path toward improving visual text understanding in multimodal language models.
>
---
#### [replaced 171] Distributional Open-Ended Evaluation of LLM Cultural Value Alignment Based on Value Codebook
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 该论文属于大语言模型文化价值观对齐任务，解决现有评估方法在文化多样性与开放生成上的不足。提出DOVE框架，通过分布对比评估模型输出与人类文本的匹配度。**

- **链接: [https://arxiv.org/pdf/2604.06210](https://arxiv.org/pdf/2604.06210)**

> **作者:** Jaehyeok Lee; Xiaoyuan Yi; Jing Yao; Hyunjin Hwang; Roy Ka-Wei Lee; Xing Xie; JinYeong Bak
>
> **备注:** ICML 2026 Camera Ready
>
> **摘要:** As LLMs are globally deployed, aligning their cultural value orientations is critical for safety and user engagement. However, existing benchmarks face the Construct-Composition-Context ($C^3$) challenge: relying on discriminative, multiple-choice formats that probe value knowledge rather than true orientations, overlook subcultural heterogeneity, and mismatch with real-world open-ended generation. We introduce DOVE, a distributional evaluation framework that directly compares human-written text distributions with LLM-generated outputs. DOVE utilizes a rate-distortion variational optimization objective to construct a compact value codebook from 10K documents, mapping text into a structured value space to filter semantic noise. Alignment is measured using unbalanced optimal transport, capturing intra-cultural distributional structures and subgroup diversity. Experiments across 12 LLMs show that DOVE achieves superior predictive validity, attaining a 31.56% correlation with downstream tasks, while maintaining high reliability with as few as 500 samples per culture.
>
---
#### [replaced 172] Self-Trained Verification for Training- and Test-Time Self-Improvement
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于推理模型的自我改进任务，旨在解决验证瓶颈问题。提出自训练验证（STV），提升测试和训练阶段的准确性。**

- **链接: [https://arxiv.org/pdf/2605.30290](https://arxiv.org/pdf/2605.30290)**

> **作者:** Chen Henry Wu; Aditi Raghunathan
>
> **摘要:** Self-improvement at scale has been a longstanding goal for reasoning models, and there are two natural places to do it: at test time, through verification-refinement (V-R) loops; and at training time, through self-training methods. Both are gated by the same bottleneck: the verifier. V-R loops stall when verifier scores inflate while accuracy stagnates, and when feedback is too generic to act on; self-training fails similarly when bad self-generated data are added to training. Better verification would unlock both, but the capability we want to train, i.e., catching self-generated errors, lacks training signal. To address this challenge, we propose self-trained verification (STV). Our key observation is that, while a model cannot catch these errors alone, it can when shown the reference solution. We turn this asymmetry into a supervision target and train the verifier to imitate a more informed version of itself. At test time, STV substantially improves V-R loops on hard problems, while alternatives (e.g., SFT, RL on verifier scores, and even meta-verifiers) do not. STV roughly doubles accuracy on hard math and lifts it 14x on scientific reasoning tasks (1.5% to 21%). At training time, we additionally train the generator using RL with STV verifier's feedback inside the V-R loop - a procedure we call verifier-in-the-loop training (ViL). Starting from an RL-converged generator, ViL yields a further 33% gain in pass@1. More notably, the generator's standalone pass@1, with no verifier at test time, climbs 30% relative past where standard RL had converged. Hence, the next frontier in reasoning on hard problems may lie in how we train for and with verification. Website: this https URL
>
---
#### [replaced 173] LISTEN to Your Preferences: An LLM Framework for Multi-Objective Selection
- **分类: cs.CL**

- **简介: 该论文提出LISTEN框架，解决多目标选择问题，通过自然语言与LLM交互，优化决策过程，提升选择效率与准确性。**

- **链接: [https://arxiv.org/pdf/2510.25799](https://arxiv.org/pdf/2510.25799)**

> **作者:** Adam S. Jovine; Tinghan Ye; Francis Bahk; Jingjing Wang; Matthew Ford; David B. Shmoys; Peter I. Frazier
>
> **备注:** Accepted at IJCAI-ECAI 2026 (the 35th International Joint Conference on Artificial Intelligence)
>
> **摘要:** Human experts often struggle to select the best option from a large set of items with multiple competing objectives, a process bottlenecked by the difficulty of formalizing complex, implicit preferences. To address this, we introduce LISTEN (LLM-based Iterative Selection with Trade-off Evaluation from Natural-language), an agentic LLM-based framework that treats the LLM as a decision-making agent capable of iteratively refining its internal preference model and taking actions (e.g., proposing utilities or selecting candidates) to maximize alignment with a user's implicit goals. To operate within LLM constraints like context windows and inference costs, we propose two iterative algorithms: LISTEN-U, which uses the LLM to refine a parametric utility function, and LISTEN-T, a non-parametric method that performs tournament-style selections over small batches of solutions. Evaluated on diverse tasks including flight booking, shopping, and exam scheduling, our results show LISTEN-U excels when preferences are parametrically aligned (a property we measure with a novel concordance metric), while LISTEN-T offers more robust performance overall. This work explores a promising direction for steering complex multi-objective decisions directly with natural language, reducing the cognitive burden of traditional preference elicitation. Code is available at this https URL data is available at this https URL.
>
---
#### [replaced 174] CECOR: Correction-oriented synthetic data construction for factual error correction
- **分类: cs.CL**

- **简介: 该论文属于事实错误修正任务，解决多跳推理中因数据不足和语义错误定位困难导致的修正难题。提出CECoR框架，通过分解与注入机制生成高质量训练数据，提升修正效果。**

- **链接: [https://arxiv.org/pdf/2605.02277](https://arxiv.org/pdf/2605.02277)**

> **作者:** Lei Zhu; Xiaobao Wang; Jianbiao Yang; Chenyang Wang; Dongxiao He; Longbiao Wang; Jianwu Dang
>
> **摘要:** Factual Error Correction (FEC) aims to revise inaccurate text into statements that are factually consistent with external evidence. Although recent methods perform well on single-hop correction, they often treat claims as atomic units and struggle with multi-hop cases that require compositional reasoning across multiple evidence sources. This challenge is further amplified by limited paired data and difficulties in locating semantic errors within complex reasoning chains. We present CECoR (Compositional Error Correction via Reasoning-aware Synthesis), a reasoning-aware framework that introduces a Decomposition and Injection paradigm for compositional error correction. CECoR decomposes multi-hop claims into interpretable reasoning steps and injects controlled perturbations to synthesize high-quality training pairs. A two-stage learning strategy combining supervised fine-tuning and reinforcement learning improves factual accuracy and robustness. Comprehensive evaluations show that CECoR achieves strong performance on multi-hop benchmarks, outperforming both distantly supervised methods and few-shot LLM baselines. It also generalizes effectively to single-hop correction and remains stable under noisy evidence, demonstrating its versatility for real-world factual correction.
>
---
#### [replaced 175] Beyond End-to-End Video Models: An LLM-Based Multi-Agent System for Educational Video Generation
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于教育视频生成任务，解决传统模型在逻辑严谨性和知识精确性上的不足。提出LASEV系统，通过多智能体协作实现高质量教学视频生成。**

- **链接: [https://arxiv.org/pdf/2602.11790](https://arxiv.org/pdf/2602.11790)**

> **作者:** Lingyong Yan; Jiulong Wu; Dong Xie; Weixian Shi; Deguo Xia; Jizhou Huang
>
> **备注:** Accepted at ACM SIGKDD 2026 (KDD '26), Applied Data Science Track. 10 pages, 2 figures, 5 tables. The project is available at \url{this https URL}
>
> **摘要:** Although recent end-to-end video generation models demonstrate impressive performance in visually oriented content creation, they remain limited in scenarios that require strict logical rigor and precise knowledge representation, such as instructional and educational media. To address this problem, we propose LASEV, a hierarchical LLM-based multi-agent system for generating high-quality instructional videos from educational problems. LASEV formulates educational video generation as a multi-objective task that simultaneously demands correct step-by-step reasoning, pedagogically coherent narration, semantically faithful visual demonstrations, and precise audio--visual alignment. To address the limitations of prior approaches--including low procedural fidelity, high production cost, and limited controllability--LASEV decomposes the generation workflow into specialized agents that collaborate through a central Orchestrating Agent, shared production state, explicit quality gates, and iterative critique mechanisms. Specifically, the Orchestrating Agent supervises a Solution Agent for rigorous problem solving, an Illustration Agent that produces executable visualization code, and a Narration Agent for learner-oriented instructional scripts. In addition, all outputs from the working agents are subject to semantic critique, rule-based constraints, and tool-based compilation checks. Rather than directly synthesizing pixels, the system constructs a structured executable video script that is deterministically compiled into synchronized visuals and narration using template-driven assembly rules, enabling fully automated production without manual editing. In large-scale deployments, LASEV achieves a throughput exceeding one million videos per day, delivering over a 95% reduction in cost compared to current industry-standard approaches while maintaining a high acceptance rate.
>
---
#### [replaced 176] CAREF: Calibration-Aware Regularization for Explanation Faithfulness Without Rationale Supervision
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出CAREF框架，用于提升大模型解释的可信度。属于可解释性任务，解决无标注解释的模型微调问题，通过联合优化准确性和解释一致性实现。**

- **链接: [https://arxiv.org/pdf/2605.27835](https://arxiv.org/pdf/2605.27835)**

> **作者:** Naphat Nithisopa; Teerapong Panboonyuen
>
> **备注:** 10 pages
>
> **摘要:** We introduce CAREF, a parameter-efficient fine-tuning framework that jointly optimizes predictive accuracy and explanation faithfulness via calibration-aware regularization. At its core, CAREF couples entropy-based calibration with token-level sparsity control through a single unified loss, the Calibration-Aware Regularization for Explanation Faithfulness (LSCED), without requiring rationale supervision. Evaluated on four NLE benchmarks (COS-E, ECQA, ComVE, e-SNLI) with Flan-T5, our lightweight CAREF-AQ variant attains the best average accuracy (89.04) and explanation alignment (81.00 nBERT) using only 6.43% of trainable parameters, outperforming LoRA and AdaLoRA. To our knowledge, CAREF is the first method to unify entropy and sparsity regularization in a single training objective for interpretable LLM fine-tuning.
>
---
#### [replaced 177] Interpreto: An Explainability Library for Transformers
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Interpreto，一个用于解释Transformer模型的开源库，解决模型可解释性问题。工作包括提供归因方法和基于概念的解释，实现端到端的解释流程。**

- **链接: [https://arxiv.org/pdf/2512.09730](https://arxiv.org/pdf/2512.09730)**

> **作者:** Antonin Poché; Thomas Mullor; Gabriele Sarti; Frédéric Boisnard; Corentin Friedrich; Charlotte Claye; François Hoofd; Raphael Bernas; Nicholas Asher; Céline Hudelot; Fanny Jourdan
>
> **备注:** Accepted to ACL 2026 System Demonstration. Equal contribution: Poché and Jourdan
>
> **摘要:** Interpreto is an open-source Python library for interpreting HuggingFace language models, from early BERT variants to LLMs. It provides two complementary families of methods: attribution methods and concept-based explanations. The library bridges recent research and practical tooling by exposing explanation workflows through a unified API for both classification and text generation. A key differentiator is its end-to-end concept-based pipeline (from activation extraction to concept learning, interpretation, and scoring), which goes beyond feature-level attributions and is uncommon in existing libraries. See GitHub: this https URL and the demo website: this https URL.
>
---
#### [replaced 178] Domain-Shift-Aware Conformal Prediction for Large Language Models
- **分类: stat.ML; cs.AI; cs.CL; cs.LG; stat.AP**

- **简介: 该论文属于不确定性量化任务，旨在解决大语言模型在领域变化下的过自信问题。提出DS-CP框架，通过重加权校准样本提升预测可靠性。**

- **链接: [https://arxiv.org/pdf/2510.05566](https://arxiv.org/pdf/2510.05566)**

> **作者:** Zhexiao Lin; Yuanyuan Li; Neeraj Sarna; Yuanyuan Gao; Michael von Gablenz
>
> **备注:** Accepted to Forty-Third International Conference on Machine Learning (ICML), 2026
>
> **摘要:** Large language models have achieved impressive performance across diverse tasks. However, their tendency to produce overconfident and factually incorrect outputs, known as hallucinations, poses risks in real-world applications. Conformal prediction provides finite-sample, distribution-free coverage guarantees, but standard conformal prediction breaks down under domain shift, often leading to under-coverage and unreliable prediction sets. We propose a new framework called Domain-Shift-Aware Conformal Prediction (DS-CP). Our framework adapts conformal prediction to large language models under domain shift, by systematically reweighting calibration samples based on their proximity to the test prompt, thereby preserving validity while enhancing adaptivity. Our theoretical analysis and experiments on the MMLU benchmark demonstrate that the proposed method delivers more reliable coverage than standard conformal prediction, especially under substantial distribution shifts, while maintaining efficiency. This provides a practical step toward trustworthy uncertainty quantification for large language models in real-world deployment.
>
---
#### [replaced 179] Global PIQA: Evaluating Commonsense Reasoning Across 100+ Languages and Cultures
- **分类: cs.CL**

- **简介: 该论文提出Global PIQA，一个覆盖100+语言和文化的常识推理基准，旨在评估大语言模型的跨文化理解能力，解决多语言环境下模型表现不均的问题。**

- **链接: [https://arxiv.org/pdf/2510.24081](https://arxiv.org/pdf/2510.24081)**

> **作者:** Tyler A. Chang; Catherine Arnett; Abdelrahman Sadallah; Abdelrahman Eldesokey; Abeer Kashar; Abolade Daud; Abosede Grace Olanihun; Adamu Labaran Mohammed; Adeyemi Praise; Adhikarimayum Meerajita Sharma; Aditi Gupta; Adril Putra Merin; Adwoa Bremang; Afitab Iyigun; Afonso Simplício; Ahmed Essouaied; Aicha Chorana; Akhil Eppa; Akintunde Oladipo; Akriti Kuri; Akshay Ramesh; Aleksei Dorkin; Alfred Malengo Kondoro; Alham Fikri Aji; Ali Eren Çetintaş; Allan Hanbury; Alou Dembele; Alp Niksarli; Álvaro Arroyo; Amin Bajand; Amol Khanna; Ana Chkhaidze; Ana Carolina Condez; Anamaria-Roberta Hartl; Andiswa Mkhonto; Andrew Hoblitzell; Andrew Tran; Angelos Poulis; Anirban Majumder; Anjali Chaudhary; Anna Vacalopoulou; Annette Kuuipolani Kanahele Wong; Annika Simonsen; Anton Kovalev; Anupam Nayak; Ashvanth S; Ayodeji Lana; Ayu Purwarianti; Bashar Alhafni; Benedict Busole; Bernard Ghanem; Bharti Nathani; Biljana Stojanovska Đurić; Blessing Ogundipe; Bolaotan Agbonile; Bragi Bergsson; Bruce Torres Fischer; Burak Tutar; Burcu Çınar; Cade Kane; Can Udomcharoenchaikit; Chadi Helwe; Chaithra Reddy Nerella; Chen Cecilia Liu; Chiamaka Nwokolo; Christopher Homan; Clément Sampebgo; Cristina España-Bonet; Cynthia Amol; Daeyoep Lee; Dan Saattrup Smart; Dana Arad; Daniil Dzenhaliou; Dasol Choi; David Liu; David Semedo; David Anugraha; Deborah Popoola; Deividas Mataciunas; Delphine Nyaboke; Dennis Owusu; Dhyuthy Krishna Kumar; Diogo Tavares; Diogo Glória-Silva; Divyanshu Goyal; DongGeon Lee; E. Kelly Buchanan; Ebele Nwamaka Anajemba; Egonu Ngozi Grace; Elena Mickel; Elias Herranen; Eliza Acharya; Eman Nisar; Emile Anand; Emmanuel Habumuremyi; Emuobonuvie Maria Ajiboye; Eryawan Presma Yulianrifat; Esther Adenuga; Ewa Rudnicka; Faith Itiola
>
> **备注:** Preprint
>
> **摘要:** To date, there exist almost no culturally-specific evaluation benchmarks for large language models (LLMs) that cover a large number of languages and cultures. In this paper, we present Global PIQA, a participatory commonsense reasoning benchmark for over 100 languages, constructed by hand by over 350 researchers from over 65 countries around the world. The 141 language varieties in Global PIQA cover five continents, 19 language families, and 24 writing systems. In the non-parallel split of Global PIQA, over 50% of examples reference local foods, customs, traditions, or other culturally-specific elements. In the parallel split, we translate more "culturally agnostic" commonsense reasoning questions into 131 language varieties, for direct cross-lingual comparisons. In both splits, all examples have been verified by native speakers of the languages. We find that state-of-the-art LLMs perform well on Global PIQA in aggregate, but they exhibit weaker performance in lower-resource languages (e.g. up to a 68% accuracy gap between languages in the parallel split). Global PIQA highlights that in many languages and cultures, everyday knowledge remains an area for improvement in LLMs, alongside more widely-discussed capabilities such as complex reasoning and expert knowledge. Beyond its uses for LLM evaluation, Global PIQA provides a glimpse into the wide diversity of cultures in which human language is embedded.
>
---
#### [replaced 180] Beyond Scalar Rewards: Dense Feedback for LLM Policy Synthesis in Sequential Social Dilemmas
- **分类: cs.CL; cs.GT**

- **简介: 该论文属于多智能体强化学习任务，解决LLM在序列社会困境中生成有效策略的问题。通过引入密集反馈（含社会指标）替代传统稀疏奖励，提升策略质量与多样性。**

- **链接: [https://arxiv.org/pdf/2603.19453](https://arxiv.org/pdf/2603.19453)**

> **作者:** Víctor Gallego
>
> **备注:** Accepted to NExT-Game 2026: New Frontiers in Game-Theoretic Learning, ICML 2026 Workshop
>
> **摘要:** We study LLM policy synthesis: using a language model to iteratively generate programmatic agent policies for multi-agent environments. Rather than training neural policies via reinforcement learning, our framework prompts an LLM to produce Python policy functions, evaluates them in self-play, and refines them using performance feedback across iterations. We investigate feedback engineering (the design of what evaluation information is shown to the LLM during refinement) comparing sparse feedback (scalar reward only) against dense feedback (reward plus social metrics: efficiency, equality, sustainability, peace). Across two canonical Sequential Social Dilemmas (Gathering and Cleanup) and two frontier LLMs (Claude Sonnet 4.6, Gemini 3.1 Pro), dense feedback consistently matches or exceeds sparse feedback on all metrics. We explain the asymmetry through feedback aliasing: when scalar reward alone maps distinct failure modes to the same value (e.g., under- vs. over-cleaning), social metrics break the alias and let the LLM diagnose which corrective direction to take. Social metrics thus function as a coordination signal rather than a distraction, yielding strategies such as Voronoi territory partitioning and waste-adaptive cleaner schedules. Code at this https URL.
>
---
#### [replaced 181] OARelatedWork: A Large-Scale Dataset of Related Work Sections with Full-texts from Open Access Sources
- **分类: cs.CL**

- **简介: 该论文提出OARelatedWork数据集，用于相关工作生成任务，解决从全文中生成完整相关工作段落的问题。通过基准测试和分析，揭示LLMs在处理全文字上下文时的挑战，并引入新的评估框架。**

- **链接: [https://arxiv.org/pdf/2405.01930](https://arxiv.org/pdf/2405.01930)**

> **作者:** Martin Docekal; Martin Fajcik; Pavel Smrz
>
> **摘要:** This paper introduces OARelatedWork: a dataset for related work generation from open-access sources. It is the first large-scale multi-document summarization dataset for related work generation, containing whole related work sections and full texts of cited papers. Its validation and test splits are constructed so that every cited paper is available in full text, enabling controlled evaluation of full-text related work generation. The dataset includes 94 450 papers and 5 824 689 unique referenced papers from multiple domains. With OARelatedWork, we aim to shift the field from generating parts of related work sections from abstracts only to generating entire related work sections from all available content. We (i) benchmark a wide spectrum of models, highlighting that synthesizing massive full-text contexts remains challenge even for modern Large Language Models (LLMs): under our statement-level judge, GPT-4o-mini's evidence-grounded True rate drops from 92.9% with abstracts to 83.8% with full texts. We (ii) empirically analyze human writing behavior through a human evaluation over 40 papers and 408 factual statements, revealing that authors frequently introduce abstractive claims ungrounded in localized source texts; consequently, advanced LLMs actually surpass human baselines in strict, evidence-grounded factuality. Finally, we (iii) conduct a fine-grained meta-evaluation, revealing that standard reference-based metrics are inadequate for evaluating such long-form structured outputs, and introduce a robust statement-level evaluation framework to address this gap.
>
---
#### [replaced 182] Demystifying Multi-Agent Debate: The Role of Confidence and Diversity
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在提升多智能体辩论效果。针对传统方法效率低、效果不佳的问题，提出通过增强观点多样性和引入置信度沟通来改进辩论机制。**

- **链接: [https://arxiv.org/pdf/2601.19921](https://arxiv.org/pdf/2601.19921)**

> **作者:** Xiaochen Zhu; Caiqi Zhang; Yizhou Chi; Tom Stafford; Nigel Collier; Andreas Vlachos
>
> **摘要:** Multi-agent debate (MAD) is widely used to improve large language model (LLM) performance through test-time scaling, yet recent work shows that vanilla MAD often underperforms simple majority vote despite higher computational cost. Studies show that, under homogeneous agents and uniform belief updates, debate preserves expected correctness and therefore cannot reliably improve outcomes. Drawing on findings from human deliberation and collective decision-making, we identify two key mechanisms missing from vanilla MAD: (i) diversity of initial viewpoints and (ii) explicit, calibrated confidence communication. We propose two lightweight interventions. First, a diversity-aware initialisation that selects a more diverse pool of candidate answers, increasing the likelihood that a correct hypothesis is present at the start of debate. Second, a confidence-modulated debate protocol in which agents express calibrated confidence and condition their updates on others' confidence. We show theoretically that diversity-aware initialisation improves the prior probability of MAD success without changing the underlying update dynamics, while confidence-modulated updates enable debate to systematically drift to the correct hypothesis. Empirically, across six reasoning-oriented QA benchmarks, our methods consistently outperform vanilla MAD and majority vote. Our results connect human deliberation with LLM-based debate and demonstrate that simple, principled modifications can substantially enhance debate effectiveness.
>
---
#### [replaced 183] Uncovering Competency Gaps in Large Language Models and Their Benchmarks
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于模型评估任务，旨在解决基准测试中隐藏的模型弱点和覆盖不均问题。通过概念激活方法，识别模型与基准中的细粒度能力缺口。**

- **链接: [https://arxiv.org/pdf/2512.20638](https://arxiv.org/pdf/2512.20638)**

> **作者:** Maty Bohacek; Nino Scherrer; Nicholas Dufour; Thomas Leung; Christoph Bregler; Stephanie C. Y. Chan
>
> **摘要:** The evaluation of large language models relies heavily on standardized benchmarks. These benchmarks provide useful aggregated metrics, but can obscure (i) particular sub-areas where the models are weak ("model gaps") and (ii) imbalanced coverage in the benchmarks themselves ("benchmark gaps"). To automatically uncover both types of gaps, we propose a simple new method using concept activations from sparse autoencoders, to identify fine-grained gaps on a per-concept basis. The method also benefits from grounding evaluation in the model's internal representations, as well as easy comparison across benchmarks. We applied the method to five popular open-source models and more than a dozen benchmarks, as illustrative examples. As validation of the approach, we found that our automatic, unsupervised method was able to recover model gaps that have been previously documented in the literature (e.g. relating to sycophancy), in addition to identifying novel model gaps. We were also able to automatically uncover benchmark gaps: core concepts that should fall within the scope of a given benchmark. Our "competency gaps" method can be used to complement existing benchmarks, by providing a concept-level decomposition of model behavior, and by helping benchmark developers iterate upon benchmark design. Code is available at this https URL.
>
---
