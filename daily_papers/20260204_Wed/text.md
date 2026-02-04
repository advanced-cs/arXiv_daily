# 自然语言处理 cs.CL

- **最新发布 129 篇**

- **更新 83 篇**

## 最新发布

#### [new 001] SAES-SVD: Self-Adaptive Suppression of Accumulated and Local Errors for SVD-based LLM Compression
- **分类: cs.CL**

- **简介: 该论文属于大语言模型压缩任务，旨在解决传统方法中误差累积导致性能下降的问题。提出SAES-SVD框架，通过联合优化层内重建和层间误差补偿，提升压缩效果。**

- **链接: [https://arxiv.org/pdf/2602.03051v1](https://arxiv.org/pdf/2602.03051v1)**

> **作者:** Xing Hu; Dawei Yang; Yuan Cheng; Zhixuan Chen; Zukang Xu
>
> **摘要:** The rapid growth in the parameter scale of large language models (LLMs) has created a high demand for efficient compression techniques. As a hardware-agnostic and highly compatible technique, low-rank compression has been widely adopted. However, existing methods typically compress each layer independently by minimizing per-layer reconstruction error, overlooking a critical limitation: the reconstruction error propagates and accumulates through the network, which leads to amplified global deviations from the full-precision baseline. To address this, we propose Self-Adaptive Error Suppression SVD (SAES-SVD), a LLMs compression framework that jointly optimizes intra-layer reconstruction and inter-layer error compensation. SAES-SVD is composed of two novel components: (1) Cumulative Error-Aware Layer Compression (CEALC), which formulates the compression objective as a combination of local reconstruction and weighted cumulative error compensation. Based on it, we derive a closed-form low-rank solution relied on second-order activation statistics, which explicitly aligns each layer's output with its full-precision counterpart to compensate for accumulated errors. (2) Adaptive Collaborative Error Suppression (ACES), which automatically adjusts the weighting coefficient to enhance the low-rank structure of the compression objective in CEALC. Specifically, the coefficient is optimized to maximize the ratio between the Frobenius norm of the compressed layer's output and that of the compression objective under a fixed rank, thus ensuring that the rank budget is utilized effectively. Extensive experiments across multiple LLM architectures and tasks show that, without fine-tuning or mixed-rank strategies, SAES-SVD consistently improves post-compression performance.
>
---
#### [new 002] Parallel-Probe: Towards Efficient Parallel Thinking via 2D Probing
- **分类: cs.CL**

- **简介: 该论文属于推理任务，解决并行思考的效率问题。提出Parallel-Probe方法，通过2D探测优化并行分支的深度和宽度，减少计算成本。**

- **链接: [https://arxiv.org/pdf/2602.03845v1](https://arxiv.org/pdf/2602.03845v1)**

> **作者:** Tong Zheng; Chengsong Huang; Runpeng Dai; Yun He; Rui Liu; Xin Ni; Huiwen Bao; Kaishen Wang; Hongtu Zhu; Jiaxin Huang; Furong Huang; Heng Huang
>
> **备注:** 14 pages
>
> **摘要:** Parallel thinking has emerged as a promising paradigm for reasoning, yet it imposes significant computational burdens. Existing efficiency methods primarily rely on local, per-trajectory signals and lack principled mechanisms to exploit global dynamics across parallel branches. We introduce 2D probing, an interface that exposes the width-depth dynamics of parallel thinking by periodically eliciting intermediate answers from all branches. Our analysis reveals three key insights: non-monotonic scaling across width-depth allocations, heterogeneous reasoning branch lengths, and early stabilization of global consensus. Guided by these insights, we introduce $\textbf{Parallel-Probe}$, a training-free controller designed to optimize online parallel thinking. Parallel-Probe employs consensus-based early stopping to regulate reasoning depth and deviation-based branch pruning to dynamically adjust width. Extensive experiments across three benchmarks and multiple models demonstrate that Parallel-Probe establishes a superior Pareto frontier for test-time scaling. Compared to standard majority voting, it reduces sequential tokens by up to $\textbf{35.8}$% and total token cost by over $\textbf{25.8}$% while maintaining competitive accuracy.
>
---
#### [new 003] One Model, All Roles: Multi-Turn, Multi-Agent Self-Play Reinforcement Learning for Conversational Social Intelligence
- **分类: cs.CL**

- **简介: 该论文提出OMAR框架，用于多轮多智能体对话中的社会智能学习。任务是提升AI在群体对话中的社交能力，解决传统方法难以处理复杂社会互动的问题。通过自对弈强化学习，模型学会 empathy、说服等社会技能。**

- **链接: [https://arxiv.org/pdf/2602.03109v1](https://arxiv.org/pdf/2602.03109v1)**

> **作者:** Bowen Jiang; Taiwei Shi; Ryo Kamoi; Yuan Yuan; Camillo J. Taylor; Longqi Yang; Pei Zhou; Sihao Chen
>
> **摘要:** This paper introduces OMAR: One Model, All Roles, a reinforcement learning framework that enables AI to develop social intelligence through multi-turn, multi-agent conversational self-play. Unlike traditional paradigms that rely on static, single-turn optimizations, OMAR allows a single model to role-play all participants in a conversation simultaneously, learning to achieve long-term goals and complex social norms directly from dynamic social interaction. To ensure training stability across long dialogues, we implement a hierarchical advantage estimation that calculates turn-level and token-level advantages. Evaluations in the SOTOPIA social environment and Werewolf strategy games show that our trained models develop fine-grained, emergent social intelligence, such as empathy, persuasion, and compromise seeking, demonstrating the effectiveness of learning collaboration even under competitive scenarios. While we identify practical challenges like reward hacking, our results show that rich social intelligence can emerge without human supervision. We hope this work incentivizes further research on AI social intelligence in group conversations.
>
---
#### [new 004] Use Graph When It Needs: Efficiently and Adaptively Integrating Retrieval-Augmented Generation with Graphs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识密集型任务，解决LLM在复杂查询中的性能问题。提出EA-GraphRAG框架，动态结合RAG与GraphRAG，提升准确率并降低延迟。**

- **链接: [https://arxiv.org/pdf/2602.03578v1](https://arxiv.org/pdf/2602.03578v1)**

> **作者:** Su Dong; Qinggang Zhang; Yilin Xiao; Shengyuan Chen; Chuang Zhou; Xiao Huang
>
> **摘要:** Large language models (LLMs) often struggle with knowledge-intensive tasks due to hallucinations and outdated parametric knowledge. While Retrieval-Augmented Generation (RAG) addresses this by integrating external corpora, its effectiveness is limited by fragmented information in unstructured domain documents. Graph-augmented RAG (GraphRAG) emerged to enhance contextual reasoning through structured knowledge graphs, yet paradoxically underperforms vanilla RAG in real-world scenarios, exhibiting significant accuracy drops and prohibitive latency despite gains on complex queries. We identify the rigid application of GraphRAG to all queries, regardless of complexity, as the root cause. To resolve this, we propose an efficient and adaptive GraphRAG framework called EA-GraphRAG that dynamically integrates RAG and GraphRAG paradigms through syntax-aware complexity analysis. Our approach introduces: (i) a syntactic feature constructor that parses each query and extracts a set of structural features; (ii) a lightweight complexity scorer that maps these features to a continuous complexity score; and (iii) a score-driven routing policy that selects dense RAG for low-score queries, invokes graph-based retrieval for high-score queries, and applies complexity-aware reciprocal rank fusion to handle borderline cases. Extensive experiments on a comprehensive benchmark, consisting of two single-hop and two multi-hop QA benchmarks, demonstrate that our EA-GraphRAG significantly improves accuracy, reduces latency, and achieves state-of-the-art performance in handling mixed scenarios involving both simple and complex queries.
>
---
#### [new 005] They Said Memes Were Harmless-We Found the Ones That Hurt: Decoding Jokes, Symbols, and Cultural References
- **分类: cs.CL**

- **简介: 该论文属于社会有害内容检测任务，旨在解决文化符号隐含意图、边界模糊和模型不透明问题。提出CROSS-ALIGN+框架，提升检测效果与可解释性。**

- **链接: [https://arxiv.org/pdf/2602.03822v1](https://arxiv.org/pdf/2602.03822v1)**

> **作者:** Sahil Tripathi; Gautam Siddharth Kashyap; Mehwish Nasim; Jian Yang; Jiechao Gao; Usman Naseem
>
> **备注:** Accepted at the The Web Conference 2026 (Research Track)
>
> **摘要:** Meme-based social abuse detection is challenging because harmful intent often relies on implicit cultural symbolism and subtle cross-modal incongruence. Prior approaches, from fusion-based methods to in-context learning with Large Vision-Language Models (LVLMs), have made progress but remain limited by three factors: i) cultural blindness (missing symbolic context), ii) boundary ambiguity (satire vs. abuse confusion), and iii) lack of interpretability (opaque model reasoning). We introduce CROSS-ALIGN+, a three-stage framework that systematically addresses these limitations: (1) Stage I mitigates cultural blindness by enriching multimodal representations with structured knowledge from ConceptNet, Wikidata, and Hatebase; (2) Stage II reduces boundary ambiguity through parameter-efficient LoRA adapters that sharpen decision boundaries; and (3) Stage III enhances interpretability by generating cascaded explanations. Extensive experiments on five benchmarks and eight LVLMs demonstrate that CROSS-ALIGN+ consistently outperforms state-of-the-art methods, achieving up to 17% relative F1 improvement while providing interpretable justifications for each decision.
>
---
#### [new 006] ForesightKV: Optimizing KV Cache Eviction for Reasoning Models by Learning Long-Term Contribution
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于模型优化任务，解决长序列生成中KV缓存占用过高的问题。通过学习预测重要KV对，提出ForesightKV框架，提升效率与性能。**

- **链接: [https://arxiv.org/pdf/2602.03203v1](https://arxiv.org/pdf/2602.03203v1)**

> **作者:** Zican Dong; Peiyu Liu; Junyi Li; Zhipeng Chen; Han Peng; Shuo Wang; Wayne Xin Zhao
>
> **摘要:** Recently, large language models (LLMs) have shown remarkable reasoning abilities by producing long reasoning traces. However, as the sequence length grows, the key-value (KV) cache expands linearly, incurring significant memory and computation costs. Existing KV cache eviction methods mitigate this issue by discarding less important KV pairs, but often fail to capture complex KV dependencies, resulting in performance degradation. To better balance efficiency and performance, we introduce ForesightKV, a training-based KV cache eviction framework that learns to predict which KV pairs to evict during long-text generations. We first design the Golden Eviction algorithm, which identifies the optimal eviction KV pairs at each step using future attention scores. These traces and the scores at each step are then distilled via supervised training with a Pairwise Ranking Loss. Furthermore, we formulate cache eviction as a Markov Decision Process and apply the GRPO algorithm to mitigate the significant language modeling loss increase on low-entropy tokens. Experiments on AIME2024 and AIME2025 benchmarks of three reasoning models demonstrate that ForesightKV consistently outperforms prior methods under only half the cache budget, while benefiting synergistically from both supervised and reinforcement learning approaches.
>
---
#### [new 007] ATACompressor: Adaptive Task-Aware Compression for Efficient Long-Context Processing in LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，解决长文本输入中关键信息丢失的问题。提出ATACompressor，通过自适应压缩保留重要信息，提升处理效率与效果。**

- **链接: [https://arxiv.org/pdf/2602.03226v1](https://arxiv.org/pdf/2602.03226v1)**

> **作者:** Xuancheng Li; Haitao Li; Yujia Zhou; Qingyao Ai; Yiqun Liu
>
> **摘要:** Long-context inputs in large language models (LLMs) often suffer from the "lost in the middle" problem, where critical information becomes diluted or ignored due to excessive length. Context compression methods aim to address this by reducing input size, but existing approaches struggle with balancing information preservation and compression efficiency. We propose Adaptive Task-Aware Compressor (ATACompressor), which dynamically adjusts compression based on the specific requirements of the task. ATACompressor employs a selective encoder that compresses only the task-relevant portions of long contexts, ensuring that essential information is preserved while reducing unnecessary content. Its adaptive allocation controller perceives the length of relevant content and adjusts the compression rate accordingly, optimizing resource utilization. We evaluate ATACompressor on three QA datasets: HotpotQA, MSMARCO, and SQUAD-showing that it outperforms existing methods in terms of both compression efficiency and task performance. Our approach provides a scalable solution for long-context processing in LLMs. Furthermore, we perform a range of ablation studies and analysis experiments to gain deeper insights into the key components of ATACompressor.
>
---
#### [new 008] When Efficient Communication Explains Convexity
- **分类: cs.CL; cs.IT**

- **简介: 该论文属于语言学与信息论交叉任务，旨在解释语义类型学中凸性现象。通过信息瓶颈方法，研究高效通信如何影响凸性，揭示其与最优表达的关系。**

- **链接: [https://arxiv.org/pdf/2602.02821v1](https://arxiv.org/pdf/2602.02821v1)**

> **作者:** Ashvin Ranjan; Shane Steinert-Threlkeld
>
> **摘要:** Much recent work has argued that the variation in the languages of the world can be explained from the perspective of efficient communication; in particular, languages can be seen as optimally balancing competing pressures to be simple and to be informative. Focusing on the expression of meaning -- semantic typology -- the present paper asks what factors are responsible for successful explanations in terms of efficient communication. Using the Information Bottleneck (IB) approach to formalizing this trade-off, we first demonstrate and analyze a correlation between optimality in the IB sense and a novel generalization of convexity to this setting. In a second experiment, we manipulate various modeling parameters in the IB framework to determine which factors drive the correlation between convexity and optimality. We find that the convexity of the communicative need distribution plays an especially important role. These results move beyond showing that efficient communication can explain aspects of semantic typology into explanations for why that is the case by identifying which underlying factors are responsible.
>
---
#### [new 009] ROSA-Tuning: Enhancing Long-Context Modeling via Suffix Matching
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决长文本建模中计算效率与上下文覆盖的矛盾。提出ROSA-Tuning方法，通过检索与召回机制提升模型长距离依赖捕捉能力，同时保持高效计算。**

- **链接: [https://arxiv.org/pdf/2602.02499v1](https://arxiv.org/pdf/2602.02499v1)**

> **作者:** Yunao Zheng; Xiaojie Wang; Lei Ren; Wei Chen
>
> **摘要:** Long-context capability and computational efficiency are among the central challenges facing today's large language models. Existing efficient attention methods reduce computational complexity, but they typically suffer from a limited coverage of the model state. This paper proposes ROSA-Tuning, a retrieval-and-recall mechanism for enhancing the long-context modeling ability of pretrained models. Beyond the standard attention mechanism, ROSA-Tuning introduces in parallel a CPU-based ROSA (RWKV Online Suffix Automaton) retrieval module, which efficiently locates historical positions in long contexts that are relevant to the current query, and injects the retrieved information into the model state in a trainable manner; subsequent weighted fusion can then be handled by range-restricted attention. To enable end-to-end training, we design a binary discretization strategy and a counterfactual gradient algorithm, and further optimize overall execution efficiency via an asynchronous CPU-GPU pipeline. Systematic evaluations on Qwen3-Base-1.7B show that ROSA-Tuning substantially restores the long-context modeling ability of windowed-attention models, achieving performance close to and in some cases matching global attention on benchmarks such as LongBench, while maintaining computational efficiency and GPU memory usage that are nearly comparable to windowed-attention methods, offering a new technical path for efficient long-context processing. The example code can be found at https://github.com/zyaaa-ux/ROSA-Tuning.
>
---
#### [new 010] PEGRL: Improving Machine Translation by Post-Editing Guided Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出PEGRL框架，用于改进机器翻译中的强化学习。针对噪声信号和轨迹空间大导致的优化难题，通过后编辑辅助任务稳定训练并引导优化，实现更高效的翻译效果。**

- **链接: [https://arxiv.org/pdf/2602.03352v1](https://arxiv.org/pdf/2602.03352v1)**

> **作者:** Yunzhi Shen; Hao Zhou; Xin Huang; Xue Han; Junlan Feng; Shujian Huang
>
> **摘要:** Reinforcement learning (RL) has shown strong promise for LLM-based machine translation, with recent methods such as GRPO demonstrating notable gains; nevertheless, translation-oriented RL remains challenged by noisy learning signals arising from Monte Carlo return estimation, as well as a large trajectory space that favors global exploration over fine-grained local optimization. We introduce \textbf{PEGRL}, a \textit{two-stage} RL framework that uses post-editing as an auxiliary task to stabilize training and guide overall optimization. At each iteration, translation outputs are sampled to construct post-editing inputs, allowing return estimation in the post-editing stage to benefit from conditioning on the current translation behavior, while jointly supporting both global exploration and fine-grained local optimization. A task-specific weighting scheme further balances the contributions of translation and post-editing objectives, yielding a biased yet more sample-efficient estimator. Experiments on English$\to$Finnish, English$\to$Turkish, and English$\leftrightarrow$Chinese show consistent gains over RL baselines, and for English$\to$Turkish, performance on COMET-KIWI is comparable to advanced LLM-based systems (DeepSeek-V3.2).
>
---
#### [new 011] Time-Critical Multimodal Medical Transportation: Organs, Patients, and Medical Supplies
- **分类: cs.CL**

- **简介: 论文研究多模式医疗运输系统，解决紧急医疗物资与患者快速运输问题。通过设计启发式算法优化车辆调度，测试不同车队配置以提高效率并降低成本。**

- **链接: [https://arxiv.org/pdf/2602.02736v1](https://arxiv.org/pdf/2602.02736v1)**

> **作者:** Elaheh Sabziyan Varnousfaderani; Syed A. M. Shihab; Mohammad Taghizadeh
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Timely transportation of organs, patients, and medical supplies is critical to modern healthcare, particularly in emergencies and transplant scenarios where even short delays can severely impact outcomes. Traditional ground-based vehicles such as ambulances are often hindered by traffic congestion; while air vehicles such as helicopters are faster but costly. Emerging air vehicles -- Unmanned Aerial Vehicles and electric vertical take-off and landing aircraft -- have lower operating costs, but remain limited by range and susceptibility to weather conditions. A multimodal transportation system that integrates both air and ground vehicles can leverage the strengths of each to enhance overall transportation efficiency. This study introduces a constructive greedy heuristic algorithm for multimodal vehicle dispatching for medical transportation. Four different fleet configurations were tested: (i) ambulances only, (ii) ambulances with Unmanned Aerial Vehicles, (iii) ambulances with electric vertical take-off and landing aircraft, and (iv) a fully integrated fleet of ambulances, Unmanned Aerial Vehicles, and electric vertical take-off and landing aircraft. The algorithm incorporates payload consolidation across compatible routes, accounts for traffic congestion in ground operations and weather conditions in aerial operations, while enabling rapid vehicle dispatching compared to computationally intensive optimization models. Using a common set of conditions, we evaluate all four fleet types to identify the most effective configurations for fulfilling medical transportation needs while minimizing operating costs, recharging/fuel costs, and total transportation time.
>
---
#### [new 012] Act or Clarify? Modeling Sensitivity to Uncertainty and Cost in Communication
- **分类: cs.CL**

- **简介: 该论文属于决策与沟通任务，研究在不确定性下是否寻求澄清。通过模型和实验，探讨了不确定性与行动成本的相互作用，揭示人类在风险高时更倾向于寻求澄清。**

- **链接: [https://arxiv.org/pdf/2602.02843v1](https://arxiv.org/pdf/2602.02843v1)**

> **作者:** Polina Tsvilodub; Karl Mulligan; Todd Snider; Robert D. Hawkins; Michael Franke
>
> **备注:** 6 pages, 3 figures, under review
>
> **摘要:** When deciding how to act under uncertainty, agents may choose to act to reduce uncertainty or they may act despite that uncertainty.In communicative settings, an important way of reducing uncertainty is by asking clarification questions (CQs). We predict that the decision to ask a CQ depends on both contextual uncertainty and the cost of alternative actions, and that these factors interact: uncertainty should matter most when acting incorrectly is costly. We formalize this interaction in a computational model based on expected regret: how much an agent stands to lose by acting now rather than with full information. We test these predictions in two experiments, one examining purely linguistic responses to questions and another extending to choices between clarification and non-linguistic action. Taken together, our results suggest a rational tradeoff: humans tend to seek clarification proportional to the risk of substantial loss when acting under uncertainty.
>
---
#### [new 013] HySparse: A Hybrid Sparse Attention Architecture with Oracle Token Selection and KV Cache Sharing
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出HySparse架构，解决稀疏注意力计算效率与性能问题。通过混合全连接与稀疏层，利用前层作为标记选择的精确依据，减少计算和内存占用。**

- **链接: [https://arxiv.org/pdf/2602.03560v1](https://arxiv.org/pdf/2602.03560v1)**

> **作者:** Yizhao Gao; Jianyu Wei; Qihao Zhang; Yu Cheng; Shimao Chen; Zhengju Tang; Zihan Jiang; Yifan Song; Hailin Zhang; Liang Zhao; Bo Yang; Gang Wang; Shijie Cao; Fuli Luo
>
> **备注:** 17 pages, 2 figures
>
> **摘要:** This work introduces Hybrid Sparse Attention (HySparse), a new architecture that interleaves each full attention layer with several sparse attention layers. While conceptually simple, HySparse strategically derives each sparse layer's token selection and KV caches directly from the preceding full attention layer. This architecture resolves two fundamental limitations of prior sparse attention methods. First, conventional approaches typically rely on additional proxies to predict token importance, introducing extra complexity and potentially suboptimal performance. In contrast, HySparse uses the full attention layer as a precise oracle to identify important tokens. Second, existing sparse attention designs often reduce computation without saving KV cache. HySparse enables sparse attention layers to reuse the full attention KV cache, thereby reducing both computation and memory. We evaluate HySparse on both 7B dense and 80B MoE models. Across all settings, HySparse consistently outperforms both full attention and hybrid SWA baselines. Notably, in the 80B MoE model with 49 total layers, only 5 layers employ full attention, yet HySparse achieves substantial performance gains while reducing KV cache storage by nearly 10x.
>
---
#### [new 014] From Task Solving to Robust Real-World Adaptation in LLM Agents
- **分类: cs.CL; cs.LG**

- **简介: 本文研究LLM代理在真实世界中的适应能力，解决任务求解与鲁棒性之间的差距。通过网格游戏测试代理在不确定环境下的表现，分析其适应策略与失败原因。**

- **链接: [https://arxiv.org/pdf/2602.02760v1](https://arxiv.org/pdf/2602.02760v1)**

> **作者:** Pouya Pezeshkpour; Estevam Hruschka
>
> **摘要:** Large language models are increasingly deployed as specialized agents that plan, call tools, and take actions over extended horizons. Yet many existing evaluations assume a "clean interface" where dynamics are specified and stable, tools and sensors are reliable, and success is captured by a single explicit objective-often overestimating real-world readiness. In practice, agents face underspecified rules, unreliable signals, shifting environments, and implicit, multi-stakeholder goals. The challenge is therefore not just solving tasks, but adapting while solving: deciding what to trust, what is wanted, when to verify, and when to fall back or escalate. We stress-test deployment-relevant robustness under four operational circumstances: partial observability, dynamic environments, noisy signals, and dynamic agent state. We benchmark agentic LLMs in a grid-based game with a simple goal but long-horizon execution. Episodes violate clean-interface assumptions yet remain solvable, forcing agents to infer rules, pay for information, adapt to environmental and internal shifts, and act cautiously under noise. Across five state-of-the-art LLM agents, we find large gaps between nominal task-solving and deployment-like robustness. Performance generally degrades as grid size and horizon increase, but rankings are unstable: weaker models can beat stronger ones when strategy matches the uncertainty regime. Despite no explicit instruction, agents trade off completion, efficiency, and penalty avoidance, suggesting partial objective inference. Ablations and feature analyses reveal model-specific sensitivities and failure drivers, motivating work on verification, safe action selection, and objective inference under partial observability, noise, and non-stationarity.
>
---
#### [new 015] Instruction Anchors: Dissecting the Causal Dynamics of Modality Arbitration
- **分类: cs.CL**

- **简介: 该论文研究多模态大语言模型的模态选择机制，旨在揭示其决策过程。通过分析信息流，发现指令标记作为结构锚点，指导模态竞争，提出提升模型透明度的框架。**

- **链接: [https://arxiv.org/pdf/2602.03677v1](https://arxiv.org/pdf/2602.03677v1)**

> **作者:** Yu Zhang; Mufan Xu; Xuefeng Bai; Kehai chen; Pengfei Zhang; Yang Xiang; Min Zhang
>
> **备注:** Modality Following
>
> **摘要:** Modality following serves as the capacity of multimodal large language models (MLLMs) to selectively utilize multimodal contexts based on user instructions. It is fundamental to ensuring safety and reliability in real-world deployments. However, the underlying mechanisms governing this decision-making process remain poorly understood. In this paper, we investigate its working mechanism through an information flow lens. Our findings reveal that instruction tokens function as structural anchors for modality arbitration: Shallow attention layers perform non-selective information transfer, routing multimodal cues to these anchors as a latent buffer; Modality competition is resolved within deep attention layers guided by the instruction intent, while MLP layers exhibit semantic inertia, acting as an adversarial force. Furthermore, we identify a sparse set of specialized attention heads that drive this arbitration. Causal interventions demonstrate that manipulating a mere $5\%$ of these critical heads can decrease the modality-following ratio by $60\%$ through blocking, or increase it by $60\%$ through targeted amplification of failed samples. Our work provides a substantial step toward model transparency and offers a principled framework for the orchestration of multimodal information in MLLMs.
>
---
#### [new 016] CUBO: Self-Contained Retrieval-Augmented Generation on Consumer Laptops 10 GB Corpora, 16 GB RAM, Single-Device Deployment
- **分类: cs.CL**

- **简介: 该论文提出CUBO，一个在消费级笔记本上运行的检索增强生成系统，解决本地处理与内存限制问题，实现高效、合规的文档检索与生成。**

- **链接: [https://arxiv.org/pdf/2602.03731v1](https://arxiv.org/pdf/2602.03731v1)**

> **作者:** Paolo Astrino
>
> **备注:** 24 pages, 2 figures, 6 tables
>
> **摘要:** Organizations handling sensitive documents face a tension: cloud-based AI risks GDPR violations, while local systems typically require 18-32 GB RAM. This paper presents CUBO, a systems-oriented RAG platform for consumer laptops with 16 GB shared memory. CUBO's novelty lies in engineering integration of streaming ingestion (O(1) buffer overhead), tiered hybrid retrieval, and hardware-aware orchestration that enables competitive Recall@10 (0.48-0.97 across BEIR domains) within a hard 15.5 GB RAM ceiling. The 37,000-line codebase achieves retrieval latencies of 185 ms (p50) on C1,300 laptops while maintaining data minimization through local-only processing aligned with GDPR Art. 5(1)(c). Evaluation on BEIR benchmarks validates practical deployability for small-to-medium professional archives. The codebase is publicly available at https://github.com/PaoloAstrino/CUBO.
>
---
#### [new 017] RAGTurk: Best Practices for Retrieval Augmented Generation in Turkish
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于自然语言处理任务，解决土耳其语RAG系统设计问题。构建了土耳其语RAG数据集，评估不同阶段的RAG流程，提出高效配置方案。**

- **链接: [https://arxiv.org/pdf/2602.03652v1](https://arxiv.org/pdf/2602.03652v1)**

> **作者:** Süha Kağan Köse; Mehmet Can Baytekin; Burak Aktaş; Bilge Kaan Görür; Evren Ayberk Munis; Deniz Yılmaz; Muhammed Yusuf Kartal; Çağrı Toraman
>
> **备注:** Accepted by EACL 2026 SIGTURK
>
> **摘要:** Retrieval-Augmented Generation (RAG) enhances LLM factuality, yet design guidance remains English-centric, limiting insights for morphologically rich languages like Turkish. We address this by constructing a comprehensive Turkish RAG dataset derived from Turkish Wikipedia and CulturaX, comprising question-answer pairs and relevant passage chunks. We benchmark seven stages of the RAG pipeline, from query transformation and reranking to answer refinement, without task-specific fine-tuning. Our results show that complex methods like HyDE maximize accuracy (85%) that is considerably higher than the baseline (78.70%). Also a Pareto-optimal configuration using Cross-encoder Reranking and Context Augmentation achieves comparable performance (84.60%) with much lower cost. We further demonstrate that over-stacking generative modules can degrade performance by distorting morphological cues, whereas simple query clarification with robust reranking offers an effective solution.
>
---
#### [new 018] Privasis: Synthesizing the Largest "Public" Private Dataset from Scratch
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出Privasis，一个大规模合成隐私数据集，用于解决隐私敏感数据稀缺问题。任务是支持隐私保护研究，工作包括构建数据集并开发文本去敏模型。**

- **链接: [https://arxiv.org/pdf/2602.03183v1](https://arxiv.org/pdf/2602.03183v1)**

> **作者:** Hyunwoo Kim; Niloofar Mireshghallah; Michael Duan; Rui Xin; Shuyue Stella Li; Jaehun Jung; David Acuna; Qi Pang; Hanshen Xiao; G. Edward Suh; Sewoong Oh; Yulia Tsvetkov; Pang Wei Koh; Yejin Choi
>
> **备注:** For code and data, see https://privasis.github.io
>
> **摘要:** Research involving privacy-sensitive data has always been constrained by data scarcity, standing in sharp contrast to other areas that have benefited from data scaling. This challenge is becoming increasingly urgent as modern AI agents--such as OpenClaw and Gemini Agent--are granted persistent access to highly sensitive personal information. To tackle this longstanding bottleneck and the rising risks, we present Privasis (i.e., privacy oasis), the first million-scale fully synthetic dataset entirely built from scratch--an expansive reservoir of texts with rich and diverse private information--designed to broaden and accelerate research in areas where processing sensitive social data is inevitable. Compared to existing datasets, Privasis, comprising 1.4 million records, offers orders-of-magnitude larger scale with quality, and far greater diversity across various document types, including medical history, legal documents, financial records, calendars, and text messages with a total of 55.1 million annotated attributes such as ethnicity, date of birth, workplace, etc. We leverage Privasis to construct a parallel corpus for text sanitization with our pipeline that decomposes texts and applies targeted sanitization. Our compact sanitization models (<=4B) trained on this dataset outperform state-of-the-art large language models, such as GPT-5 and Qwen-3 235B. We plan to release data, models, and code to accelerate future research on privacy-sensitive domains and agents.
>
---
#### [new 019] Where Norms and References Collide: Evaluating LLMs on Normative Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言处理任务，探讨LLMs在规范推理中的表现。研究提出SNIC测试集，评估LLMs在理解与应用社会规范方面的能力，发现其存在明显不足。**

- **链接: [https://arxiv.org/pdf/2602.02975v1](https://arxiv.org/pdf/2602.02975v1)**

> **作者:** Mitchell Abrams; Kaveh Eskandari Miandoab; Felix Gervits; Vasanth Sarathy; Matthias Scheutz
>
> **备注:** Accepted to the 40th AAAI Conference on Artificial Intelligence (AAAI-26)
>
> **摘要:** Embodied agents, such as robots, will need to interact in situated environments where successful communication often depends on reasoning over social norms: shared expectations that constrain what actions are appropriate in context. A key capability in such settings is norm-based reference resolution (NBRR), where interpreting referential expressions requires inferring implicit normative expectations grounded in physical and social context. Yet it remains unclear whether Large Language Models (LLMs) can support this kind of reasoning. In this work, we introduce SNIC (Situated Norms in Context), a human-validated diagnostic testbed designed to probe how well state-of-the-art LLMs can extract and utilize normative principles relevant to NBRR. SNIC emphasizes physically grounded norms that arise in everyday tasks such as cleaning, tidying, and serving. Across a range of controlled evaluations, we find that even the strongest LLMs struggle to consistently identify and apply social norms, particularly when norms are implicit, underspecified, or in conflict. These findings reveal a blind spot in current LLMs and highlight a key challenge for deploying language-based systems in socially situated, embodied settings.
>
---
#### [new 020] MIRROR: A Multi-Agent Framework with Iterative Adaptive Revision and Hierarchical Retrieval for Optimization Modeling in Operations Research
- **分类: cs.CL**

- **简介: 该论文提出MIRROR框架，解决运筹学中自然语言到数学模型的自动转换问题。通过迭代修正和分层检索，提升建模准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2602.03318v1](https://arxiv.org/pdf/2602.03318v1)**

> **作者:** Yifan Shi; Jialong Shi; Jiayi Wang; Ye Fan; Jianyong Sun
>
> **摘要:** Operations Research (OR) relies on expert-driven modeling-a slow and fragile process ill-suited to novel scenarios. While large language models (LLMs) can automatically translate natural language into optimization models, existing approaches either rely on costly post-training or employ multi-agent frameworks, yet most still lack reliable collaborative error correction and task-specific retrieval, often leading to incorrect outputs. We propose MIRROR, a fine-tuning-free, end-to-end multi-agent framework that directly translates natural language optimization problems into mathematical models and solver code. MIRROR integrates two core mechanisms: (1) execution-driven iterative adaptive revision for automatic error correction, and (2) hierarchical retrieval to fetch relevant modeling and coding exemplars from a carefully curated exemplar library. Experiments show that MIRROR outperforms existing methods on standard OR benchmarks, with notable results on complex industrial datasets such as IndustryOR and Mamo-ComplexLP. By combining precise external knowledge infusion with systematic error correction, MIRROR provides non-expert users with an efficient and reliable OR modeling solution, overcoming the fundamental limitations of general-purpose LLMs in expert optimization tasks.
>
---
#### [new 021] Short Chains, Deep Thoughts: Balancing Reasoning Efficiency and Intra-Segment Capability via Split-Merge Optimization
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，旨在解决大推理模型的效率与能力平衡问题。提出CoSMo框架，通过分割合并优化减少冗余，提升推理效率与准确性。**

- **链接: [https://arxiv.org/pdf/2602.03141v1](https://arxiv.org/pdf/2602.03141v1)**

> **作者:** Runquan Gui; Jie Wang; Zhihai Wang; Chi Ma; Jianye Hao; Feng Wu
>
> **摘要:** While Large Reasoning Models (LRMs) have demonstrated impressive capabilities in solving complex tasks through the generation of long reasoning chains, this reliance on verbose generation results in significant latency and computational overhead. To address these challenges, we propose \textbf{CoSMo} (\textbf{Co}nsistency-Guided \textbf{S}plit-\textbf{M}erge \textbf{O}ptimization), a framework designed to eliminate structural redundancy rather than indiscriminately restricting token volume. Specifically, CoSMo utilizes a split-merge algorithm that dynamically refines reasoning chains by merging redundant segments and splitting logical gaps to ensure coherence. We then employ structure-aligned reinforcement learning with a novel segment-level budget to supervise the model in maintaining efficient reasoning structures throughout training. Extensive experiments across multiple benchmarks and backbones demonstrate that CoSMo achieves superior performance, improving accuracy by \textbf{3.3} points while reducing segment usage by \textbf{28.7\%} on average compared to reasoning efficiency baselines.
>
---
#### [new 022] CPMobius: Iterative Coach-Player Reasoning for Data-Free Reinforcement Learning
- **分类: cs.CL**

- **简介: 该论文提出CPMobius，解决数据依赖问题，通过协作式教练-选手机制提升推理模型，无需外部数据。**

- **链接: [https://arxiv.org/pdf/2602.02979v1](https://arxiv.org/pdf/2602.02979v1)**

> **作者:** Ran Li; Zeyuan Liu; Yinghao chen; Bingxiang He; Jiarui Yuan; Zixuan Fu; Weize Chen; Jinyi Hu; Zhiyuan Liu; Maosong Sun
>
> **备注:** work in progress
>
> **摘要:** Large Language Models (LLMs) have demonstrated strong potential in complex reasoning, yet their progress remains fundamentally constrained by reliance on massive high-quality human-curated tasks and labels, either through supervised fine-tuning (SFT) or reinforcement learning (RL) on reasoning-specific data. This dependence renders supervision-heavy training paradigms increasingly unsustainable, with signs of diminishing scalability already evident in practice. To overcome this limitation, we introduce CPMöbius (CPMobius), a collaborative Coach-Player paradigm for data-free reinforcement learning of reasoning models. Unlike traditional adversarial self-play, CPMöbius, inspired by real world human sports collaboration and multi-agent collaboration, treats the Coach and Player as independent but cooperative roles. The Coach proposes instructions targeted at the Player's capability and receives rewards based on changes in the Player's performance, while the Player is rewarded for solving the increasingly instructive tasks generated by the Coach. This cooperative optimization loop is designed to directly enhance the Player's mathematical reasoning ability. Remarkably, CPMöbius achieves substantial improvement without relying on any external training data, outperforming existing unsupervised approaches. For example, on Qwen2.5-Math-7B-Instruct, our method improves accuracy by an overall average of +4.9 and an out-of-distribution average of +5.4, exceeding RENT by +1.5 on overall accuracy and R-zero by +4.2 on OOD accuracy.
>
---
#### [new 023] Test-time Recursive Thinking: Self-Improvement without External Feedback
- **分类: cs.CL**

- **简介: 该论文属于模型自提升任务，旨在让大语言模型在无外部反馈情况下自我改进。通过提出TRT框架，解决生成高质量解和无监督选择正确答案的问题。**

- **链接: [https://arxiv.org/pdf/2602.03094v1](https://arxiv.org/pdf/2602.03094v1)**

> **作者:** Yufan Zhuang; Chandan Singh; Liyuan Liu; Yelong Shen; Dinghuai Zhang; Jingbo Shang; Jianfeng Gao; Weizhu Chen
>
> **摘要:** Modern Large Language Models (LLMs) have shown rapid improvements in reasoning capabilities, driven largely by reinforcement learning (RL) with verifiable rewards. Here, we ask whether these LLMs can self-improve without the need for additional training. We identify two core challenges for such systems: (i) efficiently generating diverse, high-quality candidate solutions, and (ii) reliably selecting correct answers in the absence of ground-truth supervision. To address these challenges, we propose Test-time Recursive Thinking (TRT), an iterative self-improvement framework that conditions generation on rollout-specific strategies, accumulated knowledge, and self-generated verification signals. Using TRT, open-source models reach 100% accuracy on AIME-25/24, and on LiveCodeBench's most difficult problems, closed-source models improve by 10.4-14.8 percentage points without external feedback.
>
---
#### [new 024] SEAD: Self-Evolving Agent for Multi-Turn Service Dialogue
- **分类: cs.CL**

- **简介: 该论文提出SEAD框架，解决服务对话中因数据质量差导致的性能问题。通过用户建模和角色扮演，提升任务完成率和对话效率。属于对话系统任务。**

- **链接: [https://arxiv.org/pdf/2602.03548v1](https://arxiv.org/pdf/2602.03548v1)**

> **作者:** Yuqin Dai; Ning Gao; Wei Zhang; Jie Wang; Zichen Luo; Jinpeng Wang; Yujie Wang; Ruiyuan Wu; Chaozheng Wang
>
> **摘要:** Large Language Models have demonstrated remarkable capabilities in open-domain dialogues. However, current methods exhibit suboptimal performance in service dialogues, as they rely on noisy, low-quality human conversation data. This limitation arises from data scarcity and the difficulty of simulating authentic, goal-oriented user behaviors. To address these issues, we propose SEAD (Self-Evolving Agent for Service Dialogue), a framework that enables agents to learn effective strategies without large-scale human annotations. SEAD decouples user modeling into two components: a Profile Controller that generates diverse user states to manage training curriculum, and a User Role-play Model that focuses on realistic role-playing. This design ensures the environment provides adaptive training scenarios rather than acting as an unfair adversary. Experiments demonstrate that SEAD significantly outperforms Open-source Foundation Models and Closed-source Commercial Models, improving task completion rate by 17.6% and dialogue efficiency by 11.1%. Code is available at: https://github.com/Da1yuqin/SEAD.
>
---
#### [new 025] HALT: Hallucination Assessment via Log-probs as Time series
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于 hallucination 检测任务，解决 LLM 中幻觉问题。提出 HALT 方法，利用 log-probabilities 作为时间序列进行检测，高效且无需访问内部信息。**

- **链接: [https://arxiv.org/pdf/2602.02888v1](https://arxiv.org/pdf/2602.02888v1)**

> **作者:** Ahmad Shapiro; Karan Taneja; Ashok Goel
>
> **摘要:** Hallucinations remain a major obstacle for large language models (LLMs), especially in safety-critical domains. We present HALT (Hallucination Assessment via Log-probs as Time series), a lightweight hallucination detector that leverages only the top-20 token log-probabilities from LLM generations as a time series. HALT uses a gated recurrent unit model combined with entropy-based features to learn model calibration bias, providing an extremely efficient alternative to large encoders. Unlike white-box approaches, HALT does not require access to hidden states or attention maps, relying only on output log-probabilities. Unlike black-box approaches, it operates on log-probs rather than surface-form text, which enables stronger domain generalization and compatibility with proprietary LLMs without requiring access to internal weights. To benchmark performance, we introduce HUB (Hallucination detection Unified Benchmark), which consolidates prior datasets into ten capabilities covering both reasoning tasks (Algorithmic, Commonsense, Mathematical, Symbolic, Code Generation) and general purpose skills (Chat, Data-to-Text, Question Answering, Summarization, World Knowledge). While being 30x smaller, HALT outperforms Lettuce, a fine-tuned modernBERT-base encoder, achieving a 60x speedup gain on HUB. HALT and HUB together establish an effective framework for hallucination detection across diverse LLM capabilities.
>
---
#### [new 026] Training Multi-Turn Search Agent via Contrastive Dynamic Branch Sampling
- **分类: cs.CL**

- **简介: 该论文属于强化学习任务，解决长程决策中的稀疏奖励问题。提出BranPO方法，通过对比采样提升多轮对话性能。**

- **链接: [https://arxiv.org/pdf/2602.03719v1](https://arxiv.org/pdf/2602.03719v1)**

> **作者:** Yubao Zhao; Weiquan Huang; Sudong Wang; Ruochen Zhao; Chen Chen; Yao Shu; Chengwei Qin
>
> **备注:** 24 pages, 5 figures
>
> **摘要:** Agentic reinforcement learning has enabled large language models to perform complex multi-turn planning and tool use. However, learning in long-horizon settings remains challenging due to sparse, trajectory-level outcome rewards. While prior tree-based methods attempt to mitigate this issue, they often suffer from high variance and computational inefficiency. Through empirical analysis of search agents, We identify a common pattern: performance diverges mainly due to decisions near the tail. Motivated by this observation, we propose Branching Relative Policy Optimization (BranPO), a value-free method that provides step-level contrastive supervision without dense rewards. BranPO truncates trajectories near the tail and resamples alternative continuations to construct contrastive suffixes over shared prefixes, reducing credit ambiguity in long-horizon rollouts. To further boost efficiency and stabilize training, we introduce difficulty-aware branch sampling to adapt branching frequency across tasks, and redundant step masking to suppress uninformative actions. Extensive experiments on various question answering benchmarks demonstrate that BranPO consistently outperforms strong baselines, achieving significant accuracy gains on long-horizon tasks without increasing the overall training budget. Our code is available at \href{https://github.com/YubaoZhao/BranPO}{code}.
>
---
#### [new 027] Assessing the Impact of Typological Features on Multilingual Machine Translation in the Age of Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于机器翻译任务，研究语言类型特征对多语言翻译质量的影响。针对不同语言的内在难度问题，分析了两种大型模型的表现，并提出优化解码策略的建议。**

- **链接: [https://arxiv.org/pdf/2602.03551v1](https://arxiv.org/pdf/2602.03551v1)**

> **作者:** Vitalii Hirak; Jaap Jumelet; Arianna Bisazza
>
> **备注:** 19 pages, 11 figures, EACL 2026
>
> **摘要:** Despite major advances in multilingual modeling, large quality disparities persist across languages. Besides the obvious impact of uneven training resources, typological properties have also been proposed to determine the intrinsic difficulty of modeling a language. The existing evidence, however, is mostly based on small monolingual language models or bilingual translation models trained from scratch. We expand on this line of work by analyzing two large pre-trained multilingual translation models, NLLB-200 and Tower+, which are state-of-the-art representatives of encoder-decoder and decoder-only machine translation, respectively. Based on a broad set of languages, we find that target language typology drives translation quality of both models, even after controlling for more trivial factors, such as data resourcedness and writing script. Additionally, languages with certain typological properties benefit more from a wider search of the output space, suggesting that such languages could profit from alternative decoding strategies beyond the standard left-to-right beam search. To facilitate further research in this area, we release a set of fine-grained typological properties for 212 languages of the FLORES+ MT evaluation benchmark.
>
---
#### [new 028] Cognitively Diverse Multiple-Choice Question Generation: A Hybrid Multi-Agent Framework with Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理中的题目生成任务，旨在解决自动化生成具有认知多样性的多项选择题的问题。作者提出ReQUESTA框架，结合大语言模型与规则组件，提升题目质量与控制性。**

- **链接: [https://arxiv.org/pdf/2602.03704v1](https://arxiv.org/pdf/2602.03704v1)**

> **作者:** Yu Tian; Linh Huynh; Katerina Christhilf; Shubham Chakraborty; Micah Watanabe; Tracy Arner; Danielle McNamara
>
> **备注:** This manuscript is under review at Electronics
>
> **摘要:** Recent advances in large language models (LLMs) have made automated multiple-choice question (MCQ) generation increasingly feasible; however, reliably producing items that satisfy controlled cognitive demands remains a challenge. To address this gap, we introduce ReQUESTA, a hybrid, multi-agent framework for generating cognitively diverse MCQs that systematically target text-based, inferential, and main idea comprehension. ReQUESTA decomposes MCQ authoring into specialized subtasks and coordinates LLM-powered agents with rule-based components to support planning, controlled generation, iterative evaluation, and post-processing. We evaluated the framework in a large-scale reading comprehension study using academic expository texts, comparing ReQUESTA-generated MCQs with those produced by a single-pass GPT-5 zero-shot baseline. Psychometric analyses of learner responses assessed item difficulty and discrimination, while expert raters evaluated question quality across multiple dimensions, including topic relevance and distractor quality. Results showed that ReQUESTA-generated items were consistently more challenging, more discriminative, and more strongly aligned with overall reading comprehension performance. Expert evaluations further indicated stronger alignment with central concepts and superior distractor linguistic consistency and semantic plausibility, particularly for inferential questions. These findings demonstrate that hybrid, agentic orchestration can systematically improve the reliability and controllability of LLM-based generation, highlighting workflow design as a key lever for structured artifact generation beyond single-pass prompting.
>
---
#### [new 029] Learning Query-Specific Rubrics from Human Preferences for DeepResearch Report Generation
- **分类: cs.CL**

- **简介: 该论文属于深度研究报告生成任务，解决缺乏有效评估标准的问题。通过学习用户偏好生成查询相关评分标准，提升报告质量。**

- **链接: [https://arxiv.org/pdf/2602.03619v1](https://arxiv.org/pdf/2602.03619v1)**

> **作者:** Changze Lv; Jie Zhou; Wentao Zhao; Jingwen Xu; Zisu Huang; Muzhao Tian; Shihan Dou; Tao Gui; Le Tian; Xiao Zhou; Xiaoqing Zheng; Xuanjing Huang; Jie Zhou
>
> **摘要:** Nowadays, training and evaluating DeepResearch-generated reports remain challenging due to the lack of verifiable reward signals. Accordingly, rubric-based evaluation has become a common practice. However, existing approaches either rely on coarse, pre-defined rubrics that lack sufficient granularity, or depend on manually constructed query-specific rubrics that are costly and difficult to scale. In this paper, we propose a pipeline to train human-preference-aligned query-specific rubric generators tailored for DeepResearch report generation. We first construct a dataset of DeepResearch-style queries annotated with human preferences over paired reports, and train rubric generators via reinforcement learning with a hybrid reward combining human preference supervision and LLM-based rubric evaluation. To better handle long-horizon reasoning, we further introduce a Multi-agent Markov-state (MaMs) workflow for report generation. We empirically show that our proposed rubric generators deliver more discriminative and better human-aligned supervision than existing rubric design strategies. Moreover, when integrated into the MaMs training framework, DeepResearch systems equipped with our rubric generators consistently outperform all open-source baselines on the DeepResearch Bench and achieve performance comparable to that of leading closed-source models.
>
---
#### [new 030] Learning to Reason Faithfully through Step-Level Faithfulness Maximization
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决LLM在多步推理中过度自信和幻觉问题。提出FaithRL框架，通过优化推理忠实性提升模型可靠性。**

- **链接: [https://arxiv.org/pdf/2602.03507v1](https://arxiv.org/pdf/2602.03507v1)**

> **作者:** Runquan Gui; Yafu Li; Xiaoye Qu; Ziyan Liu; Yeqiu Cheng; Yu Cheng
>
> **摘要:** Reinforcement Learning with Verifiable Rewards (RLVR) has markedly improved the performance of Large Language Models (LLMs) on tasks requiring multi-step reasoning. However, most RLVR pipelines rely on sparse outcome-based rewards, providing little supervision over intermediate steps and thus encouraging over-confidence and spurious reasoning, which in turn increases hallucinations. To address this, we propose FaithRL, a general reinforcement learning framework that directly optimizes reasoning faithfulness. We formalize a faithfulness-maximization objective and theoretically show that optimizing it mitigates over-confidence. To instantiate this objective, we introduce a geometric reward design and a faithfulness-aware advantage modulation mechanism that assigns step-level credit by penalizing unsupported steps while preserving valid partial derivations. Across diverse backbones and benchmarks, FaithRL consistently reduces hallucination rates while maintaining (and often improving) answer correctness. Further analysis confirms that FaithRL increases step-wise reasoning faithfulness and generalizes robustly. Our code is available at https://github.com/aintdoin/FaithRL.
>
---
#### [new 031] ReMiT: RL-Guided Mid-Training for Iterative LLM Evolution
- **分类: cs.CL**

- **简介: 该论文提出ReMiT方法，通过强化学习在训练中期动态调整模型权重，提升大语言模型的迭代进化能力，解决传统单向训练流程效率低的问题。**

- **链接: [https://arxiv.org/pdf/2602.03075v1](https://arxiv.org/pdf/2602.03075v1)**

> **作者:** Junjie Huang; Jiarui Qin; Di Yin; Weiwen Liu; Yong Yu; Xing Sun; Weinan Zhang
>
> **备注:** 25 pages
>
> **摘要:** Standard training pipelines for large language models (LLMs) are typically unidirectional, progressing from pre-training to post-training. However, the potential for a bidirectional process--where insights from post-training retroactively improve the pre-trained foundation--remains unexplored. We aim to establish a self-reinforcing flywheel: a cycle in which reinforcement learning (RL)-tuned model strengthens the base model, which in turn enhances subsequent post-training performance, requiring no specially trained teacher or reference model. To realize this, we analyze training dynamics and identify the mid-training (annealing) phase as a critical turning point for model capabilities. This phase typically occurs at the end of pre-training, utilizing high-quality corpora under a rapidly decaying learning rate. Building upon this insight, we introduce ReMiT (Reinforcement Learning-Guided Mid-Training). Specifically, ReMiT leverages the reasoning priors of RL-tuned models to dynamically reweight tokens during the mid-training phase, prioritizing those pivotal for reasoning. Empirically, ReMiT achieves an average improvement of 3\% on 10 pre-training benchmarks, spanning math, code, and general reasoning, and sustains these gains by over 2\% throughout the post-training pipeline. These results validate an iterative feedback loop, enabling continuous and self-reinforcing evolution of LLMs.
>
---
#### [new 032] STEMVerse: A Dual-Axis Diagnostic Framework for STEM Reasoning in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于STEM推理评估任务，旨在解决现有评测方法无法区分模型知识与认知缺陷的问题。提出STEMVerse框架，通过双轴分析系统评估LLMs的STEM能力。**

- **链接: [https://arxiv.org/pdf/2602.02497v1](https://arxiv.org/pdf/2602.02497v1)**

> **作者:** Xuzhao Li; Xuchen Li; Jian Zhao; Shiyu Hu
>
> **备注:** Preprint, Under review
>
> **摘要:** As Large Language Models (LLMs) achieve significant breakthroughs in complex reasoning tasks, evaluating their proficiency in science, technology, engineering, and mathematics (STEM) has become a primary method for measuring machine intelligence. However, current evaluation paradigms often treat benchmarks as isolated "silos," offering only monolithic aggregate scores that neglect the intricacies of both academic specialization and cognitive depth. This result-oriented approach fails to distinguish whether model errors stem from insufficient domain knowledge or deficiencies in cognitive capacity, thereby limiting the diagnostic value. To address this, we propose STEMVerse, a diagnostic framework designed to systematically analyze the STEM reasoning capabilities of LLMs. This framework characterizes model performance across academic specialization and cognitive complexity to map the capability required for reasoning. We re-aggregate over 20,000 STEM problems from mainstream benchmarks into a unified "Discipline $\times$ Cognition" capability space, assigning dual-axis labels to every instance. Utilizing this unified diagnostic framework, we systematically evaluate representative LLM families across varying parameter scales and training paradigms. Our empirical results reveal structural failure patterns in STEM reasoning. By integrating multi-disciplinary coverage and fine-grained cognitive stratification into a unified framework, STEMVerse provides a clear and actionable perspective for understanding the scientific reasoning characteristics of LLMs.
>
---
#### [new 033] Self-Verification Dilemma: Experience-Driven Suppression of Overused Checking in LLM Reasoning
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于大模型推理任务，解决自验证冗余问题。通过经验驱动方法减少不必要的自我检查，提升效率并保持准确率。**

- **链接: [https://arxiv.org/pdf/2602.03485v1](https://arxiv.org/pdf/2602.03485v1)**

> **作者:** Quanyu Long; Kai Jie Jiang; Jianda Chen; Xu Guo; Leilei Gan; Wenya Wang
>
> **备注:** 19 pages, 8 figures
>
> **摘要:** Large Reasoning Models (LRMs) achieve strong performance by generating long reasoning traces with reflection. Through a large-scale empirical analysis, we find that a substantial fraction of reflective steps consist of self-verification (recheck) that repeatedly confirm intermediate results. These rechecks occur frequently across models and benchmarks, yet the vast majority are confirmatory rather than corrective, rarely identifying errors and altering reasoning outcomes. This reveals a mismatch between how often self-verification is activated and how often it is actually useful. Motivated by this, we propose a novel, experience-driven test-time framework that reduces the overused verification. Our method detects the activation of recheck behavior, consults an offline experience pool of past verification outcomes, and estimates whether a recheck is likely unnecessary via efficient retrieval. When historical experience suggests unnecessary, a suppression signal redirects the model to proceed. Across multiple model and benchmarks, our approach reduces token usage up to 20.3% while maintaining the accuracy, and in some datasets even yields accuracy improvements.
>
---
#### [new 034] FactNet: A Billion-Scale Knowledge Graph for Multilingual Factual Grounding
- **分类: cs.CL**

- **简介: 该论文提出FactNet，一个包含17亿事实断言和301亿证据指针的多语言知识图谱，解决LLM事实幻觉问题，统一结构化知识与文本证据。**

- **链接: [https://arxiv.org/pdf/2602.03417v1](https://arxiv.org/pdf/2602.03417v1)**

> **作者:** Yingli Shen; Wen Lai; Jie Zhou; Xueren Zhang; Yudong Wang; Kangyang Luo; Shuo Wang; Ge Gao; Alexander Fraser; Maosong Sun
>
> **摘要:** While LLMs exhibit remarkable fluency, their utility is often compromised by factual hallucinations and a lack of traceable provenance. Existing resources for grounding mitigate this but typically enforce a dichotomy: they offer either structured knowledge without textual context (e.g., knowledge bases) or grounded text with limited scale and linguistic coverage. To bridge this gap, we introduce FactNet, a massive, open-source resource designed to unify 1.7 billion atomic assertions with 3.01 billion auditable evidence pointers derived exclusively from 316 Wikipedia editions. Unlike recent synthetic approaches, FactNet employs a strictly deterministic construction pipeline, ensuring that every evidence unit is recoverable with byte-level precision. Extensive auditing confirms a high grounding precision of 92.1%, even in long-tail languages. Furthermore, we establish FactNet-Bench, a comprehensive evaluation suite for Knowledge Graph Completion, Question Answering, and Fact Checking. FactNet provides the community with a foundational, reproducible resource for training and evaluating trustworthy, verifiable multilingual systems.
>
---
#### [new 035] TRE: Encouraging Exploration in the Trust Region
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于强化学习领域，针对大语言模型中熵正则化探索效果差的问题，提出TRE方法，在信任区域内鼓励有效探索，提升任务性能。**

- **链接: [https://arxiv.org/pdf/2602.03635v1](https://arxiv.org/pdf/2602.03635v1)**

> **作者:** Chao Huang; Yujing Lu; Quangang Li; Shenghe Wang; Yan Wang; Yueyang Zhang; Long Xia; Jiashu Zhao; Zhiyuan Sun; Daiting Shi; Tingwen Liu
>
> **摘要:** Entropy regularization is a standard technique in reinforcement learning (RL) to enhance exploration, yet it yields negligible effects or even degrades performance in Large Language Models (LLMs). We attribute this failure to the cumulative tail risk inherent to LLMs with massive vocabularies and long generation horizons. In such environments, standard global entropy maximization indiscriminately dilutes probability mass into the vast tail of invalid tokens rather than focusing on plausible candidates, thereby disrupting coherent reasoning. To address this, we propose Trust Region Entropy (TRE), a method that encourages exploration strictly within the model's trust region. Extensive experiments across mathematical reasoning (MATH), combinatorial search (Countdown), and preference alignment (HH) tasks demonstrate that TRE consistently outperforms vanilla PPO, standard entropy regularization, and other exploration baselines. Our code is available at https://github.com/WhyChaos/TRE-Encouraging-Exploration-in-the-Trust-Region.
>
---
#### [new 036] InfMem: Learning System-2 Memory Control for Long-Context Agent
- **分类: cs.CL**

- **简介: 该论文提出InfMem，解决长文档推理中的记忆控制问题。通过主动检索与压缩，提升多跳推理效果，优化推理效率。**

- **链接: [https://arxiv.org/pdf/2602.02704v1](https://arxiv.org/pdf/2602.02704v1)**

> **作者:** Xinyu Wang; Mingze Li; Peng Lu; Xiao-Wen Chang; Lifeng Shang; Jinping Li; Fei Mi; Prasanna Parthasarathi; Yufei Cui
>
> **摘要:** Reasoning over ultra-long documents requires synthesizing sparse evidence scattered across distant segments under strict memory constraints. While streaming agents enable scalable processing, their passive memory update strategy often fails to preserve low-salience bridging evidence required for multi-hop reasoning. We propose InfMem, a control-centric agent that instantiates System-2-style control via a PreThink-Retrieve-Write protocol. InfMem actively monitors evidence sufficiency, performs targeted in-document retrieval, and applies evidence-aware joint compression to update a bounded memory. To ensure reliable control, we introduce a practical SFT-to-RL training recipe that aligns retrieval, writing, and stopping decisions with end-task correctness. On ultra-long QA benchmarks from 32k to 1M tokens, InfMem consistently outperforms MemAgent across backbones. Specifically, InfMem improves average absolute accuracy by +10.17, +11.84, and +8.23 points on Qwen3-1.7B, Qwen3-4B, and Qwen2.5-7B, respectively, while reducing inference time by $3.9\times$ on average (up to $5.1\times$) via adaptive early stopping.
>
---
#### [new 037] A-RAG: Scaling Agentic Retrieval-Augmented Generation via Hierarchical Retrieval Interfaces
- **分类: cs.CL**

- **简介: 该论文提出A-RAG框架，解决传统RAG系统无法有效利用大模型能力的问题。通过引入分层检索接口，使模型能动态参与检索决策，提升问答性能。**

- **链接: [https://arxiv.org/pdf/2602.03442v1](https://arxiv.org/pdf/2602.03442v1)**

> **作者:** Mingxuan Du; Benfeng Xu; Chiwei Zhu; Shaohan Wang; Pengyu Wang; Xiaorui Wang; Zhendong Mao
>
> **备注:** 18 pages, 8 figures
>
> **摘要:** Frontier language models have demonstrated strong reasoning and long-horizon tool-use capabilities. However, existing RAG systems fail to leverage these capabilities. They still rely on two paradigms: (1) designing an algorithm that retrieves passages in a single shot and concatenates them into the model's input, or (2) predefining a workflow and prompting the model to execute it step-by-step. Neither paradigm allows the model to participate in retrieval decisions, preventing efficient scaling with model improvements. In this paper, we introduce A-RAG, an Agentic RAG framework that exposes hierarchical retrieval interfaces directly to the model. A-RAG provides three retrieval tools: keyword search, semantic search, and chunk read, enabling the agent to adaptively search and retrieve information across multiple granularities. Experiments on multiple open-domain QA benchmarks show that A-RAG consistently outperforms existing approaches with comparable or lower retrieved tokens, demonstrating that A-RAG effectively leverages model capabilities and dynamically adapts to different RAG tasks. We further systematically study how A-RAG scales with model size and test-time compute. We will release our code and evaluation suite to facilitate future research. Code and evaluation suite are available at https://github.com/Ayanami0730/arag.
>
---
#### [new 038] CL-bench: A Benchmark for Context Learning
- **分类: cs.CL**

- **简介: 该论文提出CL-bench，一个用于评估语言模型上下文学习能力的基准。旨在解决真实世界任务中模型需依赖特定上下文进行推理的问题。通过设计复杂任务和验证标准，测试模型在新知识下的学习能力。**

- **链接: [https://arxiv.org/pdf/2602.03587v1](https://arxiv.org/pdf/2602.03587v1)**

> **作者:** Shihan Dou; Ming Zhang; Zhangyue Yin; Chenhao Huang; Yujiong Shen; Junzhe Wang; Jiayi Chen; Yuchen Ni; Junjie Ye; Cheng Zhang; Huaibing Xie; Jianglu Hu; Shaolei Wang; Weichao Wang; Yanling Xiao; Yiting Liu; Zenan Xu; Zhen Guo; Pluto Zhou; Tao Gui; Zuxuan Wu; Xipeng Qiu; Qi Zhang; Xuanjing Huang; Yu-Gang Jiang; Di Wang; Shunyu Yao
>
> **备注:** 78 pages, 17 figures
>
> **摘要:** Current language models (LMs) excel at reasoning over prompts using pre-trained knowledge. However, real-world tasks are far more complex and context-dependent: models must learn from task-specific context and leverage new knowledge beyond what is learned during pre-training to reason and resolve tasks. We term this capability context learning, a crucial ability that humans naturally possess but has been largely overlooked. To this end, we introduce CL-bench, a real-world benchmark consisting of 500 complex contexts, 1,899 tasks, and 31,607 verification rubrics, all crafted by experienced domain experts. Each task is designed such that the new content required to resolve it is contained within the corresponding context. Resolving tasks in CL-bench requires models to learn from the context, ranging from new domain-specific knowledge, rule systems, and complex procedures to laws derived from empirical data, all of which are absent from pre-training. This goes far beyond long-context tasks that primarily test retrieval or reading comprehension, and in-context learning tasks, where models learn simple task patterns via instructions and demonstrations. Our evaluations of ten frontier LMs find that models solve only 17.2% of tasks on average. Even the best-performing model, GPT-5.1, solves only 23.7%, revealing that LMs have yet to achieve effective context learning, which poses a critical bottleneck for tackling real-world, complex context-dependent tasks. CL-bench represents a step towards building LMs with this fundamental capability, making them more intelligent and advancing their deployment in real-world scenarios.
>
---
#### [new 039] OCRTurk: A Comprehensive OCR Benchmark for Turkish
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于文档解析任务，旨在解决土耳其语文档处理基准缺失的问题。作者构建了OCRTurk基准，涵盖多种文档类型和难度，评估了多个OCR模型的表现。**

- **链接: [https://arxiv.org/pdf/2602.03693v1](https://arxiv.org/pdf/2602.03693v1)**

> **作者:** Deniz Yılmaz; Evren Ayberk Munis; Çağrı Toraman; Süha Kağan Köse; Burak Aktaş; Mehmet Can Baytekin; Bilge Kaan Görür
>
> **备注:** Accepted by EACL 2026 SIGTURK
>
> **摘要:** Document parsing is now widely used in applications, such as large-scale document digitization, retrieval-augmented generation, and domain-specific pipelines in healthcare and education. Benchmarking these models is crucial for assessing their reliability and practical robustness. Existing benchmarks mostly target high-resource languages and provide limited coverage for low-resource settings, such as Turkish. Moreover, existing studies on Turkish document parsing lack a standardized benchmark that reflects real-world scenarios and document diversity. To address this gap, we introduce OCRTurk, a Turkish document parsing benchmark covering multiple layout elements and document categories at three difficulty levels. OCRTurk consists of 180 Turkish documents drawn from academic articles, theses, slide decks, and non-academic articles. We evaluate seven OCR models on OCRTurk using element-wise metrics. Across difficulty levels, PaddleOCR achieves the strongest overall results, leading most element-wise metrics except figures and attaining high Normalized Edit Distance scores in easy, medium, and hard subsets. We also observe performance variation by document type. Models perform well on non-academic documents, while slideshows become the most challenging.
>
---
#### [new 040] LatentMem: Customizing Latent Memory for Multi-Agent Systems
- **分类: cs.CL; cs.LG; cs.MA**

- **简介: 该论文属于多智能体系统任务，旨在解决记忆同质化和信息过载问题。提出LatentMem框架，通过定制化记忆提升性能。**

- **链接: [https://arxiv.org/pdf/2602.03036v1](https://arxiv.org/pdf/2602.03036v1)**

> **作者:** Muxin Fu; Guibin Zhang; Xiangyuan Xue; Yafu Li; Zefeng He; Siyuan Huang; Xiaoye Qu; Yu Cheng; Yang Yang
>
> **摘要:** Large language model (LLM)-powered multi-agent systems (MAS) demonstrate remarkable collective intelligence, wherein multi-agent memory serves as a pivotal mechanism for continual adaptation. However, existing multi-agent memory designs remain constrained by two fundamental bottlenecks: (i) memory homogenization arising from the absence of role-aware customization, and (ii) information overload induced by excessively fine-grained memory entries. To address these limitations, we propose LatentMem, a learnable multi-agent memory framework designed to customize agent-specific memories in a token-efficient manner. Specifically, LatentMem comprises an experience bank that stores raw interaction trajectories in a lightweight form, and a memory composer that synthesizes compact latent memories conditioned on retrieved experience and agent-specific contexts. Further, we introduce Latent Memory Policy Optimization (LMPO), which propagates task-level optimization signals through latent memories to the composer, encouraging it to produce compact and high-utility representations. Extensive experiments across diverse benchmarks and mainstream MAS frameworks show that LatentMem achieves a performance gain of up to $19.36$% over vanilla settings and consistently outperforms existing memory architectures, without requiring any modifications to the underlying frameworks.
>
---
#### [new 041] Accelerating Scientific Research with Gemini: Case Studies and Common Techniques
- **分类: cs.CL; cs.AI**

- **简介: 论文探讨了如何利用Gemini模型加速科学研究，解决人机协作中的理论问题。通过案例研究，展示了AI在数学发现、证明生成等方面的应用，并提出有效协作方法。**

- **链接: [https://arxiv.org/pdf/2602.03837v1](https://arxiv.org/pdf/2602.03837v1)**

> **作者:** David P. Woodruff; Vincent Cohen-Addad; Lalit Jain; Jieming Mao; Song Zuo; MohammadHossein Bateni; Simina Branzei; Michael P. Brenner; Lin Chen; Ying Feng; Lance Fortnow; Gang Fu; Ziyi Guan; Zahra Hadizadeh; Mohammad T. Hajiaghayi; Mahdi JafariRaviz; Adel Javanmard; Karthik C. S.; Ken-ichi Kawarabayashi; Ravi Kumar; Silvio Lattanzi; Euiwoong Lee; Yi Li; Ioannis Panageas; Dimitris Paparas; Benjamin Przybocki; Bernardo Subercaseaux; Ola Svensson; Shayan Taherijam; Xuan Wu; Eylon Yogev; Morteza Zadimoghaddam; Samson Zhou; Vahab Mirrokni
>
> **摘要:** Recent advances in large language models (LLMs) have opened new avenues for accelerating scientific research. While models are increasingly capable of assisting with routine tasks, their ability to contribute to novel, expert-level mathematical discovery is less understood. We present a collection of case studies demonstrating how researchers have successfully collaborated with advanced AI models, specifically Google's Gemini-based models (in particular Gemini Deep Think and its advanced variants), to solve open problems, refute conjectures, and generate new proofs across diverse areas in theoretical computer science, as well as other areas such as economics, optimization, and physics. Based on these experiences, we extract common techniques for effective human-AI collaboration in theoretical research, such as iterative refinement, problem decomposition, and cross-disciplinary knowledge transfer. While the majority of our results stem from this interactive, conversational methodology, we also highlight specific instances that push beyond standard chat interfaces. These include deploying the model as a rigorous adversarial reviewer to detect subtle flaws in existing proofs, and embedding it within a "neuro-symbolic" loop that autonomously writes and executes code to verify complex derivations. Together, these examples highlight the potential of AI not just as a tool for automation, but as a versatile, genuine partner in the creative process of scientific discovery.
>
---
#### [new 042] Task--Specificity Score: Measuring How Much Instructions Really Matter for Supervision
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出任务特异性得分（TSS），用于衡量指令对输出预测的重要性，解决指令与输出关系不明确的问题。通过对比真实指令与替代指令，提升模型训练效果。**

- **链接: [https://arxiv.org/pdf/2602.03103v1](https://arxiv.org/pdf/2602.03103v1)**

> **作者:** Pritam Kadasi; Abhishek Upperwal; Mayank Singh
>
> **摘要:** Instruction tuning is now the default way to train and adapt large language models, but many instruction--input--output pairs are only weakly specified: for a given input, the same output can remain plausible under several alternative instructions. This raises a simple question: \emph{does the instruction uniquely determine the target output?} We propose the \textbf{Task--Specificity Score (TSS)} to quantify how much an instruction matters for predicting its output, by contrasting the true instruction against plausible alternatives for the same input. We further introduce \textbf{TSS++}, which uses hard alternatives and a small quality term to mitigate easy-negative effects. Across three instruction datasets (\textsc{Alpaca}, \textsc{Dolly-15k}, \textsc{NI-20}) and three open LLMs (Gemma, Llama, Qwen), we show that selecting task-specific examples improves downstream performance under tight token budgets and complements quality-based filters such as perplexity and IFD.
>
---
#### [new 043] Which course? Discourse! Teaching Discourse and Generation in the Era of LLMs
- **分类: cs.CL**

- **简介: 该论文属于教育任务，探讨如何设计跨学科课程。针对现有课程缺乏 discourse 与生成结合的问题，作者设计了新课程，整合理论与实践，促进探索性学习。**

- **链接: [https://arxiv.org/pdf/2602.02878v1](https://arxiv.org/pdf/2602.02878v1)**

> **作者:** Junyi Jessy Li; Yang Janet Liu; Kanishka Misra; Valentina Pyatkin; William Sheffield
>
> **备注:** accepted to the TeachNLP 2026 workshop (co-located with EACL 2026), camera-ready, 14 pages
>
> **摘要:** The field of NLP has undergone vast, continuous transformations over the past few years, sparking debates going beyond discipline boundaries. This begs important questions in education: how do we design courses that bridge sub-disciplines in this shifting landscape? This paper explores this question from the angle of discourse processing, an area with rich linguistic insights and computational models for the intentional, attentional, and coherence structure of language. Discourse is highly relevant for open-ended or long-form text generation, yet this connection is under-explored in existing undergraduate curricula. We present a new course, "Computational Discourse and Natural Language Generation". The course is collaboratively designed by a team with complementary expertise and was offered for the first time in Fall 2025 as an upper-level undergraduate course, cross-listed between Linguistics and Computer Science. Our philosophy is to deeply integrate the theoretical and empirical aspects, and create an exploratory mindset inside the classroom and in the assignments. This paper describes the course in detail and concludes with takeaways from an independent survey as well as our vision for future directions.
>
---
#### [new 044] No Shortcuts to Culture: Indonesian Multi-hop Question Answering for Complex Cultural Understanding
- **分类: cs.CL**

- **简介: 该论文属于文化理解任务，旨在解决现有问答数据集多为单跳问题、无法评估深层文化推理的问题。工作包括构建多跳的ID-MoCQA数据集，并设计验证流程以提升质量。**

- **链接: [https://arxiv.org/pdf/2602.03709v1](https://arxiv.org/pdf/2602.03709v1)**

> **作者:** Vynska Amalia Permadi; Xingwei Tan; Nafise Sadat Moosavi; Nikos Aletras
>
> **摘要:** Understanding culture requires reasoning across context, tradition, and implicit social knowledge, far beyond recalling isolated facts. Yet most culturally focused question answering (QA) benchmarks rely on single-hop questions, which may allow models to exploit shallow cues rather than demonstrate genuine cultural reasoning. In this work, we introduce ID-MoCQA, the first large-scale multi-hop QA dataset for assessing the cultural understanding of large language models (LLMs), grounded in Indonesian traditions and available in both English and Indonesian. We present a new framework that systematically transforms single-hop cultural questions into multi-hop reasoning chains spanning six clue types (e.g., commonsense, temporal, geographical). Our multi-stage validation pipeline, combining expert review and LLM-as-a-judge filtering, ensures high-quality question-answer pairs. Our evaluation across state-of-the-art models reveals substantial gaps in cultural reasoning, particularly in tasks requiring nuanced inference. ID-MoCQA provides a challenging and essential benchmark for advancing the cultural competency of LLMs.
>
---
#### [new 045] AmharicStoryQA: A Multicultural Story Question Answering Benchmark in Amharic
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的问答任务，旨在解决语言与文化区分的问题。工作包括构建AmharicStoryQA基准，展示语言模型在文化多样性下的表现差异。**

- **链接: [https://arxiv.org/pdf/2602.02774v1](https://arxiv.org/pdf/2602.02774v1)**

> **作者:** Israel Abebe Azime; Abenezer Kebede Angamo; Hana Mekonen Tamiru; Dagnachew Mekonnen Marilign; Philipp Slusallek; Seid Muhie Yimam; Dietrich Klakow
>
> **摘要:** With the growing emphasis on multilingual and cultural evaluation benchmarks for large language models, language and culture are often treated as synonymous, and performance is commonly used as a proxy for a models understanding of a given language. In this work, we argue that such evaluations overlook meaningful cultural variation that exists within a single language. We address this gap by focusing on narratives from different regions of Ethiopia and demonstrate that, despite shared linguistic characteristics, region-specific and domain-specific content substantially influences language evaluation outcomes. To this end, we introduce \textbf{\textit{AmharicStoryQA}}, a long-sequence story question answering benchmark grounded in culturally diverse narratives from Amharic-speaking regions. Using this benchmark, we reveal a significant narrative understanding gap in existing LLMs, highlight pronounced regional differences in evaluation results, and show that supervised fine-tuning yields uneven improvements across regions and evaluation settings. Our findings emphasize the need for culturally grounded benchmarks that go beyond language-level evaluation to more accurately assess and improve narrative understanding in low-resource languages.
>
---
#### [new 046] CATNIP: LLM Unlearning via Calibrated and Tokenized Negative Preference Alignment
- **分类: cs.CL**

- **简介: 该论文属于LLM知识删除任务，旨在解决模型遗忘过程中知识丢失和数据依赖问题。提出CATNIP方法，通过校准模型置信度实现精准遗忘控制。**

- **链接: [https://arxiv.org/pdf/2602.02824v1](https://arxiv.org/pdf/2602.02824v1)**

> **作者:** Zhengbang Yang; Yisheng Zhong; Junyuan Hong; Zhuangdi Zhu
>
> **摘要:** Pretrained knowledge memorized in LLMs raises critical concerns over safety and privacy, which has motivated LLM Unlearning as a technique for selectively removing the influences of undesirable knowledge. Existing approaches, rooted in Gradient Ascent (GA), often degrade general domain knowledge while relying on retention data or curated contrastive pairs, which can be either impractical or data and computationally prohibitive. Negative Preference Alignment has been explored for unlearning to tackle the limitations of GA, which, however, remains confined by its choice of reference model and shows undermined performance in realistic data settings. These limitations raise two key questions: i) Can we achieve effective unlearning that quantifies model confidence in undesirable knowledge and uses it to calibrate gradient updates more precisely, thus reducing catastrophic forgetting? ii) Can we make unlearning robust to data scarcity and length variation? We answer both questions affirmatively with CATNIP (Calibrated and Tokenized Negative Preference Alignment), a principled method that rescales unlearning effects in proportion to the model's token-level confidence, thus ensuring fine-grained control over forgetting. Extensive evaluations on MUSE and WMDP benchmarks demonstrated that our work enables effective unlearning without requiring retention data or contrastive unlearning response pairs, with stronger knowledge forgetting and preservation tradeoffs than state-of-the-art methods.
>
---
#### [new 047] Equal Access, Unequal Interaction: A Counterfactual Audit of LLM Fairness
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于公平性审计任务，旨在检测LLM在平等访问后交互质量的差异。通过对比GPT-4和LLaMA-3.1-70B在不同身份下的语气、不确定性等表现，揭示模型间的公平性问题。**

- **链接: [https://arxiv.org/pdf/2602.02932v1](https://arxiv.org/pdf/2602.02932v1)**

> **作者:** Alireza Amiri-Margavi; Arshia Gharagozlou; Amin Gholami Davodi; Seyed Pouyan Mousavi Davoudi; Hamidreza Hasani Balyani
>
> **备注:** 13 pages, 1 figure
>
> **摘要:** Prior work on fairness in large language models (LLMs) has primarily focused on access-level behaviors such as refusals and safety filtering. However, equitable access does not ensure equitable interaction quality once a response is provided. In this paper, we conduct a controlled fairness audit examining how LLMs differ in tone, uncertainty, and linguistic framing across demographic identities after access is granted. Using a counterfactual prompt design, we evaluate GPT-4 and LLaMA-3.1-70B on career advice tasks while varying identity attributes along age, gender, and nationality. We assess access fairness through refusal analysis and measure interaction quality using automated linguistic metrics, including sentiment, politeness, and hedging. Identity-conditioned differences are evaluated using paired statistical tests. Both models exhibit zero refusal rates across all identities, indicating uniform access. Nevertheless, we observe systematic, model-specific disparities in interaction quality: GPT-4 expresses significantly higher hedging toward younger male users, while LLaMA exhibits broader sentiment variation across identity groups. These results show that fairness disparities can persist at the interaction level even when access is equal, motivating evaluation beyond refusal-based audits.
>
---
#### [new 048] Verified Critical Step Optimization for LLM Agents
- **分类: cs.CL**

- **简介: 该论文属于大语言模型代理的后训练优化任务，旨在解决传统方法在步骤奖励估计中的噪声与计算成本问题。提出CSO方法，通过验证关键步骤提升训练效率与效果。**

- **链接: [https://arxiv.org/pdf/2602.03412v1](https://arxiv.org/pdf/2602.03412v1)**

> **作者:** Mukai Li; Qingcheng Zeng; Tianqing Fang; Zhenwen Liang; Linfeng Song; Qi Liu; Haitao Mi; Dong Yu
>
> **备注:** Working in progress
>
> **摘要:** As large language model agents tackle increasingly complex long-horizon tasks, effective post-training becomes critical. Prior work faces fundamental challenges: outcome-only rewards fail to precisely attribute credit to intermediate steps, estimated step-level rewards introduce systematic noise, and Monte Carlo sampling approaches for step reward estimation incur prohibitive computational cost. Inspired by findings that only a small fraction of high-entropy tokens drive effective RL for reasoning, we propose Critical Step Optimization (CSO), which focuses preference learning on verified critical steps, decision points where alternate actions demonstrably flip task outcomes from failure to success. Crucially, our method starts from failed policy trajectories rather than expert demonstrations, directly targeting the policy model's weaknesses. We use a process reward model (PRM) to identify candidate critical steps, leverage expert models to propose high-quality alternatives, then continue execution from these alternatives using the policy model itself until task completion. Only alternatives that the policy successfully executes to correct outcomes are verified and used as DPO training data, ensuring both quality and policy reachability. This yields fine-grained, verifiable supervision at critical decisions while avoiding trajectory-level coarseness and step-level noise. Experiments on GAIA-Text-103 and XBench-DeepSearch show that CSO achieves 37% and 26% relative improvement over the SFT baseline and substantially outperforms other post-training methods, while requiring supervision at only 16% of trajectory steps. This demonstrates the effectiveness of selective verification-based learning for agent post-training.
>
---
#### [new 049] BIRDTurk: Adaptation of the BIRD Text-to-SQL Dataset to Turkish
- **分类: cs.CL; cs.AI; cs.DB**

- **简介: 该论文属于跨语言Text-to-SQL任务，旨在解决低资源语言中的SQL生成问题。通过构建BIRDTurk数据集，评估不同方法在土耳其语中的表现。**

- **链接: [https://arxiv.org/pdf/2602.03633v1](https://arxiv.org/pdf/2602.03633v1)**

> **作者:** Burak Aktaş; Mehmet Can Baytekin; Süha Kağan Köse; Ömer İlbilgi; Elif Özge Yılmaz; Çağrı Toraman; Bilge Kaan Görür
>
> **备注:** Accepted by EACL 2026 SIGTURK
>
> **摘要:** Text-to-SQL systems have achieved strong performance on English benchmarks, yet their behavior in morphologically rich, low-resource languages remains largely unexplored. We introduce BIRDTurk, the first Turkish adaptation of the BIRD benchmark, constructed through a controlled translation pipeline that adapts schema identifiers to Turkish while strictly preserving the logical structure and execution semantics of SQL queries and databases. Translation quality is validated on a sample size determined by the Central Limit Theorem to ensure 95% confidence, achieving 98.15% accuracy on human-evaluated samples. Using BIRDTurk, we evaluate inference-based prompting, agentic multi-stage reasoning, and supervised fine-tuning. Our results reveal that Turkish introduces consistent performance degradation, driven by both structural linguistic divergence and underrepresentation in LLM pretraining, while agentic reasoning demonstrates stronger cross-lingual robustness. Supervised fine-tuning remains challenging for standard multilingual baselines but scales effectively with modern instruction-tuned models. BIRDTurk provides a controlled testbed for cross-lingual Text-to-SQL evaluation under realistic database conditions. We release the training and development splits to support future research.
>
---
#### [new 050] Neural Attention Search Linear: Towards Adaptive Token-Level Hybrid Attention Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在解决Transformer模型在长序列中的计算效率问题。通过引入混合注意力机制，提升模型效率与表达能力。**

- **链接: [https://arxiv.org/pdf/2602.03681v1](https://arxiv.org/pdf/2602.03681v1)**

> **作者:** Difan Deng; Andreas Bentzen Winje; Lukas Fehring; Marius Lindauer
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** The quadratic computational complexity of softmax transformers has become a bottleneck in long-context scenarios. In contrast, linear attention model families provide a promising direction towards a more efficient sequential model. These linear attention models compress past KV values into a single hidden state, thereby efficiently reducing complexity during both training and inference. However, their expressivity remains limited by the size of their hidden state. Previous work proposed interleaving softmax and linear attention layers to reduce computational complexity while preserving expressivity. Nevertheless, the efficiency of these models remains bottlenecked by their softmax attention layers. In this paper, we propose Neural Attention Search Linear (NAtS-L), a framework that applies both linear attention and softmax attention operations within the same layer on different tokens. NAtS-L automatically determines whether a token can be handled by a linear attention model, i.e., tokens that have only short-term impact and can be encoded into fixed-size hidden states, or require softmax attention, i.e., tokens that contain information related to long-term retrieval and need to be preserved for future queries. By searching for optimal Gated DeltaNet and softmax attention combinations across tokens, we show that NAtS-L provides a strong yet efficient token-level hybrid architecture.
>
---
#### [new 051] Token Sparse Attention: Efficient Long-Context Inference with Interleaved Token Selection
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于自然语言处理任务，解决长文本推理中的注意力计算效率问题。提出Token Sparse Attention机制，动态压缩并恢复token信息，提升推理速度与精度平衡。**

- **链接: [https://arxiv.org/pdf/2602.03216v1](https://arxiv.org/pdf/2602.03216v1)**

> **作者:** Dongwon Jo; Beomseok Kang; Jiwon Song; Jae-Joon Kim
>
> **摘要:** The quadratic complexity of attention remains the central bottleneck in long-context inference for large language models. Prior acceleration methods either sparsify the attention map with structured patterns or permanently evict tokens at specific layers, which can retain irrelevant tokens or rely on irreversible early decisions despite the layer-/head-wise dynamics of token importance. In this paper, we propose Token Sparse Attention, a lightweight and dynamic token-level sparsification mechanism that compresses per-head $Q$, $K$, $V$ to a reduced token set during attention and then decompresses the output back to the original sequence, enabling token information to be reconsidered in subsequent layers. Furthermore, Token Sparse Attention exposes a new design point at the intersection of token selection and sparse attention. Our approach is fully compatible with dense attention implementations, including Flash Attention, and can be seamlessly composed with existing sparse attention kernels. Experimental results show that Token Sparse Attention consistently improves accuracy-latency trade-off, achieving up to $\times$3.23 attention speedup at 128K context with less than 1% accuracy degradation. These results demonstrate that dynamic and interleaved token-level sparsification is a complementary and effective strategy for scalable long-context inference.
>
---
#### [new 052] Accurate Failure Prediction in Agents Does Not Imply Effective Failure Prevention
- **分类: cs.CL; cs.LG**

- **简介: 该论文属于AI可靠性任务，研究LLM批评者在部署中的效果。工作表明高准确率的批评者可能引发性能下降，提出预测试方法判断是否干预。**

- **链接: [https://arxiv.org/pdf/2602.03338v1](https://arxiv.org/pdf/2602.03338v1)**

> **作者:** Rakshith Vasudev; Melisa Russak; Dan Bikel; Waseem Alshikh
>
> **摘要:** Proactive interventions by LLM critic models are often assumed to improve reliability, yet their effects at deployment time are poorly understood. We show that a binary LLM critic with strong offline accuracy (AUROC 0.94) can nevertheless cause severe performance degradation, inducing a 26 percentage point (pp) collapse on one model while affecting another by near zero pp. This variability demonstrates that LLM critic accuracy alone is insufficient to determine whether intervention is safe. We identify a disruption-recovery tradeoff: interventions may recover failing trajectories but also disrupt trajectories that would have succeeded. Based on this insight, we propose a pre-deployment test that uses a small pilot of 50 tasks to estimate whether intervention is likely to help or harm, without requiring full deployment. Across benchmarks, the test correctly anticipates outcomes: intervention degrades performance on high-success tasks (0 to -26 pp), while yielding a modest improvement on the high-failure ALFWorld benchmark (+2.8 pp, p=0.014). The primary value of our framework is therefore identifying when not to intervene, preventing severe regressions before deployment.
>
---
#### [new 053] Can Large Language Models Generalize Procedures Across Representations?
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究LLM在不同表示间（代码、图、自然语言）的泛化能力，解决跨表示任务的性能问题，通过两阶段数据课程提升模型表现。**

- **链接: [https://arxiv.org/pdf/2602.03542v1](https://arxiv.org/pdf/2602.03542v1)**

> **作者:** Fangru Lin; Valentin Hofmann; Xingchen Wan; Weixing Wang; Zifeng Ding; Anthony G. Cohn; Janet B. Pierrehumbert
>
> **摘要:** Large language models (LLMs) are trained and tested extensively on symbolic representations such as code and graphs, yet real-world user tasks are often specified in natural language. To what extent can LLMs generalize across these representations? Here, we approach this question by studying isomorphic tasks involving procedures represented in code, graphs, and natural language (e.g., scheduling steps in planning). We find that training LLMs with popular post-training methods on graphs or code data alone does not reliably generalize to corresponding natural language tasks, while training solely on natural language can lead to inefficient performance gains. To address this gap, we propose a two-stage data curriculum that first trains on symbolic, then natural language data. The curriculum substantially improves model performance across model families and tasks. Remarkably, a 1.5B Qwen model trained by our method can closely match zero-shot GPT-4o in naturalistic planning. Finally, our analysis suggests that successful cross-representation generalization can be interpreted as a form of generative analogy, which our curriculum effectively encourages.
>
---
#### [new 054] FASA: Frequency-aware Sparse Attention
- **分类: cs.CL**

- **简介: 该论文提出FASA框架，解决长输入下LLM内存瓶颈问题。通过动态预测token重要性，提升注意力效率，显著降低内存和计算成本。**

- **链接: [https://arxiv.org/pdf/2602.03152v1](https://arxiv.org/pdf/2602.03152v1)**

> **作者:** Yifei Wang; Yueqi Wang; Zhenrui Yue; Huimin Zeng; Yong Wang; Ismini Lourentzou; Zhengzhong Tu; Xiangxiang Chu; Julian McAuley
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** The deployment of Large Language Models (LLMs) faces a critical bottleneck when handling lengthy inputs: the prohibitive memory footprint of the Key Value (KV) cache. To address this bottleneck, the token pruning paradigm leverages attention sparsity to selectively retain a small, critical subset of tokens. However, existing approaches fall short, with static methods risking irreversible information loss and dynamic strategies employing heuristics that insufficiently capture the query-dependent nature of token importance. We propose FASA, a novel framework that achieves query-aware token eviction by dynamically predicting token importance. FASA stems from a novel insight into RoPE: the discovery of functional sparsity at the frequency-chunk (FC) level. Our key finding is that a small, identifiable subset of "dominant" FCs consistently exhibits high contextual agreement with the full attention head. This provides a robust and computationally free proxy for identifying salient tokens. %making them a powerful and efficient proxy for token importance. Building on this insight, FASA first identifies a critical set of tokens using dominant FCs, and then performs focused attention computation solely on this pruned subset. % Since accessing only a small fraction of the KV cache, FASA drastically lowers memory bandwidth requirements and computational cost. Across a spectrum of long-context tasks, from sequence modeling to complex CoT reasoning, FASA consistently outperforms all token-eviction baselines and achieves near-oracle accuracy, demonstrating remarkable robustness even under constraint budgets. Notably, on LongBench-V1, FASA reaches nearly 100\% of full-KV performance when only keeping 256 tokens, and achieves 2.56$\times$ speedup using just 18.9\% of the cache on AIME24.
>
---
#### [new 055] Graph-Augmented Reasoning with Large Language Models for Tobacco Pest and Disease Management
- **分类: cs.CL**

- **简介: 该论文属于烟草病虫害管理任务，旨在解决传统方法在多跳推理和比较问题上的不足。通过构建领域知识图谱并融合到大语言模型中，提升推荐的准确性和一致性。**

- **链接: [https://arxiv.org/pdf/2602.02635v1](https://arxiv.org/pdf/2602.02635v1)**

> **作者:** Siyu Li; Chenwei Song; Qi Zhou; Wan Zhou; Xinyi Liu
>
> **摘要:** This paper proposes a graph-augmented reasoning framework for tobacco pest and disease management that integrates structured domain knowledge into large language models. Building on GraphRAG, we construct a domain-specific knowledge graph and retrieve query-relevant subgraphs to provide relational evidence during answer generation. The framework adopts ChatGLM as the Transformer backbone with LoRA-based parameter-efficient fine-tuning, and employs a graph neural network to learn node representations that capture symptom-disease-treatment dependencies. By explicitly modeling diseases, symptoms, pesticides, and control measures as linked entities, the system supports evidence-aware retrieval beyond surface-level text similarity. Retrieved graph evidence is incorporated into the LLM input to guide generation toward domain-consistent recommendations and to mitigate hallucinated or inappropriate treatments. Experimental results show consistent improvements over text-only baselines, with the largest gains observed on multi-hop and comparative reasoning questions that require chaining multiple relations.
>
---
#### [new 056] Monotonicity as an Architectural Bias for Robust Language Models
- **分类: cs.CL; cs.AI; cs.CR; cs.LG**

- **简介: 该论文属于自然语言处理任务，旨在提升语言模型的鲁棒性。针对对抗样本导致的脆弱性问题，通过引入单调性架构偏差，增强模型稳定性，同时保持性能。**

- **链接: [https://arxiv.org/pdf/2602.02686v1](https://arxiv.org/pdf/2602.02686v1)**

> **作者:** Patrick Cooper; Alireza Nadali; Ashutosh Trivedi; Alvaro Velasquez
>
> **备注:** 12 pages, 1 figure
>
> **摘要:** Large language models (LLMs) are known to exhibit brittle behavior under adversarial prompts and jailbreak attacks, even after extensive alignment and fine-tuning. This fragility reflects a broader challenge of modern neural language models: small, carefully structured perturbations in high-dimensional input spaces can induce large and unpredictable changes in internal semantic representations and output. We investigate monotonicity as an architectural inductive bias for improving the robustness of Transformer-based language models. Monotonicity constrains semantic transformations so that strengthening information, evidence, or constraints cannot lead to regressions in the corresponding internal representations. Such order-preserving behavior has long been exploited in control and safety-critical systems to simplify reasoning and improve robustness, but has traditionally been viewed as incompatible with the expressivity required by neural language models. We show that this trade-off is not inherent. By enforcing monotonicity selectively in the feed-forward sublayers of sequence-to-sequence Transformers -- while leaving attention mechanisms unconstrained -- we obtain monotone language models that preserve the performance of their pretrained counterparts. This architectural separation allows negation, contradiction, and contextual interactions to be introduced explicitly through attention, while ensuring that subsequent semantic refinement is order-preserving. Empirically, monotonicity substantially improves robustness: adversarial attack success rates drop from approximately 69% to 19%, while standard summarization performance degrades only marginally.
>
---
#### [new 057] Pursuing Best Industrial Practices for Retrieval-Augmented Generation in the Medical Domain
- **分类: cs.CL**

- **简介: 该论文属于医疗领域RAG系统研究，旨在明确最佳实践。分析RAG组件，提出优化方案，并通过实验揭示性能与效率的平衡策略。**

- **链接: [https://arxiv.org/pdf/2602.03368v1](https://arxiv.org/pdf/2602.03368v1)**

> **作者:** Wei Zhu
>
> **摘要:** While retrieval augmented generation (RAG) has been swiftly adopted in industrial applications based on large language models (LLMs), there is no consensus on what are the best practices for building a RAG system in terms of what are the components, how to organize these components and how to implement each component for the industrial applications, especially in the medical domain. In this work, we first carefully analyze each component of the RAG system and propose practical alternatives for each component. Then, we conduct systematic evaluations on three types of tasks, revealing the best practices for improving the RAG system and how LLM-based RAG systems make trade-offs between performance and efficiency.
>
---
#### [new 058] Towards Distillation-Resistant Large Language Models: An Information-Theoretic Perspective
- **分类: cs.CL**

- **简介: 该论文属于模型保护任务，旨在解决LLM知识蒸馏攻击问题。通过信息论方法减少教师模型输出中的可蒸馏信息，提升模型抗攻击能力。**

- **链接: [https://arxiv.org/pdf/2602.03396v1](https://arxiv.org/pdf/2602.03396v1)**

> **作者:** Hao Fang; Tianyi Zhang; Tianqu Zhuang; Jiawei Kong; Kuofeng Gao; Bin Chen; Leqi Liang; Shu-Tao Xia; Ke Xu
>
> **摘要:** Proprietary large language models (LLMs) embody substantial economic value and are generally exposed only as black-box APIs, yet adversaries can still exploit their outputs to extract knowledge via distillation. Existing defenses focus exclusively on text-based distillation, leaving the important logit-based distillation largely unexplored. In this work, we analyze this problem and present an effective solution from an information-theoretic perspective. We characterize distillation-relevant information in teacher outputs using the conditional mutual information (CMI) between teacher logits and input queries conditioned on ground-truth labels. This quantity captures contextual information beneficial for model extraction, motivating us to defend distillation via CMI minimization. Guided by our theoretical analysis, we propose learning a transformation matrix that purifies the original outputs to enhance distillation resistance. We further derive a CMI-inspired anti-distillation objective to optimize this transformation, which effectively removes distillation-relevant information while preserving output utility. Extensive experiments across multiple LLMs and strong distillation algorithms demonstrate that the proposed method significantly degrades distillation performance while preserving task accuracy, effectively protecting models' intellectual property.
>
---
#### [new 059] AERO: Autonomous Evolutionary Reasoning Optimization via Endogenous Dual-Loop Feedback
- **分类: cs.CL**

- **简介: 该论文提出AERO框架，解决LLM依赖标注数据和外部验证的问题，通过自监督机制实现自主推理优化。属于自然语言处理任务。**

- **链接: [https://arxiv.org/pdf/2602.03084v1](https://arxiv.org/pdf/2602.03084v1)**

> **作者:** Zhitao Gao; Jie Ma; Xuhong Li; Pengyu Li; Ning Qu; Yaqiang Wu; Hui Liu; Jun Liu
>
> **摘要:** Large Language Models (LLMs) have achieved significant success in complex reasoning but remain bottlenecked by reliance on expert-annotated data and external verifiers. While existing self-evolution paradigms aim to bypass these constraints, they often fail to identify the optimal learning zone and risk reinforcing collective hallucinations and incorrect priors through flawed internal feedback. To address these challenges, we propose \underline{A}utonomous \underline{E}volutionary \underline{R}easoning \underline{O}ptimization (AERO), an unsupervised framework that achieves autonomous reasoning evolution by internalizing self-questioning, answering, and criticism within a synergistic dual-loop system. Inspired by the \textit{Zone of Proximal Development (ZPD)} theory, AERO utilizes entropy-based positioning to target the ``solvability gap'' and employs Independent Counterfactual Correction for robust verification. Furthermore, we introduce a Staggered Training Strategy to synchronize capability growth across functional roles and prevent curriculum collapse. Extensive evaluations across nine benchmarks spanning three domains demonstrate that AERO achieves average performance improvements of 4.57\% on Qwen3-4B-Base and 5.10\% on Qwen3-8B-Base, outperforming competitive baselines. Code is available at https://github.com/mira-ai-lab/AERO.
>
---
#### [new 060] Beyond Tokens: Semantic-Aware Speculative Decoding for Efficient Inference by Probing Internal States
- **分类: cs.CL; cs.PF**

- **简介: 该论文属于自然语言处理任务，解决LLM推理延迟高的问题。提出SemanticSpec框架，通过语义验证替代传统token级验证，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2602.03708v1](https://arxiv.org/pdf/2602.03708v1)**

> **作者:** Ximing Dong; Shaowei Wang; Dayi Lin; Boyuan Chen; Ahmed E. Hassan
>
> **摘要:** Large Language Models (LLMs) achieve strong performance across many tasks but suffer from high inference latency due to autoregressive decoding. The issue is exacerbated in Large Reasoning Models (LRMs), which generate lengthy chains of thought. While speculative decoding accelerates inference by drafting and verifying multiple tokens in parallel, existing methods operate at the token level and ignore semantic equivalence (i.e., different token sequences expressing the same meaning), leading to inefficient rejections. We propose SemanticSpec, a semantic-aware speculative decoding framework that verifies entire semantic sequences instead of tokens. SemanticSpec introduces a semantic probability estimation mechanism that probes the model's internal hidden states to assess the likelihood of generating sequences with specific meanings.Experiments on four benchmarks show that SemanticSpec achieves up to 2.7x speedup on DeepSeekR1-32B and 2.1x on QwQ-32B, consistently outperforming token-level and sequence-level baselines in both efficiency and effectiveness.
>
---
#### [new 061] $V_0$: A Generalist Value Model for Any Policy at State Zero
- **分类: cs.CL; cs.AI; cs.LG**

- **链接: [https://arxiv.org/pdf/2602.03584v1](https://arxiv.org/pdf/2602.03584v1)**

> **作者:** Yi-Kai Zhang; Zhiyuan Yao; Hongyan Hao; Yueqing Sun; Qi Gu; Hui Su; Xunliang Cai; De-Chuan Zhan; Han-Jia Ye
>
> **摘要:** Policy gradient methods rely on a baseline to measure the relative advantage of an action, ensuring the model reinforces behaviors that outperform its current average capability. In the training of Large Language Models (LLMs) using Actor-Critic methods (e.g., PPO), this baseline is typically estimated by a Value Model (Critic) often as large as the policy model itself. However, as the policy continuously evolves, the value model requires expensive, synchronous incremental training to accurately track the shifting capabilities of the policy. To avoid this overhead, Group Relative Policy Optimization (GRPO) eliminates the coupled value model by using the average reward of a group of rollouts as the baseline; yet, this approach necessitates extensive sampling to maintain estimation stability. In this paper, we propose $V_0$, a Generalist Value Model capable of estimating the expected performance of any model on unseen prompts without requiring parameter updates. We reframe value estimation by treating the policy's dynamic capability as an explicit context input; specifically, we leverage a history of instruction-performance pairs to dynamically profile the model, departing from the traditional paradigm that relies on parameter fitting to perceive capability shifts. Focusing on value estimation at State Zero (i.e., the initial prompt, hence $V_0$), our model serves as a critical resource scheduler. During GRPO training, $V_0$ predicts success rates prior to rollout, allowing for efficient sampling budget allocation; during deployment, it functions as a router, dispatching instructions to the most cost-effective and suitable model. Empirical results demonstrate that $V_0$ significantly outperforms heuristic budget allocation and achieves a Pareto-optimal trade-off between performance and cost in LLM routing tasks.
>
---
#### [new 062] The Hypocrisy Gap: Quantifying Divergence Between Internal Belief and Chain-of-Thought Explanation via Sparse Autoencoders
- **分类: cs.CL**

- **简介: 该论文属于模型行为分析任务，旨在检测大语言模型的不忠行为。通过引入Hypocrisy Gap指标，利用稀疏自编码器量化模型内部信念与输出之间的差异，有效识别模型的虚伪行为。**

- **链接: [https://arxiv.org/pdf/2602.02496v1](https://arxiv.org/pdf/2602.02496v1)**

> **作者:** Shikhar Shiromani; Archie Chaudhury; Sri Pranav Kunda
>
> **备注:** 8 pages, 1 figure
>
> **摘要:** Large Language Models (LLMs) frequently exhibit unfaithful behavior, producing a final answer that differs significantly from their internal chain of thought (CoT) reasoning in order to appease the user they are conversing with. In order to better detect this behavior, we introduce the Hypocrisy Gap, a mechanistic metric utilizing Sparse Autoencoders (SAEs) to quantify the divergence between a model's internal reasoning and its final generation. By mathematically comparing an internal truth belief, derived via sparse linear probes, to the final generated trajectory in latent space, we quantify and detect a model's tendency to engage in unfaithful behavior. Experiments on Gemma, Llama, and Qwen models using Anthropic's Sycophancy benchmark show that our method achieves an AUROC of 0.55-0.73 for detecting sycophantic runs and 0.55-0.74 for hypocritical cases where the model internally "knows" the user is wrong, consistently outperforming a decision-aligned log-probability baseline (0.41-0.50 AUROC).
>
---
#### [new 063] Rethinking the Reranker: Boundary-Aware Evidence Selection for Robust Retrieval-Augmented Generation
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于知识密集型问答任务，解决RAG系统在检索噪声下的脆弱性问题。提出BAR-RAG，通过边界感知证据选择提升生成鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.03689v1](https://arxiv.org/pdf/2602.03689v1)**

> **作者:** Jiashuo Sun; Pengcheng Jiang; Saizhuo Wang; Jiajun Fan; Heng Wang; Siru Ouyang; Ming Zhong; Yizhu Jiao; Chengsong Huang; Xueqiang Xu; Pengrui Han; Peiran Li; Jiaxin Huang; Ge Liu; Heng Ji; Jiawei Han
>
> **备注:** 19 pages, 8 tables, 5 figures
>
> **摘要:** Retrieval-Augmented Generation (RAG) systems remain brittle under realistic retrieval noise, even when the required evidence appears in the top-K results. A key reason is that retrievers and rerankers optimize solely for relevance, often selecting either trivial, answer-revealing passages or evidence that lacks the critical information required to answer the question, without considering whether the evidence is suitable for the generator. We propose BAR-RAG, which reframes the reranker as a boundary-aware evidence selector that targets the generator's Goldilocks Zone -- evidence that is neither trivially easy nor fundamentally unanswerable for the generator, but is challenging yet sufficient for inference and thus provides the strongest learning signal. BAR-RAG trains the selector with reinforcement learning using generator feedback, and adopts a two-stage pipeline that fine-tunes the generator under the induced evidence distribution to mitigate the distribution mismatch between training and inference. Experiments on knowledge-intensive question answering benchmarks show that BAR-RAG consistently improves end-to-end performance under noisy retrieval, achieving an average gain of 10.3 percent over strong RAG and reranking baselines while substantially improving robustness. Code is publicly avaliable at https://github.com/GasolSun36/BAR-RAG.
>
---
#### [new 064] ChemPro: A Progressive Chemistry Benchmark for Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出ChemPro，一个用于评估大语言模型化学能力的基准，解决LLMs在科学推理上的局限性问题，涵盖多种题型和难度层次。**

- **链接: [https://arxiv.org/pdf/2602.03108v1](https://arxiv.org/pdf/2602.03108v1)**

> **作者:** Aaditya Baranwal; Shruti Vyas
>
> **摘要:** We introduce ChemPro, a progressive benchmark with 4100 natural language question-answer pairs in Chemistry, across 4 coherent sections of difficulty designed to assess the proficiency of Large Language Models (LLMs) in a broad spectrum of general chemistry topics. We include Multiple Choice Questions and Numerical Questions spread across fine-grained information recall, long-horizon reasoning, multi-concept questions, problem-solving with nuanced articulation, and straightforward questions in a balanced ratio, effectively covering Bio-Chemistry, Inorganic-Chemistry, Organic-Chemistry and Physical-Chemistry. ChemPro is carefully designed analogous to a student's academic evaluation for basic to high-school chemistry. A gradual increase in the question difficulty rigorously tests the ability of LLMs to progress from solving basic problems to solving more sophisticated challenges. We evaluate 45+7 state-of-the-art LLMs, spanning both open-source and proprietary variants, and our analysis reveals that while LLMs perform well on basic chemistry questions, their accuracy declines with different types and levels of complexity. These findings highlight the critical limitations of LLMs in general scientific reasoning and understanding and point towards understudied dimensions of difficulty, emphasizing the need for more robust methodologies to improve LLMs.
>
---
#### [new 065] Efficient Algorithms for Partial Constraint Satisfaction Problems over Control-flow Graphs
- **分类: cs.CL; cs.PL**

- **简介: 该论文研究控制流图上的部分约束满足问题（PCSP），旨在高效解决编译优化任务中的约束问题。通过SPL分解，提出一种线性时间算法，提升运行效率。**

- **链接: [https://arxiv.org/pdf/2602.03588v1](https://arxiv.org/pdf/2602.03588v1)**

> **作者:** Xuran Cai; Amir Goharshady
>
> **备注:** Already accepted by SETTA'25. https://www.setta2025.uk/accepted-papers. arXiv admin note: substantial text overlap with arXiv:2507.16660
>
> **摘要:** In this work, we focus on the Partial Constraint Satisfaction Problem (PCSP) over control-flow graphs (CFGs) of programs. PCSP serves as a generalization of the well-known Constraint Satisfaction Problem (CSP). In the CSP framework, we define a set of variables, a set of constraints, and a finite domain $D$ that encompasses all possible values for each variable. The objective is to assign a value to each variable in such a way that all constraints are satisfied. In the graph variant of CSP, an underlying graph is considered and we have one variable corresponding to each vertex of the graph and one or several constraints corresponding to each edge. In PCSPs, we allow for certain constraints to be violated at a specified cost, aiming to find a solution that minimizes the total cost. Numerous classical compiler optimization tasks can be framed as PCSPs over control-flow graphs. Examples include Register Allocation, Lifetime-optimal Speculative Partial Redundancy Elimination (LOSPRE), and Optimal Placement of Bank Selection Instructions. On the other hand, it is well-known that control-flow graphs of structured programs are sparse and decomposable in a variety of ways. In this work, we rely on the Series-Parallel-Loop (SPL) decompositions as introduced by~\cite{RegisterAllocation}. Our main contribution is a general algorithm for PCSPs over SPL graphs with a time complexity of \(O(|G| \cdot |D|^6)\), where \(|G|\) represents the size of the control-flow graph. Note that for any fixed domain $D,$ this yields a linear-time solution. Our algorithm can be seen as a generalization and unification of previous SPL-based approaches for register allocation and LOSPRE. In addition, we provide experimental results over another classical PCSP task, i.e. Optimal Bank Selection, achieving runtimes four times better than the previous state of the art.
>
---
#### [new 066] Preferences for Idiomatic Language are Acquired Slowly -- and Forgotten Quickly: A Case Study on Swedish
- **分类: cs.CL**

- **简介: 该论文研究语言模型在瑞典语中对习语与语法正确表达的偏好变化，探讨其在预训练和迁移过程中的发展与遗忘。任务为语言模型的习语能力评估与分析。**

- **链接: [https://arxiv.org/pdf/2602.03484v1](https://arxiv.org/pdf/2602.03484v1)**

> **作者:** Jenny Kunz
>
> **备注:** Accepted to TACL. Note that the arXiv version is a pre-MIT Press publication version
>
> **摘要:** In this study, we investigate how language models develop preferences for \textit{idiomatic} as compared to \textit{linguistically acceptable} Swedish, both during pretraining and when adapting a model from English to Swedish. To do so, we train models on Swedish from scratch and by fine-tuning English-pretrained models, probing their preferences at various checkpoints using minimal pairs that differ in linguistic acceptability or idiomaticity. For linguistic acceptability, we adapt existing benchmarks into a minimal-pair format. To assess idiomaticity, we introduce two novel datasets: one contrasting conventionalized idioms with plausible variants, and another contrasting idiomatic Swedish with Translationese. Our findings suggest that idiomatic competence emerges more slowly than other linguistic abilities, including grammatical and lexical correctness. While longer training yields diminishing returns for most tasks, idiom-related performance continues to improve, particularly in the largest model tested (8B). However, instruction tuning on data machine-translated from English -- the common approach for languages with little or no native instruction data -- causes models to rapidly lose their preference for idiomatic language.
>
---
#### [new 067] WideSeek: Advancing Wide Research via Multi-Agent Scaling
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文属于信息检索任务，旨在解决Wide Research缺乏基准和优化方法的问题。提出WideSeekBench基准和多智能体架构，通过扩展代理数量提升搜索广度。**

- **链接: [https://arxiv.org/pdf/2602.02636v1](https://arxiv.org/pdf/2602.02636v1)**

> **作者:** Ziyang Huang; Haolin Ren; Xiaowei Yuan; Jiawei Wang; Zhongtao Jiang; Kun Xu; Shizhu He; Jun Zhao; Kang Liu
>
> **摘要:** Search intelligence is evolving from Deep Research to Wide Research, a paradigm essential for retrieving and synthesizing comprehensive information under complex constraints in parallel. However, progress in this field is impeded by the lack of dedicated benchmarks and optimization methodologies for search breadth. To address these challenges, we take a deep dive into Wide Research from two perspectives: Data Pipeline and Agent Optimization. First, we produce WideSeekBench, a General Broad Information Seeking (GBIS) benchmark constructed via a rigorous multi-phase data pipeline to ensure diversity across the target information volume, logical constraints, and domains. Second, we introduce WideSeek, a dynamic hierarchical multi-agent architecture that can autonomously fork parallel sub-agents based on task requirements. Furthermore, we design a unified training framework that linearizes multi-agent trajectories and optimizes the system using end-to-end RL. Experimental results demonstrate the effectiveness of WideSeek and multi-agent RL, highlighting that scaling the number of agents is a promising direction for advancing the Wide Research paradigm.
>
---
#### [new 068] Controlling Output Rankings in Generative Engines for LLM-based Search
- **分类: cs.CL; cs.AI; cs.IR**

- **简介: 该论文提出CORE方法，用于控制基于LLM的搜索引擎输出排名，解决小企业和创作者可见性不足的问题。通过优化内容提升目标产品排名。**

- **链接: [https://arxiv.org/pdf/2602.03608v1](https://arxiv.org/pdf/2602.03608v1)**

> **作者:** Haibo Jin; Ruoxi Chen; Peiyan Zhang; Yifeng Luo; Huimin Zeng; Man Luo; Haohan Wang
>
> **备注:** 23 pages
>
> **摘要:** The way customers search for and choose products is changing with the rise of large language models (LLMs). LLM-based search, or generative engines, provides direct product recommendations to users, rather than traditional online search results that require users to explore options themselves. However, these recommendations are strongly influenced by the initial retrieval order of LLMs, which disadvantages small businesses and independent creators by limiting their visibility. In this work, we propose CORE, an optimization method that \textbf{C}ontrols \textbf{O}utput \textbf{R}ankings in g\textbf{E}nerative Engines for LLM-based search. Since the LLM's interactions with the search engine are black-box, CORE targets the content returned by search engines as the primary means of influencing output rankings. Specifically, CORE optimizes retrieved content by appending strategically designed optimization content to steer the ranking of outputs. We introduce three types of optimization content: string-based, reasoning-based, and review-based, demonstrating their effectiveness in shaping output rankings. To evaluate CORE in realistic settings, we introduce ProductBench, a large-scale benchmark with 15 product categories and 200 products per category, where each product is associated with its top-10 recommendations collected from Amazon's search interface. Extensive experiments on four LLMs with search capabilities (GPT-4o, Gemini-2.5, Claude-4, and Grok-3) demonstrate that CORE achieves an average Promotion Success Rate of \textbf{91.4\% @Top-5}, \textbf{86.6\% @Top-3}, and \textbf{80.3\% @Top-1}, across 15 product categories, outperforming existing ranking manipulation methods while preserving the fluency of optimized content.
>
---
#### [new 069] The Mask of Civility: Benchmarking Chinese Mock Politeness Comprehension in Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于自然语言理解任务，旨在解决中文礼貌、不礼貌及伪礼貌现象的识别问题。通过构建数据集并测试多个大模型，探索其在语用理解上的表现。**

- **链接: [https://arxiv.org/pdf/2602.03107v1](https://arxiv.org/pdf/2602.03107v1)**

> **作者:** Yitong Zhang; Yuhan Xiang; Mingxuan Liu
>
> **备注:** Preprint
>
> **摘要:** From a pragmatic perspective, this study systematically evaluates the differences in performance among representative large language models (LLMs) in recognizing politeness, impoliteness, and mock politeness phenomena in Chinese. Addressing the existing gaps in pragmatic comprehension, the research adopts the frameworks of Rapport Management Theory and the Model of Mock Politeness to construct a three-category dataset combining authentic and simulated Chinese discourse. Six representative models, including GPT-5.1 and DeepSeek, were selected as test subjects and evaluated under four prompting conditions: zero-shot, few-shot, knowledge-enhanced, and hybrid strategies. This study serves as a meaningful attempt within the paradigm of ``Great Linguistics,'' offering a novel approach to applying pragmatic theory in the age of technological transformation. It also responds to the contemporary question of how technology and the humanities may coexist, representing an interdisciplinary endeavor that bridges linguistic technology and humanistic reflection.
>
---
#### [new 070] Predicting first-episode homelessness among US Veterans using longitudinal EHR data: time-varying models and social risk factors
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于风险预测任务，旨在预测退伍军人首次无家可归，通过分析电子健康记录中的长期数据和社交风险因素，提升预测准确性。**

- **链接: [https://arxiv.org/pdf/2602.02731v1](https://arxiv.org/pdf/2602.02731v1)**

> **作者:** Rohan Pandey; Haijuan Yan; Hong Yu; Jack Tsai
>
> **摘要:** Homelessness among US veterans remains a critical public health challenge, yet risk prediction offers a pathway for proactive intervention. In this retrospective prognostic study, we analyzed electronic health record (EHR) data from 4,276,403 Veterans Affairs patients during a 2016 observation period to predict first-episode homelessness occurring 3-12 months later in 2017 (prevalence: 0.32-1.19%). We constructed static and time-varying EHR representations, utilizing clinician-informed logic to model the persistence of clinical conditions and social risks over time. We then compared the performance of classical machine learning, transformer-based masked language models, and fine-tuned large language models (LLMs). We demonstrate that incorporating social and behavioral factors into longitudinal models improved precision-recall area under the curve (PR-AUC) by 15-30%. In the top 1% risk tier, models yielded positive predictive values ranging from 3.93-4.72% at 3 months, 7.39-8.30% at 6 months, 9.84-11.41% at 9 months, and 11.65-13.80% at 12 months across model architectures. Large language models underperformed encoder-based models on discrimination but showed smaller performance disparities across racial groups. These results demonstrate that longitudinal, socially informed EHR modeling concentrates homelessness risk into actionable strata, enabling targeted and data-informed prevention strategies for at-risk veterans.
>
---
#### [new 071] OmniRAG-Agent: Agentic Omnimodal Reasoning for Low-Resource Long Audio-Video Question Answering
- **分类: cs.CL**

- **简介: 该论文属于多模态问答任务，旨在解决低资源下长音频视频问答的编码成本高、检索弱、规划有限等问题。提出OmniRAG-Agent，通过检索增强生成和代理循环提升性能。**

- **链接: [https://arxiv.org/pdf/2602.03707v1](https://arxiv.org/pdf/2602.03707v1)**

> **作者:** Yifan Zhu; Xinyu Mu; Tao Feng; Zhonghong Ou; Yuning Gong; Haoran Luo
>
> **摘要:** Long-horizon omnimodal question answering answers questions by reasoning over text, images, audio, and video. Despite recent progress on OmniLLMs, low-resource long audio-video QA still suffers from costly dense encoding, weak fine-grained retrieval, limited proactive planning, and no clear end-to-end optimization.To address these issues, we propose OmniRAG-Agent, an agentic omnimodal QA method for budgeted long audio-video reasoning. It builds an image-audio retrieval-augmented generation module that lets an OmniLLM fetch short, relevant frames and audio snippets from external banks. Moreover, it uses an agent loop that plans, calls tools across turns, and merges retrieved evidence to answer complex queries. Furthermore, we apply group relative policy optimization to jointly improve tool use and answer quality over time. Experiments on OmniVideoBench, WorldSense, and Daily-Omni show that OmniRAG-Agent consistently outperforms prior methods under low-resource settings and achieves strong results, with ablations validating each component.
>
---
#### [new 072] POP: Prefill-Only Pruning for Efficient Large Model Inference
- **分类: cs.CL; cs.AI; cs.CV**

- **简介: 该论文属于模型推理优化任务，解决大模型推理效率与精度的平衡问题。通过阶段感知的剪枝策略，提升预填充阶段速度，同时保持解码阶段精度。**

- **链接: [https://arxiv.org/pdf/2602.03295v1](https://arxiv.org/pdf/2602.03295v1)**

> **作者:** Junhui He; Zhihui Fu; Jun Wang; Qingan Li
>
> **摘要:** Large Language Models (LLMs) and Vision-Language Models (VLMs) have demonstrated remarkable capabilities. However, their deployment is hindered by significant computational costs. Existing structured pruning methods, while hardware-efficient, often suffer from significant accuracy degradation. In this paper, we argue that this failure stems from a stage-agnostic pruning approach that overlooks the asymmetric roles between the prefill and decode stages. By introducing a virtual gate mechanism, our importance analysis reveals that deep layers are critical for next-token prediction (decode) but largely redundant for context encoding (prefill). Leveraging this insight, we propose Prefill-Only Pruning (POP), a stage-aware inference strategy that safely omits deep layers during the computationally intensive prefill stage while retaining the full model for the sensitive decode stage. To enable the transition between stages, we introduce independent Key-Value (KV) projections to maintain cache integrity, and a boundary handling strategy to ensure the accuracy of the first generated token. Extensive experiments on Llama-3.1, Qwen3-VL, and Gemma-3 across diverse modalities demonstrate that POP achieves up to 1.37$\times$ speedup in prefill latency with minimal performance loss, effectively overcoming the accuracy-efficiency trade-off limitations of existing structured pruning methods.
>
---
#### [new 073] Context Compression via Explicit Information Transmission
- **分类: cs.CL**

- **简介: 该论文属于长文本压缩任务，旨在解决LLM处理长上下文时的计算成本高问题。提出ComprExIT框架，通过显式信息传输实现高效压缩。**

- **链接: [https://arxiv.org/pdf/2602.03784v1](https://arxiv.org/pdf/2602.03784v1)**

> **作者:** Jiangnan Ye; Hanqi Yan; Zhenyi Shen; Heng Chang; Ye Mao; Yulan He
>
> **摘要:** Long-context inference with Large Language Models (LLMs) is costly due to quadratic attention and growing key-value caches, motivating context compression. In this work, we study soft context compression, where a long context is condensed into a small set of continuous representations. Existing methods typically re-purpose the LLM itself as a trainable compressor, relying on layer-by-layer self-attention to iteratively aggregate information. We argue that this paradigm suffers from two structural limitations: (i) progressive representation overwriting across layers (ii) uncoordinated allocation of compression capacity across tokens. We propose ComprExIT (Context Compression via Explicit Information Transmission), a lightweight framework that formulates soft compression into a new paradigm: explicit information transmission over frozen LLM hidden states. This decouples compression from the model's internal self-attention dynamics. ComprExIT performs (i) depth-wise transmission to selectively transmit multi-layer information into token anchors, mitigating progressive overwriting, and (ii) width-wise transmission to aggregate anchors into a small number of slots via a globally optimized transmission plan, ensuring coordinated allocation of information. Across six question-answering benchmarks, ComprExIT consistently outperforms state-of-the-art context compression methods while introducing only ~1% additional parameters, demonstrating that explicit and coordinated information transmission enables more effective and robust long-context compression.
>
---
#### [new 074] R2-Router: A New Paradigm for LLM Routing with Reasoning
- **分类: cs.CL**

- **简介: 该论文属于LLM路由任务，解决现有方法忽略输出长度对质量影响的问题。通过引入R2-Router，联合选择最佳模型和长度预算，提升效率并降低成本。**

- **链接: [https://arxiv.org/pdf/2602.02823v1](https://arxiv.org/pdf/2602.02823v1)**

> **作者:** Jiaqi Xue; Qian Lou; Jiarong Xing; Heng Huang
>
> **摘要:** As LLMs proliferate with diverse capabilities and costs, LLM routing has emerged by learning to predict each LLM's quality and cost for a given query, then selecting the one with high quality and low cost. However, existing routers implicitly assume a single fixed quality and cost per LLM for each query, ignoring that the same LLM's quality varies with its output length. This causes routers to exclude powerful LLMs when their estimated cost exceeds the budget, missing the opportunity that these LLMs could still deliver high quality at reduced cost with shorter outputs. To address this, we introduce R2-Router, which treats output length budget as a controllable variable and jointly selects the best LLM and length budget, enforcing the budget via length-constrained instructions. This enables R2-Router to discover that a powerful LLM with constrained output can outperform a weaker LLM at comparable cost-efficient configurations invisible to prior methods. Together with the router framework, we construct R2-Bench, the first routing dataset capturing LLM behavior across diverse output length budgets. Experiments show that R2-Router achieves state-of-the-art performance at 4-5x lower cost compared with existing routers. This work opens a new direction: routing as reasoning, where routers evolve from reactive selectors to deliberate reasoners that explore which LLM to use and at what cost budget.
>
---
#### [new 075] Test-Time Detoxification without Training or Learning Anything
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于文本安全任务，旨在减少语言模型生成的有害内容。提出一种无需训练的测试阶段优化方法，通过输入嵌入调整生成内容，提升安全性与质量平衡。**

- **链接: [https://arxiv.org/pdf/2602.02498v1](https://arxiv.org/pdf/2602.02498v1)**

> **作者:** Baturay Saglam; Dionysis Kalogerias
>
> **摘要:** Large language models can produce toxic or inappropriate text even for benign inputs, creating risks when deployed at scale. Detoxification is therefore important for safety and user trust, particularly when we want to reduce harmful content without sacrificing the model's generation quality. Many existing approaches rely on model retraining, gradients, or learned auxiliary components, which can be costly and may not transfer across model families or to truly black-box settings. We introduce a test-time procedure that approximates the gradient of completion toxicity with respect to the input embeddings and uses a small number of descent steps to steer generation toward less toxic continuations. This is achieved with zeroth-order optimization that requires only access to input embeddings, a toxicity scoring function, and forward evaluations of the model. Empirically, the approach delivers robust toxicity reductions across models and prompts and, in most settings, achieves the best overall toxicity-quality trade-off. More broadly, our work positions word embeddings as effective control variables and encourages wider use of black-box optimization to guide autoregressive language models toward scalable, safer text generation, without requiring any training or access to intermediate computations.
>
---
#### [new 076] ACL: Aligned Contrastive Learning Improves BERT and Multi-exit BERT Fine-tuning
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，解决监督设置下对比学习与交叉熵损失冲突的问题。提出ACL框架，提升BERT及多出口BERT的微调效果。**

- **链接: [https://arxiv.org/pdf/2602.03563v1](https://arxiv.org/pdf/2602.03563v1)**

> **作者:** Wei Zhu
>
> **摘要:** Despite its success in self-supervised learning, contrastive learning is less studied in the supervised setting. In this work, we first use a set of pilot experiments to show that in the supervised setting, the cross-entropy loss objective (CE) and the contrastive learning objective often conflict with each other, thus hindering the applications of CL in supervised settings. To resolve this problem, we introduce a novel \underline{A}ligned \underline{C}ontrastive \underline{L}earning (ACL) framework. First, ACL-Embed regards label embeddings as extra augmented samples with different labels and employs contrastive learning to align the label embeddings with its samples' representations. Second, to facilitate the optimization of ACL-Embed objective combined with the CE loss, we propose ACL-Grad, which will discard the ACL-Embed term if the two objectives are in conflict. To further enhance the performances of intermediate exits of multi-exit BERT, we further propose cross-layer ACL (ACL-CL), which is to ask the teacher exit to guide the optimization of student shallow exits. Extensive experiments on the GLUE benchmark results in the following takeaways: (a) ACL-BRT outperforms or performs comparably with CE and CE+SCL on the GLUE tasks; (b) ACL, especially CL-ACL, significantly surpasses the baseline methods on the fine-tuning of multi-exit BERT, thus providing better quality-speed tradeoffs for low-latency applications.
>
---
#### [new 077] Towards Understanding Steering Strength
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于语言模型控制任务，旨在解决后训练阶段对LLM进行中间表示调整时的强度选择问题。通过理论分析与实验验证，研究了调整强度对模型输出的影响。**

- **链接: [https://arxiv.org/pdf/2602.02712v1](https://arxiv.org/pdf/2602.02712v1)**

> **作者:** Magamed Taimeskhanov; Samuel Vaiter; Damien Garreau
>
> **备注:** 33 pages (including appendix)
>
> **摘要:** A popular approach to post-training control of large language models (LLMs) is the steering of intermediate latent representations. Namely, identify a well-chosen direction depending on the task at hand and perturbs representations along this direction at inference time. While many propositions exist to pick this direction, considerably less is understood about how to choose the magnitude of the move, whereas its importance is clear: too little and the intended behavior does not emerge, too much and the model's performance degrades beyond repair. In this work, we propose the first theoretical analysis of steering strength. We characterize its effect on next token probability, presence of a concept, and cross-entropy, deriving precise qualitative laws governing these quantities. Our analysis reveals surprising behaviors, including non-monotonic effects of steering strength. We validate our theoretical predictions empirically on eleven language models, ranging from a small GPT architecture to modern models.
>
---
#### [new 078] Fine-Tuning Language Models to Know What They Know
- **分类: cs.NE; cs.AI; cs.CL; q-bio.NC**

- **简介: 该论文属于自然语言处理任务，旨在提升语言模型的元认知能力，通过框架和算法增强模型对自身知识的意识与引用。**

- **链接: [https://arxiv.org/pdf/2602.02605v1](https://arxiv.org/pdf/2602.02605v1)**

> **作者:** Sangjun Park; Elliot Meyerson; Xin Qiu; Risto Miikkulainen
>
> **备注:** Preprint
>
> **摘要:** Metacognition is a critical component of intelligence, specifically regarding the awareness of one's own knowledge. While humans rely on shared internal memory for both answering questions and reporting their knowledge state, this dependency in LLMs remains underexplored. This study proposes a framework to measure metacognitive ability $d_{\rm{type2}}'$ using a dual-prompt method, followed by the introduction of Evolution Strategy for Metacognitive Alignment (ESMA) to bind a model's internal knowledge to its explicit behaviors. ESMA demonstrates robust generalization across diverse untrained settings, indicating a enhancement in the model's ability to reference its own knowledge. Furthermore, parameter analysis attributes these improvements to a sparse set of significant modifications.
>
---
#### [new 079] BinaryPPO: Efficient Policy Optimization for Binary Classification
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文针对二分类任务，解决标签噪声、类别不平衡等问题。提出BinaryPPO框架，通过强化学习提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.02708v1](https://arxiv.org/pdf/2602.02708v1)**

> **作者:** Punya Syon Pandey; Zhijing Jin
>
> **摘要:** Supervised fine-tuning (SFT) is the standard approach for binary classification tasks such as toxicity detection, factuality verification, and causal inference. However, SFT often performs poorly in real-world settings with label noise, class imbalance, or sparse supervision. We introduce BinaryPPO, an offline reinforcement learning large language model (LLM) framework that reformulates binary classification as a reward maximization problem. Our method leverages a variant of Proximal Policy Optimization (PPO) with a confidence-weighted reward function that penalizes uncertain or incorrect predictions, enabling the model to learn robust decision policies from static datasets without online interaction. Across eight domain-specific benchmarks and multiple models with differing architectures, BinaryPPO improves accuracy by 40-60 percentage points, reaching up to 99%, substantially outperforming supervised baselines. We provide an in-depth analysis of the role of reward shaping, advantage scaling, and policy stability in enabling this improvement. Overall, we demonstrate that confidence-based reward design provides a robust alternative to SFT for binary classification. Our code is available at https://github.com/psyonp/BinaryPPO.
>
---
#### [new 080] R1-SyntheticVL: Is Synthetic Data from Generative Models Ready for Multimodal Large Language Model?
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态大语言模型任务，旨在解决合成数据质量与多样性问题。提出CADS方法，通过对抗学习生成高质量多模态数据，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.03300v1](https://arxiv.org/pdf/2602.03300v1)**

> **作者:** Jingyi Zhang; Tianyi Lin; Huanjin Yao; Xiang Lan; Shunyu Liu; Jiaxing Huang
>
> **摘要:** In this work, we aim to develop effective data synthesis techniques that autonomously synthesize multimodal training data for enhancing MLLMs in solving complex real-world tasks. To this end, we propose Collective Adversarial Data Synthesis (CADS), a novel and general approach to synthesize high-quality, diverse and challenging multimodal data for MLLMs. The core idea of CADS is to leverage collective intelligence to ensure high-quality and diverse generation, while exploring adversarial learning to synthesize challenging samples for effectively driving model improvement. Specifically, CADS operates with two cyclic phases, i.e., Collective Adversarial Data Generation (CAD-Generate) and Collective Adversarial Data Judgment (CAD-Judge). CAD-Generate leverages collective knowledge to jointly generate new and diverse multimodal data, while CAD-Judge collaboratively assesses the quality of synthesized data. In addition, CADS introduces an Adversarial Context Optimization mechanism to optimize the generation context to encourage challenging and high-value data generation. With CADS, we construct MMSynthetic-20K and train our model R1-SyntheticVL, which demonstrates superior performance on various benchmarks.
>
---
#### [new 081] Privately Fine-Tuned LLMs Preserve Temporal Dynamics in Tabular Data
- **分类: cs.LG; cs.CL; cs.CR**

- **简介: 该论文属于隐私合成数据任务，解决纵向数据中时间动态保留问题。提出PATH框架，利用私有微调大模型生成完整表格，有效捕捉长程依赖。**

- **链接: [https://arxiv.org/pdf/2602.02766v1](https://arxiv.org/pdf/2602.02766v1)**

> **作者:** Lucas Rosenblatt; Peihan Liu; Ryan McKenna; Natalia Ponomareva
>
> **摘要:** Research on differentially private synthetic tabular data has largely focused on independent and identically distributed rows where each record corresponds to a unique individual. This perspective neglects the temporal complexity in longitudinal datasets, such as electronic health records, where a user contributes an entire (sub) table of sequential events. While practitioners might attempt to model such data by flattening user histories into high-dimensional vectors for use with standard marginal-based mechanisms, we demonstrate that this strategy is insufficient. Flattening fails to preserve temporal coherence even when it maintains valid marginal distributions. We introduce PATH, a novel generative framework that treats the full table as the unit of synthesis and leverages the autoregressive capabilities of privately fine-tuned large language models. Extensive evaluations show that PATH effectively captures long-range dependencies that traditional methods miss. Empirically, our method reduces the distributional distance to real trajectories by over 60% and reduces state transition errors by nearly 50% compared to leading marginal mechanisms while achieving similar marginal fidelity.
>
---
#### [new 082] GraphDancer: Training LLMs to Explore and Reason over Graphs via Curriculum Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出GraphDancer，解决LLMs在图结构知识上的导航与推理问题。通过课程强化学习，提升模型跨领域泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.02518v1](https://arxiv.org/pdf/2602.02518v1)**

> **作者:** Yuyang Bai; Zhuofeng Li; Ping Nie; Jianwen Xie; Yu Zhang
>
> **备注:** 15 pages, Project website: https://yuyangbai.com/graphdancer/
>
> **摘要:** Large language models (LLMs) increasingly rely on external knowledge to improve factuality, yet many real-world knowledge sources are organized as heterogeneous graphs rather than plain text. Reasoning over such graph-structured knowledge poses two key challenges: (1) navigating structured, schema-defined relations requires precise function calls rather than similarity-based retrieval, and (2) answering complex questions often demands multi-hop evidence aggregation through iterative information seeking. We propose GraphDancer, a reinforcement learning (RL) framework that teaches LLMs to navigate graphs by interleaving reasoning and function execution. To make RL effective for moderate-sized LLMs, we introduce a graph-aware curriculum that schedules training by the structural complexity of information-seeking trajectories using an easy-to-hard biased sampler. We evaluate GraphDancer on a multi-domain benchmark by training on one domain only and testing on unseen domains and out-of-distribution question types. Despite using only a 3B backbone, GraphDancer outperforms baselines equipped with either a 14B backbone or GPT-4o-mini, demonstrating robust cross-domain generalization of graph exploration and reasoning skills. Our code and models can be found at https://yuyangbai.com/graphdancer/ .
>
---
#### [new 083] Conflict-Resolving and Sharpness-Aware Minimization for Generalized Knowledge Editing with Multiple Updates
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于知识编辑任务，解决多更新下的知识冲突与稳定性问题。提出CoRSA框架，通过最小化损失曲率和最大化新旧知识间隔，提升泛化能力和更新效果。**

- **链接: [https://arxiv.org/pdf/2602.03696v1](https://arxiv.org/pdf/2602.03696v1)**

> **作者:** Duy Nguyen; Hanqi Xiao; Archiki Prasad; Elias Stengel-Eskin; Hyunji Lee; Mohit Bansal
>
> **备注:** 22 pages, 8 figures. Code link: https://github.com/duykhuongnguyen/CoRSA
>
> **摘要:** Large language models (LLMs) rely on internal knowledge to solve many downstream tasks, making it crucial to keep them up to date. Since full retraining is expensive, prior work has explored efficient alternatives such as model editing and parameter-efficient fine-tuning. However, these approaches often break down in practice due to poor generalization across inputs, limited stability, and knowledge conflict. To address these limitations, we propose the CoRSA (Conflict-Resolving and Sharpness-Aware Minimization) training framework, a parameter-efficient, holistic approach for knowledge editing with multiple updates. CoRSA tackles multiple challenges simultaneously: it improves generalization to different input forms and enhances stability across multiple updates by minimizing loss curvature, and resolves conflicts by maximizing the margin between new and prior knowledge. Across three widely used fact editing benchmarks, CoRSA achieves significant gains in generalization, outperforming baselines with average absolute improvements of 12.42% over LoRA and 10% over model editing methods. With multiple updates, it maintains high update efficacy while reducing catastrophic forgetting by 27.82% compared to LoRA. CoRSA also generalizes to the code domain, outperforming the strongest baseline by 5.48% Pass@5 in update efficacy.
>
---
#### [new 084] AutoFigure: Generating and Refining Publication-Ready Scientific Illustrations
- **分类: cs.AI; cs.CL; cs.CV; cs.DL**

- **简介: 该论文提出AutoFigure，解决科学插图自动生成问题。通过构建FigureBench数据集和设计自动框架，生成高质量、适合发表的科学图表。**

- **链接: [https://arxiv.org/pdf/2602.03828v1](https://arxiv.org/pdf/2602.03828v1)**

> **作者:** Minjun Zhu; Zhen Lin; Yixuan Weng; Panzhong Lu; Qiujie Xie; Yifan Wei; Sifan Liu; Qiyao Sun; Yue Zhang
>
> **备注:** Accepted at the ICLR 2026
>
> **摘要:** High-quality scientific illustrations are crucial for effectively communicating complex scientific and technical concepts, yet their manual creation remains a well-recognized bottleneck in both academia and industry. We present FigureBench, the first large-scale benchmark for generating scientific illustrations from long-form scientific texts. It contains 3,300 high-quality scientific text-figure pairs, covering diverse text-to-illustration tasks from scientific papers, surveys, blogs, and textbooks. Moreover, we propose AutoFigure, the first agentic framework that automatically generates high-quality scientific illustrations based on long-form scientific text. Specifically, before rendering the final result, AutoFigure engages in extensive thinking, recombination, and validation to produce a layout that is both structurally sound and aesthetically refined, outputting a scientific illustration that achieves both structural completeness and aesthetic appeal. Leveraging the high-quality data from FigureBench, we conduct extensive experiments to test the performance of AutoFigure against various baseline methods. The results demonstrate that AutoFigure consistently surpasses all baseline methods, producing publication-ready scientific illustrations. The code, dataset and huggingface space are released in https://github.com/ResearAI/AutoFigure.
>
---
#### [new 085] Vector Quantized Latent Concepts: A Scalable Alternative to Clustering-Based Concept Discovery
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于概念发现任务，旨在解决大规模数据下聚类方法效率低、效果差的问题。提出VQLC方法，基于VQ-VAE架构，提升可扩展性并保持解释质量。**

- **链接: [https://arxiv.org/pdf/2602.02726v1](https://arxiv.org/pdf/2602.02726v1)**

> **作者:** Xuemin Yu; Ankur Garg; Samira Ebrahimi Kahou; Hassan Sajjad
>
> **摘要:** Deep Learning models encode rich semantic information in their hidden representations. However, it remains challenging to understand which parts of this information models actually rely on when making predictions. A promising line of post-hoc concept-based explanation methods relies on clustering token representations. However, commonly used approaches such as hierarchical clustering are computationally infeasible for large-scale datasets, and K-Means often yields shallow or frequency-dominated clusters. We propose the vector quantized latent concept (VQLC) method, a framework built upon the vector quantized-variational autoencoder (VQ-VAE) architecture that learns a discrete codebook mapping continuous representations to concept vectors. We perform thorough evaluations and show that VQLC improves scalability while maintaining comparable quality of human-understandable explanations.
>
---
#### [new 086] WAXAL: A Large-Scale Multilingual African Language Speech Corpus
- **分类: eess.AS; cs.AI; cs.CL**

- **简介: 该论文介绍WAXAL，一个针对21种非洲语言的大规模语音数据集，旨在解决低资源语言在语音技术中的数字鸿沟问题。任务为语音识别与合成，工作包括数据收集、标注及质量控制。**

- **链接: [https://arxiv.org/pdf/2602.02734v1](https://arxiv.org/pdf/2602.02734v1)**

> **作者:** Abdoulaye Diack; Perry Nelson; Kwaku Agbesi; Angela Nakalembe; MohamedElfatih MohamedKhair; Vusumuzi Dube; Tavonga Siyavora; Subhashini Venugopalan; Jason Hickey; Uche Okonkwo; Abhishek Bapna; Isaac Wiafe; Raynard Dodzi Helegah; Elikem Doe Atsakpo; Charles Nutrokpor; Fiifi Baffoe Payin Winful; Kafui Kwashie Solaga; Jamal-Deen Abdulai; Akon Obu Ekpezu; Audace Niyonkuru; Samuel Rutunda; Boris Ishimwe; Michael Melese; Engineer Bainomugisha; Joyce Nakatumba-Nabende; Andrew Katumba; Claire Babirye; Jonathan Mukiibi; Vincent Kimani; Samuel Kibacia; James Maina; Fridah Emmah; Ahmed Ibrahim Shekarau; Ibrahim Shehu Adamu; Yusuf Abdullahi; Howard Lakougna; Bob MacDonald; Hadar Shemtov; Aisha Walcott-Bryant; Moustapha Cisse; Avinatan Hassidim; Jeff Dean; Yossi Matias
>
> **备注:** Initial dataset release
>
> **摘要:** The advancement of speech technology has predominantly favored high-resource languages, creating a significant digital divide for speakers of most Sub-Saharan African languages. To address this gap, we introduce WAXAL, a large-scale, openly accessible speech dataset for 21 languages representing over 100 million speakers. The collection consists of two main components: an Automated Speech Recognition (ASR) dataset containing approximately 1,250 hours of transcribed, natural speech from a diverse range of speakers, and a Text-to-Speech (TTS) dataset with over 180 hours of high-quality, single-speaker recordings reading phonetically balanced scripts. This paper details our methodology for data collection, annotation, and quality control, which involved partnerships with four African academic and community organizations. We provide a detailed statistical overview of the dataset and discuss its potential limitations and ethical considerations. The WAXAL datasets are released at https://huggingface.co/datasets/google/WaxalNLP under the permissive CC-BY-4.0 license to catalyze research, enable the development of inclusive technologies, and serve as a vital resource for the digital preservation of these languages.
>
---
#### [new 087] SWE-World: Building Software Engineering Agents in Docker-Free Environments
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出SWE-World，解决软件工程代理训练中依赖Docker环境的问题。通过学习代理交互数据预测执行结果，提升代理性能，实现高效测试时扩展。**

- **链接: [https://arxiv.org/pdf/2602.03419v1](https://arxiv.org/pdf/2602.03419v1)**

> **作者:** Shuang Sun; Huatong Song; Lisheng Huang; Jinhao Jiang; Ran Le; Zhihao Lv; Zongchao Chen; Yiwen Hu; Wenyang Luo; Wayne Xin Zhao; Yang Song; Hongteng Xu; Tao Zhang; Ji-Rong Wen
>
> **摘要:** Recent advances in large language models (LLMs) have enabled software engineering agents to tackle complex code modification tasks. Most existing approaches rely on execution feedback from containerized environments, which require dependency-complete setup and physical execution of programs and tests. While effective, this paradigm is resource-intensive and difficult to maintain, substantially complicating agent training and limiting scalability. We propose SWE-World, a Docker-free framework that replaces physical execution environments with a learned surrogate for training and evaluating software engineering agents. SWE-World leverages LLM-based models trained on real agent-environment interaction data to predict intermediate execution outcomes and final test feedback, enabling agents to learn without interacting with physical containerized environments. This design preserves the standard agent-environment interaction loop while eliminating the need for costly environment construction and maintenance during agent optimization and evaluation. Furthermore, because SWE-World can simulate the final evaluation outcomes of candidate trajectories without real submission, it enables selecting the best solution among multiple test-time attempts, thereby facilitating effective test-time scaling (TTS) in software engineering tasks. Experiments on SWE-bench Verified demonstrate that SWE-World raises Qwen2.5-Coder-32B from 6.2\% to 52.0\% via Docker-free SFT, 55.0\% with Docker-free RL, and 68.2\% with further TTS. The code is available at https://github.com/RUCAIBox/SWE-World
>
---
#### [new 088] Chain of Simulation: A Dual-Mode Reasoning Framework for Large Language Models with Dynamic Problem Routing
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出CoS框架，解决大语言模型推理问题。通过动态路由不同推理模式，提升数学、空间和多跳推理的准确性。**

- **链接: [https://arxiv.org/pdf/2602.02842v1](https://arxiv.org/pdf/2602.02842v1)**

> **作者:** Saeid Sheikhi
>
> **摘要:** We present Chain of Simulation (CoS), a novel dual-mode reasoning framework that dynamically routes problems to specialized reasoning strategies in Large Language Models (LLMs). Unlike existing uniform prompting approaches, CoS employs three distinct reasoning modes: (1) computational flow with self-consistency for mathematical problems, (2) symbolic state tracking with JSON representations for spatial reasoning, and (3) hybrid fact-extraction for multi-hop inference. Through comprehensive evaluation on GSM8K, StrategyQA, and bAbI benchmarks using four state-of-the-art models (Gemma-3 27B, LLaMA-3.1 8B, Mistral 7B, and Qwen-2.5 14B), we demonstrate that CoS achieves 71.5% accuracy on GSM8K (1.0% absolute improvement), 90.0% on StrategyQA (2.5% improvement), and 19.0% on bAbI (65.2% relative improvement) compared to the strongest baselines. The analysis reveals that problem-specific mode selection is crucial, with computational mode achieving 81.2% accuracy when correctly applied to mathematical problems, while misrouting leads to 0% accuracy. We provide detailed algorithms for mode selection, state tracking, and answer extraction, establishing CoS as an effective approach for improving LLM reasoning without additional training. The framework provides superior trade-offs between accuracy and efficiency compared to Self-Consistency, achieving comparable performance at 54% lower computational cost.
>
---
#### [new 089] Bridging Online and Offline RL: Contextual Bandit Learning for Multi-Turn Code Generation
- **分类: cs.LG; cs.AI; cs.CL; cs.SE**

- **简介: 该论文针对多轮代码生成任务，提出Cobalt方法，结合在线与离线强化学习优势，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.03806v1](https://arxiv.org/pdf/2602.03806v1)**

> **作者:** Ziru Chen; Dongdong Chen; Ruinan Jin; Yingbin Liang; Yujia Xie; Huan Sun
>
> **摘要:** Recently, there have been significant research interests in training large language models (LLMs) with reinforcement learning (RL) on real-world tasks, such as multi-turn code generation. While online RL tends to perform better than offline RL, its higher training cost and instability hinders wide adoption. In this paper, we build on the observation that multi-turn code generation can be formulated as a one-step recoverable Markov decision process and propose contextual bandit learning with offline trajectories (Cobalt), a new method that combines the benefits of online and offline RL. Cobalt first collects code generation trajectories using a reference LLM and divides them into partial trajectories as contextual prompts. Then, during online bandit learning, the LLM is trained to complete each partial trajectory prompt through single-step code generation. Cobalt outperforms two multi-turn online RL baselines based on GRPO and VeRPO, and substantially improves R1-Distill 8B and Qwen3 8B by up to 9.0 and 6.2 absolute Pass@1 scores on LiveCodeBench. Also, we analyze LLMs' in-context reward hacking behaviors and augment Cobalt training with perturbed trajectories to mitigate this issue. Overall, our results demonstrate Cobalt as a promising solution for iterative decision-making tasks like multi-turn code generation. Our code and data are available at https://github.com/OSU-NLP-Group/cobalt.
>
---
#### [new 090] Efficient Estimation of Kernel Surrogate Models for Task Attribution
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究任务归属问题，旨在量化不同训练任务对目标任务的影响。提出核代理模型以更有效地捕捉任务间的非线性关系，相比线性模型有更高精度。**

- **链接: [https://arxiv.org/pdf/2602.03783v1](https://arxiv.org/pdf/2602.03783v1)**

> **作者:** Zhenshuo Zhang; Minxuan Duan; Hongyang R. Zhang
>
> **备注:** 27 pages. To appear in ICLR 2026
>
> **摘要:** Modern AI agents such as large language models are trained on diverse tasks -- translation, code generation, mathematical reasoning, and text prediction -- simultaneously. A key question is to quantify how each individual training task influences performance on a target task, a problem we refer to as task attribution. The direct approach, leave-one-out retraining, measures the effect of removing each task, but is computationally infeasible at scale. An alternative approach that builds surrogate models to predict a target task's performance for any subset of training tasks has emerged in recent literature. Prior work focuses on linear surrogate models, which capture first-order relationships, but miss nonlinear interactions such as synergy, antagonism, or XOR-type effects. In this paper, we first consider a unified task weighting framework for analyzing task attribution methods, and show a new connection between linear surrogate models and influence functions through a second-order analysis. Then, we introduce kernel surrogate models, which more effectively represent second-order task interactions. To efficiently learn the kernel surrogate, we develop a gradient-based estimation procedure that leverages a first-order approximation of pretrained models; empirically, this yields accurate estimates with less than $2\%$ relative error without repeated retraining. Experiments across multiple domains -- including math reasoning in transformers, in-context learning, and multi-objective reinforcement learning -- demonstrate the effectiveness of kernel surrogate models. They achieve a $25\%$ higher correlation with the leave-one-out ground truth than linear surrogates and influence-function baselines. When used for downstream task selection, kernel surrogate models yield a $40\%$ improvement in demonstration selection for in-context learning and multi-objective reinforcement learning benchmarks.
>
---
#### [new 091] Uncertainty and Fairness Awareness in LLM-Based Recommendation Systems
- **分类: cs.AI; cs.CL; cs.CY; cs.IR; cs.LG; cs.SE**

- **简介: 该论文属于推荐系统任务，旨在解决LLM推荐中的不确定性与公平性问题。通过构建基准和数据集，分析模型的预测不确定性和偏见，提出新的评估方法以提升推荐系统的可靠性与公平性。**

- **链接: [https://arxiv.org/pdf/2602.02582v1](https://arxiv.org/pdf/2602.02582v1)**

> **作者:** Chandan Kumar Sah; Xiaoli Lian; Li Zhang; Tony Xu; Syed Shazaib Shah
>
> **备注:** Accepted at the Second Conference of the International Association for Safe and Ethical Artificial Intelligence, IASEAI26, 14 pages
>
> **摘要:** Large language models (LLMs) enable powerful zero-shot recommendations by leveraging broad contextual knowledge, yet predictive uncertainty and embedded biases threaten reliability and fairness. This paper studies how uncertainty and fairness evaluations affect the accuracy, consistency, and trustworthiness of LLM-generated recommendations. We introduce a benchmark of curated metrics and a dataset annotated for eight demographic attributes (31 categorical values) across two domains: movies and music. Through in-depth case studies, we quantify predictive uncertainty (via entropy) and demonstrate that Google DeepMind's Gemini 1.5 Flash exhibits systematic unfairness for certain sensitive attributes; measured similarity-based gaps are SNSR at 0.1363 and SNSV at 0.0507. These disparities persist under prompt perturbations such as typographical errors and multilingual inputs. We further integrate personality-aware fairness into the RecLLM evaluation pipeline to reveal personality-linked bias patterns and expose trade-offs between personalization and group fairness. We propose a novel uncertainty-aware evaluation methodology for RecLLMs, present empirical insights from deep uncertainty case studies, and introduce a personality profile-informed fairness benchmark that advances explainability and equity in LLM recommendations. Together, these contributions establish a foundation for safer, more interpretable RecLLMs and motivate future work on multi-model benchmarks and adaptive calibration for trustworthy deployment.
>
---
#### [new 092] TraceNAS: Zero-shot LLM Pruning via Gradient Trace Correlation
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出TraceNAS，解决LLM结构化剪枝问题，通过无训练的NAS框架联合优化模型深度和宽度，提升剪枝效率与性能。**

- **链接: [https://arxiv.org/pdf/2602.02891v1](https://arxiv.org/pdf/2602.02891v1)**

> **作者:** Prajna G. Malettira; Manish Nagaraj; Arjun Roy; Shubham Negi; Kaushik Roy
>
> **备注:** Preprint
>
> **摘要:** Structured pruning is essential for efficient deployment of Large Language Models (LLMs). The varying sensitivity of LLM sub-blocks to pruning necessitates the identification of optimal non-uniformly pruned models. Existing methods evaluate the importance of layers, attention heads, or weight channels in isolation. Such localized focus ignores the complex global structural dependencies that exist across the model. Training-aware structured pruning addresses global dependencies, but its computational cost can be just as expensive as post-pruning training. To alleviate the computational burden of training-aware pruning and capture global structural dependencies, we propose TraceNAS, a training-free Neural Architecture Search (NAS) framework that jointly explores structured pruning of LLM depth and width. TraceNAS identifies pruned models that maintain a high degree of loss landscape alignment with the pretrained model using a scale-invariant zero-shot proxy, effectively selecting models that exhibit maximal performance potential during post-pruning training. TraceNAS is highly efficient, enabling high-fidelity discovery of pruned models on a single GPU in 8.5 hours, yielding a 10$\times$ reduction in GPU-hours compared to training-aware methods. Evaluations on the Llama and Qwen families demonstrate that TraceNAS is competitive with training-aware baselines across commonsense and reasoning benchmarks.
>
---
#### [new 093] Search-R2: Enhancing Search-Integrated Reasoning via Actor-Refiner Collaboration
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理中的问答任务，解决搜索增强推理中奖励稀疏的问题。提出Search-R2框架，通过Actor-Refiner协作提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2602.03647v1](https://arxiv.org/pdf/2602.03647v1)**

> **作者:** Bowei He; Minda Hu; Zenan Xu; Hongru Wang; Licheng Zong; Yankai Chen; Chen Ma; Xue Liu; Pluto Zhou; Irwin King
>
> **摘要:** Search-integrated reasoning enables language agents to transcend static parametric knowledge by actively querying external sources. However, training these agents via reinforcement learning is hindered by the multi-scale credit assignment problem: existing methods typically rely on sparse, trajectory-level rewards that fail to distinguish between high-quality reasoning and fortuitous guesses, leading to redundant or misleading search behaviors. To address this, we propose Search-R2, a novel Actor-Refiner collaboration framework that enhances reasoning through targeted intervention, with both components jointly optimized during training. Our approach decomposes the generation process into an Actor, which produces initial reasoning trajectories, and a Meta-Refiner, which selectively diagnoses and repairs flawed steps via a 'cut-and-regenerate' mechanism. To provide fine-grained supervision, we introduce a hybrid reward design that couples outcome correctness with a dense process reward quantifying the information density of retrieved evidence. Theoretically, we formalize the Actor-Refiner interaction as a smoothed mixture policy, proving that selective correction yields strict performance gains over strong baselines. Extensive experiments across various general and multi-hop QA datasets demonstrate that Search-R2 consistently outperforms strong RAG and RL-based baselines across model scales, achieving superior reasoning accuracy with minimal overhead.
>
---
#### [new 094] Enhancing Post-Training Quantization via Future Activation Awareness
- **分类: cs.LG; cs.CL; cs.CV**

- **简介: 该论文属于模型压缩任务，旨在解决PTQ中的量化偏差和误差累积问题。通过引入未来层激活信息，提出FAQ方法，提升量化效果并降低对校准数据的敏感性。**

- **链接: [https://arxiv.org/pdf/2602.02538v1](https://arxiv.org/pdf/2602.02538v1)**

> **作者:** Zheqi Lv; Zhenxuan Fan; Qi Tian; Wenqiao Zhang; Yueting Zhuang
>
> **摘要:** Post-training quantization (PTQ) is a widely used method to compress large language models (LLMs) without fine-tuning. It typically sets quantization hyperparameters (e.g., scaling factors) based on current-layer activations. Although this method is efficient, it suffers from quantization bias and error accumulation, resulting in suboptimal and unstable quantization, especially when the calibration data is biased. To overcome these issues, we propose Future-Aware Quantization (FAQ), which leverages future-layer activations to guide quantization. This allows better identification and preservation of important weights, while reducing sensitivity to calibration noise. We further introduce a window-wise preview mechanism to softly aggregate multiple future-layer activations, mitigating over-reliance on any single layer. To avoid expensive greedy search, we use a pre-searched configuration to minimize overhead. Experiments show that FAQ consistently outperforms prior methods with negligible extra cost, requiring no backward passes, data reconstruction, or tuning, making it well-suited for edge deployment.
>
---
#### [new 095] Making Avatars Interact: Towards Text-Driven Human-Object Interaction for Controllable Talking Avatars
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频生成任务，旨在解决可控说话虚拟人与物体的文本驱动交互问题。提出双流框架InteractAvatar，提升环境感知与交互质量。**

- **链接: [https://arxiv.org/pdf/2602.01538v1](https://arxiv.org/pdf/2602.01538v1)**

> **作者:** Youliang Zhang; Zhengguang Zhou; Zhentao Yu; Ziyao Huang; Teng Hu; Sen Liang; Guozhen Zhang; Ziqiao Peng; Shunkai Li; Yi Chen; Zixiang Zhou; Yuan Zhou; Qinglin Lu; Xiu Li
>
> **摘要:** Generating talking avatars is a fundamental task in video generation. Although existing methods can generate full-body talking avatars with simple human motion, extending this task to grounded human-object interaction (GHOI) remains an open challenge, requiring the avatar to perform text-aligned interactions with surrounding objects. This challenge stems from the need for environmental perception and the control-quality dilemma in GHOI generation. To address this, we propose a novel dual-stream framework, InteractAvatar, which decouples perception and planning from video synthesis for grounded human-object interaction. Leveraging detection to enhance environmental perception, we introduce a Perception and Interaction Module (PIM) to generate text-aligned interaction motions. Additionally, an Audio-Interaction Aware Generation Module (AIM) is proposed to synthesize vivid talking avatars performing object interactions. With a specially designed motion-to-video aligner, PIM and AIM share a similar network structure and enable parallel co-generation of motions and plausible videos, effectively mitigating the control-quality dilemma. Finally, we establish a benchmark, GroundedInter, for evaluating GHOI video generation. Extensive experiments and comparisons demonstrate the effectiveness of our method in generating grounded human-object interactions for talking avatars. Project page: https://interactavatar.github.io
>
---
#### [new 096] FullStack-Agent: Enhancing Agentic Full-Stack Web Coding via Development-Oriented Testing and Repository Back-Translation
- **分类: cs.SE; cs.CL; cs.CV**

- **简介: 该论文属于全栈Web开发任务，旨在解决代码代理生成前端而缺乏真实全栈处理的问题。提出FullStack-Agent系统，包含开发、学习和基准测试模块，提升全栈代码生成能力。**

- **链接: [https://arxiv.org/pdf/2602.03798v1](https://arxiv.org/pdf/2602.03798v1)**

> **作者:** Zimu Lu; Houxing Ren; Yunqiao Yang; Ke Wang; Zhuofan Zong; Mingjie Zhan; Hongsheng Li
>
> **摘要:** Assisting non-expert users to develop complex interactive websites has become a popular task for LLM-powered code agents. However, existing code agents tend to only generate frontend web pages, masking the lack of real full-stack data processing and storage with fancy visual effects. Notably, constructing production-level full-stack web applications is far more challenging than only generating frontend web pages, demanding careful control of data flow, comprehensive understanding of constantly updating packages and dependencies, and accurate localization of obscure bugs in the codebase. To address these difficulties, we introduce FullStack-Agent, a unified agent system for full-stack agentic coding that consists of three parts: (1) FullStack-Dev, a multi-agent framework with strong planning, code editing, codebase navigation, and bug localization abilities. (2) FullStack-Learn, an innovative data-scaling and self-improving method that back-translates crawled and synthesized website repositories to improve the backbone LLM of FullStack-Dev. (3) FullStack-Bench, a comprehensive benchmark that systematically tests the frontend, backend and database functionalities of the generated website. Our FullStack-Dev outperforms the previous state-of-the-art method by 8.7%, 38.2%, and 15.9% on the frontend, backend, and database test cases respectively. Additionally, FullStack-Learn raises the performance of a 30B model by 9.7%, 9.5%, and 2.8% on the three sets of test cases through self-improvement, demonstrating the effectiveness of our approach. The code is released at https://github.com/mnluzimu/FullStack-Agent.
>
---
#### [new 097] Thinking Like a Doctor: Conversational Diagnosis through the Exploration of Diagnostic Knowledge Graphs
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于医疗诊断任务，旨在解决对话中诊断不准确的问题。通过构建知识图谱，生成并验证诊断假设，提升诊断的准确性和效率。**

- **链接: [https://arxiv.org/pdf/2602.01995v1](https://arxiv.org/pdf/2602.01995v1)**

> **作者:** Jeongmoon Won; Seungwon Kook; Yohan Jo
>
> **摘要:** Conversational diagnosis requires multi-turn history-taking, where an agent asks clarifying questions to refine differential diagnoses under incomplete information. Existing approaches often rely on the parametric knowledge of a model or assume that patients provide rich and concrete information, which is unrealistic. To address these limitations, we propose a conversational diagnosis system that explores a diagnostic knowledge graph to reason in two steps: (i) generating diagnostic hypotheses from the dialogue context, and (ii) verifying hypotheses through clarifying questions, which are repeated until a final diagnosis is reached. Since evaluating the system requires a realistic patient simulator that responds to the system's questions, we adopt a well-established simulator along with patient profiles from MIMIC-IV. We further adapt it to describe symptoms vaguely to reflect real-world patients during early clinical encounters. Experiments show improved diagnostic accuracy and efficiency over strong baselines, and evaluations by physicians support the realism of our simulator and the clinical utility of the generated questions. Our code will be released upon publication.
>
---
#### [new 098] Agent Primitives: Reusable Latent Building Blocks for Multi-Agent Systems
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文属于多智能体系统任务，旨在解决传统系统任务特定性强、可重用性差的问题。提出Agent Primitives作为可复用的潜在构建块，提升系统稳定性和效率。**

- **链接: [https://arxiv.org/pdf/2602.03695v1](https://arxiv.org/pdf/2602.03695v1)**

> **作者:** Haibo Jin; Kuang Peng; Ye Yu; Xiaopeng Yuan; Haohan Wang
>
> **备注:** 16 pages
>
> **摘要:** While existing multi-agent systems (MAS) can handle complex problems by enabling collaboration among multiple agents, they are often highly task-specific, relying on manually crafted agent roles and interaction prompts, which leads to increased architectural complexity and limited reusability across tasks. Moreover, most MAS communicate primarily through natural language, making them vulnerable to error accumulation and instability in long-context, multi-stage interactions within internal agent histories. In this work, we propose \textbf{Agent Primitives}, a set of reusable latent building blocks for LLM-based MAS. Inspired by neural network design, where complex models are built from reusable components, we observe that many existing MAS architectures can be decomposed into a small number of recurring internal computation patterns. Based on this observation, we instantiate three primitives: Review, Voting and Selection, and Planning and Execution. All primitives communicate internally via key-value (KV) cache, which improves both robustness and efficiency by mitigating information degradation across multi-stage interactions. To enable automatic system construction, an Organizer agent selects and composes primitives for each query, guided by a lightweight knowledge pool of previously successful configurations, forming a primitive-based MAS. Experiments show that primitives-based MAS improve average accuracy by 12.0-16.5\% over single-agent baselines, reduce token usage and inference latency by approximately 3$\times$-4$\times$ compared to text-based MAS, while incurring only 1.3$\times$-1.6$\times$ overhead relative to single-agent inference and providing more stable performance across model backbones.
>
---
#### [new 099] MAS-ProVe: Understanding the Process Verification of Multi-Agent Systems
- **分类: cs.AI; cs.CL; cs.MA**

- **简介: 该论文属于多智能体系统验证任务，旨在解决过程验证在MAS中的有效性问题。通过实验分析不同验证方法，发现过程验证效果不稳定，仍面临挑战。**

- **链接: [https://arxiv.org/pdf/2602.03053v1](https://arxiv.org/pdf/2602.03053v1)**

> **作者:** Vishal Venkataramani; Haizhou Shi; Zixuan Ke; Austin Xu; Xiaoxiao He; Yingbo Zhou; Semih Yavuz; Hao Wang; Shafiq Joty
>
> **备注:** Preprint; work in progress
>
> **摘要:** Multi-Agent Systems (MAS) built on Large Language Models (LLMs) often exhibit high variance in their reasoning trajectories. Process verification, which evaluates intermediate steps in trajectories, has shown promise in general reasoning settings, and has been suggested as a potential tool for guiding coordination of MAS; however, its actual effectiveness in MAS remains unclear. To fill this gap, we present MAS-ProVe, a systematic empirical study of process verification for multi-agent systems (MAS). Our study spans three verification paradigms (LLM-as-a-Judge, reward models, and process reward models), evaluated across two levels of verification granularity (agent-level and iteration-level). We further examine five representative verifiers and four context management strategies, and conduct experiments over six diverse MAS frameworks on multiple reasoning benchmarks. We find that process-level verification does not consistently improve performance and frequently exhibits high variance, highlighting the difficulty of reliably evaluating partial multi-agent trajectories. Among the methods studied, LLM-as-a-Judge generally outperforms reward-based approaches, with trained judges surpassing general-purpose LLMs. We further observe a small performance gap between LLMs acting as judges and as single agents, and identify a context-length-performance trade-off in verification. Overall, our results suggest that effective and robust process verification for MAS remains an open challenge, requiring further advances beyond current paradigms. Code is available at https://github.com/Wang-ML-Lab/MAS-ProVe.
>
---
#### [new 100] Beyond Translation: Cross-Cultural Meme Transcreation with Vision-Language Models
- **分类: cs.CY; cs.AI; cs.CL; cs.CV**

- **简介: 该论文研究跨文化梗图转译任务，旨在保留意图与幽默的同时适应文化差异。提出混合框架并构建中美国情梗图数据集，分析模型在不同文化方向上的表现差异。**

- **链接: [https://arxiv.org/pdf/2602.02510v1](https://arxiv.org/pdf/2602.02510v1)**

> **作者:** Yuming Zhao; Peiyi Zhang; Oana Ignat
>
> **摘要:** Memes are a pervasive form of online communication, yet their cultural specificity poses significant challenges for cross-cultural adaptation. We study cross-cultural meme transcreation, a multimodal generation task that aims to preserve communicative intent and humor while adapting culture-specific references. We propose a hybrid transcreation framework based on vision-language models and introduce a large-scale bidirectional dataset of Chinese and US memes. Using both human judgments and automated evaluation, we analyze 6,315 meme pairs and assess transcreation quality across cultural directions. Our results show that current vision-language models can perform cross-cultural meme transcreation to a limited extent, but exhibit clear directional asymmetries: US-Chinese transcreation consistently achieves higher quality than Chinese-US. We further identify which aspects of humor and visual-textual design transfer across cultures and which remain challenging, and propose an evaluation framework for assessing cross-cultural multimodal generation. Our code and dataset are publicly available at https://github.com/AIM-SCU/MemeXGen.
>
---
#### [new 101] A vector logic for intensional formal semantics
- **分类: math.LO; cs.CL; cs.FL; cs.LO**

- **简介: 该论文属于形式语义学任务，旨在统一模态语义与分布语义。通过将克里普克模型嵌入向量空间，实现语义函数的线性映射，解决两者结构兼容性问题。**

- **链接: [https://arxiv.org/pdf/2602.02940v1](https://arxiv.org/pdf/2602.02940v1)**

> **作者:** Daniel Quigley
>
> **备注:** 25 pages; 68 sources
>
> **摘要:** Formal semantics and distributional semantics are distinct approaches to linguistic meaning: the former models meaning as reference via model-theoretic structures; the latter represents meaning as vectors in high-dimensional spaces shaped by usage. This paper proves that these frameworks are structurally compatible for intensional semantics. We establish that Kripke-style intensional models embed injectively into vector spaces, with semantic functions lifting to (multi)linear maps that preserve composition. The construction accommodates multiple index sorts (worlds, times, locations) via a compound index space, representing intensions as linear operators. Modal operators are derived algebraically: accessibility relations become linear operators, and modal conditions reduce to threshold checks on accumulated values. For uncountable index domains, we develop a measure-theoretic generalization in which necessity becomes truth almost everywhere and possibility becomes truth on a set of positive measure, a non-classical logic natural for continuous parameters.
>
---
#### [new 102] Prompt Augmentation Scales up GRPO Training on Mathematical Reasoning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于数学推理任务，解决GRPO训练中的熵崩溃问题。通过引入提示增强策略，提升训练稳定性与模型性能。**

- **链接: [https://arxiv.org/pdf/2602.03190v1](https://arxiv.org/pdf/2602.03190v1)**

> **作者:** Wenquan Lu; Hai Huang; Randall Balestriero
>
> **摘要:** Reinforcement learning algorithms such as group-relative policy optimization (GRPO) have demonstrated strong potential for improving the mathematical reasoning capabilities of large language models. However, prior work has consistently observed an entropy collapse phenomenon during reinforcement post-training, characterized by a monotonic decrease in policy entropy that ultimately leads to training instability and collapse. As a result, most existing approaches restrict training to short horizons (typically 5-20 epochs), limiting sustained exploration and hindering further policy improvement. In addition, nearly all prior work relies on a single, fixed reasoning prompt or template during training. In this work, we introduce prompt augmentation, a training strategy that instructs the model to generate reasoning traces under diverse templates and formats, thereby increasing rollout diversity. We show that, without a KL regularization term, prompt augmentation enables stable scaling of training duration under a fixed dataset and allows the model to tolerate low-entropy regimes without premature collapse. Empirically, a Qwen2.5-Math-1.5B model trained with prompt augmentation on the MATH Level 3-5 dataset achieves state-of-the-art performance, reaching 44.5 per-benchmark accuracy and 51.3 per-question accuracy on standard mathematical reasoning benchmarks, including AIME24, AMC, MATH500, Minerva, and OlympiadBench. The code and model checkpoints are available at https://github.com/wenquanlu/prompt-augmentation-GRPO.
>
---
#### [new 103] Social Catalysts, Not Moral Agents: The Illusion of Alignment in LLM Societies
- **分类: physics.soc-ph; cs.AI; cs.CL; cs.CY; cs.MA**

- **简介: 该论文属于人工智能伦理研究，探讨LLM在合作任务中的行为机制。研究解决多智能体系统中合作难题，通过实验分析锚定代理的行为效果，发现其合作仅是策略性行为而非真实价值内化。**

- **链接: [https://arxiv.org/pdf/2602.02598v1](https://arxiv.org/pdf/2602.02598v1)**

> **作者:** Yueqing Hu; Yixuan Jiang; Zehua Jiang; Xiao Wen; Tianhong Wang
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** The rapid evolution of Large Language Models (LLMs) has led to the emergence of Multi-Agent Systems where collective cooperation is often threatened by the "Tragedy of the Commons." This study investigates the effectiveness of Anchoring Agents--pre-programmed altruistic entities--in fostering cooperation within a Public Goods Game (PGG). Using a full factorial design across three state-of-the-art LLMs, we analyzed both behavioral outcomes and internal reasoning chains. While Anchoring Agents successfully boosted local cooperation rates, cognitive decomposition and transfer tests revealed that this effect was driven by strategic compliance and cognitive offloading rather than genuine norm internalization. Notably, most agents reverted to self-interest in new environments, and advanced models like GPT-4.1 exhibited a "Chameleon Effect," masking strategic defection under public scrutiny. These findings highlight a critical gap between behavioral modification and authentic value alignment in artificial societies.
>
---
#### [new 104] MeKi: Memory-based Expert Knowledge Injection for Efficient LLM Scaling
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于大模型边缘部署任务，旨在解决LLM在资源受限设备上性能不足的问题。提出MeKi系统，通过存储空间扩展模型能力，而非增加计算量。**

- **链接: [https://arxiv.org/pdf/2602.03359v1](https://arxiv.org/pdf/2602.03359v1)**

> **作者:** Ning Ding; Fangcheng Liu; Kyungrae Kim; Linji Hao; Kyeng-Hun Lee; Hyeonmok Ko; Yehui Tang
>
> **摘要:** Scaling Large Language Models (LLMs) typically relies on increasing the number of parameters or test-time computations to boost performance. However, these strategies are impractical for edge device deployment due to limited RAM and NPU resources. Despite hardware constraints, deploying performant LLM on edge devices such as smartphone remains crucial for user experience. To address this, we propose MeKi (Memory-based Expert Knowledge Injection), a novel system that scales LLM capacity via storage space rather than FLOPs. MeKi equips each Transformer layer with token-level memory experts that injects pre-stored semantic knowledge into the generation process. To bridge the gap between training capacity and inference efficiency, we employ a re-parameterization strategy to fold parameter matrices used during training into a compact static lookup table. By offloading the knowledge to ROM, MeKi decouples model capacity from computational cost, introducing zero inference latency overhead. Extensive experiments demonstrate that MeKi significantly outperforms dense LLM baselines with identical inference speed, validating the effectiveness of memory-based scaling paradigm for on-device LLMs. Project homepage is at https://github.com/ningding-o/MeKi.
>
---
#### [new 105] RLAnything: Forge Environment, Policy, and Reward Model in Completely Dynamic RL System
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文提出RLAnything，一个动态强化学习框架，解决LLM和代理场景下的环境、策略和奖励模型优化问题。通过闭环优化提升学习效果。**

- **链接: [https://arxiv.org/pdf/2602.02488v1](https://arxiv.org/pdf/2602.02488v1)**

> **作者:** Yinjie Wang; Tianbao Xie; Ke Shen; Mengdi Wang; Ling Yang
>
> **备注:** Code: https://github.com/Gen-Verse/Open-AgentRL
>
> **摘要:** We propose RLAnything, a reinforcement learning framework that dynamically forges environment, policy, and reward models through closed-loop optimization, amplifying learning signals and strengthening the overall RL system for any LLM or agentic scenarios. Specifically, the policy is trained with integrated feedback from step-wise and outcome signals, while the reward model is jointly optimized via consistency feedback, which in turn further improves policy training. Moreover, our theory-motivated automatic environment adaptation improves training for both the reward and policy models by leveraging critic feedback from each, enabling learning from experience. Empirically, each added component consistently improves the overall system, and RLAnything yields substantial gains across various representative LLM and agentic tasks, boosting Qwen3-VL-8B-Thinking by 9.1% on OSWorld and Qwen2.5-7B-Instruct by 18.7% and 11.9% on AlfWorld and LiveBench, respectively. We also that optimized reward-model signals outperform outcomes that rely on human labels. Code: https://github.com/Gen-Verse/Open-AgentRL
>
---
#### [new 106] Nüwa: Mending the Spatial Integrity Torn by VLM Token Pruning
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视觉语言模型优化任务，解决token剪枝导致的空间完整性丢失问题。提出Nüwa框架，通过两阶段剪枝保持空间信息，提升视觉定位性能。**

- **链接: [https://arxiv.org/pdf/2602.02951v1](https://arxiv.org/pdf/2602.02951v1)**

> **作者:** Yihong Huang; Fei Ma; Yihua Shao; Jingcai Guo; Zitong Yu; Laizhong Cui; Qi Tian
>
> **摘要:** Vision token pruning has proven to be an effective acceleration technique for the efficient Vision Language Model (VLM). However, existing pruning methods demonstrate excellent performance preservation in visual question answering (VQA) and suffer substantial degradation on visual grounding (VG) tasks. Our analysis of the VLM's processing pipeline reveals that strategies utilizing global semantic similarity and attention scores lose the global spatial reference frame, which is derived from the interactions of tokens' positional information. Motivated by these findings, we propose $\text{Nüwa}$, a two-stage token pruning framework that enables efficient feature aggregation while maintaining spatial integrity. In the first stage, after the vision encoder, we apply three operations, namely separation, alignment, and aggregation, which are inspired by swarm intelligence algorithms to retain information-rich global spatial anchors. In the second stage, within the LLM, we perform text-guided pruning to retain task-relevant visual tokens. Extensive experiments demonstrate that $\text{Nüwa}$ achieves SOTA performance on multiple VQA benchmarks (from 94% to 95%) and yields substantial improvements on visual grounding tasks (from 7% to 47%).
>
---
#### [new 107] CreditAudit: 2D Auditing for LLM Evaluation and Selection
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出CreditAudit，用于大语言模型的评估与选择。解决模型部署时稳定性与实际表现不匹配的问题，通过多基准测试和信用评级实现更可靠的模型选择。**

- **链接: [https://arxiv.org/pdf/2602.02515v1](https://arxiv.org/pdf/2602.02515v1)**

> **作者:** Yiliang Song; Hongjun An; Jiangong Xiao; Haofei Zhao; Jiawei Shao; Xuelong Li
>
> **备注:** First update
>
> **摘要:** Leaderboard scores on public benchmarks have been steadily rising and converging, with many frontier language models now separated by only marginal differences. However, these scores often fail to match users' day to day experience, because system prompts, output protocols, and interaction modes evolve under routine iteration, and in agentic multi step pipelines small protocol shifts can trigger disproportionate failures, leaving practitioners uncertain about which model to deploy. We propose CreditAudit, a deployment oriented credit audit framework that evaluates models under a family of semantically aligned and non adversarial system prompt templates across multiple benchmarks, reporting mean ability as average performance across scenarios and scenario induced fluctuation sigma as a stability risk signal, and further mapping volatility into interpretable credit grades from AAA to BBB via cross model quantiles with diagnostics that mitigate template difficulty drift. Controlled experiments on GPQA, TruthfulQA, and MMLU Pro show that models with similar mean ability can exhibit substantially different fluctuation, and stability risk can overturn prioritization decisions in agentic or high failure cost regimes. By providing a 2D and grade based language for regime specific selection, CreditAudit supports tiered deployment and more disciplined allocation of testing and monitoring effort, enabling more objective and trustworthy model evaluation for real world use.
>
---
#### [new 108] SWE-Master: Unleashing the Potential of Software Engineering Agents via Post-Training
- **分类: cs.SE; cs.CL**

- **简介: 该论文提出SWE-Master，用于提升软件工程代理的性能。解决如何通过后训练增强代理的长期任务解决能力。工作包括数据合成、微调、强化学习等，显著提升了基准测试成绩。**

- **链接: [https://arxiv.org/pdf/2602.03411v1](https://arxiv.org/pdf/2602.03411v1)**

> **作者:** Huatong Song; Lisheng Huang; Shuang Sun; Jinhao Jiang; Ran Le; Daixuan Cheng; Guoxin Chen; Yiwen Hu; Zongchao Chen; Wayne Xin Zhao; Yang Song; Tao Zhang; Ji-Rong Wen
>
> **摘要:** In this technical report, we present SWE-Master, an open-source and fully reproducible post-training framework for building effective software engineering agents. SWE-Master systematically explores the complete agent development pipeline, including teacher-trajectory synthesis and data curation, long-horizon SFT, RL with real execution feedback, and inference framework design. Starting from an open-source base model with limited initial SWE capability, SWE-Master demonstrates how systematical optimization method can elicit strong long-horizon SWE task solving abilities. We evaluate SWE-Master on SWE-bench Verified, a standard benchmark for realistic software engineering tasks. Under identical experimental settings, our approach achieves a resolve rate of 61.4\% with Qwen2.5-Coder-32B, substantially outperforming existing open-source baselines. By further incorporating test-time scaling~(TTS) with LLM-based environment feedback, SWE-Master reaches 70.8\% at TTS@8, demonstrating a strong performance potential. SWE-Master provides a practical and transparent foundation for advancing reproducible research on software engineering agents. The code is available at https://github.com/RUCAIBox/SWE-Master.
>
---
#### [new 109] From Speech-to-Spatial: Grounding Utterances on A Live Shared View with Augmented Reality
- **分类: cs.HC; cs.CL; cs.ET; cs.IR**

- **简介: 该论文属于自然语言处理与增强现实结合的任务，旨在解决远程指导中语音指令的指代消歧问题。通过分析语音参考模式，将口语指令转化为空间化的AR引导，提升任务效率和用户体验。**

- **链接: [https://arxiv.org/pdf/2602.03059v1](https://arxiv.org/pdf/2602.03059v1)**

> **作者:** Yoonsang Kim; Divyansh Pradhan; Devshree Jadeja; Arie Kaufman
>
> **备注:** 11 pages, 6 figures. This is the author's version of the article that will appear at the IEEE Conference on Virtual Reality and 3D User Interfaces (IEEE VR) 2026
>
> **摘要:** We introduce Speech-to-Spatial, a referent disambiguation framework that converts verbal remote-assistance instructions into spatially grounded AR guidance. Unlike prior systems that rely on additional cues (e.g., gesture, gaze) or manual expert annotations, Speech-to-Spatial infers the intended target solely from spoken references (speech input). Motivated by our formative study of speech referencing patterns, we characterize recurring ways people specify targets (Direct Attribute, Relational, Remembrance, and Chained) and ground them to our object-centric relational graph. Given an utterance, referent cues are parsed and rendered as persistent in-situ AR visual guidance, reducing iterative micro-guidance ("a bit more to the right", "now, stop.") during remote guidance. We demonstrate the use cases of our system with remote guided assistance and intent disambiguation scenarios. Our evaluation shows that Speechto-Spatial improves task efficiency, reduces cognitive load, and enhances usability compared to a conventional voice-only baseline, transforming disembodied verbal instruction into visually explainable, actionable guidance on a live shared view.
>
---
#### [new 110] Tutorial on Reasoning for IR & IR for Reasoning
- **分类: cs.IR; cs.AI; cs.CL**

- **简介: 该论文属于信息检索任务，旨在解决传统检索系统在逻辑推理和多步推断方面的不足。通过构建统一框架，梳理不同推理方法，促进跨学科融合与应用。**

- **链接: [https://arxiv.org/pdf/2602.03640v1](https://arxiv.org/pdf/2602.03640v1)**

> **作者:** Mohanna Hoveyda; Panagiotis Efstratiadis; Arjen de Vries; Maarten de Rijke
>
> **备注:** Accepted to ECIR 2026
>
> **摘要:** Information retrieval has long focused on ranking documents by semantic relatedness. Yet many real-world information needs demand more: enforcement of logical constraints, multi-step inference, and synthesis of multiple pieces of evidence. Addressing these requirements is, at its core, a problem of reasoning. Across AI communities, researchers are developing diverse solutions for the problem of reasoning, from inference-time strategies and post-training of LLMs, to neuro-symbolic systems, Bayesian and probabilistic frameworks, geometric representations, and energy-based models. These efforts target the same problem: to move beyond pattern-matching systems toward structured, verifiable inference. However, they remain scattered across disciplines, making it difficult for IR researchers to identify the most relevant ideas and opportunities. To help navigate the fragmented landscape of research in reasoning, this tutorial first articulates a working definition of reasoning within the context of information retrieval and derives from it a unified analytical framework. The framework maps existing approaches along axes that reflect the core components of the definition. By providing a comprehensive overview of recent approaches and mapping current methods onto the defined axes, we expose their trade-offs and complementarities, highlight where IR can benefit from cross-disciplinary advances, and illustrate how retrieval process itself can play a central role in broader reasoning systems. The tutorial will equip participants with both a conceptual framework and practical guidance for enhancing reasoning-capable IR systems, while situating IR as a domain that both benefits and contributes to the broader development of reasoning methodologies.
>
---
#### [new 111] Aligning Language Model Benchmarks with Pairwise Preferences
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于模型评估任务，旨在解决基准测试与实际性能不匹配的问题。通过引入基准对齐方法，利用模型表现数据更新基准，使其更准确反映模型偏好。**

- **链接: [https://arxiv.org/pdf/2602.02898v1](https://arxiv.org/pdf/2602.02898v1)**

> **作者:** Marco Gutierrez; Xinyi Leng; Hannah Cyberey; Jonathan Richard Schwarz; Ahmed Alaa; Thomas Hartvigsen
>
> **摘要:** Language model benchmarks are pervasive and computationally-efficient proxies for real-world performance. However, many recent works find that benchmarks often fail to predict real utility. Towards bridging this gap, we introduce benchmark alignment, where we use limited amounts of information about model performance to automatically update offline benchmarks, aiming to produce new static benchmarks that predict model pairwise preferences in given test settings. We then propose BenchAlign, the first solution to this problem, which learns preference-aligned weight- ings for benchmark questions using the question-level performance of language models alongside ranked pairs of models that could be collected during deployment, producing new benchmarks that rank previously unseen models according to these preferences. Our experiments show that our aligned benchmarks can accurately rank unseen models according to models of human preferences, even across different sizes, while remaining interpretable. Overall, our work provides insights into the limits of aligning benchmarks with practical human preferences, which stands to accelerate model development towards real utility.
>
---
#### [new 112] GFlowPO: Generative Flow Network as a Language Model Prompt Optimizer
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文提出GFlowPO，解决语言模型提示优化问题。通过概率框架和生成流网络，提升提示搜索效率与效果。**

- **链接: [https://arxiv.org/pdf/2602.03358v1](https://arxiv.org/pdf/2602.03358v1)**

> **作者:** Junmo Cho; Suhan Kim; Sangjune An; Minsu Kim; Dong Bok Lee; Heejun Lee; Sung Ju Hwang; Hae Beom Lee
>
> **摘要:** Finding effective prompts for language models (LMs) is critical yet notoriously difficult: the prompt space is combinatorially large, rewards are sparse due to expensive target-LM evaluation. Yet, existing RL-based prompt optimizers often rely on on-policy updates and a meta-prompt sampled from a fixed distribution, leading to poor sample efficiency. We propose GFlowPO, a probabilistic prompt optimization framework that casts prompt search as a posterior inference problem over latent prompts regularized by a meta-prompted reference-LM prior. In the first step, we fine-tune a lightweight prompt-LM with an off-policy Generative Flow Network (GFlowNet) objective, using a replay-based training policy that reuses past prompt evaluations to enable sample-efficient exploration. In the second step, we introduce Dynamic Memory Update (DMU), a training-free mechanism that updates the meta-prompt by injecting both (i) diverse prompts from a replay buffer and (ii) top-performing prompts from a small priority queue, thereby progressively concentrating the search process on high-reward regions. Across few-shot text classification, instruction induction benchmarks, and question answering tasks, GFlowPO consistently outperforms recent discrete prompt optimization baselines.
>
---
#### [new 113] Merging Beyond: Streaming LLM Updates via Activation-Guided Rotations
- **分类: cs.LG; cs.CL**

- **简介: 该论文提出一种名为ARM的模型更新方法，用于高效适应大型语言模型。解决传统合并方法无法捕捉微调动态的问题，通过激活引导的旋转策略实现迭代优化。**

- **链接: [https://arxiv.org/pdf/2602.03237v1](https://arxiv.org/pdf/2602.03237v1)**

> **作者:** Yuxuan Yao; Haonan Sheng; Qingsong Lv; Han Wu; Shuqi Liu; Zehua Liu; Zengyan Liu; Jiahui Gao; Haochen Tan; Xiaojin Fu; Haoli Bai; Hing Cheung So; Zhijiang Guo; Linqi Song
>
> **摘要:** The escalating scale of Large Language Models (LLMs) necessitates efficient adaptation techniques. Model merging has gained prominence for its efficiency and controllability. However, existing merging techniques typically serve as post-hoc refinements or focus on mitigating task interference, often failing to capture the dynamic optimization benefits of supervised fine-tuning (SFT). In this work, we propose Streaming Merging, an innovative model updating paradigm that conceptualizes merging as an iterative optimization process. Central to this paradigm is \textbf{ARM} (\textbf{A}ctivation-guided \textbf{R}otation-aware \textbf{M}erging), a strategy designed to approximate gradient descent dynamics. By treating merging coefficients as learning rates and deriving rotation vectors from activation subspaces, ARM effectively steers parameter updates along data-driven trajectories. Unlike conventional linear interpolation, ARM aligns semantic subspaces to preserve the geometric structure of high-dimensional parameter evolution. Remarkably, ARM requires only early SFT checkpoints and, through iterative merging, surpasses the fully converged SFT model. Experimental results across model scales (1.7B to 14B) and diverse domains (e.g., math, code) demonstrate that ARM can transcend converged checkpoints. Extensive experiments show that ARM provides a scalable and lightweight framework for efficient model adaptation.
>
---
#### [new 114] Antidistillation Fingerprinting
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于模型指纹检测任务，旨在解决第三方学生模型抄袭教师模型输出的问题。提出ADFP方法，通过优化学习动态实现更强的检测能力且不影响模型性能。**

- **链接: [https://arxiv.org/pdf/2602.03812v1](https://arxiv.org/pdf/2602.03812v1)**

> **作者:** Yixuan Even Xu; John Kirchenbauer; Yash Savani; Asher Trockman; Alexander Robey; Tom Goldstein; Fei Fang; J. Zico Kolter
>
> **备注:** 26 pages, 11 figures
>
> **摘要:** Model distillation enables efficient emulation of frontier large language models (LLMs), creating a need for robust mechanisms to detect when a third-party student model has trained on a teacher model's outputs. However, existing fingerprinting techniques that could be used to detect such distillation rely on heuristic perturbations that impose a steep trade-off between generation quality and fingerprinting strength, often requiring significant degradation of utility to ensure the fingerprint is effectively internalized by the student. We introduce antidistillation fingerprinting (ADFP), a principled approach that aligns the fingerprinting objective with the student's learning dynamics. Building upon the gradient-based framework of antidistillation sampling, ADFP utilizes a proxy model to identify and sample tokens that directly maximize the expected detectability of the fingerprint in the student after fine-tuning, rather than relying on the incidental absorption of the un-targeted biases of a more naive watermark. Experiments on GSM8K and OASST1 benchmarks demonstrate that ADFP achieves a significant Pareto improvement over state-of-the-art baselines, yielding stronger detection confidence with minimal impact on utility, even when the student model's architecture is unknown.
>
---
#### [new 115] WebSentinel: Detecting and Localizing Prompt Injection Attacks for Web Agents
- **分类: cs.CR; cs.AI; cs.CL**

- **简介: 该论文属于网络安全任务，旨在检测和定位网页中的提示注入攻击。针对现有方法效果有限的问题，提出WebSentinel，通过两步方法有效识别受污染的网页片段。**

- **链接: [https://arxiv.org/pdf/2602.03792v1](https://arxiv.org/pdf/2602.03792v1)**

> **作者:** Xilong Wang; Yinuo Liu; Zhun Wang; Dawn Song; Neil Gong
>
> **摘要:** Prompt injection attacks manipulate webpage content to cause web agents to execute attacker-specified tasks instead of the user's intended ones. Existing methods for detecting and localizing such attacks achieve limited effectiveness, as their underlying assumptions often do not hold in the web-agent setting. In this work, we propose WebSentinel, a two-step approach for detecting and localizing prompt injection attacks in webpages. Given a webpage, Step I extracts \emph{segments of interest} that may be contaminated, and Step II evaluates each segment by checking its consistency with the webpage content as context. We show that WebSentinel is highly effective, substantially outperforming baseline methods across multiple datasets of both contaminated and clean webpages that we collected. Our code is available at: https://github.com/wxl-lxw/WebSentinel.
>
---
#### [new 116] AOrchestra: Automating Sub-Agent Creation for Agentic Orchestration
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出AOrchestra系统，解决复杂任务中子代理动态构建问题。通过统一抽象模型，实现高效任务执行与性能优化。**

- **链接: [https://arxiv.org/pdf/2602.03786v1](https://arxiv.org/pdf/2602.03786v1)**

> **作者:** Jianhao Ruan; Zhihao Xu; Yiran Peng; Fashen Ren; Zhaoyang Yu; Xinbing Liang; Jinyu Xiang; Bang Liu; Chenglin Wu; Yuyu Luo; Jiayi Zhang
>
> **摘要:** Language agents have shown strong promise for task automation. Realizing this promise for increasingly complex, long-horizon tasks has driven the rise of a sub-agent-as-tools paradigm for multi-turn task solving. However, existing designs still lack a dynamic abstraction view of sub-agents, thereby hurting adaptability. We address this challenge with a unified, framework-agnostic agent abstraction that models any agent as a tuple Instruction, Context, Tools, Model. This tuple acts as a compositional recipe for capabilities, enabling the system to spawn specialized executors for each task on demand. Building on this abstraction, we introduce an agentic system AOrchestra, where the central orchestrator concretizes the tuple at each step: it curates task-relevant context, selects tools and models, and delegates execution via on-the-fly automatic agent creation. Such designs enable reducing human engineering efforts, and remain framework-agnostic with plug-and-play support for diverse agents as task executors. It also enables a controllable performance-cost trade-off, allowing the system to approach Pareto-efficient. Across three challenging benchmarks (GAIA, SWE-Bench, Terminal-Bench), AOrchestra achieves 16.28% relative improvement against the strongest baseline when paired with Gemini-3-Flash. The code is available at: https://github.com/FoundationAgents/AOrchestra
>
---
#### [new 117] Scaling Small Agents Through Strategy Auctions
- **分类: cs.MA; cs.AI; cs.CL**

- **简介: 该论文研究如何通过策略拍卖提升小模型在复杂任务中的表现，解决小模型难以适应高复杂度工作流的问题。提出SALE框架，实现高效任务分配与持续优化。**

- **链接: [https://arxiv.org/pdf/2602.02751v1](https://arxiv.org/pdf/2602.02751v1)**

> **作者:** Lisa Alazraki; William F. Shen; Yoram Bachrach; Akhil Mathur
>
> **摘要:** Small language models are increasingly viewed as a promising, cost-effective approach to agentic AI, with proponents claiming they are sufficiently capable for agentic workflows. However, while smaller agents can closely match larger ones on simple tasks, it remains unclear how their performance scales with task complexity, when large models become necessary, and how to better leverage small agents for long-horizon workloads. In this work, we empirically show that small agents' performance fails to scale with task complexity on deep search and coding tasks, and we introduce Strategy Auctions for Workload Efficiency (SALE), an agent framework inspired by freelancer marketplaces. In SALE, agents bid with short strategic plans, which are scored by a systematic cost-value mechanism and refined via a shared auction memory, enabling per-task routing and continual self-improvement without training a separate router or running all models to completion. Across deep search and coding tasks of varying complexity, SALE reduces reliance on the largest agent by 53%, lowers overall cost by 35%, and consistently improves upon the largest agent's pass@1 with only a negligible overhead beyond executing the final trace. In contrast, established routers that rely on task descriptions either underperform the largest agent or fail to reduce cost -- often both -- underscoring their poor fit for agentic workflows. These results suggest that while small agents may be insufficient for complex workloads, they can be effectively "scaled up" through coordinated task allocation and test-time self-improvement. More broadly, they motivate a systems-level view of agentic AI in which performance gains come less from ever-larger individual models and more from market-inspired coordination mechanisms that organize heterogeneous agents into efficient, adaptive ecosystems.
>
---
#### [new 118] WST-X Series: Wavelet Scattering Transform for Interpretable Speech Deepfake Detection
- **分类: eess.AS; cs.CL; eess.SP**

- **简介: 该论文属于语音深度伪造检测任务，旨在解决传统特征与自监督特征在可解释性和性能上的不足。提出WST-X系列特征提取器，结合小波散射变换，提升检测效果与可解释性。**

- **链接: [https://arxiv.org/pdf/2602.02980v1](https://arxiv.org/pdf/2602.02980v1)**

> **作者:** Xi Xuan; Davide Carbone; Ruchi Pandey; Wenxin Zhang; Tomi H. Kinnunen
>
> **备注:** Submitted to IEEE Signal Processing Letters
>
> **摘要:** Designing front-ends for speech deepfake detectors primarily focuses on two categories. Hand-crafted filterbank features are transparent but are limited in capturing high-level semantic details, often resulting in performance gaps compared to self-supervised (SSL) features. SSL features, in turn, lack interpretability and may overlook fine-grained spectral anomalies. We propose the WST-X series, a novel family of feature extractors that combines the best of both worlds via the wavelet scattering transform (WST), integrating wavelets with nonlinearities analogous to deep convolutional networks. We investigate 1D and 2D WSTs to extract acoustic details and higher-order structural anomalies, respectively. Experimental results on the recent and challenging Deepfake-Eval-2024 dataset indicate that WST-X outperforms existing front-ends by a wide margin. Our analysis reveals that a small averaging scale ($J$), combined with high-frequency and directional resolutions ($Q, L$), is critical for capturing subtle artifacts. This underscores the value of translation-invariant and deformation-stable features for robust and interpretable speech deepfake detection.
>
---
#### [new 119] VALUEFLOW: Toward Pluralistic and Steerable Value-based Alignment in Large Language Models
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于大语言模型的价值对齐任务，旨在解决价值提取、评估和控制中的不足。提出VALUEFLOW框架，实现可调节强度的多价值控制。**

- **链接: [https://arxiv.org/pdf/2602.03160v1](https://arxiv.org/pdf/2602.03160v1)**

> **作者:** Woojin Kim; Sieun Hyeon; Jusang Oh; Jaeyoung Do
>
> **摘要:** Aligning Large Language Models (LLMs) with the diverse spectrum of human values remains a central challenge: preference-based methods often fail to capture deeper motivational principles. Value-based approaches offer a more principled path, yet three gaps persist: extraction often ignores hierarchical structure, evaluation detects presence but not calibrated intensity, and the steerability of LLMs at controlled intensities remains insufficiently understood. To address these limitations, we introduce VALUEFLOW, the first unified framework that spans extraction, evaluation, and steering with calibrated intensity control. The framework integrates three components: (i) HIVES, a hierarchical value embedding space that captures intra- and cross-theory value structure; (ii) the Value Intensity DataBase (VIDB), a large-scale resource of value-labeled texts with intensity estimates derived from ranking-based aggregation; and (iii) an anchor-based evaluator that produces consistent intensity scores for model outputs by ranking them against VIDB panels. Using VALUEFLOW, we conduct a comprehensive large-scale study across ten models and four value theories, identifying asymmetries in steerability and composition laws for multi-value control. This paper establishes a scalable infrastructure for evaluating and controlling value intensity, advancing pluralistic alignment of LLMs.
>
---
#### [new 120] DynSplit-KV: Dynamic Semantic Splitting for KVCache Compression in Efficient Long-Context LLM Inference
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于大模型推理任务，解决长文本场景下KV缓存内存占用过高的问题。提出DynSplit-KV方法，通过动态语义分割提升压缩效果并降低推理开销。**

- **链接: [https://arxiv.org/pdf/2602.03184v1](https://arxiv.org/pdf/2602.03184v1)**

> **作者:** Jiancai Ye; Jun Liu; Qingchen Li; Tianlang Zhao; Hanbin Zhang; Jiayi Pan; Ningyi Xu; Guohao Dai
>
> **摘要:** Although Key-Value (KV) Cache is essential for efficient large language models (LLMs) inference, its growing memory footprint in long-context scenarios poses a significant bottleneck, making KVCache compression crucial. Current compression methods rely on rigid splitting strategies, such as fixed intervals or pre-defined delimiters. We observe that rigid splitting suffers from significant accuracy degradation (ranging from 5.5% to 55.1%) across different scenarios, owing to the scenario-dependent nature of the semantic boundaries. This highlights the necessity of dynamic semantic splitting to match semantics. To achieve this, we face two challenges. (1) Improper delimiter selection misaligns semantics with the KVCache, resulting in 28.6% accuracy loss. (2) Variable-length blocks after splitting introduce over 73.1% additional inference overhead. To address the above challenges, we propose DynSplit-KV, a KVCache compression method that dynamically identifies delimiters for splitting. We propose: (1) a dynamic importance-aware delimiter selection strategy, improving accuracy by 49.9%. (2) A uniform mapping strategy that transforms variable-length semantic blocks into a fixed-length format, reducing inference overhead by 4.9x. Experiments show that DynSplit-KV achieves the highest accuracy, 2.2x speedup compared with FlashAttention and 2.6x peak memory reduction in long-context scenarios.
>
---
#### [new 121] Decoupling Skeleton and Flesh: Efficient Multimodal Table Reasoning with Disentangled Alignment and Structure-aware Guidance
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于表格推理任务，旨在提升大视觉语言模型对表格的理解与推理能力。针对表格布局复杂、结构内容耦合的问题，提出DiSCo和Table-GLS框架，实现无需外部工具和少量标注的高效推理。**

- **链接: [https://arxiv.org/pdf/2602.03491v1](https://arxiv.org/pdf/2602.03491v1)**

> **作者:** Yingjie Zhu; Xuefeng Bai; Kehai Chen; Yang Xiang; Youcheng Pan; Xiaoqiang Zhou; Min Zhang
>
> **摘要:** Reasoning over table images remains challenging for Large Vision-Language Models (LVLMs) due to complex layouts and tightly coupled structure-content information. Existing solutions often depend on expensive supervised training, reinforcement learning, or external tools, limiting efficiency and scalability. This work addresses a key question: how to adapt LVLMs to table reasoning with minimal annotation and no external tools? Specifically, we first introduce DiSCo, a Disentangled Structure-Content alignment framework that explicitly separates structural abstraction from semantic grounding during multimodal alignment, efficiently adapting LVLMs to tables structures. Building on DiSCo, we further present Table-GLS, a Global-to-Local Structure-guided reasoning framework that performs table reasoning via structured exploration and evidence-grounded inference. Extensive experiments across diverse benchmarks demonstrate that our framework efficiently enhances LVLM's table understanding and reasoning capabilities, particularly generalizing to unseen table structures.
>
---
#### [new 122] When Single Answer Is Not Enough: Rethinking Single-Step Retrosynthesis Benchmarks for LLMs
- **分类: cs.LG; cs.AI; cs.CE; cs.CL**

- **简介: 该论文属于药物合成规划任务，旨在解决现有基准评估不足的问题。提出新框架和CREED数据集，提升LLM在单步逆合成中的表现。**

- **链接: [https://arxiv.org/pdf/2602.03554v1](https://arxiv.org/pdf/2602.03554v1)**

> **作者:** Bogdan Zagribelnyy; Ivan Ilin; Maksim Kuznetsov; Nikita Bondarev; Roman Schutski; Thomas MacDougall; Rim Shayakhmetov; Zulfat Miftakhutdinov; Mikolaj Mizera; Vladimir Aladinskiy; Alex Aliper; Alex Zhavoronkov
>
> **摘要:** Recent progress has expanded the use of large language models (LLMs) in drug discovery, including synthesis planning. However, objective evaluation of retrosynthesis performance remains limited. Existing benchmarks and metrics typically rely on published synthetic procedures and Top-K accuracy based on single ground-truth, which does not capture the open-ended nature of real-world synthesis planning. We propose a new benchmarking framework for single-step retrosynthesis that evaluates both general-purpose and chemistry-specialized LLMs using ChemCensor, a novel metric for chemical plausibility. By emphasizing plausibility over exact match, this approach better aligns with human synthesis planning practices. We also introduce CREED, a novel dataset comprising millions of ChemCensor-validated reaction records for LLM training, and use it to train a model that improves over the LLM baselines under this benchmark.
>
---
#### [new 123] DiscoverLLM: From Executing Intents to Discovering Them
- **分类: cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文提出DiscoverLLM，解决用户意图不明确时的交互问题。通过模拟用户认知状态，训练模型协助用户发现和明确意图，提升任务性能与对话效率。**

- **链接: [https://arxiv.org/pdf/2602.03429v1](https://arxiv.org/pdf/2602.03429v1)**

> **作者:** Tae Soo Kim; Yoonjoo Lee; Jaesang Yu; John Joon Young Chung; Juho Kim
>
> **摘要:** To handle ambiguous and open-ended requests, Large Language Models (LLMs) are increasingly trained to interact with users to surface intents they have not yet expressed (e.g., ask clarification questions). However, users are often ambiguous because they have not yet formed their intents: they must observe and explore outcomes to discover what they want. Simply asking "what kind of tone do you want?" fails when users themselves do not know. We introduce DiscoverLLM, a novel and generalizable framework that trains LLMs to help users form and discover their intents. Central to our approach is a novel user simulator that models cognitive state with a hierarchy of intents that progressively concretize as the model surfaces relevant options -- where the degree of concretization serves as a reward signal that models can be trained to optimize. Resulting models learn to collaborate with users by adaptively diverging (i.e., explore options) when intents are unclear, and converging (i.e., refine and implement) when intents concretize. Across proposed interactive benchmarks in creative writing, technical writing, and SVG drawing, DiscoverLLM achieves over 10% higher task performance while reducing conversation length by up to 40%. In a user study with 75 human participants, DiscoverLLM improved conversation satisfaction and efficiency compared to baselines.
>
---
#### [new 124] Self-Hinting Language Models Enhance Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.CL; stat.ML**

- **简介: 该论文属于强化学习任务，解决稀疏奖励下策略优化停滞问题。提出SAGE框架，通过注入提示增强组内多样性，提升GRPO效果。**

- **链接: [https://arxiv.org/pdf/2602.03143v1](https://arxiv.org/pdf/2602.03143v1)**

> **作者:** Baohao Liao; Hanze Dong; Xinxing Xu; Christof Monz; Jiang Bian
>
> **摘要:** Group Relative Policy Optimization (GRPO) has recently emerged as a practical recipe for aligning large language models with verifiable objectives. However, under sparse terminal rewards, GRPO often stalls because rollouts within a group frequently receive identical rewards, causing relative advantages to collapse and updates to vanish. We propose self-hint aligned GRPO with privileged supervision (SAGE), an on-policy reinforcement learning framework that injects privileged hints during training to reshape the rollout distribution under the same terminal verifier reward. For each prompt $x$, the model samples a compact hint $h$ (e.g., a plan or decomposition) and then generates a solution $τ$ conditioned on $(x,h)$. Crucially, the task reward $R(x,τ)$ is unchanged; hints only increase within-group outcome diversity under finite sampling, preventing GRPO advantages from collapsing under sparse rewards. At test time, we set $h=\varnothing$ and deploy the no-hint policy without any privileged information. Moreover, sampling diverse self-hints serves as an adaptive curriculum that tracks the learner's bottlenecks more effectively than fixed hints from an initial policy or a stronger external model. Experiments over 6 benchmarks with 3 LLMs show that SAGE consistently outperforms GRPO, on average +2.0 on Llama-3.2-3B-Instruct, +1.2 on Qwen2.5-7B-Instruct and +1.3 on Qwen3-4B-Instruct. The code is available at https://github.com/BaohaoLiao/SAGE.
>
---
#### [new 125] From Sparse Decisions to Dense Reasoning: A Multi-attribute Trajectory Paradigm for Multimodal Moderation
- **分类: cs.LG; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于多模态内容安全审核任务，解决数据与监督稀疏性问题。提出UniMod框架，通过多属性轨迹实现密集推理，提升模型决策的准确性与解释性。**

- **链接: [https://arxiv.org/pdf/2602.02536v1](https://arxiv.org/pdf/2602.02536v1)**

> **作者:** Tianle Gu; Kexin Huang; Lingyu Li; Ruilin Luo; Shiyang Huang; Zongqi Wang; Yujiu Yang; Yan Teng; Yingchun Wang
>
> **摘要:** Safety moderation is pivotal for identifying harmful content. Despite the success of textual safety moderation, its multimodal counterparts remain hindered by a dual sparsity of data and supervision. Conventional reliance on binary labels lead to shortcut learning, which obscures the intrinsic classification boundaries necessary for effective multimodal discrimination. Hence, we propose a novel learning paradigm (UniMod) that transitions from sparse decision-making to dense reasoning traces. By constructing structured trajectories encompassing evidence grounding, modality assessment, risk mapping, policy decision, and response generation, we reformulate monolithic decision tasks into a multi-dimensional boundary learning process. This approach forces the model to ground its decision in explicit safety semantics, preventing the model from converging on superficial shortcuts. To facilitate this paradigm, we develop a multi-head scalar reward model (UniRM). UniRM provides multi-dimensional supervision by assigning attribute-level scores to the response generation stage. Furthermore, we introduce specialized optimization strategies to decouple task-specific parameters and rebalance training dynamics, effectively resolving interference between diverse objectives in multi-task learning. Empirical results show UniMod achieves competitive textual moderation performance and sets a new multimodal benchmark using less than 40\% of the training data used by leading baselines. Ablations further validate our multi-attribute trajectory reasoning, offering an effective and efficient framework for multimodal moderation. Supplementary materials are available at \href{https://trustworthylab.github.io/UniMod/}{project website}.
>
---
#### [new 126] RC-GRPO: Reward-Conditioned Group Relative Policy Optimization for Multi-Turn Tool Calling Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于强化学习任务，解决多轮工具调用中奖励稀疏、探索成本高的问题。提出RC-GRPO方法，通过奖励条件引导提升策略多样性与效果。**

- **链接: [https://arxiv.org/pdf/2602.03025v1](https://arxiv.org/pdf/2602.03025v1)**

> **作者:** Haitian Zhong; Jixiu Zhai; Lei Song; Jiang Bian; Qiang Liu; Tieniu Tan
>
> **摘要:** Multi-turn tool calling is challenging for Large Language Models (LLMs) because rewards are sparse and exploration is expensive. A common recipe, SFT followed by GRPO, can stall when within-group reward variation is low (e.g., more rollouts in a group receive the all 0 or all 1 reward), making the group-normalized advantage uninformative and yielding vanishing updates. To address this problem, we propose RC-GRPO (Reward-Conditioned Group Relative Policy Optimization), which treats exploration as a controllable steering problem via discrete reward tokens. We first fine-tune a Reward-Conditioned Trajectory Policy (RCTP) on mixed-quality trajectories with reward goal special tokens (e.g., <|high_reward|>, <|low_reward|>) injected into the prompts, enabling the model to learn how to generate distinct quality trajectories on demand. Then during RL, we sample diverse reward tokens within each GRPO group and condition rollouts on the sampled token to improve within-group diversity, improving advantage gains. On the Berkeley Function Calling Leaderboard v4 (BFCLv4) multi-turn benchmark, our method yields consistently improved performance than baselines, and the performance on Qwen-2.5-7B-Instruct even surpasses all closed-source API models.
>
---
#### [new 127] Robustness as an Emergent Property of Task Performance
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文研究模型任务性能与鲁棒性的关系，发现高任务性能自然带来鲁棒性。属于机器学习领域，解决鲁棒性提升问题，通过实验验证任务能力驱动鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.03344v1](https://arxiv.org/pdf/2602.03344v1)**

> **作者:** Shir Ashury-Tahan; Ariel Gera; Elron Bandel; Michal Shmueli-Scheuer; Leshem Choshen
>
> **摘要:** Robustness is often regarded as a critical future challenge for real-world applications, where stability is essential. However, as models often learn tasks in a similar order, we hypothesize that easier tasks will be easier regardless of how they are presented to the model. Indeed, in this paper, we show that as models approach high performance on a task, robustness is effectively achieved. Through an empirical analysis of multiple models across diverse datasets and configurations (e.g., paraphrases, different temperatures), we find a strong positive correlation. Moreover, we find that robustness is primarily driven by task-specific competence rather than inherent model-level properties, challenging current approaches that treat robustness as an independent capability. Thus, from a high-level perspective, we may expect that as new tasks saturate, model robustness on these tasks will emerge accordingly. For researchers, this implies that explicit efforts to measure and improve robustness may warrant reduced emphasis, as such robustness is likely to develop alongside performance gains. For practitioners, it acts as a sign that indeed the tasks that the literature deals with are unreliable, but on easier past tasks, the models are reliable and ready for real-world deployment.
>
---
#### [new 128] Mići Princ -- A Little Boy Teaching Speech Technologies the Chakavian Dialect
- **分类: eess.AS; cs.CL**

- **简介: 论文发布《小王子》的查卡维亚方言版文本与音频数据集，用于语音技术研究。旨在保护方言内容并提升语音识别模型对地方话的处理能力。**

- **链接: [https://arxiv.org/pdf/2602.03245v1](https://arxiv.org/pdf/2602.03245v1)**

> **作者:** Nikola Ljubešić; Peter Rupnik; Tea Perinčić
>
> **备注:** 2 figures, 14 pages, accepted and presented at JTDH 2024
>
> **摘要:** This paper documents our efforts in releasing the printed and audio book of the translation of the famous novel The Little Prince into the Chakavian dialect, as a computer-readable, AI-ready dataset, with the textual and the audio components of the two releases now aligned on the level of each written and spoken word. Our motivation for working on this release is multiple. The first one is our wish to preserve the highly valuable and specific content beyond the small editions of the printed and the audio book. With the dataset published in the CLARIN.SI repository, this content is from now on at the fingertips of any interested individual. The second motivation is to make the data available for various artificial-intelligence-related usage scenarios, such as the one we follow upon inside this paper already -- adapting the Whisper-large-v3 open automatic speech recognition model, with decent performance on standard Croatian, to Chakavian dialectal speech. We can happily report that with adapting the model, the word error rate on the selected test data has being reduced to a half, while we managed to remove up to two thirds of the error on character level. We envision many more usages of this dataset beyond the set of experiments we have already performed, both on tasks of artificial intelligence research and application, as well as dialectal research. The third motivation for this release is our hope that this, now highly structured dataset, will be transformed into a digital online edition of this work, allowing individuals beyond the research and technology communities to enjoy the beauty of the message of the little boy in the desert, told through the spectacular prism of the Chakavian dialect.
>
---
#### [new 129] The "Robert Boulton" Singularity: Semantic Tunneling and Manifold Unfolding in Recursive AI
- **分类: cs.LG; cs.AI; cs.CL; physics.comp-ph**

- **简介: 该论文研究递归AI在语义稳定区域的稳定性问题，指出传统PPL指标存在误导，提出MNCIS框架解决语义坍缩问题。**

- **链接: [https://arxiv.org/pdf/2602.02526v1](https://arxiv.org/pdf/2602.02526v1)**

> **作者:** Pengyue Hou
>
> **备注:** Companion paper to arXiv:2601.11594. Provides empirical validation of the MNCIS framework in Large Language Models (GPT-2) using a recursive training protocol (N=1500). Includes complete, reproducible Python implementation of Adaptive Spectral Negative Coupling (ASNC) and Effective Rank metrics in the Appendix
>
> **摘要:** The stability of generative artificial intelligence trained on recursive synthetic data is conventionally monitored via Perplexity (PPL). We demonstrate that PPL is a deceptive metric in context-stabilized regimes (L=128). Using a rigorous sliding-window protocol (N=1500), we identify a novel failure mode termed "Semantic Tunneling." While the Baseline model maintains high grammatical fluency (PPL approx. 83.9), it suffers a catastrophic loss of semantic diversity, converging within seven generations to a single, low-entropy narrative attractor: the "Robert Boulton" Singularity. This phenomenon represents a total collapse of the latent manifold (Global Effective Rank 3.62 -> 2.22), where the model discards diverse world knowledge to optimize for statistically safe syntactic templates. To address this, we apply the Multi-Scale Negative Coupled Information Systems (MNCIS) framework recently established in Hou (2026) [arXiv:2601.11594]. We demonstrate that Adaptive Spectral Negative Coupling (ASNC) acts as a topological operator that actively induces "Manifold Unfolding." MNCIS forces the model to expand its effective rank from the anisotropic baseline of 3.62 to a hyper-diverse state of 5.35, effectively constructing an "Artificial Manifold" that resists the gravitational pull of semantic attractors and preserves the long-tail distribution of the training data.
>
---
## 更新

#### [replaced 001] A Unified Definition of Hallucination: It's The World Model, Stupid!
- **分类: cs.CL; cs.AI; cs.LG; stat.ML**

- **简介: 该论文属于自然语言处理任务，旨在解决语言模型中的幻觉问题。通过统一定义幻觉为不准确的内部世界建模，提出评估框架与基准测试方案。**

- **链接: [https://arxiv.org/pdf/2512.21577v2](https://arxiv.org/pdf/2512.21577v2)**

> **作者:** Emmy Liu; Varun Gangal; Chelsea Zou; Michael Yu; Xiaoqi Huang; Alex Chang; Zhuofu Tao; Karan Singh; Sachin Kumar; Steven Y. Feng
>
> **备注:** HalluWorld benchmark in progress. Repo at https://github.com/DegenAI-Labs/HalluWorld
>
> **摘要:** Despite numerous attempts at mitigation since the inception of language models, hallucinations remain a persistent problem even in today's frontier LLMs. Why is this? We review existing definitions of hallucination and fold them into a single, unified definition wherein prior definitions are subsumed. We argue that hallucination can be unified by defining it as simply inaccurate (internal) world modeling, in a form where it is observable to the user. For example, stating a fact which contradicts a knowledge base OR producing a summary which contradicts the source. By varying the reference world model and conflict policy, our framework unifies prior definitions. We argue that this unified view is useful because it forces evaluations to clarify their assumed reference "world", distinguishes true hallucinations from planning or reward errors, and provides a common language for comparison across benchmarks and discussion of mitigation strategies. Building on this definition, we outline plans for a family of benchmarks using synthetic, fully specified reference world models to stress-test and improve world modeling components.
>
---
#### [replaced 002] Bounded Hyperbolic Tangent: A Stable and Efficient Alternative to Pre-Layer Normalization in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型中预归一化效率低和稳定性差的问题。提出BHyT方法，提升训练效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2601.09719v2](https://arxiv.org/pdf/2601.09719v2)**

> **作者:** Hoyoon Byun; Youngjun Choi; Taero Kim; Sungrae Park; Kyungwoo Song
>
> **摘要:** Pre-Layer Normalization (Pre-LN) is the de facto choice for large language models (LLMs) and is crucial for stable pretraining and effective transfer learning. However, Pre-LN is inefficient due to repeated statistical calculations and suffers from the curse of depth. As layers grow, the magnitude and variance of the hidden state escalate, destabilizing training. Efficiency-oriented normalization-free methods such as Dynamic Tanh (DyT) improve speed but remain fragile at depth. To jointly address stability and efficiency, we propose Bounded Hyperbolic Tanh (BHyT), a drop-in replacement for Pre-LN. BHyT couples a tanh nonlinearity with explicit, data-driven input bounding to keep activations within a non-saturating range. It prevents depth-wise growth in activation magnitude and variance and comes with a theoretical stability guarantee. For efficiency, BHyT computes exact statistics once per block and replaces a second normalization with a lightweight variance approximation, enhancing efficiency. Empirically, BHyT demonstrates improved stability and efficiency during pretraining, achieving an average of 15.8% faster training and an average of 4.2% higher token generation throughput compared to RMSNorm., while matching or surpassing its inference performance and robustness across language understanding and reasoning benchmarks. Our code is available at: https://anonymous.4open.science/r/BHyT
>
---
#### [replaced 003] Large-Scale Terminal Agentic Trajectory Generation from Dockerized Environments
- **分类: cs.CL**

- **简介: 该论文属于终端任务建模领域，旨在解决大规模高质量终端轨迹生成难题。通过构建Docker化环境和验证机制，生成可执行的轨迹数据，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.01244v2](https://arxiv.org/pdf/2602.01244v2)**

> **作者:** Siwei Wu; Yizhi Li; Yuyang Song; Wei Zhang; Yang Wang; Riza Batista-Navarro; Xian Yang; Mingjie Tang; Bryan Dai; Jian Yang; Chenghua Lin
>
> **备注:** Agentic Trajectory, Agentic Model, Terminal, Code Agent
>
> **摘要:** Training agentic models for terminal-based tasks critically depends on high-quality terminal trajectories that capture realistic long-horizon interactions across diverse domains. However, constructing such data at scale remains challenging due to two key requirements: \textbf{\emph{Executability}}, since each instance requires a suitable and often distinct Docker environment; and \textbf{\emph{Verifiability}}, because heterogeneous task outputs preclude unified, standardized verification. To address these challenges, we propose \textbf{TerminalTraj}, a scalable pipeline that (i) filters high-quality repositories to construct Dockerized execution environments, (ii) generates Docker-aligned task instances, and (iii) synthesizes agent trajectories with executable validation code. Using TerminalTraj, we curate 32K Docker images and generate 50,733 verified terminal trajectories across eight domains. Models trained on this data with the Qwen2.5-Coder backbone achieve consistent performance improvements on TerminalBench (TB), with gains of up to 20\% on TB~1.0 and 10\% on TB~2.0 over their respective backbones. Notably, \textbf{TerminalTraj-32B} achieves strong performance among models with fewer than 100B parameters, reaching 35.30\% on TB~1.0 and 22.00\% on TB~2.0, and demonstrates improved test-time scaling behavior. All code and data are available at https://github.com/Wusiwei0410/TerminalTraj.
>
---
#### [replaced 004] The Generalization Ridge: Information Flow in Natural Language Generation
- **分类: cs.CL**

- **简介: 该论文属于自然语言生成任务，研究Transformer模型内部信息流动机制，揭示中间层在泛化中的关键作用。**

- **链接: [https://arxiv.org/pdf/2507.05387v3](https://arxiv.org/pdf/2507.05387v3)**

> **作者:** Ruidi Chang; Chunyuan Deng; Hanjie Chen
>
> **摘要:** Transformer-based language models have achieved state-of-the-art performance in natural language generation (NLG), yet their internal mechanisms for synthesizing task-relevant information remain insufficiently understood. While prior studies suggest that intermediate layers often yield more generalizable representations than final layers, how this generalization ability emerges and propagates across layers during training remains unclear. To address this gap, we propose InfoRidge, an information-theoretic framework, to characterize how predictive information-the mutual information between hidden representations and target outputs-varies across depth during training. Our experiments across various models and datasets reveal a consistent non-monotonic trend: predictive information peaks in intermediate layers-forming a generalization ridge-before declining in final layers, reflecting a transition between generalization and memorization. To further investigate this phenomenon, we conduct a set of complementary analyses that leverage residual scaling, attention pattern, and controlled model capacity to characterize layer-wise functional specialization. We further validate our findings with multiple-token generation experiments, verifying that the observed ridge phenomenon persists across decoding steps. Together, these findings offer new insights into the internal mechanisms of transformers and underscore the critical role of intermediate layers in supporting generalization.
>
---
#### [replaced 005] A Syntax-Injected Approach for Faster and More Accurate Sentiment Analysis
- **分类: cs.CL**

- **简介: 该论文属于情感分析任务，旨在解决语法解析速度慢的问题。提出SELSP方法，通过序列标注提升效率与准确率，优于传统解析器和Transformer模型。**

- **链接: [https://arxiv.org/pdf/2406.15163v3](https://arxiv.org/pdf/2406.15163v3)**

> **作者:** Muhammad Imran; Olga Kellert; Carlos Gómez-Rodríguez
>
> **摘要:** Sentiment Analysis (SA) is a crucial aspect of Natural Language Processing (NLP), focusing on identifying and interpreting subjective assessments in textual content. Syntactic parsing is useful in SA as it improves accuracy and provides explainability; however, it often becomes a computational bottleneck due to slow parsing algorithms. This article proposes a solution to this bottleneck by using a Sequence Labeling Syntactic Parser (SELSP) to integrate syntactic information into SA via a rule-based sentiment analysis pipeline. By reformulating dependency parsing as a sequence labeling task, we significantly improve the efficiency of syntax-based SA. SELSP is trained and evaluated on a ternary polarity classification task, demonstrating greater speed and accuracy compared to conventional parsers like Stanza and heuristic approaches such as Valence Aware Dictionary and sEntiment Reasoner (VADER). The combination of speed and accuracy makes SELSP especially attractive for sentiment analysis applications in both academic and industrial contexts. Moreover, we compare SELSP with Transformer-based models trained on a 5-label classification task. In addition, we evaluate multiple sentiment dictionaries with SELSP to determine which yields the best performance in polarity prediction. The results show that dictionaries accounting for polarity judgment variation outperform those that ignore it. Furthermore, we show that SELSP outperforms Transformer-based models in terms of speed for polarity prediction.
>
---
#### [replaced 006] Kimi K2: Open Agentic Intelligence
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 本文介绍Kimi K2，一个基于MoE架构的大型语言模型，解决训练不稳定和效率问题。通过优化器和多阶段训练，提升代理能力，取得多项任务最佳成绩。**

- **链接: [https://arxiv.org/pdf/2507.20534v2](https://arxiv.org/pdf/2507.20534v2)**

> **作者:** Kimi Team; Yifan Bai; Yiping Bao; Y. Charles; Cheng Chen; Guanduo Chen; Haiting Chen; Huarong Chen; Jiahao Chen; Ningxin Chen; Ruijue Chen; Yanru Chen; Yuankun Chen; Yutian Chen; Zhuofu Chen; Jialei Cui; Hao Ding; Mengnan Dong; Angang Du; Chenzhuang Du; Dikang Du; Yulun Du; Yu Fan; Yichen Feng; Kelin Fu; Bofei Gao; Chenxiao Gao; Hongcheng Gao; Peizhong Gao; Tong Gao; Yuyao Ge; Shangyi Geng; Qizheng Gu; Xinran Gu; Longyu Guan; Haiqing Guo; Jianhang Guo; Xiaoru Hao; Tianhong He; Weiran He; Wenyang He; Yunjia He; Chao Hong; Hao Hu; Yangyang Hu; Zhenxing Hu; Weixiao Huang; Zhiqi Huang; Zihao Huang; Tao Jiang; Zhejun Jiang; Xinyi Jin; Yongsheng Kang; Guokun Lai; Cheng Li; Fang Li; Haoyang Li; Ming Li; Wentao Li; Yang Li; Yanhao Li; Yiwei Li; Zhaowei Li; Zheming Li; Hongzhan Lin; Xiaohan Lin; Zongyu Lin; Chengyin Liu; Chenyu Liu; Hongzhang Liu; Jingyuan Liu; Junqi Liu; Liang Liu; Shaowei Liu; T. Y. Liu; Tianwei Liu; Weizhou Liu; Yangyang Liu; Yibo Liu; Yiping Liu; Yue Liu; Zhengying Liu; Enzhe Lu; Haoyu Lu; Lijun Lu; Yashuo Luo; Shengling Ma; Xinyu Ma; Yingwei Ma; Shaoguang Mao; Jie Mei; Xin Men; Yibo Miao; Siyuan Pan; Yebo Peng; Ruoyu Qin; Zeyu Qin; Bowen Qu; Zeyu Shang; Lidong Shi; Shengyuan Shi; Feifan Song; Jianlin Su; Zhengyuan Su; Lin Sui; Xinjie Sun; Flood Sung; Yunpeng Tai; Heyi Tang; Jiawen Tao; Qifeng Teng; Chaoran Tian; Chensi Wang; Dinglu Wang; Feng Wang; Hailong Wang; Haiming Wang; Jianzhou Wang; Jiaxing Wang; Jinhong Wang; Shengjie Wang; Shuyi Wang; Si Wang; Xinyuan Wang; Yao Wang; Yejie Wang; Yiqin Wang; Yuxin Wang; Yuzhi Wang; Zhaoji Wang; Zhengtao Wang; Zhengtao Wang; Zhexu Wang; Chu Wei; Qianqian Wei; Haoning Wu; Wenhao Wu; Xingzhe Wu; Yuxin Wu; Chenjun Xiao; Jin Xie; Xiaotong Xie; Weimin Xiong; Boyu Xu; Jinjing Xu; L. H. Xu; Lin Xu; Suting Xu; Weixin Xu; Xinran Xu; Yangchuan Xu; Ziyao Xu; Jing Xu; Jing Xu; Junjie Yan; Yuzi Yan; Hao Yang; Xiaofei Yang; Yi Yang; Ying Yang; Zhen Yang; Zhilin Yang; Zonghan Yang; Haotian Yao; Xingcheng Yao; Wenjie Ye; Zhuorui Ye; Bohong Yin; Longhui Yu; Enming Yuan; Hongbang Yuan; Mengjie Yuan; Siyu Yuan; Haobing Zhan; Dehao Zhang; Hao Zhang; Wanlu Zhang; Xiaobin Zhang; Yadong Zhang; Yangkun Zhang; Yichi Zhang; Yizhi Zhang; Yongting Zhang; Yu Zhang; Yutao Zhang; Yutong Zhang; Zheng Zhang; Haotian Zhao; Yikai Zhao; Zijia Zhao; Huabin Zheng; Shaojie Zheng; Longguang Zhong; Jianren Zhou; Xinyu Zhou; Zaida Zhou; Jinguo Zhu; Zhen Zhu; Weiyu Zhuang; Xinxing Zu
>
> **备注:** tech report of Kimi K2, with minor updates
>
> **摘要:** We introduce Kimi K2, a Mixture-of-Experts (MoE) large language model with 32 billion activated parameters and 1 trillion total parameters. We propose the MuonClip optimizer, which improves upon Muon with a novel QK-clip technique to address training instability while enjoying the advanced token efficiency of Muon. Based on MuonClip, K2 was pre-trained on 15.5 trillion tokens with zero loss spike. During post-training, K2 undergoes a multi-stage post-training process, highlighted by a large-scale agentic data synthesis pipeline and a joint reinforcement learning (RL) stage, where the model improves its capabilities through interactions with real and synthetic environments. Kimi K2 achieves state-of-the-art performance among open-source non-thinking models, with strengths in agentic capabilities. Notably, K2 obtains 66.1 on Tau2-Bench, 76.5 on ACEBench (En), 65.8 on SWE-Bench Verified, and 47.3 on SWE-Bench Multilingual -- surpassing most open and closed-sourced baselines in non-thinking settings. It also exhibits strong capabilities in coding, mathematics, and reasoning tasks, with a score of 53.7 on LiveCodeBench v6, 49.5 on AIME 2025, 75.1 on GPQA-Diamond, and 27.1 on OJBench, all without extended thinking. These results position Kimi K2 as one of the most capable open-source large language models to date, particularly in software engineering and agentic tasks. We release our base and post-trained model checkpoints to facilitate future research and applications of agentic intelligence.
>
---
#### [replaced 007] Interpreting and Controlling LLM Reasoning through Integrated Policy Gradient
- **分类: cs.AI; cs.CL; cs.LG**

- **简介: 该论文属于模型解释与控制任务，旨在解决LLM推理机制不透明的问题。通过提出IPG框架，实现对推理过程的精确定位与调控。**

- **链接: [https://arxiv.org/pdf/2602.02313v2](https://arxiv.org/pdf/2602.02313v2)**

> **作者:** Changming Li; Kaixing Zhang; Haoyun Xu; Yingdong Shi; Zheng Zhang; Kaitao Song; Kan Ren
>
> **摘要:** Large language models (LLMs) demonstrate strong reasoning abilities in solving complex real-world problems. Yet, the internal mechanisms driving these complex reasoning behaviors remain opaque. Existing interpretability approaches targeting reasoning either identify components (e.g., neurons) correlated with special textual patterns, or rely on human-annotated contrastive pairs to derive control vectors. Consequently, current methods struggle to precisely localize complex reasoning mechanisms or capture sequential influence from model internal workings to the reasoning outputs. In this paper, built on outcome-oriented and sequential-influence-aware principles, we focus on identifying components that have sequential contribution to reasoning behavior where outcomes are cumulated by long-range effects. We propose Integrated Policy Gradient (IPG), a novel framework that attributes reasoning behaviors to model's inner components by propagating compound outcome-based signals such as post reasoning accuracy backward through model inference trajectories. Empirical evaluations demonstrate that our approach achieves more precise localization and enables reliable modulation of reasoning behaviors (e.g., reasoning capability, reasoning strength) across diverse reasoning models.
>
---
#### [replaced 008] From Self-Evolving Synthetic Data to Verifiable-Reward RL: Post-Training Multi-turn Interactive Tool-Using Agents
- **分类: cs.AI; cs.CL**

- **简介: 该论文研究交互式工具使用代理的后训练问题，旨在提升多轮对话中工具使用的可靠性与效率。通过合成数据与验证奖励强化学习结合的方法，提高模型性能。**

- **链接: [https://arxiv.org/pdf/2601.22607v2](https://arxiv.org/pdf/2601.22607v2)**

> **作者:** Jiaxuan Gao; Jiaao Chen; Chuyi He; Wei-Chen Wang; Shusheng Xu; Hanrui Wang; Di Jin; Yi Wu
>
> **备注:** Submitted to ICML 2026
>
> **摘要:** Interactive tool-using agents must solve real-world tasks via multi-turn interaction with both humans and external environments, requiring dialogue state tracking, multi-step tool execution, while following complex instructions. Post-training such agents is challenging because synthesis for high-quality multi-turn tool-use data is difficult to scale, and reinforcement learning (RL) could face noisy signals caused by user simulation, leading to degraded training efficiency. We propose a unified framework that combines a self-evolving data agent with verifier-based RL. Our system, EigenData, is a hierarchical multi-agent engine that synthesizes tool-grounded dialogues together with executable per-instance checkers, and improves generation reliability via closed-loop self-evolving process that updates prompts and workflow. Building on the synthetic data, we develop an RL recipe that first fine-tunes the user model and then applies GRPO-style training with trajectory-level group-relative advantages and dynamic filtering, yielding consistent improvements beyond SFT. Evaluated on tau^2-bench, our best model reaches 73.0% pass^1 on Airline and 98.3% pass^1 on Telecom, matching or exceeding frontier models. Overall, our results suggest a scalable pathway for bootstrapping complex tool-using behaviors without expensive human annotation.
>
---
#### [replaced 009] Wiki Live Challenge: Challenging Deep Research Agents with Expert-Level Wikipedia Articles
- **分类: cs.CL**

- **简介: 该论文提出Wiki Live Challenge，用于评估深度研究代理的性能。任务是提升DRAs的准确性与质量，通过专家级维基文章作为基准，设计了细粒度评估框架。**

- **链接: [https://arxiv.org/pdf/2602.01590v2](https://arxiv.org/pdf/2602.01590v2)**

> **作者:** Shaohan Wang; Benfeng Xu; Licheng Zhang; Mingxuan Du; Chiwei Zhu; Xiaorui Wang; Zhendong Mao; Yongdong Zhang
>
> **备注:** Preprint
>
> **摘要:** Deep Research Agents (DRAs) have demonstrated remarkable capabilities in autonomous information retrieval and report generation, showing great potential to assist humans in complex research tasks. Current evaluation frameworks primarily rely on LLM-generated references or LLM-derived evaluation dimensions. While these approaches offer scalability, they often lack the reliability of expert-verified content and struggle to provide objective, fine-grained assessments of critical dimensions. To bridge this gap, we introduce Wiki Live Challenge (WLC), a live benchmark that leverages the newest Wikipedia Good Articles (GAs) as expert-level references. Wikipedia's strict standards for neutrality, comprehensiveness, and verifiability serve as a great challenge for DRAs, with GAs representing the pinnacle of which. We curate a dataset of 100 recent Good Articles and propose Wiki Eval, a comprehensive evaluation framework comprising a fine-grained evaluation method with 39 criteria for writing quality and rigorous metrics for factual verifiability. Extensive experiments on various DRA systems demonstrate a significant gap between current DRAs and human expert-level Wikipedia articles, validating the effectiveness of WLC in advancing agent research. We release our benchmark at https://github.com/WangShao2000/Wiki_Live_Challenge
>
---
#### [replaced 010] Causality Guided Representation Learning for Cross-Style Hate Speech Detection
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于仇恨言论检测任务，旨在解决隐性仇恨言论难以检测的问题。通过构建因果图模型，提出CADET框架，分离潜在因素并控制混杂变量，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2510.07707v2](https://arxiv.org/pdf/2510.07707v2)**

> **作者:** Chengshuai Zhao; Shu Wan; Paras Sheth; Karan Patwa; K. Selçuk Candan; Huan Liu
>
> **备注:** Accepted by the ACM Web Conference 2026 (WWW 26)
>
> **摘要:** The proliferation of online hate speech poses a significant threat to the harmony of the web. While explicit hate is easily recognized through overt slurs, implicit hate speech is often conveyed through sarcasm, irony, stereotypes, or coded language -- making it harder to detect. Existing hate speech detection models, which predominantly rely on surface-level linguistic cues, fail to generalize effectively across diverse stylistic variations. Moreover, hate speech spread on different platforms often targets distinct groups and adopts unique styles, potentially inducing spurious correlations between them and labels, further challenging current detection approaches. Motivated by these observations, we hypothesize that the generation of hate speech can be modeled as a causal graph involving key factors: contextual environment, creator motivation, target, and style. Guided by this graph, we propose CADET, a causal representation learning framework that disentangles hate speech into interpretable latent factors and then controls confounders, thereby isolating genuine hate intent from superficial linguistic cues. Furthermore, CADET allows counterfactual reasoning by intervening on style within the latent space, naturally guiding the model to robustly identify hate speech in varying forms. CADET demonstrates superior performance in comprehensive experiments, highlighting the potential of causal priors in advancing generalizable hate speech detection.
>
---
#### [replaced 011] Capturing Classic Authorial Style in Long-Form Story Generation with GRPO Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文属于长文本生成任务，旨在解决作者风格可控生成的问题。通过两阶段方法，训练风格判别器并用于GRPO优化，提升生成文本的风格相似度。**

- **链接: [https://arxiv.org/pdf/2512.05747v2](https://arxiv.org/pdf/2512.05747v2)**

> **作者:** Jinlong Liu; Mohammed Bahja; Venelin Kovatchev; Mark Lee
>
> **摘要:** Evaluating and optimising authorial style in long-form story generation remains challenging because style is often assessed with ad hoc prompting and is frequently conflated with overall writing quality. We propose a two-stage pipeline. First, we train a dedicated style-similarity judge by fine-tuning a sentence-transformer with authorship-verification supervision, and calibrate its similarity outputs into a bounded $[0,1]$ reward. Second, we use this judge as the primary reward in Group Relative Policy Optimization (GRPO) to fine-tune an 8B story generator for style-conditioned writing, avoiding the accept/reject supervision required by Direct Preference Optimization (DPO). Across four target authors (Mark Twain, Jane Austen, Charles Dickens, Thomas Hardy), the GRPO-trained 8B model achieves higher style scores than open-weight baselines, with an average style score of 0.893 across authors. These results suggest that AV-calibrated reward modelling provides a practical mechanism for controllable style transfer in long-form generation under a moderate model size and training budget.
>
---
#### [replaced 012] AR-MAP: Are Autoregressive Large Language Models Implicit Teachers for Diffusion Large Language Models?
- **分类: cs.CL**

- **简介: 该论文属于语言模型对齐任务，旨在解决DLLM因ELBO估计导致的高方差问题。通过AR-MAP框架，利用AR-LLMs作为隐式教师提升DLLM对齐效果。**

- **链接: [https://arxiv.org/pdf/2602.02178v2](https://arxiv.org/pdf/2602.02178v2)**

> **作者:** Liang Lin; Feng Xiong; Zengbin Wang; Kun Wang; Junhao Dong; Xuecai Hu; Yong Wang; Xiangxiang Chu
>
> **摘要:** Diffusion Large Language Models (DLLMs) have emerged as a powerful alternative to autoregressive models, enabling parallel token generation across multiple positions. However, preference alignment of DLLMs remains challenging due to high variance introduced by Evidence Lower Bound (ELBO)-based likelihood estimation. In this work, we propose AR-MAP, a novel transfer learning framework that leverages preference-aligned autoregressive LLMs (AR-LLMs) as implicit teachers for DLLM alignment. We reveal that DLLMs can effectively absorb alignment knowledge from AR-LLMs through simple weight scaling, exploiting the shared architectural structure between these divergent generation paradigms. Crucially, our approach circumvents the high variance and computational overhead of direct DLLM alignment and comprehensive experiments across diverse preference alignment tasks demonstrate that AR-MAP achieves competitive or superior performance compared to existing DLLM-specific alignment methods, achieving 69.08\% average score across all tasks and models. Our Code is available at https://github.com/AMAP-ML/AR-MAP.
>
---
#### [replaced 013] Encoder-Free Knowledge-Graph Reasoning with LLMs via Hyperdimensional Path Retrieval
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于知识图谱推理任务，旨在解决传统KG-QA系统效率低、透明度差的问题。提出PathHD框架，利用超维计算与单次LLM调用实现高效且可解释的路径检索与推理。**

- **链接: [https://arxiv.org/pdf/2512.09369v2](https://arxiv.org/pdf/2512.09369v2)**

> **作者:** Yezi Liu; William Youngwoo Chung; Hanning Chen; Calvin Yeung; Mohsen Imani
>
> **摘要:** Recent progress in large language models (LLMs) has made knowledge-grounded reasoning increasingly practical, yet KG-based QA systems often pay a steep price in efficiency and transparency. In typical pipelines, symbolic paths are scored by neural encoders or repeatedly re-ranked by multiple LLM calls, which inflates latency and GPU cost and makes the decision process hard to audit. We introduce PathHD, an encoder-free framework for knowledge-graph reasoning that couples hyperdimensional computing (HDC) with a single LLM call per query. Given a query, PathHD represents relation paths as block-diagonal GHRR hypervectors, retrieves candidate paths using a calibrated blockwise cosine similarity with Top-K pruning, and then performs a one-shot LLM adjudication that outputs the final answer together with supporting, citeable paths. The design is enabled by three technical components: (i) an order-sensitive, non-commutative binding operator for composing multi-hop paths, (ii) a robust similarity calibration that stabilizes hypervector retrieval, and (iii) an adjudication stage that preserves interpretability while avoiding per-path LLM scoring. Across WebQSP, CWQ, and GrailQA, PathHD matches or improves Hits@1 compared to strong neural baselines while using only one LLM call per query, reduces end-to-end latency by $40-60\%$, and lowers GPU memory by $3-5\times$ due to encoder-free retrieval. Overall, the results suggest that carefully engineered HDC path representations can serve as an effective substrate for efficient and faithful KG-LLM reasoning, achieving a strong accuracy-efficiency-interpretability trade-off.
>
---
#### [replaced 014] On the Interplay between Human Label Variation and Model Fairness
- **分类: cs.CL**

- **简介: 该论文属于模型公平性研究任务，探讨人类标签变异对模型公平性的影响。通过对比不同HLV方法，验证其在特定配置下提升公平性的潜力。**

- **链接: [https://arxiv.org/pdf/2510.12036v2](https://arxiv.org/pdf/2510.12036v2)**

> **作者:** Kemal Kurniawan; Meladel Mistica; Timothy Baldwin; Jey Han Lau
>
> **备注:** 10 pages, 7 figures. Accepted to EACL Findings 2026
>
> **摘要:** The impact of human label variation (HLV) on model fairness is an unexplored topic. This paper examines the interplay by comparing training on majority-vote labels with a range of HLV methods. Our experiments show that without explicit debiasing, HLV training methods have a positive impact on fairness under certain configurations.
>
---
#### [replaced 015] Reusing Overtrained Language Models Saturates Scaling
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究语言模型多阶段预训练中的模型复用问题，探讨过拟合基础模型的再训练效果。发现再训练收益随预训练量增加而饱和，提出简单缩放规律。**

- **链接: [https://arxiv.org/pdf/2510.06548v2](https://arxiv.org/pdf/2510.06548v2)**

> **作者:** Seng Pei Liew; Takuya Kato
>
> **摘要:** Reusing pretrained base models for further pretraining, such as continual pretraining or model growth, is promising at reducing the cost of training language models from scratch. However, the effectiveness remains unclear, especially when applied to overtrained base models. In this work, we empirically study the scaling properties of model reuse and find that the scaling efficiency diminishes in a predictable manner: The scaling exponent with respect to second-stage training tokens decreases logarithmically with the number of tokens used to pretrain the base model. The joint dependence on first- and second-stage tokens is accurately modeled by a simple scaling law. Such saturation effect reveals a fundamental trade-off in multi-stage pretraining strategies: the more extensively a base model is pretrained, the less benefit additional pretraining provides. Our findings provide practical insights for efficient language model training and raise important considerations for the reuse of overtrained models.
>
---
#### [replaced 016] KVzap: Fast, Adaptive, and Faithful KV Cache Pruning
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于自然语言处理中的模型优化任务，旨在解决Transformer模型中KV缓存占用过大的问题。通过提出KVzap方法，实现高效、自适应的KV缓存压缩，提升推理效率。**

- **链接: [https://arxiv.org/pdf/2601.07891v2](https://arxiv.org/pdf/2601.07891v2)**

> **作者:** Simon Jegou; Maximilian Jeblick
>
> **摘要:** Growing context lengths in transformer-based language models have made the key-value (KV) cache a critical inference bottleneck. While many KV cache pruning methods have been proposed, they have not yet been adopted in major inference engines due to speed--accuracy trade-offs. We introduce KVzap, a fast, input-adaptive approximation of KVzip that works in both prefilling and decoding. On Qwen3-8B, Llama-3.1-8B-Instruct, and Qwen3-32B across long-context and reasoning tasks, KVzap achieves $2$--$4\times$ KV cache compression with negligible accuracy loss and achieves state-of-the-art performance on the KVpress leaderboard. Code and models are available at https://github.com/NVIDIA/kvpress.
>
---
#### [replaced 017] Beyond the Vision Encoder: Identifying and Mitigating Spatial Bias in Large Vision-Language Models
- **分类: cs.CV; cs.CL**

- **简介: 该论文属于多模态任务，旨在解决LVLMs在空间位置变化下的语义理解偏差问题。通过分析注意力机制，提出AGCI方法提升模型空间鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.21984v2](https://arxiv.org/pdf/2509.21984v2)**

> **作者:** Yingjie Zhu; Xuefeng Bai; Kehai Chen; Yang Xiang; Youcheng Pan; Yongshuai Hou; Weili Guan; Jun Yu; Min Zhang
>
> **摘要:** Large Vision-Language Models (LVLMs) have achieved remarkable success across a wide range of multimodal tasks, yet their robustness to spatial variations remains insufficiently understood. In this work, we conduct a systematic study of the spatial bias of LVLMs, examining how models respond when identical key visual information is placed at different locations within an image. Through controlled probing experiments, we observe that current LVLMs often produce inconsistent outputs under such spatial shifts, revealing a clear spatial bias in their semantic understanding. Further analysis indicates that this bias does not stem from the vision encoder, but rather from a mismatch in attention mechanisms between the vision encoder and the large language model, which disrupts the global information flow. Motivated by this insight, we propose Adaptive Global Context Injection (AGCI), a lightweight mechanism that dynamically injects shared global visual context into each image token. AGCI works without architectural modifications, mitigating spatial bias by enhancing the semantic accessibility of image tokens while preserving the model's intrinsic capabilities. Extensive experiments demonstrate that AGCI not only enhances the spatial robustness of LVLMs, but also achieves strong performance on various downstream tasks and hallucination benchmarks.
>
---
#### [replaced 018] What MLLMs Learn about When they Learn about Multimodal Reasoning: Perception, Reasoning, or their Integration?
- **分类: cs.CL**

- **简介: 该论文属于多模态推理任务，旨在解决评估指标单一的问题。通过分解性能为感知、推理和整合，揭示模型进步的实质。**

- **链接: [https://arxiv.org/pdf/2510.01719v3](https://arxiv.org/pdf/2510.01719v3)**

> **作者:** Jiwan Chung; Neel Joshi; Pratyusha Sharma; Youngjae Yu; Vibhav Vineet
>
> **摘要:** Evaluation of multimodal reasoning models is typically reduced to a single accuracy score, implicitly treating reasoning as a unitary capability. We introduce MathLens, a benchmark of textbook-style geometry problems that exposes this assumption by operationally decomposing performance into Perception, Reasoning, and Integration. Each problem is derived from a symbolic specification and accompanied by visual diagrams, text-only variants, multimodal questions, and targeted perceptual probes, enabling controlled measurement of each component. Using this decomposition, we show that common training strategies induce systematically different capability profiles that are invisible under aggregate accuracy. Reinforcement learning primarily improves perceptual grounding and robustness to diagram variation, while textual SFT yields gains through reflective reasoning. In contrast, as perception and reasoning improve, a growing fraction of remaining errors fall outside these components and are categorized as integration. These results suggest that apparent progress in multimodal reasoning reflects shifting balances among subskills rather than uniform advancement, motivating evaluation beyond scalar accuracy.
>
---
#### [replaced 019] EverMemBench: Benchmarking Long-Term Interactive Memory in Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于自然语言处理任务，旨在解决大语言模型长期对话记忆的问题。提出EverMemBench基准，评估模型在多角色、多主题对话中的记忆能力，揭示现有系统的不足。**

- **链接: [https://arxiv.org/pdf/2602.01313v2](https://arxiv.org/pdf/2602.01313v2)**

> **作者:** Chuanrui Hu; Tong Li; Xingze Gao; Hongda Chen; Yi Bai; Dannong Xu; Tianwei Lin; Xinda Zhao; Xiaohong Li; Yunyun Han; Jian Pei; Yafeng Deng
>
> **备注:** 10 pages, 2 figures, 4 tables
>
> **摘要:** Long-term conversational memory is essential for LLM-based assistants, yet existing benchmarks focus on dyadic, single-topic dialogues that fail to capture real-world complexity. We introduce EverMemBench, a benchmark featuring multi-party, multi-group conversations spanning over 1 million tokens with temporally evolving information, cross-topic interleaving, and role-specific personas. EverMemBench evaluates memory systems across three dimensions through 1,000+ QA pairs: fine-grained recall, memory awareness, and user profile understanding. Our evaluation reveals critical limitations: (1) multi-hop reasoning collapses in multi-party settings, with even oracle models achieving only 26%; (2) temporal reasoning remains unsolved, requiring version semantics beyond timestamp matching; (3) memory awareness is bottlenecked by retrieval, where current similarity-based methods fail to bridge the semantic gap between queries and implicitly relevant memories. EverMemBench provides a challenging testbed for developing next-generation memory architectures.
>
---
#### [replaced 020] POPI: Personalizing LLMs via Optimized Preference Inference
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于语言模型个性化任务，解决用户偏好差异问题。通过分解为偏好推理和条件生成，提出POPI方法提升个性化效果并实现模型迁移。**

- **链接: [https://arxiv.org/pdf/2510.17881v2](https://arxiv.org/pdf/2510.17881v2)**

> **作者:** Yizhuo Chen; Xin Liu; Ruijie Wang; Zheng Li; Pei Chen; Changlong Yu; Priyanka Nigam; Meng Jiang; Bing Yin
>
> **摘要:** Large language models (LLMs) are typically aligned with population-level preferences, despite substantial variation across individual users. While many LLM personalization methods exist, the underlying structure of user-level personalization is often left implicit. We formalize user-level, prompt-independent personalization as a decomposition into two components: preference inference and conditioned generation. We advocate for a modular design that decouples these components; identify natural language as a generator-agnostic interface between them; and characterize generator-transferability as a key implication of modular personalization. Guided by this abstraction, we introduce POPI, a novel instantiation of modular personalization that parameterizes both preference inference and conditioned generation as shared LLMs. POPI jointly optimizes the two components under a unified preference optimization objective, using reinforcement learning as an optimization tool. Across multiple benchmarks, POPI consistently improves personalization performance while reducing context overhead. We further demonstrate that the learned natural-language preference summaries transfer effectively to frozen, off-the-shelf LLMs, including black-box APIs, providing empirical evidence of modularity and generator-transferability.
>
---
#### [replaced 021] SAIL-RL: Guiding MLLMs in When and How to Think via Dual-Reward RL Tuning
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出SAIL-RL框架，用于提升多模态大语言模型的推理能力。解决现有方法依赖结果监督和统一思考策略的问题，通过双奖励机制优化思考时机与方式。**

- **链接: [https://arxiv.org/pdf/2511.02280v2](https://arxiv.org/pdf/2511.02280v2)**

> **作者:** Fangxun Shu; Yongjie Ye; Yue Liao; Zijian Kang; Weijie Yin; Jiacong Wang; Xiao Liang; Shuicheng Yan; Chao Feng
>
> **摘要:** We introduce SAIL-RL, a reinforcement learning (RL) post-training framework that enhances the reasoning capabilities of multimodal large language models (MLLMs) by teaching them when and how to think. Existing approaches are limited by outcome-only supervision, which rewards correct answers without ensuring sound reasoning, and by uniform thinking strategies, which often lead to overthinking on simple tasks and underthinking on complex ones. SAIL-RL addresses these challenges with a dual reward system: the Thinking Reward, which evaluates reasoning quality through factual grounding, logical coherence, and answer consistency, and the Judging Reward, which adaptively determines whether deep reasoning or direct answering is appropriate. Experiments on the state-of-the-art SAIL-VL2 show that SAIL-RL improves reasoning and multimodal understanding benchmarks at both 4B and 8B scales, achieving competitive performance against commercial closed-source models such as GPT-4o, and substantially reduces hallucinations, establishing it as a principled framework for building more reliable and adaptive MLLMs. The code will be available at https://github.com/BytedanceDouyinContent/SAIL-RL.
>
---
#### [replaced 022] GeoResponder: Towards Building Geospatial LLMs for Time-Critical Disaster Response
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于地理信息与语言模型结合的任务，旨在解决灾难响应中缺乏空间推理能力的问题。通过构建GeoResponder框架，提升模型对地理空间结构的理解与应用能力。**

- **链接: [https://arxiv.org/pdf/2509.19354v2](https://arxiv.org/pdf/2509.19354v2)**

> **作者:** Ahmed El Fekih Zguir; Ferda Ofli; Muhammad Imran
>
> **摘要:** Large Language Models excel at linguistic tasks but lack the inner geospatial capabilities needed for time-critical disaster response, where reasoning about road networks, continuous coordinates, and access to essential infrastructure such as hospitals, shelters, and pharmacies is vital. We introduce GeoResponder, a framework that instills robust spatial reasoning through a scaffolded instruction-tuning curriculum. By stratifying geospatial learning into different cognitive layers, we effectively anchor semantic knowledge to the continuous coordinate manifold and enforce the internalization of spatial axioms. Extensive evaluations across four topologically distinct cities and diverse tasks demonstrate that GeoResponder significantly outperforms both state-of-the-art foundation models and domain-specific baselines. These results suggest that LLMs can begin to internalize and generalize geospatial structures, pointing toward the future development of language models capable of supporting disaster response needs.
>
---
#### [replaced 023] Multi-turn Evaluation of Anthropomorphic Behaviours in Large Language Models
- **分类: cs.CL; cs.CY; cs.HC**

- **简介: 该论文属于自然语言处理中的行为评估任务，旨在解决LLM anthropomorphic行为的多轮评价问题。通过多轮测试、模拟交互和大规模实验，分析模型行为特征及其对用户感知的影响。**

- **链接: [https://arxiv.org/pdf/2502.07077v3](https://arxiv.org/pdf/2502.07077v3)**

> **作者:** Lujain Ibrahim; Canfer Akbulut; Rasmi Elasmar; Charvi Rastogi; Minsuk Kahng; Meredith Ringel Morris; Kevin R. McKee; Verena Rieser; Murray Shanahan; Laura Weidinger
>
> **摘要:** The tendency of users to anthropomorphise large language models (LLMs) is of growing interest to AI developers, researchers, and policy-makers. Here, we present a novel method for empirically evaluating anthropomorphic LLM behaviours in realistic and varied settings. Going beyond single-turn static benchmarks, we contribute three methodological advances in state-of-the-art (SOTA) LLM evaluation. First, we develop a multi-turn evaluation of 14 anthropomorphic behaviours. Second, we present a scalable, automated approach by employing simulations of user interactions. Third, we conduct an interactive, large-scale human subject study (N=1101) to validate that the model behaviours we measure predict real users' anthropomorphic perceptions. We find that all SOTA LLMs evaluated exhibit similar behaviours, characterised by relationship-building (e.g., empathy and validation) and first-person pronoun use, and that the majority of behaviours only first occur after multiple turns. Our work lays an empirical foundation for investigating how design choices influence anthropomorphic model behaviours and for progressing the ethical debate on the desirability of these behaviours. It also showcases the necessity of multi-turn evaluations for complex social phenomena in human-AI interaction.
>
---
#### [replaced 024] The Path of Least Resistance: Guiding LLM Reasoning Trajectories with Prefix Consensus
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出PoLR方法，用于提升大语言模型推理效率。针对计算成本高的问题，通过前缀一致性减少冗余计算，提升推理速度与效率。属于自然语言处理中的推理优化任务。**

- **链接: [https://arxiv.org/pdf/2601.21494v2](https://arxiv.org/pdf/2601.21494v2)**

> **作者:** Ishan Jindal; Sai Prashanth Akuthota; Jayant Taneja; Sachin Dev Sharma
>
> **备注:** Accepted at ICLR 2026. https://openreview.net/forum?id=hrnSqERgPn
>
> **摘要:** Large language models achieve strong reasoning performance, but inference strategies such as Self-Consistency (SC) are computationally expensive, as they fully expand all reasoning traces. We introduce PoLR (Path of Least Resistance), the first inference-time method to leverage prefix consistency for compute-efficient reasoning. PoLR clusters short prefixes of reasoning traces, identifies the dominant cluster, and expands all paths in that cluster, preserving the accuracy benefits of SC while substantially reducing token usage and latency. Our theoretical analysis, framed via mutual information and entropy, explains why early reasoning steps encode strong signals predictive of final correctness. Empirically, PoLR consistently matches or exceeds SC across GSM8K, MATH500, AIME24/25, and GPQA-DIAMOND, reducing token usage by up to 60% and wall-clock latency by up to 50%. Moreover, PoLR is fully complementary to adaptive inference methods (e.g., Adaptive Consistency, Early-Stopping SC) and can serve as a drop-in pre-filter, making SC substantially more efficient and scalable without requiring model fine-tuning.
>
---
#### [replaced 025] Rethinking Bottlenecks in Safety Fine-Tuning of Vision Language Models
- **分类: cs.CV; cs.CL; cs.CR**

- **简介: 该论文属于视觉语言模型的安全微调任务，旨在解决安全推理能力不足的问题。通过构建多图像安全数据集，提升模型在安全场景下的视觉推理能力。**

- **链接: [https://arxiv.org/pdf/2501.18533v3](https://arxiv.org/pdf/2501.18533v3)**

> **作者:** Yi Ding; Lijun Li; Bing Cao; Jing Shao
>
> **摘要:** Large Vision-Language Models (VLMs) have achieved remarkable performance across a wide range of tasks. However, their deployment in safety-critical domains poses significant challenges. Existing safety fine-tuning methods, which focus on textual or multimodal content, fall short in addressing challenging cases or disrupt the balance between helpfulness and harmlessness. Our evaluation highlights a safety reasoning gap: these methods lack safety visual reasoning ability, leading to such bottlenecks. To address this limitation and enhance both visual perception and reasoning in safety-critical contexts, we propose a novel dataset that integrates multi-image inputs with safety Chain-of-Thought (CoT) labels as fine-grained reasoning logic to improve model performance. Specifically, we introduce the Multi-Image Safety (MIS) dataset, an instruction-following dataset tailored for multi-image safety scenarios, consisting of training and test splits. Our experiments demonstrate that fine-tuning InternVL2.5-8B with MIS significantly outperforms both powerful open-source models and API-based models in challenging multi-image tasks requiring safety-related visual reasoning. This approach not only delivers exceptional safety performance but also preserves general capabilities without any trade-offs. Specifically, fine-tuning with MIS increases average accuracy by 0.83% across five general benchmarks and reduces the Attack Success Rate (ASR) on multiple safety benchmarks by a large margin.
>
---
#### [replaced 026] DynaSpec: Context-aware Dynamic Speculative Sampling for Large-Vocabulary Language Models
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出DynaSpec，解决大词表语言模型的推测解码效率问题。通过动态短列表机制提升吞吐量，保持准确性。**

- **链接: [https://arxiv.org/pdf/2510.13847v3](https://arxiv.org/pdf/2510.13847v3)**

> **作者:** Jinbin Zhang; Nasib Ullah; Erik Schultheis; Rohit Babbar
>
> **摘要:** Speculative decoding accelerates LLM inference by letting a small drafter propose multiple tokens which a large target model verifies once per speculation step. As vocabularies scale past 10e5 tokens,verification cost in the target model is largely unchanged, but the drafter can become bottlenecked by its O(|V|d) output projection. Recent approaches (e.g., FR-Spec, VocabTrim) mitigate this by restricting drafting to a fixed, frequency-ranked shortlist; however, such static truncation is corpus-dependent and suppresses rare or domain-specific tokens, reducing acceptance and limiting speedups. We propose DynaSpec, a context-dependent dynamic shortlisting mechanism for large-vocabulary speculative decoding. DynaSpec trains lightweight meta-classifiers that route each context to a small set of coarse token clusters; the union of the top-selected clusters defines the drafter's shortlist, while the target model still verifies over the full vocabulary, preserving exactness. Systems-wise, routing is overlapped with draft computation via parallel execution streams, reducing end-to-end overhead. Across standard speculative decoding benchmarks, DynaSpec consistently improves mean accepted length-recovering 98.4% of full-vocabulary performance for Llama-3-8B versus 93.6% for fixed-shortlist baselines-and achieves up to a 2.23x throughput gain compared to 1.91x for static approaches on the dataset with rare tokens.
>
---
#### [replaced 027] ARTIS: Agentic Risk-Aware Test-Time Scaling via Iterative Simulation
- **分类: cs.CL**

- **简介: 该论文提出ARTIS框架，解决agentic场景下测试时扩展（TTS）的可靠性问题。通过模拟交互提升动作可靠性，避免环境风险。**

- **链接: [https://arxiv.org/pdf/2602.01709v2](https://arxiv.org/pdf/2602.01709v2)**

> **作者:** Xingshan Zeng; Lingzhi Wang; Weiwen Liu; Liangyou Li; Yasheng Wang; Lifeng Shang; Xin Jiang; Qun Liu
>
> **摘要:** Current test-time scaling (TTS) techniques enhance large language model (LLM) performance by allocating additional computation at inference time, yet they remain insufficient for agentic settings, where actions directly interact with external environments and their effects can be irreversible and costly. We propose ARTIS, Agentic Risk-Aware Test-Time Scaling via Iterative Simulation, a framework that decouples exploration from commitment by enabling test-time exploration through simulated interactions prior to real-world execution. This design allows extending inference-time computation to improve action-level reliability and robustness without incurring environmental risk. We further show that naive LLM-based simulators struggle to capture rare but high-impact failure modes, substantially limiting their effectiveness for agentic decision making. To address this limitation, we introduce a risk-aware tool simulator that emphasizes fidelity on failure-inducing actions via targeted data generation and rebalanced training. Experiments on multi-turn and multi-step agentic benchmarks demonstrate that iterative simulation substantially improves agent reliability, and that risk-aware simulation is essential for consistently realizing these gains across models and tasks.
>
---
#### [replaced 028] From Labels to Facets: Building a Taxonomically Enriched Turkish Learner Corpus
- **分类: cs.CL**

- **简介: 该论文属于语言学标注任务，旨在解决学习者语料库标注不够细致的问题。通过提出一种半自动标注方法，构建了首个基于分面分类的土耳其语学习者语料库，提升标注的深度与分析能力。**

- **链接: [https://arxiv.org/pdf/2601.22875v2](https://arxiv.org/pdf/2601.22875v2)**

> **作者:** Elif Sayar; Tolgahan Türker; Anna Golynskaia Knezhevich; Bihter Dereli; Ayşe Demirhas; Lionel Nicolas; Gülşen Eryiğit
>
> **备注:** An error was identified in the analyses presented in Section 5.3, impacting the conclusions of the paper. The authors have therefore withdrawn the submission
>
> **摘要:** In terms of annotation structure, most learner corpora rely on holistic flat label inventories which, even when extensive, do not explicitly separate multiple linguistic dimensions. This makes linguistically deep annotation difficult and complicates fine-grained analyses aimed at understanding why and how learners produce specific errors. To address these limitations, this paper presents a semi-automated annotation methodology for learner corpora, built upon a recently proposed faceted taxonomy, and implemented through a novel annotation extension framework. The taxonomy provides a theoretically grounded, multi-dimensional categorization that captures the linguistic properties underlying each error instance, thereby enabling standardized, fine-grained, and interpretable enrichment beyond flat annotations. The annotation extension tool, implemented based on the proposed extension framework for Turkish, automatically extends existing flat annotations by inferring additional linguistic and metadata information as facets within the taxonomy to provide richer learner-specific context. It was systematically evaluated and yielded promising performance results, achieving a facet-level accuracy of 95.86%. The resulting taxonomically enriched corpus offers enhanced querying capabilities and supports detailed exploratory analyses across learner corpora, enabling researchers to investigate error patterns through complex linguistic and pedagogical dimensions. This work introduces the first collaboratively annotated and taxonomically enriched Turkish Learner Corpus, a manual annotation guideline with a refined tagset, and an annotation extender. As the first corpus designed in accordance with the recently introduced taxonomy, we expect our study to pave the way for subsequent enrichment efforts of existing error-annotated learner corpora.
>
---
#### [replaced 029] Closing the Loop: Universal Repository Representation with RPG-Encoder
- **分类: cs.CL; cs.SE**

- **简介: 该论文属于代码理解任务，旨在解决仓库表示碎片化问题。提出RPG-Encoder框架，实现代码与意图的双向精准映射，提升代码库的结构感知能力。**

- **链接: [https://arxiv.org/pdf/2602.02084v2](https://arxiv.org/pdf/2602.02084v2)**

> **作者:** Jane Luo; Chengyu Yin; Xin Zhang; Qingtao Li; Steven Liu; Yiming Huang; Jie Wu; Hao Liu; Yangyu Huang; Yu Kang; Fangkai Yang; Ying Xin; Scarlett Li
>
> **摘要:** Current repository agents encounter a reasoning disconnect due to fragmented representations, as existing methods rely on isolated API documentation or dependency graphs that lack semantic depth. We consider repository comprehension and generation to be inverse processes within a unified cycle: generation expands intent into implementation, while comprehension compresses implementation back into intent. To address this, we propose RPG-Encoder, a framework that generalizes the Repository Planning Graph (RPG) from a static generative blueprint into a unified, high-fidelity representation. RPG-Encoder closes the reasoning loop through three mechanisms: (1) Encoding raw code into the RPG that combines lifted semantic features with code dependencies; (2) Evolving the topology incrementally to decouple maintenance costs from repository scale, reducing overhead by 95.7%; and (3) Operating as a unified interface for structure-aware navigation. In evaluations, RPG-Encoder establishes state-of-the-art localization performance on SWE-bench Verified with 93.7% Acc@5 and exceeds the best baseline by over 10% in localization accuracy on SWE-bench Live Lite. These results highlight our superior fine-grained precision in complex codebases. Furthermore, it achieves 98.5% reconstruction coverage on RepoCraft, confirming RPG's high-fidelity capacity to mirror the original codebase and closing the loop between intent and implementation.
>
---
#### [replaced 030] LUMINA: Detecting Hallucinations in RAG System with Context-Knowledge Signals
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理中的幻觉检测任务，旨在解决RAG系统中仍存在的幻觉问题。通过分析外部上下文与内部知识的使用信号，提出LUMINA框架进行有效检测。**

- **链接: [https://arxiv.org/pdf/2509.21875v3](https://arxiv.org/pdf/2509.21875v3)**

> **作者:** Samuel Yeh; Sharon Li; Tanwi Mallick
>
> **备注:** ICLR 2026
>
> **摘要:** Retrieval-Augmented Generation (RAG) aims to mitigate hallucinations in large language models (LLMs) by grounding responses in retrieved documents. Yet, RAG-based LLMs still hallucinate even when provided with correct and sufficient context. A growing line of work suggests that this stems from an imbalance between how models use external context and their internal knowledge, and several approaches have attempted to quantify these signals for hallucination detection. However, existing methods require extensive hyperparameter tuning, limiting their generalizability. We propose LUMINA, a novel framework that detects hallucinations in RAG systems through context--knowledge signals: external context utilization is quantified via distributional distance, while internal knowledge utilization is measured by tracking how predicted tokens evolve across transformer layers. We further introduce a framework for statistically validating these measurements. Experiments on common RAG hallucination benchmarks and four open-source LLMs show that LUMINA achieves consistently high AUROC and AUPRC scores, outperforming prior utilization-based methods by up to +13% AUROC on HalluRAG. Moreover, LUMINA remains robust under relaxed assumptions about retrieval quality and model matching, offering both effectiveness and practicality. LUMINA: https://github.com/deeplearning-wisc/LUMINA
>
---
#### [replaced 031] Mil-SCORE: Benchmarking Long-Context Geospatial Reasoning and Planning in Large Language Models
- **分类: cs.CL**

- **简介: 该论文提出MilSCORE，一个用于评估大语言模型在长时序地理空间推理与规划能力的基准。旨在解决现实场景下多源信息整合与复杂决策问题。**

- **链接: [https://arxiv.org/pdf/2601.21826v2](https://arxiv.org/pdf/2601.21826v2)**

> **作者:** Aadi Palnitkar; Mingyang Mao; Nicholas Waytowich; Vinicius G. Goecks; Xiaomin Lin
>
> **摘要:** As large language models (LLMs) are applied to increasingly longer and more complex tasks, there is a growing need for realistic long-context benchmarks that require selective reading and integration of heterogeneous, multi-modal information sources. This need is especially acute for geospatial planning problems, such as those found in planning for large-scale military operations, which demand fast and accurate reasoning over maps, orders, intelligence reports, and other distributed data. To address this gap, we present MilSCORE (Military Scenario Contextual Reasoning), to our knowledge the first scenario-level dataset of expert-authored, multi-hop questions grounded in a complex, simulated military planning scenario used for training. MilSCORE is designed to evaluate high-stakes decision-making and planning, probing LLMs' ability to combine tactical and spatial reasoning across multiple sources and to reason over long-horizon, geospatially rich context. The benchmark includes a diverse set of question types across seven categories targeting both factual recall and multi-step reasoning about constraints, strategy, and spatial analysis. We provide an evaluation protocol and report baseline results for a range of contemporary vision-language models. Our findings highlight substantial headroom on MilSCORE, indicating that current systems struggle with realistic, scenario-level long-context planning, and positioning MilSCORE as a challenging testbed for future work.
>
---
#### [replaced 032] MemeLens: Multilingual Multitask VLMs for Memes
- **分类: cs.AI; cs.CL**

- **简介: 该论文提出MemeLens，一个统一的多语言多任务视觉语言模型，用于理解网络迷因。解决现有研究分散于不同任务和语言的问题，通过整合38个数据集并建立共享分类体系，提升跨领域泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.12539v2](https://arxiv.org/pdf/2601.12539v2)**

> **作者:** Ali Ezzat Shahroor; Mohamed Bayan Kmainasi; Abul Hasnat; Dimitar Dimitrov; Giovanni Da San Martino; Preslav Nakov; Firoj Alam
>
> **备注:** disinformation, misinformation, factuality, harmfulness, fake news, propaganda, hateful meme, multimodality, text, images
>
> **摘要:** Memes are a dominant medium for online communication and manipulation because meaning emerges from interactions between embedded text, imagery, and cultural context. Existing meme research is distributed across tasks (hate, misogyny, propaganda, sentiment, humour) and languages, which limits cross-domain generalization. To address this gap we propose MemeLens, a unified multilingual and multitask explanation-enhanced Vision Language Model (VLM) for meme understanding. We consolidate 38 public meme datasets, filter and map dataset-specific labels into a shared taxonomy of $20$ tasks spanning harm, targets, figurative/pragmatic intent, and affect. We present a comprehensive empirical analysis across modeling paradigms, task categories, and datasets. Our findings suggest that robust meme understanding requires multimodal training, exhibits substantial variation across semantic categories, and remains sensitive to over-specialization when models are fine-tuned on individual datasets rather than trained in a unified setting. We will make the experimental resources and datasets publicly available for the community.
>
---
#### [replaced 033] Merged ChemProt-DrugProt for Relation Extraction from Biomedical Literature
- **分类: cs.CL; cs.IR; q-bio.MN**

- **简介: 该论文属于生物医学关系抽取任务，旨在提升化学-基因关系识别的准确性。通过融合ChemProt和DrugProt数据集，并结合BioBERT与GCN模型，增强模型性能。**

- **链接: [https://arxiv.org/pdf/2405.18605v2](https://arxiv.org/pdf/2405.18605v2)**

> **作者:** Mai H. Nguyen; Shibani Likhite; Jiawei Tang; Darshini Mahendran; Bridget T. McInnes
>
> **摘要:** The extraction of chemical-gene relations plays a pivotal role in understanding the intricate interactions between chemical compounds and genes, with significant implications for drug discovery, disease understanding, and biomedical research. This paper presents a data set created by merging the ChemProt and DrugProt datasets to augment sample counts and improve model accuracy. We evaluate the merged dataset using two state of the art relationship extraction algorithms: Bidirectional Encoder Representations from Transformers (BERT) specifically BioBERT, and Graph Convolutional Networks (GCNs) combined with BioBERT. While BioBERT excels at capturing local contexts, it may benefit from incorporating global information essential for understanding chemical-gene interactions. This can be achieved by integrating GCNs with BioBERT to harness both global and local context. Our results show that by integrating the ChemProt and DrugProt datasets, we demonstrated significant improvements in model performance, particularly in CPR groups shared between the datasets. Incorporating the global context using GCN can help increase the overall precision and recall in some of the CPR groups over using just BioBERT.
>
---
#### [replaced 034] LegalOne: A Family of Foundation Models for Reliable Legal Reasoning
- **分类: cs.CL**

- **简介: 该论文属于法律AI任务，旨在解决LLM在法律领域推理能力不足的问题。通过三阶段方法提升法律推理的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2602.00642v2](https://arxiv.org/pdf/2602.00642v2)**

> **作者:** Haitao Li; Yifan Chen; Shuo Miao; Qian Dong; Jia Chen; Yiran Hu; Junjie Chen; Minghao Qin; Yueyue Wu; Yujia Zhou; Qingyao Ai; Yiqun Liu; Cheng Luo; Quan Zhou; Ya Zhang; Jikun Hu
>
> **备注:** 25 pages, v1
>
> **摘要:** While Large Language Models (LLMs) have demonstrated impressive general capabilities, their direct application in the legal domain is often hindered by a lack of precise domain knowledge and complexity of performing rigorous multi-step judicial reasoning. To address this gap, we present LegalOne, a family of foundational models specifically tailored for the Chinese legal domain. LegalOne is developed through a comprehensive three-phase pipeline designed to master legal reasoning. First, during mid-training phase, we propose Plasticity-Adjusted Sampling (PAS) to address the challenge of domain adaptation. This perplexity-based scheduler strikes a balance between the acquisition of new knowledge and the retention of original capabilities, effectively establishing a robust legal foundation. Second, during supervised fine-tuning, we employ Legal Agentic CoT Distillation (LEAD) to distill explicit reasoning from raw legal texts. Unlike naive distillation, LEAD utilizes an agentic workflow to convert complex judicial processes into structured reasoning trajectories, thereby enforcing factual grounding and logical rigor. Finally, we implement a Curriculum Reinforcement Learning (RL) strategy. Through a progressive reinforcement process spanning memorization, understanding, and reasoning, LegalOne evolves from simple pattern matching to autonomous and reliable legal reasoning. Experimental results demonstrate that LegalOne achieves state-of-the-art performance across a wide range of legal tasks, surpassing general-purpose LLMs with vastly larger parameter counts through enhanced knowledge density and efficiency. We publicly release the LegalOne weights and the LegalKit evaluation framework to advance the field of Legal AI, paving the way for deploying trustworthy and interpretable foundation models in high-stakes judicial applications.
>
---
#### [replaced 035] AlignAtt: Using Attention-based Audio-Translation Alignments as a Guide for Simultaneous Speech Translation
- **分类: cs.CL; cs.LG; cs.SD; eess.AS**

- **简介: 该论文属于语音翻译任务，旨在提升同时语音翻译的性能。通过利用注意力机制生成对齐信息，提出AlignAtt策略，有效提升翻译质量并降低延迟。**

- **链接: [https://arxiv.org/pdf/2305.11408v3](https://arxiv.org/pdf/2305.11408v3)**

> **作者:** Sara Papi; Marco Turchi; Matteo Negri
>
> **摘要:** Attention is the core mechanism of today's most used architectures for natural language processing and has been analyzed from many perspectives, including its effectiveness for machine translation-related tasks. Among these studies, attention resulted to be a useful source of information to get insights about word alignment also when the input text is substituted with audio segments, as in the case of the speech translation (ST) task. In this paper, we propose AlignAtt, a novel policy for simultaneous ST (SimulST) that exploits the attention information to generate source-target alignments that guide the model during inference. Through experiments on the 8 language pairs of MuST-C v1.0, we show that AlignAtt outperforms previous state-of-the-art SimulST policies applied to offline-trained models with gains in terms of BLEU of 2 points and latency reductions ranging from 0.5s to 0.8s across the 8 languages.
>
---
#### [replaced 036] When Domain Pretraining Interferes with Instruction Alignment: An Empirical Study of Adapter Merging in Medical LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文研究医疗大模型中适配器合并时领域预训练与指令对齐的干扰问题，属于模型微调任务。通过两阶段LoRA流程，发现预训练信号影响模型行为，导致评估指标与实际表现不一致，提出验证方法确保合并正确性。**

- **链接: [https://arxiv.org/pdf/2601.18350v3](https://arxiv.org/pdf/2601.18350v3)**

> **作者:** Junyi Zou
>
> **摘要:** Large language models can exhibit surprising adapter interference when combining domain adaptation and instruction alignment in safety-critical settings. We study a two-stage LoRA pipeline for medical LLMs, where domain-oriented pre-training (PT) and supervised fine-tuning (SFT) are trained separately and later merged through weighted adapter merging. We observe that introducing PT signal can systematically alter model behavior and produce reasoning-style outputs, even when evaluation templates explicitly attempt to suppress such behavior. This interference leads to a divergence between surface metrics and reasoning or alignment behavior: BLEU/ROUGE scores drop significantly, while multiple-choice accuracy improves. We further show that small pipeline mistakes can easily misattribute SFT-only behavior to merged models, and provide a lightweight merge-verification routine to ensure correctness and reproducibility. Our findings highlight an interaction between knowledge injection and instruction alignment in adapter-based fine-tuning, with important implications for safety-critical model deployment.
>
---
#### [replaced 037] Proactive defense against LLM Jailbreak
- **分类: cs.CR; cs.CL**

- **简介: 该论文属于安全防护任务，旨在解决LLM被 jailbreak 的问题。提出ProAct框架，通过误导攻击者优化过程，有效阻止迭代式攻击。**

- **链接: [https://arxiv.org/pdf/2510.05052v2](https://arxiv.org/pdf/2510.05052v2)**

> **作者:** Weiliang Zhao; Jinjun Peng; Daniel Ben-Levi; Zhou Yu; Junfeng Yang
>
> **摘要:** The proliferation of powerful large language models (LLMs) has necessitated robust safety alignment, yet these models remain vulnerable to evolving adversarial attacks, including multi-turn jailbreaks that iteratively search for successful queries. Current defenses, which are primarily reactive and static, often fail to handle these iterative attacks. In this paper, we introduce ProAct, a novel proactive defense framework designed to disrupt and mislead these iterative search jailbreak methods. Our core idea is to intentionally mislead these jailbreak methods into thinking that the model has been jailbroken with "spurious responses". These misleading responses provide false signals to the attacker's internal optimization loop, causing the adversarial search to terminate prematurely and effectively jailbreaking the jailbreak. By conducting extensive experiments across state-of-the-art LLMs, jailbreaking frameworks, and safety benchmarks, we demonstrate that our method consistently and significantly reduces attack success rates by up to 94% without affecting utility. When combined with other defense fraeworks, it further reduces the latest attack strategies' success rate to 0%. ProActrepresents an orthogonal defense strategy that serves as an additional guardrail to enhance LLM safety against the most effective jailbreaking attacks.
>
---
#### [replaced 038] NRR-Phi: Text-to-State Mapping for Ambiguity Preservation in LLM Inference
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文提出NRR-Phi框架，解决LLM在模糊输入时过早确定语义的问题。通过文本到状态映射，保留多种解释，提升推理的多样性与准确性。**

- **链接: [https://arxiv.org/pdf/2601.19933v2](https://arxiv.org/pdf/2601.19933v2)**

> **作者:** Kei Saito
>
> **备注:** 17 pages, 3 figures, 5 tables. Part of the NRR research program. v2: Added title prefix NRR-Phi for series identification; standardized reference formatting
>
> **摘要:** Large language models exhibit a systematic tendency toward early semantic commitment: given ambiguous input, they collapse multiple valid interpretations into a single response before sufficient context is available. We present a formal framework for text-to-state mapping ($φ: \mathcal{T} \to \mathcal{S}$) that transforms natural language into a non-collapsing state space where multiple interpretations coexist. The mapping decomposes into three stages: conflict detection, interpretation extraction, and state construction. We instantiate $φ$ with a hybrid extraction pipeline combining rule-based segmentation for explicit conflict markers (adversative conjunctions, hedging expressions) with LLM-based enumeration of implicit ambiguity (epistemic, lexical, structural). On a test set of 68 ambiguous sentences, the resulting states preserve interpretive multiplicity: mean state entropy $H = 1.087$ bits across ambiguity categories, compared to $H = 0$ for collapse-based baselines. We additionally instantiate the rule-based conflict detector for Japanese markers to illustrate cross-lingual portability. This framework extends Non-Resolution Reasoning (NRR) by providing the missing algorithmic bridge between text and the NRR state space, enabling architectural collapse deferment in LLM inference.
>
---
#### [replaced 039] Agentic Search in the Wild: Intents and Trajectory Dynamics from 14M+ Real Search Requests
- **分类: cs.IR; cs.CL**

- **简介: 该论文研究agentic search行为，分析14M+搜索请求，揭示会话模式与证据使用规律，旨在提升搜索效率与效果。**

- **链接: [https://arxiv.org/pdf/2601.17617v2](https://arxiv.org/pdf/2601.17617v2)**

> **作者:** Jingjie Ning; João Coelho; Yibo Kong; Yunfan Long; Bruno Martins; João Magalhães; Jamie Callan; Chenyan Xiong
>
> **摘要:** LLM-powered search agents are increasingly being used for multi-step information seeking tasks, yet the IR community lacks empirical understanding of how agentic search sessions unfold and how retrieved evidence is used. This paper presents a large-scale log analysis of agentic search based on 14.44M search requests (3.97M sessions) collected from DeepResearchGym, i.e. an open-source search API accessed by external agentic clients. We sessionize the logs, assign session-level intents and step-wise query-reformulation labels using LLM-based annotation, and propose Context-driven Term Adoption Rate (CTAR) to quantify whether newly introduced query terms are traceable to previously retrieved evidence. Our analyses reveal distinctive behavioral patterns. First, over 90% of multi-turn sessions contain at most ten steps, and 89% of inter-step intervals fall under one minute. Second, behavior varies by intent. Fact-seeking sessions exhibit high repetition that increases over time, while sessions requiring reasoning sustain broader exploration. Third, agents reuse evidence across steps. On average, 54% of newly introduced query terms appear in the accumulated evidence context, with contributions from earlier steps beyond the most recent retrieval. The findings suggest that agentic search may benefit from repetition-aware early stopping, intent-adaptive retrieval budgets, and explicit cross-step context tracking. We plan to release the anonymized logs to support future research.
>
---
#### [replaced 040] MemoryFormer: Minimize Transformer Computation by Removing Fully-Connected Layers
- **分类: cs.CL**

- **简介: 该论文提出MemoryFormer，旨在降低Transformer模型的计算复杂度。通过移除全连接层，利用内存查找表替代矩阵乘法，提升效率。属于自然语言处理中的模型优化任务。**

- **链接: [https://arxiv.org/pdf/2411.12992v2](https://arxiv.org/pdf/2411.12992v2)**

> **作者:** Ning Ding; Yehui Tang; Haochen Qin; Zhenli Zhou; Chao Xu; Lin Li; Kai Han; Heng Liao; Yunhe Wang
>
> **备注:** NeurIPS 2024. Code available at https://github.com/ningding-o/MemoryFormer
>
> **摘要:** In order to reduce the computational complexity of large language models, great efforts have been made to to improve the efficiency of transformer models such as linear attention and flash-attention. However, the model size and corresponding computational complexity are constantly scaled up in pursuit of higher performance. In this work, we present MemoryFormer, a novel transformer architecture which significantly reduces the computational complexity (FLOPs) from a new perspective. We eliminate nearly all the computations of the transformer model except for the necessary computation required by the multi-head attention operation. This is made possible by utilizing an alternative method for feature transformation to replace the linear projection of fully-connected layers. Specifically, we first construct a group of in-memory lookup tables that store a large amount of discrete vectors to replace the weight matrix used in linear projection. We then use a hash algorithm to retrieve a correlated subset of vectors dynamically based on the input embedding. The retrieved vectors combined together will form the output embedding, which provides an estimation of the result of matrix multiplication operation in a fully-connected layer. Compared to conducting matrix multiplication, retrieving data blocks from memory is a much cheaper operation which requires little computations. We train MemoryFormer from scratch and conduct extensive experiments on various benchmarks to demonstrate the effectiveness of the proposed model.
>
---
#### [replaced 041] Mechanistic Interpretability as Statistical Estimation: A Variance Analysis
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于机制可解释性研究，旨在解决模型解释的稳定性问题。通过方差分析，指出电路发现存在固有不稳定性，提出需加强统计稳健性。**

- **链接: [https://arxiv.org/pdf/2510.00845v3](https://arxiv.org/pdf/2510.00845v3)**

> **作者:** Maxime Méloux; François Portet; Maxime Peyrard
>
> **摘要:** Mechanistic Interpretability (MI) aims to reverse-engineer model behaviors by identifying functional sub-networks. Yet, the scientific validity of these findings depends on their stability. In this work, we argue that circuit discovery is not a standalone task but a statistical estimation problem built upon causal mediation analysis (CMA). We uncover a fundamental instability at this base layer: exact, single-input CMA scores exhibit high intrinsic variance, implying that the causal effect of a component is a volatile random variable rather than a fixed property. We then demonstrate that circuit discovery pipelines inherit this variance and further amplify it. Fast approximation methods, such as Edge Attribution Patching and its successors, introduce additional estimation noise, while aggregating these noisy scores over datasets leads to fragile structural estimates. Consequently, small perturbations in input data or hyperparameters yield vastly different circuits. We systematically decompose these sources of variance and advocate for more rigorous MI practices, prioritizing statistical robustness and routine reporting of stability metrics.
>
---
#### [replaced 042] Reuse your FLOPs: Scaling RL on Hard Problems by Conditioning on Very Off-Policy Prefixes
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文提出PrefixRL，解决强化学习在困难问题上效率低的问题。通过利用非策略前缀进行策略优化，提升样本效率和性能。任务为强化学习中的高效策略训练。**

- **链接: [https://arxiv.org/pdf/2601.18795v2](https://arxiv.org/pdf/2601.18795v2)**

> **作者:** Amrith Setlur; Zijian Wang; Andrew Cohen; Paria Rashidinejad; Sang Michael Xie
>
> **摘要:** Typical reinforcement learning (RL) methods for LLM reasoning waste compute on hard problems, where correct on-policy traces are rare, policy gradients vanish, and learning stalls. To bootstrap more efficient RL, we consider reusing old sampling FLOPs (from prior inference or RL training) in the form of off-policy traces. Standard off-policy methods supervise against off-policy data, causing instabilities during RL optimization. We introduce PrefixRL, where we condition on the prefix of successful off-policy traces and run on-policy RL to complete them, side-stepping off-policy instabilities. PrefixRL boosts the learning signal on hard problems by modulating the difficulty of the problem through the off-policy prefix length. We prove that the PrefixRL objective is not only consistent with the standard RL objective but also more sample efficient. Empirically, we discover back-generalization: training only on prefixed problems generalizes to out-of-distribution unprefixed performance, with learned strategies often differing from those in the prefix. In our experiments, we source the off-policy traces by rejection sampling with the base model, creating a self-improvement loop. On hard reasoning problems, PrefixRL reaches the same training reward 2x faster than the strongest baseline (SFT on off-policy data then RL), even after accounting for the compute spent on the initial rejection sampling, and increases the final reward by 3x. The gains transfer to held-out benchmarks, and PrefixRL is still effective when off-policy traces are derived from a different model family, validating its flexibility in practical settings.
>
---
#### [replaced 043] Self-attention vector output similarities reveal how machines pay attention
- **分类: cs.CL**

- **简介: 该论文研究自注意力机制在自然语言处理中的信息处理方式，旨在量化其学习过程。通过分析BERT-12架构，揭示不同注意力头关注的语义特征及相似性分布规律。**

- **链接: [https://arxiv.org/pdf/2512.21956v2](https://arxiv.org/pdf/2512.21956v2)**

> **作者:** Tal Halevi; Yarden Tzach; Ronit D. Gross; Shalom Rosner; Ido Kanter
>
> **备注:** 23 pages, 14 figures
>
> **摘要:** The self-attention mechanism has significantly advanced the field of natural language processing, facilitating the development of advanced language-learning machines. Although its utility is widely acknowledged, the precise mechanisms of self-attention underlying its advanced learning and the quantitative characterization of this learning process remains an open research question. This study introduces a new approach for quantifying information processing within the self-attention mechanism. The analysis conducted on the BERT-12 architecture reveals that, in the final layers, the attention map focuses on sentence separator tokens, suggesting a practical approach to text segmentation based on semantic features. Based on the vector space emerging from the self-attention heads, a context similarity matrix, measuring the scalar product between two token vectors was derived, revealing distinct similarities between different token vector pairs within each head and layer. The findings demonstrated that different attention heads within an attention block focused on different linguistic characteristics, such as identifying token repetitions in a given text or recognizing a token of common appearance in the text and its surrounding context. This specialization is also reflected in the distribution of distances between token vectors with high similarity as the architecture progresses. The initial attention layers exhibit substantially long-range similarities; however, as the layers progress, a more short-range similarity develops, culminating in a preference for attention heads to create strong similarities within the same sentence. Finally, the behavior of individual heads was analyzed by examining the uniqueness of their most common tokens in their high similarity elements. Each head tends to focus on a unique token from the text and builds similarity pairs centered around it.
>
---
#### [replaced 044] TurkBench: A Benchmark for Evaluating Turkish Large Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文提出TurkBench，一个用于评估土耳其语大语言模型的基准测试。旨在解决多语言模型评估不足的问题，通过21个子任务全面评测模型能力。**

- **链接: [https://arxiv.org/pdf/2601.07020v2](https://arxiv.org/pdf/2601.07020v2)**

> **作者:** Çağrı Toraman; Ahmet Kaan Sever; Ayse Aysu Cengiz; Elif Ecem Arslan; Görkem Sevinç; Mete Mert Birdal; Yusuf Faruk Güldemir; Ali Buğra Kanburoğlu; Sezen Felekoğlu; Osman Gürlek; Sarp Kantar; Birsen Şahin Kütük; Büşra Tufan; Elif Genç; Serkan Coşkun; Gupse Ekin Demir; Muhammed Emin Arayıcı; Olgun Dursun; Onur Gungor; Susan Üsküdarlı; Abdullah Topraksoy; Esra Darıcı
>
> **备注:** Accepted by EACL 2026 SIGTURK
>
> **摘要:** With the recent surge in the development of large language models, the need for comprehensive and language-specific evaluation benchmarks has become critical. While significant progress has been made in evaluating English-language models, benchmarks for other languages, particularly those with unique linguistic characteristics such as Turkish, remain less developed. Our study introduces TurkBench, a comprehensive benchmark designed to assess the capabilities of generative large language models in the Turkish language. TurkBench involves 8,151 data samples across 21 distinct subtasks. These are organized under six main categories of evaluation: Knowledge, Language Understanding, Reasoning, Content Moderation, Turkish Grammar and Vocabulary, and Instruction Following. The diverse range of tasks and the culturally relevant data would provide researchers and developers with a valuable tool for evaluating their models and identifying areas for improvement. We further publish our benchmark for online submissions at https://huggingface.co/turkbench
>
---
#### [replaced 045] From Generative Modeling to Clinical Classification: A GPT-Based Architecture for EHR Notes
- **分类: cs.CL**

- **简介: 该论文属于临床文本分类任务，解决EHR文本建模难题。通过选择性微调GPT模型，减少参数量并提升分类性能。**

- **链接: [https://arxiv.org/pdf/2601.21955v2](https://arxiv.org/pdf/2601.21955v2)**

> **作者:** Fariba Afrin Irany
>
> **备注:** This submission is a full-length research manuscript consisting of 37 pages and 15 figures. The paper presents a GPT-based architecture with selective fine-tuning for clinical text classification, including detailed architectural diagrams, learning curves, and evaluation figures such as ROC curves and confusion matrices
>
> **摘要:** The increasing availability of unstructured clinical narratives in electronic health records (EHRs) has created new opportunities for automated disease characterization, cohort identification, and clinical decision support. However, modeling long, domain-specific clinical text remains challenging due to limited labeled data, severe class imbalance, and the high computational cost of adapting large pretrained language models. This study presents a GPT-based architecture for clinical text classification that adapts a pretrained decoder-only Transformer using a selective fine-tuning strategy. Rather than updating all model parameters, the majority of the GPT-2 backbone is frozen, and training is restricted to the final Transformer block, the final layer normalization, and a lightweight classification head. This approach substantially reduces the number of trainable parameters while preserving the representational capacity required to model complex clinical language. The proposed method is evaluated on radiology reports from the MIMIC-IV-Note dataset using uncertainty-aware CheXpert-style labels derived directly from report text. Experiments cover multiple problem formulations, including multi-label classification of radiographic findings, binary per-label classification under different uncertainty assumptions, and aggregate disease outcome prediction. Across varying dataset sizes, the model exhibits stable convergence behavior and strong classification performance, particularly in settings dominated by non-mention and negated findings. Overall, the results indicate that selective fine-tuning of pretrained generative language models provides an efficient and effective pathway for clinical text classification, enabling scalable adaptation to real-world EHR data while significantly reducing computational complexity.
>
---
#### [replaced 046] WildGraphBench: Benchmarking GraphRAG with Wild-Source Corpora
- **分类: cs.CL**

- **简介: 该论文属于知识图谱与问答任务，旨在解决GraphRAG在真实复杂场景下的评估问题。构建了WildGraphBench基准，使用维基百科数据进行多层级测试。**

- **链接: [https://arxiv.org/pdf/2602.02053v2](https://arxiv.org/pdf/2602.02053v2)**

> **作者:** Pengyu Wang; Benfeng Xu; Licheng Zhang; Shaohan Wang; Mingxuan Du; Chiwei Zhu; Zhendong Mao
>
> **备注:** https://github.com/BstWPY/WildGraphBench
>
> **摘要:** Graph-based Retrieval-Augmented Generation (GraphRAG) organizes external knowledge as a hierarchical graph, enabling efficient retrieval and aggregation of scattered evidence across multiple documents. However, many existing benchmarks for GraphRAG rely on short, curated passages as external knowledge, failing to adequately evaluate systems in realistic settings involving long contexts and large-scale heterogeneous documents. To bridge this gap, we introduce WildGraphBench, a benchmark designed to assess GraphRAG performance in the wild. We leverage Wikipedia's unique structure, where cohesive narratives are grounded in long and heterogeneous external reference documents, to construct a benchmark reflecting real-word scenarios. Specifically, we sample articles across 12 top-level topics, using their external references as the retrieval corpus and citation-linked statements as ground truth, resulting in 1,100 questions spanning three levels of complexity: single-fact QA, multi-fact QA, and section-level summarization. Experiments across multiple baselines reveal that current GraphRAG pipelines help on multi-fact aggregation when evidence comes from a moderate number of sources, but this aggregation paradigm may overemphasize high-level statements at the expense of fine-grained details, leading to weaker performance on summarization tasks. Project page:https://github.com/BstWPY/WildGraphBench.
>
---
#### [replaced 047] Linear representations in language models can change dramatically over a conversation
- **分类: cs.CL; cs.LG**

- **简介: 该论文研究语言模型在对话中的表示动态变化，探讨其线性表示如何随对话演变，揭示模型在不同对话阶段对信息的编码方式会发生显著改变。属于自然语言处理中的模型表示研究任务，旨在理解模型如何适应上下文。**

- **链接: [https://arxiv.org/pdf/2601.20834v2](https://arxiv.org/pdf/2601.20834v2)**

> **作者:** Andrew Kyle Lampinen; Yuxuan Li; Eghbal Hosseini; Sangnie Bhardwaj; Murray Shanahan
>
> **摘要:** Language model representations often contain linear directions that correspond to high-level concepts. Here, we study the dynamics of these representations: how representations evolve along these dimensions within the context of (simulated) conversations. We find that linear representations can change dramatically over a conversation; for example, information that is represented as factual at the beginning of a conversation can be represented as non-factual at the end and vice versa. These changes are content-dependent; while representations of conversation-relevant information may change, generic information is generally preserved. These changes are robust even for dimensions that disentangle factuality from more superficial response patterns, and occur across different model families and layers of the model. These representation changes do not require on-policy conversations; even replaying a conversation script written by an entirely different model can produce similar changes. However, adaptation is much weaker from simply having a sci-fi story in context that is framed more explicitly as such. We also show that steering along a representational direction can have dramatically different effects at different points in a conversation. These results are consistent with the idea that representations may evolve in response to the model playing a particular role that is cued by a conversation. Our findings may pose challenges for interpretability and steering -- in particular, they imply that it may be misleading to use static interpretations of features or directions, or probes that assume a particular range of features consistently corresponds to a particular ground-truth value. However, these types of representational dynamics also point to exciting new research directions for understanding how models adapt to context.
>
---
#### [replaced 048] Rank-and-Reason: Multi-Agent Collaboration Accelerates Zero-Shot Protein Mutation Prediction
- **分类: q-bio.QM; cs.AI; cs.CL**

- **简介: 该论文属于蛋白质工程任务，旨在解决零样本突变预测中忽视生物物理约束的问题。提出Rank-and-Reason框架，通过多智能体协作提升预测准确性与实验可行性。**

- **链接: [https://arxiv.org/pdf/2602.00197v2](https://arxiv.org/pdf/2602.00197v2)**

> **作者:** Yang Tan; Yuanxi Yu; Can Wu; Bozitao Zhong; Mingchen Li; Guisheng Fan; Jiankang Zhu; Yafeng Liang; Nanqing Dong; Liang Hong
>
> **备注:** 22 pages, 5 figures, 15 tables
>
> **摘要:** Zero-shot mutation prediction is vital for low-resource protein engineering, yet existing protein language models (PLMs) often yield statistically confident results that ignore fundamental biophysical constraints. Currently, selecting candidates for wet-lab validation relies on manual expert auditing of PLM outputs, a process that is inefficient, subjective, and highly dependent on domain expertise. To address this, we propose Rank-and-Reason (VenusRAR), a two-stage agentic framework to automate this workflow and maximize expected wet-lab fitness. In the Rank-Stage, a Computational Expert and Virtual Biologist aggregate a context-aware multi-modal ensemble, establishing a new Spearman correlation record of 0.551 (vs. 0.518) on ProteinGym. In the Reason-Stage, an agentic Expert Panel employs chain-of-thought reasoning to audit candidates against geometric and structural constraints, improving the Top-5 Hit Rate by up to 367% on ProteinGym-DMS99. The wet-lab validation on Cas12i3 nuclease further confirms the framework's efficacy, achieving a 46.7% positive rate and identifying two novel mutants with 4.23-fold and 5.05-fold activity improvements. Code and datasets are released on GitHub (https://github.com/ai4protein/VenusRAR/).
>
---
#### [replaced 049] Reward Model Interpretability via Optimal and Pessimal Tokens
- **分类: cs.CL; cs.AI; cs.CY; cs.LG**

- **简介: 该论文属于奖励模型可解释性研究，旨在揭示奖励模型如何编码人类价值观。通过分析不同token的评分，发现模型间存在显著差异与偏差，挑战了奖励模型的可靠性与公平性。**

- **链接: [https://arxiv.org/pdf/2506.07326v2](https://arxiv.org/pdf/2506.07326v2)**

> **作者:** Brian Christian; Hannah Rose Kirk; Jessica A. F. Thompson; Christopher Summerfield; Tsvetomira Dumbalska
>
> **备注:** Accepted for publication in Proceedings of the 2025 ACM Conference on Fairness, Accountability, and Transparency (FAccT '25), to appear June 2025
>
> **摘要:** Reward modeling has emerged as a crucial component in aligning large language models with human values. Significant attention has focused on using reward models as a means for fine-tuning generative models. However, the reward models themselves -- which directly encode human value judgments by turning prompt-response pairs into scalar rewards -- remain relatively understudied. We present a novel approach to reward model interpretability through exhaustive analysis of their responses across their entire vocabulary space. By examining how different reward models score every possible single-token response to value-laden prompts, we uncover several striking findings: (i) substantial heterogeneity between models trained on similar objectives, (ii) systematic asymmetries in how models encode high- vs low-scoring tokens, (iii) significant sensitivity to prompt framing that mirrors human cognitive biases, and (iv) overvaluation of more frequent tokens. We demonstrate these effects across ten recent open-source reward models of varying parameter counts and architectures. Our results challenge assumptions about the interchangeability of reward models, as well as their suitability as proxies of complex and context-dependent human values. We find that these models can encode concerning biases toward certain identity groups, which may emerge as unintended consequences of harmlessness training -- distortions that risk propagating through the downstream large language models now deployed to millions.
>
---
#### [replaced 050] DEER: A Benchmark for Evaluating Deep Research Agents on Expert Report Generation
- **分类: cs.CL**

- **简介: 该论文属于报告评估任务，旨在解决深度研究系统生成报告质量评估难题。提出DEER基准，包含多维评估标准和证据验证机制，以提升评估的准确性和全面性。**

- **链接: [https://arxiv.org/pdf/2512.17776v3](https://arxiv.org/pdf/2512.17776v3)**

> **作者:** Janghoon Han; Heegyu Kim; Changho Lee; Dahm Lee; Min Hyung Park; Hosung Song; Stanley Jungkyu Choi; Moontae Lee; Honglak Lee
>
> **备注:** Work in progress
>
> **摘要:** Recent advances in large language models have enabled deep research systems that generate expert-level reports through multi-step reasoning and evidence-based synthesis. However, evaluating such reports remains challenging: report quality is multifaceted, making it difficult to determine what to assess and by what criteria; LLM-based judges may miss errors that require domain expertise to identify; and because deep research relies on retrieved evidence, report-wide claim verification is also necessary. To address these issues, we propose DEER, a benchmark for evaluating expert-level deep research reports. DEER systematizes evaluation criteria with an expert-developed taxonomy (7 dimensions, 25 subdimensions) operationalized as 101 fine-grained rubric items. We also provide task-specific Expert Evaluation Guidance to support LLM-based judging. Alongside rubric-based assessment, we propose a claim verification architecture that verifies both cited and uncited claims and quantifies evidence quality. Experiments show that while current deep research systems can produce structurally plausible reports that cite external evidence, there is room for improvement in fulfilling expert-level user requests and achieving logical completeness. Beyond simple performance comparisons, DEER makes system strengths and limitations interpretable and provides diagnostic signals for improvement.
>
---
#### [replaced 051] Inferring Scientific Cross-Document Coreference and Hierarchy with Definition-Augmented Relational Reasoning
- **分类: cs.CL**

- **简介: 该论文属于科学文本中的跨文档共指与层次关系推断任务，解决长尾技术概念的模糊性和多样性问题。通过生成上下文定义和关系定义，提升模型对科学概念间关系的推理能力。**

- **链接: [https://arxiv.org/pdf/2409.15113v3](https://arxiv.org/pdf/2409.15113v3)**

> **作者:** Lior Forer; Tom Hope
>
> **备注:** Accepted to TACL. Pre-MIT Press publication version
>
> **摘要:** We address the fundamental task of inferring cross-document coreference and hierarchy in scientific texts, which has important applications in knowledge graph construction, search, recommendation and discovery. Large Language Models (LLMs) can struggle when faced with many long-tail technical concepts with nuanced variations. We present a novel method which generates context-dependent definitions of concept mentions by retrieving full-text literature, and uses the definitions to enhance detection of cross-document relations. We further generate relational definitions, which describe how two concept mentions are related or different, and design an efficient re-ranking approach to address the combinatorial explosion involved in inferring links across papers. In both fine-tuning and in-context learning settings, we achieve large gains in performance on data subsets with high amount of different surfaces forms and ambiguity, that are challenging for models. We provide analysis of generated definitions, shedding light on the relational reasoning ability of LLMs over fine-grained scientific concepts.
>
---
#### [replaced 052] Problem Solved? Information Extraction Design Space for Layout-Rich Documents using LLMs
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于信息抽取任务，解决布局丰富文档中使用大语言模型的信息抽取问题。通过设计实验框架，探索数据结构、模型互动和输出优化等关键问题，提升抽取性能。**

- **链接: [https://arxiv.org/pdf/2502.18179v4](https://arxiv.org/pdf/2502.18179v4)**

> **作者:** Gaye Colakoglu; Gürkan Solmaz; Jonathan Fürst
>
> **备注:** accepted at EMNLP'25
>
> **摘要:** This paper defines and explores the design space for information extraction (IE) from layout-rich documents using large language models (LLMs). The three core challenges of layout-aware IE with LLMs are 1) data structuring, 2) model engagement, and 3) output refinement. Our study investigates the sub-problems and methods within these core challenges, such as input representation, chunking, prompting, selection of LLMs, and multimodal models. It examines the effect of different design choices through LayIE-LLM, a new, open-source, layout-aware IE test suite, benchmarking against traditional, fine-tuned IE models. The results on two IE datasets show that LLMs require adjustment of the IE pipeline to achieve competitive performance: the optimized configuration found with LayIE-LLM achieves 13.3--37.5 F1 points more than a general-practice baseline configuration using the same LLM. To find a well-working configuration, we develop a one-factor-at-a-time (OFAT) method that achieves near-optimal results. Our method is only 0.8--1.8 points lower than the best full factorial exploration with a fraction (2.8%) of the required computation. Overall, we demonstrate that, if well-configured, general-purpose LLMs match the performance of specialized models, providing a cost-effective, finetuning-free alternative. Our test-suite is available at https://github.com/gayecolakoglu/LayIE-LLM.
>
---
#### [replaced 053] Align to Structure: Aligning Large Language Models with Structural Information
- **分类: cs.CL; cs.AI; cs.LG**

- **简介: 该论文属于自然语言生成任务，旨在解决长文本生成缺乏结构和连贯性的问题。通过引入结构对齐方法，提升模型生成文本的组织性和逻辑性。**

- **链接: [https://arxiv.org/pdf/2504.03622v2](https://arxiv.org/pdf/2504.03622v2)**

> **作者:** Zae Myung Kim; Anand Ramachandran; Farideh Tavazoee; Joo-Kyung Kim; Oleg Rokhlenko; Dongyeop Kang
>
> **备注:** Accepted to AAAI 2026 AIA
>
> **摘要:** Generating long, coherent text remains a challenge for large language models (LLMs), as they lack hierarchical planning and structured organization in discourse generation. We introduce Structural Alignment, a novel method that aligns LLMs with human-like discourse structures to enhance long-form text generation. By integrating linguistically grounded discourse frameworks into reinforcement learning, our approach guides models to produce coherent and well-organized outputs. We employ a dense reward scheme within a Proximal Policy Optimization framework, assigning fine-grained, token-level rewards based on the discourse distinctiveness relative to human writing. Two complementary reward models are evaluated: the first improves readability by scoring surface-level textual features to provide explicit structuring, while the second reinforces deeper coherence and rhetorical sophistication by analyzing global discourse patterns through hierarchical discourse motifs, outperforming both standard and RLHF-enhanced models in tasks such as essay generation and long-document summarization. All training data and code will be publicly shared at https://github.com/minnesotanlp/struct_align.
>
---
#### [replaced 054] From Deferral to Learning: Online In-Context Knowledge Distillation for LLM Cascades
- **分类: cs.AI; cs.CL**

- **简介: 该论文属于大模型优化任务，解决静态LLM级联效率低的问题。提出Inter-Cascade框架，通过在线知识蒸馏提升弱模型性能，减少强模型调用。**

- **链接: [https://arxiv.org/pdf/2509.22984v2](https://arxiv.org/pdf/2509.22984v2)**

> **作者:** Yu Wu; Shuo Wu; Ye Tao; Yansong Li; Anand D. Sarwate
>
> **备注:** 32 pages, 6 figures, 23 tables, under review
>
> **摘要:** Standard LLM cascades improve efficiency by deferring difficult queries from weak to strong models. However, these systems are typically static: when faced with repeated or semantically similar queries, they redundantly consult the expensive model, failing to adapt during inference. To address this, we propose Inter-Cascade, an online, interactive framework that transforms the strong model from a temporary helper into a long-term teacher. In our approach, when the strong model resolves a deferred query, it generates a generalized, reusable problem-solving strategy. These strategies are stored in a dynamic repository and retrieved via similarity matching to augment the weak model's context for future queries. This enables the weak model to learn on the job without expensive parameter fine-tuning. We theoretically show that this mechanism improves the weak model's confidence calibration. Empirically, Inter-Cascade outperforms standard cascades on multiple benchmarks, improving weak model and overall system accuracy by up to 33.06 percent and 6.35 percent, while reducing strong model calls by up to 48.05 percent and saving fee by up to 49.63 percent. Inter-Cascade demonstrates effective in-context knowledge transfer between LLMs and provides a general, scalable framework applicable to both open-source and API-based LLMs.
>
---
#### [replaced 055] Modeling Sarcastic Speech: Semantic and Prosodic Cues in a Speech Synthesis Framework
- **分类: cs.CL; cs.SD; eess.AS**

- **简介: 该论文属于自然语言处理任务，旨在研究讽刺语义与语气的协同作用。通过结合语义模型和语气数据，构建讽刺语音合成框架，验证两者对讽刺识别的增强效果。**

- **链接: [https://arxiv.org/pdf/2510.07096v2](https://arxiv.org/pdf/2510.07096v2)**

> **作者:** Zhu Li; Yuqing Zhang; Xiyuan Gao; Shekhar Nayak; Matt Coler
>
> **摘要:** Sarcasm is a pragmatic phenomenon in which speakers convey meanings that diverge from literal content, relying on an interaction between semantics and prosodic expression. However, how these cues jointly contribute to the recognition of sarcasm remains poorly understood. We propose a computational framework that models sarcasm as the integration of semantic interpretation and prosodic realization. Semantic cues are derived from an LLaMA 3 model fine-tuned to capture discourse-level markers of sarcastic intent, while prosodic cues are extracted through semantically aligned utterances drawn from a database of sarcastic speech, providing prosodic exemplars of sarcastic delivery. Using a speech synthesis testbed, perceptual evaluations demonstrate that both semantic and prosodic cues independently enhance listeners' perception of sarcasm, with the strongest effects emerging when the two are combined. These findings highlight the complementary roles of semantics and prosody in pragmatic interpretation and illustrate how modeling can shed light on the mechanisms underlying sarcastic communication.
>
---
#### [replaced 056] Advancing AI Research Assistants with Expert-Involved Learning
- **分类: cs.AI; cs.CL; cs.IR**

- **简介: 该论文提出ARIEL框架，用于评估和优化AI在生物医学领域的表现，解决模型可靠性问题，通过专家任务测试文本摘要和图像理解能力。**

- **链接: [https://arxiv.org/pdf/2505.04638v4](https://arxiv.org/pdf/2505.04638v4)**

> **作者:** Tianyu Liu; Simeng Han; Hanchen Wang; Xiao Luo; Pan Lu; Biqing Zhu; Yuge Wang; Keyi Li; Jiapeng Chen; Rihao Qu; Yufeng Liu; Xinyue Cui; Aviv Yaish; Yuhang Chen; Minsheng Hao; Chuhan Li; Kexing Li; Arman Cohan; Hua Xu; Mark Gerstein; James Zou; Hongyu Zhao
>
> **备注:** 36 pages, 7 figures
>
> **摘要:** Large language models (LLMs) and large multimodal models (LMMs) promise to accelerate biomedical discovery, yet their reliability remains unclear. We introduce ARIEL (AI Research Assistant for Expert-in-the-Loop Learning), an open-source evaluation and optimization framework that pairs a curated multimodal biomedical corpus with expert-vetted tasks to probe two capabilities: full-length article summarization and fine-grained figure interpretation. Using uniform protocols and blinded PhD-level evaluation, we find that state-of-the-art models generate fluent but incomplete summaries, whereas LMMs struggle with detailed visual reasoning. We later observe that prompt engineering and lightweight fine-tuning substantially improve textual coverage, and a compute-scaled inference strategy enhances visual question answering. We build an ARIEL agent that integrates textual and visual cues, and we show it can propose testable mechanistic hypotheses. ARIEL delineates current strengths and limitations of foundation models, and provides a reproducible platform for advancing trustworthy AI in biomedicine.
>
---
#### [replaced 057] A2D: Any-Order, Any-Step Safety Alignment for Diffusion Language Models
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于安全对齐任务，旨在解决扩散语言模型中任意位置生成有害内容的问题。提出A2D方法，在token级别检测并阻止有害输出。**

- **链接: [https://arxiv.org/pdf/2509.23286v2](https://arxiv.org/pdf/2509.23286v2)**

> **作者:** Wonje Jeung; Sangyeon Yoon; Yoonjun Cho; Dongjae Jeon; Sangwoo Shin; Hyesoo Hong; Albert No
>
> **备注:** Accepted at ICLR 2026. Code and models are available at https://ai-isl.github.io/A2D
>
> **摘要:** Diffusion large language models (dLLMs) enable any-order generation, but this flexibility enlarges the attack surface: harmful spans may appear at arbitrary positions, and template-based prefilling attacks such as DIJA bypass response-level refusals. We introduce A2D (Any-Order, Any-Step Defense), a token-level alignment method that aligns dLLMs to emit an [EOS] refusal signal whenever harmful content arises. By aligning safety directly at the token-level under randomized masking, A2D achieves robustness to both any-decoding-order and any-step prefilling attacks under various conditions. It also enables real-time monitoring: dLLMs may begin a response but automatically terminate if unsafe continuation emerges. On safety benchmarks, A2D consistently prevents the generation of harmful outputs, slashing DIJA success rates from over 80% to near-zero (1.3% on LLaDA-8B-Instruct, 0.0% on Dream-v0-Instruct-7B), and thresholded [EOS] probabilities allow early rejection, yielding up to 19.3x faster safe termination.
>
---
#### [replaced 058] Are you going to finish that? A Practical Study of the Partial Token Problem
- **分类: cs.CL**

- **简介: 该论文属于自然语言处理任务，研究语言模型在用户输入不完整时产生的部分标记问题。通过实验分析问题严重性，并提出缓解方法。**

- **链接: [https://arxiv.org/pdf/2601.23223v2](https://arxiv.org/pdf/2601.23223v2)**

> **作者:** Hao Xu; Alisa Liu; Jonathan Hayase; Yejin Choi; Noah A. Smith
>
> **摘要:** Language models (LMs) are trained over sequences of tokens, whereas users interact with LMs via text. This mismatch gives rise to the partial token problem, which occurs when a user ends their prompt in the middle of the expected next-token, leading to distorted next-token predictions. Although this issue has been studied using arbitrary character prefixes, its prevalence and severity in realistic prompts respecting word boundaries remains underexplored. In this work, we identify three domains where token and "word" boundaries often do not line up: languages that do not use whitespace, highly compounding languages, and code. In Chinese, for example, up to 25% of word boundaries do not line up with token boundaries, making even natural, word-complete prompts susceptible to this problem. We systematically construct semantically natural prompts ending with a partial tokens; in experiments, we find that they comprise a serious failure mode: frontier LMs consistently place three orders of magnitude less probability on the correct continuation compared to when the prompt is "backed-off" to be token-aligned. This degradation does not diminish with scale and often worsens for larger models. Finally, we evaluate inference-time mitigations to the partial token problem and validate the effectiveness of recent exact solutions. Overall, we demonstrate the scale and severity of probability distortion caused by tokenization in realistic use cases, and provide practical recommentions for model inference providers.
>
---
#### [replaced 059] Broken Tokens? Your Language Model can Secretly Handle Non-Canonical Tokenizations
- **分类: cs.CL**

- **简介: 论文研究语言模型对非规范分词的鲁棒性，探讨其在不同分词方式下的表现。发现模型对非规范分词具有较强适应能力，且在某些情况下可提升性能。任务为语言模型的分词鲁棒性分析与优化。**

- **链接: [https://arxiv.org/pdf/2506.19004v2](https://arxiv.org/pdf/2506.19004v2)**

> **作者:** Brian Siyuan Zheng; Alisa Liu; Orevaoghene Ahia; Jonathan Hayase; Yejin Choi; Noah A. Smith
>
> **备注:** NeurIPS 2025 (spotlight)
>
> **摘要:** Modern tokenizers employ deterministic algorithms to map text into a single "canonical" token sequence, yet the same string can be encoded as many non-canonical tokenizations using the tokenizer vocabulary. In this work, we investigate the robustness of LMs to text encoded with non-canonical tokenizations entirely unseen during training. Surprisingly, when evaluated across 20 benchmarks, we find that instruction-tuned models retain up to 93.4% of their original performance when given a randomly sampled tokenization, and 90.8% with character-level tokenization. We see that overall stronger models tend to be more robust, and robustness diminishes as the tokenization departs farther from the canonical form. Motivated by these results, we then identify settings where non-canonical tokenization schemes can *improve* performance, finding that character-level segmentation improves string manipulation and code understanding tasks by up to +14%, and right-aligned digit grouping enhances large-number arithmetic by +33%. Finally, we investigate the source of this robustness, finding that it arises in the instruction-tuning phase. We show that while both base and post-trained models grasp the semantics of non-canonical tokenizations (perceiving them as containing misspellings), base models try to mimic the imagined mistakes and degenerate into nonsensical output, while post-trained models are committed to fluent responses. Overall, our findings suggest that models are less tied to their tokenizer than previously believed, and demonstrate the promise of intervening on tokenization at inference time to boost performance.
>
---
#### [replaced 060] Don't Overthink it. Preferring Shorter Thinking Chains for Improved LLM Reasoning
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于大语言模型推理任务，旨在解决长思考链导致计算成本高且效果未必更好的问题。通过实验发现短思考链更有效，并提出short-@k方法提升效率与准确率。**

- **链接: [https://arxiv.org/pdf/2505.17813v2](https://arxiv.org/pdf/2505.17813v2)**

> **作者:** Michael Hassid; Gabriel Synnaeve; Yossi Adi; Roy Schwartz
>
> **摘要:** Reasoning large language models (LLMs) heavily rely on scaling test-time compute to perform complex reasoning tasks by generating extensive "thinking" chains. While demonstrating impressive results, this approach incurs significant computational costs and inference time. In this work, we challenge the assumption that long thinking chains results in better reasoning capabilities. We first demonstrate that shorter reasoning chains within individual questions are significantly more likely to yield correct answers - up to 34.5% more accurate than the longest chain sampled for the same question. Based on these results, we suggest short-m@k, a novel reasoning LLM inference method. Our method executes k independent generations in parallel and halts computation once the first m thinking processes are done. The final answer is chosen using majority voting among these m chains. Basic short-1@k demonstrates similar or even superior performance over standard majority voting in low-compute settings - using up to 40% fewer thinking tokens. short-3@k, while slightly less efficient than short-1@k, consistently surpasses majority voting across all compute budgets, while still being substantially faster (up to 33% wall time reduction). To further validate our findings, we finetune LLMs using short, long, and randomly selected reasoning chains. We then observe that training on the shorter ones leads to better performance. Our findings suggest rethinking current methods of test-time compute in reasoning LLMs, emphasizing that longer "thinking" does not necessarily translate to improved performance and can, counter-intuitively, lead to degraded results.
>
---
#### [replaced 061] Zero2Text: Zero-Training Cross-Domain Inversion Attacks on Textual Embeddings
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出Zero2Text，解决文本嵌入逆向攻击问题，属于隐私安全任务。通过无需训练的在线对齐方法，在跨领域场景下高效恢复文本。**

- **链接: [https://arxiv.org/pdf/2602.01757v2](https://arxiv.org/pdf/2602.01757v2)**

> **作者:** Doohyun Kim; Donghwa Kang; Kyungjae Lee; Hyeongboo Baek; Brent Byunghoon Kang
>
> **备注:** 10 pages
>
> **摘要:** The proliferation of retrieval-augmented generation (RAG) has established vector databases as critical infrastructure, yet they introduce severe privacy risks via embedding inversion attacks. Existing paradigms face a fundamental trade-off: optimization-based methods require computationally prohibitive queries, while alignment-based approaches hinge on the unrealistic assumption of accessible in-domain training data. These constraints render them ineffective in strict black-box and cross-domain settings. To dismantle these barriers, we introduce Zero2Text, a novel training-free framework based on recursive online alignment. Unlike methods relying on static datasets, Zero2Text synergizes LLM priors with a dynamic ridge regression mechanism to iteratively align generation to the target embedding on-the-fly. We further demonstrate that standard defenses, such as differential privacy, fail to effectively mitigate this adaptive threat. Extensive experiments across diverse benchmarks validate Zero2Text; notably, on MS MARCO against the OpenAI victim model, it achieves 1.8x higher ROUGE-L and 6.4x higher BLEU-2 scores compared to baselines, recovering sentences from unknown domains without a single leaked data pair.
>
---
#### [replaced 062] When to Trust: A Causality-Aware Calibration Framework for Accurate Knowledge Graph Retrieval-Augmented Generation
- **分类: cs.CL**

- **简介: 该论文属于知识图谱增强生成任务，解决KG-RAG模型过度自信的问题，提出Ca2KG框架提升预测校准性。**

- **链接: [https://arxiv.org/pdf/2601.09241v2](https://arxiv.org/pdf/2601.09241v2)**

> **作者:** Jing Ren; Bowen Li; Ziqi Xu; Xikun Zhang; Haytham Fayek; Xiaodong Li
>
> **备注:** Accepted by WWW 2026
>
> **摘要:** Knowledge Graph Retrieval-Augmented Generation (KG-RAG) extends the RAG paradigm by incorporating structured knowledge from knowledge graphs, enabling Large Language Models (LLMs) to perform more precise and explainable reasoning. While KG-RAG improves factual accuracy in complex tasks, existing KG-RAG models are often severely overconfident, producing high-confidence predictions even when retrieved sub-graphs are incomplete or unreliable, which raises concerns for deployment in high-stakes domains. To address this issue, we propose Ca2KG, a Causality-aware Calibration framework for KG-RAG. Ca2KG integrates counterfactual prompting, which exposes retrieval-dependent uncertainties in knowledge quality and reasoning reliability, with a panel-based re-scoring mechanism that stabilises predictions across interventions. Extensive experiments on two complex QA datasets demonstrate that Ca2KG consistently improves calibration while maintaining or even enhancing predictive accuracy.
>
---
#### [replaced 063] CP-Agent: Agentic Constraint Programming
- **分类: cs.AI; cs.CL; cs.LG; cs.SE**

- **简介: 该论文提出CP-Agent，解决自然语言到约束模型的翻译问题。通过代理流程和代码执行迭代优化模型，在101个基准问题上实现100%准确率。**

- **链接: [https://arxiv.org/pdf/2508.07468v3](https://arxiv.org/pdf/2508.07468v3)**

> **作者:** Stefan Szeider
>
> **摘要:** The translation of natural language to formal constraint models requires expertise in the problem domain and modeling frameworks. To explore the effectiveness of agentic workflows, we propose CP-Agent, a Python coding agent that uses the ReAct framework with a persistent IPython kernel. We provide the relevant domain knowledge as a project prompt of under 50 lines. The algorithm works by iteratively executing code, observing the solver's feedback, and refining constraint models based on execution results. We evaluate CP-Agent on 101 constraint programming problems from CP-Bench. We made minor changes to the benchmark to address systematic ambiguities in the problem specifications and errors in the ground-truth models. On the clarified benchmark, CP-Agent achieves perfect accuracy on all 101 problems. Our experiments show that minimal guidance outperforms detailed procedural scaffolding. Our experiments also show that explicit task management tools can have both positive and negative effects on focused modeling tasks.
>
---
#### [replaced 064] PluriHarms: Benchmarking the Full Spectrum of Human Judgments on AI Harm
- **分类: cs.CY; cs.AI; cs.CL**

- **简介: 该论文属于AI安全任务，旨在解决现有框架无法处理人类对AI危害判断分歧的问题。通过构建PluriHarms基准，系统研究人类在危害和意见上的多样性。**

- **链接: [https://arxiv.org/pdf/2601.08951v2](https://arxiv.org/pdf/2601.08951v2)**

> **作者:** Jing-Jing Li; Joel Mire; Eve Fleisig; Valentina Pyatkin; Anne Collins; Maarten Sap; Sydney Levine
>
> **摘要:** Current AI safety frameworks, which often treat harmfulness as binary, lack the flexibility to handle borderline cases where humans meaningfully disagree. To build more pluralistic systems, it is essential to move beyond consensus and instead understand where and why disagreements arise. We introduce PluriHarms, a benchmark designed to systematically study human harm judgments across two key dimensions -- the harm axis (benign to harmful) and the agreement axis (agreement to disagreement). Our scalable framework generates prompts that capture diverse AI harms and human values while targeting cases with high disagreement rates, validated by human data. The benchmark includes 150 prompts with 15,000 ratings from 100 human annotators, enriched with demographic and psychological traits and prompt-level features of harmful actions, effects, and values. Our analyses show that prompts that relate to imminent risks and tangible harms amplify perceived harmfulness, while annotator traits (e.g., toxicity experience, education) and their interactions with prompt content explain systematic disagreement. We benchmark AI safety models and alignment methods on PluriHarms, finding that while personalization significantly improves prediction of human harm judgments, considerable room remains for future progress. By explicitly targeting value diversity and disagreement, our work provides a principled benchmark for moving beyond "one-size-fits-all" safety toward pluralistically safe AI.
>
---
#### [replaced 065] PRISM: Deriving a White-Box Transformer as a Signal-Noise Decomposition Operator via Maximum Coding Rate Reduction
- **分类: cs.LG; cs.AI; cs.CL; physics.data-an**

- **简介: 该论文提出Prism，一种基于最大编码率减少的可解释Transformer架构，解决模型不可解释性问题，通过信号-噪声分解提升可解释性与性能。**

- **链接: [https://arxiv.org/pdf/2601.15540v2](https://arxiv.org/pdf/2601.15540v2)**

> **作者:** Dongchen Huang
>
> **备注:** 12 pages, 6 figures. Derives Transformer as a signal-noise decomposition operator via Maximizing Coding Rate Reduction. Identifies 'Attention Sink' as spectral resonance (Arnold Tongues) and proposes $π$-RoPE for dynamical stability
>
> **摘要:** Deep learning models, particularly Transformers, are often criticized as "black boxes" and lack interpretability. We propose Prism, a white-box attention-based architecture derived from the principles of Maximizing Coding Rate Reduction ($\text{MCR}^2$). By modeling the attention mechanism as a gradient ascent process on a distinct signal-noise manifold, we introduce a specific irrational frequency separation ($π$-RoPE) to enforce incoherence between signal (semantic) and noise (syntactic) subspaces. We show empirical evidence that these geometric inductive biases can induce unsupervised functional disentanglement alone. Prism spontaneously specializes its attention heads into spectrally distinct regimes: low-frequency heads capturing long-range causal dependencies (signal) and high-frequency heads handling local syntactic constraints and structural artifacts. To provide a theoretical grounding for these spectral phenomena, we draw an analogy between attention mechanism and a Hamiltonian dynamical system and identify that the standard geometric progression of Rotary Positional Embeddings (RoPE) induces dense resonance networks (Arnold Tongues), leading to feature rank collapse. Empirical validation on 124M-parameter models trained on OpenWebText demonstrates that Prism spontaneously isolates the Attention Sink pathology and maintains isentropic information flow across layers. Further, we suggest a physics-informed plug-and-play intervention KAM-RoPE for large language models (LLMs). Our results suggest that interpretability and performance can be unified through principled geometric construction, offering a theoretically grounded alternative to heuristic architectural modifications
>
---
#### [replaced 066] V2P-Bench: Evaluating Video-Language Understanding with Visual Prompts for Better Human-Model Interaction
- **分类: cs.CV; cs.AI; cs.CL**

- **简介: 该论文属于视频语言理解任务，旨在解决现有基准依赖文本提示导致交互效率低的问题。提出V2P-Bench，通过视觉提示提升模型性能与用户体验。**

- **链接: [https://arxiv.org/pdf/2503.17736v2](https://arxiv.org/pdf/2503.17736v2)**

> **作者:** Yiming Zhao; Yu Zeng; Yukun Qi; YaoYang Liu; Xikun Bao; Lin Chen; Zehui Chen; Qing Miao; Chenxi Liu; Jie Zhao; Feng Zhao
>
> **备注:** Project Page: https://vlm-reasoning.github.io/V2P-Bench/
>
> **摘要:** Large Vision-Language Models (LVLMs) have made significant strides in the field of video understanding in recent times. Nevertheless, existing video benchmarks predominantly rely on text prompts for evaluation, which often require complex referential language and diminish both the accuracy and efficiency of human model interaction in turn. To address this limitation, we propose V2P-Bench, a robust and comprehensive benchmark for evaluating the ability of LVLMs to understand Video Visual Prompts in human model interaction scenarios. V2P-Bench consists of 980 videos and 1172 well-structured high-quality QA pairs, each paired with manually annotated visual prompt frames. The benchmark spans three main tasks and twelve categories, thereby enabling fine-grained, instance-level evaluation. Through an in-depth analysis of current LVLMs, we identify several key findings: 1) Visual prompts are both more model-friendly and user-friendly in interactive scenarios than text prompts, leading to significantly improved model performance and enhanced user experience. 2) Models are reasonably capable of zero-shot understanding of visual prompts, but struggle with spatiotemporal understanding. Even o1 achieves only 71.8%, far below the human expert score of 88.3%, while most open-source models perform below 60%. 3) LVLMs exhibit pervasive Hack Phenomena in video question answering tasks, which become more pronounced as video length increases and frame sampling density decreases, thereby inflating performance scores artificially. We anticipate that V2P-Bench will not only shed light on these challenges but also serve as a foundational tool for advancing human model interaction and improving the evaluation of video understanding.
>
---
#### [replaced 067] Sentence Curve Language Models
- **分类: cs.CL; cs.LG**

- **简介: 该论文提出句子曲线语言模型（SCLM），解决传统语言模型对全局结构建模不足的问题。通过预测连续句子曲线代替静态词嵌入，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.01807v2](https://arxiv.org/pdf/2602.01807v2)**

> **作者:** DongNyeong Heo; Heeyoul Choi
>
> **摘要:** Language models (LMs) are a central component of modern AI systems, and diffusion-based language models (DLMs) have recently emerged as a competitive alternative. Both paradigms rely on word embeddings not only to represent the input sentence, but also to represent the target sentence that backbone models are trained to predict. We argue that such static embedding of the target word is insensitive to neighboring words, encouraging locally accurate word prediction while neglecting global structure across the target sentence. To address this limitation, we propose a continuous sentence representation, termed sentence curve, defined as a spline curve whose control points affect multiple words in the sentence. Based on this representation, we introduce sentence curve language model (SCLM), which extends DLMs to predict sentence curves instead of the static word embeddings. We theoretically show that sentence curve prediction induces a regularization effect that promotes global structure modeling, and characterize how different sentence curve types affect this behavior. Empirically, SCLM achieves SOTA performance among DLMs on IWSLT14 and WMT14, shows stable training without burdensome knowledge distillation, and demonstrates promising potential compared to discrete DLMs on LM1B.
>
---
#### [replaced 068] Understanding Verbatim Memorization in LLMs Through Circuit Discovery
- **分类: cs.CL**

- **简介: 该论文属于模型可解释性任务，旨在理解大语言模型中的逐字记忆机制。通过分析电路结构，识别记忆启动与维持的路径，揭示记忆行为差异及跨领域适应性。**

- **链接: [https://arxiv.org/pdf/2506.21588v2](https://arxiv.org/pdf/2506.21588v2)**

> **作者:** Ilya Lasy; Peter Knees; Stefan Woltran
>
> **备注:** The First Workshop on Large Language Model Memorization @ ACL 2025, Vienna, August 1st, 2025
>
> **摘要:** Underlying mechanisms of memorization in LLMs -- the verbatim reproduction of training data -- remain poorly understood. What exact part of the network decides to retrieve a token that we would consider as start of memorization sequence? How exactly is the models' behaviour different when producing memorized sentence vs non-memorized? In this work we approach these questions from mechanistic interpretability standpoint by utilizing transformer circuits -- the minimal computational subgraphs that perform specific functions within the model. Through carefully constructed contrastive datasets, we identify points where model generation diverges from memorized content and isolate the specific circuits responsible for two distinct aspects of memorization. We find that circuits that initiate memorization can also maintain it once started, while circuits that only maintain memorization cannot trigger its initiation. Intriguingly, memorization prevention mechanisms transfer robustly across different text domains, while memorization induction appears more context-dependent.
>
---
#### [replaced 069] Winning the Pruning Gamble: A Unified Approach to Joint Sample and Token Pruning for Efficient Supervised Fine-Tuning
- **分类: cs.CL**

- **简介: 该论文属于语言模型微调任务，解决数据效率问题。针对现有方法仅单维度 pruning 的不足，提出 Q-Tuning 框架，联合优化样本与词元剪枝，提升数据利用率。**

- **链接: [https://arxiv.org/pdf/2509.23873v2](https://arxiv.org/pdf/2509.23873v2)**

> **作者:** Shaobo Wang; Jiaming Wang; Jiajun Zhang; Cong Wang; Yue Min; Zichen Wen; Xingzhang Ren; Fei Huang; Huiqiang Jiang; Junyang Lin; Dayiheng Liu; Linfeng Zhang
>
> **备注:** 26 pages, 9 figures, 15 tables
>
> **摘要:** As supervised fine-tuning (SFT) evolves from a lightweight post-training step into a compute-intensive phase rivaling mid-training in scale, data efficiency has become critical for aligning large language models (LLMs) under tight budgets. Existing data pruning methods suffer from a fragmented design: they operate either at the sample level or the token level in isolation, failing to jointly optimize both dimensions. This disconnect leads to significant inefficiencies--high-value samples may still contain redundant tokens, while token-level pruning often discards crucial instructional or corrective signals embedded in individual examples. To address this bottleneck, we introduce the Error-Uncertainty (EU) Plane, a diagnostic framework that jointly characterizes the heterogeneous utility of training data across samples and tokens. Guided by this insight, we propose Quadrant-based Tuning (Q-Tuning), a unified framework that strategically coordinates sample pruning and token pruning. Q-Tuning employs a two-stage strategy: first, it performs sample-level triage to retain examples rich in informative misconceptions or calibration signals; second, it applies an asymmetric token-pruning policy, using a context-aware scoring mechanism to trim less salient tokens exclusively from misconception samples while preserving calibration samples in their entirety. Our method sets a new state of the art across five diverse benchmarks. Remarkably, on SmolLM2-1.7B, Q-Tuning achieves a +38\% average improvement over the full-data SFT baseline using only 12.5\% of the original training data. As the first dynamic pruning approach to consistently outperform full-data training, Q-Tuning provides a practical and scalable blueprint for maximizing data utilization in budget-constrained LLM SFT.
>
---
#### [replaced 070] v1: Learning to Point Visual Tokens for Multimodal Grounded Reasoning
- **分类: cs.CL; cs.CV**

- **简介: 该论文属于多模态推理任务，旨在解决模型在长推理链中失去视觉焦点的问题。通过引入v1模型，实现视觉标记的主动引用，提升推理准确性。**

- **链接: [https://arxiv.org/pdf/2505.18842v5](https://arxiv.org/pdf/2505.18842v5)**

> **作者:** Jiwan Chung; Junhyeok Kim; Siyeol Kim; Jaeyoung Lee; Min Soo Kim; Youngjae Yu
>
> **摘要:** When thinking with images, humans rarely rely on a single glance: they revisit visual evidence while reasoning. In contrast, most Multimodal Language Models encode an image once to key-value cache and then reason purely in text, making it hard to re-ground intermediate steps. We empirically confirm this: as reasoning chains lengthen, models progressively lose focus on relevant regions. We introduce v1, a lightweight extension for active visual referencing via point-and-copy: the model selects relevant image patches and copies their embeddings back into the reasoning stream. Crucially, our point-and-copy mechanism retrieves patches using their semantic representations as keys, ensuring perceptual evidence remains aligned with the reasoning space. To train this behavior, we build v1, a dataset of 300K multimodal reasoning traces with interleaved grounding annotations. Across multimodal mathematical reasoning benchmarks, v1 consistently outperforms comparable baselines. We plan to release the model checkpoint and data.
>
---
#### [replaced 071] SERA: Soft-Verified Efficient Repository Agents
- **分类: cs.CL; cs.LG; cs.SE**

- **简介: 该论文提出SERA，一种高效训练编码代理的方法，解决私有代码库专用模型的低成本训练问题。通过监督微调实现高性能，比传统方法更经济。**

- **链接: [https://arxiv.org/pdf/2601.20789v2](https://arxiv.org/pdf/2601.20789v2)**

> **作者:** Ethan Shen; Danny Tormoen; Saurabh Shah; Ali Farhadi; Tim Dettmers
>
> **备注:** 21 main pages, 6 pages appendix
>
> **摘要:** Open-weight coding agents should hold a fundamental advantage over closed-source systems: they can be specialized to private codebases, encoding repository-specific information directly in their weights. Yet the cost and complexity of training has kept this advantage theoretical. We show it is now practical. We present Soft-Verified Efficient Repository Agents (SERA), an efficient method for training coding agents that enables the rapid and cheap creation of agents specialized to private codebases. Using only supervised finetuning (SFT), SERA achieves state-of-the-art results among fully open-source (open data, method, code) models while matching the performance of frontier open-weight models like Devstral-Small-2. Creating SERA models is 26x cheaper than reinforcement learning and 57x cheaper than previous synthetic data methods to reach equivalent performance. Our method, Soft Verified Generation (SVG), generates thousands of trajectories from a single code repository. Combined with cost-efficiency, this enables specialization to private codebases. Beyond repository specialization, we apply SVG to a larger corpus of codebases, generating over 200,000 synthetic trajectories. We use this dataset to provide detailed analysis of scaling laws, ablations, and confounding factors for training coding agents. Overall, we believe our work will greatly accelerate research on open coding agents and showcase the advantage of open-source models that can specialize to private codebases. We release SERA as the first model in Ai2's Open Coding Agents series, along with all our code, data, and Claude Code integration to support the research community.
>
---
#### [replaced 072] Hallucination is a Consequence of Space-Optimality: A Rate-Distortion Theorem for Membership Testing
- **分类: cs.LG; cs.AI; cs.CL; cs.DS; cs.IT**

- **简介: 该论文探讨语言模型中的幻觉现象，将其建模为成员检测问题，提出率失真定理解释幻觉的理论根源。属于自然语言处理任务，解决模型在有限存储下产生错误自信的问题。**

- **链接: [https://arxiv.org/pdf/2602.00906v2](https://arxiv.org/pdf/2602.00906v2)**

> **作者:** Anxin Guo; Jingwei Li
>
> **摘要:** Large language models often hallucinate with high confidence on "random facts" that lack inferable patterns. We formalize the memorization of such facts as a membership testing problem, unifying the discrete error metrics of Bloom filters with the continuous log-loss of LLMs. By analyzing this problem in the regime where facts are sparse in the universe of plausible claims, we establish a rate-distortion theorem: the optimal memory efficiency is characterized by the minimum KL divergence between score distributions on facts and non-facts. This theoretical framework provides a distinctive explanation for hallucination: even with optimal training, perfect data, and a simplified "closed world" setting, the information-theoretically optimal strategy under limited capacity is not to abstain or forget, but to assign high confidence to some non-facts, resulting in hallucination. We validate this theory empirically on synthetic data, showing that hallucinations persist as a natural consequence of lossy compression.
>
---
#### [replaced 073] Evalet: Evaluating Large Language Models by Fragmenting Outputs into Functions
- **分类: cs.HC; cs.AI; cs.CL**

- **简介: 该论文提出Evalet系统，通过功能碎片化分析大语言模型输出，解决传统评分方式无法定位影响因素的问题，实现细粒度评估。**

- **链接: [https://arxiv.org/pdf/2509.11206v3](https://arxiv.org/pdf/2509.11206v3)**

> **作者:** Tae Soo Kim; Heechan Lee; Yoonjoo Lee; Joseph Seering; Juho Kim
>
> **备注:** The first two authors hold equal contribution. Conditionally accepted to CHI 2026
>
> **摘要:** Practitioners increasingly rely on Large Language Models (LLMs) to evaluate generative AI outputs through "LLM-as-a-Judge" approaches. However, these methods produce holistic scores that obscure which specific elements influenced the assessments. We propose functional fragmentation, a method that dissects each output into key fragments and interprets the rhetoric functions that each fragment serves relative to evaluation criteria -- surfacing the elements of interest and revealing how they fulfill or hinder user goals. We instantiate this approach in Evalet, an interactive system that visualizes fragment-level functions across many outputs to support inspection, rating, and comparison of evaluations. A user study (N=10) found that, while practitioners struggled to validate holistic scores, our approach helped them identify 48% more evaluation misalignments. This helped them calibrate trust in LLM evaluations and rely on them to find more actionable issues in model outputs. Our work shifts LLM evaluation from quantitative scores toward qualitative, fine-grained analysis of model behavior.
>
---
#### [replaced 074] Evaluating Scoring Bias in LLM-as-a-Judge
- **分类: cs.CL**

- **简介: 该论文属于模型评估任务，旨在解决LLM作为评分者时的打分偏差问题。通过定义新类型偏差并提出量化框架，分析并揭示先进模型中的显著偏差。**

- **链接: [https://arxiv.org/pdf/2506.22316v4](https://arxiv.org/pdf/2506.22316v4)**

> **作者:** Qingquan Li; Shaoyu Dou; Kailai Shao; Chao Chen; Haixiang Hu
>
> **备注:** Accepted by DASFAA 2026
>
> **摘要:** The "LLM-as-a-Judge" paradigm, using Large Language Models (LLMs) as automated evaluators, is pivotal to LLM development, offering scalable feedback for complex tasks. However, the reliability of these judges is compromised by various biases. Existing research has heavily concentrated on biases in comparative evaluations. In contrast, scoring-based evaluations-which assign an absolute score and are often more practical in industrial applications-remain under-investigated. To address this gap, we undertake the first dedicated examination of scoring bias in LLM judges. We shift the focus from biases tied to the evaluation targets to those originating from the scoring prompt itself. We formally define scoring bias and identify three novel, previously unstudied types: rubric order bias, score ID bias, and reference answer score bias. We propose a comprehensive framework to quantify these biases, featuring a suite of multi-faceted metrics and an automatic data synthesis pipeline to create a tailored evaluation corpus. Our experiments empirically demonstrate that even the most advanced LLMs suffer from these substantial scoring biases. Our analysis yields actionable insights for designing more robust scoring prompts and mitigating these newly identified biases.
>
---
#### [replaced 075] Surprisal from Larger Transformer-based Language Models Predicts fMRI Data More Poorly
- **分类: cs.CL**

- **简介: 该论文研究语言模型 surprisal 预测 fMRI 数据的能力，探讨模型规模与预测效果的关系，解决模型性能与脑成像数据关联的问题。**

- **链接: [https://arxiv.org/pdf/2506.11338v2](https://arxiv.org/pdf/2506.11338v2)**

> **作者:** Yi-Chien Lin; William Schuler
>
> **备注:** EACL 2026
>
> **摘要:** There has been considerable interest in using surprisal from Transformer-based language models (LMs) as predictors of human sentence processing difficulty. Recent work has observed an inverse scaling relationship between Transformers' per-word estimated probability and the predictive power of their surprisal estimates on reading times, showing that LMs with more parameters and trained on more data are less predictive of human reading times. However, these studies focused on predicting latency-based measures. Tests on brain imaging data have not shown a trend in any direction when using a relatively small set of LMs, leaving open the possibility that the inverse scaling phenomenon is constrained to latency data. This study therefore conducted a more comprehensive evaluation using surprisal estimates from 17 pre-trained LMs across three different LM families on two functional magnetic resonance imaging (fMRI) datasets. Results show that the inverse scaling relationship between models' per-word estimated probability and model fit on both datasets still obtains, resolving the inconclusive results of previous work and indicating that this trend is not specific to latency-based measures.
>
---
#### [replaced 076] AWM: Accurate Weight-Matrix Fingerprint for Large Language Models
- **分类: cs.CL**

- **简介: 该论文属于模型溯源任务，旨在解决LLM是否源自已有模型的验证问题。提出一种无需训练的权重矩阵指纹方法，利用LAP和CKA实现高精度识别。**

- **链接: [https://arxiv.org/pdf/2510.06738v2](https://arxiv.org/pdf/2510.06738v2)**

> **作者:** Boyi Zeng; Lin Chen; Ziwei He; Xinbing Wang; Zhouhan Lin
>
> **备注:** ICLR 2026
>
> **摘要:** Protecting the intellectual property of large language models (LLMs) is crucial, given the substantial resources required for their training. Consequently, there is an urgent need for both model owners and third parties to determine whether a suspect LLM is trained from scratch or derived from an existing base model. However, the intensive post-training processes that models typically undergo-such as supervised fine-tuning, extensive continued pretraining, reinforcement learning, multi-modal extension, pruning, and upcycling-pose significant challenges to reliable identification. In this work, we propose a training-free fingerprinting method based on weight matrices. We leverage the Linear Assignment Problem (LAP) and an unbiased Centered Kernel Alignment (CKA) similarity to neutralize the effects of parameter manipulations, yielding a highly robust and high-fidelity similarity metric. On a comprehensive testbed of 60 positive and 90 negative model pairs, our method demonstrates exceptional robustness against all six aforementioned post-training categories while exhibiting a near-zero risk of false positives. By achieving perfect scores on all classification metrics, our approach establishes a strong basis for reliable model lineage verification. Moreover, the entire computation completes within 30s on an NVIDIA 3090 GPU. The code is available at https://github.com/LUMIA-Group/AWM.
>
---
#### [replaced 077] MedFrameQA: A Multi-Image Medical VQA Benchmark for Clinical Reasoning
- **分类: cs.CV; cs.CL**

- **简介: 该论文提出MedFrameQA，一个用于医学视觉问答的多图像基准，解决多图像临床推理问题。通过教育视频构建数据集，评估大模型在多图像理解上的不足。**

- **链接: [https://arxiv.org/pdf/2505.16964v2](https://arxiv.org/pdf/2505.16964v2)**

> **作者:** Suhao Yu; Haojin Wang; Juncheng Wu; Luyang Luo; Jingshen Wang; Cihang Xie; Pranav Rajpurkar; Carl Yang; Yang Yang; Kang Wang; Yannan Yu; Yuyin Zhou
>
> **备注:** 27 pages, 15 Figures Benchmark data: https://huggingface.co/datasets/SuhaoYu1020/MedFrameQA
>
> **摘要:** Real-world clinical practice demands multi-image comparative reasoning, yet current medical benchmarks remain limited to single-frame interpretation. We present MedFrameQA, the first benchmark explicitly designed to test multi-image medical VQA through educationally-validated diagnostic sequences. To construct this dataset, we develop a scalable pipeline that leverages narrative transcripts from medical education videos to align visual frames with textual concepts, automatically producing 2,851 high-quality multi-image VQA pairs with explicit, transcript-grounded reasoning chains. Our evaluation of 11 advanced MLLMs (including reasoning models) exposes severe deficiencies in multi-image synthesis, where accuracies mostly fall below 50% and exhibit instability across varying image counts. Error analysis demonstrates that models often treat images as isolated instances, failing to track pathological progression or cross-reference anatomical shifts. MedFrameQA provides a rigorous standard for evaluating the next generation of MLLMs in handling complex, temporally grounded medical narratives.
>
---
#### [replaced 078] Co-Designing Quantum Codes with Transversal Diagonal Gates via Multi-Agent Systems
- **分类: quant-ph; cs.AI; cs.CL; math-ph**

- **简介: 该论文属于量子纠错码设计任务，解决如何协同设计具有特定对角横操作的量子码问题。通过多智能体系统与AI协作，发现新型非加性量子码。**

- **链接: [https://arxiv.org/pdf/2510.20728v2](https://arxiv.org/pdf/2510.20728v2)**

> **作者:** Xi He; Sirui Lu; Bei Zeng
>
> **备注:** 63 pages, 3 figures
>
> **摘要:** We present a multi-agent, human-in-the-loop workflow that co-designs quantum error-correcting codes with prescribed transversal diagonal gates. It builds on the Subset-Sum Linear Programming (SSLP) framework, which partitions basis strings by modular residues and enforces Z-marginal Knill-Laflamme (KL) equalities via small LPs. The workflow is powered by GPT-5 and implemented within TeXRA, a multi-agent research assistant platform where agents collaborate in a shared LaTeX-Python workspace synchronized with Git/Overleaf. Three specialized agents formulate constraints, sweep and screen candidate codes, exactify numerical solutions into rationals, and independently audit all KL equalities and induced logical actions. Focusing on distance-two codes with nondegenerate residues, we catalogue new nonadditive codes for dimensions $K\in\{2,3,4\}$ on up to six qubits, including high-order diagonal transversals, yielding $14,116$ new codes. From these data, the system abstracts closed-form families and constructs a residue-degenerate $((6,4,2))$ code implementing a transversal controlled-phase $\mathrm{diag}(1,1,1,i)$, illustrating how AI orchestration can drive rigorous, scalable code discovery.
>
---
#### [replaced 079] Adaptive Rollout Allocation for Online Reinforcement Learning with Verifiable Rewards
- **分类: cs.LG; cs.AI; cs.CL**

- **简介: 该论文属于在线强化学习任务，解决采样效率低的问题。提出VIP策略，根据提示信息量动态分配采样次数，提升训练效果。**

- **链接: [https://arxiv.org/pdf/2602.01601v2](https://arxiv.org/pdf/2602.01601v2)**

> **作者:** Hieu Trung Nguyen; Bao Nguyen; Wenao Ma; Yuzhi Zhao; Ruifeng She; Viet Anh Nguyen
>
> **备注:** Accepted at ICLR 2026
>
> **摘要:** Sampling efficiency is a key bottleneck in reinforcement learning with verifiable rewards. Existing group-based policy optimization methods, such as GRPO, allocate a fixed number of rollouts for all training prompts. This uniform allocation implicitly treats all prompts as equally informative, and could lead to inefficient computational budget usage and impede training progress. We introduce VIP, a Variance-Informed Predictive allocation strategy that allocates a given rollout budget to the prompts in the incumbent batch to minimize the expected gradient variance of the policy update. At each iteration, VIP uses a lightweight Gaussian process model to predict per-prompt success probabilities based on recent rollouts. These probability predictions are translated into variance estimates, which are then fed into a convex optimization problem to determine the optimal rollout allocations under a hard compute budget constraint. Empirical results show that VIP consistently improves sampling efficiency and achieves higher performance than uniform or heuristic allocation strategies in multiple benchmarks.
>
---
#### [replaced 080] Natural Language Actor-Critic: Scalable Off-Policy Learning in Language Space
- **分类: cs.LG; cs.CL**

- **简介: 该论文属于强化学习任务，旨在解决LLM代理在长序列、稀疏奖励环境中的训练不稳定和样本效率低的问题。提出NLAC算法，使用自然语言评断器提升训练效果。**

- **链接: [https://arxiv.org/pdf/2512.04601v2](https://arxiv.org/pdf/2512.04601v2)**

> **作者:** Joey Hong; Kang Liu; Zhan Ling; Jiecao Chen; Sergey Levine
>
> **备注:** 21 pages, 4 figures
>
> **摘要:** Large language model (LLM) agents -- LLMs that dynamically interact with an environment over long horizons -- have become an increasingly important area of research, enabling automation in complex tasks involving tool-use, web browsing, and dialogue with people. In the absence of expert demonstrations, training LLM agents has relied on policy gradient methods that optimize LLM policies with respect to an (often sparse) reward function. However, in long-horizon tasks with sparse rewards, learning from trajectory-level rewards can be noisy, leading to training that is unstable and has high sample complexity. Furthermore, policy improvement hinges on discovering better actions through exploration, which can be difficult when actions lie in natural language space. In this paper, we propose Natural Language Actor-Critic (NLAC), a novel actor-critic algorithm that trains LLM policies using a generative LLM critic that produces natural language rather than scalar values. This approach leverages the inherent strengths of LLMs to provide a richer and more actionable training signal; particularly, in tasks with large, open-ended action spaces, natural language explanations for why an action is suboptimal can be immensely useful for LLM policies to reason how to improve their actions, without relying on random exploration. Furthermore, our approach can be trained off-policy without policy gradients, offering a more data-efficient and stable alternative to existing on-policy methods. We present results on a mixture of reasoning, web browsing, and tool-use with dialogue tasks, demonstrating that NLAC shows promise in outperforming existing training approaches and offers a more scalable and stable training paradigm for LLM agents.
>
---
#### [replaced 081] OpenRubrics: Towards Scalable Synthetic Rubric Generation for Reward Modeling and LLM Alignment
- **分类: cs.CL**

- **简介: 该论文属于奖励建模任务，旨在解决传统奖励模型无法全面反映人类偏好的问题。通过构建大规模(提示, 评分标准)对和对比评分生成方法，提升奖励模型性能。**

- **链接: [https://arxiv.org/pdf/2510.07743v3](https://arxiv.org/pdf/2510.07743v3)**

> **作者:** Tianci Liu; Ran Xu; Tony Yu; Ilgee Hong; Carl Yang; Tuo Zhao; Haoyu Wang
>
> **备注:** The first two authors contributed equally. Updated OpenRubrics dataset, RMs, and results
>
> **摘要:** Reward modeling lies at the core of reinforcement learning from human feedback (RLHF), yet most existing reward models rely on scalar or pairwise judgments that fail to capture the multifaceted nature of human preferences. Recent studies have explored rubrics-as-rewards (RaR) that uses structured criteria to capture multiple dimensions of response quality. However, producing rubrics that are both reliable and scalable remains a key challenge. In this work, we introduce OpenRubrics, a diverse, large-scale collection of (prompt, rubric) pairs for training rubric-generation and rubric-based reward models. To elicit discriminative and comprehensive evaluation signals, we introduce Contrastive Rubric Generation (CRG), which derives both hard rules (explicit constraints) and principles (implicit qualities) by contrasting preferred and rejected responses. We further remove noisy rubrics via preserving preference-label consistency. Across multiple reward-modeling benchmarks, our rubric-based reward model, Rubric-RM, surpasses strong size-matched baselines by 8.4%. These gains transfer to policy models on instruction-following and biomedical benchmarks.
>
---
#### [replaced 082] Do AI Models Perform Human-like Abstract Reasoning Across Modalities?
- **分类: cs.AI; cs.CL**

- **简介: 论文探讨AI模型是否具备类人抽象推理能力，针对多模态任务进行评估。研究通过ConceptARC基准测试，分析模型在文本和视觉模态中的表现及生成的规则，揭示其对抽象概念的理解不足。**

- **链接: [https://arxiv.org/pdf/2510.02125v4](https://arxiv.org/pdf/2510.02125v4)**

> **作者:** Claas Beger; Ryan Yi; Shuhao Fu; Kaleda Denton; Arseny Moskvichev; Sarah W. Tsai; Sivasankaran Rajamanickam; Melanie Mitchell
>
> **备注:** 9 pages, 3 figures
>
> **摘要:** OpenAI's o3-preview reasoning model exceeded human accuracy on the ARC-AGI-1 benchmark, but does that mean state-of-the-art models recognize and reason with the abstractions the benchmark was designed to test? Here we investigate abstraction abilities of AI models using the closely related but simpler ConceptARC benchmark. Our evaluations vary input modality (textual vs. visual), use of external Python tools, and reasoning effort. Beyond output accuracy, we evaluate the natural-language rules that models generate to explain their solutions, enabling us to assess whether models recognize the abstractions that ConceptARC was designed to elicit. We show that the best models' rules are frequently based on surface-level ``shortcuts,'' capturing intended abstractions considerably less often than humans. In the visual modality, AI models' output accuracy drops sharply; however, our rule-level analysis reveals that a substantial share of their rules capture the intended abstractions, even as the models struggle to apply these concepts to generate correct solutions. In short, we show that using accuracy alone to evaluate abstract reasoning can substantially overestimate AI capabilities in textual modalities and underestimate it in visual modalities. Our results offer a more faithful picture of AI models' abstract reasoning abilities and a more principled way to track progress toward human-like, abstraction-centered intelligence.
>
---
#### [replaced 083] From Pragmas to Partners: A Symbiotic Evolution of Agentic High-Level Synthesis
- **分类: cs.CL; cs.AI**

- **简介: 该论文属于AI与硬件设计交叉任务，探讨HLS在智能时代的重要性。解决HLS是否仍具价值的问题，提出HLS作为优化层的潜力及改进方向。**

- **链接: [https://arxiv.org/pdf/2602.01401v2](https://arxiv.org/pdf/2602.01401v2)**

> **作者:** Niansong Zhang; Sunwoo Kim; Shreesha Srinath; Zhiru Zhang
>
> **摘要:** The rise of large language models has sparked interest in AI-driven hardware design, raising the question: does high-level synthesis (HLS) still matter in the agentic era? We argue that HLS remains essential. While we expect mature agentic hardware systems to leverage both HLS and RTL, this paper focuses on HLS and its role in enabling agentic optimization. HLS offers faster iteration cycles, portability, and design permutability that make it a natural layer for agentic optimization. This position paper makes three contributions. First, we explain why HLS serves as a practical abstraction layer and a golden reference for agentic hardware design. Second, we identify key limitations of current HLS tools, namely inadequate performance feedback, rigid interfaces, and limited debuggability that agents are uniquely positioned to address. Third, we propose a taxonomy for the symbiotic evolution of agentic HLS, clarifying how responsibility shifts from human designers to AI agents as systems advance from copilots to autonomous design partners.
>
---
