# 机器人 cs.RO

- **最新发布 46 篇**

- **更新 27 篇**

## 最新发布

#### [new 001] Learning Semantic Priorities for Autonomous Target Search
- **分类: cs.RO**

- **简介: 该论文属于目标搜索任务，旨在解决机器人在未知环境中高效搜索的问题。通过引入专家知识训练语义优先级模型，提升搜索效率与覆盖性。**

- **链接: [https://arxiv.org/pdf/2603.29391](https://arxiv.org/pdf/2603.29391)**

> **作者:** Max Lodel; Nils Wilde; Robert Babuška; Javier Alonso-Mora
>
> **备注:** accepted to ICRA2026
>
> **摘要:** The use of semantic features can improve the efficiency of target search in unknown environments for robotic search and rescue missions. Current target search methods rely on training with large datasets of similar domains, which limits the adaptability to diverse environments. However, human experts possess high-level knowledge about semantic relationships necessary to effectively guide a robot during target search missions in diverse and previously unseen environments. In this paper, we propose a target search method that leverages expert input to train a model of semantic priorities. By employing the learned priorities in a frontier exploration planner using combinatorial optimization, our approach achieves efficient target search driven by semantic features while ensuring robustness and complete coverage. The proposed semantic priority model is trained with several synthetic datasets of simulated expert guidance for target search. Simulation tests in previously unseen environments show that our method consistently achieves faster target recovery than a coverage-driven exploration planner.
>
---
#### [new 002] CLaD: Planning with Grounded Foresight via Cross-Modal Latent Dynamics
- **分类: cs.RO**

- **简介: 该论文提出CLaD框架，解决机器人操作中语义与运动状态耦合问题，通过跨模态注意力建模两者联合演化，提升动作规划效果。**

- **链接: [https://arxiv.org/pdf/2603.29409](https://arxiv.org/pdf/2603.29409)**

> **作者:** Andrew Jeong; Jaemin Kim; Sebin Lee; Sung-Eui Yoon
>
> **备注:** Project page: this https URL
>
> **摘要:** Robotic manipulation involves kinematic and semantic transitions that are inherently coupled via underlying actions. However, existing approaches plan within either semantic or latent space without explicitly aligning these cross-modal transitions. To address this, we propose CLaD, a framework that models how proprioceptive and semantic states jointly evolve under actions through asymmetric cross-attention that allows kinematic transitions to query semantic ones. CLaD predicts grounded latent foresights via self-supervised objectives with EMA target encoders and auxiliary reconstruction losses, preventing representation collapse while anchoring predictions to observable states. Predicted foresights are modulated with observations to condition a diffusion policy for action generation. On LIBERO-LONG benchmark, CLaD achieves 94.7\% success rate, competitive with large VLAs with significantly fewer parameters.
>
---
#### [new 003] DIAL: Decoupling Intent and Action via Latent World Modeling for End-to-End VLA
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出DIAL框架，解决VLA模型中高阶决策与低阶执行脱节的问题，通过解耦意图与动作提升性能。**

- **链接: [https://arxiv.org/pdf/2603.29844](https://arxiv.org/pdf/2603.29844)**

> **作者:** Yi Chen; Yuying Ge; Hui Zhou; Mingyu Ding; Yixiao Ge; Xihui Liu
>
> **备注:** Project page: this https URL
>
> **摘要:** The development of Vision-Language-Action (VLA) models has been significantly accelerated by pre-trained Vision-Language Models (VLMs). However, most existing end-to-end VLAs treat the VLM primarily as a multimodal encoder, directly mapping vision-language features to low-level actions. This paradigm underutilizes the VLM's potential in high-level decision making and introduces training instability, frequently degrading its rich semantic representations. To address these limitations, we introduce DIAL, a framework bridging high-level decision making and low-level motor execution through a differentiable latent intent bottleneck. Specifically, a VLM-based System-2 performs latent world modeling by synthesizing latent visual foresight within the VLM's native feature space; this foresight explicitly encodes intent and serves as the structural bottleneck. A lightweight System-1 policy then decodes this predicted intent together with the current observation into precise robot actions via latent inverse dynamics. To ensure optimization stability, we employ a two-stage training paradigm: a decoupled warmup phase where System-2 learns to predict latent futures while System-1 learns motor control under ground-truth future guidance within a unified feature space, followed by seamless end-to-end joint optimization. This enables action-aware gradients to refine the VLM backbone in a controlled manner, preserving pre-trained knowledge. Extensive experiments on the RoboCasa GR1 Tabletop benchmark show that DIAL establishes a new state-of-the-art, achieving superior performance with 10x fewer demonstrations than prior methods. Furthermore, by leveraging heterogeneous human demonstrations, DIAL learns physically grounded manipulation priors and exhibits robust zero-shot generalization to unseen objects and novel configurations during real-world deployment on a humanoid robot.
>
---
#### [new 004] Semantic Zone-Based Map Management for Stable AI-Integrated Mobile Robots
- **分类: cs.RO**

- **简介: 该论文属于移动机器人任务，解决内存受限下密集地图与AI模型协同使用的问题。通过语义区域管理优化关键帧加载，提升系统稳定性与效率。**

- **链接: [https://arxiv.org/pdf/2603.29627](https://arxiv.org/pdf/2603.29627)**

> **作者:** Huichang Yun; Seungho Yoo
>
> **摘要:** Recent advances in large AI models (VLMs and LLMs) and joint use of the 3D dense maps, enable mobile robots to provide more powerful and interactive services grounded in rich spatial context. However, deploying both heavy AI models and dense maps on edge robots is challenging under strict memory budgets. When the memory budget is exceeded, required keyframes may not be loaded in time, which can degrade the stability of position estimation and interfering model performance. We proposes a semantic zone-based map management approach to stabilize dense-map utilization under memory constraints. We associate keyframes with semantic indoor regions (e.g., rooms and corridors) and keyframe management at the semantic zone level prioritizes spatially relevant map content while respecting memory constraints. This reduces keyframe loading and unloading frequency and memory usage. We evaluate the proposed approach in large-scale simulated indoor environments and on an NVIDIA Jetson Orin Nano under concurrent SLAM-VLM execution. With Qwen3.5:0.8b, the proposed method improves throughput by 3.3 tokens/s and reduces latency by 21.7% relative to a geometric map-management strategy. Furthermore, while the geometric strategy suffers from out-of-memory failures and stalled execution under memory pressure, the proposed method eliminates both issues, preserving localization stability and enabling robust VLM operation. These results demonstrate that the proposed approach enables efficient dense map utilization for memory constrained, AI-integrated mobile robots. Code is available at: this https URL
>
---
#### [new 005] GraSP-STL: A Graph-Based Framework for Zero-Shot Signal Temporal Logic Planning via Offline Goal-Conditioned Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于零样本信号时序逻辑规划任务，解决在无环境交互情况下，基于离线数据生成满足未见STL规范的控制策略问题。工作包括提出GraSP-STL框架，结合图搜索与目标条件强化学习实现有效规划。**

- **链接: [https://arxiv.org/pdf/2603.29533](https://arxiv.org/pdf/2603.29533)**

> **作者:** Ancheng Hou; Ruijia Liu; Xiang Yin
>
> **摘要:** This paper studies offline, zero-shot planning under Signal Temporal Logic (STL) specifications. We assume access only to an offline dataset of state-action-state transitions collected by a task-agnostic behavior policy, with no analytical dynamics model, no further environment interaction, and no task-specific retraining. The objective is to synthesize a control strategy whose resulting trajectory satisfies an arbitrary unseen STL specification. To this end, we propose GraSP-STL, a graph-search-based framework for zero-shot STL planning from offline trajectories. The method learns a goal-conditioned value function from offline data and uses it to induce a finite-horizon reachability metric over the state space. Based on this metric, it constructs a directed graph abstraction whose nodes represent representative states and whose edges encode feasible short-horizon transitions. Planning is then formulated as a graph search over waypoint sequences, evaluated using arithmetic-geometric mean robustness and its interval semantics, and executed by a learned goal-conditioned policy. The proposed framework separates reusable reachability learning from task-conditioned planning, enabling zero-shot generalization to unseen STL tasks and long-horizon planning through the composition of short-horizon behaviors from offline data. Experimental results demonstrate its effectiveness on a range of offline STL planning tasks.
>
---
#### [new 006] World2Rules: A Neuro-Symbolic Framework for Learning World-Governing Safety Rules for Aviation
- **分类: cs.RO**

- **简介: 该论文提出World2Rules，解决安全规则学习任务，旨在从多模态航空数据中自动提取可解释的安全规则，提升安全关键系统的可靠性。**

- **链接: [https://arxiv.org/pdf/2603.28952](https://arxiv.org/pdf/2603.28952)**

> **作者:** Haichuan Wang; Jay Patrikar; Sebastian Scherer
>
> **备注:** 19 pages, 6 figures
>
> **摘要:** Many real-world safety-critical systems are governed by explicit rules that define unsafe world configurations and constrain agent interactions. In practice, these rules are complex and context-dependent, making manual specification incomplete and error-prone. Learning such rules from real-world multimodal data is further challenged by noise, inconsistency, and sparse failure cases. Neural models can extract structure from text and visual data but lack formal guarantees, while symbolic methods provide verifiability yet are brittle when applied directly to imperfect observations. We present World2Rules, a neuro-symbolic framework for learning world-governing safety rules from real-world multimodal aviation data. World2Rules learns from both nominal operational data and aviation crash and incident reports, treating neural models as proposal mechanisms for candidate symbolic facts and inductive logic programming as a verification layer. The framework employs hierarchical reflective reasoning, enforcing consistency across examples, subsets, and rules to filter unreliable evidence, aggregate only mutually consistent components, and prune unsupported hypotheses. This design limits error propagation from noisy neural extractions and yields compact, interpretable first-order logic rules that characterize unsafe world configurations. We evaluate World2Rules on real-world aviation safety data and show that it learns rules that achieve 23.6% higher F1 score than purely neural and 43.2% higher F1 score than single-pass neuro-symbolic baseline, while remaining suitable for safety-critical reasoning and formal analysis.
>
---
#### [new 007] Bootstrap Perception Under Hardware Depth Failure for Indoor Robot Navigation
- **分类: cs.RO**

- **简介: 该论文属于室内机器人导航任务，解决硬件深度传感器失效问题。通过自校准方法融合LiDAR与学习到的单目深度，提升障碍物检测覆盖率，实现高效可靠导航。**

- **链接: [https://arxiv.org/pdf/2603.28890](https://arxiv.org/pdf/2603.28890)**

> **作者:** Nishant Pushparaju; Vivek Mattam; Aliasghar Arab
>
> **摘要:** We present a bootstrap perception system for indoor robot navigation under hardware depth failure. In our corridor data, the time-of-flight camera loses up to 78% of its depth pixels on reflective surfaces, yet a 2D LiDAR alone cannot sense obstacles above its scan plane. Our system exploits a self-referential property of this failure: the sensor's surviving valid pixels calibrate learned monocular depth to metric scale, so the system fills its own gaps without external data. The architecture forms a failure-aware sensing hierarchy, conservative when sensors work and filling in when they fail: LiDAR remains the geometric anchor, hardware depth is kept where valid, and learned depth enters only where needed. In corridor and dynamic pedestrian evaluations, selective fusion increases costmap obstacle coverage by 55-110% over LiDAR alone. A compact distilled student runs at 218\,FPS on a Jetson Orin Nano and achieves 9/10 navigation success with zero collisions in closed-loop simulation, matching the ground-truth depth baseline at a fraction of the foundation model's cost.
>
---
#### [new 008] Long-Reach Robotic Manipulation for Assembly and Outfitting of Lunar Structures
- **分类: cs.RO**

- **简介: 该论文属于月球结构装配任务，旨在解决长距离电缆布线的机器人操作问题。研究提出一种可展开复合臂机械臂，通过控制策略减少变形和振动，实现精准操作。**

- **链接: [https://arxiv.org/pdf/2603.29226](https://arxiv.org/pdf/2603.29226)**

> **作者:** Stanley Wang; Venny Kojouharov; Long Yin Chung; Daniel Morton; Mark Cutkosky
>
> **备注:** 7 pages, 6 figures, to appear in the proceedings of iSpaRo 2025
>
> **摘要:** Future infrastructure construction on the lunar surface will require semi- or fully-autonomous operation from robots deployed at the build site. In particular, tasks such as electrical outfitting necessitate transport, routing, and fine manipulation of cables across large structures. To address this need, we present a compact and long-reach manipulator incorporating a deployable composite boom, capable of performing manipulation tasks across large structures and workspaces. We characterize the deflection, vibration, and blossoming characteristics inherent to the deployable structure, and present a manipulation control strategy to mitigate these effects. Experiments indicate an average endpoint accuracy error of less than 15 mm for boom lengths up to 1.8 m. We demonstrate the approach with a cable routing task to illustrate the potential for lunar outfitting applications that benefit from long reach.
>
---
#### [new 009] Large Neighborhood Search for Multi-Agent Task Assignment and Path Finding with Precedence Constraints
- **分类: cs.RO; cs.MA**

- **简介: 该论文研究多智能体任务分配与路径规划问题（TAPF-PC），解决在优先级约束下如何优化任务分配和路径以提高效率。提出一种大邻域搜索方法，通过动态重分配提升解的质量。**

- **链接: [https://arxiv.org/pdf/2603.28968](https://arxiv.org/pdf/2603.28968)**

> **作者:** Viraj Parimi; Brian C. Williams
>
> **摘要:** Many multi-robot applications require tasks to be completed efficiently and in the correct order, so that downstream operations can proceed at the right time. Multi-agent path finding with precedence constraints (MAPF-PC) is a well-studied framework for computing collision-free plans that satisfy ordering relations when task sequences are fixed in advance. In many applications, however, solution quality depends not only on how agents move, but also on which agent performs which task. This motivates the lifted problem of task assignment and path finding with precedence constraints (TAPF-PC), which extends MAPF-PC by jointly optimizing assignment, precedence satisfaction, and routing cost. To address the resulting coupled TAPF-PC search space, we develop a large neighborhood search approach that starts from a feasible MAPF-PC seed and iteratively improves it through reassignment-based neighborhood repair, restoring feasibility within each selected neighborhood. Experiments across multiple benchmark families and scaling regimes show that the best-performing configuration improves 89.1% of instances over fixed-assignment seed solutions, demonstrating that large neighborhood search effectively captures the gains from flexible reassignment under precedence constraints.
>
---
#### [new 010] A Classification of Heterogeneity in Uncrewed Vehicle Swarms and the Effects of Its Inclusion on Overall Swarm Resilience
- **分类: cs.RO**

- **简介: 该论文属于分类任务，旨在解决异质无人飞行器编队设计问题。通过分析异质性对编队韧性的影响，提出分类框架并探讨实现方法。**

- **链接: [https://arxiv.org/pdf/2603.28831](https://arxiv.org/pdf/2603.28831)**

> **作者:** Abhishek Joshi; Abhishek Phadke; Tianxing Chu; F. Antonio Medrano
>
> **摘要:** Combining different types of agents in uncrewed vehicle (UV) swarms has emerged as an approach to enhance mission resilience and operational capabilities across a wide range of applications. This study offers a systematic framework for grouping different types of swarms based on three main factors: agent nature (behavior and function), hardware structure (physical configuration and sensing capabilities), and operational space (domain of operation). A literature review indicates that strategic heterogeneity significantly improves swarm performance. Operational challenges, including communication architecture constraints, energy-aware coordination strategies, and control system integration, are also discussed. The analysis shows that heterogeneous swarms are more resilient because they can leverage diverse capabilities, adapt roles on the fly, and integrate data from multidimensional sensor feeds. Some important factors to consider when implementing are sim-to-real-world transfer for learned policies, standardized evaluation metrics, and control architectures that can work together. Learning-based coordination, GPS (Global Positioning System)-denied multi-robot SLAM (Simultaneous Localization and Mapping), and domain-specific commercial deployments collectively demonstrate that heterogeneous swarm technology is moving closer to readiness for high-value applications. This study offers a single taxonomy and evidence-based observations on methods for designing mission-ready heterogeneous swarms that balance complexity and increased capability.
>
---
#### [new 011] Hybrid Framework for Robotic Manipulation: Integrating Reinforcement Learning and Large Language Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，旨在提升机器人执行复杂指令的能力。通过融合强化学习与大语言模型，实现高效、灵活的控制与任务规划。**

- **链接: [https://arxiv.org/pdf/2603.30022](https://arxiv.org/pdf/2603.30022)**

> **作者:** Md Saad; Sajjad Hussain; Mohd Suhaib
>
> **摘要:** This paper introduces a new hybrid framework that combines Reinforcement Learning (RL) and Large Language Models (LLMs) to improve robotic manipulation tasks. By utilizing RL for accurate low-level control and LLMs for high level task planning and understanding of natural language, the proposed framework effectively connects low-level execution with high-level reasoning in robotic systems. This integration allows robots to understand and carry out complex, human-like instructions while adapting to changing environments in real time. The framework is tested in a PyBullet-based simulation environment using the Franka Emika Panda robotic arm, with various manipulation scenarios as benchmarks. The results show a 33.5% decrease in task completion time and enhancements of 18.1% and 36.4% in accuracy and adaptability, respectively, when compared to systems that use only RL. These results underscore the potential of LLM-enhanced robotic systems for practical applications, making them more efficient, adaptable, and capable of interacting with humans. Future research will aim to explore sim-to-real transfer, scalability, and multi-robot systems to further broaden the framework's applicability.
>
---
#### [new 012] SafeDMPs: Integrating Formal Safety with DMPs for Adaptive HRI
- **分类: cs.RO; eess.SY; math.DS**

- **简介: 该论文属于人机协作机器人安全控制任务，解决DMPs缺乏形式化安全性和CBFs计算成本高的问题，提出SafeDMPs框架实现高效安全运动规划。**

- **链接: [https://arxiv.org/pdf/2603.29708](https://arxiv.org/pdf/2603.29708)**

> **作者:** Soumyodipta Nath; Pranav Tiwari; Ravi Prakash
>
> **备注:** 8 pages, 8 figures and 1 table
>
> **摘要:** Robots operating in human-centric environments must be both robust to disturbances and provably safe from collisions. Achieving these properties simultaneously and efficiently remains a central challenge. While Dynamic Movement Primitives (DMPs) offer inherent stability and generalization from single demonstrations, they lack formal safety guarantees. Conversely, formal methods like Control Barrier Functions (CBFs) provide provable safety but often rely on computationally expensive, real-time optimization, hindering their use in high-frequency control. This paper introduces SafeDMPs, a novel framework that resolves this trade-off. We integrate the closed-form efficiency and dynamic robustness of DMPs with a provably safe, non-optimization-based control law derived from Spatio-Temporal Tubes (STTs). This synergy allows us to generate motions that are not only robust to perturbations and adaptable to new goals, but also guaranteed to avoid static and dynamic obstacles. Our approach achieves a closed-form solution for a problem that traditionally requires online optimization. Experimental results on a 7-DOF robot manipulator demonstrate that SafeDMPs is orders of magnitude faster and more accurate than optimization-based baselines, making it an ideal solution for real-time, safe, and collaborative robotics.
>
---
#### [new 013] AutoWorld: Scaling Multi-Agent Traffic Simulation with Self-Supervised World Models
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于交通仿真任务，旨在利用未标注数据提升模拟性能。提出AutoWorld框架，通过自监督学习构建世界模型，优化多智能体运动生成。**

- **链接: [https://arxiv.org/pdf/2603.28963](https://arxiv.org/pdf/2603.28963)**

> **作者:** Mozhgan Pourkeshavatz; Tianran Liu; Nicholas Rhinehart
>
> **摘要:** Multi-agent traffic simulation is central to developing and testing autonomous driving systems. Recent data-driven simulators have achieved promising results, but rely heavily on supervised learning from labeled trajectories or semantic annotations, making it costly to scale their performance. Meanwhile, large amounts of unlabeled sensor data can be collected at scale but remain largely unused by existing traffic simulation frameworks. This raises a key question: How can a method harness unlabeled data to improve traffic simulation performance? In this work, we propose AutoWorld, a traffic simulation framework that employs a world model learned from unlabeled occupancy representations of LiDAR data. Given world model samples, AutoWorld constructs a coarse-to-fine predictive scene context as input to a multi-agent motion generation model. To promote sample diversity, AutoWorld uses a cascaded Determinantal Point Process framework to guide the sampling processes of both the world model and the motion model. Furthermore, we designed a motion-aware latent supervision objective that enhances AutoWorld's representation of scene dynamics. Experiments on the WOSAC benchmark show that AutoWorld ranks first on the leaderboard according to the primary Realism Meta Metric (RMM). We further show that simulation performance consistently improves with the inclusion of unlabeled LiDAR data, and study the efficacy of each component with ablations. Our method paves the way for scaling traffic simulation realism without additional labeling. Our project page contains additional visualizations and released code.
>
---
#### [new 014] CREST: Constraint-Release Execution for Multi-Robot Warehouse Shelf Rearrangement
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文提出CREST框架，解决多机器人仓库货架重新排列问题，通过动态释放轨迹约束提升执行效率，减少路径冗余和切换。**

- **链接: [https://arxiv.org/pdf/2603.28803](https://arxiv.org/pdf/2603.28803)**

> **作者:** Jiaqi Tan; Yudong Luo; Sophia Huang; Yifan Yang; Hang Ma
>
> **摘要:** Double-Deck Multi-Agent Pickup and Delivery (DD-MAPD) models the multi-robot shelf rearrangement problem in automated warehouses. MAPF-DECOMP is a recent framework that first computes collision-free shelf trajectories with a MAPF solver and then assigns agents to execute them. While efficient, it enforces strict trajectory dependencies, often leading to poor execution quality due to idle agents and unnecessary shelf switching. We introduce CREST, a new execution framework that achieves more continuous shelf carrying by proactively releasing trajectory constraints during execution. Experiments on diverse warehouse layouts show that CREST consistently outperforms MAPF-DECOMP, reducing metrics related to agent travel, makespan, and shelf switching by up to 40.5\%, 33.3\%, and 44.4\%, respectively, with even greater benefits under lift/place overhead. These results underscore the importance of execution-aware constraint release for scalable warehouse rearrangement. Code and data are available at this https URL.
>
---
#### [new 015] RAAP: Retrieval-Augmented Affordance Prediction with Cross-Image Action Alignment
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出RAAP框架，解决机器人在未见物体上的交互问题。通过结合检索与对齐学习，提升对象可操作性预测的泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.29419](https://arxiv.org/pdf/2603.29419)**

> **作者:** Qiyuan Zhuang; He-Yang Xu; Yijun Wang; Xin-Yang Zhao; Yang-Yang Li; Xiu-Shen Wei
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** Understanding object affordances is essential for enabling robots to perform purposeful and fine-grained interactions in diverse and unstructured environments. However, existing approaches either rely on retrieval, which is fragile due to sparsity and coverage gaps, or on large-scale models, which frequently mislocalize contact points and mispredict post-contact actions when applied to unseen categories, thereby hindering robust generalization. We introduce Retrieval-Augmented Affordance Prediction (RAAP), a framework that unifies affordance retrieval with alignment-based learning. By decoupling static contact localization and dynamic action direction, RAAP transfers contact points via dense correspondence and predicts action directions through a retrieval-augmented alignment model that consolidates multiple references with dual-weighted attention. Trained on compact subsets of DROID and HOI4D with as few as tens of samples per task, RAAP achieves consistent performance across unseen objects and categories, and enables zero-shot robotic manipulation in both simulation and the real world. Project website: this https URL.
>
---
#### [new 016] CReF: Cross-modal and Recurrent Fusion for Depth-conditioned Humanoid Locomotion
- **分类: cs.RO**

- **简介: 该论文属于人形机器人行走任务，解决复杂地形稳定行走问题。提出CReF框架，直接从深度信息学习行走特征，提升对复杂环境的适应能力。**

- **链接: [https://arxiv.org/pdf/2603.29452](https://arxiv.org/pdf/2603.29452)**

> **作者:** Yuan Hao; Ruiqi Yu; Shixin Luo; Guoteng Zhang; Jun Wu; Qiuguo Zhu
>
> **摘要:** Stable traversal over geometrically complex terrain increasingly requires exteroceptive perception, yet prior perceptive humanoid locomotion methods often remain tied to explicit geometric abstractions, either by mediating control through robot-centric 2.5D terrain representations or by shaping depth learning with auxiliary geometry-related targets. Such designs inherit the representational bias of the intermediate or supervisory target and can be restrictive for vertical structures, perforated obstacles, and complex real-world clutter. We propose CReF (Cross-modal and Recurrent Fusion), a single-stage depth-conditioned humanoid locomotion framework that learns locomotion-relevant features directly from raw forward-facing depth without explicit geometric intermediates. CReF couples proprioception and depth tokens through proprioception-queried cross-modal attention, fuses the resulting representation with a gated residual fusion block, and performs temporal integration with a Gated Recurrent Unit (GRU) regulated by a highway-style output gate for state-dependent blending of recurrent and feedforward features. To further improve terrain interaction, we introduce a terrain-aware foothold placement reward that extracts supportable foothold candidates from foot-end point-cloud samples and rewards touchdown locations that lie close to the nearest supportable candidate. Experiments in simulation and on a physical humanoid demonstrate robust traversal over diverse terrains and effective zero-shot transfer to real-world scenes containing handrails, hollow pallet assemblies, severe reflective interference, and visually cluttered outdoor surroundings.
>
---
#### [new 017] A Semantic Observer Layer for Autonomous Vehicles: Pre-Deployment Feasibility Study of VLMs for Low-Latency Anomaly Detection
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶安全任务，旨在解决语义异常检测问题。通过构建语义观察层，使用量化VLM实现低延迟异常检测，提升系统安全性。**

- **链接: [https://arxiv.org/pdf/2603.28888](https://arxiv.org/pdf/2603.28888)**

> **作者:** Kunal Runwal; Swaraj Gajare; Daniel Adejumo; Omkar Ankalkope; Siddhant Baroth; Aliasghar Arab
>
> **摘要:** Semantic anomalies-context-dependent hazards that pixel-level detectors cannot reason about-pose a critical safety risk in autonomous driving. We propose a \emph{semantic observer layer}: a quantized vision-language model (VLM) running at 1--2\,Hz alongside the primary AV control loop, monitoring for semantic edge cases, and triggering fail-safe handoffs when detected. Using Nvidia Cosmos-Reason1-7B with NVFP4 quantization and FlashAttention2, we achieve ~500 ms inference a ~50x speedup over the unoptimized FP16 baseline (no quantization, standard PyTorch attention) on the same hardware--satisfying the observer timing budget. We benchmark accuracy, latency, and quantization behavior in static and video conditions, identify NF4 recall collapse (10.6%) as a hard deployment constraint, and a hazard analysis mapping performance metrics to safety goals. The results establish a pre-deployment feasibility case for the semantic observer architecture on embodied-AI AV platforms.
>
---
#### [new 018] SuperGrasp: Single-View Object Grasping via Superquadric Similarity Matching, Evaluation, and Refinement
- **分类: cs.RO**

- **简介: 该论文属于单视角物体抓取任务，旨在解决因几何信息不全导致的抓取不稳定问题。提出SuperGrasp框架，通过相似性匹配和评估优化生成稳定抓取姿态。**

- **链接: [https://arxiv.org/pdf/2603.29254](https://arxiv.org/pdf/2603.29254)**

> **作者:** Lijingze Xiao; Jinhong Du; Yang Cong; Supeng Diao; Yu Ren
>
> **摘要:** Robotic grasping from single-view observations remains a critical challenge in manipulation. Existing methods still struggle to generate stable and valid grasp poses when confronted with incomplete geometric information. To address these limitations, we propose SuperGrasp, a novel two-stage framework for single-view grasping with parallel-jaw grippers that decomposes the grasping process into initial grasp pose generation and subsequent grasp evaluation and refinement. In the first stage, we introduce a Similarity Matching Module that efficiently retrieves grasp candidates by matching the input single-view point cloud with a pre-computed primitive dataset based on superquadric coefficients. In the second stage, we propose E-RNet, an end-to-end network that expands the graspaware region and takes the initial grasp closure region as a local anchor region, enabling more accurate and reliable evaluation and refinement of grasp candidates. To enhance generalization, we construct a primitive dataset containing 1.5k primitives for similarity matching and collect a large-scale point cloud dataset with 100k stable grasp labels from 124 objects for network training. Extensive experiments in both simulation and realworld environments demonstrate that our method achieves stable grasping performance and strong generalization across varying scenes and novel objects.
>
---
#### [new 019] Long-Reach Robotic Cleaning for Lunar Solar Arrays
- **分类: cs.RO**

- **简介: 该论文属于机器人维护任务，旨在解决月球太阳能板清洁问题。研究提出一种配备长臂机器人的移动系统，实现高效、稳定清洁，提升设备寿命与性能。**

- **链接: [https://arxiv.org/pdf/2603.29240](https://arxiv.org/pdf/2603.29240)**

> **作者:** Stanley Wang; Velin Kojouharov; Long Yin Chung; Daniel Morton; Mark Cutkosky
>
> **备注:** Extended abstract, 4 pages, 3 figures, accepted to and presented at the Sustainable Space Robotics Workshop at iSpaRo 2025
>
> **摘要:** Commercial lunar activity is accelerating the need for reliable surface infrastructure and routine operations to keep it functioning. Maintenance tasks such as inspection, cleaning, dust mitigation, and minor repair are essential to preserve performance and extend system life. A specific application is the cleaning of lunar solar arrays. Solar arrays are expected to provide substantial fraction of lunar surface power and operate for months to years, supplying continuous energy to landers, habitats, and surface assets, making sustained output mission-critical. However, over time lunar dust accumulates on these large solar arrays, which can rapidly degrade panel output and reduce mission lifetime. We propose a small mobile robot equipped with a long-reach, lightweight deployable boom and interchangeable cleaning tool to perform gentle cleaning over meter-scale workspaces with minimal human involvement. Building on prior vision-guided long-reach manipulation, we add a compliant wrist with distal force sensing and a velocity-based admittance controller to regulate stable contact during surface cleaning. In preliminary benchtop experiments on a planar surface, the system maintained approximately 2 N normal force while executing a simple cleaning motion over boom lengths from 0.3 m to 1.0 m, with RMS force error of approximately 0.2 N after initial contact. These early results suggest that deployable long-reach manipulators are a promising architecture for robotic maintenance of lunar infrastructure such as solar arrays, radiators, and optical surfaces.
>
---
#### [new 020] HapCompass: A Rotational Haptic Device for Contact-Rich Robotic Teleoperation
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于机器人远程操作任务，旨在解决接触丰富操作中方向反馈不足的问题。提出HapCompass设备，通过旋转线性共振执行器提供2D方向触觉提示，提升操作成功率和效率。**

- **链接: [https://arxiv.org/pdf/2603.30042](https://arxiv.org/pdf/2603.30042)**

> **作者:** Xiangshan Tan; Jingtian Ji; Tianchong Jiang; Pedro Lopes; Matthew R. Walter
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA), 2026. 8 pages, 5 figures. Project page: this https URL
>
> **摘要:** The contact-rich nature of manipulation makes it a significant challenge for robotic teleoperation. While haptic feedback is critical for contact-rich tasks, providing intuitive directional cues within wearable teleoperation interfaces remains a bottleneck. Existing solutions, such as non-directional vibrations from handheld controllers, provide limited information, while vibrotactile arrays are prone to perceptual interference. To address these limitations, we propose HapCompass, a novel, low-cost wearable haptic device that renders 2D directional cues by mechanically rotating a single linear resonant actuator (LRA). We evaluated HapCompass's ability to convey directional cues to human operators and showed that it increased the success rate, decreased the completion time and the maximum contact force for teleoperated manipulation tasks when compared to vision-only and non-directional feedback baselines. Furthermore, we conducted a preliminary imitation-learning evaluation, suggesting that the directional feedback provided by HapCompass enhances the quality of demonstration data and, in turn, the trained policy. We release the design of the HapCompass device along with the code that implements our teleoperation interface: this https URL.
>
---
#### [new 021] Industrial-Grade Robust Robot Vision for Screw Detection and Removal under Uneven Conditions
- **分类: cs.RO**

- **简介: 该论文属于工业机器人视觉任务，旨在解决空调外机拆解中的螺丝检测与移除问题。针对环境复杂和尺寸差异，提出两阶段检测与基于格网的校准方法，提升检测精度与拆解效率。**

- **链接: [https://arxiv.org/pdf/2603.29363](https://arxiv.org/pdf/2603.29363)**

> **作者:** Tomoki Ishikura; Genichiro Matsuda; Takuya Kiyokawa; Kensuke Harada
>
> **备注:** 19 pages, 14 figures
>
> **摘要:** As the amount of used home appliances is expected to increase despite the decreasing labor force in Japan, there is a need to automate disassembling processes at recycling plants. The automation of disassembling air conditioner outdoor units, however, remains a challenge due to unit size variations and exposure to dirt and rust. To address these challenges, this study proposes an automated system that integrates a task-specific two-stage detection method and a lattice-based local calibration strategy. This approach achieved a screw detection recall of 99.8% despite severe degradation and ensured a manipulation accuracy of +/-0.75 mm without pre-programmed coordinates. In real-world validation with 120 units, the system attained a disassembly success rate of 78.3% and an average cycle time of 193 seconds, confirming its feasibility for industrial application.
>
---
#### [new 022] Kilohertz-Safe: A Scalable Framework for Constrained Dexterous Retargeting
- **分类: cs.RO**

- **简介: 该论文属于机械臂运动重定向任务，解决高频率实时控制与安全约束问题。提出一种可扩展框架，将非线性问题转化为凸二次规划，提升效率与安全性。**

- **链接: [https://arxiv.org/pdf/2603.29213](https://arxiv.org/pdf/2603.29213)**

> **作者:** Yinxiao Tian; Ziyi Yang; Zinan Zhao; Zhen Kan
>
> **备注:** 8 pages,6 Figures,Under Reiview
>
> **摘要:** Dexterous hand teleoperation requires motion re-targeting methods that simultaneously achieve high-frequency real-time performance and enforcement of heterogeneous kinematic and safety constraints. Existing nonlinear optimization-based approaches often incur prohibitive computational cost, limiting their applicability to kilohertz-level control, while learning-based methods typically lack formal safety guarantees. This paper proposes a scalable motion retargeting framework that reformulates the nonlinear retargeting problem into a convex quadratic program in joint differential space. Heterogeneous constraints, including kinematic limits and collision avoidance, are incorporated through systematic linearization, resulting in improved computational efficiency and numerical stability. Control barrier functions are further integrated to provide formal safety guarantees during the retargeting process. The proposed framework is validated through simulations and hardware experiments on the Wuji Hand platform, outperforming state-of-the-art methods such as Dex-Retargeting and GeoRT. The framework achieves high-frequency operation with an average latency of 9.05 ms, while over 95% of retargeted frames satisfy the safety criteria, effectively mitigating self-collision and penetration during complex manipulation tasks.
>
---
#### [new 023] Passive iFIR filters for data-driven velocity control in robotics
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人速度控制任务，解决非线性机械臂的跟踪性能问题。通过数据驱动方法设计被动iFIR控制器，提升控制精度并保证系统稳定性。**

- **链接: [https://arxiv.org/pdf/2603.29882](https://arxiv.org/pdf/2603.29882)**

> **作者:** Yi Zhang; Zixing Wang; Fulvio Forni
>
> **摘要:** We present a passive, data-driven velocity control method for nonlinear robotic manipulators that achieves better tracking performance than optimized PID with comparable design complexity. Using only three minutes of probing data, a VRFT-based design identifies passive iFIR controllers that (i) preserve closed-loop stability via passivity constraints and (ii) outperform a VRFT-tuned PID baseline on the Franka Research 3 robot in both joint-space and Cartesian-space velocity control, achieving up to a 74.5% reduction in tracking error for the Cartesian velocity tracking experiment with the most demanding reference model. When the robot end-effector dynamics change, the controller can be re-learned from new data, regaining nominal performance. This study bridges learning-based control and stability-guaranteed design: passive iFIR learns from data while retaining passivity-based stability guarantees, unlike many learning-based approaches.
>
---
#### [new 024] Scaling Whole-Body Human Musculoskeletal Behavior Emulation for Specificity and Diversity
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人体运动控制模拟任务，旨在解决高维肌肉控制的优化难题。通过构建大规模并行计算框架，实现精准的全身运动再现与控制策略探索。**

- **链接: [https://arxiv.org/pdf/2603.29332](https://arxiv.org/pdf/2603.29332)**

> **作者:** Yunyue Wei; Chenhui Zuo; Shanning Zhuang; Haixin Gong; Yaming Liu; Yanan Sui
>
> **摘要:** The embodied learning of human motor control requires whole-body neuro-actuated musculoskeletal dynamics, while the internal muscle-driven processes underlying movement remain inaccessible to direct measurement. Computational modeling offers an alternative, but inverse dynamics methods struggled to resolve redundant control from observed kinematics in the high-dimensional, over-actuated system. Forward imitation approaches based on deep reinforcement learning exhibited inadequate tracking performance due to the curse of dimensionality in both control and reward design. Here we introduce a large-scale parallel musculoskeletal computation framework for biomechanically grounded whole-body motion reproduction. By integrating large-scale parallel GPU simulation with adversarial reward aggregation and value-guided flow exploration, the MS-Emulator framework overcomes key optimization bottlenecks in high-dimensional reinforcement learning for musculoskeletal control, which accurately reproduces a broad repertoire of motions in a whole-body human musculoskeletal system actuated by approximately 700 muscles. It achieved high joint angle accuracy and body position alignment for highly dynamic tasks such as dance, cartwheel, and backflip. The framework was also used to explore the musculoskeletal control solution space, identifying distinct musculoskeletal control policies that converge to nearly identical external kinematic and mechanical measurements. This work establishes a tractable computational route to analyzing the specificity and diversity underlying human embodied control of movement. Project page: this https URL.
>
---
#### [new 025] Efficient Camera Pose Augmentation for View Generalization in Robotic Policy Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人策略学习任务，解决2D视觉策略在新视角下的泛化问题。通过3D高斯点云框架GenSplat，提升策略对3D结构的依赖，增强空间扰动下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.29192](https://arxiv.org/pdf/2603.29192)**

> **作者:** Sen Wang; Huaiyi Dong; Jingyi Tian; Jiayi Li; Zhuo Yang; Tongtong Cao; Anlin Chen; Shuang Wu; Le Wang; Sanping Zhou
>
> **摘要:** Prevailing 2D-centric visuomotor policies exhibit a pronounced deficiency in novel view generalization, as their reliance on static observations hinders consistent action mapping across unseen views. In response, we introduce GenSplat, a feed-forward 3D Gaussian Splatting framework that facilitates view-generalized policy learning through novel view rendering. GenSplat employs a permutation-equivariant architecture to reconstruct high-fidelity 3D scenes from sparse, uncalibrated inputs in a single forward pass. To ensure structural integrity, we design a 3D-prior distillation strategy that regularizes the 3DGS optimization, preventing the geometric collapse typical of purely photometric supervision. By rendering diverse synthetic views from these stable 3D representations, we systematically augment the observational manifold during training. This augmentation forces the policy to ground its decisions in underlying 3D structures, thereby ensuring robust execution under severe spatial perturbations where baselines severely degrade.
>
---
#### [new 026] See Something, Say Something: Context-Criticality-Aware Mobile Robot Communication for Hazard Mitigations
- **分类: cs.RO**

- **简介: 该论文属于移动机器人通信任务，旨在解决危险情境下响应延迟问题。通过上下文感知的危急程度评估，提升机器人通信的及时性与有效性。**

- **链接: [https://arxiv.org/pdf/2603.28901](https://arxiv.org/pdf/2603.28901)**

> **作者:** Bhavya Oza; Devam Shah; Ghanashyama Prabhu; Devika Kodi; Aliasghar Arab
>
> **摘要:** The proverb ``see something, say something'' captures a core responsibility of autonomous mobile robots in safety-critical situations: when they detect a hazard, they must communicate--and do so quickly. In emergency scenarios, delayed or miscalibrated responses directly increase the time to action and the risk of damage. We argue that a systematic context-sensitive assessment of the criticality level, time sensitivity, and feasibility of mitigation is necessary for AMRs to reduce time to action and respond effectively. This paper presents a framework in which VLM/LLM-based perception drives adaptive message generation, for example, a knife in a kitchen produces a calm acknowledgment; the same object in a corridor triggers an urgent coordinated alert. Validation in 60+ runs using a patrolling mobile robot not only empowers faster response, but also brings user trusts to 82\% compared to fixed-priority baselines, validating that structured criticality assessment improves both response speed and mitigation effectiveness.
>
---
#### [new 027] Why That Robot? A Qualitative Analysis of Justification Strategies for Robot Color Selection Across Occupational Contexts
- **分类: cs.RO**

- **简介: 该论文属于人机交互研究，探讨机器人颜色选择中的社会偏见问题。通过分析用户理由，揭示隐性偏见对机器人设计的影响，并提出减少偏见的设计建议。**

- **链接: [https://arxiv.org/pdf/2603.28919](https://arxiv.org/pdf/2603.28919)**

> **作者:** Jiangen He; Wanqi Zhang; Jessica K. Barfield
>
> **摘要:** As robots increasingly enter the workforce, human-robot interaction (HRI) must address how implicit social biases influence user preferences. This paper investigates how users rationalize their selections of robots varying in skin tone and anthropomorphic features across different occupations. By qualitatively analyzing 4,146 open-ended justifications from 1,038 participants, we map the reasoning frameworks driving robot color selection across four professional contexts. We developed and validated a comprehensive, multidimensional coding scheme via human--AI consensus ($\kappa = 0.73$). Our results demonstrate that while utilitarian \textit{Functionalism} is the dominant justification strategy (52\%), participants systematically adapted these practical rationales to align with established racial and occupational stereotypes. Furthermore, we reveal that bias frequently operates beneath conscious rationalization: exposure to racial stereotype primes significantly shifted participants' color choices, yet their spoken justifications remained masked by standard affective or task-related reasoning. We also found that demographic backgrounds significantly shape justification strategies, and that robot shape strongly modulates color interpretation. Specifically, as robots become highly anthropomorphic, users increasingly retreat from functional reasoning toward \textit{Machine-Centric} de-racialization. Through these empirical results, we provide actionable design implications to help reduce the perpetuation of societal biases in future workforce robots.
>
---
#### [new 028] Interacting Multiple Model Proprioceptive Odometry for Legged Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人状态估计任务，旨在解决腿式机器人在缺乏可靠外部传感器时的位姿估计问题。通过引入IMM框架，提升 proprioceptive 里程计的准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.29383](https://arxiv.org/pdf/2603.29383)**

> **作者:** Wanlei Li; Zichang Chen; Shilei Li; Xiaogang Xiong; Yunjiang Lou
>
> **摘要:** State estimation for legged robots remains challenging because legged odometry generally suffers from limited observability and therefore depends critically on measurement constraints to suppress drift. When exteroceptive sensors are unreliable or degraded, such constraints are mainly derived from proprioceptive measurements, particularly contact-related leg kinematics information. However, most existing proprioceptive odometry methods rely on an idealized point-contact assumption, which is often violated during real locomotion. Consequently, the effectiveness of proprioceptive constraints may be significantly reduced, resulting in degraded estimation accuracy. To address these limitations, we propose an interacting multiple model (IMM)-based proprioceptive odometry framework for legged robots. By incorporating multiple contact hypotheses within a unified probabilistic framework, the proposed method enables online mode switching and probabilistic fusion under varying contact conditions. Extensive simulations and real-world experiments demonstrate that the proposed method achieves superior pose estimation accuracy over state-of-the-art methods while maintaining comparable computational efficiency.
>
---
#### [new 029] Communication Outage-Resistant UUV State Estimation: A Variational History Distillation Approach
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于UUV状态估计任务，旨在解决通信中断导致的估计误差问题。通过引入VHD方法，结合历史轨迹与物理模型，提升长时间通信中断下的状态估计精度。**

- **链接: [https://arxiv.org/pdf/2603.29512](https://arxiv.org/pdf/2603.29512)**

> **作者:** Shuyue Li; Miguel López-Benítez; Eng Gee Lim; Fei Ma; Qian Dong; Mengze Cao; Limin Yu; Xiaohui Qin
>
> **备注:** 7 pages, 2 figures,conference
>
> **摘要:** The reliable operation of Unmanned Underwater Vehicle (UUV) clusters is highly dependent on continuous acoustic communication. However, this communication method is highly susceptible to intermittent interruptions. When communication outages occur, standard state estimators such as the Unscented Kalman Filter (UKF) will be forced to make open-loop predictions. If the environment contains unmodeled dynamic factors, such as unknown ocean currents, this estimation error will grow rapidly, which may eventually lead to mission failure. To address this critical issue, this paper proposes a Variational History Distillation (VHD) approach. VHD regards trajectory prediction as an approximate Bayesian reasoning process, which links a standard motion model based on physics with a pattern extracted directly from the past trajectory of the UUV. This is achieved by synthesizing ``virtual measurements'' distilled from historical trajectories. Recognizing that the reliability of extrapolated historical trends degrades over extended prediction horizons, an adaptive confidence mechanism is introduced. This mechanism allows the filter to gradually reduce the trust of virtual measurements as the communication outage time is extended. Extensive Monte Carlo simulations in a high-fidelity environment demonstrate that the proposed method achieves a 91\% reduction in prediction Root Mean Square Error (RMSE), reducing the error from approximately 170 m to 15 m during a 40-second communication outage. These results demonstrate that VHD can maintain robust state estimation performance even under complete communication loss.
>
---
#### [new 030] IMPASTO: Integrating Model-Based Planning with Learned Dynamics Models for Robotic Oil Painting Reproduction
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出IMPASTO系统，解决机器人油彩绘画再现问题。通过结合模型预测与学习动力学模型，实现力控、轨迹规划与颜色控制，提升绘画准确性。**

- **链接: [https://arxiv.org/pdf/2603.29315](https://arxiv.org/pdf/2603.29315)**

> **作者:** Yingke Wang; Hao Li; Yifeng Zhu; Hong-Xing Yu; Ken Goldberg; Li Fei-Fei; Jiajun Wu; Yunzhu Li; Ruohan Zhang
>
> **摘要:** Robotic reproduction of oil paintings using soft brushes and pigments requires force-sensitive control of deformable tools, prediction of brushstroke effects, and multi-step stroke planning, often without human step-by-step demonstrations or faithful simulators. Given only a sequence of target oil painting images, can a robot infer and execute the stroke trajectories, forces, and colors needed to reproduce it? We present IMPASTO, a robotic oil-painting system that integrates learned pixel dynamics models with model-based planning. The dynamics models predict canvas updates from image observations and parameterized stroke actions; a receding-horizon model predictive control optimizer then plans trajectories and forces, while a force-sensitive controller executes strokes on a 7-DoF robot arm. IMPASTO integrates low-level force control, learned dynamics models, and high-level closed-loop planning, learns solely from robot self-play, and approximates human artists' single-stroke datasets and multi-stroke artworks, outperforming baselines in reproduction accuracy. Project website: this https URL
>
---
#### [new 031] Robust Multi-Agent Reinforcement Learning for Small UAS Separation Assurance under GPS Degradation and Spoofing
- **分类: cs.RO; cs.AI; cs.LG; eess.SY**

- **简介: 该论文属于多智能体强化学习任务，解决GPS受损和欺骗下的小型无人机间隔保障问题。通过构建对抗模型，设计鲁棒策略以提升安全性。**

- **链接: [https://arxiv.org/pdf/2603.28900](https://arxiv.org/pdf/2603.28900)**

> **作者:** Alex Zongo; Filippos Fotiadis; Ufuk Topcu; Peng Wei
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** We address robust separation assurance for small Unmanned Aircraft Systems (sUAS) under GPS degradation and spoofing via Multi-Agent Reinforcement Learning (MARL). In cooperative surveillance, each aircraft (or agent) broadcasts its GPS-derived position; when such position broadcasts are corrupted, the entire observed air traffic state becomes unreliable. We cast this state observation corruption as a zero-sum game between the agents and an adversary: with probability R, the adversary perturbs the observed state to maximally degrade each agent's safety performance. We derive a closed-form expression for this adversarial perturbation, bypassing adversarial training entirely and enabling linear-time evaluation in the state dimension. We show that this expression approximates the true worst-case adversarial perturbation with second-order accuracy. We further bound the safety performance gap between clean and corrupted observations, showing that it degrades at most linearly with the corruption probability under Kullback-Leibler regularization. Finally, we integrate the closed-form adversarial policy into a MARL policy gradient algorithm to obtain a robust counter-policy for the agents. In a high-density sUAS simulation, we observe near-zero collision rates under corruption levels up to 35%, outperforming a baseline policy trained without adversarial perturbations.
>
---
#### [new 032] Design and Aerodynamic Modeling of MetaMorpher: A Hybrid Rotary andFixed-Wing Morphing UAV
- **分类: cs.RO**

- **简介: 该论文属于无人机设计任务，旨在解决垂直起降与固定翼巡航效率的矛盾。提出一种新型可变形无人机模型，建立非线性动力学模型并进行仿真测试。**

- **链接: [https://arxiv.org/pdf/2603.29646](https://arxiv.org/pdf/2603.29646)**

> **作者:** Anja Bosak; Dorian Erić; Ana Milas; Stjepan Bogdan
>
> **备注:** 8 pages, 12 figures
>
> **摘要:** In this paper, we present a generalized, comprehensive nonlinear mathematical model and conceptual design for the MetaMorpher, a metamorphic Unmanned Aerial Vehicle (UAV) designed to bridge the gap between vertical takeoff and landing agility and fixed-wing cruising efficiency. Building on the successful design of the spincopter platform, this work introduces a simplified mechanical architecture using lightweight materials and a novel wing-folding strategy. Unlike traditional rigid-body approximations, we derive a nonlinear flight dynamics model that enables arbitrary force distributions across a segmented wing structure. This modularity allows for testing different airfoils, mass distributions, and chord lengths in a single environment. As part of this work, various flight modes were specifically tested and analyzed in the Simulink environment. The results show that the model behaves predictably under different structural configurations, demonstrating its reliability as a tool for rapid design evaluation.
>
---
#### [new 033] Reconfiguration of supernumerary robotic limbs for human augmentation
- **分类: cs.RO**

- **简介: 该论文属于人机协作任务，旨在解决SRL在动态环境中的适应性问题。提出一种可重构框架，通过量化分析优化SRL配置与控制，提升人类增强效果。**

- **链接: [https://arxiv.org/pdf/2603.29808](https://arxiv.org/pdf/2603.29808)**

> **作者:** Mustafa Mete; Anastasia Bolotnikova; Alexander Schuessler; Jamie Paik
>
> **摘要:** Wearable robots aim to seamlessly adapt to humans and their environment with personalized interactions. Existing supernumerary robotic limbs (SRLs), which enhance the physical capabilities of humans with additional extremities, have thus far been developed primarily for task-specific applications in structured industrial settings, limiting their adaptability to dynamic and unstructured environments. Here, we introduce a novel reconfigurable SRL framework grounded in a quantitative analysis of human augmentation to guide the development of more adaptable SRLs for diverse scenarios. This framework captures how SRL configuration shapes workspace extension and human-robot collaboration. We define human augmentation ratios to evaluate collaborative, visible extended, and non-visible extended workspaces, enabling systematic selection of SRL placement, morphology, and autonomy for a given task. Using these metrics, we demonstrate how quantitative augmentation analysis can guide the reconfiguration and control of SRLs to better match task requirements. We validate the proposed approach through experiments with a reconfigurable SRL composed of origami-inspired modular elements. Our results suggest that reconfigurable SRLs, informed by quantitative human augmentation analysis, offer a new perspective for providing adaptable human augmentation and assistance in everyday environments.
>
---
#### [new 034] Kernel-SDF: An Open-Source Library for Real-Time Signed Distance Function Estimation using Kernel Regression
- **分类: cs.RO**

- **简介: 该论文属于环境建模任务，解决机器人应用中实时准确的SDF估计问题。提出Kernel-SDF库，利用核回归实现带有不确定性量化的真实SDF估计。**

- **链接: [https://arxiv.org/pdf/2603.29227](https://arxiv.org/pdf/2603.29227)**

> **作者:** Zhirui Dai; Tianxing Fan; Mani Amani; Jaemin Seo; Ki Myung Brian Lee; Hyondong Oh; Nikolay Atanasov
>
> **摘要:** Accurate and efficient environment representation is crucial for robotic applications such as motion planning, manipulation, and navigation. Signed distance functions (SDFs) have emerged as a powerful representation for encoding distance to obstacle boundaries, enabling efficient collision-checking and trajectory optimization techniques. However, existing SDF reconstruction methods have limitations when it comes to large-scale uncertainty-aware SDF estimation from streaming sensor data. Voxel-based approaches are limited by fixed resolution and lack uncertainty quantification, neural network methods require significant training time, while Gaussian process (GP) methods struggle with scalability, sign estimation, and uncertainty calibration. In this letter, we develop an open-source library, Kernel-SDF, which uses kernel regression to learn SDF with calibrated uncertainty quantification in real-time. Our approach consists of a front-end that learns a continuous occupancy field via kernel regression, and a back-end that estimates accurate SDF via GP regression using samples from the front-end surface boundaries. Kernel-SDF provides accurate SDF, SDF gradient, SDF uncertainty, and mesh construction in real-time. Evaluation results show that Kernel-SDF achieves superior accuracy compared to existing methods, while maintaining real-time performance, making it suitable for various robotics applications requiring reliable uncertainty-aware geometric information.
>
---
#### [new 035] Gleanmer: A 6 mW SoC for Real-Time 3D Gaussian Occupancy Mapping
- **分类: cs.RO**

- **简介: 论文提出Gleanmer SoC，解决边缘计算中实时3D占用映射的高功耗问题。通过算法硬件协同优化，实现低功耗高效映射。**

- **链接: [https://arxiv.org/pdf/2603.29005](https://arxiv.org/pdf/2603.29005)**

> **作者:** Zih-Sing Fu; Peter Zhi Xuan Li; Sertac Karaman; Vivienne Sze
>
> **备注:** Accepted to IEEE Symposium on VLSI Technology & Circuits (VLSI), 2026. To appear
>
> **摘要:** High-fidelity 3D occupancy mapping is essential for many edge-based applications (such as AR/VR and autonomous navigation) but is limited by power constraints. We present Gleanmer, a system on chip (SoC) with an accelerator for GMMap, a 3D occupancy map using Gaussians. Through algorithm-hardware co-optimizations for direct computation and efficient reuse of these compact Gaussians, Gleanmer reduces construction and query energy by up to 63% and 81%, respectively. Approximate computation on Gaussians reduces accelerator area by 38%. Using 16nm CMOS, Gleanmer processes 640x480 images in real time beyond 88 fps during map construction and processes over 540K coordinates per second during map query. To our knowledge, Gleanmer is the first fabricated SoC to achieve real-time 3D occupancy mapping under 6 mW for edge-based applications.
>
---
#### [new 036] HCLSM: Hierarchical Causal Latent State Machines for Object-Centric World Modeling
- **分类: cs.LG; cs.CV; cs.RO**

- **简介: 该论文提出HCLSM，用于物体中心世界建模任务，解决传统模型在物体纠缠、因果结构和时序动态上的不足，通过分层时序和因果学习提升状态预测效果。**

- **链接: [https://arxiv.org/pdf/2603.29090](https://arxiv.org/pdf/2603.29090)**

> **作者:** Jaber Jaber; Osama Jaber
>
> **备注:** 10 pages, 3 tables, 4 figures, 1 algorithm. Code: this https URL
>
> **摘要:** World models that predict future states from video remain limited by flat latent representations that entangle objects, ignore causal structure, and collapse temporal dynamics into a single scale. We present HCLSM, a world model architecture that operates on three interconnected principles: object-centric decomposition via slot attention with spatial broadcast decoding, hierarchical temporal dynamics through a three-level engine combining selective state space models for continuous physics, sparse transformers for discrete events, and compressed transformers for abstract goals, and causal structure learning through graph neural network interaction patterns. HCLSM introduces a two-stage training protocol where spatial reconstruction forces slot specialization before dynamics prediction begins. We train a 68M-parameter model on the PushT robotic manipulation benchmark from the Open X-Embodiment dataset, achieving 0.008 MSE next-state prediction loss with emerging spatial decomposition (SBD loss: 0.0075) and learned event boundaries. A custom Triton kernel for the SSM scan delivers 38x speedup over sequential PyTorch. The full system spans 8,478 lines of Python across 51 modules with 171 unit tests. Code: this https URL
>
---
#### [new 037] An Interactive LLM-Based Simulator for Dementia-Related Activities of Daily Living
- **分类: cs.HC; cs.GR; cs.RO**

- **简介: 该论文属于 dementia care 任务，旨在解决缺乏真实 ADL 数据的问题。通过构建基于 LLM 的模拟器生成患者行为，支持护理训练与 AI 开发。**

- **链接: [https://arxiv.org/pdf/2603.29856](https://arxiv.org/pdf/2603.29856)**

> **作者:** Kruthika Gangaraju; Shu-Fen Wung; Kevin Berner; Jing Wang; Fengpei Yuan
>
> **摘要:** Effective dementia caregiving requires training and adaptive communication, but assistive AI and robotics are constrained by a lack of context-rich, privacy-sensitive data on how people living with Alzheimer's disease and related dementias (ADRD) behave during activities of daily living (ADLs). We introduce a web-based simulator that uses a large language model (gpt-5-mini) to generate multi-turn, severity- and care-setting-conditioned patient behaviors during ADL assistance, pairing utterances with lightweight behavioral cues (in parentheses). Users set dementia severity, care setting (and time in setting), and ADL; after each patient turn they rate realism (1-5) with optional critique, then respond as the caregiver via free text or by selecting/editing one of four strategy-scaffolded suggestions (Recognition, Negotiation, Facilitation, Validation). We ran an online formative expert-in-the-loop study (14 dementia-care experts, 18 sessions, 112 rated turns). Simulated behavior was judged moderately to highly plausible, with a typical session length of six turns. Experts wrote custom replies for 54.5 percent of turns; Recognition and Facilitation were the most-used suggested strategies. Thematic analysis of critiques produced a six-category failure-mode taxonomy, revealing recurring breakdowns in ADL grounding and care-setting consistency and guiding prompt/workflow refinements. The simulator and logged interactions enable an evidence-driven refinement loop toward validated patient-caregiver co-simulation and support data collection, caregiver training, and assistive AI and robot policy development.
>
---
#### [new 038] Stable Walking for Bipedal Locomotion under Foot-Slip via Virtual Nonholonomic Constraints
- **分类: eess.SY; cs.RO; math.DS; math.OC**

- **简介: 该论文属于双足行走控制任务，旨在解决低摩擦地形下的足部滑动问题。通过引入虚拟非完整约束，构建稳定步行模型，提升系统在滑动情况下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.29050](https://arxiv.org/pdf/2603.29050)**

> **作者:** Leonardo Colombo; Álvaro Rodríguez Abella; Alexandre Anahory Simoes; Anthony Bloch
>
> **摘要:** Foot slip is a major source of instability in bipedal locomotion on low-friction or uncertain terrain. Standard control approaches typically assume no-slip contact and therefore degrade when slip occurs. We propose a control framework that explicitly incorporates slip into the locomotion model through virtual nonholonomic constraints, which regulate the tangential stance-foot velocity while remaining compatible with the virtual holonomic constraints used to generate the walking gait. The resulting closed-loop system is formulated as a hybrid dynamical system with continuous swing dynamics and discrete impact events. A nonlinear feedback law enforces both classes of constraints and yields a slip-compatible hybrid zero dynamics manifold for the reduced-order locomotion dynamics. Stability of periodic walking gaits is characterized through the associated Poincaré map, and numerical results illustrate stabilization under slip conditions.
>
---
#### [new 039] Koopman Operator Framework for Modeling and Control of Off-Road Vehicle on Deformable Terrain
- **分类: eess.SY; cs.RO; math.DS**

- **简介: 该论文属于自主车辆控制任务，解决复杂地形下车辆建模与控制问题。通过融合物理模型与数据驱动的Koopman框架，实现高效预测与稳定控制。**

- **链接: [https://arxiv.org/pdf/2603.28965](https://arxiv.org/pdf/2603.28965)**

> **作者:** Kartik Loya; Phanindra Tallapragada
>
> **备注:** Submitted to ASME Journal of Autonomous Vehicles (JAVS-26-1012)
>
> **摘要:** This work presents a hybrid physics-informed and data-driven modeling framework for predictive control of autonomous off-road vehicles operating on deformable terrain. Traditional high-fidelity terramechanics models are often too computationally demanding to be directly used in control design. Modern Koopman operator methods can be used to represent the complex terramechanics and vehicle dynamics in a linear form. We develop a framework whereby a Koopman linear system can be constructed using data from simulations of a vehicle moving on deformable terrain. For vehicle simulations, the deformable-terrain terramechanics are modeled using Bekker-Wong theory, and the vehicle is represented as a simplified five-degree-of-freedom (5-DOF) system. The Koopman operators are identified from large simulation datasets for sandy loam and clay using a recursive subspace identification method, where Grassmannian distance is used to prioritize informative data segments during training. The advantage of this approach is that the Koopman operator learned from simulations can be updated with data from the physical system in a seamless manner, making this a hybrid physics-informed and data-driven approach. Prediction results demonstrate stable short-horizon accuracy and robustness under mild terrain-height variations. When embedded in a constrained MPC, the learned predictor enables stable closed-loop tracking of aggressive maneuvers while satisfying steering and torque limits.
>
---
#### [new 040] Model Predictive Path Integral PID Control for Learning-Based Path Following
- **分类: eess.SY; cs.LG; cs.RO; math.OC**

- **简介: 该论文属于路径跟踪控制任务，旨在解决传统PID和MPPI控制的不足，通过结合MPPI与PID提出新方法，提升跟踪性能与样本效率。**

- **链接: [https://arxiv.org/pdf/2603.29499](https://arxiv.org/pdf/2603.29499)**

> **作者:** Teruki Kato; Koshi Oishi; Seigo Ito
>
> **备注:** Submitted to IFAC Journal of Systems and Control
>
> **摘要:** Classical proportional--integral--derivative (PID) control is widely employed in industrial applications; however, achieving higher performance often motivates the adoption of model predictive control (MPC). Although gradient-based methods are the standard for real-time optimization, sampling-based approaches have recently gained attention. In particular, model predictive path integral (MPPI) control enables gradient-free optimization and accommodates non-differentiable models and objective functions. However, directly sampling control input sequences may yield discontinuous inputs and increase the optimization dimensionality in proportion to the prediction horizon. This study proposes MPPI--PID control, which applies MPPI to optimize PID gains at each control step, thereby replacing direct high-dimensional input-sequence optimization with low-dimensional gain-space optimization. This formulation enhances sample efficiency and yields smoother inputs via the PID structure. We also provide theoretical insights, including an information-theoretic interpretation that unifies MPPI and MPPI--PID, an analysis of the effect of optimization dimensionality on sample efficiency, and a characterization of input continuity induced by the PID structure. The proposed method is evaluated on the learning-based path following of a mini forklift using a residual-learning dynamics model that integrates a physical model with a neural network. System identification is performed with real driving data. Numerical path-following experiments demonstrate that MPPI--PID improves tracking performance compared with fixed-gain PID and achieves performance comparable to conventional MPPI while significantly reducing input increments. Furthermore, the proposed method maintains favorable performance even with substantially fewer samples, demonstrating its improved sample efficiency.
>
---
#### [new 041] Native-Domain Cross-Attention for Camera-LiDAR Extrinsic Calibration Under Large Initial Perturbations
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于相机-LiDAR外参标定任务，旨在解决大初始偏差下的跨模态匹配问题。提出一种原生域交叉注意力框架，直接对齐图像块与点云组，提升标定精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.29414](https://arxiv.org/pdf/2603.29414)**

> **作者:** Ni Ou; Zhuo Chen; Xinru Zhang; Junzheng Wang
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Accurate camera-LiDAR fusion relies on precise extrinsic calibration, which fundamentally depends on establishing reliable cross-modal correspondences under potentially large misalignments. Existing learning-based methods typically project LiDAR points into depth maps for feature fusion, which distorts 3D geometry and degrades performance when the extrinsic initialization is far from the ground truth. To address this issue, we propose an extrinsic-aware cross-attention framework that directly aligns image patches and LiDAR point groups in their native domains. The proposed attention mechanism explicitly injects extrinsic parameter hypotheses into the correspondence modeling process, enabling geometry-consistent cross-modal interaction without relying on projected 2D depth maps. Extensive experiments on the KITTI and nuScenes benchmarks demonstrate that our method consistently outperforms state-of-the-art approaches in both accuracy and robustness. Under large extrinsic perturbations, our approach achieves accurate calibration in 88% of KITTI cases and 99% of nuScenes cases, substantially surpassing the second-best baseline. We have open sourced our code on this https URL to benefit the community.
>
---
#### [new 042] OccSim: Multi-kilometer Simulation with Long-horizon Occupancy World Models
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出OccSim，用于自动驾驶仿真，解决传统依赖真实数据或地图的局限。通过占用世界模型生成大规模3D场景，实现长距离、多场景的模拟。**

- **链接: [https://arxiv.org/pdf/2603.28887](https://arxiv.org/pdf/2603.28887)**

> **作者:** Tianran Liu; Shengwen Zhao; Mozhgan Pourkeshavarz; Weican Li; Nicholas Rhinehart
>
> **摘要:** Data-driven autonomous driving simulation has long been constrained by its heavy reliance on pre-recorded driving logs or spatial priors, such as HD maps. This fundamental dependency severely limits scalability, restricting open-ended generation capabilities to the finite scale of existing collected datasets. To break this bottleneck, we present OccSim, the first occupancy world model-driven 3D simulator. OccSim obviates the requirement for continuous logs or HD maps; conditioned only on a single initial frame and a sequence of future ego-actions, it can stably generate over 3,000 continuous frames, enabling the continuous construction of large-scale 3D occupancy maps spanning over 4 kilometers for simulation. This represents an >80x improvement in stable generation length over previous state-of-the-art occupancy world models. OccSim is powered by two modules: W-DiT based static occupancy world model and the Layout Generator. W-DiT handles the ultra-long-horizon generation of static environments by explicitly introducing known rigid transformations in architecture design, while the Layout Generator populates the dynamic foreground with reactive agents based on the synthesized road topology. With these designs, OccSim can synthesize massive, diverse simulation streams. Extensive experiments demonstrate its downstream utility: data collected directly from OccSim can pre-train 4D semantic occupancy forecasting models to achieve up to 67% zero-shot performance on unseen data, outperforming previous asset-based simulator by 11%. When scaling the OccSim dataset to 5x the size, the zero-shot performance increases to about 74%, while the improvement over asset-based simulators expands to 22.1%.
>
---
#### [new 043] MaskAdapt: Learning Flexible Motion Adaptation via Mask-Invariant Prior for Physics-Based Characters
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文属于物理角色运动适应任务，解决部分观测缺失下的灵活运动调整问题。通过两阶段学习框架，实现对特定身体部位的精准控制与行为保持。**

- **链接: [https://arxiv.org/pdf/2603.29272](https://arxiv.org/pdf/2603.29272)**

> **作者:** Soomin Park; Eunseong Lee; Kwang Bin Lee; Sung-Hee Lee
>
> **备注:** CVPR 2026
>
> **摘要:** We present MaskAdapt, a framework for flexible motion adaptation in physics-based humanoid control. The framework follows a two-stage residual learning paradigm. In the first stage, we train a mask-invariant base policy using stochastic body-part masking and a regularization term that enforces consistent action distributions across masking conditions. This yields a robust motion prior that remains stable under missing observations, anticipating later adaptation in those regions. In the second stage, a residual policy is trained atop the frozen base controller to modify only the targeted body parts while preserving the original behaviors elsewhere. We demonstrate the versatility of this design through two applications: (i) motion composition, where varying masks enable multi-part adaptation within a single sequence, and (ii) text-driven partial goal tracking, where designated body parts follow kinematic targets provided by a pre-trained text-conditioned autoregressive motion generator. Through experiments, MaskAdapt demonstrates strong robustness and adaptability, producing diverse behaviors under masked observations and delivering superior targeted motion adaptation compared to prior work.
>
---
#### [new 044] PRISM: A Multi-View Multi-Capability Retail Video Dataset for Embodied Vision-Language Models
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出PRISM数据集，用于提升实体视觉语言模型在零售环境中的表现。解决实体AI在空间、物理和动作理解上的不足，通过多视角视频数据增强模型能力。**

- **链接: [https://arxiv.org/pdf/2603.29281](https://arxiv.org/pdf/2603.29281)**

> **作者:** Amirreza Rouhi; Parikshit Sakurikar; Satya Sai Reddy; Narsimha Menga; Anirudh Govil; Sri Harsha Chittajallu; Rajat Aggarwal; Anoop Namboodiri; Sashi Reddi
>
> **摘要:** A critical gap exists between the general-purpose visual understanding of state-of-the-art physical AI models and the specialized perceptual demands of structured real-world deployment environments. We present PRISM, a 270K-sample multi-view video supervised fine-tuning (SFT) corpus for embodied vision-language-models (VLMs) in real-world retail environments. PRISM is motivated by a simple observation - physical AI systems fail not because of poor visual recognition, but because they do not understand space, physical dynamics and embodied action well enough to operate reliably in the world. To this end, PRISM is grounded in a novel three-dimensional knowledge ontology that spans spatial knowledge, temporal and physical knowledge, and embodied action knowledge. It covers 20+ capability probes across four evaluation dimensions - Embodied Reasoning (ER), Common Sense (CS), Spatial Perception (SP), and Intuitive Physics (IP), and to our knowledge, PRISM is the first dataset to instantiate all three knowledge dimensions within a single real-world deployment domain. The corpus captures data from egocentric, exocentric and 360° viewpoints across five supermarket locations and includes open-ended, chain-of-thought, and multiple-choice supervision. At 4 fps, PRISM spans approximately 11.8M video frames and approximately 730M tokens, placing it among the largest domain-specific video SFT corpora. Fine-tuning on PRISM reduces the error rate across all 20+ probes by 66.6% over the pre-trained baseline, with significant gains in embodied action understanding where the accuracy improves by 36.4%. Our results suggest that ontology-structured, domain specific SFT can meaningfully strengthen embodied VLMs for real-world settings. The PRISM dataset and more details are available at this https URL
>
---
#### [new 045] LatentPilot: Scene-Aware Vision-and-Language Navigation by Dreaming Ahead with Latent Visual Reasoning
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决模型对动作与视觉变化因果关系理解不足的问题。通过引入未来观察和潜在视觉推理，提升导航决策能力。**

- **链接: [https://arxiv.org/pdf/2603.29165](https://arxiv.org/pdf/2603.29165)**

> **作者:** Haihong Hao; Lei Chen; Mingfei Han; Changlin Li; Dong An; Yuqiang Yang; Zhihui Li; Xiaojun Chang
>
> **备注:** Project page:this https URL
>
> **摘要:** Existing vision-and-language navigation (VLN) models primarily reason over past and current visual observations, while largely ignoring the future visual dynamics induced by actions. As a result, they often lack an effective understanding of the causal relationship between actions and how the visual world changes, limiting robust decision-making. Humans, in contrast, can imagine the near future by leveraging action-dynamics causality, which improves both environmental understanding and navigation choices. Inspired by this capability, we propose LatentPilot, a new paradigm that exploits future observations during training as a valuable data source to learn action-conditioned visual dynamics, while requiring no access to future frames at inference. Concretely, we propose a flywheel-style training mechanism that iteratively collects on-policy trajectories and retrains the model to better match the agent's behavior distribution, with an expert takeover triggered when the agent deviates excessively. LatentPilot further learns visual latent tokens without explicit supervision; these latent tokens attend globally in a continuous latent space and are carried across steps, serving as both the current output and the next input, thereby enabling the agent to dream ahead and reason about how actions will affect subsequent observations. Experiments on R2R-CE, RxR-CE, and R2R-PE benchmarks achieve new SOTA results, and real-robot tests across diverse environments demonstrate LatentPilot's superior understanding of environment-action dynamics in scene. Project page:this https URL
>
---
#### [new 046] Distributed Predictive Control Barrier Functions: Towards Scalable Safety Certification in Modular Multi-Agent Systems
- **分类: eess.SY; cs.RO; math.OC**

- **简介: 该论文属于多智能体系统安全控制任务，解决分布式控制中安全性不足的问题。通过引入D-PCBF框架，确保网络拓扑变化时系统的安全恢复。**

- **链接: [https://arxiv.org/pdf/2603.29560](https://arxiv.org/pdf/2603.29560)**

> **作者:** Jonas Ohnemus; Alexandre Didier; Ahmed Aboudonia; Andrea Carron; Melanie N. Zeilinger
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** We consider safety-critical multi-agent systems with distributed control architectures and potentially varying network topologies. While learning-based distributed control enables scalability and high performance, a lack of formal safety guarantees in the face of unforeseen disturbances and unsafe network topology changes may lead to system failure. To address this challenge, we introduce structured control barrier functions (s-CBFs) as a multi-agent safety framework. The s-CBFs are augmented to a distributed predictive control barrier function (D-PCBF), a predictive, optimization-based safety layer that uses model predictions to guarantee recoverable safety at all times. The proposed approach enables a permissive yet formal plug-and-play protocol, allowing agents to join or leave the network while ensuring safety recovery if a change in network topology requires temporarily unsafe behavior. We validate the formulation through simulations and real-time experiments of a miniature race-car platoon.
>
---
## 更新

#### [replaced 001] LeLaR: The First In-Orbit Demonstration of an AI-Based Satellite Attitude Controller
- **分类: cs.RO; cs.AI; cs.LG; eess.SY**

- **简介: 论文介绍了一种基于AI的卫星姿态控制器，用于解决传统控制器设计复杂、对模型不确定性强的问题。该控制器在仿真中训练并成功部署到卫星上，验证了其在轨稳定性与性能。**

- **链接: [https://arxiv.org/pdf/2512.19576](https://arxiv.org/pdf/2512.19576)**

> **作者:** Kirill Djebko; Tom Baumann; Erik Dilger; Frank Puppe; Sergio Montenegro
>
> **备注:** Accepted for publication in IEEE Access (DOI: https://doi.org/10.1109/ACCESS.2026.3678816). This is the author's version which has not been fully edited and content may change prior to final publication. 20 pages, 15 figures, 18 tables. The maneuver telemetry datasets are available in the GitHub repository under this https URL
>
> **摘要:** Attitude control is essential for many satellite missions. Classical controllers, however, are time-consuming to design and sensitive to model uncertainties and variations in operational boundary conditions. Deep Reinforcement Learning (DRL) offers a promising alternative by learning adaptive control strategies through autonomous interaction with a simulation environment. Overcoming the Sim2Real gap, which involves deploying an agent trained in simulation onto the real physical satellite, remains a significant challenge. In this work, we present the first successful in-orbit demonstration of an AI-based attitude controller for inertial pointing maneuvers. The controller was trained entirely in simulation and deployed to the InnoCube 3U nanosatellite, which was developed by the Julius-Maximilians-Universität Würzburg in cooperation with the Technische Universität Berlin, and launched in January 2025. We present the AI agent design, the methodology of the training procedure, the discrepancies between the simulation and the observed behavior of the real satellite, and a comparison of the AI-based attitude controller with the classical PD controller of InnoCube. Steady-state metrics confirm the robust performance of the AI-based controller during repeated in-orbit maneuvers.
>
---
#### [replaced 002] Beyond Hard Constraints: Budget-Conditioned Reachability For Safe Offline Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于安全离线强化学习任务，解决如何在满足安全约束下最大化奖励的问题。提出一种预算条件可达集，实现安全策略学习。**

- **链接: [https://arxiv.org/pdf/2603.22292](https://arxiv.org/pdf/2603.22292)**

> **作者:** Janaka Chathuranga Brahmanage; Akshat Kumar
>
> **备注:** Accepted to the 36th International Conference on Automated Planning and Scheduling (ICAPS 2026)
>
> **摘要:** Sequential decision making using Markov Decision Process underpins many realworld applications. Both model-based and model free methods have achieved strong results in these settings. However, real-world tasks must balance reward maximization with safety constraints, often conflicting objectives, that can lead to unstable min/max, adversarial optimization. A promising alternative is safety reachability analysis, which precomputes a forward-invariant safe state, action set, ensuring that an agent starting inside this set remains safe indefinitely. Yet, most reachability based methods address only hard safety constraints, and little work extends reachability to cumulative cost constraints. To address this, first, we define a safetyconditioned reachability set that decouples reward maximization from cumulative safety cost constraints. Second, we show how this set enforces safety constraints without unstable min/max or Lagrangian optimization, yielding a novel offline safe RL algorithm that learns a safe policy from a fixed dataset without environment interaction. Finally, experiments on standard offline safe RL benchmarks, and a real world maritime navigation task demonstrate that our method matches or outperforms state of the art baselines while maintaining safety.
>
---
#### [replaced 003] AGILE: Hand-Object Interaction Reconstruction from Video via Agentic Generation
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文提出AGILE框架，解决单目视频中手物交互重建问题，通过生成式方法和鲁棒跟踪策略，提升几何精度与物理合理性。**

- **链接: [https://arxiv.org/pdf/2602.04672](https://arxiv.org/pdf/2602.04672)**

> **作者:** Jin-Chuan Shi; Binhong Ye; Tao Liu; Xiaoyang Liu; Yangjinhui Xu; Junzhe He; Zeju Li; Hao Chen; Chunhua Shen
>
> **备注:** 11 pages
>
> **摘要:** Reconstructing dynamic hand-object interactions from monocular videos is critical for dexterous manipulation data collection and creating realistic digital twins for robotics and VR. However, current methods face two prohibitive barriers: (1) reliance on neural rendering often yields fragmented, non-simulation-ready geometries under heavy occlusion, and (2) dependence on brittle Structure-from-Motion (SfM) initialization leads to frequent failures on in-the-wild footage. To overcome these limitations, we introduce AGILE, a robust framework that shifts the paradigm from reconstruction to agentic generation for interaction learning. First, we employ an agentic pipeline where a Vision-Language Model (VLM) guides a generative model to synthesize a complete, watertight object mesh with high-fidelity texture, independent of video occlusions. Second, bypassing fragile SfM entirely, we propose a robust anchor-and-track strategy. We initialize the object pose at a single interaction onset frame using a foundation model and propagate it temporally by leveraging the strong visual similarity between our generated asset and video observations. Finally, a contact-aware optimization integrates semantic, geometric, and interaction stability constraints to enforce physical plausibility. Extensive experiments on HO3D, DexYCB, and in-the-wild videos reveal that AGILE outperforms baselines in global geometric accuracy while demonstrating exceptional robustness on challenging sequences where prior art frequently collapses. By prioritizing physical validity, our method produces simulation-ready assets validated via real-to-sim retargeting for robotic applications.
>
---
#### [replaced 004] MSG: Multi-Stream Generative Policies for Sample-Efficient Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出MSG框架，解决机器人操作中生成策略样本效率低的问题。通过组合多个对象中心策略，提升泛化能力和效率，减少演示次数并提高性能。**

- **链接: [https://arxiv.org/pdf/2509.24956](https://arxiv.org/pdf/2509.24956)**

> **作者:** Jan Ole von Hartz; Lukas Schweizer; Joschka Boedecker; Abhinav Valada
>
> **摘要:** Generative robot policies such as Flow Matching offer flexible, multi-modal policy learning but are sample-inefficient. Although object-centric policies improve sample efficiency, it does not resolve this limitation. In this work, we propose Multi-Stream Generative Policy (MSG), an inference-time composition framework that trains multiple object-centric policies and combines them at inference to improve generalization and sample efficiency. MSG is model-agnostic and inference-only, hence widely applicable to various generative policies and training paradigms. We perform extensive experiments both in simulation and on a real robot, demonstrating that our approach learns high-quality generative policies from as few as five demonstrations, resulting in a 95% reduction in demonstrations, and improves policy performance by 89 percent compared to single-stream approaches. Furthermore, we present comprehensive ablation studies on various composition strategies and provide practical recommendations for deployment. Finally, MSG enables zero-shot object instance transfer. We make our code publicly available at this https URL.
>
---
#### [replaced 005] Towards High-Consistency Embodied World Model with Multi-View Trajectory Videos
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人视觉控制任务，旨在解决低级动作与实际物理交互不一致的问题。通过多视角轨迹视频实现高一致性世界模型，提升运动精度与物体交互准确性。**

- **链接: [https://arxiv.org/pdf/2511.12882](https://arxiv.org/pdf/2511.12882)**

> **作者:** Taiyi Su; Jian Zhu; Yaxuan Li; Chong Ma; Jianjun Zhang; Zitai Huang; Hanli Wang; Yi Xu
>
> **备注:** 12 pages, 5 figures
>
> **摘要:** Embodied world models aim to predict and interact with the physical world through visual observations and actions. However, existing models struggle to accurately translate low-level actions (e.g., joint positions) into precise robotic movements in predicted frames, leading to inconsistencies with real-world physical interactions. To address these limitations, we propose MTV-World, an embodied world model that introduces Multi-view Trajectory-Video control for precise visuomotor prediction. Specifically, instead of directly using low-level actions for control, we employ trajectory videos obtained through camera intrinsic and extrinsic parameters and Cartesian-space transformation as control signals. However, projecting 3D raw actions onto 2D images inevitably causes a loss of spatial information, making a single view insufficient for accurate interaction modeling. To overcome this, we introduce a multi-view framework that compensates for spatial information loss and ensures high-consistency with physical world. MTV-World forecasts future frames based on multi-view trajectory videos as input and conditioning on an initial frame per view. Furthermore, to systematically evaluate both robotic motion precision and object interaction accuracy, we develop an auto-evaluation pipeline leveraging multimodal large models and referring video object segmentation models. To measure spatial consistency, we formulate it as an object location matching problem and adopt the Jaccard Index as the evaluation metric. Extensive experiments demonstrate that MTV-World achieves precise control execution and accurate physical interaction modeling in complex dual-arm scenarios.
>
---
#### [replaced 006] Interactive Force-Impedance Control
- **分类: cs.RO**

- **简介: 该论文属于人机协作控制任务，旨在解决机器人在复杂接触环境中的安全交互问题。提出一种统一的力-阻抗控制框架，确保系统稳定性与安全性。**

- **链接: [https://arxiv.org/pdf/2510.17341](https://arxiv.org/pdf/2510.17341)**

> **作者:** Fan Shao; Satoshi Endo; Sandra Hirche; Fanny Ficuciello
>
> **摘要:** Human collaboration with robots requires flexible role adaptation, enabling the robot to switch between an active leader and a passive follower. Effective role switching depends on accurately estimating human intentions, which is typically achieved through external force analysis, nominal robot dynamics, or data-driven approaches. However, these methods are primarily effective in contact-sparse environments. When robots under hybrid or unified force-impedance control physically interact with active humans or non-passive environments, the robotic system may lose passivity and thus compromise safety. To address this challenge, this paper proposes a unified Interactive Force-Impedance Control (IFIC) framework that adapts to interaction power flow, ensuring safe and effortless interaction in contact-rich environments. The proposed control architecture is formulated within a port-Hamiltonian framework, incorporating both interaction and task control ports, thereby guaranteeing autonomous system passivity. Experiments in both rigid and soft contact scenarios demonstrate that IFIC ensures stable collaboration under active human interaction, reduces contact impact forces and interaction force oscillations.
>
---
#### [replaced 007] Heracles: Bridging Precise Tracking and Generative Synthesis for General Humanoid Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，旨在解决 humanoid 在复杂环境中的适应性问题。提出 Heracles 框架，结合精确跟踪与生成合成，提升控制鲁棒性与自然性。**

- **链接: [https://arxiv.org/pdf/2603.27756](https://arxiv.org/pdf/2603.27756)**

> **作者:** Zelin Tao; Zeran Su; Peiran Liu; Jingkai Sun; Wenqiang Que; Jiahao Ma; Jialin Yu; Jiahang Cao; Pihai Sun; Hao Liang; Gang Han; Wen Zhao; Zhiyuan Xu; Jian Tang; Qiang Zhang; Yijie Guo
>
> **备注:** 26 pages, 7 figures, 6 tables
>
> **摘要:** Achieving general-purpose humanoid control requires a delicate balance between the precise execution of commanded motions and the flexible, anthropomorphic adaptability needed to recover from unpredictable environmental perturbations. Current general controllers predominantly formulate motion control as a rigid reference-tracking problem. While effective in nominal conditions, these trackers often exhibit brittle, non-anthropomorphic failure modes under severe disturbances, lacking the generative adaptability inherent to human motor control. To overcome this limitation, we propose Heracles, a novel state-conditioned diffusion middleware that bridges precise motion tracking and generative synthesis. Rather than relying on rigid tracking paradigms or complex explicit mode-switching, Heracles operates as an intermediary layer between high-level reference motions and low-level physics trackers. By conditioning on the robot's real-time state, the diffusion model implicitly adapts its behavior: it approximates an identity map when the state closely aligns with the reference, preserving zero-shot tracking fidelity. Conversely, when encountering significant state deviations, it seamlessly transitions into a generative synthesizer to produce natural, anthropomorphic recovery trajectories. Our framework demonstrates that integrating generative priors into the control loop not only significantly enhances robustness against extreme perturbations but also elevates humanoid control from a rigid tracking paradigm to an open-ended, generative general-purpose architecture.
>
---
#### [replaced 008] SIMPACT: Simulation-Enabled Action Planning using Vision-Language Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SIMPACT框架，解决机器人操作中物理理解不足的问题。通过模拟增强视觉语言模型，实现物理推理与动作规划，提升操作任务性能。**

- **链接: [https://arxiv.org/pdf/2512.05955](https://arxiv.org/pdf/2512.05955)**

> **作者:** Haowen Liu; Shaoxiong Yao; Haonan Chen; Jiawei Gao; Jiayuan Mao; Jia-Bin Huang; Yilun Du
>
> **备注:** Accepted to CVPR 2026; camera-ready version
>
> **摘要:** Vision-Language Models (VLMs) exhibit remarkable common-sense and semantic reasoning capabilities. However, they lack a grounded understanding of physical dynamics. This limitation arises from training VLMs on static internet-scale visual-language data that contain no causal interactions or action-conditioned changes. Consequently, it remains challenging to leverage VLMs for fine-grained robotic manipulation tasks that require physical understanding, reasoning, and corresponding action planning. To overcome this, we present SIMPACT, a test-time, SIMulation-enabled ACTion Planning framework that equips VLMs with physical reasoning through simulation-in-the-loop world modeling, without requiring any additional training. From a single RGB-D observation, SIMPACT efficiently constructs physics simulations, enabling the VLM to propose informed actions, observe simulated rollouts, and iteratively refine its reasoning. By integrating language reasoning with physics prediction, our simulation-enabled VLM can understand contact dynamics and action outcomes in a physically grounded way. Our method demonstrates state-of-the-art performance on five challenging, real-world rigid-body and deformable manipulation tasks that require fine-grained physical reasoning, outperforming existing general-purpose robotic manipulation models. Our results demonstrate that embedding physics understanding via efficient simulation into VLM reasoning at test time offers a promising path towards generalizable embodied intelligence. Project webpage can be found at this https URL
>
---
#### [replaced 009] Scaling Cross-Environment Failure Reasoning Data for Vision-Language Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决失败检测泛化不足的问题。通过生成跨环境的失败数据，构建FailCoT数据集，并训练Guardian模型提升失败检测性能。**

- **链接: [https://arxiv.org/pdf/2512.01946](https://arxiv.org/pdf/2512.01946)**

> **作者:** Paul Pacaud; Ricardo Garcia; Shizhe Chen; Cordelia Schmid
>
> **备注:** Code, Data, and Models available at this https URL. The paper contains 8 pages, 7 figures, 7 tables
>
> **摘要:** Robust robotic manipulation requires reliable failure detection and recovery. Although recent Vision-Language Models (VLMs) show promise in robot failure detection, their generalization is severely limited by the scarcity and narrow coverage of failure data. To address this bottleneck, we propose an automatic framework for generating diverse robotic planning and execution failures across both simulated and real-world environments. Our approach perturbs successful manipulation trajectories to synthesize failures that reflect realistic failure distributions, and leverages VLMs to produce structured step-by-step reasoning traces. This yields FailCoT, a large-scale failure reasoning dataset built upon the RLBench simulator and the BridgeDataV2 real-robot dataset. Using FailCoT, we train Guardian, a multi-view reasoning VLM for unified planning and execution verification. Guardian achieves state-of-the-art performance on three unseen real-world benchmarks: RoboFail, RoboVQA, and our newly introduced UR5-Fail. When integrated with a state-of-the-art LLM-based manipulation policy, it consistently boosts task success rates in both simulation and real-world deployment. These results demonstrate that scaling high-quality failure reasoning data is critical for improving generalization in robotic failure detection. Code, Data, and Models available at this https URL.
>
---
#### [replaced 010] Context-Triggered Contingency Games for Strategic Multi-Agent Interaction
- **分类: cs.RO**

- **简介: 该论文属于多智能体协作任务，旨在解决自主系统中长期战略与短期适应的平衡问题。提出上下文触发的应急博弈框架，实现安全高效的多智能体交互。**

- **链接: [https://arxiv.org/pdf/2512.03639](https://arxiv.org/pdf/2512.03639)**

> **作者:** Kilian Schweppe; Anne-Kathrin Schmuck
>
> **摘要:** We address the challenge of reliable and efficient interaction in autonomous multi-agent systems, where agents must balance long-term strategic objectives with short-term dynamic adaptation. We propose context-triggered contingency games, a novel integration of strategic games derived from temporal logic specifications with dynamic contingency games solved in real time. Our two-layered architecture leverages strategy templates to guarantee satisfaction of high-level objectives, while a new factor-graph-based solver enables scalable, real-time model predictive control of dynamic interactions. The resulting framework ensures both safety and progress in uncertain, interactive environments. We validate our approach through simulations and hardware experiments in autonomous driving and robotic navigation, demonstrating efficient, reliable, and adaptive multi-agent interaction.
>
---
#### [replaced 011] A Novel Camera-to-Robot Calibration Method for Vision-Based Floor Measurements
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人视觉标定任务，旨在解决机器人与相机的协同测量问题。通过设计参考板融合激光跟踪与相机成像，实现高精度的机器人-相机标定。**

- **链接: [https://arxiv.org/pdf/2603.15126](https://arxiv.org/pdf/2603.15126)**

> **作者:** Jan Andre Rudolph; Dennis Haitz; Markus Ulrich
>
> **备注:** 8 pages; accepted for publication in the ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences
>
> **摘要:** A novel hand-eye calibration method for ground-observing mobile robots is proposed. While cameras on mobile robots are common, they are rarely used for ground-observing measurement tasks. Laser trackers are increasingly used in robotics for precise localization. A referencing plate is designed to combine the two measurement modalities of laser-tracker 3D metrology and camera-based 2D imaging. It incorporates reflector nests for pose acquisition using a laser tracker and a camera calibration target that is observed by the robot-mounted camera. The procedure comprises estimating the plate pose, the plate-camera pose, and the robot pose, followed by computing the robot-camera transformation. Experiments indicate sub-millimeter repeatability.
>
---
#### [replaced 012] TRANS: Terrain-aware Reinforcement Learning for Agile Navigation of Quadruped Robots under Social Interactions
- **分类: cs.RO**

- **简介: 该论文提出TRANS框架，解决四足机器人在复杂地形和社交环境中的敏捷导航问题。通过两阶段DRL方法，实现地形感知与社会互动导航。**

- **链接: [https://arxiv.org/pdf/2602.12724](https://arxiv.org/pdf/2602.12724)**

> **作者:** Wei Zhu; Irfan Tito Kurniawan; Ye Zhao; Mitsuhiro Hayashibe
>
> **摘要:** This study introduces TRANS: Terrain-aware Reinforcement learning for Agile Navigation under Social interactions, a deep reinforcement learning (DRL) framework for quadrupedal social navigation over unstructured terrains. Conventional quadrupedal navigation typically separates motion planning from locomotion control, neglecting whole-body constraints and terrain awareness. On the other hand, end-to-end methods are more integrated but require high-frequency sensing, which is often noisy and computationally costly. In addition, most existing approaches assume static environments, limiting their use in human-populated settings. To address these limitations, we propose a two-stage training framework with three DRL pipelines. (1) TRANS-Loco employs an asymmetric actor-critic (AC) model for quadrupedal locomotion, enabling traversal of uneven terrains without explicit terrain or contact observations. (2) TRANS-Nav applies a symmetric AC framework for social navigation, directly mapping transformed LiDAR data to ego-agent actions under differential-drive kinematics. (3) A unified pipeline, TRANS, integrates TRANS-Loco and TRANS-Nav, supporting terrain-aware quadrupedal navigation in uneven and socially interactive environments. Comprehensive benchmarks against locomotion and social navigation baselines demonstrate the effectiveness of TRANS. Hardware experiments further confirm its potential for sim-to-real transfer.
>
---
#### [replaced 013] DCReg: Decoupled Characterization for Efficient Degenerate LiDAR Registration
- **分类: cs.RO**

- **简介: 该论文属于LiDAR点云配准任务，解决几何退化环境下的注册不稳定问题。提出DCReg方法，通过分解Hessian矩阵、特征分析和结构化预处理，提升配准精度与速度。**

- **链接: [https://arxiv.org/pdf/2509.06285](https://arxiv.org/pdf/2509.06285)**

> **作者:** Xiangcheng Hu; Xieyuanli Chen; Mingkai Jia; Jin Wu; Ping Tan; Steven L. Waslander
>
> **备注:** 27 pages, 19 figures, 9 tables
>
> **摘要:** LiDAR point cloud registration is fundamental to robotic perception and navigation. In geometrically degenerate environments (e.g., corridors), registration becomes ill-conditioned: certain motion directions are weakly constrained, causing unstable solutions and degraded accuracy. Existing detect-then-mitigate methods fail to reliably detect, physically interpret, and stabilize this ill-conditioning without corrupting the optimization. We introduce DCReg (Decoupled Characterization for Ill-conditioned Registration), establishing a detect-characterize-mitigate paradigm that systematically addresses ill-conditioned registration via three innovations. First, DCReg achieves reliable ill-conditioning detection by employing Schur complement decomposition on the Hessian matrix. This decouples the 6-DoF registration into 3-DoF clean rotational and translational subspaces, eliminating coupling effects that mask degeneracy in full-Hessian analyses. Second, within these subspaces, we develop interpretable characterization techniques resolving eigen-basis ambiguities via basis alignment. This establishes stable mappings between eigenspaces and physical motion directions, providing actionable insights on which motions lack constraints and to what extent. Third, leveraging this spectral information, we design a targeted mitigation via a structured preconditioner. Guided by MAP regularization, we implement eigenvalue clamping exclusively within the preconditioner rather than modifying the original problem. This preserves the least-squares objective and minimizer, enabling efficient optimization via Preconditioned Conjugate Gradient with a single interpretable parameter. Experiments demonstrate DCReg achieves 20-50% higher long-duration localization accuracy and 5-30x speedups (up to 116x) over degeneracy-aware baselines across diverse environments. Code: this https URL
>
---
#### [replaced 014] Bridging the Basilisk Astrodynamics Framework with ROS 2 for Modular Spacecraft Simulation and Hardware Integration
- **分类: cs.RO**

- **简介: 该论文属于航天与机器人融合任务，旨在解决 spacecraft 模拟与 ROS 2 集成难题。构建了轻量级通信桥梁，实现双向实时数据交换，支持仿真与硬件的无缝过渡。**

- **链接: [https://arxiv.org/pdf/2512.09833](https://arxiv.org/pdf/2512.09833)**

> **作者:** Elias Krantz; Ngai Nam Chan; Gunnar Tibert; Huina Mao; Christer Fuglesang
>
> **备注:** Presented at the International Conference on Space Robotics (iSpaRo) 2025
>
> **摘要:** Integrating high-fidelity spacecraft simulators with modular robotics frameworks remains a challenge for autonomy development. This paper presents a lightweight, open-source communication bridge between the Basilisk astrodynamics simulator and the Robot Operating System 2 (ROS 2), enabling real-time, bidirectional data exchange for spacecraft control. The bridge requires no changes to Basilisk's core and integrates seamlessly with ROS 2 nodes. We demonstrate its use in a leader-follower formation flying scenario using nonlinear model predictive control, deployed identically in both simulation and on the ATMOS planar microgravity testbed. This setup supports rapid development, hardware-in-the-loop testing, and seamless transition from simulation to hardware. The bridge offers a flexible and scalable platform for modular spacecraft autonomy and reproducible research workflows.
>
---
#### [replaced 015] DFM-VLA: Iterative Action Refinement for Robot Manipulation via Discrete Flow Matching
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决动作解码中早期错误无法修正的问题。提出DFM-VLA框架，通过离散流匹配实现动作迭代优化，提升操作性能。**

- **链接: [https://arxiv.org/pdf/2603.26320](https://arxiv.org/pdf/2603.26320)**

> **作者:** Jiayi Chen; Wenxuan Song; Shuai Chen; Jingbo Wang; Zhijun Li; Haoang Li
>
> **摘要:** Vision--Language--Action (VLA) models that encode actions using a discrete tokenization scheme are increasingly adopted for robotic manipulation, but existing decoding paradigms remain fundamentally limited. Whether actions are decoded sequentially by autoregressive VLAs or in parallel by discrete diffusion VLAs, once a token is generated, it is typically fixed and cannot be revised in subsequent iterations, so early token errors cannot be effectively corrected later. We propose DFM-VLA, a discrete flow matching VLA for iterative refinement of action tokens. DFM-VLA~models a token-level probability velocity field that dynamically updates the full action sequence across refinement iterations. We investigate two ways to construct the velocity field: an auxiliary velocity-head formulation and an action-embedding-guided formulation. Our framework further adopts a two-stage decoding strategy with an iterative refinement stage followed by deterministic validation for stable convergence. Extensive experiments on CALVIN, LIBERO, and real-world manipulation tasks show that DFM-VLA consistently outperforms strong autoregressive, discrete diffusion, and continuous diffusion baselines in manipulation performance while retaining high inference efficiency. In particular, DFM-VLA achieves an average success length of 4.44 on CALVIN and an average success rate of 95.7\% on LIBERO, highlighting the value of action refinement via discrete flow matching for robotic manipulation. Our project is available this https URL
>
---
#### [replaced 016] "You've got a friend in me": Co-Designing a Peer Social Robot for Young Newcomers' Language and Cultural Learning
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于教育技术任务，旨在解决社区语言学习中师资不足的问题。通过设计一款协作型社交机器人，支持儿童英语和文化学习。**

- **链接: [https://arxiv.org/pdf/2603.18804](https://arxiv.org/pdf/2603.18804)**

> **作者:** Neil Fernandes; Cheng Tang; Tehniyat Shahbaz; Alex Hauschildt; Emily Davies-Robinson; Yue Hu; Kerstin Dautenhahn
>
> **摘要:** Community literacy programs supporting young newcomer children in Canada face limited staffing and scarce one-to-one time, which constrains personalized English and cultural learning support. This paper reports on a co-design study with United for Literacy tutors that informed Maple, a table-top, peer-like Socially Assistive Robot (SAR) designed as a practice partner within tutor-mediated sessions. From shadowing and co-design interviews, we derived newcomer-specific requirements and added them in an integrated prototype that uses short story-based activities, multi-modal scaffolding and embedded quizzes that support attention while producing tutor-actionable formative signals. We contribute system design implications for tutor-in-the-loop SARs supporting language socialization in community settings and outline directions for child-centered evaluation in authentic programs.
>
---
#### [replaced 017] Masked IRL: LLM-Guided Reward Disambiguation from Demonstrations and Language
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人学习任务，旨在解决从示范和语言中准确学习奖励函数的问题。通过结合示范与语言指令，提出Masked IRL方法提升模型的泛化与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2511.14565](https://arxiv.org/pdf/2511.14565)**

> **作者:** Minyoung Hwang; Alexandra Forsey-Smerek; Nathaniel Dennler; Andreea Bobu
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** Robots can adapt to user preferences by learning reward functions from demonstrations, but with limited data, reward models often overfit to spurious correlations and fail to generalize. This happens because demonstrations show robots how to do a task but not what matters for that task, causing the model to focus on irrelevant state details. Natural language can more directly specify what the robot should focus on, and, in principle, disambiguate between many reward functions consistent with the demonstrations. However, existing language-conditioned reward learning methods typically treat instructions as simple conditioning signals, without fully exploiting their potential to resolve ambiguity. Moreover, real instructions are often ambiguous themselves, so naive conditioning is unreliable. Our key insight is that these two input types carry complementary information: demonstrations show how to act, while language specifies what is important. We propose Masked Inverse Reinforcement Learning (Masked IRL), a framework that uses large language models (LLMs) to combine the strengths of both input types. Masked IRL infers state-relevance masks from language instructions and enforces invariance to irrelevant state components. When instructions are ambiguous, it uses LLM reasoning to clarify them in the context of the demonstrations. In simulation and on a real robot, Masked IRL outperforms prior language-conditioned IRL methods by up to 15% while using up to 4.7 times less data, demonstrating improved sample-efficiency, generalization, and robustness to ambiguous language. Project page: this https URL and Code: this https URL
>
---
#### [replaced 018] Real-Time Operator Takeover for Visuomotor Diffusion Policy Training
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出RTOT框架，解决机器人视觉运动控制中的实时干预问题，通过操作员介入提升策略性能。**

- **链接: [https://arxiv.org/pdf/2502.02308](https://arxiv.org/pdf/2502.02308)**

> **作者:** Marco Moletta; Michael C. Welle; Nils Ingelhag; Jesper Munkeby; Danica Kragic
>
> **摘要:** We present a Real-Time Operator Takeover (RTOT) paradigm that enables operators to seamlessly take control of a live visuomotor diffusion policy, guiding the system back to desirable states or providing targeted corrective demonstrations. Within this framework, the operator can intervene to correct the robot's motion, after which control is smoothly returned to the policy until further intervention is needed. We evaluate the takeover framework on three tasks spanning rigid, deformable, and granular objects, and show that incorporating targeted takeover demonstrations significantly improves policy performance compared with training on an equivalent number of initial demonstrations alone. Additionally, we provide an in-depth analysis of the Mahalanobis distance as a signal for automatically identifying undesirable or out-of-distribution states during execution. Supporting materials, including videos of the initial and takeover demonstrations and all experiments, are available on the project website: this https URL
>
---
#### [replaced 019] Stein-based Optimization of Sampling Distributions in Model Predictive Path Integral Control
- **分类: cs.RO**

- **简介: 该论文属于控制领域，解决MPPI采样分布优化问题。通过引入SVGD方法，动态更新采样分布，提升控制性能。**

- **链接: [https://arxiv.org/pdf/2511.02015](https://arxiv.org/pdf/2511.02015)**

> **作者:** Jace Aldrich; Odest Chadwicke Jenkins
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** This paper introduces a method for Model Predictive Path Integral (MPPI) control that optimizes sample generation towards an optimal trajectory through Stein Variational Gradient Descent (SVGD). MPPI relies upon predictive rollout of trajectories sampled from a distribution of possible actions. Traditionally, these action distributions are assumed to be unimodal and represented as Gaussian. The result can lead suboptimal rollout predictions due to sample deprivation and, in the case of differentiable simulation, sensitivity to noise in the cost gradients. Through introducing SVGD updates in between MPPI environment steps, we present Stein-Optimized Path-Integral Inference (SOPPI), an MPPI/SVGD algorithm that can dynamically update noise distributions at runtime to better capture action sampling distributions without an excessive increase in computational requirements. We demonstrate the efficacy of SOPPI through experiments on a planar cart-pole, 7-DOF robot arm, and a planar bipedal walker. These results indicate improved system performance compared to state-of-the-art MPPI algorithms across a range of hyper-parameters and demonstrate feasibility at lower particle counts.
>
---
#### [replaced 020] Zero-Shot Coordination in Ad Hoc Teams with Generalized Policy Improvement and Difference Rewards
- **分类: cs.MA; cs.AI; cs.RO**

- **简介: 该论文研究多智能体系统中的零样本协作问题，旨在让智能体在未见过的团队中有效协作。通过引入广义策略改进和差异奖励，提升跨团队的知识迁移效果。**

- **链接: [https://arxiv.org/pdf/2510.16187](https://arxiv.org/pdf/2510.16187)**

> **作者:** Rupal Nigam; Niket Parikh; Hamid Osooli; Mikihisa Yuasa; Jacob Heglund; Huy T. Tran
>
> **备注:** 10 pages, 8 figures. To appear in proceedings of 25th International Conference on Autonomous Agents and Multiagent Systems (AAMAS 2026)
>
> **摘要:** Real-world multi-agent systems may require ad hoc teaming, where an agent must coordinate with other previously unseen teammates to solve a task in a zero-shot manner. Prior work often either selects a pretrained policy based on an inferred model of the new teammates or pretrains a single policy that is robust to potential teammates. Instead, we propose to leverage all pretrained policies in a zero-shot transfer setting. We formalize this problem as an ad hoc multi-agent Markov decision process and present a solution that uses two key ideas, generalized policy improvement and difference rewards, for efficient and effective knowledge transfer between different teams. We empirically demonstrate that our algorithm, Generalized Policy improvement for Ad hoc Teaming (GPAT), successfully enables zero-shot transfer to new teams in three simulated environments: cooperative foraging, predator-prey, and Overcooked. We also demonstrate our algorithm in a real-world multi-robot setting.
>
---
#### [replaced 021] TUGS: Physics-based Compact Representation of Underwater Scenes by Tensorized Gaussian
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出TUGS，解决水下3D场景重建问题，通过物理建模和张量化策略提升渲染效率与质量。**

- **链接: [https://arxiv.org/pdf/2505.08811](https://arxiv.org/pdf/2505.08811)**

> **作者:** Shijie Lian; Ziyi Zhang; Hua Li; Laurence Tianruo Yang; Mengyu Ren; Debin Liu; Wenhui Wu
>
> **摘要:** Underwater 3D scene reconstruction is crucial for multimedia applications in adverse environments, such as underwater robotic perception and navigation. However, the complexity of interactions between light propagation, water medium, and object surfaces poses significant difficulties for existing methods in accurately simulating their interplay. Additionally, expensive training and rendering costs limit their practical application. Therefore, we propose Tensorized Underwater Gaussian Splatting (TUGS), a compact underwater 3D representation based on physical modeling of complex underwater light fields. TUGS includes a physics-based underwater Adaptive Medium Estimation (AME) module, enabling accurate simulation of both light attenuation and backscatter effects in underwater environments, and introduces Tensorized Densification Strategies (TDS) to efficiently refine the tensorized representation during optimization. TUGS is able to render high-quality underwater images with faster rendering speeds and less memory usage. Extensive experiments on real-world underwater datasets have demonstrated that TUGS can efficiently achieve superior reconstruction quality using a limited number of parameters. The code is available at this https URL
>
---
#### [replaced 022] UniLGL: Learning Uniform Place Recognition for FOV-limited/Panoramic LiDAR Global Localization
- **分类: cs.RO**

- **简介: 该论文提出UniLGL方法，解决LiDAR全局定位中的均匀性问题，通过多BEV融合网络实现空间、材质和传感器类型的统一特征提取。**

- **链接: [https://arxiv.org/pdf/2507.12194](https://arxiv.org/pdf/2507.12194)**

> **作者:** Hongming Shen; Xun Chen; Yulin Hui; Zhenyu Wu; Wei Wang; Qiyang Lyu; Tianchen Deng; Danwei Wang
>
> **摘要:** Existing LGL methods typically consider only partial information (e.g., geometric features) from LiDAR observations or are designed for homogeneous LiDAR sensors, overlooking the uniformity in LGL. In this work, a uniform LGL method is proposed, termed UniLGL, which simultaneously achieves spatial and material uniformity, as well as sensor-type uniformity. The key idea of the proposed method is to encode the complete point cloud, which contains both geometric and material information, into a pair of BEV images (i.e., a spatial BEV image and an intensity BEV image). An end-to-end multi-BEV fusion network is designed to extract uniform features, equipping UniLGL with spatial and material uniformity. To ensure robust LGL across heterogeneous LiDAR sensors, a viewpoint invariance hypothesis is introduced, which replaces the conventional translation equivariance assumption commonly used in existing LPR networks and supervises UniLGL to achieve sensor-type uniformity in both global descriptors and local feature representations. Finally, based on the mapping between local features on the 2D BEV image and the point cloud, a robust global pose estimator is derived that determines the global minimum of the global pose on SE(3) without requiring additional registration. To validate the effectiveness of the proposed uniform LGL, extensive benchmarks are conducted in real-world environments, and the results show that the proposed UniLGL is demonstratively competitive compared to other State-of-the-Art LGL methods. Furthermore, UniLGL has been deployed on diverse platforms, including full-size trucks and agile Micro Aerial Vehicles (MAVs), to enable high-precision localization and mapping as well as multi-MAV collaborative exploration in port and forest environments, demonstrating the applicability of UniLGL in industrial and field scenarios.
>
---
#### [replaced 023] IndoorR2X: Indoor Robot-to-Everything Coordination with LLM-Driven Planning
- **分类: cs.RO; cs.MA**

- **简介: 该论文提出IndoorR2X，解决多机器人与室内物联网设备的协同问题，通过LLM驱动的规划提升场景理解与任务效率。**

- **链接: [https://arxiv.org/pdf/2603.20182](https://arxiv.org/pdf/2603.20182)**

> **作者:** Fan Yang; Soumya Teotia; Shaunak A. Mehta; Prajit KrisshnaKumar; Quanting Xie; Jun Liu; Yueqi Song; Wenkai Li; Atsunori Moteki; Kanji Uchino; Yonatan Bisk
>
> **摘要:** Although robot-to-robot (R2R) communication improves indoor scene understanding beyond what a single robot can achieve, R2R alone cannot overcome partial observability without substantial exploration overhead or scaling team size. In contrast, many indoor environments already include low-cost Internet of Things (IoT) sensors (e.g., cameras) that provide persistent, building-wide context beyond onboard perception. We therefore introduce IndoorR2X, the first benchmark and simulation framework for Large Language Model (LLM)-driven multi-robot task planning with Robot-to-Everything (R2X) perception and communication in indoor environments. IndoorR2X integrates observations from mobile robots and static IoT devices to construct a global semantic state that supports scalable scene understanding, reduces redundant exploration, and enables high-level coordination through LLM-based planning. IndoorR2X provides configurable simulation environments, sensor layouts, robot teams, and task suites to systematically evaluate high-level semantic coordination strategies. Extensive experiments across diverse settings demonstrate that IoT-augmented world modeling improves multi-robot efficiency and reliability, and we highlight key insights and failure modes for advancing LLM-based collaboration between robot teams and indoor IoT sensors. See our project website: this https URL.
>
---
#### [replaced 024] Detection of Adversarial Attacks in Robotic Perception
- **分类: cs.CV; cs.AI; cs.CR; cs.RO**

- **简介: 论文属于机器人感知中的安全任务，旨在检测针对语义分割的对抗攻击。研究提出专用检测策略以提升系统安全性。**

- **链接: [https://arxiv.org/pdf/2603.28594](https://arxiv.org/pdf/2603.28594)**

> **作者:** Ziad Sharawy; Mohammad Nakshbandi; Sorin Mihai Grigorescu
>
> **备注:** 9 pages, 6 figures. Accepted and presented at STE 2025, Transilvania University of Brasov, Romania
>
> **摘要:** Deep Neural Networks (DNNs) achieve strong performance in semantic segmentation for robotic perception but remain vulnerable to adversarial attacks, threatening safety-critical applications. While robustness has been studied for image classification, semantic segmentation in robotic contexts requires specialized architectures and detection strategies.
>
---
#### [replaced 025] RAD-LAD: Rule and Language Grounded Autonomous Driving in Real-Time
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出LAD和RAD两种方法，解决自动驾驶中的实时决策与规划问题，结合规则与语言模型提升系统可靠性与适应性。**

- **链接: [https://arxiv.org/pdf/2603.28522](https://arxiv.org/pdf/2603.28522)**

> **作者:** Anurag Ghosh; Srinivasa Narasimhan; Manmohan Chandraker; Francesco Pittaluga
>
> **摘要:** We present LAD, a real-time language--action planner with an interruptible architecture that produces a motion plan in a single forward pass (~20 Hz) or generates textual reasoning alongside a motion plan (~10 Hz). LAD is fast enough for real-time closed-loop deployment, achieving ~3x lower latency than prior driving language models while setting a new learning-based state of the art on nuPlan Test14-Hard and InterPlan. We also introduce RAD, a rule-based planner designed to address structural limitations of PDM-Closed. RAD achieves state-of-the-art performance among rule-based planners on nuPlan Test14-Hard and InterPlan. Finally, we show that combining RAD and LAD enables hybrid planning that captures the strengths of both approaches. This hybrid system demonstrates that rules and learning provide complementary capabilities: rules support reliable maneuvering, while language enables adaptive and explainable decision-making.
>
---
#### [replaced 026] VLA Models Are More Generalizable Than You Think: Revisiting Physical and Spatial Modeling
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于视觉-语言-动作模型任务，解决视角和视觉扰动下的泛化问题。通过轻量级适配方法提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.02902](https://arxiv.org/pdf/2512.02902)**

> **作者:** Weiqi Li; Quande Zhang; Ruifeng Zhai; Liang Lin; Guangrun Wang
>
> **摘要:** Vision-language-action (VLA) models achieve strong in-distribution performance but degrade sharply under novel camera viewpoints and visual perturbations. We show that this brittleness primarily arises from misalignment in Spatial Modeling, rather than Physical Modeling. To address this, we propose a one-shot adaptation framework that recalibrates visual representations through lightweight, learnable updates. Our first method, Feature Token Modulation (FTM), applies a global affine transformation to visual tokens and improves Libero viewpoint accuracy from 48.5% to 87.1% with only 4K parameters. Building on this, Feature Linear Adaptation (FLA) introduces low-rank updates to the ViT encoder, achieving 90.8% success with 4.7M parameters -- matching LoRA-scale finetuning at far lower cost. Together, these results reveal substantial untapped robustness in pretrained VLA models and demonstrate that targeted, minimal visual adaptation is sufficient to restore viewpoint generalization.
>
---
#### [replaced 027] Generation of Indoor Open Street Maps for Robot Navigation from CAD Files
- **分类: cs.RO**

- **简介: 该论文属于室内地图生成任务，旨在解决传统SLAM方法在时间、人力和鲁棒性上的不足。通过CAD文件自动生成结构化、语义化的OpenStreetMap，提升机器人导航的可靠性与效率。**

- **链接: [https://arxiv.org/pdf/2507.00552](https://arxiv.org/pdf/2507.00552)**

> **作者:** Jiajie Zhang; Shenrui Wu; Xu Ma; Sören Schwertfeger
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** The deployment of autonomous mobile robots is predicated on the availability of environmental maps, yet conventional generation via SLAM (Simultaneous Localization and Mapping) suffers from significant limitations in time, labor, and robustness, particularly in dynamic, large-scale indoor environments where map obsolescence can lead to critical localization failures. To address these challenges, this paper presents a complete and automated system for converting architectural Computer-Aided Design (CAD) files into a hierarchical topometric OpenStreetMap (OSM) representation, tailored for robust life-long robot navigation. Our core methodology involves a multi-stage pipeline that first isolates key structural layers from the raw CAD data and then employs an AreaGraph-based topological segmentation to partition the building layout into a hierarchical graph of navigable spaces. This process yields a comprehensive and semantically rich map, further enhanced by automatically associating textual labels from the CAD source and cohesively merging multiple building floors into a unified, topologically-correct model. By leveraging the permanent structural information inherent in CAD files, our system circumvents the inefficiencies and fragility of SLAM, offering a practical and scalable solution for deploying robots in complex indoor spaces. The software is encapsulated within an intuitive Graphical User Interface (GUI) to facilitate practical use. The code and dataset are available at this https URL.
>
---
