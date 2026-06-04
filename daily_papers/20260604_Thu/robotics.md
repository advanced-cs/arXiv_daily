# 机器人 cs.RO

- **最新发布 46 篇**

- **更新 28 篇**

## 最新发布

#### [new 001] CoRe-MoE: Contrastive Reweighted Mixture of Experts for Multi-Terrain Humanoid Locomotion with Gait Adaptation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人形机器人运动控制任务，解决多地形下步态适应与平滑过渡问题。提出CoRe-MoE框架，通过两阶段强化学习实现稳定步态与地形适应的分离与融合。**

- **链接: [https://arxiv.org/pdf/2606.04718](https://arxiv.org/pdf/2606.04718)**

> **作者:** Kailun Huang; Zikang Xie; Yanzhe Xie; Panpan Liao; Fanghai Zhang; Yanheng Mai; Wenhao Xu; Yunheng Wang; Renjing Xu; Haohui Huang
>
> **备注:** Kailun Huang, Zikang Xie, Yanzhe Xie and Panpan Liao contributed equally to this work. Corresponding authors: Renjing Xu and Haohui Huang
>
> **摘要:** Humans primarily rely on walking and running to traverse complex terrains, without resorting to unnecessarily complex motion patterns. Similarly, humanoid robots should achieve smooth transitions between walking and running while maintaining natural and stable locomotion. However, unifying gait transition and multi-terrain adaptation within a single policy remains challenging due to gradient interference and the distribution shift induced by terrain-dependent visual and dynamic variations. Although Mixture-of-Experts (MoE) architectures can alleviate multi-skill interference, naive joint training often fails to yield clear expert specialization, limiting their effectiveness. To address these challenges, we propose CoRe-MoE, a two-stage reinforcement learning framework that decouples gait generation from terrain adaptation. In the first stage, a stable locomotion policy is learned to produce natural walking and running behaviors with smooth transitions. In the second stage, a terrain-aware MoE branch is introduced and trained with a contrastive objective to shape the gating network, enabling it to capture structured terrain representations and promote expert specialization. The final action is obtained via weighted fusion of the base gait policy and the terrain-aware branch, allowing the policy to preserve stable locomotion patterns while adapting to complex terrains. Extensive simulation results demonstrate that the proposed method outperforms baseline approaches in terms of success rate, locomotion stability, and multi-terrain adaptability. Furthermore, zero-shot deployment on a Unitree G1 humanoid robot validates the effectiveness of our framework, achieving robust walking and running across stairs, slopes, steps, obstacles, and unstructured outdoor terrains, while maintaining accurate foothold placement and dynamic stability under external disturbances.
>
---
#### [new 002] MAD: Mapping-Aware World Models for Agile Quadrotor Flight
- **分类: cs.RO**

- **简介: 该论文提出MAD模型，用于视觉导航的四旋翼飞行器，解决复杂环境中路径规划与避障问题。通过学习隐空间动态，结合占据网格和本体感知状态，提升飞行效率与任务迁移能力。**

- **链接: [https://arxiv.org/pdf/2606.04534](https://arxiv.org/pdf/2606.04534)**

> **作者:** Xinhong Zhang; Runqing Wang; Yunfan Ren; Ding Yu; Boyu Zhou; Jian Sun; Fang Deng; Jie Chen; Gang Wang
>
> **备注:** 12 pages, 14 figures
>
> **摘要:** Agile quadrotor flight in cluttered scenes requires more than a reactive mapping from a depth image to a control command: the vehicle must remember which regions have been observed, infer nearby occupied space, and act under partial visibility and tight latency. In this paper, we present Mapping-Aware Dreamer (MAD), a geometry-aware world model for vision-based quadrotor flight. Instead of using raw-image reconstruction as the main self-supervised objective, MAD learns recurrent latent dynamics that reconstruct robocentric occupancy and visibility grid maps together with proprioceptive states. This design forces the latent state to encode local geometry, visibility history, and ego-motion in a form that is directly relevant to collision avoidance. MAD is trained in DiffAero using a GPU-parallel map-construction module that provides high-throughput supervision for occupancy and visibility. The learned representation is used in three policy-learning modes: imagination-based MAD-Dreamer and feature-extractor variants based on PPO and SHAC. Across visual navigation and racing tasks, MAD-based agents achieve higher success rates, faster flight, and better cross-task transfer than corresponding vision-only baselines. The model also produces interpretable map predictions and accurate ego-motion estimates from depth observations. We further deploy the learned policy on a physical quadrotor with an Intel RealSense D435i and demonstrate safe indoor and outdoor flight under limited sensing, reaching 9.66 m/s in simulation and 5.05 m/s in real-world forest experiments. These results show that mapping-aware world models provide a practical middle ground between modular aerial navigation and end-to-end learning.
>
---
#### [new 003] Multi-Agent Next-Best-View Optimization for Risk-Averse Planning
- **分类: cs.RO**

- **简介: 该论文属于多智能体路径规划任务，解决不确定环境中安全路径规划问题。提出一种分布式框架，通过协作最大化信息增益并降低通信开销。**

- **链接: [https://arxiv.org/pdf/2606.04158](https://arxiv.org/pdf/2606.04158)**

> **作者:** Amirhossein Mollaei Khass; Vivek Pandey; Guangyi Liu; Athanasios Cosse; Emrah Bayrak; Nader Motee
>
> **备注:** 8 pages, 5 figures. Submitted to IROS 2026
>
> **摘要:** Multi-agent Next-Best-View (NBV) selection for safe path planning in uncertain and unknown environments requires informative, safety-aware, and efficient coordination. Centralized approaches rely on sharing raw sensor data or significant communication overhead, resulting in limited scalability. We propose a distributed, risk-aware multi-agent NBV framework in which each robot maintains a private local 3D Gaussian Splatting map and the team jointly maximizes expected information gain (EIG) restricted to masked zones along planned trajectories. The resulting distributed objective is solved by Consensus ADMM (C-ADMM) over a communication graph, with each robot exchanging only candidate viewpoints, planned trajectory descriptors, and scalar EIG contributions. Collision risk along each trajectory is modeled via Average Value-at-Risk (AV@R) over the local 3DGS map and used both to shape the masking radius and to score planned paths. Experiments in Gibson environments at multiple team sizes show that the distributed formulation approaches the centralized baseline in mapping quality and trajectory safety while reducing communication by orders of magnitude.
>
---
#### [new 004] MineXplore: An Open-Source Reinforcement Learning Exploration Benchmark for GNSS-Denied Underground Environment
- **分类: cs.RO**

- **简介: 该论文提出MineXplore，一个用于GNSS拒收地下环境的强化学习探索基准，解决自主导航难题。通过高精度模拟地下隧道结构，支持稳定策略学习。**

- **链接: [https://arxiv.org/pdf/2606.04569](https://arxiv.org/pdf/2606.04569)**

> **作者:** Abhishek S; Badrikanath Praharaj; Sreeram MV
>
> **备注:** 7 pages,11 figures, Submitted to the workshop Xplore:Cross-Disciplinary aspects of Exploration in Robotics, Reinforcement Learning and Search Held at International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Underground mines present extreme conditions for autonomous robot navigation: GPS is denied, lighting is degraded, and tunnel topology is loop-rich and non-convex. Simulation benchmarks grounded in real production-mine geometry and compatible with GPU-accelerated learning pipelines do not yet exist in the open-source ecosystem. We present MineXplore, an open-source MuJoCo-based navigation benchmark derived from the Leung et al. 2017 Chilean underground copper mine dataset. The environment reconstructs a 104,423 sq.m tunnel network through an six-stage contour-to-MJCF pipeline incorporating octagonal wall cross-sections, LiDAR-sourced jagged wall geometry, three terrain friction zones, a global 5 degree incline, and periodic spot lighting. Geometric fidelity is validated at an Intersection over Union (IoU) of 0.9538 against the source survey map, and surface texture similarity scores 79.4% across six structural dimensions. A single-agent PPO baseline trained via RLlib across five independent random seeds achieves a best rolling coverage of 88.89% (3 of 5 seeds reaching the 90% coverage target), confirming that MineXplore supports stable and reproducible policy learning under realistic underground sensing and topology.
>
---
#### [new 005] X4Val: Learning Neural Surrogates for Variance-Reduced Policy Evaluation
- **分类: cs.RO**

- **简介: 该论文提出X4Val框架，解决机器人系统评估中因真实数据稀缺导致的方差问题。通过融合多源异构数据，提升评估效率与准确性。**

- **链接: [https://arxiv.org/pdf/2606.05159](https://arxiv.org/pdf/2606.05159)**

> **作者:** Rachel Luo; Michael Watson; Apoorva Sharma; Heng Yang; Han Qi; Edward Schmerling; Sushant Veer; Boris Ivanovic; Marco Pavone
>
> **摘要:** Rigorous evaluation of learning-based robotic systems is an essential prerequisite for deployment. However, real-world test data is expensive to gather; moreover, in a typical iterative development context, data gathered from the latest policy is necessarily limited in scale. This motivates evaluation methodologies that make use of heterogeneous data sources, including simulation, historical policy logs, and data collected from related platforms or environments. While such auxiliary data are abundant and inexpensive, they are generally not directly representative of real-world outcomes -- for example, performance in simulation may differ substantially from performance in the real world -- making their principled use for high-confidence performance estimation challenging. In this paper, we introduce X4Val, a general framework for variance-reduced real-world metric estimation in the presence of non-paired, multi-domain data. X4Val embeds samples from real and auxiliary domains into a shared representation space and learns a transferable predictor of real-world metrics; this learned predictor is then incorporated into a control-variates estimator, enabling variance reduction even when paired samples are unavailable. We provide theoretical analysis and empirical evaluations on autonomous driving and real-world robot manipulation tasks, domains across which X4Val achieves up to 38.4% variance reduction and demonstrates consistent improvements over strong baselines. These results show that non-paired, heterogeneous data can be leveraged to substantially improve the sample efficiency of rigorous robotic system validation.
>
---
#### [new 006] CoPark: Learning Reactive Parking via Self-Play
- **分类: cs.RO**

- **简介: 该论文属于自主泊车任务，解决多车辆在精确泊入同时安全交互的问题。提出CoPark方法，通过自博弈强化学习实现精准且反应灵敏的泊车。**

- **链接: [https://arxiv.org/pdf/2606.04149](https://arxiv.org/pdf/2606.04149)**

> **作者:** Jiarong Wei; Yanxing Chen; Sinuo Song; Yin Wu; Anna Rehr; Abhinav Valada
>
> **摘要:** Learning a single policy that reaches a goal with high geometric precision while interacting safely with nearby agents poses conflicting objectives. Precision favors commitment to a fixed geometric plan, whereas interaction requires immediate deviation when another agent intrudes, causing policies optimized for one objective to often fail at the other. We study this problem in the context of reactive autonomous parking, where multiple vehicles must reach assigned slots with sub-meter terminal accuracy while remaining responsive to neighboring vehicles throughout the maneuver. We propose CoPark, a multi-agent self-play RL approach built on a residual-policy architecture. A precomputed offline plan provides a fixed action prior, while a residual head learns the reactive corrections. The residual policy learns behaviors under self-play, where data and scripting fall short, while the fixed prior holds the slot-frame geometry that pure policies struggle to reach reliably. The key design is a partner-threat-modulated, channel-asymmetric release of the prior. A continuous threat signal shifts authority of the longitudinal channel to the residual head to enable yielding, while the lateral channel remains anchored to the precomputed reference to preserve sub-meter slot alignment. A closed-loop refinement layer corrects residual terminal error from action-grid discretization. We train our policy on six parking lots and evaluate zero-shot on our new reactive-parking benchmark spanning Dragon Lake Parking (DLP) and DeepScenario Open 3D (DSC3D). CoPark achieves ~70-85% success with only 3-6% collision rate, substantially outperforming classical, imitation-learning, and large-scale RL baselines. Importantly, the results demonstrate emergent interaction behaviors such as reverse-yielding, mid-maneuver yielding, tight-corridor passing, and queuing.
>
---
#### [new 007] PerceptTwin: Semantic Scene Reconstruction for Iterative LLM Planning and Verification
- **分类: cs.RO; cs.AI**

- **简介: 论文提出PerceptTwin，用于从机器人感知生成交互式模拟，解决传统仿真构建困难的问题。通过语义场景重建，提升规划安全性和可靠性。**

- **链接: [https://arxiv.org/pdf/2606.04226](https://arxiv.org/pdf/2606.04226)**

> **作者:** Charlie Gauthier; Sacha Morin; Liam Paull
>
> **备注:** Accepted at ICRA 2026 (Vienna); published on arxiv for archival purposes. See also this https URL
>
> **摘要:** Simulation environments are useful for both robot policy learning and planning verification and validation. Traditionally, the process of creating a simulation was onerous. Creating a bespoke simulation environment for each individual environment that a robot would operate in was simply infeasible. In this work, we introduce PerceptTwin, a fully automatic pipeline that constructs interactive simulations directly from semantic scene representations produced by a robot's perception stack. PerceptTwin combines open-vocabulary object maps with 3D asset generation, affordance prediction, and commonsense condition checking. These interactive simulations can be used to validate and refine plans before they are executed on the robot hardware. Borrowing from the AI alignment literature, we also introduce an LLM judge that verifies plan correctness and alignment with human preferences. Experiments show that PerceptTwin feedback allows LLM planners to refine plans, enhance safety, and resist harmful black-box prompting attacks. In our suite of tasks, PerceptTwin improves plan success by an average of approximately 39% for GPT5, GPT5Mini, and GPT5Nano planners. Additionally, PerceptTwin also improves human plan verification by up to 18% on average for plans that fail due to unfilled skill preconditions. Our results demonstrate the potential of open-vocabulary scene simulation from robot perception as a foundation for safer, more reliable robot planning.
>
---
#### [new 008] VISTA: Vision-Grounded and Physics-Validated Adaptation of UMI data for VLA Training
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人学习任务，旨在解决UMI数据与VLA模型间的视觉与物理不匹配问题。提出VISTA框架，包含数据集、物理验证和联合训练方法，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2606.04708](https://arxiv.org/pdf/2606.04708)**

> **作者:** Siyuan Yang; Linzheng Guo; Ouyang Lu; Zhaxizhuoma; Daoran Zhang; Xinmiao Wang; Ting Xiao; Fangzheng Yan; Zhijun Chen; Yan Ding; Chao Yu; Chenjia Bai; Xuelong Li
>
> **摘要:** Universal Manipulation Interface (UMI) enables scalable real-world robot data collection without hardware-specific teleoperation, yet leveraging UMI data to train large-scale Vision-Language-Action (VLA) models remains fundamentally challenging. We identify two critical mismatches: wrist-mounted fisheye views, with severe radial distortion and local gripper-centric perspectives, are out-of-distribution for pretrained VLMs; and human-collected trajectories frequently violate kinematic limits, incur collisions, or exceed controller bandwidth, teaching VLA policies physically infeasible actions. To address the challenges, we present VISTA, a framework that bridges this dual gap through three synergistic components. (i)~UMI-VQA, the first large-scale VQA dataset tailored to wrist-mounted fisheye observations, aligns VLM representations to the distorted visual regime via auxiliary vision-language supervision. (ii)~A systematic physical-validation pipeline performs a data-completeness pre-check and scores each valid trajectory for trajectory continuity, self-collision risk, and execution fidelity before it enters training. (iii)~A two-stage co-training recipe jointly learns vision-language grounding on UMI-VQA and action prediction on validated trajectories. Our experiments empirically show that incorporating UMI-VQA consistently improves downstream policy performance, and that physical-validation scores are strongly predictive of deployment success. On diverse simulation and real-world manipulation tasks, VISTA significantly outperforms strong baselines including $\pi_{0.5}$, LingBot-VLA, and Wall-X. We release the physical-validation pipeline, UMI-VQA, validated trajectory data, and the pre-trained model for the community.
>
---
#### [new 009] OSCAR: Omni-Embodiment Skeleton-Conditioned World Action Model for Robotics
- **分类: cs.RO**

- **简介: 该论文提出OSCAR，解决机器人策略评估中的动作跟随和跨形态泛化问题。通过构建标准化数据集并使用骨架条件控制，提升视频世界模型的精度与通用性。**

- **链接: [https://arxiv.org/pdf/2606.04463](https://arxiv.org/pdf/2606.04463)**

> **作者:** Zhuoyuan Wu; Jun Gao
>
> **备注:** Project page: this https URL
>
> **摘要:** We present OSCAR, a precise action-conditioned video world model that generalizes across different robot embodiments and enables robot policy evaluation. Existing video world models face three main challenges for real-world robot evaluation: limited scenario diversity in current robot training datasets, imprecise action following, and poor generalization across embodiments for broad adoption. We tackle these challenges from two perspectives. At its core is a large-scale standardized data pipeline that curates, filters, and deduplicates broad robotics and egocentric human datasets, yielding a clean joint-training dataset that spans diverse tasks, scenarios, actions, and robot embodiments. To condition the video model, we adopt 2D kinematic skeleton rendering as a unified conditioning representation that generalizes across different robot arms or even human hands. We finetune the Cosmos-Predict2.5-2B model on a single GH200 GPU. Our model achieves significant improvement on action following, appearance quality, and motion consistency, compared to existing baselines, which either have a much larger model size or require more GPUs. We further deploy OSCAR to evaluate robot policies from RoboArena. Extensive experiments demonstrate the significant correlation between our virtual policy evaluation in OSCAR and real-world evaluation, paving the way for the future where robot policies can be purely evaluated in virtual generated worlds.
>
---
#### [new 010] Affordance2Action: Task-Conditioned Scene-level Affordance Grounding for Real-Time Manipulation
- **分类: cs.RO**

- **简介: 该论文提出Affordance2Action框架，解决场景级任务条件化功能区域定位问题，通过构建基准和标注流程提升实时操作性能。**

- **链接: [https://arxiv.org/pdf/2606.04172](https://arxiv.org/pdf/2606.04172)**

> **作者:** Litao Liu; Yifan Han; Pengfei Yi; Wenbo Yu; Hanqing Wang; Haoran Du; Enze Yuan; Zilin Yuan; Ruiding Feng; Michael Liu; Qi Zhang; Jingjin Yu
>
> **备注:** 23 pages
>
> **摘要:** Task-conditioned manipulation requires grounding instructions to task-relevant functional parts rather than object categories. This setting is scene-dependent and often one-to-many in cluttered scenes: the same object may afford different interactions across tasks, while a single task may correspond to either one functional region or multiple valid functional regions, depending on the scene layout. Existing affordance datasets and benchmarks remain misaligned with this setting, as they typically focus on grasping or object-level affordances, rely on synthetic scenes, or assume a single instruction-region correspondence. We present Affordance2Action (A2A), a benchmark-centered learning framework for scene-level, task-conditioned part affordance grounding. At its core is A2A-Bench, a manipulation-oriented benchmark that covers both single-region and multi-region instruction correspondences in everyday scenes, with the latter highlighting the ambiguity and diversity of affordance grounding in realistic multi-object environments. To construct it at scale, we build A2A-AffordGen, an agent-assisted annotation pipeline that combines language-model filtering, interactive part segmentation, instance-level mask-out refinement, task-reasoning instruction generation, and human verification. A2A-Bench's supervision further supports diverse downstream applications, with real-time affordance grounding and affordance-conditioned manipulation policies as two representative examples. Experiments show that A2A exposes substantial gaps in generic segmentation, VLM-based grounding, and affordance distillation baselines, while improving task-level localization and providing useful spatial priors for downstream manipulation. All datasets and code will be publicly released to promote open research.
>
---
#### [new 011] Think Fast and Far: Long-Horizon Online POMDP Planning via Rapid State Sampling
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，解决长周期POMDP求解难题。提出ROP-RAS3方法，通过快速状态采样生成宏动作，提升求解效率与成功率。**

- **链接: [https://arxiv.org/pdf/2606.04355](https://arxiv.org/pdf/2606.04355)**

> **作者:** Yuanchu Liang; Edward Kim; J. Arden Knoll; Wil Thomason; Zachary Kingston; Lydia E. Kavraki; Hanna Kurniawati
>
> **备注:** @inproceedings{Liang2026Thinking, title = {Think Fast and Far: Long-Horizon Online POMDP Planning via Rapid State Sampling}, author = {Yuanchu Liang and Edward Kim and this http URL Knoll and Wil Thomason and Zachary Kingston and Lydia E. Kavraki and Hanna Kurniawati}, year = 2026, booktitle = {International Journal of Robotics Research (to appear)} }
>
> **摘要:** Partially Observable Markov Decision Processes (POMDPs) are a general and principled framework for motion planning under uncertainty. Despite tremendous improvement in the scalability of POMDP solvers, long-horizon POMDPs remain difficult to solve. To alleviate the difficulty, this paper proposes a new approximate online POMDP solver, called Reference-Based Online POMDP Planning via Rapid State Space Sampling (ROP-RAS3). ROP-RAS3 uses novel extremely fast sampling-based motion planning techniques to sample the state space and generate a diverse set of macro actions online, which are then used to bias belief-space sampling and infer high-quality policies without requiring exhaustive enumeration of the action space -- a fundamental constraint for modern online POMDP solvers. ROP-RAS3 converges to a near-optimal reference-based solution at a rate that depends on the number of sampled actions, rather than the size of the action space. ROP-RAS3 is evaluated on various long-horizon POMDPs with up to 3000 lookahead steps and 35-dimensional state spaces, where the state, action and observation spaces can be continuous, discrete, or a hybrid of discrete and continuous. Although the reference-based optimal solution may not be the same as the optimal POMDP solution, empirical results indicate that in all of these problems, in terms of success rate, ROP-RAS3 outperforms other state-of-the-art methods by up to multiple folds. We also demonstrate the capability of our approach on a physical robot demonstration. This work extends the theory and empirical results of our ISRR24 paper. Code can be found at \texttt{this https URL}.
>
---
#### [new 012] CADENCE: Predicting Realized MAPF Execution Time Beyond Sum of Costs
- **分类: cs.RO**

- **简介: 该论文属于多智能体路径规划任务，旨在解决MAPF算法评估指标与实际执行时间不匹配的问题。通过分析不同特征对执行时间的预测能力，发现原始运动负担是更可靠的预测信号。**

- **链接: [https://arxiv.org/pdf/2606.04746](https://arxiv.org/pdf/2606.04746)**

> **作者:** Abhishek S; Badrikanath Praharaj; Sreeram MV
>
> **备注:** 7 pages, 4 figures, 3 tables and this paper was accepted at Multi-Agent Robotic Systems: Real-World Collaboration and Interaction a workshop at the international conference of robotics and automation (ICRA 2026)
>
> **摘要:** Multi-Agent Path Finding (MAPF) algorithms are increasingly used to plan motion for robot teams in industrial warehouses and robotic shared workspaces, but standard MAPF algorithm evaluation metrics, such as Sum of Costs (SoC), makespan, and planner runtime, can obscure how planner choices translate into realistic execution performance. We present CADENCE (Coordination and Action-Driven Estimation for Networked Continuous Execution), a hardware study of this evaluation gap on a fixed 7 by 7 workcell with seven differential drive robots, asking which features available before execution can best predict final wall-clock completion time. We compare SoC, total planned travel cost, primitive motion burden (how much basic motion the plan requires, such as makespan, turns, consecutive moves, and start-stop transitions), and interaction aware coordination structure (how much inter-robot coordination the plan induces, such as dependency links, interacting robot pairs, dependency depth, and crowding exposure). To test this, we generate 120 plans across 15 scenarios -- 5 Empty, 5 Medium Random, and 5 Bottleneck and execute each plan four times, yielding a 480 trial hardware corpus. Using both a scenario-held -- out ridge model and a trial-level mixed-effects model, we find that SoC alone is informative but incomplete, while primitive motion burden gives the strongest improvement, reducing held out error by about 48.6%-59.8% in MAE and 44.2%-61.4% in RMSE relative to SoC-only models. Interaction-aware coordination features add smaller, less uniform gains, most clearly in the mixed-effects analysis. Across both models and uncertainty checks, primitive motion burden is the most reliable additional signal beyond SoC, suggesting that much of the execution time gap is already visible in the offline plan before any robot starts moving.
>
---
#### [new 013] Selecting haptic guidance models in teleoperation: guidelines from a comparative user study
- **分类: cs.RO**

- **简介: 该论文属于遥操作中的触觉引导模型选择任务，旨在解决如何根据任务和环境选择最优模型的问题。通过用户研究比较了三种模型，提出评估指标和选择指南。**

- **链接: [https://arxiv.org/pdf/2606.04157](https://arxiv.org/pdf/2606.04157)**

> **作者:** Alexis Boulay; Margot Vulliez; David Daney
>
> **备注:** EUROHAPTICS 2026 - EuroHaptics International Conference, Jul 2026, Sienna, Italy
>
> **摘要:** Haptic guidance in teleoperation enhances operator performance through force feedback. This paper presents guidelines to select the most appropriate model considering the task, the environment and the operator. We define a unified formulation expressing most common models (spring-damper, potential field, and guiding tube) as variations of a stiffness-damping system with model-specific guiding functions. We conducted a user study comparing the three classical models across six scenarios with varying environmental conditions in a vertical farming task. Results show no universally superior model: spring-damper excels in cluttered environments, potential field in free spaces (but it shows risks near obstacles), and guiding tube offers a balanced compromise. We propose novel objective metrics to evaluate the interaction, and show that guiding force magnitude correlates with comfort and trust scores. These findings provide practical model selection guidelines through environmental characteristics and real-time evaluation metrics.
>
---
#### [new 014] What Are We Actually Benchmarking in Robot Manipulation?
- **分类: cs.RO**

- **简介: 该论文属于机器人操作领域，旨在解决基准测试有效性问题。指出现有基准存在四个失效模式，提出诊断方法并评估多个基准，揭示其可能无法真实反映泛化能力。**

- **链接: [https://arxiv.org/pdf/2606.04233](https://arxiv.org/pdf/2606.04233)**

> **作者:** Tianchong Jiang; Xiangshan Tan; Samuel Wheeler; Luzhe Sun; Tewodros W. Ayalew; Matthew Walter
>
> **备注:** 31 pages, 6 figures
>
> **摘要:** A robotics benchmark score measures success under one fixed evaluation setup, yet is routinely treated as evidence of general manipulation capability. We identify four failure modes, each of which weakens or invalidates a benchmark's role as a valid proxy for that capability: shortcut solvability, lack of statistical significance, creeping overfitting, and data-source dependence. We propose one diagnostic per failure mode. We audit LIBERO, CALVIN, SimplerEnv, RoboCasa, and RoboTwin 2.0 under these diagnostics. LIBERO and CALVIN fail multiple diagnostics. RoboCasa and RoboTwin 2.0 fail fewer, despite appearing far less often in recent progress claims. On LIBERO, a 0.09B probe with no language encoder scores at or near reported SOTA, and most reported gains are not provably statistically significant. On CALVIN, randomizing block poses within the training range drops performance for every tested policy. We release the four diagnostics with reference implementations for authors and reviewers to apply before treating a benchmark score as evidence of progress. Code and artifacts are available at this https URL.
>
---
#### [new 015] Instant-Fold: In-Context Imitation Learning for Deformable Object Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出Instant-Fold，用于可变形物体操作的任务，解决如何从单个示范中学习多种操作模式的问题。通过视觉表征和流匹配策略，实现无需梯度更新的模仿学习。**

- **链接: [https://arxiv.org/pdf/2606.04269](https://arxiv.org/pdf/2606.04269)**

> **作者:** Yilong Wang; Cheng Qian; Edward Johns
>
> **摘要:** Deformable object manipulation (DOM) is challenging due to high-dimensional, partially observable states that evolve through long-horizon, topology-changing interactions with multiple valid manipulation modes. We introduce Instant-Fold, an in-context imitation learning framework for DOM. Given a single human demonstration, our policy infers and executes diverse manipulation modes directly from the demonstration, including variations in spatial execution and ordering, without requiring gradient updates. Our approach first learns deformation-aware visual representations via temporal contrastive pretraining, after which a flow-matching transformer policy conditioned on the demonstration predicts actions to execute the intended manipulation mode. Trained entirely in simulation, Instant-Fold generalizes across diverse folding modes and transfers zero-shot to real-world settings without additional data collection or finetuning. Videos are available at this https URL.
>
---
#### [new 016] COP-Q: Safety-First Reinforcement Learning for Robot Control via Cholesky-Ordered Projection
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于安全强化学习任务，旨在解决机器人控制中的安全约束问题。通过引入协方差建模，提升样本效率与安全性。**

- **链接: [https://arxiv.org/pdf/2606.04749](https://arxiv.org/pdf/2606.04749)**

> **作者:** Guopeng Li; Moritz A. Zanger; Matthijs T. J. Spaan; Julian F. P. Kooij
>
> **备注:** 7 pages, 6 figures, 2 tables
>
> **摘要:** Safe robot control requires maximizing return while satisfying safety constraints. In off-policy safe reinforcement learning, reward and safety Q-values are commonly learned by separate critic ensembles, with uncertainty handled independently for each objective. This objective-wise treatment neglects inter-objective correlation and can lead to overly conservative value estimates, thereby reducing sample efficiency. To address this issue, we propose Cholesky-Ordered Projection Q-learning (COP-Q), a safety-first method that incorporates inter-objective covariance into vector-valued Q-value estimation. COP-Q constructs a generalized confidence bound in the joint Q-value space and uses Cholesky factorization to encode objective priority in a sequential form. This preserves conservatism on safety while adaptively reducing excessive conservatism on the reward objective. The resulting estimate is used in both temporal-difference target computation and actor optimization. COP-Q incurs minimal computational overhead and is readily compatible with most existing deep Q-learning frameworks. Experiments on robot locomotion in Brax and safe navigation in Safety-Gymnasium, covering both hard- and soft-safety settings, demonstrate that COP-Q achieves strong safety performance together with competitive or improved sample efficiency relative to representative baselines.
>
---
#### [new 017] HapTile: A Haptic-Informed Vision-Tactile-Language-Action Dataset for Contact-Rich Imitation Learning
- **分类: cs.RO**

- **简介: 该论文提出HapTile数据集，用于接触丰富的模仿学习任务。解决现有数据集缺乏触觉与语言联合信息的问题，通过整合触觉反馈和语言指令，提升操作稳定性与任务多样性。**

- **链接: [https://arxiv.org/pdf/2606.04825](https://arxiv.org/pdf/2606.04825)**

> **作者:** Amirhosein Alian; Yongqiang Zhao; Shiyi Gu; Xuyang Zhang; Zhuo Chen; Christopher E. Mower; Haitham Bou-Ammar; Shan Luo
>
> **摘要:** Despite the importance of tactile sensing for reliable manipulation, most existing Vision-Language-Action (VLA) datasets remain vision-only, and those that do incorporate tactile information typically lack the joint combination of task diversity, language conditioning, and action trajectories. Furthermore, existing teleoperation pipelines rarely provide haptic feedback to the operator, despite its established role in demonstration quality and manipulation stability. In this work, we present HapTile, a contact-grounded visuotactile manipulation dataset that advances beyond vision-only trajectory datasets by embedding physical interaction sensing at two levels: fingertip tactile feedback at the robot end-effector, and haptic-informed demonstrations at the teleoperator side. The data collection platform integrates haptic feedback directly into the teleoperation controller, enabling the operator to perceive contact interactions in real time. It is built around a standard and reproducible robotic system equipped with custom-designed fingertip tactile sensors. The dataset comprises everyday manipulation tasks spanning a broad range of contact-rich skills, including pick-and-place, folding, pressing, stacking, and other routine activities. Each task is paired with language instructions that condition the policy on the manipulation objective, together with synchronized visuotactile observations and action trajectories. In addition, we provide a benchmarking study on contact-rich policy learning using two baseline models to evaluate the effectiveness of the proposed contact-grounded dataset. The dataset and additional details are available on our website: this http URL.
>
---
#### [new 018] Distribution-Free Risk-Aware Planning and Control Under Uncertainty Using Conformal Spectral Risk Control
- **分类: cs.RO**

- **简介: 该论文属于安全控制任务，解决不确定环境下风险感知的规划与控制问题。提出一种无需假设不确定性分布的RA-MPC框架，通过预测集确保风险控制在阈值内。**

- **链接: [https://arxiv.org/pdf/2606.04185](https://arxiv.org/pdf/2606.04185)**

> **作者:** Junsik Eom; Tulga Ersal
>
> **备注:** Submitted to IEEE Robotics and Automation Letters
>
> **摘要:** Safe navigation in dynamic and uncertain environments often relies on accurate estimation of, or assumptions about, the true underlying uncertainty. However, accurately characterizing the true uncertainty distribution is often difficult due to limited data or imperfect information. An incorrect understanding of the uncertainty and its associated risk may lead to dangerous decisions even under high levels of risk aversion. To address this issue, we propose a risk-aware model predictive control (RA-MPC) framework that incorporates prediction sets to guarantee risk control below a user-specified threshold without requiring assumptions about the underlying uncertainty distribution. To generate the prediction sets, we develop a distribution-free risk quantification framework that extends conformal risk control (CRC) to general spectral risk measures. We then show that incorporating the prediction sets into the MPC framework provides statistical safety guarantees in terms of spectral risk constraint satisfaction even under uncertainty misspecification. We validate the proposed framework in simulated vehicle obstacle avoidance scenarios, demonstrating improved safety and reduced solve time compared to a baseline RA-MPC framework.
>
---
#### [new 019] TransTac: Visuo-Tactile Modality Transition via Ultraviolet-Encoded Transparent Elastomers
- **分类: cs.RO**

- **简介: 该论文提出TransTac，一种透明触觉传感器，解决视觉与触觉信息融合难题，通过UV编码实现高精度触觉重建与视觉感知。**

- **链接: [https://arxiv.org/pdf/2606.04477](https://arxiv.org/pdf/2606.04477)**

> **作者:** Lingyue Yang; Bin Fang
>
> **备注:** Accepted at IEEE International Conference on Robotics and Automation (ICRA) 2026. 8 pages, 7 figures
>
> **摘要:** Vision-based tactile sensors (VBTS) recover high-resolution contact geometry but typically rely on opaque elastomer layers that prevent visual transparency, while RGB-D cameras provide global depth perception yet degrade significantly at close range. To address this limitation, we present TransTac, a transparent ultraviolet (UV)-encoded binocular VBTS that integrates visual observation and marker-based tactile reconstruction within a single compact device. The system employs a transparent elastomer embedded with UV-reflective markers and a prior-guided Delaunay stereo matching algorithm for robust sparse triangulation. To reliably detect densely distributed semitransparent markers, we develop a lightweight detector that enables stable localization under contact and deformation. The proposed prior-guided Delaunay matching improves correspondence robustness by approximately 21% compared with global assignment baselines while maintaining high reconstruction accuracy. In semantic evaluation, TransTac achieves up to 83.3% zero-shot recognition accuracy on tactile images, exceeding opaque tactile baselines by approximately 50 percentage points. Embedding analysis further reveals substantially stronger cross-modal alignment with natural images, with class-center similarity increasing from around 0.2 to over 0.77. Controlled near-distance experiments quantify the degradation of RGB-D depth reliability and demonstrate extended geometric coverage enabled by visuo-tactile integration. Finally, a compact prototype is implemented with an approximate hardware cost of $70.
>
---
#### [new 020] GRAIL: Generating Humanoid Loco-Manipulation from 3D Assets and Video Priors
- **分类: cs.RO**

- **简介: 该论文提出GRAIL系统，用于生成类人机器人操作与运动数据。解决传统方法依赖物理环境的问题，通过虚拟生成实现高效数据合成与任务训练。**

- **链接: [https://arxiv.org/pdf/2606.05160](https://arxiv.org/pdf/2606.05160)**

> **作者:** Tianyi Xie; Haotian Zhang; Jinhyung Park; Zi Wang; Bowen Wen; Jiefeng Li; Xueting Li; Qingwei Ben; Haoyang Weng; Yufei Ye; David Minor; Tingwu Wang; Chenfanfu Jiang; Sanja Fidler; Jan Kautz; Linxi Fan; Yuke Zhu; Zhengyi Luo; Umar Iqbal; Ye Yuan
>
> **备注:** Project page: this https URL
>
> **摘要:** Scaling humanoid loco-manipulation requires robot-compatible demonstrations across diverse objects, whole-body motions, and scene geometries, but teleoperation and motion capture are difficult to scale because each collection depends on physical setups, instrumented actors, and robot operation. We present GRAIL, a digital generation pipeline that remains fully virtual until deployment: it composes 3D assets, simulator-ready scenes, and priors from video foundation models (VFMs) to synthesize interactions without rebuilding physical environments or teleoperating the robot. Rather than reconstructing unconstrained in-the-wild videos, GRAIL starts from fully specified 3D configurations in which object geometry, camera parameters, metric scale, environment depth, and a robot-proportioned character are known before video generation and reused during reconstruction. This privileged setup better conditions 4D recovery, allowing model-based object tracking, human motion estimation, and interaction-aware optimization to reconstruct metric 4D human-object interaction (HOI) trajectories with reduced depth ambiguity and morphology mismatch. We retarget the recovered motions to a humanoid robot and train complementary task-general trackers: an object-aware latent adaptor for manipulation and a scene-aware tracker for terrain traversal. GRAIL produces over 20,000 sequences spanning pick-up, object manipulation, sitting, and terrain traversal. Using only GRAIL-generated data, we train egocentric visual policies through a sim-to-real pipeline and deploy them on a Unitree G1 humanoid, achieving 84\% real-world success on diverse object pick-up and 90\% success on stair-climbing.
>
---
#### [new 021] PointAction: 3D Points as Universal Action Representations for Robot Control
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出PointAction，将3D点云作为机器人控制的通用动作表示，解决RGB视频动作接地不明确的问题，通过4D点动态实现跨任务和机器人的动作迁移。**

- **链接: [https://arxiv.org/pdf/2606.03943](https://arxiv.org/pdf/2606.03943)**

> **作者:** Mutian Tong; Han Jiang; Qiao Feng; Lingjie Liu; Jiatao Gu
>
> **备注:** Project page: this https URL
>
> **摘要:** Video-Action Models (VAMs) leverage the broad visual dynamics captured by pre-trained video diffusion models, offering a promising path toward generalizable robot manipulation. However, RGB-only video rollouts are not directly actionable: they leave metric 3D motion, contact geometry, and fine-grained spatial constraints under-specified, making action grounding ambiguous. Meanwhile, scaling action supervision across diverse tasks and embodiments remains costly. We present PointAction, a framework that bridges video predictions to robot actions through explicit point-based 4D modeling. PointAction fine-tunes a foundation video generation model to jointly predict future RGB frames and dynamic 3D pointmaps, producing temporally consistent 3D motion of task-relevant scene geometry. These point dynamics serve as a structured, embodiment-agnostic action interface, which a diffusion-based action decoder maps to executable robot actions. By using metric 3D point dynamics as the interface between video prediction and control, PointAction reduces the ambiguity of RGB-only action grounding and supports transfer across tasks and embodiments with limited action supervision. Experiments show that PointAction achieves state-of-the-art 4D generation quality on robot scenes, outperforms existing baselines in simulation, and generalizes to two real robot arms unseen during pretraining.
>
---
#### [new 022] DLO-Lab: Benchmarking Deformable Linear Object Manipulations with Differentiable Physics
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决变形线性物体（如绳索）的操控问题。提出一种可微分模拟器和基准测试集，以提升学习与评估操控技能的效果。**

- **链接: [https://arxiv.org/pdf/2606.04206](https://arxiv.org/pdf/2606.04206)**

> **作者:** Junyi Cao; Yian Wang; Ziyan Xiong; Chunru Lin; Zhehuan Chen; Chuang Gan
>
> **备注:** ICML 2026, the project page: this https URL
>
> **摘要:** We address the challenge of enabling robots to manipulate deformable linear objects (DLOs), such as ropes, cables, and rubber bands. Prior work has primarily focused on narrow, task-specific problems, often relying on real-world demonstrations or handcrafted heuristics. Such approaches, however, struggle to scale to the wide variety of materials and tasks encountered in practice, and collecting sufficiently diverse real-world data is often impractical. Additionally, existing simulation environments offer limited support for the broad spectrum of material behaviors necessary for generalizable DLO manipulation. To overcome these limitations, we introduce a differentiable simulator explicitly designed for versatile DLO manipulation. Our simulator models a wide range of material properties-including (in)extensibility, elasticity, bending plasticity, and complex interactions with other objects-providing a robust foundation for learning and evaluating manipulation skills. Building on this simulator, we propose a benchmark suite of representative tasks that highlight the unique challenges of DLO manipulation. The successful execution of these tasks is often hindered by the topological complexity and grasp sensitivity inherent to DLOs. Therefore, we introduce a specialized DLO agent that explicitly manages these challenges by proposing strategic grasping points and decomposing long-horizon tasks to maximize control authority. Finally, we evaluate various policy-learning algorithms using our framework, alongside sim-to-real transfer experiments, demonstrating our platform's potential to advance DLO manipulation.
>
---
#### [new 023] SoftPINCH: EMG-Driven Soft Exoskeleton Assistance for Finger Flexion and Grasping
- **分类: cs.RO**

- **简介: 该论文属于手部辅助控制任务，旨在解决EMG信号不稳定和传统手部外骨骼限制自然运动的问题。提出SoftPINCH系统，结合EMG解码与软体外骨骼，实现高效指屈和抓握辅助。**

- **链接: [https://arxiv.org/pdf/2606.04776](https://arxiv.org/pdf/2606.04776)**

> **作者:** Nicklas Nikolaj Grønvall; Magnus Malthe Sigsgaard Nielsen; Xiaofeng Xiong; Saravana Prashanth Murali Babu
>
> **备注:** Submitted to 18th International Conference on the Simulation of Adaptive Behavior (SAB 2026)
>
> **摘要:** Surface electromyography (sEMG) provides a non-invasive interface for detecting hand-movement intention and controlling wearable assistive devices. However, reliable EMG-driven hand assistance remains challenging because EMG signals are affected by noise, motion artifacts, electrode placement, muscle fatigue, and inter-subject variability. At the same time, many hand exoskeletons remain mechanically restrictive or bulky, limiting comfort and natural hand motion. This work presents SoftPINCH, an EMG-driven soft wearable exoskeleton for thumb-index finger flexion and pinch grasp assistance. The system combines a tendon-driven soft exoskeleton, fingertip magnetic contact sensing, and neural EMG decoding for intention-based assistance. Surface EMG was recorded from forearm muscles during index and thumb movements, and three subject-independent decoding architectures were evaluated: LSTM, CNN+LSTM, and CNN+LSTM with attention. The CNN+LSTM and CNN+LSTM-attention models both achieved 99.4% LOSO test accuracy, outperforming the standalone LSTM, which reached 97.8%. However, the attention mechanism did not provide a significant improvement over CNN+LSTM, indicating that CNN-based feature extraction was sufficient for robust EMG representation. The CNN+LSTM model was therefore selected for real-time deployment due to its high accuracy and lower architectural complexity. Functional evaluation showed that active exoskeleton assistance reduced muscular effort during isolated finger flexion and object grasping. During weighted grasping, assistance reduced muscular effort across all tested loads, with a 92.6% reduction at the highest load. These results demonstrate the potential of SoftPINCH for intuitive, low-effort pinch assistance using real-time EMG-driven soft robotic control.
>
---
#### [new 024] HORIZON: Recoverability-Governed Curriculum for Physical-Domain Scaling
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决物理领域泛化问题。提出HORIZON方法，通过可恢复性引导课程，实现物理域的持续扩展与学习。**

- **链接: [https://arxiv.org/pdf/2606.05143](https://arxiv.org/pdf/2606.05143)**

> **作者:** Chenhao Bai; Liqin Lu; Kaijun Wang; Hui Chen; Jin-Chuan Shi; Yuyang Liu; Hao Chen; Chunhua Shen
>
> **备注:** 16 pages, 9 figures
>
> **摘要:** Scaling robust robot policies requires more than broader randomization, because physical-domain experience must remain organized and learnable throughout training. We study when a policy can benefit from harder physics and identify recoverability as a central constraint in on-policy physical-domain scaling. In on-policy training, new dynamics are useful only insofar as they remain close enough to the current policy to generate corrective on-policy data, rather than collapsing rollouts into unrecoverable failures. Using quadruped locomotion as a physically demanding benchmark for embodied generalization, we introduce HORIZON, a checkpointed frontier curriculum that expands physical domains only within the current policy's recoverable boundary. HORIZON uses rollback and boundary refinement to govern each expansion step, turning fixed randomization into a continual process of physical-domain growth. Experiments reveal three regularities of physical-domain expansion. First, direct domain widening is uneven across physical axes and often unlearnable without staged ordering. Second, domain composition is non-monotonic, and adding more domains beyond a compact core can dilute recoverable joint samples and reduce overall robustness. Third, offline distillation of isolated experts cannot substitute for the joint interaction generated by on-policy curriculum. Together, these results frame physical-domain generalization as a continual growth problem for embodied control, with recoverability as the organizing principle for on-policy expansion.
>
---
#### [new 025] AgenticDiffusion: Agentic Diffusion-based Path Planning for Vision-Based UAV Navigation
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文属于视觉导航任务，解决室内UAV路径规划问题。提出AgenticDiffusion框架，结合多视角和语言引导，提升导航效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2606.04111](https://arxiv.org/pdf/2606.04111)**

> **作者:** Faryal Batool; Muhammad Ahsan Mustafa; Fawad Mehboob; Valerii Serpiva; Dzmitry Tsetserukou
>
> **摘要:** Indoor UAV navigation requires efficient exploration, scene understanding, and reliable trajectory execution under limited field-of-view observations. Existing vision-based navigation frameworks typically rely on single-view observations, limiting their ability to reason about occlusions, target visibility, and global scene structure. In this work, we propose AgenticDiffusion, a multi-view UAV navigation framework that coordinates language-guided reasoning, open-vocabulary target grounding, vision-based diffusion planning, and NMPC within a unified aerial navigation pipeline. Given a natural language instruction and synchronized first-person-view (FPV) and top-view observations, the framework determines the most informative viewpoint for navigation and generates a mission plan prior to trajectory execution. The targets are localized using an open-vocabulary grounding model, after which viewpoint-specific diffusion planners generate navigation trajectories for UAV execution. Using complementary viewpoints, the proposed framework reduces repeated target exploration and improves navigation efficiency in cluttered indoor environments. The framework was validated in four real-world UAV navigation scenarios involving adaptive viewpoint selection, multi-stage mission execution, long-horizon navigation, and safe landing-site selection. The experimental results demonstrated an overall mission success rate of 80% in 40 real-world trials, while the diffusion planners achieved a trajectory generation success rate of 100%.
>
---
#### [new 026] D$^3$-MoE:Dual Disentangled Diffusion Mixture-of-Experts for Style-Controllable End-to-End Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文提出D$^3$-MoE，解决自动驾驶中风格不可控问题，通过双轴解耦生成多风格轨迹并优化物理一致性。**

- **链接: [https://arxiv.org/pdf/2606.04884](https://arxiv.org/pdf/2606.04884)**

> **作者:** Renju Feng; Rukang Wang; Ning Xi; Jianguo Yu; Liping Lu; Pan Zhou; Duanfeng Chu
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Traditional end-to-end autonomous driving frameworks frequently suffer from the "style-averaging" dilemma when trained on high-variance human demonstrations, yielding homogenized, style-uncontrollable, and even kinematically unsafe policies. To overcome this limitation, we present D$^3$-MoE (Dual Disentangled Diffusion Mixture-of-Experts), which disentangles trajectory modeling along two complementary axes. On the behavioral axis, generation is decoupled from selection: a style-conditioned diffusion process synthesizes multi-style candidate trajectories in parallel within a single scene, allowing a downstream module to select the optimal trajectory based on user preference or an evaluation score. On the physical axis, decoupled longitudinal and lateral routers activate their respective experts during inference time, trained without manual labels using self-supervised targets from orthogonal ground-truth kinematics. These activated experts, architected as Diffusion Transformers (DiT) and equipped with style-conditioned AdaLN and asymmetric lateral-fusion cross-attention, independently predict their corresponding physical state before being reassembled into a unified, kinematically coherent trajectory. Extensive evaluations on the challenging NAVSIM benchmark demonstrate that D$^3$-MoE achieves state-of-the-art planning performance, reaching 88.2 PDMS and 84.3 EPDMS by default. Moreover, our Best-of-Three ensemble strategy effectively broadens the multi-modal solution space, raising performance to 91.3 PDMS and 87.5 EPDMS. Both quantitative and qualitative analyses jointly confirm the framework's advantages in planning quality and style controllability.
>
---
#### [new 027] WAM-Nav: Asymmetric Latent World-Action Modeling for Unified Visual Navigation
- **分类: cs.RO**

- **简介: 该论文提出WAM-Nav，用于具身视觉导航任务，解决传统方法缺乏预见性和效率低的问题。通过联合学习动作生成与视觉预测，提升导航鲁棒性与成功率。**

- **链接: [https://arxiv.org/pdf/2606.04907](https://arxiv.org/pdf/2606.04907)**

> **作者:** Ning Yang; Yan Huang; Kaiwen Peng; Ziheng He; Kai Wang; Cui Miao; Kailin Lyu; Guo Li; Xiaofeng Wang; Zheng Zhu; Jing Liu; Nianfeng Liu
>
> **摘要:** Visual navigation requires generating smooth and collision-free trajectories under complex geometric and physical constraints. Existing reactive policies that directly map observations to actions lack anticipatory reasoning, limiting their ability to proactively avoid obstacles. While visual imagination offers predictive foresight, conventional modular approaches separate scene prediction from policy learning, often leading to error accumulation and inefficient inference. To address these limitations, we propose WAM-Nav, a Latent World-Action Model for embodied visual navigation that jointly learns action generation and latent visual foresight, enabling more robust and foresighted navigation decisions without compromising inference efficiency. Specifically, WAM-Nav utilizes a shared Diffusion Transformer for asymmetric joint diffusion to concurrently generate long-horizon actions and short-horizon visual foresight, reducing the inference latency and visual error accumulation inherent in multi-step autoregressive rollouts. To further encourage smooth and consistent trajectory generation, we introduce a dual-stream contextual conditioning mechanism that integrates episode-level ego-motion history with sequential visual observations. Combined with a unified goal alignment module that preserves balanced representations across goal types, WAM-Nav naturally supports Image-Goal, Point-Goal, and No-Goal exploration within a single policy. Extensive experiments on the challenging ClutterScenes and InternScenes benchmarks demonstrate strong generalization of WAM-Nav, particularly on Image-Goal and Point-Goal navigation, where it improves success rates by 15.7% and 3.3%, respectively. Real-world deployment further validates effective zero-shot sim-to-real transfer, achieving an average 85% task success rate across diverse indoor and outdoor environments.
>
---
#### [new 028] Potential-Guided Flow Matching for Vision-Language-Action Policy Improvement
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作策略改进任务，解决模仿学习中高质量样本难以利用的问题。提出ForesightFlow方法，通过自引导流匹配提升策略性能，减少计算量并提高真实世界成功率。**

- **链接: [https://arxiv.org/pdf/2606.04968](https://arxiv.org/pdf/2606.04968)**

> **作者:** Yunpeng Mei; Jiakai He; Hongjie Cao; Chenyu Wang; Xiaowen Zhu; Yihan Zhou; Jiamin Wang; Chenbo Xin; Peng Cheng; Yuxuan Yang; Yijie Wang; Xinhu Zheng; Gao Huang; Jie Chen; Gang Wang
>
> **摘要:** Large vision-language-action (VLA) policies are increasingly trained as conditional generative models over action chunks. Yet deployment produces mixed-quality experience-successful demonstrations, partial completions, recoverable mistakes, and failures-that is difficult to use with standard imitation. Full behavior cloning (BC) imitates failures, filtered BC discards useful sub-trajectories, and offline reinforcement learning adds a large critic. We introduce ForesightFlow, a self-guided flow-matching policy that augments each generated action chunk with a learned success-potential trajectory. The same flow proposes and scores candidate actions, enabling best-of-$K$ inference without an external critic. The key issue is that policy improvement and value calibration require different supervision: advantage weighting should emphasize high-quality actions, but applying the same weights to potential coordinates suppresses failure gradients and creates overconfident scores. We address this with decoupled advantage-weighted flow matching, applying exponentiated advantage weights only to action velocities while training potential velocities uniformly. We further derive a one-step boundary estimator for conditional flow matching, allowing advantage computation with a single stop-gradient forward pass. Across five BEHAVIOR-1K simulation tasks and five real-world bimanual tasks, ForesightFlow improves over imitation baselines, matches the strongest separate-critic baseline in simulation success, improves real-world success, and reduces training compute by $38\%$. Ablations show that decoupling prevents value hallucination, the one-step estimator preserves candidate-ranking fidelity, and self-guided sampling improves long-horizon execution.
>
---
#### [new 029] Generalization of World Models under Environmental Variability for Vision-based Quadrotor Navigation
- **分类: cs.RO**

- **简介: 该论文研究视觉导航中世界模型的泛化能力，解决环境变化下的鲁棒性问题。通过实验分析SSL预训练和RL微调对模型性能的影响，发现潜在空间大小和序列长度是关键因素。**

- **链接: [https://arxiv.org/pdf/2606.05015](https://arxiv.org/pdf/2606.05015)**

> **作者:** Luca Zanatta; Grzegorz Malczyk; Kostas Alexis
>
> **摘要:** World models, learned generative models that predict how an environment evolves, have become a promising tool for sample-efficient robot learning. Yet how robust they are to environmental variability remains poorly understood. To address this, we conduct a systematic study using vision-based quadrotor navigation as a testbed problem, training DreamerV3-based world models under varying levels of environmental randomness and evaluating them across all levels through cross-environment validation, spanning both Self-Supervised Learning (SSL) pretraining and Reinforcement Learning (RL) fine-tuning. We then deploy all world models and associated navigation policies on a real quadrotor in unseen environments, including an open-loop run where the model receives just 2.5s of real sensory input before all sensors are cut off, leaving the system to navigate entirely in imagination over a 12m traverse. Our results show that world model robustness during SSL pretraining is a strong predictor of sim-to-real transfer: every model that generalized well in cross-environment SSL validation deployed successfully in the real world, passing through gaps as narrow as 0.67m, whereas the model that dominated simulation policy evaluation failed on the real platform. We further identify (a) the discrete latent size and (b) the training-sequence length as the dominant factors governing world model quality.
>
---
#### [new 030] BPDA-GMM: Bayesian Probabilistic Data Association via Gaussian Mixture Models for Semantic SLAM
- **分类: cs.RO**

- **简介: 该论文属于语义SLAM任务，解决感知混淆下的数据关联问题。提出BPDA-GMM框架，利用高斯混合模型和贝叶斯方法提升地图构建的准确性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2606.04618](https://arxiv.org/pdf/2606.04618)**

> **作者:** Thanh Nguyen Canh; Haolan Zhang; Xiem HoangVan; Antonio Sgorbissa; Nak Young Chong
>
> **摘要:** Probabilistic data association (PDA) improves semantic SLAM in perceptually aliased scenes, but existing methods often assume a fixed landmark set, recompute association weights as the map grows, or rely on hand-tuned null-hypothesis weights. To address these limitations, we propose \textbf{BPDA-GMM}, an online Bayesian PDA framework for semantic SLAM with a growing object-level map. BPDA-GMM uses a Dirichlet-process prior to induce a Chinese Restaurant Process (CRP) association model, where accumulated evidence favors existing landmarks, and the concentration parameter assigns probability mass to new landmarks. For each semantic detection, plausible candidates are selected by a joint semantic-geometric gate, CRP-weighted association probabilities are computed, and object landmarks are updated as semantic Gaussians in closed form. The resulting landmark set forms a Gaussian mixture model, and its dominant component is passed to the back-end as a max-mixture semantic factor. When association weights are inconclusive, an ambiguity-triggered $\alpha$-divergence tempering step improves discrimination. Finally, a decoupled back-end zeroes the pose Jacobian of semantic factors, allowing noisy detections to refine landmarks without directly perturbing the trajectory. Experiments in simulation and on a real indoor dataset demonstrate improved trajectory accuracy, semantic mapping quality, and robustness to perceptual aliasing and classifier errors over state-of-the-art baselines. Code and video are publicly available at this https URL.
>
---
#### [new 031] Real-World Deployment of a 5G-Connected Edge-Controlled Aerial Robot in Industrial Subterranean Mines
- **分类: cs.RO**

- **简介: 该论文研究5G连接的边缘控制无人机在工业地下矿井中的自主飞行，解决远程安全高效控制问题，通过边缘计算实现实时路径规划与控制。**

- **链接: [https://arxiv.org/pdf/2606.04818](https://arxiv.org/pdf/2606.04818)**

> **作者:** Achilleas Santi Seisa; Emanuele Pagliari; Gerasimos Damigos; Elias Small; George Nikolakopoulos
>
> **备注:** 6 pages, 8 figures, MED 2026
>
> **摘要:** This article presents the first real-world autonomous flight of a 5G-connected aerial robot controlled by an edge-offloaded controller, and aims to bridge the gap between controlled and factual setups. The robot operates within an active industrial subterranean mine, while the high-level controller is deployed in a nearby Kubernetes-based edge cluster. Communication between the robot and the edge is enabled via a 5G New Radio (NR) Standalone (SA) network. The chosen controller is a Model Predictive Controller (MPC), which generates control actions to allow the robot to navigate seamlessly through the mining environment. A human operator selects waypoints for the aerial robot, and the MPC generates smooth, collision-free paths for autonomous executions. The proposed 5G edge-based closed-loop system is evaluated in a real industrial setting and demonstrates the potential of edge-controlled robotic systems toward time-critical, safe and efficient future deployments.
>
---
#### [new 032] Teaching Robots to Say 'I Don't Know' : SENTINEL for Uncertainty-Aware SLAM
- **分类: cs.RO**

- **简介: 该论文属于机器人SLAM任务，解决低精度LiDAR无法诊断测量失败的问题。提出SENTINEL框架，通过几何统计和深度一致性计算扫描可靠性，提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2606.04853](https://arxiv.org/pdf/2606.04853)**

> **作者:** Abhishek S; Badrikanath Praharaj; Sreeram MV
>
> **备注:** 6 pages, 10 figures, 3 tables, This paper was accepted at Uncertainty in Open-World Robotics Workshop in conjunction with Internation conference of robotics and automation (ICRA 2026)
>
> **摘要:** Low-cost 2D LiDARs lack the intensity channel that higher-end sensors use to diagnose measurement failures, yet they are widely used on educational and budget robotics platforms. We present SENTINEL, a training - free, label - free reliability estimation framework that gives range - only LiDAR an effective diagnostic signal. SENTINEL combines geometry-based scan statistics with cross - modal depth consistency between LiDAR and an RGB - D camera to compute a per - scan reliability score between 0 and 1. When the score falls below a threshold, corrupted scans are rejected and the robot falls back to calibrated wheel odometry, preventing silent SLAM corruption. We evaluate SENTINEL on a GEFIER R1 four - wheel skid-steer robot equipped with an RPLidar A2M12 and an Intel RealSense D435i in a 185 cm by 245 cm arena containing controlled transparent and reflective failure elements on a central obstacle. Spatial reliability maps across five surface conditions, including glass, mirror, shiny paper, and a mixed mirror and shiny-paper condition, show clear separation between clean and failure cases, allowing affected regions to be identified as reject or noise. Because these failure modes are absent in simulation, validation is performed entirely on real hardware.
>
---
#### [new 033] Cooperative Circumnavigation for Multiple Unmanned Surface Vehicles Without External Localization
- **分类: cs.RO**

- **简介: 该论文属于多无人水面舰艇协同任务，解决无外部定位下的目标环绕问题。通过感知与控制策略，实现USVs在无外部定位情况下的均匀环绕。**

- **链接: [https://arxiv.org/pdf/2606.04518](https://arxiv.org/pdf/2606.04518)**

> **作者:** Xueming Liu; Lin Li; Xiang Zhou; Tianjiang Hu; Qingrui Zhang
>
> **备注:** 17 pages, 15 figures
>
> **摘要:** This paper proposes a cooperative target circumnavigation framework for multiple unmanned surface vehicles (USVs) operating without external localization. The objective is to maintain a uniform circular formation of a specified radius around a target using only limited onboard sensing. The framework adopts a heterogeneous perception strategy that distinguishes between the asymmetric sensing relationships with the target and among the USVs. Specifically, the USVs obtain relative range and displacement measurements through active perception and inter-vehicle communication, while bearing measurements to a non-cooperative target are acquired via passive sensors. To estimate relative positions--both among USVs and between each USV and the target--we employ a Maximum Correntropy Kalman Filter and a Pseudo-Linear Kalman Filter, respectively. A coupled oscillator-based formation controller is designed to ensure system observability while achieving circumnavigation. Theoretical analysis demonstrates that the controller ensures the relative motions between the USVs, as well as that between each USV and the target, satisfy the persistent excitation condition, thereby guaranteeing observability of the Kalman-based filters. The effectiveness of the proposed approach is validated through numerical simulations.
>
---
#### [new 034] Towards Estimating Normal and Shear Interface Pressures in Prosthetic Sockets via Least Squares and Mechanics Modeling
- **分类: cs.RO**

- **简介: 该论文属于 prosthetic socket 压力估计任务，旨在解决接口压力测量不全与耦合问题，通过模型与实验结合提升压力分析精度。**

- **链接: [https://arxiv.org/pdf/2606.04222](https://arxiv.org/pdf/2606.04222)**

> **作者:** Axel González Cornejo; Tianhao Yu; Chi Hwan Lee; Edgar Bolívar-Nieto
>
> **摘要:** Prosthetic socket fitting remains largely manual and iterative, and objective fit metrics are still limited. Part of the challenge is the lack of long-term real-life pressure data at the residual limb--socket interface. Traditional pressure sensors are prone to drift over time, and capture only normal pressures at sparse locations within the socket, missing a critical component for biomechanical analysis: shear. Although some sensors can report both normal and shear interface stresses, these components are often difficult to decouple because of measurement crosstalk. One potential path forward is to develop models that can augment available measurements. This work introduces a testbed to evaluate model performance under sparse pressure sensing using two complementary validation signals: (i) the global wrench (\ie, total forces and moments expressed in an orthonormal frame) transmitted through the socket, by an artificial residual-limb, and (ii) local interface loads (\ie, decoupled normal and shear pressure components in a right-hand-rule orthogonal frame that lives in each instrumented location) measured by sparse sensing clusters, each composed of four capacitance-sensing channels. Rather than presenting full-field pressure estimates, the focus is on an analysis sequence that quantifies how well candidate mechanical models explain both global and local measurements under controlled conditions. A quasi-static spring--mass contact model is evaluated, and its parameters are identified via a two-stage convex least-squares problem. Validation under static loading shows that estimating constant bias terms reduces steady offsets in the wrench channels and improves agreement with local measurements. A Pareto-front sensitivity analysis further illustrates how the trade-off between global and local objectives changes when bias terms are included.
>
---
#### [new 035] CADET: A Modular Platform for Evaluating Distributed Cooperative Autonomy in Connected Autonomous Vehicles
- **分类: cs.RO; cs.DC; cs.LG; eess.SY**

- **简介: 该论文属于自动驾驶领域，解决分布式协同自主系统的评估问题。提出CADET平台，支持模块化部署与真实环境模拟，提升安全与效率。**

- **链接: [https://arxiv.org/pdf/2606.04072](https://arxiv.org/pdf/2606.04072)**

> **作者:** Pragya Sharma; Brian Wang; Mani Srivastava
>
> **摘要:** Deep learning models are increasingly central to autonomous vehicle (AV) pipelines, yet their integration has traditionally followed a monolithic design where perception, planning, and control execute on a single onboard computer. This design overlooks the emerging paradigm of cooperative autonomy, where vehicles interact with roadside units (RSUs), edge servers, and cloud-hosted intelligence through vehicle-to-everything (V2X) connectivity. Cooperative perception and control improve safety and efficiency, but also introduce systems-level challenges: network latency, compute heterogeneity, and multi-tenant contention, all critically affect real-time decision-making. These challenges are further amplified by the increasing reliance on large foundation models, whose scale necessitates cloud deployment. We present CADET (Cooperative Autonomy through Distributed Experimentation Toolkit), a modular platform for systematic and reproducible evaluation of distributed cooperative autonomy systems under realistic deployment conditions. CADET decouples the AV stack into composable modules that can be flexibly deployed across vehicles, infrastructure, and edge/cloud tiers. The framework integrates state-of-the-art models, incorporates trace-driven network and workload emulation, and provides synchronized model-, system-, and task-level instrumentation. Through V2V and V2I experiments, we show that distributed deployment choices fundamentally shape safety, with V2V intent packets outperforming cloud-based perception and RSU-assisted perception sustaining safety until overloaded by concurrent requests. Although designed for AV pipelines, CADET also supports dataset-driven experimentation, enabling systems and ML researchers to benchmark distributed inference workloads independently of full vehicle simulation. CADET is open source, with code and demo available at this https URL.
>
---
#### [new 036] RSC: Decentralized Rigid Formation Flocking for Large-Scale Swarms via Hybrid Predictive Control and Online Reconfiguration
- **分类: cs.RO**

- **简介: 该论文属于多智能体协同控制任务，解决复杂环境中大规模无人机编队保持与避障问题。提出RSC框架，结合预测控制与在线重配置，提升编队稳定性与灵活性。**

- **链接: [https://arxiv.org/pdf/2606.04248](https://arxiv.org/pdf/2606.04248)**

> **作者:** Ganyu Zou; Linhan Wang; Chen Dai; Siji Chen; Chang-Tien Lu
>
> **备注:** 8 pages, 4 figures, two-column format
>
> **摘要:** Decentralized rigid formation flocking requires a swarm of autonomous agents to maintain a predetermined geometric configuration while moving, relying solely on local sensing and communication. However, existing decentralized control methods struggle to maintain strict inter-agent distance constraints in cluttered environments, often suffering from local minima deadlocks, high frequency control oscillations, or limited flexibility during obstacle navigation, resulting in low success rate. To address these limitations, we propose Rigid Swarm Control (RSC), a decentralized control framework for large-scale rigid formation flocking. To escape local minima via robust long-term planning while ensuring short-term safety, RSC integrates finite-horizon trajectory predictions with a reactive artificial potential field (APF) safety controller within a hybrid architecture. Furthermore, to accelerate formation reassembly after obstacle traversal without interrupting task execution, RSC introduces an online leader-follower reconfiguration mechanism based on stable role exchange. Extensive evaluations in challenging cluttered environments with 25 UAVs demonstrate that RSC reliably unifies rigid formation maintenance, obstacle avoidance, and target tracking. Under strict success criteria - collision-free operation with a maximum relative edge-length error below 10%, RSC achieves an 83% success rate, significantly outperforming existing heuristic and learning-based baselines that fall below 5%.
>
---
#### [new 037] CLAW: Learning Continuous Latent Action World Models via Adversarial Latent Regularization
- **分类: cs.RO**

- **简介: 该论文提出CLAW，用于从无动作视频中学习连续潜在动作世界模型，解决无监督动作表示与环境建模问题，通过对抗正则化和扩散生成实现有效模仿学习与目标规划。**

- **链接: [https://arxiv.org/pdf/2606.04130](https://arxiv.org/pdf/2606.04130)**

> **作者:** Tewodros Ayalew; Matthew Jeung; Samuel Wheeler; Xiao Zhang; Andre de la Cruz Arce; Kaylene Stocking; Michael Maire; Matthew R. Walter
>
> **备注:** 8 pages, 15 pages of supplementary material
>
> **摘要:** We introduce CLAW, a fully end-to-end self-supervised framework for learning a world model jointly with continuous latent action representations directly from action-free videos. Our approach leverages adversarial latent regularization and diffusion-based video generation to capture structured and semantically meaningful action representations while modeling rich, predictive environment dynamics, without relying on any action labels or annotations. By simultaneously training the Latent Action Model and world model, CLAW learns to reason about how inferred actions induce environment transitions from visual observations alone. We show that the resulting latent action world model supports both imitation learning from observation and goal-directed planning. In imitation learning, latent actions extracted from raw videos enable behavior cloning. For planning, CLAW generates sequences of latent actions and maps them to executable actions to reach desired goals. Extensive experiments across diverse tasks and embodiments demonstrate that CLAW produces semantically meaningful latent action representations, supports effective action transfer, and enables planning and imitation from observation, outperforming existing methods.
>
---
#### [new 038] M3imic: Learning a Versatile Whole-Body Controller for Multimodal Motion Mimicking
- **分类: cs.RO**

- **简介: 该论文提出M3imic框架，解决多模态运动模仿问题，统一关节角度、人体姿态和末端位姿，实现通用全身控制。**

- **链接: [https://arxiv.org/pdf/2606.04829](https://arxiv.org/pdf/2606.04829)**

> **作者:** Zuxing Lu; Ziang Zheng; Yao Lyu; Jingyu Liu; Feihong Zhang; Song Lu; Xin Yuan; Changyin Sun; Xingxing Zuo; Shengbo Eben Li
>
> **摘要:** Building a general-purpose whole-body controller is essential for enabling diverse motion capabilities in humanoid robots across a wide range of downstream tasks, including locomotion and loco-manipulation. Different tasks rely on distinct motion reference modalities: locomotion primarily depends on coordinated robot joint trajectories, whereas manipulation requires precise end-effector trajectory tracking. Existing methods often overlook the representational mismatch between dense robot joint angles and sparse end-effector poses. To address this, we propose Multi-Modal Mimic (M3imic), a versatile multi-modal whole-body control framework that unifies heterogeneous motion reference modalities, including robot joint angles, human pose trajectories, and end-effector poses, using modality-specific encoders to map them into a shared latent space. Leveraging large-scale reinforcement learning in the simulator, we train a single policy that achieves sim-to-real transfer across multiple motion reference modalities without modality-specific retraining. Extensive simulation and real-world experiments on the Unitree G1 robot are conducted to evaluate the proposed framework. In simulation, the policy achieves a peak success rate of 98.42\% on an unseen test dataset, demonstrating its exceptional generalization capability. The code is available at this https URL
>
---
#### [new 039] Z-FLoc: Zero-Shot Floorplan Localization via Geometric Primitives
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位任务，解决相机在未知环境中基于平面图的定位问题。通过提取几何基元并匹配平面图，实现无需重新训练的零样本定位。**

- **链接: [https://arxiv.org/pdf/2606.04788](https://arxiv.org/pdf/2606.04788)**

> **作者:** Ayumi Umemura; Toshinori Kuwahara; Marc Pollefeys; Daniel Barath
>
> **摘要:** Visual localization -- estimating a camera pose within a pre-existing map -- is a fundamental problem in computer vision. Floorplans are an attractive map representation: they are readily available for most buildings, compact, and inherently invariant to visual appearance changes. However, bridging the severe domain gap between camera observations and floorplan geometry remains challenging. Existing methods address this gap through data-driven learning, yet they require large-scale training data and environment-specific retraining, limiting their practical deployment. We propose a zero-shot floorplan localization method that generalizes to novel environments without any retraining. Our key insight is that dominant geometric primitives -- lines and circles -- are ubiquitous in human-made environments and provide appearance-invariant structural constraints. We extract these primitives from a bird's-eye-view (BEV) projection of monocular 3D reconstructions and match them to the floorplan via dedicated minimal solvers within a robust estimation framework. Experiments on both simulated and real-world datasets show that our approach outperforms state-of-the-art learning-based methods on unseen environments, while using a single fixed set of hyperparameters across all experiments. The source code will be made publicly available.
>
---
#### [new 040] When Freshness Is Not Enough: Distribution-Aware Age of Information for Networked LQR Control
- **分类: eess.SY; cs.MA; cs.RO; math.DS; math.OC**

- **简介: 该论文属于网络化控制任务，研究如何优化更新系统以提高控制性能。针对传统基于平均AoI的策略不足，提出考虑分布特性的AoI指标，分析其对LQR控制的影响，并通过实验证明平均AoI不足以指导网络设计。**

- **链接: [https://arxiv.org/pdf/2606.04361](https://arxiv.org/pdf/2606.04361)**

> **作者:** Abdullah Y. Etcibasi; C. Emre Koksal; Eylem Ekici
>
> **摘要:** Age of Information (AoI) has become a central metric for the design of wireless update systems, especially in applications where fresh measurements support tracking, estimation, and control. Despite its popularity, the use of mean AoI or peak AoI as a surrogate for closed-loop performance is often motivated by intuition rather than by a control-theoretic derivation. This paper examines whether minimizing the mean AoI is in fact optimal for networked control systems. For scalar linear time-invariant systems with delayed intermittent updates, we show that, under state-independent scheduling policies, the infinite-horizon LQR tracking problem reduces to an optimization over the distribution of inter-scheduling intervals. The resulting objective depends on higher-order statistical moments, and in unstable or correlated regimes on exponential moments, of the inter-scheduling process rather than only on its mean. Consequently, policies with identical mean AoI can induce substantially different tracking costs. We further extend the analysis to disturbances with exponentially decaying autocorrelation and derive equivalent cost formulations that expose the role of the full interval distribution. Finally, we validate the theory using real vehicle trajectories from the NGSIM US-101 dataset. The empirical results match the predicted performance trends, demonstrating that mean AoI alone is insufficient for control-oriented network design.
>
---
#### [new 041] CIPER: A Unified Framework for Cross-view Image-retrieval and Pose-estimation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出CIPER框架，解决跨视角图像检索与姿态估计问题，实现城市级检索与精确3-DoF姿态估计的统一。**

- **链接: [https://arxiv.org/pdf/2606.05011](https://arxiv.org/pdf/2606.05011)**

> **作者:** Yurim Jeon; Dongseong Seo; Seung-Woo Seo
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** Cross-view geo-localization estimates the geographic location of a ground image by matching it against an aerial image database. Existing methods tackle this through either large-scale retrieval or precise pose estimation, but not both: retrieval-based methods enable wide-area search at the cost of localization accuracy, while pose estimation methods achieve high precision within only a narrow search space. Naively cascading these pipelines introduces error propagation and inconsistent feature representations. We formulate cross-view geo-localization as a unified problem requiring simultaneous city-scale retrieval and precise 3-DoF pose estimation. We propose CIPER (Cross-view Image-retrieval and Pose-estimation transformER), a single architecture that jointly performs both tasks through mutually beneficial feature learning. CIPER uses a shared transformer encoder with task-specific tokens to disentangle global retrieval features from spatial localization cues. To bridge the large domain gap between ground and aerial views, we introduce a two-way transformer pose decoder that uses ground features as spatial queries for bidirectional cross-attention. A set prediction strategy further enables stable 3-DoF regression under a unified multi-task objective. Experiments on VIGOR, KITTI, and Ford Multi-AV demonstrate competitive performance, especially under limited field-of-view and arbitrary orientation conditions. Code is available at this https URL.
>
---
#### [new 042] What Can Eye Gaze Teach Us About Real-World Cycling? Insights From the Oxford RobotCycle Project
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互任务，旨在研究骑行者对危险的感知。通过眼动追踪分析不同道路环境下的认知挑战和压力，揭示骑行安全感知差异。**

- **链接: [https://arxiv.org/pdf/2606.04989](https://arxiv.org/pdf/2606.04989)**

> **作者:** Benjamin Hardin; Efimia Panagiotaki; Daniele De Martini; Lars Kunze
>
> **摘要:** Although much is known about the physical danger of cycling situations, less is understood about the perceived danger of cycling. Furthermore, perception of danger may be filtered at a subconscious level and therefore difficult for one to self-report. To this end, these subconscious perceptions can be revealed through physiological metrics such as eye gaze. This paper explores the perceived safety of cycling in Oxford, United Kingdom and explores the ability of wearable eye tracking glasses to produce insights about the differences in perception under different environments and events. This paper finds that eye gaze patterns change between using bike lanes, car lanes and shared bus lanes, representing different cognitive challenges of each lane type. This paper presents that different intersections have significantly different eye gaze patterns which may have implications for cyclist stress. Finally, eye gaze patterns differ in the presence of events such as passes and pedestrians in the road compared to when cycling with no events. This paper draws conclusions on the benefits and limitations of using wearable eye trackers to estimate stress and cyclist workload.
>
---
#### [new 043] Dive into the Scene: Breaking the Perceptual Bottleneck in Vision-Language Decision Making via Focus Plan Generation
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文针对具身视觉-语言决策任务中的感知瓶颈问题，提出SceneDiver方法，通过分层聚焦计划生成提升模型对关键对象的识别能力，减少视觉幻觉。**

- **链接: [https://arxiv.org/pdf/2606.04046](https://arxiv.org/pdf/2606.04046)**

> **作者:** Boyuan Xiao; Bohong Chen; Yumeng Li; Ji Feng; Yao-Xiang Ding; Kun Zhou
>
> **备注:** Accepted at ICML 2026
>
> **摘要:** In embodied vision-language decision making tasks such as robotic manipulation and navigation, Vision-Language and Vision-Language-Action Models (VLMs & VLAs) are powerful tools with different benefits: VLMs are better at long-term planning, while VLAs are better at reactive control. However, their performance is limited by the same perceptual bottleneck: visual hallucinations arise due to the models' inability to distinguish task-relevant objects from distractors. In principle, accurate identification and focus on critical objects while filtering out irrelevant ones is the key to break this limitation. A straightforward solution is one-step focus: directly attending to essential objects. However, this approach proves ineffective because effective focus inherently requires deep scene understanding. To this end, we propose SceneDiver, a coarse-to-fine focus plan generation method for VLMs leveraging their long-term planning abilities, that first constructs a holistic scene graph to establish initial comprehension, then progressively decomposes the task into simpler sub-problems through an iterative cycle of recognition, understanding, and analysis. To enable reactive control, we also design a lightweight adapter for distilling the deliberate focus ability into VLAs. Evaluations on standard embodied AI benchmarks confirm that our method substantially reduces visual hallucinations for both VLMs and VLAs, while preserving computational efficiency in tasks requiring fast execution. Our code and data are released at: this https URL.
>
---
#### [new 044] Dual Advantage Fields
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文提出Dual Advantage Fields，解决目标条件强化学习中的策略提取问题。通过双目标表示生成局部优势信号，提升任务完成效果。**

- **链接: [https://arxiv.org/pdf/2606.04188](https://arxiv.org/pdf/2606.04188)**

> **作者:** Alexey Zemtsov; Maxim Bobrin; Alexander Nikulin; Dmitry V. Dylov; Fakhri Karray; Vladislav Kurenkov; Martin Takáč; Arip Asadulaev
>
> **备注:** Accepted by ICML 2026 Workshop on Decision-Making from Offline Datasets to Online Adaptation: Black-Box Optimization to Reinforcement Learning
>
> **摘要:** Offline goal-conditioned reinforcement learning requires both long-horizon reachability estimates and local action comparisons. Dual goal representations provide value fields that capture global goal reachability, but they do not directly specify which action should be preferred at a given state. We propose Dual Advantage Fields, a policy-extraction method that turns a bilinear dual value model into a local advantage signal. Under bilinear dual parameterization, the goal embedding is the gradient of the value field with respect to the state representation. DAF learns an action-effect model that predicts the discounted feature displacement induced by an action and scores actions by the alignment between this displacement and the goal direction. In the realizable case, this score equals the goal-conditioned Bellman advantage, yielding a standard local policy-improvement guarantee. On OGBench locomotion, manipulation, and puzzle tasks, DAF improves aggregate RLiable metrics and performs strongly in settings where locally correct actions differ from direct movement toward the final goal.
>
---
#### [new 045] Semantic Constraint Synthesis for Adaptive Trajectory Optimization via Large Language Models
- **分类: math.OC; cs.AI; cs.RO**

- **简介: 该论文属于轨迹优化任务，旨在将自然语言任务描述转化为可执行的优化代码。通过大语言模型实现语义约束合成，提升轨迹设计效率与灵活性。**

- **链接: [https://arxiv.org/pdf/2606.04123](https://arxiv.org/pdf/2606.04123)**

> **作者:** Eleanor Brosius; Yuji Takubo; Daniele Gammelli; Simone D'Amico; Marco Pavone
>
> **备注:** 7 pages, 4 figures, Presented as a short paper at IEEE CVPR 2026, AI4Space Workshop
>
> **摘要:** Trajectory optimization is a critical component for enabling safe and reliable autonomous operations in space exploration. As space missions increase in frequency, complexity, and scope, there is a growing need to rapidly formulate mathematically sound trajectory optimization problems that accurately reflect mission objectives and operational constraints. However, translating mission intent into tractable analytical formulations for trajectory optimization requires substantial domain expertise. This paper presents a framework that leverages large language models (LLMs) to translate natural language descriptions of mission requirements and constraints into executable trajectory optimization code and corresponding mathematical formulations. Experiments in spacecraft rendezvous scenarios demonstrate a high success rate in reconditioning a convex trajectory optimization problem from semantic mission requirements. Ultimately, this work highlights the potential of LLMs to bridge high-level intent and formal optimization models, enabling more flexible and efficient trajectory design of spacecraft.
>
---
#### [new 046] 3DThinkVLA: Endowing Vision-Language-Action Models with Latent 3D Priors via 3D-Thinking-Guided Co-training
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决3D空间推理不足的问题。通过引入3D先验知识，提升模型在动作预测中的3D推理能力。**

- **链接: [https://arxiv.org/pdf/2606.04436](https://arxiv.org/pdf/2606.04436)**

> **作者:** Jiaxin Shi; Xidong Zhang; Fucai Zhu; Zhe Li; Siyu Zhu; Weihao Yuan
>
> **摘要:** We propose a 3D-thinking-guided co-training framework that enables vision-language-action (VLA) models to perform 3D spatial reasoning implicitly during action prediction. Our core insight is that 3D geometry perception and 3D spatial reasoning are distinct capabilities that can be disentangled and injected at different feature hierarchies. During training, three tightly coupled components work in concert primarily within the latent space: (1) To gain geometric priors, a latent 3D geometry perception module aligns intermediate visual features with a 3D foundation model, acquiring low-level geometric cues without architectural modifications to the VLM backbone. (2) Complementing this, an online 3D reasoning distillation module mitigates the prompt-induced reasoning gap via a shared reasoning anchor token. During 3D VLM co-training, this anchor is emitted as the first output token to robustly encode spatial priors. During VLA training, it serves as an input token inserted between the task and action instructions, transferring high-level spatial thinking from explicit teacher reasoning prompts to student action prompts without chain-of-thought text generation. (3) These disentangled geometric and reasoning features are then united by a spatially augmented action integration, which jointly injects them into the action-query tokens as hierarchical spatial conditions to prevent action shortcuts. At deployment, our method retains only its lightweight adapters to perform implicit 3D reasoning, discarding the 3D foundation model and the teacher branch used for supervision. Consequently, it operates purely on 2D images without 3D sensors, external models, or explicit text generation while preventing catastrophic forgetting of the pretrained VLM, achieving state-of-the-art performance on LIBERO, LIBERO-PLUS, SimplerEnv, and real-world manipulation tasks.
>
---
## 更新

#### [replaced 001] Continuum Robot State Estimation with Actuation Uncertainty
- **分类: cs.RO**

- **简介: 该论文属于机器人状态估计任务，解决手术环境中连续机器人形状估计问题。通过联合估计形状、负载和驱动输入，提升估计精度与实时性。**

- **链接: [https://arxiv.org/pdf/2601.04493](https://arxiv.org/pdf/2601.04493)**

> **作者:** James M. Ferguson; Alan Kuntz; Tucker Hermans
>
> **备注:** Public preprint for IEEE RAL. Accepted May 2026
>
> **摘要:** Continuum robots are flexible, slender manipulators well suited for confined surgical environments. In these settings, unknown interaction forces and model uncertainty significantly affect robot shape, motivating state estimation from external observations. Existing estimation methods either neglect actuation modeling or rely on simplified deterministic actuation models. In contrast, we jointly estimate robot shape, external loads, and actuation inputs using mechanically principled actuation priors. To achieve this, we present a discrete Cosserat rod formulation with piecewise-linear strain integration that provides high numerical accuracy while inducing a sparse factor graph structure for efficient nonlinear optimization. We extend the framework to tendon-driven and parallel robots in simulation and validate it experimentally on a surgical concentric tube robot. Overall, our approach enables principled real-time estimation across multiple robot architectures while providing direct access to manipulator Jacobians through the linearized factor graph.
>
---
#### [replaced 002] Right Model, Right Time: Real-Time Cascaded-Fidelity MPC for Bipedal Walking
- **分类: cs.RO**

- **简介: 该论文属于双足行走控制任务，旨在提高实时控制效率。通过结合不同精度模型，降低计算复杂度，实现高效最优控制。**

- **链接: [https://arxiv.org/pdf/2605.04607](https://arxiv.org/pdf/2605.04607)**

> **作者:** Franek Stark; Felix Wiebe; Shubham Vyas; Dennis Mronga; Frank Kirchner
>
> **备注:** Presented at IEEE ICRA 2026 Workshop "2cnd Workshop on Frontiers of Optimization for Robotics"
>
> **摘要:** This paper presents a multi-phase whole-body model predictive control (MPC) approach for bipedal walking, combining a detailed whole-body model in the near horizon with a simplified single-rigid-body model in the later prediction steps. This reduces computational complexity while retaining prediction capabilities. The resulting nonlinear optimal control problem is solved entirely within the general-purpose, off-the-shelf nonlinear MPC framework acados, using sequential quadratic programming (SQP). Given a contact schedule and a target walking speed, the controller optimizes joint torques without depending on preselected footstep locations. The controller is validated in MuJoCo simulation on the 18-DoF bipedal robot HyPer-2.
>
---
#### [replaced 003] Sem-NaVAE: Semantically-Guided Outdoor Mapless Navigation via Generative Trajectory Priors
- **分类: cs.RO**

- **简介: 该论文提出Sem-NaVAE，用于户外无地图导航任务，通过生成轨迹并结合语义分割选择最佳路径，解决导航多样性与实时性问题。**

- **链接: [https://arxiv.org/pdf/2602.01429](https://arxiv.org/pdf/2602.01429)**

> **作者:** Gonzalo Olguín; Javier Ruiz-del-Solar
>
> **备注:** Accepted for publication in IEEE Robotics and Automation Letters (RA-L). 8 pages, 5 figures
>
> **摘要:** This work presents a mapless navigation approach for outdoor applications. It combines the exploratory capacity of conditional variational autoencoders (CVAEs) to generate trajectories and the semantic segmentation capabilities of a lightweight visual language model (VLM) to select the trajectory to execute. Open-vocabulary segmentation is used to score and select the generated trajectories based on natural language, and a state-of-the-art local planner executes velocity commands. One of the key features of the proposed approach is its ability to generate a large variability of trajectories and select them to navigate in real-time. In real-world outdoor experiments, Sem-NaVAE achieves a 90% success rate across routes of 120-240m in unseen environments, outperforming the nearest baseline by 10% while remaining within 7% of a map-based upper bound. A video showing an experimental run of the system can be found in this https URL.
>
---
#### [replaced 004] Revisiting Embodied Chain-of-Thought for Generalizable Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决 embodied chain-of-thought（CoT）的有效集成问题。通过构建大规模数据集并提出 ERVLA 模型，提升机器人在复杂任务中的泛化能力和稳定性。**

- **链接: [https://arxiv.org/pdf/2606.03784](https://arxiv.org/pdf/2606.03784)**

> **作者:** Nan Sun; Yuan Zhang; Yongkun Yang; Wentao Zhao; Peiyan Li; Jun Guo; Wenxuan Song; Pengxiang Ding; Runze Suo; Yifei Su; Xin Xiao; Xinghang Li; Huaping Liu
>
> **摘要:** Embodied chain-of-thought (CoT) aims to bridge linguistic reasoning and robotic control, but its effective form and integration strategy remain underexplored. In this paper, we revisit embodied CoT for vision-language-action (VLA) models at large scale. We construct the largest embodied CoT corpus to date, comprising 978,743 trajectories, 226.3M samples, and 2592.5 hours of robot data. Through extensive experiments, we find that effective embodied CoT should ground high-level semantic understanding into concrete action guidance, such as end-effector movement descriptions and image-space trajectories, while high-level reasoning alone brings only marginal gains. We further show that explicit CoT does not scale reliably when used as an autoregressive action prefix, as it suffers from compounding inference errors and unstable reasoning-action coupling. To address these limitations, we propose ERVLA, a VLA model that uses embodied CoT as representation-shaping supervision rather than mandatory test-time reasoning. ERVLA is trained with a reasoning-dropout strategy, enabling the model to absorb rich reasoning traces during training while predicting actions directly without CoT decoding during inference. This design improves scalability with increasing pre-training data and avoids autoregressive instability. ERVLA achieves state-of-the-art performance on LIBERO-Plus with an 86.9% success rate and reaches 53.2% success rate on VLABench, demonstrating strong out-of-distribution generalization. In real-robot experiments, ERVLA further outperforms competitive state-of-the-art baselines, especially on tasks requiring semantic disambiguation and long-horizon execution.
>
---
#### [replaced 005] DVGT: Driving Visual Geometry Transformer
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶中的3D场景重建任务，旨在解决无标定多视角图像下精确几何感知的问题。提出DVGT模型，通过注意力机制实现全局点云重建，无需依赖相机参数。**

- **链接: [https://arxiv.org/pdf/2512.16919](https://arxiv.org/pdf/2512.16919)**

> **作者:** Sicheng Zuo; Zixun Xie; Wenzhao Zheng; Shaoqing Xu; Fang Li; Shengyin Jiang; Long Chen; Zhi-Xin Yang; Jiwen Lu
>
> **备注:** Code is available at this https URL
>
> **摘要:** Perceiving and reconstructing 3D scene geometry from visual inputs is crucial for autonomous driving. However, there still lacks a driving-targeted dense geometry perception model that can adapt to different scenarios and camera configurations. To bridge this gap, we propose a Driving Visual Geometry Transformer (DVGT), which reconstructs a global dense 3D point map from a sequence of unposed multi-view visual inputs. We first extract visual features for each image using a DINO backbone, and employ alternating intra-view local attention, cross-view spatial attention, and cross-frame temporal attention to infer geometric relations across images. We then use multiple heads to decode a global point map in the ego coordinate of the first frame and the ego poses for each frame. Unlike conventional methods that rely on precise camera parameters, DVGT is free of explicit 3D geometric priors, enabling flexible processing of arbitrary camera configurations. DVGT directly predicts metric-scaled geometry from image sequences, eliminating the need for post-alignment with external sensors. Trained on a large mixture of driving datasets including nuScenes, OpenScene, Waymo, KITTI, and DDAD, DVGT significantly outperforms existing models on various scenarios. Code is available at this https URL.
>
---
#### [replaced 006] 3PoinTr: 3D Point Tracks for Learning Manipulation from Unconstrained Human Videos
- **分类: cs.RO**

- **简介: 该论文提出3PoinTr，用于从非约束人类视频中学习机器人操作策略，解决传统方法依赖标注或预设关键点的问题。通过预测3D点轨迹，提升机器人操作成功率。**

- **链接: [https://arxiv.org/pdf/2603.08485](https://arxiv.org/pdf/2603.08485)**

> **作者:** Adam Hung; Bardienus Pieter Duisterhof; Jeffrey Ichnowski
>
> **摘要:** Learning manipulation policies from human videos could greatly reduce the need for expensive robot demonstrations, but existing approaches typically require restrictive assumptions such as choreographed human motions, predefined keypoints, manual annotations, or known grasp locations. We propose 3PoinTr, a method for pretraining sample-efficient robot policies from unconstrained human videos by predicting dense 3D point tracks. In the unconstrained human demonstration videos, humans are free to follow whatever trajectories and manipulation strategies they see fit, rather than choreographing their motions to mimic a robot. 3PoinTr uses a lightweight visibility-aware transformer to learn how scene points should move from human videos, and then trains a closed-loop multitask robot policy to flexibly extract action-relevant priors from those predicted point tracks. With only 20 action-labeled robot demonstrations, 3PoinTr achieves a 25.0 percentage point higher average success rate than the strongest behavior cloning and video-pretraining baselines on real-world tasks, and a 29.6 percentage point higher average success rate in simulation. Targeted ablations support the key design choices and confirm the benefit of learning from actionless videos. We further show that 3PoinTr's point track prediction transformer outperforms a strong baseline by preserving supervision over partially occluded points. Project page: this https URL.
>
---
#### [replaced 007] LDA-1B: Scaling Latent Dynamics Action Model via Universal Embodied Data Ingestion
- **分类: cs.RO**

- **简介: 该论文提出LDA-1B，解决机器人基础模型在异构数据中高效学习动态知识的问题。通过联合学习动力学、策略和视觉预测，提升任务性能。**

- **链接: [https://arxiv.org/pdf/2602.12215](https://arxiv.org/pdf/2602.12215)**

> **作者:** Jiangran Lyu; Kai Liu; Xuheng Zhang; Haoran Liao; Yusen Feng; Wenxuan Zhu; Tingrui Shen; Jiayi Chen; Jiazhao Zhang; Yifei Dong; Wenbo Cui; Senmao Qi; Shuo Wang; Yixin Zheng; Mi Yan; Xuesong Shi; Haoran Li; Dongbin Zhao; Ming-Yu Liu; Zhizheng Zhang; Li Yi; Yizhou Wang; He Wang
>
> **备注:** Accepted at RSS 2026, Project Page:this https URL
>
> **摘要:** Recent robot foundation models largely rely on large-scale behavior cloning, which imitates expert actions but discards transferable dynamics knowledge embedded in heterogeneous embodied data. While the Unified World Model (UWM) formulation has the potential to leverage such diverse data, existing instantiations struggle to scale to foundation-level due to coarse data usage and fragmented datasets. We introduce LDA-1B, a robot foundation model that scales through universal embodied data ingestion by jointly learning dynamics, policy, and visual forecasting, assigning distinct roles to data of varying quality. To support this regime at scale, we assemble and standardize EI-30k, an embodied interaction dataset comprising over 30k hours of human and robot trajectories in a unified format. Scalable dynamics learning over such heterogeneous data is enabled by prediction in a structured DINO latent space, which avoids redundant pixel-space appearance modeling. Complementing this representation, LDA-1B employs a multi-modal diffusion transformer to handle asynchronous vision and action streams, enabling stable training at the 1B-parameter scale. Experiments in simulation and the real world show LDA-1B outperforms prior methods (e.g., $\pi_{0.5}$) by up to 21\%, 48\%, and 23\% on contact-rich, dexterous, and long-horizon tasks, respectively. Notably, LDA-1B enables data-efficient fine-tuning, gaining 10\% by leveraging 30\% low-quality trajectories typically harmful and discarded.
>
---
#### [replaced 008] Evaluating Zero-Shot and One-Shot Adaptation of Small Language Models in Leader-Follower Interaction
- **分类: cs.HC; cs.AI; cs.LG; cs.RO; eess.SY**

- **简介: 该论文属于角色分类任务，解决资源受限机器人实时角色分配问题。通过构建数据集，对比零样本和少量样本适应策略，评估小型语言模型的性能。**

- **链接: [https://arxiv.org/pdf/2602.23312](https://arxiv.org/pdf/2602.23312)**

> **作者:** Rafael R. Baptista; André de Lima Salgado; Ricardo V. Godoy; Marcelo Becker; Thiago Boaventura; Gustavo J. G. Lahr
>
> **摘要:** Leader-follower interaction is an important paradigm in human-robot interaction (HRI). Yet, assigning roles in real time remains challenging for resource-constrained mobile and assistive robots. While large language models (LLMs) have shown promise for natural communication, their size and latency limit on-device deployment. Small language models (SLMs) offer a potential alternative, but their effectiveness for role classification in HRI has not been systematically evaluated. In this paper, we present a benchmark of SLMs for leader-follower communication, introducing a novel dataset derived from a published database and augmented with synthetic samples to capture interaction-specific dynamics. We investigate two adaptation strategies: prompt engineering and fine-tuning, studied under zero-shot and one-shot interaction modes, compared with an untrained baseline. Experiments with Qwen2.5-0.5B reveal that zero-shot fine-tuning achieves robust classification performance (86.66% accuracy) while maintaining low latency (22.2 ms per sample), significantly outperforming baseline and prompt-engineered approaches. However, results also indicate a performance degradation in one-shot modes, where increased context length challenges the model's architectural capacity. These findings demonstrate that fine-tuned SLMs provide an effective solution for direct role assignment, while highlighting critical trade-offs between dialogue complexity and classification reliability on the edge.
>
---
#### [replaced 009] From Video to Control: A Survey of Learning Manipulation Interfaces from Temporal Visual Data
- **分类: cs.RO**

- **简介: 该论文属于机器人操控任务，旨在从视频中学习控制接口。解决视频缺乏动作标注的问题，通过三种方法实现视频到控制的转换。**

- **链接: [https://arxiv.org/pdf/2604.04974](https://arxiv.org/pdf/2604.04974)**

> **作者:** Linfang Zheng; Zikai Ouyang; Chen Wang; Jia Pan; Wei Zhang
>
> **摘要:** Video is a scalable observation of physical dynamics: it captures how objects move, how contact unfolds, and how scenes evolve under interaction -- all without requiring robot action labels. Yet translating this temporal structure into reliable robotic control remains an open challenge, because video lacks action supervision and differs from robot experience in embodiment, viewpoint, and physical constraints. This survey reviews methods that exploit non-action-annotated temporal video to learn control interfaces for robotic manipulation. We introduce an interface-centric taxonomy organized by where the video-to-control interface is constructed and what control properties it enables, identifying three families: direct video-action policies, which keep the interface implicit; latent-action methods, which route temporal structure through a compact learned intermediate; and explicit visual interfaces, which predict interpretable targets for downstream control. For each family, we analyze control-integration properties -- how the loop is closed, what can be verified before execution, and where failures enter. A cross-family synthesis reveals that the most pressing open challenges center on the robotics integration layer -- the mechanisms that connect video-derived predictions to dependable robot behavior -- and we outline research directions toward closing this gap.
>
---
#### [replaced 010] Ask When It Pays: Cost-Aware Open-Ended Interaction for Instance Goal Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于实例目标导航任务，解决语言描述模糊导致的导航问题。通过构建成本敏感的交互机制，提升导航效率与准确性。**

- **链接: [https://arxiv.org/pdf/2606.03175](https://arxiv.org/pdf/2606.03175)**

> **作者:** Xunyi Zhao; Sihao Lin; Gengze Zhou; Zerui Li; Shijie Li; Wei Tao; Jiajun Liu; Qi Wu
>
> **摘要:** Instance Goal Navigation (IGN) requires an embodied agent to find a specific object instance among distractors from an under-specified natural-language description. Such ambiguity often cannot be resolved from perception and language alone, making interaction with an oracle a natural mechanism for disambiguation. Prior interactive methods allow oracle queries but treat lightweight clarification and route-level guidance alike, letting agents boost success rate through repeated high-information questions rather than by resolving the underlying ambiguity efficiently. We recast interactive IGN as a cost-sensitive uncertainty-reduction problem, where the agent should ask the question whose answer provides the largest reduction in navigation uncertainty relative to its penalty. To this end, we apply an information-gain analysis on existing navigation corpora to identify which cues reduce navigation uncertainty, yielding a compact set of question types and data-derived weights. However, existing interactive navigation benchmarks do not model the cost of different question types or evaluate how efficiently agents use interaction, making them unsuitable for studying cost-sensitive interaction. Based on this taxonomy, we construct a benchmark for diagnosing interaction behavior and efficiency, together with a Weighted Success Rate metric that penalizes each query by its derived cost. We further propose a zero-shot MLLM navigator that selectively queries at each decision step only when the expected uncertainty reduction justifies the interaction cost.
>
---
#### [replaced 011] PerchRL: Vision-Based Agile Perching on Inclined Platforms under Rapid and Irregular Motion
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出PerchRL，用于解决无人机在快速不规则运动的倾斜平台上的视觉自主着陆问题。通过分阶段强化学习方法提升适应性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2606.03441](https://arxiv.org/pdf/2606.03441)**

> **作者:** Zihong Lu; Zongzhuo Liu; Huaxu Li; Jinqiang Cui; Jie Mei; Youmin Gong; U Kei Cheang; Boyu Zhou
>
> **摘要:** Autonomous vision-based perching of quadrotors on moving inclined platforms is critical for air-ground collaboration but remains challenging due to the limited field of view (FOV). In this paper, we propose PerchRL, a reinforcement learning (RL) framework for vision-based agile perching on inclined platforms under rapid and irregular motion. Specifically, we employ a two-stage learning strategy consisting of state-based pre-training followed by vision-based fine-tuning. To improve generalization across diverse platform motions, we employ randomized platform trajectories to prevent overfitting and temporal augmentation methods to capture latent motion patterns from historical observations. During vision-based fine-tuning, a hybrid learning framework consisting of visibility-aware state augmentation and active perception rewards is presented to improve robustness under intermittent visual loss. Extensive simulation and real-world experiments demonstrate the feasibility, stability, and real-time performance of PerchRL, while successful deployment across distinct quadrotor platforms further validates its adaptability. The source code will be released to benefit the community.
>
---
#### [replaced 012] Transformer-Based Autonomous Driving Models and Deployment-Oriented Compression: A Survey
- **分类: cs.LG; cs.AI; cs.CV; cs.RO; eess.SY**

- **简介: 本文综述基于Transformer的自动驾驶模型及其部署压缩技术，探讨如何在效率约束下优化模型设计，提升系统可部署性与安全性。**

- **链接: [https://arxiv.org/pdf/2304.10891](https://arxiv.org/pdf/2304.10891)**

> **作者:** Juan Zhong; Yuhang Shi; Zukang Xu; Xi Chen
>
> **摘要:** Transformer-based models are becoming a central paradigm in autonomous driving because they can capture long-range spatial dependencies, multi-agent interactions, and multimodal context across perception, prediction, and planning. At the same time, their deployment in real vehicles remains difficult because high-capacity attention-based architectures impose substantial latency, memory, and energy overhead. This survey reviews representative Transformer-based autonomous driving models and organizes them by task role, sensing configuration, and architectural design. More importantly, it examines these models from a deployment-oriented perspective and analyzes how efficiency constraints reshape model design choices in practice. We further review compression and acceleration strategies relevant to Transformer-based driving systems, including quantization, pruning, knowledge distillation, low-rank approximation, and efficient attention, and discuss their benefits, limitations, and task-dependent applicability. Rather than treating compression as an isolated post-processing step, we highlight it as a system-level design consideration that directly affects deployability, robustness, and safety. Finally, we identify open challenges and future research directions toward standardized, safety-aware, and hardware-conscious evaluation of efficient autonomous driving systems.
>
---
#### [replaced 013] Dynamic Policy Learning for Legged Robot with Simplified Model Pretraining and Model-Homotopy-Inspired Transfer
- **分类: cs.RO**

- **简介: 该论文属于机器人动态运动生成任务，旨在解决腿式机器人在复杂动态行为中的策略迁移问题。通过简化模型预训练和模型同伦迁移方法，提升策略转移效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2512.24698](https://arxiv.org/pdf/2512.24698)**

> **作者:** Dongyun Kang; Min-Gyu Kim; Tae-Gyu Song; Hajun Kim; Sehoon Ha; Hae-Won Park
>
> **备注:** 8 pages
>
> **摘要:** Generating dynamic motions for legged robots remains a challenging problem. While reinforcement learning has achieved notable success in various legged locomotion tasks, producing highly dynamic behaviors often requires extensive reward tuning or high-quality demonstrations. Leveraging reduced-order models can help mitigate these challenges. However, the model discrepancy poses a significant challenge when transferring policies to full-body dynamics environments. In this work, we introduce a continuation-based learning framework that combines simplified model pretraining and model-homotopy-inspired transfer to efficiently generate and refine complex dynamic behaviors. First, we pretrain the policy using a single rigid body model to capture core motion patterns in a simplified environment. Next, we employ a continuation strategy to progressively transfer the policy to the full-body environment, minimizing performance loss. To define the continuation path, we introduce a parametric transition path from the single rigid body model to the full-body model by gradually redistributing mass and inertia between the trunk and legs. The proposed method achieves faster convergence and demonstrates superior stability during the transfer process compared to baseline methods. Our framework is validated on a range of dynamic tasks, including flips and wall-assisted maneuvers, and is successfully deployed on a real quadrupedal robot.
>
---
#### [replaced 014] A Reproducible and Physically Feasible Dynamic Parameter Identification Framework for a Low-Cost Robot Arm
- **分类: cs.RO**

- **简介: 该论文属于机器人动力学参数辨识任务，旨在解决低成本机械臂动态模型的可重复性和物理可行性问题。通过简化模型、设计识别运动并结合优化算法，提升模型精度与可靠性。**

- **链接: [https://arxiv.org/pdf/2605.15949](https://arxiv.org/pdf/2605.15949)**

> **作者:** Junji Oaki; Koki Yamane; Koki Inami; Sho Sakaino
>
> **备注:** 11 pages, 8 figures, 7 tables, 1 algorithm and 2 appendices
>
> **摘要:** This paper presents a reproducible and physically feasible dynamic parameter identification framework for CRANE-X7, a low-cost robot arm driven by modular smart actuators. To improve practical identifiability, products of inertia are removed according to approximate link symmetry, reducing the rigid-body model from 65 to 39 base parameters. Identification motions are hand-designed from structured single-joint and adjacent-joint primitives under practical joint-range limits. The proposed pipeline combines preprocessing, inverse-dynamics-regressor-based ordinary least squares (OLS), conditional semidefinite-programming (SDP) projection for feasibility recovery, and closed-loop input error (CLIE) refinement. Candidate solutions from 40 structured trajectories are analyzed in a common principal component analysis (PCA) space to select a statistically central representative model. Because statistical centrality alone does not ensure physical acceptability, the selected model is finally screened by an all-pose positive-definiteness audit of the inertia matrix and, when necessary, corrected by a localized post-CLIE SDP rescue step. Experiments show that the parameter cloud becomes progressively more concentrated from OLS to SDP and CLIE, while the final accepted model preserves high predictive accuracy on held-out validation motions. These results demonstrate a practical route to statistically coherent and physically feasible dynamic models for low-cost robot platforms.
>
---
#### [replaced 015] Lost in Fog: Sensor Perturbations Expose Reasoning Fragility in Driving VLAs
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究自动驾驶中视觉-语言-动作模型的鲁棒性，通过传感器扰动实验分析推理一致性与轨迹可靠性关系，提出以因果链解释作为安全评估指标。**

- **链接: [https://arxiv.org/pdf/2605.21446](https://arxiv.org/pdf/2605.21446)**

> **作者:** Abhinaw Priyadershi; Jelena Frtunikj
>
> **摘要:** Interpretable autonomous driving planners depend not only on generating explanations, but also on those explanations remaining reliable under real-world sensor degradation. In this paper we present a controlled perturbation study of Vision-Language-Action (VLA) robustness in autonomous driving, evaluating Alpamayo R1 (10B parameters) across 1,996 scenarios under eight sensor perturbations (Gaussian noise at four intensities, two lighting extremes, and two fog levels; ${\sim}18{,}000$ inference trials). We find that reasoning consistency is a high-fidelity indicator of trajectory reliability: when Chain-of-Causation (CoC) explanations change after perturbation, trajectory deviation spikes $5.3{\times}$ (21.8m vs 4.1m), with $r\!=\!0.99$ across attack types and $r_{pb}\!=\!0.53$ per-sample (Cohen's $d\!=\!1.12$). A controlled ablation provides evidence that enabling CoC generation is associated with improved trajectory accuracy (11.8% on average across conditions; $p < 0.0001$) under matched inference settings. Over the tested noise range ($\sigma \in \{10, 30, 50, 70\}$), degradation is approximately linear ($R^2\!=\!0.957$), while standard input preprocessing defenses provide only marginal relief. Together, these results establish CoC consistency as a quantitative proxy for planning safety and motivate reasoning-based runtime monitoring for safer VLA deployment.
>
---
#### [replaced 016] BiPneu: Design and Control of a Bipolar-Pressure Pneumatic System for Soft Robots
- **分类: cs.RO**

- **简介: 该论文属于软体机器人控制任务，旨在解决正负压调节难题。提出BiPneu系统与DM-SMC控制器，实现高精度、快速响应的压力控制。**

- **链接: [https://arxiv.org/pdf/2605.12804](https://arxiv.org/pdf/2605.12804)**

> **作者:** Yu Mei; Xinyu Zhou; Vedant Naik; Alan Gao; Xiaobo Tan
>
> **备注:** Full Version of BiPenu, including the supplementary materials
>
> **摘要:** Positive-negative pressure regulation is critical to soft robotic actuators, enabling large motion ranges and versatile actuation modes. However, achieving high-performance regulation across both pressure polarities remains challenging due to asymmetric inflation-deflation dynamics, valve nonlinearities, and switching-induced flow disturbances. This paper presents BiPneu, a scalable and cost-efficient multi-channel bipolar-pressure pneumatic system for soft robots that enables wide-range, accurate, and responsive pressure regulation while providing seamless compatibility with high-level software ecosystems. A dual-mode sliding-mode controller (DM-SMC) with hysteresis-supervised mode selection is proposed based on a hybrid electro-pneumatic model. Extensive simulation and experiments demonstrate the superior performance of DM-SMC in tracking step and sinusoidal pressure references compared with both advanced model predictive controllers and well-tuned PID controllers. Experimental results show average absolute errors of 1.44 kPa in multi-step tests and 4.23 kPa in sinusoidal tracking, corresponding to reductions of 11.9% and 35.6% relative to PID control, along with improved control effort, valve switching rate, and transient response. Robustness of DM-SMC is further verified on a bellow actuator with pressure-dependent volume. Finally, BiPneu's capability is demonstrated via two soft robotic examples, quick ball-maneuvering with a soft parallel manipulator and real-time finite element method (FEM)-based teleoperation of a soft bellows actuator.
>
---
#### [replaced 017] Simplicial Embeddings Improve Sample Efficiency in Actor-Critic Agents
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，旨在提升策略-评估代理的样本效率。通过引入单纯形嵌入，增强表示结构，改善泛化能力，提升性能且不牺牲训练速度。**

- **链接: [https://arxiv.org/pdf/2510.13704](https://arxiv.org/pdf/2510.13704)**

> **作者:** Johan Obando-Ceron; Walter Mayor; Samuel Lavoie; Scott Fujimoto; Aaron Courville; Pablo Samuel Castro
>
> **摘要:** Recent works have proposed accelerating the wall-clock training time of actor-critic methods via the use of large-scale environment parallelization; unfortunately, these can sometimes still require large number of environment interactions to achieve a desired level of performance. Noting that well-structured representations can improve the generalization and sample efficiency of deep reinforcement learning (RL) agents, we propose the use of simplicial embeddings: lightweight representation layers that constrain embeddings to simplicial structures. This geometric inductive bias results in sparse and discrete features that stabilize critic bootstrapping and strengthen policy gradients. When applied to FastTD3, FastSAC, and PPO, simplicial embeddings consistently improve sample efficiency and final performance across a variety of continuous- and discrete-control environments, without any loss in runtime speed.
>
---
#### [replaced 018] How Users Understand Robot Foundation Model Performance through Task Success Rates and Beyond
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究非专家用户如何理解机器人基础模型的性能，解决用户对模型能力认知不足的问题。通过实验分析用户对任务成功率及其他信息的使用情况。**

- **链接: [https://arxiv.org/pdf/2602.03920](https://arxiv.org/pdf/2602.03920)**

> **作者:** Isaac Sheidlower; Jindan Huang; James Staley; Bingyu Wu; Qicong Chen; Reuben Aronson; Elaine Short
>
> **摘要:** Robot Foundation Models (RFMs) represent a promising approach to developing general-purpose home robots. Given the broad capabilities of RFMs, users will inevitably ask an RFM-based robot to perform tasks that the RFM was not trained or evaluated on. In these cases, it is crucial that users understand the risks associated with attempting novel tasks due to the relatively high cost of failure. Furthermore, an informed user who understands an RFM's capabilities will know what situations and tasks the robot can handle. In this paper, we study how non-roboticists interpret performance information from RFM evaluations. These evaluations typically report task success rate (TSR) as the primary performance metric. While TSR is intuitive to experts, it is necessary to validate whether novices also use this information as intended. Toward this end, we conducted a study in which users saw real evaluation data, including TSR, failure case descriptions, and videos from multiple published RFM research projects. The results highlight that non-experts not only use TSR in a manner consistent with expert expectations but also highly value other information types, such as failure cases that are not often reported in RFM evaluations. Furthermore, we find that users want access to both real data from previous evaluations of the RFM and estimates from the robot about how well it will do on a novel task.
>
---
#### [replaced 019] DiscreteRTC: Discrete Diffusion Policies are Natural Asynchronous Executors
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决动态环境下异步执行问题。提出DiscreteRTC，利用离散扩散策略实现自然异步执行，提升成功率与效率。**

- **链接: [https://arxiv.org/pdf/2604.25050](https://arxiv.org/pdf/2604.25050)**

> **作者:** Pengcheng Wang; Kaiwen Hong; Chensheng Peng; Katherine Driggs-Campbell; Masayoshi Tomizuka; Chenfeng Xu; Chen Tang
>
> **摘要:** Unlike chatbots, physical AI must act while the world keeps evolving. Therefore, the inter-chunk pause of synchronous executors are fatal for dynamic tasks regardless of how fast the inference is. Asynchronous execution -- thinking while acting -- is therefore a structural requirement, and real-time chunking (RTC) makes it viable by recasting chunk transitions as inpainting: freezing committed actions and consistently generating the remainder. However, RTC with flow-matching policy is structurally suboptimal: its inpainting comes from inference-time corrections rather than the base policy, yielding little pre-training benefit, specific fine-tuning, heuristic guidance, and extra computation that inflates the latency. In this work, we observe that discrete diffusion policies, which generate actions by iteratively unmasking, are natural asynchronous executors that resolve all limitations at once: they are fine-tuning free since inpainting is their native operation, while early stopping further provides adaptive guidance and reduces inference cost. We propose DiscreteRTC, which replaces external corrections with native unmasking, and show on dynamic simulated benchmarks and real-world dynamic manipulation tasks that it achieves higher success rates than continuous RTC and other baselines. In summary, DiscreteRTC is simpler to implement with 0 lines of additional code to enable async inpainting, faster at inference with only ~0.7 computation compared with generating actions from scratch, and better at execution with 65% higher success rate in real-world hockey defend task compared with flow-matching RTC, and 30% higher compared with training-time flow-matching RTC. More visualizations are on this https URL.
>
---
#### [replaced 020] Contextual Multi-Task Reinforcement Learning for Autonomous Reef Monitoring
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主水下监测任务，旨在解决水下环境不确定性带来的控制难题。通过上下文多任务强化学习，提升策略的泛化与复用能力。**

- **链接: [https://arxiv.org/pdf/2604.12645](https://arxiv.org/pdf/2604.12645)**

> **作者:** Melvin Laux; Yi-Ling Liu; Rina Alo; Sören Töpper; Mariela De Lucas Alvarez; Frank Kirchner; Rebecca Adam
>
> **备注:** To be published in IEEE OCEANS 2026 (Sanya) conference proceedings
>
> **摘要:** Although autonomous underwater vehicles promise the capability of marine ecosystem monitoring, their deployment is fundamentally limited by the difficulty of controlling vehicles under highly uncertain and non-stationary underwater dynamics. To address these challenges, we employ a data-driven reinforcement learning approach to compensate for unknown dynamics and task variations. Traditional single-task reinforcement learning has a tendency to overfit the training environment, thus, limit the long-term usefulness of the learnt policy. Hence, we propose to use a contextual multi-task reinforcement learning paradigm instead, allowing us to learn controllers that can be reused for various tasks, e.g., detecting oysters in one reef and detecting corals in another. We evaluate whether contextual multi-task reinforcement learning can efficiently learn robust and generalisable control policies for autonomous underwater reef monitoring. We train a single context-dependent policy that is able to solve multiple related monitoring tasks in a simulated reef environment in HoloOcean. In our experiments, we empirically evaluate the contextual policies regarding sample-efficiency, zero-shot generalisation to unseen tasks, and robustness to varying water currents. By utilising multi-task reinforcement learning, we aim to improve the training effectiveness, as well as the reusability of learnt policies to take a step towards more sustainable procedures in autonomous reef monitoring.
>
---
#### [replaced 021] DEFLECT: Temporal Counterfactual Preference Learning for Delay-Robust Asynchronous VLAs
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出DEFLECT框架，解决异步VLA中的延迟鲁棒性问题，通过时序反事实偏好学习提升动作与执行状态的一致性。**

- **链接: [https://arxiv.org/pdf/2605.19294](https://arxiv.org/pdf/2605.19294)**

> **作者:** Yixiang Zhu; Yonghao Chen; Zijie Yang; Yusong Hu; Xinyu Chen
>
> **摘要:** Vision-Language-Action (VLA) policies increasingly rely on asynchronous inference to hide large-model latency behind ongoing robot motion. While this avoids the stop-and-go behavior of synchronous action-chunk execution, it creates a prediction-execution mismatch: the next chunk is computed from a stale observation at inference start but executed only after the robot and scene have evolved. As a result, actions that fit the prediction-time state can become misaligned with the execution-time state. Existing runtime repair, behavior-cloning, and preference-alignment approaches do not directly teach the policy to resolve this stale-input mismatch. We propose DEFLECT, an offline post-training framework for delay-robust asynchronous VLAs. DEFLECT converts latency-induced mismatch into counterfactual preference supervision: a frozen reference VLA generates a preferred chunk from the future execution-time observation and a rejected chunk from the stale prediction-time observation. The trainable policy scores both chunks under the same deployment-time input, learning to favor execution-time-aligned actions while a supervised fine-tuning anchor preserves the expert action manifold. DEFLECT requires no human preference labels, reward models, online robot rollouts, architectural changes, or additional inference-time computation. Across Kinetix, LIBERO, and three real-robot tasks, DEFLECT improves delay robustness over strong asynchronous VLA baselines, raising high-latency success by up to 6.4 percentage points and achieving a 4.6 percentage-point gain at the longest delay on a real-scale VLA.
>
---
#### [replaced 022] Vectorized Online POMDP Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人决策任务，解决部分可观测环境下的规划问题。针对POMDP求解的并行化难题，提出VOPP方法，通过向量化计算实现高效在线规划。**

- **链接: [https://arxiv.org/pdf/2510.27191](https://arxiv.org/pdf/2510.27191)**

> **作者:** Marcus Hoerger; Muhammad Sudrajat; Hanna Kurniawati
>
> **备注:** 8 pages, 3 figures. Accepted at ICRA 2026
>
> **摘要:** Planning under partial observability is an essential capability of autonomous robots. The Partially Observable Markov Decision Process (POMDP) provides a powerful framework for planning under partial observability problems, capturing the stochastic effects of actions and the limited information available through noisy observations. POMDP solving could benefit tremendously from massive parallelization on today's hardware, but parallelizing POMDP solvers has been challenging. Most solvers rely on interleaving numerical optimization over actions with the estimation of their values, which creates dependencies and synchronization bottlenecks between parallel processes that can offset the benefits of parallelization. In this paper, we propose Vectorized Online POMDP Planner (VOPP), a novel parallel online solver that leverages a recent POMDP formulation which analytically solves part of the optimization component, leaving numerical computations to consist of only estimation of expectations. VOPP represents all data structures related to planning as a collection of tensors, and implements all planning steps as fully vectorized computations over this representation. The result is a massively parallel online solver with no dependencies or synchronization bottlenecks between concurrent processes. Experimental results indicate that VOPP is at least $20\times$ more efficient in computing near-optimal solutions compared to an existing state-of-the-art parallel online solver. Moreover, VOPP outperforms state-of-the-art sequential online solvers, while using a planning budget that is $1000\times$ smaller.
>
---
#### [replaced 023] Learning While Deploying: Fleet-Scale Reinforcement Learning for Generalist Robot Policies
- **分类: cs.RO**

- **简介: 该论文提出LWD框架，解决机器人部署中持续学习问题，通过 fleet-scale 数据提升通用机器人策略的鲁棒性与任务适应性。**

- **链接: [https://arxiv.org/pdf/2605.00416](https://arxiv.org/pdf/2605.00416)**

> **作者:** Yi Wang; Xinchen Li; Pengwei Xie; Pu Yang; Buqing Nie; Yunuo Cai; Qinglin Zhang; Chendi Qu; Jeffrey Wu; Jianheng Song; Xinlin Ren; Jingshun Huang; Mingjie Pan; Siyuan Feng; Zhi Chen; Jianlan Luo
>
> **备注:** No
>
> **摘要:** Generalist robot policies increasingly benefit from large-scale pretraining, but offline data alone is insufficient for robust real-world deployment. Deployed robots encounter distribution shifts, long-tail failures, task variations, and human correction opportunities that fixed demonstration datasets cannot fully capture. We present Learning While Deploying (LWD), a fleet-scale offline-to-online reinforcement learning framework for continual post-training of generalist Vision-Language-Action (VLA) policies. Starting from a pretrained VLA policy, LWD closes the loop between deployment, shared physical experience, policy improvement, and redeployment by using autonomous rollouts and human interventions collected across a robot fleet. To stabilize learning from heterogeneous, sparse-reward fleet data, LWD combines Distributional Implicit Value Learning (DIVL) for robust value estimation with Q-learning via Adjoint Matching (QAM) for policy extraction in flow-based VLA action generators. We validate LWD on a fleet of 16 dual-arm robots across eight real-world manipulation tasks, including semantic grocery restocking and 3--5 minute long-horizon tasks. A single generalist policy improves as fleet experience accumulates, reaching an average success rate of 95%, with the largest gains on long-horizon tasks.
>
---
#### [replaced 024] Too Much of a Good Thing: When sim2real Efforts Impede Policy Learning (And What to Do About It)
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于强化学习任务，解决sim2real迁移中的政策学习问题。指出过度依赖sim2real导致激励不匹配，提出sim2sim2real方法缓解此问题。**

- **链接: [https://arxiv.org/pdf/2606.02636](https://arxiv.org/pdf/2606.02636)**

> **作者:** Kyle Morgenstein; Bharath Masetty; Stephen Welch; Luis Sentis
>
> **摘要:** While sim2real efforts are necessary for effective policy transfer to hardware, there is such a thing as too much of a good thing. We argue that sim2real efforts have led to misaligned incentives with policy learning, resulting in simulator lock in and poor policy exploration due to the unreasonable constraints imposed by the real world. We offer a diagnosis and explanation of the current status of the problem, and propose a potential solution via a sim2sim2real paradigm that leverages the robot's kinematics as the sole design constraint.
>
---
#### [replaced 025] ZeroWBC: Learning Natural Whole-Body Humanoid Interaction from Human Egocentric Data
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出ZeroWBC框架，解决无监督的人形机器人全身交互控制问题。通过人类第一视角视频和文本指令，生成自然运动并实现交互。**

- **链接: [https://arxiv.org/pdf/2603.09170](https://arxiv.org/pdf/2603.09170)**

> **作者:** Haoran Yang; Jiacheng Bao; Yucheng Xin; Haoming Song; Yuyang Tian; Bin Zhao; Dong Wang; Xuelong Li
>
> **摘要:** Achieving versatile and natural whole-body humanoid interaction control remains challenging due to the high cost of whole-body teleoperation data. We present ZeroWBC, a teleoperation-free framework that learns humanoid whole-body interaction from human egocentric videos paired with synchronized whole-body motion and text annotations. ZeroWBC adopts a generation-then-tracking formulation to tackle the static scene whole-body interaction control problem. Given an initial egocentric image and a language instruction, a fine-tuned Vision-Language Model generates future human whole-body motion tokens, which are decoded into continuous motions and retargeted to the humanoid. The resulting reference motions, together with root and key body-part trajectories, are then executed by a general interactive motion tracking policy. To improve interaction performance, we introduce an interaction-oriented tracking reward that prioritizes global root and key body-part trajectory alignment while preserving natural whole-body motion. Experiments on the Unitree G1 humanoid robot show that ZeroWBC enables diverse scene-aware behaviors without robot teleoperation demonstrations. These results suggest a scalable paradigm for learning natural humanoid whole-body interaction from human egocentric data.
>
---
#### [replaced 026] A 3D Isovist World Model -- Revealing a City's Unseen Geometry and Its Emergent Cross-City Signature
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出一种3D等值体世界模型，用于城市导航，解决传统模型忽略真实三维空间的问题。通过预测可行走空间，实现城市空间特征的高效建模与分析。**

- **链接: [https://arxiv.org/pdf/2606.03609](https://arxiv.org/pdf/2606.03609)**

> **作者:** Xuhui Lin; Stephen Law; Nanjiang Chen; Kunyao Li; Tao Yang
>
> **摘要:** Embodied agents that navigate cities rely on world models that predict how their surroundings will change as they move. But for navigation, what matters is not what the buildings look like; it is where the agent can go. Most world models nonetheless predict appearance, learning how a scene looks rather than the space an agent can move through. Those that do target geometry, such as bird's-eye-view occupancy grids, flatten the three-dimensional environment onto a ground plane, discarding the above-ground and multi-level structure that shapes real navigation. What is missing is a predictive target that captures the navigable geometry an agent actually traverses, without photometric entanglement and without collapsing the third dimension. Our key idea is to model the open volume between buildings, the negative space, encoded as a 3D isovist: a spherical visibility-depth map recording the distance to the nearest surface in every direction. We introduce an embodied world model that predicts the next isovist from a short history of past isovists and a movement action. The prediction is formulated as a depth residual so the decoder inherits sharp building edges, trained with self-rollout scheduled sampling to keep corrupted context on the geometry manifold, and equipped with a persistent latent bird's-eye-view spatial map for cross-path consistency. Our central finding is emergent and unexpected: a single city-blind model trained on Manhattan and Paris develops a cross-city spatial signature, with city identity linearly decodable from its temporal latents far above single-frame baselines, so the signature lives in the learned dynamics rather than in appearance. The representation is lightweight, interpretable, and reproducible, offering a geometric substrate for spatial reasoning in embodied AI, robotics, and urban analysis, released with an open dataset and pipeline.
>
---
#### [replaced 027] ContactExplorer: Contact Coverage-Guided Exploration for General-Purpose Dexterous Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出ContactExplorer，用于解决灵活操作中的探索问题。通过接触模式引导探索，提升样本效率与成功率，适用于多种操作任务。**

- **链接: [https://arxiv.org/pdf/2603.10971](https://arxiv.org/pdf/2603.10971)**

> **作者:** Zixuan Liu; Ruoyi Qiao; Chenrui Tie; Xuanwei Liu; Yunfan Lou; Chongkai Gao; Zhixuan Xu; Lin Shao
>
> **备注:** 24 pages
>
> **摘要:** Reinforcement learning has achieved remarkable success in domains such as Atari games, navigation, and locomotion, where exploration can often be guided by novelty over states or dynamics. In contrast, dexterous manipulation requires rich physical hand--object interactions, but existing methods often suffer from unstable contact-based novelty signals, inefficient distance novelty signals, or reliance on task-specific priors. We propose ContactExplorer, a general exploration method for dexterous manipulation tasks. ContactExplorer represents contact as the intersection between object surface points and hand keypoints, encouraging dexterous hands to discover diverse and novel contact patterns, namely which fingers contact which object regions. It maintains a contact counter conditioned on discretized object states obtained via learned hash codes, capturing how frequently each finger interacts with different object regions. This counter is leveraged in two complementary ways: (1) to assign a count-based contact coverage reward that promotes exploration of novel contact patterns, and (2) an energy-based reaching reward that guides the agent toward under-explored contact regions. We evaluate ContactExplorer on a diverse set of dexterous manipulation tasks. Experimental results show that ContactExplorer substantially improves sample efficiency and success rates over existing exploration methods, and that the contact patterns learned with ContactExplorer transfer robustly to the real world. Project page is this https URL.
>
---
#### [replaced 028] PHASER: Phase-Aware and Semantic Experience Replay for Vision-Language-Action Models
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出PHASER框架，解决VLA模型在持续学习中的灾难性遗忘问题。通过相位感知和语义经验回放，提升机器人操作任务的长期性能。**

- **链接: [https://arxiv.org/pdf/2606.03598](https://arxiv.org/pdf/2606.03598)**

> **作者:** Ziyang Chen; Shaoguang Wang; Weiyu Guo; Qianyi Cai; He Zhang; Pengteng Li; Yiren Zhao; Yandong Guo
>
> **备注:** 20 pages, 8 figures, 12 tables
>
> **摘要:** Vision-Language-Action (VLA) models have achieved remarkable success in language-conditioned robotic manipulation. However, deploying these models in open-ended environments requires continuously acquiring novel skills, a process that inevitably triggers severe catastrophic forgetting of previously learned behaviors. While experience replay (ER) serves as a standard mitigating strategy, naive uniform sampling fundamentally misaligns with the temporal characteristics of manipulation trajectories. It systematically under-samples brief but causally critical sub-skills, leading to phase starvation, and completely overlooks the varying degrees of forgetting across historical tasks. To overcome these limitations, we introduce PHASER, an architecture-agnostic continual learning framework. PHASER employs a phase-centric capacity allocation to guarantee equal memory support for all sub-skills, coupled with a multi-modal interference routing strategy that dynamically prioritizes historical phases at high risk of forgetting. Furthermore, to enable fully autonomous lifelong adaptation, we integrate Auto-PC, a lightweight pipeline combining unsupervised action-signal change-point detection with VLM-based semantic verification to extract temporal boundaries without intensive manual supervision. Evaluated across three VLA backbones on LIBERO continual learning suites, PHASER yields substantial empirical improvements, increasing Average Success Rate (ASR) by up to 31% over matched-budget ER and achieving an 87.8% final ASR on the LIBERO-Goal CL setting.
>
---
