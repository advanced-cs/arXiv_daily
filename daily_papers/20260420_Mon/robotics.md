# 机器人 cs.RO

- **最新发布 29 篇**

- **更新 16 篇**

## 最新发布

#### [new 001] One-Shot Cross-Geometry Skill Transfer through Part Decomposition
- **分类: cs.RO**

- **简介: 该论文属于机器人技能迁移任务，旨在解决机器人难以适应陌生形状物体的问题。通过分解物体为语义部分，实现跨几何的技能迁移。**

- **链接: [https://arxiv.org/pdf/2604.15455](https://arxiv.org/pdf/2604.15455)**

> **作者:** Skye Thompson; Ondrej Biza; George Konidaris
>
> **备注:** ICRA 2026
>
> **摘要:** Given a demonstration, a robot should be able to generalize a skill to any object it encounters-but existing approaches to skill transfer often fail to adapt to objects with unfamiliar shapes. Motivated by examples of improved transfer from compositional modeling, we propose a method for improving transfer by decomposing objects into their constituent semantic parts. We leverage data-efficient generative shape models to accurately transfer interaction points from the parts of a demonstration object to a novel object. We autonomously construct an objective to optimize the alignment of those points on skill-relevant object parts. Our method generalizes to a wider range of object geometries than existing work, and achieves successful one-shot transfer for a range of skills and objects from a single demonstration, in both simulated and real environments.
>
---
#### [new 002] DTEA: A Dual-Topology Elastic Actuator Enabling Real-Time Switching Between Series and Parallel Compliance
- **分类: cs.RO**

- **简介: 该论文属于机器人执行器设计任务，旨在解决传统弹性执行器无法实时切换串联与并联拓扑的问题。研究提出DTEA，实现两种模式的动态切换，并验证其性能与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.15865](https://arxiv.org/pdf/2604.15865)**

> **作者:** Vishal Ramesh; Aman Singh; Shishir Kolathaya
>
> **摘要:** Series and parallel elastic actuators offer complementary but mutually exclusive advantages, yet no existing actuator enables real-time transition between these topologies during operation. This paper presents a novel actuator design called the Dual-Topology Elastic Actuator (DTEA), which enables dynamic switching between SEA and PEA topologies during operation. A proof-of-concept prototype of the DTEA is developed to demonstrate the feasibility of the topology-switching mechanism. Experiments are conducted to evaluate the robustness and timing of the switching mechanism under operational conditions. The actuator successfully performed 324 topology-switching cycles under load without damage, demonstrating the robustness of the mechanism. The measured switching time between SEA and PEA modes is under 33.33 ms. Additional experiments are conducted to characterize the static stiffness and disturbance rejection performance in both SEA and PEA modes. Static stiffness tests show that the PEA mode is 1.53x stiffer than the SEA mode, with KSEA = 5.57 +/- 0.02 Nm/rad and KPEA = 8.54 +/- 0.02 Nm/rad. Disturbance rejection experiments show that the mean peak deflection in SEA mode is 2.26x larger than in PEA mode (5.2 deg vs. 2.3 deg), while the mean settling time is 3.45x longer (1380 ms vs. 400 ms). The observed behaviors are consistent with the known characteristics of conventional SEA and PEA actuators, validating the functionality of both modes in the DTEA actuator.
>
---
#### [new 003] NeuroMesh: A Unified Neural Inference Framework for Decentralized Multi-Robot Collaboration
- **分类: cs.RO; cs.MA**

- **简介: 该论文提出NeuroMesh，解决异构多机器人协作中的模型部署难题。通过统一的神经推理框架，实现感知、控制与任务分配的高效协同。**

- **链接: [https://arxiv.org/pdf/2604.15475](https://arxiv.org/pdf/2604.15475)**

> **作者:** Yang Zhou; Yash Shetye; Long Quang; Devon Super; Jesse Milzman; Manohari Goarin; Aditya Azad; Devang Sunil Dhake; Jeffery Mao; Carlos Nieto-Granda; Giuseppe Loianno
>
> **备注:** 8 page, 8 figures, Accepted at the IEEE Robotics Automation Letter (RA-L)
>
> **摘要:** Deploying learned multi-robot models on heterogeneous robots remains challenging due to hardware heterogeneity, communication constraints, and the lack of a unified execution stack. This paper presents NeuroMesh, a multi-domain, cross-platform, and modular decentralized neural inference framework that standardizes observation encoding, message passing, aggregation, and task decoding in a unified pipeline. NeuroMesh combines a dual-aggregation paradigm for reduction- and broadcast-based information fusion with a parallelized architecture that decouples cycle time from end-to-end latency. Our high-performance C++ implementation leverages Zenoh for inter-robot communication and supports hybrid GPU/CPU inference. We validate NeuroMesh on a heterogeneous team of aerial and ground robots across collaborative perception, decentralized control, and task assignment, demonstrating robust operation across diverse task structures and payload sizes. We plan to release NeuroMesh as an open-source framework to the community.
>
---
#### [new 004] ShapeGen: Robotic Data Generation for Category-Level Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决类别级物体操作中的形状多样性问题。通过生成多样化3D操作数据，提升策略的泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.15569](https://arxiv.org/pdf/2604.15569)**

> **作者:** Yirui Wang; Xiuwei Xu; Angyuan Ma; Bingyao Yu; Jie Zhou; Jiwen Lu
>
> **备注:** 15 pages, 11 figures. Under review
>
> **摘要:** Manipulation policies deployed in uncontrolled real-world scenarios are faced with great in-category geometric diversity of everyday objects. In order to function robustly under such variations, policies need to work in a category-level manner, i.e. knowing how to interact with any object in a certain category, instead of only a specific one seen during training. This in-category generalizability is usually nurtured with shape-diversified training data; however, manually collecting such a corpus of data is infeasible due to the requirement of intense human labor and large collections of divergent objects at hand. In this paper, we propose ShapeGen, a data generation method that aims at generating shape-variated manipulation data in a simulator-free and 3D manner. ShapeGen decomposes the process into two stages: Shape Library curation and Function-Aware Generation. In the first stage, we train spatial warpings between shapes mapping points to points that correspond functionally, and aggregate 3D models along with the warpings into a plug-and-play Shape Library. In the second stage, we design a pipeline that, leveraging established Libraries, requires only minimal human annotation to generate physically plausible and functionally correct novel demonstrations. Experiments in the real world demonstrate the effectiveness of ShapeGen to boost policies' in-category shape generalizability. Project page: this https URL.
>
---
#### [new 005] Limits of Lamarckian Evolution Under Pressure of Morphological Novelty
- **分类: cs.RO**

- **简介: 该论文研究 Lamarckian 进化在形态新颖性压力下的局限性，比较其与达尔文进化的性能差异，旨在揭示继承与多样性之间的权衡。**

- **链接: [https://arxiv.org/pdf/2604.15854](https://arxiv.org/pdf/2604.15854)**

> **作者:** Jed R Muff; Karine Miras; A.E. Eiben
>
> **备注:** 8 pages, 7 figures, Submitted to WCCI 2026
>
> **摘要:** Lamarckian inheritance has been shown to be a powerful accelerator in systems where the joint evolution of robot morphologies and controllers is enhanced with individual learning. Its defining advantage lies in the offspring inheriting controllers learned by their parents. The efficacy of this option, however, relies on morphological similarity between parent and offspring. In this study, we examine how Lamarckian inheritance performs when the search process is driven toward high morphological variance, potentially straining the requirement for parent-offspring similarity. Using a system of modular robots that can evolve and learn to solve a locomotion task, we compare Darwinian and Lamarckian evolution to determine how they respond to shifting from pure task-based selection to a multi-objective pressure that also rewards morphological novelty. Our results confirm that Lamarckian evolution outperforms Darwinian evolution when optimizing task-performance alone. However, introducing selection pressure for morphological diversity causes a substantial performance drop, which is much greater in the Lamarckian system. Further analyses show that promoting diversity reduces parent-offspring similarity, which in turn reduces the benefits of inheriting controllers learned by parents. These results reveal the limits of Lamarckian evolution by exposing a fundamental trade-off between inheritance-based exploitation and diversity-driven exploration.
>
---
#### [new 006] Long-Term Memory for VLA-based Agents in Open-World Task Execution
- **分类: cs.RO**

- **简介: 该论文属于化学实验室自动化任务，解决VLA模型在长周期任务中的记忆与策略积累问题。提出ChemBot框架，集成双层记忆与协调机制，提升任务执行效率与成功率。**

- **链接: [https://arxiv.org/pdf/2604.15671](https://arxiv.org/pdf/2604.15671)**

> **作者:** Xu Huang; Weixin Mao; Yinhao Li; Hua Chen; Jiabao Zhao
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated significant potential for embodied decision-making; however, their application in complex chemical laboratory automation remains restricted by limited long-horizon reasoning and the absence of persistent experience accumulation. Existing frameworks typically treat planning and execution as decoupled processes, often failing to consolidate successful strategies, which results in inefficient trial-and-error in multi-stage protocols. In this paper, we propose ChemBot, a dual-layer, closed-loop framework that integrates an autonomous AI agent with a progress-aware VLA model (Skill-VLA) for hierarchical task decomposition and execution. ChemBot utilizes a dual-layer memory architecture to consolidate successful trajectories into retrievable assets, while a Model Context Protocol (MCP) server facilitates efficient sub-agent and tool orchestration. To address the inherent limitations of VLA models, we further implement a future-state-based asynchronous inference mechanism to mitigate trajectory discontinuities. Extensive experiments on collaborative robots demonstrate that ChemBot achieves superior operational safety, precision, and task success rates compared to existing VLA baselines in complex, long-horizon chemical experimentation.
>
---
#### [new 007] VADF: Vision-Adaptive Diffusion Policy Framework for Efficient Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出VADF框架，解决机器人操作中扩散策略的训练慢和推理失败问题。通过自适应损失网络和分层视觉任务分割，提升训练效率和推理成功率。**

- **链接: [https://arxiv.org/pdf/2604.15938](https://arxiv.org/pdf/2604.15938)**

> **作者:** Xinglei Yu; Zhenyang Liu; Shufeng Nan; Simo Wu; Yanwei Fu
>
> **摘要:** Diffusion policies are becoming mainstream in robotic manipulation but suffer from hard negative class imbalance due to uniform sampling and lack of sample difficulty awareness, leading to slow training convergence and frequent inference timeout failures. We propose VADF (Vision-Adaptive Diffusion Policy Framework), a vision-driven dual-adaptive framework that significantly reduces convergence steps and achieves early success in inference, with model-agnostic design enabling seamless integration into any diffusion policy architecture. During training, we introduce Adaptive Loss Network (ALN), a lightweight MLP-based loss predictor that quantifies per-step sample difficulty in real time. Guided by hard negative mining, it performs weighted sampling to prioritize high-loss regions, enabling adaptive weight updates and faster convergence. In inference, we design the Hierarchical Vision Task Segmenter (HVTS), which decomposes high-level task instructions into multi-stage low-level sub-instructions based on visual input. It adaptively segments action sequences into simple and complex subtasks by assigning shorter noise schedules with longer direct execution sequences to simple actions, and longer noise steps with shorter execution sequences to complex ones, thereby dramatically reducing computational overhead and significantly improving the early success rate.
>
---
#### [new 008] GaussianFlow SLAM: Monocular Gaussian Splatting SLAM Guided by GaussianFlow
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于SLAM任务，解决单目SLAM中缺乏几何线索的问题。提出GaussianFlow SLAM，利用光流引导场景结构和相机位姿优化，提升重建精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.15612](https://arxiv.org/pdf/2604.15612)**

> **作者:** Dong-Uk Seo; Jinwoo Jeon; Eungchang Mason Lee; Hyun Myung
>
> **备注:** 8 pages, 5 figures, 7 tables, accepted to IEEE RA-L
>
> **摘要:** Gaussian splatting has recently gained traction as a compelling map representation for SLAM systems, enabling dense and photo-realistic scene modeling. However, its application to monocular SLAM remains challenging due to the lack of reliable geometric cues from monocular input. Without geometric supervision, mapping or tracking could fall in local-minima, resulting in structural degeneracies and inaccuracies. To address this challenge, we propose GaussianFlow SLAM, a monocular 3DGS-SLAM that leverages optical flow as a geometry-aware cue to guide the optimization of both the scene structure and camera poses. By encouraging the projected motion of Gaussians, termed GaussianFlow, to align with the optical flow, our method introduces consistent structural cues to regularize both map reconstruction and pose estimation. Furthermore, we introduce normalized error-based densification and pruning modules to refine inactive and unstable Gaussians, thereby contributing to improved map quality and pose accuracy. Experiments conducted on public datasets demonstrate that our method achieves superior rendering quality and tracking accuracy compared with state-of-the-art algorithms. The source code is available at: this https URL.
>
---
#### [new 009] Robust Fleet Sizing for Multi-UAV Inspection Missions under Synchronized Replacement Demand
- **分类: cs.RO**

- **简介: 该论文属于多无人机任务规划领域，解决有限周期内无人机队列规模优化问题。针对同步替换需求导致的备用机不足问题，提出一种闭式 fleet-sizing 规则，提升任务可靠性。**

- **链接: [https://arxiv.org/pdf/2604.15890](https://arxiv.org/pdf/2604.15890)**

> **作者:** Vishal Ramesh; Antony Thomas
>
> **摘要:** Multi-UAV inspection missions require spare drones to replace active drones during recharging cycles. Existing fleet-sizing approaches often assume steady-state operating conditions that do not apply to finite-horizon missions, or they treat replacement requests as statistically independent events. The latter provides per-request blocking guarantees that fail to translate to mission-level reliability when demands cluster. This paper identifies a structural failure mode where efficient routing assigns similar workloads to each UAV, leading to synchronized battery depletion and replacement bursts that exhaust the spare pool even when average capacity is sufficient. We derive a closed-form sufficient fleet-sizing rule, k = m(ceil(R) + 1), where m is the number of active UAVs and R is the recovery-to-active time ratio. This additive buffer of m spares absorbs worst-case synchronized demand at recovery-cycle boundaries and ensures mission-level reliability even when all UAVs deplete simultaneously. Monte Carlo validation across five scenarios (m in [2, 10], R in [0.87, 3.39], 1000 trials each) shows that Erlang-B sizing with a per-request blocking target epsilon = 0.01 drops to 69.9% mission success at R = 3.39, with 95% of spare exhaustion events concentrated in the top-decile 5-minute demand windows. In contrast, the proposed rule maintains 99.8% success (Wilson 95% lower bound 99.3%) across all tested conditions, including wind variability up to CV = 0.30, while requiring only four additional drones in the most demanding scenario.
>
---
#### [new 010] Environment-Adaptive Solid-State LiDAR-Inertial Odometry
- **分类: cs.RO**

- **简介: 该论文属于LiDAR-Inertial定位任务，旨在解决极端环境下地图构建与定位不准确的问题。通过引入局部法向约束和退化感知的地图维护策略，提升定位精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.15864](https://arxiv.org/pdf/2604.15864)**

> **作者:** Zhi Zhang; Chalermchon Satirapod; Bingtao Ma; Changjun Gu
>
> **摘要:** Solid-state LiDAR-inertial SLAM has attracted significant attention due to its advantages in speed and robustness. However, achieving accurate mapping in extreme environments remains challenging due to severe geometric degeneracy and unreliable observations, which often lead to ill-conditioned optimization and map inconsistencies. To address these challenges, we propose an environment-adaptive solid-state LiDAR-inertial odometry that integrates local normal-vector constraints with degeneracy-aware map maintenance to enhance localization accuracy. Specifically, we introduce local normal-vector constraints to improve the stability of state estimation, effectively suppressing localization drift in degenerate scenarios. Furthermore, we design a degeneration-guided map update strategy to improve map precision. Benefiting from the refined map representation, localization accuracy is further enhanced in subsequent estimation. Experimental results demonstrate that the proposed method achieves superior mapping accuracy and robustness in extreme and perceptually degraded environments, with an average RMSE reduction of up to 12.8% compared to the baseline method.
>
---
#### [new 011] DENALI: A Dataset Enabling Non-Line-of-Sight Spatial Reasoning with Low-Cost LiDARs
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于非视距感知任务，旨在解决低成本LiDAR在隐藏物体感知中的难题。通过构建DENALI数据集，实现数据驱动的NLOS推理。**

- **链接: [https://arxiv.org/pdf/2604.16201](https://arxiv.org/pdf/2604.16201)**

> **作者:** Nikhil Behari; Diego Rivero; Luke Apostolides; Suman Ghosh; Paul Pu Liang; Ramesh Raskar
>
> **摘要:** Consumer LiDARs in mobile devices and robots typically output a single depth value per pixel. Yet internally, they record full time-resolved histograms containing direct and multi-bounce light returns; these multi-bounce returns encode rich non-line-of-sight (NLOS) cues that can enable perception of hidden objects in a scene. However, severe hardware limitations of consumer LiDARs make NLOS reconstruction with conventional methods difficult. In this work, we motivate a complementary direction: enabling NLOS perception with low-cost LiDARs through data-driven inference. We present DENALI, the first large-scale real-world dataset of space-time histograms from low-cost LiDARs capturing hidden objects. We capture time-resolved LiDAR histograms for 72,000 hidden-object scenes across diverse object shapes, positions, lighting conditions, and spatial resolutions. Using our dataset, we show that consumer LiDARs can enable accurate, data-driven NLOS perception. We further identify key scene and modeling factors that limit performance, as well as simulation-fidelity gaps that hinder current sim-to-real transfer, motivating future work toward scalable NLOS vision with consumer LiDARs.
>
---
#### [new 012] Fuzzy Logic Theory-based Adaptive Reward Shaping for Robust Reinforcement Learning (FARS)
- **分类: cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决高维状态空间中奖励稀疏导致的探索困难问题。通过引入基于模糊逻辑的自适应奖励 shaping 方法，提升学习稳定性与性能。**

- **链接: [https://arxiv.org/pdf/2604.15772](https://arxiv.org/pdf/2604.15772)**

> **作者:** Hürkan Şahin; Van Huyen Dang; Erdi Sayar; Alper Yegenoglu; Erdal Kayacan
>
> **备注:** 6 pages, 5 figures
>
> **摘要:** Reinforcement learning (RL) often struggles in real-world tasks with high-dimensional state spaces and long horizons, where sparse or fixed rewards severely slow down exploration and cause agents to get trapped in local optima. This paper presents a fuzzy logic based reward shaping method that integrates human intuition into RL reward design. By encoding expert knowledge into adaptive and interpreable terms, fuzzy rules promote stable learning and reduce sensitivity to hyperparameters. The proposed method leverages these properties to adapt reward contributions based on the agent state, enabling smoother transitions between fast motion and precise control in challenging navigation tasks. Extensive simulation results on autonomous drone racing benchmarks show stable learning behavior and consistent task performance across scenarios of increasing difficulty. The proposed method achieves faster convergence and reduced performance variability across training seeds in more challenging environments, with success rates improving by up to approximately 5 percent compared to non fuzzy reward formulations.
>
---
#### [new 013] Semantic Area Graph Reasoning for Multi-Robot Language-Guided Search
- **分类: cs.RO**

- **简介: 该论文属于多机器人语义导航任务，解决未知环境中语义搜索效率低的问题。提出SAGR框架，通过语义区域图实现机器人协同探索与目标搜索，提升任务效率。**

- **链接: [https://arxiv.org/pdf/2604.16263](https://arxiv.org/pdf/2604.16263)**

> **作者:** Ruiyang Wang; Hao-Lun Hsu; Jiwoo Kim; Miroslav Pajic
>
> **摘要:** Coordinating multi-robot systems (MRS) to search in unknown environments is particularly challenging for tasks that require semantic reasoning beyond geometric exploration. Classical coordination strategies rely on frontier coverage or information gain and cannot incorporate high-level task intent, such as searching for objects associated with specific room types. We propose \textit{Semantic Area Graph Reasoning} (SAGR), a hierarchical framework that enables Large Language Models (LLMs) to coordinate multi-robot exploration and semantic search through a structured semantic-topological abstraction of the environment. SAGR incrementally constructs a semantic area graph from a semantic occupancy map, encoding room instances, connectivity, frontier availability, and robot states into a compact task-relevant representation for LLM reasoning. The LLM performs high-level semantic room assignment based on spatial structure and task context, while deterministic frontier planning and local navigation handle geometric execution within assigned rooms. Experiments on the Habitat-Matterport3D dataset across 100 scenarios show that SAGR remains competitive with state-of-the-art exploration methods while consistently improving semantic target search efficiency, with up to 18.8\% in large environments. These results highlight the value of structured semantic abstractions as an effective interface between LLM-based reasoning and multi-robot coordination in complex indoor environments.
>
---
#### [new 014] From Seeing to Simulating: Generative High-Fidelity Simulation with Digital Cousins for Generalizable Robot Learning and Evaluation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人学习与评估任务，旨在解决真实数据采集成本高的问题。通过生成高保真模拟场景和多样化“数字双胞胎”，提升机器人学习的泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.15805](https://arxiv.org/pdf/2604.15805)**

> **作者:** Jasper Lu; Zhenhao Shen; Yuanfei Wang; Shugao Liu; Shengqiang Xu; Shawn Xie; Jingkai Xu; Feng Jiang; Jade Yang; Chen Xie; Ruihai Wu
>
> **摘要:** Learning robust robot policies in real-world environments requires diverse data augmentation, yet scaling real-world data collection is costly due to the need for acquiring physical assets and reconfiguring environments. Therefore, augmenting real-world scenes into simulation has become a practical augmentation for efficient learning and evaluation. We present a generative framework that establishes a generative real-to-sim mapping from real-world panoramas to high-fidelity simulation scenes, and further synthesize diverse cousin scenes via semantic and geometric editing. Combined with high-quality physics engines and realistic assets, the generated scenes support interactive manipulation tasks. Additionally, we incorporate multi-room stitching to construct consistent large-scale environments for long-horizon navigation across complex layouts. Experiments demonstrate a strong sim-to-real correlation validating our platform's fidelity, and show that extensively scaling up data generation leads to significantly better generalization to unseen scene and object variations, demonstrating the effectiveness of Digital Cousins for generalizable robot learning and evaluation.
>
---
#### [new 015] Contact-Aware Planning and Control of Continuum Robots in Highly Constrained Environments
- **分类: cs.RO; eess.SY; math.OC**

- **简介: 该论文属于机器人路径规划任务，解决连续机器人在受限环境中的接触问题。通过接触感知规划与控制，提升导航安全性和可靠性。**

- **链接: [https://arxiv.org/pdf/2604.15638](https://arxiv.org/pdf/2604.15638)**

> **作者:** Aedan Mangan; Kehan Long; Ki Myung Brian Lee; Miheer Potdar; Nikolay Atanasov; Tania K. Morimoto
>
> **备注:** 15 pages, 3 figures
>
> **摘要:** Continuum robots are well suited for navigating confined and fragile environments, such as vascular or endoluminal anatomy, where contact with surrounding structures is often unavoidable. While controlled contact can assist motion, unfavorable contact can degrade controllability, induce kinematic singularities, or introduce safety risks. We present a contact-aware planning approach that evaluates contact quality, penalizing hazardous interactions, while permitting benign contact. The planner produces kinematically feasible trajectories and contact-aware Jacobians which can be used for closed-loop control in hardware experiments. We validate the approach by testing the integrated system (planning, control, and mechanical design) on anatomical models from patient scans. The planner generates effective plans for three common anatomical environments, and, in all hardware trials, the continuum robot was able to reach the target while avoiding dangerous tip contact (100% success). Mean tracking errors were 1.9 +/- 0.5 mm, 1.2 +/- 0.1 mm, and 1.7 +/- 0.2 mm across the three different environments. Ablation studies showed that penalizing end-of-continuum-segment (ECS) contact improved manipulability and prevented hardware failures. Overall, this work enables reliable, contact-aware navigation in highly constrained environments.
>
---
#### [new 016] Trajectory Planning for Safe Dual Control with Active Exploration
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决在模型不确定性下的安全双控制问题。通过引入Dual-gatekeeper框架，在保证安全和预算约束下实现有效探索与任务执行的平衡。**

- **链接: [https://arxiv.org/pdf/2604.15507](https://arxiv.org/pdf/2604.15507)**

> **作者:** Kaleb Ben Naveed; Manveer Singh; Devansh R. Agrawal; Dimitra Panagou
>
> **摘要:** Planning safe trajectories under model uncertainty is a fundamental challenge. Robust planning ensures safety by considering worst-case realizations, yet ignores uncertainty reduction and leads to overly conservative behavior. Actively reducing uncertainty on-the-fly during a nominal mission defines the dual control problem. Most approaches address this by adding a weighted exploration term to the cost, tuned to trade off the nominal objective and uncertainty reduction, but without formal consideration of when exploration is beneficial. Moreover, safety is enforced in some methods but not in others. We study a budget-constrained dual control problem, where uncertainty is reduced subject to safety and a mission-level cost budget that limits the allowable degradation in task performance due to exploration. In this work, we propose Dual-gatekeeper, a framework that integrates robust planning with active exploration under formal guarantees of safety and budget feasibility. The key idea is that exploration is pursued only when it provides a verifiable improvement without compromising safety or violating the budget, enabling the system to balance immediate task performance with long-term uncertainty reduction in a principled manner. We provide two implementations of the framework based on different safety mechanisms and demonstrate its performance on quadrotor navigation and autonomous car racing case studies under parametric uncertainty.
>
---
#### [new 017] Factor Graph-Based Shape Estimation for Continuum Robots via Magnus Expansion
- **分类: cs.RO**

- **简介: 该论文属于机器人形状估计任务，旨在从稀疏噪声数据中重建连续机械臂的形状。通过结合几何变量应变参数化与因子图框架，提出一种高效且具有概率特性的方法。**

- **链接: [https://arxiv.org/pdf/2604.15619](https://arxiv.org/pdf/2604.15619)**

> **作者:** Lorenzo Ticozzi; Patricio A. Vela; Panagiotis Tsiotras
>
> **摘要:** Reconstructing the shape of continuum manipulators from sparse, noisy sensor data is a challenging task, owing to the infinite-dimensional nature of such systems. Existing approaches broadly trade off between parametric methods that yield compact state representations but lack probabilistic structure, and Cosserat rod inference on factor graphs, which provides principled uncertainty quantification at the cost of a state dimension that grows with the spatial discretization. This letter combines the strength of both paradigms by estimating the coefficients of a low-dimensional Geometric Variable Strain (GVS) parameterization within a factor graph framework. A novel kinematic factor, derived from the Magnus expansion of the strain field, encodes the closed-form rod geometry as a prior constraint linking the GVS strain coefficients to the backbone pose variables. The resulting formulation yields a compact state vector directly amenable to model-based control, while retaining the modularity, probabilistic treatment and computational efficiency of factor graph inference. The proposed method is evaluated in simulation on a 0.4 m long tendon-driven continuum robot under three measurement configurations, achieving mean position errors below 2 mm for all three scenarios and demonstrating a sixfold reduction in orientation error compared to a Gaussian process regression baseline when only position measurements are available.
>
---
#### [new 018] Foundation Models in Robotics: A Comprehensive Review of Methods, Models, Datasets, Challenges and Future Research Directions
- **分类: cs.RO**

- **简介: 该论文属于机器人学领域，综述了基础模型在机器人中的应用，解决如何提升机器人适应性和通用性的问题，分析了模型类型、架构、学习方法及挑战。**

- **链接: [https://arxiv.org/pdf/2604.15395](https://arxiv.org/pdf/2604.15395)**

> **作者:** Aggelos Psiris; Vasileios Argyriou; Evangelos K. Markakis; Panagiotis Sarigiannidis; Efstratios Gavves; Kostas Bekris; Arash Ajoudani adn Georgios Th. Papadopoulos
>
> **摘要:** Over the recent years, the field of robotics has been undergoing a transformative paradigm shift from fixed, single-task, domain-specific solutions towards adaptive, multi-function, general-purpose agents, capable of operating in complex, open-world, and dynamic environments. This tremendous advancement is primarily driven by the emergence of Foundation Models (FMs), i.e., large-scale neural-network architectures trained on massive, heterogeneous datasets that provide unprecedented capabilities in multi-modal understanding and reasoning, long-horizon planning, and cross-embodiment generalization. In this context, the current study provides a holistic, systematic, and in-depth review of the research landscape of FMs in robotics. In particular, the evolution of the field is initially delineated through five distinct research phases, spanning from the early incorporation of Natural Language Processing (NLP) and Computer Vision (CV) models to the current frontier of multi-sensory generalization and real-world deployment. Subsequently, a highly-granular taxonomic investigation of the literature is performed, examining the following key aspects: a) the employed FM types, including LLMs, VFMs, VLMs, and VLAs, b) the underlying neural-network architectures, c) the adopted learning paradigms, d) the different learning stages of knowledge incorporation, e) the major robotic tasks, and f) the main real-world application domains. For each aspect, comparative analysis and critical insights are provided. Moreover, a report on the publicly available datasets used for model training and evaluation across the considered robotic tasks is included. Furthermore, a hierarchical discussion on the current open challenges and promising future research directions in the field is incorporated.
>
---
#### [new 019] A Reconfigurable Pneumatic Joint Enabling Localized Selective Stiffening and Shape Locking in Vine-Inspired Robots
- **分类: cs.RO**

- **简介: 该论文属于软体机器人领域，旨在解决 vine- inspired 机器人在自由空间中刚度不足、形状保持差的问题。通过设计可重构气动关节，实现局部刚化与形状锁定，提升负载能力和运动性能。**

- **链接: [https://arxiv.org/pdf/2604.15907](https://arxiv.org/pdf/2604.15907)**

> **作者:** Ayodele James Oyejide; Ustaz A. Yaqub; Samir Erturk; Eray A. Baran; Fabio Stroppa
>
> **备注:** Original Article
>
> **摘要:** Vine-inspired robots achieve large workspace coverage through tip eversion, enabling safe navigation in confined and cluttered environments. However, their deployment in free space is fundamentally limited by low axial stiffness, poor load-bearing capacity, and the inability to retain shape during and after steering. In this work, we propose a reconfigurable pneumatic joint (RPJ) architecture that introduces discrete, pressure-tunable stiffness along the robot body without compromising continuous growth. Each RPJ module comprises symmetrically distributed pneumatic chambers that locally increase bending stiffness when pressurized, enabling decoupling between global compliance and localized rigidity. We integrate the RPJs into a soft growing robot with tendon-driven steering and develop a compact base station for mid-air eversion. System characterization and experimental validation demonstrate moderate pressure requirements for eversion, as well as comparable localized stiffening and steering performance to layer-jamming mechanisms. Demonstrations further show that the proposed robot achieves improved shape retention during bending, reduced gravitational deflection under load, cascading retraction, and reliable payload transport up to 202 g in free space. The RPJ mechanism establishes a practical pathway toward structurally adaptive vine robots for manipulation-oriented tasks such as object sorting and adaptive exploration in unconstrained environments.
>
---
#### [new 020] Iterated Invariant EKF for Quadruped Robot Odometry
- **分类: cs.RO**

- **简介: 该论文提出一种基于IterIEKF的腿部机器人状态估计算法，解决非线性系统下的定位问题，通过 proprioceptive 测量提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.15449](https://arxiv.org/pdf/2604.15449)**

> **作者:** Hilton Marques Souza Santana; João Carlos Virgolino Soares; Sven Goffin; Ylenia Nisticò; Silvère Bonnabel; Claudio Semini; Marco Antonio Meggiolaro
>
> **摘要:** Kalman filter-based algorithms are fundamental for mobile robots, as they provide a computationally efficient solution to the challenging problem of state estimation. However, they rely on two main assumptions that are difficult to satisfy in practice: (a) the system dynamics must be linear with Gaussian process noise, and (b) the measurement model must also be linear with Gaussian measurement noise. Previous works have extended assumption (a) to nonlinear spaces through the Invariant Extended Kalman Filter (IEKF), showing that it retains properties similar to those of the classical Kalman filter when the system dynamics are group-affine on a Lie group. More recently, the counterpart of assumption (b) for the same nonlinear setting was addressed in [1]. By means of the proposed Iterated Invariant Extended Kalman Filter (IterIEKF), the authors of that work demonstrated that the update step exhibits several compatibility properties of the classical linear Kalman filter. In this work, we introduce a novel open-source state estimation algorithm for legged robots based on the IterIEKF. The update step of the proposed filter relies solely on proprioceptive measurements, exploiting kinematic constraints on foot velocity during contact and base-frame velocity, making it inherently robust to environmental conditions. Through extensive numerical simulations and evaluation on real-world datasets, we demonstrate that the IterIEKF outperforms the vanilla IEKF, the SO(3)-based Kalman Filter, and its iterated variant in terms of both accuracy and consistency.
>
---
#### [new 021] GIST: Multimodal Knowledge Extraction and Spatial Grounding via Intelligent Semantic Topology
- **分类: cs.AI; cs.CV; cs.HC; cs.RO**

- **简介: 该论文提出GIST系统，解决复杂环境中AI的语义空间定位问题。通过多模态知识提取，构建语义导航拓扑，提升人机交互任务性能。**

- **链接: [https://arxiv.org/pdf/2604.15495](https://arxiv.org/pdf/2604.15495)**

> **作者:** Shivendra Agrawal; Bradley Hayes
>
> **摘要:** Navigating complex, densely packed environments like retail stores, warehouses, and hospitals poses a significant spatial grounding challenge for humans and embodied AI. In these spaces, dense visual features quickly become stale given the quasi-static nature of items, and long-tail semantic distributions challenge traditional computer vision. While Vision-Language Models (VLMs) help assistive systems navigate semantically-rich spaces, they still struggle with spatial grounding in cluttered environments. We present GIST (Grounded Intelligent Semantic Topology), a multimodal knowledge extraction pipeline that transforms a consumer-grade mobile point cloud into a semantically annotated navigation topology. Our architecture distills the scene into a 2D occupancy map, extracts its topological layout, and overlays a lightweight semantic layer via intelligent keyframe and semantic selection. We demonstrate the versatility of this structured spatial knowledge through critical downstream Human-AI interaction tasks: (1) an intent-driven Semantic Search engine that actively infers categorical alternatives and zones when exact matches fail; (2) a one-shot Semantic Localizer achieving a 1.04 m top-5 mean translation error; (3) a Zone Classification module that segments the walkable floor plan into high-level semantic regions; and (4) a Visually-Grounded Instruction Generator that synthesizes optimal paths into egocentric, landmark-rich natural language routing. In multi-criteria LLM evaluations, GIST outperforms sequence-based instruction generation baselines. Finally, an in-situ formative evaluation (N=5) yields an 80% navigation success rate relying solely on verbal cues, validating the system's capacity for universal design.
>
---
#### [new 022] SENSE: Stereo OpEN Vocabulary SEmantic Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于开放词汇语义分割任务，解决单视角图像精度不足的问题。通过引入立体视觉和视觉语言模型，提升分割准确性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.15946](https://arxiv.org/pdf/2604.15946)**

> **作者:** Thomas Campagnolo; Ezio Malis; Philippe Martinet; Gaétan Bahl
>
> **摘要:** Open-vocabulary semantic segmentation enables models to segment objects or image regions beyond fixed class sets, offering flexibility in dynamic environments. However, existing methods often rely on single-view images and struggle with spatial precision, especially under occlusions and near object boundaries. We propose SENSE, the first work on Stereo OpEN Vocabulary SEmantic Segmentation, which leverages stereo vision and vision-language models to enhance open-vocabulary semantic segmentation. By incorporating stereo image pairs, we introduce geometric cues that improve spatial reasoning and segmentation accuracy. Trained on the PhraseStereo dataset, our approach achieves strong performance in phrase-grounded tasks and demonstrates generalization in zero-shot settings. On PhraseStereo, we show a +2.9% improvement in Average Precision over the baseline method and +0.76% over the best competing method. SENSE also provides a relative improvement of +3.5% mIoU on Cityscapes and +18% on KITTI compared to the baseline work. By jointly reasoning over semantics and geometry, SENSE supports accurate scene understanding from natural language, essential for autonomous robots and Intelligent Transportation Systems.
>
---
#### [new 023] PLAF: Pixel-wise Language-Aligned Feature Extraction for Efficient 3D Scene Understanding
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D场景理解任务，旨在解决语义表示在2D与3D间语言对齐和冗余问题。提出PLAF框架，实现像素级语义对齐与高效存储查询。**

- **链接: [https://arxiv.org/pdf/2604.15770](https://arxiv.org/pdf/2604.15770)**

> **作者:** Junjie Wen; Junlin He; Fei Ma; Jinqiang Cui
>
> **备注:** Accepted by ICCA 2026
>
> **摘要:** Accurate open-vocabulary 3D scene understanding requires semantic representations that are both language-aligned and spatially precise at the pixel level, while remaining scalable when lifted to 3D space. However, existing representations struggle to jointly satisfy these requirements, and densely propagating pixel-wise semantics to 3D often results in substantial redundancy, leading to inefficient storage and querying in large-scale scenes. To address these challenges, we present \emph{PLAF}, a Pixel-wise Language-Aligned Feature extraction framework that enables dense and accurate semantic alignment in 2D without sacrificing open-vocabulary expressiveness. Building upon this representation, we further design an efficient semantic storage and querying scheme that significantly reduces redundancy across both 2D and 3D domains. Experimental results show that \emph{PLAF} provides a strong semantic foundation for accurate and efficient open-vocabulary 3D scene understanding. The codes are publicly available at this https URL.
>
---
#### [new 024] FineCog-Nav: Integrating Fine-grained Cognitive Modules for Zero-shot Multimodal UAV Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出FineCog-Nav，解决零样本多模态无人机导航问题。通过细粒度认知模块整合，提升导航准确性和可解释性。**

- **链接: [https://arxiv.org/pdf/2604.16298](https://arxiv.org/pdf/2604.16298)**

> **作者:** Dian Shao; Zhengzheng Xu; Peiyang Wang; Like Liu; Yule Wang; Jieqi Shi; Jing Huo
>
> **备注:** Accepted by CVPR 2026 Findings
>
> **摘要:** UAV vision-language navigation (VLN) requires an agent to navigate complex 3D environments from an egocentric perspective while following ambiguous multi-step instructions over long horizons. Existing zero-shot methods remain limited, as they often rely on large base models, generic prompts, and loosely coordinated modules. In this work, we propose FineCog-Nav, a top-down framework inspired by human cognition that organizes navigation into fine-grained modules for language processing, perception, attention, memory, imagination, reasoning, and decision-making. Each module is driven by a moderate-sized foundation model with role-specific prompts and structured input-output protocols, enabling effective collaboration and improved interpretability. To support fine-grained evaluation, we construct AerialVLN-Fine, a curated benchmark of 300 trajectories derived from AerialVLN, with sentence-level instruction-trajectory alignment and refined instructions containing explicit visual endpoints and landmark references. Experiments show that FineCog-Nav consistently outperforms zero-shot baselines in instruction adherence, long-horizon planning, and generalization to unseen environments. These results suggest the effectiveness of fine-grained cognitive modularization for zero-shot aerial navigation. Project page: this https URL.
>
---
#### [new 025] Safe and Energy-Aware Multi-Robot Density Control via PDE-Constrained Optimization for Long-Duration Autonomy
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于多机器人系统控制任务，解决长期自主下的密度控制问题。通过PDE约束优化实现安全与能耗保障，确保密度跟踪与避障。**

- **链接: [https://arxiv.org/pdf/2604.15524](https://arxiv.org/pdf/2604.15524)**

> **作者:** Longchen Niu; Andrew Nasif; Gennaro Notomista
>
> **摘要:** This paper presents a novel density control framework for multi-robot systems with spatial safety and energy sustainability guarantees. Stochastic robot motion is encoded through the Fokker-Planck Partial Differential Equation (PDE) at the density level. Control Lyapunov and control barrier functions are integrated with PDEs to enforce target density tracking, obstacle region avoidance, and energy sufficiency over multiple charging cycles. The resulting quadratic program enables fast in-the-loop implementation that adjusts commands in real-time. Multi-robot experiment and extensive simulations were conducted to demonstrate the effectiveness of the controller under localization and motion uncertainties.
>
---
#### [new 026] Continual Hand-Eye Calibration for Open-world Robotic Manipulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人手眼标定任务，解决开放场景下模型遗忘问题。提出SARS和SPDD方法，实现持续标定与知识保留。**

- **链接: [https://arxiv.org/pdf/2604.15814](https://arxiv.org/pdf/2604.15814)**

> **作者:** Fazeng Li; Gan Sun; Chenxi Liu; Yao He; Wei Cong; Yang Cong
>
> **摘要:** Hand-eye calibration through visual localization is a critical capability for robotic manipulation in open-world environments. However, most deep learning-based calibration models suffer from catastrophic forgetting when adapting into unseen data amongst open-world scene changes, while simple rehearsal-based continual learning strategy cannot well mitigate this issue. To overcome this challenge, we propose a continual hand-eye calibration framework, enabling robots to adapt to sequentially encountered open-world manipulation scenes through spatially replay strategy and structure-preserving distillation. Specifically, a Spatial-Aware Replay Strategy (SARS) constructs a geometrically uniform replay buffer that ensures comprehensive coverage of each scene pose space, replacing redundant adjacent frames with maximally informative viewpoints. Meanwhile, a Structure-Preserving Dual Distillation (SPDD) is proposed to decompose localization knowledge into coarse scene layout and fine pose precision, and distills them separately to alleviate both types of forgetting during continual adaptation. As a new manipulation scene arrives, SARS provides geometrically representative replay samples from all prior scenes, and SPDD applies structured distillation on these samples to retain previously learned knowledge. After training on the new scene, SARS incorporates selected samples from the new scene into the replay buffer for future rehearsal, allowing the model to continuously accumulate multi-scene calibration capability. Experiments on multiple public datasets show significant anti scene forgetting performance, maintaining accuracy on past scenes while preserving adaptation to new scenes, confirming the effectiveness of the framework.
>
---
#### [new 027] $π_{0.7}$: a Steerable Generalist Robotic Foundation Model with Emergent Capabilities
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出一种新型机器人基础模型π₀.₇，解决机器人在多样化任务中的泛化与自主执行问题。通过多模态条件训练，使模型能精准执行复杂任务，无需额外微调。**

- **链接: [https://arxiv.org/pdf/2604.15483](https://arxiv.org/pdf/2604.15483)**

> **作者:** Physical Intelligence; Bo Ai; Ali Amin; Raichelle Aniceto; Ashwin Balakrishna; Greg Balke; Kevin Black; George Bokinsky; Shihao Cao; Thomas Charbonnier; Vedant Choudhary; Foster Collins; Ken Conley; Grace Connors; James Darpinian; Karan Dhabalia; Maitrayee Dhaka; Jared DiCarlo; Danny Driess; Michael Equi; Adnan Esmail; Yunhao Fang; Chelsea Finn; Catherine Glossop; Thomas Godden; Ivan Goryachev; Lachlan Groom; Haroun Habeeb; Hunter Hancock; Karol Hausman; Gashon Hussein; Victor Hwang; Brian Ichter; Connor Jacobsen; Szymon Jakubczak; Rowan Jen; Tim Jones; Gregg Kammerer; Ben Katz; Liyiming Ke; Mairbek Khadikov; Chandra Kuchi; Marinda Lamb; Devin LeBlanc; Brendon LeCount; Sergey Levine; Xinyu Li; Adrian Li-Bell; Vladislav Lialin; Zhonglin Liang; Wallace Lim; Yao Lu; Enyu Luo; Vishnu Mano; Nandan Marwaha; Aikys Mongush; Liam Murphy; Suraj Nair; Tyler Patterson; Karl Pertsch; Allen Z. Ren; Gavin Schelske; Charvi Sharma; Baifeng Shi; Lucy Xiaoyang Shi; Laura Smith; Jost Tobias Springenberg; Kyle Stachowicz; Will Stoeckle; Jiaming Tang; Jimmy Tanner; Shalom Tekeste; Marcel Torne; Kyle Vedder; Quan Vuong; Anna Walling; Haohuan Wang; Jason Wang; XuDong Wang; Chris Whalen; Samuel Whitmore; Blake Williams; Charles Xu; Sukwon Yoo; Lili Yu; Wuming Zhang; Zhuoyang Zhang; Ury Zhilinsky
>
> **备注:** Website: this https URL
>
> **摘要:** We present a new robotic foundation model, called ${\pi}_{0.7}$, that can enable strong out-of-the-box performance in a wide range of scenarios. ${\pi}_{0.7}$ can follow diverse language instructions in unseen environments, including multi-stage tasks with various kitchen appliances, provide zero-shot cross-embodiment generalization, for example enabling a robot to fold laundry without seeing the task before, and perform challenging tasks such as operating an espresso machine out of the box at a level of performance that matches much more specialized RL-finetuned models. The main idea behind ${\pi}_{0.7}$ is to use diverse context conditioning during training. This conditioning information, contained in the prompt, makes it possible to steer the model precisely to perform many tasks with different strategies. It is conditioned not just on a language command that describes what it should do, but on additional multimodal information that also describes the manner or strategy in which it should do it, including metadata about task performance and subgoal images. This enables ${\pi}_{0.7}$ to use very diverse data, including demonstrations, potentially suboptimal (autonomous) data including failures, and data from non-robot sources. Our experiments evaluate ${\pi}_{0.7}$ across numerous tasks with multiple robot platforms, on tasks that require speed and dexterity, language following, and compositional task generalization.
>
---
#### [new 028] NEFFY 2.0: A Breathing Companion Robot: User-Centered Design and Findings from a Study with Ukrainian Refugees
- **分类: cs.HC; cs.ET; cs.RO**

- **简介: 本文介绍NEFFY 2.0机器人，用于压力缓解的呼吸训练。针对乌克兰难民进行用户研究，比较机器人辅助与纯音频干预的效果，探索其在压力管理中的应用价值。**

- **链接: [https://arxiv.org/pdf/2604.15325](https://arxiv.org/pdf/2604.15325)**

> **作者:** Ilona Buchem; Jessica Kazubski; Charly Goerke
>
> **备注:** 5 pages, 1 figure, 1st ACM/IEEE International Conference on Human-Robot Interaction
>
> **摘要:** This paper presents the design of NEFFY 2.0, a social robot designed as a haptic slow-paced breathing companion for stress reduction, and reports findings from a mixed-methods user study with 14 refugees from Ukraine. Developed through a user-centered design process, NEFFY 2.0 builds on NEFFY 1.0 and integrates embodiment and multi-sensory interaction to provide low-threshold, accessible guidance of slow-paced breathing for stress relief, which may be particularly valuable for individuals experiencing prolonged periods of anxiety. To evaluate effectiveness, an experimental comparison of a robot-assisted breathing intervention versus an audio-only condition was conducted. Measures included subjective ratings and physiological indicators, such as heart rate (HR), heart rate variability (HRV) using RMSSD parameter, respiratory rate (RR), and galvanic skin response (GSR), alongside qualitative data from interviews exploring user experience and perceived support. Qualitative findings showed that NEFFY 2.0 was perceived as intuitive, calming and supportive. Survey results showed a substantially larger effect in significant reduction of perceived stress in the NEFFY 2.0 condition compared to audio-only. Physiological data reveled mixed results combined with large inter-personal variability. Three patterns of breathing practice with NEFFY 2.0 were identified using k-means clustering. Despite the small sample size, this study makes a novel contribution by providing empirical evidence of stress reduction in a vulnerable population through a direct comparison of robot-assisted and non-robot conditions. The findings position NEFFY 2.0 as a promising low-threshold tool that supports stress relief and contributes to the vision of HRI empowering society.
>
---
#### [new 029] Uncertainty, Vagueness, and Ambiguity in Human-Robot Interaction: Why Conceptualization Matters
- **分类: cs.HC; cs.AI; cs.RO**

- **简介: 该论文属于概念分析任务，旨在解决HRI中不确定性、模糊性和歧义性概念混淆的问题，通过定义和区分这些概念，为后续研究提供清晰的理论基础。**

- **链接: [https://arxiv.org/pdf/2604.15339](https://arxiv.org/pdf/2604.15339)**

> **作者:** Xiaowen Sun; Cornelius Weber; Matthias Kerzel; Josua Spisak; Stefan Wermter
>
> **备注:** Accepted to InterAI@HRI'26
>
> **摘要:** Uncertainty, vagueness, and ambiguity are closely related and often confused concepts in human-robot interaction (HRI). In earlier studies, these concepts have been defined in contradictory ways and described using inconsistent terminology. This conceptual confusion and lack of terminological consistency undermine empirical comparability, thereby slowing the accumulation of theory. Consequently, consistent concepts that clarify these challenges, including their definitions, distinctions, and interrelationships, are needed in HRI. To address this lack of clarity, this paper proposes a consistent conceptual foundation for the challenges of uncertainty, vagueness, and ambiguity in HRI. First, we examine the meanings of these three terms in dictionaries. We then analyze the nature of their distinctions and interrelationships within the context of HRI. We further illustrate these characteristics through examples. Finally, we demonstrate how this consistent conceptual foundation facilitates the design of novel methods and the evaluation of existing methodologies for these phenomena.
>
---
## 更新

#### [replaced 001] COVER:COverage-VErified Roadmaps for Fixed-time Motion Planning in Continuous Semi-Static Environments
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，解决半静态环境中固定时间内的路径规划问题。提出COVER框架，构建覆盖验证的路线图，提升规划成功率和覆盖范围。**

- **链接: [https://arxiv.org/pdf/2510.03875](https://arxiv.org/pdf/2510.03875)**

> **作者:** Niranjan Kumar Ilampooranan; Constantinos Chamzas
>
> **摘要:** The ability to solve motion-planning queries within a fixed time budget is critical for deploying robotic systems in time-sensitive applications. Semi-static environments, where most of the workspace remains fixed while a subset of obstacles varies between tasks, exhibit structured variability that can be exploited to provide stronger guarantees than general-purpose planners. However, existing approaches either lack formal coverage guarantees or rely on discretizations of obstacle configurations that restrict applicability to realistic domains. This paper introduces COVER, a framework that incrementally constructs coverage-verified roadmaps for semi-static environments. COVER decomposes the arrangement space by independently partitioning the configuration space of each movable obstacle and verifies roadmap feasibility within each partition, enabling fixed-time query resolution for verified this http URL evaluate COVER on a 7-DoF manipulator performing object-picking in tabletop and shelf environments, demonstrating broader problem-space coverage and higher query success rates than prior work, particularly with obstacles of different sizes.
>
---
#### [replaced 002] Sampling-Based Multi-Modal Multi-Robot Multi-Goal Path Planning
- **分类: cs.RO**

- **简介: 该论文研究多机器人多目标路径规划问题，旨在解决传统方法非最优、不完整的缺陷。提出一种基于采样的集中式规划方法，实现概率完备和渐近最优的路径规划。**

- **链接: [https://arxiv.org/pdf/2503.03509](https://arxiv.org/pdf/2503.03509)**

> **作者:** Valentin N. Hartmann; Tirza Heinle; Yijiang Huang; Stelian Coros
>
> **备注:** 25 pages, 17 figures
>
> **摘要:** In many robotics applications, multiple robots are working in a shared workspace to complete a set of tasks as fast as possible. Such settings can be treated as multi-modal multi-robot multi-goal path planning problems, where each robot has to reach a set of goals. Existing approaches to this type of problem solve this using prioritization or assume synchronous task completion, and are thus neither optimal nor complete. We formalize this problem as a single centralized path planning problem and present planners that are probabilistically complete and asymptotically optimal. The planners plan in the composite space of all robots and are modifications of standard sampling-based planners with the required changes to work in our multi-modal, multi-robot, multi-goal setting. We validate the planners on a diverse range of problems including scenarios with various robots, planning horizons, and collaborative tasks such as handovers, and compare the planners against a suboptimal prioritized planner. Videos and code for the planners and the benchmark is available at this https URL.
>
---
#### [replaced 003] Optimal Solutions for the Moving Target Vehicle Routing Problem with Obstacles via Lazy Branch and Price
- **分类: cs.RO**

- **简介: 该论文研究MT-VRP-O任务，解决多智能体在障碍物环境中最优路径规划问题，提出Lazy BPRC算法加速求解。**

- **链接: [https://arxiv.org/pdf/2603.21880](https://arxiv.org/pdf/2603.21880)**

> **作者:** Anoop Bhat; Geordan Gutow; Surya Singh; Zhongqiang Ren; Sivakumar Rathinam; Howie Choset
>
> **摘要:** The Moving Target Vehicle Routing Problem with Obstacles (MT-VRP-O) seeks trajectories for several agents that collectively intercept a set of moving targets. Each target has one or more time windows where it must be visited, and the agents must avoid static obstacles and satisfy speed and capacity constraints. We introduce Lazy Branch-and-Price with Relaxed Continuity (Lazy BPRC), which finds optimal solutions for the MT-VRP-O. Lazy BPRC applies the branch-and-price framework for VRPs, which alternates between a restricted master problem (RMP) and a pricing problem. The RMP aims to select a sequence of target-time window pairings (called a tour) for each agent to follow, from a limited subset of tours. The pricing problem adds tours to the limited subset. Conventionally, solving the RMP requires computing the cost for an agent to follow each tour in the limited subset. Computing these costs in the MT-VRP-O is computationally intensive, since it requires collision-free motion planning between moving targets. Lazy BPRC defers cost computations by solving the RMP using lower bounds on the costs of each tour, computed via motion planning with relaxed continuity constraints. We lazily evaluate the true costs of tours as-needed. We compute a tour's cost by searching for a shortest path on a Graph of Convex Sets (GCS), and we accelerate this search using our continuity relaxation method. We demonstrate that Lazy BPRC runs up to an order of magnitude faster than two ablations.
>
---
#### [replaced 004] CLAW: Composable Language-Annotated Whole-body Motion Generation
- **分类: cs.RO**

- **简介: 该论文提出CLAW系统，用于生成人形机器人语言标注的全身运动数据。解决运动-语言数据不足问题，通过组合运动基元和生成自然语言注释，实现物理可行的运动生成。**

- **链接: [https://arxiv.org/pdf/2604.11251](https://arxiv.org/pdf/2604.11251)**

> **作者:** Jianuo Cao; Yuxin Chen; Masayoshi Tomizuka
>
> **摘要:** Training language-conditioned whole-body controllers for humanoid robots demands large-scale motion-language datasets. Existing approaches based on motion capture are costly and limited in diversity, while text-to-motion generative models produce purely kinematic outputs that are not guaranteed to be physically feasible. We present CLAW, a pipeline for scalable generation of language-annotated whole-body motion data for the Unitree G1 humanoid robot. CLAW composes motion primitives from a kinematic planner, parameterized by movement, heading, speed, pelvis height, and duration, and provides two browser-based interfaces--a real-time keyboard mode and a timeline-based sequence editor--for exploratory and batch data collection. A low-level controller tracks these references in MuJoCo simulation, yielding physically grounded trajectories. In parallel, a template-based engine generates diverse natural-language annotations at both segment and trajectory levels. To support scalable generation of motion-language paired data for humanoid robot learning, we make our system publicly available at: this https URL
>
---
#### [replaced 005] Soft Electroadhesive Feet for Micro Aerial Robots Perching on Smooth and Curved Surfaces
- **分类: cs.RO**

- **简介: 该论文属于微飞行器定点任务，解决其在光滑曲面稳定附着的问题。通过设计软电粘附足，实现高效附着与快速脱离。**

- **链接: [https://arxiv.org/pdf/2604.09270](https://arxiv.org/pdf/2604.09270)**

> **作者:** Chen Liu; Sonu Feroz; Ketao Zhang
>
> **备注:** 7 pages, 8 figures
>
> **摘要:** Electroadhesion (EA) provides electrically switchable adhesion and is a promising mechanism for perching micro aerial robots on smooth surfaces. However, practical implementations of soft and stretchable EA pads for aerial perching remain limited. This work presents (i) an efficient workflow for fabricating soft, stretchable electroadhesive pads with sinusoidal wave and concentric-circle electrodes in multiple sizes, (ii) a controlled experimental comparison of normal and shear adhesion under inactive (0 kV) and active (4.8 kV) conditions using an Instron-based setup, and (iii) a perching demonstration using a Crazyflie quadrotor equipped with electroadhesive feet on flat and curved substrates. Experimental results show that shear adhesion dominates, reaching forces on the order of 3 N with partial pad contact, while normal adhesion is comparatively small and strongly dependent on substrate properties. The Crazyflie prototype demonstrates repeatable attachment on smooth plastic surfaces, including curved geometries, as well as rapid detachment when the voltage is removed. These results highlight the potential of soft electroadhesive feet for lightweight and reliable perching in micro aerial vehicles (MAVs).
>
---
#### [replaced 006] Robust Real-Time Coordination of CAVs: A Distributed Optimization Framework under Uncertainty
- **分类: cs.RO**

- **简介: 该论文属于智能交通任务，解决动态环境下协同车辆的实时协调问题。提出一种分布式优化框架，通过轨迹分布控制、并行ADMM算法和交互注意力机制，提升安全性和计算效率。**

- **链接: [https://arxiv.org/pdf/2508.21322](https://arxiv.org/pdf/2508.21322)**

> **作者:** Haojie Bai; Tingting Zhang; Cong Guo; Yang Wang; Xiongwei Zhao; Hai Zhu
>
> **备注:** Accept by IEEE TVT
>
> **摘要:** Achieving both safety guarantees and real-time performance in cooperative vehicle coordination remains a fundamental challenge, particularly in dynamic and uncertain environments. Existing methods often suffer from insufficient uncertainty treatment in safety modeling, which intertwines with the heavy computational burden under complex multi-vehicle coupling. This paper presents a novel coordination framework that resolves this challenge through three key innovations: 1) direct control of vehicles' trajectory distributions during coordination, formulated as a robust cooperative planning problem with adaptive enhanced safety constraints, ensuring a specified level of safety regarding the uncertainty of the interactive trajectory, 2) a fully parallel ADMM-based distributed trajectory negotiation (ADMM-DTN) algorithm that efficiently solves the optimization problem while allowing configurable negotiation rounds to balance solution quality and computational resources, and 3) an interactive attention mechanism that selectively focuses on critical interactive participants to further enhance computational efficiency. Simulation results demonstrate that our framework achieves significant advantages in safety (reducing collision rates by up to 40.79\% in various scenarios) and real-time performance compared to representative benchmarks, while maintaining strong scalability with increasing vehicle numbers. The proposed interactive attention mechanism further reduces the computational demand by 15.4\%. Real-world experiments further validate robustness and real-time feasibility with unexpected dynamic obstacles, demonstrating reliable coordination in complex traffic scenes. The experiment demo could be found at this https URL.
>
---
#### [replaced 007] AutoDrive-R$^2$: Incentivizing Reasoning and Self-Reflection Capacity for VLA Model in Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自主驾驶任务，旨在提升VLA模型的推理与自省能力。通过构建CoT数据集和GRPO算法，增强决策过程的可解释性与动作序列合理性。**

- **链接: [https://arxiv.org/pdf/2509.01944](https://arxiv.org/pdf/2509.01944)**

> **作者:** Zhenlong Yuan; Chengxuan Qian; Jing Tang; Rui Chen; Zijian Song; Lei Sun; Xiangxiang Chu; Yujun Cai; Dapeng Zhang; Shuo Li
>
> **摘要:** Vision-Language-Action (VLA) models in autonomous driving systems have recently demonstrated transformative potential by integrating multimodal perception with decision-making capabilities. However, the interpretability and coherence of the decision process and the plausibility of action sequences remain largely underexplored. To address these issues, we propose AutoDrive-R$^2$, a novel VLA framework that enhances both reasoning and self-reflection capabilities of autonomous driving systems through chain-of-thought (CoT) processing and reinforcement learning (RL). Specifically, we first propose an innovative CoT dataset named nuScenesR$^2$-6K for supervised fine-tuning, which effectively builds cognitive bridges between input information and output trajectories through a four-step logical chain with self-reflection for validation. Moreover, to maximize both reasoning and self-reflection during the RL stage, we further employ the Group Relative Policy Optimization (GRPO) algorithm within a physics-grounded reward framework that incorporates spatial alignment, vehicle dynamic, and temporal smoothness criteria to ensure reliable and realistic trajectory planning. Extensive evaluation results across both nuScenes and Waymo datasets demonstrates the state-of-the-art performance and robust generalization capacity of our proposed method.
>
---
#### [replaced 008] Special Unitary Parameterized Estimators of Rotation
- **分类: cs.RO**

- **简介: 该论文属于旋转估计任务，解决如何用特殊酉矩阵表示旋转的问题。通过重构瓦尔巴问题，提出新的参数化方法及神经网络中的旋转学习模型。**

- **链接: [https://arxiv.org/pdf/2411.13109](https://arxiv.org/pdf/2411.13109)**

> **作者:** Akshay Chandrasekhar
>
> **备注:** Final version to be published at ICLR 2026; 33 pages
>
> **摘要:** This paper revisits the topic of rotation estimation through the lens of special unitary matrices. We begin by reformulating Wahba's problem using $SU(2)$ to derive multiple solutions that yield linear constraints on corresponding quaternion parameters. We then explore applications of these constraints by formulating efficient methods for related problems. Finally, from this theoretical foundation, we propose two novel continuous representations for learning rotations in neural networks. Extensive experiments validate the effectiveness of the proposed methods.
>
---
#### [replaced 009] Optimal Solutions for the Moving Target Vehicle Routing Problem via Branch-and-Price with Relaxed Continuity
- **分类: cs.RO**

- **简介: 该论文研究MT-VRP任务，解决移动目标的路径规划问题。提出BPRC算法，通过改进的标签算法高效求解，提升求解速度。**

- **链接: [https://arxiv.org/pdf/2603.00663](https://arxiv.org/pdf/2603.00663)**

> **作者:** Anoop Bhat; Geordan Gutow; Zhongqiang Ren; Sivakumar Rathinam; Howie Choset
>
> **备注:** Accepted to ICAPS 2026
>
> **摘要:** The Moving Target Vehicle Routing Problem (MT-VRP) seeks trajectories for several agents that intercept a set of moving targets, subject to speed, time window, and capacity constraints. We introduce an exact algorithm, Branch-and-Price with Relaxed Continuity (BPRC), for the MT-VRP. The main challenge in a branch-and-price approach for the MT-VRP is the pricing subproblem, which is complicated by moving targets and time-dependent travel costs between targets. Our key contribution is a new labeling algorithm that solves this subproblem by means of a novel dominance criterion tailored for problems with moving targets. Numerical results on instances with up to 25 targets show that our algorithm finds optimal solutions more than an order of magnitude faster than a baseline based on previous work, showing particular strength in scenarios with limited agent capacities.
>
---
#### [replaced 010] Angle-based Localization and Rigidity Maintenance Control for Multi-Robot Networks
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于多机器人系统任务，解决角度定位与刚性维持问题。通过建立角度刚性与方位刚性的关系，提出分布式角度定位方案和保持刚性的控制方法。**

- **链接: [https://arxiv.org/pdf/2604.11754](https://arxiv.org/pdf/2604.11754)**

> **作者:** J. Francisco Presenza; Leonardo J. Colombo; Juan I. Giribet; Ignacio Mas
>
> **摘要:** In this work, we study angle-based localization and rigidity maintenance control for multi-robot networks. First, we establish the relationship between angle rigidity and bearing rigidity considering \textit{directed} sensing graphs and \textit{body-frame} bearing measurements in both $2$ and $3$-\textit{dimensional space}. In particular, we demonstrate that a framework in $\mathrm{SE}(d)$ is infinitesimally bearing rigid if and only if it is infinitesimally angle rigid and each robot obtains at least $d-1$ bearing measurements ($d \in \{2, 3\}$). Building on these findings, this paper proposes a distributed angle-based localization scheme and establishes local exponential stability under switching sensing graphs, requiring only infinitesimal angle rigidity across the visited topologies. Then, since the set of available angles strongly depends on the robots' spatial configuration due to sensing constraints, we investigate rigidity maintenance control. The \textit{angle rigidity eigenvalue} is presented as a metric for the degree of rigidity. A decentralized gradient-based controller capable of executing mission-specific commands while maintaining a sufficient level of angle rigidity is proposed. Simulations were conducted to evaluate the scheme's effectiveness and practicality.
>
---
#### [replaced 011] Time-optimal Convexified Reeds-Shepp Paths on a Sphere
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究球面上凸化Reeds-Shepp车辆的时间最优路径规划问题，解决在转向率受限下的运动规划任务，提出六段式最优路径结构及23种路径类型。**

- **链接: [https://arxiv.org/pdf/2504.00966](https://arxiv.org/pdf/2504.00966)**

> **作者:** Sixu Li; Deepak Prakash Kumar; Swaroop Darbha; Yang Zhou
>
> **摘要:** This article studies the time-optimal path planning problem for a convexified Reeds-Shepp (CRS) vehicle on a unit sphere, capable of both forward and backward motion, with speed bounded in magnitude by 1 and turning rate bounded in magnitude by a given constant. For the case in which the turning-rate bound is at least 1, using Pontryagin's Maximum Principle and a phase-portrait analysis, we show that the optimal path connecting a given initial configuration to a desired terminal configuration consists of at most six segments drawn from three motion primitives: tight turns, great circular arcs, and turn-in-place motions. A complete classification yields a finite sufficient list of 23 optimal path types with closed-form segment angles derived. The complementary case in which the turning-rate bound is less than 1 is addressed via an equivalent reformulation. The proposed formulation is applicable to underactuated satellite attitude control, spherical rolling robots, and mobile robots operating on spherical or gently curved surfaces. The source code for solving the time-optimal path problem and visualization is publicly available at this https URL.
>
---
#### [replaced 012] ArrayTac: A Closed-loop Piezoelectric Tactile Platform for Continuously Tunable Rendering of Shape, Stiffness, and Friction
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文属于触觉显示任务，旨在解决现有设备无法连续调控形状、刚度和摩擦的问题。研究提出ArrayTac系统，实现多维触觉渲染与精准感知。**

- **链接: [https://arxiv.org/pdf/2603.13829](https://arxiv.org/pdf/2603.13829)**

> **作者:** Tianhai Liang; Shiyi Guo; Baiye Cheng; Zhengrong Xue; Han Zhang; Huazhe Xu
>
> **备注:** Project website: this https URL
>
> **摘要:** Human touch depends on the integration of shape, stiffness, and friction, yet existing tactile displays cannot render these cues together as continuously tunable, high-fidelity signals for intuitive perception. We present ArrayTac, a closed-loop piezoelectric tactile display that simultaneously renders these three dimensions with continuous tunability on a 4 by 4 actuator array. Each unit integrates a three-stage micro-lever amplifier with end-effector Hall-effect feedback, enabling up to 5 mm displacement, greater than 500 Hz array refresh, and 123 Hz closed-loop bandwidth. In psychophysical experiments, naive participants identified three-dimensional shapes and distinguished multiple stiffness and friction levels through touch alone without training. We further demonstrate image-to-touch rendering from an RGB image and remote palpation of a medical-grade breast tumor phantom over 1,000 km, in which all 11 naive participants correctly identified tumor number and type with sub-centimeter localization error. These results establish ArrayTac as a platform for multidimensional haptic rendering and interaction.
>
---
#### [replaced 013] VeriGraph: Scene Graphs for Execution Verifiable Robot Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出VeriGraph框架，解决机器人任务规划中动作序列不可行的问题。通过场景图验证和修正LLM生成的计划，提升任务完成率。**

- **链接: [https://arxiv.org/pdf/2411.10446](https://arxiv.org/pdf/2411.10446)**

> **作者:** Daniel Ekpo; Mara Levy; Saksham Suri; Chuong Huynh; Archana Swaminathan; Abhinav Shrivastava
>
> **备注:** Accepted to ICRA 2026. Project website: this https URL
>
> **摘要:** Recent progress in vision-language models (VLMs) has opened new possibilities for robot task planning, but these models often produce incorrect action sequences. To address these limitations, we propose VeriGraph, a novel framework that integrates VLMs for robotic planning while verifying action feasibility. VeriGraph uses scene graphs as an intermediate representation to capture key objects and spatial relationships, enabling more reliable plan verification and refinement. The system generates a scene graph from input images and uses it to iteratively check and correct action sequences generated by an LLM-based task planner, ensuring constraints are respected and actions are executable. Our approach significantly enhances task completion rates across diverse manipulation scenarios, outperforming baseline methods by 58% on language-based tasks, 56% on tangram puzzle tasks, and 30% on image-based tasks. Qualitative results and code can be found at this https URL.
>
---
#### [replaced 014] Scalable Unseen Objects 6-DoF Absolute Pose Estimation with Robotic Integration
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人操作任务，解决未知物体6-DoF位姿估计问题。通过单张带标注的RGB-D图像实现可扩展的位姿估计，并构建软硬件结合的机器人系统。**

- **链接: [https://arxiv.org/pdf/2503.05578](https://arxiv.org/pdf/2503.05578)**

> **作者:** Jian Liu; Wei Sun; Kai Zeng; Jin Zheng; Hui Yang; Hossein Rahmani; Ajmal Mian; Lin Wang
>
> **备注:** Accepted by TRO 2026, 18 pages, 9 figures
>
> **摘要:** Pose estimation-guided unseen object 6-DoF robotic manipulation is a key task in robotics. However, the scalability of current pose estimation methods to unseen objects remains a fundamental challenge, as they generally rely on CAD models or dense reference views of unseen objects, which are difficult to acquire, ultimately limit their scalability. In this paper, we introduce a novel task setup, referred to as SinRef-6D, which addresses 6-DoF absolute pose estimation for unseen objects using only a single pose-labeled reference RGB-D image captured during robotic manipulation. This setup is more scalable yet technically nontrivial due to large pose discrepancies and the limited geometric and spatial information contained in a single view. To address these issues, our key idea is to iteratively establish point-wise alignment in a common coordinate system with state space models (SSMs) as backbones. Specifically, to handle large pose discrepancies, we introduce an iterative object-space point-wise alignment strategy. Then, Point and RGB SSMs are proposed to capture long-range spatial dependencies from a single view, offering superior spatial modeling capability with linear complexity. Once pre-trained on synthetic data, SinRef-6D can estimate the 6-DoF absolute pose of an unseen object using only a single reference view. With the estimated pose, we further develop a hardware-software robotic system and integrate the proposed SinRef-6D into it in real-world settings. Extensive experiments on six benchmarks and in diverse real-world scenarios demonstrate that our SinRef-6D offers superior scalability. Additional robotic grasping experiments further validate the effectiveness of the developed robotic system. The code and robotic demos are available at this https URL.
>
---
#### [replaced 015] Make Tracking Easy: Neural Motion Retargeting for Humanoid Whole-body Control
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，解决人类数据与机器人运动之间的差异问题。提出NMR框架，通过学习数据分布提升运动重定向的准确性与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.22201](https://arxiv.org/pdf/2603.22201)**

> **作者:** Qingrui Zhao; Kaiyue Yang; Xiyu Wang; Shiqi Zhao; Yi Lu; Xinfang Zhang; Wei Yin; Qiu Shen; Xiao-Xiao Long; Xun Cao
>
> **备注:** Report, 12 pages, 5 figures, 4 tables, webpage: this https URL
>
> **摘要:** Humanoid robots require diverse motor skills to integrate into complex environments, but bridging the kinematic and dynamic embodiment gap from human data remains a major bottleneck. We demonstrate through Hessian analysis that traditional optimization-based retargeting is inherently non-convex and prone to local optima, leading to physical artifacts like joint jumps and self-penetration. To address this, we reformulate the targeting problem as learning data distribution rather than optimizing optimal solutions, where we propose NMR, a Neural Motion Retargeting framework that transforms static geometric mapping into a dynamics-aware learned process. We first propose Clustered-Expert Physics Refinement (CEPR), a hierarchical data pipeline that leverages VAE-based motion clustering to group heterogeneous movements into latent motifs. This strategy significantly reduces the computational overhead of massively parallel reinforcement learning experts, which project and repair noisy human demonstrations onto the robot's feasible motion manifold. The resulting high-fidelity data supervises a non-autoregressive CNN-Transformer architecture that reasons over global temporal context to suppress reconstruction noise and bypass geometric traps. Experiments on the Unitree G1 humanoid across diverse dynamic tasks (e.g., martial arts, dancing) show that NMR eliminates joint jumps and significantly reduces self-collisions compared to state-of-the-art baselines. Furthermore, NMR-generated references accelerate the convergence of downstream whole-body control policies, establishing a scalable path for bridging the human-robot embodiment gap.
>
---
#### [replaced 016] Scalable Multi-Task Learning through Spiking Neural Networks with Adaptive Task-Switching Policy for Intelligent Autonomous Agents
- **分类: cs.NE; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于多任务学习领域，旨在解决自主智能体在资源受限下多任务训练中的任务干扰问题。提出SwitchMT方法，通过自适应任务切换策略提升性能与可扩展性。**

- **链接: [https://arxiv.org/pdf/2504.13541](https://arxiv.org/pdf/2504.13541)**

> **作者:** Rachmad Vidya Wicaksana Putra; Avaneesh Devkota; Muhammad Shafique
>
> **备注:** Accepted at the 63rd ACM/IEEE Design Automation Conference (DAC), July 26-29, 2026 in Long Beach, CA, USA. [Codes: this https URL]
>
> **摘要:** Training resource-constrained autonomous agents on multiple tasks simultaneously is crucial for adapting to diverse real-world environments. Recent works employ reinforcement learning (RL) approach, but they still suffer from sub-optimal multi-task performance due to task interference. State-of-the-art works employ Spiking Neural Networks (SNNs) to improve RL-based multi-task learning and enable low-power/energy operations through network enhancements and spike-driven data stream processing. However, they rely on fixed task-switching intervals during its training, thus limiting its performance and scalability. To address this, we propose SwitchMT, a novel methodology that employs adaptive task-switching for effective, scalable, and simultaneous multi-task learning. SwitchMT employs the following key ideas: (1) leveraging a Deep Spiking Q-Network with active dendrites and dueling structure, that utilizes task-specific context signals to create specialized sub-networks; and (2) devising an adaptive task-switching policy that leverages both rewards and internal dynamics of the network parameters. Experimental results demonstrate that SwitchMT achieves competitive scores in multiple Atari games (i.e., Pong: -8.8, Breakout: 5.6, and Enduro: 355.2) and longer game episodes as compared to the state-of-the-art. These results also highlight the effectiveness of SwitchMT methodology in addressing task interference without increasing the network complexity, enabling intelligent autonomous agents with scalable multi-task learning capabilities.
>
---
