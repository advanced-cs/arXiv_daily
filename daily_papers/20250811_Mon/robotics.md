# 机器人 cs.RO

- **最新发布 33 篇**

- **更新 19 篇**

## 最新发布

#### [new 001] Latent Policy Barrier: Learning Robust Visuomotor Policies by Staying In-Distribution
- **分类: cs.RO**

- **简介: 论文提出Latent Policy Barrier框架，通过潜在嵌入作为隐式屏障区分安全与危险状态，分离精确模仿与OOD恢复，提升视觉运动策略鲁棒性与数据效率。**

- **链接: [http://arxiv.org/pdf/2508.05941v1](http://arxiv.org/pdf/2508.05941v1)**

> **作者:** Zhanyi Sun; Shuran Song
>
> **摘要:** Visuomotor policies trained via behavior cloning are vulnerable to covariate shift, where small deviations from expert trajectories can compound into failure. Common strategies to mitigate this issue involve expanding the training distribution through human-in-the-loop corrections or synthetic data augmentation. However, these approaches are often labor-intensive, rely on strong task assumptions, or compromise the quality of imitation. We introduce Latent Policy Barrier, a framework for robust visuomotor policy learning. Inspired by Control Barrier Functions, LPB treats the latent embeddings of expert demonstrations as an implicit barrier separating safe, in-distribution states from unsafe, out-of-distribution (OOD) ones. Our approach decouples the role of precise expert imitation and OOD recovery into two separate modules: a base diffusion policy solely on expert data, and a dynamics model trained on both expert and suboptimal policy rollout data. At inference time, the dynamics model predicts future latent states and optimizes them to stay within the expert distribution. Both simulated and real-world experiments show that LPB improves both policy robustness and data efficiency, enabling reliable manipulation from limited expert data and without additional human correction or annotation.
>
---
#### [new 002] Mitigating Undesired Conditions in Flexible Production with Product-Process-Resource Asset Knowledge Graphs
- **分类: cs.RO**

- **简介: 论文提出基于产品-工艺-资源资产知识图谱（PPR-AKG）的柔性生产条件缓解方法，解决传统模型对不良条件与错误处理不足的问题，通过语义技术与大语言模型实现人机交互，优化资源分配与异常识别。**

- **链接: [http://arxiv.org/pdf/2508.06278v1](http://arxiv.org/pdf/2508.06278v1)**

> **作者:** Petr Novak; Stefan Biffl; Marek Obitko; Petr Kadera
>
> **备注:** 3 pages, 1 figure
>
> **摘要:** Contemporary industrial cyber-physical production systems (CPPS) composed of robotic workcells face significant challenges in the analysis of undesired conditions due to the flexibility of Industry 4.0 that disrupts traditional quality assurance mechanisms. This paper presents a novel industry-oriented semantic model called Product-Process-Resource Asset Knowledge Graph (PPR-AKG), which is designed to analyze and mitigate undesired conditions in flexible CPPS. Built on top of the well-proven Product-Process-Resource (PPR) model originating from ISA-95 and VDI-3682, a comprehensive OWL ontology addresses shortcomings of conventional model-driven engineering for CPPS, particularly inadequate undesired condition and error handling representation. The integration of semantic technologies with large language models (LLMs) provides intuitive interfaces for factory operators, production planners, and engineers to interact with the entire model using natural language. Evaluation with the use case addressing electric vehicle battery remanufacturing demonstrates that the PPR-AKG approach efficiently supports resource allocation based on explicitly represented capabilities as well as identification and mitigation of undesired conditions in production. The key contributions include (1) a holistic PPR-AKG model capturing multi-dimensional production knowledge, and (2) the useful combination of the PPR-AKG with LLM-based chatbots for human interaction.
>
---
#### [new 003] Shortcut Learning in Generalist Robot Policies: The Role of Dataset Diversity and Fragmentation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 论文探讨一般主义机器人策略的短路学习机制，分析数据集多样性与碎片化对泛化能力的影响，提出优化数据采集和增强策略以提升泛化性能。**

- **链接: [http://arxiv.org/pdf/2508.06426v1](http://arxiv.org/pdf/2508.06426v1)**

> **作者:** Youguang Xing; Xu Luo; Junlin Xie; Lianli Gao; Hengtao Shen; Jingkuan Song
>
> **备注:** CoRL 2025
>
> **摘要:** Generalist robot policies trained on large-scale datasets such as Open X-Embodiment (OXE) demonstrate strong performance across a wide range of tasks. However, they often struggle to generalize beyond the distribution of their training data. In this paper, we investigate the underlying cause of this limited generalization capability. We identify shortcut learning -- the reliance on task-irrelevant features -- as a key impediment to generalization. Through comprehensive theoretical and empirical analysis, we uncover two primary contributors to shortcut learning: (1) limited diversity within individual sub-datasets, and (2) significant distributional disparities across sub-datasets, leading to dataset fragmentation. These issues arise from the inherent structure of large-scale datasets like OXE, which are typically composed of multiple sub-datasets collected independently across varied environments and embodiments. Our findings provide critical insights into dataset collection strategies that can reduce shortcut learning and enhance the generalization ability of generalist robot policies. Moreover, in scenarios where acquiring new large-scale data is impractical, we demonstrate that carefully selected robotic data augmentation strategies can effectively reduce shortcut learning in existing offline datasets, thereby improving generalization capabilities of generalist robot policies, e.g., $\pi_0$, in both simulation and real-world environments. More information at https://lucky-light-sun.github.io/proj/shortcut-learning-in-grps/.
>
---
#### [new 004] Modular Vacuum-Based Fixturing System for Adaptive Disassembly Workspace Integration
- **分类: cs.RO**

- **简介: 论文提出一种模块化真空固定系统，解决小家电拆卸中复杂曲面适应难题，通过软抓取器与稳定性分析框架提升螺丝去除成功率，验证其优于传统刚性夹具。**

- **链接: [http://arxiv.org/pdf/2508.05936v1](http://arxiv.org/pdf/2508.05936v1)**

> **作者:** Haohui Pan; Takuya Kiyokawa; Tomoki Ishikura; Shingo Hamada; Genichiro Matsuda; Kensuke Harada
>
> **备注:** 8 pages, 9 figures
>
> **摘要:** The disassembly of small household appliances poses significant challenges due to their complex and curved geometries, which render traditional rigid fixtures inadequate. In this paper, we propose a modular vacuum-based fixturing system that leverages commercially available balloon-type soft grippers to conform to arbitrarily shaped surfaces and provide stable support during screw-removal tasks. To enable a reliable deployment of the system, we develop a stability-aware planning framework that samples the bottom surface of the target object, filters candidate contact points based on geometric continuity, and evaluates support configurations using convex hull-based static stability criteria. We compare the quality of object placement under different numbers and configurations of balloon hands. In addition, real-world experiments were conducted to compare the success rates of traditional rigid fixtures with our proposed system. The results demonstrate that our method consistently achieves higher success rates and superior placement stability during screw removal tasks.
>
---
#### [new 005] Integrating Vision Foundation Models with Reinforcement Learning for Enhanced Object Interaction
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.SY; eess.SY; 68T07, 68T40, 90C40, 93E35; I.2.6; I.2.9; I.2.10**

- **简介: 论文提出融合视觉基础模型与强化学习，提升机器人在模拟环境中的物体交互能力，通过SAM+YOLOv5+PPO代理实现感知与交互优化，显著提高成功率和导航效率。**

- **链接: [http://arxiv.org/pdf/2508.05838v1](http://arxiv.org/pdf/2508.05838v1)**

> **作者:** Ahmad Farooq; Kamran Iqbal
>
> **备注:** Published in the Proceedings of the 2025 3rd International Conference on Robotics, Control and Vision Engineering (RCVE'25). 6 pages, 3 figures, 1 table
>
> **摘要:** This paper presents a novel approach that integrates vision foundation models with reinforcement learning to enhance object interaction capabilities in simulated environments. By combining the Segment Anything Model (SAM) and YOLOv5 with a Proximal Policy Optimization (PPO) agent operating in the AI2-THOR simulation environment, we enable the agent to perceive and interact with objects more effectively. Our comprehensive experiments, conducted across four diverse indoor kitchen settings, demonstrate significant improvements in object interaction success rates and navigation efficiency compared to a baseline agent without advanced perception. The results show a 68% increase in average cumulative reward, a 52.5% improvement in object interaction success rate, and a 33% increase in navigation efficiency. These findings highlight the potential of integrating foundation models with reinforcement learning for complex robotic tasks, paving the way for more sophisticated and capable autonomous agents.
>
---
#### [new 006] Computer Vision-based Adaptive Control for Back Exoskeleton Performance Optimization
- **分类: cs.RO**

- **简介: 论文提出基于计算机视觉的自适应控制方法，优化背外骨骼性能，解决支持策略和负载估计问题，通过实时负载估计和动态调整提升舒适度与效率。**

- **链接: [http://arxiv.org/pdf/2508.06207v1](http://arxiv.org/pdf/2508.06207v1)**

> **作者:** Andrea Dal Prete; Seyram Ofori; Chan Yon Sin; Ashwin Narayan; Francesco Braghin; Marta Gandolla; Haoyong Yu
>
> **摘要:** Back exoskeletons can reduce musculoskeletal strain, but their effectiveness depends on support modulation and adaptive control. This study addresses two challenges: defining optimal support strategies and developing adaptive control based on payload estimation. We introduce an optimization space based on muscle activity reduction, perceived discomfort, and user preference, constructing functions to identify optimal strategies. Experiments with 12 subjects revealed optimal operating regions, highlighting the need for dynamic modulation. Based on these insights, we developed a vision-based adaptive control pipeline that estimates payloads in real-time by enhancing exoskeleton contextual understanding, minimising latency and enabling support adaptation within the defined optimisation space. Validation with 12 more subjects showed over 80% accuracy and improvements across all metrics. Compared to static control, adaptive modulation reduced peak back muscle activation by up to 23% while preserving user preference and minimising discomfort. These findings validate the proposed framework and highlight the potential of intelligent, context-aware control in industrial exoskeletons.
>
---
#### [new 007] Affordance-R1: Reinforcement Learning for Generalizable Affordance Reasoning in Multimodal Large Language Model
- **分类: cs.RO; cs.CV**

- **简介: 论文提出Affordance-R1框架，通过强化学习结合认知CoT引导的GRPO，解决多模态大语言模型在跨域泛化和推理能力不足的问题，设计多奖励函数与ReasonAff数据集，实现零样本泛化与测试时推理。**

- **链接: [http://arxiv.org/pdf/2508.06206v1](http://arxiv.org/pdf/2508.06206v1)**

> **作者:** Hanqing Wang; Shaoyang Wang; Yiming Zhong; Zemin Yang; Jiamin Wang; Zhiqing Cui; Jiahao Yuan; Yifan Han; Mingyu Liu; Yuexin Ma
>
> **摘要:** Affordance grounding focuses on predicting the specific regions of objects that are associated with the actions to be performed by robots. It plays a vital role in the fields of human-robot interaction, human-object interaction, embodied manipulation, and embodied perception. Existing models often neglect the affordance shared among different objects because they lack the Chain-of-Thought(CoT) reasoning abilities, limiting their out-of-domain (OOD) generalization and explicit reasoning capabilities. To address these challenges, we propose Affordance-R1, the first unified affordance grounding framework that integrates cognitive CoT guided Group Relative Policy Optimization (GRPO) within a reinforcement learning paradigm. Specifically, we designed a sophisticated affordance function, which contains format, perception, and cognition rewards to effectively guide optimization directions. Furthermore, we constructed a high-quality affordance-centric reasoning dataset, ReasonAff, to support training. Trained exclusively via reinforcement learning with GRPO and without explicit reasoning data, Affordance-R1 achieves robust zero-shot generalization and exhibits emergent test-time reasoning capabilities. Comprehensive experiments demonstrate that our model outperforms well-established methods and exhibits open-world generalization. To the best of our knowledge, Affordance-R1 is the first to integrate GRPO-based RL with reasoning into affordance reasoning. The code of our method and our dataset is released on https://github.com/hq-King/Affordance-R1.
>
---
#### [new 008] V*: An Efficient Motion Planning Algorithm for Autonomous Vehicles
- **分类: cs.RO; math.OC**

- **简介: 论文提出V*算法用于自主车辆导航，解决时间最优、无碰撞轨迹规划问题，通过动态图生成与几何剪枝优化高维搜索，实现动态环境下的高效路径规划。**

- **链接: [http://arxiv.org/pdf/2508.06404v1](http://arxiv.org/pdf/2508.06404v1)**

> **作者:** Abdullah Zareh Andaryan; Michael G. H. Bell; Mohsen Ramezani; Glenn Geers
>
> **摘要:** Autonomous vehicle navigation in structured environments requires planners capable of generating time-optimal, collision-free trajectories that satisfy dynamic and kinematic constraints. We introduce V*, a graph-based motion planner that represents speed and direction as explicit state variables within a discretised space-time-velocity lattice. Unlike traditional methods that decouple spatial search from dynamic feasibility or rely on post-hoc smoothing, V* integrates both motion dimensions directly into graph construction through dynamic graph generation during search expansion. To manage the complexity of high-dimensional search, we employ a hexagonal discretisation strategy and provide formal mathematical proofs establishing optimal waypoint spacing and minimal node redundancy under constrained heading transitions for velocity-aware motion planning. We develop a mathematical formulation for transient steering dynamics in the kinematic bicycle model, modelling steering angle convergence with exponential behaviour, and deriving the relationship for convergence rate parameters. This theoretical foundation, combined with geometric pruning strategies that eliminate expansions leading to infeasible steering configurations, enables V* to evaluate dynamically admissible manoeuvres, ensuring each trajectory is physically realisable without further refinement. We further demonstrate V*'s performance in simulation studies with cluttered and dynamic environments involving moving obstacles, showing its ability to avoid conflicts, yield proactively, and generate safe, efficient trajectories with temporal reasoning capabilities for waiting behaviours and dynamic coordination.
>
---
#### [new 009] Bounding Distributional Shifts in World Modeling through Novelty Detection
- **分类: cs.RO; cs.AI**

- **简介: 论文提出利用变分自编码器作为新颖性检测器，缓解视觉世界模型中的分布偏移问题，提升数据效率。**

- **链接: [http://arxiv.org/pdf/2508.06096v1](http://arxiv.org/pdf/2508.06096v1)**

> **作者:** Eric Jing; Abdeslam Boularias
>
> **备注:** 7 pages, 6 figures
>
> **摘要:** Recent work on visual world models shows significant promise in latent state dynamics obtained from pre-trained image backbones. However, most of the current approaches are sensitive to training quality, requiring near-complete coverage of the action and state space during training to prevent divergence during inference. To make a model-based planning algorithm more robust to the quality of the learned world model, we propose in this work to use a variational autoencoder as a novelty detector to ensure that proposed action trajectories during planning do not cause the learned model to deviate from the training data distribution. To evaluate the effectiveness of this approach, a series of experiments in challenging simulated robot environments was carried out, with the proposed method incorporated into a model-predictive control policy loop extending the DINO-WM architecture. The results clearly show that the proposed method improves over state-of-the-art solutions in terms of data efficiency.
>
---
#### [new 010] Situationally-aware Path Planning Exploiting 3D Scene Graphs
- **分类: cs.RO**

- **简介: 论文提出基于三维场景图的情境感知路径规划方法，通过两阶段搜索与分解及重规划机制，显著提升效率并保持路径最优性，适用于室内场景。**

- **链接: [http://arxiv.org/pdf/2508.06283v1](http://arxiv.org/pdf/2508.06283v1)**

> **作者:** Saad Ejaz; Marco Giberna; Muhammad Shaheer; Jose Andres Millan-Romera; Ali Tourani; Paul Kremer; Holger Voos; Jose Luis Sanchez-Lopez
>
> **摘要:** 3D Scene Graphs integrate both metric and semantic information, yet their structure remains underutilized for improving path planning efficiency and interpretability. In this work, we present S-Path, a situationally-aware path planner that leverages the metric-semantic structure of indoor 3D Scene Graphs to significantly enhance planning efficiency. S-Path follows a two-stage process: it first performs a search over a semantic graph derived from the scene graph to yield a human-understandable high-level path. This also identifies relevant regions for planning, which later allows the decomposition of the problem into smaller, independent subproblems that can be solved in parallel. We also introduce a replanning mechanism that, in the event of an infeasible path, reuses information from previously solved subproblems to update semantic heuristics and prioritize reuse to further improve the efficiency of future planning attempts. Extensive experiments on both real-world and simulated environments show that S-Path achieves average reductions of 5.7x in planning time while maintaining comparable path optimality to classical sampling-based planners and surpassing them in complex scenarios, making it an efficient and interpretable path planner for environments represented by indoor 3D Scene Graphs.
>
---
#### [new 011] Beyond Constant Parameters: Hyper Prediction Models and HyperMPC
- **分类: cs.RO**

- **简介: 论文提出HyperPrediction Model解决传统MPC动力学建模问题，通过时间依赖参数提升效率与鲁棒性，实验证明其在复杂系统中优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.06181v1](http://arxiv.org/pdf/2508.06181v1)**

> **作者:** Jan Węgrzynowski; Piotr Kicki; Grzegorz Czechmanowski; Maciej Krupka; Krzysztof Walas
>
> **摘要:** Model Predictive Control (MPC) is among the most widely adopted and reliable methods for robot control, relying critically on an accurate dynamics model. However, existing dynamics models used in the gradient-based MPC are limited by computational complexity and state representation. To address this limitation, we propose the Hyper Prediction Model (HyperPM) - a novel approach in which we project the unmodeled dynamics onto a time-dependent dynamics model. This time-dependency is captured through time-varying model parameters, whose evolution over the MPC prediction horizon is learned using a neural network. Such formulation preserves the computational efficiency and robustness of the base model while equipping it with the capacity to anticipate previously unmodeled phenomena. We evaluated the proposed approach on several challenging systems, including real-world F1TENTH autonomous racing, and demonstrated that it significantly reduces long-horizon prediction errors. Moreover, when integrated within the MPC framework (HyperMPC), our method consistently outperforms existing state-of-the-art techniques.
>
---
#### [new 012] Dynamical Trajectory Planning of Disturbance Consciousness for Air-Land Bimodal Unmanned Aerial Vehicles
- **分类: cs.RO**

- **简介: 论文提出一种基于扰动感知的空中陆地双模无人机轨迹规划方法，解决环境干扰下的鲁棒性问题，通过实时估计与自适应安全边界调整，提升轨迹可行性与性能。**

- **链接: [http://arxiv.org/pdf/2508.05972v1](http://arxiv.org/pdf/2508.05972v1)**

> **作者:** Shaoting Liu; Zhou Liu
>
> **摘要:** Air-land bimodal vehicles provide a promising solution for navigating complex environments by combining the flexibility of aerial locomotion with the energy efficiency of ground mobility. To enhance the robustness of trajectory planning under environmental disturbances, this paper presents a disturbance-aware planning framework that incorporates real-time disturbance estimation into both path searching and trajectory optimization. A key component of the framework is a disturbance-adaptive safety boundary adjustment mechanism, which dynamically modifies the vehicle's feasible dynamic boundaries based on estimated disturbances to ensure trajectory feasibility. Leveraging the dynamics model of the bimodal vehicle, the proposed approach achieves adaptive and reliable motion planning across different terrains and operating conditions. A series of real-world experiments and benchmark comparisons on a custom-built platform validate the effectiveness and robustness of the method, demonstrating improvements in tracking accuracy, task efficiency, and energy performance under both ground and aerial disturbances.
>
---
#### [new 013] REBot: Reflexive Evasion Robot for Instantaneous Dynamic Obstacle Avoidance
- **分类: cs.RO**

- **简介: 论文提出REBot框架，通过有限状态机整合避障与恢复策略，解决四足机器人快速动态障碍避让难题，实现低延迟瞬时避障。**

- **链接: [http://arxiv.org/pdf/2508.06229v1](http://arxiv.org/pdf/2508.06229v1)**

> **作者:** Zihao Xu; Ce Hao; Chunzheng Wang; Kuankuan Sima; Fan Shi; Jin Song Dong
>
> **摘要:** Dynamic obstacle avoidance (DOA) is critical for quadrupedal robots operating in environments with moving obstacles or humans. Existing approaches typically rely on navigation-based trajectory replanning, which assumes sufficient reaction time and leading to fails when obstacles approach rapidly. In such scenarios, quadrupedal robots require reflexive evasion capabilities to perform instantaneous, low-latency maneuvers. This paper introduces Reflexive Evasion Robot (REBot), a control framework that enables quadrupedal robots to achieve real-time reflexive obstacle avoidance. REBot integrates an avoidance policy and a recovery policy within a finite-state machine. With carefully designed learning curricula and by incorporating regularization and adaptive rewards, REBot achieves robust evasion and rapid stabilization in instantaneous DOA tasks. We validate REBot through extensive simulations and real-world experiments, demonstrating notable improvements in avoidance success rates, energy efficiency, and robustness to fast-moving obstacles. Videos and appendix are available on https://rebot-2025.github.io/.
>
---
#### [new 014] Evaluating Robot Program Performance with Power Consumption Driven Metrics in Lightweight Industrial Robots
- **分类: cs.RO**

- **简介: 论文提出基于功率消耗的评估框架，解决传统CPU指标忽略物理影响的问题，通过能量利用系数、转换指标和可靠性系数等指标量化机器人程序性能，实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.06295v1](http://arxiv.org/pdf/2508.06295v1)**

> **作者:** Juan Heredia; Emil Stubbe Kolvig-Raun; Sune Lundo Sorensen; Mikkel Baun Kjaergaard
>
> **摘要:** The code performance of industrial robots is typically analyzed through CPU metrics, which overlook the physical impact of code on robot behavior. This study introduces a novel framework for assessing robot program performance from an embodiment perspective by analyzing the robot's electrical power profile. Our approach diverges from conventional CPU based evaluations and instead leverages a suite of normalized metrics, namely, the energy utilization coefficient, the energy conversion metric, and the reliability coefficient, to capture how efficiently and reliably energy is used during task execution. Complementing these metrics, the established robot wear metric provides further insight into long term reliability. Our approach is demonstrated through an experimental case study in machine tending, comparing four programs with diverse strategies using a UR5e robot. The proposed metrics directly compare and categorize different robot programs, regardless of the specific task, by linking code performance to its physical manifestation through power consumption patterns. Our results reveal the strengths and weaknesses of each strategy, offering actionable insights for optimizing robot programming practices. Enhancing energy efficiency and reliability through this embodiment centric approach not only improves individual robot performance but also supports broader industrial objectives such as sustainable manufacturing and cost reduction.
>
---
#### [new 015] Affordance-Guided Dual-Armed Disassembly Teleoperation for Mating Parts
- **分类: cs.RO**

- **简介: 论文提出基于affordance的双臂远程拆卸系统，解决机械部件拆卸中灵活性与结构可见性不足问题，通过虚拟环境可视化抓取与拆卸方向，结合混合控制器提升任务成功率及姿态精度。**

- **链接: [http://arxiv.org/pdf/2508.05937v1](http://arxiv.org/pdf/2508.05937v1)**

> **作者:** Gen Sako; Takuya Kiyokawa; Kensuke Harada; Tomoki Ishikura; Naoya Miyaji; Genichiro Matsuda
>
> **备注:** 6 pages, 9 figures
>
> **摘要:** Robotic non-destructive disassembly of mating parts remains challenging due to the need for flexible manipulation and the limited visibility of internal structures. This study presents an affordance-guided teleoperation system that enables intuitive human demonstrations for dual-arm fix-and-disassemble tasks for mating parts. The system visualizes feasible grasp poses and disassembly directions in a virtual environment, both derived from the object's geometry, to address occlusions and structural complexity. To prevent excessive position tracking under load when following the affordance, we integrate a hybrid controller that combines position and impedance control into the teleoperated disassembly arm. Real-world experiments validate the effectiveness of the proposed system, showing improved task success rates and reduced object pose deviation.
>
---
#### [new 016] GPU-Accelerated Barrier-Rate Guided MPPI Control for Tractor-Trailer Systems
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出GPU加速的BR-MPPI控制方法，用于拖挂车复杂环境导航，通过CBF约束提升探索与鲁棒性，实现高效安全控制。**

- **链接: [http://arxiv.org/pdf/2508.05773v1](http://arxiv.org/pdf/2508.05773v1)**

> **作者:** Keyvan Majd; Hardik Parwana; Bardh Hoxha; Steven Hong; Hideki Okamoto; Georgios Fainekos
>
> **备注:** Accepted to IEEE ITSC 2025
>
> **摘要:** Articulated vehicles such as tractor-trailers, yard trucks, and similar platforms must often reverse and maneuver in cluttered spaces where pedestrians are present. We present how Barrier-Rate guided Model Predictive Path Integral (BR-MPPI) control can solve navigation in such challenging environments. BR-MPPI embeds Control Barrier Function (CBF) constraints directly into the path-integral update. By steering the importance-sampling distribution toward collision-free, dynamically feasible trajectories, BR-MPPI enhances the exploration strength of MPPI and improves robustness of resulting trajectories. The method is evaluated in the high-fidelity CarMaker simulator on a 12 [m] tractor-trailer tasked with reverse and forward parking in a parking lot. BR-MPPI computes control inputs in above 100 [Hz] on a single GPU (for scenarios with eight obstacles) and maintains better parking clearance than a standard MPPI baseline and an MPPI with collision cost baseline.
>
---
#### [new 017] L2Calib: $SE(3)$-Manifold Reinforcement Learning for Robust Extrinsic Calibration with Degenerate Motion Resilience
- **分类: cs.RO**

- **简介: 论文提出基于强化学习的SE(3) manifold外参校准方法，通过轨迹奖励机制和数据筛选提升鲁棒性，解决传统方法依赖结构目标和弱激发问题，适用于多种机器人平台。**

- **链接: [http://arxiv.org/pdf/2508.06330v1](http://arxiv.org/pdf/2508.06330v1)**

> **作者:** Baorun Li; Chengrui Zhu; Siyi Du; Bingran Chen; Jie Ren; Wenfei Wang; Yong Liu; Jiajun Lv
>
> **备注:** IROS2025
>
> **摘要:** Extrinsic calibration is essential for multi-sensor fusion, existing methods rely on structured targets or fully-excited data, limiting real-world applicability. Online calibration further suffers from weak excitation, leading to unreliable estimates. To address these limitations, we propose a reinforcement learning (RL)-based extrinsic calibration framework that formulates extrinsic calibration as a decision-making problem, directly optimizes $SE(3)$ extrinsics to enhance odometry accuracy. Our approach leverages a probabilistic Bingham distribution to model 3D rotations, ensuring stable optimization while inherently retaining quaternion symmetry. A trajectory alignment reward mechanism enables robust calibration without structured targets by quantitatively evaluating estimated tightly-coupled trajectory against a reference trajectory. Additionally, an automated data selection module filters uninformative samples, significantly improving efficiency and scalability for large-scale datasets. Extensive experiments on UAVs, UGVs, and handheld platforms demonstrate that our method outperforms traditional optimization-based approaches, achieving high-precision calibration even under weak excitation conditions. Our framework simplifies deployment on diverse robotic platforms by eliminating the need for high-quality initial extrinsics and enabling calibration from routine operating data. The code is available at https://github.com/APRIL-ZJU/learn-to-calibrate.
>
---
#### [new 018] ReNiL: Relative Neural Inertial Locator with Any-Scale Bayesian Inference
- **分类: cs.RO**

- **简介: 论文提出一种基于贝叶斯深度学习的惯性定位框架ReNiL，解决传统方法难以适应不同运动尺度与不确定性不一致的问题，通过IPDPs和ASLE实现高效、鲁棒的行人定位，提升精度与不确定性可控性。**

- **链接: [http://arxiv.org/pdf/2508.06053v1](http://arxiv.org/pdf/2508.06053v1)**

> **作者:** Kaixuan Wu; Yuanzhuo Xu; Zejun Zhang; Weiping Zhu; Steve Drew; Xiaoguang Niu
>
> **摘要:** Pedestrian inertial localization is key for mobile and IoT services because it provides infrastructure-free positioning. Yet most learning-based methods depend on fixed sliding-window integration, struggle to adapt to diverse motion scales and cadences, and yield inconsistent uncertainty, limiting real-world use. We present ReNiL, a Bayesian deep-learning framework for accurate, efficient, and uncertainty-aware pedestrian localization. ReNiL introduces Inertial Positioning Demand Points (IPDPs) to estimate motion at contextually meaningful waypoints instead of dense tracking, and supports inference on IMU sequences at any scale so cadence can match application needs. It couples a motion-aware orientation filter with an Any-Scale Laplace Estimator (ASLE), a dual-task network that blends patch-based self-supervision with Bayesian regression. By modeling displacements with a Laplace distribution, ReNiL provides homogeneous Euclidean uncertainty that integrates cleanly with other sensors. A Bayesian inference chain links successive IPDPs into consistent trajectories. On RoNIN-ds and a new WUDataset covering indoor and outdoor motion from 28 participants, ReNiL achieves state-of-the-art displacement accuracy and uncertainty consistency, outperforming TLIO, CTIN, iMoT, and RoNIN variants while reducing computation. Application studies further show robustness and practicality for mobile and IoT localization, making ReNiL a scalable, uncertainty-aware foundation for next-generation positioning.
>
---
#### [new 019] Incremental Language Understanding for Online Motion Planning of Robot Manipulators
- **分类: cs.RO**

- **简介: 论文提出基于推理的增量解析器，融合在线运动规划，实现机器人实时更新运动计划以适应动态语言输入，解决传统方法需重启重规划的问题。**

- **链接: [http://arxiv.org/pdf/2508.06095v1](http://arxiv.org/pdf/2508.06095v1)**

> **作者:** Mitchell Abrams; Thies Oelerich; Christian Hartl-Nesic; Andreas Kugi; Matthias Scheutz
>
> **备注:** 8 pages, 9 figures, accepted at IROS 2025
>
> **摘要:** Human-robot interaction requires robots to process language incrementally, adapting their actions in real-time based on evolving speech input. Existing approaches to language-guided robot motion planning typically assume fully specified instructions, resulting in inefficient stop-and-replan behavior when corrections or clarifications occur. In this paper, we introduce a novel reasoning-based incremental parser which integrates an online motion planning algorithm within the cognitive architecture. Our approach enables continuous adaptation to dynamic linguistic input, allowing robots to update motion plans without restarting execution. The incremental parser maintains multiple candidate parses, leveraging reasoning mechanisms to resolve ambiguities and revise interpretations when needed. By combining symbolic reasoning with online motion planning, our system achieves greater flexibility in handling speech corrections and dynamically changing constraints. We evaluate our framework in real-world human-robot interaction scenarios, demonstrating online adaptions of goal poses, constraints, or task objectives. Our results highlight the advantages of integrating incremental language understanding with real-time motion planning for natural and fluid human-robot collaboration. The experiments are demonstrated in the accompanying video at www.acin.tuwien.ac.at/42d5.
>
---
#### [new 020] ADPro: a Test-time Adaptive Diffusion Policy for Robot Manipulation via Manifold and Initial Noise Constraints
- **分类: cs.RO**

- **简介: 论文提出一种基于流形约束和初始化策略的自适应扩散政策ADPro，解决传统方法忽略几何先验导致的探索过度和收敛慢问题，提升机器人操作的成功率与效率。**

- **链接: [http://arxiv.org/pdf/2508.06266v1](http://arxiv.org/pdf/2508.06266v1)**

> **作者:** Zezeng Li; Rui Yang; Ruochen Chen; ZhongXuan Luo; Liming Chen
>
> **摘要:** Diffusion policies have recently emerged as a powerful class of visuomotor controllers for robot manipulation, offering stable training and expressive multi-modal action modeling. However, existing approaches typically treat action generation as an unconstrained denoising process, ignoring valuable a priori knowledge about geometry and control structure. In this work, we propose the Adaptive Diffusion Policy (ADP), a test-time adaptation method that introduces two key inductive biases into the diffusion. First, we embed a geometric manifold constraint that aligns denoising updates with task-relevant subspaces, leveraging the fact that the relative pose between the end-effector and target scene provides a natural gradient direction, and guiding denoising along the geodesic path of the manipulation manifold. Then, to reduce unnecessary exploration and accelerate convergence, we propose an analytically guided initialization: rather than sampling from an uninformative prior, we compute a rough registration between the gripper and target scenes to propose a structured initial noisy action. ADP is compatible with pre-trained diffusion policies and requires no retraining, enabling test-time adaptation that tailors the policy to specific tasks, thereby enhancing generalization across novel tasks and environments. Experiments on RLBench, CALVIN, and real-world dataset show that ADPro, an implementation of ADP, improves success rates, generalization, and sampling efficiency, achieving up to 25% faster execution and 9% points over strong diffusion baselines.
>
---
#### [new 021] Real-Time 3D Vision-Language Embedding Mapping
- **分类: cs.RO**

- **简介: 论文提出实时三维视觉-语言嵌入映射方法，解决精确语义3D表示问题，通过局部掩码与信心加权整合提升定位精度，适用于多场景机器人任务。**

- **链接: [http://arxiv.org/pdf/2508.06291v1](http://arxiv.org/pdf/2508.06291v1)**

> **作者:** Christian Rauch; Björn Ellensohn; Linus Nwankwo; Vedant Dave; Elmar Rueckert
>
> **摘要:** A metric-accurate semantic 3D representation is essential for many robotic tasks. This work proposes a simple, yet powerful, way to integrate the 2D embeddings of a Vision-Language Model in a metric-accurate 3D representation at real-time. We combine a local embedding masking strategy, for a more distinct embedding distribution, with a confidence-weighted 3D integration for more reliable 3D embeddings. The resulting metric-accurate embedding representation is task-agnostic and can represent semantic concepts on a global multi-room, as well as on a local object-level. This enables a variety of interactive robotic applications that require the localisation of objects-of-interest via natural language. We evaluate our approach on a variety of real-world sequences and demonstrate that these strategies achieve a more accurate object-of-interest localisation while improving the runtime performance in order to meet our real-time constraints. We further demonstrate the versatility of our approach in a variety of interactive handheld, mobile robotics and manipulation tasks, requiring only raw image data.
>
---
#### [new 022] EcBot: Data-Driven Energy Consumption Open-Source MATLAB Library for Manipulators
- **分类: cs.RO**

- **简介: 论文提出一个开源MATLAB库EcBot，通过数据驱动方法建模机械臂能耗，解决传统模型精度不足问题，需Denavit-Hartenberg参数及实测数据，测试显示RMSE在1.42-5.25W。**

- **链接: [http://arxiv.org/pdf/2508.06276v1](http://arxiv.org/pdf/2508.06276v1)**

> **作者:** Juan Heredia; Christian Schlette; Mikkel Baun Kjærgaard
>
> **摘要:** Existing literature proposes models for estimating the electrical power of manipulators, yet two primary limitations prevail. First, most models are predominantly tested using traditional industrial robots. Second, these models often lack accuracy. To address these issues, we introduce an open source Matlab-based library designed to automatically generate \ac{ec} models for manipulators. The necessary inputs for the library are Denavit-Hartenberg parameters, link masses, and centers of mass. Additionally, our model is data-driven and requires real operational data, including joint positions, velocities, accelerations, electrical power, and corresponding timestamps. We validated our methodology by testing on four lightweight robots sourced from three distinct manufacturers: Universal Robots, Franka Emika, and Kinova. The model underwent testing, and the results demonstrated an RMSE ranging from 1.42 W to 2.80 W for the training dataset and from 1.45 W to 5.25 W for the testing dataset.
>
---
#### [new 023] Social and Telepresence Robots for Accessibility and Inclusion in Small Museums
- **分类: cs.RO; cs.HC**

- **简介: 论文研究社会与远程机器人提升小博物馆的无障碍与包容性，解决低密度地区访问障碍问题，通过社交机器人和远程机器人作为导览工具和远程访问手段，探索故事讲述、共情及协作机制。**

- **链接: [http://arxiv.org/pdf/2508.05946v1](http://arxiv.org/pdf/2508.05946v1)**

> **作者:** Nello Balossino; Rossana Damiano; Cristina Gena; Alberto Lillo; Anna Maria Marras; Claudio Mattutino; Antonio Pizzo; Alessia Prin; Fabiana Vernero
>
> **摘要:** There are still many museums that present accessibility barriers, particularly regarding perceptual, cultural, and cognitive aspects. This is especially evident in low-density population areas. The aim of the ROBSO-PM project is to improve the accessibility of small museums through the use of social robots and social telepresence robots, focusing on three museums as case studies: the Museum of the Holy Shroud in Turin, a small but globally known institution, and two lesser known mountain museums: the Museum of the Champlas du Col Carnival and the Pragelato Museum of Alpine Peoples' Costumes and Traditions. The project explores two main applications for robots: as guides supporting inclusive visits for foreign or disabled visitors, and as telepresence tools allowing people with limited mobility to access museums remotely. From a research perspective, key topics include storytelling, robot personality, empathy, personalization, and, in the case of telepresence, collaboration between the robot and the person, with clearly defined roles and autonomy.
>
---
#### [new 024] Towards Balanced Behavior Cloning from Imbalanced Datasets
- **分类: cs.RO**

- **简介: 论文提出针对不平衡数据集的行为克隆方法，解决数据量与重要性不匹配导致的政策偏差问题，通过自动重平衡算法提升模仿学习效果，引入元梯度策略优化性能。**

- **链接: [http://arxiv.org/pdf/2508.06319v1](http://arxiv.org/pdf/2508.06319v1)**

> **作者:** Sagar Parekh; Heramb Nemlekar; Dylan P. Losey
>
> **摘要:** Robots should be able to learn complex behaviors from human demonstrations. In practice, these human-provided datasets are inevitably imbalanced: i.e., the human demonstrates some subtasks more frequently than others. State-of-the-art methods default to treating each element of the human's dataset as equally important. So if -- for instance -- the majority of the human's data focuses on reaching a goal, and only a few state-action pairs move to avoid an obstacle, the learning algorithm will place greater emphasis on goal reaching. More generally, misalignment between the relative amounts of data and the importance of that data causes fundamental problems for imitation learning approaches. In this paper we analyze and develop learning methods that automatically account for mixed datasets. We formally prove that imbalanced data leads to imbalanced policies when each state-action pair is weighted equally; these policies emulate the most represented behaviors, and not the human's complex, multi-task demonstrations. We next explore algorithms that rebalance offline datasets (i.e., reweight the importance of different state-action pairs) without human oversight. Reweighting the dataset can enhance the overall policy performance. However, there is no free lunch: each method for autonomously rebalancing brings its own pros and cons. We formulate these advantages and disadvantages, helping other researchers identify when each type of approach is most appropriate. We conclude by introducing a novel meta-gradient rebalancing algorithm that addresses the primary limitations behind existing approaches. Our experiments show that dataset rebalancing leads to better downstream learning, improving the performance of general imitation learning algorithms without requiring additional data collection. See our project website: https://collab.me.vt.edu/data_curation/.
>
---
#### [new 025] Surrogate-Enhanced Modeling and Adaptive Modular Control of All-Electric Heavy-Duty Robotic Manipulators
- **分类: cs.RO**

- **简介: 论文提出一种基于代理增强建模与自适应模块化控制的全电动重型机器人机械臂系统，解决高精度实时控制问题，通过融合神经网络与虚拟分解控制架构实现动态建模与分层控制，实验验证其在负载下的有效性。**

- **链接: [http://arxiv.org/pdf/2508.06313v1](http://arxiv.org/pdf/2508.06313v1)**

> **作者:** Amir Hossein Barjini; Mohammad Bahari; Mahdi Hejrati; Jouni Mattila
>
> **备注:** This is submitted to IEEE T-ASE
>
> **摘要:** This paper presents a unified system-level modeling and control framework for an all-electric heavy-duty robotic manipulator (HDRM) driven by electromechanical linear actuators (EMLAs). A surrogate-enhanced actuator model, combining integrated electromechanical dynamics with a neural network trained on a dedicated testbed, is integrated into an extended virtual decomposition control (VDC) architecture augmented by a natural adaptation law. The derived analytical HDRM model supports a hierarchical control structure that seamlessly maps high-level force and velocity objectives to real-time actuator commands, accompanied by a Lyapunov-based stability proof. In multi-domain simulations of both cubic and a custom planar triangular trajectory, the proposed adaptive modular controller achieves sub-centimeter Cartesian tracking accuracy. Experimental validation of the same 1-DoF platform under realistic load emulation confirms the efficacy of the proposed control strategy. These findings demonstrate that a surrogate-enhanced EMLA model embedded in the VDC approach can enable modular, real-time control of an all-electric HDRM, supporting its deployment in next-generation mobile working machines.
>
---
#### [new 026] A Humanoid Social Robot as a Teaching Assistant in the Classroom
- **分类: cs.HC; cs.RO**

- **简介: 论文研究社会机器人作为教学助手的可行性，解决传统教学中教师负担过重问题，通过Pepper机器人结合ChatGPT在课堂中教学，验证其技术实现与学生接受度，发现学生认可其教学效果。**

- **链接: [http://arxiv.org/pdf/2508.05646v1](http://arxiv.org/pdf/2508.05646v1)**

> **作者:** Thomas Sievers
>
> **摘要:** Although innovation and the support of new technologies are much needed to ease the burden on the education system, social robots in schools to help teachers with educational tasks are rare. Child-Robot Interaction (CRI) could support teachers and add an embodied social component to modern multi-modal and multi-sensory learning environments already in use. The social robot Pepper, connected to the Large Language Model (LLM) ChatGPT, was used in a high school classroom to teach new learning content to groups of students. I tested the technical possibilities with the robot on site and asked the students about their acceptance and perceived usefulness of teaching with the help of a social robot. All participants felt that the robot's presentation of the learning material was appropriate or at least partially appropriate and that its use made sense.
>
---
#### [new 027] GMF-Drive: Gated Mamba Fusion with Spatial-Aware BEV Representation for End-to-End Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 论文提出GMF-Drive框架，解决自动驾驶中Transformer融合的高计算复杂度和空间先验不足问题，通过几何增强LiDAR表示与高效SSM融合，提升BEV场景建模能力。**

- **链接: [http://arxiv.org/pdf/2508.06113v1](http://arxiv.org/pdf/2508.06113v1)**

> **作者:** Jian Wang; Chaokang Jiang; Haitao Xu
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Diffusion-based models are redefining the state-of-the-art in end-to-end autonomous driving, yet their performance is increasingly hampered by a reliance on transformer-based fusion. These architectures face fundamental limitations: quadratic computational complexity restricts the use of high-resolution features, and a lack of spatial priors prevents them from effectively modeling the inherent structure of Bird's Eye View (BEV) representations. This paper introduces GMF-Drive (Gated Mamba Fusion for Driving), an end-to-end framework that overcomes these challenges through two principled innovations. First, we supersede the information-limited histogram-based LiDAR representation with a geometrically-augmented pillar format encoding shape descriptors and statistical features, preserving critical 3D geometric details. Second, we propose a novel hierarchical gated mamba fusion (GM-Fusion) architecture that substitutes an expensive transformer with a highly efficient, spatially-aware state-space model (SSM). Our core BEV-SSM leverages directional sequencing and adaptive fusion mechanisms to capture long-range dependencies with linear complexity, while explicitly respecting the unique spatial properties of the driving scene. Extensive experiments on the challenging NAVSIM benchmark demonstrate that GMF-Drive achieves a new state-of-the-art performance, significantly outperforming DiffusionDrive. Comprehensive ablation studies validate the efficacy of each component, demonstrating that task-specific SSMs can surpass a general-purpose transformer in both performance and efficiency for autonomous driving.
>
---
#### [new 028] Graph-based Robot Localization Using a Graph Neural Network with a Floor Camera and a Feature Rich Industrial Floor
- **分类: cs.CV; cs.RO**

- **简介: 论文提出基于图神经网络的机器人定位方法，利用地板特征图表示，实现高精度（0.64cm）和高效定位，解决复杂环境下的kidnapped robot问题。**

- **链接: [http://arxiv.org/pdf/2508.06177v1](http://arxiv.org/pdf/2508.06177v1)**

> **作者:** Dominik Brämer; Diana Kleingarn; Oliver Urbann
>
> **备注:** Accepted at 28th RoboCup International Symposium, Salvador, Brasil
>
> **摘要:** Accurate localization represents a fundamental challenge in robotic navigation. Traditional methodologies, such as Lidar or QR-code based systems, suffer from inherent scalability and adaptability con straints, particularly in complex environments. In this work, we propose an innovative localization framework that harnesses flooring characteris tics by employing graph-based representations and Graph Convolutional Networks (GCNs). Our method uses graphs to represent floor features, which helps localize the robot more accurately (0.64cm error) and more efficiently than comparing individual image features. Additionally, this approach successfully addresses the kidnapped robot problem in every frame without requiring complex filtering processes. These advancements open up new possibilities for robotic navigation in diverse environments.
>
---
#### [new 029] Towards Transparent Ethical AI: A Roadmap for Trustworthy Robotic Systems
- **分类: cs.CY; cs.AI; cs.HC; cs.LG; cs.RO; 68T01, 68T40; K.7.4; K.4.1; I.2.9; H.1.2**

- **简介: 论文提出透明伦理AI框架，解决机器人系统伦理问题，通过标准化、可解释AI和用户界面提升透明度，强化问责与信任，推动负责任AI发展。**

- **链接: [http://arxiv.org/pdf/2508.05846v1](http://arxiv.org/pdf/2508.05846v1)**

> **作者:** Ahmad Farooq; Kamran Iqbal
>
> **备注:** Published in the Proceedings of the 2025 3rd International Conference on Robotics, Control and Vision Engineering (RCVE'25). 6 pages, 3 tables
>
> **摘要:** As artificial intelligence (AI) and robotics increasingly permeate society, ensuring the ethical behavior of these systems has become paramount. This paper contends that transparency in AI decision-making processes is fundamental to developing trustworthy and ethically aligned robotic systems. We explore how transparency facilitates accountability, enables informed consent, and supports the debugging of ethical algorithms. The paper outlines technical, ethical, and practical challenges in implementing transparency and proposes novel approaches to enhance it, including standardized metrics, explainable AI techniques, and user-friendly interfaces. This paper introduces a framework that connects technical implementation with ethical considerations in robotic systems, focusing on the specific challenges of achieving transparency in dynamic, real-world contexts. We analyze how prioritizing transparency can impact public trust, regulatory policies, and avenues for future research. By positioning transparency as a fundamental element in ethical AI system design, we aim to add to the ongoing discussion on responsible AI and robotics, providing direction for future advancements in this vital field.
>
---
#### [new 030] PASG: A Closed-Loop Framework for Automated Geometric Primitive Extraction and Semantic Anchoring in Robotic Manipulation
- **分类: cs.CV; cs.RO**

- **简介: 论文提出PASG框架，解决机器人操作中高阶语义与低阶几何特征碎片化问题，通过几何特征聚合自动提取基本体并结合VLM实现语义锚定，建立空间语义推理基准，实现细粒度语义-具身理解。**

- **链接: [http://arxiv.org/pdf/2508.05976v1](http://arxiv.org/pdf/2508.05976v1)**

> **作者:** Zhihao Zhu; Yifan Zheng; Siyu Pan; Yaohui Jin; Yao Mu
>
> **备注:** Accepted to ICCV 2025. 8 pages main paper, 8 figures, plus supplementary material
>
> **摘要:** The fragmentation between high-level task semantics and low-level geometric features remains a persistent challenge in robotic manipulation. While vision-language models (VLMs) have shown promise in generating affordance-aware visual representations, the lack of semantic grounding in canonical spaces and reliance on manual annotations severely limit their ability to capture dynamic semantic-affordance relationships. To address these, we propose Primitive-Aware Semantic Grounding (PASG), a closed-loop framework that introduces: (1) Automatic primitive extraction through geometric feature aggregation, enabling cross-category detection of keypoints and axes; (2) VLM-driven semantic anchoring that dynamically couples geometric primitives with functional affordances and task-relevant description; (3) A spatial-semantic reasoning benchmark and a fine-tuned VLM (Qwen2.5VL-PA). We demonstrate PASG's effectiveness in practical robotic manipulation tasks across diverse scenarios, achieving performance comparable to manual annotations. PASG achieves a finer-grained semantic-affordance understanding of objects, establishing a unified paradigm for bridging geometric primitives with task semantics in robotic manipulation.
>
---
#### [new 031] Depth Jitter: Seeing through the Depth
- **分类: cs.CV; cs.RO**

- **简介: 论文提出Depth-Jitter技术，通过自适应深度偏移模拟自然深度变化，提升模型在真实深度条件下的稳定性和泛化能力，解决传统方法忽略深度感知导致的鲁棒性不足问题。**

- **链接: [http://arxiv.org/pdf/2508.06227v1](http://arxiv.org/pdf/2508.06227v1)**

> **作者:** Md Sazidur Rahman; David Cabecinhas; Ricard Marxer
>
> **摘要:** Depth information is essential in computer vision, particularly in underwater imaging, robotics, and autonomous navigation. However, conventional augmentation techniques overlook depth aware transformations, limiting model robustness in real world depth variations. In this paper, we introduce Depth-Jitter, a novel depth-based augmentation technique that simulates natural depth variations to improve generalization. Our approach applies adaptive depth offsetting, guided by depth variance thresholds, to generate synthetic depth perturbations while preserving structural integrity. We evaluate Depth-Jitter on two benchmark datasets, FathomNet and UTDAC2020 demonstrating its impact on model stability under diverse depth conditions. Extensive experiments compare Depth-Jitter against traditional augmentation strategies such as ColorJitter, analyzing performance across varying learning rates, encoders, and loss functions. While Depth-Jitter does not always outperform conventional methods in absolute performance, it consistently enhances model stability and generalization in depth-sensitive environments. These findings highlight the potential of depth-aware augmentation for real-world applications and provide a foundation for further research into depth-based learning strategies. The proposed technique is publicly available to support advancements in depth-aware augmentation. The code is publicly available on \href{https://github.com/mim-team/Depth-Jitter}{github}.
>
---
#### [new 032] ME$^3$-BEV: Mamba-Enhanced Deep Reinforcement Learning for End-to-End Autonomous Driving with BEV-Perception
- **分类: cs.AI; cs.RO**

- **简介: 论文提出ME$^3$-BEV模型，通过融合BEV感知与Mamba框架提升自主驾驶的时空特征提取能力，解决传统方法的误差传播和计算瓶颈，优化动态城市驾驶性能。**

- **链接: [http://arxiv.org/pdf/2508.06074v1](http://arxiv.org/pdf/2508.06074v1)**

> **作者:** Siyi Lu; Run Liu; Dongsheng Yang; Lei He
>
> **摘要:** Autonomous driving systems face significant challenges in perceiving complex environments and making real-time decisions. Traditional modular approaches, while offering interpretability, suffer from error propagation and coordination issues, whereas end-to-end learning systems can simplify the design but face computational bottlenecks. This paper presents a novel approach to autonomous driving using deep reinforcement learning (DRL) that integrates bird's-eye view (BEV) perception for enhanced real-time decision-making. We introduce the \texttt{Mamba-BEV} model, an efficient spatio-temporal feature extraction network that combines BEV-based perception with the Mamba framework for temporal feature modeling. This integration allows the system to encode vehicle surroundings and road features in a unified coordinate system and accurately model long-range dependencies. Building on this, we propose the \texttt{ME$^3$-BEV} framework, which utilizes the \texttt{Mamba-BEV} model as a feature input for end-to-end DRL, achieving superior performance in dynamic urban driving scenarios. We further enhance the interpretability of the model by visualizing high-dimensional features through semantic segmentation, providing insight into the learned representations. Extensive experiments on the CARLA simulator demonstrate that \texttt{ME$^3$-BEV} outperforms existing models across multiple metrics, including collision rate and trajectory accuracy, offering a promising solution for real-time autonomous driving.
>
---
#### [new 033] Safety of Embodied Navigation: A Survey
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于综述任务，旨在分析具身导航的安全性问题，涵盖攻击策略、防御机制及评估方法，探讨现有挑战与未来方向，以提升系统安全性。**

- **链接: [http://arxiv.org/pdf/2508.05855v1](http://arxiv.org/pdf/2508.05855v1)**

> **作者:** Zixia Wang; Jia Hu; Ronghui Mu
>
> **摘要:** As large language models (LLMs) continue to advance and gain influence, the development of embodied AI has accelerated, drawing significant attention, particularly in navigation scenarios. Embodied navigation requires an agent to perceive, interact with, and adapt to its environment while moving toward a specified target in unfamiliar settings. However, the integration of embodied navigation into critical applications raises substantial safety concerns. Given their deployment in dynamic, real-world environments, ensuring the safety of such systems is critical. This survey provides a comprehensive analysis of safety in embodied navigation from multiple perspectives, encompassing attack strategies, defense mechanisms, and evaluation methodologies. Beyond conducting a comprehensive examination of existing safety challenges, mitigation technologies, and various datasets and metrics that assess effectiveness and robustness, we explore unresolved issues and future research directions in embodied navigation safety. These include potential attack methods, mitigation strategies, more reliable evaluation techniques, and the implementation of verification frameworks. By addressing these critical gaps, this survey aims to provide valuable insights that can guide future research toward the development of safer and more reliable embodied navigation systems. Furthermore, the findings of this study have broader implications for enhancing societal safety and increasing industrial efficiency.
>
---
## 更新

#### [replaced 001] Learning to Initialize Trajectory Optimization for Vision-Based Autonomous Flight in Unknown Environments
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2309.10683v2](http://arxiv.org/pdf/2309.10683v2)**

> **作者:** Yicheng Chen; Jinjie Li; Wenyuan Qin; Yongzhao Hua; Xiwang Dong; Qingdong Li
>
> **备注:** Accepted to IROS 2025. Source code available
>
> **摘要:** Autonomous flight in unknown environments requires precise spatial and temporal trajectory planning, often involving computationally expensive nonconvex optimization prone to local optima. To overcome these challenges, we present the Neural-Enhanced Trajectory Planner (NEO-Planner), a novel approach that leverages a Neural Network (NN) Planner to provide informed initial values for trajectory optimization. The NN-Planner is trained on a dataset generated by an expert planner using batch sampling, capturing multimodal trajectory solutions. It learns to predict spatial and temporal parameters for trajectories directly from raw sensor observations. NEO-Planner starts optimization from these predictions, accelerating computation speed while maintaining explainability. Furthermore, we introduce a robust online replanning framework that accommodates planning latency for smooth trajectory tracking. Extensive simulations demonstrate that NEO-Planner reduces optimization iterations by 20%, leading to a 26% decrease in computation time compared with pure optimization-based methods. It maintains trajectory quality comparable to baseline approaches and generalizes well to unseen environments. Real-world experiments validate its effectiveness for autonomous drone navigation in cluttered, unknown environments.
>
---
#### [replaced 002] GhostShell: Streaming LLM Function Calls for Concurrent Embodied Programming
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.05298v2](http://arxiv.org/pdf/2508.05298v2)**

> **作者:** Jian Gong; Youwei Huang; Bo Yuan; Ming Zhu; Zhou Liao; Jianhang Liang; Juncheng Zhan; Jinke Wang; Hang Shu; Mingyue Xiong; Yanjun Ye; Yufan Zu; Yang Zhou; Yihan Ding; Xuannian Chen; Xingyu Lu; Runjie Ban; Bingchao Huang; Fusen Liu
>
> **备注:** 17 pages, 5 figures, conference
>
> **摘要:** We present GhostShell, a novel approach that leverages Large Language Models (LLMs) to enable streaming and concurrent behavioral programming for embodied systems. In contrast to conventional methods that rely on pre-scheduled action sequences or behavior trees, GhostShell drives embodied systems to act on-the-fly by issuing function calls incrementally as tokens are streamed from the LLM. GhostShell features a streaming XML function token parser, a dynamic function interface mapper, and a multi-channel scheduler that orchestrates intra-channel synchronous and inter-channel asynchronous function calls, thereby coordinating serial-parallel embodied actions across multiple robotic components under LLM guidance. We evaluate GhostShell on our robotic prototype COCO through comprehensive grounded experiments across 34 real-world interaction tasks and multiple LLM backends. The results demonstrate that our approach achieves a state-of-the-art Behavioral Correctness Metric of 0.85 with Claude-4-Sonnet, and up to 66X faster response times compared to native LLM function calling APIs. GhostShell also proves effective in long-horizon multimodal tasks, exhibiting strong robustness and generalization capabilities.
>
---
#### [replaced 003] An improved two-dimensional time-to-collision for articulated vehicles: predicting sideswipe and rear-end collisions
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.04184v2](http://arxiv.org/pdf/2507.04184v2)**

> **作者:** Abhijeet Behera; Sogol Kharrazi; Erik Frisk; Maytheewat Aramrattana
>
> **摘要:** Time-to-collision (TTC) is a widely used measure for predicting rear-end collisions, assuming constant speed and heading for both vehicles in the prediction horizon. However, this conventional formulation cannot detect sideswipe collisions. A two-dimensional extension, $\text{TTC}_{\text{2D}}$, has been proposed in the literature to address lateral interactions. However, this formulation assumes both vehicles have the same heading and that their headings remain unchanged during the manoeuvre, in addition to the constant speed and heading assumptions in the prediction horizon. Moreover, its use for articulated vehicles like a tractor-semitrailer remains unclear. This paper proposes three enhanced versions of $\text{TTC}_{\text{2D}}$ to overcome these limitations. The first incorporates the vehicle heading to account for directional differences. The standard assumption of constant speed and heading in the prediction horizon holds. The second adapts the formulation for articulated vehicles, and the third allows for constant acceleration, relaxing the constant speed assumption in the prediction horizon. All versions are evaluated in simulated cut-in scenarios, covering both sideswipe and rear-end collisions, using the CARLA simulation environment with a tractor-semitrailer model. Results show that the proposed versions predict sideswipe collisions with better accuracy compared to existing $\text{TTC}_{\text{2D}}$. They also detect rear-end collisions similar to the existing methods.
>
---
#### [replaced 004] MBA-SLAM: Motion Blur Aware Gaussian Splatting SLAM
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2411.08279v2](http://arxiv.org/pdf/2411.08279v2)**

> **作者:** Peng Wang; Lingzhe Zhao; Yin Zhang; Shiyu Zhao; Peidong Liu
>
> **备注:** Accepted to TPAMI; Deblur Gaussian Splatting SLAM
>
> **摘要:** Emerging 3D scene representations, such as Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS), have demonstrated their effectiveness in Simultaneous Localization and Mapping (SLAM) for photo-realistic rendering, particularly when using high-quality video sequences as input. However, existing methods struggle with motion-blurred frames, which are common in real-world scenarios like low-light or long-exposure conditions. This often results in a significant reduction in both camera localization accuracy and map reconstruction quality. To address this challenge, we propose a dense visual deblur SLAM pipeline (i.e. MBA-SLAM) to handle severe motion-blurred inputs and enhance image deblurring. Our approach integrates an efficient motion blur-aware tracker with either neural radiance fields or Gaussian Splatting based mapper. By accurately modeling the physical image formation process of motion-blurred images, our method simultaneously learns 3D scene representation and estimates the cameras' local trajectory during exposure time, enabling proactive compensation for motion blur caused by camera movement. In our experiments, we demonstrate that MBA-SLAM surpasses previous state-of-the-art methods in both camera localization and map reconstruction, showcasing superior performance across a range of datasets, including synthetic and real datasets featuring sharp images as well as those affected by motion blur, highlighting the versatility and robustness of our approach. Code is available at https://github.com/WU-CVGL/MBA-SLAM.
>
---
#### [replaced 005] Hand-Eye Autonomous Delivery: Learning Humanoid Navigation, Locomotion and Reaching
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.03068v2](http://arxiv.org/pdf/2508.03068v2)**

> **作者:** Sirui Chen; Yufei Ye; Zi-Ang Cao; Jennifer Lew; Pei Xu; C. Karen Liu
>
> **摘要:** We propose Hand-Eye Autonomous Delivery (HEAD), a framework that learns navigation, locomotion, and reaching skills for humanoids, directly from human motion and vision perception data. We take a modular approach where the high-level planner commands the target position and orientation of the hands and eyes of the humanoid, delivered by the low-level policy that controls the whole-body movements. Specifically, the low-level whole-body controller learns to track the three points (eyes, left hand, and right hand) from existing large-scale human motion capture data while high-level policy learns from human data collected by Aria glasses. Our modular approach decouples the ego-centric vision perception from physical actions, promoting efficient learning and scalability to novel scenes. We evaluate our method both in simulation and in the real-world, demonstrating humanoid's capabilities to navigate and reach in complex environments designed for humans.
>
---
#### [replaced 006] Would you let a humanoid play storytelling with your child? A usability study on LLM-powered narrative Human-Robot Interaction
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.02505v2](http://arxiv.org/pdf/2508.02505v2)**

> **作者:** Maria Lombardi; Carmela Calabrese; Davide Ghiglino; Caterina Foglino; Davide De Tommaso; Giulia Da Lisca; Lorenzo Natale; Agnieszka Wykowska
>
> **摘要:** A key challenge in human-robot interaction research lies in developing robotic systems that can effectively perceive and interpret social cues, facilitating natural and adaptive interactions. In this work, we present a novel framework for enhancing the attention of the iCub humanoid robot by integrating advanced perceptual abilities to recognise social cues, understand surroundings through generative models, such as ChatGPT, and respond with contextually appropriate social behaviour. Specifically, we propose an interaction task implementing a narrative protocol (storytelling task) in which the human and the robot create a short imaginary story together, exchanging in turn cubes with creative images placed on them. To validate the protocol and the framework, experiments were performed to quantify the degree of usability and the quality of experience perceived by participants interacting with the system. Such a system can be beneficial in promoting effective human robot collaborations, especially in assistance, education and rehabilitation scenarios where the social awareness and the robot responsiveness play a pivotal role.
>
---
#### [replaced 007] Code-as-Symbolic-Planner: Foundation Model-Based Robot Planning via Symbolic Code Generation
- **分类: cs.RO; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2503.01700v2](http://arxiv.org/pdf/2503.01700v2)**

> **作者:** Yongchao Chen; Yilun Hao; Yang Zhang; Chuchu Fan
>
> **备注:** 7 pages, 7 figures, 3 tables
>
> **摘要:** Recent works have shown great potentials of Large Language Models (LLMs) in robot task and motion planning (TAMP). Current LLM approaches generate text- or code-based reasoning chains with sub-goals and action plans. However, they do not fully leverage LLMs' symbolic computing and code generation capabilities. Many robot TAMP tasks involve complex optimization under multiple constraints, where pure textual reasoning is insufficient. While augmenting LLMs with predefined solvers and planners improves performance, it lacks generalization across tasks. Given LLMs' growing coding proficiency, we enhance their TAMP capabilities by steering them to generate code as symbolic planners for optimization and constraint verification. Unlike prior work that uses code to interface with robot action modules, we steer LLMs to generate code as solvers, planners, and checkers for TAMP tasks requiring symbolic computing, while still leveraging textual reasoning to incorporate common sense. With a multi-round guidance and answer evolution framework, the proposed Code-as-Symbolic-Planner improves success rates by average 24.1\% over best baseline methods across seven typical TAMP tasks and three popular LLMs. Code-as-Symbolic-Planner shows strong effectiveness and generalizability across discrete and continuous environments, 2D/3D simulations and real-world settings, as well as single- and multi-robot tasks with diverse requirements. See our project website https://yongchao98.github.io/Code-Symbol-Planner/ for prompts, videos, and code.
>
---
#### [replaced 008] Unified Multi-Rate Model Predictive Control for a Jet-Powered Humanoid Robot
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2505.16478v2](http://arxiv.org/pdf/2505.16478v2)**

> **作者:** Davide Gorbani; Giuseppe L'Erario; Hosameldin Awadalla Omer Mohamed; Daniele Pucci
>
> **备注:** This paper has been accepted for publication at the 2025 IEEE-RAS 24th International Conference on Humanoid Robots (Humanoids), Seoul, 2025
>
> **摘要:** We propose a novel Model Predictive Control (MPC) framework for a jet-powered flying humanoid robot. The controller is based on a linearised centroidal momentum model to represent the flight dynamics, augmented with a second-order nonlinear model to explicitly account for the slow and nonlinear dynamics of jet propulsion. A key contribution is the introduction of a multi-rate MPC formulation that handles the different actuation rates of the robot's joints and jet engines while embedding the jet dynamics directly into the predictive model. We validated the framework using the jet-powered humanoid robot iRonCub, performing simulations in Mujoco; the simulation results demonstrate the robot's ability to recover from external disturbances and perform stable, non-abrupt flight manoeuvres, validating the effectiveness of the proposed approach.
>
---
#### [replaced 009] Failure-Aware Multi-Robot Coordination for Resilient and Adaptive Target Tracking
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.02529v2](http://arxiv.org/pdf/2508.02529v2)**

> **作者:** Peihan Li; Jiazhen Liu; Yuwei Wu; Lifeng Zhou
>
> **摘要:** Multi-robot coordination is crucial for autonomous systems, yet real-world deployments often encounter various failures. These include both temporary and permanent disruptions in sensing and communication, which can significantly degrade system robustness and performance if not explicitly modeled. Despite its practical importance, failure-aware coordination remains underexplored in the literature. To bridge the gap between idealized conditions and the complexities of real-world environments, we propose a unified failure-aware coordination framework designed to enable resilient and adaptive multi-robot target tracking under both temporary and permanent failure conditions. Our approach systematically distinguishes between two classes of failures: (1) probabilistic and temporary disruptions, where robots recover from intermittent sensing or communication losses by dynamically adapting paths and avoiding inferred danger zones, and (2) permanent failures, where robots lose sensing or communication capabilities irreversibly, requiring sustained, decentralized behavioral adaptation. To handle these scenarios, the robot team is partitioned into subgroups. Robots that remain connected form a communication group and collaboratively plan using partially centralized nonlinear optimization. Robots experiencing permanent disconnection or failure continue to operate independently through decentralized or individual optimization, allowing them to contribute to the task within their local context. We extensively evaluate our method across a range of benchmark variations and conduct a comprehensive assessment under diverse real-world failure scenarios. Results show that our framework consistently achieves robust performance in realistic environments with unknown danger zones, offering a practical and generalizable solution for the multi-robot systems community.
>
---
#### [replaced 010] Direct Robot Configuration Space Construction using Convolutional Encoder-Decoders
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2303.05653v2](http://arxiv.org/pdf/2303.05653v2)**

> **作者:** Christopher Benka; Judah Goldfeder; Carl Gross; Riya Gupta; Hod Lipson
>
> **备注:** 8 pages, 7 figures, 4 tables; Appeared at the ICML 2025 Workshop on Building Physically Plausible World Models
>
> **摘要:** Intelligent robots must be able to perform safe and efficient motion planning in their environments. Central to modern motion planning is the configuration space. Configuration spaces define the set of configurations of a robot that result in collisions with obstacles in the workspace, $\text{C}_{\text{clsn}}$, and the set of configurations that do not, $\text{C}_{\text{free}}$. Modern approaches to motion planning first compute the configuration space and then perform motion planning using the calculated configuration space. Real-time motion planning requires accurate and efficient construction of configuration spaces. We are the first to apply a convolutional encoder-decoder framework for calculating highly accurate approximations to configuration spaces, essentially learning how the robot and physical world interact. Our model achieves an average 97.5% F1-score for predicting $\text{C}_{\text{free}}$ and $\text{C}_{\text{clsn}}$ for 2-D robotic workspaces with a dual-arm robot. Our method limits undetected collisions to less than 2.5% on robotic workspaces that involve translation, rotation, and removal of obstacles. Our model learns highly transferable features between robotic workspaces, requiring little to no fine-tuning to adapt to new transformations of obstacles in the workspace.
>
---
#### [replaced 011] CARE: Enhancing Safety of Visual Navigation through Collision Avoidance via Repulsive Estimation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.03834v4](http://arxiv.org/pdf/2506.03834v4)**

> **作者:** Joonkyung Kim; Joonyeol Sim; Woojun Kim; Katia Sycara; Changjoo Nam
>
> **备注:** 16 pages, 6 figures
>
> **摘要:** We propose CARE (Collision Avoidance via Repulsive Estimation) to improve the robustness of learning-based visual navigation methods. Recently, visual navigation models, particularly foundation models, have demonstrated promising performance by generating viable trajectories using only RGB images. However, these policies can generalize poorly to environments containing out-of-distribution (OOD) scenes characterized by unseen objects or different camera setups (e.g., variations in field of view, camera pose, or focal length). Without fine-tuning, such models could produce trajectories that lead to collisions, necessitating substantial efforts in data collection and additional training. To address this limitation, we introduce CARE, an attachable module that enhances the safety of visual navigation without requiring additional range sensors or fine-tuning of pretrained models. CARE can be integrated seamlessly into any RGB-based navigation model that generates local robot trajectories. It dynamically adjusts trajectories produced by a pretrained model using repulsive force vectors computed from depth images estimated directly from RGB inputs. We evaluate CARE by integrating it with state-of-the-art visual navigation models across diverse robot platforms. Real-world experiments show that CARE significantly reduces collisions (up to 100%) without compromising navigation performance in goal-conditioned navigation, and further improves collision-free travel distance (up to 10.7x) in exploration tasks. Project page: https://airlab-sogang.github.io/CARE/
>
---
#### [replaced 012] LaDi-WM: A Latent Diffusion-based World Model for Predictive Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.11528v2](http://arxiv.org/pdf/2505.11528v2)**

> **作者:** Yuhang Huang; JIazhao Zhang; Shilong Zou; XInwang Liu; Ruizhen Hu; Kai Xu
>
> **备注:** CoRL 2025
>
> **摘要:** Predictive manipulation has recently gained considerable attention in the Embodied AI community due to its potential to improve robot policy performance by leveraging predicted states. However, generating accurate future visual states of robot-object interactions from world models remains a well-known challenge, particularly in achieving high-quality pixel-level representations. To this end, we propose LaDi-WM, a world model that predicts the latent space of future states using diffusion modeling. Specifically, LaDi-WM leverages the well-established latent space aligned with pre-trained Visual Foundation Models (VFMs), which comprises both geometric features (DINO-based) and semantic features (CLIP-based). We find that predicting the evolution of the latent space is easier to learn and more generalizable than directly predicting pixel-level images. Building on LaDi-WM, we design a diffusion policy that iteratively refines output actions by incorporating forecasted states, thereby generating more consistent and accurate results. Extensive experiments on both synthetic and real-world benchmarks demonstrate that LaDi-WM significantly enhances policy performance by 27.9\% on the LIBERO-LONG benchmark and 20\% on the real-world scenario. Furthermore, our world model and policies achieve impressive generalizability in real-world experiments.
>
---
#### [replaced 013] Uncertainty-aware Accurate Elevation Modeling for Off-road Navigation via Neural Processes
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.03890v2](http://arxiv.org/pdf/2508.03890v2)**

> **作者:** Sanghun Jung; Daehoon Gwak; Byron Boots; James Hays
>
> **备注:** CoRL 2025
>
> **摘要:** Terrain elevation modeling for off-road navigation aims to accurately estimate changes in terrain geometry in real-time and quantify the corresponding uncertainties. Having precise estimations and uncertainties plays a crucial role in planning and control algorithms to explore safe and reliable maneuver strategies. However, existing approaches, such as Gaussian Processes (GPs) and neural network-based methods, often fail to meet these needs. They are either unable to perform in real-time due to high computational demands, underestimating sharp geometry changes, or harming elevation accuracy when learned with uncertainties. Recently, Neural Processes (NPs) have emerged as a promising approach that integrates the Bayesian uncertainty estimation of GPs with the efficiency and flexibility of neural networks. Inspired by NPs, we propose an effective NP-based method that precisely estimates sharp elevation changes and quantifies the corresponding predictive uncertainty without losing elevation accuracy. Our method leverages semantic features from LiDAR and camera sensors to improve interpolation and extrapolation accuracy in unobserved regions. Also, we introduce a local ball-query attention mechanism to effectively reduce the computational complexity of global attention by 17\% while preserving crucial local and spatial information. We evaluate our method on off-road datasets having interesting geometric features, collected from trails, deserts, and hills. Our results demonstrate superior performance over baselines and showcase the potential of neural processes for effective and expressive terrain modeling in complex off-road environments.
>
---
#### [replaced 014] ImLPR: Image-based LiDAR Place Recognition using Vision Foundation Models
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.18364v2](http://arxiv.org/pdf/2505.18364v2)**

> **作者:** Minwoo Jung; Lanke Frank Tarimo Fu; Maurice Fallon; Ayoung Kim
>
> **备注:** CoRL2025 Accepted, 23 Pages, 15 Figures and 14 Tables
>
> **摘要:** LiDAR Place Recognition (LPR) is a key component in robotic localization, enabling robots to align current scans with prior maps of their environment. While Visual Place Recognition (VPR) has embraced Vision Foundation Models (VFMs) to enhance descriptor robustness, LPR has relied on task-specific models with limited use of pre-trained foundation-level knowledge. This is due to the lack of 3D foundation models and the challenges of using VFM with LiDAR point clouds. To tackle this, we introduce ImLPR, a novel pipeline that employs a pre-trained DINOv2 VFM to generate rich descriptors for LPR. To the best of our knowledge, ImLPR is the first method to utilize a VFM for LPR while retaining the majority of pre-trained knowledge. ImLPR converts raw point clouds into novel three-channel Range Image Views (RIV) to leverage VFM in the LiDAR domain. It employs MultiConv adapters and Patch-InfoNCE loss for effective feature learning. We validate ImLPR on public datasets and outperform state-of-the-art (SOTA) methods across multiple evaluation metrics in both intra- and inter-session LPR. Comprehensive ablations on key design choices such as channel composition, RIV, adapters, and the patch-level loss quantify each component's impact. We release ImLPR as open source for the robotics community: https://github.com/minwoo0611/ImLPR.
>
---
#### [replaced 015] RoboTron-Nav: A Unified Framework for Embodied Navigation Integrating Perception, Planning, and Prediction
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.18525v4](http://arxiv.org/pdf/2503.18525v4)**

> **作者:** Yufeng Zhong; Chengjian Feng; Feng Yan; Fanfan Liu; Liming Zheng; Lin Ma
>
> **备注:** ICCV 2025
>
> **摘要:** In language-guided visual navigation, agents locate target objects in unseen environments using natural language instructions. For reliable navigation in unfamiliar scenes, agents should possess strong perception, planning, and prediction capabilities. Additionally, when agents revisit previously explored areas during long-term navigation, they may retain irrelevant and redundant historical perceptions, leading to suboptimal results. In this work, we propose RoboTron-Nav, a unified framework that integrates perception, planning, and prediction capabilities through multitask collaborations on navigation and embodied question answering tasks, thereby enhancing navigation performances. Furthermore, RoboTron-Nav employs an adaptive 3D-aware history sampling strategy to effectively and efficiently utilize historical observations. By leveraging large language model, RoboTron-Nav comprehends diverse commands and complex visual scenes, resulting in appropriate navigation actions. RoboTron-Nav achieves an 81.1% success rate in object goal navigation on the $\mathrm{CHORES}$-$\mathbb{S}$ benchmark, setting a new state-of-the-art performance. Project page: https://yvfengzhong.github.io/RoboTron-Nav
>
---
#### [replaced 016] Observability-Aware Control for Quadrotor Formation Flight with Range-only Measurement
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2411.03747v2](http://arxiv.org/pdf/2411.03747v2)**

> **作者:** H S Helson Go; Ching Lok Chong; Longhao Qian; Hugh H. -T. Liu
>
> **备注:** 36 pages, 5 figures
>
> **摘要:** Cooperative Localization (CL) is a promising approach to achieve safe quadrotor formation flight through precise positioning via low-cost inter-drone sensors. This paper develops an observability-aware control principle tailored to quadrotor formation flight with range-only inter-drone measurement. The control principle is based on a novel approximation of the local observability Gramian (LOG), which we name the Short-Term Local Observability Gramian (STLOG). The validity of STLOG is established by a proof of the link between local observability and estimation precision. We propose the Observability Predictive Controller (OPC), an implementation of our control principle under a receding-horizon framework, which optimizes a metric of the STLOG to maximize the minimum precision improvement along a trajectory. Monte Carlo simulations and experimental flight tests are conducted on a pair of quadrotors performing formation flight. The results show that the OPC improves positioning precision and estimator confidence, confirming the practical utility of the proposed approach.
>
---
#### [replaced 017] CANVAS: Commonsense-Aware Navigation System for Intuitive Human-Robot Interaction
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.01273v3](http://arxiv.org/pdf/2410.01273v3)**

> **作者:** Suhwan Choi; Yongjun Cho; Minchan Kim; Jaeyoon Jung; Myunchul Joe; Yubeen Park; Minseo Kim; Sungwoong Kim; Sungjae Lee; Hwiseong Park; Jiwan Chung; Youngjae Yu
>
> **备注:** Accepted to ICRA 2025, project page https://worv-ai.github.io/canvas
>
> **摘要:** Real-life robot navigation involves more than just reaching a destination; it requires optimizing movements while addressing scenario-specific goals. An intuitive way for humans to express these goals is through abstract cues like verbal commands or rough sketches. Such human guidance may lack details or be noisy. Nonetheless, we expect robots to navigate as intended. For robots to interpret and execute these abstract instructions in line with human expectations, they must share a common understanding of basic navigation concepts with humans. To this end, we introduce CANVAS, a novel framework that combines visual and linguistic instructions for commonsense-aware navigation. Its success is driven by imitation learning, enabling the robot to learn from human navigation behavior. We present COMMAND, a comprehensive dataset with human-annotated navigation results, spanning over 48 hours and 219 km, designed to train commonsense-aware navigation systems in simulated environments. Our experiments show that CANVAS outperforms the strong rule-based system ROS NavStack across all environments, demonstrating superior performance with noisy instructions. Notably, in the orchard environment, where ROS NavStack records a 0% total success rate, CANVAS achieves a total success rate of 67%. CANVAS also closely aligns with human demonstrations and commonsense constraints, even in unseen environments. Furthermore, real-world deployment of CANVAS showcases impressive Sim2Real transfer with a total success rate of 69%, highlighting the potential of learning from human demonstrations in simulated environments for real-world applications.
>
---
#### [replaced 018] Human-Machine Shared Control Approach for the Takeover of CACC
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2407.11551v4](http://arxiv.org/pdf/2407.11551v4)**

> **作者:** Haoran Wang; Zhexi Lian; Zhenning Li; Jiawei Wang; Arno Eichberger; Jia Hu; Yongyu Chen; Yongji Gao
>
> **备注:** This article has been published on IEEE Transactions on Intelligent Transportation Systems (2025)
>
> **摘要:** Cooperative Adaptive Cruise Control (CACC) often requires human takeover for tasks such as exiting a freeway. Direct human takeover can pose significant risks, especially given the close-following strategy employed by CACC, which might cause drivers to feel unsafe and execute hard braking, potentially leading to collisions. This research aims to develop a CACC takeover controller that ensures a smooth transition from automated to human control. The proposed CACC takeover maneuver employs an indirect human-machine shared control approach, modeled as a Stackelberg competition where the machine acts as the leader and the human as the follower. The machine guides the human to respond in a manner that aligns with the machine's expectations, aiding in maintaining following stability. Additionally, the human reaction function is integrated into the machine's predictive control system, moving beyond a simple "prediction-planning" pipeline to enhance planning optimality. The controller has been verified to i) enable a smooth takeover maneuver of CACC; ii) ensure string stability in the condition that the platoon has less than 6 CAVs and human control authority is less than 40%; iii) enhance both perceived and actual safety through machine interventions; and iv) reduce the impact on upstream traffic by up to 60%.
>
---
#### [replaced 019] From Autonomy to Agency: Agentic Vehicles for Human-Centered Mobility Systems
- **分类: cs.CY; cs.CE; cs.CL; cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.04996v2](http://arxiv.org/pdf/2507.04996v2)**

> **作者:** Jiangbo Yu
>
> **摘要:** Autonomy, from the Greek autos (self) and nomos (law), refers to the capacity to operate according to internal rules without external control. Accordingly, autonomous vehicles (AuVs) are defined as systems capable of perceiving their environment and executing preprogrammed tasks independently of external input. However, both research and real-world deployments increasingly showcase vehicles that demonstrate behaviors beyond this definition (including the SAE levels 1 to 6), such as interaction with humans and machines, goal adaptation, contextual reasoning, external tool use, and long-term planning, particularly with the integration of large language models (LLMs) and agentic AI systems. These developments reveal a conceptual gap between technical autonomy and the broader cognitive and social capabilities needed for future human-centered mobility systems. To address this, we introduce the concept of agentic vehicles (AgVs), referring to vehicles that integrate agentic AI to reason, adapt, and interact within complex environments. This paper presents a systems-level framework to characterize AgVs, focusing on their cognitive and communicative layers and differentiating them from conventional AuVs. It synthesizes relevant advances in agentic AI, robotics, multi-agent systems, and human-machine interaction, and highlights how agentic AI, through high-level reasoning and tool use, can function not merely as computational tools but as interactive agents embedded in mobility ecosystems. The paper concludes by identifying key challenges in the development and governance of AgVs, including safety, real-time control, public acceptance, ethical alignment, and regulatory frameworks.
>
---
