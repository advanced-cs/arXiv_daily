# 机器人 cs.RO

- **最新发布 45 篇**

- **更新 28 篇**

## 最新发布

#### [new 001] IG-RFT: An Interaction-Guided RL Framework for VLA Models in Long-Horizon Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型在长时序复杂任务中的泛化问题。提出IG-RFT框架，结合强化学习与交互引导策略，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2602.20715v1](https://arxiv.org/pdf/2602.20715v1)**

> **作者:** Zhian Su; Weijie Kong; Haonan Dong; Huixu Dong
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated significant potential for generalist robotic policies; however, they struggle to generalize to long-horizon complex tasks in novel real-world domains due to distribution shifts and the scarcity of high-quality demonstrations. Although reinforcement learning (RL) offers a promising avenue for policy improvement, applying it to real-world VLA fine-tuning faces challenges regarding exploration efficiency, training stability, and sample cost. To address these issues, we propose IG-RFT, a novel Interaction-Guided Reinforced Fine-Tuning system designed for flow-based VLA models. Firstly, to facilitate effective policy optimization, we introduce Interaction-Guided Advantage Weighted Regression (IG-AWR), an RL algorithm that dynamically modulates exploration intensity based on the robot's interaction status. Furthermore, to address the limitations of sparse or task-specific rewards, we design a novel hybrid dense reward function that integrates the trajectory-level reward and the subtask-level reward. Finally, we construct a three-stage RL system comprising SFT, Offline RL, and Human-in-the-Loop RL for fine-tuning VLA models. Extensive real-world experiments on four challenging long-horizon tasks demonstrate that IG-RFT achieves an average success rate of 85.0%, significantly outperforming SFT (18.8%) and standard Offline RL baselines (40.0%). Ablation studies confirm the critical contributions of IG-AWR and hybrid reward shaping. In summary, our work establishes and validates a novel reinforced fine-tuning system for VLA models in real-world robotic manipulation.
>
---
#### [new 002] Conflict-Based Search for Multi-Agent Path Finding with Elevators
- **分类: cs.RO**

- **简介: 该论文研究多智能体路径规划中的电梯问题（MAPF-E），解决多智能体在不同楼层间通过电梯移动时的冲突问题，扩展了传统路径规划任务。**

- **链接: [https://arxiv.org/pdf/2602.20512v1](https://arxiv.org/pdf/2602.20512v1)**

> **作者:** Haitong He; Xuemian Wu; Shizhe Zhao; Zhongqiang Ren
>
> **摘要:** This paper investigates a problem called Multi-Agent Path Finding with Elevators (MAPF-E), which seeks conflict-free paths for multiple agents from their start to goal locations that may locate on different floors, and the agents can use elevators to travel between floors. The existence of elevators complicates the interaction among the agents and introduces new challenges to the planning. On the one hand, elevators can cause many conflicts among the agents due to its relatively long traversal time across floors, especially when many agents need to reach a different floor. On the other hand, the planner has to reason in a larger state space including the states of the elevators, besides the locations of the agents.
>
---
#### [new 003] Efficient Hierarchical Any-Angle Path Planning on Multi-Resolution 3D Grids
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于路径规划任务，解决高分辨率地图下路径规划效率与计算可行性问题。通过利用多分辨率网格，提出一种高效、最优的任意角度路径规划方法。**

- **链接: [https://arxiv.org/pdf/2602.21174v1](https://arxiv.org/pdf/2602.21174v1)**

> **作者:** Victor Reijgwart; Cesar Cadena; Roland Siegwart; Lionel Ott
>
> **备注:** 12 pages, 9 figures, 4 tables, accepted to RSS 2025, code is open-source: https://github.com/ethz-asl/wavestar
>
> **摘要:** Hierarchical, multi-resolution volumetric mapping approaches are widely used to represent large and complex environments as they can efficiently capture their occupancy and connectivity information. Yet widely used path planning methods such as sampling and trajectory optimization do not exploit this explicit connectivity information, and search-based methods such as A* suffer from scalability issues in large-scale high-resolution maps. In many applications, Euclidean shortest paths form the underpinning of the navigation system. For such applications, any-angle planning methods, which find optimal paths by connecting corners of obstacles with straight-line segments, provide a simple and efficient solution. In this paper, we present a method that has the optimality and completeness properties of any-angle planners while overcoming computational tractability issues common to search-based methods by exploiting multi-resolution representations. Extensive experiments on real and synthetic environments demonstrate the proposed approach's solution quality and speed, outperforming even sampling-based methods. The framework is open-sourced to allow the robotics and planning community to build on our research.
>
---
#### [new 004] Sample-Efficient Learning with Online Expert Correction for Autonomous Catheter Steering in Endovascular Bifurcation Navigation
- **分类: cs.RO**

- **简介: 该论文属于自主导管导航任务，解决传统强化学习在血管分叉导航中样本效率低和泛化能力差的问题。提出一种结合在线专家修正的高效强化学习框架，提升导航精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.20216v1](https://arxiv.org/pdf/2602.20216v1)**

> **作者:** Hao Wang; Tianliang Yao; Bo Lu; Zhiqiang Pei; Liu Dong; Lei Ma; Peng Qi
>
> **备注:** This paper has been accepted by IEEE ICRA 2026. 8 pages, 5 figures, 1 table
>
> **摘要:** Robot-assisted endovascular intervention offers a safe and effective solution for remote catheter manipulation, reducing radiation exposure while enabling precise navigation. Reinforcement learning (RL) has recently emerged as a promising approach for autonomous catheter steering; however, conventional methods suffer from sparse reward design and reliance on static vascular models, limiting their sample efficiency and generalization to intraoperative variations. To overcome these challenges, this paper introduces a sample-efficient RL framework with online expert correction for autonomous catheter steering in endovascular bifurcation navigation. The proposed framework integrates three key components: (1) A segmentation-based pose estimation module for accurate real-time state feedback, (2) A fuzzy controller for bifurcation-aware orientation adjustment, and (3) A structured reward generator incorporating expert priors to guide policy learning. By leveraging online expert correction, the framework reduces exploration inefficiency and enhances policy robustness in complex vascular structures. Experimental validation on a robotic platform using a transparent vascular phantom demonstrates that the proposed approach achieves convergence in 123 training episodes -- a 25.9% reduction compared to the baseline Soft Actor-Critic (SAC) algorithm -- while reducing average positional error to 83.8% of the baseline. These results indicate that combining sample-efficient RL with online expert correction enables reliable and accurate catheter steering, particularly in anatomically challenging bifurcation scenarios critical for endovascular navigation.
>
---
#### [new 005] Energy-Based Injury Protection Database: Including Shearing Contact Thresholds for Hand and Finger Using Porcine Surrogates
- **分类: cs.RO**

- **简介: 该论文属于人机交互安全研究，旨在解决碰撞伤害评估问题。通过建立基于能量的损伤防护数据库，涵盖剪切接触场景，提升安全控制策略的有效性。**

- **链接: [https://arxiv.org/pdf/2602.20362v1](https://arxiv.org/pdf/2602.20362v1)**

> **作者:** Robin Jeanne Kirschner; Anna Huber; Carina M. Micheler; Dirk Müller; Nader Rajaei; Rainer Burgkart; Sami Haddadin
>
> **备注:** 9 pages, 11 figures
>
> **摘要:** While robotics research continues to propose strategies for collision avoidance in human-robot interaction, the reality of constrained environments and future humanoid systems makes contact inevitable. To mitigate injury risks, energy-constraining control approaches are commonly used, often relying on safety thresholds derived from blunt impact data in EN ISO 10218-2:2025. However, this dataset does not extend to edged or pointed collisions. Without scalable, clinically grounded datasets covering diverse contact scenarios, safety validation remains limited. Previous studies have laid the groundwork by assessing surrogate-based velocity and mass limits across various geometries, focusing on perpendicular impacts. This study expands those datasets by including shearing contact scenarios in unconstrained collisions, revealing that collision angle significantly affects injury outcomes. Notably, unconstrained shearing contacts result in fewer injuries than perpendicular ones. By reevaluating all prior porcine surrogate data, we establish energy thresholds across geometries and contact types, forming the first energy-based Injury Protection Database. This enables the development of meaningful energy-limiting controllers that ensure safety across a wide range of realistic collision events.
>
---
#### [new 006] Global Prior Meets Local Consistency: Dual-Memory Augmented Vision-Language-Action Model for Efficient Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型在生成动作时效率低和鲁棒性差的问题。通过引入双记忆机制提升性能与速度。**

- **链接: [https://arxiv.org/pdf/2602.20200v1](https://arxiv.org/pdf/2602.20200v1)**

> **作者:** Zaijing Li; Bing Hu; Rui Shao; Gongwei Chen; Dongmei Jiang; Pengwei Xie; Jianye Hao; Liqiang Nie
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** Hierarchical Vision-Language-Action (VLA) models have rapidly become a dominant paradigm for robotic manipulation. It typically comprising a Vision-Language backbone for perception and understanding, together with a generative policy for action generation. However, its performance is increasingly bottlenecked by the action generation proceess. (i) Low inference efficiency. A pronounced distributional gap between isotropic noise priors and target action distributions, which increases denoising steps and the incidence of infeasible samples. (ii) Poor robustness. Existing policies condition solely on the current observation, neglecting the constraint of history sequence and thus lacking awareness of task progress and temporal consistency. To address these issues, we introduce OptimusVLA, a dual-memory VLA framework with Global Prior Memory (GPM) and Local Consistency Memory (LCM). GPM replaces Gaussian noise with task-level priors retrieved from semantically similar trajectories, thereby shortening the generative path and reducing the umber of function evaluations (NFE). LCM dynamically models executed action sequence to infer task progress and injects a learned consistency constraint that enforces temporal coherence and smoothness of trajectory. Across three simulation benchmarks, OptimusVLA consistently outperforms strong baselines: it achieves 98.6% average success rate on LIBERO, improves over pi_0 by 13.5% on CALVIN, and attains 38% average success rate on RoboTwin 2.0 Hard. In Real-World evaluation, OptimusVLA ranks best on Generalization and Long-horizon suites, surpassing pi_0 by 42.9% and 52.4%, respectively, while delivering 2.9x inference speedup.
>
---
#### [new 007] Strategy-Supervised Autonomous Laparoscopic Camera Control via Event-Driven Graph Mining
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自主腹腔镜摄像控制任务，解决手术中视角稳定与安全问题。通过事件驱动图挖掘提取策略，结合视觉语言模型实现智能控制。**

- **链接: [https://arxiv.org/pdf/2602.20500v1](https://arxiv.org/pdf/2602.20500v1)**

> **作者:** Keyu Zhou; Peisen Xu; Yahao Wu; Jiming Chen; Gaofeng Li; Shunlei Li
>
> **备注:** Submitted to IEEE Transactions on Robotics (T-RO). 19 pages, 9 figures
>
> **摘要:** Autonomous laparoscopic camera control must maintain a stable and safe surgical view under rapid tool-tissue interactions while remaining interpretable to surgeons. We present a strategy-grounded framework that couples high-level vision-language inference with low-level closed-loop control. Offline, raw surgical videos are parsed into camera-relevant temporal events (e.g., interaction, working-distance deviation, and view-quality degradation) and structured as attributed event graphs. Mining these graphs yields a compact set of reusable camera-handling strategy primitives, which provide structured supervision for learning. Online, a fine-tuned Vision-Language Model (VLM) processes the live laparoscopic view to predict the dominant strategy and discrete image-based motion commands, executed by an IBVS-RCM controller under strict safety constraints; optional speech input enables intuitive human-in-the-loop conditioning. On a surgeon-annotated dataset, event parsing achieves reliable temporal localization (F1-score 0.86), and the mined strategies show strong semantic alignment with expert interpretation (cluster purity 0.81). Extensive ex vivo experiments on silicone phantoms and porcine tissues demonstrate that the proposed system outperforms junior surgeons in standardized camera-handling evaluations, reducing field-of-view centering error by 35.26% and image shaking by 62.33%, while preserving smooth motion and stable working-distance regulation.
>
---
#### [new 008] Acoustic Feedback for Closed-Loop Force Control in Robotic Grinding
- **分类: cs.RO**

- **简介: 该论文属于机器人打磨任务，旨在解决传统力传感成本高、适应性差的问题。通过音频反馈实现闭环力控制，使用低成本麦克风替代力传感器，提升一致性与经济性。**

- **链接: [https://arxiv.org/pdf/2602.20596v1](https://arxiv.org/pdf/2602.20596v1)**

> **作者:** Zongyuan Zhang; Christopher Lehnert; Will N. Browne; Jonathan M. Roberts
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026. 8 pages, 10 figures
>
> **摘要:** Acoustic feedback is a critical indicator for assessing the contact condition between the tool and the workpiece when humans perform grinding tasks with rotary tools. In contrast, robotic grinding systems typically rely on force sensing, with acoustic information largely ignored. This reliance on force sensors is costly and difficult to adapt to different grinding tools, whereas audio sensors (microphones) are low-cost and can be mounted on any medium that conducts grinding sound. This paper introduces a low-cost Acoustic Feedback Robotic Grinding System (AFRG) that captures audio signals with a contact microphone, estimates grinding force from the audio in real time, and enables closed-loop force control of the grinding process. Compared with conventional force-sensing approaches, AFRG achieves a 4-fold improvement in consistency across different grinding disc conditions. AFRG relies solely on a low-cost microphone, which is approximately 200-fold cheaper than conventional force sensors, as the sensing modality, providing an easily deployable, cost-effective robotic grinding solution.
>
---
#### [new 009] UniLACT: Depth-Aware RGB Latent Action Learning for Vision-Language-Action Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型任务，解决RGB视频缺乏3D结构的问题。提出UniLACT模型，结合深度信息进行统一潜在动作学习，提升操作精度。**

- **链接: [https://arxiv.org/pdf/2602.20231v1](https://arxiv.org/pdf/2602.20231v1)**

> **作者:** Manish Kumar Govind; Dominick Reilly; Pu Wang; Srijan Das
>
> **备注:** https://manishgovind.github.io/unilact-vla/
>
> **摘要:** Latent action representations learned from unlabeled videos have recently emerged as a promising paradigm for pretraining vision-language-action (VLA) models without explicit robot action supervision. However, latent actions derived solely from RGB observations primarily encode appearance-driven dynamics and lack explicit 3D geometric structure, which is essential for precise and contact-rich manipulation. To address this limitation, we introduce UniLACT, a transformer-based VLA model that incorporates geometric structure through depth-aware latent pretraining, enabling downstream policies to inherit stronger spatial priors. To facilitate this process, we propose UniLARN, a unified latent action learning framework based on inverse and forward dynamics objectives that learns a shared embedding space for RGB and depth while explicitly modeling their cross-modal interactions. This formulation produces modality-specific and unified latent action representations that serve as pseudo-labels for the depth-aware pretraining of UniLACT. Extensive experiments in both simulation and real-world settings demonstrate the effectiveness of depth-aware unified latent action representations. UniLACT consistently outperforms RGB-based latent action baselines under in-domain and out-of-domain pretraining regimes, as well as on both seen and unseen manipulation tasks.
>
---
#### [new 010] Grasp to Act: Dexterous Grasping for Tool Use in Dynamic Settings
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决动态环境下工具使用时的稳定抓取问题。提出Grasp-to-Act系统，结合物理优化与强化学习，提升抓取鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.20466v1](https://arxiv.org/pdf/2602.20466v1)**

> **作者:** Harsh Gupta; Mohammad Amin Mirzaee; Wenzhen Yuan
>
> **备注:** Result videos can be found at https://grasp2act.github.io/
>
> **摘要:** Achieving robust grasping with dexterous hands remains challenging, especially when manipulation involves dynamic forces such as impacts, torques, and continuous resistance--situations common in real-world tool use. Existing methods largely optimize grasps for static geometric stability and often fail once external forces arise during manipulation. We present Grasp-to-Act, a hybrid system that combines physics-based grasp optimization with reinforcement-learning-based grasp adaptation to maintain stable grasps throughout functional manipulation tasks. Our method synthesizes robust grasp configurations informed by human demonstrations and employs an adaptive controller that residually issues joint corrections to prevent in-hand slip while tracking the object trajectory. Grasp-to-Act enables robust zero-shot sim-to-real transfer across five dynamic tool-use tasks--hammering, sawing, cutting, stirring, and scooping--consistently outperforming baselines. Across simulation and real-world hardware trials with a 16-DoF dexterous hand, our method reduces translational and rotational in-hand slip and achieves the highest task completion rates, demonstrating stable functional grasps under dynamic, contact-rich conditions.
>
---
#### [new 011] Smoothly Differentiable and Efficiently Vectorizable Contact Manifold Generation
- **分类: cs.RO**

- **简介: 该论文属于机器人动力学模拟任务，旨在解决接触面生成的效率与可微性问题。提出一种新的框架，结合高效与可微性，提升模拟速度。**

- **链接: [https://arxiv.org/pdf/2602.20304v1](https://arxiv.org/pdf/2602.20304v1)**

> **作者:** Onur Beker; Andreas René Geist; Anselm Paulus; Nico Gürtler; Ji Shi; Sylvain Calinon; Georg Martius
>
> **摘要:** Simulating rigid-body dynamics with contact in a fast, massively vectorizable, and smoothly differentiable manner is highly desirable in robotics. An important bottleneck faced by existing differentiable simulation frameworks is contact manifold generation: representing the volume of intersection between two colliding geometries via a discrete set of properly distributed contact points. A major factor contributing to this bottleneck is that the related routines of commonly used robotics simulators were not designed with vectorization and differentiability as a primary concern, and thus rely on logic and control flow that hinder these goals. We instead propose a framework designed from the ground up with these goals in mind, by trying to strike a middle ground between: i) convex primitive based approaches used by common robotics simulators (efficient but not differentiable), and ii) mollified vertex-face and edge-edge unsigned distance-based approaches used by barrier methods (differentiable but inefficient). Concretely, we propose: i) a representative set of smooth analytical signed distance primitives to implement vertex-face collisions, and ii) a novel differentiable edge-edge collision routine that can provide signed distances and signed contact normals. The proposed framework is evaluated via a set of didactic experiments and benchmarked against the collision detection routine of the well-established Mujoco XLA framework, where we observe a significant speedup. Supplementary videos can be found at https://github.com/bekeronur/contax, where a reference implementation in JAX will also be made available at the conclusion of the review process.
>
---
#### [new 012] Visual Cooperative Drone Tracking for Open-Path Gas Measurements
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，旨在解决开放路径气体测量中自动采样的难题。通过视觉跟踪和GNSS技术，实现无人机与地面传感器的协同测量。**

- **链接: [https://arxiv.org/pdf/2602.20768v1](https://arxiv.org/pdf/2602.20768v1)**

> **作者:** Marius Schaab; Alisha Kiefer; Thomas Wiedemann; Patrick Hinsen; Achim J. Lilienthal
>
> **摘要:** Open-path Tunable Diode Laser Absorption Spectroscopy offers an effective method for measuring, mapping, and monitoring gas concentrations, such as leaking CO2 or methane. Compared to spatial sampling of gas distributions using in-situ sensors, open-path sensors in combination with gas tomography algorithms can cover large outdoor environments faster in a non-invasive way. However, the requirement of a dedicated reflection surface for the open-path laser makes automating the spatial sampling process challenging. This publication presents a robotic system for collecting open-path measurements, making use of a sensor mounted on a ground-based pan-tilt unit and a small drone carrying a reflector. By means of a zoom camera, the ground unit visually tracks red LED markers mounted on the drone and aligns the sensor's laser beam with the reflector. Incorporating GNSS position information provided by the drone's flight controller further improves the tracking approach. Outdoor experiments validated the system's performance, demonstrating successful autonomous tracking and valid CO2 measurements at distances up to 60 meters. Furthermore, the system successfully measured a CO2 plume without interference from the drone's propulsion system, demonstrating its superiority compared to flying in-situ sensors.
>
---
#### [new 013] KCFRC: Kinematic Collision-Aware Foothold Reachability Criteria for Legged Locomotion
- **分类: cs.RO**

- **简介: 该论文属于腿式机器人运动规划任务，解决足点可达性验证问题。提出KCFRC算法，实现实时足点可达性分析，提升机器人在复杂环境中的适应性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.20850v1](https://arxiv.org/pdf/2602.20850v1)**

> **作者:** Lei Ye; Haibo Gao; Huaiguang Yang; Peng Xu; Haoyu Wang; Tie Liu; Junqi Shan; Zongquan Deng; Liang Ding
>
> **摘要:** Legged robots face significant challenges in navigating complex environments, as they require precise real-time decisions for foothold selection and contact planning. While existing research has explored methods to select footholds based on terrain geometry or kinematics, a critical gap remains: few existing methods efficiently validate the existence of a non-collision swing trajectory. This paper addresses this gap by introducing KCFRC, a novel approach for efficient foothold reachability analysis. We first formally define the foothold reachability problem and establish a sufficient condition for foothold reachability. Based on this condition, we develop the KCFRC algorithm, which enables robots to validate foothold reachability in real time. Our experimental results demonstrate that KCFRC achieves remarkable time efficiency, completing foothold reachability checks for a single leg across 900 potential footholds in an average of 2 ms. Furthermore, we show that KCFRC can accelerate trajectory optimization and is particularly beneficial for contact planning in confined spaces, enhancing the adaptability and robustness of legged robots in challenging environments.
>
---
#### [new 014] Notes-to-Self: Scratchpad Augmented VLAs for Memory Dependent Manipulation Tasks
- **分类: cs.RO**

- **简介: 该论文研究记忆依赖的机械操作任务，解决VLAs在长序列任务中缺乏记忆的问题。通过引入语言便签增强模型的时空记忆能力，提升任务泛化性能。**

- **链接: [https://arxiv.org/pdf/2602.21013v1](https://arxiv.org/pdf/2602.21013v1)**

> **作者:** Sanjay Haresh; Daniel Dijkman; Apratim Bhattacharyya; Roland Memisevic
>
> **备注:** To appear at ICRA 2026
>
> **摘要:** Many dexterous manipulation tasks are non-markovian in nature, yet little attention has been paid to this fact in the recent upsurge of the vision-language-action (VLA) paradigm. Although they are successful in bringing internet-scale semantic understanding to robotics, existing VLAs are primarily "stateless" and struggle with memory-dependent long horizon tasks. In this work, we explore a way to impart both spatial and temporal memory to a VLA by incorporating a language scratchpad. The scratchpad makes it possible to memorize task-specific information, such as object positions, and it allows the model to keep track of a plan and progress towards subgoals within that plan. We evaluate this approach on a split of memory-dependent tasks from the ClevrSkills environment, on MemoryBench, as well as on a challenging real-world pick-and-place task. We show that incorporating a language scratchpad significantly improves generalization on these tasks for both non-recurrent and recurrent models.
>
---
#### [new 015] What Matters for Simulation to Online Reinforcement Learning on Real Robots
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究在线强化学习在真实机器人上的应用，分析影响其性能的设计选择，旨在提升学习稳定性与部署效率。**

- **链接: [https://arxiv.org/pdf/2602.20220v1](https://arxiv.org/pdf/2602.20220v1)**

> **作者:** Yarden As; Dhruva Tirumala; René Zurbrügg; Chenhao Li; Stelian Coros; Andreas Krause; Markus Wulfmeier
>
> **摘要:** We investigate what specific design choices enable successful online reinforcement learning (RL) on physical robots. Across 100 real-world training runs on three distinct robotic platforms, we systematically ablate algorithmic, systems, and experimental decisions that are typically left implicit in prior work. We find that some widely used defaults can be harmful, while a set of robust, readily adopted design choices within standard RL practice yield stable learning across tasks and hardware. These results provide the first large-sample empirical study of such design choices, enabling practitioners to deploy online RL with lower engineering effort.
>
---
#### [new 016] HALO: A Unified Vision-Language-Action Model for Embodied Multimodal Chain-of-Thought Reasoning
- **分类: cs.RO**

- **简介: 该论文提出HALO模型，解决机器人操作中长序列和分布外场景的多模态推理问题。通过统一的文本、视觉和动作推理框架提升任务成功率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.21157v1](https://arxiv.org/pdf/2602.21157v1)**

> **作者:** Quanxin Shou; Fangqi Zhu; Shawn Chen; Puxin Yan; Zhengyang Yan; Yikun Miao; Xiaoyi Pang; Zicong Hong; Ruikai Shi; Hao Huang; Jie Zhang; Song Guo
>
> **摘要:** Vision-Language-Action (VLA) models have shown strong performance in robotic manipulation, but often struggle in long-horizon or out-of-distribution scenarios due to the lack of explicit mechanisms for multimodal reasoning and anticipating how the world will evolve under action. Recent works introduce textual chain-of-thought or visual subgoal prediction within VLA models to reason, but still fail to offer a unified human-like reasoning framework for joint textual reasoning, visual foresight, and action prediction. To this end, we propose HALO, a unified VLA model that enables embodied multimodal chain-of-thought (EM-CoT) reasoning through a sequential process of textual task reasoning, visual subgoal prediction for fine-grained guidance, and EM-CoT-augmented action prediction. We instantiate HALO with a Mixture-of-Transformers (MoT) architecture that decouples semantic reasoning, visual foresight, and action prediction into specialized experts while allowing seamless cross-expert collaboration. To enable HALO learning at scale, we introduce an automated pipeline to synthesize EM-CoT training data along with a carefully crafted training recipe. Extensive experiments demonstrate that: (1) HALO achieves superior performance in both simulated and real-world environments, surpassing baseline policy pi_0 by 34.1% on RoboTwin benchmark; (2) all proposed components of the training recipe and EM-CoT design help improve task success rate; and (3) HALO exhibits strong generalization capabilities under aggressive unseen environmental randomization with our proposed EM-CoT reasoning.
>
---
#### [new 017] Squint: Fast Visual Reinforcement Learning for Sim-to-Real Robotics
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于视觉强化学习领域，解决机器人模拟到现实的高效训练问题。提出Squint方法，通过并行仿真和优化策略，提升训练速度与效果。**

- **链接: [https://arxiv.org/pdf/2602.21203v1](https://arxiv.org/pdf/2602.21203v1)**

> **作者:** Abdulaziz Almuzairee; Henrik I. Christensen
>
> **备注:** For website and code, see https://aalmuzairee.github.io/squint
>
> **摘要:** Visual reinforcement learning is appealing for robotics but expensive -- off-policy methods are sample-efficient yet slow; on-policy methods parallelize well but waste samples. Recent work has shown that off-policy methods can train faster than on-policy methods in wall-clock time for state-based control. Extending this to vision remains challenging, where high-dimensional input images complicate training dynamics and introduce substantial storage and encoding overhead. To address these challenges, we introduce Squint, a visual Soft Actor Critic method that achieves faster wall-clock training than prior visual off-policy and on-policy methods. Squint achieves this via parallel simulation, a distributional critic, resolution squinting, layer normalization, a tuned update-to-data ratio, and an optimized implementation. We evaluate on the SO-101 Task Set, a new suite of eight manipulation tasks in ManiSkill3 with heavy domain randomization, and demonstrate sim-to-real transfer to a real SO-101 robot. We train policies for 15 minutes on a single RTX 3090 GPU, with most tasks converging in under 6 minutes.
>
---
#### [new 018] Computer-Aided Design of Rational Motions for 4R and 6R Spatial Mechanism Synthesis
- **分类: cs.RO**

- **简介: 该论文属于机械设计任务，解决空间机构运动合成问题。通过几何方法生成合理运动，设计四杆和六杆机构，实现指定空间轨迹。**

- **链接: [https://arxiv.org/pdf/2602.20920v1](https://arxiv.org/pdf/2602.20920v1)**

> **作者:** Daniel Huczala; Severinas Zube; Martin Pfurner; Johannes Siegele; Frank C. Park
>
> **摘要:** This paper focuses on geometric methods for generating rational motions used in the design of single-loop rational linkages, 1-degree-of-freedom mechanisms that can execute prescribed spatial tasks. Building on established rational motion synthesis methods, we introduce a new interpolation scheme for seven 3D points based on cubic quaternionic Bezier curves. The resulting motion admits factorization, i.e. the synthesis of a spatial six-bar mechanism whose tool frame passes the specified seven points. To support engineering practice, we provide open-source CAD tools that implement also the other methods and provide fast visual evaluation of motion generation and mechanism synthesis.
>
---
#### [new 019] A Micro-Macro Model of Encounter-Driven Information Diffusion in Robot Swarms
- **分类: cs.RO; cs.MA**

- **简介: 该论文研究机器人群体中基于相遇的信息扩散问题，提出微宏观模型以模拟信息传播过程，旨在为存储与路由算法设计提供理论支持。**

- **链接: [https://arxiv.org/pdf/2602.21148v1](https://arxiv.org/pdf/2602.21148v1)**

> **作者:** Davis S. Catherman; Carlo Pinciroli
>
> **备注:** 10 pages, 5 figures, published at ANTS 2026
>
> **摘要:** In this paper, we propose the problem of Encounter-Driven Information Diffusion (EDID). In EDID, robots are allowed to exchange information only upon meeting. Crucially, EDID assumes that the robots are not allowed to schedule their meetings. As such, the robots have no means to anticipate when, where, and who they will meet. As a step towards the design of storage and routing algorithms for EDID, in this paper we propose a model of information diffusion that captures the essential dynamics of EDID. The model is derived from first principles and is composed of two levels: a micro model, based on a generalization of the concept of `mean free path'; and a macro model, which captures the global dynamics of information diffusion. We validate the model through extensive robot simulations, in which we consider swarm size, communication range, environment size, and different random motion regimes. We conclude the paper with a discussion of the implications of this model on the algorithms that best support information diffusion according to the parameters of interest.
>
---
#### [new 020] FACTO: Function-space Adaptive Constrained Trajectory Optimization for Robotic Manipulators
- **分类: cs.RO**

- **简介: 该论文提出FACTO算法，用于机器人机械臂轨迹优化。解决轨迹规划中的约束问题，通过函数空间优化提升解的质量和可行性。**

- **链接: [https://arxiv.org/pdf/2602.20225v1](https://arxiv.org/pdf/2602.20225v1)**

> **作者:** Yichang Feng; Xiao Liang; Minghui Zheng
>
> **摘要:** This paper introduces Function-space Adaptive Constrained Trajectory Optimization (FACTO), a new trajectory optimization algorithm for both single- and multi-arm manipulators. Trajectory representations are parameterized as linear combinations of orthogonal basis functions, and optimization is performed directly in the coefficient space. The constrained problem formulation consists of both an objective functional and a finite-dimensional objective defined over truncated coefficients. To address nonlinearity, FACTO uses a Gauss-Newton approximation with exponential moving averaging, yielding a smoothed quadratic subproblem. Trajectory-wide constraints are addressed using coefficient-space mappings, and an adaptive constrained update using the Levenberg-Marquardt algorithm is performed in the null space of active constraints. Comparisons with optimization-based planners (CHOMP, TrajOpt, GPMP2) and sampling-based planners (RRT-Connect, RRT*, PRM) show the improved solution quality and feasibility, especially in constrained single- and multi-arm scenarios. The experimental evaluation of FACTO on Franka robots verifies the feasibility of deployment.
>
---
#### [new 021] A Robotic Testing Platform for Pipelined Discovery of Resilient Soft Actuators
- **分类: cs.RO**

- **简介: 该论文属于机器人执行器设计任务，旨在解决线性介电弹性体执行器寿命短的问题。通过构建自动化测试平台，优化参数提升其耐用性，并应用于四足机器人。**

- **链接: [https://arxiv.org/pdf/2602.20963v1](https://arxiv.org/pdf/2602.20963v1)**

> **作者:** Ang; Li; Alexander Yin; Alexander White; Sahib Sandhu; Matthew Francoeur; Victor Jimenez-Santiago; Van Remenar; Codrin Tugui; Mihai Duduta
>
> **摘要:** Short lifetime under high electrical fields hinders the widespread robotic application of linear dielectric elastomer actuators (DEAs). Systematic scanning is difficult due to time-consuming per-sample testing and the high-dimensional parameter space affecting performance. To address this, we propose an optimization pipeline enabled by a novel testing robot capable of scanning DEA lifetime. The robot integrates electro-mechanical property measurement, programmable voltage input, and multi-channel testing capacity. Using it, we scanned the lifetime of Elastosil-based linear actuators across parameters including input voltage magnitude, frequency, electrode material concentration, and electrical connection filler. The optimal parameter combinations improved operational lifetime under boundary operating conditions by up to 100% and were subsequently scaled up to achieve higher force and displacement output. The final product demonstrated resilience on a modular, scalable quadruped walking robot with payload carrying capacity (>100% of its untethered body weight, and >700% of combined actuator weight). This work is the first to introduce a self-driving lab approach into robotic actuator design.
>
---
#### [new 022] GeCo-SRT: Geometry-aware Continual Adaptation for Robotic Cross-Task Sim-to-Real Transfer
- **分类: cs.RO**

- **简介: 该论文属于机器人跨任务模拟到现实的迁移任务，旨在解决传统方法因孤立处理导致效率低、成本高的问题。提出GeCo-SRT方法，通过几何感知的知识积累实现高效适应。**

- **链接: [https://arxiv.org/pdf/2602.20871v1](https://arxiv.org/pdf/2602.20871v1)**

> **作者:** Wenbo Yu; Wenke Xia; Weitao Zhang; Di Hu
>
> **备注:** Accepted By CVPR 2026
>
> **摘要:** Bridging the sim-to-real gap is important for applying low-cost simulation data to real-world robotic systems. However, previous methods are severely limited by treating each transfer as an isolated endeavor, demanding repeated, costly tuning and wasting prior transfer experience.To move beyond isolated sim-to-real, we build a continual cross-task sim-to-real transfer paradigm centered on knowledge accumulation across iterative transfers, thereby enabling effective and efficient adaptation to novel tasks. Thus, we propose GeCo-SRT, a geometry-aware continual adaptation method. It utilizes domain-invariant and task-invariant knowledge from local geometric features as a transferable foundation to accelerate adaptation during subsequent sim-to-real transfers. This method starts with a geometry-aware mixture-of-experts module, which dynamically activates experts to specialize in distinct geometric knowledge to bridge observation sim-to-real gap. Further, the geometry-expert-guided prioritized experience replay module preferentially samples from underutilized experts, refreshing specialized knowledge to combat forgetting and maintain robust cross-task performance. Leveraging knowledge accumulated during iterative transfer, GeCo-SRT method not only achieves 52% average performance improvement over the baseline, but also demonstrates significant data efficiency for new task adaptation with only 1/6 data.We hope this work inspires approaches for efficient, low-cost cross-task sim-to-real transfer.
>
---
#### [new 023] Task-oriented grasping for dexterous robots using postural synergies and reinforcement learning
- **分类: cs.RO**

- **简介: 该论文研究人形机器人任务导向抓取问题，旨在提升抓取动作与任务目标和人类社会规范的一致性。通过结合后抓意图与强化学习，实现多物体抓取及任务特定操作。**

- **链接: [https://arxiv.org/pdf/2602.20915v1](https://arxiv.org/pdf/2602.20915v1)**

> **作者:** Dimitrios Dimou; José Santos-Victor; Plinio Moreno
>
> **摘要:** In this paper, we address the problem of task-oriented grasping for humanoid robots, emphasizing the need to align with human social norms and task-specific objectives. Existing methods, employ a variety of open-loop and closed-loop approaches but lack an end-to-end solution that can grasp several objects while taking into account the downstream task's constraints. Our proposed approach employs reinforcement learning to enhance task-oriented grasping, prioritizing the post-grasp intention of the agent. We extract human grasp preferences from the ContactPose dataset, and train a hand synergy model based on the Variational Autoencoder (VAE) to imitate the participant's grasping actions. Based on this data, we train an agent able to grasp multiple objects while taking into account distinct post-grasp intentions that are task-specific. By combining data-driven insights from human grasping behavior with learning by exploration provided by reinforcement learning, we can develop humanoid robots capable of context-aware manipulation actions, facilitating collaboration in human-centered environments.
>
---
#### [new 024] ParkDiffusion++: Ego Intention Conditioned Joint Multi-Agent Trajectory Prediction for Automated Parking using Diffusion Models
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶中的多智能体轨迹预测任务，解决自动化停车中预测驾驶员意图及周围车辆响应的问题。提出ParkDiffusion++模型，联合学习意图预测与轨迹生成，提升预测准确性和安全性。**

- **链接: [https://arxiv.org/pdf/2602.20923v1](https://arxiv.org/pdf/2602.20923v1)**

> **作者:** Jiarong Wei; Anna Rehr; Christian Feist; Abhinav Valada
>
> **备注:** ICRA 2026 Camera Ready Version
>
> **摘要:** Automated parking is a challenging operational domain for advanced driver assistance systems, requiring robust scene understanding and interaction reasoning. The key challenge is twofold: (i) predict multiple plausible ego intentions according to context and (ii) for each intention, predict the joint responses of surrounding agents, enabling effective what-if decision-making. However, existing methods often fall short, typically treating these interdependent problems in isolation. We propose ParkDiffusion++, which jointly learns a multi-modal ego intention predictor and an ego-conditioned multi-agent joint trajectory predictor for automated parking. Our approach makes several key contributions. First, we introduce an ego intention tokenizer that predicts a small set of discrete endpoint intentions from agent histories and vectorized map polylines. Second, we perform ego-intention-conditioned joint prediction, yielding socially consistent predictions of the surrounding agents for each possible ego intention. Third, we employ a lightweight safety-guided denoiser with different constraints to refine joint scenes during training, thus improving accuracy and safety. Fourth, we propose counterfactual knowledge distillation, where an EMA teacher refined by a frozen safety-guided denoiser provides pseudo-targets that capture how agents react to alternative ego intentions. Extensive evaluations demonstrate that ParkDiffusion++ achieves state-of-the-art performance on the Dragon Lake Parking (DLP) dataset and the Intersections Drone (inD) dataset. Importantly, qualitative what-if visualizations show that other agents react appropriately to different ego intentions.
>
---
#### [new 025] Robot Local Planner: A Periodic Sampling-Based Motion Planner with Minimal Waypoints for Home Environments
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，旨在解决家庭环境中快速安全操作的问题。提出一种基于周期采样的局部规划方法，减少路径点，提升效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.20645v1](https://arxiv.org/pdf/2602.20645v1)**

> **作者:** Keisuke Takeshita; Takahiro Yamazaki; Tomohiro Ono; Takashi Yamamoto
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2025. Project Page: https://toyotafrc.github.io/RobotLocalPlanner-Proj/
>
> **摘要:** The objective of this study is to enable fast and safe manipulation tasks in home environments. Specifically, we aim to develop a system that can recognize its surroundings and identify target objects while in motion, enabling it to plan and execute actions accordingly. We propose a periodic sampling-based whole-body trajectory planning method, called the "Robot Local Planner (RLP)." This method leverages unique features of home environments to enhance computational efficiency, motion optimality, and robustness against recognition and control errors, all while ensuring safety. The RLP minimizes computation time by planning with minimal waypoints and generating safe trajectories. Furthermore, overall motion optimality is improved by periodically executing trajectory planning to select more optimal motions. This approach incorporates inverse kinematics that are robust to base position errors, further enhancing robustness. Evaluation experiments demonstrated that the RLP outperformed existing methods in terms of motion planning time, motion duration, and robustness, confirming its effectiveness in home environments. Moreover, application experiments using a tidy-up task achieved high success rates and short operation times, thereby underscoring its practical feasibility.
>
---
#### [new 026] Generalizing from References using a Multi-Task Reference and Goal-Driven RL Framework
- **分类: cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决参考轨迹泛化与运动质量的平衡问题。通过多任务框架，联合训练模仿与目标驱动任务，提升机器人运动的适应性与自然性。**

- **链接: [https://arxiv.org/pdf/2602.20375v1](https://arxiv.org/pdf/2602.20375v1)**

> **作者:** Jiashun Wang; M. Eva Mungai; He Li; Jean Pierre Sleiman; Jessica Hodgins; Farbod Farshidian
>
> **摘要:** Learning agile humanoid behaviors from human motion offers a powerful route to natural, coordinated control, but existing approaches face a persistent trade-off: reference-tracking policies are often brittle outside the demonstration dataset, while purely task-driven Reinforcement Learning (RL) can achieve adaptability at the cost of motion quality. We introduce a unified multi-task RL framework that bridges this gap by treating reference motion as a prior for behavioral shaping rather than a deployment-time constraint. A single goal-conditioned policy is trained jointly on two tasks that share the same observation and action spaces, but differ in their initialization schemes, command spaces, and reward structures: (i) a reference-guided imitation task in which reference trajectories define dense imitation rewards but are not provided as policy inputs, and (ii) a goal-conditioned generalization task in which goals are sampled independently of any reference and where rewards reflect only task success. By co-optimizing these objectives within a shared formulation, the policy acquires structured, human-like motor skills from dense reference supervision while learning to adapt these skills to novel goals and initial conditions. This is achieved without adversarial objectives, explicit trajectory tracking, phase variables, or reference-dependent inference. We evaluate the method on a challenging box-based parkour playground that demands diverse athletic behaviors (e.g., jumping and climbing), and show that the learned controller transfers beyond the reference distribution while preserving motion naturalness. Finally, we demonstrate long-horizon behavior generation by composing multiple learned skills, illustrating the flexibility of the learned polices in complex scenarios.
>
---
#### [new 027] BFA++: Hierarchical Best-Feature-Aware Token Prune for Multi-View Vision Language Action Model
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出BFA++，解决多视角视觉语言动作模型中的实时性问题，通过动态令牌剪枝提升计算效率和操作成功率。**

- **链接: [https://arxiv.org/pdf/2602.20566v1](https://arxiv.org/pdf/2602.20566v1)**

> **作者:** Haosheng Li; Weixin Mao; Zihan Lan; Hongwei Xiong; Hongan Wang; Chenyang Si; Ziwei Liu; Xiaoming Deng; Hua Chen
>
> **备注:** 9 pages, 10 figures
>
> **摘要:** Vision-Language-Action (VLA) models have achieved significant breakthroughs by leveraging Large Vision Language Models (VLMs) to jointly interpret instructions and visual inputs. However, the substantial increase in visual tokens, particularly from multi-view inputs, poses serious challenges to real-time robotic manipulation. Existing acceleration techniques for VLMs, such as token pruning, often result in degraded performance when directly applied to VLA models, as they overlook the relationships between different views and fail to account for the dynamic and task-specific characteristics of robotic operation. To address this, we propose BFA++, a dynamic token pruning framework designed specifically for VLA models. BFA++ introduces a hierarchical pruning strategy guided by two-level importance predictors: an intra-view predictor highlights task-relevant regions within each image to suppress spatial noise, while an inter-view predictor identifies critical camera views throughout different manipulation phases to reduce cross-view redundancy. This design enables efficient token selection while preserving essential visual cues, resulting in improved computational efficiency and higher manipulation success rates. Evaluations on the RoboTwin benchmark and real-world robotic tasks demonstrate that BFA++ consistently outperforms existing methods. BFA++ improves the success rate by about 10% on both the π0 and RDT models, achieving speedup of 1.8X and 1.5X, respectively. Our results highlight that context-sensitive and task-aware token pruning serves as a more effective strategy than full visual processing, enabling faster inference and improved manipulation accuracy in real-world robotic systems.
>
---
#### [new 028] An Approach to Combining Video and Speech with Large Language Models in Human-Robot Interaction
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人机交互任务，旨在提升机器人对人类意图的理解与执行能力。通过融合视觉、语音和模糊逻辑，实现精准控制机械臂，解决命令解释不准确的问题。**

- **链接: [https://arxiv.org/pdf/2602.20219v1](https://arxiv.org/pdf/2602.20219v1)**

> **作者:** Guanting Shen; Zi Tian
>
> **备注:** Preprint currently under revision
>
> **摘要:** Interpreting human intent accurately is a central challenge in human-robot interaction (HRI) and a key requirement for achieving more natural and intuitive collaboration between humans and machines. This work presents a novel multimodal HRI framework that combines advanced vision-language models, speech processing, and fuzzy logic to enable precise and adaptive control of a Dobot Magician robotic arm. The proposed system integrates Florence-2 for object detection, Llama 3.1 for natural language understanding, and Whisper for speech recognition, providing users with a seamless and intuitive interface for object manipulation through spoken commands. By jointly addressing scene perception and action planning, the approach enhances the reliability of command interpretation and execution. Experimental evaluations conducted on consumer-grade hardware demonstrate a command execution accuracy of 75\%, highlighting both the robustness and adaptability of the system. Beyond its current performance, the proposed architecture serves as a flexible and extensible foundation for future HRI research, offering a practical pathway toward more sophisticated and natural human-robot collaboration through tightly coupled speech and vision-language processing.
>
---
#### [new 029] Surface-based Manipulation Using Tunable Compliant Porous-Elastic Soft Sensing
- **分类: cs.RO**

- **简介: 该论文属于软体机器人任务，旨在解决物体柔性抓取中缺乏适应性与触觉反馈的问题。提出COPESS系统，通过可调结构实现机械柔性和传感性能的协同优化。**

- **链接: [https://arxiv.org/pdf/2602.21028v1](https://arxiv.org/pdf/2602.21028v1)**

> **作者:** Gayatri Indukumar; Muhammad Awais; Diana Cafiso; Matteo Lo Preti; Lucia Beccai
>
> **备注:** 6 pages, 6 figures, 1 table, to be published in RoboSoft 2026 proceedings
>
> **摘要:** There is a growing need for soft robotic platforms that perform gentle, precise handling of a wide variety of objects. Existing surface-based manipulation systems, however, lack the compliance and tactile feedback needed for delicate handling. This work introduces the COmpliant Porous-Elastic Soft Sensing (COPESS) integrated with inductive sensors for adaptive object manipulation and localised sensing. The design features a tunable lattice layer that simultaneously modulates mechanical compliance and sensing performance. By adjusting lattice geometry, both stiffness and sensor response can be tailored to handle objects with varying mechanical properties. Experiments demonstrate that by easily adjusting one parameter, the lattice density, from 7 % to 20 %, it is possible to significantly alter the sensitivity and operational force range (about -23x and 9x, respectively). This approach establishes a blueprint for creating adaptive, sensorized surfaces where mechanical and sensory properties are co-optimized, enabling passive, yet programmable, delicate manipulation.
>
---
#### [new 030] EKF-Based Depth Camera and Deep Learning Fusion for UAV-Person Distance Estimation and Following in SAR Operations
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于无人机目标跟踪任务，解决SAR中无人机与人员安全距离估计问题。融合深度相机与单目相机数据，利用YOLO-pose和EKF算法实现精准距离估计。**

- **链接: [https://arxiv.org/pdf/2602.20958v1](https://arxiv.org/pdf/2602.20958v1)**

> **作者:** Luka Šiktar; Branimir Ćaran; Bojan Šekoranja; Marko Švaco
>
> **摘要:** Search and rescue (SAR) operations require rapid responses to save lives or property. Unmanned Aerial Vehicles (UAVs) equipped with vision-based systems support these missions through prior terrain investigation or real-time assistance during the mission itself. Vision-based UAV frameworks aid human search tasks by detecting and recognizing specific individuals, then tracking and following them while maintaining a safe distance. A key safety requirement for UAV following is the accurate estimation of the distance between camera and target object under real-world conditions, achieved by fusing multiple image modalities. UAVs with deep learning-based vision systems offer a new approach to the planning and execution of SAR operations. As part of the system for automatic people detection and face recognition using deep learning, in this paper we present the fusion of depth camera measurements and monocular camera-to-body distance estimation for robust tracking and following. Deep learning-based filtering of depth camera data and estimation of camera-to-body distance from a monocular camera are achieved with YOLO-pose, enabling real-time fusion of depth information using the Extended Kalman Filter (EKF) algorithm. The proposed subsystem, designed for use in drones, estimates and measures the distance between the depth camera and the human body keypoints, to maintain the safe distance between the drone and the human target. Our system provides an accurate estimated distance, which has been validated against motion capture ground truth data. The system has been tested in real time indoors, where it reduces the average errors, root mean square error (RMSE) and standard deviations of distance estimation up to 15,3\% in three tested scenarios.
>
---
#### [new 031] ActionReasoning: Robot Action Reasoning in 3D Space with LLM for Robotic Brick Stacking
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决传统系统泛化能力差的问题。通过引入LLM驱动的ActionReasoning框架，实现物理一致的动作推理，提升机器人砖块堆叠的稳定性与通用性。**

- **链接: [https://arxiv.org/pdf/2602.21161v1](https://arxiv.org/pdf/2602.21161v1)**

> **作者:** Guangming Wang; Qizhen Ying; Yixiong Jing; Olaf Wysocki; Brian Sheil
>
> **备注:** 8 pages, 5 figures, accepted by the 2026 IEEE International Conference on Robotics and Automation
>
> **摘要:** Classical robotic systems typically rely on custom planners designed for constrained environments. While effective in restricted settings, these systems lack generalization capabilities, limiting the scalability of embodied AI and general-purpose robots. Recent data-driven Vision-Language-Action (VLA) approaches aim to learn policies from large-scale simulation and real-world data. However, the continuous action space of the physical world significantly exceeds the representational capacity of linguistic tokens, making it unclear if scaling data alone can yield general robotic intelligence. To address this gap, we propose ActionReasoning, an LLM-driven framework that performs explicit action reasoning to produce physics-consistent, prior-guided decisions for robotic manipulation. ActionReasoning leverages the physical priors and real-world knowledge already encoded in Large Language Models (LLMs) and structures them within a multi-agent architecture. We instantiate this framework on a tractable case study of brick stacking, where the environment states are assumed to be already accurately measured. The environmental states are then serialized and passed to a multi-agent LLM framework that generates physics-aware action plans. The experiments demonstrate that the proposed multi-agent LLM framework enables stable brick placement while shifting effort from low-level domain-specific coding to high-level tool invocation and prompting, highlighting its potential for broader generalization. This work introduces a promising approach to bridging perception and execution in robotic manipulation by integrating physical reasoning with LLMs.
>
---
#### [new 032] LST-SLAM: A Stereo Thermal SLAM System for Kilometer-Scale Dynamic Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于SLAM任务，旨在解决动态大场景下热红外相机定位与建图的难题。通过特征学习、运动跟踪和优化方法提升系统鲁棒性与精度。**

- **链接: [https://arxiv.org/pdf/2602.20925v1](https://arxiv.org/pdf/2602.20925v1)**

> **作者:** Zeyu Jiang; Kuan Xu; Changhao Chen
>
> **备注:** ICRA 2026
>
> **摘要:** Thermal cameras offer strong potential for robot perception under challenging illumination and weather conditions. However, thermal Simultaneous Localization and Mapping (SLAM) remains difficult due to unreliable feature extraction, unstable motion tracking, and inconsistent global pose and map construction, particularly in dynamic large-scale outdoor environments. To address these challenges, we propose LST-SLAM, a novel large-scale stereo thermal SLAM system that achieves robust performance in complex, dynamic scenes. Our approach combines self-supervised thermal feature learning, stereo dual-level motion tracking, and geometric pose optimization. We also introduce a semantic-geometric hybrid constraint that suppresses potentially dynamic features lacking strong inter-frame geometric consistency. Furthermore, we develop an online incremental bag-of-words model for loop closure detection, coupled with global pose optimization to mitigate accumulated drift. Extensive experiments on kilometer-scale dynamic thermal datasets show that LST-SLAM significantly outperforms recent representative SLAM systems, including AirSLAM and DROID-SLAM, in both robustness and accuracy.
>
---
#### [new 033] Cooperative-Competitive Team Play of Real-World Craft Robots
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多智能体强化学习任务，旨在解决集体机器人高效训练及模拟到现实的迁移问题。作者构建了完整系统并提出OODSI技术，提升真实场景下的性能。**

- **链接: [https://arxiv.org/pdf/2602.21119v1](https://arxiv.org/pdf/2602.21119v1)**

> **作者:** Rui Zhao; Xihui Li; Yizheng Zhang; Yuzhen Liu; Zhong Zhang; Yufeng Zhang; Cheng Zhou; Zhengyou Zhang; Lei Han
>
> **备注:** Accepted by 2026 IEEE International Conference on Robotics and Automation (ICRA 2026), Vienna, Austria
>
> **摘要:** Multi-agent deep Reinforcement Learning (RL) has made significant progress in developing intelligent game-playing agents in recent years. However, the efficient training of collective robots using multi-agent RL and the transfer of learned policies to real-world applications remain open research questions. In this work, we first develop a comprehensive robotic system, including simulation, distributed learning framework, and physical robot components. We then propose and evaluate reinforcement learning techniques designed for efficient training of cooperative and competitive policies on this platform. To address the challenges of multi-agent sim-to-real transfer, we introduce Out of Distribution State Initialization (OODSI) to mitigate the impact of the sim-to-real gap. In the experiments, OODSI improves the Sim2Real performance by 20%. We demonstrate the effectiveness of our approach through experiments with a multi-robot car competitive game and a cooperative task in real-world settings.
>
---
#### [new 034] Learning Physical Principles from Interaction: Self-Evolving Planning via Test-Time Memory
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出PhysMem框架，解决机器人在物体操作中学习物理原理的问题。通过测试时记忆与验证，提升操作成功率。**

- **链接: [https://arxiv.org/pdf/2602.20323v1](https://arxiv.org/pdf/2602.20323v1)**

> **作者:** Haoyang Li; Yang You; Hao Su; Leonidas Guibas
>
> **摘要:** Reliable object manipulation requires understanding physical properties that vary across objects and environments. Vision-language model (VLM) planners can reason about friction and stability in general terms; however, they often cannot predict how a specific ball will roll on a particular surface or which stone will provide a stable foundation without direct experience. We present PhysMem, a memory framework that enables VLM robot planners to learn physical principles from interaction at test time, without updating model parameters. The system records experiences, generates candidate hypotheses, and verifies them through targeted interaction before promoting validated knowledge to guide future decisions. A central design choice is verification before application: the system tests hypotheses against new observations rather than applying retrieved experience directly, reducing rigid reliance on prior experience when physical conditions change. We evaluate PhysMem on three real-world manipulation tasks and simulation benchmarks across four VLM backbones. On a controlled brick insertion task, principled abstraction achieves 76% success compared to 23% for direct experience retrieval, and real-world experiments show consistent improvement over 30-minute deployment sessions.
>
---
#### [new 035] Vision-Based Reasoning with Topology-Encoded Graphs for Anatomical Path Disambiguation in Robot-Assisted Endovascular Navigation
- **分类: cs.RO**

- **简介: 该论文属于医学图像分析任务，旨在解决机器人辅助血管内手术中的路径歧义问题。通过分割血管并构建图结构，利用图注意力网络识别正确路径，提升导航准确性。**

- **链接: [https://arxiv.org/pdf/2602.20215v1](https://arxiv.org/pdf/2602.20215v1)**

> **作者:** Jiyuan Zhao; Zhengyu Shi; Wentong Tian; Tianliang Yao; Dong Liu; Tao Liu; Yizhe Wu; Peng Qi
>
> **备注:** This paper has been accepted by IEEE ICRA 2026. 8 pages, 3 figures, 3 tables
>
> **摘要:** Robotic-assisted percutaneous coronary intervention (PCI) is constrained by the inherent limitations of 2D Digital Subtraction Angiography (DSA). Unlike physicians, who can directly manipulate guidewires and integrate tactile feedback with their prior anatomical knowledge, teleoperated robotic systems must rely solely on 2D projections. This mode of operation, simultaneously lacking spatial context and tactile sensation, may give rise to projection-induced ambiguities at vascular bifurcations. To address this challenge, we propose a two-stage framework (SCAR-UNet-GAT) for real-time robotic path planning. In the first stage, SCAR-UNet, a spatial-coordinate-attention-regularized U-Net, is employed for accurate coronary vessel segmentation. The integration of multi-level attention mechanisms enhances the delineation of thin, tortuous vessels and improves robustness against imaging noise. From the resulting binary masks, vessel centerlines and bifurcation points are extracted, and geometric descriptors (e.g., branch diameter, intersection angles) are fused with local DSA patches to construct node features. In the second stage, a Graph Attention Network (GAT) reasons over the vessel graph to identify anatomically consistent and clinically feasible trajectories, effectively distinguishing true bifurcations from projection-induced false crossings. On a clinical DSA dataset, SCAR-UNet achieved a Dice coefficient of 93.1%. For path disambiguation, the proposed GAT-based method attained a success rate of 95.0% and a target-arrival success rate of 90.0%, substantially outperforming conventional shortest-path planning (60.0% and 55.0%) and heuristic-based planning (75.0% and 70.0%). Validation on a robotic platform further confirmed the practical feasibility and robustness of the proposed framework.
>
---
#### [new 036] Real-time Motion Segmentation with Event-based Normal Flow
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于运动分割任务，旨在解决事件相机在实时场景中运动分割效率低的问题。通过引入法线流作为中间表示，提出一种基于法线流的运动分割框架，提升处理速度与准确性。**

- **链接: [https://arxiv.org/pdf/2602.20790v1](https://arxiv.org/pdf/2602.20790v1)**

> **作者:** Sheng Zhong; Zhongyang Ren; Xiya Zhu; Dehao Yuan; Cornelia Fermuller; Yi Zhou
>
> **摘要:** Event-based cameras are bio-inspired sensors with pixels that independently and asynchronously respond to brightness changes at microsecond resolution, offering the potential to handle visual tasks in challenging scenarios. However, due to the sparse information content in individual events, directly processing the raw event data to solve vision tasks is highly inefficient, which severely limits the applicability of state-of-the-art methods in real-time tasks, such as motion segmentation, a fundamental task for dynamic scene understanding. Incorporating normal flow as an intermediate representation to compress motion information from event clusters within a localized region provides a more effective solution. In this work, we propose a normal flow-based motion segmentation framework for event-based vision. Leveraging the dense normal flow directly learned from event neighborhoods as input, we formulate the motion segmentation task as an energy minimization problem solved via graph cuts, and optimize it iteratively with normal flow clustering and motion model fitting. By using a normal flow-based motion model initialization and fitting method, the proposed system is able to efficiently estimate the motion models of independently moving objects with only a limited number of candidate models, which significantly reduces the computational complexity and ensures real-time performance, achieving nearly a 800x speedup in comparison to the open-source state-of-the-art method. Extensive evaluations on multiple public datasets fully demonstrate the accuracy and efficiency of our framework.
>
---
#### [new 037] Localized Dynamics-Aware Domain Adaption for Off-Dynamics Offline Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于离线强化学习任务，解决目标域与源域动态不匹配问题。提出LoDADA方法，通过局部动态差异筛选数据，提升策略性能。**

- **链接: [https://arxiv.org/pdf/2602.21072v1](https://arxiv.org/pdf/2602.21072v1)**

> **作者:** Zhangjie Xia; Yu Yang; Pan Xu
>
> **备注:** 33 pages, 9 figures, 11 tables
>
> **摘要:** Off-dynamics offline reinforcement learning (RL) aims to learn a policy for a target domain using limited target data and abundant source data collected under different transition dynamics. Existing methods typically address dynamics mismatch either globally over the state space or via pointwise data filtering; these approaches can miss localized cross-domain similarities or incur high computational cost. We propose Localized Dynamics-Aware Domain Adaptation (LoDADA), which exploits localized dynamics mismatch to better reuse source data. LoDADA clusters transitions from source and target datasets and estimates cluster-level dynamics discrepancy via domain discrimination. Source transitions from clusters with small discrepancy are retained, while those from clusters with large discrepancy are filtered out. This yields a fine-grained and scalable data selection strategy that avoids overly coarse global assumptions and expensive per-sample filtering. We provide theoretical insights and extensive experiments across environments with diverse global and local dynamics shifts. Results show that LoDADA consistently outperforms state-of-the-art off-dynamics offline RL methods by better leveraging localized distribution mismatch.
>
---
#### [new 038] Event-Aided Sharp Radiance Field Reconstruction for Fast-Flying Drones
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D重建任务，解决高速飞行无人机的运动模糊和位姿漂移问题。通过融合事件流和模糊图像，优化NeRF并提升位姿估计，实现高精度辐射场重建。**

- **链接: [https://arxiv.org/pdf/2602.21101v1](https://arxiv.org/pdf/2602.21101v1)**

> **作者:** Rong Zou; Marco Cannici; Davide Scaramuzza
>
> **摘要:** Fast-flying aerial robots promise rapid inspection under limited battery constraints, with direct applications in infrastructure inspection, terrain exploration, and search and rescue. However, high speeds lead to severe motion blur in images and induce significant drift and noise in pose estimates, making dense 3D reconstruction with Neural Radiance Fields (NeRFs) particularly challenging due to their high sensitivity to such degradations. In this work, we present a unified framework that leverages asynchronous event streams alongside motion-blurred frames to reconstruct high-fidelity radiance fields from agile drone flights. By embedding event-image fusion into NeRF optimization and jointly refining event-based visual-inertial odometry priors using both event and frame modalities, our method recovers sharp radiance fields and accurate camera trajectories without ground-truth supervision. We validate our approach on both synthetic data and real-world sequences captured by a fast-flying drone. Despite highly dynamic drone flights, where RGB frames are severely degraded by motion blur and pose priors become unreliable, our method reconstructs high-fidelity radiance fields and preserves fine scene details, delivering a performance gain of over 50% on real-world data compared to state-of-the-art methods.
>
---
#### [new 039] UAMTERS: Uncertainty-Aware Mutation Analysis for DL-enabled Robotic Software
- **分类: cs.SE; cs.RO**

- **简介: 该论文属于软件测试任务，旨在解决DL-enabled机器人软件在不确定性环境下的测试有效性问题。提出UAMTERS框架，通过引入不确定性突变算子和评分指标，提升测试套件检测不确定故障的能力。**

- **链接: [https://arxiv.org/pdf/2602.20334v1](https://arxiv.org/pdf/2602.20334v1)**

> **作者:** Chengjie Lu; Jiahui Wu; Shaukat Ali; Malaika Din Hashmi; Sebastian Mathias Thomle Mason; Francois Picard; Mikkel Labori Olsen; Thomas Peyrucain
>
> **备注:** 23 pages, 6 figures, 7 tables
>
> **摘要:** Self-adaptive robots adjust their behaviors in response to unpredictable environmental changes. These robots often incorporate deep learning (DL) components into their software to support functionality such as perception, decision-making, and control, enhancing autonomy and self-adaptability. However, the inherent uncertainty of DL-enabled software makes it challenging to ensure its dependability in dynamic environments. Consequently, test generation techniques have been developed to test robot software, and classical mutation analysis injects faults into the software to assess the test suite's effectiveness in detecting the resulting failures. However, there is a lack of mutation analysis techniques to assess the effectiveness under the uncertainty inherent to DL-enabled software. To this end, we propose UAMTERS, an uncertainty-aware mutation analysis framework that introduces uncertainty-aware mutation operators to explicitly inject stochastic uncertainty into DL-enabled robotic software, simulating uncertainty in its behavior. We further propose mutation score metrics to quantify a test suite's ability to detect failures under varying levels of uncertainty. We evaluate UAMTERS across three robotic case studies, demonstrating that UAMTERS more effectively distinguishes test suite quality and captures uncertainty-induced failures in DL-enabled software.
>
---
#### [new 040] Learning from Trials and Errors: Reflective Test-Time Planning for Embodied LLMs
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文研究机器人学习任务，旨在解决Embodied LLMs无法反思错误的问题。通过引入两种反射机制，提升机器人在执行任务中的自我修正能力。**

- **链接: [https://arxiv.org/pdf/2602.21198v1](https://arxiv.org/pdf/2602.21198v1)**

> **作者:** Yining Hong; Huang Huang; Manling Li; Li Fei-Fei; Jiajun Wu; Yejin Choi
>
> **摘要:** Embodied LLMs endow robots with high-level task reasoning, but they cannot reflect on what went wrong or why, turning deployment into a sequence of independent trials where mistakes repeat rather than accumulate into experience. Drawing upon human reflective practitioners, we introduce Reflective Test-Time Planning, which integrates two modes of reflection: \textit{reflection-in-action}, where the agent uses test-time scaling to generate and score multiple candidate actions using internal reflections before execution; and \textit{reflection-on-action}, which uses test-time training to update both its internal reflection model and its action policy based on external reflections after execution. We also include retrospective reflection, allowing the agent to re-evaluate earlier decisions and perform model updates with hindsight for proper long-horizon credit assignment. Experiments on our newly-designed Long-Horizon Household benchmark and MuJoCo Cupboard Fitting benchmark show significant gains over baseline models, with ablative studies validating the complementary roles of reflection-in-action and reflection-on-action. Qualitative analyses, including real-robot trials, highlight behavioral correction through reflection.
>
---
#### [new 041] RU4D-SLAM: Reweighting Uncertainty in Gaussian Splatting SLAM for 4D Scene Reconstruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于4D场景重建任务，旨在解决动态环境中SLAM的不确定性问题。通过引入时间因素和动态感知机制，提升跟踪与重建精度。**

- **链接: [https://arxiv.org/pdf/2602.20807v1](https://arxiv.org/pdf/2602.20807v1)**

> **作者:** Yangfan Zhao; Hanwei Zhang; Ke Huang; Qiufeng Wang; Zhenzhou Shao; Dengyu Wu
>
> **摘要:** Combining 3D Gaussian splatting with Simultaneous Localization and Mapping (SLAM) has gained popularity as it enables continuous 3D environment reconstruction during motion. However, existing methods struggle in dynamic environments, particularly moving objects complicate 3D reconstruction and, in turn, hinder reliable tracking. The emergence of 4D reconstruction, especially 4D Gaussian splatting, offers a promising direction for addressing these challenges, yet its potential for 4D-aware SLAM remains largely underexplored. Along this direction, we propose a robust and efficient framework, namely Reweighting Uncertainty in Gaussian Splatting SLAM (RU4D-SLAM) for 4D scene reconstruction, that introduces temporal factors into spatial 3D representation while incorporating uncertainty-aware perception of scene changes, blurred image synthesis, and dynamic scene reconstruction. We enhance dynamic scene representation by integrating motion blur rendering, and improve uncertainty-aware tracking by extending per-pixel uncertainty modeling, which is originally designed for static scenarios, to handle blurred images. Furthermore, we propose a semantic-guided reweighting mechanism for per-pixel uncertainty estimation in dynamic scenes, and introduce a learnable opacity weight to support adaptive 4D mapping. Extensive experiments on standard benchmarks demonstrate that our method substantially outperforms state-of-the-art approaches in both trajectory accuracy and 4D scene reconstruction, particularly in dynamic environments with moving objects and low-quality inputs. Code available: https://ru4d-slam.github.io
>
---
#### [new 042] Object-Scene-Camera Decomposition and Recomposition for Data-Efficient Monocular 3D Object Detection
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于单目3D目标检测任务，旨在解决训练数据不足和过拟合问题。通过分解并重新组合物体、场景和相机姿态，提升数据效率。**

- **链接: [https://arxiv.org/pdf/2602.20627v1](https://arxiv.org/pdf/2602.20627v1)**

> **作者:** Zhaonian Kuang; Rui Ding; Meng Yang; Xinhu Zheng; Gang Hua
>
> **备注:** IJCV
>
> **摘要:** Monocular 3D object detection (M3OD) is intrinsically ill-posed, hence training a high-performance deep learning based M3OD model requires a humongous amount of labeled data with complicated visual variation from diverse scenes, variety of objects and camera poses.However, we observe that, due to strong human bias, the three independent entities, i.e., object, scene, and camera pose, are always tightly entangled when an image is captured to construct training data. More specifically, specific 3D objects are always captured in particular scenes with fixed camera poses, and hence lacks necessary diversity. Such tight entanglement induces the challenging issues of insufficient utilization and overfitting to uniform training data. To mitigate this, we propose an online object-scene-camera decomposition and recomposition data manipulation scheme to more efficiently exploit the training data. We first fully decompose training images into textured 3D object point models and background scenes in an efficient computation and storage manner. We then continuously recompose new training images in each epoch by inserting the 3D objects into the freespace of the background scenes, and rendering them with perturbed camera poses from textured 3D point representation. In this way, the refreshed training data in all epochs can cover the full spectrum of independent object, scene, and camera pose combinations. This scheme can serve as a plug-and-play component to boost M3OD models, working flexibly with both fully and sparsely supervised settings. In the sparsely-supervised setting, objects closest to the ego-camera for all instances are sparsely annotated. We then can flexibly increase the annotated objects to control annotation cost. For validation, our method is widely applied to five representative M3OD models and evaluated on both the KITTI and the more complicated Waymo datasets.
>
---
#### [new 043] Is Robot Labor Labor? Delivery Robots and the Politics of Work in Public Space
- **分类: cs.CY; cs.HC; cs.RO**

- **简介: 该论文属于社会技术研究任务，探讨配送机器人劳动的性质与社会影响。通过实地观察，分析机器人劳动如何依赖人类协作与政策支持，揭示其对劳动形态的重构及公共空间中的权力关系。**

- **链接: [https://arxiv.org/pdf/2602.20180v1](https://arxiv.org/pdf/2602.20180v1)**

> **作者:** EunJeong Cheon; Do Yeon Shin
>
> **摘要:** As sidewalk delivery robots become increasingly integrated into urban life, this paper begins with a critical provocation: Is robot labor labor? More than a rhetorical question, this inquiry invites closer attention to the social and political arrangements that robot labor entails. Drawing on ethnographic fieldwork across two smart-city districts in Seoul, we examine how delivery robot labor is collectively sustained. While robotic actions are often framed as autonomous and efficient, we show that each successful delivery is in fact a distributed sociotechnical achievement--reliant on human labor, regulatory coordination, and social accommodations. We argue that delivery robots do not replace labor but reconfigure it--rendering some forms more visible (robotic performance) while obscuring others (human and institutional support). Unlike industrial robots, delivery robots operate in shared public space, engage everyday passersby, and are embedded in policy and progress narratives. In these spaces, we identify "robot privilege"--humans routinely yielding to robots--and distinct perceptions between casual observers ("cute") and everyday coexisters ("admirable"). We contribute a conceptual reframing of robot labor as a collective assemblage, empirical insights into South Korea's smart-city automation, and a call for HRI to engage more deeply with labor and spatial politics to better theorize public-facing robots.
>
---
#### [new 044] Long-Term Multi-Session 3D Reconstruction Under Substantial Appearance Change
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D重建任务，解决长期监测中因外观变化导致的多会话模型对齐问题。通过联合SfM重建和特征匹配，实现跨时间的连贯3D模型重建。**

- **链接: [https://arxiv.org/pdf/2602.20584v1](https://arxiv.org/pdf/2602.20584v1)**

> **作者:** Beverley Gorry; Tobias Fischer; Michael Milford; Alejandro Fontan
>
> **摘要:** Long-term environmental monitoring requires the ability to reconstruct and align 3D models across repeated site visits separated by months or years. However, existing Structure-from-Motion (SfM) pipelines implicitly assume near-simultaneous image capture and limited appearance change, and therefore fail when applied to long-term monitoring scenarios such as coral reef surveys, where substantial visual and structural change is common. In this paper, we show that the primary limitation of current approaches lies in their reliance on post-hoc alignment of independently reconstructed sessions, which is insufficient under large temporal appearance change. We address this limitation by enforcing cross-session correspondences directly within a joint SfM reconstruction. Our approach combines complementary handcrafted and learned visual features to robustly establish correspondences across large temporal gaps, enabling the reconstruction of a single coherent 3D model from imagery captured years apart, where standard independent and joint SfM pipelines break down. We evaluate our method on long-term coral reef datasets exhibiting significant real-world change, and demonstrate consistent joint reconstruction across sessions in cases where existing methods fail to produce coherent reconstructions. To ensure scalability to large datasets, we further restrict expensive learned feature matching to a small set of likely cross-session image pairs identified via visual place recognition, which reduces computational cost and improves alignment robustness.
>
---
#### [new 045] Large-scale Photorealistic Outdoor 3D Scene Reconstruction from UAV Imagery Using Gaussian Splatting Techniques
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D场景重建任务，旨在解决无人机影像实时高保真重建问题。通过整合视频流、传感器融合与3DGS优化，实现低延迟、高效率的实时重建与可视化。**

- **链接: [https://arxiv.org/pdf/2602.20342v1](https://arxiv.org/pdf/2602.20342v1)**

> **作者:** Christos Maikos; Georgios Angelidis; Georgios Th. Papadopoulos
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** In this study, we present an end-to-end pipeline capable of converting drone-captured video streams into high-fidelity 3D reconstructions with minimal latency. Unmanned aerial vehicles (UAVs) are extensively used in aerial real-time perception applications. Moreover, recent advances in 3D Gaussian Splatting (3DGS) have demonstrated significant potential for real-time neural rendering. However, their integration into end-to-end UAV-based reconstruction and visualization systems remains underexplored. Our goal is to propose an efficient architecture that combines live video acquisition via RTMP streaming, synchronized sensor fusion, camera pose estimation, and 3DGS optimization, achieving continuous model updates and low-latency deployment within interactive visualization environments that supports immersive augmented and virtual reality (AR/VR) applications. Experimental results demonstrate that the proposed method achieves competitive visual fidelity, while delivering significantly higher rendering performance and substantially reduced end-to-end latency, compared to NeRF-based approaches. Reconstruction quality remains within 4-7\% of high-fidelity offline references, confirming the suitability of the proposed system for real-time, scalable augmented perception from aerial platforms.
>
---
## 更新

#### [replaced 001] MoMaGen: Generating Demonstrations under Soft and Hard Constraints for Multi-Step Bimanual Mobile Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文针对多步骤双臂移动操作任务，解决数据生成中可达性和可视性约束问题，提出MoMaGen框架生成多样化数据集，支持有效模仿学习。**

- **链接: [https://arxiv.org/pdf/2510.18316v3](https://arxiv.org/pdf/2510.18316v3)**

> **作者:** Chengshu Li; Mengdi Xu; Arpit Bahety; Hang Yin; Yunfan Jiang; Huang Huang; Josiah Wong; Sujay Garlanka; Cem Gokmen; Ruohan Zhang; Weiyu Liu; Jiajun Wu; Roberto Martín-Martín; Li Fei-Fei
>
> **备注:** Project website: momagen.github.io. The first four authors contribute equally. Accpeted to International Conference on Learning Representations (ICLR 2026)
>
> **摘要:** Imitation learning from large-scale, diverse human demonstrations has been shown to be effective for training robots, but collecting such data is costly and time-consuming. This challenge intensifies for multi-step bimanual mobile manipulation, where humans must teleoperate both the mobile base and two high-DoF arms. Prior X-Gen works have developed automated data generation frameworks for static (bimanual) manipulation tasks, augmenting a few human demos in simulation with novel scene configurations to synthesize large-scale datasets. However, prior works fall short for bimanual mobile manipulation tasks for two major reasons: 1) a mobile base introduces the problem of how to place the robot base to enable downstream manipulation (reachability) and 2) an active camera introduces the problem of how to position the camera to generate data for a visuomotor policy (visibility). To address these challenges, MoMaGen formulates data generation as a constrained optimization problem that satisfies hard constraints (e.g., reachability) while balancing soft constraints (e.g., visibility while navigation). This formulation generalizes across most existing automated data generation approaches and offers a principled foundation for developing future methods. We evaluate on four multi-step bimanual mobile manipulation tasks and find that MoMaGen enables the generation of much more diverse datasets than previous methods. As a result of the dataset diversity, we also show that the data generated by MoMaGen can be used to train successful imitation learning policies using a single source demo. Furthermore, the trained policy can be fine-tuned with a very small amount of real-world data (40 demos) to be succesfully deployed on real robotic hardware. More details are on our project page: momagen.github.io.
>
---
#### [replaced 002] PMG: Parameterized Motion Generator for Human-like Locomotion Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人形机器人运动控制任务，旨在解决传统方法对数据依赖大、适应性差的问题。提出PMG系统，通过参数化运动生成实现自然、高效的人类似运动控制。**

- **链接: [https://arxiv.org/pdf/2602.12656v2](https://arxiv.org/pdf/2602.12656v2)**

> **作者:** Chenxi Han; Yuheng Min; Zihao Huang; Ao Hong; Hang Liu; Yi Cheng; Houde Liu
>
> **备注:** Website: https://pmg-icra26.github.io/
>
> **摘要:** Recent advances in data-driven reinforcement learning and motion tracking have substantially improved humanoid locomotion, yet critical practical challenges remain. In particular, while low-level motion tracking and trajectory-following controllers are mature, whole-body reference-guided methods are difficult to adapt to higher-level command interfaces and diverse task contexts: they require large, high-quality datasets, are brittle across speed and pose regimes, and are sensitive to robot-specific calibration. To address these limitations, we propose the Parameterized Motion Generator (PMG), a real-time motion generator grounded in an analysis of human motion structure that synthesizes reference trajectories using only a compact set of parameterized motion data together with high-dimensional control commands. Combined with an imitation-learning pipeline and an optimization-based sim-to-real motor parameter identification module, we validate the complete approach on our humanoid prototype ZERITH Z1 and show that, within a single integrated system, PMG produces natural, human-like locomotion, responds precisely to high-dimensional control inputs-including VR-based teleoperation-and enables efficient, verifiable sim-to-real transfer. Together, these results establish a practical, experimentally validated pathway toward natural and deployable humanoid control. Website: https://pmg-icra26.github.io/
>
---
#### [replaced 003] Effective Reinforcement Learning Control using Conservative Soft Actor-Critic
- **分类: cs.RO**

- **简介: 该论文属于强化学习控制任务，旨在解决探索、稳定性和样本效率问题。提出CSAC算法，结合熵和相对熵正则化，提升稳定性与效率。**

- **链接: [https://arxiv.org/pdf/2505.03356v2](https://arxiv.org/pdf/2505.03356v2)**

> **作者:** Zhiwei Shang; Xinyi Yuan; Wenjun Huang; Yunduan Cui; Di Chen; Meixin Zhu
>
> **备注:** 8 pages, 9 figures
>
> **摘要:** Reinforcement Learning (RL) has shown great potential in complex control tasks, particularly when combined with deep neural networks within the Actor-Critic (AC) framework. However, in practical applications, balancing exploration, learning stability, and sample efficiency remains a significant challenge. Traditional methods such as Soft Actor-Critic (SAC) and Proximal Policy Optimization (PPO) address these issues by incorporating entropy or relative entropy regularization, but often face problems of instability and low sample efficiency. In this paper, we propose the Conservative Soft Actor-Critic (CSAC) algorithm, which seamlessly integrates entropy and relative entropy regularization within the AC framework. CSAC improves exploration through entropy regularization while avoiding overly aggressive policy updates with the use of relative entropy regularization. Evaluations on benchmark tasks and real-world robotic simulations demonstrate that CSAC offers significant improvements in stability and efficiency over existing methods. These findings suggest that CSAC provides strong robustness and application potential in control tasks under dynamic environments.
>
---
#### [replaced 004] Joint Task Assistance Planning via Nested Branch and Bound (Extended Version)
- **分类: cs.RO**

- **简介: 该论文属于机器人协作任务，解决双机器人路径规划问题，旨在最大化辅助时间。通过嵌套分支定界方法高效求解。**

- **链接: [https://arxiv.org/pdf/2602.13932v2](https://arxiv.org/pdf/2602.13932v2)**

> **作者:** Omer Daube; Oren Salzman
>
> **摘要:** We introduce and study the Joint Task Assistance Planning problem which generalizes prior work on optimizing assistance in robotic collaboration. In this setting, two robots operate over predefined roadmaps, each represented as a graph corresponding to its configuration space. One robot, the task robot, must execute a timed mission, while the other, the assistance robot, provides sensor-based support that depends on their spatial relationship. The objective is to compute a path for both robots that maximizes the total duration of assistance given. Solving this problem is challenging due to the combinatorial explosion of possible path combinations together with the temporal nature of the problem (time needs to be accounted for as well). To address this, we propose a nested branch-and-bound framework that efficiently explores the space of robot paths in a hierarchical manner. We empirically evaluate our algorithm and demonstrate a speedup of up to two orders of magnitude when compared to a baseline approach.
>
---
#### [replaced 005] Performance Asymmetry in Model-Based Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文研究强化学习中的性能不对称问题，针对MBRL在不同任务上表现差异大的问题，提出新模型JEDI以提升平衡性与效率。**

- **链接: [https://arxiv.org/pdf/2505.19698v3](https://arxiv.org/pdf/2505.19698v3)**

> **作者:** Jing Yu Lim; Rushi Shah; Zarif Ikram; Samson Yu; Haozhe Ma; Tze-Yun Leong; Dianbo Liu
>
> **备注:** Preprint
>
> **摘要:** Recently, Model-Based Reinforcement Learning (MBRL) have achieved super-human level performance on the Atari100k benchmark on average. However, we discover that conventional aggregates mask a major problem, Performance Asymmetry: MBRL agents dramatically outperform humans in certain tasks (Agent-Optimal tasks) while drastically underperform humans in other tasks (Human-Optimal tasks). Indeed, despite achieving SOTA in the overall mean Human-Normalized Scores (HNS), the SOTA agent scored the worst among baselines on Human-Optimal tasks, with a striking 21X performance gap between the Human-Optimal and Agent-Optimal subsets. To address this, we partition Atari100k evenly into Human-Optimal and Agent-Optimal subsets, and introduce a more balanced aggregate, Sym-HNS. Furthermore, we trace the striking Performance Asymmetry in the SOTA pixel diffusion world model to the curse of dimensionality and its prowess on high visual detail tasks (e.g. Breakout). To this end, we propose a novel latent end-to-end Joint Embedding DIffusion (JEDI) world model that achieves SOTA results in Sym-HNS, Human-Optimal tasks, and Breakout -- thus reversing the worsening Performance Asymmetry trend while improving computational efficiency and remaining competitive on the full Atari100k.
>
---
#### [replaced 006] Learning Humanoid End-Effector Control for Open-Vocabulary Visual Loco-Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于人形机器人操作任务，旨在解决复杂环境中物体操控的泛化问题。通过结合视觉模型与控制策略，提升末端执行器的跟踪精度，实现多样化物体的可靠操作。**

- **链接: [https://arxiv.org/pdf/2602.16705v2](https://arxiv.org/pdf/2602.16705v2)**

> **作者:** Runpei Dong; Ziyan Li; Xialin He; Saurabh Gupta
>
> **备注:** Project page: https://hero-humanoid.github.io/
>
> **摘要:** Visual loco-manipulation of arbitrary objects in the wild with humanoid robots requires accurate end-effector (EE) control and a generalizable understanding of the scene via visual inputs (e.g., RGB-D images). Existing approaches are based on real-world imitation learning and exhibit limited generalization due to the difficulty in collecting large-scale training datasets. This paper presents a new paradigm, HERO, for object loco-manipulation with humanoid robots that combines the strong generalization and open-vocabulary understanding of large vision models with strong control performance from simulated training. We achieve this by designing an accurate residual-aware EE tracking policy. This EE tracking policy combines classical robotics with machine learning. It uses a) inverse kinematics to convert residual end-effector targets into reference trajectories, b) a learned neural forward model for accurate forward kinematics, c) goal adjustment, and d) replanning. Together, these innovations help us cut down the end-effector tracking error by 3.2x. We use this accurate end-effector tracker to build a modular system for loco-manipulation, where we use open-vocabulary large vision models for strong visual generalization. Our system is able to operate in diverse real-world environments, from offices to coffee shops, where the robot is able to reliably manipulate various everyday objects (e.g., mugs, apples, toys) on surfaces ranging from 43cm to 92cm in height. Systematic modular and end-to-end tests in simulation and the real world demonstrate the effectiveness of our proposed design. We believe the advances in this paper can open up new ways of training humanoid robots to interact with daily objects.
>
---
#### [replaced 007] TouchGuide: Inference-Time Steering of Visuomotor Policies via Touch Guidance
- **分类: cs.RO**

- **简介: 该论文提出TouchGuide，解决机器人精细操作中触觉反馈利用不足的问题。通过视觉-触觉融合，在推理阶段引导预训练策略，提升接触任务性能。**

- **链接: [https://arxiv.org/pdf/2601.20239v3](https://arxiv.org/pdf/2601.20239v3)**

> **作者:** Zhemeng Zhang; Jiahua Ma; Xincheng Yang; Xin Wen; Yuzhi Zhang; Boyan Li; Yiran Qin; Jin Liu; Can Zhao; Li Kang; Haoqin Hong; Zhenfei Yin; Philip Torr; Hao Su; Ruimao Zhang; Daolin Ma
>
> **摘要:** Fine-grained and contact-rich manipulation remain challenging for robots, largely due to the underutilization of tactile feedback. To address this, we introduce TouchGuide, a novel cross-policy visuo-tactile fusion paradigm that fuses modalities within a low-dimensional action space. Specifically, TouchGuide operates in two stages to guide a pre-trained diffusion or flow-matching visuomotor policy at inference time. First, the policy produces a coarse, visually-plausible action using only visual inputs during early sampling. Second, a task-specific Contact Physical Model (CPM) provides tactile guidance to steer and refine the action, ensuring it aligns with realistic physical contact conditions. Trained through contrastive learning on limited expert demonstrations, the CPM provides a tactile-informed feasibility score to steer the sampling process toward refined actions that satisfy physical contact constraints. Furthermore, to facilitate TouchGuide training with high-quality and cost-effective data, we introduce TacUMI, a data collection system. TacUMI achieves a favorable trade-off between precision and affordability; by leveraging rigid fingertips, it obtains direct tactile feedback, thereby enabling the collection of reliable tactile data. Extensive experiments on five challenging contact-rich tasks, such as shoe lacing and chip handover, show that TouchGuide consistently and significantly outperforms state-of-the-art visuo-tactile policies.
>
---
#### [replaced 008] OVSegDT: Segmenting Transformer for Open-Vocabulary Object Goal Navigation
- **分类: cs.RO**

- **简介: 该论文属于开放词汇目标导航任务，解决模型泛化能力差和行为不安全的问题。提出OVSegDT模型，结合语义分支和熵自适应损失，提升导航性能并减少碰撞。**

- **链接: [https://arxiv.org/pdf/2508.11479v2](https://arxiv.org/pdf/2508.11479v2)**

> **作者:** Tatiana Zemskova; Aleksei Staroverov; Dmitry Yudin; Aleksandr Panov
>
> **摘要:** Open-vocabulary Object Goal Navigation requires an embodied agent to reach objects described by free-form language, including categories never seen during training. Existing end-to-end policies overfit small simulator datasets, achieving high success on training scenes but failing to generalize and exhibiting unsafe behaviour (frequent collisions). We introduce OVSegDT, a lightweight transformer policy that tackles these issues with two synergistic components. The first component is the semantic branch, which includes an encoder for the target binary mask and an auxiliary segmentation loss function, grounding the textual goal and providing precise spatial cues. The second component consists of a proposed Entropy-Adaptive Loss Modulation, a per-sample scheduler that continuously balances imitation and reinforcement signals according to the policy entropy, eliminating brittle manual phase switches. These additions cut the sample complexity of training by 33%, and reduce collision count in two times while keeping inference cost low (130M parameters, RGB-only input). On HM3D-OVON, our model matches the performance on unseen categories to that on seen ones and establishes state-of-the-art results (40.1% SR, 20.9% SPL on val unseen) without depth, odometry, or large vision-language models. Code is available at https://github.com/CognitiveAISystems/OVSegDT.
>
---
#### [replaced 009] SpikePingpong: Spike Vision-based Fast-Slow Pingpong Robot System
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人高精度操控任务，旨在解决动态环境中高速物体的精准控制问题。通过结合视觉与学习策略，实现乒乓球的快速准确击打。**

- **链接: [https://arxiv.org/pdf/2506.06690v2](https://arxiv.org/pdf/2506.06690v2)**

> **作者:** Hao Wang; Chengkai Hou; Xianglong Li; Yankai Fu; Chenxuan Li; Ning Chen; Gaole Dai; Jiaming Liu; Tiejun Huang; Shanghang Zhang
>
> **摘要:** Learning to control high-speed objects in dynamic environments represents a fundamental challenge in robotics. Table tennis serves as an ideal testbed for advancing robotic capabilities in dynamic environments. This task presents two fundamental challenges: it requires a high-precision vision system capable of accurately predicting ball trajectories under complex dynamics, and it necessitates intelligent control strategies to ensure precise ball striking to target regions. High-speed object manipulation typically demands advanced visual perception hardware capable of capturing rapid motion with exceptional temporal resolution. Drawing inspiration from Kahneman's dual-system theory, where fast intuitive processing complements slower deliberate reasoning, there exists an opportunity to develop more robust perception architectures that can handle high-speed dynamics while maintaining accuracy. To this end, we present \textit{\textbf{SpikePingpong}}, a novel system that integrates spike-based vision with imitation learning for high-precision robotic table tennis. We develop a Fast-Slow system architecture where System 1 provides rapid ball detection and preliminary trajectory prediction with millisecond-level responses, while System 2 employs spike-oriented neural calibration for precise hittable position corrections. For strategic ball striking, we introduce Imitation-based Motion Planning And Control Technology, which learns optimal robotic arm striking policies through demonstration-based learning. Experimental results demonstrate that \textit{\textbf{SpikePingpong}} achieves a remarkable 92\% success rate for 30 cm accuracy zones and 70\% in the more challenging 20 cm precision targeting. This work demonstrates the potential of Fast-Slow architectures for advancing robotic capabilities in time-critical manipulation tasks.
>
---
#### [replaced 010] Sensory-Motor Control with Large Language Models via Iterative Policy Refinement
- **分类: cs.AI; cs.HC; cs.LG; cs.RO**

- **简介: 该论文属于智能控制任务，旨在让大语言模型直接生成连续动作策略以控制实体代理。通过迭代优化，结合符号知识与感官数据，解决环境交互中的控制问题。**

- **链接: [https://arxiv.org/pdf/2506.04867v4](https://arxiv.org/pdf/2506.04867v4)**

> **作者:** Jônata Tyska Carvalho; Stefano Nolfi
>
> **备注:** Final version of the article accepted for publication on Scientific Reports. 29 pages (13 pages are from appendix), 8 figures, 2 tables, code for experiments replication and supplementary material provided at https://github.com/jtyska/llm-robotics-article/
>
> **摘要:** We propose a method that enables large language models (LLMs) to control embodied agents through the generation of control policies that directly map continuous observation vectors to continuous action vectors. At the outset, the LLMs generate a control strategy based on a textual description of the agent, its environment, and the intended goal. This strategy is then iteratively refined through a learning process in which the LLMs are repeatedly prompted to improve the current strategy, using performance feedback and sensory-motor data collected during its evaluation. The method is validated on classic control tasks from the Gymnasium library and the inverted pendulum task from the MuJoCo library. The approach proves effective with relatively compact models such as GPT-oss:120b and Qwen2.5:72b. In most cases, it successfully identifies optimal or near-optimal solutions by integrating symbolic knowledge derived through reasoning with sub-symbolic sensory-motor data gathered as the agent interacts with its environment.
>
---
#### [replaced 011] Optimal Transport-Based Decentralized Multi-Agent Distribution Matching
- **分类: eess.SY; cs.RO**

- **简介: 该论文研究多智能体系统分布匹配问题，利用最优输运理论设计去中心化控制框架，解决如何让各智能体在无全局信息下达成指定分布的问题。**

- **链接: [https://arxiv.org/pdf/2601.00548v2](https://arxiv.org/pdf/2601.00548v2)**

> **作者:** Kooktae Lee
>
> **摘要:** This paper presents a decentralized control framework for distribution matching in multi-agent systems (MAS), where agents collectively achieve a prescribed terminal spatial distribution. The problem is formulated using optimal transport (Wasserstein distance), which provides a principled measure of distributional discrepancy and serves as the basis for the control design. To avoid solving the global optimal transport problem directly, the distribution-matching objective is reformulated into a tractable per-agent decision process, enabling each agent to identify its desired terminal locations using only locally available information. A sequential weight-update rule is introduced to construct feasible local transport plans, and a memory-based correction mechanism is incorporated to maintain reliable operation under intermittent and range-limited communication. Convergence guarantees are established, showing cycle-wise improvement of a surrogate transport cost under both linear and nonlinear agent dynamics. Simulation results demonstrate that the proposed framework achieves effective and scalable distribution matching while operating fully in a decentralized manner.
>
---
#### [replaced 012] Soft Surfaced Vision-Based Tactile Sensing for Bipedal Robot Applications
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，旨在提升双足机器人的平衡与环境感知。通过视觉触觉传感器捕捉足部接触信息，提高稳定性与适应性。**

- **链接: [https://arxiv.org/pdf/2602.18638v2](https://arxiv.org/pdf/2602.18638v2)**

> **作者:** Jaeeun Kim; Junhee Lim; Yu She
>
> **备注:** 8 pages, 11 figures, RoboSoft 2026. For the supplementary video, please visit: https://youtu.be/ceJiy9q_2Aw Section IV-D updated
>
> **摘要:** Legged locomotion benefits from embodied sensing, where perception emerges from the physical interaction between body and environment. We present a soft-surfaced, vision-based tactile foot sensor that endows a bipedal robot with a skin-like deformable layer that captures contact deformations optically, turning foot-ground interactions into rich haptic signals. From a contact image stream, our method estimates contact pose (position and orientation), visualizes shear, computes center of pressure (CoP), classifies terrain, and detects geometric features of the contact patch. We validate these capabilities on a tilting platform and in visually obscured conditions, showing that foot-borne tactile feedback improves balance control and terrain awareness beyond proprioception alone. These findings suggest that integrating tactile perception into legged robot feet improves stability, adaptability, and environmental awareness, offering a promising direction toward more compliant and intelligent locomotion systems. For the supplementary video, please visit: https://youtu.be/ceJiy9q_2Aw
>
---
#### [replaced 013] Interaction-Aware Model Predictive Decision-Making for Socially-Compliant Autonomous Driving in Mixed Urban Traffic Scenarios
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于自动驾驶任务，解决混合交通中车辆与行人交互的决策问题。提出IAMPDM框架，结合MPC与意图模型，提升安全性和社会合规性。**

- **链接: [https://arxiv.org/pdf/2503.01852v2](https://arxiv.org/pdf/2503.01852v2)**

> **作者:** Balint Varga; Thomas Brand; Marcus Schmitz; Ehsan Hashemi
>
> **备注:** Major Revision
>
> **摘要:** Autonomous vehicles must negotiate with pedestrians in ways that are both safe and socially compliant. We present an interaction-aware model predictive decision-making (IAMPDM) framework that integrates a gap-acceptance-inspired intention model with MPC to jointly reason about human intent and vehicle control in real time. The pedestrian module produces a continuous crossing-propensity signal - driven by time-to-collision (TTC) with an intention discounting mechanism - that modulates MPC safety terms and minimum-distance constraints. We implement IAMPDM in a projection-based, motion-tracked simulator and compare it against a rule-based intention-aware controller (RBDM) and a conservative non-interactive baseline (NIA). In a human-in-the-decision-loop study with 25 participants, intention-aware methods shortened negotiation and completion time relative to NIA across scenarios, at the expense of tighter TTC/DST margins, with no significant difference between IAMPDM and RBDM except for TTC in one scenario. Results indicate that intention-aware decision-making algorithms reduce pedestrian crossing time and improve subjective ratings of comfort, safety, and trust relative to a non-cooperative decision-making algorithm. We discuss implications for real-world deployment of interaction-aware autonomous vehicles. We detail decision-making calibration and real-time implementation (CasADi/IPOPT) and propose deployment guardrails - minimum surrogate-safety margins, deadlock prevention - to balance efficiency with safety.
>
---
#### [replaced 014] On Robustness of Vision-Language-Action Model against Multi-Modal Perturbations
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文研究视觉-语言-动作模型在多模态扰动下的鲁棒性，解决实际部署中多模态干扰问题，提出RobustVLA方法提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.00037v4](https://arxiv.org/pdf/2510.00037v4)**

> **作者:** Jianing Guo; Zhenhong Wu; Chang Tu; Yiyao Ma; Xiangqi Kong; Zhiqian Liu; Jiaming Ji; Shuning Zhang; Yuanpei Chen; Kai Chen; Qi Dou; Yaodong Yang; Xianglong Liu; Huijie Zhao; Weifeng Lv; Simin Li
>
> **摘要:** In Vision-Language-Actionf(VLA) models, robustness to real-world perturbations is critical for deployment. Existing methods target simple visual disturbances, overlooking the broader multi-modal perturbations that arise in actions, instructions, environments, and observations. Here, we first evaluate the robustness of mainstream VLAs under 17 perturbations across four modalities. We find (1) actions as the most fragile modality, (2) Existing visual-robust VLA do not gain robustness in other modality, and (3) pi0 demonstrates superior robustness. To build multi-modal robust VLAs, we propose RobustVLA against perturbations in VLA inputs and outputs. For output robustness, we perform offline robust optimization against worst-case action noise that maximizes mismatch in flow matching objective. This can be seen as adversarial training, label smoothing, and outlier penalization. For input robustness, we enforce consistent actions across input variations that preserve task semantics. To account for multiple perturbations, we formulate robustness as a multi-armed bandit problem and apply an upper confidence bound algorithm to automatically identify the most harmful noise. Experiments on LIBERO demonstrate our RobustVLA delivers absolute gains over baselines of 12.6% on the pi0 backbone and 10.4% on the OpenVLA backbone across all 17 perturbations, achieving 50.6x faster inference than existing visual-robust BYOVLA that requires external LLMs, and a 10.4% gain under mixed perturbations. On the real-world FR5 robot, under four types of multimodal perturbations, RobustVLA shows strong low-data performance, outperforming pi0 by 65.6% success rate with 25 demonstrations. Even with abundant demos, our method still outperform pi0 by 30% success rate. Code and demo videos available at https://github.com/gakakulicc/RobustVLA.
>
---
#### [replaced 015] An Efficient LiDAR-Camera Fusion Network for Multi-Class 3D Dynamic Object Detection and Trajectory Prediction
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于3D动态目标检测与轨迹预测任务，旨在提升服务机器人在复杂环境中的感知能力。提出一种高效融合LiDAR与相机的框架，实现快速准确的目标检测与轨迹预测。**

- **链接: [https://arxiv.org/pdf/2504.13647v2](https://arxiv.org/pdf/2504.13647v2)**

> **作者:** Yushen He; Lei Zhao; Tianchen Deng; Zipeng Fang; Weidong Chen
>
> **摘要:** Service mobile robots are often required to avoid dynamic objects while performing their tasks, but they usually have only limited computational resources. To further advance the practical application of service robots in complex dynamic environments, we propose an efficient multi-modal framework for 3D object detection and trajectory prediction, which synergistically integrates LiDAR and camera inputs to achieve real-time perception of pedestrians, vehicles, and riders in 3D space.The framework incorporates two novel models: 1) a Unified modality detector with Mamba and Transformer (UniMT) for object detection, which achieves high-accuracy object detection with fast inference speed, and 2) a Reference Trajectory-based Multi-Class Transformer (RTMCT) for efficient and diverse trajectory prediction of multi-class objects with flexible-length trajectories. Evaluations on the CODa benchmark demonstrate that our method outperforms existing ones in both detection (+3.71\% in mAP) and trajectory prediction (-0.408m in minADE$_5$ of pedestrians) metrics. Furthermore, on the challenging nuScenes detection benchmark, our detection model achieves competitive performance among LiDAR-camera fusion methods, with a mAP of 72.7\% and NDS of 75.3\%. Remarkably, the system demonstrates exceptional generalizability and practical deployment potential. When transferred and implemented on a wheelchair robot with an entry-level NVIDIA RTX 3060 GPU, it achieves real-time inference at 13.9 frames per second (FPS) with satisfactory accuracy. To facilitate reproducibility and practical deployment, we release the related code of the method at \href{https://github.com/TossherO/3D_Perception}{https://github.com/TossherO/3D\_Perception} and its ROS inference version at \href{https://github.com/TossherO/ros_packages}{https://github.com/TossherO/ros\_packages}.
>
---
#### [replaced 016] "Don't Do That!": Guiding Embodied Systems through Large Language Model-based Constraint Generation
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于机器人导航任务，解决如何将自然语言约束转化为可执行代码的问题。工作包括提出STPR框架，利用LLM生成Python函数实现约束，提升导航合规性与效率。**

- **链接: [https://arxiv.org/pdf/2506.04500v2](https://arxiv.org/pdf/2506.04500v2)**

> **作者:** Amin Seffo; Aladin Djuhera; Masataro Asai; Holger Boche
>
> **备注:** Preprint; under review
>
> **摘要:** Recent advancements in large language models (LLMs) have spurred interest in robotic navigation that incorporates complex spatial, mathematical, and conditional constraints from natural language into the planning problem. Such constraints can be informal yet highly complex, making it challenging to translate into a formal description that can be passed on to a planning algorithm. In this paper, we propose STPR, a constraint generation framework that uses LLMs to translate constraints (expressed as instructions on ``what not to do'') into executable Python functions. STPR leverages the LLM's strong coding capabilities to shift the problem description from language into structured and transparent code, thus circumventing complex reasoning and avoiding potential hallucinations. We show that these LLM-generated functions accurately describe even complex mathematical constraints, and apply them to point cloud representations with traditional search algorithms. Experiments in a simulated Gazebo environment show that STPR ensures full compliance across several constraints and scenarios, while having short runtimes. We also verify that STPR can be used with smaller, code-specific LLMs, making it applicable to a wide range of compact models at low inference cost.
>
---
#### [replaced 017] NRSeg: Noise-Resilient Learning for BEV Semantic Segmentation via Driving World Models
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文针对BEV语义分割任务，解决标注数据分布单一和合成数据噪声问题，提出NRSeg框架提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2507.04002v2](https://arxiv.org/pdf/2507.04002v2)**

> **作者:** Siyu Li; Fei Teng; Yihong Cao; Kailun Yang; Zhiyong Li; Yaonan Wang
>
> **备注:** Accepted to IEEE Transactions on Image Processing (TIP). The source code will be made publicly available at https://github.com/lynn-yu/NRSeg
>
> **摘要:** Birds' Eye View (BEV) semantic segmentation is an indispensable perception task in end-to-end autonomous driving systems. Unsupervised and semi-supervised learning for BEV tasks, as pivotal for real-world applications, underperform due to the homogeneous distribution of the labeled data. In this work, we explore the potential of synthetic data from driving world models to enhance the diversity of labeled data for robustifying BEV segmentation. Yet, our preliminary findings reveal that generation noise in synthetic data compromises efficient BEV model learning. To fully harness the potential of synthetic data from world models, this paper proposes NRSeg, a noise-resilient learning framework for BEV semantic segmentation. Specifically, a Perspective-Geometry Consistency Metric (PGCM) is proposed to quantitatively evaluate the guidance capability of generated data for model learning. This metric originates from the alignment measure between the perspective road mask of generated data and the mask projected from the BEV labels. Moreover, a Bi-Distribution Parallel Prediction (BiDPP) is designed to enhance the inherent robustness of the model, where the learning process is constrained through parallel prediction of multinomial and Dirichlet distributions. The former efficiently predicts semantic probabilities, whereas the latter adopts evidential deep learning to realize uncertainty quantification. Furthermore, a Hierarchical Local Semantic Exclusion (HLSE) module is designed to address the non-mutual exclusivity inherent in BEV semantic segmentation tasks. Experimental results demonstrate that NRSeg achieves state-of-the-art performance, yielding the highest improvements in mIoU of 13.8% and 11.4% in unsupervised and semi-supervised BEV segmentation tasks, respectively. The source code will be made publicly available at https://github.com/lynn-yu/NRSeg.
>
---
#### [replaced 018] LLM-Driven Corrective Robot Operation Code Generation with Static Text-Based Simulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作代码生成任务，旨在解决LLM生成代码可靠性不足的问题。通过构建静态文本模拟，实现无需物理实验或仿真环境的代码修正与生成。**

- **链接: [https://arxiv.org/pdf/2512.02002v3](https://arxiv.org/pdf/2512.02002v3)**

> **作者:** Wenhao Wang; Yi Rong; Yanyan Li; Long Jiao; Jiawei Yuan
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** Recent advances in Large language models (LLMs) have demonstrated their promising capabilities of generating robot operation code to enable LLM-driven robots. To enhance the reliability of operation code generated by LLMs, corrective designs with feedback from the observation of executing code have been increasingly adopted in existing research. However, the code execution in these designs relies on either a physical experiment or a customized simulation environment, which limits their deployment due to the high configuration effort of the environment and the potential long execution time. In this paper, we explore the possibility of directly leveraging LLM to enable static simulation of robot operation code, and then leverage it to design a new reliable LLM-driven corrective robot operation code generation framework. Our framework configures the LLM as a static simulator with enhanced capabilities that reliably simulate robot code execution by interpreting actions, reasoning over state transitions, analyzing execution outcomes, and generating semantic observations that accurately capture trajectory dynamics. To validate the performance of our framework, we performed experiments on various operation tasks for different robots, including UAVs and small ground vehicles. The experiment results not only demonstrated the high accuracy of our static text-based simulation but also the reliable code generation of our LLM-driven corrective framework, which achieves a comparable performance with state-of-the-art research while does not rely on dynamic code execution using physical experiments or simulators.
>
---
#### [replaced 019] Fast-ThinkAct: Efficient Vision-Language-Action Reasoning via Verbalizable Latent Planning
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于视觉-语言-动作推理任务，解决推理延迟高问题。提出Fast-ThinkAct框架，通过可表述的潜在推理实现高效规划，降低延迟并保持性能。**

- **链接: [https://arxiv.org/pdf/2601.09708v2](https://arxiv.org/pdf/2601.09708v2)**

> **作者:** Chi-Pin Huang; Yunze Man; Zhiding Yu; Min-Hung Chen; Jan Kautz; Yu-Chiang Frank Wang; Fu-En Yang
>
> **备注:** CVPR 2026. Project page: https://jasper0314-huang.github.io/fast-thinkact/
>
> **摘要:** Vision-Language-Action (VLA) tasks require reasoning over complex visual scenes and executing adaptive actions in dynamic environments. While recent studies on reasoning VLAs show that explicit chain-of-thought (CoT) can improve generalization, they suffer from high inference latency due to lengthy reasoning traces. We propose Fast-ThinkAct, an efficient reasoning framework that achieves compact yet performant planning through verbalizable latent reasoning. Fast-ThinkAct learns to reason efficiently with latent CoTs by distilling from a teacher, driven by a preference-guided objective to align manipulation trajectories that transfers both linguistic and visual planning capabilities for embodied control. This enables reasoning-enhanced policy learning that effectively connects compact reasoning to action execution. Extensive experiments across diverse embodied manipulation and reasoning benchmarks demonstrate that Fast-ThinkAct achieves strong performance with up to 89.3% reduced inference latency over state-of-the-art reasoning VLAs, while maintaining effective long-horizon planning, few-shot adaptation, and failure recovery.
>
---
#### [replaced 020] Adaptive Evolutionary Framework for Safe, Efficient, and Cooperative Autonomous Vehicle Interactions
- **分类: cs.MA; cs.RO**

- **简介: 该论文属于自动驾驶车辆协同任务，解决AV间安全高效交互问题。提出基于进化博弈论的框架，通过自适应策略优化提升安全性与效率。**

- **链接: [https://arxiv.org/pdf/2509.07411v2](https://arxiv.org/pdf/2509.07411v2)**

> **作者:** Yijun Lu; Zhen Tian; Zhihao Lin
>
> **摘要:** Modern transportation systems face significant challenges in ensuring road safety, given serious injuries caused by road accidents. The rapid growth of autonomous vehicles (AVs) has prompted new traffic designs that aim to optimize interactions among AVs. However, effective interactions between AVs remains challenging due to the absence of centralized control. Besides, there is a need for balancing multiple factors, including passenger demands and overall traffic efficiency. Traditional rule-based, optimization-based, and game-theoretic approaches each have limitations in addressing these challenges. Rule-based methods struggle with adaptability and generalization in complex scenarios, while optimization-based methods often require high computational resources. Game-theoretic approaches, such as Stackelberg and Nash games, suffer from limited adaptability and potential inefficiencies in cooperative settings. This paper proposes an Evolutionary Game Theory (EGT)-based framework for AV interactions that overcomes these limitations by utilizing a decentralized and adaptive strategy evolution mechanism. A causal evaluation module (CEGT) is introduced to optimize the evolutionary rate, balancing mutation and evolution by learning from historical interactions. Simulation results demonstrate the proposed CEGT outperforms EGT and popular benchmark games in terms of lower collision rates, improved safety distances, higher speeds, and overall better performance compared to Nash and Stackelberg games across diverse scenarios and parameter settings.
>
---
#### [replaced 021] DeLTa: Demonstration and Language-Guided Novel Transparent Object Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决透明物体长时序、高精度操控问题。提出DeLTa框架，结合语言指令与视觉感知，实现无需额外训练的透明物体操作。**

- **链接: [https://arxiv.org/pdf/2510.05662v2](https://arxiv.org/pdf/2510.05662v2)**

> **作者:** Taeyeop Lee; Gyuree Kang; Bowen Wen; Youngho Kim; Seunghyeok Back; In So Kweon; David Hyunchul Shim; Kuk-Jin Yoon
>
> **备注:** Project page: https://sites.google.com/view/DeLTa25/
>
> **摘要:** Despite the prevalence of transparent object interactions in human everyday life, transparent robotic manipulation research remains limited to short-horizon tasks and basic grasping capabilities. Although some methods have partially addressed these issues, most of them have limitations in generalization to novel objects and are insufficient for precise long-horizon robot manipulation. To address this limitation, we propose DeLTa (Demonstration and Language-Guided Novel Transparent Object Manipulation), a novel framework that integrates depth estimation, 6D pose estimation, and vision-language planning for precise long-horizon manipulation of transparent objects guided by natural language task instructions. A key advantage of our method is its single-demonstration approach, which generalizes 6D trajectories to novel transparent objects without requiring category-level priors or additional training. Additionally, we present a task planner that refines the VLM-generated plan to account for the constraints of a single-arm, eye-in-hand robot for long-horizon object manipulation tasks. Through comprehensive evaluation, we demonstrate that our method significantly outperforms existing transparent object manipulation approaches, particularly in long-horizon scenarios requiring precise manipulation capabilities. Project page: https://sites.google.com/view/DeLTa25/
>
---
#### [replaced 022] Scout-Rover cooperation: online terrain strength mapping and traversal risk estimation for planetary-analog explorations
- **分类: cs.RO**

- **简介: 该论文属于行星探测任务，解决松散地形安全导航问题。通过 scout-rover 协作，实现地形强度在线映射和路径风险评估，提升探索效率与安全性。**

- **链接: [https://arxiv.org/pdf/2602.18688v2](https://arxiv.org/pdf/2602.18688v2)**

> **作者:** Shipeng Liu; J. Diego Caporale; Yifeng Zhang; Xingjue Liao; William Hoganson; Wilson Hu; Shivangi Misra; Neha Peddinti; Rachel Holladay; Ethan Fulcher; Akshay Ram Panyam; Andrik Puentes; Jordan M. Bretzfelder; Michael Zanetti; Uland Wong; Daniel E. Koditschek; Mark Yim; Douglas Jerolmack; Cynthia Sung; Feifei Qian
>
> **备注:** 8 figures
>
> **摘要:** Robot-aided exploration of planetary surfaces is essential for understanding geologic processes, yet many scientifically valuable regions, such as Martian dunes and lunar craters, remain hazardous due to loose, deformable regolith. We present a scout-rover cooperation framework that expands safe access to such terrain using a hybrid team of legged and wheeled robots. In our approach, a high-mobility legged robot serves as a mobile scout, using proprioceptive leg-terrain interactions to estimate regolith strength during locomotion and construct spatially resolved terrain maps. These maps are integrated with rover locomotion models to estimate traversal risk and inform path planning. We validate the framework through analogue missions at the NASA Ames Lunar Simulant Testbed and the White Sands Dune Field. Experiments demonstrate (1) online terrain strength mapping from legged locomotion and (2) rover-specific traversal-risk estimation enabling safe navigation to scientific targets. Results show that scout-generated terrain maps reliably capture spatial variability and predict mobility failure modes, allowing risk-aware path planning that avoids hazardous regions. By combining embodied terrain sensing with heterogeneous rover cooperation, this framework enhances operational robustness and expands the reachable science workspace in deformable planetary environments.
>
---
#### [replaced 023] DiSPo: Diffusion-SSM based Policy Learning for Coarse-to-Fine Action Discretization
- **分类: cs.RO**

- **简介: 该论文属于强化学习中的技能学习任务，解决粗到细动作离散化问题。提出DiSPo方法，利用扩散-状态空间模型实现高效、可扩展的动作控制。**

- **链接: [https://arxiv.org/pdf/2409.14719v4](https://arxiv.org/pdf/2409.14719v4)**

> **作者:** Nayoung Oh; Jaehyeong Jang; Moonkyeong Jung; Daehyung Park
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** We aim to solve the problem of generating coarse-to-fine skills learning from demonstrations (LfD). To scale precision, traditional LfD approaches often rely on extensive fine-grained demonstrations with external interpolations or dynamics models with limited generalization capabilities. For memory-efficient learning and convenient granularity change, we propose a novel diffusion-state space model (SSM) based policy (DiSPo) that learns from diverse coarse skills and produces varying control scales of actions by leveraging an SSM, Mamba. Our evaluations show the adoption of Mamba and the proposed step-scaling method enable DiSPo to outperform in three coarse-to-fine benchmark tests with maximum 81% higher success rate than baselines. In addition, DiSPo improves inference efficiency by generating coarse motions in less critical regions. We finally demonstrate the scalability of actions with simulation and real-world manipulation tasks. Code and Videos are available at https://robo-dispo.github.io.
>
---
#### [replaced 024] Noise-enabled goal attainment in crowded collectives
- **分类: cs.RO; cond-mat.soft; cs.MA**

- **简介: 该论文研究拥挤环境中噪声对群体目标达成的影响，旨在优化交通流量。通过模拟、理论和实验，发现适当噪声可减少拥堵，提升效率。任务是改善群体导航性能。**

- **链接: [https://arxiv.org/pdf/2507.08100v2](https://arxiv.org/pdf/2507.08100v2)**

> **作者:** Lucy Liu; Justin Werfel; Federico Toschi; L. Mahadevan
>
> **摘要:** In crowded environments, individuals must navigate around other occupants to reach their destinations. Understanding and controlling traffic flows in these spaces is relevant for coordinating robot swarms and designing infrastructure for dense populations. Here, we use simulations, theory, and experiments to study how adding stochasticity to agent motion can reduce traffic jams and help agents travel more quickly to prescribed goals. A computational approach reveals the collective behavior. Above a critical noise level, large jams do not persist. From this observation, we analytically approximate the swarm's goal attainment rate, which allows us to solve for the agent density and noise level that maximize the goals reached. Robotic experiments corroborate the behaviors observed in our simulated and theoretical results. Finally, we compare simple, local navigation approaches with a sophisticated but computationally costly central planner. A simple reactive scheme performs well up to moderate densities and is far more computationally efficient than a planner, motivating further research into robust, decentralized navigation methods for crowded environments. By integrating ideas from physics and engineering using simulations, theory, and experiments, our work identifies new directions for emergent traffic research.
>
---
#### [replaced 025] PegasusFlow: Parallel Rolling-Denoising Score Sampling for Robot Diffusion Planner Flow Matching
- **分类: cs.RO**

- **简介: 该论文属于机器人轨迹规划任务，解决扩散模型依赖专家数据的问题，提出PegasusFlow框架和WBFO算法，实现高效并行采样与优化。**

- **链接: [https://arxiv.org/pdf/2509.08435v2](https://arxiv.org/pdf/2509.08435v2)**

> **作者:** Lei Ye; Haibo Gao; Peng Xu; Zhelin Zhang; Junqi Shan; Ao Zhang; Wei Zhang; Ruyi Zhou; Zongquan Deng; Liang Ding
>
> **备注:** 8 pages, 7 figures, conference paper
>
> **摘要:** Diffusion models offer powerful generative capabilities for robot trajectory planning, yet their practical deployment on robots is hindered by a critical bottleneck: a reliance on imitation learning from expert demonstrations. This paradigm is often impractical for specialized robots where data is scarce and creates an inefficient, theoretically suboptimal training pipeline. To overcome this, we introduce PegasusFlow, a hierarchical rolling-denoising framework that enables direct and parallel sampling of trajectory score gradients from environmental interaction, completely bypassing the need for expert data. Our core innovation is a novel sampling algorithm, Weighted Basis Function Optimization (WBFO), which leverages spline basis representations to achieve superior sample efficiency and faster convergence compared to traditional methods like MPPI. The framework is embedded within a scalable, asynchronous parallel simulation architecture that supports massively parallel rollouts for efficient data collection. Extensive experiments on trajectory optimization and robotic navigation tasks demonstrate that our approach, particularly Action-Value WBFO (AVWBFO) combined with a reinforcement learning warm-start, significantly outperforms baselines. In a challenging barrier-crossing task, our method achieved a 100% success rate and was 18% faster than the next-best method, validating its effectiveness for complex terrain locomotion planning. https://masteryip.github.io/pegasusflow.github.io/
>
---
#### [replaced 026] Human-Exoskeleton Kinematic Calibration to Improve Hand Tracking for Dexterous Teleoperation
- **分类: cs.RO; cs.HC; eess.SY**

- **简介: 该论文属于手部运动跟踪任务，解决因个体差异导致的外骨骼定位不准问题。通过优化算法提升跟踪精度，适用于精密操作和机器人学习。**

- **链接: [https://arxiv.org/pdf/2507.23592v3](https://arxiv.org/pdf/2507.23592v3)**

> **作者:** Haiyun Zhang; Stefano Dalla Gasperina; Saad N. Yousaf; Toshimitsu Tsuboi; Tetsuya Narita; Ashish D. Deshpande
>
> **备注:** 8 pages, 10 figures, 1 supplementary video, submitted to RA-L
>
> **摘要:** Hand exoskeletons are critical tools for dexterous teleoperation and immersive manipulation interfaces, but achieving accurate hand tracking remains a challenge due to user-specific anatomical variability and donning inconsistencies. These issues lead to kinematic misalignments that degrade tracking performance and limit applicability in precision tasks. We propose a subject-specific calibration framework for exoskeleton-based hand tracking that estimates virtual link parameters through residual-weighted optimization. A data-driven approach is introduced to empirically tune cost function weights using motion capture ground truth, enabling accurate and consistent calibration across users. Implemented on the Maestro hand exoskeleton with seven healthy participants, the method achieved substantial reductions in joint and fingertip tracking errors across diverse hand geometries. Qualitative visualizations using a Unity-based virtual hand further demonstrate improved motion fidelity. The proposed framework generalizes to exoskeletons with closed-loop kinematics and minimal sensing, laying the foundation for high-fidelity teleoperation and robot learning applications.
>
---
#### [replaced 027] SimToolReal: An Object-Centric Policy for Zero-Shot Dexterous Tool Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人工具操作任务，解决sim-to-real迁移中需大量工程定制的问题。提出SimToolReal，通过模拟生成多样化工具并训练通用策略，实现零样本真实世界操作。**

- **链接: [https://arxiv.org/pdf/2602.16863v2](https://arxiv.org/pdf/2602.16863v2)**

> **作者:** Kushal Kedia; Tyler Ga Wei Lum; Jeannette Bohg; C. Karen Liu
>
> **摘要:** The ability to manipulate tools significantly expands the set of tasks a robot can perform. Yet, tool manipulation represents a challenging class of dexterity, requiring grasping thin objects, in-hand object rotations, and forceful interactions. Since collecting teleoperation data for these behaviors is challenging, sim-to-real reinforcement learning (RL) is a promising alternative. However, prior approaches typically require substantial engineering effort to model objects and tune reward functions for each task. In this work, we propose SimToolReal, taking a step towards generalizing sim-to-real RL policies for tool manipulation. Instead of focusing on a single object and task, we procedurally generate a large variety of tool-like object primitives in simulation and train a single RL policy with the universal goal of manipulating each object to random goal poses. This approach enables SimToolReal to perform general dexterous tool manipulation at test-time without any object or task-specific training. We demonstrate that SimToolReal outperforms prior retargeting and fixed-grasp methods by 37% while matching the performance of specialist RL policies trained on specific target objects and tasks. Finally, we show that SimToolReal generalizes across a diverse set of everyday tools, achieving strong zero-shot performance over 120 real-world rollouts spanning 24 tasks, 12 object instances, and 6 tool categories.
>
---
#### [replaced 028] A Very Big Video Reasoning Suite
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.RO**

- **简介: 该论文属于视频推理任务，旨在解决视频模型推理能力不足的问题。提出VBVR数据集和评估框架，推动视频推理研究的发展。**

- **链接: [https://arxiv.org/pdf/2602.20159v2](https://arxiv.org/pdf/2602.20159v2)**

> **作者:** Maijunxian Wang; Ruisi Wang; Juyi Lin; Ran Ji; Thaddäus Wiedemer; Qingying Gao; Dezhi Luo; Yaoyao Qian; Lianyu Huang; Zelong Hong; Jiahui Ge; Qianli Ma; Hang He; Yifan Zhou; Lingzi Guo; Lantao Mei; Jiachen Li; Hanwen Xing; Tianqi Zhao; Fengyuan Yu; Weihang Xiao; Yizheng Jiao; Jianheng Hou; Danyang Zhang; Pengcheng Xu; Boyang Zhong; Zehong Zhao; Gaoyun Fang; John Kitaoka; Yile Xu; Hua Xu; Kenton Blacutt; Tin Nguyen; Siyuan Song; Haoran Sun; Shaoyue Wen; Linyang He; Runming Wang; Yanzhi Wang; Mengyue Yang; Ziqiao Ma; Raphaël Millière; Freda Shi; Nuno Vasconcelos; Daniel Khashabi; Alan Yuille; Yilun Du; Ziming Liu; Bo Li; Dahua Lin; Ziwei Liu; Vikash Kumar; Yijiang Li; Lei Yang; Zhongang Cai; Hokin Deng
>
> **备注:** Homepage: https://video-reason.com/
>
> **摘要:** Rapid progress in video models has largely focused on visual quality, leaving their reasoning capabilities underexplored. Video reasoning grounds intelligence in spatiotemporally consistent visual environments that go beyond what text can naturally capture, enabling intuitive reasoning over spatiotemporal structure such as continuity, interaction, and causality. However, systematically studying video reasoning and its scaling behavior is hindered by the lack of large-scale training data. To address this gap, we introduce the Very Big Video Reasoning (VBVR) Dataset, an unprecedentedly large-scale resource spanning 200 curated reasoning tasks following a principled taxonomy and over one million video clips, approximately three orders of magnitude larger than existing datasets. We further present VBVR-Bench, a verifiable evaluation framework that moves beyond model-based judging by incorporating rule-based, human-aligned scorers, enabling reproducible and interpretable diagnosis of video reasoning capabilities. Leveraging the VBVR suite, we conduct one of the first large-scale scaling studies of video reasoning and observe early signs of emergent generalization to unseen reasoning tasks. Together, VBVR lays a foundation for the next stage of research in generalizable video reasoning. The data, benchmark toolkit, and models are publicly available at https://video-reason.com/ .
>
---
