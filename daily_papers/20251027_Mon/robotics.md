# 机器人 cs.RO

- **最新发布 27 篇**

- **更新 21 篇**

## 最新发布

#### [new 001] HRT1: One-Shot Human-to-Robot Trajectory Transfer for Mobile Manipulation
- **分类: cs.RO**

- **简介: 该论文提出HRT1系统，实现机器人通过观看一次人类演示视频完成移动操作任务。解决从人类示范到机器人执行的跨环境迁移问题。工作包括：基于AR头显采集视角视频、理解视频中物体与手部轨迹、将人手轨迹映射为机器人末端轨迹，并优化生成机器人配置空间轨迹。**

- **链接: [http://arxiv.org/pdf/2510.21026v1](http://arxiv.org/pdf/2510.21026v1)**

> **作者:** Sai Haneesh Allu; Jishnu Jaykumar P; Ninad Khargonkar; Tyler Summers; Jian Yao; Yu Xiang
>
> **备注:** 14 pages, 11 figures and 3 tables. Project page is available at \url{https://irvlutd.github.io/HRT1/}
>
> **摘要:** We introduce a novel system for human-to-robot trajectory transfer that enables robots to manipulate objects by learning from human demonstration videos. The system consists of four modules. The first module is a data collection module that is designed to collect human demonstration videos from the point of view of a robot using an AR headset. The second module is a video understanding module that detects objects and extracts 3D human-hand trajectories from demonstration videos. The third module transfers a human-hand trajectory into a reference trajectory of a robot end-effector in 3D space. The last module utilizes a trajectory optimization algorithm to solve a trajectory in the robot configuration space that can follow the end-effector trajectory transferred from the human demonstration. Consequently, these modules enable a robot to watch a human demonstration video once and then repeat the same mobile manipulation task in different environments, even when objects are placed differently from the demonstrations. Experiments of different manipulation tasks are conducted on a mobile manipulator to verify the effectiveness of our system
>
---
#### [new 002] Underwater Visual-Inertial-Acoustic-Depth SLAM with DVL Preintegration for Degraded Environments
- **分类: cs.RO**

- **简介: 该论文提出一种融合视觉、惯性、声学与深度信息的水下SLAM系统，针对水下视觉退化问题，通过多传感器紧耦合与改进的DVL预积分策略，提升在低可见度环境下的定位稳定性与精度。**

- **链接: [http://arxiv.org/pdf/2510.21215v1](http://arxiv.org/pdf/2510.21215v1)**

> **作者:** Shuoshuo Ding; Tiedong Zhang; Dapeng Jiang; Ming Lei
>
> **备注:** 10 pages, 10 figures
>
> **摘要:** Visual degradation caused by limited visibility, insufficient lighting, and feature scarcity in underwater environments presents significant challenges to visual-inertial simultaneous localization and mapping (SLAM) systems. To address these challenges, this paper proposes a graph-based visual-inertial-acoustic-depth SLAM system that integrates a stereo camera, an inertial measurement unit (IMU), the Doppler velocity log (DVL), and a pressure sensor. The key innovation lies in the tight integration of four distinct sensor modalities to ensure reliable operation, even under degraded visual conditions. To mitigate DVL drift and improve measurement efficiency, we propose a novel velocity-bias-based DVL preintegration strategy. At the frontend, hybrid tracking strategies and acoustic-inertial-depth joint optimization enhance system stability. Additionally, multi-source hybrid residuals are incorporated into a graph optimization framework. Extensive quantitative and qualitative analyses of the proposed system are conducted in both simulated and real-world underwater scenarios. The results demonstrate that our approach outperforms current state-of-the-art stereo visual-inertial SLAM systems in both stability and localization accuracy, exhibiting exceptional robustness, particularly in visually challenging environments.
>
---
#### [new 003] Enhancing Social Robots through Resilient AI
- **分类: cs.RO; cs.AI**

- **简介: 该论文聚焦于提升社交机器人在复杂环境中的韧性，解决其在压力或故障下仍需保持可靠运行的问题。通过构建具备韧性的人工智能系统，确保机器人在老年用户等敏感场景中维持信任与基本功能，增强其在医疗、教育等领域的实用性和安全性。**

- **链接: [http://arxiv.org/pdf/2510.21469v1](http://arxiv.org/pdf/2510.21469v1)**

> **作者:** Domenico Palmisano; Giuseppe Palestra; Berardina Nadja De Carolis
>
> **备注:** 8 pages, Workshop on Adaptive Social Interaction based on user's Mental mOdels and behaVior in HRI, The 17th International Conference on Social Robotics, 10-12 September 2025, Naples (IT)
>
> **摘要:** As artificial intelligence continues to advance and becomes more integrated into sensitive areas like healthcare, education, and everyday life, it's crucial for these systems to be both resilient and robust. This paper shows how resilience is a fundamental characteristic of social robots, which, through it, ensure trust in the robot itself-an essential element especially when operating in contexts with elderly people, who often have low trust in these systems. Resilience is therefore the ability to operate under adverse or stressful conditions, even when degraded or weakened, while maintaining essential operational capabilities.
>
---
#### [new 004] ROPES: Robotic Pose Estimation via Score-Based Causal Representation Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出ROPES，一种基于得分的因果表示学习方法，用于机器人位姿估计任务。针对无监督下从图像中恢复位置与姿态的问题，通过干预性因果学习分离可操控的生成因子（如关节角度），仅利用分布变化实现高保真度解耦，无需标签数据。**

- **链接: [http://arxiv.org/pdf/2510.20884v1](http://arxiv.org/pdf/2510.20884v1)**

> **作者:** Pranamya Kulkarni; Puranjay Datta; Burak Varıcı; Emre Acartürk; Karthikeyan Shanmugam; Ali Tajer
>
> **备注:** A preliminary version of this paper appeared at NeurIPS 2025 Workshop on Embodied World Models for Decision Making
>
> **摘要:** Causal representation learning (CRL) has emerged as a powerful unsupervised framework that (i) disentangles the latent generative factors underlying high-dimensional data, and (ii) learns the cause-and-effect interactions among the disentangled variables. Despite extensive recent advances in identifiability and some practical progress, a substantial gap remains between theory and real-world practice. This paper takes a step toward closing that gap by bringing CRL to robotics, a domain that has motivated CRL. Specifically, this paper addresses the well-defined robot pose estimation -- the recovery of position and orientation from raw images -- by introducing Robotic Pose Estimation via Score-Based CRL (ROPES). Being an unsupervised framework, ROPES embodies the essence of interventional CRL by identifying those generative factors that are actuated: images are generated by intrinsic and extrinsic latent factors (e.g., joint angles, arm/limb geometry, lighting, background, and camera configuration) and the objective is to disentangle and recover the controllable latent variables, i.e., those that can be directly manipulated (intervened upon) through actuation. Interventional CRL theory shows that variables that undergo variations via interventions can be identified. In robotics, such interventions arise naturally by commanding actuators of various joints and recording images under varied controls. Empirical evaluations in semi-synthetic manipulator experiments demonstrate that ROPES successfully disentangles latent generative factors with high fidelity with respect to the ground truth. Crucially, this is achieved by leveraging only distributional changes, without using any labeled data. The paper also includes a comparison with a baseline based on a recently proposed semi-supervised framework. This paper concludes by positioning robot pose estimation as a near-practical testbed for CRL.
>
---
#### [new 005] AURASeg: Attention Guided Upsampling with Residual Boundary-Assistive Refinement for Drivable-Area Segmentation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对自动驾驶与机器人导航中的可行驶区域分割任务，解决现有模型在边界精度与多尺度特征融合上的不足。提出AURASeg模型，结合残差边界精修模块与注意力引导的渐进上采样解码器，提升边缘识别准确率与整体分割性能，在保持实时性的同时显著优化mIoU与F1指标。**

- **链接: [http://arxiv.org/pdf/2510.21536v1](http://arxiv.org/pdf/2510.21536v1)**

> **作者:** Narendhiran Vijayakumar; Sridevi. M
>
> **备注:** 10 pages, 5 figures, 4 tables
>
> **摘要:** Free space ground segmentation is essential to navigate robots and autonomous vehicles, recognize drivable zones, and traverse efficiently. Fine-grained features remain challenging for existing segmentation models, particularly for robots in indoor and structured environments. These difficulties arise from ineffective multi-scale processing, suboptimal boundary refinement, and limited feature representation. In order to overcome these limitations, we propose Attention-Guided Upsampling with Residual Boundary-Assistive Refinement (AURASeg), a ground-plane semantic segmentation model that maintains high segmentation accuracy while improving border precision. Our method uses CSP-Darknet backbone by adding a Residual Border Refinement Module (RBRM) for accurate edge delineation and an Attention Progressive Upsampling Decoder (APUD) for strong feature integration. We also incorporate a lightweight Atrous Spatial Pyramid Pooling (ASPP-Lite) module to ensure multi-scale context extraction without compromising real-time performance. The proposed model beats benchmark segmentation architectures in mIoU and F1 metrics when tested on the Ground Mobile Robot Perception (GMRP) Dataset and a custom Gazebo indoor dataset. Our approach achieves an improvement in mean Intersection-over-Union (mIoU) of +1.26% and segmentation precision of +1.65% compared to state-of-the-art models. These results show that our technique is feasible for autonomous perception in both indoor and outdoor environments, enabling precise border refinement with minimal effect on inference speed.
>
---
#### [new 006] Design and Structural Validation of a Micro-UAV with On-Board Dynamic Route Planning
- **分类: cs.RO; cs.CE**

- **简介: 该论文针对微小型无人机在搜救任务中结构脆弱与路径无法动态重规划的问题，设计了一种全自研、低成本、模块化无人机。通过强化结构与开源软件实现抗冲击能力与实时动态路径规划，无需昂贵硬件，提升其在复杂环境中的实用性和可靠性。**

- **链接: [http://arxiv.org/pdf/2510.21648v1](http://arxiv.org/pdf/2510.21648v1)**

> **作者:** Inbazhagan Ravikumar; Ram Sundhar; Narendhiran Vijayakumar
>
> **备注:** 8 pages, 4 figures, 4 tables
>
> **摘要:** Micro aerial vehicles are becoming increasingly important in search and rescue operations due to their agility, speed, and ability to access confined spaces or hazardous areas. However, designing lightweight aerial systems presents significant structural, aerodynamic, and computational challenges. This work addresses two key limitations in many low-cost aerial systems under two kilograms: their lack of structural durability during flight through rough terrains and inability to replan paths dynamically when new victims or obstacles are detected. We present a fully customised drone built from scratch using only commonly available components and materials, emphasising modularity, low cost, and ease of assembly. The structural frame is reinforced with lightweight yet durable materials to withstand impact, while the onboard control system is powered entirely by free, open-source software solutions. The proposed system demonstrates real-time perception and adaptive navigation capabilities without relying on expensive hardware accelerators, offering an affordable and practical solution for real-world search and rescue missions.
>
---
#### [new 007] Robust Point Cloud Reinforcement Learning via PCA-Based Canonicalization
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于点云强化学习任务，旨在解决机器人控制中因相机姿态变化导致的视点敏感问题。提出基于主成分分析的点云规范化方法（PPC），将任意刚体变换下的点云映射到统一规范姿态，减少视点差异，提升模型对未见视角的鲁棒性，优于领域随机化。**

- **链接: [http://arxiv.org/pdf/2510.20974v1](http://arxiv.org/pdf/2510.20974v1)**

> **作者:** Michael Bezick; Vittorio Giammarino; Ahmed H. Qureshi
>
> **摘要:** Reinforcement Learning (RL) from raw visual input has achieved impressive successes in recent years, yet it remains fragile to out-of-distribution variations such as changes in lighting, color, and viewpoint. Point Cloud Reinforcement Learning (PC-RL) offers a promising alternative by mitigating appearance-based brittleness, but its sensitivity to camera pose mismatches continues to undermine reliability in realistic settings. To address this challenge, we propose PCA Point Cloud (PPC), a canonicalization framework specifically tailored for downstream robotic control. PPC maps point clouds under arbitrary rigid-body transformations to a unique canonical pose, aligning observations to a consistent frame, thereby substantially decreasing viewpoint-induced inconsistencies. In our experiments, we show that PPC improves robustness to unseen camera poses across challenging robotic tasks, providing a principled alternative to domain randomization.
>
---
#### [new 008] Revisiting Replanning from Scratch: Real-Time Incremental Planning with Fast Almost-Surely Asymptotically Optimal Planners
- **分类: cs.RO**

- **简介: 该论文研究机器人实时路径规划任务，针对动态环境中需快速重规划的问题。提出无需重用旧计划，而是通过快速几乎必然渐近最优（ASAO）算法独立求解增量问题，实现高效高质路径规划。在仿真与真实机械臂实验中验证了方法优于现有反应式规划算法。**

- **链接: [http://arxiv.org/pdf/2510.21074v1](http://arxiv.org/pdf/2510.21074v1)**

> **作者:** Mitchell E. C. Sabbadini; Andrew H. Liu; Joseph Ruan; Tyler S. Wilson; Zachary Kingston; Jonathan D. Gammell
>
> **备注:** Submitted to IEEE International Conference on Robotics and Automation (ICRA) 2026, 8 pages, 5 figures, 1 table. A video of this work can be found at https://www.youtube.com/watch?v=XaZrFy8wGZs
>
> **摘要:** Robots operating in changing environments either predict obstacle changes and/or plan quickly enough to react to them. Predictive approaches require a strong prior about the position and motion of obstacles. Reactive approaches require no assumptions about their environment but must replan quickly and find high-quality paths to navigate effectively. Reactive approaches often reuse information between queries to reduce planning cost. These techniques are conceptually sound but updating dense planning graphs when information changes can be computationally prohibitive. It can also require significant effort to detect the changes in some applications. This paper revisits the long-held assumption that reactive replanning requires updating existing plans. It shows that the incremental planning problem can alternatively be solved more efficiently as a series of independent problems using fast almost-surely asymptotically optimal (ASAO) planning algorithms. These ASAO algorithms quickly find an initial solution and converge towards an optimal solution which allows them to find consistent global plans in the presence of changing obstacles without requiring explicit plan reuse. This is demonstrated with simulated experiments where Effort Informed Trees (EIT*) finds shorter median solution paths than the tested reactive planning algorithms and is further validated using Asymptotically Optimal RRT-Connect (AORRTC) on a real-world planning problem on a robot arm.
>
---
#### [new 009] An Agnostic End-Effector Alignment Controller for Robust Assembly of Modular Space Robots
- **分类: cs.RO**

- **简介: 该论文针对月球环境下模块化航天机器人自主装配任务，解决因机械误差与传感噪声导致的对接不稳问题。提出一种无机器人依赖的末端执行器对齐控制器，通过动态超球面约束自适应调整运动速度，实现平稳、精确的对齐。在真实硬件上验证了其对不同缺陷和噪声的鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.21164v1](http://arxiv.org/pdf/2510.21164v1)**

> **作者:** Shamistan Karimov; Elian Neppel; Shreya Santra; Kentaro Uno; Kazuya Yoshida
>
> **备注:** 6 pages, 12 figures. Accepted at iSparo 2025 | Video: https://youtu.be/BW0YgSrvuDo
>
> **摘要:** Modular robots offer reconfigurability and fault tolerance essential for lunar missions, but require controllers that adapt safely to real-world disturbances. We build on our previous hardware-agnostic actuator synchronization in Motion Stack to develop a new controller enforcing adaptive velocity bounds via a dynamic hypersphere clamp. Using only real-time end-effector and target pose measurements, the controller adjusts its translational and rotational speed limits to ensure smooth, stable alignment without abrupt motions. We implemented two variants, a discrete, step-based version and a continuous, velocity-based version, and tested them on two MoonBot limbs in JAXA's lunar environment simulator. Field trials demonstrate that the step-based variant produces highly predictable, low-wobble motions, while the continuous variant converges more quickly and maintains millimeter-level positional accuracy, and both remain robust across limbs with differing mechanical imperfections and sensing noise (e.g., backlash and flex). These results highlight the flexibility and robustness of our robot-agnostic framework for autonomous self-assembly and reconfiguration under harsh conditions.
>
---
#### [new 010] Sequentially Teaching Sequential Tasks $(ST)^2$: Teaching Robots Long-horizon Manipulation Skills
- **分类: cs.RO**

- **简介: 该论文研究机器人长时程操作技能的教学问题，针对传统一次性示范导致误差累积和教师疲劳的问题，提出$(ST)^2$顺序教学方法。通过用户实验对比单体与分步示范，验证了顺序教学在可控性上的优势，实现了高效、结构化的技能传授。**

- **链接: [http://arxiv.org/pdf/2510.21046v1](http://arxiv.org/pdf/2510.21046v1)**

> **作者:** Zlatan Ajanović; Ravi Prakash; Leandro de Souza Rosa; Jens Kober
>
> **摘要:** Learning from demonstration is effective for teaching robots complex skills with high sample efficiency. However, teaching long-horizon tasks with multiple skills is difficult, as deviations accumulate, distributional shift increases, and human teachers become fatigued, raising the chance of failure. In this work, we study user responses to two teaching frameworks: (i) a traditional monolithic approach, where users demonstrate the entire trajectory of a long-horizon task; and (ii) a sequential approach, where the task is segmented by the user and demonstrations are provided step by step. To support this study, we introduce $(ST)^2$, a sequential method for learning long-horizon manipulation tasks that allows users to control the teaching flow by defining key points, enabling incremental and structured demonstrations. We conducted a user study on a restocking task with 16 participants in a realistic retail environment to evaluate both user preference and method effectiveness. Our objective and subjective results show that both methods achieve similar trajectory quality and success rates. Some participants preferred the sequential approach for its iterative control, while others favored the monolithic approach for its simplicity.
>
---
#### [new 011] SutureBot: A Precision Framework & Benchmark For Autonomous End-to-End Suturing
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出SutureBot，一个面向达芬奇手术机器人平台的自主缝合基准。针对长期精细操作中缝合全流程自动化难题，构建了包含1890次示范的高保真数据集，设计目标条件框架提升穿刺精度，并评估多种视觉-语言-动作模型，推动手术机器人自主化发展。**

- **链接: [http://arxiv.org/pdf/2510.20965v1](http://arxiv.org/pdf/2510.20965v1)**

> **作者:** Jesse Haworth; Juo-Tung Chen; Nigel Nelson; Ji Woong Kim; Masoud Moghani; Chelsea Finn; Axel Krieger
>
> **备注:** 10 pages, 5 figures, 4 tables, NeurIPS 2025
>
> **摘要:** Robotic suturing is a prototypical long-horizon dexterous manipulation task, requiring coordinated needle grasping, precise tissue penetration, and secure knot tying. Despite numerous efforts toward end-to-end autonomy, a fully autonomous suturing pipeline has yet to be demonstrated on physical hardware. We introduce SutureBot: an autonomous suturing benchmark on the da Vinci Research Kit (dVRK), spanning needle pickup, tissue insertion, and knot tying. To ensure repeatability, we release a high-fidelity dataset comprising 1,890 suturing demonstrations. Furthermore, we propose a goal-conditioned framework that explicitly optimizes insertion-point precision, improving targeting accuracy by 59\%-74\% over a task-only baseline. To establish this task as a benchmark for dexterous imitation learning, we evaluate state-of-the-art vision-language-action (VLA) models, including $\pi_0$, GR00T N1, OpenVLA-OFT, and multitask ACT, each augmented with a high-level task-prediction policy. Autonomous suturing is a key milestone toward achieving robotic autonomy in surgery. These contributions support reproducible evaluation and development of precision-focused, long-horizon dexterous manipulation policies necessary for end-to-end suturing. Dataset is available at: https://huggingface.co/datasets/jchen396/suturebot
>
---
#### [new 012] Enhancing Tactile-based Reinforcement Learning for Robotic Control
- **分类: cs.RO; cs.LG**

- **简介: 该论文聚焦于基于触觉的机器人操控任务，旨在解决强化学习中触觉感知效能不稳的问题。通过自监督学习提升稀疏二值触觉信号利用效率，验证其对精细操作的关键作用，并提出分离记忆结构以优化性能。研究发布RoTO基准，推动该领域发展。**

- **链接: [http://arxiv.org/pdf/2510.21609v1](http://arxiv.org/pdf/2510.21609v1)**

> **作者:** Elle Miller; Trevor McInroe; David Abel; Oisin Mac Aodha; Sethu Vijayakumar
>
> **摘要:** Achieving safe, reliable real-world robotic manipulation requires agents to evolve beyond vision and incorporate tactile sensing to overcome sensory deficits and reliance on idealised state information. Despite its potential, the efficacy of tactile sensing in reinforcement learning (RL) remains inconsistent. We address this by developing self-supervised learning (SSL) methodologies to more effectively harness tactile observations, focusing on a scalable setup of proprioception and sparse binary contacts. We empirically demonstrate that sparse binary tactile signals are critical for dexterity, particularly for interactions that proprioceptive control errors do not register, such as decoupled robot-object motions. Our agents achieve superhuman dexterity in complex contact tasks (ball bouncing and Baoding ball rotation). Furthermore, we find that decoupling the SSL memory from the on-policy memory can improve performance. We release the Robot Tactile Olympiad (RoTO) benchmark to standardise and promote future research in tactile-based manipulation. Project page: https://elle-miller.github.io/tactile_rl
>
---
#### [new 013] Load-bearing Assessment for Safe Locomotion of Quadruped Robots on Collapsing Terrain
- **分类: cs.RO**

- **简介: 该论文针对四足机器人在坍塌地形上的安全行走问题，提出一种融合地形探测、承载力分析与模型预测控制的运动框架。通过关节传感数据评估地形稳定性，动态调整步态与落脚点，实现无需额外传感器的安全导航。实验验证了方法在复杂不稳定地形中的有效性。**

- **链接: [http://arxiv.org/pdf/2510.21369v1](http://arxiv.org/pdf/2510.21369v1)**

> **作者:** Vivian S. Medeiros; Giovanni B. Dessy; Thiago Boaventura; Marcelo Becker; Claudio Semini; Victor Barasuol
>
> **摘要:** Collapsing terrains, often present in search and rescue missions or planetary exploration, pose significant challenges for quadruped robots. This paper introduces a robust locomotion framework for safe navigation over unstable surfaces by integrating terrain probing, load-bearing analysis, motion planning, and control strategies. Unlike traditional methods that rely on specialized sensors or external terrain mapping alone, our approach leverages joint measurements to assess terrain stability without hardware modifications. A Model Predictive Control (MPC) system optimizes robot motion, balancing stability and probing constraints, while a state machine coordinates terrain probing actions, enabling the robot to detect collapsible regions and dynamically adjust its footholds. Experimental results on custom-made collapsing platforms and rocky terrains demonstrate the framework's ability to traverse collapsing terrain while maintaining stability and prioritizing safety.
>
---
#### [new 014] Remote Autonomy for Multiple Small Lowcost UAVs in GNSS-denied Search and Rescue Operations
- **分类: cs.RO**

- **简介: 该论文针对GNSS拒止环境下多架低成本无人机的自主搜救任务，解决手动操作依赖人力、自动化系统成本高的问题。提出基于安卓遥控器运行状态估计与避障的轻量级自主飞行系统，支持单操作员协同控制异构无人机，并融合多机观测构建统一3D环境模型，提升搜救效率与态势感知能力。**

- **链接: [http://arxiv.org/pdf/2510.21357v1](http://arxiv.org/pdf/2510.21357v1)**

> **作者:** Daniel Schleich; Jan Quenzel; Sven Behnke
>
> **备注:** Accepted final version. IEEE International Symposium on Safety, Security, and Rescue Robotics (SSRR), Galway, Ireland, 2025
>
> **摘要:** In recent years, consumer-grade UAVs have been widely adopted by first responders. In general, they are operated manually, which requires trained pilots, especially in unknown GNSS-denied environments and in the vicinity of structures. Autonomous flight can facilitate the application of UAVs and reduce operator strain. However, autonomous systems usually require special programming interfaces, custom sensor setups, and strong onboard computers, which limits a broader deployment. We present a system for autonomous flight using lightweight consumer-grade DJI drones. They are controlled by an Android app for state estimation and obstacle avoidance directly running on the UAV's remote control. Our ground control station enables a single operator to configure and supervise multiple heterogeneous UAVs at once. Furthermore, it combines the observations of all UAVs into a joint 3D environment model for improved situational awareness.
>
---
#### [new 015] Aircraft Collision Avoidance Systems: Technological Challenges and Solutions on the Path to Regulatory Acceptance
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 该论文聚焦航空器防撞系统，旨在解决监测、决策与验证中的技术难题。通过综述已获监管机构认可的成熟方案，为安全关键系统提供可借鉴的实践路径。**

- **链接: [http://arxiv.org/pdf/2510.20916v1](http://arxiv.org/pdf/2510.20916v1)**

> **作者:** Sydney M. Katz; Robert J. Moss; Dylan M. Asmar; Wesley A. Olson; James K. Kuchar; Mykel J. Kochenderfer
>
> **备注:** 32 pages, 9 figures
>
> **摘要:** Aircraft collision avoidance systems is critical to modern aviation. These systems are designed to predict potential collisions between aircraft and recommend appropriate avoidance actions. Creating effective collision avoidance systems requires solutions to a variety of technical challenges related to surveillance, decision making, and validation. These challenges have sparked significant research and development efforts over the past several decades that have resulted in a variety of proposed solutions. This article provides an overview of these challenges and solutions with an emphasis on those that have been put through a rigorous validation process and accepted by regulatory bodies. The challenges posed by the collision avoidance problem are often present in other domains, and aircraft collision avoidance systems can serve as case studies that provide valuable insights for a wide range of safety-critical systems.
>
---
#### [new 016] Generalizable Hierarchical Skill Learning via Object-Centric Representation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出通用分层技能学习框架GSL，用于机器人操作任务。针对政策泛化与样本效率问题，利用物体中心技能作为视觉语言模型与低层视觉-运动策略的接口，通过基础模型分解演示并生成可迁移的物体规范技能，实现高效学习与跨场景泛化，在仿真和真实世界实验中均显著优于基线方法。**

- **链接: [http://arxiv.org/pdf/2510.21121v1](http://arxiv.org/pdf/2510.21121v1)**

> **作者:** Haibo Zhao; Yu Qi; Boce Hu; Yizhe Zhu; Ziyan Chen; Heng Tian; Xupeng Zhu; Owen Howell; Haojie Huang; Robin Walters; Dian Wang; Robert Platt
>
> **摘要:** We present Generalizable Hierarchical Skill Learning (GSL), a novel framework for hierarchical policy learning that significantly improves policy generalization and sample efficiency in robot manipulation. One core idea of GSL is to use object-centric skills as an interface that bridges the high-level vision-language model and the low-level visual-motor policy. Specifically, GSL decomposes demonstrations into transferable and object-canonicalized skill primitives using foundation models, ensuring efficient low-level skill learning in the object frame. At test time, the skill-object pairs predicted by the high-level agent are fed to the low-level module, where the inferred canonical actions are mapped back to the world frame for execution. This structured yet flexible design leads to substantial improvements in sample efficiency and generalization of our method across unseen spatial arrangements, object appearances, and task compositions. In simulation, GSL trained with only 3 demonstrations per task outperforms baselines trained with 30 times more data by 15.5 percent on unseen tasks. In real-world experiments, GSL also surpasses the baseline trained with 10 times more data.
>
---
#### [new 017] Scalable Vision-Language-Action Model Pretraining for Robotic Manipulation with Real-Life Human Activity Videos
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文面向机器人灵巧操作任务，提出基于真实人类活动视频的视觉-语言-动作（VLA）模型预训练方法。通过自动化分析无标注的视角视频，生成带3D手部运动与语言描述的高质量数据，构建100万条任务数据集。模型在零样本和微调后均表现出色，显著提升机器人对新物体的泛化能力，验证了大规模预训练的有效性。**

- **链接: [http://arxiv.org/pdf/2510.21571v1](http://arxiv.org/pdf/2510.21571v1)**

> **作者:** Qixiu Li; Yu Deng; Yaobo Liang; Lin Luo; Lei Zhou; Chengtang Yao; Lingqi Zeng; Zhiyuan Feng; Huizhi Liang; Sicheng Xu; Yizhong Zhang; Xi Chen; Hao Chen; Lily Sun; Dong Chen; Jiaolong Yang; Baining Guo
>
> **备注:** Project page: https://microsoft.github.io/VITRA/
>
> **摘要:** This paper presents a novel approach for pretraining robotic manipulation Vision-Language-Action (VLA) models using a large corpus of unscripted real-life video recordings of human hand activities. Treating human hand as dexterous robot end-effector, we show that "in-the-wild" egocentric human videos without any annotations can be transformed into data formats fully aligned with existing robotic V-L-A training data in terms of task granularity and labels. This is achieved by the development of a fully-automated holistic human activity analysis approach for arbitrary human hand videos. This approach can generate atomic-level hand activity segments and their language descriptions, each accompanied with framewise 3D hand motion and camera motion. We process a large volume of egocentric videos and create a hand-VLA training dataset containing 1M episodes and 26M frames. This training data covers a wide range of objects and concepts, dexterous manipulation tasks, and environment variations in real life, vastly exceeding the coverage of existing robot data. We design a dexterous hand VLA model architecture and pretrain the model on this dataset. The model exhibits strong zero-shot capabilities on completely unseen real-world observations. Additionally, fine-tuning it on a small amount of real robot action data significantly improves task success rates and generalization to novel objects in real robotic experiments. We also demonstrate the appealing scaling behavior of the model's task performance with respect to pretraining data scale. We believe this work lays a solid foundation for scalable VLA pretraining, advancing robots toward truly generalizable embodied intelligence.
>
---
#### [new 018] PREVENT: Proactive Risk Evaluation and Vigilant Execution of Tasks for Mobile Robotic Chemists using Multi-Modal Behavior Trees
- **分类: cs.RO**

- **简介: 该论文针对移动机器人在化学实验中缺乏流程感知能力的问题，提出PREVENT系统，基于多模态行为树实现主动风险评估与警觉执行。通过融合视觉与气体传感器数据，提升感知准确性，有效避免误报与漏报，保障实验安全高效运行。**

- **链接: [http://arxiv.org/pdf/2510.21438v1](http://arxiv.org/pdf/2510.21438v1)**

> **作者:** Satheeshkumar Veeramani; Zhengxue Zhou; Francisco Munguia-Galeano; Hatem Fakhruldeen; Thomas Roddelkopf; Mohammed Faeik Ruzaij Al-Okby; Kerstin Thurow; Andrew Ian Cooper
>
> **备注:** 25 pages, 8 figures, paper submitted to Robotics and Autonomous Systems Journal
>
> **摘要:** Mobile robotic chemists are a fast growing trend in the field of chemistry and materials research. However, so far these mobile robots lack workflow awareness skills. This poses the risk that even a small anomaly, such as an improperly capped sample vial could disrupt the entire workflow. This wastes time, and resources, and could pose risks to human researchers, such as exposure to toxic materials. Existing perception mechanisms can be used to predict anomalies but they often generate excessive false positives. This may halt workflow execution unnecessarily, requiring researchers to intervene and to resume the workflow when no problem actually exists, negating the benefits of autonomous operation. To address this problem, we propose PREVENT a system comprising navigation and manipulation skills based on a multimodal Behavior Tree (BT) approach that can be integrated into existing software architectures with minimal modifications. Our approach involves a hierarchical perception mechanism that exploits AI techniques and sensory feedback through Dexterous Vision and Navigational Vision cameras and an IoT gas sensor module for execution-related decision-making. Experimental evaluations show that the proposed approach is comparatively efficient and completely avoids both false negatives and false positives when tested in simulated risk scenarios within our robotic chemistry workflow. The results also show that the proposed multi-modal perception skills achieved deployment accuracies that were higher than the average of the corresponding uni-modal skills, both for navigation and for manipulation.
>
---
#### [new 019] Safety Assessment in Reinforcement Learning via Model Predictive Control
- **分类: cs.LG; cs.RO**

- **简介: 该论文针对强化学习中的安全问题，提出基于模型预测路径积分控制的方法，利用系统可逆性在训练中实时检测并阻止不安全动作。无需显式安全约束或动态模型，仅需黑箱查询，即可保障安全性，同时保持与基线方法相当的学习效果。**

- **链接: [http://arxiv.org/pdf/2510.20955v1](http://arxiv.org/pdf/2510.20955v1)**

> **作者:** Jeff Pflueger; Michael Everett
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Model-free reinforcement learning approaches are promising for control but typically lack formal safety guarantees. Existing methods to shield or otherwise provide these guarantees often rely on detailed knowledge of the safety specifications. Instead, this work's insight is that many difficult-to-specify safety issues are best characterized by invariance. Accordingly, we propose to leverage reversibility as a method for preventing these safety issues throughout the training process. Our method uses model-predictive path integral control to check the safety of an action proposed by a learned policy throughout training. A key advantage of this approach is that it only requires the ability to query the black-box dynamics, not explicit knowledge of the dynamics or safety constraints. Experimental results demonstrate that the proposed algorithm successfully aborts before all unsafe actions, while still achieving comparable training progress to a baseline PPO approach that is allowed to violate safety.
>
---
#### [new 020] PhysWorld: From Real Videos to World Models of Deformable Objects via Physics-Aware Demonstration Synthesis
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出PhysWorld框架，旨在从有限真实视频中学习可变形物体的物理一致动态模型。针对数据稀缺与物理一致性难题，利用物理模拟器构建数字孪生，通过局部属性扰动生成多样化演示，训练轻量级图神经网络世界模型，并用真实视频优化物理参数，实现高效精准的未来预测与新交互泛化。**

- **链接: [http://arxiv.org/pdf/2510.21447v1](http://arxiv.org/pdf/2510.21447v1)**

> **作者:** Yu Yang; Zhilu Zhang; Xiang Zhang; Yihan Zeng; Hui Li; Wangmeng Zuo
>
> **备注:** 17 pages, 5 figures
>
> **摘要:** Interactive world models that simulate object dynamics are crucial for robotics, VR, and AR. However, it remains a significant challenge to learn physics-consistent dynamics models from limited real-world video data, especially for deformable objects with spatially-varying physical properties. To overcome the challenge of data scarcity, we propose PhysWorld, a novel framework that utilizes a simulator to synthesize physically plausible and diverse demonstrations to learn efficient world models. Specifically, we first construct a physics-consistent digital twin within MPM simulator via constitutive model selection and global-to-local optimization of physical properties. Subsequently, we apply part-aware perturbations to the physical properties and generate various motion patterns for the digital twin, synthesizing extensive and diverse demonstrations. Finally, using these demonstrations, we train a lightweight GNN-based world model that is embedded with physical properties. The real video can be used to further refine the physical properties. PhysWorld achieves accurate and fast future predictions for various deformable objects, and also generalizes well to novel interactions. Experiments show that PhysWorld has competitive performance while enabling inference speeds 47 times faster than the recent state-of-the-art method, i.e., PhysTwin.
>
---
#### [new 021] ESCORT: Efficient Stein-variational and Sliced Consistency-Optimized Temporal Belief Representation for POMDPs
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文针对部分可观测马尔可夫决策过程（POMDP）中的信念表示问题，提出ESCORT框架。针对复杂多模态高维信念分布难以准确建模的问题，通过改进Stein变分梯度下降，引入相关性感知投影与时间一致性约束，实现高效、稳定的粒子表示，显著提升信念精度与决策质量。**

- **链接: [http://arxiv.org/pdf/2510.21107v1](http://arxiv.org/pdf/2510.21107v1)**

> **作者:** Yunuo Zhang; Baiting Luo; Ayan Mukhopadhyay; Gabor Karsai; Abhishek Dubey
>
> **备注:** Proceeding of the 39th Conference on Neural Information Processing Systems (NeurIPS'25). Code would be available at https://github.com/scope-lab-vu/ESCORT
>
> **摘要:** In Partially Observable Markov Decision Processes (POMDPs), maintaining and updating belief distributions over possible underlying states provides a principled way to summarize action-observation history for effective decision-making under uncertainty. As environments grow more realistic, belief distributions develop complexity that standard mathematical models cannot accurately capture, creating a fundamental challenge in maintaining representational accuracy. Despite advances in deep learning and probabilistic modeling, existing POMDP belief approximation methods fail to accurately represent complex uncertainty structures such as high-dimensional, multi-modal belief distributions, resulting in estimation errors that lead to suboptimal agent behaviors. To address this challenge, we present ESCORT (Efficient Stein-variational and sliced Consistency-Optimized Representation for Temporal beliefs), a particle-based framework for capturing complex, multi-modal distributions in high-dimensional belief spaces. ESCORT extends SVGD with two key innovations: correlation-aware projections that model dependencies between state dimensions, and temporal consistency constraints that stabilize updates while preserving correlation structures. This approach retains SVGD's attractive-repulsive particle dynamics while enabling accurate modeling of intricate correlation patterns. Unlike particle filters prone to degeneracy or parametric methods with fixed representational capacity, ESCORT dynamically adapts to belief landscape complexity without resampling or restrictive distributional assumptions. We demonstrate ESCORT's effectiveness through extensive evaluations on both POMDP domains and synthetic multi-modal distributions of varying dimensionality, where it consistently outperforms state-of-the-art methods in terms of belief approximation accuracy and downstream decision quality.
>
---
#### [new 022] Learning Neural Control Barrier Functions from Expert Demonstrations using Inverse Constraint Learning
- **分类: cs.AI; cs.RO**

- **简介: 该论文针对自主系统安全控制中难以明确定义故障集的问题，提出基于逆约束学习（ICL）从专家演示中学习神经控制屏障函数（CBF）。通过ICL识别安全与不安全状态，利用模拟轨迹训练神经CBF，实现高效安全控制。实验证明其性能优于基线方法，接近使用真实标签的模型。**

- **链接: [http://arxiv.org/pdf/2510.21560v1](http://arxiv.org/pdf/2510.21560v1)**

> **作者:** Yuxuan Yang; Hussein Sibai
>
> **摘要:** Safety is a fundamental requirement for autonomous systems operating in critical domains. Control barrier functions (CBFs) have been used to design safety filters that minimally alter nominal controls for such systems to maintain their safety. Learning neural CBFs has been proposed as a data-driven alternative for their computationally expensive optimization-based synthesis. However, it is often the case that the failure set of states that should be avoided is non-obvious or hard to specify formally, e.g., tailgating in autonomous driving, while a set of expert demonstrations that achieve the task and avoid the failure set is easier to generate. We use ICL to train a constraint function that classifies the states of the system under consideration to safe, i.e., belong to a controlled forward invariant set that is disjoint from the unspecified failure set, and unsafe ones, i.e., belong to the complement of that set. We then use that function to label a new set of simulated trajectories to train our neural CBF. We empirically evaluate our approach in four different environments, demonstrating that it outperforms existing baselines and achieves comparable performance to a neural CBF trained with the same data but annotated with ground-truth safety labels.
>
---
#### [new 023] MATrack: Efficient Multiscale Adaptive Tracker for Real-Time Nighttime UAV Operations
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对夜间无人机跟踪任务，解决低光下视觉退化、背景杂乱和视角变化导致的跟踪漂移问题。提出MATrack系统，通过多尺度融合、自适应关键令牌门和夜间模板校准三模块协同，提升特征一致性与跟踪稳定性，在UAVDark135上显著优于SOTA方法，实现81 FPS实时性能。**

- **链接: [http://arxiv.org/pdf/2510.21586v1](http://arxiv.org/pdf/2510.21586v1)**

> **作者:** Xuzhao Li; Xuchen Li; Shiyu Hu
>
> **备注:** Preprint, Under Review
>
> **摘要:** Nighttime UAV tracking faces significant challenges in real-world robotics operations. Low-light conditions not only limit visual perception capabilities, but cluttered backgrounds and frequent viewpoint changes also cause existing trackers to drift or fail during deployment. To address these difficulties, researchers have proposed solutions based on low-light enhancement and domain adaptation. However, these methods still have notable shortcomings in actual UAV systems: low-light enhancement often introduces visual artifacts, domain adaptation methods are computationally expensive and existing lightweight designs struggle to fully leverage dynamic object information. Based on an in-depth analysis of these key issues, we propose MATrack-a multiscale adaptive system designed specifically for nighttime UAV tracking. MATrack tackles the main technical challenges of nighttime tracking through the collaborative work of three core modules: Multiscale Hierarchy Blende (MHB) enhances feature consistency between static and dynamic templates. Adaptive Key Token Gate accurately identifies object information within complex backgrounds. Nighttime Template Calibrator (NTC) ensures stable tracking performance over long sequences. Extensive experiments show that MATrack achieves a significant performance improvement. On the UAVDark135 benchmark, its precision, normalized precision and AUC surpass state-of-the-art (SOTA) methods by 5.9%, 5.4% and 4.2% respectively, while maintaining a real-time processing speed of 81 FPS. Further tests on a real-world UAV platform validate the system's reliability, demonstrating that MATrack can provide stable and effective nighttime UAV tracking support for critical robotics applications such as nighttime search and rescue and border patrol.
>
---
#### [new 024] ZING-3D: Zero-shot Incremental 3D Scene Graphs via Vision-Language Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出ZING-3D框架，解决3D场景图生成中缺乏零样本识别、增量更新与3D几何接地的问题。通过视觉语言模型生成2D语义场景图，并结合深度信息实现3D空间对齐，支持开放词汇、增量更新与空间关系建模，适用于机器人等具身应用。**

- **链接: [http://arxiv.org/pdf/2510.21069v1](http://arxiv.org/pdf/2510.21069v1)**

> **作者:** Pranav Saxena; Jimmy Chiun
>
> **摘要:** Understanding and reasoning about complex 3D environments requires structured scene representations that capture not only objects but also their semantic and spatial relationships. While recent works on 3D scene graph generation have leveraged pretrained VLMs without task-specific fine-tuning, they are largely confined to single-view settings, fail to support incremental updates as new observations arrive and lack explicit geometric grounding in 3D space, all of which are essential for embodied scenarios. In this paper, we propose, ZING-3D, a framework that leverages the vast knowledge of pretrained foundation models to enable open-vocabulary recognition and generate a rich semantic representation of the scene in a zero-shot manner while also enabling incremental updates and geometric grounding in 3D space, making it suitable for downstream robotics applications. Our approach leverages VLM reasoning to generate a rich 2D scene graph, which is grounded in 3D using depth information. Nodes represent open-vocabulary objects with features, 3D locations, and semantic context, while edges capture spatial and semantic relations with inter-object distances. Our experiments on scenes from the Replica and HM3D dataset show that ZING-3D is effective at capturing spatial and relational knowledge without the need of task-specific training.
>
---
#### [new 025] Track-to-Track Association for Collective Perception based on Stochastic Optimization
- **分类: eess.SP; cs.RO**

- **简介: 该论文针对智能城市中自动驾驶的集体感知任务，解决多车传感器融合中的轨迹关联问题。提出基于随机优化的关联算法，融合轨迹数量与空间分布信息，高效生成高概率关联假设，显著提升复杂场景下的关联准确性与计算效率。**

- **链接: [http://arxiv.org/pdf/2510.21278v1](http://arxiv.org/pdf/2510.21278v1)**

> **作者:** Laura M. Wolf; Vincent Albert Wolff; Simon Steuernagel; Kolja Thormann; Marcus Baum
>
> **摘要:** Collective perception is a key aspect for autonomous driving in smart cities as it aims to combine the local environment models of multiple intelligent vehicles in order to overcome sensor limitations. A crucial part of multi-sensor fusion is track-to-track association. Previous works often suffer from high computational complexity or are based on heuristics. We propose an association algorithms based on stochastic optimization, which leverages a multidimensional likelihood incorporating the number of tracks and their spatial distribution and furthermore computes several association hypotheses. We demonstrate the effectiveness of our approach in Monte Carlo simulations and a realistic collective perception scenario computing high-likelihood associations in ambiguous settings.
>
---
#### [new 026] Towards Reliable Code-as-Policies: A Neuro-Symbolic Framework for Embodied Task Planning
- **分类: cs.AI; cs.RO**

- **简介: 该论文针对机器人任务规划中代码即策略（Code-as-Policies）因环境感知不足导致的可靠性问题，提出一种神经符号框架。通过引入符号验证与交互式验证机制，生成能主动探索环境以获取缺失信息的代码，增强代码与环境的对齐性，显著提升动态、部分可观测场景下的任务成功率与动作可执行性。**

- **链接: [http://arxiv.org/pdf/2510.21302v1](http://arxiv.org/pdf/2510.21302v1)**

> **作者:** Sanghyun Ahn; Wonje Choi; Junyong Lee; Jinwoo Park; Honguk Woo
>
> **备注:** Accepted at NeurIPS 2025 Spotlight
>
> **摘要:** Recent advances in large language models (LLMs) have enabled the automatic generation of executable code for task planning and control in embodied agents such as robots, demonstrating the potential of LLM-based embodied intelligence. However, these LLM-based code-as-policies approaches often suffer from limited environmental grounding, particularly in dynamic or partially observable settings, leading to suboptimal task success rates due to incorrect or incomplete code generation. In this work, we propose a neuro-symbolic embodied task planning framework that incorporates explicit symbolic verification and interactive validation processes during code generation. In the validation phase, the framework generates exploratory code that actively interacts with the environment to acquire missing observations while preserving task-relevant states. This integrated process enhances the grounding of generated code, resulting in improved task reliability and success rates in complex environments. We evaluate our framework on RLBench and in real-world settings across dynamic, partially observable scenarios. Experimental results demonstrate that our framework improves task success rates by 46.2% over Code-as-Policies baselines and attains over 86.8% executability of task-relevant actions, thereby enhancing the reliability of task planning in dynamic environments.
>
---
#### [new 027] An Experimental Study of Trojan Vulnerabilities in UAV Autonomous Landing
- **分类: cs.CR; cs.AI; cs.CV; cs.RO**

- **简介: 该论文研究无人机自主着陆系统中基于深度学习模型的后门攻击漏洞。针对卷积神经网络，通过在训练数据中嵌入隐蔽触发器，导致模型在特定条件下失效。作者构建了定制数据集与评估框架，验证了攻击有效性，并揭示了城市空中交通系统的安全风险，为提升系统鲁棒性提供基础。**

- **链接: [http://arxiv.org/pdf/2510.20932v1](http://arxiv.org/pdf/2510.20932v1)**

> **作者:** Reza Ahmari; Ahmad Mohammadi; Vahid Hemmati; Mohammed Mynuddin; Mahmoud Nabil Mahmoud; Parham Kebria; Abdollah Homaifar; Mehrdad Saif
>
> **备注:** 6 pages
>
> **摘要:** This study investigates the vulnerabilities of autonomous navigation and landing systems in Urban Air Mobility (UAM) vehicles. Specifically, it focuses on Trojan attacks that target deep learning models, such as Convolutional Neural Networks (CNNs). Trojan attacks work by embedding covert triggers within a model's training data. These triggers cause specific failures under certain conditions, while the model continues to perform normally in other situations. We assessed the vulnerability of Urban Autonomous Aerial Vehicles (UAAVs) using the DroNet framework. Our experiments showed a significant drop in accuracy, from 96.4% on clean data to 73.3% on data triggered by Trojan attacks. To conduct this study, we collected a custom dataset and trained models to simulate real-world conditions. We also developed an evaluation framework designed to identify Trojan-infected models. This work demonstrates the potential security risks posed by Trojan attacks and lays the groundwork for future research on enhancing the resilience of UAM systems.
>
---
## 更新

#### [replaced 001] MEReQ: Max-Ent Residual-Q Inverse RL for Sample-Efficient Alignment from Intervention
- **分类: cs.RO; cs.AI; cs.LG; I.2.6; I.2.9**

- **链接: [http://arxiv.org/pdf/2406.16258v4](http://arxiv.org/pdf/2406.16258v4)**

> **作者:** Yuxin Chen; Chen Tang; Jianglan Wei; Chenran Li; Ran Tian; Xiang Zhang; Wei Zhan; Peter Stone; Masayoshi Tomizuka
>
> **摘要:** Aligning robot behavior with human preferences is crucial for deploying embodied AI agents in human-centered environments. A promising solution is interactive imitation learning from human intervention, where a human expert observes the policy's execution and provides interventions as feedback. However, existing methods often fail to utilize the prior policy efficiently to facilitate learning, thus hindering sample efficiency. In this work, we introduce MEReQ (Maximum-Entropy Residual-Q Inverse Reinforcement Learning), designed for sample-efficient alignment from human intervention. Instead of inferring the complete human behavior characteristics, MEReQ infers a residual reward function that captures the discrepancy between the human expert's and the prior policy's underlying reward functions. It then employs Residual Q-Learning (RQL) to align the policy with human preferences using this residual reward function. Extensive evaluations on simulated and real-world tasks demonstrate that MEReQ achieves sample-efficient policy alignment from human intervention.
>
---
#### [replaced 002] Online Intrinsic Rewards for Decision Making Agents from Large Language Model Feedback
- **分类: cs.LG; cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2410.23022v4](http://arxiv.org/pdf/2410.23022v4)**

> **作者:** Qinqing Zheng; Mikael Henaff; Amy Zhang; Aditya Grover; Brandon Amos
>
> **备注:** RLC 2025
>
> **摘要:** Automatically synthesizing dense rewards from natural language descriptions is a promising paradigm in reinforcement learning (RL), with applications to sparse reward problems, open-ended exploration, and hierarchical skill design. Recent works have made promising steps by exploiting the prior knowledge of large language models (LLMs). However, these approaches suffer from important limitations: they are either not scalable to problems requiring billions of environment samples, due to requiring LLM annotations for each observation, or they require a diverse offline dataset, which may not exist or be impossible to collect. In this work, we address these limitations through a combination of algorithmic and systems-level contributions. We propose ONI, a distributed architecture that simultaneously learns an RL policy and an intrinsic reward function using LLM feedback. Our approach annotates the agent's collected experience via an asynchronous LLM server, which is then distilled into an intrinsic reward model. We explore a range of algorithmic choices for reward modeling with varying complexity, including hashing, classification, and ranking models. Our approach achieves state-of-the-art performance across a range of challenging tasks from the NetHack Learning Environment, while removing the need for large offline datasets required by prior work. We make our code available at https://github.com/facebookresearch/oni.
>
---
#### [replaced 003] Intrinsic Goals for Autonomous Agents: Model-Based Exploration in Virtual Zebrafish Predicts Ethological Behavior and Whole-Brain Dynamics
- **分类: q-bio.NC; cs.AI; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.00138v2](http://arxiv.org/pdf/2506.00138v2)**

> **作者:** Reece Keller; Alyn Kirsch; Felix Pei; Xaq Pitkow; Leo Kozachkov; Aran Nayebi
>
> **备注:** 17 pages, 7 figures
>
> **摘要:** Autonomy is a hallmark of animal intelligence, enabling adaptive and intelligent behavior in complex environments without relying on external reward or task structure. Existing reinforcement learning approaches to exploration in reward-free environments, including a class of methods known as model-based intrinsic motivation, exhibit inconsistent exploration patterns and do not converge to an exploratory policy, thus failing to capture robust autonomous behaviors observed in animals. Moreover, systems neuroscience has largely overlooked the neural basis of autonomy, focusing instead on experimental paradigms where animals are motivated by external reward rather than engaging in ethological, naturalistic and task-independent behavior. To bridge these gaps, we introduce a novel model-based intrinsic drive explicitly designed after the principles of autonomous exploration in animals. Our method (3M-Progress) achieves animal-like exploration by tracking divergence between an online world model and a fixed prior learned from an ecological niche. To the best of our knowledge, we introduce the first autonomous embodied agent that predicts brain data entirely from self-supervised optimization of an intrinsic goal -- without any behavioral or neural training data -- demonstrating that 3M-Progress agents capture the explainable variance in behavioral patterns and whole-brain neural-glial dynamics recorded from autonomously behaving larval zebrafish, thereby providing the first goal-driven, population-level model of neural-glial computation. Our findings establish a computational framework connecting model-based intrinsic motivation to naturalistic behavior, providing a foundation for building artificial agents with animal-like autonomy.
>
---
#### [replaced 004] Knot So Simple: A Minimalistic Environment for Spatial Reasoning
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.18028v2](http://arxiv.org/pdf/2505.18028v2)**

> **作者:** Zizhao Chen; Yoav Artzi
>
> **摘要:** We propose KnotGym, an interactive environment for complex, spatial reasoning and manipulation. KnotGym includes goal-oriented rope manipulation tasks with varying levels of complexity, all requiring acting from pure image observations. Tasks are defined along a clear and quantifiable axis of complexity based on the number of knot crossings, creating a natural generalization test. KnotGym has a simple observation space, allowing for scalable development, yet it highlights core challenges in integrating acute perception, spatial reasoning, and grounded manipulation. We evaluate methods of different classes, including model-based RL, model-predictive control, and chain-of-thought reasoning, and illustrate the challenges KnotGym presents. KnotGym is available at https://github.com/lil-lab/knotgym.
>
---
#### [replaced 005] LightPlanner: Unleashing the Reasoning Capabilities of Lightweight Large Language Models in Task Planning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.08508v2](http://arxiv.org/pdf/2503.08508v2)**

> **作者:** Weijie Zhou; Manli Tao; Chaoyang Zhao; Honghui Dong; Ming Tang; Jinqiao Wang
>
> **备注:** The 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** In recent years, lightweight large language models (LLMs) have garnered significant attention in the robotics field due to their low computational resource requirements and suitability for edge deployment. However, in task planning -- particularly for complex tasks that involve dynamic semantic logic reasoning -- lightweight LLMs have underperformed. To address this limitation, we propose a novel task planner, LightPlanner, which enhances the performance of lightweight LLMs in complex task planning by fully leveraging their reasoning capabilities. Unlike conventional planners that use fixed skill templates, LightPlanner controls robot actions via parameterized function calls, dynamically generating parameter values. This approach allows for fine-grained skill control and improves task planning success rates in complex scenarios. Furthermore, we introduce hierarchical deep reasoning. Before generating each action decision step, LightPlanner thoroughly considers three levels: action execution (feedback verification), semantic parsing (goal consistency verification), and parameter generation (parameter validity verification). This ensures the correctness of subsequent action controls. Additionally, we incorporate a memory module to store historical actions, thereby reducing context length and enhancing planning efficiency for long-term tasks. We train the LightPlanner-1.5B model on our LightPlan-40k dataset, which comprises 40,000 action controls across tasks with 2 to 13 action steps. Experiments demonstrate that our model achieves the highest task success rate despite having the smallest number of parameters. In tasks involving spatial semantic reasoning, the success rate exceeds that of ReAct by 14.9 percent. Moreover, we demonstrate LightPlanner's potential to operate on edge devices.
>
---
#### [replaced 006] High-Precision Climbing Robot Localization Using Planar Array UWB/GPS/IMU/Barometer Integration
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.23801v2](http://arxiv.org/pdf/2509.23801v2)**

> **作者:** Shuning Zhang; Zhanchen Zhu; Xiangyu Chen; Yunheng Wang; Xu Jiang; Peibo Duan; Renjing Xu
>
> **摘要:** To address the need for high-precision localization of climbing robots in complex high-altitude environments, this paper proposes a multi-sensor fusion system that overcomes the limitations of single-sensor approaches. Firstly, the localization scenarios and the problem model are analyzed. An integrated architecture of Attention Mechanism-based Fusion Algorithm (AMFA) incorporating planar array Ultra-Wideband (UWB), GPS, Inertial Measurement Unit (IMU), and barometer is designed to handle challenges such as GPS occlusion and UWB Non-Line-of-Sight (NLOS) problem. Then, End-to-end neural network inference models for UWB and barometer are developed, along with a multimodal attention mechanism for adaptive data fusion. An Unscented Kalman Filter (UKF) is applied to refine the trajectory, improving accuracy and robustness. Finally, real-world experiments show that the method achieves 0.48 m localization accuracy and lower MAX error of 1.50 m, outperforming baseline algorithms such as GPS/INS-EKF and demonstrating stronger robustness.
>
---
#### [replaced 007] Augmenting Neural Networks-Based Model Approximators in Robotic Force-Tracking Tasks
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.08440v3](http://arxiv.org/pdf/2509.08440v3)**

> **作者:** Kevin Saad; Vincenzo Petrone; Enrico Ferrentino; Pasquale Chiacchio; Francesco Braghin; Loris Roveda
>
> **备注:** In Proceedings of the 22nd International Conference on Informatics in Control, Automation and Robotics - Volume 2: ICINCO, 394-401, 2025 , Marbella, Spain
>
> **摘要:** As robotics gains popularity, interaction control becomes crucial for ensuring force tracking in manipulator-based tasks. Typically, traditional interaction controllers either require extensive tuning, or demand expert knowledge of the environment, which is often impractical in real-world applications. This work proposes a novel control strategy leveraging Neural Networks (NNs) to enhance the force-tracking behavior of a Direct Force Controller (DFC). Unlike similar previous approaches, it accounts for the manipulator's tangential velocity, a critical factor in force exertion, especially during fast motions. The method employs an ensemble of feedforward NNs to predict contact forces, then exploits the prediction to solve an optimization problem and generate an optimal residual action, which is added to the DFC output and applied to an impedance controller. The proposed Velocity-augmented Artificial intelligence Interaction Controller for Ambiguous Models (VAICAM) is validated in the Gazebo simulator on a Franka Emika Panda robot. Against a vast set of trajectories, VAICAM achieves superior performance compared to two baseline controllers.
>
---
#### [replaced 008] DeltaFlow: An Efficient Multi-frame Scene Flow Estimation Method
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.17054v2](http://arxiv.org/pdf/2508.17054v2)**

> **作者:** Qingwen Zhang; Xiaomeng Zhu; Yushan Zhang; Yixi Cai; Olov Andersson; Patric Jensfelt
>
> **备注:** NeurIPS 2025 Spotlight, 18 pages (10 main pages + 8 supp materail), 11 figures, code at https://github.com/Kin-Zhang/DeltaFlow
>
> **摘要:** Previous dominant methods for scene flow estimation focus mainly on input from two consecutive frames, neglecting valuable information in the temporal domain. While recent trends shift towards multi-frame reasoning, they suffer from rapidly escalating computational costs as the number of frames grows. To leverage temporal information more efficiently, we propose DeltaFlow ($\Delta$Flow), a lightweight 3D framework that captures motion cues via a $\Delta$ scheme, extracting temporal features with minimal computational cost, regardless of the number of frames. Additionally, scene flow estimation faces challenges such as imbalanced object class distributions and motion inconsistency. To tackle these issues, we introduce a Category-Balanced Loss to enhance learning across underrepresented classes and an Instance Consistency Loss to enforce coherent object motion, improving flow accuracy. Extensive evaluations on the Argoverse 2, Waymo and nuScenes datasets show that $\Delta$Flow achieves state-of-the-art performance with up to 22% lower error and $2\times$ faster inference compared to the next-best multi-frame supervised method, while also demonstrating a strong cross-domain generalization ability. The code is open-sourced at https://github.com/Kin-Zhang/DeltaFlow along with trained model weights.
>
---
#### [replaced 009] Trust-Aware Assistance Seeking in Human-Supervised Autonomy
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.20496v2](http://arxiv.org/pdf/2410.20496v2)**

> **作者:** Dong Hae Mangalindan; Ericka Rovira; Vaibhav Srivastava
>
> **摘要:** Our goal is to model and experimentally assess trust evolution to predict future beliefs and behaviors of human-robot teams in dynamic environments. Research suggests that maintaining trust among team members in a human-robot team is vital for successful team performance. Research suggests that trust is a multi-dimensional and latent entity that relates to past experiences and future actions in a complex manner. Employing a human-robot collaborative task, we design an optimal assistance-seeking strategy for the robot using a POMDP framework. In the task, the human supervises an autonomous mobile manipulator collecting objects in an environment. The supervisor's task is to ensure that the robot safely executes its task. The robot can either choose to attempt to collect the object or seek human assistance. The human supervisor actively monitors the robot's activities, offering assistance upon request, and intervening if they perceive the robot may fail. In this setting, human trust is the hidden state, and the primary objective is to optimize team performance. We execute two sets of human-robot interaction experiments. The data from the first experiment are used to estimate POMDP parameters, which are used to compute an optimal assistance-seeking policy evaluated in the second experiment. The estimated POMDP parameters reveal that, for most participants, human intervention is more probable when trust is low, particularly in high-complexity tasks. Our estimates suggest that the robot's action of asking for assistance in high-complexity tasks can positively impact human trust. Our experimental results show that the proposed trust-aware policy is better than an optimal trust-agnostic policy. By comparing model estimates of human trust, obtained using only behavioral data, with the collected self-reported trust values, we show that model estimates are isomorphic to self-reported responses.
>
---
#### [replaced 010] LIBERO-Plus: In-depth Robustness Analysis of Vision-Language-Action Models
- **分类: cs.RO; cs.CL; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.13626v2](http://arxiv.org/pdf/2510.13626v2)**

> **作者:** Senyu Fei; Siyin Wang; Junhao Shi; Zihao Dai; Jikun Cai; Pengfang Qian; Li Ji; Xinzhe He; Shiduo Zhang; Zhaoye Fei; Jinlan Fu; Jingjing Gong; Xipeng Qiu
>
> **摘要:** Visual-Language-Action (VLA) models report impressive success rates on robotic manipulation benchmarks, yet these results may mask fundamental weaknesses in robustness. We perform a systematic vulnerability analysis by introducing controlled perturbations across seven dimensions: objects layout, camera viewpoints, robot initial states, language instructions, light conditions, background textures and sensor noise. We comprehensively analyzed multiple state-of-the-art models and revealed consistent brittleness beneath apparent competence. Our analysis exposes critical weaknesses: models exhibit extreme sensitivity to perturbation factors, including camera viewpoints and robot initial states, with performance dropping from 95% to below 30% under modest perturbations. Surprisingly, models are largely insensitive to language variations, with further experiments revealing that models tend to ignore language instructions completely. Our findings challenge the assumption that high benchmark scores equate to true competency and highlight the need for evaluation practices that assess reliability under realistic variation.
>
---
#### [replaced 011] RESample: A Robust Data Augmentation Framework via Exploratory Sampling for Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.17640v2](http://arxiv.org/pdf/2510.17640v2)**

> **作者:** Yuquan Xue; Guanxing Lu; Zhenyu Wu; Chuanrui Zhang; Bofang Jia; Zhengyi Gu; Yansong Tang; Ziwei Wang
>
> **备注:** 9 pages,7 figures, submitted to ICRA2026
>
> **摘要:** Vision-Language-Action models (VLAs) have demonstrated remarkable performance on complex robotic manipulation tasks through imitation learning. However, existing imitation learning datasets contain only successful trajectories and lack failure or recovery data, especially for out-of-distribution (OOD) states where the robot deviates from the main policy due to minor perturbations or errors, leading VLA models to struggle with states deviating from the training distribution. To this end, we propose an automated OOD data augmentation framework named RESample through exploratory sampling. Specifically, we first leverage offline reinforcement learning to obtain an action-value network that accurately identifies sub-optimal actions under the current manipulation policy. We further sample potential OOD states from trajectories via rollout, and design an exploratory sampling mechanism that adaptively incorporates these action proxies into the training dataset to ensure efficiency. Subsequently, our framework explicitly encourages the VLAs to recover from OOD states and enhances their robustness against distributional shifts. We conduct extensive experiments on the LIBERO benchmark as well as real-world robotic manipulation tasks, demonstrating that RESample consistently improves the stability and generalization ability of VLA models.
>
---
#### [replaced 012] Mix Q-learning for Lane Changing: A Collaborative Decision-Making Method in Multi-Agent Deep Reinforcement Learning
- **分类: cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2406.09755v2](http://arxiv.org/pdf/2406.09755v2)**

> **作者:** Xiaojun Bi; Mingjie He; Yiwen Sun
>
> **摘要:** Lane-changing decisions, which are crucial for autonomous vehicle path planning, face practical challenges due to rule-based constraints and limited data. Deep reinforcement learning has become a major research focus due to its advantages in data acquisition and interpretability. However, current models often overlook collaboration, which affects not only impacts overall traffic efficiency but also hinders the vehicle's own normal driving in the long run. To address the aforementioned issue, this paper proposes a method named Mix Q-learning for Lane Changing(MQLC) that integrates a hybrid value Q network, taking into account both collective and individual benefits for the greater good. At the collective level, our method coordinates the individual Q and global Q networks by utilizing global information. This enables agents to effectively balance their individual interests with the collective benefit. At the individual level, we integrated a deep learning-based intent recognition module into our observation and enhanced the decision network. These changes provide agents with richer decision information and more accurate feature extraction for improved lane-changing decisions. This strategy enables the multi-agent system to learn and formulate optimal decision-making strategies effectively. Our MQLC model, through extensive experimental results, impressively outperforms other state-of-the-art multi-agent decision-making methods, achieving significantly safer and faster lane-changing decisions. The code is available at https:github.com/pku-smart-city/source_code/tree/main/MQLC.
>
---
#### [replaced 013] HAVT-IVD: Heterogeneity-Aware Cross-Modal Network for Audio-Visual Surveillance: Idling Vehicles Detection With Multichannel Audio and Multiscale Visual Cues
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.16102v2](http://arxiv.org/pdf/2504.16102v2)**

> **作者:** Xiwen Li; Xiaoya Tang; Tolga Tasdizen
>
> **摘要:** Idling vehicle detection (IVD) uses surveillance video and multichannel audio to localize and classify vehicles in the last frame as moving, idling, or engine-off in pick-up zones. IVD faces three challenges: (i) modality heterogeneity between visual cues and audio patterns; (ii) large box scale variation requiring multi-resolution detection; and (iii) training instability due to coupled detection heads. The previous end-to-end (E2E) model with simple CBAM-based bi-modal attention fails to handle these issues and often misses vehicles. We propose HAVT-IVD, a heterogeneity-aware network with a visual feature pyramid and decoupled heads. Experiments show HAVT-IVD improves mAP by 7.66 over the disjoint baseline and 9.42 over the E2E baseline.
>
---
#### [replaced 014] Reinforcement Learning with Action Chunking
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **链接: [http://arxiv.org/pdf/2507.07969v3](http://arxiv.org/pdf/2507.07969v3)**

> **作者:** Qiyang Li; Zhiyuan Zhou; Sergey Levine
>
> **备注:** The Thirty-Ninth Annual Conference on Neural Information Processing Systems (NeurIPS 2025); 36 pages, 17 figures
>
> **摘要:** We present Q-chunking, a simple yet effective recipe for improving reinforcement learning (RL) algorithms for long-horizon, sparse-reward tasks. Our recipe is designed for the offline-to-online RL setting, where the goal is to leverage an offline prior dataset to maximize the sample-efficiency of online learning. Effective exploration and sample-efficient learning remain central challenges in this setting, as it is not obvious how the offline data should be utilized to acquire a good exploratory policy. Our key insight is that action chunking, a technique popularized in imitation learning where sequences of future actions are predicted rather than a single action at each timestep, can be applied to temporal difference (TD)-based RL methods to mitigate the exploration challenge. Q-chunking adopts action chunking by directly running RL in a 'chunked' action space, enabling the agent to (1) leverage temporally consistent behaviors from offline data for more effective online exploration and (2) use unbiased $n$-step backups for more stable and efficient TD learning. Our experimental results demonstrate that Q-chunking exhibits strong offline performance and online sample efficiency, outperforming prior best offline-to-online methods on a range of long-horizon, sparse-reward manipulation tasks.
>
---
#### [replaced 015] Real-Time Gait Adaptation for Quadrupeds using Model Predictive Control and Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.20706v2](http://arxiv.org/pdf/2510.20706v2)**

> **作者:** Prakrut Kotecha; Ganga Nair B; Shishir Kolathaya
>
> **备注:** 7 pages
>
> **摘要:** Model-free reinforcement learning (RL) has enabled adaptable and agile quadruped locomotion; however, policies often converge to a single gait, leading to suboptimal performance. Traditionally, Model Predictive Control (MPC) has been extensively used to obtain task-specific optimal policies but lacks the ability to adapt to varying environments. To address these limitations, we propose an optimization framework for real-time gait adaptation in a continuous gait space, combining the Model Predictive Path Integral (MPPI) algorithm with a Dreamer module to produce adaptive and optimal policies for quadruped locomotion. At each time step, MPPI jointly optimizes the actions and gait variables using a learned Dreamer reward that promotes velocity tracking, energy efficiency, stability, and smooth transitions, while penalizing abrupt gait changes. A learned value function is incorporated as terminal reward, extending the formulation to an infinite-horizon planner. We evaluate our framework in simulation on the Unitree Go1, demonstrating an average reduction of up to 36.48 % in energy consumption across varying target speeds, while maintaining accurate tracking and adaptive, task-appropriate gaits.
>
---
#### [replaced 016] SimuRA: A World-Model-Driven Simulative Reasoning Architecture for General Goal-Oriented Agents
- **分类: cs.AI; cs.CL; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.23773v2](http://arxiv.org/pdf/2507.23773v2)**

> **作者:** Mingkai Deng; Jinyu Hou; Zhiting Hu; Eric Xing
>
> **备注:** This submission has been updated to adjust the scope and presentation of the work
>
> **摘要:** AI agents built on foundation models hold enormous promise. Current practice, however, focuses on a one-task-one-agent approach, which not only falls short of scalability and generality, but also faces practical limitations from black-box autoregressive reasoning, where decisions unfold token by token without explicit simulation or counterfactual evaluation of outcomes. Humans, on the other hand, reason and plan by mentally simulating the consequences of actions within an internal model of the world -- a capability that supports flexible, goal-directed behavior across diverse contexts. Moving towards a more general and powerful AI agent, we introduce SimuRA, a goal-oriented architecture for generalized agentic reasoning. Based on a principled formulation of an optimal agent in any general environment, SimuRA addresses the limitations of black-box autoregressive reasoning by incorporating the world model for planning via simulation. Our prototype world model is implemented using LLMs as a substrate, leveraging the natural language as a discrete, hierarchical representation grounded in concepts for planning, while remaining model-agnostic. On complex web-browsing tasks such as flight search, SimuRA improves the success rate from 0% to 32.2% compared to a representative open-web agent baseline. Across tasks, world-model-based planning achieves up to 124% higher task completion rates than a matched black-box autoregressive baseline, demonstrating the advantages of simulative reasoning. We release ReasonerAgent-Web, a web-browsing agent built on SimuRA, as an open-source research demo.
>
---
#### [replaced 017] Grasp2Grasp: Vision-Based Dexterous Grasp Translation via Schrödinger Bridges
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.02489v2](http://arxiv.org/pdf/2506.02489v2)**

> **作者:** Tao Zhong; Jonah Buchanan; Christine Allen-Blanchette
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** We propose a new approach to vision-based dexterous grasp translation, which aims to transfer grasp intent across robotic hands with differing morphologies. Given a visual observation of a source hand grasping an object, our goal is to synthesize a functionally equivalent grasp for a target hand without requiring paired demonstrations or hand-specific simulations. We frame this problem as a stochastic transport between grasp distributions using the Schr\"odinger Bridge formalism. Our method learns to map between source and target latent grasp spaces via score and flow matching, conditioned on visual observations. To guide this translation, we introduce physics-informed cost functions that encode alignment in base pose, contact maps, wrench space, and manipulability. Experiments across diverse hand-object pairs demonstrate our approach generates stable, physically grounded grasps with strong generalization. This work enables semantic grasp transfer for heterogeneous manipulators and bridges vision-based grasping with probabilistic generative modeling. Additional details at https://grasp2grasp.github.io/
>
---
#### [replaced 018] Rectified Point Flow: Generic Point Cloud Pose Estimation
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.05282v2](http://arxiv.org/pdf/2506.05282v2)**

> **作者:** Tao Sun; Liyuan Zhu; Shengyu Huang; Shuran Song; Iro Armeni
>
> **备注:** NeurIPS 2025 Camera-ready. Project page: https://rectified-pointflow.github.io/
>
> **摘要:** We introduce Rectified Point Flow, a unified parameterization that formulates pairwise point cloud registration and multi-part shape assembly as a single conditional generative problem. Given unposed point clouds, our method learns a continuous point-wise velocity field that transports noisy points toward their target positions, from which part poses are recovered. In contrast to prior work that regresses part-wise poses with ad-hoc symmetry handling, our method intrinsically learns assembly symmetries without symmetry labels. Together with a self-supervised encoder focused on overlapping points, our method achieves a new state-of-the-art performance on six benchmarks spanning pairwise registration and shape assembly. Notably, our unified formulation enables effective joint training on diverse datasets, facilitating the learning of shared geometric priors and consequently boosting accuracy. Project page: https://rectified-pointflow.github.io/.
>
---
#### [replaced 019] BEAST: Efficient Tokenization of B-Splines Encoded Action Sequences for Imitation Learning
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.06072v3](http://arxiv.org/pdf/2506.06072v3)**

> **作者:** Hongyi Zhou; Weiran Liao; Xi Huang; Yucheng Tang; Fabian Otto; Xiaogang Jia; Xinkai Jiang; Simon Hilber; Ge Li; Qian Wang; Ömer Erdinç Yağmurlu; Nils Blank; Moritz Reuss; Rudolf Lioutikov
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** We present the B-spline Encoded Action Sequence Tokenizer (BEAST), a novel action tokenizer that encodes action sequences into compact discrete or continuous tokens using B-splines. In contrast to existing action tokenizers based on vector quantization or byte pair encoding, BEAST requires no separate tokenizer training and consistently produces tokens of uniform length, enabling fast action sequence generation via parallel decoding. Leveraging our B-spline formulation, BEAST inherently ensures generating smooth trajectories without discontinuities between adjacent segments. We extensively evaluate BEAST by integrating it with three distinct model architectures: a Variational Autoencoder (VAE) with continuous tokens, a decoder-only Transformer with discrete tokens, and Florence-2, a pretrained Vision-Language Model with an encoder-decoder architecture, demonstrating BEAST's compatibility and scalability with large pretrained models. We evaluate BEAST across three established benchmarks consisting of 166 simulated tasks and on three distinct robot settings with a total of 8 real-world tasks. Experimental results demonstrate that BEAST (i) significantly reduces both training and inference computational costs, and (ii) consistently generates smooth, high-frequency control signals suitable for continuous control tasks while (iii) reliably achieves competitive task success rates compared to state-of-the-art methods.
>
---
#### [replaced 020] Visual Cues Enhance Predictive Turn-Taking for Two-Party Human Interaction
- **分类: cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.21043v2](http://arxiv.org/pdf/2505.21043v2)**

> **作者:** Sam O'Connor Russell; Naomi Harte
>
> **备注:** Accepted to ACL 2025, Findings of the Association for Computational Linguistics
>
> **摘要:** Turn-taking is richly multimodal. Predictive turn-taking models (PTTMs) facilitate naturalistic human-robot interaction, yet most rely solely on speech. We introduce MM-VAP, a multimodal PTTM which combines speech with visual cues including facial expression, head pose and gaze. We find that it outperforms the state-of-the-art audio-only in videoconferencing interactions (84% vs. 79% hold/shift prediction accuracy). Unlike prior work which aggregates all holds and shifts, we group by duration of silence between turns. This reveals that through the inclusion of visual features, MM-VAP outperforms a state-of-the-art audio-only turn-taking model across all durations of speaker transitions. We conduct a detailed ablation study, which reveals that facial expression features contribute the most to model performance. Thus, our working hypothesis is that when interlocutors can see one another, visual cues are vital for turn-taking and must therefore be included for accurate turn-taking prediction. We additionally validate the suitability of automatic speech alignment for PTTM training using telephone speech. This work represents the first comprehensive analysis of multimodal PTTMs. We discuss implications for future work and make all code publicly available.
>
---
#### [replaced 021] HYPE: Hybrid Planning with Ego Proposal-Conditioned Predictions
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.12733v2](http://arxiv.org/pdf/2510.12733v2)**

> **作者:** Hang Yu; Julian Jordan; Julian Schmidt; Silvan Lindner; Alessandro Canevaro; Wilhelm Stork
>
> **备注:** Accepted to IEEE ITSC 2025
>
> **摘要:** Safe and interpretable motion planning in complex urban environments needs to reason about bidirectional multi-agent interactions. This reasoning requires to estimate the costs of potential ego driving maneuvers. Many existing planners generate initial trajectories with sampling-based methods and refine them by optimizing on learned predictions of future environment states, which requires a cost function that encodes the desired vehicle behavior. Designing such a cost function can be very challenging, especially if a wide range of complex urban scenarios has to be considered. We propose HYPE: HYbrid Planning with Ego proposal-conditioned predictions, a planner that integrates multimodal trajectory proposals from a learned proposal model as heuristic priors into a Monte Carlo Tree Search (MCTS) refinement. To model bidirectional interactions, we introduce an ego-conditioned occupancy prediction model, enabling consistent, scene-aware reasoning. Our design significantly simplifies cost function design in refinement by considering proposal-driven guidance, requiring only minimalistic grid-based cost terms. Evaluations on large-scale real-world benchmarks nuPlan and DeepUrban show that HYPE effectively achieves state-of-the-art performance, especially in safety and adaptability.
>
---
