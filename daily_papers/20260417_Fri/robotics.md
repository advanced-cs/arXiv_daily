# 机器人 cs.RO

- **最新发布 35 篇**

- **更新 22 篇**

## 最新发布

#### [new 001] Differentiable Object Pose Connectivity Metrics for Regrasp Sequence Optimization
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决多步重抓规划问题。通过构建可微的位姿连通性度量，实现连续优化和自适应步骤选择，提升规划鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.14733](https://arxiv.org/pdf/2604.14733)**

> **作者:** Liang Qin; Weiwei Wan; Kensuke Harada
>
> **摘要:** Regrasp planning is often required when one pick-and-place cannot transfer an object from an initial pose to a goal pose while maintaining grasp feasibility. The main challenge is to reason about shared-grasp connectivity across intermediate poses, where discrete search becomes brittle. We propose an implicit multi-step regrasp planning framework based on differentiable pose sequence connectivity metrics. We model grasp feasibility under an object pose using an Energy-Based Model (EBM) and leverage energy additivity to construct a continuous energy landscape that measures pose-pair connectivity, enabling gradient-based optimization of intermediate object poses. An adaptive iterative deepening strategy is introduced to determine the minimum number of intermediate steps automatically. Experiments show that the proposed cost formulation provides smooth and informative gradients, improving planning robustness over other alternatives. They also demonstrate generalization to unseen grasp poses and cross-end-effector transfer, where a model trained with suction constraints can guide parallel gripper grasp manipulation. The multi-step planning results further highlight the effectiveness of adaptive deepening and minimum-step search.
>
---
#### [new 002] RoSLAC: Robust Simultaneous Localization and Calibration of Multiple Magnetometers
- **分类: cs.RO**

- **简介: 该论文属于自主机器人定位任务，解决磁力计在复杂环境中因铁磁材料干扰导致的定位不准问题，提出RoSLAC方法实现定位与校准的联合优化。**

- **链接: [https://arxiv.org/pdf/2604.14353](https://arxiv.org/pdf/2604.14353)**

> **作者:** Qiyang Lyu; Zhenyu Wu; Wei Wang; Hongming Shen; Danwei Wang
>
> **摘要:** Localization of autonomous mobile robots (AMRs) in enclosed or semi-enclosed environments such as offices, hotels, hospitals, indoor parking facilities, and underground spaces where GPS signals are weak or unavailable remains a major obstacle to the deployment of fully autonomous systems. Infrastructure-based localization approaches, such as QR codes and RFID, are constrained by high installation and maintenance costs as well as limited flexibility, while onboard sensor-based methods, including LiDAR- and vision-based solutions, are affected by ambiguous geometric features and frequent occlusions caused by dynamic obstacles such as pedestrians. Ambient magnetic field (AMF)-based localization has therefore attracted growing interest in recent years because it does not rely on external infrastructure or geometric features, making it well-suited for AMR applications such as service robots and security robots. However, magnetometer measurements are often corrupted by distortions caused by ferromagnetic materials present on the sensor platform, which bias the AMF and degrade localization reliability. As a result, accurate magnetometer calibration to estimate distortion parameters becomes essential. Conventional calibration methods that rely on rotating the magnetometer are impractical for large and heavy platforms. To address this limitation, this paper proposes a robust simultaneous localization and calibration (RoSLAC) approach based on alternating optimization, which iteratively and efficiently estimates both the platform pose and magnetometer calibration parameters. Extensive evaluations conducted in high-fidelity simulation and real-world environments demonstrate that the proposed RoSLAC method achieves high localization accuracy while maintaining low computational cost compared with state-of-the-art magnetometer calibration techniques.
>
---
#### [new 003] CART: Context-Aware Terrain Adaptation using Temporal Sequence Selection for Legged Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人地形适应任务，解决视觉与本体感知不一致导致的行走不稳定问题。提出CART方法，融合多模态传感信息，提升机器人在复杂地形上的稳定性和成功率。**

- **链接: [https://arxiv.org/pdf/2604.14344](https://arxiv.org/pdf/2604.14344)**

> **作者:** Kartikeya Singh; Youngjin Kim; Yash Turkar; Karthik Dantu
>
> **摘要:** Animals in nature combine multiple modalities, such as sight and feel, to perceive terrain and develop an understanding of how to walk on uneven terrain in a stable manner. Similarly, legged robots need to develop their ability to stably walk on complex terrains by developing an understanding of the relationship between vision and proprioception. Most current terrain adaptation methods are susceptible to failure on complex, off-road terrain as they rely on prior experience, particularly observations from a vision sensor. This experience-based learning often creates a Visual-Texture Paradox between what has been seen and how it actually feels. In this work, we introduce CART, a high-level controller built on a context-aware terrain adaptation approach that integrates proprioception and exteroception from onboard sensing to achieve a robust understanding of terrain. We evaluate our method on multiple terrains using an ANYmal-C robot on the IsaacSim simulator and a Boston Dynamics SPOT robot for our real-world experiments. To evaluate the learned contextual terrain properties, we adapt vibrational stability on the base of the robot as a metric. We compare CART with various state-of-the-art baselines equipped with multimodal sensing in both simulation and the real world. CART achieves an average success rate improvement of 5% over all baselines in simulation and improves the overall stability up to 45% and 24% in the real world without increasing the time taken by the robot to accomplish locomotion tasks.
>
---
#### [new 004] Graph Theoretical Outlier Rejection for 4D Radar Registration in Feature-Poor Environments
- **分类: cs.RO**

- **简介: 该论文属于雷达点云配准任务，解决特征稀疏环境下的异常点剔除问题。通过图论方法提升ICP算法的鲁棒性，减少定位误差。**

- **链接: [https://arxiv.org/pdf/2604.14857](https://arxiv.org/pdf/2604.14857)**

> **作者:** Georg Dorndorf; Daniel Adolfsson; Masrur Doostdar
>
> **备注:** under review
>
> **摘要:** Automotive 4D imaging radar is well suited for operation in dusty and low-visibility environments, but scan registration remains challenging due to scan sparsity and spurious detections caused by noise and multipath reflections. This difficulty is compounded in feature-poor open-pit mines, where the lack of distinctive landmarks reduces correspondence reliability. We integrate graph-based pairwise consistency maximization (PCM) as an outlier rejection step within the iterative closest points (ICP) loop. We propose a radar-adapted pairwise distance-invariant scoring function for graph-based (PCM) that incorporates anisotropic, per-detection uncertainty derived from a radar measurement model. The consistency maximization problem is approximated with a greedy heuristic that finds a large clique in the pairwise consistency graph. The refined correspondence set improves robustness when the initial association set is heavily contaminated. We evaluate a standard Euclidean distance residual and our uncertainty-aware residual on an open-pit mine dataset collected with a 4D imaging radar. Compared to the generalized ICP (GICP) baseline without PCM, our method reduces segment relative position error (RPE) by 29.6% on 1 m segments and by up to 55% on 100 m segments. The presented method is intended for integration into localization pipelines and is suitable for online use due to the greedy heuristic in graph-based (PCM).
>
---
#### [new 005] DigiForest: Digital Analytics and Robotics for Sustainable Forestry
- **分类: cs.RO**

- **简介: 该论文介绍DigiForest，一种利用数字技术和机器人实现可持续林业的方案，旨在解决森林管理与生态保护问题，通过数据采集、决策支持和智能采伐提升林业效率。**

- **链接: [https://arxiv.org/pdf/2604.14652](https://arxiv.org/pdf/2604.14652)**

> **作者:** Marco Camurri; Enrico Tomelleri; Matías Mattamala; Sebastián Barbas Laina; Martin Jacquet; Jens Behley; Sunni Kanta Prasad Kushwaha; Fang Nan; Nived Chebrolu; Leonard Freißmuth; Marvin Chayton Harms; Meher V.R. Malladi; Fan Yang; Jonas Frey; Cesar Cadena; Marco Hutter; Janine Schweier; Kostas Alexis; Cyrill Stachniss; Maurice Fallon; Stefan Leutenegger
>
> **备注:** 34 pages, 24 figures
>
> **摘要:** Covering one third of Earth's land surface, forests are vital to global biodiversity, climate regulation, and human well-being. In Europe, forests and woodlands reach approximately 40% of land area, and the forestry sector is central to achieving the EU's climate neutrality and biodiversity goals; these emphasize sustainable forest management, increased use of long-lived wood products, and resilient forest ecosystems. To meet these goals and properly address their inherent challenges, current practices require further innovation. This chapter introduces DigiForest, a novel, large-scale precision forestry approach leveraging digital technologies and autonomous robotics. DigiForest is structured around four main components: (1) autonomous, heterogeneous mobile robots (aerial, legged, and marsupial) for tree-level data collection; (2) automated extraction of tree traits to build forest inventories; (3) a Decision Support System (DSS) for forecasting forest growth and supporting decision-making; and (4) low-impact selective logging using purpose-built autonomous harvesters. These technologies have been extensively validated in real-world conditions in several locations, including forests in Finland, the UK, and Switzerland.
>
---
#### [new 006] Trajectory Planning for a Multi-UAV Rigid-Payload Cascaded Transportation System Based on Enhanced Tube-RRT*
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于多无人机协同运输轨迹规划任务，解决密集环境下的安全路径生成与姿态控制问题。提出增强型Tube-RRT*算法和二次优化方法，实现高效、平滑的轨迹规划。**

- **链接: [https://arxiv.org/pdf/2604.15074](https://arxiv.org/pdf/2604.15074)**

> **作者:** Jianqiao Yu; Jia Li; Tianhua Gao
>
> **备注:** 15 pages, 7 figures. Under review at IEEE Transactions on Aerospace and Electronic Systems (TAES). This work has been submitted to the IEEE for possible publication
>
> **摘要:** This paper presents a two-stage trajectory planning framework for a multi-UAV rigid-payload cascaded transportation system, aiming to address planning challenges in densely cluttered environments. In Stage I, an Enhanced Tube-RRT* algorithm is developed by integrating active hybrid sampling and an adaptive expansion strategy, enabling rapid generation of a safe and feasible virtual tube in environments with dense obstacles. Moreover, a trajectory smoothness cost is explicitly incorporated into the edge cost to reduce excessive turns and thereby mitigate cable-induced oscillations. Simulation results demonstrate that the proposed Enhanced Tube-RRT* achieves a higher success rate and effective sampling rate than mixed-sampling Tube-RRT* (STube-RRT*) and adaptive-extension Tube-RRT* (AETube-RRT*), while producing a shorter optimal path with a smaller cumulative turning angle. In Stage II, a convex quadratic program is formulated by considering payload translational and rotational dynamics, cable tension constraints, and collision-safety constraints, yielding a smooth, collision-free desired payload trajectory. Finally, a centralized geometric control scheme is applied to the cascaded system to validate the effectiveness and feasibility of the proposed planning framework, offering a practical solution for payload attitude maneuvering in densely cluttered environments.
>
---
#### [new 007] Switch: Learning Agile Skills Switching for Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人技能切换任务，旨在解决 humanoid 机器人在不同运动技能间切换不灵活的问题。提出 Switch 系统，通过技能图和在线调度实现高效、稳定的技能过渡。**

- **链接: [https://arxiv.org/pdf/2604.14834](https://arxiv.org/pdf/2604.14834)**

> **作者:** Yuen-Fui Lau; Qihan Zhao; Yinhuai Wang; Runyi Yu; Hok Wai Tsui; Qifeng Chen; Ping Tan
>
> **摘要:** Recent advancements in whole-body control through deep reinforcement learning have enabled humanoid robots to achieve remarkable progress in real-world chal lenging locomotion skills. However, existing approaches often struggle with flexible transitions between distinct skills, cre ating safety concerns and practical limitations. To address this challenge, we introduce a hierarchical multi-skill system, Switch, enabling seamless skill transitions at any moment. Our approach comprises three key components: (1) a Skill Graph (SG) that establishes potential cross-skill transitions based on kinematic similarity within multi-skill motion data, (2) a whole-body tracking policy trained on this skill graph through deep reinforcement learning, and (3) an online skill scheduler to drive the tracking policy for robust skill execution and smooth transitions. For skill switching or significant tracking deviations, the scheduler performs online graph search to find the optimal feasible path, which ensures efficient, stable, and real-time execution of diverse locomotion skills. Comprehensive experiments demonstrate that Switch empowers humanoid to execute agile skill transitions with high success rates while maintaining strong motion imitation performance.
>
---
#### [new 008] CooperDrive: Enhancing Driving Decisions Through Cooperative Perception
- **分类: cs.RO; cs.CV**

- **简介: 论文提出CooperDrive，解决自动驾驶在遮挡和非视距场景下的感知局限问题。通过协作感知提升决策安全性，实现低延迟、高效的信息共享与融合。**

- **链接: [https://arxiv.org/pdf/2604.14454](https://arxiv.org/pdf/2604.14454)**

> **作者:** Deyuan Qu; Qi Chen; Takayuki Shimizu; Onur Altintas
>
> **备注:** Accepted at ICRA 2026
>
> **摘要:** Autonomous vehicles equipped with robust onboard perception, localization, and planning still face limitations in occlusion and non-line-of-sight (NLOS) scenarios, where delayed reactions can increase collision risk. We propose CooperDrive, a cooperative perception framework that augments situational awareness and enables earlier, safer driving decisions. CooperDrive offers two key advantages: (i) each vehicle retains its native perception, localization, and planning stack, and (ii) a lightweight object-level sharing and fusion strategy bridges perception and planning. Specifically, CooperDrive reuses detector Bird's-Eye View (BEV) features to estimate accurate vehicle poses without additional heavy encoders, thereby reconstructing BEV representations and feeding the planner with low latency. On the planning side, CooperDrive leverages the expanded object set to anticipate potential conflicts earlier and adjust speed and trajectory proactively, thereby transforming reactive behaviors into predictive and safer driving decisions. Real-world closed-loop tests at occlusion-heavy NLOS intersections demonstrate that CooperDrive increases reaction lead time, minimum time-to-collision (TTC), and stopping margin, while requiring only 90 kbps bandwidth and maintaining an average end-to-end latency of 89 ms.
>
---
#### [new 009] Abstract Sim2Real through Approximate Information States
- **分类: cs.RO**

- **简介: 该论文研究sim2real任务，解决抽象模拟器与真实世界间的策略迁移问题。通过状态抽象和动态修正，提升策略在真实环境中的表现。**

- **链接: [https://arxiv.org/pdf/2604.15289](https://arxiv.org/pdf/2604.15289)**

> **作者:** Yunfu Deng; Yuhao Li; Josiah P. Hanna
>
> **摘要:** In recent years, reinforcement learning (RL) has shown remarkable success in robotics when a fast and accurate simulator is available for a given task. When using RL and simulation, more simulator realism is generally beneficial but becomes harder to obtain as robots are deployed in increasingly complex and widescale domains. In such settings, simulators will likely fail to model all relevant details of a given target task and this observation motivates the study of sim2real with simulators that leave out key task details. In this paper, we formalize and study the abstract sim2real problem: given an abstract simulator that models a target task at a coarse level of abstraction, how can we train a policy with RL in the abstract simulator and successfully transfer it to the real-world? Our first contribution is to formalize this problem using the language of state abstraction from the RL literature. This framing shows that an abstract simulator can be grounded to match the target task if the grounded abstract dynamics take the history of states into account. Based on the formalism, we then introduce a method that uses real-world task data to correct the dynamics of the abstract simulator. We then show that this method enables successful policy transfer both in sim2sim and sim2real evaluation.
>
---
#### [new 010] Model-Based Reinforcement Learning Exploits Passive Body Dynamics for High-Performance Biped Robot Locomotion
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人运动控制任务，旨在提升双足机器人的行走性能。通过模型强化学习利用被动身体动力学，解决能耗与稳定性问题，验证了被动元件对高效运动的重要性。**

- **链接: [https://arxiv.org/pdf/2604.14565](https://arxiv.org/pdf/2604.14565)**

> **作者:** Tomoya Kamimura; Haruka Washiyama; Akihito Sano
>
> **摘要:** Embodiment is a significant keyword in recent machine learning fields. This study focused on the passive nature of the body of a biped robot to generate walking and running locomotion using model-based deep reinforcement learning. We constructed two models in a simulator, one with passive elements (e.g., springs) and the other, which is similar to general humanoids, without passive elements. The training of the model with passive elements was highly affected by the attractor of the system. This lead that although the trajectories quickly converged to limit cycles, it took a long time to obtain large rewards. However, thanks to the attractor-driven learning, the acquired locomotion was robust and energy-efficient. The results revealed that robots with passive elements could efficiently acquire high-performance locomotion by utilizing stable limit cycles generated through dynamic interaction between the body and ground. This study demonstrates the importance of implementing passive properties in the body for future embodied AI.
>
---
#### [new 011] 4D Radar Gaussian Modeling and Scan Matching with RCS
- **分类: cs.RO**

- **简介: 该论文属于激光雷达建模与配准任务，旨在提升4D雷达数据的使用效率。通过引入RCS信息，增强场景建模与扫描匹配效果。**

- **链接: [https://arxiv.org/pdf/2604.14868](https://arxiv.org/pdf/2604.14868)**

> **作者:** Fernando Amodeo; Luis Merino; Fernando Caballero
>
> **备注:** This version is an Extended Abstract, sent to the Radar in Robotics: New Frontiers workshop of ICRA 2026
>
> **摘要:** 4D millimeter-wave (mmWave) radars are increasingly used in robotics, as they offer robustness against adverse environmental conditions. Besides the usual XYZ position, they provide Doppler velocity measurements as well as Radar Cross Section (RCS) information for every point. While Doppler is widely used to filter out dynamic points, RCS is often overlooked and not usually used in modeling and scan matching processes. Building on previous 3D Gaussian modeling and scan matching work, we propose incorporating the physical behavior of RCS in the model, in order to further enrich the summarized information about the scene, and improve the scan matching process.
>
---
#### [new 012] BIEVR-LIO: Robust LiDAR-Inertial Odometry through Bump-Image-Enhanced Voxel Maps
- **分类: cs.RO**

- **简介: 该论文属于机器人定位任务，解决LiDAR-Inertial Odometry在信息匮乏环境中的精度下降问题。提出BIEVR-LIO方法，通过高分辨率体素地图和区域采样策略提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.14421](https://arxiv.org/pdf/2604.14421)**

> **作者:** Patrick Pfreundschuh; Turcan Tuna; Cedric Le Gentil; Roland Siegwart; Cesar Cadena; Helen Oleynikova
>
> **摘要:** Reliable odometry is essential for mobile robots as they increasingly enter more challenging environments, which often contain little information to constrain point cloud registration, resulting in degraded LiDAR-Inertial Odometry (LIO) accuracy or even divergence. To address this, we present BIEVR-LIO, a novel approach designed specifically to exploit subtle variations in the available geometry for improved robustness. We propose a high-resolution map representation that stores surfaces as compact voxel-wise oriented height images. This representation can directly be used for registration without the calculation of intermediate geometric primitives while still supporting efficient updates. Since informative geometry is often sparsely distributed in the environment, we further propose a map-informed point sampling strategy to focus registration on geometrically informative regions, improving robustness in uninformative environments while reducing computational cost compared to global high-resolution sampling. Experiments across multiple sensors, platforms, and environments demonstrates state-of-the-art performance in well-constrained scenes and substantial improvements in challenging scenarios where baseline methods diverge. Additionally, we demonstrate that the fine-grained geometry captured by BIEVR-LIO can be used for downstream tasks such as elevation mapping for robot locomotion.
>
---
#### [new 013] POMDP-based Object Search with Growing State Space and Hybrid Action Domain
- **分类: cs.RO**

- **简介: 该论文属于机器人目标搜索任务，解决复杂室内环境中高效定位目标物体的问题。通过构建高维POMDP模型并提出新型求解方法，提升搜索效率与可靠性。**

- **链接: [https://arxiv.org/pdf/2604.14965](https://arxiv.org/pdf/2604.14965)**

> **作者:** Yongbo Chen; Hesheng Wang; Shoudong Huang; Hanna Kurniawati
>
> **摘要:** Efficiently locating target objects in complex indoor environments with diverse furniture, such as shelves, tables, and beds, is a significant challenge for mobile robots. This difficulty arises from factors like localization errors, limited fields of view, and visual occlusion. We address this by framing the object-search task as a highdimensional Partially Observable Markov Decision Process (POMDP) with a growing state space and hybrid (continuous and discrete) action spaces in 3D environments. Based on a meticulously designed perception module, a novel online POMDP solver named the growing neural process filtered k-center clustering tree (GNPF-kCT) is proposed to tackle this problem. Optimal actions are selected using Monte Carlo Tree Search (MCTS) with belief tree reuse for growing state space, a neural process network to filter useless primitive actions, and k-center clustering hypersphere discretization for efficient refinement of high-dimensional action spaces. A modified upper-confidence bound (UCB), informed by belief differences and action value functions within cells of estimated diameters, guides MCTS expansion. Theoretical analysis validates the convergence and performance potential of our method. To address scenarios with limited information or rewards, we also introduce a guessed target object with a grid-world model as a key strategy to enhance search efficiency. Extensive Gazebo simulations with Fetch and Stretch robots demonstrate faster and more reliable target localization than POMDP-based baselines and state-of-the-art (SOTA) non-POMDP-based solvers, especially large language model (LLM) based methods, in object search under the same computational constraints and perception systems. Real-world tests in office environments confirm the practical applicability of our approach. Project page: this https URL.
>
---
#### [new 014] DEX-Mouse: A Low-cost Portable and Universal Interface with Force Feedback for Data Collection of Dexterous Robotic Hands
- **分类: cs.RO**

- **简介: 该论文提出DEX-Mouse，一种低成本、便携的力反馈操作接口，用于抓取机器人手的数据显示。解决传统方法在跨平台兼容性和便携性上的不足，通过无需校准的设计实现快速部署。**

- **链接: [https://arxiv.org/pdf/2604.15013](https://arxiv.org/pdf/2604.15013)**

> **作者:** Joonho Koh; Haechan Jung; Nayoung Kim; Wook Ko; Changjoo Nam
>
> **摘要:** Data-driven dexterous hand manipulation requires large-scale, physically consistent demonstration data. Simulation and video-based methods suffer from sim-to-real gaps and retargeting problems, while MoCap glove-based teleoperation systems require per-operator calibration and lack portability, as the robot hand is typically fixed to a stationary arm. Portable alternatives improve mobility but lack cross-platform and cross-operator compatibility. We present DEX-Mouse, a portable, calibration-free hand-held teleoperation interface with integrated kinesthetic force feedback, built from commercial off-the-shelf components under USD 150. The operator-agnostic design requires no calibration or structural modification, enabling immediate deployment across diverse environments and platforms. The interface supports a configuration in which the target robot hand is mounted directly on the forearm of an operator, producing robot-aligned data. In a comparative user study across various dexterous manipulation tasks, operators using the proposed system achieved an 86.67% task completion rate under the attached configuration. Also, we found that the attached configuration reduced the perceived workload of the operators compared to spatially separated teleoperation setups across all compared interfaces. The complete hardware and software stack, including bill of materials, CAD models, and firmware, is open-sourced at this https URL to facilitate replication and adoption.
>
---
#### [new 015] SpaceMind: A Modular and Self-Evolving Embodied Vision-Language Agent Framework for Autonomous On-orbit Servicing
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文提出SpaceMind框架，用于自主在轨服务任务，解决视觉-语言代理的模块化与自进化问题，通过多模态感知、动态技能管理和自我优化实现高效任务执行。**

- **链接: [https://arxiv.org/pdf/2604.14399](https://arxiv.org/pdf/2604.14399)**

> **作者:** Aodi Wu; Haodong Han; Xubo Luo; Ruisuo Wang; Shan He; Xue Wan
>
> **备注:** 23 pages, 6 figures, 7 tables. Code available at this https URL
>
> **摘要:** Autonomous on-orbit servicing demands embodied agents that perceive through visual sensors, reason about 3D spatial situations, and execute multi-phase tasks over extended horizons. We present SpaceMind, a modular and self-evolving vision-language model (VLM) agent framework that decomposes knowledge, tools, and reasoning into three independently extensible dimensions: skill modules with dynamic routing, Model Context Protocol (MCP) tools with configurable profiles, and injectable reasoning-mode skills. An MCP-Redis interface layer enables the same codebase to operate across simulation and physical hardware without modification, and a Skill Self-Evolution mechanism distills operational experience into persistent skill files without model fine-tuning. We validate SpaceMind through 192 closed-loop runs across five satellites, three task types, and two environments, a UE5 simulation and a physical laboratory, deliberately including degraded conditions to stress-test robustness. Under nominal conditions all modes achieve 90--100% navigation success; under degradation, the Prospective mode uniquely succeeds in search-and-approach tasks where other modes fail. A self-evolution study shows that the agent recovers from failure in four of six groups from a single failed episode, including complete failure to 100% success and inspection scores improving from 12 to 59 out of 100. Real-world validation confirms zero-code-modification transfer to a physical robot with 100% rendezvous success. Code: this https URL
>
---
#### [new 016] CAVERS: Multimodal SLAM Data from a Natural Karstic Cave with Ground Truth Motion Capture
- **分类: cs.RO**

- **简介: 该论文提出CAVERS数据集，用于自然溶洞中的多模态SLAM研究，解决复杂环境下的自主导航问题。**

- **链接: [https://arxiv.org/pdf/2604.15052](https://arxiv.org/pdf/2604.15052)**

> **作者:** Giacomo Franchini; David Rodríguez-Martínez; Alfonso Martínez-Petersen; C. J. Pérez-del-Pulgar; Marcello Chiaberge
>
> **备注:** 8 pages, 5 figures, preprint version
>
> **摘要:** Autonomous robots operating in natural karstic caves face perception and navigation challenges that are qualitatively distinct from those encountered in mines or tunnels: irregular geometry, reflective wet surfaces, near-zero ambient light, and complex branching passages. Yet publicly available datasets targeting this environment remain scarce and offer limited sensing modalities and environmental diversity. We present CAVERS, a multimodal dataset acquired in two structurally distinct rooms of Cueva de la Victoria, Málaga, Spain, comprising 24 sequences totaling approximately 335 GB of recorded data. The sensor suite combines an Intel RealSense D435i RGB-D-I camera, an Optris PI640i near-IR thermal camera, and a Velodyne VLP-16 LiDAR, operated both handheld and mounted on a wheeled rover under full darkness and artificial illumination. For most of the sequences, mm-accurate 6-DoF ground truth pose and velocity at 120 Hz are provided by an Optirack motion capture system installed directly inside the cave. We benchmark seven state-of-the-art SLAM and odometry algorithms spanning visual, visual-inertial, thermal-inertial, and LiDAR-based pipelines, as well as a 3D reconstruction pipeline, demonstrating the dataset's usability. %The dataset and all supplementary material are publicly available at: this https URL.
>
---
#### [new 017] Vision-Based Safe Human-Robot Collaboration with Uncertainty Guarantees
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于人机协作任务，旨在提升视觉引导下的人机协作安全性。通过融合不确定性估计与异常检测，提供可验证的安全保障。**

- **链接: [https://arxiv.org/pdf/2604.15221](https://arxiv.org/pdf/2604.15221)**

> **作者:** Jakob Thumm; Marian Frei; Tianle Ni; Matthias Althoff; Marco Pavone
>
> **摘要:** We propose a framework for vision-based human pose estimation and motion prediction that gives conformal prediction guarantees for certifiably safe human-robot collaboration. Our framework combines aleatoric uncertainty estimation with OOD detection for high probabilistic confidence. To integrate our pipeline in certifiable safety frameworks, we propose conformal prediction sets for human motion predictions with high, valid confidence. We evaluate our pipeline on recorded human motion data and a real-world human-robot collaboration setting.
>
---
#### [new 018] Keep It CALM: Toward Calibration-Free Kilometer-Level SLAM with Visual Geometry Foundation Models via an Assistant Eye
- **分类: cs.RO**

- **简介: 该论文属于SLAM任务，解决千米级定位与建图中的几何失准问题。提出CAL2M框架，通过辅助视觉和全局映射策略，实现无需标定的精准对齐与一致重建。**

- **链接: [https://arxiv.org/pdf/2604.14795](https://arxiv.org/pdf/2604.14795)**

> **作者:** Tianjun Zhang; Fengyi Zhang; Tianchen Deng; Lin Zhang; Hesheng Wang
>
> **备注:** 19 pages, 8 figures, submitted to IEEE TPAMI
>
> **摘要:** Visual Geometry Foundation Models (VGFMs) demonstrate remarkable zero-shot capabilities in local reconstruction. However, deploying them for kilometer-level Simultaneous Localization and Mapping (SLAM) remains challenging. In such scenarios, current approaches mainly rely on linear transforms (e.g., Sim3 and SL4) for sub-map alignment, while we argue that a single linear transform is fundamentally insufficient to model the complex, non-linear geometric distortions inherent in VGFM outputs. Forcing such rigid alignment leads to the rapid accumulation of uncorrected residuals, eventually resulting in significant trajectory drift and map divergence. To address these limitations, we present CAL2M (Calibration-free Assistant-eye based Large-scale Localization and Mapping), a plug-and-play framework compatible with arbitrary VGFMs. Distinct from traditional systems, CAL2M introduces an "assistant eye" solely to leverage the prior of constant physical spacing, effectively eliminating scale ambiguity without any temporal or spatial pre-calibration. Furthermore, leveraging the assumption of accurate feature matching, we propose an epipolar-guided intrinsic and pose correction model. Supported by an online intrinsic search module, it can effectively rectify rotation and translation errors caused by inaccurate intrinsics through fundamental matrix decomposition. Finally, to ensure accurate mapping, we introduce a globally consistent mapping strategy based on anchor propagation. By constructing and fusing anchors across the trajectory, we establish a direct local-to-global mapping relationship. This enables the application of nonlinear transformations to elastically align sub-maps, effectively eliminating geometric misalignments and ensuring a globally consistent reconstruction. The source code of CAL2M will be publicly available at this https URL.
>
---
#### [new 019] A multi-platform LiDAR dataset for standardized forest inventory measurement at long term ecological monitoring sites
- **分类: cs.RO**

- **简介: 该论文属于森林结构测量任务，旨在解决长期生态监测中的3D数据标准化问题。通过多平台LiDAR数据融合，提升森林 inventory 的精度与效率。**

- **链接: [https://arxiv.org/pdf/2604.14635](https://arxiv.org/pdf/2604.14635)**

> **作者:** Michael R. Chang; Anna Candotti; Karl von Ellenrieder; Enrico Tomelleri; Marco Camurri
>
> **备注:** 30 pages, 7 figures
>
> **摘要:** We present a curated multi-platform LiDAR reference dataset from an instrumented ICOS forest plot, explicitly designed to support calibration, benchmarking, and integration of 3D structural data with ecological observations and standard allometric models. The dataset integrates UAV-borne laser scanning (ULS) to measure canopy coverage, terrestrial laser scanning (TLS) for detailed stem mapping, and backpack mobile laser scanning (MLS) with real-time SLAM for efficient sub-canopy acquisition. We focus on the control plot with the most complete and internally consistent registration, where TLS point clouds (~333 million points) are complemented by ULS and MLS data capturing canopy and understory strata. Marker-free, SLAM-aware protocols were used to reduce field and processing time, while manual and automated methods were combined. Final products are available in LAZ and E57 formats with UTM coordinates, together with registration reports for reproducibility. The dataset provides a benchmark for testing registration methods, evaluating scanning efficiency, and linking point clouds with segmentation, quantitative structure models, and allometric biomass estimation. By situating the acquisitions at a long-term ICOS site, it is explicitly linked to 3D structure with decades of ecological and flux measurements. More broadly, it illustrates how TLS, MLS, and ULS can be combined for repeated inventories and digital twins of forest ecosystems.
>
---
#### [new 020] DockAnywhere: Data-Efficient Visuomotor Policy Learning for Mobile Manipulation via Novel Demonstration Generation
- **分类: cs.RO**

- **简介: 该论文属于移动操作任务，解决视点泛化问题。通过生成多样演示，提升策略在不同停靠点的泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.15023](https://arxiv.org/pdf/2604.15023)**

> **作者:** Ziyu Shan; Yuheng Zhou; Gaoyuan Wu; Ziheng Ji; Zhenyu Wu; Ziwei Wang
>
> **备注:** Accepted to RA-L
>
> **摘要:** Mobile manipulation is a fundamental capability that enables robots to interact in expansive environments such as homes and factories. Most existing approaches follow a two-stage paradigm, where the robot first navigates to a docking point and then performs fixed-base manipulation using powerful visuomotor policies. However, real-world mobile manipulation often suffers from the view generalization problem due to shifts of docking points. To address this issue, we propose a novel low-cost demonstration generation framework named DockAnywhere, which improves viewpoint generalization under docking variability by lifting a single demonstration to diverse feasible docking configurations. Specifically, DockAnywhere lifts a trajectory to any feasible docking points by decoupling docking-dependent base motions from contact-rich manipulation skills that remain invariant across viewpoints. Feasible docking proposals are sampled under feasibility constraints, and corresponding trajectories are generated via structure-preserving augmentation. Visual observations are synthesized in 3D space by representing the robot and objects as point clouds and applying point-level spatial editing to ensure the consistency of observation and action across viewpoints. Extensive experiments on ManiSkill and real-world platforms demonstrate that DockAnywhere substantially improves policy success rates and easily generalizes to novel viewpoints from unseen docking points during training, significantly enhancing the generalization capability of mobile manipulation policy in real-world deployment.
>
---
#### [new 021] A Hierarchical Spatiotemporal Action Tokenizer for In-Context Imitation Learning in Robotics
- **分类: cs.RO**

- **简介: 该论文属于机器人模仿学习任务，旨在提升上下文模仿学习效果。通过提出分层时空动作分词器，解决动作表示与时空信息融合问题，实现更精确的动作恢复与学习。**

- **链接: [https://arxiv.org/pdf/2604.15215](https://arxiv.org/pdf/2604.15215)**

> **作者:** Fawad Javed Fateh; Ali Shah Ali; Murad Popattia; Usman Nizamani; Andrey Konin; M. Zeeshan Zia; Quoc-Huy Tran
>
> **摘要:** We present a novel hierarchical spatiotemporal action tokenizer for in-context imitation learning. We first propose a hierarchical approach, which consists of two successive levels of vector quantization. In particular, the lower level assigns input actions to fine-grained subclusters, while the higher level further maps fine-grained subclusters to clusters. Our hierarchical approach outperforms the non-hierarchical counterpart, while mainly exploiting spatial information by reconstructing input actions. Furthermore, we extend our approach by utilizing both spatial and temporal cues, forming a hierarchical spatiotemporal action tokenizer, namely HiST-AT. Specifically, our hierarchical spatiotemporal approach conducts multi-level clustering, while simultaneously recovering input actions and their associated timestamps. Finally, extensive evaluations on multiple simulation and real robotic manipulation benchmarks show that our approach establishes a new state-of-the-art performance in in-context imitation learning.
>
---
#### [new 022] Momentum-constrained Hybrid Heuristic Trajectory Optimization Framework with Residual-enhanced DRL for Visually Impaired Scenarios
- **分类: cs.RO**

- **简介: 该论文属于路径优化任务，解决视觉障碍场景下的安全高效导航问题。提出MHHTOF框架，结合启发式采样、MTO和改进DRL，提升轨迹优化的稳定性与适应性。**

- **链接: [https://arxiv.org/pdf/2604.14986](https://arxiv.org/pdf/2604.14986)**

> **作者:** Yuting Zeng; Zhiwen Zheng; Jingya Wang; You Zhou; JiaLing Xiao; Yongbin Yu; Manping Fan; Bo Gong; Liyong Ren
>
> **备注:** 24 pages, 14 figures. arXiv admin note: text overlap with arXiv:2509.15582
>
> **摘要:** Safe and efficient assistive planning for visually impaired scenarios remains challenging, since existing methods struggle with multi-objective optimization, generalization, and interpretability. In response, this paper proposes a Momentum-Constrained Hybrid Heuristic Trajectory Optimization Framework (MHHTOF). To balance multiple objectives of comfort and safety, the framework designs a Heuristic Trajectory Sampling Cluster (HTSC) with a Momentum-Constrained Trajectory Optimization (MTO), which suppresses abrupt velocity and acceleration changes. In addition, a novel residual-enhanced deep reinforcement learning (DRL) module refines candidate trajectories, advancing temporal modeling and policy generalization. Finally, a dual-stage cost modeling mechanism (DCMM) is introduced to regulate optimization, where costs in the Frenet space ensure consistency, and reward-driven adaptive weights in the Cartesian space integrate user preferences for interpretability and user-centric decision-making. Experimental results show that the proposed framework converges in nearly half the iterations of baselines and achieves lower and more stable costs. In complex dynamic scenarios, MHHTOF further demonstrates stable velocity and acceleration curves with reduced risk, confirming its advantages in robustness, safety, and efficiency.
>
---
#### [new 023] An Intelligent Robotic and Bio-Digestor Framework for Smart Waste Management
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于智能废物管理任务，旨在解决传统废物处理效率低的问题。通过集成机器人分拣与生物消化系统，实现高效自动废物处理。**

- **链接: [https://arxiv.org/pdf/2604.14882](https://arxiv.org/pdf/2604.14882)**

> **作者:** Radhika Khatri; Adit Tewari; Nikhil Sharma; M. B. Srinivas
>
> **备注:** 8 pages, 10 figures, submitted to 7th International Conference on Smart Systems and Inventive Technology (ICSSIT 2026)
>
> **摘要:** Rapid urbanization and continuous population growth have made municipal solid waste management increasingly challenging. These challenges highlight the need for smarter and automated waste management solutions. This paper presents the design and evaluation of an integrated waste management framework that combines two connected systems, a robotic waste segregation module and an optimized bio-digestor. The robotic waste segregation system uses a MyCobot 280 Jetson Nano robotic arm along with YOLOv8 object detection and robot operating system (ROS)-based path planning to identify and sort waste in real time. It classifies waste into four different categories with high precision, reducing the need for manual intervention. After segregation, the biodegradable waste is transferred to a bio-digestor system equipped with multiple sensors. These sensors continuously monitor key parameters, including temperature, pH, pressure, and motor revolutions per minute. The Particle Swarm Optimization (PSO) algorithm, combined with a regression model, is used to dynamically adjust system parameters. This intelligent optimization approach ensures stable operation and maximizes digestion efficiency under varying environmental conditions. System testing under dynamic conditions demonstrates a sorting accuracy of 98% along with highly efficient biological conversion. The proposed framework offers a scalable, intelligent, and practical solution for modern waste management, making it suitable for both residential and industrial applications.
>
---
#### [new 024] Dual Pose-Graph Semantic Localization for Vision-Based Autonomous Drone Racing
- **分类: cs.RO**

- **简介: 该论文属于视觉导航任务，解决高速无人机竞速中的定位问题。提出双图结构融合里程计与语义检测，提升定位精度与实时性。**

- **链接: [https://arxiv.org/pdf/2604.15168](https://arxiv.org/pdf/2604.15168)**

> **作者:** David Perez-Saura; Miguel Fernandez-Cortizas; Alvaro J. Gaona; Pascual Campoy
>
> **摘要:** Autonomous drone racing demands robust real-time localization under extreme conditions: high-speed flight, aggressive maneuvers, and payload-constrained platforms that often rely on a single camera for perception. Existing visual SLAM systems, while effective in general scenarios, struggle with motion blur and feature instability inherent to racing dynamics, and do not exploit the structured nature of racing environments. In this work, we present a dual pose-graph architecture that fuses odometry with semantic detections for robust localization. A temporary graph accumulates multiple gate observations between keyframes and optimizes them into a single refined constraint per landmark, which is then promoted to a persistent main graph. This design preserves the information richness of frequent detections while preventing graph growth from degrading real-time performance. The system is designed to be sensor-agnostic, although in this work we validate it using monocular visual-inertial odometry and visual gate detections. Experimental evaluation on the TII-RATM dataset shows a 56% to 74% reduction in ATE compared to standalone VIO, while an ablation study confirms that the dual-graph architecture achieves 10% to 12% higher accuracy than a single-graph baseline at identical computational cost. Deployment in the A2RL competition demonstrated that the system performs real-time onboard localization during flight, reducing the drift of the odometry baseline by up to 4.2 m per lap.
>
---
#### [new 025] Benchmarking Classical Coverage Path Planning Heuristics on Irregular Hexagonal Grids for Maritime Coverage Scenarios
- **分类: cs.RO; cs.AI; math.OC**

- **简介: 该论文研究海洋覆盖路径规划问题，针对不规则六边形网格设计基准测试，评估不同启发式算法的性能，旨在提供可复现的分析平台。**

- **链接: [https://arxiv.org/pdf/2604.15202](https://arxiv.org/pdf/2604.15202)**

> **作者:** Carlos S. Sepúlveda; Gonzalo A. Ruz
>
> **摘要:** Coverage path planning on irregular hexagonal grids is relevant to maritime surveillance, search and rescue and environmental monitoring, yet classical methods are often compared on small ad hoc examples or on rectangular grids. This paper presents a reproducible benchmark of deterministic single-vehicle coverage path planning heuristics on irregular hexagonal graphs derived from synthetic but maritime-motivated areas of interest. The benchmark contains 10,000 Hamiltonian-feasible instances spanning compact, elongated, and irregular morphologies, 17 heuristics from seven families, and a common evaluation protocol covering Hamiltonian success, complete-coverage success, revisits, path length, heading changes, and CPU latency. Across the released dataset, heuristics with explicit shortest-path reconnection solve the relaxed coverage task reliably but almost never produce zero-revisit tours. Exact Depth-First Search confirms that every released instance is Hamiltonian-feasible. The strongest classical Hamiltonian baseline is a Warnsdorff variant that uses an index-based tie-break together with a terminal-inclusive residual-degree policy, reaching 79.0% Hamiltonian success. The dominant design choice is not tie-breaking alone, but how the residual degree is defined when the endpoint is reserved until the final move. This shows that underreported implementation details can materially affect performance on sparse geometric graphs with bottlenecks. The benchmark is intended as a controlled testbed for heuristic analysis rather than as a claim of operational optimality at fleet scale.
>
---
#### [new 026] A Nonasymptotic Theory of Gain-Dependent Error Dynamics in Behavior Cloning
- **分类: cs.RO; cs.AI; math.OC**

- **简介: 该论文研究行为克隆中的误差动态，解决控制器增益对失败概率的影响问题，提出非渐近理论分析。**

- **链接: [https://arxiv.org/pdf/2604.14484](https://arxiv.org/pdf/2604.14484)**

> **作者:** Junghoon Seo
>
> **摘要:** Behavior cloning (BC) policies on position-controlled robots inherit the closed-loop response of the underlying PD controller, yet the effect of controller gains on BC failure lacks a nonasymptotic theory. We show that independent sub-Gaussian action errors propagate through the gain-dependent closed-loop dynamics to yield sub-Gaussian position errors whose proxy matrix $X_\infty(K)$ governs the failure tail. The probability of horizon-$T$ task failure factorizes into a gain-dependent amplification index $\Gamma_T(K)$ and the validation loss plus a generalization slack, so training loss alone cannot predict closed-loop performance. Under shape-preserving upper-bound structural assumptions the proxy admits the scalar bound $X_\infty(K)\preceq\Psi(K)\bar X$ with $\Psi(K)$ decomposed into label difficulty, injection strength, and contraction, ranking the four canonical regimes with compliant-overdamped (CO) tightest, stiff-underdamped (SU) loosest, and the stiff-overdamped versus compliant-underdamped ordering system-dependent. For the canonical scalar second-order PD system the closed-form continuous-time stationary variance $X_\infty^{\mathrm{c}}(\alpha,\beta)=\sigma^2\alpha/(2\beta)$ is strictly monotone in stiffness and damping over the entire stable orthant, covering both underdamped and overdamped regimes, and the exact zero-order-hold (ZOH) discretization inherits this monotonicity. The analysis provides the first nonasymptotic explanation of the empirical finding that compliant, overdamped controllers improve BC success rates.
>
---
#### [new 027] NEAT-NC: NEAT guided Navigation Cells for Robot Path Planning
- **分类: cs.RO; cs.AI; cs.NE**

- **简介: 该论文属于机器人路径规划任务，旨在提升NEAT算法在动态环境中的表现。通过引入生物导航细胞，演化递归神经网络以实现更优路径规划。**

- **链接: [https://arxiv.org/pdf/2604.15076](https://arxiv.org/pdf/2604.15076)**

> **作者:** Hibatallah Meliani; Khadija Slimani; Samira Khoulji
>
> **备注:** To appear in short form in Genetic and Evolutionary Computation Conference (GECCO '26), 2026
>
> **摘要:** To navigate a space, the brain makes an internal representation of the environment using different cells such as place cells, grid cells, head direction cells, border cells, and speed cells. All these cells, along with sensory inputs, enable an organism to explore the space around it. Inspired by these biological principles, we developed NEATNC, a Neuro-Evolution of Augmenting Topology guided Navigation Cells. The goal of the paper is to improve NEAT algorithm performance in path planning in dynamic environments using spatial cognitive cells. This approach uses navigation cells as inputs and evolves recurrent neural networks, representing the hippocampus part of the brain. The performance of the proposed algorithm is evaluated in different static and dynamic scenarios. This study highlights NEAT's adaptability to complex and different environments, showcasing the utility of biological theories. This suggests that our approach is well-suited for real-time dynamic path planning for robotics and games.
>
---
#### [new 028] World-Value-Action Model: Implicit Planning for Vision-Language-Action Systems
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于视觉-语言-动作系统任务，解决长周期决策问题。提出WAV模型，通过隐式规划提升复杂任务性能。**

- **链接: [https://arxiv.org/pdf/2604.14732](https://arxiv.org/pdf/2604.14732)**

> **作者:** Runze Li; Hongyin Zhang; Junxi Jin; Qixin Zeng; Zifeng Zhuang; Yiqi Tang; Shangke Lyu; Donglin Wang
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a promising paradigm for building embodied agents that ground perception and language into action. However, most existing approaches rely on direct action prediction, lacking the ability to reason over long-horizon trajectories and evaluate their consequences, which limits performance in complex decision-making tasks. In this work, we introduce World-Value-Action (WAV) model, a unified framework that enables implicit planning in VLA systems. Rather than performing explicit trajectory optimization, WAV model learn a structured latent representation of future trajectories conditioned on visual observations and language instructions. A learned world model predicts future states, while a trajectory value function evaluates their long-horizon utility. Action generation is then formulated as inference in this latent space, where the model progressively concentrates probability mass on high-value and dynamically feasible trajectories. We provide a theoretical perspective showing that planning directly in action space suffers from an exponential decay in the probability of feasible trajectories as the horizon increases. In contrast, latent-space inference reshapes the search distribution toward feasible regions, enabling efficient long-horizon decision making. Extensive simulations and real-world experiments demonstrate that the WAV model consistently outperforms state-of-the-art methods, achieving significant improvements in task success rate, generalization ability, and robustness, especially in long-horizon and compositional scenarios.
>
---
#### [new 029] HRDexDB: A Large-Scale Dataset of Dexterous Human and Robotic Hand Grasps
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出HRDexDB，一个包含人类和机器人手高精度抓取数据的多模态基准数据集，用于解决跨域灵巧操作与多模态策略学习问题。**

- **链接: [https://arxiv.org/pdf/2604.14944](https://arxiv.org/pdf/2604.14944)**

> **作者:** Jongbin Lim; Taeyun Ha; Mingi Choi; Jisoo Kim; Byungjun Kim; Subin Jeon; Hanbyul Joo
>
> **摘要:** We present HRDexDB, a large-scale, multi-modal dataset of high-fidelity dexterous grasping sequences featuring both human and diverse robotic hands. Unlike existing datasets, HRDexDB provides a comprehensive collection of grasping trajectories across human hands and multiple robot hand embodiments, spanning 100 diverse objects. Leveraging state-of-the-art vision methods and a new dedicated multi-camera system, our HRDexDB offers high-precision spatiotemporal 3D ground-truth motion for both the agent and the manipulated object. To facilitate the study of physical interaction, HRDexDB includes high-resolution tactile signals, synchronized multi-view video, and egocentric video streams. The dataset comprises 1.4K grasping trials, encompassing both successes and failures, each enriched with visual, kinematic, and tactile modalities. By providing closely aligned captures of human dexterity and robotic execution on the same target objects under comparable grasping motions, HRDexDB serves as a foundational benchmark for multi-modal policy learning and cross-domain dexterous manipulation.
>
---
#### [new 030] CT-VIR: Continuous-Time Visual-Inertial-Ranging Fusion for Indoor Localization with Sparse Anchors
- **分类: cs.RO**

- **简介: 该论文属于室内定位任务，旨在解决VIO长期精度下降和UWB依赖密集锚点的问题。提出一种基于样条的连续时间VIR融合方法，提升定位精度与效率。**

- **链接: [https://arxiv.org/pdf/2604.14545](https://arxiv.org/pdf/2604.14545)**

> **作者:** Yu-An Liu; Li Zhang
>
> **摘要:** Visual-inertial odometry (VIO) is widely used for mobile robot localization, but its long-term accuracy degrades without global constraints. Incorporating ranging sensors such as ultra-wideband (UWB) can mitigate drift; however, high-accuracy ranging usually requires well-deployed anchors, which is difficult to ensure in narrow or low-power environments. Moreover, most existing visual-inertial-ranging (VIR) fusion methods rely on discrete time-based filtering or optimization, making it difficult to balance positioning accuracy, trajectory consistency, and fusion efficiency under asynchronous multi-sensor sampling. To address these issues, we propose a spline-based continuous-time state estimation method for VIR fusion localization. In the preprocessing stage, VIO motion priors and UWB ranging measurements are used to construct virtual anchors and reject outliers, thereby alleviating geometric degeneration and improving range reliability. In the estimation stage, the pose trajectory is parameterized in continuous time using a B-spline, while inertial, visual, and ranging constraints are formulated as factors in a sliding-window graph. The spline control points, together with a small set of auxiliary parameters, are then jointly optimized to obtain a continuous-time trajectory estimate. Evaluations on public datasets and real-world experiments demonstrate the effectiveness and practical potential of the proposed approach.
>
---
#### [new 031] Efficient closed-form approaches for pose estimation using Sylvester forms
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于姿态估计任务，旨在解决实时计算机视觉中的非线性最小二乘问题。通过利用Sylvester形式构造高效闭式解法，降低计算复杂度，提升求解速度。**

- **链接: [https://arxiv.org/pdf/2604.14747](https://arxiv.org/pdf/2604.14747)**

> **作者:** Jana Vráblíková; Ezio Malis; Laurent Busé
>
> **摘要:** Solving non-linear least-squares problem for pose estimation (rotation and translation) is often a time consuming yet fundamental problem in several real-time computer vision applications. With an adequate rotation parametrization, the optimization problem can be reduced to the solution of a~system of polynomial equations and solved in closed form. Recent advances in efficient closed form solvers utilizing resultant matrices have shown a promising research direction to decrease the computation time while preserving the estimation accuracy. In this paper, we propose a new class of resultant-based solvers that exploit Sylvester forms to further reduce the complexity of the resolution. We demonstrate that our proposed methods are numerically as accurate as the state-of-the-art solvers, and outperform them in terms of computational time. We show that this approach can be applied for pose estimation in two different types of problems: estimating a pose from 3D to 3D correspondences, and estimating a pose from 3D points to 2D points correspondences.
>
---
#### [new 032] R3D: Revisiting 3D Policy Learning
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D策略学习任务，旨在解决训练不稳定和过拟合问题。通过引入3D数据增强和新架构，提升模型泛化能力与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.15281](https://arxiv.org/pdf/2604.15281)**

> **作者:** Zhengdong Hong; Shenrui Wu; Haozhe Cui; Boyi Zhao; Ran Ji; Yiyang He; Hangxing Zhang; Zundong Ke; Jun Wang; Guofeng Zhang; Jiayuan Gu
>
> **摘要:** 3D policy learning promises superior generalization and cross-embodiment transfer, but progress has been hindered by training instabilities and severe overfitting, precluding the adoption of powerful 3D perception models. In this work, we systematically diagnose these failures, identifying the omission of 3D data augmentation and the adverse effects of Batch Normalization as primary causes. We propose a new architecture coupling a scalable transformer-based 3D encoder with a diffusion decoder, engineered specifically for stability at scale and designed to leverage large-scale pre-training. Our approach significantly outperforms state-of-the-art 3D baselines on challenging manipulation benchmarks, establishing a new and robust foundation for scalable 3D imitation learning. Project Page: this https URL
>
---
#### [new 033] ADAPT: Benchmarking Commonsense Planning under Unspecified Affordance Constraints
- **分类: cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文属于智能体规划任务，解决现实环境中对象可操作性未明确时的推理问题。提出ADAPT模块，增强规划器的可操作性推理能力，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2604.14902](https://arxiv.org/pdf/2604.14902)**

> **作者:** Pei-An Chen; Yong-Ching Liang; Jia-Fong Yeh; Hung-Ting Su; Yi-Ting Chen; Min Sun; Winston Hsu
>
> **摘要:** Intelligent embodied agents should not simply follow instructions, as real-world environments often involve unexpected conditions and exceptions. However, existing methods usually focus on directly executing instructions, without considering whether the target objects can actually be manipulated, meaning they fail to assess available affordances. To address this limitation, we introduce DynAfford, a benchmark that evaluates embodied agents in dynamic environments where object affordances may change over time and are not specified in the instruction. DynAfford requires agents to perceive object states, infer implicit preconditions, and adapt their actions accordingly. To enable this capability, we introduce ADAPT, a plug-and-play module that augments existing planners with explicit affordance reasoning. Experiments demonstrate that incorporating ADAPT significantly improves robustness and task success across both seen and unseen environments. We also show that a domain-adapted, LoRA-finetuned vision-language model used as the affordance inference backend outperforms a commercial LLM (GPT-4o), highlighting the importance of task-aligned affordance grounding.
>
---
#### [new 034] Separation is Optimal for LQR under Intermittent Feedback
- **分类: math.OC; cs.IT; cs.MA; cs.RO; eess.SY**

- **简介: 该论文研究通信受限下的LQR控制问题，证明分离原理成立，并提出最优调度策略与控制器设计。**

- **链接: [https://arxiv.org/pdf/2603.27833](https://arxiv.org/pdf/2603.27833)**

> **作者:** Abdullah Y. Etcibasi; C. Emre Koksal; Eylem Ekici
>
> **摘要:** In this work, we first prove that the separation principle holds for communication-constrained LQR problems under i.i.d. zero-mean disturbances with a symmetric distribution. We then solve the dynamic programming problem and show that the optimal scheduling policy is a symmetric threshold rule on the accumulated disturbance since the most recent update, while the optimal controller is a discounted linear feedback law independent of the scheduling policy.
>
---
#### [new 035] Energy-based Regularization for Learning Residual Dynamics in Neural MPC for Omnidirectional Aerial Robots
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于控制理论领域，旨在解决神经模型对物理特性理解不足的问题。通过引入能量正则化损失函数，提升神经MPC在全向飞行器上的控制精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2604.14678](https://arxiv.org/pdf/2604.14678)**

> **作者:** Johannes Kübel; Henrik Krauss; Jinjie Li; Moju Zhao
>
> **摘要:** Data-driven Model Predictive Control (MPC) has lately been the core research subject in the field of control theory. The combination of an optimal control framework with deep learning paradigms opens up the possibility to accurately track control tasks without the need for complex analytical models. However, the system dynamics are often nuanced and the neural model lacks the potential to understand physical properties such as inertia and conservation of energy. In this work, we propose a novel energy-based regularization loss function which is applied to the training of a neural model that learns the residual dynamics of an omnidirectional aerial robot. Our energy-based regularization encourages the neural network to cause control corrections that stabilize the energy of the system. The residual dynamics are integrated into the MPC framework and improve the positional mean absolute error (MAE) over three real-world experiments by 23% compared to an analytical MPC. We also compare our method to a standard neural MPC implementation without regularization and primarily achieve a significantly increased flight stability implicitly due to the energy regularization and up to 15% lower MAE. Our code is available under: this https URL.
>
---
## 更新

#### [replaced 001] XRZero-G0: Pushing the Frontier of Dexterous Robotic Manipulation with Interfaces, Quality and Ratios
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决高质量演示数据获取难题。通过XRZero-G0系统，提升数据收集效率与质量，实现低成本、高效果的机器人学习。**

- **链接: [https://arxiv.org/pdf/2604.13001](https://arxiv.org/pdf/2604.13001)**

> **作者:** James Wang; Primo Pu; Zephyr Fung; Alex Wang; Sam Wang; Bender Deng; Kevin Wang; Zivid Liu; Chris Pan; Panda Yang; Andy Zhai; Lucy Liang; Shalfun Li; Johnny Sun; Jacky Xu; Will Tian; Kai Yan; Kohler Ye; Scott Li; Qian Wang; Roy Gan; Hao Wang
>
> **备注:** Technical Report
>
> **摘要:** The acquisition of high-quality, action-aligned demonstration data remains a fundamental bottleneck in scaling foundation models for dexterous robot manipulation. Although robot-free human demonstrations (e.g., the UMI paradigm) offer a scalable alternative to traditional teleoperation, current systems are constrained by sub-optimal hardware ergonomics, open-loop workflows, and a lack of systematic data-mixing strategies. To address these limitations, we present XRZero-G0, a hardware-software co-designed system for embodied data collection and policy learning. The system features an ergonomic, virtual reality interface equipped with a top-view camera and dual specialized grippers to directly improve collection efficiency. To ensure dataset reliability, we propose a closed-loop collection, inspection, training, and evaluation pipeline for non-proprioceptive data. This workflow achieves an 85% data validity rate and establishes a transparent mechanism for quality control. Furthermore, we investigate the empirical scaling behaviors and optimal mixing ratios of robot-free data. Extensive experiments indicate that combining a minimal volume of real-robot data with large-scale robot-free data (e.g., a 10:1 ratio) achieves performance comparable to exclusively real-robot datasets, while reducing acquisition costs by a factor of twenty. Utilizing XRZero-G0, we construct a 2,000-hour robot-free dataset that enables zero-shot cross-embodiment transfer to a target physical robot, demonstrating a highly scalable methodology for generalized real-world this http URL project repository: this https URL
>
---
#### [replaced 002] Simple but Stable, Fast and Safe: Achieve End-to-end Control by High-Fidelity Differentiable Simulation
- **分类: cs.RO**

- **简介: 该论文属于无人机控制任务，解决高速飞行中轨迹动态不可行问题。通过强化学习直接从深度图像生成低级指令，实现端到端控制，提升稳定性和安全性。**

- **链接: [https://arxiv.org/pdf/2604.10548](https://arxiv.org/pdf/2604.10548)**

> **作者:** Fanxing Li; Shengyang Wang; Yuxiang Huang; Fangyu Sun; Shuyu Wu; Yufei Yan; Danping Zou; Wenxian Yu
>
> **摘要:** Obstacle avoidance is a fundamental vision-based task essential for enabling quadrotors to perform advanced applications. When planning the trajectory, existing approaches both on optimization and learning typically regard quadrotor as a point-mass model, giving path or velocity commands then tracking the commands by outer-loop controller. However, at high speeds, planned trajectories sometimes become dynamically infeasible in actual flight, which beyond the capacity of controller. In this paper, we propose a novel end-to-end policy that directly maps depth images to low-level bodyrate commands by reinforcement learning via differentiable simulation. The high-fidelity simulation in training after parameter identification significantly reduces all the gaps between training, simulation and real world. Analytical process by differentiable simulation provides accurate gradient to ensure efficiently training the low-level policy without expert guidance. The policy employs a lightweight and the most simple inference pipeline that runs without explicit mapping, backbone networks, primitives, recurrent structures, or backend controllers, nor curriculum or privileged guidance. By inferring low-level command directly to the hardware controller, the method enables full flight envelope control and avoids the dynamic-infeasible this http URL results demonstrate that the proposed approach achieves the highest success rate and the lowest jerk among state-of-the-art baselines across multiple benchmarks. The policy also exhibits strong generalization, successfully deploying zero-shot in unseen, outdoor environments while reaching speeds of up to 7.5m/s as well as stably flying in the super-dense forest. This work is released at this https URL.
>
---
#### [replaced 003] Constrained Decoding for Safe Robot Navigation Foundation Models
- **分类: cs.RO; cs.LG; cs.LO**

- **简介: 该论文属于机器人导航任务，旨在解决基础模型缺乏行为正确性保障的问题。提出SafeDec框架，在解码阶段通过STL约束确保动作安全，无需重新训练。**

- **链接: [https://arxiv.org/pdf/2509.01728](https://arxiv.org/pdf/2509.01728)**

> **作者:** Parv Kapoor; Akila Ganlath; Michael Clifford; Changliu Liu; Sebastian Scherer; Eunsuk Kang
>
> **摘要:** Recent advances in the development of robotic foundation models have led to promising end-to-end and general-purpose capabilities in robotic systems. Trained on vast datasets of simulated and real-world trajectories, these policies map multimodal observations directly to action sequences for physical execution. Despite promising real-world capabilities, these models are still data-driven and, therefore, lack explicit notions of behavioral correctness. We address this gap by introducing SafeDec, a constrained decoding framework for autoregressive, transformer-based robot navigation foundation models that enforces safety specifications expressed as Signal Temporal Logic (STL) formulas. Our method ensures that generated actions provably satisfy STL specifications under assumed dynamics at runtime without retraining while remaining agnostic of the underlying policy. We evaluate SafeDec on tasks from the CHORES benchmark for state-of-the-art embodied navigation policies across hundreds of procedurally generated environments and show that our decoding-time interventions are useful not only for filtering unsafe actions but also for conditional action generation. Videos are available at this http URL
>
---
#### [replaced 004] cuRoboV2: Dynamics-Aware Motion Generation with Depth-Fused Distance Fields for High-DoF Robots
- **分类: cs.RO**

- **简介: 该论文提出cuRoboV2，解决高自由度机器人运动生成问题，集成轨迹优化、感知与计算框架，提升安全性、效率与可扩展性。**

- **链接: [https://arxiv.org/pdf/2603.05493](https://arxiv.org/pdf/2603.05493)**

> **作者:** Balakumar Sundaralingam; Adithyavairavan Murali; Stan Birchfield
>
> **备注:** cuRoboV2 Technical Report with code url
>
> **摘要:** Effective robot autonomy requires motion generation that is safe, feasible, and reactive. Current methods are fragmented: fast planners output physically unexecutable trajectories, reactive controllers struggle with high-fidelity perception, and existing solvers fail on high-DoF systems. We present cuRoboV2, a unified framework with three key innovations: (1) B-spline trajectory optimization that enforces smoothness and torque limits; (2) a GPU-native TSDF/ESDF perception pipeline that generates dense signed distance fields covering the full workspace, unlike existing methods that only provide distances within sparsely allocated blocks, up to 10x faster and in 8x less memory than the state-of-the-art at manipulation scale, with up to 99% collision recall; and (3) scalable GPU-native whole-body computation, namely topology-aware kinematics, differentiable inverse dynamics, and map-reduce self-collision, that achieves up to 61x speedup while also extending to high-DoF humanoids (where previous GPU implementations fail). On benchmarks, cuRoboV2 achieves 99.7% success under 3kg payload (where baselines achieve only 72--77%), 99.6% collision-free IK on a 48-DoF humanoid (where prior methods fail entirely), and 89.5% retargeting constraint satisfaction (vs. 61% for PyRoki); these collision-free motions yield locomotion policies with 21% lower tracking error than PyRoki and 12x lower cross-seed variance than GMR. A ground-up codebase redesign for discoverability enabled LLM coding assistants to author up to 73% of new modules, including hand-optimized CUDA kernels, demonstrating that well-structured robotics code can unlock productive human-LLM collaboration. Together, these advances provide a unified, dynamics-aware motion generation stack that scales from single-arm manipulators to full humanoids. Code is available at this https URL.
>
---
#### [replaced 005] TwinOR: Photorealistic Digital Twins of Dynamic Operating Rooms for Embodied AI Research
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出TwinOR，用于构建动态手术室的高保真数字孪生，解决真实手术环境难以安全实验的问题。属于 embodied AI 领域，旨在提供可控制的仿真环境以支持智能手术系统研究。**

- **链接: [https://arxiv.org/pdf/2511.07412](https://arxiv.org/pdf/2511.07412)**

> **作者:** Han Zhang; Yiqing Shen; Roger D. Soberanis-Mukul; Ankita Ghosh; Hao Ding; Lalithkumar Seenivasan; Jose L. Porras; Zhekai Mao; Chenjia Li; Wenjie Xiao; Lonny Yarmus; Angela Christine Argento; Masaru Ishii; Mathias Unberath
>
> **摘要:** Developing embodied AI for intelligent surgical systems requires safe, controllable environments for continual learning and evaluation. However, safety regulations and operational constraints in operating rooms (ORs) limit agents from freely perceiving and interacting in realistic settings. Digital twins provide high-fidelity, risk-free environments for exploration and training. How we may create dynamic digital representations of ORs that capture relevant spatial, visual, and behavioral complexity remains an open challenge. We introduce TwinOR, a real-to-sim infrastructure for constructing photorealistic and dynamic digital twins of ORs. The system reconstructs static geometry and continuously models human and equipment motion. The static and dynamic components are fused into an immersive 3D environment that supports controllable simulation and facilitates future embodied exploration. The proposed framework reconstructs complete OR geometry with centimeter-level accuracy while preserving dynamic interaction across surgical workflows. In our experiments, TwinOR synthesizes stereo and monocular RGB streams as well as depth observations for geometry understanding and visual localization tasks. Models such as FoundationStereo and ORB-SLAM3 evaluated on TwinOR-synthesized data achieve performance within their reported accuracy ranges on real-world indoor datasets, demonstrating that TwinOR provides sensor-level realism sufficient for emulating real-world perception and localization challenge. By establishing a perception-grounded real-to-sim pipeline, TwinOR enables the automatic construction of dynamic, photorealistic digital twins of ORs. As a safe and scalable environment for experimentation, TwinOR opens new opportunities for translating embodied intelligence from simulation to real-world clinical environments.
>
---
#### [replaced 006] An Active Perception Game for Robust Exploration
- **分类: cs.RO**

- **简介: 该论文属于机器人主动感知任务，解决信息估计不准确问题，通过博弈论方法提升感知系统性能。**

- **链接: [https://arxiv.org/pdf/2404.00769](https://arxiv.org/pdf/2404.00769)**

> **作者:** Siming He; Yuezhan Tao; Igor Spasojevic; Vijay Kumar; Pratik Chaudhari
>
> **摘要:** Active perception approaches select future viewpoints by using some estimate of the information gain. An inaccurate estimate can be detrimental in critical situations, e.g., locating a person in distress. However the true information gained can only be calculated post hoc, i.e., after the observation is realized. We present an approach to estimate the discrepancy between the estimated information gain (which is the expectation over putative future observations while neglecting correlations among them) and the true information gain. The key idea is to analyze the mathematical relationship between active perception and the estimation error of the information gain in a game-theoretic setting. Using this, we develop an online estimation approach that achieves sub-linear regret (in the number of time-steps) for the estimation of the true information gain and reduces the sub-optimality of active perception systems. We demonstrate our approach for active perception using a comprehensive set of experiments on: (a) different types of environments, including a quadrotor in a photorealistic simulation, real-world robotic data, and real-world experiments with ground robots exploring indoor and outdoor scenes; (b) different types of robotic perception data; and (c) different map representations. On average, our approach reduces information gain estimation errors by 42%, increases the information gain by 7%, PSNR by 5%, and semantic accuracy (measured as the number of objects that are localized correctly) by 6%. In real-world experiments with a Jackal ground robot, our approach demonstrated complex trajectories to explore occluded regions.
>
---
#### [replaced 007] Generative Models and Connected and Automated Vehicles: A Survey in Exploring the Intersection of Transportation and AI
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 论文属于综述任务，探讨生成模型与自动驾驶车辆的结合，解决如何提升自动驾驶预测、模拟和决策的问题，分析其优势与挑战。**

- **链接: [https://arxiv.org/pdf/2403.10559](https://arxiv.org/pdf/2403.10559)**

> **作者:** Bo Shu; Yiting Zhang; Saisai Hu; Dong Shu
>
> **摘要:** This report investigates the history and impact of Generative Models and Connected and Automated Vehicles (CAVs), two groundbreaking forces pushing progress in technology and transportation. By focusing on the application of generative models within the context of CAVs, the study aims to unravel how this integration could enhance predictive modeling, simulation accuracy, and decision-making processes in autonomous vehicles. This thesis discusses the benefits and challenges of integrating generative models and CAV technology in transportation. It aims to highlight the progress made, the remaining obstacles, and the potential for advancements in safety and innovation.
>
---
#### [replaced 008] Flow with the Force Field: Learning 3D Compliant Flow Matching Policies from Force and Demonstration-Guided Simulation Data
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究机器人操作中的接触任务，解决传统视觉-运动策略忽略物理接触的问题。通过结合力信息和人类示范，在仿真中生成数据，提升策略的接触适应能力。**

- **链接: [https://arxiv.org/pdf/2510.02738](https://arxiv.org/pdf/2510.02738)**

> **作者:** Tianyu Li; Yihan Li; Zizhe Zhang; Nadia Figueroa
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** While visuomotor policy has made advancements in recent years, contact-rich tasks still remain a challenge. Robotic manipulation tasks that require continuous contact demand explicit handling of compliance and force. However, most visuomotor policies ignore compliance, overlooking the importance of physical interaction with the real world, often leading to excessive contact forces or fragile behavior under uncertainty. Introducing force information into vision-based imitation learning could help improve awareness of contacts, but could also require a lot of data to perform well. One remedy for data scarcity is to generate data in simulation, yet computationally taxing processes are required to generate data good enough not to suffer from the Sim2Real gap. In this work, we introduce a framework for generating force-informed data in simulation, instantiated by a single human demonstration, and show how coupling with a compliant policy improves the performance of a visuomotor policy learned from synthetic data. We validate our approach on real-robot tasks, including non-prehensile block flipping and a bi-manual object moving, where the learned policy exhibits reliable contact maintenance and adaptation to novel conditions. Project Website: this https URL
>
---
#### [replaced 009] Reference-Free Sampling-Based Model Predictive Control
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人运动控制任务，旨在无需预设步态或接触序列实现自适应运动。通过优化高阶目标，提出一种基于采样的模型预测控制方法，实现多样化的运动模式和实时控制。**

- **链接: [https://arxiv.org/pdf/2511.19204](https://arxiv.org/pdf/2511.19204)**

> **作者:** Fabian Schramm; Pierre Fabre; Nicolas Perrin-Gilbert; Justin Carpentier
>
> **备注:** Accepted to the 2026 IEEE International Conference on Robotics and Automation (ICRA), Vienna, Austria
>
> **摘要:** We present a sampling-based model predictive control (MPC) framework that enables emergent locomotion without relying on handcrafted gait patterns or predefined contact sequences. Our method discovers diverse motion patterns, ranging from trotting to galloping, robust standing policies, jumping, and handstand balancing, purely through the optimization of high-level objectives. Building on model predictive path integral (MPPI), we propose a cubic Hermite spline parameterization that operates on position and velocity control points. Our approach enables contact-making and contact-breaking strategies that adapt automatically to task requirements, requiring only a limited number of sampled trajectories. This sample efficiency enables real-time control on standard CPU hardware, eliminating the GPU acceleration typically required by other state-of-the-art MPPI methods. We validate our approach on the Go2 quadrupedal robot, demonstrating a range of emergent gaits and basic jumping capabilities. In simulation, we further showcase more complex behaviors, such as backflips, dynamic handstand balancing and locomotion on a Humanoid, all without requiring reference tracking or offline pre-training.
>
---
#### [replaced 010] Multi-Modal Manipulation via Multi-Modal Policy Consensus
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人操作任务，解决多模态信息融合问题。通过分解策略为多个扩散模型并使用路由网络自适应组合，提升多模态推理能力与系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.23468](https://arxiv.org/pdf/2509.23468)**

> **作者:** Haonan Chen; Jiaming Xu; Hongyu Chen; Kaiwen Hong; Binghao Huang; Chaoqi Liu; Jiayuan Mao; Yunzhu Li; Yilun Du; Katherine Driggs-Campbell
>
> **备注:** 8 pages, 7 figures. Project website: this https URL
>
> **摘要:** Effectively integrating diverse sensory modalities is crucial for robotic manipulation. However, the typical approach of feature concatenation is often suboptimal: dominant modalities such as vision can overwhelm sparse but critical signals like touch in contact-rich tasks, and monolithic architectures cannot flexibly incorporate new or missing modalities without retraining. Our method factorizes the policy into a set of diffusion models, each specialized for a single representation (e.g., vision or touch), and employs a router network that learns consensus weights to adaptively combine their contributions, enabling incremental of new representations. We evaluate our approach on simulated manipulation tasks in {RLBench}, as well as real-world tasks such as occluded object picking, in-hand spoon reorientation, and puzzle insertion, where it significantly outperforms feature-concatenation baselines on scenarios requiring multimodal reasoning. Our policy further demonstrates robustness to physical perturbations and sensor corruption. We further conduct perturbation-based importance analysis, which reveals adaptive shifts between modalities.
>
---
#### [replaced 011] Sixth-Sense: Self-Supervised Learning of Spatial Awareness of Humans from a Planar Lidar
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于目标检测任务，旨在解决机器人在复杂环境中对人定位不准确的问题。通过自监督学习，利用1D LiDAR数据检测人体并估计姿态，提升机器人环境感知能力。**

- **链接: [https://arxiv.org/pdf/2502.21029](https://arxiv.org/pdf/2502.21029)**

> **作者:** Simone Arreghini; Nicholas Carlotti; Mirko Nava; Antonio Paolillo; Alessandro Giusti
>
> **摘要:** Reliable localization of people is fundamental for service and social robots that must operate in close interaction with humans. State-of-the-art human detectors often rely on RGB-D cameras or costly 3D LiDARs. However, most commercial robots are equipped with cameras with a narrow field of view, leaving them unaware of users approaching from other directions, or inexpensive 1D LiDARs whose readings are hard to interpret. To address these limitations, we propose a self-supervised approach to detect humans and estimate their 2D pose from 1D LiDAR data, using detections from an RGB-D camera as supervision. Trained on 70 minutes of autonomously collected data, our model detects humans omnidirectionally in unseen environments with 71% precision, 80% recall, and mean absolute errors of 13cm in distance and 44° in orientation, measured against ground truth data. Beyond raw detection accuracy, this capability is relevant for robots operating in shared public spaces, where omnidirectional awareness of nearby people is crucial for safe navigation, appropriate approach behavior, and timely human-robot interaction initiation using low-cost, privacy-preserving sensing. Deployment in two additional public environments further suggests that the approach can serve as a practical wide-FOV awareness layer for socially aware service robotics.
>
---
#### [replaced 012] Towards Deploying VLA without Fine-Tuning: Plug-and-Play Inference-Time VLA Policy Steering via Embodied Evolutionary Diffusion
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人视觉-语言-动作（VLA）任务，旨在解决预训练VLA模型在实际部署中性能下降的问题。提出VLA-Pilot方法，在不微调的情况下实现零样本部署和提升成功率。**

- **链接: [https://arxiv.org/pdf/2511.14178](https://arxiv.org/pdf/2511.14178)**

> **作者:** Zhuo Li; Junjia Liu; Zhipeng Dong; Tao Teng; Quentin Rouxel; Darwin Caldwell; Fei Chen
>
> **备注:** 9 pages, 8 figures, submitted to IEEE RA-L
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated significant potential in real-world robotic manipulation. However, pre-trained VLA policies still suffer from substantial performance degradation during downstream deployment. Although fine-tuning can mitigate this issue, its reliance on costly demonstration collection and intensive computation makes it impractical in real-world settings. In this work, we introduce VLA-Pilot, a plug-and-play inference-time policy steering method for zero-shot deployment of pre-trained VLA without any additional fine-tuning or data collection. We evaluate VLA-Pilot on six real-world downstream manipulation tasks across two distinct robotic embodiments, encompassing both in-distribution and out-of-distribution scenarios. Experimental results demonstrate that VLA-Pilot substantially boosts the success rates of off-the-shelf pre-trained VLA policies, enabling robust zero-shot generalization to diverse tasks and embodiments. Experimental videos and code are available at: this https URL.
>
---
#### [replaced 013] A Robust Approach for LiDAR-Inertial Odometry Without Sensor-Specific Modeling
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决LiDAR与IMU融合的里程计问题。提出一种无需传感器特异性建模的方法，通过简化运动模型和直接扫描配准提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.06593](https://arxiv.org/pdf/2509.06593)**

> **作者:** Meher V.R. Malladi; Tiziano Guadagnino; Luca Lobefaro; Cyrill Stachniss
>
> **摘要:** Accurate odometry is a critical component in a robotic navigation stack, and subsequent modules such as planning and control often rely on an estimate of the robot's motion. Sensor-based odometry approaches should be robust across sensor types and deployable in different target domains, from solid-state LiDARs mounted on cars in urban-driving scenarios to spinning LiDARs on handheld packages used in unstructured natural environments. In this paper, we propose a robust LiDAR-inertial odometry system that does not rely on sensor-specific modeling. Sensor fusion techniques for LiDAR and inertial measurement unit (IMU) data typically integrate IMU data iteratively in a Kalman filter or use pre-integration in a factor graph framework, combined with LiDAR scan matching often exploiting some form of feature extraction. We propose an alternative strategy that only requires a simplified motion model for IMU integration and directly registers LiDAR scans in a scan-to-map approach. Our approach allows us to impose a novel regularization on the LiDAR registration, improving the overall odometry performance. We detail extensive experiments on a number of datasets covering a wide array of commonly used robotic sensors and platforms. We show that our approach works with the exact same configuration in all these scenarios, demonstrating its robustness. We have open-sourced our implementation so that the community can build further on our work and use it in their navigation stacks.
>
---
#### [replaced 014] Hoi! -- A Multimodal Dataset for Force-Grounded, Cross-View Articulated Manipulation
- **分类: cs.RO**

- **简介: 该论文提出一个用于力感知、跨视角操作的多模态数据集，解决人机交互理解问题，涵盖多种操作方式和传感信息。**

- **链接: [https://arxiv.org/pdf/2512.04884](https://arxiv.org/pdf/2512.04884)**

> **作者:** Tim Engelbracht; René Zurbrügg; Matteo Wohlrapp; Martin Büchner; Abhinav Valada; Marc Pollefeys; Hermann Blum; Zuria Bauer
>
> **摘要:** We present a dataset for force-grounded, cross-view articulated manipulation that couples what is seen with what is done and what is felt during real human interaction. The dataset contains 3048 sequences across 381 articulated objects in 38 environments. Each object is operated in four embodiments - (i) human hand, (ii) human hand with a wrist-mounted camera, (iii) handheld UMI gripper, and (iv) a custom Hoi! gripper, where the tool embodiment provides end-effector forces and tactile sensing. Our dataset offers a holistic view of interaction understanding from video, enabling researchers to evaluate how well methods transfer between human and robotic viewpoints, but also investigate underexplored modalities such as interaction forces. The Project Website can be found at this https URL.
>
---
#### [replaced 015] AFFORD2ACT: Affordance-Guided Automatic Keypoint Selection for Generalizable and Lightweight Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出AFFORD2ACT，解决机器人操作中关键点选择问题。通过语义关键点提取与Transformer策略学习，提升操作的泛化性和轻量化，实现高效实时控制。**

- **链接: [https://arxiv.org/pdf/2510.01433](https://arxiv.org/pdf/2510.01433)**

> **作者:** Anukriti Singh; Kasra Torshizi; Khuzema Habib; Kelin Yu; Ruohan Gao; Pratap Tokekar
>
> **摘要:** Vision-based robot learning often relies on dense image or point-cloud inputs, which are computationally heavy and entangle irrelevant background features. Existing keypoint-based approaches can focus on manipulation-centric features and be lightweight, but either depend on manual heuristics or task-coupled selection, limiting scalability and semantic understanding. To address this, we propose AFFORD2ACT, an affordance-guided framework that distills a minimal set of semantic 2D keypoints from a text prompt and a single image. AFFORD2ACT follows a three-stage pipeline: affordance filtering, category-level keypoint construction, and transformer-based policy learning with embedded gating to reason about the most relevant keypoints, yielding a compact 38-dimensional state policy that can be trained in 15 minutes, which performs well in real-time without proprioception or dense representations. Across diverse real-world manipulation tasks, AFFORD2ACT consistently improves data efficiency, achieving an 82% success rate on unseen objects, novel categories, backgrounds, and distractors.
>
---
#### [replaced 016] Emergent Neural Automaton Policies: Learning Symbolic Structure from Visuomotor Trajectories
- **分类: cs.RO**

- **简介: 该论文提出ENAP框架，解决长周期机器人任务学习问题。通过结合神经符号方法与视觉运动数据，自动生成可解释的高阶规划结构，提升样本效率与任务理解。**

- **链接: [https://arxiv.org/pdf/2603.25903](https://arxiv.org/pdf/2603.25903)**

> **作者:** Yiyuan Pan; Xusheng Luo; Hanjiang Hu; Peiqi Yu; Changliu Liu
>
> **摘要:** Scaling robot learning to long-horizon tasks remains a formidable challenge. While end-to-end policies often lack the structural priors needed for effective long-term reasoning, traditional neuro-symbolic methods rely heavily on hand-crafted symbolic priors. To address the issue, we introduce ENAP (Emergent Neural Automaton Policy), a framework that allows a bi-level neuro-symbolic policy adaptively emerge from visuomotor demonstrations. Specifically, we first employ adaptive clustering and an extension of the L* algorithm to infer a Mealy state machine from visuomotor data, which serves as an interpretable high-level planner capturing latent task modes. Then, this discrete structure guides a low-level reactive residual network to learn precise continuous control via behavior cloning (BC). By explicitly modeling the task structure with discrete transitions and continuous residuals, ENAP achieves high sample efficiency and interpretability without requiring task-specific labels. Extensive experiments on complex manipulation and long-horizon tasks demonstrate that ENAP outperforms state-of-the-art (SoTA) end-to-end VLA policies by up to 27% in low-data regimes, while offering a structured representation of robotic intent (Fig. 1).
>
---
#### [replaced 017] A Multimodal Data Collection Framework for Dialogue-Driven Assistive Robotics to Clarify Ambiguities: A Wizard-of-Oz Pilot Study
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决对话驱动的助行机器人中语义模糊问题。提出多模态数据收集框架，通过对话协议和Wizard-of-Oz实验获取自然交互数据，支持更灵活的辅助控制。**

- **链接: [https://arxiv.org/pdf/2601.16870](https://arxiv.org/pdf/2601.16870)**

> **作者:** Guangping Liu; Nicholas Hawkins; Billy Madden; Tipu Sultan; Flavio Esposito; Madi Babaiasl
>
> **备注:** Accepted to IEEE RAS/EMBS 11th International Conference on Biomedical Robotics and Biomechatronics (BioRob) 2026
>
> **摘要:** Integrated control of wheelchairs and wheelchair-mounted robotic arms (WMRAs) has strong potential to increase independence for users with severe motor limitations, yet existing interfaces often lack the flexibility needed for intuitive assistive interaction. Although data-driven AI methods show promise, progress is limited by the lack of multimodal datasets that capture natural Human-Robot Interaction (HRI), particularly conversational ambiguity in dialogue-driven control. To address this gap, we propose a multimodal data collection framework that employs a dialogue-based interaction protocol and a two-room Wizard-of-Oz (WoZ) setup to simulate robot autonomy while eliciting natural user behavior. The framework records five synchronized modalities: RGB-D video, conversational audio, inertial measurement unit (IMU) signals, end-effector Cartesian pose, and whole-body joint states across five assistive tasks. Using this framework, we collected a pilot dataset of 53 trials from five participants and validated its quality through motion smoothness analysis and user feedback. The results show that the framework effectively captures diverse ambiguity types and supports natural dialogue-driven interaction, demonstrating its suitability for scaling to a larger dataset for learning, benchmarking, and evaluation of ambiguity-aware assistive control.
>
---
#### [replaced 018] Trajectory-based actuator identification via differentiable simulation
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决无扭矩传感器的执行器建模问题。通过轨迹匹配和可微仿真，从编码器数据中优化执行器模型参数，提升仿真与真实机器人行为的一致性。**

- **链接: [https://arxiv.org/pdf/2604.10351](https://arxiv.org/pdf/2604.10351)**

> **作者:** Vyacheslav Kovalev; Ekaterina Chaikovskaia; Egor Davydenko; Roman Gorbachev
>
> **摘要:** Accurate actuation models are critical for bridging the gap between simulation and real robot behavior, yet obtaining high-fidelity actuator dynamics typically requires dedicated test stands and torque sensing. We present a trajectory-based actuator identification method that uses differentiable simulation to fit system-level actuator models from encoder motion alone. Identification is posed as a trajectory-matching problem: given commanded joint positions and measured joint angles and velocities, we optimize actuator and simulator parameters by backpropagating through the simulator, without torque sensors, current/voltage measurements, or access to embedded motor-control internals. The framework supports multiple model classes, ranging from compact structured parameterizations to neural actuator mappings, within a unified optimization pipeline. On held-out real-robot trajectories for a high-gear-ratio actuator with an embedded PD controller, the proposed torque-sensor-free identification achieves much tighter trajectory alignment than a supervised stand-trained baseline dominated by steady-state data, reducing mean absolute position error from 14.20 mrad to as low as 7.54 mrad (1.88 times). Finally, we demonstrate downstream impact for the same actuator class in a real-robot locomotion study: training policies with the refined actuator model increases travel distance by 46% and reduces rotational deviation by 75% relative to the baseline.
>
---
#### [replaced 019] Humanoid Factors: Design Principles for AI Humanoids in Human Worlds
- **分类: cs.RO**

- **简介: 该论文属于人机交互领域，旨在解决人形机器人与人类共存的设计问题。提出“人形因素”框架，涵盖物理、认知、社交和伦理四方面，指导人形机器人的设计与评估。**

- **链接: [https://arxiv.org/pdf/2602.10069](https://arxiv.org/pdf/2602.10069)**

> **作者:** Xinyuan Liu; Eren Sadikoglu; Ransalu Senanayake; Lixiao Huang
>
> **摘要:** Human factors research has long focused on optimizing environments, tools, and systems to account for human performance. Yet, as humanoid robots begin to share our workplaces, homes, and public spaces, the design challenge expands. We must now consider not only factors for humans but also factors for humanoids, since both will coexist and interact within the same environments. Unlike conventional machines, humanoids introduce expectations of human-like behavior, communication, and social presence, which reshape usability, trust, and safety considerations. In this article, we introduce the concept of humanoid factors as a framework structured around four pillars - physical, cognitive, social, and ethical - that shape the development of humanoids to help them effectively coexist and collaborate with humans. This framework characterizes the overlap and divergence between human capabilities and those of general-purpose humanoids powered by AI foundation models. To demonstrate our framework's practical utility, we then apply the framework to evaluate a real-world humanoid control algorithm, illustrating how conventional task completion metrics in robotics overlook key human cognitive and interaction principles. We thus position humanoid factors as a foundational framework for designing, evaluating, and governing sustained human-humanoid coexistence.
>
---
#### [replaced 020] Towards a Multi-Embodied Grasping Agent
- **分类: cs.RO**

- **简介: 该论文属于多机械臂抓取任务，旨在解决不同夹爪设计下的通用抓取问题。提出一种数据高效、等变的抓取生成架构，利用夹爪和场景几何信息进行有效抓取。**

- **链接: [https://arxiv.org/pdf/2510.27420](https://arxiv.org/pdf/2510.27420)**

> **作者:** Roman Freiberg; Alexander Qualmann; Ngo Anh Vien; Gerhard Neumann
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Multi-embodiment grasping focuses on developing approaches that exhibit generalist behavior across diverse gripper designs. Existing methods often learn the kinematic structure of the robot implicitly and face challenges due to the difficulty of sourcing the required large-scale data. In this work, we present a data-efficient, flow-based, equivariant grasp synthesis architecture that can handle different gripper types with variable degrees of freedom and successfully exploit the underlying kinematic model, deducing all necessary information solely from the gripper and scene geometry. Unlike previous equivariant grasping methods, we translated all modules from the ground up to JAX and provide a model with batching capabilities over scenes, grippers, and grasps, resulting in smoother learning, improved performance and faster inference time. Our dataset encompasses grippers ranging from humanoid hands to parallel yaw grippers and includes 25,000 scenes and 20 million grasps.
>
---
#### [replaced 021] Learning to Plan, Planning to Learn: Adaptive Hierarchical RL-MPC for Sample-Efficient Decision Making
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习与模型预测控制的融合任务，旨在提升决策效率与鲁棒性。通过结合深度强化学习与MPPI采样，实现自适应规划，提高数据效率和任务成功率。**

- **链接: [https://arxiv.org/pdf/2512.17091](https://arxiv.org/pdf/2512.17091)**

> **作者:** Toshiaki Hori; Jonathan DeCastro; Deepak Gopinath; Avinash Balachandran; Guy Rosman
>
> **备注:** 27 pages, 10 figures, 8th Annual Learning for Dynamics & Control Conference (L4DC)
>
> **摘要:** We propose a new approach for solving planning problems with a hierarchical structure, fusing reinforcement learning and MPC planning. Our formulation tightly and elegantly couples the two planning paradigms. It leverages reinforcement learning actions to inform the MPPI sampler, and adaptively aggregates MPPI samples to inform the value estimation. The resulting adaptive process leverages further MPPI exploration where value estimates are uncertain, and improves training robustness and the overall resulting policies. This results in a robust planning approach that can handle complex planning problems and easily adapts to different applications, as demonstrated over several domains, including race driving, modified Acrobot, and Lunar Lander with added obstacles. Our results in these domains show better data efficiency and overall performance in terms of both rewards and task success, with up to a 72% increase in success rate compared to existing approaches, as well as accelerated convergence (x2.1) compared to non-adaptive sampling.
>
---
#### [replaced 022] IROSA: Interactive Robot Skill Adaptation using Natural Language
- **分类: cs.RO; cs.AI; cs.CL; cs.HC; cs.LG**

- **简介: 该论文属于机器人技能适应任务，旨在通过自然语言实现机器人技能的灵活调整。工作包括提出一个框架，利用预训练语言模型选择工具，无需微调即可完成轨迹修正、避障等操作。**

- **链接: [https://arxiv.org/pdf/2603.03897](https://arxiv.org/pdf/2603.03897)**

> **作者:** Markus Knauer; Samuel Bustamante; Thomas Eiband; Alin Albu-Schäffer; Freek Stulp; João Silvério
>
> **备注:** Accepted IEEE Robotics and Automation Letters (RA-L) journal, 8 pages, 5 figures, 3 tables, 1 listing. Code available: this https URL
>
> **摘要:** Foundation models have demonstrated impressive capabilities across diverse domains, while imitation learning provides principled methods for robot skill adaptation from limited data. Combining these approaches holds significant promise for direct application to robotics, yet this combination has received limited attention, particularly for industrial deployment. We present a novel framework that enables open-vocabulary skill adaptation through a tool-based architecture, maintaining a protective abstraction layer between the language model and robot hardware. Our approach leverages pre-trained LLMs to select and parameterize specific tools for adapting robot skills without requiring fine-tuning or direct model-to-robot interaction. We demonstrate the framework on a 7-DoF torque-controlled robot performing an industrial bearing ring insertion task, showing successful skill adaptation through natural language commands for speed adjustment, trajectory correction, and obstacle avoidance while maintaining safety, transparency, and interpretability.
>
---
