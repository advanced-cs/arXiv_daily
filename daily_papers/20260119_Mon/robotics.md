# 机器人 cs.RO

- **最新发布 27 篇**

- **更新 10 篇**

## 最新发布

#### [new 001] Is open robotics innovation a threat to international peace and security?
- **分类: cs.RO**

- **简介: 该论文属于科技伦理研究，探讨开放机器人创新对国际和平与安全的潜在威胁，提出四项实践建议以促进负责任的机器人研发。**

- **链接: [https://arxiv.org/pdf/2601.10877v1](https://arxiv.org/pdf/2601.10877v1)**

> **作者:** Ludovic Righetti; Vincent Boulanin
>
> **摘要:** Open access to publication, software and hardware is central to robotics: it lowers barriers to entry, supports reproducible science and accelerates reliable system development. However, openness also exacerbates the inherent dual-use risks associated with research and innovation in robotics. It lowers barriers for states and non-state actors to develop and deploy robotics systems for military use and harmful purposes. Compared to other fields of engineering where dual-use risks are present - e.g., those that underlie the development of weapons of mass destruction (chemical, biological, radiological, and nuclear weapons) and even the field of AI, robotics offers no specific regulation and little guidance as to how research and innovation may be conducted and disseminated responsibly. While other fields can be used for guidance, robotics has its own needs and specificities which have to be taken into account. The robotics community should therefore work toward its own set of sector-specific guidance and possibly regulation. To that end, we propose a roadmap focusing on four practices: a) education in responsible robotics; b) incentivizing risk assessment; c) moderating the diffusion of high-risk material; and d) developing red lines.
>
---
#### [new 002] H-AIM: Orchestrating LLMs, PDDL, and Behavior Trees for Hierarchical Multi-Robot Planning
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.MA**

- **简介: 该论文提出H-AIM框架，解决多机器人长周期任务规划问题。融合LLM、PDDL与行为树，提升指令解析与动态协调能力。**

- **链接: [https://arxiv.org/pdf/2601.11063v1](https://arxiv.org/pdf/2601.11063v1)**

> **作者:** Haishan Zeng; Peng Li
>
> **摘要:** In embodied artificial intelligence, enabling heterogeneous robot teams to execute long-horizon tasks from high-level instructions remains a critical challenge. While large language models (LLMs) show promise in instruction parsing and preliminary planning, they exhibit limitations in long-term reasoning and dynamic multi-robot coordination. We propose Hierarchical Autonomous Intelligent Multi-Robot Planning(H-AIM), a novel embodied multi-robot task planning framework that addresses these issues through a three-stage cascaded architecture: 1) It leverages an LLM to parse instructions and generate Planning Domain Definition Language (PDDL) problem descriptions, thereby transforming commands into formal planning problems; 2) It combines the semantic reasoning of LLMs with the search capabilities of a classical planner to produce optimized action sequences; 3) It compiles the resulting plan into behavior trees for reactive control. The framework supports dynamically sized heterogeneous robot teams via a shared blackboard mechanism for communication and state synchronization. To validate our approach, we introduce the MACE-THOR benchmark dataset, comprising 42 complex tasks across 8 distinct household layouts. Experimental results demonstrate that H-AIM achieves a remarkable performance improvement, elevating the task success rate from 12% to 55% and boosting the goal condition recall from 32% to 72% against the strongest baseline, LaMMA-P.
>
---
#### [new 003] Skill-Aware Diffusion for Generalizable Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在提升机器人在不同环境中的泛化能力。通过引入技能级信息，提出SADiff模型，增强运动模式的泛化与执行效果。**

- **链接: [https://arxiv.org/pdf/2601.11266v1](https://arxiv.org/pdf/2601.11266v1)**

> **作者:** Aoshen Huang; Jiaming Chen; Jiyu Cheng; Ran Song; Wei Pan; Wei Zhang
>
> **摘要:** Robust generalization in robotic manipulation is crucial for robots to adapt flexibly to diverse environments. Existing methods usually improve generalization by scaling data and networks, but model tasks independently and overlook skill-level information. Observing that tasks within the same skill share similar motion patterns, we propose Skill-Aware Diffusion (SADiff), which explicitly incorporates skill-level information to improve generalization. SADiff learns skill-specific representations through a skill-aware encoding module with learnable skill tokens, and conditions a skill-constrained diffusion model to generate object-centric motion flow. A skill-retrieval transformation strategy further exploits skill-specific trajectory priors to refine the mapping from 2D motion flow to executable 3D actions. Furthermore, we introduce IsaacSkill, a high-fidelity dataset containing fundamental robotic skills for comprehensive evaluation and sim-to-real transfer. Experiments in simulation and real-world settings show that SADiff achieves good performance and generalization across various manipulation tasks. Code, data, and videos are available at https://sites.google.com/view/sa-diff.
>
---
#### [new 004] The Great March 100: 100 Detail-oriented Tasks for Evaluating Embodied AI Agents
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出GM-100任务集，用于评估具身AI代理。解决当前数据集任务设计缺乏系统性的问题，通过100个多样化任务全面评测机器人能力。**

- **链接: [https://arxiv.org/pdf/2601.11421v1](https://arxiv.org/pdf/2601.11421v1)**

> **作者:** Ziyu Wang; Chenyuan Liu; Yushun Xiang; Runhao Zhang; Qingbo Hao; Hongliang Lu; Houyu Chen; Zhizhong Feng; Kaiyue Zheng; Dehao Ye; Xianchao Zeng; Xinyu Zhou; Boran Wen; Jiaxin Li; Mingyu Zhang; Kecheng Zheng; Qian Zhu; Ran Cheng; Yong-Lu Li
>
> **摘要:** Recently, with the rapid development of robot learning and imitation learning, numerous datasets and methods have emerged. However, these datasets and their task designs often lack systematic consideration and principles. This raises important questions: Do the current datasets and task designs truly advance the capabilities of robotic agents? Do evaluations on a few common tasks accurately reflect the differentiated performance of various methods proposed by different teams and evaluated on different tasks? To address these issues, we introduce the Great March 100 (\textbf{GM-100}) as the first step towards a robot learning Olympics. GM-100 consists of 100 carefully designed tasks that cover a wide range of interactions and long-tail behaviors, aiming to provide a diverse and challenging set of tasks to comprehensively evaluate the capabilities of robotic agents and promote diversity and complexity in robot dataset task designs. These tasks are developed through systematic analysis and expansion of existing task designs, combined with insights from human-object interaction primitives and object affordances. We collect a large amount of trajectory data on different robotic platforms and evaluate several baseline models. Experimental results demonstrate that the GM-100 tasks are 1) feasible to execute and 2) sufficiently challenging to effectively differentiate the performance of current VLA models. Our data and code are available at https://rhos.ai/research/gm-100.
>
---
#### [new 005] Energy-Efficient Omnidirectional Locomotion for Wheeled Quadrupeds via Predictive Energy-Aware Nominal Gait Selection
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，旨在解决轮足机器人在复杂环境中能耗过高的问题。通过预测能量模型和强化学习优化步态选择与调整，显著降低能耗并保持性能。**

- **链接: [https://arxiv.org/pdf/2601.10723v1](https://arxiv.org/pdf/2601.10723v1)**

> **作者:** Xu Yang; Wei Yang; Kaibo He; Bo Yang; Yanan Sui; Yilin Mo
>
> **备注:** Published in IEEE IROS 2025
>
> **摘要:** Wheeled-legged robots combine the efficiency of wheels with the versatility of legs, but face significant energy optimization challenges when navigating diverse environments. In this work, we present a hierarchical control framework that integrates predictive power modeling with residual reinforcement learning to optimize omnidirectional locomotion efficiency for wheeled quadrupedal robots. Our approach employs a novel power prediction network that forecasts energy consumption across different gait patterns over a 1-second horizon, enabling intelligent selection of the most energy-efficient nominal gait. A reinforcement learning policy then generates residual adjustments to this nominal gait, fine-tuning the robot's actions to balance energy efficiency with performance objectives. Comparative analysis shows our method reduces energy consumption by up to 35\% compared to fixed-gait approaches while maintaining comparable velocity tracking performance. We validate our framework through extensive simulations and real-world experiments on a modified Unitree Go1 platform, demonstrating robust performance even under external disturbances. Videos and implementation details are available at \href{https://sites.google.com/view/switching-wpg}{https://sites.google.com/view/switching-wpg}.
>
---
#### [new 006] Verified Design of Robotic Autonomous Systems using Probabilistic Model Checking
- **分类: cs.RO**

- **简介: 该论文属于系统设计任务，旨在解决机器人自主系统设计中的风险评估与方案选择问题。通过概率模型检测方法，验证设计概念的可靠性。**

- **链接: [https://arxiv.org/pdf/2601.10720v1](https://arxiv.org/pdf/2601.10720v1)**

> **作者:** Atef Azaiez; Alireza David Anisi
>
> **备注:** Accepted in ModelSWARD 2026 conference, 7 figures
>
> **摘要:** Safety and reliability play a crucial role when designing Robotic Autonomous Systems (RAS). Early consideration of hazards, risks and mitigation actions -- already in the concept study phase -- are important steps in building a solid foundations for the subsequent steps in the system engineering life cycle. The complex nature of RAS, as well as the uncertain and dynamic environments the robots operate within, do not merely effect fault management and operation robustness, but also makes the task of system design concept selection, a hard problem to address. Approaches to tackle the mentioned challenges and their implications on system design, range from ad-hoc concept development and design practices, to systematic, statistical and analytical techniques of Model Based Systems Engineering. In this paper, we propose a methodology to apply a formal method, namely Probabilistic Model Checking (PMC), to enable systematic evaluation and analysis of a given set of system design concepts, ultimately leading to a set of Verified Designs (VD). We illustrate the application of the suggested methodology -- using PRISM as probabilistic model checker -- to a practical RAS concept selection use-case from agriculture robotics. Along the way, we also develop and present a domain-specific Design Evaluation Criteria for agri-RAS.
>
---
#### [new 007] Learning Semantic-Geometric Task Graph-Representations from Human Demonstrations
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究如何从人类示范中学习语义-几何任务图表示，解决长时程操作行为理解问题。通过结合MPNN和Transformer，实现任务进展推理与动作预测。**

- **链接: [https://arxiv.org/pdf/2601.11460v1](https://arxiv.org/pdf/2601.11460v1)**

> **作者:** Franziska Herbert; Vignesh Prasad; Han Liu; Dorothea Koert; Georgia Chalvatzaki
>
> **备注:** 9 pages, 7 figures, preprint
>
> **摘要:** Learning structured task representations from human demonstrations is essential for understanding long-horizon manipulation behaviors, particularly in bimanual settings where action ordering, object involvement, and interaction geometry can vary significantly. A key challenge lies in jointly capturing the discrete semantic structure of tasks and the temporal evolution of object-centric geometric relations in a form that supports reasoning over task progression. In this work, we introduce a semantic-geometric task graph-representation that encodes object identities, inter-object relations, and their temporal geometric evolution from human demonstrations. Building on this formulation, we propose a learning framework that combines a Message Passing Neural Network (MPNN) encoder with a Transformer-based decoder, decoupling scene representation learning from action-conditioned reasoning about task progression. The encoder operates solely on temporal scene graphs to learn structured representations, while the decoder conditions on action-context to predict future action sequences, associated objects, and object motions over extended time horizons. Through extensive evaluation on human demonstration datasets, we show that semantic-geometric task graph-representations are particularly beneficial for tasks with high action and object variability, where simpler sequence-based models struggle to capture task progression. Finally, we demonstrate that task graph representations can be transferred to a physical bimanual robot and used for online action selection, highlighting their potential as reusable task abstractions for downstream decision-making in manipulation systems.
>
---
#### [new 008] Multi-Agent Formation Navigation Using Diffusion-Based Trajectory Generation
- **分类: cs.RO; math.OC**

- **简介: 该论文属于多智能体路径规划任务，解决复杂环境中领导者-跟随者编队控制问题。通过扩散模型生成平面上的轨迹，实现稳定编队与低跟踪误差。**

- **链接: [https://arxiv.org/pdf/2601.10725v1](https://arxiv.org/pdf/2601.10725v1)**

> **作者:** Hieu Do Quang; Chien Truong-Quoc; Quoc Van Tran
>
> **备注:** 8 pages, 3 figures, full version of a paper submitted to a conference
>
> **摘要:** This paper introduces a diffusion-based planner for leader--follower formation control in cluttered environments. The diffusion policy is used to generate the trajectory of the midpoint of two leaders as a rigid bar in the plane, thereby defining their desired motion paths in a planar formation. While the followers track the leaders and form desired foramtion geometry using a distance-constrained formation controller based only on the relative positions in followers' local coordinates. The proposed approach produces smooth motions and low tracking errors, with most failures occurring in narrow obstacle-free space, or obstacle configurations that are not in the training data set. Simulation results demonstrate the potential of diffusion models for reliable multi-agent formation planning.
>
---
#### [new 009] A Survey of Real-Time Support, Analysis, and Advancements in ROS 2
- **分类: cs.RO; cs.DC; cs.SE**

- **简介: 该论文属于ROS 2实时性研究任务，旨在提升其实时执行能力。工作包括分析调度机制、评估时间指标、改进通信与执行器设计。**

- **链接: [https://arxiv.org/pdf/2601.10722v1](https://arxiv.org/pdf/2601.10722v1)**

> **作者:** Daniel Casini; Jian-Jia Chen; Jing Li; Federico Reghenzani; Harun Teper
>
> **摘要:** The Robot Operating System 2 (ROS~2) has emerged as a relevant middleware framework for robotic applications, offering modularity, distributed execution, and communication. In the last six years, ROS~2 has drawn increasing attention from the real-time systems community and industry. This survey presents a comprehensive overview of research efforts that analyze, enhance, and extend ROS~2 to support real-time execution. We first provide a detailed description of the internal scheduling mechanisms of ROS~2 and its layered architecture, including the interaction with DDS-based communication and other communication middleware. We then review key contributions from the literature, covering timing analysis for both single- and multi-threaded executors, metrics such as response time, reaction time, and data age, and different communication modes. The survey also discusses community-driven enhancements to the ROS~2 runtime, including new executor algorithm designs, real-time GPU management, and microcontroller support via micro-ROS. Furthermore, we summarize techniques for bounding DDS communication delays, message filters, and profiling tools that have been developed to support analysis and experimentation. To help systematize this growing body of work, we introduce taxonomies that classify the surveyed contributions based on different criteria. This survey aims to guide both researchers and practitioners in understanding and improving the real-time capabilities of ROS~2.
>
---
#### [new 010] SurfSLAM: Sim-to-Real Underwater Stereo Reconstruction For Real-Time SLAM
- **分类: cs.RO**

- **简介: 该论文属于 underwater SLAM 任务，旨在解决水下立体重建中的深度估计难题。通过 sim-to-real 训练和多传感器融合，提升水下机器人定位与建图精度。**

- **链接: [https://arxiv.org/pdf/2601.10814v1](https://arxiv.org/pdf/2601.10814v1)**

> **作者:** Onur Bagoren; Seth Isaacson; Sacchin Sundar; Yung-Ching Sun; Anja Sheppard; Haoyu Ma; Abrar Shariff; Ram Vasudevan; Katherine A. Skinner
>
> **摘要:** Localization and mapping are core perceptual capabilities for underwater robots. Stereo cameras provide a low-cost means of directly estimating metric depth to support these tasks. However, despite recent advances in stereo depth estimation on land, computing depth from image pairs in underwater scenes remains challenging. In underwater environments, images are degraded by light attenuation, visual artifacts, and dynamic lighting conditions. Furthermore, real-world underwater scenes frequently lack rich texture useful for stereo depth estimation and 3D reconstruction. As a result, stereo estimation networks trained on in-air data cannot transfer directly to the underwater domain. In addition, there is a lack of real-world underwater stereo datasets for supervised training of neural networks. Poor underwater depth estimation is compounded in stereo-based Simultaneous Localization and Mapping (SLAM) algorithms, making it a fundamental challenge for underwater robot perception. To address these challenges, we propose a novel framework that enables sim-to-real training of underwater stereo disparity estimation networks using simulated data and self-supervised finetuning. We leverage our learned depth predictions to develop \algname, a novel framework for real-time underwater SLAM that fuses stereo cameras with IMU, barometric, and Doppler Velocity Log (DVL) measurements. Lastly, we collect a challenging real-world dataset of shipwreck surveys using an underwater robot. Our dataset features over 24,000 stereo pairs, along with high-quality, dense photogrammetry models and reference trajectories for evaluation. Through extensive experiments, we demonstrate the advantages of the proposed training approach on real-world data for improving stereo estimation in the underwater domain and for enabling accurate trajectory estimation and 3D reconstruction of complex shipwreck sites.
>
---
#### [new 011] Distributed Control Barrier Functions for Safe Multi-Vehicle Navigation in Heterogeneous USV Fleets
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于多智能体安全导航任务，解决异构无人船队的避撞问题。通过分布式控制屏障函数方法，提升安全性和效率。**

- **链接: [https://arxiv.org/pdf/2601.11335v1](https://arxiv.org/pdf/2601.11335v1)**

> **作者:** Tyler Paine; Brendan Long; Jeremy Wenger; Michael DeFilippo; James Usevitch; Michael Benjamin
>
> **备注:** 8 pages, 10 figures
>
> **摘要:** Collision avoidance in heterogeneous fleets of uncrewed vessels is challenging because the decision-making processes and controllers often differ between platforms, and it is further complicated by the limitations on sharing trajectories and control values in real-time. This paper presents a pragmatic approach that addresses these issues by adding a control filter on each autonomous vehicle that assumes worst-case behavior from other contacts, including crewed vessels. This distributed safety control filter is developed using control barrier function (CBF) theory and the application is clearly described to ensure explainability of these safety-critical methods. This work compares the worst-case CBF approach with a Collision Regulations (COLREGS) behavior-based approach in simulated encounters. Real-world experiments with three different uncrewed vessels and a human operated vessel were performed to confirm the approach is effective across a range of platforms and is robust to uncooperative behavior from human operators. Results show that combining both CBF methods and COLREGS behaviors achieves the best safety and efficiency.
>
---
#### [new 012] Crane Lowering Guidance Using a Attachable Camera Module for Driver Vision Support
- **分类: cs.RO**

- **简介: 该论文属于起重机操作辅助任务，旨在解决操作员视线受阻的问题。通过安装可附着的摄像头模块，实时传输负载下方图像，提供视觉参考以提升施工安全。**

- **链接: [https://arxiv.org/pdf/2601.11026v1](https://arxiv.org/pdf/2601.11026v1)**

> **作者:** HyoJae Kang; SunWoo Ahn; InGyu Choi; GeonYeong Go; KunWoo Son; Min-Sung Kang
>
> **备注:** Presented at ICCR 2025(International COnference on Control and Robotics 2025). Submitted to the IEEE for possible publication
>
> **摘要:** Cranes have long been essential equipment for lifting and placing heavy loads in construction projects. This study focuses on the lowering phase of crane operation, the stage in which the load is moved to the desired location. During this phase, a constant challenge exists: the load obstructs the operator's view of the landing point. As a result, operators traditionally have to rely on verbal or gestural instructions from ground personnel, which significantly impacts site safety. To alleviate this constraint, the proposed system incorporates a attachable camera module designed to be attached directly to the load via a suction cup. This module houses a single-board computer, battery, and compact camera. After installation, it streams and processes images of the ground directly below the load in real time to generate installation guidance. Simultaneously, this guidance is transmitted to and monitored by a host computer. Preliminary experiments were conducted by attaching this module to a test object, confirming the feasibility of real-time image acquisition and transmission. This approach has the potential to significantly improve safety on construction sites by providing crane operators with an instant visual reference of hidden landing zones.
>
---
#### [new 013] Approximately Optimal Global Planning for Contact-Rich SE(2) Manipulation on a Graph of Reachable Sets
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文研究接触丰富操作的全局最优规划问题，提出一种分阶段方法，构建可达集图并在线优化路径，提升操作效率与成功率。**

- **链接: [https://arxiv.org/pdf/2601.10827v1](https://arxiv.org/pdf/2601.10827v1)**

> **作者:** Simin Liu; Tong Zhao; Bernhard Paus Graesdal; Peter Werner; Jiuguang Wang; John Dolan; Changliu Liu; Tao Pang
>
> **备注:** 17 pages, 14 figures; under submission to IEEE Transactions on Robotics
>
> **摘要:** If we consider human manipulation, it is clear that contact-rich manipulation (CRM)-the ability to use any surface of the manipulator to make contact with objects-can be far more efficient and natural than relying solely on end-effectors (i.e., fingertips). However, state-of-the-art model-based planners for CRM are still focused on feasibility rather than optimality, limiting their ability to fully exploit CRM's advantages. We introduce a new paradigm that computes approximately optimal manipulator plans. This approach has two phases. Offline, we construct a graph of mutual reachable sets, where each set contains all object orientations reachable from a starting object orientation and grasp. Online, we plan over this graph, effectively computing and sequencing local plans for globally optimized motion. On a challenging, representative contact-rich task, our approach outperforms a leading planner, reducing task cost by 61%. It also achieves a 91% success rate across 250 queries and maintains sub-minute query times, ultimately demonstrating that globally optimized contact-rich manipulation is now practical for real-world tasks.
>
---
#### [new 014] IMU-based Real-Time Crutch Gait Phase and Step Detections in Lower-Limb Exoskeletons
- **分类: cs.RO**

- **简介: 该论文属于实时步态识别任务，解决下肢外骨骼中步态与步态阶段检测问题。通过单个IMU和深度学习方法实现高精度、低延迟检测。**

- **链接: [https://arxiv.org/pdf/2601.10832v1](https://arxiv.org/pdf/2601.10832v1)**

> **作者:** Anis R. Shakkour; David Hexner; Yehuda Bitton; Avishai Sintov
>
> **摘要:** Lower limb exoskeletons and prostheses require precise, real time gait phase and step detections to ensure synchronized motion and user safety. Conventional methods often rely on complex force sensing hardware that introduces control latency. This paper presents a minimalist framework utilizing a single, low cost Inertial-Measurement Unit (IMU) integrated into the crutch hand grip, eliminating the need for mechanical modifications. We propose a five phase classification system, including standard gait phases and a non locomotor auxiliary state, to prevent undesired motion. Three deep learning architectures were benchmarked on both a PC and an embedded system. To improve performance under data constrained conditions, models were augmented with a Finite State Machine (FSM) to enforce biomechanical consistency. The Temporal Convolutional Network (TCN) emerged as the superior architecture, yielding the highest success rates and lowest latency. Notably, the model generalized to a paralyzed user despite being trained exclusively on healthy participants. Achieving a 94% success rate in detecting crutch steps, this system provides a high performance, cost effective solution for real time exoskeleton control.
>
---
#### [new 015] A3D: Adaptive Affordance Assembly with Dual-Arm Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人家具装配任务，解决双臂协作中支持策略适应与泛化问题。提出A3D框架，通过学习自适应 affordance 实现有效支持与稳定。**

- **链接: [https://arxiv.org/pdf/2601.11076v1](https://arxiv.org/pdf/2601.11076v1)**

> **作者:** Jiaqi Liang; Yue Chen; Qize Yu; Yan Shen; Haipeng Zhang; Hao Dong; Ruihai Wu
>
> **备注:** AAAI2026 oral
>
> **摘要:** Furniture assembly is a crucial yet challenging task for robots, requiring precise dual-arm coordination where one arm manipulates parts while the other provides collaborative support and stabilization. To accomplish this task more effectively, robots need to actively adapt support strategies throughout the long-horizon assembly process, while also generalizing across diverse part geometries. We propose A3D, a framework which learns adaptive affordances to identify optimal support and stabilization locations on furniture parts. The method employs dense point-level geometric representations to model part interaction patterns, enabling generalization across varied geometries. To handle evolving assembly states, we introduce an adaptive module that uses interaction feedback to dynamically adjust support strategies during assembly based on previous interactions. We establish a simulation environment featuring 50 diverse parts across 8 furniture types, designed for dual-arm collaboration evaluation. Experiments demonstrate that our framework generalizes effectively to diverse part geometries and furniture categories in both simulation and real-world settings.
>
---
#### [new 016] Learning Quadrupedal Locomotion for a Heavy Hydraulic Robot Using an Actuator Model
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，旨在解决液压四足机器人从仿真到现实的迁移难题。通过构建解析型执行器模型，提升强化学习效率，成功实现重载液压机器人的稳定运动控制。**

- **链接: [https://arxiv.org/pdf/2601.11143v1](https://arxiv.org/pdf/2601.11143v1)**

> **作者:** Minho Lee; Hyeonseok Kim; Jin Tak Kim; Sangshin Park; Jeong Hyun Lee; Jungsan Cho; Jemin Hwangbo
>
> **备注:** 9 pages, Accepted to IEEE Robotics and Automation Letters (RA-L) 2025
>
> **摘要:** The simulation-to-reality (sim-to-real) transfer of large-scale hydraulic robots presents a significant challenge in robotics because of the inherent slow control response and complex fluid dynamics. The complex dynamics result from the multiple interconnected cylinder structure and the difference in fluid rates of the cylinders. These characteristics complicate detailed simulation for all joints, making it unsuitable for reinforcement learning (RL) applications. In this work, we propose an analytical actuator model driven by hydraulic dynamics to represent the complicated actuators. The model predicts joint torques for all 12 actuators in under 1 microsecond, allowing rapid processing in RL environments. We compare our model with neural network-based actuator models and demonstrate the advantages of our model in data-limited scenarios. The locomotion policy trained in RL with our model is deployed on a hydraulic quadruped robot, which is over 300 kg. This work is the first demonstration of a successful transfer of stable and robust command-tracking locomotion with RL on a heavy hydraulic quadruped robot, demonstrating advanced sim-to-real transferability.
>
---
#### [new 017] Collaborative Continuum Robots: A Survey
- **分类: cs.RO**

- **简介: 本文属于机器人领域，探讨协作连续体机器人（CCRs）的研究现状。旨在解决多连续体机器人协同工作的优化问题，通过分类与综述结构设计、控制等技术，总结进展并指出未来方向。**

- **链接: [https://arxiv.org/pdf/2601.10721v1](https://arxiv.org/pdf/2601.10721v1)**

> **作者:** Xinyu Li; Qian Tang; Guoxin Yin; Gang Zheng; Jessica Burgner-Kahrs; Cesare Stefanini; Ke Wu
>
> **摘要:** Continuum robots (CRs), owing to their compact structure, inherent compliance, and flexible deformation, have been widely applied in various fields. By coordinating multiple CRs to form collaborative continuum robots (CCRs), task adaptability, workspace, flexibility, load capacity, and operational stability can be further improved, thus offering significant advantages. In recent years, interest in this emerging field has grown steadily within the continuum-robotics community, accompanied by a consistent rise in related publications. By presenting a comprehensive overview of recent progress from different system-architecture levels, this survey provides a clear framework for research on CCRs. First, CCRs are classified into the three collaboration modes of separated collaboration, assistance collaboration, and parallel collaboration, with definitions provided. Next, advances in structural design, modeling, motion planning, and control for each mode are systematically summarized. Finally, current challenges and future opportunities for CCRs are discussed.
>
---
#### [new 018] Visual Marker Search for Autonomous Drone Landing in Diverse Urban Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主无人机着陆任务，旨在解决复杂城市环境中标记识别与导航问题。通过仿真测试不同探索策略，提升着陆可靠性与效率。**

- **链接: [https://arxiv.org/pdf/2601.11078v1](https://arxiv.org/pdf/2601.11078v1)**

> **作者:** Jiaohong Yao; Linfeng Liang; Yao Deng; Xi Zheng; Richard Han; Yuankai Qi
>
> **摘要:** Marker-based landing is widely used in drone delivery and return-to-base systems for its simplicity and reliability. However, most approaches assume idealized landing site visibility and sensor performance, limiting robustness in complex urban settings. We present a simulation-based evaluation suite on the AirSim platform with systematically varied urban layouts, lighting, and weather to replicate realistic operational diversity. Using onboard camera sensors (RGB for marker detection and depth for obstacle avoidance), we benchmark two heuristic coverage patterns and a reinforcement learning-based agent, analyzing how exploration strategy and scene complexity affect success rate, path efficiency, and robustness. Results underscore the need to evaluate marker-based autonomous landing under diverse, sensor-relevant conditions to guide the development of reliable aerial navigation systems.
>
---
#### [new 019] Adaptive Monitoring of Stochastic Fire Front Processes via Information-seeking Predictive Control
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于 wildfire 监测任务，解决如何通过无人机动态采集数据以提高火势预测准确性的问题。提出一种结合感知、估计与控制的随机最优控制方法。**

- **链接: [https://arxiv.org/pdf/2601.11231v1](https://arxiv.org/pdf/2601.11231v1)**

> **作者:** Savvas Papaioannou; Panayiotis Kolios; Christos G. Panayiotou; Marios M. Polycarpou
>
> **备注:** 2025 IEEE 64th Conference on Decision and Control (CDC)
>
> **摘要:** We consider the problem of adaptively monitoring a wildfire front using a mobile agent (e.g., a drone), whose trajectory determines where sensor data is collected and thus influences the accuracy of fire propagation estimation. This is a challenging problem, as the stochastic nature of wildfire evolution requires the seamless integration of sensing, estimation, and control, often treated separately in existing methods. State-of-the-art methods either impose linear-Gaussian assumptions to establish optimality or rely on approximations and heuristics, often without providing explicit performance guarantees. To address these limitations, we formulate the fire front monitoring task as a stochastic optimal control problem that integrates sensing, estimation, and control. We derive an optimal recursive Bayesian estimator for a class of stochastic nonlinear elliptical-growth fire front models. Subsequently, we transform the resulting nonlinear stochastic control problem into a finite-horizon Markov decision process and design an information-seeking predictive control law obtained via a lower confidence bound-based adaptive search algorithm with asymptotic convergence to the optimal policy.
>
---
#### [new 020] The Mini Wheelbot Dataset: High-Fidelity Data for Robot Learning
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出一个高保真机器人数据集，用于不稳定系统的学习控制。解决真实世界数据获取难的问题，通过多种实验和控制方法收集数据，支持动力学建模、状态估计等任务。**

- **链接: [https://arxiv.org/pdf/2601.11394v1](https://arxiv.org/pdf/2601.11394v1)**

> **作者:** Henrik Hose; Paul Brunzema; Devdutt Subhasish; Sebastian Trimpe
>
> **摘要:** The development of robust learning-based control algorithms for unstable systems requires high-quality, real-world data, yet access to specialized robotic hardware remains a significant barrier for many researchers. This paper introduces a comprehensive dynamics dataset for the Mini Wheelbot, an open-source, quasi-symmetric balancing reaction wheel unicycle. The dataset provides 1 kHz synchronized data encompassing all onboard sensor readings, state estimates, ground-truth poses from a motion capture system, and third-person video logs. To ensure data diversity, we include experiments across multiple hardware instances and surfaces using various control paradigms, including pseudo-random binary excitation, nonlinear model predictive control, and reinforcement learning agents. We include several example applications in dynamics model learning, state estimation, and time-series classification to illustrate common robotics algorithms that can be benchmarked on our dataset.
>
---
#### [new 021] ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文提出ACoT-VLA模型，解决机器人操作任务中动作生成的精准性问题。通过引入显式和隐式动作推理模块，提升动作空间的直接推理能力。**

- **链接: [https://arxiv.org/pdf/2601.11404v1](https://arxiv.org/pdf/2601.11404v1)**

> **作者:** Linqing Zhong; Yi Liu; Yifei Wei; Ziyu Xiong; Maoqing Yao; Si Liu; Guanghui Ren
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as essential generalist robot policies for diverse manipulation tasks, conventionally relying on directly translating multimodal inputs into actions via Vision-Language Model (VLM) embeddings. Recent advancements have introduced explicit intermediary reasoning, such as sub-task prediction (language) or goal image synthesis (vision), to guide action generation. However, these intermediate reasoning are often indirect and inherently limited in their capacity to convey the full, granular information required for precise action execution. Instead, we posit that the most effective form of reasoning is one that deliberates directly in the action space. We introduce Action Chain-of-Thought (ACoT), a paradigm where the reasoning process itself is formulated as a structured sequence of coarse action intents that guide the final policy. In this paper, we propose ACoT-VLA, a novel architecture that materializes the ACoT paradigm. Specifically, we introduce two complementary components: an Explicit Action Reasoner (EAR) and Implicit Action Reasoner (IAR). The former proposes coarse reference trajectories as explicit action-level reasoning steps, while the latter extracts latent action priors from internal representations of multimodal input, co-forming an ACoT that conditions the downstream action head to enable grounded policy learning. Extensive experiments in real-world and simulation environments demonstrate the superiority of our proposed method, which achieves 98.5%, 84.1%, and 47.4% on LIBERO, LIBERO-Plus and VLABench, respectively.
>
---
#### [new 022] VLAgents: A Policy Server for Efficient VLA Inference
- **分类: cs.RO**

- **简介: 该论文提出VLAgents，解决VLA模型在机器人部署中的接口碎片化和通信延迟问题，通过统一协议提升推理效率。**

- **链接: [https://arxiv.org/pdf/2601.11250v1](https://arxiv.org/pdf/2601.11250v1)**

> **作者:** Tobias Jülg; Khaled Gamal; Nisarga Nilavadi; Pierre Krack; Seongjin Bien; Michael Krawez; Florian Walter; Wolfram Burgard
>
> **摘要:** The rapid emergence of Vision-Language-Action models (VLAs) has a significant impact on robotics. However, their deployment remains complex due to the fragmented interfaces and the inherent communication latency in distributed setups. To address this, we introduce VLAgents, a modular policy server that abstracts VLA inferencing behind a unified Gymnasium-style protocol. Crucially, its communication layer transparently adapts to the context by supporting both zero-copy shared memory for high-speed simulation and compressed streaming for remote hardware. In this work, we present the architecture of VLAgents and validate it by integrating seven policies -- including OpenVLA and Pi Zero. In a benchmark with both local and remote communication, we further demonstrate how it outperforms the default policy servers provided by OpenVLA, OpenPi, and LeRobot. VLAgents is available at https://github.com/RobotControlStack/vlagents
>
---
#### [new 023] Bidirectional Human-Robot Communication for Physical Human-Robot Interaction
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决物理交互中沟通不畅的问题。通过BRIDGE系统实现双向语言交流，允许用户实时修改机器人轨迹并获得反馈，提升交互性和透明度。**

- **链接: [https://arxiv.org/pdf/2601.10796v1](https://arxiv.org/pdf/2601.10796v1)**

> **作者:** Junxiang Wang; Cindy Wang; Rana Soltani Zarrin; Zackory Erickson
>
> **备注:** 12 pages, 8 figures. To be published in 2026 ACM/IEEE International Conference on Human-Robot Interaction
>
> **摘要:** Effective physical human-robot interaction requires systems that are not only adaptable to user preferences but also transparent about their actions. This paper introduces BRIDGE, a system for bidirectional human-robot communication in physical assistance. Our method allows users to modify a robot's planned trajectory -- position, velocity, and force -- in real time using natural language. We utilize a large language model (LLM) to interpret any trajectory modifications implied by user commands in the context of the planned motion and conversation history. Importantly, our system provides verbal feedback in response to the user, either assuring any resulting changes or posing a clarifying question. We evaluated our method in a user study with 18 older adults across three assistive tasks, comparing BRIDGE to an ablation without verbal feedback and a baseline. Results show that participants successfully used the system to modify trajectories in real time. Moreover, the bidirectional feedback led to significantly higher ratings of interactivity and transparency, demonstrating that the robot's verbal response is critical for a more intuitive user experience. Videos and code can be found on our project website: https://bidir-comm.github.io/
>
---
#### [new 024] Adaptive Sliding Mode Control for Vehicle Platoons with State-Dependent Friction Uncertainty
- **分类: cs.RO**

- **简介: 该论文属于车辆编队控制任务，旨在解决摩擦力不确定性带来的控制难题。提出一种自适应滑模控制器，以维持编队距离和速度，应对外部干扰和系统参数不确定。**

- **链接: [https://arxiv.org/pdf/2601.10724v1](https://arxiv.org/pdf/2601.10724v1)**

> **作者:** Rishabh Dev Yadav
>
> **备注:** Extended version based on the author MSc thesis. Related to an earlier IEEE ICAR 2021 publication
>
> **摘要:** Multi-robot formation control has various applications in domains such as vehicle troops, platoons, payload transportation, and surveillance. Maintaining formation in a vehicle platoon requires designing a suitable control scheme that can tackle external disturbances and uncertain system parameters while maintaining a predefined safe distance between the robots. A crucial challenge in this context is dealing with the unknown/uncertain friction forces between wheels and the ground, which vary with changes in road surface, wear in tires, and speed of the vehicle. Although state-of-the-art adaptive controllers can handle a priori bounded uncertainties, they struggle with accurately modeling and identifying frictional forces, which are often state-dependent and cannot be a priori bounded. This thesis proposes a new adaptive sliding mode controller for wheeled mobile robot-based vehicle platoons that can handle the unknown and complex behavior of frictional forces without prior knowledge of their parameters and structures. The controller uses the adaptive sliding mode control techniques to regulate the platoon's speed and maintain a predefined inter-robot distance, even in the presence of external disturbances and uncertain system parameters. This approach involves a two-stage process: first, the kinematic controller calculates the desired velocities based on the desired trajectory; and second, the dynamics model generates the commands to achieve the desired motion. By separating the kinematics and dynamics of the robot, this approach can simplify the control problem and allow for more efficient and robust control of the wheeled mobile robot.
>
---
#### [new 025] Where to Touch, How to Contact: Hierarchical RL-MPC Framework for Geometry-Aware Long-Horizon Dexterous Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决复杂接触环境下的精细操控问题。提出分层RL-MPC框架，结合几何与动力学规划，提升任务成功率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.10930v1](https://arxiv.org/pdf/2601.10930v1)**

> **作者:** Zhixian Xie; Yu Xiang; Michael Posa; Wanxin Jin
>
> **备注:** 13 Pages, Plan to submit RSS
>
> **摘要:** A key challenge in contact-rich dexterous manipulation is the need to jointly reason over geometry, kinematic constraints, and intricate, nonsmooth contact dynamics. End-to-end visuomotor policies bypass this structure, but often require large amounts of data, transfer poorly from simulation to reality, and generalize weakly across tasks/embodiments. We address those limitations by leveraging a simple insight: dexterous manipulation is inherently hierarchical - at a high level, a robot decides where to touch (geometry) and move the object (kinematics); at a low level it determines how to realize that plan through contact dynamics. Building on this insight, we propose a hierarchical RL--MPC framework in which a high-level reinforcement learning (RL) policy predicts a contact intention, a novel object-centric interface that specifies (i) an object-surface contact location and (ii) a post-contact object-level subgoal pose. Conditioned on this contact intention, a low-level contact-implicit model predictive control (MPC) optimizes local contact modes and replans with contact dynamics to generate robot actions that robustly drive the object toward each subgoal. We evaluate the framework on non-prehensile tasks, including geometry-generalized pushing and object 3D reorientation. It achieves near-100% success with substantially reduced data (10x less than end-to-end baselines), highly robust performance, and zero-shot sim-to-real transfer.
>
---
#### [new 026] Haptic Light-Emitting Diodes: Miniature, Luminous Tactile Actuators
- **分类: cs.HC; cs.RO**

- **简介: 该论文提出Haptic Light-Emitting Diodes（HLEDs），一种将光脉冲直接转换为机械力的微型触觉执行器，解决人机交互中的触觉反馈问题。**

- **链接: [https://arxiv.org/pdf/2601.11043v1](https://arxiv.org/pdf/2601.11043v1)**

> **作者:** Max Linnander; Yon Visell
>
> **摘要:** We present Haptic Light-Emitting Diodes (HLEDs), luminous thermopneumatic actuators that directly convert pulsed light into mechanical forces and displacements. Each device packages a miniature surface-mount LED in a gas-filled cavity that contains a low-inertia graphite photoabsorber. The cavity is sealed by an elastic membrane, which functions as a working diaphragm. Brief optical pulses heat the photoabsorber, which heats the gas. The resulting rapid pressure increases generate forces and displacements at the working diaphragm. Millimeter-scale HLEDs produce forces exceeding 0.4 N and displacements of 1 mm at low voltages, with 5 to 100 ms response times, making them attractive as actuators providing tactile feedback in human-machine interfaces. Perceptual testing revealed that the strength of tactile feedback increased linearly with optical power. HLEDs devices are mechanically simple and efficient to fabricate. Unusually, these actuators are also light-emitting, as a fraction of optical energy is transmitted through the membrane. These opto-mechanical actuators have many potential applications in tactile displays, human interface engineering, wearable computing, and other areas.
>
---
#### [new 027] Learning-Based Shrinking Disturbance-Invariant Tubes for State- and Input-Dependent Uncertainty
- **分类: eess.SY; cs.RO; math.OC**

- **简介: 该论文属于控制领域的安全验证任务，解决状态与输入相关不确定性下的鲁棒性问题，通过学习构建收缩不变区间，确保系统安全。**

- **链接: [https://arxiv.org/pdf/2601.11426v1](https://arxiv.org/pdf/2601.11426v1)**

> **作者:** Abdelrahman Ramadan; Sidney Givigi
>
> **摘要:** We develop a learning-based framework for constructing shrinking disturbance-invariant tubes under state- and input-dependent uncertainty, intended as a building block for tube Model Predictive Control (MPC), and certify safety via a lifted, isotone (order-preserving) fixed-point map. Gaussian Process (GP) posteriors become $(1-α)$ credible ellipsoids, then polytopic outer sets for deterministic set operations. A two-time-scale scheme separates learning epochs, where these polytopes are frozen, from an inner, outside-in iteration that converges to a compact fixed point $Z^\star\!\subseteq\!\mathcal G$; its state projection is RPI for the plant. As data accumulate, disturbance polytopes tighten, and the associated tubes nest monotonically, resolving the circular dependence between the set to be verified and the disturbance model while preserving hard constraints. A double-integrator study illustrates shrinking tube cross-sections in data-rich regions while maintaining invariance.
>
---
## 更新

#### [replaced 001] Probabilistic Mission Design for Neuro-Symbolic Unmanned Aircraft Systems
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于无人飞行器任务规划领域，解决在动态不确定环境中遵循法律框架的导航问题。提出ProMis系统，结合概率逻辑与机器学习，生成概率任务景观以指导UAS决策。**

- **链接: [https://arxiv.org/pdf/2501.01439v2](https://arxiv.org/pdf/2501.01439v2)**

> **作者:** Simon Kohaut; Benedict Flade; Daniel Ochs; Devendra Singh Dhami; Julian Eggert; Kristian Kersting
>
> **备注:** arXiv admin note: text overlap with arXiv:2406.03454
>
> **摘要:** Advanced Air Mobility (AAM) is a growing field that demands accurate and trustworthy models of legal concepts and restrictions for navigating Unmanned Aircraft Systems (UAS). In addition, any implementation of AAM needs to face the challenges posed by inherently dynamic and uncertain human-inhabited spaces robustly. Nevertheless, the employment of UAS beyond visual line of sight (BVLOS) is an endearing task that promises to significantly enhance today's logistics and emergency response capabilities. Hence, we propose Probabilistic Mission Design (ProMis), a novel neuro-symbolic approach to navigating UAS within legal frameworks. ProMis is an interpretable and adaptable system architecture that links uncertain geospatial data and noisy perception with declarative, Hybrid Probabilistic Logic Programs (HPLP) to reason over the agent's state space and its legality. To inform planning with legal restrictions and uncertainty in mind, ProMis yields Probabilistic Mission Landscapes (PML). These scalar fields quantify the belief that the HPLP is satisfied across the agent's state space. Extending prior work on ProMis' reasoning capabilities and computational characteristics, we show its integration with potent machine learning models such as Large Language Models (LLM) and Transformer-based vision models. Hence, our experiments underpin the application of ProMis with multi-modal input data and how our method applies to many AAM scenarios.
>
---
#### [replaced 002] Collaborative Representation Learning for Alignment of Tactile, Language, and Vision Modalities
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于多模态对齐任务，解决 tactile、language 和 vision 模态间融合不足的问题。提出 TLV-CoRe 方法，提升跨传感器泛化与模态交互。**

- **链接: [https://arxiv.org/pdf/2511.11512v4](https://arxiv.org/pdf/2511.11512v4)**

> **作者:** Yiyun Zhou; Mingjing Xu; Jingwei Shi; Quanjiang Li; Jingyuan Chen
>
> **摘要:** Tactile sensing offers rich and complementary information to vision and language, enabling robots to perceive fine-grained object properties. However, existing tactile sensors lack standardization, leading to redundant features that hinder cross-sensor generalization. Moreover, existing methods fail to fully integrate the intermediate communication among tactile, language, and vision modalities. To address this, we propose TLV-CoRe, a CLIP-based Tactile-Language-Vision Collaborative Representation learning method. TLV-CoRe introduces a Sensor-Aware Modulator to unify tactile features across different sensors and employs tactile-irrelevant decoupled learning to disentangle irrelevant tactile features. Additionally, a Unified Bridging Adapter is introduced to enhance tri-modal interaction within the shared representation space. To fairly evaluate the effectiveness of tactile models, we further propose the RSS evaluation framework, focusing on Robustness, Synergy, and Stability across different methods. Experimental results demonstrate that TLV-CoRe significantly improves sensor-agnostic representation learning and cross-modal alignment, offering a new direction for multimodal tactile representation.
>
---
#### [replaced 003] Robot-R1: Reinforcement Learning for Enhanced Embodied Reasoning in Robotics
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Robot-R1，用于增强机器人控制的具身推理。针对SFT方法的不足，采用强化学习提升任务完成能力，实验表明其效果优于SFT和GPT-4o。**

- **链接: [https://arxiv.org/pdf/2506.00070v3](https://arxiv.org/pdf/2506.00070v3)**

> **作者:** Dongyoung Kim; Sumin Park; Huiwon Jang; Jinwoo Shin; Jaehyung Kim; Younggyo Seo
>
> **备注:** NeurIPS 2025
>
> **摘要:** Large Vision-Language Models (LVLMs) have recently shown great promise in advancing robotics by combining embodied reasoning with robot control. A common approach involves training on embodied reasoning tasks related to robot control using Supervised Fine-Tuning (SFT). However, SFT datasets are often heuristically constructed and not explicitly optimized for improving robot control. Furthermore, SFT often leads to issues such as catastrophic forgetting and reduced generalization performance. To address these limitations, we introduce Robot-R1, a novel framework that leverages reinforcement learning to enhance embodied reasoning specifically for robot control. Robot-R1 learns to predict the next keypoint state required for task completion, conditioned on the current scene image and environment metadata derived from expert demonstrations. Inspired by the DeepSeek-R1 learning approach, Robot-R1 samples reasoning-based responses and reinforces those that lead to more accurate predictions. To rigorously evaluate Robot-R1, we also introduce a new benchmark that demands the diverse embodied reasoning capabilities for the task. Our experiments show that models trained with Robot-R1 outperform SFT methods on embodied reasoning tasks. Despite having only 7B parameters, Robot-R1 even surpasses GPT-4o on reasoning tasks related to low-level action control, such as spatial and movement reasoning.
>
---
#### [replaced 004] LeLaR: The First In-Orbit Demonstration of an AI-Based Satellite Attitude Controller
- **分类: cs.RO; cs.AI; cs.LG; eess.SY**

- **简介: 该论文属于卫星姿态控制任务，旨在解决传统控制器设计复杂、对模型不确定性敏感的问题。通过AI深度强化学习方法，在轨验证了新型姿态控制器的有效性。**

- **链接: [https://arxiv.org/pdf/2512.19576v3](https://arxiv.org/pdf/2512.19576v3)**

> **作者:** Kirill Djebko; Tom Baumann; Erik Dilger; Frank Puppe; Sergio Montenegro
>
> **备注:** This work has been submitted to the IEEE for possible publication. 55 pages, 27 figures, 29 tables. The maneuver telemetry datasets generated and analyzed during this work are available in the GitHub repository under https://github.com/kdjebko/lelar-in-orbit-data
>
> **摘要:** Attitude control is essential for many satellite missions. Classical controllers, however, are time-consuming to design and sensitive to model uncertainties and variations in operational boundary conditions. Deep Reinforcement Learning (DRL) offers a promising alternative by learning adaptive control strategies through autonomous interaction with a simulation environment. Overcoming the Sim2Real gap, which involves deploying an agent trained in simulation onto the real physical satellite, remains a significant challenge. In this work, we present the first successful in-orbit demonstration of an AI-based attitude controller for inertial pointing maneuvers. The controller was trained entirely in simulation and deployed to the InnoCube 3U nanosatellite, which was developed by the Julius-Maximilians-Universität Würzburg in cooperation with the Technische Universität Berlin, and launched in January 2025. We present the AI agent design, the methodology of the training procedure, the discrepancies between the simulation and the observed behavior of the real satellite, and a comparison of the AI-based attitude controller with the classical PD controller of InnoCube. Steady-state metrics confirm the robust performance of the AI-based controller during repeated in-orbit maneuvers.
>
---
#### [replaced 005] Off Policy Lyapunov Stability in Reinforcement Learning
- **分类: eess.SY; cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决传统算法缺乏稳定性保障的问题。通过提出离线学习Lyapunov函数的方法，提升算法的数据效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2509.09863v2](https://arxiv.org/pdf/2509.09863v2)**

> **作者:** Sarvan Gill; Daniela Constantinescu
>
> **备注:** Conference on Robot Learning (CORL) 2025
>
> **摘要:** Traditional reinforcement learning lacks the ability to provide stability guarantees. More recent algorithms learn Lyapunov functions alongside the control policies to ensure stable learning. However, the current self-learned Lyapunov functions are sample inefficient due to their on-policy nature. This paper introduces a method for learning Lyapunov functions off-policy and incorporates the proposed off-policy Lyapunov function into the Soft Actor Critic and Proximal Policy Optimization algorithms to provide them with a data efficient stability certificate. Simulations of an inverted pendulum and a quadrotor illustrate the improved performance of the two algorithms when endowed with the proposed off-policy Lyapunov function.
>
---
#### [replaced 006] Fine-Tuning of Neural Network Approximate MPC without Retraining via Bayesian Optimization
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于控制领域，解决AMPC部署中参数调优困难的问题。通过贝叶斯优化实现无需重训练的自动调优，提升控制性能。**

- **链接: [https://arxiv.org/pdf/2512.14350v3](https://arxiv.org/pdf/2512.14350v3)**

> **作者:** Henrik Hose; Paul Brunzema; Alexander von Rohr; Alexander Gräfe; Angela P. Schoellig; Sebastian Trimpe
>
> **备注:** Presented at the 13th International Conference on Robot Intelligence Technology and Applications
>
> **摘要:** Approximate model-predictive control (AMPC) aims to imitate an MPC's behavior with a neural network, removing the need to solve an expensive optimization problem at runtime. However, during deployment, the parameters of the underlying MPC must usually be fine-tuned. This often renders AMPC impractical as it requires repeatedly generating a new dataset and retraining the neural network. Recent work addresses this problem by adapting AMPC without retraining using approximated sensitivities of the MPC's optimization problem. Currently, this adaption must be done by hand, which is labor-intensive and can be unintuitive for high-dimensional systems. To solve this issue, we propose using Bayesian optimization to tune the parameters of AMPC policies based on experimental data. By combining model-based control with direct and local learning, our approach achieves superior performance to nominal AMPC on hardware, with minimal experimentation. This allows automatic and data-efficient adaptation of AMPC to new system instances and fine-tuning to cost functions that are difficult to directly implement in MPC. We demonstrate the proposed method in hardware experiments for the swing-up maneuver on an inverted cartpole and yaw control of an under-actuated balancing unicycle robot, a challenging control problem.
>
---
#### [replaced 007] Vision-Conditioned Variational Bayesian Last Layer Dynamics Models
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决环境变化下的动态适应问题。提出一种视觉条件的变分贝叶斯模型，实现环境变化的提前预测与控制优化。**

- **链接: [https://arxiv.org/pdf/2601.09178v2](https://arxiv.org/pdf/2601.09178v2)**

> **作者:** Paul Brunzema; Thomas Lew; Ray Zhang; Takeru Shirasawa; John Subosits; Marcus Greiff
>
> **备注:** 9 pages, 7 figures, currently under review
>
> **摘要:** Agile control of robotic systems often requires anticipating how the environment affects system behavior. For example, a driver must perceive the road ahead to anticipate available friction and plan actions accordingly. Achieving such proactive adaptation within autonomous frameworks remains a challenge, particularly under rapidly changing conditions. Traditional modeling approaches often struggle to capture abrupt variations in system behavior, while adaptive methods are inherently reactive and may adapt too late to ensure safety. We propose a vision-conditioned variational Bayesian last-layer dynamics model that leverages visual context to anticipate changes in the environment. The model first learns nominal vehicle dynamics and is then fine-tuned with feature-wise affine transformations of latent features, enabling context-aware dynamics prediction. The resulting model is integrated into an optimal controller for vehicle racing. We validate our method on a Lexus LC500 racing through water puddles. With vision-conditioning, the system completed all 12 attempted laps under varying conditions. In contrast, all baselines without visual context consistently lost control, demonstrating the importance of proactive dynamics adaptation in high-performance applications.
>
---
#### [replaced 008] Generalizable Domain Adaptation for Sim-and-Real Policy Co-Training
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操控领域，解决模拟与真实环境间策略迁移的问题。通过联合训练框架，利用大量模拟数据和少量真实数据，提升策略的泛化能力。**

- **链接: [https://arxiv.org/pdf/2509.18631v3](https://arxiv.org/pdf/2509.18631v3)**

> **作者:** Shuo Cheng; Liqian Ma; Zhenyang Chen; Ajay Mandlekar; Caelan Garrett; Danfei Xu
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Behavior cloning has shown promise for robot manipulation, but real-world demonstrations are costly to acquire at scale. While simulated data offers a scalable alternative, particularly with advances in automated demonstration generation, transferring policies to the real world is hampered by various simulation and real domain gaps. In this work, we propose a unified sim-and-real co-training framework for learning generalizable manipulation policies that primarily leverages simulation and only requires a few real-world demonstrations. Central to our approach is learning a domain-invariant, task-relevant feature space. Our key insight is that aligning the joint distributions of observations and their corresponding actions across domains provides a richer signal than aligning observations (marginals) alone. We achieve this by embedding an Optimal Transport (OT)-inspired loss within the co-training framework, and extend this to an Unbalanced OT framework to handle the imbalance between abundant simulation data and limited real-world examples. We validate our method on challenging manipulation tasks, showing it can leverage abundant simulation data to achieve up to a 30% improvement in the real-world success rate and even generalize to scenarios seen only in simulation. Project webpage: https://ot-sim2real.github.io/.
>
---
#### [replaced 009] EqVIO: An Equivariant Filter for Visual Inertial Odometry
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于视觉惯性里程计任务，解决轨迹估计问题。提出一种新的李群对称性，设计等变滤波器，提升估计一致性与精度。**

- **链接: [https://arxiv.org/pdf/2205.01980v3](https://arxiv.org/pdf/2205.01980v3)**

> **作者:** Pieter van Goor; Robert Mahony
>
> **备注:** 28 pages, 17 figures, published in IEEE TRO
>
> **摘要:** Visual-Inertial Odometry (VIO) is the problem of estimating a robot's trajectory by combining information from an inertial measurement unit (IMU) and a camera, and is of great interest to the robotics community. This paper develops a novel Lie group symmetry for the VIO problem and applies the recently proposed equivariant filter. The proposed symmetry is compatible with the invariance of the VIO reference frame, leading to improved filter consistency. The bias-free IMU dynamics are group-affine, ensuring that filter linearisation errors depend only on the bias estimation error and measurement noise. Furthermore, visual measurements are equivariant with respect to the symmetry, enabling the application of the higher-order equivariant output approximation to reduce approximation error in the filter update equation. As a result, the equivariant filter (EqF) based on this Lie group is a consistent estimator for VIO with lower linearisation error in the propagation of state dynamics and a higher order equivariant output approximation than standard formulations. Experimental results on the popular EuRoC and UZH FPV datasets demonstrate that the proposed system outperforms other state-of-the-art VIO algorithms in terms of both speed and accuracy.
>
---
#### [replaced 010] SceneFoundry: Generating Interactive Infinite 3D Worlds
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出SceneFoundry，用于生成可交互的3D虚拟环境，解决机器人学习中缺乏真实复杂场景的问题。通过语言引导和扩散模型，生成具有功能家具的多样化场景。**

- **链接: [https://arxiv.org/pdf/2601.05810v2](https://arxiv.org/pdf/2601.05810v2)**

> **作者:** ChunTeng Chen; YiChen Hsu; YiWen Liu; WeiFang Sun; TsaiChing Ni; ChunYi Lee; Min Sun; YuanFu Yang
>
> **备注:** 15 pages
>
> **摘要:** The ability to automatically generate large-scale, interactive, and physically realistic 3D environments is crucial for advancing robotic learning and embodied intelligence. However, existing generative approaches often fail to capture the functional complexity of real-world interiors, particularly those containing articulated objects with movable parts essential for manipulation and navigation. This paper presents SceneFoundry, a language-guided diffusion framework that generates apartment-scale 3D worlds with functionally articulated furniture and semantically diverse layouts for robotic training. From natural language prompts, an LLM module controls floor layout generation, while diffusion-based posterior sampling efficiently populates the scene with articulated assets from large-scale 3D repositories. To ensure physical usability, SceneFoundry employs differentiable guidance functions to regulate object quantity, prevent articulation collisions, and maintain sufficient walkable space for robotic navigation. Extensive experiments demonstrate that our framework generates structurally valid, semantically coherent, and functionally interactive environments across diverse scene types and conditions, enabling scalable embodied AI research. project page: https://anc891203.github.io/SceneFoundry-Demo/
>
---
