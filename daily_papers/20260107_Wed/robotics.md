# 机器人 cs.RO

- **最新发布 33 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] SOP: A Scalable Online Post-Training System for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文提出SOP系统，解决VLA模型在真实世界中高效、可扩展的后训练问题。通过在线学习与多机器人协作，提升模型性能并保持通用性。**

- **链接: [https://arxiv.org/pdf/2601.03044v1](https://arxiv.org/pdf/2601.03044v1)**

> **作者:** Mingjie Pan; Siyuan Feng; Qinglin Zhang; Xinchen Li; Jianheng Song; Chendi Qu; Yi Wang; Chuankang Li; Ziyu Xiong; Zhi Chen; Yi Liu; Jianlan Luo
>
> **摘要:** Vision-language-action (VLA) models achieve strong generalization through large-scale pre-training, but real-world deployment requires expert-level task proficiency in addition to broad generality. Existing post-training approaches for VLA models are typically offline, single-robot, or task-specific, limiting effective on-policy adaptation and scalable learning from real-world interaction. We introduce a Scalable Online Post-training (SOP) system that enables online, distributed, multi-task post-training of generalist VLA models directly in the physical world. SOP tightly couples execution and learning through a closed-loop architecture in which a fleet of robots continuously streams on-policy experience and human intervention signals to a centralized cloud learner, and asynchronously receives updated policies. This design supports prompt on-policy correction, scales experience collection through parallel deployment, and preserves generality during adaptation. SOP is agnostic to the choice of post-training algorithm; we instantiate it with both interactive imitation learning (HG-DAgger) and reinforcement learning (RECAP). Across a range of real-world manipulation tasks including cloth folding, box assembly, and grocery restocking, we show that SOP substantially improves the performance of large pretrained VLA models while maintaining a single shared policy across tasks. Effective post-training can be achieved within hours of real-world interaction, and performance scales near-linearly with the number of robots in the fleet. These results suggest that tightly coupling online learning with fleet-scale deployment is instrumental to enabling efficient, reliable, and scalable post-training of generalist robot policies in the physical world.
>
---
#### [new 002] Validating Generalist Robots with Situation Calculus and STL Falsification
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人验证任务，旨在解决通用机器人验证难题。通过结合情境演算与STL falsification，构建两层验证框架，有效发现控制器故障案例。**

- **链接: [https://arxiv.org/pdf/2601.03038v1](https://arxiv.org/pdf/2601.03038v1)**

> **作者:** Changwen Li; Rongjie Yan; Chih-Hong Cheng; Jian Zhang
>
> **摘要:** Generalist robots are becoming a reality, capable of interpreting natural language instructions and executing diverse operations. However, their validation remains challenging because each task induces its own operational context and correctness specification, exceeding the assumptions of traditional validation methods. We propose a two-layer validation framework that combines abstract reasoning with concrete system falsification. At the abstract layer, situation calculus models the world and derives weakest preconditions, enabling constraint-aware combinatorial testing to systematically generate diverse, semantically valid world-task configurations with controllable coverage strength. At the concrete layer, these configurations are instantiated for simulation-based falsification with STL monitoring. Experiments on tabletop manipulation tasks show that our framework effectively uncovers failure cases in the NVIDIA GR00T controller, demonstrating its promise for validating general-purpose robot autonomy.
>
---
#### [new 003] Advancing Assistive Robotics: Multi-Modal Navigation and Biophysical Monitoring for Next-Generation Wheelchairs
- **分类: cs.RO; cs.AR**

- **简介: 本文属于 assistive robotics 任务，旨在提升轮椅的自主导航与健康监测能力。通过多模态控制与生物信号监控，解决患者独立性与护理监督问题。**

- **链接: [https://arxiv.org/pdf/2601.02766v1](https://arxiv.org/pdf/2601.02766v1)**

> **作者:** Md. Anowar Hossain; Mohd. Ehsanul Hoque
>
> **摘要:** Assistive electric-powered wheelchairs (EPWs) have become essential mobility aids for people with disabilities such as amyotrophic lateral sclerosis (ALS), post-stroke hemiplegia, and dementia-related mobility impairment. This work presents a novel multi-modal EPW control system designed to prioritize patient needs while allowing seamless switching between control modes. Four complementary interfaces, namely joystick, speech, hand gesture, and electrooculography (EOG), are integrated with a continuous vital sign monitoring framework measuring heart rate variability, oxygen saturation (SpO2), and skin temperature. This combination enables greater patient independence while allowing caregivers to maintain real-time supervision and early intervention capability. Two-point calibration of the biophysical sensors against clinical reference devices resulted in root mean square errors of at most 2 bpm for heart rate, 0.5 degree Celsius for skin temperature, and 1 percent for SpO2. Experimental evaluation involved twenty participants with mobility impairments executing a total of 500 indoor navigation commands. The achieved command recognition accuracies were 99 percent for joystick control, 97 percent plus or minus 2 percent for speech, and 95 percent plus or minus 3 percent for hand gesture, with an average closed-loop latency of 20 plus or minus 0.5 milliseconds. Caregivers receive real-time alerts through an Android application following encrypted cloud transmission of physiological data. By integrating multi-modal mobility control with cloud-enabled health monitoring and reporting latency and energy budgets, the proposed prototype addresses key challenges in assistive robotics, contributes toward compliance with ISO 7176-31 and IEC 80601-2-78 safety standards, and establishes a foundation for future adaptive machine learning enhancements.
>
---
#### [new 004] Learning to Nudge: A Scalable Barrier Function Framework for Safe Robot Interaction in Dense Clutter
- **分类: cs.RO**

- **简介: 该论文属于机器人安全交互任务，解决密集环境中碰撞问题。提出DCBF方法，通过学习可组合的安全函数实现高效、安全的接触交互。**

- **链接: [https://arxiv.org/pdf/2601.02686v1](https://arxiv.org/pdf/2601.02686v1)**

> **作者:** Haixin Jin; Nikhil Uday Shinde; Soofiyan Atar; Hongzhan Yu; Dylan Hirsch; Sicun Gao; Michael C. Yip; Sylvia Herbert
>
> **摘要:** Robots operating in everyday environments must navigate and manipulate within densely cluttered spaces, where physical contact with surrounding objects is unavoidable. Traditional safety frameworks treat contact as unsafe, restricting robots to collision avoidance and limiting their ability to function in dense, everyday settings. As the number of objects grows, model-based approaches for safe manipulation become computationally intractable; meanwhile, learned methods typically tie safety to the task at hand, making them hard to transfer to new tasks without retraining. In this work we introduce Dense Contact Barrier Functions(DCBF). Our approach bypasses the computational complexity of explicitly modeling multi-object dynamics by instead learning a composable, object-centric function that implicitly captures the safety constraints arising from physical interactions. Trained offline on interactions with a few objects, the learned DCBFcomposes across arbitrary object sets at runtime, producing a single global safety filter that scales linearly and transfers across tasks without retraining. We validate our approach through simulated experiments in dense clutter, demonstrating its ability to enable collision-free navigation and safe, contact-rich interaction in suitable settings.
>
---
#### [new 005] Closing the Reality Gap: Zero-Shot Sim-to-Real Deployment for Dexterous Force-Based Grasping and Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操控任务，旨在解决仿真到现实的部署难题。通过结合触觉与扭矩感知，提出高效模拟与校准方法，实现无需调优的零样本真实机械手操作。**

- **链接: [https://arxiv.org/pdf/2601.02778v1](https://arxiv.org/pdf/2601.02778v1)**

> **作者:** Haoyu Dong; Zhengmao He; Yang Li; Zhibin Li; Xinyu Yi; Zhe Zhao
>
> **摘要:** Human-like dexterous hands with multiple fingers offer human-level manipulation capabilities, but training control policies that can directly deploy on real hardware remains difficult due to contact-rich physics and imperfect actuation. We close this gap with a practical sim-to-real reinforcement learning (RL) framework that utilizes dense tactile feedback combined with joint torque sensing to explicitly regulate physical interactions. To enable effective sim-to-real transfer, we introduce (i) a computationally fast tactile simulation that computes distances between dense virtual tactile units and the object via parallel forward kinematics, providing high-rate, high-resolution touch signals needed by RL; (ii) a current-to-torque calibration that eliminates the need for torque sensors on dexterous hands by mapping motor current to joint torque; and (iii) actuator dynamics modeling to bridge the actuation gaps with randomization of non-ideal effects such as backlash, torque-speed saturation. Using an asymmetric actor-critic PPO pipeline trained entirely in simulation, our policies deploy directly to a five-finger hand. The resulting policies demonstrated two essential skills: (1) command-based, controllable grasp force tracking, and (2) reorientation of objects in the hand, both of which were robustly executed without fine-tuning on the robot. By combining tactile and torque in the observation space with effective sensing/actuation modeling, our system provides a practical solution to achieve reliable dexterous manipulation. To our knowledge, this is the first demonstration of controllable grasping on a multi-finger dexterous hand trained entirely in simulation and transferred zero-shot on real hardware.
>
---
#### [new 006] Parameter-Robust MPPI for Safe Online Learning of Unknown Parameters
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决动态环境中参数不确定带来的安全问题。通过PRMPPI框架，结合在线参数学习与安全约束，提升控制安全性与性能。**

- **链接: [https://arxiv.org/pdf/2601.02948v1](https://arxiv.org/pdf/2601.02948v1)**

> **作者:** Matti Vahs; Jaeyoun Choi; Niklas Schmid; Jana Tumova; Chuchu Fan
>
> **摘要:** Robots deployed in dynamic environments must remain safe even when key physical parameters are uncertain or change over time. We propose Parameter-Robust Model Predictive Path Integral (PRMPPI) control, a framework that integrates online parameter learning with probabilistic safety constraints. PRMPPI maintains a particle-based belief over parameters via Stein Variational Gradient Descent, evaluates safety constraints using Conformal Prediction, and optimizes both a nominal performance-driven and a safety-focused backup trajectory in parallel. This yields a controller that is cautious at first, improves performance as parameters are learned, and ensures safety throughout. Simulation and hardware experiments demonstrate higher success rates, lower tracking error, and more accurate parameter estimates than baselines.
>
---
#### [new 007] PiDR: Physics-Informed Inertial Dead Reckoning for Autonomous Platforms
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于自主平台导航任务，解决纯惯性导航中的误差漂移问题。通过引入物理信息的残差组件，提升导航精度与可靠性。**

- **链接: [https://arxiv.org/pdf/2601.03040v1](https://arxiv.org/pdf/2601.03040v1)**

> **作者:** Arup Kumar Sahoo; Itzik Klein
>
> **备注:** 11 pages and 7 figures
>
> **摘要:** A fundamental requirement for full autonomy is the ability to sustain accurate navigation in the absence of external data, such as GNSS signals or visual information. In these challenging environments, the platform must rely exclusively on inertial sensors, leading to pure inertial navigation. However, the inherent noise and other error terms of the inertial sensors in such real-world scenarios will cause the navigation solution to drift over time. Although conventional deep-learning models have emerged as a possible approach to inertial navigation, they are inherently black-box in nature. Furthermore, they struggle to learn effectively with limited supervised sensor data and often fail to preserve physical principles. To address these limitations, we propose PiDR, a physics-informed inertial dead-reckoning framework for autonomous platforms in situations of pure inertial navigation. PiDR offers transparency by explicitly integrating inertial navigation principles into the network training process through the physics-informed residual component. PiDR plays a crucial role in mitigating abrupt trajectory deviations even under limited or sparse supervision. We evaluated PiDR on real-world datasets collected by a mobile robot and an autonomous underwater vehicle. We obtained more than 29% positioning improvement in both datasets, demonstrating the ability of PiDR to generalize different platforms operating in various environments and dynamics. Thus, PiDR offers a robust, lightweight, yet effective architecture and can be deployed on resource-constrained platforms, enabling real-time pure inertial navigation in adverse scenarios.
>
---
#### [new 008] Warm-Starting Collision-Free Model Predictive Control With Object-Centric Diffusion
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，解决复杂环境中快速生成无碰撞轨迹的问题。结合扩散模型与MPC，利用物体中心表示生成高效可行轨迹。**

- **链接: [https://arxiv.org/pdf/2601.02873v1](https://arxiv.org/pdf/2601.02873v1)**

> **作者:** Arthur Haffemayer; Alexandre Chapin; Armand Jordana; Krzysztof Wojciechowski; Florent Lamiraux; Nicolas Mansard; Vladimir Petrik
>
> **备注:** An open-source implementation is provided https://cozy-fairy-0e0139.netlify.app/
>
> **摘要:** Acting in cluttered environments requires predicting and avoiding collisions while still achieving precise control. Conventional optimization-based controllers can enforce physical constraints, but they struggle to produce feasible solutions quickly when many obstacles are present. Diffusion models can generate diverse trajectories around obstacles, yet prior approaches lacked a general and efficient way to condition them on scene structure. In this paper, we show that combining diffusion-based warm-starting conditioned with a latent object-centric representation of the scene and with a collision-aware model predictive controller (MPC) yields reliable and efficient motion generation under strict time limits. Our approach conditions a diffusion transformer on the system state, task, and surroundings, using an object-centric slot attention mechanism to provide a compact obstacle representation suitable for control. The sampled trajectories are refined by an optimal control problem that enforces rigid-body dynamics and signed-distance collision constraints, producing feasible motions in real time. On benchmark tasks, this hybrid method achieved markedly higher success rates and lower latency than sampling-based planners or either component alone. Real-robot experiments with a torque-controlled Panda confirm reliable and safe execution with MPC.
>
---
#### [new 009] Reinforcement Learning for Follow-the-Leader Robotic Endoscopic Navigation via Synthetic Data
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，旨在解决内窥镜机器人接触肠道壁的问题。通过深度强化学习和合成数据训练，提升导航精度与安全性。**

- **链接: [https://arxiv.org/pdf/2601.02798v1](https://arxiv.org/pdf/2601.02798v1)**

> **作者:** Sicong Gao; Chen Qian; Laurence Xian; Liao Wu; Maurice Pagnucco; Yang Song
>
> **摘要:** Autonomous navigation is crucial for both medical and industrial endoscopic robots, enabling safe and efficient exploration of narrow tubular environments without continuous human intervention, where avoiding contact with the inner walls has been a longstanding challenge for prior approaches. We present a follow-the-leader endoscopic robot based on a flexible continuum structure designed to minimize contact between the endoscope body and intestinal walls, thereby reducing patient discomfort. To achieve this objective, we propose a vision-based deep reinforcement learning framework guided by monocular depth estimation. A realistic intestinal simulation environment was constructed in \textit{NVIDIA Omniverse} to train and evaluate autonomous navigation strategies. Furthermore, thousands of synthetic intraluminal images were generated using NVIDIA Replicator to fine-tune the Depth Anything model, enabling dense three-dimensional perception of the intestinal environment with a single monocular camera. Subsequently, we introduce a geometry-aware reward and penalty mechanism to enable accurate lumen tracking. Compared with the original Depth Anything model, our method improves $δ_{1}$ depth accuracy by 39.2% and reduces the navigation J-index by 0.67 relative to the second-best method, demonstrating the robustness and effectiveness of the proposed approach.
>
---
#### [new 010] Making Infeasible Tasks Feasible: Planning to Reconfigure Disconnected 3D Environments with Movable Objects
- **分类: cs.RO**

- **简介: 该论文研究3D环境中机器人路径规划问题，解决因环境断开导致目标不可达的任务。通过重新配置可移动物体，构建可行路径，提出BRiDGE算法实现有效规划。**

- **链接: [https://arxiv.org/pdf/2601.02645v1](https://arxiv.org/pdf/2601.02645v1)**

> **作者:** Samarth Kalluraya; Yiannis Kantaros
>
> **摘要:** Several planners have been developed to compute dynamically feasible, collision-free robot paths from an initial to a goal configuration. A key assumption in these works is that the goal region is reachable; an assumption that often fails in practice when environments are disconnected. Motivated by this limitation, we consider known 3D environments comprising objects, also called blocks, that form distinct navigable support surfaces (planes), and that are either non-movable (e.g., tables) or movable (e.g., boxes). These surfaces may be mutually disconnected due to height differences, holes, or lateral separations. Our focus is on tasks where the robot must reach a goal region residing on an elevated plane that is unreachable. Rather than declaring such tasks infeasible, an effective strategy is to enable the robot to interact with the environment, rearranging movable objects to create new traversable connections; a problem known as Navigation Among Movable Objects (NAMO). Existing NAMO planners typically address 2D environments, where obstacles are pushed aside to clear a path. These methods cannot directly handle the considered 3D setting; in such cases, obstacles must be placed strategically to bridge these physical disconnections. We address this challenge by developing BRiDGE (Block-based Reconfiguration in Disconnected 3D Geometric Environments), a sampling-based planner that incrementally builds trees over robot and object configurations to compute feasible plans specifying which objects to move, where to place them, and in what order, while accounting for a limited number of movable objects. To accelerate planning, we introduce non-uniform sampling strategies. We show that our method is probabilistically complete and we provide extensive numerical and hardware experiments validating its effectiveness.
>
---
#### [new 011] A Bi-directional Adaptive Framework for Agile UAV Landing
- **分类: cs.RO**

- **简介: 该论文属于无人机自主着陆任务，解决动态平台着陆效率低的问题。提出双向协作框架，实现无人机与平台的协同优化，提升着陆效率与精度。**

- **链接: [https://arxiv.org/pdf/2601.03037v1](https://arxiv.org/pdf/2601.03037v1)**

> **作者:** Chunhui Zhao; Xirui Kao; Yilin Lu; Yang Lyu
>
> **备注:** This work has been submitted to the IEEE Robotics and Automation Letters (RA-L) for possible publication
>
> **摘要:** Autonomous landing on mobile platforms is crucial for extending quadcopter operational flexibility, yet conventional methods are often too inefficient for highly dynamic scenarios. The core limitation lies in the prevalent ``track-then-descend'' paradigm, which treats the platform as a passive target and forces the quadcopter to perform complex, sequential maneuvers. This paper challenges that paradigm by introducing a bi-directional cooperative landing framework that redefines the roles of the vehicle and the platform. The essential innovation is transforming the problem from a single-agent tracking challenge into a coupled system optimization. Our key insight is that the mobile platform is not merely a target, but an active agent in the landing process. It proactively tilts its surface to create an optimal, stable terminal attitude for the approaching quadcopter. This active cooperation fundamentally breaks the sequential model by parallelizing the alignment and descent phases. Concurrently, the quadcopter's planning pipeline focuses on generating a time-optimal and dynamically feasible trajectory that minimizes energy consumption. This bi-directional coordination allows the system to execute the recovery in an agile manner, characterized by aggressive trajectory tracking and rapid state synchronization within transient windows. The framework's effectiveness, validated in dynamic scenarios, significantly improves the efficiency, precision, and robustness of autonomous quadrotor recovery in complex and time-constrained missions.
>
---
#### [new 012] InternVLA-A1: Unifying Understanding, Generation and Action for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决视觉-语言-动作模型在物理世界动态预测上的不足。提出InternVLA-A1模型，融合语义理解与动态预测，提升机器人操作性能。**

- **链接: [https://arxiv.org/pdf/2601.02456v1](https://arxiv.org/pdf/2601.02456v1)**

> **作者:** Junhao Cai; Zetao Cai; Jiafei Cao; Yilun Chen; Zeyu He; Lei Jiang; Hang Li; Hengjie Li; Yang Li; Yufei Liu; Yanan Lu; Qi Lv; Haoxiang Ma; Jiangmiao Pang; Yu Qiao; Zherui Qiu; Yanqing Shen; Xu Shi; Yang Tian; Bolun Wang; Hanqing Wang; Jiaheng Wang; Tai Wang; Xueyuan Wei; Chao Wu; Yiman Xie; Boyang Xing; Yuqiang Yang; Yuyin Yang; Qiaojun Yu; Feng Yuan; Jia Zeng; Jingjing Zhang; Shenghan Zhang; Shi Zhang; Zhuoma Zhaxi; Bowen Zhou; Yuanzhen Zhou; Yunsong Zhou; Hongrui Zhu; Yangkun Zhu; Yuchen Zhu
>
> **备注:** Homepage: https://internrobotics.github.io/internvla-a1.github.io/
>
> **摘要:** Prevalent Vision-Language-Action (VLA) models are typically built upon Multimodal Large Language Models (MLLMs) and demonstrate exceptional proficiency in semantic understanding, but they inherently lack the capability to deduce physical world dynamics. Consequently, recent approaches have shifted toward World Models, typically formulated via video prediction; however, these methods often suffer from a lack of semantic grounding and exhibit brittleness when handling prediction errors. To synergize semantic understanding with dynamic predictive capabilities, we present InternVLA-A1. This model employs a unified Mixture-of-Transformers architecture, coordinating three experts for scene understanding, visual foresight generation, and action execution. These components interact seamlessly through a unified masked self-attention mechanism. Building upon InternVL3 and Qwen3-VL, we instantiate InternVLA-A1 at 2B and 3B parameter scales. We pre-train these models on hybrid synthetic-real datasets spanning InternData-A1 and Agibot-World, covering over 533M frames. This hybrid training strategy effectively harnesses the diversity of synthetic simulation data while minimizing the sim-to-real gap. We evaluated InternVLA-A1 across 12 real-world robotic tasks and simulation benchmark. It significantly outperforms leading models like pi0 and GR00T N1.5, achieving a 14.5\% improvement in daily tasks and a 40\%-73.3\% boost in dynamic settings, such as conveyor belt sorting.
>
---
#### [new 013] HEXAR: a Hierarchical Explainability Architecture for Robots
- **分类: cs.RO**

- **简介: 该论文提出HEXAR框架，解决机器人系统解释性不足的问题。通过分层模块化方法，提升决策透明度与效率。**

- **链接: [https://arxiv.org/pdf/2601.03070v1](https://arxiv.org/pdf/2601.03070v1)**

> **作者:** Tamlin Love; Ferran Gebellí; Pradip Pramanick; Antonio Andriella; Guillem Alenyà; Anais Garrell; Raquel Ros; Silvia Rossi
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** As robotic systems become increasingly complex, the need for explainable decision-making becomes critical. Existing explainability approaches in robotics typically either focus on individual modules, which can be difficult to query from the perspective of high-level behaviour, or employ monolithic approaches, which do not exploit the modularity of robotic architectures. We present HEXAR (Hierarchical EXplainability Architecture for Robots), a novel framework that provides a plug-in, hierarchical approach to generate explanations about robotic systems. HEXAR consists of specialised component explainers using diverse explanation techniques (e.g., LLM-based reasoning, causal models, feature importance, etc) tailored to specific robot modules, orchestrated by an explainer selector that chooses the most appropriate one for a given query. We implement and evaluate HEXAR on a TIAGo robot performing assistive tasks in a home environment, comparing it against end-to-end and aggregated baseline approaches across 180 scenario-query variations. We observe that HEXAR significantly outperforms baselines in root cause identification, incorrect information exclusion, and runtime, offering a promising direction for transparent autonomous systems.
>
---
#### [new 014] Dual-quaternion learning control for autonomous vehicle trajectory tracking with safety guarantees
- **分类: cs.RO; eess.SY; math.OC**

- **简介: 该论文属于自主车辆轨迹跟踪任务，解决运动不确定性下的安全控制问题。通过双四元数框架结合高斯过程，实现鲁棒的在线学习控制。**

- **链接: [https://arxiv.org/pdf/2601.03097v1](https://arxiv.org/pdf/2601.03097v1)**

> **作者:** Omayra Yago Nieto; Alexandre Anahory Simoes; Juan I. Giribet; Leonardo Colombo
>
> **摘要:** We propose a learning-based trajectory tracking controller for autonomous robotic platforms whose motion can be described kinematically on $\mathrm{SE}(3)$. The controller is formulated in the dual quaternion framework and operates at the velocity level, assuming direct command of angular and linear velocities, as is standard in many aerial vehicles and omnidirectional mobile robots. Gaussian Process (GP) regression is integrated into a geometric feedback law to learn and compensate online for unknown, state-dependent disturbances and modeling imperfections affecting both attitude and position, while preserving the algebraic structure and coupling properties inherent to rigid-body motion. The proposed approach does not rely on explicit parametric models of the unknown effects, making it well-suited for robotic systems subject to sensor-induced disturbances, unmodeled actuation couplings, and environmental uncertainties. A Lyapunov-based analysis establishes probabilistic ultimate boundedness of the pose tracking error under bounded GP uncertainty, providing formal stability guarantees for the learning-based controller. Simulation results demonstrate accurate and smooth trajectory tracking in the presence of realistic, localized disturbances, including correlated rotational and translational effects arising from magnetometer perturbations. These results illustrate the potential of combining geometric modeling and probabilistic learning to achieve robust, data-efficient pose control for autonomous robotic systems.
>
---
#### [new 015] Movement Primitives in Robotics: A Comprehensive Survey
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人学领域，旨在探讨运动基元在机器人控制中的应用，解决如何有效生成和学习复杂运动轨迹的问题。论文系统综述了相关方法、应用及挑战。**

- **链接: [https://arxiv.org/pdf/2601.02379v1](https://arxiv.org/pdf/2601.02379v1)**

> **作者:** Nolan B. Gutierrez; William J. Beksi
>
> **备注:** 105 pages, 3 figures, and 6 tables
>
> **摘要:** Biological systems exhibit a continuous stream of movements, consisting of sequential segments, that allow them to perform complex tasks in a creative and versatile fashion. This observation has led researchers towards identifying elementary building blocks of motion known as movement primitives, which are well-suited for generating motor commands in autonomous systems, such as robots. In this survey, we provide an encyclopedic overview of movement primitive approaches and applications in chronological order. Concretely, we present movement primitive frameworks as a way of representing robotic control trajectories acquired through human demonstrations. Within the area of robotics, movement primitives can encode basic motions at the trajectory level, such as how a robot would grasp a cup or the sequence of motions necessary to toss a ball. Furthermore, movement primitives have been developed with the desirable analytical properties of a spring-damper system, probabilistic coupling of multiple demonstrations, using neural networks in high-dimensional systems, and more, to address difficult challenges in robotics. Although movement primitives have widespread application to a variety of fields, the goal of this survey is to inform practitioners on the use of these frameworks in the context of robotics. Specifically, we aim to (i) present a systematic review of major movement primitive frameworks and examine their strengths and weaknesses; (ii) highlight applications that have successfully made use of movement primitives; and (iii) examine open questions and discuss practical challenges when applying movement primitives in robotics.
>
---
#### [new 016] Learning and Optimizing the Efficacy of Spatio-Temporal Task Allocation under Temporal and Resource Constraints
- **分类: cs.RO**

- **简介: 该论文研究多机器人系统的时空任务分配问题，解决在时间与资源约束下优化任务效能的问题。提出STEAM模型和E-ITAGS算法，实现高效任务分配与路径规划。**

- **链接: [https://arxiv.org/pdf/2601.02505v1](https://arxiv.org/pdf/2601.02505v1)**

> **作者:** Jiazhen Liu; Glen Neville; Jinwoo Park; Sonia Chernova; Harish Ravichandar
>
> **备注:** The journal extension version of our conference paper: arXiv:2404.07902, which has been accepted by ISRR 2024
>
> **摘要:** Complex multi-robot missions often require heterogeneous teams to jointly optimize task allocation, scheduling, and path planning to improve team performance under strict constraints. We formalize these complexities into a new class of problems, dubbed Spatio-Temporal Efficacy-optimized Allocation for Multi-robot systems (STEAM). STEAM builds upon trait-based frameworks that model robots using their capabilities (e.g., payload and speed), but goes beyond the typical binary success-failure model by explicitly modeling the efficacy of allocations as trait-efficacy maps. These maps encode how the aggregated capabilities assigned to a task determine performance. Further, STEAM accommodates spatio-temporal constraints, including a user-specified time budget (i.e., maximum makespan). To solve STEAM problems, we contribute a novel algorithm named Efficacy-optimized Incremental Task Allocation Graph Search (E-ITAGS) that simultaneously optimizes task performance and respects time budgets by interleaving task allocation, scheduling, and path planning. Motivated by the fact that trait-efficacy maps are difficult, if not impossible, to specify, E-ITAGS efficiently learns them using a realizability-aware active learning module. Our approach is realizability-aware since it explicitly accounts for the fact that not all combinations of traits are realizable by the robots available during learning. Further, we derive experimentally-validated bounds on E-ITAGS' suboptimality with respect to efficacy. Detailed numerical simulations and experiments using an emergency response domain demonstrate that E-ITAGS generates allocations of higher efficacy compared to baselines, while respecting resource and spatio-temporal constraints. We also show that our active learning approach is sample efficient and establishes a principled tradeoff between data and computational efficiency.
>
---
#### [new 017] A High-Fidelity Digital Twin for Robotic Manipulation Based on 3D Gaussian Splatting
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决数字孪生重建慢、视觉质量低及碰撞模型转换难的问题。通过3D高斯泼溅技术实现快速高保真场景重建，并提升几何精度以支持真实机器人操作。**

- **链接: [https://arxiv.org/pdf/2601.03200v1](https://arxiv.org/pdf/2601.03200v1)**

> **作者:** Ziyang Sun; Lingfan Bao; Tianhu Peng; Jingcheng Sun; Chengxu Zhou
>
> **备注:** Under review of Journal of Robot Learning
>
> **摘要:** Developing high-fidelity, interactive digital twins is crucial for enabling closed-loop motion planning and reliable real-world robot execution, which are essential to advancing sim-to-real transfer. However, existing approaches often suffer from slow reconstruction, limited visual fidelity, and difficulties in converting photorealistic models into planning-ready collision geometry. We present a practical framework that constructs high-quality digital twins within minutes from sparse RGB inputs. Our system employs 3D Gaussian Splatting (3DGS) for fast, photorealistic reconstruction as a unified scene representation. We enhance 3DGS with visibility-aware semantic fusion for accurate 3D labelling and introduce an efficient, filter-based geometry conversion method to produce collision-ready models seamlessly integrated with a Unity-ROS2-MoveIt physics engine. In experiments with a Franka Emika Panda robot performing pick-and-place tasks, we demonstrate that this enhanced geometric accuracy effectively supports robust manipulation in real-world trials. These results demonstrate that 3DGS-based digital twins, enriched with semantic and geometric consistency, offer a fast, reliable, and scalable path from perception to manipulation in unstructured environments.
>
---
#### [new 018] Loop Closure using AnyLoc Visual Place Recognition in DPV-SLAM
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉SLAM任务，旨在提升回环检测的准确性与鲁棒性。通过引入AnyLoc替代传统BoVW方法，并设计自适应阈值机制，优化DPV-SLAM的回环闭合性能。**

- **链接: [https://arxiv.org/pdf/2601.02723v1](https://arxiv.org/pdf/2601.02723v1)**

> **作者:** Wenzheng Zhang; Kazuki Adachi; Yoshitaka Hara; Sousuke Nakamura
>
> **备注:** Accepted at IEEE/SICE International Symposium on System Integration(SII) 2026. 6 pages, 14 figures
>
> **摘要:** Loop closure is crucial for maintaining the accuracy and consistency of visual SLAM. We propose a method to improve loop closure performance in DPV-SLAM. Our approach integrates AnyLoc, a learning-based visual place recognition technique, as a replacement for the classical Bag of Visual Words (BoVW) loop detection method. In contrast to BoVW, which relies on handcrafted features, AnyLoc utilizes deep feature representations, enabling more robust image retrieval across diverse viewpoints and lighting conditions. Furthermore, we propose an adaptive mechanism that dynamically adjusts similarity threshold based on environmental conditions, removing the need for manual tuning. Experiments on both indoor and outdoor datasets demonstrate that our method significantly outperforms the original DPV-SLAM in terms of loop closure accuracy and robustness. The proposed method offers a practical and scalable solution for enhancing loop closure performance in modern SLAM systems.
>
---
#### [new 019] Effective Online 3D Bin Packing with Lookahead Parcels Using Monte Carlo Tree Search
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于在线3D装箱任务，解决物流中因货物批次变化导致的性能下降问题。通过MCTS结合前瞻信息和辅助奖励，提升装箱效率。**

- **链接: [https://arxiv.org/pdf/2601.02649v1](https://arxiv.org/pdf/2601.02649v1)**

> **作者:** Jiangyi Fang; Bowen Zhou; Haotian Wang; Xin Zhu; Leye Wang
>
> **摘要:** Online 3D Bin Packing (3D-BP) with robotic arms is crucial for reducing transportation and labor costs in modern logistics. While Deep Reinforcement Learning (DRL) has shown strong performance, it often fails to adapt to real-world short-term distribution shifts, which arise as different batches of goods arrive sequentially, causing performance drops. We argue that the short-term lookahead information available in modern logistics systems is key to mitigating this issue, especially during distribution shifts. We formulate online 3D-BP with lookahead parcels as a Model Predictive Control (MPC) problem and adapt the Monte Carlo Tree Search (MCTS) framework to solve it. Our framework employs a dynamic exploration prior that automatically balances a learned RL policy and a robust random policy based on the lookahead characteristics. Additionally, we design an auxiliary reward to penalize long-term spatial waste from individual placements. Extensive experiments on real-world datasets show that our method consistently outperforms state-of-the-art baselines, achieving over 10\% gains under distributional shifts, 4\% average improvement in online deployment, and up to more than 8\% in the best case--demonstrating the effectiveness of our framework.
>
---
#### [new 020] Soft Responsive Materials Enhance Humanoid Safety
- **分类: cs.RO**

- **简介: 该论文属于人形机器人安全任务，旨在解决机器人跌落时对自身和环境的伤害问题。通过软硬结合设计，使用响应性材料提升安全性。**

- **链接: [https://arxiv.org/pdf/2601.02857v1](https://arxiv.org/pdf/2601.02857v1)**

> **作者:** Chunzheng Wang; Yiyuan Zhang; Annan Tang; Ziqiu Zeng; Haoran Chen; Quan Gao; Zixuan Zhuang; Boyu Li; Zhilin Xiong; Aoqian Zhang; Ce Hao; Siyuan Luo; Tongyang Zhao; Cecilia Laschi; Fan Shi
>
> **备注:** 40 pages, 11 figures
>
> **摘要:** Humanoid robots are envisioned as general-purpose platforms in human-centered environments, yet their deployment is limited by vulnerability to falls and the risks posed by rigid metal-plastic structures to people and surroundings. We introduce a soft-rigid co-design framework that leverages non-Newtonian fluid-based soft responsive materials to enhance humanoid safety. The material remains compliant during normal interaction but rapidly stiffens under impact, absorbing and dissipating fall-induced forces. Physics-based simulations guide protector placement and thickness and enable learning of active fall policies. Applied to a 42 kg life-size humanoid, the protector markedly reduces peak impact and allows repeated falls without hardware damage, including drops from 3 m and tumbles down long staircases. Across diverse scenarios, the approach improves robot robustness and environmental safety. By uniting responsive materials, structural co-design, and learning-based control, this work advances interact-safe, industry-ready humanoid robots.
>
---
#### [new 021] LOST-3DSG: Lightweight Open-Vocabulary 3D Scene Graphs with Semantic Tracking in Dynamic Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于动态环境中的3D目标跟踪任务，解决传统方法依赖重型模型效率低的问题，提出LOST-3DSG轻量级方法，利用语义嵌入实现高效跟踪。**

- **链接: [https://arxiv.org/pdf/2601.02905v1](https://arxiv.org/pdf/2601.02905v1)**

> **作者:** Sara Micol Ferraina; Michele Brienza; Francesco Argenziano; Emanuele Musumeci; Vincenzo Suriani; Domenico D. Bloisi; Daniele Nardi
>
> **摘要:** Tracking objects that move within dynamic environments is a core challenge in robotics. Recent research has advanced this topic significantly; however, many existing approaches remain inefficient due to their reliance on heavy foundation models. To address this limitation, we propose LOST-3DSG, a lightweight open-vocabulary 3D scene graph designed to track dynamic objects in real-world environments. Our method adopts a semantic approach to entity tracking based on word2vec and sentence embeddings, enabling an open-vocabulary representation while avoiding the necessity of storing dense CLIP visual features. As a result, LOST-3DSG achieves superior performance compared to approaches that rely on high-dimensional visual embeddings. We evaluate our method through qualitative and quantitative experiments conducted in a real 3D environment using a TIAGo robot. The results demonstrate the effectiveness and efficiency of LOST-3DSG in dynamic object tracking. Code and supplementary material are publicly available on the project website at https://lab-rococo-sapienza.github.io/lost-3dsg/.
>
---
#### [new 022] Learning to Act Robustly with View-Invariant Latent Actions
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人视觉控制任务，旨在解决视角变化导致的策略性能下降问题。通过引入基于物理动态的视图不变潜在动作模型VILA，提升策略的泛化与迁移能力。**

- **链接: [https://arxiv.org/pdf/2601.02994v1](https://arxiv.org/pdf/2601.02994v1)**

> **作者:** Youngjoon Jeong; Junha Chun; Taesup Kim
>
> **备注:** Website: https://joon-stack.github.io/VILA/
>
> **摘要:** Vision-based robotic policies often struggle with even minor viewpoint changes, underscoring the need for view-invariant visual representations. This challenge becomes more pronounced in real-world settings, where viewpoint variability is unavoidable and can significantly disrupt policy performance. Existing methods typically learn invariance from multi-view observations at the scene level, but such approaches rely on visual appearance and fail to incorporate the physical dynamics essential for robust generalization. We propose View-Invariant Latent Action (VILA), which models a latent action capturing transition patterns across trajectories to learn view-invariant representations grounded in physical dynamics. VILA aligns these latent actions across viewpoints using an action-guided objective based on ground-truth action sequences. Experiments in both simulation and the real world show that VILA-based policies generalize effectively to unseen viewpoints and transfer well to new tasks, establishing VILA as a strong pretraining framework that improves robustness and downstream learning performance.
>
---
#### [new 023] Modeling the Mental World for Embodied AI: A Comprehensive Review
- **分类: cs.RO**

- **简介: 该论文属于认知建模任务，旨在解决 embodied AI 在社会交互中的理解难题。通过构建 MWM 理论框架，分析 ToM 方法与评估基准，推动人机协作发展。**

- **链接: [https://arxiv.org/pdf/2601.02378v1](https://arxiv.org/pdf/2601.02378v1)**

> **作者:** Biyuan Liu; Daigang Xu; Lei Jiang; Wenjun Guo; Ping Chen
>
> **摘要:** As the application of Embodied AI Agents in avatars, wearable devices, and robotic systems continues to deepen, their core research challenges have gradually shifted from physical environment interaction to the accurate understanding of social interactions. Traditional physical world models (PWM) focus on quantifiable physical attributes such as space and motion, failing to meet the needs of social intelligence modeling. In contrast, the Mental World Model (MWM), as a structured representation of humans' internal mental states, has become the critical cognitive foundation for embodied agents to achieve natural human-machine collaboration and dynamic social adaptation. However, current MWM research faces significant bottlenecks: such as fragmented conceptual framework with vague boundaries between MWM and PWM, disjointed reasoning mechanisms for the technical pathways and applicable scenarios of different Theory of Mind (ToM) reasoning paradigms, and detachment between evaluation and practice. To address these issues, this review systematically synthesizes over 100 authoritative studies to provide a comprehensive overview of MWM research for embodied AI. Its core contributions are threefold: First, it constructs a complete theoretical framework for MWM for the first time. Specifically, it distinguishes the essential differences between MWM and PWMs. Second, it systematically defines the key components of MWM through two paradigms for mental element representation. Third, it comprehensively analyzes two core ToM reasoning paradigms with 19 ToM methods. Finally, it also clarifies the integration trend of neuro-symbolic hybrid architectures, and synthesizes 26 ToM evaluation benchmarks. This work aims to promote the integration of embodied agents into human society and advance the in-depth development of human-machine collaborative interaction.
>
---
#### [new 024] Analysis of Various Manipulator Configurations Based on Multi-Objective Black-Box Optimization
- **分类: cs.RO**

- **简介: 该论文属于机械臂结构优化任务，旨在解决不同配置机械臂的最优设计问题。通过多目标优化分析末端可达性和关节扭矩，评估现有结构并为未来设计提供指导。**

- **链接: [https://arxiv.org/pdf/2601.02704v1](https://arxiv.org/pdf/2601.02704v1)**

> **作者:** Kento Kawaharazuka; Keita Yoneda; Takahiro Hattori; Shintaro Inoue; Kei Okada
>
> **备注:** Accepted to Advanced Robotics, website: https://haraduka.github.io/bbo-manip-design
>
> **摘要:** Various 6-degree-of-freedom (DOF) and 7-DOF manipulators have been developed to date. Over a long history, their joint configurations and link length ratios have been determined empirically. In recent years, the development of robotic foundation models has become increasingly active, leading to the continuous proposal of various manipulators to support these models. However, none of these manipulators share exactly the same structure, as the order of joints and the ratio of link lengths differ among robots. Therefore, in order to discuss the optimal structure of a manipulator, we performed multi-objective optimization from the perspectives of end-effector reachability and joint torque. We analyze where existing manipulator structures stand within the sampling results of the optimization and provide insights for future manipulator design.
>
---
#### [new 025] A Fast Semidefinite Convex Relaxation for Optimal Control Problems With Spatio-Temporal Constraints
- **分类: cs.RO**

- **简介: 该论文属于最优控制任务，解决具有时空约束的非凸优化问题。通过时间缩放直接多段射击和半定规划松弛方法，提升求解效率与最优性。**

- **链接: [https://arxiv.org/pdf/2601.03055v1](https://arxiv.org/pdf/2601.03055v1)**

> **作者:** Shiying Dong; Zhipeng Shen; Rudolf Reiter; Hailong Huang; Bingzhao Gao; Hong Chen; Wen-Hua Chen
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** Solving optimal control problems (OCPs) of autonomous agents operating under spatial and temporal constraints fast and accurately is essential in applications ranging from eco-driving of autonomous vehicles to quadrotor navigation. However, the nonlinear programs approximating the OCPs are inherently nonconvex due to the coupling between the dynamics and the event timing, and therefore, they are challenging to solve. Most approaches address this challenge by predefining waypoint times or just using nonconvex trajectory optimization, which simplifies the problem but often yields suboptimal solutions. To significantly improve the numerical properties, we propose a formulation with a time-scaling direct multiple shooting scheme that partitions the prediction horizon into segments aligned with characteristic time constraints. Moreover, we develop a fast semidefinite-programming-based convex relaxation that exploits the sparsity pattern of the lifted formulation. Comprehensive simulation studies demonstrate the solution optimality and computational efficiency. Furthermore, real-world experiments on a quadrotor waypoint flight task with constrained open time windows validate the practical applicability of the approach in complex environments.
>
---
#### [new 026] Trust in LLM-controlled Robotics: a Survey of Security Threats, Defenses and Challenges
- **分类: cs.RO**

- **简介: 本文探讨LLM控制机器人中的安全威胁与防御策略，属于安全防护任务。解决LLM与机器人物理执行间的漏洞问题，总结攻击类型并分析防御方法。**

- **链接: [https://arxiv.org/pdf/2601.02377v1](https://arxiv.org/pdf/2601.02377v1)**

> **作者:** Xinyu Huang; Shyam Karthick V B; Taozhao Chen; Mitch Bryson; Thomas Chaffey; Huaming Chen; Kim-Kwang Raymond Choo; Ian R. Manchester
>
> **摘要:** The integration of Large Language Models (LLMs) into robotics has revolutionized their ability to interpret complex human commands and execute sophisticated tasks. However, such paradigm shift introduces critical security vulnerabilities stemming from the ''embodiment gap'', a discord between the LLM's abstract reasoning and the physical, context-dependent nature of robotics. While security for text-based LLMs is an active area of research, existing solutions are often insufficient to address the unique threats for the embodied robotic agents, where malicious outputs manifest not merely as harmful text but as dangerous physical actions. In this work, we present a systematic survey, summarizing the emerging threat landscape and corresponding defense strategies for LLM-controlled robotics. Specifically, we discuss a comprehensive taxonomy of attack vectors, covering topics such as jailbreaking, backdoor attacks, and multi-modal prompt injection. In response, we analyze and categorize a range of defense mechanisms, from formal safety specifications and runtime enforcement to multi-LLM oversight and prompt hardening. Furthermore, we review key datasets and benchmarks used to evaluate the robustness of these embodied systems. By synthesizing current research, this work highlights the urgent need for context-aware security solutions and provides a foundational roadmap for the development of safe, secure, and reliable LLM-controlled robotics.
>
---
#### [new 027] Optimizing Control-Friendly Trajectories with Self-Supervised Residual Learning
- **分类: cs.RO**

- **简介: 该论文属于轨迹优化任务，旨在解决复杂机器人系统中因残余物理效应导致的轨迹跟踪难题。通过自监督残差学习构建混合模型，并优化出控制友好的轨迹。**

- **链接: [https://arxiv.org/pdf/2601.02738v1](https://arxiv.org/pdf/2601.02738v1)**

> **作者:** Kexin Guo; Zihan Yang; Yuhang Liu; Jindou Jia; Xiang Yu
>
> **备注:** 10 pages, 9 figures
>
> **摘要:** Real-world physics can only be analytically modeled with a certain level of precision for modern intricate robotic systems. As a result, tracking aggressive trajectories accurately could be challenging due to the existence of residual physics during controller synthesis. This paper presents a self-supervised residual learning and trajectory optimization framework to address the aforementioned challenges. At first, unknown dynamic effects on the closed-loop model are learned and treated as residuals of the nominal dynamics, jointly forming a hybrid model. We show that learning with analytic gradients can be achieved using only trajectory-level data while enjoying accurate long-horizon prediction with an arbitrary integration step size. Subsequently, a trajectory optimizer is developed to compute the optimal reference trajectory with the residual physics along it minimized. It ends up with trajectories that are friendly to the following control level. The agile flight of quadrotors illustrates that by utilizing the hybrid dynamics, the proposed optimizer outputs aggressive motions that can be precisely tracked.
>
---
#### [new 028] Unified Meta-Representation and Feedback Calibration for General Disturbance Estimation
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决未知时变扰动的估计问题。通过元学习和反馈校准方法，构建通用扰动估计框架，提升控制精度。**

- **链接: [https://arxiv.org/pdf/2601.02762v1](https://arxiv.org/pdf/2601.02762v1)**

> **作者:** Zihan Yang; Jindou Jia; Meng Wang; Yuhang Liu; Kexin Guo; Xiang Yu
>
> **备注:** 8 pages, 10 figures
>
> **摘要:** Precise control in modern robotic applications is always an open issue due to unknown time-varying disturbances. Existing meta-learning-based approaches require a shared representation of environmental structures, which lack flexibility for realistic non-structural disturbances. Besides, representation error and the distribution shifts can lead to heavy degradation in prediction accuracy. This work presents a generalizable disturbance estimation framework that builds on meta-learning and feedback-calibrated online adaptation. By extracting features from a finite time window of past observations, a unified representation that effectively captures general non-structural disturbances can be learned without predefined structural assumptions. The online adaptation process is subsequently calibrated by a state-feedback mechanism to attenuate the learning residual originating from the representation and generalizability limitations. Theoretical analysis shows that simultaneous convergence of both the online learning error and the disturbance estimation error can be achieved. Through the unified meta-representation, our framework effectively estimates multiple rapidly changing disturbances, as demonstrated by quadrotor flight experiments. See the project page for video, supplementary material and code: https://nonstructural-metalearn.github.io.
>
---
#### [new 029] M-SEVIQ: A Multi-band Stereo Event Visual-Inertial Quadruped-based Dataset for Perception under Rapid Motion and Challenging Illumination
- **分类: cs.RO**

- **简介: 该论文提出M-SEVIQ数据集，用于解决快速运动和复杂光照下的机器人感知问题。融合多光谱事件相机、帧相机、IMU等传感器，支持视觉-惯性里程计与语义分割研究。**

- **链接: [https://arxiv.org/pdf/2601.02777v1](https://arxiv.org/pdf/2601.02777v1)**

> **作者:** Jingcheng Cao; Chaoran Xiong; Jianmin Song; Shang Yan; Jiachen Liu; Ling Pei
>
> **备注:** 6 pages, 7 figures
>
> **摘要:** Agile locomotion in legged robots poses significant challenges for visual perception. Traditional frame-based cameras often fail in these scenarios for producing blurred images, particularly under low-light conditions. In contrast, event cameras capture changes in brightness asynchronously, offering low latency, high temporal resolution, and high dynamic range. These advantages make them suitable for robust perception during rapid motion and under challenging illumination. However, existing event camera datasets exhibit limitations in stereo configurations and multi-band sensing domains under various illumination conditions. To address this gap, we present M-SEVIQ, a multi-band stereo event visual and inertial quadruped dataset collected using a Unitree Go2 equipped with stereo event cameras, a frame-based camera, an inertial measurement unit (IMU), and joint encoders. This dataset contains more than 30 real-world sequences captured across different velocity levels, illumination wavelengths, and lighting conditions. In addition, comprehensive calibration data, including intrinsic, extrinsic, and temporal alignments, are provided to facilitate accurate sensor fusion and benchmarking. Our M-SEVIQ can be used to support research in agile robot perception, sensor fusion, semantic segmentation and multi-modal vision in challenging environments.
>
---
#### [new 030] Limited Linguistic Diversity in Embodied AI Datasets
- **分类: cs.CL; cs.AI; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型研究，旨在分析VLA数据集的语言多样性。工作包括对多个数据集进行系统审计，评估其语言特征，发现数据集存在语言重复、结构单一的问题。**

- **链接: [https://arxiv.org/pdf/2601.03136v1](https://arxiv.org/pdf/2601.03136v1)**

> **作者:** Selma Wanna; Agnes Luhtaru; Jonathan Salfity; Ryan Barron; Juston Moore; Cynthia Matuszek; Mitch Pryor
>
> **摘要:** Language plays a critical role in Vision-Language-Action (VLA) models, yet the linguistic characteristics of the datasets used to train and evaluate these systems remain poorly documented. In this work, we present a systematic dataset audit of several widely used VLA corpora, aiming to characterize what kinds of instructions these datasets actually contain and how much linguistic variety they provide. We quantify instruction language along complementary dimensions-including lexical variety, duplication and overlap, semantic similarity, and syntactic complexity. Our analysis shows that many datasets rely on highly repetitive, template-like commands with limited structural variation, yielding a narrow distribution of instruction forms. We position these findings as descriptive documentation of the language signal available in current VLA training and evaluation data, intended to support more detailed dataset reporting, more principled dataset selection, and targeted curation or augmentation strategies that broaden language coverage.
>
---
#### [new 031] Towards Zero-Shot Point Cloud Registration Across Diverse Scales, Scenes, and Sensor Setups
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于点云配准任务，解决零样本泛化问题。针对参数固定、特征迁移差和尺度不一致等问题，提出BUFFER-X框架，实现无需训练的跨场景、跨传感器配准。**

- **链接: [https://arxiv.org/pdf/2601.02759v1](https://arxiv.org/pdf/2601.02759v1)**

> **作者:** Hyungtae Lim; Minkyun Seo; Luca Carlone; Jaesik Park
>
> **备注:** 18 pages, 15 figures. Extended version of our ICCV 2025 highlight paper [arXiv:2503.07940]. arXiv admin note: substantial text overlap with arXiv:2503.07940
>
> **摘要:** Some deep learning-based point cloud registration methods struggle with zero-shot generalization, often requiring dataset-specific hyperparameter tuning or retraining for new environments. We identify three critical limitations: (a) fixed user-defined parameters (e.g., voxel size, search radius) that fail to generalize across varying scales, (b) learned keypoint detectors exhibit poor cross-domain transferability, and (c) absolute coordinates amplify scale mismatches between datasets. To address these three issues, we present BUFFER-X, a training-free registration framework that achieves zero-shot generalization through: (a) geometric bootstrapping for automatic hyperparameter estimation, (b) distribution-aware farthest point sampling to replace learned detectors, and (c) patch-level coordinate normalization to ensure scale consistency. Our approach employs hierarchical multi-scale matching to extract correspondences across local, middle, and global receptive fields, enabling robust registration in diverse environments. For efficiency-critical applications, we introduce BUFFER-X-Lite, which reduces total computation time by 43% (relative to BUFFER-X) through early exit strategies and fast pose solvers while preserving accuracy. We evaluate on a comprehensive benchmark comprising 12 datasets spanning object-scale, indoor, and outdoor scenes, including cross-sensor registration between heterogeneous LiDAR configurations. Results demonstrate that our approach generalizes effectively without manual tuning or prior knowledge of test domains. Code: https://github.com/MIT-SPARK/BUFFER-X.
>
---
#### [new 032] AMC26: High-performance DOb for robust position control
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于运动控制任务，旨在提升扰动估计的精度与鲁棒性。提出一种新型HPDOb，通过引入一阶截断误差动态，优于传统DObs的零阶方法。**

- **链接: [https://arxiv.org/pdf/2601.02560v1](https://arxiv.org/pdf/2601.02560v1)**

> **作者:** Emre Sariyildiz
>
> **摘要:** This paper presents a new HPDOb that significantly improves disturbance estimation accuracy and robustness in motion control systems, surpassing the capabilities of conventional DObs. The proposed observer is analysed and synthesised in the discrete-time domain, providing a realistic representation of their dynamic behaviour and enabling enhanced controller design for practical applications. The core contribution of the HPDOb is a novel synthesis method that incorporates higher-order truncation error dynamics into disturbance estimation. Unlike conventional DObs, which are limited to zero-order truncation error, the HPDOb achieves first-order truncation error, yielding markedly improved estimation accuracy and robustness against disturbances in motion control systems. Simulation and experiments verify the stability and performance of HPDOb.
>
---
#### [new 033] Nonlinear Spectral Modeling and Control of Soft-Robotic Muscles from Data
- **分类: math.DS; cs.CE; cs.RO; eess.SY; math.OC**

- **简介: 该论文属于软体机器人控制任务，解决人工肌肉非线性动态控制问题。通过数据驱动的谱子流形方法，建立输入输出映射，实现高效控制。**

- **链接: [https://arxiv.org/pdf/2601.03247v1](https://arxiv.org/pdf/2601.03247v1)**

> **作者:** Leonardo Bettini; Amirhossein Kazemipour; Robert K. Katzschmann; George Haller
>
> **摘要:** Artificial muscles are essential for compliant musculoskeletal robotics but complicate control due to nonlinear multiphysics dynamics. Hydraulically amplified electrostatic (HASEL) actuators, a class of soft artificial muscles, offer high performance but exhibit memory effects and hysteresis. Here we present a data-driven reduction and control strategy grounded in spectral submanifold (SSM) theory. In the adiabatic regime, where inputs vary slowly relative to intrinsic transients, trajectories rapidly converge to a low-dimensional slow manifold. We learn an explicit input-to-output map on this manifold from forced-response trajectories alone, avoiding decay experiments that can trigger hysteresis. We deploy the SSM-based model for real-time control of an antagonistic HASEL-clutch joint. This approach yields a substantial reduction in tracking error compared to feedback-only and feedforward-only baselines under identical settings. This record-and-control workflow enables rapid characterization and high-performance control of soft muscles and muscle-driven joints without detailed physics-based modeling.
>
---
## 更新

#### [replaced 001] VLN-MME: Diagnosing MLLMs as Language-guided Visual Navigation agents
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉导航任务，研究MLLMs作为零样本导航代理的性能。通过构建VLN-MME框架，发现增强推理反而降低效果，揭示其空间推理能力不足。**

- **链接: [https://arxiv.org/pdf/2512.24851v2](https://arxiv.org/pdf/2512.24851v2)**

> **作者:** Xunyi Zhao; Gengze Zhou; Qi Wu
>
> **摘要:** Multimodal Large Language Models (MLLMs) have demonstrated remarkable capabilities across a wide range of vision-language tasks. However, their performance as embodied agents, which requires multi-round dialogue spatial reasoning and sequential action prediction, needs further exploration. Our work investigates this potential in the context of Vision-and-Language Navigation (VLN) by introducing a unified and extensible evaluation framework to probe MLLMs as zero-shot agents by bridging traditional navigation datasets into a standardized benchmark, named VLN-MME. We simplify the evaluation with a highly modular and accessible design. This flexibility streamlines experiments, enabling structured comparisons and component-level ablations across diverse MLLM architectures, agent designs, and navigation tasks. Crucially, enabled by our framework, we observe that enhancing our baseline agent with Chain-of-Thought (CoT) reasoning and self-reflection leads to an unexpected performance decrease. This suggests MLLMs exhibit poor context awareness in embodied navigation tasks; although they can follow instructions and structure their output, their 3D spatial reasoning fidelity is low. VLN-MME lays the groundwork for systematic evaluation of general-purpose MLLMs in embodied navigation settings and reveals limitations in their sequential decision-making capabilities. We believe these findings offer crucial guidance for MLLM post-training as embodied agents.
>
---
#### [replaced 002] Evaluating Gemini Robotics Policies in a Veo World Simulator
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于机器人政策评估任务，旨在解决视频模型在机器人领域中的泛化与安全性评估问题。工作包括构建基于Veo的生成评估系统，支持多视角一致性与场景编辑，验证政策在不同条件下的表现。**

- **链接: [https://arxiv.org/pdf/2512.10675v2](https://arxiv.org/pdf/2512.10675v2)**

> **作者:** Gemini Robotics Team; Krzysztof Choromanski; Coline Devin; Yilun Du; Debidatta Dwibedi; Ruiqi Gao; Abhishek Jindal; Thomas Kipf; Sean Kirmani; Isabel Leal; Fangchen Liu; Anirudha Majumdar; Andrew Marmon; Carolina Parada; Yulia Rubanova; Dhruv Shah; Vikas Sindhwani; Jie Tan; Fei Xia; Ted Xiao; Sherry Yang; Wenhao Yu; Allan Zhou
>
> **摘要:** Generative world models hold significant potential for simulating interactions with visuomotor policies in varied environments. Frontier video models can enable generation of realistic observations and environment interactions in a scalable and general manner. However, the use of video models in robotics has been limited primarily to in-distribution evaluations, i.e., scenarios that are similar to ones used to train the policy or fine-tune the base video model. In this report, we demonstrate that video models can be used for the entire spectrum of policy evaluation use cases in robotics: from assessing nominal performance to out-of-distribution (OOD) generalization, and probing physical and semantic safety. We introduce a generative evaluation system built upon a frontier video foundation model (Veo). The system is optimized to support robot action conditioning and multi-view consistency, while integrating generative image-editing and multi-view completion to synthesize realistic variations of real-world scenes along multiple axes of generalization. We demonstrate that the system preserves the base capabilities of the video model to enable accurate simulation of scenes that have been edited to include novel interaction objects, novel visual backgrounds, and novel distractor objects. This fidelity enables accurately predicting the relative performance of different policies in both nominal and OOD conditions, determining the relative impact of different axes of generalization on policy performance, and performing red teaming of policies to expose behaviors that violate physical or semantic safety constraints. We validate these capabilities through 1600+ real-world evaluations of eight Gemini Robotics policy checkpoints and five tasks for a bimanual manipulator.
>
---
#### [replaced 003] RoboTracer: Mastering Spatial Trace with Reasoning in Vision-Language Models for Robotics
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出RoboTracer，解决机器人空间追踪任务中的多步度量推理与空间指代问题，通过3D感知和强化学习提升追踪性能。**

- **链接: [https://arxiv.org/pdf/2512.13660v2](https://arxiv.org/pdf/2512.13660v2)**

> **作者:** Enshen Zhou; Cheng Chi; Yibo Li; Jingkun An; Jiayuan Zhang; Shanyu Rong; Yi Han; Yuheng Ji; Mengzhen Liu; Pengwei Wang; Zhongyuan Wang; Lu Sheng; Shanghang Zhang
>
> **备注:** Project page: https://zhoues.github.io/RoboTracer
>
> **摘要:** Spatial tracing, as a fundamental embodied interaction ability for robots, is inherently challenging as it requires multi-step metric-grounded reasoning compounded with complex spatial referring and real-world metric measurement. However, existing methods struggle with this compositional task. To this end, we propose RoboTracer, a 3D-aware VLM that first achieves both 3D spatial referring and measuring via a universal spatial encoder and a regression-supervised decoder to enhance scale awareness during supervised fine-tuning (SFT). Moreover, RoboTracer advances multi-step metric-grounded reasoning via reinforcement fine-tuning (RFT) with metric-sensitive process rewards, supervising key intermediate perceptual cues to accurately generate spatial traces. To support SFT and RFT training, we introduce TraceSpatial, a large-scale dataset of 30M QA pairs, spanning outdoor/indoor/tabletop scenes and supporting complex reasoning processes (up to 9 steps). We further present TraceSpatial-Bench, a challenging benchmark filling the gap to evaluate spatial tracing. Experimental results show that RoboTracer surpasses baselines in spatial understanding, measuring, and referring, with an average success rate of 79.1%, and also achieves SOTA performance on TraceSpatial-Bench by a large margin, exceeding Gemini-2.5-Pro by 36% accuracy. Notably, RoboTracer can be integrated with various control policies to execute long-horizon, dynamic tasks across diverse robots (UR5, G1 humanoid) in cluttered real-world scenes. See the project page at https://zhoues.github.io/RoboTracer.
>
---
#### [replaced 004] Characterizing the Robustness of Black-Box LLM Planners Under Perturbed Observations with Adaptive Stress Testing
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文属于安全评估任务，旨在解决LLM在噪声环境下的鲁棒性问题。通过自适应压力测试，探索扰动空间，识别可能导致模型失效的场景和配置。**

- **链接: [https://arxiv.org/pdf/2505.05665v3](https://arxiv.org/pdf/2505.05665v3)**

> **作者:** Neeloy Chakraborty; John Pohovey; Melkior Ornik; Katherine Driggs-Campbell
>
> **备注:** 30 pages, 24 figures, 6 tables
>
> **摘要:** Large language models (LLMs) have recently demonstrated success in decision-making tasks including planning, control, and prediction, but their tendency to hallucinate unsafe and undesired outputs poses risks. This unwanted behavior is further exacerbated in environments where sensors are noisy or unreliable. Characterizing the behavior of LLM planners to varied observations is necessary to proactively avoid failures in safety-critical scenarios. We specifically investigate the response of LLMs along two different perturbation dimensions. Like prior works, one dimension generates semantically similar prompts with varied phrasing by randomizing order of details, modifying access to few-shot examples, etc. Unique to our work, the second dimension simulates access to varied sensors and noise to mimic raw sensor or detection algorithm failures. An initial case study in which perturbations are manually applied show that both dimensions lead LLMs to hallucinate in a multi-agent driving environment. However, manually covering the entire perturbation space for several scenarios is infeasible. As such, we propose a novel method for efficiently searching the space of prompt perturbations using adaptive stress testing (AST) with Monte-Carlo tree search (MCTS). Our AST formulation enables discovery of scenarios, sensor configurations, and prompt phrasing that cause language models to act with high uncertainty or even crash. By generating MCTS prompt perturbation trees across diverse scenarios, we show through extensive experiments that offline analyses can be used to proactively understand potential failures that may arise at runtime.
>
---
#### [replaced 005] RobotDiffuse: Diffusion-Based Motion Planning for Redundant Manipulators with the ROP Obstacle Avoidance Dataset
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人运动规划任务，旨在解决冗余机械臂在复杂环境中的避障问题。提出RobotDiffuse方法，结合扩散模型与物理约束，提升规划效果。**

- **链接: [https://arxiv.org/pdf/2412.19500v2](https://arxiv.org/pdf/2412.19500v2)**

> **作者:** Xudong Mou; Xiaohan Zhang; Tiejun Wang; Tianyu Wo; Cangbai Xu; Ningbo Gu; Rui Wang; Xudong Liu
>
> **摘要:** Redundant manipulators, with their higher Degrees of Freedom (DoFs), offer enhanced kinematic performance and versatility, making them suitable for applications like manufacturing, surgical robotics, and human-robot collaboration. However, motion planning for these manipulators is challenging due to increased DoFs and complex, dynamic environments. While traditional motion planning algorithms struggle with high-dimensional spaces, deep learning-based methods often face instability and inefficiency in complex tasks. This paper introduces RobotDiffuse, a diffusion model-based approach for motion planning in redundant manipulators. By integrating physical constraints with a point cloud encoder and replacing the U-Net structure with an encoder-only transformer, RobotDiffuse improves the model's ability to capture temporal dependencies and generate smoother, more coherent motion plans. We validate the approach using a complex simulator and release a new dataset, Robot-obtalcles-panda (ROP), with 35M robot poses and 0.14M obstacle avoidance scenarios. The highest overall score obtained in the experiment demonstrates the effectiveness of RobotDiffuse and the promise of diffusion models for motion planning tasks. The dataset can be accessed at https://github.com/ACRoboT-buaa/RobotDiffuse.
>
---
#### [replaced 006] DDBot: Differentiable Physics-based Digging Robot for Unknown Granular Materials
- **分类: cs.RO**

- **简介: 该论文研究未知颗粒材料的挖掘任务，解决复杂接触动力学和材料不确定性问题。提出DDBot框架，结合可微物理模拟与优化方法，实现高效精准的挖掘技能优化。**

- **链接: [https://arxiv.org/pdf/2510.17335v4](https://arxiv.org/pdf/2510.17335v4)**

> **作者:** Xintong Yang; Minglun Wei; Yu-Kun Lai; Ze Ji
>
> **备注:** Published as a regular paper by the IEEE Transactions on Robotics
>
> **摘要:** Automating the manipulation of granular materials poses significant challenges due to complex contact dynamics, unpredictable material properties, and intricate system states. Existing approaches often fail to achieve efficiency and accuracy in such tasks. To fill the research gap, this article studies the small-scale and high-precision granular material digging task with unknown physical properties. A key scientific problem addressed is the feasibility of applying first-order gradient-based optimization to complex differentiable granular material simulation and overcoming associated numerical instability. A new framework, named differentiable digging robot (DDBot), is proposed to manipulate granular materials, including sand and soil. Specifically, we equip DDBot with a differentiable physics-based simulator, tailored for granular material manipulation, powered by GPU-accelerated parallel computing and automatic differentiation. DDBot can perform efficient differentiable system identification and high-precision digging skill optimization for unknown granular materials, which is enabled by a differentiable skill-to-action mapping, a task-oriented demonstration method, gradient clipping and line search-based gradient descent. Experimental results show that DDBot can efficiently (converge within 5 to 20 minutes) identify unknown granular material dynamics and optimize digging skills, with high-precision results in zero-shot real-world deployments, highlighting its practicality. Benchmark results against state-of-the-art baselines also confirm the robustness and efficiency of DDBot in such digging tasks.
>
---
#### [replaced 007] An Informative Planning Framework for Target Tracking and Active Mapping in Dynamic Environments with ASVs
- **分类: cs.RO**

- **简介: 该论文属于目标跟踪与主动建图任务，解决动态环境中漂浮目标的跟踪问题。通过集成动态栅格地图和时空预测网络，提升跟踪性能。**

- **链接: [https://arxiv.org/pdf/2508.14636v3](https://arxiv.org/pdf/2508.14636v3)**

> **作者:** Sanjeev Ramkumar Sudha; Marija Popović; Erlend M. Coates
>
> **备注:** Accepted for publication in Robotics and Automation Letters (RA-L)
>
> **摘要:** Mobile robot platforms are increasingly being used to automate information gathering tasks such as environmental monitoring. Efficient target tracking in dynamic environments is critical for applications such as search and rescue and pollutant cleanups. In this letter, we study active mapping of floating targets that drift due to environmental disturbances such as wind and currents. This is a challenging problem as it involves predicting both spatial and temporal variations in the map due to changing conditions. We introduce an integrated framework combining dynamic occupancy grid mapping and an informative planning approach to actively map and track freely drifting targets with an autonomous surface vehicle. A key component of our adaptive planning approach is a spatiotemporal prediction network that predicts target position distributions over time. We further propose a planning objective for target tracking that leverages these predictions. Simulation experiments show that this planning objective improves target tracking performance compared to existing methods that consider only entropy reduction as the planning objective. Finally, we validate our approach in field tests, showcasing its ability to track targets in real-world monitoring scenarios.
>
---
#### [replaced 008] Openpi Comet: Competition Solution For 2025 BEHAVIOR Challenge
- **分类: cs.RO**

- **简介: 该论文为2025 BEHAVIOR挑战赛的竞赛解决方案，旨在解决长时序物理代理在模拟环境中的任务执行问题。通过优化训练方法和数据，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2512.10071v3](https://arxiv.org/pdf/2512.10071v3)**

> **作者:** Junjie Bai; Yu-Wei Chao; Qizhi Chen; Jinwei Gu; Moo Jin Kim; Zhaoshuo Li; Xuan Li; Tsung-Yi Lin; Ming-Yu Liu; Nic Ma; Kaichun Mo; Delin Qu; Shangkun Sun; Hongchi Xia; Fangyin Wei; Xiaohui Zeng
>
> **备注:** Post-challenge bug fix
>
> **摘要:** The 2025 BEHAVIOR Challenge is designed to rigorously track progress toward solving long-horizon tasks by physical agents in simulated environments. BEHAVIOR-1K focuses on everyday household tasks that people most want robots to assist with and these tasks introduce long-horizon mobile manipulation challenges in realistic settings, bridging the gap between current research and real-world, human-centric applications. This report presents our solution to the 2025 BEHAVIOR Challenge in a very close 2nd place and substantially outperforms the rest of the submissions. Building on $π_{0.5}$, we focus on systematically building our solution by studying the effects of training techniques and data. Through careful ablation studies, we reveal the scaling benefits in both the pre-training and post-training phases, leading to a validation Q-score of 0.345, significantly surpassing previous state-of-the-art performance. We summarize our practical lessons and design recommendations that we hope will provide actionable insights for the broader embodied AI community when adapting powerful foundation models to complex embodied scenarios. Project page: https://github.com/mli0603/openpi-comet
>
---
#### [replaced 009] DarkEQA: Benchmarking Vision-Language Models for Embodied Question Answering in Low-Light Indoor Environments
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于视觉语言模型在低光环境下的问答任务，旨在解决现有基准未覆盖低光条件的问题。工作包括构建DarkEQA基准，模拟真实低光场景，评估模型性能。**

- **链接: [https://arxiv.org/pdf/2512.24985v2](https://arxiv.org/pdf/2512.24985v2)**

> **作者:** Yohan Park; Hyunwoo Ha; Wonjun Jo; Tae-Hyun Oh
>
> **备注:** Submitted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Vision Language Models (VLMs) are increasingly adopted as central reasoning modules for embodied agents. Existing benchmarks evaluate their capabilities under ideal, well-lit conditions, yet robust 24/7 operation demands performance under a wide range of visual degradations, including low-light conditions at night or in dark environments--a core necessity that has been largely overlooked. To address this underexplored challenge, we present DarkEQA, an open-source benchmark for evaluating EQA-relevant perceptual primitives under multi-level low-light conditions. DarkEQA isolates the perception bottleneck by evaluating question answering from egocentric observations under controlled degradations, enabling attributable robustness analysis. A key design feature of DarkEQA is its physical fidelity: visual degradations are modeled in linear RAW space, simulating physics-based illumination drop and sensor noise followed by an ISP-inspired rendering pipeline. We demonstrate the utility of DarkEQA by evaluating a wide range of state-of-the-art VLMs and Low-Light Image Enhancement (LLIE) models. Our analysis systematically reveals VLMs' limitations when operating under these challenging visual conditions. Project website: https://darkeqa-benchmark.github.io/
>
---
#### [replaced 010] Efficient Swept Volume-Based Trajectory Generation for Arbitrary-Shaped Ground Robot Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决复杂环境中任意形状地面机器人高效安全导航问题。提出一种分阶段框架，提升计算效率并保证连续避障。**

- **链接: [https://arxiv.org/pdf/2504.07554v2](https://arxiv.org/pdf/2504.07554v2)**

> **作者:** Yisheng Li; Longji Yin; Yixi Cai; Jianheng Liu; Fangcheng Zhu; Mingpu Ma; Siqi Liang; Haotian Li; Fu Zhang
>
> **摘要:** Navigating an arbitrary-shaped ground robot safely in cluttered environments remains a challenging problem. The existing trajectory planners that account for the robot's physical geometry severely suffer from the intractable runtime. To achieve both computational efficiency and Continuous Collision Avoidance (CCA) of arbitrary-shaped ground robot planning, we proposed a novel coarse-to-fine navigation framework that significantly accelerates planning. In the first stage, a sampling-based method selectively generates distinct topological paths that guarantee a minimum inflated margin. In the second stage, a geometry-aware front-end strategy is designed to discretize these topologies into full-state robot motion sequences while concurrently partitioning the paths into SE(2) sub-problems and simpler R2 sub-problems for back-end optimization. In the final stage, an SVSDF-based optimizer generates trajectories tailored to these sub-problems and seamlessly splices them into a continuous final motion plan. Extensive benchmark comparisons show that the proposed method is one to several orders of magnitude faster than the cutting-edge methods in runtime while maintaining a high planning success rate and ensuring CCA.
>
---
#### [replaced 011] RoboMIND 2.0: A Multimodal, Bimanual Mobile Manipulation Dataset for Generalizable Embodied Intelligence
- **分类: cs.RO**

- **简介: 该论文提出RoboMIND 2.0数据集，解决机器人在复杂环境中泛化能力不足的问题，包含多模态、双臂和移动操作数据，支持模仿学习与强化学习研究。**

- **链接: [https://arxiv.org/pdf/2512.24653v2](https://arxiv.org/pdf/2512.24653v2)**

> **作者:** Chengkai Hou; Kun Wu; Jiaming Liu; Zhengping Che; Di Wu; Fei Liao; Guangrun Li; Jingyang He; Qiuxuan Feng; Zhao Jin; Chenyang Gu; Zhuoyang Liu; Nuowei Han; Xiangju Mi; Yaoxu Lv; Yankai Fu; Gaole Dai; Langzhe Gu; Tao Li; Yuheng Zhang; Yixue Zhang; Xinhua Wang; Shichao Fan; Meng Li; Zhen Zhao; Ning Liu; Zhiyuan Xu; Pei Ren; Junjie Ji; Haonan Liu; Kuan Cheng; Shanghang Zhang; Jian Tang
>
> **摘要:** While data-driven imitation learning has revolutionized robotic manipulation, current approaches remain constrained by the scarcity of large-scale, diverse real-world demonstrations. Consequently, the ability of existing models to generalize across long-horizon bimanual tasks and mobile manipulation in unstructured environments remains limited. To bridge this gap, we present RoboMIND 2.0, a comprehensive real-world dataset comprising over 310K dual-arm manipulation trajectories collected across six distinct robot embodiments and 739 complex tasks. Crucially, to support research in contact-rich and spatially extended tasks, the dataset incorporates 12K tactile-enhanced episodes and 20K mobile manipulation trajectories. Complementing this physical data, we construct high-fidelity digital twins of our real-world environments, releasing an additional 20K-trajectory simulated dataset to facilitate robust sim-to-real transfer. To fully exploit the potential of RoboMIND 2.0, we propose MIND-2 system, a hierarchical dual-system frame-work optimized via offline reinforcement learning. MIND-2 integrates a high-level semantic planner (MIND-2-VLM) to decompose abstract natural language instructions into grounded subgoals, coupled with a low-level Vision-Language-Action executor (MIND-2-VLA), which generates precise, proprioception-aware motor actions.
>
---
#### [replaced 012] Steering Flexible Linear Objects in Planar Environments by Two Robot Hands Using Euler's Elastica Solutions
- **分类: cs.RO**

- **简介: 该论文研究机器人手在平面环境中操控柔性线性物体的任务，解决其路径规划与避障问题。通过欧拉弹性解建立模型，实现柔性物体的稳定控制与非自交约束。**

- **链接: [https://arxiv.org/pdf/2501.02874v4](https://arxiv.org/pdf/2501.02874v4)**

> **作者:** Aharon Levin; Elon Rimon; Amir Shapiro
>
> **摘要:** The manipulation of flexible objects such as cables, wires and fresh food items by robot hands forms a special challenge in robot grasp mechanics. This paper considers the steering of flexible linear objects in planar environments by two robot hands. The flexible linear object, modeled as an elastic non-stretchable rod, is manipulated by varying the gripping endpoint positions while keeping equal endpoint tangents. The flexible linear object shape has a closed form solution in terms of the grasp endpoint positions and tangents, called Euler's elastica. This paper obtains the elastica solutions under the optimal control framework, then uses the elastica solutions to obtain closed-form criteria for non self-intersection, stability and obstacle avoidance of the flexible linear object. The new tools are incorporated into a planning scheme for steering flexible linear objects in planar environments populated by sparsely spaced obstacles. The scheme is fully implemented and demonstrated with detailed examples.
>
---
#### [replaced 013] Indicating Robot Vision Capabilities with Augmented Reality
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机协作任务，旨在解决人类对机器人视野认知不准确的问题。通过设计AR视觉指示器并进行实验，验证其提升任务准确性与用户信心的效果。**

- **链接: [https://arxiv.org/pdf/2511.03550v2](https://arxiv.org/pdf/2511.03550v2)**

> **作者:** Hong Wang; Ridhima Phatak; James Ocampo; Zhao Han
>
> **摘要:** Research indicates that humans can mistakenly assume that robots and humans have the same field of view, possessing an inaccurate mental model of robots. This misperception may lead to failures during human-robot collaboration tasks where robots might be asked to complete impossible tasks about out-of-view objects. The issue is more severe when robots do not have a chance to scan the scene to update their world model while focusing on assigned tasks. To help align humans' mental models of robots' vision capabilities, we propose four field-of-view indicators in augmented reality and conducted a human-subjects experiment (N=41) to evaluate them in a collaborative assembly task regarding accuracy, confidence, task efficiency, and workload. These indicators span a spectrum of positions: two at robot's eye and head space -- deepening eye socket and adding blocks to two sides of the eyes (i.e., egocentric), and two anchoring in the robot's task space -- adding extended blocks from the sides of eyes to the table and placing blocks directly on the tables (i.e., allocentric). Results showed that, when placed directly in the task space, the allocentric indicator yields the highest accuracy, although with a delay in interpreting the robot's field of view. When placed at the robot's eyes, the egocentric indicator of deeper eye sockets, possible for physical alteration, also increased accuracy. In all indicators, participants' confidence was high while cognitive load remained low. Finally, we contribute six guidelines for practitioners to apply our augmented reality indicators or physical alterations to align humans' mental models with robots' vision capabilities.
>
---
#### [replaced 014] Chain-of-Action: Trajectory Autoregressive Modeling for Robotic Manipulation
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文提出CoA，一种基于轨迹自回归建模的视觉-运动策略，用于机器人操作任务。旨在解决传统方法预测局部动作不足的问题，通过逆向推理生成全局约束的完整轨迹。**

- **链接: [https://arxiv.org/pdf/2506.09990v2](https://arxiv.org/pdf/2506.09990v2)**

> **作者:** Wenbo Zhang; Tianrun Hu; Hanbo Zhang; Yanyuan Qiao; Yuchu Qin; Yang Li; Jiajun Liu; Tao Kong; Lingqiao Liu; Xiao Ma
>
> **摘要:** We present Chain-of-Action (CoA), a novel visuo-motor policy paradigm built upon Trajectory Autoregressive Modeling. Unlike conventional approaches that predict next step action(s) forward, CoA generates an entire trajectory by explicit backward reasoning with task-specific goals through an action-level Chain-of-Thought (CoT) process. This process is unified within a single autoregressive structure: (1) the first token corresponds to a stable keyframe action that encodes the task-specific goals; and (2) subsequent action tokens are generated autoregressively, conditioned on the initial keyframe and previously predicted actions. This backward action reasoning enforces a global-to-local structure, allowing each local action to be tightly constrained by the final goal. To further realize the action reasoning structure, CoA incorporates four complementary designs: continuous action token representation; dynamic stopping for variable-length trajectory generation; reverse temporal ensemble; and multi-token prediction to balance action chunk modeling with global structure. As a result, CoA gives strong spatial generalization capabilities while preserving the flexibility and simplicity of a visuo-motor policy. Empirically, we observe CoA achieves the state-of-the-art performance across 60 RLBench tasks and 8 real-world manipulation tasks.
>
---
#### [replaced 015] Augmented Reality for RObots (ARRO): Pointing Visuomotor Policies Towards Visual Robustness
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉任务，旨在解决视觉策略对环境变化敏感的问题。通过ARRO技术实时过滤干扰信息，提升策略的鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2505.08627v3](https://arxiv.org/pdf/2505.08627v3)**

> **作者:** Reihaneh Mirjalili; Tobias Jülg; Florian Walter; Wolfram Burgard
>
> **摘要:** Visuomotor policies trained on human expert demonstrations have recently shown strong performance across a wide range of robotic manipulation tasks. However, these policies remain highly sensitive to domain shifts stemming from background or robot embodiment changes, which limits their generalization capabilities. In this paper, we present ARRO, a novel visual representation that leverages zero-shot open-vocabulary segmentation and object detection models to efficiently mask out task-irrelevant regions of the scene in real time without requiring additional training, modeling of the setup, or camera calibration. By filtering visual distractors and overlaying virtual guides during both training and inference, ARRO improves robustness to scene variations and reduces the need for additional data collection. We extensively evaluate ARRO with Diffusion Policy on a range of tabletop manipulation tasks in both simulation and real-world environments, and further demonstrate its compatibility and effectiveness with generalist robot policies, such as Octo, OpenVLA and Pi Zero. Across all settings in our evaluation, ARRO yields consistent performance gains, allows for selective masking to choose between different objects, and shows robustness even to challenging segmentation conditions. Videos showcasing our results are available at: https://augmented-reality-for-robots.github.io/
>
---
#### [replaced 016] FICO: Finite-Horizon Closed-Loop Factorization for Unified Multi-Agent Path Finding
- **分类: cs.RO**

- **简介: 该论文属于多智能体路径规划任务，解决传统方法分离规划与执行的问题。提出FICO算法，整合规划与执行，提升实时性与扩展性。**

- **链接: [https://arxiv.org/pdf/2511.13961v3](https://arxiv.org/pdf/2511.13961v3)**

> **作者:** Jiarui Li; Alessandro Zanardi; Federico Pecora; Runyu Zhang; Gioele Zardini
>
> **摘要:** Multi-Agent Path Finding is a fundamental problem in robotics and AI, yet most existing formulations treat planning and execution separately and address variants of the problem in an ad hoc manner. This paper presents a system-level framework for MAPF that integrates planning and execution, generalizes across variants, and explicitly models uncertainties. At its core is the MAPF system, a formal model that casts MAPF as a control design problem encompassing classical and uncertainty-aware formulations. To solve it, we introduce Finite-Horizon Closed-Loop Factorization (FICO), a factorization-based algorithm inspired by receding-horizon control that exploits compositional structure for efficient closed-loop operation. FICO enables real-time responses -- commencing execution within milliseconds -- while scaling to thousands of agents and adapting seamlessly to execution-time uncertainties. Extensive case studies demonstrate that it reduces computation time by up to two orders of magnitude compared with open-loop baselines, while delivering significantly higher throughput under stochastic delays and agent arrivals. These results establish a principled foundation for analyzing and advancing MAPF through system-level modeling, factorization, and closed-loop design.
>
---
#### [replaced 017] AdaVLN: Towards Visual Language Navigation in Continuous Indoor Environments with Moving Humans
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决机器人在动态室内环境中避开移动人类的导航问题。提出AdaVLN框架及配套数据集，增强导航真实性和可复现性。**

- **链接: [https://arxiv.org/pdf/2411.18539v3](https://arxiv.org/pdf/2411.18539v3)**

> **作者:** Dillon Loh; Tomasz Bednarz; Xinxing Xia; Frank Guan
>
> **摘要:** Visual Language Navigation is a task that challenges robots to navigate in realistic environments based on natural language instructions. While previous research has largely focused on static settings, real-world navigation must often contend with dynamic human obstacles. Hence, we propose an extension to the task, termed Adaptive Visual Language Navigation (AdaVLN), which seeks to narrow this gap. AdaVLN requires robots to navigate complex 3D indoor environments populated with dynamically moving human obstacles, adding a layer of complexity to navigation tasks that mimic the real-world. To support exploration of this task, we also present AdaVLN simulator and AdaR2R datasets. The AdaVLN simulator enables easy inclusion of fully animated human models directly into common datasets like Matterport3D. We also introduce a "freeze-time" mechanism for both the navigation task and simulator, which pauses world state updates during agent inference, enabling fair comparisons and experimental reproducibility across different hardware. We evaluate several baseline models on this task, analyze the unique challenges introduced by AdaVLN, and demonstrate its potential to bridge the sim-to-real gap in VLN research.
>
---
