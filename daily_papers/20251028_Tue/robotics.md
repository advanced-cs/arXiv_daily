# 机器人 cs.RO

- **最新发布 92 篇**

- **更新 45 篇**

## 最新发布

#### [new 001] TWC-SLAM: Multi-Agent Cooperative SLAM with Text Semantics and WiFi Features Integration for Similar Indoor Environments
- **分类: cs.RO**

- **简介: 该论文提出TWC-SLAM框架，解决多智能体在相似室内环境中因结构重复导致的定位与建图误差问题。通过融合文本语义与WiFi信号特征，提升位置识别与回环检测精度，实现更准确的协同建图与点云对齐。**

- **链接: [http://arxiv.org/pdf/2510.22754v1](http://arxiv.org/pdf/2510.22754v1)**

> **作者:** Chunyu Li; Shoubin Chen; Dong Li; Weixing Xue; Qingquan Li
>
> **备注:** Accepted by the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Multi-agent cooperative SLAM often encounters challenges in similar indoor environments characterized by repetitive structures, such as corridors and rooms. These challenges can lead to significant inaccuracies in shared location identification when employing point cloud-based techniques. To mitigate these issues, we introduce TWC-SLAM, a multi-agent cooperative SLAM framework that integrates text semantics and WiFi signal features to enhance location identification and loop closure detection. TWC-SLAM comprises a single-agent front-end odometry module based on FAST-LIO2, a location identification and loop closure detection module that leverages text semantics and WiFi features, and a global mapping module. The agents are equipped with sensors capable of capturing textual information and detecting WiFi signals. By correlating these data sources, TWC-SLAM establishes a common location, facilitating point cloud alignment across different agents' maps. Furthermore, the system employs loop closure detection and optimization modules to achieve global optimization and cohesive mapping. We evaluated our approach using an indoor dataset featuring similar corridors, rooms, and text signs. The results demonstrate that TWC-SLAM significantly improves the performance of cooperative SLAM systems in complex environments with repetitive architectural features.
>
---
#### [new 002] Kinematically Controllable Cable Robots with Reconfigurable End-effectors
- **分类: cs.RO**

- **简介: 该论文针对缆绳机器人因增加缆绳导致旋转工作空间受限及张力解非唯一的问题，设计了结构简单的可重构末端执行器。通过弹簧、螺旋槽轴与螺母实现直线运动转旋转，结合轴承引入额外旋转自由度，使系统非冗余，仅靠运动学即可实现精确控制。**

- **链接: [http://arxiv.org/pdf/2510.22825v1](http://arxiv.org/pdf/2510.22825v1)**

> **作者:** Nan Zhang
>
> **备注:** 7 pages, 12 figures, Technical Report
>
> **摘要:** To enlarge the translational workspace of cable-driven robots, one common approach is to increase the number of cables. However, this introduces two challenges: (1) cable interference significantly reduces the rotational workspace, and (2) the solution of tensions in cables becomes non-unique, resulting in difficulties for kinematic control of the robot. In this work, we design structurally simple reconfigurable end-effectors for cable robots. By incorporating a spring, a helical-grooved shaft, and a matching nut, relative linear motions between end-effector components are converted into relative rotations, thereby expanding the rotational workspace of the mechanism. Meanwhile, a bearing is introduced to provide an additional rotational degree of freedom, making the mechanism non-redundant. As a result, the robot's motion can be controlled purely through kinematics without additional tension sensing and control.
>
---
#### [new 003] Learn2Drive: A neural network-based framework for socially compliant automated vehicle control
- **分类: cs.RO; cs.AI; cs.LG; cs.MA; cs.SY; eess.SY**

- **简介: 该论文提出Learn2Drive框架，解决自动驾驶车辆在自适应巡航中忽视与人类驾驶车辆互动及交通流整体效率的问题。通过引入社会价值取向的LSTM神经网络控制模型，优化车辆间协同行为，提升交通流畅性与能效。**

- **链接: [http://arxiv.org/pdf/2510.21736v1](http://arxiv.org/pdf/2510.21736v1)**

> **作者:** Yuhui Liu; Samannita Halder; Shian Wang; Tianyi Li
>
> **摘要:** This study introduces a novel control framework for adaptive cruise control (ACC) in automated driving, leveraging Long Short-Term Memory (LSTM) networks and physics-informed constraints. As automated vehicles (AVs) adopt advanced features like ACC, transportation systems are becoming increasingly intelligent and efficient. However, existing AV control strategies primarily focus on optimizing the performance of individual vehicles or platoons, often neglecting their interactions with human-driven vehicles (HVs) and the broader impact on traffic flow. This oversight can exacerbate congestion and reduce overall system efficiency. To address this critical research gap, we propose a neural network-based, socially compliant AV control framework that incorporates social value orientation (SVO). This framework enables AVs to account for their influence on HVs and traffic dynamics. By leveraging AVs as mobile traffic regulators, the proposed approach promotes adaptive driving behaviors that reduce congestion, improve traffic efficiency, and lower energy consumption. Within this framework, we define utility functions for both AVs and HVs, which are optimized based on the SVO of each AV to balance its own control objectives with broader traffic flow considerations. Numerical results demonstrate the effectiveness of the proposed method in adapting to varying traffic conditions, thereby enhancing system-wide efficiency. Specifically, when the AV's control mode shifts from prioritizing energy consumption to optimizing traffic flow efficiency, vehicles in the following platoon experience at least a 58.99% increase in individual energy consumption alongside at least a 38.39% improvement in individual average speed, indicating significant enhancements in traffic dynamics.
>
---
#### [new 004] Reliable Robotic Task Execution in the Face of Anomalies
- **分类: cs.RO**

- **简介: 该论文针对机器人在开放环境中执行任务时因异常导致失败的问题，提出一种融合视觉异常检测与三级恢复机制的框架。通过训练异常检测模型并集成至在线执行流程，实现对异常的识别与响应，提升机器人任务执行的可靠性与安全性。**

- **链接: [http://arxiv.org/pdf/2510.23121v1](http://arxiv.org/pdf/2510.23121v1)**

> **作者:** Bharath Santhanam; Alex Mitrevski; Santosh Thoduka; Sebastian Houben; Teena Hassan
>
> **备注:** Accepted for publication in IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Learned robot policies have consistently been shown to be versatile, but they typically have no built-in mechanism for handling the complexity of open environments, making them prone to execution failures; this implies that deploying policies without the ability to recognise and react to failures may lead to unreliable and unsafe robot behaviour. In this paper, we present a framework that couples a learned policy with a method to detect visual anomalies during policy deployment and to perform recovery behaviours when necessary, thereby aiming to prevent failures. Specifically, we train an anomaly detection model using data collected during nominal executions of a trained policy. This model is then integrated into the online policy execution process, so that deviations from the nominal execution can trigger a three-level sequential recovery process that consists of (i) pausing the execution temporarily, (ii) performing a local perturbation of the robot's state, and (iii) resetting the robot to a safe state by sampling from a learned execution success model. We verify our proposed method in two different scenarios: (i) a door handle reaching task with a Kinova Gen3 arm using a policy trained in simulation and transferred to the real robot, and (ii) an object placing task with a UFactory xArm 6 using a general-purpose policy model. Our results show that integrating policy execution with anomaly detection and recovery increases the execution success rate in environments with various anomalies, such as trajectory deviations and adversarial human interventions.
>
---
#### [new 005] Analytical Swarm Chemistry: Characterization and Analysis of Emergent Swarm Behaviors
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出“解析群集化学”框架，融合工程、多智能体与化学概念，通过相图分析系统化研究群集参数对涌现行为的影响。旨在解决群集机器人在真实场景中因难以预测涌现行为而导致部署困难的问题。工作包括定义宏观状态、构建参数空间相图，并验证其在真实机器人上的有效性，实现可预测的群集行为设计。**

- **链接: [http://arxiv.org/pdf/2510.22821v1](http://arxiv.org/pdf/2510.22821v1)**

> **作者:** Ricardo Vega; Connor Mattson; Kevin Zhu; Daniel S. Brown; Cameron Nowzari
>
> **备注:** 9 pages, 8 figures, 1 table
>
> **摘要:** Swarm robotics has potential for a wide variety of applications, but real-world deployments remain rare due to the difficulty of predicting emergent behaviors arising from simple local interactions. Traditional engineering approaches design controllers to achieve desired macroscopic outcomes under idealized conditions, while agent-based and artificial life studies explore emergent phenomena in a bottom-up, exploratory manner. In this work, we introduce Analytical Swarm Chemistry, a framework that integrates concepts from engineering, agent-based and artificial life research, and chemistry. This framework combines macrostate definitions with phase diagram analysis to systematically explore how swarm parameters influence emergent behavior. Inspired by concepts from chemistry, the framework treats parameters like thermodynamic variables, enabling visualization of regions in parameter space that give rise to specific behaviors. Applying this framework to agents with minimally viable capabilities, we identify sufficient conditions for behaviors such as milling and diffusion and uncover regions of the parameter space that reliably produce these behaviors. Preliminary validation on real robots demonstrates that these regions correspond to observable behaviors in practice. By providing a principled, interpretable approach, this framework lays the groundwork for predictable and reliable emergent behavior in real-world swarm systems.
>
---
#### [new 006] Two-Steps Diffusion Policy for Robotic Manipulation via Genetic Denoising
- **分类: cs.RO; cs.AI; 68T40, 93C85, 68T07, 68U35**

- **简介: 该论文针对机器人操作中的扩散策略推理效率低的问题，提出基于遗传去噪的两步扩散策略。通过适应动作分布的低维结构，显著减少神经函数评估次数（最低仅2次），在14个任务上实现更高或相当性能，大幅降低推理成本。**

- **链接: [http://arxiv.org/pdf/2510.21991v1](http://arxiv.org/pdf/2510.21991v1)**

> **作者:** Mateo Clemente; Leo Brunswic; Rui Heng Yang; Xuan Zhao; Yasser Khalil; Haoyu Lei; Amir Rasouli; Yinchuan Li
>
> **备注:** 16 pages, 11 figure, 2 tables, accepted at Neurips 2025
>
> **摘要:** Diffusion models, such as diffusion policy, have achieved state-of-the-art results in robotic manipulation by imitating expert demonstrations. While diffusion models were originally developed for vision tasks like image and video generation, many of their inference strategies have been directly transferred to control domains without adaptation. In this work, we show that by tailoring the denoising process to the specific characteristics of embodied AI tasks -- particularly structured, low-dimensional nature of action distributions -- diffusion policies can operate effectively with as few as 5 neural function evaluations (NFE). Building on this insight, we propose a population-based sampling strategy, genetic denoising, which enhances both performance and stability by selecting denoising trajectories with low out-of-distribution risk. Our method solves challenging tasks with only 2 NFE while improving or matching performance. We evaluate our approach across 14 robotic manipulation tasks from D4RL and Robomimic, spanning multiple action horizons and inference budgets. In over 2 million evaluations, our method consistently outperforms standard diffusion-based policies, achieving up to 20\% performance gains with significantly fewer inference steps.
>
---
#### [new 007] RL-AVIST: Reinforcement Learning for Autonomous Visual Inspection of Space Targets
- **分类: cs.RO**

- **简介: 该论文提出RL-AVIST框架，用于航天器自主视觉巡检任务。针对传统控制在模型不确定性与复杂场景下的适应性不足问题，采用基于模型的强化学习（DreamerV3）在高保真仿真中训练智能体，实现对月球门户等目标的高精度近距机动。对比了泛化与专用策略，并验证了模型在多构型任务中的鲁棒性与可扩展性。**

- **链接: [http://arxiv.org/pdf/2510.22699v1](http://arxiv.org/pdf/2510.22699v1)**

> **作者:** Matteo El-Hariry; Andrej Orsula; Matthieu Geist; Miguel Olivares-Mendez
>
> **摘要:** The growing need for autonomous on-orbit services such as inspection, maintenance, and situational awareness calls for intelligent spacecraft capable of complex maneuvers around large orbital targets. Traditional control systems often fall short in adaptability, especially under model uncertainties, multi-spacecraft configurations, or dynamically evolving mission contexts. This paper introduces RL-AVIST, a Reinforcement Learning framework for Autonomous Visual Inspection of Space Targets. Leveraging the Space Robotics Bench (SRB), we simulate high-fidelity 6-DOF spacecraft dynamics and train agents using DreamerV3, a state-of-the-art model-based RL algorithm, with PPO and TD3 as model-free baselines. Our investigation focuses on 3D proximity maneuvering tasks around targets such as the Lunar Gateway and other space assets. We evaluate task performance under two complementary regimes: generalized agents trained on randomized velocity vectors, and specialized agents trained to follow fixed trajectories emulating known inspection orbits. Furthermore, we assess the robustness and generalization of policies across multiple spacecraft morphologies and mission domains. Results demonstrate that model-based RL offers promising capabilities in trajectory fidelity, and sample efficiency, paving the way for scalable, retrainable control solutions for future space operations
>
---
#### [new 008] Curriculum-Based Iterative Self-Play for Scalable Multi-Drone Racing
- **分类: cs.RO; cs.AI; cs.MA; cs.SY; eess.SY; I.2.9; I.2.11; I.2.6**

- **简介: 该论文针对多无人机高速竞速中的协同控制难题，提出CRUISE框架，结合渐进式难度课程与高效自对弈机制，实现可扩展的强化学习训练。实验表明，该方法显著提升竞速速度与成功率，优于基线与现有博弈规划方法，验证了课程设计的关键作用。**

- **链接: [http://arxiv.org/pdf/2510.22570v1](http://arxiv.org/pdf/2510.22570v1)**

> **作者:** Onur Akgün
>
> **备注:** 13 pages, 5 figures. This paper is currently under review at the journal Engineering Applications of Artificial Intelligence. Supplementary video: https://drive.google.com/file/d/1k7necen2DgIxaYT2alKK8-b20sE_AyDA/view Source code and models: https://doi.org/10.5281/zenodo.17256943
>
> **摘要:** The coordination of multiple autonomous agents in high-speed, competitive environments represents a significant engineering challenge. This paper presents CRUISE (Curriculum-Based Iterative Self-Play for Scalable Multi-Drone Racing), a reinforcement learning framework designed to solve this challenge in the demanding domain of multi-drone racing. CRUISE overcomes key scalability limitations by synergistically combining a progressive difficulty curriculum with an efficient self-play mechanism to foster robust competitive behaviors. Validated in high-fidelity simulation with realistic quadrotor dynamics, the resulting policies significantly outperform both a standard reinforcement learning baseline and a state-of-the-art game-theoretic planner. CRUISE achieves nearly double the planner's mean racing speed, maintains high success rates, and demonstrates robust scalability as agent density increases. Ablation studies confirm that the curriculum structure is the critical component for this performance leap. By providing a scalable and effective training methodology, CRUISE advances the development of autonomous systems for dynamic, competitive tasks and serves as a blueprint for future real-world deployment.
>
---
#### [new 009] Deductive Chain-of-Thought Augmented Socially-aware Robot Navigation World Model
- **分类: cs.RO**

- **简介: 该论文针对社交机器人导航中大语言模型（LLM）缺乏物理约束与逻辑一致性的问题，提出NaviWM世界模型。通过融合时空世界模型与基于一阶逻辑的演绎推理链，增强LLM在动态人群中的安全、合规决策能力。实验表明，该方法显著提升导航成功率并减少社交违规。**

- **链接: [http://arxiv.org/pdf/2510.23509v1](http://arxiv.org/pdf/2510.23509v1)**

> **作者:** Weizheng Wang; Obi Ike; Soyun Choi; Sungeun Hong; Byung-Cheol Min
>
> **摘要:** Social robot navigation increasingly relies on large language models for reasoning, path planning, and enabling movement in dynamic human spaces. However, relying solely on LLMs for planning often leads to unpredictable and unsafe behaviors, especially in dynamic human spaces, due to limited physical grounding and weak logical consistency. In this work, we introduce NaviWM, a socially-aware robot Navigation World Model that augments LLM reasoning with a structured world model and a logic-driven chain-of-thought process. NaviWM consists of two main components: (1) a spatial-temporal world model that captures the positions, velocities, and activities of agents in the environment, and (2) a deductive reasoning module that guides LLMs through a multi-step, logic-based inference process. This integration enables the robot to generate navigation decisions that are both socially compliant and physically safe, under well-defined constraints such as personal space, collision avoidance, and timing. Unlike previous methods based on prompting or fine-tuning, NaviWM encodes social norms as first-order logic, enabling interpretable and verifiable reasoning. Experiments show that NaviWM improves success rates and reduces social violations, particularly in crowded environments. These results demonstrate the benefit of combining formal reasoning with LLMs for robust social navigation. Additional experimental details and demo videos for this work can be found at: https://sites.google.com/view/NaviWM.
>
---
#### [new 010] Full-Dynamics Real-Time Nonlinear Model Predictive Control of Heavy-Duty Hydraulic Manipulator for Trajectory Tracking Tasks
- **分类: cs.RO**

- **简介: 该论文针对重型液压机械臂的高精度轨迹跟踪任务，解决其在实时控制中满足关节与末端执行器多重物理约束（力、速度、位置）的问题。提出一种基于多段射击法的非线性模型预测控制框架，结合实时反馈与虚拟分解控制，实现1kHz下全动态约束满足的实时控制，实验验证了其高精度与安全性。**

- **链接: [http://arxiv.org/pdf/2510.23386v1](http://arxiv.org/pdf/2510.23386v1)**

> **作者:** Alvaro Paz; Mahdi Hejrati; Pauli Mustalahti; Jouni Mattila
>
> **备注:** This work has been submitted for possible publication in IEEE
>
> **摘要:** Heavy-duty hydraulic manipulators (HHMs) operate under strict physical and safety-critical constraints due to their large size, high power, and complex nonlinear dynamics. Ensuring that both joint-level and end-effector trajectories remain compliant with actuator capabilities, such as force, velocity, and position limits, is essential for safe and reliable operation, yet remains largely underexplored in real-time control frameworks. This paper presents a nonlinear model predictive control (NMPC) framework designed to guarantee constraint satisfaction throughout the full nonlinear dynamics of HHMs, while running at a real-time control frequency of 1 kHz. The proposed method combines a multiple-shooting strategy with real-time sensor feedback, and is supported by a robust low-level controller based on virtual decomposition control (VDC) for precise joint tracking. Experimental validation on a full-scale hydraulic manipulator shows that the NMPC framework not only enforces actuator constraints at the joint level, but also ensures constraint-compliant motion in Cartesian space for the end-effector. These results demonstrate the method's capability to deliver high-accuracy trajectory tracking while strictly respecting safety-critical limits, setting a new benchmark for real-time control in large-scale hydraulic systems.
>
---
#### [new 011] Estimation of Minimum Stride Frequency for the Frontal Plane Stability of Bipedal Systems
- **分类: cs.RO**

- **简介: 该论文研究双足系统在矢状面的稳定性，旨在确定维持稳定步态所需的最小步频。通过分析质量、刚度、腿长和髋宽等参数影响，提出预测最小步频的方法，并验证其准确性，以实现无需反馈控制的前馈稳定，降低控制复杂度与能耗。**

- **链接: [http://arxiv.org/pdf/2510.22030v1](http://arxiv.org/pdf/2510.22030v1)**

> **作者:** Harsha Karunanayaka; Siavash Rezazadeh
>
> **摘要:** Stability of bipedal systems in frontal plane is affected by the hip offset, to the extent that adjusting stride time using feedforward retraction and extension of the legs can lead to stable oscillations without feedback control. This feedforward stabilization can be leveraged to reduce the control effort and energy expenditure and increase the locomotion robustness. However, there is limited understanding of how key parameters, such as mass, stiffness, leg length, and hip width, affect stability and the minimum stride frequency needed to maintain it. This study aims to address these gaps through analyzing how individual model parameters and the system's natural frequency influence the minimum stride frequency required to maintain a stable cycle. We propose a method to predict the minimum stride frequency, and compare the predicted stride frequencies with actual values for randomly generated models. The findings of this work provide a better understanding of the frontal plane stability mechanisms and how feedforward stabilization can be leveraged to reduce the control effort.
>
---
#### [new 012] SPIRAL: Self-Play Incremental Racing Algorithm for Learning in Multi-Drone Competitions
- **分类: cs.RO; cs.AI; cs.MA; cs.SY; eess.SY; I.2.9; I.2.11; I.2.6**

- **简介: 该论文提出SPIRAL，一种用于多无人机竞速的自对弈增量学习算法。针对多智能体竞速中复杂协作与动态挑战难题，通过自对弈机制逐步提升对手难度，驱动无人机从基础飞行到高级协同策略的进化。方法可兼容主流深度强化学习算法，显著提升训练效率与策略鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.22568v1](http://arxiv.org/pdf/2510.22568v1)**

> **作者:** Onur Akgün
>
> **备注:** \c{opyright} 2025 IEEE. Personal use of this material is permitted. Permission from IEEE must be obtained for all other uses, in any current or future media, including reprinting/republishing this material for advertising or promotional purposes, creating new collective works, for resale or redistribution to servers or lists, or reuse of any copyrighted component of this work in other works
>
> **摘要:** This paper introduces SPIRAL (Self-Play Incremental Racing Algorithm for Learning), a novel approach for training autonomous drones in multi-agent racing competitions. SPIRAL distinctively employs a self-play mechanism to incrementally cultivate complex racing behaviors within a challenging, dynamic environment. Through this self-play core, drones continuously compete against increasingly proficient versions of themselves, naturally escalating the difficulty of competitive interactions. This progressive learning journey guides agents from mastering fundamental flight control to executing sophisticated cooperative multi-drone racing strategies. Our method is designed for versatility, allowing integration with any state-of-the-art Deep Reinforcement Learning (DRL) algorithms within its self-play framework. Simulations demonstrate the significant advantages of SPIRAL and benchmark the performance of various DRL algorithms operating within it. Consequently, we contribute a versatile, scalable, and self-improving learning framework to the field of autonomous drone racing. SPIRAL's capacity to autonomously generate appropriate and escalating challenges through its self-play dynamic offers a promising direction for developing robust and adaptive racing strategies in multi-agent environments. This research opens new avenues for enhancing the performance and reliability of autonomous racing drones in increasingly complex and competitive scenarios.
>
---
#### [new 013] Awakening Facial Emotional Expressions in Human-Robot
- **分类: cs.RO**

- **简介: 该论文聚焦于人形机器人面部表情自动生成任务，旨在解决现有机器人依赖人工编程、缺乏自主学习能力的问题。研究设计了类生物机器人面部系统，构建首个开源人脸表情数据集，并提出基于KAN与注意力机制的端到端学习框架，实现机器人对人类表情的精准多样模仿。**

- **链接: [http://arxiv.org/pdf/2510.23059v1](http://arxiv.org/pdf/2510.23059v1)**

> **作者:** Yongtong Zhu; Lei Li; Iggy Qian; WenBin Zhou; Ye Yuan; Qingdu Li; Na Liu; Jianwei Zhang
>
> **备注:** Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025). 8 pages, 7 figures, IEEE two-column format
>
> **摘要:** The facial expression generation capability of humanoid social robots is critical for achieving natural and human-like interactions, playing a vital role in enhancing the fluidity of human-robot interactions and the accuracy of emotional expression. Currently, facial expression generation in humanoid social robots still relies on pre-programmed behavioral patterns, which are manually coded at high human and time costs. To enable humanoid robots to autonomously acquire generalized expressive capabilities, they need to develop the ability to learn human-like expressions through self-training. To address this challenge, we have designed a highly biomimetic robotic face with physical-electronic animated facial units and developed an end-to-end learning framework based on KAN (Kolmogorov-Arnold Network) and attention mechanisms. Unlike previous humanoid social robots, we have also meticulously designed an automated data collection system based on expert strategies of facial motion primitives to construct the dataset. Notably, to the best of our knowledge, this is the first open-source facial dataset for humanoid social robots. Comprehensive evaluations indicate that our approach achieves accurate and diverse facial mimicry across different test subjects.
>
---
#### [new 014] Real-time Mixed-Integer Quadratic Programming for Driving Behavior-Inspired Speed Bump Optimal Trajectory Planning
- **分类: cs.RO**

- **简介: 该论文针对自动驾驶车辆在城市道路中安全舒适通过减速带的轨迹规划问题，提出基于实时混合整数二次规划（MIQP）的方法。通过融合人类驾驶行为与模型预测控制（MPC），实现对减速带的高效、平顺穿越，兼顾乘客舒适性与计算效率，适用于复杂动态环境下的实时决策。**

- **链接: [http://arxiv.org/pdf/2510.21751v1](http://arxiv.org/pdf/2510.21751v1)**

> **作者:** Van Nam Dinh; Van Vy Phan; Thai Son Dang; Van Du Phan; The Anh Mai; Van Chuong Le; Sy Phuong Ho; Dinh Tu Duong; Hung Cuong Ta
>
> **摘要:** This paper proposes a novel methodology for trajectory planning in autonomous vehicles (AVs), addressing the complex challenge of negotiating speed bumps within a unified Mixed-Integer Quadratic Programming (MIQP) framework. By leveraging Model Predictive Control (MPC), we develop trajectories that optimize both the traversal of speed bumps and overall passenger comfort. A key contribution of this work is the formulation of speed bump handling constraints that closely emulate human driving behavior, seamlessly integrating these with broader road navigation requirements. Through extensive simulations in varied urban driving environments, we demonstrate the efficacy of our approach, highlighting its ability to ensure smooth speed transitions over speed bumps while maintaining computational efficiency suitable for real-time deployment. The method's capability to handle both static road features and dynamic constraints, alongside expert human driving, represents a significant step forward in trajectory planning for urban
>
---
#### [new 015] An Automated Tape Laying System Employing a Uniaxial Force Control Device
- **分类: cs.RO**

- **简介: 该论文针对复合材料自动化铺放中的层间粘结问题，提出一种集成单轴力控与温控的自动铺带系统。通过固定铺带头、机器人移动工件的方式，实现复杂曲面高效铺放，验证了碳纤维增强HDPE带的可行性，解决了铺放过程中的压力与温度控制难题。**

- **链接: [http://arxiv.org/pdf/2510.23109v1](http://arxiv.org/pdf/2510.23109v1)**

> **作者:** Bernhard Rameder; Hubert Gattringer; Ronald Naderer; Andreas Mueller
>
> **备注:** Proceedings ECCM21 - 21st European Conference on Composite Materials, Nantes, France, 7-2024
>
> **摘要:** This paper deals with the design of a cost effective automated tape laying system (ATL system) with integrated uniaxial force control to ensure the necessary compaction forces as well as with an accurate temperature control to guarantee the used tape being melted appropriate. It is crucial to control the substrate and the oncoming tape onto a specific temperature level to ensure an optimal consolidation between the different layers of the product. Therefore, it takes several process steps from the spooled tape on the coil until it is finally tacked onto the desired mold. The different modules are divided into the tape storage spool, a tape-guiding roller, a tape processing unit, a heating zone and the consolidation unit. Moreover, a special robot control concept for testing the ATL system is presented. In contrast to many other systems, with this approach, the tape laying device is spatially fixed and the shape is moved accordingly by the robot, which allows for handling of rather compact and complex shapes. The functionality of the subsystems and the taping process itself was finally approved in experimental results using a carbon fiber reinforced HDPE tape.
>
---
#### [new 016] Never Too Rigid to Reach: Adaptive Virtual Model Control with LLM- and Lyapunov-Based Reinforcement Learning
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对机器人臂在不确定环境中控制僵化、稳定性差的问题，提出基于大语言模型（LLM）与李雅普诺夫理论的自适应虚拟模型控制方法。通过LLM提升任务理解与协调能力，结合Lyapunov约束保障在线适应的安全性，实现高效、稳定、可解释的动态任务执行。**

- **链接: [http://arxiv.org/pdf/2510.22892v1](http://arxiv.org/pdf/2510.22892v1)**

> **作者:** Jingzehua Xu; Yangyang Li; Yangfei Chen; Guanwen Xie; Shuai Zhang
>
> **摘要:** Robotic arms are increasingly deployed in uncertain environments, yet conventional control pipelines often become rigid and brittle when exposed to perturbations or incomplete information. Virtual Model Control (VMC) enables compliant behaviors by embedding virtual forces and mapping them into joint torques, but its reliance on fixed parameters and limited coordination among virtual components constrains adaptability and may undermine stability as task objectives evolve. To address these limitations, we propose Adaptive VMC with Large Language Model (LLM)- and Lyapunov-Based Reinforcement Learning (RL), which preserves the physical interpretability of VMC while supporting stability-guaranteed online adaptation. The LLM provides structured priors and high-level reasoning that enhance coordination among virtual components, improve sample efficiency, and facilitate flexible adjustment to varying task requirements. Complementarily, Lyapunov-based RL enforces theoretical stability constraints, ensuring safe and reliable adaptation under uncertainty. Extensive simulations on a 7-DoF Panda arm demonstrate that our approach effectively balances competing objectives in dynamic tasks, achieving superior performance while highlighting the synergistic benefits of LLM guidance and Lyapunov-constrained adaptation.
>
---
#### [new 017] Precise Time Delay Measurement and Compensation for Tightly Coupled Underwater SINS/piUSBL Navigation
- **分类: cs.RO**

- **简介: 该论文针对水下惯性/声学组合导航中的时间延迟问题，提出一种紧耦合框架。通过同步定时与声信号处理，实现对声传播及系统处理延迟的精确测量与补偿，提升导航精度。解决了多传感器时间不同步导致的定位误差问题。**

- **链接: [http://arxiv.org/pdf/2510.23286v1](http://arxiv.org/pdf/2510.23286v1)**

> **作者:** Jin Huang; Yingqiang Wang; Haoda Li; Zichen Liu; Zhikun Wang; Ying Chen
>
> **摘要:** In multi-sensor systems, time synchronization between sensors is a significant challenge, and this issue is particularly pronounced in underwater integrated navigation systems incorporating acoustic positioning. Such systems are highly susceptible to time delay, which can significantly degrade accuracy when measurement and fusion moments are misaligned. To address this challenge, this paper introduces a tightly coupled navigation framework that integrates a passive inverted ultra-short baseline (piUSBL) acoustic positioning system, a strapdown inertial navigation system (SINS), and a depth gauge under precise time synchronization. The framework fuses azimuth and slant range from the piUSBL with depth data, thereby avoiding poor vertical-angle observability in planar arrays. A novel delay measurement strategy is introduced, combining synchronized timing with acoustic signal processing, which redefines delay-traditionally an unobservable error-into a quantifiable parameter, enabling explicit estimation of both acoustic propagation and system processing delays. Simulations and field experiments confirm the feasibility of the proposed method, with delay-compensated navigation reducing RMSE by 40.45% and maximum error by 32.55%. These findings show that precise delay measurement and compensation not only enhance underwater navigation accuracy but also establish a generalizable framework for acoustic positioning integration, offering valuable insights into time alignment and data fusion in latency-sensitive multi-sensor systems.
>
---
#### [new 018] A Novel Multi-Timescale Stability-Preserving Hierarchical Reinforcement Learning Controller Framework for Adaptive Control in High-Dimensional Dynamical Systems
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对高维随机系统控制中的维数灾难、时间抽象缺失与稳定性难保障问题，提出多时标李雅普诺夫约束分层强化学习框架（MTLHRL）。通过分层策略与神经李雅普诺夫函数，实现稳定、高效控制，显著提升性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.22420v1](http://arxiv.org/pdf/2510.22420v1)**

> **作者:** Mohammad Ali Labbaf Khaniki; Fateme Taroodi; Benyamin Safizadeh
>
> **摘要:** Controlling high-dimensional stochastic systems, critical in robotics, autonomous vehicles, and hyperchaotic systems, faces the curse of dimensionality, lacks temporal abstraction, and often fails to ensure stochastic stability. To overcome these limitations, this study introduces the Multi-Timescale Lyapunov-Constrained Hierarchical Reinforcement Learning (MTLHRL) framework. MTLHRL integrates a hierarchical policy within a semi-Markov Decision Process (SMDP), featuring a high-level policy for strategic planning and a low-level policy for reactive control, which effectively manages complex, multi-timescale decision-making and reduces dimensionality overhead. Stability is rigorously enforced using a neural Lyapunov function optimized via Lagrangian relaxation and multi-timescale actor-critic updates, ensuring mean-square boundedness or asymptotic stability in the face of stochastic dynamics. The framework promotes efficient and reliable learning through trust-region constraints and decoupled optimization. Extensive simulations on an 8D hyperchaotic system and a 5-DOF robotic manipulator demonstrate MTLHRL's empirical superiority. It significantly outperforms baseline methods in both stability and performance, recording the lowest error indices (e.g., Integral Absolute Error (IAE): 3.912 in hyperchaotic control and IAE: 1.623 in robotics), achieving faster convergence, and exhibiting superior disturbance rejection. MTLHRL offers a theoretically grounded and practically viable solution for robust control of complex stochastic systems.
>
---
#### [new 019] Avi: Action from Volumetric Inference
- **分类: cs.RO**

- **简介: 该论文提出Avi，一种基于3D视觉-语言-动作的架构，将机器人动作生成视为3D感知与空间推理问题。针对现有模型依赖2D图像和端到端策略学习导致泛化性差的问题，Avi利用3D点云与语言理解，通过几何变换显式计算动作，无需训练历史动作，提升对遮挡、视角变化的鲁棒性，实现从语言指令到动作的可解释推理。**

- **链接: [http://arxiv.org/pdf/2510.21746v1](http://arxiv.org/pdf/2510.21746v1)**

> **作者:** Harris Song; Long Le
>
> **备注:** NeurIPS 2025 Workshop on Embodied World Models for Decision Making. URL: https://avi-3drobot.github.io/
>
> **摘要:** We propose Avi, a novel 3D Vision-Language-Action (VLA) architecture that reframes robotic action generation as a problem of 3D perception and spatial reasoning, rather than low-level policy learning. While existing VLA models primarily operate on 2D visual inputs and are trained end-to-end on task-specific action policies, Avi leverages 3D point clouds and language-grounded scene understanding to compute actions through classical geometric transformations. Most notably, Avi does not train on previous action tokens, rather, we build upon a 3D Multi-modal Large Language Model (MLLM) to generate the next point cloud and explicitly calculate the actions through classical transformations. This approach enables generalizable behaviors that are robust to occlusions, camera pose variations, and changes in viewpoint. By treating the robotic decision-making process as a structured reasoning task over 3D representations, Avi bridges the gap between high-level language instructions and low-level actuation without requiring opaque policy learning. Our preliminary results highlight the potential of 3D vision-language reasoning as a foundation for scalable, robust robotic systems. Check it out at https://avi-3drobot.github.io/.
>
---
#### [new 020] Seq-DeepIPC: Sequential Sensing for End-to-End Control in Legged Robot Navigation
- **分类: cs.RO; cs.CV; cs.SY; eess.IV; eess.SY**

- **简介: 该论文提出Seq-DeepIPC，面向足式机器人在真实环境中的端到端导航任务。针对感知与控制耦合不足的问题，融合RGB-D与GNSS多模态数据，通过时序融合提升感知精度，并简化航向估计。采用轻量编码器实现高效部署，验证了序列输入对性能的提升。**

- **链接: [http://arxiv.org/pdf/2510.23057v1](http://arxiv.org/pdf/2510.23057v1)**

> **作者:** Oskar Natan; Jun Miura
>
> **备注:** Preprint notice, this manuscript has been submitted to IEEE sensors journal for possible publication
>
> **摘要:** We present Seq-DeepIPC, a sequential end-to-end perception-to-control model for legged robot navigation in realworld environments. Seq-DeepIPC advances intelligent sensing for autonomous legged navigation by tightly integrating multi-modal perception (RGB-D + GNSS) with temporal fusion and control. The model jointly predicts semantic segmentation and depth estimation, giving richer spatial features for planning and control. For efficient deployment on edge devices, we use EfficientNet-B0 as the encoder, reducing computation while maintaining accuracy. Heading estimation is simplified by removing the noisy IMU and instead computing the bearing angle directly from consecutive GNSS positions. We collected a larger and more diverse dataset that includes both road and grass terrains, and validated Seq-DeepIPC on a robot dog. Comparative and ablation studies show that sequential inputs improve perception and control in our models, while other baselines do not benefit. Seq-DeepIPC achieves competitive or better results with reasonable model size; although GNSS-only heading is less reliable near tall buildings, it is robust in open areas. Overall, Seq-DeepIPC extends end-to-end navigation beyond wheeled robots to more versatile and temporally-aware systems. To support future research, we will release the codes to our GitHub repository at https://github.com/oskarnatan/Seq-DeepIPC.
>
---
#### [new 021] Estimating Continuum Robot Shape under External Loading using Spatiotemporal Neural Networks
- **分类: cs.RO**

- **简介: 该论文针对柔性连续体机器人在外部载荷下的三维形状估计问题，提出一种融合时空神经网络的感知方法。通过融合肌腱位移与视觉图像数据，实现高精度点云重建，并用贝塞尔曲线拟合获得连续形状。实验表明，该方法在无载和有载条件下均优于现有技术。**

- **链接: [http://arxiv.org/pdf/2510.22339v1](http://arxiv.org/pdf/2510.22339v1)**

> **作者:** Enyi Wang; Zhen Deng; Chuanchuan Pan; Bingwei He; Jianwei Zhang
>
> **备注:** 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** This paper presents a learning-based approach for accurately estimating the 3D shape of flexible continuum robots subjected to external loads. The proposed method introduces a spatiotemporal neural network architecture that fuses multi-modal inputs, including current and historical tendon displacement data and RGB images, to generate point clouds representing the robot's deformed configuration. The network integrates a recurrent neural module for temporal feature extraction, an encoding module for spatial feature extraction, and a multi-modal fusion module to combine spatial features extracted from visual data with temporal dependencies from historical actuator inputs. Continuous 3D shape reconstruction is achieved by fitting B\'ezier curves to the predicted point clouds. Experimental validation demonstrates that our approach achieves high precision, with mean shape estimation errors of 0.08 mm (unloaded) and 0.22 mm (loaded), outperforming state-of-the-art methods in shape sensing for TDCRs. The results validate the efficacy of deep learning-based spatiotemporal data fusion for precise shape estimation under loading conditions.
>
---
#### [new 022] A short methodological review on social robot navigation benchmarking
- **分类: cs.RO; I.2.9**

- **简介: 该论文聚焦社会机器人导航的基准测试问题，针对2020年1月至2025年7月间85篇相关研究，系统分析了评估指标、算法应用、人类问卷使用及结论推导方法，旨在推动该领域标准化发展。**

- **链接: [http://arxiv.org/pdf/2510.22448v1](http://arxiv.org/pdf/2510.22448v1)**

> **作者:** Pranup Chhetri; Alejandro Torrejon; Sergio Eslava; Luis J. Manso
>
> **备注:** 18 pages, 14 of which references. 3 figures, 2 tables
>
> **摘要:** Social Robot Navigation is the skill that allows robots to move efficiently in human-populated environments while ensuring safety, comfort, and trust. Unlike other areas of research, the scientific community has not yet achieved an agreement on how Social Robot Navigation should be benchmarked. This is notably important, as the lack of a de facto standard to benchmark Social Robot Navigation can hinder the progress of the field and may lead to contradicting conclusions. Motivated by this gap, we contribute with a short review focused exclusively on benchmarking trends in the period from January 2020 to July 2025. Of the 130 papers identified by our search using IEEE Xplore, we analysed the 85 papers that met the criteria of the review. This review addresses the metrics used in the literature for benchmarking purposes, the algorithms employed in such benchmarks, the use of human surveys for benchmarking, and how conclusions are drawn from the benchmarking results, when applicable.
>
---
#### [new 023] Combining High Level Scheduling and Low Level Control to Manage Fleets of Mobile Robots
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对工业环境中大规模移动机器人车队的协同调度问题，提出一种两层框架：上层用ComSat算法进行任务分配与时间参数化调度，下层通过分布式模型预测控制实现实时路径规划与避障。有效应对动态干扰，保障安全、高效运行，支持快速重调度。**

- **链接: [http://arxiv.org/pdf/2510.23129v1](http://arxiv.org/pdf/2510.23129v1)**

> **作者:** Sabino Francesco Roselli; Ze Zhang; Knut Åkesson
>
> **摘要:** The deployment of mobile robots for material handling in industrial environments requires scalable coordination of large fleets in dynamic settings. This paper presents a two-layer framework that combines high-level scheduling with low-level control. Tasks are assigned and scheduled using the compositional algorithm ComSat, which generates time-parameterized routes for each robot. These schedules are then used by a distributed Model Predictive Control (MPC) system in real time to compute local reference trajectories, accounting for static and dynamic obstacles. The approach ensures safe, collision-free operation, and supports rapid rescheduling in response to disruptions such as robot failures or environmental changes. We evaluate the method in simulated 2D environments with varying road capacities and traffic conditions, demonstrating high task completion rates and robust behavior even under congestion. The modular structure of the framework allows for computational tractability and flexibility, making it suitable for deployment in complex, real-world industrial scenarios.
>
---
#### [new 024] An Intelligent Water-Saving Irrigation System Based on Multi-Sensor Fusion and Visual Servoing Control
- **分类: cs.RO; cs.CV; cs.SY; eess.SY**

- **简介: 该论文针对精准农业中灌溉效率低、地形适应性差的问题，提出一种基于多传感器融合与视觉伺服控制的智能节水灌溉系统。通过轻量YOLO模型实现高精度植株检测，结合简化手眼标定与主动调平技术，提升机器人定位与平台稳定性，在多种环境下实现30-50%节水率，水利用效率超92%。**

- **链接: [http://arxiv.org/pdf/2510.23003v1](http://arxiv.org/pdf/2510.23003v1)**

> **作者:** ZhengKai Huang; YiKun Wang; ChenYu Hui; XiaoCheng
>
> **摘要:** This paper introduces an intelligent water-saving irrigation system designed to address critical challenges in precision agriculture, such as inefficient water use and poor terrain adaptability. The system integrates advanced computer vision, robotic control, and real-time stabilization technologies via a multi-sensor fusion approach. A lightweight YOLO model, deployed on an embedded vision processor (K210), enables real-time plant container detection with over 96% accuracy under varying lighting conditions. A simplified hand-eye calibration algorithm-designed for 'handheld camera' robot arm configurations-ensures that the end effector can be precisely positioned, with a success rate exceeding 90%. The active leveling system, driven by the STM32F103ZET6 main control chip and JY901S inertial measurement data, can stabilize the irrigation platform on slopes up to 10 degrees, with a response time of 1.8 seconds. Experimental results across three simulated agricultural environments (standard greenhouse, hilly terrain, complex lighting) demonstrate a 30-50% reduction in water consumption compared to conventional flood irrigation, with water use efficiency exceeding 92% in all test cases.
>
---
#### [new 025] A phase-aware AI car-following model for electric vehicles with adaptive cruise control: Development and validation using real-world data
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 该论文针对电动汽车（EV）独特驾驶动态，提出一种基于人工智能的分相感知车距跟随模型（PAAI），以解决现有模型无法准确描述EV加速与再生制动行为的问题。利用实测自适应巡航数据验证，显著提升预测精度，为交通仿真中EV行为建模提供有效工具。**

- **链接: [http://arxiv.org/pdf/2510.21735v1](http://arxiv.org/pdf/2510.21735v1)**

> **作者:** Yuhui Liu; Shian Wang; Ansel Panicker; Kate Embry; Ayana Asanova; Tianyi Li
>
> **摘要:** Internal combustion engine (ICE) vehicles and electric vehicles (EVs) exhibit distinct vehicle dynamics. EVs provide rapid acceleration, with electric motors producing peak power across a wider speed range, and achieve swift deceleration through regenerative braking. While existing microscopic models effectively capture the driving behavior of ICE vehicles, a modeling framework that accurately describes the unique car-following dynamics of EVs is lacking. Developing such a model is essential given the increasing presence of EVs in traffic, yet creating an easy-to-use and accurate analytical model remains challenging. To address these gaps, this study develops and validates a Phase-Aware AI (PAAI) car-following model specifically for EVs. The proposed model enhances traditional physics-based frameworks with an AI component that recognizes and adapts to different driving phases, such as rapid acceleration and regenerative braking. Using real-world trajectory data from vehicles equipped with adaptive cruise control (ACC), we conduct comprehensive simulations to validate the model's performance. The numerical results demonstrate that the PAAI model significantly improves prediction accuracy over traditional car-following models, providing an effective tool for accurately representing EV behavior in traffic simulations.
>
---
#### [new 026] Uncertainty-Aware Autonomous Vehicles: Predicting the Road Ahead
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对自动驾驶中感知系统对罕见场景过度自信的问题，提出基于随机集神经网络（RS-NN）的不确定性感知方法。通过量化预测不确定性，在真实赛车系统中实现道路布局识别与动态速度调节，显著提升安全性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.22680v1](http://arxiv.org/pdf/2510.22680v1)**

> **作者:** Shireen Kudukkil Manchingal; Armand Amaritei; Mihir Gohad; Maryam Sultana; Julian F. P. Kooij; Fabio Cuzzolin; Andrew Bradley
>
> **摘要:** Autonomous Vehicle (AV) perception systems have advanced rapidly in recent years, providing vehicles with the ability to accurately interpret their environment. Perception systems remain susceptible to errors caused by overly-confident predictions in the case of rare events or out-of-sample data. This study equips an autonomous vehicle with the ability to 'know when it is uncertain', using an uncertainty-aware image classifier as part of the AV software stack. Specifically, the study exploits the ability of Random-Set Neural Networks (RS-NNs) to explicitly quantify prediction uncertainty. Unlike traditional CNNs or Bayesian methods, RS-NNs predict belief functions over sets of classes, allowing the system to identify and signal uncertainty clearly in novel or ambiguous scenarios. The system is tested in a real-world autonomous racing vehicle software stack, with the RS-NN classifying the layout of the road ahead and providing the associated uncertainty of the prediction. Performance of the RS-NN under a range of road conditions is compared against traditional CNN and Bayesian neural networks, with the RS-NN achieving significantly higher accuracy and superior uncertainty calibration. This integration of RS-NNs into Robot Operating System (ROS)-based vehicle control pipeline demonstrates that predictive uncertainty can dynamically modulate vehicle speed, maintaining high-speed performance under confident predictions while proactively improving safety through speed reductions in uncertain scenarios. These results demonstrate the potential of uncertainty-aware neural networks - in particular RS-NNs - as a practical solution for safer and more robust autonomous driving.
>
---
#### [new 027] ACG: Action Coherence Guidance for Flow-based VLA models
- **分类: cs.RO**

- **简介: 该论文针对视觉-语言-动作（VLA）模型在模仿学习中因人类演示噪声导致的动作不连贯问题，提出无需训练的测试时引导方法ACG。通过提升动作一致性，显著改善了机器人在复杂操作任务中的稳定性与成功率，适用于多样化真实场景下的精细操控任务。**

- **链接: [http://arxiv.org/pdf/2510.22201v1](http://arxiv.org/pdf/2510.22201v1)**

> **作者:** Minho Park; Kinam Kim; Junha Hyung; Hyojin Jang; Hoiyeong Jin; Jooyeol Yun; Hojoon Lee; Jaegul Choo
>
> **摘要:** Diffusion and flow matching models have emerged as powerful robot policies, enabling Vision-Language-Action (VLA) models to generalize across diverse scenes and instructions. Yet, when trained via imitation learning, their high generative capacity makes them sensitive to noise in human demonstrations: jerks, pauses, and jitter which reduce action coherence. Reduced action coherence causes instability and trajectory drift during deployment, failures that are catastrophic in fine-grained manipulation where precision is crucial. In this paper, we present Action Coherence Guidance (ACG) for VLA models, a training-free test-time guidance algorithm that improves action coherence and thereby yields performance gains. Evaluated on RoboCasa, DexMimicGen, and real-world SO-101 tasks, ACG consistently improves action coherence and boosts success rates across diverse manipulation tasks. Code and project page are available at https://github.com/DAVIAN-Robotics/ACG and https://DAVIAN-Robotics.github.io/ACG , respectively.
>
---
#### [new 028] A Literature Review On Stewart-Gough Platform Calibrations A Literature Review On Stewart-Gough Platform Calibrations
- **分类: cs.RO**

- **简介: 该论文属于综述任务，旨在梳理斯特林-戈夫平台（Stewart-Gough平台）的标定技术。针对平台在高精度应用中因结构与运动学误差导致定位不准的问题，系统回顾了基于逆运动学的标定方法，涵盖外部测量、约束运动、自校准传感器等手段，总结了关键误差源及标定效果。**

- **链接: [http://arxiv.org/pdf/2510.21854v1](http://arxiv.org/pdf/2510.21854v1)**

> **作者:** Sourabh Karmakar; Cameron J. Turner
>
> **摘要:** Researchers have studied Stewart-Gough platforms, also known as Gough-Stewart platforms or hexapod platforms extensively for their inherent fine control characteristics. Their studies led to the potential deployment opportunities of Stewart-Gough Platforms in many critical applications such as the medical field, engineering machines, space research, electronic chip manufacturing, automobile manufacturing, etc. Some of these applications need micro and nano-level movement control in 3D space for the motions to be precise, complicated, and repeatable; a Stewart-Gough platform fulfills these challenges smartly. For this, the platform must be more accurate than the specified application accuracy level and thus proper calibration for a parallel robot is crucial. Forward kinematics-based calibration for these hexapod machines becomes unnecessarily complex and inverse kinematics complete this task with much ease. To experiment with different calibration techniques, various calibration approaches were implemented by using external instruments, constraining one or more motions of the system, and using extra sensors for auto or self-calibration. This survey paid attention to those key methodologies, their outcome, and important details related to inverse kinematic-based parallel robot calibrations. It was observed during this study that the researchers focused on improving the accuracy of the platform position and orientation considering the errors contributed by one source or multiple sources. The error sources considered are mainly kinematic and structural, in some cases, environmental factors also are reviewed, however, those calibrations are done under no-load conditions. This study aims to review the present state of the art in this field and highlight the processes and errors considered for the calibration of Stewart-Gough platforms.
>
---
#### [new 029] ManiDP: Manipulability-Aware Diffusion Policy for Posture-Dependent Bimanual Manipulation
- **分类: cs.RO**

- **简介: 该论文针对双臂操作中姿势依赖性任务需求忽视的问题，提出ManiDP方法。通过提取专家示范中的双臂可操作性特征，利用黎曼概率模型编码姿势先验，并融入条件扩散过程，生成更符合任务需求的双臂运动轨迹。实验表明，该方法显著提升操作成功率与任务兼容性。**

- **链接: [http://arxiv.org/pdf/2510.23016v1](http://arxiv.org/pdf/2510.23016v1)**

> **作者:** Zhuo Li; Junjia Liu; Dianxi Li; Tao Teng; Miao Li; Sylvain Calinon; Darwin Caldwell; Fei Chen
>
> **备注:** 7 pages, 6 figures, Accepted and published in IROS 2025
>
> **摘要:** Recent work has demonstrated the potential of diffusion models in robot bimanual skill learning. However, existing methods ignore the learning of posture-dependent task features, which are crucial for adapting dual-arm configurations to meet specific force and velocity requirements in dexterous bimanual manipulation. To address this limitation, we propose Manipulability-Aware Diffusion Policy (ManiDP), a novel imitation learning method that not only generates plausible bimanual trajectories, but also optimizes dual-arm configurations to better satisfy posture-dependent task requirements. ManiDP achieves this by extracting bimanual manipulability from expert demonstrations and encoding the encapsulated posture features using Riemannian-based probabilistic models. These encoded posture features are then incorporated into a conditional diffusion process to guide the generation of task-compatible bimanual motion sequences. We evaluate ManiDP on six real-world bimanual tasks, where the experimental results demonstrate a 39.33$\%$ increase in average manipulation success rate and a 0.45 improvement in task compatibility compared to baseline methods. This work highlights the importance of integrating posture-relevant robotic priors into bimanual skill diffusion to enable human-like adaptability and dexterity.
>
---
#### [new 030] Optimal Dimensioning of Elastic-Link Manipulators regarding Lifetime Estimation
- **分类: cs.RO**

- **简介: 该论文针对轻量化柔性机械臂的寿命与振动问题，提出一种结合疲劳分析与多目标优化的设计方法。通过雨流计数和临界平面法评估寿命，优化机械臂几何以平衡重量、振动与寿命，适用于三自由度机械臂的拾放任务。**

- **链接: [http://arxiv.org/pdf/2510.23234v1](http://arxiv.org/pdf/2510.23234v1)**

> **作者:** Klaus Zauner; Hubert Gattringer; Andreas Mueller
>
> **备注:** Mechanics Based Design of Structures and Machines, December 2024
>
> **摘要:** Resourceful operation and design of robots is key for sustainable industrial automation. This will be enabled by lightweight design along with time and energy optimal control of robotic manipulators. Design and control of such systems is intertwined as the control must take into account inherent mechanical compliance while the design must accommodate the dynamic requirements demanded by the control. As basis for such design optimization, a method for estimating the lifetime of elastic link robotic manipulators is presented. This is applied to the geometry optimization of flexible serial manipulators performing pick-and-place operations, where the optimization objective is a combination of overall weight and vibration amplitudes. The lifetime estimation draws from a fatigue analysis combining the rainflow counting algorithm and the method of critical cutting plane. Tresca hypothesis is used to formulate an equivalent stress, and linear damage accumulation is assumed. The final robot geometry is selected from a Pareto front as a tradeoff of lifetime and vibration characteristic. The method is illustrated for a three degrees of freedom articulated robotic manipulator.
>
---
#### [new 031] Policies over Poses: Reinforcement Learning based Distributed Pose-Graph Optimization for Multi-Robot SLAM
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文针对多机器人SLAM中的分布式位姿图优化问题，提出基于多智能体强化学习的可扩展、抗异常值框架。通过图分割与图神经网络实现局部优化，结合记忆机制与共识策略，显著提升精度与效率，且无需重训练即可扩展至更大团队。**

- **链接: [http://arxiv.org/pdf/2510.22740v1](http://arxiv.org/pdf/2510.22740v1)**

> **作者:** Sai Krishna Ghanta; Ramviyas Parasuraman
>
> **备注:** IEEE International Symposium on Multi-Robot & Multi-Agent Systems (MRS) 2025
>
> **摘要:** We consider the distributed pose-graph optimization (PGO) problem, which is fundamental in accurate trajectory estimation in multi-robot simultaneous localization and mapping (SLAM). Conventional iterative approaches linearize a highly non-convex optimization objective, requiring repeated solving of normal equations, which often converge to local minima and thus produce suboptimal estimates. We propose a scalable, outlier-robust distributed planar PGO framework using Multi-Agent Reinforcement Learning (MARL). We cast distributed PGO as a partially observable Markov game defined on local pose-graphs, where each action refines a single edge's pose estimate. A graph partitioner decomposes the global pose graph, and each robot runs a recurrent edge-conditioned Graph Neural Network (GNN) encoder with adaptive edge-gating to denoise noisy edges. Robots sequentially refine poses through a hybrid policy that utilizes prior action memory and graph embeddings. After local graph correction, a consensus scheme reconciles inter-robot disagreements to produce a globally consistent estimate. Our extensive evaluations on a comprehensive suite of synthetic and real-world datasets demonstrate that our learned MARL-based actors reduce the global objective by an average of 37.5% more than the state-of-the-art distributed PGO framework, while enhancing inference efficiency by at least 6X. We also demonstrate that actor replication allows a single learned policy to scale effortlessly to substantially larger robot teams without any retraining. Code is publicly available at https://github.com/herolab-uga/policies-over-poses.
>
---
#### [new 032] LT-Exosense: A Vision-centric Multi-session Mapping System for Lifelong Safe Navigation of Exoskeletons
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出LT-Exosense，一种面向外骨骼长期安全导航的视觉中心多会话建图系统。针对动态环境中持久感知与路径规划难题，通过增量融合多时段空间知识，实现环境变化检测与全局地图更新，支持自适应路径规划。实验证明其定位精度优于5cm，具备在复杂室内环境中长期稳定运行的能力。**

- **链接: [http://arxiv.org/pdf/2510.22164v1](http://arxiv.org/pdf/2510.22164v1)**

> **作者:** Jianeng Wang; Matias Mattamala; Christina Kassab; Nived Chebrolu; Guillaume Burger; Fabio Elnecave; Marine Petriaux; Maurice Fallon
>
> **备注:** 8 pages, 4 figures
>
> **摘要:** Self-balancing exoskeletons offer a promising mobility solution for individuals with lower-limb disabilities. For reliable long-term operation, these exoskeletons require a perception system that is effective in changing environments. In this work, we introduce LT-Exosense, a vision-centric, multi-session mapping system designed to support long-term (semi)-autonomous navigation for exoskeleton users. LT-Exosense extends single-session mapping capabilities by incrementally fusing spatial knowledge across multiple sessions, detecting environmental changes, and updating a persistent global map. This representation enables intelligent path planning, which can adapt to newly observed obstacles and can recover previous routes when obstructions are removed. We validate LT-Exosense through several real-world experiments, demonstrating a scalable multi-session map that achieves an average point-to-point error below 5 cm when compared to ground-truth laser scans. We also illustrate the potential application of adaptive path planning in dynamically changing indoor environments.
>
---
#### [new 033] Transferable Deep Reinforcement Learning for Cross-Domain Navigation: from Farmland to the Moon
- **分类: cs.RO**

- **简介: 该论文研究跨域自主导航任务，旨在解决机器人在不同环境间迁移时需重新训练的问题。通过在农田模拟环境中训练深度强化学习策略，并零样本迁移至月球模拟环境，验证了策略的泛化能力，实现了近50%的成功率，证明了跨域DRL迁移的有效性与低成本优势。**

- **链接: [http://arxiv.org/pdf/2510.23329v1](http://arxiv.org/pdf/2510.23329v1)**

> **作者:** Shreya Santra; Thomas Robbins; Kazuya Yoshida
>
> **备注:** 6 pages, 7 figures. Accepted at IEEE iSpaRo 2025
>
> **摘要:** Autonomous navigation in unstructured environments is essential for field and planetary robotics, where robots must efficiently reach goals while avoiding obstacles under uncertain conditions. Conventional algorithmic approaches often require extensive environment-specific tuning, limiting scalability to new domains. Deep Reinforcement Learning (DRL) provides a data-driven alternative, allowing robots to acquire navigation strategies through direct interactions with their environment. This work investigates the feasibility of DRL policy generalization across visually and topographically distinct simulated domains, where policies are trained in terrestrial settings and validated in a zero-shot manner in extraterrestrial environments. A 3D simulation of an agricultural rover is developed and trained using Proximal Policy Optimization (PPO) to achieve goal-directed navigation and obstacle avoidance in farmland settings. The learned policy is then evaluated in a lunar-like simulated environment to assess transfer performance. The results indicate that policies trained under terrestrial conditions retain a high level of effectiveness, achieving close to 50\% success in lunar simulations without the need for additional training and fine-tuning. This underscores the potential of cross-domain DRL-based policy transfer as a promising approach to developing adaptable and efficient autonomous navigation for future planetary exploration missions, with the added benefit of minimizing retraining costs.
>
---
#### [new 034] Improving the performance of AI-powered Affordable Robotics for Assistive Tasks
- **分类: cs.RO**

- **简介: 该论文针对高龄化社会中助老机器人成本高、难部署的问题，提出一种低成本智能机械臂系统。通过模仿学习从视频中获取动作数据，利用PACT模型与TE方法实现高效动作识别与轨迹优化，在喂食、清洁、取药等任务中达到90%以上准确率，显著提升性能并降低模型规模。**

- **链接: [http://arxiv.org/pdf/2510.21771v1](http://arxiv.org/pdf/2510.21771v1)**

> **作者:** Dharunish Yugeswardeenoo
>
> **备注:** 6 pages, 5 figures. Accepted to Conference on Robot Learning (CoRL 2025), Seoul, Korea
>
> **摘要:** By 2050, the global demand for assistive care is expected to reach 3.5 billion people, far outpacing the availability of human caregivers. Existing robotic solutions remain expensive and require technical expertise, limiting accessibility. This work introduces a low-cost robotic arm for assistive tasks such as feeding, cleaning spills, and fetching medicine. The system uses imitation learning from demonstration videos, requiring no task-specific programming or manual labeling. The robot consists of six servo motors, dual cameras, and 3D-printed grippers. Data collection via teleoperation with a leader arm yielded 50,000 video frames across the three tasks. A novel Phased Action Chunking Transformer (PACT) captures temporal dependencies and segments motion dynamics, while a Temporal Ensemble (TE) method refines trajectories to improve accuracy and smoothness. Evaluated across five model sizes and four architectures, with ten hours of real-world testing, the system achieved over 90% task accuracy, up to 40% higher than baselines. PACT enabled a 5x model size reduction while maintaining 75% accuracy. Saliency analysis showed reliance on key visual cues, and phase token gradients peaked at critical trajectory moments, indicating effective temporal reasoning. Future work will explore bimanual manipulation and mobility for expanded assistive capabilities.
>
---
#### [new 035] Breaking the Static Assumption: A Dynamic-Aware LIO Framework Via Spatio-Temporal Normal Analysis
- **分类: cs.RO**

- **简介: 该论文针对动态环境中激光惯性里程计（LIO）因依赖静态假设而失效的问题，提出一种基于时空法向分析的动态感知LIO框架。通过将动态感知融入点云配准，实现静态特征可靠识别与精准位姿估计的解耦，显著提升复杂动态场景下的定位性能。**

- **链接: [http://arxiv.org/pdf/2510.22313v1](http://arxiv.org/pdf/2510.22313v1)**

> **作者:** Chen Zhiqiang; Le Gentil Cedric; Lin Fuling; Lu Minghao; Qiao Qiyuan; Xu Bowen; Qi Yuhua; Lu Peng
>
> **备注:** 8 pages, 7 figures, Accepted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** This paper addresses the challenge of Lidar-Inertial Odometry (LIO) in dynamic environments, where conventional methods often fail due to their static-world assumptions. Traditional LIO algorithms perform poorly when dynamic objects dominate the scenes, particularly in geometrically sparse environments. Current approaches to dynamic LIO face a fundamental challenge: accurate localization requires a reliable identification of static features, yet distinguishing dynamic objects necessitates precise pose estimation. Our solution breaks this circular dependency by integrating dynamic awareness directly into the point cloud registration process. We introduce a novel dynamic-aware iterative closest point algorithm that leverages spatio-temporal normal analysis, complemented by an efficient spatial consistency verification method to enhance static map construction. Experimental evaluations demonstrate significant performance improvements over state-of-the-art LIO systems in challenging dynamic environments with limited geometric structure. The code and dataset are available at https://github.com/thisparticle/btsa.
>
---
#### [new 036] Real-Time QP Solvers: A Concise Review and Practical Guide Towards Legged Robots
- **分类: cs.RO**

- **简介: 该论文聚焦于腿式机器人实时运动控制中的二次规划（QP）求解问题，系统综述并对比了四类主流QP求解器。针对嵌入式平台的计算与能耗约束，通过基准测试评估其速度、精度与鲁棒性，提出基于任务与硬件特性的选型指导，助力高效、自主的腿式机器人系统实现。**

- **链接: [http://arxiv.org/pdf/2510.21773v1](http://arxiv.org/pdf/2510.21773v1)**

> **作者:** Van Nam Dinh
>
> **备注:** 6 pages, 1 figure, 2 tables
>
> **摘要:** Quadratic programming (QP) underpins real-time robotics by enabling efficient, constrained optimization in state estimation, motion planning, and control. In legged locomotion and manipulation, essential modules like inverse dynamics, Model Predictive Control (MPC), and Whole-Body Control (WBC) are inherently QP-based, demanding reliable solutions amid tight timing, energy, and computational limits on embedded platforms. This paper presents a comprehensive analysis and benchmarking study of cutting-edge QP solvers for legged robotics. We begin by formulating the standard convex QP and classify solvers into four principal algorithmic approaches, including interior-point methods, active-set strategies, operator splitting schemes, and augmented Lagrangian approaches. Each solver is examined in terms of algorithmic structure, computational characteristics, and its ability to exploit problem structure and warm-starting. Performance is evaluated using publicly available benchmarks, focusing on metrics such as computation time, constraint satisfaction, and robustness under perturbations. Feature tables and comparisons yield practical guidance for solver selection, underscoring trade-offs in speed, accuracy, and energy efficiency. Our findings emphasize the synergy between solver, task, and hardware, sparse IPMs for long-horizon MPC, and dense active-set for high frequency WBC to advance agile, autonomous legged systems, with emerging extensions to nonconvex and distributed QP.
>
---
#### [new 037] RaycastGrasp: Eye-Gaze Interaction with Wearable Devices for Robotic Manipulation
- **分类: cs.RO; cs.HC**

- **简介: 该论文提出RaycastGrasp，一种基于可穿戴混合现实头显的眼动交互系统，用于辅助机器人抓取。针对传统操纵杆控制精度要求高、参考系不直观的问题，利用第一人称视角下的眼动聚焦与增强视觉提示，结合预训练视觉模型实现意图与物体的单次识别，显著提升操作准确率与响应速度，有效增强了辅助机器人系统的易用性与可及性。**

- **链接: [http://arxiv.org/pdf/2510.22113v1](http://arxiv.org/pdf/2510.22113v1)**

> **作者:** Zitiantao Lin; Yongpeng Sang; Yang Ye
>
> **备注:** 5 pages, 5 figures; Accepted to: 2025 IEEE 4th International Conference on Intelligent Reality (ICIR 2025); Zitiantao Lin and Yongpeng Sang contributed equally to this work (co-first authors). Corresponding author: Yang Ye (y.ye@northeastern.edu)
>
> **摘要:** Robotic manipulators are increasingly used to assist individuals with mobility impairments in object retrieval. However, the predominant joystick-based control interfaces can be challenging due to high precision requirements and unintuitive reference frames. Recent advances in human-robot interaction have explored alternative modalities, yet many solutions still rely on external screens or restrictive control schemes, limiting their intuitiveness and accessibility. To address these challenges, we present an egocentric, gaze-guided robotic manipulation interface that leverages a wearable Mixed Reality (MR) headset. Our system enables users to interact seamlessly with real-world objects using natural gaze fixation from a first-person perspective, while providing augmented visual cues to confirm intent and leveraging a pretrained vision model and robotic arm for intent recognition and object manipulation. Experimental results demonstrate that our approach significantly improves manipulation accuracy, reduces system latency, and achieves single-pass intention and object recognition accuracy greater than 88% across multiple real-world scenarios. These results demonstrate the system's effectiveness in enhancing intuitiveness and accessibility, underscoring its practical significance for assistive robotics applications.
>
---
#### [new 038] OmniDexGrasp: Generalizable Dexterous Grasping via Foundation Model and Force Feedback
- **分类: cs.RO**

- **简介: 该论文提出OmniDexGrasp框架，解决机器人在多样对象与任务下难以泛化抓取的问题。通过融合基础模型、人体图像到机器人动作的迁移策略及力反馈自适应抓取，实现基于自然语言指令的通用灵巧抓取与操作，在仿真与真实机器人上验证了其有效性与扩展性。**

- **链接: [http://arxiv.org/pdf/2510.23119v1](http://arxiv.org/pdf/2510.23119v1)**

> **作者:** Yi-Lin Wei; Zhexi Luo; Yuhao Lin; Mu Lin; Zhizhao Liang; Shuoyu Chen; Wei-Shi Zheng
>
> **备注:** Project page: https://isee-laboratory.github.io/OmniDexGrasp/
>
> **摘要:** Enabling robots to dexterously grasp and manipulate objects based on human commands is a promising direction in robotics. However, existing approaches are challenging to generalize across diverse objects or tasks due to the limited scale of semantic dexterous grasp datasets. Foundation models offer a new way to enhance generalization, yet directly leveraging them to generate feasible robotic actions remains challenging due to the gap between abstract model knowledge and physical robot execution. To address these challenges, we propose OmniDexGrasp, a generalizable framework that achieves omni-capabilities in user prompting, dexterous embodiment, and grasping tasks by combining foundation models with the transfer and control strategies. OmniDexGrasp integrates three key modules: (i) foundation models are used to enhance generalization by generating human grasp images supporting omni-capability of user prompt and task; (ii) a human-image-to-robot-action transfer strategy converts human demonstrations into executable robot actions, enabling omni dexterous embodiment; (iii) force-aware adaptive grasp strategy ensures robust and stable grasp execution. Experiments in simulation and on real robots validate the effectiveness of OmniDexGrasp on diverse user prompts, grasp task and dexterous hands, and further results show its extensibility to dexterous manipulation tasks.
>
---
#### [new 039] BLIP-FusePPO: A Vision-Language Deep Reinforcement Learning Framework for Lane Keeping in Autonomous Vehicles
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.SE**

- **简介: 该论文提出BLIP-FusePPO框架，用于自动驾驶车道保持任务。通过将视觉-语言模型的语义嵌入直接融合到状态表示中，结合几何与控制信号，提升策略学习的鲁棒性与可解释性。相比仅用语义模型奖励的方法，本方案减少推理开销，增强实时性与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.22370v1](http://arxiv.org/pdf/2510.22370v1)**

> **作者:** Seyed Ahmad Hosseini Miangoleh; Amin Jalal Aghdasian; Farzaneh Abdollahi
>
> **备注:** https://github.com/Amin-A96/BLIP-FusePPO-A-Vision-Language-Deep-Reinforcement-Learning-Framework-for-Lane-Keeping-in-Autonomous.git
>
> **摘要:** In this paper, we propose Bootstrapped Language-Image Pretraining-driven Fused State Representation in Proximal Policy Optimization (BLIP-FusePPO), a novel multimodal reinforcement learning (RL) framework for autonomous lane-keeping (LK), in which semantic embeddings generated by a vision-language model (VLM) are directly fused with geometric states, LiDAR observations, and Proportional-Integral-Derivative-based (PID) control feedback within the agent observation space. The proposed method lets the agent learn driving rules that are aware of their surroundings and easy to understand by combining high-level scene understanding from the VLM with low-level control and spatial signals. Our architecture brings together semantic, geometric, and control-aware representations to make policy learning more robust. A hybrid reward function that includes semantic alignment, LK accuracy, obstacle avoidance, and speed regulation helps learning to be more efficient and generalizable. Our method is different from the approaches that only use semantic models to shape rewards. Instead, it directly embeds semantic features into the state representation. This cuts down on expensive runtime inference and makes sure that semantic guidance is always available. The simulation results show that the proposed model is better at LK stability and adaptability than the best vision-based and multimodal RL baselines in a wide range of difficult driving situations. We make our code publicly available.
>
---
#### [new 040] Butter-Bench: Evaluating LLM Controlled Robots for Practical Intelligence
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Butter-Bench基准，评估大语言模型（LLM）在机器人任务中的实用智能，聚焦于处理物理世界复杂性的能力。针对当前LLM在多步空间规划与社会理解上的短板，研究发现人类表现远超LLM（95% vs 40%），且微调训练未能提升性能，揭示了现有LLM在具身推理中的局限性。**

- **链接: [http://arxiv.org/pdf/2510.21860v1](http://arxiv.org/pdf/2510.21860v1)**

> **作者:** Callum Sharrock; Lukas Petersson; Hanna Petersson; Axel Backlund; Axel Wennström; Kristoffer Nordström; Elias Aronsson
>
> **摘要:** We present Butter-Bench, a benchmark evaluating large language model (LLM) controlled robots for practical intelligence, defined as the ability to navigate the messiness of the physical world. Current state-of-the-art robotic systems use a hierarchical architecture with LLMs in charge of high-level reasoning, and a Vision Language Action (VLA) model for low-level control. Butter-Bench evaluates the LLM part in isolation from the VLA. Although LLMs have repeatedly surpassed humans in evaluations requiring analytical intelligence, we find humans still outperform LLMs on Butter-Bench. The best LLMs score 40% on Butter-Bench, while the mean human score is 95%. LLMs struggled the most with multi-step spatial planning and social understanding. We also evaluate LLMs that are fine-tuned for embodied reasoning and conclude that this training does not improve their score on Butter-Bench.
>
---
#### [new 041] FORGE-Tree: Diffusion-Forcing Tree Search for Long-Horizon Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文针对长时序机器人操作任务中因漂移和暴露偏差导致的轨迹误差累积问题，提出FORGE-Tree方法。通过阶段对齐扩散强迫与测试时蒙特卡洛树扩散，实现局部轨迹修正与智能搜索，结合场景图提升推理效率。在LIBERO数据集上显著提升成功率，且在相同计算预算下表现更优。**

- **链接: [http://arxiv.org/pdf/2510.21744v1](http://arxiv.org/pdf/2510.21744v1)**

> **作者:** Yanjia Huang; Shuo Liu; Sheng Liu; Qingxiao Xu; Mingyang Wu; Xiangbo Gao; Zhengzhong Tu
>
> **摘要:** Long-horizon robot manipulation tasks remain challenging for Vision-Language-Action (VLA) policies due to drift and exposure bias, often denoise the entire trajectory with fixed hyperparameters, causing small geometric errors to compound across stages and offering no mechanism to allocate extra test-time compute where clearances are tight. To address these challenges, we introduce FORGE-Tree, a plug-in control layer that couples a stage-aligned Diffusion Forcing (DF) head with test-time Monte Carlo Tree Diffusion (MCTD). With a frozen VLA encoder, DF aligns timesteps to subtask stages; during inference we partially denoise only a target segment while keeping other tokens frozen, turning trajectory refinement into a sequence of local edits. We then apply Monte Carlo Tree Diffusion to select the next segment to refine. A scene graph supplies priors for expansion and geometry relation-aware scoring for rollouts, yielding tree-structured denoising whose performance scales with search budget while preserving the executed prefix. Evaluation on LIBERO, FORGE-Tree improves success rate by 13.4 to 17.2 pp over the native VLA baselines with both OpenVLA and Octo-Base. Gains remain consistent under comparable compute budgets, especially on long-horizon variants. Videos available at: https://taco-group.github.io/FORGE-Tree/
>
---
#### [new 042] PIP-LLM: Integrating PDDL-Integer Programming with LLMs for Coordinating Multi-Robot Teams Using Natural Language
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出PIP-LLM框架，解决多机器人团队执行自然语言指令时的协调难题。通过分层规划：先用PDDL进行团队级任务分解，再用整数规划优化机器人级任务分配，提升计划成功率、降低路径成本并实现负载均衡。**

- **链接: [http://arxiv.org/pdf/2510.22784v1](http://arxiv.org/pdf/2510.22784v1)**

> **作者:** Guangyao Shi; Yuwei Wu; Vijay Kumar; Gaurav S. Sukhatme
>
> **摘要:** Enabling robot teams to execute natural language commands requires translating high-level instructions into feasible, efficient multi-robot plans. While Large Language Models (LLMs) combined with Planning Domain Description Language (PDDL) offer promise for single-robot scenarios, existing approaches struggle with multi-robot coordination due to brittle task decomposition, poor scalability, and low coordination efficiency. We introduce PIP-LLM, a language-based coordination framework that consists of PDDL-based team-level planning and Integer Programming (IP) based robot-level planning. PIP-LLMs first decomposes the command by translating the command into a team-level PDDL problem and solves it to obtain a team-level plan, abstracting away robot assignment. Each team-level action represents a subtask to be finished by the team. Next, this plan is translated into a dependency graph representing the subtasks' dependency structure. Such a dependency graph is then used to guide the robot-level planning, in which each subtask node will be formulated as an IP-based task allocation problem, explicitly optimizing travel costs and workload while respecting robot capabilities and user-defined constraints. This separation of planning from assignment allows PIP-LLM to avoid the pitfalls of syntax-based decomposition and scale to larger teams. Experiments across diverse tasks show that PIP-LLM improves plan success rate, reduces maximum and average travel costs, and achieves better load balancing compared to state-of-the-art baselines.
>
---
#### [new 043] UrbanVLA: A Vision-Language-Action Model for Urban Micromobility
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文针对城市微移动机器人在复杂城市环境中的长距离导航难题，提出UrbanVLA框架。通过视觉-语言-动作联合建模，实现路线与视觉的动态对齐，并结合两阶段训练提升模型在真实场景下的安全性和适应性，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.23576v1](http://arxiv.org/pdf/2510.23576v1)**

> **作者:** Anqi Li; Zhiyong Wang; Jiazhao Zhang; Minghan Li; Yunpeng Qi; Zhibo Chen; Zhizheng Zhang; He Wang
>
> **摘要:** Urban micromobility applications, such as delivery robots, demand reliable navigation across large-scale urban environments while following long-horizon route instructions. This task is particularly challenging due to the dynamic and unstructured nature of real-world city areas, yet most existing navigation methods remain tailored to short-scale and controllable scenarios. Effective urban micromobility requires two complementary levels of navigation skills: low-level capabilities such as point-goal reaching and obstacle avoidance, and high-level capabilities, such as route-visual alignment. To this end, we propose UrbanVLA, a route-conditioned Vision-Language-Action (VLA) framework designed for scalable urban navigation. Our method explicitly aligns noisy route waypoints with visual observations during execution, and subsequently plans trajectories to drive the robot. To enable UrbanVLA to master both levels of navigation, we employ a two-stage training pipeline. The process begins with Supervised Fine-Tuning (SFT) using simulated environments and trajectories parsed from web videos. This is followed by Reinforcement Fine-Tuning (RFT) on a mixture of simulation and real-world data, which enhances the model's safety and adaptability in real-world settings. Experiments demonstrate that UrbanVLA surpasses strong baselines by more than 55% in the SocialNav task on MetaUrban. Furthermore, UrbanVLA achieves reliable real-world navigation, showcasing both scalability to large-scale urban environments and robustness against real-world uncertainties.
>
---
#### [new 044] Workspace Registration and Collision Detection for Industrial Robotics Applications
- **分类: cs.RO**

- **简介: 该论文属于工业机器人运动规划任务，旨在解决环境建模与碰撞检测问题。通过对比不同传感器获取点云数据，采用区域生长与VCCS算法识别碰撞物体，并对点云进行拟合，构建完整碰撞环境，实现机器人与环境的精确碰撞检测。**

- **链接: [http://arxiv.org/pdf/2510.23227v1](http://arxiv.org/pdf/2510.23227v1)**

> **作者:** Klaus Zauner; Josef El Dib; Hubert Gattringer; Andreas Mueller
>
> **摘要:** Motion planning for robotic manipulators relies on precise knowledge of the environment in order to be able to define restricted areas and to take collision objects into account. To capture the workspace, point clouds of the environment are acquired using various sensors. The collision objects are identified by region growing segmentation and VCCS algorithm. Subsequently the point clusters are approximated. The aim of the present paper is to compare different sensors, to illustrate the process from detection to the finished collision environment and to detect collisions between the robot and this environment.
>
---
#### [new 045] SCAL for Pinch-Lifting: Complementary Rotational and Linear Prototypes for Environment-Adaptive Grasping
- **分类: cs.RO**

- **简介: 该论文针对环境自适应夹持任务，提出一种槽约束可变形连杆（SCAL）结构，设计出旋转驱动（SCAL-R）与直线驱动（SCAL-L）两种互补手指。二者通过表面跟随实现向上抓举，保持指尖朝向，无需复杂传感与控制，可稳定抓取薄型或低轮廓物体。实验验证了其在多种场景下的鲁棒性，为简单驱动下的自适应抓取提供了有效方案。**

- **链接: [http://arxiv.org/pdf/2510.22738v1](http://arxiv.org/pdf/2510.22738v1)**

> **作者:** Wentao Guo; Wenzeng Zhang
>
> **备注:** Preliminary version presented at the IROS 2025 CIM Workshop, where it was selected as a Best Demo Award (Finalist) and subsequently received the Best Demo Award after oral presentation
>
> **摘要:** This paper presents environment-adaptive pinch-lifting built on a slot-constrained adaptive linkage (SCAL) and instantiated in two complementary fingers: SCAL-R, a rotational-drive design with an active fingertip that folds inward after contact to form an envelope, and SCAL-L, a linear-drive design that passively opens on contact to span wide or weak-feature objects. Both fingers convert surface following into an upward lifting branch while maintaining fingertip orientation, enabling thin or low-profile targets to be raised from supports with minimal sensing and control. Two-finger grippers are fabricated via PLA-based 3D printing. Experiments evaluate (i) contact-preserving sliding and pinch-lifting on tabletops, (ii) ramp negotiation followed by lift, and (iii) handling of bulky objects via active enveloping (SCAL-R) or contact-triggered passive opening (SCAL-L). Across dozens of trials on small parts, boxes, jars, and tape rolls, both designs achieve consistent grasps with limited tuning. A quasi-static analysis provides closed-form fingertip-force models for linear parallel pinching and two-point enveloping, offering geometry-aware guidance for design and operation. Overall, the results indicate complementary operating regimes and a practical path to robust, environment-adaptive grasping with simple actuation.
>
---
#### [new 046] VITA-E: Natural Embodied Interaction with Concurrent Seeing, Hearing, Speaking, and Acting
- **分类: cs.RO; cs.CL; cs.LG**

- **简介: 该论文提出VITA-E框架，解决现有视觉-语言-动作模型无法并发处理感知、听觉、语言与行动的问题。通过双模型架构与“模型即控制器”机制，实现人机交互中多任务并发与实时中断响应，提升机器人自然协同能力。**

- **链接: [http://arxiv.org/pdf/2510.21817v1](http://arxiv.org/pdf/2510.21817v1)**

> **作者:** Xiaoyu Liu; Chaoyou Fu; Chi Yan; Chu Wu; Haihan Gao; Yi-Fan Zhang; Shaoqi Dong; Cheng Qian; Bin Luo; Xiuyong Yang; Guanwu Li; Yusheng Cai; Yunhang Shen; Deqiang Jiang; Haoyu Cao; Xing Sun; Caifeng Shan; Ran He
>
> **备注:** Homepage: https://lxysl.github.io/VITA-E/
>
> **摘要:** Current Vision-Language-Action (VLA) models are often constrained by a rigid, static interaction paradigm, which lacks the ability to see, hear, speak, and act concurrently as well as handle real-time user interruptions dynamically. This hinders seamless embodied collaboration, resulting in an inflexible and unresponsive user experience. To address these limitations, we introduce VITA-E, a novel embodied interaction framework designed for both behavioral concurrency and nearly real-time interruption. The core of our approach is a dual-model architecture where two parallel VLA instances operate as an ``Active Model'' and a ``Standby Model'', allowing the embodied agent to observe its environment, listen to user speech, provide verbal responses, and execute actions, all concurrently and interruptibly, mimicking human-like multitasking capabilities. We further propose a ``model-as-controller'' paradigm, where we fine-tune the VLM to generate special tokens that serve as direct system-level commands, coupling the model's reasoning with the system's behavior. Experiments conducted on a physical humanoid platform demonstrate that VITA-E can reliably handle complex interactive scenarios. Our framework is compatible with various dual-system VLA models, achieving an extremely high success rate on emergency stops and speech interruptions while also successfully performing concurrent speech and action. This represents a significant step towards more natural and capable embodied assistants.
>
---
#### [new 047] End-to-End Design and Validation of a Low-Cost Stewart Platform with Nonlinear Estimation and Control
- **分类: cs.RO; cs.SY; eess.SY; 93C10; I.2.9; I.2.8; J.2**

- **简介: 该论文针对低成本六自由度Stewart平台的完整设计与控制问题，融合硬件集成、非线性建模与实时控制。通过反馈线性化与LQR结合实现精准轨迹跟踪，利用扩展卡尔曼滤波融合传感器数据提升状态估计精度，完成了软硬件一体化平台的仿真与实验验证。**

- **链接: [http://arxiv.org/pdf/2510.22949v1](http://arxiv.org/pdf/2510.22949v1)**

> **作者:** Benedictus C. G. Cinun; Tua A. Tamba; Immanuel R. Santjoko; Xiaofeng Wang; Michael A. Gunarso; Bin Hu
>
> **备注:** 24 pages, journal
>
> **摘要:** This paper presents the complete design, control, and experimental validation of a low-cost Stewart platform prototype developed as an affordable yet capable robotic testbed for research and education. The platform combines off the shelf components with 3D printed and custom fabricated parts to deliver full six degrees of freedom motions using six linear actuators connecting a moving platform to a fixed base. The system software integrates dynamic modeling, data acquisition, and real time control within a unified framework. A robust trajectory tracking controller based on feedback linearization, augmented with an LQR scheme, compensates for the platform's nonlinear dynamics to achieve precise motion control. In parallel, an Extended Kalman Filter fuses IMU and actuator encoder feedback to provide accurate and reliable state estimation under sensor noise and external disturbances. Unlike prior efforts that emphasize only isolated aspects such as modeling or control, this work delivers a complete hardware-software platform validated through both simulation and experiments on static and dynamic trajectories. Results demonstrate effective trajectory tracking and real-time state estimation, highlighting the platform's potential as a cost effective and versatile tool for advanced research and educational applications.
>
---
#### [new 048] EasyUUV: An LLM-Enhanced Universal and Lightweight Sim-to-Real Reinforcement Learning Framework for UUV Attitude Control
- **分类: cs.RO**

- **简介: 该论文提出EasyUUV，一个基于大语言模型的轻量级仿真到现实强化学习框架，用于水下无人艇姿态控制。针对泛化性差、抗干扰弱和部署效率低的问题，融合并行强化学习与自适应S-表面控制器，并利用多模态LLM实时调整参数，实现无需重训练的动态适应。**

- **链接: [http://arxiv.org/pdf/2510.22126v1](http://arxiv.org/pdf/2510.22126v1)**

> **作者:** Guanwen Xie; Jingzehua Xu; Jiwei Tang; Yubo Huang; Shuai Zhang; Xiaofan Li
>
> **备注:** 8 pages, 15 figures
>
> **摘要:** Despite recent advances in Unmanned Underwater Vehicle (UUV) attitude control, existing methods still struggle with generalizability, robustness to real-world disturbances, and efficient deployment. To address the above challenges, this paper presents EasyUUV, a Large Language Model (LLM)-enhanced, universal, and lightweight simulation-to-reality reinforcement learning (RL) framework for robust attitude control of UUVs. EasyUUV combines parallelized RL training with a hybrid control architecture, where a learned policy outputs high-level attitude corrections executed by an adaptive S-Surface controller. A multimodal LLM is further integrated to adaptively tune controller parameters at runtime using visual and textual feedback, enabling training-free adaptation to unmodeled dynamics. Also, we have developed a low-cost 6-DoF UUV platform and applied an RL policy trained through efficient parallelized simulation. Extensive simulation and real-world experiments validate the effectiveness and outstanding performance of EasyUUV in achieving robust and adaptive UUV attitude control across diverse underwater conditions. The source code is available from the following website: https://360zmem.github.io/easyuuv/
>
---
#### [new 049] TARC: Time-Adaptive Robotic Control
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出TARC方法，解决机器人控制中固定频率导致的效率与鲁棒性权衡问题。通过强化学习让策略自主选择动作及持续时间，实现动态调整控制频率。在真实机器人上验证，显著降低控制频率并适应实际场景。**

- **链接: [http://arxiv.org/pdf/2510.23176v1](http://arxiv.org/pdf/2510.23176v1)**

> **作者:** Arnav Sukhija; Lenart Treven; Jin Cheng; Florian Dörfler; Stelian Coros; Andreas Krause
>
> **摘要:** Fixed-frequency control in robotics imposes a trade-off between the efficiency of low-frequency control and the robustness of high-frequency control, a limitation not seen in adaptable biological systems. We address this with a reinforcement learning approach in which policies jointly select control actions and their application durations, enabling robots to autonomously modulate their control frequency in response to situational demands. We validate our method with zero-shot sim-to-real experiments on two distinct hardware platforms: a high-speed RC car and a quadrupedal robot. Our method matches or outperforms fixed-frequency baselines in terms of rewards while significantly reducing the control frequency and exhibiting adaptive frequency control under real-world conditions.
>
---
#### [new 050] J-ORA: A Framework and Multimodal Dataset for Japanese Object Identification, Reference, Action Prediction in Robot Perception
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出J-ORA框架与多模态数据集，聚焦日语人机对话中的机器人感知任务，解决物体识别、指代消解与动作预测问题。通过丰富物体属性标注，提升视觉语言模型性能，揭示专有与开源模型间的差距，并强调上下文敏感属性对动态环境感知的重要性。**

- **链接: [http://arxiv.org/pdf/2510.21761v1](http://arxiv.org/pdf/2510.21761v1)**

> **作者:** Jesse Atuhurra; Hidetaka Kamigaito; Taro Watanabe; Koichiro Yoshino
>
> **备注:** Accepted to IROS2025
>
> **摘要:** We introduce J-ORA, a novel multimodal dataset that bridges the gap in robot perception by providing detailed object attribute annotations within Japanese human-robot dialogue scenarios. J-ORA is designed to support three critical perception tasks, object identification, reference resolution, and next-action prediction, by leveraging a comprehensive template of attributes (e.g., category, color, shape, size, material, and spatial relations). Extensive evaluations with both proprietary and open-source Vision Language Models (VLMs) reveal that incorporating detailed object attributes substantially improves multimodal perception performance compared to without object attributes. Despite the improvement, we find that there still exists a gap between proprietary and open-source VLMs. In addition, our analysis of object affordances demonstrates varying abilities in understanding object functionality and contextual relationships across different VLMs. These findings underscore the importance of rich, context-sensitive attribute annotations in advancing robot perception in dynamic environments. See project page at https://jatuhurrra.github.io/J-ORA/.
>
---
#### [new 051] If They Disagree, Will You Conform? Exploring the Role of Robots' Value Awareness in a Decision-Making Task
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究社交机器人价值意识对人类决策的影响，旨在探索机器人是否因表现出价值理解而更影响人类判断。通过对比价值感知与非感知机器人在图像标注任务中的表现，发现参与者更关注价值感知机器人，且其分歧引发决策犹豫，揭示了机器人潜在的引导与反思作用。**

- **链接: [http://arxiv.org/pdf/2510.23204v1](http://arxiv.org/pdf/2510.23204v1)**

> **作者:** Giulia Pusceddu; Giulio Antonio Abbo; Francesco Rea; Tony Belpaeme; Alessandra Sciutti
>
> **摘要:** This study investigates whether the opinions of robotic agents are more likely to influence human decision-making when the robots are perceived as value-aware (i.e., when they display an understanding of human principles). We designed an experiment in which participants interacted with two Furhat robots - one programmed to be Value-Aware and the other Non-Value-Aware - during a labeling task for images representing human values. Results indicate that participants distinguished the Value-Aware robot from the Non-Value-Aware one. Although their explicit choices did not indicate a clear preference for one robot over the other, participants directed their gaze more toward the Value-Aware robot. Additionally, the Value-Aware robot was perceived as more loyal, suggesting that value awareness in a social robot may enhance its perceived commitment to the group. Finally, when both robots disagreed with the participant, conformity occurred in about one out of four trials, and participants took longer to confirm their responses, suggesting that two robots expressing dissent may introduce hesitation in decision-making. On one hand, this highlights the potential risk that robots, if misused, could manipulate users for unethical purposes. On the other hand, it reinforces the idea that social robots might encourage reflection in ambiguous situations and help users avoid scams.
>
---
#### [new 052] Ant-inspired Walling Strategies for Scalable Swarm Separation: Reinforcement Learning Approaches Based on Finite State Machines
- **分类: cs.RO**

- **简介: 该论文研究群体机器人协同任务中的空间分离问题，旨在实现异构集群在执行并行任务时的自主避让。受蚂蚁筑墙行为启发，提出基于有限状态机的分布式控制策略，并结合深度强化学习优化分离效果，显著降低群体混杂，提升适应性与收敛速度。**

- **链接: [http://arxiv.org/pdf/2510.22524v1](http://arxiv.org/pdf/2510.22524v1)**

> **作者:** Shenbagaraj Kannapiran; Elena Oikonomou; Albert Chu; Spring Berman; Theodore P. Pavlic
>
> **摘要:** In natural systems, emergent structures often arise to balance competing demands. Army ants, for example, form temporary "walls" that prevent interference between foraging trails. Inspired by this behavior, we developed two decentralized controllers for heterogeneous robotic swarms to maintain spatial separation while executing concurrent tasks. The first is a finite-state machine (FSM)-based controller that uses encounter-triggered transitions to create rigid, stable walls. The second integrates FSM states with a Deep Q-Network (DQN), dynamically optimizing separation through emergent "demilitarized zones." In simulation, both controllers reduce mixing between subgroups, with the DQN-enhanced controller improving adaptability and reducing mixing by 40-50% while achieving faster convergence.
>
---
#### [new 053] Breaking the Circle: An Autonomous Control-Switching Strategy for Stable Orographic Soaring in MAVs
- **分类: cs.RO**

- **简介: 该论文针对微小型无人机（MAVs）在地形升力中持续滑翔时因纵向与垂直轴控制冲突导致的盘旋问题，提出一种自主控制切换策略SAOS。通过选择性控制水平或垂直轴，将系统从欠驱动转为全驱动，结合攻角优化力估计，显著提升位置收敛性、降低能耗与滚转振荡，增强飞行稳定性和能效。**

- **链接: [http://arxiv.org/pdf/2510.23084v1](http://arxiv.org/pdf/2510.23084v1)**

> **作者:** Sunyou Hwang; Christophe De Wagter; Bart Remes; Guido de Croon
>
> **备注:** 13 pages, 15 figures
>
> **摘要:** Orographic soaring can significantly extend the endurance of micro aerial vehicles (MAVs), but circling behavior, arising from control conflicts between the longitudinal and vertical axes, increases energy consumption and the risk of divergence. We propose a control switching method, named SAOS: Switched Control for Autonomous Orographic Soaring, which mitigates circling behavior by selectively controlling either the horizontal or vertical axis, effectively transforming the system from underactuated to fully actuated during soaring. Additionally, the angle of attack is incorporated into the INDI controller to improve force estimation. Simulations with randomized initial positions and wind tunnel experiments on two MAVs demonstrate that the SAOS improves position convergence, reduces throttle usage, and mitigates roll oscillations caused by pitch-roll coupling. These improvements enhance energy efficiency and flight stability in constrained soaring environments.
>
---
#### [new 054] Large language model-based task planning for service robots: A review
- **分类: cs.RO**

- **简介: 该论文聚焦于大语言模型（LLM）在服务机器人任务规划中的应用，旨在提升机器人在复杂家庭环境中自主决策与执行任务的能力。论文综述了LLM的基础技术及其作为机器人“大脑”的作用，分析了多模态输入下的任务规划进展，并指出了当前挑战与未来方向。**

- **链接: [http://arxiv.org/pdf/2510.23357v1](http://arxiv.org/pdf/2510.23357v1)**

> **作者:** Shaohan Bian; Ying Zhang; Guohui Tian; Zhiqiang Miao; Edmond Q. Wu; Simon X. Yang; Changchun Hua
>
> **备注:** Submitted to Biomimetic Intelligence and Robotics for possible publication
>
> **摘要:** With the rapid advancement of large language models (LLMs) and robotics, service robots are increasingly becoming an integral part of daily life, offering a wide range of services in complex environments. To deliver these services intelligently and efficiently, robust and accurate task planning capabilities are essential. This paper presents a comprehensive overview of the integration of LLMs into service robotics, with a particular focus on their role in enhancing robotic task planning. First, the development and foundational techniques of LLMs, including pre-training, fine-tuning, retrieval-augmented generation (RAG), and prompt engineering, are reviewed. We then explore the application of LLMs as the cognitive core-`brain'-of service robots, discussing how LLMs contribute to improved autonomy and decision-making. Furthermore, recent advancements in LLM-driven task planning across various input modalities are analyzed, including text, visual, audio, and multimodal inputs. Finally, we summarize key challenges and limitations in current research and propose future directions to advance the task planning capabilities of service robots in complex, unstructured domestic environments. This review aims to serve as a valuable reference for researchers and practitioners in the fields of artificial intelligence and robotics.
>
---
#### [new 055] Deep Active Inference with Diffusion Policy and Multiple Timescale World Model for Real-World Exploration and Navigation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文针对真实场景下机器人自主导航中的探索与目标导向导航问题，提出基于自由能最小化的深度主动推理框架。结合扩散策略与多时标状态空间模型，实现长期后果预测与动作选择，有效统一探索与导航行为，在真实环境中显著提升成功率并减少碰撞。**

- **链接: [http://arxiv.org/pdf/2510.23258v1](http://arxiv.org/pdf/2510.23258v1)**

> **作者:** Riko Yokozawa; Kentaro Fujii; Yuta Nomura; Shingo Murata
>
> **备注:** Preprint version
>
> **摘要:** Autonomous robotic navigation in real-world environments requires exploration to acquire environmental information as well as goal-directed navigation in order to reach specified targets. Active inference (AIF) based on the free-energy principle provides a unified framework for these behaviors by minimizing the expected free energy (EFE), thereby combining epistemic and extrinsic values. To realize this practically, we propose a deep AIF framework that integrates a diffusion policy as the policy model and a multiple timescale recurrent state-space model (MTRSSM) as the world model. The diffusion policy generates diverse candidate actions while the MTRSSM predicts their long-horizon consequences through latent imagination, enabling action selection that minimizes EFE. Real-world navigation experiments demonstrated that our framework achieved higher success rates and fewer collisions compared with the baselines, particularly in exploration-demanding scenarios. These results highlight how AIF based on EFE minimization can unify exploration and goal-directed navigation in real-world robotic settings.
>
---
#### [new 056] RoGER-SLAM: A Robust Gaussian Splatting SLAM System for Noisy and Low-light Environment Resilience
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对低光与噪声环境下3D高斯溅射SLAM系统性能下降的问题，提出RoGER-SLAM。通过结构保持的鲁棒融合、自适应跟踪优化和基于CLIP的增强模块，提升系统在恶劣视觉条件下的定位精度与重建质量。**

- **链接: [http://arxiv.org/pdf/2510.22600v1](http://arxiv.org/pdf/2510.22600v1)**

> **作者:** Huilin Yin; Zhaolin Yang; Linchuan Zhang; Gerhard Rigoll; Johannes Betz
>
> **备注:** 13 pages, 11 figures, under review
>
> **摘要:** The reliability of Simultaneous Localization and Mapping (SLAM) is severely constrained in environments where visual inputs suffer from noise and low illumination. Although recent 3D Gaussian Splatting (3DGS) based SLAM frameworks achieve high-fidelity mapping under clean conditions, they remain vulnerable to compounded degradations that degrade mapping and tracking performance. A key observation underlying our work is that the original 3DGS rendering pipeline inherently behaves as an implicit low-pass filter, attenuating high-frequency noise but also risking over-smoothing. Building on this insight, we propose RoGER-SLAM, a robust 3DGS SLAM system tailored for noise and low-light resilience. The framework integrates three innovations: a Structure-Preserving Robust Fusion (SP-RoFusion) mechanism that couples rendered appearance, depth, and edge cues; an adaptive tracking objective with residual balancing regularization; and a Contrastive Language-Image Pretraining (CLIP)-based enhancement module, selectively activated under compounded degradations to restore semantic and structural fidelity. Comprehensive experiments on Replica, TUM, and real-world sequences show that RoGER-SLAM consistently improves trajectory accuracy and reconstruction quality compared with other 3DGS-SLAM systems, especially under adverse imaging conditions.
>
---
#### [new 057] Explicit Memory through Online 3D Gaussian Splatting Improves Class-Agnostic Video Segmentation
- **分类: cs.RO**

- **简介: 该论文针对类无关视频分割任务，解决模型缺乏有效历史记忆导致预测不一致的问题。提出基于在线3D高斯点阵的显式记忆机制，增强FastSAM与SAM2模型，通过显式存储和融合历史分割结果，提升准确性和一致性。**

- **链接: [http://arxiv.org/pdf/2510.23521v1](http://arxiv.org/pdf/2510.23521v1)**

> **作者:** Anthony Opipari; Aravindhan K Krishnan; Shreekant Gayaka; Min Sun; Cheng-Hao Kuo; Arnie Sen; Odest Chadwicke Jenkins
>
> **备注:** Accepted in IEEE Robotics and Automation Letters September 2025
>
> **摘要:** Remembering where object segments were predicted in the past is useful for improving the accuracy and consistency of class-agnostic video segmentation algorithms. Existing video segmentation algorithms typically use either no object-level memory (e.g. FastSAM) or they use implicit memories in the form of recurrent neural network features (e.g. SAM2). In this paper, we augment both types of segmentation models using an explicit 3D memory and show that the resulting models have more accurate and consistent predictions. For this, we develop an online 3D Gaussian Splatting (3DGS) technique to store predicted object-level segments generated throughout the duration of a video. Based on this 3DGS representation, a set of fusion techniques are developed, named FastSAM-Splat and SAM2-Splat, that use the explicit 3DGS memory to improve their respective foundation models' predictions. Ablation experiments are used to validate the proposed techniques' design and hyperparameter settings. Results from both real-world and simulated benchmarking experiments show that models which use explicit 3D memories result in more accurate and consistent predictions than those which use no memory or only implicit neural network memories. Project Page: https://topipari.com/projects/FastSAM-Splat/
>
---
#### [new 058] Forward Kinematics Solution For A General Stewart Platform Through Iteration Based Simulation
- **分类: cs.RO**

- **简介: 该论文针对斯特林平台正向运动学解的多解性问题，提出基于迭代仿真的唯一解求解方法。利用改进的D-H参数与逆运动学数据，实现无需人工验证的精确正解，支撑六自由度材料测试系统中高精度力控与在位标定。**

- **链接: [http://arxiv.org/pdf/2510.22465v1](http://arxiv.org/pdf/2510.22465v1)**

> **作者:** Sourabh Karmakar; Cameron J. Turner
>
> **摘要:** This paper presents a method to generate feasible, unique forward-kinematic solutions for a general Stewart platform. This is done by using inverse kinematics to obtain valid workspace data and corresponding actuator lengths for the moving platform. For parallel kinematic machines, such as the Stewart Platform, inverse kinematics are straight forward, but the forward kinematics are complex and generates multiple solutions due to the closed loop structure of the kinematic links. In this research, a simple iterative algorithm has been used employing modified Denavit-Hartenberg convention. The outcome is encouraging as this method generates a single feasible forward kinematic solution for each valid pose with the solved DH parameters and unlike earlier forward kinematics solutions, this unique solution does not need to be manually verified. Therefore, the forward kinematic solutions can be used directly for further calculations without the need for manual pose verification. This capability is essential for the six degree of freedom materials testing system developed by the authors in their laboratory. The developed system is aimed at characterizing additively manufactured materials under complex combined multiple loading conditions. The material characterization is done by enabling high precision force control on the moving platform via in situ calibration of the as-built kinematics of the Stewart Gough Platform.
>
---
#### [new 059] A Robotic Stirring Method with Trajectory Optimization and Adaptive Speed Control for Accurate Pest Counting in Water Traps
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对水 trap 中害虫计数因遮挡导致精度低的问题，提出基于机械臂的轨迹优化与自适应速度控制搅拌方法。通过对比多种搅拌轨迹，确定最优路径，并设计闭环系统依据计数置信度动态调节搅拌速度，提升计数准确性。属于精准农业中的害虫动态监测任务。**

- **链接: [http://arxiv.org/pdf/2510.21732v1](http://arxiv.org/pdf/2510.21732v1)**

> **作者:** Xumin Gao; Mark Stevens; Grzegorz Cielniak
>
> **备注:** This paper has been submitted to ICRA 2026 and is currently under review
>
> **摘要:** Accurate monitoring of pest population dynamics is crucial for informed decision-making in precision agriculture. Currently, mainstream image-based pest counting methods primarily rely on image processing combined with machine learning or deep learning for pest counting. However, these methods have limitations and struggle to handle situations involving pest occlusion. To address this issue, this paper proposed a robotic stirring method with trajectory optimization and adaptive speed control for accurate pest counting in water traps. First, we developed an automated stirring system for pest counting in yellow water traps based on a robotic arm. Stirring alters the distribution of pests in the yellow water trap, making some of the occluded individuals visible for detection and counting. Then, we investigated the impact of different stirring trajectories on pest counting performance and selected the optimal trajectory for pest counting. Specifically, we designed six representative stirring trajectories, including circle, square, triangle, spiral, four small circles, and random lines, for the robotic arm to stir. And by comparing the overall average counting error and counting confidence of different stirring trajectories across various pest density scenarios, we determined the optimal trajectory. Finally, we proposed a counting confidence-driven closed-loop control system to achieve adaptive-speed stirring. It uses changes in pest counting confidence between consecutive frames as feedback to adjust the stirring speed. To the best of our knowledge, this is the first study dedicated to investigating the effects of different stirring trajectories on object counting in the dynamic liquid environment and to implement adaptive-speed stirring for this type of task. Experimental results show ...
>
---
#### [new 060] Toward Humanoid Brain-Body Co-design: Joint Optimization of Control and Morphology for Fall Recovery
- **分类: cs.RO; cs.AI**

- **简介: 该论文聚焦于人形机器人跌倒恢复的脑-体协同设计任务，旨在通过联合优化控制策略与物理形态提升恢复能力。提出RoboCraft框架，实现控制与形态的迭代协同优化，利用预训练策略与优先级缓冲机制高效搜索高性能设计，实验表明平均性能提升44.55%，形态优化贡献超40%。**

- **链接: [http://arxiv.org/pdf/2510.22336v1](http://arxiv.org/pdf/2510.22336v1)**

> **作者:** Bo Yue; Sheng Xu; Kui Jia; Guiliang Liu
>
> **摘要:** Humanoid robots represent a central frontier in embodied intelligence, as their anthropomorphic form enables natural deployment in humans' workspace. Brain-body co-design for humanoids presents a promising approach to realizing this potential by jointly optimizing control policies and physical morphology. Within this context, fall recovery emerges as a critical capability. It not only enhances safety and resilience but also integrates naturally with locomotion systems, thereby advancing the autonomy of humanoids. In this paper, we propose RoboCraft, a scalable humanoid co-design framework for fall recovery that iteratively improves performance through the coupled updates of control policy and morphology. A shared policy pretrained across multiple designs is progressively finetuned on high-performing morphologies, enabling efficient adaptation without retraining from scratch. Concurrently, morphology search is guided by human-inspired priors and optimization algorithms, supported by a priority buffer that balances reevaluation of promising candidates with the exploration of novel designs. Experiments show that \ourmethod{} achieves an average performance gain of 44.55% on seven public humanoid robots, with morphology optimization drives at least 40% of improvements in co-designing four humanoid robots, underscoring the critical role of humanoid co-design.
>
---
#### [new 061] On Steerability Factors for Growing Vine Robots
- **分类: cs.RO**

- **简介: 该论文研究藤蔓机器人的可操控性，旨在优化其在复杂环境中的弯曲运动能力。针对负载、压力、尺寸及制造工艺等因素，通过实验分析各参数影响，提出优化设计原则，显著提升机器人在垂直与水平方向的曲率表现，解决软体机器人在救援等场景中运动性能受限的问题。**

- **链接: [http://arxiv.org/pdf/2510.22504v1](http://arxiv.org/pdf/2510.22504v1)**

> **作者:** Ciera McFarland; Antonio Alvarez; Sarah Taher; Nathaniel Hanson; Margaret McGuinness
>
> **摘要:** Vine robots extend their tubular bodies by everting material from the tip, enabling navigation in complex environments with a minimalist soft body. Despite their promise for field applications, especially in the urban search and rescue domain, performance is constrained by the weight of attached sensors or tools, as well as other design and control choices. This work investigates how tip load, pressure, length, diameter, and fabrication method shape vine robot steerability--the ability to maneuver with controlled curvature--for robots that steer with series pouch motor-style pneumatic actuators. We conduct two groups of experiments: (1) studying tip load, chamber pressure, length, and diameter in a robot supporting itself against gravity, and (2) studying fabrication method and ratio of actuator to chamber pressure in a robot supported on the ground. Results show that steerability decreases with increasing tip load, is best at moderate chamber pressure, increases with length, and is largely unaffected by diameter. Robots with actuators attached on their exterior begin curving at low pressure ratios, but curvature saturates at high pressure ratios; those with actuators integrated into the robot body require higher pressure ratios to begin curving but achieve higher curvature overall. We demonstrate that robots optimized with these principles outperform those with ad hoc parameters in a mobility task that involves maximizing upward and horizontal curvatures.
>
---
#### [new 062] HyPerNav: Hybrid Perception for Object-Oriented Navigation in Unknown Environment
- **分类: cs.RO; cs.AI**

- **简介: 该论文聚焦于未知环境中的目标导向导航任务，旨在提升机器人在复杂场景下定位目标物体的能力。针对现有方法多依赖单一感知模态的问题，提出HyPerNav框架，融合RGB-D相机的局部视觉与俯视图地图的全局信息，利用视觉语言模型实现多模态协同感知，显著提升导航效率与智能水平。**

- **链接: [http://arxiv.org/pdf/2510.22917v1](http://arxiv.org/pdf/2510.22917v1)**

> **作者:** Zecheng Yin; Hao Zhao; Zhen Li
>
> **备注:** under review
>
> **摘要:** Objective-oriented navigation(ObjNav) enables robot to navigate to target object directly and autonomously in an unknown environment. Effective perception in navigation in unknown environment is critical for autonomous robots. While egocentric observations from RGB-D sensors provide abundant local information, real-time top-down maps offer valuable global context for ObjNav. Nevertheless, the majority of existing studies focus on a single source, seldom integrating these two complementary perceptual modalities, despite the fact that humans naturally attend to both. With the rapid advancement of Vision-Language Models(VLMs), we propose Hybrid Perception Navigation (HyPerNav), leveraging VLMs' strong reasoning and vision-language understanding capabilities to jointly perceive both local and global information to enhance the effectiveness and intelligence of navigation in unknown environments. In both massive simulation evaluation and real-world validation, our methods achieved state-of-the-art performance against popular baselines. Benefiting from hybrid perception approach, our method captures richer cues and finds the objects more effectively, by simultaneously leveraging information understanding from egocentric observations and the top-down map. Our ablation study further proved that either of the hybrid perception contributes to the navigation performance.
>
---
#### [new 063] Bridging Perception and Reasoning: Dual-Pipeline Neuro-Symbolic Landing for UAVs in Cluttered Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对无人机在复杂环境中的自主着陆任务，提出神经符号框架NeuroSymLand。通过离线生成可验证的符号知识与在线实时推理结合，融合感知与逻辑推理，提升鲁棒性、可解释性与安全性，有效应对环境变化与数据依赖问题。**

- **链接: [http://arxiv.org/pdf/2510.22204v1](http://arxiv.org/pdf/2510.22204v1)**

> **作者:** Weixian Qian; Sebastian Schroder; Yao Deng; Jiaohong Yao; Linfeng Liang; Xiao Cheng; Richard Han; Xi Zheng
>
> **摘要:** Autonomous landing in unstructured (cluttered, uneven, and map-poor) environments is a core requirement for Unmanned Aerial Vehicles (UAVs), yet purely vision-based or deep learning models often falter under covariate shift and provide limited interpretability. We propose NeuroSymLand, a neuro-symbolic framework that tightly couples two complementary pipelines: (i) an offline pipeline, where Large Language Models (LLMs) and human-in-the-loop refinement synthesize Scallop code from diverse landing scenarios, distilling generalizable and verifiable symbolic knowledge; and (ii) an online pipeline, where a compact foundation-based semantic segmentation model generates probabilistic Scallop facts that are composed into semantic scene graphs for real-time deductive reasoning. This design combines the perceptual strengths of lightweight foundation models with the interpretability and verifiability of symbolic reasoning. Node attributes (e.g., flatness, area) and edge relations (adjacency, containment, proximity) are computed with geometric routines rather than learned, avoiding the data dependence and latency of train-time graph builders. The resulting Scallop program encodes landing principles (avoid water and obstacles; prefer large, flat, accessible regions) and yields calibrated safety scores with ranked Regions of Interest (ROIs) and human-readable justifications. Extensive evaluations across datasets, diverse simulation maps, and real UAV hardware show that NeuroSymLand achieves higher accuracy, stronger robustness to covariate shift, and superior efficiency compared with state-of-the-art baselines, while advancing UAV safety and reliability in emergency response, surveillance, and delivery missions.
>
---
#### [new 064] Localising under the drape: proprioception in the era of distributed surgical robotic system
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出一种无标记的本体感知方法，解决分布式手术机器人在无视觉线索下的精确定位问题。基于轻量级双目相机与新型Transformer模型，利用大规模自标注数据实现对全机器人及术野的鲁棒跟踪，提升视野覆盖25%，支持多机器人协同与智能控制，推动模块化自主手术发展。**

- **链接: [http://arxiv.org/pdf/2510.23512v1](http://arxiv.org/pdf/2510.23512v1)**

> **作者:** Martin Huber; Nicola A. Cavalcanti; Ayoob Davoodi; Ruixuan Li; Christopher E. Mower; Fabio Carrillo; Christoph J. Laux; Francois Teyssere; Thibault Chandanson; Antoine Harlé; Elie Saghbiny; Mazda Farshad; Guillaume Morel; Emmanuel Vander Poorten; Philipp Fürnstahl; Sébastien Ourselin; Christos Bergeles; Tom Vercauteren
>
> **摘要:** Despite their mechanical sophistication, surgical robots remain blind to their surroundings. This lack of spatial awareness causes collisions, system recoveries, and workflow disruptions, issues that will intensify with the introduction of distributed robots with independent interacting arms. Existing tracking systems rely on bulky infrared cameras and reflective markers, providing only limited views of the surgical scene and adding hardware burden in crowded operating rooms. We present a marker-free proprioception method that enables precise localisation of surgical robots under their sterile draping despite associated obstruction of visual cues. Our method solely relies on lightweight stereo-RGB cameras and novel transformer-based deep learning models. It builds on the largest multi-centre spatial robotic surgery dataset to date (1.4M self-annotated images from human cadaveric and preclinical in vivo studies). By tracking the entire robot and surgical scene, rather than individual markers, our approach provides a holistic view robust to occlusions, supporting surgical scene understanding and context-aware control. We demonstrate an example of potential clinical benefits during in vivo breathing compensation with access to tissue dynamics, unobservable under state of the art tracking, and accurately locate in multi-robot systems for future intelligent interaction. In addition, and compared with existing systems, our method eliminates markers and improves tracking visibility by 25%. To our knowledge, this is the first demonstration of marker-free proprioception for fully draped surgical robots, reducing setup complexity, enhancing safety, and paving the way toward modular and autonomous robotic surgery.
>
---
#### [new 065] A Physics-Informed Neural Network Approach for UAV Path Planning in Dynamic Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文针对动态风场中无人机路径规划问题，提出一种融合物理规律的神经网络方法。通过嵌入动力学模型与环境约束，无需监督数据即可生成安全、平滑且节能的轨迹，有效克服传统算法在平滑性与实时性上的不足。**

- **链接: [http://arxiv.org/pdf/2510.21874v1](http://arxiv.org/pdf/2510.21874v1)**

> **作者:** Shuning Zhang
>
> **备注:** 15 pages, 8 figures
>
> **摘要:** Unmanned aerial vehicles (UAVs) operating in dynamic wind fields must generate safe and energy-efficient trajectories under physical and environmental constraints. Traditional planners, such as A* and kinodynamic RRT*, often yield suboptimal or non-smooth paths due to discretization and sampling limitations. This paper presents a physics-informed neural network (PINN) framework that embeds UAV dynamics, wind disturbances, and obstacle avoidance directly into the learning process. Without requiring supervised data, the PINN learns dynamically feasible and collision-free trajectories by minimizing physical residuals and risk-aware objectives. Comparative simulations show that the proposed method outperforms A* and Kino-RRT* in control energy, smoothness, and safety margin, while maintaining similar flight efficiency. The results highlight the potential of physics-informed learning to unify model-based and data-driven planning, providing a scalable and physically consistent framework for UAV trajectory optimization.
>
---
#### [new 066] Learning Neural Observer-Predictor Models for Limb-level Sampling-based Locomotion Planning
- **分类: cs.RO**

- **简介: 该论文针对足式机器人在复杂环境中的安全导航问题，提出一种基于神经网络的观测-预测框架。通过可证明稳定的神经观测器实现精确状态估计，并驱动高效预测器，支持采样规划中数千条轨迹的快速评估，实现肢体级碰撞检测与避障。**

- **链接: [http://arxiv.org/pdf/2510.22789v1](http://arxiv.org/pdf/2510.22789v1)**

> **作者:** Abhijeet M. Kulkarni; Ioannis Poulakakis; Guoquan Huang
>
> **摘要:** Accurate full-body motion prediction is essential for the safe, autonomous navigation of legged robots, enabling critical capabilities like limb-level collision checking in cluttered environments. Simplified kinematic models often fail to capture the complex, closed-loop dynamics of the robot and its low-level controller, limiting their predictions to simple planar motion. To address this, we present a learning-based observer-predictor framework that accurately predicts this motion. Our method features a neural observer with provable UUB guarantees that provides a reliable latent state estimate from a history of proprioceptive measurements. This stable estimate initializes a computationally efficient predictor, designed for the rapid, parallel evaluation of thousands of potential trajectories required by modern sampling-based planners. We validated the system by integrating our neural predictor into an MPPI-based planner on a Vision 60 quadruped. Hardware experiments successfully demonstrated effective, limb-aware motion planning in a challenging, narrow passage and over small objects, highlighting our system's ability to provide a robust foundation for high-performance, collision-aware planning on dynamic robotic platforms.
>
---
#### [new 067] COOPERA: Continual Open-Ended Human-Robot Assistance
- **分类: cs.RO**

- **简介: 该论文提出COOPERA框架，解决机器人在长期、开放环境中个性化协作的问题。通过模拟具有心理特质和长期意图的虚拟人类，实现持续人机协同。工作包括构建基准、学习人类特质与情境意图，并验证其对长期协作的有效性。**

- **链接: [http://arxiv.org/pdf/2510.23495v1](http://arxiv.org/pdf/2510.23495v1)**

> **作者:** Chenyang Ma; Kai Lu; Ruta Desai; Xavier Puig; Andrew Markham; Niki Trigoni
>
> **备注:** NeurIPS 2025 (Spotlight); Project Page: https://dannymcy.github.io/coopera/
>
> **摘要:** To understand and collaborate with humans, robots must account for individual human traits, habits, and activities over time. However, most robotic assistants lack these abilities, as they primarily focus on predefined tasks in structured environments and lack a human model to learn from. This work introduces COOPERA, a novel framework for COntinual, OPen-Ended human-Robot Assistance, where simulated humans, driven by psychological traits and long-term intentions, interact with robots in complex environments. By integrating continuous human feedback, our framework, for the first time, enables the study of long-term, open-ended human-robot collaboration (HRC) in different collaborative tasks across various time-scales. Within COOPERA, we introduce a benchmark and an approach to personalize the robot's collaborative actions by learning human traits and context-dependent intents. Experiments validate the extent to which our simulated humans reflect realistic human behaviors and demonstrate the value of inferring and personalizing to human intents for open-ended and long-term HRC. Project Page: https://dannymcy.github.io/coopera/
>
---
#### [new 068] Dexbotic: Open-Source Vision-Language-Action Toolbox
- **分类: cs.RO**

- **简介: 该论文提出Dexbotic，一个基于PyTorch的开源视觉-语言-动作（VLA）模型工具箱，旨在支持具身智能领域的研究。它提供多模型统一代码库与强预训练模型，简化实验复现与开发，解决VLA研究环境复杂、复现困难的问题，持续集成最新模型以推动领域发展。**

- **链接: [http://arxiv.org/pdf/2510.23511v1](http://arxiv.org/pdf/2510.23511v1)**

> **作者:** Bin Xie; Erjin Zhou; Fan Jia; Hao Shi; Haoqiang Fan; Haowei Zhang; Hebei Li; Jianjian Sun; Jie Bin; Junwen Huang; Kai Liu; Kaixin Liu; Kefan Gu; Lin Sun; Meng Zhang; Peilong Han; Ruitao Hao; Ruitao Zhang; Saike Huang; Songhan Xie; Tiancai Wang; Tianle Liu; Wenbin Tang; Wenqi Zhu; Yang Chen; Yingfei Liu; Yizhuang Zhou; Yu Liu; Yucheng Zhao; Yunchao Ma; Yunfei Wei; Yuxiang Chen; Ze Chen; Zeming Li; Zhao Wu; Ziheng Zhang; Ziming Liu; Ziwei Yan; Ziyu Zhang
>
> **备注:** Authors are listed in alphabetical order. The official website is located at https://dexbotic.com/. Code is available at https://github.com/Dexmal/dexbotic
>
> **摘要:** In this paper, we present Dexbotic, an open-source Vision-Language-Action (VLA) model toolbox based on PyTorch. It aims to provide a one-stop VLA research service for professionals in the field of embodied intelligence. It offers a codebase that supports multiple mainstream VLA policies simultaneously, allowing users to reproduce various VLA methods with just a single environment setup. The toolbox is experiment-centric, where the users can quickly develop new VLA experiments by simply modifying the Exp script. Moreover, we provide much stronger pretrained models to achieve great performance improvements for state-of-the-art VLA policies. Dexbotic will continuously update to include more of the latest pre-trained foundation models and cutting-edge VLA models in the industry.
>
---
#### [new 069] Taxonomy and Trends in Reinforcement Learning for Robotics and Control Systems: A Structured Review
- **分类: cs.RO; cs.LG**

- **简介: 该论文综述强化学习在机器人与控制系统中的应用，聚焦高维连续控制任务。针对理论与实践脱节问题，构建分类体系，系统梳理MDP框架、主流DRL算法及部署模式，归纳技术趋势与设计范式，推动RL在真实机器人场景的落地。**

- **链接: [http://arxiv.org/pdf/2510.21758v1](http://arxiv.org/pdf/2510.21758v1)**

> **作者:** Kumater Ter; RexCharles Donatus; Ore-Ofe Ajayi; Daniel Udekwe
>
> **摘要:** Reinforcement learning (RL) has become a foundational approach for enabling intelligent robotic behavior in dynamic and uncertain environments. This work presents an in-depth review of RL principles, advanced deep reinforcement learning (DRL) algorithms, and their integration into robotic and control systems. Beginning with the formalism of Markov Decision Processes (MDPs), the study outlines essential elements of the agent-environment interaction and explores core algorithmic strategies including actor-critic methods, value-based learning, and policy gradients. Emphasis is placed on modern DRL techniques such as DDPG, TD3, PPO, and SAC, which have shown promise in solving high-dimensional, continuous control tasks. A structured taxonomy is introduced to categorize RL applications across domains such as locomotion, manipulation, multi-agent coordination, and human-robot interaction, along with training methodologies and deployment readiness levels. The review synthesizes recent research efforts, highlighting technical trends, design patterns, and the growing maturity of RL in real-world robotics. Overall, this work aims to bridge theoretical advances with practical implementations, providing a consolidated perspective on the evolving role of RL in autonomous robotic systems.
>
---
#### [new 070] Next-Generation LLM for UAV: From Natural Language to Autonomous Flight
- **分类: cs.RO; cs.AI; cs.CL; cs.SY; eess.SY**

- **简介: 该论文提出下一代无人机大语言模型系统NeLV，旨在将自然语言指令转化为多尺度无人机自主飞行任务。针对现有研究局限于小型无人机、缺乏全流程自动化的问题，构建五组件架构实现从指令解析到飞行控制的闭环，并提出五级自动化分级体系，推动无人机向全自主飞行演进。**

- **链接: [http://arxiv.org/pdf/2510.21739v1](http://arxiv.org/pdf/2510.21739v1)**

> **作者:** Liangqi Yuan; Chuhao Deng; Dong-Jun Han; Inseok Hwang; Sabine Brunswicker; Christopher G. Brinton
>
> **摘要:** With the rapid advancement of Large Language Models (LLMs), their capabilities in various automation domains, particularly Unmanned Aerial Vehicle (UAV) operations, have garnered increasing attention. Current research remains predominantly constrained to small-scale UAV applications, with most studies focusing on isolated components such as path planning for toy drones, while lacking comprehensive investigation of medium- and long-range UAV systems in real-world operational contexts. Larger UAV platforms introduce distinct challenges, including stringent requirements for airport-based take-off and landing procedures, adherence to complex regulatory frameworks, and specialized operational capabilities with elevated mission expectations. This position paper presents the Next-Generation LLM for UAV (NeLV) system -- a comprehensive demonstration and automation roadmap for integrating LLMs into multi-scale UAV operations. The NeLV system processes natural language instructions to orchestrate short-, medium-, and long-range UAV missions through five key technical components: (i) LLM-as-Parser for instruction interpretation, (ii) Route Planner for Points of Interest (POI) determination, (iii) Path Planner for waypoint generation, (iv) Control Platform for executable trajectory implementation, and (v) UAV monitoring. We demonstrate the system's feasibility through three representative use cases spanning different operational scales: multi-UAV patrol, multi-POI delivery, and multi-hop relocation. Beyond the current implementation, we establish a five-level automation taxonomy that charts the evolution from current LLM-as-Parser capabilities (Level 1) to fully autonomous LLM-as-Autopilot systems (Level 5), identifying technical prerequisites and research challenges at each stage.
>
---
#### [new 071] Force-Displacement Profiling for Robot-Assisted Deployment of a Left Atrial Appendage Occluder Using FBG-EM Distal Sensing
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对左心耳封堵术中定位不准、辐射暴露等问题，提出基于光纤光栅与电磁追踪的力-位移动态监测方法。通过机器人辅助在模拟环境中实现无辐射实时反馈，精准识别部署关键步骤，降低机械应力，提升手术安全性与准确性。**

- **链接: [http://arxiv.org/pdf/2510.21734v1](http://arxiv.org/pdf/2510.21734v1)**

> **作者:** Giovanni Battista Regazzo; Wim-Alexander Beckers; Xuan Thao Ha; Mouloud Ourak; Johan Vlekken; Emmanuel Vander Poorten
>
> **备注:** Presented at the Conference on New Technologies for Computer and Robot Assisted Surgery (CRAS2025)
>
> **摘要:** Atrial fibrillation (AF) increases the risk of thromboembolic events due to impaired function of the left atrial appendage (LAA). Left atrial appendage closure (LAAC) is a minimally invasive intervention designed to reduce stroke risk by sealing the LAA with an expandable occluder device. Current deployment relies on manual catheter control and imaging modalities like fluoroscopy and transesophageal echocardiography, which carry limitations including radiation exposure and limited positioning precision. In this study, we leverage a previously developed force-sensing delivery sheath integrating fiber Bragg gratings (FBGs) at the interface between the catheter and the occluder. Combined with electromagnetic (EM) tracking, this setup enables real-time measurement of interaction forces and catheter tip position during robot-assisted LAAC deployment in an anatomical phantom. We present a novel force-displacement profiling method that characterizes occluder deployment dynamics and identifies key procedural steps without relying on ionizing radiation. The force profiles reveal low-magnitude interaction forces, suggesting minimal mechanical stress on the surrounding anatomy. This approach shows promise in providing clinicians with enhanced intraoperative feedback, improving deployment outcome. Future work will focus on automating deployment steps classification and validating the sensing strategy in dynamic, realistic environments.
>
---
#### [new 072] T-ESKF: Transformed Error-State Kalman Filter for Consistent Visual-Inertial Navigation
- **分类: cs.RO**

- **简介: 该论文针对视觉惯性导航系统（VINS）中的可观测性不一致问题，提出变换误差状态卡尔曼滤波器（T-ESKF）。通过引入时变线性变换，使误差状态的不可观测子空间与状态解耦，提升估计一致性。同时设计高效协方差传播方法，实验证明其性能优于或媲美现有方法。**

- **链接: [http://arxiv.org/pdf/2510.23359v1](http://arxiv.org/pdf/2510.23359v1)**

> **作者:** Chungeng Tian; Ning Hao; Fenghua He
>
> **备注:** This paper was submitted to IEEE RA-L on July 14, 2024, and accepted on December 18, 2024. This version serves as the 'plus edition' of the accepted paper, incorporating supplementary materials for completeness
>
> **摘要:** This paper presents a novel approach to address the inconsistency problem caused by observability mismatch in visual-inertial navigation systems (VINS). The key idea involves applying a linear time-varying transformation to the error-state within the Error-State Kalman Filter (ESKF). This transformation ensures that \textrr{the unobservable subspace of the transformed error-state system} becomes independent of the state, thereby preserving the correct observability of the transformed system against variations in linearization points. We introduce the Transformed ESKF (T-ESKF), a consistent VINS estimator that performs state estimation using the transformed error-state system. Furthermore, we develop an efficient propagation technique to accelerate the covariance propagation based on the transformation relationship between the transition and accumulated matrices of T-ESKF and ESKF. We validate the proposed method through extensive simulations and experiments, demonstrating better (or competitive at least) performance compared to state-of-the-art methods. The code is available at github.com/HITCSC/T-ESKF.
>
---
#### [new 073] RobotArena $\infty$: Scalable Robot Benchmarking via Real-to-Sim Translation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出RobotArena∞，一个基于真实到仿真转换的可扩展机器人评估框架。针对真实测试耗时、安全风险高及仿真基准局限性问题，利用视觉语言模型与3D生成技术将真实视频演示转为仿真环境，结合自动评分与众包偏好判断，实现对机器人政策的高效、可复现评估，并通过环境扰动测试其泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.23571v1](http://arxiv.org/pdf/2510.23571v1)**

> **作者:** Yash Jangir; Yidi Zhang; Kashu Yamazaki; Chenyu Zhang; Kuan-Hsun Tu; Tsung-Wei Ke; Lei Ke; Yonatan Bisk; Katerina Fragkiadaki
>
> **备注:** Website: https://robotarenainf.github.io
>
> **摘要:** The pursuit of robot generalists - instructable agents capable of performing diverse tasks across diverse environments - demands rigorous and scalable evaluation. Yet real-world testing of robot policies remains fundamentally constrained: it is labor-intensive, slow, unsafe at scale, and difficult to reproduce. Existing simulation benchmarks are similarly limited, as they train and test policies within the same synthetic domains and cannot assess models trained from real-world demonstrations or alternative simulation environments. As policies expand in scope and complexity, these barriers only intensify, since defining "success" in robotics often hinges on nuanced human judgments of execution quality. In this paper, we introduce a new benchmarking framework that overcomes these challenges by shifting VLA evaluation into large-scale simulated environments augmented with online human feedback. Leveraging advances in vision-language models, 2D-to-3D generative modeling, and differentiable rendering, our approach automatically converts video demonstrations from widely used robot datasets into simulated counterparts. Within these digital twins, we assess VLA policies using both automated VLM-guided scoring and scalable human preference judgments collected from crowdworkers, transforming human involvement from tedious scene setup, resetting, and safety supervision into lightweight preference comparisons. To measure robustness, we systematically perturb simulated environments along multiple axes, such as textures and object placements, stress-testing policy generalization under controlled variation. The result is a continuously evolving, reproducible, and scalable benchmark for real-world trained robot manipulation policies, addressing a critical missing capability in today's robotics landscape.
>
---
#### [new 074] Multi-Agent Pose Uncertainty: A Differentiable Rendering Cramér-Rao Bound
- **分类: cs.CV; cs.GR; cs.LG; cs.RO**

- **简介: 该论文针对计算机视觉中的姿态估计任务，解决密集或学习模型下姿态不确定性量化难题。通过将可微渲染器视为测量函数，推导出基于流形扰动的可微分Cramér-Rao下界，实现姿态协方差的闭式下界计算，并自然扩展至多相机协同场景，无需关键点匹配即可用于协同感知与新视角合成。**

- **链接: [http://arxiv.org/pdf/2510.21785v1](http://arxiv.org/pdf/2510.21785v1)**

> **作者:** Arun Muthukkumar
>
> **备注:** 5 pages, 3 figures, 1 table. Presented at IEEE/CVF International Conference on Computer Vision (ICCV 2025) and IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** Pose estimation is essential for many applications within computer vision and robotics. Despite its uses, few works provide rigorous uncertainty quantification for poses under dense or learned models. We derive a closed-form lower bound on the covariance of camera pose estimates by treating a differentiable renderer as a measurement function. Linearizing image formation with respect to a small pose perturbation on the manifold yields a render-aware Cram\'er-Rao bound. Our approach reduces to classical bundle-adjustment uncertainty, ensuring continuity with vision theory. It also naturally extends to multi-agent settings by fusing Fisher information across cameras. Our statistical formulation has downstream applications for tasks such as cooperative perception and novel view synthesis without requiring explicit keypoint correspondences.
>
---
#### [new 075] Embodied Navigation with Auxiliary Task of Action Description Prediction
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究多模态机器人室内导航任务，针对强化学习决策系统缺乏可解释性的问题，提出将动作语言描述作为辅助任务。通过知识蒸馏融合预训练视觉-语言模型，实现高精度导航与自然语言描述的协同优化，在语义视听导航任务中达到领先性能。**

- **链接: [http://arxiv.org/pdf/2510.21809v1](http://arxiv.org/pdf/2510.21809v1)**

> **作者:** Haru Kondoh; Asako Kanezaki
>
> **备注:** ICCV 2025 Poster
>
> **摘要:** The field of multimodal robot navigation in indoor environments has garnered significant attention in recent years. However, as tasks and methods become more advanced, the action decision systems tend to become more complex and operate as black-boxes. For a reliable system, the ability to explain or describe its decisions is crucial; however, there tends to be a trade-off in that explainable systems can not outperform non-explainable systems in terms of performance. In this paper, we propose incorporating the task of describing actions in language into the reinforcement learning of navigation as an auxiliary task. Existing studies have found it difficult to incorporate describing actions into reinforcement learning due to the absence of ground-truth data. We address this issue by leveraging knowledge distillation from pre-trained description generation models, such as vision-language models. We comprehensively evaluate our approach across various navigation tasks, demonstrating that it can describe actions while attaining high navigation performance. Furthermore, it achieves state-of-the-art performance in the particularly challenging multimodal navigation task of semantic audio-visual navigation.
>
---
#### [new 076] CGoT: A Novel Inference Mechanism for Embodied Multi-Agent Systems Using Composable Graphs of Thoughts
- **分类: cs.MA; cs.RO**

- **简介: 该论文提出CGoT机制，用于协同运输机器人与自动驾驶车辆的多智能体系统。针对车辆携带机器人执行任务时的推理效率问题，利用可组合思维图（Composable Graphs of Thoughts）提升协作智能体的决策能力。实验验证了该方法在提升系统运行效率方面的有效性。**

- **链接: [http://arxiv.org/pdf/2510.22235v1](http://arxiv.org/pdf/2510.22235v1)**

> **作者:** Yixiao Nie; Yang Zhang; Yingjie Jin; Zhepeng Wang; Xiu Li; Xiang Li
>
> **摘要:** The integration of self-driving cars and service robots is becoming increasingly prevalent across a wide array of fields, playing a crucial and expanding role in both industrial applications and everyday life. In parallel, the rapid advancements in Large Language Models (LLMs) have garnered substantial attention and interest within the research community. This paper introduces a novel vehicle-robot system that leverages the strengths of both autonomous vehicles and service robots. In our proposed system, two autonomous ego-vehicles transports service robots to locations within an office park, where they perform a series of tasks. The study explores the feasibility and potential benefits of incorporating LLMs into this system, with the aim of enhancing operational efficiency and maximizing the potential of the cooperative mechanisms between the vehicles and the robots. This paper proposes a novel inference mechanism which is called CGOT toward this type of system where an agent can carry another agent. Experimental results are presented to validate the performance of the proposed method.
>
---
#### [new 077] DPGLA: Bridging the Gap between Synthetic and Real Data for Unsupervised Domain Adaptation in 3D LiDAR Semantic Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对3D LiDAR语义分割中的无监督域自适应问题，解决合成数据与真实数据间的域偏移及未标注数据利用不足。提出动态伪标签过滤（DPLF）和先验引导的数据增强（PG-DAP），结合数据混合一致性损失，有效提升模型性能。**

- **链接: [http://arxiv.org/pdf/2510.23525v1](http://arxiv.org/pdf/2510.23525v1)**

> **作者:** Wanmeng Li; Simone Mosco; Daniel Fusaro; Alberto Pretto
>
> **备注:** This paper has been accepted for publication at the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Annotating real-world LiDAR point clouds for use in intelligent autonomous systems is costly. To overcome this limitation, self-training-based Unsupervised Domain Adaptation (UDA) has been widely used to improve point cloud semantic segmentation by leveraging synthetic point cloud data. However, we argue that existing methods do not effectively utilize unlabeled data, as they either rely on predefined or fixed confidence thresholds, resulting in suboptimal performance. In this paper, we propose a Dynamic Pseudo-Label Filtering (DPLF) scheme to enhance real data utilization in point cloud UDA semantic segmentation. Additionally, we design a simple and efficient Prior-Guided Data Augmentation Pipeline (PG-DAP) to mitigate domain shift between synthetic and real-world point clouds. Finally, we utilize data mixing consistency loss to push the model to learn context-free representations. We implement and thoroughly evaluate our approach through extensive comparisons with state-of-the-art methods. Experiments on two challenging synthetic-to-real point cloud semantic segmentation tasks demonstrate that our approach achieves superior performance. Ablation studies confirm the effectiveness of the DPLF and PG-DAP modules. We release the code of our method in this paper.
>
---
#### [new 078] Bag-of-Word-Groups (BoWG): A Robust and Efficient Loop Closure Detection Method Under Perceptual Aliasing
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对SLAM中的回环检测任务，解决感知相似环境下因重复纹理导致的误检问题。提出BoWG方法，通过视觉词组捕捉空间共现关系，结合时序一致性与特征分布分析，提升精度与效率，在公开及自建数据集上均优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.22529v1](http://arxiv.org/pdf/2510.22529v1)**

> **作者:** Xiang Fei; Tina Tian; Howie Choset; Lu Li
>
> **备注:** This paper has been accepted by IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Loop closure is critical in Simultaneous Localization and Mapping (SLAM) systems to reduce accumulative drift and ensure global mapping consistency. However, conventional methods struggle in perceptually aliased environments, such as narrow pipes, due to vector quantization, feature sparsity, and repetitive textures, while existing solutions often incur high computational costs. This paper presents Bag-of-Word-Groups (BoWG), a novel loop closure detection method that achieves superior precision-recall, robustness, and computational efficiency. The core innovation lies in the introduction of word groups, which captures the spatial co-occurrence and proximity of visual words to construct an online dictionary. Additionally, drawing inspiration from probabilistic transition models, we incorporate temporal consistency directly into similarity computation with an adaptive scheme, substantially improving precision-recall performance. The method is further strengthened by a feature distribution analysis module and dedicated post-verification mechanisms. To evaluate the effectiveness of our method, we conduct experiments on both public datasets and a confined-pipe dataset we constructed. Results demonstrate that BoWG surpasses state-of-the-art methods, including both traditional and learning-based approaches, in terms of precision-recall and computational efficiency. Our approach also exhibits excellent scalability, achieving an average processing time of 16 ms per image across 17,565 images in the Bicocca25b dataset.
>
---
#### [new 079] Track, Inpaint, Resplat: Subject-driven 3D and 4D Generation with Progressive Texture Infilling
- **分类: cs.CV; cs.AI; cs.GR; cs.LG; cs.RO**

- **简介: 该论文提出TIRE方法，解决3D/4D生成中主体身份一致性差的问题。通过视频追踪定位需修改区域，利用主体驱动的2D修复模型渐进填充，并将结果重投影回3D，有效提升跨视角的身份保真度。**

- **链接: [http://arxiv.org/pdf/2510.23605v1](http://arxiv.org/pdf/2510.23605v1)**

> **作者:** Shuhong Zheng; Ashkan Mirzaei; Igor Gilitschenski
>
> **备注:** NeurIPS 2025, 38 pages, 22 figures
>
> **摘要:** Current 3D/4D generation methods are usually optimized for photorealism, efficiency, and aesthetics. However, they often fail to preserve the semantic identity of the subject across different viewpoints. Adapting generation methods with one or few images of a specific subject (also known as Personalization or Subject-driven generation) allows generating visual content that align with the identity of the subject. However, personalized 3D/4D generation is still largely underexplored. In this work, we introduce TIRE (Track, Inpaint, REsplat), a novel method for subject-driven 3D/4D generation. It takes an initial 3D asset produced by an existing 3D generative model as input and uses video tracking to identify the regions that need to be modified. Then, we adopt a subject-driven 2D inpainting model for progressively infilling the identified regions. Finally, we resplat the modified 2D multi-view observations back to 3D while still maintaining consistency. Extensive experiments demonstrate that our approach significantly improves identity preservation in 3D/4D generation compared to state-of-the-art methods. Our project website is available at https://zsh2000.github.io/track-inpaint-resplat.github.io/.
>
---
#### [new 080] Payload trajectory tracking control for aerial transportation systems with cable length online optimization
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文针对可变缆长的空中运输系统，解决多旋翼与载荷间复杂非线性耦合下的轨迹跟踪难题。提出基于反步法的控制策略，并设计在线优化缆长的生成器，实现无需人工规划的协同控制，保证系统渐近稳定。**

- **链接: [http://arxiv.org/pdf/2510.23296v1](http://arxiv.org/pdf/2510.23296v1)**

> **作者:** Hai Yu; Zhichao Yang; Wei He; Jianda Han; Yongchun Fang; Xiao Liang
>
> **摘要:** Cable-suspended aerial transportation systems are employed extensively across various industries. The capability to flexibly adjust the relative position between the multirotor and the payload has spurred growing interest in the system equipped with variable-length cable, promising broader application potential. Compared to systems with fixed-length cables, introducing the variable-length cable adds a new degree of freedom. However, it also results in increased nonlinearity and more complex dynamic coupling among the multirotor, the cable and the payload, posing significant challenges in control design. This paper introduces a backstepping control strategy tailored for aerial transportation systems with variable-length cable, designed to precisely track the payload trajectory while dynamically adjusting cable length. Then, a cable length generator has been developed that achieves online optimization of the cable length while satisfying state constraints, thus balancing the multirotor's motion and cable length changes without the need for manual trajectory planning. The asymptotic stability of the closed-loop system is guaranteed through Lyapunov techniques and the growth restriction condition. Finally, simulation results confirm the efficacy of the proposed method in managing trajectory tracking and cable length adjustments effectively.
>
---
#### [new 081] Planning Oriented Integrated Sensing and Communication
- **分类: eess.SP; cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出面向路径规划的感知与通信一体化（PISAC）框架，解决传统ISAC忽略关键障碍物对行驶效率影响的问题。通过建立功率与感知不确定性间的闭式安全边界，实现功率分配与运动规划的联合优化，显著提升自动驾驶车辆在复杂环境中的安全性和通行效率。**

- **链接: [http://arxiv.org/pdf/2510.23021v1](http://arxiv.org/pdf/2510.23021v1)**

> **作者:** Xibin Jin; Guoliang Li; Shuai Wang; Fan Liu; Miaowen Wen; Huseyin Arslan; Derrick Wing Kwan Ng; Chengzhong Xu
>
> **摘要:** Integrated sensing and communication (ISAC) enables simultaneous localization, environment perception, and data exchange for connected autonomous vehicles. However, most existing ISAC designs prioritize sensing accuracy and communication throughput, treating all targets uniformly and overlooking the impact of critical obstacles on motion efficiency. To overcome this limitation, we propose a planning-oriented ISAC (PISAC) framework that reduces the sensing uncertainty of planning-bottleneck obstacles and expands the safe navigable path for the ego-vehicle, thereby bridging the gap between physical-layer optimization and motion-level planning. The core of PISAC lies in deriving a closed-form safety bound that explicitly links ISAC transmit power to sensing uncertainty, based on the Cram\'er-Rao Bound and occupancy inflation principles. Using this model, we formulate a bilevel power allocation and motion planning (PAMP) problem, where the inner layer optimizes the ISAC beam power distribution and the outer layer computes a collision-free trajectory under uncertainty-aware safety constraints. Comprehensive simulations in high-fidelity urban driving environments demonstrate that PISAC achieves up to 40% higher success rates and over 5% shorter traversal times than existing ISAC-based and communication-oriented benchmarks, validating its effectiveness in enhancing both safety and efficiency.
>
---
#### [new 082] MOGRAS: Human Motion with Grasping in 3D Scenes
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文聚焦于3D场景中人体与物体交互的运动生成任务。针对现有方法在场景感知与精细抓握间缺失的问题，提出MOGRAS数据集，包含带标注的完整人体抓握动作与场景信息，并设计方法提升现有模型在3D场景中的生成能力，显著改善了真实感与物理合理性。**

- **链接: [http://arxiv.org/pdf/2510.22199v1](http://arxiv.org/pdf/2510.22199v1)**

> **作者:** Kunal Bhosikar; Siddharth Katageri; Vivek Madhavaram; Kai Han; Charu Sharma
>
> **备注:** British Machine Vision Conference Workshop - From Scene Understanding to Human Modeling
>
> **摘要:** Generating realistic full-body motion interacting with objects is critical for applications in robotics, virtual reality, and human-computer interaction. While existing methods can generate full-body motion within 3D scenes, they often lack the fidelity for fine-grained tasks like object grasping. Conversely, methods that generate precise grasping motions typically ignore the surrounding 3D scene. This gap, generating full-body grasping motions that are physically plausible within a 3D scene, remains a significant challenge. To address this, we introduce MOGRAS (Human MOtion with GRAsping in 3D Scenes), a large-scale dataset that bridges this gap. MOGRAS provides pre-grasping full-body walking motions and final grasping poses within richly annotated 3D indoor scenes. We leverage MOGRAS to benchmark existing full-body grasping methods and demonstrate their limitations in scene-aware generation. Furthermore, we propose a simple yet effective method to adapt existing approaches to work seamlessly within 3D scenes. Through extensive quantitative and qualitative experiments, we validate the effectiveness of our dataset and highlight the significant improvements our proposed method achieves, paving the way for more realistic human-scene interactions.
>
---
#### [new 083] ATLAS: Actor-Critic Task-Completion with Look-ahead Action Simulation
- **分类: cs.LG; cs.AI; cs.CL; cs.IR; cs.MA; cs.RO**

- **简介: 该论文提出ATLAS，一种无需微调的网页智能体，解决现有方法在新环境中规划效率低的问题。通过构建认知地图、动作模拟与回溯优化，实现高效任务完成，在WebArena-Lite上达63%成功率，显著优于此前最优。**

- **链接: [http://arxiv.org/pdf/2510.22732v1](http://arxiv.org/pdf/2510.22732v1)**

> **作者:** Jiali Cheng; Anjishnu Kumar; Roshan Lal; Rishi Rajasekaran; Hani Ramezani; Omar Zia Khan; Oleg Rokhlenko; Sunny Chiu-Webster; Gang Hua; Hadi Amiri
>
> **备注:** 9 pages, NeurIPS 2025 Workshop on Language Agents and World Models
>
> **摘要:** We observe that current state-of-the-art web-agents are unable to effectively adapt to new environments without neural network fine-tuning, without which they produce inefficient execution plans due to a lack of awareness of the structure and dynamics of the new environment. To address this limitation, we introduce ATLAS (Actor-Critic Task-completion with Look-ahead Action Simulation), a memory-augmented agent that is able to make plans grounded in a model of the environment by simulating the consequences of those actions in cognitive space. Our agent starts by building a "cognitive map" by performing a lightweight curiosity driven exploration of the environment. The planner proposes candidate actions; the simulator predicts their consequences in cognitive space; a critic analyzes the options to select the best roll-out and update the original plan; and a browser executor performs the chosen action. On the WebArena-Lite Benchmark, we achieve a 63% success rate compared to 53.9% success rate for the previously published state-of-the-art. Unlike previous systems, our modular architecture requires no website-specific LLM fine-tuning. Ablations show sizable drops without the world-model, hierarchical planner, and look-ahead-based replanner confirming their complementary roles within the design of our system
>
---
#### [new 084] LOC: A General Language-Guided Framework for Open-Set 3D Occupancy Prediction
- **分类: cs.CV; cs.CL; cs.LG; cs.RO; eess.IV**

- **简介: 该论文提出LOC框架，解决3D场景理解中因数据稀缺导致的开放集占用预测难题。通过语言引导融合多帧激光雷达点云与语义信息，结合对比学习增强特征区分性，实现无需额外训练即可识别未知类别的高精度3D占用预测。**

- **链接: [http://arxiv.org/pdf/2510.22141v1](http://arxiv.org/pdf/2510.22141v1)**

> **作者:** Yuhang Gao; Xiang Xiang; Sheng Zhong; Guoyou Wang
>
> **摘要:** Vision-Language Models (VLMs) have shown significant progress in open-set challenges. However, the limited availability of 3D datasets hinders their effective application in 3D scene understanding. We propose LOC, a general language-guided framework adaptable to various occupancy networks, supporting both supervised and self-supervised learning paradigms. For self-supervised tasks, we employ a strategy that fuses multi-frame LiDAR points for dynamic/static scenes, using Poisson reconstruction to fill voids, and assigning semantics to voxels via K-Nearest Neighbor (KNN) to obtain comprehensive voxel representations. To mitigate feature over-homogenization caused by direct high-dimensional feature distillation, we introduce Densely Contrastive Learning (DCL). DCL leverages dense voxel semantic information and predefined textual prompts. This efficiently enhances open-set recognition without dense pixel-level supervision, and our framework can also leverage existing ground truth to further improve performance. Our model predicts dense voxel features embedded in the CLIP feature space, integrating textual and image pixel information, and classifies based on text and semantic similarity. Experiments on the nuScenes dataset demonstrate the method's superior performance, achieving high-precision predictions for known classes and distinguishing unknown classes without additional training data.
>
---
#### [new 085] EndoWave: Rational-Wavelet 4D Gaussian Splatting for Endoscopic Reconstruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对内窥镜视频三维重建任务，解决光照不一致、组织非刚性运动和视点相关高光等问题。提出EndoWave框架，通过4D高斯溅射结合光流几何约束与多分辨率有理小波监督，提升时空一致性与细节还原能力，显著改善重建质量。**

- **链接: [http://arxiv.org/pdf/2510.23087v1](http://arxiv.org/pdf/2510.23087v1)**

> **作者:** Taoyu Wu; Yiyi Miao; Jiaxin Guo; Ziyan Chen; Sihang Zhao; Zhuoxiao Li; Zhe Tang; Baoru Huang; Limin Yu
>
> **摘要:** In robot-assisted minimally invasive surgery, accurate 3D reconstruction from endoscopic video is vital for downstream tasks and improved outcomes. However, endoscopic scenarios present unique challenges, including photometric inconsistencies, non-rigid tissue motion, and view-dependent highlights. Most 3DGS-based methods that rely solely on appearance constraints for optimizing 3DGS are often insufficient in this context, as these dynamic visual artifacts can mislead the optimization process and lead to inaccurate reconstructions. To address these limitations, we present EndoWave, a unified spatio-temporal Gaussian Splatting framework by incorporating an optical flow-based geometric constraint and a multi-resolution rational wavelet supervision. First, we adopt a unified spatio-temporal Gaussian representation that directly optimizes primitives in a 4D domain. Second, we propose a geometric constraint derived from optical flow to enhance temporal coherence and effectively constrain the 3D structure of the scene. Third, we propose a multi-resolution rational orthogonal wavelet as a constraint, which can effectively separate the details of the endoscope and enhance the rendering performance. Extensive evaluations on two real surgical datasets, EndoNeRF and StereoMIS, demonstrate that our method EndoWave achieves state-of-the-art reconstruction quality and visual accuracy compared to the baseline method.
>
---
#### [new 086] Look and Tell: A Dataset for Multimodal Grounding Across Egocentric and Exocentric Views
- **分类: cs.CV; cs.CL; cs.RO; I.2.10; I.2.9; I.2.7; H.5.2**

- **简介: 该论文提出Look and Tell数据集，用于研究第一人称与第三人称视角下的多模态语义对齐。针对跨视角指代理解难题，通过同步记录眼动、语音与视频，结合3D场景重建，提供2.7k条标注的指代表达，推动具身智能体在情境对话中的理解能力发展。**

- **链接: [http://arxiv.org/pdf/2510.22672v1](http://arxiv.org/pdf/2510.22672v1)**

> **作者:** Anna Deichler; Jonas Beskow
>
> **备注:** 10 pages, 6 figures, 2 tables. Accepted to the NeurIPS 2025 Workshop on SPACE in Vision, Language, and Embodied AI (SpaVLE)
>
> **摘要:** We introduce Look and Tell, a multimodal dataset for studying referential communication across egocentric and exocentric perspectives. Using Meta Project Aria smart glasses and stationary cameras, we recorded synchronized gaze, speech, and video as 25 participants instructed a partner to identify ingredients in a kitchen. Combined with 3D scene reconstructions, this setup provides a benchmark for evaluating how different spatial representations (2D vs. 3D; ego vs. exo) affect multimodal grounding. The dataset contains 3.67 hours of recordings, including 2,707 richly annotated referential expressions, and is designed to advance the development of embodied agents that can understand and engage in situated dialogue.
>
---
#### [new 087] When Robots Say No: Temporal Trust Recovery Through Explanation
- **分类: cs.HC; cs.CL; cs.RO**

- **简介: 该论文研究人机协作中机器人拒绝请求时的信任恢复问题。针对高风险任务中机器人因自主决策导致用户信任受损的问题，通过模拟灭火场景实验发现：提供合理解释可促进信任随时间恢复，有效缓解信任危机。**

- **链接: [http://arxiv.org/pdf/2510.21716v1](http://arxiv.org/pdf/2510.21716v1)**

> **作者:** Nicola Webb; Zijun Huang; Sanja Milivojevic; Chris Baber; Edmund R. Hunt
>
> **摘要:** Mobile robots with some degree of autonomy could deliver significant advantages in high-risk missions such as search and rescue and firefighting. Integrated into a human-robot team (HRT), robots could work effectively to help search hazardous buildings. User trust is a key enabler for HRT, but during a mission, trust can be damaged. With distributed situation awareness, such as when team members are working in different locations, users may be inclined to doubt a robot's integrity if it declines to immediately change its priorities on request. In this paper, we present the results of a computer-based study investigating on-mission trust dynamics in a high-stakes human-robot teaming scenario. Participants (n = 38) played an interactive firefighting game alongside a robot teammate, where a trust violation occurs owing to the robot declining to help the user immediately. We find that when the robot provides an explanation for declining to help, trust better recovers over time, albeit following an initial drop that is comparable to a baseline condition where an explanation for refusal is not provided. Our findings indicate that trust can vary significantly during a mission, notably when robots do not immediately respond to user requests, but that this trust violation can be largely ameliorated over time if adequate explanation is provided.
>
---
#### [new 088] Drone Carry-on Weight and Wind Flow Assessment via Micro-Doppler Analysis
- **分类: physics.app-ph; cs.RO**

- **简介: 该论文属于无人机监控任务，旨在区分风速与载重对无人机飞行的影响。通过微多普勒分析，利用风洞与消声室实验，基于谱图分支特征，实现对风速和载重的分离识别，为无人机安全监管提供新方法。**

- **链接: [http://arxiv.org/pdf/2510.22846v1](http://arxiv.org/pdf/2510.22846v1)**

> **作者:** Dmytro Vovchuk; Oleg Torgovitsky; Mykola Khobzei; Vladyslav Tkach; Sergey Geyman; Anton Kharchevskii; Andrey Sheleg; Toms Salgals; Vjaceslavs Bobrovs; Shai Gizach; Aviel Glam; Niv Haim Mizrahi; Alexander Liberzon; Pavel Ginzburg
>
> **摘要:** Remote monitoring of drones has become a global objective due to emerging applications in national security and managing aerial delivery traffic. Despite their relatively small size, drones can carry significant payloads, which require monitoring, especially in cases of unauthorized transportation of dangerous goods. A drone's flight dynamics heavily depend on outdoor wind conditions and the carry-on weight, which affect the tilt angle of a drone's body and the rotation velocity of the blades. A surveillance radar can capture both effects, provided a sufficient signal-to-noise ratio for the received echoes and an adjusted postprocessing detection algorithm. Here, we conduct a systematic study to demonstrate that micro-Doppler analysis enables the disentanglement of the impacts of wind and weight on a hovering drone. The physics behind the effect is related to the flight controller, as the way the drone counteracts weight and wind differs. When the payload is balanced, it imposes an additional load symmetrically on all four rotors, causing them to rotate faster, thereby generating a blade-related micro-Doppler shift at a higher frequency. However, the impact of the wind is different. The wind attempts to displace the drone, and to counteract this, the drone tilts to the side. As a result, the forward and rear rotors rotate at different velocities to maintain the tilt angle of the drone body relative to the airflow direction. This causes the splitting in the micro-Doppler spectra. By performing a set of experiments in a controlled environment, specifically, an anechoic chamber for electromagnetic isolation and a wind tunnel for imposing deterministic wind conditions, we demonstrate that both wind and payload details can be extracted using a simple deterministic algorithm based on branching in the micro-Doppler spectra.
>
---
#### [new 089] Mixed Density Diffuser: Efficient Planning with Non-uniform Temporal Resolution
- **分类: cs.AI; cs.RO**

- **简介: 该论文针对强化学习中的轨迹规划任务，解决传统扩散模型在时间密度上均匀分布导致的效率与精度失衡问题。提出可调节时序密度的Mixed Density Diffuser（MDD），实现非均匀时间分辨率规划，在保持高效的同时提升长程依赖建模能力，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.23026v1](http://arxiv.org/pdf/2510.23026v1)**

> **作者:** Crimson Stambaugh; Rajesh P. N. Rao
>
> **备注:** European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESSAN) (under review)
>
> **摘要:** Recent studies demonstrate that diffusion planners benefit from sparse-step planning over single-step planning. Training models to skip steps in their trajectories helps capture long-term dependencies without additional or memory computational cost. However, predicting excessively sparse plans degrades performance. We hypothesize this temporal density threshold is non-uniform across a temporal horizon and that certain parts of a planned trajectory should be more densely planned. We propose Mixed Density Diffuser (MDD), a diffusion planner where the densities throughout the horizon are tunable hyperparameters. MDD achieves a new SOTA across the Maze2D, Franka Kitchen, and Antmaze D4RL task domains.
>
---
#### [new 090] Collaborative Task Assignment, Sequencing and Multi-agent Path-finding
- **分类: cs.MA; cs.RO**

- **简介: 该论文研究协同任务分配、排序与多智能体路径规划（TSPF）问题，旨在最小化流程时间并避免碰撞。提出CBS-TS算法，结合MILP优化任务序列与基于MLA*的冲突消解，实现最优完整求解。实验表明其优于基线方法CBSS，成功率与解质量更优。**

- **链接: [http://arxiv.org/pdf/2510.21738v1](http://arxiv.org/pdf/2510.21738v1)**

> **作者:** Yifan Bai; Shruti Kotpalliwar; Christoforos Kanellakis; George Nikolakopoulos
>
> **摘要:** In this article, we address the problem of collaborative task assignment, sequencing, and multi-agent pathfinding (TSPF), where a team of agents must visit a set of task locations without collisions while minimizing flowtime. TSPF incorporates agent-task compatibility constraints and ensures that all tasks are completed. We propose a Conflict-Based Search with Task Sequencing (CBS-TS), an optimal and complete algorithm that alternates between finding new task sequences and resolving conflicts in the paths of current sequences. CBS-TS uses a mixed-integer linear program (MILP) to optimize task sequencing and employs Conflict-Based Search (CBS) with Multi-Label A* (MLA*) for collision-free path planning within a search forest. By invoking MILP for the next-best sequence only when needed, CBS-TS efficiently limits the search space, enhancing computational efficiency while maintaining optimality. We compare the performance of our CBS-TS against Conflict-based Steiner Search (CBSS), a baseline method that, with minor modifications, can address the TSPF problem. Experimental results demonstrate that CBS-TS outperforms CBSS in most testing scenarios, achieving higher success rates and consistently optimal solutions, whereas CBSS achieves near-optimal solutions in some cases. The supplementary video is available at https://youtu.be/QT8BYgvefmU.
>
---
#### [new 091] Separation of Unconscious Robots with Obstructed Visibility
- **分类: cs.DC; cs.RO**

- **简介: 该论文研究无意识机器人在遮挡可见性下的分离问题，目标是使同色机器人聚成同心半圆。相比以往透明模型，新模型中机器人为不透明，会遮挡视线。提出一种无需知晓总数的碰撞避免算法，在半同步调度下以O(n)轮完成分离。**

- **链接: [http://arxiv.org/pdf/2510.22434v1](http://arxiv.org/pdf/2510.22434v1)**

> **作者:** Prajyot Pyati; Navjot Kaur; Saswata Jana; Adri Bhattacharya; Partha Sarathi Mandal
>
> **摘要:** We study a recently introduced \textit{unconscious} mobile robot model, where each robot is associated with a \textit{color}, which is visible to other robots but not to itself. The robots are autonomous, anonymous, oblivious and silent, operating in the Euclidean plane under the conventional \textit{Look-Compute-Move} cycle. A primary task in this model is the \textit{separation problem}, where unconscious robots sharing the same color must separate from others, forming recognizable geometric shapes such as circles, points, or lines. All prior works model the robots as \textit{transparent}, enabling each to know the positions and colors of all other robots. In contrast, we model the robots as \textit{opaque}, where a robot can obstruct the visibility of two other robots, if it lies on the line segment between them. Under this obstructed visibility, we consider a variant of the separation problem in which robots, starting from any arbitrary initial configuration, are required to separate into concentric semicircles. We present a collision-free algorithm that solves the separation problem under a semi-synchronous scheduler in $O(n)$ epochs, where $n$ is the number of robots. The robots agree on one coordinate axis but have no knowledge of $n$.
>
---
#### [new 092] Clinic-Oriented Feasibility of a Sensor-Fused Wearable for Upper-Limb Function
- **分类: eess.SP; cs.HC; cs.LG; cs.RO; q-bio.NC**

- **简介: 该论文研究可穿戴设备在上肢功能康复中的技术可行性，旨在改善上肢震颤与无力导致的日常活动障碍。通过融合肌电、惯性与力传感器，实现低延迟、安全约束的闭环辅助，验证了其在提升运动质量与任务效率方面的潜力，为后续临床试验奠定基础。**

- **链接: [http://arxiv.org/pdf/2510.22913v1](http://arxiv.org/pdf/2510.22913v1)**

> **作者:** Thanyanee Srichaisak; Arissa Ieochai; Aueaphum Aueawattthanaphisut
>
> **备注:** 19 pages, 7 figures, 5 Tables
>
> **摘要:** Background: Upper-limb weakness and tremor (4--12 Hz) limit activities of daily living (ADL) and reduce adherence to home rehabilitation. Objective: To assess technical feasibility and clinician-relevant signals of a sensor-fused wearable targeting the triceps brachii and extensor pollicis brevis. Methods: A lightweight node integrates surface EMG (1 kHz), IMU (100--200 Hz), and flex/force sensors with on-device INT8 inference (Tiny 1D-CNN/Transformer) and a safety-bounded assist policy (angle/torque/jerk limits; stall/time-out). Healthy adults (n = 12) performed three ADL-like tasks. Primary outcomes: Tremor Index (TI), range of motion (ROM), repetitions (Reps min$^{-1}$). Secondary: EMG median-frequency slope (fatigue trend), closed-loop latency, session completion, and device-related adverse events. Analyses used subject-level paired medians with BCa 95\% CIs; exact Wilcoxon $p$-values are reported in the Results. Results: Assistance was associated with lower tremor prominence and improved task throughput: TI decreased by $-0.092$ (95\% CI [$-0.102$, $-0.079$]), ROM increased by $+12.65\%$ (95\% CI [$+8.43$, $+13.89$]), and Reps rose by $+2.99$ min$^{-1}$ (95\% CI [$+2.61$, $+3.35$]). Median on-device latency was 8.7 ms at a 100 Hz loop rate; all sessions were completed with no device-related adverse events. Conclusions: Multimodal sensing with low-latency, safety-bounded assistance produced improved movement quality (TI $\downarrow$) and throughput (ROM, Reps $\uparrow$) in a pilot technical-feasibility setting, supporting progression to IRB-approved patient studies. Trial registration: Not applicable (pilot non-clinical).
>
---
## 更新

#### [replaced 001] World-Env: Leveraging World Model as a Virtual Environment for VLA Post-Training
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.24948v2](http://arxiv.org/pdf/2509.24948v2)**

> **作者:** Junjin Xiao; Yandan Yang; Xinyuan Chang; Ronghan Chen; Feng Xiong; Mu Xu; Wei-Shi Zheng; Qing Zhang
>
> **摘要:** Vision-Language-Action (VLA) models trained via imitation learning suffer from significant performance degradation in data-scarce scenarios due to their reliance on large-scale demonstration datasets. Although reinforcement learning (RL)-based post-training has proven effective in addressing data scarcity, its application to VLA models is hindered by the non-resettable nature of real-world environments. This limitation is particularly critical in high-risk domains such as industrial automation, where interactions often induce state changes that are costly or infeasible to revert. Furthermore, existing VLA approaches lack a reliable mechanism for detecting task completion, leading to redundant actions that reduce overall task success rates. To address these challenges, we propose World-Env, an RL-based post-training framework that replaces physical interaction with a low-cost, world model-based virtual simulator. World-Env consists of two key components: (1) a video-based world simulator that generates temporally consistent future visual observations, and (2) a vision-language model (VLM)-guided instant reflector that provides continuous reward signals and predicts action termination. This simulated environment enables VLA models to safely explore and generalize beyond their initial imitation learning distribution. Our method achieves notable performance gains with as few as five expert demonstrations per task. Experiments on complex robotic manipulation tasks demonstrate that World-Env effectively overcomes the data inefficiency, safety constraints, and inefficient execution of conventional VLA models that rely on real-world interaction, offering a practical and scalable solution for post-training in resource-constrained settings. Our code is available at https://github.com/junjxiao/world-env.
>
---
#### [replaced 002] D-LIO: 6DoF Direct LiDAR-Inertial Odometry based on Simultaneous Truncated Distance Field Mapping
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.16726v2](http://arxiv.org/pdf/2505.16726v2)**

> **作者:** Lucia Coto-Elena; J. E. Maese; L. Merino; F. Caballero
>
> **备注:** 9 pages, 3 figures and 43 references
>
> **摘要:** This paper presents a new approach for 6DoF Direct LiDAR-Inertial Odometry (D-LIO) based on the simultaneous mapping of truncated distance fields on CPU. Such continuous representation (in the vicinity of the points) enables working with raw 3D LiDAR data online, avoiding the need of LiDAR feature selection and tracking, simplifying the odometry pipeline and easily generalizing to many scenarios. The method is based on the proposed Fast Truncated Distance Field (Fast-TDF) method as a convenient tool to represent the environment. Such representation enables i) solving the LiDAR point-cloud registration as a nonlinear optimization process without the need of selecting/tracking LiDAR features in the input data, ii) simultaneously producing an accurate truncated distance field map of the environment, and iii) updating such map at constant time independently of its size. The approach is tested using open datasets, aerial and ground. It is also benchmarked against other state-of-the-art odometry approaches, demonstrating the same or better level of accuracy with the added value of an online-generated TDF representation of the environment, that can be used for other robotics tasks as planning or collision avoidance. The source code is publicly available at https://anonymous.4open.science/r/D-LIO
>
---
#### [replaced 003] Prognostic Framework for Robotic Manipulators Operating Under Dynamic Task Severities
- **分类: cs.RO; cs.LG; cs.SY; eess.SY; stat.AP**

- **链接: [http://arxiv.org/pdf/2412.00538v3](http://arxiv.org/pdf/2412.00538v3)**

> **作者:** Ayush Mohanty; Jason Dekarske; Stephen K. Robinson; Sanjay Joshi; Nagi Gebraeel
>
> **备注:** Accepted for Publication in IEEE Transactions on Systems, Man, and Cybernetics: Systems
>
> **摘要:** Robotic manipulators are critical in many applications but are known to degrade over time. This degradation is influenced by the nature of the tasks performed by the robot. Tasks with higher severity, such as handling heavy payloads, can accelerate the degradation process. One way this degradation is reflected is in the position accuracy of the robot's end-effector. In this paper, we present a prognostic modeling framework that predicts a robotic manipulator's Remaining Useful Life (RUL) while accounting for the effects of task severity. Our framework represents the robot's position accuracy as a Brownian motion process with a random drift parameter that is influenced by task severity. The dynamic nature of task severity is modeled using a continuous-time Markov chain (CTMC). To evaluate RUL, we discuss two approaches -- (1) a novel closed-form expression for Remaining Lifetime Distribution (RLD), and (2) Monte Carlo simulations, commonly used in prognostics literature. Theoretical results establish the equivalence between these RUL computation approaches. We validate our framework through experiments using two distinct physics-based simulators for planar and spatial robot fleets. Our findings show that robots in both fleets experience shorter RUL when handling a higher proportion of high-severity tasks.
>
---
#### [replaced 004] DDBot: Differentiable Physics-based Digging Robot for Unknown Granular Materials
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.17335v3](http://arxiv.org/pdf/2510.17335v3)**

> **作者:** Xintong Yang; Minglun Wei; Yu-Kun Lai; Ze Ji
>
> **备注:** Accepted as a regular paper by the IEEE Transactions on Robotics
>
> **摘要:** Automating the manipulation of granular materials poses significant challenges due to complex contact dynamics, unpredictable material properties, and intricate system states. Existing approaches often fail to achieve efficiency and accuracy in such tasks. To fill the research gap, this paper studies the small-scale and high-precision granular material digging task with unknown physical properties. A new framework, named differentiable digging robot (DDBot), is proposed to manipulate granular materials, including sand and soil. Specifically, we equip DDBot with a differentiable physics-based simulator, tailored for granular material manipulation, powered by GPU-accelerated parallel computing and automatic differentiation. DDBot can perform efficient differentiable system identification and high-precision digging skill optimisation for unknown granular materials, which is enabled by a differentiable skill-to-action mapping, a task-oriented demonstration method, gradient clipping and line search-based gradient descent. Experimental results show that DDBot can efficiently (converge within 5 to 20 minutes) identify unknown granular material dynamics and optimise digging skills, with high-precision results in zero-shot real-world deployments, highlighting its practicality. Benchmark results against state-of-the-art baselines also confirm the robustness and efficiency of DDBot in such digging tasks.
>
---
#### [replaced 005] Stability Criteria and Motor Performance in Delayed Haptic Dyadic Interactions Mediated by Robots
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2510.14511v2](http://arxiv.org/pdf/2510.14511v2)**

> **作者:** Mingtian Du; Suhas Raghavendra Kulkarni; Simone Kager; Domenico Campolo
>
> **摘要:** This paper establishes analytical stability criteria for robot-mediated human-human (dyadic) interaction systems, focusing on haptic communication under network-induced time delays. Through frequency-domain analysis supported by numerical simulations, we identify both delay-independent and delay-dependent stability criteria. The delay-independent criterion guarantees stability irrespective of the delay, whereas the delay-dependent criterion is characterised by a maximum tolerable delay before instability occurs. The criteria demonstrate dependence on controller and robot dynamic parameters, where increasing stiffness reduces the maximum tolerable delay in a non-linear manner, thereby heightening system vulnerability. The proposed criteria can be generalised to a wide range of robot-mediated interactions and serve as design guidelines for stable remote dyadic systems. Experiments with robots performing human-like movements further illustrate the correlation between stability and motor performance. The findings of this paper suggest the prerequisites for effective delay-compensation strategies.
>
---
#### [replaced 006] Using Non-Expert Data to Robustify Imitation Learning via Offline Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.19495v2](http://arxiv.org/pdf/2510.19495v2)**

> **作者:** Kevin Huang; Rosario Scalise; Cleah Winston; Ayush Agrawal; Yunchu Zhang; Rohan Baijal; Markus Grotz; Byron Boots; Benjamin Burchfiel; Masha Itkina; Paarth Shah; Abhishek Gupta
>
> **摘要:** Imitation learning has proven effective for training robots to perform complex tasks from expert human demonstrations. However, it remains limited by its reliance on high-quality, task-specific data, restricting adaptability to the diverse range of real-world object configurations and scenarios. In contrast, non-expert data -- such as play data, suboptimal demonstrations, partial task completions, or rollouts from suboptimal policies -- can offer broader coverage and lower collection costs. However, conventional imitation learning approaches fail to utilize this data effectively. To address these challenges, we posit that with right design decisions, offline reinforcement learning can be used as a tool to harness non-expert data to enhance the performance of imitation learning policies. We show that while standard offline RL approaches can be ineffective at actually leveraging non-expert data under the sparse data coverage settings typically encountered in the real world, simple algorithmic modifications can allow for the utilization of this data, without significant additional assumptions. Our approach shows that broadening the support of the policy distribution can allow imitation algorithms augmented by offline RL to solve tasks robustly, showing considerably enhanced recovery and generalization behavior. In manipulation tasks, these innovations significantly increase the range of initial conditions where learned policies are successful when non-expert data is incorporated. Moreover, we show that these methods are able to leverage all collected data, including partial or suboptimal demonstrations, to bolster task-directed policy performance. This underscores the importance of algorithmic techniques for using non-expert data for robust policy learning in robotics. Website: https://uwrobotlearning.github.io/RISE-offline/
>
---
#### [replaced 007] MOSAIC: Modular Foundation Models for Assistive and Interactive Cooking
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2402.18796v3](http://arxiv.org/pdf/2402.18796v3)**

> **作者:** Huaxiaoyue Wang; Kushal Kedia; Juntao Ren; Rahma Abdullah; Atiksh Bhardwaj; Angela Chao; Kelly Y Chen; Nathaniel Chin; Prithwish Dan; Xinyi Fan; Gonzalo Gonzalez-Pumariega; Aditya Kompella; Maximus Adrian Pace; Yash Sharma; Xiangwan Sun; Neha Sunkara; Sanjiban Choudhury
>
> **备注:** 22 pages, 13 figures; CoRL 2024
>
> **摘要:** We present MOSAIC, a modular architecture for coordinating multiple robots to (a) interact with users using natural language and (b) manipulate an open vocabulary of everyday objects. MOSAIC employs modularity at several levels: it leverages multiple large-scale pre-trained models for high-level tasks like language and image recognition, while using streamlined modules designed for low-level task-specific control. This decomposition allows us to reap the complementary benefits of foundation models as well as precise, more specialized models. Pieced together, our system is able to scale to complex tasks that involve coordinating multiple robots and humans. First, we unit-test individual modules with 180 episodes of visuomotor picking, 60 episodes of human motion forecasting, and 46 online user evaluations of the task planner. We then extensively evaluate MOSAIC with 60 end-to-end trials. We discuss crucial design decisions, limitations of the current system, and open challenges in this domain. The project's website is at https://portal-cornell.github.io/MOSAIC/
>
---
#### [replaced 008] Controllable Collision Scenario Generation via Collision Pattern Prediction
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2510.12206v2](http://arxiv.org/pdf/2510.12206v2)**

> **作者:** Pin-Lun Chen; Chi-Hsi Kung; Che-Han Chang; Wei-Chen Chiu; Yi-Ting Chen
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Evaluating the safety of autonomous vehicles (AVs) requires diverse, safety-critical scenarios, with collisions being especially important yet rare and unsafe to collect in the real world. Therefore, the community has been focusing on generating safety-critical scenarios in simulation. However, controlling attributes such as collision type and time-to-accident (TTA) remains challenging. We introduce a new task called controllable collision scenario generation, where the goal is to produce trajectories that realize a user-specified collision type and TTA, to investigate the feasibility of automatically generating desired collision scenarios. To support this task, we present COLLIDE, a large-scale collision scenario dataset constructed by transforming real-world driving logs into diverse collisions, balanced across five representative collision types and different TTA intervals. We propose a framework that predicts Collision Pattern, a compact and interpretable representation that captures the spatial configuration of the ego and the adversarial vehicles at impact, before rolling out full adversarial trajectories. Experiments show that our approach outperforms strong baselines in both collision rate and controllability. Furthermore, generated scenarios consistently induce higher planner failure rates, revealing limitations of existing planners. We demonstrate that these scenarios fine-tune planners for robustness improvements, contributing to safer AV deployment in different collision scenarios. Project page is available at https://submit-user.github.io/anon2025
>
---
#### [replaced 009] A Single Motor Nano Aerial Vehicle with Novel Peer-to-Peer Communication and Sensing Mechanism
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2405.14144v3](http://arxiv.org/pdf/2405.14144v3)**

> **作者:** Jingxian Wang; Andrew G. Curtis; Mark Yim; Michael Rubenstein
>
> **摘要:** Communication and position sensing are among the most important capabilities for swarm robots to interact with their peers and perform tasks collaboratively. However, the hardware required to facilitate communication and position sensing is often too complicated, expensive, and bulky to be carried on swarm robots. Here we present Maneuverable Piccolissimo 3 (MP3), a minimalist, single motor drone capable of executing inter-robot communication via infrared light and triangulation-based sensing of relative bearing, distance, and elevation using message arrival time. Thanks to its novel design, MP3 can communicate with peers and localize itself using simple components, keeping its size and mass small and making it inherently safe for human interaction. We present the hardware and software design of MP3 and demonstrate its capability to localize itself, fly stably, and maneuver in the environment using peer-to-peer communication and sensing.
>
---
#### [replaced 010] Pretraining a Unified PDDL Domain from Real-World Demonstrations for Generalizable Robot Task Planning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.21545v2](http://arxiv.org/pdf/2507.21545v2)**

> **作者:** Haoming Ye; Yunxiao Xiao; Cewu Lu; Panpan Cai
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Robotic task planning in real-world environments requires reasoning over implicit constraints from language and vision. While LLMs and VLMs offer strong priors, they struggle with long-horizon structure and symbolic grounding. Existing methods that combine LLMs with symbolic planning often rely on handcrafted or narrow domains, limiting generalization. We propose UniDomain, a framework that pre-trains a PDDL domain from robot manipulation demonstrations and applies it for online robotic task planning. It extracts atomic domains from 12,393 manipulation videos to form a unified domain with 3137 operators, 2875 predicates, and 16481 causal edges. Given a target class of tasks, it retrieves relevant atomics from the unified domain and systematically fuses them into high-quality meta-domains to support compositional generalization in planning. Experiments on diverse real-world tasks show that UniDomain solves complex, unseen tasks in a zero-shot manner, achieving up to 58% higher task success and 160% improvement in plan optimality over state-of-the-art LLM and LLM-PDDL baselines.
>
---
#### [replaced 011] DERD-Net: Learning Depth from Event-based Ray Densities
- **分类: cs.CV; cs.LG; cs.RO; eess.SP**

- **链接: [http://arxiv.org/pdf/2504.15863v2](http://arxiv.org/pdf/2504.15863v2)**

> **作者:** Diego Hitzges; Suman Ghosh; Guillermo Gallego
>
> **备注:** 17 pages, 3 figures, 15 tables. Project page: https://github.com/tub-rip/DERD-Net. 39th Conference on Neural Information Processing Systems (NeurIPS), San Diego, 2025
>
> **摘要:** Event cameras offer a promising avenue for multi-view stereo depth estimation and Simultaneous Localization And Mapping (SLAM) due to their ability to detect blur-free 3D edges at high-speed and over broad illumination conditions. However, traditional deep learning frameworks designed for conventional cameras struggle with the asynchronous, stream-like nature of event data, as their architectures are optimized for discrete, image-like inputs. We propose a scalable, flexible and adaptable framework for pixel-wise depth estimation with event cameras in both monocular and stereo setups. The 3D scene structure is encoded into disparity space images (DSIs), representing spatial densities of rays obtained by back-projecting events into space via known camera poses. Our neural network processes local subregions of the DSIs combining 3D convolutions and a recurrent structure to recognize valuable patterns for depth prediction. Local processing enables fast inference with full parallelization and ensures constant ultra-low model complexity and memory costs, regardless of camera resolution. Experiments on standard benchmarks (MVSEC and DSEC datasets) demonstrate unprecedented effectiveness: (i) using purely monocular data, our method achieves comparable results to existing stereo methods; (ii) when applied to stereo data, it strongly outperforms all state-of-the-art (SOTA) approaches, reducing the mean absolute error by at least 42%; (iii) our method also allows for increases in depth completeness by more than 3-fold while still yielding a reduction in median absolute error of at least 30%. Given its remarkable performance and effective processing of event-data, our framework holds strong potential to become a standard approach for using deep learning for event-based depth estimation and SLAM. Project page: https://github.com/tub-rip/DERD-Net
>
---
#### [replaced 012] Zero-Shot Trajectory Planning for Signal Temporal Logic Tasks
- **分类: cs.RO; cs.AI; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2501.13457v2](http://arxiv.org/pdf/2501.13457v2)**

> **作者:** Ruijia Liu; Ancheng Hou; Xiao Yu; Xiang Yin
>
> **摘要:** Signal Temporal Logic (STL) is a powerful specification language for describing complex temporal behaviors of continuous signals, making it well-suited for high-level robotic task descriptions. However, generating executable plans for STL tasks is challenging, as it requires consideration of the coupling between the task specification and the system dynamics. Existing approaches either follow a model-based setting that explicitly requires knowledge of the system dynamics or adopt a task-oriented data-driven approach to learn plans for specific tasks. In this work, we address the problem of generating executable STL plans for systems with unknown dynamics. We propose a hierarchical planning framework that enables zero-shot generalization to new STL tasks by leveraging only task-agnostic trajectory data during offline training. The framework consists of three key components: (i) decomposing the STL specification into several progresses and time constraints, (ii) searching for timed waypoints that satisfy all progresses under time constraints, and (iii) generating trajectory segments using a pre-trained diffusion model and stitching them into complete trajectories. We formally prove that our method guarantees STL satisfaction, and simulation results demonstrate its effectiveness in generating dynamically feasible trajectories across diverse long-horizon STL tasks.
>
---
#### [replaced 013] Adversarial Locomotion and Motion Imitation for Humanoid Policy Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.14305v3](http://arxiv.org/pdf/2504.14305v3)**

> **作者:** Jiyuan Shi; Xinzhe Liu; Dewei Wang; Ouyang Lu; Sören Schwertfeger; Chi Zhang; Fuchun Sun; Chenjia Bai; Xuelong Li
>
> **备注:** NeurIPS 2025. Code: https://github.com/TeleHuman/ALMI-Open, Dataset: https://huggingface.co/datasets/TeleEmbodied/ALMI-X
>
> **摘要:** Humans exhibit diverse and expressive whole-body movements. However, attaining human-like whole-body coordination in humanoid robots remains challenging, as conventional approaches that mimic whole-body motions often neglect the distinct roles of upper and lower body. This oversight leads to computationally intensive policy learning and frequently causes robot instability and falls during real-world execution. To address these issues, we propose Adversarial Locomotion and Motion Imitation (ALMI), a novel framework that enables adversarial policy learning between upper and lower body. Specifically, the lower body aims to provide robust locomotion capabilities to follow velocity commands while the upper body tracks various motions. Conversely, the upper-body policy ensures effective motion tracking when the robot executes velocity-based movements. Through iterative updates, these policies achieve coordinated whole-body control, which can be extended to loco-manipulation tasks with teleoperation systems. Extensive experiments demonstrate that our method achieves robust locomotion and precise motion tracking in both simulation and on the full-size Unitree H1 robot. Additionally, we release a large-scale whole-body motion control dataset featuring high-quality episodic trajectories from MuJoCo simulations deployable on real robots. The project page is https://almi-humanoid.github.io.
>
---
#### [replaced 014] Online POMDP Planning with Anytime Deterministic Optimality Guarantees
- **分类: cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2310.01791v5](http://arxiv.org/pdf/2310.01791v5)**

> **作者:** Moran Barenboim; Vadim Indelman
>
> **摘要:** Decision-making under uncertainty is a critical aspect of many practical autonomous systems due to incomplete information. Partially Observable Markov Decision Processes (POMDPs) offer a mathematically principled framework for formulating decision-making problems under such conditions. However, finding an optimal solution for a POMDP is generally intractable. In recent years, there has been a significant progress of scaling approximate solvers from small to moderately sized problems, using online tree search solvers. Often, such approximate solvers are limited to probabilistic or asymptotic guarantees towards the optimal solution. In this paper, we derive a deterministic relationship for discrete POMDPs between an approximated and the optimal solution. We show that at any time, we can derive bounds that relate between the existing solution and the optimal one. We show that our derivations provide an avenue for a new set of algorithms and can be attached to existing algorithms that have a certain structure to provide them with deterministic guarantees with marginal computational overhead. In return, not only do we certify the solution quality, but we demonstrate that making a decision based on the deterministic guarantee may result in superior performance compared to the original algorithm without the deterministic certification.
>
---
#### [replaced 015] Lazy-DaSH: Lazy Approach for Hypergraph-based Multi-robot Task and Motion Planning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.05552v2](http://arxiv.org/pdf/2504.05552v2)**

> **作者:** Seongwon Lee; James Motes; Isaac Ngui; Marco Morales; Nancy M. Amato
>
> **摘要:** We introduce Lazy-DaSH, an improvement over the recent state of the art multi-robot task and motion planning method DaSH, which scales to more than double the number of robots and objects compared to the original method and achieves an order of magnitude faster planning time when applied to a multi-manipulator object rearrangement problem. We achieve this improvement through a hierarchical approach, where a high-level task planning layer identifies planning spaces required for task completion, and motion feasibility is validated lazily only within these spaces. In contrast, DaSH precomputes the motion feasibility of all possible actions, resulting in higher costs for constructing state space representations. Lazy-DaSH maintains efficient query performance by utilizing a constraint feedback mechanism within its hierarchical structure, ensuring that motion feasibility is effectively conveyed to the query process. By maintaining smaller state space representations, our method significantly reduces both representation construction time and query time. We evaluate Lazy-DaSH in four distinct scenarios, demonstrating its scalability to increasing numbers of robots and objects, as well as its adaptability in resolving conflicts through the constraint feedback mechanism.
>
---
#### [replaced 016] A Cycle Ride to HDR: Semantics Aware Self-Supervised Framework for Unpaired LDR-to-HDR Image Reconstruction
- **分类: cs.CV; cs.AI; cs.GR; cs.LG; cs.RO; Artificial intelligence, Computer vision, Machine learning, Deep
  learning; I.3.3; I.4.5**

- **链接: [http://arxiv.org/pdf/2410.15068v4](http://arxiv.org/pdf/2410.15068v4)**

> **作者:** Hrishav Bakul Barua; Kalin Stefanov; Lemuel Lai En Che; Abhinav Dhall; KokSheik Wong; Ganesh Krishnasamy
>
> **摘要:** Reconstruction of High Dynamic Range (HDR) from Low Dynamic Range (LDR) images is an important computer vision task. There is a significant amount of research utilizing both conventional non-learning methods and modern data-driven approaches, focusing on using both single-exposed and multi-exposed LDR for HDR image reconstruction. However, most current state-of-the-art methods require high-quality paired {LDR;HDR} datasets with limited literature use of unpaired datasets, that is, methods that learn the LDR-HDR mapping between domains. This paper proposes CycleHDR, a method that integrates self-supervision into a modified semantic- and cycle-consistent adversarial architecture that utilizes unpaired LDR and HDR datasets for training. Our method introduces novel artifact- and exposure-aware generators to address visual artifact removal. It also puts forward an encoder and loss to address semantic consistency, another under-explored topic. CycleHDR is the first to use semantic and contextual awareness for the LDR-HDR reconstruction task in a self-supervised setup. The method achieves state-of-the-art performance across several benchmark datasets and reconstructs high-quality HDR images. The official website of this work is available at: https://github.com/HrishavBakulBarua/Cycle-HDR
>
---
#### [replaced 017] RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.04308v3](http://arxiv.org/pdf/2506.04308v3)**

> **作者:** Enshen Zhou; Jingkun An; Cheng Chi; Yi Han; Shanyu Rong; Chi Zhang; Pengwei Wang; Zhongyuan Wang; Tiejun Huang; Lu Sheng; Shanghang Zhang
>
> **备注:** Accepted by NeurIPS 2025. Project page: https://zhoues.github.io/RoboRefer/
>
> **摘要:** Spatial referring is a fundamental capability of embodied robots to interact with the 3D physical world. However, even with the powerful pretrained vision language models (VLMs), recent approaches are still not qualified to accurately understand the complex 3D scenes and dynamically reason about the instruction-indicated locations for interaction. To this end, we propose RoboRefer, a 3D-aware VLM that can first achieve precise spatial understanding by integrating a disentangled but dedicated depth encoder via supervised fine-tuning (SFT). Moreover, RoboRefer advances generalized multi-step spatial reasoning via reinforcement fine-tuning (RFT), with metric-sensitive process reward functions tailored for spatial referring tasks. To support SFT and RFT training, we introduce RefSpatial, a large-scale dataset of 20M QA pairs (2x prior), covering 31 spatial relations (vs. 15 prior) and supporting complex reasoning processes (up to 5 steps). In addition, we introduce RefSpatial-Bench, a challenging benchmark filling the gap in evaluating spatial referring with multi-step reasoning. Experiments show that SFT-trained RoboRefer achieves state-of-the-art spatial understanding, with an average success rate of 89.6%. RFT-trained RoboRefer further outperforms all other baselines by a large margin, even surpassing Gemini-2.5-Pro by 17.4% in average accuracy on RefSpatial-Bench. Notably, RoboRefer can be integrated with various control policies to execute long-horizon, dynamic tasks across diverse robots (e,g., UR5, G1 humanoid) in cluttered real-world scenes. Please see the project page at https://zhoues.github.io/RoboRefer.
>
---
#### [replaced 018] Correspondence-Free, Function-Based Sim-to-Real Learning for Deformable Surface Control
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.00060v2](http://arxiv.org/pdf/2509.00060v2)**

> **作者:** Yingjun Tian; Guoxin Fang; Renbo Su; Aoran Lyu; Neelotpal Dutta; Weiming Wang; Simeon Gill; Andrew Weightman; Charlie C. L. Wang
>
> **摘要:** This paper presents a correspondence-free, function-based sim-to-real learning method for controlling deformable freeform surfaces. Unlike traditional sim-to-real transfer methods that strongly rely on marker points with full correspondences, our approach simultaneously learns a deformation function space and a confidence map -- both parameterized by a neural network -- to map simulated shapes to their real-world counterparts. As a result, the sim-to-real learning can be conducted by input from either a 3D scanner as point clouds (without correspondences) or a motion capture system as marker points (tolerating missed markers). The resultant sim-to-real transfer can be seamlessly integrated into a neural network-based computational pipeline for inverse kinematics and shape control. We demonstrate the versatility and adaptability of our method on both vision devices and across four pneumatically actuated soft robots: a deformable membrane, a robotic mannequin, and two soft manipulators.
>
---
#### [replaced 019] SceneComplete: Open-World 3D Scene Completion in Cluttered Real World Environments for Robot Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.23643v4](http://arxiv.org/pdf/2410.23643v4)**

> **作者:** Aditya Agarwal; Gaurav Singh; Bipasha Sen; Tomás Lozano-Pérez; Leslie Pack Kaelbling
>
> **摘要:** Careful robot manipulation in every-day cluttered environments requires an accurate understanding of the 3D scene, in order to grasp and place objects stably and reliably and to avoid colliding with other objects. In general, we must construct such a 3D interpretation of a complex scene based on limited input, such as a single RGB-D image. We describe SceneComplete, a system for constructing a complete, segmented, 3D model of a scene from a single view. SceneComplete is a novel pipeline for composing general-purpose pretrained perception modules (vision-language, segmentation, image-inpainting, image-to-3D, visual-descriptors and pose-estimation) to obtain highly accurate results. We demonstrate its accuracy and effectiveness with respect to ground-truth models in a large benchmark dataset and show that its accurate whole-object reconstruction enables robust grasp proposal generation, including for a dexterous hand. We release the code and additional results on our website.
>
---
#### [replaced 020] On the Importance of Tactile Sensing for Imitation Learning: A Case Study on Robotic Match Lighting
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.13618v3](http://arxiv.org/pdf/2504.13618v3)**

> **作者:** Niklas Funk; Changqi Chen; Tim Schneider; Georgia Chalvatzaki; Roberto Calandra; Jan Peters
>
> **摘要:** The field of robotic manipulation has advanced significantly in recent years. At the sensing level, several novel tactile sensors have been developed, capable of providing accurate contact information. On a methodological level, learning from demonstrations has proven an efficient paradigm to obtain performant robotic manipulation policies. The combination of both holds the promise to extract crucial contact-related information from the demonstration data and actively exploit it during policy rollouts. However, this integration has so far been underexplored, most notably in dynamic, contact-rich manipulation tasks where precision and reactivity are essential. This work therefore proposes a multimodal, visuotactile imitation learning framework that integrates a modular transformer architecture with a flow-based generative model, enabling efficient learning of fast and dexterous manipulation policies. We evaluate our framework on the dynamic, contact-rich task of robotic match lighting - a task in which tactile feedback influences human manipulation performance. The experimental results highlight the effectiveness of our approach and show that adding tactile information improves policy performance, thereby underlining their combined potential for learning dynamic manipulation from few demonstrations. Project website: https://sites.google.com/view/tactile-il .
>
---
#### [replaced 021] GOPLA: Generalizable Object Placement Learning via Synthetic Augmentation of Human Arrangement
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.14627v2](http://arxiv.org/pdf/2510.14627v2)**

> **作者:** Yao Zhong; Hanzhi Chen; Simon Schaefer; Anran Zhang; Stefan Leutenegger
>
> **摘要:** Robots are expected to serve as intelligent assistants, helping humans with everyday household organization. A central challenge in this setting is the task of object placement, which requires reasoning about both semantic preferences (e.g., common-sense object relations) and geometric feasibility (e.g., collision avoidance). We present GOPLA, a hierarchical framework that learns generalizable object placement from augmented human demonstrations. A multi-modal large language model translates human instructions and visual inputs into structured plans that specify pairwise object relationships. These plans are then converted into 3D affordance maps with geometric common sense by a spatial mapper, while a diffusion-based planner generates placement poses guided by test-time costs, considering multi-plan distributions and collision avoidance. To overcome data scarcity, we introduce a scalable pipeline that expands human placement demonstrations into diverse synthetic training data. Extensive experiments show that our approach improves placement success rates by 30.04 percentage points over the runner-up, evaluated on positioning accuracy and physical plausibility, demonstrating strong generalization across a wide range of real-world robotic placement scenarios.
>
---
#### [replaced 022] CCDP: Composition of Conditional Diffusion Policies with Guided Sampling
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.15386v3](http://arxiv.org/pdf/2503.15386v3)**

> **作者:** Amirreza Razmjoo; Sylvain Calinon; Michael Gienger; Fan Zhang
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** Imitation Learning offers a promising approach to learn directly from data without requiring explicit models, simulations, or detailed task definitions. During inference, actions are sampled from the learned distribution and executed on the robot. However, sampled actions may fail for various reasons, and simply repeating the sampling step until a successful action is obtained can be inefficient. In this work, we propose an enhanced sampling strategy that refines the sampling distribution to avoid previously unsuccessful actions. We demonstrate that by solely utilizing data from successful demonstrations, our method can infer recovery actions without the need for additional exploratory behavior or a high-level controller. Furthermore, we leverage the concept of diffusion model decomposition to break down the primary problem, which may require long-horizon history to manage failures, into multiple smaller, more manageable sub-problems in learning, data collection, and inference, thereby enabling the system to adapt to variable failure counts. Our approach yields a low-level controller that dynamically adjusts its sampling space to improve efficiency when prior samples fall short. We validate our method across several tasks, including door opening with unknown directions, object manipulation, and button-searching scenarios, demonstrating that our approach outperforms traditional baselines.
>
---
#### [replaced 023] Real-Time Knee Angle Prediction Using EMG and Kinematic Data with an Attention-Based CNN-LSTM Network and Transfer Learning Across Multiple Datasets
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.13443v2](http://arxiv.org/pdf/2510.13443v2)**

> **作者:** Mojtaba Mollahossein; Gholamreza Vossoughi; Mohammad Hossein Rohban
>
> **摘要:** Electromyography (EMG) signals are widely used for predicting body joint angles through machine learning (ML) and deep learning (DL) methods. However, these approaches often face challenges such as limited real-time applicability, non-representative test conditions, and the need for large datasets to achieve optimal performance. This paper presents a transfer-learning framework for knee joint angle prediction that requires only a few gait cycles from new subjects. Three datasets - Georgia Tech, the University of California Irvine (UCI), and the Sharif Mechatronic Lab Exoskeleton (SMLE) - containing four EMG channels relevant to knee motion were utilized. A lightweight attention-based CNN-LSTM model was developed and pre-trained on the Georgia Tech dataset, then transferred to the UCI and SMLE datasets. The proposed model achieved Normalized Mean Absolute Errors (NMAE) of 6.8 percent and 13.7 percent for one-step and 50-step predictions on abnormal subjects using EMG inputs alone. Incorporating historical knee angles reduced the NMAE to 3.1 percent and 3.5 percent for normal subjects, and to 2.8 percent and 7.5 percent for abnormal subjects. When further adapted to the SMLE exoskeleton with EMG, kinematic, and interaction force inputs, the model achieved 1.09 percent and 3.1 percent NMAE for one- and 50-step predictions, respectively. These results demonstrate robust performance and strong generalization for both short- and long-term rehabilitation scenarios.
>
---
#### [replaced 024] DexSinGrasp: Learning a Unified Policy for Dexterous Object Singulation and Grasping in Densely Cluttered Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.04516v3](http://arxiv.org/pdf/2504.04516v3)**

> **作者:** Lixin Xu; Zixuan Liu; Zhewei Gui; Jingxiang Guo; Zeyu Jiang; Tongzhou Zhang; Zhixuan Xu; Chongkai Gao; Lin Shao
>
> **摘要:** Grasping objects in cluttered environments remains a fundamental yet challenging problem in robotic manipulation. While prior works have explored learning-based synergies between pushing and grasping for two-fingered grippers, few have leveraged the high degrees of freedom (DoF) in dexterous hands to perform efficient singulation for grasping in cluttered settings. In this work, we introduce DexSinGrasp, a unified policy for dexterous object singulation and grasping. DexSinGrasp enables high-dexterity object singulation to facilitate grasping, significantly improving efficiency and effectiveness in cluttered environments. We incorporate clutter arrangement curriculum learning to enhance success rates and generalization across diverse clutter conditions, while policy distillation enables a deployable vision-based grasping strategy. To evaluate our approach, we introduce a set of cluttered grasping tasks with varying object arrangements and occlusion levels. Experimental results show that our method outperforms baselines in both efficiency and grasping success rate, particularly in dense clutter. Codes, appendix, and videos are available on our website https://nus-lins-lab.github.io/dexsingweb/.
>
---
#### [replaced 025] SAC Flow: Sample-Efficient Reinforcement Learning of Flow-Based Policies via Velocity-Reparameterized Sequential Modeling
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.25756v2](http://arxiv.org/pdf/2509.25756v2)**

> **作者:** Yixian Zhang; Shu'ang Yu; Tonghe Zhang; Mo Guang; Haojia Hui; Kaiwen Long; Yu Wang; Chao Yu; Wenbo Ding
>
> **摘要:** Training expressive flow-based policies with off-policy reinforcement learning is notoriously unstable due to gradient pathologies in the multi-step action sampling process. We trace this instability to a fundamental connection: the flow rollout is algebraically equivalent to a residual recurrent computation, making it susceptible to the same vanishing and exploding gradients as RNNs. To address this, we reparameterize the velocity network using principles from modern sequential models, introducing two stable architectures: Flow-G, which incorporates a gated velocity, and Flow-T, which utilizes a decoded velocity. We then develop a practical SAC-based algorithm, enabled by a noise-augmented rollout, that facilitates direct end-to-end training of these policies. Our approach supports both from-scratch and offline-to-online learning and achieves state-of-the-art performance on continuous control and robotic manipulation benchmarks, eliminating the need for common workarounds like policy distillation or surrogate objectives.
>
---
#### [replaced 026] Raw2Drive: Reinforcement Learning with Aligned World Models for End-to-End Autonomous Driving (in CARLA v2)
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2505.16394v2](http://arxiv.org/pdf/2505.16394v2)**

> **作者:** Zhenjie Yang; Xiaosong Jia; Qifeng Li; Xue Yang; Maoqing Yao; Junchi Yan
>
> **备注:** Accepted by NeurIPS 2025
>
> **摘要:** Reinforcement Learning (RL) can mitigate the causal confusion and distribution shift inherent to imitation learning (IL). However, applying RL to end-to-end autonomous driving (E2E-AD) remains an open problem for its training difficulty, and IL is still the mainstream paradigm in both academia and industry. Recently Model-based Reinforcement Learning (MBRL) have demonstrated promising results in neural planning; however, these methods typically require privileged information as input rather than raw sensor data. We fill this gap by designing Raw2Drive, a dual-stream MBRL approach. Initially, we efficiently train an auxiliary privileged world model paired with a neural planner that uses privileged information as input. Subsequently, we introduce a raw sensor world model trained via our proposed Guidance Mechanism, which ensures consistency between the raw sensor world model and the privileged world model during rollouts. Finally, the raw sensor world model combines the prior knowledge embedded in the heads of the privileged world model to effectively guide the training of the raw sensor policy. Raw2Drive is so far the only RL based end-to-end method on CARLA Leaderboard 2.0, and Bench2Drive and it achieves state-of-the-art performance.
>
---
#### [replaced 027] CIVIL: Causal and Intuitive Visual Imitation Learning
- **分类: cs.RO; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2504.17959v3](http://arxiv.org/pdf/2504.17959v3)**

> **作者:** Yinlong Dai; Robert Ramirez Sanchez; Ryan Jeronimus; Shahabedin Sagheb; Cara M. Nunez; Heramb Nemlekar; Dylan P. Losey
>
> **摘要:** Today's robots attempt to learn new tasks by imitating human examples. These robots watch the human complete the task, and then try to match the actions taken by the human expert. However, this standard approach to visual imitation learning is fundamentally limited: the robot observes what the human does, but not why the human chooses those behaviors. Without understanding which features of the system or environment factor into the human's decisions, robot learners often misinterpret the human's examples. In practice, this results in causal confusion, inefficient learning, and robot policies that fail when the environment changes. We therefore propose a shift in perspective: instead of asking human teachers just to show what actions the robot should take, we also enable humans to intuitively indicate why they made those decisions. Under our paradigm human teachers attach markers to task-relevant objects and use natural language prompts to describe their state representation. Our proposed algorithm, CIVIL, leverages this augmented demonstration data to filter the robot's visual observations and extract a feature representation that aligns with the human teacher. CIVIL then applies these causal features to train a transformer-based policy that -- when tested on the robot -- is able to emulate human behaviors without being confused by visual distractors or irrelevant items. Our simulations and real-world experiments demonstrate that robots trained with CIVIL learn both what actions to take and why to take those actions, resulting in better performance than state-of-the-art baselines. From the human's perspective, our user study reveals that this new training paradigm actually reduces the total time required for the robot to learn the task, and also improves the robot's performance in previously unseen scenarios. See videos at our project website: https://civil2025.github.io
>
---
#### [replaced 028] HAND Me the Data: Fast Robot Adaptation via Hand Path Retrieval
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.20455v4](http://arxiv.org/pdf/2505.20455v4)**

> **作者:** Matthew Hong; Anthony Liang; Kevin Kim; Harshitha Rajaprakash; Jesse Thomason; Erdem Bıyık; Jesse Zhang
>
> **摘要:** We hand the community HAND, a simple and time-efficient method for teaching robots new manipulation tasks through human hand demonstrations. Instead of relying on task-specific robot demonstrations collected via teleoperation, HAND uses easy-to-provide hand demonstrations to retrieve relevant behaviors from task-agnostic robot play data. Using a visual tracking pipeline, HAND extracts the motion of the human hand from the hand demonstration and retrieves robot sub-trajectories in two stages: first filtering by visual similarity, then retrieving trajectories with similar behaviors to the hand. Fine-tuning a policy on the retrieved data enables real-time learning of tasks in under four minutes, without requiring calibrated cameras or detailed hand pose estimation. Experiments also show that HAND outperforms retrieval baselines by over 2x in average task success rates on real robots. Videos can be found at our project website: https://liralab.usc.edu/handretrieval/.
>
---
#### [replaced 029] ASC-SW: Atrous strip convolution network with sliding windows
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.12744v3](http://arxiv.org/pdf/2507.12744v3)**

> **作者:** Cheng Liu; Fan Zhu; Yifeng Xu; Baoru Huang; Mohd Rizal Arshad
>
> **备注:** The data of model comparsion in chapter 4 need to be modified
>
> **摘要:** With the rapid development of lightweight visual neural network architectures, traditional high-performance vision models have undergone significant compression, enhancing their computational and energy efficiency and enabling deployment on resource-constrained edge devices. In order to enable the mobile robot to avoid the ground wires, we propose a visual-assisted navigation framework called Atrous Strip Convolution Sliding Window (ASC-SW). This framework compensates for the limitations of traditional light detection and range (LiDAR) sensors to detect ground-level obstacles such as wires. A lightweight and efficient segmentation model, Atrous Strip Convolution Network (ASCnet) was proposed, for detecting deformable linear objects (DLOs). Atrous Strip Convolution Spatial Pyramid Pooling (ASCSPP) is designed to extract DLOs features effectively. Atrous Strip Convolution is integrated into ASCSPP to accurately identify the linear structure of DLOs with low computational cost. Additionally, a Sliding Window (SW) post processing module is proposed to denoise the output in complex environments, improving recognition accuracy. ASC-SW achieves 75.3% MIoU at 217 FPS on a self-built real world dataset and real-robot experiment was demonstrated that our proposed framework. It can be successfully verified on the real-robot on the edge device(Jetson platform) at that were originally inoperable.
>
---
#### [replaced 030] Robust Understanding of Human-Robot Social Interactions through Multimodal Distillation
- **分类: cs.RO; cs.HC**

- **链接: [http://arxiv.org/pdf/2505.06278v2](http://arxiv.org/pdf/2505.06278v2)**

> **作者:** Tongfei Bian; Mathieu Chollet; Tanaya Guha
>
> **备注:** Accepted by ACM Multimedia 2025, camera-ready version
>
> **摘要:** There is a growing need for social robots and intelligent agents that can effectively interact with and support users. For the interactions to be seamless, the agents need to analyse social scenes and behavioural cues from their (robot's) perspective. Works that model human-agent interactions in social situations are few; and even those existing ones are computationally too intensive to be deployed in real time or perform poorly in real-world scenarios when only limited information is available. We propose a knowledge distillation framework that models social interactions through various multimodal cues, and yet is robust against incomplete and noisy information during inference. We train a teacher model with multimodal input (body, face and hand gestures, gaze, raw images) that transfers knowledge to a student model which relies solely on body pose. Extensive experiments on two publicly available human-robot interaction datasets demonstrate that our student model achieves an average accuracy gain of 14.75% over competitive baselines on multiple downstream social understanding tasks, even with up to 51% of its input being corrupted. The student model is also highly efficient - less than 1% in size of the teacher model in terms of parameters and its latency is 11.9% of the teacher model. Our code and related data are available at github.com/biantongfei/SocialEgoMobile.
>
---
#### [replaced 031] Soft and Compliant Contact-Rich Hair Manipulation and Care
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2501.02630v2](http://arxiv.org/pdf/2501.02630v2)**

> **作者:** Uksang Yoo; Nathaniel Dennler; Eliot Xing; Maja Matarić; Stefanos Nikolaidis; Jeffrey Ichnowski; Jean Oh
>
> **摘要:** Hair care robots can help address labor shortages in elderly care while enabling those with limited mobility to maintain their hair-related identity. We present MOE-Hair, a soft robot system that performs three hair-care tasks: head patting, finger combing, and hair grasping. The system features a tendon-driven soft robot end-effector (MOE) with a wrist-mounted RGBD camera, leveraging both mechanical compliance for safety and visual force sensing through deformation. In testing with a force-sensorized mannequin head, MOE achieved comparable hair-grasping effectiveness while applying significantly less force than rigid grippers. Our novel force estimation method combines visual deformation data and tendon tensions from actuators to infer applied forces, reducing sensing errors by up to 60.1% and 20.3% compared to actuator current load-only and depth image-only baselines, respectively. A user study with 12 participants demonstrated statistically significant preferences for MOE-Hair over a baseline system in terms of comfort, effectiveness, and appropriate force application. These results demonstrate the unique advantages of soft robots in contact-rich hair-care tasks, while highlighting the importance of precise force control despite the inherent compliance of the system.
>
---
#### [replaced 032] KungfuBot: Physics-Based Humanoid Whole-Body Control for Learning Highly-Dynamic Skills
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.12851v2](http://arxiv.org/pdf/2506.12851v2)**

> **作者:** Weiji Xie; Jinrui Han; Jiakun Zheng; Huanyu Li; Xinzhe Liu; Jiyuan Shi; Weinan Zhang; Chenjia Bai; Xuelong Li
>
> **备注:** NeurIPS 2025. Project Page: https://kungfu-bot.github.io/
>
> **摘要:** Humanoid robots are promising to acquire various skills by imitating human behaviors. However, existing algorithms are only capable of tracking smooth, low-speed human motions, even with delicate reward and curriculum design. This paper presents a physics-based humanoid control framework, aiming to master highly-dynamic human behaviors such as Kungfu and dancing through multi-steps motion processing and adaptive motion tracking. For motion processing, we design a pipeline to extract, filter out, correct, and retarget motions, while ensuring compliance with physical constraints to the maximum extent. For motion imitation, we formulate a bi-level optimization problem to dynamically adjust the tracking accuracy tolerance based on the current tracking error, creating an adaptive curriculum mechanism. We further construct an asymmetric actor-critic framework for policy training. In experiments, we train whole-body control policies to imitate a set of highly-dynamic motions. Our method achieves significantly lower tracking errors than existing approaches and is successfully deployed on the Unitree G1 robot, demonstrating stable and expressive behaviors. The project page is https://kungfu-bot.github.io.
>
---
#### [replaced 033] Steering Flexible Linear Objects in Planar Environments by Two Robot Hands Using Euler's Elastica Solutions
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2501.02874v2](http://arxiv.org/pdf/2501.02874v2)**

> **作者:** Aharon Levin; Elon Rimon; Amir Shapiro
>
> **摘要:** The manipulation of flexible objects such as cables, wires and fresh food items by robot hands forms a special challenge in robot grasp mechanics. This paper considers the steering of flexible linear objects in planar environments by two robot hands. The flexible linear object, modeled as an elastic non-stretchable rod, is manipulated by varying the gripping endpoint positions while keeping equal endpoint tangents. The flexible linear object shape has a closed form solution in terms of the grasp endpoint positions and tangents, called Euler's elastica. This paper obtains the elastica solutions under the optimal control framework, then uses the elastica solutions to obtain closed-form criteria for non self-intersection, stability and obstacle avoidance of the flexible linear object. The new tools are incorporated into a planning scheme for steering flexible linear objects in planar environments populated by sparsely spaced obstacles. The scheme is fully implemented and demonstrated with detailed examples.
>
---
#### [replaced 034] Hierarchical Language Models for Semantic Navigation and Manipulation in an Aerial-Ground Robotic System
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.05020v3](http://arxiv.org/pdf/2506.05020v3)**

> **作者:** Haokun Liu; Zhaoqi Ma; Yunong Li; Junichiro Sugihara; Yicheng Chen; Jinjie Li; Moju Zhao
>
> **备注:** 18 pages, 10 figures
>
> **摘要:** Heterogeneous multirobot systems show great potential in complex tasks requiring coordinated hybrid cooperation. However, existing methods that rely on static or task-specific models often lack generalizability across diverse tasks and dynamic environments. This highlights the need for generalizable intelligence that can bridge high-level reasoning with low-level execution across heterogeneous agents. To address this, we propose a hierarchical multimodal framework that integrates a prompted large language model (LLM) with a fine-tuned vision-language model (VLM). At the system level, the LLM performs hierarchical task decomposition and constructs a global semantic map, while the VLM provides semantic perception and object localization, where the proposed GridMask significantly enhances the VLM's spatial accuracy for reliable fine-grained manipulation. The aerial robot leverages this global map to generate semantic paths and guide the ground robot's local navigation and manipulation, ensuring robust coordination even in target-absent or ambiguous scenarios. We validate the framework through extensive simulation and real-world experiments on long-horizon object arrangement tasks, demonstrating zero-shot adaptability, robust semantic navigation, and reliable manipulation in dynamic environments. To the best of our knowledge, this work presents the first heterogeneous aerial-ground robotic system that integrates VLM-based perception with LLM-driven reasoning for global high-level task planning and execution.
>
---
#### [replaced 035] TreeIRL: Safe Urban Driving with Tree Search and Inverse Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.13579v4](http://arxiv.org/pdf/2509.13579v4)**

> **作者:** Momchil S. Tomov; Sang Uk Lee; Hansford Hendrago; Jinwook Huh; Teawon Han; Forbes Howington; Rafael da Silva; Gianmarco Bernasconi; Marc Heim; Samuel Findler; Xiaonan Ji; Alexander Boule; Michael Napoli; Kuo Chen; Jesse Miller; Boaz Floor; Yunqing Hu
>
> **摘要:** We present TreeIRL, a novel planner for autonomous driving that combines Monte Carlo tree search (MCTS) and inverse reinforcement learning (IRL) to achieve state-of-the-art performance in simulation and in real-world driving. The core idea is to use MCTS to find a promising set of safe candidate trajectories and a deep IRL scoring function to select the most human-like among them. We evaluate TreeIRL against both classical and state-of-the-art planners in large-scale simulations and on 500+ miles of real-world autonomous driving in the Las Vegas metropolitan area. Test scenarios include dense urban traffic, adaptive cruise control, cut-ins, and traffic lights. TreeIRL achieves the best overall performance, striking a balance between safety, progress, comfort, and human-likeness. To our knowledge, our work is the first demonstration of MCTS-based planning on public roads and underscores the importance of evaluating planners across a diverse set of metrics and in real-world environments. TreeIRL is highly extensible and could be further improved with reinforcement learning and imitation learning, providing a framework for exploring different combinations of classical and learning-based approaches to solve the planning bottleneck in autonomous driving.
>
---
#### [replaced 036] Data-Driven Soft Robot Control via Adiabatic Spectral Submanifolds
- **分类: cs.RO; cs.SY; eess.SY; nlin.PS**

- **链接: [http://arxiv.org/pdf/2503.10919v2](http://arxiv.org/pdf/2503.10919v2)**

> **作者:** Roshan S. Kaundinya; John Irvin Alora; Jonas G. Matt; Luis A. Pabon; Marco Pavone; George Haller
>
> **备注:** 41 pages, 24 figures
>
> **摘要:** The mechanical complexity of soft robots creates significant challenges for their model-based control. Specifically, linear data-driven models have struggled to control soft robots on complex, spatially extended paths that explore regions with significant nonlinear behavior. To account for these nonlinearities, we develop here a model-predictive control strategy based on the recent theory of adiabatic spectral submanifolds (aSSMs). This theory is applicable because the internal vibrations of heavily overdamped robots decay at a speed that is much faster than the desired speed of the robot along its intended path. In that case, low-dimensional attracting invariant manifolds (aSSMs) emanate from the path and carry the dominant dynamics of the robot. Aided by this recent theory, we devise an aSSM-based model-predictive control scheme purely from data. We demonstrate our data-driven model's effectiveness in tracking dynamic trajectories across diverse tasks, validated on a high-fidelity, high-dimensional finite-element model of a soft trunk robot and a Cosserat rod-based elastic soft arm. Notably, we find that five- or six-dimensional aSSM-reduced models outperform the tracking performance of other data-driven modeling methods by a factor up to $10$ across all closed-loop control tasks.
>
---
#### [replaced 037] SceneWeaver: All-in-One 3D Scene Synthesis with an Extensible and Self-Reflective Agent
- **分类: cs.GR; cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.20414v2](http://arxiv.org/pdf/2509.20414v2)**

> **作者:** Yandan Yang; Baoxiong Jia; Shujie Zhang; Siyuan Huang
>
> **备注:** Accepted by NeurIPS 2025, 26 pages
>
> **摘要:** Indoor scene synthesis has become increasingly important with the rise of Embodied AI, which requires 3D environments that are not only visually realistic but also physically plausible and functionally diverse. While recent approaches have advanced visual fidelity, they often remain constrained to fixed scene categories, lack sufficient object-level detail and physical consistency, and struggle to align with complex user instructions. In this work, we present SceneWeaver, a reflective agentic framework that unifies diverse scene synthesis paradigms through tool-based iterative refinement. At its core, SceneWeaver employs a language model-based planner to select from a suite of extensible scene generation tools, ranging from data-driven generative models to visual- and LLM-based methods, guided by self-evaluation of physical plausibility, visual realism, and semantic alignment with user input. This closed-loop reason-act-reflect design enables the agent to identify semantic inconsistencies, invoke targeted tools, and update the environment over successive iterations. Extensive experiments on both common and open-vocabulary room types demonstrate that SceneWeaver not only outperforms prior methods on physical, visual, and semantic metrics, but also generalizes effectively to complex scenes with diverse instructions, marking a step toward general-purpose 3D environment generation. Project website: https://scene-weaver.github.io/.
>
---
#### [replaced 038] ExAMPC: the Data-Driven Explainable and Approximate NMPC with Physical Insights
- **分类: eess.SY; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2503.00654v2](http://arxiv.org/pdf/2503.00654v2)**

> **作者:** Jean Pierre Allamaa; Panagiotis Patrinos; Tong Duy Son
>
> **备注:** This paper has been accepted for publication in the 2025 IEEE/RSJ IROS Conference
>
> **摘要:** Amidst the surge in the use of Artificial Intelligence (AI) for control purposes, classical and model-based control methods maintain their popularity due to their transparency and deterministic nature. However, advanced controllers like Nonlinear Model Predictive Control (NMPC), despite proven capabilities, face adoption challenges due to their computational complexity and unpredictable closed-loop performance in complex validation systems. This paper introduces ExAMPC, a methodology bridging classical control and explainable AI by augmenting the NMPC with data-driven insights to improve the trustworthiness and reveal the optimization solution and closed-loop performance's sensitivities to physical variables and system parameters. By employing a low-order spline embedding, we reduce the open-loop trajectory dimensionality by over 95%, and integrate it with SHAP and Symbolic Regression from eXplainable AI (XAI) for an approximate NMPC, enabling intuitive physical insights into the NMPC's optimization routine. The prediction accuracy of the approximate NMPC is enhanced through physics-inspired continuous-time constraints penalties, reducing the predicted continuous trajectory violations by 93%. ExAMPC also enables accurate forecasting of the NMPC's computational requirements with explainable insights on worst-case scenarios. Experimental validation on automated valet parking and autonomous racing with lap-time optimization, demonstrates the methodology's practical effectiveness for potential real-world applications.
>
---
#### [replaced 039] Optimal Kinematic Synthesis and Prototype Development of Knee Exoskeleton
- **分类: cs.RO; math.OC**

- **链接: [http://arxiv.org/pdf/2409.02635v2](http://arxiv.org/pdf/2409.02635v2)**

> **作者:** Shashank Mani Gautam; Ekta Singla; Ashish Singla
>
> **摘要:** This study focuses on enhancing the design of an existing knee exoskeleton by addressing limitations in the range of motion (ROM) during Sit-to-Stand (STS) motions. While current knee exoskeletons emphasize toughness and rehabilitation, their closed-loop mechanisms hinder optimal ROM, which is crucial for effective rehabilitation. This research aims to optimize the exoskeleton design to achieve the necessary ROM, improving its functionality in rehabilitation. This can be achieved by utilizing kinematic modeling and formulation, the existing design was represented in the non-linear and non-convex mathematical functions. Optimization techniques, considering constraints based on human leg measurements, were applied to determine the best dimensions for the exoskeleton. This resulted in a significant increase in ROM compared to existing models. A MATLAB program was developed to compare the ROM of the optimized exoskeleton with the original design. To validate the practicality of the optimized design, analysis was conducted using a mannequin with average human dimensions, followed by constructing a cardboard dummy model to confirm simulation results. The STS motion of an average human was captured using a camera and TRACKER software, and the motion was compared with that of the dummy model to identify any misalignments between the human and exoskeleton knee joints. Furthermore, a prototype of the knee joint exoskeleton is being developed to further investigate misalignments and improve the design. Future work includes the use of EMG sensors for more detailed analysis and better results.
>
---
#### [replaced 040] Onboard Mission Replanning for Adaptive Cooperative Multi-Robot Systems
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.06094v3](http://arxiv.org/pdf/2506.06094v3)**

> **作者:** Elim Kwan; Rehman Qureshi; Liam Fletcher; Colin Laganier; Victoria Nockles; Richard Walters
>
> **备注:** 9 pages, 5 figures, 1 table
>
> **摘要:** Cooperative autonomous robotic systems have significant potential for executing complex multi-task missions across space, air, ground, and maritime domains. But they commonly operate in remote, dynamic and hazardous environments, requiring rapid in-mission adaptation without reliance on fragile or slow communication links to centralised compute. Fast, on-board replanning algorithms are therefore needed to enhance resilience. Reinforcement Learning shows strong promise for efficiently solving mission planning tasks when formulated as Travelling Salesperson Problems (TSPs), but existing methods: 1) are unsuitable for replanning, where agents do not start at a single location; 2) do not allow cooperation between agents; 3) are unable to model tasks with variable durations; or 4) lack practical considerations for on-board deployment. Here we define the Cooperative Mission Replanning Problem as a novel variant of multiple TSP with adaptations to overcome these issues, and develop a new encoder/decoder-based model using Graph Attention Networks and Attention Models to solve it effectively and efficiently. Using a simple example of cooperative drones, we show our replanner consistently (90% of the time) maintains performance within 10% of the state-of-the-art LKH3 heuristic solver, whilst running 85-370 times faster on a Raspberry Pi. This work paves the way for increased resilience in autonomous multi-agent systems.
>
---
#### [replaced 041] FailSafe: Reasoning and Recovery from Failures in Vision-Language-Action Models
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.01642v2](http://arxiv.org/pdf/2510.01642v2)**

> **作者:** Zijun Lin; Jiafei Duan; Haoquan Fang; Dieter Fox; Ranjay Krishna; Cheston Tan; Bihan Wen
>
> **备注:** Project Page: https://jimntu.github.io/FailSafe
>
> **摘要:** Recent advances in robotic manipulation have integrated low-level robotic control into Vision-Language Models (VLMs), extending them into Vision-Language-Action (VLA) models. Although state-of-the-art VLAs achieve strong performance in downstream robotic applications, supported by large-scale crowd-sourced robot training data, they still inevitably encounter failures during execution. Enabling robots to reason and recover from unpredictable and abrupt failures remains a critical challenge. Existing robotic manipulation datasets, collected in either simulation or the real world, primarily provide only ground-truth trajectories, leaving robots unable to recover once failures occur. Moreover, the few datasets that address failure detection typically offer only textual explanations, which are difficult to utilize directly in VLA models. To address this gap, we introduce FailSafe, a novel failure generation and recovery system that automatically produces diverse failure cases paired with executable recovery actions. FailSafe can be seamlessly applied to any manipulation task in any simulator, enabling scalable creation of failure action data. To demonstrate its effectiveness, we fine-tune LLaVa-OneVision-7B (LLaVa-OV-7B) to build FailSafe-VLM. Experimental results show that FailSafe-VLM successfully helps robotic arms detect and recover from potential failures, improving the performance of three state-of-the-art VLA models (pi0-FAST, OpenVLA, OpenVLA-OFT) by up to 22.6% on average across several tasks in Maniskill. Furthermore, FailSafe-VLM could generalize across different spatial configurations, camera viewpoints, object and robotic embodiments. We plan to release the FailSafe code to the community.
>
---
#### [replaced 042] Open-Set 3D Semantic Instance Maps for Vision Language Navigation -- O3D-SIM
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2404.17922v2](http://arxiv.org/pdf/2404.17922v2)**

> **作者:** Laksh Nanwani; Kumaraditya Gupta; Aditya Mathur; Swayam Agrawal; A. H. Abdul Hafez; K. Madhava Krishna
>
> **摘要:** Humans excel at forming mental maps of their surroundings, equipping them to understand object relationships and navigate based on language queries. Our previous work, SI Maps (Nanwani L, Agarwal A, Jain K, et al. Instance-level semantic maps for vision language navigation. In: 2023 32nd IEEE International Conference on Robot and Human Interactive Communication (RO-MAN). IEEE; 2023 Aug.), showed that having instance-level information and the semantic understanding of an environment helps significantly improve performance for language-guided tasks. We extend this instance-level approach to 3D while increasing the pipeline's robustness and improving quantitative and qualitative results. Our method leverages foundational models for object recognition, image segmentation, and feature extraction. We propose a representation that results in a 3D point cloud map with instance-level embeddings, which bring in the semantic understanding that natural language commands can query. Quantitatively, the work improves upon the success rate of language-guided tasks. At the same time, we qualitatively observe the ability to identify instances more clearly and leverage the foundational models and language and image-aligned embeddings to identify objects that, otherwise, a closed-set approach wouldn't be able to identify. Project Page - https://smart-wheelchair-rrc.github.io/o3d-sim-webpage
>
---
#### [replaced 043] Open-Vocabulary Spatio-Temporal Scene Graph for Robot Perception and Teleoperation Planning
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.23107v2](http://arxiv.org/pdf/2509.23107v2)**

> **作者:** Yi Wang; Zeyu Xue; Mujie Liu; Tongqin Zhang; Yan Hu; Zhou Zhao; Chenguang Yang; Zhenyu Lu
>
> **摘要:** Teleoperation via natural-language reduces operator workload and enhances safety in high-risk or remote settings. However, in dynamic remote scenes, transmission latency during bidirectional communication creates gaps between remote perceived states and operator intent, leading to command misunderstanding and incorrect execution. To mitigate this, we introduce the Spatio-Temporal Open-Vocabulary Scene Graph (ST-OVSG), a representation that enriches open-vocabulary perception with temporal dynamics and lightweight latency annotations. ST-OVSG leverages LVLMs to construct open-vocabulary 3D object representations, and extends them into the temporal domain via Hungarian assignment with our temporal matching cost, yielding a unified spatio-temporal scene graph. A latency tag is embedded to enable LVLM planners to retrospectively query past scene states, thereby resolving local-remote state mismatches caused by transmission delays. To further reduce redundancy and highlight task-relevant cues, we propose a task-oriented subgraph filtering strategy that produces compact inputs for the planner. ST-OVSG generalizes to novel categories and enhances planning robustness against transmission latency without requiring fine-tuning. Experiments show that our method achieves 74 percent node accuracy on the Replica benchmark, outperforming ConceptGraph. Notably, in the latency-robustness experiment, the LVLM planner assisted by ST-OVSG achieved a planning success rate of 70.5 percent.
>
---
#### [replaced 044] Depth-Constrained ASV Navigation with Deep RL and Limited Sensing
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.18253v3](http://arxiv.org/pdf/2504.18253v3)**

> **作者:** Amirhossein Zhalehmehrabi; Daniele Meli; Francesco Dal Santo; Francesco Trotti; Alessandro Farinelli
>
> **备注:** 8 pages, 8 figures, Accepted to IEEE Robotics and Automation Letters (this is not the final version)
>
> **摘要:** Autonomous Surface Vehicles (ASVs) play a crucial role in maritime operations, yet their navigation in shallow-water environments remains challenging due to dynamic disturbances and depth constraints. Traditional navigation strategies struggle with limited sensor information, making safe and efficient operation difficult. In this paper, we propose a reinforcement learning (RL) framework for ASV navigation under depth constraints, where the vehicle must reach a target while avoiding unsafe areas with only a single depth measurement per timestep from a downward-facing Single Beam Echosounder (SBES). To enhance environmental awareness, we integrate Gaussian Process (GP) regression into the RL framework, enabling the agent to progressively estimate a bathymetric depth map from sparse sonar readings. This approach improves decision-making by providing a richer representation of the environment. Furthermore, we demonstrate effective sim-to-real transfer, ensuring that trained policies generalize well to real-world aquatic conditions. Experimental results validate our method's capability to improve ASV navigation performance while maintaining safety in challenging shallow-water environments.
>
---
#### [replaced 045] iWalker: Imperative Visual Planning for Walking Humanoid Robot
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2409.18361v5](http://arxiv.org/pdf/2409.18361v5)**

> **作者:** Xiao Lin; Yuhao Huang; Taimeng Fu; Xiaobin Xiong; Chen Wang
>
> **摘要:** Humanoid robots, designed to operate in human-centric environments, serve as a fundamental platform for a broad range of tasks. Although humanoid robots have been extensively studied for decades, a majority of existing humanoid robots still heavily rely on complex modular frameworks, leading to inflexibility and potential compounded errors from independent sensing, planning, and acting components. In response, we propose an end-to-end humanoid sense-plan-act walking system, enabling vision-based obstacle avoidance and footstep planning for whole body balancing simultaneously. We designed two imperative learning (IL)-based bilevel optimizations for model-predictive step planning and whole body balancing, respectively, to achieve self-supervised learning for humanoid robot walking. This enables the robot to learn from arbitrary unlabeled data, improving its adaptability and generalization capabilities. We refer to our method as iWalker and demonstrate its effectiveness in both simulated and real-world environments, representing a significant advancement toward autonomous humanoid robots.
>
---
