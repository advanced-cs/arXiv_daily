# 机器人 cs.RO

- **最新发布 69 篇**

- **更新 51 篇**

## 最新发布

#### [new 001] INF-3DP: Implicit Neural Fields for Collision-Free Multi-Axis 3D Printing
- **分类: cs.RO; cs.CG**

- **简介: 该论文提出INF-3DP框架，利用隐式神经场统一生成多轴3D打印路径与避障运动规划，解决复杂模型的高效、无碰撞打印问题，提升速度并减少误差。**

- **链接: [http://arxiv.org/pdf/2509.05345v1](http://arxiv.org/pdf/2509.05345v1)**

> **作者:** Jiasheng Qu; Zhuo Huang; Dezhao Guo; Hailin Sun; Aoran Lyu; Chengkai Dai; Yeung Yam; Guoxin Fang
>
> **摘要:** We introduce a general, scalable computational framework for multi-axis 3D printing based on implicit neural fields (INFs) that unifies all stages of toolpath generation and global collision-free motion planning. In our pipeline, input models are represented as signed distance fields, with fabrication objectives such as support-free printing, surface finish quality, and extrusion control being directly encoded in the optimization of an implicit guidance field. This unified approach enables toolpath optimization across both surface and interior domains, allowing shell and infill paths to be generated via implicit field interpolation. The printing sequence and multi-axis motion are then jointly optimized over a continuous quaternion field. Our continuous formulation constructs the evolving printing object as a time-varying SDF, supporting differentiable global collision handling throughout INF-based motion planning. Compared to explicit-representation-based methods, INF-3DP achieves up to two orders of magnitude speedup and significantly reduces waypoint-to-surface error. We validate our framework on diverse, complex models and demonstrate its efficiency with physical fabrication experiments using a robot-assisted multi-axis system.
>
---
#### [new 002] RoboBallet: Planning for Multi-Robot Reaching with Graph Neural Networks and Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **简介: 论文提出一种基于图神经网络和强化学习的框架，用于多机器人协同完成任务。解决多机器人在复杂环境中自动分配任务、调度和路径规划的问题，实现高效、可扩展的联合运动规划。**

- **链接: [http://arxiv.org/pdf/2509.05397v1](http://arxiv.org/pdf/2509.05397v1)**

> **作者:** Matthew Lai; Keegan Go; Zhibin Li; Torsten Kroger; Stefan Schaal; Kelsey Allen; Jonathan Scholz
>
> **备注:** Published in Science Robotics
>
> **摘要:** Modern robotic manufacturing requires collision-free coordination of multiple robots to complete numerous tasks in shared, obstacle-rich workspaces. Although individual tasks may be simple in isolation, automated joint task allocation, scheduling, and motion planning under spatio-temporal constraints remain computationally intractable for classical methods at real-world scales. Existing multi-arm systems deployed in the industry rely on human intuition and experience to design feasible trajectories manually in a labor-intensive process. To address this challenge, we propose a reinforcement learning (RL) framework to achieve automated task and motion planning, tested in an obstacle-rich environment with eight robots performing 40 reaching tasks in a shared workspace, where any robot can perform any task in any order. Our approach builds on a graph neural network (GNN) policy trained via RL on procedurally-generated environments with diverse obstacle layouts, robot configurations, and task distributions. It employs a graph representation of scenes and a graph policy neural network trained through reinforcement learning to generate trajectories of multiple robots, jointly solving the sub-problems of task allocation, scheduling, and motion planning. Trained on large randomly generated task sets in simulation, our policy generalizes zero-shot to unseen settings with varying robot placements, obstacle geometries, and task poses. We further demonstrate that the high-speed capability of our solution enables its use in workcell layout optimization, improving solution times. The speed and scalability of our planner also open the door to new capabilities such as fault-tolerant planning and online perception-based re-planning, where rapid adaptation to dynamic task sets is required.
>
---
#### [new 003] ZLATTE: A Geometry-Aware, Learning-Free Framework for Language-Driven Trajectory Reshaping in Human-Robot Interaction
- **分类: cs.RO**

- **简介: 该论文提出ZLATTE框架，解决语言驱动下人机交互中轨迹重塑问题。利用视觉-语言模型和大语言模型，将自然语言指令转化为几何约束，并通过势场优化调整轨迹，实现安全、平滑的路径修改。**

- **链接: [http://arxiv.org/pdf/2509.06031v1](http://arxiv.org/pdf/2509.06031v1)**

> **作者:** Junhui Huang; Yuhe Gong; Changsheng Li; Xingguang Duan; Luis Figueredo
>
> **摘要:** We present ZLATTE, a geometry-aware, learning-free framework for language-driven trajectory reshaping in human-robot interaction. Unlike prior learning-based methods, ZLATTE leverages Vision-Language Models to register objects as geometric primitives and employs a Large Language Model to translate natural language instructions into explicit geometric and kinematic constraints. These constraints are integrated into a potential field optimization to adapt initial trajectories while preserving feasibility and safety. A multi-agent strategy further enhances robustness under complex or conflicting commands. Simulation and real-world experiments demonstrate that ZLATTE achieves smoother, safer, and more interpretable trajectory modifications compared to state-of-the-art baselines.
>
---
#### [new 004] Learning in ImaginationLand: Omnidirectional Policies through 3D Generative Models (OP-Gen)
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 论文提出OP-Gen方法，利用3D生成模型扩展真实演示数据集，使机器人能从不同初始状态执行任务。属于强化学习与数据增强任务，解决少量演示下策略泛化问题。**

- **链接: [http://arxiv.org/pdf/2509.06191v1](http://arxiv.org/pdf/2509.06191v1)**

> **作者:** Yifei Ren; Edward Johns
>
> **备注:** Project webpage with robot videos: https://www.robot-learning.uk/op-gen
>
> **摘要:** Recent 3D generative models, which are capable of generating full object shapes from just a few images, now open up new opportunities in robotics. In this work, we show that 3D generative models can be used to augment a dataset from a single real-world demonstration, after which an omnidirectional policy can be learned within this imagined dataset. We found that this enables a robot to perform a task when initialised from states very far from those observed during the demonstration, including starting from the opposite side of the object relative to the real-world demonstration, significantly reducing the number of demonstrations required for policy learning. Through several real-world experiments across tasks such as grasping objects, opening a drawer, and placing trash into a bin, we study these omnidirectional policies by investigating the effect of various design choices on policy behaviour, and we show superior performance to recent baselines which use alternative methods for data augmentation.
>
---
#### [new 005] Evaluation of Large Language Models for Anomaly Detection in Autonomous Vehicles
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 论文评估大语言模型在自动驾驶中的异常检测能力，针对现实世界中自动驾驶易失败的边缘案例。提出结合开放词汇检测与提示工程的架构，分析LLM作为补充模块的潜力。**

- **链接: [http://arxiv.org/pdf/2509.05315v1](http://arxiv.org/pdf/2509.05315v1)**

> **作者:** Petros Loukas; David Bassir; Savvas Chatzichristofis; Angelos Amanatiadis
>
> **摘要:** The rapid evolution of large language models (LLMs) has pushed their boundaries to many applications in various domains. Recently, the research community has started to evaluate their potential adoption in autonomous vehicles and especially as complementary modules in the perception and planning software stacks. However, their evaluation is limited in synthetic datasets or manually driving datasets without the ground truth knowledge and more precisely, how the current perception and planning algorithms would perform in the cases under evaluation. For this reason, this work evaluates LLMs on real-world edge cases where current autonomous vehicles have been proven to fail. The proposed architecture consists of an open vocabulary object detector coupled with prompt engineering and large language model contextual reasoning. We evaluate several state-of-the-art models against real edge cases and provide qualitative comparison results along with a discussion on the findings for the potential application of LLMs as anomaly detectors in autonomous vehicles.
>
---
#### [new 006] Learning Tool-Aware Adaptive Compliant Control for Autonomous Regolith Excavation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 论文提出一种基于模型的强化学习框架，用于自主挖掘月壤。任务是解决复杂颗粒介质交互与多工具操作问题，通过模拟训练使机器人具备自适应控制与工具感知能力，提升挖掘成功率。**

- **链接: [http://arxiv.org/pdf/2509.05475v1](http://arxiv.org/pdf/2509.05475v1)**

> **作者:** Andrej Orsula; Matthieu Geist; Miguel Olivares-Mendez; Carol Martinez
>
> **备注:** The source code is available at https://github.com/AndrejOrsula/space_robotics_bench
>
> **摘要:** Autonomous regolith excavation is a cornerstone of in-situ resource utilization for a sustained human presence beyond Earth. However, this task is fundamentally hindered by the complex interaction dynamics of granular media and the operational need for robots to use diverse tools. To address these challenges, this work introduces a framework where a model-based reinforcement learning agent learns within a parallelized simulation. This environment leverages high-fidelity particle physics and procedural generation to create a vast distribution of both lunar terrains and excavation tool geometries. To master this diversity, the agent learns an adaptive interaction strategy by dynamically modulating its own stiffness and damping at each control step through operational space control. Our experiments demonstrate that training with a procedural distribution of tools is critical for generalization and enables the development of sophisticated tool-aware behavior. Furthermore, we show that augmenting the agent with visual feedback significantly improves task success. These results represent a validated methodology for developing the robust and versatile autonomous systems required for the foundational tasks of future space missions.
>
---
#### [new 007] Energy-Efficient Path Planning with Multi-Location Object Pickup for Mobile Robots on Uneven Terrain
- **分类: cs.RO; cs.DB**

- **简介: 论文提出一种能量高效的路径规划方法，解决移动机器人在不平地形上多地点取物并送至目的地的问题。引入OM EPP问题，设计PCPD数据库优化搜索效率，实现近最优解且计算速度显著提升。**

- **链接: [http://arxiv.org/pdf/2509.06061v1](http://arxiv.org/pdf/2509.06061v1)**

> **作者:** Faiza Babakano; Ahmed Fahmin; Bojie Shen; Muhammad Aamir Cheema; Isma Farah Siddiqui
>
> **摘要:** Autonomous Mobile Robots (AMRs) operate on battery power, making energy efficiency a critical consideration, particularly in outdoor environments where terrain variations affect energy consumption. While prior research has primarily focused on computing energy-efficient paths from a source to a destination, these approaches often overlook practical scenarios where a robot needs to pick up an object en route - an action that can significantly impact energy consumption due to changes in payload. This paper introduces the Object-Pickup Minimum Energy Path Problem (OMEPP), which addresses energy-efficient route planning for AMRs required to pick up an object from one of many possible locations and deliver it to a destination. To address OMEPP, we first introduce a baseline algorithm that employs the Z star algorithm, a variant of A star tailored for energy-efficient routing, to iteratively visit each pickup point. While this approach guarantees optimality, it suffers from high computational cost due to repeated searches at each pickup location. To mitigate this inefficiency, we propose a concurrent PCPD search that manages multiple Z star searches simultaneously across all pickup points. Central to our solution is the Payload-Constrained Path Database (PCPD), an extension of the Compressed Path Database (CPD) that incorporates payload constraints. We demonstrate that PCPD significantly reduces branching factors during search, improving overall performance. Although the concurrent PCPD search may produce slightly suboptimal solutions, extensive experiments on real-world datasets show it achieves near-optimal performance while being one to two orders of magnitude faster than the baseline algorithm.
>
---
#### [new 008] Plantbot: Integrating Plant and Robot through LLM Modular Agent Networks
- **分类: cs.RO; cs.AI**

- **简介: 论文提出Plantbot，通过LLM模块网络连接植物与机器人，实现异步自然语言交互。任务是构建生物与人工系统的新型交互模型，解决跨领域协调问题，工作包括设计LLM接口与传感器动作协调机制。**

- **链接: [http://arxiv.org/pdf/2509.05338v1](http://arxiv.org/pdf/2509.05338v1)**

> **作者:** Atsushi Masumori; Norihiro Maruyama; Itsuki Doi; johnsmith; Hiroki Sato; Takashi Ikegami
>
> **摘要:** We introduce Plantbot, a hybrid lifeform that connects a living plant with a mobile robot through a network of large language model (LLM) modules. Each module - responsible for sensing, vision, dialogue, or action - operates asynchronously and communicates via natural language, enabling seamless interaction across biological and artificial domains. This architecture leverages the capacity of LLMs to serve as hybrid interfaces, where natural language functions as a universal protocol, translating multimodal data (soil moisture, temperature, visual context) into linguistic messages that coordinate system behaviors. The integrated network transforms plant states into robotic actions, installing normativity essential for agency within the sensor-motor loop. By combining biological and robotic elements through LLM-mediated communication, Plantbot behaves as an embodied, adaptive agent capable of responding autonomously to environmental conditions. This approach suggests possibilities for a new model of artificial life, where decentralized, LLM modules coordination enable novel interactions between biological and artificial systems.
>
---
#### [new 009] Dynamic Modeling and Efficient Data-Driven Optimal Control for Micro Autonomous Surface Vehicles
- **分类: cs.RO**

- **简介: 论文提出一种基于物理驱动模型和数据驱动的最优控制框架，用于解决微型自主水面车辆（MicroASV）在复杂环境下的精确控制问题。通过实时在线学习方法优化模型，提升轨迹跟踪精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.06882v1](http://arxiv.org/pdf/2509.06882v1)**

> **作者:** Zhiheng Chen; Wei Wang
>
> **备注:** This work has been accepted to the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Micro Autonomous Surface Vehicles (MicroASVs) offer significant potential for operations in confined or shallow waters and swarm robotics applications. However, achieving precise and robust control at such small scales remains highly challenging, mainly due to the complexity of modeling nonlinear hydrodynamic forces and the increased sensitivity to self-motion effects and environmental disturbances, including waves and boundary effects in confined spaces. This paper presents a physics-driven dynamics model for an over-actuated MicroASV and introduces a data-driven optimal control framework that leverages a weak formulation-based online model learning method. Our approach continuously refines the physics-driven model in real time, enabling adaptive control that adjusts to changing system parameters. Simulation results demonstrate that the proposed method substantially enhances trajectory tracking accuracy and robustness, even under unknown payloads and external disturbances. These findings highlight the potential of data-driven online learning-based optimal control to improve MicroASV performance, paving the way for more reliable and precise autonomous surface vehicle operations.
>
---
#### [new 010] Safety Meets Speed: Accelerated Neural MPC with Safety Guarantees and No Retraining
- **分类: cs.RO**

- **简介: 该论文提出BAN-MPC框架，解决传统MPC实时性不足问题。通过结合神经网络与CBF，实现安全约束下的快速控制，无需重训练，适用于嵌入式系统。**

- **链接: [http://arxiv.org/pdf/2509.06404v1](http://arxiv.org/pdf/2509.06404v1)**

> **作者:** Kaikai Wang; Tianxun Li; Liang Xu; Qinglei Hu; Keyou You
>
> **备注:** 12 pages, 9 figures, accepted to RA-L
>
> **摘要:** While Model Predictive Control (MPC) enforces safety via constraints, its real-time execution can exceed embedded compute budgets. We propose a Barrier-integrated Adaptive Neural Model Predictive Control (BAN-MPC) framework that synergizes neural networks' fast computation with MPC's constraint-handling capability. To ensure strict safety, we replace traditional Euclidean distance with Control Barrier Functions (CBFs) for collision avoidance. We integrate an offline-learned neural value function into the optimization objective of a Short-horizon MPC, substantially reducing online computational complexity. Additionally, we use a second neural network to learn the sensitivity of the value function to system parameters, and adaptively adjust the neural value function based on this neural sensitivity when model parameters change, eliminating the need for retraining and reducing offline computation costs. The hardware in-the-loop (HIL) experiments on Jetson Nano show that BAN-MPC solves 200 times faster than traditional MPC, enabling collision-free navigation with control error below 5\% under model parameter variations within 15\%, making it an effective embedded MPC alternative.
>
---
#### [new 011] HapMorph: A Pneumatic Framework for Multi-Dimensional Haptic Property Rendering
- **分类: cs.RO**

- **简介: 该论文提出HapMorph，一种气动框架，解决可穿戴设备中同时调控物体尺寸和刚度的难题。通过双腔压力调节实现独立控制，并验证了用户感知能力，为多维触觉渲染提供新方法。**

- **链接: [http://arxiv.org/pdf/2509.05433v1](http://arxiv.org/pdf/2509.05433v1)**

> **作者:** Rui Chen; Domenico Chiaradia; Antonio Frisoli; Daniele Leonardis
>
> **备注:** 20 pages, 5 figures
>
> **摘要:** Haptic interfaces that can simultaneously modulate multiple physical properties remain a fundamental challenge in human-robot interaction. Existing systems typically allow the rendering of either geometric features or mechanical properties, but rarely both, within wearable form factors. Here, we introduce HapMorph, a pneumatic framework that enables continuous, simultaneous modulation of object size and stiffness through antagonistic fabric-based pneumatic actuators (AFPAs). We implemented a HapMorph protoytpe designed for hands interaction achieving size variation from 50 to 104 mm, stiffness modulation up to 4.7 N/mm and mass of the wearable parts of just 21 g. Through systematic characterization, we demonstrate decoupled control of size and stiffness properties via dual-chamber pressure regulation. Human perception studies with 10 participants reveal that users can distinguish nine discrete states across three size categories and three stiffness levels with 89.4% accuracy and 6.7 s average response time. We further demonstrate extended architectures that combine AFPAs with complementary pneumatic structures to enable shape or geometry morphing with concurrent stiffness control. Our results establish antagonistic pneumatic principle as a pathway toward next-generation haptic interfaces, capable of multi-dimensiona rendering properties within practical wearable constraints.
>
---
#### [new 012] Real-time Photorealistic Mapping for Situational Awareness in Robot Teleoperation
- **分类: cs.RO**

- **简介: 该论文提出一种基于GPU的实时高精度三维地图生成方法，用于提升机器人远程操作效率。通过结合高斯点云SLAM与在线地图系统，解决传统系统计算成本高、映射不准确的问题，实验证明其在决策速度和环境交互上表现更优。**

- **链接: [http://arxiv.org/pdf/2509.06433v1](http://arxiv.org/pdf/2509.06433v1)**

> **作者:** Ian Page; Pierre Susbielle; Olivier Aycard; Pierre-Brice Wieber
>
> **摘要:** Achieving efficient remote teleoperation is particularly challenging in unknown environments, as the teleoperator must rapidly build an understanding of the site's layout. Online 3D mapping is a proven strategy to tackle this challenge, as it enables the teleoperator to progressively explore the site from multiple perspectives. However, traditional online map-based teleoperation systems struggle to generate visually accurate 3D maps in real-time due to the high computational cost involved, leading to poor teleoperation performances. In this work, we propose a solution to improve teleoperation efficiency in unknown environments. Our approach proposes a novel, modular and efficient GPU-based integration between recent advancement in gaussian splatting SLAM and existing online map-based teleoperation systems. We compare the proposed solution against state-of-the-art teleoperation systems and validate its performances through real-world experiments using an aerial vehicle. The results show significant improvements in decision-making speed and more accurate interaction with the environment, leading to greater teleoperation efficiency. In doing so, our system enhances remote teleoperation by seamlessly integrating photorealistic mapping generation with real-time performances, enabling effective teleoperation in unfamiliar environments.
>
---
#### [new 013] Scenario-based Decision-making Using Game Theory for Interactive Autonomous Driving: A Survey
- **分类: cs.RO**

- **简介: 该论文综述了基于博弈论的场景化自动驾驶决策方法，旨在解决复杂交通场景下的智能决策问题。论文系统总结了各类场景中的算法进展，分析其机制与性能，并指出当前局限与未来方向。**

- **链接: [http://arxiv.org/pdf/2509.05777v1](http://arxiv.org/pdf/2509.05777v1)**

> **作者:** Zhihao Lin; Zhen Tian
>
> **备注:** This paper provides a comprehensive review for scenario-based game-theoretic methods
>
> **摘要:** Game-based interactive driving simulations have emerged as versatile platforms for advancing decision-making algorithms in road transport mobility. While these environments offer safe, scalable, and engaging settings for testing driving strategies, ensuring both realism and robust performance amid dynamic and diverse scenarios remains a significant challenge. Recently, the integration of game-based techniques with advanced learning frameworks has enabled the development of adaptive decision-making models that effectively manage the complexities inherent in varied driving conditions. These models outperform traditional simulation methods, especially when addressing scenario-specific challenges, ranging from obstacle avoidance on highways and precise maneuvering during on-ramp merging to navigation in roundabouts, unsignalized intersections, and even the high-speed demands of autonomous racing. Despite numerous innovations in game-based interactive driving, a systematic review comparing these approaches across different scenarios is still missing. This survey provides a comprehensive evaluation of game-based interactive driving methods by summarizing recent advancements and inherent roadway features in each scenario. Furthermore, the reviewed algorithms are critically assessed based on their adaptation of the standard game model and an analysis of their specific mechanisms to understand their impact on decision-making performance. Finally, the survey discusses the limitations of current approaches and outlines promising directions for future research.
>
---
#### [new 014] Deep Reactive Policy: Learning Reactive Manipulator Motion Planning for Dynamic Environments
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.SY; eess.SY**

- **简介: 论文提出Deep Reactive Policy（DRP），用于动态环境中的机械臂运动规划。针对传统方法依赖完整环境信息且速度慢、神经策略泛化能力差的问题，DRP基于点云输入，结合预训练的IMPACT模型与动态避障模块DCP-RMP，实现高效、通用的反应式运动控制。**

- **链接: [http://arxiv.org/pdf/2509.06953v1](http://arxiv.org/pdf/2509.06953v1)**

> **作者:** Jiahui Yang; Jason Jingzhou Liu; Yulong Li; Youssef Khaky; Kenneth Shaw; Deepak Pathak
>
> **备注:** Website at \url{deep-reactive-policy.com}
>
> **摘要:** Generating collision-free motion in dynamic, partially observable environments is a fundamental challenge for robotic manipulators. Classical motion planners can compute globally optimal trajectories but require full environment knowledge and are typically too slow for dynamic scenes. Neural motion policies offer a promising alternative by operating in closed-loop directly on raw sensory inputs but often struggle to generalize in complex or dynamic settings. We propose Deep Reactive Policy (DRP), a visuo-motor neural motion policy designed for reactive motion generation in diverse dynamic environments, operating directly on point cloud sensory input. At its core is IMPACT, a transformer-based neural motion policy pretrained on 10 million generated expert trajectories across diverse simulation scenarios. We further improve IMPACT's static obstacle avoidance through iterative student-teacher finetuning. We additionally enhance the policy's dynamic obstacle avoidance at inference time using DCP-RMP, a locally reactive goal-proposal module. We evaluate DRP on challenging tasks featuring cluttered scenes, dynamic moving obstacles, and goal obstructions. DRP achieves strong generalization, outperforming prior classical and neural methods in success rate across both simulated and real-world settings. Video results and code available at https://deep-reactive-policy.com
>
---
#### [new 015] Safe Robust Predictive Control-based Motion Planning of Automated Surface Vessels in Inland Waterways
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出基于鲁棒模型预测控制（RMPC）和控制屏障函数（CBFs）的自动水面船舶运动规划方法，解决内河狭窄航道中避障与鲁棒导航问题，提升安全性和适应性。**

- **链接: [http://arxiv.org/pdf/2509.06687v1](http://arxiv.org/pdf/2509.06687v1)**

> **作者:** Sajad Ahmadi; Hossein Nejatbakhsh Esfahani; Javad Mohammadpour Velni
>
> **摘要:** Deploying self-navigating surface vessels in inland waterways offers a sustainable alternative to reduce road traffic congestion and emissions. However, navigating confined waterways presents unique challenges, including narrow channels, higher traffic density, and hydrodynamic disturbances. Existing methods for autonomous vessel navigation often lack the robustness or precision required for such environments. This paper presents a new motion planning approach for Automated Surface Vessels (ASVs) using Robust Model Predictive Control (RMPC) combined with Control Barrier Functions (CBFs). By incorporating channel borders and obstacles as safety constraints within the control design framework, the proposed method ensures both collision avoidance and robust navigation on complex waterways. Simulation results demonstrate the efficacy of the proposed method in safely guiding ASVs under realistic conditions, highlighting its improved safety and adaptability compared to the state-of-the-art.
>
---
#### [new 016] Co-Located VR with Hybrid SLAM-based HMD Tracking and Motion Capture Synchronization
- **分类: cs.RO; cs.HC**

- **简介: 论文提出一种结合SLAM与动作捕捉的多用户VR协同框架，解决传统方法存在的延迟、漂移及校准问题，实现低延迟、高精度的空间同步与多人交互。**

- **链接: [http://arxiv.org/pdf/2509.06582v1](http://arxiv.org/pdf/2509.06582v1)**

> **作者:** Carlos A. Pinheiro de Sousa; Niklas Gröne; Mathias Günther; Oliver Deussen
>
> **备注:** Accepted at the Gesellschaft f\"ur Informatik (GI) VR/AR Workshop 2025 (Lecture Notes in Informatics)
>
> **摘要:** We introduce a multi-user VR co-location framework that synchronizes users within a shared virtual environment aligned to physical space. Our approach combines a motion capture system with SLAM-based inside-out tracking to deliver smooth, high-framerate, low-latency performance. Previous methods either rely on continuous external tracking, which introduces latency and jitter, or on one-time calibration, which cannot correct drift over time. In contrast, our approach combines the responsiveness of local HMD SLAM tracking with the flexibility to realign to an external source when needed. It also supports real-time pose sharing across devices, ensuring consistent spatial alignment and engagement between users. Our evaluation demonstrates that our framework achieves the spatial accuracy required for natural multi-user interaction while offering improved comfort, scalability, and robustness over existing co-located VR solutions.
>
---
#### [new 017] Spiking Neural Networks for Continuous Control via End-to-End Model-Based Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究利用脉冲神经网络（SNN）实现连续控制任务。针对SNN在连续运动控制中的应用不足，提出端到端模型预测控制框架，结合漏积分火神经元动态与代理梯度，优化动力学模型和策略网络，验证其在二维抓取和六自由度机械臂任务中的有效性。**

- **链接: [http://arxiv.org/pdf/2509.05356v1](http://arxiv.org/pdf/2509.05356v1)**

> **作者:** Justus Huebotter; Pablo Lanillos; Marcel van Gerven; Serge Thill
>
> **摘要:** Despite recent progress in training spiking neural networks (SNNs) for classification, their application to continuous motor control remains limited. Here, we demonstrate that fully spiking architectures can be trained end-to-end to control robotic arms with multiple degrees of freedom in continuous environments. Our predictive-control framework combines Leaky Integrate-and-Fire dynamics with surrogate gradients, jointly optimizing a forward model for dynamics prediction and a policy network for goal-directed action. We evaluate this approach on both a planar 2D reaching task and a simulated 6-DOF Franka Emika Panda robot. Results show that SNNs can achieve stable training and accurate torque control, establishing their viability for high-dimensional motor tasks. An extensive ablation study highlights the role of initialization, learnable time constants, and regularization in shaping training dynamics. We conclude that while stable and effective control can be achieved, recurrent spiking networks remain highly sensitive to hyperparameter settings, underscoring the importance of principled design choices.
>
---
#### [new 018] Super-LIO: A Robust and Efficient LiDAR-Inertial Odometry System with a Compact Mapping Strategy
- **分类: cs.RO**

- **简介: 该论文提出Super-LIO系统，解决资源受限平台下LiDAR-Inertial Odometry的高效与准确问题。采用OctVox地图结构和HKNN策略，提升计算效率与鲁棒性，适用于无人机等自主系统。**

- **链接: [http://arxiv.org/pdf/2509.05723v1](http://arxiv.org/pdf/2509.05723v1)**

> **作者:** Liansheng Wang; Xinke Zhang; Chenhui Li; Dongjiao He; Yihan Pan; Jianjun Yi
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** LiDAR-Inertial Odometry (LIO) is a foundational technique for autonomous systems, yet its deployment on resource-constrained platforms remains challenging due to computational and memory limitations. We propose Super-LIO, a robust LIO system that demands both high performance and accuracy, ideal for applications such as aerial robots and mobile autonomous systems. At the core of Super-LIO is a compact octo-voxel-based map structure, termed OctVox, that limits each voxel to eight fused subvoxels, enabling strict point density control and incremental denoising during map updates. This design enables a simple yet efficient and accurate map structure, which can be easily integrated into existing LIO frameworks. Additionally, Super-LIO designs a heuristic-guided KNN strategy (HKNN) that accelerates the correspondence search by leveraging spatial locality, further reducing runtime overhead. We evaluated the proposed system using four publicly available datasets and several self-collected datasets, totaling more than 30 sequences. Extensive testing on both X86 and ARM platforms confirms that Super-LIO offers superior efficiency and robustness, while maintaining competitive accuracy. Super-LIO processes each frame approximately 73% faster than SOTA, while consuming less CPU resources. The system is fully open-source and plug-and-play compatible with a wide range of LiDAR sensors and platforms. The implementation is available at: https://github.com/Liansheng-Wang/Super-LIO.git
>
---
#### [new 019] Microrobot Vascular Parkour: Analytic Geometry-based Path Planning with Real-time Dynamic Obstacle Avoidance
- **分类: cs.RO; cs.AI**

- **简介: 论文提出一种结合解析几何全局规划与实时避障控制器的路径规划框架，用于解决微机器人在血管中自主导航时的动态障碍物避让问题，提升其在复杂血管环境中的实时性和可靠性。**

- **链接: [http://arxiv.org/pdf/2509.05500v1](http://arxiv.org/pdf/2509.05500v1)**

> **作者:** Yanda Yang; Max Sokolich; Fatma Ceren Kirmizitas; Sambeeta Das; Andreas A. Malikopoulos
>
> **备注:** 56 pages, 19 figures including Supplementary Materials. Supplementary videos available at https://robotyyd.github.io/yanda-yang.github.io/vascular-parkour.html. Preprint. This version has not been peer reviewed
>
> **摘要:** Autonomous microrobots in blood vessels could enable minimally invasive therapies, but navigation is challenged by dense, moving obstacles. We propose a real-time path planning framework that couples an analytic geometry global planner (AGP) with two reactive local escape controllers, one based on rules and one based on reinforcement learning, to handle sudden moving obstacles. Using real-time imaging, the system estimates the positions of the microrobot, obstacles, and targets and computes collision-free motions. In simulation, AGP yields shorter paths and faster planning than weighted A* (WA*), particle swarm optimization (PSO), and rapidly exploring random trees (RRT), while maintaining feasibility and determinism. We extend AGP from 2D to 3D without loss of speed. In both simulations and experiments, the combined global planner and local controllers reliably avoid moving obstacles and reach targets. The average planning time is 40 ms per frame, compatible with 25 fps image acquisition and real-time closed-loop control. These results advance autonomous microrobot navigation and targeted drug delivery in vascular environments.
>
---
#### [new 020] CRISP - Compliant ROS2 Controllers for Learning-Based Manipulation Policies and Teleoperation
- **分类: cs.RO**

- **简介: 论文提出CRISP，一种轻量级ROS2合规控制器，用于实现基于学习的机械臂控制与遥操作。解决高阶策略生成低频或不连续状态变化的问题，提供平滑轨迹跟踪与接触交互的柔顺行为。**

- **链接: [http://arxiv.org/pdf/2509.06819v1](http://arxiv.org/pdf/2509.06819v1)**

> **作者:** Daniel San José Pro; Oliver Hausdörfer; Ralf Römer; Maximilian Dösch; Martin Schuck; Angela P. Schöllig
>
> **备注:** 5 pages, 5 figures
>
> **摘要:** Learning-based controllers, such as diffusion policies and vision-language action models, often generate low-frequency or discontinuous robot state changes. Achieving smooth reference tracking requires a low-level controller that converts high-level targets commands into joint torques, enabling compliant behavior during contact interactions. We present CRISP, a lightweight C++ implementation of compliant Cartesian and joint-space controllers for the ROS2 control standard, designed for seamless integration with high-level learning-based policies as well as teleoperation. The controllers are compatible with any manipulator that exposes a joint-torque interface. Through our Python and Gymnasium interfaces, CRISP provides a unified pipeline for recording data from hardware and simulation and deploying high-level learning-based policies seamlessly, facilitating rapid experimentation. The system has been validated on hardware with the Franka Robotics FR3 and in simulation with the Kuka IIWA14 and Kinova Gen3. Designed for rapid integration, flexible deployment, and real-time performance, our implementation provides a unified pipeline for data collection and policy execution, lowering the barrier to applying learning-based methods on ROS2-compatible manipulators. Detailed documentation is available at the project website - https://utiasDSL.github.io/crisp_controllers.
>
---
#### [new 021] Learning to Walk with Less: a Dyna-Style Approach to Quadrupedal Locomotion
- **分类: cs.RO; cs.AI**

- **简介: 论文提出一种基于模型的强化学习框架，通过合成数据提升四足机器人运动控制的数据效率。属于运动控制任务，解决传统RL样本效率低的问题，采用Dyna风格方法，结合预测模型与调度策略优化策略训练。**

- **链接: [http://arxiv.org/pdf/2509.06296v1](http://arxiv.org/pdf/2509.06296v1)**

> **作者:** Francisco Affonso; Felipe Andrade G. Tommaselli; Juliano Negri; Vivian S. Medeiros; Mateus V. Gasparino; Girish Chowdhary; Marcelo Becker
>
> **备注:** Under review at IEEE Robotics and Automation Letters. 8 pages
>
> **摘要:** Traditional RL-based locomotion controllers often suffer from low data efficiency, requiring extensive interaction to achieve robust performance. We present a model-based reinforcement learning (MBRL) framework that improves sample efficiency for quadrupedal locomotion by appending synthetic data to the end of standard rollouts in PPO-based controllers, following the Dyna-Style paradigm. A predictive model, trained alongside the policy, generates short-horizon synthetic transitions that are gradually integrated using a scheduling strategy based on the policy update iterations. Through an ablation study, we identified a strong correlation between sample efficiency and rollout length, which guided the design of our experiments. We validated our approach in simulation on the Unitree Go1 robot and showed that replacing part of the simulated steps with synthetic ones not only mimics extended rollouts but also improves policy return and reduces variance. Finally, we demonstrate that this improvement transfers to the ability to track a wide range of locomotion commands using fewer simulated steps.
>
---
#### [new 022] TeleopLab: Accessible and Intuitive Teleoperation of a Robotic Manipulator for Remote Labs
- **分类: cs.RO; cs.HC**

- **简介: 论文提出TeleopLab系统，解决远程教育中实操设备缺乏的问题。通过手机控制机械臂和实验设备，提升远程STEM学习体验。系统降低任务时间，提高可用性与用户满意度。**

- **链接: [http://arxiv.org/pdf/2509.05547v1](http://arxiv.org/pdf/2509.05547v1)**

> **作者:** Ziling Chen; Yeo Jung Yoon; Rolando Bautista-Montesano; Zhen Zhao; Ajay Mandlekar; John Liu
>
> **摘要:** Teleoperation offers a promising solution for enabling hands-on learning in remote education, particularly in environments requiring interaction with real-world equipment. However, such remote experiences can be costly or non-intuitive. To address these challenges, we present TeleopLab, a mobile device teleoperation system that allows students to control a robotic arm and operate lab equipment. TeleopLab comprises a robotic arm, an adaptive gripper, cameras, lab equipment for a diverse range of applications, a user interface accessible through smartphones, and video call software. We conducted a user study, focusing on task performance, students' perspectives toward the system, usability, and workload assessment. Our results demonstrate a 46.1% reduction in task completion time as users gained familiarity with the system. Quantitative feedback highlighted improvements in students' perspectives after using the system, while NASA TLX and SUS assessments indicated a manageable workload of 38.2 and a positive usability of 73.8. TeleopLab successfully bridges the gap between physical labs and remote education, offering a scalable and effective platform for remote STEM learning.
>
---
#### [new 023] Robotic Manipulation Framework Based on Semantic Keypoints for Packing Shoes of Different Sizes, Shapes, and Softness
- **分类: cs.RO**

- **简介: 该论文提出一种基于语义关键点的机器人抓取框架，用于解决不同尺寸、形状和软硬度鞋类的配对包装任务。研究设计了感知模块、再定向规划器和包装规划器，实现任意初始状态下的鞋类包装，提升变形物体再定向效率与包装策略有效性。**

- **链接: [http://arxiv.org/pdf/2509.06048v1](http://arxiv.org/pdf/2509.06048v1)**

> **作者:** Yi Dong; Yangjun Liu; Jinjun Duan; Yang Li; Zhendong Dai
>
> **备注:** Yi Dong and Yangjun Liu contributed equally to the work. Accepted by Robotics and Autonomous Systems. https://authors.elsevier.com/c/1lgjX3HdG3supQ
>
> **摘要:** With the rapid development of the warehousing and logistics industries, the packing of goods has gradually attracted the attention of academia and industry. The packing of footwear products is a typical representative paired-item packing task involving irregular shapes and deformable objects. Although studies on shoe packing have been conducted, different initial states due to the irregular shapes of shoes and standard packing placement poses have not been considered. This study proposes a robotic manipulation framework, including a perception module, reorientation planners, and a packing planner, that can complete the packing of pairs of shoes in any initial state. First, to adapt to the large intraclass variations due to the state, shape, and deformation of the shoe, we propose a vision module based on semantic keypoints, which can also infer more information such as size, state, pose, and manipulation points by combining geometric features. Subsequently, we not only proposed primitive-based reorientation methods for different states of a single deformable shoe but also proposed a fast reorientation method for the top state using box edge contact and gravity, which further improved the efficiency of reorientation. Finally, based on the perception module and reorientation methods, we propose a task planner for shoe pair packing in any initial state to provide an optimal packing strategy. Real-world experiments were conducted to verify the robustness of the reorientation methods and the effectiveness of the packing strategy for various types of shoes. In this study, we highlight the potential of semantic keypoint representation methods, introduce new perspectives on the reorientation of 3D deformable objects and multi-object manipulation, and provide a reference for paired object packing.
>
---
#### [new 024] Learning to Walk in Costume: Adversarial Motion Priors for Aesthetically Constrained Humanoids
- **分类: cs.RO; cs.AI; cs.SY; eess.SY; 68T40; I.2.9; I.2.6**

- **简介: 论文提出基于强化学习的运动系统，解决娱乐机器人Cosmo因美学设计导致的运动稳定性问题。通过对抗运动先验（AMP）和定制奖励机制，使机器人实现自然且稳定的行走，为兼顾外观与功能的机器人设计提供新方法。**

- **链接: [http://arxiv.org/pdf/2509.05581v1](http://arxiv.org/pdf/2509.05581v1)**

> **作者:** Arturo Flores Alvarez; Fatemeh Zargarbashi; Havel Liu; Shiqi Wang; Liam Edwards; Jessica Anz; Alex Xu; Fan Shi; Stelian Coros; Dennis W. Hong
>
> **备注:** 8 pages, 11 figures, accepted at IEEE-RAS International Conference on Humanoid Robots (Humanoids) 2025
>
> **摘要:** We present a Reinforcement Learning (RL)-based locomotion system for Cosmo, a custom-built humanoid robot designed for entertainment applications. Unlike traditional humanoids, entertainment robots present unique challenges due to aesthetic-driven design choices. Cosmo embodies these with a disproportionately large head (16% of total mass), limited sensing, and protective shells that considerably restrict movement. To address these challenges, we apply Adversarial Motion Priors (AMP) to enable the robot to learn natural-looking movements while maintaining physical stability. We develop tailored domain randomization techniques and specialized reward structures to ensure safe sim-to-real, protecting valuable hardware components during deployment. Our experiments demonstrate that AMP generates stable standing and walking behaviors despite Cosmo's extreme mass distribution and movement constraints. These results establish a promising direction for robots that balance aesthetic appeal with functional performance, suggesting that learning-based methods can effectively adapt to aesthetic-driven design constraints.
>
---
#### [new 025] Sharing but Not Caring: Similar Outcomes for Shared Control and Switching Control in Telepresence-Robot Navigation
- **分类: cs.RO**

- **简介: 论文研究远程机器人导航中的共享控制与切换控制方法，旨在提高效率并降低用户负担。通过两项用户实验发现，共享控制与切换控制在导航效率上无显著差异，但未明显减轻任务负荷，需进一步探索影响用户偏好的因素。属于人机交互与机器人控制任务。**

- **链接: [http://arxiv.org/pdf/2509.05672v1](http://arxiv.org/pdf/2509.05672v1)**

> **作者:** Juho Kalliokoski; Evan G. Center; Steven M. LaValle; Timo Ojala; Basak Sakcak
>
> **备注:** Immersive telepresence, shared control
>
> **摘要:** Telepresence robots enable users to interact with remote environments, but efficient and intuitive navigation remains a challenge. In this work, we developed and evaluated a shared control method, in which the robot navigates autonomously while allowing users to affect the path generation to better suit their needs. We compared this with control switching, where users toggle between direct and automated control. We hypothesized that shared control would maintain efficiency comparable to control switching while potentially reducing user workload. The results of two consecutive user studies (each with final sample of n=20) showed that shared control does not degrade navigation efficiency, but did not show a significant reduction in task load compared to control switching. Further research is needed to explore the underlying factors that influence user preference and performance in these control systems.
>
---
#### [new 026] Hybrid A* Path Planning with Multi-Modal Motion Extension for Four-Wheel Steering Mobile Robots
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出一种改进的Hybrid A*算法，用于四轮独立转向移动机器人的路径规划。通过引入多模态运动模式和状态空间扩展，解决传统方法无法充分利用4WIS平台灵活性的问题，提升复杂环境下的规划性能。**

- **链接: [http://arxiv.org/pdf/2509.06115v1](http://arxiv.org/pdf/2509.06115v1)**

> **作者:** Runjiao Bao; Lin Zhang; Tianwei Niu; Haoyu Yuan; Shoukun Wang
>
> **摘要:** Four-wheel independent steering (4WIS) systems provide mobile robots with a rich set of motion modes, such as Ackermann steering, lateral steering, and parallel movement, offering superior maneuverability in constrained environments. However, existing path planning methods generally assume a single kinematic model and thus fail to fully exploit the multi-modal capabilities of 4WIS platforms. To address this limitation, we propose an extended Hybrid A* framework that operates in a four-dimensional state space incorporating both spatial states and motion modes. Within this framework, we design multi-modal Reeds-Shepp curves tailored to the distinct kinematic constraints of each motion mode, develop an enhanced heuristic function that accounts for mode-switching costs, and introduce a terminal connection strategy with intelligent mode selection to ensure smooth transitions between different steering patterns. The proposed planner enables seamless integration of multiple motion modalities within a single path, significantly improving flexibility and adaptability in complex environments. Results demonstrate significantly improved planning performance for 4WIS robots in complex environments.
>
---
#### [new 027] Human-LLM Synergy in Context-Aware Adaptive Architecture for Scalable Drone Swarm Operation
- **分类: cs.RO; cs.MA**

- **简介: 论文提出一种基于大语言模型的自适应架构，用于无人机群在灾难响应中的协同操作。该架构根据任务参数动态选择最佳结构，提升可扩展性、适应性和鲁棒性，解决传统固定架构在动态环境下的效率与连接问题。**

- **链接: [http://arxiv.org/pdf/2509.05355v1](http://arxiv.org/pdf/2509.05355v1)**

> **作者:** Ahmed R. Sadik; Muhammad Ashfaq; Niko Mäkitalo; Tommi Mikkonen
>
> **摘要:** The deployment of autonomous drone swarms in disaster response missions necessitates the development of flexible, scalable, and robust coordination systems. Traditional fixed architectures struggle to cope with dynamic and unpredictable environments, leading to inefficiencies in energy consumption and connectivity. This paper addresses this gap by proposing an adaptive architecture for drone swarms, leveraging a Large Language Model to dynamically select the optimal architecture as centralized, hierarchical, or holonic based on real time mission parameters such as task complexity, swarm size, and communication stability. Our system addresses the challenges of scalability, adaptability, and robustness,ensuring efficient energy consumption and maintaining connectivity under varying conditions. Extensive simulations demonstrate that our adaptive architecture outperforms traditional static models in terms of scalability, energy efficiency, and connectivity. These results highlight the potential of our approach to provide a scalable, adaptable, and resilient solution for real world disaster response scenarios.
>
---
#### [new 028] LocoMamba: Vision-Driven Locomotion via End-to-End Deep Reinforcement Learning with Mamba
- **分类: cs.RO; cs.AI; cs.CV; cs.SY; eess.IV; eess.SY**

- **简介: 论文提出LocoMamba，一种基于视觉的强化学习框架，用于机器人运动控制。通过结合Mamba模型与深度强化学习，实现高效长序列建模，提升在复杂地形和障碍物环境中的移动性能与泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.11849v2](http://arxiv.org/pdf/2508.11849v2)**

> **作者:** Yinuo Wang; Gavin Tao
>
> **备注:** 13 pages
>
> **摘要:** We introduce LocoMamba, a vision-driven cross-modal DRL framework built on selective state-space models, specifically leveraging Mamba, that achieves near-linear-time sequence modeling, effectively captures long-range dependencies, and enables efficient training with longer sequences. First, we embed proprioceptive states with a multilayer perceptron and patchify depth images with a lightweight convolutional neural network, producing compact tokens that improve state representation. Second, stacked Mamba layers fuse these tokens via near-linear-time selective scanning, reducing latency and memory footprint, remaining robust to token length and image resolution, and providing an inductive bias that mitigates overfitting. Third, we train the policy end-to-end with Proximal Policy Optimization under terrain and appearance randomization and an obstacle-density curriculum, using a compact state-centric reward that balances progress, smoothness, and safety. We evaluate our method in challenging simulated environments with static and moving obstacles as well as uneven terrain. Compared with state-of-the-art baselines, our method achieves higher returns and success rates with fewer collisions, exhibits stronger generalization to unseen terrains and obstacle densities, and improves training efficiency by converging in fewer updates under the same compute budget.
>
---
#### [new 029] ManipDreamer3D : Synthesizing Plausible Robotic Manipulation Video with Occupancy-aware 3D Trajectory
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 论文提出ManipDreamer3D框架，用于生成具有合理3D轨迹的机器人操作视频。该方法结合3D轨迹规划与占用地图重建，解决现有方法依赖2D轨迹导致的3D空间模糊问题，提升视频生成的视觉质量与合理性。**

- **链接: [http://arxiv.org/pdf/2509.05314v1](http://arxiv.org/pdf/2509.05314v1)**

> **作者:** Ying Li; Xiaobao Wei; Xiaowei Chi; Yuming Li; Zhongyu Zhao; Hao Wang; Ningning Ma; Ming Lu; Shanghang Zhang
>
> **备注:** 8pages; 7figures; 4 tables
>
> **摘要:** Data scarcity continues to be a major challenge in the field of robotic manipulation. Although diffusion models provide a promising solution for generating robotic manipulation videos, existing methods largely depend on 2D trajectories, which inherently face issues with 3D spatial ambiguity. In this work, we present a novel framework named ManipDreamer3D for generating plausible 3D-aware robotic manipulation videos from the input image and the text instruction. Our method combines 3D trajectory planning with a reconstructed 3D occupancy map created from a third-person perspective, along with a novel trajectory-to-video diffusion model. Specifically, ManipDreamer3D first reconstructs the 3D occupancy representation from the input image and then computes an optimized 3D end-effector trajectory, minimizing path length while avoiding collisions. Next, we employ a latent editing technique to create video sequences from the initial image latent and the optimized 3D trajectory. This process conditions our specially trained trajectory-to-video diffusion model to produce robotic pick-and-place videos. Our method generates robotic videos with autonomously planned plausible 3D trajectories, significantly reducing human intervention requirements. Experimental results demonstrate superior visual quality compared to existing methods.
>
---
#### [new 030] An Adaptive Coverage Control Approach for Multiple Autonomous Off-road Vehicles in Dynamic Agricultural Fields
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出一种自适应覆盖控制方法，用于多辆越野无人车在动态农业环境中的路径规划。通过集成无人机检测障碍和评估地形，实现无人车动态调整覆盖路径，提升导航效率与覆盖率。**

- **链接: [http://arxiv.org/pdf/2509.06682v1](http://arxiv.org/pdf/2509.06682v1)**

> **作者:** Sajad Ahmadi; Mohammadreza Davoodi; Javad Mohammadpour Velni
>
> **摘要:** This paper presents an adaptive coverage control method for a fleet of off-road and Unmanned Ground Vehicles (UGVs) operating in dynamic (time-varying) agricultural environments. Traditional coverage control approaches often assume static conditions, making them unsuitable for real-world farming scenarios where obstacles, such as moving machinery and uneven terrains, create continuous challenges. To address this, we propose a real-time path planning framework that integrates Unmanned Aerial Vehicles (UAVs) for obstacle detection and terrain assessment, allowing UGVs to dynamically adjust their coverage paths. The environment is modeled as a weighted directed graph, where the edge weights are continuously updated based on the UAV observations to reflect obstacle motion and terrain variations. The proposed approach incorporates Voronoi-based partitioning, adaptive edge weight assignment, and cost-based path optimization to enhance navigation efficiency. Simulation results demonstrate the effectiveness of the proposed method in improving path planning, reducing traversal costs, and maintaining robust coverage in the presence of dynamic obstacles and muddy terrains.
>
---
#### [new 031] Grasp-MPC: Closed-Loop Visual Grasping via Value-Guided Model Predictive Control
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 论文提出Grasp-MPC，一种基于视觉的闭环6-DoF抓取策略，用于解决杂乱环境中新型物体抓取问题。通过结合价值函数与MPC框架，提升抓取成功率，优于现有方法。属于机器人抓取任务。**

- **链接: [http://arxiv.org/pdf/2509.06201v1](http://arxiv.org/pdf/2509.06201v1)**

> **作者:** Jun Yamada; Adithyavairavan Murali; Ajay Mandlekar; Clemens Eppner; Ingmar Posner; Balakumar Sundaralingam
>
> **备注:** 14 pages, 17 figures
>
> **摘要:** Grasping of diverse objects in unstructured environments remains a significant challenge. Open-loop grasping methods, effective in controlled settings, struggle in cluttered environments. Grasp prediction errors and object pose changes during grasping are the main causes of failure. In contrast, closed-loop methods address these challenges in simplified settings (e.g., single object on a table) on a limited set of objects, with no path to generalization. We propose Grasp-MPC, a closed-loop 6-DoF vision-based grasping policy designed for robust and reactive grasping of novel objects in cluttered environments. Grasp-MPC incorporates a value function, trained on visual observations from a large-scale synthetic dataset of 2 million grasp trajectories that include successful and failed attempts. We deploy this learned value function in an MPC framework in combination with other cost terms that encourage collision avoidance and smooth execution. We evaluate Grasp-MPC on FetchBench and real-world settings across diverse environments. Grasp-MPC improves grasp success rates by up to 32.6% in simulation and 33.3% in real-world noisy conditions, outperforming open-loop, diffusion policy, transformer policy, and IQL approaches. Videos and more at http://grasp-mpc.github.io.
>
---
#### [new 032] Event Driven CBBA with Reduced Communication
- **分类: cs.RO**

- **简介: 论文提出一种事件驱动的CBBA算法（ED-CBBA），用于多机器人任务分配。针对传统CBBA通信开销大的问题，通过减少通信次数提升效率，理论证明其性能与CBBA相当，并通过仿真验证可减少52%的消息传输。**

- **链接: [http://arxiv.org/pdf/2509.06481v1](http://arxiv.org/pdf/2509.06481v1)**

> **作者:** Vinita Sao; Tu Dac Ho; Sujoy Bhore; P. B. Sujit
>
> **摘要:** In various scenarios such as multi-drone surveillance and search-and-rescue operations, deploying multiple robots is essential to accomplish multiple tasks at once. Due to the limited communication range of these vehicles, a decentralised task allocation algorithm is crucial for effective task distribution among robots. The consensus-based bundle algorithm (CBBA) has been promising for multi-robot operation, offering theoretical guarantees. However, CBBA demands continuous communication, leading to potential congestion and packet loss that can hinder performance. In this study, we introduce an event-driven communication mechanism designed to address these communication challenges while maintaining the convergence and performance bounds of CBBA. We demonstrate theoretically that the solution quality matches that of CBBA and validate the approach with Monte-Carlo simulations across varying targets, agents, and bundles. Results indicate that the proposed algorithm (ED-CBBA) can reduce message transmissions by up to 52%.
>
---
#### [new 033] Adaptive Evolution Factor Risk Ellipse Framework for Reliable and Safe Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文提出ERPF-MPC框架，解决自动驾驶中动态交通场景下的安全与效率问题。通过风险椭圆和自适应演化因子，实现动态风险评估与路径规划，提升轨迹平滑性与避撞能力。属于自动驾驶路径规划与安全控制任务。**

- **链接: [http://arxiv.org/pdf/2509.06375v1](http://arxiv.org/pdf/2509.06375v1)**

> **作者:** Fujiang Yuan; Zhen Tian; Yangfan He; Guojian Zou; Chunhong Yuan; Yanhong Peng; Zhihao Lin
>
> **摘要:** In recent years, ensuring safety, efficiency, and comfort in interactive autonomous driving has become a critical challenge. Traditional model-based techniques, such as game-theoretic methods and robust control, are often overly conservative or computationally intensive. Conversely, learning-based approaches typically require extensive training data and frequently exhibit limited interpretability and generalizability. Simpler strategies, such as Risk Potential Fields (RPF), provide lightweight alternatives with minimal data demands but are inherently static and struggle to adapt effectively to dynamic traffic conditions. To overcome these limitations, we propose the Evolutionary Risk Potential Field (ERPF), a novel approach that dynamically updates risk assessments in dynamical scenarios based on historical obstacle proximity data. We introduce a Risk-Ellipse construct that combines longitudinal reach and lateral uncertainty into a unified spatial temporal collision envelope. Additionally, we define an adaptive Evolution Factor metric, computed through sigmoid normalization of Time to Collision (TTC) and Time-Window-of-Hazard (TWH), which dynamically adjusts the dimensions of the ellipse axes in real time. This adaptive risk metric is integrated seamlessly into a Model Predictive Control (MPC) framework, enabling autonomous vehicles to proactively address complex interactive driving scenarios in terms of uncertain driving of surrounding vehicles. Comprehensive comparative experiments demonstrate that our ERPF-MPC approach consistently achieves smoother trajectories, higher average speeds, and collision-free navigation, offering a robust and adaptive solution suitable for complex interactive driving environments.
>
---
#### [new 034] Interactive Shaping of Granular Media Using Reinforcement Learning
- **分类: cs.RO**

- **简介: 论文提出一种基于强化学习的框架，使机械臂能通过视觉策略塑造沙等颗粒介质。任务为自主操控颗粒材料，解决其高维状态空间与复杂动态问题，验证了紧凑观测与奖励设计的有效性，优于传统方法。**

- **链接: [http://arxiv.org/pdf/2509.06469v1](http://arxiv.org/pdf/2509.06469v1)**

> **作者:** Benedikt Kreis; Malte Mosbach; Anny Ripke; Muhammad Ehsan Ullah; Sven Behnke; Maren Bennewitz
>
> **备注:** Accepted to IEEE-RAS International Conference on Humanoid Robots (Humanoids) 2025
>
> **摘要:** Autonomous manipulation of granular media, such as sand, is crucial for applications in construction, excavation, and additive manufacturing. However, shaping granular materials presents unique challenges due to their high-dimensional configuration space and complex dynamics, where traditional rule-based approaches struggle without extensive engineering efforts. Reinforcement learning (RL) offers a promising alternative by enabling agents to learn adaptive manipulation strategies through trial and error. In this work, we present an RL framework that enables a robotic arm with a cubic end-effector and a stereo camera to shape granular media into desired target structures. We show the importance of compact observations and concise reward formulations for the large configuration space, validating our design choices with an ablation study. Our results demonstrate the effectiveness of the proposed approach for the training of visual policies that manipulate granular media including their real-world deployment, outperforming two baseline approaches.
>
---
#### [new 035] eKalibr-Inertial: Continuous-Time Spatiotemporal Calibration for Event-Based Visual-Inertial Systems
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出eKalibr-Inertial，用于事件相机与惯性测量单元的时空标定。针对事件视觉惯性系统，解决其精准标定问题，通过网格板初始化和连续时间优化提升标定精度。**

- **链接: [http://arxiv.org/pdf/2509.05923v1](http://arxiv.org/pdf/2509.05923v1)**

> **作者:** Shuolong Chen; Xingxing Li; Liu Yuan
>
> **摘要:** The bioinspired event camera, distinguished by its exceptional temporal resolution, high dynamic range, and low power consumption, has been extensively studied in recent years for motion estimation, robotic perception, and object detection. In ego-motion estimation, the visual-inertial setup is commonly adopted due to complementary characteristics between sensors (e.g., scale perception and low drift). For optimal event-based visual-inertial fusion, accurate spatiotemporal (extrinsic and temporal) calibration is required. In this work, we present eKalibr-Inertial, an accurate spatiotemporal calibrator for event-based visual-inertial systems, utilizing the widely used circle grid board. Building upon the grid pattern recognition and tracking methods in eKalibr and eKalibr-Stereo, the proposed method starts with a rigorous and efficient initialization, where all parameters in the estimator would be accurately recovered. Subsequently, a continuous-time-based batch optimization is conducted to refine the initialized parameters toward better states. The results of extensive real-world experiments show that eKalibr-Inertial can achieve accurate event-based visual-inertial spatiotemporal calibration. The implementation of eKalibr-Inertial is open-sourced at (https://github.com/Unsigned-Long/eKalibr) to benefit the research community.
>
---
#### [new 036] Embodied Hazard Mitigation using Vision-Language Models for Autonomous Mobile Robots
- **分类: cs.RO**

- **简介: 该论文提出一种融合视觉-语言模型的异常检测与缓解系统，用于自主移动机器人。旨在实时识别并应对城市和环境中的危险与冲突，提升机器人安全性和操作连续性。通过集成危害状态到决策框架，实现高效检测与响应。**

- **链接: [http://arxiv.org/pdf/2509.06768v1](http://arxiv.org/pdf/2509.06768v1)**

> **作者:** Oluwadamilola Sotomi; Devika Kodi; Kiruthiga Chandra Shekar; Aliasghar Arab
>
> **摘要:** Autonomous robots operating in dynamic environments should identify and report anomalies. Embodying proactive mitigation improves safety and operational continuity. This paper presents a multimodal anomaly detection and mitigation system that integrates vision-language models and large language models to identify and report hazardous situations and conflicts in real-time. The proposed system enables robots to perceive, interpret, report, and if possible respond to urban and environmental anomalies through proactive detection mechanisms and automated mitigation actions. A key contribution in this paper is the integration of Hazardous and Conflict states into the robot's decision-making framework, where each anomaly type can trigger specific mitigation strategies. User studies (n = 30) demonstrated the effectiveness of the system in anomaly detection with 91.2% prediction accuracy and relatively low latency response times using edge-ai architecture.
>
---
#### [new 037] O$^3$Afford: One-Shot 3D Object-to-Object Affordance Grounding for Generalizable Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出O$^3$Afford，解决机器人操作中对象间 affordance 接地问题。在有限数据下，结合3D点云与视觉基础模型，实现单次学习的泛化能力，并融合大语言模型提升任务约束生成能力。属于机器人操作中的对象间 affordance 接地任务。**

- **链接: [http://arxiv.org/pdf/2509.06233v1](http://arxiv.org/pdf/2509.06233v1)**

> **作者:** Tongxuan Tian; Xuhui Kang; Yen-Ling Kuo
>
> **备注:** Conference on Robot Learning (CoRL) 2025. Project website: https://o3afford.github.io/
>
> **摘要:** Grounding object affordance is fundamental to robotic manipulation as it establishes the critical link between perception and action among interacting objects. However, prior works predominantly focus on predicting single-object affordance, overlooking the fact that most real-world interactions involve relationships between pairs of objects. In this work, we address the challenge of object-to-object affordance grounding under limited data contraints. Inspired by recent advances in few-shot learning with 2D vision foundation models, we propose a novel one-shot 3D object-to-object affordance learning approach for robotic manipulation. Semantic features from vision foundation models combined with point cloud representation for geometric understanding enable our one-shot learning pipeline to generalize effectively to novel objects and categories. We further integrate our 3D affordance representation with large language models (LLMs) for robotics manipulation, significantly enhancing LLMs' capability to comprehend and reason about object interactions when generating task-specific constraint functions. Our experiments on 3D object-to-object affordance grounding and robotic manipulation demonstrate that our O$^3$Afford significantly outperforms existing baselines in terms of both accuracy and generalization capability.
>
---
#### [new 038] LLaDA-VLA: Vision Language Diffusion Action Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出LLaDA-VLA模型，用于机器人操作任务。旨在解决将扩散语言模型应用于机器人策略学习的问题，通过特殊标记分类和分层解码策略提升性能，实验证明其优于现有方法。**

- **链接: [http://arxiv.org/pdf/2509.06932v1](http://arxiv.org/pdf/2509.06932v1)**

> **作者:** Yuqing Wen; Hebei Li; Kefan Gu; Yucheng Zhao; Tiancai Wang; Xiaoyan Sun
>
> **摘要:** The rapid progress of auto-regressive vision-language models (VLMs) has inspired growing interest in vision-language-action models (VLA) for robotic manipulation. Recently, masked diffusion models, a paradigm distinct from autoregressive models, have begun to demonstrate competitive performance in text generation and multimodal applications, leading to the development of a series of diffusion-based VLMs (d-VLMs). However, leveraging such models for robot policy learning remains largely unexplored. In this work, we present LLaDA-VLA, the first Vision-Language-Diffusion-Action model built upon pretrained d-VLMs for robotic manipulation. To effectively adapt d-VLMs to robotic domain, we introduce two key designs: (1) a localized special-token classification strategy that replaces full-vocabulary classification with special action token classification, reducing adaptation difficulty; (2) a hierarchical action-structured decoding strategy that decodes action sequences hierarchically considering the dependencies within and across actions. Extensive experiments demonstrate that LLaDA-VLA significantly outperforms state-of-the-art VLAs on both simulation and real-world robots.
>
---
#### [new 039] MonoGlass3D: Monocular 3D Glass Detection with Plane Regression and Adaptive Feature Fusion
- **分类: cs.RO**

- **简介: 该论文提出MonoGlass3D方法，解决单目3D玻璃检测问题。针对玻璃光学特性带来的检测困难，构建新数据集，并设计自适应特征融合模块与平面回归管道，提升透明表面理解效果。**

- **链接: [http://arxiv.org/pdf/2509.05599v1](http://arxiv.org/pdf/2509.05599v1)**

> **作者:** Kai Zhang; Guoyang Zhao; Jianxing Shi; Bonan Liu; Weiqing Qi; Jun Ma
>
> **摘要:** Detecting and localizing glass in 3D environments poses significant challenges for visual perception systems, as the optical properties of glass often hinder conventional sensors from accurately distinguishing glass surfaces. The lack of real-world datasets focused on glass objects further impedes progress in this field. To address this issue, we introduce a new dataset featuring a wide range of glass configurations with precise 3D annotations, collected from distinct real-world scenarios. On the basis of this dataset, we propose MonoGlass3D, a novel approach tailored for monocular 3D glass detection across diverse environments. To overcome the challenges posed by the ambiguous appearance and context diversity of glass, we propose an adaptive feature fusion module that empowers the network to effectively capture contextual information in varying conditions. Additionally, to exploit the distinct planar geometry of glass surfaces, we present a plane regression pipeline, which enables seamless integration of geometric properties within our framework. Extensive experiments demonstrate that our method outperforms state-of-the-art approaches in both glass segmentation and monocular glass depth estimation. Our results highlight the advantages of combining geometric and contextual cues for transparent surface understanding.
>
---
#### [new 040] A*-PRM: A Dynamic Weight-Based Probabilistic Roadmap Algorithm
- **分类: cs.RO**

- **简介: 该论文提出A*-PRM算法，结合A*与PRM方法，优化路径质量和计算效率。属于机器人路径规划任务，解决复杂环境中高效避障问题，通过动态权重和分层采样策略提升适应性与稳定性。**

- **链接: [http://arxiv.org/pdf/2509.05701v1](http://arxiv.org/pdf/2509.05701v1)**

> **作者:** Siyuan Wang; Shuyi Zhang; Zhen Tian; Yuheng Yao; Gongsen Wang; Yu Zhao
>
> **摘要:** Robot path planning is a fundamental challenge in enhancing the environmental adaptability of autonomous navigation systems. This paper presents a hybrid path planning algorithm, A-star PRM, which incorporates dynamic weights. By embedding the Manhattan distance heuristic of the A-star algorithm into the random sampling process of PRM, the algorithm achieves a balanced optimization of path quality and computational efficiency. The approach uses a hierarchical sampling strategy and a dynamic connection mechanism, greatly improving adaptability to complex obstacle distributions. Experiments show that under a baseline configuration with one thousand sampled vertices, the path length of A-star PRM is 1073.23 plus or minus 14.8 meters and is 42.3 percent shorter than that of PRM with p value less than 0.01. With high-density sampling using three thousand vertices, the path length is reduced by 0.94 percent, 1036.61 meters compared with 1046.42 meters, while the increase in computational time is cut to about one tenth of the PRM increase, 71 percent compared with 785 percent. These results confirm the comprehensive advantages of A-star PRM in path quality, stability, and computational efficiency. Compared with existing hybrid algorithms, the proposed method shows clear benefits, especially in narrow channels and scenarios with dynamic obstacles.
>
---
#### [new 041] F1: A Vision-Language-Action Model Bridging Understanding and Generation to Actions
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出F1模型，解决动态视觉环境中语言指令任务执行问题。通过集成视觉预测与决策机制，提升动作生成的远见性与鲁棒性，实现更高效的任务完成与泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.06951v1](http://arxiv.org/pdf/2509.06951v1)**

> **作者:** Qi Lv; Weijie Kong; Hao Li; Jia Zeng; Zherui Qiu; Delin Qu; Haoming Song; Qizhi Chen; Xiang Deng; Jiangmiao Pang
>
> **摘要:** Executing language-conditioned tasks in dynamic visual environments remains a central challenge in embodied AI. Existing Vision-Language-Action (VLA) models predominantly adopt reactive state-to-action mappings, often leading to short-sighted behaviors and poor robustness in dynamic scenes. In this paper, we introduce F1, a pretrained VLA framework which integrates the visual foresight generation into decision-making pipeline. F1 adopts a Mixture-of-Transformer architecture with dedicated modules for perception, foresight generation, and control, thereby bridging understanding, generation, and actions. At its core, F1 employs a next-scale prediction mechanism to synthesize goal-conditioned visual foresight as explicit planning targets. By forecasting plausible future visual states, F1 reformulates action generation as a foresight-guided inverse dynamics problem, enabling actions that implicitly achieve visual goals. To endow F1 with robust and generalizable capabilities, we propose a three-stage training recipe on an extensive dataset comprising over 330k trajectories across 136 diverse tasks. This training scheme enhances modular reasoning and equips the model with transferable visual foresight, which is critical for complex and dynamic environments. Extensive evaluations on real-world tasks and simulation benchmarks demonstrate F1 consistently outperforms existing approaches, achieving substantial gains in both task success rate and generalization ability.
>
---
#### [new 042] LiHRA: A LiDAR-Based HRI Dataset for Automated Risk Monitoring Methods
- **分类: cs.RO**

- **简介: 该论文提出LiHRA数据集，用于开发人机交互中的风险监测方法。针对工业协作机器人缺乏高质量数据的问题，构建包含LiDAR点云、人体关键点和机器人状态的多模态数据集，支持实时风险评估与安全策略研究。**

- **链接: [http://arxiv.org/pdf/2509.06597v1](http://arxiv.org/pdf/2509.06597v1)**

> **作者:** Frederik Plahl; Georgios Katranis; Ilshat Mamaev; Andrey Morozov
>
> **备注:** Preprint of final paper that will appear in the Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** We present LiHRA, a novel dataset designed to facilitate the development of automated, learning-based, or classical risk monitoring (RM) methods for Human-Robot Interaction (HRI) scenarios. The growing prevalence of collaborative robots in industrial environments has increased the need for reliable safety systems. However, the lack of high-quality datasets that capture realistic human-robot interactions, including potentially dangerous events, slows development. LiHRA addresses this challenge by providing a comprehensive, multi-modal dataset combining 3D LiDAR point clouds, human body keypoints, and robot joint states, capturing the complete spatial and dynamic context of human-robot collaboration. This combination of modalities allows for precise tracking of human movement, robot actions, and environmental conditions, enabling accurate RM during collaborative tasks. The LiHRA dataset covers six representative HRI scenarios involving collaborative and coexistent tasks, object handovers, and surface polishing, with safe and hazardous versions of each scenario. In total, the data set includes 4,431 labeled point clouds recorded at 10 Hz, providing a rich resource for training and benchmarking classical and AI-driven RM algorithms. Finally, to demonstrate LiHRA's utility, we introduce an RM method that quantifies the risk level in each scenario over time. This method leverages contextual information, including robot states and the dynamic model of the robot. With its combination of high-resolution LiDAR data, precise human tracking, robot state data, and realistic collision events, LiHRA offers an essential foundation for future research into real-time RM and adaptive safety strategies in human-robot workspaces.
>
---
#### [new 043] A Hybrid TDMA/CSMA Protocol for Time-Sensitive Traffic in Robot Applications
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出一种兼容IEEE 802.11的混合TDMA/CSMA协议，解决机器人应用中高负载下关键任务通信延迟与碰撞问题。通过时隙同步、动态TDMA分配和信标保护机制，显著降低丢包率与轨迹误差，提升实时性与安全性。**

- **链接: [http://arxiv.org/pdf/2509.06119v1](http://arxiv.org/pdf/2509.06119v1)**

> **作者:** Shiqi Xu; Lihao Zhang; Yuyang Du; Qun Yang; Soung Chang Liew
>
> **摘要:** Recent progress in robotics has underscored the demand for real-time control in applications such as manufacturing, healthcare, and autonomous systems, where the timely delivery of mission-critical commands under heterogeneous robotic traffic is paramount for operational efficacy and safety. In these scenarios, mission-critical traffic follows a strict deadline-constrained communication pattern: commands must arrive within defined QoS deadlines, otherwise late arrivals can degrade performance or destabilize control loops.In this work, we demonstrate on a real-time SDR platform that CSMA, widely adopted in robotic communications,suffers severe degradation under high robot traffic loads, with contention-induced collisions and delays disrupting the on-time arrival of mission-critical packets. To address this problem, we propose an IEEE 802.11-compatible hybrid TDMA/CSMA protocol that combines TDMA's deterministic slot scheduling with CSMA's adaptability for heterogeneous robot traffic.The protocol achieves collision-free, low-latency mission-critical command delivery and IEEE 802.11 compatibility through the synergistic integration of sub-microsecond PTP-based slot synchronization-essential for establishing precise timing for TDMA, a three-session superframe with dynamic TDMA allocation for structured and adaptable traffic management,and beacon-NAV protection to preemptively secure these critical communication sessions from interference. Emulation experiments on real-time SDR testbed and Robot Operating System (ROS) simulation show that the proposed protocol reduces missed-deadline errors by 93% compared to the CSMA baseline. In high-speed robot path-tracking ROS simulations, the protocol lowers Root Mean Square (RMS) trajectory error by up to 90% compared with a CSMA baseline, all while maintaining throughput for non-critical traffic within +-2%.
>
---
#### [new 044] A Robust Approach for LiDAR-Inertial Odometry Without Sensor-Specific Modeling
- **分类: cs.RO**

- **简介: 该论文提出一种无需传感器特定建模的鲁棒LiDAR-惯性里程计方法，通过简化IMU运动模型和直接扫描配准提升性能，并在多种数据集上验证其鲁棒性。属于机器人导航中的里程计任务，解决跨传感器和场景的适应性问题。**

- **链接: [http://arxiv.org/pdf/2509.06593v1](http://arxiv.org/pdf/2509.06593v1)**

> **作者:** Meher V. R. Malladi; Tiziano Guadagnino; Luca Lobefaro; Cyrill Stachniss
>
> **摘要:** Accurate odometry is a critical component in a robotic navigation stack, and subsequent modules such as planning and control often rely on an estimate of the robot's motion. Sensor-based odometry approaches should be robust across sensor types and deployable in different target domains, from solid-state LiDARs mounted on cars in urban-driving scenarios to spinning LiDARs on handheld packages used in unstructured natural environments. In this paper, we propose a robust LiDAR-inertial odometry system that does not rely on sensor-specific modeling. Sensor fusion techniques for LiDAR and inertial measurement unit (IMU) data typically integrate IMU data iteratively in a Kalman filter or use pre-integration in a factor graph framework, combined with LiDAR scan matching often exploiting some form of feature extraction. We propose an alternative strategy that only requires a simplified motion model for IMU integration and directly registers LiDAR scans in a scan-to-map approach. Our approach allows us to impose a novel regularization on the LiDAR registration, improving the overall odometry performance. We detail extensive experiments on a number of datasets covering a wide array of commonly used robotic sensors and platforms. We show that our approach works with the exact same configuration in all these scenarios, demonstrating its robustness. We have open-sourced our implementation so that the community can build further on our work and use it in their navigation stacks.
>
---
#### [new 045] DCReg: Decoupled Characterization for Efficient Degenerate LiDAR Registration
- **分类: cs.RO**

- **简介: 论文提出DCReg框架，解决LiDAR点云配准在退化环境下的不稳定问题。通过Schur补分解、子空间特征分析和预处理策略，提升配准精度与效率，适用于机器人感知与导航任务。**

- **链接: [http://arxiv.org/pdf/2509.06285v1](http://arxiv.org/pdf/2509.06285v1)**

> **作者:** Xiangcheng Hu; Xieyuanli Chen; Mingkai Jia; Jin Wu; Ping Tan; Steven L. Waslander
>
> **备注:** 24 pages, 19 figures, 9 tables
>
> **摘要:** LiDAR point cloud registration is fundamental to robotic perception and navigation. However, in geometrically degenerate or narrow environments, registration problems become ill-conditioned, leading to unstable solutions and degraded accuracy. While existing approaches attempt to handle these issues, they fail to address the core challenge: accurately detection, interpret, and resolve this ill-conditioning, leading to missed detections or corrupted solutions. In this study, we introduce DCReg, a principled framework that systematically addresses the ill-conditioned registration problems through three integrated innovations. First, DCReg achieves reliable ill-conditioning detection by employing a Schur complement decomposition to the hessian matrix. This technique decouples the registration problem into clean rotational and translational subspaces, eliminating coupling effects that mask degeneracy patterns in conventional analyses. Second, within these cleanly subspaces, we develop quantitative characterization techniques that establish explicit mappings between mathematical eigenspaces and physical motion directions, providing actionable insights about which specific motions lack constraints. Finally, leveraging this clean subspace, we design a targeted mitigation strategy: a novel preconditioner that selectively stabilizes only the identified ill-conditioned directions while preserving all well-constrained information in observable space. This enables efficient and robust optimization via the Preconditioned Conjugate Gradient method with a single physical interpretable parameter. Extensive experiments demonstrate DCReg achieves at least 20% - 50% improvement in localization accuracy and 5-100 times speedup over state-of-the-art methods across diverse environments. Our implementation will be available at https://github.com/JokerJohn/DCReg.
>
---
#### [new 046] Long-Horizon Visual Imitation Learning via Plan and Code Reflection
- **分类: cs.RO; cs.AI; cs.LG; I.2.9; I.2.10**

- **简介: 该论文属于视觉模仿学习任务，旨在解决长时序复杂动作序列的学习问题。提出包含计划与代码反思模块的新框架，提升动作计划与代码生成的准确性，并引入LongVILBench基准进行评估。**

- **链接: [http://arxiv.org/pdf/2509.05368v1](http://arxiv.org/pdf/2509.05368v1)**

> **作者:** Quan Chen; Chenrui Shi; Qi Chen; Yuwei Wu; Zhi Gao; Xintong Zhang; Rui Gao; Kun Wu; Yunde Jia
>
> **备注:** 9 pages, 4 figures. Submitted to AAAI 2026
>
> **摘要:** Learning from long-horizon demonstrations with complex action sequences presents significant challenges for visual imitation learning, particularly in understanding temporal relationships of actions and spatial relationships between objects. In this paper, we propose a new agent framework that incorporates two dedicated reflection modules to enhance both plan and code generation. The plan generation module produces an initial action sequence, which is then verified by the plan reflection module to ensure temporal coherence and spatial alignment with the demonstration video. The code generation module translates the plan into executable code, while the code reflection module verifies and refines the generated code to ensure correctness and consistency with the generated plan. These two reflection modules jointly enable the agent to detect and correct errors in both the plan generation and code generation, improving performance in tasks with intricate temporal and spatial dependencies. To support systematic evaluation, we introduce LongVILBench, a benchmark comprising 300 human demonstrations with action sequences of up to 18 steps. LongVILBench emphasizes temporal and spatial complexity across multiple task types. Experimental results demonstrate that existing methods perform poorly on this benchmark, whereas our new framework establishes a strong baseline for long-horizon visual imitation learning.
>
---
#### [new 047] T-araVLN: Translator for Agricultural Robotic Agents on Vision-and-Language Navigation
- **分类: cs.RO**

- **简介: 该论文提出T-araVLN方法，解决农业机器人对复杂自然语言指令理解不足的问题。通过指令翻译模块提升指令精度，在A2A基准测试中显著提升导航成功率和减少偏离距离，属于视觉-语言导航任务。**

- **链接: [http://arxiv.org/pdf/2509.06644v1](http://arxiv.org/pdf/2509.06644v1)**

> **作者:** Xiaobei Zhao; Xingqi Lyu; Xiang Li
>
> **摘要:** Agricultural robotic agents have been becoming powerful helpers in a wide range of agricultural tasks, nevertheless, still heavily rely on manual operation or untransportable railway for movement. The AgriVLN method and the A2A benchmark pioneeringly extend Vision-and-Language Navigation (VLN) to the agricultural domain, enabling agents navigate to the target position following the natural language instructions. AgriVLN effectively understands the simple instructions, however, often misunderstands the complicated instructions. To bridge this gap, we propose the method of Translator for Agricultural Robotic Agents on Vision-and-Language Navigation (T-araVLN), in which the Instruction Translator module translates the original instruction to be both refined and precise. Being evaluated on the A2A benchmark, our T-araVLN effectively improves SR from 0.47 to 0.63 and reduces NE from 2.91m to 2.28m, demonstrating the state-of-the-art performance in the agricultural domain. Code: https://github.com/AlexTraveling/T-araVLN.
>
---
#### [new 048] Towards bridging the gap: Systematic sim-to-real transfer for diverse legged robots
- **分类: cs.RO**

- **简介: 论文提出一种系统化的仿真到现实迁移框架，用于多足机器人。解决控制器在仿真中训练难以迁移到现实的问题，通过集成强化学习与物理能量模型，提升运动鲁棒性与能效，实现可靠策略迁移并降低能耗。**

- **链接: [http://arxiv.org/pdf/2509.06342v1](http://arxiv.org/pdf/2509.06342v1)**

> **作者:** Filip Bjelonic; Fabian Tischhauser; Marco Hutter
>
> **备注:** Submitted to The International Journal of Robotics Research (IJRR), 25 Figures, 7 Tables, Open Source Data available at ETH Research Collection. Open Source software available soon
>
> **摘要:** Legged robots must achieve both robust locomotion and energy efficiency to be practical in real-world environments. Yet controllers trained in simulation often fail to transfer reliably, and most existing approaches neglect actuator-specific energy losses or depend on complex, hand-tuned reward formulations. We propose a framework that integrates sim-to-real reinforcement learning with a physics-grounded energy model for permanent magnet synchronous motors. The framework requires a minimal parameter set to capture the simulation-to-reality gap and employs a compact four-term reward with a first-principle-based energetic loss formulation that balances electrical and mechanical dissipation. We evaluate and validate the approach through a bottom-up dynamic parameter identification study, spanning actuators, full-robot in-air trajectories and on-ground locomotion. The framework is tested on three primary platforms and deployed on ten additional robots, demonstrating reliable policy transfer without randomization of dynamic parameters. Our method improves energetic efficiency over state-of-the-art methods, achieving a 32 percent reduction in the full Cost of Transport of ANYmal (value 1.27). All code, models, and datasets will be released.
>
---
#### [new 049] Evaluating Magic Leap 2 Tool Tracking for AR Sensor Guidance in Industrial Inspections
- **分类: cs.RO; cs.HC; cs.MM**

- **简介: 论文评估Magic Leap 2控制器在工业检测中的AR工具追踪性能，解决商业AR硬件缺乏公开基准的问题。通过机器人臂和光学系统测试静态与动态表现，提供定量基准与可移植的评估方法，用于判断其在工业传感器引导任务中的适用性。**

- **链接: [http://arxiv.org/pdf/2509.05391v1](http://arxiv.org/pdf/2509.05391v1)**

> **作者:** Christian Masuhr; Julian Koch; Thorsten Schüppstuhl
>
> **摘要:** Rigorous evaluation of commercial Augmented Reality (AR) hardware is crucial, yet public benchmarks for tool tracking on modern Head-Mounted Displays (HMDs) are limited. This paper addresses this gap by systematically assessing the Magic Leap 2 (ML2) controllers tracking performance. Using a robotic arm for repeatable motion (EN ISO 9283) and an optical tracking system as ground truth, our protocol evaluates static and dynamic performance under various conditions, including realistic paths from a hydrogen leak inspection use case. The results provide a quantitative baseline of the ML2 controller's accuracy and repeatability and present a robust, transferable evaluation methodology. The findings provide a basis to assess the controllers suitability for the inspection use case and similar industrial sensor-based AR guidance tasks.
>
---
#### [new 050] Advancing Resource Extraction Systems in Martian Volcanic Terrain: Rover Design, Power Consumption and Hazard Analysis
- **分类: astro-ph.IM; astro-ph.EP; cs.RO; physics.space-ph**

- **简介: 论文提出火星火山地形资源开采方案，设计适应复杂地形的中型探测车，分析能源需求与运输方式，解决环境挑战与机械稳定性问题，实现可持续资源利用。**

- **链接: [http://arxiv.org/pdf/2509.06103v1](http://arxiv.org/pdf/2509.06103v1)**

> **作者:** Divij Gupta; Arkajit Aich
>
> **备注:** 23 pages, 5 figures
>
> **摘要:** This study proposes a schematic plan for in-situ resource utilization (ISRU) in Martian volcanic terrains. The work investigated the complexity of volcanic terrains and Martian environmental hazards and suggested comprehensive engineering strategies to overcome the odds and establish a successful mining program in Martian volcanic regions. Slope stabilization methods - such as terracing and anchored drilling rigs - with terrain-adaptive rovers capable of autonomous operations on steep unstable slopes has been suggested as feasible solutions to navigate the complex geological terrains of Martian volcanoes. The mid range rover design with a mass of approximately 2.1 t, proposed here for mining operations, incorporates a six-wheel rocker-bogie suspension, anchoring-enabled drilling arm, dust-mitigation solar arrays, and advanced sensing systems for hazard detection and navigation. A comparative analysis regarding choice of roads and rails for building transport infrastructure has also been performed. We have also looked into the energy requirement of the rover to work under extreme environmental conditions of Mars and suggested a combination of solar and nuclear power to account for the huge energy requirements of sustained operations on Mars. The results demonstrate that mission success in these environments depends on integrating mechanical resilience, environmental adaptability, and operational autonomy, enabling sustainable access to resources in one of Mars' most geologically challenging settings.
>
---
#### [new 051] VehicleWorld: A Highly Integrated Multi-Device Environment for Intelligent Vehicle Interaction
- **分类: cs.AI; cs.CL; cs.RO**

- **简介: 该论文提出VehicleWorld，一个集成多设备的智能汽车交互环境，解决传统函数调用效率低的问题，提出基于状态的函数调用（SFC）方法，提升执行准确性和效率。**

- **链接: [http://arxiv.org/pdf/2509.06736v1](http://arxiv.org/pdf/2509.06736v1)**

> **作者:** Jie Yang; Jiajun Chen; Zhangyue Yin; Shuo Chen; Yuxin Wang; Yiran Guo; Yuan Li; Yining Zheng; Xuanjing Huang; Xipeng Qiu
>
> **摘要:** Intelligent vehicle cockpits present unique challenges for API Agents, requiring coordination across tightly-coupled subsystems that exceed typical task environments' complexity. Traditional Function Calling (FC) approaches operate statelessly, requiring multiple exploratory calls to build environmental awareness before execution, leading to inefficiency and limited error recovery. We introduce VehicleWorld, the first comprehensive environment for the automotive domain, featuring 30 modules, 250 APIs, and 680 properties with fully executable implementations that provide real-time state information during agent execution. This environment enables precise evaluation of vehicle agent behaviors across diverse, challenging scenarios. Through systematic analysis, we discovered that direct state prediction outperforms function calling for environmental control. Building on this insight, we propose State-based Function Call (SFC), a novel approach that maintains explicit system state awareness and implements direct state transitions to achieve target conditions. Experimental results demonstrate that SFC significantly outperforms traditional FC approaches, achieving superior execution accuracy and reduced latency. We have made all implementation code publicly available on Github https://github.com/OpenMOSS/VehicleWorld.
>
---
#### [new 052] Stereovision Image Processing for Planetary Navigation Maps with Semi-Global Matching and Superpixel Segmentation
- **分类: astro-ph.IM; astro-ph.EP; cs.CV; cs.RO**

- **简介: 论文提出一种结合半全局匹配（SGM）与超像素分割的立体视觉图像处理方法，用于生成火星探测器导航地形图。旨在解决传统方法在低纹理、遮挡和重复模式下的深度估计问题，提升地形建模精度与一致性，适用于行星探索任务。**

- **链接: [http://arxiv.org/pdf/2509.05645v1](http://arxiv.org/pdf/2509.05645v1)**

> **作者:** Yan-Shan Lu; Miguel Arana-Catania; Saurabh Upadhyay; Leonard Felicetti
>
> **备注:** 8 pages, 6 figures, 2 tables. ESA ASTRA 2025
>
> **摘要:** Mars exploration requires precise and reliable terrain models to ensure safe rover navigation across its unpredictable and often hazardous landscapes. Stereoscopic vision serves a critical role in the rover's perception, allowing scene reconstruction by generating precise depth maps through stereo matching. State-of-the-art Martian planetary exploration uses traditional local block-matching, aggregates cost over square windows, and refines disparities via smoothness constraints. However, this method often struggles with low-texture images, occlusion, and repetitive patterns because it considers only limited neighbouring pixels and lacks a wider understanding of scene context. This paper uses Semi-Global Matching (SGM) with superpixel-based refinement to mitigate the inherent block artefacts and recover lost details. The approach balances the efficiency and accuracy of SGM and adds context-aware segmentation to support more coherent depth inference. The proposed method has been evaluated in three datasets with successful results: In a Mars analogue, the terrain maps obtained show improved structural consistency, particularly in sloped or occlusion-prone regions. Large gaps behind rocks, which are common in raw disparity outputs, are reduced, and surface details like small rocks and edges are captured more accurately. Another two datasets, evaluated to test the method's general robustness and adaptability, show more precise disparity maps and more consistent terrain models, better suited for the demands of autonomous navigation on Mars, and competitive accuracy across both non-occluded and full-image error metrics. This paper outlines the entire terrain modelling process, from finding corresponding features to generating the final 2D navigation maps, offering a complete pipeline suitable for integration in future planetary exploration missions.
>
---
#### [new 053] LiDAR-BIND-T: Improving SLAM with Temporally Consistent Cross-Modal LiDAR Reconstruction
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出LiDAR-BIND-T，改进SLAM任务中的多模态融合，解决时空一致性问题。通过时序嵌入、运动对齐损失和时序融合模块，提升轨迹精度与地图质量，增强SLAM系统的鲁棒性与性能。**

- **链接: [http://arxiv.org/pdf/2509.05728v1](http://arxiv.org/pdf/2509.05728v1)**

> **作者:** Niels Balemans; Ali Anwar; Jan Steckel; Siegfried Mercelis
>
> **摘要:** This paper extends LiDAR-BIND, a modular multi-modal fusion framework that binds heterogeneous sensors (radar, sonar) to a LiDAR-defined latent space, with mechanisms that explicitly enforce temporal consistency. We introduce three contributions: (i) temporal embedding similarity that aligns consecutive latents, (ii) a motion-aligned transformation loss that matches displacement between predictions and ground truth LiDAR, and (iii) windows temporal fusion using a specialised temporal module. We further update the model architecture to better preserve spatial structure. Evaluations on radar/sonar-to-LiDAR translation demonstrate improved temporal and spatial coherence, yielding lower absolute trajectory error and better occupancy map accuracy in Cartographer-based SLAM (Simultaneous Localisation and Mapping). We propose different metrics based on the Fr\'echet Video Motion Distance (FVMD) and a correlation-peak distance metric providing practical temporal quality indicators to evaluate SLAM performance. The proposed temporal LiDAR-BIND, or LiDAR-BIND-T, maintains plug-and-play modality fusion while substantially enhancing temporal stability, resulting in improved robustness and performance for downstream SLAM.
>
---
#### [new 054] SpecPrune-VLA: Accelerating Vision-Language-Action Models via Action-Aware Self-Speculative Pruning
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉-语言-动作（VLA）模型加速任务，旨在解决现有剪枝方法因忽略全局上下文导致的成功率下降问题。提出SpecPrune-VLA方法，结合局部与全局信息进行两阶段剪枝，实现高效加速且成功率损失极小。**

- **链接: [http://arxiv.org/pdf/2509.05614v1](http://arxiv.org/pdf/2509.05614v1)**

> **作者:** Hanzhen Wang; Jiaming Xu; Jiayi Pan; Yongkang Zhou; Guohao Dai
>
> **备注:** 8pages, 10 figures,
>
> **摘要:** Pruning accelerates compute-bound models by reducing computation. Recently applied to Vision-Language-Action (VLA) models, existing methods prune tokens using only local info from current action, ignoring global context from prior actions, causing >20% success rate drop and limited speedup. We observe high similarity across consecutive actions and propose leveraging both local (current) and global (past) info for smarter token selection. We introduce SpecPrune-VLA, a training-free method with two-level pruning and heuristic control: (1) Static pruning at action level: uses global history and local context to reduce visual tokens per action; (2) Dynamic pruning at layer level: prunes tokens per layer based on layer-specific importance; (3) Lightweight action-aware controller: classifies actions as coarse/fine-grained (by speed), adjusting pruning aggressiveness since fine-grained actions are pruning-sensitive. Experiments on LIBERO show SpecPrune-VLA achieves 1.46 times speedup on NVIDIA A800 and 1.57 times on NVIDIA GeForce RTX 3090 vs. OpenVLA-OFT, with negligible success rate loss.
>
---
#### [new 055] Anticipatory Fall Detection in Humans with Hybrid Directed Graph Neural Networks and Long Short-Term Memory
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出一种结合动态图神经网络与LSTM的混合模型，用于提前检测人类跌倒。通过解耦运动预测与步态分类任务，实现对稳定、过渡和跌倒状态的识别，提升预测准确率，适用于辅助机器人系统。**

- **链接: [http://arxiv.org/pdf/2509.05337v1](http://arxiv.org/pdf/2509.05337v1)**

> **作者:** Younggeol Cho; Gokhan Solak; Olivia Nocentini; Marta Lorenzini; Andrea Fortuna; Arash Ajoudani
>
> **备注:** Presented at IEEE RO-MAN 2025
>
> **摘要:** Detecting and preventing falls in humans is a critical component of assistive robotic systems. While significant progress has been made in detecting falls, the prediction of falls before they happen, and analysis of the transient state between stability and an impending fall remain unexplored. In this paper, we propose a anticipatory fall detection method that utilizes a hybrid model combining Dynamic Graph Neural Networks (DGNN) with Long Short-Term Memory (LSTM) networks that decoupled the motion prediction and gait classification tasks to anticipate falls with high accuracy. Our approach employs real-time skeletal features extracted from video sequences as input for the proposed model. The DGNN acts as a classifier, distinguishing between three gait states: stable, transient, and fall. The LSTM-based network then predicts human movement in subsequent time steps, enabling early detection of falls. The proposed model was trained and validated using the OUMVLP-Pose and URFD datasets, demonstrating superior performance in terms of prediction error and recognition accuracy compared to models relying solely on DGNN and models from literature. The results indicate that decoupling prediction and classification improves performance compared to addressing the unified problem using only the DGNN. Furthermore, our method allows for the monitoring of the transient state, offering valuable insights that could enhance the functionality of advanced assistance systems.
>
---
#### [new 056] InterAct: A Large-Scale Dataset of Dynamic, Expressive and Interactive Activities between Two People in Daily Scenarios
- **分类: cs.CV; cs.AI; cs.LG; cs.MA; cs.RO; I.5.4**

- **简介: 该论文提出InterAct数据集，用于捕捉两人日常互动中的动态行为。解决传统方法忽略身体动态和长期交互的问题，通过多模态数据和扩散模型生成互动表情与动作，推动相关研究。**

- **链接: [http://arxiv.org/pdf/2509.05747v1](http://arxiv.org/pdf/2509.05747v1)**

> **作者:** Leo Ho; Yinghao Huang; Dafei Qin; Mingyi Shi; Wangpok Tse; Wei Liu; Junichi Yamagishi; Taku Komura
>
> **备注:** The first two authors contributed equally to this work
>
> **摘要:** We address the problem of accurate capture of interactive behaviors between two people in daily scenarios. Most previous works either only consider one person or solely focus on conversational gestures of two people, assuming the body orientation and/or position of each actor are constant or barely change over each interaction. In contrast, we propose to simultaneously model two people's activities, and target objective-driven, dynamic, and semantically consistent interactions which often span longer duration and cover bigger space. To this end, we capture a new multi-modal dataset dubbed InterAct, which is composed of 241 motion sequences where two people perform a realistic and coherent scenario for one minute or longer over a complete interaction. For each sequence, two actors are assigned different roles and emotion labels, and collaborate to finish one task or conduct a common interaction activity. The audios, body motions, and facial expressions of both persons are captured. InterAct contains diverse and complex motions of individuals and interesting and relatively long-term interaction patterns barely seen before. We also demonstrate a simple yet effective diffusion-based method that estimates interactive face expressions and body motions of two people from speech inputs. Our method regresses the body motions in a hierarchical manner, and we also propose a novel fine-tuning mechanism to improve the lip accuracy of facial expressions. To facilitate further research, the data and code is made available at https://hku-cg.github.io/interact/ .
>
---
#### [new 057] "It was Tragic": Exploring the Impact of a Robot's Shutdown
- **分类: cs.HC; cs.RO**

- **简介: 该论文研究机器人关机动作对人类感知的影响。通过对比两种关机方式，发现设计感强的“入睡”动作使机器人更受喜爱、显得更智能和生动。任务是探讨机器人边缘交互设计的重要性，解决如何通过关机动作提升人机互动体验的问题。**

- **链接: [http://arxiv.org/pdf/2509.06934v1](http://arxiv.org/pdf/2509.06934v1)**

> **作者:** Agam Oberlender; Hadas Erel
>
> **备注:** 8 pages, 4 figures, 1 table, submitted to IEEE RO-MAN 2025
>
> **摘要:** It is well established that people perceive robots as social entities, even when they are not designed for social interaction. We evaluated whether the social interpretation of robotic gestures should also be considered when turning off a robot. In the experiment, participants engaged in a brief preliminary neutral interaction while a robotic arm showed interest in their actions. At the end of the task, participants were asked to turn off the robotic arm under two conditions: (1) a Non-designed condition, where all of the robot's engines were immediately and simultaneously turned off, as robots typically shut down; (2) a Designed condition, where the robot's engines gradually folded inward in a motion resembling "falling asleep." Our findings revealed that all participants anthropomorphized the robot's movement when it was turned off. In the Non-designed condition, most participants interpreted the robot's turn-off movement negatively, as if the robot had "died." In the Designed condition, most participants interpreted it more neutrally, stating that the robot "went to sleep." The robot's turn-off movement also impacted its perception, leading to higher likeability, perceived intelligence, and animacy in the Designed condition. We conclude that the impact of common edge interactions, such as turning off a robot, should be carefully designed while considering people's automatic tendency to perceive robots as social entities.
>
---
#### [new 058] Quaternion Approximation Networks for Enhanced Image Classification and Oriented Object Detection
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出Quaternion Approximation Networks（QUAN），用于图像分类和定向目标检测。通过实值操作近似四元数卷积，提升几何不变性与效率，并引入IQBN和空间注意力机制。实验表明其在多个任务中参数更少、精度更高，适用于资源受限的机器人系统。**

- **链接: [http://arxiv.org/pdf/2509.05512v1](http://arxiv.org/pdf/2509.05512v1)**

> **作者:** Bryce Grant; Peng Wang
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** This paper introduces Quaternion Approximate Networks (QUAN), a novel deep learning framework that leverages quaternion algebra for rotation equivariant image classification and object detection. Unlike conventional quaternion neural networks attempting to operate entirely in the quaternion domain, QUAN approximates quaternion convolution through Hamilton product decomposition using real-valued operations. This approach preserves geometric properties while enabling efficient implementation with custom CUDA kernels. We introduce Independent Quaternion Batch Normalization (IQBN) for training stability and extend quaternion operations to spatial attention mechanisms. QUAN is evaluated on image classification (CIFAR-10/100, ImageNet), object detection (COCO, DOTA), and robotic perception tasks. In classification tasks, QUAN achieves higher accuracy with fewer parameters and faster convergence compared to existing convolution and quaternion-based models. For objection detection, QUAN demonstrates improved parameter efficiency and rotation handling over standard Convolutional Neural Networks (CNNs) while establishing the SOTA for quaternion CNNs in this downstream task. These results highlight its potential for deployment in resource-constrained robotic systems requiring rotation-aware perception and application in other domains.
>
---
#### [new 059] OccVLA: Vision-Language-Action Model with Implicit 3D Occupancy Supervision
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出OccVLA模型，解决自动驾驶中多模态模型缺乏三维空间理解的问题。通过引入隐式三维占用表示，从二维视觉输入中学习精细空间结构，实现轨迹规划与三维问答任务的先进性能。属于自动驾驶中的三维视觉语言联合推理任务。**

- **链接: [http://arxiv.org/pdf/2509.05578v1](http://arxiv.org/pdf/2509.05578v1)**

> **作者:** Ruixun Liu; Lingyu Kong; Derun Li; Hang Zhao
>
> **摘要:** Multimodal large language models (MLLMs) have shown strong vision-language reasoning abilities but still lack robust 3D spatial understanding, which is critical for autonomous driving. This limitation stems from two key challenges: (1) the difficulty of constructing accessible yet effective 3D representations without expensive manual annotations, and (2) the loss of fine-grained spatial details in VLMs due to the absence of large-scale 3D vision-language pretraining. To address these challenges, we propose OccVLA, a novel framework that integrates 3D occupancy representations into a unified multimodal reasoning process. Unlike prior approaches that rely on explicit 3D inputs, OccVLA treats dense 3D occupancy as both a predictive output and a supervisory signal, enabling the model to learn fine-grained spatial structures directly from 2D visual inputs. The occupancy predictions are regarded as implicit reasoning processes and can be skipped during inference without performance degradation, thereby adding no extra computational overhead. OccVLA achieves state-of-the-art results on the nuScenes benchmark for trajectory planning and demonstrates superior performance on 3D visual question-answering tasks, offering a scalable, interpretable, and fully vision-based solution for autonomous driving.
>
---
#### [new 060] Cumplimiento del Reglamento (UE) 2024/1689 en robótica y sistemas autónomos: una revisión sistemática de la literatura
- **分类: cs.CY; cs.AI; cs.CR; cs.RO**

- **简介: 该论文系统综述了欧盟2024/1689法规在自主机器人系统中的合规情况，分析了网络安全框架与方法。通过PRISMA协议筛选22项研究，发现法规在风险管理和加密通信方面有进展，但在可解释性、实时监督和知识库追踪等方面存在不足，提出需整合模块化方法以满足AI法案要求。**

- **链接: [http://arxiv.org/pdf/2509.05380v1](http://arxiv.org/pdf/2509.05380v1)**

> **作者:** Yoana Pita Lorenzo
>
> **备注:** in Spanish language
>
> **摘要:** This systematic literature review analyzes the current state of compliance with Regulation (EU) 2024/1689 in autonomous robotic systems, focusing on cybersecurity frameworks and methodologies. Using the PRISMA protocol, 22 studies were selected from 243 initial records across IEEE Xplore, ACM DL, Scopus, and Web of Science. Findings reveal partial regulatory alignment: while progress has been made in risk management and encrypted communications, significant gaps persist in explainability modules, real-time human oversight, and knowledge base traceability. Only 40% of reviewed solutions explicitly address transparency requirements, and 30% implement failure intervention mechanisms. The study concludes that modular approaches integrating risk, supervision, and continuous auditing are essential to meet the AI Act mandates in autonomous robotics.
>
---
#### [new 061] Investigating Location-Regularised Self-Supervised Feature Learning for Seafloor Visual Imagery
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究了利用位置信息正则化提升自监督特征学习在海底图像分析中的效果。任务为改进SSL方法以提高分类性能，通过评估六种SSL框架发现位置正则化能有效提升CNN和ViT模型的下游任务表现。**

- **链接: [http://arxiv.org/pdf/2509.06660v1](http://arxiv.org/pdf/2509.06660v1)**

> **作者:** Cailei Liang; Adrian Bodenmann; Emma J Curtis; Samuel Simmons; Kazunori Nagano; Stan Brown; Adam Riese; Blair Thornton
>
> **摘要:** High-throughput interpretation of robotically gathered seafloor visual imagery can increase the efficiency of marine monitoring and exploration. Although recent research has suggested that location metadata can enhance self-supervised feature learning (SSL), its benefits across different SSL strategies, models and seafloor image datasets are underexplored. This study evaluates the impact of location-based regularisation on six state-of-the-art SSL frameworks, which include Convolutional Neural Network (CNN) and Vision Transformer (ViT) models with varying latent-space dimensionality. Evaluation across three diverse seafloor image datasets finds that location-regularisation consistently improves downstream classification performance over standard SSL, with average F1-score gains of $4.9 \pm 4.0%$ for CNNs and $6.3 \pm 8.9%$ for ViTs, respectively. While CNNs pretrained on generic datasets benefit from high-dimensional latent representations, dataset-optimised SSL achieves similar performance across the high (512) and low (128) dimensional latent representations. Location-regularised SSL improves CNN performance over pre-trained models by $2.7 \pm 2.7%$ and $10.1 \pm 9.4%$ for high and low-dimensional latent representations, respectively. For ViTs, high-dimensionality benefits both pre-trained and dataset-optimised SSL. Although location-regularisation improves SSL performance compared to standard SSL methods, pre-trained ViTs show strong generalisation, matching the best-performing location-regularised SSL with F1-scores of $0.795 \pm 0.075$ and $0.795 \pm 0.077$, respectively. The findings highlight the value of location metadata for SSL regularisation, particularly when using low-dimensional latent representations, and demonstrate strong generalisation of high-dimensional ViTs for seafloor image analysis.
>
---
#### [new 062] Musculoskeletal simulation of limb movement biomechanics in Drosophila melanogaster
- **分类: q-bio.NC; cs.AI; cs.LG; cs.RO**

- **简介: 该论文构建了果蝇腿部的首个三维肌肉骨骼模型，用于模拟肢体运动生物力学。通过整合高分辨率影像数据与仿真环境，实现了肌肉驱动行为回放与学习策略测试，解决神经控制与肢体运动协调机制问题。**

- **链接: [http://arxiv.org/pdf/2509.06426v1](http://arxiv.org/pdf/2509.06426v1)**

> **作者:** Pembe Gizem Özdil; Chuanfang Ning; Jasper S. Phelps; Sibo Wang-Chen; Guy Elisha; Alexander Blanke; Auke Ijspeert; Pavan Ramdya
>
> **备注:** 23 pages, 11 figures
>
> **摘要:** Computational models are critical to advance our understanding of how neural, biomechanical, and physical systems interact to orchestrate animal behaviors. Despite the availability of near-complete reconstructions of the Drosophila melanogaster central nervous system, musculature, and exoskeleton, anatomically and physically grounded models of fly leg muscles are still missing. These models provide an indispensable bridge between motor neuron activity and joint movements. Here, we introduce the first 3D, data-driven musculoskeletal model of Drosophila legs, implemented in both OpenSim and MuJoCo simulation environments. Our model incorporates a Hill-type muscle representation based on high-resolution X-ray scans from multiple fixed specimens. We present a pipeline for constructing muscle models using morphological imaging data and for optimizing unknown muscle parameters specific to the fly. We then combine our musculoskeletal models with detailed 3D pose estimation data from behaving flies to achieve muscle-actuated behavioral replay in OpenSim. Simulations of muscle activity across diverse walking and grooming behaviors predict coordinated muscle synergies that can be tested experimentally. Furthermore, by training imitation learning policies in MuJoCo, we test the effect of different passive joint properties on learning speed and find that damping and stiffness facilitate learning. Overall, our model enables the investigation of motor control in an experimentally tractable model organism, providing insights into how biomechanics contribute to generation of complex limb movements. Moreover, our model can be used to control embodied artificial agents to generate naturalistic and compliant locomotion in simulated environments.
>
---
#### [new 063] Nanobot Algorithms for Treatment of Diffuse Cancer
- **分类: cs.MA; cs.RO; q-bio.QM**

- **简介: 论文研究纳米机器人在治疗弥散性癌症中的算法。任务是设计协调算法以定位并按需分配药物。提出三种算法（KM、KMA、KMAR），通过模拟验证其在不同癌灶分布下的性能，提升治疗效率与成功率。**

- **链接: [http://arxiv.org/pdf/2509.06893v1](http://arxiv.org/pdf/2509.06893v1)**

> **作者:** Noble Harasha; Nancy Lynch
>
> **备注:** Abridged abstract shown here; 34 pages, 9 figures
>
> **摘要:** Motile nanosized particles, or "nanobots", promise more effective and less toxic targeted drug delivery because of their unique scale and precision. We consider the case in which the cancer is "diffuse", dispersed such that there are multiple distinct cancer sites. We investigate the problem of a swarm of nanobots locating these sites and treating them by dropping drug payloads at the sites. To improve the success of the treatment, the drug payloads must be allocated between sites according to their "demands"; this requires extra nanobot coordination. We present a mathematical model of the behavior of the nanobot agents and of their colloidal environment. This includes a movement model for agents based upon experimental findings from actual nanoparticles in which bots noisily ascend and descend chemical gradients. We present three algorithms: The first algorithm, called KM, is the most representative of reality, with agents simply following naturally existing chemical signals that surround each cancer site. The second algorithm, KMA, includes an additional chemical payload which amplifies the existing natural signals. The third algorithm, KMAR, includes another additional chemical payload which counteracts the other signals, instead inducing negative chemotaxis in agents such that they are repelled from sites that are already sufficiently treated. We present simulation results for all algorithms across different types of cancer arrangements. For KM, we show that the treatment is generally successful unless the natural chemical signals are weak, in which case the treatment progresses too slowly. For KMA, we demonstrate a significant improvement in treatment speed but a drop in eventual success, except for concentrated cancer patterns. For KMAR, our results show great performance across all types of cancer patterns, demonstrating robustness and adaptability.
>
---
#### [new 064] Multi-Modal Camera-Based Detection of Vulnerable Road Users
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于目标检测任务，旨在解决复杂环境下弱势道路使用者（VRUs）检测难题。研究提出融合RGB与热成像的多模态检测框架，采用改进的YOLOv8模型，提升检测精度与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.06333v1](http://arxiv.org/pdf/2509.06333v1)**

> **作者:** Penelope Brown; Julie Stephany Berrio Perez; Mao Shan; Stewart Worrall
>
> **摘要:** Vulnerable road users (VRUs) such as pedestrians, cyclists, and motorcyclists represent more than half of global traffic deaths, yet their detection remains challenging in poor lighting, adverse weather, and unbalanced data sets. This paper presents a multimodal detection framework that integrates RGB and thermal infrared imaging with a fine-tuned YOLOv8 model. Training leveraged KITTI, BDD100K, and Teledyne FLIR datasets, with class re-weighting and light augmentations to improve minority-class performance and robustness, experiments show that 640-pixel resolution and partial backbone freezing optimise accuracy and efficiency, while class-weighted losses enhance recall for rare VRUs. Results highlight that thermal models achieve the highest precision, and RGB-to-thermal augmentation boosts recall, demonstrating the potential of multimodal detection to improve VRU safety at intersections.
>
---
#### [new 065] MAPF-HD: Multi-Agent Path Finding in High-Density Environments
- **分类: cs.MA; cs.RO**

- **简介: 该论文提出MAPF-HD框架，解决高密度环境下多智能体路径规划问题。传统ILP方法计算成本高，论文引入PHANS方法，通过启发式交换实现高效路径规划，适用于仓库物流等实际场景。**

- **链接: [http://arxiv.org/pdf/2509.06374v1](http://arxiv.org/pdf/2509.06374v1)**

> **作者:** Hiroya Makino; Seigo Ito
>
> **备注:** 9 pages, 12 figures
>
> **摘要:** Multi-agent path finding (MAPF) involves planning efficient paths for multiple agents to move simultaneously while avoiding collisions. In typical warehouse environments, agents are often sparsely distributed along aisles. However, increasing the agent density can improve space efficiency. When the agent density is high, we must optimize the paths not only for goal-assigned agents but also for those obstructing them. This study proposes a novel MAPF framework for high-density environments (MAPF-HD). Several studies have explored MAPF in similar settings using integer linear programming (ILP). However, ILP-based methods require substantial computation time to optimize all agent paths simultaneously. Even in small grid-based environments with fewer than $100$ cells, these computations can incur tens to hundreds of seconds. These high computational costs render these methods impractical for large-scale applications such as automated warehouses and valet parking. To address these limitations, we introduce the phased null-agent swapping (PHANS) method. PHANS employs a heuristic approach to incrementally swap positions between agents and empty vertices. This method solves the MAPF-HD problem within seconds to tens of seconds, even in large environments containing more than $700$ cells. The proposed method can potentially improve efficiency in various real-world applications such as warehouse logistics, traffic management, or crowd control. Code is available at https://github.com/ToyotaCRDL/MAPF-in-High-Density-Envs.
>
---
#### [new 066] OpenEgo: A Large-Scale Multimodal Egocentric Dataset for Dexterous Manipulation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出OpenEgo数据集，用于解决模仿学习中缺乏精细手部动作描述的问题。其整合多数据源，提供标准化手部姿态和动作描述，支持视觉-语言-动作学习研究。**

- **链接: [http://arxiv.org/pdf/2509.05513v1](http://arxiv.org/pdf/2509.05513v1)**

> **作者:** Ahad Jawaid; Yu Xiang
>
> **备注:** 4 pages, 1 figure
>
> **摘要:** Egocentric human videos provide scalable demonstrations for imitation learning, but existing corpora often lack either fine-grained, temporally localized action descriptions or dexterous hand annotations. We introduce OpenEgo, a multimodal egocentric manipulation dataset with standardized hand-pose annotations and intention-aligned action primitives. OpenEgo totals 1107 hours across six public datasets, covering 290 manipulation tasks in 600+ environments. We unify hand-pose layouts and provide descriptive, timestamped action primitives. To validate its utility, we train language-conditioned imitation-learning policies to predict dexterous hand trajectories. OpenEgo is designed to lower the barrier to learning dexterous manipulation from egocentric video and to support reproducible research in vision-language-action learning. All resources and instructions will be released at www.openegocentric.com.
>
---
#### [new 067] Online Clustering of Seafloor Imagery for Interpretation during Long-Term AUV Operations
- **分类: cs.CV; cs.RO**

- **简介: 论文提出一种在线聚类框架（OCF），用于实时解读水下机器人采集的海底图像，解决长期任务中无法依赖人工标注和完整数据集的问题。该方法无需监督，在保证聚类精度的同时降低计算成本，适用于自主海洋探索中的路径规划与数据总结。**

- **链接: [http://arxiv.org/pdf/2509.06678v1](http://arxiv.org/pdf/2509.06678v1)**

> **作者:** Cailei Liang; Adrian Bodenmann; Sam Fenton; Blair Thornton
>
> **摘要:** As long-endurance and seafloor-resident AUVs become more capable, there is an increasing need for extended, real-time interpretation of seafloor imagery to enable adaptive missions and optimise communication efficiency. Although offline image analysis methods are well established, they rely on access to complete datasets and human-labelled examples to manage the strong influence of environmental and operational conditions on seafloor image appearance-requirements that cannot be met in real-time settings. To address this, we introduce an online clustering framework (OCF) capable of interpreting seafloor imagery without supervision, which is designed to operate in real-time on continuous data streams in a scalable, adaptive, and self-consistent manner. The method enables the efficient review and consolidation of common patterns across the entire data history in constant time by identifying and maintaining a set of representative samples that capture the evolving feature distribution, supporting dynamic cluster merging and splitting without reprocessing the full image history. We evaluate the framework on three diverse seafloor image datasets, analysing the impact of different representative sampling strategies on both clustering accuracy and computational cost. The OCF achieves the highest average F1 score of 0.68 across the three datasets among all comparative online clustering approaches, with a standard deviation of 3% across three distinct survey trajectories, demonstrating its superior clustering capability and robustness to trajectory variation. In addition, it maintains consistently lower and bounded computational time as the data volume increases. These properties are beneficial for generating survey data summaries and supporting informative path planning in long-term, persistent autonomous marine exploration.
>
---
#### [new 068] Programming tension in 3D printed networks inspired by spiderwebs
- **分类: cs.GR; cs.RO**

- **简介: 论文提出一种3D打印算法，用于制造具有编程张力梯度的网络结构，解决传统方法中因压缩导致的张力误差问题。通过力密度法定义张力，并优化顶点位置以实现精准打印，验证显示误差低于1.0%。**

- **链接: [http://arxiv.org/pdf/2509.05855v1](http://arxiv.org/pdf/2509.05855v1)**

> **作者:** Thijs Masmeijer; Caleb Swain; Jeff Hill; Ed Habtour
>
> **摘要:** Each element in tensioned structural networks -- such as tensegrity, architectural fabrics, or medical braces/meshes -- requires a specific tension level to achieve and maintain the desired shape, stability, and compliance. These structures are challenging to manufacture, 3D print, or assemble because flattening the network during fabrication introduces multiplicative inaccuracies in the network's final tension gradients. This study overcomes this challenge by offering a fabrication algorithm for direct 3D printing of such networks with programmed tension gradients, an approach analogous to the spinning of spiderwebs. The algorithm: (i) defines the desired network and prescribes its tension gradients using the force density method; (ii) converts the network into an unstretched counterpart by numerically optimizing vertex locations toward target element lengths and converting straight elements into arcs to resolve any remaining error; and (iii) decomposes the network into printable toolpaths; Optional additional steps are: (iv) flattening curved 2D networks or 3D networks to ensure 3D printing compatibility; and (v) automatically resolving any unwanted crossings introduced by the flattening process. The proposed method is experimentally validated using 2D unit cells of viscoelastic filaments, where accurate tension gradients are achieved with an average element strain error of less than 1.0\%. The method remains effective for networks with element minimum length and maximum stress of 5.8 mm and 7.3 MPa, respectively. The method is used to demonstrate the fabrication of three complex cases: a flat spiderweb, a curved mesh, and a tensegrity system. The programmable tension gradient algorithm can be utilized to produce compact, integrated cable networks, enabling novel applications such as moment-exerting structures in medical braces and splints.
>
---
#### [new 069] Event Spectroscopy: Event-based Multispectral and Depth Sensing using Structured Light
- **分类: cs.CV; cs.RO**

- **简介: 论文提出一种基于事件的多光谱与深度感知系统，用于无人机在森林环境中的导航与数据采集。系统通过结构光实现高精度深度重建和可控波段的光谱成像，提升环境感知能力，解决传统方法在延迟、深度分辨率和光照依赖性方面的不足。**

- **链接: [http://arxiv.org/pdf/2509.06741v1](http://arxiv.org/pdf/2509.06741v1)**

> **作者:** Christian Geckeler; Niklas Neugebauer; Manasi Muglikar; Davide Scaramuzza; Stefano Mintchev
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Uncrewed aerial vehicles (UAVs) are increasingly deployed in forest environments for tasks such as environmental monitoring and search and rescue, which require safe navigation through dense foliage and precise data collection. Traditional sensing approaches, including passive multispectral and RGB imaging, suffer from latency, poor depth resolution, and strong dependence on ambient light - especially under forest canopies. In this work, we present a novel event spectroscopy system that simultaneously enables high-resolution, low-latency depth reconstruction and multispectral imaging using a single sensor. Depth is reconstructed using structured light, and by modulating the wavelength of the projected structured light, our system captures spectral information in controlled bands between 650 nm and 850 nm. We demonstrate up to $60\%$ improvement in RMSE over commercial depth sensors and validate the spectral accuracy against a reference spectrometer and commercial multispectral cameras, demonstrating comparable performance. A portable version limited to RGB (3 wavelengths) is used to collect real-world depth and spectral data from a Masoala Rainforest. We demonstrate the use of this prototype for color image reconstruction and material differentiation between leaves and branches using spectral and depth data. Our results show that adding depth (available at no extra effort with our setup) to material differentiation improves the accuracy by over $30\%$ compared to color-only method. Our system, tested in both lab and real-world rainforest environments, shows strong performance in depth estimation, RGB reconstruction, and material differentiation - paving the way for lightweight, integrated, and robust UAV perception and data collection in complex natural environments.
>
---
## 更新

#### [replaced 001] ER-LoRA: Effective-Rank Guided Adaptation for Weather-Generalized Depth Estimation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2509.00665v2](http://arxiv.org/pdf/2509.00665v2)**

> **作者:** Weilong Yan; Xin Zhang; Robby T. Tan
>
> **摘要:** Monocular depth estimation under adverse weather conditions (e.g.\ rain, fog, snow, and nighttime) remains highly challenging due to the lack of reliable ground truth and the difficulty of learning from unlabeled real-world data. Existing methods often rely on synthetic adverse data with pseudo-labels, which suffer from domain gaps, or employ self-supervised learning, which violates photometric assumptions in adverse scenarios. In this work, we propose to achieve weather-generalized depth estimation by Parameter-Efficient Fine-Tuning (PEFT) of Vision Foundation Models (VFMs), using only a small amount of high-visibility (normal) data. While PEFT has shown strong performance in semantic tasks such as segmentation, it remains underexplored for geometry -- centric tasks like depth estimation -- especially in terms of balancing effective adaptation with the preservation of pretrained knowledge. To this end, we introduce the Selecting-Tuning-Maintaining (STM) strategy, which structurally decomposes the pretrained weights of VFMs based on two kinds of effective ranks (entropy-rank and stable-rank). In the tuning phase, we adaptively select the proper rank number as well as the task-aware singular directions for initialization, based on the entropy-rank and full-tuned weight; while in the maintaining stage, we enforce a principal direction regularization based on the stable-rank. This design guarantees flexible task adaptation while preserving the strong generalization capability of the pretrained VFM. Extensive experiments on four real-world benchmarks across diverse weather conditions demonstrate that STM not only outperforms existing PEFT methods and full fine-tuning but also surpasses methods trained with adverse synthetic data, and even the depth foundation model
>
---
#### [replaced 002] Bridging the Sim2Real Gap: Vision Encoder Pre-Training for Visuomotor Policy Transfer
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.16389v2](http://arxiv.org/pdf/2501.16389v2)**

> **作者:** Yash Yardi; Samuel Biruduganti; Lars Ankile
>
> **备注:** 6 pages, 4 figures, 1 table, GitHub: https://github.com/yyardi/Bridging-the-Sim2Real-Gap
>
> **摘要:** Simulation offers a scalable and efficient alternative to real-world data collection for learning visuomotor robotic policies. However, the simulation-to-reality, or Sim2Real distribution shift -- introduced by employing simulation-trained policies in real-world environments -- frequently prevents successful policy transfer. We present an offline framework to evaluate the performance of using large-scale pre-trained vision encoders to address the Sim2Real gap. We examine a diverse collection of encoders, assessing their ability to extract features necessary for robot control (Action Score) while remaining invariant to task-irrelevant environmental variations (Domain Invariance Score). Evaluating 23 encoders, we reveal patterns across architectures, pre-training datasets, and parameter scales. Our findings show that manipulation-pretrained encoders consistently achieve higher Action Scores, CNN-based encoders demonstrate stronger domain invariance than ViTs, and the best-performing models combine both properties, underscoring DIS and AS as complementary predictors of Sim2Real transferability.
>
---
#### [replaced 003] Generation of Indoor Open Street Maps for Robot Navigation from CAD Files
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.00552v2](http://arxiv.org/pdf/2507.00552v2)**

> **作者:** Jiajie Zhang; Shenrui Wu; Xu Ma; Sören Schwertfeger
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** The deployment of autonomous mobile robots is predicated on the availability of environmental maps, yet conventional generation via SLAM (Simultaneous Localization and Mapping) suffers from significant limitations in time, labor, and robustness, particularly in dynamic, large-scale indoor environments where map obsolescence can lead to critical localization failures. To address these challenges, this paper presents a complete and automated system for converting architectural Computer-Aided Design (CAD) files into a hierarchical topometric OpenStreetMap (OSM) representation, tailored for robust life-long robot navigation. Our core methodology involves a multi-stage pipeline that first isolates key structural layers from the raw CAD data and then employs an AreaGraph-based topological segmentation to partition the building layout into a hierarchical graph of navigable spaces. This process yields a comprehensive and semantically rich map, further enhanced by automatically associating textual labels from the CAD source and cohesively merging multiple building floors into a unified, topologically-correct model. By leveraging the permanent structural information inherent in CAD files, our system circumvents the inefficiencies and fragility of SLAM, offering a practical and scalable solution for deploying robots in complex indoor spaces. The software is encapsulated within an intuitive Graphical User Interface (GUI) to facilitate practical use. The code and dataset are available at https://github.com/jiajiezhang7/osmAG-from-cad.
>
---
#### [replaced 004] Conversational Code Generation: a Case Study of Designing a Dialogue System for Generating Driving Scenarios for Testing Autonomous Vehicles
- **分类: cs.CL; cs.IR; cs.RO**

- **链接: [http://arxiv.org/pdf/2410.09829v3](http://arxiv.org/pdf/2410.09829v3)**

> **作者:** Rimvydas Rubavicius; Antonio Valerio Miceli-Barone; Alex Lascarides; Subramanian Ramamoorthy
>
> **备注:** In Proceedings of GeCoIn 2025: Generative Code Intelligence Workshop, co-located with ECAI-2025
>
> **摘要:** Cyber-physical systems like autonomous vehicles are tested in simulation before deployment, using domain-specific programs for scenario specification. To aid the testing of autonomous vehicles in simulation, we design a natural language interface, using an instruction-following large language model, to assist a non-coding domain expert in synthesising the desired scenarios and vehicle behaviours. We show that using it to convert utterances to the symbolic program is feasible, despite the very small training dataset. Human experiments show that dialogue is critical to successful simulation generation, leading to a 4.5 times higher success rate than a generation without engaging in extended conversation.
>
---
#### [replaced 005] Output-Feedback Boundary Control of Thermally and Flow-Induced Vibrations in Slender Timoshenko Beams
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.21281v2](http://arxiv.org/pdf/2503.21281v2)**

> **作者:** Chengyi Wang; Ji Wang
>
> **摘要:** This work is motivated by the engineering challenge of suppressing vibrations in turbine blades of aero engines, which often operate under extreme thermal conditions and high-Mach aerodynamic environments that give rise to complex vibration phenomena, commonly referred to as thermally-induced and flow-induced vibrations. Using Hamilton's variational principle, the system is modeled as a rotating slender Timoshenko beam under thermal and aerodynamic loads, described by a coupled system of 2*2 hyperbolic PIDEs, parabolic PDE, and ODEs, where the nonlocal terms exist in the hyperbolic PDE domain, and where the external disturbance (heat flux) flows into one boundary of the heat PDE. For the general form of such mixed systems, we present the state-feedback control design based on the PDE backstepping method, and then design an extended state observer for the unmeasurable distributed states and external disturbances using only available boundary measurements. In the resulting output-feedback closed-loop system, the state of the uncontrolled boundary, i.e., the furthest state from the control input, is proved to be exponentially convergent to zero, and all signals are proved to be uniformly ultimately bounded. Moreover, if the external disturbance vanishes, the exponential stability of the overall system is obtained. The proposed control design is validated on an aero-engine flexible blade under extreme thermal and aerodynamic conditions.
>
---
#### [replaced 006] COLLAGE: Adaptive Fusion-based Retrieval for Augmented Policy Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2508.01131v2](http://arxiv.org/pdf/2508.01131v2)**

> **作者:** Sateesh Kumar; Shivin Dass; Georgios Pavlakos; Roberto Martín-Martín
>
> **备注:** Accepted at the Conference on Robot Learning (CoRL), 2025. Project page: https://robin-lab.cs.utexas.edu/COLLAGE
>
> **摘要:** In this work, we study the problem of data retrieval for few-shot imitation learning: selecting data from a large dataset to train a performant policy for a specific task, given only a few target demonstrations. Prior methods retrieve data using a single-feature distance heuristic, assuming that the best demonstrations are those that most closely resemble the target examples in visual, semantic, or motion space. However, this approach captures only a subset of the relevant information and can introduce detrimental demonstrations, e.g., retrieving data from unrelated tasks due to similar scene layouts, or selecting similar motions from tasks with divergent goals. We present COLLAGE, a method for COLLective data AGgrEgation in few-shot imitation learning that uses an adaptive late fusion mechanism to guide the selection of relevant demonstrations based on a task-specific combination of multiple cues. COLLAGE follows a simple, flexible, and efficient recipe: it assigns weights to subsets of the dataset that are pre-selected using a single feature (e.g., appearance, shape, or language similarity), based on how well a policy trained on each subset predicts actions in the target demonstrations. These weights are then used to perform importance sampling during policy training, sampling data more densely or sparsely according to estimated relevance. COLLAGE is general and feature-agnostic, allowing it to combine any number of subsets selected by any retrieval heuristic, and to identify which subsets provide the greatest benefit for the target task. In extensive experiments, COLLAGE outperforms state-of-the-art retrieval and multi-task learning approaches by 5.1% in simulation across 10 tasks, and by 16.6% in the real world across 6 tasks, where we perform retrieval from the large-scale DROID dataset. More information at https://robin-lab.cs.utexas.edu/COLLAGE .
>
---
#### [replaced 007] QuadKAN: KAN-Enhanced Quadruped Motion Control via End-to-End Reinforcement Learning
- **分类: cs.RO; cs.AI; cs.CV; cs.SY; eess.IV; eess.SY**

- **链接: [http://arxiv.org/pdf/2508.19153v2](http://arxiv.org/pdf/2508.19153v2)**

> **作者:** Yinuo Wang; Gavin Tao
>
> **备注:** 14pages, 9 figures, Journal paper
>
> **摘要:** We address vision-guided quadruped motion control with reinforcement learning (RL) and highlight the necessity of combining proprioception with vision for robust control. We propose QuadKAN, a spline-parameterized cross-modal policy instantiated with Kolmogorov-Arnold Networks (KANs). The framework incorporates a spline encoder for proprioception and a spline fusion head for proprioception-vision inputs. This structured function class aligns the state-to-action mapping with the piecewise-smooth nature of gait, improving sample efficiency, reducing action jitter and energy consumption, and providing interpretable posture-action sensitivities. We adopt Multi-Modal Delay Randomization (MMDR) and perform end-to-end training with Proximal Policy Optimization (PPO). Evaluations across diverse terrains, including both even and uneven surfaces and scenarios with static or dynamic obstacles, demonstrate that QuadKAN achieves consistently higher returns, greater distances, and fewer collisions than state-of-the-art (SOTA) baselines. These results show that spline-parameterized policies offer a simple, effective, and interpretable alternative for robust vision-guided locomotion. A repository will be made available upon acceptance.
>
---
#### [replaced 008] ARCH: Hierarchical Hybrid Learning for Long-Horizon Contact-Rich Robotic Assembly
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.16451v2](http://arxiv.org/pdf/2409.16451v2)**

> **作者:** Jiankai Sun; Aidan Curtis; Yang You; Yan Xu; Michael Koehle; Qianzhong Chen; Suning Huang; Leonidas Guibas; Sachin Chitta; Mac Schwager; Hui Li
>
> **备注:** The Conference on Robot Learning (CoRL) 2025
>
> **摘要:** Generalizable long-horizon robotic assembly requires reasoning at multiple levels of abstraction. While end-to-end imitation learning (IL) is a promising approach, it typically requires large amounts of expert demonstration data and often struggles to achieve the high precision demanded by assembly tasks. Reinforcement learning (RL) approaches, on the other hand, have shown some success in high-precision assembly, but suffer from sample inefficiency, which limits their effectiveness in long-horizon tasks. To address these challenges, we propose a hierarchical modular approach, named Adaptive Robotic Compositional Hierarchy (ARCH), which enables long-horizon, high-precision robotic assembly in contact-rich settings. ARCH employs a hierarchical planning framework, including a low-level primitive library of parameterized skills and a high-level policy. The low-level primitive library includes essential skills for assembly tasks, such as grasping and inserting. These primitives consist of both RL and model-based policies. The high-level policy, learned via IL from a handful of demonstrations, without the need for teleoperation, selects the appropriate primitive skills and instantiates them with input parameters. We extensively evaluate our approach in simulation and on a real robotic manipulation platform. We show that ARCH generalizes well to unseen objects and outperforms baseline methods in terms of success rate and data efficiency. More details are available at: https://long-horizon-assembly.github.io.
>
---
#### [replaced 009] The GOOSE Dataset for Perception in Unstructured Environments
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2310.16788v2](http://arxiv.org/pdf/2310.16788v2)**

> **作者:** Peter Mortimer; Raphael Hagmanns; Miguel Granero; Thorsten Luettel; Janko Petereit; Hans-Joachim Wuensche
>
> **备注:** Accepted at ICRA 2024, Github link: https://github.com/FraunhoferIOSB/goose_dataset
>
> **摘要:** The potential for deploying autonomous systems can be significantly increased by improving the perception and interpretation of the environment. However, the development of deep learning-based techniques for autonomous systems in unstructured outdoor environments poses challenges due to limited data availability for training and testing. To address this gap, we present the German Outdoor and Offroad Dataset (GOOSE), a comprehensive dataset specifically designed for unstructured outdoor environments. The GOOSE dataset incorporates 10 000 labeled pairs of images and point clouds, which are utilized to train a range of state-of-the-art segmentation models on both image and point cloud data. We open source the dataset, along with an ontology for unstructured terrain, as well as dataset standards and guidelines. This initiative aims to establish a common framework, enabling the seamless inclusion of existing datasets and a fast way to enhance the perception capabilities of various robots operating in unstructured environments. The dataset, pre-trained models for offroad perception, and additional documentation can be found at https://goose-dataset.de/.
>
---
#### [replaced 010] Bipedal Balance Control with Whole-body Musculoskeletal Standing and Falling Simulations
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2506.09383v2](http://arxiv.org/pdf/2506.09383v2)**

> **作者:** Chengtian Ma; Yunyue Wei; Chenhui Zuo; Chen Zhang; Yanan Sui
>
> **摘要:** Balance control is important for human and bipedal robotic systems. While dynamic balance during locomotion has received considerable attention, quantitative understanding of static balance and falling remains limited. This work presents a hierarchical control pipeline for simulating human balance via a comprehensive whole-body musculoskeletal system. We identified spatiotemporal dynamics of balancing during stable standing, revealed the impact of muscle injury on balancing behavior, and generated fall contact patterns that aligned with clinical data. Furthermore, our simulated hip exoskeleton assistance demonstrated improvement in balance maintenance and reduced muscle effort under perturbation. This work offers unique muscle-level insights into human balance dynamics that are challenging to capture experimentally. It could provide a foundation for developing targeted interventions for individuals with balance impairments and support the advancement of humanoid robotic systems.
>
---
#### [replaced 011] Active Illumination for Visual Ego-Motion Estimation in the Dark
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.13708v2](http://arxiv.org/pdf/2502.13708v2)**

> **作者:** Francesco Crocetti; Alberto Dionigi; Raffaele Brilli; Gabriele Costante; Paolo Valigi
>
> **摘要:** Visual Odometry (VO) and Visual SLAM (V-SLAM) systems often struggle in low-light and dark environments due to the lack of robust visual features. In this paper, we propose a novel active illumination framework to enhance the performance of VO and V-SLAM algorithms in these challenging conditions. The developed approach dynamically controls a moving light source to illuminate highly textured areas, thereby improving feature extraction and tracking. Specifically, a detector block, which incorporates a deep learning-based enhancing network, identifies regions with relevant features. Then, a pan-tilt controller is responsible for guiding the light beam toward these areas, so that to provide information-rich images to the ego-motion estimation algorithm. Experimental results on a real robotic platform demonstrate the effectiveness of the proposed method, showing a reduction in the pose estimation error up to 75% with respect to a traditional fixed lighting technique.
>
---
#### [replaced 012] Why Report Failed Interactions With Robots?! Towards Vignette-based Interaction Quality
- **分类: cs.RO; cs.HC**

- **链接: [http://arxiv.org/pdf/2508.10603v3](http://arxiv.org/pdf/2508.10603v3)**

> **作者:** Agnes Axelsson; Merle Reimann; Ronald Cumbal; Hannah Pelikan; Divesh Lala
>
> **备注:** Accepted at the workshop on Real-World HRI in Public and Private Spaces: Successes, Failures, and Lessons Learned (PubRob-Fails), held at the IEEE RO-MAN Conference, 2025. 6 pages
>
> **摘要:** Although the quality of human-robot interactions has improved with the advent of LLMs, there are still various factors that cause systems to be sub-optimal when compared to human-human interactions. The nature and criticality of failures are often dependent on the context of the interaction and so cannot be generalized across the wide range of scenarios and experiments which have been implemented in HRI research. In this work we propose the use of a technique overlooked in the field of HRI, ethnographic vignettes, to clearly highlight these failures, particularly those that are rarely documented. We describe the methodology behind the process of writing vignettes and create our own based on our personal experiences with failures in HRI systems. We emphasize the strength of vignettes as the ability to communicate failures from a multi-disciplinary perspective, promote transparency about the capabilities of robots, and document unexpected behaviours which would otherwise be omitted from research reports. We encourage the use of vignettes to augment existing interaction evaluation methods.
>
---
#### [replaced 013] DEXOP: A Device for Robotic Transfer of Dexterous Human Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.HC**

- **链接: [http://arxiv.org/pdf/2509.04441v2](http://arxiv.org/pdf/2509.04441v2)**

> **作者:** Hao-Shu Fang; Branden Romero; Yichen Xie; Arthur Hu; Bo-Ruei Huang; Juan Alvarez; Matthew Kim; Gabriel Margolis; Kavya Anbarasu; Masayoshi Tomizuka; Edward Adelson; Pulkit Agrawal
>
> **备注:** project page: https://dex-op.github.io
>
> **摘要:** We introduce perioperation, a paradigm for robotic data collection that sensorizes and records human manipulation while maximizing the transferability of the data to real robots. We implement this paradigm in DEXOP, a passive hand exoskeleton designed to maximize human ability to collect rich sensory (vision + tactile) data for diverse dexterous manipulation tasks in natural environments. DEXOP mechanically connects human fingers to robot fingers, providing users with direct contact feedback (via proprioception) and mirrors the human hand pose to the passive robot hand to maximize the transfer of demonstrated skills to the robot. The force feedback and pose mirroring make task demonstrations more natural for humans compared to teleoperation, increasing both speed and accuracy. We evaluate DEXOP across a range of dexterous, contact-rich tasks, demonstrating its ability to collect high-quality demonstration data at scale. Policies learned with DEXOP data significantly improve task performance per unit time of data collection compared to teleoperation, making DEXOP a powerful tool for advancing robot dexterity. Our project page is at https://dex-op.github.io.
>
---
#### [replaced 014] Data-Driven Robust Optimization for Energy-Aware Safe Motion Planning of Electric Vehicles
- **分类: eess.SY; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2304.12887v3](http://arxiv.org/pdf/2304.12887v3)**

> **作者:** Simran Kumari; Ashish R. Hota; Siddhartha Mukhopadhyay
>
> **摘要:** In this paper, we simultaneously address the problems of energy optimal and safe motion planning of electric vehicles (EVs) in a data-driven robust optimization framework. Safe maneuvers, especially in urban traffic, are characterized by frequent lateral motions, such as lane changes, overtakes and turning along curved roads. Motivated by our previous work which shows a 3-10 % increase in energy consumption due to lateral motion when an electric vehicle changes its lane once every kilometer while following standard drive cycles, we incorporate vehicle lateral dynamics in the modeling and control synthesis, which is in contrast with most prior works. In the context of safety, we leverage past data of obstacle motion to construct a future occupancy set with probabilistic guarantees, and formulate robust collision avoidance constraints with respect to such an occupancy set using convex programming duality. Consequently, we formulate a finite-horizon optimal control problem subject to robust collision avoidance constraints while penalizing resulting energy consumption, and solve it in a receding horizon fashion. Finally, we show the effectiveness of the proposed approach in reducing energy consumption and collision avoidance via numerical simulations involving curved roads and multiple obstacles. A detailed analysis of energy consumption along different components of EV motion highlights appreciable improvement under the proposed approach.
>
---
#### [replaced 015] Efficient Alignment of Unconditioned Action Prior for Language-conditioned Pick and Place in Clutter
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.09423v3](http://arxiv.org/pdf/2503.09423v3)**

> **作者:** Kechun Xu; Xunlong Xia; Kaixuan Wang; Yifei Yang; Yunxuan Mao; Bing Deng; Jieping Ye; Rong Xiong; Yue Wang
>
> **备注:** Accepted by T-ASE and CoRL25 GenPriors Workshop
>
> **摘要:** We study the task of language-conditioned pick and place in clutter, where a robot should grasp a target object in open clutter and move it to a specified place. Some approaches learn end-to-end policies with features from vision foundation models, requiring large datasets. Others combine foundation models in a zero-shot setting, suffering from cascading errors. In addition, they primarily leverage vision and language foundation models, focusing less on action priors. In this paper, we aim to develop an effective policy by integrating foundation priors from vision, language, and action. We propose A$^2$, an action prior alignment method that aligns unconditioned action priors with 3D vision-language priors by learning one attention layer. The alignment formulation enables our policy to train with less data and preserve zero-shot generalization capabilities. We show that a shared policy for both pick and place actions enhances the performance for each task, and introduce a policy adaptation scheme to accommodate the multi-modal nature of actions. Extensive experiments in simulation and the real-world show that our policy achieves higher task success rates with fewer steps for both pick and place tasks in clutter, effectively generalizing to unseen objects and language instructions. Videos and codes are available at https://xukechun.github.io/papers/A2.
>
---
#### [replaced 016] AARK: An Open Toolkit for Autonomous Racing Research
- **分类: cs.RO; cs.LG; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2410.00358v2](http://arxiv.org/pdf/2410.00358v2)**

> **作者:** James Bockman; Matthew Howe; Adrian Orenstein; Feras Dayoub
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** Autonomous racing demands safe control of vehicles at their physical limits for extended periods of time, providing insights into advanced vehicle safety systems which increasingly rely on intervention provided by vehicle autonomy. Participation in this field carries with it a high barrier to entry. Physical platforms and their associated sensor suites require large capital outlays before any demonstrable progress can be made. Simulators allow researches to develop soft autonomous systems without purchasing a platform. However, currently available simulators lack visual and dynamic fidelity, can still be expensive to buy, lack customisation, and are difficult to use. AARK provides three packages, ACI, ACDG, and ACMPC. These packages enable research into autonomous control systems in the demanding environment of racing to bring more people into the field and improve reproducibility: ACI provides researchers with a computer vision-friendly interface to Assetto Corsa for convenient comparison and evaluation of autonomous control solutions; ACDG enables generation of depth, normal and semantic segmentation data for training computer vision models to use in perception systems; and ACMPC gives newcomers to the field a modular full-stack autonomous control solution, capable of controlling vehicles to build from. AARK aims to unify and democratise research into a field critical to providing safer roads and trusted autonomous systems.
>
---
#### [replaced 017] Beyond Pairwise Comparisons: Unveiling Structural Landscape of Mobile Robot Models
- **分类: cs.DC; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.19805v2](http://arxiv.org/pdf/2508.19805v2)**

> **作者:** Shota Naito; Tsukasa Ninomiya; Koichi Wada
>
> **摘要:** Understanding the computational power of mobile robot systems is a fundamental challenge in distributed computing. While prior work has focused on pairwise separations between models, we explore how robot capabilities, light observability, and scheduler synchrony interact in more complex ways. We first show that the Exponential Times Expansion (ETE) problem is solvable only in the strongest model -- fully-synchronous robots with full mutual lights ($\mathcal{LUMT}^F$). We then introduce the Hexagonal Edge Traversal (HET) and TAR(d)* problems to demonstrate how internal memory and lights interact with synchrony: under weak synchrony, internal memory alone is insufficient, while full synchrony can substitute for both lights and memory. In the asynchronous setting, we classify problems such as LP-MLCv, VEC, and ZCC to show fine-grained separations between $\mathcal{FSTA}$ and $\mathcal{FCOM}$ robots. We also analyze Vertex Traversal Rendezvous (VTR) and Leave Place Convergence (LP-Cv), illustrating the limitations of internal memory in symmetric settings. These results extend the known separation map of 14 canonical robot models, revealing structural phenomena only visible through higher-order comparisons. Our work provides new impossibility criteria and deepens the understanding of how observability, memory, and synchrony collectively shape the computational power of mobile robots.
>
---
#### [replaced 018] Ontology Neural Network and ORTSF: A Framework for Topological Reasoning and Delay-Robust Control
- **分类: cs.RO; cs.SY; eess.SY; 68T40, 93C41; I.2.9; I.2.8; F.2.2**

- **链接: [http://arxiv.org/pdf/2506.19277v3](http://arxiv.org/pdf/2506.19277v3)**

> **作者:** Jaehong Oh
>
> **备注:** 12 pages, 5 figures, includes theoretical proofs and simulation results
>
> **摘要:** The advancement of autonomous robotic systems has led to impressive capabilities in perception, localization, mapping, and control. Yet, a fundamental gap remains: existing frameworks excel at geometric reasoning and dynamic stability but fall short in representing and preserving relational semantics, contextual reasoning, and cognitive transparency essential for collaboration in dynamic, human-centric environments. This paper introduces a unified architecture comprising the Ontology Neural Network (ONN) and the Ontological Real-Time Semantic Fabric (ORTSF) to address this gap. The ONN formalizes relational semantic reasoning as a dynamic topological process. By embedding Forman-Ricci curvature, persistent homology, and semantic tensor structures within a unified loss formulation, ONN ensures that relational integrity and topological coherence are preserved as scenes evolve over time. The ORTSF transforms reasoning traces into actionable control commands while compensating for system delays. It integrates predictive and delay-aware operators that ensure phase margin preservation and continuity of control signals, even under significant latency conditions. Empirical studies demonstrate the ONN + ORTSF framework's ability to unify semantic cognition and robust control, providing a mathematically principled and practically viable solution for cognitive robotics.
>
---
#### [replaced 019] ILeSiA: Interactive Learning of Robot Situational Awareness from Camera Input
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.20173v3](http://arxiv.org/pdf/2409.20173v3)**

> **作者:** Petr Vanc; Giovanni Franzese; Jan Kristof Behrens; Cosimo Della Santina; Karla Stepanova; Jens Kober; Robert Babuska
>
> **备注:** 8 pages, 9 figures. IEEE Robotics and Automation Letters. Accepted August 2025
>
> **摘要:** Learning from demonstration is a promising approach for teaching robots new skills. However, a central challenge in the execution of acquired skills is the ability to recognize faults and prevent failures. This is essential because demonstrations typically cover only a limited set of scenarios and often only the successful ones. During task execution, unforeseen situations may arise, such as changes in the robot's environment or interaction with human operators. To recognize such situations, this paper focuses on teaching the robot situational awareness by using a camera input and labeling frames as safe or risky. We train a Gaussian Process (GP) regression model fed by a low-dimensional latent space representation of the input images. The model outputs a continuous risk score ranging from zero to one, quantifying the degree of risk at each timestep. This allows for pausing task execution in unsafe situations and directly adding new training data, labeled by the human user. Our experiments on a robotic manipulator show that the proposed method can reliably detect both known and novel faults using only a single example for each new fault. In contrast, a standard multi-layer perceptron (MLP) performs well only on faults it has encountered during training. Our method enables the next generation of cobots to be rapidly deployed with easy-to-set-up, vision-based risk assessment, proactively safeguarding humans and detecting misaligned parts or missing objects before failures occur. We provide all the code and data required to reproduce our experiments at imitrob.ciirc.cvut.cz/publications/ilesia.
>
---
#### [replaced 020] Action Flow Matching for Continual Robot Learning
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2504.18471v2](http://arxiv.org/pdf/2504.18471v2)**

> **作者:** Alejandro Murillo-Gonzalez; Lantao Liu
>
> **备注:** Robotics: Science and Systems 2025
>
> **摘要:** Continual learning in robotics seeks systems that can constantly adapt to changing environments and tasks, mirroring human adaptability. A key challenge is refining dynamics models, essential for planning and control, while addressing issues such as safe adaptation, catastrophic forgetting, outlier management, data efficiency, and balancing exploration with exploitation -- all within task and onboard resource constraints. Towards this goal, we introduce a generative framework leveraging flow matching for online robot dynamics model alignment. Rather than executing actions based on a misaligned model, our approach refines planned actions to better match with those the robot would take if its model was well aligned. We find that by transforming the actions themselves rather than exploring with a misaligned model -- as is traditionally done -- the robot collects informative data more efficiently, thereby accelerating learning. Moreover, we validate that the method can handle an evolving and possibly imperfect model while reducing, if desired, the dependency on replay buffers or legacy model snapshots. We validate our approach using two platforms: an unmanned ground vehicle and a quadrotor. The results highlight the method's adaptability and efficiency, with a record 34.2\% higher task success rate, demonstrating its potential towards enabling continual robot learning. Code: https://github.com/AlejandroMllo/action_flow_matching.
>
---
#### [replaced 021] VIPER: Visual Perception and Explainable Reasoning for Sequential Decision-Making
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.15108v2](http://arxiv.org/pdf/2503.15108v2)**

> **作者:** Mohamed Salim Aissi; Clemence Grislain; Mohamed Chetouani; Olivier Sigaud; Laure Soulier; Nicolas Thome
>
> **摘要:** While Large Language Models (LLMs) excel at reasoning on text and Vision-Language Models (VLMs) are highly effective for visual perception, applying those models for visual instruction-based planning remains a widely open problem. In this paper, we introduce VIPER, a novel framework for multimodal instruction-based planning that integrates VLM-based perception with LLM-based reasoning. Our approach uses a modular pipeline where a frozen VLM generates textual descriptions of image observations, which are then processed by an LLM policy to predict actions based on the task goal. We fine-tune the reasoning module using behavioral cloning and reinforcement learning, improving our agent's decision-making capabilities. Experiments on the ALFWorld benchmark show that VIPER significantly outperforms state-of-the-art visual instruction-based planners while narrowing the gap with purely text-based oracles. By leveraging text as an intermediate representation, VIPER also enhances explainability, paving the way for a fine-grained analysis of perception and reasoning components.
>
---
#### [replaced 022] Robotic Fire Risk Detection based on Dynamic Knowledge Graph Reasoning: An LLM-Driven Approach with Graph Chain-of-Thought
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2509.00054v2](http://arxiv.org/pdf/2509.00054v2)**

> **作者:** Haimei Pan; Jiyun Zhang; Qinxi Wei; Xiongnan Jin; Chen Xinkai; Jie Cheng
>
> **备注:** We have decided to withdraw this paper as the work is still undergoing further refinement. To ensure the clarity of the results, we prefer to make additional improvements before resubmission. We appreciate the readers' understanding
>
> **摘要:** Fire is a highly destructive disaster, but effective prevention can significantly reduce its likelihood of occurrence. When it happens, deploying emergency robots in fire-risk scenarios can help minimize the danger to human responders. However, current research on pre-disaster warnings and disaster-time rescue still faces significant challenges due to incomplete perception, inadequate fire situational awareness, and delayed response. To enhance intelligent perception and response planning for robots in fire scenarios, we first construct a knowledge graph (KG) by leveraging large language models (LLMs) to integrate fire domain knowledge derived from fire prevention guidelines and fire rescue task information from robotic emergency response documents. We then propose a new framework called Insights-on-Graph (IOG), which integrates the structured fire information of KG and Large Multimodal Models (LMMs). The framework generates perception-driven risk graphs from real-time scene imagery to enable early fire risk detection and provide interpretable emergency responses for task module and robot component configuration based on the evolving risk situation. Extensive simulations and real-world experiments show that IOG has good applicability and practical application value in fire risk detection and rescue decision-making.
>
---
#### [replaced 023] Automated Planning Domain Inference for Task and Motion Planning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.16445v3](http://arxiv.org/pdf/2410.16445v3)**

> **作者:** Jinbang Huang; Allen Tao; Rozilyn Marco; Miroslav Bogdanovic; Jonathan Kelly; Florian Shkurti
>
> **备注:** Published in the Proceedings of the 2025 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Task and motion planning (TAMP) frameworks address long and complex planning problems by integrating high-level task planners with low-level motion planners. However, existing TAMP methods rely heavily on the manual design of planning domains that specify the preconditions and postconditions of all high-level actions. This paper proposes a method to automate planning domain inference from a handful of test-time trajectory demonstrations, reducing the reliance on human design. Our approach incorporates a deep learning-based estimator that predicts the appropriate components of a domain for a new task and a search algorithm that refines this prediction, reducing the size and ensuring the utility of the inferred domain. Our method is able to generate new domains from minimal demonstrations at test time, enabling robots to handle complex tasks more efficiently. We demonstrate that our approach outperforms behavior cloning baselines, which directly imitate planner behavior, in terms of planning performance and generalization across a variety of tasks. Additionally, our method reduces computational costs and data amount requirements at test time for inferring new planning domains.
>
---
#### [replaced 024] Generative Modeling for Adversarial Lane-Change Scenarios
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.12055v2](http://arxiv.org/pdf/2503.12055v2)**

> **作者:** Chuancheng Zhang; Zhenhao Wang; Jiangcheng Wang; Kun Su; Qiang Lv; Bin Jiang; Kunkun Hao; Wenyu Wang
>
> **摘要:** Decision-making in long-tail scenarios is pivotal to autonomous-driving development, and realistic and challenging simulations play a crucial role in testing safety-critical situations. However, existing open-source datasets lack systematic coverage of long-tail scenes, and lane-change maneuvers being emblematic, rendering such data exceedingly scarce. To bridge this gap, we introduce a data mining framework that exhaustively analyzes two widely used datasets, NGSIM and INTERACTION, to identify sequences marked by hazardous behavior, thereby replenishing these neglected scenarios. Using Generative Adversarial Imitation Learning (GAIL) enhanced with Proximal Policy Optimization (PPO), and enriched by vehicular-environment interaction analytics, our method iteratively refines and parameterizes newly generated trajectories. Distinguished by a rationally adversarial and sensitivity-aware perspective, the approach optimizes the creation of challenging scenes. Experiments show that, compared to unfiltered data and baseline models, our method produces behaviors that are simultaneously both adversarial and natural, judged by collision frequency, acceleration profiles, and lane-change dynamics, offering constructive insights to amplifying long-tailed lane-change instances in datasets and advancing decision-making training.
>
---
#### [replaced 025] Inverse Kinematics for a 6-Degree-of-Freedom Robot Manipulator Using Comprehensive Gröbner Systems
- **分类: cs.RO; cs.SC; math.AC; 68W30, 13P10, 13P25**

- **链接: [http://arxiv.org/pdf/2509.00823v2](http://arxiv.org/pdf/2509.00823v2)**

> **作者:** Takumu Okazaki; Akira Terui; Masahiko Mikawa
>
> **备注:** 24 pages
>
> **摘要:** We propose an effective method for solving the inverse kinematic problem of a specific model of 6-degree-of-freedom (6-DOF) robot manipulator using computer algebra. It is known that when the rotation axes of three consecutive rotational joints of a manipulator intersect at a single point, the inverse kinematics problem can be divided into determining position and orientation. We extend this method to more general manipulators in which the rotational axes of two consecutive joints intersect. This extension broadens the class of 6-DOF manipulators for which the inverse kinematics problem can be solved, and is expected to enable more efficient solutions. The inverse kinematic problem is solved using the Comprehensive Gr\"obner System (CGS) with joint parameters of the robot appearing as parameters in the coefficients to prevent repetitive calculations of the Gr\"obner bases. The effectiveness of the proposed method is shown by experiments.
>
---
#### [replaced 026] Ask1: Development and Reinforcement Learning-Based Control of a Custom Quadruped Robot
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.08019v2](http://arxiv.org/pdf/2412.08019v2)**

> **作者:** Yang Zhang; Yuxing Lu; Guiyang Xin; Yufei Xue; Chenkun Qi; Kairong Qin; Yan Zhuang
>
> **摘要:** In this work, we present the design, development, and experimental validation of a custom-built quadruped robot, Ask1. The Ask1 robot shares similar morphology with the Unitree Go1, but features custom hardware components and a different control architecture. We transfer and extend previous reinforcement learning (RL)-based control methods to the Ask1 robot, demonstrating the applicability of our approach in real-world scenarios. By eliminating the need for Adversarial Motion Priors (AMP) and reference trajectories, we introduce a novel reward function to guide the robot's motion style. We demonstrate the generalization capability of the proposed RL algorithm by training it on both the Go1 and Ask1 robots. Simulation and real-world experiments validate the effectiveness of this method, showing that Ask1, like the Go1, is capable of navigating various rugged terrains.
>
---
#### [replaced 027] BeSimulator: A Large Language Model Powered Text-based Behavior Simulator
- **分类: cs.RO; cs.AI; cs.CL**

- **链接: [http://arxiv.org/pdf/2409.15865v2](http://arxiv.org/pdf/2409.15865v2)**

> **作者:** Jianan Wang; Bin Li; Jingtao Qi; Xueying Wang; Fu Li; Hanxun Li
>
> **备注:** 19 pages, 5 figures, 8 tables
>
> **摘要:** Traditional robot simulators focus on physical process modeling and realistic rendering, often suffering from high computational costs, inefficiencies, and limited adaptability. To handle this issue, we concentrate on behavior simulation in robotics to analyze and validate the logic behind robot behaviors, aiming to achieve preliminary evaluation before deploying resource-intensive simulators and thus enhance simulation efficiency. In this paper, we propose BeSimulator, a modular and novel LLM-powered framework, as an attempt towards behavior simulation in the context of text-based environments. By constructing text-based virtual environments and performing semantic-level simulation, BeSimulator can generalize across scenarios and achieve long-horizon complex simulation. Inspired by human cognition paradigm, it employs a ``consider-decide-capture-transfer'' four-phase simulation process, termed Chain of Behavior Simulation (CBS), which excels at analyzing action feasibility and state transition. Additionally, BeSimulator incorporates code-driven reasoning to enable arithmetic operations and enhance reliability, and reflective feedback to refine simulation. Based on our manually constructed behavior-tree-based simulation benchmark, BTSIMBENCH, our experiments show a significant performance improvement in behavior simulation compared to baselines, ranging from 13.60% to 24.80%. Code and data are available at https://github.com/Dawn888888/BeSimulator.
>
---
#### [replaced 028] ESVO2: Direct Visual-Inertial Odometry with Stereo Event Cameras
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2410.09374v4](http://arxiv.org/pdf/2410.09374v4)**

> **作者:** Junkai Niu; Sheng Zhong; Xiuyuan Lu; Shaojie Shen; Guillermo Gallego; Yi Zhou
>
> **摘要:** Event-based visual odometry is a specific branch of visual Simultaneous Localization and Mapping (SLAM) techniques, which aims at solving tracking and mapping subproblems (typically in parallel), by exploiting the special working principles of neuromorphic (i.e., event-based) cameras. Due to the motion-dependent nature of event data, explicit data association (i.e., feature matching) under large-baseline view-point changes is difficult to establish, making direct methods a more rational choice. However, state-of-the-art direct methods are limited by the high computational complexity of the mapping sub-problem and the degeneracy of camera pose tracking in certain degrees of freedom (DoF) in rotation. In this paper, we tackle these issues by building an event-based stereo visual-inertial odometry system on top of a direct pipeline. Specifically, to speed up the mapping operation, we propose an efficient strategy for sampling contour points according to the local dynamics of events. The mapping performance is also improved in terms of structure completeness and local smoothness by merging the temporal stereo and static stereo results. To circumvent the degeneracy of camera pose tracking in recovering the pitch and yaw components of general 6-DoF motion, we introduce IMU measurements as motion priors via pre-integration. To this end, a compact back-end is proposed for continuously updating the IMU bias and predicting the linear velocity, enabling an accurate motion prediction for camera pose tracking. The resulting system scales well with modern high-resolution event cameras and leads to better global positioning accuracy in large-scale outdoor environments. Extensive evaluations on five publicly available datasets featuring different resolutions and scenarios justify the superior performance of the proposed system against five state-of-the-art methods.
>
---
#### [replaced 029] Stochastic Adaptive Estimation in Polynomial Curvature Shape State Space for Continuum Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2210.08427v4](http://arxiv.org/pdf/2210.08427v4)**

> **作者:** Guoqing Zhang; Long Wang
>
> **备注:** 20 pages. IEEE Transactions on Robotics - conditionally accepted; this arXiv version corresponds to the final revision (submitted 2025-09-07). Supplementary appendix provided as an ancillary PDF
>
> **摘要:** In continuum robotics, real-time robust shape estimation is crucial for planning and control tasks that involve physical manipulation in complex environments. In this paper, we present a novel stochastic observer-based shape estimation framework designed specifically for continuum robots. The shape state space is uniquely represented by the modal coefficients of a polynomial, enabled by leveraging polynomial curvature kinematics (PCK) to describe the curvature distribution along the arclength. Our framework processes noisy measurements from limited discrete position, orientation, or pose sensors to estimate the shape state robustly. We derive a novel noise-weighted observability matrix, providing a detailed assessment of observability variations under diverse sensor configurations. To overcome the limitations of a single model, our observer employs the Interacting Multiple Model (IMM) method, coupled with Extended Kalman Filters (EKFs), to mix polynomial curvature models of different orders. The IMM approach, rooted in Markov processes, effectively manages multiple model scenarios by dynamically adapting to different polynomial orders based on real-time model probabilities. This adaptability is key to ensuring robust shape estimation of the robot's behaviors under various conditions. Our comprehensive analysis, supported by both simulation studies and experimental validations, confirms the robustness and accuracy of our methods.
>
---
#### [replaced 030] SAIL: Faster-than-Demonstration Execution of Imitation Learning Policies
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.11948v2](http://arxiv.org/pdf/2506.11948v2)**

> **作者:** Nadun Ranawaka Arachchige; Zhenyang Chen; Wonsuhk Jung; Woo Chul Shin; Rohan Bansal; Pierre Barroso; Yu Hang He; Yingyang Celine Lin; Benjamin Joffe; Shreyas Kousik; Danfei Xu
>
> **备注:** The first two authors contributed equally. Accepted to CoRL 2025
>
> **摘要:** Offline Imitation Learning (IL) methods such as Behavior Cloning are effective at acquiring complex robotic manipulation skills. However, existing IL-trained policies are confined to executing the task at the same speed as shown in demonstration data. This limits the task throughput of a robotic system, a critical requirement for applications such as industrial automation. In this paper, we introduce and formalize the novel problem of enabling faster-than-demonstration execution of visuomotor policies and identify fundamental challenges in robot dynamics and state-action distribution shifts. We instantiate the key insights as SAIL (Speed Adaptation for Imitation Learning), a full-stack system integrating four tightly-connected components: (1) a consistency-preserving action inference algorithm for smooth motion at high speed, (2) high-fidelity tracking of controller-invariant motion targets, (3) adaptive speed modulation that dynamically adjusts execution speed based on motion complexity, and (4) action scheduling to handle real-world system latencies. Experiments on 12 tasks across simulation and two real, distinct robot platforms show that SAIL achieves up to a 4x speedup over demonstration speed in simulation and up to 3.2x speedup in the real world. Additional detail is available at https://nadunranawaka1.github.io/sail-policy
>
---
#### [replaced 031] LanternNet: A Hub-and-Spoke System to Seek and Suppress Spotted Lanternfly Populations
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.20800v4](http://arxiv.org/pdf/2507.20800v4)**

> **作者:** Vinil Polepalli
>
> **备注:** The submission is being withdrawn pending coordination with co-authors before resubmission
>
> **摘要:** The invasive spotted lanternfly (SLF) poses a significant threat to agriculture and ecosystems, causing widespread damage. Current control methods, such as egg scraping, pesticides, and quarantines, prove labor-intensive, environmentally hazardous, and inadequate for long-term SLF suppression. This research introduces LanternNet, a novel autonomous robotic Hub-and-Spoke system designed for scalable detection and suppression of SLF populations. A central, tree-mimicking hub utilizes a YOLOv8 computer vision model for precise SLF identification. Three specialized robotic spokes perform targeted tasks: pest neutralization, environmental monitoring, and navigation/mapping. Field deployment across multiple infested sites over 5 weeks demonstrated LanternNet's efficacy. Quantitative analysis revealed significant reductions (p < 0.01, paired t-tests) in SLF populations and corresponding improvements in tree health indicators across the majority of test sites. Compared to conventional methods, LanternNet offers substantial cost advantages and improved scalability. Furthermore, the system's adaptability for enhanced autonomy and targeting of other invasive species presents significant potential for broader ecological impact. LanternNet demonstrates the transformative potential of integrating robotics and AI for advanced invasive species management and improved environmental outcomes.
>
---
#### [replaced 032] Skill-Nav: Enhanced Navigation with Versatile Quadrupedal Locomotion via Waypoint Interface
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.21853v3](http://arxiv.org/pdf/2506.21853v3)**

> **作者:** Dewei Wang; Chenjia Bai; Chenhui Li; Jiyuan Shi; Yan Ding; Chi Zhang; Bin Zhao
>
> **备注:** 17pages, 6 figures
>
> **摘要:** Quadrupedal robots have demonstrated exceptional locomotion capabilities through Reinforcement Learning (RL), including extreme parkour maneuvers. However, integrating locomotion skills with navigation in quadrupedal robots has not been fully investigated, which holds promise for enhancing long-distance movement capabilities. In this paper, we propose Skill-Nav, a method that incorporates quadrupedal locomotion skills into a hierarchical navigation framework using waypoints as an interface. Specifically, we train a waypoint-guided locomotion policy using deep RL, enabling the robot to autonomously adjust its locomotion skills to reach targeted positions while avoiding obstacles. Compared with direct velocity commands, waypoints offer a simpler yet more flexible interface for high-level planning and low-level control. Utilizing waypoints as the interface allows for the application of various general planning tools, such as large language models (LLMs) and path planning algorithms, to guide our locomotion policy in traversing terrains with diverse obstacles. Extensive experiments conducted in both simulated and real-world scenarios demonstrate that Skill-Nav can effectively traverse complex terrains and complete challenging navigation tasks.
>
---
#### [replaced 033] ManiCM: Real-time 3D Diffusion Policy via Consistency Model for Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2406.01586v3](http://arxiv.org/pdf/2406.01586v3)**

> **作者:** Guanxing Lu; Zifeng Gao; Tianxing Chen; Wenxun Dai; Ziwei Wang; Wenbo Ding; Yansong Tang
>
> **备注:** https://manicm-fast.github.io/
>
> **摘要:** Diffusion models have been verified to be effective in generating complex distributions from natural images to motion trajectories. Recent diffusion-based methods show impressive performance in 3D robotic manipulation tasks, whereas they suffer from severe runtime inefficiency due to multiple denoising steps, especially with high-dimensional observations. To this end, we propose a real-time robotic manipulation model named ManiCM that imposes the consistency constraint to the diffusion process, so that the model can generate robot actions in only one-step inference. Specifically, we formulate a consistent diffusion process in the robot action space conditioned on the point cloud input, where the original action is required to be directly denoised from any point along the ODE trajectory. To model this process, we design a consistency distillation technique to predict the action sample directly instead of predicting the noise within the vision community for fast convergence in the low-dimensional action manifold. We evaluate ManiCM on 31 robotic manipulation tasks from Adroit and Metaworld, and the results demonstrate that our approach accelerates the state-of-the-art method by 10 times in average inference speed while maintaining competitive average success rate.
>
---
#### [replaced 034] Generative World Explorer
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2411.11844v3](http://arxiv.org/pdf/2411.11844v3)**

> **作者:** Taiming Lu; Tianmin Shu; Alan Yuille; Daniel Khashabi; Jieneng Chen
>
> **备注:** Website: generative-world-explorer.github.io
>
> **摘要:** Planning with partial observation is a central challenge in embodied AI. A majority of prior works have tackled this challenge by developing agents that physically explore their environment to update their beliefs about the world state. In contrast, humans can $\textit{imagine}$ unseen parts of the world through a mental exploration and $\textit{revise}$ their beliefs with imagined observations. Such updated beliefs can allow them to make more informed decisions, without necessitating the physical exploration of the world at all times. To achieve this human-like ability, we introduce the $\textit{Generative World Explorer (Genex)}$, an egocentric world exploration framework that allows an agent to mentally explore a large-scale 3D world (e.g., urban scenes) and acquire imagined observations to update its belief. This updated belief will then help the agent to make a more informed decision at the current step. To train $\textit{Genex}$, we create a synthetic urban scene dataset, Genex-DB. Our experimental results demonstrate that (1) $\textit{Genex}$ can generate high-quality and consistent observations during long-horizon exploration of a large virtual physical world and (2) the beliefs updated with the generated observations can inform an existing decision-making model (e.g., an LLM agent) to make better plans.
>
---
#### [replaced 035] The best approximation pair problem relative to two subsets in a normed space
- **分类: math.OC; cs.GR; cs.RO; math.FA; math.MG; 41A50, 41A52, 41A65, 90C25, 46N10, 90C26, 46B20, 68U05, 65D18; G.1.6; G.1.2; I.3.5**

- **链接: [http://arxiv.org/pdf/2403.18767v3](http://arxiv.org/pdf/2403.18767v3)**

> **作者:** Daniel Reem; Yair Censor
>
> **备注:** Correction of a misprint in the Acknowledgments
>
> **摘要:** In the classical best approximation pair (BAP) problem, one is given two nonempty, closed, convex and disjoint subsets in a finite- or an infinite-dimensional Hilbert space, and the goal is to find a pair of points, each from each subset, which realizes the distance between the subsets. We discuss the problem in more general normed spaces and with possibly non-convex subsets, and focus our attention on the issues of uniqueness and existence of the solution to the problem. As far as we know, these fundamental issues have not received much attention. We present several sufficient geometric conditions for the (at most) uniqueness of a BAP. These conditions are related to the structure and the relative orientation of the boundaries of the subsets and to the norm. We also present many sufficient conditions for the existence of a BAP. Our results significantly extend the horizon of a recent algorithm for solving the BAP problem [Censor, Mansour, Reem, J. Approx. Theory (2024)]. The paper also shows, perhaps for the first time, how wide is the scope of the BAP problem in terms of the scientific communities which are involved in it (frequently independently) and in terms of its applications.
>
---
#### [replaced 036] Scaling Laws of Motion Forecasting and Planning - Technical Report
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.08228v2](http://arxiv.org/pdf/2506.08228v2)**

> **作者:** Mustafa Baniodeh; Kratarth Goel; Scott Ettinger; Carlos Fuertes; Ari Seff; Tim Shen; Cole Gulino; Chenjie Yang; Ghassen Jerfel; Dokook Choe; Rui Wang; Benjamin Charrow; Vinutha Kallem; Sergio Casas; Rami Al-Rfou; Benjamin Sapp; Dragomir Anguelov
>
> **摘要:** We study the empirical scaling laws of a family of encoder-decoder autoregressive transformer models on the task of joint motion forecasting and planning in the autonomous driving domain. Using a 500 thousand hours driving dataset, we demonstrate that, similar to language modeling, model performance improves as a power-law function of the total compute budget, and we observe a strong correlation between model training loss and model evaluation metrics. Most interestingly, closed-loop metrics also improve with scaling, which has important implications for the suitability of open-loop metrics for model development and hill climbing. We also study the optimal scaling of the number of transformer parameters and the training data size for a training compute-optimal model. We find that as the training compute budget grows, optimal scaling requires increasing the model size 1.5x as fast as the dataset size. We also study inference-time compute scaling, where we observe that sampling and clustering the output of smaller models makes them competitive with larger models, up to a crossover point beyond which a larger models becomes more inference-compute efficient. Overall, our experimental results demonstrate that optimizing the training and inference-time scaling properties of motion forecasting and planning models is a key lever for improving their performance to address a wide variety of driving scenarios. Finally, we briefly study the utility of training on general logged driving data of other agents to improve the performance of the ego-agent, an important research area to address the scarcity of robotics data for large capacity models training.
>
---
#### [replaced 037] Modular Recurrence in Contextual MDPs for Universal Morphology Control
- **分类: cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.08630v2](http://arxiv.org/pdf/2506.08630v2)**

> **作者:** Laurens Engwegen; Daan Brinks; Wendelin Böhmer
>
> **摘要:** A universal controller for any robot morphology would greatly improve computational and data efficiency. By utilizing contextual information about the properties of individual robots and exploiting their modular structure in the architecture of deep reinforcement learning agents, steps have been made towards multi-robot control. Generalization to new, unseen robots, however, remains a challenge. In this paper we hypothesize that the relevant contextual information is partially observable, but that it can be inferred through interactions for better generalization to contexts that are not seen during training. To this extent, we implement a modular recurrent architecture and evaluate its generalization performance on a large set of MuJoCo robots. The results show a substantial improved performance on robots with unseen dynamics, kinematics, and topologies, in four different environments.
>
---
#### [replaced 038] A High Efficient and Scalable Obstacle-Avoiding VLSI Global Routing Flow
- **分类: cs.OH; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.07268v3](http://arxiv.org/pdf/2503.07268v3)**

> **作者:** Junhao Guo; Hongxin Kong; Lang Feng
>
> **备注:** Accepted by ACM TODAES
>
> **摘要:** Routing is a crucial step in the VLSI design flow. With the advancement of manufacturing technologies, more constraints have emerged in design rules, particularly regarding obstacles during routing, leading to increased routing complexity. Unfortunately, many global routers struggle to efficiently generate obstacle-free solutions due to the lack of scalable obstacle-avoiding tree generation methods and the capability of handling modern designs with complex obstacles and nets. In this work, we propose an efficient obstacle-aware global routing flow for VLSI designs with obstacles. The flow includes a rule-based obstacle-avoiding rectilinear Steiner minimal tree (OARSMT) algorithm during the tree generation phase. This algorithm is both scalable and fast to provide tree topologies avoiding obstacles in the early stage globally. With its guidance, OARSMT-guided and obstacle-aware sparse maze routing are proposed in the later stages to minimize obstacle violations further and reduce overflow costs. Compared to advanced methods on the benchmark with obstacles, our approach successfully eliminates obstacle violations, and reduces wirelength and overflow cost, while sacrificing only a limited number of via counts and runtime overhead.
>
---
#### [replaced 039] Stability analysis through folds: An end-loaded elastic with a lever arm
- **分类: math.OC; cond-mat.soft; cs.RO**

- **链接: [http://arxiv.org/pdf/2501.04729v5](http://arxiv.org/pdf/2501.04729v5)**

> **作者:** Siva Prasad Chakri Dhanakoti
>
> **备注:** 22 pages, 12 figures
>
> **摘要:** Many physical systems can be modelled as parameter-dependent variational problems. In numerous cases, multiple equilibria co-exist, requiring the evaluation of their stability, and the monitoring of transitions between them. Generally, the stability characteristics of the equilibria change near folds in the parameter space. The direction of stability changes is embedded in a specific projection of the solutions, known as distinguished bifurcation diagrams. In this article, we identify such projections for variational problems characterized by fixed-free ends -- a class of problems frequently encountered in mechanics. Using these diagrams, we study an Elastica subject to an end load applied through a rigid lever arm. Several instances of snap-back instability are reported, along with their dependence on system parameters through numerical examples. These findings have potential applications in the design of soft robot arms and other actuator designs.
>
---
#### [replaced 040] Forbal: Force Balanced 2-5 Degree of Freedom Robot Manipulator Built from a Five Bar Linkage
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2509.03119v2](http://arxiv.org/pdf/2509.03119v2)**

> **作者:** Yash Vyas; Matteo Bottin
>
> **摘要:** A force balanced manipulator design based on the closed chain planar five bar linkage is developed and experimentally validated. We present 2 variants as a modular design: Forbal-2, a planar 2-DOF manipulator, and its extension to 5-DOF spatial motion called Forbal-5. The design considerations in terms of geometric, kinematic, and dynamic design that fulfill the force balance conditions while maximizing workspace are discussed. Then, the inverse kinematics of both variants are derived from geometric principles. We validate the improvements from force balancing the manipulator through comparative experiments with counter mass balanced and unbalanced configurations. The results show how the balanced configuration yields a reduction in the average reaction moments of up to 66%, a reduction of average joint torques of up to 79%, as well as a noticeable reduction in position error for Forbal-2. For Forbal-5, which has a higher end effector payload mass, the joint torques are reduced up to 84% for the balanced configuration. Experimental results validate that the balanced manipulator design is suitable for applications where the reduction of joint torques and reaction forces/moments helps achieve millimeter level precision.
>
---
#### [replaced 041] Efficient Virtuoso: A Latent Diffusion Transformer Model for Goal-Conditioned Trajectory Planning
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.03658v2](http://arxiv.org/pdf/2509.03658v2)**

> **作者:** Antonio Guillen-Perez
>
> **摘要:** The ability to generate a diverse and plausible distribution of future trajectories is a critical capability for autonomous vehicle planning systems. While recent generative models have shown promise, achieving high fidelity, computational efficiency, and precise control remains a significant challenge. In this paper, we present the Efficient Virtuoso, a conditional latent diffusion model for goal-conditioned trajectory planning. Our approach introduces a novel two-stage normalization pipeline that first scales trajectories to preserve their geometric aspect ratio and then normalizes the resulting PCA latent space to ensure a stable training target. The denoising process is performed efficiently in this low-dimensional latent space by a simple MLP denoiser, which is conditioned on a rich scene context fused by a powerful Transformer-based StateEncoder. We demonstrate that our method achieves state-of-the-art performance on the Waymo Open Motion Dataset, achieving a minimum Average Displacement Error (minADE) of 0.25. Furthermore, through a rigorous ablation study on goal representation, we provide a key insight: while a single endpoint goal can resolve strategic ambiguity, a richer, multi-step sparse route is essential for enabling the precise, high-fidelity tactical execution that mirrors nuanced human driving behavior.
>
---
#### [replaced 042] Towards No-Code Programming of Cobots: Experiments with Code Synthesis by Large Code Models for Conversational Programming
- **分类: cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2409.11041v4](http://arxiv.org/pdf/2409.11041v4)**

> **作者:** Chalamalasetti Kranti; Sherzod Hakimov; David Schlangen
>
> **备注:** Accepted to ITL4HRI workshop at RO-MAN 2025 conference
>
> **摘要:** While there has been a lot of research recently on robots in household environments, at the present time, most robots in existence can be found on shop floors, and most interactions between humans and robots happen there. ``Collaborative robots'' (cobots) designed to work alongside humans on assembly lines traditionally require expert programming, limiting ability to make changes, or manual guidance, limiting expressivity of the resulting programs. To address these limitations, we explore using Large Language Models (LLMs), and in particular, their abilities of doing in-context learning, for conversational code generation. As a first step, we define RATS, the ``Repetitive Assembly Task'', a 2D building task designed to lay the foundation for simulating industry assembly scenarios. In this task, a `programmer' instructs a cobot, using natural language, on how a certain assembly is to be built; that is, the programmer induces a program, through natural language. We create a dataset that pairs target structures with various example instructions (human-authored, template-based, and model-generated) and example code. With this, we systematically evaluate the capabilities of state-of-the-art LLMs for synthesising this kind of code, given in-context examples. Evaluating in a simulated environment, we find that LLMs are capable of generating accurate `first order code' (instruction sequences), but have problems producing `higher-order code' (abstractions such as functions, or use of loops).
>
---
#### [replaced 043] Sense4FL: Vehicular Crowdsensing Enhanced Federated Learning for Object Detection in Autonomous Driving
- **分类: cs.RO; cs.DC**

- **链接: [http://arxiv.org/pdf/2503.17697v2](http://arxiv.org/pdf/2503.17697v2)**

> **作者:** Yanan Ma; Senkang Hu; Zhengru Fang; Yun Ji; Yiqin Deng; Yuguang Fang
>
> **备注:** 18 pages, 8 figures
>
> **摘要:** To accommodate constantly changing road conditions, real-time vision model training is essential for autonomous driving (AD). Federated learning (FL) serves as a promising paradigm to enable autonomous vehicles to train models collaboratively with their onboard computing resources. However, existing vehicle selection schemes for FL all assume predetermined and location-independent vehicles' datasets, neglecting the fact that vehicles collect training data along their routes, thereby resulting in suboptimal vehicle selection. In this paper, we focus on the fundamental perception problem and propose Sense4FL, a vehicular crowdsensing-enhanced FL framework featuring \textit{trajectory-dependent} vehicular \textit{training data collection} to \rev{improve the object detection quality} in AD for a region. To this end, we first derive the convergence bound of FL by considering the impact of both vehicles' uncertain trajectories and uploading probabilities, from which we discover that minimizing the training loss is equivalent to minimizing a weighted sum of local and global earth mover's distance (EMD) between vehicles' collected data distribution and global data distribution. Based on this observation, we formulate the trajectory-dependent vehicle selection and data collection problem for FL in AD. Given that the problem is NP-hard, we develop an efficient algorithm to find the solution with an approximation guarantee. Extensive simulation results have demonstrated the effectiveness of our approach in improving object detection performance compared with existing benchmarks.
>
---
#### [replaced 044] Modeling, Observability, and Inertial Parameter Estimation of a Planar Multi-Link System with Thrusters
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.14790v2](http://arxiv.org/pdf/2503.14790v2)**

> **作者:** Nicholas B. Andrews; Kristi A. Morgansen
>
> **备注:** 8 pages, 4 figures, 4 tables
>
> **摘要:** This research provides a theoretical foundation for modeling and real-time estimation of both the pose and inertial parameters of a free-floating multi-link system with link thrusters, which are essential for safe and effective controller design and performance. First, we adapt a planar nonlinear multi-link snake robot model to represent a planar chain of bioinspired salp robots by removing joint actuators, introducing link thrusters, and allowing for non-uniform link lengths, masses, and moments of inertia. Second, we conduct a nonlinear observability analysis of the multi-link system with link thrusters, proving that the link angles, angular velocities, masses, and moments of inertia are locally observable when equipped with inertial measurement units and operating under specific thruster conditions. The analytical results are demonstrated in simulation with a three-link system.
>
---
#### [replaced 045] Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.19757v2](http://arxiv.org/pdf/2503.19757v2)**

> **作者:** Zhi Hou; Tianyi Zhang; Yuwen Xiong; Haonan Duan; Hengjun Pu; Ronglei Tong; Chengyang Zhao; Xizhou Zhu; Yu Qiao; Jifeng Dai; Yuntao Chen
>
> **备注:** Preprint; https://robodita.github.io; To appear in ICCV2025
>
> **摘要:** While recent vision-language-action models trained on diverse robot datasets exhibit promising generalization capabilities with limited in-domain data, their reliance on compact action heads to predict discretized or continuous actions constrains adaptability to heterogeneous action spaces. We present Dita, a scalable framework that leverages Transformer architectures to directly denoise continuous action sequences through a unified multimodal diffusion process. Departing from prior methods that condition denoising on fused embeddings via shallow networks, Dita employs in-context conditioning -- enabling fine-grained alignment between denoised actions and raw visual tokens from historical observations. This design explicitly models action deltas and environmental nuances. By scaling the diffusion action denoiser alongside the Transformer's scalability, Dita effectively integrates cross-embodiment datasets across diverse camera perspectives, observation scenes, tasks, and action spaces. Such synergy enhances robustness against various variances and facilitates the successful execution of long-horizon tasks. Evaluations across extensive benchmarks demonstrate state-of-the-art or comparative performance in simulation. Notably, Dita achieves robust real-world adaptation to environmental variances and complex long-horizon tasks through 10-shot finetuning, using only third-person camera inputs. The architecture establishes a versatile, lightweight and open-source baseline for generalist robot policy learning. Project Page: https://robodita.github.io.
>
---
#### [replaced 046] Multi-Modal Multi-Task (M3T) Federated Foundation Models for Embodied AI: Potentials and Challenges for Edge Integration
- **分类: cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.11191v2](http://arxiv.org/pdf/2505.11191v2)**

> **作者:** Kasra Borazjani; Payam Abdisarabshali; Fardis Nadimi; Naji Khosravan; Minghui Liwang; Xianbin Wang; Yiguang Hong; Seyyedali Hosseinalipour
>
> **备注:** Accepted for Publication in IEEE Internet of Things Magazine, 2025
>
> **摘要:** As embodied AI systems become increasingly multi-modal, personalized, and interactive, they must learn effectively from diverse sensory inputs, adapt continually to user preferences, and operate safely under resource and privacy constraints. These challenges expose a pressing need for machine learning models capable of swift, context-aware adaptation while balancing model generalization and personalization. Here, two methods emerge as suitable candidates, each offering parts of these capabilities: multi-modal multi-task foundation models (M3T-FMs) provide a pathway toward generalization across tasks and modalities, whereas federated learning (FL) offers the infrastructure for distributed, privacy-preserving model updates and user-level model personalization. However, when used in isolation, each of these approaches falls short of meeting the complex and diverse capability requirements of real-world embodied AI environments. In this vision paper, we introduce multi-modal multi-task federated foundation models (M3T-FFMs) for embodied AI, a new paradigm that unifies the strengths of M3T-FMs with the privacy-preserving distributed training nature of FL, enabling intelligent systems at the wireless edge. We collect critical deployment dimensions of M3T-FFMs in embodied AI ecosystems under a unified framework, which we name "EMBODY": Embodiment heterogeneity, Modality richness and imbalance, Bandwidth and compute constraints, On-device continual learning, Distributed control and autonomy, and Yielding safety, privacy, and personalization. For each, we identify concrete challenges and envision actionable research directions. We also present an evaluation framework for deploying M3T-FFMs in embodied AI systems, along with the associated trade-offs. Finally, we present a prototype implementation of M3T-FFMs and evaluate their energy and latency performance.
>
---
#### [replaced 047] Learning Multi-Stage Pick-and-Place with a Legged Mobile Manipulator
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.03859v3](http://arxiv.org/pdf/2509.03859v3)**

> **作者:** Haichao Zhang; Haonan Yu; Le Zhao; Andrew Choi; Qinxun Bai; Yiqing Yang; Wei Xu
>
> **备注:** Accepted to IEEE Robotics and Automation Letters (RA-L). Tech Report: arXiv:2501.09905
>
> **摘要:** Quadruped-based mobile manipulation presents significant challenges in robotics due to the diversity of required skills, the extended task horizon, and partial observability. After presenting a multi-stage pick-and-place task as a succinct yet sufficiently rich setup that captures key desiderata for quadruped-based mobile manipulation, we propose an approach that can train a visuo-motor policy entirely in simulation, and achieve nearly 80\% success in the real world. The policy efficiently performs search, approach, grasp, transport, and drop into actions, with emerged behaviors such as re-grasping and task chaining. We conduct an extensive set of real-world experiments with ablation studies highlighting key techniques for efficient training and effective sim-to-real transfer. Additional experiments demonstrate deployment across a variety of indoor and outdoor environments. Demo videos and additional resources are available on the project page: https://horizonrobotics.github.io/gail/SLIM.
>
---
#### [replaced 048] Parallel Computing Architectures for Robotic Applications: A Comprehensive Review
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2407.01011v2](http://arxiv.org/pdf/2407.01011v2)**

> **作者:** Md Rafid Islam
>
> **摘要:** With the growing complexity and capability of contemporary robotic systems, the necessity of sophisticated computing solutions to efficiently handle tasks such as real-time processing, sensor integration, decision-making, and control algorithms is also increasing. Conventional serial computing frequently fails to meet these requirements, underscoring the necessity for high-performance computing alternatives. Parallel computing, the utilization of several processing elements simultaneously to solve computational problems, offers a possible answer. Various parallel computing designs, such as multi-core CPUs, GPUs, FPGAs, and distributed systems, provide substantial enhancements in processing capacity and efficiency. By utilizing these architectures, robotic systems can attain improved performance in functionalities such as real-time image processing, sensor fusion, and path planning. The transformative potential of parallel computing architectures in advancing robotic technology has been underscored, real-life case studies of these architectures in the robotics field have been discussed, and comparisons are presented. Challenges pertaining to these architectures have been explored, and possible solutions have been mentioned for further research and enhancement of the robotic applications.
>
---
#### [replaced 049] An Effective Trajectory Planning and an Optimized Path Planning for a 6-Degree-of-Freedom Robot Manipulator
- **分类: cs.RO; cs.SC; math.AC; 68W30, 13P10, 13P25, 68U07, 68R10**

- **链接: [http://arxiv.org/pdf/2509.00828v2](http://arxiv.org/pdf/2509.00828v2)**

> **作者:** Takumu Okazaki; Akira Terui; Masahiko Mikawa
>
> **备注:** 26 pages
>
> **摘要:** An effective method for optimizing path planning for a specific model of a 6-degree-of-freedom (6-DOF) robot manipulator is presented as part of the motion planning of the manipulator using computer algebra. We assume that we are given a path in the form of a set of line segments that the end-effector should follow. We also assume that we have a method to solve the inverse kinematic problem of the manipulator at each via-point of the trajectory. The proposed method consists of three steps. First, we calculate the feasible region of the manipulator under a specific configuration of the end-effector. Next, we aim to find a trajectory on the line segments and a sequence of joint configurations the manipulator should follow to move the end-effector along the specified trajectory. Finally, we find the optimal combination of solutions to the inverse kinematic problem at each via-point along the trajectory by reducing the problem to a shortest-path problem of the graph and applying Dijkstra's algorithm. We show the effectiveness of the proposed method by experiments.
>
---
#### [replaced 050] Driver-Net: Multi-Camera Fusion for Assessing Driver Take-Over Readiness in Automated Vehicles
- **分类: cs.CV; cs.AI; cs.ET; cs.LG; cs.RO; I.4.9**

- **链接: [http://arxiv.org/pdf/2507.04139v2](http://arxiv.org/pdf/2507.04139v2)**

> **作者:** Mahdi Rezaei; Mohsen Azarmi
>
> **摘要:** Ensuring safe transition of control in automated vehicles requires an accurate and timely assessment of driver readiness. This paper introduces Driver-Net, a novel deep learning framework that fuses multi-camera inputs to estimate driver take-over readiness. Unlike conventional vision-based driver monitoring systems that focus on head pose or eye gaze, Driver-Net captures synchronised visual cues from the driver's head, hands, and body posture through a triple-camera setup. The model integrates spatio-temporal data using a dual-path architecture, comprising a Context Block and a Feature Block, followed by a cross-modal fusion strategy to enhance prediction accuracy. Evaluated on a diverse dataset collected from the University of Leeds Driving Simulator, the proposed method achieves an accuracy of up to 95.8% in driver readiness classification. This performance significantly enhances existing approaches and highlights the importance of multimodal and multi-view fusion. As a real-time, non-intrusive solution, Driver-Net contributes meaningfully to the development of safer and more reliable automated vehicles and aligns with new regulatory mandates and upcoming safety standards.
>
---
#### [replaced 051] Semi-SMD: Semi-Supervised Metric Depth Estimation via Surrounding Cameras for Autonomous Driving
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.19713v2](http://arxiv.org/pdf/2503.19713v2)**

> **作者:** Yusen Xie; Zhengmin Huang; Shaojie Shen; Jun Ma
>
> **摘要:** In this paper, we introduce Semi-SD, a novel metric depth estimation framework tailored for surrounding cameras equipment in autonomous driving. In this work, the input data consists of adjacent surrounding frames and camera parameters. We propose a unified spatial-temporal-semantic fusion module to construct the visual fused features. Cross-attention components for surrounding cameras and adjacent frames are utilized to focus on metric scale information refinement and temporal feature matching. Building on this, we propose a pose estimation framework using surrounding cameras, their corresponding estimated depths, and extrinsic parameters, which effectively address the scale ambiguity in multi-camera setups. Moreover, semantic world model and monocular depth estimation world model are integrated to supervised the depth estimation, which improve the quality of depth estimation. We evaluate our algorithm on DDAD and nuScenes datasets, and the results demonstrate that our method achieves state-of-the-art performance in terms of surrounding camera based depth estimation quality. The source code will be available on https://github.com/xieyuser/Semi-SD.
>
---
