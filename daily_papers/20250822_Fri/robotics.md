# 机器人 cs.RO

- **最新发布 18 篇**

- **更新 14 篇**

## 最新发布

#### [new 001] Decentralized Vision-Based Autonomous Aerial Wildlife Monitoring
- **分类: cs.RO; cs.AI; cs.CV; cs.MA; I.2.9**

- **简介: 该论文提出一种去中心化多无人机系统，用于自主监测野生动物。通过单相机视觉算法，在动态环境中实现大规模物种的鲁棒识别与跟踪，解决传统方法效率低、依赖集中控制的问题。**

- **链接: [http://arxiv.org/pdf/2508.15038v1](http://arxiv.org/pdf/2508.15038v1)**

> **作者:** Makram Chahine; William Yang; Alaa Maalouf; Justin Siriska; Ninad Jadhav; Daniel Vogt; Stephanie Gil; Robert Wood; Daniela Rus
>
> **摘要:** Wildlife field operations demand efficient parallel deployment methods to identify and interact with specific individuals, enabling simultaneous collective behavioral analysis, and health and safety interventions. Previous robotics solutions approach the problem from the herd perspective, or are manually operated and limited in scale. We propose a decentralized vision-based multi-quadrotor system for wildlife monitoring that is scalable, low-bandwidth, and sensor-minimal (single onboard RGB camera). Our approach enables robust identification and tracking of large species in their natural habitat. We develop novel vision-based coordination and tracking algorithms designed for dynamic, unstructured environments without reliance on centralized communication or control. We validate our system through real-world experiments, demonstrating reliable deployment in diverse field conditions.
>
---
#### [new 002] Mind and Motion Aligned: A Joint Evaluation IsaacSim Benchmark for Task Planning and Low-Level Policies in Mobile Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 论文提出Kitchen-R基准，解决任务规划与低层控制评估断层问题，通过模拟厨房环境整合两者，提供基线方法和多模式评估。**

- **链接: [http://arxiv.org/pdf/2508.15663v1](http://arxiv.org/pdf/2508.15663v1)**

> **作者:** Nikita Kachaev; Andrei Spiridonov; Andrey Gorodetsky; Kirill Muravyev; Nikita Oskolkov; Aditya Narendra; Vlad Shakhuro; Dmitry Makarov; Aleksandr I. Panov; Polina Fedotova; Alexey K. Kovalev
>
> **摘要:** Benchmarks are crucial for evaluating progress in robotics and embodied AI. However, a significant gap exists between benchmarks designed for high-level language instruction following, which often assume perfect low-level execution, and those for low-level robot control, which rely on simple, one-step commands. This disconnect prevents a comprehensive evaluation of integrated systems where both task planning and physical execution are critical. To address this, we propose Kitchen-R, a novel benchmark that unifies the evaluation of task planning and low-level control within a simulated kitchen environment. Built as a digital twin using the Isaac Sim simulator and featuring more than 500 complex language instructions, Kitchen-R supports a mobile manipulator robot. We provide baseline methods for our benchmark, including a task-planning strategy based on a vision-language model and a low-level control policy based on diffusion policy. We also provide a trajectory collection system. Our benchmark offers a flexible framework for three evaluation modes: independent assessment of the planning module, independent assessment of the control policy, and, crucially, an integrated evaluation of the whole system. Kitchen-R bridges a key gap in embodied AI research, enabling more holistic and realistic benchmarking of language-guided robotic agents.
>
---
#### [new 003] Lang2Lift: A Framework for Language-Guided Pallet Detection and Pose Estimation Integrated in Autonomous Outdoor Forklift Operation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Lang2Lift框架，通过自然语言指导实现户外叉车自主托盘检测与6D位姿估计，解决复杂环境下的自动化搬运问题，结合Florence-2/SAM-2和FoundationPose提升鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.15427v1](http://arxiv.org/pdf/2508.15427v1)**

> **作者:** Huy Hoang Nguyen; Johannes Huemer; Markus Murschitz; Tobias Glueck; Minh Nhat Vu; Andreas Kugi
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** The logistics and construction industries face persistent challenges in automating pallet handling, especially in outdoor environments with variable payloads, inconsistencies in pallet quality and dimensions, and unstructured surroundings. In this paper, we tackle automation of a critical step in pallet transport: the pallet pick-up operation. Our work is motivated by labor shortages, safety concerns, and inefficiencies in manually locating and retrieving pallets under such conditions. We present Lang2Lift, a framework that leverages foundation models for natural language-guided pallet detection and 6D pose estimation, enabling operators to specify targets through intuitive commands such as "pick up the steel beam pallet near the crane." The perception pipeline integrates Florence-2 and SAM-2 for language-grounded segmentation with FoundationPose for robust pose estimation in cluttered, multi-pallet outdoor scenes under variable lighting. The resulting poses feed into a motion planning module for fully autonomous forklift operation. We validate Lang2Lift on the ADAPT autonomous forklift platform, achieving 0.76 mIoU pallet segmentation accuracy on a real-world test dataset. Timing and error analysis demonstrate the system's robustness and confirm its feasibility for deployment in operational logistics and construction environments. Video demonstrations are available at https://eric-nguyen1402.github.io/lang2lift.github.io/
>
---
#### [new 004] Neural Robot Dynamics
- **分类: cs.RO; cs.AI; cs.GR; cs.LG**

- **简介: 该论文旨在解决机器人动态模拟中泛化能力不足的问题，提出NeRD神经模拟器，通过机器人中心的时空不变状态表示替代传统方法，实现跨任务和环境的稳定高效模拟。**

- **链接: [http://arxiv.org/pdf/2508.15755v1](http://arxiv.org/pdf/2508.15755v1)**

> **作者:** Jie Xu; Eric Heiden; Iretiayo Akinola; Dieter Fox; Miles Macklin; Yashraj Narang
>
> **摘要:** Accurate and efficient simulation of modern robots remains challenging due to their high degrees of freedom and intricate mechanisms. Neural simulators have emerged as a promising alternative to traditional analytical simulators, capable of efficiently predicting complex dynamics and adapting to real-world data; however, existing neural simulators typically require application-specific training and fail to generalize to novel tasks and/or environments, primarily due to inadequate representations of the global state. In this work, we address the problem of learning generalizable neural simulators for robots that are structured as articulated rigid bodies. We propose NeRD (Neural Robot Dynamics), learned robot-specific dynamics models for predicting future states for articulated rigid bodies under contact constraints. NeRD uniquely replaces the low-level dynamics and contact solvers in an analytical simulator and employs a robot-centric and spatially-invariant simulation state representation. We integrate the learned NeRD models as an interchangeable backend solver within a state-of-the-art robotics simulator. We conduct extensive experiments to show that the NeRD simulators are stable and accurate over a thousand simulation steps; generalize across tasks and environment configurations; enable policy learning exclusively in a neural engine; and, unlike most classical simulators, can be fine-tuned from real-world data to bridge the gap between simulation and reality.
>
---
#### [new 005] Sensing, Social, and Motion Intelligence in Embodied Navigation: A Comprehensive Survey
- **分类: cs.RO**

- **简介: 该论文研究具身导航任务，解决传统导航依赖显式定位和预定义地图的局限性。提出TOFRA框架，综述感知、社交、运动智能的最新进展，评估平台与指标，并识别关键研究挑战。**

- **链接: [http://arxiv.org/pdf/2508.15354v1](http://arxiv.org/pdf/2508.15354v1)**

> **作者:** Chaoran Xiong; Yulong Huang; Fangwen Yu; Changhao Chen; Yue Wang; Songpengchen Xia; Ling Pei
>
> **摘要:** Embodied navigation (EN) advances traditional navigation by enabling robots to perform complex egocentric tasks through sensing, social, and motion intelligence. In contrast to classic methodologies that rely on explicit localization and pre-defined maps, EN leverages egocentric perception and human-like interaction strategies. This survey introduces a comprehensive EN formulation structured into five stages: Transition, Observation, Fusion, Reward-policy construction, and Action (TOFRA). The TOFRA framework serves to synthesize the current state of the art, provide a critical review of relevant platforms and evaluation metrics, and identify critical open research challenges. A list of studies is available at https://github.com/Franky-X/Awesome-Embodied-Navigation.
>
---
#### [new 006] LLM-Driven Self-Refinement for Embodied Drone Task Planning
- **分类: cs.RO; cs.AI**

- **简介: 论文提出SRDrone系统，通过LLM驱动的连续状态评估与分层行为树优化，解决工业无人机在动态环境中的任务规划问题，提升任务成功率至96.25%。**

- **链接: [http://arxiv.org/pdf/2508.15501v1](http://arxiv.org/pdf/2508.15501v1)**

> **作者:** Deyu Zhang; Xicheng Zhang; Jiahao Li; Tingting Long; Xunhua Dai; Yongjian Fu; Jinrui Zhang; Ju Ren; Yaoxue Zhang
>
> **备注:** 14pages
>
> **摘要:** We introduce SRDrone, a novel system designed for self-refinement task planning in industrial-grade embodied drones. SRDrone incorporates two key technical contributions: First, it employs a continuous state evaluation methodology to robustly and accurately determine task outcomes and provide explanatory feedback. This approach supersedes conventional reliance on single-frame final-state assessment for continuous, dynamic drone operations. Second, SRDrone implements a hierarchical Behavior Tree (BT) modification model. This model integrates multi-level BT plan analysis with a constrained strategy space to enable structured reflective learning from experience. Experimental results demonstrate that SRDrone achieves a 44.87% improvement in Success Rate (SR) over baseline methods. Furthermore, real-world deployment utilizing an experience base optimized through iterative self-refinement attains a 96.25% SR. By embedding adaptive task refinement capabilities within an industrial-grade BT planning framework, SRDrone effectively integrates the general reasoning intelligence of Large Language Models (LLMs) with the stringent physical execution constraints inherent to embodied drones. Code is available at https://github.com/ZXiiiC/SRDrone.
>
---
#### [new 007] Exploiting Policy Idling for Dexterous Manipulation
- **分类: cs.RO; cs.LG; 68T40; I.2.9**

- **简介: 该论文针对机器人灵巧操作中策略因训练数据导致的"政策空闲"问题，提出通过检测并扰动空闲状态以提升鲁棒性的方法，实验证明在模拟与真实任务中显著提高成功率。**

- **链接: [http://arxiv.org/pdf/2508.15669v1](http://arxiv.org/pdf/2508.15669v1)**

> **作者:** Annie S. Chen; Philemon Brakel; Antonia Bronars; Annie Xie; Sandy Huang; Oliver Groth; Maria Bauza; Markus Wulfmeier; Nicolas Heess; Dushyant Rao
>
> **备注:** A similar version to this paper was accepted at IROS 2025
>
> **摘要:** Learning-based methods for dexterous manipulation have made notable progress in recent years. However, learned policies often still lack reliability and exhibit limited robustness to important factors of variation. One failure pattern that can be observed across many settings is that policies idle, i.e. they cease to move beyond a small region of states when they reach certain states. This policy idling is often a reflection of the training data. For instance, it can occur when the data contains small actions in areas where the robot needs to perform high-precision motions, e.g., when preparing to grasp an object or object insertion. Prior works have tried to mitigate this phenomenon e.g. by filtering the training data or modifying the control frequency. However, these approaches can negatively impact policy performance in other ways. As an alternative, we investigate how to leverage the detectability of idling behavior to inform exploration and policy improvement. Our approach, Pause-Induced Perturbations (PIP), applies perturbations at detected idling states, thus helping it to escape problematic basins of attraction. On a range of challenging simulated dual-arm tasks, we find that this simple approach can already noticeably improve test-time performance, with no additional supervision or training. Furthermore, since the robot tends to idle at critical points in a movement, we also find that learning from the resulting episodes leads to better iterative policy improvement compared to prior approaches. Our perturbation strategy also leads to a 15-35% improvement in absolute success rate on a real-world insertion task that requires complex multi-finger manipulation.
>
---
#### [new 008] Understanding and Utilizing Dynamic Coupling in Free-Floating Space Manipulators for On-Orbit Servicing
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对自由浮动空间机械臂轨迹规划问题，提出利用动态耦合进行优化的方法，通过SVD分析耦合特性并设计控制策略，提升轨道服务任务效率。**

- **链接: [http://arxiv.org/pdf/2508.15732v1](http://arxiv.org/pdf/2508.15732v1)**

> **作者:** Gargi Das; Daegyun Choi; Donghoon Kim
>
> **备注:** 17 pages, 7 figures, 2025 AAS/AIAA Astrodynamics Specialist Conference
>
> **摘要:** This study proposes a dynamic coupling-informed trajectory optimization algorithm for free-floating space manipulator systems (SMSs). Dynamic coupling between the base and the manipulator arms plays a critical role in influencing the system's behavior. While prior research has predominantly focused on minimizing this coupling, often overlooking its potential advantages, this work investigates how dynamic coupling can instead be leveraged to improve trajectory planning. Singular value decomposition (SVD) of the dynamic coupling matrix is employed to identify the dominant components governing coupling behavior. A quantitative metric is then formulated to characterize the strength and directionality of the coupling and is incorporated into a trajectory optimization framework. To assess the feasibility of the optimized trajectory, a sliding mode control-based tracking controller is designed to generate the required joint torque inputs. Simulation results demonstrate that explicitly accounting for dynamic coupling in trajectory planning enables more informed and potentially more efficient operation, offering new directions for the control of free-floating SMSs.
>
---
#### [new 009] Survey of Vision-Language-Action Models for Embodied Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文调研了用于具身操作的视觉-语言-动作模型，分析其结构、数据集等五个方面，总结挑战与未来方向，旨在提升机器人与环境交互能力。**

- **链接: [http://arxiv.org/pdf/2508.15201v1](http://arxiv.org/pdf/2508.15201v1)**

> **作者:** Haoran Li; Yuhui Chen; Wenbo Cui; Weiheng Liu; Kai Liu; Mingcai Zhou; Zhengtao Zhang; Dongbin Zhao
>
> **备注:** in Chinese language
>
> **摘要:** Embodied intelligence systems, which enhance agent capabilities through continuous environment interactions, have garnered significant attention from both academia and industry. Vision-Language-Action models, inspired by advancements in large foundation models, serve as universal robotic control frameworks that substantially improve agent-environment interaction capabilities in embodied intelligence systems. This expansion has broadened application scenarios for embodied AI robots. This survey comprehensively reviews VLA models for embodied manipulation. Firstly, it chronicles the developmental trajectory of VLA architectures. Subsequently, we conduct a detailed analysis of current research across 5 critical dimensions: VLA model structures, training datasets, pre-training methods, post-training methods, and model evaluation. Finally, we synthesize key challenges in VLA development and real-world deployment, while outlining promising future research directions.
>
---
#### [new 010] A Vision-Based Shared-Control Teleoperation Scheme for Controlling the Robotic Arm of a Four-Legged Robot
- **分类: cs.RO; cs.CV; cs.LG; cs.SY; eess.SY**

- **简介: 该论文提出基于视觉的共享控制方案，通过外部相机和机器学习模型实时映射操作员手腕动作到四足机器人机械臂，结合轨迹规划避免碰撞，提升工业遥控安全性与易用性。**

- **链接: [http://arxiv.org/pdf/2508.14994v1](http://arxiv.org/pdf/2508.14994v1)**

> **作者:** Murilo Vinicius da Silva; Matheus Hipolito Carvalho; Juliano Negri; Thiago Segreto; Gustavo J. G. Lahr; Ricardo V. Godoy; Marcelo Becker
>
> **摘要:** In hazardous and remote environments, robotic systems perform critical tasks demanding improved safety and efficiency. Among these, quadruped robots with manipulator arms offer mobility and versatility for complex operations. However, teleoperating quadruped robots is challenging due to the lack of integrated obstacle detection and intuitive control methods for the robotic arm, increasing collision risks in confined or dynamically changing workspaces. Teleoperation via joysticks or pads can be non-intuitive and demands a high level of expertise due to its complexity, culminating in a high cognitive load on the operator. To address this challenge, a teleoperation approach that directly maps human arm movements to the robotic manipulator offers a simpler and more accessible solution. This work proposes an intuitive remote control by leveraging a vision-based pose estimation pipeline that utilizes an external camera with a machine learning-based model to detect the operator's wrist position. The system maps these wrist movements into robotic arm commands to control the robot's arm in real-time. A trajectory planner ensures safe teleoperation by detecting and preventing collisions with both obstacles and the robotic arm itself. The system was validated on the real robot, demonstrating robust performance in real-time control. This teleoperation approach provides a cost-effective solution for industrial applications where safety, precision, and ease of use are paramount, ensuring reliable and intuitive robotic control in high-risk environments.
>
---
#### [new 011] Mag-Match: Magnetic Vector Field Features for Map Matching and Registration
- **分类: cs.RO**

- **简介: 该论文提出Mag-Match，通过提取3D磁矢量场的高阶导数特征，解决传统传感器在烟雾等恶劣环境下的地图匹配与注册问题，实现无需重力对齐的鲁棒配准。**

- **链接: [http://arxiv.org/pdf/2508.15300v1](http://arxiv.org/pdf/2508.15300v1)**

> **作者:** William McDonald; Cedric Le Gentil; Jennifer Wakulicz; Teresa Vidal-Calleja
>
> **备注:** To be published in IROS: IEEE/RSJ International Conference on Intelligent Robots and Systems, 2025
>
> **摘要:** Map matching and registration are essential tasks in robotics for localisation and integration of multi-session or multi-robot data. Traditional methods rely on cameras or LiDARs to capture visual or geometric information but struggle in challenging conditions like smoke or dust. Magnetometers, on the other hand, detect magnetic fields, revealing features invisible to other sensors and remaining robust in such environments. In this paper, we introduce Mag-Match, a novel method for extracting and describing features in 3D magnetic vector field maps to register different maps of the same area. Our feature descriptor, based on higher-order derivatives of magnetic field maps, is invariant to global orientation, eliminating the need for gravity-aligned mapping. To obtain these higher-order derivatives map-wide given point-wise magnetometer data, we leverage a physics-informed Gaussian Process to perform efficient and recursive probabilistic inference of both the magnetic field and its derivatives. We evaluate Mag-Match in simulated and real-world experiments against a SIFT-based approach, demonstrating accurate map-to-map, robot-to-map, and robot-to-robot transformations - even without initial gravitational alignment.
>
---
#### [new 012] Hardware Implementation of a Zero-Prior-Knowledge Approach to Lifelong Learning in Kinematic Control of Tendon-Driven Quadrupeds
- **分类: cs.RO**

- **简介: 该论文研究四足机器人在零先验知识下的终身学习控制，解决环境动态变化中的适应性问题。提出生物启发的G2P算法，通过泛化运动与精炼试验实现腱驱动机器人快速学习非凸循环运动，验证硬件系统在短时间内达成功能控制的可行性。**

- **链接: [http://arxiv.org/pdf/2508.15160v1](http://arxiv.org/pdf/2508.15160v1)**

> **作者:** Hesam Azadjou; Suraj Chakravarthi Raja; Ali Marjaninejad; Francisco J. Valero-Cuevas
>
> **摘要:** Like mammals, robots must rapidly learn to control their bodies and interact with their environment despite incomplete knowledge of their body structure and surroundings. They must also adapt to continuous changes in both. This work presents a bio-inspired learning algorithm, General-to-Particular (G2P), applied to a tendon-driven quadruped robotic system developed and fabricated in-house. Our quadruped robot undergoes an initial five-minute phase of generalized motor babbling, followed by 15 refinement trials (each lasting 20 seconds) to achieve specific cyclical movements. This process mirrors the exploration-exploitation paradigm observed in mammals. With each refinement, the robot progressively improves upon its initial "good enough" solution. Our results serve as a proof-of-concept, demonstrating the hardware-in-the-loop system's ability to learn the control of a tendon-driven quadruped with redundancies in just a few minutes to achieve functional and adaptive cyclical non-convex movements. By advancing autonomous control in robotic locomotion, our approach paves the way for robots capable of dynamically adjusting to new environments, ensuring sustained adaptability and performance.
>
---
#### [new 013] In-Context Iterative Policy Improvement for Dynamic Manipulation
- **分类: cs.RO**

- **简介: 本文研究动态操作中的策略改进，利用上下文学习解决高维、复杂动态和部分可观测性问题，通过迭代调整参数策略，在模拟和物理机器人上优于其他方法。**

- **链接: [http://arxiv.org/pdf/2508.15021v1](http://arxiv.org/pdf/2508.15021v1)**

> **作者:** Mark Van der Merwe; Devesh Jha
>
> **备注:** 14 pages. Accepted at CoRL 2025
>
> **摘要:** Attention-based architectures trained on internet-scale language data have demonstrated state of the art reasoning ability for various language-based tasks, such as logic problems and textual reasoning. Additionally, these Large Language Models (LLMs) have exhibited the ability to perform few-shot prediction via in-context learning, in which input-output examples provided in the prompt are generalized to new inputs. This ability furthermore extends beyond standard language tasks, enabling few-shot learning for general patterns. In this work, we consider the application of in-context learning with pre-trained language models for dynamic manipulation. Dynamic manipulation introduces several crucial challenges, including increased dimensionality, complex dynamics, and partial observability. To address this, we take an iterative approach, and formulate our in-context learning problem to predict adjustments to a parametric policy based on previous interactions. We show across several tasks in simulation and on a physical robot that utilizing in-context learning outperforms alternative methods in the low data regime. Video summary of this work and experiments can be found https://youtu.be/2inxpdrq74U?si=dAdDYsUEr25nZvRn.
>
---
#### [new 014] GraspQP: Differentiable Optimization of Force Closure for Diverse and Robust Dexterous Grasping
- **分类: cs.RO**

- **简介: 论文针对灵巧抓取中的力闭合优化，解决现有方法生成抓取多样性和物理可行性不足的问题，提出可微分能量公式与优化方法，并提供新数据集。**

- **链接: [http://arxiv.org/pdf/2508.15002v1](http://arxiv.org/pdf/2508.15002v1)**

> **作者:** René Zurbrügg; Andrei Cramariuc; Marco Hutter
>
> **摘要:** Dexterous robotic hands enable versatile interactions due to the flexibility and adaptability of multi-fingered designs, allowing for a wide range of task-specific grasp configurations in diverse environments. However, to fully exploit the capabilities of dexterous hands, access to diverse and high-quality grasp data is essential -- whether for developing grasp prediction models from point clouds, training manipulation policies, or supporting high-level task planning with broader action options. Existing approaches for dataset generation typically rely on sampling-based algorithms or simplified force-closure analysis, which tend to converge to power grasps and often exhibit limited diversity. In this work, we propose a method to synthesize large-scale, diverse, and physically feasible grasps that extend beyond simple power grasps to include refined manipulations, such as pinches and tri-finger precision grasps. We introduce a rigorous, differentiable energy formulation of force closure, implicitly defined through a Quadratic Program (QP). Additionally, we present an adjusted optimization method (MALA*) that improves performance by dynamically rejecting gradient steps based on the distribution of energy values across all samples. We extensively evaluate our approach and demonstrate significant improvements in both grasp diversity and the stability of final grasp predictions. Finally, we provide a new, large-scale grasp dataset for 5,700 objects from DexGraspNet, comprising five different grippers and three distinct grasp types. Dataset and Code:https://graspqp.github.io/
>
---
#### [new 015] Open-Universe Assistance Games
- **分类: cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文研究Embodied AI在开放环境中的目标推理任务，提出GOOD方法通过对话提取自然语言目标，利用LLM模拟用户意图进行概率推理，无需大规模数据，实验显示优于基线。**

- **链接: [http://arxiv.org/pdf/2508.15119v1](http://arxiv.org/pdf/2508.15119v1)**

> **作者:** Rachel Ma; Jingyi Qu; Andreea Bobu; Dylan Hadfield-Menell
>
> **备注:** 7 pages + 2 pages references + 7 pages appendix
>
> **摘要:** Embodied AI agents must infer and act in an interpretable way on diverse human goals and preferences that are not predefined. To formalize this setting, we introduce Open-Universe Assistance Games (OU-AGs), a framework where the agent must reason over an unbounded and evolving space of possible goals. In this context, we introduce GOOD (GOals from Open-ended Dialogue), a data-efficient, online method that extracts goals in the form of natural language during an interaction with a human, and infers a distribution over natural language goals. GOOD prompts an LLM to simulate users with different complex intents, using its responses to perform probabilistic inference over candidate goals. This approach enables rich goal representations and uncertainty estimation without requiring large offline datasets. We evaluate GOOD in a text-based grocery shopping domain and in a text-operated simulated household robotics environment (AI2Thor), using synthetic user profiles. Our method outperforms a baseline without explicit goal tracking, as confirmed by both LLM-based and human evaluations.
>
---
#### [new 016] Discrete VHCs for Propeller Motion of a Devil-Stick using purely Impulsive Inputs
- **分类: eess.SY; cs.RO; cs.SY; math.DS**

- **简介: 论文研究如何用脉冲输入控制魔鬼棒在垂直面实现螺旋运动，提出离散虚拟约束方法，设计控制器并验证有效性。**

- **链接: [http://arxiv.org/pdf/2508.15040v1](http://arxiv.org/pdf/2508.15040v1)**

> **作者:** Aakash Khandelwal; Ranjan Mukherjee
>
> **备注:** 16 pages, 11 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** The control problem of realizing propeller motion of a devil-stick in the vertical plane using impulsive forces applied normal to the stick is considered. This problem is an example of underactuated robotic juggling and has not been considered in the literature before. Inspired by virtual holonomic constraints, the concept of discrete virtual holonomic constraints (DVHC) is introduced for the first time to solve this orbital stabilization problem. At the discrete instants when impulsive inputs are applied, the location of the center-of-mass of the devil-stick is specified in terms of its orientation angle. This yields the discrete zero dynamics (DZD), which provides conditions for stable propeller motion. In the limiting case, when the rotation angle between successive applications of impulsive inputs is chosen to be arbitrarily small, the problem reduces to that of propeller motion under continuous forcing. A controller that enforces the DVHC, and an orbit stabilizing controller based on the impulse controlled Poincar\'e map approach are presented. The efficacy of the approach to trajectory design and stabilization is validated through simulations.
>
---
#### [new 017] Learning to Drive Ethically: Embedding Moral Reasoning into Autonomous Driving
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 论文提出分层Safe RL框架，整合道德推理与驾驶目标，通过复合伦理风险成本和动态经验回放，结合PID/Stanley控制器生成安全轨迹，在真实数据上验证，有效降低伦理风险并提升性能。**

- **链接: [http://arxiv.org/pdf/2508.14926v1](http://arxiv.org/pdf/2508.14926v1)**

> **作者:** Dianzhao Li; Ostap Okhrin
>
> **摘要:** Autonomous vehicles hold great promise for reducing traffic fatalities and improving transportation efficiency, yet their widespread adoption hinges on embedding robust ethical reasoning into routine and emergency maneuvers. Here, we present a hierarchical Safe Reinforcement Learning (Safe RL) framework that explicitly integrates moral considerations with standard driving objectives. At the decision level, a Safe RL agent is trained using a composite ethical risk cost, combining collision probability and harm severity, to generate high-level motion targets. A dynamic Prioritized Experience Replay mechanism amplifies learning from rare but critical, high-risk events. At the execution level, polynomial path planning coupled with Proportional-Integral-Derivative (PID) and Stanley controllers translates these targets into smooth, feasible trajectories, ensuring both accuracy and comfort. We train and validate our approach on rich, real-world traffic datasets encompassing diverse vehicles, cyclists, and pedestrians, and demonstrate that it outperforms baseline methods in reducing ethical risk and maintaining driving performance. To our knowledge, this is the first study of ethical decision-making for autonomous vehicles via Safe RL in real-world scenarios. Our results highlight the potential of combining formal control theory and data-driven learning to advance ethically accountable autonomy in complex, human-mixed traffic environments.
>
---
#### [new 018] You Only Pose Once: A Minimalist's Detection Transformer for Monocular RGB Category-level 9D Multi-Object Pose Estimation
- **分类: cs.CV; cs.RO**

- **简介: 论文提出YOPO框架，解决单目RGB图像中多物体类别级9D姿态估计问题，通过统一检测与姿态估计，无需伪深度或CAD模型，实现端到端训练，刷新基准性能。**

- **链接: [http://arxiv.org/pdf/2508.14965v1](http://arxiv.org/pdf/2508.14965v1)**

> **作者:** Hakjin Lee; Junghoon Seo; Jaehoon Sim
>
> **备注:** https://mikigom.github.io/YOPO-project-page
>
> **摘要:** Accurately recovering the full 9-DoF pose of unseen instances within specific categories from a single RGB image remains a core challenge for robotics and automation. Most existing solutions still rely on pseudo-depth, CAD models, or multi-stage cascades that separate 2D detection from pose estimation. Motivated by the need for a simpler, RGB-only alternative that learns directly at the category level, we revisit a longstanding question: Can object detection and 9-DoF pose estimation be unified with high performance, without any additional data? We show that they can with our method, YOPO, a single-stage, query-based framework that treats category-level 9-DoF estimation as a natural extension of 2D detection. YOPO augments a transformer detector with a lightweight pose head, a bounding-box-conditioned translation module, and a 6D-aware Hungarian matching cost. The model is trained end-to-end only with RGB images and category-level pose labels. Despite its minimalist design, YOPO sets a new state of the art on three benchmarks. On the REAL275 dataset, it achieves 79.6% $\rm{IoU}_{50}$ and 54.1% under the $10^\circ$$10{\rm{cm}}$ metric, surpassing prior RGB-only methods and closing much of the gap to RGB-D systems. The code, models, and additional qualitative results can be found on our project.
>
---
## 更新

#### [replaced 001] Embodied Long Horizon Manipulation with Closed-loop Code Generation and Incremental Few-shot Adaptation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.21969v3](http://arxiv.org/pdf/2503.21969v3)**

> **作者:** Yuan Meng; Xiangtong Yao; Haihui Ye; Yirui Zhou; Shengqiang Zhang; Zhenguo Sun; Xukun Li; Zhenshan Bing; Alois Knoll
>
> **备注:** update ICRA 6 page
>
> **摘要:** Embodied long-horizon manipulation requires robotic systems to process multimodal inputs-such as vision and natural language-and translate them into executable actions. However, existing learning-based approaches often depend on large, task-specific datasets and struggle to generalize to unseen scenarios. Recent methods have explored using large language models (LLMs) as high-level planners that decompose tasks into subtasks using natural language and guide pretrained low-level controllers. Yet, these approaches assume perfect execution from low-level policies, which is unrealistic in real-world environments with noise or suboptimal behaviors. To overcome this, we fully discard the pretrained low-level policy and instead use the LLM to directly generate executable code plans within a closed-loop framework. Our planner employs chain-of-thought (CoT)-guided few-shot learning with incrementally structured examples to produce robust and generalizable task plans. Complementing this, a reporter evaluates outcomes using RGB-D and delivers structured feedback, enabling recovery from misalignment and replanning under partial observability. This design eliminates per-step inference, reduces computational overhead, and limits error accumulation that was observed in previous methods. Our framework achieves state-of-the-art performance on 30+ diverse seen and unseen long-horizon tasks across LoHoRavens, CALVIN, Franka Kitchen, and cluttered real-world settings.
>
---
#### [replaced 002] Towards High Precision: An Adaptive Self-Supervised Learning Framework for Force-Based Verification
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.02153v2](http://arxiv.org/pdf/2508.02153v2)**

> **作者:** Zebin Duan; Frederik Hagelskjær; Aljaz Kramberger; Juan Heredia; Norbert Krüger
>
> **备注:** 7 pages, 7 figures, 3 tables
>
> **摘要:** The automation of robotic tasks requires high precision and adaptability, particularly in force-based operations such as insertions. Traditional learning-based approaches either rely on static datasets, which limit their ability to generalize, or require frequent manual intervention to maintain good performances. As a result, ensuring long-term reliability without human supervision remains a significant challenge. To address this, we propose an adaptive self-supervised learning framework for insertion classification that continuously improves its precision over time. The framework operates in real-time, incrementally refining its classification decisions by integrating newly acquired force data. Unlike conventional methods, it does not rely on pre-collected datasets but instead evolves dynamically with each task execution. Through real-world experiments, we demonstrate how the system progressively reduces execution time while maintaining near-perfect precision as more samples are processed. This adaptability ensures long-term reliability in force-based robotic tasks while minimizing the need for manual intervention.
>
---
#### [replaced 003] CaLiV: LiDAR-to-Vehicle Calibration of Arbitrary Sensor Setups
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.01987v2](http://arxiv.org/pdf/2504.01987v2)**

> **作者:** Ilir Tahiraj; Markus Edinger; Dominik Kulmer; Markus Lienkamp
>
> **摘要:** In autonomous systems, sensor calibration is essential for safe and efficient navigation in dynamic environments. Accurate calibration is a prerequisite for reliable perception and planning tasks such as object detection and obstacle avoidance. Many existing LiDAR calibration methods require overlapping fields of view, while others use external sensing devices or postulate a feature-rich environment. In addition, Sensor-to-Vehicle calibration is not supported by the vast majority of calibration algorithms. In this work, we propose a novel target-based technique for extrinsic Sensor-to-Sensor and Sensor-to-Vehicle calibration of multi-LiDAR systems called CaLiV. This algorithm works for non-overlapping fields of view and does not require any external sensing devices. First, we apply motion to produce field of view overlaps and utilize a simple Unscented Kalman Filter to obtain vehicle poses. Then, we use the Gaussian mixture model-based registration framework GMMCalib to align the point clouds in a common calibration frame. Finally, we reduce the task of recovering the sensor extrinsics to a minimization problem. We show that both translational and rotational Sensor-to-Sensor errors can be solved accurately by our method. In addition, all Sensor-to-Vehicle rotation angles can also be calibrated with high accuracy. We validate the simulation results in real-world experiments. The code is open-source and available on https://github.com/TUMFTM/CaLiV.
>
---
#### [replaced 004] UAV-ON: A Benchmark for Open-World Object Goal Navigation with Aerial Agents
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.00288v3](http://arxiv.org/pdf/2508.00288v3)**

> **作者:** Jianqiang Xiao; Yuexuan Sun; Yixin Shao; Boxi Gan; Rongqiang Liu; Yanjing Wu; Weili Gua; Xiang Deng
>
> **备注:** Accepted to ACM MM Dataset Track 2025
>
> **摘要:** Aerial navigation is a fundamental yet underexplored capability in embodied intelligence, enabling agents to operate in large-scale, unstructured environments where traditional navigation paradigms fall short. However, most existing research follows the Vision-and-Language Navigation (VLN) paradigm, which heavily depends on sequential linguistic instructions, limiting its scalability and autonomy. To address this gap, we introduce UAV-ON, a benchmark for large-scale Object Goal Navigation (ObjectNav) by aerial agents in open-world environments, where agents operate based on high-level semantic goals without relying on detailed instructional guidance as in VLN. UAV-ON comprises 14 high-fidelity Unreal Engine environments with diverse semantic regions and complex spatial layouts, covering urban, natural, and mixed-use settings. It defines 1270 annotated target objects, each characterized by an instance-level instruction that encodes category, physical footprint, and visual descriptors, allowing grounded reasoning. These instructions serve as semantic goals, introducing realistic ambiguity and complex reasoning challenges for aerial agents. To evaluate the benchmark, we implement several baseline methods, including Aerial ObjectNav Agent (AOA), a modular policy that integrates instruction semantics with egocentric observations for long-horizon, goal-directed exploration. Empirical results show that all baselines struggle in this setting, highlighting the compounded challenges of aerial navigation and semantic goal grounding. UAV-ON aims to advance research on scalable UAV autonomy driven by semantic goal descriptions in complex real-world environments.
>
---
#### [replaced 005] Continual Learning for Multimodal Data Fusion of a Soft Gripper
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2409.13792v2](http://arxiv.org/pdf/2409.13792v2)**

> **作者:** Nilay Kushawaha; Egidio Falotico
>
> **备注:** Accepted in Wiley Advanced Robotics Research
>
> **摘要:** Continual learning (CL) refers to the ability of an algorithm to continuously and incrementally acquire new knowledge from its environment while retaining previously learned information. A model trained on one data modality often fails when tested with a different modality. A straightforward approach might be to fuse the two modalities by concatenating their features and training the model on the fused data. However, this requires retraining the model from scratch each time it encounters a new domain. In this paper, we introduce a continual learning algorithm capable of incrementally learning different data modalities by leveraging both class-incremental and domain-incremental learning scenarios in an artificial environment where labeled data is scarce, yet non-iid (independent and identical distribution) unlabeled data from the environment is plentiful. The proposed algorithm is efficient and only requires storing prototypes for each class. We evaluate the algorithm's effectiveness on a challenging custom multimodal dataset comprising of tactile data from a soft pneumatic gripper, and visual data from non-stationary images of objects extracted from video sequences. Additionally, we conduct an ablation study on the custom dataset and the Core50 dataset to highlight the contributions of different components of the algorithm. To further demonstrate the robustness of the algorithm, we perform a real-time experiment for object classification using the soft gripper and an external independent camera setup, all synchronized with the Robot Operating System (ROS) framework.
>
---
#### [replaced 006] TripleMixer: A 3D Point Cloud Denoising Model for Adverse Weather
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2408.13802v2](http://arxiv.org/pdf/2408.13802v2)**

> **作者:** Xiongwei Zhao; Congcong Wen; Xu Zhu; Yang Wang; Haojie Bai; Wenhao Dou
>
> **备注:** 15 pages, submit to IEEE TIP
>
> **摘要:** Adverse weather conditions such as snow, fog, and rain pose significant challenges to LiDAR-based perception models by introducing noise and corrupting point cloud measurements. To address this issue, we propose TripleMixer, a robust and efficient point cloud denoising network that integrates spatial, frequency, and channel-wise processing through three specialized mixer modules. TripleMixer effectively suppresses high-frequency noise while preserving essential geometric structures and can be seamlessly deployed as a plug-and-play module within existing LiDAR perception pipelines. To support the development and evaluation of denoising methods, we construct two large-scale simulated datasets, Weather-KITTI and Weather-NuScenes, covering diverse weather scenarios with dense point-wise semantic and noise annotations. Based on these datasets, we establish four benchmarks: Denoising, Semantic Segmentation (SS), Place Recognition (PR), and Object Detection (OD). These benchmarks enable systematic evaluation of denoising generalization, transferability, and downstream impact under both simulated and real-world adverse weather conditions. Extensive experiments demonstrate that TripleMixer achieves state-of-the-art denoising performance and yields substantial improvements across all downstream tasks without requiring retraining. Our results highlight the potential of denoising as a task-agnostic preprocessing strategy to enhance LiDAR robustness in real-world autonomous driving applications.
>
---
#### [replaced 007] A MILP-Based Solution to Multi-Agent Motion Planning and Collision Avoidance in Constrained Environments
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2506.21982v2](http://arxiv.org/pdf/2506.21982v2)**

> **作者:** Akshay Jaitly; Jack Cline; Siavash Farzan
>
> **备注:** Accepted to 2025 IEEE International Conference on Automation Science and Engineering (CASE 2025). This arXiv version adds a supplementary appendix with figures not in the IEEE proceedings
>
> **摘要:** We propose a mixed-integer linear program (MILP) for multi-agent motion planning that embeds Polytopic Action-based Motion Planning (PAAMP) into a sequence-then-solve pipeline. Region sequences confine each agent to adjacent convex polytopes, while a big-M hyperplane model enforces inter-agent separation. Collision constraints are applied only to agents sharing or neighboring a region, which reduces binary variables exponentially compared with naive formulations. An L1 path-length-plus-acceleration cost yields smooth trajectories. We prove finite-time convergence and demonstrate on representative multi-agent scenarios with obstacles that our formulation produces collision-free trajectories an order of magnitude faster than an unstructured MILP baseline.
>
---
#### [replaced 008] ILeSiA: Interactive Learning of Robot Situational Awareness from Camera Input
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2409.20173v2](http://arxiv.org/pdf/2409.20173v2)**

> **作者:** Petr Vanc; Giovanni Franzese; Jan Kristof Behrens; Cosimo Della Santina; Karla Stepanova; Jens Kober; Robert Babuska
>
> **备注:** 8 pages, 9 figures. Accepted to IEEE Robotics and Automation Letters (Early Access)
>
> **摘要:** Learning from demonstration is a promising approach for teaching robots new skills. However, a central challenge in the execution of acquired skills is the ability to recognize faults and prevent failures. This is essential because demonstrations typically cover only a limited set of scenarios and often only the successful ones. During task execution, unforeseen situations may arise, such as changes in the robot's environment or interaction with human operators. To recognize such situations, this paper focuses on teaching the robot situational awareness by using a camera input and labeling frames as safe or risky. We train a Gaussian Process (GP) regression model fed by a low-dimensional latent space representation of the input images. The model outputs a continuous risk score ranging from zero to one, quantifying the degree of risk at each timestep. This allows for pausing task execution in unsafe situations and directly adding new training data, labeled by the human user. Our experiments on a robotic manipulator show that the proposed method can reliably detect both known and novel faults using only a single example for each new fault. In contrast, a standard multi-layer perceptron (MLP) performs well only on faults it has encountered during training. Our method enables the next generation of cobots to be rapidly deployed with easy-to-set-up, vision-based risk assessment, proactively safeguarding humans and detecting misaligned parts or missing objects before failures occur. We provide all the code and data required to reproduce our experiments at imitrob.ciirc.cvut.cz/publications/ilesia.
>
---
#### [replaced 009] Polytope Volume Monitoring Problem: Formulation and Solution via Parametric Linear Program Based Control Barrier Function
- **分类: math.OC; cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.12546v3](http://arxiv.org/pdf/2503.12546v3)**

> **作者:** Shizhen Wu; Jinyang Dong; Xu Fang; Ning Sun; Yongchun Fang
>
> **备注:** An extension version of the accepted CDC2025
>
> **摘要:** Motivated by the latest research on feasible space monitoring of multiple control barrier functions (CBFs) as well as polytopic collision avoidance, this paper studies the Polytope Volume Monitoring (PVM) problem, whose goal is to design a control law for inputs of nonlinear systems to prevent the volume of some state-dependent polytope from decreasing to zero. Recent studies have explored the idea of applying Chebyshev ball method in optimization theory to solve the case study of PVM; however, the underlying difficulties caused by nonsmoothness have not been addressed. This paper continues the study on this topic, where our main contribution is to establish the relationship between nonsmooth CBF and parametric optimization theory through directional derivatives for the first time, to solve PVM problems more conveniently. In detail, inspired by Chebyshev ball approach, a parametric linear program (PLP) based nonsmooth barrier function candidate is established for PVM, and then, sufficient conditions for it to be a nonsmooth CBF are proposed, based on which a quadratic program (QP) based safety filter with guaranteed feasibility is proposed to address PVM problems. Finally, a numerical simulation example is given to show the efficiency of the proposed safety filter.
>
---
#### [replaced 010] An Informative Planning Framework for Target Tracking and Active Mapping in Dynamic Environments with ASVs
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.14636v2](http://arxiv.org/pdf/2508.14636v2)**

> **作者:** Sanjeev Ramkumar Sudha; Marija Popović; Erlend M. Coates
>
> **备注:** Submitted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Mobile robot platforms are increasingly being used to automate information gathering tasks such as environmental monitoring. Efficient target tracking in dynamic environments is critical for applications such as search and rescue and pollutant cleanups. In this letter, we study active mapping of floating targets that drift due to environmental disturbances such as wind and currents. This is a challenging problem as it involves predicting both spatial and temporal variations in the map due to changing conditions. We propose an informative path planning framework to map an arbitrary number of moving targets with initially unknown positions in dynamic environments. A key component of our approach is a spatiotemporal prediction network that predicts target position distributions over time. We propose an adaptive planning objective for target tracking that leverages these predictions. Simulation experiments show that our proposed planning objective improves target tracking performance compared to existing methods that consider only entropy reduction as the planning objective. Finally, we validate our approach in field tests using an autonomous surface vehicle, showcasing its ability to track targets in real-world monitoring scenarios.
>
---
#### [replaced 011] CaRL: Learning Scalable Planning Policies with Simple Rewards
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.17838v3](http://arxiv.org/pdf/2504.17838v3)**

> **作者:** Bernhard Jaeger; Daniel Dauner; Jens Beißwenger; Simon Gerstenecker; Kashyap Chitta; Andreas Geiger
>
> **备注:** Accepted at the Conference on Robot Learning 2025
>
> **摘要:** We investigate reinforcement learning (RL) for privileged planning in autonomous driving. State-of-the-art approaches for this task are rule-based, but these methods do not scale to the long tail. RL, on the other hand, is scalable and does not suffer from compounding errors like imitation learning. Contemporary RL approaches for driving use complex shaped rewards that sum multiple individual rewards, \eg~progress, position, or orientation rewards. We show that PPO fails to optimize a popular version of these rewards when the mini-batch size is increased, which limits the scalability of these approaches. Instead, we propose a new reward design based primarily on optimizing a single intuitive reward term: route completion. Infractions are penalized by terminating the episode or multiplicatively reducing route completion. We find that PPO scales well with higher mini-batch sizes when trained with our simple reward, even improving performance. Training with large mini-batch sizes enables efficient scaling via distributed data parallelism. We scale PPO to 300M samples in CARLA and 500M samples in nuPlan with a single 8-GPU node. The resulting model achieves 64 DS on the CARLA longest6 v2 benchmark, outperforming other RL methods with more complex rewards by a large margin. Requiring only minimal adaptations from its use in CARLA, the same method is the best learning-based approach on nuPlan. It scores 91.3 in non-reactive and 90.6 in reactive traffic on the Val14 benchmark while being an order of magnitude faster than prior work.
>
---
#### [replaced 012] Taming VR Teleoperation and Learning from Demonstration for Multi-Task Bimanual Table Service Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.14542v2](http://arxiv.org/pdf/2508.14542v2)**

> **作者:** Weize Li; Zhengxiao Han; Lixin Xu; Xiangyu Chen; Harrison Bounds; Chenrui Zhang; Yifan Xu
>
> **备注:** Technical Report of First-place/Champion solution at IEEE ICRA 2025 What Bimanuals Can Do (WBCD) Challenge - Table Services Track
>
> **摘要:** This technical report presents the champion solution of the Table Service Track in the ICRA 2025 What Bimanuals Can Do (WBCD) competition. We tackled a series of demanding tasks under strict requirements for speed, precision, and reliability: unfolding a tablecloth (deformable-object manipulation), placing a pizza into the container (pick-and-place), and opening and closing a food container with the lid. Our solution combines VR-based teleoperation and Learning from Demonstrations (LfD) to balance robustness and autonomy. Most subtasks were executed through high-fidelity remote teleoperation, while the pizza placement was handled by an ACT-based policy trained from 100 in-person teleoperated demonstrations with randomized initial configurations. By carefully integrating scoring rules, task characteristics, and current technical capabilities, our approach achieved both high efficiency and reliability, ultimately securing the first place in the competition.
>
---
#### [replaced 013] EgoDex: Learning Dexterous Manipulation from Large-Scale Egocentric Video
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.11709v2](http://arxiv.org/pdf/2505.11709v2)**

> **作者:** Ryan Hoque; Peide Huang; David J. Yoon; Mouli Sivapurapu; Jian Zhang
>
> **摘要:** Imitation learning for manipulation has a well-known data scarcity problem. Unlike natural language and 2D computer vision, there is no Internet-scale corpus of data for dexterous manipulation. One appealing option is egocentric human video, a passively scalable data source. However, existing large-scale datasets such as Ego4D do not have native hand pose annotations and do not focus on object manipulation. To this end, we use Apple Vision Pro to collect EgoDex: the largest and most diverse dataset of dexterous human manipulation to date. EgoDex has 829 hours of egocentric video with paired 3D hand and finger tracking data collected at the time of recording, where multiple calibrated cameras and on-device SLAM can be used to precisely track the pose of every joint of each hand. The dataset covers a wide range of diverse manipulation behaviors with everyday household objects in 194 different tabletop tasks ranging from tying shoelaces to folding laundry. Furthermore, we train and systematically evaluate imitation learning policies for hand trajectory prediction on the dataset, introducing metrics and benchmarks for measuring progress in this increasingly important area. By releasing this large-scale dataset, we hope to push the frontier of robotics, computer vision, and foundation models. EgoDex is publicly available for download at https://github.com/apple/ml-egodex.
>
---
#### [replaced 014] Automatic Geometric Decomposition for Analytical Inverse Kinematics
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.14815v2](http://arxiv.org/pdf/2409.14815v2)**

> **作者:** Daniel Ostermeier; Jonathan Külz; Matthias Althoff
>
> **备注:** Website: https://eaik.cps.cit.tum.de/
>
> **摘要:** Calculating the inverse kinematics (IK) is a fundamental challenge in robotics. Compared to numerical or learning-based approaches, analytical IK provides higher efficiency and accuracy. However, existing analytical approaches are difficult to use in most applications, as they require human ingenuity in the derivation process, are numerically unstable, or rely on time-consuming symbolic manipulation. In contrast, we propose a method that, for the first time, enables an analytical IK derivation and computation in less than a millisecond in total. Our work is based on an automatic online decomposition of the IK into pre-solved, numerically stable subproblems via a kinematic classification of the respective manipulator. In numerical experiments, we demonstrate that our approach is orders of magnitude faster in deriving the IK than existing tools that employ symbolic manipulation. Following this one-time derivation, our method matches and often surpasses baselines, such as IKFast, in terms of speed and accuracy during the computation of explicit IK solutions. Finally, we provide an open-source C++ toolbox with Python wrappers that substantially reduces the entry barrier to using analytical IK in applications like rapid prototyping and kinematic robot design.
>
---
