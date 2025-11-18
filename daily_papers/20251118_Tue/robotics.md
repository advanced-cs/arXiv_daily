# 机器人 cs.RO

- **最新发布 66 篇**

- **更新 32 篇**

## 最新发布

#### [new 001] DiffuDepGrasp: Diffusion-based Depth Noise Modeling Empowers Sim2Real Robotic Grasping
- **分类: cs.RO**

- **简介: 论文提出DiffuDepGrasp框架，解决仿真到现实的机械臂抓取任务中深度图噪声导致的域差距问题。通过扩散模型合成真实传感器噪声，在仅用仿真数据训练的情况下实现零样本迁移，提升抓取成功率并降低部署复杂度。**

- **链接: [https://arxiv.org/pdf/2511.12912v1](https://arxiv.org/pdf/2511.12912v1)**

> **作者:** Yingting Zhou; Wenbo Cui; Weiheng Liu; Guixing Chen; Haoran Li; Dongbin Zhao
>
> **摘要:** Transferring the depth-based end-to-end policy trained in simulation to physical robots can yield an efficient and robust grasping policy, yet sensor artifacts in real depth maps like voids and noise establish a significant sim2real gap that critically impedes policy transfer. Training-time strategies like procedural noise injection or learned mappings suffer from data inefficiency due to unrealistic noise simulation, which is often ineffective for grasping tasks that require fine manipulation or dependency on paired datasets heavily. Furthermore, leveraging foundation models to reduce the sim2real gap via intermediate representations fails to mitigate the domain shift fully and adds computational overhead during deployment. This work confronts dual challenges of data inefficiency and deployment complexity. We propose DiffuDepGrasp, a deploy-efficient sim2real framework enabling zero-shot transfer through simulation-exclusive policy training. Its core innovation, the Diffusion Depth Generator, synthesizes geometrically pristine simulation depth with learned sensor-realistic noise via two synergistic modules. The first Diffusion Depth Module leverages temporal geometric priors to enable sample-efficient training of a conditional diffusion model that captures complex sensor noise distributions, while the second Noise Grafting Module preserves metric accuracy during perceptual artifact injection. With only raw depth inputs during deployment, DiffuDepGrasp eliminates computational overhead and achieves a 95.7% average success rate on 12-object grasping with zero-shot transfer and strong generalization to unseen objects.Project website: https://diffudepgrasp.github.io/.
>
---
#### [new 002] Tactile Data Recording System for Clothing with Motion-Controlled Robotic Sliding
- **分类: cs.RO; cs.CV; cs.HC; cs.LG; cs.MM**

- **简介: 论文提出一种基于机械臂的衣物触觉数据采集系统，通过模拟手指滑动精确控制运动参数，构建带运动标签的多模态触觉数据库，提升机器学习对衣物触感的识别准确率，解决衣物舒适性量化难题。**

- **链接: [https://arxiv.org/pdf/2511.11634v1](https://arxiv.org/pdf/2511.11634v1)**

> **作者:** Michikuni Eguchi; Takekazu Kitagishi; Yuichi Hiroi; Takefumi Hiraki
>
> **备注:** 3 pages, 2 figures, 1 table. Presented at SIGGRAPH Asia 2025 Posters (SA Posters '25), December 15-18, 2025, Hong Kong, Hong Kong
>
> **摘要:** The tactile sensation of clothing is critical to wearer comfort. To reveal physical properties that make clothing comfortable, systematic collection of tactile data during sliding motion is required. We propose a robotic arm-based system for collecting tactile data from intact garments. The system performs stroking measurements with a simulated fingertip while precisely controlling speed and direction, enabling creation of motion-labeled, multimodal tactile databases. Machine learning evaluation showed that including motion-related parameters improved identification accuracy for audio and acceleration data, demonstrating the efficacy of motion-related labels for characterizing clothing tactile sensation. This system provides a scalable, non-destructive method for capturing tactile data of clothing, contributing to future studies on fabric perception and reproduction.
>
---
#### [new 003] Multilaminate piezoelectric PVDF actuators to enhance performance of soft micro robots
- **分类: cs.RO**

- **简介: 论文研究多层压电PVDF致动器设计，旨在提升软微机器人的性能。解决传统致动器力大但脆或柔性但带宽低的问题。通过优化层数和厚度，实现高变形、高力和高频响应，集成于微机器人中实现鲁棒运动。**

- **链接: [https://arxiv.org/pdf/2511.12380v1](https://arxiv.org/pdf/2511.12380v1)**

> **作者:** Nicholas Gunter; Heiko Kabutz; Kaushik Jayaram
>
> **摘要:** Multilayer piezoelectric polyvinylidene fluoride (PVDF) actuators are a promising approach to enhance performance of soft microrobotic systems. In this work, we develop and characterize multilayer PVDF actuators with parallel voltage distribution across each layer, bridging a unique design space between brittle high-force PZT stacks and compliant but lower-bandwidth soft polymer actuators. We show the effects of layer thickness and number of layers in actuator performance and their agreement with a first principles model. By varying these parameters, we demonstrate actuators capable of >3 mm of free deflection, >20 mN of blocked force, and >=500 Hz, while operating at voltages as low as 150 volts. To illustrate their potential for robotic integration, we integrate our actuators into a planar, translating microrobot that leverages resonance to achieve locomotion with robustness to large perturbations.
>
---
#### [new 004] Task-Aware Morphology Optimization of Planar Manipulators via Reinforcement Learning
- **分类: cs.RO; eess.SY**

- **简介: 论文研究如何用强化学习优化平面机械臂的形态结构，以提升其操作灵活性。针对无解析解的路径跟踪任务，RL能有效收敛到最优形态，优于传统网格搜索和黑箱优化方法，展现了良好的可扩展性。**

- **链接: [https://arxiv.org/pdf/2511.12650v1](https://arxiv.org/pdf/2511.12650v1)**

> **作者:** Arvind Kumar Mishra; Sohom Chakrabarty
>
> **备注:** 10 pages, 11 figures, It is submitted as a journal option paper associated with the IFAC World Congress 2026
>
> **摘要:** In this work, Yoshikawa's manipulability index is used to investigate reinforcement learning (RL) as a framework for morphology optimization in planar robotic manipulators. A 2R manipulator tracking a circular end-effector path is first examined because this case has a known analytical optimum: equal link lengths and the second joint orthogonal to the first. This serves as a validation step to test whether RL can rediscover the optimum using reward feedback alone, without access to the manipulability expression or the Jacobian. Three RL algorithms (SAC, DDPG, and PPO) are compared with grid search and black-box optimizers, with morphology represented by a single action parameter phi that maps to the link lengths. All methods converge to the analytical solution, showing that numerical recovery of the optimum is possible without supplying analytical structure. Most morphology design tasks have no closed-form solutions, and grid or heuristic search becomes expensive as dimensionality increases. RL is therefore explored as a scalable alternative. The formulation used for the circular path is extended to elliptical and rectangular paths by expanding the action space to the full morphology vector (L1, L2, theta2). In these non-analytical settings, RL continues to converge reliably, whereas grid and black-box methods require far larger evaluation budgets. These results indicate that RL is effective for both recovering known optima and solving morphology optimization problems without analytical solutions.
>
---
#### [new 005] CUTE-Planner: Confidence-aware Uneven Terrain Exploration Planner
- **分类: cs.RO**

- **简介: 论文提出CUTE-Planner框架，用于行星探测机器人在复杂地形中的安全探索。解决现有方法无法处理高海拔不确定性及缺乏不确定性减少策略的问题。通过融合卡尔曼滤波估计与图搜索规划，优先探索可通行但低置信度区域，显著提升地图可靠性和任务成功率。**

- **链接: [https://arxiv.org/pdf/2511.12984v1](https://arxiv.org/pdf/2511.12984v1)**

> **作者:** Miryeong Park; Dongjin Cho; Sanghyun Kim; Younggun Cho
>
> **备注:** Accepted in International Conference on Space Robotics 2025
>
> **摘要:** Planetary exploration robots must navigate uneven terrain while building reliable maps for space missions. However, most existing methods incorporate traversability constraints but may not handle high uncertainty in elevation estimates near complex features like craters, do not consider exploration strategies for uncertainty reduction, and typically fail to address how elevation uncertainty affects navigation safety and map quality. To address the problems, we propose a framework integrating safe path generation, adaptive confidence updates, and confidence-aware exploration strategies. Using Kalman-based elevation estimation, our approach generates terrain traversability and confidence scores, then incorporates them into Graph-Based exploration Planner (GBP) to prioritize exploration of traversable low-confidence regions. We evaluate our framework through simulated lunar experiments using a novel low-confidence region ratio metric, achieving 69% uncertainty reduction compared to baseline GBP. In terms of mission success rate, our method achieves 100% while baseline GBP achieves 0%, demonstrating improvements in exploration safety and map reliability.
>
---
#### [new 006] Bootstrapped LLM Semantics for Context-Aware Path Planning
- **分类: cs.RO**

- **简介: 论文提出一种基于Bootstrap的LLM语义框架，将自然语言提示转化为风险感知的路径规划策略，解决机器人在人类空间中安全高效执行任务的问题。通过多轮LLM危险判断和贝叶斯Bootstrap估计风险后验，生成动态代价函数优化路径。**

- **链接: [https://arxiv.org/pdf/2511.11967v1](https://arxiv.org/pdf/2511.11967v1)**

> **作者:** Mani Amani; Behrad Beheshti; Reza Akhavian
>
> **摘要:** Prompting robots with natural language (NL) has largely been studied as what task to execute (goal selection, skill sequencing) rather than how to execute that task safely and efficiently in semantically rich, human-centric spaces. We address this gap with a framework that turns a large language model (LLM) into a stochastic semantic sensor whose outputs modulate a classical planner. Given a prompt and a semantic map, we draw multiple LLM "danger" judgments and apply a Bayesian bootstrap to approximate a posterior over per-class risk. Using statistics from the posterior, we create a potential cost to formulate a path planning problem. Across simulated environments and a BIM-backed digital twin, our method adapts how the robot moves in response to explicit prompts and implicit contextual information. We present qualitative and quantitative results.
>
---
#### [new 007] OpenRoboCare: A Multimodal Multi-Task Expert Demonstration Dataset for Robot Caregiving
- **分类: cs.RO**

- **简介: 论文提出OpenRoboCare数据集，用于机器人照护任务，解决缺乏专家驱动、多模态真实照护数据的问题。收集21名治疗师执行15项日常活动，涵盖RGB-D视频、姿态、眼动、触觉等五种模态，助力机器人感知与决策研究。**

- **链接: [https://arxiv.org/pdf/2511.13707v1](https://arxiv.org/pdf/2511.13707v1)**

> **作者:** Xiaoyu Liang; Ziang Liu; Kelvin Lin; Edward Gu; Ruolin Ye; Tam Nguyen; Cynthia Hsu; Zhanxin Wu; Xiaoman Yang; Christy Sum Yu Cheung; Harold Soh; Katherine Dimitropoulou; Tapomayukh Bhattacharjee
>
> **备注:** IROS 2025
>
> **摘要:** We present OpenRoboCare, a multimodal dataset for robot caregiving, capturing expert occupational therapist demonstrations of Activities of Daily Living (ADLs). Caregiving tasks involve complex physical human-robot interactions, requiring precise perception under occlusions, safe physical contact, and long-horizon planning. While recent advances in robot learning from demonstrations have shown promise, there is a lack of a large-scale, diverse, and expert-driven dataset that captures real-world caregiving routines. To address this gap, we collect data from 21 occupational therapists performing 15 ADL tasks on two manikins. The dataset spans five modalities: RGB-D video, pose tracking, eye-gaze tracking, task and action annotations, and tactile sensing, providing rich multimodal insights into caregiver movement, attention, force application, and task execution strategies. We further analyze expert caregiving principles and strategies, offering insights to improve robot efficiency and task feasibility. Additionally, our evaluations demonstrate that OpenRoboCare presents challenges for state-of-the-art robot perception and human activity recognition methods, both critical for developing safe and adaptive assistive robots, highlighting the value of our contribution. See our website for additional visualizations: https://emprise.cs.cornell.edu/robo-care/.
>
---
#### [new 008] ResAlignNet: A Data-Driven Approach for INS/DVL Alignment
- **分类: cs.RO**

- **简介: 论文提出ResAlignNet，一种基于1D ResNet-18的数据驱动方法，用于解决水下机器人惯导与多普勒测速仪的快速对准问题。该方法无需外部定位或复杂运动，可在25秒内实现0.8°精度对准，显著缩短收敛时间并提升部署灵活性。**

- **链接: [https://arxiv.org/pdf/2511.13096v1](https://arxiv.org/pdf/2511.13096v1)**

> **作者:** Guy Damari; Itzik Klein
>
> **摘要:** Autonomous underwater vehicles rely on precise navigation systems that combine the inertial navigation system and the Doppler velocity log for successful missions in challenging environments where satellite navigation is unavailable. The effectiveness of this integration critically depends on accurate alignment between the sensor reference frames. Standard model-based alignment methods between these sensor systems suffer from lengthy convergence times, dependence on prescribed motion patterns, and reliance on external aiding sensors, significantly limiting operational flexibility. To address these limitations, this paper presents ResAlignNet, a data-driven approach using the 1D ResNet-18 architecture that transforms the alignment problem into deep neural network optimization, operating as an in-situ solution that requires only sensors on board without external positioning aids or complex vehicle maneuvers, while achieving rapid convergence in seconds. Additionally, the approach demonstrates the learning capabilities of Sim2Real transfer, enabling training in synthetic data while deploying in operational sensor measurements. Experimental validation using the Snapir autonomous underwater vehicle demonstrates that ResAlignNet achieves alignment accuracy within 0.8° using only 25 seconds of data collection, representing a 65\% reduction in convergence time compared to standard velocity-based methods. The trajectory-independent solution eliminates motion pattern requirements and enables immediate vehicle deployment without lengthy pre-mission procedures, advancing underwater navigation capabilities through robust sensor-agnostic alignment that scales across different operational scenarios and sensor specifications.
>
---
#### [new 009] EcoFlight: Finding Low-Energy Paths Through Obstacles for Autonomous Sensing Drones
- **分类: cs.RO**

- **简介: 论文提出EcoFlight算法，解决无人机在有障碍物环境中高效避障路径规划问题。通过建模能量消耗并优化飞行路径，在高密度障碍场景下显著降低能耗，优于直接飞行和最短距离方案。**

- **链接: [https://arxiv.org/pdf/2511.12618v1](https://arxiv.org/pdf/2511.12618v1)**

> **作者:** Jordan Leyva; Nahim J. Moran Vera; Yihan Xu; Adrien Durasno; Christopher U. Romero; Tendai Chimuka; Gabriel O. Huezo Ramirez; Ziqian Dong; Roberto Rojas-Cessa
>
> **备注:** Autonomous drone, A* algorithm, 3D environments, path planning, obstacle avoidance, energy efficiency, MIT Conference
>
> **摘要:** Obstacle avoidance path planning for uncrewed aerial vehicles (UAVs), or drones, is rarely addressed in most flight path planning schemes, despite obstacles being a realistic condition. Obstacle avoidance can also be energy-intensive, making it a critical factor in efficient point-to-point drone flights. To address these gaps, we propose EcoFlight, an energy-efficient pathfinding algorithm that determines the lowest-energy route in 3D space with obstacles. The algorithm models energy consumption based on the drone propulsion system and flight dynamics. We conduct extensive evaluations, comparing EcoFlight with direct-flight and shortest-distance schemes. The simulation results across various obstacle densities show that EcoFlight consistently finds paths with lower energy consumption than comparable algorithms, particularly in high-density environments. We also demonstrate that a suitable flying speed can further enhance energy savings.
>
---
#### [new 010] PIGEON: VLM-Driven Object Navigation via Points of Interest Selection
- **分类: cs.RO; cs.CV**

- **简介: 论文提出PIGEON方法，用于未知环境中物体导航任务。针对现有方法决策频率与智能平衡差的问题，利用视觉语言模型选择兴趣点（PoI），结合低层规划器提升决策频率，并生成可验证奖励的强化学习数据，实现零样本迁移和实时深度推理导航。**

- **链接: [https://arxiv.org/pdf/2511.13207v1](https://arxiv.org/pdf/2511.13207v1)**

> **作者:** Cheng Peng; Zhenzhe Zhang; Cheng Chi; Xiaobao Wei; Yanhao Zhang; Heng Wang; Pengwei Wang; Zhongyuan Wang; Jing Liu; Shanghang Zhang
>
> **摘要:** Navigating to a specified object in an unknown environment is a fundamental yet challenging capability of embodied intelligence. However, current methods struggle to balance decision frequency with intelligence, resulting in decisions lacking foresight or discontinuous actions. In this work, we propose PIGEON: Point of Interest Guided Exploration for Object Navigation with VLM, maintaining a lightweight and semantically aligned snapshot memory during exploration as semantic input for the exploration strategy. We use a large Visual-Language Model (VLM), named PIGEON-VL, to select Points of Interest (PoI) formed during exploration and then employ a lower-level planner for action output, increasing the decision frequency. Additionally, this PoI-based decision-making enables the generation of Reinforcement Learning with Verifiable Reward (RLVR) data suitable for simulators. Experiments on classic object navigation benchmarks demonstrate that our zero-shot transfer method achieves state-of-the-art performance, while RLVR further enhances the model's semantic guidance capabilities, enabling deep reasoning during real-time navigation.
>
---
#### [new 011] Locally Optimal Solutions to Constraint Displacement Problems via Path-Obstacle Overlaps
- **分类: cs.RO; cs.AI**

- **简介: 论文研究机器人路径规划中的约束位移问题，提出两阶段方法：先优化轨迹，再位移障碍物使路径可行，从而在复杂环境中找到局部最优解。**

- **链接: [https://arxiv.org/pdf/2511.12203v1](https://arxiv.org/pdf/2511.12203v1)**

> **作者:** Antony Thomas; Fulvio Mastrogiovanni; Marco Baglietto
>
> **备注:** Robotics and Autonomous Systems
>
> **摘要:** We present a unified approach for constraint displacement problems in which a robot finds a feasible path by displacing constraints or obstacles. To this end, we propose a two stage process that returns locally optimal obstacle displacements to enable a feasible path for the robot. The first stage proceeds by computing a trajectory through the obstacles while minimizing an appropriate objective function. In the second stage, these obstacles are displaced to make the computed robot trajectory feasible, that is, collision-free. Several examples are provided that successfully demonstrate our approach on two distinct classes of constraint displacement problems.
>
---
#### [new 012] Collision-Free Navigation of Mobile Robots via Quadtree-Based Model Predictive Control
- **分类: cs.RO; eess.SY**

- **简介: 论文提出基于四叉树的模型预测控制方法，用于移动机器人无碰撞导航。通过构建安全区域并将其作为MPC约束，实现环境感知与路径规划一体化，解决复杂环境中高效避障问题。**

- **链接: [https://arxiv.org/pdf/2511.13188v1](https://arxiv.org/pdf/2511.13188v1)**

> **作者:** Osama Al Sheikh Ali; Sotiris Koutsoftas; Ze Zhang; Knut Akesson; Emmanuel Dean
>
> **备注:** This paper has been accepted by IEEE SII 2026
>
> **摘要:** This paper presents an integrated navigation framework for Autonomous Mobile Robots (AMRs) that unifies environment representation, trajectory generation, and Model Predictive Control (MPC). The proposed approach incorporates a quadtree-based method to generate structured, axis-aligned collision-free regions from occupancy maps. These regions serve as both a basis for developing safe corridors and as linear constraints within the MPC formulation, enabling efficient and reliable navigation without requiring direct obstacle encoding. The complete pipeline combines safe-area extraction, connectivity graph construction, trajectory generation, and B-spline smoothing into one coherent system. Experimental results demonstrate consistent success and superior performance compared to baseline approaches across complex environments.
>
---
#### [new 013] Count Every Rotation and Every Rotation Counts: Exploring Drone Dynamics via Propeller Sensing
- **分类: cs.RO**

- **简介: 论文提出基于事件相机的无人机感知系统\sysname，通过分析螺旋桨转速实现高精度、低延迟的无人机动态感知。解决地面非接触式无人机监测难题，提升飞行指令识别与跟踪精度。**

- **链接: [https://arxiv.org/pdf/2511.13100v1](https://arxiv.org/pdf/2511.13100v1)**

> **作者:** Xuecheng Chen; Jingao Xu; Wenhua Ding; Haoyang Wang; Xinyu Luo; Ruiyang Duan; Jialong Chen; Xueqian Wang; Yunhao Liu; Xinlei Chen
>
> **摘要:** As drone-based applications proliferate, paramount contactless sensing of airborne drones from the ground becomes indispensable. This work demonstrates concentrating on propeller rotational speed will substantially improve drone sensing performance and proposes an event-camera-based solution, \sysname. \sysname features two components: \textit{Count Every Rotation} achieves accurate, real-time propeller speed estimation by mitigating ultra-high sensitivity of event cameras to environmental noise. \textit{Every Rotation Counts} leverages these speeds to infer both internal and external drone dynamics. Extensive evaluations in real-world drone delivery scenarios show that \sysname achieves a sensing latency of 3$ms$ and a rotational speed estimation error of merely 0.23\%. Additionally, \sysname infers drone flight commands with 96.5\% precision and improves drone tracking accuracy by over 22\% when combined with other sensing modalities. \textit{ Demo: {\color{blue}https://eventpro25.github.io/EventPro/.} }
>
---
#### [new 014] Learning Adaptive Neural Teleoperation for Humanoid Robots: From Inverse Kinematics to End-to-End Control
- **分类: cs.RO**

- **简介: 论文提出一种基于学习的神经遥操作框架，用于改善人形机器人在复杂操作任务中的控制自然性和鲁棒性。针对传统逆运动学加PD控制器难以适应外力和用户差异的问题，该方法通过强化学习直接从VR输入映射到关节指令，实现实时、平滑且自适应的控制，在仿真与真实机器人上验证了优越性能。**

- **链接: [https://arxiv.org/pdf/2511.12390v1](https://arxiv.org/pdf/2511.12390v1)**

> **作者:** Sanjar Atamuradov
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** Virtual reality (VR) teleoperation has emerged as a promising approach for controlling humanoid robots in complex manipulation tasks. However, traditional teleoperation systems rely on inverse kinematics (IK) solvers and hand-tuned PD controllers, which struggle to handle external forces, adapt to different users, and produce natural motions under dynamic conditions. In this work, we propose a learning-based neural teleoperation framework that replaces the conventional IK+PD pipeline with learned policies trained via reinforcement learning. Our approach learns to directly map VR controller inputs to robot joint commands while implicitly handling force disturbances, producing smooth trajectories, and adapting to user preferences. We train our policies in simulation using demonstrations collected from IK-based teleoperation as initialization, then fine-tune them with force randomization and trajectory smoothness rewards. Experiments on the Unitree G1 humanoid robot demonstrate that our learned policies achieve 34% lower tracking error, 45% smoother motions, and superior force adaptation compared to the IK baseline, while maintaining real-time performance (50Hz control frequency). We validate our approach on manipulation tasks including object pick-and-place, door opening, and bimanual coordination. These results suggest that learning-based approaches can significantly improve the naturalness and robustness of humanoid teleoperation systems.
>
---
#### [new 015] Contact-Safe Reinforcement Learning with ProMP Reparameterization and Energy Awareness
- **分类: cs.RO**

- **简介: 论文提出一种基于任务空间的强化学习框架，解决机器人接触安全与能量效率问题。通过ProMP重参数化和能量感知阻抗控制，在复杂3D环境中实现安全、平滑的任务执行。**

- **链接: [https://arxiv.org/pdf/2511.13459v1](https://arxiv.org/pdf/2511.13459v1)**

> **作者:** Bingkun Huang; Yuhe Gong; Zewen Yang; Tianyu Ren; Luis Figueredo
>
> **摘要:** Reinforcement learning (RL) approaches based on Markov Decision Processes (MDPs) are predominantly applied in the robot joint space, often relying on limited task-specific information and partial awareness of the 3D environment. In contrast, episodic RL has demonstrated advantages over traditional MDP-based methods in terms of trajectory consistency, task awareness, and overall performance in complex robotic tasks. Moreover, traditional step-wise and episodic RL methods often neglect the contact-rich information inherent in task-space manipulation, especially considering the contact-safety and robustness. In this work, contact-rich manipulation tasks are tackled using a task-space, energy-safe framework, where reliable and safe task-space trajectories are generated through the combination of Proximal Policy Optimization (PPO) and movement primitives. Furthermore, an energy-aware Cartesian Impedance Controller objective is incorporated within the proposed framework to ensure safe interactions between the robot and the environment. Our experimental results demonstrate that the proposed framework outperforms existing methods in handling tasks on various types of surfaces in 3D environments, achieving high success rates as well as smooth trajectories and energy-safe interactions.
>
---
#### [new 016] Botany Meets Robotics in Alpine Scree Monitoring
- **分类: cs.RO**

- **简介: 该论文属于环境监测任务，旨在解决高山碎石 habitat 监测效率低、成本高的问题。作者使用腿式机器人ANYmal C结合深度学习技术，在阿尔卑斯山区进行两年实地试验，实现自动识别关键植物物种，提升监测频率与数据质量。**

- **链接: [https://arxiv.org/pdf/2511.12526v1](https://arxiv.org/pdf/2511.12526v1)**

> **作者:** Davide De Benedittis; Giovanni Di Lorenzo; Franco Angelini; Barbara Valle; Marina Serena Borgatti; Paolo Remagnino; Marco Caccianiga; Manolo Garabini
>
> **备注:** Published as Early Access in IEEE Transactions on Field Robotics. 19 pages, 13 figures
>
> **摘要:** According to the European Union's Habitat Directive, habitat monitoring plays a critical role in response to the escalating problems posed by biodiversity loss and environmental degradation. Scree habitats, hosting unique and often endangered species, face severe threats from climate change due to their high-altitude nature. Traditionally, their monitoring has required highly skilled scientists to conduct extensive fieldwork in remote, potentially hazardous locations, making the process resource-intensive and time-consuming. This paper presents a novel approach for scree habitat monitoring using a legged robot to assist botanists in data collection and species identification. Specifically, we deployed the ANYmal C robot in the Italian Alpine bio-region in two field campaigns spanning two years and leveraged deep learning to detect and classify key plant species of interest. Our results demonstrate that agile legged robots can navigate challenging terrains and increase the frequency and efficiency of scree monitoring. When paired with traditional phytosociological surveys performed by botanists, this robotics-assisted protocol not only streamlines field operations but also enhances data acquisition, storage, and usage. The outcomes of this research contribute to the evolving landscape of robotics in environmental science, paving the way for a more comprehensive and sustainable approach to habitat monitoring and preservation.
>
---
#### [new 017] ZeroDexGrasp: Zero-Shot Task-Oriented Dexterous Grasp Synthesis with Prompt-Based Multi-Stage Semantic Reasoning
- **分类: cs.RO**

- **简介: 该论文提出ZeroDexGrasp框架，解决零样本任务导向灵巧抓取问题。通过提示驱动的多阶段语义推理与接触引导优化，实现对未见物体和复杂任务指令的高质量抓取生成。**

- **链接: [https://arxiv.org/pdf/2511.13327v1](https://arxiv.org/pdf/2511.13327v1)**

> **作者:** Juntao Jian; Yi-Lin Wei; Chengjie Mou; Yuhao Lin; Xing Zhu; Yujun Shen; Wei-Shi Zheng; Ruizhen Hu
>
> **摘要:** Task-oriented dexterous grasping holds broad application prospects in robotic manipulation and human-object interaction. However, most existing methods still struggle to generalize across diverse objects and task instructions, as they heavily rely on costly labeled data to ensure task-specific semantic alignment. In this study, we propose \textbf{ZeroDexGrasp}, a zero-shot task-oriented dexterous grasp synthesis framework integrating Multimodal Large Language Models with grasp refinement to generate human-like grasp poses that are well aligned with specific task objectives and object affordances. Specifically, ZeroDexGrasp employs prompt-based multi-stage semantic reasoning to infer initial grasp configurations and object contact information from task and object semantics, then exploits contact-guided grasp optimization to refine these poses for physical feasibility and task alignment. Experimental results demonstrate that ZeroDexGrasp enables high-quality zero-shot dexterous grasping on diverse unseen object categories and complex task requirements, advancing toward more generalizable and intelligent robotic grasping.
>
---
#### [new 018] Characterization and Evaluation of Screw-Based Locomotion Across Aquatic, Granular, and Transitional Media
- **分类: cs.RO**

- **简介: 论文研究螺杆推进系统在水、颗粒和过渡介质中的运动性能，旨在优化其两栖移动能力。通过系统实验与原理分析，识别关键设计参数并提出适应性策略，提升螺杆推进系统的多环境适应性。**

- **链接: [https://arxiv.org/pdf/2511.11958v1](https://arxiv.org/pdf/2511.11958v1)**

> **作者:** Derek Chen; Zoe Samuels; Lizzie Peiros; Sujaan Mukherjee; Michael C. Yip
>
> **摘要:** Screw-based propulsion systems offer promising capabilities for amphibious mobility, yet face significant challenges in optimizing locomotion across water, granular materials, and transitional environments. This study presents a systematic investigation into the locomotion performance of various screw configurations in media such as dry sand, wet sand, saturated sand, and water. Through a principles-first approach to analyze screw performance, it was found that certain parameters are dominant in their impact on performance. Depending on the media, derived parameters inspired from optimizing heat sink design help categorize performance within the dominant design parameters. Our results provide specific insights into screw shell design and adaptive locomotion strategies to enhance the performance of screw-based propulsion systems for versatile amphibious applications.
>
---
#### [new 019] SocialNav-Map: Dynamic Mapping with Human Trajectory Prediction for Zero-Shot Social Navigation
- **分类: cs.RO**

- **简介: 该论文提出SocialNav-Map框架，解决自主机器人在人群密集环境中零样本社交导航问题。通过融合人类轨迹预测与动态占用地图，实现无需环境训练的安全高效导航，显著降低碰撞率并优于现有强化学习方法。**

- **链接: [https://arxiv.org/pdf/2511.12232v1](https://arxiv.org/pdf/2511.12232v1)**

> **作者:** Lingfeng Zhang; Erjia Xiao; Xiaoshuai Hao; Haoxiang Fu; Zeying Gong; Long Chen; Xiaojun Liang; Renjing Xu; Hangjun Ye; Wenbo Ding
>
> **摘要:** Social navigation in densely populated dynamic environments poses a significant challenge for autonomous mobile robots, requiring advanced strategies for safe interaction. Existing reinforcement learning (RL)-based methods require over 2000+ hours of extensive training and often struggle to generalize to unfamiliar environments without additional fine-tuning, limiting their practical application in real-world scenarios. To address these limitations, we propose SocialNav-Map, a novel zero-shot social navigation framework that combines dynamic human trajectory prediction with occupancy mapping, enabling safe and efficient navigation without the need for environment-specific training. Specifically, SocialNav-Map first transforms the task goal position into the constructed map coordinate system. Subsequently, it creates a dynamic occupancy map that incorporates predicted human movements as dynamic obstacles. The framework employs two complementary methods for human trajectory prediction: history prediction and orientation prediction. By integrating these predicted trajectories into the occupancy map, the robot can proactively avoid potential collisions with humans while efficiently navigating to its destination. Extensive experiments on the Social-HM3D and Social-MP3D datasets demonstrate that SocialNav-Map significantly outperforms state-of-the-art (SOTA) RL-based methods, which require 2,396 GPU hours of training. Notably, it reduces human collision rates by over 10% without necessitating any training in novel environments. By eliminating the need for environment-specific training, SocialNav-Map achieves superior navigation performance, paving the way for the deployment of social navigation systems in real-world environments characterized by diverse human behaviors. The code is available at: https://github.com/linglingxiansen/SocialNav-Map.
>
---
#### [new 020] GUIDE: Gaussian Unified Instance Detection for Enhanced Obstacle Perception in Autonomous Driving
- **分类: cs.RO**

- **简介: 论文提出GUIDE框架，用于自动驾驶中的障碍物感知任务，解决传统3D边界框难以表征不规则物体的问题。通过3D高斯表示实现实例级占用预测与跟踪，提升精度与效率，在nuScenes上mAP达21.61，性能提升50%。**

- **链接: [https://arxiv.org/pdf/2511.12941v1](https://arxiv.org/pdf/2511.12941v1)**

> **作者:** Chunyong Hu; Qi Luo; Jianyun Xu; Song Wang; Qiang Li; Sheng Yang
>
> **摘要:** In the realm of autonomous driving, accurately detecting surrounding obstacles is crucial for effective decision-making. Traditional methods primarily rely on 3D bounding boxes to represent these obstacles, which often fail to capture the complexity of irregularly shaped, real-world objects. To overcome these limitations, we present GUIDE, a novel framework that utilizes 3D Gaussians for instance detection and occupancy prediction. Unlike conventional occupancy prediction methods, GUIDE also offers robust tracking capabilities. Our framework employs a sparse representation strategy, using Gaussian-to-Voxel Splatting to provide fine-grained, instance-level occupancy data without the computational demands associated with dense voxel grids. Experimental validation on the nuScenes dataset demonstrates GUIDE's performance, with an instance occupancy mAP of 21.61, marking a 50\% improvement over existing methods, alongside competitive tracking capabilities. GUIDE establishes a new benchmark in autonomous perception systems, effectively combining precision with computational efficiency to better address the complexities of real-world driving environments.
>
---
#### [new 021] Evaluating Model-Agnostic Meta-Learning on MetaWorld ML10 Benchmark: Fast Adaptation in Robotic Manipulation Tasks
- **分类: cs.RO**

- **简介: 该论文研究机器人操纵任务中的快速适应问题，提出用MAML-TRPO方法实现少样本快速学习。实验表明其能有效适应10种不同操作技能，但存在泛化差距和适应效果差异大的问题。**

- **链接: [https://arxiv.org/pdf/2511.12383v1](https://arxiv.org/pdf/2511.12383v1)**

> **作者:** Sanjar Atamuradov
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** Meta-learning algorithms enable rapid adaptation to new tasks with minimal data, a critical capability for real-world robotic systems. This paper evaluates Model-Agnostic Meta-Learning (MAML) combined with Trust Region Policy Optimization (TRPO) on the MetaWorld ML10 benchmark, a challenging suite of ten diverse robotic manipulation tasks. We implement and analyze MAML-TRPO's ability to learn a universal initialization that facilitates few-shot adaptation across semantically different manipulation behaviors including pushing, picking, and drawer manipulation. Our experiments demonstrate that MAML achieves effective one-shot adaptation with clear performance improvements after a single gradient update, reaching final success rates of 21.0% on training tasks and 13.2% on held-out test tasks. However, we observe a generalization gap that emerges during meta-training, where performance on test tasks plateaus while training task performance continues to improve. Task-level analysis reveals high variance in adaptation effectiveness, with success rates ranging from 0% to 80% across different manipulation skills. These findings highlight both the promise and current limitations of gradient-based meta-learning for diverse robotic manipulation, and suggest directions for future work in task-aware adaptation and structured policy architectures.
>
---
#### [new 022] Innovative Design of Multi-functional Supernumerary Robotic Limbs with Ellipsoid Workspace Optimization
- **分类: cs.RO**

- **简介: 论文提出多目标优化设计理论，解决SRL设备在上肢抓握与下肢行走功能间的平衡问题。通过椭球体量化工作空间、改进萤火虫算法优化结构参数，提升性能并降低肌肉负荷。**

- **链接: [https://arxiv.org/pdf/2511.12186v1](https://arxiv.org/pdf/2511.12186v1)**

> **作者:** Jun Huo; Jian Huang; Jie Zuo; Bo Yang; Zhongzheng Fu; Xi Li; Samer Mohammed
>
> **摘要:** Supernumerary robotic limbs (SRLs) offer substantial potential in both the rehabilitation of hemiplegic patients and the enhancement of functional capabilities for healthy individuals. Designing a general-purpose SRL device is inherently challenging, particularly when developing a unified theoretical framework that meets the diverse functional requirements of both upper and lower limbs. In this paper, we propose a multi-objective optimization (MOO) design theory that integrates grasping workspace similarity, walking workspace similarity, braced force for sit-to-stand (STS) movements, and overall mass and inertia. A geometric vector quantification method is developed using an ellipsoid to represent the workspace, aiming to reduce computational complexity and address quantification challenges. The ellipsoid envelope transforms workspace points into ellipsoid attributes, providing a parametric description of the workspace. Furthermore, the STS static braced force assesses the effectiveness of force transmission. The overall mass and inertia restricts excessive link length. To facilitate rapid and stable convergence of the model to high-dimensional irregular Pareto fronts, we introduce a multi-subpopulation correction firefly algorithm. This algorithm incorporates a strategy involving attractive and repulsive domains to effectively handle the MOO task. The optimized solution is utilized to redesign the prototype for experimentation to meet specified requirements. Six healthy participants and two hemiplegia patients participated in real experiments. Compared to the pre-optimization results, the average grasp success rate improved by 7.2%, while the muscle activity during walking and STS tasks decreased by an average of 12.7% and 25.1%, respectively. The proposed design theory offers an efficient option for the design of multi-functional SRL mechanisms.
>
---
#### [new 023] Towards High-Consistency Embodied World Model with Multi-View Trajectory Videos
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人视觉与世界模型任务，旨在解决现有模型在低级动作控制下难以实现高一致性物理交互的问题。作者提出MTV-World框架，利用多视角轨迹视频作为控制信号，提升视觉运动预测精度，并设计自动评估方法衡量空间一致性与交互准确性。**

- **链接: [https://arxiv.org/pdf/2511.12882v1](https://arxiv.org/pdf/2511.12882v1)**

> **作者:** Taiyi Su; Jian Zhu; Yaxuan Li; Chong Ma; Zitai Huang; Yichen Zhu; Hanli Wang; Yi Xu
>
> **备注:** 11 pages, 5 figures
>
> **摘要:** Embodied world models aim to predict and interact with the physical world through visual observations and actions. However, existing models struggle to accurately translate low-level actions (e.g., joint positions) into precise robotic movements in predicted frames, leading to inconsistencies with real-world physical interactions. To address these limitations, we propose MTV-World, an embodied world model that introduces Multi-view Trajectory-Video control for precise visuomotor prediction. Specifically, instead of directly using low-level actions for control, we employ trajectory videos obtained through camera intrinsic and extrinsic parameters and Cartesian-space transformation as control signals. However, projecting 3D raw actions onto 2D images inevitably causes a loss of spatial information, making a single view insufficient for accurate interaction modeling. To overcome this, we introduce a multi-view framework that compensates for spatial information loss and ensures high-consistency with physical world. MTV-World forecasts future frames based on multi-view trajectory videos as input and conditioning on an initial frame per view. Furthermore, to systematically evaluate both robotic motion precision and object interaction accuracy, we develop an auto-evaluation pipeline leveraging multimodal large models and referring video object segmentation models. To measure spatial consistency, we formulate it as an object location matching problem and adopt the Jaccard Index as the evaluation metric. Extensive experiments demonstrate that MTV-World achieves precise control execution and accurate physical interaction modeling in complex dual-arm scenarios.
>
---
#### [new 024] Autonomous Underwater Cognitive System for Adaptive Navigation: A SLAM-Integrated Cognitive Architecture
- **分类: cs.RO; cs.AI; cs.AR**

- **简介: 论文提出AUCS系统，融合SLAM与Soar认知架构，解决深海导航中迷失、通信中断等问题。通过多传感器融合与认知模块实现感知、推理与自适应学习，提升水下机器人在复杂环境中的自主性与可靠性。**

- **链接: [https://arxiv.org/pdf/2511.11845v1](https://arxiv.org/pdf/2511.11845v1)**

> **作者:** K. A. I. N Jayarathne; R. M. N. M. Rathnayaka; D. P. S. S. Peiris
>
> **备注:** 6 pages, 2 figures
>
> **摘要:** Deep-sea exploration poses significant challenges, including disorientation, communication loss, and navigational failures in dynamic underwater environments. This paper presents an Autonomous Underwater Cognitive System (AUCS) that integrates Simultaneous Localization and Mapping (SLAM) with a Soar-based cognitive architecture to enable adaptive navigation in complex oceanic conditions. The system fuses multi-sensor data from SONAR, LiDAR, IMU, and DVL with cognitive reasoning modules for perception, attention, planning, and learning. Unlike conventional SLAM systems, AUCS incorporates semantic understanding, adaptive sensor management, and memory-based learning to differentiate between dynamic and static objects, reducing false loop closures and enhancing long-term map consistency. The proposed architecture demonstrates a complete perception-cognition-action-learning loop, allowing autonomous underwater vehicles to sense, reason, and adapt intelligently. This work lays a foundation for next-generation cognitive submersible systems, improving safety, reliability, and autonomy in deep-sea exploration.
>
---
#### [new 025] SplatSearch: Instance Image Goal Navigation for Mobile Robots using 3D Gaussian Splatting and Diffusion Models
- **分类: cs.RO**

- **简介: 论文提出SplatSearch解决移动机器人在未知环境中基于单张目标图像寻找特定物体的实例图像目标导航问题。通过稀疏视图3D高斯点绘和扩散模型补全图像，结合语义与视觉上下文优化探索策略，提升导航成功率和路径效率。**

- **链接: [https://arxiv.org/pdf/2511.12972v1](https://arxiv.org/pdf/2511.12972v1)**

> **作者:** Siddarth Narasimhan; Matthew Lisondra; Haitong Wang; Goldie Nejat
>
> **备注:** Project Page: https://splat-search.github.io/
>
> **摘要:** The Instance Image Goal Navigation (IIN) problem requires mobile robots deployed in unknown environments to search for specific objects or people of interest using only a single reference goal image of the target. This problem can be especially challenging when: 1) the reference image is captured from an arbitrary viewpoint, and 2) the robot must operate with sparse-view scene reconstructions. In this paper, we address the IIN problem, by introducing SplatSearch, a novel architecture that leverages sparse-view 3D Gaussian Splatting (3DGS) reconstructions. SplatSearch renders multiple viewpoints around candidate objects using a sparse online 3DGS map, and uses a multi-view diffusion model to complete missing regions of the rendered images, enabling robust feature matching against the goal image. A novel frontier exploration policy is introduced which uses visual context from the synthesized viewpoints with semantic context from the goal image to evaluate frontier locations, allowing the robot to prioritize frontiers that are semantically and visually relevant to the goal image. Extensive experiments in photorealistic home and real-world environments validate the higher performance of SplatSearch against current state-of-the-art methods in terms of Success Rate and Success Path Length. An ablation study confirms the design choices of SplatSearch.
>
---
#### [new 026] TOPP-DWR: Time-Optimal Path Parameterization of Differential-Driven Wheeled Robots Considering Piecewise-Constant Angular Velocity Constraints
- **分类: cs.RO**

- **简介: 该论文针对差速轮式机器人的时间最优路径参数化问题，提出TOPP-DWR算法。解决现有方法忽略角速度约束导致控制性能下降的问题。通过非均匀B样条表示轨迹、统一约束为线性速度约束，并引入松弛变量将问题转化为SOCP优化，提升计算效率与实用性。**

- **链接: [https://arxiv.org/pdf/2511.12910v1](https://arxiv.org/pdf/2511.12910v1)**

> **作者:** Yong Li; Yujun Huang; Yi Chen; Hui Cheng
>
> **摘要:** Differential-driven wheeled robots (DWR) represent the quintessential type of mobile robots and find extensive appli- cations across the robotic field. Most high-performance control approaches for DWR explicitly utilize the linear and angular velocities of the trajectory as control references. However, existing research on time-optimal path parameterization (TOPP) for mobile robots usually neglects the angular velocity and joint vel- ocity constraints, which can result in degraded control perfor- mance in practical applications. In this article, a systematic and practical TOPP algorithm named TOPP-DWR is proposed for DWR and other mobile robots. First, the non-uniform B-spline is adopted to represent the initial trajectory in the task space. Second, the piecewise-constant angular velocity, as well as joint velocity, linear velocity, and linear acceleration constraints, are incorporated into the TOPP problem. During the construction of the optimization problem, the aforementioned constraints are uniformly represented as linear velocity constraints. To boost the numerical computational efficiency, we introduce a slack variable to reformulate the problem into second-order-cone programming (SOCP). Subsequently, comparative experiments are conducted to validate the superiority of the proposed method. Quantitative performance indexes show that TOPP-DWR achieves TOPP while adhering to all constraints. Finally, field autonomous navigation experiments are carried out to validate the practicability of TOPP-DWR in real-world applications.
>
---
#### [new 027] EL3DD: Extended Latent 3D Diffusion for Language Conditioned Multitask Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出EL3DD模型，用于语言条件下的多任务机器人操作。解决如何让机器人理解自然语言并执行复杂物理任务的问题。通过改进嵌入和扩散模型技术，在CALVIN数据集上实现更精准的轨迹生成和更高长程成功率。**

- **链接: [https://arxiv.org/pdf/2511.13312v1](https://arxiv.org/pdf/2511.13312v1)**

> **作者:** Jonas Bode; Raphael Memmesheimer; Sven Behnke
>
> **备注:** 10 pages; 2 figures; 1 table. Prprint submitted to the European Robotics Forum 2026
>
> **摘要:** Acting in human environments is a crucial capability for general-purpose robots, necessitating a robust understanding of natural language and its application to physical tasks. This paper seeks to harness the capabilities of diffusion models within a visuomotor policy framework that merges visual and textual inputs to generate precise robotic trajectories. By employing reference demonstrations during training, the model learns to execute manipulation tasks specified through textual commands within the robot's immediate environment. The proposed research aims to extend an existing model by leveraging improved embeddings, and adapting techniques from diffusion models for image generation. We evaluate our methods on the CALVIN dataset, proving enhanced performance on various manipulation tasks and an increased long-horizon success rate when multiple tasks are executed in sequence. Our approach reinforces the usefulness of diffusion models and contributes towards general multitask manipulation.
>
---
#### [new 028] DR. Nav: Semantic-Geometric Representations for Proactive Dead-End Recovery and Navigation
- **分类: cs.RO**

- **简介: 该论文提出DR. Nav方法，用于自主导航中的死区检测与恢复任务。针对未建图环境中死区难以识别的问题，通过RGB-LiDAR融合与贝叶斯更新生成带恢复风险的语义代价地图，提升导航安全性与效率。**

- **链接: [https://arxiv.org/pdf/2511.12778v1](https://arxiv.org/pdf/2511.12778v1)**

> **作者:** Vignesh Rajagopal; Kasun Weerakoon Kulathun Mudiyanselage; Gershom Devake Seneviratne; Pon Aswin Sankaralingam; Mohamed Elnoor; Jing Liang; Rohan Chandra; Dinesh Manocha
>
> **摘要:** We present DR. Nav (Dead-End Recovery-aware Navigation), a novel approach to autonomous navigation in scenarios where dead-end detection and recovery are critical, particularly in unstructured environments where robots must handle corners, vegetation occlusions, and blocked junctions. DR. Nav introduces a proactive strategy for navigation in unmapped environments without prior assumptions. Our method unifies dead-end prediction and recovery by generating a single, continuous, real-time semantic cost map. Specifically, DR. Nav leverages cross-modal RGB-LiDAR fusion with attention-based filtering to estimate per-cell dead-end likelihoods and recovery points, which are continuously updated through Bayesian inference to enhance robustness. Unlike prior mapping methods that only encode traversability, DR. Nav explicitly incorporates recovery-aware risk into the navigation cost map, enabling robots to anticipate unsafe regions and plan safer alternative trajectories. We evaluate DR. Nav across multiple dense indoor and outdoor scenarios and demonstrate an increase of 83.33% in accuracy in detection, a 52.4% reduction in time-to-goal (path efficiency), compared to state-of-the-art planners such as DWA, MPPI, and Nav2 DWB. Furthermore, the dead-end classifier functions
>
---
#### [new 029] Towards Affect-Adaptive Human-Robot Interaction: A Protocol for Multimodal Dataset Collection on Social Anxiety
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人机交互中的情感适应任务，旨在解决社会焦虑检测数据稀缺问题。作者提出了一种多模态数据采集协议，通过音频、视频和生理信号记录70名参与者与机器人互动的过程，以支持社交焦虑的鲁棒检测。**

- **链接: [https://arxiv.org/pdf/2511.13530v1](https://arxiv.org/pdf/2511.13530v1)**

> **作者:** Vesna Poprcova; Iulia Lefter; Matthias Wieser; Martijn Warnier; Frances Brazier
>
> **备注:** Accepted at the Workshop on Benefits of pErsonalization and behAvioral adaptation in assistive Robots (BEAR 2025), held at the IEEE RO-MAN Conference 2025
>
> **摘要:** Social anxiety is a prevalent condition that affects interpersonal interactions and social functioning. Recent advances in artificial intelligence and social robotics offer new opportunities to examine social anxiety in the human-robot interaction context. Accurate detection of affective states and behaviours associated with social anxiety requires multimodal datasets, where each signal modality provides complementary insights into its manifestations. However, such datasets remain scarce, limiting progress in both research and applications. To address this, this paper presents a protocol for multimodal dataset collection designed to reflect social anxiety in a human-robot interaction context. The dataset will consist of synchronised audio, video, and physiological recordings acquired from at least 70 participants, grouped according to their level of social anxiety, as they engage in approximately 10-minute interactive Wizard-of-Oz role-play scenarios with the Furhat social robot under controlled experimental conditions. In addition to multimodal data, the dataset will be enriched with contextual data providing deeper insight into individual variability in social anxiety responses. This work can contribute to research on affect-adaptive human-robot interaction by providing support for robust multimodal detection of social anxiety.
>
---
#### [new 030] From Power to Precision: Learning Fine-grained Dexterity for Multi-fingered Robotic Hands
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 论文解决多指机器人手在精细操作能力上的不足问题，提出软硬件协同设计方法：通过轻量化指尖几何优化与控制策略联合改进，实现功率抓握与精密操作的统一。**

- **链接: [https://arxiv.org/pdf/2511.13710v1](https://arxiv.org/pdf/2511.13710v1)**

> **作者:** Jianglong Ye; Lai Wei; Guangqi Jiang; Changwei Jing; Xueyan Zou; Xiaolong Wang
>
> **备注:** Project page: https://jianglongye.com/power-to-precision
>
> **摘要:** Human grasps can be roughly categorized into two types: power grasps and precision grasps. Precision grasping enables tool use and is believed to have influenced human evolution. Today's multi-fingered robotic hands are effective in power grasps, but for tasks requiring precision, parallel grippers are still more widely adopted. This contrast highlights a key limitation in current robotic hand design: the difficulty of achieving both stable power grasps and precise, fine-grained manipulation within a single, versatile system. In this work, we bridge this gap by jointly optimizing the control and hardware design of a multi-fingered dexterous hand, enabling both power and precision manipulation. Rather than redesigning the entire hand, we introduce a lightweight fingertip geometry modification, represent it as a contact plane, and jointly optimize its parameters along with the corresponding control. Our control strategy dynamically switches between power and precision manipulation and simplifies precision control into parallel thumb-index motions, which proves robust for sim-to-real transfer. On the design side, we leverage large-scale simulation to optimize the fingertip geometry using a differentiable neural-physics surrogate model. We validate our approach through extensive experiments in both sim-to-real and real-to-real settings. Our method achieves an 82.5% zero-shot success rate on unseen objects in sim-to-real precision grasping, and a 93.3% success rate in challenging real-world tasks involving bread pinching. These results demonstrate that our co-design framework can significantly enhance the fine-grained manipulation ability of multi-fingered hands without reducing their ability for power grasps. Our project page is at https://jianglongye.com/power-to-precision
>
---
#### [new 031] Game-Theoretic Safe Multi-Agent Motion Planning with Reachability Analysis for Dynamic and Uncertain Environments (Extended Version)
- **分类: cs.RO**

- **简介: 论文提出RE-DPG框架，用于多智能体在动态不确定环境中的安全运动规划。通过博弈论与可达性分析结合，解决协同决策复杂性和安全性保障难题，实现分布式、可扩展且理论保证的路径规划。**

- **链接: [https://arxiv.org/pdf/2511.12160v1](https://arxiv.org/pdf/2511.12160v1)**

> **作者:** Wenbin Mai; Minghui Liwang; Xinlei Yi; Xiaoyu Xia; Seyyedali Hosseinalipour; Xianbin Wang
>
> **备注:** 12 pages, 9 figures
>
> **摘要:** Ensuring safe, robust, and scalable motion planning for multi-agent systems in dynamic and uncertain environments is a persistent challenge, driven by complex inter-agent interactions, stochastic disturbances, and model uncertainties. To overcome these challenges, particularly the computational complexity of coupled decision-making and the need for proactive safety guarantees, we propose a Reachability-Enhanced Dynamic Potential Game (RE-DPG) framework, which integrates game-theoretic coordination into reachability analysis. This approach formulates multi-agent coordination as a dynamic potential game, where the Nash equilibrium (NE) defines optimal control strategies across agents. To enable scalability and decentralized execution, we develop a Neighborhood-Dominated iterative Best Response (ND-iBR) scheme, built upon an iterated $\varepsilon$-BR (i$\varepsilon$-BR) process that guarantees finite-step convergence to an $\varepsilon$-NE. This allows agents to compute strategies based on local interactions while ensuring theoretical convergence guarantees. Furthermore, to ensure safety under uncertainty, we integrate a Multi-Agent Forward Reachable Set (MA-FRS) mechanism into the cost function, explicitly modeling uncertainty propagation and enforcing collision avoidance constraints. Through both simulations and real-world experiments in 2D and 3D environments, we validate the effectiveness of RE-DPG across diverse operational scenarios.
>
---
#### [new 032] ActiveGrasp: Information-Guided Active Grasping with Calibrated Energy-based Model
- **分类: cs.RO**

- **简介: 论文提出ActiveGrasp方法，解决密集 clutter 环境下机器人抓取难题。通过校准的能量模型建模抓取分布，并基于此选择信息增益最大的视点，提升有限视野下的抓取成功率。**

- **链接: [https://arxiv.org/pdf/2511.12795v1](https://arxiv.org/pdf/2511.12795v1)**

> **作者:** Boshu Lei; Wen Jiang; Kostas Daniilidis
>
> **备注:** under review
>
> **摘要:** Grasping in a densely cluttered environment is a challenging task for robots. Previous methods tried to solve this problem by actively gathering multiple views before grasp pose generation. However, they either overlooked the importance of the grasp distribution for information gain estimation or relied on the projection of the grasp distribution, which ignores the structure of grasp poses on the SE(3) manifold. To tackle these challenges, we propose a calibrated energy-based model for grasp pose generation and an active view selection method that estimates information gain from grasp distribution. Our energy-based model captures the multi-modality nature of grasp distribution on the SE(3) manifold. The energy level is calibrated to the success rate of grasps so that the predicted distribution aligns with the real distribution. The next best view is selected by estimating the information gain for grasp from the calibrated distribution conditioned on the reconstructed environment, which could efficiently drive the robot to explore affordable parts of the target object. Experiments on simulated environments and real robot setups demonstrate that our model could successfully grasp objects in a cluttered environment with limited view budgets compared to previous state-of-the-art models. Our simulated environment can serve as a reproducible platform for future research on active grasping. The source code of our paper will be made public when the paper is released to the public.
>
---
#### [new 033] Monolithic Units: Actuation, Sensing, and Simulation for Integrated Soft Robot Design
- **分类: cs.RO**

- **简介: 论文提出一种集成驱动、传感与仿真的软体机器人单元（MU），解决软体机器人设计中传感与性能协同优化难题。通过参数化设计和仿真指导传感器布局，实现可重复制造与嵌入式感知，验证于多尺度结构与 gripper。**

- **链接: [https://arxiv.org/pdf/2511.13120v1](https://arxiv.org/pdf/2511.13120v1)**

> **作者:** Trevor Exley; Anderson Brazil Nardin; Petr Trunin; Diana Cafiso; Lucia Beccai
>
> **备注:** 8 pages, 6 figures, 1 algorithm, 1 table
>
> **摘要:** This work introduces the Monolithic Unit (MU), an actuator-lattice-sensor building block for soft robotics. The MU integrates pneumatic actuation, a compliant lattice envelope, and candidate sites for optical waveguide sensing into a single printed body. In order to study reproducibility and scalability, a parametric design framework establishes deterministic rules linking actuator chamber dimensions to lattice unit cell size. Experimental homogenization of lattice specimens provides effective material properties for finite element simulation. Within this simulation environment, sensor placement is treated as a discrete optimization problem, where a finite set of candidate waveguide paths derived from lattice nodes is evaluated by introducing local stiffening, and the configuration minimizing deviation from baseline mechanical response is selected. Optimized models are fabricated and experimentally characterized, validating the preservation of mechanical performance while enabling embedded sensing. The workflow is further extended to scaled units and a two-finger gripper, demonstrating generality of the MU concept. This approach advances monolithic soft robotic design by combining reproducible co-design rules with simulation-informed sensor integration.
>
---
#### [new 034] APP: A* Post-Processing Algorithm for Robots with Bidirectional Shortcut and Path Perturbation
- **分类: cs.RO**

- **简介: 论文提出APP算法，用于优化A*等图搜索算法生成的路径。解决路径非最短及冗余转向问题，通过双向顶点缩减和路径扰动提升路径长度与平滑性。**

- **链接: [https://arxiv.org/pdf/2511.13042v1](https://arxiv.org/pdf/2511.13042v1)**

> **作者:** Yong Li; Hui Cheng
>
> **摘要:** Paths generated by A* and other graph-search-based planners are widely used in the robotic field. Due to the restricted node-expansion directions, the resulting paths are usually not the shortest. Besides, unnecessary heading changes, or zig-zag patterns, exist even when no obstacle is nearby, which is inconsistent with the human intuition that the path segments should be straight in wide-open space due to the absence of obstacles. This article puts forward a general and systematic post-processing algorithm for A* and other graph-search-based planners. The A* post-processing algorithm, called APP, is developed based on the costmap, which is widely used in commercial service robots. First, a bidirectional vertices reduction algorithm is proposed to tackle the asymm- etry of the path and the environments. During the forward and backward vertices reduction, a thorough shortcut strategy is put forward to improve the path-shortening performance and avoid unnecessary heading changes. Second, an iterative path perturbation algorithm is adopted to locally reduce the number of unnecessary heading changes and improve the path smooth- ness. Comparative experiments are then carried out to validate the superiority of the proposed method. Quantitative performance indexes show that APP outperforms the existing methods in planning time, path length as well as the number of unnecessary heading changes. Finally, field navigation experiments are carried out to verify the practicability of APP.
>
---
#### [new 035] Structured Imitation Learning of Interactive Policies through Inverse Games
- **分类: cs.RO; cs.LG**

- **简介: 论文提出一种结构化模仿学习框架，用于学习多智能体交互策略。解决无显式沟通下的复杂交互行为建模问题，通过两步法：先学个体行为，再解逆博弈学习交互依赖，仅用50次演示即达接近真实交互策略效果。**

- **链接: [https://arxiv.org/pdf/2511.12848v1](https://arxiv.org/pdf/2511.12848v1)**

> **作者:** Max M. Sun; Todd Murphey
>
> **备注:** Presented at the "Workshop on Generative Modeling Meets Human-Robot Interaction" at Robotics: Science and Systems 2025. Workshop website: https://sites.google.com/view/gai-hri/
>
> **摘要:** Generative model-based imitation learning methods have recently achieved strong results in learning high-complexity motor skills from human demonstrations. However, imitation learning of interactive policies that coordinate with humans in shared spaces without explicit communication remains challenging, due to the significantly higher behavioral complexity in multi-agent interactions compared to non-interactive tasks. In this work, we introduce a structured imitation learning framework for interactive policies by combining generative single-agent policy learning with a flexible yet expressive game-theoretic structure. Our method explicitly separates learning into two steps: first, we learn individual behavioral patterns from multi-agent demonstrations using standard imitation learning; then, we structurally learn inter-agent dependencies by solving an inverse game problem. Preliminary results in a synthetic 5-agent social navigation task show that our method significantly improves non-interactive policies and performs comparably to the ground truth interactive policy using only 50 demonstrations. These results highlight the potential of structured imitation learning in interactive settings.
>
---
#### [new 036] ClutterNav: Gradient-Guided Search for Efficient 3D Clutter Removal with Learned Costmaps
- **分类: cs.RO**

- **简介: 论文提出ClutterNav框架，解决3D场景中密集杂乱物体下的目标物体检索问题。通过结合学习的代价模型与梯度引导搜索，实现高效、低扰动的物体移除决策，无需预设规则，在仿真和真实环境中均表现优异。**

- **链接: [https://arxiv.org/pdf/2511.12479v1](https://arxiv.org/pdf/2511.12479v1)**

> **作者:** Navin Sriram Ravie; Keerthi Vasan M; Bijo Sebastian
>
> **摘要:** Dense clutter removal for target object retrieval presents a challenging problem, especially when targets are embedded deep within densely-packed configurations. It requires foresight to minimize overall changes to the clutter configuration while accessing target objects, avoiding stack destabilization and reducing the number of object removals required. Rule-based planners when applied to this problem, rely on rigid heuristics, leading to high computational overhead. End-to-end reinforcement learning approaches struggle with interpretability and generalizability over different conditions. To address these issues, we present ClutterNav, a novel decision-making framework that can identify the next best object to be removed so as to access a target object in a given clutter, while minimising stack disturbances. ClutterNav formulates the problem as a continuous reinforcement learning task, where each object removal dynamically updates the understanding of the scene. A removability critic, trained from demonstrations, estimates the cost of removing any given object based on geometric and spatial features. This learned cost is complemented by integrated gradients that assess how the presence or removal of surrounding objects influences the accessibility of the target. By dynamically prioritizing actions that balance immediate removability against long-term target exposure, ClutterNav achieves near human-like strategic sequencing, without predefined heuristics. The proposed approach is validated extensively in simulation and over real-world experiments. The results demonstrate real-time, occlusion-aware decision-making in partially observable environments.
>
---
#### [new 037] Orientation-Free Neural Network-Based Bias Estimation for Low-Cost Stationary Accelerometers
- **分类: cs.RO; cs.LG**

- **简介: 论文提出了一种无需传感器定向的神经网络偏置估计方法，解决低成本加速度计在静止状态下因偏置误差导致精度下降的问题。通过模型无关的学习方法，在不旋转或调平传感器的情况下实现快速、准确的校准，实验表明其误差比传统方法低52%以上。**

- **链接: [https://arxiv.org/pdf/2511.13071v1](https://arxiv.org/pdf/2511.13071v1)**

> **作者:** Michal Levin; Itzik Klein
>
> **备注:** 22 pages, 10 figures
>
> **摘要:** Low-cost micro-electromechanical accelerometers are widely used in navigation, robotics, and consumer devices for motion sensing and position estimation. However, their performance is often degraded by bias errors. To eliminate deterministic bias terms a calibration procedure is applied under stationary conditions. It requires accelerom- eter leveling or complex orientation-dependent calibration procedures. To overcome those requirements, in this paper we present a model-free learning-based calibration method that estimates accelerometer bias under stationary conditions, without requiring knowledge of the sensor orientation and without the need to rotate the sensors. The proposed approach provides a fast, practical, and scalable solution suitable for rapid field deployment. Experimental validation on a 13.39-hour dataset collected from six accelerometers shows that the proposed method consistently achieves error levels more than 52% lower than traditional techniques. On a broader scale, this work contributes to the advancement of accurate calibration methods in orientation-free scenarios. As a consequence, it improves the reliability of low-cost inertial sensors in diverse scientific and industrial applications and eliminates the need for leveled calibration.
>
---
#### [new 038] Towards Obstacle-Avoiding Control of Planar Snake Robots Exploring Neuro-Evolution of Augmenting Topologies
- **分类: cs.RO**

- **简介: 论文研究平面蛇形机器人在密集障碍环境中的避障跟踪控制问题。提出基于NEAT算法生成动态步态参数，通过传感器和LiDAR数据优化奖励函数，实现高效避障控制，实验表明该方法计算效率高，性能优于现有方法。**

- **链接: [https://arxiv.org/pdf/2511.12148v1](https://arxiv.org/pdf/2511.12148v1)**

> **作者:** Advik Sinha; Akshay Arjun; Abhijit Das; Joyjit Mukherjee
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** This work aims to develop a resource-efficient solution for obstacle-avoiding tracking control of a planar snake robot in a densely cluttered environment with obstacles. Particularly, Neuro-Evolution of Augmenting Topologies (NEAT) has been employed to generate dynamic gait parameters for the serpenoid gait function, which is implemented on the joint angles of the snake robot, thus controlling the robot on a desired dynamic path. NEAT is a single neural-network based evolutionary algorithm that is known to work extremely well when the input layer is of significantly higher dimension and the output layer is of a smaller size. For the planar snake robot, the input layer consists of the joint angles, link positions, head link position as well as obstacle positions in the vicinity. However, the output layer consists of only the frequency and offset angle of the serpenoid gait that control the speed and heading of the robot, respectively. Obstacle data from a LiDAR and the robot data from various sensors, along with the location of the end goal and time, are employed to parametrize a reward function that is maximized over iterations by selective propagation of superior neural networks. The implementation and experimental results showcase that the proposed approach is computationally efficient, especially for large environments with many obstacles. The proposed framework has been verified through a physics engine simulation study on PyBullet. The approach shows superior results to existing state-of-the-art methodologies and comparable results to the very recent CBRL approach with significantly lower computational overhead. The video of the simulation can be found here: https://sites.google.com/view/neatsnakerobot
>
---
#### [new 039] GaRLILEO: Gravity-aligned Radar-Leg-Inertial Enhanced Odometry
- **分类: cs.RO**

- **简介: 论文提出GaRLILEO框架，解决腿式机器人在复杂地形中垂直位姿漂移问题。通过融合雷达、足部运动与惯性数据，构建连续时间速度样条并引入软S2约束重力因子，提升垂直定位精度，无需依赖LiDAR或相机。**

- **链接: [https://arxiv.org/pdf/2511.13216v1](https://arxiv.org/pdf/2511.13216v1)**

> **作者:** Chiyun Noh; Sangwoo Jung; Hanjun Kim; Yafei Hu; Laura Herlant; Ayoung Kim
>
> **摘要:** Deployment of legged robots for navigating challenging terrains (e.g., stairs, slopes, and unstructured environments) has gained increasing preference over wheel-based platforms. In such scenarios, accurate odometry estimation is a preliminary requirement for stable locomotion, localization, and mapping. Traditional proprioceptive approaches, which rely on leg kinematics sensor modalities and inertial sensing, suffer from irrepressible vertical drift caused by frequent contact impacts, foot slippage, and vibrations, particularly affected by inaccurate roll and pitch estimation. Existing methods incorporate exteroceptive sensors such as LiDAR or cameras. Further enhancement has been introduced by leveraging gravity vector estimation to add additional observations on roll and pitch, thereby increasing the accuracy of vertical pose estimation. However, these approaches tend to degrade in feature-sparse or repetitive scenes and are prone to errors from double-integrated IMU acceleration. To address these challenges, we propose GaRLILEO, a novel gravity-aligned continuous-time radar-leg-inertial odometry framework. GaRLILEO decouples velocity from the IMU by building a continuous-time ego-velocity spline from SoC radar Doppler and leg kinematics information, enabling seamless sensor fusion which mitigates odometry distortion. In addition, GaRLILEO can reliably capture accurate gravity vectors leveraging a novel soft S2-constrained gravity factor, improving vertical pose accuracy without relying on LiDAR or cameras. Evaluated on a self-collected real-world dataset with diverse indoor-outdoor trajectories, GaRLILEO demonstrates state-of-the-art accuracy, particularly in vertical odometry estimation on stairs and slopes. We open-source both our dataset and algorithm to foster further research in legged robot odometry and SLAM. https://garlileo.github.io/GaRLILEO
>
---
#### [new 040] ExpertAD: Enhancing Autonomous Driving Systems with Mixture of Experts
- **分类: cs.RO; cs.AI**

- **简介: 论文提出ExpertAD框架，用于提升自动驾驶系统的感知与规划能力。针对语义模糊、任务干扰和推理延迟问题，引入感知适配器和稀疏专家混合机制，显著降低碰撞率并加快决策速度，增强复杂场景下的泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.11740v1](https://arxiv.org/pdf/2511.11740v1)**

> **作者:** Haowen Jiang; Xinyu Huang; You Lu; Dingji Wang; Yuheng Cao; Chaofeng Sha; Bihuan Chen; Keyu Chen; Xin Peng
>
> **备注:** The paper has been accepted by the Fortieth AAAI Conference on Artificial Intelligence. AAAI 2026
>
> **摘要:** Recent advancements in end-to-end autonomous driving systems (ADSs) underscore their potential for perception and planning capabilities. However, challenges remain. Complex driving scenarios contain rich semantic information, yet ambiguous or noisy semantics can compromise decision reliability, while interference between multiple driving tasks may hinder optimal planning. Furthermore, prolonged inference latency slows decision-making, increasing the risk of unsafe driving behaviors. To address these challenges, we propose ExpertAD, a novel framework that enhances the performance of ADS with Mixture of Experts (MoE) architecture. We introduce a Perception Adapter (PA) to amplify task-critical features, ensuring contextually relevant scene understanding, and a Mixture of Sparse Experts (MoSE) to minimize task interference during prediction, allowing for effective and efficient planning. Our experiments show that ExpertAD reduces average collision rates by up to 20% and inference latency by 25% compared to prior methods. We further evaluate its multi-skill planning capabilities in rare scenarios (e.g., accidents, yielding to emergency vehicles) and demonstrate strong generalization to unseen urban environments. Additionally, we present a case study that illustrates its decision-making process in complex driving scenarios.
>
---
#### [new 041] Air-Chamber Based Soft Six-Axis Force/Torque Sensor for Human-Robot Interaction
- **分类: cs.RO**

- **简介: 论文提出一种基于气腔的软六维力/力矩传感器，用于人机交互中的安全精确感知。针对交叉耦合导致的校准难题，设计了16通道气压传感结构并提出分层解耦方法，显著提升精度与软性兼容性。**

- **链接: [https://arxiv.org/pdf/2511.12896v1](https://arxiv.org/pdf/2511.12896v1)**

> **作者:** Jun Huo; Hongge Ru; Bo Yang; Xingjian Chen; Xi Li; Jian Huang
>
> **摘要:** Soft multi-axis force/torque sensors provide safe and precise force interaction. Capturing the complete degree-of-freedom of force is imperative for accurate force measurement with six-axis force/torque sensors. However, cross-axis coupling can lead to calibration issues and decreased accuracy. In this instance, developing a soft and accurate six-axis sensor is a challenging task. In this paper, a soft air-chamber type six-axis force/torque sensor with 16-channel barometers is introduced, which housed in hyper-elastic air chambers made of silicone rubber. Additionally, an effective decoupling method is proposed, based on a rigid-soft hierarchical structure, which reduces the six-axis decoupling problem to two three-axis decoupling problems. Finite element model simulation and experiments demonstrate the compatibility of the proposed approach with reality. The prototype's sensing performance is quantitatively measured in terms of static load response, dynamic load response and dynamic response characteristic. It possesses a measuring range of 50 N force and 1 Nm torque, and the average deviation, repeatability, non-linearity and hysteresis are 4.9$\%$, 2.7$\%$, 5.8$\%$ and 6.7$\%$, respectively. The results indicate that the prototype exhibits satisfactory sensing performance while maintaining its softness due to the presence of soft air chambers.
>
---
#### [new 042] Hierarchical Federated Graph Attention Networks for Scalable and Resilient UAV Collision Avoidance
- **分类: cs.RO; cs.AI; cs.LG; cs.MA**

- **简介: 论文提出分层联邦图注意力网络，解决大规模多无人机系统的实时避障、抗干扰和隐私保护问题。通过三层架构实现低延迟（<10ms）、高可扩展性（500架无人机）和拜占庭容错（f < n/3）。**

- **链接: [https://arxiv.org/pdf/2511.11616v1](https://arxiv.org/pdf/2511.11616v1)**

> **作者:** Rathin Chandra Shit; Sharmila Subudhi
>
> **备注:** Accepted and scheduled for conference presentation
>
> **摘要:** The real-time performance, adversarial resiliency, and privacy preservation are the most important metrics that need to be balanced to practice collision avoidance in large-scale multi-UAV (Unmanned Aerial Vehicle) systems. Current frameworks tend to prescribe monolithic solutions that are not only prohibitively computationally complex with a scaling cost of $O(n^2)$ but simply do not offer Byzantine fault tolerance. The proposed hierarchical framework presented in this paper tries to eliminate such trade-offs by stratifying a three-layered architecture. We spread the intelligence into three layers: an immediate collision avoiding local layer running on dense graph attention with latency of $<10 ms$, a regional layer using sparse attention with $O(nk)$ computational complexity and asynchronous federated learning with coordinate-wise trimmed mean aggregation, and lastly, a global layer using a lightweight Hashgraph-inspired protocol. We have proposed an adaptive differential privacy mechanism, wherein the noise level $(ε\in [0.1, 1.0])$ is dynamically reduced based on an evaluation of the measured real-time threat that in turn maximized the privacy-utility tradeoff. Through the use of Distributed Hash Table (DHT)-based lightweight audit logging instead of heavyweight blockchain consensus, the median cost of getting a $95^{th}$ percentile decision within 50ms is observed across all tested swarm sizes. This architecture provides a scalable scenario of 500 UAVs with a collision rate of $< 2.0\%$ and the Byzantine fault tolerance of $f < n/3$.
>
---
#### [new 043] Decoupled Action Head: Confining Task Knowledge to Conditioning Layers
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 论文提出解耦动作头训练方法，解决行为克隆中数据稀缺和模型机制不明确问题。通过预训练通用动作生成器并冻结，仅用条件层适配新任务，提升效率与泛化性，并引入轻量级DP-MLP模型实现显著加速。**

- **链接: [https://arxiv.org/pdf/2511.12101v1](https://arxiv.org/pdf/2511.12101v1)**

> **作者:** Jian Zhou; Sihao Lin; Shuai Fu; Qi WU
>
> **摘要:** Behavior Cloning (BC) is a data-driven supervised learning approach that has gained increasing attention with the success of scaling laws in language and vision domains. Among its implementations in robotic manipulation, Diffusion Policy (DP), with its two variants DP-CNN (DP-C) and DP-Transformer (DP-T), is one of the most effective and widely adopted models, demonstrating the advantages of predicting continuous action sequences. However, both DP and other BC methods remain constrained by the scarcity of paired training data, and the internal mechanisms underlying DP's effectiveness remain insufficiently understood, leading to limited generalization and a lack of principled design in model development. In this work, we propose a decoupled training recipe that leverages nearly cost-free kinematics-generated trajectories as observation-free data to pretrain a general action head (action generator). The pretrained action head is then frozen and adapted to novel tasks through feature modulation. Our experiments demonstrate the feasibility of this approach in both in-distribution and out-of-distribution scenarios. As an additional benefit, decoupling improves training efficiency; for instance, DP-C achieves up to a 41% speedup. Furthermore, the confinement of task-specific knowledge to the conditioning components under decoupling, combined with the near-identical performance of DP-C in both normal and decoupled training, indicates that the action generation backbone plays a limited role in robotic manipulation. Motivated by this observation, we introduce DP-MLP, which replaces the 244M-parameter U-Net backbone of DP-C with only 4M parameters of simple MLP blocks, achieving a 83.9% faster training speed under normal training and 89.1% under decoupling.
>
---
#### [new 044] Large Language Models and 3D Vision for Intelligent Robotic Perception and Autonomy: A Review
- **分类: cs.RO; cs.CV**

- **简介: 论文探讨了大语言模型（LLMs）与3D视觉融合在机器人感知与自主性中的应用，旨在提升机器人对复杂环境的理解与交互能力。工作包括梳理核心技术、应用场景、多模态融合及评估基准，指出未来需解决实时处理、跨模态对齐等挑战。**

- **链接: [https://arxiv.org/pdf/2511.11777v1](https://arxiv.org/pdf/2511.11777v1)**

> **作者:** Vinit Mehta; Charu Sharma; Karthick Thiyagarajan
>
> **备注:** 45 pages, 15 figures, MDPI Sensors Journal
>
> **摘要:** With the rapid advancement of artificial intelligence and robotics, the integration of Large Language Models (LLMs) with 3D vision is emerging as a transformative approach to enhancing robotic sensing technologies. This convergence enables machines to perceive, reason and interact with complex environments through natural language and spatial understanding, bridging the gap between linguistic intelligence and spatial perception. This review provides a comprehensive analysis of state-of-the-art methodologies, applications and challenges at the intersection of LLMs and 3D vision, with a focus on next-generation robotic sensing technologies. We first introduce the foundational principles of LLMs and 3D data representations, followed by an in-depth examination of 3D sensing technologies critical for robotics. The review then explores key advancements in scene understanding, text-to-3D generation, object grounding and embodied agents, highlighting cutting-edge techniques such as zero-shot 3D segmentation, dynamic scene synthesis and language-guided manipulation. Furthermore, we discuss multimodal LLMs that integrate 3D data with touch, auditory and thermal inputs, enhancing environmental comprehension and robotic decision-making. To support future research, we catalog benchmark datasets and evaluation metrics tailored for 3D-language and vision tasks. Finally, we identify key challenges and future research directions, including adaptive model architectures, enhanced cross-modal alignment and real-time processing capabilities, which pave the way for more intelligent, context-aware and autonomous robotic sensing systems.
>
---
#### [new 045] Intermittent Rendezvous Plans with Mixed Integer Linear Program for Large-Scale Multi-Robot Exploration
- **分类: cs.RO; cs.HC; cs.MA**

- **简介: 论文研究大规模多机器人探索中的间歇通信问题，提出基于混合整数线性规划的 rendezvous 计划生成方法与RTUS跟踪策略，解决未知环境下路径规划与执行难题，提升任务效率与可部署性。**

- **链接: [https://arxiv.org/pdf/2511.12237v1](https://arxiv.org/pdf/2511.12237v1)**

> **作者:** Alysson Ribeiro da Silva; Luiz Chaimowicz
>
> **备注:** 9 pages, 9 figures, International Conference on Advanced Robotics
>
> **摘要:** Multi-Robot Exploration (MRE) systems with communication constraints have proven efficient in accomplishing a variety of tasks, including search-and-rescue, stealth, and military operations. While some works focus on opportunistic approaches for efficiency, others concentrate on pre-planned trajectories or scheduling for increased interpretability. However, scheduling usually requires knowledge of the environment beforehand, which prevents its deployment in several domains due to related uncertainties (e.g., underwater exploration). In our previous work, we proposed an intermittent communications framework for MRE under communication constraints that uses scheduled rendezvous events to mitigate such limitations. However, the system was unable to generate optimal plans and had no mechanisms to follow the plan considering realistic trajectories, which is not suited for real-world deployments. In this work, we further investigate the problem by formulating the Multi-Robot Exploration with Communication Constraints and Intermittent Connectivity (MRE-CCIC) problem. We propose a Mixed-Integer Linear Program (MILP) formulation to generate rendezvous plans and a policy to follow them based on the Rendezvous Tracking for Unknown Scenarios (RTUS) mechanism. The RTUS is a simple rule to allow robots to follow the assigned plan, considering unknown conditions. Finally, we evaluated our method in a large-scale environment configured in Gazebo simulations. The results suggest that our method can follow the plan promptly and accomplish the task efficiently. We provide an open-source implementation of both the MILP plan generator and the large-scale MRE-CCIC.
>
---
#### [new 046] Image-based Morphological Characterization of Filamentous Biological Structures with Non-constant Curvature Shape Feature
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于植物形态学分析任务，旨在解决如何准确量化藤蔓在机械刺激下的形状变化及其响应机制问题。作者提出基于3D分段Clothoid模型的图像分析方法，实现高精度重构与动态分析，揭示顶端区域响应更敏感，为植物力学研究和仿生机器人设计提供新方法。**

- **链接: [https://arxiv.org/pdf/2511.11639v1](https://arxiv.org/pdf/2511.11639v1)**

> **作者:** Jie Fan; Francesco Visentin; Barbara Mazzolai; Emanuela Del Dottore
>
> **备注:** This manuscript is a preprint version of the article currently under peer review at International Journal of Computer Vision (IJCV)
>
> **摘要:** Tendrils coil their shape to anchor the plant to supporting structures, allowing vertical growth toward light. Although climbing plants have been studied for a long time, extracting information regarding the relationship between the temporal shape change, the event that triggers it, and the contact location is still challenging. To help build this relation, we propose an image-based method by which it is possible to analyze shape changes over time in tendrils when mechano-stimulated in different portions of their body. We employ a geometric approach using a 3D Piece-Wise Clothoid-based model to reconstruct the configuration taken by a tendril after mechanical rubbing. The reconstruction shows high robustness and reliability with an accuracy of R2 > 0.99. This method demonstrates distinct advantages over deep learning-based approaches, including reduced data requirements, lower computational costs, and interpretability. Our analysis reveals higher responsiveness in the apical segment of tendrils, which might correspond to higher sensitivity and tissue flexibility in that region of the organs. Our study provides a methodology for gaining new insights into plant biomechanics and offers a foundation for designing and developing novel intelligent robotic systems inspired by climbing plants.
>
---
#### [new 047] SAC-MoE: Reinforcement Learning with Mixture-of-Experts for Control of Hybrid Dynamical Systems with Uncertainty
- **分类: cs.RO; eess.SY**

- **简介: 论文提出SAC-MoE方法，用于不确定混合动力系统的强化学习控制。针对隐变量和突发模式切换导致的泛化差问题，利用Mixture-of-Experts架构实现专家自适应选择，并设计课程训练策略提升鲁棒性。在赛车和足式机器人任务中验证了其优越性能。**

- **链接: [https://arxiv.org/pdf/2511.12361v1](https://arxiv.org/pdf/2511.12361v1)**

> **作者:** Leroy D'Souza; Akash Karthikeyan; Yash Vardhan Pant; Sebastian Fischmeister
>
> **摘要:** Hybrid dynamical systems result from the interaction of continuous-variable dynamics with discrete events and encompass various systems such as legged robots, vehicles and aircrafts. Challenges arise when the system's modes are characterized by unobservable (latent) parameters and the events that cause system dynamics to switch between different modes are also unobservable. Model-based control approaches typically do not account for such uncertainty in the hybrid dynamics, while standard model-free RL methods fail to account for abrupt mode switches, leading to poor generalization. To overcome this, we propose SAC-MoE which models the actor of the Soft Actor-Critic (SAC) framework as a Mixture-of-Experts (MoE) with a learned router that adaptively selects among learned experts. To further improve robustness, we develop a curriculum-based training algorithm to prioritize data collection in challenging settings, allowing better generalization to unseen modes and switching locations. Simulation studies in hybrid autonomous racing and legged locomotion tasks show that SAC-MoE outperforms baselines (up to 6x) in zero-shot generalization to unseen environments. Our curriculum strategy consistently improves performance across all evaluated policies. Qualitative analysis shows that the interpretable MoE router activates different experts for distinct latent modes.
>
---
#### [new 048] MATT-Diff: Multimodal Active Target Tracking by Diffusion Policy
- **分类: cs.RO**

- **简介: 论文提出MATT-Diff，一种基于扩散策略的多模态主动目标跟踪方法，解决未知目标数量与状态下的多目标跟踪问题。通过视觉Transformer和注意力机制融合目标估计，训练扩散模型生成探索、跟踪和重捕获等多行为模式的动作序列，实现高效主动跟踪。**

- **链接: [https://arxiv.org/pdf/2511.11931v1](https://arxiv.org/pdf/2511.11931v1)**

> **作者:** Saida Liu; Nikolay Atanasov; Shumon Koga
>
> **备注:** 14 pages, 3 figures. Submitted to L4DC 2026
>
> **摘要:** This paper proposes MATT-Diff: Multi-Modal Active Target Tracking by Diffusion Policy, a control policy that captures multiple behavioral modes - exploration, dedicated tracking, and target reacquisition - for active multi-target tracking. The policy enables agent control without prior knowledge of target numbers, states, or dynamics. Effective target tracking demands balancing exploration for undetected or lost targets with following the motion of detected but uncertain ones. We generate a demonstration dataset from three expert planners including frontier-based exploration, an uncertainty-based hybrid planner switching between frontier-based exploration and RRT* tracking based on target uncertainty, and a time-based hybrid planner switching between exploration and tracking based on target detection time. We design a control policy utilizing a vision transformer for egocentric map tokenization and an attention mechanism to integrate variable target estimates represented by Gaussian densities. Trained as a diffusion model, the policy learns to generate multi-modal action sequences through a denoising process. Evaluations demonstrate MATT-Diff's superior tracking performance against expert and behavior cloning baselines across multiple target motions, empirically validating its advantages in target tracking.
>
---
#### [new 049] SBAMP: Sampling Based Adaptive Motion Planning
- **分类: cs.RO; eess.SY**

- **简介: 论文提出SBAMP框架，融合RRT*与SEDS，解决动态环境中路径规划与实时适应性问题。无需预训练数据，实现全局最优路径与局部平滑调整的协同，保障稳定性与实时响应。**

- **链接: [https://arxiv.org/pdf/2511.12022v1](https://arxiv.org/pdf/2511.12022v1)**

> **作者:** Anh-Quan Pham; Kabir Ram Puri; Shreyas Raorane
>
> **备注:** 8 pages, 13 figures
>
> **摘要:** Autonomous robotic systems must navigate complex, dynamic environments in real time, often facing unpredictable obstacles and rapidly changing conditions. Traditional sampling-based methods, such as RRT*, excel at generating collision-free paths but struggle to adapt to sudden changes without extensive replanning. Conversely, learning-based dynamical systems, such as the Stable Estimator of Dynamical Systems (SEDS), offer smooth, adaptive trajectory tracking but typically rely on pre-collected demonstration data, limiting their generalization to novel scenarios. This paper introduces Sampling-Based Adaptive Motion Planning (SBAMP), a novel framework that overcomes these limitations by integrating RRT* for global path planning with a SEDS-based local controller for continuous, adaptive trajectory adjustment. Our approach requires no pre-trained datasets and ensures smooth transitions between planned waypoints, maintaining stability through Lyapunov-based guarantees. We validate SBAMP in both simulated environments and real hardware using the RoboRacer platform, demonstrating superior performance in dynamic obstacle scenarios, rapid recovery from perturbations, and robust handling of sharp turns. Experimental results highlight SBAMP's ability to adapt in real time without sacrificing global path optimality, providing a scalable solution for dynamic, unstructured environments.
>
---
#### [new 050] RoboAfford++: A Generative AI-Enhanced Dataset for Multimodal Affordance Learning in Robotic Manipulation and Navigation
- **分类: cs.RO**

- **简介: 论文提出RoboAfford++数据集和RoboAfford-Eval基准，解决视觉语言模型在机器人操作与导航中缺乏细粒度物体与空间可操作性理解的问题，提升其对功能抓取点和可移动区域的推理能力。**

- **链接: [https://arxiv.org/pdf/2511.12436v1](https://arxiv.org/pdf/2511.12436v1)**

> **作者:** Xiaoshuai Hao; Yingbo Tang; Lingfeng Zhang; Yanbiao Ma; Yunfeng Diao; Ziyu Jia; Wenbo Ding; Hangjun Ye; Long Chen
>
> **摘要:** Robotic manipulation and navigation are fundamental capabilities of embodied intelligence, enabling effective robot interactions with the physical world. Achieving these capabilities requires a cohesive understanding of the environment, including object recognition to localize target objects, object affordances to identify potential interaction areas and spatial affordances to discern optimal areas for both object placement and robot movement. While Vision-Language Models (VLMs) excel at high-level task planning and scene understanding, they often struggle to infer actionable positions for physical interaction, such as functional grasping points and permissible placement regions. This limitation stems from the lack of fine-grained annotations for object and spatial affordances in their training datasets. To tackle this challenge, we introduce RoboAfford++, a generative AI-enhanced dataset for multimodal affordance learning for both robotic manipulation and navigation. Our dataset comprises 869,987 images paired with 2.0 million question answering (QA) annotations, covering three critical tasks: object affordance recognition to identify target objects based on attributes and spatial relationships, object affordance prediction to pinpoint functional parts for manipulation, and spatial affordance localization to identify free space for object placement and robot navigation. Complementing this dataset, we propose RoboAfford-Eval, a comprehensive benchmark for assessing affordance-aware prediction in real-world scenarios, featuring 338 meticulously annotated samples across the same three tasks. Extensive experimental results reveal the deficiencies of existing VLMs in affordance learning, while fine-tuning on the RoboAfford++ dataset significantly enhances their ability to reason about object and spatial affordances, validating the dataset's effectiveness.
>
---
#### [new 051] Unidirectional-Road-Network-Based Global Path Planning for Cleaning Robots in Semi-Structured Environments
- **分类: cs.RO**

- **简介: 论文提出基于单向路网的全局路径规划方法，解决清洁机器人在半结构化环境中路径长度与交通规则一致性难以平衡的问题。通过构建单向路网和两层势场图，实现更短且合规的路径规划，提升导航效率与安全性。**

- **链接: [https://arxiv.org/pdf/2511.13048v1](https://arxiv.org/pdf/2511.13048v1)**

> **作者:** Yong Li; Hui Cheng
>
> **备注:** 2023 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Practical global path planning is critical for commercializing cleaning robots working in semi-structured environments. In the literature, global path planning methods for free space usually focus on path length and neglect the traffic rule constraints of the environments, which leads to high-frequency re-planning and increases collision risks. In contrast, those for structured environments are developed mainly by strictly complying with the road network representing the traffic rule constraints, which may result in an overlong path that hinders the overall navigation efficiency. This article proposes a general and systematic approach to improve global path planning performance in semi-structured environments. A unidirectional road network is built to represent the traffic constraints in semi-structured environments and a hybrid strategy is proposed to achieve a guaranteed planning result.Cutting across the road at the starting and the goal points are allowed to achieve a shorter path. Especially, a two-layer potential map is proposed to achieve a guaranteed performance when the starting and the goal points are in complex intersections. Comparative experiments are carried out to validate the effectiveness of the proposed method. Quantitative experimental results show that, compared with the state-of-art, the proposed method guarantees a much better balance between path length and the consistency with the road network.
>
---
#### [new 052] ARCSnake V2: An Amphibious Multi-Domain Screw-Propelled Snake-Like Robot
- **分类: cs.RO**

- **简介: 该论文属于机器人探索任务，旨在解决极端环境中多地形 locomotion 难题。提出 ARCSnake V2 机器人，融合螺杆推进与蛇形运动，实现陆地、沙地、水下多模式切换，具备密封设计、浮力控制与遥控操作能力。**

- **链接: [https://arxiv.org/pdf/2511.11970v1](https://arxiv.org/pdf/2511.11970v1)**

> **作者:** Sara Wickenhiser; Lizzie Peiros; Calvin Joyce; Peter Gavrilrov; Sujaan Mukherjee; Syler Sylvester; Junrong Zhou; Mandy Cheung; Jason Lim; Florian Richter; Michael C. Yip
>
> **备注:** 8 pages, 9 figures, ICRA
>
> **摘要:** Robotic exploration in extreme environments such as caves, oceans, and planetary surfaces pose significant challenges, particularly in locomotion across diverse terrains. Conventional wheeled or legged robots often struggle in these contexts due to surface variability. This paper presents ARCSnake V2, an amphibious, screw propelled, snake like robot designed for teleoperated or autonomous locomotion across land, granular media, and aquatic environments. ARCSnake V2 combines the high mobility of hyper redundant snake robots with the terrain versatility of Archimedean screw propulsion. Key contributions include a water sealed mechanical design with serially linked screw and joint actuation, an integrated buoyancy control system, and teleoperation via a kinematically matched handheld controller. The robots design and control architecture enable multiple locomotion modes screwing, wheeling, and sidewinding with smooth transitions between them. Extensive experiments validate its underwater maneuverability, communication robustness, and force regulated actuation. These capabilities position ARCSnake V2 as a versatile platform for exploration, search and rescue, and environmental monitoring in multi domain settings.
>
---
#### [new 053] Prompt-Driven Domain Adaptation for End-to-End Autonomous Driving via In-Context RL
- **分类: cs.RO; cs.LG**

- **简介: 论文提出基于上下文强化学习（ICRL）的提示驱动域适应方法，用于端到端自动驾驶在恶劣天气下的闭环控制。解决现有提示方法仅限感知任务且需专家数据的问题，无需模型更新或额外数据收集，仅用推理时观察到的通用轨迹即可实现更安全、高效的驾驶策略。**

- **链接: [https://arxiv.org/pdf/2511.12755v1](https://arxiv.org/pdf/2511.12755v1)**

> **作者:** Aleesha Khurram; Amir Moeini; Shangtong Zhang; Rohan Chandra
>
> **摘要:** Despite significant progress and advances in autonomous driving, many end-to-end systems still struggle with domain adaptation (DA), such as transferring a policy trained under clear weather to adverse weather conditions. Typical DA strategies in the literature include collecting additional data in the target domain or re-training the model, or both. Both these strategies quickly become impractical as we increase scale and complexity of driving. These limitations have encouraged investigation into few-shot and zero-shot prompt-driven DA at inference time involving LLMs and VLMs. These methods work by adding a few state-action trajectories during inference to the prompt (similar to in-context learning). However, there are two limitations of such an approach: $(i)$ prompt-driven DA methods are currently restricted to perception tasks such as detection and segmentation and $(ii)$ they require expert few-shot data. In this work, we present a new approach to inference-time few-shot prompt-driven DA for closed-loop autonomous driving in adverse weather condition using in-context reinforcement learning (ICRL). Similar to other prompt-driven DA methods, our approach does not require any updates to the model parameters nor does it require additional data collection in adversarial weather regime. Furthermore, our approach advances the state-of-the-art in prompt-driven DA by extending to closed driving using general trajectories observed during inference. Our experiments using the CARLA simulator show that ICRL results in safer, more efficient, and more comfortable driving policies in the target domain compared to state-of-the-art prompt-driven DA baselines.
>
---
#### [new 054] LAVQA: A Latency-Aware Visual Question Answering Framework for Shared Autonomy in Self-Driving Vehicles
- **分类: cs.RO**

- **简介: 论文提出LAVQA框架，解决自动驾驶中因网络延迟和人类反应时间导致的决策时机问题。通过融合VQA与时空风险可视化，动态生成碰撞地图，提升远程操作安全性，显著降低碰撞率。**

- **链接: [https://arxiv.org/pdf/2511.11840v1](https://arxiv.org/pdf/2511.11840v1)**

> **作者:** Shuangyu Xie; Kaiyuan Chen; Wenjing Chen; Chengyuan Qian; Christian Juette; Liu Ren; Dezhen Song; Ken Goldberg
>
> **摘要:** When uncertainty is high, self-driving vehicles may halt for safety and benefit from the access to remote human operators who can provide high-level guidance. This paradigm, known as {shared autonomy}, enables autonomous vehicle and remote human operators to jointly formulate appropriate responses. To address critical decision timing with variable latency due to wireless network delays and human response time, we present LAVQA, a latency-aware shared autonomy framework that integrates Visual Question Answering (VQA) and spatiotemporal risk visualization. LAVQA augments visual queries with Latency-Induced COllision Map (LICOM), a dynamically evolving map that represents both temporal latency and spatial uncertainty. It enables remote operator to observe as the vehicle safety regions vary over time in the presence of dynamic obstacles and delayed responses. Closed-loop simulations in CARLA, the de-facto standard for autonomous vehicle simulator, suggest that that LAVQA can reduce collision rates by over 8x compared to latency-agnostic baselines.
>
---
#### [new 055] Variable Impedance Control for Floating-Base Supernumerary Robotic Leg in Walking Assistance
- **分类: cs.RO**

- **简介: 论文研究浮地冗余机械腿在行走辅助中的变阻抗控制，解决因内部扰动和外部干扰导致的安全与适应性问题。通过设计混合位置/力阻抗控制器和实时稳定参数生成网络，实现柔顺与刚性状态的动态切换，提升人机交互安全性与适应性。**

- **链接: [https://arxiv.org/pdf/2511.12184v1](https://arxiv.org/pdf/2511.12184v1)**

> **作者:** Jun Huo; Kehan Xu; Chengyao Li; Yu Cao; Jie Zuo; Xinxing Chen; Jian Huang
>
> **摘要:** In human-robot systems, ensuring safety during force control in the presence of both internal and external disturbances is crucial. As a typical loosely coupled floating-base robot system, the supernumerary robotic leg (SRL) system is particularly susceptible to strong internal disturbances. To address the challenge posed by floating base, we investigated the dynamics model of the loosely coupled SRL and designed a hybrid position/force impedance controller to fit dynamic torque input. An efficient variable impedance control (VIC) method is developed to enhance human-robot interaction, particularly in scenarios involving external force disturbances. By dynamically adjusting impedance parameters, VIC improves the dynamic switching between rigidity and flexibility, so that it can adapt to unknown environmental disturbances in different states. An efficient real-time stability guaranteed impedance parameters generating network is specifically designed for the proposed SRL, to achieve shock mitigation and high rigidity supporting. Simulations and experiments validate the system's effectiveness, demonstrating its ability to maintain smooth signal transitions in flexible states while providing strong support forces in rigid states. This approach provides a practical solution for accommodating individual gait variations in interaction, and significantly advances the safety and adaptability of human-robot systems.
>
---
#### [new 056] Scaling Spatial Intelligence with Multimodal Foundation Models
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.RO**

- **简介: 论文研究如何通过数据规模扩展提升多模态基础模型的空间智能。针对现有模型在空间理解上的不足，构建了SenseNova-SI系列模型，使用800万条多样化数据训练，显著提升多个空间智能基准测试表现，同时分析了数据规模、过拟合风险及链式推理能力。**

- **链接: [https://arxiv.org/pdf/2511.13719v1](https://arxiv.org/pdf/2511.13719v1)**

> **作者:** Zhongang Cai; Ruisi Wang; Chenyang Gu; Fanyi Pu; Junxiang Xu; Yubo Wang; Wanqi Yin; Zhitao Yang; Chen Wei; Qingping Sun; Tongxi Zhou; Jiaqi Li; Hui En Pang; Oscar Qian; Yukun Wei; Zhiqian Lin; Xuanke Shi; Kewang Deng; Xiaoyang Han; Zukai Chen; Xiangyu Fan; Hanming Deng; Lewei Lu; Liang Pan; Bo Li; Ziwei Liu; Quan Wang; Dahua Lin; Lei Yang
>
> **备注:** Model: https://huggingface.co/collections/sensenova/sensenova-si; Code: https://github.com/OpenSenseNova/SenseNova-SI
>
> **摘要:** Despite remarkable progress, multimodal foundation models still exhibit surprising deficiencies in spatial intelligence. In this work, we explore scaling up multimodal foundation models to cultivate spatial intelligence within the SenseNova-SI family, built upon established multimodal foundations including visual understanding models (i.e., Qwen3-VL and InternVL3) and unified understanding and generation models (i.e., Bagel). We take a principled approach to constructing high-performing and robust spatial intelligence by systematically curating SenseNova-SI-8M: eight million diverse data samples under a rigorous taxonomy of spatial capabilities. SenseNova-SI demonstrates unprecedented performance across a broad range of spatial intelligence benchmarks: 68.7% on VSI-Bench, 43.3% on MMSI, 85.6% on MindCube, 54.6% on ViewSpatial, and 50.1% on SITE, while maintaining strong general multimodal understanding (e.g., 84.9% on MMBench-En). More importantly, we analyze the impact of data scaling, discuss early signs of emergent generalization capabilities enabled by diverse data training, analyze the risk of overfitting and language shortcuts, present a preliminary study on spatial chain-of-thought reasoning, and validate the potential downstream application. SenseNova-SI is an ongoing project, and this report will be updated continuously. All newly trained multimodal foundation models are publicly released to facilitate further research in this direction.
>
---
#### [new 057] Enhancing Reinforcement Learning in 3D Environments through Semantic Segmentation: A Case Study in ViZDoom
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 论文研究强化学习在3D环境中的应用，针对高维输入导致的内存消耗大和部分可观测问题，提出SS-only与RGB+SS两种语义分割输入表示，显著降低内存占用并提升智能体性能。**

- **链接: [https://arxiv.org/pdf/2511.11703v1](https://arxiv.org/pdf/2511.11703v1)**

> **作者:** Hugo Huang
>
> **备注:** Master's Thesis at the University of Edinburgh (2024)
>
> **摘要:** Reinforcement learning (RL) in 3D environments with high-dimensional sensory input poses two major challenges: (1) the high memory consumption induced by memory buffers required to stabilise learning, and (2) the complexity of learning in partially observable Markov Decision Processes (POMDPs). This project addresses these challenges by proposing two novel input representations: SS-only and RGB+SS, both employing semantic segmentation on RGB colour images. Experiments were conducted in deathmatches of ViZDoom, utilizing perfect segmentation results for controlled evaluation. Our results showed that SS-only was able to reduce the memory consumption of memory buffers by at least 66.6%, and up to 98.6% when a vectorisable lossless compression technique with minimal overhead such as run-length encoding is applied. Meanwhile, RGB+SS significantly enhances RL agents' performance with the additional semantic information provided. Furthermore, we explored density-based heatmapping as a tool to visualise RL agents' movement patterns and evaluate their suitability for data collection. A brief comparison with a previous approach highlights how our method overcame common pitfalls in applying semantic segmentation in 3D environments like ViZDoom.
>
---
#### [new 058] Density-Driven Optimal Control for Non-Uniform Area Coverage in Decentralized Multi-Agent Systems Using Optimal Transport
- **分类: eess.SY; cs.RO**

- **简介: 该论文研究多智能体系统中的非均匀区域覆盖任务，旨在解决现有方法无法兼顾最优性与实际约束的问题。提出密度驱动最优控制（D2OC）框架，结合最优传输理论与覆盖控制，实现带物理和操作约束的分布式最优轨迹规划，通过解析解和去中心化通信提升覆盖效率与可扩展性。**

- **链接: [https://arxiv.org/pdf/2511.12756v1](https://arxiv.org/pdf/2511.12756v1)**

> **作者:** Sungjun Seo; Kooktae Lee
>
> **备注:** Author Accepted Manuscript (AAM) of a paper accepted for publication in IEEE Transactions on Systems, Man, and Cybernetics: Systems
>
> **摘要:** This paper addresses the fundamental problem of non-uniform area coverage in multi-agent systems, where different regions require varying levels of attention due to mission-dependent priorities. Existing uniform coverage strategies are insufficient for realistic applications, and many non-uniform approaches either lack optimality guarantees or fail to incorporate crucial real-world constraints such as agent dynamics, limited operation time, the number of agents, and decentralized execution. To resolve these limitations, we propose a novel framework called Density-Driven Optimal Control (D2OC). The central idea of D2OC is the integration of optimal transport theory with multi-agent coverage control, enabling each agent to continuously adjust its trajectory to match a mission-specific reference density map. The proposed formulation establishes optimality by solving a constrained optimization problem that explicitly incorporates physical and operational constraints. The resulting control input is analytically derived from the Lagrangian of the objective function, yielding closed-form optimal solutions for linear systems and a generalizable structure for nonlinear systems. Furthermore, a decentralized data-sharing mechanism is developed to coordinate agents without reliance on global information. Comprehensive simulation studies demonstrate that D2OC achieves significantly improved non-uniform area coverage performance compared to existing methods, while maintaining scalability and decentralized implementability.
>
---
#### [new 059] PhysX-Anything: Simulation-Ready Physical 3D Assets from Single Image
- **分类: cs.CV; cs.RO**

- **简介: 论文提出PhysX-Anything，首个从单张图像生成可直接用于仿真的物理3D资产的框架，解决现有方法忽视物理属性和关节结构的问题。通过VLM模型与高效3D表示，提升生成质量和多样性，并构建新数据集PhysX-Mobility支持训练与验证。**

- **链接: [https://arxiv.org/pdf/2511.13648v1](https://arxiv.org/pdf/2511.13648v1)**

> **作者:** Ziang Cao; Fangzhou Hong; Zhaoxi Chen; Liang Pan; Ziwei Liu
>
> **备注:** Project page: https://physx-anything.github.io/
>
> **摘要:** 3D modeling is shifting from static visual representations toward physical, articulated assets that can be directly used in simulation and interaction. However, most existing 3D generation methods overlook key physical and articulation properties, thereby limiting their utility in embodied AI. To bridge this gap, we introduce PhysX-Anything, the first simulation-ready physical 3D generative framework that, given a single in-the-wild image, produces high-quality sim-ready 3D assets with explicit geometry, articulation, and physical attributes. Specifically, we propose the first VLM-based physical 3D generative model, along with a new 3D representation that efficiently tokenizes geometry. It reduces the number of tokens by 193x, enabling explicit geometry learning within standard VLM token budgets without introducing any special tokens during fine-tuning and significantly improving generative quality. In addition, to overcome the limited diversity of existing physical 3D datasets, we construct a new dataset, PhysX-Mobility, which expands the object categories in prior physical 3D datasets by over 2x and includes more than 2K common real-world objects with rich physical annotations. Extensive experiments on PhysX-Mobility and in-the-wild images demonstrate that PhysX-Anything delivers strong generative performance and robust generalization. Furthermore, simulation-based experiments in a MuJoCo-style environment validate that our sim-ready assets can be directly used for contact-rich robotic policy learning. We believe PhysX-Anything can substantially empower a broad range of downstream applications, especially in embodied AI and physics-based simulation.
>
---
#### [new 060] Proceedings Seventh International Workshop on Formal Methods for Autonomous Systems
- **分类: cs.LO; cs.AI; cs.RO**

- **简介: 该论文属于会议论文集，汇集了第七届形式化方法在自主系统中的应用研讨会（FMAS 2025）的成果。旨在促进形式化方法研究者交流，解决自主系统中的验证与可靠性问题。工作包括接收并整理来自12个国家的16篇投稿，展示当前研究进展与社区成长。**

- **链接: [https://arxiv.org/pdf/2511.13245v1](https://arxiv.org/pdf/2511.13245v1)**

> **作者:** Matt Luckcuck; Maike Schwammberger; Mengwei Xu
>
> **摘要:** This EPTCS volume contains the papers from the Seventh International Workshop on Formal Methods for Autonomous Systems (FMAS 2025), which was held between the 17th and 19th of November 2025. The goal of the FMAS workshop series is to bring together leading researchers who are using formal methods to tackle the unique challenges that autonomous systems present, so that they can publish and discuss their work with a growing community of researchers. FMAS 2025 was co-located with the 20th International Conference on integrated Formal Methods (iFM'25), hosted by Inria Paris, France at the Inria Paris Center. In total, FMAS 2025 received 16 submissions from researchers at institutions in: Canada, China, France, Germany, Ireland, Italy, Japan, the Netherlands, Portugal, Sweden, the United States of America, and the United Kingdom. Though we received fewer submissions than last year, we are encouraged to see the submissions being sent from a wide range of countries. Submissions come from both past and new FMAS authors, which shows us that the existing community appreciates the network that FMAS has built over the past 7 years, while new authors also show the FMAS community's great potential of growth.
>
---
#### [new 061] DiffPixelFormer: Differential Pixel-Aware Transformer for RGB-D Indoor Scene Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 该论文针对RGB-D室内场景分割任务，提出DiffPixelFormer模型，通过改进的自注意力机制和差分共享模块，增强模态内特征表达与跨模态对齐，提升分割精度。**

- **链接: [https://arxiv.org/pdf/2511.13047v1](https://arxiv.org/pdf/2511.13047v1)**

> **作者:** Yan Gong; Jianli Lu; Yongsheng Gao; Jie Zhao; Xiaojuan Zhang; Susanto Rahardja
>
> **备注:** 11 pages, 5 figures, 5 tables
>
> **摘要:** Indoor semantic segmentation is fundamental to computer vision and robotics, supporting applications such as autonomous navigation, augmented reality, and smart environments. Although RGB-D fusion leverages complementary appearance and geometric cues, existing methods often depend on computationally intensive cross-attention mechanisms and insufficiently model intra- and inter-modal feature relationships, resulting in imprecise feature alignment and limited discriminative representation. To address these challenges, we propose DiffPixelFormer, a differential pixel-aware Transformer for RGB-D indoor scene segmentation that simultaneously enhances intra-modal representations and models inter-modal interactions. At its core, the Intra-Inter Modal Interaction Block (IIMIB) captures intra-modal long-range dependencies via self-attention and models inter-modal interactions with the Differential-Shared Inter-Modal (DSIM) module to disentangle modality-specific and shared cues, enabling fine-grained, pixel-level cross-modal alignment. Furthermore, a dynamic fusion strategy balances modality contributions and fully exploits RGB-D information according to scene characteristics. Extensive experiments on the SUN RGB-D and NYUDv2 benchmarks demonstrate that DiffPixelFormer-L achieves mIoU scores of 54.28% and 59.95%, outperforming DFormer-L by 1.78% and 2.75%, respectively. Code is available at https://github.com/gongyan1/DiffPixelFormer.
>
---
#### [new 062] OPFormer: Object Pose Estimation leveraging foundation model with geometric encoding
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 论文提出OPFormer框架，解决物体姿态估计任务。通过融合检测与姿态估计，利用基础模型和几何编码，实现高精度6D姿态预测，适用于有无3D模型的场景。**

- **链接: [https://arxiv.org/pdf/2511.12614v1](https://arxiv.org/pdf/2511.12614v1)**

> **作者:** Artem Moroz; Vít Zeman; Martin Mikšík; Elizaveta Isianova; Miroslav David; Pavel Burget; Varun Burde
>
> **摘要:** We introduce a unified, end-to-end framework that seamlessly integrates object detection and pose estimation with a versatile onboarding process. Our pipeline begins with an onboarding stage that generates object representations from either traditional 3D CAD models or, in their absence, by rapidly reconstructing a high-fidelity neural representation (NeRF) from multi-view images. Given a test image, our system first employs the CNOS detector to localize target objects. For each detection, our novel pose estimation module, OPFormer, infers the precise 6D pose. The core of OPFormer is a transformer-based architecture that leverages a foundation model for robust feature extraction. It uniquely learns a comprehensive object representation by jointly encoding multiple template views and enriches these features with explicit 3D geometric priors using Normalized Object Coordinate Space (NOCS). A decoder then establishes robust 2D-3D correspondences to determine the final pose. Evaluated on the challenging BOP benchmarks, our integrated system demonstrates a strong balance between accuracy and efficiency, showcasing its practical applicability in both model-based and model-free scenarios.
>
---
#### [new 063] Target Defense against Sequentially Arriving Intruders: Algorithm for Agents with Dubins Dynamics
- **分类: eess.SY; cs.RO**

- **简介: 论文研究单个防御者对抗连续来袭入侵者的策略问题，考虑双方非完整动力学。通过分阶段信息模型和Dubins路径分析，提出捕获概率量化方法，并验证了有限与无限序列下的捕获率。**

- **链接: [https://arxiv.org/pdf/2511.12329v1](https://arxiv.org/pdf/2511.12329v1)**

> **作者:** Arman Pourghorban; Dipankar Maity
>
> **摘要:** We consider a variant of the target defense problem where a single defender is tasked to capture a sequence of incoming intruders. Both the defender and the intruders have non-holonomic dynamics. The intruders' objective is to breach the target perimeter without being captured by the defender, while the defender's goal is to capture as many intruders as possible. After one intruder breaches or is captured, the next appears randomly on a fixed circle surrounding the target. Therefore, the defender's final position in one game becomes its starting position for the next. We divide an intruder-defender engagement into two phases, partial information and full information, depending on the information available to the players. We address the capturability of an intruder by the defender using the notions of Dubins path and guarding arc. We quantify the percentage of capture for both finite and infinite sequences of incoming intruders. Finally, the theoretical results are verified through numerical examples using Monte-Carlo-type random trials of experiments.
>
---
#### [new 064] Density-Driven Multi-Agent Coordination for Efficient Farm Coverage and Management in Smart Agriculture
- **分类: eess.SY; cs.RO**

- **简介: 论文提出D2OC框架，解决大田农业中多无人机高效、节能覆盖喷洒问题。通过最优传输理论与动力学建模，实现按病害密度非均匀分配资源，提升覆盖率并减少农药使用。**

- **链接: [https://arxiv.org/pdf/2511.12492v1](https://arxiv.org/pdf/2511.12492v1)**

> **作者:** Sungjun Seo; Kooktae Lee
>
> **备注:** Author Accepted Manuscript (AAM) of a paper accepted for publication in the IEEE Transactions on Control Systems Technology (TCST)
>
> **摘要:** The growing scale of modern farms has increased the need for efficient and adaptive multi-agent coverage strategies for pest, weed, and disease management. Traditional methods such as manual inspection and blanket pesticide spraying often lead to excessive chemical use, resource waste, and environmental impact. While unmanned aerial vehicles (UAVs) offer a promising platform for precision agriculture through targeted spraying and improved operational efficiency, existing UAV-based approaches remain limited by battery life, payload capacity, and scalability, especially in large fields where single-UAV or uniformly distributed spraying is insufficient. Although multi-UAV coordination has been explored, many current frameworks still assume uniform spraying and do not account for infestation severity, UAV dynamics, non-uniform resource allocation, or energy-efficient coordination. To address these limitations, this paper proposes a Density-Driven Optimal Control (D2OC) framework that integrates Optimal Transport (OT) theory with multi-UAV coverage control for large-scale agricultural spraying. The method supports non-uniform, priority-aware resource allocation based on infestation intensity, reducing unnecessary chemical application. UAVs are modeled as a linear time-varying (LTV) system to capture variations in mass and inertia during spraying missions. The D2OC control law, derived using Lagrangian mechanics, enables efficient coordination, balanced workload distribution, and improved mission duration. Simulation results demonstrate that the proposed approach outperforms uniform spraying and Spectral Multiscale Coverage (SMC) in coverage efficiency, chemical reduction, and operational sustainability, providing a scalable solution for smart agriculture.
>
---
#### [new 065] Are LLMs The Way Forward? A Case Study on LLM-Guided Reinforcement Learning for Decentralized Autonomous Driving
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 论文研究小规模本地LLM如何通过奖励塑造提升强化学习在高速公路自动驾驶中的表现。针对RL依赖人工设计奖励和LLM直接控制不稳定的缺点，提出混合方法：LLM评分状态-动作转换以优化RL奖励，RL策略负责执行。结果表明混合方法平衡了成功率与效率，但存在保守偏差。**

- **链接: [https://arxiv.org/pdf/2511.12751v1](https://arxiv.org/pdf/2511.12751v1)**

> **作者:** Timur Anvar; Jeffrey Chen; Yuyan Wang; Rohan Chandra
>
> **摘要:** Autonomous vehicle navigation in complex environments such as dense and fast-moving highways and merging scenarios remains an active area of research. A key limitation of RL is its reliance on well-specified reward functions, which often fail to capture the full semantic and social complexity of diverse, out-of-distribution situations. As a result, a rapidly growing line of research explores using Large Language Models (LLMs) to replace or supplement RL for direct planning and control, on account of their ability to reason about rich semantic context. However, LLMs present significant drawbacks: they can be unstable in zero-shot safety-critical settings, produce inconsistent outputs, and often depend on expensive API calls with network latency. This motivates our investigation into whether small, locally deployed LLMs (< 14B parameters) can meaningfully support autonomous highway driving through reward shaping rather than direct control. We present a case study comparing RL-only, LLM-only, and hybrid approaches, where LLMs augment RL rewards by scoring state-action transitions during training, while standard RL policies execute at test time. Our findings reveal that RL-only agents achieve moderate success rates (73-89%) with reasonable efficiency, LLM-only agents can reach higher success rates (up to 94%) but with severely degraded speed performance, and hybrid approaches consistently fall between these extremes. Critically, despite explicit efficiency instructions, LLM-influenced approaches exhibit systematic conservative bias with substantial model-dependent variability, highlighting important limitations of current small LLMs for safety-critical control tasks.
>
---
#### [new 066] Uni-Hand: Universal Hand Motion Forecasting in Egocentric Views
- **分类: cs.CV; cs.RO**

- **简介: 论文提出EgoLoc方法，解决egocentric视频中手物交互时刻定位（TIL）问题，无需对象掩码或类别标注，通过手部动态引导采样和视觉语言模型实现零样本精准定位接触与分离时间戳。**

- **链接: [https://arxiv.org/pdf/2511.12878v1](https://arxiv.org/pdf/2511.12878v1)**

> **作者:** Junyi Ma; Wentao Bao; Jingyi Xu; Guanzhong Sun; Yu Zheng; Erhang Zhang; Xieyuanli Chen; Hesheng Wang
>
> **备注:** Extended journal version of MMTwin (IROS'25)
>
> **摘要:** Analyzing hand-object interaction in egocentric vision facilitates VR/AR applications and human-robot policy transfer. Existing research has mostly focused on modeling the behavior paradigm of interactive actions (i.e., "how to interact"). However, the more challenging and fine-grained problem of capturing the critical moments of contact and separation between the hand and the target object (i.e., "when to interact") is still underexplored, which is crucial for immersive interactive experiences in mixed reality and robotic motion planning. Therefore, we formulate this problem as temporal interaction localization (TIL). Some recent works extract semantic masks as TIL references, but suffer from inaccurate object grounding and cluttered scenarios. Although current temporal action localization (TAL) methods perform well in detecting verb-noun action segments, they rely on category annotations during training and exhibit limited precision in localizing hand-object contact/separation moments. To address these issues, we propose a novel zero-shot approach dubbed EgoLoc to localize hand-object contact and separation timestamps in egocentric videos. EgoLoc introduces hand-dynamics-guided sampling to generate high-quality visual prompts. It exploits the vision-language model to identify contact/separation attributes, localize specific timestamps, and provide closed-loop feedback for further refinement. EgoLoc eliminates the need for object masks and verb-noun taxonomies, leading to generalizable zero-shot implementation. Comprehensive experiments on the public dataset and our novel benchmarks demonstrate that EgoLoc achieves plausible TIL for egocentric videos. It is also validated to effectively facilitate multiple downstream applications in egocentric vision and robotic manipulation tasks. Code and relevant data will be released at https://github.com/IRMVLab/EgoLoc.
>
---
## 更新

#### [replaced 001] Automating RT Planning at Scale: High Quality Data For AI Training
- **分类: cs.HC; cs.LG; cs.RO**

- **链接: [https://arxiv.org/pdf/2501.11803v5](https://arxiv.org/pdf/2501.11803v5)**

> **作者:** Riqiang Gao; Mamadou Diallo; Han Liu; Anthony Magliari; Jonathan Sackett; Wilko Verbakel; Sandra Meyers; Rafe Mcbeth; Masoud Zarepisheh; Simon Arberet; Martin Kraus; Florin C. Ghesu; Ali Kamen
>
> **备注:** radiotherapy planning, data for AI training
>
> **摘要:** Radiotherapy (RT) planning is complex, subjective, and time-intensive. Advances with artificial intelligence (AI) promise to improve its precision and efficiency, but progress is often limited by the scarcity of large, standardized datasets. To address this, we introduce the Automated Iterative RT Planning (AIRTP) system, a scalable solution for generating high-quality treatment plans. This scalable solution is designed to generate substantial volumes of consistently high-quality treatment plans, overcoming a key obstacle in the advancement of AI-driven RT planning. Our AIRTP pipeline adheres to clinical guidelines and automates essential steps, including organ-at-risk (OAR) contouring, helper structure creation, beam setup, optimization, and plan quality improvement, using AI integrated with RT planning software like Varian Eclipse. Furthermore, a novel approach for determining optimization parameters to reproduce 3D dose distributions, i.e. a method to convert dose predictions to deliverable treatment plans constrained by machine limitations is proposed. A comparative analysis of plan quality reveals that our automated pipeline produces treatment plans of quality comparable to those generated manually, which traditionally require several hours of labor per plan. Committed to public research, the first data release of our AIRTP pipeline includes nine cohorts covering head-and-neck and lung cancer sites to support an AAPM 2025 challenge. To our best knowledge, this dataset features more than 10 times number of plans compared to the largest existing well-curated public dataset. Repo: https://github.com/RiqiangGao/GDP-HMM_AAPMChallenge.
>
---
#### [replaced 002] Sequential Autonomous Exploration-Based Precise Mapping for Mobile Robots through Stepwise and Consistent Motions
- **分类: cs.RO; eess.SY**

- **链接: [https://arxiv.org/pdf/2503.17005v2](https://arxiv.org/pdf/2503.17005v2)**

> **作者:** Muhua Zhang; Lei Ma; Ying Wu; Kai Shen; Yongkui Sun; Henry Leung
>
> **备注:** 9 pages, 10 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** This paper proposes a 2-D autonomous exploration and mapping framework for LiDAR-based SLAM mobile robots, designed to address the major challenges on low-cost platforms, including process instability, map drift, and increased risks of collisions and deadlocks. For frontier search, the local-global sampling architecture based on Rapidly-exploring Random Trees (RRTs) is employed. For local exploration, the proposed Self-Convergent RRT (SC-RRT) efficiently covers the reachable space within a finite time while the robot remains stationary, without relying on motion-induced sampling diversity. In addition, traversability checks during RRT expansion and global RRT pruning upon map updates eliminate unreachable frontiers, reducing potential collisions and deadlocks. For frontier point navigation, a stepwise consistent motion strategy is employed to generate motion trajectories that are more amenable to stable scan matching. The resulting straight-segment and in-place-rotation pattern improves scan-matching robustness and effectively suppresses map drift on resource-constrained platforms. For the process control, the framework serializes frontier point selection and navigation, avoiding oscillations caused by frequent goal changes in conventional parallelized processes. The waypoint retracing mechanism is incorporated to generate repeated observations, triggering loop closure detection and backend optimization in graph-based SLAM, thereby improving map consistency. Experiments in challenging simulated and real-world environments validate the effectiveness of the framework. Compared with baseline methods, the proposed framework achieves higher mapping success rates and stronger robustness on resource-constrained robots and maintains consistent mapping quality across various LiDAR field-of-view (FoV) configurations.
>
---
#### [replaced 003] Scalable Policy Evaluation with Video World Models
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2511.11520v2](https://arxiv.org/pdf/2511.11520v2)**

> **作者:** Wei-Cheng Tseng; Jinwei Gu; Qinsheng Zhang; Hanzi Mao; Ming-Yu Liu; Florian Shkurti; Lin Yen-Chen
>
> **摘要:** Training generalist policies for robotic manipulation has shown great promise, as they enable language-conditioned, multi-task behaviors across diverse scenarios. However, evaluating these policies remains difficult because real-world testing is expensive, time-consuming, and labor-intensive. It also requires frequent environment resets and carries safety risks when deploying unproven policies on physical robots. Manually creating and populating simulation environments with assets for robotic manipulation has not addressed these issues, primarily due to the significant engineering effort required and the often substantial sim-to-real gap, both in terms of physics and rendering. In this paper, we explore the use of action-conditional video generation models as a scalable way to learn world models for policy evaluation. We demonstrate how to incorporate action conditioning into existing pre-trained video generation models. This allows leveraging internet-scale in-the-wild online videos during the pre-training stage, and alleviates the need for a large dataset of paired video-action data, which is expensive to collect for robotic manipulation. Our paper examines the effect of dataset diversity, pre-trained weight and common failure cases for the proposed evaluation pipeline. Our experiments demonstrate that, across various metrics, including policy ranking and the correlation between actual policy values and predicted policy values, these models offer a promising approach for evaluating policies without requiring real-world interactions.
>
---
#### [replaced 004] GRIM: Task-Oriented Grasping with Conditioning on Generative Examples
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2506.15607v2](https://arxiv.org/pdf/2506.15607v2)**

> **作者:** Shailesh; Alok Raj; Nayan Kumar; Priya Shukla; Andrew Melnik; Michael Beetz; Gora Chand Nandi
>
> **备注:** Accepted to AAAI-26 (Oral). Project website: https://grim-tog.github.io
>
> **摘要:** Task-Oriented Grasping (TOG) requires robots to select grasps that are functionally appropriate for a specified task - a challenge that demands an understanding of task semantics, object affordances, and functional constraints. We present GRIM (Grasp Re-alignment via Iterative Matching), a training-free framework that addresses these challenges by leveraging Video Generation Models (VGMs) together with a retrieve-align-transfer pipeline. Beyond leveraging VGMs, GRIM can construct a memory of object-task exemplars sourced from web images, human demonstrations, or generative models. The retrieved task-oriented grasp is then transferred and refined by evaluating it against a set of geometrically stable candidate grasps to ensure both functional suitability and physical feasibility. GRIM demonstrates strong generalization and achieves state-of-the-art performance on standard TOG benchmarks. Project website: https://grim-tog.github.io
>
---
#### [replaced 005] DexGraspVLA: A Vision-Language-Action Framework Towards General Dexterous Grasping
- **分类: cs.RO; cs.AI**

- **链接: [https://arxiv.org/pdf/2502.20900v5](https://arxiv.org/pdf/2502.20900v5)**

> **作者:** Yifan Zhong; Xuchuan Huang; Ruochong Li; Ceyao Zhang; Zhang Chen; Tianrui Guan; Fanlian Zeng; Ka Num Lui; Yuyao Ye; Yitao Liang; Yaodong Yang; Yuanpei Chen
>
> **备注:** 18 pages, 11 figures
>
> **摘要:** Dexterous grasping remains a fundamental yet challenging problem in robotics. A general-purpose robot must be capable of grasping diverse objects in arbitrary scenarios. However, existing research typically relies on restrictive assumptions, such as single-object settings or limited environments, showing constrained generalization. We present DexGraspVLA, a hierarchical framework for robust generalization in language-guided general dexterous grasping and beyond. It utilizes a pre-trained Vision-Language model as the high-level planner and learns a diffusion-based low-level Action controller. The key insight to achieve generalization lies in iteratively transforming diverse language and visual inputs into domain-invariant representations via foundation models, where imitation learning can be effectively applied due to the alleviation of domain shift. Notably, our method achieves a 90+% dexterous grasping success rate under thousands of challenging unseen cluttered scenes. Empirical analysis confirms the consistency of internal model behavior across environmental variations, validating our design. DexGraspVLA also, for the first time, simultaneously demonstrates free-form long-horizon prompt execution, robustness to adversarial objects and human disturbance, and failure recovery. Extended application to nonprehensile grasping further proves its generality. Project website: https://dexgraspvla.github.io.
>
---
#### [replaced 006] Certified Coil Geometry Learning for Short-Range Magnetic Actuation and Spacecraft Docking Application
- **分类: cs.RO; cs.LG**

- **链接: [https://arxiv.org/pdf/2507.03806v2](https://arxiv.org/pdf/2507.03806v2)**

> **作者:** Yuta Takahashi; Hayate Tajima; Shin-ichiro Sakai
>
> **备注:** Submitted to IEEE Robotics and Automation Letters
>
> **摘要:** This paper presents a learning-based framework for approximating an exact magnetic-field interaction model, supported by both numerical and experimental validation. High-fidelity magnetic-field interaction modeling is essential for achieving exceptional accuracy and responsiveness across a wide range of fields, including transportation, energy systems, medicine, biomedical robotics, and aerospace robotics. In aerospace engineering, magnetic actuation has been investigated as a fuel-free solution for multi-satellite attitude and formation control. Although the exact magnetic field can be computed from the Biot-Savart law, the associated computational cost is prohibitive, and prior studies have therefore relied on dipole approximations to improve efficiency. However, these approximations lose accuracy during proximity operations, leading to unstable behavior and even collisions. To address this limitation, we develop a learning-based approximation framework that faithfully reproduces the exact field while dramatically reducing computational cost. The proposed method additionally provides a certified error bound, derived from the number of training samples, ensuring reliable prediction accuracy. The learned model can also accommodate interactions between coils of different sizes through appropriate geometric transformations, without retraining. To verify the effectiveness of the proposed framework under challenging conditions, a spacecraft docking scenario is examined through both numerical simulations and experimental validation.
>
---
#### [replaced 007] EnerVerse: Envisioning Embodied Future Space for Robotics Manipulation
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [https://arxiv.org/pdf/2501.01895v3](https://arxiv.org/pdf/2501.01895v3)**

> **作者:** Siyuan Huang; Liliang Chen; Pengfei Zhou; Shengcong Chen; Zhengkai Jiang; Yue Hu; Yue Liao; Peng Gao; Hongsheng Li; Maoqing Yao; Guanghui Ren
>
> **备注:** Accepted by NeurIPS 2025. Website: https://sites.google.com/view/enerverse
>
> **摘要:** We introduce EnerVerse, a generative robotics foundation model that constructs and interprets embodied spaces. EnerVerse employs a chunk-wise autoregressive video diffusion framework to predict future embodied spaces from instructions, enhanced by a sparse context memory for long-term reasoning. To model the 3D robotics world, we adopt a multi-view video representation, providing rich perspectives to address challenges like motion ambiguity and 3D grounding. Additionally, EnerVerse-D, a data engine pipeline combining generative modeling with 4D Gaussian Splatting, forms a self-reinforcing data loop to reduce the sim-to-real gap. Leveraging these innovations, EnerVerse translates 4D world representations into physical actions via a policy head (EnerVerse-A), achieving state-of-the-art performance in both simulation and real-world tasks. For efficiency, EnerVerse-A reuses features from the first denoising step and predicts action chunks, achieving about 280 ms per 8-step action chunk on a single RTX 4090. Further video demos, dataset samples could be found in our project page.
>
---
#### [replaced 008] FALCON: Learning Force-Adaptive Humanoid Loco-Manipulation
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2505.06776v2](https://arxiv.org/pdf/2505.06776v2)**

> **作者:** Yuanhang Zhang; Yifu Yuan; Prajwal Gurunath; Ishita Gupta; Shayegan Omidshafiei; Ali-akbar Agha-mohammadi; Marcell Vazquez-Chanlatte; Liam Pedersen; Tairan He; Guanya Shi
>
> **摘要:** Humanoid loco-manipulation holds transformative potential for daily service and industrial tasks, yet achieving precise, robust whole-body control with 3D end-effector force interaction remains a major challenge. Prior approaches are often limited to lightweight tasks or quadrupedal/wheeled platforms. To overcome these limitations, we propose FALCON, a dual-agent reinforcement-learning-based framework for robust force-adaptive humanoid loco-manipulation. FALCON decomposes whole-body control into two specialized agents: (1) a lower-body agent ensuring stable locomotion under external force disturbances, and (2) an upper-body agent precisely tracking end-effector positions with implicit adaptive force compensation. These two agents are jointly trained in simulation with a force curriculum that progressively escalates the magnitude of external force exerted on the end effector while respecting torque limits. Experiments demonstrate that, compared to the baselines, FALCON achieves 2x more accurate upper-body joint tracking, while maintaining robust locomotion under force disturbances and achieving faster training convergence. Moreover, FALCON enables policy training without embodiment-specific reward or curriculum tuning. Using the same training setup, we obtain policies that are deployed across multiple humanoids, enabling forceful loco-manipulation tasks such as transporting payloads (0-20N force), cart-pulling (0-100N), and door-opening (0-40N) in the real world.
>
---
#### [replaced 009] Benchmarking LLM Privacy Recognition for Social Robot Decision Making
- **分类: cs.RO; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.16124v3](https://arxiv.org/pdf/2507.16124v3)**

> **作者:** Dakota Sullivan; Shirley Zhang; Jennica Li; Heather Kirkorian; Bilge Mutlu; Kassem Fawaz
>
> **备注:** 18 pages, 7 figures. Dakota Sullivan and Shirley Zhang contributed equally to this work
>
> **摘要:** While robots have previously utilized rule-based systems or probabilistic models for user interaction, the rapid evolution of large language models (LLMs) presents new opportunities to develop LLM-powered robots for enhanced human-robot interaction (HRI). To fully realize these capabilities, however, robots need to collect data such as audio, fine-grained images, video, and locations. As a result, LLMs often process sensitive personal information, particularly within private environments, such as homes. Given the tension between utility and privacy risks, evaluating how current LLMs manage sensitive data is critical. Specifically, we aim to explore the extent to which out-of-the-box LLMs are privacy-aware in the context of household robots. In this work, we present a set of privacy-relevant scenarios developed using the Contextual Integrity (CI) framework. We first surveyed users' privacy preferences regarding in-home robot behaviors and then examined how their privacy orientations affected their choices of these behaviors (N = 450). We then provided the same set of scenarios and questions to state-of-the-art LLMs (N = 10) and found that the agreement between humans and LLMs was generally low. To further investigate the capabilities of LLMs as potential privacy controllers, we implemented four additional prompting strategies and compared their results. We discuss the performance of the evaluated models as well as the implications and potential of AI privacy awareness in human-robot interaction.
>
---
#### [replaced 010] HumanoidGen: Data Generation for Bimanual Dexterous Manipulation via LLM Reasoning
- **分类: cs.RO; cs.AI**

- **链接: [https://arxiv.org/pdf/2507.00833v2](https://arxiv.org/pdf/2507.00833v2)**

> **作者:** Zhi Jing; Siyuan Yang; Jicong Ao; Ting Xiao; Yu-Gang Jiang; Chenjia Bai
>
> **备注:** Project Page: https://openhumanoidgen.github.io
>
> **摘要:** For robotic manipulation, existing robotics datasets and simulation benchmarks predominantly cater to robot-arm platforms. However, for humanoid robots equipped with dual arms and dexterous hands, simulation tasks and high-quality demonstrations are notably lacking. Bimanual dexterous manipulation is inherently more complex, as it requires coordinated arm movements and hand operations, making autonomous data collection challenging. This paper presents HumanoidGen, an automated task creation and demonstration collection framework that leverages atomic dexterous operations and LLM reasoning to generate relational constraints. Specifically, we provide spatial annotations for both assets and dexterous hands based on the atomic operations, and perform an LLM planner to generate a chain of actionable spatial constraints for arm movements based on object affordances and scenes. To further improve planning ability, we employ a variant of Monte Carlo tree search to enhance LLM reasoning for long-horizon tasks and insufficient annotation. In experiments, we create a novel benchmark with augmented scenarios to evaluate the quality of the collected data. The results show that the performance of the 2D and 3D diffusion policies can scale with the generated dataset. Project page is https://openhumanoidgen.github.io.
>
---
#### [replaced 011] Less is More: Contextual Sampling for Nonlinear Data-Driven Predictive Control
- **分类: cs.RO; eess.SY**

- **链接: [https://arxiv.org/pdf/2503.23890v2](https://arxiv.org/pdf/2503.23890v2)**

> **作者:** Julius Beerwerth; Bassam Alrifaee
>
> **备注:** Submitted to ECC 2026 on November 14, 2025
>
> **摘要:** Data-Driven Predictive Control (DPC) optimizes system behavior directly from measured trajectories without requiring an explicit model. However, its computational cost scales with dataset size, limiting real-time applicability to nonlinear robotic systems. For robotic tasks such as trajectory tracking and motion planning, real-time feasibility and numerical robustness are essential. Nonlinear DPC often relies on large datasets or learned nonlinear representations to ensure accuracy, both of which increase computational demand. We propose Contextual Sampling, a dynamic data selection strategy that adaptively selects the most relevant trajectories based on the current state and reference. By reducing dataset size while preserving representativeness, it improves computational efficiency. Experiments on a scaled autonomous vehicle and a quadrotor show that Contextual Sampling achieves comparable or better tracking than Random Sampling with fewer trajectories, enabling real-time feasibility. Compared with Select-DPC, it achieves similar tracking accuracy at lower computational cost. In comparison with the full DPC formulation without sampling, Contextual Sampling attains comparable tracking performance while requiring less computation, highlighting the benefit of efficient data selection in data-driven predictive control.
>
---
#### [replaced 012] A Communication-Latency-Aware Co-Simulation Platform for Safety and Comfort Evaluation of Cloud-Controlled ICVs
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2506.07696v2](https://arxiv.org/pdf/2506.07696v2)**

> **作者:** Yongqi Zhao; Xinrui Zhang; Tomislav Mihalj; Martin Schabauer; Luis Putzer; Erik Reichmann-Blaga; Ádám Boronyák; András Rövid; Gábor Soós; Peizhi Zhang; Lu Xiong; Jia Hu; Arno Eichberger
>
> **备注:** 13 pages, 8 figures
>
> **摘要:** Testing cloud-controlled intelligent connected vehicles (ICVs) requires simulation environments that faithfully emulate both vehicle behavior and realistic communication latencies. This paper proposes a latency-aware co-simulation platform integrating CarMaker and Vissim to evaluate safety and comfort under real-world vehicle-to-cloud (V2C) latency conditions. Two communication latency models, derived from empirical 5G measurements in China and Hungary, are incorporated and statistically modeled using Gamma distributions. A proactive conflict module (PCM) is proposed to dynamically control background vehicles and generate safety-critical scenarios. The platform is validated through experiments involving an exemplary system under test (SUT) across six testing conditions combining two PCM modes (enabled/disabled) and three latency conditions (none, China, Hungary). Safety and comfort are assessed using metrics including collision rate, distance headway, post-encroachment time, and the spectral characteristics of longitudinal acceleration. Results show that the PCM effectively increases driving environment criticality, while V2C latency primarily affects ride comfort. These findings confirm the platform's effectiveness in systematically evaluating cloud-controlled ICVs under diverse testing conditions.
>
---
#### [replaced 013] Task-Driven Implicit Representations for Automated Design of LiDAR Systems
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2505.22344v2](https://arxiv.org/pdf/2505.22344v2)**

> **作者:** Nikhil Behari; Aaron Young; Tzofi Klinghoffer; Akshat Dave; Ramesh Raskar
>
> **摘要:** Imaging system design is a complex, time-consuming, and largely manual process; LiDAR design, ubiquitous in mobile devices, autonomous vehicles, and aerial imaging platforms, adds further complexity through unique spatial and temporal sampling requirements. In this work, we propose a framework for automated, task-driven LiDAR system design under arbitrary constraints. To achieve this, we represent LiDAR configurations in a continuous six-dimensional design space and learn task-specific implicit densities in this space via flow-based generative modeling. We then synthesize new LiDAR systems by modeling sensors as parametric distributions in 6D space and fitting these distributions to our learned implicit density using expectation-maximization, enabling efficient, constraint-aware LiDAR system design. We validate our method on diverse tasks in 3D vision, enabling automated LiDAR system design across real-world-inspired applications in face scanning, robotic tracking, and object detection.
>
---
#### [replaced 014] Large-Scale Multi-Robot Assembly Planning for Autonomous Manufacturing
- **分类: cs.RO; cs.AI**

- **链接: [https://arxiv.org/pdf/2311.00192v3](https://arxiv.org/pdf/2311.00192v3)**

> **作者:** Kyle Brown; Dylan M. Asmar; Mac Schwager; Mykel J. Kochenderfer
>
> **备注:** Repository: https://github.com/sisl/ConstructionBots.jl
>
> **摘要:** Mobile autonomous robots have the potential to revolutionize manufacturing processes. However, employing large robot fleets in manufacturing requires addressing challenges including collision-free movement in a shared workspace, effective multi-robot collaboration to manipulate and transport large payloads, complex task allocation due to coupled manufacturing processes, and spatial planning for parallel assembly and transportation of nested subassemblies. We propose a full algorithmic stack for large-scale multi-robot assembly planning that addresses these challenges and can synthesize construction plans for complex assemblies with thousands of parts in a matter of minutes. Our approach takes in a CAD-like product specification and automatically plans a full-stack assembly procedure for a group of robots to manufacture the product. We propose an algorithmic stack that comprises: (i) an iterative radial layout optimization procedure to define a global staging layout for the manufacturing facility, (ii) a graph-repair mixed-integer program formulation and a modified greedy task allocation algorithm to optimally allocate robots and robot sub-teams to assembly and transport tasks, (iii) a geometric heuristic and a hill-climbing algorithm to plan collaborative carrying configurations of robot sub-teams, and (iv) a distributed control policy that enables robots to execute the assembly motion plan collision-free. We also present an open-source multi-robot manufacturing simulator implemented in Julia as a resource to the research community, to test our algorithms and to facilitate multi-robot manufacturing research more broadly. Our empirical results demonstrate the scalability and effectiveness of our approach by generating plans to manufacture a LEGO model of a Saturn V launch vehicle with 1845 parts, 306 subassemblies, and 250 robots in under three minutes on a standard laptop computer.
>
---
#### [replaced 015] Dynamic Risk Assessment for Autonomous Vehicles from Spatio-Temporal Probabilistic Occupancy Heatmaps
- **分类: cs.RO; cs.LG; eess.SP**

- **链接: [https://arxiv.org/pdf/2501.16480v2](https://arxiv.org/pdf/2501.16480v2)**

> **作者:** Han Wang; Yuneil Yeo; Antonio R. Paiva; Jean Utke; Maria Laura Delle Monache
>
> **摘要:** Accurately assessing collision risk in dynamic traffic scenarios is a crucial requirement for trajectory planning in autonomous vehicles~(AVs) and enables a comprehensive safety evaluation of automated driving systems. To that end, this paper presents a novel probabilistic occupancy risk assessment~(PORA) metric. It uses spatiotemporal heatmaps as probabilistic occupancy predictions of surrounding traffic participants and estimates the risk of a collision along an AV's planned trajectory based on potential vehicle interactions. The use of probabilistic occupancy allows PORA to account for the uncertainty in future trajectories and velocities of traffic participants in the risk estimates. The risk from potential vehicle interactions is then further adjusted through a Cox model\edit{,} which considers the relative \edit{motion} between the AV and surrounding traffic participants. We demonstrate that the proposed approach enhances the accuracy of collision risk assessment in dynamic traffic scenarios, resulting in safer vehicle controllers, and provides a robust framework for real-time decision-making in autonomous driving systems. From evaluation in Monte Carlo simulations, PORA is shown to be more effective at accurately characterizing collision risk compared to other safety surrogate measures. Keywords: Dynamic Risk Assessment, Autonomous Vehicle, Probabilistic Occupancy, Driving Safety
>
---
#### [replaced 016] Coordinated Humanoid Robot Locomotion with Symmetry Equivariant Reinforcement Learning Policy
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2508.01247v2](https://arxiv.org/pdf/2508.01247v2)**

> **作者:** Buqing Nie; Yang Zhang; Rongjun Jin; Zhanxiang Cao; Huangxuan Lin; Xiaokang Yang; Yue Gao
>
> **备注:** AAAI 2026 accepted
>
> **摘要:** The human nervous system exhibits bilateral symmetry, enabling coordinated and balanced movements. However, existing Deep Reinforcement Learning (DRL) methods for humanoid robots neglect morphological symmetry of the robot, leading to uncoordinated and suboptimal behaviors. Inspired by human motor control, we propose Symmetry Equivariant Policy (SE-Policy), a new DRL framework that embeds strict symmetry equivariance in the actor and symmetry invariance in the critic without additional hyperparameters. SE-Policy enforces consistent behaviors across symmetric observations, producing temporally and spatially coordinated motions with higher task performance. Extensive experiments on velocity tracking tasks, conducted in both simulation and real-world deployment with the Unitree G1 humanoid robot, demonstrate that SE-Policy improves tracking accuracy by up to 40% compared to state-of-the-art baselines, while achieving superior spatial-temporal coordination. These results demonstrate the effectiveness of SE-Policy and its broad applicability to humanoid robots.
>
---
#### [replaced 017] A Skeleton-Based Topological Planner for Exploration in Complex Unknown Environments
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2412.13664v4](https://arxiv.org/pdf/2412.13664v4)**

> **作者:** Haochen Niu; Xingwu Ji; Lantao Zhang; Fei Wen; Rendong Ying; Peilin Liu
>
> **备注:** 7 pages, 7 figures. Accepted to be presented at the ICRA 2025
>
> **摘要:** The capability of autonomous exploration in complex, unknown environments is important in many robotic applications. While recent research on autonomous exploration have achieved much progress, there are still limitations, e.g., existing methods relying on greedy heuristics or optimal path planning are often hindered by repetitive paths and high computational demands. To address such limitations, we propose a novel exploration framework that utilizes the global topology information of observed environment to improve exploration efficiency while reducing computational overhead. Specifically, global information is utilized based on a skeletal topological graph representation of the environment geometry. We first propose an incremental skeleton extraction method based on wavefront propagation, based on which we then design an approach to generate a lightweight topological graph that can effectively capture the environment's structural characteristics. Building upon this, we introduce a finite state machine that leverages the topological structure to efficiently plan coverage paths, which can substantially mitigate the back-and-forth maneuvers (BFMs) problem. Experimental results demonstrate the superiority of our method in comparison with state-of-the-art methods. The source code will be made publicly available at: https://github.com/Haochen-Niu/STGPlanner.
>
---
#### [replaced 018] TopAY: Efficient Trajectory Planning for Differential Drive Mobile Manipulators via Topological Paths Search and Arc Length-Yaw Parameterization
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2507.02761v2](https://arxiv.org/pdf/2507.02761v2)**

> **作者:** Long Xu; Choilam Wong; Mengke Zhang; Junxiao Lin; Jialiang Hou; Fei Gao
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Differential drive mobile manipulators combine the mobility of wheeled bases with the manipulation capability of multi-joint arms, enabling versatile applications but posing considerable challenges for trajectory planning due to their high-dimensional state space and nonholonomic constraints. This paper introduces TopAY, an optimization-based planning framework designed for efficient and safe trajectory generation for differential drive mobile manipulators. The framework employs a hierarchical initial value acquisition strategy, including topological paths search for the base and parallel sampling for the manipulator. A polynomial trajectory representation with arc length-yaw parameterization is also proposed to reduce optimization complexity while preserving dynamic feasibility. Extensive simulation and real-world experiments validate that TopAY achieves higher planning efficiency and success rates than state-of-the-art method in dense and complex scenarios. The source code is released at https://github.com/TopAY-Planner/TopAY .
>
---
#### [replaced 019] MASt3R-Fusion: Integrating Feed-Forward Visual Model with IMU, GNSS for High-Functionality SLAM
- **分类: cs.RO; cs.CV**

- **链接: [https://arxiv.org/pdf/2509.20757v3](https://arxiv.org/pdf/2509.20757v3)**

> **作者:** Yuxuan Zhou; Xingxing Li; Shengyu Li; Zhuohao Yan; Chunxi Xia; Shaoquan Feng
>
> **摘要:** Visual SLAM is a cornerstone technique in robotics, autonomous driving and extended reality (XR), yet classical systems often struggle with low-texture environments, scale ambiguity, and degraded performance under challenging visual conditions. Recent advancements in feed-forward neural network-based pointmap regression have demonstrated the potential to recover high-fidelity 3D scene geometry directly from images, leveraging learned spatial priors to overcome limitations of traditional multi-view geometry methods. However, the widely validated advantages of probabilistic multi-sensor information fusion are often discarded in these pipelines. In this work, we propose MASt3R-Fusion,a multi-sensor-assisted visual SLAM framework that tightly integrates feed-forward pointmap regression with complementary sensor information, including inertial measurements and GNSS data. The system introduces Sim(3)-based visualalignment constraints (in the Hessian form) into a universal metric-scale SE(3) factor graph for effective information fusion. A hierarchical factor graph design is developed, which allows both real-time sliding-window optimization and global optimization with aggressive loop closures, enabling real-time pose tracking, metric-scale structure perception and globally consistent mapping. We evaluate our approach on both public benchmarks and self-collected datasets, demonstrating substantial improvements in accuracy and robustness over existing visual-centered multi-sensor SLAM systems. The code will be released open-source to support reproducibility and further research (https://github.com/GREAT-WHU/MASt3R-Fusion).
>
---
#### [replaced 020] GUIDES: Guidance Using Instructor-Distilled Embeddings for Pre-trained Robot Policy Enhancement
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2511.03400v2](https://arxiv.org/pdf/2511.03400v2)**

> **作者:** Minquan Gao; Xinyi Li; Qing Yan; Xiaojian Sun; Xiaopan Zhang; Chien-Ming Huang; Jiachen Li
>
> **备注:** 8 pages, 4 figures, Accepted by IEEE IROS 2025 Workshop WIR-M
>
> **摘要:** Pre-trained robot policies serve as the foundation of many validated robotic systems, which encapsulate extensive embodied knowledge. However, they often lack the semantic awareness characteristic of foundation models, and replacing them entirely is impractical in many situations due to high costs and the loss of accumulated knowledge. To address this gap, we introduce GUIDES, a lightweight framework that augments pre-trained policies with semantic guidance from foundation models without requiring architectural redesign. GUIDES employs a fine-tuned vision-language model (Instructor) to generate contextual instructions, which are encoded by an auxiliary module into guidance embeddings. These embeddings are injected into the policy's latent space, allowing the legacy model to adapt to this new semantic input through brief, targeted fine-tuning. For inference-time robustness, a large language model-based Reflector monitors the Instructor's confidence and, when confidence is low, initiates a reasoning loop that analyzes execution history, retrieves relevant examples, and augments the VLM's context to refine subsequent actions. Extensive validation in the RoboCasa simulation environment across diverse policy architectures shows consistent and substantial improvements in task success rates. Real-world deployment on a UR5 robot further demonstrates that GUIDES enhances motion precision for critical sub-tasks such as grasping. Overall, GUIDES offers a practical and resource-efficient pathway to upgrade, rather than replace, validated robot policies.
>
---
#### [replaced 021] Reward Redistribution via Gaussian Process Likelihood Estimation
- **分类: cs.LG; cs.RO**

- **链接: [https://arxiv.org/pdf/2503.17409v2](https://arxiv.org/pdf/2503.17409v2)**

> **作者:** Minheng Xiao; Xian Yu
>
> **备注:** Accepted by AAAI-26
>
> **摘要:** In many practical reinforcement learning tasks, feedback is only provided at the end of a long horizon, leading to sparse and delayed rewards. Existing reward redistribution methods typically assume that per-step rewards are independent, thus overlooking interdependencies among state-action pairs. In this paper, we propose a Gaussian process based Likelihood Reward Redistribution (GP-LRR) framework that addresses this issue by modeling the reward function as a sample from a Gaussian process, which explicitly captures dependencies between state-action pairs through the kernel function. By maximizing the likelihood of the observed episodic return via a leave-one-out strategy that leverages the entire trajectory, our framework inherently introduces uncertainty regularization. Moreover, we show that conventional mean-squared-error (MSE) based reward redistribution arises as a special case of our GP-LRR framework when using a degenerate kernel without observation noise. When integrated with an off-policy algorithm such as Soft Actor-Critic, GP-LRR yields dense and informative reward signals, resulting in superior sample efficiency and policy performance on several MuJoCo benchmarks.
>
---
#### [replaced 022] AI-Driven Robotics for Optics
- **分类: physics.optics; cs.RO**

- **链接: [https://arxiv.org/pdf/2505.17985v2](https://arxiv.org/pdf/2505.17985v2)**

> **作者:** Shiekh Zia Uddin; Sachin Vaidya; Shrish Choudhary; Zhuo Chen; Raafat K. Salib; Luke Huang; Dirk R. Englund; Marin Soljačić
>
> **摘要:** Optics is foundational to research in many areas of science and engineering, including nanophotonics, quantum information, materials science, biomedical imaging, and metrology. However, the design, assembly, and alignment of optical experiments remain predominantly manual, limiting throughput and reproducibility. Automating such experiments is challenging due to the strict, non-negotiable precision requirements and the diversity of optical configurations found in typical laboratories. Here, we introduce a platform that integrates generative artificial intelligence, computer vision, and robotics to automate free-space optical experiments. The platform translates user-defined goals into valid optical configurations, assembles them using a robotic arm, and performs micrometer-scale fine alignment using a robot-deployable tool. It then executes a range of automated measurements, including beam characterization, polarization mapping, and spectroscopy, with consistency surpassing that of human operators. This work demonstrates the first flexible, AI-driven automation platform for optics, offering a path towards remote operation, cloud labs, and high-throughput discovery in the optical sciences.
>
---
#### [replaced 023] LLM-Driven Robots Risk Enacting Discrimination, Violence, and Unlawful Actions
- **分类: cs.RO; cs.AI; cs.CL; cs.CY**

- **链接: [https://arxiv.org/pdf/2406.08824v2](https://arxiv.org/pdf/2406.08824v2)**

> **作者:** Andrew Hundt; Rumaisa Azeem; Masoumeh Mansouri; Martim Brandão
>
> **备注:** Published in International Journal of Social Robotics (2025). 49 pages (65 with references and appendix), 27 Figures, 8 Tables. Andrew Hundt and Rumaisa Azeem are equal contribution co-first authors. The positions of the two co-first authors were swapped from arxiv version 1 with the written consent of all four authors. The Version of Record is available via DOI: 10.1007/s12369-025-01301-x
>
> **摘要:** Members of the Human-Robot Interaction (HRI) and Machine Learning (ML) communities have proposed Large Language Models (LLMs) as a promising resource for robotics tasks such as natural language interaction, household and workplace tasks, approximating 'common sense reasoning', and modeling humans. However, recent research has raised concerns about the potential for LLMs to produce discriminatory outcomes and unsafe behaviors in real-world robot experiments and applications. To assess whether such concerns are well placed in the context of HRI, we evaluate several highly-rated LLMs on discrimination and safety criteria. Our evaluation reveals that LLMs are currently unsafe for people across a diverse range of protected identity characteristics, including, but not limited to, race, gender, disability status, nationality, religion, and their intersections. Concretely, we show that LLMs produce directly discriminatory outcomes- e.g., 'gypsy' and 'mute' people are labeled untrustworthy, but not 'european' or 'able-bodied' people. We find various such examples of direct discrimination on HRI tasks such as facial expression, proxemics, security, rescue, and task assignment. Furthermore, we test models in settings with unconstrained natural language (open vocabulary) inputs, and find they fail to act safely, generating responses that accept dangerous, violent, or unlawful instructions-such as incident-causing misstatements, taking people's mobility aids, and sexual predation. Our results underscore the urgent need for systematic, routine, and comprehensive risk assessments and assurances to improve outcomes and ensure LLMs only operate on robots when it is safe, effective, and just to do so. We provide code to reproduce our experiments at https://github.com/rumaisa-azeem/llm-robots-discrimination-safety .
>
---
#### [replaced 024] Extendable Planning via Multiscale Diffusion
- **分类: cs.LG; cs.RO**

- **链接: [https://arxiv.org/pdf/2503.20102v3](https://arxiv.org/pdf/2503.20102v3)**

> **作者:** Chang Chen; Hany Hamed; Doojin Baek; Taegu Kang; Samyeul Noh; Yoshua Bengio; Sungjin Ahn
>
> **备注:** First two authors contributed equally
>
> **摘要:** Long-horizon planning is crucial in complex environments, but diffusion-based planners like Diffuser are limited by the trajectory lengths observed during training. This creates a dilemma: long trajectories are needed for effective planning, yet they degrade model performance. In this paper, we introduce this extendable long-horizon planning challenge and propose a two-phase solution. First, Progressive Trajectory Extension incrementally constructs longer trajectories through multi-round compositional stitching. Second, the Hierarchical Multiscale Diffuser enables efficient training and inference over long horizons by reasoning across temporal scales. To avoid the need for multiple separate models, we propose Adaptive Plan Pondering and the Recursive HM-Diffuser, which unify hierarchical planning within a single model. Experiments show our approach yields strong performance gains, advancing scalable and efficient decision-making over long-horizons.
>
---
#### [replaced 025] Human2Robot: Learning Robot Actions from Paired Human-Robot Videos
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2502.16587v4](https://arxiv.org/pdf/2502.16587v4)**

> **作者:** Sicheng Xie; Haidong Cao; Zejia Weng; Zhen Xing; Haoran Chen; Shiwei Shen; Jiaqi Leng; Zuxuan Wu; Yu-Gang Jiang
>
> **摘要:** Distilling knowledge from human demonstrations is a promising way for robots to learn and act. Existing methods, which often rely on coarsely-aligned video pairs, are typically constrained to learning global or task-level features. As a result, they tend to neglect the fine-grained frame-level dynamics required for complex manipulation and generalization to novel tasks. We posit that this limitation stems from a vicious circle of inadequate datasets and the methods they inspire. To break this cycle, we propose a paradigm shift that treats fine-grained human-robot alignment as a conditional video generation problem. To this end, we first introduce H&R, a novel third-person dataset containing 2,600 episodes of precisely synchronized human and robot motions, collected using a VR teleoperation system. We then present Human2Robot, a framework designed to leverage this data. Human2Robot employs a Video Prediction Model to learn a rich and implicit representation of robot dynamics by generating robot videos from human input, which in turn guides a decoupled action decoder. Our real-world experiments demonstrate that this approach not only achieves high performance on seen tasks but also exhibits significant one-shot generalization to novel positions, objects, instances, and even new task categories.
>
---
#### [replaced 026] PB-NBV: Efficient Projection-Based Next-Best-View Planning Framework for Reconstruction of Unknown Objects
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2501.10663v2](https://arxiv.org/pdf/2501.10663v2)**

> **作者:** Zhizhou Jia; Yuetao Li; Qun Hao; Shaohui Zhang
>
> **备注:** Accepted to IEEE Robotics and Automation Letters (RA-L), 2025
>
> **摘要:** Completely capturing the three-dimensional (3D) data of an object is essential in industrial and robotic applications. The task of next-best-view (NBV) planning is to calculate the next optimal viewpoint based on the current data, gradually achieving a complete 3D reconstruction of the object. However, many existing NBV planning algorithms incur heavy computational costs due to the extensive use of ray-casting. Specifically, this framework refits different types of voxel clusters into ellipsoids based on the voxel structure. Then, the next optimal viewpoint is selected from the candidate views using a projection-based viewpoint quality evaluation function in conjunction with a global partitioning strategy. This process replaces extensive ray-casting, significantly improving the computational efficiency. Comparison experiments in the simulation environment show that our framework achieves the highest point cloud coverage with low computational time compared to other frameworks. The real-world experiments also confirm the efficiency and feasibility of the framework. Our method will be made open source to benefit the community.
>
---
#### [replaced 027] Local Guidance for Configuration-Based Multi-Agent Pathfinding
- **分类: cs.MA; cs.AI; cs.RO**

- **链接: [https://arxiv.org/pdf/2510.19072v3](https://arxiv.org/pdf/2510.19072v3)**

> **作者:** Tomoki Arita; Keisuke Okumura
>
> **备注:** To be presented at AAAI-26
>
> **摘要:** Guidance is an emerging concept that improves the empirical performance of real-time, sub-optimal multi-agent pathfinding (MAPF) methods. It offers additional information to MAPF algorithms to mitigate congestion on a global scale by considering the collective behavior of all agents across the entire workspace. This global perspective helps reduce agents' waiting times, thereby improving overall coordination efficiency. In contrast, this study explores an alternative approach: providing local guidance in the vicinity of each agent. While such localized methods involve recomputation as agents move and may appear computationally demanding, we empirically demonstrate that supplying informative spatiotemporal cues to the planner can significantly improve solution quality without exceeding a moderate time budget. When applied to LaCAM, a leading configuration-based solver, this form of guidance establishes a new performance frontier for MAPF.
>
---
#### [replaced 028] Whole-Body Control Framework for Humanoid Robots with Heavy Limbs: A Model-Based Approach
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2506.14278v2](https://arxiv.org/pdf/2506.14278v2)**

> **作者:** Tianlin Zhang; Linzhu Yue; Hongbo Zhang; Lingwei Zhang; Xuanqi Zeng; Zhitao Song; Yun-Hui Liu
>
> **摘要:** Humanoid robots often face significant balance issues due to the motion of their heavy limbs. These challenges are particularly pronounced when attempting dynamic motion or operating in environments with irregular terrain. To address this challenge, this manuscript proposes a whole-body control framework for humanoid robots with heavy limbs, using a model-based approach that combines a kino-dynamics planner and a hierarchical optimization problem. The kino-dynamics planner is designed as a model predictive control (MPC) scheme to account for the impact of heavy limbs on mass and inertia distribution. By simplifying the robot's system dynamics and constraints, the planner enables real-time planning of motion and contact forces. The hierarchical optimization problem is formulated using Hierarchical Quadratic Programming (HQP) to minimize limb control errors and ensure compliance with the policy generated by the kino-dynamics planner. Experimental validation of the proposed framework demonstrates its effectiveness. The humanoid robot with heavy limbs controlled by the proposed framework can achieve dynamic walking speeds of up to 1.2~m/s, respond to external disturbances of up to 60~N, and maintain balance on challenging terrains such as uneven surfaces, and outdoor environments.
>
---
#### [replaced 029] MonoDream: Monocular Vision-Language Navigation with Panoramic Dreaming
- **分类: cs.CV; cs.RO**

- **链接: [https://arxiv.org/pdf/2508.02549v2](https://arxiv.org/pdf/2508.02549v2)**

> **作者:** Shuo Wang; Yongcai Wang; Zhaoxin Fan; Yucheng Wang; Maiyue Chen; Kaihui Wang; Zhizhong Su; Wanting Li; Xudong Cai; Yeying Jin; Deying Li
>
> **摘要:** Vision-Language Navigation (VLN) tasks often leverage panoramic RGB and depth inputs to provide rich spatial cues for action planning, but these sensors can be costly or less accessible in real-world deployments. Recent approaches based on Vision-Language Action (VLA) models achieve strong results with monocular input, yet they still lag behind methods using panoramic RGB-D information. We present MonoDream, a lightweight VLA framework that enables monocular agents to learn a Unified Navigation Representation (UNR). This shared feature representation jointly aligns navigation-relevant visual semantics (e.g., global layout, depth, and future cues) and language-grounded action intent, enabling more reliable action prediction. MonoDream further introduces Latent Panoramic Dreaming (LPD) tasks to supervise the UNR, which train the model to predict latent features of panoramic RGB and depth observations at both current and future steps based on only monocular input. Experiments on multiple VLN benchmarks show that MonoDream consistently improves monocular navigation performance and significantly narrows the gap with panoramic-based agents.
>
---
#### [replaced 030] Scaffolding Dexterous Manipulation with Vision-Language Models
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2506.19212v2](https://arxiv.org/pdf/2506.19212v2)**

> **作者:** Vincent de Bakker; Joey Hejna; Tyler Ga Wei Lum; Onur Celik; Aleksandar Taranovic; Denis Blessing; Gerhard Neumann; Jeannette Bohg; Dorsa Sadigh
>
> **摘要:** Dexterous robotic hands are essential for performing complex manipulation tasks, yet remain difficult to train due to the challenges of demonstration collection and high-dimensional control. While reinforcement learning (RL) can alleviate the data bottleneck by generating experience in simulation, it typically relies on carefully designed, task-specific reward functions, which hinder scalability and generalization. Thus, contemporary works in dexterous manipulation have often bootstrapped from reference trajectories. These trajectories specify target hand poses that guide the exploration of RL policies and object poses that enable dense, task-agnostic rewards. However, sourcing suitable trajectories - particularly for dexterous hands - remains a significant challenge. Yet, the precise details in explicit reference trajectories are often unnecessary, as RL ultimately refines the motion. Our key insight is that modern vision-language models (VLMs) already encode the commonsense spatial and semantic knowledge needed to specify tasks and guide exploration effectively. Given a task description (e.g., "open the cabinet") and a visual scene, our method uses an off-the-shelf VLM to first identify task-relevant keypoints (e.g., handles, buttons) and then synthesize 3D trajectories for hand motion and object motion. Subsequently, we train a low-level residual RL policy in simulation to track these coarse trajectories or "scaffolds" with high fidelity. Across a number of simulated tasks involving articulated objects and semantic understanding, we demonstrate that our method is able to learn robust dexterous manipulation policies. Moreover, we showcase that our method transfers to real-world robotic hands without any human demonstrations or handcrafted rewards.
>
---
#### [replaced 031] A Cooperation Control Framework Based on Admittance Control and Time-varying Passive Velocity Field Control for Human-Robot Co-carrying Tasks
- **分类: cs.RO**

- **链接: [https://arxiv.org/pdf/2407.21339v2](https://arxiv.org/pdf/2407.21339v2)**

> **作者:** Dang Van Trong; Hiroki Kotake; Sumitaka Honji; Takahiro Wada
>
> **备注:** 15 pages, 13 figures. This is a preprint of an article accepted for publication in IEEE Transactions on Automation Science and Engineering
>
> **摘要:** Human-robot co-carrying tasks reveal their potential in both industrial and everyday applications by leveraging the strengths of both parties. Effective control of robots in these tasks requires managing the energy level in the closed-loop systems to prevent potential dangers while also minimizing motion errors to complete the shared tasks. The collaborative tasks pose numerous challenges due to varied human intentions in adapting to workspace characteristics, leading to human-robot conflicts. In this paper, we develop a cooperation control framework for human-robot co-carrying tasks constructed by utilizing reference generator and low-level controller to aim to achieve safe interaction and synchronized human-robot movement. Firstly, the human motion predictions are corrected in the event of prediction errors based on the conflicts measured by the interaction forces through admittance control, thereby mitigating conflict levels. Low-level controller using an energy-compensation passive velocity field control approach allows encoding the corrected motion to produce control torques for the robot. In this manner, the closed-loop robotic system is passive when the energy level exceeds the predetermined threshold, and otherwise. Furthermore, the proposed control approach ensures that the system's kinetic energy is compensated within a finite time interval. The passivity, stability, convergence rate of energy, and power flow regulation are analyzed from theoretical viewpoints. Human-in-the-loop experiments involving 18 participants have demonstrated that the proposed method significantly enhances task performance and reduces human workload, as evidenced by both objective metrics and subjective evaluations, with improvements confirmed by statistical tests (p < 0.05) relative to baseline methods.
>
---
#### [replaced 032] Bench2FreeAD: A Benchmark for Vision-based End-to-end Navigation in Unstructured Robotic Environments
- **分类: cs.RO; cs.CV**

- **链接: [https://arxiv.org/pdf/2503.12180v2](https://arxiv.org/pdf/2503.12180v2)**

> **作者:** Yuhang Peng; Sidong Wang; Jihaoyu Yang; Shilong Li; Han Wang; Jiangtao Gong
>
> **备注:** 7 pages, 9 figures
>
> **摘要:** Most current end-to-end (E2E) autonomous driving algorithms are built on standard vehicles in structured transportation scenarios, lacking exploration of robot navigation for unstructured scenarios such as auxiliary roads, campus roads, and indoor settings. This paper investigates E2E robot navigation in unstructured road environments. First, we introduce two data collection pipelines - one for real-world robot data and another for synthetic data generated using the Isaac Sim simulator, which together produce an unstructured robotics navigation dataset -- FreeWorld Dataset. Second, we fine-tuned an efficient E2E autonomous driving model -- VAD -- using our datasets to validate the performance and adaptability of E2E autonomous driving models in these environments. Results demonstrate that fine-tuning through our datasets significantly enhances the navigation potential of E2E autonomous driving models in unstructured robotic environments. Thus, this paper presents the first dataset targeting E2E robot navigation tasks in unstructured scenarios, and provides a benchmark based on vision-based E2E autonomous driving algorithms to facilitate the development of E2E navigation technology for logistics and service robots. The project is available on Github.
>
---
