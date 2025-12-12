# 机器人 cs.RO

- **最新发布 33 篇**

- **更新 23 篇**

## 最新发布

#### [new 001] LEO-RobotAgent: A General-purpose Robotic Agent for Language-driven Embodied Operator
- **分类: cs.RO**

- **简介: 该论文提出LEO-RobotAgent框架，解决现有机器人任务规划中泛化性差、结构复杂的问题。其面向语言驱动的通用机器人代理任务，实现大模型操控多类机器人完成跨场景复杂任务，具备强泛化性、鲁棒性和人机协作能力。**

- **链接: [https://arxiv.org/pdf/2512.10605v1](https://arxiv.org/pdf/2512.10605v1)**

> **作者:** Lihuang Chen; Xiangyu Luo; Jun Meng
>
> **摘要:** We propose LEO-RobotAgent, a general-purpose language-driven intelligent agent framework for robots. Under this framework, LLMs can operate different types of robots to complete unpredictable complex tasks across various scenarios. This framework features strong generalization, robustness, and efficiency. The application-level system built around it can fully enhance bidirectional human-robot intent understanding and lower the threshold for human-robot interaction. Regarding robot task planning, the vast majority of existing studies focus on the application of large models in single-task scenarios and for single robot types. These algorithms often have complex structures and lack generalizability. Thus, the proposed LEO-RobotAgent framework is designed with a streamlined structure as much as possible, enabling large models to independently think, plan, and act within this clear framework. We provide a modular and easily registrable toolset, allowing large models to flexibly call various tools to meet different requirements. Meanwhile, the framework incorporates a human-robot interaction mechanism, enabling the algorithm to collaborate with humans like a partner. Experiments have verified that this framework can be easily adapted to mainstream robot platforms including unmanned aerial vehicles (UAVs), robotic arms, and wheeled robot, and efficiently execute a variety of carefully designed tasks with different complexity levels. Our code is available at https://github.com/LegendLeoChen/LEO-RobotAgent.
>
---
#### [new 002] Mr. Virgil: Learning Multi-robot Visual-range Relative Localization
- **分类: cs.RO**

- **简介: 该论文研究多机器人视觉-测距相对定位任务，解决UWB与视觉匹配易出错的问题。提出Mr. Virgil框架，采用图神经网络进行数据关联，并结合可微姿态图优化，实现鲁棒、精确的相对定位。**

- **链接: [https://arxiv.org/pdf/2512.10540v1](https://arxiv.org/pdf/2512.10540v1)**

> **作者:** Si Wang; Zhehan Li; Jiadong Lu; Rong Xiong; Yanjun Cao; Yue Wang
>
> **备注:** Accepted by 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Ultra-wideband (UWB)-vision fusion localization has achieved extensive applications in the domain of multi-agent relative localization. The challenging matching problem between robots and visual detection renders existing methods highly dependent on identity-encoded hardware or delicate tuning algorithms. Overconfident yet erroneous matches may bring about irreversible damage to the localization system. To address this issue, we introduce Mr. Virgil, an end-to-end learning multi-robot visual-range relative localization framework, consisting of a graph neural network for data association between UWB rangings and visual detections, and a differentiable pose graph optimization (PGO) back-end. The graph-based front-end supplies robust matching results, accurate initial position predictions, and credible uncertainty estimates, which are subsequently integrated into the PGO back-end to elevate the accuracy of the final pose estimation. Additionally, a decentralized system is implemented for real-world applications. Experiments spanning varying robot numbers, simulation and real-world, occlusion and non-occlusion conditions showcase the stability and exactitude under various scenes compared to conventional methods. Our code is available at: https://github.com/HiOnes/Mr-Virgil.
>
---
#### [new 003] Contact SLAM: An Active Tactile Exploration Policy Based on Physical Reasoning Utilized in Robotic Fine Blind Manipulation Tasks
- **分类: cs.RO**

- **简介: 该论文研究机器人在视觉受限下的精细操作任务，提出“Contact SLAM”方法，利用触觉和物理推理实现环境状态估计，并设计主动探索策略以降低不确定性，成功应用于插孔装配和推块等高接触性操作。**

- **链接: [https://arxiv.org/pdf/2512.10481v1](https://arxiv.org/pdf/2512.10481v1)**

> **作者:** Gaozhao Wang; Xing Liu; Zhenduo Ye; Zhengxiong Liu; Panfeng Huang
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Contact-rich manipulation is difficult for robots to execute and requires accurate perception of the environment. In some scenarios, vision is occluded. The robot can then no longer obtain real-time scene state information through visual feedback. This is called ``blind manipulation". In this manuscript, a novel physically-driven contact cognition method, called ``Contact SLAM", is proposed. It estimates the state of the environment and achieves manipulation using only tactile sensing and prior knowledge of the scene. To maximize exploration efficiency, this manuscript also designs an active exploration policy. The policy gradually reduces uncertainties in the manipulation scene. The experimental results demonstrated the effectiveness and accuracy of the proposed method in several contact-rich tasks, including the difficult and delicate socket assembly task and block-pushing task.
>
---
#### [new 004] Task-Oriented Grasping Using Reinforcement Learning with a Contextual Reward Machine
- **分类: cs.RO**

- **简介: 该论文研究任务导向的抓取问题，提出结合上下文奖励机的强化学习方法。通过分解任务、定义阶段上下文和转移奖励，提升学习效率与成功率，并在仿真和真实机器人中验证有效性。**

- **链接: [https://arxiv.org/pdf/2512.10235v1](https://arxiv.org/pdf/2512.10235v1)**

> **作者:** Hui Li; Akhlak Uz Zaman; Fujian Yan; Hongsheng He
>
> **摘要:** This paper presents a reinforcement learning framework that incorporates a Contextual Reward Machine for task-oriented grasping. The Contextual Reward Machine reduces task complexity by decomposing grasping tasks into manageable sub-tasks. Each sub-task is associated with a stage-specific context, including a reward function, an action space, and a state abstraction function. This contextual information enables efficient intra-stage guidance and improves learning efficiency by reducing the state-action space and guiding exploration within clearly defined boundaries. In addition, transition rewards are introduced to encourage or penalize transitions between stages which guides the model toward desirable stage sequences and further accelerates convergence. When integrated with the Proximal Policy Optimization algorithm, the proposed method achieved a 95% success rate across 1,000 simulated grasping tasks encompassing diverse objects, affordances, and grasp topologies. It outperformed the state-of-the-art methods in both learning speed and success rate. The approach was transferred to a real robot, where it achieved a success rate of 83.3% in 60 grasping tasks over six affordances. These experimental results demonstrate superior accuracy, data efficiency, and learning efficiency. They underscore the model's potential to advance task-oriented grasping in both simulated and real-world settings.
>
---
#### [new 005] AERMANI-Diffusion: Regime-Conditioned Diffusion for Dynamics Learning in Aerial Manipulators
- **分类: cs.RO**

- **简介: 该论文针对无人机机械臂动力学建模难题，提出一种基于条件扩散的框架，通过轻量时序编码器捕捉运动状态，建模残差力分布，实现跨工况的高精度动态预测，提升控制性能。**

- **链接: [https://arxiv.org/pdf/2512.10773v1](https://arxiv.org/pdf/2512.10773v1)**

> **作者:** Samaksh Ujjawal; Shivansh Pratap Singh; Naveen Sudheer Nair; Rishabh Dev Yadav; Wei Pan; Spandan Roy
>
> **摘要:** Aerial manipulators undergo rapid, configuration-dependent changes in inertial coupling forces and aerodynamic forces, making accurate dynamics modeling a core challenge for reliable control. Analytical models lose fidelity under these nonlinear and nonstationary effects, while standard data-driven methods such as deep neural networks and Gaussian processes cannot represent the diverse residual behaviors that arise across different operating conditions. We propose a regime-conditioned diffusion framework that models the full distribution of residual forces using a conditional diffusion process and a lightweight temporal encoder. The encoder extracts a compact summary of recent motion and configuration, enabling consistent residual predictions even through abrupt transitions or unseen payloads. When combined with an adaptive controller, the framework enables dynamics uncertainty compensation and yields markedly improved tracking accuracy in real-world tests.
>
---
#### [new 006] Curriculum-Based Reinforcement Learning for Autonomous UAV Navigation in Unknown Curved Tubular Conduit
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究无人机在未知弯曲管道中的自主导航任务，解决缺乏先验几何信息和感知受限下的导航难题。提出基于课程学习的强化学习方法，利用LiDAR与视觉线索实现稳定飞行，实验表明其优于依赖中心线先验的纯追踪算法。**

- **链接: [https://arxiv.org/pdf/2512.10934v1](https://arxiv.org/pdf/2512.10934v1)**

> **作者:** Zamirddine Mari; Jérôme Pasquet; Julien Seinturier
>
> **摘要:** Autonomous drone navigation in confined tubular environments remains a major challenge due to the constraining geometry of the conduits, the proximity of the walls, and the perceptual limitations inherent to such scenarios. We propose a reinforcement learning approach enabling a drone to navigate unknown three-dimensional tubes without any prior knowledge of their geometry, relying solely on local observations from LiDAR and a conditional visual detection of the tube center. In contrast, the Pure Pursuit algorithm, used as a deterministic baseline, benefits from explicit access to the centerline, creating an information asymmetry designed to assess the ability of RL to compensate for the absence of a geometric model. The agent is trained through a progressive Curriculum Learning strategy that gradually exposes it to increasingly curved geometries, where the tube center frequently disappears from the visual field. A turning-negotiation mechanism, based on the combination of direct visibility, directional memory, and LiDAR symmetry cues, proves essential for ensuring stable navigation under such partial observability conditions. Experiments show that the PPO policy acquires robust and generalizable behavior, consistently outperforming the deterministic controller despite its limited access to geometric information. Validation in a high-fidelity 3D environment further confirms the transferability of the learned behavior to a continuous physical dynamics. The proposed approach thus provides a complete framework for autonomous navigation in unknown tubular environments and opens perspectives for industrial, underground, or medical applications where progressing through narrow and weakly perceptive conduits represents a central challenge.
>
---
#### [new 007] RoboNeuron: A Modular Framework Linking Foundation Models and ROS for Embodied AI
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出RoboNeuron，旨在解决 embodied AI 中跨场景适应性差、模块耦合紧密和推理加速碎片化问题。通过融合大模型与ROS，利用MCP协议实现感知、推理、控制解耦，提升系统模块化与通用性。**

- **链接: [https://arxiv.org/pdf/2512.10394v1](https://arxiv.org/pdf/2512.10394v1)**

> **作者:** Weifan Guan; Huasen Xi; Chenxiao Zhang; Aosheng Li; Qinghao Hu; Jian Cheng
>
> **摘要:** Current embodied AI systems face severe engineering impediments, primarily characterized by poor cross-scenario adaptability, rigid inter-module coupling, and fragmented inference acceleration. To overcome these limitations, we propose RoboNeuron, a universal deployment framework for embodied intelligence. RoboNeuron is the first framework to deeply integrate the cognitive capabilities of Large Language Models (LLMs) and Vision-Language-Action (VLA) models with the real-time execution backbone of the Robot Operating System (ROS). We utilize the Model Context Protocol (MCP) as a semantic bridge, enabling the LLM to dynamically orchestrate underlying robotic tools. The framework establishes a highly modular architecture that strictly decouples sensing, reasoning, and control by leveraging ROS's unified communication interfaces. Crucially, we introduce an automated tool to translate ROS messages into callable MCP functions, significantly streamlining development. RoboNeuron significantly enhances cross-scenario adaptability and component flexibility, while establishing a systematic platform for horizontal performance benchmarking, laying a robust foundation for scalable real-world embodied applications.
>
---
#### [new 008] ImplicitRDP: An End-to-End Visual-Force Diffusion Policy with Structural Slow-Fast Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究接触丰富的机器人操作任务，解决视觉与力觉信号融合难、模态失衡问题。提出ImplicitRDP方法，通过结构化慢-快学习和虚拟目标正则化，实现端到端视觉-力觉扩散策略，提升反应性与成功率。**

- **链接: [https://arxiv.org/pdf/2512.10946v1](https://arxiv.org/pdf/2512.10946v1)**

> **作者:** Wendi Chen; Han Xue; Yi Wang; Fangyuan Zhou; Jun Lv; Yang Jin; Shirun Tang; Chuan Wen; Cewu Lu
>
> **备注:** Project page: https://implicit-rdp.github.io
>
> **摘要:** Human-level contact-rich manipulation relies on the distinct roles of two key modalities: vision provides spatially rich but temporally slow global context, while force sensing captures rapid, high-frequency local contact dynamics. Integrating these signals is challenging due to their fundamental frequency and informational disparities. In this work, we propose ImplicitRDP, a unified end-to-end visual-force diffusion policy that integrates visual planning and reactive force control within a single network. We introduce Structural Slow-Fast Learning, a mechanism utilizing causal attention to simultaneously process asynchronous visual and force tokens, allowing the policy to perform closed-loop adjustments at the force frequency while maintaining the temporal coherence of action chunks. Furthermore, to mitigate modality collapse where end-to-end models fail to adjust the weights across different modalities, we propose Virtual-target-based Representation Regularization. This auxiliary objective maps force feedback into the same space as the action, providing a stronger, physics-grounded learning signal than raw force prediction. Extensive experiments on contact-rich tasks demonstrate that ImplicitRDP significantly outperforms both vision-only and hierarchical baselines, achieving superior reactivity and success rates with a streamlined training pipeline. Code and videos will be publicly available at https://implicit-rdp.github.io.
>
---
#### [new 009] Motion Planning for Safe Landing of a Human-Piloted Parafoil
- **分类: cs.RO**

- **简介: 该论文研究人控降落伞安全着陆路径规划，旨在减少人为操作失误导致的事故。通过改进SST算法生成低控制成本的安全轨迹，并与人类飞行数据对比，验证算法在安全性与效率上的优势，为训练模拟器提供技术基础。**

- **链接: [https://arxiv.org/pdf/2512.10595v1](https://arxiv.org/pdf/2512.10595v1)**

> **作者:** Maximillian Fainkich; Kiril Solovey; Anna Clarke
>
> **摘要:** Most skydiving accidents occur during the parafoil-piloting and landing stages and result from human lapses in judgment while piloting the parafoil. Training of novice pilots is protracted due to the lack of functional and easily accessible training simulators. Moreover, work on parafoil trajectory planning suitable for aiding human training remains limited. To bridge this gap, we study the problem of computing safe trajectories for human-piloted parafoil flight and examine how such trajectories fare against human-generated solutions. For the algorithmic part, we adapt the sampling-based motion planner Stable Sparse RRT (SST) by Li et al., to cope with the problem constraints while minimizing the bank angle (control effort) as a proxy for safety. We then compare the computer-generated solutions with data from human-generated parafoil flight, where the algorithm offers a relative cost improvement of 20\%-80\% over the performance of the human pilot. We observe that human pilots tend to, first, close the horizontal distance to the landing area, and then address the vertical gap by spiraling down to the suitable altitude for starting a landing maneuver. The algorithm considered here makes smoother and more gradual descents, arriving at the landing area at the precise altitude necessary for the final approach while maintaining safety constraints. Overall, the study demonstrates the potential of computer-generated guidelines, rather than traditional rules of thumb, which can be integrated into future simulators to train pilots for safer and more cost-effective flights.
>
---
#### [new 010] Design and Validation of an Under-actuated Robotic Finger with Synchronous Tendon Routing
- **分类: cs.RO**

- **简介: 该论文设计并验证了一种同步肌腱传动的欠驱动机器人手指，旨在以单驱动器实现高负载与自适应柔顺性。通过建立考虑肌腱弹性的模型，实现了刚度预测与可靠抓取，解决了多指手结构复杂与驱动冗余问题。**

- **链接: [https://arxiv.org/pdf/2512.10349v1](https://arxiv.org/pdf/2512.10349v1)**

> **作者:** Quan Yuan; Zhenting Du; Daqian Cao; Weibang Bai
>
> **备注:** 7 pages and 11 figures
>
> **摘要:** Tendon-driven under-actuated robotic fingers provide advantages for dexterous manipulation through reduced actuator requirements and simplified mechanical design. However, achieving both high load capacity and adaptive compliance in a compact form remains challenging. This paper presents an under-actuated tendon-driven robotic finger (UTRF) featuring a synchronous tendon routing that mechanically couples all joints with fixed angular velocity ratios, enabling the entire finger to be actuated by a single actuator. This approach significantly reduces the number of actuators required in multi-finger hands, resulting in a lighter and more compact structure without sacrificing stiffness or compliance. The kinematic and static models of the finger are derived, incorporating tendon elasticity to predict structural stiffness. A single-finger prototype was fabricated and tested under static loading, showing an average deflection prediction error of 1.0 mm (0.322% of total finger length) and a measured stiffness of 1.2x10^3 N/m under a 3 kg tip load. Integration into a five-finger robotic hand (UTRF-RoboHand) demonstrates effective object manipulation across diverse scenarios, confirming that the proposed routing achieves predictable stiffness and reliable grasping performance with a minimal actuator count.
>
---
#### [new 011] Symphony: A Heuristic Normalized Calibrated Advantage Actor and Critic Algorithm in application for Humanoid Robots
- **分类: cs.RO; cs.NE**

- **简介: 该论文研究 humanoid 机器人从零开始的强化学习控制任务，旨在解决样本效率、动作安全与训练稳定性问题。提出 Symphony 算法，结合确定性策略、弱噪声探索、Swaddling 正则化与 Fading 回放缓冲，提升学习安全性与效率。**

- **链接: [https://arxiv.org/pdf/2512.10477v1](https://arxiv.org/pdf/2512.10477v1)**

> **作者:** Timur Ishuov; Michele Folgheraiter; Madi Nurmanov; Goncalo Gordo; Richárd Farkas; József Dombi
>
> **摘要:** In our work we not explicitly hint that it is a misconception to think that humans learn fast. Learning process takes time. Babies start learning to move in the restricted liquid area called placenta. Children often are limited by underdeveloped body. Even adults are not allowed to participate in complex competitions right away. However, with robots, when learning from scratch, we often don't have the privilege of waiting for dozen millions of steps. "Swaddling" regularization is responsible for restraining an agent in rapid but unstable development penalizing action strength in a specific way not affecting actions directly. The Symphony, Transitional-policy Deterministic Actor and Critic algorithm, is a concise combination of different ideas for possibility of training humanoid robots from scratch with Sample Efficiency, Sample Proximity and Safety of Actions in mind. It is no secret that continuous increase in Gaussian noise without appropriate smoothing is harmful for motors and gearboxes. Compared to Stochastic algorithms, we set a limited parametric noise and promote a reduced strength of actions, safely increasing entropy, since the actions are kind of immersed in weaker noise. When actions require more extreme values, actions rise above the weak noise. Training becomes empirically much safer for both the environment around and the robot's mechanisms. We use Fading Replay Buffer: using a fixed formula containing the hyperbolic tangent, we adjust the batch sampling probability: the memory contains a recent memory and a long-term memory trail. Fading Replay Buffer allows us to use Temporal Advantage when we improve the current Critic Network prediction compared to the exponential moving average. Temporal Advantage allows us to update Actor and Critic in one pass, as well as combine Actor and Critic in one Object and implement their Losses in one line.
>
---
#### [new 012] CLASH: Collaborative Large-Small Hierarchical Framework for Continuous Vision-and-Language Navigation
- **分类: cs.RO**

- **简介: 该论文研究视觉-语言导航（VLN）任务，旨在解决大模型推理强但执行弱、小模型反之的问题。提出CLASH框架，融合大模型的推理与小模型的规划，并引入协作机制和可学习控制器，在仿真与真实场景中均取得优异性能。**

- **链接: [https://arxiv.org/pdf/2512.10360v1](https://arxiv.org/pdf/2512.10360v1)**

> **作者:** Liuyi Wang; Zongtao He; Jinlong Li; Xiaoyan Qi; Mengxian Hu; Chenpeng Yao; Chengju Liu; Qijun Chen
>
> **摘要:** Vision-and-Language Navigation (VLN) requires robots to follow natural language instructions and navigate complex environments without prior maps. While recent vision-language large models demonstrate strong reasoning abilities, they often underperform task-specific panoramic small models in VLN tasks. To address this, we propose CLASH (Collaborative Large-Small Hierarchy), a VLN-CE framework that integrates a reactive small-model planner (RSMP) with a reflective large-model reasoner (RLMR). RSMP adopts a causal-learning-based dual-branch architecture to enhance generalization, while RLMR leverages panoramic visual prompting with chain-of-thought reasoning to support interpretable spatial understanding and navigation. We further introduce an uncertainty-aware collaboration mechanism (UCM) that adaptively fuses decisions from both models. For obstacle avoidance, in simulation, we replace the rule-based controller with a fully learnable point-goal policy, and in real-world deployment, we design a LiDAR-based clustering module for generating navigable waypoints and pair it with an online SLAM-based local controller. CLASH achieves state-of-the-art (SoTA) results (ranking 1-st) on the VLN-CE leaderboard, significantly improving SR and SPL on the test-unseen set over the previous SoTA methods. Real-world experiments demonstrate CLASH's strong robustness, validating its effectiveness in both simulation and deployment scenarios.
>
---
#### [new 013] Neural Ranging Inertial Odometry
- **分类: cs.RO**

- **简介: 该论文研究GPS拒止环境下的定位任务，解决UWB定位受多径干扰和传感器布局影响的问题。提出一种神经融合框架，结合图注意力与循环网络，实现无需校准的高精度惯性里程计，适应任意锚点数量，并在多种环境中验证了其鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.10531v1](https://arxiv.org/pdf/2512.10531v1)**

> **作者:** Si Wang; Bingqi Shen; Fei Wang; Yanjun Cao; Rong Xiong; Yue Wang
>
> **备注:** Accepted by 2025 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Ultra-wideband (UWB) has shown promising potential in GPS-denied localization thanks to its lightweight and drift-free characteristics, while the accuracy is limited in real scenarios due to its sensitivity to sensor arrangement and non-Gaussian pattern induced by multi-path or multi-signal interference, which commonly occurs in many typical applications like long tunnels. We introduce a novel neural fusion framework for ranging inertial odometry which involves a graph attention UWB network and a recurrent neural inertial network. Our graph net learns scene-relevant ranging patterns and adapts to any number of anchors or tags, realizing accurate positioning without calibration. Additionally, the integration of least squares and the incorporation of nominal frame enhance overall performance and scalability. The effectiveness and robustness of our methods are validated through extensive experiments on both public and self-collected datasets, spanning indoor, outdoor, and tunnel environments. The results demonstrate the superiority of our proposed IR-ULSG in handling challenging conditions, including scenarios outside the convex envelope and cases where only a single anchor is available.
>
---
#### [new 014] Seamless Outdoor-Indoor Pedestrian Positioning System with GNSS/UWB/IMU Fusion: A Comparison of EKF, FGO, and PF
- **分类: cs.RO**

- **简介: 该论文研究无缝室内外行人定位，解决GNSS、UWB和IMU各自在信号遮挡、多径和漂移下的局限性。提出融合GNSS/UWB/IMU的统一框架，比较EKF、FGO和PF三种后端，并引入轻量地图约束提升过渡鲁棒性，实现穿戴设备上的实时定位。**

- **链接: [https://arxiv.org/pdf/2512.10480v1](https://arxiv.org/pdf/2512.10480v1)**

> **作者:** Jiaqiang Zhang; Xianjia Yu; Sier Ha; Paola Torrico Moron; Sahar Salimpour; Farhad Kerama; Haizhou Zhang; Tomi Westerlund
>
> **备注:** 8 pages, 4 figures, submitted to The 17th International Conference on Ambient Systems, Networks and Technologies
>
> **摘要:** Accurate and continuous pedestrian positioning across outdoor-indoor environments remains challenging because GNSS, UWB, and inertial PDR are complementary yet individually fragile under signal blockage, multipath, and drift. This paper presents a unified GNSS/UWB/IMU fusion framework for seamless pedestrian localization and provides a controlled comparison of three probabilistic back-ends: an error-state extended Kalman filter, sliding-window factor graph optimization, and a particle filter. The system uses chest-mounted IMU-based PDR as the motion backbone and integrates absolute updates from GNSS outdoors and UWB indoors. To enhance transition robustness and mitigate urban GNSS degradation, we introduce a lightweight map-based feasibility constraint derived from OpenStreetMap building footprints, treating most building interiors as non-navigable while allowing motion inside a designated UWB-instrumented building. The framework is implemented in ROS 2 and runs in real time on a wearable platform, with visualization in Foxglove. We evaluate three scenarios: indoor (UWB+PDR), outdoor (GNSS+PDR), and seamless outdoor-indoor (GNSS+UWB+PDR). Results show that the ESKF provides the most consistent overall performance in our implementation.
>
---
#### [new 015] On the Stabilization of Rigid Formations on Regular Curves
- **分类: cs.RO**

- **简介: 该论文研究多智能体刚性编队在平面曲线上的稳定控制问题。针对闭合可微曲线，提出随机多起点牛顿类算法搜索内接正多边形，并设计连续反馈律实现编队沿曲线扫掠与定点收敛，确保避障。通过数值仿真验证方法有效性。**

- **链接: [https://arxiv.org/pdf/2512.10700v1](https://arxiv.org/pdf/2512.10700v1)**

> **作者:** Mohamed Elobaid; Shinkyu Park; Eric Feron
>
> **摘要:** This work deals with the problem of stabilizing a multi-agent rigid formation on a general class of planar curves. Namely, we seek to stabilize an equilateral polygonal formation on closed planar differentiable curves after a path sweep. The task of finding an inscribed regular polygon centered at the point of interest is solved via a randomized multi-start Newton-Like algorithm for which one is able to ascertain the existence of a minimizer. Then we design a continuous feedback law that guarantees convergence to, and sufficient sweeping of the curve, followed by convergence to the desired formation vertices while ensuring inter-agent avoidance. The proposed approach is validated through numerical simulations for different classes of curves and different rigid formations. Code: https://github.com/mebbaid/paper-elobaid-ifacwc-2026
>
---
#### [new 016] Design of a six wheel suspension and a three-axis linear actuation mechanism for a laser weeding robot
- **分类: cs.RO; cs.CV; eess.SY**

- **简介: 该论文设计了一种六轮激光除草机器人，旨在解决传统除草方式效率低或破坏生态的问题。通过双四杆悬挂提升越障能力，三轴线性驱动机构精确引导激光，实现高效、精准的非化学除草。**

- **链接: [https://arxiv.org/pdf/2512.10319v1](https://arxiv.org/pdf/2512.10319v1)**

> **作者:** Muhammad Usama; Muhammad Ibrahim Khan; Ahmad Hasan; Muhammad Shaaf Nadeem; Khawaja Fahad Iqbal; Jawad Aslam; Mian Ashfaq Ali; Asad Nisar Awan
>
> **备注:** 15 Pages, 10 figures
>
> **摘要:** Mobile robots are increasingly utilized in agriculture to automate labor-intensive tasks such as weeding, sowing, harvesting and soil analysis. Recently, agricultural robots have been developed to detect and remove weeds using mechanical tools or precise herbicide sprays. Mechanical weeding is inefficient over large fields, and herbicides harm the soil ecosystem. Laser weeding with mobile robots has emerged as a sustainable alternative in precision farming. In this paper, we present an autonomous weeding robot that uses controlled exposure to a low energy laser beam for weed removal. The proposed robot is six-wheeled with a novel double four-bar suspension for higher stability. The laser is guided towards the detected weeds by a three-dimensional linear actuation mechanism. Field tests have demonstrated the robot's capability to navigate agricultural terrains effectively by overcoming obstacles up to 15 cm in height. At an optimal speed of 42.5 cm/s, the robot achieves a weed detection rate of 86.2\% and operating time of 87 seconds per meter. The laser actuation mechanism maintains a minimal mean positional error of 1.54 mm, combined with a high hit rate of 97\%, ensuring effective and accurate weed removal. This combination of speed, accuracy, and efficiency highlights the robot's potential for significantly enhancing precision farming practices.
>
---
#### [new 017] Push Smarter, Not Harder: Hierarchical RL-Diffusion Policy for Efficient Nonprehensile Manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究非抓取式操作中的推动物体任务，旨在解决复杂接触动力学与长视野规划难题。作者提出HeRD方法，结合强化学习与扩散模型，分层决策目标并生成高效轨迹，在2D仿真中实现了更高成功率与泛化性。**

- **链接: [https://arxiv.org/pdf/2512.10099v1](https://arxiv.org/pdf/2512.10099v1)**

> **作者:** Steven Caro; Stephen L. Smith
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** Nonprehensile manipulation, such as pushing objects across cluttered environments, presents a challenging control problem due to complex contact dynamics and long-horizon planning requirements. In this work, we propose HeRD, a hierarchical reinforcement learning-diffusion policy that decomposes pushing tasks into two levels: high-level goal selection and low-level trajectory generation. We employ a high-level reinforcement learning (RL) agent to select intermediate spatial goals, and a low-level goal-conditioned diffusion model to generate feasible, efficient trajectories to reach them. This architecture combines the long-term reward maximizing behaviour of RL with the generative capabilities of diffusion models. We evaluate our method in a 2D simulation environment and show that it outperforms the state-of-the-art baseline in success rate, path efficiency, and generalization across multiple environment configurations. Our results suggest that hierarchical control with generative low-level planning is a promising direction for scalable, goal-directed nonprehensile manipulation. Code, documentation, and trained models are available: https://github.com/carosteven/HeRD.
>
---
#### [new 018] Evaluating Gemini Robotics Policies in a Veo World Simulator
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于机器人策略评估任务，旨在解决现有视频模型仅限于分布内评估的问题。作者基于Veo视频模型构建生成式评估系统，支持动作条件、多视角一致性与场景编辑，实现对机器人策略在常态、分布外及安全性的全面仿真评估，并通过真实实验验证有效性。**

- **链接: [https://arxiv.org/pdf/2512.10675v1](https://arxiv.org/pdf/2512.10675v1)**

> **作者:** Gemini Robotics Team; Coline Devin; Yilun Du; Debidatta Dwibedi; Ruiqi Gao; Abhishek Jindal; Thomas Kipf; Sean Kirmani; Fangchen Liu; Anirudha Majumdar; Andrew Marmon; Carolina Parada; Yulia Rubanova; Dhruv Shah; Vikas Sindhwani; Jie Tan; Fei Xia; Ted Xiao; Sherry Yang; Wenhao Yu; Allan Zhou
>
> **摘要:** Generative world models hold significant potential for simulating interactions with visuomotor policies in varied environments. Frontier video models can enable generation of realistic observations and environment interactions in a scalable and general manner. However, the use of video models in robotics has been limited primarily to in-distribution evaluations, i.e., scenarios that are similar to ones used to train the policy or fine-tune the base video model. In this report, we demonstrate that video models can be used for the entire spectrum of policy evaluation use cases in robotics: from assessing nominal performance to out-of-distribution (OOD) generalization, and probing physical and semantic safety. We introduce a generative evaluation system built upon a frontier video foundation model (Veo). The system is optimized to support robot action conditioning and multi-view consistency, while integrating generative image-editing and multi-view completion to synthesize realistic variations of real-world scenes along multiple axes of generalization. We demonstrate that the system preserves the base capabilities of the video model to enable accurate simulation of scenes that have been edited to include novel interaction objects, novel visual backgrounds, and novel distractor objects. This fidelity enables accurately predicting the relative performance of different policies in both nominal and OOD conditions, determining the relative impact of different axes of generalization on policy performance, and performing red teaming of policies to expose behaviors that violate physical or semantic safety constraints. We validate these capabilities through 1600+ real-world evaluations of eight Gemini Robotics policy checkpoints and five tasks for a bimanual manipulator.
>
---
#### [new 019] Openpi Comet: Competition Solution For 2025 BEHAVIOR Challenge
- **分类: cs.RO**

- **简介: 该论文针对2025 BEHAVIOR挑战赛，解决机器人在仿真环境中执行长视野家庭任务的移动操作问题。基于π₀.₅模型，通过系统优化训练方法与数据，提升预训练与后训练的扩展性，最终取得第二名并超越其余方案。**

- **链接: [https://arxiv.org/pdf/2512.10071v1](https://arxiv.org/pdf/2512.10071v1)**

> **作者:** Junjie Bai; Yu-Wei Chao; Qizhi Chen; Jinwei Gu; Moo Jin Kim; Zhaoshuo Li; Xuan Li; Tsung-Yi Lin; Ming-Yu Liu; Nic Ma; Kaichun Mo; Delin Qu; Shangkun Sun; Hongchi Xia; Fangyin Wei; Xiaohui Zeng
>
> **备注:** preprint
>
> **摘要:** The 2025 BEHAVIOR Challenge is designed to rigorously track progress toward solving long-horizon tasks by physical agents in simulated environments. BEHAVIOR-1K focuses on everyday household tasks that people most want robots to assist with and these tasks introduce long-horizon mobile manipulation challenges in realistic settings, bridging the gap between current research and real-world, human-centric applications. This report presents our solution to the 2025 BEHAVIOR Challenge in a very close 2nd place and substantially outperforms the rest of the submissions. Building on $π_{0.5}$, we focus on systematically building our solution by studying the effects of training techniques and data. Through careful ablations, we show the scaling power in pre-training and post-training phases for competitive performance. We summarize our practical lessons and design recommendations that we hope will provide actionable insights for the broader embodied AI community when adapting powerful foundation models to complex embodied scenarios.
>
---
#### [new 020] How to Brake? Ethical Emergency Braking with Deep Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究自动驾驶车辆在紧急制动时的伦理决策问题，旨在通过结合深度强化学习与解析方法，提升多车协同下的整体安全性和伤害最小化能力，解决传统保守策略灵活性不足的问题。**

- **链接: [https://arxiv.org/pdf/2512.10698v1](https://arxiv.org/pdf/2512.10698v1)**

> **作者:** Jianbo Wang; Galina Sidorenko; Johan Thunberg
>
> **摘要:** Connected and automated vehicles (CAVs) have the potential to enhance driving safety, for example by enabling safe vehicle following and more efficient traffic scheduling. For such future deployments, safety requirements should be addressed, where the primary such are avoidance of vehicle collisions and substantial mitigating of harm when collisions are unavoidable. However, conservative worst-case-based control strategies come at the price of reduced flexibility and may compromise overall performance. In light of this, we investigate how Deep Reinforcement Learning (DRL) can be leveraged to improve safety in multi-vehicle-following scenarios involving emergency braking. Specifically, we investigate how DRL with vehicle-to-vehicle communication can be used to ethically select an emergency breaking profile in scenarios where overall, or collective, three-vehicle harm reduction or collision avoidance shall be obtained instead of single-vehicle such. As an algorithm, we provide a hybrid approach that combines DRL with a previously published method based on analytical expressions for selecting optimal constant deceleration. By combining DRL with the previous method, the proposed hybrid approach increases the reliability compared to standalone DRL, while achieving superior performance in terms of overall harm reduction and collision avoidance.
>
---
#### [new 021] Iterative Compositional Data Generation for Robot Control
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究机器人控制中的数据生成任务，旨在解决多任务组合下数据昂贵且难泛化的难题。提出语义组合扩散Transformer，分解状态转移并利用注意力学习组件交互，结合迭代自优化提升零样本生成质量，实现对未见任务的高效策略学习。**

- **链接: [https://arxiv.org/pdf/2512.10891v1](https://arxiv.org/pdf/2512.10891v1)**

> **作者:** Anh-Quan Pham; Marcel Hussing; Shubhankar P. Patankar; Dani S. Bassett; Jorge Mendez-Mendez; Eric Eaton
>
> **摘要:** Collecting robotic manipulation data is expensive, making it impractical to acquire demonstrations for the combinatorially large space of tasks that arise in multi-object, multi-robot, and multi-environment settings. While recent generative models can synthesize useful data for individual tasks, they do not exploit the compositional structure of robotic domains and struggle to generalize to unseen task combinations. We propose a semantic compositional diffusion transformer that factorizes transitions into robot-, object-, obstacle-, and objective-specific components and learns their interactions through attention. Once trained on a limited subset of tasks, we show that our model can zero-shot generate high-quality transitions from which we can learn control policies for unseen task combinations. Then, we introduce an iterative self-improvement procedure in which synthetic data is validated via offline reinforcement learning and incorporated into subsequent training rounds. Our approach substantially improves zero-shot performance over monolithic and hard-coded compositional baselines, ultimately solving nearly all held-out tasks and demonstrating the emergence of meaningful compositional structure in the learned representations.
>
---
#### [new 022] Lies We Can Trust: Quantifying Action Uncertainty with Inaccurate Stochastic Dynamics through Conformalized Nonholonomic Lie Groups
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究机器人动作不确定性量化，提出CLAPS算法，利用李群结构与共形预测，在无需强假设下为非欧系统（如SE(2)）构建可靠预测集，提升安全控制中的置信度与效率。**

- **链接: [https://arxiv.org/pdf/2512.10294v1](https://arxiv.org/pdf/2512.10294v1)**

> **作者:** Luís Marques; Maani Ghaffari; Dmitry Berenson
>
> **备注:** 13 pages, 7 figures. Under review
>
> **摘要:** We propose Conformal Lie-group Action Prediction Sets (CLAPS), a symmetry-aware conformal prediction-based algorithm that constructs, for a given action, a set guaranteed to contain the resulting system configuration at a user-defined probability. Our assurance holds under both aleatoric and epistemic uncertainty, non-asymptotically, and does not require strong assumptions about the true system dynamics, the uncertainty sources, or the quality of the approximate dynamics model. Typically, uncertainty quantification is tackled by making strong assumptions about the error distribution or magnitude, or by relying on uncalibrated uncertainty estimates - i.e., with no link to frequentist probabilities - which are insufficient for safe control. Recently, conformal prediction has emerged as a statistical framework capable of providing distribution-free probabilistic guarantees on test-time prediction accuracy. While current conformal methods treat robots as Euclidean points, many systems have non-Euclidean configurations, e.g., some mobile robots have SE(2). In this work, we rigorously analyze configuration errors using Lie groups, extending previous Euclidean Space theoretical guarantees to SE(2). Our experiments on a simulated JetBot, and on a real MBot, suggest that by considering the configuration space's structure, our symmetry-informed nonconformity score leads to more volume-efficient prediction regions which represent the underlying uncertainty better than existing approaches.
>
---
#### [new 023] Inertial Magnetic SLAM Systems Using Low-Cost Sensors
- **分类: cs.RO; eess.SP**

- **简介: 该论文研究惯性-磁性SLAM任务，旨在解决低-cost传感器下定位误差大的问题。提出松耦合和紧耦合两种IM-SLAM系统，利用IMU、磁力计和气压计实现3D定位，实验证明紧耦合方法精度更高，适用于救援等弱光环境。**

- **链接: [https://arxiv.org/pdf/2512.10128v1](https://arxiv.org/pdf/2512.10128v1)**

> **作者:** Chuan Huang; Gustaf Hendeby; Isaac Skog
>
> **摘要:** Spatially inhomogeneous magnetic fields offer a valuable, non-visual information source for positioning. Among systems leveraging this, magnetic field-based simultaneous localization and mapping (SLAM) systems are particularly attractive because they can provide positioning information and build a magnetic field map on the fly. Moreover, they have bounded error within mapped regions. However, state-of-the-art methods typically require low-drift odometry data provided by visual odometry or a wheel encoder, etc. This is because these systems need to minimize/reduce positioning errors while exploring, which happens when they are in unmapped regions. To address these limitations, this work proposes a loosely coupled and a tightly coupled inertial magnetic SLAM (IM-SLAM) system. The proposed systems use commonly available low-cost sensors: an inertial measurement unit (IMU), a magnetometer array, and a barometer. The use of non-visual data provides a significant advantage over visual-based systems, making it robust to low-visibility conditions. Both systems employ state-space representations, and magnetic field models on different scales. The difference lies in how they use a local and global magnetic field model. The loosely coupled system uses these models separately in two state-space models, while the tightly coupled system integrates them into one state-space model. Experiment results show that the tightly coupled IM-SLAM system achieves lower positioning errors than the loosely coupled system in most scenarios, with typical errors on the order of meters per 100 meters traveled. These results demonstrate the feasiblity of developing a full 3D IM-SLAM systems using low-cost sensors and the potential of applying these systems in emergency response scenarios such as mine/fire rescue.
>
---
#### [new 024] Fast Functionally Redundant Inverse Kinematics for Robotic Toolpath Optimisation in Manufacturing Tasks
- **分类: cs.RO**

- **简介: 该论文针对六轴机器人在制造任务中的功能冗余问题，提出一种快速求解冗余逆运动学的新算法，通过任务空间分解与优化方法，实现工具路径的高效优化，并在冷喷涂工艺中验证了其有效性。**

- **链接: [https://arxiv.org/pdf/2512.10116v1](https://arxiv.org/pdf/2512.10116v1)**

> **作者:** Andrew Razjigaev; Hans Lohr; Alejandro Vargas-Uscategui; Peter King; Tirthankar Bandyopadhyay
>
> **备注:** Published at the Australasian Conference on Robotics and Automation (ACRA 2025) https://ssl.linklings.net/conferences/acra/acra2025_proceedings/views/includes/files/pap149s2.pdf
>
> **摘要:** Industrial automation with six-axis robotic arms is critical for many manufacturing tasks, including welding and additive manufacturing applications; however, many of these operations are functionally redundant due to the symmetrical tool axis, which effectively makes the operation a five-axis task. Exploiting this redundancy is crucial for achieving the desired workspace and dexterity required for the feasibility and optimisation of toolpath planning. Inverse kinematics algorithms can solve this in a fast, reactive framework, but these techniques are underutilised over the more computationally expensive offline planning methods. We propose a novel algorithm to solve functionally redundant inverse kinematics for robotic manipulation utilising a task space decomposition approach, the damped least-squares method and Halley's method to achieve fast and robust solutions with reduced joint motion. We evaluate our methodology in the case of toolpath optimisation in a cold spray coating application on a non-planar surface. The functionally redundant inverse kinematics algorithm can quickly solve motion plans that minimise joint motion, expanding the feasible operating space of the complex toolpath. We validate our approach on an industrial ABB manipulator and cold-spray gun executing the computed toolpath.
>
---
#### [new 025] Distribution-Free Stochastic MPC for Joint-in-Time Chance-Constrained Linear Systems
- **分类: eess.SY; cs.RO**

- **简介: 该论文研究数据驱动的随机模型预测控制，解决未知分布噪声下线性系统的联合概率约束控制问题。提出基于共形预测的方法构建误差置信域，实现无需分布假设的实时约束满足与递归可行性，适用于状态和输出反馈情形。**

- **链接: [https://arxiv.org/pdf/2512.10738v1](https://arxiv.org/pdf/2512.10738v1)**

> **作者:** Lukas Vogel; Andrea Carron; Eleftherios E. Vlahakis; Dimos V. Dimarogonas
>
> **摘要:** This work presents a stochastic model predictive control (MPC) framework for linear systems subject to joint-in-time chance constraints under unknown disturbance distributions. Unlike existing stochastic MPC formulations that rely on parametric or Gaussian assumptions or require expensive offline computations, the proposed method leverages conformal prediction (CP) as a streamlined tool to construct finite-sample confidence regions for the system's stochastic error trajectories with minimal computational effort. These regions enable the relaxation of probabilistic constraints while providing formal guarantees. By employing an indirect feedback mechanism and a probabilistic set-based formulation, we prove recursive feasibility of the relaxed optimization problem and establish chance constraint satisfaction in closed-loop. Furthermore, we extend the approach to the more general output feedback setting with unknown measurement noise distributions. Given available noise samples, we establish satisfaction of the joint chance constraints and recursive feasibility via output measurements alone. Numerical examples demonstrate the effectiveness and advantages of the proposed method compared to existing approaches.
>
---
#### [new 026] Latent Chain-of-Thought World Modeling for End-to-End Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究端到端自动驾驶中的推理建模，提出Latent-CoT-Drive模型。它用动作对齐的隐空间语言替代自然语言链式思维，联合建模动作生成与环境预测，通过监督预训练和强化学习提升推理效率与驾驶性能。**

- **链接: [https://arxiv.org/pdf/2512.10226v1](https://arxiv.org/pdf/2512.10226v1)**

> **作者:** Shuhan Tan; Kashyap Chitta; Yuxiao Chen; Ran Tian; Yurong You; Yan Wang; Wenjie Luo; Yulong Cao; Philipp Krahenbuhl; Marco Pavone; Boris Ivanovic
>
> **备注:** Technical Report
>
> **摘要:** Recent Vision-Language-Action (VLA) models for autonomous driving explore inference-time reasoning as a way to improve driving performance and safety in challenging scenarios. Most prior work uses natural language to express chain-of-thought (CoT) reasoning before producing driving actions. However, text may not be the most efficient representation for reasoning. In this work, we present Latent-CoT-Drive (LCDrive): a model that expresses CoT in a latent language that captures possible outcomes of the driving actions being considered. Our approach unifies CoT reasoning and decision making by representing both in an action-aligned latent space. Instead of natural language, the model reasons by interleaving (1) action-proposal tokens, which use the same vocabulary as the model's output actions; and (2) world model tokens, which are grounded in a learned latent world model and express future outcomes of these actions. We cold start latent CoT by supervising the model's action proposals and world model tokens based on ground-truth future rollouts of the scene. We then post-train with closed-loop reinforcement learning to strengthen reasoning capabilities. On a large-scale end-to-end driving benchmark, LCDrive achieves faster inference, better trajectory quality, and larger improvements from interactive reinforcement learning compared to both non-reasoning and text-reasoning baselines.
>
---
#### [new 027] Any4D: Unified Feed-Forward Metric 4D Reconstruction
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出Any4D，解决多视角下密集度量级4D重建问题。通过统一前馈架构和模块化4D场景表示，支持多模态输入，实现高效、高精度的像素级运动与几何预测，适用于多样传感器配置。**

- **链接: [https://arxiv.org/pdf/2512.10935v1](https://arxiv.org/pdf/2512.10935v1)**

> **作者:** Jay Karhade; Nikhil Keetha; Yuchen Zhang; Tanisha Gupta; Akash Sharma; Sebastian Scherer; Deva Ramanan
>
> **备注:** Project Website: https://any-4d.github.io/
>
> **摘要:** We present Any4D, a scalable multi-view transformer for metric-scale, dense feed-forward 4D reconstruction. Any4D directly generates per-pixel motion and geometry predictions for N frames, in contrast to prior work that typically focuses on either 2-view dense scene flow or sparse 3D point tracking. Moreover, unlike other recent methods for 4D reconstruction from monocular RGB videos, Any4D can process additional modalities and sensors such as RGB-D frames, IMU-based egomotion, and Radar Doppler measurements, when available. One of the key innovations that allows for such a flexible framework is a modular representation of a 4D scene; specifically, per-view 4D predictions are encoded using a variety of egocentric factors (depthmaps and camera intrinsics) represented in local camera coordinates, and allocentric factors (camera extrinsics and scene flow) represented in global world coordinates. We achieve superior performance across diverse setups - both in terms of accuracy (2-3X lower error) and compute efficiency (15X faster), opening avenues for multiple downstream applications.
>
---
#### [new 028] Digital Twin Supervised Reinforcement Learning Framework for Autonomous Underwater Navigation
- **分类: cs.LG; cs.RO**

- **简介: 该论文研究水下自主导航，针对无GPS、低能见度和障碍物问题，提出基于数字孪生监督的深度强化学习框架，采用PPO算法训练BlueROV2，在仿真与真实环境中验证其优于传统DWA方法的避障与导航能力。**

- **链接: [https://arxiv.org/pdf/2512.10925v1](https://arxiv.org/pdf/2512.10925v1)**

> **作者:** Zamirddine Mari; Mohamad Motasem Nawaf; Pierre Drap
>
> **摘要:** Autonomous navigation in underwater environments remains a major challenge due to the absence of GPS, degraded visibility, and the presence of submerged obstacles. This article investigates these issues through the case of the BlueROV2, an open platform widely used for scientific experimentation. We propose a deep reinforcement learning approach based on the Proximal Policy Optimization (PPO) algorithm, using an observation space that combines target-oriented navigation information, a virtual occupancy grid, and ray-casting along the boundaries of the operational area. The learned policy is compared against a reference deterministic kinematic planner, the Dynamic Window Approach (DWA), commonly employed as a robust baseline for obstacle avoidance. The evaluation is conducted in a realistic simulation environment and complemented by validation on a physical BlueROV2 supervised by a 3D digital twin of the test site, helping to reduce risks associated with real-world experimentation. The results show that the PPO policy consistently outperforms DWA in highly cluttered environments, notably thanks to better local adaptation and reduced collisions. Finally, the experiments demonstrate the transferability of the learned behavior from simulation to the real world, confirming the relevance of deep RL for autonomous navigation in underwater robotics.
>
---
#### [new 029] V-OCBF: Learning Safety Filters from Offline Data via Value-Guided Offline Control Barrier Functions
- **分类: cs.AI; cs.RO**

- **简介: 该论文研究安全控制任务，旨在解决离线强化学习中缺乏严格安全保证的问题。作者提出V-OCBF框架，通过值引导的离线控制屏障函数，从历史数据中学习神经屏障函数，实现无需在线交互或动力学模型的安全控制器合成。**

- **链接: [https://arxiv.org/pdf/2512.10822v1](https://arxiv.org/pdf/2512.10822v1)**

> **作者:** Mumuksh Tayal; Manan Tayal; Aditya Singh; Shishir Kolathaya; Ravi Prakash
>
> **备注:** 23 pages, 8 figure, 7 tables
>
> **摘要:** Ensuring safety in autonomous systems requires controllers that satisfy hard, state-wise constraints without relying on online interaction. While existing Safe Offline RL methods typically enforce soft expected-cost constraints, they do not guarantee forward invariance. Conversely, Control Barrier Functions (CBFs) provide rigorous safety guarantees but usually depend on expert-designed barrier functions or full knowledge of the system dynamics. We introduce Value-Guided Offline Control Barrier Functions (V-OCBF), a framework that learns a neural CBF entirely from offline demonstrations. Unlike prior approaches, V-OCBF does not assume access to the dynamics model; instead, it derives a recursive finite-difference barrier update, enabling model-free learning of a barrier that propagates safety information over time. Moreover, V-OCBF incorporates an expectile-based objective that avoids querying the barrier on out-of-distribution actions and restricts updates to the dataset-supported action set. The learned barrier is then used with a Quadratic Program (QP) formulation to synthesize real-time safe control. Across multiple case studies, V-OCBF yields substantially fewer safety violations than baseline methods while maintaining strong task performance, highlighting its scalability for offline synthesis of safety-critical controllers without online interaction or hand-engineered barriers.
>
---
#### [new 030] CHyLL: Learning Continuous Neural Representations of Hybrid Systems
- **分类: cs.LG; cs.AI; cs.RO; eess.SP; eess.SY**

- **简介: 该论文提出CHyLL方法，解决混合系统中连续与离散动态学习难的问题。通过将状态空间重构为分段光滑商流形，实现无需模式切换的连续神经表示学习，准确预测系统流并识别拓扑不变量，应用于随机最优控制。**

- **链接: [https://arxiv.org/pdf/2512.10117v1](https://arxiv.org/pdf/2512.10117v1)**

> **作者:** Sangli Teng; Hang Liu; Jingyu Song; Koushil Sreenath
>
> **摘要:** Learning the flows of hybrid systems that have both continuous and discrete time dynamics is challenging. The existing method learns the dynamics in each discrete mode, which suffers from the combination of mode switching and discontinuities in the flows. In this work, we propose CHyLL (Continuous Hybrid System Learning in Latent Space), which learns a continuous neural representation of a hybrid system without trajectory segmentation, event functions, or mode switching. The key insight of CHyLL is that the reset map glues the state space at the guard surface, reformulating the state space as a piecewise smooth quotient manifold where the flow becomes spatially continuous. Building upon these insights and the embedding theorems grounded in differential topology, CHyLL concurrently learns a singularity-free neural embedding in a higher-dimensional space and the continuous flow in it. We showcase that CHyLL can accurately predict the flow of hybrid systems with superior accuracy and identify the topological invariants of the hybrid systems. Finally, we apply CHyLL to the stochastic optimal control problem.
>
---
#### [new 031] Decoupled Q-Chunking
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **简介: 该论文研究离线强化学习中的值函数偏差问题，提出解耦的Q-分块方法（Decoupled Q-Chunking），通过构建面向短动作片段的策略与扩展价值目标的分块批评者，缓解自举偏差与开环策略次优性，提升长视野任务性能。**

- **链接: [https://arxiv.org/pdf/2512.10926v1](https://arxiv.org/pdf/2512.10926v1)**

> **作者:** Qiyang Li; Seohong Park; Sergey Levine
>
> **备注:** 76 pages, 14 figures
>
> **摘要:** Temporal-difference (TD) methods learn state and action values efficiently by bootstrapping from their own future value predictions, but such a self-bootstrapping mechanism is prone to bootstrapping bias, where the errors in the value targets accumulate across steps and result in biased value estimates. Recent work has proposed to use chunked critics, which estimate the value of short action sequences ("chunks") rather than individual actions, speeding up value backup. However, extracting policies from chunked critics is challenging: policies must output the entire action chunk open-loop, which can be sub-optimal for environments that require policy reactivity and also challenging to model especially when the chunk length grows. Our key insight is to decouple the chunk length of the critic from that of the policy, allowing the policy to operate over shorter action chunks. We propose a novel algorithm that achieves this by optimizing the policy against a distilled critic for partial action chunks, constructed by optimistically backing up from the original chunked critic to approximate the maximum value achievable when a partial action chunk is extended to a complete one. This design retains the benefits of multi-step value propagation while sidestepping both the open-loop sub-optimality and the difficulty of learning action chunking policies for long action chunks. We evaluate our method on challenging, long-horizon offline goal-conditioned tasks and show that it reliably outperforms prior methods. Code: github.com/ColinQiyangLi/dqc.
>
---
#### [new 032] Design and Implementation of a High-Precision Wind-Estimation UAV with Onboard Sensors
- **分类: cs.ET; cs.RO**

- **简介: 该论文研究基于机载传感器的实时风场估计任务，解决传统方法依赖外部设备或简化动力学导致精度不足的问题。提出结合扰动观测器与薄板样条模型的方法，并设计风感吊舱提升灵敏度，实现了高精度三维风速估计。**

- **链接: [https://arxiv.org/pdf/2512.10428v1](https://arxiv.org/pdf/2512.10428v1)**

> **作者:** Haowen Yu; Na Fan; Xing Liu; Ximin Lyu
>
> **备注:** https://www.sciencedirect.com/science/article/abs/pii/S0263224125032415?via%3Dihub
>
> **摘要:** Accurate real-time wind vector estimation is essential for enhancing the safety, navigation accuracy, and energy efficiency of unmanned aerial vehicles (UAVs). Traditional approaches rely on external sensors or simplify vehicle dynamics, which limits their applicability during agile flight or in resource-constrained platforms. This paper proposes a real-time wind estimation method based solely on onboard sensors. The approach first estimates external aerodynamic forces using a disturbance observer (DOB), and then maps these forces to wind vectors using a thin-plate spline (TPS) model. A custom-designed wind barrel mounted on the UAV enhances aerodynamic sensitivity, further improving estimation accuracy. The system is validated through comprehensive experiments in wind tunnels, indoor and outdoor flights. Experimental results demonstrate that the proposed method achieves consistently high-accuracy wind estimation across controlled and real-world conditions, with speed RMSEs as low as \SI{0.06}{m/s} in wind tunnel tests, \SI{0.22}{m/s} during outdoor hover, and below \SI{0.38}{m/s} in indoor and outdoor dynamic flights, and direction RMSEs under \ang{7.3} across all scenarios, outperforming existing baselines. Moreover, the method provides vertical wind estimates -- unavailable in baselines -- with RMSEs below \SI{0.17}{m/s} even during fast indoor translations.
>
---
#### [new 033] Active Optics for Hyperspectral Imaging of Reflective Agricultural Leaf Sensors
- **分类: eess.IV; cs.RO**

- **简介: 该论文针对农业中难以高效定位和读取植物叶片传感器的问题，提出一种集成主动光学的自主检测系统。通过LiDAR识别传感器，结合快速反射镜、液态镜头等实现动态追踪与高光谱成像，实现低成本、可扩展的植物健康实时监测。**

- **链接: [https://arxiv.org/pdf/2512.10213v1](https://arxiv.org/pdf/2512.10213v1)**

> **作者:** Dexter Burns; Sanjeev Koppal
>
> **摘要:** Monitoring plant health increasingly relies on leaf-mounted sensors that provide real-time physiological data, yet efficiently locating and sampling these sensors in complex agricultural environments remains a major challenge. We present an integrated, adaptive, and scalable system that autonomously detects and interrogates plant sensors using a coordinated suite of low-cost optical components including a LiDAR, liquid lens, monochrome camera, filter wheel, and Fast Steering Mirror (FSM). The system first uses LiDAR to identify the distinct reflective signatures of sensors within the field, then dynamically redirects the camera s field of view via the FSM to target each sensor for hyperspectral imaging. The liquid lens continuously adjusts focus to maintain image sharpness across varying depths, enabling precise spectral measurements. We validated the system in controlled indoor experiments, demonstrating accurate detection and tracking of reflective plant sensors and successful acquisition of their spectral data. To our knowledge, no other system currently integrates these sensing and optical modalities for agricultural monitoring. This work establishes a foundation for adaptive, low-cost, and automated plant sensor interrogation, representing a significant step toward scalable, real-time plant health monitoring in precision agriculture.
>
---
## 更新

#### [replaced 001] MaskedManipulator: Versatile Whole-Body Manipulation
- **分类: cs.RO; cs.AI; cs.GR**

- **简介: 该论文研究物理仿真中的人体全身物体操作任务，旨在解决现有方法依赖精细轨迹控制、缺乏灵活性的问题。作者提出MaskedManipulator，通过两阶段学习从动捕数据中蒸馏生成控制策略，实现用户指定高层目标（如物体/姿态）的多样化操作行为。**

- **链接: [https://arxiv.org/pdf/2505.19086v3](https://arxiv.org/pdf/2505.19086v3)**

> **作者:** Chen Tessler; Yifeng Jiang; Erwin Coumans; Zhengyi Luo; Gal Chechik; Xue Bin Peng
>
> **备注:** SIGGRAPH Asia 2025 (Project page: https://research.nvidia.com/labs/par/maskedmanipulator/ )
>
> **摘要:** We tackle the challenges of synthesizing versatile, physically simulated human motions for full-body object manipulation. Unlike prior methods that are focused on detailed motion tracking, trajectory following, or teleoperation, our framework enables users to specify versatile high-level objectives such as target object poses or body poses. To achieve this, we introduce MaskedManipulator, a generative control policy distilled from a tracking controller trained on large-scale human motion capture data. This two-stage learning process allows the system to perform complex interaction behaviors, while providing intuitive user control over both character and object motions. MaskedManipulator produces goal-directed manipulation behaviors that expand the scope of interactive animation systems beyond task-specific solutions.
>
---
#### [replaced 002] WAM-Flow: Parallel Coarse-to-Fine Motion Planning via Discrete Flow Matching for Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出WAM-Flow，用于自动驾驶中轨迹规划任务。它将规划建模为离散流匹配，实现并行双向去噪，支持粗到精的生成。通过结构化分词、几何感知目标和强化学习对齐，提升性能与安全性，在NAVSIM上取得优于自回归和扩散模型的效果。**

- **链接: [https://arxiv.org/pdf/2512.06112v2](https://arxiv.org/pdf/2512.06112v2)**

> **作者:** Yifang Xu; Jiahao Cui; Feipeng Cai; Zhihao Zhu; Hanlin Shang; Shan Luan; Mingwang Xu; Neng Zhang; Yaoyi Li; Jia Cai; Siyu Zhu
>
> **备注:** 18 pages, 11 figures. Code & Model: https://github.com/fudan-generative-vision/WAM-Flow
>
> **摘要:** We introduce WAM-Flow, a vision-language-action (VLA) model that casts ego-trajectory planning as discrete flow matching over a structured token space. In contrast to autoregressive decoders, WAM-Flow performs fully parallel, bidirectional denoising, enabling coarse-to-fine refinement with a tunable compute-accuracy trade-off. Specifically, the approach combines a metric-aligned numerical tokenizer that preserves scalar geometry via triplet-margin learning, a geometry-aware flow objective and a simulator-guided GRPO alignment that integrates safety, ego progress, and comfort rewards while retaining parallel generation. A multi-stage adaptation converts a pre-trained auto-regressive backbone (Janus-1.5B) from causal decoding to non-causal flow model and strengthens road-scene competence through continued multimodal pretraining. Thanks to the inherent nature of consistency model training and parallel decoding inference, WAM-Flow achieves superior closed-loop performance against autoregressive and diffusion-based VLA baselines, with 1-step inference attaining 89.1 PDMS and 5-step inference reaching 90.3 PDMS on NAVSIM v1 benchmark. These results establish discrete flow matching as a new promising paradigm for end-to-end autonomous driving. The code will be publicly available soon.
>
---
#### [replaced 003] Multi-Robot Path Planning Combining Heuristics and Multi-Agent Reinforcement Learning
- **分类: cs.AI; cs.LG; cs.RO**

- **简介: 该论文研究多机器人路径规划任务，旨在解决动态环境中避障与高效路径规划问题。提出MAPPOHR方法，结合启发式搜索、经验规则与多智能体强化学习，提升规划性能与学习效率。**

- **链接: [https://arxiv.org/pdf/2306.01270v2](https://arxiv.org/pdf/2306.01270v2)**

> **作者:** Shaoming Peng
>
> **摘要:** Multi-robot path finding in dynamic environments is a highly challenging classic problem. In the movement process, robots need to avoid collisions with other moving robots while minimizing their travel distance. Previous methods for this problem either continuously replan paths using heuristic search methods to avoid conflicts or choose appropriate collision avoidance strategies based on learning approaches. The former may result in long travel distances due to frequent replanning, while the latter may have low learning efficiency due to low sample exploration and utilization, and causing high training costs for the model. To address these issues, we propose a path planning method, MAPPOHR, which combines heuristic search, empirical rules, and multi-agent reinforcement learning. The method consists of two layers: a real-time planner based on the multi-agent reinforcement learning algorithm, MAPPO, which embeds empirical rules in the action output layer and reward functions, and a heuristic search planner used to create a global guiding path. During movement, the heuristic search planner replans new paths based on the instructions of the real-time planner. We tested our method in 10 different conflict scenarios. The experiments show that the planning performance of MAPPOHR is better than that of existing learning and heuristic methods. Due to the utilization of empirical knowledge and heuristic search, the learning efficiency of MAPPOHR is higher than that of existing learning methods.
>
---
#### [replaced 004] Semantic Trajectory Generation for Goal-Oriented Spacecraft Rendezvous
- **分类: cs.RO; cs.AI; math.OC**

- **简介: 该论文研究语言驱动的航天器轨迹生成任务，旨在减少专家依赖。提出SAGES框架，将自然语言指令转化为满足非凸约束的轨迹，实现高语义一致性，支持自主交会任务中的安全与行为交互引导。**

- **链接: [https://arxiv.org/pdf/2512.09111v2](https://arxiv.org/pdf/2512.09111v2)**

> **作者:** Yuji Takubo; Arpit Dwivedi; Sukeerth Ramkumar; Luis A. Pabon; Daniele Gammelli; Marco Pavone; Simone D'Amico
>
> **备注:** 28 pages, 12 figures. Submitted to AIAA SCITECH 2026
>
> **摘要:** Reliable real-time trajectory generation is essential for future autonomous spacecraft. While recent progress in nonconvex guidance and control is paving the way for onboard autonomous trajectory optimization, these methods still rely on extensive expert input (e.g., waypoints, constraints, mission timelines, etc.), which limits the operational scalability in real rendezvous missions. This paper introduces SAGES (Semantic Autonomous Guidance Engine for Space), a trajectory-generation framework that translates natural-language commands into spacecraft trajectories that reflect high-level intent while respecting nonconvex constraints. Experiments in two settings -- fault-tolerant proximity operations with continuous-time constraint enforcement and a free-flying robotic platform -- demonstrate that SAGES reliably produces trajectories aligned with human commands, achieving over 90% semantic-behavioral consistency across diverse behavior modes. Ultimately, this work marks an initial step toward language-conditioned, constraint-aware spacecraft trajectory generation, enabling operators to interactively guide both safety and behavior through intuitive natural-language commands with reduced expert burden.
>
---
#### [replaced 005] Development and Testing for Perception Based Autonomous Landing of a Long-Range QuadPlane
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究基于视觉的长航程QuadPlane自主降落任务，解决GPS拒止环境下无结构着陆区的感知与稳定降落问题。提出轻量级系统架构，集成视觉-惯性里程计与高效DNN感知，实现资源受限边缘设备上的实时检测与姿态估计。**

- **链接: [https://arxiv.org/pdf/2512.09343v2](https://arxiv.org/pdf/2512.09343v2)**

> **作者:** Ashik E Rasul; Humaira Tasnim; Ji Yu Kim; Young Hyun Lim; Scott Schmitz; Bruce W. Jo; Hyung-Jin Yoon
>
> **摘要:** QuadPlanes combine the range efficiency of fixed-wing aircraft with the maneuverability of multi-rotor platforms for long-range autonomous missions. In GPS-denied or cluttered urban environments, perception-based landing is vital for reliable operation. Unlike structured landing zones, real-world sites are unstructured and highly variable, requiring strong generalization capabilities from the perception system. Deep neural networks (DNNs) provide a scalable solution for learning landing site features across diverse visual and environmental conditions. While perception-driven landing has been shown in simulation, real-world deployment introduces significant challenges. Payload and volume constraints limit high-performance edge AI devices like the NVIDIA Jetson Orin Nano, which are crucial for real-time detection and control. Accurate pose estimation during descent is necessary, especially in the absence of GPS, and relies on dependable visual-inertial odometry. Achieving this with limited edge AI resources requires careful optimization of the entire deployment framework. The flight characteristics of large QuadPlanes further complicate the problem. These aircraft exhibit high inertia, reduced thrust vectoring, and slow response times further complicate stable landing maneuvers. This work presents a lightweight QuadPlane system for efficient vision-based autonomous landing and visual-inertial odometry, specifically developed for long-range QuadPlane operations such as aerial monitoring. It describes the hardware platform, sensor configuration, and embedded computing architecture designed to meet demanding real-time, physical constraints. This establishes a foundation for deploying autonomous landing in dynamic, unstructured, GPS-denied environments.
>
---
#### [replaced 006] Enhancing Hand Palm Motion Gesture Recognition by Eliminating Reference Frame Bias via Frame-Invariant Similarity Measures
- **分类: cs.RO; cs.CV; cs.HC**

- **简介: 该论文研究手部掌心动作手势识别，旨在解决因参考系变化导致的识别不稳定问题。提出并评测了多种不变性轨迹描述方法，在新构建的HPM数据集上验证其鲁棒性，最终通过实时原型系统实现92.3% F1分数，证明其有效性。**

- **链接: [https://arxiv.org/pdf/2503.11352v2](https://arxiv.org/pdf/2503.11352v2)**

> **作者:** Arno Verduyn; Maxim Vochten; Joris De Schutter
>
> **备注:** This is the preprint version of a paper accepted for publication at the 2025 IEEE International Conference on Automation Science and Engineering (CASE). The final published version is available at DOI: 10.1109/CASE58245.2025.11163910
>
> **摘要:** The ability of robots to recognize human gestures facilitates a natural and accessible human-robot collaboration. However, most work in gesture recognition remains rooted in reference frame-dependent representations. This poses a challenge when reference frames vary due to different work cell layouts, imprecise frame calibrations, or other environmental changes. This paper investigated the use of invariant trajectory descriptors for robust hand palm motion gesture recognition under reference frame changes. First, a novel dataset of recorded Hand Palm Motion (HPM) gestures is introduced. The motion gestures in this dataset were specifically designed to be distinguishable without dependence on specific reference frames or directional cues. Afterwards, multiple invariant trajectory descriptor approaches were benchmarked to assess how their performances generalize to this novel HPM dataset. After this offline benchmarking, the best scoring approach is validated for online recognition by developing a real-time Proof of Concept (PoC). In this PoC, hand palm motion gestures were used to control the real-time movement of a manipulator arm. The PoC demonstrated a high recognition reliability in real-time operation, achieving an $F_1$-score of 92.3%. This work demonstrates the effectiveness of the invariant descriptor approach as a standalone solution. Moreover, we believe that the invariant descriptor approach can also be utilized within other state-of-the-art pattern recognition and learning systems to improve their robustness against reference frame variations.
>
---
#### [replaced 007] Compliant Residual DAgger: Improving Real-World Contact-Rich Manipulation with Human Corrections
- **分类: cs.RO; cs.LG**

- **简介: 该论文研究真实场景中高接触复杂度的机器人操作任务，提出Compliant Residual DAgger（CR-DAgger），通过柔顺干预接口和残差策略学习，利用少量人类修正数据提升策略性能，显著提高书本翻页和皮带装配等任务的成功率。**

- **链接: [https://arxiv.org/pdf/2506.16685v4](https://arxiv.org/pdf/2506.16685v4)**

> **作者:** Xiaomeng Xu; Yifan Hou; Zeyi Liu; Shuran Song
>
> **摘要:** We address key challenges in Dataset Aggregation (DAgger) for real-world contact-rich manipulation: how to collect informative human correction data and how to effectively update policies with this new data. We introduce Compliant Residual DAgger (CR-DAgger), which contains two novel components: 1) a Compliant Intervention Interface that leverages compliance control, allowing humans to provide gentle, accurate delta action corrections without interrupting the ongoing robot policy execution; and 2) a Compliant Residual Policy formulation that learns from human corrections while incorporating force feedback and force control. Our system significantly enhances performance on precise contact-rich manipulation tasks using minimal correction data, improving base policy success rates by over 50\% on two challenging tasks (book flipping and belt assembly) while outperforming both retraining-from-scratch and finetuning approaches. Through extensive real-world experiments, we provide practical guidance for implementing effective DAgger in real-world robot learning tasks. Result videos are available at: https://compliant-residual-dagger.github.io/
>
---
#### [replaced 008] ShapeForce: Low-Cost Soft Robotic Wrist for Contact-Rich Manipulation
- **分类: cs.RO**

- **简介: 该论文提出ShapeForce，一种低成本软体机器人腕部，用于接触丰富的操作任务。它通过形变感知力变化，替代昂贵的六轴力矩传感器，实现免校准、即插即用的力反馈，支持多种接触操作，性能相当但成本极低。**

- **链接: [https://arxiv.org/pdf/2511.19955v2](https://arxiv.org/pdf/2511.19955v2)**

> **作者:** Jinxuan Zhu; Zihao Yan; Yangyu Xiao; Jingxiang Guo; Chenrui Tie; Xinyi Cao; Yuhang Zheng; Lin Shao
>
> **摘要:** Contact feedback is essential for contact-rich robotic manipulation, as it allows the robot to detect subtle interaction changes and adjust its actions accordingly. Six-axis force-torque sensors are commonly used to obtain contact feedback, but their high cost and fragility have discouraged many researchers from adopting them in contact-rich tasks. To offer a more cost-efficient and easy-accessible source of contact feedback, we present ShapeForce, a low-cost, plug-and-play soft wrist that provides force-like signals for contact-rich robotic manipulation. Inspired by how humans rely on relative force changes in contact rather than precise force magnitudes, ShapeForce converts external force and torque into measurable deformations of its compliant core, which are then estimated via marker-based pose tracking and converted into force-like signals. Our design eliminates the need for calibration or specialized electronics to obtain exact values, and instead focuses on capturing force and torque changes sufficient for enabling contact-rich manipulation. Extensive experiments across diverse contact-rich tasks and manipulation policies demonstrate that ShapeForce delivers performance comparable to six-axis force-torque sensors at an extremely low cost.
>
---
#### [replaced 009] FICO: Finite-Horizon Closed-Loop Factorization for Unified Multi-Agent Path Finding
- **分类: cs.RO**

- **简介: 该论文研究多智能体路径规划（MAPF），旨在统一建模并解决规划与执行割裂、缺乏不确定性处理的问题。作者提出FICO框架，将MAPF视为控制系统问题，通过有限时域闭环因子化算法实现高效实时响应，支持大规模 agents 并提升不确定环境下的鲁棒性与吞吐量。**

- **链接: [https://arxiv.org/pdf/2511.13961v2](https://arxiv.org/pdf/2511.13961v2)**

> **作者:** Jiarui Li; Alessandro Zanardi; Federico Pecora; Runyu Zhang; Gioele Zardini
>
> **摘要:** Multi-Agent Path Finding is a fundamental problem in robotics and AI, yet most existing formulations treat planning and execution separately and address variants of the problem in an ad hoc manner. This paper presents a system-level framework for MAPF that integrates planning and execution, generalizes across variants, and explicitly models uncertainties. At its core is the MAPF system, a formal model that casts MAPF as a control design problem encompassing classical and uncertainty-aware formulations. To solve it, we introduce Finite-Horizon Closed-Loop Factorization (FICO), a factorization-based algorithm inspired by receding-horizon control that exploits compositional structure for efficient closed-loop operation. FICO enables real-time responses -- commencing execution within milliseconds -- while scaling to thousands of agents and adapting seamlessly to execution-time uncertainties. Extensive case studies demonstrate that it reduces computation time by up to two orders of magnitude compared with open-loop baselines, while delivering significantly higher throughput under stochastic delays and agent arrivals. These results establish a principled foundation for analyzing and advancing MAPF through system-level modeling, factorization, and closed-loop design.
>
---
#### [replaced 010] Multi-Modal Graph Convolutional Network with Sinusoidal Encoding for Robust Human Action Segmentation
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究人类动作分割任务，旨在解决因姿态与物体检测噪声导致的过分割问题。提出多模态图卷积网络，结合高低帧率数据，引入正弦编码、时序融合模块和SmoothLabelMix增强方法，提升分割准确性和时序连贯性。**

- **链接: [https://arxiv.org/pdf/2507.00752v2](https://arxiv.org/pdf/2507.00752v2)**

> **作者:** Hao Xing; Kai Zhe Boey; Yuankai Wu; Darius Burschka; Gordon Cheng
>
> **备注:** 8 pages, 5 figures, accepted in IROS25, Hangzhou, China
>
> **摘要:** Accurate temporal segmentation of human actions is critical for intelligent robots in collaborative settings, where a precise understanding of sub-activity labels and their temporal structure is essential. However, the inherent noise in both human pose estimation and object detection often leads to over-segmentation errors, disrupting the coherence of action sequences. To address this, we propose a Multi-Modal Graph Convolutional Network (MMGCN) that integrates low-frame-rate (e.g., 1 fps) visual data with high-frame-rate (e.g., 30 fps) motion data (skeleton and object detections) to mitigate fragmentation. Our framework introduces three key contributions. First, a sinusoidal encoding strategy that maps 3D skeleton coordinates into a continuous sin-cos space to enhance spatial representation robustness. Second, a temporal graph fusion module that aligns multi-modal inputs with differing resolutions via hierarchical feature aggregation, Third, inspired by the smooth transitions inherent to human actions, we design SmoothLabelMix, a data augmentation technique that mixes input sequences and labels to generate synthetic training examples with gradual action transitions, enhancing temporal consistency in predictions and reducing over-segmentation artifacts. Extensive experiments on the Bimanual Actions Dataset, a public benchmark for human-object interaction understanding, demonstrate that our approach outperforms state-of-the-art methods, especially in action segmentation accuracy, achieving F1@10: 94.5% and F1@25: 92.8%.
>
---
#### [replaced 011] Panoramic Out-of-Distribution Segmentation
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文提出全景异常分割（PanOoS）任务，解决现有方法在360°图像中难以识别异常物体的问题。作者设计POS模型，结合文本引导的提示分布学习，提升跨域泛化与语义解码，并构建两个新数据集DenseOoS和QuadOoS，显著提升检测性能。**

- **链接: [https://arxiv.org/pdf/2505.03539v3](https://arxiv.org/pdf/2505.03539v3)**

> **作者:** Mengfei Duan; Yuheng Zhang; Yihong Cao; Fei Teng; Kai Luo; Jiaming Zhang; Kailun Yang; Zhiyong Li
>
> **备注:** Code and datasets will be available at https://github.com/MengfeiD/PanOoS
>
> **摘要:** Panoramic imaging enables capturing 360° images with an ultra-wide Field-of-View (FoV) for dense omnidirectional perception, which is critical to applications, such as autonomous driving and augmented reality, etc. However, current panoramic semantic segmentation methods fail to identify outliers, and pinhole Out-of-distribution Segmentation (OoS) models perform unsatisfactorily in the panoramic domain due to pixel distortions and background clutter. To address these issues, we introduce a new task, Panoramic Out-of-distribution Segmentation (PanOoS), with the aim of achieving comprehensive and safe scene understanding. Furthermore, we propose the first solution, POS, which adapts to the characteristics of panoramic images through text-guided prompt distribution learning. Specifically, POS integrates a disentanglement strategy designed to materialize the cross-domain generalization capability of CLIP. The proposed Prompt-based Restoration Attention (PRA) optimizes semantic decoding by prompt guidance and self-adaptive correction, while Bilevel Prompt Distribution Learning (BPDL) refines the manifold of per-pixel mask embeddings via semantic prototype supervision. Besides, to compensate for the scarcity of PanOoS datasets, we establish two benchmarks: DenseOoS, which features diverse outliers in complex environments, and QuadOoS, captured by a quadruped robot with a panoramic annular lens system. Extensive experiments demonstrate superior performance of POS, with AuPRC improving by 34.25% and FPR95 decreasing by 21.42% on DenseOoS, outperforming state-of-the-art pinhole-OoS methods. Moreover, POS achieves leading closed-set segmentation capabilities and advances the development of panoramic understanding. Code and datasets will be available at https://github.com/MengfeiD/PanOoS.
>
---
#### [replaced 012] Reframing Human-Robot Interaction Through Extended Reality: Unlocking Safer, Smarter, and More Empathic Interactions with Virtual Robots and Foundation Models
- **分类: cs.HC; cs.RO**

- **简介: 该论文探讨通过扩展现实（XR）重构人机交互，利用大基础模型驱动虚拟机器人，解决物理机器人受限问题。提出虚拟代理可实现安全、共情、可扩展的交互，并讨论技术挑战与伦理风险，倡导以人为本的XR代理研究方向。**

- **链接: [https://arxiv.org/pdf/2512.02569v2](https://arxiv.org/pdf/2512.02569v2)**

> **作者:** Yuchong Zhang; Yong Ma; Danica Kragic
>
> **备注:** This paper is under review
>
> **摘要:** This perspective reframes human-robot interaction (HRI) through extended reality (XR), arguing that virtual robots powered by large foundation models (FMs) can serve as cognitively grounded, empathic agents. Unlike physical robots, XR-native agents are unbound by hardware constraints and can be instantiated, adapted, and scaled on demand, while still affording embodiment and co-presence. We synthesize work across XR, HRI, and cognitive AI to show how such agents can support safety-critical scenarios, socially and cognitively empathic interaction across domains, and outreaching physical capabilities with XR and AI integration. We then discuss how multimodal large FMs (e.g., large language model, large vision model, and vision-language model) enable context-aware reasoning, affect-sensitive situations, and long-term adaptation, positioning virtual robots as cognitive and empathic mediators rather than mere simulation assets. At the same time, we highlight challenges and potential risks, including overtrust, cultural and representational bias, privacy concerns around biometric sensing, and data governance and transparency. The paper concludes by outlining a research agenda for human-centered, ethically grounded XR agents - emphasizing multi-layered evaluation frameworks, multi-user ecosystems, mixed virtual-physical embodiment, and societal and ethical design practices to envision XR-based virtual agents powered by FMs as reshaping future HRI into a more efficient and adaptive paradigm.
>
---
#### [replaced 013] Mastering Diverse, Unknown, and Cluttered Tracks for Robust Vision-Based Drone Racing
- **分类: cs.RO**

- **简介: 该论文研究视觉无人机竞速，解决在未知、杂乱环境中兼顾高速飞行与避障的难题。提出两阶段强化学习框架，结合自适应课程学习与Lipschitz约束，实现从仿真到实机的泛化，提升复杂赛道的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.09571v2](https://arxiv.org/pdf/2512.09571v2)**

> **作者:** Feng Yu; Yu Hu; Yang Su; Yang Deng; Linzuo Zhang; Danping Zou
>
> **备注:** 8 pages, 9 figures, accepted to Robotics and Automation Letters
>
> **摘要:** Most reinforcement learning(RL)-based methods for drone racing target fixed, obstacle-free tracks, leaving the generalization to unknown, cluttered environments largely unaddressed. This challenge stems from the need to balance racing speed and collision avoidance, limited feasible space causing policy exploration trapped in local optima during training, and perceptual ambiguity between gates and obstacles in depth maps-especially when gate positions are only coarsely specified. To overcome these issues, we propose a two-phase learning framework: an initial soft-collision training phase that preserves policy exploration for high-speed flight, followed by a hard-collision refinement phase that enforces robust obstacle avoidance. An adaptive, noise-augmented curriculum with an asymmetric actor-critic architecture gradually shifts the policy's reliance from privileged gate-state information to depth-based visual input. We further impose Lipschitz constraints and integrate a track-primitive generator to enhance motion stability and cross-environment generalization. We evaluate our framework through extensive simulation and ablation studies, and validate it in real-world experiments on a computationally constrained quadrotor. The system achieves agile flight while remaining robust to gate-position errors, developing a generalizable drone racing framework with the capability to operate in diverse, partially unknown and cluttered environments. https://yufengsjtu.github.io/MasterRacing.github.io/
>
---
#### [replaced 014] SEA: Semantic Map Prediction for Active Exploration of Uncertain Areas
- **分类: cs.RO**

- **简介: 该论文研究主动探索任务，旨在解决机器人在未知环境中高效构建语义地图的问题。提出SEA方法，通过语义地图预测与分层强化学习策略，迭代预测缺失区域并指导探索，提升全局地图覆盖率。**

- **链接: [https://arxiv.org/pdf/2510.19766v2](https://arxiv.org/pdf/2510.19766v2)**

> **作者:** Hongyu Ding; Xinyue Liang; Yudong Fang; You Wu; Jieqi Shi; Jing Huo; Wenbin Li; Jing Wu; Yu-Kun Lai; Yang Gao
>
> **备注:** Project page: https://robo-lavira.github.io/sea-active-exp
>
> **摘要:** In this paper, we propose SEA, a novel approach for active robot exploration through semantic map prediction and a reinforcement learning-based hierarchical exploration policy. Unlike existing learning-based methods that rely on one-step waypoint prediction, our approach enhances the agent's long-term environmental understanding to facilitate more efficient exploration. We propose an iterative prediction-exploration framework that explicitly predicts the missing areas of the map based on current observations. The difference between the actual accumulated map and the predicted global map is then used to guide exploration. Additionally, we design a novel reward mechanism that leverages reinforcement learning to update the long-term exploration strategies, enabling us to construct an accurate semantic map within limited steps. Experimental results demonstrate that our method significantly outperforms state-of-the-art exploration strategies, achieving superior coverage ares of the global map within the same time constraints.
>
---
#### [replaced 015] From Generated Human Videos to Physically Plausible Robot Trajectories
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究如何将生成的人类视频转化为机器人可执行的动作。针对生成视频噪声多、形态不一致的问题，提出两阶段方法：先将视频转为4D人体表示并适配机器人形态，再通过物理感知的强化学习策略GenMimic实现动作模仿，在仿真和真实机器人上均实现零样本稳定运动跟踪。**

- **链接: [https://arxiv.org/pdf/2512.05094v2](https://arxiv.org/pdf/2512.05094v2)**

> **作者:** James Ni; Zekai Wang; Wei Lin; Amir Bar; Yann LeCun; Trevor Darrell; Jitendra Malik; Roei Herzig
>
> **备注:** For project website, see https://genmimic.github.io
>
> **摘要:** Video generation models are rapidly improving in their ability to synthesize human actions in novel contexts, holding the potential to serve as high-level planners for contextual robot control. To realize this potential, a key research question remains open: how can a humanoid execute the human actions from generated videos in a zero-shot manner? This challenge arises because generated videos are often noisy and exhibit morphological distortions that make direct imitation difficult compared to real video. To address this, we introduce a two-stage pipeline. First, we lift video pixels into a 4D human representation and then retarget to the humanoid morphology. Second, we propose GenMimic-a physics-aware reinforcement learning policy conditioned on 3D keypoints, and trained with symmetry regularization and keypoint-weighted tracking rewards. As a result, GenMimic can mimic human actions from noisy, generated videos. We curate GenMimicBench, a synthetic human-motion dataset generated using two video generation models across a spectrum of actions and contexts, establishing a benchmark for assessing zero-shot generalization and policy robustness. Extensive experiments demonstrate improvements over strong baselines in simulation and confirm coherent, physically stable motion tracking on a Unitree G1 humanoid robot without fine-tuning. This work offers a promising path to realizing the potential of video generation models as high-level policies for robot control.
>
---
#### [replaced 016] Continuous Vision-Language-Action Co-Learning with Semantic-Physical Alignment for Behavioral Cloning
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文研究语言条件下的行为克隆任务，旨在解决序列动作中累积误差与语义-物理错配问题。提出CCoL框架，通过视觉-语言-动作连续协同学习和双向交叉注意力实现语义-物理对齐，提升动作克隆的连贯性与准确性。**

- **链接: [https://arxiv.org/pdf/2511.14396v2](https://arxiv.org/pdf/2511.14396v2)**

> **作者:** Xiuxiu Qi; Yu Yang; Jiannong Cao; Luyao Bai; Chongshan Fan; Chengtai Cao; Hongpeng Wang
>
> **备注:** Accepted at AAAI 2026, the Project website is available at https://qhemu.github.io/CCoL/
>
> **摘要:** Language-conditioned manipulation facilitates human-robot interaction via behavioral cloning (BC), which learns control policies from human demonstrations and serves as a cornerstone of embodied AI. Overcoming compounding errors in sequential action decisions remains a central challenge to improving BC performance. Existing approaches mitigate compounding errors through data augmentation, expressive representation, or temporal abstraction. However, they suffer from physical discontinuities and semantic-physical misalignment, leading to inaccurate action cloning and intermittent execution. In this paper, we present Continuous vision-language-action Co-Learning with Semantic-Physical Alignment (CCoL), a novel BC framework that ensures temporally consistent execution and fine-grained semantic grounding. It generates robust and smooth action execution trajectories through continuous co-learning across vision, language, and proprioceptive inputs (e.g., robot internal states). Meanwhile, we anchor language semantics to visuomotor representations by a bidirectional cross-attention to learn contextual information for action generation, successfully overcoming the problem of semantic-physical misalignment. Extensive experiments show that CCoL achieves an average 8.0% relative improvement across three simulation suites, with up to 19.2% relative gain in human-demonstrated bimanual insertion tasks. Real-world tests on a 7-DoF robot further confirm CCoL's generalization under unseen and noisy object states.
>
---
#### [replaced 017] From the Laboratory to Real-World Application: Evaluating Zero-Shot Scene Interpretation on Edge Devices for Mobile Robotics
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究面向移动机器人场景理解的零样本视觉语言模型在边缘设备上的应用，旨在平衡精度与推理速度。作者评估了多种小型VLM在真实城市场景、校园及室内环境中的表现，分析其实际部署的潜力与挑战。**

- **链接: [https://arxiv.org/pdf/2511.02427v2](https://arxiv.org/pdf/2511.02427v2)**

> **作者:** Nicolas Schuler; Lea Dewald; Nick Baldig; Jürgen Graf
>
> **备注:** 15 pages, 6 figures, 1 table; accepted for AI-2025 Forty-fifth SGAI International Conference on Artificial Intelligence CAMBRIDGE, ENGLAND 16-18 DECEMBER 2025
>
> **摘要:** Video Understanding, Scene Interpretation and Commonsense Reasoning are highly challenging tasks enabling the interpretation of visual information, allowing agents to perceive, interact with and make rational decisions in its environment. Large Language Models (LLMs) and Visual Language Models (VLMs) have shown remarkable advancements in these areas in recent years, enabling domain-specific applications as well as zero-shot open vocabulary tasks, combining multiple domains. However, the required computational complexity poses challenges for their application on edge devices and in the context of Mobile Robotics, especially considering the trade-off between accuracy and inference time. In this paper, we investigate the capabilities of state-of-the-art VLMs for the task of Scene Interpretation and Action Recognition, with special regard to small VLMs capable of being deployed to edge devices in the context of Mobile Robotics. The proposed pipeline is evaluated on a diverse dataset consisting of various real-world cityscape, on-campus and indoor scenarios. The experimental evaluation discusses the potential of these small models on edge devices, with particular emphasis on challenges, weaknesses, inherent model biases and the application of the gained information. Supplementary material is provided via the following repository: https://datahub.rz.rptu.de/hstr-csrl-public/publications/scene-interpretation-on-edge-devices/
>
---
#### [replaced 018] Never too Cocky to Cooperate: An FIM and RL-based USV-AUV Collaborative System for Underwater Tasks in Extreme Sea Conditions
- **分类: cs.RO; eess.SY**

- **简介: 该论文针对极端海况下水下任务执行难题，提出一种基于FIM优化与强化学习的USV-AUV协同系统，实现高精度多AUV定位与稳定协作。通过开源仿真工具验证了系统在定位与任务协作上的有效性。**

- **链接: [https://arxiv.org/pdf/2504.14894v2](https://arxiv.org/pdf/2504.14894v2)**

> **作者:** Jingzehua Xu; Guanwen Xie; Jiwei Tang; Yimian Ding; Weiyi Liu; Shuai Zhang; Yi Li
>
> **备注:** This paper has been submitted to IEEE Transactions on Mobile Computing, and is currently under minor revision
>
> **摘要:** This paper develops a novel unmanned surface vehicle (USV)-autonomous underwater vehicle (AUV) collaborative system designed to enhance underwater task performance in extreme sea conditions. The system integrates a dual strategy: (1) high-precision multi-AUV localization enabled by Fisher information matrix-optimized USV path planning, and (2) reinforcement learning-based cooperative planning and control method for multi-AUV task execution. Extensive experimental evaluations in the underwater data collection task demonstrate the system's operational feasibility, with quantitative results showing significant performance improvements over baseline methods. The proposed system exhibits robust coordination capabilities between USV and AUVs while maintaining stability in extreme sea conditions. To facilitate reproducibility and community advancement, we provide an open-source simulation toolkit available at: https://github.com/360ZMEM/USV-AUV-colab .
>
---
#### [replaced 019] Towards Open-World Human Action Segmentation Using Graph Convolutional Networks
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究开放世界下的人类动作分割任务，旨在解决模型对未见动作的检测与分割问题。提出EPGCN网络、Mixup数据增强和时序聚类损失，实现无需人工标注的未见动作识别，在多个指标上显著优于现有方法。**

- **链接: [https://arxiv.org/pdf/2507.00756v2](https://arxiv.org/pdf/2507.00756v2)**

> **作者:** Hao Xing; Kai Zhe Boey; Gordon Cheng
>
> **备注:** 8 pages, 3 figures, accepted in IROS25, Hangzhou, China
>
> **摘要:** Human-object interaction segmentation is a fundamental task of daily activity understanding, which plays a crucial role in applications such as assistive robotics, healthcare, and autonomous systems. Most existing learning-based methods excel in closed-world action segmentation, they struggle to generalize to open-world scenarios where novel actions emerge. Collecting exhaustive action categories for training is impractical due to the dynamic diversity of human activities, necessitating models that detect and segment out-of-distribution actions without manual annotation. To address this issue, we formally define the open-world action segmentation problem and propose a structured framework for detecting and segmenting unseen actions. Our framework introduces three key innovations: 1) an Enhanced Pyramid Graph Convolutional Network (EPGCN) with a novel decoder module for robust spatiotemporal feature upsampling. 2) Mixup-based training to synthesize out-of-distribution data, eliminating reliance on manual annotations. 3) A novel Temporal Clustering loss that groups in-distribution actions while distancing out-of-distribution samples. We evaluate our framework on two challenging human-object interaction recognition datasets: Bimanual Actions and 2 Hands and Object (H2O) datasets. Experimental results demonstrate significant improvements over state-of-the-art action segmentation models across multiple open-set evaluation metrics, achieving 16.9% and 34.6% relative gains in open-set segmentation (F1@50) and out-of-distribution detection performances (AUROC), respectively. Additionally, we conduct an in-depth ablation study to assess the impact of each proposed component, identifying the optimal framework configuration for open-world action segmentation.
>
---
#### [replaced 020] Risk-Bounded Multi-Agent Visual Navigation via Iterative Risk Allocation
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文研究多智能体视觉导航中的风险约束路径规划问题，提出通过迭代风险分配动态分发全局风险预算，结合目标条件强化学习与冲突搜索，实现安全与效率的平衡。**

- **链接: [https://arxiv.org/pdf/2509.08157v2](https://arxiv.org/pdf/2509.08157v2)**

> **作者:** Viraj Parimi; Brian C. Williams
>
> **摘要:** Safe navigation is essential for autonomous systems operating in hazardous environments, especially when multiple agents must coordinate using only high-dimensional visual observations. While recent approaches successfully combine Goal-Conditioned RL (GCRL) for graph construction with Conflict-Based Search (CBS) for planning, they typically rely on static edge pruning to enforce safety. This binary strategy is overly conservative, precluding feasible missions that require traversing high-risk regions, even when the aggregate risk is acceptable. To address this, we introduce a framework for Risk-Bounded Multi-Agent Path Finding (\problem{}), where agents share a user-specified global risk budget ($Δ$). Rather than permanently discarding edges, our framework dynamically distributes per-agent risk budgets ($δ_i$) during search via an Iterative Risk Allocation (IRA) layer that integrates with a standard CBS planner. We investigate two distribution strategies: a greedy surplus-deficit scheme for rapid feasibility repair, and a market-inspired mechanism that treats risk as a priced resource to guide improved allocation. This yields a tunable trade-off wherein agents exploit available risk to secure shorter, more efficient paths, but revert to longer, safer detours under tighter budgets. Experiments in complex visual environments show that, our dynamic allocation framework achieves higher success rates than baselines and effectively leverages the available safety budget to reduce travel time.
>
---
#### [replaced 021] Transformer Driven Visual Servoing and Dual Arm Impedance Control for Fabric Texture Matching
- **分类: cs.RO**

- **简介: 该论文研究双臂机器人基于视觉的布料纹理匹配任务，解决布料对齐与平整放置问题。提出融合Transformer视觉伺服与双臂阻抗控制的新方法，结合合成数据训练的DEAM网络实现零样本真实场景部署，准确完成纹理匹配。**

- **链接: [https://arxiv.org/pdf/2511.21203v2](https://arxiv.org/pdf/2511.21203v2)**

> **作者:** Fuyuki Tokuda; Akira Seino; Akinari Kobayashi; Kai Tang; Kazuhiro Kosuge
>
> **备注:** 8 pages, 11 figures. Accepted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** In this paper, we propose a method to align and place a fabric piece on top of another using a dual-arm manipulator and a grayscale camera, so that their surface textures are accurately matched. We propose a novel control scheme that combines Transformer-driven visual servoing with dualarm impedance control. This approach enables the system to simultaneously control the pose of the fabric piece and place it onto the underlying one while applying tension to keep the fabric piece flat. Our transformer-based network incorporates pretrained backbones and a newly introduced Difference Extraction Attention Module (DEAM), which significantly enhances pose difference prediction accuracy. Trained entirely on synthetic images generated using rendering software, the network enables zero-shot deployment in real-world scenarios without requiring prior training on specific fabric textures. Real-world experiments demonstrate that the proposed system accurately aligns fabric pieces with different textures.
>
---
#### [replaced 022] An Introduction to Deep Reinforcement and Imitation Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文介绍面向具身智能体的深度强化学习与模仿学习，旨在解决复杂序列决策任务中控制器设计困难的问题。通过深入讲解基础算法，如PPO、DAgger和GAIL，帮助读者建立扎实的理论理解。**

- **链接: [https://arxiv.org/pdf/2512.08052v2](https://arxiv.org/pdf/2512.08052v2)**

> **作者:** Pedro Santana
>
> **摘要:** Embodied agents, such as robots and virtual characters, must continuously select actions to execute tasks effectively, solving complex sequential decision-making problems. Given the difficulty of designing such controllers manually, learning-based approaches have emerged as promising alternatives, most notably Deep Reinforcement Learning (DRL) and Deep Imitation Learning (DIL). DRL leverages reward signals to optimize behavior, while DIL uses expert demonstrations to guide learning. This document introduces DRL and DIL in the context of embodied agents, adopting a concise, depth-first approach to the literature. It is self-contained, presenting all necessary mathematical and machine learning concepts as they are needed. It is not intended as a survey of the field; rather, it focuses on a small set of foundational algorithms and techniques, prioritizing in-depth understanding over broad coverage. The material ranges from Markov Decision Processes to REINFORCE and Proximal Policy Optimization (PPO) for DRL, and from Behavioral Cloning to Dataset Aggregation (DAgger) and Generative Adversarial Imitation Learning (GAIL) for DIL.
>
---
#### [replaced 023] Learning Agile Striker Skills for Humanoid Soccer Robots from Noisy Sensory Input
- **分类: cs.RO**

- **简介: 该论文研究 humanoid 足球机器人在感知噪声下的敏捷射门技能学习。针对感知不准确和环境扰动问题，提出四阶段教师-学生强化学习框架，结合噪声建模与在线约束强化学习，实现从仿真到真实的稳健连续射门控制。**

- **链接: [https://arxiv.org/pdf/2512.06571v2](https://arxiv.org/pdf/2512.06571v2)**

> **作者:** Zifan Xu; Myoungkyu Seo; Dongmyeong Lee; Hao Fu; Jiaheng Hu; Jiaxun Cui; Yuqian Jiang; Zhihan Wang; Anastasiia Brund; Joydeep Biswas; Peter Stone
>
> **摘要:** Learning fast and robust ball-kicking skills is a critical capability for humanoid soccer robots, yet it remains a challenging problem due to the need for rapid leg swings, postural stability on a single support foot, and robustness under noisy sensory input and external perturbations (e.g., opponents). This paper presents a reinforcement learning (RL)-based system that enables humanoid robots to execute robust continual ball-kicking with adaptability to different ball-goal configurations. The system extends a typical teacher-student training framework -- in which a "teacher" policy is trained with ground truth state information and the "student" learns to mimic it with noisy, imperfect sensing -- by including four training stages: (1) long-distance ball chasing (teacher); (2) directional kicking (teacher); (3) teacher policy distillation (student); and (4) student adaptation and refinement (student). Key design elements -- including tailored reward functions, realistic noise modeling, and online constrained RL for adaptation and refinement -- are critical for closing the sim-to-real gap and sustaining performance under perceptual uncertainty. Extensive evaluations in both simulation and on a real robot demonstrate strong kicking accuracy and goal-scoring success across diverse ball-goal configurations. Ablation studies further highlight the necessity of the constrained RL, noise modeling, and the adaptation stage. This work presents a system for learning robust continual humanoid ball-kicking under imperfect perception, establishing a benchmark task for visuomotor skill learning in humanoid whole-body control.
>
---
