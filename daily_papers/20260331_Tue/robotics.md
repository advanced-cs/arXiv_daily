# 机器人 cs.RO

- **最新发布 102 篇**

- **更新 52 篇**

## 最新发布

#### [new 001] Pandora: Articulated 3D Scene Graphs from Egocentric Vision
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Pandora，解决机器人感知环境不完整的问题，通过人类第一视角数据构建可动3D场景图，提升机器人对物体动态和容器关系的理解，增强移动操作能力。**

- **链接: [https://arxiv.org/pdf/2603.28732](https://arxiv.org/pdf/2603.28732)**

> **作者:** Alan Yu; Yun Chang; Christopher Xie; Luca Carlone
>
> **备注:** 14 pages, 5 figures. Presented at the 2025 British Machine Vision Conference (BMVC) in Sheffield, UK
>
> **摘要:** Robotic mapping systems typically approach building metric-semantic scene representations from the robot's own sensors and cameras. However, these "first person" maps inherit the robot's own limitations due to its embodiment or skillset, which may leave many aspects of the environment unexplored. For example, the robot might not be able to open drawers or access wall cabinets. In this sense, the map representation is not as complete, and requires a more capable robot to fill in the gaps. We narrow these blind spots in current methods by leveraging egocentric data captured as a human naturally explores a scene wearing Project Aria glasses, giving a way to directly transfer knowledge about articulation from the human to any deployable robot. We demonstrate that, by using simple heuristics, we can leverage egocentric data to recover models of articulate object parts, with quality comparable to those of state-of-the-art methods based on other input modalities. We also show how to integrate these models into 3D scene graph representations, leading to a better understanding of object dynamics and object-container relationships. We finally demonstrate that these articulated 3D scene graphs enhance a robot's ability to perform mobile manipulation tasks, showcasing an application where a Boston Dynamics Spot is tasked with retrieving concealed target items, given only the 3D scene graph as input.
>
---
#### [new 002] osmAG-Nav: A Hierarchical Semantic Topometric Navigation Stack for Robust Lifelong Indoor Autonomy
- **分类: cs.RO**

- **简介: 该论文属于室内自主导航任务，解决大规模多层环境下的导航效率与稳定性问题。提出osmAG-Nav系统，采用分层语义拓扑地图，提升规划速度与长期定位可靠性。**

- **链接: [https://arxiv.org/pdf/2603.28271](https://arxiv.org/pdf/2603.28271)**

> **作者:** Yongqi Zhang; Jiajie Zhang; Chengqian Li; Fujing Xie; Sören Schwertfeger
>
> **备注:** 42 pages, 10 figures
>
> **摘要:** The deployment of mobile robots in large-scale, multi-floor environments demands navigation systems that achieve spatial scalability without compromising local kinematic precision. Traditional navigation stacks, reliant on monolithic occupancy grid maps, face severe bottlenecks in storage efficiency, cross-floor reasoning, and long-horizon planning. To address these limitations, this paper presents osmAG-Nav, a complete, open-source ROS2 navigation stack built upon the hierarchical semantic topometric OpenStreetMap Area Graph (osmAG) map standard. The system follows a "System of Systems" architecture that decouples global topological reasoning from local metric execution. A Hierarchical osmAG planner replaces dense grid searches with an LCA-anchored pipeline on a passage-centric graph whose edge costs derive from local raster traversability rather than Euclidean distance, yielding low-millisecond planning on long campus-scale routes. A Rolling Window mechanism rasterizes a fixed-size local metric grid around the robot, keeping the local costmap memory footprint independent of the total mapped area, while a Segmented Execution strategy dispatches intermediate goals to standard ROS2 controllers for smooth handoffs. System robustness is reinforced by a structure-aware LiDAR localization framework that filters dynamic clutter against permanent architectural priors. Extensive experiments on a real-world multi-story indoor-outdoor campus (>11,025 m^2) show that, on the same-floor benchmark subset, osmAG-Nav delivers up to 7816x lower planning latency than a grid-based baseline on long routes while maintaining low path-length overhead and lifelong localization stability. A single-floor long-range robot mission further validates the integrated stack reliability. The full stack is released as modular ROS2 Lifecycle Nodes.
>
---
#### [new 003] SpatialPoint: Spatial-aware Point Prediction for Embodied Localization
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出SpatialPoint，解决具身定位任务，通过融合深度信息提升视觉语言模型在3D空间中的定位能力。**

- **链接: [https://arxiv.org/pdf/2603.26690](https://arxiv.org/pdf/2603.26690)**

> **作者:** Qiming Zhu; Zhirui Fang; Tianming Zhang; Chuanxiu Liu; Xiaoke Jiang; Lei Zhang
>
> **备注:** 19 pages, 12 figures, supplementary material included
>
> **摘要:** Embodied intelligence fundamentally requires a capability to determine where to act in 3D space. We formalize this requirement as embodied localization -- the problem of predicting executable 3D points conditioned on visual observations and language instructions. We instantiate embodied localization with two complementary target types: touchable points, surface-grounded 3D points enabling direct physical interaction, and air points, free-space 3D points specifying placement and navigation goals, directional constraints, or geometric relations. Embodied localization is inherently a problem of embodied 3D spatial reasoning -- yet most existing vision-language systems rely predominantly on RGB inputs, necessitating implicit geometric reconstruction that limits cross-scene generalization, despite the widespread adoption of RGB-D sensors in robotics. To address this gap, we propose SpatialPoint, a spatial-aware vision-language framework with careful design that integrates structured depth into a vision-language model (VLM) and generates camera-frame 3D coordinates. We construct a 2.6M-sample RGB-D dataset covering both touchable and air points QA pairs for training and evaluation. Extensive experiments demonstrate that incorporating depth into VLMs significantly improves embodied localization performance. We further validate SpatialPoint through real-robot deployment across three representative tasks: language-guided robotic arm grasping at specified locations, object placement to target destinations, and mobile robot navigation to goal positions.
>
---
#### [new 004] A Position Statement on Endovascular Models and Effectiveness Metrics for Mechanical Thrombectomy Navigation, on behalf of the Stakeholder Taskforce for AI-assisted Robotic Thrombectomy (START)
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人领域，旨在解决机械取栓技术的标准化问题，通过建立测试环境和评估指标，推动AI辅助机器人的发展。**

- **链接: [https://arxiv.org/pdf/2603.28129](https://arxiv.org/pdf/2603.28129)**

> **作者:** Harry Robertshaw; Anna Barnes; Phil Blakelock; Raphael Blanc; Robert Crossley; Rebecca Fahrig; Ameer E. Hassan; Benjamin Jackson; Lennart Karstensen; Neelam Kaur; Markus Kowarschik; Jeremy Lynch; Franziska Mathis-Ullrich; Dwight Meglan; Vitor Mendes Pereira; Mouloud Ourak; Matteo Pantano; S.M.Hadi Sadati; Alice Taylor-Gee; Tom Vercauteren; Phil White; Alejandro Granados; Thomas C. Booth
>
> **备注:** Published in Journal of the American Heart Association
>
> **摘要:** While we are making progress in overcoming infectious diseases and cancer; one of the major medical challenges of the mid-21st century will be the rising prevalence of stroke. Large vessels occlusions are especially debilitating, yet effective treatment (needed within hours to achieve best outcomes) remains limited due to geography. One solution for improving timely access to mechanical thrombectomy in geographically diverse populations is the deployment of robotic surgical systems. Artificial intelligence (AI) assistance may enable the upskilling of operators in this emerging therapeutic delivery approach. Our aim was to establish consensus frameworks for developing and validating AI-assisted robots for thrombectomy. Objectives included standardizing effectiveness metrics and defining reference testbeds across in silico, in vitro, ex vivo, and in vivo environments. To achieve this, we convened experts in neurointervention, robotics, data science, health economics, policy, statistics, and patient advocacy. Consensus was built through an incubator day, a Delphi process, and a final Position Statement. We identified that the four essential testbed environments each had distinct validation roles. Realism requirements vary: simpler testbeds should include realistic vessel anatomy compatible with guidewire and catheter use, while standard testbeds should incorporate deformable vessels. More advanced testbeds should include blood flow, pulsatility, and disease features. There are two macro-classes of effectiveness metrics: one for in silico, in vitro, and ex vivo stages focusing on technical navigation, and another for in vivo stages, focused on clinical outcomes. Patient safety is central to this technology's development. One requisite patient safety task needed now is to correlate in vitro measurements to in vivo complications.
>
---
#### [new 005] An End-to-end Flight Control Network for High-speed UAV Obstacle Avoidance based on Event-Depth Fusion
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于高速无人机避障任务，解决静态与动态障碍物感知不全的问题。通过融合深度与事件相机数据，设计端到端控制网络和高效规划器，提升避障可靠性。**

- **链接: [https://arxiv.org/pdf/2603.27181](https://arxiv.org/pdf/2603.27181)**

> **作者:** Dikai Shang; Jingyue Zhao; Shi Xu; Nanyang Ye; Lei Wang
>
> **备注:** 7 pages, 10 figures
>
> **摘要:** Achieving safe, high-speed autonomous flight in complex environments with static, dynamic, or mixed obstacles remains challenging, as a single perception modality is incomplete. Depth cameras are effective for static objects but suffer from motion blur at high speeds. Conversely, event cameras excel at capturing rapid motion but struggle to perceive static scenes. To exploit the complementary strengths of both sensors, we propose an end-to-end flight control network that achieves feature-level fusion of depth images and event data through a bidirectional crossattention module. The end-to-end network is trained via imitation learning, which relies on high-quality supervision. Building on this insight, we design an efficient expert planner using Spherical Principal Search (SPS). This planner reduces computational complexity from $O(n^2)$ to $O(n)$ while generating smoother trajectories, achieving over 80% success rate at 17m/s--nearly 20% higher than traditional planners. Simulation experiments show that our method attains a 70-80% success rate at 17 m/s across varied scenes, surpassing single-modality and unidirectional fusion models by 10-20%. These results demonstrate that bidirectional fusion effectively integrates event and depth information, enabling more reliable obstacle avoidance in complex environments with both static and dynamic objects.
>
---
#### [new 006] EBuddy: a workflow orchestrator for industrial human-machine collaboration
- **分类: cs.RO**

- **简介: 该论文提出EBuddy，一种用于工业人机协作的语音引导工作流协调器，解决专家知识难以规模化和操作质量下降的问题。通过有限状态机实现可解释的决策框架，提升协作效率与一致性。**

- **链接: [https://arxiv.org/pdf/2603.28579](https://arxiv.org/pdf/2603.28579)**

> **作者:** Michele Banfi; Rocco Felici; Stefano Baraldo; Oliver Avram; Anna Valente
>
> **摘要:** This paper presents EBuddy, a voice-guided workflow orchestrator for natural human-machine collaboration in industrial environments. EBuddy targets a recurrent bottleneck in tool-intensive workflows: expert know-how is effective but difficult to scale, and execution quality degrades when procedures are reconstructed ad hoc across operators and sessions. EBuddy operationalizes expert practice as a finite state machine (FSM) driven application that provides an interpretable decision frame at runtime (current state and admissible actions), so that spoken requests are interpreted within state-grounded constraints, while the system executes and monitors the corresponding tool interactions. Through modular workflow artifacts, EBuddy coordinates heterogeneous resources, including GUI-driven software and a collaborative robot, leveraging fully voice-based interaction through automatic speech recognition and intent understanding. An industrial pilot on impeller blade inspection and repair preparation for directed energy deposition (DED), realized by human-robot collaboration, shows substantial reductions in end-to-end process duration across onboarding, 3D scanning and processing, and repair program generation, while preserving repeatability and low operator burden.
>
---
#### [new 007] Uni-World VLA: Interleaved World Modeling and Planning for Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出Uni-World VLA模型，解决自动驾驶中环境建模与路径规划的协同问题。通过交替预测未来帧和规划动作，实现闭环控制，提升动态场景下的决策能力。**

- **链接: [https://arxiv.org/pdf/2603.27287](https://arxiv.org/pdf/2603.27287)**

> **作者:** Qiqi Liu; Huan Xu; Jingyu Li; Bin Sun; Zhihui Hao; Dangen She; Xiatian Zhu; Li Zhang
>
> **备注:** 22 pages, 8 figures. Submitted to ECCV 2026. Code will be released
>
> **摘要:** Autonomous driving requires reasoning about how the environment evolves and planning actions accordingly. Existing world-model-based approaches typically predict future scenes first and plan afterwards, resulting in open-loop imagination that may drift from the actual decision process. In this paper, we present Uni-World VLA, a unified vision-language-action (VLA) model that tightly interleaves future frame prediction and trajectory planning. Instead of generating a full world rollout before planning, our model alternates between predicting future frames and ego actions step by step, allowing planning decisions to be continuously conditioned on the imagined future observations. This interleaved generation forms a closed-loop interaction between world modeling and control, enabling more adaptive decision-making in dynamic traffic scenarios. In addition, we incorporate monocular depth information into frames to provide stronger geometric cues for world modeling, improving long-horizon scene prediction. Experiments on the NAVSIM benchmark show that our approach achieves competitive closed-loop planning performance while producing high-fidelity future frame predictions. These results demonstrate that tightly coupling world prediction and planning is a promising direction for scalable VLA driving systems.
>
---
#### [new 008] Tele-Catch: Adaptive Teleoperation for Dexterous Dynamic 3D Object Catching
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于动态物体抓取任务，解决纯远程操作在动态物体捕捉中因时间、姿态和力误差导致的失败问题。提出Tele-Catch框架，结合人类输入与自主策略，提升抓取精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.28427](https://arxiv.org/pdf/2603.28427)**

> **作者:** Weiguang Zhao; Junting Dong; Rui Zhang; Kailin Li; Qin Zhao; Kaizhu Huang
>
> **摘要:** Teleoperation is a key paradigm for transferring human dexterity to robots, yet most prior work targets objects that are initially static, such as grasping or manipulation. Dynamic object catch, where objects move before contact, remains underexplored. Pure teleoperation in this task often fails due to timing, pose, and force errors, highlighting the need for shared autonomy that combines human input with autonomous policies. To this end, we present Tele-Catch, a systematic framework for dexterous hand teleoperation in dynamic object catching. At its core, we design DAIM, a dynamics-aware adaptive integration mechanism that realizes shared autonomy by fusing glove-based teleoperation signals into the diffusion policy denoising process. It adaptively modulates control based on the interaction object state. To improve policy robustness, we introduce DP-U3R, which integrates unsupervised geometric representations from point cloud observations into diffusion policy learning, enabling geometry-aware decision making. Extensive experiments demonstrate that Tele-Catch significantly improves accuracy and robustness in dynamic catching tasks, while also exhibiting consistent gains across distinct dexterous hand embodiments and previously unseen object categories.
>
---
#### [new 009] Communications-Aware NMPC for Multi-Rotor Aerial Relay Networks Under Jamming Interference
- **分类: cs.RO**

- **简介: 该论文属于通信与控制协同优化任务，解决多旋翼无人机在干扰下的通信质量下降问题。通过结合轨迹规划与非线性预测控制，提升通信可靠性。**

- **链接: [https://arxiv.org/pdf/2603.28467](https://arxiv.org/pdf/2603.28467)**

> **作者:** Giuseppe Silano; Daniel Bonilla Licea; Davide Liuzza; Antonio Franchi; Martin Saska
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Multi-Rotor Aerial Vehicles (MRAVs) are increasingly used in communication-dependent missions where connectivity loss directly compromises task execution. Existing anti-jamming strategies often decouple motion from communication, overlooking that link quality depends on vehicle attitude and antenna orientation. In coplanar platforms, "tilt-to-translate" maneuvers can inadvertently align antenna nulls with communication partners, causing severe degradation under interference. This paper presents a modular communications-aware control framework that combines a high-level max-min trajectory generator with an actuator-level Nonlinear Model Predictive Controller (NMPC). The trajectory layer optimizes the weakest link under jamming, while the NMPC enforces vehicle dynamics, actuator limits, and antenna-alignment constraints. Antenna directionality is handled geometrically, avoiding explicit radiation-pattern parametrization. The method is evaluated in a relay scenario with an active jammer and compared across coplanar and tilted-propeller architectures. Results show a near two-order-of-magnitude increase in minimum end-to-end capacity, markedly reducing outage events, with moderate average-capacity gains. Tilted platforms preserve feasibility and link quality, whereas coplanar vehicles show recurrent degradation. These findings indicate that full actuation is a key enabler of reliable communications-aware operation under adversarial directional constraints.
>
---
#### [new 010] Probe-to-Grasp Manipulation Using Self-Sensing Pneumatic Variable-Stiffness Joints
- **分类: cs.RO**

- **简介: 该论文属于柔性抓取任务，旨在解决变形物体 stiffness 估计问题。通过自感知气动可变刚度关节，实现物体相对刚度的探测与抓取。**

- **链接: [https://arxiv.org/pdf/2603.27808](https://arxiv.org/pdf/2603.27808)**

> **作者:** Ngoc Duy Tran; Yeman Fan; Feng Dai; Khang Nguyen; Anh Nguyen; Hoang Hiep Ly; Tung D. Ta; Shigeru Chiba
>
> **摘要:** Grasping deformable objects with varying stiffness remains a significant challenge in robotics. Estimating the local stiffness of a target object is important for determining an optimal grasp pose that enables stable pickup without damaging the object. This paper presents a probe-to-grasp manipulation framework for estimating the relative stiffness of objects using a passive soft-rigid two-finger hybrid gripper equipped with self-sensing pneumatic variable-stiffness joints. Each finger of the gripper consists of two rigid links connected by a soft pneumatic ring placed at the joint, enabling both compliant interaction and controllable joint stiffness via internal pressurization. By measuring the pressure inside the pneumatic ring, we can estimate the interaction force during contact. Building on this, we propose a practical probing strategy to infer relative object stiffness by correlating the estimated normal force with known gripper closing displacement. We validate the self-sensing model through stiffness characterization experiments across bending angles and pressure ranges, and demonstrate stiffness-aware probing-and-grasping in real-life applications: selecting grasp locations on fruits with spatially varying stiffness. The proposed system offers a minimal, low-cost sensing approach for stiffness-aware soft manipulation while retaining probing and grasping capability.
>
---
#### [new 011] Transferability Through Cooperative Competitions
- **分类: cs.RO**

- **简介: 该论文属于机器人协作任务，旨在解决模块在异构系统间的可迁移性和可组合性问题。通过设计 coopetition 框架，促进团队合作与模块 reuse，提升系统兼容性与集成效率。**

- **链接: [https://arxiv.org/pdf/2603.27770](https://arxiv.org/pdf/2603.27770)**

> **作者:** Rodrigo Serra; Carlos Azevedo; André Silva; Kevin Alcedo; Quentin Rouxel; Peter So; Alejandro Suarez; Alin Albu-Schäeffer; Pedro U. Lima
>
> **备注:** Description of the cooperative competition concept, with a case study in EU project euROBIN, held in Nancy, November 2024
>
> **摘要:** This paper presents a novel framework for cooperative robotics competitions (coopetitions) that promote the transferability and composability of robotics modules, including software, hardware, and data, across heterogeneous robotic systems. The framework is designed to incentivize collaboration between teams through structured task design, shared infrastructure, and a royalty-based scoring system. As a case study, the paper details the implementation and outcomes of the first euROBIN Coopetition, held under the European Robotics and AI Network (euROBIN), which featured fifteen robotic platforms competing across Industrial, Service, and Outdoor domains. The study highlights the practical challenges of achieving module reuse in real-world scenarios, particularly in terms of integration complexity and system compatibility. It also examines participant performance, integration behavior, and team feedback to assess the effectiveness of the framework. The paper concludes with lessons learned and recommendations for future coopetitions, including improveme
>
---
#### [new 012] Agent-Driven Autonomous Reinforcement Learning Research: Iterative Policy Improvement for Quadruped Locomotion
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究四足机器人运动的自主强化学习，通过代理执行迭代策略优化，解决机器人在复杂地形中的运动控制问题。**

- **链接: [https://arxiv.org/pdf/2603.27416](https://arxiv.org/pdf/2603.27416)**

> **作者:** Nimesh Khandelwal; Shakti S. Gupta
>
> **摘要:** This paper documents a case study in agent-driven autonomous reinforcement learning research for quadruped locomotion. The setting was not a fully self-starting research system. A human provided high-level directives through an agentic coding environment, while an agent carried out most of the execution loop: reading code, diagnosing failures, editing reward and terrain configurations, launching and monitoring jobs, analyzing intermediate metrics, and proposing the next wave of experiments. Across more than 70 experiments organized into fourteen waves on a DHAV1 12-DoF quadruped in Isaac Lab, the agent progressed from early rough-terrain runs with mean reward around 7 to a best logged Wave 12 run, exp063, with velocity error 0.263 and 97\% timeout over 2000 iterations, independently reproduced five times across different GPUs. The archive also records several concrete autonomous research decisions: isolating PhysX deadlocks to terrain sets containing boxes and stair-like primitives, porting four reward terms from openly available reference implementations \cite{deeprobotics, rlsar}, correcting Isaac Sim import and bootstrapping issues, reducing environment count for diagnosis, terminating hung runs, and pivoting effort away from HIM after repeated terrain=0.0 outcomes. Relative to the AutoResearch paradigm \cite{autoresearch}, this case study operates in a more failure-prone robotics RL setting with multi-GPU experiment management and simulator-specific engineering constraints. The contribution is empirical and documentary: it shows that an agent can materially execute the iterative RL research loop in this domain with limited human intervention, while also making clear where human direction still shaped the agenda.
>
---
#### [new 013] HiFlow: Tokenization-Free Scale-Wise Autoregressive Policy Learning via Flow Matching
- **分类: cs.RO**

- **简介: 该论文属于视觉-运动策略学习任务，解决传统方法依赖离散动作分词导致的量化误差问题。提出HiFlow，直接生成连续动作，实现端到端训练。**

- **链接: [https://arxiv.org/pdf/2603.27281](https://arxiv.org/pdf/2603.27281)**

> **作者:** Daichi Yashima; Koki Seno; Shuhei Kurita; Yusuke Oda; Komei Sugiura
>
> **摘要:** Coarse-to-fine autoregressive modeling has recently shown strong promise for visuomotor policy learning, combining the inference efficiency of autoregressive methods with the global trajectory coherence of diffusion-based policies. However, existing approaches rely on discrete action tokenizers that map continuous action sequences to codebook indices, a design inherited from image generation where learned compression is necessary for high-dimensional pixel data. We observe that robot actions are inherently low-dimensional continuous vectors, for which such tokenization introduces unnecessary quantization error and a multi-stage training pipeline. In this work, we propose Hierarchical Flow Policy (HiFlow), a tokenization-free coarse-to-fine autoregressive policy that operates directly on raw continuous actions. HiFlow constructs multi-scale continuous action targets from each action chunk via simple temporal pooling. Specifically, it averages contiguous action windows to produce coarse summaries that are refined at finer temporal resolutions. The entire model is trained end-to-end in a single stage, eliminating the need for a separate tokenizer. Experiments on MimicGen, RoboTwin 2.0, and real-world environments demonstrate that HiFlow consistently outperforms existing methods including diffusion-based and tokenization-based autoregressive policies.
>
---
#### [new 014] ManipArena: Comprehensive Real-world Evaluation of Reasoning-Oriented Generalist Robot Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出ManipArena，用于评估机器人推理导向的通用操作能力。解决真实场景下评估标准缺失的问题，通过20个任务和多维度测试，提升评估的公平性与真实性。**

- **链接: [https://arxiv.org/pdf/2603.28545](https://arxiv.org/pdf/2603.28545)**

> **作者:** Yu Sun; Meng Cao; Ping Yang; Rongtao Xu; Yunxiao Yan; Runze Xu; Liang Ma; Roy Gan; Andy Zhai; Qingxuan Chen; Zunnan Xu; Hao Wang; Jincheng Yu; Lucy Liang; Qian Wang; Ivan Laptev; Ian D Reid; Xiaodan Liang
>
> **备注:** Technical report for CVPR 2026 Challenge ManipArena
>
> **摘要:** Vision-Language-Action (VLA) models and world models have recently emerged as promising paradigms for general-purpose robotic intelligence, yet their progress is hindered by the lack of reliable evaluation protocols that reflect real-world deployment. Existing benchmarks are largely simulator-centric, which provide controllability but fail to capture the reality gap caused by perception noise, complex contact dynamics, hardware constraints, and system latency. Moreover, fragmented real-world evaluations across different robot platforms prevent fair and reproducible comparison. To address these challenges, we introduce ManipArena, a standardized evaluation framework designed to bridge simulation and real-world execution. ManipArena comprises 20 diverse tasks across 10,812 expert trajectories emphasizing reasoning-oriented manipulation tasks requiring semantic and spatial reasoning, supports multi-level generalization through controlled out-of-distribution settings, and incorporates long-horizon mobile manipulation beyond tabletop scenarios. The framework further provides rich sensory diagnostics, including low-level motor signals, and synchronized real-to-sim environments constructed via high-quality 3D scanning. Together, these features enable fair, realistic, and reproducible evaluation for both VLA and world model approaches, providing a scalable foundation for diagnosing and advancing embodied intelligence systems.
>
---
#### [new 015] Safety Guardrails in the Sky: Realizing Control Barrier Functions on the VISTA F-16 Jet
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自主系统安全控制任务，旨在解决动态环境下的安全执行问题。提出Guardrails机制，结合控制屏障函数，确保飞行器安全操作。**

- **链接: [https://arxiv.org/pdf/2603.27912](https://arxiv.org/pdf/2603.27912)**

> **作者:** Andrew W. Singletary; Max H. Cohen; Tamas G. Molnar; Aaron D. Ames
>
> **摘要:** The advancement of autonomous systems -- from legged robots to self-driving vehicles and aircraft -- necessitates executing increasingly high-performance and dynamic motions without ever putting the system or its environment in harm's way. In this paper, we introduce Guardrails -- a novel runtime assurance mechanism that guarantees dynamic safety for autonomous systems, allowing them to safely evolve on the edge of their operational domains. Rooted in the theory of control barrier functions, Guardrails offers a control strategy that carefully blends commands from a human or AI operator with safe control actions to guarantee safe behavior. To demonstrate its capabilities, we implemented Guardrails on an F-16 fighter jet and conducted flight tests where Guardrails supervised a human pilot to enforce g-limits, altitude bounds, geofence constraints, and combinations thereof. Throughout extensive flight testing, Guardrails successfully ensured safety, keeping the pilot in control when safe to do so and minimally modifying unsafe pilot inputs otherwise.
>
---
#### [new 016] Dynamic Lookahead Distance via Reinforcement Learning-Based Pure Pursuit for Autonomous Racing
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文属于自主赛车任务，解决纯追踪算法中固定前瞻距离适应性差的问题。通过结合强化学习与纯追踪，动态调整前瞻距离，提升赛道表现。**

- **链接: [https://arxiv.org/pdf/2603.28625](https://arxiv.org/pdf/2603.28625)**

> **作者:** Mohamed Elgouhary; Amr S. El-Wakeel
>
> **摘要:** Pure Pursuit (PP) is a widely used path-tracking algorithm in autonomous vehicles due to its simplicity and real-time performance. However, its effectiveness is sensitive to the choice of lookahead distance: shorter values improve cornering but can cause instability on straights, while longer values improve smoothness but reduce accuracy in curves. We propose a hybrid control framework that integrates Proximal Policy Optimization (PPO) with the classical Pure Pursuit controller to adjust the lookahead distance dynamically during racing. The PPO agent maps vehicle speed and multi-horizon curvature features to an online lookahead command. It is trained using Stable-Baselines3 in the F1TENTH Gym simulator with a KL penalty and learning-rate decay for stability, then deployed in a ROS2 environment to guide the controller. Experiments in simulation compare the proposed method against both fixed-lookahead Pure Pursuit and an adaptive Pure Pursuit baseline. Additional real-car experiments compare the learned controller against a fixed-lookahead Pure Pursuit controller. Results show that the learned policy improves lap-time performance and repeated lap completion on unseen tracks, while also transferring zero-shot to hardware. The learned controller adapts the lookahead by increasing it on straights and reducing it in curves, demonstrating effectiveness in augmenting a classical controller by online adaptation of a single interpretable parameter. On unseen tracks, the proposed method achieved 33.16 s on Montreal and 46.05 s on Yas Marina, while tolerating more aggressive speed-profile scaling than the baselines and achieving the best lap times among the tested settings. Initial real-car experiments further support sim-to-real transfer on a 1:10-scale autonomous racing platform
>
---
#### [new 017] Off-Axis Compliant RCM Joint with Near-Isotropic Stiffness and Minimal Parasitic Error
- **分类: cs.RO**

- **简介: 该论文属于机械设计任务，旨在解决神经内镜操作中RCM关节的刚度各向同性和寄生运动问题。通过优化结构设计与仿真验证，实现高精度、小误差的运动控制。**

- **链接: [https://arxiv.org/pdf/2603.28240](https://arxiv.org/pdf/2603.28240)**

> **作者:** Federico Mariano; Elena De Momi; Giovanni Berselli; Jovana Jovanova; Just L. Herder; Leonardo S. Mattos
>
> **摘要:** This paper presents an off-axis, monolithic compliant Remote Center of Motion (RCM) joint for neuroendoscopic manipulation, combining near-isotropic stiffness with minimal parasitic motion. Based on the Tetra II concept, the end-effector is placed outside the tetrahedral flexure to improve line of sight, facilitate sterilization, and allow rapid tool release. Design proceeds in two stages: mobility panels are sized with a compliance-based isotropy objective, then constraining panels are synthesized through finite-element feasibility exploration to trade stiffness isotropy against RCM drift. The joint is modeled with beam elements and validated via detailed finite-element analyses, including fatigue-bounded stress constraints. A PA12 prototype is fabricated by selective laser sintering and characterized on a benchtop: a 2 N radial load is applied at the end-effector while a 6-DOF electromagnetic sensor records pose. The selected configuration produces a stiffness-ellipse principal axis ratio (PAR) of 1.37 and a parasitic-to-useful rotation ratio (PRR) of 0.63%. Under a 4.5° commanded rotation, the predicted RCM drift remains sub-millimetric (0.015-0.172 mm). Fatigue analysis predicts a usable rotational workspace of 12.1°-34.4° depending on direction. Experiments reproduce the simulated directional stiffness trend with typical deviations of 6-30%, demonstrating a compact, fabrication-ready RCM module for constrained surgical access.
>
---
#### [new 018] SutureAgent: Learning Surgical Trajectories via Goal-conditioned Offline RL in Pixel Space
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于手术导航任务，解决手术缝合轨迹预测问题。通过目标条件的离线强化学习，建模像素级连续动作序列，提升轨迹预测精度。**

- **链接: [https://arxiv.org/pdf/2603.26720](https://arxiv.org/pdf/2603.26720)**

> **作者:** Huanrong Liu; Chunlin Tian; Tongyu Jia; Tailai Zhou; Qin Liu; Yu Gao; Yutong Ban; Yun Gu; Guy Rosman; Xin Ma; Qingbiao Li
>
> **摘要:** Predicting surgical needle trajectories from endoscopic video is critical for robot-assisted suturing, enabling anticipatory planning, real-time guidance, and safer motion execution. Existing methods that directly learn motion distributions from visual observations tend to overlook the sequential dependency among adjacent motion steps. Moreover, sparse waypoint annotations often fail to provide sufficient supervision, further increasing the difficulty of supervised or imitation learning methods. To address these challenges, we formulate image-based needle trajectory prediction as a sequential decision-making problem, in which the needle tip is treated as an agent that moves step by step in pixel space. This formulation naturally captures the continuity of needle motion and enables the explicit modeling of physically plausible pixel-wise state transitions over time. From this perspective, we propose SutureAgent, a goal-conditioned offline reinforcement learning framework that leverages sparse annotations to dense reward signals via cubic spline interpolation, encouraging the policy to exploit limited expert guidance while exploring plausible future motion paths. SutureAgent encodes variable-length clips using an observation encoder to capture both local spatial cues and long-range temporal dynamics, and autoregressively predicts future waypoints through actions composed of discrete directions and continuous magnitudes. To enable stable offline policy optimization from expert demonstrations, we adopt Conservative Q-Learning with Behavioral Cloning regularization. Experiments on a new kidney wound suturing dataset containing 1,158 trajectories from 50 patients show that SutureAgent reduces Average Displacement Error by 58.6% compared with the strongest baseline, demonstrating the effectiveness of modeling needle trajectory prediction as pixel-level sequential action learning.
>
---
#### [new 019] Cost-Matching Model Predictive Control for Efficient Reinforcement Learning in Humanoid Locomotion
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于人形机器人运动控制任务，解决高效强化学习中的动作价值函数近似问题。通过参数化MPC框架，利用轨迹数据优化成本函数，提升运动性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.28243](https://arxiv.org/pdf/2603.28243)**

> **作者:** Wenqi Cai; Kyriakos G. Vamvoudakis; Sébastien Gros; Anthony Tzes
>
> **摘要:** In this paper, we propose a cost-matching approach for optimal humanoid locomotion within a Model Predictive Control (MPC)-based Reinforcement Learning (RL) framework. A parameterized MPC formulation with centroidal dynamics is trained to approximate the action-value function obtained from high-fidelity closed-loop data. Specifically, the MPC cost-to-go is evaluated along recorded state-action trajectories, and the parameters are updated to minimize the discrepancy between MPC-predicted values and measured returns. This formulation enables efficient gradient-based learning while avoiding the computational burden of repeatedly solving the MPC problem during training. The proposed method is validated in simulation using a commercial humanoid platform. Results demonstrate improved locomotion performance and robustness to model mismatch and external disturbances compared with manually tuned baselines.
>
---
#### [new 020] Design of an In-Pipe Robot with Contact-Angle-Guided Kinematic Decoupling for Crosstalk-Suppressed Locomotion
- **分类: cs.RO**

- **简介: 该论文属于管道机器人设计任务，解决传统机器人在管道中运动时存在的耦合问题。通过结构设计实现驱动与转向解耦，提升运动稳定性与适应性。**

- **链接: [https://arxiv.org/pdf/2603.27245](https://arxiv.org/pdf/2603.27245)**

> **作者:** Min Yang; Yang Tian; Longchuang Li; Jun Ma; Shugen Ma
>
> **摘要:** In-pipe inspection robots must traverse confined pipeline networks with elbows and three-dimensional fittings, requiring both reliable axial traction and rapid rolling reorientation for posture correction. In compact V-shaped platforms, these functions often rely on shared contacts or indirect actuation, which introduces strong kinematic coupling and makes performance sensitive to geometry and friction variations. This paper presents a V-shaped in-pipe robot with a joint-axis-and-wheel-separation layout that provides two physically independent actuation channels, with all-wheel-drive propulsion and motorized rolling reorientation while using only two motors. To make the decoupling mechanism explicit and designable, we formulate an actuation transmission matrix and identify the spherical-wheel contact angle as the key geometric variable governing the dominant roll-to-propulsion leakage and roll-channel efficiency. A geometric transmission analysis maps mounting parameters to the contact angle, leakage, and efficiency, yielding a structural guideline for suppressing crosstalk by driving the contact angle toward zero. A static stability model further provides a stability-domain map for selecting torsion-spring stiffness under friction uncertainty to ensure vertical-pipe stability with a margin. Experiments validate the decoupling effect, where during high-dynamic rolling in a vertical pipe, the propulsion torque remains nearly invariant. On a multi-material testbed including out-of-plane double elbows, the robot achieved a 100% success rate in more than 10 independent round-trip trials.
>
---
#### [new 021] S3KF: Spherical State-Space Kalman Filtering for Panoramic 3D Multi-Object Tracking
- **分类: cs.RO**

- **简介: 该论文属于全景3D多目标跟踪任务，解决广域场景下目标关联与定位问题，提出S³KF框架融合LiDAR与全景相机数据实现高精度实时跟踪。**

- **链接: [https://arxiv.org/pdf/2603.27534](https://arxiv.org/pdf/2603.27534)**

> **作者:** Zhongyuan Liu; Shaonan Yu; Jianping Li; Pengfei Wan; Xinhang Xu; Pengfei Wang; Maggie Y. Gao; Lihua Xie
>
> **摘要:** Panoramic multi-object tracking is important for industrial safety monitoring, wide-area robotic perception, and infrastructure-light deployment in large workspaces. In these settings, the sensing system must provide full-surround coverage, metric geometric cues, and stable target association under wide field-of-view distortion and occlusion. Existing image-plane trackers are tightly coupled to the camera projection and become unreliable in panoramic imagery, while conventional Euclidean 3D formulations introduce redundant directional parameters and do not naturally unify angular, scale, and depth estimation. In this paper, we present $\mathbf{S^3KF}$, a panoramic 3D multi-object tracking framework built on a motorized rotating LiDAR and a quad-fisheye camera rig. The key idea is a geometry-consistent state representation on the unit sphere $\mathbb{S}^2$, where object bearing is modeled by a two-degree-of-freedom tangent-plane parameterization and jointly estimated with box scale and depth dynamics. Based on this state, we derive an extended spherical Kalman filtering pipeline that fuses panoramic camera detections with LiDAR depth observations for multimodal tracking. We further establish a map-based ground-truth generation pipeline using wearable localization devices registered to a shared global LiDAR map, enabling quantitative evaluation without motion-capture infrastructure. Experiments on self-collected real-world sequences show decimeter-level planar tracking accuracy, improved identity continuity over a 2D panoramic baseline in dynamic scenes, and real-time onboard operation on a Jetson AGX Orin platform. These results indicate that the proposed framework is a practical solution for panoramic perception and industrial-scale multi-object this http URL project page can be found at this https URL.
>
---
#### [new 022] Flip Stunts on Bicycle Robots using Iterative Motion Imitation
- **分类: cs.RO**

- **简介: 该论文研究自行车机器人的前空翻动作，通过迭代运动模仿方法训练策略，解决不可行参考轨迹的模仿问题，实现真实世界中的敏捷翻转行为。**

- **链接: [https://arxiv.org/pdf/2603.27944](https://arxiv.org/pdf/2603.27944)**

> **作者:** Jeonghwan Kim; Shamel Fahmi; Seungeun Rho; Sehoon Ha; Gabriel Nelson
>
> **备注:** 8 Pages, Accepted to the IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** This work demonstrates a front-flip on bicycle robots via reinforcement learning, particularly by imitating reference motions that are infeasible and imperfect. To address this, we propose Iterative Motion Imitation(IMI), a method that iteratively imitates trajectories generated by prior policy rollouts. Starting from an initial reference that is kinematically or dynamically infeasible, IMI helps train policies that lead to feasible and agile behaviors. We demonstrate our method on Ultra-Mobility Vehicle (UMV), a bicycle robot that is designed to enable agile behaviors. From a self-colliding table-to-ground flip reference generated by a model-based controller, we are able to train policies that enable ground-to-ground and ground-to-table front-flips. We show that compared to a single-shot motion imitation, IMI results in policies with higher success rates and can transfer robustly to the real world. To our knowledge, this is the first unassisted acrobatic flip behavior on such a platform.
>
---
#### [new 023] A Deep Reinforcement Learning Framework for Closed-loop Guidance of Fish Schools via Virtual Agents
- **分类: cs.RO; cs.LG; q-bio.PE**

- **简介: 该论文属于生物群体引导任务，旨在通过深度强化学习实现对鱼群的闭环控制。研究设计虚拟代理，利用PPO算法训练并验证其在真实实验中的引导效果。**

- **链接: [https://arxiv.org/pdf/2603.28200](https://arxiv.org/pdf/2603.28200)**

> **作者:** Takato Shibayama; Hiroaki Kawashima
>
> **备注:** 18 pages, 8 figures
>
> **摘要:** Guiding collective motion in biological groups is a fundamental challenge in understanding social interaction rules and developing automated systems for animal management. In this study, we propose a deep reinforcement learning (RL) framework for the closed-loop guidance of fish schools using virtual agents. These agents are controlled by policies trained via Proximal Policy Optimization (PPO) in simulation and deployed in physical experiments with rummy-nose tetras (Petitella bleheri), enabling real-time interaction between artificial agents and live individuals. To cope with the stochastic behavior of live individuals, we design a composite reward function to balance directional guidance with social cohesion. Our systematic evaluation of visual parameters shows that a white background and larger stimulus sizes maximize guidance efficacy in physical trials. Furthermore, evaluation across group sizes revealed that while the system demonstrates effective guidance for groups of five individuals, this capability markedly degrades as group size increases to eight. This study highlights the potential of deep RL for automated guidance of biological collectives and identifies challenges in maintaining artificial influence in larger groups.
>
---
#### [new 024] Reasoning Systems for Semantic Navigation in Mobile Robots
- **分类: cs.RO**

- **简介: 该论文属于语义导航任务，解决移动机器人环境理解与路径规划问题。通过构建语义表示和推理系统，提出两种基于本体的解决方案，并在机器人上验证。**

- **链接: [https://arxiv.org/pdf/2603.26753](https://arxiv.org/pdf/2603.26753)**

> **作者:** Jonathan Crespo; Ramón Barber; O. M. Mozos; Daniel Beßler; Michael Beetz
>
> **备注:** This is the authors' manuscript. The final published article is available at this https URL
>
> **摘要:** Semantic navigation is the navigation paradigm in which environmental semantic concepts and their relationships are taken into account to plan the route of a mobile robot. This paradigm facilitates the interaction with humans and the understanding of human environments in terms of navigation goals and tasks. At the high level, a semantic navigation system requires two main components: a semantic representation of the environment, and a reasoner system. This paper is focused on develop a model of the environment using semantic concepts. This paper presents two solutions for the semantic navigation paradigm. Both systems implement an ontological model. Whilst the first one uses a relational database, the second one is based on KnowRob. Both systems have been integrated in a semantic navigator. We compare both systems at the qualitative and quantitative levels, and present an implementation on a mobile robot as a proof of concept.
>
---
#### [new 025] ContraMap: Contrastive Uncertainty Mapping for Robot Environment Representation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出ContraMap，用于机器人环境建模中的不确定性映射。解决感知可靠性问题，通过对比学习实现实时环境预测与不确定性估计。**

- **链接: [https://arxiv.org/pdf/2603.27632](https://arxiv.org/pdf/2603.27632)**

> **作者:** Chi Cuong Le; Weiming Zhi
>
> **摘要:** Reliable robot perception requires not only predicting scene structure, but also identifying where predictions should be treated as unreliable due to sparse or missing observations. We present ContraMap, a contrastive continuous mapping method that augments kernel-based discriminative maps with an explicit uncertainty class trained using synthetic noise samples. This formulation treats unobserved regions as a contrastive class, enabling joint environment prediction and spatial uncertainty estimation in real time without Bayesian inference. Under a simple mixture-model view, we show that the probability assigned to the uncertainty class is a monotonic function of a distance-aware uncertainty surrogate. Experiments in 2D occupancy mapping, 3D semantic mapping, and tabletop scene reconstruction show that ContraMap preserves mapping quality, produces spatially coherent uncertainty estimates, and is substantially more efficient than Bayesian kernelmap baselines.
>
---
#### [new 026] Where-to-Learn: Analytical Policy Gradient Directed Exploration for On-Policy Robotic Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人强化学习任务，旨在解决有效探索问题。通过分析策略梯度，引导智能体向高奖励区域学习，提升策略效率。**

- **链接: [https://arxiv.org/pdf/2603.27317](https://arxiv.org/pdf/2603.27317)**

> **作者:** Leixin Chang; Xinchen Yao; Ben Liu; Liangjing Yang; Hua Chen
>
> **备注:** 8 pages, 10 figures
>
> **摘要:** On-policy reinforcement learning (RL) algorithms have demonstrated great potential in robotic control, where effective exploration is crucial for efficient and high-quality policy learning. However, how to encourage the agent to explore the better trajectories efficiently remains a challenge. Most existing methods incentivize exploration by maximizing the policy entropy or encouraging novel state visiting regardless of the potential state value. We propose a new form of directed exploration that uses analytical policy gradients from a differentiable dynamics model to inject task-aware, physics-guided guidance, thereby steering the agent towards high-reward regions for accelerated and more effective policy learning.
>
---
#### [new 027] UMI-Underwater: Learning Underwater Manipulation without Underwater Teleoperation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于水下机器人抓取任务，解决水下视觉差和数据稀缺问题。通过自监督数据收集和基于深度的表征迁移，实现水下抓取的零样本迁移与泛化。**

- **链接: [https://arxiv.org/pdf/2603.27012](https://arxiv.org/pdf/2603.27012)**

> **作者:** Hao Li; Long Yin Chung; Jack Goler; Ryan Zhang; Xiaochi Xie; Huy Ha; Shuran Song; Mark Cutkosky
>
> **摘要:** Underwater robotic grasping is difficult due to degraded, highly variable imagery and the expense of collecting diverse underwater demonstrations. We introduce a system that (i) autonomously collects successful underwater grasp demonstrations via a self-supervised data collection pipeline and (ii) transfers grasp knowledge from on-land human demonstrations through a depth-based affordance representation that bridges the on-land-to-underwater domain gap and is robust to lighting and color shift. An affordance model trained on on-land handheld demonstrations is deployed underwater zero-shot via geometric alignment, and an affordance-conditioned diffusion policy is then trained on underwater demonstrations to generate control actions. In pool experiments, our approach improves grasping performance and robustness to background shifts, and enables generalization to objects seen only in on-land data, outperforming RGB-only baselines. Code, videos, and additional results are available at this https URL.
>
---
#### [new 028] Why Cognitive Robotics Matters: Lessons from OntoAgent and LLM Deployment in HARMONIC for Safety-Critical Robot Teaming
- **分类: cs.RO**

- **简介: 论文探讨认知机器人在安全关键任务中的重要性，解决LLM在长期规划中缺乏自我评估的问题。通过HARMONIC架构对比OntoAgent与LLMs，发现LLMs在诊断和决策上存在缺陷。**

- **链接: [https://arxiv.org/pdf/2603.26730](https://arxiv.org/pdf/2603.26730)**

> **作者:** Sanjay Oruganti; Sergei Nirenburg; Marjorie McShane; Jesse English; Michael Roberts; Christian Arndt; Ramviyas Parasuraman; Luis Sentis
>
> **摘要:** Deploying embodied AI agents in the physical world demands cognitive capabilities for long-horizon planning that execute reliably, deterministically, and transparently. We present HARMONIC, a cognitive-robotic architecture that pairs OntoAgent, a content-centric cognitive architecture providing metacognitive self-monitoring, domain-grounded diagnosis, and consequence-based action selection over ontologically structured knowledge, with a modular reactive tactical layer. HARMONIC's modular design enables a functional evaluation of whether LLMs can replicate OntoAgent's cognitive capabilities, evaluated within the same robotic system under identical conditions. Six LLMs spanning frontier and efficient tiers replace OntoAgent in a collaborative maintenance scenario under native and knowledge-equalized conditions. Results reveal that LLMs do not consistently assess their own knowledge state before acting, causing downstream failures in diagnostic reasoning and action selection. These deficits persist even with equivalent procedural knowledge, indicating the issues are architectural rather than knowledge-based. These findings support the design of physically embodied systems in which cognitive architectures retain primary authority for reasoning, owing to their deterministic and transparent characteristics.
>
---
#### [new 029] RAD-LAD: Rule and Language Grounded Autonomous Driving in Real-Time
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出LAD和RAD两种方法，解决自动驾驶中的实时决策问题。LAD实现快速运动规划，RAD提升规则性，二者结合提升系统可靠性与适应性。**

- **链接: [https://arxiv.org/pdf/2603.28522](https://arxiv.org/pdf/2603.28522)**

> **作者:** Anurag Ghosh; Srinivasa Narasimhan; Manmohan Chandraker; Francesco Pittaluga
>
> **摘要:** We present LAD, a real-time language--action planner with an interruptible architecture that produces a motion plan in a single forward pass (~20 Hz) or generates textual reasoning alongside a motion plan (~10 Hz). LAD is fast enough for real-time closed-loop deployment, achieving ~3x lower latency than prior driving language models while setting a new learning-based state of the art on nuPlan Test14-Hard and InterPlan. We also introduce RAD, a rule-based planner designed to address structural limitations of PDM-Closed. RAD achieves state-of-the-art performance among rule-based planners on nuPlan Test14-Hard and InterPlan. Finally, we show that combining RAD and LAD enables hybrid planning that captures the strengths of both approaches. This hybrid system demonstrates that rules and learning provide complementary capabilities: rules support reliable maneuvering, while language enables adaptive and explainable decision-making.
>
---
#### [new 030] ROSClaw: An OpenClaw ROS 2 Framework for Agentic Robot Control and Interaction
- **分类: cs.RO; cs.HC**

- **简介: 该论文提出ROSClaw，解决机器人与基础模型集成问题，通过ROS 2框架实现模型与机器人的灵活交互，提升安全性与可移植性。**

- **链接: [https://arxiv.org/pdf/2603.26997](https://arxiv.org/pdf/2603.26997)**

> **作者:** Irvin Steve Cardenas; Marcus Anthony Arnett; Natalie Catherine Yeo; Lucky Sah; Jong-Hoon Kim
>
> **摘要:** Foundation models can endow robots with open-ended reasoning, language understanding, and adaptive planning, yet connecting a model to a physical robot today requires bespoke integration that couples perception, actuation, and safety to a single model and platform. We present ROSClaw, a model-agnostic executive layer that integrates the OpenClaw agent runtime with ROS 2, enabling any foundation model to perceive, reason about, and act on any ROS-enabled robot through (i) dynamic capability discovery with standardized affordance injection, (ii) multimodal observation normalization, (iii) pre-execution action validation within a configurable safety envelope, and (iv) structured audit logging. Swapping model backends or robot platforms is a configuration change; tool schemas, safety enforcement, and provenance logging remain invariant. We deploy ROSClaw on three platforms (wheeled, quadruped, humanoid) with four foundation-model backends. Under this controlled substrate, models exhibit up to 4.8 x differences in out-of-policy action proposal rates (3.4 x among frontier models alone) and produce qualitatively distinct physical behaviors from identical commands. A cross-framework parity protocol against ROSA confirms that executive-layer design, not just prompt wording, significantly affects both task completion and safety behavior, establishing ROSClaw as both practical agentic-robot infrastructure and a reproducible measurement instrument for embodied AI.
>
---
#### [new 031] ProgressVLA: Progress-Guided Diffusion Policy for Vision-Language Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于视觉-语言机器人操作任务，旨在解决现有模型缺乏进度感知的问题。通过引入进展估计和可微进展引导，提升长时序任务的执行效果。**

- **链接: [https://arxiv.org/pdf/2603.27670](https://arxiv.org/pdf/2603.27670)**

> **作者:** Hongyu Yan; Qiwei Li; Jiaolong Yang; Yadong Mu
>
> **摘要:** Most existing vision-language-action (VLA) models for robotic manipulation lack progress awareness, typically relying on hand-crafted heuristics for task termination. This limitation is particularly severe in long-horizon tasks involving cascaded sub-goals. In this work, we investigate the estimation and integration of task progress, proposing a novel model named {\textbf \vla}. Our technical contributions are twofold: (1) \emph{robust progress estimation}: We pre-train a progress estimator on large-scale, unsupervised video-text robotic datasets. This estimator achieves a low prediction residual (0.07 on a scale of $[0, 1]$) in simulation and demonstrates zero-shot generalization to unseen real-world samples, and (2) \emph{differentiable progress guidance}: We introduce an inverse dynamics world model that maps predicted action tokens into future latent visual states. These latents are then processed by the progress estimator; by applying a maximal progress regularization, we establish a differentiable pipeline that provides progress-piloted guidance to refine action tokens. Extensive experiments on the CALVIN and LIBERO benchmarks, alongside real-world robot deployment, consistently demonstrate substantial improvements in success rates and generalization over strong baselines.
>
---
#### [new 032] Surface-Constrained Offline Warping with Contact-Aware Online Pose Projection for Safe Robotic Trajectory Execution
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人轨迹执行任务，解决非平面表面轨迹规划中的几何不一致问题。通过分阶段方法实现轨迹的表面适应与在线安全约束，提升轨迹稳定性与接触保持。**

- **链接: [https://arxiv.org/pdf/2603.26711](https://arxiv.org/pdf/2603.26711)**

> **作者:** Farong Wang; Sai Swaminathan; Fei Liu
>
> **备注:** 7 pages, 7 figures. Submitted to IROS 2026
>
> **摘要:** Robotic manipulation tasks that require repeated tool motion along curved surfaces frequently arise in surface finishing, inspection, and guided interaction. In practice, nominal motion primitives are often designed independently of the deployment surface and later reused across varying geometries. Directly tiling such primitives onto nonplanar surfaces introduces geometric inconsistencies, leading to interpenetration, orientation discontinuities, and cumulative drift over repeated cycles. We present a two-stage framework that separates geometric embedding from execution-level regulation. An offline surface-constrained warping operator embeds a nominal periodic primitive onto curved surfaces through asymmetric diffeomorphic deformation of dual-track waypoints and axis-consistent orientation completion, producing a surface-adapted reference trajectory. An online contact-aware projection operator then enforces bounded deviation relative to this reference using FSR-driven disturbance adaptation and a conic orientation safety constraint. Experiments across multiple analytic surface families and real-robot validation on a sinusoidal surface demonstrate improved geometric continuity, reduced large orientation jumps, and robust contact maintenance compared with direct tiling. These results show that decoupling offline geometric remapping from lightweight online projection enables stable and repeatable surface-embedded trajectory execution under sensor-lite feedbacks.
>
---
#### [new 033] MetaTune: Adjoint-based Meta-tuning via Robotic Differentiable Dynamics
- **分类: cs.RO**

- **简介: 该论文提出MetaTune，解决机器人控制中控制器与观测器参数联合调优难题，通过可微分闭环元学习实现自适应增益调整。**

- **链接: [https://arxiv.org/pdf/2603.27313](https://arxiv.org/pdf/2603.27313)**

> **作者:** Xiexin Peng; Bingheng Wang; Tao Zhang; Ying Zheng
>
> **摘要:** Disturbance observer-based control has shown promise in robustifying robotic systems against uncertainties. However, tuning such systems remains challenging due to the strong coupling between controller gains and observer parameters. In this work, we propose MetaTune, a unified framework for joint auto-tuning of feedback controllers and disturbance observers through differentiable closed-loop meta-learning. MetaTune integrates a portable neural policy with physics-informed gradients derived from differentiable system dynamics, enabling adaptive gain across tasks and operating conditions. We develop an adjoint method that efficiently computes the meta-gradients with respect to adaptive gains backward in time to directly minimize the cost-to-go. Compared to existing forward methods, our approach reduces the computational complexity to be linear in the data horizon. Experimental results on quadrotor control show that MetaTune achieves consistent improvements over state-of-the-art differentiable tuning methods while reducing gradient computation time by more than 50 percent. In high-fidelity PX4-Gazebo hardware-in-the-loop simulation, the learned adaptive policy yields 15-20 percent average tracking error reduction at aggressive flight speeds and up to 40 percent improvement under strong disturbances, while demonstrating zero-shot sim-to-sim transfer without fine-tuning.
>
---
#### [new 034] StreamingVLA: Streaming Vision-Language-Action Model with Action Flow Matching and Adaptive Early Observation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出StreamingVLA，解决VLA模型在边缘设备上的效率问题。通过动作流匹配和自适应观测机制，实现各阶段并行，提升执行速度与流畅性。**

- **链接: [https://arxiv.org/pdf/2603.28565](https://arxiv.org/pdf/2603.28565)**

> **作者:** Yiran Shi; Dongqi Guo; Tianchen Zhao; Feng Gao; Liangzhi Shi; Chao Yu; ZhiJian Mo; Qihua Xiao; XiaoShuai Peng; Qingmin Liao; Yu Wang
>
> **摘要:** Vision-language-action (VLA) models have demonstrated exceptional performance in natural language-driven perception and control. However, the high computational cost of VLA models poses significant efficiency challenges, particularly for resource-constrained edge platforms in real-world deployments. However, since different stages of VLA (observation, action generation and execution) must proceed sequentially, and wait for the completion of the preceding stage, the system suffers from frequent halting and high latency. To address this, We conduct a systematic analysis to identify the challenges for fast and fluent generation, and propose enabling VLAs with the ability to asynchronously parallelize across VLA stages in a "streaming" manner. First, we eliminate the reliance on action chunking and adopt action flow matching, which learns the trajectory of action flows rather than denoising chunk-wise actions. It overlaps the latency of action generation and execution. Second, we design an action saliency-aware adaptive observation mechanism, thereby overlapping the latency of execution and observation. Without sacrificing performance, StreamingVLA achieves substantial speedup and improves the fluency of execution. It achieves a 2.4 $\times$ latency speedup and reduces execution halting by 6.5 $\times$.
>
---
#### [new 035] Active Stereo-Camera Outperforms Multi-Sensor Setup in ACT Imitation Learning for Humanoid Manipulation
- **分类: cs.RO**

- **简介: 该论文研究人形机器人操作中的感知配置问题，旨在优化模仿学习的传感器组合。通过实验比较不同传感器设置，发现主动立体相机在数据有限情况下表现最佳。**

- **链接: [https://arxiv.org/pdf/2603.28422](https://arxiv.org/pdf/2603.28422)**

> **作者:** Robin Kühn; Moritz Schappler; Thomas Seel; Dennis Bank
>
> **备注:** 7 pages
>
> **摘要:** The complexity of teaching humanoid robots new tasks is one of the major reasons hindering their widespread adoption in the industry. While Imitation Learning (IL), particularly Action Chunking with Transformers (ACT), enables rapid task acquisition, there is no consensus yet on the optimal sensory hardware required for manipulation tasks. This paper benchmarks 14 sensor combinations on the Unitree G1 humanoid robot equipped with three-finger hands for two manipulation tasks. We explicitly evaluate the integration of tactile and proprioceptive modalities alongside active vision. Our analysis demonstrates that strategic sensor selection can outperform complex configurations in data-limited regimes while reducing computational overhead. We develop an open-source Unified Ablation Framework that utilizes sensor masking on a comprehensive master dataset. Results indicate that additional modalities often degrade performance for IL with limited data. A minimal active stereo-camera setup outperformed complex multi-sensor configurations, achieving 87.5% success in a spatial generalization task and 94.4% in a structured manipulation task. Conversely, adding pressure sensors to this setup reduced success to 67.3% in the latter task due to a low signal-to-noise ratio. We conclude that in data-limited regimes, active vision offers a superior trade-off between robustness and complexity. While tactile modalities may require larger datasets to be effective, our findings validate that strategic sensor selection is critical for designing an efficient learning process.
>
---
#### [new 036] D-SPEAR: Dual-Stream Prioritized Experience Adaptive Replay for Stable Reinforcement Learninging Robotic Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于强化学习任务，解决机器人操作中的训练不稳定问题。提出D-SPEAR框架，分离策略和评论家的经验回放，提升稳定性和性能。**

- **链接: [https://arxiv.org/pdf/2603.27346](https://arxiv.org/pdf/2603.27346)**

> **作者:** Yu Zhang; Karl Mason
>
> **备注:** Accepted at IEEE 11th International Conference on Control and Robotics Engineering (ICCRE 2026)
>
> **摘要:** Robotic manipulation remains challenging for reinforcement learning due to contact-rich dynamics, long horizons, and training instability. Although off-policy actor-critic algorithms such as SAC and TD3 perform well in simulation, they often suffer from policy oscillations and performance collapse in realistic settings, partly due to experience replay strategies that ignore the differing data requirements of the actor and the critic. We propose D-SPEAR: Dual-Stream Prioritized Experience Adaptive Replay, a replay framework that decouples actor and critic sampling while maintaining a shared replay buffer. The critic leverages prioritized replay for efficient value learning, whereas the actor is updated using low-error transitions to stabilize policy optimization. An adaptive anchor mechanism balances uniform and prioritized sampling based on the coefficient of variation of TD errors, and a Huber-based critic objective further improves robustness under heterogeneous reward scales. We evaluate D-SPEAR on challenging robotic manipulation tasks from the robosuite benchmark, including Block-Lifting and Door-Opening. Results demonstrate that D-SPEAR consistently outperforms strong off-policy baselines, including SAC, TD3, and DDPG, in both final performance and training stability, with ablation studies confirming the complementary roles of the actorside and critic-side replay streams.
>
---
#### [new 037] Robot Arm Control via Cognitive Map Learners
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，解决多关节机械臂的运动控制问题。通过独立训练的CML模块，无需逆运动学计算即可实现目标点定位。**

- **链接: [https://arxiv.org/pdf/2603.26773](https://arxiv.org/pdf/2603.26773)**

> **作者:** Nathan McDonald; Colyn Seeley; Christian Brazeau
>
> **摘要:** Cognitive map learners (CML) have been shown to enable hierarchical, compositional machine learning. That is, interpedently trained CML modules can be arbitrarily composed together to solve more complex problems without task-specific retraining. This work applies this approach to control the movement of a multi-jointed robot arm, whereby each arm segment's angular position is governed by an independently trained CML. Operating in a 2D Cartesian plane, target points are encoded as phasor hypervectors according to fractional power encoding (FPE). This phasor hypervector is then factorized into a set of arm segment angles either via a resonator network or a modern Hopfield network. These arm segment angles are subsequently fed to their respective arm segment CMLs, which reposition the robot arm to the target point without the use of inverse kinematic equations. This work presents both a general solution for both a 2D robot arm with an arbitrary number of arm segments and a particular solution for a 3D arm with a single rotating base.
>
---
#### [new 038] Which Reconstruction Model Should a Robot Use? Routing Image-to-3D Models for Cost-Aware Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决3D重建模型选择问题。针对不同任务需求，提出SCOUT框架，实现高效模型路由与成本约束下的优化选择。**

- **链接: [https://arxiv.org/pdf/2603.27797](https://arxiv.org/pdf/2603.27797)**

> **作者:** Akash Anand; Aditya Agarwal; Leslie Pack Kaelbling
>
> **备注:** 8 pages, 7 tables, 3 figures. Supplementary material included. Project page: this https URL
>
> **摘要:** Robotic manipulation tasks require 3D mesh reconstructions of varying quality: dexterous manipulation demands fine-grained surface detail, while collision-free planning tolerates coarser representations. Multiple reconstruction methods offer different cost-quality tradeoffs, from Image-to-3D models - whose output quality depends heavily on the input viewpoint - to view-invariant methods such as structured light scanning. Querying all models is computationally prohibitive, motivating per-input model selection. We propose SCOUT, a novel routing framework that decouples reconstruction scores into two components: (1) the relative performance of viewpoint-dependent models, captured by a learned probability distribution, and (2) the overall image difficulty, captured by a scalar partition function estimate. As the learned network operates only over the viewpoint-dependent models, view-invariant pipelines can be added, removed, or reconfigured without retraining. SCOUT also supports arbitrary cost constraints at inference time, accommodating the multi-dimensional cost constraints common in robotics. We evaluate on the Google Scanned Objects, BigBIRD, and YCB datasets under multiple mesh quality metrics, demonstrating consistent improvements over routing baselines adapted from the LLM literature across various cost constraints. We further validate the framework through robotic grasping and dexterous manipulation experiments. We release the code and additional results on our website.
>
---
#### [new 039] Spectral Decomposition of Inverse Dynamics for Fast Exploration in Model-Based Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决长时域操作中轨迹探索难题。通过逆动力学谱分解生成可行轨迹，提升长时域规划效率。**

- **链接: [https://arxiv.org/pdf/2603.27796](https://arxiv.org/pdf/2603.27796)**

> **作者:** Solvin Sigurdson; Benjamin Riviere; Joel Burdick
>
> **备注:** 8 pages, 8 figures, accepted to the 2026 IEEE International Conference on Robotics and Automation
>
> **摘要:** Planning long duration robotic manipulation sequences is challenging because of the complexity of exploring feasible trajectories through nonlinear contact dynamics and many contact modes. Moreover, this complexity grows with the problem's horizon length. We propose a search tree method that generates trajectories using the spectral decomposition of the inverse dynamics equation. This equation maps actuator displacement to object displacement, and its spectrum is efficient for exploration because its components are orthogonal and they approximate the reachable set of the object while remaining dynamically feasible. These trajectories can be combined with any search based method, such as Rapidly-Exploring Random Trees (RRT), for long-horizon planning. Our method performs similarly to recent work in model-based planning for short-horizon tasks, and differentiates itself with its ability to solve long-horizon tasks: whereas existing methods fail, ours can generate 45 second duration, 10+ contact mode plans using 15 seconds of computation, demonstrating real-time capability in highly complex domains.
>
---
#### [new 040] SCRAMPPI: Efficient Contingency Planning for Mobile Robot Navigation via Hamilton-Jacobi Reachability
- **分类: cs.RO**

- **简介: 该论文提出SCRAMPPI算法，解决移动机器人导航中的应急规划问题，通过HJ可达性分析确保安全轨迹，提升采样效率。**

- **链接: [https://arxiv.org/pdf/2603.26995](https://arxiv.org/pdf/2603.26995)**

> **作者:** Raj Harshit Srirangam; Leonard Jung; Rohith Poola; Michael Everett
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Autonomous robots commonly aim to complete a nominal behavior while minimizing a cost; this leaves them vulnerable to failure or unplanned scenarios, where a backup or contingency plan to a safe set is needed to avoid a total mission failure. This is formalized as a trajectory optimization problem over the nominal cost with a safety constraint: from any point along the nominal plan, a feasible trajectory to a designated safe set must exist. Previous methods either relax this hard constraint, or use an expensive sampling-based strategy to optimize for this constraint. Instead, we formalize this requirement as a reach-avoid problem and leverage Hamilton-Jacobi (HJ) reachability analysis to certify contingency feasibility. By computing the value function of our safe-set's backward reachable set online as the environment is revealed and integrating it with a sampling based planner (MPPI) via resampling based rollouts, we guarantee satisfaction of the hard constraint while greatly increasing sampling efficiency. Finally, we present simulated and hardware experiments demonstrating our algorithm generating nominal and contingency plans in real time on a mobile robot in an adversarial evasion task.
>
---
#### [new 041] $AutoDrive\text{-}P^3$: Unified Chain of Perception-Prediction-Planning Thought via Reinforcement Fine-Tuning
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出AutoDrive-P³框架，解决VLM在自动驾驶中缺乏连贯推理和模块协同的问题。通过整合感知、预测与规划，提升决策安全性和可解释性。**

- **链接: [https://arxiv.org/pdf/2603.28116](https://arxiv.org/pdf/2603.28116)**

> **作者:** Yuqi Ye; Zijian Zhang; Junhong Lin; Shangkun Sun; Changhao Peng; Wei Gao
>
> **备注:** Accepted at ICLR 2026 (International Conference on Learning Representations)
>
> **摘要:** Vision-language models (VLMs) are increasingly being adopted for end-to-end autonomous driving systems due to their exceptional performance in handling long-tail scenarios. However, current VLM-based approaches suffer from two major limitations: 1) Some VLMs directly output planning results without chain-of-thought (CoT) reasoning, bypassing crucial perception and prediction stages which creates a significant domain gap and compromises decision-making capability; 2) Other VLMs can generate outputs for perception, prediction, and planning tasks but employ a fragmented decision-making approach where these modules operate separately, leading to a significant lack of synergy that undermines true planning performance. To address these limitations, we propose ${AutoDrive\text{-}P^3}$, a novel framework that seamlessly integrates $\textbf{P}$erception, $\textbf{P}$rediction, and $\textbf{P}$lanning through structured reasoning. We introduce the ${P^3\text{-}CoT}$ dataset to facilitate coherent reasoning and propose ${P^3\text{-}GRPO}$, a hierarchical reinforcement learning algorithm that provides progressive supervision across all three tasks. Specifically, ${AutoDrive\text{-}P^3}$ progressively generates CoT reasoning and answers for perception, prediction, and planning, where perception provides essential information for subsequent prediction and planning, while both perception and prediction collectively contribute to the final planning decisions, enabling safer and more interpretable autonomous driving. Additionally, to balance inference efficiency with performance, we introduce dual thinking modes: detailed thinking and fast thinking. Extensive experiments on both open-loop (nuScenes) and closed-loop (NAVSIMv1/v2) benchmarks demonstrate that our approach achieves state-of-the-art performance in planning tasks. Code is available at this https URL.
>
---
#### [new 042] Heracles: Bridging Precise Tracking and Generative Synthesis for General Humanoid Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，旨在解决人形机器人在复杂环境中的鲁棒控制问题。提出Heracles框架，结合精确跟踪与生成合成，提升适应性与稳定性。**

- **链接: [https://arxiv.org/pdf/2603.27756](https://arxiv.org/pdf/2603.27756)**

> **作者:** Zelin Tao; Zeran Su; Peiran Liu; Jingkai Sun; Wenqiang Que; Jiahao Ma; Jialin Yu; Jiahang Cao; Pihai Sun; Hao Liang; Gang Han; Wen Zhao; Zhiyuan Xu; Yijie Guo; Jian Tang; Qiang Zhang
>
> **备注:** 26 pages, 7 figures, 6 tables
>
> **摘要:** Achieving general-purpose humanoid control requires a delicate balance between the precise execution of commanded motions and the flexible, anthropomorphic adaptability needed to recover from unpredictable environmental perturbations. Current general controllers predominantly formulate motion control as a rigid reference-tracking problem. While effective in nominal conditions, these trackers often exhibit brittle, non-anthropomorphic failure modes under severe disturbances, lacking the generative adaptability inherent to human motor control. To overcome this limitation, we propose Heracles, a novel state-conditioned diffusion middleware that bridges precise motion tracking and generative synthesis. Rather than relying on rigid tracking paradigms or complex explicit mode-switching, Heracles operates as an intermediary layer between high-level reference motions and low-level physics trackers. By conditioning on the robot's real-time state, the diffusion model implicitly adapts its behavior: it approximates an identity map when the state closely aligns with the reference, preserving zero-shot tracking fidelity. Conversely, when encountering significant state deviations, it seamlessly transitions into a generative synthesizer to produce natural, anthropomorphic recovery trajectories. Our framework demonstrates that integrating generative priors into the control loop not only significantly enhances robustness against extreme perturbations but also elevates humanoid control from a rigid tracking paradigm to an open-ended, generative general-purpose architecture.
>
---
#### [new 043] TerraSkipper: A Centimeter-Scale Robot for Multi-Terrain Skipping and Crawling
- **分类: cs.RO**

- **简介: 该论文属于仿生机器人任务，旨在解决多地形移动问题。通过模仿弹涂鱼的运动机制，设计出可跳跃和爬行的微型机器人，提升其在复杂环境中的适应性。**

- **链接: [https://arxiv.org/pdf/2603.27725](https://arxiv.org/pdf/2603.27725)**

> **作者:** Shashwat Singh; Sheri Zhang; Spencer Matonis; Zeynep Temel
>
> **备注:** 8 pages, 9 figures, Accepted - IEEE International Conference on Robotics & Automation (ICRA), Vienna, Austria, 2026
>
> **摘要:** Mudskippers are unique amphibious fish capable of locomotion in diverse environments, including terrestrial surfaces, aquatic habitats, and highly viscous substrates such as mud. This versatile locomotion is largely enabled by their powerful tail, which stores and rapidly releases energy to produce impulsive jumps. Inspired by this biological mechanism, we present the design and development of a multi-terrain centimeter-scale skipping and crawling robot. The robot is predominantly 3D printed and features onboard sensing, computation, and power. It is equipped with two side fins for crawling, each integrated with a hall effect sensor for gait control, while a rotary springtail driven by a 10mm planetary gear motor enables continuous impulsive skipping across a range of substrates to achieve multi-terrain locomotion. We modeled and experimentally characterized the tail, identifying an optimal length of 25mm that maximizes the mean propulsive force (4N, peaks up to 6N) for forward motion. In addition, we evaluated skipping on substrates where fin based crawling alone fails, and varied the moisture content of uniform sand and bentonite clay powder to compare skipping with crawling. Skipping consistently produced higher mean velocities than crawling, particularly on viscous and granular media. Finally, outdoor tests on grass, loose sand, and hard ground confirmed that combining skipping on entangling and granular terrain with crawling on firm ground extends the operational range of the robot in real-world environments.
>
---
#### [new 044] LARD 2.0: Enhanced Datasets and Benchmarking for Autonomous Landing Systems
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主着陆系统任务，旨在解决数据集不足和检测模型评估问题。通过扩展数据源、优化设计域并提出评估框架，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2603.26748](https://arxiv.org/pdf/2603.26748)**

> **作者:** Yassine Bougacha; Geoffrey Delhomme; Mélanie Ducoffe; Augustin Fuchs; Jean-Brice Ginestet; Jacques Girard; Sofiane Kraiem; Franck Mamalet; Vincent Mussot; Claire Pagetti; Thierry Sammour
>
> **摘要:** This paper addresses key challenges in the development of autonomous landing systems, focusing on dataset limitations for supervised training of Machine Learning (ML) models for object detection. Our main contributions include: (1) Enhancing dataset diversity, by advocating for the inclusion of new sources such as BingMap aerial images and Flight Simulator, to widen the generation scope of an existing dataset generator used to produce the dataset LARD; (2) Refining the Operational Design Domain (ODD), addressing issues like unrealistic landing scenarios and expanding coverage to multi-runway airports; (3) Benchmarking ML models for autonomous landing systems, introducing a framework for evaluating object detection subtask in a complex multi-instances setting, and providing associated open-source models as a baseline for AI models' performance.
>
---
#### [new 045] Autonomous overtaking trajectory optimization using reinforcement learning and opponent pose estimation
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决自主超车轨迹优化问题。通过强化学习与传感器数据融合，实现精准的对手位置估计和高效超车。**

- **链接: [https://arxiv.org/pdf/2603.27207](https://arxiv.org/pdf/2603.27207)**

> **作者:** Matej Rene Cihlar; Luka Šiktar; Branimir Ćaran; Marko Švaco
>
> **备注:** The paper is accepted and presented on the 35th International Conference on Robotics in Alpe-Adria-Danube Region, RAAD 2026, Bratislava, Slovakia
>
> **摘要:** Vehicle overtaking is one of the most complex driving maneuvers for autonomous vehicles. To achieve optimal autonomous overtaking, driving systems rely on multiple sensors that enable safe trajectory optimization and overtaking efficiency. This paper presents a reinforcement learning mechanism for multi-agent autonomous racing environments, enabling overtaking trajectory optimization, based on LiDAR and depth image data. The developed reinforcement learning agent uses pre-generated raceline data and sensor inputs to compute the steering angle and linear velocity for optimal overtaking. The system uses LiDAR with a 2D detection algorithm and a depth camera with YOLO-based object detection to identify the vehicle to be overtaken and its pose. The LiDAR and the depth camera detection data are fused using a UKF for improved opponent pose estimation and trajectory optimization for overtaking in racing scenarios. The results show that the proposed algorithm successfully performs overtaking maneuvers in both simulation and real-world experiments, with pose estimation RMSE of (0.0816, 0.0531) m in (x, y).
>
---
#### [new 046] Beyond Viewpoint Generalization: What Multi-View Demonstrations Offer and How to Synthesize Them for Robot Manipulation?
- **分类: cs.RO**

- **简介: 该论文研究多视角演示对机器人操作的影响，解决单视角数据局限问题。通过实验分析多视角优势，并提出RoboNVS合成新视角视频，提升策略性能。**

- **链接: [https://arxiv.org/pdf/2603.26757](https://arxiv.org/pdf/2603.26757)**

> **作者:** Boyang Cai; Qiwei Liang; Jiawei Li; Shihang Weng; Zhaoxin Zhang; Tao Lin; Xiangyu Chen; Wenjie Zhang; Jiaqi Mao; Weisheng Xu; Bin Yang; Jiaming Liang; Junhao Cai; Renjing Xu
>
> **摘要:** Does multi-view demonstration truly improve robot manipulation, or merely enhance cross-view robustness? We present a systematic study quantifying the performance gains, scaling behavior, and underlying mechanisms of multi-view data for robot manipulation. Controlled experiments show that, under both fixed and randomized backgrounds, multi-view demonstrations consistently improve single-view policy success and generalization. Performance varies non-monotonically with view coverage, revealing effective regimes rather than a simple "more is better" trend. Notably, multi-view data breaks the scaling limitation of single-view datasets and continues to raise performance ceilings after saturation. Mechanistic analysis shows that multi-view learning promotes manipulation-relevant visual representations, better aligns the action head with the learned feature distribution, and reduces overfitting. Motivated by the importance of multi-view data and its scarcity in large-scale robotic datasets, as well as the difficulty of collecting additional viewpoints in real world settings, we propose RoboNVS, a geometry-aware self-supervised framework that synthesizes novel-view videos from monocular inputs. The generated data consistently improves downstream policies in both simulation and real-world environments.
>
---
#### [new 047] CARLA-Air: Fly Drones Inside a CARLA World -- A Unified Infrastructure for Air-Ground Embodied Intelligence
- **分类: cs.RO; cs.AI; cs.CV; cs.HC**

- **简介: 该论文提出CARLA-Air，融合空中与地面模拟，解决多模态智能体协同仿真问题。它统一了高保真驾驶与飞行物理，支持多种任务的开发与测试。**

- **链接: [https://arxiv.org/pdf/2603.28032](https://arxiv.org/pdf/2603.28032)**

> **作者:** Tianle Zeng; Hanxuan Chen; Yanci Wen; Hong Zhang
>
> **备注:** Prebuilt binaries, project page, full source code, and community discussion group are all available at: this https URL
>
> **摘要:** The convergence of low-altitude economies, embodied intelligence, and air-ground cooperative systems creates growing demand for simulation infrastructure capable of jointly modeling aerial and ground agents within a single physically coherent environment. Existing open-source platforms remain domain-segregated: driving simulators lack aerial dynamics, while multirotor simulators lack realistic ground scenes. Bridge-based co-simulation introduces synchronization overhead and cannot guarantee strict spatial-temporal consistency. We present CARLA-Air, an open-source infrastructure that unifies high-fidelity urban driving and physics-accurate multirotor flight within a single Unreal Engine process. The platform preserves both CARLA and AirSim native Python APIs and ROS 2 interfaces, enabling zero-modification code reuse. Within a shared physics tick and rendering pipeline, CARLA-Air delivers photorealistic environments with rule-compliant traffic, socially-aware pedestrians, and aerodynamically consistent UAV dynamics, synchronously capturing up to 18 sensor modalities across all platforms at each tick. The platform supports representative air-ground embodied intelligence workloads spanning cooperation, embodied navigation and vision-language action, multi-modal perception and dataset construction, and reinforcement-learning-based policy training. An extensible asset pipeline allows integration of custom robot platforms into the shared world. By inheriting AirSim's aerial capabilities -- whose upstream development has been archived -- CARLA-Air ensures this widely adopted flight stack continues to evolve within a modern infrastructure. Released with prebuilt binaries and full source: this https URL
>
---
#### [new 048] Robust Global-Local Behavior Arbitration via Continuous Command Fusion Under LiDAR Errors
- **分类: cs.RO; cs.AI; eess.SY**

- **简介: 该论文属于自动驾驶任务，解决传感器误差下的控制指令仲裁问题。通过融合全局与局部控制器，提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.27273](https://arxiv.org/pdf/2603.27273)**

> **作者:** Mohamed Elgouhary; Amr S. El-Wakeel
>
> **摘要:** Modular autonomous driving systems must coordinate global progress objectives with local safety-driven reactions under imperfect sensing and strict real-time constraints. This paper presents a ROS2-native arbitration module that continuously fuses the outputs of two unchanged and interpretable controllers: a global reference-tracking controller based on Pure Pursuit and a reactive LiDAR-based Gap Follow controller. At each control step, both controllers propose Ackermann commands, and a PPO-trained policy predicts a continuous gate from a compact feature observation to produce a single fused drive command, augmented with practical safety checks. For comparison under identical ROS topic inputs and control rate, we implement a lightweight sampling-based predictive baseline. Robustness is evaluated using a ROS2 impairment protocol that injects LiDAR noise, delay, and dropout, and additionally sweeps forward-cone false short-range outliers. In a repeatable close-proximity passing scenario, we report safe success and failure rates together with per-step end-to-end controller runtime as sensing stress increases. The study is intended as a command-level robustness evaluation in a modular ROS2 setting, not as a replacement for planning-level interaction reasoning.
>
---
#### [new 049] Functionalization of Situated Robots via Vapour
- **分类: cs.RO; cond-mat.mtrl-sci**

- **简介: 该论文属于机器人功能化任务，解决 situated 机器人结构功能化难题。通过环境材料实现结构功能化，演示了利用吡咯蒸汽将PVDF纤维转化为导电结构。**

- **链接: [https://arxiv.org/pdf/2603.26752](https://arxiv.org/pdf/2603.26752)**

> **作者:** Kadri-Ann Pankratov; Leonid Zinatullin; Adele Metsniit; Marie Vihmar; Indrek Must
>
> **备注:** Accepted in 9th IEEE-RAS International Conference on Soft Robotics (Robosoft 2026) as Extended Abstract (preliminary results)
>
> **摘要:** Tight matching with the environment is key to effective robot operation in complex settings. Situated robots that build their bodies in situ (e.g. by spinning) are uniquely positioned to exploit their surroundings, yet functionalization of these structures remains an integration challenge - multimaterial spinning requires complex spinneret multiplexing, and mixture doping is limited by additive availability and chemical stability. We propose instead using materials available in the environment to functionalize in situ spun webs, reducing payload and uniquely matching the structure to its surroundings. As a demonstration, we transform an optically scattering PVDF fiber web into an optically absorbing, polypyrrole-grafted structure via pyrrole vapour exposure. Two activator-delivery strategies are shown: liquid infusion into a prefabricated web, and activator pre-embedding in the spinning mixture. Beyond this proof-of-concept, we foresee broader applications including biohybrid robots that exploit bacterial genomes for specific biomolecule synthesis in situ.
>
---
#### [new 050] Feel Robot Feels: Tactile Feedback Array Glove for Dexterous Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决teleoperation中运动映射不准确和触觉反馈不足的问题，提出TAG手套系统，实现高精度运动捕捉和高分辨率触觉反馈。**

- **链接: [https://arxiv.org/pdf/2603.28542](https://arxiv.org/pdf/2603.28542)**

> **作者:** Feiyu Jia; Xiaojie Niu; Sizhe Yang; Qingwei Ben; Tao Huang; Feng zhao; Jingbo Wang; Jiangmiao Pang
>
> **备注:** 13 pages, 16 figures
>
> **摘要:** Teleoperation is a key approach for collecting high-quality, physically consistent demonstrations for robotic manipulation. However, teleoperation for dexterous manipulation remains constrained by: (i) inaccurate hand-robot motion mapping, which limits teleoperated dexterity, and (ii) limited tactile feedback that forces vision-dominated interaction and hinders perception of contact geometry and force variation. To address these challenges, we present TAG, a low-cost glove system that integrates precise hand motion capture with high-resolution tactile feedback, enabling effective tactile-in-the-loop dexterous teleoperation. For motion capture, TAG employs a non-contact magnetic sensing design that provides drift-free, electromagnetically robust 21-DoF joint tracking with joint angle estimation errors below 1 degree. Meanwhile, to restore tactile sensation, TAG equips each finger with a 32-actuator tactile array within a compact 2 cm^2 module, allowing operators to directly feel physical interactions at the robot end-effector through spatial activation patterns. Through real-world teleoperation experiments and user studies, we show that TAG enables reliable real-time perception of contact geometry and dynamic force, improves success rates in contact-rich teleoperation tasks, and increases the reliability of demonstration data collection for learning-based manipulation.
>
---
#### [new 051] FocusVLA: Focused Visual Utilization for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决视觉信息利用不足导致的行动质量下降问题。提出FocusVLA，通过注意力机制提升对关键视觉区域的聚焦能力。**

- **链接: [https://arxiv.org/pdf/2603.28740](https://arxiv.org/pdf/2603.28740)**

> **作者:** Yichi Zhang; Weihao Yuan; Yizhuo Zhang; Xidong Zhang; Jia Wan
>
> **备注:** 25 pages, 18 figures
>
> **摘要:** Vision-Language-Action (VLA) models improve action generation by conditioning policies on rich vision-language information. However, current auto-regressive policies are constrained by three bottlenecks: (1) architectural bias drives models to overlook visual details, (2) an excessive number of visual tokens makes attention difficult to focus on the correct regions, and (3) task-irrelevant visual information introduces substantial noise - together severely impairing the quality of action. In this paper, we investigate how to effectively utilize different visual representations for action generation. To this end, we first empirically validate the above issues and show that VLA performance is primarily limited by how visual information is utilized, rather than by the quality of visual representations. Based on these insights, we introduce FocusVLA, a novel paradigm that directs the model's attention to task-relevant visual regions to effectively bridge vision to action. Specifically, we first propose Modality Cascaded Attention to eliminate shortcut pathways, thereby compelling VLA models to rely on task-relevant visual details for action generation. Furthermore, we propose Focus Attention, which dynamically selects task-relevant visual patches to control information quantity while explicitly modulating their influence to suppress task-irrelevant noise. Extensive experiments on both simulated and real-world robotic benchmarks demonstrate that FocusVLA not only effectively leverages visual details to perform dexterous manipulations, but also substantially improves performance and accelerates convergence across a variety of tasks.
>
---
#### [new 052] Rainbow-DemoRL: Combining Improvements in Demonstration-Augmented Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于强化学习任务，旨在提升在线学习的样本效率。通过分析和组合不同演示增强方法，发现直接使用离线数据优于复杂预训练方法。**

- **链接: [https://arxiv.org/pdf/2603.27400](https://arxiv.org/pdf/2603.27400)**

> **作者:** Dwait Bhatt; Shih-Chieh Chou; Nikolay Atanasov
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** Several approaches have been proposed to improve the sample efficiency of online reinforcement learning (RL) by leveraging demonstrations collected offline. The offline data can be used directly as transitions to optimize RL objectives, or offline policy and value functions can first be learned from the data and then used for online finetuning or to provide reference actions. While each of these strategies has shown compelling results, it is unclear which method has the most impact on sample efficiency, whether these approaches can be combined, and if there are cumulative benefits. We classify existing demonstration-augmented RL approaches into three categories and perform an extensive empirical study of their strengths, weaknesses, and combinations to isolate the contribution of each strategy and determine effective hybrid combinations for sample-efficient online RL. Our analysis reveals that directly reusing offline data and initializing with behavior cloning consistently outperform more complex offline RL pretraining methods for improving online sample efficiency.
>
---
#### [new 053] Contextual Graph Representations for Task-Driven 3D Perception and Planning
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文研究机器人任务规划与3D场景图的结合，旨在解决任务规划中状态空间过大的问题。通过构建基准和使用图神经网络优化表示。**

- **链接: [https://arxiv.org/pdf/2603.26685](https://arxiv.org/pdf/2603.26685)**

> **作者:** Christopher Agia
>
> **备注:** University of Toronto Undergraduate Thesis, 2021. 85 pages, 24 figures
>
> **摘要:** Recent advances in computer vision facilitate fully automatic extraction of object-centric relational representations from visual-inertial data. These state representations, dubbed 3D scene graphs, are a hierarchical decomposition of real-world scenes with a dense multiplex graph structure. While 3D scene graphs claim to promote efficient task planning for robot systems, they contain numerous objects and relations when only small subsets are required for a given task. This magnifies the state space that task planners must operate over and prohibits deployment in resource constrained settings. This thesis tests the suitability of existing embodied AI environments for research at the intersection of robot task planning and 3D scene graphs and constructs a benchmark for empirical comparison of state-of-the-art classical planners. Furthermore, we explore the use of graph neural networks to harness invariances in the relational structure of planning domains and learn representations that afford faster planning.
>
---
#### [new 054] Bridging the Awareness Gap: Socially Mediated State Externalization for Transparent Distributed Home Robots
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机交互任务，旨在解决分布式家庭机器人系统中用户对机器人状态不透明的问题。通过社交媒介实现状态外化，提升用户信任与控制感，同时保持任务性能。**

- **链接: [https://arxiv.org/pdf/2603.26686](https://arxiv.org/pdf/2603.26686)**

> **作者:** Wenzheng Zhao; Manideep Duggi; Fengpei Yuan
>
> **备注:** 9 pages, 7 figures, 6 tables. Under review for IROS 2026
>
> **摘要:** Distributed multi-robot systems for the home often require robots to operate out of the user's sight, creating a state awareness gap that can diminish trust and perceived transparency and control. This paper investigates whether real-time, socially mediated state externalization can bridge this gap without compromising task performance. We developed a system where a co-located social mediator robot (Pepper) externalizes the hidden execution states of an out-of-sight mobile manipulator (Stretch~3) for voice-driven object retrieval and delivery, where task-level states are synchronized and externalized through verbal updates and visual progress display. In a counterbalanced within-subject study (N=30), we compared a baseline of Autonomous Hidden Execution against Socially Mediated State Externalization. Our results show that externalization significantly increases user task-focused attention (from 15.8% to 84.6%, p<.001) and substantially improves perceived perspicuity, dependability, stimulation, and attractiveness (all p<.001). Furthermore, 83% of participants preferred the externalized condition, and this improvement in user experience was achieved without a statistically significant increase in end-to-end task completion time (p=.271). The results suggest that socially mediated state externalization is an effective architectural mechanism for designing more transparent and trustworthy distributed robot systems, ultimately enhancing user experience without sacrificing performance in distributed home robot deployments.
>
---
#### [new 055] LLM-Enabled Low-Altitude UAV Natural Language Navigation via Signal Temporal Logic Specification Translation and Repair
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自然语言导航任务，解决低空无人机在城市环境中安全执行指令的问题。通过将自然语言翻译为STL规范并生成可行轨迹，提升导航安全性与适应性。**

- **链接: [https://arxiv.org/pdf/2603.27583](https://arxiv.org/pdf/2603.27583)**

> **作者:** Yuqi Ping; Huahao Ding; Tianhao Liang; Longyu Zhou; Guangyu Lei; Xinglin Chen; Junwei Wu; Jieyu Zhou; Tingting Zhang
>
> **摘要:** Natural language (NL) navigation for low-altitude unmanned aerial vehicles (UAVs) offers an intelligent and convenient solution for low-altitude aerial services by enabling an intuitive interface for non-expert operators. However, deploying this capability in urban environments necessitates the precise grounding of underspecified instructions into safety-critical, dynamically feasible motion plans subject to spatiotemporal constraints. To address this challenge, we propose a unified framework that translates NL instructions into Signal Temporal Logic (STL) specifications and subsequently synthesizes trajectories via mixed-integer linear programming (MILP). Specifically, to generate executable STL formulas from free-form NL, we develop a reasoning-enhanced large language model (LLM) leveraging chain-of-thought (CoT) supervision and group-relative policy optimization (GRPO), which ensures high syntactic validity and semantic consistency. Furthermore, to resolve infeasibilities induced by stringent logical or spatial requirements, we introduce a specification repair mechanism. This module combines MILP-based diagnosis with LLM-guided semantic reasoning to selectively relax task constraints while strictly enforcing safety guarantees. Extensive simulations and real-world flight experiments demonstrate that the proposed closed-loop framework significantly improves NL-to-STL translation robustness, enabling safe, interpretable, and adaptable UAV navigation in complex scenarios.
>
---
#### [new 056] Fine-Tuning Large Language Models for Cooperative Tactical Deconfliction of Small Unmanned Aerial Systems
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多智能体协同决策任务，旨在解决小无人机在低空空域的冲突解脱问题。通过微调大语言模型，提升其在安全约束下的决策准确性与一致性。**

- **链接: [https://arxiv.org/pdf/2603.28561](https://arxiv.org/pdf/2603.28561)**

> **作者:** Iman Sharifi; Alex Zongo; Peng Wei
>
> **备注:** 15 pages, 6 figures, to be published in CVPR 2026 Workshop Proceedings
>
> **摘要:** The growing deployment of small Unmanned Aerial Systems (sUASs) in low-altitude airspaces has increased the need for reliable tactical deconfliction under safety-critical constraints. Tactical deconfliction involves short-horizon decision-making in dense, partially observable, and heterogeneous multi-agent environments, where both cooperative separation assurance and operational efficiency must be maintained. While Large Language Models (LLMs) exhibit strong reasoning capabilities, their direct application to air traffic control remains limited by insufficient domain grounding and unpredictable output inconsistency. This paper investigates LLMs as decision-makers in cooperative multi-agent tactical deconfliction using fine-tuning strategies that align model outputs to human operator heuristics. We propose a simulation-to-language data generation pipeline based on the BlueSky air traffic simulator that produces rule-consistent deconfliction datasets reflecting established safety practices. A pretrained Qwen-Math-7B model is fine-tuned using two parameter-efficient strategies: supervised fine-tuning with Low-Rank Adaptation (LoRA) and preference-based fine-tuning combining LoRA with Group-Relative Policy Optimization (GRPO). Experimental results on validation datasets and closed-loop simulations demonstrate that supervised LoRA fine-tuning substantially improves decision accuracy, consistency, and separation performance compared to the pretrained LLM, with significant reductions in near mid-air collisions. GRPO provides additional coordination benefits but exhibits reduced robustness when interacting with heterogeneous agent policies.
>
---
#### [new 057] Control Without Control: Defining Implicit Interaction Paradigms for Autonomous Assistive Robots
- **分类: cs.RO**

- **简介: 论文探讨了自主助人机器人的隐式交互设计，旨在解决用户控制感与自动化之间的矛盾。通过两个案例研究，提出提升系统直观性与适应性的方法。**

- **链接: [https://arxiv.org/pdf/2603.28079](https://arxiv.org/pdf/2603.28079)**

> **作者:** Janavi Gupta; Kavya Puthuveetil; Dimitra Tsakona; Akhil Padmanabha; Yiannis Demiris; Zackory Erickson
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** Assistive robotic systems have shown growing potential to improve the quality of life of those with disabilities. As researchers explore the automation of various caregiving tasks, considerations for how the technology can still preserve the user's sense of control become paramount to ensuring that robotic systems are aligned with fundamental user needs and motivations. In this work, we present two previously developed systems as design cases through which to explore an interaction paradigm that we call implicit control, where the behavior of an autonomous robot is modified based on users' natural behavioral cues, instead of some direct input. Our selected design cases, unlike systems in past work, specifically probe users' perception of the interaction. We find, from a new thematic analysis of qualitative feedback on both cases, that designing for effective implicit control enables both a reduction in perceived workload and the preservation of the users' sense of control through the system's intuitiveness and responsiveness, contextual awareness, and ability to adapt to preferences. We further derive a set of core guidelines for designers in deciding when and how to apply implicit interaction paradigms for their assistive applications.
>
---
#### [new 058] Learning Energy-Efficient Air--Ground Actuation for Hybrid Robots on Stair-Like Terrain
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，旨在解决混合机器人在楼梯状地形上的能效问题。通过强化学习，训练出协调推进器、轮子和舵机的策略，提升能效并实现有效攀爬。**

- **链接: [https://arxiv.org/pdf/2603.26687](https://arxiv.org/pdf/2603.26687)**

> **作者:** Jiaxing Li; Wen Tian; Xinhang Xu; Junbin Yuan; Sebastian Scherer; Muqing Cao
>
> **摘要:** Hybrid aerial--ground robots offer both traversability and endurance, but stair-like discontinuities create a trade-off: wheels alone often stall at edges, while flight is energy-hungry for small height gains. We propose an energy-aware reinforcement learning framework that trains a single continuous policy to coordinate propellers, wheels, and tilt servos without predefined aerial and ground modes. We train policies from proprioception and a local height scan in Isaac Lab with parallel environments, using hardware-calibrated thrust/power models so the reward penalizes true electrical energy. The learned policy discovers thrust-assisted driving that blends aerial thrust and ground traction. In simulation it achieves about 4 times lower energy than propeller-only control. We transfer the policy to a DoubleBee prototype on an 8cm gap-climbing task; it achieves 38% lower average power than a rule-based decoupled controller. These results show that efficient hybrid actuation can emerge from learning and deploy on hardware.
>
---
#### [new 059] DRIVE-Nav: Directional Reasoning, Inspection, and Verification for Efficient Open-Vocabulary Navigation
- **分类: cs.RO**

- **简介: 该论文提出DRIVE-Nav，解决开放词汇导航任务中的路径效率与稳定性问题，通过方向推理与验证提升导航效果。**

- **链接: [https://arxiv.org/pdf/2603.28691](https://arxiv.org/pdf/2603.28691)**

> **作者:** Maoguo Gao; Zejun Zhu; Zhiming Sun; Zhengwei Ma; Longze Yuan; Zhongjing Ma; Zhigang Gao; Jinhui Zhang; Suli Zou
>
> **备注:** 8 pages, 4 figures. Project page: this https URL
>
> **摘要:** Open-Vocabulary Object Navigation (OVON) requires an embodied agent to locate a language-specified target in unknown environments. Existing zero-shot methods often reason over dense frontier points under incomplete observations, causing unstable route selection, repeated revisits, and unnecessary action overhead. We present DRIVE-Nav, a structured framework that organizes exploration around persistent directions rather than raw frontiers. By inspecting encountered directions more completely and restricting subsequent decisions to still-relevant directions within a forward 240 degree view range, DRIVE-Nav reduces redundant revisits and improves path efficiency. The framework extracts and tracks directional candidates from weighted Fast Marching Method (FMM) paths, maintains representative views for semantic inspection, and combines vision-language-guided prompt enrichment with cross-frame verification to improve grounding reliability. Experiments on HM3D-OVON, HM3Dv2, and MP3D demonstrate strong overall performance and consistent efficiency gains. On HM3D-OVON, DRIVE-Nav achieves 50.2% SR and 32.6% SPL, improving the previous best method by 1.9% SR and 5.6% SPL. It also delivers the best SPL on HM3Dv2 and MP3D and transfers to a physical humanoid robot. Real-world deployment also demonstrates its effectiveness. Project page: this https URL
>
---
#### [new 060] Topological Motion Planning Diffusion: Generative Tangle-Free Path Planning for Tethered Robots in Obstacle-Rich Environments
- **分类: cs.RO**

- **简介: 该论文属于路径规划任务，解决绳索机器人在复杂环境中避免缠绕的问题。提出TMPD框架，结合拓扑记忆与扩散模型生成无缠绕路径。**

- **链接: [https://arxiv.org/pdf/2603.26696](https://arxiv.org/pdf/2603.26696)**

> **作者:** Yifu Tian; Xinhang Xu; Thien-Minh Nguyen; Muqing Cao
>
> **摘要:** In extreme environments such as underwater exploration and post-disaster rescue, tethered robots require continuous navigation while avoiding cable entanglement. Traditional planners struggle in these lifelong planning scenarios due to topological unawareness, while topology-augmented graph-search methods face computational bottlenecks in obstacle-rich environments where the number of candidate topological classes increases. To address these challenges, we propose Topological Motion Planning Diffusion (TMPD), a novel generative planning framework that integrates lifelong topological memory. Instead of relying on sequential graph search, TMPD leverages a diffusion model to propose a multimodal front-end of kinematically feasible trajectory candidates across various homotopy classes. A tether-aware topological back-end then filters and optimizes these candidates by computing generalized winding numbers to evaluate their topological energy against the accumulated tether configuration. Benchmarking in obstacle-rich simulated environments demonstrates that TMPD achieves a collision-free reach of 100% and a tangle-free rate of 97.0%, outperforming traditional topological search and purely kinematic diffusion baselines in both geometric smoothness and computational efficiency. Simulation with realistic cable dynamics further validates the practicality of the proposed approach.
>
---
#### [new 061] ReMemNav: A Rethinking and Memory-Augmented Framework for Zero-Shot Object Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于零样本目标导航任务，旨在解决未知环境中定位未见目标的问题。提出ReMemNav框架，结合视觉语言模型与记忆机制，提升导航成功率和效率。**

- **链接: [https://arxiv.org/pdf/2603.26788](https://arxiv.org/pdf/2603.26788)**

> **作者:** Feng Wu; Wei Zuo; Wenliang Yang; Jun Xiao; Yang Liu; Xinhua Zeng
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Zero-shot object navigation requires agents to locate unseen target objects in unfamiliar environments without prior maps or task-specific training which remains a significant challenge. Although recent advancements in vision-language models(VLMs) provide promising commonsense reasoning capabilities for this task, these models still suffer from spatial hallucinations, local exploration deadlocks, and a disconnect between high-level semantic intent and low-level control. In this regard, we propose a novel hierarchical navigation framework named ReMemNav, which seamlessly integrates panoramic semantic priors and episodic memory with VLMs. We introduce the Recognize Anything Model to anchor the spatial reasoning process of the VLM. We also design an adaptive dual-modal rethinking mechanism based on an episodic semantic buffer queue. The proposed mechanism actively verifies target visibility and corrects decisions using historical memory to prevent deadlocks. For low-level action execution, ReMemNav extracts a sequence of feasible actions using depth masks, allowing the VLM to select the optimal action for mapping into actual spatial movement. Extensive evaluations on HM3D and MP3D demonstrate that ReMemNav outperforms existing training-free zero-shot baselines in both success rate and exploration efficiency. Specifically, we achieve significant absolute performance improvements, with SR and SPL increasing by 1.7% and 7.0% on HM3D v0.1, 18.2% and 11.1% on HM3D v0.2, and 8.7% and 7.9% on MP3D.
>
---
#### [new 062] SpatialAnt: Autonomous Zero-Shot Robot Navigation via Active Scene Reconstruction and Visual Anticipation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉导航任务，解决零样本机器人导航中因自建场景不完整导致的性能下降问题。提出SpatialAnt框架，结合物理对齐和视觉前瞻机制，提升导航鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.26837](https://arxiv.org/pdf/2603.26837)**

> **作者:** Jiwen Zhang; Xiangyu Shi; Siyuan Wang; Zerui Li; Zhongyu Wei; Qi Wu
>
> **备注:** 10 pages, 4 figures, 5 tables. Homepage: this https URL
>
> **摘要:** Vision-and-Language Navigation (VLN) has recently benefited from Multimodal Large Language Models (MLLMs), enabling zero-shot navigation. While recent exploration-based zero-shot methods have shown promising results by leveraging global scene priors, they rely on high-quality human-crafted scene reconstructions, which are impractical for real-world robot deployment. When encountering an unseen environment, a robot should build its own priors through pre-exploration. However, these self-built reconstructions are inevitably incomplete and noisy, which severely degrade methods that depend on high-quality scene reconstructions. To address these issues, we propose SpatialAnt, a zero-shot navigation framework designed to bridge the gap between imperfect self-reconstructions and robust execution. SpatialAnt introduces a physical grounding strategy to recover the absolute metric scale for monocular-based reconstructions. Furthermore, rather than treating the noisy self-reconstructed scenes as absolute spatial references, we propose a novel visual anticipation mechanism. This mechanism leverages the noisy point clouds to render future observations, enabling the agent to perform counterfactual reasoning and prune paths that contradict human instructions. Extensive experiments in both simulated and real-world environments demonstrate that SpatialAnt significantly outperforms existing zero-shot methods. We achieve a 66% Success Rate (SR) on R2R-CE and 50.8% SR on RxR-CE benchmarks. Physical deployment on a Hello Robot further confirms the efficiency and efficacy of our framework, achieving a 52% SR in challenging real-world settings.
>
---
#### [new 063] A Foldable and Agile Soft Electromagnetic Robot for Multimodal Navigation in Confined and Unstructured Environments
- **分类: cs.RO; cond-mat.mtrl-sci; cond-mat.soft; physics.app-ph**

- **简介: 该论文属于软体机器人领域，旨在解决复杂环境中多模式移动问题。研究设计了一种可折叠、灵活的电磁软机器人，具备多种运动方式，适用于生物体内导航。**

- **链接: [https://arxiv.org/pdf/2603.28362](https://arxiv.org/pdf/2603.28362)**

> **作者:** Zhihao Lv; Xiaoyong Zhang; Mengfan Zhang; Xiaoyu Song; Xingyue Liu; Yide Liu; Shaoxing Qu; Guoyong Mao
>
> **摘要:** Multimodal locomotion is crucial for an animal's adaptability in unstructured wild environments. Similarly, in the human gastrointestinal tract, characterized by viscoelastic mucus, complex rugae, and narrow sphincters like the cardia, multimodal locomotion is also essential for a small-scale soft robot to conduct tasks. Here, we introduce a small-scale compact, foldable, and robust soft electromagnetic robot (M-SEMR) with more than nine locomotion modes designed for such a scenario. Featuring a six-spoke elastomer body embedded with liquid metal channels and driven by Laplace forces under a static magnetic field, the M-SEMR is capable of rapid transitions (< 0.35 s) among different locomotion modes. It achieves exceptional agility, including high-speed rolling (818 mm/s, 26 BL/s), omnidirectional crawling, jumping, and swimming. Notably, the robot can fold to reduce its volume by 79%, enabling it to traverse confined spaces. We further validate its navigation capabilities on complex terrains, including discrete obstacles, viscoelastic gelatin surfaces, viscous fluids, and simulated biological tissues. This system offers a versatile strategy for developing high-mobility soft robots for future biomedical applications.
>
---
#### [new 064] A Self-Rotating Tri-Rotor UAV for Field of View Expansion and Autonomous Flight
- **分类: cs.RO**

- **简介: 该论文属于无人机感知任务，旨在解决传感器视场角窄的问题。通过自旋三旋翼设计扩展感知视野，实现自主飞行与高精度轨迹跟踪。**

- **链接: [https://arxiv.org/pdf/2603.28581](https://arxiv.org/pdf/2603.28581)**

> **作者:** Xiaobin Zhou; Zihao Zheng; Aoxu Jin; Lei Qiang; Bo Zhu
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) perception relies on onboard sensors like cameras and LiDAR, which are limited by the narrow field of view (FoV). We present Self-Perception INertial Navigation Enabled Rotorcraft (SPINNER), a self-rotating tri-rotor UAV for the FoV expansion and autonomous flight. Without adding extra sensors or energy consumption, SPINNER significantly expands the FoV of onboard camera and LiDAR sensors through continuous spin motion, thereby enhancing environmental perception efficiency. SPINNER achieves full 3-dimensional position and roll--pitch attitude control using only three brushless motors, while adjusting the rotation speed via anti-torque plates design. To address the strong coupling, severe nonlinearity, and complex disturbances induced by spinning flight, we develop a disturbance compensation control framework that combines nonlinear model predictive control (MPC) with incremental nonlinear dynamic inversion. Experimental results demonstrate that SPINNER maintains robust flight under wind disturbances up to 4.8 \,m/s and achieves high-precision trajectory tracking at a maximum speed of 2.0\,m/s. Moreover, tests in parking garages and forests show that the rotational perception mechanism substantially improves FoV coverage and enhances perception capability of SPINNER.
>
---
#### [new 065] Robotic Dexterous Manipulation via Anisotropic Friction Modulation using Passive Rollers
- **分类: cs.RO**

- **简介: 论文提出一种基于被动滚轮的机械手指设计，用于调节接触摩擦，解决机器人灵巧操作中摩擦控制困难的问题。通过滚轮的制动与旋转，实现多样化的抓取与操作任务。**

- **链接: [https://arxiv.org/pdf/2603.27452](https://arxiv.org/pdf/2603.27452)**

> **作者:** Ethan Fisk; Taeyoon Lee; Shenli Yuan
>
> **备注:** 2026 IEEE International Conference on Robotics & Automation
>
> **摘要:** Controlling friction at the fingertip is fundamental to dexterous manipulation, yet remains difficult to realize in robotic hands. We present the design and analysis of a robotic fingertip equipped with passive rollers that can be selectively braked or pivoted to modulate contact friction and constraint directions. When unbraked, the rollers permit unconstrained sliding of the contact point along the rolling direction; when braked, they resist motion like a conventional fingertip. The rollers are mounted on a pivoting mechanism, allowing reorientation of the constraint frame to accommodate different manipulation tasks. We develop a constraint-based model of the fingertip integrated into a parallel-jaw gripper and analyze its ability to support diverse manipulation strategies. Experiments show that the proposed design enables a wide range of dexterous actions that are conventionally challenging for robotic grippers, including sliding and pivoting within the grasp, robust adaptation to uncertain contacts, multi-object or multi-part manipulation, and interactions requiring asymmetric friction across fingers. These results demonstrate the versatility of passive roller fingertips as a low-complexity, mechanically efficient approach to friction modulation, advancing the development of more adaptable and robust robotic manipulation.
>
---
#### [new 066] Reducing Mental Workload through On-Demand Human Assistance for Physical Action Failures in LLM-based Multi-Robot Coordination
- **分类: cs.RO**

- **简介: 该论文属于多机器人协作任务，旨在解决LLM执行物理失败后任务停滞的问题。提出REPAIR框架，通过人机协作实现远程错误恢复，提升任务完成效率。**

- **链接: [https://arxiv.org/pdf/2603.28156](https://arxiv.org/pdf/2603.28156)**

> **作者:** Shoichi Hasegawa; Akira Taniguchi; Lotfi El Hafi; Gustavo Alfonso Garcia Ricardez; Tadahiro Taniguchi
>
> **备注:** Under review in IEEE RO-MAN 2026. Project page is this https URL
>
> **摘要:** Multi-robot coordination based on large language models (LLMs) has attracted growing attention, since LLMs enable the direct translation of natural language instructions into robot action plans by decomposing tasks and generating high-level plans. However, recovering from physical execution failures remains difficult, and tasks often stagnate due to the repetition of the same unsuccessful actions. While frameworks for remote robot operation using Mixed Reality were proposed, there have been few attempts to implement remote error resolution specifically for physical failures in multi-robot environments. In this study, we propose REPAIR (Robot Execution with Planned And Interactive Recovery), a human-in-the-loop framework that integrates remote error resolution into LLM-based multi-robot planning. In this method, robots execute tasks autonomously; however, when an irrecoverable failure occurs, the LLM requests assistance from an operator, enabling task continuity through remote intervention. Evaluations using a multi-robot trash collection task in a real-world environment confirmed that REPAIR significantly improves task progress (the number of items cleared within a time limit) compared to fully autonomous methods. Furthermore, for easily collectable items, it achieved task progress equivalent to full remote control. The results also suggested that the mental workload on the operator may differ in terms of physical demand and effort. The project website is this https URL.
>
---
#### [new 067] Learning Smooth and Robust Space Robotic Manipulation of Dynamic Target via Inter-frame Correlation
- **分类: cs.RO**

- **简介: 该论文属于空间机械臂操作任务，旨在解决微重力下动态目标的精准操控问题。通过引入帧间相关性机制，提升轨迹平稳性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.27537](https://arxiv.org/pdf/2603.27537)**

> **作者:** Siyi Lang; Hongyi Gao; Yingxin Zhang; Zihao Liu; Hanlin Dong; Zhaoke Ning; Zhiqiang Ma; Panfeng Huang
>
> **备注:** none
>
> **摘要:** On-orbit servicing represents a critical frontier in future aerospace engineering, with the manipulation of dynamic non-cooperative targets serving as a key technology. In microgravity environments, objects are typically free-floating, lacking the support and frictional constraints found on Earth, which significantly escalates the complexity of tasks involving space robotic manipulation. Conventional planning and control-based methods are primarily limited to known, static scenarios and lack real-time responsiveness. To achieve precise robotic manipulation of dynamic targets in unknown and unstructured space environments, this letter proposes a data-driven space robotic manipulation approach that integrates historical temporal information and inter-frame correlation mechanisms. By exploiting the temporal correlation between historical and current frames, the system can effectively capture motion features within the scene, thereby producing stable and smooth manipulation trajectories for dynamic targets. To validate the effectiveness of the proposed method, we developed a ground-based experimental platform consisting of a PIPER X robotic arm and a dual-axis linear stage, which accurately simulates micro-gravity free-floating motion in a 2D plane.
>
---
#### [new 068] Co-designing a Social Robot for Newcomer Children's Cultural and Language Learning
- **分类: cs.RO; cs.CY; cs.HC**

- **简介: 该论文属于教育技术任务，旨在解决 newcomer 儿童语言和文化学习问题。通过与教师合作设计社交机器人 Maple，探索其在文化适应和语言学习中的应用，提出设计指南与挑战分析。**

- **链接: [https://arxiv.org/pdf/2603.26674](https://arxiv.org/pdf/2603.26674)**

> **作者:** Neil Fernandes; Tehniyat Shahbaz; Emily Davies-Robinson; Yue Hu; Kerstin Dautenhahn
>
> **备注:** In proceedings of the 21st ACM/IEEE International Conference on Human-Robot Interaction (HRI 2026)
>
> **摘要:** Newcomer children face barriers in acquiring the host country's language and literacy programs are often constrained by limited staffing, mixed-proficiency cohorts, and short contact time. While Socially Assistive Robots (SARs) show promise in education, their use in these socio-emotionally sensitive settings remains underexplored. This research presents a co-design study with program tutors and coordinators, to explore the design space for a social robot, Maple. We contribute (1) a domain summary outlining four recurring challenges, (2) a discussion on cultural orientation and community belonging with robots, (3) an expert-grounded discussion of the perceived role of an SAR in cultural and language learning, and (4) preliminary design guidelines for integrating an SAR into a classroom. These expert-grounded insights lay the foundation for iterative design and evaluation with newcomer children and their families.
>
---
#### [new 069] Vision-Based Robotic Disassembly Combined with Real-Time MFA Data Acquisition
- **分类: cs.RO; cs.CE**

- **简介: 该论文属于机器人拆解与材料流分析任务，旨在解决CRMs回收与实时MFA数据获取问题，通过视觉检测实现PC拆解与同步材料数据采集。**

- **链接: [https://arxiv.org/pdf/2603.28690](https://arxiv.org/pdf/2603.28690)**

> **作者:** Federico Zocco; Maria Pozzi; Monica Malvezzi
>
> **备注:** Submitted
>
> **摘要:** Stable and reliable supplies of rare-Earth minerals and critical raw materials (CRMs) are essential for the development of the European Union. Since a large share of these materials enters the Union from outside, a valid option for CRMs supply resilience and security is to recover them from end-of-use products. Hence, in this paper we present the preliminary phases of the development of real-time visual detection of PC desktop components running on edge devices to simultaneously achieve two goals. The first goal is to perform robotic disassembly of PC desktops, where the adaptivity of learning-based vision can enable the processing of items with unpredictable geometry caused by accidental damages. We also discuss the robot end-effectors for different PC components with the object contact points derivable from neural detector bounding boxes. The second goal is to provide in an autonomous, highly-granular, and timely fashion, the data needed to perform material flow analysis (MFA) since, to date, MFA often lacks of the data needed to accurately study material stocks and flows. The second goal is achievable thanks to the recently-proposed synchromaterials, which can generate both local and wide-area (e.g., national) material mass information in a real-time and synchronized fashion.
>
---
#### [new 070] Proposing a Game Theory Approach to Explore Group Dynamics with Social Robot
- **分类: cs.RO; cs.HC**

- **简介: 论文属于社会机器人研究任务，旨在探索社交机器人如何促进群体合作。通过博弈论方法，设计实验评估机器人对群体决策的影响。**

- **链接: [https://arxiv.org/pdf/2603.28348](https://arxiv.org/pdf/2603.28348)**

> **作者:** Giulia Pusceddu
>
> **备注:** Honorable Mention at HRI Pioneers 2025. Peer-reviewed. this https URL
>
> **摘要:** Integrating social robots in our group-based society, beyond the technical challenges, requires considering the social group dynamics. Following the results from preliminary exploratory studies on the influence of social robots on group decisions, the proposed research investigates whether social robots can foster cooperation among group members. To achieve this, I propose a game theory approach, employing the Public Good Game to recreate a simplified and controlled social situation where the robot's influence can be evaluated. Clarifying the role of robots in promoting collaboration among humans might have a significant impact in educational environments, enhancing student learning, as well as in workplace settings, where they could facilitate problem-solving and lead to shared solutions.
>
---
#### [new 071] Point of View: How Perspective Affects Perceived Robot Sociability
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决机器人导航社会性评估中的视角差异问题。通过VR实验，研究不同视角对机器人轨迹感知的影响，并探索手势信号提升舒适度的效果。**

- **链接: [https://arxiv.org/pdf/2603.28272](https://arxiv.org/pdf/2603.28272)**

> **作者:** Subham Agrawal; Aftab Akthar; Nils Dengler; Maren Bennewitz
>
> **摘要:** Ensuring that robot navigation is safe and socially acceptable is crucial for comfortable human-robot interaction in shared environments. However, existing validation methods often rely on a bird's-eye (allocentric) perspective, which fails to capture the subjective first-person experience of pedestrians encountering robots in the real world. In this paper, we address the perceptual gap between allocentric validation and egocentric experience by investigating how different perspectives affect the perceived sociability and disturbance of robot trajectories. Our approach uses an immersive VR environment to evaluate identical robot trajectories across allocentric, egocentric-proximal, and egocentric-distal viewpoints in a user study. We perform this analysis for trajectories generated from two different navigation policies to understand if the observed differences are unique to a single type of trajectory or more generalizable. We further examine whether augmenting a trajectory with a head-nod gesture can bridge the perceptual gap and improve human comfort. Our experiments suggest that trajectories rated as sociable from an allocentric view may be perceived as significantly more disturbing when experienced from a first-person perspective in close proximity. Our results also demonstrate that while passing distance affects perceived disturbance, communicative social signaling, such as a head-nod, can effectively enhance the perceived sociability of the robot's behavior.
>
---
#### [new 072] Copilot-Assisted Second-Thought Framework for Brain-to-Robot Hand Motion Decoding
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于脑机接口任务，旨在提升脑电到机器人手部运动的解码精度。通过混合模型和多模态融合，提高运动轨迹预测性能，并引入辅助框架优化解码结果。**

- **链接: [https://arxiv.org/pdf/2603.27492](https://arxiv.org/pdf/2603.27492)**

> **作者:** Yizhe Li; Shixiao Wang; Jian K. Liu
>
> **摘要:** Motor kinematics prediction (MKP) from electroencephalography (EEG) is an important research area for developing movement-related brain-computer interfaces (BCIs). While traditional methods often rely on convolutional neural networks (CNNs) or recurrent neural networks (RNNs), Transformer-based models have shown strong ability in modeling long sequential EEG data. In this study, we propose a CNN-attention hybrid model for decoding hand kinematics from EEG during grasp-and-lift tasks, achieving strong performance in within-subject experiments. We further extend this approach to EEG-EMG multimodal decoding, which yields substantially improved results. Within-subject tests achieve PCC values of 0.9854, 0.9946, and 0.9065 for the X, Y, and Z axes, respectively, computed on the midpoint trajectory between the thumb and index finger, while cross-subject tests result in 0.9643, 0.9795, and 0.5852. The decoded trajectories from both modalities are then used to control a Franka Panda robotic arm in a MuJoCo simulation. To enhance trajectory fidelity, we introduce a copilot framework that filters low-confidence decoded points using a motion-state-aware critic within a finite-state machine. This post-processing step improves the overall within-subject PCC of EEG-only decoding to 0.93 while excluding fewer than 20% of the data points.
>
---
#### [new 073] SOLE-R1: Video-Language Reasoning as the Sole Reward for On-Robot Reinforcement Learning
- **分类: cs.RO; cs.CL; cs.CV**

- **简介: 该论文提出SOLE-R1，用于机器人强化学习的视频-语言推理模型，解决无监督任务学习问题。通过视频和自然语言目标生成密集奖励信号，实现零样本在线学习。**

- **链接: [https://arxiv.org/pdf/2603.28730](https://arxiv.org/pdf/2603.28730)**

> **作者:** Philip Schroeder; Thomas Weng; Karl Schmeckpeper; Eric Rosen; Stephen Hart; Ondrej Biza
>
> **摘要:** Vision-language models (VLMs) have shown impressive capabilities across diverse tasks, motivating efforts to leverage these models to supervise robot learning. However, when used as evaluators in reinforcement learning (RL), today's strongest models often fail under partial observability and distribution shift, enabling policies to exploit perceptual errors rather than solve the task. To address this limitation, we introduce SOLE-R1 (Self-Observing LEarner), a video-language reasoning model explicitly designed to serve as the sole reward signal for online RL. Given only raw video observations and a natural-language goal, SOLE-R1 performs per-timestep spatiotemporal chain-of-thought (CoT) reasoning and produces dense estimates of task progress that can be used directly as rewards. To train SOLE-R1, we develop a large-scale video trajectory and reasoning synthesis pipeline that generates temporally grounded CoT traces aligned with continuous progress supervision. This data is combined with foundational spatial and multi-frame temporal reasoning, and used to train the model with a hybrid framework that couples supervised fine-tuning with RL from verifiable rewards. Across four different simulation environments and a real-robot setting, SOLE-R1 enables zero-shot online RL from random initialization: robots learn previously unseen manipulation tasks without ground-truth rewards, success indicators, demonstrations, or task-specific tuning. SOLE-R1 succeeds on 24 unseen tasks and substantially outperforms strong vision-language rewarders, including GPT-5 and Gemini-3-Pro, while exhibiting markedly greater robustness to reward hacking.
>
---
#### [new 074] A Predictive Control Strategy to Offset-Point Tracking for Agricultural Mobile Robots
- **分类: cs.RO**

- **简介: 该论文属于农业机器人路径跟踪任务，解决传统控制器忽略附属装置位置导致的跟踪误差问题。通过提出一种预测控制策略，建模附属装置为刚性偏移点，提升跟踪精度与作业安全。**

- **链接: [https://arxiv.org/pdf/2603.28439](https://arxiv.org/pdf/2603.28439)**

> **作者:** Stephane Ngnepiepaye Wembe; Vincent Rousseau; Johann Laconte; Roland Lenain
>
> **备注:** Accepted in the journal IEEE Transaction on Field Robotics
>
> **摘要:** Robots are increasingly being deployed in agriculture to support sustainable practices and improve productivity. They offer strong potential to enable precise, efficient, and environmentally friendly operations. However, most existing path-following controllers focus solely on the robot's center of motion and neglect the spatial footprint and dynamics of attached implements. In practice, implements such as mechanical weeders or spring-tine cultivators are often large, rigidly mounted, and directly interacting with crops and soil; ignoring their position can degrade tracking performance and increase the risk of crop damage. To address this limitation, we propose a closed-form predictive control strategy extending the approach introduced in [1]. The method is developed specifically for Ackermann-type agricultural vehicles and explicitly models the implement as a rigid offset point, while accounting for lateral slip and lever-arm effects. The approach is benchmarked against state-of-the-art baseline controllers, including a reactive geometric method, a reactive backstepping method, and a model-based predictive scheme. Real-world agricultural experiments with two different implements show that the proposed method reduces the median tracking error by 24% to 56%, and decreases peak errors during curvature transitions by up to 70%. These improvements translate into enhanced operational safety, particularly in scenarios where the implement operates in close proximity to crop rows.
>
---
#### [new 075] Multi-AUV Ad-hoc Networks-Based Multi-Target Tracking Based on Scene-Adaptive Embodied Intelligence
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多AUV协同跟踪任务，解决动态水下环境中通信受限与拓扑变化问题，提出场景自适应的EI架构和SA-MARL算法，提升跟踪精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.27194](https://arxiv.org/pdf/2603.27194)**

> **作者:** Kai Tian; Jialun Wang; Chuan Lin; Guangjie Han; Shengchao Zhu; Ying Liu; Qian Zhu
>
> **摘要:** With the rapid advancement of underwater net-working and multi-agent coordination technologies, autonomous underwater vehicle (AUV) ad-hoc networks have emerged as a pivotal framework for executing complex maritime missions, such as multi-target tracking. However, traditional data-centricarchitectures struggle to maintain operational consistency under highly dynamic topological fluctuations and severely constrained acoustic communication bandwidth. This article proposes a scene-adaptive embodied intelligence (EI) architecture for multi-AUV ad-hoc networks, which re-envisions AUVs as embodied entities by integrating perception, decision-making, and physical execution into a unified cognitive loop. To materialize the functional interaction between these layers, we define a beacon-based communication and control model that treats the communication link as a dynamic constraint-aware channel, effectively bridging the gap between high-level policy inference and decentralized physical actuation. Specifically, the proposed architecture employs a three-layer functional framework and introduces a Scene-Adaptive MARL (SA-MARL) algorithm featuring a dual-path critic mechanism. By integrating a scene critic network and a general critic network through a weight-based dynamic fusion process, SA-MARL effectively decouples specialized tracking tasks from global safety constraints, facilitating autonomous policy evolution. Evaluation results demonstrate that the proposedscheme significantly accelerates policy convergence and achieves superior tracking accuracy compared to mainstream MARL approaches, maintaining robust performance even under intense environmental interference and fluid topological shifts.
>
---
#### [new 076] Online Inertia Tensor Identification for Non-Cooperative Spacecraft via Augmented UKF
- **分类: cs.RO**

- **简介: 该论文属于非合作航天器相对导航任务，解决参数不确定导致的导航失效问题。通过改进UKF框架，同时估计姿态和惯性张量，提升深空自主导航精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.27361](https://arxiv.org/pdf/2603.27361)**

> **作者:** Batu Candan; Simone Servadio
>
> **摘要:** Autonomous proximity operations, such as active debris removal and on-orbit servicing, require high-fidelity relative navigation solutions that remain robust in the presence of parametric uncertainty. Standard estimation frameworks typically assume that the target spacecraft's mass properties are known a priori; however, for non-cooperative or tumbling targets, these parameters are often unknown or uncertain, leading to rapid divergence in model-based propagators. This paper presents an augmented Unscented Kalman Filter (UKF) framework designed to jointly estimate the relative 6-DOF pose and the full inertia tensor of a non-cooperative target spacecraft. The proposed architecture fuses visual measurements from monocular vision-based Convolutional Neural Networks (CNN) with depth information from LiDAR to constrain the coupled rigid-body dynamics. By augmenting the state vector to include the six independent elements of the inertia tensor, the filter dynamically recovers the target's normalized mass distribution in real-time without requiring ground-based pre-calibration. To ensure numerical stability and physical consistency during the estimation of constant parameters, the filter employs an adaptive process noise formulation that prevents covariance collapse while allowing for the gradual convergence of the inertial parameters. Numerical validation is performed via Monte Carlo simulations, demonstrating that the proposed Augmented UKF enables the simultaneous convergence of kinematic states and inertial parameters, thereby facilitating accurate long-term trajectory prediction and robust guidance in non-cooperative deep-space environments.
>
---
#### [new 077] Predictive Modeling in AUV Navigation: A Perspective from Kalman Filtering
- **分类: cs.RO**

- **简介: 该论文属于AUV导航任务，解决通信丢失时的定位与搜索问题。通过融合多浮标TDOA数据与卡尔曼滤波，提升定位精度和轨迹预测能力，实现可靠搜索区域定义。**

- **链接: [https://arxiv.org/pdf/2603.27422](https://arxiv.org/pdf/2603.27422)**

> **作者:** Zizhan Tang; Yao Liu; Jessica Liu
>
> **备注:** 7pages and 9 figures
>
> **摘要:** We present a safety-oriented framework for autonomous underwater vehicles (AUVs) that improves localization accuracy, enhances trajectory prediction, and supports efficient search operations during communication loss. Acoustic signals emitted by the AUV are detected by a network of fixed buoys, which compute Time-Difference-of-Arrival (TDOA) range-difference measurements serving as position observations. These observations are subsequently fused with a Kalman-based prediction model to obtain continuous, noise-robust state estimates. The combined method achieves significantly better localization precision and trajectory stability than TDOA-only baselines. Beyond real-time tracking, our framework offers targeted search-and-recovery capability by predicting post-disconnection motion and explicitly modeling uncertainty growth. The search module differentiates between continued navigation and propulsion failure, allowing search resources to be deployed toward the most probable recovery region. Our framework fuses multi-buoy acoustic data with Kalman filtering and uncertainty propagation to maintain navigation accuracy and yield robust search-region definitions during communication loss.
>
---
#### [new 078] Motion as a Sensing Modality for Metric Scale in Monocular Visual-Inertial Odometry
- **分类: cs.RO**

- **简介: 该论文属于视觉-惯性里程计任务，解决单目VIO无法单独恢复度量尺度的问题。通过分析轨迹对可观测性的影响，提出基于惯性数据的激励度量，提升尺度恢复精度。**

- **链接: [https://arxiv.org/pdf/2603.26740](https://arxiv.org/pdf/2603.26740)**

> **作者:** Hadush Hailu; Bruk Gebregziabher
>
> **备注:** 10 pages
>
> **摘要:** Monocular visual-inertial odometry (VIO) cannot recover metric scale from vision alone; scale must be resolved through inertial measurements. We present a trajectory-dependent observability analysis showing that translational acceleration, produced by curvature, not constant-speed straight-line travel, is the fundamental source that couples scale to the inertial state. This relationship is formalized through the gravity-acceleration asymmetry in the IMU model, from which we derive rank conditions on the observability matrix and propose a lightweight excitation metric computable from raw IMU data. Controlled experiments on a differential-drive robot with a monocular camera and consumer-grade IMU validate the theory, with straight-line motion yielding 9.2% scale error, circular motion 6.4%, and figure-eight motion 4.8%, with excitation spanning four orders of magnitude. These results establish trajectory design as a practical mechanism for improving metric scale recovery.
>
---
#### [new 079] Tac2Real: Reliable and GPU Visuotactile Simulation for Online Reinforcement Learning and Zero-Shot Real-World Deployment
- **分类: cs.RO**

- **简介: 该论文属于机器人触觉模拟任务，旨在解决在线强化学习中仿真与现实的差距问题。提出Tac2Real框架和TacAlign方法，实现高效、可靠的零样本仿真到现实迁移。**

- **链接: [https://arxiv.org/pdf/2603.28475](https://arxiv.org/pdf/2603.28475)**

> **作者:** Ningyu Yan; Shuai Wang; Xing Shen; Hui Wang; Hanqing Wang; Yang Xiang; Jiangmiao Pang
>
> **备注:** 27 pages, 12 figures
>
> **摘要:** Visuotactile sensors are indispensable for contact-rich robotic manipulation tasks. However, policy learning with tactile feedback in simulation, especially for online reinforcement learning (RL), remains a critical challenge, as it demands a delicate balance between physics fidelity and computational efficiency. To address this challenge, we present Tac2Real, a lightweight visuotactile simulation framework designed to enable efficient online RL training. Tac2Real integrates the Preconditioned Nonlinear Conjugate Gradient Incremental Potential Contact (PNCG-IPC) method with a multi-node, multi-GPU high-throughput parallel simulation architecture, which can generate marker displacement fields at interactive rates. Meanwhile, we propose a systematic approach, TacAlign, to narrow both structured and stochastic sources of domain gap, ensuring a reliable zero-shot sim-to-real transfer. We further evaluate Tac2Real on the contact-rich peg insertion task. The zero-shot transfer results achieve a high success rate in the real-world scenario, verifying the effectiveness and robustness of our framework. The project page is: this https URL
>
---
#### [new 080] Serialized Red-Green-Gray: Quicker Heuristic Validation of Edges in Dynamic Roadmap Graphs
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决动态环境中道路图边快速验证问题。提出RGG框架，通过几何近似分类边的有效性，提升验证效率。**

- **链接: [https://arxiv.org/pdf/2603.28674](https://arxiv.org/pdf/2603.28674)**

> **作者:** Yulie Arad; Stav Ashur; Marta Markowicz; James D. Motes; Marco Morales; Nancy M. Amato
>
> **摘要:** Motion planning in dynamic environments, such as robotic warehouses, requires fast adaptation to frequent changes in obstacle poses. Traditional roadmap-based methods struggle in such settings, relying on inefficient reconstruction of a roadmap or expensive collision detection to update the existing roadmap. To address these challenges we introduce the Red-Green-Gray (RGG) framework, a method that builds on SPITE to quickly classify roadmap edges as invalid (red), valid (green), or uncertain (gray) using conservative geometric approximations. Serial RGG provides a high-performance variant leveraging batch serialization and vectorization to enable efficient GPU acceleration. Empirical results demonstrate that while RGG effectively reduces the number of unknown edges requiring full validation, SerRGG achieves a 2-9x speedup compared to the sequential implementation. This combination of geometric precision and computational speed makes SerRGG highly effective for time-critical robotic applications.
>
---
#### [new 081] Path-Following Guidance for Unmanned Aerial Vehicle with Bounded Lateral Acceleration
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于无人机路径跟踪任务，解决受限横向加速度下的引导问题。提出一种非线性引导框架，确保控制输入有限且误差指数收敛。**

- **链接: [https://arxiv.org/pdf/2603.27177](https://arxiv.org/pdf/2603.27177)**

> **作者:** Vinay Kathiriya; Saurabh Kumar; Shashi Ranjan Kumar
>
> **摘要:** This paper addresses the three-dimensional path-following guidance problem for unmanned aerial vehicles under explicit actuator constraints. Unlike conventional approaches that assume unbounded control inputs or handle saturation heuristically, the proposed method incorporates bounded lateral acceleration directly into the guidance design. A nonlinear guidance framework is developed employing a nested saturation-based control technique. The proposed guidance strategy guarantees bounded control inputs while ensuring exponential convergence of cross-track errors to zero. The formulation is applicable to general smooth paths and is systematically extended from planar to three-dimensional scenarios using a path-tangent coordinate framework. Rigorous stability analysis based on Lyapunov theory establishes convergence and feasibility properties of the closed-loop system. Numerical simulations on representative paths, including straight-line, circular, and sinusoidal paths, demonstrate that the proposed method achieves superior tracking performance, reduced control effort, and robustness against disturbances compared to existing guidance laws. The simplicity of the design and its compatibility with practical actuator limits make it suitable for real-world UAV applications.
>
---
#### [new 082] E-TIDE: Fast, Structure-Preserving Motion Forecasting from Event Sequences
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出E-TIDE，解决事件流中的运动预测任务，通过轻量结构实现高效、低功耗的未来事件表示预测。**

- **链接: [https://arxiv.org/pdf/2603.27757](https://arxiv.org/pdf/2603.27757)**

> **作者:** Biswadeep Sen; Benoit R. Cottereau; Nicolas Cuperlier; Terence Sim
>
> **摘要:** Event-based cameras capture visual information as asynchronous streams of per-pixel brightness changes, generating sparse, temporally precise data. Compared to conventional frame-based sensors, they offer significant advantages in capturing high-speed dynamics while consuming substantially less power. Predicting future event representations from past observations is an important problem, enabling downstream tasks such as future semantic segmentation or object tracking without requiring access to future sensor measurements. While recent state-of-the-art approaches achieve strong performance, they often rely on computationally heavy backbones and, in some cases, large-scale pretraining, limiting their applicability in resource-constrained scenarios. In this work, we introduce E-TIDE, a lightweight, end-to-end trainable architecture for event-tensor prediction that is designed to operate efficiently without large-scale pretraining. Our approach employs the TIDE module (Temporal Interaction for Dynamic Events), motivated by efficient spatiotemporal interaction design for sparse event tensors, to capture temporal dependencies via large-kernel mixing and activity-aware gating while maintaining low computational complexity. Experiments on standard event-based datasets demonstrate that our method achieves competitive performance with significantly reduced model size and training requirements, making it well-suited for real-time deployment under tight latency and memory budgets.
>
---
#### [new 083] Effort-Based Criticality Metrics for Evaluating 3D Perception Errors in Autonomous Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶感知误差评估任务，旨在解决传统关键性指标无法区分误报与漏报问题。提出三种基于努力的新指标，量化感知错误的安全影响。**

- **链接: [https://arxiv.org/pdf/2603.28029](https://arxiv.org/pdf/2603.28029)**

> **作者:** Sharang Kaul; Simon Bultmann; Mario Berk; Abhinav Valada
>
> **摘要:** Criticality metrics such as time-to-collision (TTC) quantify collision urgency but conflate the consequences of false-positive (FP) and false-negative (FN) perception errors. We propose two novel effort-based metrics: False Speed Reduction (FSR), the cumulative velocity loss from persistent phantom detections, and Maximum Deceleration Rate (MDR), the peak braking demand from missed objects under a constant-acceleration model. These longitudinal metrics are complemented by Lateral Evasion Acceleration (LEA), adapted from prior lateral evasion kinematics and coupled with reachability-based collision timing to quantify the minimum steering effort to avoid a predicted collision. A reachability-based ellipsoidal collision filter ensures only dynamically plausible threats are scored, with frame-level matching and track-level aggregation. Evaluation of different perception pipelines on nuScenes and Argoverse~2 shows that 65-93% of errors are non-critical, and Spearman correlation analysis confirms that all three metrics capture safety-relevant information inaccessible to established time-based, deceleration-based, or normalized criticality measures, enabling targeted mining of the most critical perception failures.
>
---
#### [new 084] An Annotation-to-Detection Framework for Autonomous and Robust Vine Trunk Localization in the Field by Mobile Agricultural Robots
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于农业机器人领域的物体检测任务，旨在解决无标记数据下 vine trunk 的定位问题。通过多模态数据融合与增量标注方法，提升检测鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.26724](https://arxiv.org/pdf/2603.26724)**

> **作者:** Dimitrios Chatziparaschis; Elia Scudiero; Brent Sams; Konstantinos Karydis
>
> **备注:** 7 pages, 6 figures, conference
>
> **摘要:** The dynamic and heterogeneous nature of agricultural fields presents significant challenges for object detection and localization, particularly for autonomous mobile robots that are tasked with surveying previously unseen unstructured environments. Concurrently, there is a growing need for real-time detection systems that do not depend on large-scale manually labeled real-world datasets. In this work, we introduce a comprehensive annotation-to-detection framework designed to train a robust multi-modal detector using limited and partially labeled training data. The proposed methodology incorporates cross-modal annotation transfer and an early-stage sensor fusion pipeline, which, in conjunction with a multi-stage detection architecture, effectively trains and enhances the system's multi-modal detection capabilities. The effectiveness of the framework was demonstrated through vine trunk detection in novel vineyard settings that featured diverse lighting conditions and varying crop densities to validate performance. When integrated with a customized multi-modal LiDAR and Odometry Mapping (LOAM) algorithm and a tree association module, the system demonstrated high-performance trunk localization, successfully identifying over 70% of trees in a single traversal with a mean distance error of less than 0.37m. The results reveal that by leveraging multi-modal, incremental-stage annotation and training, the proposed framework achieves robust detection performance regardless of limited starting annotations, showcasing its potential for real-world and near-ground agricultural applications.
>
---
#### [new 085] Deep Learning Aided Vision System for Planetary Rovers
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文提出一种用于行星探测器的视觉系统，结合实时感知与离线地形重建，解决自主导航中的距离估计和目标检测问题。**

- **链接: [https://arxiv.org/pdf/2603.26802](https://arxiv.org/pdf/2603.26802)**

> **作者:** Lomash Relia; Jai G Singla; Amitabh; Nitant Dube
>
> **摘要:** This study presents a vision system for planetary rovers, combining real-time perception with offline terrain reconstruction. The real-time module integrates CLAHE enhanced stereo imagery, YOLOv11n based object detection, and a neural network to estimate object distances. The offline module uses the Depth Anything V2 metric monocular depth estimation model to generate depth maps from captured images, which are fused into dense point clouds using Open3D. Real world distance estimates from the real time pipeline provide reliable metric context alongside the qualitative reconstructions. Evaluation on Chandrayaan 3 NavCam stereo imagery, benchmarked against a CAHV based utility, shows that the neural network achieves a median depth error of 2.26 cm within a 1 to 10 meter range. The object detection model maintains a balanced precision recall tradeoff on grayscale lunar scenes. This architecture offers a scalable, compute-efficient vision solution for autonomous planetary exploration.
>
---
#### [new 086] Engineering Mythology: A Digital-Physical Framework for Culturally-Inspired Public Art
- **分类: cs.GR; cs.CV; cs.CY; cs.RO**

- **简介: 该论文属于跨学科艺术与工程任务，旨在融合文化传统与现代技术。通过数字-物理流程，实现文化灵感公共艺术的创作与协作，解决文化传承与技术创新结合的问题。**

- **链接: [https://arxiv.org/pdf/2603.27801](https://arxiv.org/pdf/2603.27801)**

> **作者:** Jnaneshwar Das; Christopher Filkins; Rajesh Moharana; Ekadashi Barik; Bishweshwar Das; David Ayers; Christopher Skiba; Rodney Staggers Jr; Mark Dill; Swig Miller; Daniel Tulberg; Patrick Smith; Seth Brink; Kyle Breen; Harish Anand; Ramon Arrowsmith
>
> **备注:** 19 pages, 28 figures, 4 tables
>
> **摘要:** Navagunjara Reborn: The Phoenix of Odisha was built for Burning Man 2025 as both a sculpture and an experiment-a fusion of myth, craft, and computation. This paper describes the digital-physical workflow developed for the project: a pipeline that linked digital sculpting, distributed fabrication by artisans in Odisha (India), modular structural optimization in the U.S., iterative feedback through photogrammetry and digital twins, and finally, one-shot full assembly at the art site in Black Rock Desert, Nevada. The desert installation tested not just materials, but also systems of collaboration: between artisans and engineers, between myth and technology, between cultural specificity and global experimentation. We share the lessons learned in design, fabrication, and deployment and offer a framework for future interdisciplinary projects at the intersection of cultural heritage, STEAM education, and public art. In retrospect, this workflow can be read as a convergence of many knowledge systems-artisan practice, structural engineering, mythic narrative, and environmental constraint-rather than as execution of a single fixed blueprint.
>
---
#### [new 087] Critic-Free Deep Reinforcement Learning for Maritime Coverage Path Planning on Irregular Hexagonal Grids
- **分类: cs.LG; cs.AI; cs.NE; cs.RO**

- **简介: 该论文属于海洋覆盖路径规划任务，解决传统方法在复杂海域中的效率与适应性问题。提出一种基于Transformer的DRL框架，实现高效路径生成。**

- **链接: [https://arxiv.org/pdf/2603.28385](https://arxiv.org/pdf/2603.28385)**

> **作者:** Carlos S. Sepúlveda; Gonzalo A. Ruz
>
> **摘要:** Maritime surveillance missions, such as search and rescue and environmental monitoring, rely on the efficient allocation of sensing assets over vast and geometrically complex areas. Traditional Coverage Path Planning (CPP) approaches depend on decomposition techniques that struggle with irregular coastlines, islands, and exclusion zones, or require computationally expensive re-planning for every instance. We propose a Deep Reinforcement Learning (DRL) framework to solve CPP on hexagonal grid representations of irregular maritime areas. Unlike conventional methods, we formulate the problem as a neural combinatorial optimization task where a Transformer-based pointer policy autoregressively constructs coverage tours. To overcome the instability of value estimation in long-horizon routing problems, we implement a critic-free Group-Relative Policy Optimization (GRPO) scheme. This method estimates advantages through within-instance comparisons of sampled trajectories rather than relying on a value function. Experiments on 1,000 unseen synthetic maritime environments demonstrate that a trained policy achieves a 99.0% Hamiltonian success rate, more than double the best heuristic (46.0%), while producing paths 7% shorter and with 24% fewer heading changes than the closest baseline. All three inference modes (greedy, stochastic sampling, and sampling with 2-opt refinement) operate under 50~ms per instance on a laptop GPU, confirming feasibility for real-time on-board deployment.
>
---
#### [new 088] Decoupling Geometric Planning and Execution in Scalable Multi-Agent Path Finding
- **分类: cs.MA; cs.RO**

- **简介: 该论文属于多智能体路径规划任务，解决大规模场景下的路径冲突问题。提出分离几何规划与执行阶段的方法，提升可扩展性与效率。**

- **链接: [https://arxiv.org/pdf/2603.26684](https://arxiv.org/pdf/2603.26684)**

> **作者:** Fernando Salanova; Cristian Mahulea; Eduardo Montijano
>
> **备注:** 6 pages, 3 figures WODES submission
>
> **摘要:** Multi-Agent Path Finding (MAPF) requires collision-free trajectories for multiple agents on a shared graph, often with the objective of minimizing the sum-of-costs (SOC). Many optimal and bounded-suboptimal solvers rely on time-expanded models and centralized conflict resolution, which limits scalability in large or dense instances. We propose a hybrid prioritized framework that separates geometric planning from execution-time conflict resolution. In the first stage, Geometric Conflict Preemption (GCP) plans agents sequentially with A* on the original graph while inflating costs for transitions entering vertices used by higher-priority paths, encouraging spatial detours without explicit time reasoning. In the second stage, a Decentralized Local Controller (DLC) executes the geometric paths using per-vertex FIFO authorization queues and inserts wait actions only when required to avoid vertex and edge-swap conflicts. Experiments on standard benchmark maps with up to 1000 agents show that the method scales with an empirically near-linear runtime trend and attains a 100% success rate on instances satisfying the geometric feasibility assumption. On bottleneck-heavy maps, GCP reduces synchronization-induced waiting and often improves SOC on bottleneck-heavy maps
>
---
#### [new 089] Structured Observation Language for Efficient and Generalizable Vision-Language Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决环境变化下的泛化能力不足问题。提出SOL-Nav框架，将视觉信息转化为结构化语言描述，提升导航效率与泛化性。**

- **链接: [https://arxiv.org/pdf/2603.27577](https://arxiv.org/pdf/2603.27577)**

> **作者:** Daojie Peng; Fulong Ma; Jun Ma
>
> **摘要:** Vision-Language Navigation (VLN) requires an embodied agent to navigate complex environments by following natural language instructions, which typically demands tight fusion of visual and language modalities. Existing VLN methods often convert raw images into visual tokens or implicit features, requiring large-scale visual pre-training and suffering from poor generalization under environmental variations (e.g., lighting, texture). To address these issues, we propose SOL-Nav (Structured Observation Language for Navigation), a novel framework that translates egocentric visual observations into compact structured language descriptions for efficient and generalizable navigation. Specifically, we divide RGB-D images into a N*N grid, extract representative semantic, color, and depth information for each grid cell to form structured text, and concatenate this with the language instruction as pure language input to a pre-trained language model (PLM). Experimental results on standard VLN benchmarks (R2R, RxR) and real-world deployments demonstrate that SOL-Nav significantly reduces the model size and training data dependency, fully leverages the reasoning and representation capabilities of PLMs, and achieves strong generalization to unseen environments.
>
---
#### [new 090] Benchmarking Multi-View BEV Object Detection with Mixed Pinhole and Fisheye Cameras
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D目标检测任务，解决混合相机（针孔与鱼眼）下BEV检测性能下降的问题。通过数据转换、视图变换模块和极坐标表示等方法，提升模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.27818](https://arxiv.org/pdf/2603.27818)**

> **作者:** Xiangzhong Liu; Hao Shen
>
> **备注:** 8 pages,5 figures, IEEE International Conference on Robotics and Automation (ICRA),Vienna, Austria, 1-5 June 2026
>
> **摘要:** Modern autonomous driving systems increasingly rely on mixed camera configurations with pinhole and fisheye cameras for full view perception. However, Bird's-Eye View (BEV) 3D object detection models are predominantly designed for pinhole cameras, leading to performance degradation under fisheye distortion. To bridge this gap, we introduce a multi-view BEV detection benchmark with mixed cameras by converting KITTI-360 into nuScenes format. Our study encompasses three adaptations: rectification for zero-shot evaluation and fine-tuning of nuScenes-trained models, distortion-aware view transformation modules (VTMs) via the MEI camera model, and polar coordinate representations to better align with radial distortion. We systematically evaluate three representative BEV architectures, BEVFormer, BEVDet and PETR, across these strategies. We demonstrate that projection-free architectures are inherently more robust and effective against fisheye distortion than other VTMs. This work establishes the first real-data 3D detection benchmark with fisheye and pinhole images and provides systematic adaptation and practical guidelines for designing robust and cost-effective 3D perception systems. The code is available at this https URL.
>
---
#### [new 091] MPC as a Copilot: A Predictive Filter Framework with Safety and Stability Guarantees
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于控制领域，解决学习控制中安全与稳定不足的问题，提出PS2F框架，通过两级优化确保约束满足和系统稳定。**

- **链接: [https://arxiv.org/pdf/2603.27893](https://arxiv.org/pdf/2603.27893)**

> **作者:** Yunda Yan; Chenxi Tao; Jinya Su; Cunjia Liu; Shihua Li
>
> **备注:** 21 pages, 11 figures, 1 table
>
> **摘要:** Ensuring both safety and stability remains a fundamental challenge in learning-based control, where goal-oriented policies often neglect system constraints and closed-loop state convergence. To address this limitation, this paper introduces the Predictive Safety--Stability Filter (PS2F), a unified predictive filter framework that guarantees constraint satisfaction and asymptotic stability within a single architecture. The PS2F framework comprises two cascaded optimal control problems: a nominal model predictive control (MPC) layer that serves solely as a copilot, implicitly defining a Lyapunov function and generating safety- and stability-certified predicted trajectories, and a secondary filtering layer that adjusts external command to remain within a provably safe and stable region. This cascaded structure enables PS2F to inherit the theoretical guarantees of nominal MPC while accommodating goal-oriented external commands. Rigorous analysis establishes recursive feasibility and asymptotic stability of the closed-loop system without introducing additional conservatism beyond that associated with the nominal MPC. Furthermore, a time-varying parameterisation allows PS2F to transition smoothly between safety-prioritised and stability-oriented operation modes, providing a principled mechanism for balancing exploration and exploitation. The effectiveness of the proposed framework is demonstrated through comparative numerical experiments.
>
---
#### [new 092] SHARP: Short-Window Streaming for Accurate and Robust Prediction in Motion Forecasting
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于运动预测任务，解决动态环境中异构观测长度下的预测准确性问题。提出SHARP框架，通过增量处理和上下文流提升预测鲁棒性与精度。**

- **链接: [https://arxiv.org/pdf/2603.28091](https://arxiv.org/pdf/2603.28091)**

> **作者:** Alexander Prutsch; Christian Fruhwirth-Reisinger; David Schinagl; Horst Possegger
>
> **备注:** CVPR 2026. Project page at this https URL
>
> **摘要:** In dynamic traffic environments, motion forecasting models must be able to accurately estimate future trajectories continuously. Streaming-based methods are a promising solution, but despite recent advances, their performance often degrades when exposed to heterogeneous observation lengths. To address this, we propose a novel streaming-based motion forecasting framework that explicitly focuses on evolving scenes. Our method incrementally processes incoming observation windows and leverages an instance-aware context streaming to maintain and update latent agent representations across inference steps. A dual training objective further enables consistent forecasting accuracy across diverse observation horizons. Extensive experiments on Argoverse 2, nuScenes, and Argoverse 1 demonstrate the robustness of our approach under evolving scene conditions and also on the single-agent benchmarks. Our model achieves state-of-the-art performance in streaming inference on the Argoverse 2 multi-agent benchmark, while maintaining minimal latency, highlighting its suitability for real-world deployment.
>
---
#### [new 093] Language-Conditioned World Modeling for Visual Navigation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文研究语言条件视觉导航任务，解决在无目标图像情况下，根据语言指令进行导航的问题。构建了LCVN数据集，并提出两种框架实现语言理解、状态预测与动作生成的联合学习。**

- **链接: [https://arxiv.org/pdf/2603.26741](https://arxiv.org/pdf/2603.26741)**

> **作者:** Yifei Dong; Fengyi Wu; Yilong Dai; Lingdong Kong; Guangyu Chen; Xu Zhu; Qiyu Hu; Tianyu Wang; Johnalbert Garnica; Feng Liu; Siyu Huang; Qi Dai; Zhi-Qi Cheng
>
> **备注:** 19 pages, 6 figures, Code: this https URL
>
> **摘要:** We study language-conditioned visual navigation (LCVN), in which an embodied agent is asked to follow a natural language instruction based only on an initial egocentric observation. Without access to goal images, the agent must rely on language to shape its perception and continuous control, making the grounding problem particularly challenging. We formulate this problem as open-loop trajectory prediction conditioned on linguistic instructions and introduce the LCVN Dataset, a benchmark of 39,016 trajectories and 117,048 human-verified instructions that supports reproducible research across a range of environments and instruction styles. Using this dataset, we develop LCVN frameworks that link language grounding, future-state prediction, and action generation through two complementary model families. The first family combines LCVN-WM, a diffusion-based world model, with LCVN-AC, an actor-critic agent trained in the latent space of the world model. The second family, LCVN-Uni, adopts an autoregressive multimodal architecture that predicts both actions and future observations. Experiments show that these families offer different advantages: the former provides more temporally coherent rollouts, whereas the latter generalizes better to unseen environments. Taken together, these observations point to the value of jointly studying language grounding, imagination, and policy learning in a unified task setting, and LCVN provides a concrete basis for further investigation of language-conditioned world models. The code is available at this https URL.
>
---
#### [new 094] Users and Wizards in Conversations: How WoZ Interface Choices Define Human-Robot Interactions
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互任务，研究WoZ接口对人机对话的影响。通过对比不同界面，分析其对用户和操作者体验的影响，旨在提升机器人社交互动质量。**

- **链接: [https://arxiv.org/pdf/2603.28338](https://arxiv.org/pdf/2603.28338)**

> **作者:** Ekaterina Torubarova; Jura Miniota; Andre Pereira
>
> **备注:** Published in Robotics: Science and Systems (2025)
>
> **摘要:** In this paper, we investigated how the choice of a Wizard-of-Oz (WoZ) interface affects communication with a robot from both the user's and the wizard's perspective. In a conversational setting, we used three WoZ interfaces with varying levels of dialogue input and output restrictions: a) a restricted perception GUI that showed fixed-view video and ASR transcripts and let the wizard trigger pre-scripted utterances and gestures; b) an unrestricted perception GUI that added real-time audio from the participant and the robot c) a VR telepresence interface that streamed immersive stereo video and audio to the wizard and forwarded the wizard's spontaneous speech, gaze and facial expressions to the robot. We found that the interaction mediated by the VR interface was preferred by users in terms of robot features and perceived social presence. For the wizards, the VR condition turned out to be the most demanding but elicited a higher social connection with the users. VR interface also induced the most connected interaction in terms of inter-speaker gaps and overlaps, while Restricted GUI induced the least connected flow and the largest silences. Given these results, we argue for more WoZ studies using telepresence interfaces. These studies better reflect the robots of tomorrow and offer a promising path to automation based on naturalistic contextualized verbal and non-verbal behavioral data.
>
---
#### [new 095] Data is All You Need: Markov Chain Car-Following (MC-CF) Model
- **分类: eess.SY; cs.LG; cs.RO**

- **简介: 该论文属于交通流建模任务，旨在解决传统模型无法准确捕捉驾驶随机性的问题。通过提出MC-CF模型，利用马尔可夫链和实证分布进行轨迹预测，显著提升预测精度与真实性。**

- **链接: [https://arxiv.org/pdf/2603.27909](https://arxiv.org/pdf/2603.27909)**

> **作者:** Sungyong Chung; Yanlin Zhang; Nachuan Li; Dana Monzer; Alireza Talebpour
>
> **摘要:** Car-following behavior is fundamental to traffic flow theory, yet traditional models often fail to capture the stochasticity of naturalistic driving. This paper introduces a new car-following modeling category called the empirical probabilistic paradigm, which bypasses conventional parametric assumptions. Within this paradigm, we propose the Markov Chain Car-Following (MC-CF) model, which represents state transitions as a Markov process and predicts behavior by randomly sampling accelerations from empirical distributions within discretized state bins. Evaluation of the MC-CF model trained on the Waymo Open Motion Dataset (WOMD) demonstrates that its variants significantly outperform physics-based models including IDM, Gipps, FVDM, and SIDM in both one-step and open-loop trajectory prediction accuracy. Statistical analysis of transition probabilities confirms that the model-generated trajectories are indistinguishable from real-world behavior, successfully reproducing the probabilistic structure of naturalistic driving across all interaction types. Zero-shot generalization on the Naturalistic Phoenix (PHX) dataset further confirms the model's robustness. Finally, microscopic ring road simulations validate the framework's scalability. By incrementally integrating unconstrained free-flow trajectories and high-speed freeway data (TGSIM) alongside a conservative inference strategy, the model drastically reduces collisions, achieving zero crashes in multiple equilibrium and shockwave scenarios, while successfully reproducing naturalistic and stochastic shockwave propagation. Overall, the proposed MC-CF model provides a robust, scalable, and calibration-free foundation for high-fidelity stochastic traffic modeling, uniquely suited for the data-rich future of intelligent transportation.
>
---
#### [new 096] arg-VU: Affordance Reasoning with Physics-Aware 3D Geometry for Visual Understanding in Robotic Surgery
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉理解任务，解决手术中软组织变形带来的感知与动作关联问题。提出arg-VU框架，结合物理约束与3D几何，提升手术场景的 affordance 预测准确性。**

- **链接: [https://arxiv.org/pdf/2603.26814](https://arxiv.org/pdf/2603.26814)**

> **作者:** Nan Xiao; Yunxin Fan; Farong Wang; Fei Liu
>
> **摘要:** Affordance reasoning provides a principled link between perception and action, yet remains underexplored in surgical robotics, where tissues are highly deformable, compliant, and dynamically coupled with tool motion. We present arg-VU, a physics-aware affordance reasoning framework that integrates temporally consistent geometry tracking with constraint-induced mechanical modeling for surgical visual understanding. Surgical scenes are reconstructed using 3D Gaussian Splatting (3DGS) and converted into a temporally tracked surface representation. Extended Position-Based Dynamics (XPBD) embeds local deformation constraints and produces representative geometry points (RGPs) whose constraint sensitivities define anisotropic stiffness metrics capturing the local constraint-manifold geometry. Robotic tool poses in SE(3) are incorporated to compute rigidly induced displacements at RGPs, from which we derive two complementary measures: a physics-aware compliance energy that evaluates mechanical feasibility with respect to local deformation constraints, and a positional agreement score that captures motion alignment (as kinematic motion baseline). Experiments on surgical video datasets show that arg-VU yields more stable, physically consistent, and interpretable affordance predictions than kinematic baselines. These results demonstrate that physics-aware geometric representations enable reliable affordance reasoning for deformable surgical environments and support embodied robotic interaction.
>
---
#### [new 097] Neural Aided Adaptive Innovation-Based Invariant Kalman Filter
- **分类: eess.SY; cs.RO; eess.SP**

- **简介: 该论文属于自主导航任务，解决传感器融合中的噪声估计问题。结合不变滤波与神经网络，提出一种改进的卡尔曼滤波方法，提升水下导航精度。**

- **链接: [https://arxiv.org/pdf/2603.26709](https://arxiv.org/pdf/2603.26709)**

> **作者:** Barak Diker; Itzik Klein
>
> **备注:** 11 pages and 2 figures
>
> **摘要:** Autonomous platforms require accurate positioning to complete their tasks. To this end, a Kalman filter-based algorithms, such as the extended Kalman filter or invariant Kalman filter, utilizing inertial and external sensor fusion are applied. To cope with real-world scenarios, adaptive noise estimation methods have been developed primarily for classical Euclidean formulations. However, these methods remain largely unexplored in the tangent Lie space, despite it provides a principled geometric framework with favorable error dynamics on Lie groups. To fill this gap, we combine invariant filtering theory with neural-aided adaptive noise estimation in real-world settings. To this end, we derive a novel theoretical extension of classical innovation-based process noise adaptation formulated directly within the Lie-group framework. We further propose a lightweight neural network that estimates the process noise covariance parameters directly from raw inertial data. Trained entirely in a sim2real framework via domain adaptation, the network captures motion-dependent and sensor-dependent noise characteristics without requiring labeled real-world data. To examine our proposed neural-aided adaptive invariant Kalman filter, we focus on the challenging real-world scenario of autonomous underwater navigation. Experimental results demonstrate superior performance compared to existing methods in terms of position root mean square error. These results validate our sim2real pipeline and further confirm that geometric invariance significantly enhances learning-based adaptation and that adaptive noise estimation in the tangent Lie space offers a powerful mechanism for improving navigation accuracy in nonlinear autonomous platforms.
>
---
#### [new 098] SHOW3D: Capturing Scenes of 3D Hands and Objects in the Wild
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于人手与物体交互的3D理解任务，旨在解决真实环境中手和物体3D标注不准确的问题。提出SHOW3D数据集和多相机系统，实现高精度3D标注与真实环境结合。**

- **链接: [https://arxiv.org/pdf/2603.28760](https://arxiv.org/pdf/2603.28760)**

> **作者:** Patrick Rim; Kevin Harris; Braden Copple; Shangchen Han; Xu Xie; Ivan Shugurov; Sizhe An; He Wen; Alex Wong; Tomas Hodan; Kun He
>
> **备注:** CVPR 2026
>
> **摘要:** Accurate 3D understanding of human hands and objects during manipulation remains a significant challenge for egocentric computer vision. Existing hand-object interaction datasets are predominantly captured in controlled studio settings, which limits both environmental diversity and the ability of models trained on such data to generalize to real-world scenarios. To address this challenge, we introduce a novel marker-less multi-camera system that allows for nearly unconstrained mobility in genuinely in-the-wild conditions, while still having the ability to generate precise 3D annotations of hands and objects. The capture system consists of a lightweight, back-mounted, multi-camera rig that is synchronized and calibrated with a user-worn VR headset. For 3D ground-truth annotation of hands and objects, we develop an ego-exo tracking pipeline and rigorously evaluate its quality. Finally, we present SHOW3D, the first large-scale dataset with 3D annotations that show hands interacting with objects in diverse real-world environments, including outdoor settings. Our approach significantly reduces the fundamental trade-off between environmental realism and accuracy of 3D annotations, which we validate with experiments on several downstream tasks. this http URL
>
---
#### [new 099] Liquid Networks with Mixture Density Heads for Efficient Imitation Learning
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于模仿学习任务，旨在提升策略的效率与鲁棒性。通过对比液体神经网络与扩散策略，验证了其在参数量、预测误差和推理速度上的优势。**

- **链接: [https://arxiv.org/pdf/2603.27058](https://arxiv.org/pdf/2603.27058)**

> **作者:** Nikolaus Correll
>
> **摘要:** We compare liquid neural networks with mixture density heads against diffusion policies on Push-T, RoboMimic Can, and PointMaze under a shared-backbone comparison protocol that isolates policy-head effects under matched inputs, training budgets, and evaluation settings. Across tasks, liquid policies use roughly half the parameters (4.3M vs. 8.6M), achieve 2.4x lower offline prediction error, and run 1.8 faster at inference. In sample-efficiency experiments spanning 1% to 46.42% of training data, liquid models remain consistently more robust, with especially large gains in low-data and medium-data regimes. Closed-loop results on Push-T and PointMaze are directionally consistent with offline rankings but noisier, indicating that strong offline density modeling helps deployment while not fully determining closed-loop success. Overall, liquid recurrent multimodal policies provide a compact and practical alternative to iterative denoising for imitation learning.
>
---
#### [new 100] Detection of Adversarial Attacks in Robotic Perception
- **分类: cs.CV; cs.AI; cs.CR; cs.RO**

- **简介: 论文属于机器人感知任务，旨在检测深度神经网络在语义分割中面临的对抗攻击，解决其安全性问题，提出针对性的检测策略。**

- **链接: [https://arxiv.org/pdf/2603.28594](https://arxiv.org/pdf/2603.28594)**

> **作者:** Ziad Sharawy; Mohammad Nakshbandiand; Sorin Mihai Grigorescu
>
> **备注:** 9 pages, 6 figures. Accepted and presented at STE 2025, Transilvania University of Brasov, Romania
>
> **摘要:** Deep Neural Networks (DNNs) achieve strong performance in semantic segmentation for robotic perception but remain vulnerable to adversarial attacks, threatening safety-critical applications. While robustness has been studied for image classification, semantic segmentation in robotic contexts requires specialized architectures and detection strategies.
>
---
#### [new 101] Sim-to-Real Fruit Detection Using Synthetic Data: Quantitative Evaluation and Embedded Deployment with Isaac Sim
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于目标检测任务，研究合成数据在模拟到真实场景中的有效性，解决数据不足和部署限制问题。通过对比不同训练策略，验证了合成与真实数据结合的优越性。**

- **链接: [https://arxiv.org/pdf/2603.28670](https://arxiv.org/pdf/2603.28670)**

> **作者:** Martina Hutter-Mironovova
>
> **备注:** 18 pages, 6 figures
>
> **摘要:** This study investigates the effectiveness of synthetic data for sim-to-real transfer in object detection under constrained data conditions and embedded deployment requirements. Synthetic datasets were generated in NVIDIA Isaac Sim and combined with limited real-world fruit images to train YOLO-based detection models under real-only, synthetic-only, and hybrid regimes. Performance was evaluated on two test datasets: an in-domain dataset with conditions matching the training data and a domain shift dataset containing real fruit and different background conditions. Results show that models trained exclusively on real data achieve the highest accuracy, while synthetic-only models exhibit reduced performance due to a domain gap. Hybrid training strategies significantly improve performance compared to synthetic-only approaches and achieve results close to real-only training while reducing the need for manual annotation. Under domain shift conditions, all models show performance degradation, with hybrid models providing improved robustness. The trained models were successfully deployed on a Jetson Orin NX using TensorRT optimization, achieving real-time inference performance. The findings highlight that synthetic data is most effective when used in combination with real data and that deployment constraints must be considered alongside detection accuracy.
>
---
#### [new 102] Kernel Dynamics under Path Entropy Maximization
- **分类: cs.LG; cs.AI; cs.RO; math.DS**

- **简介: 该论文提出一种基于路径熵最大化的核函数动态框架，研究核函数在信息几何中的演化，解决核结构优化问题，探讨其在神经网络和科学理论中的应用。**

- **链接: [https://arxiv.org/pdf/2603.27880](https://arxiv.org/pdf/2603.27880)**

> **作者:** Jnaneshwar Das
>
> **备注:** 7 pages, 2 figures
>
> **摘要:** We propose a variational framework in which the kernel function k : X x X -> R, interpreted as the foundational object encoding what distinctions an agent can represent, is treated as a dynamical variable subject to path entropy maximization (Maximum Caliber, MaxCal). Each kernel defines a representational structure over which an information geometry on probability space may be analyzed; a trajectory through kernel space therefore corresponds to a trajectory through a family of effective geometries, making the optimization landscape endogenous to its own traversal. We formulate fixed-point conditions for self-consistent kernels, propose renormalization group (RG) flow as a structured special case, and suggest neural tangent kernel (NTK) evolution during deep network training as a candidate empirical instantiation. Under explicit information-thermodynamic assumptions, the work required for kernel change is bounded below by delta W >= k_B T delta I_k, where delta I_k is the mutual information newly unlocked by the updated kernel. In this view, stable fixed points of MaxCal over kernels correspond to self-reinforcing distinction structures, with biological niches, scientific paradigms, and craft mastery offered as conjectural interpretations. We situate the framework relative to assembly theory and the MaxCal literature, separate formal results from structured correspondences and conjectural bridges, and pose six open questions that make the program empirically and mathematically testable.
>
---
## 更新

#### [replaced 001] Scaling Spatial Intelligence with Multimodal Foundation Models
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.RO**

- **简介: 该论文属于多模态基础模型任务，旨在提升模型的空间智能。通过构建大规模数据集，增强模型在多个空间智能基准上的表现，并分析数据扩展与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.13719](https://arxiv.org/pdf/2511.13719)**

> **作者:** Zhongang Cai; Ruisi Wang; Chenyang Gu; Fanyi Pu; Junxiang Xu; Yubo Wang; Wanqi Yin; Zhitao Yang; Chen Wei; Qingping Sun; Tongxi Zhou; Jiaqi Li; Hui En Pang; Oscar Qian; Yukun Wei; Zhiqian Lin; Xuanke Shi; Kewang Deng; Xiaoyang Han; Zukai Chen; Xiangyu Fan; Hanming Deng; Lewei Lu; Liang Pan; Bo Li; Ziwei Liu; Quan Wang; Dahua Lin; Lei Yang
>
> **备注:** Codebase: this https URL ; Models: this https URL . This report is based on the v1.1 version of SenseNova-SI. Accepted to CVPR 2026
>
> **摘要:** Despite remarkable progress, multimodal foundation models still exhibit surprising deficiencies in spatial intelligence. In this work, we explore scaling up multimodal foundation models to cultivate spatial intelligence within the SenseNova-SI family, built upon established multimodal foundations including visual understanding models (i.e., Qwen3-VL and InternVL3) and unified understanding and generation models (i.e., Bagel). We take a principled approach to constructing high-performing and robust spatial intelligence by systematically curating SenseNova-SI-8M: eight million diverse data samples under a rigorous taxonomy of spatial capabilities. SenseNova-SI demonstrates unprecedented performance across a broad range of spatial intelligence benchmarks: 68.8% on VSI-Bench, 43.3% on MMSI, 85.7% on MindCube, 54.7% on ViewSpatial, 47.7% on SITE, 63.9% on BLINK, 55.5% on 3DSR, and 72.0% on EmbSpatial, while maintaining strong general multimodal understanding (e.g., 84.9% on MMBench-En). More importantly, we analyze the impact of data scaling, discuss early signs of emergent generalization capabilities enabled by diverse data training, analyze the risk of overfitting and language shortcuts, present a preliminary study on spatial chain-of-thought reasoning, and validate the potential downstream application. All newly trained multimodal foundation models are publicly released.
>
---
#### [replaced 002] 3D CAVLA: Leveraging Depth and 3D Context to Generalize Vision Language Action Models for Unseen Tasks
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出3D-CAVLA，解决机器人在3D环境中任务泛化问题。通过引入深度感知、思维链和区域检测，提升视觉语言动作模型的泛化能力。**

- **链接: [https://arxiv.org/pdf/2505.05800](https://arxiv.org/pdf/2505.05800)**

> **作者:** Vineet Bhat; Yu-Hsiang Lan; Prashanth Krishnamurthy; Ramesh Karri; Farshad Khorrami
>
> **备注:** Accepted at the 1st Workshop on 3D LLM/VLA, CVPR 2025. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Robotic manipulation in 3D requires effective computation of N degree-of-freedom joint-space trajectories that enable precise and robust control. To achieve this, robots must integrate semantic understanding with visual perception to transform real-world observations into low-level control for object interaction. Recent advances in Vision-Language-Action (VLA) models have shown promise by mapping RGB images and language instructions to task space velocities, typically trained on large datasets of teleoperated demonstrations. However, these models often struggle with generalization beyond their training distributions. In this work, we introduce 3D-CAVLA, a novel finetuning framework that enhances task generalization of VLA policies by incorporating three key components: (i) chain-of-thought reasoning for structured decision-making, (ii) depth-aware perception for 3D spatial understanding, and (iii) task-oriented region-of-interest detection for focused manipulation. Extensive experiments in the LIBERO simulation environment demonstrate that 3D-CAVLA achieves an average success rate of 98.1% across diverse in-domain task suites. On unseen tasks, 3D-CAVLA delivers an absolute improvement of 8.8% in success rate, underscoring the benefits of 3D scene awareness for robust generalization. We validate our approach on real-world tabletop experiments demonstrating that the proposed model translates effectively from simulation to physical robots. 3D-CAVLA achieves over a 3X faster training convergence and delivers a 25% gain in success rate on unseen real world tasks. We will open-source our code and the unseen tasks dataset to promote community-driven research here: this https URL
>
---
#### [replaced 003] Object-Reconstruction-Aware Whole-body Control of Mobile Manipulators
- **分类: cs.RO**

- **简介: 该论文属于移动机械臂的全身控制任务，旨在提高物体重建效率。通过计算关键视点并保持其在视野内，减少计算成本，提升路径规划速度。**

- **链接: [https://arxiv.org/pdf/2509.04094](https://arxiv.org/pdf/2509.04094)**

> **作者:** Fatih Dursun; Bruno Vilhena Adorno; Simon Watson; Wei Pan
>
> **备注:** 19 pages, 17 figures, 5 tables. Under Review for the IEEE Transactions on Robotics (T-RO)
>
> **摘要:** Object reconstruction and inspection tasks play a crucial role in various robotics applications. Identifying paths that reveal the most unknown areas of the object is paramount in this context, as it directly affects reconstruction efficiency. Current methods often use sampling based path planning techniques, evaluating views along the path to enhance reconstruction performance. However, these methods are computationally expensive as they require evaluating several candidate views on the path. To this end, we propose a computationally efficient solution that relies on calculating a focus point in the most informative region and having the robot maintain this point in the camera field of view along the path. In this way, object reconstruction related information is incorporated into the whole body control of a mobile manipulator employing a visibility constraint without the need for an additional path planner. We conducted comprehensive and realistic simulations using a large dataset of 114 diverse objects of varying sizes from 57 categories to compare our method with a sampling based planning strategy and a strategy that does not employ informative paths using Bayesian data analysis. Furthermore, to demonstrate the applicability and generality of the proposed approach, we conducted real world experiments with an 8 DoF omnidirectional mobile manipulator and a legged manipulator. Our results suggest that, compared to a sampling based strategy, there is no statistically significant difference in object reconstruction entropy, and there is a 52.3% probability that they are practically equivalent in terms of coverage. In contrast, our method is 6.2 to 19.36 times faster in terms of computation time and reduces the total time the robot spends between views by 13.76% to 27.9%, depending on the camera FoV and model resolution.
>
---
#### [replaced 004] RobotSeg: A Model and Dataset for Segmenting Robots in Image and Video
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出RobotSeg，解决机器人图像和视频分割任务。针对机器人结构复杂、形态多样的问题，改进分割模型，提升准确性与效率。**

- **链接: [https://arxiv.org/pdf/2511.22950](https://arxiv.org/pdf/2511.22950)**

> **作者:** Haiyang Mei; Qiming Huang; Hai Ci; Mike Zheng Shou
>
> **备注:** CVPR 2026. Project page: this https URL
>
> **摘要:** Accurate robot segmentation is a fundamental capability for robotic perception. It enables precise visual servoing for VLA systems, scalable robot-centric data augmentation, accurate real-to-sim transfer, and reliable safety monitoring in dynamic human-robot environments. Despite the strong capabilities of modern segmentation models, surprisingly it remains challenging to segment robots. This is due to robot embodiment diversity, appearance ambiguity, structural complexity, and rapid shape changes. Embracing these challenges, we introduce RobotSeg, a foundation model for robot segmentation in image and video. RobotSeg is built upon the versatile SAM 2 foundation model but addresses its three limitations for robot segmentation, namely the lack of adaptation to articulated robots, reliance on manual prompts, and the need for per-frame training mask annotations, by introducing a structure-enhanced memory associator, a robot prompt generator, and a label-efficient training strategy. These innovations collectively enable a structure-aware, automatic, and label-efficient solution. We further construct the video robot segmentation (VRS) dataset comprising over 2.8k videos (138k frames) with diverse robot embodiments and environments. Extensive experiments demonstrate that RobotSeg achieves state-of-the-art performance on both images and videos, establishing a strong foundation for future advances in robot perception.
>
---
#### [replaced 005] Mimic Intent, Not Just Trajectories
- **分类: cs.RO**

- **简介: 该论文属于模仿学习任务，旨在解决环境变化适应与技能迁移问题。通过分离行为意图与执行细节，提出MINT方法提升泛化与效率。**

- **链接: [https://arxiv.org/pdf/2602.08602](https://arxiv.org/pdf/2602.08602)**

> **作者:** Renming Huang; Chendong Zeng; Wenjing Tang; Jintian Cai; Cewu Lu; Panpan Cai
>
> **摘要:** While imitation learning (IL) has achieved impressive success in dexterous manipulation through generative modeling and pretraining, state-of-the-art approaches like Vision-Language-Action (VLA) models still struggle with adaptation to environmental changes and skill transfer. We argue this stems from mimicking raw trajectories without understanding the underlying intent. To address this, we propose explicitly disentangling behavior intent from execution details in end-2-end IL: Mimic Intent, Not just Trajectories(MINT). We achieve this via multi-scale frequency-space tokenization, which enforces a spectral decomposition of action chunk representation. We learn action tokens with a multi-scale coarse-to-fine structure, and force the coarsest token to capture low-frequency global structure and finer tokens to encode high-frequency details. This yields an abstract Intent token that facilitates planning and transfer, and multi-scale Execution tokens that enable precise adaptation to environmental dynamics. Building on this hierarchy, our policy generates trajectories through next-scale autoregression, performing progressive intent-to-execution reasoning, thus boosting learning efficiency and generalization. Crucially, this disentanglement enables one-shot transfer of skills, by simply injecting the Intent token from a demonstration into the autoregressive generation process. Experiments on several manipulation benchmarks and on a real robot demonstrate state-of-the-art success rates, superior inference efficiency, robust generalization against disturbances, and effective one-shot transfer.
>
---
#### [replaced 006] AffordGrasp: Cross-Modal Diffusion for Affordance-Aware Grasp Synthesis
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言生成任务，旨在解决3D物体与文本指令间语义不一致的问题。提出AffordGrasp框架，结合扩散模型与语义标注，生成物理合理且语义准确的抓取姿态。**

- **链接: [https://arxiv.org/pdf/2603.08021](https://arxiv.org/pdf/2603.08021)**

> **作者:** Xiaofei Wu; Yi Zhang; Yumeng Liu; Yuexin Ma; Yujiao Shi; Xuming He
>
> **备注:** CVPR 2026
>
> **摘要:** Generating human grasping poses that accurately reflect both object geometry and user-specified interaction semantics is essential for natural hand-object interactions in AR/VR and embodied AI. However, existing semantic grasping approaches struggle with the large modality gap between 3D object representations and textual instructions, and often lack explicit spatial or semantic constraints, leading to physically invalid or semantically inconsistent grasps. In this work, we present AffordGrasp, a diffusion-based framework that produces physically stable and semantically faithful human grasps with high precision. We first introduce a scalable annotation pipeline that automatically enriches hand-object interaction datasets with fine-grained structured language labels capturing interaction intent. Building upon these annotations, AffordGrasp integrates an affordance-aware latent representation of hand poses with a dual-conditioning diffusion process, enabling the model to jointly reason over object geometry, spatial affordances, and instruction semantics. A distribution adjustment module further enforces physical contact consistency and semantic alignment. We evaluate AffordGrasp across four instruction-augmented benchmarks derived from HO-3D, OakInk, GRAB, and AffordPose, and observe substantial improvements over state-of-the-art methods in grasp quality, semantic accuracy, and diversity.
>
---
#### [replaced 007] Onboard MuJoCo-based Model Predictive Control for Shipboard Crane with Double-Pendulum Sway Suppression
- **分类: cs.RO**

- **简介: 该论文属于船舶起重机控制任务，旨在解决双摆振荡问题。通过基于MuJoCo的模型预测控制，实现实时有效抑制外部扰动，优于传统方法。**

- **链接: [https://arxiv.org/pdf/2603.16407](https://arxiv.org/pdf/2603.16407)**

> **作者:** Oscar Pang; Lisa Coiffard; Paul Templier; Luke Beddow; Kamil Dreczkowski; Antoine Cully
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Transferring heavy payloads in maritime settings relies on efficient crane operation, limited by hazardous double-pendulum payload sway. This sway motion is further exacerbated in offshore environments by external perturbations from wind and ocean waves. Manual suppression of these oscillations on an underactuated crane system by human operators is challenging. Existing control methods struggle in such settings, often relying on simplified analytical models, while deep reinforcement learning (RL) approaches tend to generalise poorly to unseen conditions. Deploying a predictive controller onto compute-constrained, highly non-linear physical systems without relying on extensive offline training or complex analytical models remains a significant challenge. Here we show a complete real-time control pipeline centered on the MuJoCo MPC framework that leverages a cross-entropy method planner to evaluate candidate action sequences directly within a physics simulator. By using simulated rollouts, this sampling-based approach successfully reconciles the conflicting objectives of dynamic target tracking and sway damping without relying on complex analytical models. We demonstrate that the controller can run effectively on a resource-constrained embedded hardware, while outperforming traditional PID and RL baselines in counteracting external base perturbations. Furthermore, our system demonstrates robustness even when subjected to unmodeled physical discrepancies like the introduction of a second payload.
>
---
#### [replaced 008] Optimal Solutions for the Moving Target Vehicle Routing Problem with Obstacles via Lazy Branch and Price
- **分类: cs.RO**

- **简介: 该论文研究MT-VRP-O任务，解决多代理在障碍物环境下最优路径规划问题。提出Lazy BPRC方法，通过延迟成本计算提升求解效率。**

- **链接: [https://arxiv.org/pdf/2603.21880](https://arxiv.org/pdf/2603.21880)**

> **作者:** Anoop Bhat; Geordan Gutow; Surya Singh; Zhongqiang Ren; Sivakumar Rathinam; Howie Choset
>
> **摘要:** The Moving Target Vehicle Routing Problem with Obstacles (MT-VRP-O) seeks trajectories for several agents that collectively intercept a set of moving targets. Each target has one or more time windows where it must be visited, and the agents must avoid static obstacles and satisfy speed and capacity constraints. We introduce Lazy Branch-and-Price with Relaxed Continuity (Lazy BPRC), which finds optimal solutions for the MT-VRP-O. Lazy BPRC applies the branch-and-price framework for VRPs, which alternates between a restricted master problem (RMP) and a pricing problem. The RMP aims to select a sequence of target-time window pairings (called a tour) for each agent to follow, from a limited subset of tours. The pricing problem adds tours to the limited subset. Conventionally, solving the RMP requires computing the cost for an agent to follow each tour in the limited subset. Computing these costs in the MT-VRP-O is computationally intensive, since it requires collision-free motion planning between moving targets. Lazy BPRC defers cost computations by solving the RMP using lower bounds on the costs of each tour, computed via motion planning with relaxed continuity constraints. We lazily evaluate the true costs of tours as-needed. We compute a tour's cost by searching for a shortest path on a Graph of Convex Sets (GCS), and we accelerate this search using our continuity relaxation method. We demonstrate that Lazy BPRC runs up to an order of magnitude faster than two ablations.
>
---
#### [replaced 009] Learning Underwater Active Perception in Simulation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于 underwater active perception 任务，旨在解决水下视觉质量受浊度和散射影响的问题。通过构建合成数据集和MLP模型，提升不同水况下的图像质量与可视覆盖。**

- **链接: [https://arxiv.org/pdf/2504.17817](https://arxiv.org/pdf/2504.17817)**

> **作者:** Alexandre Cardaillac; Donald G. Dansereau
>
> **摘要:** When employing underwater vehicles for the autonomous inspection of assets, it is crucial to consider and assess the water conditions. These conditions significantly impact visibility and directly affect robotic operations. Turbidity can jeopardise the mission by preventing accurate visual documentation of inspected structures. Previous works have introduced methods to adapt to turbidity and backscattering, however, they also include manoeuvring and setup constraints. We propose a simple yet efficient approach to enable high-quality image acquisition of assets in a broad range of water conditions. This active perception framework includes a multi-layer perceptron (MLP) trained to predict image quality given a distance to a target and artificial light intensity. We generate a large synthetic dataset that includes ten water types with varying levels of turbidity and backscattering. For this, we modified the modelling software Blender to better account for the underwater light propagation properties. We validated the approach in simulation and demonstrate significant improvements in visual coverage and image quality compared to traditional methods. The project code is available on our project page at this https URL.
>
---
#### [replaced 010] Goal-VLA: Image-Generative VLMs as Object-Centric World Models Empowering Zero-shot Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决零样本泛化能力不足的问题。通过利用图像生成的VLM作为世界模型，生成目标状态以实现通用操作。**

- **链接: [https://arxiv.org/pdf/2506.23919](https://arxiv.org/pdf/2506.23919)**

> **作者:** Haonan Chen; Jingxiang Guo; Bangjun Wang; Tianrui Zhang; Xuchuan Huang; Boren Zheng; Yiwen Hou; Chenrui Tie; Jiajun Deng; Lin Shao
>
> **摘要:** Generalization remains a fundamental challenge in robotic manipulation. To tackle this challenge, recent Vision-Language-Action (VLA) models build policies on top of Vision-Language Models (VLMs), seeking to transfer their open-world semantic knowledge. However, their zero-shot capability lags significantly behind the base VLMs, as the instruction-vision-action data is too limited to cover diverse scenarios, tasks, and robot embodiments. In this work, we present Goal-VLA, a zero-shot framework that leverages Image-Generative VLMs as world models to generate desired goal states, from which the target object pose is derived to enable generalizable manipulation. The key insight is that object state representation is the golden interface, naturally separating a manipulation system into high-level and low-level policies. This representation abstracts away explicit action annotations, allowing the use of highly generalizable VLMs while simultaneously providing spatial cues for training-free low-level control. To further improve robustness, we introduce a Reflection-through-Synthesis process that iteratively validates and refines the generated goal image before execution. Both simulated and real-world experiments demonstrate that our \name achieves strong performance and inspiring generalizability in manipulation tasks. Supplementary materials are available at this https URL.
>
---
#### [replaced 011] Context-Triggered Contingency Games for Strategic Multi-Agent Interaction
- **分类: cs.RO**

- **简介: 该论文属于多智能体协作任务，旨在解决自主系统中长期战略与短期适应的平衡问题。提出上下文触发的应急博弈框架，实现安全高效的多智能体交互。**

- **链接: [https://arxiv.org/pdf/2512.03639](https://arxiv.org/pdf/2512.03639)**

> **作者:** Kilian Schweppe; Anne-Kathrin Schmuck
>
> **摘要:** We address the challenge of reliable and efficient interaction in autonomous multi-agent systems, where agents must balance long-term strategic objectives with short-term dynamic adaptation. We propose context-triggered contingency games, a novel integration of strategic games derived from temporal logic specifications with dynamic contingency games solved in real time. Our two-layered architecture leverages strategy templates to guarantee satisfaction of high-level objectives, while a new factor-graph-based solver enables scalable, real-time model predictive control of dynamic interactions. The resulting framework ensures both safety and progress in uncertain, interactive environments. We validate our approach through simulations and hardware experiments in autonomous driving and robotic navigation, demonstrating efficient, reliable, and adaptive multi-agent interaction.
>
---
#### [replaced 012] Ruka-v2: Tendon Driven Open-Source Dexterous Hand with Wrist and Abduction for Robot Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文介绍Ruka-v2，一款开源、肌腱驱动的人类手型机械臂，解决机器人操作灵活性不足的问题。新增手腕和手指外展功能，提升抓取与操作能力，并验证其在机器人学习中的应用效果。**

- **链接: [https://arxiv.org/pdf/2603.26660](https://arxiv.org/pdf/2603.26660)**

> **作者:** Xinqi Lucas Liu; Ruoxi Hu; Alejandro Ojeda Olarte; Zhuoran Chen; Kenny Ma; Charles Cheng Ji; Lerrel Pinto; Raunaq Bhirangi; Irmak Guzey
>
> **摘要:** Lack of accessible and dexterous robot hardware has been a significant bottleneck to achieving human-level dexterity in robots. Last year, we released Ruka, a fully open-sourced, tendon-driven humanoid hand with 11 degrees of freedom - 2 per finger and 3 at the thumb - buildable for under $1,300. It was one of the first fully open-sourced humanoid hands, and introduced a novel data-driven approach to finger control that captures tendon dynamics within the control system. Despite these contributions, Ruka lacked two degrees of freedom essential for closely imitating human behavior: wrist mobility and finger adduction/abduction. In this paper, we introduce Ruka-v2: a fully open-sourced, tendon-driven humanoid hand featuring a decoupled 2-DOF parallel wrist and abduction/adduction at the fingers. The parallel wrist adds smooth, independent flexion/extension and radial/ulnar deviation, enabling manipulation in confined environments such as cabinets. Abduction enables motions such as grasping thin objects, in-hand rotation, and calligraphy. We present the design of Ruka-v2 and evaluate it against Ruka through user studies on teleoperated tasks, finding a 51.3% reduction in completion time and a 21.2% increase in success rate. We further demonstrate its full range of applications for robot learning: bimanual and single-arm teleoperation across 13 dexterous tasks, and autonomous policy learning on 3 tasks. All 3D print files, assembly instructions, controller software, and videos are available at this https URL .
>
---
#### [replaced 013] Dream to Recall: Imagination-Guided Experience Retrieval for Memory-Persistent Vision-and-Language Navigation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决记忆持久型VLN中记忆访问机制不足的问题。提出Memoir模型，利用想象引导检索环境与行为记忆，提升导航效果。**

- **链接: [https://arxiv.org/pdf/2510.08553](https://arxiv.org/pdf/2510.08553)**

> **作者:** Yunzhe Xu; Yiyuan Pan; Zhe Liu
>
> **备注:** Accepted by IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
>
> **摘要:** Vision-and-Language Navigation (VLN) requires agents to follow natural language instructions through environments, with memory-persistent variants demanding progressive improvement through accumulated experience. Existing approaches for memory-persistent VLN face critical limitations: they lack effective memory access mechanisms, instead relying on entire memory incorporation or fixed-horizon lookup, and predominantly store only environmental observations while neglecting navigation behavioral patterns that encode valuable decision-making strategies. We present Memoir, which employs imagination as a retrieval mechanism grounded by explicit memory: a world model imagines future navigation states as queries to selectively retrieve relevant environmental observations and behavioral histories. The approach comprises: 1) a language-conditioned world model that imagines future states serving dual purposes: encoding experiences for storage and generating retrieval queries; 2) Hybrid Viewpoint-Level Memory that anchors both observations and behavioral patterns to viewpoints, enabling hybrid retrieval; and 3) an experience-augmented navigation model that integrates retrieved knowledge through specialized encoders. Extensive evaluation across diverse memory-persistent VLN benchmarks with 10 distinct testing scenarios demonstrates Memoir's effectiveness: significant improvements across all scenarios, with 5.4% SPL gains on IR2R over the best memory-persistent baseline, accompanied by 8.3x training speedup and 74% inference memory reduction. The results validate that predictive retrieval of both environmental and behavioral memories enables more effective navigation, with analysis indicating substantial headroom (73.3% vs 93.4% upper bound) for this imagination-guided paradigm.
>
---
#### [replaced 014] FlexiCup: Wireless Multimodal Suction Cup with Dual-Zone Vision-Tactile Sensing
- **分类: cs.RO**

- **简介: 该论文提出FlexiCup，一种具备双区视觉触觉感知的无线多模态吸盘，解决非结构化环境中接触感知不足的问题。通过模块化设计实现真空与伯努利两种抓取模式，验证了传感与执行的解耦。**

- **链接: [https://arxiv.org/pdf/2511.14139](https://arxiv.org/pdf/2511.14139)**

> **作者:** Junhao Gong; Shoujie Li; Kit-Wa Sou; Changqing Guo; Hourong Huang; Tong Wu; Yifan Xie; Chenxin Liang; Chuqiao Lyu; Xiaojun Liang; Wenbo Ding
>
> **备注:** Accepted by IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Conventional suction cups lack sensing capabilities for contact-aware manipulation in unstructured environments. This paper presents FlexiCup, a multimodal suction cup with wireless electronics that integrate dual-zone vision-tactile sensing. The central zone dynamically switches between vision and tactile modalities via illumination control, while the peripheral zone provides continuous spatial awareness. The modular mechanical design supports both vacuum (sustained-contact adhesion) and Bernoulli (contactless lifting) actuation while maintaining the identical dual-zone sensing architecture, demonstrating sensing-actuation decoupling where sensing and actuation principles are orthogonally separable. We validate hardware versatility through dual control paradigms. Modular perception-driven grasping achieves comparable success rates across vacuum (90.0%) and Bernoulli (86.7%) modes using identical sensing and control pipelines, validating the sensing architecture's effectiveness across fundamentally different pneumatic principles. Diffusion-based end-to-end learning achieves 73.3% and 66.7% success on contact-aware manipulation tasks, with ablation studies confirming 13% improvements from multi-head attention coordinating dual-zone observations. Hardware designs, firmware, and experimental videos are available at the companion website: this https URL.
>
---
#### [replaced 015] Grip as Needed, Glide on Demand: Ultrasonic Lubrication for Robotic Locomotion
- **分类: cs.RO; physics.app-ph**

- **简介: 该论文属于机器人运动控制任务，旨在解决传统摩擦固定导致的效率低问题。通过超声润滑主动调控摩擦，实现高效双向运动。**

- **链接: [https://arxiv.org/pdf/2602.15608](https://arxiv.org/pdf/2602.15608)**

> **作者:** Mostafa A. Atalla; Daan van Bemmel; Jack Cummings; Paul Breedveld; Michaël Wiertlewski; Aimée Sakes
>
> **备注:** Accepted for publication in the 2026 IEEE International Conference on Robotics and Automation (ICRA) in Vienna
>
> **摘要:** Friction is the essential mediator of terrestrial locomotion, yet in robotic systems it is almost always treated as a passive property fixed by surface materials and conditions. Here, we introduce ultrasonic lubrication as a method to actively control friction in robotic locomotion. By exciting resonant structures at ultrasonic frequencies, contact interfaces can dynamically switch between "grip" and "slip" states, enabling locomotion. We developed two friction control modules, a cylindrical design for lumen-like environments and a flat-plate design for external surfaces, and integrated them into bio-inspired systems modeled after inchworm and wasp ovipositor locomotion. Both systems achieved bidirectional locomotion with nearly perfect locomotion efficiencies that exceeded 90%. Friction characterization experiments further demonstrated substantial friction reduction across various surfaces, including rigid, soft, granular, and biological tissue interfaces, under dry and wet conditions, and on surfaces with different levels of roughness, confirming the broad applicability of ultrasonic lubrication to locomotion tasks. These findings establish ultrasonic lubrication as a viable active friction control mechanism for robotic locomotion, with the potential to reduce design complexity and improve efficiency of robotic locomotion systems.
>
---
#### [replaced 016] IndoorR2X: Indoor Robot-to-Everything Coordination with LLM-Driven Planning
- **分类: cs.RO; cs.MA**

- **简介: 该论文提出IndoorR2X，解决多机器人与室内物联网设备的协同问题，通过LLM驱动的规划提升场景理解与任务效率。**

- **链接: [https://arxiv.org/pdf/2603.20182](https://arxiv.org/pdf/2603.20182)**

> **作者:** Fan Yang; Soumya Teotia; Shaunak A. Mehta; Prajit KrisshnaKumar; Quanting Xie; Jun Liu; Yueqi Song; Wenkai Li; Atsunori Moteki; Kanji Uchino; Yonatan Bisk
>
> **摘要:** Although robot-to-robot (R2R) communication improves indoor scene understanding beyond what a single robot can achieve, R2R alone cannot overcome partial observability without substantial exploration overhead or scaling team size. In contrast, many indoor environments already include low-cost Internet of Things (IoT) sensors (e.g., cameras) that provide persistent, building-wide context beyond onboard perception. We therefore introduce IndoorR2X, the first benchmark and simulation framework for Large Language Model (LLM)-driven multi-robot task planning with Robot-to-Everything (R2X) perception and communication in indoor environments. IndoorR2X integrates observations from mobile robots and static IoT devices to construct a global semantic state that supports scalable scene understanding, reduces redundant exploration, and enables high-level coordination through LLM-based planning. IndoorR2X provides configurable simulation environments, sensor layouts, robot teams, and task suites to systematically evaluate high-level semantic coordination strategies. Extensive experiments across diverse settings demonstrate that IoT-augmented world modeling improves multi-robot efficiency and reliability, and we highlight key insights and failure modes for advancing LLM-based collaboration between robot teams and indoor IoT sensors. See our project website: this https URL.
>
---
#### [replaced 017] DIV-Nav: Open-Vocabulary Spatial Relationships for Multi-Object Navigation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出DIV-Nav系统，解决多物体导航中复杂空间关系查询的问题，通过分解指令、计算语义地图交集和验证实现高效导航。**

- **链接: [https://arxiv.org/pdf/2510.16518](https://arxiv.org/pdf/2510.16518)**

> **作者:** Jesús Ortega-Peimbert; Finn Lukas Busch; Timon Homberger; Quantao Yang; Olov Andersson
>
> **摘要:** Advances in open-vocabulary semantic mapping and object navigation have enabled robots to perform an informed search of their environment for an arbitrary object. However, such zero-shot object navigation is typically designed for simple queries with an object name like "television" or "blue rug". Here, we consider more complex free-text queries with spatial relationships, such as "find the remote on the table" while still leveraging robustness of a semantic map. We present DIV-Nav, a real-time navigation system that efficiently addresses this problem through a series of relaxations: i) Decomposing natural language instructions with complex spatial constraints into simpler object-level queries on a semantic map, ii) computing the Intersection of individual semantic belief maps to identify regions where all objects co-exist, and iii) Validating the discovered objects against the original, complex spatial constrains via a LVLM. We further investigate how to adapt the frontier exploration objectives of online semantic mapping to such spatial search queries to more effectively guide the search process. We validate our system through extensive experiments on the MultiON benchmark and real-world deployment on a Boston Dynamics Spot robot using a Jetson Orin AGX. More details and videos are available at this https URL
>
---
#### [replaced 018] ROBOGATE: Adaptive Failure Discovery for Safe Robot Policy Deployment via Two-Stage Boundary-Focused Sampling
- **分类: cs.RO**

- **简介: 该论文提出ROBOGATE框架，用于安全部署机器人策略时的故障发现。任务是评估机器人在工业环境中的部署风险，解决高维参数空间中故障边界难以全面测试的问题。通过两阶段采样策略，结合物理仿真与风险建模，高效识别危险区域。**

- **链接: [https://arxiv.org/pdf/2603.22126](https://arxiv.org/pdf/2603.22126)**

> **作者:** Azuki Kim
>
> **备注:** 12 pages, 5 figures, open-source code and 30K failure pattern dataset available at this https URL
>
> **摘要:** Deploying learned robot manipulation policies in industrial settings requires rigorous pre-deployment validation, yet exhaustive testing across high-dimensional parameter spaces is intractable. We present ROBOGATE, a deployment risk management framework that combines physics-based simulation with a two-stage adaptive sampling strategy to efficiently discover failure boundaries in the operational parameter space. Stage 1 employs Latin Hypercube Sampling (LHS) across an 8-dimensional parameter space to establish a coarse failure landscape from 20,000 uniformly distributed experiments. Stage 2 applies boundary-focused sampling that concentrates 10,000 additional experiments in the 30-70% success rate transition zone, enabling precise failure boundary mapping. Using NVIDIA Isaac Sim with Newton physics, we evaluate a scripted pick-and-place controller on two robot embodiments -- Franka Panda (7-DOF) and UR5e (6-DOF) -- across 30,000 total experiments. Our logistic regression risk model achieves an AUC of 0.780 on the combined dataset (vs. 0.754 for Stage 1 alone), identifies a closed-form failure boundary equation, and reveals four universal danger zones affecting both robot platforms. We further demonstrate the framework on VLA (Vision-Language-Action) model evaluation, where Octo-Small achieves 0.0% success rate on 68 adversarial scenarios versus 100% for the scripted baseline -- a 100-point gap that underscores the challenge of deploying foundation models in industrial settings. ROBOGATE is open-source and runs on a single GPU workstation.
>
---
#### [replaced 019] ViPRA: Video Prediction for Robot Actions
- **分类: cs.RO; cs.AI; cs.CL; cs.CV; cs.LG**

- **简介: 该论文提出ViPRA，解决机器人控制中缺乏标注动作的问题。通过视频预测和隐式动作表示，实现无需大量标注的连续控制，提升泛化能力和控制频率。**

- **链接: [https://arxiv.org/pdf/2511.07732](https://arxiv.org/pdf/2511.07732)**

> **作者:** Sandeep Routray; Hengkai Pan; Unnat Jain; Shikhar Bahl; Deepak Pathak
>
> **备注:** In ICLR 2026. Website: this https URL
>
> **摘要:** Can we turn a video prediction model into a robot policy? Videos, including those of humans or teleoperated robots, capture rich physical interactions. However, most of them lack labeled actions, which limits their use in robot learning. We present Video Prediction for Robot Actions (ViPRA), a simple pretraining-finetuning framework that learns continuous robot control from these actionless videos. Instead of directly predicting actions, we train a video-language model to predict both future visual observations and motion-centric latent actions, which serve as intermediate representations of scene dynamics. We train these latent actions using perceptual losses and optical flow consistency to ensure they reflect physically grounded behavior. For downstream control, we introduce a chunked flow matching decoder that maps latent actions to robot-specific continuous action sequences, using only 100 to 200 teleoperated demonstrations. This approach avoids expensive action annotation, supports generalization across embodiments, and enables smooth, high-frequency continuous control upto 22 Hz via chunked action decoding. Unlike prior latent action works that treat pretraining as autoregressive policy learning, ViPRA explicitly models both what changes and how. Our method outperforms strong baselines, with a 16% gain on the SIMPLER benchmark and a 13% improvement across real world manipulation tasks. We have released models and code at this https URL
>
---
#### [replaced 020] Integrating Maneuverable Planning and Adaptive Control for Robot Cart-Pushing under Disturbances
- **分类: cs.RO**

- **简介: 该论文研究移动机器人在干扰下的灵活推车任务，解决运动规划与控制难题。提出新的规划与控制框架，提升推车灵活性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2506.18410](https://arxiv.org/pdf/2506.18410)**

> **作者:** Zhe Zhang; Peijia Xie; Yuhan Pang; Zhirui Sun; Bingyi Xia; Bi-Ke Zhu; Jiankun Wang
>
> **备注:** 11 pages, 11 figures
>
> **摘要:** Precise and flexible cart-pushing is a challenging task for mobile robots. The motion constraints during cart-pushing and the robot's redundancy lead to complex motion planning problems, while variable payloads and disturbances present complicated dynamics. In this work, we propose a novel planning and control framework for flexible whole-body coordination and robust adaptive control. Our motion planning method employs a local coordinate representation and a novel kinematic model to solve a nonlinear optimization problem, thereby enhancing motion maneuverability by generating feasible and flexible push poses. Furthermore, we present a disturbance rejection control method to resist disturbances and reduce control errors for the complex control problem without requiring an accurate dynamic model. We validate our method through extensive experiments in simulation and real-world settings, demonstrating its superiority over existing approaches. To the best of our knowledge, this is the first work to systematically evaluate the flexibility and robustness of cart-pushing methods in experiments. The video supplement is available at this https URL.
>
---
#### [replaced 021] VLM-SAFE: Vision-Language Model-Guided Safety-Aware Reinforcement Learning with World Models for Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶任务，解决RL在安全性和样本效率上的问题。提出VLM-SAFE框架，结合视觉语言模型与世界模型，提升安全决策与策略学习效果。**

- **链接: [https://arxiv.org/pdf/2505.16377](https://arxiv.org/pdf/2505.16377)**

> **作者:** Yansong Qu; Zilin Huang; Zihao Sheng; Jiancong Chen; Yue Leng; Samuel Labi; Sikai Chen
>
> **备注:** N/A
>
> **摘要:** Autonomous driving policy learning with reinforcement learning (RL) is fundamentally limited by low sample efficiency, weak generalization, and a dependence on unsafe online trial-and-error interactions. Although safe RL introduces explicit constraints or costs, existing methods often fail to capture the semantic meaning of safety in real driving scenes, leading to conservative behaviors in simple cases and insufficient risk awareness in complex ones. To address this issue, we propose VLM-SAFE, an offline safe RL framework that follows a human cognitive loop of observe-imagine-evaluate-act. Starting from offline driving data, VLM-SAFE observes traffic scenarios and leverages a vision-language model (VLM) to provide semantic safety signals grounded in scene understanding. A learned world model then imagines future trajectories from the observed context, enabling the agent to reason about possible consequences without interacting with the real environment. Rather than using imagined rollouts solely for return estimation, VLM-SAFE further evaluates these predicted futures with VLM-based safety guidance, explicitly coupling future anticipation with semantic risk assessment. The resulting safety-aware imagined experience is finally used to optimize the policy via actor-critic learning, such that actions are chosen based on both predicted outcomes and their safety implications. By tightly integrating observation, imagination, evaluation, and action into a unified closed loop, VLM-SAFE enables safer and more efficient offline policy learning for autonomous driving. Extensive experiments in simulation show that VLM-SAFE achieves improved safety, stronger robustness under traffic-density shift, and a better safety-performance trade-off than representative baselines.
>
---
#### [replaced 022] OVSegDT: Segmenting Transformer for Open-Vocabulary Object Goal Navigation
- **分类: cs.RO**

- **简介: 该论文属于开放词汇目标导航任务，解决模型泛化能力差和碰撞问题。提出OVSegDT模型，结合语义分支和自适应损失机制，提升导航性能并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2508.11479](https://arxiv.org/pdf/2508.11479)**

> **作者:** Tatiana Zemskova; Aleksei Staroverov; Dmitry Yudin; Aleksandr Panov
>
> **摘要:** Open-vocabulary Object Goal Navigation requires an embodied agent to reach objects described by free-form language, including categories never seen during training. Existing end-to-end policies overfit small simulator datasets, achieving high success on training scenes but failing to generalize and exhibiting unsafe behaviour (frequent collisions). We introduce OVSegDT, a lightweight transformer policy that tackles these issues with two synergistic components. The first component is the semantic branch, which includes an encoder for the target binary mask and an auxiliary segmentation loss function, grounding the textual goal and providing precise spatial cues. The second component consists of a proposed Entropy-Adaptive Loss Modulation, a per-sample scheduler that continuously balances imitation and reinforcement signals according to the policy entropy, eliminating brittle manual phase switches. These additions cut the sample complexity of training by 33%, and reduce collision count in two times while keeping inference cost low (130M parameters, RGB-only input). On HM3D-OVON, our model matches the performance on unseen categories to that on seen ones and establishes state-of-the-art results (40.1% SR, 20.9% SPL on val unseen) without depth, odometry, or large vision-language models. Code is available at this https URL.
>
---
#### [replaced 023] ThermoAct:Thermal-Aware Vision-Language-Action Models for Robotic Perception and Decision-Making
- **分类: cs.RO**

- **简介: 该论文提出ThermoAct框架，将热力信息融入视觉-语言-动作模型，解决机器人感知与决策中的安全与效率问题，提升任务成功率和环境安全性。**

- **链接: [https://arxiv.org/pdf/2603.25044](https://arxiv.org/pdf/2603.25044)**

> **作者:** Young-Chae Son; Dae-Kwan Ko; Yoon-Ji Choi; Soo-Chul Lim
>
> **备注:** 2026 RA-L
>
> **摘要:** In recent human-robot collaboration environments, there is a growing focus on integrating diverse sensor data beyond visual information to enable safer and more intelligent task execution. Although thermal data can be crucial for enhancing robot safety and operational efficiency, its integration has been relatively overlooked in prior research. This paper proposes a novel Vision-Language-Action (VLA) framework that incorporates thermal information for robot task execution. The proposed system leverages a Vision-Language Model (VLM) as a high-level planner to interpret complex natural language commands and decompose them into simpler sub-tasks. This approach facilitates efficient data collection and robust reasoning for complex operations. Unlike conventional methods that rely solely on visual data, our approach integrates thermal information, enabling the robot to perceive physical properties and proactively ensure environmental safety. Experimental results from real-world task scenarios validate the feasibility of our proposed framework, suggesting its potential to enhance task success rates and safety compared to existing vision-based systems.
>
---
#### [replaced 024] Continual Robot Skill and Task Learning via Dialogue
- **分类: cs.RO; cs.AI; cs.CL**

- **简介: 该论文属于人机交互中的持续学习任务，旨在解决机器人高效学习新技能与任务的问题。通过对话交互获取人类指导，提出ACT-LoRA模型实现少量样本下的持续学习。**

- **链接: [https://arxiv.org/pdf/2409.03166](https://arxiv.org/pdf/2409.03166)**

> **作者:** Weiwei Gu; Suresh Kondepudi; Anmol Gupta; Lixiao Huang; Nakul Gopalan
>
> **摘要:** Interactive robot learning is a challenging problem as the robot is present with human users who expect the robot to learn novel skills to solve novel tasks perpetually with sample efficiency. In this work we present a framework for robots to continually learn tasks and visuo-motor skills and query for novel skills via dialog interactions with human users. Our robot agent maintains a skill library, and uses an existing LLM to perform grounded dialog interactions to query unknown skills from real human users. We developed a novel visual-motor control policy Action Chunking Transformer with Low Rank Adaptation (ACT-LoRA) that can continually learn novel skills using only a few demonstrations which is critical in human-robot interaction scenarios. The paper has twin goals: Firstly to demonstrate better continual learning in simulation; and secondly, to demonstrate the use of our dialog based learning framework in a realistic human-robot interaction use case. Our ACT-LoRA policy consistently outperforms a GMM-LoRA baseline on multiple continual learning simulation benchmarks by achieving > 300% improvements on novel skills, while achieving comparable performance in existing skills. Moreover, with our IRB approved human-subjects study we demonstrate that our dialog based continual learning framework allows users to teach robots cooking skills successfully (100%) while spending a higher ratio of time on finishing an auxiliary distraction tasks in the test phase of the study compared to a non-learning language based agent (p < 0.001).
>
---
#### [replaced 025] Omni-LIVO: Robust RGB-Colored Multi-Camera Visual-Inertial-LiDAR Odometry via Photometric Migration and ESIKF Fusion
- **分类: cs.RO**

- **简介: 该论文提出Omni-LIVO，解决多相机LiDAR惯性视觉里程计问题，通过跨视图对齐和ESIKF融合提升定位精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2509.15673](https://arxiv.org/pdf/2509.15673)**

> **作者:** Yinong Cao; Chenyang Zhang; Xin He; Yuwei Chen; Chengyu Pu; Bingtao Wang; Kaile Wu; Shouzheng Zhu; Fei Han; Shijie Liu; Chunlai Li; Jianyu Wang
>
> **备注:** Accepted by IEEE Robotics and Automation Letters (RA-L). Early Access version available. This version supersedes all previous versions and is the official accepted manuscript for citation
>
> **摘要:** Wide field-of-view (FoV) LiDAR sensors provide dense geometry across large environments, but existing LiDAR-inertial-visual odometry (LIVO) systems generally rely on a single camera, limiting their ability to fully exploit LiDAR-derived depth for photometric alignment and scene colorization. We present Omni-LIVO, a tightly coupled multi-camera LIVO system that leverages multi-view observations to comprehensively utilize LiDAR geometric information across extended spatial regions. Omni-LIVO introduces a Cross-View direct alignment strategy that maintains photometric consistency across non-overlapping views, and extends the Error-State Iterated Kalman Filter (ESIKF) with multi-view updates and adaptive covariance. The system is evaluated on public benchmarks and our custom dataset, showing improved accuracy and robustness over state-of-the-art LIVO, LIO, and visual-inertial SLAM baselines. Code and dataset will be released upon publication.
>
---
#### [replaced 026] The Multi-AMR Buffer Storage, Retrieval, and Reshuffling Problem: Exact and Heuristic Approaches
- **分类: cs.RO; cs.AI; cs.MA; math.OC**

- **简介: 该论文研究多AMR协同的缓冲区存储、取货与调位问题，旨在解决高密度生产环境中的自动化调度难题。提出精确模型与分层启发式算法以提高计算效率。**

- **链接: [https://arxiv.org/pdf/2603.26542](https://arxiv.org/pdf/2603.26542)**

> **作者:** Max Disselnmeyer; Thomas Bömer; Laura Dörr; Bastian Amberg; Anne Meyer
>
> **备注:** 52 pages, 15 figures and tables
>
> **摘要:** Buffer zones are essential in production systems to decouple sequential processes. In dense floor storage environments, such as space-constrained brownfield facilities, manual operation is increasingly challenged by severe labor shortages and rising operational costs. Automating these zones requires solving the Buffer Storage, Retrieval, and Reshuffling Problem (BSRRP). While previous work has addressed scenarios where the focus is limited to reshuffling and retrieving a fixed set of items, real-world manufacturing necessitates an adaptive approach that also incorporates arriving unit loads. This paper introduces the Multi-AMR BSRRP, coordinating a robot fleet to manage concurrent reshuffling, alongside time-windowed storage and retrieval tasks, within a shared floor area. We formulate a Binary Integer Programming (IP) model to obtain exact solutions for benchmarking purposes. As the problem is NP-hard, rendering exact methods computationally intractable for industrial scales, we propose a hierarchical heuristic. This approach decomposes the problem into an A* search for task-level sequence planning of unit load placements, and a Constraint Programming (CP) approach for multi-robot coordination and scheduling. Experiments demonstrate orders-of-magnitude computation time reductions compared to the exact formulation. These results confirm the heuristic's viability as responsive control logic for high-density production environments.
>
---
#### [replaced 027] From Observation to Action: Latent Action-based Primitive Segmentation for VLA Pre-training in Industrial Settings
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉-语言-动作（VLA）预训练任务，旨在从工业视频中自动提取结构化数据。通过动作分割和语义聚类，解决无监督数据利用问题，提升制造中的具身AI性能。**

- **链接: [https://arxiv.org/pdf/2511.21428](https://arxiv.org/pdf/2511.21428)**

> **作者:** Jiajie Zhang; Sören Schwertfeger; Alexander Kleiner
>
> **备注:** 10 pages, 5 figures, Accepted to CVPR 2026
>
> **摘要:** We present a novel unsupervised framework to unlock vast unlabeled human demonstration data from continuous industrial video streams for Vision-Language-Action (VLA) model pre-training. Our method first trains a lightweight motion tokenizer to encode motion dynamics, then employs an unsupervised action segmenter leveraging a novel "Latent Action Energy" metric to discover and segment semantically coherent action primitives. The pipeline outputs both segmented video clips and their corresponding latent action sequences, providing structured data directly suitable for VLA pre-training. Evaluations on public benchmarks and a proprietary electric motor assembly dataset demonstrate effective segmentation of key tasks performed by humans at workstations. Further clustering and quantitative assessment via a Vision-Language Model confirm the semantic coherence of the discovered action primitives. To our knowledge, this is the first fully automated end-to-end system for extracting and organizing VLA pre-training data from unstructured industrial videos, offering a scalable solution for embodied AI integration in manufacturing.
>
---
#### [replaced 028] ExtremControl: Low-Latency Humanoid Teleoperation with Direct Extremity Control
- **分类: cs.RO**

- **简介: 该论文属于人形机器人遥操作任务，旨在解决高延迟导致响应慢的问题。通过直接控制肢体姿态和速度前馈，实现低延迟（50ms）的实时操作。**

- **链接: [https://arxiv.org/pdf/2602.11321](https://arxiv.org/pdf/2602.11321)**

> **作者:** Ziyan Xiong; Lixing Fang; Junyun Huang; Kashu Yamazaki; Hao Zhang; Chuang Gan
>
> **备注:** Project website: this https URL
>
> **摘要:** Building a low-latency humanoid teleoperation system is essential for collecting diverse reactive and dynamic demonstrations. However, existing approaches rely on heavily pre-processed human-to-humanoid motion retargeting and position-only PD control, resulting in substantial latency that severely limits responsiveness and prevents tasks requiring rapid feedback and fast reactions. To address this problem, we propose ExtremControl, a low latency whole-body control framework that: (1) operates directly on SE(3) poses of selected rigid links, primarily humanoid extremities, to avoid full-body retargeting; (2) utilizes a Cartesian-space mapping to directly convert human motion to humanoid link targets; and (3) incorporates velocity feedforward control at low level to support highly responsive behavior under rapidly changing control interfaces. We further provide a unified theoretical formulation of ExtremControl and systematically validate its effectiveness through experiments in both simulation and real-world environments. Building on ExtremControl, we implement a low-latency humanoid teleoperation system that supports both optical motion capture and VR-based motion tracking, achieving end-to-end latency as low as 50ms and enabling highly responsive behaviors such as ping-pong ball balancing, juggling, and real-time return, thereby substantially surpassing the 200ms latency limit observed in prior work.
>
---
#### [replaced 029] DADP: Domain Adaptive Diffusion Policy
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决策略在不同动态环境中的泛化问题。通过无监督解耦和领域感知扩散注入，提出DADP方法提升零样本适应能力。**

- **链接: [https://arxiv.org/pdf/2602.04037](https://arxiv.org/pdf/2602.04037)**

> **作者:** Pengcheng Wang; Qinghang Liu; Haotian Lin; Yiheng Li; Guojian Zhan; Masayoshi Tomizuka; Yixiao Wang
>
> **摘要:** Learning domain adaptive policies that can generalize to unseen transition dynamics, remains a fundamental challenge in learning-based control. Substantial progress has been made through domain representation learning to capture domain-specific information, thus enabling domain-aware decision making. We analyze the process of learning domain representations through dynamical prediction and find that selecting contexts adjacent to the current step causes the learned representations to entangle static domain information with varying dynamical properties. Such mixture can confuse the conditioned policy, thereby constraining zero-shot adaptation. To tackle the challenge, we propose DADP (Domain Adaptive Diffusion Policy), which achieves robust adaptation through unsupervised disentanglement and domain-aware diffusion injection. First, we introduce Lagged Context Dynamical Prediction, a strategy that conditions future state estimation on a historical offset context; by increasing this temporal gap, we unsupervisedly disentangle static domain representations by filtering out transient properties. Second, we integrate the learned domain representations directly into the generative process by biasing the prior distribution and reformulating the diffusion target. Extensive experiments on challenging benchmarks across locomotion and manipulation demonstrate the superior performance, and the generalizability of DADP over prior methods. More visualization results are available on the this https URL.
>
---
#### [replaced 030] Scaling Sim-to-Real Reinforcement Learning for Robot VLAs with Generative 3D Worlds
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究机器人视觉语言动作（VLA）模型的强化学习微调问题，旨在解决真实世界训练泛化性差与仿真场景设计成本高的矛盾。通过生成式3D世界提升训练效率和泛化能力。**

- **链接: [https://arxiv.org/pdf/2603.18532](https://arxiv.org/pdf/2603.18532)**

> **作者:** Andrew Choi; Xinjie Wang; Zhizhong Su; Wei Xu
>
> **摘要:** The strong performance of large vision-language models (VLMs) trained with reinforcement learning (RL) has motivated similar approaches for fine-tuning vision-language-action (VLA) models in robotics. Many recent works fine-tune VLAs directly in the real world to avoid addressing the sim-to-real gap. While real-world RL circumvents sim-to-real issues, it inherently limits the generality of the resulting VLA, as scaling scene and object diversity in the physical world is prohibitively difficult. This leads to the paradoxical outcome of transforming a broadly pretrained model into an overfitted, scene-specific policy. Training in simulation can instead provide access to diverse scenes, but designing those scenes is also costly. In this work, we show that VLAs can be RL fine-tuned without sacrificing generality and with reduced labor by leveraging 3D world generative models. Using these models together with a language-driven scene designer, we generate hundreds of diverse interactive scenes containing unique objects and backgrounds, enabling scalable and highly parallel policy learning. Starting from a pretrained imitation baseline, our approach increases simulation success from 9.7% to 79.8% while achieving a 1.25$\times$ speedup in task completion time. We further demonstrate successful sim-to-real transfer enabled by the quality of the generated digital twins together with domain randomization, improving real-world success from 21.7% to 75% and achieving a 1.13$\times$ speedup. Finally, we further highlight the benefits of leveraging the effectively unlimited data from 3D world generative models through an ablation study showing that increasing scene diversity directly improves zero-shot generalization.
>
---
#### [replaced 031] CycleManip: Enabling Cyclic Task Manipulation via Effective Historical Perception and Understanding
- **分类: cs.RO**

- **简介: 该论文属于机器人周期任务操作领域，解决周期性动作执行效率低和缺乏基准的问题。提出CycleManip框架，通过历史感知与多任务学习提升性能，并构建了相关基准与评估方法。**

- **链接: [https://arxiv.org/pdf/2512.01022](https://arxiv.org/pdf/2512.01022)**

> **作者:** Yi-Lin Wei; Haoran Liao; Yuhao Lin; Pengyue Wang; Zhizhao Liang; Guiliang Liu; Wei-Shi Zheng
>
> **备注:** Accepted by CVPR2026. Project page: this https URL
>
> **摘要:** In this paper, we explore an important yet underexplored task in robot manipulation: cycle-based manipulation, where robots need to perform cyclic or repetitive actions with an expected terminal time. These tasks are crucial in daily life, such as shaking a bottle or knocking a nail. However, few prior works have explored this task, leading to two main challenges: 1) the imitation methods often fail to complete these tasks within the expected terminal time due to the ineffective utilization of history; 2) the absence of a benchmark with sufficient data and automatic evaluation tools hinders development of effective solutions in this area. To address these challenges, we first propose the CycleManip framework to achieve cycle-based task manipulation in an end-to-end imitation manner without requiring any extra models, hierarchical structure or significant computational overhead. The core insight is to enhance effective history perception by a cost-aware sampling strategy and to improve historical understanding by multi-task learning. Second, we introduce a cycle-based task manipulation benchmark, which provides diverse cycle-based tasks, and an automatic evaluation method. Extensive experiments conducted in both simulation and real-world settings demonstrate that our method achieves high success rates in cycle-based task manipulation. The results further show strong adaptability performance in general manipulation, and the plug-and-play ability on imitation policies such as Vision-Language-Action (VLA) models. Moreover, the results show that our approach can be applied across diverse robotic platforms, including bi-arm grippers, dexterous hands, and humanoid robots.
>
---
#### [replaced 032] A Class of Axis-Angle Attitude Control Laws for Rotational Systems
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于姿态控制任务，解决旋转系统的稳定控制问题。提出一种基于轴角表示的控制方法，提高控制灵活性和效率。**

- **链接: [https://arxiv.org/pdf/2512.19846](https://arxiv.org/pdf/2512.19846)**

> **作者:** Francisco M. F. R. Gonçalves; Ryan M. Bena; Néstor O. Pérez-Arancibia
>
> **备注:** 6 pages, 4 figures. Published in IEEE Control Systems Letters
>
> **摘要:** We introduce a new class of attitude control laws for rotational systems; the proposed framework generalizes the use of the Euler \mbox{axis--angle} representation beyond quaternion-based formulations. Using basic Lyapunov stability theory and the notion of extended class $\mathcal{K}$ function, we developed a method for determining and enforcing the global asymptotic stability of the single fixed point of the resulting \mbox{\textit{closed-loop}} (CL) scheme. In contrast with traditional \mbox{quaternion-based} methods, the introduced generalized \mbox{axis--angle} approach enables greater flexibility in the design of the control law, which is of great utility when employed in combination with a switching scheme whose transition state depends on the angular velocity of the controlled rotational system. Through simulation and \mbox{real-time} experimental results, we demonstrate the effectiveness of the developed formulation. According to the recorded data, in the execution of \mbox{high-speed} \mbox{tumble-recovery} maneuvers, the new method consistently achieves shorter stabilization times and requires lower control effort relative to those corresponding to the \mbox{quaternion-based} and \mbox{geometric-control} methods used as benchmarks.
>
---
#### [replaced 033] Mobile Robot Exploration Without Maps via Out-of-Distribution Deep Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决无地图、动态环境下的移动机器人探索问题。通过OOD DRL方法实现高效避障与探索，无需高阶规划或实时制图。**

- **链接: [https://arxiv.org/pdf/2402.05066](https://arxiv.org/pdf/2402.05066)**

> **作者:** Shathushan Sivashangaran; Apoorva Khairnar; Azim Eskandarian
>
> **备注:** \c{opyright} 2025 the authors. This work has been accepted to IFAC for publication under a Creative Commons License CC-BY-NC-ND
>
> **摘要:** Autonomous Mobile Robot (AMR) navigation in dynamic environments that may be GPS denied, without a-priori maps, is an unsolved problem with potential to improve humanity's capabilities. Conventional modular methods are computationally inefficient, and require explicit feature extraction and engineering that inhibit generalization and deployment at scale. We present an Out-of-Distribution (OOD) Deep Reinforcement Learning (DRL) approach that includes functionality in unstructured terrain and dynamic obstacle avoidance capabilities. We leverage accelerated simulation training in a racetrack with a transition probability to parameterize spatial reasoning with intrinsic exploratory behavior, in a compact, computationally efficient Artificial Neural Network (ANN), which we transfer zero-shot with a reward component to mitigate differences between simulation and real world physics. Our approach enables utility without a separate high-level planner or real-time cartography and utilizes a fraction of the computation resources of modular methods, enabling execution in a range of AMRs with different embedded computer payloads.
>
---
#### [replaced 034] Integrated Shape-Force Estimation for Continuum Robots: A Virtual-Work and Polynomial-Curvature Framework
- **分类: cs.RO**

- **简介: 该论文属于机器人形状与力估计任务，解决CDCR在大变形和稀疏传感下的精度问题。提出结合缆绳张力与末端位姿的框架，利用多项式曲率和虚功原理实现形状重建与力估计。**

- **链接: [https://arxiv.org/pdf/2501.05418](https://arxiv.org/pdf/2501.05418)**

> **作者:** Guoqing Zhang; Zihan Chen; Long Wang
>
> **摘要:** Cable-driven continuum robots (CDCRs) are widely used in surgical and inspection tasks that require dexterous manipulation in confined spaces. Existing model-based estimation methods either assume constant curvature or rely on geometry-space interpolants, both of which struggle with accuracy under large deformations and sparse sensing. This letter introduces an integrated shape-force estimation framework that combines cable-tension measurements with tip-pose data to reconstruct backbone shape and estimate external tip force simultaneously. The framework employs polynomial curvature kinematics (PCK) and a virtual-work-based static formulation expressed directly in curvature space, where polynomial modal coefficients serve as generalized coordinates. The proposed method is validated through Cosserat-rod-based simulations and hardware experiments on a torque-cell-enabled CDCR prototype. Results show that the second-order PCK model achieves superior shape and force accuracy, combining a lightweight shape optimization with a closed-form, iteration-free force estimation, offering a compact and robust alternative to prior constant-curvature and geometry-space approaches.
>
---
#### [replaced 035] Service Discovery-Based Hybrid Network Middleware for Efficient Communication in Distributed Robotic Systems
- **分类: cs.RO; cs.DC; cs.NI**

- **简介: 该论文属于分布式机器人系统通信优化任务，解决大尺度自动驾驶中通信效率与确定性问题。提出RIMAOS2C中间件，通过服务发现和共享内存提升数据传输效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2508.00947](https://arxiv.org/pdf/2508.00947)**

> **作者:** Shiyao Sang; Yinggang Ling
>
> **备注:** 8 pages, 8 figures, accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Robotic middleware is fundamental to ensuring reliable communication among system components and is crucial for intelligent robotics, autonomous vehicles, and smart manufacturing. However, existing robotic middleware often struggles to meet the diverse communication demands, optimize data transmission efficiency, and maintain scheduling determinism between Orin computing units in large-scale L4 autonomous vehicle deployments. This paper presents RIMAOS2C, a service discovery-based hybrid network communication middleware designed to tackle these challenges. By leveraging multi-level service discovery multicast, RIMAOS2C supports a wide variety of communication modes, including multiple cross-chip Ethernet protocols and PCIe communication capabilities. Its core mechanism, the Message Bridge, optimizes data flow forwarding and employs shared memory for centralized message distribution, reducing message redundancy and minimizing transmission delay uncertainty. Tested on L4 vehicles and Jetson Orin domain controllers, RIMAOS2C leverages TCP-based ZeroMQ to overcome the large-message transmission bottleneck in native CyberRT. In scenarios with two cross-chip subscribers, it eliminates message redundancy and improves large-data transmission efficiency by 36 to 40 percent while reducing callback latency variation by 42 to 906 percent. This research advances the communication capabilities of robotic operating systems and proposes a novel approach to optimizing communication in distributed computing architectures for autonomous driving.
>
---
#### [replaced 036] Securing the Skies: A Comprehensive Survey on Anti-UAV Methods, Benchmarking, and Future Directions
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 本文属于反无人机任务，旨在解决UAV的安全威胁问题。综述了分类、检测与跟踪方法，分析了现有技术及挑战，提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2504.11967](https://arxiv.org/pdf/2504.11967)**

> **作者:** Yifei Dong; Fengyi Wu; Sanjian Zhang; Guangyu Chen; Yuzhi Hu; Masumi Yano; Jingdong Sun; Siyu Huang; Feng Liu; Qi Dai; Zhi-Qi Cheng
>
> **备注:** Accepted to CVPR 2025 Anti-UAV Workshop (Best Paper Award), 16 pages
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are indispensable for infrastructure inspection, surveillance, and related tasks, yet they also introduce critical security challenges. This survey provides a wide-ranging examination of the anti-UAV domain, centering on three core objectives-classification, detection, and tracking-while detailing emerging methodologies such as diffusion-based data synthesis, multi-modal fusion, vision-language modeling, self-supervised learning, and reinforcement learning. We systematically evaluate state-of-the-art solutions across both single-modality and multi-sensor pipelines (spanning RGB, infrared, audio, radar, and RF) and discuss large-scale as well as adversarially oriented benchmarks. Our analysis reveals persistent gaps in real-time performance, stealth detection, and swarm-based scenarios, underscoring pressing needs for robust, adaptive anti-UAV systems. By highlighting open research directions, we aim to foster innovation and guide the development of next-generation defense strategies in an era marked by the extensive use of UAVs.
>
---
#### [replaced 037] SimULi: Real-Time LiDAR and Camera Simulation with Unscented Transforms
- **分类: cs.CV; cs.GR; cs.LG; cs.RO**

- **简介: 该论文提出SimULi，解决多传感器实时模拟问题，支持任意相机模型和LiDAR数据，提升模拟速度与精度。**

- **链接: [https://arxiv.org/pdf/2510.12901](https://arxiv.org/pdf/2510.12901)**

> **作者:** Haithem Turki; Qi Wu; Xin Kang; Janick Martinez Esturo; Shengyu Huang; Ruilong Li; Zan Gojcic; Riccardo de Lutio
>
> **备注:** ICLR 2026 - project page: this https URL
>
> **摘要:** Rigorous testing of autonomous robots, such as self-driving vehicles, is essential to ensure their safety in real-world deployments. This requires building high-fidelity simulators to test scenarios beyond those that can be safely or exhaustively collected in the real-world. Existing neural rendering methods based on NeRF and 3DGS hold promise but suffer from low rendering speeds or can only render pinhole camera models, hindering their suitability to applications that commonly require high-distortion lenses and LiDAR data. Multi-sensor simulation poses additional challenges as existing methods handle cross-sensor inconsistencies by favoring the quality of one modality at the expense of others. To overcome these limitations, we propose SimULi, the first method capable of rendering arbitrary camera models and LiDAR data in real-time. Our method extends 3DGUT, which natively supports complex camera models, with LiDAR support, via an automated tiling strategy for arbitrary spinning LiDAR models and ray-based culling. To address cross-sensor inconsistencies, we design a factorized 3D Gaussian representation and anchoring strategy that reduces mean camera and depth error by up to 40% compared to existing methods. SimULi renders 10-20x faster than ray tracing approaches and 1.5-10x faster than prior rasterization-based work (and handles a wider range of camera models). When evaluated on two widely benchmarked autonomous driving datasets, SimULi matches or exceeds the fidelity of existing state-of-the-art methods across numerous camera and LiDAR metrics.
>
---
#### [replaced 038] R3DP: Real-Time 3D-Aware Policy for Embodied Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出R3DP，解决实时具身操作中的3D感知问题。通过融合大模型3D先验，提升操作成功率并保持实时性。**

- **链接: [https://arxiv.org/pdf/2603.14498](https://arxiv.org/pdf/2603.14498)**

> **作者:** Yuhao Zhang; Wanxi Dong; Yue Shi; Yi Liang; Jingnan Gao; Qiaochu Yang; Yaxing Lyu; Zhixuan Liang; Yibin Liu; Congsheng Xu; Xianda Guo; Wei Sui; Yaohui Jin; Xiaokang Yang; Yanyan Xu; Yao Mu
>
> **备注:** Project Page: this https URL Github Repo: this https URL
>
> **摘要:** Embodied manipulation requires accurate 3D understanding of objects and their spatial relations to plan and execute contact-rich actions. While large-scale 3D vision models provide strong priors, their computational cost incurs prohibitive latency for real-time control. We propose Real-time 3D-aware Policy (R3DP), which integrates powerful 3D priors into manipulation policies without sacrificing real-time performance. A core innovation of R3DP is the asynchronous fast-slow collaboration module, which seamlessly integrates large-scale 3D priors into the policy without compromising real-time performance. The system maintains real-time efficiency by querying the pre-trained slow system (VGGT) only on sparse key frames, while simultaneously employing a lightweight Temporal Feature Prediction Network (TFPNet) to predict features for all intermediate frames. By leveraging historical data to exploit temporal correlations, TFPNet explicitly improves task success rates through consistent feature estimation. Additionally, to enable more effective multi-view fusion, we introduce a Multi-View Feature Fuser (MVFF) that aggregates features across views by explicitly incorporating camera intrinsics and extrinsics. R3DP offers a plug-and-play solution for integrating large models into real-time inference systems. We evaluate R3DP against multiple baselines across different visual configurations. R3DP effectively harnesses large-scale 3D priors to achieve superior results, outperforming single-view and multi-view DP by 32.9% and 51.4% in average success rate, respectively. Furthermore, by decoupling heavy 3D reasoning from policy execution, R3DP achieves a 44.8% reduction in inference time compared to a naive DP+VGGT integration.
>
---
#### [replaced 039] Resolving Spatio-Temporal Entanglement in Video Prediction via Multi-Modal Attention
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于视频预测任务，旨在解决传统模型在长期时间一致性和高分辨率细节上的不足。提出MAUCell架构，结合多模态注意力机制，提升视频生成的准确性和实时性。**

- **链接: [https://arxiv.org/pdf/2501.16997](https://arxiv.org/pdf/2501.16997)**

> **作者:** Shreyam Gupta; P. Agrawal; Priyam Gupta
>
> **备注:** 11 pages, 3 figures, 5 tables, and 3 Algorithms
>
> **摘要:** The fast progress in computer vision has necessitated more advanced methods for temporal sequence modeling. This area is essential for the operation of autonomous systems, real-time surveillance, and predicting anomalies. As the demand for accurate video prediction increases, the limitations of traditional deterministic models, particularly their struggle to maintain long-term temporal coherence while providing high-frequency spatial detail, have become very clear. This report provides an exhaustive analysis of the Multi-Attention Unit Cell (MAUCell), a novel architectural framework that represents a significant leap forward in video frame prediction. By synergizing Generative Adversarial Networks (GANs) with a hierarchical "STAR-GAN" processing strategy and a triad of specialized attention mechanisms (Temporal, Spatial, and Pixel-wise), the MAUCell addresses the persistent "deep-in-time" dilemma that plagues Recurrent Neural Networks (RNNs). Our analysis shows that the MAUCell framework successfully establishes a new state-of-the-art benchmark, especially in its ability to produce realistic video sequences that closely resemble real-world footage while ensuring efficient inference for real-time deployment. Through rigorous evaluation on datasets: Moving MNIST, KTH Action, and CASIA-B, the framework shows superior performance metrics, especially in Learned Perceptual Image Patch Similarity (LPIPS) and Structural Similarity Index (SSIM). This success confirms its dual-pathway information transformation system. This report details the theoretical foundations, detailed structure and broader significance of MAUCell, presenting it as a valuable solution for video forecasting tasks that require high precision and limited resources.
>
---
#### [replaced 040] RoboManipBaselines: A Unified Framework for Imitation Learning in Robotic Manipulation across Real and Simulation Environments
- **分类: cs.RO**

- **简介: 该论文提出RoboManipBaselines，一个用于机器人操作模仿学习的统一框架，解决仿真与真实环境中的学习问题，实现数据收集、训练与评估的全流程支持。**

- **链接: [https://arxiv.org/pdf/2509.17057](https://arxiv.org/pdf/2509.17057)**

> **作者:** Masaki Murooka; Tomohiro Motoda; Ryoichi Nakajo; Hanbit Oh; Koshi Makihara; Keisuke Shirai; Tetsuya Ogata; Yukiyasu Domae
>
> **备注:** Minor title revision. Added one author. Expanded the description and added application examples
>
> **摘要:** We present RoboManipBaselines, an open-source software framework for imitation learning research in robotic manipulation. The framework supports the entire imitation learning pipeline, including data collection, policy training, and rollout, across both simulation and real-world environments. Its design emphasizes integration through a consistent workflow, generality across diverse environments and robot platforms, extensibility for easily adding new robots, tasks, and policies, and reproducibility through evaluations using publicly available datasets. RoboManipBaselines systematically implements the core components of imitation learning: environment, dataset, and policy. Through a unified interface, the framework supports multiple simulators and real robot environments, as well as multimodal sensors and a wide variety of policy models. We further present benchmark evaluations in both simulation and real-world environments and introduce several research applications, including data augmentation, integration with tactile models, interactive robotic systems, 3D sensing evaluation, and hardware extensions. These results demonstrate that RoboManipBaselines provides a useful foundation for advancing research and experimental validation in robotic manipulation using imitation learning. this https URL
>
---
#### [replaced 041] AIM-SLAM: Dense Monocular SLAM via Adaptive and Informative Multi-View Keyframe Prioritization with Foundation Model
- **分类: cs.RO**

- **简介: 该论文属于视觉SLAM任务，解决单目密集重建中视图选择不足的问题。提出AIM-SLAM框架，通过自适应多视角关键帧优先级策略提升定位精度和重建效果。**

- **链接: [https://arxiv.org/pdf/2603.05097](https://arxiv.org/pdf/2603.05097)**

> **作者:** Jinwoo Jeon; Dong-Uk Seo; Eungchang Mason Lee; Hyun Myung
>
> **备注:** 8 pages
>
> **摘要:** Recent advances in geometric foundation models have emerged as a promising alternative for addressing the challenge of dense reconstruction in monocular visual simultaneous localization and mapping (SLAM). Although geometric foundation models enable SLAM to leverage variable input views, the previous methods remain confined to two-view pairs or fixed-length inputs without sufficient deliberation of geometric context for view selection. To tackle this problem, we propose AIM-SLAM, a dense monocular SLAM framework that exploits an adaptive and informative multi-view keyframe prioritization with dense pointmap predictions from visual geometry grounded transformer (VGGT). Specifically, we introduce the selective information- and geometric-aware multi-view adaptation (SIGMA) module, which employs voxel overlap and information gain to retrieve a candidate set of keyframes and adaptively determine its size. Furthermore, we formulate a joint multi-view Sim(3) optimization that enforces consistent alignment across selected views, substantially improving pose estimation accuracy. The effectiveness of AIM-SLAM is demonstrated on real-world datasets, where it achieves state-of-the-art pose estimation performance and accurate dense reconstruction results. Our system supports ROS integration, with code is available at this https URL.
>
---
#### [replaced 042] PhysMem: Scaling Test-time Physical Memory for Robot Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出PhysMem，用于机器人操作中的物理记忆学习，解决VLM无法准确预测具体物理行为的问题。通过测试验证假设，提升操作成功率。**

- **链接: [https://arxiv.org/pdf/2602.20323](https://arxiv.org/pdf/2602.20323)**

> **作者:** Haoyang Li; Yang You; Hao Su; Leonidas Guibas
>
> **摘要:** Reliable object manipulation requires understanding physical properties that vary across objects and environments. Vision-language model (VLM) planners can reason about friction and stability in general terms; however, they often cannot predict how a specific ball will roll on a particular surface or which stone will provide a stable foundation without direct experience. We present PhysMem, a memory framework that enables VLM robot planners to learn physical principles from interaction at test time, without updating model parameters. The system records experiences, generates candidate hypotheses, and verifies them through targeted interaction before promoting validated knowledge to guide future decisions. A central design choice is verification before application: the system tests hypotheses against new observations rather than applying retrieved experience directly, reducing rigid reliance on prior experience when physical conditions change. We evaluate PhysMem on three real-world manipulation tasks and simulation benchmarks across four VLM backbones. On a controlled brick insertion task, principled abstraction achieves 76% success compared to 23% for direct experience retrieval, and real-world experiments show consistent improvement over 30-minute deployment sessions.
>
---
#### [replaced 043] LaST$_{0}$: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision-Language-Action Model
- **分类: cs.RO**

- **简介: 该论文提出LaST$_0$，解决机器人视觉-语言-动作模型中的推理效率与物理动态建模问题，通过潜在时空思维链提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2601.05248](https://arxiv.org/pdf/2601.05248)**

> **作者:** Zhuoyang Liu; Jiaming Liu; Hao Chen; Jiale Yu; Ziyu Guo; Chengkai Hou; Chenyang Gu; Xiangju Mi; Renrui Zhang; Kun Wu; Zhengping Che; Jian Tang; Pheng-Ann Heng; Shanghang Zhang
>
> **备注:** Project page: this https URL
>
> **摘要:** Vision-Language-Action (VLA) models have recently shown strong generalization, with some approaches seeking to explicitly generate linguistic reasoning traces or predict future observations prior to execution. However, explicit reasoning typically incurs non-negligible inference latency, which constrains the temporal resolution required for robotic manipulation. Moreover, such reasoning is confined to the linguistic space, imposing a representational bottleneck that struggles to faithfully capture ineffable physical attributes. To mitigate these limitations, we propose LaST$_0$, a framework that enables efficient reasoning before acting through a Latent Spatio-Temporal Chain-of-Thought (CoT), capturing fine-grained physical and robotic dynamics that are often difficult to verbalize. Specifically, we introduce a token-efficient latent CoT space that models future visual dynamics, 3D structural information, and robot proprioceptive states, and further extends these representations across time to enable temporally consistent implicit reasoning trajectories. Furthermore, LaST$_0$ adopts a dual-system architecture implemented via a Mixture-of-Transformers design, where a reasoning expert conducts low-frequency latent inference and an acting expert generates high-frequency actions conditioned on robotics-oriented latent representations. To facilitate coordination, LaST$_0$ is trained with heterogeneous operation frequencies, enabling adaptive switching during deployment. Across 10 real-world tasks spanning tabletop, mobile, and dexterous hand manipulation, LaST$_0$ improves mean success rates by 13%, 14% and 14% over prior SOTA VLA methods, respectively.
>
---
#### [replaced 044] Vega: Learning to Drive with Natural Language Instructions
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文提出Vega模型，解决自动驾驶中根据自然语言指令进行个性化驾驶的问题。通过构建大规模数据集并融合视觉、语言和动作模态，提升指令跟随与路径规划能力。**

- **链接: [https://arxiv.org/pdf/2603.25741](https://arxiv.org/pdf/2603.25741)**

> **作者:** Sicheng Zuo; Yuxuan Li; Wenzhao Zheng; Zheng Zhu; Jie Zhou; Jiwen Lu
>
> **备注:** Code is available at this https URL
>
> **摘要:** Vision-language-action models have reshaped autonomous driving to incorporate languages into the decision-making process. However, most existing pipelines only utilize the language modality for scene descriptions or reasoning and lack the flexibility to follow diverse user instructions for personalized driving. To address this, we first construct a large-scale driving dataset (InstructScene) containing around 100,000 scenes annotated with diverse driving instructions with the corresponding trajectories. We then propose a unified Vision-Language-World-Action model, Vega, for instruction-based generation and planning. We employ the autoregressive paradigm to process visual inputs (vision) and language instructions (language) and the diffusion paradigm to generate future predictions (world modeling) and trajectories (action). We perform joint attention to enable interactions between the modalities and use individual projection layers for different modalities for more capabilities. Extensive experiments demonstrate that our method not only achieves superior planning performance but also exhibits strong instruction-following abilities, paving the way for more intelligent and personalized driving systems.
>
---
#### [replaced 045] DecompGrind: A Decomposition Framework for Robotic Grinding via Cutting-Surface Planning and Contact-Force Adaptation
- **分类: cs.RO**

- **简介: 该论文属于机器人打磨任务，旨在解决不同形状和硬度工件的高效自动化打磨问题。通过分解为形状规划和力适应两部分，提出 DecompGrind 框架，实现安全高效的打磨。**

- **链接: [https://arxiv.org/pdf/2603.22859](https://arxiv.org/pdf/2603.22859)**

> **作者:** Shunsuke Araki; Takumi Hachimine; Yuki Saito; Kouhei Ohnishi; Jun Morimoto; Takamitsu Matsubara
>
> **备注:** Under review
>
> **摘要:** Robotic grinding is widely used for shaping workpieces in manufacturing, but it remains difficult to automate this process efficiently. In particular, efficiently grinding workpieces of different shapes and material hardness is challenging because removal resistance varies with local contact conditions. Moreover, it is difficult to achieve accurate estimation of removal resistance and analytical modeling of shape transition, and learning-based approaches often require large amounts of training data to cover diverse processing conditions. To address these challenges, we decompose robotic grinding into two components: removal-shape planning and contact-force adaptation. Based on this formulation, we propose DecompGrind, a framework that combines Global Cutting-Surface Planning (GCSP) and Local Contact-Force Adaptation (LCFA). GCSP determines removal shapes through geometric analysis of the current and target shapes without learning, while LCFA learns a contact-force adaptation policy using bilateral control-based imitation learning during the grinding of each removal shape. This decomposition restricts learning to local contact-force adaptation, allowing the policy to be learned from a small number of demonstrations, while handling global shape transition geometrically. Experiments using a robotic grinding system and 3D-printed workpieces demonstrate efficient robotic grinding of workpieces having different shapes and material hardness while maintaining safe levels of contact force.
>
---
#### [replaced 046] Hybrid Action Based Reinforcement Learning for Multi-Objective Compatible Autonomous Driving
- **分类: cs.RO; cs.AI; cs.ET; cs.LG**

- **简介: 该论文属于自主驾驶任务，解决多目标兼容问题。提出一种基于混合动作的强化学习方法，提升决策灵活性与安全性。**

- **链接: [https://arxiv.org/pdf/2501.08096](https://arxiv.org/pdf/2501.08096)**

> **作者:** Guizhe Jin; Zhuoren Li; Bo Leng; Wei Han; Lu Xiong; Chen Sun
>
> **备注:** 14 pages, accepted for publication in IEEE Transactions on Neural Networks and Learning Systems (T-NNLS)
>
> **摘要:** Reinforcement Learning (RL) has shown excellent performance in solving decision-making and control problems of autonomous driving, which is increasingly applied in diverse driving scenarios. However, driving is a multi-attribute problem, leading to challenges in achieving multi-objective compatibility for current RL methods, especially in both policy updating and policy execution. On the one hand, a single value evaluation network limits the policy updating in complex scenarios with coupled driving objectives. On the other hand, the common single-type action space structure limits driving flexibility or results in large behavior fluctuations during policy execution. To this end, we propose a Multi-objective Ensemble-Critic reinforcement learning method with Hybrid Parametrized Action for multi-objective compatible autonomous driving. Specifically, an advanced MORL architecture is constructed, in which the ensemble-critic focuses on different objectives through independent reward functions. The architecture integrates a hybrid parameterized action space structure, and the generated driving actions contain both abstract guidance that matches the hybrid road modality and concrete control commands. Additionally, an uncertainty-based exploration mechanism that supports hybrid actions is developed to learn multi-objective compatible policies more quickly. Experimental results demonstrate that, in both simulator-based and HighD dataset-based multi-lane highway scenarios, our method efficiently learns multi-objective compatible autonomous driving with respect to efficiency, action consistency, and safety.
>
---
#### [replaced 047] MALLVI: A Multi-Agent Framework for Integrated Generalized Robotics Manipulation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出MALLVI框架，解决机器人操作中的任务规划问题。通过多智能体协作实现闭环反馈，提升动态环境下的操作成功率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.16898](https://arxiv.org/pdf/2602.16898)**

> **作者:** Mehrshad Taji; Arad Mahdinezhad Kashani; Iman Ahmadi; AmirHossein Jadidi; Saina Kashani; Babak Khalaj
>
> **摘要:** Task planning for robotic manipulation with large language models (LLMs) is an emerging area. Prior approaches rely on specialized models, fine tuning, or prompt tuning, and often operate in an open loop manner without robust environmental feedback, making them fragile in dynamic settings. MALLVI presents a Multi Agent Large Language and Vision framework that enables closed-loop feedback driven robotic manipulation. Given a natural language instruction and an image of the environment, MALLVI generates executable atomic actions for a robot manipulator. After action execution, a Vision Language Model (VLM) evaluates environmental feedback and decides whether to repeat the process or proceed to the next step. Rather than using a single model, MALLVI coordinates specialized agents, Decomposer, Localizer, Thinker, and Reflector, to manage perception, localization, reasoning, and high level planning. An optional Descriptor agent provides visual memory of the initial state. The Reflector supports targeted error detection and recovery by reactivating only relevant agents, avoiding full replanning. Experiments in simulation and real-world settings show that iterative closed loop multi agent coordination improves generalization and increases success rates in zero shot manipulation tasks. Code available at this https URL .
>
---
#### [replaced 048] EgoDemoGen: Egocentric Demonstration Generation for Viewpoint Generalization in Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决视觉-动作策略在第一视角变化下的泛化问题。通过生成新的视角下观测与动作演示，提升策略在不同视角下的成功率。**

- **链接: [https://arxiv.org/pdf/2509.22578](https://arxiv.org/pdf/2509.22578)**

> **作者:** Yuan Xu; Jiabing Yang; Xiaofeng Wang; Yixiang Chen; Zheng Zhu; Bowen Fang; Guan Huang; Xinze Chen; Yun Ye; Qiang Zhang; Peiyan Li; Xiangnan Wu; Kai Wang; Bing Zhan; Shuo Lu; Jing Liu; Nianfeng Liu; Yan Huang; Liang Wang
>
> **摘要:** Imitation learning based visuomotor policies have achieved strong performance in robotic manipulation, yet they often remain sensitive to egocentric viewpoint shifts. Unlike third-person viewpoint changes that only move the camera, egocentric shifts simultaneously alter both the camera pose and the robot action coordinate frame, making it necessary to jointly transfer action trajectories and synthesize corresponding observations under novel egocentric viewpoints. To address this challenge, we present EgoDemoGen, a framework that generates paired observation--action demonstrations under novel egocentric viewpoints through two key components: 1{)} EgoTrajTransfer, which transfers robot trajectories to the novel egocentric coordinate frame through motion-skill segmentation, geometry-aware transformation, and inverse kinematics filtering; and 2{)} EgoViewTransfer, a conditional video generation model that fuses a novel-viewpoint reprojected scene video and a robot motion video rendered from the transferred trajectory to synthesize photorealistic observations, trained with a self-supervised double reprojection strategy without requiring multi-viewpoint data. Experiments in simulation and real-world settings show that EgoDemoGen consistently improves policy success rates under both standard and novel egocentric viewpoints, with absolute gains of +24.6\% and +16.9\% in simulation and +16.0\% and +23.0\% on the real robot. Moreover, EgoViewTransfer achieves superior video generation quality for novel egocentric observations.
>
---
#### [replaced 049] ACoT-VLA: Action Chain-of-Thought for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决动作执行中信息传达不足的问题。提出ACoT-VLA模型，通过显式和隐式动作推理直接指导动作生成。**

- **链接: [https://arxiv.org/pdf/2601.11404](https://arxiv.org/pdf/2601.11404)**

> **作者:** Linqing Zhong; Yi Liu; Yifei Wei; Ziyu Xiong; Maoqing Yao; Si Liu; Guanghui Ren
>
> **备注:** Accepted by Conference on Computer Vision and Pattern Recognition (CVPR) 2026
>
> **摘要:** Vision-Language-Action models have emerged as essential generalist robot policies for diverse manipulation tasks, conventionally relying on directly translating multimodal inputs into actions via Vision-Language Model embeddings. Recent advancements have introduced explicit intermediary reasoning-such as sub-task prediction (language) or goal image synthesis (vision)-to guide action generation. However, these intermediate reasoning are often indirect and inherently limited in their capacity to convey the full, granular information required for precise action execution. Instead, we posit that the most effective form of reasoning is one that deliberates directly in the action space. We introduce Action Chain-of-Thought (ACoT), a paradigm where the reasoning process itself is formulated as a structured sequence of coarse action intents that guide the final policy. In this paper, we propose ACoT-VLA, a novel architecture that materializes the ACoT paradigm. Specifically, we introduce two complementary components: an Explicit Action Reasoner (EAR) and Implicit Action Reasoner (IAR). The former proposes coarse reference trajectories as explicit action-level reasoning steps, while the latter extracts latent action priors from internal representations of multimodal input, co-forming an ACoT that conditions the downstream action head to enable grounded policy learning. Extensive experiments in real-world and simulation environments demonstrate the superiority of our proposed method. Code is available at: this https URL.
>
---
#### [replaced 050] Assessing Vision-Language Models for Perception in Autonomous Underwater Robotic Software
- **分类: cs.SE; cs.RO**

- **简介: 该论文属于自主水下机器人感知任务，旨在解决水下环境对深度学习模型的挑战。研究评估了视觉-语言模型在水下垃圾检测中的性能与不确定性。**

- **链接: [https://arxiv.org/pdf/2602.10655](https://arxiv.org/pdf/2602.10655)**

> **作者:** Muhammad Yousaf; Aitor Arrieta; Shaukat Ali; Paolo Arcaini; Shuai Wang
>
> **备注:** 16 pages, 5 figures
>
> **摘要:** Autonomous Underwater Robots (AURs) operate in challenging underwater environments, including low visibility and harsh water conditions. Such conditions present challenges for software engineers developing perception modules for the AUR software. To successfully carry out these tasks, deep learning has been incorporated into the AUR software to support its operations. However, the unique challenges of underwater environments pose difficulties for deep learning models, which often rely on labeled data that is scarce and noisy. This may undermine the trustworthiness of AUR software that relies on perception modules. Vision-Language Models (VLMs) offer promising solutions for AUR software as they generalize to unseen objects and remain robust in noisy conditions by inferring information from contextual cues. Despite this potential, their performance and uncertainty in underwater environments remain understudied from a software engineering perspective. Motivated by the needs of an industrial partner in assurance and risk management for maritime systems to assess the potential use of VLMs in this context, we present an empirical evaluation of VLM-based perception modules within the AUR software. We assess their ability to detect underwater trash by computing performance, uncertainty, and their relationship, to enable software engineers to select appropriate VLMs for their AUR software.
>
---
#### [replaced 051] Captivity-Escape Games as a Means for Safety in Online Motion Generation
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于在线运动生成安全任务，解决传统方法保守性高、计算量大和精度不足的问题。通过引入捕获-逃脱博弈提升安全性与效率。**

- **链接: [https://arxiv.org/pdf/2506.01399](https://arxiv.org/pdf/2506.01399)**

> **作者:** Christopher Bohn; Manuel Hess; Sören Hohmann
>
> **摘要:** This paper presents a method that addresses the conservatism, computational effort, and limited numerical accuracy of existing frameworks and methods that ensure safety in online model-based motion generation, commonly referred to as fast and safe tracking. Computational limitations restrict online motion planning to low-fidelity models. However, planning with low-fidelity models compromises safety, as the dynamic feasibility of resulting references is not ensured. This potentially leads to unavoidable tracking errors that may cause safety-critical constraint violations. Existing frameworks mitigate this safety risk by augmenting safety-critical constraints in motion planning by a safety margin that prevents constraint violations under worst-case tracking errors. However, the methods employed in these frameworks determine the safety margin based on a heuristically selected performance of the model used for planning, which likely results in overly conservative references. Furthermore, these methods are computationally intensive, and the state-of-the-art method is limited in numerical accuracy. We adopt a different perspective and address these limitations with a method that mitigates conservatism in existing frameworks by adapting the performance of the model used for planning to a given safety margin. Our method achieves numerical accuracy and requires significantly less computation time than existing methods by leveraging a captivity-escape game, which is a novel zero-sum differential game formulated in this paper. We demonstrate our method using a numerical example and compare it to the state of the art.
>
---
#### [replaced 052] Deconfounded Lifelong Learning for Autonomous Driving via Dynamic Knowledge Spaces
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于自主驾驶任务，解决E2E-AD的持续学习问题，提出DeLL框架，结合DPMM和因果调整机制，提升知识迁移与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2603.14354](https://arxiv.org/pdf/2603.14354)**

> **作者:** Jiayuan Du; Yuebing Song; Yiming Zhao; Xianghui Pan; Jiawei Lian; Yuchu Lu; Liuyi Wang; Chengju Liu; Qijun Chen
>
> **摘要:** End-to-End autonomous driving (E2E-AD) systems face challenges in lifelong learning, including catastrophic forgetting, difficulty in knowledge transfer across diverse scenarios, and spurious correlations between unobservable confounders and true driving intents. To address these issues, we propose DeLL, a Deconfounded Lifelong Learning framework that integrates a Dirichlet process mixture model (DPMM) with the front-door adjustment mechanism from causal inference. The DPMM is employed to construct two dynamic knowledge spaces: a trajectory knowledge space for clustering explicit driving behaviors and an implicit feature knowledge space for discovering latent driving abilities. Leveraging the non-parametric Bayesian nature of DPMM, our framework enables adaptive expansion and incremental updating of knowledge without predefining the number of clusters, thereby mitigating catastrophic forgetting. Meanwhile, the front-door adjustment mechanism utilizes the DPMM-derived knowledge as valid mediators to deconfound spurious correlations, such as those induced by sensor noise or environmental changes, and enhances the causal expressiveness of the learned representations. Additionally, we introduce an evolutionary trajectory decoder that enables non-autoregressive planning. To evaluate the lifelong learning performance of E2E-AD, we propose new evaluation protocols and metrics based on Bench2Drive. Extensive evaluations in the closed-loop CARLA simulator demonstrate that our framework significantly improves adaptability to new driving scenarios and overall driving performance, while effectively retaining previous acquired knowledge.
>
---
