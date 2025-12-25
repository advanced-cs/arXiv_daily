# 机器人 cs.RO

- **最新发布 27 篇**

- **更新 10 篇**

## 最新发布

#### [new 001] Certifiable Alignment of GNSS and Local Frames via Lagrangian Duality
- **分类: cs.RO**

- **简介: 提出一种基于拉格朗日对偶的GNSS与局部坐标系全局最优对齐方法，可在卫星稀少时仍保证解的可验证最优性，优于传统易陷局部最优的方法。**

- **链接: [https://arxiv.org/pdf/2512.20931v1](https://arxiv.org/pdf/2512.20931v1)**

> **作者:** Baoshan Song; Matthew Giamou; Penggao Yan; Chunxi Xia; Li-Ta Hsu
>
> **摘要:** Estimating the absolute orientation of a local system relative to a global navigation satellite system (GNSS) reference often suffers from local minima and high dependency on satellite availability. Existing methods for this alignment task rely on abundant satellites unavailable in GNSS-degraded environments, or use local optimization methods which cannot guarantee the optimality of a solution. This work introduces a globally optimal solver that transforms raw pseudo-range or Doppler measurements into a convexly relaxed problem. The proposed method is certifiable, meaning it can numerically verify the correctness of the result, filling a gap where existing local optimizers fail. We first formulate the original frame alignment problem as a nonconvex quadratically constrained quadratic program (QCQP) problem and relax the QCQP problem to a concave Lagrangian dual problem that provides a lower cost bound for the original problem. Then we perform relaxation tightness and observability analysis to derive criteria for certifiable optimality of the solution. Finally, simulation and real world experiments are conducted to evaluate the proposed method. The experiments show that our method provides certifiably optimal solutions even with only 2 satellites with Doppler measurements and 2D vehicle motion, while the traditional velocity-based VOBA method and the advanced GVINS alignment technique may fail or converge to local optima without notice. To support the development of GNSS-based navigation techniques in robotics, all code and data are open-sourced at https://github.com/Baoshan-Song/Certifiable-Doppler-alignment.
>
---
#### [new 002] Quadrupped-Legged Robot Movement Plan Generation using Large Language Model
- **分类: cs.RO; cs.HC**

- **简介: 论文提出用大模型实现四足机器人自然语言控制，通过云端卸载计算，结合传感器生成可执行指令，实测成功率超90%，降低操作门槛。**

- **链接: [https://arxiv.org/pdf/2512.21293v1](https://arxiv.org/pdf/2512.21293v1)**

> **作者:** Muhtadin; Vincentius Gusti Putu A. B. M.; Ahmad Zaini; Mauridhi Hery Purnomo; I Ketut Eddy Purnama; Chastine Fatichah
>
> **摘要:** Traditional control interfaces for quadruped robots often impose a high barrier to entry, requiring specialized technical knowledge for effective operation. To address this, this paper presents a novel control framework that integrates Large Language Models (LLMs) to enable intuitive, natural language-based navigation. We propose a distributed architecture where high-level instruction processing is offloaded to an external server to overcome the onboard computational constraints of the DeepRobotics Jueying Lite 3 platform. The system grounds LLM-generated plans into executable ROS navigation commands using real-time sensor fusion (LiDAR, IMU, and Odometry). Experimental validation was conducted in a structured indoor environment across four distinct scenarios, ranging from single-room tasks to complex cross-zone navigation. The results demonstrate the system's robustness, achieving an aggregate success rate of over 90\% across all scenarios, validating the feasibility of offloaded LLM-based planning for autonomous quadruped deployment in real-world settings.
>
---
#### [new 003] Tracing Energy Flow: Learning Tactile-based Grasping Force Control to Prevent Slippage in Dynamic Object Interaction
- **分类: cs.RO**

- **简介: 论文提出基于触觉的抓取力控制方法，通过能量流建模与学习，实时优化抓力防滑，无需外部传感或先验物体知识，适用于动态交互场景。**

- **链接: [https://arxiv.org/pdf/2512.21043v1](https://arxiv.org/pdf/2512.21043v1)**

> **作者:** Cheng-Yu Kuo; Hirofumi Shin; Takamitsu Matsubara
>
> **备注:** 8 pages. Accepted by IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Regulating grasping force to reduce slippage during dynamic object interaction remains a fundamental challenge in robotic manipulation, especially when objects are manipulated by multiple rolling contacts, have unknown properties (such as mass or surface conditions), and when external sensing is unreliable. In contrast, humans can quickly regulate grasping force by touch, even without visual cues. Inspired by this ability, we aim to enable robotic hands to rapidly explore objects and learn tactile-driven grasping force control under motion and limited sensing. We propose a physics-informed energy abstraction that models the object as a virtual energy container. The inconsistency between the fingers' applied power and the object's retained energy provides a physically grounded signal for inferring slip-aware stability. Building on this abstraction, we employ model-based learning and planning to efficiently model energy dynamics from tactile sensing and perform real-time grasping force optimization. Experiments in both simulation and hardware demonstrate that our method can learn grasping force control from scratch within minutes, effectively reduce slippage, and extend grasp duration across diverse motion-object pairs, all without relying on external sensing or prior object knowledge.
>
---
#### [new 004] Robust and Efficient MuJoCo-based Model Predictive Control via Web of Affine Spaces Derivatives
- **分类: cs.RO**

- **简介: 论文用WASP导数替代MJPC中低效的有限差分，加速MPC优化，提升高维系统实时控制效率与鲁棒性，开源实现支持广泛应用。**

- **链接: [https://arxiv.org/pdf/2512.21109v1](https://arxiv.org/pdf/2512.21109v1)**

> **作者:** Chen Liang; Daniel Rakita
>
> **备注:** Submitted to 2026 IEEE International Conference on Robotics & Automation (ICRA 2026)
>
> **摘要:** MuJoCo is a powerful and efficient physics simulator widely used in robotics. One common way it is applied in practice is through Model Predictive Control (MPC), which uses repeated rollouts of the simulator to optimize future actions and generate responsive control policies in real time. To make this process more accessible, the open source library MuJoCo MPC (MJPC) provides ready-to-use MPC algorithms and implementations built directly on top of the MuJoCo simulator. However, MJPC relies on finite differencing (FD) to compute derivatives through the underlying MuJoCo simulator, which is often a key bottleneck that can make it prohibitively costly for time-sensitive tasks, especially in high-DOF systems or complex scenes. In this paper, we introduce the use of Web of Affine Spaces (WASP) derivatives within MJPC as a drop-in replacement for FD. WASP is a recently developed approach for efficiently computing sequences of accurate derivative approximations. By reusing information from prior, related derivative calculations, WASP accelerates and stabilizes the computation of new derivatives, making it especially well suited for MPC's iterative, fine-grained updates over time. We evaluate WASP across a diverse suite of MJPC tasks spanning multiple robot embodiments. Our results suggest that WASP derivatives are particularly effective in MJPC: it integrates seamlessly across tasks, delivers consistently robust performance, and achieves up to a 2$\mathsf{x}$ speedup compared to an FD backend when used with derivative-based planners, such as iLQG. In addition, WASP-based MPC outperforms MJPC's stochastic sampling-based planners on our evaluation tasks, offering both greater efficiency and reliability. To support adoption and future research, we release an open-source implementation of MJPC with WASP derivatives fully integrated.
>
---
#### [new 005] Schrödinger's Navigator: Imagining an Ensemble of Futures for Zero-Shot Object Navigation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 提出Schrödinger's Navigator框架，通过轨迹条件3D想象应对遮挡与动态目标，提升零样本物体导航在未知环境中的成功率。**

- **链接: [https://arxiv.org/pdf/2512.21201v1](https://arxiv.org/pdf/2512.21201v1)**

> **作者:** Yu He; Da Huang; Zhenyang Liu; Zixiao Gu; Qiang Sun; Guangnan Ye; Yanwei Fu
>
> **摘要:** Zero-shot object navigation (ZSON) requires a robot to locate a target object in a previously unseen environment without relying on pre-built maps or task-specific training. However, existing ZSON methods often struggle in realistic and cluttered environments, particularly when the scene contains heavy occlusions, unknown risks, or dynamically moving target objects. To address these challenges, we propose \textbf{Schrödinger's Navigator}, a navigation framework inspired by Schrödinger's thought experiment on uncertainty. The framework treats unobserved space as a set of plausible future worlds and reasons over them before acting. Conditioned on egocentric visual inputs and three candidate trajectories, a trajectory-conditioned 3D world model imagines future observations along each path. This enables the agent to see beyond occlusions and anticipate risks in unseen regions without requiring extra detours or dense global mapping. The imagined 3D observations are fused into the navigation map and used to update a value map. These updates guide the policy toward trajectories that avoid occlusions, reduce exposure to uncertain space, and better track moving targets. Experiments on a Go2 quadruped robot across three challenging scenarios, including severe static occlusions, unknown risks, and dynamically moving targets, show that Schrödinger's Navigator consistently outperforms strong ZSON baselines in self-localization, object localization, and overall Success Rate in occlusion-heavy environments. These results demonstrate the effectiveness of trajectory-conditioned 3D imagination in enabling robust zero-shot object navigation.
>
---
#### [new 006] SparScene: Efficient Traffic Scene Representation via Sparse Graph Learning for Large-Scale Trajectory Generation
- **分类: cs.RO**

- **简介: 论文提出SparScene，用稀疏图学习高效建模交通场景，解决大规模轨迹生成中密集图效率低问题，显著提升推理速度与可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.21133v1](https://arxiv.org/pdf/2512.21133v1)**

> **作者:** Xiaoyu Mo; Jintian Ge; Zifan Wang; Chen Lv; Karl Henrik Johansson
>
> **备注:** 13 pages, 7 figures, 5 tables
>
> **摘要:** Multi-agent trajectory generation is a core problem for autonomous driving and intelligent transportation systems. However, efficiently modeling the dynamic interactions between numerous road users and infrastructures in complex scenes remains an open problem. Existing methods typically employ distance-based or fully connected dense graph structures to capture interaction information, which not only introduces a large number of redundant edges but also requires complex and heavily parameterized networks for encoding, thereby resulting in low training and inference efficiency, limiting scalability to large and complex traffic scenes. To overcome the limitations of existing methods, we propose SparScene, a sparse graph learning framework designed for efficient and scalable traffic scene representation. Instead of relying on distance thresholds, SparScene leverages the lane graph topology to construct structure-aware sparse connections between agents and lanes, enabling efficient yet informative scene graph representation. SparScene adopts a lightweight graph encoder that efficiently aggregates agent-map and agent-agent interactions, yielding compact scene representations with substantially improved efficiency and scalability. On the motion prediction benchmark of the Waymo Open Motion Dataset (WOMD), SparScene achieves competitive performance with remarkable efficiency. It generates trajectories for more than 200 agents in a scene within 5 ms and scales to more than 5,000 agents and 17,000 lanes with merely 54 ms of inference time with a GPU memory of 2.9 GB, highlighting its superior scalability for large-scale traffic scenes.
>
---
#### [new 007] Flocking phase transition and threat responses in bio-inspired autonomous drone swarms
- **分类: cs.RO; eess.SY; nlin.AO**

- **简介: 论文研究无人机群仿生集群相变与威胁响应，通过调节局部交互参数实现群集状态切换，提升抗扰动能力与恢复效率。**

- **链接: [https://arxiv.org/pdf/2512.21196v1](https://arxiv.org/pdf/2512.21196v1)**

> **作者:** Matthieu Verdoucq; Dari Trendafilov; Clément Sire; Ramón Escobedo; Guy Theraulaz; Gautier Hattenberger
>
> **摘要:** Collective motion inspired by animal groups offers powerful design principles for autonomous aerial swarms. We present a bio-inspired 3D flocking algorithm in which each drone interacts only with a minimal set of influential neighbors, relying solely on local alignment and attraction cues. By systematically tuning these two interaction gains, we map a phase diagram revealing sharp transitions between swarming and schooling, as well as a critical region where susceptibility, polarization fluctuations, and reorganization capacity peak. Outdoor experiments with a swarm of ten drones, combined with simulations using a calibrated flight-dynamics model, show that operating near this transition enhances responsiveness to external disturbances. When confronted with an intruder, the swarm performs rapid collective turns, transient expansions, and reliably recovers high alignment within seconds. These results demonstrate that minimal local-interaction rules are sufficient to generate multiple collective phases and that simple gain modulation offers an efficient mechanism to adjust stability, flexibility, and resilience in drone swarms.
>
---
#### [new 008] Language-Guided Grasp Detection with Coarse-to-Fine Learning for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 提出LGGD方法，用粗到精学习实现语言引导抓取，解决语义对齐弱问题，提升机器人按指令抓物的准确性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.21065v1](https://arxiv.org/pdf/2512.21065v1)**

> **作者:** Zebin Jiang; Tianle Jin; Xiangtong Yao; Alois Knoll; Hu Cao
>
> **备注:** Submitted to IEEE Journal
>
> **摘要:** Grasping is one of the most fundamental challenging capabilities in robotic manipulation, especially in unstructured, cluttered, and semantically diverse environments. Recent researches have increasingly explored language-guided manipulation, where robots not only perceive the scene but also interpret task-relevant natural language instructions. However, existing language-conditioned grasping methods typically rely on shallow fusion strategies, leading to limited semantic grounding and weak alignment between linguistic intent and visual grasp reasoning.In this work, we propose Language-Guided Grasp Detection (LGGD) with a coarse-to-fine learning paradigm for robotic manipulation. LGGD leverages CLIP-based visual and textual embeddings within a hierarchical cross-modal fusion pipeline, progressively injecting linguistic cues into the visual feature reconstruction process. This design enables fine-grained visual-semantic alignment and improves the feasibility of the predicted grasps with respect to task instructions. In addition, we introduce a language-conditioned dynamic convolution head (LDCH) that mixes multiple convolution experts based on sentence-level features, enabling instruction-adaptive coarse mask and grasp predictions. A final refinement module further enhances grasp consistency and robustness in complex scenes.Experiments on the OCID-VLG and Grasp-Anything++ datasets show that LGGD surpasses existing language-guided grasping methods, exhibiting strong generalization to unseen objects and diverse language queries. Moreover, deployment on a real robotic platform demonstrates the practical effectiveness of our approach in executing accurate, instruction-conditioned grasp actions. The code will be released publicly upon acceptance.
>
---
#### [new 009] YCB-Handovers Dataset: Analyzing Object Weight Impact on Human Handovers to Adapt Robotic Handover Motion
- **分类: cs.RO; cs.HC**

- **简介: 论文构建YCB-Handovers数据集，分析物体重量对人类传递动作的影响，旨在优化机器人自适应传递行为，提升人机协作自然性。**

- **链接: [https://arxiv.org/pdf/2512.20847v1](https://arxiv.org/pdf/2512.20847v1)**

> **作者:** Parag Khanna; Karen Jane Dsouza; Chunyu Wang; Mårten Björkman; Christian Smith
>
> **备注:** Paper presented at the IEEE International Conference on Robot and Human Interactive Communication (RO-MAN), 2025
>
> **摘要:** This paper introduces the YCB-Handovers dataset, capturing motion data of 2771 human-human handovers with varying object weights. The dataset aims to bridge a gap in human-robot collaboration research, providing insights into the impact of object weight in human handovers and readiness cues for intuitive robotic motion planning. The underlying dataset for object recognition and tracking is the YCB (Yale-CMU-Berkeley) dataset, which is an established standard dataset used in algorithms for robotic manipulation, including grasping and carrying objects. The YCB-Handovers dataset incorporates human motion patterns in handovers, making it applicable for data-driven, human-inspired models aimed at weight-sensitive motion planning and adaptive robotic behaviors. This dataset covers an extensive range of weights, allowing for a more robust study of handover behavior and weight variation. Some objects also require careful handovers, highlighting contrasts with standard handovers. We also provide a detailed analysis of the object's weight impact on the human reaching motion in these handovers.
>
---
#### [new 010] RoboCade: Gamifying Robot Data Collection
- **分类: cs.RO**

- **简介: 论文提出RoboCade平台，通过游戏化远程操控吸引大众参与机器人演示数据收集，提升数据规模与用户参与度，有效提高下游策略训练成功率。**

- **链接: [https://arxiv.org/pdf/2512.21235v1](https://arxiv.org/pdf/2512.21235v1)**

> **作者:** Suvir Mirchandani; Mia Tang; Jiafei Duan; Jubayer Ibn Hamid; Michael Cho; Dorsa Sadigh
>
> **备注:** 10 pages, 9 figures
>
> **摘要:** Imitation learning from human demonstrations has become a dominant approach for training autonomous robot policies. However, collecting demonstration datasets is costly: it often requires access to robots and needs sustained effort in a tedious, long process. These factors limit the scale of data available for training policies. We aim to address this scalability challenge by involving a broader audience in a gamified data collection experience that is both accessible and motivating. Specifically, we develop a gamified remote teleoperation platform, RoboCade, to engage general users in collecting data that is beneficial for downstream policy training. To do this, we embed gamification strategies into the design of the system interface and data collection tasks. In the system interface, we include components such as visual feedback, sound effects, goal visualizations, progress bars, leaderboards, and badges. We additionally propose principles for constructing gamified tasks that have overlapping structure with useful downstream target tasks. We instantiate RoboCade on three manipulation tasks -- including spatial arrangement, scanning, and insertion. To illustrate the viability of gamified robot data collection, we collect a demonstration dataset through our platform, and show that co-training robot policies with this data can improve success rate on non-gamified target tasks (+16-56%). Further, we conduct a user study to validate that novice users find the gamified platform significantly more enjoyable than a standard non-gamified platform (+24%). These results highlight the promise of gamified data collection as a scalable, accessible, and engaging method for collecting demonstration data.
>
---
#### [new 011] Anytime Metaheuristic Framework for Global Route Optimization in Expected-Time Mobile Search
- **分类: cs.RO**

- **简介: 论文提出Milaps框架，用元启发式算法优化移动搜索中全局路径，以最小化发现隐藏目标的期望时间，兼顾解质与效率。**

- **链接: [https://arxiv.org/pdf/2512.20711v1](https://arxiv.org/pdf/2512.20711v1)**

> **作者:** Jan Mikula; Miroslav Kulich
>
> **备注:** 20 pages, 42 figures (including subfigures); submitted to IEEE Transactions on Robotics (T-RO) in February 2025
>
> **摘要:** Expected-time mobile search (ETS) is a fundamental robotics task where a mobile sensor navigates an environment to minimize the expected time required to locate a hidden object. Global route optimization for ETS in static 2D continuous environments remains largely underexplored due to the intractability of objective evaluation, stemming from the continuous nature of the environment and the interplay of motion and visibility constraints. Prior work has addressed this through partial discretization, leading to discrete-sensing formulations tackled via utility-greedy heuristics. Others have taken an indirect approach by heuristically approximating the objective using minimum latency problems on fixed graphs, enabling global route optimization via efficient metaheuristics. This paper builds on and significantly extends the latter by introducing Milaps (Minimum latency problems), a model-based solution framework for ETS. Milaps integrates novel auxiliary objectives and adapts a recent anytime metaheuristic for the traveling deliveryman problem, chosen for its strong performance under tight runtime constraints. Evaluations on a novel large-scale dataset demonstrate superior trade-offs between solution quality and runtime compared to state-of-the-art baselines. The best-performing strategy rapidly generates a preliminary solution, assigns static weights to sensing configurations, and optimizes global costs metaheuristically. Additionally, a qualitative study highlights the framework's flexibility across diverse scenarios.
>
---
#### [new 012] Multimodal Sensing for Robot-Assisted Sub-Tissue Feature Detection in Physiotherapy Palpation
- **分类: cs.RO**

- **简介: 论文提出多模态传感方案，融合触觉成像与力传感，解决机器人理疗按压中软组织下细微结构检测难题，提升特征识别鲁棒性与操作安全性。**

- **链接: [https://arxiv.org/pdf/2512.20992v1](https://arxiv.org/pdf/2512.20992v1)**

> **作者:** Tian-Ao Ren; Jorge Garcia; Seongheon Hong; Jared Grinberg; Hojung Choi; Julia Di; Hao Li; Dmitry Grinberg; Mark R. Cutkosky
>
> **备注:** 6 pages, 9 figures, submitted to DMD2026
>
> **摘要:** Robotic palpation relies on force sensing, but force signals in soft-tissue environments are variable and cannot reliably reveal subtle subsurface features. We present a compact multimodal sensor that integrates high-resolution vision-based tactile imaging with a 6-axis force-torque sensor. In experiments on silicone phantoms with diverse subsurface tendon geometries, force signals alone frequently produce ambiguous responses, while tactile images reveal clear structural differences in presence, diameter, depth, crossings, and multiplicity. Yet accurate force tracking remains essential for maintaining safe, consistent contact during physiotherapeutic interaction. Preliminary results show that combining tactile and force modalities enables robust subsurface feature detection and controlled robotic palpation.
>
---
#### [new 013] Stretchable and High-Precision Optical Tactile Sensor for Trajectory Tracking of Parallel Mechanisms
- **分类: cs.RO**

- **简介: 提出一种基于光谱滤波的可拉伸触觉传感器，解决高分辨率与抗干扰难题，实现7μm空间精度，并用于并联机构轨迹实时跟踪。**

- **链接: [https://arxiv.org/pdf/2512.20888v1](https://arxiv.org/pdf/2512.20888v1)**

> **作者:** Yiding Nie; Dongliang Fan; Jiatai Huang; Chunyu Liu; Jian S. Dai
>
> **备注:** Accepted by 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Stretchable sensors indicate promising prospects for soft robotics, medical devices, and human-machine interactions due to the high compliance of soft materials. Discrete sensing strategies, including sensor arrays and distributed sensors, are broadly involved in tactile sensors across versatile applications. However, it remains a challenge to achieve high spatial resolution with self-decoupled capacity and insensitivity to other off-axis stimuli for stretchable tactile sensors. Herein, we develop a stretchable tactile sensor based on the proposed continuous spectral-filtering principle, allowing superhigh resolution for applied stimuli. This proposed sensor enables a high-linear spatial response (0.996) even during stretching and bending, and high continuous spatial (7 μm) and force (5 mN) resolutions with design scalability and interaction robustness to survive piercing and cutting. We further demonstrate the sensors' performance by integrating them into a planar parallel mechanism for precise trajectory tracking (rotational resolution: 0.02°) in real time.
>
---
#### [new 014] LookPlanGraph: Embodied Instruction Following Method with VLM Graph Augmentation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 论文提出LookPlanGraph，用VLM动态更新场景图，解决静态图无法应对环境变化问题，提升机器人指令跟随效果，含仿真与真实实验及新数据集GraSIF。**

- **链接: [https://arxiv.org/pdf/2512.21243v1](https://arxiv.org/pdf/2512.21243v1)**

> **作者:** Anatoly O. Onishchenko; Alexey K. Kovalev; Aleksandr I. Panov
>
> **摘要:** Methods that use Large Language Models (LLM) as planners for embodied instruction following tasks have become widespread. To successfully complete tasks, the LLM must be grounded in the environment in which the robot operates. One solution is to use a scene graph that contains all the necessary information. Modern methods rely on prebuilt scene graphs and assume that all task-relevant information is available at the start of planning. However, these approaches do not account for changes in the environment that may occur between the graph construction and the task execution. We propose LookPlanGraph - a method that leverages a scene graph composed of static assets and object priors. During plan execution, LookPlanGraph continuously updates the graph with relevant objects, either by verifying existing priors or discovering new entities. This is achieved by processing the agents egocentric camera view using a Vision Language Model. We conducted experiments with changed object positions VirtualHome and OmniGibson simulated environments, demonstrating that LookPlanGraph outperforms methods based on predefined static scene graphs. To demonstrate the practical applicability of our approach, we also conducted experiments in a real-world setting. Additionally, we introduce the GraSIF (Graph Scenes for Instruction Following) dataset with automated validation framework, comprising 514 tasks drawn from SayPlan Office, BEHAVIOR-1K, and VirtualHome RobotHow. Project page available at https://lookplangraph.github.io .
>
---
#### [new 015] Relative Localization System Design for SnailBot: A Modular Self-reconfigurable Robot
- **分类: cs.RO; eess.SY**

- **简介: 论文设计SnailBot模块化机器人相对定位系统，融合ArUco、光流与IMU数据，实现实时鲁棒协作定位，解决动态场景下精准自定位问题。**

- **链接: [https://arxiv.org/pdf/2512.21226v1](https://arxiv.org/pdf/2512.21226v1)**

> **作者:** Shuhan Zhang; Tin Lun Lam
>
> **备注:** 7 pages, 7 figures, 4 algorithms
>
> **摘要:** This paper presents the design and implementation of a relative localization system for SnailBot, a modular self reconfigurable robot. The system integrates ArUco marker recognition, optical flow analysis, and IMU data processing into a unified fusion framework, enabling robust and accurate relative positioning for collaborative robotic tasks. Experimental validation demonstrates the effectiveness of the system in realtime operation, with a rule based fusion strategy ensuring reliability across dynamic scenarios. The results highlight the potential for scalable deployment in modular robotic systems.
>
---
#### [new 016] ETP-R1: Evolving Topological Planning with Reinforcement Fine-tuning for Vision-Language Navigation in Continuous Environments
- **分类: cs.RO**

- **简介: 论文提出ETP-R1框架，用于连续环境视觉语言导航，结合大规模预训练与强化微调，提升图基方法性能，在R2R/RxR-CE达SOTA。**

- **链接: [https://arxiv.org/pdf/2512.20940v1](https://arxiv.org/pdf/2512.20940v1)**

> **作者:** Shuhao Ye; Sitong Mao; Yuxiang Cui; Xuan Yu; Shichao Zhai; Wen Chen; Shunbo Zhou; Rong Xiong; Yue Wang
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Vision-Language Navigation in Continuous Environments (VLN-CE) requires an embodied agent to navigate towards target in continuous environments, following natural language instructions. While current graph-based methods offer an efficient, structured approach by abstracting the environment into a topological map and simplifying the action space to waypoint selection, they lag behind methods based on Large Vision-Language Models (LVLMs) in leveraging large-scale data and advanced training paradigms. In this paper, we try to bridge this gap by introducing ETP-R1, a framework that applies the paradigm of scaling up data and Reinforcement Fine-Tuning (RFT) to a graph-based VLN-CE model. To build a strong foundation, we first construct a high-quality, large-scale pretraining dataset using the Gemini API. This dataset consists of diverse, low-hallucination instructions for topological trajectories, providing rich supervision for our graph-based policy to map language to topological paths. This foundation is further strengthened by unifying data from both R2R and RxR tasks for joint pretraining. Building on this, we introduce a three-stage training paradigm, which culminates in the first application of closed-loop, online RFT to a graph-based VLN-CE model, powered by the Group Relative Policy Optimization (GRPO) algorithm. Extensive experiments demonstrate that our approach is highly effective, establishing new state-of-the-art performance across all major metrics on both the R2R-CE and RxR-CE benchmarks. Our code is available at https://github.com/Cepillar/ETP-R1.
>
---
#### [new 017] From Human Bias to Robot Choice: How Occupational Contexts and Racial Priming Shape Robot Selection
- **分类: cs.RO; cs.HC**

- **简介: 研究人类职业偏见如何影响机器人选择，通过实验揭示肤色与职业刻板印象对人机决策的影响，警示机器人部署或加剧社会不平等。**

- **链接: [https://arxiv.org/pdf/2512.20951v1](https://arxiv.org/pdf/2512.20951v1)**

> **作者:** Jiangen He; Wanqi Zhang; Jessica Barfield
>
> **备注:** HRI '26
>
> **摘要:** As artificial agents increasingly integrate into professional environments, fundamental questions have emerged about how societal biases influence human-robot selection decisions. We conducted two comprehensive experiments (N = 1,038) examining how occupational contexts and stereotype activation shape robotic agent choices across construction, healthcare, educational, and athletic domains. Participants made selections from artificial agents that varied systematically in skin tone and anthropomorphic characteristics. Our study revealed distinct context-dependent patterns. Healthcare and educational scenarios demonstrated strong favoritism toward lighter-skinned artificial agents, while construction and athletic contexts showed greater acceptance of darker-toned alternatives. Participant race was associated with systematic differences in selection patterns across professional domains. The second experiment demonstrated that exposure to human professionals from specific racial backgrounds systematically shifted later robotic agent preferences in stereotype-consistent directions. These findings show that occupational biases and color-based discrimination transfer directly from human-human to human-robot evaluation contexts. The results highlight mechanisms through which robotic deployment may unintentionally perpetuate existing social inequalities.
>
---
#### [new 018] Wireless Center of Pressure Feedback System for Humanoid Robot Balance Control using ESP32-C3
- **分类: cs.RO; eess.SY**

- **简介: 提出无线CoP反馈系统，用ESP32-C3实时估算足底压力中心，通过PID控制提升人形机器人单腿站立平衡能力，解决传统有线传感限制关节灵活性问题。**

- **链接: [https://arxiv.org/pdf/2512.21219v1](https://arxiv.org/pdf/2512.21219v1)**

> **作者:** Muhtadin; Faris Rafi Pramana; Dion Hayu Fandiantoro; Moh Ismarintan Zazuli; Atar Fuady Babgei
>
> **摘要:** Maintaining stability during the single-support phase is a fundamental challenge in humanoid robotics, particularly in dance robots that require complex maneuvers and high mechanical freedom. Traditional tethered sensor configurations often restrict joint movement and introduce mechanical noises. This study proposes a wireless embedded balance system designed to maintain stability on uneven surfaces. The system utilizes a custom-designed foot unit integrated with four load cells and an ESP32-C3 microcontroller to estimate the Center of Pressure (CoP) in real time. The CoP data were transmitted wirelessly to the main controller to minimize the wiring complexity of the 29-DoF VI-ROSE humanoid robot. A PID control strategy is implemented to adjust the torso, hip, and ankle roll joints based on CoP feedback. Experimental characterization demonstrated high sensor precision with an average measurement error of 14.8 g. Furthermore, the proposed control system achieved a 100% success rate in maintaining balance during single-leg lifting tasks at a 3-degree inclination with optimized PID parameters (Kp=0.10, Kd=0.005). These results validate the efficacy of wireless CoP feedback in enhancing the postural stability of humanoid robots, without compromising their mechanical flexibility.
>
---
#### [new 019] Proprioception Enhances Vision Language Model in Generating Captions and Subtask Segmentations for Robot Task
- **分类: cs.RO**

- **简介: 论文提升VLM对机器人任务理解，利用本体感知数据增强视频描述与子任务分割，解决纯视觉语言模型缺乏运动信息的问题。**

- **链接: [https://arxiv.org/pdf/2512.20876v1](https://arxiv.org/pdf/2512.20876v1)**

> **作者:** Kanata Suzuki; Shota Shimizu; Tetsuya Ogata
>
> **摘要:** From the perspective of future developments in robotics, it is crucial to verify whether foundation models trained exclusively on offline data, such as images and language, can understand the robot motion. In particular, since Vision Language Models (VLMs) do not include low-level motion information from robots in their training datasets, video understanding including trajectory information remains a significant challenge. In this study, we assess two capabilities of VLMs through a video captioning task with low-level robot motion information: (1) automatic captioning of robot tasks and (2) segmentation of a series of tasks. Both capabilities are expected to enhance the efficiency of robot imitation learning by linking language and motion and serve as a measure of the foundation model's performance. The proposed method generates multiple "scene" captions using image captions and trajectory data from robot tasks. The full task caption is then generated by summarizing these individual captions. Additionally, the method performs subtask segmentation by comparing the similarity between text embeddings of image captions. In both captioning tasks, the proposed method aims to improve performance by providing the robot's motion data - joint and end-effector states - as input to the VLM. Simulator experiments were conducted to validate the effectiveness of the proposed method.
>
---
#### [new 020] Early warning signals for loss of control
- **分类: cs.RO; eess.SY**

- **简介: 提出无模型动态指标监测反馈系统失稳前兆，基于临界减速现象，适用于无人机等复杂系统实时预警与韧性设计。**

- **链接: [https://arxiv.org/pdf/2512.20868v1](https://arxiv.org/pdf/2512.20868v1)**

> **作者:** Jasper J. van Beers; Marten Scheffer; Prashant Solanki; Ingrid A. van de Leemput; Egbert H. van Nes; Coen C. de Visser
>
> **摘要:** Maintaining stability in feedback systems, from aircraft and autonomous robots to biological and physiological systems, relies on monitoring their behavior and continuously adjusting their inputs. Incremental damage can make such control fragile. This tends to go unnoticed until a small perturbation induces instability (i.e. loss of control). Traditional methods in the field of engineering rely on accurate system models to compute a safe set of operating instructions, which become invalid when the, possibly damaged, system diverges from its model. Here we demonstrate that the approach of such a feedback system towards instability can nonetheless be monitored through dynamical indicators of resilience. This holistic system safety monitor does not rely on a system model and is based on the generic phenomenon of critical slowing down, shown to occur in the climate, biology and other complex nonlinear systems approaching criticality. Our findings for engineered devices opens up a wide range of applications involving real-time early warning systems as well as an empirical guidance of resilient system design exploration, or "tinkering". While we demonstrate the validity using drones, the generic nature of the underlying principles suggest that these indicators could apply across a wider class of controlled systems including reactors, aircraft, and self-driving cars.
>
---
#### [new 021] UniTacHand: Unified Spatio-Tactile Representation for Human to Robotic Hand Skill Transfer
- **分类: cs.RO**

- **简介: 提出UniTacHand，统一人手与机器人触觉表征，实现零样本技能迁移，解决触觉数据稀缺与异构对齐难题，提升灵巧操作泛化性与数据效率。**

- **链接: [https://arxiv.org/pdf/2512.21233v1](https://arxiv.org/pdf/2512.21233v1)**

> **作者:** Chi Zhang; Penglin Cai; Haoqi Yuan; Chaoyi Xu; Zongqing Lu
>
> **摘要:** Tactile sensing is crucial for robotic hands to achieve human-level dexterous manipulation, especially in scenarios with visual occlusion. However, its application is often hindered by the difficulty of collecting large-scale real-world robotic tactile data. In this study, we propose to collect low-cost human manipulation data using haptic gloves for tactile-based robotic policy learning. The misalignment between human and robotic tactile data makes it challenging to transfer policies learned from human data to robots. To bridge this gap, we propose UniTacHand, a unified representation to align robotic tactile information captured by dexterous hands with human hand touch obtained from gloves. First, we project tactile signals from both human hands and robotic hands onto a morphologically consistent 2D surface space of the MANO hand model. This unification standardizes the heterogeneous data structures and inherently embeds the tactile signals with spatial context. Then, we introduce a contrastive learning method to align them into a unified latent space, trained on only 10 minutes of paired data from our data collection system. Our approach enables zero-shot tactile-based policy transfer from humans to a real robot, generalizing to objects unseen in the pre-training data. We also demonstrate that co-training on mixed data, including both human and robotic demonstrations via UniTacHand, yields better performance and data efficiency compared with using only robotic data. UniTacHand paves a path toward general, scalable, and data-efficient learning for tactile-based dexterous hands.
>
---
#### [new 022] Global End-Effector Pose Control of an Underactuated Aerial Manipulator via Reinforcement Learning
- **分类: cs.RO**

- **简介: 论文用强化学习控制欠驱动空中机械臂，实现六自由度末端精准操控，解决轻量化设计下的扰动鲁棒性问题，实验验证厘米级定位与抗外力能力。**

- **链接: [https://arxiv.org/pdf/2512.21085v1](https://arxiv.org/pdf/2512.21085v1)**

> **作者:** Shlok Deshmukh; Javier Alonso-Mora; Sihao Sun
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** Aerial manipulators, which combine robotic arms with multi-rotor drones, face strict constraints on arm weight and mechanical complexity. In this work, we study a lightweight 2-degree-of-freedom (DoF) arm mounted on a quadrotor via a differential mechanism, capable of full six-DoF end-effector pose control. While the minimal design enables simplicity and reduced payload, it also introduces challenges such as underactuation and sensitivity to external disturbances, including manipulation of heavy loads and pushing tasks. To address these, we employ reinforcement learning, training a Proximal Policy Optimization (PPO) agent in simulation to generate feedforward commands for quadrotor acceleration and body rates, along with joint angle targets. These commands are tracked by an incremental nonlinear dynamic inversion (INDI) attitude controller and a PID joint controller, respectively. Flight experiments demonstrate centimeter-level position accuracy and degree-level orientation precision, with robust performance under external force disturbances. The results highlight the potential of learning-based control strategies for enabling contact-rich aerial manipulation using simple, lightweight platforms.
>
---
#### [new 023] A General Purpose Method for Robotic Interception of Non-Cooperative Dynamic Targets
- **分类: cs.RO**

- **简介: 提出通用视觉拦截框架，解决无合作动态目标的自主拦截问题，适配多平台，融合滤波、预测与实时规划，实测高效鲁棒。**

- **链接: [https://arxiv.org/pdf/2512.20769v1](https://arxiv.org/pdf/2512.20769v1)**

> **作者:** Tanmay P. Patel; Erica L. Tevere; Erik H. Kramer; Rudranarayan M. Mukherjee
>
> **备注:** 10 pages, 11 figures, 5 tables. Accepted to IEEE Aerospace Conference 2026
>
> **摘要:** This paper presents a general purpose framework for autonomous, vision-based interception of dynamic, non-cooperative targets, validated across three distinct mobility platforms: an unmanned aerial vehicle (UAV), a four-wheeled ground rover, and an air-thruster spacecraft testbed. The approach relies solely on a monocular camera with fiducials for target tracking and operates entirely in the local observer frame without the need for global information. The core contribution of this work is a streamlined and general approach to autonomous interception that can be adapted across robots with varying dynamics, as well as our comprehensive study of the robot interception problem across heterogenous mobility systems under limited observability and no global localization. Our method integrates (1) an Extended Kalman Filter for relative pose estimation amid intermittent measurements, (2) a history-conditioned motion predictor for dynamic target trajectory propagation, and (3) a receding-horizon planner solving a constrained convex program in real time to ensure time-efficient and kinematically feasible interception paths. Our operating regime assumes that observability is restricted by partial fields of view, sensor dropouts, and target occlusions. Experiments are performed in these conditions and include autonomous UAV landing on dynamic targets, rover rendezvous and leader-follower tasks, and spacecraft proximity operations. Results from simulated and physical experiments demonstrate robust performance with low interception errors (both during station-keeping and upon scenario completion), high success rates under deterministic and stochastic target motion profiles, and real-time execution on embedded processors such as the Jetson Orin, VOXL2, and Raspberry Pi 5. These results highlight the framework's generalizability, robustness, and computational efficiency.
>
---
#### [new 024] Towards Optimal Performance and Action Consistency Guarantees in Dec-POMDPs with Inconsistent Beliefs and Limited Communication
- **分类: cs.MA; cs.AI; cs.RO**

- **简介: 论文解决多智能体信念不一致下的协同决策问题，提出新框架保障动作一致性与性能，按需通信，优于现有算法。**

- **链接: [https://arxiv.org/pdf/2512.20778v1](https://arxiv.org/pdf/2512.20778v1)**

> **作者:** Moshe Rafaeli Shimron; Vadim Indelman
>
> **备注:** 9 pages, 3 figures, 2 tables
>
> **摘要:** Multi-agent decision-making under uncertainty is fundamental for effective and safe autonomous operation. In many real-world scenarios, each agent maintains its own belief over the environment and must plan actions accordingly. However, most existing approaches assume that all agents have identical beliefs at planning time, implying these beliefs are conditioned on the same data. Such an assumption is often impractical due to limited communication. In reality, agents frequently operate with inconsistent beliefs, which can lead to poor coordination and suboptimal, potentially unsafe, performance. In this paper, we address this critical challenge by introducing a novel decentralized framework for optimal joint action selection that explicitly accounts for belief inconsistencies. Our approach provides probabilistic guarantees for both action consistency and performance with respect to open-loop multi-agent POMDP (which assumes all data is always communicated), and selectively triggers communication only when needed. Furthermore, we address another key aspect of whether, given a chosen joint action, the agents should share data to improve expected performance in inference. Simulation results show our approach outperforms state-of-the-art algorithms.
>
---
#### [new 025] RoboSafe: Safeguarding Embodied Agents via Executable Safety Logic
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 提出RoboSafe，通过可执行安全逻辑动态防护具身智能体，解决其执行中隐式风险问题，结合回溯与前瞻推理，显著降险保效。**

- **链接: [https://arxiv.org/pdf/2512.21220v1](https://arxiv.org/pdf/2512.21220v1)**

> **作者:** Le Wang; Zonghao Ying; Xiao Yang; Quanchen Zou; Zhenfei Yin; Tianlin Li; Jian Yang; Yaodong Yang; Aishan Liu; Xianglong Liu
>
> **备注:** 11 pages, 6 figures
>
> **摘要:** Embodied agents powered by vision-language models (VLMs) are increasingly capable of executing complex real-world tasks, yet they remain vulnerable to hazardous instructions that may trigger unsafe behaviors. Runtime safety guardrails, which intercept hazardous actions during task execution, offer a promising solution due to their flexibility. However, existing defenses often rely on static rule filters or prompt-level control, which struggle to address implicit risks arising in dynamic, temporally dependent, and context-rich environments. To address this, we propose RoboSafe, a hybrid reasoning runtime safeguard for embodied agents through executable predicate-based safety logic. RoboSafe integrates two complementary reasoning processes on a Hybrid Long-Short Safety Memory. We first propose a Backward Reflective Reasoning module that continuously revisits recent trajectories in short-term memory to infer temporal safety predicates and proactively triggers replanning when violations are detected. We then propose a Forward Predictive Reasoning module that anticipates upcoming risks by generating context-aware safety predicates from the long-term safety memory and the agent's multimodal observations. Together, these components form an adaptive, verifiable safety logic that is both interpretable and executable as code. Extensive experiments across multiple agents demonstrate that RoboSafe substantially reduces hazardous actions (-36.8% risk occurrence) compared with leading baselines, while maintaining near-original task performance. Real-world evaluations on physical robotic arms further confirm its practicality. Code will be released upon acceptance.
>
---
#### [new 026] Fixed-time control with prescribed performance for path following of underwater gliders
- **分类: eess.SY; cs.RO; math.OC**

- **简介: 提出固定时间控制方案，解决水下滑翔机在强流扰动下路径跟踪问题，确保误差限内快速收敛，提升鲁棒性与安全性。**

- **链接: [https://arxiv.org/pdf/2512.20748v1](https://arxiv.org/pdf/2512.20748v1)**

> **作者:** Hanzhi Yang; Nina Mahmoudian
>
> **备注:** 22 pages, 13 figures, 2 tables, Submitted to Ocean Engineering
>
> **摘要:** Underwater gliders are increasingly deployed in challenging missions - such as hurricane-season observations and long-endurance environmental monitoring - where strong currents and turbulence pose significant risks to navigation safety. To address these practical challenges, this paper presents a fixed-time prescribed performance control scheme for the 3D path following of underwater gliders subject to model uncertainties and environmental disturbances. The primary contribution is the integration of a finite-time performance function within a fixed-time control framework. This synthesis ensures that the tracking errors are constrained within prescribed performance bounds and converge to a compact set within a fixed time, independent of initial conditions. A second key contribution is the development of a fixed-time sliding mode disturbance observer that provides accurate finite-time estimation of lumped disturbances, enhancing the system's robustness. Integrated with an iLOS guidance law, the proposed controller enables precise and safe waypoint following. Numerical simulations demonstrate that the proposed method outperforms conventional sliding mode and prescribed performance controllers in tracking accuracy, convergence speed, and control effort smoothness, validating its efficacy for robust underwater navigation.
>
---
#### [new 027] Generalised Linear Models in Deep Bayesian RL with Learnable Basis Functions
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 提出GLiBRL方法，用可学习基函数的广义线性模型改进深度贝叶斯强化学习，解决模型难优化、任务参数模糊问题，提升泛化与成功率。**

- **链接: [https://arxiv.org/pdf/2512.20974v1](https://arxiv.org/pdf/2512.20974v1)**

> **作者:** Jingyang You; Hanna Kurniawati
>
> **摘要:** Bayesian Reinforcement Learning (BRL) provides a framework for generalisation of Reinforcement Learning (RL) problems from its use of Bayesian task parameters in the transition and reward models. However, classical BRL methods assume known forms of transition and reward models, reducing their applicability in real-world problems. As a result, recent deep BRL methods have started to incorporate model learning, though the use of neural networks directly on the joint data and task parameters requires optimising the Evidence Lower Bound (ELBO). ELBOs are difficult to optimise and may result in indistinctive task parameters, hence compromised BRL policies. To this end, we introduce a novel deep BRL method, Generalised Linear Models in Deep Bayesian RL with Learnable Basis Functions (GLiBRL), that enables efficient and accurate learning of transition and reward models, with fully tractable marginal likelihood and Bayesian inference on task parameters and model noises. On challenging MetaWorld ML10/45 benchmarks, GLiBRL improves the success rate of one of the state-of-the-art deep BRL methods, VariBAD, by up to 2.7x. Comparing against representative or recent deep BRL / Meta-RL methods, such as MAML, RL2, SDVT, TrMRL and ECET, GLiBRL also demonstrates its low-variance and decent performance consistently.
>
---
## 更新

#### [replaced 001] RGMP: Recurrent Geometric-prior Multimodal Policy for Generalizable Humanoid Robot Manipulation
- **分类: cs.RO**

- **简介: 提出RGMP框架，融合几何先验与递归高斯网络，提升人形机器人多模态操作的泛化能力与数据效率，解决现有方法忽视几何推理、训练低效问题。**

- **链接: [https://arxiv.org/pdf/2511.09141v2](https://arxiv.org/pdf/2511.09141v2)**

> **作者:** Xuetao Li; Wenke Huang; Nengyuan Pan; Kaiyan Zhao; Songhua Yang; Yiming Wang; Mengde Li; Mang Ye; Jifeng Xuan; Miao Li
>
> **摘要:** Humanoid robots exhibit significant potential in executing diverse human-level skills. However, current research predominantly relies on data-driven approaches that necessitate extensive training datasets to achieve robust multimodal decision-making capabilities and generalizable visuomotor control. These methods raise concerns due to the neglect of geometric reasoning in unseen scenarios and the inefficient modeling of robot-target relationships within the training data, resulting in significant waste of training resources. To address these limitations, we present the Recurrent Geometric-prior Multimodal Policy (RGMP), an end-to-end framework that unifies geometric-semantic skill reasoning with data-efficient visuomotor control. For perception capabilities, we propose the Geometric-prior Skill Selector, which infuses geometric inductive biases into a vision language model, producing adaptive skill sequences for unseen scenes with minimal spatial common sense tuning. To achieve data-efficient robotic motion synthesis, we introduce the Adaptive Recursive Gaussian Network, which parameterizes robot-object interactions as a compact hierarchy of Gaussian processes that recursively encode multi-scale spatial relationships, yielding dexterous, data-efficient motion synthesis even from sparse demonstrations. Evaluated on both our humanoid robot and desktop dual-arm robot, the RGMP framework achieves 87% task success in generalization tests and exhibits 5x greater data efficiency than the state-of-the-art model. This performance underscores its superior cross-domain generalization, enabled by geometric-semantic reasoning and recursive-Gaussion adaptation.
>
---
#### [replaced 002] NavDP: Learning Sim-to-Real Navigation Diffusion Policy with Privileged Information Guidance
- **分类: cs.RO**

- **简介: 提出NavDP，用扩散策略实现端到端仿真训练导航，零样本迁移到真实环境，利用特权信息提升安全性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2505.08712v3](https://arxiv.org/pdf/2505.08712v3)**

> **作者:** Wenzhe Cai; Jiaqi Peng; Yuqiang Yang; Yujian Zhang; Meng Wei; Hanqing Wang; Yilun Chen; Tai Wang; Jiangmiao Pang
>
> **备注:** Project Page: https://wzcai99.github.io/navigation-diffusion-policy.github.io/
>
> **摘要:** Learning to navigate in dynamic and complex open-world environments is a critical yet challenging capability for autonomous robots. Existing approaches often rely on cascaded modular frameworks, which require extensive hyperparameter tuning or learning from limited real-world demonstration data. In this paper, we propose Navigation Diffusion Policy (NavDP), an end-to-end network trained solely in simulation that enables zero-shot sim-to-real transfer across diverse environments and robot embodiments. The core of NavDP is a unified transformer-based architecture that jointly learns trajectory generation and trajectory evaluation, both conditioned solely on local RGB-D observation. By learning to predict critic values for contrastive trajectory samples, our proposed approach effectively leverages supervision from privileged information available in simulation, thereby fostering accurate spatial understanding and enabling the distinction between safe and dangerous behaviors. To support this, we develop an efficient data generation pipeline in simulation and construct a large-scale dataset encompassing over one million meters of navigation experience across 3,000 scenes. Empirical experiments in both simulated and real-world environments demonstrate that NavDP significantly outperforms prior state-of-the-art methods. Furthermore, we identify key factors influencing the generalization performance of NavDP. The dataset and code are publicly available at https://wzcai99.github.io/navigation-diffusion-policy.github.io.
>
---
#### [replaced 003] DAPPER: Discriminability-Aware Policy-to-Policy Preference-Based Reinforcement Learning for Query-Efficient Robot Skill Acquisition
- **分类: cs.RO**

- **简介: 论文提出DAPPER方法，通过多策略生成高区分度偏好查询，提升强化学习中人类偏好获取的查询效率，解决单策略轨迹多样性不足问题。**

- **链接: [https://arxiv.org/pdf/2505.06357v2](https://arxiv.org/pdf/2505.06357v2)**

> **作者:** Yuki Kadokawa; Jonas Frey; Takahiro Miki; Takamitsu Matsubara; Marco Hutter
>
> **备注:** Accepted for IEEE Robotics & Automation Magazine (RAM)
>
> **摘要:** Preference-based Reinforcement Learning (PbRL) enables policy learning through simple queries comparing trajectories from a single policy. While human responses to these queries make it possible to learn policies aligned with human preferences, PbRL suffers from low query efficiency, as policy bias limits trajectory diversity and reduces the number of discriminable queries available for learning preferences. This paper identifies preference discriminability, which quantifies how easily a human can judge which trajectory is closer to their ideal behavior, as a key metric for improving query efficiency. To address this, we move beyond comparisons within a single policy and instead generate queries by comparing trajectories from multiple policies, as training them from scratch promotes diversity without policy bias. We propose Discriminability-Aware Policy-to-Policy Preference-Based Efficient Reinforcement Learning (DAPPER), which integrates preference discriminability with trajectory diversification achieved by multiple policies. DAPPER trains new policies from scratch after each reward update and employs a discriminator that learns to estimate preference discriminability, enabling the prioritized sampling of more discriminable queries. During training, it jointly maximizes the preference reward and preference discriminability score, encouraging the discovery of highly rewarding and easily distinguishable policies. Experiments in simulated and real-world legged robot environments demonstrate that DAPPER outperforms previous methods in query efficiency, particularly under challenging preference discriminability conditions.
>
---
#### [replaced 004] Analyzing Key Objectives in Human-to-Robot Retargeting for Dexterous Manipulation
- **分类: cs.RO**

- **简介: 论文分析人手到机器人手运动重定向的关键优化目标，通过实验评估各因素重要性，旨在提升灵巧操作的重定向算法效果。**

- **链接: [https://arxiv.org/pdf/2506.09384v2](https://arxiv.org/pdf/2506.09384v2)**

> **作者:** Chendong Xin; Mingrui Yu; Yongpeng Jiang; Zhefeng Zhang; Xiang Li
>
> **备注:** v2: Extended the main text with additional analysis and implementation details
>
> **摘要:** Kinematic retargeting from human hands to robot hands is essential for transferring dexterity from humans to robots in manipulation teleoperation and imitation learning. However, due to mechanical differences between human and robot hands, completely reproducing human motions on robot hands is impossible. Existing works on retargeting incorporate various optimization objectives, focusing on different aspects of hand configuration. However, the lack of experimental comparative studies leaves the significance and effectiveness of these objectives unclear. This work aims to analyze these retargeting objectives for dexterous manipulation through extensive real-world comparative experiments. Specifically, we propose a comprehensive retargeting objective formulation that integrates intuitively crucial factors appearing in recent approaches. The significance of each factor is evaluated through experimental ablation studies on the full objective in kinematic posture retargeting and real-world teleoperated manipulation tasks. Experimental results and conclusions provide valuable insights for designing more accurate and effective retargeting algorithms for real-world dexterous manipulation.
>
---
#### [replaced 005] CoDrone: Autonomous Drone Navigation Assisted by Edge and Cloud Foundation Models
- **分类: cs.RO**

- **简介: 提出CoDrone框架，融合云边端与基础模型，解决无人机算力受限与高延迟问题，提升自主导航性能与环境适应力。**

- **链接: [https://arxiv.org/pdf/2512.19083v2](https://arxiv.org/pdf/2512.19083v2)**

> **作者:** Pengyu Chen; Tao Ouyang; Ke Luo; Weijie Hong; Xu Chen
>
> **备注:** This paper is accepted by the IEEE Internet of Things Journal (IoT-J) for publication in the Special Issue on "Augmented Edge Sensing Intelligence for Low-Altitude IoT Systems"
>
> **摘要:** Autonomous navigation for Unmanned Aerial Vehicles faces key challenges from limited onboard computational resources, which restrict deployed deep neural networks to shallow architectures incapable of handling complex environments. Offloading tasks to remote edge servers introduces high latency, creating an inherent trade-off in system design. To address these limitations, we propose CoDrone - the first cloud-edge-end collaborative computing framework integrating foundation models into autonomous UAV cruising scenarios - effectively leveraging foundation models to enhance performance of resource-constrained unmanned aerial vehicle platforms. To reduce onboard computation and data transmission overhead, CoDrone employs grayscale imagery for the navigation model. When enhanced environmental perception is required, CoDrone leverages the edge-assisted foundation model Depth Anything V2 for depth estimation and introduces a novel one-dimensional occupancy grid-based navigation method - enabling fine-grained scene understanding while advancing efficiency and representational simplicity of autonomous navigation. A key component of CoDrone is a Deep Reinforcement Learning-based neural scheduler that seamlessly integrates depth estimation with autonomous navigation decisions, enabling real-time adaptation to dynamic environments. Furthermore, the framework introduces a UAV-specific vision language interaction module incorporating domain-tailored low-level flight primitives to enable effective interaction between the cloud foundation model and the UAV. The introduction of VLM enhances open-set reasoning capabilities in complex unseen scenarios. Experimental results show CoDrone outperforms baseline methods under varying flight speeds and network conditions, achieving a 40% increase in average flight distance and a 5% improvement in average Quality of Navigation.
>
---
#### [replaced 006] Towards Autonomous Navigation in Endovascular Interventions
- **分类: cs.RO**

- **简介: 论文提出AI框架实现血管内导丝自主导航，解决现有系统低自主性、缺触觉反馈问题，开发高保真仿真平台、多模态控制网络及几何建模方法。**

- **链接: [https://arxiv.org/pdf/2512.18081v2](https://arxiv.org/pdf/2512.18081v2)**

> **作者:** Tudor Jianu
>
> **摘要:** Cardiovascular diseases remain the leading cause of global mortality, with minimally invasive treatment options offered through endovascular interventions. However, the precision and adaptability of current robotic systems for endovascular navigation are limited by heuristic control, low autonomy, and the absence of haptic feedback. This thesis presents an integrated AI-driven framework for autonomous guidewire navigation in complex vascular environments, addressing key challenges in data availability, simulation fidelity, and navigational accuracy. A high-fidelity, real-time simulation platform, CathSim, is introduced for reinforcement learning based catheter navigation, featuring anatomically accurate vascular models and contact dynamics. Building on CathSim, the Expert Navigation Network is developed, a policy that fuses visual, kinematic, and force feedback for autonomous tool control. To mitigate data scarcity, the open-source, bi-planar fluoroscopic dataset Guide3D is proposed, comprising more than 8,700 annotated images for 3D guidewire reconstruction. Finally, SplineFormer, a transformer-based model, is introduced to directly predict guidewire geometry as continuous B-spline parameters, enabling interpretable, real-time navigation. The findings show that combining high-fidelity simulation, multimodal sensory fusion, and geometric modelling substantially improves autonomous endovascular navigation and supports safer, more precise minimally invasive procedures.
>
---
#### [replaced 007] Nonholonomic Robot Parking by Feedback -- Part I: Modular Strict CLF Designs
- **分类: eess.SY; cs.RO; math.DS; math.OC**

- **简介: 提出模块化严格CLF框架，解决非完整机器人泊车的全局渐近稳定控制问题，支持角约束与最优/自适应扩展。**

- **链接: [https://arxiv.org/pdf/2511.15119v2](https://arxiv.org/pdf/2511.15119v2)**

> **作者:** Velimir Todorovski; Kwang Hak Kim; Alessandro Astolfi; Miroslav Krstic
>
> **备注:** arXiv admin note: text overlap with arXiv:2509.25575
>
> **摘要:** It has been known in the robotics literature since about 1995 that, in polar coordinates, the nonholonomic unicycle is asymptotically stabilizable by smooth feedback, even globally. We introduce a modular design framework that selects the forward velocity to decouple the radial coordinate, allowing the steering subsystem to be stabilized independently. Within this structure, we develop families of feedback laws using passivity, backstepping, and integrator forwarding. Each law is accompanied by a strict control Lyapunov function, including barrier variants that enforce angular constraints. These strict CLFs provide constructive class KL convergence estimates and enable eigenvalue assignment at the target equilibrium. The framework generalizes and extends prior modular and nonmodular approaches, while preparing the ground for inverse optimal and adaptive redesigns in the sequel paper.
>
---
#### [replaced 008] TacMan-Turbo: Proactive Tactile Control for Robust and Efficient Articulated Object Manipulation
- **分类: cs.RO**

- **简介: 提出TacMan-Turbo框架，通过主动利用触觉偏差预测最优操作，解决灵巧操作中效率与鲁棒性难以兼顾问题，无需预设运动学模型。**

- **链接: [https://arxiv.org/pdf/2508.02204v3](https://arxiv.org/pdf/2508.02204v3)**

> **作者:** Zihang Zhao; Zhenghao Qi; Yuyang Li; Leiyao Cui; Zhi Han; Lecheng Ruan; Yixin Zhu
>
> **摘要:** Adept manipulation of articulated objects is essential for robots to operate successfully in human environments. Such manipulation requires both effectiveness--reliable operation despite uncertain object structures--and efficiency--swift execution with minimal redundant steps and smooth actions. Existing approaches struggle to achieve both objectives simultaneously: methods relying on predefined kinematic models lack effectiveness when encountering structural variations, while tactile-informed approaches achieve robust manipulation without kinematic priors but compromise efficiency through reactive, step-by-step exploration-compensation cycles. This paper introduces TacMan-Turbo, a novel proactive tactile control framework for articulated object manipulation that mitigates this fundamental trade-off. Unlike previous approaches that treat tactile contact deviations merely as error signals requiring compensation, our method interprets these deviations as rich sources of local kinematic information. This new perspective enables our controller to predict optimal future interactions and make proactive adjustments, significantly enhancing manipulation efficiency. In comprehensive evaluations across 200 diverse simulated articulated objects and real-world experiments, our approach maintains a 100% success rate while significantly outperforming the previous tactile-informed method in time efficiency, action efficiency, and trajectory smoothness (all p-values < 0.0001). These results demonstrate that the long-standing trade-off between effectiveness and efficiency in articulated object manipulation can be successfully resolved without relying on prior kinematic knowledge.
>
---
#### [replaced 009] STARE-VLA: Progressive Stage-Aware Reinforcement for Fine-Tuning Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 提出STARE模块，分阶段优化VLA模型动作轨迹，结合IPI三阶段微调，提升机器人长程操作成功率至SOTA水平。**

- **链接: [https://arxiv.org/pdf/2512.05107v2](https://arxiv.org/pdf/2512.05107v2)**

> **作者:** Feng Xu; Guangyao Zhai; Xin Kong; Tingzhong Fu; Daniel F. N. Gordon; Xueli An; Benjamin Busam
>
> **摘要:** Recent advances in Vision-Language-Action (VLA) models, powered by large language models and reinforcement learning-based fine-tuning, have shown remarkable progress in robotic manipulation. Existing methods often treat long-horizon actions as linguistic sequences and apply trajectory-level optimization methods such as Trajectory-wise Preference Optimization (TPO) or Proximal Policy Optimization (PPO), leading to coarse credit assignment and unstable training. However, unlike language, where a unified semantic meaning is preserved despite flexible sentence order, action trajectories progress through causally chained stages with different learning difficulties. This motivates progressive stage optimization. Thereby, we present Stage-Aware Reinforcement (STARE), a module that decomposes a long-horizon action trajectory into semantically meaningful stages and provides dense, interpretable, and stage-aligned reinforcement signals. Integrating STARE into TPO and PPO, we yield Stage-Aware TPO (STA-TPO) and Stage-Aware PPO (STA-PPO) for offline stage-wise preference and online intra-stage interaction, respectively. Further building on supervised fine-tuning as initialization, we propose the Imitation -> Preference -> Interaction (IPI), a serial fine-tuning pipeline for improving action accuracy in VLA models. Experiments on SimplerEnv and ManiSkill3 demonstrate substantial gains, achieving state-of-the-art success rates of 98.0 percent on SimplerEnv and 96.4 percent on ManiSkill3 tasks.
>
---
#### [replaced 010] State-Conditional Adversarial Learning: An Off-Policy Visual Domain Transfer Method for End-to-End Imitation Learning
- **分类: cs.RO**

- **简介: 论文提出SCAL方法，解决视觉模仿学习中目标域无专家、离策略数据稀缺的跨域迁移问题，通过状态条件对抗对齐隐空间分布。**

- **链接: [https://arxiv.org/pdf/2512.05335v2](https://arxiv.org/pdf/2512.05335v2)**

> **作者:** Yuxiang Liu; Shengfan Cao
>
> **摘要:** We study visual domain transfer for end-to-end imitation learning in a realistic and challenging setting where target-domain data are strictly off-policy, expert-free, and scarce. We first provide a theoretical analysis showing that the target-domain imitation loss can be upper bounded by the source-domain loss plus a state-conditional latent KL divergence between source and target observation models. Guided by this result, we propose State- Conditional Adversarial Learning, an off-policy adversarial framework that aligns latent distributions conditioned on system state using a discriminator-based estimator of the conditional KL term. Experiments on visually diverse autonomous driving environments built on the BARC-CARLA simulator demonstrate that SCAL achieves robust transfer and strong sample efficiency.
>
---
