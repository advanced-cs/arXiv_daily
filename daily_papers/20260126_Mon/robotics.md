# 机器人 cs.RO

- **最新发布 18 篇**

- **更新 15 篇**

## 最新发布

#### [new 001] DMV-AVP: Distributed Multi-Vehicle Autonomous Valet Parking using Autoware
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文属于多车辆自动驾驶停车任务，解决分布式协同停车问题。提出DMV-AVP系统，实现多车协调泊车，提升系统可扩展性和自主控制能力。**

- **链接: [https://arxiv.org/pdf/2601.16327v1](https://arxiv.org/pdf/2601.16327v1)**

> **作者:** Zubair Islam; Mohamed El-Darieby
>
> **备注:** 7 pages, 5 figures, 1 table. Demo videos and source code available
>
> **摘要:** This paper presents the DMV-AVP System, a distributed simulation of Multi-Vehicle Autonomous Valet Parking (AVP). The system was implemented as an application of the Distributed Multi-Vehicle Architecture (DMAVA) for synchronized multi-host execution. Most existing simulation approaches rely on centralized or non-distributed designs that constrain scalability and limit fully autonomous control. This work introduces two modules built on top of the DMAVA: 1) a Multi-Vehicle AVP Node that performs state-based coordination, queuing, and reservation management across multiple vehicles, and 2) a Unity-Integrated YOLOv5 Parking Spot Detection Module that provides real-time, vision-based perception within AWSIM Labs. Both modules integrate seamlessly with the DMAVA and extend it specifically for multi-vehicle AVP operation, supported by a Zenoh-based communication layer that ensures low-latency topic synchronization and coordinated behavior across hosts. Experiments conducted on two- and three-host configurations demonstrate deterministic coordination, conflict-free parking behavior, and scalable performance across distributed Autoware instances. The results confirm that the proposed Distributed Multi-Vehicle AVP System supports cooperative AVP simulation and establishes a foundation for future real-world and hardware-in-the-loop validation. Demo videos and source code are available at https://github.com/zubxxr/multi-vehicle-avp
>
---
#### [new 002] DMAVA: Distributed Multi-Autonomous Vehicle Architecture Using Autoware
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文提出DMAVA架构，解决多自动驾驶车辆协同仿真难题。通过分布式设计实现多车同步控制，集成多种工具支持多AV并行运行。**

- **链接: [https://arxiv.org/pdf/2601.16336v1](https://arxiv.org/pdf/2601.16336v1)**

> **作者:** Zubair Islam; Mohamed El-Darieby
>
> **备注:** 9 pages, 4 figures, 5 tables, Submitted to IEEE IV 2026, Demo videos and source code available
>
> **摘要:** Simulating and validating coordination among multiple autonomous vehicles (AVs) is a challenging task as most existing simulation architectures are limited to single-vehicle operation or rely on centralized control. This paper presents a Distributed Multi-AV Architecture (DMAVA) that enables synchronized, real-time autonomous driving simulation across multiple physical hosts. Each vehicle runs its own complete AV stack and operates independently from other AVs. The vehicles in the simulation maintain synchronized coordination through a low-latency data-centric communication layer. The proposed system integrates ROS 2 Humble, Autoware Universe, AWSIM Labs, and Zenoh to support concurrent execution of multiple Autoware stacks within a shared Unity-based environment. Experiments conducted on multiple-host configurations demonstrate stable localization, reliable inter-host communication, and fully synchronized closed-loop control. The DMAVA also serves as a foundation for Multi-Vehicle Autonomous Valet Parking, demonstrating its extensibility toward higher-level cooperative autonomy. Demo videos and source code are available at: https://github.com/zubxxr/distributed-multi-autonomous-vehicle-architecture.
>
---
#### [new 003] A Multimodal Data Collection Framework for Dialogue-Driven Assistive Robotics to Clarify Ambiguities: A Wizard-of-Oz Pilot Study
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决对话驱动的助行机器人中模糊性识别问题。提出多模态数据收集框架，通过对话和实验获取自然交互数据，支持更智能的辅助控制。**

- **链接: [https://arxiv.org/pdf/2601.16870v1](https://arxiv.org/pdf/2601.16870v1)**

> **作者:** Guangping Liu; Nicholas Hawkins; Billy Madden; Tipu Sultan; Flavio Esposito; Madi Babaiasl
>
> **摘要:** Integrated control of wheelchairs and wheelchair-mounted robotic arms (WMRAs) has strong potential to increase independence for users with severe motor limitations, yet existing interfaces often lack the flexibility needed for intuitive assistive interaction. Although data-driven AI methods show promise, progress is limited by the lack of multimodal datasets that capture natural Human-Robot Interaction (HRI), particularly conversational ambiguity in dialogue-driven control. To address this gap, we propose a multimodal data collection framework that employs a dialogue-based interaction protocol and a two-room Wizard-of-Oz (WoZ) setup to simulate robot autonomy while eliciting natural user behavior. The framework records five synchronized modalities: RGB-D video, conversational audio, inertial measurement unit (IMU) signals, end-effector Cartesian pose, and whole-body joint states across five assistive tasks. Using this framework, we collected a pilot dataset of 53 trials from five participants and validated its quality through motion smoothness analysis and user feedback. The results show that the framework effectively captures diverse ambiguity types and supports natural dialogue-driven interaction, demonstrating its suitability for scaling to a larger dataset for learning, benchmarking, and evaluation of ambiguity-aware assistive control.
>
---
#### [new 004] Zero-Shot MARL Benchmark in the Cyber-Physical Mobility Lab
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于多智能体强化学习（MARL）的模拟到现实迁移任务，旨在评估CAV运动规划策略的泛化能力。通过构建包含仿真、数字孪生和物理测试平台的基准，分析了架构差异与环境真实度带来的性能下降问题。**

- **链接: [https://arxiv.org/pdf/2601.16578v1](https://arxiv.org/pdf/2601.16578v1)**

> **作者:** Julius Beerwerth; Jianye Xu; Simon Schäfer; Fynn Belderink; Bassam Alrifaee
>
> **摘要:** We present a reproducible benchmark for evaluating sim-to-real transfer of Multi-Agent Reinforcement Learning (MARL) policies for Connected and Automated Vehicles (CAVs). The platform, based on the Cyber-Physical Mobility Lab (CPM Lab) [1], integrates simulation, a high-fidelity digital twin, and a physical testbed, enabling structured zero-shot evaluation of MARL motion-planning policies. We demonstrate its use by deploying a SigmaRL-trained policy [2] across all three domains, revealing two complementary sources of performance degradation: architectural differences between simulation and hardware control stacks, and the sim-to-real gap induced by increasing environmental realism. The open-source setup enables systematic analysis of sim-to-real challenges in MARL under realistic, reproducible conditions.
>
---
#### [new 005] RENEW: Risk- and Energy-Aware Navigation in Dynamic Waterways
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出RENEW，用于动态水域中自主水面车辆的路径规划任务，解决外部干扰下的安全与能耗问题，通过风险与能量感知策略实现自适应导航。**

- **链接: [https://arxiv.org/pdf/2601.16424v1](https://arxiv.org/pdf/2601.16424v1)**

> **作者:** Mingi Jeong; Alberto Quattrini Li
>
> **备注:** 9 pages, 10 figure, 4 tables, AAAI 2026 (main track; oral acceptance)
>
> **摘要:** We present RENEW, a global path planner for Autonomous Surface Vehicle (ASV) in dynamic environments with external disturbances (e.g., water currents). RENEW introduces a unified risk- and energy-aware strategy that ensures safety by dynamically identifying non-navigable regions and enforcing adaptive safety constraints. Inspired by maritime contingency planning, it employs a best-effort strategy to maintain control under adverse conditions. The hierarchical architecture combines high-level constrained triangulation for topological diversity with low-level trajectory optimization within safe corridors. Validated with real-world ocean data, RENEW is the first framework to jointly address adaptive non-navigability and topological path diversity for robust maritime navigation.
>
---
#### [new 006] Adaptive Reinforcement and Model Predictive Control Switching for Safe Human-Robot Cooperative Navigation
- **分类: cs.RO**

- **简介: 该论文属于人机协作导航任务，解决安全导航与路径规划问题。提出ARMS框架，结合强化学习与模型预测控制，提升复杂环境下的导航成功率与效率。**

- **链接: [https://arxiv.org/pdf/2601.16686v1](https://arxiv.org/pdf/2601.16686v1)**

> **作者:** Ning Liu; Sen Shen; Zheng Li; Matthew D'Souza; Jen Jen Chung; Thomas Braunl
>
> **摘要:** This paper addresses the challenge of human-guided navigation for mobile collaborative robots under simultaneous proximity regulation and safety constraints. We introduce Adaptive Reinforcement and Model Predictive Control Switching (ARMS), a hybrid learning-control framework that integrates a reinforcement learning follower trained with Proximal Policy Optimization (PPO) and an analytical one-step Model Predictive Control (MPC) formulated as a quadratic program safety filter. To enable robust perception under partial observability and non-stationary human motion, ARMS employs a decoupled sensing architecture with a Long Short-Term Memory (LSTM) temporal encoder for the human-robot relative state and a spatial encoder for 360-degree LiDAR scans. The core contribution is a learned adaptive neural switcher that performs context-aware soft action fusion between the two controllers, favoring conservative, constraint-aware QP-based control in low-risk regions while progressively shifting control authority to the learned follower in highly cluttered or constrained scenarios where maneuverability is critical, and reverting to the follower action when the QP becomes infeasible. Extensive evaluations against Pure Pursuit, Dynamic Window Approach (DWA), and an RL-only baseline demonstrate that ARMS achieves an 82.5 percent success rate in highly cluttered environments, outperforming DWA and RL-only approaches by 7.1 percent and 3.1 percent, respectively, while reducing average computational latency by 33 percent to 5.2 milliseconds compared to a multi-step MPC baseline. Additional simulation transfer in Gazebo and initial real-world deployment results further indicate the practicality and robustness of ARMS for safe and efficient human-robot collaboration. Source code and a demonstration video are available at https://github.com/21ning/ARMS.git.
>
---
#### [new 007] A Unified Calibration Framework for High-Accuracy Articulated Robot Kinematics
- **分类: cs.RO**

- **简介: 该论文属于机器人标定任务，解决工业机器人定位误差问题。通过单一实验建立统一模型，整合几何与非几何误差，提升标定精度。**

- **链接: [https://arxiv.org/pdf/2601.16638v1](https://arxiv.org/pdf/2601.16638v1)**

> **作者:** Philip Tobuschat; Simon Duenser; Markus Bambach; Ivo Aschwanden
>
> **摘要:** Researchers have identified various sources of tool positioning errors for articulated industrial robots and have proposed dedicated compensation strategies. However, these typically require individual, specialized experiments with separate models and identification procedures. This article presents a unified approach to the static calibration of industrial robots that identifies a robot model, including geometric and non-geometric effects (compliant bending, thermal deformation, gear transmission errors), using only a single, straightforward experiment for data collection. The model augments the kinematic chain with virtual joints for each modeled effect and realizes the identification using Gauss-Newton optimization with analytic gradients. Fisher information spectra show that the estimation is well-conditioned and the parameterization near-minimal, whereas systematic temporal cross-validation and model ablations demonstrate robustness of the model identification. The resulting model is very accurate and its identification robust, achieving a mean position error of 26.8 $μm$ on a KUKA KR30 industrial robot compared to 102.3 $μm$ for purely geometric calibration.
>
---
#### [new 008] Sim-to-Real Transfer via a Style-Identified Cycle Consistent Generative Adversarial Network: Zero-Shot Deployment on Robotic Manipulators through Visual Domain Adaptation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人学习任务，旨在解决深度强化学习中的sim-to-real迁移问题。通过提出SICGAN模型，实现虚拟到真实的零样本迁移，提升工业机器人部署效率。**

- **链接: [https://arxiv.org/pdf/2601.16677v1](https://arxiv.org/pdf/2601.16677v1)**

> **作者:** Lucía Güitta-López; Lionel Güitta-López; Jaime Boal; Álvaro Jesús López-López
>
> **摘要:** The sample efficiency challenge in Deep Reinforcement Learning (DRL) compromises its industrial adoption due to the high cost and time demands of real-world training. Virtual environments offer a cost-effective alternative for training DRL agents, but the transfer of learned policies to real setups is hindered by the sim-to-real gap. Achieving zero-shot transfer, where agents perform directly in real environments without additional tuning, is particularly desirable for its efficiency and practical value. This work proposes a novel domain adaptation approach relying on a Style-Identified Cycle Consistent Generative Adversarial Network (StyleID-CycleGAN or SICGAN), an original Cycle Consistent Generative Adversarial Network (CycleGAN) based model. SICGAN translates raw virtual observations into real-synthetic images, creating a hybrid domain for training DRL agents that combines virtual dynamics with real-like visual inputs. Following virtual training, the agent can be directly deployed, bypassing the need for real-world training. The pipeline is validated with two distinct industrial robots in the approaching phase of a pick-and-place operation. In virtual environments agents achieve success rates of 90 to 100\%, and real-world deployment confirms robust zero-shot transfer (i.e., without additional training in the physical environment) with accuracies above 95\% for most workspace regions. We use augmented reality targets to improve the evaluation process efficiency, and experimentally demonstrate that the agent successfully generalizes to real objects of varying colors and shapes, including LEGO\textsuperscript{\textregistered}~cubes and a mug. These results establish the proposed pipeline as an efficient, scalable solution to the sim-to-real problem.
>
---
#### [new 009] Scalable Screw-Theoretic Synthesis for PDE-Based Dynamic Modeling of Multibody Flexible Manipulators
- **分类: cs.RO**

- **简介: 该论文属于多体柔性机械臂动力学建模任务，解决如何高效构建可扩展的PDE模型问题。工作包括建立基于螺旋理论的柔性链接动力学模型，并通过变分原理推导出统一的控制方程。**

- **链接: [https://arxiv.org/pdf/2601.16242v1](https://arxiv.org/pdf/2601.16242v1)**

> **作者:** S. Yaqubi; J. Mattila
>
> **摘要:** This paper presents a novel and scalable screw-theoretic multibody synthesis framework for PDE-based dynamic modeling of serial robotic manipulators with an arbitrary number of flexible links in three-dimensional space. The proposed approach systematically constructs screw-theoretic PDE models for individual flexible links and rigorously enforces holonomic joint constraints through interaction forces. The dynamics of each link are formulated using a set of dual screws expressed in body-fixed coordinates: one describing the motion of the body-fixed frame relative to the inertial frame, a second relating the body-fixed frame to the undeformed configuration, and a third capturing elastic deformations. By expressing the system energy and applying variational principles, the governing dynamics of each link had been previously derived in a unified manner. Synthesizing the individual link models yields an infinitely scalable multibody representation capable of capturing both local (subsystem-level) and global (system-level) dynamics. The framework explicitly recovers all dynamic states, including the motion of each body-fixed frame and the distributed deformation fields of the flexible links. For computational tractability and mathematical rigor, the resulting governing equations are formulated as a semi-explicit index-1 differential-algebraic system. Furthermore, by applying separation of variables, the PDE model is recast as an abstract Cauchy problem, and well-posedness of the resulting system is established.
>
---
#### [new 010] GNSS-based Lunar Orbit and Clock Estimation With Stochastic Cloning UD Filter
- **分类: cs.RO**

- **简介: 该论文属于月球导航任务，解决远距离下轨道与钟差估计问题。提出一种改进的滤波方法，结合多星座GNSS数据，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2601.16393v1](https://arxiv.org/pdf/2601.16393v1)**

> **作者:** Keidai Iiyama; Grace Gao
>
> **备注:** Submitted to the Journal of Guidance, Control, and Dynamics
>
> **摘要:** This paper presents a terrestrial GNSS-based orbit and clock estimation framework for lunar navigation satellites. To enable high-precision estimation under the low-observability conditions encountered at lunar distances, we develop a stochastic-cloning UD-factorized filter and delayed-state smoother that provide enhanced numerical stability when processing precise time-differenced carrier phase (TDCP) measurements. A comprehensive dynamics and measurement model is formulated, explicitly accounting for relativistic coupling between orbital and clock states, lunar time-scale transformations, and signal propagation delays including ionospheric, plasmaspheric, and Shapiro effects. The proposed approach is evaluated using high-fidelity Monte-Carlo simulations incorporating realistic multi-constellation GNSS geometry, broadcast ephemeris errors, lunar satellite dynamics, and ionospheric and plasmaspheric delay computed from empirical electron density models. Simulation results demonstrate that combining ionosphere-free pseudorange and TDCP measurements achieves meter-level orbit accuracy and sub-millimeter-per-second velocity accuracy, satisfying the stringent signal-in-space error requirements of future Lunar Augmented Navigation Services (LANS).
>
---
#### [new 011] Boosting Deep Reinforcement Learning with Semantic Knowledge for Robotic Manipulators
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，旨在解决DRL样本效率低的问题。通过融合语义知识图嵌入，提升学习效率，减少训练时间并提高任务准确性。**

- **链接: [https://arxiv.org/pdf/2601.16866v1](https://arxiv.org/pdf/2601.16866v1)**

> **作者:** Lucía Güitta-López; Vincenzo Suriani; Jaime Boal; Álvaro J. López-López; Daniele Nardi
>
> **摘要:** Deep Reinforcement Learning (DRL) is a powerful framework for solving complex sequential decision-making problems, particularly in robotic control. However, its practical deployment is often hindered by the substantial amount of experience required for learning, which results in high computational and time costs. In this work, we propose a novel integration of DRL with semantic knowledge in the form of Knowledge Graph Embeddings (KGEs), aiming to enhance learning efficiency by providing contextual information to the agent. Our architecture combines KGEs with visual observations, enabling the agent to exploit environmental knowledge during training. Experimental validation with robotic manipulators in environments featuring both fixed and randomized target attributes demonstrates that our method achieves up to {60}{\%} reduction in learning time and improves task accuracy by approximately 15 percentage points, without increasing training time or computational complexity. These results highlight the potential of semantic knowledge to reduce sample complexity and improve the effectiveness of DRL in robotic applications.
>
---
#### [new 012] Creating a biologically more accurate spider robot to study active vibration sensing
- **分类: cs.RO**

- **简介: 该论文属于仿生机器人研究，旨在解决蜘蛛腿部动作如何影响振动感知的问题。通过设计更生物逼真的八足机器人，改进腿部结构与运动方式，以更好地模拟蜘蛛的传感机制。**

- **链接: [https://arxiv.org/pdf/2601.16691v1](https://arxiv.org/pdf/2601.16691v1)**

> **作者:** Siyuan Sun; Eugene H. Lin; Nathan Brown; Hsin-Yi Hung; Andrew Gordus; Jochen Mueller; Chen Li
>
> **备注:** 8 pages, 12 figures
>
> **摘要:** Orb-weaving spiders detect prey on a web using vibration sensors at leg joints. They often dynamically crouch their legs during prey sensing, likely an active sensing strategy. However, how leg crouching enhances sensing is poorly understood, because measuring system vibrations in behaving animals is difficult. We use robophysical modeling to study this problem. Our previous spider robot had only four legs, simplified leg morphology, and a shallow crouching range of motion. Here, we developed a new spider robot, with eight legs, each with four joints that better approximated spider leg morphology. Leg exoskeletons were 3-D printed and joint stiffness was tuned using integrated silicone molding with variable materials and geometry. Tendon-driven actuation allowed a motor in the body to crouch all eight legs deeply as spiders do, while accelerometers at leg joints record leg vibrations. Experiments showed that our new spider robot reproduced key vibration features observed in the previous robot while improving biological accuracy. Our new robot provides a biologically more accurate robophysical model for studying how leg behaviors modulate vibration sensing on a web.
>
---
#### [new 013] Reinforcement Learning-Based Energy-Aware Coverage Path Planning for Precision Agriculture
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于农业机器人路径规划任务，解决能量约束下的覆盖路径问题。提出基于SAC的框架，结合CNN和LSTM，优化覆盖效率与能耗。**

- **链接: [https://arxiv.org/pdf/2601.16405v1](https://arxiv.org/pdf/2601.16405v1)**

> **作者:** Beining Wu; Zihao Ding; Leo Ostigaard; Jun Huang
>
> **备注:** Accepted by RACS '25: International Conference on Research in Adaptive and Convergent Systems, November 16-19, 2025, Ho Chi Minh, Vietnam. 10 pages, 5 figures
>
> **摘要:** Coverage Path Planning (CPP) is a fundamental capability for agricultural robots; however, existing solutions often overlook energy constraints, resulting in incomplete operations in large-scale or resource-limited environments. This paper proposes an energy-aware CPP framework grounded in Soft Actor-Critic (SAC) reinforcement learning, designed for grid-based environments with obstacles and charging stations. To enable robust and adaptive decision-making under energy limitations, the framework integrates Convolutional Neural Networks (CNNs) for spatial feature extraction and Long Short-Term Memory (LSTM) networks for temporal dynamics. A dedicated reward function is designed to jointly optimize coverage efficiency, energy consumption, and return-to-base constraints. Experimental results demonstrate that the proposed approach consistently achieves over 90% coverage while ensuring energy safety, outperforming traditional heuristic algorithms such as Rapidly-exploring Random Tree (RRT), Particle Swarm Optimization (PSO), and Ant Colony Optimization (ACO) baselines by 13.4-19.5% in coverage and reducing constraint violations by 59.9-88.3%. These findings validate the proposed SAC-based framework as an effective and scalable solution for energy-constrained CPP in agricultural robotics.
>
---
#### [new 014] A Feature Extraction Pipeline for Enhancing Lightweight Neural Networks in sEMG-based Joint Torque Estimation
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于肌电信号驱动的关节扭矩估计任务，旨在提升轻量神经网络的性能。通过设计特征提取管道，提高MLP在有限数据下的预测精度，实现与TCN相当的效果。**

- **链接: [https://arxiv.org/pdf/2601.16712v1](https://arxiv.org/pdf/2601.16712v1)**

> **作者:** Kartik Chari; Raid Dokhan; Anas Homsi; Niklas Kueper; Elsa Andrea Kirchner
>
> **摘要:** Robot-assisted rehabilitation offers an effective approach, wherein exoskeletons adapt to users' needs and provide personalized assistance. However, to deliver such assistance, accurate prediction of the user's joint torques is essential. In this work, we propose a feature extraction pipeline using 8-channel surface electromyography (sEMG) signals to predict elbow and shoulder joint torques. For preliminary evaluation, this pipeline was integrated into two neural network models: the Multilayer Perceptron (MLP) and the Temporal Convolutional Network (TCN). Data were collected from a single subject performing elbow and shoulder movements under three load conditions (0 kg, 1.10 kg, and 1.85 kg) using three motion-capture cameras. Reference torques were estimated from center-of-mass kinematics under the assumption of static equilibrium. Our offline analyses showed that, with our feature extraction pipeline, MLP model achieved mean RMSE of 0.963 N m, 1.403 N m, and 1.434 N m (over five seeds) for elbow, front-shoulder, and side-shoulder joints, respectively, which were comparable to the TCN performance. These results demonstrate that the proposed feature extraction pipeline enables a simple MLP to achieve performance comparable to that of a network designed explicitly for temporal dependencies. This finding is particularly relevant for applications with limited training data, a common scenario patient care.
>
---
#### [new 015] ReViP: Reducing False Completion in Vision-Language-Action Models with Vision-Proprioception Rebalance
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉-语言-动作模型任务，解决因模态不平衡导致的错误执行问题。提出ReViP框架，通过视觉-本体感知再平衡提升模型鲁棒性与准确性。**

- **链接: [https://arxiv.org/pdf/2601.16667v1](https://arxiv.org/pdf/2601.16667v1)**

> **作者:** Zhuohao Li; Yinghao Li; Jian-Jian Jiang; Lang Zhou; Tianyu Zhang; Wei-Shi Zheng
>
> **摘要:** Vision-Language-Action (VLA) models have advanced robotic manipulation by combining vision, language, and proprioception to predict actions. However, previous methods fuse proprioceptive signals directly with VLM-encoded vision-language features, resulting in state-dominant bias and false completions despite visible execution failures. We attribute this to modality imbalance, where policies over-rely on internal state while underusing visual evidence. To address this, we present ReViP, a novel VLA framework with Vision-Proprioception Rebalance to enhance visual grounding and robustness under perturbations. The key insight is to introduce auxiliary task-aware environment priors to adaptively modulate the coupling between semantic perception and proprioceptive dynamics. Specifically, we use an external VLM as a task-stage observer to extract real-time task-centric visual cues from visual observations, which drive a Vision-Proprioception Feature-wise Linear Modulation to enhance environmental awareness and reduce state-driven errors. Moreover, to evaluate false completion, we propose the first False-Completion Benchmark Suite built on LIBERO with controlled settings such as Object-Drop. Extensive experiments show that ReViP effectively reduces false-completion rates and improves success rates over strong VLA baselines on our suite, with gains extending to LIBERO, RoboTwin 2.0, and real-world evaluations.
>
---
#### [new 016] AnyView: Synthesizing Any Novel View in Dynamic Scenes
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出AnyView，解决动态场景中任意视角视频生成任务，通过扩散模型实现多视角一致性，无需几何假设。**

- **链接: [https://arxiv.org/pdf/2601.16982v1](https://arxiv.org/pdf/2601.16982v1)**

> **作者:** Basile Van Hoorick; Dian Chen; Shun Iwase; Pavel Tokmakov; Muhammad Zubair Irshad; Igor Vasiljevic; Swati Gupta; Fangzhou Cheng; Sergey Zakharov; Vitor Campagnolo Guizilini
>
> **备注:** Project webpage: https://tri-ml.github.io/AnyView/
>
> **摘要:** Modern generative video models excel at producing convincing, high-quality outputs, but struggle to maintain multi-view and spatiotemporal consistency in highly dynamic real-world environments. In this work, we introduce \textbf{AnyView}, a diffusion-based video generation framework for \emph{dynamic view synthesis} with minimal inductive biases or geometric assumptions. We leverage multiple data sources with various levels of supervision, including monocular (2D), multi-view static (3D) and multi-view dynamic (4D) datasets, to train a generalist spatiotemporal implicit representation capable of producing zero-shot novel videos from arbitrary camera locations and trajectories. We evaluate AnyView on standard benchmarks, showing competitive results with the current state of the art, and propose \textbf{AnyViewBench}, a challenging new benchmark tailored towards \emph{extreme} dynamic view synthesis in diverse real-world scenarios. In this more dramatic setting, we find that most baselines drastically degrade in performance, as they require significant overlap between viewpoints, while AnyView maintains the ability to produce realistic, plausible, and spatiotemporally consistent videos when prompted from \emph{any} viewpoint. Results, data, code, and models can be viewed at: https://tri-ml.github.io/AnyView/
>
---
#### [new 017] GPA-VGGT:Adapting VGGT to Large scale Localization by self-Supervised learning with Geometry and Physics Aware loss
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位任务，解决无标签数据下大场景定位问题。通过自监督学习和几何物理损失，提升VGGT模型的定位能力。**

- **链接: [https://arxiv.org/pdf/2601.16885v1](https://arxiv.org/pdf/2601.16885v1)**

> **作者:** Yangfan Xu; Lilian Zhang; Xiaofeng He; Pengdong Wu; Wenqi Wu; Jun Mao
>
> **摘要:** Transformer-based general visual geometry frameworks have shown promising performance in camera pose estimation and 3D scene understanding. Recent advancements in Visual Geometry Grounded Transformer (VGGT) models have shown great promise in camera pose estimation and 3D reconstruction. However, these models typically rely on ground truth labels for training, posing challenges when adapting to unlabeled and unseen scenes. In this paper, we propose a self-supervised framework to train VGGT with unlabeled data, thereby enhancing its localization capability in large-scale environments. To achieve this, we extend conventional pair-wise relations to sequence-wise geometric constraints for self-supervised learning. Specifically, in each sequence, we sample multiple source frames and geometrically project them onto different target frames, which improves temporal feature consistency. We formulate physical photometric consistency and geometric constraints as a joint optimization loss to circumvent the requirement for hard labels. By training the model with this proposed method, not only the local and global cross-view attention layers but also the camera and depth heads can effectively capture the underlying multi-view geometry. Experiments demonstrate that the model converges within hundreds of iterations and achieves significant improvements in large-scale localization. Our code will be released at https://github.com/X-yangfan/GPA-VGGT.
>
---
#### [new 018] An Efficient Insect-inspired Approach for Visual Point-goal Navigation
- **分类: cs.AI; cs.RO**

- **简介: 论文提出一种仿昆虫的视觉点目标导航方法，用于解决机器人导航任务。通过模拟昆虫大脑结构，实现高效路径学习与规划，提升导航效率并降低计算成本。**

- **链接: [https://arxiv.org/pdf/2601.16806v1](https://arxiv.org/pdf/2601.16806v1)**

> **作者:** Lu Yihe; Barbara Webb
>
> **摘要:** In this work we develop a novel insect-inspired agent for visual point-goal navigation. This combines abstracted models of two insect brain structures that have been implicated, respectively, in associative learning and path integration. We draw an analogy between the formal benchmark of the Habitat point-goal navigation task and the ability of insects to learn and refine visually guided paths around obstacles between a discovered food location and their nest. We demonstrate that the simple insect-inspired agent exhibits performance comparable to recent SOTA models at many orders of magnitude less computational cost. Testing in a more realistic simulated environment shows the approach is robust to perturbations.
>
---
## 更新

#### [replaced 001] DAVOS: An Autonomous Vehicle Operating System in the Vehicle Computing Era
- **分类: cs.OS; cs.RO**

- **简介: 该论文属于车载系统领域，旨在解决自动驾驶车辆操作系统在安全、可靠、高效和可扩展性方面的挑战。提出DAVOS系统，统一管理实时驾驶与数据服务。**

- **链接: [https://arxiv.org/pdf/2601.05072v3](https://arxiv.org/pdf/2601.05072v3)**

> **作者:** Yuxin Wang; Yuankai He; Boyang Tian; Lichen Xian; Weisong Shi
>
> **摘要:** Vehicle computing represents a fundamental shift in how autonomous vehicles are designed and deployed, transforming them from isolated transportation systems into mobile computing platforms that support both safety-critical, real-time driving and data-centric services. In this setting, vehicles simultaneously support real-time driving pipelines and a growing set of data-driven applications, placing increased responsibility on the vehicle operating system to coordinate computation, data movement, storage, and access. These demands highlight recurring system considerations related to predictable execution, data and execution protection, efficient handling of high-rate sensor data, and long-term system evolvability, commonly summarized as Safety, Security, Efficiency, and Extensibility (SSEE). Existing vehicle operating systems and runtimes address these concerns in isolation, resulting in fragmented software stacks that limit coordination between autonomy workloads and vehicle data services. This paper presents DAVOS, the Dependable Autonomous Vehicle Operating System, a unified vehicle operating system architecture designed for the vehicle computing context. DAVOS provides a cohesive operating system foundation that supports both real-time autonomy and extensible vehicle computing within a single system framework.
>
---
#### [replaced 002] CLASH: Collaborative Large-Small Hierarchical Framework for Continuous Vision-and-Language Navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决机器人根据自然语言指令导航复杂环境的问题。提出CLASH框架，结合大模型与小模型，提升导航性能与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.10360v2](https://arxiv.org/pdf/2512.10360v2)**

> **作者:** Liuyi Wang; Zongtao He; Jinlong Li; Ruihao Xia; Mengxian Hu; Chenpeng Yao; Chengju Liu; Yang Tang; Qijun Chen
>
> **摘要:** Vision-and-Language Navigation (VLN) requires robots to follow natural language instructions and navigate complex environments without prior maps. While recent vision-language large models demonstrate strong reasoning abilities, they often underperform task-specific panoramic small models in VLN tasks. To address this, we propose CLASH (Collaborative Large-Small Hierarchy), a VLN-CE framework that integrates a reactive small-model planner (RSMP) with a reflective large-model reasoner (RLMR). RSMP adopts a causal-learning-based dual-branch architecture to enhance generalization, while RLMR leverages panoramic visual prompting with chain-of-thought reasoning to support interpretable spatial understanding and navigation. We further introduce an uncertainty-aware collaboration mechanism (UCM) that adaptively fuses decisions from both models. For obstacle avoidance, in simulation, we replace the rule-based controller with a fully learnable point-goal policy, and in real-world deployment, we design a LiDAR-based clustering module for generating navigable waypoints and pair it with an online SLAM-based local controller. CLASH achieves state-of-the-art (SoTA) results (ranking 1-st) on the VLN-CE leaderboard, significantly improving SR and SPL on the test-unseen set over the previous SoTA methods. Real-world experiments demonstrate CLASH's strong robustness, validating its effectiveness in both simulation and deployment scenarios.
>
---
#### [replaced 003] VL-LN Bench: Towards Long-horizon Goal-oriented Navigation with Active Dialogs
- **分类: cs.RO**

- **简介: 该论文提出IIGN任务，解决真实场景下导航指令模糊的问题，通过主动对话增强导航模型，构建VL-LN基准进行评估与训练。**

- **链接: [https://arxiv.org/pdf/2512.22342v4](https://arxiv.org/pdf/2512.22342v4)**

> **作者:** Wensi Huang; Shaohao Zhu; Meng Wei; Jinming Xu; Xihui Liu; Hanqing Wang; Tai Wang; Feng Zhao; Jiangmiao Pang
>
> **摘要:** In most existing embodied navigation tasks, instructions are well-defined and unambiguous, such as instruction following and object searching. Under this idealized setting, agents are required solely to produce effective navigation outputs conditioned on vision and language inputs. However, real-world navigation instructions are often vague and ambiguous, requiring the agent to resolve uncertainty and infer user intent through active dialog. To address this gap, we propose Interactive Instance Goal Navigation (IIGN), a task that requires agents not only to generate navigation actions but also to produce language outputs via active dialog, thereby aligning more closely with practical settings. IIGN extends Instance Goal Navigation (IGN) by allowing agents to freely consult an oracle in natural language while navigating. Building on this task, we present the Vision Language-Language Navigation (VL-LN) benchmark, which provides a large-scale, automatically generated dataset and a comprehensive evaluation protocol for training and assessing dialog-enabled navigation models. VL-LN comprises over 41k long-horizon dialog-augmented trajectories for training and an automatic evaluation protocol with an oracle capable of responding to agent queries. Using this benchmark, we train a navigation model equipped with dialog capabilities and show that it achieves significant improvements over the baselines. Extensive experiments and analyses further demonstrate the effectiveness and reliability of VL-LN for advancing research on dialog-enabled embodied navigation. Code and dataset: https://0309hws.github.io/VL-LN.github.io/
>
---
#### [replaced 004] VALISENS: A Validated Innovative Multi-Sensor System for Cooperative Automated Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自动驾驶感知任务，旨在解决复杂环境下的感知可靠性问题。通过多传感器融合与V2X协作，提升车辆对行人等目标的识别能力，并实现传感器健康监测。**

- **链接: [https://arxiv.org/pdf/2505.06980v2](https://arxiv.org/pdf/2505.06980v2)**

> **作者:** Lei Wan; Prabesh Gupta; Andreas Eich; Marcel Kettelgerdes; Hannan Ejaz Keen; Michael Klöppel-Gersdorf; Alexey Vinel
>
> **备注:** 8 pages, 5 figures, submitted to IEEE VNC
>
> **摘要:** Reliable perception remains a key challenge for Connected Automated Vehicles (CAVs) in complex real-world environments, where varying lighting conditions and adverse weather degrade sensing performance. While existing multi-sensor solutions improve local robustness, they remain constrained by limited sensing range, line-of-sight occlusions, and sensor failures on individual vehicles. This paper introduces VALISENS, a validated cooperative perception system that extends multi-sensor fusion beyond a single vehicle through Vehicle-to-Everything (V2X)-enabled collaboration between Connected Automated Vehicles (CAVs) and intelligent infrastructure. VALISENS integrates onboard and roadside LiDARs, radars, RGB cameras, and thermal cameras within a unified multi-agent perception framework. Thermal cameras enhances the detection of Vulnerable Road Users (VRUs) under challenging lighting conditions, while roadside sensors reduce occlusions and expand the effective perception range. In addition, an integrated sensor monitoring module continuously assesses sensor health and detects anomalies before system degradation occurs. The proposed system is implemented and evaluated in a dedicated real-world testbed. Experimental results show that VALISENS improves pedestrian situational awareness by up to 18% compared with vehicle-only sensing, while the sensor monitoring module achieves over 97% accuracy, demonstrating its effectiveness and its potential to support future Cooperative Intelligent Transport Systems (C-ITS) applications.
>
---
#### [replaced 005] Digital twins for the design, interactive control, and deployment of modular, fiber-reinforced soft continuum arms
- **分类: cs.RO**

- **简介: 该论文属于软体机器人领域，解决SCA设计与控制难题。提出数字孪生框架，实现模块化软臂的仿真与控制，提升部署效率。**

- **链接: [https://arxiv.org/pdf/2507.10121v2](https://arxiv.org/pdf/2507.10121v2)**

> **作者:** Seung Hyun Kim; Jiamiao Guo; Arman Tekinalp; Heng-Sheng Chang; Ugur Akcal; Tixian Wang; Darren Biskup; Benjamin Walt; Girish Chowdhary; Girish Krishnan; Prashant G. Mehta; Mattia Gazzola
>
> **备注:** 8 pages, 4 figures This work has been submitted for possible publication
>
> **摘要:** Soft continuum arms (SCAs) promise versatile manipulation through mechanical compliance, for assistive devices, agriculture, search applications, or surgery. However, the strong nonlinear coupling between materials, morphology, and actuation renders design and control challenging, hindering real-world deployment. In this context, a modular fabrication strategy paired with reliable, interactive simulations would be highly beneficial, streamlining prototyping and control design. Here, we present a digital twin framework for modular SCAs realized using pneumatic Fiber-Reinforced Elastomeric Enclosures (FREEs). The approach models assemblies of FREE actuators through networks of Cosserat rods, favoring the accurate simulation of three-dimensional arm reconfigurations, while explicitly preserving internal modular architecture. This enables the quantitative analysis and scalable development of composite soft robot arms, overcoming limitations of current monolithic continuum models. To validate the framework, we introduce a three-dimensional reconstruction pipeline tailored to soft, slender, small-volume, and highly deformable structures, allowing reliable recovery of arm kinematics and strain distributions. Experimental results across multiple configurations and actuation regimes demonstrate close agreement with simulations. Finally, we embed the digital twins in a virtual environment to allow interactive control design and sim-to-real deployment, establishing a foundation for principled co-design and remote operation of modular soft continuum manipulators.
>
---
#### [replaced 006] Q-learning with Adjoint Matching
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **简介: 该论文提出QAM算法，解决连续动作强化学习中高效优化表达性策略的问题。通过邻接匹配技术，避免不稳定反向传播，提升策略性能。**

- **链接: [https://arxiv.org/pdf/2601.14234v2](https://arxiv.org/pdf/2601.14234v2)**

> **作者:** Qiyang Li; Sergey Levine
>
> **备注:** 32 pages, 8 figures, 7 tables
>
> **摘要:** We propose Q-learning with Adjoint Matching (QAM), a novel TD-based reinforcement learning (RL) algorithm that tackles a long-standing challenge in continuous-action RL: efficient optimization of an expressive diffusion or flow-matching policy with respect to a parameterized Q-function. Effective optimization requires exploiting the first-order information of the critic, but it is challenging to do so for flow or diffusion policies because direct gradient-based optimization via backpropagation through their multi-step denoising process is numerically unstable. Existing methods work around this either by only using the value and discarding the gradient information, or by relying on approximations that sacrifice policy expressivity or bias the learned policy. QAM sidesteps both of these challenges by leveraging adjoint matching, a recently proposed technique in generative modeling, which transforms the critic's action gradient to form a step-wise objective function that is free from unstable backpropagation, while providing an unbiased, expressive policy at the optimum. Combined with temporal-difference backup for critic learning, QAM consistently outperforms prior approaches on hard, sparse reward tasks in both offline and offline-to-online RL.
>
---
#### [replaced 007] XR$^3$: An Extended Reality Platform for Social-Physical Human-Robot Interaction
- **分类: cs.RO; cs.HC**

- **简介: 该论文提出XR$^3$平台，解决社交物理人机交互研究中成本高、缺乏触觉反馈的问题。通过双VR头显实现互动，支持精确的肢体与面部动作映射及触觉同步。**

- **链接: [https://arxiv.org/pdf/2601.12395v3](https://arxiv.org/pdf/2601.12395v3)**

> **作者:** Chao Wang; Anna Belardinelli; Michael Gienger
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** Social-physical human-robot interaction (spHRI) is difficult to study: building and programming robots that integrate multiple interaction modalities is costly and slow, while VR-based prototypes often lack physical contact, breaking users' visuo-tactile expectations. We present XR$^3$, a co-located dual-VR-headset platform for HRI research in which an attendee and a hidden operator share the same physical space while experiencing different virtual embodiments. The attendee sees an expressive virtual robot that interacts face-to-face in a shared virtual environment. In real time, the robot's upper-body motion, head and gaze behavior, and facial expressions are mapped from the operator's tracked limbs and face signals. Because the operator is co-present and calibrated in the same coordinate frame, the operator can also touch the attendee, enabling perceived robot touch synchronized with the robot's visible hands. Finger and hand motion is mapped to the robot avatar using inverse kinematics to support precise contact. Beyond motion retargeting, XR$^3$ supports social retargeting of multiple nonverbal cues that can be experimentally varied while keeping physical interaction constant. We detail the system design and calibration, and demonstrate the platform in a touch-based Wizard-of-Oz study, lowering the barrier to prototyping and evaluating embodied, contact-based robot behaviors.
>
---
#### [replaced 008] HumanDiffusion: A Vision-Based Diffusion Trajectory Planner with Human-Conditioned Goals for Search and Rescue UAV
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出HumanDiffusion，用于搜救无人机的人工智能轨迹规划任务，解决在动态环境中自主导航并接近人类的问题。通过图像生成轨迹，实现安全、高效的救援操作。**

- **链接: [https://arxiv.org/pdf/2601.14973v2](https://arxiv.org/pdf/2601.14973v2)**

> **作者:** Faryal Batool; Iana Zhura; Valerii Serpiva; Roohan Ahmed Khan; Ivan Valuev; Issatay Tokmurziyev; Dzmitry Tsetserukou
>
> **备注:** This paper has been accepted at HRI, Late Breaking Report, 2026
>
> **摘要:** Reliable human--robot collaboration in emergency scenarios requires autonomous systems that can detect humans, infer navigation goals, and operate safely in dynamic environments. This paper presents HumanDiffusion, a lightweight image-conditioned diffusion planner that generates human-aware navigation trajectories directly from RGB imagery. The system combines YOLO-11 based human detection with diffusion-driven trajectory generation, enabling a quadrotor to approach a target person and deliver medical assistance without relying on prior maps or computationally intensive planning pipelines. Trajectories are predicted in pixel space, ensuring smooth motion and a consistent safety margin around humans. We evaluate HumanDiffusion in simulation and real-world indoor mock-disaster scenarios. On a 300-sample test set, the model achieves a mean squared error of 0.02 in pixel-space trajectory reconstruction. Real-world experiments demonstrate an overall mission success rate of 80% across accident-response and search-and-locate tasks with partial occlusions. These results indicate that human-conditioned diffusion planning offers a practical and robust solution for human-aware UAV navigation in time-critical assistance settings.
>
---
#### [replaced 009] FoldNet: Learning Generalizable Closed-Loop Policy for Garment Folding via Keypoint-Driven Asset and Demonstration Synthesis
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对服装折叠任务，解决数据生成困难问题，提出合成数据集和关键点驱动的策略，提升机器人折叠成功率。**

- **链接: [https://arxiv.org/pdf/2505.09109v4](https://arxiv.org/pdf/2505.09109v4)**

> **作者:** Yuxing Chen; Bowen Xiao; He Wang
>
> **摘要:** Due to the deformability of garments, generating a large amount of high-quality data for robotic garment manipulation tasks is highly challenging. In this paper, we present a synthetic garment dataset that can be used for robotic garment folding. We begin by constructing geometric garment templates based on keypoints and applying generative models to generate realistic texture patterns. Leveraging these keypoint annotations, we generate folding demonstrations in simulation and train folding policies via closed-loop imitation learning. To improve robustness, we propose KG-DAgger, which uses a keypoint-based strategy to generate demonstration data for recovering from failures. KG-DAgger significantly improves the model performance, boosting the real-world success rate by 25\%. After training with 15K trajectories (about 2M image-action pairs), the model achieves a 75\% success rate in the real world. Experiments in both simulation and real-world settings validate the effectiveness of our proposed framework.
>
---
#### [replaced 010] HEIGHT: Heterogeneous Interaction Graph Transformer for Robot Navigation in Crowded and Constrained Environments
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人导航任务，解决拥挤环境中安全高效路径规划问题。提出HEIGHT模型，通过图结构和注意力机制捕捉异质交互，提升导航性能。**

- **链接: [https://arxiv.org/pdf/2411.12150v4](https://arxiv.org/pdf/2411.12150v4)**

> **作者:** Shuijing Liu; Haochen Xia; Fatemeh Cheraghi Pouria; Kaiwen Hong; Neeloy Chakraborty; Zichao Hu; Joydeep Biswas; Katherine Driggs-Campbell
>
> **备注:** Accepted to IEEE Transactions of Automation Science and Engineering (T-ASE)
>
> **摘要:** We study the problem of robot navigation in dense and interactive crowds with static constraints such as corridors and furniture. Previous methods fail to consider all types of spatial and temporal interactions among agents and obstacles, leading to unsafe and inefficient robot paths. In this article, we leverage a graph-based representation of crowded and constrained scenarios and propose a structured framework to learn robot navigation policies with deep reinforcement learning. We first split the representations of different inputs and propose a heterogeneous spatio-temporal graph to model distinct interactions among humans, robots, and obstacles. Based on the heterogeneous spatio-temporal graph, we propose HEIGHT, a novel navigation policy network architecture with different components to capture heterogeneous interactions through space and time. HEIGHT utilizes attention mechanisms to prioritize important interactions and a recurrent network to track changes in the dynamic scene over time, encouraging the robot to avoid collisions adaptively. Through extensive simulation and real-world experiments, we demonstrate that HEIGHT outperforms state-of-the-art baselines in terms of success, navigation time, and generalization to domain shifts in challenging navigation scenarios. More information is available at https://sites.google.com/view/crowdnav-height/home.
>
---
#### [replaced 011] MapAnything: Universal Feed-Forward Metric 3D Reconstruction
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文提出MapAnything，一种统一的Transformer模型，用于3D重建任务。解决多视角几何与相机参数联合估计问题，通过输入图像和几何信息直接回归出度量3D场景。**

- **链接: [https://arxiv.org/pdf/2509.13414v3](https://arxiv.org/pdf/2509.13414v3)**

> **作者:** Nikhil Keetha; Norman Müller; Johannes Schönberger; Lorenzo Porzi; Yuchen Zhang; Tobias Fischer; Arno Knapitsch; Duncan Zauss; Ethan Weber; Nelson Antunes; Jonathon Luiten; Manuel Lopez-Antequera; Samuel Rota Bulò; Christian Richardt; Deva Ramanan; Sebastian Scherer; Peter Kontschieder
>
> **备注:** 3DV 2026. Project Page: https://map-anything.github.io/
>
> **摘要:** We introduce MapAnything, a unified transformer-based feed-forward model that ingests one or more images along with optional geometric inputs such as camera intrinsics, poses, depth, or partial reconstructions, and then directly regresses the metric 3D scene geometry and cameras. MapAnything leverages a factored representation of multi-view scene geometry, i.e., a collection of depth maps, local ray maps, camera poses, and a metric scale factor that effectively upgrades local reconstructions into a globally consistent metric frame. Standardizing the supervision and training across diverse datasets, along with flexible input augmentation, enables MapAnything to address a broad range of 3D vision tasks in a single feed-forward pass, including uncalibrated structure-from-motion, calibrated multi-view stereo, monocular depth estimation, camera localization, depth completion, and more. We provide extensive experimental analyses and model ablations demonstrating that MapAnything outperforms or matches specialist feed-forward models while offering more efficient joint training behavior, thus paving the way toward a universal 3D reconstruction backbone.
>
---
#### [replaced 012] Tunable Passivity Control for Centralized Multiport Networked Systems
- **分类: cs.RO**

- **简介: 该论文属于控制理论任务，解决CMND系统的稳定性问题。提出一种集中式可调被动控制方法，优化能量分布以确保系统稳定性和性能。**

- **链接: [https://arxiv.org/pdf/2511.05026v2](https://arxiv.org/pdf/2511.05026v2)**

> **作者:** Xingyuan Zhou; Peter Paik; S. Farokh Atashzar
>
> **摘要:** Centralized Multiport Networked Dynamic (CMND) systems have emerged as a key architecture with applications in several complex network systems, such as multilateral telerobotics and multi-agent control. These systems consist of a hub node/subsystem connecting with multiple remote nodes/subsystems via a networked architecture. One challenge for this system is stability, which can be affected by non-ideal network artifacts. Conventional passivity-based approaches can stabilize the system under specialized applications like small-scale networked systems. However, those conventional passive stabilizers have several restrictions, such as distributing compensation across subsystems in a decentralized manner, limiting flexibility, and, at the same time, relying on the restrictive assumptions of node passivity. This paper synthesizes a centralized optimal passivity-based stabilization framework for CMND systems. It consists of a centralized passivity observer monitoring overall energy flow and an optimal passivity controller that distributes the just-needed dissipation among various nodes, guaranteeing strict passivity and, thus, L2 stability. The proposed data-driven model-free approach, i.e., Tunable Centralized Optimal Passivity Control (TCoPC), optimizes total performance based on the prescribed dissipation distribution strategy while ensuring stability. The controller can put high dissipation loads on some sub-networks while relaxing the dissipation on other nodes. Simulation results demonstrate the proposed frameworks performance in a complex task under different time-varying delay scenarios while relaxing the remote nodes minimum phase and passivity assumption, enhancing the scalability and generalizability.
>
---
#### [replaced 013] FantasyVLN: Unified Multimodal Chain-of-Thought Reasoning for Vision-Language Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决传统方法在推理与实时性上的不足。提出FantasyVLN框架，通过隐式推理实现高效导航。**

- **链接: [https://arxiv.org/pdf/2601.13976v2](https://arxiv.org/pdf/2601.13976v2)**

> **作者:** Jing Zuo; Lingzhou Mu; Fan Jiang; Chengcheng Ma; Mu Xu; Yonggang Qi
>
> **摘要:** Achieving human-level performance in Vision-and-Language Navigation (VLN) requires an embodied agent to jointly understand multimodal instructions and visual-spatial context while reasoning over long action sequences. Recent works, such as NavCoT and NavGPT-2, demonstrate the potential of Chain-of-Thought (CoT) reasoning for improving interpretability and long-horizon planning. Moreover, multimodal extensions like OctoNav-R1 and CoT-VLA further validate CoT as a promising pathway toward human-like navigation reasoning. However, existing approaches face critical drawbacks: purely textual CoTs lack spatial grounding and easily overfit to sparse annotated reasoning steps, while multimodal CoTs incur severe token inflation by generating imagined visual observations, making real-time navigation impractical. In this work, we propose FantasyVLN, a unified implicit reasoning framework that preserves the benefits of CoT reasoning without explicit token overhead. Specifically, imagined visual tokens are encoded into a compact latent space using a pretrained Visual AutoRegressor (VAR) during CoT reasoning training, and the model jointly learns from textual, visual, and multimodal CoT modes under a unified multi-CoT strategy. At inference, our model performs direct instruction-to-action mapping while still enjoying reasoning-aware representations. Extensive experiments on LH-VLN show that our approach achieves reasoning-aware yet real-time navigation, improving success rates and efficiency while reducing inference latency by an order of magnitude compared to explicit CoT methods.
>
---
#### [replaced 014] Collision-Free Humanoid Traversal in Cluttered Indoor Scenes
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决复杂室内场景中人形机器人避障问题。提出HumanoidPF表示法和混合场景生成方法，提升避障技能学习效果，并实现真实世界部署。**

- **链接: [https://arxiv.org/pdf/2601.16035v2](https://arxiv.org/pdf/2601.16035v2)**

> **作者:** Han Xue; Sikai Liang; Zhikai Zhang; Zicheng Zeng; Yun Liu; Yunrui Lian; Jilong Wang; Qingtao Liu; Xuesong Shi; Li Yi
>
> **摘要:** We study the problem of collision-free humanoid traversal in cluttered indoor scenes, such as hurdling over objects scattered on the floor, crouching under low-hanging obstacles, or squeezing through narrow passages. To achieve this goal, the humanoid needs to map its perception of surrounding obstacles with diverse spatial layouts and geometries to the corresponding traversal skills. However, the lack of an effective representation that captures humanoid-obstacle relationships during collision avoidance makes directly learning such mappings difficult. We therefore propose Humanoid Potential Field (HumanoidPF), which encodes these relationships as collision-free motion directions, significantly facilitating RL-based traversal skill learning. We also find that HumanoidPF exhibits a surprisingly negligible sim-to-real gap as a perceptual representation. To further enable generalizable traversal skills through diverse and challenging cluttered indoor scenes, we further propose a hybrid scene generation method, incorporating crops of realistic 3D indoor scenes and procedurally synthesized obstacles. We successfully transfer our policy to the real world and develop a teleoperation system where users could command the humanoid to traverse in cluttered indoor scenes with just a single click. Extensive experiments are conducted in both simulation and the real world to validate the effectiveness of our method. Demos and code can be found in our website: https://axian12138.github.io/CAT/.
>
---
#### [replaced 015] Where to Touch, How to Contact: Hierarchical RL-MPC Framework for Geometry-Aware Long-Horizon Dexterous Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决复杂接触环境下的精细操控问题。通过分层RL-MPC框架，结合几何与动力学规划，提升操控效果与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.10930v2](https://arxiv.org/pdf/2601.10930v2)**

> **作者:** Zhixian Xie; Yu Xiang; Michael Posa; Wanxin Jin
>
> **摘要:** A key challenge in contact-rich dexterous manipulation is the need to jointly reason over geometry, kinematic constraints, and intricate, nonsmooth contact dynamics. End-to-end visuomotor policies bypass this structure, but often require large amounts of data, transfer poorly from simulation to reality, and generalize weakly across tasks/embodiments. We address those limitations by leveraging a simple insight: dexterous manipulation is inherently hierarchical - at a high level, a robot decides where to touch (geometry) and move the object (kinematics); at a low level it determines how to realize that plan through contact dynamics. Building on this insight, we propose a hierarchical RL--MPC framework in which a high-level reinforcement learning (RL) policy predicts a contact intention, a novel object-centric interface that specifies (i) an object-surface contact location and (ii) a post-contact object-level subgoal pose. Conditioned on this contact intention, a low-level contact-implicit model predictive control (MPC) optimizes local contact modes and replans with contact dynamics to generate robot actions that robustly drive the object toward each subgoal. We evaluate the framework on non-prehensile tasks, including geometry-generalized pushing and object 3D reorientation. It achieves near-100% success with substantially reduced data (10x less than end-to-end baselines), highly robust performance, and zero-shot sim-to-real transfer.
>
---
