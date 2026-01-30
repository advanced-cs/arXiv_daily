# 机器人 cs.RO

- **最新发布 40 篇**

- **更新 18 篇**

## 最新发布

#### [new 001] Multi-Robot Decentralized Collaborative SLAM in Planetary Analogue Environments: Dataset, Challenges, and Lessons Learned
- **分类: cs.RO**

- **简介: 论文研究多机器人在未知环境下的协同定位与建图（C-SLAM）任务，解决通信受限和行星环境带来的挑战，通过实验分析并提供新数据集。**

- **链接: [https://arxiv.org/pdf/2601.21063v1](https://arxiv.org/pdf/2601.21063v1)**

> **作者:** Pierre-Yves Lajoie; Karthik Soma; Haechan Mark Bong; Alice Lemieux-Bourque; Rongge Zhang; Vivek Shankar Varadharajan; Giovanni Beltrame
>
> **摘要:** Decentralized collaborative simultaneous localization and mapping (C-SLAM) is essential to enable multirobot missions in unknown environments without relying on preexisting localization and communication infrastructure. This technology is anticipated to play a key role in the exploration of the Moon, Mars, and other planets. In this article, we share insights and lessons learned from C-SLAM experiments involving three robots operating on a Mars analogue terrain and communicating over an ad hoc network. We examine the impact of limited and intermittent communication on C-SLAM performance, as well as the unique localization challenges posed by planetary-like environments. Additionally, we introduce a novel dataset collected during our experiments, which includes real-time peer-to-peer inter-robot throughput and latency measurements. This dataset aims to support future research on communication-constrained, decentralized multirobot operations.
>
---
#### [new 002] ReactEMG Stroke: Healthy-to-Stroke Few-shot Adaptation for sEMG-Based Intent Detection
- **分类: cs.RO**

- **简介: 该论文属于sEMG意图检测任务，解决中风患者需长时间校准的问题。通过健康数据预训练模型并微调，提升检测准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.22090v1](https://arxiv.org/pdf/2601.22090v1)**

> **作者:** Runsheng Wang; Katelyn Lee; Xinyue Zhu; Lauren Winterbottom; Dawn M. Nilsen; Joel Stein; Matei Ciocarlie
>
> **摘要:** Surface electromyography (sEMG) is a promising control signal for assist-as-needed hand rehabilitation after stroke, but detecting intent from paretic muscles often requires lengthy, subject-specific calibration and remains brittle to variability. We propose a healthy-to-stroke adaptation pipeline that initializes an intent detector from a model pretrained on large-scale able-bodied sEMG, then fine-tunes it for each stroke participant using only a small amount of subject-specific data. Using a newly collected dataset from three individuals with chronic stroke, we compare adaptation strategies (head-only tuning, parameter-efficient LoRA adapters, and full end-to-end fine-tuning) and evaluate on held-out test sets that include realistic distribution shifts such as within-session drift, posture changes, and armband repositioning. Across conditions, healthy-pretrained adaptation consistently improves stroke intent detection relative to both zero-shot transfer and stroke-only training under the same data budget; the best adaptation methods improve average transition accuracy from 0.42 to 0.61 and raw accuracy from 0.69 to 0.78. These results suggest that transferring a reusable healthy-domain EMG representation can reduce calibration burden while improving robustness for real-time post-stroke intent detection.
>
---
#### [new 003] IROS: A Dual-Process Architecture for Real-Time VLM-Based Indoor Navigation
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于室内导航任务，解决机器人快速响应与语义理解的矛盾。提出IROS框架，结合VLM与轻量模块，提升导航准确性与实时性。**

- **链接: [https://arxiv.org/pdf/2601.21506v1](https://arxiv.org/pdf/2601.21506v1)**

> **作者:** Joonhee Lee; Hyunseung Shin; Jeonggil Ko
>
> **摘要:** Indoor mobile robot navigation requires fast responsiveness and robust semantic understanding, yet existing methods struggle to provide both. Classical geometric approaches such as SLAM offer reliable localization but depend on detailed maps and cannot interpret human-targeted cues (e.g., signs, room numbers) essential for indoor reasoning. Vision-Language-Action (VLA) models introduce semantic grounding but remain strictly reactive, basing decisions only on visible frames and failing to anticipate unseen intersections or reason about distant textual cues. Vision-Language Models (VLMs) provide richer contextual inference but suffer from high computational latency, making them unsuitable for real-time operation on embedded platforms. In this work, we present IROS, a real-time navigation framework that combines VLM-level contextual reasoning with the efficiency of lightweight perceptual modules on low-cost, on-device hardware. Inspired by Dual Process Theory, IROS separates fast reflexive decisions (System One) from slow deliberative reasoning (System Two), invoking the VLM only when necessary. Furthermore, by augmenting compact VLMs with spatial and textual cues, IROS delivers robust, human-like navigation with minimal latency. Across five real-world buildings, IROS improves decision accuracy and reduces latency by 66% compared to continuous VLM-based navigation.
>
---
#### [new 004] DSCD-Nav: Dual-Stance Cooperative Debate for Object Navigation
- **分类: cs.RO**

- **简介: 该论文属于室内导航任务，旨在解决机器人在未知环境中导航时的决策可靠性问题。提出DSCD-Nav机制，通过双立场协作辩论提升导航准确性与效率。**

- **链接: [https://arxiv.org/pdf/2601.21409v1](https://arxiv.org/pdf/2601.21409v1)**

> **作者:** Weitao An; Qi Liu; Chenghao Xu; Jiayi Chai; Xu Yang; Kun Wei; Cheng Deng
>
> **摘要:** Adaptive navigation in unfamiliar indoor environments is crucial for household service robots. Despite advances in zero-shot perception and reasoning from vision-language models, existing navigation systems still rely on single-pass scoring at the decision layer, leading to overconfident long-horizon errors and redundant exploration. To tackle these problems, we propose Dual-Stance Cooperative Debate Navigation (DSCD-Nav), a decision mechanism that replaces one-shot scoring with stance-based cross-checking and evidence-aware arbitration to improve action reliability under partial observability. Specifically, given the same observation and candidate action set, we explicitly construct two stances by conditioning the evaluation on diverse and complementary objectives: a Task-Scene Understanding (TSU) stance that prioritizes goal progress from scene-layout cues, and a Safety-Information Balancing (SIB) stance that emphasizes risk and information value. The stances conduct a cooperative debate and make policy by cross-checking their top candidates with cue-grounded arguments. Then, a Navigation Consensus Arbitration (NCA) agent is employed to consolidate both sides' reasons and evidence, optionally triggering lightweight micro-probing to verify uncertain choices, preserving NCA's primary intent while disambiguating. Experiments on HM3Dv1, HM3Dv2, and MP3D demonstrate consistent improvements in success and path efficiency while reducing exploration redundancy.
>
---
#### [new 005] CoFreeVLA: Collision-Free Dual-Arm Manipulation via Vision-Language-Action Model and Risk Estimation
- **分类: cs.RO**

- **简介: 该论文属于双臂操作任务，解决自碰撞安全问题。通过引入风险估计模块，提升VLA模型的碰撞规避能力，提高操作安全性与成功率。**

- **链接: [https://arxiv.org/pdf/2601.21712v1](https://arxiv.org/pdf/2601.21712v1)**

> **作者:** Xuanran Zhai; Binkai Ou; Yemin Wang; Hui Yi Leong; Qiaojun Yu; Ce Hao; Yaohua Liu
>
> **摘要:** Vision Language Action (VLA) models enable instruction following manipulation, yet dualarm deployment remains unsafe due to under modeled selfcollisions between arms and grasped objects. We introduce CoFreeVLA, which augments an endtoend VLA with a short horizon selfcollision risk estimator that predicts collision likelihood from proprioception, visual embeddings, and planned actions. The estimator gates risky commands, recovers to safe states via risk-guided adjustments, and shapes policy refinement for safer rollouts. It is pre-trained with model-based collision labels and posttrained on real robot rollouts for calibration. On five bimanual tasks with the PiPER robot arm, CoFreeVLA reduces selfcollisions and improves success rates versus RDT and APEX.
>
---
#### [new 006] GAZELOAD A Multimodal Eye-Tracking Dataset for Mental Workload in Industrial Human-Robot Collaboration
- **分类: cs.RO**

- **简介: 该论文提出GAZELOAD数据集，用于工业人机协作中的心理负荷估计。通过多模态数据融合，解决心理负荷评估问题，支持算法开发与环境影响研究。**

- **链接: [https://arxiv.org/pdf/2601.21829v1](https://arxiv.org/pdf/2601.21829v1)**

> **作者:** Bsher Karbouj; Baha Eddin Gaaloul; Jorg Kruger
>
> **摘要:** This article describes GAZELOAD, a multimodal dataset for mental workload estimation in industrial human-robot collaboration. The data were collected in a laboratory assembly testbed where 26 participants interacted with two collaborative robots (UR5 and Franka Emika Panda) while wearing Meta ARIA smart glasses. The dataset time-synchronizes eye-tracking signals (pupil diameter, fixations, saccades, eye gaze, gaze transition entropy, fixation dispersion index) with environmental real-time and continuous measurements (illuminance) and task and robot context (bench, task block, induced faults), under controlled manipulations of task difficulty and ambient conditions. For each participant and workload-graded task block, we provide CSV files with ocular metrics aggregated into 250 ms windows, environmental logs, and self-reported mental workload ratings on a 1-10 Likert scale, organized in participant-specific folders alongside documentation. These data can be used to develop and benchmark algorithms for mental workload estimation, feature extraction, and temporal modeling in realistic industrial HRC scenarios, and to investigate the influence of environmental factors such as lighting on eye-based workload markers.
>
---
#### [new 007] Flocking behavior for dynamic and complex swarm structures
- **分类: cs.RO**

- **简介: 该论文属于多无人机编队控制任务，旨在解决复杂结构和轨迹保持的问题。通过引入虚拟质心概念，提出一种 flocking 算法，实现动态控制无人机数量和编队。**

- **链接: [https://arxiv.org/pdf/2601.21772v1](https://arxiv.org/pdf/2601.21772v1)**

> **作者:** Carmen D. R. Pita-Romero; Pedro Arias-Perez; Miguel Fernandez-Cortizas; Rafael Perez-Segui; Pascual Campoy
>
> **摘要:** Maintaining the formation of complex structures with multiple UAVs and achieving complex trajectories remains a major challenge. This work presents an algorithm for implementing the flocking behavior of UAVs based on the concept of Virtual Centroid to easily develop a structure for the flock. The approach builds on the classical virtual-based behavior, providing a theoretical framework for incorporating enhancements to dynamically control both the number of agents and the formation of the structure. Simulation tests and real-world experiments were conducted, demonstrating its simplicity even with complex formations and complex trajectories.
>
---
#### [new 008] Training slow silicon neurons to control extremely fast robots with spiking reinforcement learning
- **分类: cs.RO; cs.AI; cs.ET**

- **简介: 该论文属于机器人控制任务，旨在解决高速环境下实时决策问题。通过脉冲神经网络与类脑硬件结合，实现快速、高效的学习与控制。**

- **链接: [https://arxiv.org/pdf/2601.21548v1](https://arxiv.org/pdf/2601.21548v1)**

> **作者:** Irene Ambrosini; Ingo Blakowski; Dmitrii Zendrikov; Cristiano Capone; Luna Gava; Giacomo Indiveri; Chiara De Luca; Chiara Bartolozzi
>
> **摘要:** Air hockey demands split-second decisions at high puck velocities, a challenge we address with a compact network of spiking neurons running on a mixed-signal analog/digital neuromorphic processor. By co-designing hardware and learning algorithms, we train the system to achieve successful puck interactions through reinforcement learning in a remarkably small number of trials. The network leverages fixed random connectivity to capture the task's temporal structure and adopts a local e-prop learning rule in the readout layer to exploit event-driven activity for fast and efficient learning. The result is real-time learning with a setup comprising a computer and the neuromorphic chip in-the-loop, enabling practical training of spiking neural networks for robotic autonomous systems. This work bridges neuroscience-inspired hardware with real-world robotic control, showing that brain-inspired approaches can tackle fast-paced interaction tasks while supporting always-on learning in intelligent machines.
>
---
#### [new 009] Multi-Modular MANTA-RAY: A Modular Soft Surface Platform for Distributed Multi-Object Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决多对象操控的可扩展性问题。提出多模块MANTA-RAY平台，通过减少执行器密度实现高效、协调的物体操控。**

- **链接: [https://arxiv.org/pdf/2601.21884v1](https://arxiv.org/pdf/2601.21884v1)**

> **作者:** Pratik Ingle; Jørn Lambertsen; Kasper Støy; Andres Faina
>
> **备注:** 8 pages
>
> **摘要:** Manipulation surfaces control objects by actively deforming their shape rather than directly grasping them. While dense actuator arrays can generate complex deformations, they also introduce high degrees of freedom (DOF), increasing system complexity and limiting scalability. The MANTA-RAY (Manipulation with Adaptive Non-rigid Textile Actuation with Reduced Actuation densitY) platform addresses these challenges by leveraging a soft, fabric-based surface with reduced actuator density to manipulate fragile and heterogeneous objects. Previous studies focused on single-module implementations supported by four actuators, whereas the feasibility and benefits of a scalable, multi-module configuration remain unexplored. In this work, we present a distributed, modular, and scalable variant of the MANTA-RAY platform that maintains manipulation performance with a reduced actuator density. The proposed multi-module MANTA-RAY platform and control strategy employs object passing between modules and a geometric transformation driven PID controller that directly maps tilt-angle control outputs to actuator commands, eliminating the need for extensive data-driven or black-box training. We evaluate system performance in simulation across surface configurations of varying modules (3x3 and 4x4) and validate its feasibility through experiments on a physical 2x2 hardware prototype. The system successfully manipulates objects with diverse geometries, masses, and textures including fragile items such as eggs and apples as well as enabling parallel manipulation. The results demonstrate that the multi-module MANTA-RAY improves scalability and enables coordinated manipulation of multiple objects across larger areas, highlighting its potential for practical, real-world applications.
>
---
#### [new 010] HPTune: Hierarchical Proactive Tuning for Collision-Free Model Predictive Control
- **分类: cs.RO**

- **简介: 该论文属于运动规划任务，解决MPC参数调优效率低的问题。通过引入HPTune框架，结合预测风险指标和闭环优化，提升碰撞避免能力。**

- **链接: [https://arxiv.org/pdf/2601.21346v1](https://arxiv.org/pdf/2601.21346v1)**

> **作者:** Wei Zuo; Chengyang Li; Yikun Wang; Bingyang Cheng; Zeyi Ren; Shuai Wang; Derrick Wing Kwan Ng; Yik-Chung Wu
>
> **备注:** Accepted by IEEE ICASSP 2026
>
> **摘要:** Parameter tuning is a powerful approach to enhance adaptability in model predictive control (MPC) motion planners. However, existing methods typically operate in a myopic fashion that only evaluates executed actions, leading to inefficient parameter updates due to the sparsity of failure events (e.g., obstacle nearness or collision). To cope with this issue, we propose to extend evaluation from executed to non-executed actions, yielding a hierarchical proactive tuning (HPTune) framework that combines both a fast-level tuning and a slow-level tuning. The fast one adopts risk indicators of predictive closing speed and predictive proximity distance, and the slow one leverages an extended evaluation loss for closed-loop backpropagation. Additionally, we integrate HPTune with the Doppler LiDAR that provides obstacle velocities apart from position-only measurements for enhanced motion predictions, thus facilitating the implementation of HPTune. Extensive experiments on high-fidelity simulator demonstrate that HPTune achieves efficient MPC tuning and outperforms various baseline schemes in complex environments. It is found that HPTune enables situation-tailored motion planning by formulating a safe, agile collision avoidance strategy.
>
---
#### [new 011] Disentangling perception and reasoning for improving data efficiency in learning cloth manipulation without demonstrations
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究机器人布料操作任务，旨在提升数据效率。通过分离感知与推理模块，减少模型规模和训练时间，并实现有效的仿真到现实迁移。**

- **链接: [https://arxiv.org/pdf/2601.21713v1](https://arxiv.org/pdf/2601.21713v1)**

> **作者:** Donatien Delehelle; Fei Chen; Darwin Caldwell
>
> **备注:** 6 pages, 4 figures,
>
> **摘要:** Cloth manipulation is a ubiquitous task in everyday life, but it remains an open challenge for robotics. The difficulties in developing cloth manipulation policies are attributed to the high-dimensional state space, complex dynamics, and high propensity to self-occlusion exhibited by fabrics. As analytical methods have not been able to provide robust and general manipulation policies, reinforcement learning (RL) is considered a promising approach to these problems. However, to address the large state space and complex dynamics, data-based methods usually rely on large models and long training times. The resulting computational cost significantly hampers the development and adoption of these methods. Additionally, due to the challenge of robust state estimation, garment manipulation policies often adopt an end-to-end learning approach with workspace images as input. While this approach enables a conceptually straightforward sim-to-real transfer via real-world fine-tuning, it also incurs a significant computational cost by training agents on a highly lossy representation of the environment state. This paper questions this common design choice by exploring an efficient and modular approach to RL for cloth manipulation. We show that, through careful design choices, model size and training time can be significantly reduced when learning in simulation. Furthermore, we demonstrate how the resulting simulation-trained model can be transferred to the real world. We evaluate our approach on the SoftGym benchmark and achieve significant performance improvements over available baselines on our task, while using a substantially smaller model.
>
---
#### [new 012] Quick Heuristic Validation of Edges in Dynamic Roadmap Graphs
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决动态环境下道路图更新问题。通过引入"红-绿-灰"范式，快速分类边的有效性，实现高效半懒惰更新。**

- **链接: [https://arxiv.org/pdf/2601.20968v1](https://arxiv.org/pdf/2601.20968v1)**

> **作者:** Yulie Arad; Stav Ashur; Nancy M. Amato
>
> **摘要:** In this paper we tackle the problem of adjusting roadmap graphs for robot motion planning to non-static environments. We introduce the "Red-Green-Gray" paradigm, a modification of the SPITE method, capable of classifying the validity status of nodes and edges using cheap heuristic checks, allowing fast semi-lazy roadmap updates. Given a roadmap, we use simple computational geometry methods to approximate the swept volumes of robots and perform lazy collision checks, and label a subset of the edges as invalid (red), valid (green), or unknown (gray). We present preliminary experimental results comparing our method to the well-established technique of Leven and Hutchinson, and showing increased accuracy as well as the ability to correctly label edges as invalid while maintaining comparable update runtimes.
>
---
#### [new 013] From Instruction to Event: Sound-Triggered Mobile Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究声音触发的移动操作任务，解决传统指令驱动方式限制自主性的问题。通过构建数据平台和基线模型，使智能体能主动感知并响应声音事件。**

- **链接: [https://arxiv.org/pdf/2601.21667v1](https://arxiv.org/pdf/2601.21667v1)**

> **作者:** Hao Ju; Shaofei Huang; Hongyu Li; Zihan Ding; Si Liu; Meng Wang; Zhedong Zheng
>
> **摘要:** Current mobile manipulation research predominantly follows an instruction-driven paradigm, where agents rely on predefined textual commands to execute tasks. However, this setting confines agents to a passive role, limiting their autonomy and ability to react to dynamic environmental events. To address these limitations, we introduce sound-triggered mobile manipulation, where agents must actively perceive and interact with sound-emitting objects without explicit action instructions. To support these tasks, we develop Habitat-Echo, a data platform that integrates acoustic rendering with physical interaction. We further propose a baseline comprising a high-level task planner and low-level policy models to complete these tasks. Extensive experiments show that the proposed baseline empowers agents to actively detect and respond to auditory events, eliminating the need for case-by-case instructions. Notably, in the challenging dual-source scenario, the agent successfully isolates the primary source from overlapping acoustic interference to execute the first interaction, and subsequently proceeds to manipulate the secondary object, verifying the robustness of the baseline.
>
---
#### [new 014] AIR-VLA: Vision-Language-Action Systems for Aerial Manipulation
- **分类: cs.RO**

- **简介: 该论文属于航空操作任务，旨在解决VLA模型在空中机械系统中的应用问题。提出AIR-VLA平台，构建仿真环境与数据集，评估模型性能。**

- **链接: [https://arxiv.org/pdf/2601.21602v1](https://arxiv.org/pdf/2601.21602v1)**

> **作者:** Jianli Sun; Bin Tian; Qiyao Zhang; Chengxiang Li; Zihan Song; Zhiyong Cui; Yisheng Lv; Yonglin Tian
>
> **摘要:** While Vision-Language-Action (VLA) models have achieved remarkable success in ground-based embodied intelligence, their application to Aerial Manipulation Systems (AMS) remains a largely unexplored frontier. The inherent characteristics of AMS, including floating-base dynamics, strong coupling between the UAV and the manipulator, and the multi-step, long-horizon nature of operational tasks, pose severe challenges to existing VLA paradigms designed for static or 2D mobile bases. To bridge this gap, we propose AIR-VLA, the first VLA benchmark specifically tailored for aerial manipulation. We construct a physics-based simulation environment and release a high-quality multimodal dataset comprising 3000 manually teleoperated demonstrations, covering base manipulation, object & spatial understanding, semantic reasoning, and long-horizon planning. Leveraging this platform, we systematically evaluate mainstream VLA models and state-of-the-art VLM models. Our experiments not only validate the feasibility of transferring VLA paradigms to aerial systems but also, through multi-dimensional metrics tailored to aerial tasks, reveal the capabilities and boundaries of current models regarding UAV mobility, manipulator control, and high-level planning. AIR-VLA establishes a standardized testbed and data foundation for future research in general-purpose aerial robotics. The resource of AIR-VLA will be available at https://anonymous.4open.science/r/AIR-VLA-dataset-B5CC/.
>
---
#### [new 015] DexTac: Learning Contact-aware Visuotactile Policies via Hand-by-hand Teaching
- **分类: cs.RO**

- **简介: 该论文提出DexTac框架，解决高接触任务中触觉信息不足的问题。通过人类示范获取多维触觉数据，提升机械手的精确操作能力。**

- **链接: [https://arxiv.org/pdf/2601.21474v1](https://arxiv.org/pdf/2601.21474v1)**

> **作者:** Xingyu Zhang; Chaofan Zhang; Boyue Zhang; Zhinan Peng; Shaowei Cui; Shuo Wang
>
> **摘要:** For contact-intensive tasks, the ability to generate policies that produce comprehensive tactile-aware motions is essential. However, existing data collection and skill learning systems for dexterous manipulation often suffer from low-dimensional tactile information. To address this limitation, we propose DexTac, a visuo-tactile manipulation learning framework based on kinesthetic teaching. DexTac captures multi-dimensional tactile data-including contact force distributions and spatial contact regions-directly from human demonstrations. By integrating these rich tactile modalities into a policy network, the resulting contact-aware agent enables a dexterous hand to autonomously select and maintain optimal contact regions during complex interactions. We evaluate our framework on a challenging unimanual injection task. Experimental results demonstrate that DexTac achieves a 91.67% success rate. Notably, in high-precision scenarios involving small-scale syringes, our approach outperforms force-only baselines by 31.67%. These results underscore that learning multi-dimensional tactile priors from human demonstrations is critical for achieving robust, human-like dexterous manipulation in contact-rich environments.
>
---
#### [new 016] PocketDP3: Efficient Pocket-Scale 3D Visuomotor Policy
- **分类: cs.RO**

- **简介: 该论文提出PocketDP3，解决3D视觉-动作策略的参数效率问题。通过轻量架构提升推理速度与实用性，适用于真实场景部署。**

- **链接: [https://arxiv.org/pdf/2601.22018v1](https://arxiv.org/pdf/2601.22018v1)**

> **作者:** Jinhao Zhang; Zhexuan Zhou; Huizhe Li; Yichen Lai; Wenlong Xia; Haoming Song; Youmin Gong; Jie Me
>
> **摘要:** Recently, 3D vision-based diffusion policies have shown strong capability in learning complex robotic manipulation skills. However, a common architectural mismatch exists in these models: a tiny yet efficient point-cloud encoder is often paired with a massive decoder. Given a compact scene representation, we argue that this may lead to substantial parameter waste in the decoder. Motivated by this observation, we propose PocketDP3, a pocket-scale 3D diffusion policy that replaces the heavy conditional U-Net decoder used in prior methods with a lightweight Diffusion Mixer (DiM) built on MLP-Mixer blocks. This architecture enables efficient fusion across temporal and channel dimensions, significantly reducing model size. Notably, without any additional consistency distillation techniques, our method supports two-step inference without sacrificing performance, improving practicality for real-time deployment. Across three simulation benchmarks--RoboTwin2.0, Adroit, and MetaWorld--PocketDP3 achieves state-of-the-art performance with fewer than 1% of the parameters of prior methods, while also accelerating inference. Real-world experiments further demonstrate the practicality and transferability of our method in real-world settings. Code will be released.
>
---
#### [new 017] Meta-ROS: A Next-Generation Middleware Architecture for Adaptive and Scalable Robotic Systems
- **分类: cs.RO; cs.MA; cs.OS; cs.SE**

- **简介: 该论文提出Meta-ROS，解决机器人系统中中间件复杂与不兼容问题，通过优化通信协议提升性能与跨平台支持。**

- **链接: [https://arxiv.org/pdf/2601.21011v1](https://arxiv.org/pdf/2601.21011v1)**

> **作者:** Anshul Ranjan; Anoosh Damodar; Neha Chougule; Dhruva S Nayak; Anantharaman P. N; Shylaja S S
>
> **备注:** Checkout the Python Library - https://pypi.org/project/metaros/ To be Submitted in ACM Transactions on Autonomous and Adaptive Systems (TAAS) Journal
>
> **摘要:** The field of robotics faces significant challenges related to the complexity and interoperability of existing middleware frameworks, like ROS2, which can be difficult for new developers to adopt. To address these issues, we propose Meta-ROS, a novel middleware solution designed to streamline robotics development by simplifying integration, enhancing performance, and ensuring cross-platform compatibility. Meta-ROS leverages modern communication protocols, such as Zenoh and ZeroMQ, to enable efficient and low-latency communication across diverse hardware platforms, while also supporting various data types like audio, images, and video. We evaluated Meta-ROS's performance through comprehensive testing, comparing it with existing middleware frameworks like ROS1 and ROS2. The results demonstrated that Meta-ROS outperforms ROS2, achieving up to 30% higher throughput, significantly reducing message latency, and optimizing resource usage. Additionally, its robust hardware support and developer-centric design facilitate seamless integration and ease of use, positioning Meta-ROS as an ideal solution for modern, real-time robotics AI applications.
>
---
#### [new 018] LLM-Driven Scenario-Aware Planning for Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属于自主驾驶任务，旨在解决混合规划框架在复杂交通中的效率与安全性问题。提出LAP方法，利用大语言模型实现场景感知与路径规划的自适应切换。**

- **链接: [https://arxiv.org/pdf/2601.21876v1](https://arxiv.org/pdf/2601.21876v1)**

> **作者:** He Li; Zhaowei Chen; Rui Gao; Guoliang Li; Qi Hao; Shuai Wang; Chengzhong Xu
>
> **摘要:** Hybrid planner switching framework (HPSF) for autonomous driving needs to reconcile high-speed driving efficiency with safe maneuvering in dense traffic. Existing HPSF methods often fail to make reliable mode transitions or sustain efficient driving in congested environments, owing to heuristic scene recognition and low-frequency control updates. To address the limitation, this paper proposes LAP, a large language model (LLM) driven, adaptive planning method, which switches between high-speed driving in low-complexity scenes and precise driving in high-complexity scenes, enabling high qualities of trajectory generation through confined gaps. This is achieved by leveraging LLM for scene understanding and integrating its inference into the joint optimization of mode configuration and motion planning. The joint optimization is solved using tree-search model predictive control and alternating minimization. We implement LAP by Python in Robot Operating System (ROS). High-fidelity simulation results show that the proposed LAP outperforms other benchmarks in terms of both driving time and success rate.
>
---
#### [new 019] Disturbance-Aware Flight Control of Robotic Gliding Blimp via Moving Mass Actuation
- **分类: cs.RO**

- **简介: 该论文属于飞行控制任务，旨在解决风扰对轻于空气飞行器的影响。通过结合移动质量作动与预测控制，提升飞行稳定性。**

- **链接: [https://arxiv.org/pdf/2601.21188v1](https://arxiv.org/pdf/2601.21188v1)**

> **作者:** Hao Cheng; Feitian Zhang
>
> **摘要:** Robotic blimps, as lighter-than-air (LTA) aerial systems, offer long endurance and inherently safe operation but remain highly susceptible to wind disturbances. Building on recent advances in moving mass actuation, this paper addresses the lack of disturbance-aware control frameworks for LTA platforms by explicitly modeling and compensating for wind-induced effects. A moving horizon estimator (MHE) infers real-time wind perturbations and provides these estimates to a model predictive controller (MPC), enabling robust trajectory and heading regulation under varying wind conditions. The proposed approach leverages a two-degree-of-freedom (2-DoF) moving-mass mechanism to generate both inertial and aerodynamic moments for attitude and heading control, thereby enhancing flight stability in disturbance-prone environments. Extensive flight experiments under headwind and crosswind conditions show that the integrated MHE-MPC framework significantly outperforms baseline PID control, demonstrating its effectiveness for disturbance-aware LTA flight.
>
---
#### [new 020] Nimbus: A Unified Embodied Synthetic Data Generation Framework
- **分类: cs.RO; cs.DC**

- **简介: 该论文提出Nimbus框架，解决合成数据生成效率低的问题，通过统一架构提升数据生成的吞吐量和稳定性，适用于多任务场景。**

- **链接: [https://arxiv.org/pdf/2601.21449v1](https://arxiv.org/pdf/2601.21449v1)**

> **作者:** Zeyu He; Yuchang Zhang; Yuanzhen Zhou; Miao Tao; Hengjie Li; Yang Tian; Jia Zeng; Tai Wang; Wenzhe Cai; Yilun Chen; Ning Gao; Jiangmiao Pang
>
> **摘要:** Scaling data volume and diversity is critical for generalizing embodied intelligence. While synthetic data generation offers a scalable alternative to expensive physical data acquisition, existing pipelines remain fragmented and task-specific. This isolation leads to significant engineering inefficiency and system instability, failing to support the sustained, high-throughput data generation required for foundation model training. To address these challenges, we present Nimbus, a unified synthetic data generation framework designed to integrate heterogeneous navigation and manipulation pipelines. Nimbus introduces a modular four-layer architecture featuring a decoupled execution model that separates trajectory planning, rendering, and storage into asynchronous stages. By implementing dynamic pipeline scheduling, global load balancing, distributed fault tolerance, and backend-specific rendering optimizations, the system maximizes resource utilization across CPU, GPU, and I/O resources. Our evaluation demonstrates that Nimbus achieves a 2-3X improvement in end-to-end throughput compared to unoptimized baselines and ensuring robust, long-term operation in large-scale distributed environments. This framework serves as the production backbone for the InternData suite, enabling seamless cross-domain data synthesis.
>
---
#### [new 021] Don't double it: Efficient Agent Prediction in Occlusions
- **分类: cs.RO**

- **简介: 该论文属于目标预测任务，解决遮挡下交通参与者重复预测问题。提出MatchInformer方法，通过匈牙利匹配和轨迹解耦提升预测准确性和效率。**

- **链接: [https://arxiv.org/pdf/2601.21504v1](https://arxiv.org/pdf/2601.21504v1)**

> **作者:** Anna Rothenhäusler; Markus Mazzola; Andreas Look; Raghu Rajan; Joschka Bödecker
>
> **摘要:** Occluded traffic agents pose a significant challenge for autonomous vehicles, as hidden pedestrians or vehicles can appear unexpectedly, yet this problem remains understudied. Existing learning-based methods, while capable of inferring the presence of hidden agents, often produce redundant occupancy predictions where a single agent is identified multiple times. This issue complicates downstream planning and increases computational load. To address this, we introduce MatchInformer, a novel transformer-based approach that builds on the state-of-the-art SceneInformer architecture. Our method improves upon prior work by integrating Hungarian Matching, a state-of-the-art object matching algorithm from object detection, into the training process to enforce a one-to-one correspondence between predictions and ground truth, thereby reducing redundancy. We further refine trajectory forecasts by decoupling an agent's heading from its motion, a strategy that improves the accuracy and interpretability of predicted paths. To better handle class imbalances, we propose using the Matthews Correlation Coefficient (MCC) to evaluate occupancy predictions. By considering all entries in the confusion matrix, MCC provides a robust measure even in sparse or imbalanced scenarios. Experiments on the Waymo Open Motion Dataset demonstrate that our approach improves reasoning about occluded regions and produces more accurate trajectory forecasts than prior methods.
>
---
#### [new 022] mjlab: A Lightweight Framework for GPU-Accelerated Robot Learning
- **分类: cs.RO**

- **简介: 该论文提出mjlab，一个轻量级的GPU加速机器人学习框架，解决传统环境配置复杂的问题。通过模块化设计和MuJoCo Warp实现高效仿真，支持快速开发与测试。**

- **链接: [https://arxiv.org/pdf/2601.22074v1](https://arxiv.org/pdf/2601.22074v1)**

> **作者:** Kevin Zakka; Qiayuan Liao; Brent Yi; Louis Le Lay; Koushil Sreenath; Pieter Abbeel
>
> **备注:** Code is available at https://github.com/mujocolab/mjlab
>
> **摘要:** We present mjlab, a lightweight, open-source framework for robot learning that combines GPU-accelerated simulation with composable environments and minimal setup friction. mjlab adopts the manager-based API introduced by Isaac Lab, where users compose modular building blocks for observations, rewards, and events, and pairs it with MuJoCo Warp for GPU-accelerated physics. The result is a framework installable with a single command, requiring minimal dependencies, and providing direct access to native MuJoCo data structures. mjlab ships with reference implementations of velocity tracking, motion imitation, and manipulation tasks.
>
---
#### [new 023] MoE-ACT: Improving Surgical Imitation Learning Policies through Supervised Mixture-of-Experts
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于手术机器人学习任务，解决数据少、环境复杂的问题。提出监督混合专家架构，提升模仿学习性能，增强泛化与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.21971v1](https://arxiv.org/pdf/2601.21971v1)**

> **作者:** Lorenzo Mazza; Ariel Rodriguez; Rayan Younis; Martin Lelis; Ortrun Hellig; Chenpan Li; Sebastian Bodenstedt; Martin Wagner; Stefanie Speidel
>
> **摘要:** Imitation learning has achieved remarkable success in robotic manipulation, yet its application to surgical robotics remains challenging due to data scarcity, constrained workspaces, and the need for an exceptional level of safety and predictability. We present a supervised Mixture-of-Experts (MoE) architecture designed for phase-structured surgical manipulation tasks, which can be added on top of any autonomous policy. Unlike prior surgical robot learning approaches that rely on multi-camera setups or thousands of demonstrations, we show that a lightweight action decoder policy like Action Chunking Transformer (ACT) can learn complex, long-horizon manipulation from less than 150 demonstrations using solely stereo endoscopic images, when equipped with our architecture. We evaluate our approach on the collaborative surgical task of bowel grasping and retraction, where a robot assistant interprets visual cues from a human surgeon, executes targeted grasping on deformable tissue, and performs sustained retraction. We benchmark our method against state-of-the-art Vision-Language-Action (VLA) models and the standard ACT baseline. Our results show that generalist VLAs fail to acquire the task entirely, even under standard in-distribution conditions. Furthermore, while standard ACT achieves moderate success in-distribution, adopting a supervised MoE architecture significantly boosts its performance, yielding higher success rates in-distribution and demonstrating superior robustness in out-of-distribution scenarios, including novel grasp locations, reduced illumination, and partial occlusions. Notably, it generalizes to unseen testing viewpoints and also transfers zero-shot to ex vivo porcine tissue without additional training, offering a promising pathway toward in vivo deployment. To support this, we present qualitative preliminary results of policy roll-outs during in vivo porcine surgery.
>
---
#### [new 024] InspecSafe-V1: A Multimodal Benchmark for Safety Assessment in Industrial Inspection Scenarios
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出InspecSafe-V1基准数据集，用于工业检测安全评估。解决真实场景下多模态感知与安全分析的问题，包含多种传感器数据和精细标注。**

- **链接: [https://arxiv.org/pdf/2601.21173v1](https://arxiv.org/pdf/2601.21173v1)**

> **作者:** Zeyi Liu; Shuang Liu; Jihai Min; Zhaoheng Zhang; Jun Cen; Pengyu Han; Songqiao Hu; Zihan Meng; Xiao He; Donghua Zhou
>
> **备注:** 15 pages, 7 figures
>
> **摘要:** With the rapid development of industrial intelligence and unmanned inspection, reliable perception and safety assessment for AI systems in complex and dynamic industrial sites has become a key bottleneck for deploying predictive maintenance and autonomous inspection. Most public datasets remain limited by simulated data sources, single-modality sensing, or the absence of fine-grained object-level annotations, which prevents robust scene understanding and multimodal safety reasoning for industrial foundation models. To address these limitations, InspecSafe-V1 is released as the first multimodal benchmark dataset for industrial inspection safety assessment that is collected from routine operations of real inspection robots in real-world environments. InspecSafe-V1 covers five representative industrial scenarios, including tunnels, power facilities, sintering equipment, oil and gas petrochemical plants, and coal conveyor trestles. The dataset is constructed from 41 wheeled and rail-mounted inspection robots operating at 2,239 valid inspection sites, yielding 5,013 inspection instances. For each instance, pixel-level segmentation annotations are provided for key objects in visible-spectrum images. In addition, a semantic scene description and a corresponding safety level label are provided according to practical inspection tasks. Seven synchronized sensing modalities are further included, including infrared video, audio, depth point clouds, radar point clouds, gas measurements, temperature, and humidity, to support multimodal anomaly recognition, cross-modal fusion, and comprehensive safety assessment in industrial environments.
>
---
#### [new 025] Singularity-Free Lie Group Integration and Geometrically Consistent Evaluation of Multibody System Models Described in Terms of Standard Absolute Coordinates
- **分类: cs.RO**

- **简介: 该论文属于多体系统仿真任务，解决绝对坐标下运动学奇异性和几何一致性问题，提出接口框架和几何集成方法。**

- **链接: [https://arxiv.org/pdf/2601.21413v1](https://arxiv.org/pdf/2601.21413v1)**

> **作者:** Andreas Mueller
>
> **备注:** 10 pages
>
> **摘要:** A classical approach to the multibody systems (MBS) modeling is to use absolute coordinates, i.e., a set of (possibly redundant) coordinates that describe the absolute position and orientation of the individual bodies with respect to an inertial frame (IFR). A well-known problem for the time integration of the equations of motion (EOM) is the lack of a singularity-free parameterization of spatial motions, which is usually tackled by using unit quaternions. Lie group integration methods were proposed as an alternative approach to the singularity-free time integration. At the same time, Lie group formulations of EOM naturally respect the geometry of spatial motions during integration. Lie group integration methods, operating directly on the configuration space Lie group, are incompatible with standard formulations of the EOM, and cannot be implemented in existing MBS simulation codes without a major restructuring. The contribution of this paper is twofold: (1) A framework for interfacing Lie group integrators to standard EOM formulations is presented. It allows describing MBS in terms of various absolute coordinates and at the same using Lie group integration schemes. (2) A method for consistently incorporating the geometry of rigid body motions into the evaluation of EOM in absolute coordinates integrated with standard vector space integration schemes. The direct product group and the semidirect product group SO(3)xR3 and the semidirect product group SE(3) are used for representing rigid body motions. The key element is the local-global transitions (LGT) transition map, which facilitates the update of (global) absolute coordinates in terms of the (local) coordinates on the Lie group. This LGT map is specific to the absolute coordinates, the local coordinates on the Lie group, and the Lie group used to represent rigid body configurations.
>
---
#### [new 026] 4D-CAAL: 4D Radar-Camera Calibration and Auto-Labeling for Autonomous Driving
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于多模态感知任务，旨在解决4D雷达与相机的标定及自动标注问题。提出4D-CAAL框架，设计双用途标定靶，实现精准标定与高效标注。**

- **链接: [https://arxiv.org/pdf/2601.21454v1](https://arxiv.org/pdf/2601.21454v1)**

> **作者:** Shanliang Yao; Zhuoxiao Li; Runwei Guan; Kebin Cao; Meng Xia; Fuping Hu; Sen Xu; Yong Yue; Xiaohui Zhu; Weiping Ding; Ryan Wen Liu
>
> **摘要:** 4D radar has emerged as a critical sensor for autonomous driving, primarily due to its enhanced capabilities in elevation measurement and higher resolution compared to traditional 3D radar. Effective integration of 4D radar with cameras requires accurate extrinsic calibration, and the development of radar-based perception algorithms demands large-scale annotated datasets. However, existing calibration methods often employ separate targets optimized for either visual or radar modalities, complicating correspondence establishment. Furthermore, manually labeling sparse radar data is labor-intensive and unreliable. To address these challenges, we propose 4D-CAAL, a unified framework for 4D radar-camera calibration and auto-labeling. Our approach introduces a novel dual-purpose calibration target design, integrating a checkerboard pattern on the front surface for camera detection and a corner reflector at the center of the back surface for radar detection. We develop a robust correspondence matching algorithm that aligns the checkerboard center with the strongest radar reflection point, enabling accurate extrinsic calibration. Subsequently, we present an auto-labeling pipeline that leverages the calibrated sensor relationship to transfer annotations from camera-based segmentations to radar point clouds through geometric projection and multi-feature optimization. Extensive experiments demonstrate that our method achieves high calibration accuracy while significantly reducing manual annotation effort, thereby accelerating the development of robust multi-modal perception systems for autonomous driving.
>
---
#### [new 027] Information Filtering via Variational Regularization for Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决视觉-运动策略中的噪声问题。通过引入变分正则化模块，提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.21926v1](https://arxiv.org/pdf/2601.21926v1)**

> **作者:** Jinhao Zhang; Wenlong Xia; Yaojia Wang; Zhexuan Zhou; Huizhe Li; Yichen Lai; Haoming Song; Youmin Gong; Jie Me
>
> **摘要:** Diffusion-based visuomotor policies built on 3D visual representations have achieved strong performance in learning complex robotic skills. However, most existing methods employ an oversized denoising decoder. While increasing model capacity can improve denoising, empirical evidence suggests that it also introduces redundancy and noise in intermediate feature blocks. Crucially, we find that randomly masking backbone features at inference time (without changing training) can improve performance, confirming the presence of task-irrelevant noise in intermediate features. To this end, we propose Variational Regularization (VR), a lightweight module that imposes a timestep-conditioned Gaussian over backbone features and applies a KL-divergence regularizer, forming an adaptive information bottleneck. Extensive experiments on three simulation benchmarks (RoboTwin2.0, Adroit, and MetaWorld) show that, compared to the baseline DP3, our approach improves the success rate by 6.1% on RoboTwin2.0 and by 4.1% on Adroit and MetaWorld, achieving new state-of-the-art results. Real-world experiments further demonstrate that our method performs well in practical deployments. Code will released.
>
---
#### [new 028] Towards Space-Based Environmentally-Adaptive Grasping
- **分类: cs.RO**

- **简介: 论文研究空间环境下的抓取任务，解决高维动作空间、稀疏奖励和泛化能力差的问题。通过融合多模态的潜在空间学习控制策略，实现高效且鲁棒的抓取。**

- **链接: [https://arxiv.org/pdf/2601.21394v1](https://arxiv.org/pdf/2601.21394v1)**

> **作者:** Leonidas Askianakis; Aleksandr Artemov
>
> **摘要:** Robotic manipulation in unstructured environments requires reliable execution under diverse conditions, yet many state-of-the-art systems still struggle with high-dimensional action spaces, sparse rewards, and slow generalization beyond carefully curated training scenarios. We study these limitations through the example of grasping in space environments. We learn control policies directly in a learned latent manifold that fuses (grammarizes) multiple modalities into a structured representation for policy decision-making. Building on GPU-accelerated physics simulation, we instantiate a set of single-shot manipulation tasks and achieve over 95% task success with Soft Actor-Critic (SAC)-based reinforcement learning in less than 1M environment steps, under continuously varying grasping conditions from step 1. This empirically shows faster convergence than representative state-of-the-art visual baselines under the same open-loop single-shot conditions. Our analysis indicates that explicitly reasoning in latent space yields more sample-efficient learning and improved robustness to novel object and gripper geometries, environmental clutter, and sensor configurations compared to standard baselines. We identify remaining limitations and outline directions toward fully adaptive and generalizable grasping in the extreme conditions of space.
>
---
#### [new 029] Spotlighting Task-Relevant Features: Object-Centric Representations for Better Generalization in Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在提升视觉表示的泛化能力。针对现有全局和密集特征混合无关信息的问题，提出基于槽的对象中心表示（SBOCR），有效减少噪声，提高任务执行效果。**

- **链接: [https://arxiv.org/pdf/2601.21416v1](https://arxiv.org/pdf/2601.21416v1)**

> **作者:** Alexandre Chapin; Bruno Machado; Emmanuel Dellandréa; Liming Chen
>
> **摘要:** The generalization capabilities of robotic manipulation policies are heavily influenced by the choice of visual representations. Existing approaches typically rely on representations extracted from pre-trained encoders, using two dominant types of features: global features, which summarize an entire image via a single pooled vector, and dense features, which preserve a patch-wise embedding from the final encoder layer. While widely used, both feature types mix task-relevant and irrelevant information, leading to poor generalization under distribution shifts, such as changes in lighting, textures, or the presence of distractors. In this work, we explore an intermediate structured alternative: Slot-Based Object-Centric Representations (SBOCR), which group dense features into a finite set of object-like entities. This representation permits to naturally reduce the noise provided to the robotic manipulation policy while keeping enough information to efficiently perform the task. We benchmark a range of global and dense representations against intermediate slot-based representations, across a suite of simulated and real-world manipulation tasks ranging from simple to complex. We evaluate their generalization under diverse visual conditions, including changes in lighting, texture, and the presence of distractors. Our findings reveal that SBOCR-based policies outperform dense and global representation-based policies in generalization settings, even without task-specific pretraining. These insights suggest that SBOCR is a promising direction for designing visual systems that generalize effectively in dynamic, real-world robotic environments.
>
---
#### [new 030] WheelArm-Sim: A Manipulation and Navigation Combined Multimodal Synthetic Data Generation Simulator for Unified Control in Assistive Robotics
- **分类: cs.RO**

- **简介: 该论文提出WheelArm-Sim，用于辅助机器人统一控制的数据生成模拟器，解决轮椅与机械臂集成控制问题，通过合成数据提升模型性能。**

- **链接: [https://arxiv.org/pdf/2601.21129v1](https://arxiv.org/pdf/2601.21129v1)**

> **作者:** Guangping Liu; Tipu Sultan; Vittorio Di Giorgio; Nick Hawkins; Flavio Esposito; Madi Babaiasl
>
> **备注:** Accepted to IEEE International Symposium on Medical Robotics (ISMR) 2026
>
> **摘要:** Wheelchairs and robotic arms enhance independent living by assisting individuals with upper-body and mobility limitations in their activities of daily living (ADLs). Although recent advancements in assistive robotics have focused on Wheelchair-Mounted Robotic Arms (WMRAs) and wheelchairs separately, integrated and unified control of the combination using machine learning models remains largely underexplored. To fill this gap, we introduce the concept of WheelArm, an integrated cyber-physical system (CPS) that combines wheelchair and robotic arm controls. Data collection is the first step toward developing WheelArm models. In this paper, we present WheelArm-Sim, a simulation framework developed in Isaac Sim for synthetic data collection. We evaluate its capability by collecting a manipulation and navigation combined multimodal dataset, comprising 13 tasks, 232 trajectories, and 67,783 samples. To demonstrate the potential of the WheelArm dataset, we implement a baseline model for action prediction in the mustard-picking task. The results illustrate that data collected from WheelArm-Sim is feasible for a data-driven machine learning model for integrated control.
>
---
#### [new 031] Macro-Scale Electrostatic Origami Motor
- **分类: cs.RO; physics.app-ph**

- **简介: 该论文属于机器人领域，解决宏尺度折叠旋转致动器缺失的问题。提出一种可折叠展开的静电折纸电机，实现连续旋转运动。**

- **链接: [https://arxiv.org/pdf/2601.21976v1](https://arxiv.org/pdf/2601.21976v1)**

> **作者:** Alex S. Miller; Leo McElroy; Jeffrey H. Lang
>
> **摘要:** Foldable robots have been an active area of robotics research due to their high volume-to-mass ratio, easy packability, and shape adaptability. For locomotion, previously developed foldable robots have either embedded linear actuators in, or attached non-folding rotary motors to, their structure. Further, those actuators directly embedded in the structure of the folding medium all contributed to linear or folding motion, not to continuous rotary motion. On the macro-scale there has not yet been a folding continuous rotary actuator. This paper details the development and testing of the first macro-scale origami rotary motor that can be folded flat, and then unfurled to operate. Using corona discharge for torque production, the prototype motor achieved an expansion ratio of 2.5:1, reached a top speed of 1440 rpm when driven at -29 kV, and exhibited a maximum output torque over 0.15 mN m with an active component torque density of 0.04 Nm/kg.
>
---
#### [new 032] Towards Bridging the Gap between Large-Scale Pretraining and Efficient Finetuning for Humanoid Control
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决人形机器人在预训练与微调间的效率问题。通过结合SAC算法与模型-based方法，提升训练效率与适应性。**

- **链接: [https://arxiv.org/pdf/2601.21363v1](https://arxiv.org/pdf/2601.21363v1)**

> **作者:** Weidong Huang; Zhehan Li; Hangxin Liu; Biao Hou; Yao Su; Jingwen Zhang
>
> **备注:** ICLR 2026
>
> **摘要:** Reinforcement learning (RL) is widely used for humanoid control, with on-policy methods such as Proximal Policy Optimization (PPO) enabling robust training via large-scale parallel simulation and, in some cases, zero-shot deployment to real robots. However, the low sample efficiency of on-policy algorithms limits safe adaptation to new environments. Although off-policy RL and model-based RL have shown improved sample efficiency, the gap between large-scale pretraining and efficient finetuning on humanoids still exists. In this paper, we find that off-policy Soft Actor-Critic (SAC), with large-batch update and a high Update-To-Data (UTD) ratio, reliably supports large-scale pretraining of humanoid locomotion policies, achieving zero-shot deployment on real robots. For adaptation, we demonstrate that these SAC-pretrained policies can be finetuned in new environments and out-of-distribution tasks using model-based methods. Data collection in the new environment executes a deterministic policy while stochastic exploration is instead confined to a physics-informed world model. This separation mitigates the risks of random exploration during adaptation while preserving exploratory coverage for improvement. Overall, the approach couples the wall-clock efficiency of large-scale simulation during pretraining with the sample efficiency of model-based learning during fine-tuning.
>
---
#### [new 033] Abstracting Robot Manipulation Skills via Mixture-of-Experts Diffusion Policies
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决多任务学习中模型规模大、成本高的问题。提出SMP方法，通过专家混合策略实现高效、可迁移的技能学习与应用。**

- **链接: [https://arxiv.org/pdf/2601.21251v1](https://arxiv.org/pdf/2601.21251v1)**

> **作者:** Ce Hao; Xuanran Zhai; Yaohua Liu; Harold Soh
>
> **摘要:** Diffusion-based policies have recently shown strong results in robot manipulation, but their extension to multi-task scenarios is hindered by the high cost of scaling model size and demonstrations. We introduce Skill Mixture-of-Experts Policy (SMP), a diffusion-based mixture-of-experts policy that learns a compact orthogonal skill basis and uses sticky routing to compose actions from a small, task-relevant subset of experts at each step. A variational training objective supports this design, and adaptive expert activation at inference yields fast sampling without oversized backbones. We validate SMP in simulation and on a real dual-arm platform with multi-task learning and transfer learning tasks, where SMP achieves higher success rates and markedly lower inference cost than large diffusion baselines. These results indicate a practical path toward scalable, transferable multi-task manipulation: learn reusable skills once, activate only what is needed, and adapt quickly when tasks change.
>
---
#### [new 034] Track-centric Iterative Learning for Global Trajectory Optimization in Autonomous Racing
- **分类: cs.RO**

- **简介: 该论文属于自主赛车轨迹优化任务，旨在解决不确定动力学下的全局轨迹优化问题。通过轨道中心的迭代学习方法，优化全周期轨迹以减少圈速。**

- **链接: [https://arxiv.org/pdf/2601.21027v1](https://arxiv.org/pdf/2601.21027v1)**

> **作者:** Youngim Nam; Jungbin Kim; Kyungtae Kang; Cheolhyeon Kwon
>
> **摘要:** This paper presents a global trajectory optimization framework for minimizing lap time in autonomous racing under uncertain vehicle dynamics. Optimizing the trajectory over the full racing horizon is computationally expensive, and tracking such a trajectory in the real world hardly assures global optimality due to uncertain dynamics. Yet, existing work mostly focuses on dynamics learning at the tracking level, without updating the trajectory itself to account for the learned dynamics. To address these challenges, we propose a track-centric approach that directly learns and optimizes the full-horizon trajectory. We first represent trajectories through a track-agnostic parametric space in light of the wavelet transform. This space is then efficiently explored using Bayesian optimization, where the lap time of each candidate is evaluated by running simulations with the learned dynamics. This optimization is embedded in an iterative learning framework, where the optimized trajectory is deployed to collect real-world data for updating the dynamics, progressively refining the trajectory over the iterations. The effectiveness of the proposed framework is validated through simulations and real-world experiments, demonstrating lap time improvement of up to 20.7% over a nominal baseline and consistently outperforming state-of-the-art methods.
>
---
#### [new 035] DynamicVLA: A Vision-Language-Action Model for Dynamic Object Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于动态物体操作任务，解决VLA模型在动态场景中的感知、预测和控制问题。提出DynamicVLA框架，结合时间推理与闭环适应，提升动态操作性能。**

- **链接: [https://arxiv.org/pdf/2601.22153v1](https://arxiv.org/pdf/2601.22153v1)**

> **作者:** Haozhe Xie; Beichen Wen; Jiarui Zheng; Zhaoxi Chen; Fangzhou Hong; Haiwen Diao; Ziwei Liu
>
> **备注:** Project Page: https://www.infinitescript.com/project/dynamic-vla/ GitHub: https://github.com/hzxie/DynamicVLA
>
> **摘要:** Manipulating dynamic objects remains an open challenge for Vision-Language-Action (VLA) models, which, despite strong generalization in static manipulation, struggle in dynamic scenarios requiring rapid perception, temporal anticipation, and continuous control. We present DynamicVLA, a framework for dynamic object manipulation that integrates temporal reasoning and closed-loop adaptation through three key designs: 1) a compact 0.4B VLA using a convolutional vision encoder for spatially efficient, structurally faithful encoding, enabling fast multimodal inference; 2) Continuous Inference, enabling overlapping reasoning and execution for lower latency and timely adaptation to object motion; and 3) Latent-aware Action Streaming, which bridges the perception-execution gap by enforcing temporally aligned action execution. To fill the missing foundation of dynamic manipulation data, we introduce the Dynamic Object Manipulation (DOM) benchmark, built from scratch with an auto data collection pipeline that efficiently gathers 200K synthetic episodes across 2.8K scenes and 206 objects, and enables fast collection of 2K real-world episodes without teleoperation. Extensive evaluations demonstrate remarkable improvements in response speed, perception, and generalization, positioning DynamicVLA as a unified framework for general dynamic object manipulation across embodiments.
>
---
#### [new 036] Deep QP Safety Filter: Model-free Learning for Reachability-based Safety Filter
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于安全控制任务，旨在解决黑箱动态系统中的安全问题。通过结合HJ可达性与无模型学习，提出Deep QP Safety Filter，实现安全过滤器的无模型学习。**

- **链接: [https://arxiv.org/pdf/2601.21297v1](https://arxiv.org/pdf/2601.21297v1)**

> **作者:** Byeongjun Kim; H. Jin Kim
>
> **备注:** Accepted at L4DC 2026
>
> **摘要:** We introduce Deep QP Safety Filter, a fully data-driven safety layer for black-box dynamical systems. Our method learns a Quadratic-Program (QP) safety filter without model knowledge by combining Hamilton-Jacobi (HJ) reachability with model-free learning. We construct contraction-based losses for both the safety value and its derivatives, and train two neural networks accordingly. In the exact setting, the learned critic converges to the viscosity solution (and its derivative), even for non-smooth values. Across diverse dynamical systems -- even including a hybrid system -- and multiple RL tasks, Deep QP Safety Filter substantially reduces pre-convergence failures while accelerating learning toward higher returns than strong baselines, offering a principled and practical route to safe, model-free control.
>
---
#### [new 037] EmboCoach-Bench: Benchmarking AI Agents on Developing Embodied Robots
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于 embodied AI 领域，旨在解决机器人自主策略开发问题。提出 EmboCoach-Bench 基准，评估 LLM 代理自主设计策略的能力。**

- **链接: [https://arxiv.org/pdf/2601.21570v1](https://arxiv.org/pdf/2601.21570v1)**

> **作者:** Zixing Lei; Genjia Liu; Yuanshuo Zhang; Qipeng Liu; Chuan Wen; Shanghang Zhang; Wenzhao Lian; Siheng Chen
>
> **备注:** 37 pages, 13 figures
>
> **摘要:** The field of Embodied AI is witnessing a rapid evolution toward general-purpose robotic systems, fueled by high-fidelity simulation and large-scale data collection. However, this scaling capability remains severely bottlenecked by a reliance on labor-intensive manual oversight from intricate reward shaping to hyperparameter tuning across heterogeneous backends. Inspired by LLMs' success in software automation and science discovery, we introduce \textsc{EmboCoach-Bench}, a benchmark evaluating the capacity of LLM agents to autonomously engineer embodied policies. Spanning 32 expert-curated RL and IL tasks, our framework posits executable code as the universal interface. We move beyond static generation to assess a dynamic closed-loop workflow, where agents leverage environment feedback to iteratively draft, debug, and optimize solutions, spanning improvements from physics-informed reward design to policy architectures such as diffusion policies. Extensive evaluations yield three critical insights: (1) autonomous agents can qualitatively surpass human-engineered baselines by 26.5\% in average success rate; (2) agentic workflow with environment feedback effectively strengthens policy development and substantially narrows the performance gap between open-source and proprietary models; and (3) agents exhibit self-correction capabilities for pathological engineering cases, successfully resurrecting task performance from near-total failures through iterative simulation-in-the-loop debugging. Ultimately, this work establishes a foundation for self-evolving embodied intelligence, accelerating the paradigm shift from labor-intensive manual tuning to scalable, autonomous engineering in embodied AI field.
>
---
#### [new 038] Causal World Modeling for Robot Control
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决长期操作和数据效率问题。提出LingBot-VA模型，结合视觉与动作信息，实现高效控制与泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.21998v1](https://arxiv.org/pdf/2601.21998v1)**

> **作者:** Lin Li; Qihang Zhang; Yiming Luo; Shuai Yang; Ruilin Wang; Fei Han; Mingrui Yu; Zelin Gao; Nan Xue; Xing Zhu; Yujun Shen; Yinghao Xu
>
> **备注:** Project page: https://technology.robbyant.com/lingbot-va Code: https://github.com/robbyant/lingbot-va
>
> **摘要:** This work highlights that video world modeling, alongside vision-language pre-training, establishes a fresh and independent foundation for robot learning. Intuitively, video world models provide the ability to imagine the near future by understanding the causality between actions and visual dynamics. Inspired by this, we introduce LingBot-VA, an autoregressive diffusion framework that learns frame prediction and policy execution simultaneously. Our model features three carefully crafted designs: (1) a shared latent space, integrating vision and action tokens, driven by a Mixture-of-Transformers (MoT) architecture, (2) a closed-loop rollout mechanism, allowing for ongoing acquisition of environmental feedback with ground-truth observations, (3) an asynchronous inference pipeline, parallelizing action prediction and motor execution to support efficient control. We evaluate our model on both simulation benchmarks and real-world scenarios, where it shows significant promise in long-horizon manipulation, data efficiency in post-training, and strong generalizability to novel configurations. The code and model are made publicly available to facilitate the community.
>
---
#### [new 039] AI-Augmented Density-Driven Optimal Control (D2OC) for Decentralized Environmental Mapping
- **分类: cs.MA; cs.RO**

- **简介: 该论文属于多智能体环境映射任务，解决有限传感与通信下的精准地图构建问题。提出AI增强的D2OC方法，通过自适应密度优化提升映射精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.21126v1](https://arxiv.org/pdf/2601.21126v1)**

> **作者:** Kooktae Lee; Julian Martinez
>
> **摘要:** This paper presents an AI-augmented decentralized framework for multi-agent (multi-robot) environmental mapping under limited sensing and communication. While conventional coverage formulations achieve effective spatial allocation when an accurate reference map is available, their performance deteriorates under uncertain or biased priors. The proposed method introduces an adaptive and self-correcting mechanism that enables agents to iteratively refine local density estimates within an optimal transport-based framework, ensuring theoretical consistency and scalability. A dual multilayer perceptron (MLP) module enhances adaptivity by inferring local mean-variance statistics and regulating virtual uncertainty for long-unvisited regions, mitigating stagnation around local minima. Theoretical analysis rigorously proves convergence under the Wasserstein metric, while simulation results demonstrate that the proposed AI-augmented Density-Driven Optimal Control consistently achieves robust and precise alignment with the ground-truth density, yielding substantially higher-fidelity reconstruction of complex multi-modal spatial distributions compared with conventional decentralized baselines.
>
---
#### [new 040] Generalized Information Gathering Under Dynamics Uncertainty
- **分类: cs.LG; cs.AI; cs.MA; cs.RO; eess.SY**

- **简介: 该论文属于强化学习中的信息获取任务，解决在动态不确定性下如何有效收集信息的问题。提出统一框架，推导通用信息获取成本，验证其理论基础并展示应用效果。**

- **链接: [https://arxiv.org/pdf/2601.21988v1](https://arxiv.org/pdf/2601.21988v1)**

> **作者:** Fernando Palafox; Jingqi Li; Jesse Milzman; David Fridovich-Keil
>
> **摘要:** An agent operating in an unknown dynamical system must learn its dynamics from observations. Active information gathering accelerates this learning, but existing methods derive bespoke costs for specific modeling choices: dynamics models, belief update procedures, observation models, and planners. We present a unifying framework that decouples these choices from the information-gathering cost by explicitly exposing the causal dependencies between parameters, beliefs, and controls. Using this framework, we derive a general information-gathering cost based on Massey's directed information that assumes only Markov dynamics with additive noise and is otherwise agnostic to modeling choices. We prove that the mutual information cost used in existing literature is a special case of our cost. Then, we leverage our framework to establish an explicit connection between the mutual information cost and information gain in linearized Bayesian estimation, thereby providing theoretical justification for mutual information-based active learning approaches. Finally, we illustrate the practical utility of our framework through experiments spanning linear, nonlinear, and multi-agent systems.
>
---
## 更新

#### [replaced 001] AsterNav: Autonomous Aerial Robot Navigation In Darkness Using Passive Computation
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决黑暗环境中微型无人机的导航问题。通过结合红外摄像头和结构光，实现无需外部设施的深度估计与导航。**

- **链接: [https://arxiv.org/pdf/2601.17550v2](https://arxiv.org/pdf/2601.17550v2)**

> **作者:** Deepak Singh; Shreyas Khobragade; Nitin J. Sanket
>
> **备注:** 8 pages, 10 figures, Published in IEEE Robotics And Automation Letters
>
> **摘要:** Autonomous aerial navigation in absolute darkness is crucial for post-disaster search and rescue operations, which often occur from disaster-zone power outages. Yet, due to resource constraints, tiny aerial robots, perfectly suited for these operations, are unable to navigate in the darkness to find survivors safely. In this paper, we present an autonomous aerial robot for navigation in the dark by combining an Infra-Red (IR) monocular camera with a large-aperture coded lens and structured light without external infrastructure like GPS or motion-capture. Our approach obtains depth-dependent defocus cues (each structured light point appears as a pattern that is depth dependent), which acts as a strong prior for our AsterNet deep depth estimation model. The model is trained in simulation by generating data using a simple optical model and transfers directly to the real world without any fine-tuning or retraining. AsterNet runs onboard the robot at 20 Hz on an NVIDIA Jetson Orin$^\text{TM}$ Nano. Furthermore, our network is robust to changes in the structured light pattern and relative placement of the pattern emitter and IR camera, leading to simplified and cost-effective construction. We successfully evaluate and demonstrate our proposed depth navigation approach AsterNav using depth from AsterNet in many real-world experiments using only onboard sensing and computation, including dark matte obstacles and thin ropes (diameter 6.25mm), achieving an overall success rate of 95.5% with unknown object shapes, locations and materials. To the best of our knowledge, this is the first work on monocular, structured-light-based quadrotor navigation in absolute darkness.
>
---
#### [replaced 002] Parallels Between VLA Model Post-Training and Human Motor Learning: Progress, Challenges, and Trends
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于VLA模型后训练研究，旨在提升模型在具体任务中的表现。通过分析人类运动学习，提出四类后训练方法，解决模型与实际应用不匹配的问题。**

- **链接: [https://arxiv.org/pdf/2506.20966v2](https://arxiv.org/pdf/2506.20966v2)**

> **作者:** Tian-Yu Xiang; Ao-Qun Jin; Xiao-Hu Zhou; Mei-Jiang Gui; Xiao-Liang Xie; Shi-Qi Liu; Shuang-Yi Wang; Sheng-Bin Duan; Fu-Chao Xie; Wen-Kai Wang; Si-Cheng Wang; Ling-Yun Li; Tian Tu; Zeng-Guang Hou
>
> **摘要:** Vision-language-action (VLA) models extend vision-language models (VLM) by integrating action generation modules for robotic manipulation. Leveraging the strengths of VLM in vision perception and instruction understanding, VLA models exhibit promising generalization across diverse manipulation tasks. However, applications demanding high precision and accuracy reveal performance gaps without further adaptation. Evidence from multiple domains highlights the critical role of post-training to align foundational models with downstream applications, spurring extensive research on post-training VLA models. VLA model post-training aims to enhance an embodiment's ability to interact with the environment for the specified tasks. This perspective aligns with Newell's constraints-led theory of skill acquisition, which posits that motor behavior arises from interactions among task, environmental, and organismic (embodiment) constraints. Accordingly, this survey structures post-training methods into four categories: (i) enhancing environmental perception, (ii) improving embodiment awareness, (iii) deepening task comprehension, and (iv) multi-component integration. Experimental results on standard benchmarks are synthesized to distill actionable guidelines. Finally, open challenges and emerging trends are outlined, relating insights from human learning to prospective methods for VLA post-training. This work delivers both a comprehensive overview of current VLA model post-training methods from a human motor learning perspective and practical insights for VLA model development. Project website: https://github.com/AoqunJin/Awesome-VLA-Post-Training.
>
---
#### [replaced 003] EROAM: Event-based Camera Rotational Odometry and Mapping in Real-time
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出EROAM，用于实时事件相机旋转定位与建图，解决高动态场景下的精度与效率问题。通过球面事件表示和优化框架提升性能。**

- **链接: [https://arxiv.org/pdf/2411.11004v2](https://arxiv.org/pdf/2411.11004v2)**

> **作者:** Wanli Xing; Shijie Lin; Linhan Yang; Zeqing Zhang; Yanjun Du; Maolin Lei; Yipeng Pan; Chen Wang; Jia Pan
>
> **备注:** Accepted by IEEE Transactions on Robotics (T-RO), 2026. Project page: https://wlxing1901.github.io/eroam/
>
> **摘要:** This paper presents EROAM, a novel event-based rotational odometry and mapping system that achieves real-time, accurate camera rotation estimation. Unlike existing approaches that rely on event generation models or contrast maximization, EROAM employs a spherical event representation by projecting events onto a unit sphere and introduces Event Spherical Iterative Closest Point (ES-ICP), a novel geometric optimization framework designed specifically for event camera data. The spherical representation simplifies rotational motion formulation while operating in a continuous spherical domain, enabling enhanced spatial resolution. Our system features an efficient map management approach using incremental k-d tree structures and intelligent regional density control, ensuring optimal computational performance during long-term operation. Combined with parallel point-to-line optimization, EROAM achieves efficient computation without compromising accuracy. Extensive experiments on both synthetic and real-world datasets show that EROAM significantly outperforms state-of-the-art methods in terms of accuracy, robustness, and computational efficiency. Our method maintains consistent performance under challenging conditions, including high angular velocities and extended sequences, where other methods often fail or show significant drift. Additionally, EROAM produces high-quality panoramic reconstructions with preserved fine structural details.
>
---
#### [replaced 004] Designing Effective Human-Swarm Interaction Interfaces: Insights from a User Study on Task Performance
- **分类: cs.HC; cs.RO**

- **简介: 论文研究人机群体交互界面设计，解决如何有效引导机器人集群完成任务的问题。通过用户实验评估界面性能，结果显示界面在多数情况下能成功引导机器人接近目标并减少损失。**

- **链接: [https://arxiv.org/pdf/2504.02250v3](https://arxiv.org/pdf/2504.02250v3)**

> **作者:** Wasura D. Wattearachchi; Erandi Lakshika; Kathryn Kasmarik; Michael Barlow
>
> **备注:** 8 pages, 4 figures, 5 tables
>
> **摘要:** In this paper, we present a systematic method of design for human-swarm interaction interfaces, combining theoretical insights with empirical evaluation. We first derived ten design principles from existing literature, applying them to key information dimensions identified through goal-directed task analysis and developed a tablet-based interface for a target search task. We then conducted a user study with 31 participants where humans were required to guide a robotic swarm to a target in the presence of three types of hazards that pose a risk to the robots: Distributed, Moving, and Spreading. Performance was measured based on the proximity of the robots to the target and the number of deactivated robots at the end of the task. Results indicate that at least one robot was brought closer to the target in 98% of tasks, demonstrating the interface's success in fulfilling the primary objective of the task. Additionally, in nearly 67% of tasks, more than 50% of the robots reached the target. Moreover, particularly better performance was noted in moving hazards. Additionally, the interface appeared to help minimise robot deactivation, as evidenced by nearly 94% of tasks where participants managed to keep more than 50% of the robots active, ensuring that most of the swarm remained operational. However, its effectiveness varied across hazards, with robot deactivation being lowest in distributed hazard scenarios, suggesting that the interface provided the most support in these conditions.
>
---
#### [replaced 005] FLARE: Agile Flights for Quadrotor Cable-Suspended Payload System via Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于无人机悬吊负载系统的敏捷飞行任务，解决其非线性、欠驱动和混合动力学带来的控制难题。通过强化学习方法FLARE，实现高效实时控制与真实环境验证。**

- **链接: [https://arxiv.org/pdf/2508.09797v2](https://arxiv.org/pdf/2508.09797v2)**

> **作者:** Dongcheng Cao; Jin Zhou; Xian Wang; Shuo Li
>
> **摘要:** Agile flight for the quadrotor cable-suspended payload system is a formidable challenge due to its underactuated, highly nonlinear, and hybrid dynamics. Traditional optimization-based methods often struggle with high computational costs and the complexities of cable mode transitions, limiting their real-time applicability and maneuverability exploitation. In this letter, we present FLARE, a reinforcement learning (RL) framework that directly learns agile navigation policy from high-fidelity simulation. Our method is validated across three designed challenging scenarios, notably outperforming a state-of-the-art optimization-based approach by a 3x speedup during gate traversal maneuvers. Furthermore, the learned policies achieve successful zero-shot sim-to-real transfer, demonstrating remarkable agility and safety in real-world experiments, running in real time on an onboard computer.
>
---
#### [replaced 006] Industrial Internet Robot Collaboration System and Edge Computing Optimization
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究工业场景下机器人路径规划与控制问题，通过优化算法和神经网络实现避障、导航和边缘计算，提升路径效率与系统响应速度。**

- **链接: [https://arxiv.org/pdf/2504.02492v2](https://arxiv.org/pdf/2504.02492v2)**

> **作者:** Haopeng Zhao; Dajun Tao; Tian Qi; Jingyuan Xu; Zijie Zhou; Lipeng Liu
>
> **摘要:** In industrial Internet environments, mobile robots must generate collision-free global routes under stochastic obstacle layouts and random perturbations in commanded linear and angular velocities. This paper models a differential-drive robot with nonholonomic constraints, then decomposes motion into obstacle avoidance, target turning, and target approaching behaviors to parameterize the control variables. Global path planning is formulated as a constrained optimization problem and converted into a weighted energy function that balances path length and collision penalties. A three-layer neural network represents the planning model, while simulated annealing searches for near-global minima and mitigates local traps. During execution, a fuzzy controller uses heading and lateral-offset errors to output wheel-speed differentials for rapid correction; edge-side computation is discussed to reduce robot-server traffic and latency. Matlab 2024 simulations report deviation within +-5 cm, convergence within 10 ms, and shorter paths than two baseline methods. The approach improves robustness of global navigation in practice.
>
---
#### [replaced 007] Designing Underactuated Graspers with Dynamically Variable Geometry Using Potential Energy Map Based Analysis
- **分类: cs.RO**

- **简介: 该论文属于机械臂抓取任务，旨在解决不同尺寸物体的稳定抓取问题。通过分析能量图，优化 tendon-pulley 参数，提升抓取适应性与稳定性。**

- **链接: [https://arxiv.org/pdf/2203.07456v3](https://arxiv.org/pdf/2203.07456v3)**

> **作者:** C. L. Yako; Shenli Yuan; J. Kenneth Salisbury
>
> **备注:** This is an updated version of my original paper (with the same title) published in IROS 2022. Parts of this work were refined or corrected in Chapter 3 of my dissertation, Good Vibrations: Toward Vibration-Based Robotic In-Hand Manipulation (DOI: 10.25740/mm182vq8220), and many of those changes have been incorporated here
>
> **摘要:** This paper introduces an extension to the energy map method for in-hand manipulation. Energy maps are used to predict how a part will evolve in the grasp given a specific actuation input to the gripper. Previous approaches assumed frictionless contacts, but we show analytically that friction can be included in the energy maps when using two-link underactuated fingers by understanding the evolution of the part-finger contact. These friction-based energy maps were used to evaluate the importance of various tendon-pulley gripper parameters across nearly 6 million simulated grasping scenarios. Specifically, a variable palm width is needed to manipulate parts of varying scales, and a variable transmission ratio, or the ratio of the distal to the proximal pulley radii, is needed to draw parts into a cage or to maintain a tip prehension grasp.
>
---
#### [replaced 008] Large-Scale Autonomous Gas Monitoring for Volcanic Environments: A Legged Robot on Mount Etna
- **分类: cs.RO**

- **简介: 该论文属于自主气体监测任务，旨在解决火山环境中近地表气体测量的危险与困难。研究使用腿式机器人ANYmal进行自主气体分析，成功实现了高自主率的气体源检测。**

- **链接: [https://arxiv.org/pdf/2601.07362v2](https://arxiv.org/pdf/2601.07362v2)**

> **作者:** Julia Richter; Turcan Tuna; Manthan Patel; Takahiro Miki; Devon Higgins; James Fox; Cesar Cadena; Andres Diaz; Marco Hutter
>
> **备注:** This work has been submitted to the IEEE for possible publication. Submitted to IEEE Robotics & Automation Magazine (RAM)
>
> **摘要:** Volcanic gas emissions are key precursors of eruptive activity. Yet, obtaining accurate near-surface measurements remains hazardous and logistically challenging, motivating the need for autonomous solutions. Limited mobility in rough volcanic terrain has prevented wheeled systems from performing reliable in situ gas measurements, reducing their usefulness as sensing platforms. We present a legged robotic system for autonomous volcanic gas analysis, utilizing the quadruped ANYmal, equipped with a quadrupole mass spectrometer system. Our modular autonomy stack integrates a mission planning interface, global planner, localization framework, and terrain-aware local navigation. We evaluated the system on Mount Etna across three autonomous missions in varied terrain, achieving successful gas-source detections with autonomy rates of 93-100%. In addition, we conducted a teleoperated mission in which the robot measured natural fumaroles, detecting sulfur dioxide and carbon dioxide. We discuss lessons learned from the gas-analysis and autonomy perspectives, emphasizing the need for adaptive sensing strategies, tighter integration of global and local planning, and improved hardware design.
>
---
#### [replaced 009] Listen, Look, Drive: Coupling Audio Instructions for User-aware VLA-based Autonomous Driving
- **分类: eess.AS; cs.MM; cs.RO**

- **简介: 该论文属于自主驾驶任务，旨在解决VLA模型无法实时接收用户意图的问题。通过融合音频指令与视觉信息，提出EchoVLA模型，提升驾驶决策的准确性和情感适应性。**

- **链接: [https://arxiv.org/pdf/2601.12142v3](https://arxiv.org/pdf/2601.12142v3)**

> **作者:** Ziang Guo; Feng Yang; Xuefeng Zhang; Jiaqi Guo; Kun Zhao; Yixiao Zhou; Peng Lu; Sifa Zheng; Zufeng Zhang
>
> **备注:** Accepted by IV
>
> **摘要:** Vision Language Action (VLA) models promise an open-vocabulary interface that can translate perceptual ambiguity into semantically grounded driving decisions, yet they still treat language as a static prior fixed at inference time. As a result, the model must infer continuously shifting objectives from pixels alone, yielding delayed or overly conservative maneuvers. We argue that effective VLAs for autonomous driving need an online channel in which users can influence driving with specific intentions. To this end, we present EchoVLA, a user-aware VLA that couples camera streams with in situ audio instructions. We augment the nuScenes dataset with temporally aligned, intent-specific speech commands generated by converting ego-motion descriptions into synthetic audios. Further, we compose emotional speech-trajectory pairs into a multimodal Chain-of-Thought (CoT) for fine-tuning a Multimodal Large Model (MLM) based on Qwen2.5-Omni. Specifically, we synthesize the audio-augmented dataset with different emotion types paired with corresponding driving behaviors, leveraging the emotional cues embedded in tone, pitch, and speech tempo to reflect varying user states, such as urgent or hesitant intentions, thus enabling our EchoVLA to interpret not only the semantic content but also the emotional context of audio commands for more nuanced and emotionally adaptive driving behavior. In open-loop benchmarks, our approach reduces the average L2 error by $59.4\%$ and the collision rate by $74.4\%$ compared to the baseline of vision-only perception. More experiments on nuScenes dataset validate that EchoVLA not only steers the trajectory through audio instructions, but also modulates driving behavior in response to the emotions detected in the user's speech.
>
---
#### [replaced 010] SKETCH: Semantic Key-Point Conditioning for Long-Horizon Vessel Trajectory Prediction
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于船舶轨迹预测任务，旨在解决长时 horizon 预测中的方向不一致问题。通过引入语义关键点条件，提升预测准确性与合理性。**

- **链接: [https://arxiv.org/pdf/2601.18537v2](https://arxiv.org/pdf/2601.18537v2)**

> **作者:** Linyong Gan; Zimo Li; Wenxin Xu; Xingjian Li; Jianhua Z. Huang; Enmei Tu; Shuhang Chen
>
> **摘要:** Accurate long-horizon vessel trajectory prediction remains challenging due to compounded uncertainty from complex navigation behaviors and environmental factors. Existing methods often struggle to maintain global directional consistency, leading to drifting or implausible trajectories when extrapolated over long time horizons. To address this issue, we propose a semantic-key-point-conditioned trajectory modeling framework, in which future trajectories are predicted by conditioning on a high-level Next Key Point (NKP) that captures navigational intent. This formulation decomposes long-horizon prediction into global semantic decision-making and local motion modeling, effectively restricting the support of future trajectories to semantically feasible subsets. To efficiently estimate the NKP prior from historical observations, we adopt a pretrain-finetune strategy. Extensive experiments on real-world AIS data demonstrate that the proposed method consistently outperforms state-of-the-art approaches, particularly for long travel durations, directional accuracy, and fine-grained trajectory prediction.
>
---
#### [replaced 011] Bring My Cup! Personalizing Vision-Language-Action Models with Visual Attentive Prompting
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究个性化指令理解与操作任务，解决机器人识别并操控特定个人物品的问题。提出VAP方法，通过视觉提示提升模型实例级控制能力。**

- **链接: [https://arxiv.org/pdf/2512.20014v2](https://arxiv.org/pdf/2512.20014v2)**

> **作者:** Sangoh Lee; Sangwoo Mo; Wook-Shin Han
>
> **备注:** Project page with videos and code: https://vap-project.github.io/
>
> **摘要:** While Vision-Language-Action (VLA) models generalize well to generic instructions, they struggle with personalized commands such as "bring my cup," where the robot must act on one specific instance among visually similar objects. We study this setting of manipulating personal objects, in which a VLA must identify and control a user-specific object unseen during training using only a few reference images. To address this challenge, we propose Visual Attentive Prompting (VAP), a simple-yet-effective training-free perceptual adapter that equips frozen VLAs with top-down selective attention. VAP treats the reference images as a non-parametric visual memory, grounds the personal object in the scene through open-vocabulary detection and embedding-based matching, and then injects this grounding as a visual prompt by highlighting the object and rewriting the instruction. We construct two simulation benchmarks, Personalized-SIMPLER and Personalized-VLABench, and a real-world tabletop benchmark to evaluate personalized manipulation across multiple robots and tasks. Experiments show that VAP consistently outperforms generic policies and token-learning baselines in both success rate and correct-object manipulation, helping to bridge the gap between semantic understanding and instance-level control.
>
---
#### [replaced 012] When Context Is Not Enough: Modeling Unexplained Variability in Car-Following Behavior
- **分类: stat.AP; cs.LG; cs.RO**

- **简介: 该论文属于交通仿真任务，旨在解决传统模型无法捕捉驾驶行为不确定性的难题。通过引入可解释的随机建模框架，结合深度学习与高斯过程，提升预测精度与不确定性量化能力。**

- **链接: [https://arxiv.org/pdf/2507.07012v2](https://arxiv.org/pdf/2507.07012v2)**

> **作者:** Chengyuan Zhang; Zhengbing He; Cathy Wu; Lijun Sun
>
> **备注:** Accepted to ISTTT26
>
> **摘要:** Modeling car-following behavior is fundamental to microscopic traffic simulation, yet traditional deterministic models often fail to capture the full extent of variability and unpredictability in human driving. While many modern approaches incorporate context-aware inputs (e.g., spacing, speed, relative speed), they frequently overlook structured stochasticity that arises from latent driver intentions, perception errors, and memory effects -- factors that are not directly observable from context alone. To fill the gap, this study introduces an interpretable stochastic modeling framework that captures not only context-dependent dynamics but also residual variability beyond what context can explain. Leveraging deep neural networks integrated with nonstationary Gaussian processes (GPs), our model employs a scenario-adaptive Gibbs kernel to learn dynamic temporal correlations in acceleration decisions, where the strength and duration of correlations between acceleration decisions evolve with the driving context. This formulation enables a principled, data-driven quantification of uncertainty in acceleration, speed, and spacing, grounded in both observable context and latent behavioral variability. Comprehensive experiments on the naturalistic vehicle trajectory dataset collected from the German highway, i.e., the HighD dataset, demonstrate that the proposed stochastic simulation method within this framework surpasses conventional methods in both predictive performance and interpretable uncertainty quantification. The integration of interpretability and accuracy makes this framework a promising tool for traffic analysis and safety-critical applications.
>
---
#### [replaced 013] Virtual Reflections on a Dynamic 2D Eye Model Improve Spatial Reference Identification
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决眼模型空间参考模糊问题。通过引入动态虚拟反射，提升空间参考识别准确性和用户体验。**

- **链接: [https://arxiv.org/pdf/2412.07344v4](https://arxiv.org/pdf/2412.07344v4)**

> **作者:** Matti Krüger; Yutaka Oshima; Yu Fang
>
> **备注:** This article has been accepted for publication in IEEE Transactions on Human-Machine Systems. Citation information: DOI 10.1109/THMS.2026.3651818
>
> **摘要:** The visible orientation of human eyes creates some transparency about people's spatial attention and other mental states. This leads to a dual role of the eyes as a means of sensing and communication. Accordingly, artificial eye models are being explored as communication media in human-machine interaction scenarios. One challenge in the use of eye models for communication consists of resolving spatial reference ambiguities, especially for screen-based models. To address this challenge, we introduce an approach that incorporates reflection-like features that are contingent on the movements of artificial eyes. We conducted a user study with 30 participants in which participants had to use spatial references provided by dynamic eye models to advance in a fast-paced group interaction task. Compared to a non-reflective eye model and a pure reflection mode, the superimposition of screen-based eyes with gaze-contingent virtual reflections resulted in a higher identification accuracy and user experience, suggesting a synergistic benefit.
>
---
#### [replaced 014] ALRM: Agentic LLM for Robotic Manipulation
- **分类: cs.RO; cs.CL**

- **简介: 该论文属于机器人操作任务，旨在解决LLM在机器人控制中模块化执行和多步骤推理评估不足的问题。提出ALRM框架，结合代码与工具生成策略，提升自然语言到机器人执行的可靠性。**

- **链接: [https://arxiv.org/pdf/2601.19510v2](https://arxiv.org/pdf/2601.19510v2)**

> **作者:** Vitor Gaboardi dos Santos; Ibrahim Khadraoui; Ibrahim Farhat; Hamza Yous; Samy Teffahi; Hakim Hacid
>
> **摘要:** Large Language Models (LLMs) have recently empowered agentic frameworks to exhibit advanced reasoning and planning capabilities. However, their integration in robotic control pipelines remains limited in two aspects: (1) prior \ac{llm}-based approaches often lack modular, agentic execution mechanisms, limiting their ability to plan, reflect on outcomes, and revise actions in a closed-loop manner; and (2) existing benchmarks for manipulation tasks focus on low-level control and do not systematically evaluate multistep reasoning and linguistic variation. In this paper, we propose Agentic LLM for Robot Manipulation (ALRM), an LLM-driven agentic framework for robotic manipulation. ALRM integrates policy generation with agentic execution through a ReAct-style reasoning loop, supporting two complementary modes: Code-asPolicy (CaP) for direct executable control code generation, and Tool-as-Policy (TaP) for iterative planning and tool-based action execution. To enable systematic evaluation, we also introduce a novel simulation benchmark comprising 56 tasks across multiple environments, capturing linguistically diverse instructions. Experiments with ten LLMs demonstrate that ALRM provides a scalable, interpretable, and modular approach for bridging natural language reasoning with reliable robotic execution. Results reveal Claude-4.1-Opus as the top closed-source model and Falcon-H1-7B as the top open-source model under CaP.
>
---
#### [replaced 015] Transport and Delivery of Objects with a Soft Everting Robot
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究软体自翻转机器人的物体运输任务，解决其在复杂环境中搬运重物的问题。通过实验分析和模型建立，验证了机器人可运输多种形状物体及较大重量，并通过狭窄空间与复杂地形。**

- **链接: [https://arxiv.org/pdf/2507.22188v2](https://arxiv.org/pdf/2507.22188v2)**

> **作者:** Ethan DeVries; Jack Ferlazzo; Mustafa Ugur; Laura H. Blumenschein
>
> **备注:** 8 pages, 11 figures, Published in IEEE Robotics and Automation Letters Link to publication: https://ieeexplore.ieee.org/abstract/document/11359009 Citation: E. M. DeVries, J. Ferlazzo, M. Ugur and L. H. Blumenschein, "Transport and Delivery of Objects With a Soft Everting Robot," in IEEE Robotics and Automation Letters, vol. 11, no. 3, pp. 2935-2942, March 2026, doi: 10.1109/LRA.2026.3655537
>
> **摘要:** Soft everting robots present significant advantages over traditional rigid robots, including enhanced dexterity, improved environmental interaction, and safe navigation in unpredictable environments. While soft everting robots have been widely demonstrated for exploration type tasks, their potential to move and deploy payloads in such tasks has been less investigated, with previous work focusing on sensors and tools for the robot. Leveraging the navigation capabilities, and deployed body, of the soft everting robot to deliver payloads in hazardous areas, e.g. carrying a water bottle to a person stuck under debris, would represent a significant capability in many applications. In this work, we present an analysis of how soft everting robots can be used to deploy larger, heavier payloads through the inside of the robot. We analyze both what objects can be deployed and what terrain features they can be carried through. Building on existing models, we present methods to quantify the effects of payloads on robot growth and self-support, and develop a model to predict payload slip. We then experimentally quantify payload transport using soft everting robot with a variety of payload shapes, sizes, and weights and though a series of tasks: steering, vertical transport, movement through holes, and movement across gaps. Overall, the results show that we can transport payloads in a variety of shapes and up to 1.5kg in weight and that we can move through circular apertures with as little as 0.01cm clearance around payloads, carry out discrete turns up to 135 degrees, and move across unsupported gaps of 1.15m in length.
>
---
#### [replaced 016] Visual Localization via Semantic Structures in Autonomous Photovoltaic Power Plant Inspection
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自主导航任务，解决光伏电站巡检中无人机精确定位问题。通过整合模块检测与导航，实现精准定位，并评估不同分割方法的性能。**

- **链接: [https://arxiv.org/pdf/2501.14587v2](https://arxiv.org/pdf/2501.14587v2)**

> **作者:** Viktor Kozák; Karel Košnar; Jan Chudoba; Miroslav Kulich; Libor Přeučil
>
> **备注:** 50 pages, 23 figures. Submitted for review to Array
>
> **摘要:** Inspection systems utilizing unmanned aerial vehicles (UAVs) equipped with thermal cameras are increasingly popular for the maintenance of photovoltaic (PV) power plants. However, automation of the inspection task is a challenging problem as it requires precise navigation to capture images from optimal distances and viewing angles. This paper presents a novel localization pipeline that directly integrates PV module detection with UAV navigation, allowing precise positioning during inspection. The detections are used to identify the power plant structures in the image. These are associated with the power plant model and used to infer the UAV position relative to the inspected PV installation. We define visually recognizable anchor points for the initial association and use object tracking to discern global associations. Additionally, we present three different methods for visual segmentation of PV modules and evaluate their performance in relation to the proposed localization pipeline. The presented methods were verified and evaluated using custom aerial inspection data sets, demonstrating their robustness and applicability for real-time navigation. Additionally, we evaluate the influence of the power plant model precision on the localization methods.
>
---
#### [replaced 017] Unifying Perception and Action: A Hybrid-Modality Pipeline with Implicit Visual Chain-of-Thought for Robotic Action Generation
- **分类: cs.RO**

- **简介: 该论文属于机器人动作生成任务，旨在解决视觉与动作间模态差距及训练不稳定问题。提出VITA框架，通过共享潜在空间和隐式视觉思维链，提升机器人操作性能。**

- **链接: [https://arxiv.org/pdf/2511.19859v2](https://arxiv.org/pdf/2511.19859v2)**

> **作者:** Xiangkai Ma; Lekai Xing; Han Zhang; Wenzhong Li; Sanglu Lu
>
> **摘要:** Vision-Language-Action (VLA) models built upon Chain-of-Thought (CoT) have achieved remarkable success in advancing general-purpose robotic agents, owing to its significant perceptual comprehension. Recently, since text-only CoT struggles to adequately capture scene details in complex spatial environments, a highly promising strategy involves leveraging visual priors to guide robotic action generation. Nevertheless, these strategies face two inherent challenges: (i) a modality gap between visual observations and low-level actions, and (ii) unstable training due to competing objectives between visual prediction and action generation. To address these challenges, we propose a Vision-Integrated Trajectory Alignment (VITA) framework that learns a shared discrete latent space for vision and action, enabling joint modeling of perception and motor control. VITA introduces a implicit visual CoT: autoregressively generated tokens is simultaneously decoded into future frames predictions and robot actions, thereby internalizing visual dynamics as an inductive bias for motion planning. Extensive experiments on simulated and real-world environments demonstrate state-of-the-art performance. VITA improves 14.5\%, 9.6\% and 12.1\% over existing baselines on CALVIN, LIBERO and SimplerEnv. Furthermore, VITA attains an average success rate of 80.5\% across six real-world tasks, demonstrating its potential as a generalist robotic manipulation model.
>
---
#### [replaced 018] OMP: One-step Meanflow Policy with Directional Alignment
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决数据驱动策略框架中推理延迟与复杂架构的矛盾。提出OMP框架，通过方向对齐和DDE方法提升精度与效率。**

- **链接: [https://arxiv.org/pdf/2512.19347v2](https://arxiv.org/pdf/2512.19347v2)**

> **作者:** Han Fang; Yize Huang; Yuheng Zhao; Paul Weng; Xiao Li; Yutong Ban
>
> **摘要:** Robot manipulation has increasingly adopted data-driven generative policy frameworks, yet the field faces a persistent trade-off: diffusion models suffer from high inference latency, while flow-based methods often require complex architectural constraints. Although in image generation domain, the MeanFlow paradigm offers a path to single-step inference, its direct application to robotics is impeded by critical theoretical pathologies, specifically spectral bias and gradient starvation in low-velocity regimes. To overcome these limitations, we propose the One-step MeanFlow Policy (OMP), a novel framework designed for high-fidelity, real-time manipulation. We introduce a lightweight directional alignment mechanism to explicitly synchronize predicted velocities with true mean velocities. Furthermore, we implement a Differential Derivation Equation (DDE) to approximate the Jacobian-Vector Product (JVP) operator, which decouples forward and backward passes to significantly reduce memory complexity. Extensive experiments on the Adroit and Meta-World benchmarks demonstrate that OMP outperforms state-of-the-art methods in success rate and trajectory accuracy, particularly in high-precision tasks, while retaining the efficiency of single-step generation.
>
---
