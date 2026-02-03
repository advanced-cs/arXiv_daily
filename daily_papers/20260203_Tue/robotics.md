# 机器人 cs.RO

- **最新发布 110 篇**

- **更新 52 篇**

## 最新发布

#### [new 001] Toward Reliable Sim-to-Real Predictability for MoE-based Robust Quadrupedal Locomotion
- **分类: cs.RO**

- **简介: 该论文属于四足机器人运动控制任务，解决sim-to-real迁移和奖励过拟合问题。提出MoE策略与RoboGauge评估框架，提升复杂地形下的运动鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.00678v1](https://arxiv.org/pdf/2602.00678v1)**

> **作者:** Tianyang Wu; Hanwei Guo; Yuhang Wang; Junshu Yang; Xinyang Sui; Jiayi Xie; Xingyu Chen; Zeyang Liu; Xuguang Lan
>
> **摘要:** Reinforcement learning has shown strong promise for quadrupedal agile locomotion, even with proprioception-only sensing. In practice, however, sim-to-real gap and reward overfitting in complex terrains can produce policies that fail to transfer, while physical validation remains risky and inefficient. To address these challenges, we introduce a unified framework encompassing a Mixture-of-Experts (MoE) locomotion policy for robust multi-terrain representation with RoboGauge, a predictive assessment suite that quantifies sim-to-real transferability. The MoE policy employs a gated set of specialist experts to decompose latent terrain and command modeling, achieving superior deployment robustness and generalization via proprioception alone. RoboGauge further provides multi-dimensional proprioception-based metrics via sim-to-sim tests over terrains, difficulty levels, and domain randomizations, enabling reliable MoE policy selection without extensive physical trials. Experiments on a Unitree Go2 demonstrate robust locomotion on unseen challenging terrains, including snow, sand, stairs, slopes, and 30 cm obstacles. In dedicated high-speed tests, the robot reaches 4 m/s and exhibits an emergent narrow-width gait associated with improved stability at high velocity.
>
---
#### [new 002] Physics-informed Diffusion Mamba Transformer for Real-world Driving
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶轨迹预测任务，旨在解决模型难以融合物理约束和长期序列依赖的问题。提出Diffusion Mamba Transformer和Port-Hamiltonian模块，提升预测准确性与物理合理性。**

- **链接: [https://arxiv.org/pdf/2602.00808v1](https://arxiv.org/pdf/2602.00808v1)**

> **作者:** Hang Zhou; Qiang Zhang; Peiran Liu; Yihao Qin; Zhaoxu Yan; Yiding Ji
>
> **摘要:** Autonomous driving systems demand trajectory planners that not only model the inherent uncertainty of future motions but also respect complex temporal dependencies and underlying physical laws. While diffusion-based generative models excel at capturing multi-modal distributions, they often fail to incorporate long-term sequential contexts and domain-specific physical priors. In this work, we bridge these gaps with two key innovations. First, we introduce a Diffusion Mamba Transformer architecture that embeds mamba and attention into the diffusion process, enabling more effective aggregation of sequential input contexts from sensor streams and past motion histories. Second, we design a Port-Hamiltonian Neural Network module that seamlessly integrates energy-based physical constraints into the diffusion model, thereby enhancing trajectory predictions with both consistency and interpretability. Extensive evaluations on standard autonomous driving benchmarks demonstrate that our unified framework significantly outperforms state-of-the-art baselines in predictive accuracy, physical plausibility, and robustness, thereby advancing safe and reliable motion planning.
>
---
#### [new 003] Frictional Contact Solving for Material Point Method
- **分类: cs.RO**

- **简介: 该论文属于接触力学任务，解决MPM中摩擦接触的精确处理问题。提出一种基于NCP和ADMM的摩擦接触算法，提升模拟的准确性与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.02038v1](https://arxiv.org/pdf/2602.02038v1)**

> **作者:** Etienne Ménager; Justin Carpentier
>
> **摘要:** Accurately handling contact with friction remains a core bottleneck for Material Point Method (MPM), from reliable contact point detection to enforcing frictional contact laws (non-penetration, Coulomb friction, and maximum dissipation principle). In this paper, we introduce a frictional-contact pipeline for implicit MPM that is both precise and robust. During the collision detection phase, contact points are localized with particle-centric geometric primitives; during the contact resolution phase, we cast frictional contact as a Nonlinear Complementarity Problem (NCP) over contact impulses and solve it with an Alternating Direction Method of Multipliers (ADMM) scheme. Crucially, the formulation reuses the same implicit MPM linearization, yielding efficiency and numerical stability. The method integrates seamlessly into the implicit MPM loop and is agnostic to modeling choices, including material laws, interpolation functions, and transfer schemes. We evaluate it across seven representative scenes that span elastic and elasto-plastic responses, simple and complex deformable geometries, and a wide range of contact conditions. Overall, the proposed method enables accurate contact localization, reliable frictional handling, and broad generality, making it a practical solution for MPM-based simulations in robotics and related domains.
>
---
#### [new 004] MapDream: Task-Driven Map Learning for Vision-Language Navigation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于视觉语言导航任务，旨在解决传统地图构建与导航策略脱节的问题。提出MapDream框架，通过生成式地图学习提升导航性能。**

- **链接: [https://arxiv.org/pdf/2602.00222v1](https://arxiv.org/pdf/2602.00222v1)**

> **作者:** Guoxin Lian; Shuo Wang; Yucheng Wang; Yongcai Wang; Maiyue Chen; Kaihui Wang; Bo Zhang; Zhizhong Su; Deying Li; Zhaoxin Fan
>
> **摘要:** Vision-Language Navigation (VLN) requires agents to follow natural language instructions in partially observed 3D environments, motivating map representations that aggregate spatial context beyond local perception. However, most existing approaches rely on hand-crafted maps constructed independently of the navigation policy. We argue that maps should instead be learned representations shaped directly by navigation objectives rather than exhaustive reconstructions. Based on this insight, we propose MapDream, a map-in-the-loop framework that formulates map construction as autoregressive bird's-eye-view (BEV) image synthesis. The framework jointly learns map generation and action prediction, distilling environmental context into a compact three-channel BEV map that preserves only navigation-critical affordances. Supervised pre-training bootstraps a reliable mapping-to-control interface, while the autoregressive design enables end-to-end joint optimization through reinforcement fine-tuning. Experiments on R2R-CE and RxR-CE achieve state-of-the-art monocular performance, validating task-driven generative map learning.
>
---
#### [new 005] Concept-Based Dictionary Learning for Inference-Time Safety in Vision Language Action Models
- **分类: cs.RO**

- **简介: 该论文属于视觉语言动作模型的安全防护任务，解决嵌入式系统中因指令误解引发的危险行为问题。通过概念字典学习，在推理时检测并阻断有害激活，提升模型安全性。**

- **链接: [https://arxiv.org/pdf/2602.01834v1](https://arxiv.org/pdf/2602.01834v1)**

> **作者:** Siqi Wen; Shu Yang; Shaopeng Fu; Jingfeng Zhang; Lijie Hu; Di Wang
>
> **摘要:** Vision Language Action (VLA) models close the perception action loop by translating multimodal instructions into executable behaviors, but this very capability magnifies safety risks: jailbreaks that merely yield toxic text in LLMs can trigger unsafe physical actions in embodied systems. Existing defenses alignment, filtering, or prompt hardening intervene too late or at the wrong modality, leaving fused representations exploitable. We introduce a concept-based dictionary learning framework for inference-time safety control. By constructing sparse, interpretable dictionaries from hidden activations, our method identifies harmful concept directions and applies threshold-based interventions to suppress or block unsafe activations. Experiments on Libero-Harm, BadRobot, RoboPair, and IS-Bench show that our approach achieves state-of-the-art defense performance, cutting attack success rates by over 70\% while maintaining task success. Crucially, the framework is plug-in and model-agnostic, requiring no retraining and integrating seamlessly with diverse VLAs. To our knowledge, this is the first inference-time concept-based safety method for embodied systems, advancing both interpretability and safe deployment of VLA models.
>
---
#### [new 006] Bridging the Sim-to-Real Gap with multipanda ros2: A Real-Time ROS2 Framework for Multimanual Systems
- **分类: cs.RO; cs.AI; cs.SE; eess.SY**

- **简介: 该论文属于多机器人控制任务，旨在解决sim2real差距和实时扭矩控制问题。提出multipanda_ros2框架，实现高精度多机械臂控制与仿真验证。**

- **链接: [https://arxiv.org/pdf/2602.02269v1](https://arxiv.org/pdf/2602.02269v1)**

> **作者:** Jon Škerlj; Seongjin Bien; Abdeldjallil Naceri; Sami Haddadin
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** We present $multipanda\_ros2$, a novel open-source ROS2 architecture for multi-robot control of Franka Robotics robots. Leveraging ros2 control, this framework provides native ROS2 interfaces for controlling any number of robots from a single process. Our core contributions address key challenges in real-time torque control, including interaction control and robot-environment modeling. A central focus of this work is sustaining a 1kHz control frequency, a necessity for real-time control and a minimum frequency required by safety standards. Moreover, we introduce a controllet-feature design pattern that enables controller-switching delays of $\le 2$ ms, facilitating reproducible benchmarking and complex multi-robot interaction scenarios. To bridge the simulation-to-reality (sim2real) gap, we integrate a high-fidelity MuJoCo simulation with quantitative metrics for both kinematic accuracy and dynamic consistency (torques, forces, and control errors). Furthermore, we demonstrate that real-world inertial parameter identification can significantly improve force and torque accuracy, providing a methodology for iterative physics refinement. Our work extends approaches from soft robotics to rigid dual-arm, contact-rich tasks, showcasing a promising method to reduce the sim2real gap and providing a robust, reproducible platform for advanced robotics research.
>
---
#### [new 007] Green-VLA: Staged Vision-Language-Action Model for Generalist Robots
- **分类: cs.RO**

- **简介: 该论文提出Green-VLA框架，解决机器人通用控制问题。通过多阶段训练和强化学习，提升机器人在不同形态下的任务执行能力与安全性。**

- **链接: [https://arxiv.org/pdf/2602.00919v1](https://arxiv.org/pdf/2602.00919v1)**

> **作者:** I. Apanasevich; M. Artemyev; R. Babakyan; P. Fedotova; D. Grankin; E. Kupryashin; A. Misailidi; D. Nerus; A. Nutalapati; G. Sidorov; I. Efremov; M. Gerasyov; D. Pikurov; Y. Senchenko; S. Davidenko; D. Kulikov; M. Sultankin; K. Askarbek; O. Shamanin; D. Statovoy; E. Zalyaev; I. Zorin; A. Letkin; E. Rusakov; A. Silchenko; V. Vorobyov; S. Sobolnikov; A. Postnikov
>
> **备注:** 22 pages, 14 figures
>
> **摘要:** We introduce Green-VLA, a staged Vision-Language-Action (VLA) framework for real-world deployment on the Green humanoid robot while maintaining generalization across diverse embodiments. Green-VLA follows a five stage curriculum: (L0) foundational VLMs, (L1) multimodal grounding, (R0) multi-embodiment pretraining, (R1) embodiment-specific adaptation, and (R2) reinforcement-learning (RL) policy alignment. We couple a scalable data-processing pipeline (3,000 hours of demonstrations) with temporal alignment and quality filtering, and use a unified, embodiment-aware action interface enabling a single policy to control humanoids, mobile manipulators, and fixed-base arms. At inference, the VLA controller is enhanced with episode-progress prediction, out-of-distribution detection, and joint-prediction-based guidance to improve safety and precise target selection. Experiments on Simpler BRIDGE WidowX and CALVIN ABC-D, as well as real-robot evaluations, show strong generalization and performance gains from RL alignment in success rate, robustness, and long-horizon efficiency.
>
---
#### [new 008] Towards Autonomous Instrument Tray Assembly for Sterile Processing Applications
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于手术器械自动装配任务，旨在解决人工装配效率低、易出错的问题。通过构建自动化系统实现器械的检测、分类与安全打包。**

- **链接: [https://arxiv.org/pdf/2602.01679v1](https://arxiv.org/pdf/2602.01679v1)**

> **作者:** Raghavasimhan Sankaranarayanan; Paul Stuart; Nicholas Ahn; Arno Sungarian; Yash Chitalia
>
> **备注:** 7 pages, 9 figures, 2026 International Symposium on Medical Robotics
>
> **摘要:** The Sterile Processing and Distribution (SPD) department is responsible for cleaning, disinfecting, inspecting, and assembling surgical instruments between surgeries. Manual inspection and preparation of instrument trays is a time-consuming, error-prone task, often prone to contamination and instrument breakage. In this work, we present a fully automated robotic system that sorts and structurally packs surgical instruments into sterile trays, focusing on automation of the SPD assembly stage. A custom dataset comprising 31 surgical instruments and 6,975 annotated images was collected to train a hybrid perception pipeline using YOLO12 for detection and a cascaded ResNet-based model for fine-grained classification. The system integrates a calibrated vision module, a 6-DOF Staubli TX2-60L robotic arm with a custom dual electromagnetic gripper, and a rule-based packing algorithm that reduces instrument collisions during transport. The packing framework uses 3D printed dividers and holders to physically isolate instruments, reducing collision and friction during transport. Experimental evaluations show high perception accuracy and statistically significant reduction in tool-to-tool collisions compared to human-assembled trays. This work serves as the scalable first step toward automating SPD workflows, improving safety, and consistency of surgical preparation while reducing SPD processing times.
>
---
#### [new 009] LIEREx: Language-Image Embeddings for Robotic Exploration
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人探索任务，旨在解决传统地图无法处理未知对象的问题。通过融合视觉-语言模型与3D语义图，实现目标导向的自主探索。**

- **链接: [https://arxiv.org/pdf/2602.01930v1](https://arxiv.org/pdf/2602.01930v1)**

> **作者:** Felix Igelbrink; Lennart Niecksch; Marian Renz; Martin Günther; Martin Atzmueller
>
> **备注:** This preprint has not undergone peer review or any post-submission improvements or corrections. The Version of Record of this article is published in KI - Künstliche Intelligenz, and is available online at https://doi.org/10.1007/s13218-026-00902-6
>
> **摘要:** Semantic maps allow a robot to reason about its surroundings to fulfill tasks such as navigating known environments, finding specific objects, and exploring unmapped areas. Traditional mapping approaches provide accurate geometric representations but are often constrained by pre-designed symbolic vocabularies. The reliance on fixed object classes makes it impractical to handle out-of-distribution knowledge not defined at design time. Recent advances in Vision-Language Foundation Models, such as CLIP, enable open-set mapping, where objects are encoded as high-dimensional embeddings rather than fixed labels. In LIEREx, we integrate these VLFMs with established 3D Semantic Scene Graphs to enable target-directed exploration by an autonomous agent in partially unknown environments.
>
---
#### [new 010] A Unified Control Architecture for Macro-Micro Manipulation using a Active Remote Center of Compliance for Manufacturing Applications
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决宏微操作中交互控制带宽不足的问题。通过统一控制架构和主动顺应控制，提升控制性能并优化控制器设计。**

- **链接: [https://arxiv.org/pdf/2602.01948v1](https://arxiv.org/pdf/2602.01948v1)**

> **作者:** Patrick Frank; Christian Friedrich
>
> **备注:** 17 pages, 14 figures, submitted to Robotics and Computer-Integrated Manufacturing (RCIM)
>
> **摘要:** Macro-micro manipulators combine a macro manipulator with a large workspace, such as an industrial robot, with a lightweight, high-bandwidth micro manipulator. This enables highly dynamic interaction control while preserving the wide workspace of the robot. Traditionally, position control is assigned to the macro manipulator, while the micro manipulator handles the interaction with the environment, limiting the achievable interaction control bandwidth. To solve this, we propose a novel control architecture that incorporates the macro manipulator into the active interaction control. This leads to a increase in control bandwidth by a factor of 2.1 compared to the state of the art architecture, based on the leader-follower approach and factor 12.5 compared to traditional robot-based force control. Further we propose surrogate models for a more efficient controller design and easy adaptation to hardware changes. We validate our approach by comparing it against the other control schemes in different experiments, like collision with an object, following a force trajectory and industrial assembly tasks.
>
---
#### [new 011] Estimating Force Interactions of Deformable Linear Objects from their Shapes
- **分类: cs.RO**

- **简介: 该论文属于机器人与柔性线性物体交互任务，旨在无需传感器直接估计外部力。通过分析物体形状，建立方程求解力的位置和大小，提升轨迹规划安全性。**

- **链接: [https://arxiv.org/pdf/2602.01085v1](https://arxiv.org/pdf/2602.01085v1)**

> **作者:** Qi Jing Chen; Shilin Shan; Timothy Bretl; Quang-Cuong Pham
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** This work introduces an analytical approach for detecting and estimating external forces acting on deformable linear objects (DLOs) using only their observed shapes. In many robot-wire interaction tasks, contact occurs not at the end-effector but at other points along the robot's body. Such scenarios arise when robots manipulate wires indirectly (e.g., by nudging) or when wires act as passive obstacles in the environment. Accurately identifying these interactions is crucial for safe and efficient trajectory planning, helping to prevent wire damage, avoid restricted robot motions, and mitigate potential hazards. Existing approaches often rely on expensive external force-torque sensor or that contacts occur at the end-effector for accurate force estimation. Using wire shape information acquired from a depth camera and under the assumption that the wire is in or near its static equilibrium, our method estimates both the location and magnitude of external forces without additional prior knowledge. This is achieved by exploiting derived consistency conditions and solving a system of linear equations based on force-torque balance along the wire. The approach was validated through simulation, where it achieved high accuracy, and through real-world experiments, where accurate estimation was demonstrated in selected interaction scenarios.
>
---
#### [new 012] UniMorphGrasp: Diffusion Model with Morphology-Awareness for Cross-Embodiment Dexterous Grasp Generation
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决跨形态灵巧抓取问题。通过引入形态感知的扩散模型，实现不同机械手的通用抓取生成与泛化。**

- **链接: [https://arxiv.org/pdf/2602.00915v1](https://arxiv.org/pdf/2602.00915v1)**

> **作者:** Zhiyuan Wu; Xiangyu Zhang; Zhuo Chen; Jiankang Deng; Rolandos Alexandros Potamias; Shan Luo
>
> **摘要:** Cross-embodiment dexterous grasping aims to generate stable and diverse grasps for robotic hands with heterogeneous kinematic structures. Existing methods are often tailored to specific hand designs and fail to generalize to unseen hand morphologies outside the training distribution. To address these limitations, we propose \textbf{UniMorphGrasp}, a diffusion-based framework that incorporates hand morphological information into the grasp generation process for unified cross-embodiment grasp synthesis. The proposed approach maps grasps from diverse robotic hands into a unified human-like canonical hand pose representation, providing a common space for learning. Grasp generation is then conditioned on structured representations of hand kinematics, encoded as graphs derived from hand configurations, together with object geometry. In addition, a loss function is introduced that exploits the hierarchical organization of hand kinematics to guide joint-level supervision. Extensive experiments demonstrate that UniMorphGrasp achieves state-of-the-art performance on existing dexterous grasp benchmarks and exhibits strong zero-shot generalization to previously unseen hand structures, enabling scalable and practical cross-embodiment grasp deployment.
>
---
#### [new 013] Reinforcement Learning for Active Perception in Autonomous Navigation
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决复杂环境中主动感知问题。通过强化学习框架，使机器人同时实现避障和相机控制，提升环境感知能力。**

- **链接: [https://arxiv.org/pdf/2602.01266v1](https://arxiv.org/pdf/2602.01266v1)**

> **作者:** Grzegorz Malczyk; Mihir Kulkarni; Kostas Alexis
>
> **备注:** Accepted to the IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** This paper addresses the challenge of active perception within autonomous navigation in complex, unknown environments. Revisiting the foundational principles of active perception, we introduce an end-to-end reinforcement learning framework in which a robot must not only reach a goal while avoiding obstacles, but also actively control its onboard camera to enhance situational awareness. The policy receives observations comprising the robot state, the current depth frame, and a particularly local geometry representation built from a short history of depth readings. To couple collision-free motion planning with information-driven active camera control, we augment the navigation reward with a voxel-based information metric. This enables an aerial robot to learn a robust policy that balances goal-directed motion with exploratory sensing. Extensive evaluation demonstrates that our strategy achieves safer flight compared to using fixed, non-actuated camera baselines while also inducing intrinsic exploratory behaviors.
>
---
#### [new 014] A Closed-Form Geometric Retargeting Solver for Upper Body Humanoid Robot Teleoperation
- **分类: cs.RO**

- **简介: 该论文属于机器人遥操作任务，解决人体运动到机器人姿态的映射问题。提出SEW-Mimic方法，通过上肢方向对齐实现快速准确的映射。**

- **链接: [https://arxiv.org/pdf/2602.01632v1](https://arxiv.org/pdf/2602.01632v1)**

> **作者:** Chuizheng Kong; Yunho Cho; Wonsuhk Jung; Idris Wibowo; Parth Shinde; Sundhar Vinodh-Sangeetha; Long Kiu Chung; Zhenyang Chen; Andrew Mattei; Advaith Nidumukkala; Alexander Elias; Danfei Xu; Taylor Higgins; Shreyas Kousik
>
> **备注:** Project page at https://sew-mimic.com/
>
> **摘要:** Retargeting human motion to robot poses is a practical approach for teleoperating bimanual humanoid robot arms, but existing methods can be suboptimal and slow, often causing undesirable motion or latency. This is due to optimizing to match robot end-effector to human hand position and orientation, which can also limit the robot's workspace to that of the human. Instead, this paper reframes retargeting as an orientation alignment problem, enabling a closed-form, geometric solution algorithm with an optimality guarantee. The key idea is to align a robot arm to a human's upper and lower arm orientations, as identified from shoulder, elbow, and wrist (SEW) keypoints; hence, the method is called SEW-Mimic. The method has fast inference (3 kHz) on standard commercial CPUs, leaving computational overhead for downstream applications; an example in this paper is a safety filter to avoid bimanual self-collision. The method suits most 7-degree-of-freedom robot arms and humanoids, and is agnostic to input keypoint source. Experiments show that SEW-Mimic outperforms other retargeting methods in computation time and accuracy. A pilot user study suggests that the method improves teleoperation task success. Preliminary analysis indicates that data collected with SEW-Mimic improves policy learning due to being smoother. SEW-Mimic is also shown to be a drop-in way to accelerate full-body humanoid retargeting. Finally, hardware demonstrations illustrate SEW-Mimic's practicality. The results emphasize the utility of SEW-Mimic as a fundamental building block for bimanual robot manipulation and humanoid robot teleoperation.
>
---
#### [new 015] ConLA: Contrastive Latent Action Learning from Human Videos for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出ConLA框架，用于从人类视频中无监督学习机器人操作策略。解决数据获取成本高、泛化能力弱的问题，通过对比解耦机制提升表示质量。**

- **链接: [https://arxiv.org/pdf/2602.00557v1](https://arxiv.org/pdf/2602.00557v1)**

> **作者:** Weisheng Dai; Kai Lan; Jianyi Zhou; Bo Zhao; Xiu Su; Junwen Tong; Weili Guan; Shuo Yang
>
> **摘要:** Vision-Language-Action (VLA) models achieve preliminary generalization through pretraining on large scale robot teleoperation datasets. However, acquiring datasets that comprehensively cover diverse tasks and environments is extremely costly and difficult to scale. In contrast, human demonstration videos offer a rich and scalable source of diverse scenes and manipulation behaviors, yet their lack of explicit action supervision hinders direct utilization. Prior work leverages VQ-VAE based frameworks to learn latent actions from human videos in an unsupervised manner. Nevertheless, since the training objective primarily focuses on reconstructing visual appearances rather than capturing inter-frame dynamics, the learned representations tend to rely on spurious visual cues, leading to shortcut learning and entangled latent representations that hinder transferability. To address this, we propose ConLA, an unsupervised pretraining framework for learning robotic policies from human videos. ConLA introduces a contrastive disentanglement mechanism that leverages action category priors and temporal cues to isolate motion dynamics from visual content, effectively mitigating shortcut learning. Extensive experiments show that ConLA achieves strong performance across diverse benchmarks. Notably, by pretraining solely on human videos, our method for the first time surpasses the performance obtained with real robot trajectory pretraining, highlighting its ability to extract pure and semantically consistent latent action representations for scalable robot learning.
>
---
#### [new 016] USS-Nav: Unified Spatio-Semantic Scene Graph for Lightweight UAV Zero-Shot Object Navigation
- **分类: cs.RO**

- **简介: 该论文属于无人机零样本目标导航任务，解决未知环境中语义推理与计算资源受限的矛盾。提出USS-Nav框架，构建统一时空场景图，提升导航效率与实时性。**

- **链接: [https://arxiv.org/pdf/2602.00708v1](https://arxiv.org/pdf/2602.00708v1)**

> **作者:** Weiqi Gai; Yuman Gao; Yuan Zhou; Yufan Xie; Zhiyang Liu; Yuze Wu; Xin Zhou; Fei Gao; Zhijun Meng
>
> **摘要:** Zero-Shot Object Navigation in unknown environments poses significant challenges for Unmanned Aerial Vehicles (UAVs) due to the conflict between high-level semantic reasoning requirements and limited onboard computational resources. To address this, we present USS-Nav, a lightweight framework that incrementally constructs a Unified Spatio-Semantic scene graph and enables efficient Large Language Model (LLM)-augmented Zero-Shot Object Navigation in unknown environments. Specifically, we introduce an incremental Spatial Connectivity Graph generation method utilizing polyhedral expansion to capture global geometric topology, which is dynamically partitioned into semantic regions via graph clustering. Concurrently, open-vocabulary object semantics are instantiated and anchored to this topology to form a hierarchical environmental representation. Leveraging this hierarchical structure, we present a coarse-to-fine exploration strategy: LLM grounded in the scene graph's semantics to determine global target regions, while a local planner optimizes frontier coverage based on information gain. Experimental results demonstrate that our framework outperforms state-of-the-art methods in terms of computational efficiency and real-time update frequency (15 Hz) on a resource-constrained platform. Furthermore, ablation studies confirm the effectiveness of our framework, showing substantial improvements in Success weighted by Path Length (SPL). The source code will be made publicly available to foster further research.
>
---
#### [new 017] SA-VLA: Spatially-Aware Flow-Matching for Vision-Language-Action Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于视觉-语言-动作强化学习任务，旨在解决RL微调中因空间分布变化导致的鲁棒性下降问题。提出SA-VLA框架，通过空间感知机制提升策略稳定性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.00743v1](https://arxiv.org/pdf/2602.00743v1)**

> **作者:** Xu Pan; Zhenglin Wan; Xingrui Yu; Xianwei Zheng; Youkai Ke; Ming Sun; Rui Wang; Ziwei Wang; Ivor Tsang
>
> **备注:** Version 1
>
> **摘要:** Vision-Language-Action (VLA) models exhibit strong generalization in robotic manipulation, yet reinforcement learning (RL) fine-tuning often degrades robustness under spatial distribution shifts. For flow-matching VLA policies, this degradation is closely associated with the erosion of spatial inductive bias during RL adaptation, as sparse rewards and spatially agnostic exploration increasingly favor short-horizon visual cues. To address this issue, we propose \textbf{SA-VLA}, a spatially-aware RL adaptation framework that preserves spatial grounding during policy optimization by aligning representation learning, reward design, and exploration with task geometry. SA-VLA fuses implicit spatial representations with visual tokens, provides dense rewards that reflect geometric progress, and employs \textbf{SCAN}, a spatially-conditioned annealed exploration strategy tailored to flow-matching dynamics. Across challenging multi-object and cluttered manipulation benchmarks, SA-VLA enables stable RL fine-tuning and improves zero-shot spatial generalization, yielding more robust and transferable behaviors. Code and project page are available at https://xupan.top/Projects/savla.
>
---
#### [new 018] ZEST: Zero-shot Embodied Skill Transfer for Athletic Robot Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出ZEST框架，解决机器人零样本技能迁移问题。通过强化学习从多种数据源训练策略，实现跨行为、平台的鲁棒控制，无需接触标签或复杂调参。**

- **链接: [https://arxiv.org/pdf/2602.00401v1](https://arxiv.org/pdf/2602.00401v1)**

> **作者:** Jean Pierre Sleiman; He Li; Alphonsus Adu-Bredu; Robin Deits; Arun Kumar; Kevin Bergamin; Mohak Bhardwaj; Scott Biddlestone; Nicola Burger; Matthew A. Estrada; Francesco Iacobelli; Twan Koolen; Alexander Lambert; Erica Lin; M. Eva Mungai; Zach Nobles; Shane Rozen-Levy; Yuyao Shi; Jiashun Wang; Jakob Welner; Fangzhou Yu; Mike Zhang; Alfred Rizzi; Jessica Hodgins; Sylvain Bertrand; Yeuhi Abe; Scott Kuindersma; Farbod Farshidian
>
> **摘要:** Achieving robust, human-like whole-body control on humanoid robots for agile, contact-rich behaviors remains a central challenge, demanding heavy per-skill engineering and a brittle process of tuning controllers. We introduce ZEST (Zero-shot Embodied Skill Transfer), a streamlined motion-imitation framework that trains policies via reinforcement learning from diverse sources -- high-fidelity motion capture, noisy monocular video, and non-physics-constrained animation -- and deploys them to hardware zero-shot. ZEST generalizes across behaviors and platforms while avoiding contact labels, reference or observation windows, state estimators, and extensive reward shaping. Its training pipeline combines adaptive sampling, which focuses training on difficult motion segments, and an automatic curriculum using a model-based assistive wrench, together enabling dynamic, long-horizon maneuvers. We further provide a procedure for selecting joint-level gains from approximate analytical armature values for closed-chain actuators, along with a refined model of actuators. Trained entirely in simulation with moderate domain randomization, ZEST demonstrates remarkable generality. On Boston Dynamics' Atlas humanoid, ZEST learns dynamic, multi-contact skills (e.g., army crawl, breakdancing) from motion capture. It transfers expressive dance and scene-interaction skills, such as box-climbing, directly from videos to Atlas and the Unitree G1. Furthermore, it extends across morphologies to the Spot quadruped, enabling acrobatics, such as a continuous backflip, through animation. Together, these results demonstrate robust zero-shot deployment across heterogeneous data sources and embodiments, establishing ZEST as a scalable interface between biological movements and their robotic counterparts.
>
---
#### [new 019] RFS: Reinforcement learning with Residual flow steering for dexterous manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决预训练策略泛化能力不足的问题。提出RFS框架，通过残差修正和潜在空间调节实现高效微调。**

- **链接: [https://arxiv.org/pdf/2602.01789v1](https://arxiv.org/pdf/2602.01789v1)**

> **作者:** Entong Su; Tyler Westenbroek; Anusha Nagabandi; Abhishek Gupta
>
> **摘要:** Imitation learning has emerged as an effective approach for bootstrapping sequential decision-making in robotics, achieving strong performance even in high-dimensional dexterous manipulation tasks. Recent behavior cloning methods further leverage expressive generative models, such as diffusion models and flow matching, to represent multimodal action distributions. However, policies pretrained in this manner often exhibit limited generalization and require additional fine-tuning to achieve robust performance at deployment time. Such adaptation must preserve the global exploration benefits of pretraining while enabling rapid correction of local execution errors.We propose \emph{Residual Flow Steering} (RFS), a data-efficient reinforcement learning framework for adapting pretrained generative policies. RFS steers a pretrained flow-matching policy by jointly optimizing a residual action and a latent noise distribution, enabling complementary forms of exploration: local refinement through residual corrections and global exploration through latent-space modulation. This design allows efficient adaptation while retaining the expressive structure of the pretrained policy.We demonstrate the effectiveness of RFS on dexterous manipulation tasks, showing efficient fine-tuning both in simulation and in real-world settings when adapting pretrained base policies.Project website:https://weirdlabuw.github.io/rfs.
>
---
#### [new 020] TTT-Parkour: Rapid Test-Time Training for Perceptive Robot Parkour
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人动态行走任务，解决复杂地形下机器人运动控制问题。通过快速测试时训练（TTT）提升机器人在未知地形上的适应能力。**

- **链接: [https://arxiv.org/pdf/2602.02331v1](https://arxiv.org/pdf/2602.02331v1)**

> **作者:** Shaoting Zhu; Baijun Ye; Jiaxuan Wang; Jiakang Chen; Ziwen Zhuang; Linzhan Mou; Runhan Huang; Hang Zhao
>
> **备注:** Project Page: https://ttt-parkour.github.io/
>
> **摘要:** Achieving highly dynamic humanoid parkour on unseen, complex terrains remains a challenge in robotics. Although general locomotion policies demonstrate capabilities across broad terrain distributions, they often struggle with arbitrary and highly challenging environments. To overcome this limitation, we propose a real-to-sim-to-real framework that leverages rapid test-time training (TTT) on novel terrains, significantly enhancing the robot's capability to traverse extremely difficult geometries. We adopt a two-stage end-to-end learning paradigm: a policy is first pre-trained on diverse procedurally generated terrains, followed by rapid fine-tuning on high-fidelity meshes reconstructed from real-world captures. Specifically, we develop a feed-forward, efficient, and high-fidelity geometry reconstruction pipeline using RGB-D inputs, ensuring both speed and quality during test-time training. We demonstrate that TTT-Parkour empowers humanoid robots to master complex obstacles, including wedges, stakes, boxes, trapezoids, and narrow beams. The whole pipeline of capturing, reconstructing, and test-time training requires less than 10 minutes on most tested terrains. Extensive experiments show that the policy after test-time training exhibits robust zero-shot sim-to-real transfer capability.
>
---
#### [new 021] Mapping-Guided Task Discovery and Allocation for Robotic Inspection of Underwater Structures
- **分类: cs.RO**

- **简介: 该论文属于水下多机器人任务分配问题，解决无先验几何信息下的任务生成与优化。通过分析SLAM数据生成任务，并优化关键点评分和距离剪枝，提升检测效率。**

- **链接: [https://arxiv.org/pdf/2602.02389v1](https://arxiv.org/pdf/2602.02389v1)**

> **作者:** Marina Ruediger; Ashis G. Banerjee
>
> **备注:** This paper will appear in the proceedings of the 2026 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Task generation for underwater multi-robot inspections without prior knowledge of existing geometry can be achieved and optimized through examination of simultaneous localization and mapping (SLAM) data. By considering hardware parameters and environmental conditions, a set of tasks is generated from SLAM meshes and optimized through expected keypoint scores and distance-based pruning. In-water tests are used to demonstrate the effectiveness of the algorithm and determine the appropriate parameters. These results are compared to simulated Voronoi partitions and boustrophedon patterns for inspection coverage on a model of the test environment. The key benefits of the presented task discovery method include adaptability to unexpected geometry and distributions that maintain coverage while focusing on areas more likely to present defects or damage.
>
---
#### [new 022] Ocean Current-Harnessing Stage-Gated MPC: Monotone Cost Shaping and Speed-to-Fly for Energy-Efficient AUV Navigation
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于AUV导航任务，旨在提升能源效率。通过引入阶段门控MPC，结合单调成本塑造和速度飞行策略，有效利用洋流降低能耗。**

- **链接: [https://arxiv.org/pdf/2602.00823v1](https://arxiv.org/pdf/2602.00823v1)**

> **作者:** Spyridon Syntakas; Kostas Vlachos
>
> **摘要:** Autonomous Underwater Vehicles (AUVs) are a highly promising technology for ocean exploration and diverse offshore operations, yet their practical deployment is constrained by energy efficiency and endurance. To address this, we propose Current-Harnessing Stage-Gated MPC, which exploits ocean currents via a per-stage scalar which indicates the "helpfulness" of ocean currents. This scalar is computed along the prediction horizon to gate lightweight cost terms only where the ocean currents truly aids the control goal. The proposed cost terms, that are merged in the objective function, are (i) a Monotone Cost Shaping (MCS) term, a help-gated, non-worsening modification that relaxes along-track position error and provides a bounded translational energy rebate, guaranteeing the shaped objective is never larger than a set baseline, and (ii) a speed-to-fly (STF) cost component that increases the price of thrust and softly matches ground velocity to the ocean current, enabling near zero water-relative "gliding". All terms are C1 and integrate as a plug-and-play in MPC designs. Extensive simulations with the BlueROV2 model under realistic ocean current fields show that the proposed approach achieves substantially lower energy consumption than conventional predictive control while maintaining comparable arrival times and constraint satisfaction.
>
---
#### [new 023] RoDiF: Robust Direct Fine-Tuning of Diffusion Policies with Corrupted Human Feedback
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，解决扩散策略微调中人类偏好被干扰的问题。提出RoDiF方法，通过几何视角优化目标，提升鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.00886v1](https://arxiv.org/pdf/2602.00886v1)**

> **作者:** Amitesh Vatsa; Zhixian Xie; Wanxin Jin
>
> **摘要:** Diffusion policies are a powerful paradigm for robotic control, but fine-tuning them with human preferences is fundamentally challenged by the multi-step structure of the denoising process. To overcome this, we introduce a Unified Markov Decision Process (MDP) formulation that coherently integrates the diffusion denoising chain with environmental dynamics, enabling reward-free Direct Preference Optimization (DPO) for diffusion policies. Building on this formulation, we propose RoDiF (Robust Direct Fine-Tuning), a method that explicitly addresses corrupted human preferences. RoDiF reinterprets the DPO objective through a geometric hypothesis-cutting perspective and employs a conservative cutting strategy to achieve robustness without assuming any specific noise distribution. Extensive experiments on long-horizon manipulation tasks show that RoDiF consistently outperforms state-of-the-art baselines, effectively steering pretrained diffusion policies of diverse architectures to human-preferred modes, while maintaining strong performance even under 30% corrupted preference labels.
>
---
#### [new 024] UniForce: A Unified Latent Force Model for Robot Manipulation with Diverse Tactile Sensors
- **分类: cs.RO**

- **简介: 该论文提出UniForce，解决机器人操作中多类型触觉传感器数据不统一的问题，通过学习共享的潜在力空间，实现跨传感器的力感知与任务迁移。**

- **链接: [https://arxiv.org/pdf/2602.01153v1](https://arxiv.org/pdf/2602.01153v1)**

> **作者:** Zhuo Chen; Fei Ni; Kaiyao Luo; Zhiyuan Wu; Xuyang Zhang; Emmanouil Spyrakos-Papastavridis; Lorenzo Jamone; Nathan F. Lepora; Jiankang Deng; Shan Luo
>
> **摘要:** Force sensing is essential for dexterous robot manipulation, but scaling force-aware policy learning is hindered by the heterogeneity of tactile sensors. Differences in sensing principles (e.g., optical vs. magnetic), form factors, and materials typically require sensor-specific data collection, calibration, and model training, thereby limiting generalisability. We propose UniForce, a novel unified tactile representation learning framework that learns a shared latent force space across diverse tactile sensors. UniForce reduces cross-sensor domain shift by jointly modeling inverse dynamics (image-to-force) and forward dynamics (force-to-image), constrained by force equilibrium and image reconstruction losses to produce force-grounded representations. To avoid reliance on expensive external force/torque (F/T) sensors, we exploit static equilibrium and collect force-paired data via direct sensor--object--sensor interactions, enabling cross-sensor alignment with contact force. The resulting universal tactile encoder can be plugged into downstream force-aware robot manipulation tasks with zero-shot transfer, without retraining or finetuning. Extensive experiments on heterogeneous tactile sensors including GelSight, TacTip, and uSkin, demonstrate consistent improvements in force estimation over prior methods, and enable effective cross-sensor coordination in Vision-Tactile-Language-Action (VTLA) models for a robotic wiping task. Code and datasets will be released.
>
---
#### [new 025] Co-Design of Rover Wheels and Control using Bayesian Optimization and Rover-Terrain Simulations
- **分类: cs.RO**

- **简介: 该论文属于机器人优化任务，解决越野车轮设计与控制协同优化问题。通过贝叶斯优化和高保真仿真，联合优化车轮参数和控制器，提升行驶性能。**

- **链接: [https://arxiv.org/pdf/2602.01535v1](https://arxiv.org/pdf/2602.01535v1)**

> **作者:** Huzaifa Mustafa Unjhawala; Khizar Shaikh; Luning Bakke; Radu Serban; Dan Negrut
>
> **备注:** 19 pages, 15 figures
>
> **摘要:** While simulation is vital for optimizing robotic systems, the cost of modeling deformable terrain has long limited its use in full-vehicle studies of off-road autonomous mobility. For example, Discrete Element Method (DEM) simulations are often confined to single-wheel tests, which obscures coupled wheel-vehicle-controller interactions and prevents joint optimization of mechanical design and control. This paper presents a Bayesian optimization framework that co-designs rover wheel geometry and steering controller parameters using high-fidelity, full-vehicle closed-loop simulations on deformable terrain. Using the efficiency and scalability of a continuum-representation model (CRM) for terramechanics, we evaluate candidate designs on trajectories of varying complexity while towing a fixed load. The optimizer tunes wheel parameters (radius, width, and grouser features) and steering PID gains under a multi-objective formulation that balances traversal speed, tracking error, and energy consumption. We compare two strategies: simultaneous co-optimization of wheel and controller parameters versus a sequential approach that decouples mechanical and control design. We analyze trade-offs in performance and computational cost. Across 3,000 full-vehicle simulations, campaigns finish in five to nine days, versus months with the group's earlier DEM-based workflow. Finally, a preliminary hardware study suggests the simulation-optimized wheel designs preserve relative performance trends on the physical rover. Together, these results show that scalable, high-fidelity simulation can enable practical co-optimization of wheel design and control for off-road vehicles on deformable terrain without relying on prohibitively expensive DEM studies. The simulation infrastructure (scripts and models) is released as open source in a public repository to support reproducibility and further research.
>
---
#### [new 026] World-Gymnast: Training Robots with Reinforcement Learning in a World Model
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人学习任务，旨在解决物理交互成本高和模拟到现实差距的问题。通过在世界模型中进行强化学习，提升机器人性能。**

- **链接: [https://arxiv.org/pdf/2602.02454v1](https://arxiv.org/pdf/2602.02454v1)**

> **作者:** Ansh Kumar Sharma; Yixiang Sun; Ninghao Lu; Yunzhe Zhang; Jiarao Liu; Sherry Yang
>
> **备注:** https://world-gymnast.github.io/
>
> **摘要:** Robot learning from interacting with the physical world is fundamentally bottlenecked by the cost of physical interaction. The two alternatives, supervised finetuning (SFT) from expert demonstrations and reinforcement learning (RL) in a software-based simulator, are limited by the amount of expert data available and the sim-to-real gap for manipulation. With the recent emergence of world models learned from real-world video-action data, we ask the question of whether training a policy in a world model can be more effective than supervised learning or software simulation in achieving better real-robot performance. We propose World-Gymnast, which performs RL finetuning of a vision-language-action (VLA) policy by rolling out the policy in an action-conditioned video world model and rewarding the rollouts with a vision-language model (VLM). On the Bridge robot setup, World-Gymnast outperforms SFT by as much as 18x and outperforms software simulator by as much as 2x. More importantly, World-Gymnast demonstrates intriguing capabilities of RL with a world model, including training on diverse language instructions and novel scenes from the world model, test-time training in a novel scene, and online iterative world model and policy improvement. Our results suggest learning a world model and training robot policies in the cloud could be the key to bridging the gap between robots that work in demonstrations and robots that can work in anyone's household.
>
---
#### [new 027] FISC: A Fluid-Inspired Framework for Decentralized and Scalable Swarm Control
- **分类: cs.RO**

- **简介: 该论文属于多智能体系统控制任务，旨在解决大规模机器人集群的可扩展协调问题。通过类流体框架实现去中心化控制，无需通信即可保持群体结构与一致性。**

- **链接: [https://arxiv.org/pdf/2602.00480v1](https://arxiv.org/pdf/2602.00480v1)**

> **作者:** Mohini Priya Kolluri; Ammar Waheed; Zohaib Hasnain
>
> **摘要:** Achieving scalable coordination in large robotic swarms is often constrained by reliance on inter-agent communication, which introduces latency, bandwidth limitations, and vulnerability to failure. To address this gap, a decentralized approach for outer-loop control of large multi-agent systems based on the paradigm of how a fluid moves through a volume is proposed and evaluated. A relationship between fundamental fluidic element properties and individual robotic agent states is developed such that the corresponding swarm "flows" through a space, akin to a fluid when forced via a pressure boundary condition. By ascribing fluid-like properties to subsets of agents, the swarm evolves collectively while maintaining desirable structure and coherence without explicit communication of agent states within or outside of the swarm. The approach is evaluated using simulations involving $O(10^3)$ quadcopter agents and compared against Computational Fluid Dynamics (CFD) solutions for a converging-diverging domain. Quantitative agreement between swarm-derived and CFD fields is assessed using Root-Mean-Square Error (RMSE), yielding normalized errors of 0.15-0.9 for velocity, 0.61-0.98 for density, 0-0.937 for pressure. These results demonstrate the feasibility of treating large robotic swarms as continuum systems that retain the macroscopic structure derived from first principles, providing a basis for scalable and decentralized control.
>
---
#### [new 028] Instance-Guided Unsupervised Domain Adaptation for Robotic Semantic Segmentation
- **分类: cs.RO**

- **简介: 该论文属于机器人语义分割任务，解决域适应问题。通过生成多视角一致伪标签并利用基础模型提升实例一致性，实现无监督模型微调，提升部署性能。**

- **链接: [https://arxiv.org/pdf/2602.01389v1](https://arxiv.org/pdf/2602.01389v1)**

> **作者:** Michele Antonazzi; Lorenzo Signorelli; Matteo Luperto; Nicola Basilico
>
> **备注:** Accepted for publication at ICRA 2026
>
> **摘要:** Semantic segmentation networks, which are essential for robotic perception, often suffer from performance degradation when the visual distribution of the deployment environment differs from that of the source dataset on which they were trained. Unsupervised Domain Adaptation (UDA) addresses this challenge by adapting the network to the robot's target environment without external supervision, leveraging the large amounts of data a robot might naturally collect during long-term operation. In such settings, UDA methods can exploit multi-view consistency across the environment's map to fine-tune the model in an unsupervised fashion and mitigate domain shift. However, these approaches remain sensitive to cross-view instance-level inconsistencies. In this work, we propose a method that starts from a volumetric 3D map to generate multi-view consistent pseudo-labels. We then refine these labels using the zero-shot instance segmentation capabilities of a foundation model, enforcing instance-level coherence. The refined annotations serve as supervision for self-supervised fine-tuning, enabling the robot to adapt its perception system at deployment time. Experiments on real-world data demonstrate that our approach consistently improves performance over state-of-the-art UDA baselines based on multi-view consistency, without requiring any ground-truth labels in the target domain.
>
---
#### [new 029] Agentic Reward Modeling: Verifying GUI Agent via Online Proactive Interaction
- **分类: cs.RO**

- **简介: 该论文属于GUI代理评估任务，旨在解决传统评估方法在可扩展性和状态感知上的不足。提出VAGEN框架，通过主动交互提升验证准确性。**

- **链接: [https://arxiv.org/pdf/2602.00575v1](https://arxiv.org/pdf/2602.00575v1)**

> **作者:** Chaoqun Cui; Jing Huang; Shijing Wang; Liming Zheng; Qingchao Kong; Zhixiong Zeng
>
> **备注:** 21 pages, 11 figures
>
> **摘要:** Reinforcement learning with verifiable rewards (RLVR) is pivotal for the continuous evolution of GUI agents, yet existing evaluation paradigms face significant limitations. Rule-based methods suffer from poor scalability and cannot handle open-ended tasks, while LLM-as-a-Judge approaches rely on passive visual observation, often failing to capture latent system states due to partial state observability. To address these challenges, we advocate for a paradigm shift from passive evaluation to Agentic Interactive Verification. We introduce VAGEN, a framework that employs a verifier agent equipped with interaction tools to autonomously plan verification strategies and proactively probe the environment for evidence of task completion. Leveraging the insight that GUI tasks are typically "easy to verify but hard to solve", VAGEN overcomes the bottlenecks of visual limitations. Experimental results on OSWorld-Verified and AndroidWorld benchmarks demonstrate that VAGEN significantly improves evaluation accuracy compared to LLM-as-a-Judge baselines and further enhances performance through test-time scaling strategies.
>
---
#### [new 030] Multi-Task Learning for Robot Perception with Imbalanced Data
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于多任务学习领域，解决机器人感知中数据不平衡的问题。提出一种无需所有任务真实标签的方法，并分析任务间交互影响。**

- **链接: [https://arxiv.org/pdf/2602.01899v1](https://arxiv.org/pdf/2602.01899v1)**

> **作者:** Ozgur Erkent
>
> **备注:** 16 pages
>
> **摘要:** Multi-task problem solving has been shown to improve the accuracy of the individual tasks, which is an important feature for robots, as they have a limited resource. However, when the number of labels for each task is not equal, namely imbalanced data exist, a problem may arise due to insufficient number of samples, and labeling is not very easy for mobile robots in every environment. We propose a method that can learn tasks even in the absence of the ground truth labels for some of the tasks. We also provide a detailed analysis of the proposed method. An interesting finding is related to the interaction of the tasks. We show a methodology to find out which tasks can improve the performance of other tasks. We investigate this by training the teacher network with the task outputs such as depth as inputs. We further provide empirical evidence when trained with a small amount of data. We use semantic segmentation and depth estimation tasks on different datasets, NYUDv2 and Cityscapes.
>
---
#### [new 031] UniMotion: A Unified Motion Framework for Simulation, Prediction and Planning
- **分类: cs.RO**

- **简介: 该论文提出UniMotion，一个统一的运动框架，解决自动驾驶中运动模拟、预测与规划任务的协同问题，通过共享结构提升性能与可扩展性。**

- **链接: [https://arxiv.org/pdf/2602.00566v1](https://arxiv.org/pdf/2602.00566v1)**

> **作者:** Nan Song; Junzhe Jiang; Jingyu Li; Xiatian Zhu; Li Zhang
>
> **备注:** Accepted at NeurIPS 2025
>
> **摘要:** Motion simulation, prediction and planning are foundational tasks in autonomous driving, each essential for modeling and reasoning about dynamic traffic scenarios. While often addressed in isolation due to their differing objectives, such as generating diverse motion states or estimating optimal trajectories, these tasks inherently depend on shared capabilities: understanding multi-agent interactions, modeling motion behaviors, and reasoning over temporal and spatial dynamics. Despite this underlying commonality, existing approaches typically adopt specialized model designs, which hinders cross-task generalization and system scalability. More critically, this separation overlooks the potential mutual benefits among tasks. Motivated by these observations, we propose UniMotion, a unified motion framework that captures shared structures across motion tasks while accommodating their individual requirements. Built on a decoder-only Transformer architecture, UniMotion employs dedicated interaction modes and tailored training strategies to simultaneously support these motion tasks. This unified design not only enables joint optimization and representation sharing but also allows for targeted fine-tuning to specialize in individual tasks when needed. Extensive experiments on the Waymo Open Motion Dataset demonstrate that joint training leads to robust generalization and effective task integration. With further fine-tuning, UniMotion achieves state-of-the-art performance across a range of motion tasks, establishing it as a versatile and scalable solution for autonomous driving.
>
---
#### [new 032] Reformulating AI-based Multi-Object Relative State Estimation for Aleatoric Uncertainty-based Outlier Rejection of Partial Measurements
- **分类: cs.RO**

- **简介: 该论文属于机器人定位任务，解决AI测量中的不确定性与异常值问题。通过重构测量方程，提升状态估计的精度与一致性。**

- **链接: [https://arxiv.org/pdf/2602.02006v1](https://arxiv.org/pdf/2602.02006v1)**

> **作者:** Thomas Jantos; Giulio Delama; Stephan Weiss; Jan Steinbrener
>
> **备注:** Accepted for publication at ICRA 2026, Vienna, Austria
>
> **摘要:** Precise localization with respect to a set of objects of interest enables mobile robots to perform various tasks. With the rise of edge devices capable of deploying deep neural networks (DNNs) for real-time inference, it stands to reason to use artificial intelligence (AI) for the extraction of object-specific, semantic information from raw image data, such as the object class and the relative six degrees of freedom (6-DoF) pose. However, fusing such AI-based measurements in an Extended Kalman Filter (EKF) requires quantifying the DNNs' uncertainty and outlier rejection capabilities. This paper presents the benefits of reformulating the measurement equation in AI-based, object-relative state estimation. By deriving an EKF using the direct object-relative pose measurement, we can decouple the position and rotation measurements, thus limiting the influence of erroneous rotation measurements and allowing partial measurement rejection. Furthermore, we investigate the performance and consistency improvements for state estimators provided by replacing the fixed measurement covariance matrix of the 6-DoF object-relative pose measurements with the predicted aleatoric uncertainty of the DNN.
>
---
#### [new 033] Online Fine-Tuning of Pretrained Controllers for Autonomous Driving via Real-Time Recurrent RL
- **分类: cs.RO; cs.LG; cs.NE; eess.SY**

- **简介: 该论文属于自主驾驶任务，解决预训练控制器在环境变化下的性能下降问题。通过实时循环强化学习（RTRRL）和液态电阻电容RNN模型，实现在线微调，提升驾驶性能。**

- **链接: [https://arxiv.org/pdf/2602.02236v1](https://arxiv.org/pdf/2602.02236v1)**

> **作者:** Julian Lemmel; Felix Resch; Mónika Farsang; Ramin Hasani; Daniela Rus; Radu Grosu
>
> **摘要:** Deploying pretrained policies in real-world applications presents substantial challenges that fundamentally limit the practical applicability of learning-based control systems. When autonomous systems encounter environmental changes in system dynamics, sensor drift, or task objectives, fixed policies rapidly degrade in performance. We show that employing Real-Time Recurrent Reinforcement Learning (RTRRL), a biologically plausible algorithm for online adaptation, can effectively fine-tune a pretrained policy to improve autonomous agents' performance on driving tasks. We further show that RTRRL synergizes with a recent biologically inspired recurrent network model, the Liquid-Resistance Liquid-Capacitance RNN. We demonstrate the effectiveness of this closed-loop approach in a simulated CarRacing environment and in a real-world line-following task with a RoboRacer car equipped with an event camera.
>
---
#### [new 034] Offline Discovery of Interpretable Skills from Multi-Task Trajectories
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多任务轨迹中的可解释技能发现任务，旨在从无奖励、无标注的离线数据中自动提取有用技能。工作包括设计三阶段框架LOKI，实现技能分割与层次模仿学习。**

- **链接: [https://arxiv.org/pdf/2602.01018v1](https://arxiv.org/pdf/2602.01018v1)**

> **作者:** Chongyu Zhu; Mithun Vanniasinghe; Jiayu Chen; Chi-Guhn Lee
>
> **摘要:** Hierarchical Imitation Learning is a powerful paradigm for acquiring complex robot behaviors from demonstrations. A central challenge, however, lies in discovering reusable skills from long-horizon, multi-task offline data, especially when the data lacks explicit rewards or subtask annotations. In this work, we introduce LOKI, a three-stage end-to-end learning framework designed for offline skill discovery and hierarchical imitation. The framework commences with a two-stage, weakly supervised skill discovery process: Stage one performs coarse, task-aware macro-segmentation by employing an alignment-enforced Vector Quantized VAE guided by weak task labels. Stage two then refines these segments at a micro-level using a self-supervised sequential model, followed by an iterative clustering process to consolidate skill boundaries. The third stage then leverages these precise boundaries to construct a hierarchical policy within an option-based framework-complete with a learned termination condition beta for explicit skill switching. LOKI achieves high success rates on the challenging D4RL Kitchen benchmark and outperforms standard HIL baselines. Furthermore, we demonstrate that the discovered skills are semantically meaningful, aligning with human intuition, and exhibit compositionality by successfully sequencing them to solve a novel, unseen task.
>
---
#### [new 035] SanD-Planner: Sample-Efficient Diffusion Planner in B-Spline Space for Robust Local Navigation
- **分类: cs.RO**

- **简介: 该论文提出SanD-Planner，解决复杂环境中高效局部路径规划问题。通过扩散模型与B样条空间结合，提升样本效率和规划可靠性。**

- **链接: [https://arxiv.org/pdf/2602.00923v1](https://arxiv.org/pdf/2602.00923v1)**

> **作者:** Jincheng Wang; Lingfan Bao; Tong Yang; Diego Martinez Plasencia; Jianhao Jiao; Dimitrios Kanoulas
>
> **备注:** Under review. 11 pages
>
> **摘要:** The challenge of generating reliable local plans has long hindered practical applications in highly cluttered and dynamic environments. Key fundamental bottlenecks include acquiring large-scale expert demonstrations across diverse scenes and improving learning efficiency with limited data. This paper proposes SanD-Planner, a sample-efficient diffusion-based local planner that conducts depth image-based imitation learning within the clamped B-spline space. By operating within this compact space, the proposed algorithm inherently yields smooth outputs with bounded prediction errors over local supports, naturally aligning with receding-horizon execution. Integration of an ESDF-based safety checker with explicit clearance and time-to-completion metrics further reduces the training burden associated with value-function learning for feasibility assessment. Experiments show that training with $500$ episodes (merely $0.25\%$ of the demonstration scale used by the baseline), SanD-Planner achieves state-of-the-art performance on the evaluated open benchmark, attaining success rates of $90.1\%$ in simulated cluttered environments and $72.0\%$ in indoor simulations. The performance is further proven by demonstrating zero-shot transferability to realistic experimentation in both 2D and 3D scenes. The dataset and pre-trained models will also be open-sourced.
>
---
#### [new 036] APEX: A Decoupled Memory-based Explorer for Asynchronous Aerial Object Goal Navigation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于无人机目标导航任务，解决空中环境中的记忆、决策和探索效率问题。提出APEX框架，通过分层异步设计提升导航性能。**

- **链接: [https://arxiv.org/pdf/2602.00551v1](https://arxiv.org/pdf/2602.00551v1)**

> **作者:** Daoxuan Zhang; Ping Chen; Xiaobo Xia; Xiu Su; Ruichen Zhen; Jianqiang Xiao; Shuo Yang
>
> **备注:** 15 pages, 8 figures
>
> **摘要:** Aerial Object Goal Navigation, a challenging frontier in Embodied AI, requires an Unmanned Aerial Vehicle (UAV) agent to autonomously explore, reason, and identify a specific target using only visual perception and language description. However, existing methods struggle with the memorization of complex spatial representations in aerial environments, reliable and interpretable action decision-making, and inefficient exploration and information gathering. To address these challenges, we introduce \textbf{APEX} (Aerial Parallel Explorer), a novel hierarchical agent designed for efficient exploration and target acquisition in complex aerial settings. APEX is built upon a modular, three-part architecture: 1) Dynamic Spatio-Semantic Mapping Memory, which leverages the zero-shot capability of a Vision-Language Model (VLM) to dynamically construct high-resolution 3D Attraction, Exploration, and Obstacle maps, serving as an interpretable memory mechanism. 2) Action Decision Module, trained with reinforcement learning, which translates this rich spatial understanding into a fine-grained and robust control policy. 3) Target Grounding Module, which employs an open-vocabulary detector to achieve definitive and generalizable target identification. All these components are integrated into a hierarchical, asynchronous, and parallel framework, effectively bypassing the VLM's inference latency and boosting the agent's proactivity in exploration. Extensive experiments show that APEX outperforms the previous state of the art by +4.2\% SR and +2.8\% SPL on challenging UAV-ON benchmarks, demonstrating its superior efficiency and the effectiveness of its hierarchical asynchronous design. Our source code is provided in \href{https://github.com/4amGodvzx/apex}{GitHub}
>
---
#### [new 037] FD-VLA: Force-Distilled Vision-Language-Action Model for Contact-Rich Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出FD-VLA，解决接触密集操作中缺乏力传感器的问题。通过力蒸馏模块，将力信息融入视觉-语言-动作框架，提升操作鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.02142v1](https://arxiv.org/pdf/2602.02142v1)**

> **作者:** Ruiteng Zhao; Wenshuo Wang; Yicheng Ma; Xiaocong Li; Francis E. H. Tay; Marcelo H. Ang; Haiyue Zhu
>
> **摘要:** Force sensing is a crucial modality for Vision-Language-Action (VLA) frameworks, as it enables fine-grained perception and dexterous manipulation in contact-rich tasks. We present Force-Distilled VLA (FD-VLA), a novel framework that integrates force awareness into contact-rich manipulation without relying on physical force sensors. The core of our approach is a Force Distillation Module (FDM), which distills force by mapping a learnable query token, conditioned on visual observations and robot states, into a predicted force token aligned with the latent representation of actual force signals. During inference, this distilled force token is injected into the pretrained VLM, enabling force-aware reasoning while preserving the integrity of its vision-language semantics. This design provides two key benefits: first, it allows practical deployment across a wide range of robots that lack expensive or fragile force-torque sensors, thereby reducing hardware cost and complexity; second, the FDM introduces an additional force-vision-state fusion prior to the VLM, which improves cross-modal alignment and enhances perception-action robustness in contact-rich scenarios. Surprisingly, our physical experiments show that the distilled force token outperforms direct sensor force measurements as well as other baselines, which highlights the effectiveness of this force-distilled VLA approach.
>
---
#### [new 038] Latent Reasoning VLA: Latent Thinking and Prediction for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文提出LaRA-VLA，解决VLA模型推理效率低和表示不匹配问题，通过连续潜空间实现高效、实时的视觉-语言-动作控制。**

- **链接: [https://arxiv.org/pdf/2602.01166v1](https://arxiv.org/pdf/2602.01166v1)**

> **作者:** Shuanghao Bai; Jing Lyu; Wanqi Zhou; Zhe Li; Dakai Wang; Lei Xing; Xiaoguang Zhao; Pengwei Wang; Zhongyuan Wang; Cheng Chi; Badong Chen; Shanghang Zhang
>
> **摘要:** Vision-Language-Action (VLA) models benefit from chain-of-thought (CoT) reasoning, but existing approaches incur high inference overhead and rely on discrete reasoning representations that mismatch continuous perception and control. We propose Latent Reasoning VLA (\textbf{LaRA-VLA}), a unified VLA framework that internalizes multi-modal CoT reasoning into continuous latent representations for embodied action. LaRA-VLA performs unified reasoning and prediction in latent space, eliminating explicit CoT generation at inference time and enabling efficient, action-oriented control. To realize latent embodied reasoning, we introduce a curriculum-based training paradigm that progressively transitions from explicit textual and visual CoT supervision to latent reasoning, and finally adapts latent reasoning dynamics to condition action generation. We construct two structured CoT datasets and evaluate LaRA-VLA on both simulation benchmarks and long-horizon real-robot manipulation tasks. Experimental results show that LaRA-VLA consistently outperforms state-of-the-art VLA methods while reducing inference latency by up to 90\% compared to explicit CoT-based approaches, demonstrating latent reasoning as an effective and efficient paradigm for real-time embodied control. Project Page: \href{https://loveju1y.github.io/Latent-Reasoning-VLA/}{LaRA-VLA Website}.
>
---
#### [new 039] ForSim: Stepwise Forward Simulation for Traffic Policy Fine-Tuning
- **分类: cs.RO**

- **简介: 该论文属于交通仿真任务，旨在解决仿真中行为多样性不足与交互不真实的问题。提出ForSim方法，通过闭环模拟提升仿真真实性和安全性。**

- **链接: [https://arxiv.org/pdf/2602.01916v1](https://arxiv.org/pdf/2602.01916v1)**

> **作者:** Keyu Chen; Wenchao Sun; Hao Cheng; Zheng Fu; Sifa Zheng
>
> **备注:** Accepted by ICRA 2026
>
> **摘要:** As the foundation of closed-loop training and evaluation in autonomous driving, traffic simulation still faces two fundamental challenges: covariate shift introduced by open-loop imitation learning and limited capacity to reflect the multimodal behaviors observed in real-world traffic. Although recent frameworks such as RIFT have partially addressed these issues through group-relative optimization, their forward simulation procedures remain largely non-reactive, leading to unrealistic agent interactions within the virtual domain and ultimately limiting simulation fidelity. To address these issues, we propose ForSim, a stepwise closed-loop forward simulation paradigm. At each virtual timestep, the traffic agent propagates the virtual candidate trajectory that best spatiotemporally matches the reference trajectory through physically grounded motion dynamics, thereby preserving multimodal behavioral diversity while ensuring intra-modality consistency. Other agents are updated with stepwise predictions, yielding coherent and interaction-aware evolution. When incorporated into the RIFT traffic simulation framework, ForSim operates in conjunction with group-relative optimization to fine-tune traffic policy. Extensive experiments confirm that this integration consistently improves safety while maintaining efficiency, realism, and comfort. These results underscore the importance of modeling closed-loop multimodal interactions within forward simulation and enhance the fidelity and reliability of traffic simulation for autonomous driving. Project Page: https://currychen77.github.io/ForSim/
>
---
#### [new 040] 3D Foundation Model-Based Loop Closing for Decentralized Collaborative SLAM
- **分类: cs.RO**

- **简介: 该论文属于多机器人协同SLAM任务，解决因视角差异导致的地图重叠识别问题。通过引入3D基础模型，实现可靠的相对位姿估计与优化，提升定位与建图精度及效率。**

- **链接: [https://arxiv.org/pdf/2602.02430v1](https://arxiv.org/pdf/2602.02430v1)**

> **作者:** Pierre-Yves Lajoie; Benjamin Ramtoula; Daniele De Martini; Giovanni Beltrame
>
> **摘要:** Decentralized Collaborative Simultaneous Localization And Mapping (C-SLAM) techniques often struggle to identify map overlaps due to significant viewpoint variations among robots. Motivated by recent advancements in 3D foundation models, which can register images despite large viewpoint differences, we propose a robust loop closing approach that leverages these models to establish inter-robot measurements. In contrast to resource-intensive methods requiring full 3D reconstruction within a centralized map, our approach integrates foundation models into existing SLAM pipelines, yielding scalable and robust multi-robot mapping. Our contributions include: (1) integrating 3D foundation models to reliably estimate relative poses from monocular image pairs within decentralized C-SLAM; (2) introducing robust outlier mitigation techniques critical to the use of these relative poses; and (3) developing specialized pose graph optimization formulations that efficiently resolve scale ambiguities. We evaluate our method against state-of-the-art approaches, demonstrating improvements in localization and mapping accuracy, alongside significant gains in computational and memory efficiency. These results highlight the potential of our approach for deployment in large-scale multi-robot scenarios.
>
---
#### [new 041] Failure-Aware Bimanual Teleoperation via Conservative Value Guided Assistance
- **分类: cs.RO**

- **简介: 该论文属于双臂遥操作任务，旨在解决操作中潜在故障难以预测的问题。通过保守价值学习构建成功评分，提供安全辅助，提升任务成功率并降低操作负担。**

- **链接: [https://arxiv.org/pdf/2602.01092v1](https://arxiv.org/pdf/2602.01092v1)**

> **作者:** Peng Zhou; Zhongxuan Li; Jinsong Wu; Jiaming Qi; Jun Hu; David Navarro-Alarcon; Jia Pan; Lihua Xie; Shiyao Zhang; Zeqing Zhang
>
> **摘要:** Teleoperation of high-precision manipulation is con-strained by tight success tolerances and complex contact dy-namics, which make impending failures difficult for human operators to anticipate under partial observability. This paper proposes a value-guided, failure-aware framework for bimanual teleoperation that provides compliant haptic assistance while pre-serving continuous human authority. The framework is trained entirely from heterogeneous offline teleoperation data containing both successful and failed executions. Task feasibility is mod-eled as a conservative success score learned via Conservative Value Learning, yielding a risk-sensitive estimate that remains reliable under distribution shift. During online operation, the learned success score regulates the level of assistance, while a learned actor provides a corrective motion direction. Both are integrated through a joint-space impedance interface on the master side, yielding continuous guidance that steers the operator away from failure-prone actions without overriding intent. Experimental results on contact-rich manipulation tasks demonstrate improved task success rates and reduced operator workload compared to conventional teleoperation and shared-autonomy baselines, indicating that conservative value learning provides an effective mechanism for embedding failure awareness into bilateral teleoperation. Experimental videos are available at https://www.youtube.com/watch?v=XDTsvzEkDRE
>
---
#### [new 042] Learning Adaptive Cross-Embodiment Visuomotor Policy with Contrastive Prompt Orchestration
- **分类: cs.RO**

- **简介: 该论文属于视觉运动策略学习任务，解决跨身体配置的适应性问题。提出CAPO方法，通过对比提示学习和自适应提示编排，提升策略在不同环境中的泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.01040v1](https://arxiv.org/pdf/2602.01040v1)**

> **作者:** Yuhang Zhang; Chao Yan; Jiaxi Yu; Jiaping Xiao; Mir Feroskhan
>
> **摘要:** Learning adaptive visuomotor policies for embodied agents remains a formidable challenge, particularly when facing cross-embodiment variations such as diverse sensor configurations and dynamic properties. Conventional learning approaches often struggle to separate task-relevant features from domain-specific variations (e.g., lighting, field-of-view, and rotation), leading to poor sample efficiency and catastrophic failure in unseen environments. To bridge this gap, we propose ContrAstive Prompt Orchestration (CAPO), a novel approach for learning visuomotor policies that integrates contrastive prompt learning and adaptive prompt orchestration. For prompt learning, we devise a hybrid contrastive learning strategy that integrates visual, temporal action, and text objectives to establish a pool of learnable prompts, where each prompt induces a visual representation encapsulating fine-grained domain factors. Based on these learned prompts, we introduce an adaptive prompt orchestration mechanism that dynamically aggregates these prompts conditioned on current observations. This enables the agent to adaptively construct optimal state representations by identifying dominant domain factors instantaneously. Consequently, the policy optimization is effectively shielded from irrelevant interference, preventing the common issue of overfitting to source domains. Extensive experiments demonstrate that CAPO significantly outperforms state-of-the-art baselines in sample efficiency and asymptotic performance. Crucially, it exhibits superior zero-shot adaptation across unseen target domains characterized by drastic environmental (e.g., illumination) and physical shifts (e.g., field-of-view and rotation), validating its effectiveness as a viable solution for cross-embodiment visuomotor policy adaptation.
>
---
#### [new 043] CLAMP: Contrastive Learning for 3D Multi-View Action-Conditioned Robotic Manipulation Pretraining
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文提出CLAMP框架，解决机器人操作中3D空间信息不足的问题。通过对比学习和点云数据预训练，提升操作精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.00937v1](https://arxiv.org/pdf/2602.00937v1)**

> **作者:** I-Chun Arthur Liu; Krzysztof Choromanski; Sandy Huang; Connor Schenck
>
> **摘要:** Leveraging pre-trained 2D image representations in behavior cloning policies has achieved great success and has become a standard approach for robotic manipulation. However, such representations fail to capture the 3D spatial information about objects and scenes that is essential for precise manipulation. In this work, we introduce Contrastive Learning for 3D Multi-View Action-Conditioned Robotic Manipulation Pretraining (CLAMP), a novel 3D pre-training framework that utilizes point clouds and robot actions. From the merged point cloud computed from RGB-D images and camera extrinsics, we re-render multi-view four-channel image observations with depth and 3D coordinates, including dynamic wrist views, to provide clearer views of target objects for high-precision manipulation tasks. The pre-trained encoders learn to associate the 3D geometric and positional information of objects with robot action patterns via contrastive learning on large-scale simulated robot trajectories. During encoder pre-training, we pre-train a Diffusion Policy to initialize the policy weights for fine-tuning, which is essential for improving fine-tuning sample efficiency and performance. After pre-training, we fine-tune the policy on a limited amount of task demonstrations using the learned image and action representations. We demonstrate that this pre-training and fine-tuning design substantially improves learning efficiency and policy performance on unseen tasks. Furthermore, we show that CLAMP outperforms state-of-the-art baselines across six simulated tasks and five real-world tasks.
>
---
#### [new 044] Multi-Agent Monte Carlo Tree Search for Makespan-Efficient Object Rearrangement in Cluttered Spaces
- **分类: cs.RO**

- **简介: 该论文研究多智能体在杂乱环境中高效物体重排任务，解决非单调任务下的协作效率问题，提出CAM-MCTS框架以减少完成时间。**

- **链接: [https://arxiv.org/pdf/2602.02411v1](https://arxiv.org/pdf/2602.02411v1)**

> **作者:** Hanwen Ren; Junyong Kim; Aathman Tharmasanthiran; Ahmed H. Qureshi
>
> **摘要:** Object rearrangement planning in complex, cluttered environments is a common challenge in warehouses, households, and rescue sites. Prior studies largely address monotone instances, whereas real-world tasks are often non-monotone-objects block one another and must be temporarily relocated to intermediate positions before reaching their final goals. In such settings, effective multi-agent collaboration can substantially reduce the time required to complete tasks. This paper introduces Centralized, Asynchronous, Multi-agent Monte Carlo Tree Search (CAM-MCTS), a novel framework for general-purpose makespan-efficient object rearrangement planning in challenging environments. CAM-MCTS combines centralized task assignment-where agents remain aware of each other's intended actions to facilitate globally optimized planning-with an asynchronous task execution strategy that enables agents to take on new tasks at appropriate time steps, rather than waiting for others, guided by a one-step look-ahead cost estimate. This design minimizes idle time, prevents unnecessary synchronization delays, and enhances overall system efficiency. We evaluate CAM-MCTS across a diverse set of monotone and non-monotone tasks in cluttered environments, demonstrating consistent reductions in makespan compared to strong baselines. Finally, we validate our approach on a real-world multi-agent system under different configurations, further confirming its effectiveness and robustness.
>
---
#### [new 045] Towards a Novel Wearable Robotic Vest for Hemorrhage Suppression
- **分类: cs.RO**

- **简介: 该论文属于医疗救援任务，旨在解决严重出血的紧急处理问题。设计了一种可变形的可穿戴机械背心，通过调整形状和压力来覆盖不同部位伤口。**

- **链接: [https://arxiv.org/pdf/2602.01448v1](https://arxiv.org/pdf/2602.01448v1)**

> **作者:** Harshith Jella; Pejman Kheradmand; Joseph Klein; Behnam Moradkhani; Yash Chitalia
>
> **摘要:** This paper introduces a novel robotic system designed to manage severe bleeding in emergency scenarios, including unique environments like space stations. The robot features a shape-adjustable "ring mechanism", transitioning from a circular to an elliptical configuration to adjust wound coverage across various anatomical regions. We developed various arms for this ring mechanism with varying flexibilities to improve adaptability when applied to non-extremities of the body (abdomen, back, neck, etc.). To apply equal and constant pressure across the wound, we developed an inflatable ring and airbag balloon that are compatible with this shape-changing ring mechanism. A series of experiments focused on evaluating various ring arm configurations to characterize their bending stiffness. Subsequent experiments measured the force exerted by the airbag balloon system using a digital scale. Despite its promising performance, certain limitations related to coverage area are identified. The shape-changing effect of the device is limited to scenarios involving partially inflated or deflated airbag balloons, and cannot fully conform to complex anatomical regions. Finally, the device was tested on casualty simulation kits, where it successfully demonstrated its ability to control simulated bleeding.
>
---
#### [new 046] Multimodal Large Language Models for Real-Time Situated Reasoning
- **分类: cs.RO**

- **简介: 该论文研究多模态大语言模型在实时情境推理中的应用，解决机器人决策与价值观对齐问题，通过结合GPT-4o与机器人平台实现环境感知与合理决策。**

- **链接: [https://arxiv.org/pdf/2602.01880v1](https://arxiv.org/pdf/2602.01880v1)**

> **作者:** Giulio Antonio Abbo; Senne Lenaerts; Tony Belpaeme
>
> **备注:** Submitted to the interactivity track of the 21st ACM/IEEE International Conference on Human-Robot Interaction on December 2025, accepted January 2026
>
> **摘要:** In this work, we explore how multimodal large language models can support real-time context- and value-aware decision-making. To do so, we combine the GPT-4o language model with a TurtleBot 4 platform simulating a smart vacuum cleaning robot in a home. The model evaluates the environment through vision input and determines whether it is appropriate to initiate cleaning. The system highlights the ability of these models to reason about domestic activities, social norms, and user preferences and take nuanced decisions aligned with the values of the people involved, such as cleanliness, comfort, and safety. We demonstrate the system in a realistic home environment, showing its ability to infer context and values from limited visual input. Our results highlight the promise of multimodal large language models in enhancing robotic autonomy and situational awareness, while also underscoring challenges related to consistency, bias, and real-time performance.
>
---
#### [new 047] TreeLoc: 6-DoF LiDAR Global Localization in Forests via Inter-Tree Geometric Matching
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出TreeLoc，用于森林中基于LiDAR的6-DoF全局定位。解决GPS失效环境下森林场景的定位问题，通过树干特征进行匹配与姿态估计。**

- **链接: [https://arxiv.org/pdf/2602.01501v1](https://arxiv.org/pdf/2602.01501v1)**

> **作者:** Minwoo Jung; Nived Chebrolu; Lucas Carvalho de Lima; Haedam Oh; Maurice Fallon; Ayoung Kim
>
> **备注:** An 8-page paper with 7 tables and 8 figures, accepted to ICRA 2026
>
> **摘要:** Reliable localization is crucial for navigation in forests, where GPS is often degraded and LiDAR measurements are repetitive, occluded, and structurally complex. These conditions weaken the assumptions of traditional urban-centric localization methods, which assume that consistent features arise from unique structural patterns, necessitating forest-centric solutions to achieve robustness in these environments. To address these challenges, we propose TreeLoc, a LiDAR-based global localization framework for forests that handles place recognition and 6-DoF pose estimation. We represent scenes using tree stems and their Diameter at Breast Height (DBH), which are aligned to a common reference frame via their axes and summarized using the tree distribution histogram (TDH) for coarse matching, followed by fine matching with a 2D triangle descriptor. Finally, pose estimation is achieved through a two-step geometric verification. On diverse forest benchmarks, TreeLoc outperforms baselines, achieving precise localization. Ablation studies validate the contribution of each component. We also propose applications for long-term forest management using descriptors from a compact global tree database. TreeLoc is open-sourced for the robotics community at https://github.com/minwoo0611/TreeLoc.
>
---
#### [new 048] Inject Once Survive Later: Backdooring Vision-Language-Action Models to Persist Through Downstream Fine-tuning
- **分类: cs.RO**

- **简介: 该论文属于安全领域，针对VLA模型在微调后背门失效的问题，提出INFUSE框架，使背门在微调后仍有效，保障模型安全性。**

- **链接: [https://arxiv.org/pdf/2602.00500v1](https://arxiv.org/pdf/2602.00500v1)**

> **作者:** Jianyi Zhou; Yujie Wei; Ruichen Zhen; Bo Zhao; Xiaobo Xia; Rui Shao; Xiu Su; Shuo Yang
>
> **摘要:** Vision-Language-Action (VLA) models have become foundational to modern embodied AI systems. By integrating visual perception, language understanding, and action planning, they enable general-purpose task execution across diverse environments. Despite their importance, the security of VLA models remains underexplored -- particularly in the context of backdoor attacks, which pose realistic threats in physical-world deployments. While recent methods attempt to inject backdoors into VLA models, these backdoors are easily erased during downstream adaptation, as user-side fine-tuning with clean data significantly alters model parameters, rendering them impractical for real-world applications. To address these challenges, we propose INFUSE (INjection into Fine-tUne-inSensitive modulEs), the first backdoor attack framework for VLA base models that remains effective even with arbitrary user fine-tuning. INFUSE begins by analyzing parameter sensitivity across diverse fine-tuning scenarios to identify modules that remain largely unchanged -- the fine-tune-insensitive modules. It then injects backdoors into these stable modules while freezing the rest, ensuring malicious behavior persists after extensive user fine-tuning. Comprehensive experiments across multiple VLA architectures demonstrate INFUSE's effectiveness. After user-side fine-tuning, INFUSE maintains mean attack success rates of 91.0% on simulation environments and 79.8% on real-world robot tasks, substantially surpassing BadVLA (38.8% and 36.6%, respectively), while preserving clean-task performance comparable to standard models. These results uncover a critical threat: backdoors implanted before distribution can persist through fine-tuning and remain effective at deployment.
>
---
#### [new 049] Learning When to Jump for Off-road Navigation
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决越野行驶中速度与安全的平衡问题。通过引入MAT表示，建模地形成本与速度的关系，提升导航效率与安全性。**

- **链接: [https://arxiv.org/pdf/2602.00877v1](https://arxiv.org/pdf/2602.00877v1)**

> **作者:** Zhipeng Zhao; Taimeng Fu; Shaoshu Su; Qiwei Du; Ehsan Tarkesh Esfahani; Karthik Dantu; Souma Chowdhury; Chen Wang
>
> **摘要:** Low speed does not always guarantee safety in off-road driving. For instance, crossing a ditch may be risky at a low speed due to the risk of getting stuck, yet safe at a higher speed with a controlled, accelerated jump. Achieving such behavior requires path planning that explicitly models complex motion dynamics, whereas existing methods often neglect this aspect and plan solely based on positions or a fixed velocity. To address this gap, we introduce Motion-aware Traversability (MAT) representation to explicitly model terrain cost conditioned on actual robot motion. Instead of assigning a single scalar score for traversability, MAT models each terrain region as a Gaussian function of velocity. During online planning, we decompose the terrain cost computation into two stages: (1) predict terrain-dependent Gaussian parameters from perception in a single forward pass, (2) efficiently update terrain costs for new velocities inferred from current dynamics by evaluating these functions without repeated inference. We develop a system that integrates MAT to enable agile off-road navigation and evaluate it in both simulated and real-world environments with various obstacles. Results show that MAT achieves real-time efficiency and enhances the performance of off-road navigation, reducing path detours by 75% while maintaining safety across challenging terrains.
>
---
#### [new 050] From Knowing to Doing Precisely: A General Self-Correction and Termination Framework for VLA models
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决抓取偏差和任务完成识别问题。提出VLA-SCT框架，通过自校正和终止机制提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2602.01811v1](https://arxiv.org/pdf/2602.01811v1)**

> **作者:** Wentao Zhang; Aolan Sun; Wentao Mo; Xiaoyang Qu; Yuxin Zheng; Jianzong Wang
>
> **备注:** Accepted to 2026 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP 2026)
>
> **摘要:** While vision-language-action (VLA) models for embodied agents integrate perception, reasoning, and control, they remain constrained by two critical weaknesses: first, during grasping tasks, the action tokens generated by the language model often exhibit subtle spatial deviations from the target object, resulting in grasp failures; second, they lack the ability to reliably recognize task completion, which leads to redundant actions and frequent timeout errors. To address these challenges and enhance robustness, we propose a lightweight, training-free framework, VLA-SCT. This framework operates as a self-correcting control loop, combining data-driven action refinement with conditional logic for termination. Consequently, compared to baseline approaches, our method achieves consistent improvements across all datasets in the LIBERO benchmark, significantly increasing the success rate of fine manipulation tasks and ensuring accurate task completion, thereby promoting the deployment of more reliable VLA agents in complex, unstructured environments.
>
---
#### [new 051] BTGenBot-2: Efficient Behavior Tree Generation with Small Language Models
- **分类: cs.RO**

- **简介: 该论文属于机器人任务规划领域，解决LLM在机器人部署中的效率与通用性问题。提出BTGenBot-2模型，实现自然语言到行为树的高效生成与错误恢复。**

- **链接: [https://arxiv.org/pdf/2602.01870v1](https://arxiv.org/pdf/2602.01870v1)**

> **作者:** Riccardo Andrea Izzo; Gianluca Bardaro; Matteo Matteucci
>
> **摘要:** Recent advances in robot learning increasingly rely on LLM-based task planning, leveraging their ability to bridge natural language with executable actions. While prior works showcased great performances, the widespread adoption of these models in robotics has been challenging as 1) existing methods are often closed-source or computationally intensive, neglecting the actual deployment on real-world physical systems, and 2) there is no universally accepted, plug-and-play representation for robotic task generation. Addressing these challenges, we propose BTGenBot-2, a 1B-parameter open-source small language model that directly converts natural language task descriptions and a list of robot action primitives into executable behavior trees in XML. Unlike prior approaches, BTGenBot-2 enables zero-shot BT generation, error recovery at inference and runtime, while remaining lightweight enough for resource-constrained robots. We further introduce the first standardized benchmark for LLM-based BT generation, covering 52 navigation and manipulation tasks in NVIDIA Isaac Sim. Extensive evaluations demonstrate that BTGenBot-2 consistently outperforms GPT-5, Claude Opus 4.1, and larger open-source models across both functional and non-functional metrics, achieving average success rates of 90.38% in zero-shot and 98.07% in one-shot, while delivering up to 16x faster inference compared to the previous BTGenBot.
>
---
#### [new 052] Path Tracking with Dynamic Control Point Blending for Autonomous Vehicles: An Experimental Study
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于自主车辆路径跟踪任务，解决固定控制点适应性差的问题，通过动态控制点融合和曲率感知控制提升跟踪精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.01892v1](https://arxiv.org/pdf/2602.01892v1)**

> **作者:** Alexandre Lombard; Florent Perronnet; Nicolas Gaud; Abdeljalil Abbas-Turki
>
> **摘要:** This paper presents an experimental study of a path-tracking framework for autonomous vehicles in which the lateral control command is applied to a dynamic control point along the wheelbase. Instead of enforcing a fixed reference at either the front or rear axle, the proposed method continuously interpolates between both, enabling smooth adaptation across driving contexts, including low-speed maneuvers and reverse motion. The lateral steering command is obtained by barycentric blending of two complementary controllers: a front-axle Stanley formulation and a rear-axle curvature-based geometric controller, yielding continuous transitions in steering behavior and improved tracking stability. In addition, we introduce a curvature-aware longitudinal control strategy based on virtual track borders and ray-tracing, which converts upcoming geometric constraints into a virtual obstacle distance and regulates speed accordingly. The complete approach is implemented in a unified control stack and validated in simulation and on a real autonomous vehicle equipped with GPS-RTK, radar, odometry, and IMU. The results in closed-loop tracking and backward maneuvers show improved trajectory accuracy, smoother steering profiles, and increased adaptability compared to fixed control-point baselines.
>
---
#### [new 053] Sem-NaVAE: Semantically-Guided Outdoor Mapless Navigation via Generative Trajectory Priors
- **分类: cs.RO**

- **简介: 该论文提出一种无地图的户外导航方法，结合生成轨迹先验和语义分割，实现实时轨迹选择与导航，解决户外自主导航问题。**

- **链接: [https://arxiv.org/pdf/2602.01429v1](https://arxiv.org/pdf/2602.01429v1)**

> **作者:** Gonzalo Olguin; Javier Ruiz-del-Solar
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** This work presents a mapless global navigation approach for outdoor applications. It combines the exploratory capacity of conditional variational autoencoders (CVAEs) to generate trajectories and the semantic segmentation capabilities of a lightweight visual language model (VLM) to select the trajectory to execute. Open-vocabulary segmentation is used to score and select the generated trajectories based on natural language, and a state-of-the-art local planner executes velocity commands. One of the key features of the proposed approach is its ability to generate a large variability of trajectories and to select them and navigate in real-time. The approach was validated through real-world outdoor navigation experiments, achieving superior performance compared to state-of-the-art methods. A video showing an experimental run of the system can be found in https://www.youtube.com/watch?v=i3R5ey5O2yk.
>
---
#### [new 054] StreamVLA: Breaking the Reason-Act Cycle via Completion-State Gating
- **分类: cs.RO**

- **简介: 该论文提出StreamVLA，解决长周期机器人操作中的高延迟与目标不稳定问题。通过双系统架构和“锁-门”机制，实现高效任务分解与动作生成。**

- **链接: [https://arxiv.org/pdf/2602.01100v1](https://arxiv.org/pdf/2602.01100v1)**

> **作者:** Hang Wu; Tongqing Chen; Jiasen Wang; Xiaotao Li; Lu Fang
>
> **摘要:** Long-horizon robotic manipulation requires bridging the gap between high-level planning (System 2) and low-level control (System 1). Current Vision-Language-Action (VLA) models often entangle these processes, performing redundant multimodal reasoning at every timestep, which leads to high latency and goal instability. To address this, we present StreamVLA, a dual-system architecture that unifies textual task decomposition, visual goal imagination, and continuous action generation within a single parameter-efficient backbone. We introduce a "Lock-and-Gated" mechanism to intelligently modulate computation: only when a sub-task transition is detected, the model triggers slow thinking to generate a textual instruction and imagines the specific visual completion state, rather than generic future frames. Crucially, this completion state serves as a time-invariant goal anchor, making the policy robust to execution speed variations. During steady execution, these high-level intents are locked to condition a Flow Matching action head, allowing the model to bypass expensive autoregressive decoding for 72% of timesteps. This hierarchical abstraction ensures sub-goal focus while significantly reducing inference latency. Extensive evaluations demonstrate that StreamVLA achieves state-of-the-art performance, with a 98.5% success rate on the LIBERO benchmark and robust recovery in real-world interference scenarios, achieving a 48% reduction in latency compared to full-reasoning baselines.
>
---
#### [new 055] LLM-Based Behavior Tree Generation for Construction Machinery
- **分类: cs.RO**

- **简介: 该论文属于自动化任务，旨在解决施工机械协作中行为树手动设计的局限性。通过LLM生成行为树，提升自动化水平。**

- **链接: [https://arxiv.org/pdf/2602.01041v1](https://arxiv.org/pdf/2602.01041v1)**

> **作者:** Akinosuke Tsutsumi; Tomoya Itsuka; Yuichiro Kasahara; Tomoya Kouno; Kota Akinari; Genki Yamauchi; Daisuke Endo; Taro Abe; Takeshi Hashimoto; Keiji Nagatani; Ryo Kurazume
>
> **备注:** 7 pages, 7 figures
>
> **摘要:** Earthwork operations are facing an increasing demand, while workforce aging and skill loss create a pressing need for automation. ROS2-TMS for Construction, a Cyber-Physical System framework designed to coordinate construction machinery, has been proposed for autonomous operation; however, its reliance on manually designed Behavior Trees (BTs) limits scalability, particularly in scenarios involving heterogeneous machine cooperation. Recent advances in large language models (LLMs) offer new opportunities for task planning and BT generation. However, most existing approaches remain confined to simulations or simple manipulators, with relatively few applications demonstrated in real-world contexts, such as complex construction sites involving multiple machines. This paper proposes an LLM-based workflow for BT generation, introducing synchronization flags to enable safe and cooperative operation. The workflow consists of two steps: high-level planning, where the LLM generates synchronization flags, and BT generation using structured templates. Safety is ensured by planning with parameters stored in the system database. The proposed method is validated in simulation and further demonstrated through real-world experiments, highlighting its potential to advance automation in civil engineering.
>
---
#### [new 056] SoMA: A Real-to-Sim Neural Simulator for Robotic Soft-body Manipulation
- **分类: cs.RO; cs.AI; cs.CV; physics.app-ph**

- **简介: 该论文属于机器人软体操作任务，旨在解决真实到仿真模拟中的动态建模问题。提出SoMA，通过神经网络联合环境与机器人动作，实现稳定、可推广的软体操控模拟。**

- **链接: [https://arxiv.org/pdf/2602.02402v1](https://arxiv.org/pdf/2602.02402v1)**

> **作者:** Mu Huang; Hui Wang; Kerui Ren; Linning Xu; Yunsong Zhou; Mulin Yu; Bo Dai; Jiangmiao Pang
>
> **备注:** Project page: https://city-super.github.io/SoMA/
>
> **摘要:** Simulating deformable objects under rich interactions remains a fundamental challenge for real-to-sim robot manipulation, with dynamics jointly driven by environmental effects and robot actions. Existing simulators rely on predefined physics or data-driven dynamics without robot-conditioned control, limiting accuracy, stability, and generalization. This paper presents SoMA, a 3D Gaussian Splat simulator for soft-body manipulation. SoMA couples deformable dynamics, environmental forces, and robot joint actions in a unified latent neural space for end-to-end real-to-sim simulation. Modeling interactions over learned Gaussian splats enables controllable, stable long-horizon manipulation and generalization beyond observed trajectories without predefined physical models. SoMA improves resimulation accuracy and generalization on real-world robot manipulation by 20%, enabling stable simulation of complex tasks such as long-horizon cloth folding.
>
---
#### [new 057] A Systematic Study of Data Modalities and Strategies for Co-training Large Behavior Models for Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文研究机器人操作中通过多模态数据协同训练提升模型泛化能力的问题。通过实验分析不同数据模态对策略性能的影响，旨在构建更通用的机器人策略。**

- **链接: [https://arxiv.org/pdf/2602.01067v1](https://arxiv.org/pdf/2602.01067v1)**

> **作者:** Fanqi Lin; Kushal Arora; Jean Mercat; Haruki Nishimura; Paarth Shah; Chen Xu; Mengchao Zhang; Mark Zolotas; Maya Angeles; Owen Pfannenstiehl; Andrew Beaulieu; Jose Barreiros
>
> **摘要:** Large behavior models have shown strong dexterous manipulation capabilities by extending imitation learning to large-scale training on multi-task robot data, yet their generalization remains limited by the insufficient robot data coverage. To expand this coverage without costly additional data collection, recent work relies on co-training: jointly learning from target robot data and heterogeneous data modalities. However, how different co-training data modalities and strategies affect policy performance remains poorly understood. We present a large-scale empirical study examining five co-training data modalities: standard vision-language data, dense language annotations for robot trajectories, cross-embodiment robot data, human videos, and discrete robot action tokens across single- and multi-phase training strategies. Our study leverages 4,000 hours of robot and human manipulation data and 50M vision-language samples to train vision-language-action policies. We evaluate 89 policies over 58,000 simulation rollouts and 2,835 real-world rollouts. Our results show that co-training with forms of vision-language and cross-embodiment robot data substantially improves generalization to distribution shifts, unseen tasks, and language following, while discrete action token variants yield no significant benefits. Combining effective modalities produces cumulative gains and enables rapid adaptation to unseen long-horizon dexterous tasks via fine-tuning. Training exclusively on robot data degrades the visiolinguistic understanding of the vision-language model backbone, while co-training with effective modalities restores these capabilities. Explicitly conditioning action generation on chain-of-thought traces learned from co-training data does not improve performance in our simulation benchmark. Together, these results provide practical guidance for building scalable generalist robot policies.
>
---
#### [new 058] SPOT: Spatio-Temporal Obstacle-free Trajectory Planning for UAVs in an Unknown Dynamic Environment
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于无人机路径规划任务，解决未知动态环境中实时避障问题。提出一种无需地图的四维时空规划方法，结合视觉感知与轨迹优化，提升避障能力。**

- **链接: [https://arxiv.org/pdf/2602.01189v1](https://arxiv.org/pdf/2602.01189v1)**

> **作者:** Astik Srivastava; Thomas J Chackenkulam. Bitla Bhanu Teja; Antony Thomas; Madhava Krishna
>
> **摘要:** We address the problem of reactive motion planning for quadrotors operating in unknown environments with dynamic obstacles. Our approach leverages a 4-dimensional spatio-temporal planner, integrated with vision-based Safe Flight Corridor (SFC) generation and trajectory optimization. Unlike prior methods that rely on map fusion, our framework is mapless, enabling collision avoidance directly from perception while reducing computational overhead. Dynamic obstacles are detected and tracked using a vision-based object segmentation and tracking pipeline, allowing robust classification of static versus dynamic elements in the scene. To further enhance robustness, we introduce a backup planning module that reactively avoids dynamic obstacles when no direct path to the goal is available, mitigating the risk of collisions during deadlock situations. We validate our method extensively in both simulation and real-world hardware experiments, and benchmark it against state-of-the-art approaches, showing significant advantages for reactive UAV navigation in dynamic, unknown environments.
>
---
#### [new 059] Learning to Accelerate Vision-Language-Action Models through Adaptive Visual Token Caching
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉-语言-动作模型的优化任务，旨在解决计算开销大、效率低的问题。通过引入可学习的动态缓存策略，提升推理速度并提高成功率。**

- **链接: [https://arxiv.org/pdf/2602.00686v1](https://arxiv.org/pdf/2602.00686v1)**

> **作者:** Yujie Wei; Jiahan Fan; Jiyu Guo; Ruichen Zhen; Rui Shao; Xiu Su; Zeke Xie; Shuo Yang
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated remarkable generalization capabilities in robotic manipulation tasks, yet their substantial computational overhead remains a critical obstacle to real-world deployment. Improving inference efficiency is therefore essential for practical robotic applications. Existing acceleration methods often rely on heuristic or static strategies--such as rule-based token caching or pruning--that are decoupled from task objectives and fail to adapt to dynamic scene changes. In this work, we reformulate inference acceleration as a learnable policy optimization problem and propose a novel framework that integrates a dynamic, task-aware decision-making process directly into the VLA model. At its core are two lightweight, cooperative modules: a Cached Token Selector, which determines which tokens should be reused, and a Cache Ratio Predictor, which controls how many tokens to reuse. Training these modules is non-trivial due to their discrete decisions. We address this by adopting a differentiable relaxation that allows gradient-based end-to-end optimization. Extensive experiments on the LIBERO and SIMPLER benchmarks, as well as real-robot evaluations, show that our method achieves a 1.76x wall-clock inference speedup while simultaneously improving the average success rate by 1.9 percentage points (from 75.0% to 76.9%) on LIBERO and by 5.0 percentage points on real-world tasks, significantly outperforming existing baselines. This work highlights the potential of learning task-aware computational allocation policies, paving the way for VLA models that are both powerful and efficient.
>
---
#### [new 060] GSR: Learning Structured Reasoning for Embodied Manipulation
- **分类: cs.RO**

- **简介: 该论文提出GSR方法，解决具身操作中的长周期任务问题，通过显式建模世界状态演变，提升任务推理与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.01693v1](https://arxiv.org/pdf/2602.01693v1)**

> **作者:** Kewei Hu; Michael Zhang; Wei Ying; Tianhao Liu; Guoqiang Hao; Zimeng Li; Wanchan Yu; Jiajian Jing; Fangwen Chen; Hanwen Kang
>
> **摘要:** Despite rapid progress, embodied agents still struggle with long-horizon manipulation that requires maintaining spatial consistency, causal dependencies, and goal constraints. A key limitation of existing approaches is that task reasoning is implicitly embedded in high-dimensional latent representations, making it challenging to separate task structure from perceptual variability. We introduce Grounded Scene-graph Reasoning (GSR), a structured reasoning paradigm that explicitly models world-state evolution as transitions over semantically grounded scene graphs. By reasoning step-wise over object states and spatial relations, rather than directly mapping perception to actions, GSR enables explicit reasoning about action preconditions, consequences, and goal satisfaction in a physically grounded space. To support learning such reasoning, we construct Manip-Cognition-1.6M, a large-scale dataset that jointly supervises world understanding, action planning, and goal interpretation. Extensive evaluations across RLBench, LIBERO, GSR-benchmark, and real-world robotic tasks show that GSR significantly improves zero-shot generalization and long-horizon task completion over prompting-based baselines. These results highlight explicit world-state representations as a key inductive bias for scalable embodied reasoning.
>
---
#### [new 061] Bandwidth-Efficient Multi-Agent Communication through Information Bottleneck and Vector Quantization
- **分类: cs.RO; cs.AI; cs.IT; cs.LG; cs.MA**

- **简介: 该论文属于多智能体强化学习任务，解决通信约束下的协调问题。通过信息瓶颈与向量量化结合，实现高效通信，提升性能并降低带宽消耗。**

- **链接: [https://arxiv.org/pdf/2602.02035v1](https://arxiv.org/pdf/2602.02035v1)**

> **作者:** Ahmad Farooq; Kamran Iqbal
>
> **备注:** Accepted at the 2026 IEEE International Conference on Robotics and Automation (ICRA 2026), Vienna, Austria. 9 pages, 4 figures, 6 tables
>
> **摘要:** Multi-agent reinforcement learning systems deployed in real-world robotics applications face severe communication constraints that significantly impact coordination effectiveness. We present a framework that combines information bottleneck theory with vector quantization to enable selective, bandwidth-efficient communication in multi-agent environments. Our approach learns to compress and discretize communication messages while preserving task-critical information through principled information-theoretic optimization. We introduce a gated communication mechanism that dynamically determines when communication is necessary based on environmental context and agent states. Experimental evaluation on challenging coordination tasks demonstrates that our method achieves 181.8% performance improvement over no-communication baselines while reducing bandwidth usage by 41.4%. Comprehensive Pareto frontier analysis shows dominance across the entire success-bandwidth spectrum with area-under-curve of 0.198 vs 0.142 for next-best methods. Our approach significantly outperforms existing communication strategies and establishes a theoretically grounded framework for deploying multi-agent systems in bandwidth-constrained environments such as robotic swarms, autonomous vehicle fleets, and distributed sensor networks.
>
---
#### [new 062] Safe Stochastic Explorer: Enabling Safe Goal Driven Exploration in Stochastic Environments and Safe Interaction with Unknown Objects
- **分类: cs.RO**

- **简介: 该论文属于机器人安全探索任务，解决在不确定环境中安全导航与交互的问题。提出S.S.Explorer框架，结合高斯过程实现安全驱动的探索。**

- **链接: [https://arxiv.org/pdf/2602.00868v1](https://arxiv.org/pdf/2602.00868v1)**

> **作者:** Nikhil Uday Shinde; Dylan Hirsch; Michael C. Yip; Sylvia Herbert
>
> **摘要:** Autonomous robots operating in unstructured, safety-critical environments, from planetary exploration to warehouses and homes, must learn to safely navigate and interact with their surroundings despite limited prior knowledge. Current methods for safe control, such as Hamilton-Jacobi Reachability and Control Barrier Functions, assume known system dynamics. Meanwhile existing safe exploration techniques often fail to account for the unavoidable stochasticity inherent when operating in unknown real world environments, such as an exploratory rover skidding over an unseen surface or a household robot pushing around unmapped objects in a pantry. To address this critical gap, we propose Safe Stochastic Explorer (S.S.Explorer) a novel framework for safe, goal-driven exploration under stochastic dynamics. Our approach strategically balances safety and information gathering to reduce uncertainty about safety in the unknown environment. We employ Gaussian Processes to learn the unknown safety function online, leveraging their predictive uncertainty to guide information-gathering actions and provide probabilistic bounds on safety violations. We first present our method for discrete state space environments and then introduce a scalable relaxation to effectively extend this approach to continuous state spaces. Finally we demonstrate how this framework can be naturally applied to ensure safe physical interaction with multiple unknown objects. Extensive validation in simulation and demonstrative hardware experiments showcase the efficacy of our method, representing a step forward toward enabling reliable widespread robot autonomy in complex, uncertain environments.
>
---
#### [new 063] Uncertainty-Aware Non-Prehensile Manipulation with Mobile Manipulators under Object-Induced Occlusion
- **分类: cs.RO**

- **简介: 该论文属于移动机械臂的非抓取操作任务，解决物体遮挡传感器视野导致碰撞的问题。提出CURA-PPO框架，通过建模不确定性提升操作安全性与成功率。**

- **链接: [https://arxiv.org/pdf/2602.01731v1](https://arxiv.org/pdf/2602.01731v1)**

> **作者:** Jiwoo Hwang; Taegeun Yang; Jeil Jeong; Minsung Yoon; Sung-Eui Yoon
>
> **备注:** 8 pages, 7 figures, Accepted to ICRA 2026, Webpage: https://jiw0o.github.io/cura-ppo/
>
> **摘要:** Non-prehensile manipulation using onboard sensing presents a fundamental challenge: the manipulated object occludes the sensor's field of view, creating occluded regions that can lead to collisions. We propose CURA-PPO, a reinforcement learning framework that addresses this challenge by explicitly modeling uncertainty under partial observability. By predicting collision possibility as a distribution, we extract both risk and uncertainty to guide the robot's actions. The uncertainty term encourages active perception, enabling simultaneous manipulation and information gathering to resolve occlusions. When combined with confidence maps that capture observation reliability, our approach enables safe navigation despite severe sensor occlusion. Extensive experiments across varying object sizes and obstacle configurations demonstrate that CURA-PPO achieves up to 3X higher success rates than the baselines, with learned behaviors that handle occlusions. Our method provides a practical solution for autonomous manipulation in cluttered environments using only onboard sensing.
>
---
#### [new 064] Tilt-Ropter: A Novel Hybrid Aerial and Terrestrial Vehicle with Tilt Rotors and Passive Wheels
- **分类: cs.RO**

- **简介: 该论文提出Tilt-Ropter，一种新型空地混合机器人，解决多模式高效运动问题。通过倾转旋翼与被动轮结合，实现能量高效的空地转换与轨迹跟踪。**

- **链接: [https://arxiv.org/pdf/2602.01700v1](https://arxiv.org/pdf/2602.01700v1)**

> **作者:** Ruoyu Wang; Xuchen Liu; Zongzhou Wu; Zixuan Guo; Wendi Ding; Ben M. Chen
>
> **备注:** 8 pages, 10 figures
>
> **摘要:** In this work, we present Tilt-Ropter, a novel hybrid aerial-terrestrial vehicle (HATV) that combines tilt rotors with passive wheels to achieve energy-efficient multi-mode locomotion. Unlike existing under-actuated HATVs, the fully actuated design of Tilt-Ropter enables decoupled force and torque control, greatly enhancing its mobility and environmental adaptability. A nonlinear model predictive controller (NMPC) is developed to track reference trajectories and handle contact constraints across locomotion modes, while a dedicated control allocation module exploits actuation redundancy to achieve energy-efficient control of actuators. Additionally, to enhance robustness during ground contact, we introduce an external wrench estimation algorithm that estimates environmental interaction forces and torques in real time. The system is validated through both simulation and real-world experiments, including seamless air-ground transitions and trajectory tracking. Results show low tracking errors in both modes and highlight a 92.8% reduction in power consumption during ground locomotion, demonstrating the system's potential for long-duration missions across large-scale and energy-constrained environments.
>
---
#### [new 065] RAPT: Model-Predictive Out-of-Distribution Detection and Failure Diagnosis for Sim-to-Real Humanoid Robots
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出RAPT，用于人形机器人Sim-to-Real迁移中的异常检测与故障诊断，解决控制策略在分布外状态下的失效问题。**

- **链接: [https://arxiv.org/pdf/2602.01515v1](https://arxiv.org/pdf/2602.01515v1)**

> **作者:** Humphrey Munn; Brendan Tidd; Peter Bohm; Marcus Gallagher; David Howard
>
> **摘要:** Deploying learned control policies on humanoid robots is challenging: policies that appear robust in simulation can execute confidently in out-of-distribution (OOD) states after Sim-to-Real transfer, leading to silent failures that risk hardware damage. Although anomaly detection can mitigate these failures, prior methods are often incompatible with high-rate control, poorly calibrated at the extremely low false-positive rates required for practical deployment, or operate as black boxes that provide a binary stop signal without explaining why the robot drifted from nominal behavior. We present RAPT, a lightweight, self-supervised deployment-time monitor for 50Hz humanoid control. RAPT learns a probabilistic spatio-temporal manifold of nominal execution from simulation and evaluates execution-time predictive deviation as a calibrated, per-dimension signal. This yields (i) reliable online OOD detection under strict false-positive constraints and (ii) a continuous, interpretable measure of Sim-to-Real mismatch that can be tracked over time to quantify how far deployment has drifted from training. Beyond detection, we introduce an automated post-hoc root-cause analysis pipeline that combines gradient-based temporal saliency derived from RAPT's reconstruction objective with LLM-based reasoning conditioned on saliency and joint kinematics to produce semantic failure diagnoses in a zero-shot setting. We evaluate RAPT on a Unitree G1 humanoid across four complex tasks in simulation and on physical hardware. In large-scale simulation, RAPT improves True Positive Rate (TPR) by 37% over the strongest baseline at a fixed episode-level false positive rate of 0.5%. On real-world deployments, RAPT achieves a 12.5% TPR improvement and provides actionable interpretability, reaching 75% root-cause classification accuracy across 16 real-world failures using only proprioceptive data.
>
---
#### [new 066] PRISM: Performer RS-IMLE for Single-pass Multisensory Imitation Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出PRISM，一种单次通过的多模态模仿学习策略，解决机器人实时控制与多传感器融合问题。通过改进IMLE方法，提升任务成功率与控制频率。**

- **链接: [https://arxiv.org/pdf/2602.02396v1](https://arxiv.org/pdf/2602.02396v1)**

> **作者:** Amisha Bhaskar; Pratap Tokekar; Stefano Di Cairano; Alexander Schperberg
>
> **备注:** 10 pages main text and 4 figures, and 11 pages appendix and 10 figures, total 21 pages and 14 figures
>
> **摘要:** Robotic imitation learning typically requires models that capture multimodal action distributions while operating at real-time control rates and accommodating multiple sensing modalities. Although recent generative approaches such as diffusion models, flow matching, and Implicit Maximum Likelihood Estimation (IMLE) have achieved promising results, they often satisfy only a subset of these requirements. To address this, we introduce PRISM, a single-pass policy based on a batch-global rejection-sampling variant of IMLE. PRISM couples a temporal multisensory encoder (integrating RGB, depth, tactile, audio, and proprioception) with a linear-attention generator using a Performer architecture. We demonstrate the efficacy of PRISM on a diverse real-world hardware suite, including loco-manipulation using a Unitree Go2 with a 7-DoF arm D1 and tabletop manipulation with a UR5 manipulator. Across challenging physical tasks such as pre-manipulation parking, high-precision insertion, and multi-object pick-and-place, PRISM outperforms state-of-the-art diffusion policies by 10-25% in success rate while maintaining high-frequency (30-50 Hz) closed-loop control. We further validate our approach on large-scale simulation benchmarks, including CALVIN, MetaWorld, and Robomimic. In CALVIN (10% data split), PRISM improves success rates by approximately 25% over diffusion and approximately 20% over flow matching, while simultaneously reducing trajectory jerk by 20x-50x. These results position PRISM as a fast, accurate, and multisensory imitation policy that retains multimodal action coverage without the latency of iterative sampling.
>
---
#### [new 067] Flow Policy Gradients for Robot Control
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，旨在解决传统策略梯度方法在表达能力上的限制。通过引入流匹配策略梯度，提升政策的表达能力和训练效果。**

- **链接: [https://arxiv.org/pdf/2602.02481v1](https://arxiv.org/pdf/2602.02481v1)**

> **作者:** Brent Yi; Hongsuk Choi; Himanshu Gaurav Singh; Xiaoyu Huang; Takara E. Truong; Carmelo Sferrazza; Yi Ma; Rocky Duan; Pieter Abbeel; Guanya Shi; Karen Liu; Angjoo Kanazawa
>
> **备注:** Project webpage: https://hongsukchoi.github.io/fpo-control
>
> **摘要:** Likelihood-based policy gradient methods are the dominant approach for training robot control policies from rewards. These methods rely on differentiable action likelihoods, which constrain policy outputs to simple distributions like Gaussians. In this work, we show how flow matching policy gradients -- a recent framework that bypasses likelihood computation -- can be made effective for training and fine-tuning more expressive policies in challenging robot control settings. We introduce an improved objective that enables success in legged locomotion, humanoid motion tracking, and manipulation tasks, as well as robust sim-to-real transfer on two humanoid robots. We then present ablations and analysis on training dynamics. Results show how policies can exploit the flow representation for exploration when training from scratch, as well as improved fine-tuning robustness over baselines.
>
---
#### [new 068] A Low-Cost Vision-Based Tactile Gripper with Pretraining Learning for Contact-Rich Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决接触密集环境下的抓取与操作问题。提出低成本视觉触觉夹爪LVTG，结合视觉与触觉反馈，提升抓取稳定性与效率。**

- **链接: [https://arxiv.org/pdf/2602.00514v1](https://arxiv.org/pdf/2602.00514v1)**

> **作者:** Yaohua Liu; Binkai Ou; Zicheng Qiu; Ce Hao; Yemin Wang; Hengjun Zhang
>
> **摘要:** Robotic manipulation in contact-rich environments remains challenging, particularly when relying on conventional tactile sensors that suffer from limited sensing range, reliability, and cost-effectiveness. In this work, we present LVTG, a low-cost visuo-tactile gripper designed for stable, robust, and efficient physical interaction. Unlike existing visuo-tactile sensors, LVTG enables more effective and stable grasping of larger and heavier everyday objects, thanks to its enhanced tactile sensing area and greater opening angle. Its surface skin is made of highly wear-resistant material, significantly improving durability and extending operational lifespan. The integration of vision and tactile feedback allows LVTG to provide rich, high-fidelity sensory data, facilitating reliable perception during complex manipulation tasks. Furthermore, LVTG features a modular design that supports rapid maintenance and replacement. To effectively fuse vision and touch, We adopt a CLIP-inspired contrastive learning objective to align tactile embeddings with their corresponding visual observations, enabling a shared cross-modal representation space for visuo-tactile perception. This alignment improves the performance of an Action Chunking Transformer (ACT) policy in contact-rich manipulation, leading to more efficient data collection and more effective policy learning. Compared to the original ACT method, the proposed LVTG with pretraining achieves significantly higher success rates in manipulation tasks.
>
---
#### [new 069] Synchronized Online Friction Estimation and Adaptive Grasp Control for Robust Gentle Grasp
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决柔性抓取中摩擦系数不确定的问题。通过实时估计摩擦系数并自适应调整抓取力，提升抓取稳定性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.02026v1](https://arxiv.org/pdf/2602.02026v1)**

> **作者:** Zhenwei Niu; Xiaoyi Chen; Jiayu Hu; Zhaoyang Liu; Xiaozu Ju
>
> **摘要:** We introduce a unified framework for gentle robotic grasping that synergistically couples real-time friction estimation with adaptive grasp control. We propose a new particle filter-based method for real-time estimation of the friction coefficient using vision-based tactile sensors. This estimate is seamlessly integrated into a reactive controller that dynamically modulates grasp force to maintain a stable grip. The two processes operate synchronously in a closed-loop: the controller uses the current best estimate to adjust the force, while new tactile feedback from this action continuously refines the estimation. This creates a highly responsive and robust sensorimotor cycle. The reliability and efficiency of the complete framework are validated through extensive robotic experiments.
>
---
#### [new 070] AgenticLab: A Real-World Robot Agent Platform that Can See, Think, and Act
- **分类: cs.RO**

- **简介: 该论文提出AgenticLab平台，解决真实机器人在开放环境中的操作问题。通过构建闭环代理流程，评估视觉语言模型的现实任务表现，揭示传统测试未捕捉的失败模式。**

- **链接: [https://arxiv.org/pdf/2602.01662v1](https://arxiv.org/pdf/2602.01662v1)**

> **作者:** Pengyuan Guo; Zhonghao Mai; Zhengtong Xu; Kaidi Zhang; Heng Zhang; Zichen Miao; Arash Ajoudani; Zachary Kingston; Qiang Qiu; Yu She
>
> **摘要:** Recent advances in large vision-language models (VLMs) have demonstrated generalizable open-vocabulary perception and reasoning, yet their real-robot manipulation capability remains unclear for long-horizon, closed-loop execution in unstructured, in-the-wild environments. Prior VLM-based manipulation pipelines are difficult to compare across different research groups' setups, and many evaluations rely on simulation, privileged state, or specially designed setups. We present AgenticLab, a model-agnostic robot agent platform and benchmark for open-world manipulation. AgenticLab provides a closed-loop agent pipeline for perception, task decomposition, online verification, and replanning. Using AgenticLab, we benchmark state-of-the-art VLM-based agents on real-robot tasks in unstructured environments. Our benchmark reveals several failure modes that offline vision-language tests (e.g., VQA and static image understanding) fail to capture, including breakdowns in multi-step grounding consistency, object grounding under occlusion and scene changes, and insufficient spatial reasoning for reliable manipulation. We will release the full hardware and software stack to support reproducible evaluation and accelerate research on general-purpose robot agents.
>
---
#### [new 071] HumanX: Toward Agile and Generalizable Humanoid Interaction Skills from Human Videos
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出HumanX框架，解决人形机器人交互技能学习问题。通过视频生成通用技能，无需任务特定奖励，提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.02473v1](https://arxiv.org/pdf/2602.02473v1)**

> **作者:** Yinhuai Wang; Qihan Zhao; Yuen Fui Lau; Runyi Yu; Hok Wai Tsui; Qifeng Chen; Jingbo Wang; Jiangmiao Pang; Ping Tan
>
> **摘要:** Enabling humanoid robots to perform agile and adaptive interactive tasks has long been a core challenge in robotics. Current approaches are bottlenecked by either the scarcity of realistic interaction data or the need for meticulous, task-specific reward engineering, which limits their scalability. To narrow this gap, we present HumanX, a full-stack framework that compiles human video into generalizable, real-world interaction skills for humanoids, without task-specific rewards. HumanX integrates two co-designed components: XGen, a data generation pipeline that synthesizes diverse and physically plausible robot interaction data from video while supporting scalable data augmentation; and XMimic, a unified imitation learning framework that learns generalizable interaction skills. Evaluated across five distinct domains--basketball, football, badminton, cargo pickup, and reactive fighting--HumanX successfully acquires 10 different skills and transfers them zero-shot to a physical Unitree G1 humanoid. The learned capabilities include complex maneuvers such as pump-fake turnaround fadeaway jumpshots without any external perception, as well as interactive tasks like sustained human-robot passing sequences over 10 consecutive cycles--learned from a single video demonstration. Our experiments show that HumanX achieves over 8 times higher generalization success than prior methods, demonstrating a scalable and task-agnostic pathway for learning versatile, real-world robot interactive skills.
>
---
#### [new 072] Extending the Law of Intersegmental Coordination: Implications for Powered Prosthetic Controls
- **分类: cs.RO**

- **简介: 该论文属于下肢假肢控制研究，旨在降低截肢者步行能耗。通过分析肢体协调规律，提出新的力矩协调模型，并开发工具箱用于3D数据分析，以优化假肢控制。**

- **链接: [https://arxiv.org/pdf/2602.02181v1](https://arxiv.org/pdf/2602.02181v1)**

> **作者:** Elad Siman Tov; Nili E. Krausz
>
> **备注:** Submitted to 2026 IEEE International Conference on Biomedical Robotics and Biomechatronics (BioRob)
>
> **摘要:** Powered prostheses are capable of providing net positive work to amputees and have advanced in the past two decades. However, reducing amputee metabolic cost of walking remains an open problem. The Law of Intersegmental Coordination (ISC) has been observed across gaits and has been previously implicated in energy expenditure of walking, yet it has rarely been analyzed or applied within the context of lower-limb amputee gait. This law states that the elevation angles of the thigh, shank and foot over the gait cycle are not independent. In this work, we developed a method to analyze intersegmental coordination for lower-limb 3D kinematic data, to simplify ISC analysis. Moreover, inspired by motor control, biomechanics and robotics literature, we used our method to broaden ISC toward a new law of coordination of moments. We find these Elevation Space Moments (ESM), and present results showing a moment-based coordination for able bodied gait. We also analyzed ISC for amputee gait walking with powered and passive prosthesis, and found that while elevation angles remained planar, the ESM showed less coordination. We use ISC as a constraint to predict the shank angles/moments that would compensate for alterations due to a passive foot so as to mimic a healthy thigh angle/moment profile. This may have implications for improving powered prosthetic control. We developed the ISC3d toolbox that is freely available online, which may be used to compute kinematic and kinetic ISC in 3D. This provides a means to further study the role of coordination in gait and may help address fundamental questions of the neural control of human movement.
>
---
#### [new 073] TriphiBot: A Triphibious Robot Combining FOC-based Propulsion with Eccentric Design
- **分类: cs.RO**

- **简介: 该论文属于多域运动机器人研究，旨在解决现有机器人在不同环境中的运动效率和控制问题。提出一种新型三栖机器人，结合FOC推进与偏心设计，实现空中、地面和水中的高效运动与平滑过渡。**

- **链接: [https://arxiv.org/pdf/2602.01385v1](https://arxiv.org/pdf/2602.01385v1)**

> **作者:** Xiangyu Li; Mingwei Lai; Mengke Zhang; Junxiao Lin; Tiancheng Lai; Junping Zhi; Chao Xu; Fei Gao; Yanjun Cao
>
> **摘要:** Triphibious robots capable of multi-domain motion and cross-domain transitions are promising to handle complex tasks across diverse environments. However, existing designs primarily focus on dual-mode platforms, and some designs suffer from high mechanical complexity or low propulsion efficiency, which limits their application. In this paper, we propose a novel triphibious robot capable of aerial, terrestrial, and aquatic motion, by a minimalist design combining a quadcopter structure with two passive wheels, without extra actuators. To address inefficiency of ground-support motion (moving on land/seabed) for quadcopter based designs, we introduce an eccentric Center of Gravity (CoG) design that inherently aligns thrust with motion, enhancing efficiency without specialized mechanical transformation designs. Furthermore, to address the drastic differences in motion control caused by different fluids (air and water), we develop a unified propulsion system based on Field-Oriented Control (FOC). This method resolves torque matching issues and enables precise, rapid bidirectional thrust across different mediums. Grounded in the perspective of living condition and ground support, we analyse the robot's dynamics and propose a Hybrid Nonlinear Model Predictive Control (HNMPC)-PID control system to ensure stable multi-domain motion and seamless transitions. Experimental results validate the robot's multi-domain motion and cross-mode transition capability, along with the efficiency and adaptability of the proposed propulsion system.
>
---
#### [new 074] Vision-only UAV State Estimation for Fast Flights Without External Localization Systems: A2RL Drone Racing Finalist Approach
- **分类: cs.RO**

- **简介: 该论文属于无人机状态估计任务，解决高速飞行中GNSS失效环境下的精准定位问题。通过融合单目视觉与IMU数据，修正VIO漂移，提升状态估计精度。**

- **链接: [https://arxiv.org/pdf/2602.01860v1](https://arxiv.org/pdf/2602.01860v1)**

> **作者:** Filip Novák; Matěj Petrlík; Matej Novosad; Parakh M. Gupta; Robert Pěnička; Martin Saska
>
> **备注:** Visit our webpage for more details: https://mrs.fel.cvut.cz/papers/vision-only-uav-state-estimation
>
> **摘要:** Fast flights with aggressive maneuvers in cluttered GNSS-denied environments require fast, reliable, and accurate UAV state estimation. In this paper, we present an approach for onboard state estimation of a high-speed UAV using a monocular RGB camera and an IMU. Our approach fuses data from Visual-Inertial Odometry (VIO), an onboard landmark-based camera measurement system, and an IMU to produce an accurate state estimate. Using onboard measurement data, we estimate and compensate for VIO drift through a novel mathematical drift model. State-of-the-art approaches often rely on more complex hardware (e.g., stereo cameras or rangefinders) and use uncorrected drifting VIO velocities, orientation, and angular rates, leading to errors during fast maneuvers. In contrast, our method corrects all VIO states (position, orientation, linear and angular velocity), resulting in accurate state estimation even during rapid and dynamic motion. Our approach was thoroughly validated through 1600 simulations and numerous real-world experiments. Furthermore, we applied the proposed method in the A2RL Drone Racing Challenge 2025, where our team advanced to the final four out of 210 teams and earned a medal.
>
---
#### [new 075] Towards Exploratory and Focused Manipulation with Bimanual Active Perception: A New Problem, Benchmark and Strategy
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Exploratory and Focused Manipulation（EFM）问题，解决机器人操作中因视觉遮挡导致的信息缺失。通过建立EFM-10基准和BAP策略，提升操作任务的感知与执行能力。**

- **链接: [https://arxiv.org/pdf/2602.01939v1](https://arxiv.org/pdf/2602.01939v1)**

> **作者:** Yuxin He; Ruihao Zhang; Tianao Shen; Cheng Liu; Qiang Nie
>
> **备注:** ICRA 2026
>
> **摘要:** Recently, active vision has reemerged as an important concept for manipulation, since visual occlusion occurs more frequently when main cameras are mounted on the robot heads. We reflect on the visual occlusion issue and identify its essence as the absence of information useful for task completion. Inspired by this, we come up with the more fundamental problem of Exploratory and Focused Manipulation (EFM). The proposed problem is about actively collecting information to complete challenging manipulation tasks that require exploration or focus. As an initial attempt to address this problem, we establish the EFM-10 benchmark that consists of 4 categories of tasks that align with our definition (10 tasks in total). We further come up with a Bimanual Active Perception (BAP) strategy, which leverages one arm to provide active vision and another arm to provide force sensing while manipulating. Based on this idea, we collect a dataset named BAPData for the tasks in EFM-10. With the dataset, we successfully verify the effectiveness of the BAP strategy in an imitation learning manner. We hope that the EFM-10 benchmark along with the BAP strategy can become a cornerstone that facilitates future research towards this direction. Project website: EFManipulation.github.io.
>
---
#### [new 076] TIC-VLA: A Think-in-Control Vision-Language-Action Model for Robot Navigation in Dynamic Environments
- **分类: cs.RO**

- **简介: 该论文提出TIC-VLA，解决动态环境中机器人导航问题，通过延迟感知框架实现语义推理与实时控制的同步。**

- **链接: [https://arxiv.org/pdf/2602.02459v1](https://arxiv.org/pdf/2602.02459v1)**

> **作者:** Zhiyu Huang; Yun Zhang; Johnson Liu; Rui Song; Chen Tang; Jiaqi Ma
>
> **摘要:** Robots in dynamic, human-centric environments must follow language instructions while maintaining real-time reactive control. Vision-language-action (VLA) models offer a promising framework, but they assume temporally aligned reasoning and control, despite semantic inference being inherently delayed relative to real-time action. We introduce Think-in-Control (TIC)-VLA, a latency-aware framework that explicitly models delayed semantic reasoning during action generation. TIC-VLA defines a delayed semantic-control interface that conditions action generation on delayed vision-language semantic states and explicit latency metadata, in addition to current observations, enabling policies to compensate for asynchronous reasoning. We further propose a latency-consistent training pipeline that injects reasoning inference delays during imitation learning and online reinforcement learning, aligning training with asynchronous deployment. To support realistic evaluation, we present DynaNav, a physics-accurate, photo-realistic simulation suite for language-guided navigation in dynamic environments. Extensive experiments in simulation and on a real robot show that TIC-VLA consistently outperforms prior VLA models while maintaining robust real-time control under multi-second reasoning latency. Project website: https://ucla-mobility.github.io/TIC-VLA/
>
---
#### [new 077] UniDWM: Towards a Unified Driving World Model via Multifaceted Representation Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出UniDWM，属于自动驾驶任务，旨在解决复杂环境下的场景理解与规划问题。通过多模态表征学习构建统一的世界模型，提升轨迹规划与4D重建效果。**

- **链接: [https://arxiv.org/pdf/2602.01536v1](https://arxiv.org/pdf/2602.01536v1)**

> **作者:** Shuai Liu; Siheng Ren; Xiaoyao Zhu; Quanmin Liang; Zefeng Li; Qiang Li; Xin Hu; Kai Huang
>
> **摘要:** Achieving reliable and efficient planning in complex driving environments requires a model that can reason over the scene's geometry, appearance, and dynamics. We present UniDWM, a unified driving world model that advances autonomous driving through multifaceted representation learning. UniDWM constructs a structure- and dynamic-aware latent world representation that serves as a physically grounded state space, enabling consistent reasoning across perception, prediction, and planning. Specifically, a joint reconstruction pathway learns to recover the scene's structure, including geometry and visual texture, while a collaborative generation framework leverages a conditional diffusion transformer to forecast future world evolution within the latent space. Furthermore, we show that our UniDWM can be deemed as a variation of VAE, which provides theoretical guidance for the multifaceted representation learning. Extensive experiments demonstrate the effectiveness of UniDWM in trajectory planning, 4D reconstruction and generation, highlighting the potential of multifaceted world representations as a foundation for unified driving intelligence. The code will be publicly available at https://github.com/Say2L/UniDWM.
>
---
#### [new 078] SkySim: A ROS2-based Simulation Environment for Natural Language Control of Drone Swarms using Large Language Models
- **分类: cs.RO; cs.HC**

- **简介: 该论文提出SkySim，用于无人机群的自然语言控制，解决安全与灵活性问题。结合LLM与ROS2，实现高效避障与动态环境适应。**

- **链接: [https://arxiv.org/pdf/2602.01226v1](https://arxiv.org/pdf/2602.01226v1)**

> **作者:** Aditya Shibu; Marah Saleh; Mohamed Al-Musleh; Nidhal Abdulaziz
>
> **摘要:** Unmanned Aerial Vehicle (UAV) swarms offer versatile applications in logistics, agriculture, and surveillance, yet controlling them requires expert knowledge for safety and feasibility. Traditional static methods limit adaptability, while Large Language Models (LLMs) enable natural language control but generate unsafe trajectories due to lacking physical grounding. This paper introduces SkySim, a ROS2-based simulation framework in Gazebo that decouples LLM high-level planning from low-level safety enforcement. Using Gemini 3.5 Pro, SkySim translates user commands (e.g., "Form a circle") into spatial waypoints, informed by real-time drone states. An Artificial Potential Field (APF) safety filter applies minimal adjustments for collision avoidance, kinematic limits, and geo-fencing, ensuring feasible execution at 20 Hz. Experiments with swarms of 3, 10, and 30 Crazyflie drones validate spatial reasoning accuracy (100% across tested geometric primitives), real-time collision prevention, and scalability. SkySim empowers non-experts to iteratively refine behaviors, bridging AI cognition with robotic safety for dynamic environments. Future work targets hardware integration.
>
---
#### [new 079] Meanshift Shape Formation Control Using Discrete Mass Distribution
- **分类: cs.RO**

- **简介: 该论文属于群体机器人控制任务，旨在解决复杂形状生成与群体规模变化适应问题。提出一种基于离散质量分布的去中心化控制策略，实现高效形状形成与动态适应。**

- **链接: [https://arxiv.org/pdf/2602.00980v1](https://arxiv.org/pdf/2602.00980v1)**

> **作者:** Yichen Cai; Yuan Gao; Pengpeng Li; Wei Wang; Guibin Sun; Jinhu Lü
>
> **摘要:** The density-distribution method has recently become a promising paradigm owing to its adaptability to variations in swarm size. However, existing studies face practical challenges in achieving complex shape representation and decentralized implementation. This motivates us to develop a fully decentralized, distribution-based control strategy with the dual capability of forming complex shapes and adapting to swarm-size variations. Specifically, we first propose a discrete mass-distribution function defined over a set of sample points to model swarm formation. In contrast to the continuous density-distribution method, our model eliminates the requirement for defining continuous density functions-a task that is difficult for complex shapes. Second, we design a decentralized meanshift control law to coordinate the swarm's global distribution to fit the sample-point distribution by feeding back mass estimates. The mass estimates for all sample points are achieved by the robots in a decentralized manner via the designed mass estimator. It is shown that the mass estimates of the sample points can asymptotically converge to the true global values. To validate the proposed strategy, we conduct comprehensive simulations and real-world experiments to evaluate the efficiency of complex shape formation and adaptability to swarm-size variations.
>
---
#### [new 080] KAN We Flow? Advancing Robotic Manipulation with 3D Flow Matching via KAN & RWKV
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在解决传统方法计算效率低的问题。通过引入KAN和RWKV构建轻量模型，提升3D操作性能。**

- **链接: [https://arxiv.org/pdf/2602.01115v1](https://arxiv.org/pdf/2602.01115v1)**

> **作者:** Zhihao Chen; Yiyuan Ge; Ziyang Wang
>
> **备注:** Accepted By ICRA2026
>
> **摘要:** Diffusion-based visuomotor policies excel at modeling action distributions but are inference-inefficient, since recursively denoising from noise to policy requires many steps and heavy UNet backbones, which hinders deployment on resource-constrained robots. Flow matching alleviates the sampling burden by learning a one-step vector field, yet prior implementations still inherit large UNet-style architectures. In this work, we present KAN-We-Flow, a flow-matching policy that draws on recent advances in Receptance Weighted Key Value (RWKV) and Kolmogorov-Arnold Networks (KAN) from vision to build a lightweight and highly expressive backbone for 3D manipulation. Concretely, we introduce an RWKV-KAN block: an RWKV first performs efficient time/channel mixing to propagate task context, and a subsequent GroupKAN layer applies learnable spline-based, groupwise functional mappings to perform feature-wise nonlinear calibration of the action mapping on RWKV outputs. Moreover, we introduce an Action Consistency Regularization (ACR), a lightweight auxiliary loss that enforces alignment between predicted action trajectories and expert demonstrations via Euler extrapolation, providing additional supervision to stabilize training and improve policy precision. Without resorting to large UNets, our design reduces parameters by 86.8\%, maintains fast runtime, and achieves state-of-the-art success rates on Adroit, Meta-World, and DexArt benchmarks. Our project page can be viewed in \href{https://zhihaochen-2003.github.io/KAN-We-Flow.github.io/}{\textcolor{red}{link}}
>
---
#### [new 081] Minimal Footprint Grasping Inspired by Ants
- **分类: cs.RO**

- **简介: 该论文属于机械抓取任务，旨在解决复杂环境中物体抓取问题。通过模仿蚂蚁前足结构，设计了一种低成本、高鲁棒性的夹爪，有效提升抓取性能。**

- **链接: [https://arxiv.org/pdf/2602.00935v1](https://arxiv.org/pdf/2602.00935v1)**

> **作者:** Mohamed Sorour; Barbara Webb
>
> **摘要:** Ants are highly capable of grasping objects in clutter, and we have recently observed that this involves substantial use of their forelegs. The forelegs, more specifically the tarsi, have high friction microstructures (setal pads), are covered in hairs, and have a flexible under-actuated tip. Here we abstract these features to test their functional advantages for a novel low-cost gripper design, suitable for bin-picking applications. In our implementation, the gripper legs are long and slim, with high friction gripping pads, low friction hairs and single-segment tarsus-like structure to mimic the insect's setal pads, hairs, and the tarsi's interactive compliance. Experimental evaluation shows this design is highly robust for grasping a wide variety of individual consumer objects, with all grasp attempts successful. In addition, we demonstrate this design is effective for picking single objects from dense clutter, a task at which ants also show high competence. The work advances grasping technology and shed new light on the mechanical importance of hairy structures and tarsal flexibility in insects.
>
---
#### [new 082] Factored Reasoning with Inner Speech and Persistent Memory for Evidence-Grounded Human-Robot Interaction
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机交互任务，旨在解决机器人在对话中保持上下文、处理不明确请求及基于证据回应的问题。工作包括提出JANUS架构，分解行为模块并实现可验证的决策流程。**

- **链接: [https://arxiv.org/pdf/2602.00675v1](https://arxiv.org/pdf/2602.00675v1)**

> **作者:** Valerio Belcamino; Mariya Kilina; Alessandro Carfì; Valeria Seidita; Fulvio Mastrogiovanni; Antonio Chella
>
> **摘要:** Dialogue-based human-robot interaction requires robot cognitive assistants to maintain persistent user context, recover from underspecified requests, and ground responses in external evidence, while keeping intermediate decisions verifiable. In this paper we introduce JANUS, a cognitive architecture for assistive robots that models interaction as a partially observable Markov decision process and realizes control as a factored controller with typed interfaces. To this aim, Janus (i) decomposes the overall behavior into specialized modules, related to scope detection, intent recognition, memory, inner speech, query generation, and outer speech, and (ii) exposes explicit policies for information sufficiency, execution readiness, and tool grounding. A dedicated memory agent maintains a bounded recent-history buffer, a compact core memory, and an archival store with semantic retrieval, coupled through controlled consolidation and revision policies. Models inspired by the notion of inner speech in cognitive theories provide a control-oriented internal textual flow that validates parameter completeness and triggers clarification before grounding, while a faithfulness constraint ties robot-to-human claims to an evidence bundle combining working context and retrieved tool outputs. We evaluate JANUS through module-level unit tests in a dietary assistance domain grounded on a knowledge graph, reporting high agreement with curated references and practical latency profiles. These results support factored reasoning as a promising path to scalable, auditable, and evidence-grounded robot assistance over extended interaction horizons.
>
---
#### [new 083] SyNeT: Synthetic Negatives for Traversability Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于自主机器人导航任务，旨在解决缺乏负样本导致的可行驶性估计不准确问题。通过构造合成负样本，提升模型识别非可行驶区域的能力。**

- **链接: [https://arxiv.org/pdf/2602.00814v1](https://arxiv.org/pdf/2602.00814v1)**

> **作者:** Bomena Kim; Hojun Lee; Younsoo Park; Yaoyu Hu; Sebastian Scherer; Inwook Shim
>
> **摘要:** Reliable traversability estimation is crucial for autonomous robots to navigate complex outdoor environments safely. Existing self-supervised learning frameworks primarily rely on positive and unlabeled data; however, the lack of explicit negative data remains a critical limitation, hindering the model's ability to accurately identify diverse non-traversable regions. To address this issue, we introduce a method to explicitly construct synthetic negatives, representing plausible but non-traversable, and integrate them into vision-based traversability learning. Our approach is formulated as a training strategy that can be seamlessly integrated into both Positive-Unlabeled (PU) and Positive-Negative (PN) frameworks without modifying inference architectures. Complementing standard pixel-wise metrics, we introduce an object-centric FPR evaluation approach that analyzes predictions in regions where synthetic negatives are inserted. This evaluation provides an indirect measure of the model's ability to consistently identify non-traversable regions without additional manual labeling. Extensive experiments on both public and self-collected datasets demonstrate that our approach significantly enhances robustness and generalization across diverse environments. The source code and demonstration videos are publicly available at the project page: https://anonymous-synet.github.io/SyNet.github.io/
>
---
#### [new 084] HERMES: A Holistic End-to-End Risk-Aware Multimodal Embodied System with Vision-Language Models for Long-Tail Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶任务，旨在解决长尾场景下的安全驾驶问题。提出HERMES系统，融合多模态信息与视觉-语言模型，提升轨迹规划的准确性和风险感知能力。**

- **链接: [https://arxiv.org/pdf/2602.00993v1](https://arxiv.org/pdf/2602.00993v1)**

> **作者:** Weizhe Tang; Junwei You; Jiaxi Liu; Zhaoyi Wang; Rui Gan; Zilin Huang; Feng Wei; Bin Ran
>
> **摘要:** End-to-end autonomous driving models increasingly benefit from large vision--language models for semantic understanding, yet ensuring safe and accurate operation under long-tail conditions remains challenging. These challenges are particularly prominent in long-tail mixed-traffic scenarios, where autonomous vehicles must interact with heterogeneous road users, including human-driven vehicles and vulnerable road users, under complex and uncertain conditions. This paper proposes HERMES, a holistic risk-aware end-to-end multimodal driving framework designed to inject explicit long-tail risk cues into trajectory planning. HERMES employs a foundation-model-assisted annotation pipeline to produce structured Long-Tail Scene Context and Long-Tail Planning Context, capturing hazard-centric cues together with maneuver intent and safety preference, and uses these signals to guide end-to-end planning. HERMES further introduces a Tri-Modal Driving Module that fuses multi-view perception, historical motion cues, and semantic guidance, ensuring risk-aware accurate trajectory planning under long-tail scenarios. Experiments on the real-world long-tail dataset demonstrate that HERMES consistently outperforms representative end-to-end and VLM-driven baselines under long-tail mixed-traffic scenarios. Ablation studies verify the complementary contributions of key components.
>
---
#### [new 085] Geometry-Aware Sampling-Based Motion Planning on Riemannian Manifolds
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，解决非欧几里得配置空间中的路径规划问题。提出一种基于黎曼流形的采样规划框架，提升路径优化效果。**

- **链接: [https://arxiv.org/pdf/2602.00992v1](https://arxiv.org/pdf/2602.00992v1)**

> **作者:** Phone Thiha Kyaw; Jonathan Kelly
>
> **备注:** Submitted to WAFR 2026 (17th World Symposium on the Algorithmic Foundations of Robotics (WAFR))
>
> **摘要:** In many robot motion planning problems, task objectives and physical constraints induce non-Euclidean geometry on the configuration space, yet many planners operate using Euclidean distances that ignore this structure. We address the problem of planning collision-free motions that minimize length under configuration-dependent Riemannian metrics, corresponding to geodesics on the configuration manifold. Conventional numerical methods for computing such paths do not scale well to high-dimensional systems, while sampling-based planners trade scalability for geometric fidelity. To bridge this gap, we propose a sampling-based motion planning framework that operates directly on Riemannian manifolds. We introduce a computationally efficient midpoint-based approximation of the Riemannian geodesic distance and prove that it matches the true Riemannian distance with third-order accuracy. Building on this approximation, we design a local planner that traces the manifold using first-order retractions guided by Riemannian natural gradients. Experiments on a two-link planar arm and a 7-DoF Franka manipulator under a kinetic-energy metric, as well as on rigid-body planning in $\mathrm{SE}(2)$ with non-holonomic motion constraints, demonstrate that our approach consistently produces lower-cost trajectories than Euclidean-based planners and classical numerical geodesic-solver baselines.
>
---
#### [new 086] Relationship-Aware Hierarchical 3D Scene Graph for Task Reasoning
- **分类: cs.RO**

- **简介: 该论文提出一种关系感知的层次化3D场景图，用于任务推理。旨在解决3D环境结构化表示与关系推理问题，结合视觉语言模型和大语言模型提升智能体环境理解与任务处理能力。**

- **链接: [https://arxiv.org/pdf/2602.02456v1](https://arxiv.org/pdf/2602.02456v1)**

> **作者:** Albert Gassol Puigjaner; Angelos Zacharia; Kostas Alexis
>
> **备注:** ICRA 2026, 8 pages
>
> **摘要:** Representing and understanding 3D environments in a structured manner is crucial for autonomous agents to navigate and reason about their surroundings. While traditional Simultaneous Localization and Mapping (SLAM) methods generate metric reconstructions and can be extended to metric-semantic mapping, they lack a higher level of abstraction and relational reasoning. To address this gap, 3D scene graphs have emerged as a powerful representation for capturing hierarchical structures and object relationships. In this work, we propose an enhanced hierarchical 3D scene graph that integrates open-vocabulary features across multiple abstraction levels and supports object-relational reasoning. Our approach leverages a Vision Language Model (VLM) to infer semantic relationships. Notably, we introduce a task reasoning module that combines Large Language Models (LLM) and a VLM to interpret the scene graph's semantic and relational information, enabling agents to reason about tasks and interact with their environment more intelligently. We validate our method by deploying it on a quadruped robot in multiple environments and tasks, highlighting its ability to reason about them.
>
---
#### [new 087] Real-Time Loop Closure Detection in Visual SLAM via NetVLAD and Faiss
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉SLAM中的回环检测任务，旨在解决传统方法在外观变化下的性能下降问题。通过评估NetVLAD并结合Faiss加速，提升实时性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.01673v1](https://arxiv.org/pdf/2602.01673v1)**

> **作者:** Enguang Fan
>
> **摘要:** Loop closure detection (LCD) is a core component of simultaneous localization and mapping (SLAM): it identifies revisited places and enables pose-graph constraints that correct accumulated drift. Classic bag-of-words approaches such as DBoW are efficient but often degrade under appearance change and perceptual aliasing. In parallel, deep learning-based visual place recognition (VPR) descriptors (e.g., NetVLAD and Transformer-based models) offer stronger robustness, but their computational cost is often viewed as a barrier to real-time SLAM. In this paper, we empirically evaluate NetVLAD as an LCD module and compare it against DBoW on the KITTI dataset. We introduce a Fine-Grained Top-K precision-recall curve that better reflects LCD settings where a query may have zero or multiple valid matches. With Faiss-accelerated nearestneighbor search, NetVLAD achieves real-time query speed while improving accuracy and robustness over DBoW, making it a practical drop-in alternative for LCD in SLAM.
>
---
#### [new 088] LangMap: A Hierarchical Benchmark for Open-Vocabulary Goal Navigation
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出LangMap，一个用于开放词汇目标导航的层次化基准任务，解决AI理解语言指令进行导航的问题。工作包括构建大规模标注数据集，涵盖不同语义层级的导航任务。**

- **链接: [https://arxiv.org/pdf/2602.02220v1](https://arxiv.org/pdf/2602.02220v1)**

> **作者:** Bo Miao; Weijia Liu; Jun Luo; Lachlan Shinnick; Jian Liu; Thomas Hamilton-Smith; Yuhe Yang; Zijie Wu; Vanja Videnovic; Feras Dayoub; Anton van den Hengel
>
> **摘要:** The relationships between objects and language are fundamental to meaningful communication between humans and AI, and to practically useful embodied intelligence. We introduce HieraNav, a multi-granularity, open-vocabulary goal navigation task where agents interpret natural language instructions to reach targets at four semantic levels: scene, room, region, and instance. To this end, we present Language as a Map (LangMap), a large-scale benchmark built on real-world 3D indoor scans with comprehensive human-verified annotations and tasks spanning these levels. LangMap provides region labels, discriminative region descriptions, discriminative instance descriptions covering 414 object categories, and over 18K navigation tasks. Each target features both concise and detailed descriptions, enabling evaluation across different instruction styles. LangMap achieves superior annotation quality, outperforming GOAT-Bench by 23.8% in discriminative accuracy using four times fewer words. Comprehensive evaluations of zero-shot and supervised models on LangMap reveal that richer context and memory improve success, while long-tailed, small, context-dependent, and distant goals, as well as multi-goal completion, remain challenging. HieraNav and LangMap establish a rigorous testbed for advancing language-driven embodied navigation. Project: https://bo-miao.github.io/LangMap
>
---
#### [new 089] Improving Robustness of Vision-Language-Action Models by Restoring Corrupted Visual Inputs
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型的鲁棒性提升任务，解决视觉干扰导致的性能下降问题。提出CRT模型，通过修复受损图像恢复模型性能。**

- **链接: [https://arxiv.org/pdf/2602.01158v1](https://arxiv.org/pdf/2602.01158v1)**

> **作者:** Daniel Yezid Guarnizo Orjuela; Leonardo Scappatura; Veronica Di Gennaro; Riccardo Andrea Izzo; Gianluca Bardaro; Matteo Matteucci
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a dominant paradigm for generalist robotic manipulation, unifying perception and control within a single end-to-end architecture. However, despite their success in controlled environments, reliable real-world deployment is severely hindered by their fragility to visual disturbances. While existing literature extensively addresses physical occlusions caused by scene geometry, a critical mode remains largely unexplored: image corruptions. These sensor-level artifacts, ranging from electronic noise and dead pixels to lens contaminants, directly compromise the integrity of the visual signal prior to interpretation. In this work, we quantify this vulnerability, demonstrating that state-of-the-art VLAs such as $π_{0.5}$ and SmolVLA, suffer catastrophic performance degradation, dropping from 90\% success rates to as low as 2\%, under common signal artifacts. To mitigate this, we introduce the Corruption Restoration Transformer (CRT), a plug-and-play and model-agnostic vision transformer designed to immunize VLA models against sensor disturbances. Leveraging an adversarial training objective, CRT restores clean observations from corrupted inputs without requiring computationally expensive fine-tuning of the underlying model. Extensive experiments across the LIBERO and Meta-World benchmarks demonstrate that CRT effectively recovers lost performance, enabling VLAs to maintain near-baseline success rates, even under severe visual corruption.
>
---
#### [new 090] Real-Time 2D LiDAR Object Detection Using Three-Frame RGB Scan Encoding
- **分类: eess.SP; cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于2D LiDAR目标检测任务，旨在实现嵌入式室内机器人实时、隐私友好的物体检测。通过三帧扫描编码为RGB图像，提升检测精度与速度。**

- **链接: [https://arxiv.org/pdf/2602.02167v1](https://arxiv.org/pdf/2602.02167v1)**

> **作者:** Soheil Behnam Roudsari; Alexandre S. Brandão; Felipe N. Martins
>
> **备注:** 6 pages, 6 figures, submitted to IEEE SAS 2026
>
> **摘要:** Indoor service robots need perception that is robust, more privacy-friendly than RGB video, and feasible on embedded hardware. We present a camera-free 2D LiDAR object detection pipeline that encodes short-term temporal context by stacking three consecutive scans as RGB channels, yielding a compact YOLOv8n input without occupancy-grid construction while preserving angular structure and motion cues. Evaluated in Webots across 160 randomized indoor scenarios with strict scenario-level holdout, the method achieves 98.4% mAP@0.5 (0.778 mAP@0.5:0.95) with 94.9% precision and 94.7% recall on four object classes. On a Raspberry Pi 5, it runs in real time with a mean post-warm-up end-to-end latency of 47.8ms per frame, including scan encoding and postprocessing. Relative to a closely related occupancy-grid LiDAR-YOLO pipeline reported on the same platform, the proposed representation is associated with substantially lower reported end-to-end latency. Although results are simulation-based, they suggest that lightweight temporal encoding can enable accurate and real-time LiDAR-only detection for embedded indoor robotics without capturing RGB appearance.
>
---
#### [new 091] PovNet+: A Deep Learning Architecture for Socially Assistive Robots to Learn and Assist with Multiple Activities of Daily Living
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于多活动识别任务，解决机器人无法同时感知和协助多种日常活动的问题。提出POVNet+架构，通过多模态学习区分已知、未知及异常活动，提升助人交互能力。**

- **链接: [https://arxiv.org/pdf/2602.00131v1](https://arxiv.org/pdf/2602.00131v1)**

> **作者:** Fraser Robinson; Souren Pashangpour; Matthew Lisondra; Goldie Nejat
>
> **备注:** Submitted to Advanced Robotics (Taylor & Francis)
>
> **摘要:** A significant barrier to the long-term deployment of autonomous socially assistive robots is their inability to both perceive and assist with multiple activities of daily living (ADLs). In this paper, we present the first multimodal deep learning architecture, POVNet+, for multi-activity recognition for socially assistive robots to proactively initiate assistive behaviors. Our novel architecture introduces the use of both ADL and motion embedding spaces to uniquely distinguish between a known ADL being performed, a new unseen ADL, or a known ADL being performed atypically in order to assist people in real scenarios. Furthermore, we apply a novel user state estimation method to the motion embedding space to recognize new ADLs while monitoring user performance. This ADL perception information is used to proactively initiate robot assistive interactions. Comparison experiments with state-of-the-art human activity recognition methods show our POVNet+ method has higher ADL classification accuracy. Human-robot interaction experiments in a cluttered living environment with multiple users and the socially assistive robot Leia using POVNet+ demonstrate the ability of our multi-modal ADL architecture in successfully identifying different seen and unseen ADLs, and ADLs being performed atypically, while initiating appropriate assistive human-robot interactions.
>
---
#### [new 092] DDP-WM: Disentangled Dynamics Prediction for Efficient World Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DDP-WM，解决世界模型计算效率低的问题。通过解耦动态预测，提升实时性能，适用于导航与操作任务。**

- **链接: [https://arxiv.org/pdf/2602.01780v1](https://arxiv.org/pdf/2602.01780v1)**

> **作者:** Shicheng Yin; Kaixuan Yin; Weixing Chen; Yang Liu; Guanbin Li; Liang Lin
>
> **备注:** Codes will be available at https://github.com/HCPLabSYSU/DDP-WM
>
> **摘要:** World models are essential for autonomous robotic planning. However, the substantial computational overhead of existing dense Transformerbased models significantly hinders real-time deployment. To address this efficiency-performance bottleneck, we introduce DDP-WM, a novel world model centered on the principle of Disentangled Dynamics Prediction (DDP). We hypothesize that latent state evolution in observed scenes is heterogeneous and can be decomposed into sparse primary dynamics driven by physical interactions and secondary context-driven background updates. DDP-WM realizes this decomposition through an architecture that integrates efficient historical processing with dynamic localization to isolate primary dynamics. By employing a crossattention mechanism for background updates, the framework optimizes resource allocation and provides a smooth optimization landscape for planners. Extensive experiments demonstrate that DDP-WM achieves significant efficiency and performance across diverse tasks, including navigation, precise tabletop manipulation, and complex deformable or multi-body interactions. Specifically, on the challenging Push-T task, DDP-WM achieves an approximately 9 times inference speedup and improves the MPC success rate from 90% to98% compared to state-of-the-art dense models. The results establish a promising path for developing efficient, high-fidelity world models. Codes will be available at https://github.com/HCPLabSYSU/DDP-WM.
>
---
#### [new 093] VVLoc: Prior-free 3-DoF Vehicle Visual Localization
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于车辆视觉定位任务，解决传统方法依赖先验信息、无法量化置信度的问题。提出VVLoc，通过单神经网络实现拓扑与度量定位，提升定位精度与实用性。**

- **链接: [https://arxiv.org/pdf/2602.00810v1](https://arxiv.org/pdf/2602.00810v1)**

> **作者:** Ze Huang; Zhongyang Xiao; Mingliang Song; Longan Yang; Hongyuan Yuan; Li Sun
>
> **摘要:** Localization is a critical technology in autonomous driving, encompassing both topological localization, which identifies the most similar map keyframe to the current observation, and metric localization, which provides precise spatial coordinates. Conventional methods typically address these tasks independently, rely on single-camera setups, and often require additional 3D semantic or pose priors, while lacking mechanisms to quantify the confidence of localization results, making them less feasible for real industrial applications. In this paper, we propose VVLoc, a unified pipeline that employs a single neural network to concurrently achieve topological and metric vehicle localization using multi-camera system. VVLoc first evaluates the geo-proximity between visual observations, then estimates their relative metric poses using a matching strategy, while also providing a confidence measure. Additionally, the training process for VVLoc is highly efficient, requiring only pairs of visual data and corresponding ground-truth poses, eliminating the need for complex supplementary data. We evaluate VVLoc not only on the publicly available datasets, but also on a more challenging self-collected dataset, demonstrating its ability to deliver state-of-the-art localization accuracy across a wide range of localization tasks.
>
---
#### [new 094] Stealthy Coverage Control for Human-enabled Real-Time 3D Reconstruction
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于3D重建任务，解决如何高效采集图像以重构复杂场景的问题。通过半自主系统，结合人类判断与无人机自动采样，提升重建质量。**

- **链接: [https://arxiv.org/pdf/2602.00466v1](https://arxiv.org/pdf/2602.00466v1)**

> **作者:** Reiji Terunuma; Yuta Nakamura; Takuma Abe; Takeshi Hatanaka
>
> **备注:** This work has been submitted to the 23rd IFAC World Congress for possible publication
>
> **摘要:** In this paper, we propose a novel semi-autonomous image sampling strategy, called stealthy coverage control, for human-enabled 3D structure reconstruction. The present mission involves a fundamental problem: while the number of images required to accurately reconstruct a 3D model depends on the structural complexity of the target scene to be reconstructed, it is not realistic to assume prior knowledge of the spatially non-uniform structural complexity. We approach this issue by leveraging human flexible reasoning and situational recognition capabilities. Specifically, we design a semi-autonomous system that leaves identification of regions that need more images and navigation of the drones to such regions to a human operator. To this end, we first present a way to reflect the human intention in autonomous coverage control. Subsequently, in order to avoid operational conflicts between manual control and autonomous coverage control, we develop the stealthy coverage control that decouples the drone motion for efficient image sampling from navigation by the human. Simulation studies on a Unity/ROS2-based simulator demonstrate that the present semi-autonomous system outperforms the one without human interventions in the sense of the reconstructed model quality.
>
---
#### [new 095] Efficient UAV trajectory prediction: A multi-modal deep diffusion framework
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文属于无人机轨迹预测任务，旨在提升低空经济中非法无人机的轨迹预测精度。通过融合LiDAR与雷达数据，提出多模态深度融合框架，显著提高预测效果。**

- **链接: [https://arxiv.org/pdf/2602.00107v1](https://arxiv.org/pdf/2602.00107v1)**

> **作者:** Yuan Gao; Xinyu Guo; Wenjing Xie; Zifan Wang; Hongwen Yu; Gongyang Li; Shugong Xu
>
> **备注:** in Chinese language
>
> **摘要:** To meet the requirements for managing unauthorized UAVs in the low-altitude economy, a multi-modal UAV trajectory prediction method based on the fusion of LiDAR and millimeter-wave radar information is proposed. A deep fusion network for multi-modal UAV trajectory prediction, termed the Multi-Modal Deep Fusion Framework, is designed. The overall architecture consists of two modality-specific feature extraction networks and a bidirectional cross-attention fusion module, aiming to fully exploit the complementary information of LiDAR and radar point clouds in spatial geometric structure and dynamic reflection characteristics. In the feature extraction stage, the model employs independent but structurally identical feature encoders for LiDAR and radar. After feature extraction, the model enters the Bidirectional Cross-Attention Mechanism stage to achieve information complementarity and semantic alignment between the two modalities. To verify the effectiveness of the proposed model, the MMAUD dataset used in the CVPR 2024 UG2+ UAV Tracking and Pose-Estimation Challenge is adopted as the training and testing dataset. Experimental results show that the proposed multi-modal fusion model significantly improves trajectory prediction accuracy, achieving a 40% improvement compared to the baseline model. In addition, ablation experiments are conducted to demonstrate the effectiveness of different loss functions and post-processing strategies in improving model performance. The proposed model can effectively utilize multi-modal data and provides an efficient solution for unauthorized UAV trajectory prediction in the low-altitude economy.
>
---
#### [new 096] Before Autonomy Takes Control: Software Testing in Robotics
- **分类: cs.SE; cs.RO**

- **简介: 该论文属于软件测试任务，旨在解决机器人系统测试难题。通过分析247篇文献，探讨机器人测试现状与挑战，提出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2602.02293v1](https://arxiv.org/pdf/2602.02293v1)**

> **作者:** Nils Chur; Thiago Santos de Moura; Argentina Ortega; Sven Peldszus; Thorsten Berger; Nico Hochgeschwender; Yannic Noller
>
> **摘要:** Robotic systems are complex and safety-critical software systems. As such, they need to be tested thoroughly. Unfortunately, robot software is intrinsically hard to test compared to traditional software, mainly since the software needs to closely interact with hardware, account for uncertainty in its operational environment, handle disturbances, and act highly autonomously. However, given the large space in which robots operate, anticipating possible failures when designing tests is challenging. This paper presents a mapping study by considering robotics testing papers and relating them to the software testing theory. We consider 247 robotics testing papers and map them to software testing, discussing the state-of-the-art software testing in robotics with an illustrated example, and discuss current challenges. Forming the basis to introduce both the robotics and software engineering communities to software testing challenges. Finally, we identify open questions and lessons learned.
>
---
#### [new 097] Dual Quaternion SE(3) Synchronization with Recovery Guarantees
- **分类: math.OC; cs.CV; cs.RO; eess.SP**

- **简介: 该论文研究SE(3)同步问题，旨在从噪声相对变换中恢复绝对位姿。通过双四元数表示，提出一种两阶段算法，提升准确性和效率。**

- **链接: [https://arxiv.org/pdf/2602.00324v1](https://arxiv.org/pdf/2602.00324v1)**

> **作者:** Jianing Zhao; Linglingzhi Zhu; Anthony Man-Cho So
>
> **摘要:** Synchronization over the special Euclidean group SE(3) aims to recover absolute poses from noisy pairwise relative transformations and is a core primitive in robotics and 3D vision. Standard approaches often require multi-step heuristic procedures to recover valid poses, which are difficult to analyze and typically lack theoretical guarantees. This paper adopts a dual quaternion representation and formulates SE(3) synchronization directly over the unit dual quaternion. A two-stage algorithm is developed: A spectral initializer computed via the power method on a Hermitian dual quaternion measurement matrix, followed by a dual quaternion generalized power method (DQGPM) that enforces feasibility through per-iteration projection. The estimation error bounds are established for spectral estimators, and DQGPM is shown to admit a finite-iteration error bound and achieves linear error contraction up to an explicit noise-dependent threshold. Experiments on synthetic benchmarks and real-world multi-scan point-set registration demonstrate that the proposed pipeline improves both accuracy and efficiency over representative matrix-based methods.
>
---
#### [new 098] Navigating Simply, Aligning Deeply: Winning Solutions for Mouse vs. AI 2025
- **分类: cs.CV; cs.AI; cs.NE; cs.RO**

- **简介: 该论文属于视觉鲁棒性与神经对齐任务，旨在提升AI在视觉寻食中的表现。通过简化架构和深度网络设计，分别在两个赛道取得最佳成绩。**

- **链接: [https://arxiv.org/pdf/2602.00982v1](https://arxiv.org/pdf/2602.00982v1)**

> **作者:** Phu-Hoa Pham; Chi-Nguyen Tran; Dao Sy Duy Minh; Nguyen Lam Phu Quy; Huynh Trung Kiet
>
> **备注:** 15 pages, 8 tables. Technical Report for winning solutions (Track 1 & Track 2) at the NeurIPS 2025 Mouse vs. AI Challenge
>
> **摘要:** Visual robustness and neural alignment remain critical challenges in developing artificial agents that can match biological vision systems. We present the winning approaches from Team HCMUS_TheFangs for both tracks of the NeurIPS 2025 Mouse vs. AI: Robust Visual Foraging Competition. For Track 1 (Visual Robustness), we demonstrate that architectural simplicity combined with targeted components yields superior generalization, achieving 95.4% final score with a lightweight two-layer CNN enhanced by Gated Linear Units and observation normalization. For Track 2 (Neural Alignment), we develop a deep ResNet-like architecture with 16 convolutional layers and GLU-based gating that achieves top-1 neural prediction performance with 17.8 million parameters. Our systematic analysis of ten model checkpoints trained between 60K to 1.14M steps reveals that training duration exhibits a non-monotonic relationship with performance, with optimal results achieved around 200K steps. Through comprehensive ablation studies and failure case analysis, we provide insights into why simpler architectures excel at visual robustness while deeper models with increased capacity achieve better neural alignment. Our results challenge conventional assumptions about model complexity in visuomotor learning and offer practical guidance for developing robust, biologically-inspired visual agents.
>
---
#### [new 099] OASIS-DC: Generalizable Depth Completion via Output-level Alignment of Sparse-Integrated Monocular Pseudo Depth
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于深度补全任务，解决单目深度估计缺乏度量尺度的问题。通过结合稀疏测量与基础模型，生成伪度量深度先验，并设计网络进行优化，提升少量标注数据下的精度。**

- **链接: [https://arxiv.org/pdf/2602.01268v1](https://arxiv.org/pdf/2602.01268v1)**

> **作者:** Jaehyeon Cho; Jhonghyun An
>
> **备注:** Accepted to ICRA 2026
>
> **摘要:** Recent monocular foundation models excel at zero-shot depth estimation, yet their outputs are inherently relative rather than metric, limiting direct use in robotics and autonomous driving. We leverage the fact that relative depth preserves global layout and boundaries: by calibrating it with sparse range measurements, we transform it into a pseudo metric depth prior. Building on this prior, we design a refinement network that follows the prior where reliable and deviates where necessary, enabling accurate metric predictions from very few labeled samples. The resulting system is particularly effective when curated validation data are unavailable, sustaining stable scale and sharp edges across few-shot regimes. These findings suggest that coupling foundation priors with sparse anchors is a practical route to robust, deployment-ready depth completion under real-world label scarcity.
>
---
#### [new 100] DISK: Dynamic Inference SKipping for World Models
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出DISK，用于自动回归世界模型的动态推理跳过方法，解决视频与轨迹预测中的效率问题，通过双分支控制器提升推理速度并保持预测质量。**

- **链接: [https://arxiv.org/pdf/2602.00440v1](https://arxiv.org/pdf/2602.00440v1)**

> **作者:** Anugunj Naman; Gaibo Zhang; Ayushman Singh; Yaguang Zhang
>
> **摘要:** We present DISK, a training-free adaptive inference method for autoregressive world models. DISK coordinates two coupled diffusion transformers for video and ego-trajectory via dual-branch controllers with cross-modal skip decisions, preserving motion-appearance consistency without retraining. We extend higher-order latent-difference skip testing to the autoregressive chain-of-forward regime and propagate controller statistics through rollout loops for long-horizon stability. When integrated into closed-loop driving rollouts on 1500 NuPlan and NuScenes samples using an NVIDIA L40S GPU, DISK achieves 2x speedup on trajectory diffusion and 1.6x speedup on video diffusion while maintaining L2 planning error, visual quality (FID/FVD), and NAVSIM PDMS scores, demonstrating practical long-horizon video-and-trajectory prediction at substantially reduced cost.
>
---
#### [new 101] AdaptNC: Adaptive Nonconformity Scores for Uncertainty-Aware Autonomous Systems in Dynamic Environments
- **分类: cs.LG; cs.RO; eess.SY**

- **简介: 该论文属于不确定性量化任务，旨在解决动态环境中自主系统预测可靠性问题。提出AdaptNC框架，联合调整非合规模型和阈值，提升预测效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2602.01629v1](https://arxiv.org/pdf/2602.01629v1)**

> **作者:** Renukanandan Tumu; Aditya Singh; Rahul Mangharam
>
> **摘要:** Rigorous uncertainty quantification is essential for the safe deployment of autonomous systems in unconstrained environments. Conformal Prediction (CP) provides a distribution-free framework for this task, yet its standard formulations rely on exchangeability assumptions that are violated by the distribution shifts inherent in real-world robotics. Existing online CP methods maintain target coverage by adaptively scaling the conformal threshold, but typically employ a static nonconformity score function. We show that this fixed geometry leads to highly conservative, volume-inefficient prediction regions when environments undergo structural shifts. To address this, we propose \textbf{AdaptNC}, a framework for the joint online adaptation of both the nonconformity score parameters and the conformal threshold. AdaptNC leverages an adaptive reweighting scheme to optimize score functions, and introduces a replay buffer mechanism to mitigate the coverage instability that occurs during score transitions. We evaluate AdaptNC on diverse robotic benchmarks involving multi-agent policy changes, environmental changes and sensor degradation. Our results demonstrate that AdaptNC significantly reduces prediction region volume compared to state-of-the-art threshold-only baselines while maintaining target coverage levels.
>
---
#### [new 102] Efficiently Solving Mixed-Hierarchy Games with Quasi-Policy Approximations
- **分类: cs.GT; cs.RO**

- **简介: 该论文研究多机器人协同中的混合层次博弈问题，旨在解决同时存在纳什和斯塔克尔伯格决策结构的协调难题。通过引入准策略近似和牛顿法，实现高效求解。**

- **链接: [https://arxiv.org/pdf/2602.01568v1](https://arxiv.org/pdf/2602.01568v1)**

> **作者:** Hamzah Khan; Dong Ho Lee; Jingqi Li; Tianyu Qiu; Christian Ellis; Jesse Milzman; Wesley Suttle; David Fridovich-Keil
>
> **摘要:** Multi-robot coordination often exhibits hierarchical structure, with some robots' decisions depending on the planned behaviors of others. While game theory provides a principled framework for such interactions, existing solvers struggle to handle mixed information structures that combine simultaneous (Nash) and hierarchical (Stackelberg) decision-making. We study N-robot forest-structured mixed-hierarchy games, in which each robot acts as a Stackelberg leader over its subtree while robots in different branches interact via Nash equilibria. We derive the Karush-Kuhn-Tucker (KKT) first-order optimality conditions for this class of games and show that they involve increasingly high-order derivatives of robots' best-response policies as the hierarchy depth grows, rendering a direct solution intractable. To overcome this challenge, we introduce a quasi-policy approximation that removes higher-order policy derivatives and develop an inexact Newton method for efficiently solving the resulting approximated KKT systems. We prove local exponential convergence of the proposed algorithm for games with non-quadratic objectives and nonlinear constraints. The approach is implemented in a highly optimized Julia library (MixedHierarchyGames.jl) and evaluated in simulated experiments, demonstrating real-time convergence for complex mixed-hierarchy information structures.
>
---
#### [new 103] SDCM: Simulated Densifying and Compensatory Modeling Fusion for Radar-Vision 3-D Object Detection in Internet of Vehicles
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文针对车联网中雷达-视觉3D目标检测任务，解决点云稀疏和视觉信息退化问题，提出SDCM框架，包含模拟增密、补偿映射和交互融合模块。**

- **链接: [https://arxiv.org/pdf/2602.00149v1](https://arxiv.org/pdf/2602.00149v1)**

> **作者:** Shucong Li; Xiaoluo Zhou; Yuqian He; Zhenyu Liu
>
> **摘要:** 3-D object detection based on 4-D radar-vision is an important part in Internet of Vehicles (IoV). However, there are two challenges which need to be faced. First, the 4-D radar point clouds are sparse, leading to poor 3-D representation. Second, vision datas exhibit representation degradation under low-light, long distance detection and dense occlusion scenes, which provides unreliable texture information during fusion stage. To address these issues, a framework named SDCM is proposed, which contains Simulated Densifying and Compensatory Modeling Fusion for radar-vision 3-D object detection in IoV. Firstly, considering point generation based on Gaussian simulation of key points obtained from 3-D Kernel Density Estimation (3-D KDE), and outline generation based on curvature simulation, Simulated Densifying (SimDen) module is designed to generate dense radar point clouds. Secondly, considering that radar data could provide more real time information than vision data, due to the all-weather property of 4-D radar. Radar Compensatory Mapping (RCM) module is designed to reduce the affects of vision datas' representation degradation. Thirdly, considering that feature tensor difference values contain the effective information of every modality, which could be extracted and modeled for heterogeneity reduction and modalities interaction, Mamba Modeling Interactive Fusion (MMIF) module is designed for reducing heterogeneous and achieving interactive Fusion. Experiment results on the VoD, TJ4DRadSet and Astyx HiRes 2019 dataset show that SDCM achieves best performance with lower parameter quantity and faster inference speed. Our code will be available.
>
---
#### [new 104] Motion Planning with Metric Temporal Logic Using Reachability Analysis and Hybrid Zonotopes
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于自主车辆运动规划任务，旨在解决满足时间约束的控制优化问题。通过可达性分析与混合zonotope表示，高效编码MTL规范并生成运动计划。**

- **链接: [https://arxiv.org/pdf/2602.00325v1](https://arxiv.org/pdf/2602.00325v1)**

> **作者:** Andrew F. Thompson; Joshua A. Robbins; Jonah J. Glunt; Sean B. Brennan; Herschel C. Pangborn
>
> **摘要:** Metric temporal logic (MTL) provides a formal framework for defining time-dependent mission requirements on autonomous vehicles. However, optimizing control decisions subject to these constraints is often computationally expensive. This article presents a method that uses reachability analysis to implicitly express the set of states satisfying an MTL specification and then optimizes to find a motion plan. The hybrid zonotope set representation is used to efficiently and conveniently encode MTL specifications into reachable sets. A numerical benchmark highlights the proposed method's computational advantages as compared to existing methods in the literature. Further numerical examples and an experimental application demonstrate the ability to address time-varying environments, region-dependent disturbances, and multi-agent coordination.
>
---
#### [new 105] PolicyFlow: Policy Optimization with Continuous Normalizing Flow in Reinforcement Learning
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出PolicyFlow，解决PPO在使用连续归一化流（CNF）策略时的计算难题，通过近似重要性比率和引入布朗正则化，提升策略多样性与性能。**

- **链接: [https://arxiv.org/pdf/2602.01156v1](https://arxiv.org/pdf/2602.01156v1)**

> **作者:** Shunpeng Yang; Ben Liu; Hua Chen
>
> **备注:** Submitted to ICLR 2026
>
> **摘要:** Among on-policy reinforcement learning algorithms, Proximal Policy Optimization (PPO) demonstrates is widely favored for its simplicity, numerical stability, and strong empirical performance. Standard PPO relies on surrogate objectives defined via importance ratios, which require evaluating policy likelihood that is typically straightforward when the policy is modeled as a Gaussian distribution. However, extending PPO to more expressive, high-capacity policy models such as continuous normalizing flows (CNFs), also known as flow-matching models, is challenging because likelihood evaluation along the full flow trajectory is computationally expensive and often numerically unstable. To resolve this issue, we propose PolicyFlow, a novel on-policy CNF-based reinforcement learning algorithm that integrates expressive CNF policies with PPO-style objectives without requiring likelihood evaluation along the full flow path. PolicyFlow approximates importance ratios using velocity field variations along a simple interpolation path, reducing computational overhead without compromising training stability. To further prevent mode collapse and further encourage diverse behaviors, we propose the Brownian Regularizer, an implicit policy entropy regularizer inspired by Brownian motion, which is conceptually elegant and computationally lightweight. Experiments on diverse tasks across various environments including MultiGoal, PointMaze, IsaacLab and MuJoCo Playground show that PolicyFlow achieves competitive or superior performance compared to PPO using Gaussian policies and flow-based baselines including FPO and DPPO. Notably, results on MultiGoal highlight PolicyFlow's ability to capture richer multimodal action distributions.
>
---
#### [new 106] From Perception to Action: Spatial AI Agents and World Models
- **分类: cs.LG; cs.AI; cs.CV; cs.MA; cs.RO**

- **简介: 该论文属于空间智能与自主系统领域，旨在解决物理世界中代理的感知与行动问题。通过构建统一框架，整合代理能力、空间任务和尺度，提出关键技术和未来研究方向。**

- **链接: [https://arxiv.org/pdf/2602.01644v1](https://arxiv.org/pdf/2602.01644v1)**

> **作者:** Gloria Felicia; Nolan Bryant; Handi Putra; Ayaan Gazali; Eliel Lobo; Esteban Rojas
>
> **备注:** 61 pages, 742 citations, 1 figure, 3 tables. Survey paper on spatial AI agents, embodied AI, graph neural networks, and world models
>
> **摘要:** While large language models have become the prevailing approach for agentic reasoning and planning, their success in symbolic domains does not readily translate to the physical world. Spatial intelligence, the ability to perceive 3D structure, reason about object relationships, and act under physical constraints, is an orthogonal capability that proves important for embodied agents. Existing surveys address either agentic architectures or spatial domains in isolation. None provide a unified framework connecting these complementary capabilities. This paper bridges that gap. Through a thorough review of over 2,000 papers, citing 742 works from top-tier venues, we introduce a unified three-axis taxonomy connecting agentic capabilities with spatial tasks across scales. Crucially, we distinguish spatial grounding (metric understanding of geometry and physics) from symbolic grounding (associating images with text), arguing that perception alone does not confer agency. Our analysis reveals three key findings mapped to these axes: (1) hierarchical memory systems (Capability axis) are important for long-horizon spatial tasks. (2) GNN-LLM integration (Task axis) is a promising approach for structured spatial reasoning. (3) World models (Scale axis) are essential for safe deployment across micro-to-macro spatial scales. We conclude by identifying six grand challenges and outlining directions for future research, including the need for unified evaluation frameworks to standardize cross-domain assessment. This taxonomy provides a foundation for unifying fragmented research efforts and enabling the next generation of spatially-aware autonomous systems in robotics, autonomous vehicles, and geospatial intelligence.
>
---
#### [new 107] RVDebloater: Mode-based Adaptive Firmware Debloating for Robotic Vehicles
- **分类: cs.CR; cs.RO; cs.SE**

- **简介: 该论文属于嵌入式系统安全任务，旨在解决嵌入式设备固件攻击面过大问题。通过分析模式化设备的运行需求，自动去除冗余代码，实现动态固件精简。**

- **链接: [https://arxiv.org/pdf/2602.00270v1](https://arxiv.org/pdf/2602.00270v1)**

> **作者:** Mohsen Salehi; Karthik Pattabiraman
>
> **摘要:** As the number of embedded devices grows and their functional requirements increase, embedded firmware is becoming increasingly larger, thereby expanding its attack surface. Despite the increase in firmware size, many embedded devices, such as robotic vehicles (RVs), operate in distinct modes, each requiring only a small subset of the firmware code at runtime. We refer to such devices as mode-based embedded devices. Debloating is an approach to reduce attack surfaces by removing or restricting unneeded code, but existing techniques suffer from significant limitations, such as coarse granularity and irreversible code removal, limiting their applicability. To address these limitations, we propose RVDebloater, a novel adaptive debloating technique for mode-based embedded devices that automatically identifies unneeded firmware code for each mode using either static or dynamic analysis, and dynamically debloats the firmware for each mode at the function level at runtime. RVDebloater introduces a new software-based enforcement approach that supports diverse mode-based embedded devices. We implemented RVDebloater using the LLVM compiler and evaluated its efficiency and effectiveness on six different RVs, including both simulated and real ones, with different real-world missions. We find that device requirements change throughout its lifetime for each mode, and that many critical firmware functions can be restricted in other modes, with an average of 85% of functions not being required. The results showed that none of the missions failed after debloating with RVDebloater, indicating that it neither incurred false positives nor false negatives. Further, RVDebloater prunes the firmware call graph by an average of 45% across different firmware. Finally, RVDebloater incurred an average performance overhead of 3.9% and memory overhead of 4% (approximately 0.25 MB) on real RVs.
>
---
#### [new 108] LatentTrack: Sequential Weight Generation via Latent Filtering
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **简介: 该论文提出LatentTrack，用于在线概率预测任务，解决非平稳动态下的不确定性建模问题。通过潜在空间过滤和轻量超网络生成参数，实现高效在线适应。**

- **链接: [https://arxiv.org/pdf/2602.00458v1](https://arxiv.org/pdf/2602.00458v1)**

> **作者:** Omer Haq
>
> **摘要:** We introduce LatentTrack (LT), a sequential neural architecture for online probabilistic prediction under nonstationary dynamics. LT performs causal Bayesian filtering in a low-dimensional latent space and uses a lightweight hypernetwork to generate predictive model parameters at each time step, enabling constant-time online adaptation without per-step gradient updates. At each time step, a learned latent model predicts the next latent distribution, which is updated via amortized inference using new observations, yielding a predict--generate--update filtering framework in function space. The formulation supports both structured (Markovian) and unstructured latent dynamics within a unified objective, while Monte Carlo inference over latent trajectories produces calibrated predictive mixtures with fixed per-step cost. Evaluated on long-horizon online regression using the Jena Climate benchmark, LT consistently achieves lower negative log-likelihood and mean squared error than stateful sequential and static uncertainty-aware baselines, with competitive calibration, demonstrating that latent-conditioned function evolution is an effective alternative to traditional latent-state modeling under distribution shift.
>
---
#### [new 109] Parallel Stochastic Gradient-Based Planning for World Models
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习中的规划任务，解决视觉输入下长期控制问题。提出GRASP方法，通过随机优化和梯度松弛实现高效规划。**

- **链接: [https://arxiv.org/pdf/2602.00475v1](https://arxiv.org/pdf/2602.00475v1)**

> **作者:** Michael Psenka; Michael Rabbat; Aditi Krishnapriyan; Yann LeCun; Amir Bar
>
> **备注:** 23 pages, 7 figures
>
> **摘要:** World models simulate environment dynamics from raw sensory inputs like video. However, using them for planning can be challenging due to the vast and unstructured search space. We propose a robust and highly parallelizable planner that leverages the differentiability of the learned world model for efficient optimization, solving long-horizon control tasks from visual input. Our method treats states as optimization variables ("virtual states") with soft dynamics constraints, enabling parallel computation and easier optimization. To facilitate exploration and avoid local optima, we introduce stochasticity into the states. To mitigate sensitive gradients through high-dimensional vision-based world models, we modify the gradient structure to descend towards valid plans while only requiring action-input gradients. Our planner, which we call GRASP (Gradient RelAxed Stochastic Planner), can be viewed as a stochastic version of a non-condensed or collocation-based optimal controller. We provide theoretical justification and experiments on video-based world models, where our resulting planner outperforms existing planning algorithms like the cross-entropy method (CEM) and vanilla gradient-based optimization (GD) on long-horizon experiments, both in success rate and time to convergence.
>
---
#### [new 110] Any3D-VLA: Enhancing VLA Robustness via Diverse Point Clouds
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言-动作（VLA）任务，旨在提升模型在复杂场景中的空间理解。针对3D数据稀缺和领域差异问题，提出Any3D-VLA，融合多源点云与2D图像，增强模型鲁棒性。**

- **链接: [https://arxiv.org/pdf/2602.00807v1](https://arxiv.org/pdf/2602.00807v1)**

> **作者:** Xianzhe Fan; Shengliang Deng; Xiaoyang Wu; Yuxiang Lu; Zhuoling Li; Mi Yan; Yujia Zhang; Zhizheng Zhang; He Wang; Hengshuang Zhao
>
> **摘要:** Existing Vision-Language-Action (VLA) models typically take 2D images as visual input, which limits their spatial understanding in complex scenes. How can we incorporate 3D information to enhance VLA capabilities? We conduct a pilot study across different observation spaces and visual representations. The results show that explicitly lifting visual input into point clouds yields representations that better complement their corresponding 2D representations. To address the challenges of (1) scarce 3D data and (2) the domain gap induced by cross-environment differences and depth-scale biases, we propose Any3D-VLA. It unifies the simulator, sensor, and model-estimated point clouds within a training pipeline, constructs diverse inputs, and learns domain-agnostic 3D representations that are fused with the corresponding 2D representations. Simulation and real-world experiments demonstrate Any3D-VLA's advantages in improving performance and mitigating the domain gap. Our project homepage is available at https://xianzhefan.github.io/Any3D-VLA.github.io.
>
---
## 更新

#### [replaced 001] Unified Task and Motion Planning using Object-centric Abstractions of Motion Constraints
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于任务与运动规划（TAMP）领域，旨在解决抽象描述与物理约束不匹配的问题。通过对象中心的运动约束抽象，将任务与运动规划统一为单一启发式搜索，提高规划效率。**

- **链接: [https://arxiv.org/pdf/2312.17605v2](https://arxiv.org/pdf/2312.17605v2)**

> **作者:** Alejandro Agostini; Justus Piater
>
> **备注:** This version corrects notation in Eqs. (7) and (8) and includes minor changes
>
> **摘要:** In task and motion planning (TAMP), the ambiguity and underdetermination of abstract descriptions used by task planning methods make it difficult to characterize physical constraints needed to successfully execute a task. The usual approach is to overlook such constraints at task planning level and to implement expensive sub-symbolic geometric reasoning techniques that perform multiple calls on unfeasible actions, plan corrections, and re-planning until a feasible solution is found. We propose an alternative TAMP approach that unifies task and motion planning into a single heuristic search. Our approach is based on an object-centric abstraction of motion constraints that permits leveraging the computational efficiency of off-the-shelf AI heuristic search to yield physically feasible plans. These plans can be directly transformed into object and motion parameters for task execution without the need of intensive sub-symbolic geometric reasoning.
>
---
#### [replaced 002] Line-Search Filter Differential Dynamic Programming for Optimal Control with Nonlinear Equality Constraints
- **分类: math.OC; cs.RO; eess.SY**

- **简介: 该论文提出FilterDDP算法，用于解决带有非线性等式约束的离散时间最优控制问题。通过步骤过滤器和线搜索处理约束，提升数值稳定性与收敛性。**

- **链接: [https://arxiv.org/pdf/2504.08278v5](https://arxiv.org/pdf/2504.08278v5)**

> **作者:** Ming Xu; Stephen Gould; Iman Shames
>
> **备注:** Accepted for publication in the IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** We present FilterDDP, a differential dynamic programming algorithm for solving discrete-time, optimal control problems (OCPs) with nonlinear equality constraints. Unlike prior methods based on merit functions or the augmented Lagrangian class of algorithms, FilterDDP uses a step filter in conjunction with a line search to handle equality constraints. We identify two important design choices for the step filter criteria which lead to robust numerical performance: 1) we use the Lagrangian instead of the cost in the step acceptance criterion and, 2) in the backward pass, we perturb the value function Hessian. Both choices are rigorously justified, for 2) in particular by a formal proof of local quadratic convergence. In addition to providing a primal-dual interior point extension for handling OCPs with both equality and inequality constraints, we validate FilterDDP on three contact implicit trajectory optimisation problems which arise in robotics.
>
---
#### [replaced 003] MineInsight: A Multi-sensor Dataset for Humanitarian Demining Robotics in Off-Road Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出MineInsight数据集，用于解决人道主义排雷中机器人检测地雷的算法验证问题。整合多传感器数据，提升检测准确性与环境适应性。**

- **链接: [https://arxiv.org/pdf/2506.04842v2](https://arxiv.org/pdf/2506.04842v2)**

> **作者:** Mario Malizia; Charles Hamesse; Ken Hasselmann; Geert De Cubber; Nikolaos Tsiogkas; Eric Demeester; Rob Haelterman
>
> **摘要:** The use of robotics in humanitarian demining increasingly involves computer vision techniques to improve landmine detection capabilities. However, in the absence of diverse and realistic datasets, the reliable validation of algorithms remains a challenge for the research community. In this paper, we introduce MineInsight, a publicly available multi-sensor, multi-spectral dataset designed for off-road landmine detection. The dataset features 35 different targets (15 landmines and 20 commonly found objects) distributed along three distinct tracks, providing a diverse and realistic testing environment. MineInsight is, to the best of our knowledge, the first dataset to integrate dual-view sensor scans from both an Unmanned Ground Vehicle and its robotic arm, offering multiple viewpoints to mitigate occlusions and improve spatial awareness. It features two LiDARs, as well as images captured at diverse spectral ranges, including visible (RGB, monochrome), visible short-wave infrared (VIS-SWIR), and long-wave infrared (LWIR). Additionally, the dataset provides bounding boxes generated by an automated pipeline and refined with human supervision. We recorded approximately one hour of data in both daylight and nighttime conditions, resulting in around 38,000 RGB frames, 53,000 VIS-SWIR frames, and 108,000 LWIR frames. MineInsight serves as a benchmark for developing and evaluating landmine detection algorithms. Our dataset is available at https://github.com/mariomlz99/MineInsight.
>
---
#### [replaced 004] A Gait Driven Reinforcement Learning Framework for Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人步态控制任务，旨在提升人形机器人行走效率。通过设计新型步态规划器和奖励函数，实现快速学习与稳定步态。**

- **链接: [https://arxiv.org/pdf/2506.08416v2](https://arxiv.org/pdf/2506.08416v2)**

> **作者:** Bolin Li; Yuzhi Jiang; Linwei Sun; Xuecong Huang; Lijun Zhu; Han Ding
>
> **摘要:** This paper presents a real-time gait driven training framework for humanoid robots. First, we introduce a novel gait planner that incorporates dynamics to design the desired joint trajectory. In the gait design process, the 3D robot model is decoupled into two 2D models, which are then approximated as hybrid inverted pendulums (H-LIP) for trajectory planning. The gait planner operates in parallel in real time within the robot's learning environment. Second, based on this gait planner, we design three effective reward functions within a reinforcement learning framework, forming a reward composition to achieve periodic bipedal gait. This reward composition reduces the robot's learning time and enhances locomotion performance. Finally, a gait design example, along with simulation and experimental comparisons, is presented to demonstrate the effectiveness of the proposed method.
>
---
#### [replaced 005] An Extended Generalized Prandtl-Ishlinskii Hysteresis Model for I2RIS Robot
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决柔性手术机器人中的多阶段迟滞问题。通过提出扩展的普朗特-伊什林斯基模型，提高了迟滞建模的精度。**

- **链接: [https://arxiv.org/pdf/2504.12114v2](https://arxiv.org/pdf/2504.12114v2)**

> **作者:** Yiyao Yue; Mojtaba Esfandiari; Pengyuan Du; Peter Gehlbach; Makoto Jinno; Adnan Munawar; Peter Kazanzides; Iulian Iordachita
>
> **备注:** Published in IFAC-PapersOnLine, Vol. 59, No. 30, pp. 323-328, 2025. 5th Conference on Modeling, Estimation and Control (MECC 2025)
>
> **摘要:** Retinal surgery requires extreme precision due to constrained anatomical spaces in the human retina. To assist surgeons achieve this level of accuracy, the Improved Integrated Robotic Intraocular Snake (I2RIS) with dexterous capability has been developed. However, such flexible tendon-driven robots often suffer from hysteresis problems, which significantly challenges precise control and positioning. In particular, we observed multi-stage hysteresis phenomena in the small-scale I2RIS. In this paper, we propose an Extended Generalized Prandtl-Ishlinskii (EGPI) model to increase the fitting accuracy of the hysteresis. The model incorporates a novel switching mechanism that enables it to describe multi-stage hysteresis in the regions of monotonic input. Experimental validation on I2RIS data demonstrates that the EGPI model outperforms the conventional Generalized Prandtl-Ishlinskii (GPI) model in terms of RMSE, NRMSE, and MAE across multiple motor input directions. The EGPI model in our study highlights the potential in modeling multi-stage hysteresis in minimally invasive flexible robots.
>
---
#### [replaced 006] CADGrasp: Learning Contact and Collision Aware General Dexterous Grasping in Cluttered Scenes
- **分类: cs.RO**

- **简介: 该论文属于机械臂抓取任务，旨在解决杂乱场景中灵活抓取的碰撞与接触问题。提出CADGrasp算法，通过两阶段优化实现稳定抓取。**

- **链接: [https://arxiv.org/pdf/2601.15039v2](https://arxiv.org/pdf/2601.15039v2)**

> **作者:** Jiyao Zhang; Zhiyuan Ma; Tianhao Wu; Zeyuan Chen; Hao Dong
>
> **摘要:** Dexterous grasping in cluttered environments presents substantial challenges due to the high degrees of freedom of dexterous hands, occlusion, and potential collisions arising from diverse object geometries and complex layouts. To address these challenges, we propose CADGrasp, a two-stage algorithm for general dexterous grasping using single-view point cloud inputs. In the first stage, we predict sparse IBS, a scene-decoupled, contact- and collision-aware representation, as the optimization target. Sparse IBS compactly encodes the geometric and contact relationships between the dexterous hand and the scene, enabling stable and collision-free dexterous grasp pose optimization. To enhance the prediction of this high-dimensional representation, we introduce an occupancy-diffusion model with voxel-level conditional guidance and force closure score filtering. In the second stage, we develop several energy functions and ranking strategies for optimization based on sparse IBS to generate high-quality dexterous grasp poses. Extensive experiments in both simulated and real-world settings validate the effectiveness of our approach, demonstrating its capability to mitigate collisions while maintaining a high grasp success rate across diverse objects and complex scenes.
>
---
#### [replaced 007] ARCAS: An Augmented Reality Collision Avoidance System with SLAM-Based Tracking for Enhancing VRU Safety
- **分类: eess.SY; cs.AR; cs.CV; cs.ET; cs.RO; eess.IV**

- **简介: 该论文提出ARCAS系统，用于增强VRU安全。属于交通安全任务，解决VRU碰撞风险问题，通过AR技术提供实时预警。**

- **链接: [https://arxiv.org/pdf/2512.05299v2](https://arxiv.org/pdf/2512.05299v2)**

> **作者:** Ahmad Yehia; Jiseop Byeon; Tianyi Wang; Huihai Wang; Yiming Xu; Junfeng Jiao; Christian Claudel
>
> **备注:** 8 pages, 3 figures, 1 table, accepted for IEEE Intelligent Vehicles (IV) Symposium 2026
>
> **摘要:** Vulnerable road users (VRUs) face high collision risks in mixed traffic, yet most existing safety systems prioritize driver or vehicle assistance over direct VRU support. This paper presents ARCAS, a real-time augmented reality (AR) collision avoidance system that provides personalized spatial alerts to VRUs via wearable AR headsets. By fusing roadside 360° 3D LiDAR with SLAM-based headset tracking and an automatic 3D calibration procedure, ARCAS accurately overlays world-locked 3D bounding boxes and directional arrows onto approaching hazards in the user's passthrough view. The system also enables multi-headset coordination through shared world anchoring. Evaluated in real-world pedestrian interactions with e-scooters and vehicles (180 trials), ARCAS nearly doubles pedestrians' time to collision and increases counterparts' reaction margins by up to 4x compared to unaided eye conditions. Results validate the feasibility and effectiveness of LiDAR-driven AR guidance and highlight the potential of wearable AR as a promising next generation safety tool for urban mobility.
>
---
#### [replaced 008] Transferring Kinesthetic Demonstrations across Diverse Objects for Manipulation Planning
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决不同几何物体间运动规划问题。通过识别关键位置并传递参考帧，生成有效运动计划。**

- **链接: [https://arxiv.org/pdf/2503.10904v2](https://arxiv.org/pdf/2503.10904v2)**

> **作者:** Dibyendu Das; Aditya Patankar; Nilanjan Chakraborty; C. R. Ramakrishnan; I. V. Ramakrishnan
>
> **备注:** Published in: 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) ---- External Link: https://doi.org/10.1109/IROS60139.2025.11246024
>
> **摘要:** Given a demonstration of a complex manipulation task, such as pouring liquid from one container to another, we seek to generate a motion plan for a new task instance involving objects with different geometries. This is nontrivial since we need to simultaneously ensure that the implicit motion constraints are satisfied (glass held upright while moving), that the motion is collision-free, and that the task is successful (e.g., liquid is poured into the target container). We solve this problem by identifying the positions of critical locations and associating a reference frame (called "motion transfer frames") on the manipulated object and the target, selected based on their geometries and the task at hand. By tracking and transferring the path of the motion transfer frames, we generate motion plans for arbitrary task instances with objects of different geometries and poses. We show results from simulation as well as robot experiments on physical objects to evaluate the effectiveness of our solution. A video supplement is available on YouTube: https://youtu.be/RuG9zMXnfR8
>
---
#### [replaced 009] LaST$_{0}$: Latent Spatio-Temporal Chain-of-Thought for Robotic Vision-Language-Action Model
- **分类: cs.RO**

- **简介: 该论文提出LaST$_0$，解决机器人视觉-语言-动作模型中的推理效率与物理表征不足问题，通过潜在时空思维链实现高效、精准的机器人操作。**

- **链接: [https://arxiv.org/pdf/2601.05248v2](https://arxiv.org/pdf/2601.05248v2)**

> **作者:** Zhuoyang Liu; Jiaming Liu; Hao Chen; Jiale Yu; Ziyu Guo; Chengkai Hou; Chenyang Gu; Xiangju Mi; Renrui Zhang; Kun Wu; Zhengping Che; Jian Tang; Pheng-Ann Heng; Shanghang Zhang
>
> **摘要:** Vision-Language-Action (VLA) models have recently shown strong generalization, with some approaches seeking to explicitly generate linguistic reasoning traces or predict future observations prior to execution. However, explicit reasoning typically incurs non-negligible inference latency, which constrains the temporal resolution required for robotic manipulation. Moreover, such reasoning is confined to the linguistic space, imposing a representational bottleneck that struggles to faithfully capture ineffable physical attributes. To mitigate these limitations, we propose LaST$_0$, a framework that enables efficient reasoning before acting through a Latent Spatio-Temporal Chain-of-Thought (CoT), capturing fine-grained physical and robotic dynamics that are often difficult to verbalize. Specifically, we introduce a token-efficient latent CoT space that models future visual dynamics, 3D structural information, and robot proprioceptive states, and further extends these representations across time to enable temporally consistent implicit reasoning trajectories. Furthermore, LaST$_0$ adopts a dual-system architecture implemented via a Mixture-of-Transformers design, where a reasoning expert conducts low-frequency latent inference and an acting expert generates high-frequency actions conditioned on robotics-oriented latent representations. To facilitate coordination, LaST$_0$ is trained with heterogeneous operation frequencies, enabling adaptive switching during deployment. Across 10 real-world tasks spanning tabletop, mobile, and dexterous hand manipulation, LaST$_0$ improves mean success rates by 13%, 14% and 14% over prior SOTA VLA methods, respectively.
>
---
#### [replaced 010] EgoFSD: Ego-Centric Fully Sparse Paradigm with Uncertainty Denoising and Iterative Refinement for Efficient End-to-End Self-Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出EgoFSD，解决端到端自动驾驶效率与性能问题。通过稀疏感知、层级交互和迭代规划，提升轨迹预测精度与运行效率。**

- **链接: [https://arxiv.org/pdf/2409.09777v5](https://arxiv.org/pdf/2409.09777v5)**

> **作者:** Haisheng Su; Wei Wu; Zhenjie Yang; Isabel Guan
>
> **备注:** Accepted to ICRA2026
>
> **摘要:** Current End-to-End Autonomous Driving (E2E-AD) methods resort to unifying modular designs for various tasks (e.g. perception, prediction and planning). Although optimized with a fully differentiable framework in a planning-oriented manner, existing end-to-end driving systems lacking ego-centric designs still suffer from unsatisfactory performance and inferior efficiency, due to rasterized scene representation learning and redundant information transmission. In this paper, we propose an ego-centric fully sparse paradigm, named EgoFSD, for end-to-end self-driving. Specifically, EgoFSD consists of sparse perception, hierarchical interaction and iterative motion planner. The sparse perception module performs detection and online mapping based on sparse representation of the driving scene. The hierarchical interaction module aims to select the Closest In-Path Vehicle / Stationary (CIPV / CIPS) from coarse to fine, benefiting from an additional geometric prior. As for the iterative motion planner, both selected interactive agents and ego-vehicle are considered for joint motion prediction, where the output multi-modal ego-trajectories are optimized in an iterative fashion. In addition, position-level motion diffusion and trajectory-level planning denoising are introduced for uncertainty modeling, thereby enhancing the training stability and convergence speed. Extensive experiments are conducted on nuScenes and Bench2Drive datasets, which significantly reduces the average L2 error by 59% and collision rate by 92% than UniAD while achieves 6.9x faster running efficiency.
>
---
#### [replaced 011] One-Shot Real-World Demonstration Synthesis for Scalable Bimanual Manipulation
- **分类: cs.RO**

- **简介: 该论文提出BiDemoSyn框架，解决双臂操作中演示数据获取效率与真实感的矛盾，通过单次真实示例生成大量物理可行的演示，提升策略泛化与迁移能力。**

- **链接: [https://arxiv.org/pdf/2512.09297v2](https://arxiv.org/pdf/2512.09297v2)**

> **作者:** Huayi Zhou; Kui Jia
>
> **备注:** Under review. The project link is https://hnuzhy.github.io/projects/BiDemoSyn/
>
> **摘要:** Learning dexterous bimanual manipulation policies critically depends on large-scale, high-quality demonstrations, yet current paradigms face inherent trade-offs: teleoperation provides physically grounded data but is prohibitively labor-intensive, while simulation-based synthesis scales efficiently but suffers from sim-to-real gaps. We present BiDemoSyn, a framework that synthesizes contact-rich, physically feasible bimanual demonstrations from a single real-world example. The key idea is to decompose tasks into invariant coordination blocks and variable, object-dependent adjustments, then adapt them through vision-guided alignment and lightweight trajectory optimization. This enables the generation of thousands of diverse and feasible demonstrations within several hour, without repeated teleoperation or reliance on imperfect simulation. Across six dual-arm tasks, we show that policies trained on BiDemoSyn data generalize robustly to novel object poses and shapes, significantly outperforming recent strong baselines. Beyond the one-shot setting, BiDemoSyn naturally extends to few-shot-based synthesis, improving object-level diversity and out-of-distribution generalization while maintaining strong data efficiency. Moreover, policies trained on BiDemoSyn data exhibit zero-shot cross-embodiment transfer to new robotic platforms, enabled by object-centric observations and a simplified 6-DoF end-effector action representation that decouples policies from embodiment-specific dynamics. By bridging the gap between efficiency and real-world fidelity, BiDemoSyn provides a scalable path toward practical imitation learning for complex bimanual manipulation without compromising physical grounding.
>
---
#### [replaced 012] RF-MatID: Dataset and Benchmark for Radio Frequency Material Identification
- **分类: cs.RO; eess.SP**

- **简介: 该论文属于材料识别任务，旨在解决RF材料识别数据集缺失和模型基准不足的问题。提出RF-MatID数据集，包含多频段、多几何参数的样本，并建立多协议基准评估模型性能。**

- **链接: [https://arxiv.org/pdf/2601.20377v2](https://arxiv.org/pdf/2601.20377v2)**

> **作者:** Xinyan Chen; Qinchun Li; Ruiqin Ma; Jiaqi Bai; Li Yi; Jianfei Yang
>
> **备注:** Accepted by ICLR 2026
>
> **摘要:** Accurate material identification plays a crucial role in embodied AI systems, enabling a wide range of applications. However, current vision-based solutions are limited by the inherent constraints of optical sensors, while radio-frequency (RF) approaches, which can reveal intrinsic material properties, have received growing attention. Despite this progress, RF-based material identification remains hindered by the lack of large-scale public datasets and the limited benchmarking of learning-based approaches. In this work, we present RF-MatID, the first open-source, large-scale, wide-band, and geometry-diverse RF dataset for fine-grained material identification. RF-MatID includes 16 fine-grained categories grouped into 5 superclasses, spanning a broad frequency range from 4 to 43.5 GHz, and comprises 142k samples in both frequency- and time-domain representations. The dataset systematically incorporates controlled geometry perturbations, including variations in incidence angle and stand-off distance. We further establish a multi-setting, multi-protocol benchmark by evaluating state-of-the-art deep learning models, assessing both in-distribution performance and out-of-distribution robustness under cross-angle and cross-distance shifts. The 5 frequency-allocation protocols enable systematic frequency- and region-level analysis, thereby facilitating real-world deployment. RF-MatID aims to enable reproducible research, accelerate algorithmic advancement, foster cross-domain robustness, and support the development of real-world application in RF-based material identification.
>
---
#### [replaced 013] Toward Learning POMDPs Beyond Full-Rank Actions and State Observability
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文研究如何在部分可观测环境下学习POMDP模型，解决隐藏状态和非全秩动作的问题。通过PSR和张量方法，学习观测和转移矩阵，提升模型精度。**

- **链接: [https://arxiv.org/pdf/2601.18930v2](https://arxiv.org/pdf/2601.18930v2)**

> **作者:** Seiji Shaw; Travis Manderson; Chad Kessens; Nicholas Roy
>
> **摘要:** We are interested in enabling autonomous agents to learn and reason about systems with hidden states, such as furniture with hidden locking mechanisms. We cast this problem as learning the parameters of a discrete Partially Observable Markov Decision Process (POMDP). The agent begins with knowledge of the POMDP's actions and observation spaces, but not its state space, transitions, or observation models. These properties must be constructed from action-observation sequences. Spectral approaches to learning models of partially observable domains, such as learning Predictive State Representations (PSRs), are known to directly estimate the number of hidden states. These methods cannot, however, yield direct estimates of transition and observation likelihoods, which are important for many downstream reasoning tasks. Other approaches leverage tensor decompositions to estimate transition and observation likelihoods but often assume full state observability and full-rank transition matrices for all actions. To relax these assumptions, we study how PSRs learn transition and observation matrices up to a similarity transform, which may be estimated via tensor methods. Our method learns observation matrices and transition matrices up to a partition of states, where the states in a single partition have the same observation distributions corresponding to actions whose transition matrices are full-rank. Our experiments suggest that these partition-level transition models learned by our method, with a sufficient amount of data, meets the performance of PSRs as models to be used by standard sampling-based POMDP solvers. Furthermore, the explicit observation and transition likelihoods can be leveraged to specify planner behavior after the model has been learned.
>
---
#### [replaced 014] Virtual Community: An Open World for Humans, Robots, and Society
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文提出Virtual Community平台，用于研究人机共存的开放世界社会。任务是解决人机协作与社会智能问题，通过模拟器和挑战任务评估多智能体的规划与合作能力。**

- **链接: [https://arxiv.org/pdf/2508.14893v2](https://arxiv.org/pdf/2508.14893v2)**

> **作者:** Qinhong Zhou; Hongxin Zhang; Xiangye Lin; Zheyuan Zhang; Yutian Chen; Wenjun Liu; Zunzhe Zhang; Sunli Chen; Lixing Fang; Qiushi Lyu; Xinyu Sun; Jincheng Yang; Zeyuan Wang; Bao Chi Dang; Zhehuan Chen; Daksha Ladia; Jiageng Liu; Chuang Gan
>
> **备注:** website https://virtual-community-ai.github.io/
>
> **摘要:** The rapid progress in AI and Robotics may lead to a profound societal transformation, as humans and robots begin to coexist within shared communities, introducing both opportunities and challenges. To explore this future, we present Virtual Community-an open-world platform for humans, robots, and society-built on a universal physics engine and grounded in real-world 3D scenes. With Virtual Community, we aim to enable the study of embodied social intelligence at scale. To support these, Virtual Community features: 1) An open-source multi-agent physics simulator that supports robots, humans, and their interactions within a society; 2) A large-scale, real-world aligned community generation pipeline, including vast outdoor space, diverse indoor scenes, and a community of grounded agents with rich characters and appearances. Leveraging Virtual Community, we propose two novel challenges. The Community Planning Challenge evaluates multi-agent reasoning and planning ability in open-world settings, such as cooperating to help agents with daily activities and efficiently connecting other agents. The Community Robot Challenge requires multiple heterogeneous robots to collaborate in solving complex open-world tasks. We evaluate various baselines on these tasks and demonstrate the challenges in both high-level open-world task planning and low-level cooperation controls. We hope that Virtual Community will unlock further study of human-robot coexistence within open-world environments.
>
---
#### [replaced 015] Safe and Stable Neural Network Dynamical Systems for Robot Motion Planning
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人运动规划任务，旨在解决从演示中学习安全稳定运动的问题。提出S²-NNDS框架，结合神经动力系统与安全证书，提升运动的鲁棒性与安全性。**

- **链接: [https://arxiv.org/pdf/2511.20593v2](https://arxiv.org/pdf/2511.20593v2)**

> **作者:** Allen Emmanuel Binny; Mahathi Anand; Hugo T. M. Kussaba; Lingyun Chen; Shreenabh Agrawal; Fares J. Abu-Dakka; Abdalla Swikir
>
> **备注:** Accepted for publication in IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Learning safe and stable robot motions from demonstrations remains a challenge, especially in complex, nonlinear tasks involving dynamic, obstacle-rich environments. In this paper, we propose Safe and Stable Neural Network Dynamical Systems S$^2$-NNDS, a learning-from-demonstration framework that simultaneously learns expressive neural dynamical systems alongside neural Lyapunov stability and barrier safety certificates. Unlike traditional approaches with restrictive polynomial parameterizations, S$^2$-NNDS leverages neural networks to capture complex robot motions, providing probabilistic guarantees through split conformal prediction in learned certificates. Experimental results in various 2D and 3D datasets -- including LASA handwriting and demonstrations recorded kinesthetically from the Franka Emika Panda robot -- validate the effectiveness of S$^2$-NNDS in learning robust, safe, and stable motions from potentially unsafe demonstrations. The source code, supplementary material and experiment videos can be accessed via https://github.com/allemmbinn/S2NNDS
>
---
#### [replaced 016] Model Reconciliation through Explainability and Collaborative Recovery in Assistive Robotics
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机协作任务，解决机器人与人类模型不一致问题。通过框架实现模型协调，利用大语言模型解释差异，并允许人类修正机器人模型。**

- **链接: [https://arxiv.org/pdf/2601.06552v2](https://arxiv.org/pdf/2601.06552v2)**

> **作者:** Britt Besch; Tai Mai; Jeremias Thun; Markus Huff; Jörn Vogel; Freek Stulp; Samuel Bustamante
>
> **摘要:** Whenever humans and robots work together, it is essential that unexpected robot behavior can be explained to the user. Especially in applications such as shared control the user and the robot must share the same model of the objects in the world, and the actions that can be performed on these objects. In this paper, we achieve this with a so-called model reconciliation framework. We leverage a Large Language Model to predict and explain the difference between the robot's and the human's mental models, without the need of a formal mental model of the user. Furthermore, our framework aims to solve the model divergence after the explanation by allowing the human to correct the robot. We provide an implementation in an assistive robotics domain, where we conduct a set of experiments with a real wheelchair-based mobile manipulator and its digital twin.
>
---
#### [replaced 017] System Identification for Virtual Sensor-Based Model Predictive Control: Application to a 2-DoF Direct-Drive Robotic Arm
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于控制任务，解决非线性系统建模与关键变量测量问题。通过虚拟传感器识别框架，实现无需高成本传感器的精确控制。**

- **链接: [https://arxiv.org/pdf/2505.23138v2](https://arxiv.org/pdf/2505.23138v2)**

> **作者:** Kosei Tsuji; Ichiro Maruta; Kenji Fujimoto; Tomoyuki Maeda; Yoshihisa Tamase; Tsukasa Shinohara
>
> **备注:** 6 pages, 5 figures. Published in the proceedings of the 2025 IEEE 64th Conference on Decision and Control (CDC 2025)
>
> **摘要:** Nonlinear Model Predictive Control (NMPC) offers a powerful approach for controlling complex nonlinear systems, yet faces two key challenges. First, accurately modeling nonlinear dynamics remains difficult. Second, variables directly related to control objectives often cannot be directly measured during operation. Although high-cost sensors can acquire these variables during model development, their use in practical deployment is typically infeasible. To overcome these limitations, we propose a Predictive Virtual Sensor Identification (PVSID) framework that leverages temporary high-cost sensors during the modeling phase to create virtual sensors for NMPC implementation. We validate PVSID on a Two-Degree-of-Freedom (2-DoF) direct-drive robotic arm with complex joint interactions, capturing tip position via motion capture during modeling and utilize an Inertial Measurement Unit (IMU) in NMPC. Experimental results show our NMPC with identified virtual sensors achieves precise tip trajectory tracking without requiring the motion capture system during operation. PVSID offers a practical solution for implementing optimal control in nonlinear systems where the measurement of key variables is constrained by cost or operational limitations.
>
---
#### [replaced 018] Collision-free Source Seeking and Flocking Control of Multi-agents with Connectivity Preservation
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于多智能体协同控制任务，解决非完整约束下多智能体的避障与编队问题，通过分布式控制算法实现源寻踪与群集保持。**

- **链接: [https://arxiv.org/pdf/2301.04576v3](https://arxiv.org/pdf/2301.04576v3)**

> **作者:** Tinghua Li; Bayu Jayawardhana
>
> **备注:** Published in IEEE Transactions on Automatic Control
>
> **摘要:** In this article, we present a distributed source-seeking and flocking control method for networked multi-agent systems with non-holonomic constraints. Based solely on identical on-board sensor systems, which measure the source local field, the group objective is attained by appointing a leader agent to seek the source while the remaining follower agents safely form a cohesive flocking with their neighbors using a distributed flocking control law in a connectivity-preserved undirected network. To guarantee safe separation and group motion for all agents and to solve the conflicts with the "cohesion" flocking rule of Reynolds, the distributed control algorithm is solved individually through feasible CBF-based optimization problem with complex constraints, which guarantees the inter-agent collision avoidance and connectivity preservation. Stability analysis of the closed-loop system is presented and the efficacy of the methods is shown in simulation results.
>
---
#### [replaced 019] Flexible Multitask Learning with Factorized Diffusion Policy
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人学习任务，解决多任务中策略适应性差的问题。通过因子化扩散策略框架，提升策略灵活性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.21898v2](https://arxiv.org/pdf/2512.21898v2)**

> **作者:** Chaoqi Liu; Haonan Chen; Sigmund H. Høeg; Shaoxiong Yao; Yunzhu Li; Kris Hauser; Yilun Du
>
> **摘要:** Multitask learning poses significant challenges due to the highly multimodal and diverse nature of robot action distributions. However, effectively fitting policies to these complex task distributions is often difficult, and existing monolithic models often underfit the action distribution and lack the flexibility required for efficient adaptation. We introduce a novel modular diffusion policy framework that factorizes complex action distributions into a composition of specialized diffusion models, each capturing a distinct sub-mode of the behavior space for a more effective overall policy. In addition, this modular structure enables flexible policy adaptation to new tasks by adding or fine-tuning components, which inherently mitigates catastrophic forgetting. Empirically, across both simulation and real-world robotic manipulation settings, we illustrate how our method consistently outperforms strong modular and monolithic baselines.
>
---
#### [replaced 020] NeMo-map: Neural Implicit Flow Fields for Spatio-Temporal Motion Mapping
- **分类: cs.RO**

- **简介: 该论文提出NeMo-map，用于建模时空运动模式，解决传统方法离散采样和计算成本高的问题，通过隐式神经函数实现连续高效建模。**

- **链接: [https://arxiv.org/pdf/2510.14827v2](https://arxiv.org/pdf/2510.14827v2)**

> **作者:** Yufei Zhu; Shih-Min Yang; Andrey Rudenko; Tomasz P. Kucner; Achim J. Lilienthal; Martin Magnusson
>
> **备注:** Published as a conference paper at ICLR 2026
>
> **摘要:** Safe and efficient robot operation in complex human environments can benefit from good models of site-specific motion patterns. Maps of Dynamics (MoDs) provide such models by encoding statistical motion patterns in a map, but existing representations use discrete spatial sampling and typically require costly offline construction. We propose a continuous spatio-temporal MoD representation based on implicit neural functions that directly map coordinates to the parameters of a Semi-Wrapped Gaussian Mixture Model. This removes the need for discretization and imputation for unevenly sampled regions, enabling smooth generalization across both space and time. Evaluated on two public datasets with real-world people tracking data, our method achieves better accuracy of motion representation and smoother velocity distributions in sparse regions while still being computationally efficient, compared to available baselines. The proposed approach demonstrates a powerful and efficient way of modeling complex human motion patterns and high performance in the trajectory prediction downstream task. Project code is available at https://github.com/test-bai-cpu/nemo-map
>
---
#### [replaced 021] SPARC: Spine with Prismatic and Revolute Compliance for Quadruped Robots
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出SPARC脊柱模块，解决四足机器人动态运动中缺乏主动顺应性的问题。通过设计具有3-DoF的脊柱结构，提升运动效率与敏捷性。**

- **链接: [https://arxiv.org/pdf/2510.01984v3](https://arxiv.org/pdf/2510.01984v3)**

> **作者:** Yue Wang
>
> **摘要:** Quadruped mammals coordinate spinal bending and axial compression to enhance locomotion agility and efficiency. However, existing robotic spines typically lack the active compliance required to support such dynamic behaviours. We present SPARC, a compact 3-DoF sagittal-plane spine module that enables simultaneous revolute and prismatic motions within a 1.26 kg package. Using a floating-base impedance controller, we facilitate independent, task-space tuning of spinal stiffness and damping to mimic biological load-bearing strategies. Benchtop experiments confirm high-fidelity rendering of commanded impedance, with linear force-displacement error within 1.5%. Systematic locomotion simulations reveal a critical speed-dependency: while low-speed efficiency is insensitive to spinal properties, precise impedance tuning becomes indispensable for high-speed performance. Our results demonstrate that an optimally compliant spine reduces power consumption by 21% at 0.9 m/s compared to a rigid-spine baseline. This efficiency gain is mechanistically attributed to the spine's role in augmenting stride length and acting as a mechanical low-pass filter to attenuate high-frequency torque fluctuations. SPARC provides an open-source platform for systematic studies of spine compliance in legged locomotion. Available at: github.com/YueWang996/sparc
>
---
#### [replaced 022] Robust Trajectory Tracking of Autonomous Surface Vehicle via Lie Algebraic Online MPC
- **分类: cs.RO**

- **简介: 该论文属于自主水面航行器轨迹跟踪任务，旨在解决环境扰动下的精确控制问题。通过结合李代数在线MPC与学习模块，实现高效鲁棒跟踪控制。**

- **链接: [https://arxiv.org/pdf/2511.18683v2](https://arxiv.org/pdf/2511.18683v2)**

> **作者:** Yinan Dong; Ziyu Xu; Tsimafei Lazouski; Sangli Teng; Maani Ghaffari
>
> **摘要:** Autonomous surface vehicles (ASVs) are influenced by environmental disturbances such as wind and waves, making accurate trajectory tracking a persistent challenge in dynamic marine conditions. In this paper, we propose an efficient controller for trajectory tracking of marine vehicles under unknown disturbances by combining a convex error-state MPC on the Lie group augmented by an online learning module to compensate for these disturbances in real time. This design enables adaptive and robust tracking control while maintaining computational efficiency. Extensive evaluations in the Virtual RobotX (VRX) simulator, and real-world field experiments demonstrate that our method achieves superior tracking accuracy under various disturbance scenarios compared with existing approaches.
>
---
#### [replaced 023] Policy Contrastive Decoding for Robotic Foundation Models
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决机器人策略在预训练中学习虚假关联导致泛化能力差的问题。提出Policy Contrastive Decoding方法，通过对比视觉输入提升策略性能。**

- **链接: [https://arxiv.org/pdf/2505.13255v5](https://arxiv.org/pdf/2505.13255v5)**

> **作者:** Shihan Wu; Xu Luo; Ji Zhang; Junlin Xie; Jingkuan Song; Heng Tao Shen; Lianli Gao
>
> **摘要:** Robotic foundation models, or generalist robot policies, hold immense potential to enable flexible, general-purpose and dexterous robotic systems. Despite their advancements, our empirical experiments reveal that existing robot policies are prone to learning spurious correlations from pre-training trajectories, adversely affecting their generalization capabilities beyond the training data. To tackle this, we propose a novel Policy Contrastive Decoding (PCD) approach, which redirects the robot policy's focus toward object-relevant visual clues by contrasting action probability distributions derived from original and object-masked visual inputs. As a training-free method, our PCD can be used as a plugin to improve different types of robot policies without needing to finetune or access model weights. We conduct extensive experiments on top of three open-source robot policies, including the autoregressive policy OpenVLA and the diffusion-based policies Octo and $π_0$. The obtained results in both simulation and real-world environments prove PCD's flexibility and effectiveness, e.g., PCD enhances the state-of-the-art policy $π_0$ by 8.9% in the simulation environment and by 108% in the real-world environment. Code and demos are publicly available at: https://koorye.github.io/PCD.
>
---
#### [replaced 024] DMV-AVP: Distributed Multi-Vehicle Autonomous Valet Parking Using Autoware
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文属于多车辆自动驾驶停车任务，解决分布式协同控制问题。提出DMV-AVP系统，实现多车协调停车模拟，提升系统可扩展性和自主控制能力。**

- **链接: [https://arxiv.org/pdf/2601.16327v2](https://arxiv.org/pdf/2601.16327v2)**

> **作者:** Zubair Islam; Mohamed El-Darieby
>
> **备注:** 6 pages, 3 figures, 3 tables. Submitted to IEEE IV 2026, Demo videos and source code available
>
> **摘要:** This paper presents DMV-AVP, a distributed simulation of Multi-Vehicle Autonomous Valet Parking (AVP). The system was implemented as an application of the Distributed Multi-Autonomous Vehicle Architecture (DMAVA) for synchronized multi-host execution. Most existing simulation approaches rely on centralized or non-distributed designs that constrain scalability and limit fully autonomous control. This work introduces two modules built on top of DMAVA: 1) the Multi-Vehicle AVP Coordination Framework, composed of AVP Managers and a per-vehicle AVP Node, is responsible for global parking state tracking, vehicle queuing, parking spot reservation, lifecycle coordination, and conflict resolution across multiple vehicles, and 2) the Unity-Integrated YOLOv5 Parking Spot Detection Module, that provides real-time, vision-based perception within AWSIM Labs. Both modules integrate seamlessly with DMAVA and extend it specifically for multi-vehicle AVP operation, supported by a Zenoh communication layer that ensures high data accuracy and controllability across hosts. Experiments conducted on two- and three-host configurations demonstrate consistent coordination, conflict-free parking behavior, and scalable performance across distributed Autoware instances. The results confirm that the proposed DMV-AVP supports cooperative AVP simulation and establishes a foundation for future real-world and hardware-in-the-loop validation. Demo videos and source code are available at: https://github.com/zubxxr/multi-vehicle-avp
>
---
#### [replaced 025] Deep Transformer Network for Monocular Pose Estimation of Shipborne Unmanned Aerial Vehicle
- **分类: cs.CV; cs.AI; cs.RO; eess.IV**

- **简介: 该论文属于单目UAV位姿估计任务，旨在准确估算UAV相对于船舶的6D位姿。通过构建合成数据集并训练Transformer模型检测关键点，结合贝叶斯融合实现高精度位姿估计。**

- **链接: [https://arxiv.org/pdf/2406.09260v3](https://arxiv.org/pdf/2406.09260v3)**

> **作者:** Maneesha Wickramasuriya; Taeyoung Lee; Murray Snyder
>
> **备注:** 23 pages, 25 figures, 3 tables
>
> **摘要:** This paper introduces a deep transformer network for estimating the relative 6D pose of a Unmanned Aerial Vehicle (UAV) with respect to a ship using monocular images. A synthetic dataset of ship images is created and annotated with 2D keypoints of multiple ship parts. A Transformer Neural Network model is trained to detect these keypoints and estimate the 6D pose of each part. The estimates are integrated using Bayesian fusion. The model is tested on synthetic data and in-situ flight experiments, demonstrating robustness and accuracy in various lighting conditions. The position estimation error is approximately 0.8\% and 1.0\% of the distance to the ship for the synthetic data and the flight experiments, respectively. The method has potential applications for ship-based autonomous UAV landing and navigation.
>
---
#### [replaced 026] Versatile Behavior Diffusion for Generalized Traffic Agent Simulation
- **分类: cs.RO**

- **简介: 该论文提出VBD框架，用于生成真实、可控的多交通参与者交互场景，解决传统交通模拟不足的问题，提升自动驾驶系统测试效果。**

- **链接: [https://arxiv.org/pdf/2404.02524v3](https://arxiv.org/pdf/2404.02524v3)**

> **作者:** Zhiyu Huang; Zixu Zhang; Ameya Vaidya; Yuxiao Chen; Chen Lv; Jaime Fernández Fisac
>
> **摘要:** Existing traffic simulation models often fall short in capturing the intricacies of real-world scenarios, particularly the interactive behaviors among multiple traffic participants, thereby limiting their utility in the evaluation and validation of autonomous driving systems. We introduce Versatile Behavior Diffusion (VBD), a novel traffic scenario generation framework based on diffusion generative models that synthesizes scene-consistent, realistic, and controllable multi-agent interactions. VBD achieves strong performance in closed-loop traffic simulation, generating scene-consistent agent behaviors that reflect complex agent interactions. A key capability of VBD is inference-time scenario editing through multi-step refinement, guided by behavior priors and model-based optimization objectives, enabling flexible and controllable behavior generation. Despite being trained on real-world traffic datasets with only normal conditions, we introduce conflict-prior and game-theoretic guidance approaches. These approaches enable the generation of interactive, customizable, or long-tail safety-critical scenarios, which are essential for comprehensive testing and validation of autonomous driving systems. Extensive experiments validate the effectiveness and versatility of VBD and highlight its promise as a foundational tool for advancing traffic simulation and autonomous vehicle development. Project website: https://sites.google.com/view/versatile-behavior-diffusion
>
---
#### [replaced 027] GenTrack2: An Improved Hybrid Approach for Visual Multi-Object Tracking
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于多目标跟踪任务，解决未知且动态变化的目标数量下的标识一致性问题。融合随机与确定性机制，提升跟踪精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.24410v2](https://arxiv.org/pdf/2510.24410v2)**

> **作者:** Toan Van Nguyen; Rasmus G. K. Christiansen; Dirk Kraft; Leon Bodenhagen
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** This paper proposes a visual multi-object tracking method that jointly employs stochastic and deterministic mechanisms to ensure identifier consistency for unknown and time-varying target numbers under nonlinear dynamics. A stochastic particle filter addresses nonlinear dynamics and non-Gaussian noise, with support from particle swarm optimization (PSO) to guide particles toward state distribution modes and mitigate divergence through proposed fitness measures incorporating motion consistency, appearance similarity, and social-interaction cues with neighboring targets. Deterministic association further enforces identifier consistency via a proposed cost matrix incorporating spatial consistency between particles and current detections, detection confidences, and track penalties. Subsequently, a novel scheme is proposed for the smooth updating of target states while preserving their identities, particularly for weak tracks during interactions with other targets and prolonged occlusions. Moreover, velocity regression over past states provides trend-seed velocities, enhancing particle sampling and state updates. The proposed tracker is designed to operate flexibly for both pre-recorded videos and camera live streams, where future frames are unavailable. Experimental results confirm superior performance compared to state-of-the-art trackers. The source-code reference implementations of both the proposed method and compared-trackers are provided on GitHub: https://github.com/SDU-VelKoTek/GenTrack2
>
---
#### [replaced 028] Dichotomous Diffusion Policy Optimization
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出DIPOLE算法，解决扩散策略在强化学习中训练不稳定和计算效率低的问题。通过分解策略为最大化与最小化两部分，实现稳定可控的策略优化，应用于自动驾驶等复杂任务。**

- **链接: [https://arxiv.org/pdf/2601.00898v2](https://arxiv.org/pdf/2601.00898v2)**

> **作者:** Ruiming Liang; Yinan Zheng; Kexin Zheng; Tianyi Tan; Jianxiong Li; Liyuan Mao; Zhihao Wang; Guang Chen; Hangjun Ye; Jingjing Liu; Jinqiao Wang; Xianyuan Zhan
>
> **摘要:** Diffusion-based policies have gained growing popularity in solving a wide range of decision-making tasks due to their superior expressiveness and controllable generation during inference. However, effectively training large diffusion policies using reinforcement learning (RL) remains challenging. Existing methods either suffer from unstable training due to directly maximizing value objectives, or face computational issues due to relying on crude Gaussian likelihood approximation, which requires a large amount of sufficiently small denoising steps. In this work, we propose DIPOLE (Dichotomous diffusion Policy improvement), a novel RL algorithm designed for stable and controllable diffusion policy optimization. We begin by revisiting the KL-regularized objective in RL, which offers a desirable weighted regression objective for diffusion policy extraction, but often struggles to balance greediness and stability. We then formulate a greedified policy regularization scheme, which naturally enables decomposing the optimal policy into a pair of stably learned dichotomous policies: one aims at reward maximization, and the other focuses on reward minimization. Under such a design, optimized actions can be generated by linearly combining the scores of dichotomous policies during inference, thereby enabling flexible control over the level of greediness.Evaluations in offline and offline-to-online RL settings on ExORL and OGBench demonstrate the effectiveness of our approach. We also use DIPOLE to train a large vision-language-action (VLA) model for end-to-end autonomous driving (AD) and evaluate it on the large-scale real-world AD benchmark NAVSIM, highlighting its potential for complex real-world applications.
>
---
#### [replaced 029] DMAVA: Distributed Multi-Autonomous Vehicle Architecture Using Autoware
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文提出DMAVA，解决多自动驾驶车辆协同仿真难题，通过分布式架构实现多车协同控制，提升仿真准确性与可扩展性。**

- **链接: [https://arxiv.org/pdf/2601.16336v2](https://arxiv.org/pdf/2601.16336v2)**

> **作者:** Zubair Islam; Mohamed El-Darieby
>
> **备注:** 7 pages, 3 figures, 5 tables, Submitted to IEEE IV 2026, Demo videos and source code available
>
> **摘要:** Simulating and validating coordination among multiple autonomous vehicles remains challenging, as many existing simulation architectures are limited to single-vehicle operation or rely on centralized control. This paper presents the Distributed Multi-Autonomous Vehicle Architecture (DMAVA), a simulation architecture that enables concurrent execution of multiple independent vehicle autonomy stacks distributed across multiple physical hosts within a shared simulation environment. Each vehicle operates its own complete autonomous driving stack while maintaining coordinated behavior through a data-centric communication layer. The proposed system integrates ROS 2 Humble, Autoware Universe, AWSIM Labs, and Zenoh to support high data accuracy and controllability during multi-vehicle simulation, enabling consistent perception, planning, and control behavior under distributed execution. Experiments conducted on multiple-host configurations demonstrate stable localization, reliable inter-host communication, and consistent closed-loop control under distributed execution. DMAVA also serves as a foundation for Multi-Vehicle Autonomous Valet Parking, demonstrating its extensibility toward higher-level cooperative autonomy. Demo videos and source code are available at: https://github.com/zubxxr/distributed-multi-autonomous-vehicle-architecture.
>
---
#### [replaced 030] HI-SLAM2: Geometry-Aware Gaussian SLAM for Fast Monocular Scene Reconstruction
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉SLAM任务，旨在解决单目场景重建中的几何精度与渲染质量平衡问题。通过结合单目先验与学习方法，利用3D高斯泼溅实现高效场景建模。**

- **链接: [https://arxiv.org/pdf/2411.17982v3](https://arxiv.org/pdf/2411.17982v3)**

> **作者:** Wei Zhang; Qing Cheng; David Skuddis; Niclas Zeller; Daniel Cremers; Norbert Haala
>
> **摘要:** We present HI-SLAM2, a geometry-aware Gaussian SLAM system that achieves fast and accurate monocular scene reconstruction using only RGB input. Existing Neural SLAM or 3DGS-based SLAM methods often trade off between rendering quality and geometry accuracy, our research demonstrates that both can be achieved simultaneously with RGB input alone. The key idea of our approach is to enhance the ability for geometry estimation by combining easy-to-obtain monocular priors with learning-based dense SLAM, and then using 3D Gaussian splatting as our core map representation to efficiently model the scene. Upon loop closure, our method ensures on-the-fly global consistency through efficient pose graph bundle adjustment and instant map updates by explicitly deforming the 3D Gaussian units based on anchored keyframe updates. Furthermore, we introduce a grid-based scale alignment strategy to maintain improved scale consistency in prior depths for finer depth details. Through extensive experiments on Replica, ScanNet, and ScanNet++, we demonstrate significant improvements over existing Neural SLAM methods and even surpass RGB-D-based methods in both reconstruction and rendering quality. The project page and source code will be made available at https://hi-slam2.github.io/.
>
---
#### [replaced 031] A Three-Level Whole-Body Disturbance Rejection Control Framework for Dynamic Motions in Legged Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在提升腿式机器人在不确定环境下的稳定性与鲁棒性。提出一种三层次全身体扰动抑制控制框架，解决模型不确定性、外部扰动和故障问题。**

- **链接: [https://arxiv.org/pdf/2508.13531v3](https://arxiv.org/pdf/2508.13531v3)**

> **作者:** Bolin Li; Gewei Zuo; Zhixiang Wang; Xiaotian Ke; Lijun Zhu; Han Ding
>
> **备注:** has been accepted for publication as a SPECIAL ISSUE paper in the IEEE Transactions on Automation Science and Engineering
>
> **摘要:** This paper presents a control framework designed to enhance the stability and robustness of legged robots in the presence of uncertainties, including model uncertainties, external disturbances, and faults. The framework enables the full-state feedback estimator to estimate and compensate for uncertainties in the whole-body dynamics of the legged robots. First, we propose a novel moving horizon extended state observer (MH-ESO) to estimate uncertainties and mitigate noise in legged systems, which can be integrated into the framework for disturbance compensation. Second, we introduce a three-level whole-body disturbance rejection control framework (T-WB-DRC). Unlike the previous two-level approach, this three-level framework considers both the plan based on whole-body dynamics without uncertainties and the plan based on dynamics with uncertainties, significantly improving payload transportation, external disturbance rejection, and fault tolerance. Third, simulations of both humanoid and quadruped robots in the Gazebo simulator demonstrate the effectiveness and versatility of T-WB-DRC. Finally, extensive experimental trials on a quadruped robot validate the robustness and stability of the system when using T-WB-DRC under various disturbance conditions.
>
---
#### [replaced 032] Human-Inspired Neuro-Symbolic World Modeling and Logic Reasoning for Interpretable Safe UAV Landing Site Assessment
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于无人机安全着陆点评估任务，解决传统方法在复杂环境下的可靠性与可解释性问题。提出NeuroSymLand框架，结合感知建模与符号推理，提升评估准确性和透明度。**

- **链接: [https://arxiv.org/pdf/2510.22204v2](https://arxiv.org/pdf/2510.22204v2)**

> **作者:** Weixian Qian; Tianyi Yang; Sebastian Schroder; Yao Deng; Jiaohong Yao; Xiao Cheng; Richard Han; Xi Zheng
>
> **摘要:** Reliable assessment of safe landing sites in unstructured environments is essential for deploying Unmanned Aerial Vehicles (UAVs) in real-world applications such as delivery, inspection, and surveillance. Existing learning-based approaches often degrade under covariate shift and offer limited transparency, making their decisions difficult to interpret and validate on resource-constrained platforms. We present NeuroSymLand, a neuro-symbolic framework for marker-free UAV landing site safety assessment that explicitly separates perception-driven world modeling from logic-based safety reasoning. A lightweight segmentation model incrementally constructs a probabilistic semantic scene graph encoding objects, attributes, and spatial relations. Symbolic safety rules, synthesized offline via large language models with human-in-the-loop refinement, are executed directly over this world model at runtime to perform white-box reasoning, producing ranked landing candidates with human-readable explanations of the underlying safety constraints. Across 72 simulated and hardware-in-the-loop landing scenarios, NeuroSymLand achieves 61 successful assessments, outperforming four competitive baselines, which achieve between 37 and 57 successes. Qualitative analysis highlights its superior interpretability and transparent reasoning, while deployment incurs negligible edge overhead. Our results suggest that combining explicit world modeling with symbolic reasoning can support accurate, interpretable, and edge-deployable safety assessment in mobile systems, as demonstrated through UAV landing site assessment.
>
---
#### [replaced 033] Collaborative Representation Learning for Alignment of Tactile, Language, and Vision Modalities
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于多模态对齐任务，旨在解决触觉、语言和视觉模态间特征冗余与交互不足的问题。提出TLV-CoRe方法，提升跨传感器泛化与模态融合效果。**

- **链接: [https://arxiv.org/pdf/2511.11512v5](https://arxiv.org/pdf/2511.11512v5)**

> **作者:** Yiyun Zhou; Mingjing Xu; Jingwei Shi; Quanjiang Li; Jingyuan Chen
>
> **摘要:** Tactile sensing offers rich and complementary information to vision and language, enabling robots to perceive fine-grained object properties. However, existing tactile sensors lack standardization, leading to redundant features that hinder cross-sensor generalization. Moreover, existing methods fail to fully integrate the intermediate communication among tactile, language, and vision modalities. To address this, we propose TLV-CoRe, a CLIP-based Tactile-Language-Vision Collaborative Representation learning method. TLV-CoRe introduces a Sensor-Aware Modulator to unify tactile features across different sensors and employs tactile-irrelevant decoupled learning to disentangle irrelevant tactile features. Additionally, a Unified Bridging Adapter is introduced to enhance tri-modal interaction within the shared representation space. To fairly evaluate the effectiveness of tactile models, we further propose the RSS evaluation framework, focusing on Robustness, Synergy, and Stability across different methods. Experimental results demonstrate that TLV-CoRe significantly improves sensor-agnostic representation learning and cross-modal alignment, offering a new direction for multimodal tactile representation.
>
---
#### [replaced 034] UAV-Based Infrastructure Inspections: A Literature Review and Proposed Framework for AEC+FM
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于基础设施检测任务，旨在解决传统检测方法效率低的问题。通过综述150余篇文献，提出融合多模态数据的UAV检测框架，提升结构缺陷识别的准确性与可靠性。**

- **链接: [https://arxiv.org/pdf/2601.11665v2](https://arxiv.org/pdf/2601.11665v2)**

> **作者:** Amir Farzin Nikkhah; Dong Chen; Bradford Campbell; Somayeh Asadi; Arsalan Heydarian
>
> **备注:** Withdrawn at the request of the authors to allow further revisions
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) are transforming infrastructure inspections in the Architecture, Engineering, Construction, and Facility Management (AEC+FM) domain. By synthesizing insights from over 150 studies, this review paper highlights UAV-based methodologies for data acquisition, photogrammetric modeling, defect detection, and decision-making support. Key innovations include path optimization, thermal integration, and advanced machine learning (ML) models such as YOLO and Faster R-CNN for anomaly detection. UAVs have demonstrated value in structural health monitoring (SHM), disaster response, urban infrastructure management, energy efficiency evaluations, and cultural heritage preservation. Despite these advancements, challenges in real-time processing, multimodal data fusion, and generalizability remain. A proposed workflow framework, informed by literature and a case study, integrates RGB imagery, LiDAR, and thermal sensing with transformer-based architectures to improve accuracy and reliability in detecting structural defects, thermal anomalies, and geometric inconsistencies. The proposed framework ensures precise and actionable insights by fusing multimodal data and dynamically adapting path planning for complex environments, presented as a comprehensive step-by-step guide to address these challenges effectively. This paper concludes with future research directions emphasizing lightweight AI models, adaptive flight planning, synthetic datasets, and richer modality fusion to streamline modern infrastructure inspections.
>
---
#### [replaced 035] UNIC: Learning Unified Multimodal Extrinsic Contact Estimation
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于接触估计任务，旨在解决无先验知识下的外在接触估计问题。提出UNIC框架，融合视觉、本体和触觉信息，实现可靠、通用的接触检测。**

- **链接: [https://arxiv.org/pdf/2601.04356v2](https://arxiv.org/pdf/2601.04356v2)**

> **作者:** Zhengtong Xu; Yuki Shirai
>
> **摘要:** Contact-rich manipulation requires reliable estimation of extrinsic contacts-the interactions between a grasped object and its environment which provide essential contextual information for planning, control, and policy learning. However, existing approaches often rely on restrictive assumptions, such as predefined contact types, fixed grasp configurations, or camera calibration, that hinder generalization to novel objects and deployment in unstructured environments. In this paper, we present UNIC, a unified multimodal framework for extrinsic contact estimation that operates without any prior knowledge or camera calibration. UNIC directly encodes visual observations in the camera frame and integrates them with proprioceptive and tactile modalities in a fully data-driven manner. It introduces a unified contact representation based on scene affordance maps that captures diverse contact formations and employs a multimodal fusion mechanism with random masking, enabling robust multimodal representation learning. Extensive experiments demonstrate that UNIC performs reliably. It achieves a 9.6 mm average Chamfer distance error on unseen contact locations, performs well on unseen objects, remains robust under missing modalities, and adapts to dynamic camera viewpoints. These results establish extrinsic contact estimation as a practical and versatile capability for contact-rich manipulation. The overview and hardware experiment videos are at https://youtu.be/xpMitkxN6Ls?si=7Vgj-aZ_P1wtnWZN
>
---
#### [replaced 036] CostNav: A Navigation Benchmark for Real-World Economic-Cost Evaluation of Physical AI Agents
- **分类: cs.AI; cs.CE; cs.CV; cs.LG; cs.RO**

- **简介: 该论文提出CostNav，一个用于评估物理AI代理经济成本的导航基准，解决传统导航研究忽略实际经济约束的问题。通过整合真实数据，评估导航策略的商业可行性。**

- **链接: [https://arxiv.org/pdf/2511.20216v2](https://arxiv.org/pdf/2511.20216v2)**

> **作者:** Haebin Seong; Sungmin Kim; Yongjun Cho; Myunchul Joe; Geunwoo Kim; Yubeen Park; Sunhoo Kim; Yoonshik Kim; Suhwan Choi; Jaeyoon Jung; Jiyong Youn; Jinmyung Kwak; Sunghee Ahn; Jaemin Lee; Younggil Do; Seungyeop Yi; Woojin Cheong; Minhyeok Oh; Minchan Kim; Yoonseok Kang; Seongjae Kang; Samwoo Seong; Youngjae Yu; Yunsung Lee
>
> **摘要:** While current navigation benchmarks prioritize task success in simplified settings, they neglect the multidimensional economic constraints essential for the real-world commercialization of autonomous delivery systems. We introduce CostNav, an Economic Navigation Benchmark that evaluates physical AI agents through comprehensive economic cost-revenue analysis aligned with real-world business operations. By integrating industry-standard data - such as SEC filings and AIS injury reports - with Isaac Sim's detailed collision and cargo dynamics, CostNav transcends simple task completion to accurately evaluate business value in complex, real-world scenarios. To our knowledge, CostNav is the first work to quantitatively expose the gap between navigation research metrics and commercial viability, revealing that optimizing for task success on a simplified task fundamentally differs from optimizing for real-world economic deployment. Our evaluation of rule-based Nav2 navigation shows that current approaches are not economically viable: the contribution margin is -22.81/run (AMCL) and -12.87/run (GPS), resulting in no break-even point. We challenge the community to develop navigation policies that achieve economic viability on CostNav. We remain method-agnostic, evaluating success solely on the metric of cost rather than the underlying architecture. All resources are available at https://github.com/worv-ai/CostNav.
>
---
#### [replaced 037] DSCD-Nav: Dual-Stance Cooperative Debate for Object Navigation
- **分类: cs.RO**

- **简介: 该论文属于室内导航任务，解决服务机器人在未知环境中的适应性导航问题。提出DSCD-Nav机制，通过双立场协作辩论提升决策可靠性，减少错误和冗余探索。**

- **链接: [https://arxiv.org/pdf/2601.21409v2](https://arxiv.org/pdf/2601.21409v2)**

> **作者:** Weitao An; Qi Liu; Chenghao Xu; Jiayi Chai; Xu Yang; Kun Wei; Cheng Deng
>
> **摘要:** Adaptive navigation in unfamiliar indoor environments is crucial for household service robots. Despite advances in zero-shot perception and reasoning from vision-language models, existing navigation systems still rely on single-pass scoring at the decision layer, leading to overconfident long-horizon errors and redundant exploration. To tackle these problems, we propose Dual-Stance Cooperative Debate Navigation (DSCD-Nav), a decision mechanism that replaces one-shot scoring with stance-based cross-checking and evidence-aware arbitration to improve action reliability under partial observability. Specifically, given the same observation and candidate action set, we explicitly construct two stances by conditioning the evaluation on diverse and complementary objectives: a Task-Scene Understanding (TSU) stance that prioritizes goal progress from scene-layout cues, and a Safety-Information Balancing (SIB) stance that emphasizes risk and information value. The stances conduct a cooperative debate and make policy by cross-checking their top candidates with cue-grounded arguments. Then, a Navigation Consensus Arbitration (NCA) agent is employed to consolidate both sides' reasons and evidence, optionally triggering lightweight micro-probing to verify uncertain choices, preserving NCA's primary intent while disambiguating. Experiments on HM3Dv1, HM3Dv2, and MP3D demonstrate consistent improvements in success and path efficiency while reducing exploration redundancy.
>
---
#### [replaced 038] Information Filtering via Variational Regularization for Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决视觉-运动策略中中间特征冗余问题。提出变分正则化模块，提升信息过滤效果，显著提高任务成功率。**

- **链接: [https://arxiv.org/pdf/2601.21926v2](https://arxiv.org/pdf/2601.21926v2)**

> **作者:** Jinhao Zhang; Wenlong Xia; Yaojia Wang; Zhexuan Zhou; Huizhe Li; Yichen Lai; Haoming Song; Youmin Gong; Jie Mei
>
> **摘要:** Diffusion-based visuomotor policies built on 3D visual representations have achieved strong performance in learning complex robotic skills. However, most existing methods employ an oversized denoising decoder. While increasing model capacity can improve denoising, empirical evidence suggests that it also introduces redundancy and noise in intermediate feature blocks. Crucially, we find that randomly masking backbone features at inference time (without changing training) can improve performance, confirming the presence of task-irrelevant noise in intermediate features. To this end, we propose Variational Regularization (VR), a lightweight module that imposes a timestep-conditioned Gaussian over backbone features and applies a KL-divergence regularizer, forming an adaptive information bottleneck. Extensive experiments on three simulation benchmarks (RoboTwin2.0, Adroit, and MetaWorld) show that, compared to the baseline DP3, our approach improves the success rate by 6.1% on RoboTwin2.0 and by 4.1% on Adroit and MetaWorld, achieving new state-of-the-art results. Real-world experiments further demonstrate that our method performs well in practical deployments. Code will released.
>
---
#### [replaced 039] VGGT-SLAM 2.0: Real-time Dense Feed-forward Scene Reconstruction
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出VGGT-SLAM 2.0，解决实时稠密场景重建问题。通过改进因子图设计和利用注意力层提升图像检索，提高定位精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.19887v2](https://arxiv.org/pdf/2601.19887v2)**

> **作者:** Dominic Maggio; Luca Carlone
>
> **摘要:** We present VGGT-SLAM 2.0, a real-time RGB feed-forward SLAM system which substantially improves upon VGGT-SLAM for incrementally aligning submaps created from VGGT. Firstly, we remove high-dimensional 15-degree-of-freedom drift and planar degeneracy from VGGT-SLAM by creating a new factor graph design while still addressing the reconstruction ambiguity of VGGT given unknown camera intrinsics. Secondly, by studying the attention layers of VGGT, we show that one of the layers is well suited to assist in image retrieval verification for free without additional training, which enables both rejecting false positive matches and allows for completing more loop closures. Finally, we conduct a suite of experiments which includes showing VGGT-SLAM 2.0 can easily be adapted for open-set object detection and demonstrating real-time performance while running online onboard a ground robot using a Jetson Thor. We test in environments ranging from cluttered indoor apartments and office scenes to a 4,200 square foot barn, and we also demonstrate VGGT-SLAM 2.0 achieves the highest accuracy on the TUM dataset with about 23 percent less pose error than VGGT-SLAM. Code will be released upon publication.
>
---
#### [replaced 040] Reward Evolution with Graph-of-Thoughts: A Bi-Level Language Model Framework for Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决奖励函数设计困难的问题。通过结合大语言模型与视觉语言模型，利用图式思维进行奖励进化，提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2509.16136v2](https://arxiv.org/pdf/2509.16136v2)**

> **作者:** Changwei Yao; Xinzi Liu; Chen Li; Marios Savvides
>
> **摘要:** Designing effective reward functions remains a major challenge in reinforcement learning (RL), often requiring considerable human expertise and iterative refinement. Recent advances leverage Large Language Models (LLMs) for automated reward design, but these approaches are limited by hallucinations, reliance on human feedback, and challenges with handling complex, multi-step tasks. In this work, we introduce Reward Evolution with Graph-of-Thoughts (RE-GoT), a novel bi-level framework that enhances LLMs with structured graph-based reasoning and integrates Visual Language Models (VLMs) for automated rollout evaluation. RE-GoT first decomposes tasks into text-attributed graphs, enabling comprehensive analysis and reward function generation, and then iteratively refines rewards using visual feedback from VLMs without human intervention. Extensive experiments on 10 RoboGen and 4 ManiSkill2 tasks demonstrate that RE-GoT consistently outperforms existing LLM-based baselines. On RoboGen, our method improves average task success rates by 32.25%, with notable gains on complex multi-step tasks. On ManiSkill2, RE-GoT achieves an average success rate of 93.73% across four diverse manipulation tasks, significantly surpassing prior LLM-based approaches and even exceeding expert-designed rewards. Our results indicate that combining LLMs and VLMs with graph-of-thoughts reasoning provides a scalable and effective solution for autonomous reward evolution in RL.
>
---
#### [replaced 041] LangForce: Bayesian Decomposition of Vision Language Action Models via Latent Action Queries
- **分类: cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型在新指令和复杂任务中的泛化问题。通过引入贝叶斯分解框架LangForce，提升语言指导的准确性。**

- **链接: [https://arxiv.org/pdf/2601.15197v5](https://arxiv.org/pdf/2601.15197v5)**

> **作者:** Shijie Lian; Bin Yu; Xiaopeng Lin; Laurence T. Yang; Zhaolong Shen; Changti Wu; Yuzhuo Miao; Cong Huang; Kai Chen
>
> **摘要:** Vision-Language-Action (VLA) models have shown promise in robot manipulation but often struggle to generalize to new instructions or complex multi-task scenarios. We identify a critical pathology in current training paradigms where goal-driven data collection creates a dataset bias. In such datasets, language instructions are highly predictable from visual observations alone, causing the conditional mutual information between instructions and actions to vanish, a phenomenon we term Information Collapse. Consequently, models degenerate into vision-only policies that ignore language constraints and fail in out-of-distribution (OOD) settings. To address this, we propose LangForce, a novel framework that enforces instruction following via Bayesian decomposition. By introducing learnable Latent Action Queries, we construct a dual-branch architecture to estimate both a vision-only prior $p(a \mid v)$ and a language-conditioned posterior $π(a \mid v, \ell)$. We then optimize the policy to maximize the conditional Pointwise Mutual Information (PMI) between actions and instructions. This objective effectively penalizes the vision shortcut and rewards actions that explicitly explain the language command. Without requiring new data, LangForce significantly improves generalization. Extensive experiments across on SimplerEnv and RoboCasa demonstrate substantial gains, including an 11.3% improvement on the challenging OOD SimplerEnv benchmark, validating the ability of our approach to robustly ground language in action.
>
---
#### [replaced 042] Fast Policy Learning for 6-DOF Position Control of Underwater Vehicles
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于水下机器人控制任务，旨在解决6-DOF位置控制问题。针对传统控制器性能下降和RL训练慢的问题，提出GPU加速的RL训练方法，实现快速仿真与真实环境有效控制。**

- **链接: [https://arxiv.org/pdf/2512.13359v2](https://arxiv.org/pdf/2512.13359v2)**

> **作者:** Sümer Tunçay; Alain Andres; Ignacio Carlucho
>
> **摘要:** Autonomous Underwater Vehicles (AUVs) require reliable six-degree-of-freedom (6-DOF) position control to operate effectively in complex and dynamic marine environments. Traditional controllers are effective under nominal conditions but exhibit degraded performance when faced with unmodeled dynamics or environmental disturbances. Reinforcement learning (RL) provides a powerful alternative but training is typically slow and sim-to-real transfer remains challenging. This work introduces a GPU accelerated RL training pipeline built in JAX and MuJoCo-XLA (MJX). By jointly JIT-compiling large-scale parallel physics simulation and learning updates, we achieve training times of under two minutes. Through systematic evaluation of multiple RL algorithms, we show robust 6-DOF trajectory tracking and effective disturbance rejection in real underwater experiments, with policies transferred zero-shot from simulation.
>
---
#### [replaced 043] 3D Dynamics-Aware Manipulation: Endowing Manipulation Policies with 3D Foresight
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决深度方向运动不足的问题。通过引入3D动态建模与策略学习，提升操作策略的3D预见能力，增强鲁棒性。**

- **链接: [https://arxiv.org/pdf/2502.10028v2](https://arxiv.org/pdf/2502.10028v2)**

> **作者:** Yuxin He; Ruihao Zhang; Xianzu Wu; Zhiyuan Zhang; Cheng Ding; Qiang Nie
>
> **备注:** ICRA 2026
>
> **摘要:** The incorporation of world modeling into manipulation policy learning has pushed the boundary of manipulation performance. However, existing efforts simply model the 2D visual dynamics, which is insufficient for robust manipulation when target tasks involve prominent depth-wise movement. To address this, we present a 3D dynamics-aware manipulation framework that seamlessly integrates 3D world modeling and policy learning. Three self-supervised learning tasks (current depth estimation, future RGB-D prediction, 3D flow prediction) are introduced within our framework, which complement each other and endow the policy model with 3D foresight. Extensive experiments on simulation and the real world show that 3D foresight can greatly boost the performance of manipulation policies without sacrificing inference speed. Code is available at https://github.com/Stardust-hyx/3D-Foresight.
>
---
#### [replaced 044] A Survey on Efficient Vision-Language-Action Models
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型研究，旨在提升模型效率。针对计算和数据需求高的问题，提出统一分类框架，涵盖模型设计、训练和数据收集三方面。**

- **链接: [https://arxiv.org/pdf/2510.24795v2](https://arxiv.org/pdf/2510.24795v2)**

> **作者:** Zhaoshu Yu; Bo Wang; Pengpeng Zeng; Haonan Zhang; Ji Zhang; Zheng Wang; Lianli Gao; Jingkuan Song; Nicu Sebe; Heng Tao Shen
>
> **备注:** 28 pages, 8 figures
>
> **摘要:** Vision-Language-Action models (VLAs) represent a significant frontier in embodied intelligence, aiming to bridge digital knowledge with physical-world interaction. Despite their remarkable performance, foundational VLAs are hindered by the prohibitive computational and data demands inherent to their large-scale architectures. While a surge of recent research has focused on enhancing VLA efficiency, the field lacks a unified framework to consolidate these disparate advancements. To bridge this gap, this survey presents the first comprehensive review of Efficient Vision-Language-Action models (Efficient VLAs) across the entire model-training-data pipeline. Specifically, we introduce a unified taxonomy to systematically organize the disparate efforts in this domain, categorizing current techniques into three core pillars: (1) Efficient Model Design, focusing on efficient architectures and model compression; (2) Efficient Training, which reduces computational burdens during model learning; and (3) Efficient Data Collection, which addresses the bottlenecks in acquiring and utilizing robotic data. Through a critical review of state-of-the-art methods within this framework, this survey not only establishes a foundational reference for the community but also summarizes representative applications, delineates key challenges, and charts a roadmap for future research. We maintain a continuously updated project page to track our latest developments: https://evla-survey.github.io/.
>
---
#### [replaced 045] Terrain Costmap Generation via Scaled Preference Conditioning
- **分类: cs.RO**

- **简介: 该论文属于自主导航任务，解决地形成本地图生成问题。提出SPACER方法，既可泛化到新地形，又能快速适应用户偏好。**

- **链接: [https://arxiv.org/pdf/2511.11529v2](https://arxiv.org/pdf/2511.11529v2)**

> **作者:** Luisa Mao; Garrett Warnell; Peter Stone; Joydeep Biswas
>
> **摘要:** Successful autonomous robot navigation in off-road domains requires the ability to generate high-quality terrain costmaps that are able to both generalize well over a wide variety of terrains and rapidly adapt relative costs at test time to meet mission-specific needs. Existing approaches for costmap generation allow for either rapid test-time adaptation of relative costs (e.g., semantic segmentation methods) or generalization to new terrain types (e.g., representation learning methods), but not both. In this work, we present scaled preference conditioned all-terrain costmap generation (SPACER), a novel approach for generating terrain costmaps that leverages synthetic data during training in order to generalize well to new terrains, and allows for rapid test-time adaptation of relative costs by conditioning on a user-specified scaled preference context. Using large-scale aerial maps, we provide empirical evidence that SPACER outperforms other approaches at generating costmaps for terrain navigation, with the lowest measured regret across varied preferences in five of seven environments for global path planning.
>
---
#### [replaced 046] MOBIUS: A Multi-Modal Bipedal Robot that can Walk, Crawl, Climb, and Roll
- **分类: cs.RO; eess.SY**

- **简介: 该论文介绍MOBIUS机器人，解决多模式移动与操作问题，通过混合控制和规划实现行走、攀爬等动作，提升机器人交互能力。**

- **链接: [https://arxiv.org/pdf/2511.01774v2](https://arxiv.org/pdf/2511.01774v2)**

> **作者:** Alexander Schperberg; Yusuke Tanaka; Stefano Di Cairano; Dennis Hong
>
> **备注:** Collaborative work between the Robotics and Mechanisms Laboratory (RoMeLa) and Mitsubishi Electric Research Laboratories (MERL)
>
> **摘要:** This paper presents the MOBIUS platform, a bipedal robot capable of walking, crawling, climbing, and rolling. MOBIUS features four limbs, two 6-DoF arms with two-finger grippers for manipulation and climbing, and two 4-DoF legs for locomotion--enabling smooth transitions across diverse terrains without reconfiguration. A hybrid control architecture combines reinforcement learning for locomotion and force control for compliant contact interactions during manipulation. A high-level MIQCP planner autonomously selects locomotion modes to balance stability and energy efficiency. Hardware experiments demonstrate robust gait transitions, dynamic climbing, and full-body load support via pinch grasp. Overall, MOBIUS demonstrates the importance of tight integration between morphology, high-level planning, and control to enable mobile loco-manipulation and grasping, substantially expanding its interaction capabilities, workspace, and traversability.
>
---
#### [replaced 047] Field evaluation and optimization of a lightweight autonomous lidar-based UAV system based on a rigorous experimental setup in boreal forest environments
- **分类: cs.RO**

- **简介: 该论文属于无人机自主飞行任务，旨在解决森林环境下自主导航评估标准缺失的问题。提出标准化实验方案，并优化了轻量级激光雷达无人机系统。**

- **链接: [https://arxiv.org/pdf/2512.14340v2](https://arxiv.org/pdf/2512.14340v2)**

> **作者:** Aleksi Karhunen; Teemu Hakala; Väinö Karjalainen; Eija Honkavaara
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Interest in utilizing autonomous uncrewed aerial vehicles (UAVs) for under-canopy forest remote sensing has increased in recent years, resulting in the publication of numerous autonomous flight algorithms in the scientific literature. To support the selection and development of such algorithms, a reliable comparison of existing approaches based on published studies is essential. However, reliable comparisons are currently challenging due to widely varying experimental setups and incomplete reporting practices. This study proposes a standardized experimental setup for evaluating autonomous under-canopy UAV systems to fill this gap. The proposed setup emphasizes quantitative reporting of forest complexity, visual representation of test environments, execution of multiple repeated flights, and reporting of flight success rates alongside qualitative flight results. In addition, flights at multiple target speeds are encouraged, with reporting of realized flight speed, mission completion time, and point-to-point flight distance. The proposed setup is demonstrated using a lightweight lidar-based quadrotor employing state-of-the-art open-source algorithms, evaluated through extensive experiments in two natural boreal forest environments. Based on a systematic evaluation of the original system, several improvements were introduced. The same experimental protocol was then repeated with the optimized system, resulting in a total of 93 real-world flights. The optimized system achieved success rates of 12/15 and 15/15 at target flight speeds of 1 m/s and 2 m/s, respectively, in a medium-difficulty forest, and 12/15 and 5/15 in a difficult forest. Adoption of the proposed experimental setup would facilitate the literature-based comparison of autonomous under-canopy flight systems and support systematic performance improvement of future UAV-based forest robotics solutions.
>
---
#### [replaced 048] Graph-Based Floor Separation Using Node Embeddings and Clustering of WiFi Trajectories
- **分类: cs.NI; cs.AI; cs.CR; cs.LG; cs.RO**

- **简介: 该论文属于室内定位任务，解决多层建筑中的楼层分离问题。通过构建Wi-Fi轨迹图并应用节点嵌入与聚类方法，实现无需建筑信息的楼层划分。**

- **链接: [https://arxiv.org/pdf/2505.08088v3](https://arxiv.org/pdf/2505.08088v3)**

> **作者:** Rabia Yasa Kostas; Kahraman Kostas
>
> **备注:** 10 pages,4 figures, 3 tables
>
> **摘要:** Vertical localization, particularly floor separation, remains a major challenge in indoor positioning systems operating in GPS-denied multistory environments. This paper proposes a fully data-driven, graph-based framework for blind floor separation using only Wi-Fi fingerprint trajectories, without requiring prior building information or knowledge of the number of floors. In the proposed method, Wi-Fi fingerprints are represented as nodes in a trajectory graph, where edges capture both signal similarity and sequential movement context. Structural node embeddings are learned via Node2Vec, and floor-level partitions are obtained using K-Means clustering with automatic cluster number estimation. The framework is evaluated on multiple publicly available datasets, including a newly released Huawei University Challenge 2021 dataset and a restructured version of the UJIIndoorLoc benchmark. Experimental results demonstrate that the proposed approach effectively captures the intrinsic vertical structure of multistory buildings using only received signal strength data. By eliminating dependence on building-specific metadata, the proposed method provides a scalable and practical solution for vertical localization in indoor environments.
>
---
#### [replaced 049] A Dataset and Benchmark for Robotic Cloth Unfolding Grasp Selection: The ICRA 2024 Cloth Competition
- **分类: cs.RO**

- **简介: 该论文聚焦于机器人布料展开抓取任务，旨在解决缺乏标准化基准和数据集的问题。通过构建数据集并组织竞赛，评估不同抓取方法，推动布料操作研究发展。**

- **链接: [https://arxiv.org/pdf/2508.16749v2](https://arxiv.org/pdf/2508.16749v2)**

> **作者:** Victor-Louis De Gusseme; Thomas Lips; Remko Proesmans; Julius Hietala; Giwan Lee; Jiyoung Choi; Jeongil Choi; Geon Kim; Phayuth Yonrith; Domen Tabernik; Andrej Gams; Peter Nimac; Matej Urbas; Jon Muhovič; Danijel Skočaj; Matija Mavsar; Hyojeong Yu; Minseo Kwon; Young J. Kim; Yang Cong; Ronghan Chen; Yu Ren; Supeng Diao; Jiawei Weng; Jiayue Liu; Haoran Sun; Linhan Yang; Zeqing Zhang; Ning Guo; Lei Yang; Fang Wan; Chaoyang Song; Jia Pan; Yixiang Jin; Yong A; Jun Shi; Dingzhe Li; Yong Yang; Kakeru Yamasaki; Takumi Kajiwara; Yuki Nakadera; Krati Saxena; Tomohiro Shibata; Chongkun Xia; Kai Mo; Yanzhao Yu; Qihao Lin; Binqiang Ma; Uihun Sagong; JungHyun Choi; JeongHyun Park; Dongwoo Lee; Yeongmin Kim; Myun Joong Hwang; Yusuke Kuribayashi; Naoki Hiratsuka; Daisuke Tanaka; Solvi Arnold; Kimitoshi Yamazaki; Carlos Mateo-Agullo; Andreas Verleysen; Francis Wyffels
>
> **备注:** The International Journal of Robotics Research. 2026;0(0). Published at IJRR - https://journals.sagepub.com/doi/10.1177/02783649251414885
>
> **摘要:** Robotic cloth manipulation suffers from a lack of standardized benchmarks and shared datasets for evaluating and comparing different approaches. To address this, we created a benchmark and organized the ICRA 2024 Cloth Competition, a unique head-to-head evaluation focused on grasp pose selection for in-air robotic cloth unfolding. Eleven diverse teams participated in the competition, utilizing our publicly released dataset of real-world robotic cloth unfolding attempts and a variety of methods to design their unfolding approaches. Afterwards, we also expanded our dataset with 176 competition evaluation trials, resulting in a dataset of 679 unfolding demonstrations across 34 garments. Analysis of the competition results revealed insights about the trade-off between grasp success and coverage, the surprisingly strong achievements of hand-engineered methods and a significant discrepancy between competition performance and prior work, underscoring the importance of independent, out-of-the-lab evaluation in robotic cloth manipulation. The associated dataset is a valuable resource for developing and evaluating grasp selection methods, particularly for learning-based approaches. We hope that our benchmark, dataset and competition results can serve as a foundation for future benchmarks and drive further progress in data-driven robotic cloth manipulation. The dataset and benchmarking code are available at https://airo.ugent.be/cloth_competition.
>
---
#### [replaced 050] What does really matter in image goal navigation?
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究图像目标导航任务，探讨是否可通过强化学习端到端训练实现相对位姿估计。工作包括分析架构选择对导航性能的影响，并验证其在真实场景中的迁移能力。**

- **链接: [https://arxiv.org/pdf/2507.01667v2](https://arxiv.org/pdf/2507.01667v2)**

> **作者:** Gianluca Monaci; Philippe Weinzaepfel; Christian Wolf
>
> **摘要:** Image goal navigation requires two different skills: firstly, core navigation skills, including the detection of free space and obstacles, and taking decisions based on an internal representation; and secondly, computing directional information by comparing visual observations to the goal image. Current state-of-the-art methods either rely on dedicated image-matching, or pre-training of computer vision modules on relative pose estimation. In this paper, we study whether this task can be efficiently solved with end-to-end training of full agents with RL, as has been claimed by recent work. A positive answer would have impact beyond Embodied AI and allow training of relative pose estimation from reward for navigation alone. In this large experimental study we investigate the effect of architectural choices like late fusion, channel stacking, space-to-depth projections and cross-attention, and their role in the emergence of relative pose estimators from navigation training. We show that the success of recent methods is influenced up to a certain extent by simulator settings, leading to shortcuts in simulation. However, we also show that these capabilities can be transferred to more realistic setting, up to some extent. We also find evidence for correlations between navigation performance and probed (emerging) relative pose estimation performance, an important sub skill.
>
---
#### [replaced 051] InterMimic: Towards Universal Whole-Body Control for Physics-Based Human-Object Interactions
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文提出InterMimic框架，解决物理模拟中人类与物体交互的仿真实现问题。通过课程学习和强化学习，实现多样且真实的全身体互动生成。**

- **链接: [https://arxiv.org/pdf/2502.20390v2](https://arxiv.org/pdf/2502.20390v2)**

> **作者:** Sirui Xu; Hung Yu Ling; Yu-Xiong Wang; Liang-Yan Gui
>
> **备注:** CVPR 2025. Project Page: https://sirui-xu.github.io/InterMimic/
>
> **摘要:** Achieving realistic simulations of humans interacting with a wide range of objects has long been a fundamental goal. Extending physics-based motion imitation to complex human-object interactions (HOIs) is challenging due to intricate human-object coupling, variability in object geometries, and artifacts in motion capture data, such as inaccurate contacts and limited hand detail. We introduce InterMimic, a framework that enables a single policy to robustly learn from hours of imperfect MoCap data covering diverse full-body interactions with dynamic and varied objects. Our key insight is to employ a curriculum strategy -- perfect first, then scale up. We first train subject-specific teacher policies to mimic, retarget, and refine motion capture data. Next, we distill these teachers into a student policy, with the teachers acting as online experts providing direct supervision, as well as high-quality references. Notably, we incorporate RL fine-tuning on the student policy to surpass mere demonstration replication and achieve higher-quality solutions. Our experiments demonstrate that InterMimic produces realistic and diverse interactions across multiple HOI datasets. The learned policy generalizes in a zero-shot manner and seamlessly integrates with kinematic generators, elevating the framework from mere imitation to generative modeling of complex human-object interactions.
>
---
#### [replaced 052] From Edge to Edge: A Flow-Inspired Scheduling Planner for Multi-Robot Systems
- **分类: cs.RO**

- **简介: 该论文属于多机器人路径规划任务，解决复杂环境中机器人协同高效穿越的问题。提出一种基于网络流的实时调度方案，优化机器人路径分配，提升整体 traversal 效率。**

- **链接: [https://arxiv.org/pdf/2409.06952v3](https://arxiv.org/pdf/2409.06952v3)**

> **作者:** Han Liu; Yu Jin; Mingyue Cui; Boyang Li; Tianjiang Hu; Kai Huang
>
> **摘要:** Trajectory planning is crucial in multi-robot systems, particularly in environments with numerous obstacles. While extensive research has been conducted in this field, the challenge of coordinating multiple robots to flow collectively from one side of the map to the other-such as in crossing missions through obstacle-rich spaces-has received limited attention. This paper focuses on this directional traversal scenario by introducing a real-time scheduling scheme that enables multi-robot systems to move from edge to edge, emulating the smooth and efficient flow of water. Inspired by network flow optimization, our scheme decomposes the environment into a flow-based network structure, enabling the efficient allocation of robots to paths based on real-time congestion levels. The proposed scheduling planner operates on top of existing collision avoidance algorithms, aiming to minimize overall traversal time by balancing detours and waiting times. Simulation results demonstrate the effectiveness of the proposed scheme in achieving fast and coordinated traversal. Furthermore, real-world flight tests with ten drones validate its practical feasibility. This work contributes a flow-inspired, real-time scheduling planner tailored for directional multi-robot traversal in complex, obstacle-rich environments. Code: https://github.com/chengji253/FlowPlanner
>
---
