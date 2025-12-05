# 机器人 cs.RO

- **最新发布 32 篇**

- **更新 18 篇**

## 最新发布

#### [new 001] One Ring to Rule Them All: Constrained Distributional Control for Massive-Scale Heterogeneous Robotic Ensemble Systems
- **分类: cs.RO**

- **简介: 该论文研究大规模异构机器人集群的协同控制问题，旨在通过单一共享控制器实现群体在复杂约束环境下的安全高效运动。提出基于矩核变换的约束集控框架，结合信号时序逻辑编码任务，统一处理避障与路径规划。**

- **链接: [https://arxiv.org/pdf/2512.04502v1](https://arxiv.org/pdf/2512.04502v1)**

> **作者:** Andres Arias; Wei Zhang; Haoyu Qian; Jr-Shin Li; Chuangchuang Sun
>
> **备注:** 9 pages, 8 figures
>
> **摘要:** Ensemble control aims to steer a population of dynamical systems using a shared control input. This paper introduces a constrained ensemble control framework for parameterized, heterogeneous robotic systems operating under state and environmental constraints, such as obstacle avoidance. We develop a moment kernel transform that maps the parameterized ensemble dynamics to the moment system in a kernel space, enabling the characterization of population-level behavior. The state-space constraints, such as polyhedral waypoints to be visited and obstacles to be avoided, are also transformed into the moment space, leading to a unified formulation for safe, large-scale ensemble control. Expressive signal temporal logic specifications are employed to encode complex visit-avoid tasks, which are achieved through a single shared controller synthesized from our constrained ensemble control formulation. Simulation and hardware experiments demonstrate the effectiveness of the proposed approach in safely and efficiently controlling robotic ensembles within constrained environments.
>
---
#### [new 002] TEMPO-VINE: A Multi-Temporal Sensor Fusion Dataset for Localization and Mapping in Vineyards
- **分类: cs.RO; eess.SY**

- **简介: 该论文聚焦农业机器人定位与建图任务，针对缺乏真实复杂环境下多模态基准数据的问题，构建了首个涵盖多传感器、多时段、多种条件的公开葡萄园数据集TEMPO-VINE，支持传感器融合、SLAM与场景识别方法评估。**

- **链接: [https://arxiv.org/pdf/2512.04772v1](https://arxiv.org/pdf/2512.04772v1)**

> **作者:** Mauro Martini; Marco Ambrosio; Judith Vilella-Cantos; Alessandro Navone; Marcello Chiaberge
>
> **摘要:** In recent years, precision agriculture has been introducing groundbreaking innovations in the field, with a strong focus on automation. However, research studies in robotics and autonomous navigation often rely on controlled simulations or isolated field trials. The absence of a realistic common benchmark represents a significant limitation for the diffusion of robust autonomous systems under real complex agricultural conditions. Vineyards pose significant challenges due to their dynamic nature, and they are increasingly drawing attention from both academic and industrial stakeholders interested in automation. In this context, we introduce the TEMPO-VINE dataset, a large-scale multi-temporal dataset specifically designed for evaluating sensor fusion, simultaneous localization and mapping (SLAM), and place recognition techniques within operational vineyard environments. TEMPO-VINE is the first multi-modal public dataset that brings together data from heterogeneous LiDARs of different price levels, AHRS, RTK-GPS, and cameras in real trellis and pergola vineyards, with multiple rows exceeding 100 m in length. In this work, we address a critical gap in the landscape of agricultural datasets by providing researchers with a comprehensive data collection and ground truth trajectories in different seasons, vegetation growth stages, terrain and weather conditions. The sequence paths with multiple runs and revisits will foster the development of sensor fusion, localization, mapping and place recognition solutions for agricultural fields. The dataset, the processing tools and the benchmarking results will be available at the dedicated webpage upon acceptance.
>
---
#### [new 003] RoboBPP: Benchmarking Robotic Online Bin Packing with Physics-based Simulation
- **分类: cs.RO**

- **简介: 该论文针对机器人在线装箱任务，解决因标准不一、数据失真和评估不足导致的研究瓶颈。提出RoboBPP，集成物理仿真、真实工业数据集、新评估指标与开源平台，支持可复现、可扩展的算法评测。**

- **链接: [https://arxiv.org/pdf/2512.04415v1](https://arxiv.org/pdf/2512.04415v1)**

> **作者:** Zhoufeng Wang; Hang Zhao; Juzhan Xu; Shishun Zhang; Zeyu Xiong; Ruizhen Hu; Chenyang Zhu; Kai Xu
>
> **备注:** Under review at the International Journal of Robotics Research (IJRR)
>
> **摘要:** Physical feasibility in 3D bin packing is a key requirement in modern industrial logistics and robotic automation. With the growing adoption of industrial automation, online bin packing has gained increasing attention. However, inconsistencies in problem settings, test datasets, and evaluation metrics have hindered progress in the field, and there is a lack of a comprehensive benchmarking system. Direct testing on real hardware is costly, and building a realistic simulation environment is also challenging. To address these limitations, we introduce RoboBPP, a benchmarking system designed for robotic online bin packing. RoboBPP integrates a physics-based simulator to assess physical feasibility. In our simulation environment, we introduce a robotic arm and boxes at real-world scales to replicate real industrial packing workflows. By simulating conditions that arise in real industrial applications, we ensure that evaluated algorithms are practically deployable. In addition, prior studies often rely on synthetic datasets whose distributions differ from real-world industrial data. To address this issue, we collect three datasets from real industrial workflows, including assembly-line production, logistics packing, and furniture manufacturing. The benchmark comprises three carefully designed test settings and extends existing evaluation metrics with new metrics for structural stability and operational safety. We design a scoring system and derive a range of insights from the evaluation results. RoboBPP is fully open-source and is equipped with visualization tools and an online leaderboard, providing a reproducible and extensible foundation for future research and industrial applications (https://robot-bin-packing-benchmark.github.io).
>
---
#### [new 004] Vertical Planetary Landing on Sloped Terrain Using Optical Flow Divergence Estimates
- **分类: cs.RO**

- **简介: 该论文研究小型航天器在斜坡地形上的自主着陆任务，旨在解决资源受限下着陆控制难题。提出基于局部光流散度估计的非线性控制方法，通过调节推力与姿态，实现稳定着陆与地形对齐。**

- **链接: [https://arxiv.org/pdf/2512.04373v1](https://arxiv.org/pdf/2512.04373v1)**

> **作者:** Hann Woei Ho; Ye Zhou
>
> **备注:** This paper is accepted at International Astronautical Congress (IAC 2025)
>
> **摘要:** Autonomous landing on sloped terrain poses significant challenges for small, lightweight spacecraft, such as rotorcraft and landers. These vehicles have limited processing capability and payload capacity, which makes advanced deep learning methods and heavy sensors impractical. Flying insects, such as bees, achieve remarkable landings with minimal neural and sensory resources, relying heavily on optical flow. By regulating flow divergence, a measure of vertical velocity divided by height, they perform smooth landings in which velocity and height decay exponentially together. However, adapting this bio-inspired strategy for spacecraft landings on sloped terrain presents two key challenges: global flow-divergence estimates obscure terrain inclination, and the nonlinear nature of divergence-based control can lead to instability when using conventional controllers. This paper proposes a nonlinear control strategy that leverages two distinct local flow divergence estimates to regulate both thrust and attitude during vertical landings. The control law is formulated based on Incremental Nonlinear Dynamic Inversion to handle the nonlinear flow divergence. The thrust control ensures a smooth vertical descent by keeping a constant average of the local flow divergence estimates, while the attitude control aligns the vehicle with the inclined surface at touchdown by exploiting their difference. The approach is evaluated in numerical simulations using a simplified 2D spacecraft model across varying slopes and divergence setpoints. Results show that regulating the average divergence yields stable landings with exponential decay of velocity and height, and using the divergence difference enables effective alignment with inclined terrain. Overall, the method offers a robust, low-resource landing strategy that enhances the feasibility of autonomous planetary missions with small spacecraft.
>
---
#### [new 005] Bridging Simulation and Reality: Cross-Domain Transfer with Semantic 2D Gaussian Splatting
- **分类: cs.RO**

- **简介: 该论文研究机器人操作中的跨域迁移问题，旨在缩小仿真与现实的差距。提出语义2D高斯点阵（S2GS）方法，提取对象中心、域不变特征，结合语义滤波提升策略迁移性能，实现在真实场景中高效部署。**

- **链接: [https://arxiv.org/pdf/2512.04731v1](https://arxiv.org/pdf/2512.04731v1)**

> **作者:** Jian Tang; Pu Pang; Haowen Sun; Chengzhong Ma; Xingyu Chen; Hua Huang; Xuguang Lan
>
> **摘要:** Cross-domain transfer in robotic manipulation remains a longstanding challenge due to the significant domain gap between simulated and real-world environments. Existing methods such as domain randomization, adaptation, and sim-real calibration often require extensive tuning or fail to generalize to unseen scenarios. To address this issue, we observe that if domain-invariant features are utilized during policy training in simulation, and the same features can be extracted and provided as the input to policy during real-world deployment, the domain gap can be effectively bridged, leading to significantly improved policy generalization. Accordingly, we propose Semantic 2D Gaussian Splatting (S2GS), a novel representation method that extracts object-centric, domain-invariant spatial features. S2GS constructs multi-view 2D semantic fields and projects them into a unified 3D space via feature-level Gaussian splatting. A semantic filtering mechanism removes irrelevant background content, ensuring clean and consistent inputs for policy learning. To evaluate the effectiveness of S2GS, we adopt Diffusion Policy as the downstream learning algorithm and conduct experiments in the ManiSkill simulation environment, followed by real-world deployment. Results demonstrate that S2GS significantly improves sim-to-real transferability, maintaining high and stable task performance in real-world scenarios.
>
---
#### [new 006] Hoi! - A Multimodal Dataset for Force-Grounded, Cross-View Articulated Manipulation
- **分类: cs.RO**

- **简介: 该论文构建了一个力感知、多视角的铰接物体操作数据集Hoi!，包含人类与机器人在不同工具形态下的操作视频、力觉与触觉同步数据，旨在推动跨视角操作理解及力觉预测等多模态交互研究。**

- **链接: [https://arxiv.org/pdf/2512.04884v1](https://arxiv.org/pdf/2512.04884v1)**

> **作者:** Tim Engelbracht; René Zurbrügg; Matteo Wohlrapp; Martin Büchner; Abhinav Valada; Marc Pollefeys; Hermann Blum; Zuria Bauer
>
> **摘要:** We present a dataset for force-grounded, cross-view articulated manipulation that couples what is seen with what is done and what is felt during real human interaction. The dataset contains 3048 sequences across 381 articulated objects in 38 environments. Each object is operated under four embodiments - (i) human hand, (ii) human hand with a wrist-mounted camera, (iii) handheld UMI gripper, and (iv) a custom Hoi! gripper - where the tool embodiment provides synchronized end-effector forces and tactile sensing. Our dataset offers a holistic view of interaction understanding from video, enabling researchers to evaluate how well methods transfer between human and robotic viewpoints, but also investigate underexplored modalities such as force sensing and prediction.
>
---
#### [new 007] FALCON: Actively Decoupled Visuomotor Policies for Loco-Manipulation with Foundation-Model-Based Coordination
- **分类: cs.RO**

- **简介: 该论文研究具身智能中的移动操作（loco-manipulation）任务，旨在解决多模态感知融合导致的性能下降问题。作者提出FALCON框架，将移动与操作解耦为两个独立策略，利用视觉语言基础模型协调二者，并引入阶段进展头和对比损失增强协调与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.04381v1](https://arxiv.org/pdf/2512.04381v1)**

> **作者:** Chengyang He; Ge Sun; Yue Bai; Junkai Lu; Jiadong Zhao; Guillaume Sartoretti
>
> **摘要:** We present FoundAtion-model-guided decoupled LoCO-maNipulation visuomotor policies (FALCON), a framework for loco-manipulation that combines modular diffusion policies with a vision-language foundation model as the coordinator. Our approach explicitly decouples locomotion and manipulation into two specialized visuomotor policies, allowing each subsystem to rely on its own observations. This mitigates the performance degradation that arise when a single policy is forced to fuse heterogeneous, potentially mismatched observations from locomotion and manipulation. Our key innovation lies in restoring coordination between these two independent policies through a vision-language foundation model, which encodes global observations and language instructions into a shared latent embedding conditioning both diffusion policies. On top of this backbone, we introduce a phase-progress head that uses textual descriptions of task stages to infer discrete phase and continuous progress estimates without manual phase labels. To further structure the latent space, we incorporate a coordination-aware contrastive loss that explicitly encodes cross-subsystem compatibility between arm and base actions. We evaluate FALCON on two challenging loco-manipulation tasks requiring navigation, precise end-effector placement, and tight base-arm coordination. Results show that it surpasses centralized and decentralized baselines while exhibiting improved robustness and generalization to out-of-distribution scenarios.
>
---
#### [new 008] Introducing V-Soft Pro: a Modular Platform for a Transhumeral Prosthesis with Controllable Stiffness
- **分类: cs.RO**

- **简介: 该论文提出一种带可控刚度的模块化上臂假肢V-Soft Pro，旨在解决现有假肢无法模拟人臂自然运动与交互的问题。通过引入变刚度驱动器和弹性元件，实现刚度调节与自然运动，提升假肢适应性与安全性。**

- **链接: [https://arxiv.org/pdf/2512.04998v1](https://arxiv.org/pdf/2512.04998v1)**

> **作者:** Giuseppe Milazzo; Giorgio Grioli; Antonio Bicchi; Manuel G. Catalano
>
> **备注:** This article has been accepted for publication in Proceedings of the International Conference On Rehabilitation Robotics (ICORR), 2025. This is the author's version, which has not been fully edited, and content may change prior to final publication. Citation information: DOI 10.1109/ICORR66766.2025.11062964
>
> **摘要:** Current upper limb prostheses aim to enhance user independence in daily activities by incorporating basic motor functions. However, they fall short of replicating the natural movement and interaction capabilities of the human arm. In contrast, human limbs leverage intrinsic compliance and actively modulate joint stiffness, enabling adaptive responses to varying tasks, impact absorption, and efficient energy transfer during dynamic actions. Inspired by this adaptability, we developed a transhumeral prosthesis with Variable Stiffness Actuators (VSAs) to replicate the controllable compliance found in biological joints. The proposed prosthesis features a modular design, allowing customization for different residual limb shapes and accommodating a range of independent control signals derived from users' biological cues. Integrated elastic elements passively support more natural movements, facilitate safe interactions with the environment, and adapt to diverse task requirements. This paper presents a comprehensive overview of the platform and its functionalities, highlighting its potential applications in the field of prosthetics.
>
---
#### [new 009] Bridging Probabilistic Inference and Behavior Trees: An Interactive Framework for Adaptive Multi-Robot Cooperation
- **分类: cs.RO**

- **简介: 该论文提出交互式推理行为树（IIBT）框架，将概率推理融入行为树，解决多机器人在动态环境中自适应协作决策问题。通过自由能最小化实现联合规划与执行，兼容传统BT，降低70%节点复杂度，提升不确定性下的协作鲁棒性与可解释性。**

- **链接: [https://arxiv.org/pdf/2512.04404v1](https://arxiv.org/pdf/2512.04404v1)**

> **作者:** Chaoran Wang; Jingyuan Sun; Yanhui Zhang; Changju Wu
>
> **备注:** 34 pages, is submitted RAS Journal
>
> **摘要:** This paper proposes an Interactive Inference Behavior Tree (IIBT) framework that integrates behavior trees (BTs) with active inference under the free energy principle for distributed multi-robot decision-making. The proposed IIBT node extends conventional BTs with probabilistic reasoning, enabling online joint planning and execution across multiple robots. It remains fully com- patible with standard BT architectures, allowing seamless integration into existing multi-robot control systems. Within this framework, multi-robot cooperation is formulated as a free-energy minimization process, where each robot dynamically updates its preference matrix based on perceptual inputs and peer intentions, thereby achieving adaptive coordination in partially observ- able and dynamic environments. The proposed approach is validated through both simulation and real-world experiments, including a multi-robot maze navigation and a collaborative ma- nipulation task, compared against traditional BTs(https://youtu.be/KX_oT3IDTf4). Experimental results demonstrate that the IIBT framework reduces BT node complexity by over 70%, while maintaining robust, interpretable, and adaptive cooperative behavior under environmental uncertainty.
>
---
#### [new 010] Preliminary Analysis and Simulation of a Compact Variable Stiffness Wrist
- **分类: cs.RO**

- **简介: 该论文研究紧凑型变刚度腕关节设计，解决传统变刚度驱动器体积大、重量重的问题。提出一种四电机驱动的3自由度并联腕，实现轻量化，并建立理论模型与控制策略，通过仿真验证其高精度与柔顺性能。**

- **链接: [https://arxiv.org/pdf/2512.04973v1](https://arxiv.org/pdf/2512.04973v1)**

> **作者:** Giuseppe Milazzo; Manuel G. Catalano; Antonio Bicchi; Giorgio Grioli
>
> **备注:** This article has been accepted for publication in Springer Proceedings in Advanced Robotics, vol 31. Springer, Cham. This is the author's version, which has not been fully edited, and the content may change prior to final publication. Citation information: DOI https://doi.org/10.1007/978-3-031-64057-5_9
>
> **摘要:** Variable Stiffness Actuators prove invaluable for robotics applications in unstructured environments, fostering safe interactions and enhancing task adaptability. Nevertheless, their mechanical design inevitably results in larger and heavier structures compared to classical rigid actuators. This paper introduces a novel 3 Degrees of Freedom (DoFs) parallel wrist that achieves variable stiffness through redundant elastic actuation. Leveraging its parallel architecture, the device employs only four motors, rendering it compact and lightweight. This characteristic makes it particularly well-suited for applications in prosthetics or humanoid robotics. The manuscript delves into the theoretical model of the device and proposes a sophisticated control strategy for independent regulation of joint position and stiffness. Furthermore, it validates the proposed controller through simulation, utilizing a comprehensive analysis of the system dynamics. The reported results affirm the ability of the device to achieve high accuracy and disturbance rejection in rigid configurations while minimizing interaction forces with its compliant behavior.
>
---
#### [new 011] Using Machine Learning to Take Stay-or-Go Decisions in Data-driven Drone Missions
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究数据驱动无人机任务中的“留-走”决策问题，即无人机需实时判断是否就地处理数据。为减少无效等待或返航，提出基于分支预测和强化学习的机器学习方法，显著优化任务时间，性能接近理想情况。**

- **链接: [https://arxiv.org/pdf/2512.04773v1](https://arxiv.org/pdf/2512.04773v1)**

> **作者:** Giorgos Polychronis; Foivos Pournaropoulos; Christos D. Antonopoulos; Spyros Lalis
>
> **备注:** 19 pages, 3 figures, to appear in the proceedings of MobiQuitous 2025
>
> **摘要:** Drones are becoming indispensable in many application domains. In data-driven missions, besides sensing, the drone must process the collected data at runtime to decide whether additional action must be taken on the spot, before moving to the next point of interest. If processing does not reveal an event or situation that requires such an action, the drone has waited in vain instead of moving to the next point. If, however, the drone starts moving to the next point and it turns out that a follow-up action is needed at the previous point, it must spend time to fly-back. To take this decision, we propose different machine-learning methods based on branch prediction and reinforcement learning. We evaluate these methods for a wide range of scenarios where the probability of event occurrence changes with time. Our results show that the proposed methods consistently outperform the regression-based method proposed in the literature and can significantly improve the worst-case mission time by up to 4.1x. Also, the achieved median mission time is very close, merely up to 2.7% higher, to that of a method with perfect knowledge of the current underlying event probability at each point of interest.
>
---
#### [new 012] STARE-VLA: Progressive Stage-Aware Reinforcement for Fine-Tuning Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文研究视觉-语言-动作（VLA）模型的细调，解决长视野动作轨迹中信用分配粗略、训练不稳定的问题。提出阶段感知强化（STARE）模块，实现分阶段优化，并设计IPI串行细调流程，显著提升机器人操作成功率。**

- **链接: [https://arxiv.org/pdf/2512.05107v1](https://arxiv.org/pdf/2512.05107v1)**

> **作者:** Feng Xu; Guangyao Zhai; Xin Kong; Tingzhong Fu; Daniel F. N. Gordon; Xueli An; Benjamin Busam
>
> **摘要:** Recent advances in Vision-Language-Action (VLA) models, powered by large language models and reinforcement learning-based fine-tuning, have shown remarkable progress in robotic manipulation. Existing methods often treat long-horizon actions as linguistic sequences and apply trajectory-level optimization methods such as Trajectory-wise Preference Optimization (TPO) or Proximal Policy Optimization (PPO), leading to coarse credit assignment and unstable training. However, unlike language, where a unified semantic meaning is preserved despite flexible sentence order, action trajectories progress through causally chained stages with different learning difficulties. This motivates progressive stage optimization. Thereby, we present Stage-Aware Reinforcement (STARE), a module that decomposes a long-horizon action trajectory into semantically meaningful stages and provides dense, interpretable, and stage-aligned reinforcement signals. Integrating STARE into TPO and PPO, we yield Stage-Aware TPO (STA-TPO) and Stage-Aware PPO (STA-PPO) for offline stage-wise preference and online intra-stage interaction, respectively. Further building on supervised fine-tuning as initialization, we propose the Imitation -> Preference -> Interaction (IPI), a serial fine-tuning pipeline for improving action accuracy in VLA models. Experiments on SimplerEnv and ManiSkill3 demonstrate substantial gains, achieving state-of-the-art success rates of 98.0 percent on SimplerEnv and 96.4 percent on ManiSkill3 tasks.
>
---
#### [new 013] MOVE: A Simple Motion-Based Data Collection Paradigm for Spatial Generalization in Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作中的模仿学习任务，旨在解决因数据稀缺导致的空间泛化能力差的问题。作者提出MOVE方法，通过在数据采集时引入物体运动，增强空间多样性，提升数据效率和模型泛化性能。**

- **链接: [https://arxiv.org/pdf/2512.04813v1](https://arxiv.org/pdf/2512.04813v1)**

> **作者:** Huanqian Wang; Chi Bene Chen; Yang Yue; Danhua Tao; Tong Guo; Shaoxuan Xie; Denghang Huang; Shiji Song; Guocai Yao; Gao Huang
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** Imitation learning method has shown immense promise for robotic manipulation, yet its practical deployment is fundamentally constrained by the data scarcity. Despite prior work on collecting large-scale datasets, there still remains a significant gap to robust spatial generalization. We identify a key limitation: individual trajectories, regardless of their length, are typically collected from a \emph{single, static spatial configuration} of the environment. This includes fixed object and target spatial positions as well as unchanging camera viewpoints, which significantly restricts the diversity of spatial information available for learning. To address this critical bottleneck in data efficiency, we propose \textbf{MOtion-Based Variability Enhancement} (\emph{MOVE}), a simple yet effective data collection paradigm that enables the acquisition of richer spatial information from dynamic demonstrations. Our core contribution is an augmentation strategy that injects motion into any movable objects within the environment for each demonstration. This process implicitly generates a dense and diverse set of spatial configurations within a single trajectory. We conduct extensive experiments in both simulation and real-world environments to validate our approach. For example, in simulation tasks requiring strong spatial generalization, \emph{MOVE} achieves an average success rate of 39.1\%, a 76.1\% relative improvement over the static data collection paradigm (22.2\%), and yields up to 2--5$\times$ gains in data efficiency on certain tasks. Our code is available at https://github.com/lucywang720/MOVE.
>
---
#### [new 014] Sliding Mode Control and Subspace Stabilization Methodology for the Orbital Stabilization of Periodic Trajectories
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究欠驱动机械系统的周期轨迹轨道镇定问题。提出结合滑模控制与子空间稳定的方法，通过部分反馈线性化和横向线性化设计稳定子空间，实现鲁棒控制，避免求解复杂周期LQR问题，并在蝴蝶机器人上验证。**

- **链接: [https://arxiv.org/pdf/2512.04249v1](https://arxiv.org/pdf/2512.04249v1)**

> **作者:** Maksim Surov; Leonid Freidovich
>
> **摘要:** This paper presents a combined sliding-mode control and subspace stabilization methodology for orbital stabilization of periodic trajectories in underactuated mechanical systems with one degree of underactuation. The approach starts with partial feedback linearization and stabilization. Then, transverse linearization along the reference orbit is computed, resulting in a periodic linear time-varying system with a stable subspace. Sliding-mode control drives trajectories toward this subspace. The proposed design avoids solving computationally intensive periodic LQR problems and improves robustness to matched disturbances. The methodology is validated through experiments on the Butterfly robot.
>
---
#### [new 015] Driving Beyond Privilege: Distilling Dense-Reward Knowledge into Sparse-Reward Policies
- **分类: cs.RO**

- **简介: 该论文研究自动驾驶中稀疏奖励策略学习，解决密集特权奖励导致策略偏离部署目标的问题。提出两阶段蒸馏框架：教师用密集奖励训练，学生仅通过其潜动力学蒸馏知识，专精于稀疏任务奖励，提升泛化性与安全性。**

- **链接: [https://arxiv.org/pdf/2512.04279v1](https://arxiv.org/pdf/2512.04279v1)**

> **作者:** Feeza Khan Khanzada; Jaerock Kwon
>
> **摘要:** We study how to exploit dense simulator-defined rewards in vision-based autonomous driving without inheriting their misalignment with deployment metrics. In realistic simulators such as CARLA, privileged state (e.g., lane geometry, infractions, time-to-collision) can be converted into dense rewards that stabilize and accelerate model-based reinforcement learning, but policies trained directly on these signals often overfit and fail to generalize when evaluated on sparse objectives such as route completion and collision-free overtaking. We propose reward-privileged world model distillation, a two-stage framework in which a teacher DreamerV3-style agent is first trained with a dense privileged reward, and only its latent dynamics are distilled into a student trained solely on sparse task rewards. Teacher and student share the same observation space (semantic bird's-eye-view images); privileged information enters only through the teacher's reward, and the student does not imitate the teacher's actions or value estimates. Instead, the student's world model is regularized to match the teacher's latent dynamics while its policy is learned from scratch on sparse success/failure signals. In CARLA lane-following and overtaking benchmarks, sparse-reward students outperform both dense-reward teachers and sparse-from-scratch baselines. On unseen lane-following routes, reward-privileged distillation improves success by about 23 percent relative to the dense teacher while maintaining comparable or better safety. On overtaking, students retain near-perfect performance on training routes and achieve up to a 27x improvement in success on unseen routes, with improved lane keeping. These results show that dense rewards can be leveraged to learn richer dynamics models while keeping the deployed policy optimized strictly for sparse, deployment-aligned objectives.
>
---
#### [new 016] Contact-Implicit Modeling and Simulation of a Snake Robot on Compliant and Granular Terrain
- **分类: cs.RO**

- **简介: 该论文研究蛇形机器人在不同地形上的运动建模与仿真，旨在解决复杂 terrain下运动预测不准的问题。提出融合接触隐式模型与多体动力学的方法，结合连续介质与离散颗粒模拟，实现高效精准的跨地形运动分析。**

- **链接: [https://arxiv.org/pdf/2512.05008v1](https://arxiv.org/pdf/2512.05008v1)**

> **作者:** Haroon Hublikar
>
> **摘要:** This thesis presents a unified modeling and simulation framework for analyzing sidewinding and tumbling locomotion of the COBRA snake robot across rigid, compliant, and granular terrains. A contact-implicit formulation is used to model distributed frictional interactions during sidewinding, and validated through MATLAB Simscape simulations and physical experiments on rigid ground and loose sand. To capture terrain deformation effects, Project Chrono's Soil Contact Model (SCM) is integrated with the articulated multibody dynamics, enabling prediction of slip, sinkage, and load redistribution that reduce stride efficiency on deformable substrates. For high-energy rolling locomotion on steep slopes, the Chrono DEM Engine is used to simulate particle-resolved granular interactions, revealing soil failure, intermittent lift-off, and energy dissipation mechanisms not captured by rigid models. Together, these methods span real-time control-oriented simulation and high-fidelity granular physics. Results demonstrate that rigid-ground models provide accurate short-horizon motion prediction, while continuum and particle-based terrain modeling becomes necessary for reliable mobility analysis in soft and highly dynamic environments. This work establishes a hierarchical simulation pipeline that advances robust, terrain-aware locomotion for robots operating in challenging unstructured settings.
>
---
#### [new 017] On Disturbance-Aware Minimum-Time Trajectory Planning: Evidence from Tests on a Dynamic Driving Simulator
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究干扰感知的最快轨迹规划，旨在平衡圈速与驾驶 effort。通过动态驾驶模拟器测试三种鲁棒轨迹（TLC、FLC、NOM），验证其在真实驾驶中的性能，发现FLC在小幅增加圈时下显著降低转向 effort，优于无引导驾驶，表明干扰感知规划可提升驾驶效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2512.04917v1](https://arxiv.org/pdf/2512.04917v1)**

> **作者:** Matteo Masoni; Vincenzo Palermo; Marco Gabiccini; Martino Gulisano; Giorgio Previati; Massimiliano Gobbi; Francesco Comolli; Gianpiero Mastinu; Massimo Guiggiani
>
> **备注:** 18 pages, 11 figures, 5 tables
>
> **摘要:** This work investigates how disturbance-aware, robustness-embedded reference trajectories translate into driving performance when executed by professional drivers in a dynamic simulator. Three planned reference trajectories are compared against a free-driving baseline (NOREF) to assess trade-offs between lap time (LT) and steering effort (SE): NOM, the nominal time-optimal trajectory; TLC, a track-limit-robust trajectory obtained by tightening margins to the track edges; and FLC, a friction-limit-robust trajectory obtained by tightening against axle and tire saturation. All trajectories share the same minimum lap-time objective with a small steering-smoothness regularizer and are evaluated by two professional drivers using a high-performance car on a virtual track. The trajectories derive from a disturbance-aware minimum-lap-time framework recently proposed by the authors, where worst-case disturbance growth is propagated over a finite horizon and used to tighten tire-friction and track-limit constraints, preserving performance while providing probabilistic safety margins. LT and SE are used as performance indicators, while RMS lateral deviation, speed error, and drift angle characterize driving style. Results show a Pareto-like LT-SE trade-off: NOM yields the shortest LT but highest SE; TLC minimizes SE at the cost of longer LT; FLC lies near the efficient frontier, substantially reducing SE relative to NOM with only a small LT increase. Removing trajectory guidance (NOREF) increases both LT and SE, confirming that reference trajectories improve pace and control efficiency. Overall, the findings highlight reference-based and disturbance-aware planning, especially FLC, as effective tools for training and for achieving fast yet stable trajectories.
>
---
#### [new 018] Development of a 15-Degree-of-Freedom Bionic Hand with Cable-Driven Transmission and Distributed Actuation
- **分类: cs.RO**

- **简介: 该论文研究仿生灵巧手设计，旨在减少驱动器数量的同时保持人手自由度与尺寸。提出一种15自由度缆索驱动、分布式驱动的仿生手，集成15个电机（5个前臂，10个掌部），实现轻量化、高灵活性与强抓握能力。**

- **链接: [https://arxiv.org/pdf/2512.04399v1](https://arxiv.org/pdf/2512.04399v1)**

> **作者:** Haoqi Han; Yi Yang; Yifei Yu; Yixuan Zhou; Xiaohan Zhu; Hesheng Wang
>
> **摘要:** In robotic hand research, minimizing the number of actuators while maintaining human-hand-consistent dimensions and degrees of freedom constitutes a fundamental challenge. Drawing bio-inspiration from human hand kinematic configurations and muscle distribution strategies, this work proposes a novel 15-DoF dexterous robotic hand, with detailed analysis of its mechanical architecture, electrical system, and control system. The bionic hand employs a new tendon-driven mechanism, significantly reducing the number of motors required by traditional tendon-driven systems while enhancing motion performance and simplifying the mechanical structure. This design integrates five motors in the forearm to provide strong gripping force, while ten small motors are installed in the palm to support fine manipulation tasks. Additionally, a corresponding joint sensing and motor driving electrical system was developed to ensure efficient control and feedback. The entire system weighs only 1.4kg, combining lightweight and high-performance features. Through experiments, the bionic hand exhibited exceptional dexterity and robust grasping capabilities, demonstrating significant potential for robotic manipulation tasks.
>
---
#### [new 019] Embodied Co-Design for Rapidly Evolving Agents: Taxonomy, Frontiers, and Challenges
- **分类: cs.RO; cs.AI; cs.ET; eess.SY**

- **简介: 该论文综述了具身协同设计（ECD）的研究进展，旨在通过脑-体-环境联合优化提升智能体性能。它提出分层分类体系，整合四类ECD框架，并系统梳理了方法、基准与应用，指出挑战与未来方向。**

- **链接: [https://arxiv.org/pdf/2512.04770v1](https://arxiv.org/pdf/2512.04770v1)**

> **作者:** Yuxing Wang; Zhiyu Chen; Tiantian Zhang; Qiyue Yin; Yongzhe Chang; Zhiheng Li; Liang Wang; Xueqian Wang
>
> **摘要:** Brain-body co-evolution enables animals to develop complex behaviors in their environments. Inspired by this biological synergy, embodied co-design (ECD) has emerged as a transformative paradigm for creating intelligent agents-from virtual creatures to physical robots-by jointly optimizing their morphologies and controllers rather than treating control in isolation. This integrated approach facilitates richer environmental interactions and robust task performance. In this survey, we provide a systematic overview of recent advances in ECD. We first formalize the concept of ECD and position it within related fields. We then introduce a hierarchical taxonomy: a lower layer that breaks down agent design into three fundamental components-controlling brain, body morphology, and task environment-and an upper layer that integrates these components into four major ECD frameworks: bi-level, single-level, generative, and open-ended. This taxonomy allows us to synthesize insights from more than one hundred recent studies. We further review notable benchmarks, datasets, and applications in both simulated and real-world scenarios. Finally, we identify significant challenges and offer insights into promising future research directions. A project associated with this survey has been created at https://github.com/Yuxing-Wang-THU/SurveyBrainBody.
>
---
#### [new 020] CRAFT-E: A Neuro-Symbolic Framework for Embodied Affordance Grounding
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究具身场景下的物体功能理解任务，旨在解决助手机器人对“物体可执行动作”的可靠推理问题。提出CRAFT-E框架，结合知识图谱、视觉语言对齐与抓取可行性分析，实现可解释的物体选择，并构建新数据集验证其有效性。**

- **链接: [https://arxiv.org/pdf/2512.04231v1](https://arxiv.org/pdf/2512.04231v1)**

> **作者:** Zhou Chen; Joe Lin; Carson Bulgin; Sathyanarayanan N. Aakur
>
> **备注:** 20 pages. 3 figures, 4 tables. Under Review
>
> **摘要:** Assistive robots operating in unstructured environments must understand not only what objects are, but what they can be used for. This requires grounding language-based action queries to objects that both afford the requested function and can be physically retrieved. Existing approaches often rely on black-box models or fixed affordance labels, limiting transparency, controllability, and reliability for human-facing applications. We introduce CRAFT-E, a modular neuro-symbolic framework that composes a structured verb-property-object knowledge graph with visual-language alignment and energy-based grasp reasoning. The system generates interpretable grounding paths that expose the factors influencing object selection and incorporates grasp feasibility as an integral part of affordance inference. We further construct a benchmark dataset with unified annotations for verb-object compatibility, segmentation, and grasp candidates, and deploy the full pipeline on a physical robot. CRAFT-E achieves competitive performance in static scenes, ImageNet-based functional retrieval, and real-world trials involving 20 verbs and 39 objects. The framework remains robust under perceptual noise and provides transparent, component-level diagnostics. By coupling symbolic reasoning with embodied perception, CRAFT-E offers an interpretable and customizable alternative to end-to-end models for affordance-grounded object selection, supporting trustworthy decision-making in assistive robotic systems.
>
---
#### [new 021] Open-Ended Goal Inference through Actions and Language for Human-Robot Collaboration
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究人机协作中的开放性目标推理任务，旨在解决人类目标模糊、难以预定义的问题。提出BALI方法，融合语言与动作线索进行双向推理，在烹饪场景中实现更稳定的目标预测，并减少误判。**

- **链接: [https://arxiv.org/pdf/2512.04453v1](https://arxiv.org/pdf/2512.04453v1)**

> **作者:** Debasmita Ghose; Oz Gitelson; Marynel Vazquez; Brian Scassellati
>
> **备注:** Accepted to ACM/IEEE International Conference on Human-Robot Interaction, 2026 (HRI 2026), 10 pages, 4 figures
>
> **摘要:** To collaborate with humans, robots must infer goals that are often ambiguous, difficult to articulate, or not drawn from a fixed set. Prior approaches restrict inference to a predefined goal set, rely only on observed actions, or depend exclusively on explicit instructions, making them brittle in real-world interactions. We present BALI (Bidirectional Action-Language Inference) for goal prediction, a method that integrates natural language preferences with observed human actions in a receding-horizon planning tree. BALI combines language and action cues from the human, asks clarifying questions only when the expected information gain from the answer outweighs the cost of interruption, and selects supportive actions that align with inferred goals. We evaluate the approach in collaborative cooking tasks, where goals may be novel to the robot and unbounded. Compared to baselines, BALI yields more stable goal predictions and significantly fewer mistakes.
>
---
#### [new 022] ResponsibleRobotBench: Benchmarking Responsible Robot Manipulation using Multi-modal Large Language Models
- **分类: cs.RO**

- **简介: 该论文提出 ResponsibleRobotBench，旨在评估机器人在高风险环境中的负责任操作能力。属于机器人操纵与安全决策任务，解决多模态大模型下风险感知、道德决策与物理规划的可靠性问题，构建了含23项多阶段任务的评测基准及综合框架。**

- **链接: [https://arxiv.org/pdf/2512.04308v1](https://arxiv.org/pdf/2512.04308v1)**

> **作者:** Lei Zhang; Ju Dong; Kaixin Bai; Minheng Ni; Zoltan-Csaba Marton; Zhaopeng Chen; Jianwei Zhang
>
> **备注:** https://sites.google.com/view/responsible-robotbench
>
> **摘要:** Recent advances in large multimodal models have enabled new opportunities in embodied AI, particularly in robotic manipulation. These models have shown strong potential in generalization and reasoning, but achieving reliable and responsible robotic behavior in real-world settings remains an open challenge. In high-stakes environments, robotic agents must go beyond basic task execution to perform risk-aware reasoning, moral decision-making, and physically grounded planning. We introduce ResponsibleRobotBench, a systematic benchmark designed to evaluate and accelerate progress in responsible robotic manipulation from simulation to real world. This benchmark consists of 23 multi-stage tasks spanning diverse risk types, including electrical, chemical, and human-related hazards, and varying levels of physical and planning complexity. These tasks require agents to detect and mitigate risks, reason about safety, plan sequences of actions, and engage human assistance when necessary. Our benchmark includes a general-purpose evaluation framework that supports multimodal model-based agents with various action representation modalities. The framework integrates visual perception, context learning, prompt construction, hazard detection, reasoning and planning, and physical execution. It also provides a rich multimodal dataset, supports reproducible experiments, and includes standardized metrics such as success rate, safety rate, and safe success rate. Through extensive experimental setups, ResponsibleRobotBench enables analysis across risk categories, task types, and agent configurations. By emphasizing physical reliability, generalization, and safety in decision-making, this benchmark provides a foundation for advancing the development of trustworthy, real-world responsible dexterous robotic systems. https://sites.google.com/view/responsible-robotbench
>
---
#### [new 023] From Generated Human Videos to Physically Plausible Robot Trajectories
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究如何让机器人零样本模仿生成视频中的人类动作。针对生成视频噪声多、形态失真问题，提出两阶段方法：先将视频转为4D人体表示并适配机器人形态，再用物理感知强化学习策略GenMimic实现动作复现，并构建合成数据集GenMimicBench验证其泛化性与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.05094v1](https://arxiv.org/pdf/2512.05094v1)**

> **作者:** James Ni; Zekai Wang; Wei Lin; Amir Bar; Yann LeCun; Trevor Darrell; Jitendra Malik; Roei Herzig
>
> **备注:** For project website, see https://genmimic.github.io
>
> **摘要:** Video generation models are rapidly improving in their ability to synthesize human actions in novel contexts, holding the potential to serve as high-level planners for contextual robot control. To realize this potential, a key research question remains open: how can a humanoid execute the human actions from generated videos in a zero-shot manner? This challenge arises because generated videos are often noisy and exhibit morphological distortions that make direct imitation difficult compared to real video. To address this, we introduce a two-stage pipeline. First, we lift video pixels into a 4D human representation and then retarget to the humanoid morphology. Second, we propose GenMimic-a physics-aware reinforcement learning policy conditioned on 3D keypoints, and trained with symmetry regularization and keypoint-weighted tracking rewards. As a result, GenMimic can mimic human actions from noisy, generated videos. We curate GenMimicBench, a synthetic human-motion dataset generated using two video generation models across a spectrum of actions and contexts, establishing a benchmark for assessing zero-shot generalization and policy robustness. Extensive experiments demonstrate improvements over strong baselines in simulation and confirm coherent, physically stable motion tracking on a Unitree G1 humanoid robot without fine-tuning. This work offers a promising path to realizing the potential of video generation models as high-level policies for robot control.
>
---
#### [new 024] Vision-Language-Action Models for Selective Robotic Disassembly: A Case Study on Critical Component Extraction from Desktops
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究废旧台式机中关键部件（如RAM、CPU）的机器人选择性拆卸任务，旨在解决因产品差异大导致的自动化拆卸难题。作者通过构建定制数据集，微调视觉-语言-动作（VLA）模型，并结合规则控制器，探索端到端拆卸的可行性。**

- **链接: [https://arxiv.org/pdf/2512.04446v1](https://arxiv.org/pdf/2512.04446v1)**

> **作者:** Chang Liu; Sibo Tian; Sara Behdad; Xiao Liang; Minghui Zheng
>
> **摘要:** Automating disassembly of critical components from end-of-life (EoL) desktops, such as high-value items like RAM modules and CPUs, as well as sensitive parts like hard disk drives, remains challenging due to the inherent variability and uncertainty of these products. Moreover, their disassembly requires sequential, precise, and dexterous operations, further increasing the complexity of automation. Current robotic disassembly processes are typically divided into several stages: perception, sequence planning, task planning, motion planning, and manipulation. Each stage requires explicit modeling, which limits generalization to unfamiliar scenarios. Recent development of vision-language-action (VLA) models has presented an end-to-end approach for general robotic manipulation tasks. Although VLAs have demonstrated promising performance on simple tasks, the feasibility of applying such models to complex disassembly remains largely unexplored. In this paper, we collected a customized dataset for robotic RAM and CPU disassembly and used it to fine-tune two well-established VLA approaches, OpenVLA and OpenVLA-OFT, as a case study. We divided the whole disassembly task into several small steps, and our preliminary experimental results indicate that the fine-tuned VLA models can faithfully complete multiple early steps but struggle with certain critical subtasks, leading to task failure. However, we observed that a simple hybrid strategy that combines VLA with a rule-based controller can successfully perform the entire disassembly operation. These findings highlight the current limitations of VLA models in handling the dexterity and precision required for robotic EoL product disassembly. By offering a detailed analysis of the observed results, this study provides insights that may inform future research to address current challenges and advance end-to-end robotic automated disassembly.
>
---
#### [new 025] Hybrid-Diffusion Models: Combining Open-loop Routines with Visuomotor Diffusion Policies
- **分类: cs.RO**

- **简介: 该论文针对机器人操作中模仿学习精度与速度不足的问题，提出Hybrid-Diffusion模型，融合开环例程与视觉运动扩散策略。通过引入可由操作者在示范中触发的Teleoperation Augmentation Primitives（TAPs），使模型在推理时自主调用相应例程，提升复杂任务表现。实验验证于药瓶抽吸、液体转移和容器旋拧等高难度真实任务。**

- **链接: [https://arxiv.org/pdf/2512.04960v1](https://arxiv.org/pdf/2512.04960v1)**

> **作者:** Jonne Van Haastregt; Bastian Orthmann; Michael C. Welle; Yuchong Zhang; Danica Kragic
>
> **摘要:** Despite the fact that visuomotor-based policies obtained via imitation learning demonstrate good performances in complex manipulation tasks, they usually struggle to achieve the same accuracy and speed as traditional control based methods. In this work, we introduce Hybrid-Diffusion models that combine open-loop routines with visuomotor diffusion policies. We develop Teleoperation Augmentation Primitives (TAPs) that allow the operator to perform predefined routines, such as locking specific axes, moving to perching waypoints, or triggering task-specific routines seamlessly during demonstrations. Our Hybrid-Diffusion method learns to trigger such TAPs during inference. We validate the method on challenging real-world tasks: Vial Aspiration, Open-Container Liquid Transfer, and container unscrewing. All experimental videos are available on the project's website: https://hybriddiffusion.github.io/
>
---
#### [new 026] When Robots Should Say "I Don't Know": Benchmarking Abstention in Embodied Question Answering
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文研究具身问答（EQA）中的“ abstention”任务，即机器人在信息不足时应选择不回答。针对现有基准强制回答的问题，作者构建AbstainEQA数据集，包含需 abstention 的五类模糊问题，评估模型拒答能力，发现当前模型表现远低于人类，凸显拒答对可靠交互的重要性。**

- **链接: [https://arxiv.org/pdf/2512.04597v1](https://arxiv.org/pdf/2512.04597v1)**

> **作者:** Tao Wu; Chuhao Zhou; Guangyu Zhao; Haozhi Cao; Yewen Pu; Jianfei Yang
>
> **摘要:** Embodied Question Answering (EQA) requires an agent to interpret language, perceive its environment, and navigate within 3D scenes to produce responses. Existing EQA benchmarks assume that every question must be answered, but embodied agents should know when they do not have sufficient information to answer. In this work, we focus on a minimal requirement for EQA agents, abstention: knowing when to withhold an answer. From an initial study of 500 human queries, we find that 32.4% contain missing or underspecified context. Drawing on this initial study and cognitive theories of human communication errors, we derive five representative categories requiring abstention: actionability limitation, referential underspecification, preference dependence, information unavailability, and false presupposition. We augment OpenEQA by having annotators transform well-posed questions into ambiguous variants outlined by these categories. The resulting dataset, AbstainEQA, comprises 1,636 annotated abstention cases paired with 1,636 original OpenEQA instances for balanced evaluation. Evaluating on AbstainEQA, we find that even the best frontier model only attains 42.79% abstention recall, while humans achieve 91.17%. We also find that scaling, prompting, and reasoning only yield marginal gains, and that fine-tuned models overfit to textual cues. Together, these results position abstention as a fundamental prerequisite for reliable interaction in embodied settings and as a necessary basis for effective clarification.
>
---
#### [new 027] SIMA 2: A Generalist Embodied Agent for Virtual Worlds
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出SIMA 2，一种基于Gemini的通用具身智能体，旨在解决虚拟世界中复杂任务理解与自主学习问题。它能理解图文指令、对话交互、泛化至新环境，并实现自我改进，推动虚拟及物理世界智能体的发展。**

- **链接: [https://arxiv.org/pdf/2512.04797v1](https://arxiv.org/pdf/2512.04797v1)**

> **作者:** SIMA team; Adrian Bolton; Alexander Lerchner; Alexandra Cordell; Alexandre Moufarek; Andrew Bolt; Andrew Lampinen; Anna Mitenkova; Arne Olav Hallingstad; Bojan Vujatovic; Bonnie Li; Cong Lu; Daan Wierstra; Daniel P. Sawyer; Daniel Slater; David Reichert; Davide Vercelli; Demis Hassabis; Drew A. Hudson; Duncan Williams; Ed Hirst; Fabio Pardo; Felix Hill; Frederic Besse; Hannah Openshaw; Harris Chan; Hubert Soyer; Jane X. Wang; Jeff Clune; John Agapiou; John Reid; Joseph Marino; Junkyung Kim; Karol Gregor; Kaustubh Sridhar; Kay McKinney; Laura Kampis; Lei M. Zhang; Loic Matthey; Luyu Wang; Maria Abi Raad; Maria Loks-Thompson; Martin Engelcke; Matija Kecman; Matthew Jackson; Maxime Gazeau; Ollie Purkiss; Oscar Knagg; Peter Stys; Piermaria Mendolicchio; Raia Hadsell; Rosemary Ke; Ryan Faulkner; Sarah Chakera; Satinder Singh Baveja; Shane Legg; Sheleem Kashem; Tayfun Terzi; Thomas Keck; Tim Harley; Tim Scholtes; Tyson Roberts; Volodymyr Mnih; Yulan Liu; Zhengdong Wang; Zoubin Ghahramani
>
> **摘要:** We introduce SIMA 2, a generalist embodied agent that understands and acts in a wide variety of 3D virtual worlds. Built upon a Gemini foundation model, SIMA 2 represents a significant step toward active, goal-directed interaction within an embodied environment. Unlike prior work (e.g., SIMA 1) limited to simple language commands, SIMA 2 acts as an interactive partner, capable of reasoning about high-level goals, conversing with the user, and handling complex instructions given through language and images. Across a diverse portfolio of games, SIMA 2 substantially closes the gap with human performance and demonstrates robust generalization to previously unseen environments, all while retaining the base model's core reasoning capabilities. Furthermore, we demonstrate a capacity for open-ended self-improvement: by leveraging Gemini to generate tasks and provide rewards, SIMA 2 can autonomously learn new skills from scratch in a new environment. This work validates a path toward creating versatile and continuously learning agents for both virtual and, eventually, physical worlds.
>
---
#### [new 028] Object Reconstruction under Occlusion with Generative Priors and Contact-induced Constraints
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究被遮挡物体的几何重建，属机器人操作中的感知任务。针对视觉观测不完整问题，结合生成模型的形状先验与接触产生的边界约束，提出接触引导的3D生成方法，提升重建精度。**

- **链接: [https://arxiv.org/pdf/2512.05079v1](https://arxiv.org/pdf/2512.05079v1)**

> **作者:** Minghan Zhu; Zhiyi Wang; Qihang Sun; Maani Ghaffari; Michael Posa
>
> **备注:** Project page: https://contactgen3d.github.io/
>
> **摘要:** Object geometry is key information for robot manipulation. Yet, object reconstruction is a challenging task because cameras only capture partial observations of objects, especially when occlusion occurs. In this paper, we leverage two extra sources of information to reduce the ambiguity of vision signals. First, generative models learn priors of the shapes of commonly seen objects, allowing us to make reasonable guesses of the unseen part of geometry. Second, contact information, which can be obtained from videos and physical interactions, provides sparse constraints on the boundary of the geometry. We combine the two sources of information through contact-guided 3D generation. The guidance formulation is inspired by drag-based editing in generative models. Experiments on synthetic and real-world data show that our approach improves the reconstruction compared to pure 3D generation and contact-based optimization.
>
---
#### [new 029] Gauss-Newton accelerated MPPI Control
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于最优控制任务，旨在解决MPPI在高维场景下性能下降的问题。作者提出Gauss-Newton加速MPPI方法，结合雅可比重构与二阶广义高斯-牛顿法，提升算法的可扩展性与计算效率，同时保持原有优势。**

- **链接: [https://arxiv.org/pdf/2512.04579v1](https://arxiv.org/pdf/2512.04579v1)**

> **作者:** Hannes Homburger; Katrin Baumgärtner; Moritz Diehl; Johannes Reuter
>
> **备注:** 6 pages, 3 figures, submitted to the IFAC World Congress 2026
>
> **摘要:** Model Predictive Path Integral (MPPI) control is a sampling-based optimization method that has recently attracted attention, particularly in the robotics and reinforcement learning communities. MPPI has been widely applied as a GPU-accelerated random search method to deterministic direct single-shooting optimal control problems arising in model predictive control (MPC) formulations. MPPI offers several key advantages, including flexibility, robustness, ease of implementation, and inherent parallelizability. However, its performance can deteriorate in high-dimensional settings since the optimal control problem is solved via Monte Carlo sampling. To address this limitation, this paper proposes an enhanced MPPI method that incorporates a Jacobian reconstruction technique and the second-order Generalized Gauss-Newton method. This novel approach is called \textit{Gauss-Newton accelerated MPPI}. The numerical results show that the Gauss-Newton accelerated MPPI approach substantially improves MPPI scalability and computational efficiency while preserving the key benefits of the classical MPPI framework, making it a promising approach even for high-dimensional problems.
>
---
#### [new 030] FASTer: Toward Efficient Autoregressive Vision Language Action Modeling via neural Action Tokenization
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究机器人操作中的视觉-语言-动作（VLA）建模，旨在解决动作标记化中重建精度与推理效率的权衡问题。作者提出FASTer框架，通过可学习的动作分词器FASTerVQ和基于其的快速自回归策略FASTerVLA，实现高效、高性能的跨任务泛化。**

- **链接: [https://arxiv.org/pdf/2512.04952v1](https://arxiv.org/pdf/2512.04952v1)**

> **作者:** Yicheng Liu; Shiduo Zhang; Zibin Dong; Baijun Ye; Tianyuan Yuan; Xiaopeng Yu; Linqi Yin; Chenhao Lu; Junhao Shi; Luca Jiang-Tao Yu; Liangtao Zheng; Tao Jiang; Jingjing Gong; Xipeng Qiu; Hang Zhao
>
> **摘要:** Autoregressive vision-language-action (VLA) models have recently demonstrated strong capabilities in robotic manipulation. However, their core process of action tokenization often involves a trade-off between reconstruction fidelity and inference efficiency. We introduce FASTer, a unified framework for efficient and generalizable robot learning that integrates a learnable tokenizer with an autoregressive policy built upon it. FASTerVQ encodes action chunks as single-channel images, capturing global spatio-temporal dependencies while maintaining a high compression ratio. FASTerVLA builds on this tokenizer with block-wise autoregressive decoding and a lightweight action expert, achieving both faster inference and higher task performance. Extensive experiments across simulated and real-world benchmarks show that FASTerVQ delivers superior reconstruction quality, high token utilization, and strong cross-task and cross-embodiment generalization, while FASTerVLA further improves overall capability, surpassing previous state-of-the-art VLA models in both inference speed and task performance.
>
---
#### [new 031] NeuralRemaster: Phase-Preserving Diffusion for Structure-Aligned Generation
- **分类: cs.CV; cs.GR; cs.LG; cs.RO**

- **简介: 该论文属于图像/视频生成任务，旨在解决标准扩散模型破坏空间结构的问题。作者提出相位保持扩散（φ-PD），保留输入相位以实现结构对齐生成，并引入频率选择性噪声控制结构刚性，无需修改模型结构，提升 sim-to-real 等任务性能。**

- **链接: [https://arxiv.org/pdf/2512.05106v1](https://arxiv.org/pdf/2512.05106v1)**

> **作者:** Yu Zeng; Charles Ochoa; Mingyuan Zhou; Vishal M. Patel; Vitor Guizilini; Rowan McAllister
>
> **摘要:** Standard diffusion corrupts data using Gaussian noise whose Fourier coefficients have random magnitudes and random phases. While effective for unconditional or text-to-image generation, corrupting phase components destroys spatial structure, making it ill-suited for tasks requiring geometric consistency, such as re-rendering, simulation enhancement, and image-to-image translation. We introduce Phase-Preserving Diffusion φ-PD, a model-agnostic reformulation of the diffusion process that preserves input phase while randomizing magnitude, enabling structure-aligned generation without architectural changes or additional parameters. We further propose Frequency-Selective Structured (FSS) noise, which provides continuous control over structural rigidity via a single frequency-cutoff parameter. φ-PD adds no inference-time cost and is compatible with any diffusion model for images or videos. Across photorealistic and stylized re-rendering, as well as sim-to-real enhancement for driving planners, φ-PD produces controllable, spatially aligned results. When applied to the CARLA simulator, φ-PD improves CARLA-to-Waymo planner performance by 50\%. The method is complementary to existing conditioning approaches and broadly applicable to image-to-image and video-to-video generation. Videos, additional examples, and code are available on our \href{https://yuzeng-at-tri.github.io/ppd-page/}{project page}.
>
---
#### [new 032] MARL Warehouse Robots
- **分类: cs.AI; cs.RO**

- **简介: 该论文研究多智能体强化学习在协作仓储机器人中的应用，解决任务分配与路径协同问题。比较QMIX与IPPO算法，在RWARE和Unity 3D环境中实验，发现QMIX性能更优但需精细调参，成功实现小规模部署。**

- **链接: [https://arxiv.org/pdf/2512.04463v1](https://arxiv.org/pdf/2512.04463v1)**

> **作者:** Price Allman; Lian Thang; Dre Simmons; Salmon Riaz
>
> **备注:** 6 pages, 4 tables. Project documentation: https://pallman14.github.io/MARL-QMIX-Warehouse-Robots/
>
> **摘要:** We present a comparative study of multi-agent reinforcement learning (MARL) algorithms for cooperative warehouse robotics. We evaluate QMIX and IPPO on the Robotic Warehouse (RWARE) environment and a custom Unity 3D simulation. Our experiments reveal that QMIX's value decomposition significantly outperforms independent learning approaches (achieving 3.25 mean return vs. 0.38 for advanced IPPO), but requires extensive hyperparameter tuning -- particularly extended epsilon annealing (5M+ steps) for sparse reward discovery. We demonstrate successful deployment in Unity ML-Agents, achieving consistent package delivery after 1M training steps. While MARL shows promise for small-scale deployments (2-4 robots), significant scaling challenges remain. Code and analyses: https://pallman14.github.io/MARL-QMIX-Warehouse-Robots/
>
---
## 更新

#### [replaced 001] Efficient Preference-Based Reinforcement Learning: Randomized Exploration Meets Experimental Design
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **简介: 该论文研究基于人类偏好的强化学习，旨在通过轨迹比较学习奖励函数。提出一种结合随机探索与最优实验设计的元算法，降低查询复杂度，支持批量并行查询，并在理论与实验上验证了有效性。**

- **链接: [https://arxiv.org/pdf/2506.09508v2](https://arxiv.org/pdf/2506.09508v2)**

> **作者:** Andreas Schlaginhaufen; Reda Ouhamma; Maryam Kamgarpour
>
> **摘要:** We study reinforcement learning from human feedback in general Markov decision processes, where agents learn from trajectory-level preference comparisons. A central challenge in this setting is to design algorithms that select informative preference queries to identify the underlying reward while ensuring theoretical guarantees. We propose a meta-algorithm based on randomized exploration, which avoids the computational challenges associated with optimistic approaches and remains tractable. We establish both regret and last-iterate guarantees under mild reinforcement learning oracle assumptions. To improve query complexity, we introduce and analyze an improved algorithm that collects batches of trajectory pairs and applies optimal experimental design to select informative comparison queries. The batch structure also enables parallelization of preference queries, which is relevant in practical deployment as feedback can be gathered concurrently. Empirical evaluation confirms that the proposed method is competitive with reward-based reinforcement learning while requiring a small number of preference queries.
>
---
#### [replaced 002] Scalable Policy Evaluation with Video World Models
- **分类: cs.RO**

- **简介: 该论文属于机器人策略评估任务，旨在解决真实世界测试成本高和仿真环境构建困难的问题。作者提出利用动作条件视频生成模型构建可扩展的世界模型，通过预训练视频模型实现无需大量真实交互的策略评估。**

- **链接: [https://arxiv.org/pdf/2511.11520v3](https://arxiv.org/pdf/2511.11520v3)**

> **作者:** Wei-Cheng Tseng; Jinwei Gu; Qinsheng Zhang; Hanzi Mao; Ming-Yu Liu; Florian Shkurti; Lin Yen-Chen
>
> **摘要:** Training generalist policies for robotic manipulation has shown great promise, as they enable language-conditioned, multi-task behaviors across diverse scenarios. However, evaluating these policies remains difficult because real-world testing is expensive, time-consuming, and labor-intensive. It also requires frequent environment resets and carries safety risks when deploying unproven policies on physical robots. Manually creating and populating simulation environments with assets for robotic manipulation has not addressed these issues, primarily due to the significant engineering effort required and the substantial sim-to-real gap, both in terms of physics and rendering. In this paper, we explore the use of action-conditional video generation models as a scalable way to learn world models for policy evaluation. We demonstrate how to incorporate action conditioning into existing pre-trained video generation models. This allows leveraging internet-scale in-the-wild online videos during the pre-training stage and alleviates the need for a large dataset of paired video-action data, which is expensive to collect for robotic manipulation. Our paper examines the effect of dataset diversity, pre-trained weights, and common failure cases for the proposed evaluation pipeline. Our experiments demonstrate that across various metrics, including policy ranking and the correlation between actual policy values and predicted policy values, these models offer a promising approach for evaluating policies without requiring real-world interactions.
>
---
#### [replaced 003] HAFO: A Force-Adaptive Control Framework for Humanoid Robots in Intense Interaction Environments
- **分类: cs.RO**

- **简介: 该论文研究人形机器人在强交互环境中的力控问题，提出HAFO框架，通过双智能体强化学习与弹簧阻尼扰动建模，实现鲁棒运动与精确操作的协同训练，提升抗干扰与负载能力。**

- **链接: [https://arxiv.org/pdf/2511.20275v3](https://arxiv.org/pdf/2511.20275v3)**

> **作者:** Chenhui Dong; Haozhe Xu; Wenhao Feng; Zhipeng Wang; Yanmin Zhou; Yifei Zhao; Bin He
>
> **摘要:** Reinforcement learning (RL) controllers have made impressive progress in humanoid locomotion and light-weight object manipulation. However, achieving robust and precise motion control with intense force interaction remains a significant challenge. To address these limitations, this paper proposes HAFO, a dual-agent reinforcement learning framework that concurrently optimizes both a robust locomotion strategy and a precise upper-body manipulation strategy via coupled training in environments with external disturbances. The external pulling disturbances are explicitly modeled using a spring-damper system, allowing for fine-grained force control through manipulation of the virtual spring. In this process, the reinforcement learning policy autonomously generates a disturbance-rejection response by utilizing environmental feedback. Furthermore, HAFO employs an asymmetric Actor-Critic framework in which the Critic network's access to privileged external forces guides the actor network to acquire generalizable force adaptation for resisting external disturbances. The experimental results demonstrate that HAFO achieves whole-body control for humanoid robots across diverse force-interaction environments, delivering outstanding performance in load-bearing tasks and maintaining stable operation even under rope suspension state.
>
---
#### [replaced 004] A Fast and Model Based Approach for Evaluating Task-Competence of Antagonistic Continuum Arms
- **分类: cs.RO; cs.CE**

- **简介: 该论文针对软体机器人设计难题，提出一种快速、基于模型的方法，用于评估对抗性连续臂在特定任务中的能力。解决了现有模型依赖参数拟合、难以指导设计的问题，实现了任务特异性分析与可视化比较。**

- **链接: [https://arxiv.org/pdf/2411.00241v4](https://arxiv.org/pdf/2411.00241v4)**

> **作者:** Bill Fan; Jacob Roulier; Gina Olson
>
> **备注:** Published in the 8th IEEE-RAS International Conference on Soft Robotics (RoboSoft 2025). See https://github.com/wfan19/antagonistic-task-competency for code, proofs, and supplementary information. Please note the officially published version of the paper in IEEE contains an error in Equation 7. That has been corrected here, so this is the final version of the paper. Apologies for the confusion!
>
> **摘要:** Soft robot arms have made significant progress towards completing human-scale tasks, but designing arms for tasks with specific load and workspace requirements remains difficult. A key challenge is the lack of model-based design tools, forcing advancement to occur through empirical iteration and observation. Existing models are focused on control and rely on parameter fits, which means they cannot provide general conclusions about the mapping between design and performance or the influence of factors outside the fitting data.As a first step toward model-based design tools, we introduce a novel method of analyzing whether a proposed arm design can complete desired tasks. Our method is informative, interpretable, and fast; it provides novel metrics for quantifying a proposed arm design's ability to perform a task, it yields a graphical interpretation of performance through segment forces, and computing it is over 80x faster than optimization based methods.Our formulation focuses on antagonistic, pneumatically-driven soft arms. We demonstrate our approach through example analysis, and also through consideration of antagonistic vs non-antagonistic designs. Our method enables fast, direct and task-specific comparison of these two architectures, and provides a new visualization of the comparative mechanics. While only a first step, the proposed approach will support advancement of model-based design tools, leading to highly capable soft arms.
>
---
#### [replaced 005] Beyond Description: Cognitively Benchmarking Fine-Grained Action for Embodied Agents
- **分类: cs.CV; cs.RO**

- **简介: 该论文聚焦于具身智能体的细粒度动作认知任务，旨在解决现有基准忽略物理交互细节与高阶推理的问题。作者提出CFG-Bench，包含1,368个视频和19,562个多模态问答对，评估四种认知能力，并通过实验证明其数据可提升模型在具身任务中的表现。**

- **链接: [https://arxiv.org/pdf/2511.18685v2](https://arxiv.org/pdf/2511.18685v2)**

> **作者:** Dayong Liu; Chao Xu; Weihong Chen; Suyu Zhang; Juncheng Wang; Jiankang Deng; Baigui Sun; Yang Liu
>
> **摘要:** Multimodal Large Language Models (MLLMs) show promising results as decision-making engines for embodied agents operating in complex, physical environments. However, existing benchmarks often prioritize high-level planning or spatial reasoning, leaving the fine-grained action intelligence required for embodied physical interaction underexplored. To address this gap, we introduce CFG-Bench, a new benchmark designed to systematically evaluate this crucial capability. CFG-Bench consists of 1,368 curated videos paired with 19,562 three-modalities question-answer pairs targeting four cognitive abilities: 1) Physical Interaction, 2) Temporal-Causal Relation, 3) Intentional Understanding, and 4) Evaluative Judgment. Together, these dimensions provide a systematic framework for assessing a model's ability to translate visual observations into actionable knowledge, moving beyond mere surface-level recognition. Our comprehensive evaluation on CFG-Bench reveals that leading MLLMs struggle to produce detailed instructions for physical interactions and exhibit profound limitations in the higher-order reasoning of intention and evaluation. Moreover, supervised fine-tuning (SFT) on our data demonstrates that teaching an MLLMs to articulate fine-grained actions directly translates to significant performance gains on established embodied benchmarks. Our analysis highlights these limitations and offers insights for developing more capable and grounded embodied agents. Project page: \href{https://cfg-bench.github.io/}{https://cfg-bench.github.io/}.
>
---
#### [replaced 006] SO-Bench: A Structural Output Evaluation of Multimodal LLMs
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **简介: 该论文聚焦多模态大模型的结构化输出任务，旨在解决视觉输入下符合预定义模式的信息提取与推理评估问题。作者构建了包含多种视觉领域的SO-Bench基准，并通过训练实验提升模型结构化输出能力。**

- **链接: [https://arxiv.org/pdf/2511.21750v2](https://arxiv.org/pdf/2511.21750v2)**

> **作者:** Di Feng; Kaixin Ma; Feng Nan; Haofeng Chen; Bohan Zhai; David Griffiths; Mingfei Gao; Zhe Gan; Eshan Verma; Yinfei Yang; Zhifeng Chen; Afshin Dehghan
>
> **备注:** v2 preprint. Fixed some typos, add a discussion about limitation, provide pseudo-codes for eval
>
> **摘要:** Multimodal large language models (MLLMs) are increasingly deployed in real-world, agentic settings where outputs must not only be correct, but also conform to predefined data schemas. Despite recent progress in structured generation in textual domain, there is still no benchmark that systematically evaluates schema-grounded information extraction and reasoning over visual inputs. In this work, we conduct a comprehensive study of visual structural output capabilities for MLLMs with our carefully designed SO-Bench benchmark. Covering four visual domains, including UI screens, natural images, documents, and charts, SO-Bench is built from over 6.5K diverse JSON schemas and 1.8K curated image-schema pairs with human-verified quality. Benchmarking experiments on open-sourced and frontier proprietary models reveal persistent gaps in predicting accurate, schema compliant outputs, highlighting the need for better multimodal structured reasoning. Beyond benchmarking, we further conduct training experiments to largely improve the model's structured output capability. We plan to make the benchmark available to the community.
>
---
#### [replaced 007] Estimating the Joint Probability of Scenario Parameters with Gaussian Mixture Copula Models
- **分类: cs.RO; cs.LG**

- **简介: 该论文针对自动驾驶安全验证中的场景参数联合概率估计问题，提出采用高斯混合Copula模型。该方法结合高斯混合模型的多模态表达能力与Copula对依赖结构的灵活建模，实现了更准确的概率估计，优于传统方法。**

- **链接: [https://arxiv.org/pdf/2506.10098v3](https://arxiv.org/pdf/2506.10098v3)**

> **作者:** Christian Reichenbächer; Philipp Rank; Jochen Hipp; Oliver Bringmann
>
> **备注:** 9 pages, 4 figures; This work has been submitted to the IEEE for possible publication; Code available at: https://codeocean.com/capsule/1003615/tree
>
> **摘要:** This paper presents the first application of Gaussian Mixture Copula Models to the statistical modeling of driving scenarios for the safety validation of automated driving systems. Knowledge of the joint probability distribution of scenario parameters is essential for scenario-based safety assessment, where risk quantification depends on the likelihood of concrete parameter combinations. Gaussian Mixture Copula Models bring together the multimodal expressivity of Gaussian Mixture Models and the flexibility of copulas, enabling separate modeling of marginal distributions and dependencies. We benchmark Gaussian Mixture Copula Models against previously proposed approaches - Gaussian Mixture Models and Gaussian Copula Models - using real-world driving data drawn from two scenarios defined in United Nations Regulation No. 157. Our evaluation on approximately 18 million instances of these two scenarios demonstrates that Gaussian Mixture Copula Models consistently surpass Gaussian Copula Models and perform better than, or at least comparably to, Gaussian Mixture Models, as measured by both log-likelihood and Sinkhorn distance. These results are promising for the adoption of Gaussian Mixture Copula Models as a statistical foundation for future scenario-based validation frameworks.
>
---
#### [replaced 008] Designing for Distributed Heterogeneous Modularity: On Software Architecture and Deployment of MoonBots
- **分类: cs.RO**

- **简介: 该论文设计面向分布式异构模块化的软件架构与部署策略，解决月球机器人系统中多组件协同、动态重构与跨环境扩展难题。提出基于组件化设计、ROS2/Zenoh通信与部署编排器的开源Motion Stack架构，实现低开销、高鲁棒的远程自治协作。**

- **链接: [https://arxiv.org/pdf/2511.01437v2](https://arxiv.org/pdf/2511.01437v2)**

> **作者:** Elian Neppel; Shamistan Karimov; Ashutosh Mishra; Gustavo Hernan Diaz Huenupan; Hazal Gozbasi; Kentaro Uno; Shreya Santra; Kazuya Yoshida
>
> **备注:** 6 pages, 8 figures. Accepted at ISPARO 2025
>
> **摘要:** This paper presents the software architecture and deployment strategy behind the MoonBot platform: a modular space robotic system composed of heterogeneous components distributed across multiple computers, networks and ultimately celestial bodies. We introduce a principled approach to distributed, heterogeneous modularity, extending modular robotics beyond physical reconfiguration to software, communication and orchestration. We detail the architecture of our system that integrates component-based design, a data-oriented communication model using ROS2 and Zenoh, and a deployment orchestrator capable of managing complex multi-module assemblies. These abstractions enable dynamic reconfiguration, decentralized control, and seamless collaboration between numerous operators and modules. At the heart of this system lies our open-source Motion Stack software, validated by months of field deployment with self-assembling robots, inter-robot cooperation, and remote operation. Our architecture tackles the significant hurdles of modular robotics by significantly reducing integration and maintenance overhead, while remaining scalable and robust. Although tested with space in mind, we propose generalizable patterns for designing robotic systems that must scale across time, hardware, teams and operational environments.
>
---
#### [replaced 009] SkillWrapper: Generative Predicate Invention for Skill Abstraction
- **分类: cs.RO**

- **简介: 该论文研究技能抽象中的生成性谓词发明任务，旨在从视觉输入中学习可规划的符号表示。提出SkillWrapper方法，利用基础模型主动采集数据，学习黑箱技能的可解释、保真抽象，实现长视野任务的可靠规划。**

- **链接: [https://arxiv.org/pdf/2511.18203v3](https://arxiv.org/pdf/2511.18203v3)**

> **作者:** Ziyi Yang; Benned Hedegaard; Ahmed Jaafar; Yichen Wei; Skye Thompson; Shreyas S. Raman; Haotian Fu; Stefanie Tellex; George Konidaris; David Paulius; Naman Shah
>
> **摘要:** Generalizing from individual skill executions to solving long-horizon tasks remains a core challenge in building autonomous agents. A promising direction is learning high-level, symbolic abstractions of the low-level skills of the agents, enabling reasoning and planning independent of the low-level state space. Among possible high-level representations, object-centric skill abstraction with symbolic predicates has been proven to be efficient because of its compatibility with domain-independent planners. Recent advances in foundation models have made it possible to generate symbolic predicates that operate on raw sensory inputs, a process we call generative predicate invention, to facilitate downstream abstraction learning. However, it remains unclear which formal properties the learned representations must satisfy, and how they can be learned to guarantee these properties. In this paper, we address both questions by presenting a formal theory of generative predicate invention for skill abstraction, resulting in symbolic operators that can be used for provably sound and complete planning. Within this framework, we propose SkillWrapper, a method that leverages foundation models to actively collect robot data and learn human-interpretable, plannable representations of black-box skills, using only RGB image observations. Our extensive empirical evaluation in simulation and on real robots shows that SkillWrapper learns abstract representations that enable solving unseen, long-horizon tasks in the real world with black-box skills.
>
---
#### [replaced 010] Surfel-LIO: Fast LiDAR-Inertial Odometry with Pre-computed Surfels and Hierarchical Z-order Voxel Hashing
- **分类: cs.RO**

- **简介: 该论文研究激光-惯性里程计（LIO）任务，旨在提升实时位姿估计效率。针对邻域搜索耗时和重复平面拟合问题，提出Surfel-LIO方法，采用预计算面元与分层Z序 voxel 结构，实现快速对应点查找与高效空间索引，显著提升处理速度并保持精度。**

- **链接: [https://arxiv.org/pdf/2512.03397v2](https://arxiv.org/pdf/2512.03397v2)**

> **作者:** Seungwon Choi; Dong-Gyu Park; Seo-Yeon Hwang; Tae-Wan Kim
>
> **摘要:** LiDAR-inertial odometry (LIO) is an active research area, as it enables accurate real-time state estimation in GPS-denied environments. Recent advances in map data structures and spatial indexing have significantly improved the efficiency of LIO systems. Nevertheless, we observe that two aspects may still leave room for improvement: (1) nearest neighbor search often requires examining multiple spatial units to gather sufficient points for plane fitting, and (2) plane parameters are typically recomputed at every iteration despite unchanged map geometry. Motivated by these observations, we propose Surfel-LIO, which employs a hierarchical voxel structure (hVox) with pre-computed surfel representation. This design enables O(1) correspondence retrieval without runtime neighbor enumeration or plane fitting, combined with Z-order curve encoding for cache-friendly spatial indexing. Experimental results on the M3DGR dataset demonstrate that our method achieves significantly faster processing speed compared to recent state-of-the-art methods while maintaining comparable state estimation accuracy. Our implementation is publicly available at https://github.com/93won/lidar_inertial_odometry.
>
---
#### [replaced 011] Q-STAC: Q-Guided Stein Variational Model Predictive Actor-Critic
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于强化学习任务，旨在解决模型偏差、手动设计代价函数和计算开销高的问题。作者提出Q-STAC框架，结合贝叶斯MPC与SAC，利用Q引导的SVGD优化动作序列，提升样本效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2507.06625v2](https://arxiv.org/pdf/2507.06625v2)**

> **作者:** Shizhe Cai; Zeya Yin; Jayadeep Jacob; Fabio Ramos
>
> **备注:** 9 pages, 10 figures
>
> **摘要:** Deep reinforcement learning (DRL) often struggles with complex robotic manipulation tasks due to low sample efficiency and biased value estimation. Model-based reinforcement learning (MBRL) improves efficiency by leveraging environment dynamics, with prior work integrating Model Predictive Control (MPC) to enhance policy robustness through online trajectory optimization. However, existing MBRL approaches still suffer from high model bias, task-specific cost function design, and significant computational overhead. To address these challenges, we propose Q-guided Stein Variational Model Predictive Actor-Critic (Q-STAC)--a unified framework that bridges Bayesian MPC and Soft Actor-Critic (SAC). Q-STAC employs Stein Variational Gradient Descent (SVGD) to iteratively optimize action sequences sampled from a learned prior distribution guided by Q-values, thereby eliminating manual cost-function engineering. By performing short-horizon model-predictive rollouts, Q-STAC reduces cumulative prediction errors, improves training stability and reduces computational complexity. Experiments on simulated particle navigation, diverse robotic manipulation tasks, and a real-world fruit-picking scenario demonstrate that Q-STAC consistently achieves superior sample efficiency, stability, and overall performance compared to both model-free and model-based baselines.
>
---
#### [replaced 012] GigaBrain-0: A World Model-Powered Vision-Language-Action Model
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出GigaBrain-0，解决VLA模型依赖大量真实机器人数据的问题。通过世界模型生成多样化数据，结合RGBD输入和具身链式思维监督，提升跨任务泛化与策略鲁棒性，实现高效灵巧操作与长视野任务控制。**

- **链接: [https://arxiv.org/pdf/2510.19430v3](https://arxiv.org/pdf/2510.19430v3)**

> **作者:** GigaBrain Team; Angen Ye; Boyuan Wang; Chaojun Ni; Guan Huang; Guosheng Zhao; Haoyun Li; Jie Li; Jiagang Zhu; Lv Feng; Peng Li; Qiuping Deng; Runqi Ouyang; Wenkang Qin; Xinze Chen; Xiaofeng Wang; Yang Wang; Yifan Li; Yilong Li; Yiran Ding; Yuan Xu; Yun Ye; Yukun Zhou; Zhehao Dong; Zhenan Wang; Zhichao Liu; Zheng Zhu
>
> **备注:** https://gigabrain0.github.io/
>
> **摘要:** Training Vision-Language-Action (VLA) models for generalist robots typically requires large-scale real-world robot data, which is expensive and time-consuming to collect. The inefficiency of physical data collection severely limits the scalability, and generalization capacity of current VLA systems. To address this challenge, we introduce GigaBrain-0, a novel VLA foundation model empowered by world model-generated data (e.g., video generation, real2real transfer, human transfer, view transfer, sim2real transfer data). By leveraging world models to generate diverse data at scale, GigaBrain-0 significantly reduces reliance on real robot data while improving cross-task generalization. Our approach further improves policy robustness through RGBD input modeling and embodied Chain-of-Thought (CoT) supervision, enabling the model to reason about spatial geometry, object states, and long-horizon dependencies during task execution. This leads to substantial gains in real-world performance on dexterous, long-horizon, and mobile manipulation tasks. Extensive experiments demonstrate that GigaBrain-0 achieves superior generalization across variations in appearances (e.g., textures, colors), object placements, and camera viewpoints. Additionally, we present GigaBrain-0-Small, an optimized lightweight variant designed to run efficiently on devices such as the NVIDIA Jetson AGX Orin.
>
---
#### [replaced 013] WeatherPrompt: Multi-modality Representation Learning for All-Weather Drone Visual Geo-Localization
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究无人机视觉地理定位任务，旨在解决天气变化导致的性能下降问题。提出WeatherPrompt框架，通过图文融合与动态门控机制，实现天气无关的特征表示，提升复杂天气下的定位精度。**

- **链接: [https://arxiv.org/pdf/2508.09560v3](https://arxiv.org/pdf/2508.09560v3)**

> **作者:** Jiahao Wen; Hang Yu; Zhedong Zheng
>
> **摘要:** Visual geo-localization for drones faces critical degradation under weather perturbations, \eg, rain and fog, where existing methods struggle with two inherent limitations: 1) Heavy reliance on limited weather categories that constrain generalization, and 2) Suboptimal disentanglement of entangled scene-weather features through pseudo weather categories. We present WeatherPrompt, a multi-modality learning paradigm that establishes weather-invariant representations through fusing the image embedding with the text context. Our framework introduces two key contributions: First, a Training-free Weather Reasoning mechanism that employs off-the-shelf large multi-modality models to synthesize multi-weather textual descriptions through human-like reasoning. It improves the scalability to unseen or complex weather, and could reflect different weather strength. Second, to better disentangle the scene and weather feature, we propose a multi-modality framework with the dynamic gating mechanism driven by the text embedding to adaptively reweight and fuse visual features across modalities. The framework is further optimized by the cross-modal objectives, including image-text contrastive learning and image-text matching, which maps the same scene with different weather conditions closer in the respresentation space. Extensive experiments validate that, under diverse weather conditions, our method achieves competitive recall rates compared to state-of-the-art drone geo-localization methods. Notably, it improves Recall@1 by +13.37\% under night conditions and by 18.69\% under fog and snow conditions.
>
---
#### [replaced 014] PPL: Point Cloud Supervised Proprioceptive Locomotion Reinforcement Learning for Legged Robots in Crawl Spaces
- **分类: cs.RO**

- **简介: 该论文研究足式机器人在狭小空间中的本体感知运动问题，提出一种点云监督的强化学习框架。通过设计状态估计网络和极坐标点云特征提取方法，实现对碰撞及环境特征的感知，提升机器人在无外部传感器情况下的敏捷通行能力。**

- **链接: [https://arxiv.org/pdf/2508.09950v2](https://arxiv.org/pdf/2508.09950v2)**

> **作者:** Bida Ma; Nuo Xu; Chenkun Qi; Xin Liu; Yule Mo; Jinkai Wang; Chunpeng Lu
>
> **备注:** Accepted by RA-L
>
> **摘要:** Legged locomotion in constrained spaces (called crawl spaces) is challenging. In crawl spaces, current proprioceptive locomotion learning methods are difficult to achieve traverse because only ground features are inferred. In this study, a point cloud supervised RL framework for proprioceptive locomotion in crawl spaces is proposed. A state estimation network is designed to estimate the robot's collision states as well as ground and spatial features for locomotion. A point cloud feature extraction method is proposed to supervise the state estimation network. The method uses representation of the point cloud in polar coordinate frame and MLPs for efficient feature extraction. Experiments demonstrate that, compared with existing methods, our method exhibits faster iteration time in the training and more agile locomotion in crawl spaces. This study enhances the ability of legged robots to traverse constrained spaces without requiring exteroceptive sensors.
>
---
#### [replaced 015] Energy-Aware Lane Planning for Connected Electric Vehicles in Urban Traffic: Design and Vehicle-in-the-Loop Validation
- **分类: eess.SY; cs.RO**

- **简介: 该论文研究面向城市交通中联网电动车的节能行驶任务，解决传统策略忽略变道对能耗影响的问题。提出一种联合优化纵向速度与横向变道的能效规划框架，并通过车联设施通信和实车实验验证，显著降低能耗。**

- **链接: [https://arxiv.org/pdf/2503.23228v3](https://arxiv.org/pdf/2503.23228v3)**

> **作者:** Hansung Kim; Eric Yongkeun Choi; Eunhyek Joa; Hotae Lee; Linda Lim; Scott Moura; Francesco Borrelli
>
> **备注:** Accepted at 2025 IEEE Conference on Decision and Control (CDC25')
>
> **摘要:** Urban driving with connected and automated vehicles (CAVs) offers potential for energy savings, yet most eco-driving strategies focus solely on longitudinal speed control within a single lane. This neglects the significant impact of lateral decisions, such as lane changes, on overall energy efficiency, especially in environments with traffic signals and heterogeneous traffic flow. To address this gap, we propose a novel energy-aware motion planning framework that jointly optimizes longitudinal speed and lateral lane-change decisions using vehicle-to-infrastructure (V2I) communication. Our approach estimates long-term energy costs using a graph-based approximation and solves short-horizon optimal control problems under traffic constraints. Using a data-driven energy model calibrated to an actual battery electric vehicle, we demonstrate with vehicle-in-the-loop experiments that our method reduces motion energy consumption by up to 24 percent compared to a human driver, highlighting the potential of connectivity-enabled planning for sustainable urban autonomy.
>
---
#### [replaced 016] Bootstrap Dynamic-Aware 3D Visual Representation for Scalable Robot Learning
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人学习中的3D视觉表征任务，旨在解决现有3D预训练方法在操作任务中性能不足的问题。作者提出AFRO框架，通过扩散模型联合学习状态-动作-状态动力学，无需动作或重建监督，提升表征质量与策略性能。**

- **链接: [https://arxiv.org/pdf/2512.00074v2](https://arxiv.org/pdf/2512.00074v2)**

> **作者:** Qiwei Liang; Boyang Cai; Minghao Lai; Sitong Zhuang; Tao Lin; Yan Qin; Yixuan Ye; Jiaming Liang; Renjing Xu
>
> **摘要:** Despite strong results on recognition and segmentation, current 3D visual pre-training methods often underperform on robotic manipulation. We attribute this gap to two factors: the lack of state-action-state dynamics modeling and the unnecessary redundancy of explicit geometric reconstruction. We introduce AFRO, a self-supervised framework that learns dynamics-aware 3D representations without action or reconstruction supervision. AFRO casts state prediction as a generative diffusion process and jointly models forward and inverse dynamics in a shared latent space to capture causal transition structure. To prevent feature leakage in action learning, we employ feature differencing and inverse-consistency supervision, improving the quality and stability of visual features. When combined with Diffusion Policy, AFRO substantially increases manipulation success rates across 16 simulated and 4 real-world tasks, outperforming existing pre-training approaches. The framework also scales favorably with data volume and task complexity. Qualitative visualizations indicate that AFRO learns semantically rich, discriminative features, offering an effective pre-training solution for 3D representation learning in robotics. Project page: https://kolakivy.github.io/AFRO/
>
---
#### [replaced 017] The Autonomy-Alignment Problem in Open-Ended Learning Robots: Formalising the Purpose Framework
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文针对开放学习机器人中的自主性与人类意图对齐难题，提出一个名为“目的”的计算框架。通过分解对齐问题为四个子问题，形式化定义并验证对齐条件，旨在指导构建既自主又符合人类目标与价值观的机器人系统。**

- **链接: [https://arxiv.org/pdf/2403.02514v3](https://arxiv.org/pdf/2403.02514v3)**

> **作者:** Gianluca Baldassarre; Richard J. Duro; Emilio Cartoni; Mehdi Khamassi; Alejandro Romero; Vieri Giuliano Santucci
>
> **备注:** 33 pages, 5 figures
>
> **摘要:** The rapid advancement of artificial intelligence is enabling the development of increasingly autonomous robots capable of operating beyond engineered factory settings and into the unstructured environments of human life. This shift raises a critical autonomy-alignment problem: how to ensure that a robot's autonomous learning focuses on acquiring knowledge and behaviours that serve human practical objectives while remaining aligned with broader human values (e.g., safety and ethics). This problem remains largely underexplored and lacks a unifying conceptual and formal framework. Here, we address one of its most challenging instances of the problem: open-ended learning (OEL) robots, which autonomously acquire new knowledge and skills through interaction with the environment, guided by intrinsic motivations and self-generated goals. We propose a computational framework, introduced qualitatively and then formalised, to guide the design of OEL architectures that balance autonomy with human control. At its core is the novel concept of purpose, which specifies what humans (designers or users) want the robot to learn, do, or avoid, independently of specific task domains. The framework decomposes the autonomy-alignment problem into four tractable sub-problems: the alignment of robot purposes (hardwired or learnt) with human purposes; the arbitration between multiple purposes; the grounding of abstract purposes into domain-specific goals; and the acquisition of competence to achieve those goals. The framework supports formal definitions of alignment across multiple cases and proofs of necessary and sufficient conditions under which alignment holds. Illustrative hypothetical scenarios showcase the applicability of the framework for guiding the development of purpose-aligned autonomous robots.
>
---
#### [replaced 018] BOP-ASK: Object-Interaction Reasoning for Vision-Language Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文聚焦视觉语言模型的物体交互推理任务，旨在解决现有模型在精细空间理解上的不足。作者构建了大规模数据集BOP-ASK，包含6D物体姿态衍生的细粒度标注，用于训练和评测模型在3D定位、物理兼容性、抓取规划等方面的推理能力。**

- **链接: [https://arxiv.org/pdf/2511.16857v2](https://arxiv.org/pdf/2511.16857v2)**

> **作者:** Vineet Bhat; Sungsu Kim; Valts Blukis; Greg Heinrich; Prashanth Krishnamurthy; Ramesh Karri; Stan Birchfield; Farshad Khorrami; Jonathan Tremblay
>
> **摘要:** Vision Language Models (VLMs) have achieved impressive performance on spatial reasoning benchmarks, yet these evaluations mask critical weaknesses in understanding object interactions. Current benchmarks test high level relationships ('left of,' 'behind', etc.) but ignore fine-grained spatial understanding needed for real world applications: precise 3D localization, physical compatibility between objects, object affordances and multi step spatial planning. In this work, we present BOP-ASK, a novel large scale dataset for object interaction reasoning for both training and benchmarking. Our data generation pipeline leverages 6D object poses from the Benchmark for Object Pose Estimation (BOP) datasets from which we derive fine grained annotations such as grasp poses, referred object poses, path planning trajectories, relative spatial and depth relationships, and object-to-object relationships. BOP-ASK comprises over 150k images and 33M question answer pairs spanning six tasks (four novel), providing a rich resource for training and evaluating VLMs. We evaluate proprietary and open sourced VLMs, and conduct human evaluations on BOP-ASK-core, a contributed test benchmark. We also release BOP-ASK-lab, an out-of-distribution benchmark with images not sourced from BOP, enabling testing of generalization. Our experiments demonstrate that models trained on BOP-ASK outperform baselines and exhibit emergent capabilities such as precise object and grasp pose estimation, trajectory planning, and fine-grained object-centric spatial reasoning in cluttered environments. We will publicly release our datasets and dataset generation pipeline.
>
---
