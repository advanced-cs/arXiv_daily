# 机器人 cs.RO

- **最新发布 89 篇**

- **更新 40 篇**

## 最新发布

#### [new 001] RePlan-Bot: Multi-Level Replanning for Embodied Instruction Following
- **分类: cs.RO**

- **简介: 该论文提出RePlan-Bot，解决 embodied instruction following 任务中的长程规划和不可逆状态问题，通过多级重规划提升任务成功率。**

- **链接: [https://arxiv.org/pdf/2605.25851](https://arxiv.org/pdf/2605.25851)**

> **作者:** Xicheng Gong; Guozheng Sun; Peiran Xu; Yadong Mu
>
> **备注:** 10 pages
>
> **摘要:** Embodied instruction following (EIF) requires agents to understand and execute complex natural language commands within interactive 3D environments. Despite recent advances, existing methods often fail in long-horizon planning and handling irreversible state changes, resulting in low task success rates. To address these challenges, we introduce RePlan-Bot, a novel EIF agent that performs multi-level, continuous replanning throughout task execution. RePlan-Bot integrates a high-level LLM-based auditor for dynamic sub-goal adjustments guided by environmental feedback, a commonsense-guided search mechanism based on a multi-layered instance map for precise and structured object localization, and a lightweight ViT-based corrector to preemptively fix risky low-level actions. Evaluated on the ALFRED benchmark, RePlan-Bot achieves state-of-the-art performance in both seen and unseen environments, demonstrating superior adaptability and reliability.
>
---
#### [new 002] Prior Policy Guided Dual-Agent Coordinated Manipulation Planning of Spacecraft-Manipulator System
- **分类: cs.RO**

- **简介: 该论文属于空间机械臂协同操作任务，旨在解决机械臂与航天器动态耦合导致的姿态不稳定问题。提出DACMP框架，结合深度强化学习提升操作精度和姿态稳定性。**

- **链接: [https://arxiv.org/pdf/2605.25362](https://arxiv.org/pdf/2605.25362)**

> **作者:** Yuhui Hu; Dong Zhou; Kaihong Ouyang; Zhongliang Yu; Jianfeng Lv; Xiangyu Shao
>
> **备注:** 36 pages, 13 figures, 6 tables. Under review
>
> **摘要:** The strong dynamic coupling between the manipulator and the base poses a significant challenge to maintaining spacecraft attitude stability, potentially compromising mission safety. In this paper, we propose a Dual-Agent Coordinated Manipulation Planning (DACMP) framework that simultaneously achieves high-precision end-effector pose reaching for a 6-DoF space manipulator and attitude stabilization of the base spacecraft. To enhance learning efficiency, we present a prior policy-guided Deep Reinforcement Learning algorithm incorporating the Timestep-level Expert Switching Guidance (TESG) mechanism, thereby promoting global convergence and improving task success rates. Extensive experiments demonstrate that DACMP significantly outperforms baseline DRL algorithms in terms of task success rate and control precision. Furthermore, the robustness of DACMP is validated under various challenging scenarios, including system constraints, environmental disturbances, and perception uncertainties. The code and simulation configurations are available on GitHub: this https URL.
>
---
#### [new 003] AcroRL: Learning Aggressive Quadrotor Inversion using Bidirectional Thrust
- **分类: cs.RO**

- **简介: 该论文提出AcroRL框架，解决四旋翼无人机在反向飞行中的控制问题，通过强化学习实现高效、精准的翻转操作。**

- **链接: [https://arxiv.org/pdf/2605.24301](https://arxiv.org/pdf/2605.24301)**

> **作者:** Gabriel Rodriguez; Henri Sayag; Abhishek Rathod; John Stecklein; Siddharth Saha; Christopher Barngrover; Wennie Tabib
>
> **备注:** 17 pages, 8 figures
>
> **摘要:** Bidirectional thrust grants quadrotors a second equilibrium condition and increased control authority, expanding the envelope of possible aggressive maneuvers and enabling inverted flight, perching, and sensing. Prior geometric control approaches extend differential flatness through Hopf fibration-based attitude representations to support bidirectional thrust, but struggle with actuator saturation and motor reversal delay during inversions, requiring heuristic thrust posture scheduling and waypoint tuning. We propose a learning-based framework that modulates a constant reference trajectory to perform compact, position-constrained quadrotor inversions while remaining compatible with traditional trajectory generation and tracking across flight regimes. Separate policies are trained via reinforcement learning for nominal-to-inverted and inverted-to-nominal transitions. In JAX-based simulation, the proposed method achieves the lowest position deviation and settling time across all evaluated baselines, reducing position root mean square error (RMSE) by 32% and settling time by 57% relative to the strongest optimization-based baseline. Hardware experiments demonstrate successful inversion across multiple yaw configurations with position RMSE below 0.35m, and compatibility with downstream trajectory generation and control through circular flight in both regimes. Additionally, we provide an open-source implementation of the proposed framework.
>
---
#### [new 004] Learning Transferable Motor Skills for Geometry-Aware Robotic Surface Tasks
- **分类: cs.RO**

- **简介: 该论文属于机器人表面操作任务，解决几何规划与运动执行脱节的问题。通过模块化框架分离几何路径与专家运动规则，提升任务迁移能力。**

- **链接: [https://arxiv.org/pdf/2605.24881](https://arxiv.org/pdf/2605.24881)**

> **作者:** Miroslav David; Karla Stepanova; Robert Babuska
>
> **备注:** 4 pages (3 text, 1 references), 2 figures
>
> **摘要:** Robotic surface-interaction tasks, such as spray painting or welding, require both accurate geometric planning and precise motion execution. While modern motion planners generate valid geometric paths, they often lack the expert motor patterns observed in human operators. Conversely, learning from demonstration often tightly couples task execution to the specific training geometry, limiting transferability. We propose a modular framework that decouples geometric motion planning from execution-level expertise. Expert behavior is represented as a vocabulary of interpretable, atomic motor rules, such as velocity scaling and orientation offsets, that systematically modify a geometrically planned reference path. We train a multimodal neural network to infer rule parameters jointly from kinematic trajectory data and CAD model geometry. We evaluate our approach through dynamic simulation on L-shaped and window-shaped objects, demonstrating on simulated data that the model successfully extracts velocity and orientation rules across both topologies.
>
---
#### [new 005] Learning, locomotion, and navigation of soft synthetic snakes in three-dimensional, heterogeneous environments
- **分类: cs.RO; cs.LG; physics.comp-ph**

- **简介: 该论文属于机器人导航任务，旨在解决软体蛇形机器人在复杂三维环境中的运动与导航问题。通过生物启发的控制模型和强化学习，实现其在真实地形中的可靠移动。**

- **链接: [https://arxiv.org/pdf/2605.24985](https://arxiv.org/pdf/2605.24985)**

> **作者:** Xiaotian Zhang; Ali Albazroun; Tixian Wang; Songyuan Cui; Prashant G. Mehta; Mattia Gazzola
>
> **备注:** 14 pages, 5 figures
>
> **摘要:** Limbless terrestrial animals exhibit exceptional locomotor versatility and control, currently unmatched by engineered counterparts. Here, we introduce a computational framework that enables soft synthetic snakes to navigate unstructured, heterogeneous 3D terrains. Our approach is grounded in bio-inspired actuation and sensing models that reduce the control complexity inherent to high-degree-of-freedom, continuum bodies. These models are integrated into a reinforcement learning architecture to derive environment-traversing policies. Training first occurs in simplified, homogeneous terrains to learn locomotion primitives. These are then composed into adaptive strategies for complex landscapes. We demonstrate robustness by deploying a snake in high-fidelity 3D environments reconstructed from real-world imaging, achieving reliable navigation. Overall, this work provides a physically-realistic simulation platform and practical insights for the control of continuum systems in natural terrains.
>
---
#### [new 006] ParkourFormer: Integrating Predictive Supervision and Sequence Modeling into Parkour Locomotion
- **分类: cs.RO**

- **简介: 该论文提出ParkourFormer，用于解决人形机器人在复杂地形上的敏捷运动问题。通过结合预测监督和序列建模，提升运动策略的鲁棒性和泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.25782](https://arxiv.org/pdf/2605.25782)**

> **作者:** Yanheng Mai; Wenhao Xu; Zirui Huang; Yifei Fu; Shengwei Dong; Xinjue Wang; Kailun Huang; Yanzhe Xie; Renjing Xu
>
> **摘要:** Humanoid parkour requires locomotion policies to coordinate whole-body dynamics across rapidly changing terrains such as stairs, gaps, slopes, and obstacles. Existing reinforcement learning policies are largely reactive, mapping observations directly to actions without explicitly modeling future body states. Such modeling becomes critical in agile locomotion tasks where successful motion execution depends strongly on anticipating upcoming contact transitions and body this http URL present ParkourFormer, a Transformer-based sequence modeling framework that reformulates humanoid locomotion as a future-conditioned decision-making problem. The current robot state queries historical sensorimotor trajectories through cross-attention, while a lightweight prediction head forecasts short-horizon future proprioceptive states. The predicted future states, trained with supervised signals, are fused with temporal features to generate actions, enabling the policy to jointly reason over motion history and anticipated future dynamics. We evaluate ParkourFormer on a diverse multi-terrain humanoid parkour benchmark including stairs, gaps, slopes, rough terrain, and obstacle traversal. Experiments in simulation and on a real humanoid robot show that ParkourFormer achieves a 93.85% average traversal success rate on highly challenging terrains, with improvements of up to 42.73% over strong MLP, MoE-based MLP, and vanilla Transformer baselines, while maintaining a single unified policy across all terrain types. These results demonstrate that explicit future-state modeling significantly improves robustness and generalization for agile whole-body locomotion.
>
---
#### [new 007] ECo-MoE: Embodiment-Conditioned Mixture of Experts Increases the Evolvability of Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人进化与学习任务，旨在解决传统方法效率低或保守的问题。提出ECo-MoE模型，联合优化设计向量和控制专家模块，提升机器人可进化性。**

- **链接: [https://arxiv.org/pdf/2605.24225](https://arxiv.org/pdf/2605.24225)**

> **作者:** Yibin Wang; Muhan Li; Zihan Guo; Sam Kriegman
>
> **摘要:** In this paper, we introduce a model of evolution and learning in robots that co-optimizes a distribution of latent design vectors (genotypes) and a mixture of control experts (neural modules), which are gated by the latent coordinates of each decoded design (phenotype). This provides a scalable alternative to co-design algorithms that either train an individual policy for every robot, which is inefficient, or a monolithic universal controller for all robots, which results in overly conservative structures and behaviors. Our approach lies somewhere between these two extremes, preserving ancestral knowledge in a unified yet modular framework in which different body plans activate and deactivate different combinations of learned sensorimotor circuits for goal-directed behavior. This allows one part of the controller to be overhauled to better suit new species of designs as they emerge without disrupting the hard-earned knowledge contained within other expert modules. It also allows pretrained expert policies to be directly plugged into the mixture, which can steer evolution into otherwise unexplored areas of latent space containing desired morphological traits. We refer to this process as "evo by demo" and explore how it may be used to guide freeform evolution toward canonical structures defined by the pretrained model. Videos and code can be found at: this https URL.
>
---
#### [new 008] Sum of Costs Diffusion with Dynamic Guidance for Motion Planning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人运动规划任务，解决轨迹生成中的泛化问题。通过动态引导扩散模型，使用碰撞成本梯度生成无碰撞路径，提升性能。**

- **链接: [https://arxiv.org/pdf/2605.24690](https://arxiv.org/pdf/2605.24690)**

> **作者:** Aysu Aylin Kaplan; Özgür Erkent
>
> **备注:** Accepted at the Frontiers of Optimization for Robotics Workshop at the IEEE International Conference of Robotics & Automation (ICRA), 2026
>
> **摘要:** The motion planning problem for robotic manipulation can be addressed through classical or deep learning approaches. Existing methods face significant challenges in generalizing to diverse settings. In this study, we present a method with high generalization capability that generates collision-free trajectories using diffusion models where the denoising process is guided by the gradient of the total collision cost. We are also presenting a dynamic approach for choosing start step of the gradient guidance. Experimental results demonstrate that guiding the diffusion model dynamically with the sum of collision costs offers more robust performance by overcoming the generalization issues faced by competing methods. The proposed model demonstrates its effectiveness by achieving the highest performance on diverse test settings in M$\pi$nets\ dataset among the compared methods.
>
---
#### [new 009] HoLoArm: Deformable Arms for Collision-Tolerant Quadrotor Flight
- **分类: cs.RO**

- **简介: 论文提出HoLoArm无人机，通过柔性臂设计和强化学习控制，解决碰撞容忍与快速恢复问题，提升飞行安全与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.25790](https://arxiv.org/pdf/2605.25790)**

> **作者:** Quang Ngoc Pham; Jonas Eschmann; Yang Zhou; Alejandro Ojeda Olarte; Giuseppe Loianno; Van Anh Ho
>
> **备注:** 8 pages, 15 figures, 1 table, Accepted at the IEEE Robotics and Automation Letters (RA-L) and the IEEE International Conference on Robotics and Automation (ICRA), 2026
>
> **摘要:** The increasing use of drones in human-centric applications highlights the need for designs that can survive collisions and recover rapidly, minimizing risks to both humans and the environment. We present HoLoArm, a quadrotor with compliant arms inspired by the nodus structure of dragonfly wings. This design provides natural flexibility and resilience while preserving flight stability, which is further reinforced by the integration of a Reinforcement Learning (RL) control policy that enhances both recovery and hovering performance. Experimental results demonstrate that HoLoArm can passively deform in any direction, including axial one, and recover within 0.3-0.6 s depending on the direction and level of the impact. The drone can survive collisions at speeds up to 7.6 m/s and carry a 540 g payload while maintaining stable flight. This work contributes to the morphological design of soft aerial robots with high agility and reliable safety, enabling operation in cluttered and human shared environments, and lays the groundwork for future fully soft drones that integrate compliant structures with intelligent control.
>
---
#### [new 010] MuGen: Multi-Skill Generative Locomotion Controller for Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文提出MuGen，用于双足机器人多技能运动控制的生成框架，解决机器人模仿人类运动的问题。通过VQ-VAEs和策略蒸馏，实现高效运动表示与任务迁移。**

- **链接: [https://arxiv.org/pdf/2605.24592](https://arxiv.org/pdf/2605.24592)**

> **作者:** Yusen Feng; Xiang Wang; Heyuan Yao; Zixi Kang; Xinyu Huo; Boyang Yu; Pengyun Qiu; Ruijie Zhao; Baoquan Chen; Libin Liu
>
> **摘要:** This paper presents MuGen, a data-driven framework for learning and deploying multi-skill locomotion on humanoid robots. MuGen enables a robot to perform expressive motions like humans under the guidance of example motion sequences. To achieve this, we employ vector-quantized autoencoders (VQ-VAEs) trained with model-based reinforcement learning, resulting in a generative representation of locomotion that captures key patterns of human motion from hours of heterogeneous human performance data. We employ a teacher-student learning framework and develop a new policy distillation strategy to enable a deployable student policy learning this efficient latent representation. This policy allows the robot to track and mimic unseen human motions and further enables the robot to reuse the learned latent space for other tasks. We demonstrate the effectiveness of our framework through a diverse set of motions and accurate execution.
>
---
#### [new 011] GreenSeg: Ground Segmentation Algorithm for Agricultural Robots in Mediterranean Greenhouses using RGB-D Point Clouds
- **分类: cs.RO**

- **简介: 该论文属于农业机器人导航任务，解决地中海温室中因环境复杂导致的地面分割问题。提出GreenSeg算法，结合RGB-D点云实现稳定导航。**

- **链接: [https://arxiv.org/pdf/2605.25279](https://arxiv.org/pdf/2605.25279)**

> **作者:** Fernando Cañadas-Aránega; José C. Moreno; José L. Blanco-Claraco
>
> **摘要:** Greenhouse agriculture in the Mediterranean region faces significant automation challenges due to its unique structural and environmental constraints. These environments are characterized by extremely narrow aisles, heterogeneous terrains ranging from concrete to tilled soil and severe optical interference caused by polyethylene covers, which induce specular reflections and "ghost points" in depth sensors. While autonomous navigation is essential for digitizing agricultural tasks, traditional solutions often rely on expensive 3D LiDAR systems that are economically unscalable for most facilities. To address this, this paper presents GreenSeg, a robust perception framework for autonomous navigation using RGB-D sensing. The proposed method introduces a dual-layer validation strategy: a robust global plane fitting combined with a surface curvature filter for terrain adaptability, and a seed-point-based Region Growing constraint to ensure the spatial continuity of the navigable plane. Experimental validation was conducted using the AGRICOBIOT I platform across four diurnal scenarios with varying solar elevations. The results show that GreenSeg consistently outperforms benchmark segmentation methods, achieving peak improvements of 11.58% in mean Recall and 19.24% in mIoU during critical rotational maneuvers at the end of corridors. These findings confirm that the proposed algorithm enables stable and safe autonomous navigation in unstructured, dynamic agricultural environments that are subject to budget constraints and sensitive to lighting conditions.
>
---
#### [new 012] FOUND-IT: Foundation-model-first Task-driven 3D Scene Graphs with Granularity on Demand
- **分类: cs.RO**

- **简介: 该论文提出FOUND-IT方法，用于实时构建可调整粒度的3D场景图，解决室内室外环境的动态任务需求。**

- **链接: [https://arxiv.org/pdf/2605.25371](https://arxiv.org/pdf/2605.25371)**

> **作者:** Dominic Maggio; Nicolas Gorlo; Luca Carlone
>
> **摘要:** We present the first approach to build hierarchical task-driven 3D scene graphs of arbitrary indoor or outdoor environments using an uncalibrated monocular camera in real-time. We leverage geometric foundation models to estimate geometric attributes of the scene graph (e.g., object bounding boxes), but we also observe that traversability information (the "places" layer of a scene graph) can be directly reconstructed by adding an extra head to existing geometric foundation models, like VGGT. Our approach is task-driven in the sense that we adjust the granularity of the objects and regions in the map depending on the task; for instance, during a manipulation task, our approach is able to resolve small knobs on a stove, while during a navigation task it can focus on large objects (e.g., the entire stove). However, in a major departure from related work, we consider the realistic case where the list of tasks is not predefined and fixed, but evolves as the robot operates. This naturally allows dealing with complex loco-manipulation tasks, where the robot can dynamically adjust its representation as the task unfolds. We dub the resulting approach FOUND-IT. FOUND-IT also includes an agentic approach to query information in the scene graph. In addition to achieving 79% higher accuracy on the ASHiTA SG3D task grounding benchmark, we demonstrate FOUND-IT runs in real-time on a ground robot using a Jetson Thor. Furthermore, to highlight the robustness of our method, we demonstrate constructing 3D scene graphs on casually captured realtor apartment tours from YouTube. Code will be made available upon publication.
>
---
#### [new 013] Vision-Guided Outdoor Flight and Obstacle Evasion via Reinforcement Learning
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于自主导航任务，旨在解决无人机在无GPS环境下的避障问题。通过强化学习与视觉感知结合，实现无人机自主飞行与避障。**

- **链接: [https://arxiv.org/pdf/2605.24449](https://arxiv.org/pdf/2605.24449)**

> **作者:** Shiladitya Dutta; Aayush Gupta; Varun Saran; Avideh Zakhor
>
> **备注:** Published in IEEE Robotics and Automation Letters, vol 11, no 2. Presented at the IEEE International Conference on Robotics and Automation 2026
>
> **摘要:** Although quadcopters boast impressive traversal capabilities enabled by their omnidirectional maneuverability, the need for continuous pilot control in complex environments impedes their application in GNSS and telemetry-denied scenarios. To this end, we propose a novel sensorimotor policy that uses stereo-vision depth and visual-inertial odometry (VIO) to autonomously navigate through obstacles in an unknown environment to reach a goal point. The policy is comprised of a pre-trained autoencoder as the perception head followed by a planning and control LSTM network which outputs velocity commands that can be followed by an off-the-shelf commercial drone. We leverage reinforcement and privileged learning paradigms to train the policy in simulation through a two-stage process: 1) initial training with optimal trajectories generated by a global motion planner acting as a supervisory backbone, 2) further fine-tuning in a curriculum environment. To bridge the sim-to-real gap, we employ domain randomization and reward shaping to create a policy that is both robust to noise and domain shift. In outdoor experiments, our approach achieves successful zero-shot transfer to both obstacle environments and a drone platform that were never encountered during training.
>
---
#### [new 014] Loosely Coupled Factor Graph Optimization for Pseudolite-Augmented Navigation
- **分类: cs.RO**

- **简介: 该论文属于导航定位任务，解决GNSS信号弱时的定位问题，通过融合GNSS、伪卫星和IMU数据，提升定位精度。**

- **链接: [https://arxiv.org/pdf/2605.24980](https://arxiv.org/pdf/2605.24980)**

> **作者:** Chih-Chun Chen; Lipeng Tan; Shiyu Bai; Heike Vallery
>
> **摘要:** In Global Navigation Satellite System (GNSS)-degraded environments, pseudolites (PLs) provide additional signal sources to enhance positioning performance, but their integration in optimization-based frameworks remains limited. This paper presents a loosely coupled factor graph optimization (FGO) framework that fuses the GNSS/PL least-squares (LS) solutions with inertial measurement unit (IMU) data. The evaluation considers low GNSS visibility scenarios with four high-elevation GNSS satellites and up to two PL transmitters over an 80~s window. FGO achieves a 22.8\% to 41.3\% reduction in mean 3D error compared to standard LS methods. Compared to a GNSS-IMU baseline, incorporating PL transmitters further improves positioning accuracy, with performance depending on geometry.
>
---
#### [new 015] Towards Low-Gravity Planetary Exploration using Reinforcement Learning for Walking, Jumping, and In-flight Attitude Control
- **分类: cs.RO; eess.SY**

- **简介: 该论文研究低重力环境下行星探索的四足机器人运动控制问题，通过强化学习实现行走、跳跃和飞行姿态控制，以应对火星复杂地形。**

- **链接: [https://arxiv.org/pdf/2605.24643](https://arxiv.org/pdf/2605.24643)**

> **作者:** Jørgen Anker Olsen; Kostas Alexis
>
> **备注:** 16 pages, 16 figures
>
> **摘要:** This paper presents reinforcement learning (RL) policies for dynamic quadrupedal locomotion in planetary exploration scenarios. Building on a taskoptimized quadruped with a 5-bar leg design, we develop RL policies for walking, vertical jumping, forward jumping, and in-flight attitude control, explicitly tailored to the reduced gravity on Mars. These policies jointly enable such robots to overcome obstacles larger than themselves through coordinated jumping and precise in-flight reorientation for safe landings. We demonstrate Sim2Real transfer of the attitude control policy on the Olympus quadruped through single-axis reorientation tests, while all locomotion policies are validated in simulation. A complete Mars exploration mission scenario demonstrates coordinated policy deployment across challenging terrain. Experimental results show 90° attitude reorientation in 2.6 seconds, with simulations demonstrating 3.1 meter vertical jumps and 3.9 meter forward jumps under Martian gravity conditions. - Supplementary video: this https URL
>
---
#### [new 016] Learning High-Frequency Continuous Action Chunks in Latent Space
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决高频率动作执行中的时空一致性问题。通过将动作学习转移到潜在空间，提升控制的平滑性和连续性。**

- **链接: [https://arxiv.org/pdf/2605.24931](https://arxiv.org/pdf/2605.24931)**

> **作者:** Kunyun Wang; Yuhang Zheng; Yupeng Zheng; Jieru Zhao; Wenchao Ding
>
> **备注:** 17 pages, 10 figures
>
> **摘要:** Modern robotic policies increasingly rely on action chunking to execute complex tasks in the physical world. While action chunking improves temporal consistency at moderate action frequencies, it becomes insufficient when the action frequency is further increased (e.g., to 60~Hz). At such high frequencies, policies often fail to generate actions that are both temporally smooth and spatially consistent. We address this challenge by shifting high-frequency action learning from the action space to a latent space with variational autoencoder (VAE). This formulation significantly improves both temporal and spatial consistency of high-frequency control. To enable smooth real-time execution, we further introduce Reuse-then-Refine, a chunk-level refine strategy that improves continuity between adjacent action chunks under asynchronous inference. As a result, robots controlled by our policy can execute complex contact-rich tasks continuously, with less pauses and jerky motions. Experiments on three real-world contact-rich robotic tasks show that our approach consistently completes tasks with smooth motions. Our code and data are available at this https URL.
>
---
#### [new 017] PoseRefer: Pathway-Local Parameters for Semantically Grounded Reference Resolution
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于语义接地任务，解决机器人理解指代性语言的问题。通过构建真实手势与场景数据集，设计解耦融合架构，提升指代消解准确率。**

- **链接: [https://arxiv.org/pdf/2605.24622](https://arxiv.org/pdf/2605.24622)**

> **作者:** Anna Deichler
>
> **备注:** ICRA 2026 Workshop on Semantics for Reliable Robot Autonomy: From Environment Understanding and Reasoning to Safe Interaction
>
> **摘要:** A robot resolving ``put the cup on that one'' must fuse gesture, language, and scene geometry, yet 3D grounding benchmarks only partially capture this regime: descriptions are written post-hoc, gestures are templated, or pointing is staged for the camera. MM-Conv captures natural co-speech gesture from dyadic VR interaction alongside full-body motion capture and 3D scene graphs. We use it to evaluate pose-language fusion with a decoupled late-fusion architecture in which pose and text pathways share no learned parameters. The two choices together make category, pose, and text contributions easier to isolate through controlled ablations. Fusion with frozen MiniLM category embeddings exceeds pose alone and the best text-only pathway on every reference type, reaching 31.9% top-1. The learned scalar gate flips between opposing policies depending on whether the text pathway has category access. This is a reliability diagnostic: fusion-accuracy claims for semantic grounding systems are indistinguishable from category-representation artifacts unless pathways are architecturally decoupled.
>
---
#### [new 018] RAMBA: 4D Radar Mapping by Bundle Adjustment
- **分类: cs.RO**

- **简介: 该论文提出RAMBA，用于4D雷达全局一致地图构建的束调整框架，解决雷达惯性里程计后的地图优化问题，通过多帧优化提升地图一致性与轨迹精度。**

- **链接: [https://arxiv.org/pdf/2605.25041](https://arxiv.org/pdf/2605.25041)**

> **作者:** Jianzhu Huai; Yiwen Chen; Binliang Wang
>
> **备注:** 5 pages, 2 figures, to present in ISPRS2026 Thematic Session 10 on Radar Perception
>
> **摘要:** 4D radar is increasingly attractive for robotic mapping because it provides range, azimuth, elevation, and Doppler measurements while remaining robust in adverse visibility conditions. Although recent radar and radar--inertial odometry methods have achieved promising online state estimation performance, offline global map refinement for 4D radar remains underexplored. This paper presents RAMBA, a radar bundle-adjustment framework for globally consistent 4D radar mapping. Given initial poses and radar frames from a radar--inertial odometry front-end, RAMBA jointly refines radar frame states using covariance-weighted geometric residuals, IMU preintegration factors, and radar ego-velocity constraints. The geometric residuals extend pairwise GICP to a multi-frame optimization by forming voxel-based correspondences across selected frames and weighting each residual with point covariances. To improve robustness against drift and revisits, RAMBA enforces temporal consistency during correspondence formation while explicitly supporting loop-closure constraints. Experiments on the ColoRadar and SNAIL Radar datasets show that RAMBA improves map consistency and usually enhances trajectory accuracy over radar--inertial odometry and pose-graph optimization baselines.
>
---
#### [new 019] X-DiffVLA: X-Embodied Diffusion Action Heads for Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉-语言-动作模型任务，旨在解决跨具身策略泛化问题。提出X-DiffVLA模型，通过扩散机制提升不同末端执行器间的知识迁移能力。**

- **链接: [https://arxiv.org/pdf/2605.25044](https://arxiv.org/pdf/2605.25044)**

> **作者:** Boyu Li; Chaoyi Xu; Haoqi Yuan; Xinrun Xu; Börje F. Karlsson; Dongbin Zhao; Haoran Li; Zongqing Lu
>
> **摘要:** Learning universal policies from cross-embodied data remains a fundamental challenge in robotics. Although Vision-Language-Action (VLA) models are pre-trained on large and diverse datasets, they typically rely on embodiment-specific fine-tuning to achieve strong performance in downstream tasks. This requirement severely limits their generalization capability and restricts knowledge transfer across embodiments performing similar tasks. To overcome these limitations, we focus on cross-embodied settings with shared robotic bases and heterogeneous end-effectors, and propose X-DiffVLA, a diffusion-based VLA model featuring a unified cross-embodied action head. X-DiffVLA can leverage the generative strengths of diffusion models to capture both the diversity and latent correlations in cross-embodied datasets. Specifically, we introduce Embodiment Forcing, a classifier-free guidance technique to implicitly steer action generation toward embodiment-specific functional components, capturing fine-grained structural nuances without explicit supervision. In addition, a Morphological Tree Diffusion approach is designed to strengthen behavioral correlations across diverse end-effectors, maximizing the transferability of heterogeneous demonstrations. Experimental results across RoboCasa and Isaac Gym, covering different embodiments from grippers to dexterous hands, show that X-DiffVLA achieves state-of-the-art performance, with improvements of 15.3% and 12.5%, respectively. Real-world evaluations further validate the robustness of the proposed framework and its effectiveness in scalable cross-embodied policy learning.
>
---
#### [new 020] When Search Becomes Memory: Turning Robot Design Trials into Transferable Skills
- **分类: cs.RO; cs.AI; cs.CL; cs.CV**

- **简介: 该论文属于机器人设计任务，旨在解决传统进化算法记忆缺失问题。通过构建可迁移技能库，将设计经验转化为可审计的知识，提升搜索效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.25832](https://arxiv.org/pdf/2605.25832)**

> **作者:** Yunfei Wang; Xiaohao Xu; Yang Li; Xiaonan Huang
>
> **备注:** 20 pages, 8 figures
>
> **摘要:** Large language models (LLMs) are increasingly used as proposal generators for evolutionary robot design, yet most loops remain memoryless: simulator results shape the next population but are not preserved as reusable design knowledge. We present Auto-Robotist, a self-evolving LLM agent that distills morphology-search traces into an explicit natural-language skill library. Each skill stores a structural archetype, evidence-grounded positive and negative rules, and the evaluated designs that support them, making design memory inspectable rather than implicit in a population. During search, the agent retrieves skills to condition LLM edits of elite bodies while retaining a Genetic Algorithm (GA) mutation path for exploration; after evaluation, it updates the library through Add, Diagnose, and Merge. Across seven EvoGym tasks spanning locomotion, traversal, and object interaction, Auto-Robotist improves cold-start 5x5 search and transfers learned skills to 10x10 design spaces, where reference-conditioned transfer outperforms GA on every task. These results suggest that LLM agents can convert expensive physical evaluations into reusable, auditable design principles. Our code will be released upon acceptance.
>
---
#### [new 021] PACT: Proactive Asking for Continual Task Assistance in Human-Robot Collaboration
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机协作任务，解决长期合作中机器人因信息不足导致的辅助效率低问题。提出PACT框架，通过主动询问提升辅助准确性与效率。**

- **链接: [https://arxiv.org/pdf/2605.24350](https://arxiv.org/pdf/2605.24350)**

> **作者:** Chengbo He; Sheng Li; Chenyang Ma; Bochao Zou; Li Sun; Jiansheng Chen; Junliang Xing; Yuanchun Shi; Huimin Ma
>
> **摘要:** Robotic assistants in long-term human-robot collaboration need to assist users under partial observations while leveraging cross-day interaction history. However, human traits and routines are often unknown at the beginning of collaboration, making passive infer-then-act assistance ineffective and inefficient. To address this challenge, we study a cross-day proactive asking setting for continual task assistance and propose PACT (Proactive Asking for Continual Task Assistance), an ask-or-act framework that determines whether clarification should be sought before taking action. PACT leverages current observations together with accumulated interaction history to evaluate contextual sufficiency, enabling the robot to provide more reliable assistance and progressively adapt to the user over time. We implement its primary learned instantiation using reinforcement learning and evaluate alternative instantiations under the same framework. To assess such behavior, we further introduce a clarification utility metric that quantifies the trade-off between assistance accuracy and the frequency of clarification requests. Experiments in multi-day embodied collaboration scenarios demonstrate that, compared with passive inference baselines, PACT consistently improves both assistance accuracy and clarification utility, highlighting the importance of proactive asking in continual human-robot collaboration.
>
---
#### [new 022] Polymander II: an amphibious salamander-inspired robot with contact and flow sensors
- **分类: cs.RO**

- **简介: 该论文属于机器人感知任务，旨在解决 amphibious 机器人在陆水环境中的交互感知问题。通过霍尔传感器实现接触力和水流力的同步检测，提升机器人适应复杂地形的能力。**

- **链接: [https://arxiv.org/pdf/2605.24465](https://arxiv.org/pdf/2605.24465)**

> **作者:** Qiyuan Fu; Sudong Lee; Andrea Grillo; Jonathan Arreguit; Louis Gevers; Josie Hughes; Auke J. Ijspeert
>
> **备注:** This work has been accepted for publication in the 2026 \it{International Conference on Robotics and Automation (ICRA)}, Vienna, Austria
>
> **摘要:** Robots benefit from sensory information to coordinate body movement, gain robustness against perturbations, and transit between different modes to adapt to various terrains. However, few amphibious robots can sense interactions with both terrestrial and aquatic environments. In this paper, we present a solution that uses Hall-effect sensors to sense foot contact forces and lateral hydrodynamic forces on a salamander-inspired amphibious robot. With two bus lines, the robot can simultaneously acquire this exteroceptive information at more than 500 Hz and proprioceptive information, such as joint positions and loads, at 100 Hz. The Hall-effect sensors used are compact, making them suitable for embedding in multiple positions within a robot, and exhibit high sensitivity to small forces. Moreover, because the sensor can be positioned separately from the measured object, waterproofing can be implemented with relative ease. Our tests demonstrate the robot's capabilities in traversing amphibious environments and its potential in using feedback control for more complex locomotion tasks.
>
---
#### [new 023] ParkingWorld: End-to-End Autonomous Parking Reinforcement Learning from Corrective Experience in 3DGS Simulation
- **分类: cs.RO**

- **简介: 该论文属于自主泊车任务，解决传统方法在训练效率和泛化能力上的不足。提出CIL-SERL框架，利用3DGS仿真和多级回放缓冲机制提升泊车性能。**

- **链接: [https://arxiv.org/pdf/2605.25029](https://arxiv.org/pdf/2605.25029)**

> **作者:** Zhengcheng Yu; Changze Li; Haoran Liu; Tong Qin
>
> **备注:** 9 pages(including 1 page of Appendix), 6 figures. Will be submitted to RA-L 2026
>
> **摘要:** Autonomous parking demands precise low-speed maneuvering within narrow, cluttered, and highly constrained environments, where vehicles must navigate tight spaces while avoiding static obstacles and complex geometric boundaries. Unlike imitation learning, which typically requires massive volumes of high-quality expert demonstrations to converge to a stable policy and often suffers from limited generalization to unseen scenarios, traditional reinforcement learning (RL) methods face persistent challenges including excessive training overhead, inefficient exploration, and even failure to learn viable parking strategies in challenging settings. To address these limitations, this paper presents a correction-in-the-loop sample-efficient reinforcement learning (CIL-SERL) framework for end-to-end autonomous parking, which is entirely trained in a photorealistic 3D Gaussian Splatting (3DGS) parking simulator that enables high-fidelity digital reconstruction of real-world scenes. Inspired by error-correction notebooks used in learning practice, we design a novel multi-level replay buffer mechanism. These buffers hierarchically organize and store standard RL rollouts, human corrective interventions, failed exploration trajectories, and rollback-based correction segments in separate yet interconnected memory regions, facilitating structured sampling and targeted learning during training. The proposed framework is systematically evaluated in both the 3DGS simulation environment and a physical vehicle platform. Extensive experimental results demonstrate that our method achieves substantial improvements in parking success rate, operational efficiency, and safety performance across diverse scenarios, validating the effectiveness and practical applicability of the proposed CIL-SERL-based end-to-end autonomous parking solution.
>
---
#### [new 024] Afford-VLA: Action-Aligned Visual Planning via Internalized Affordance
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型在复杂场景中空间推理不足的问题。通过引入内部化可操作性机制，提升视觉规划与动作的对齐性。**

- **链接: [https://arxiv.org/pdf/2605.24203](https://arxiv.org/pdf/2605.24203)**

> **作者:** Runze Wang; Yuqian Fu; Yu Li; Tao Lin; Tianwen Qian; Mohamed Elhoseiny; Bo Zhao; Yanwei Fu; Yu-Gang Jiang; Xiangyang Xue
>
> **备注:** 20 pages
>
> **摘要:** Vision-language-action (VLA) models have shown strong potential for generalist robot manipulation, yet they remain limited by insufficient spatial reasoning, particularly in determining where to interact in complex visual scenes. While recent efforts introduce various forms of visual planning to address this issue, existing approaches either rely on global geometric cues, symbolic intermediate representations, or externally generated visual signals, which are often weakly coupled with downstream action prediction. In this work, we revisit visual planning in VLA systems and argue that effective planning should be local, visually grounded, internally generated, and directly aligned with action. Based on this insight, we propose Afford-VLA, a unified framework that internalizes task-conditioned affordance as an explicit visual planning interface within VLA models. Concretely, we introduce learnable <AFF> tokens to query task-relevant interaction regions, decode affordance masks from multimodal features, and convert them into compact embeddings that directly condition action generation. This design enables affordance to be both generated and utilized within the VLA, forming a tightly coupled perception-action pathway. To further support this integration, we adopt a training strategy that allows the affordance pathway to be jointly optimized with action prediction, improving its effectiveness for downstream control. We evaluate our method on multiple simulation benchmarks, including LIBERO, LIBERO-Plus, and SimplerEnv, achieving consistent state-of-the-art performance, along with strong real-world results. These findings demonstrate that internalizing affordance as action-aligned visual planning provides a powerful paradigm for improving VLA systems.
>
---
#### [new 025] Performance Comparison of Classical and Neural Sampling Algorithms for Robotic Navigation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人路径规划任务，比较经典与神经网络采样算法的性能，旨在提升导航效率与轨迹平滑度。**

- **链接: [https://arxiv.org/pdf/2605.25010](https://arxiv.org/pdf/2605.25010)**

> **作者:** Hichem Cheriet; Badra Khellat Kihel; Samira Chouraqui
>
> **摘要:** Integrating artificial intelligence (AI) into sampling-based motion planning provides new possibilities for improving autonomous navigation efficiency. In this paper, three algorithms, namely RRT*, Neural RRT*, and Neural Informed RRT*, are implemented and evaluated on environments containing convex and concave obstacles with different obstacle densities. The obtained results indicate that neural-guided planners improve path quality, producing up to 14\% shorter paths and 55--75\% smoother trajectories compared with the conventional RRT* algorithm. Among the evaluated methods, Neural Informed RRT* achieves the best overall performance in terms of path length and trajectory smoothness. These results demonstrate the effectiveness of AI-guided sampling strategies for improving reliability and trajectory efficiency in robotic and UAV navigation, despite a slight increase in computation time. Overall, the study highlights the growing importance of artificial intelligence in real-time robotic path planning applications.
>
---
#### [new 026] Action-Prior Denoising for Smooth Real-Time Chunking
- **分类: cs.RO**

- **简介: 该论文属于实时分块任务，解决训练时分块策略的约束不足问题。提出Soft RTC，通过动作先验去噪提升分块策略的平滑性和稳定性。**

- **链接: [https://arxiv.org/pdf/2605.25537](https://arxiv.org/pdf/2605.25537)**

> **作者:** Dongyang Liu; Zhaowen Zheng; Yu Sun; Longxu Zhang; Yixuan Liu; Hao Wan
>
> **备注:** 7 pages, 5 figures, 1 table
>
> **摘要:** Real-time chunking (RTC) lets chunked action policies operate under inference delay by conditioning a newly generated action chunk on actions already committed by the previous chunk. Training-time RTC simulates this delay during learning and avoids expensive guidance at deployment, but its binary prefix mask treats all non-prefix tokens as fully unconstrained. This under-models asynchronous execution: early overlap actions are fixed, while later overlap actions remain editable but should still stay close to the previous plan. We propose Soft RTC, a training-time RTC generalization based on action-prior denoising. Soft RTC constructs corrupted overlap tokens from partially denoised states instead of pure noise and injects the aligned previous chunk as the same prior during inference through a lightweight token-wise blending rule. On the 12 released large Kinetix levels, a short soft window nearly matches hard training-time RTC in overall solve rate (0.809 vs. 0.815), while a medium window reduces high-delay action delta and jerk by 9.1% and 9.6% relative to hard RTC. Both variants keep near-naive runtime, unlike inference-time RTC baselines. A small preliminary real-robot sorting study provides additional evidence that training-time RTC can improve completion and that Soft RTC gives the lowest commanded-action finite-difference metrics among the tested policies.
>
---
#### [new 027] Micro-Swarm Locomotion Optimization in Dynamic Flow using Multi-Objective Multi-Agent Reinforcement Learning
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于微机器人集群运动优化任务，解决动态流中多目标控制问题。通过结合CFD与多智能体强化学习，实现高效、平稳的上游移动。**

- **链接: [https://arxiv.org/pdf/2605.25025](https://arxiv.org/pdf/2605.25025)**

> **作者:** Josef Berman; Oren Gal
>
> **摘要:** Coordinating micro-robotic swarms in physiologically realistic, time-dependent fluid environments remains an unsolved challenge for biomedical and environmental applications. We present a hybrid Computational Fluid Dynamics - Multi-Objective Multi-Agent Reinforcement Learning framework that directly couples a high-fidelity incompressible Navier-Stokes solver with decentralized proximal policy optimization to learn physically consistent swarm control strategies in oscillatory flow. Sixteen magnetically actuated micro-robots navigate a pulsatile arterial waveform, simultaneously optimizing upstream progression, energy conservation, and motion smoothness, reconciled using PCGrad surgery. Without PCGrad, energy efficiency and smoothness rewards collapse to near zero within 10,000 training steps while progress exhibits persistent large-amplitude oscillations, confirming that gradient conflict resolution is a structural requirement rather than an optional refinement in this domain. The converged policy achieves a progress reward of 6.5-7.0, a sustained energy efficiency of 0.63-0.65, and near-maximum smoothness (0.97-0.99), representing improvements over brute-force baselines on the primary objective while both baselines yield negative energy efficiency throughout. Training reveals three emergent behavioral phases: a collective two-layer hydrodynamic throttling formation that suppresses peak channel velocities during forward flow, a cycle-synchronized ratchet mechanism that exploits flow reversals for upstream repositioning, and an individualized final approach as agents near the success boundary. These results establish that time-dependent fluid-agent interactions can be captured directly within multi-objective reinforcement learning loops, offering a physically grounded paradigm for micro-swarm control in biomedical navigation, environmental monitoring, and industrial microfluidics.
>
---
#### [new 028] IsaacIPC: Coupling High-Fidelity Simulation and Realistic Rendering for Contact-Rich Robotic Systems
- **分类: cs.RO**

- **简介: 该论文提出IsaacIPC，用于高保真机器人系统仿真与渲染，解决接触模拟与触觉感知问题，结合GPU加速和几何方法提升真实感。**

- **链接: [https://arxiv.org/pdf/2605.24339](https://arxiv.org/pdf/2605.24339)**

> **作者:** Qixin Liang; Zhongqing Han
>
> **备注:** This is a tech report
>
> **摘要:** We present IsaacIPC, a robotic simulation framework that couples GPU accelerated incremental potential contact (IPC) with IsaacSim/Lab. IsaacIPC maps simulated deformation between simulation and visual meshes, enabling real-time realistic rendering with applications to data collection and policy evaluation. For tactile sensing, we introduce the geometric mortar contact potential (GMCP), which defines a barrier potential over contact samples on tactile surfaces to better resolve contact-pressure distributions. We evaluate GMCP on contact benchmarks and demonstrate IsaacIPC on rigid-deformable robotic simulations including a quadruped robot, a dexterous hand, and a universal manipulation interface (UMI) gripper.
>
---
#### [new 029] Parallel Differentiable Reachability for Learning and Planning with Certified Neural Dynamics and Controllers
- **分类: cs.RO; cs.AI; cs.LG; eess.SY; math.OC**

- **简介: 该论文属于机器人学习与控制领域，解决闭环神经网络系统在不确定性下的安全保证问题。提出一种可并行、可微的可达性框架，用于提升动态模型和控制器的可靠性。**

- **链接: [https://arxiv.org/pdf/2605.25346](https://arxiv.org/pdf/2605.25346)**

> **作者:** Keyi Shen; Glen Chou
>
> **备注:** Robotics: Science and Systems XXII (RSS 2026)
>
> **摘要:** Neural network (NN) dynamics models and control policies achieve strong performance in robotics, but providing sound guarantees under uncertainty remains difficult, especially for closed-loop NN systems. Existing reachability tools provide formal over-approximations, yet are often non-differentiable, overly conservative, or too slow for modern learning and online planning pipelines. To address this, we present a parallelizable, differentiable reachability framework in JAX for continuous- and discrete-time systems with analytical and NN-based dynamics and controllers. Our framework combines Taylor-model flowpipe construction with CROWN-style linear bound propagation through a unified representation that preserves affine dependencies while supporting GPU-batched computation and automatic differentiation. Building on this reachability primitive, we develop (i) a certified training method that encourages reachability-friendly dynamics models and controllers, and (ii) a reachability-aware sampling-based MPC scheme with gradient-based refinement. Experiments on non-prehensile manipulation and quadrotor tasks, including hardware and higher-dimensional evaluations (up to 72D), demonstrate practical online planning while maintaining certified reachable-set over-approximations under bounded uncertainty.
>
---
#### [new 030] EXPO-FT: Sample-Efficient Reinforcement Learning Finetuning for Vision-Language-Action Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人学习任务，旨在解决VLA模型在真实场景中可靠性不足的问题。通过EXPO-FT系统实现高效强化学习微调，提升任务成功率与样本效率。**

- **链接: [https://arxiv.org/pdf/2605.25477](https://arxiv.org/pdf/2605.25477)**

> **作者:** Perry Dong; Kuo-Han Hung; Tian Gao; Dorsa Sadigh; Chelsea Finn
>
> **摘要:** The ability to efficiently and reliably learn new tasks has been a foundational challenge in robotics. Vision-Language-Action (VLA) models have demonstrated strong generalization across diverse manipulation tasks, yet pretrained policies consistently fall short of the reliability required for real-world deployment. Reinforcement learning (RL) fine-tuning offers a promising path to bridge this gap, but existing approaches either train from scratch without fully leveraging pretrained priors, or fine-tune VLAs without achieving the sample efficiency and success rates that practical deployment demands. We present EXPO-FT, a system for stable, sample-efficient RL finetuning of pretrained VLA policies that closes this gap. Our system solves a suite of challenging manipulation tasks, including routing string lights and inserting the plug to light it up, striking a pool ball into a pocket, and inserting a flower into a wine bottle, each requiring combinations of high precision, dynamic actions, and robustness to varied initial states. Our system achieves perfect task performance (30/30 successes) across all evaluated tasks within an average of 19.1 minutes of online robot data, outperforming both prior RL-from-scratch and VLA finetuning approaches. We release an open-source codebase with the aim of facilitating broader adoption of RL finetuning of VLA models in robotics.
>
---
#### [new 031] RepSAM: Bridging Foundation Models to Robotic Vision via Representation-Guided Adaptation
- **分类: cs.RO; cs.CV**

- **简介: 该论文针对机器人视觉中的感知问题，提出RepSAM框架，通过表征引导的微调方法提升基础模型在非结构化环境中的性能。**

- **链接: [https://arxiv.org/pdf/2605.25495](https://arxiv.org/pdf/2605.25495)**

> **作者:** Wenhui Chu
>
> **备注:** Accepted to IJCAI-ECAI 2026 (Special Track on AI and Robotics). 8 pages, 4 figures, 12 tables
>
> **摘要:** Robotic perception in unstructured environments remains challenging despite the zero-shot capabilities of foundation models such as SAM. This work attributes performance degradation to non-uniform representation shifts across transformer layers: shallow layers exhibit substantial domain gaps (CKA < 0.5), whereas deep layers transfer effectively (CKA > 0.7). Based on this observation, we propose RepSAM, a representation-guided parameter-efficient fine-tuning (PEFT) framework for adapting foundation models to robotic vision. RepSAM employs a theoretically grounded CKA-guided rank allocation strategy combined with a multi-modal fusion module for robust handling of challenging robotic scenarios, including transparent objects and cluttered scenes. Experimental evaluation across six benchmarks and robotic manipulation tasks demonstrates that RepSAM achieves 97.9% of full fine-tuning performance (89.0% vs. 90.9% mIoU) while reducing trainable parameters by 158x (from 632M to 4.0M). RepSAM outperforms DoRA by 7.9% mIoU with just 4 hours of training on a single A100 GPU (a 96x reduction from full fine-tuning, which takes 384 GPU-hours). These improvements are statistically significant (p < 0.01) and translate to a 12.0% absolute improvement in robotic manipulation success rates over the LoRA (RGB) baseline.
>
---
#### [new 032] HumanEgo: Zero-Shot Robot Learning from Minutes of Human Egocentric Videos
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **简介: 该论文属于机器人技能学习任务，解决人类演示与机器人执行间的体感差距问题。通过HumanEgo框架，将人类视角视频转化为机器人可执行的策略，实现零样本迁移。**

- **链接: [https://arxiv.org/pdf/2605.24934](https://arxiv.org/pdf/2605.24934)**

> **作者:** Wang; Botao He; Kelin Yu; Seungjae Lee; Ruohan Gao; Furong Huang; Yiannis Aloimonos
>
> **备注:** Project page: this https URL
>
> **摘要:** Human egocentric video captures rich manipulation demonstrations without any robot hardware, yet transferring these skills to robots remains challenging due to the embodiment gap between human and robot in both visual appearance and kinematics. We present HumanEgo, a framework that bridges the embodiment gap by lifting each human demonstration to an entity-level representation of hand-object interaction, and training a flow matching policy with dense auxiliary objectives that amplify supervision from every trajectory. HumanEgo is robot-data-free, hardware-agnostic, data-efficient, and zero-shot human-to-robot transferable. With only 30 minutes of human videos per task, HumanEgo achieves 92.5% average success across four real-world tasks (75% with just 15 minutes), outperforms matched-time robot teleoperation by 41%, and robustly transfers zero-shot across novel robots, cameras, and environments.
>
---
#### [new 033] Soft Pneumatic Actuators for Soft Robotics: A Motion-Based Review of Actuation Mechanisms and Performance Trade-offs
- **分类: cs.RO**

- **简介: 该论文属于软体机器人领域，旨在解决软气动执行器设计与性能评估问题。通过分析不同运动模式的结构特征，探讨其对输出性能的影响，为应用选择提供依据。**

- **链接: [https://arxiv.org/pdf/2605.25109](https://arxiv.org/pdf/2605.25109)**

> **作者:** Mohammed Abboodi
>
> **摘要:** Soft pneumatic actuators are widely used in soft robotics because they can produce large motions while remaining compliant enough to interact safely with objects, environments, and the human body. However, their performance is not solely determined by pressure. Instead, the response depends on the way the actuator is built, including the shape of its chambers, the placement of reinforcements, the use of folds, material stiffness, and the constraints that guide its deformation. As the literature has expanded, it has become more difficult to determine which mechanism is most suitable for a given application and which reported results can be compared across studies. This review examines soft pneumatic actuators according to the design strategies used to generate four motion classes: linear, bending, twisting, and omnidirectional actuation. For each class, it analyzes the structural features that define the deformation path, including braid angle, fold geometry, fiber orientation, chamber arrangement, structural asymmetry, and internal constraint layers. It then discusses how the design choice affect motion output, force generation, air demand, repeatability, durability, fabrication difficulty, and robotic integration. The review further identifies key conditions that must be considered when selecting or comparing actuators, including pressure, loading condition, actuator size, pneumatic supply, and hysteresis This approach helps explain why actuators with similar motion outputs may differ substantially in design requirements, pneumatic demand, and practical suitability. It also highlights the design priorities needed for compact, efficient, repeatable, and deployable soft pneumatic systems in wearable, biomedical, and mobile robotic applications.
>
---
#### [new 034] Path Following Control System of Line-of-Sight Guidance for Robotic Dolphin with Multi-Link Mechanism in Underwater Simulator
- **分类: cs.RO**

- **简介: 该论文属于路径跟踪控制任务，旨在解决仿生水下机器人多关节机构的路径跟随问题。通过仿真设计系统并评估控制方法。**

- **链接: [https://arxiv.org/pdf/2605.25401](https://arxiv.org/pdf/2605.25401)**

> **作者:** Takumi Asada; Takao Oki; Hideo Furuhashi; Kenta Tabata; Renato Miyagusuku; Koichi Ozaki
>
> **摘要:** Biomimetic autonomous underwater vehicle (BAUV) with multi-link mechanism is widely used in aquatic life observation and environmental surveys due to its low power consumption and high maneuverability. An environmental survey requires a path following system that automatically follows specific points. However, the path following system of BAUV is limited, and its evaluation with multi-link mechanism robots has not yet been clarified. The path following system in BAUV requires prior simulation because the model differs depending on the type of biomimetics. In this study, we propose a path following system for BAUVs with a multi-link mechanism and evaluation in underwater simulation. In this result, it was possible to design a path following system suitable for BAUV, determine parameters using a simulator, and evaluate control methods.
>
---
#### [new 035] How to Mitigate the Distribution Shift Problem in Robotics Control: A Robust and Adaptive Approach Based on Offline to Online Imitation Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决模仿学习中的分布偏移问题。通过离线到在线的自适应框架，提升策略的鲁棒性和环境适应能力。**

- **链接: [https://arxiv.org/pdf/2605.25414](https://arxiv.org/pdf/2605.25414)**

> **作者:** Hyung-Suk Yoon; Seung-Woo Seo
>
> **备注:** 8 pages, 2 figures
>
> **摘要:** Distribution shift in imitation learning refers to the problem that the agent cannot plan proper actions for a state that has not been visited during the training. This problem can be largely attributed to the inherently narrow state-action coverage provided by expert demonstrations over the full environment. In this paper, we propose a robust offline to adaptive online imitation learning framework that handles the distribution shift problem in a lifelong, multi-phase scheme. In the offline learning phase, we leverage supplementary demonstrations to broaden the state-action coverage of the policy by utilizing a discriminator to effectively train the policy with supplementary demonstrations, thereby enhancing the robustness of the policy to distribution shift. In the subsequent online inference phase, our framework detects the occurrence of distribution shift and conducts self-supervised imitation learning from online experiences to adapt the policy to the online environments. Through extensive evaluations in MuJoCo environments, we demonstrate that our method exhibits better robustness to distribution shift and better adaptation performance to online environments than the baseline algorithms, which indicates superior performance of our framework against the distribution shift.
>
---
#### [new 036] A Decentralized LiDAR-SLAM System with Certifiably Optimal Pose Graph Optimization
- **分类: cs.RO**

- **简介: 该论文属于多机器人LiDAR-SLAM任务，旨在解决分布式系统中全局一致性问题，通过引入最优位姿图优化算法提升定位精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.25051](https://arxiv.org/pdf/2605.25051)**

> **作者:** Baoshan Song; Feng Huang; Li-Ta Hsu
>
> **备注:** In Proceedings of the IEEE International Conference on Robotics & Automation (ICRA'26) 1st Workshop on Robot Meets GNSS and Ranging for Seamless Autonomy, Vienna, Austria, Jun. 5, 2026
>
> **摘要:** Decentralized multi-robot LiDAR-SLAM is essential for collaborative missions but faces significant challenges in maintaining global consistency. Existing frameworks predominantly rely on local-search optimization or one-time coordinate alignment, which are prone to suboptimal convergence and long-term inconsistency, especially in large-scale or degenerate environments. To address these limitations, this paper presents the first decentralized LiDAR-SLAM system that integrates a state-of-the-art certifiably optimal Pose Graph Optimization (PGO) backend. By leveraging the Riemannian Block Coordinate Descent (RBCD) algorithm, our system ensures globally consistent trajectory estimation without requiring accurate initial guesses. Experimental results demonstrate that the proposed framework achieves superior robustness, improving trajectory RMSE by up to 48.9% compared to the state-of-the-art DiSCo-SLAM.
>
---
#### [new 037] Acting on the Unseen: Communication-Free Collaborative Filtering for Decentralized Multi-Robot Task Allocation
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究去中心化多机器人任务分配问题，解决无通信、无先验知识的协作过滤难题。提出SwarmCF方法，在无需通信情况下实现高效任务分配。**

- **链接: [https://arxiv.org/pdf/2605.25584](https://arxiv.org/pdf/2605.25584)**

> **作者:** Alexander Apartsin; Yigal Meshulam; Yehudit Aperstein
>
> **备注:** 27 pages, 12 figures
>
> **摘要:** Multi-robot task allocation usually assumes some combination of communication, known task models, or a coordinator. We study the opposite extreme, a regime common in practice but overlooked in theory, which we name Zero-Knowledge MRTA (ZK-MRTA): a robot team with no prior knowledge (no task models, not even the latent rank), no communication (no messages, no parameter sharing, no coordinator), and only a partial and privately-noisy view of a public stream of teammates' outcomes. A hidden low-rank structure governs which robot suits which task, and there are far more tasks than rounds, so most (robot, task) pairs are never attempted. Yet each robot can act well on tasks it never attempted, and onboard new tasks, by running online low-rank collaborative filtering over the broadcast (SwarmCF). The advantage over any structure-free learner is categorical, not a constant factor: a structure-free learner is provably at the prior-mean error floor on unseen pairs. We prove a matching per-robot sample complexity ({\Theta}(d) versus {\Theta}(n), in the rank d and the task count n), an anytime (cumulative-reward) separation under task scarcity, and a deterministic condition under which decentralized recovery from the masked broadcast is exact (validated empirically). Experiments quantify the value of the broadcast, a positive scaling law (per-robot unseen-pair skill rises with team size), and the strongest masking-robustness and anytime profile among low-rank methods, recovering most (about 80% on earned skill) of a centralized full-communication ceiling, and holding under capacity-1 contention and in a robotics-grounded sensing instance.
>
---
#### [new 038] Investigating the Effect of a Series Elastic Actuation Retrofit to Black-Box Actuators
- **分类: cs.RO**

- **简介: 该论文属于控制任务，旨在提升黑箱执行器的力控性能。通过引入串联弹性执行器（SEA），改善了非线性问题，提升了力控带宽和精度。**

- **链接: [https://arxiv.org/pdf/2605.24127](https://arxiv.org/pdf/2605.24127)**

> **作者:** Ivan Tregear; Ayhan Aktas; Ferdinando Rodriguez y Baena
>
> **备注:** Related GitHub repo available here: this https URL
>
> **摘要:** In robotic applications, actuators are typically designed to be stiff with minimal backlash to ensure precision and repeatability. However, this limits compliance, leading to potential damage and poor force control in uncertain environments. Series Elastic Actuation (SEA) introduces compliance to enhance disturbance rejection and enable force measurement via Hooke's Law but reduces system bandwidth. A custom Series Elastic (SE) element was retrofitted to a black-box actuator to mitigate non-linearities like backlash and static friction. Integrating the SE element enabled high-fidelity force measurements, improving force control bandwidth and performance. A torsional SE element was designed through Finite Element (FE) analysis, yielding a stiffness of 2155.4 Nm/rad. Open-loop force control bandwidth was measured for the original motor and the SEA-integrated configuration, while closed-loop bandwidth was assessed using feedback from the SEA and a commercial force sensor. The SEA module increased bandwidth from 10.32 Hz to 30.32 Hz, a 2.93X improvement. Additionally, it outperformed the commercial sensor by 7.63% despite costing 25 GBP, a fraction of the price.
>
---
#### [new 039] Implicit Null-space Manifold Generation for Redundant Robotic Systems
- **分类: cs.RO**

- **简介: 该论文研究冗余机器人系统的解空间表示问题。通过构建隐式标量场，捕捉解流形的几何结构，实现对解空间的有效建模与距离计算。**

- **链接: [https://arxiv.org/pdf/2605.25770](https://arxiv.org/pdf/2605.25770)**

> **作者:** Taiki Ishigaki; Teresa Vidal-Calleja; Ko Ayusawa; Eiichi Yoshida
>
> **备注:** Accepted to Robotics: Science and Systems (RSS) 2026
>
> **摘要:** Robotic systems with redundant degrees of freedom can achieve the same task outcome using multiple configurations, resulting in solution sets that form manifolds in the configuration space. Existing approaches typically exploit such redundancy locally through Jacobian-based techniques to compute individual solutions or trajectories. While effective for solution computation, these methods do not retain a representation of the geometry of the solution set itself. In this work, we adopt a representation-centric approach to estimate the geometric structure of the solution space. We consider solution manifolds induced by general task-defining maps and construct an implicit scalar field over the configuration space, whose zero-level set corresponds to the solution manifold. To this end, we generate samples in the neighborhood of the solution manifold using a Jacobian-guided exploration strategy, which efficiently captures its local and global structure. The resulting implicit representation is defined over the configuration space and naturally induces a continuous, distance field that encodes proximity to the solution manifold. Experiments on a planar three-link robot and a seven-degree-of-freedom Franka manipulator demonstrate the effectiveness of the proposed representation. Furthermore, the framework enables consistent modeling of solution spaces across families of tasks with continuous variation.
>
---
#### [new 040] Smoother Action Chunking Flow Policy via Prior-Corrected Orthogonal Trust-Region Guidance
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，解决动作分块带来的连续性问题。提出POTR方法，通过修正权重和约束方向提升控制平滑性与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.24433](https://arxiv.org/pdf/2605.24433)**

> **作者:** Kai Fang; Hailong Pei; Xuemin Chi
>
> **摘要:** Flow-matching robot policies commonly use action-chunking inference for efficient closed-loop control, but chunk boundaries can introduce discontinuous action transitions. Existing RTC guidance improves continuity by injecting correction signals during denoising, yet its weight schedule is weak at intermediate timesteps and its unconstrained correction direction may introduce transverse perturbations. We propose POTR, a **p**rior-corrected **o**rthogonal **t**rust-**r**egion guidance method. First, we incorporate a data-prior scale $\sigma_d$ into the RTC guidance weight, yielding stronger intermediate-time correction. Second, we decompose the guidance vector into components parallel and perpendicular to the denoising velocity, and constrain the perpendicular component within a trust region. On LIBERO with $\pi_{0.5}$, POTR improves success rate and consistently reduces chunk-boundary discontinuity, acceleration, and jerk compared with RTC. Ablations show that the prior-corrected weight provides the main correction gain, while the orthogonal trust region further improves stability.
>
---
#### [new 041] Terrain-Adaptive Grouser Wheel for Optimal Planetary Exploration: Design and Experimental Investigation
- **分类: cs.RO**

- **简介: 该论文属于行星探测任务，旨在解决轮式机器人在复杂地形中的移动问题。通过设计可调节履刺高度的轮子，提升其适应不同地形的能力。**

- **链接: [https://arxiv.org/pdf/2605.24311](https://arxiv.org/pdf/2605.24311)**

> **作者:** Vincent Griffo; Yashwanth Kumar Nakka
>
> **备注:** Under Review
>
> **摘要:** Planetary rovers operating in extraterrestrial environments often encounter significant mobility challenges due to varying terrain features such as gradients and granularity. While recent works in multimodal wheel design have explored adjustments in stiffness, compliance, and diameter as a means to improve terrain adaptability, full wheel grouser-adjustable designs remain largely unexplored. Grousers are a compelling feature to actuate, as granular terrains tend to require increased grouser height for improved wheel performance. As a result, we introduce [Anonymized Robot Name], a multimodal wheel capable of continuously adjusting its grouser height for terrain adaptation. The platform was evaluated across four representative surfaces, including vinyl flooring, coarse rock, pea gravel, and sand under two packing states, spanning a range of granular conditions. Results from 750 experimental trials demonstrate that adaptive deployment reduces slip by 30.0--58.0\% and improves travel time and energy consumption by up to 77.4\% in granular regimes relative to fixed configurations. Using the terrain trial data, a simplified scaling analysis was developed and validated, suggesting a relationship between terrain granularity and optimal grouser height for the tested configuration. No single grouser height minimized slip across all terrains, underscoring the limitations of fixed-wheel systems commonly used for planetary exploration. This observation reinforces the potential of grouser-adaptive morphology, such as [Anonymized Robot Name], as an effective solution for enhancing rover mobility across diverse and mobility-challenging extraterrestrial environments.
>
---
#### [new 042] Convex-Neural RRT*: Fast and Reliable Learning-Guided Sampling for High-Quality Robot Path Planning
- **分类: cs.RO; cs.LG; cs.NE**

- **简介: 该论文属于机器人路径规划任务，解决传统采样算法效率低的问题。通过引入神经网络和凸区域引导，提升路径规划的效率与质量。**

- **链接: [https://arxiv.org/pdf/2605.25006](https://arxiv.org/pdf/2605.25006)**

> **作者:** Hichem Cheriet; Badra Khellat Kihel; Samira Chouraqui; Bara J. Emran
>
> **摘要:** Sampling-based algorithms for robot path planning offer probabilistic completeness and strong empirical convergence properties across environments with diverse obstacle configurations. However, in practice, these methods often require many iterations to obtain high-quality solutions. This paper proposes Convex-Neural RRT*, an enhanced RRT* variant that incorporates neural guidance to predict informative waypoint regions near high-quality paths. Convex candidate regions are extracted from these predictions, enabling the planner to concentrate exploration on geometrically relevant areas while preserving global exploration. The proposed algorithm is evaluated against Neural RRT*, Neural Informed RRT*, classical RRT*, and LTA* across three environment types and 18 benchmark maps. Experimental results show that Convex-Neural RRT* reduces computation time by 30-75% compared to neural-guided variants and up to 88-98% relative to LTA*, while achieving an average path length reduction of approximately 5% compared to classical RRT*, with larger improvements observed in complex environments. The method also maintains an overall success rate above 99% across varying obstacle densities. These findings indicate that convex-guided neural sampling provides an effective balance between computational efficiency and solution quality, supporting its applicability to time-sensitive robotic navigation tasks.
>
---
#### [new 043] Safety-Critical Whole-Body Control for Humanoid Robots via Input-to-State Safe Control Barrier Functions
- **分类: cs.RO**

- **简介: 该论文属于人形机器人安全控制任务，解决复杂环境中动态安全约束问题。提出基于ISSf-CBF的分层控制框架，确保运动安全与动态可行性。**

- **链接: [https://arxiv.org/pdf/2605.25546](https://arxiv.org/pdf/2605.25546)**

> **作者:** Kwanwoo Lee; Sanghyuk Park; Gyeongjae Park; Myeong-Ju Kim; Jaeheung Park
>
> **备注:** 14 pages, 6 figures
>
> **摘要:** Safety-critical control is essential for humanoid robots operating in complex human-centered environments, where physical safety constraints such as joint limits, self-collision avoidance, obstacle avoidance, and workspace boundaries must be satisfied during real-robot operation. However, existing approaches remain limited because kinematic safety guarantees can be degraded in the presence of unknown disturbances, such as model uncertainties, trajectory-tracking errors, and external perturbations. This paper presents a hierarchical safety-critical whole-body control framework for humanoid robots based on input-to-state safe control barrier functions (ISSf-CBFs). The proposed architecture integrates a kinematic-level whole-body controller (KinWBC), an ISSf-CBF safety filter, and a dynamic-level whole-body controller (DynWBC). KinWBC generates nominal joint-motion references from prioritized tasks; the ISSf-CBF filter minimally modifies these references to satisfy kinematic safety constraints under bounded disturbances; and DynWBC tracks the filtered references while enforcing full-body dynamic feasibility and contact stability. Safety constraints are imposed on a whole-body kinematic model, and the ISSf-CBF parameters are conservatively tuned so that the resulting kinematic safety guarantees can be transferred to full-order humanoid dynamics under unknown disturbances. Simulation and real-robot experiments demonstrate that the proposed framework improves safety margins under model mismatch and reliably enforces multiple safety constraints in real time during locomotion, teleoperation, and single-leg balancing with hand control. Project website: this https URL
>
---
#### [new 044] OPAL: Omnidirectional Path-efficient Aerial 3D expLoration
- **分类: cs.RO**

- **简介: 该论文属于自主探索任务，旨在提高机器人在未知环境中的探索效率。提出OPAL框架，通过360度旋转减少计算负担，实现更短的移动距离和更高覆盖率。**

- **链接: [https://arxiv.org/pdf/2605.25423](https://arxiv.org/pdf/2605.25423)**

> **作者:** Yoga Satwik Chappidi; Avideh Zakhor
>
> **备注:** Submitted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Autonomous exploration is critical for robot mapping unknown environments. Desirable characteristics of exploration algorithms include compute efficiency and small traversed distance during the exploration process. Motivated by these, we present Omnidirectional Path-efficient Aerial 3D expLoration (OPAL), an exploration framework centered on deliberate 360-degree yaw rotation at ambiguous branch points rather than compute-heavy global tour planning. We devise multiple variants of OPAL to determine the frontier-selection strategy once the yaw pan is completed. One variant is model-free, while others use large language models (LLMs) or vision-language models (VLMs). We characterize the performance of these variants while varying the vicinity search radius to include frontiers in the selection process. Through simulations we find that although the time-consuming in-place yaw rotation increases total exploration time relative to more computationally complex baselines such as EDEN and FALCON, OPAL is computationally simpler and achieves shorter travel distances and higher coverage-versus-distance area under the curve. We also show that adjusting the frontier-selection search radius enables a tradeoff between travel distance and total exploration time. We verify our results on a Modal AI drone in two indoor environments by comparing OPAL against FALCON, and find that the traveled distance for a variant of OPAL to be as much as 25% lower than FALCON.
>
---
#### [new 045] Extending Embodied Question Answering from Perception to Decision
- **分类: cs.RO**

- **简介: 该论文属于 embodied question answering 任务，旨在解决现有数据集碎片化问题。提出 EQA-Decision 数据集和 RoboDecision 模型，全面评估感知、推理与决策能力。**

- **链接: [https://arxiv.org/pdf/2605.25813](https://arxiv.org/pdf/2605.25813)**

> **作者:** Xicheng Gong; Qiwei Li; Peiran Xu; Yadong Mu
>
> **备注:** 11 pages,4 figures
>
> **摘要:** Embodied Question Answering (EQA) connects perception, reasoning, and interaction within embodied environments. However, existing datasets and benchmarks remain fragmented, each focusing on a limited subset of reasoning skills such as spatial understanding or procedural reasoning, without offering a unified large-scale framework for comprehensive evaluation. We present EQA-Decision, a large-scale embodied QA dataset that systematically covers four complementary dimensions of embodied reasoning: static scene construction, spatial understanding, task dynamics reasoning, and instant decision. The dataset contains over four million question-answer pairs with hierarchical annotations across diverse embodied scenarios. In addition, we develop RoboDecision, a strong baseline model aligned with the EQA-Decision Benchmark, providing a unified framework that jointly evaluates perception, reasoning, and action-level decision-making in embodied environments. Results demonstrate that EQA-Decision effectively benchmarks and enhances VLM capabilities in spatial and interaction reasoning, providing a solid foundation for advancing embodied intelligence research.
>
---
#### [new 046] Decision-Making with Lightweight Confidence-Aware Language Model for Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶决策任务，解决大模型计算开销大、延迟高的问题。提出轻量级信心感知语言模型，通过多代理协作与知识蒸馏，实现高效且可靠的决策生成。**

- **链接: [https://arxiv.org/pdf/2605.25393](https://arxiv.org/pdf/2605.25393)**

> **作者:** Ruoyu Yao; Ruiguo Zhong; Pei Liu; Mingxing Peng; Rui Yang; Jun Ma
>
> **备注:** 8 Pages, 3 figures, ITSC 2026
>
> **摘要:** Large Language Models (LLMs) and Multimodal LLMs (MLLMs) have demonstrated immense potential in autonomous driving (AD) by offering human-like reasoning and open-world generalization. However, the excessive computational overhead and high inference latency of these massive models severely hinder their deployment in resource-constrained AD systems. To address this challenge, we propose a novel decision-making framework utilizing a lightweight confidence-aware language model, which bridges the gap between complex multimodal intention reasoning and efficient inference. Specifically, we design a multi-agent collaborative workflow, comprising action voting, confidence assessment, and summarization agents, to generate high-quality, confidence-annotated decision demonstrations via explicit Chain-of-Thought (CoT) reasoning. These demonstrations are then distilled into a lightweight language model featuring a dual-head architecture, enabling the joint prediction of decision probabilities and the generation of textual rationales. The distillation is realized via a confidence-aware fine-tuning strategy coupled with Retrieval Augmented Generation (RAG) to enhance the model's adaptability and data efficiency. Comprehensive closed-loop experiments on the nuPlan benchmark demonstrate that our approach achieves state-of-the-art (SOTA) success rates in both regular and long-tail scenarios while maintaining low inference latency.
>
---
#### [new 047] RoboHitch: Learning Visual Affordance from Disordered Keypoints for Hitch Knots Tying
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决DLOs knot tying中的拓扑状态跟踪问题。通过学习无序关键点和视觉信息，实现灵活的绳结打结。**

- **链接: [https://arxiv.org/pdf/2605.24394](https://arxiv.org/pdf/2605.24394)**

> **作者:** Jiahui Zuo; Boyang Zhang; Fumin Zhang
>
> **摘要:** Robotic manipulation of deformable linear objects (DLOs) presents significant challenges due to complex dynamics and frequent self-occlusions. Existing robotic knot tying methods typically rely on precise topological state tracking with ordered keypoints and explicit edge connectivity. This reliance makes them prone to failures due to tracking drift and topology mismatch caused by repeated bending and crossings during knot this http URL address these limitations, we introduce RoboHitch, a novel framework that learns to perform hitch knot tying from human demonstrations using only disordered 3D keypoints and RGB images. This eliminates the need for explicit topological order, allowing for more flexible manipulation. Our method employs a dynamic Graph Autoencoder to extract geometric features from untracked keypoints, complemented by a Convolutional Autoencoder that captures essential visual context. A bidirectional cross-attention mechanism then fuses these modalities to jointly predict pick and place affordances, facilitating implicit reasoning about the rope's state and enabling knot tying under this http URL-world experiments demonstrate the effectiveness and generalizability of our approach, successfully completing hitch knots in scenarios with self-occlusions.
>
---
#### [new 048] HumanFlow -- Diffusion-Driven MAV Navigation Among Humans via Tightly-Coupled Motion Tracking, Forecasting, and Control
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决复杂场景下人类运动预测与跟踪问题。提出HumanFlow模型，融合运动跟踪、预测与控制，提升导航安全与效率。**

- **链接: [https://arxiv.org/pdf/2605.25685](https://arxiv.org/pdf/2605.25685)**

> **作者:** Simon Schaefer; Joshua Näf; Stefan Leutenegger
>
> **备注:** Accepted to Robotics Science and Systems (RSS), 2026
>
> **摘要:** Robust and accurate perception of humans in their 3D scene context is essential for integrating robots into everyday environments. Existing approaches, however, often fail to predict plausible and accurate human motion estimates that are consistent with the surrounding scene, especially in the presence of heavy occlusions or partial visibility. This can limit both safety and efficiency for robotic operations. We introduce HumanFlow, a latent diffusion model that unifies human motion tracking and forecasting, conditioned on the 3D scene context. We show that our human motion model produces smooth and accurate predictions under challenging conditions, including heavy occlusions, and outperforms state-of-the-art methods in tracking accuracy while being significantly more efficient. Furthermore, we show how HumanFlow's latent space can be tightly coupled with control by conditioning a flow-matching-based, approximate MPC policy on these representations. We validate our policy in simulation with real human trajectories for MAV social navigation, demonstrating superior navigation performance and remaining collision-free, even under partial observability of the human.
>
---
#### [new 049] Anisotropic Diffusion-Driven Ergodic Coverage in Multi-Robot Systems
- **分类: cs.RO**

- **简介: 该论文属于多机器人系统任务，解决 ergodic 覆盖问题。通过引入各向异性扩散方法，优化潜在场，提升覆盖效率。**

- **链接: [https://arxiv.org/pdf/2605.24125](https://arxiv.org/pdf/2605.24125)**

> **作者:** Thales C. Silva; Anoop Kiran; Nora Ayanian
>
> **摘要:** We consider the problem of combining potential field and ergodic search on multi-robot systems. Traditional ergodic search algorithms use metrics for ergodicity that account for the desired distribution at different scales. Recently, a heat equation-driven ergodic approach was proposed, which adds flexibility to the smoothing of the ergodic metric. However, such an approach, as it is an isotropic diffusion, propagates the error uniformly in all directions, regardless of changes in the desired distribution. We introduce a general class of anisotropic diffusion formulation of the ergodicity problem, which generates a potential field for the ergodic search. We demonstrate that this approach generalizes previous results, which consider radial basis functions and the solution of the heat equation to represent the difference between the goal density distribution and the covered trajectories. In our solution, the agent movement is directed using the gradient of the solution of the Perona-Malik diffusion, and our formulation includes the heat equation as a special case. We demonstrate the methodology with a series of simulations in different scenarios.
>
---
#### [new 050] G-DRAGON: Geospatial Reasoning and Dynamic Planning for Retrieval-Augmented Outdoor Navigation
- **分类: cs.RO**

- **简介: 该论文提出G-DRAGON框架，解决户外导航中长距离定位与最后百米探索问题，结合语义地图与SLAM实现精准导航。**

- **链接: [https://arxiv.org/pdf/2605.25646](https://arxiv.org/pdf/2605.25646)**

> **作者:** Dongzhihan Wang; Yi Du; Jianan Sun; Yuan Xue; Yingchen Zhang; Bing Xiao; Chen Wang; Liang Xu
>
> **备注:** Accepted by IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Autonomous ground robots operating in large-scale outdoor environments require both robust long-range navigation and fine-grained ''last-mile'' exploration. Current advances in visual-language navigation (VLN) work well at short-range tasks, lacking geospatial grounding for long-distance missions. Some OpenStreetMap (OSM)-based methods relying on cloud-based Large Language Models (LLMs) are prone to factual hallucination and cannot conduct ''last-mile'' exploration based on human instruction. To address these challenges, we present G-DRAGON, a retrieval-augmented framework for outdoor, open-world navigation. This framework maps natural-language commands to versioned, local OSM entities via generative retrieval based on lightweight LLM, yielding accurate coordinates for global route planning. A high-level planning module bridges global topological routes with the SLAM system, projecting geospatial waypoints into the robot's navigable frame. For the ''last mile," the framework transitions to frontier-based exploration and open-set semantic voxel mapping to localize open-vocabulary targets. Experimental results in simulation demonstrate our framework outperforms state-of-the-art baselines. Furthermore, we validate the system in unseen real-world urban environments on an Unmanned Ground Vehicle (UGV), successfully completing person-search missions with trajectories of up to 500m.
>
---
#### [new 051] ARCANE-PedSynth: Synthetic Multi-Pedestrian Datasets with Behavioural Crossing Annotations
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出ARCANE-PedSynth框架，用于生成带行为标注的多行人数据集，解决自动驾驶中行人过街预测问题。通过混合控制架构提升过街率，并生成多模态数据。**

- **链接: [https://arxiv.org/pdf/2605.24950](https://arxiv.org/pdf/2605.24950)**

> **作者:** Muhammad Naveed Riaz; Maciej Wielgosz; Antonio M. López Peña
>
> **摘要:** We present ARCANE-PedSynth, an open-source CARLA-based software framework for generating synthetic multi-pedestrian datasets with dense behavioural annotations for pedestrian crossing prediction in autonomous driving. The framework overcomes CARLA's native 9% crossing rate through a hybrid AI-manual pedestrian control architecture, enabling configurable target rates up to 75%. A 12-state behavioural finite state machine with five character archetypes produces diverse crossing behaviours. The framework generates synchronised RGB, LiDAR, and DVS data with per-frame crossing labels, behavioural states, and estimated 2D pose keypoints. We demonstrate ARCANE-PedSynth through PedSynth++, an example dataset generated with the framework, comprising 533 multi-pedestrian clips across 12 weather conditions with RGB, LiDAR, and DVS streams. ARCANE-PedSynth is fully reproducible via CLI parameterisation and Docker containerisation.
>
---
#### [new 052] RED: Adaptive Real-Time DAG Scheduling for Robotic Inference under Environmental Dynamics
- **分类: cs.RO; cs.SE; eess.SY**

- **简介: 该论文提出RED框架，解决动态环境下机器人多任务推理的实时调度问题。通过自适应调度和MIMONet优化，提升性能与兼容性。**

- **链接: [https://arxiv.org/pdf/2605.24044](https://arxiv.org/pdf/2605.24044)**

> **作者:** Zexin Li; Tao Ren; Johnathan Liu; Xiaoxi He; Cong Liu
>
> **备注:** Extension version of RTSS'23
>
> **摘要:** Robots deployed in dynamic environments must contend with environment-driven changes that reshape computation at runtime: new tasks may appear, precedence relations can shift, and overall workload structure evolves, all of which degrade performance, especially when multi-task inference is required under tight resource and real-time budgets. We present RED, a real-time scheduling framework for multi-task deep neural network workloads on resource-constrained robotic platforms that adapts to Robotic Environmental Dynamics (RED) while preserving end-to-end timing guarantees under modeling assumptions. The core of RED is a deadline-aware scheduler that assigns intermediate sub-deadlines, allowing it to accommodate evolving computation graphs and asynchronous inference induced by unpredictable conditions. The framework also supports flexible deployment of MIMONet (multi-input multi-output neural networks), commonly used in multi-tasking robots to alleviate memory pressure through weight sharing. RED explicitly leverages this shared-parameter property via a workload refinement and graph-reconstruction procedure that aligns MIMONet structure with schedulability requirements, improving compatibility and efficiency. We implement RED on NVIDIA Jetson family platforms and on an Apple M-series MacBook and evaluate it on navigation-oriented workloads representative of real robotic scenarios. Experiments show consistent gains over existing methods in throughput, deadline satisfaction, robustness to interference, adaptability, and runtime overhead.
>
---
#### [new 053] Bridging the Gap: Enabling Soft Actor Critic for High Performance Legged Locomotion
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人运动控制任务，旨在解决SAC在大规模并行训练中性能不如PPO的问题。通过改进策略初始化、目标估计等方法，使SAC达到与PPO相当的性能。**

- **链接: [https://arxiv.org/pdf/2605.24975](https://arxiv.org/pdf/2605.24975)**

> **作者:** Gianluca Sabatini; Chenhao Li; Marco Hutter
>
> **摘要:** Proximal Policy Optimization (PPO) has become the de facto standard for training legged robots, thanks to its robustness and scalability in massively parallel simulation environments like IsaacLab. However, its on-policy nature makes it inherently sample-inefficient, preventing its use for continuous adaptation and fine-tuning on real hardware. Soft Actor-Critic (SAC), by contrast, is an off-policy algorithm that can reuse past experience, making it a natural candidate for sim-to-real transfer workflows where the same algorithm can be used both in simulation and for online learning on the real robot. Despite these advantages, SAC has consistently failed to match PPO's empirical performance in massively parallel training settings. This work identifies the root causes of this gap and introduces targeted modifications, covering policy initialization, timeout-aware critic targets, and multi-step return estimation, that enable SAC to train stably at scale. Evaluated across multiple legged robot platforms and diverse locomotion tasks, our approach closes the performance gap with PPO entirely.
>
---
#### [new 054] Elevator-LIO: Robust LiDAR-Inertial Odometry for Multi-Floor Navigation under Elevator-Induced Non-Inertial Motion
- **分类: cs.RO**

- **简介: 该论文属于机器人定位任务，解决电梯中非惯性运动下的定位问题。提出Elevator-LIO框架，分离机器人与电梯运动，实现连续稳定定位。**

- **链接: [https://arxiv.org/pdf/2605.24495](https://arxiv.org/pdf/2605.24495)**

> **作者:** Yifan Zhang; Yudong Huang; Yuchong Zhang; Changze Li; Haoran Liu; Ming Yang; Tong Qin
>
> **备注:** 16 pages, 10 figures, 5 tables
>
> **摘要:** This paper presents Elevator-LIO, a LiDAR-inertial odometry framework designed to achieve continuous robot localization during elevator travel, thereby supporting cross-floor robotic tasks. To address the state-estimation problem in non-inertial frames, Elevator-LIO establishes a decoupled state-estimation model that separately models the robot motion relative to the elevator and the elevator motion itself, and embeds it into a mode-dependent iterated error-state Kalman filter framework. This framework degenerates to conventional LIO estimation in ordinary indoor environments, while enabling the propagation and constrained update of elevator-related states in elevator non-inertial environments, thereby achieving continuous and stable localization. An elevator mode manager detects elevator entry and exit events using LiDAR ranging statistics and estimated states, and introduces event-triggered zero-velocity and zero-acceleration updates when the elevator stops to suppress accumulated vertical drift. In addition, this paper adopts an adaptive voxel downsampling strategy to maintain a stable number of effective points under significant environmental scale changes. We conduct extensive experiments on 20 real-world sequences containing 79 elevator rides, including practical challenges such as large-scale spaces, long vertical travel, dynamic pedestrian interference, and mirror reflections. The results show that Elevator-LIO maintains continuous localization accuracy in all sequences, with terminal height error below 1 cm in 17 sequences. In contrast, existing representative localization systems perform poorly on these elevator sequences. Tests on the Hilti 2022/2023 datasets further show that the proposed method remains competitive in standard indoor scenarios. The project page is available at this https URL.
>
---
#### [new 055] Geometric Workspace Analysis and Transmission-Aware Dynamics of a Serial Spherical Tool for Microsurgery
- **分类: cs.RO**

- **简介: 该论文属于微手术机器人设计任务，解决运动学与动力学分析问题，提出工作空间解析方法和传动动力学评估框架，提升手术工具设计精度与实用性。**

- **链接: [https://arxiv.org/pdf/2605.24760](https://arxiv.org/pdf/2605.24760)**

> **作者:** Anestis Mablekos-Alexiou; Lyndon da Cruz; Christos Bergeles
>
> **摘要:** We present a kinematic and transmission-aware design framework for a serial spherical mechanism with an additional translational degree of freedom for microsurgery. The first contribution is an analytical workspace formulation that provides geometric insight into reachable motion and enables rapid selection of rotation axis orientations without numerical optimization. The second contribution is a dynamics-informed methodology for mechanisms driven by self-locking transmissions, supporting evaluation of torque requirements for a prescribed workspace geometry. The framework is accompanied by an open-source software package for friction identification and inverse dynamics analysis. Experiments on a purpose-built robotic tool for vitreoretinal surgery validate the predictive capability of the models and demonstrate their practical utility for engineering design.
>
---
#### [new 056] Manifold-Constrained MPPI: Real-Time Sampling-Based Control Under Hard Constraints
- **分类: cs.RO; eess.SY**

- **简介: 该论文提出MC-MPPI，解决机器人控制中硬约束满足问题，通过降维与QP优化实现高效实时控制。**

- **链接: [https://arxiv.org/pdf/2605.24813](https://arxiv.org/pdf/2605.24813)**

> **作者:** Seulchan Lee; Sanghyun Kim
>
> **备注:** International Journal of Control, Automation, and Systems
>
> **摘要:** Sampling-based model predictive control methods, such as Model Predictive Path Integral (MPPI), offer derivative-free optimization and robustness in complex robotic systems. However, standard MPPI relies on cost-based soft penalties that cannot guarantee hard-constraint satisfaction, severely limiting its applicability to highly constrained tasks such as closed-chain manipulation. To address this, we propose Manifold-Constrained MPPI (MC-MPPI), a real-time sampling-based control framework that enforces manifold-based equality constraints while preserving the computational advantages of MPPI. The key idea is to decouple the constrained optimal control problem into latent-space planning and execution-level correction. At the planning stage, a Variational Autoencoder (VAE) learns a low-dimensional latent representation of the constraint manifold, enabling MPPI to efficiently generate near-feasible candidate trajectories without per-sample modification. Since this reference enables accurate linearization of the equality constraints, an execution-level Quadratic Programming (QP) controller resolves the residual manifold mismatch in a single solve rather than through iterative projection. Experiments on a 14-DoF closed-chain dual-arm system in both simulation and real-world settings demonstrate that MC-MPPI operates stably at 100 Hz, reliably navigates dynamic environments while effectively maintaining hard equality constraints, and significantly outperforms baseline methods in tracking accuracy. Supplementary videos and implementation details are available at this https URL.
>
---
#### [new 057] FusionCore: A 23-State Unscented Kalman Filter for IMU, Wheel Encoder, GPS, and Visual SLAM Fusion in ROS 2
- **分类: cs.RO; eess.SP**

- **简介: 该论文提出FusionCore，用于ROS 2中的多传感器融合定位任务，解决GPS信号丢失时的导航精度问题，通过23状态UKF融合IMU、轮速计、GPS和视觉SLAM数据。**

- **链接: [https://arxiv.org/pdf/2605.25239](https://arxiv.org/pdf/2605.25239)**

> **作者:** Manan Kharwar
>
> **备注:** 8 pages, 4 figures, 2 tables. Source code: this https URL (Apache 2.0)
>
> **摘要:** We present FusionCore, an open-source ROS 2 sensor fusion package that fuses IMU, wheel encoder odometry, GPS, and Visual SLAM pose into a single 100 Hz odometry stream using a 23-state Unscented Kalman Filter (UKF). The 23rd state is an online estimate of the wheel encoder's systematic yaw rate bias, identified through GPS heading cross-covariance and subtracted during GPS blackouts to reduce heading drift in coast mode. FusionCore also estimates gyroscope and accelerometer biases as explicit filter states, handles GPS natively in ECEF without a separate coordinate projection node, applies per-sensor Mahalanobis chi-squared outlier gating calibrated to measurement degrees of freedom, and adapts sensor noise covariance automatically from the innovation sequence. VSLAM pose fusion enables GPS-denied operation with any visual odometry or SLAM system, including automatic recovery from map reinitialization. We evaluate against robot_localization on twelve full-length sequences (55-92 min each) from the NCLT public dataset. FusionCore achieves lower Absolute Trajectory Error (ATE) on ten of twelve sequences, with improvements ranging from 1.2x to 22.2x on winning sequences. The robot_localization UKF diverges numerically on all twelve sequences. FusionCore is available at this https URL under the Apache 2.0 license.
>
---
#### [new 058] TapSampling: Inference-Time Sampling with a Task-Progress-Understanding Verifier for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，旨在提升生成模型在推理阶段的性能。通过引入TapSampling框架，结合动作VAE和任务进度验证器，实现更优的动作选择与采样。**

- **链接: [https://arxiv.org/pdf/2605.25547](https://arxiv.org/pdf/2605.25547)**

> **作者:** Sizhe Zhao; Shengping Zhang; Shuo Yang; Weiyu Zhao; Shuigen Wang; Xiangyang Ji
>
> **备注:** ICML 2026. Project Page: this https URL
>
> **摘要:** Existing embodied control research demonstrates remarkable performance improvements by scaling training data and model size. We instead explore inference-time strategy as an alternative axis. Non-deterministic generative models, such as diffusion and autoregressive models, have been widely adopted in the field of embodied control. However, the single-shot inference paradigm limits their performance. In this paper, we propose \textbf{TapSampling}, a plug-and-play framework for inference-time sampling. First, we introduce an Action-VAE that represents actions in a low-dimensional latent space by mapping policy-generated initial actions into a compressed posterior distribution, from which any number of latent samples can be drawn and decoded into candidate actions that approximate the true action distribution. Second, we formulate action verification as task-progress outcome prediction, using the intrinsic sequential structure of robotic datasets to train a semantically grounded verifier for interpretable action selection. Furthermore, TapSampling is a policy-agnostic framework. Extensive experiments in both simulated and real-world environments demonstrate that our method substantially improves multiple generalist policies without further policy finetuning. Code and models are available at the project page.
>
---
#### [new 059] Dynamic Neural Koopman Distillation for Real-Time Robot Control Using Diffusion Models
- **分类: cs.RO**

- **简介: 该论文属于机器人实时控制任务，解决扩散模型推理延迟问题。通过动态神经Koopman蒸馏方法，将多步推理压缩为单次前向传播，提升效率并保持多样性。**

- **链接: [https://arxiv.org/pdf/2605.24924](https://arxiv.org/pdf/2605.24924)**

> **作者:** Lei Zheng; Peiqi Yu; Zengqi Peng; Changliu Liu; Armin Lederer
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Diffusion models excel at generating diverse and multimodal trajectories for robotic planning, yet their iterative denoising process introduces latency that is incompatible with high-frequency closed-loop control. To address this problem, we propose Dynamic Neural Koopman Distillation, a framework that distills multistep diffusion inference into a single forward pass while retaining the multimodal expressivity of the teacher model. Specifically, we introduce a Factorized Dynamic Koopman layer that models the denoising process through a factorized latent transition with state-dependent modal gains. We evaluate the proposed method on standard D4RL MuJoCo locomotion benchmarks and a physical Kinova manipulator, comparing against one-step baselines. The results show that our method significantly outperforms existing one-step distillation approaches on the reported locomotion tasks, and reduces the inference latency to the millisecond regime compared with the teacher policy. Hardware experiments further demonstrate that our method enables smooth and fast closed-loop execution while maintaining task success and comparable accuracy. A project page is available at this https URL.
>
---
#### [new 060] OASIS: Observation-Action Space Alignment via SE(3) Trajectory Prediction for Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出OASIS，解决机器人操作中观察空间与动作空间不匹配的问题。通过SE(3)轨迹预测对齐中间表示，提升操作成功率和泛化能力。**

- **链接: [https://arxiv.org/pdf/2605.25829](https://arxiv.org/pdf/2605.25829)**

> **作者:** Xinzhe Chen; Sihua Ren; Liqi Huang; Haowen Sun; Mingyang Li; Xingyu Chen; Zeyang Liu; Xuguang Lan
>
> **摘要:** Recent vision-language-action (VLA) models and world action models (WAMs) advance robotic manipulation by enriching intermediate representations with auxiliary spatial features or future visual-state prediction. However, these representations largely remain within the observation space and do not share the rigid-body geometry of the action space, forcing the action decoder to implicitly recover this geometry. We propose OASIS, a visuomotor policy that aligns the intermediate representation with the action space via $SE(3)$ end-effector trajectory prediction. OASIS couples a 3D-aware feature encoder that fuses vision-language and metric-depth features with an $SE(3)$ trajectory predictor that produces a camera-frame end-effector trajectory. Conditioned on the predictor's pose-supervised hidden states, the action decoder generates action chunks consistent with rigid-body motion. Across simulation and real-world experiments, OASIS outperforms VLA and WAM baselines in success rate and out-of-distribution generalization. Our project page is available at this https URL.
>
---
#### [new 061] Stiffness Optimization for Concentrated Bending in Magnetically Actuated Catheters: Maintaining Steerability under Gradient Stiffness
- **分类: cs.RO**

- **简介: 该论文属于软体机器人任务，旨在解决磁驱动导管在推进和弯曲间的刚度矛盾。通过设计梯度刚度结构，实现稳定弯曲与高效推进。**

- **链接: [https://arxiv.org/pdf/2605.25005](https://arxiv.org/pdf/2605.25005)**

> **作者:** Jiewen Tan; Junnan Xue; Shing Shin Cheng; Shuang Song; Erli Lyu; Jiaole Wang
>
> **摘要:** Achieving both efficient pushability (propulsion transmission) and proximally concentrated bending for steerability is challenging for magnetically actuated soft catheters: higher axial/bending stiffness improves force transmission but reduces steerability, whereas lower stiffness enables large, proximally concentrated bending yet increases kinking/buckling risk under compressive push loads. To address this trade-off, we propose a stiffness-optimized multi-segment magnetically actuated catheter (SO-MAC) that integrates a decoupled steering-advancement mechanism with a gradient-stiffness architecture. The SO-MAC concentrates bending about a stable proximal pivot during advancement while the distal section passively self-straightens to transmit propulsion, aided by the optimized stiffness distribution and elastic recovery of the spring backbone against friction-induced kinking/buckling. Over $0{-}180^{\circ}$ combined steering and advancement, the pivot remained stable and the distal tip advanced near-straight toward the target direction. A 1.5 mm-diameter SO-MAC achieved up to $180^{\circ}$ steering with a 3 mm bending radius at its 10 mm tip, with an average shape error of $1.39 \pm 0.56$ mm and a steering-pivot error of $0.35 \pm 0.10$ mm. Visual feedback control in a bronchial phantom further confirmed robust navigation through highly curved, bifurcating paths.
>
---
#### [new 062] MR-LiDAR: A Multi-Resolution Roadside LiDAR Benchmark for Perception Diagnostics and Deployment Guidance
- **分类: cs.RO**

- **简介: 该论文属于道路感知任务，解决LiDAR配置选择问题。通过构建多分辨率LiDAR基准，分析不同配置对感知性能的影响，提供选型指导。**

- **链接: [https://arxiv.org/pdf/2605.24777](https://arxiv.org/pdf/2605.24777)**

> **作者:** Shunlai Cui; Peng Cao; Yuan Zhu; Yongjiang He; Jiacheng Yin; Xiao Huo; Gang Cao; Xiaobo Liu
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** LiDAR model selection is a critical issue in roadside sensing systems, as it directly determines both perception capability and deployment cost. However, the lack of empirical benchmarks for comparing perception performance across different LiDAR configurations has greatly constrained scientific sensor selection and deployment planning. To address this gap, we present MR-LiDAR, a controlled multi-resolution LiDAR benchmark for roadside perception diagnostics. Using 16-, 32-, 80-, and 128-beam LiDARs in identical roadside scenarios, we collect point clouds and ground-truth annotations for diverse traffic participants, including vehicles and vulnerable road users (VRUs), across varying distances. This controlled design isolates intrinsic LiDAR specifications, particularly beam count and beam distribution, as the key variables for precise performance diagnostics. Based on MR-LiDAR, we conduct systematic empirical analyses to examine how beam count, beam distribution, target distance, object category, and vehicle occlusion affect LiDAR perception performance. The results reveal that all of these factors have substantial impacts. In particular, contrary to the common assumption that higher beam counts always yield better perception, we show that an 80-beam LiDAR with optimized beam distribution can match or even outperform a 128-beam LiDAR with uniform beam distribution. In addition, we provide a practical reference guide for LiDAR selection, including target point-count statistics and detection performance comparisons based on two widely used detection algorithms. This work offers a diagnostic benchmark and practical guidance for determining cost-effective LiDAR configurations in roadside perception applications.
>
---
#### [new 063] Enhanced INS/GNSS State Estimation using GNSS-Based Acceleration Measurements
- **分类: cs.RO**

- **简介: 该论文属于导航定位任务，旨在解决INS/GNSS融合在低动态下观测性不足的问题。通过引入GNSS加速度信息提升滤波精度，实验显示定位误差分别降低11.40%和20.74%。**

- **链接: [https://arxiv.org/pdf/2605.24767](https://arxiv.org/pdf/2605.24767)**

> **作者:** Gal Versano; Itzik Klein
>
> **摘要:** Accurate and reliable navigation is essential for autonomous ground vehicle operations. Standard INS/GNSS fusion relies on GNSS position updates, which provide limited observability of orientation and inertial sensor error states, particularly during low-dynamic motion. In this work, we propose utilizing past GNSS measurements alongside a motion model to extract meaningful vehicle acceleration information. This acceleration measurement is then integrated into the INS/GNSS filter to improve its robustness and accuracy. The proposed approach is evaluated on two real-world unmanned ground vehicle datasets collected from different mobile platforms and inertial sensor grades. Results demonstrate consistent positioning accuracy improvements relative to the standard position-aided filter, with mean position root mean square error improvements of 11.40 % and 20.74 % on the two datasets, respectively.
>
---
#### [new 064] Compliant Non-Prehensile Pushing Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人非抓取推操作任务，解决安全交互中的合规控制问题。通过模型预测控制与能量罐滤波，实现安全、稳定的推动物体运动。**

- **链接: [https://arxiv.org/pdf/2605.25672](https://arxiv.org/pdf/2605.25672)**

> **作者:** Francesco Cufino; Mario Selvaggio; Fabio Amadio; Fabio Ruggiero
>
> **摘要:** In this paper, we address the challenge of performing non-prehensile pushing operations with a compliant robotic manipulation system. To ensure safe operations in human-populated environments, robots must comply with external physical interactions and exhibit passive behavior. To achieve this, we extend a state-of-the-art pushing model to integrate it with impedance-controlled robots. We develop a model predictive control framework built upon this model that enables compliant pushing through optimal modulation of the robot's position/velocity set-point, jointly realizing the required pushing force and contact point adaptation to obtain desired object motion. However, external interactions may induce tracking errors, causing a consequent potentially indefinite increase of the pushing force. To prevent this, we integrate an energy tank passivity filter that further modulates the robot velocity set-point to guarantee passivity and avoid uncontrolled energy buildup. The proposed method has been rigorously tested in simulation and validated through experiments on two different robotic systems, demonstrating passive compliance during human-robot interactions and assessing trajectory tracking performance and robustness to variations in the object's physical parameters.
>
---
#### [new 065] AnyScene: Towards Highly Controllable Driving Scene Generation at Anywhere and Beyond
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出AnyScene，用于生成高可控的驾驶场景。解决合成数据生成中控制不足与适用性差的问题，通过占用图生成和多视角视频合成实现灵活场景生成。**

- **链接: [https://arxiv.org/pdf/2605.26113](https://arxiv.org/pdf/2605.26113)**

> **作者:** Haiming Zhang; Junfei Zhou; Feng Jiang; Jingzhong Li; Zhenglong Guo; Penglin Dai; Jifeng Dai; Yan Xie; Benjin Zhu
>
> **备注:** Work in progress. Project page: this https URL
>
> **摘要:** Generating high-fidelity and controllable synthetic data is critical for advancing end-to-end autonomous driving, particularly for addressing the long tail of rare safety-critical scenarios. Existing occupancy-guided methods typically rely on shallow conditioning mechanisms and reference-frame-dependent video synthesis, which limits fine-grained controllability from arbitrary BEV layouts and restricts their applicability for scalable simulation. In this paper, we propose AnyScene, a unified occupancy-centric framework for driving scene generation. AnyScene generates semantic occupancy sequences from BEV layouts through a Spatial-Temporal Occupancy Diffusion Transformer that jointly tokenizes BEV and occupancy features in an autoregressive manner. This design enables precise controllability from cross-dataset and user-defined BEV inputs while naturally supporting long-horizon generation. Building upon the generated occupancy, a Geometry-Grounded View Expansion module treats occupancy as the canonical spatial representation and synthesizes temporally consistent multi-view driving videos in a reference-free and autoregressive fashion, supporting flexible camera configurations at inference time. Extensive experiments demonstrate that AnyScene achieves state-of-the-art performance in both occupancy and video generation. It exhibits strong generalization to unseen and customized layouts, and provides measurable benefits for downstream tasks such as sparse-view 3D reconstruction.
>
---
#### [new 066] MuJoCoUni:Persistent Batched Runtime Primitives for MuJoCo
- **分类: cs.RO**

- **简介: 该论文提出MuJoCoUni，用于在线机器人学习和批量物理评估。解决高吞吐并行执行问题，提供状态环境执行原语，保留上游MuJoCo语义。**

- **链接: [https://arxiv.org/pdf/2605.24922](https://arxiv.org/pdf/2605.24922)**

> **作者:** Yufei Jia; Junzhe Wu
>
> **备注:** Technical report
>
> **摘要:** We present MuJoCoUni, a downstream MuJoCo distribution for online robot learning and batched physics evaluation. Alongside the open-loop batched trajectory generation already provided by upstream this http URL, MuJoCoUni supplies runtime primitives for stateful environment execution. The target workloads need high-throughput parallel execution while retaining upstream CPU MuJoCo semantics for models, sensors, contact, and constraints. Its core object, BatchEnvPool, is a C++/pybind11 executor that owns per-environment mjModel copies, per-thread mjData workers, and an internal thread pool. It provides final-state-only short stepping, sparse reset, reset-lifecycle domain randomization, batched sensor forward evaluation without advancing dynamics, and batched Jacobian and height-field queries. The implementation is confined to the Python binding layer; MuJoCo's solver, contact model, integrator, and core source tree retain upstream semantics. This report describes the BatchEnvPool API, implementation boundary, relationship to rollout, and the validation and benchmark scripts shipped with the open-source mujoco-uni package, which is installed with \texttt{pip install mujoco-uni}.
>
---
#### [new 067] InvariantCloud: A Globally Invariant, Uniquely Indexed Point Cloud Framework for Robust 6-DoF Tactile Pose Tracking
- **分类: cs.RO**

- **简介: 该论文属于6-DoF姿态估计任务，解决 tactile 姿态跟踪中的精度与鲁棒性问题。提出 InvariantCloud 框架，利用全局不变点云实现高精度位姿估计。**

- **链接: [https://arxiv.org/pdf/2605.25216](https://arxiv.org/pdf/2605.25216)**

> **作者:** Pengfei Ye; Yuxiang Ma; Yi Zhou; Wei Chen; Wenzhen Dong; Molong Duan
>
> **摘要:** Recent advances in imitation learning and vision-language models highlight the need for high-fidelity tactile perception, with 6-DoF tactile object pose estimation providing a crucial foundation for precise robotic manipulation. We introduce InvariantCloud, a 6-DoF pose estimation framework that leverages the global invariance of surface marker constellations on vision-based tactile sensors. In contrast to recent approaches, our one-shot globally invariant point cloud registration suppresses cumulative drift and overcomes long-standing limitations in accurately estimating yaw (Z-axis) rotation. Experimental verifications show that InvariantCloud achieves superior yaw tracking accuracy and re-localization repeatability compared to existing benchmarks, demonstrating its precision and robustness in long-sequence manipulation tasks.
>
---
#### [new 068] MASt3R-Nav: WayPixel Navigation in Relative 3D Maps
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出MASt3R-Nav，解决视觉导航问题。通过像素相对连通性构建地图，提升导航精度与能力，适用于多种任务。**

- **链接: [https://arxiv.org/pdf/2605.24111](https://arxiv.org/pdf/2605.24111)**

> **作者:** Vansh Garg; Rohit Jayanti; Krish Pandya; Sarthak Chittawar; Siddharth Tourani; Muhammad Haris Khan; Sourav Garg; Madhava Krishna
>
> **备注:** 2026 IEEE International Conference on Robotics & Automation (ICRA)
>
> **摘要:** Visual navigation ability is strongly tied to its underlying representation of the world. Unlike classical 3D maps that require globally-consistent geometry, image- or object-relative topological graphs almost entirely do away with geometric understanding. But, this comes at the cost of navigation capability, often limiting it to merely teach-and-repeat. In this work, we propose a novel map representation in the form of pixel-relative connectivity, which is geometrically accurate but does not require global geometric consistency. Inspired by recent progress in 3D grounded image matching, we construct a map from an image sequence through inter-image connectivity based on pixel correspondences in the relative 3D coordinate systems of individual image pairs. We then use this pixel-level graph to perform global path planning by approximating and sparsifying intra-image pixel connectivity. Through this, we derive a ''WayPixel Costmap'' representation and train a controller conditioned on it to predict a trajectory rollout. We show that this dense pixel-level costmap based on relative geometry is a more accurate conditioning variable for control prediction than its image- and object-level counterparts. This enables a highly capable navigation system, as validated on four types of navigation tasks in the simulator and through real world demonstrations.
>
---
#### [new 069] Reason--Imagine--Act: Closed-Loop LLM Decision Making with World Models for Autonomous Driving
- **分类: cs.AI; cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于自主驾驶任务，解决LLM决策在动态交通中物理安全性不足的问题。提出RIA框架，结合LLM与世界模型进行在线安全验证，提升决策可靠性。**

- **链接: [https://arxiv.org/pdf/2605.24004](https://arxiv.org/pdf/2605.24004)**

> **作者:** Zhengqi Sun; Yiwen Sun; Boxuan Liu; Tailai Chen; Tianxu Guo; Jiabin Liu
>
> **备注:** Accepted by the 2026 IEEE International Conference on Intelligent Transportation Systems (ITSC 2026). 8 pages, 2 figures
>
> **摘要:** Large language models (LLMs) are promising for autonomous driving, but semantics-only decision policies can yield physically unsafe behavior in dynamic traffic. Existing methods either perform online language reasoning without explicit dynamics verification or use world models mainly in offline pipelines, leaving a gap between semantic intent and physical feasibility at decision time. We propose Reason--Imagine--Act (RIA), a closed-loop framework that couples an LLM reasoner with an action-conditioned world model for online safety verification. At each step, the LLM proposes an action template and candidate sub-actions, the world model performs short-horizon rollouts, and a safety scorer selects the safest executable action with feedback to the next reasoning step. Under a unified CARLA point-goal protocol (1000 episodes), RIA achieves 80.05% route completion, 51.10% arrival rate, and 0.20% collision rate. Under the same closed-loop interface, RIA consistently outperforms training-free baselines, including CARLA TM and MADA, on core closed-loop metrics. For reproducibility, code is available at this https URL.
>
---
#### [new 070] Multi-view Consistent 3D Gaussian Head Avatars 'without' Multi-view Generation
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文属于3D头像生成任务，解决无需多视角数据生成高质量3D头像的问题。工作包括提出MVCHead模型，通过多视角一致性约束直接生成3D高斯分布。**

- **链接: [https://arxiv.org/pdf/2605.25220](https://arxiv.org/pdf/2605.25220)**

> **作者:** Aviral Chharia; Fernando De la Torre
>
> **备注:** CVPR 2026; Project Website: this https URL
>
> **摘要:** High-fidelity 3D Gaussian head avatar generation is critical for applications such as AR/VR, telepresence, and digital humans. Existing methods depend on multi-view datasets, 3D captures, or intermediate 2D view synthesis. In contrast, we learn both conditional and unconditional 3D head models from randomly sampled 2D images alone, without using multi-view data, 3D supervision, or intermediate view generation. We introduce MVCHead, a single-shot state space model that enforces multi-view consistency (MVC) directly in the 3D representation while regressing 3D Gaussians under these constraints. At its core, we propose a Hierarchical State Space (HiSS) block that progressively refines Gaussians from coarse to fine, while capturing long-range dependencies. Within each HiSS block, we modify Mamba's standard unidirectional scan with the proposed Hierarchical Bi-directional State Scan (HiBiSS) that aligns recurrence with the axes along which multi-view inconsistencies are strongest. Finally, we design an SE(3) Multi-view Critic that judges whether a set of self-renders arises from a single underlying 3D configuration, rewarding cross-view pixel alignment without observing real multi-view pairs. MVCHead achieves state-of-the-art perceptual quality, surpasses prior methods in both texture and geometric consistency, and maintains comparable shape consistency. To demonstrate scalability, we release FaceGS-10K, the first large-scale dataset of ready-to-use 3D Gaussian head assets for training and evaluation of 3D head models. Project Page and code: this https URL
>
---
#### [new 071] LRDDv3: High-Resolution Long-Range Drone Detection Dataset with Range Information and Thermal Data
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出LRDDv3数据集，用于长距离无人机检测，解决高分辨率、远距离无人机数据不足的问题，包含RGB和热成像图像。**

- **链接: [https://arxiv.org/pdf/2605.25942](https://arxiv.org/pdf/2605.25942)**

> **作者:** Knut Peterson; Zaid Mayers; Azmain Yousuf; Priontu Chowdhury; Asher Zaczepinski; Solmaz Arezoomandan; Reihaneh Maarefdoust; David Han
>
> **备注:** 8 pages, 5 figures. Accepted to the 2026 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) have quickly become common in various airspaces, representing a wide range of applications from recreation flying to commercial photography and package delivery. With the increasing prevalence of UAVs, it becomes critical that both manned and unmanned aircraft can detect UAVs and other flying objects from long range to effectively track movement and ensure safe operation in shared spaces. While several datasets have been introduced for drone detection, the need for expanded high-quality data persists, especially in the area of high-resolution long-range drone data. To address this, we introduce a high-resolution dataset of 102,532 long-range RGB images of drones, sampled at 5 FPS from 128 distinct video clips taken mid flight during 17 different data collection days spread over 8 months to ensure a wide variety of lighting scenarios, flight locations, and background elements. The dataset boasts comprehensive drone range information across the dataset, as well as 29,630 IR images, all paired with RGB counterparts from the base dataset. As one of the first drone detection datasets to leverage 4K image resolution and paired 640x512 IR images, our work represents a significant advancement to enable the detection of drones at long range. For access to the complete dataset, please visit this https URL
>
---
#### [new 072] Neuromorphic LiDAR-based Bird's Eye View Object Detection using Energy-efficient Spiking Neural Networks
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于自动驾驶感知任务，旨在解决LiDAR点云的高效目标检测问题。提出一种基于脉冲神经网络的端到端编码解码架构，实现高精度与低能耗的鸟瞰图目标检测。**

- **链接: [https://arxiv.org/pdf/2605.25293](https://arxiv.org/pdf/2605.25293)**

> **作者:** Sambit Mohapatra; Senthil Yogamani; Heinrich Gotzig; Patrick Mader
>
> **摘要:** Autonomous driving perception demands accurate and efficient processing of three-dimensional sensor data under strict power constraints. Traditional convolutional neural networks achieve strong detection accuracy but are computationally intensive, limiting their suitability for deployment on resource-constrained neuromorphic platforms. Spiking neural networks offer a compelling alternative through event-driven sparse computation, yet their application to complex real-world perception tasks such as three-dimensional object detection remains limited. In this work, we propose an end-to-end spiking encoder-decoder network for object detection in bird's eye view representations of LiDAR point clouds, trained using surrogate gradient backpropagation. We train two variants: a membrane potential variant that reads continuous neuron state at the output stage for maximum accuracy, achieving $92.05$/$87.04$/$86.51$ AP at $\mathrm{IoU}\!=\!0.5$ (Easy/Moderate/Hard), and, a fully binary spiking variant that operates exclusively on spike trains at every layer for direct neuromorphic deployment. We evaluate four input spike encoding strategies and demonstrate that allowing the network to learn spike representations directly from data outperforms hand-crafted Poisson, latency, and z-axis encoding schemes on the KITTI benchmark, where sequential frames are unavailable and the BEV input is presented repeatedly across timesteps as a proxy for temporal streaming. A block-wise energy analysis demonstrates a $3.33\times$ reduction in synaptic operation energy over an equivalent CNN under conservative loop-based operation. Together, these results demonstrate the viability of spiking neural networks for accurate and energy-efficient neuromorphic perception in autonomous driving.
>
---
#### [new 073] SEIDM: A Safe and Efficient Intelligent Driver Model for Autonomous Driving Behavior
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于自动驾驶行为建模任务，旨在提升交通流效率与安全性。针对IDM模型保守性过强的问题，提出SEIDM，通过自适应安全因子优化决策，提高跟车效率。**

- **链接: [https://arxiv.org/pdf/2605.23915](https://arxiv.org/pdf/2605.23915)**

> **作者:** Yuyang Yao; Shaocheng Luo
>
> **备注:** To appear in IEEE IV 2026
>
> **摘要:** The Intelligent Driver Model (IDM) is a cornerstone of Adaptive Cruise Control (ACC), valued for its interpretable parameters and effectiveness in car-following behavior modeling. However, its inherent conservatism leads to prolonged stabilization and reduced traffic efficiency, which have received limited attention. In this paper, we propose SEIDM (Safe and Efficient Intelligent Driver Model), an enhanced IDM extension designed to improve traffic flow efficiency without sacrificing safety. SEIDM introduces an adaptive safety factor to dynamically modulate the impact of the safe deceleration term in acceleration decisions. This allows vehicles to follow more assertively under safe conditions while behaving more cautiously in potential hazards. Extensive urban traffic simulations show that SEIDM achieves significantly shorter stabilization spacing and faster convergence to traffic flow equilibrium, outperforming the original IDM and its variants in traffic stability and efficiency.
>
---
#### [new 074] Understanding the Impact of Geometric Foundation Models on Vision-Language-Action Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型任务，旨在解决几何理解不足的问题，通过实验分析不同架构和设计选择对模型性能的影响。**

- **链接: [https://arxiv.org/pdf/2605.24642](https://arxiv.org/pdf/2605.24642)**

> **作者:** Yurou Yang; Muyuan Lin; Roberto Martin-Martin; Martin Labrie; Shreekant Gayaka; Cheng-Hao Kuo; Luca Carlone
>
> **摘要:** Recent work explores new opportunities at the intersection of vision-language-action models (VLAs) and geometric foundation models (GFMs) for 3D reconstruction, such as VGGT. While the resulting geometric VLAs often show improved performance, it remains unclear (i) if modern VLAs already have sufficient geometric understanding to start with, (ii) what is the best architecture to inject geometric understanding into a VLA, and (iii) what is the effect of other design choices that affect geometric VLAs. In this paper we provide a rigorous experimental analysis to shed light on these questions, for a specific choice of VLA (GR00T-N1.5) and GFM (VGGT). Our first contribution is to formalize prior work's intuition that current VLAs lack geometric understanding, by providing a rigorous analysis based on linear probing. The analysis quantifies, for the first time, the "geometric gap" between VLAs and GFMs. Our second contribution is to identify and compare different strategies to bridge GFMs with VLAs. We implement three different architectures, which differ in the way they inject geometry in the VLA, while keeping low-level implementation details as similar as possible, to ensure a fair comparison. Finally, we analyze the impact of non-architectural choices (e.g., training data, number of cameras, reconstruction quality) on the performance of the geometric VLAs.
>
---
#### [new 075] Why We Need World Models for AGI: Where LLMs Fail and How World Models May Outperform
- **分类: cs.AI; cs.CL; cs.RO**

- **简介: 该论文探讨AGI发展中世界模型的必要性，针对LLMs在因果推理和长期规划上的不足，提出Latent Dynamics Inference概念，并通过Flux环境验证世界模型的优势。**

- **链接: [https://arxiv.org/pdf/2605.23972](https://arxiv.org/pdf/2605.23972)**

> **作者:** Feisal Alaswad; Batoul Aljaddouh; Maher Alrahhal; Poovammal E; Talal Bonny
>
> **备注:** 19 pages, 5 figures
>
> **摘要:** Large language models achieve strong performance in language generation and knowledge-intensive tasks, yet remain limited in settings requiring causal reasoning, persistent state tracking, and long-horizon planning. We argue that these limitations may arise from an objective-level mismatch between sequence prediction and reasoning over latent environment dynamics. To formalize this distinction, we introduce Latent Dynamics Inference (LDI), a conceptual perspective that interprets language and multimodal observations as partial evidence of underlying transition dynamics. To empirically investigate this perspective, we introduce Flux, a sequential reasoning environment specified entirely through natural-language rules. As a proof-of-concept case study, the rules are first compiled into an explicit state-transition simulator, illustrating that structured latent transition dynamics can, in some cases, be operationally extracted from textual rule descriptions. This enables a controlled comparison between the LLMs operating purely over textual observations and reinforcement-learning agents trained directly within the extracted latent state space. Within this case study, agents operating with explicit access to the latent state space exhibit substantially more stable behavior in long-horizon gameplay, achieving an aggregate win rate of approximately 79% versus 11% for LLMs. Qualitative analysis further reveals failure modes consistent with unstable persistent state tracking, including invalid actions, state-tracking errors, and short-horizon reasoning failures. The complete implementation of the Flux environment available at this https URL Within the evaluated setting, these results suggest that strong sequence prediction alone may struggle to support robust long-horizon dynamic reasoning without mechanisms for persistent state tracking and transition modeling
>
---
#### [new 076] UWM-JEPA: Predictive World Models That Imagine in Belief Space
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **简介: 该论文提出UWM-JEPA，解决部分可观测环境下的世界建模问题，通过密度矩阵潜空间和单元预测器提高未来模拟准确性。**

- **链接: [https://arxiv.org/pdf/2605.25313](https://arxiv.org/pdf/2605.25313)**

> **作者:** Santosh Kumar Radha; Oktay Goktas
>
> **备注:** 14 pages, 6 figures, 7 tables. Code and data: this https URL
>
> **摘要:** World models for partially observed environments must imagine multiple compatible hidden futures and steer between them under counterfactual actions. Joint Embedding Predictive Architectures (JEPAs) do this in latent space, but a vector-valued latent has no internal structure for carrying the belief over hidden continuations through blind rollout. We introduce the Unitary World Model JEPA (UWM-JEPA), a JEPA world model with a density-matrix latent on a joint system-environment space and a learned unitary predictor. The construction preserves the joint-state spectrum exactly during rollout, so the predictor itself cannot dissipate the represented uncertainty. On a hidden-velocity indicator task requiring five-step forward simulation under a given action sequence with the target observation masked, UWM-JEPA reaches 0.77 accuracy and degrades monotonically as actions are perturbed; a parameter-matched LSTM-JEPA trained under the same counterfactual-target objective and action head collapses to majority-class accuracy (0.53) under every action condition. Under blind rollout, UWM-JEPA loses fewer than ten points of probe R^2 at short horizons while vector-latent baselines lose forty-one and sixty-eight; both nevertheless tie on a held-out context probe, locating the separation in the predictor rather than the encoder. Action sensitivity itself requires training against counterfactual rather than teacher-forced targets, a finding that applies beyond the unitary parameterisation. For JEPA world models to imagine under partial observability, latent geometry and predictor dynamics matter, not frozen context-encoding capacity alone.
>
---
#### [new 077] Drift-Resistant Navigation World Model with Anchored Epipolar Guidance
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于导航世界模型任务，解决感知漂移和几何漂移问题。通过锚点引导的滚动预测，提升长期视觉质量和几何一致性。**

- **链接: [https://arxiv.org/pdf/2605.24761](https://arxiv.org/pdf/2605.24761)**

> **作者:** Po-Chien Luan; Zimin Xia; Wuyang Li; Yang Gao; Alexandre Alahi
>
> **摘要:** We propose Drift-Resistant Navigation World Model, a generative model that mitigates both perceptual drift and geometric drift in conventional rollout-based navigation world models. Existing methods recursively feed generated content into subsequent steps, causing noise accumulation and degraded predictions, i.e., perceptual drift. Meanwhile, their predictions often deviate from the agent's motion, resulting in geometry drift. We address both types of drift by redesigning world-model prediction as an anchor-guided rollout. Instead of rolling out every frame sequentially, we first predict sparse future anchors that serve as stable long-range targets, and then generate intermediate frames within each chunk conditioned on both past context and future anchors. Importantly, these sparse anchors also provide geometric constraints, supported by bidirectional epipolar geometry, to localize where corresponding content should appear in the intermediate frames. Experiments on four benchmarks demonstrate consistent improvements over strong baselines in long-horizon visual quality, geometric consistency, and multi-view coherence. These gains further translate into improved downstream planning performance under the same planners, highlighting the importance of drift-resistant, geometry-aware prediction for reliable navigation world models.
>
---
#### [new 078] ComPose: A Unified Completion-Pose Framework for Robust Category-Level Object Pose Estimation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于物体姿态估计任务，解决点云不完整导致的鲁棒性不足问题。提出ComPose框架，将形状补全与姿态估计统一，提升精度与效率。**

- **链接: [https://arxiv.org/pdf/2605.25553](https://arxiv.org/pdf/2605.25553)**

> **作者:** Huan Ren; Yihan Chen; Chuxin Wang; Nailong Liu; Wenfei Yang; Tianzhu Zhang
>
> **备注:** Accepted by CVPR 2026 (Oral, Best Paper Award Candidate). Project page is available at this http URL
>
> **摘要:** Category-level object pose estimation aims to predict the pose and size of arbitrary objects in specific categories. Existing methods struggle with the inherent incompleteness of observed point clouds, which limits their ability to capture complete object shapes for robust pose reasoning. While point cloud completion offers a promising solution, naively treating it as a separate preprocessing step for partial observations introduces compounding errors and additional computational overhead, ultimately hindering both accuracy and efficiency. To address these challenges, we propose ComPose, a novel unified framework that tightly integrates shape completion to provide complete geometric cues for enhanced pose estimation. At the core of ComPose is a keypoint-based progressive completion module, which recovers full shape representations by progressively predicting a sparse set of keypoints and their surrounding dense point sets, empowering the keypoints to capture holistic object geometries. A geometric relation encoding module further enriches keypoint features with both local and global geometric context. In addition, we introduce a novel geometric relation consistency loss to enforce structural alignment between observed keypoints and their predicted NOCS coordinates, ensuring globally coherent coordinate transformations. Extensive experiments on standard benchmarks demonstrate that our method outperforms state-of-the-art approaches without relying on category-level shape priors.
>
---
#### [new 079] AgentGrounder: Zero-Shot 3D Visual Pointcloud Grounding using Multimodal Language Models
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D视觉定位任务，解决零样本下基于自然语言描述定位3D场景中物体的问题。提出AgentGrounder框架，直接处理点云数据，提升定位精度与效率。**

- **链接: [https://arxiv.org/pdf/2605.25901](https://arxiv.org/pdf/2605.25901)**

> **作者:** Cuong Huynh; Maxim Popov; Denis Gridusov; Sergey Kolyubin
>
> **备注:** Code: this https URL
>
> **摘要:** 3D Visual Grounding (3DVG) is an essential capability for embodied AI, requiring agents to localize objects in 3D scenes based on natural language descriptions. Recent zero-shot methods leverage 2D vision-language models (LVLMs). However, they often rely on existing sets of multi-view images and struggle with the limited semantic and spatial details provided by standard 3D segmentation tools. We present $\textbf{AgentGrounder}$, a zero-shot 3D visual grounding framework that operates directly on colored point clouds without task-specific 3D training. Our approach follows a two-stage design: (1) an offline stage that applies 3D model to build an Object Lookup Table (OLT) with instance IDs, semantic labels, 3D bounding boxes; and (2) an online tool-driven agent that decomposes each query, retrieves only relevant candidates from the OLT, performs geometric scoring, and triggers image rendering on demand when additional visual evidence (e.g., color, material, or viewpoint-sensitive cues) is required. Compared with fixed anchor-target matching pipelines, this design reduces cascading matching errors and improves context-window efficiency by avoiding prompts overloaded with irrelevant objects. We evaluate on ScanRefer and Nr3D under a zero-shot setting and observe consistent improvements over SeeGround in our setup, including +2.5% Acc@0.5 on ScanRefer and +6.3% on Nr3D, with a notable +6.3% gain on Nr3D view-independent queries. These results show that combining selective retrieval, geometric reasoning, and adaptive visual inspection yields a practical and robust foundation for open-vocabulary 3D grounding. Our code is available at this https URL.
>
---
#### [new 080] DBPnet: Damper Characteristics-Based Bayesian Physics-Informed Neural Network for Wheel Load Estimation
- **分类: eess.SY; cs.AI; cs.ET; cs.LG; cs.RO**

- **简介: 该论文属于车辆状态估计任务，旨在解决轮载荷准确估算问题。通过构建物理信息神经网络，结合阻尼特性与贝叶斯推理，提升模型在噪声和不确定性下的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2605.24860](https://arxiv.org/pdf/2605.24860)**

> **作者:** Tianyi Wang; Tianyi Zeng; Zimo Zeng; Feiyang Zhang; Yujin Wang; Xiangyu Li; Yiming Xu; Sikai Chen; Junfeng Jiao; Christian Claudel; Xinbo Chen
>
> **备注:** 14 pages, 12 figures, 6 tables
>
> **摘要:** Advanced driver assistance systems (ADAS) play an important role in modern automotive intelligence, significantly enhancing vehicle safety and stability. The performance of ADAS critically relies on accurate and reliable vehicle state estimation, particularly from vehicle dynamic sensors. Among these signals, wheel load is a key variable for chassis control and safety-critical functions, yet it remains difficult to estimate robustly due to complex suspension geometry, nonlinear dynamics, and measurement noise. To address this issue, we propose DBPnet, a Bayesian physics-informed neural network (PINN) with a physics-aware embedding module inspired by damper characteristics. First, this paper presents a suspension linkage-level modeling (SLLM) approach that constructs a nonlinear instantaneous dynamic model by explicitly considering the complex geometric structure of the suspension. Building upon SLLM, Bayesian inference is integrated into the PINN to effectively cope with noise and uncertainty in the vehicle chassis system, thereby improving the model's robustness. Then, a physics-informed loss function is employed to ensure consistency with fundamental physical principles, while the damper characteristics-inspired embedding module extracts temporal variation features of input signals and incorporates them into each layer of the PINN, ensuring that physical observations guide the neural network without being constrained by fixed physical models. Extensive evaluations on high-fidelity simulations and real-world experiments demonstrate that our DBPnet consistently achieves lower RMSE and MaxError than baseline methods. These results highlight the potential of our DBPnet to advance wheel load estimation and contribute to the development of more reliable ADAS actuator functions.
>
---
#### [new 081] Beyond Predefined Learning Objects: A Thinking-Learning Interaction Model for Up-to-Date Autonomous Robot Learning
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出一种思考-学习交互模型，解决自主机器人在动态环境中适应新特征和任务的问题。通过双向机制提升机器人自适应能力。**

- **链接: [https://arxiv.org/pdf/2605.23987](https://arxiv.org/pdf/2605.23987)**

> **作者:** Hong Su
>
> **摘要:** Autonomous robots operating in open and changing environments cannot always rely on predefined inputs, outputs, and action routines. Although existing learning methods enable robots to improve their performance through environmental interaction, the objects of learning are often fixed in advance, such as input features, recognition outputs, network structures, task goals, or action sequences. This limits their ability to adapt when new features, new categories, or more efficient task routines appear during long-term operation. To address this problem, this paper proposes a thinking-learning interaction model for autonomous robots. The core idea is that thinking guides learning by identifying potential changes, selecting useful evidence, organizing training materials, and planning verification actions, while learning promotes thinking by updating task knowledge, feature-selection experience, action strategies, and future reasoning processes. Based on this bidirectional mechanism, the robot can gradually move beyond predefined learning settings and adapt its recognition relations and action relations through continuous interaction with the environment. Specifically, the proposed model supports adaptive input feature discovery, output category expansion, learning model update, and action routine reconstruction. Experimental results show that the proposed model improves the final recognition accuracy from 0.419 to 0.845 in feature adaptation, achieves higher new-category formation accuracy and model-update success rate, and reduces the average action length from 13.0 to 4.0 in action routine reconstruction. In learning-enhanced thinking, the useful evidence selection rate increases from 0.272 to 0.965, indicating that learning results can effectively improve future evidence selection and reasoning.
>
---
#### [new 082] A Reinforcement Learning Inspired Latent Yield Based Adaptive Algorithm Switching Mechanism
- **分类: cs.MA; cs.LG; cs.RO**

- **简介: 该论文属于算法选择任务，旨在解决动态环境中算法切换不稳定的问题。通过引入强化学习思想，构建适应性算法切换机制，提升算法选择的效率与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.24436](https://arxiv.org/pdf/2605.24436)**

> **作者:** Jayprakash S. Nair; Jimson Mathew; Shivashankar B. Nair
>
> **备注:** Accepted and published in the Proceedings of the 29th European Conference on Applications of Evolutionary Computation (EvoApplications 2026), held as part of EvoStar 2026, Toulouse, France, April 8 to 10, 2026. Lecture Notes in Computer Science (LNCS), Springer Nature Switzerland
>
> **摘要:** Selecting the most suitable algorithm for a given problem instance remains a challenging task, particularly in online or dynamic environments where problem characteristics evolve over time. Relying solely on instantaneous performance metrics can result in a reactive and unstable behaviour, often leading to suboptimal algorithm switching. This paper introduces a computationally efficient approach for aggregating an algorithm's performance across multiple problem instances that is fairly immune to erratic variations in instance features. Inspired by features inherent to Reinforcement Learning (RL), this technique encapsulates rewards and penalties into a latent yield that, in turn, triggers exploitation and exploration, consequently resulting in adaptive algorithm switching. The proposed technique employs island models, inspired by Genetic Algorithms, to facilitate parallel exploration and performance exchanges among algorithm populations inhabiting local repertoires. Experimental evaluations on sorting algorithms and robotic obstacle avoidance tasks demonstrate the feasibility and effectiveness of the approach, highlighting its potential in domains where adaptive algorithm selection is critical.
>
---
#### [new 083] Lifted Schrödinger Bridges for Gaussian Mixture Endpoints: Projection Gaps and Path-Space Obstructions
- **分类: math.OC; cs.LG; cs.RO; eess.SY**

- **简介: 该论文研究在布朗运动先验下，通过提升路径空间解决高斯混合分布间的随机密度控制问题，分析投影误差与路径空间障碍。**

- **链接: [https://arxiv.org/pdf/2605.24795](https://arxiv.org/pdf/2605.24795)**

> **作者:** Siddhartha Ganguly; George Rapakoulias; Panagiotis Tsiotras
>
> **备注:** 35 pages. Submitted to a journal; comments are welcome
>
> **摘要:** We study stochastic density control between Gaussian-mixture endpoint distributions under Brownian prior dynamics. Since the direct Schrödinger bridge between Gaussian mixtures is generally not available in closed form, we introduce a lifted path-space construction in which each trajectory is augmented with a source--target component label. Consequently, the problem decomposes into Gaussian component-to-component Schrödinger bridges with explicit marginal, drift, and cost formulas, while the mixture-level assignment reduces to a finite-dimensional entropic coupling problem with a Sinkhorn scaling form. We then analyze the projection obtained by discarding or forgetting the label. By construction, the projected law satisfies the original Gaussian-mixture endpoint constraints, but its relative entropy generally differs from the lifted relative entropy by a nonnegative conditional label-information gap. This gap reveals a path-space obstruction: the lifted optimizer cannot, in general, be identified with the direct unlabeled Schrödinger bridge after projection. We also derive the posterior-averaged Markov drift associated with the projected marginal flow, prove a kinetic-energy upper bound, and identify a common path-potential condition under which the projection gap vanishes. Several numerical illustrations showing density and shape control are recorded for a self-contained exposition.
>
---
#### [new 084] MIND: Multi-Scale Intent Diffusion for Text-Driven Physics-Based Humanoid Control
- **分类: cs.CV; cs.GR; cs.RO**

- **简介: 该论文属于文本驱动的物理拟人机器人控制任务，旨在解决文本与低级动作间语义对齐不足的问题。提出MIND框架，通过多尺度意图扩散实现更自然的运动生成。**

- **链接: [https://arxiv.org/pdf/2605.26006](https://arxiv.org/pdf/2605.26006)**

> **作者:** Bin Li; Ruichi Zhang; Han Liang; Jingyan Zhang; Juze Zhang; Xin Chen; Jingya Wang
>
> **摘要:** Enabling physics-based humanoids to execute diverse behaviors from high-level textual commands remains a significant challenge. Existing methods typically follow either a two-stage paradigm that combines kinematic motion generation with physics-based tracking, or an end-to-end imitation-learning paradigm that directly generates actions from text. However, the former suffers from the inherent domain shift between kinematic generation and physics-based tracking, while the latter struggles with the substantial modality gap between textual commands and low-level actions, limiting effective semantic alignment. Notably, humanoid states encode rich motion dynamics that are more semantically aligned with textual descriptions than low-level actions, making them a natural basis for deriving behavioral intent. Building upon this insight, we propose MIND, a novel end-to-end diffusion framework for text-driven physics-based humanoid control that leverages behavioral intent as a semantic bridge between textual commands and low-level actions. At its core, MIND introduces a multi-scale intent diffusion mechanism, where a holistic intent predictor captures global behavioral dynamics to guide overall behavior synthesis, while an immediate intent predictor provides step-wise, fine-grained signals for local behavior refinement at each diffusion step. This hierarchical intent formulation imposes a structured inductive bias for humanoid control, improving semantic alignment and behavioral naturalness. Furthermore, MIND encodes humanoid states into a latent space to enable more effective semantic intent modeling. Extensive experiments demonstrate that MIND outperforms existing methods and synthesizes coherent, physically plausible, and semantically aligned humanoid behaviors from text commands. Our code will be released to facilitate future research.
>
---
#### [new 085] Passivity-based Semi-autonomous Rotational Motion Navigation for Rigid-body Networks: Stability and Human Passivity Analysis
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于机器人控制任务，解决人机协作系统在SO(3)流形上的稳定性问题。提出基于被动性的半自主控制框架，确保系统稳定并分析人类操作者的被动性。**

- **链接: [https://arxiv.org/pdf/2605.24731](https://arxiv.org/pdf/2605.24731)**

> **作者:** Reiji Terunuma; Yuta Nakamura; Takeshi Hatanaka
>
> **备注:** This work is to be submitted to the 6th Workshop on Cyber-Physical Human Systems (CPHS2026) for possible publication
>
> **摘要:** This paper presents a novel passivity-based semi-autonomous attitude control framework, with a particular focus on attitude kinematics defined on the special orthogonal group $SO(3)$. While human-robot interaction facilitates the successful execution of complex tasks, ensuring stability of human-in-the-loop systems on the $SO(3)$ manifold remains a largely unsolved challenge. We first propose a new control architecture in which a multi-robot system preserves invariance of the average information fed back to the human operator through so-called stealthy control, and the human intervention is mediated through a virtual leader, which is coupled with the robots via a passivity-based attitude synchronization law. We then rigorously prove closed-loop stability of the proposed human-in-the-loop system under the assumption that the human behaves as a passive system. To support this analysis, simulation studies are conducted to identify the human operator as a dynamical system, and to examine passivity properties of the identified model.
>
---
#### [new 086] Grow-Prune-Freeze Networks: Adaptive & Continual Learning Technique for Olfactory Navigation
- **分类: cs.LG; cs.AI; cs.ET; cs.RO**

- **简介: 该论文提出GPF网络，用于嗅觉导航的持续学习任务，解决动态环境中模型适应性问题，通过生长、剪枝和冻结策略提升学习效果。**

- **链接: [https://arxiv.org/pdf/2605.25170](https://arxiv.org/pdf/2605.25170)**

> **作者:** Kordel K. France; Ovidiu Daescu
>
> **摘要:** Training data for olfaction is scattered through disparate, non-standardized datasets that limit the ability to build representative world models. Olfactory navigation is a highly dynamic and non-stationary task that benefits from real-time continual learning. We introduce an adaptive framework called Grow-Prune-Freeze (GPF) networks that enable an agent to continually learn through growing, pruning, and freezing early layers of its policy in response to world complexity. Grounding GPFs in non-linear random matrix theory, we show that the work of Pennington & Worth (2017) can be extended from single hidden layers to n-layer continual-learning models, and that eigenvalue composition of network weights is preserved as successive layers are added. We show that GPFs based on Expected SARSA achieve a 94% success rate on turbulent plume navigation - a partially observable, non-stationary task representative of the "big world" challenges that motivate adaptive learning in robotics - and provide supporting methodology for applying GPFs in other world models. Further experiments amount evidence that GPFs may generalize well to other machine learning tasks such as reinforcement learning in Atari, image classification, and autoregressive language models. We open source all code and data to encourage improvements on and more research in olfactory robotics.
>
---
#### [new 087] Cross-Domain Energy-Guided Diffusion Generation for Off-Dynamics Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO; stat.AP**

- **简介: 该论文属于离线强化学习任务，解决目标域与源域动态不匹配问题。提出CEDGE框架，通过能量引导生成目标域轨迹，提升策略学习效果。**

- **链接: [https://arxiv.org/pdf/2605.24810](https://arxiv.org/pdf/2605.24810)**

> **作者:** Yu Yang; Yihong Guo; Anqi Liu; Pan Xu
>
> **备注:** 29 pages, 3 figures, and 14 tables
>
> **摘要:** Off-dynamics offline reinforcement learning seeks to learn a target-domain policy from a large source dataset and a limited target dataset under mismatched transition dynamics. Existing approaches such as reward augmentation and data filtering are constrained to the source dataset and cannot synthesize new target behavior to improve coverage beyond the collected source trajectories. While recent model-based methods attempt to address this by learning target-aware dynamics, the generated experience is constructed only at the transition level, which leads to accumulated errors over long horizons. These limitations necessitate a shift toward trajectory-level generation for off-dynamics offline RL. We propose CEDGE, a Cross-domain Energy-guided Diffusion GEneration framework. CEDGE trains a trajectory diffusion model on source-domain trajectories and adapts the generated samples to the target domain through energy guidance. This guidance is derived by minimizing the distribution mismatch between the source and desired target-domain trajectories and is decomposed into return, domain, and behavior energy components. The resulting energy-guided trajectories are useful both for direct planning and as synthetic data for policy learning. Since target adaptation is achieved via energy guidance rather than retraining the diffusion model, CEDGE can be efficiently adapted to new target dynamics compared to previous methods. Experiments on the ODRL benchmark demonstrate that trajectory-level energy-guided generation improves diffusion planning under dynamics shifts and produces synthetic data that improves downstream target policy learning.
>
---
#### [new 088] MEMOR-E: In-Context and Fine-Tuned LLM Personalization for Alzheimer's Assistive Robotics
- **分类: cs.AI; cs.RO**

- **简介: 该论文提出MEMOR-E，用于阿尔茨海默病患者的辅助机器人系统，解决个性化互动与认知支持问题，通过微调大语言模型和上下文学习实现阶段感知的非诊断性认知总结。**

- **链接: [https://arxiv.org/pdf/2605.23941](https://arxiv.org/pdf/2605.23941)**

> **作者:** Maissa Abir Smaili; Eren Sadikoglu; Ransalu Senanayake
>
> **备注:** 8 pages 14 figures
>
> **摘要:** Alzheimer's disease is a neurodegenerative disorder marked by progressive declines in memory and language that reduce independence in daily life, motivating socially assistive robotic support. This paper presents MEMOR-E, a mobile quadruped robot with an interactive tablet interface that assists patients and caregivers through medication reminders, routine guidance, memory oriented interactions, and companionship. We evaluated the feasibility of fine tuning large language models (LLMs) to emulate stage consistent cognitive behavior and interpret responses across standard neuropsychological language tasks, using audio transcriptions from 235 Alzheimer's patients and synthetically generated healthy controls. We also report findings on using in context learning (ICL) in LLMs, where a second LLM produced domain and severity level cognitive error summaries. Our results show that MEMOR-E can generate stage aware, non diagnostic cognitive summaries that support personalized assistive interactions, while explainable AI mechanisms translate model outputs into transparent, human readable evidence to enable caregiver oversight and trustworthy human robot interaction.
>
---
#### [new 089] WideDepth: Millimeter-Accurate Benchmark for Fisheye Depth Estimation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于深度估计任务，旨在解决鱼眼相机深度基准缺失的问题。提出WideDepth数据集，包含高精度深度标签，并改进模型适应性，提升鱼眼深度估计性能。**

- **链接: [https://arxiv.org/pdf/2605.24074](https://arxiv.org/pdf/2605.24074)**

> **作者:** Ilia Indyk; Ignat Penshin; Ivan Sosin; Maxim Monastyrny; Aleksei Valenkov; Ilya Makarov
>
> **备注:** Accepted to IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Fisheye cameras are increasingly adopted in robotics for near-field manipulation, navigation, and immersive perception, yet indoor depth benchmarks with accurate ground truth are still missing. To address this, we introduce WideDepth - the first indoor dataset for fisheye depth estimation, featuring 101 scenes containing 5K high-resolution stereo pairs labeled with millimeter-level ground truth depth and disparity. Our dataset also includes paired pinhole and fisheye samples across varying fields of view and baselines in both horizontal and vertical stereo setups. We further propose a method to adapt pinhole-trained stereo models to fisheye images and introduce a novel stereo fisheye image generation pipeline based on high-resolution LiDAR scans. Leveraging these methods, we thoroughly evaluate state-of-the-art monocular depth, stereo matching, and depth completion models on our benchmark. Additionally, we provide 18K LiDAR-derived sparse depth training samples, achieving up to a 62% performance boost on fisheye data when fine-tuning pinhole-based stereo models. In summary, the high precision and versatility of our benchmark set a strong foundation for advancing research in fisheye depth estimation and robotics perception. Project page: this https URL
>
---
## 更新

#### [replaced 001] Efficient Long-Horizon Vision-Language-Action Models via Static-Dynamic Disentanglement
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人控制任务，旨在解决VLA模型的上下文限制和推理效率问题。通过静态-动态解耦提升多帧整合效率与推理速度。**

- **链接: [https://arxiv.org/pdf/2602.03983](https://arxiv.org/pdf/2602.03983)**

> **作者:** Weikang Qiu; Huashuo Lei; Tinglin Huang; Rex Ying
>
> **摘要:** Vision-Language-Action (VLA) models have recently emerged as a promising paradigm for generalist robotic control. Built upon vision-language model (VLM) architectures, VLAs predict actions conditioned on visual observations and language instructions, achieving strong performance and generalization across tasks. However, VLAs face two major challenges: a limited context window for input frames and inefficient inference due to the quadratic attention complexity and large parameter counts. To this end, we propose DySta, a framework that disentangles visual inputs into multi-level static and dynamic tokens, which enables (1) retaining a single copy of static tokens across frames to significantly reduce context length, and (2) reusing the key-value (KV) cache of static tokens through a lightweight recache gate that updates only when necessary. This design enables efficient multi-frame integration and efficient inference. In addition, we introduce a new benchmark that more effectively evaluates the multi-frame integration ability of VLAs. Experiments show that Dysta improves multi-frame integration by 24.5% across metrics on our benchmark and 23.3% in absolute success rate on real-world memory-dependent tasks, while accelerating inference by 2.0x (with +2.3% success rate) on simulation benchmarks and 2.2x (with +10.6% success rate) on real-world general tasks.
>
---
#### [replaced 002] AEROS: A Single-Agent Operating Architecture with Embodied Capability Modules
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出AEROS系统，解决机器人智能组织与执行问题，通过模块化能力包实现灵活控制与安全执行。属于机器人控制系统任务。**

- **链接: [https://arxiv.org/pdf/2604.07039](https://arxiv.org/pdf/2604.07039)**

> **作者:** Xue Qin; Simin Luan; John See; Cong Yang; Zhijun Li
>
> **备注:** Submitted to Engineering Applications of Artificial Intelligence (EAAI). 48 pages, 5 figures, 9 tables
>
> **摘要:** Robotic systems lack a principled abstraction for organizing intelligence, capabilities, and execution in a unified manner. Existing approaches either couple skills within monolithic architectures or decompose functionality into loosely coordinated modules or multiple agents, often without a coherent model of identity and control authority. We argue that a robot should be modeled as a single persistent intelligent subject whose capabilities are extended through installable packages. We formalize this view as AEROS (Agent Execution Runtime Operating System), in which each robot corresponds to one persistent agent and capabilities are provided through Embodied Capability Modules (ECMs). Each ECM encapsulates executable skills, models, and tools, while execution constraints and safety guarantees are enforced by a policy-separated runtime. This separation enables modular extensibility, composable capability execution, and consistent system-level safety. We evaluate a reference implementation in PyBullet simulation with a Franka Panda 7-DOF manipulator across eight experiments covering re-planning, failure recovery, policy enforcement, baseline comparison, cross-task generality, ECM hot-swapping, ablation, and failure boundary analysis. Over 100 randomized trials per condition, AEROS achieves 100% task success across three tasks versus baselines (this http URL-style and ProgPrompt-style at 92--93%, flat pipeline at 67--73%), the policy layer blocks all invalid actions with zero false acceptances, runtime benefits generalize across tasks without task-specific tuning, and ECMs load at runtime with 100% post-swap success.
>
---
#### [replaced 003] CollaBot: Vision-Language Guided Simultaneous Collaborative Manipulation
- **分类: cs.RO**

- **简介: 该论文提出CollaBot框架，解决多机器人协同操作大物体的问题。通过场景分割、协作抓取和路径规划，提升复杂任务的成功率。属于多机器人协作操纵任务。**

- **链接: [https://arxiv.org/pdf/2508.03526](https://arxiv.org/pdf/2508.03526)**

> **作者:** Kun Song; Gaoming Chen; Shentao Ma; Ninglong Jin; Guangbao Zhao; Mingyu Ding; Zhenhua Xiong; Jia Pan
>
> **备注:** 8 pages,6 figures
>
> **摘要:** One central goal of robotics is to enable robots to interact with the physical world. Traditional manipulation studies primarily focus on single robots and relatively small objects. However, factory and domestic environments often require large-object manipulation, such as moving tables, where multiple robots must work collaboratively. Existing studies still lack a generalizable framework that can handle diverse objects, tasks, and robot team sizes. In this work, we propose CollaBot, a generalist framework for simultaneous collaborative manipulation. First, we use SEEM for scene segmentation and target-object extraction. Then, we propose a collaborative grasping framework that decomposes the task into local grasp pose generation and global coordination. Finally, we design a two-stage planning module to generate collision-free trajectories for task execution. Experimental results across different settings with varying objects, tasks, and numbers of robots indicate that our framework achieves a 72% success rate. This marks a substantial improvement over behavior cloning-based methods, validating the advantages of the proposed framework in complex multi-robot cooperative tasks. Real-world experiments further demonstrate the feasibility of our method in practical applications.
>
---
#### [replaced 004] A Formal gatekeeper Framework for Safe Dual Control with Active Exploration
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决模型不确定性下的安全双控制问题。通过整合鲁棒规划与主动探索，确保安全并减少不确定性，提升任务效率。**

- **链接: [https://arxiv.org/pdf/2510.06351](https://arxiv.org/pdf/2510.06351)**

> **作者:** Kaleb Ben Naveed; Devansh R. Agrawal; Dimitra Panagou
>
> **备注:** Accepted at American Control Conference (ACC) 2026
>
> **摘要:** Planning safe trajectories under model uncertainty is a fundamental challenge. Robust planning ensures safety by considering worst-case realizations, yet ignores uncertainty reduction and leads to overly conservative behavior. Actively reducing uncertainty on-the-fly during a nominal mission defines the dual control problem. Most approaches address this by adding a weighted exploration term to the cost, tuned to trade off the nominal objective and uncertainty reduction, but without formal consideration of when exploration is beneficial. Moreover, safety is enforced in some methods but not in others. We propose a framework that integrates robust planning with active exploration under formal guarantees as follows: The key innovation and contribution is that exploration is pursued only when it provides a verifiable improvement without compromising safety. To achieve this, we utilize our earlier work on gatekeeper as an architecture for safety verification, and extend it so that it generates both safe and informative trajectories that reduce uncertainty and the cost of the mission, or keep it within a user-defined budget. The methodology is evaluated via simulation case studies on the online dual control of a quadrotor under parametric uncertainty.
>
---
#### [replaced 005] A neural signed configuration distance function for path planning of picking manipulators
- **分类: cs.RO**

- **简介: 该论文属于路径规划任务，解决 picking manipulators 的高效路径规划问题。提出一种神经符号配置距离函数，构建无碰撞球体，提升在线规划效率。**

- **链接: [https://arxiv.org/pdf/2502.16205](https://arxiv.org/pdf/2502.16205)**

> **作者:** Bernhard Wullt; Mikael Norrlöf; Per Mattsson; Thomas B. Schön
>
> **摘要:** Picking manipulators are task specific robots, with fewer degrees of freedom compared to general-purpose manipulators, and are heavily used in industry. The efficiency of the picking robots is highly dependent on the path planning solution, which is commonly based on sampling-based multi-query methods. The planner is robustly able to solve the problem, but its heavy use of collision-detection limits the planning capabilities for online use. We approach this problem by presenting a novel implicit obstacle representation for path planning, a neural signed configuration distance function (nSCDF), which allows us to form collision-free balls in the configuration space. We use the ball representation to re-formulate a state of the art multi-query path planner, i.e., instead of points, we use balls in the graph. Our planner returns a collision-free corridor, which allows us to use convex programming to produce optimized paths. From our numerical experiments, we observe that our planner produces paths that are close to those from an asymptotically optimal path planner, in significantly less time.
>
---
#### [replaced 006] A Closed-Form Dual-Barrier CBF Safety Filter for Holonomic Robots on Incrementally Built Occupancy Grid Maps
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人安全控制任务，解决未知环境中全向机器人实时避障问题。提出双屏障控制屏障函数安全滤波器，确保在增量构建的占用网格地图中安全运动。**

- **链接: [https://arxiv.org/pdf/2605.05182](https://arxiv.org/pdf/2605.05182)**

> **作者:** Himanshu Paudel; Basanta Joshi; Dhirendra Raj Madai; Alina Bartaula; Biman Rimal; Sanjay Neupane
>
> **摘要:** We present a dual-barrier control barrier function (CBF) safety filter for real-time, safety-critical velocity control of holonomic robots operating in incrementally built occupancy grid maps. As a robot explores an unknown environment, unmapped regions introduce irreducible uncertainty, since obstacle geometry beyond the explored frontier is unknown, making entry into such regions a source of collision risk, especially with front-facing sensors. To address this, we enforce two constraints: avoidance of mapped obstacles and restriction from unexplored regions. Both constraints are derived analytically from the occupancy grid's signed distance field, yielding a closed-form safety filter that requires only a small linear system solve per cycle. On resource-constrained platforms such as the Raspberry Pi, where SLAM and planning already consume significant compute, the low overhead of the proposed filter preserves resources. An adaptive gain schedule relaxes the frontier constraint in information-rich regions and tightens it in well-mapped areas, improving exploration efficiency while maintaining safety. The filter operates in velocity space as a minimally invasive correction and composes with arbitrary nominal controllers, including learning-based methods. Hardware flight experiments on a PX4-controlled quadrotor demonstrate zero collisions across multiple indoor runs.
>
---
#### [replaced 007] DPNet: Doppler LiDAR Motion Planning for Highly-Dynamic Environments
- **分类: cs.RO**

- **简介: 该论文属于动态环境下的运动规划任务，旨在解决快速移动障碍物的感知与规划问题。通过融合多普勒LiDAR与神经网络，提出DPNet实现高精度、高频的障碍物跟踪与路径规划。**

- **链接: [https://arxiv.org/pdf/2512.00375](https://arxiv.org/pdf/2512.00375)**

> **作者:** Wei Zuo; Zeyi Ren; Chengyang Li; Yikun Wang; Mingle Zhao; Shuai Wang; Wei Sui; Fei Gao; Yik-Chung Wu; Chengzhong Xu
>
> **备注:** Accepted to IEEE Robotics and Automation Letters in April, 2026
>
> **摘要:** Existing motion planning methods often struggle with rapid-motion obstacles due to an insufficient understanding of environmental changes. To address this, we propose integrating motion planners with Doppler LiDARs, which provide not only ranging measurements but also instantaneous point velocities. However, this integration is nontrivial due to the requirements of high accuracy and high frequency. To this end, we introduce Doppler Planning Network (DPNet), which tracks and reacts to rapid obstacles via Doppler model-based learning. We first propose a Doppler Kalman neural network (D-KalmanNet) to track obstacle states under a partially observable Gaussian state space model. We then leverage the predicted motions of obstacles to construct a Doppler-tuned model predictive control (DT-MPC) framework for ego-motion planning, enabling runtime auto-tuning of controller parameters. These two modules allow DPNet to learn fast environmental changes from minimal data while remaining lightweight, achieving high frequency and high accuracy in both tracking and planning. Experiments on high-fidelity simulator and real-world datasets demonstrate the superiority of DPNet over extensive benchmark schemes. Code available at this https URL
>
---
#### [replaced 008] What Questions Should Robots Be Able to Answer? A Dataset of User Questions for Explainable Robotics
- **分类: cs.RO; cs.CL; cs.HC**

- **简介: 该论文属于人机交互任务，旨在解决机器人如何回答用户问题的问题。通过收集用户对家用机器人的提问数据，为机器人问答系统提供基准和指导。**

- **链接: [https://arxiv.org/pdf/2510.16435](https://arxiv.org/pdf/2510.16435)**

> **作者:** Lennart Wachowiak; Andrew Coles; Gerard Canal; Oya Celiktutan
>
> **摘要:** With the growing use of large language models and conversational interfaces in human-robot interaction, robots' ability to answer user questions is more important than ever. We therefore introduce a dataset of 1,893 user questions for household robots, collected from 100 participants and organized into 12 categories and 70 subcategories. Most work in explainable robotics focuses on why-questions. In contrast, our dataset provides a wide variety of questions, from questions about simple execution details to questions about how the robot would act in hypothetical scenarios -- thus giving roboticists valuable insights into what questions their robot needs to be able to answer. To collect the dataset, we created 15 video stimuli and 7 text stimuli, depicting robots performing varied household tasks. We then asked participants on Prolific what questions they would want to ask the robot in each portrayed situation. In the final dataset, the most frequent categories are questions about task execution details (21.4%), the robot's capabilities (12.6%), and performance assessments (10.7%). Although questions about how robots would handle potentially difficult scenarios and ensure correct behavior are less frequent, users rank them as the most important for robots to be able to answer. Moreover, we find that users who identify as novices in robotics ask different questions than more experienced users. Novices are more likely to inquire about simple facts, such as what the robot did or the current state of the environment. As robots enter environments shared with humans and language becomes central to giving instructions and interaction, this dataset provides a valuable foundation for (i) identifying the information robots need to log and expose to conversational interfaces, (ii) benchmarking question-answering modules, and (iii) designing explanation strategies that align with user expectations.
>
---
#### [replaced 009] Safety in Embodied AI: A Survey of Risks, Attacks, and Defenses
- **分类: cs.CR; cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于安全研究任务，旨在解决 embodied AI 的安全性问题。通过综述攻击与防御方法，分析感知、决策到交互各环节的安全风险，提出多层级分类框架，识别关键研究缺口。**

- **链接: [https://arxiv.org/pdf/2605.02900](https://arxiv.org/pdf/2605.02900)**

> **作者:** Xiao Li; Xiang Zheng; Yifeng Gao; Xinyu Xia; Yixu Wang; Xin Wang; Ye Sun; Yunhan Zhao; Ming Wen; Jiayu Li; Zixing Chen; Xun Gong; Yi Liu; Yige Li; Yutao Wu; Cong Wang; Jun Sun; Yixin Cao; Zhineng Chen; Jingjing Chen; Tao Gui; Qi Zhang; Zuxuan Wu; Xipeng Qiu; Xuanjing Huang; Tiehua Zhang; Zhipeng Wei; Kun Wang; Xinfeng Li; Hanxun Huang; Sarah Erfani; James Bailey; Jianping Wang; Chaowei Xiao; Ran He; Bo Li; Xingjun Ma; Yu-Gang Jiang
>
> **备注:** Survey paper; 75 pages, 4 figures, 18 tables; v2 expands embodied-specific coverage of agentic threats, World Action Model threats, and contextual risk mitigation, with over 100 new references added. Project page: this https URL
>
> **摘要:** Embodied Artificial Intelligence (Embodied AI) integrates perception, cognition, planning, and interaction into agents that operate in open-world, safety-critical environments. As these systems gain autonomy and enter domains such as transportation, healthcare, and industrial or assistive robotics, ensuring their safety becomes both technically challenging and socially indispensable. Unlike digital AI systems, embodied agents must act under uncertain sensing, incomplete knowledge, and dynamic human-robot interactions, where failures can directly lead to physical harm. This survey provides a comprehensive and structured review of safety research in embodied AI, examining attacks and defenses across the full embodied pipeline, from perception and cognition to planning, action and interaction, and agentic system. We introduce a multi-level taxonomy that unifies fragmented lines of work and connects embodied-specific safety findings with broader advances in vision, language, and multimodal foundation models. Our review synthesizes insights from over 500 papers spanning adversarial, backdoor, jailbreak, and hardware-level attacks; attack detection, safe training and robust inference; and risk-aware human-agent interaction. This analysis reveals several overlooked challenges, including the fragility of multimodal perception fusion, the instability of planning under jailbreak attacks, and the trustworthiness of human-agent interaction in open-ended scenarios. By organizing the field into a coherent framework and identifying critical research gaps, this survey provides a roadmap for building embodied agents that are not only capable and autonomous but also safe, robust, and reliable in real-world deployment.
>
---
#### [replaced 010] Soft Pneumatic Grippers: Topology optimization, 3D-printing and Experimental validation
- **分类: cs.RO**

- **简介: 该论文属于软体机械臂设计任务，解决软气动夹爪的结构优化问题。通过拓扑优化、3D打印和实验验证，提升夹爪的抓取性能。**

- **链接: [https://arxiv.org/pdf/2511.19211](https://arxiv.org/pdf/2511.19211)**

> **作者:** Prabhat Kumar; Chandra Prakash; Josh Pinskier; David Howard; Matthijs Langelaar
>
> **备注:** 11 Figures
>
> **摘要:** This paper presents a systematic topology optimization framework for designing a soft pneumatic gripper (SPG), explicitly considering the design-dependent nature of the actuating load. The load is modeled using Darcy's law with an added drainage term. A 2D soft arm unit is optimized by formulating it as a compliant mechanism design problem using the robust formulation. The problem is posed as a min-max optimization, where the output deformations of blueprint and eroded designs are considered. A volume constraint is imposed on the blueprint part, while a strain-energy constraint is enforced on the eroded part. The MMA is employed to solve the optimization problem and obtain the optimized soft unit. Finite element analysis with the Ogden material model confirms that the optimized 2D unit outperforms a conventional rectangular design under pneumatic loading. The optimized 2D unit is extruded to obtain a 3D module, and ten such units are assembled to create a soft arm. Deformation profiles of the optimized arm are analysed under different pressure loads. Four arms are 3D-printed and integrated with a supporting structure to realize the proposed SPG. The gripping performance of the SPG is demonstrated on objects with different weights, sizes, stiffness, and shapes.
>
---
#### [replaced 011] TimeSpot: Benchmarking Geo-Temporal Understanding in Vision-Language Models in Real-World Settings
- **分类: cs.CV; cs.CL; cs.ET; cs.MM; cs.RO**

- **简介: 该论文提出TimeSpot基准，用于评估视觉语言模型在真实场景中的时空理解能力。解决VLM在时间与空间推理上的不足，通过图像预测地理和时间属性及进行时空推理任务。**

- **链接: [https://arxiv.org/pdf/2603.06687](https://arxiv.org/pdf/2603.06687)**

> **作者:** Azmine Toushik Wasi; Shahriyar Zaman Ridoy; Koushik Ahamed Tonmoy; Kinga Tshering; S. M. Muhtasimul Hasan; Wahid Faisal; Tasnim Mohiuddin; Md Rizwan Parvez
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Geo-temporal understanding, the ability to infer location, time, and contextual properties from visual input alone, underpins applications such as disaster management, traffic planning, embodied navigation, world modeling, and geography education. Although recent vision-language models (VLMs) have advanced image geo-localization using cues like landmarks and road signs, their ability to reason about temporal signals and physically grounded spatial cues remains limited. To address this gap, we introduce TimeSpot, a benchmark for evaluating real-world geo-temporal reasoning in VLMs. TimeSpot comprises 1,455 ground-level images from 80 countries and requires structured prediction of temporal attributes (season, month, time of day, daylight phase) and geographic attributes (continent, country, climate zone, environment type, latitude-longitude) directly from visual evidence. It also includes spatial-temporal reasoning tasks that test physical plausibility under real-world uncertainty. Evaluations of state-of-the-art open- and closed-source VLMs show low performance, particularly for temporal inference. While supervised fine-tuning yields improvements, results remain insufficient, highlighting the need for new methods to achieve robust, physically grounded geo-temporal understanding TimeSpot is available at: this https URL.
>
---
#### [replaced 012] Novel Algorithms for Smoothly Differentiable and Efficiently Vectorizable Contact Manifold Construction
- **分类: cs.RO**

- **简介: 该论文属于机器人接触建模任务，旨在解决接触环境中梯度计算困难的问题。通过设计可微且适合GPU并行的碰撞检测方法，提升优化效率。**

- **链接: [https://arxiv.org/pdf/2604.17538](https://arxiv.org/pdf/2604.17538)**

> **作者:** Onur Beker; Andreas René Geist; Anselm Paulus; Georg Martius
>
> **备注:** This version adds late-breaking results in preparation for the CR2 workshop in ICRA 2026
>
> **摘要:** Generating intelligent robot behavior in contact-rich settings is a research problem where zeroth-order methods currently prevail. Developing methods that make use of first/second order information about rigid-body dynamics in the presence of contact holds great promise in terms of increasing the solution speed and computational efficiency. The main bottleneck in this research direction is the difficulty in obtaining gradients and Hessians that are actually useful for numerical optimization, due to pathologies in all three steps of a common simulation pipeline: i) collision detection, ii) contact dynamics, iii) time integration. This abstract proposes a method that aims to address the collision detection part of the puzzle, via a novel pipeline designed from scratch with smooth (i.e. twice) differentiability and massive vectorizability on GPUs as the main priorities. This is in contrast to standard collision detection routines that are instead optimized for runtime on CPUs and minimal memory footprint, but do employ logic and control flow that hinder differentiability and vectorization. The proposed pipeline consists of the following contributions: i) highly expressive and compute efficient SDF representations, ii) differentiable broad-phase and narrow-phase routines that use these representations to generate vertex-SDF and edge-SDF contacts, iii) a differentiable routine for convex decomposition based contact blending.
>
---
#### [replaced 013] Stein Variational Ergodic Surface Coverage with SE(3) Constraints
- **分类: cs.RO**

- **简介: 该论文属于机器人表面操作任务，解决点云目标下轨迹优化问题。提出一种带SE(3)约束的预处理Stein变分梯度下降方法，提升轨迹覆盖质量与计算效率。**

- **链接: [https://arxiv.org/pdf/2603.09458](https://arxiv.org/pdf/2603.09458)**

> **作者:** Jiayun Li; Yufeng Jin; Sangli Teng; Dejian Gong; Georgia Chalvatzaki
>
> **摘要:** Surface manipulation tasks require robots to generate trajectories that comprehensively cover complex 3D surfaces while maintaining precise end-effector poses. Existing ergodic trajectory optimization (TO) methods demonstrate success in coverage tasks, while struggling with point-cloud targets due to the nonconvex optimization landscapes and the inadequate handling of SE(3) constraints in sampling-as-optimization (SAO) techniques. In this work, we introduce a preconditioned SE(3) Stein Variational Gradient Descent (SVGD) approach for SAO ergodic trajectory generation. Our proposed approach comprises multiple innovations. First, we reformulate point-cloud ergodic coverage as a manifold-aware sampling problem. Second, we derive SE(3)-specific SVGD particle updates, and, third, we develop a preconditioner to accelerate TO convergence. Our sampling-based framework consistently identifies superior local optima compared to strong optimization-based and SAO baselines while preserving the SE(3) geometric structure. Experiments on a 3D point-cloud surface coverage benchmark and robotic surface drawing tasks demonstrate that our method achieves superior coverage quality with tractable computation in our setting relative to existing TO and SAO approaches, and is validated in real-world robot experiments.
>
---
#### [replaced 014] Design, Control, and Motion Strategy for DELTA: Transformable Multilink Multirotor for Air-Ground Hybrid Locomotion and Manipulation
- **分类: cs.RO**

- **简介: 该论文属于多模态机器人任务，旨在解决空地协同运动与操作难题。设计了一种可变形多连杆多旋翼机器人，实现空地混合运动与操作。**

- **链接: [https://arxiv.org/pdf/2403.06636](https://arxiv.org/pdf/2403.06636)**

> **作者:** Kazuki Sugihara; Moju Zhao; Takuzumi Nishio; Kei Okada; Masayuki Inaba
>
> **备注:** 20 pages, 31 figures
>
> **摘要:** In recent years, multimodal locomotion capabilities have enabled robots to maneuver in both terrestrial and aerial domains. However, most of these robots are designed only for locomotion, and few possess the manipulation capabilities required for practical tasks. By adding a manipulator, ground robots can perform manipulation, and some drones with robotic arms have demonstrated aerial manipulation. Nonetheless, such multirotors cannot be directly used for manipulation on the ground, and this configuration itself is unsuitable for air-ground hybrid locomotion. This is because their thruster-centralized structure makes it difficult to achieve both sufficient degrees of freedom (DoF) for manipulation and stable motion with contact and transformation. Therefore, in this work, we develop a new multilink multirotor with thrusters on each link and capable of contact with the environments. This robot can perform terrestrial rolling locomotion, aerial flight locomotion, and manipulation in multiple environments using joint actuation. First, we introduce a minimal configuration design of the proposed robot. We also describe a kinematic model and propose a design for each component based on this model. Second, we propose a real-time control method based on nonlinear optimization that considers contact and joint motion, which can be applied to various multirotors. Third, we propose motion strategies that include contact constraints specific to air-ground hybrid multilink multirotors, and analyze the limitations of manipulation capabilities based on multi-contact model. Finally, we demonstrate a variety of motions in both domains using the implemented prototype. To the best of our knowledge, this is the first demonstration of air-ground hybrid locomotion and manipulation by a multilink multirotor.
>
---
#### [replaced 015] Data-Driven Optimization of Tactile Sensor Configurations for Efficient Dexterous Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人触觉感知领域，旨在优化触觉传感器配置以提升操作效率。研究解决了传感器布局对深度强化学习性能影响的问题，通过系统方法识别关键传感器并降低硬件成本。**

- **链接: [https://arxiv.org/pdf/2409.20473](https://arxiv.org/pdf/2409.20473)**

> **作者:** Haoran Guo; Haoyang Wang; Zhengxiong Li; He Bai; Lingfeng Tao
>
> **备注:** This work has been submitted to the ICRA for possible publication
>
> **摘要:** Tactile sensing is critical for learning-based dexterous manipulation, yet principled guidelines for sensor placement remain largely absent. While dense sensor arrays provide rich contact feedback, they impose significant hardware costs and can even degrade policy performance by introducing redundant or conflicting inputs. This paper presents the first systematic framework for quantifying the contribution of individual tactile sensors to deep reinforcement learning (DRL) policy performance. We propose a two-stage approach: a coarse empirical pruning phase that reduces the sensor count on the Shadow Hand from 92 to 21 while retaining 93\% task performance, followed by a fine-grained active learning phase that combines Gaussian Process Regression (GPR) with Lasso regression to rank the functional importance of each remaining sensor. Our analysis reveals that sensors on the thumb, ring finger, and little finger dominate manipulation performance, while middle-finger sensors exhibit negative contributions -- actively degrading policy learning. Ablation studies across three manipulation tasks (block, egg, and pen) confirm that a 14-sensor configuration preserves over 90\% of the full-array performance. Zero-shot transfer experiments on two novel objects and cross-platform validation on the Allegro and Leap Hand further demonstrate that the identified importance rankings generalize across tasks and robot morphologies. These findings establish quantitative deployment guidelines that enable practitioners to select cost-effective sensor configurations with predictable performance trade-offs.
>
---
#### [replaced 016] ESI-Bench: Towards Embodied Spatial Intelligence that Closes the Perception-Action Loop
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文提出ESI-Bench，用于评估具身空间智能，解决感知-行动闭环问题，通过主动探索提升任务表现。**

- **链接: [https://arxiv.org/pdf/2605.18746](https://arxiv.org/pdf/2605.18746)**

> **作者:** Yining Hong; Jiageng Liu; Han Yin; Manling Li; Leonidas Guibas; Li Fei-Fei; Jiajun Wu; Yejin Choi
>
> **备注:** this https URL
>
> **摘要:** Spatial intelligence unfolds through a perception-action loop: agents act to acquire observations, and reason about how observations vary as a function of action. Rather than passively processing what is seen, they actively uncover what is unseen - occluded structure, dynamics, containment, and functionality that cannot be resolved from passive sensing alone. We move beyond prior formulations of spatial intelligence that assume oracle observations by recasting the observer as an actor. We introduce ESI-BENCH, a comprehensive benchmark for embodied spatial intelligence spanning 10 task categories and 29 subcategories built on OmniGibson, grounded in Spelke's core knowledge systems. Agents must decide what abilities to deploy - perception, locomotion, and manipulation - and how to sequence them to actively accumulate task-relevant evidence. We conduct extensive experiments on state-of-the-art MLLMs and find that active exploration substantially outperforms passive counterparts, with agents spontaneously discovering emergent spatial strategies without explicit instructions, while random multi-view often adds noise rather than signal despite consuming far more images. Most failures stem not from weak perception but from action blindness: poor action choices lead to poor observations, which in turn drive cascading errors. While explicit 3D grounding stabilizes reasoning on depth-sensitive tasks, imperfect 3D representation proves more harmful than 2D baselines by distorting spatial relations. Human studies further reveal that unlike humans who seek falsifying viewpoints and revise beliefs under contradiction, models commit prematurely with high confidence regardless of evidence quality, exposing a metacognitive gap that neither better perception nor more embodied interaction alone can close.
>
---
#### [replaced 017] Neuromorphic Control of a Flapping-Wing Robot on Resource-Constrained Hardware
- **分类: cs.RO**

- **简介: 论文提出一种基于脉冲神经网络的控制框架，用于资源受限硬件上的扑翼飞行器自主控制，解决其在尺寸、重量和功耗限制下的实时控制问题。**

- **链接: [https://arxiv.org/pdf/2605.19430](https://arxiv.org/pdf/2605.19430)**

> **作者:** Rim El Filali; Chenrui Feng; Chao Gao; Weibin Gu
>
> **摘要:** Flapping-Wing Micro Aerial Vehicles (FWMAVs) provide exceptional maneuverability and aerodynamic efficiency but pose significant challenges for onboard control due to nonlinear dynamics and stringent Size, Weight, and Power (SWaP) constraints, as exemplified by a butterfly-inspired robot less than 30 gram. To this end, we present a hierarchical neuromorphic control framework that enables fully onboard, closed-loop flight on a widely available, resource-constrained ESP32 microcontroller with a unit cost of approximately $5. Specifically, our method deploys two lightweight Spiking Neural Networks (SNNs) onboard: one for state estimation from raw sensory feedback and another for control via modulation of a Central Pattern Generator (CPG) for wing actuation. Trained by imitation learning, the system achieves stable pitch and heading angle tracking during untethered real-world flight. Experimental results further reveal that the SNN-based controller reduces latency by 36% (1059us to 680us) and power by 18% (0.033W to 0.027W) for inference compared to the conventional Artificial Neural Network (ANN) baseline, demonstrating the viability of spike-based computation without specialized hardware. To the best of our knowledge, this work constitutes the first demonstration of fully onboard neuromorphic control for autonomous flight of a FWMAV, highlighting the potential of SNNs to enable energy-efficient autonomy under stringent SWaP constraints. Visual abstract: this http URL Code: this https URL
>
---
#### [replaced 018] Fundamental Limits for Sensor-Based Control via the Gibbs Variational Principle
- **分类: math.OC; cs.RO; eess.SY**

- **简介: 该论文属于控制理论领域，解决反馈控制器性能极限问题。通过吉布斯变分原理，建立部分观测下的控制器成本下界，提供可计算的性能评估方法。**

- **链接: [https://arxiv.org/pdf/2603.18454](https://arxiv.org/pdf/2603.18454)**

> **作者:** Vincent Pacelli; Evangelos A. Theodorou
>
> **备注:** First revision. Added LQG numerical example. Improved exposition throughout. 6 pages, 1 figure
>
> **摘要:** Fundamental limits on the performance of feedback controllers are essential for benchmarking algorithms, guiding sensor selection, and certifying task feasibility -- yet few general-purpose tools exist for computing them. Existing information-theoretic approaches overestimate the information a sensor must provide by evaluating it against the uncontrolled system, producing bounds that degrade precisely when feedback is most valuable. We derive a lower bound on the minimum expected cost of any causal feedback controller under partial observations by applying the Gibbs variational principle to the joint path measure over states and observations. The bound applies to nonlinear, nonholonomic, and hybrid dynamics with unbounded costs and admits a self-consistent refinement: any good controller concentrates the state, which limits the information the sensor can extract, which tightens the bound. The resulting fixed-point equation has a unique solution computable by bisection, and we provide conditions under which the free energy minimization is provably convex, yielding a certifiably correct numerical bound. On a scalar LQG problem the self-consistent bound captures over 80% of the known optimal cost at moderate sensor noise, and on a nonlinear Dubins car tracking problem it remains informative across all noise levels where a bound using the uncontrolled state distribution is vacuous.
>
---
#### [replaced 019] Action with Visual Primitives
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型中指令理解与动作控制耦合的问题。提出AVP架构，通过视觉原语提升动作生成效果。**

- **链接: [https://arxiv.org/pdf/2605.22183](https://arxiv.org/pdf/2605.22183)**

> **作者:** Weilong Guo; Yuchen Wang; Renping Zhou; Yunfeng Zhang; Rui Fang; Yuyang Pang; Wenda Xu; Gao Huang
>
> **备注:** 9 pages, 6 figures. Project page: this https URL
>
> **摘要:** Vision-Language-Action (VLA) models have emerged as a promising paradigm for generalist robotic manipulation. A common design in current architectures maps language instructions and visual observations to actions in a single forward pass. While conceptually simple, this formulation entangles instruction comprehension, spatial scene understanding, and motor control within a single learning objective. As a result, the action expert must implicitly relearn cognitive and perceptual capabilities already present in the pretrained VLM, which can limit both learning efficiency and generalization. We introduce AVP (Action with Visual Primitives), an end-to-end architecture that implements this visual-primitive-centric interface: the VLM infers the next-stage target and emits visual-primitive tokens that condition a flow-matching action expert, with supervision derived from end-effector kinematics. Real-robot experiments on general pick-and-place tasks show that AVP improves the success rate by 27.61% over pi_0.5 and outperforms other recent methods, with consistent gains in data efficiency, spatial-compositional generalization, and object-level transfer.
>
---
#### [replaced 020] VILAS: A VLA-Integrated Low-cost Architecture with Soft Grasping for Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出VILAS系统，用于低成本机器人操作任务。解决如何在无力觉传感下安全抓取脆弱物体的问题，设计软夹爪并集成视觉-语言-动作策略进行实验验证。**

- **链接: [https://arxiv.org/pdf/2605.02037](https://arxiv.org/pdf/2605.02037)**

> **作者:** Zijian An; Hadi Khezam; Bill Cai; Ran Yang; Shijie Geng; Yiming Feng; Yue Zheng; Lifeng Zhou
>
> **摘要:** We present VILAS, a fully low-cost, modular robotic manipulation platform designed to support end-to-end vision-language-action (VLA) policy learning and deployment on accessible hardware. The system integrates a Fairino FR5 collaborative arm, a Jodell RG52-50 electric gripper, and a dual-camera perception module, unified through a ZMQ-based communication architecture that seamlessly coordinates teleoperation, data collection, and policy deployment within a single framework. To enable safe manipulation of fragile objects without relying on explicit force sensing, we design a kirigami-based soft compliant gripper extension that induces predictable deformation under compressive loading, providing gentle and repeatable contact with delicate targets. We deploy and evaluate three state-of-the-art VLA models on the VILAS platform: pi_0, pi_0.5, and GR00T N1.6. All models are fine-tuned from publicly released pretrained checkpoints using an identical demonstration dataset collected via our teleoperation pipeline. Experiments on a grape grasping task validate the effectiveness of the proposed system, confirming that capable manipulation policies can be successfully trained and deployed on low-cost modular hardware. Our results further provide practical insights into the deployment characteristics of current VLA models in real-world settings.
>
---
#### [replaced 021] SpecPrune-VLA: Accelerating Vision-Language-Action Models via Action-Aware Self-Speculative Pruning
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文针对视觉-语言-动作模型的加速问题，提出SpecPrune-VLA方法，通过结合局部与全局信息进行剪枝，提升推理速度并保持高成功率。**

- **链接: [https://arxiv.org/pdf/2509.05614](https://arxiv.org/pdf/2509.05614)**

> **作者:** Hanzhen Wang; Jiaming Xu; Yushun Xiang; Jiayi Pan; Yongkang Zhou; Yong-Lu Li; Guohao Dai
>
> **备注:** Accepted to ICML 2026
>
> **摘要:** Pruning is a typical acceleration technique for compute-bound models by removing computation on unimportant values. Recently, it has been applied to accelerate Vision-Language-Action (VLA) model inference. However, existing acceleration methods focus on local information from the current action step and ignore the global context, leading to >20% success rate drop and limited speedup in some scenarios. In this paper, we point out spatial-temporal consistency in VLA tasks: input images in consecutive steps exhibit high similarity, and propose the key insight that token selection should combine local information with global context of the model. Based on this, we propose SpecPrune-VLA, a training-free, two-level pruning method with heuristic control. (1) Action-level static pruning. We leverage global history and local attention to statically reduce visual tokens per action. (2) Layer-level dynamic pruning. We prune tokens adaptively per layer based on layer-wise importance. (3) Lightweight action-aware controller: We classify actions as coarse- or fine-grained by the speed of the end effector and adjust pruning aggressiveness accordingly. Extensive experiments show that SpecPrune-VLA achieves up to 1.57$\times$ speedup in LIBERO simulation and 1.70$\times$ on real-world tasks, with negligible success rate degradation.
>
---
#### [replaced 022] SCRIPT: Scalable Diffusion Policy with Multi-stage Training for Language-driven Physics-based Humanoid Control
- **分类: cs.GR; cs.LG; cs.RO**

- **简介: 该论文属于语言驱动的物理人形控制任务，旨在解决语义表达与物理可行性之间的矛盾。提出SCRIPT框架，结合多阶段训练和扩散模型，提升指令遵循、运动质量和控制稳定性。**

- **链接: [https://arxiv.org/pdf/2605.22894](https://arxiv.org/pdf/2605.22894)**

> **作者:** Jingyan Zhang; Han Liang; Ruichi Zhang; Bin Li; Juze Zhang; Xin Chen; Jingya Wang; Lan Xu; Jingyi Yu
>
> **备注:** Project page: this https URL
>
> **摘要:** Controlling physics-based humanoids from natural-language instructions is a critical step toward general-purpose embodied agents. However, existing methods remain constrained by a tension between semantic expressiveness and physical feasibility, often failing to jointly achieve faithful instruction following, high-quality motion, and stable long-horizon control. We propose SCRIPT, a scalable diffusion policy with a multi-stage training framework for language-driven physics-based humanoid control. The core of SCRIPT is a Joint Action-State-Text Diffusion Transformer (JAST-DiT), which represents actions, physical states, and text as dedicated token streams and couples them through joint attention, enabling direct interaction between language semantics and control dynamics. To stabilize autoregressive control, we introduce a nonlinear history conditioning mechanism, which preserves the dense recent context and samples increasingly sparse cues from long-term history. Beyond supervised imitation pre-training, we propose a post-training stage, further improving the performance using Reinforcement Learning with Hybrid Rewards (RLHR). By injecting learnable noise into the flow-sampling process, RLHR effectively improves motion quality and instruction following within closed-loop simulations using hybrid physical feedback and text rewards. Quantitative evaluations demonstrate that SCRIPT outperforms prior state-of-the-art methods, with gains across text alignment, motion quality, and physical realism metrics. Furthermore, scaling studies on the 1200-hour MotionMillion dataset demonstrate consistent performance gains with model scaling, highlighting SCRIPT's robust scalability for large-scale pre-training. Our code will be publicly available for future research.
>
---
#### [replaced 023] Is VLA Reasoning Faithful? Probing Safety of Chain-of-Causation in Autonomous Driving Models
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 该论文研究自动驾驶模型的因果推理可靠性，分析VLA模型在100种场景中的表现，发现其推理与实际不符，提出安全架构以提升可信度。**

- **链接: [https://arxiv.org/pdf/2605.17268](https://arxiv.org/pdf/2605.17268)**

> **作者:** Nicanor Mayumu; Xiaoheng Deng; Patrick Mukala
>
> **备注:** Accept (Poster), CVPR 2026 Workshop DriveX NonArchival Track
>
> **摘要:** We present the first systematic study of faithfulness in Vision-Language-Action (VLA) driving models, analyzing 300 Alpamayo-R1-10B inferences across 100 diverse PhysicalAI-AV scenarios. Our main finding is that output natural-language rationales with trajectories may be significantly unfaithful: (i) overall reasoning fidelity is only 42.5%, with Chain-of-Causation matching scene reality less than half the time; (ii) 94 missed pedestrians in one-third of pedestrian-relevant scenes; (iii) 97.7% trajectory fragility under mild visual perturbations; and (iv) only 48.3% mean reasoning-action consistency, with 53.3% of inferences exhibiting low consistency, including 37.9% of stop-claimed cases where the model continues instead. We formalize faithfulness information-theoretically, define entity and action fidelity with verification criteria, and outline a four-component safety architecture aligned with these results.
>
---
#### [replaced 024] INSIGHT: INference-time Sequence Introspection for Generating Help Triggers in Vision-Language-Action Models
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于视觉-语言-动作模型的可靠性任务，旨在解决模型在执行中缺乏自我反思机制的问题。通过分析token级不确定性信号，训练模型生成求助触发器，提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.01389](https://arxiv.org/pdf/2510.01389)**

> **作者:** Ulas Berk Karli; Ziyao Shangguan; Tesca FItzgerald
>
> **摘要:** Recent Vision-Language-Action (VLA) models show strong generalization capabilities, yet they lack introspective mechanisms for anticipating failures and requesting help from a human supervisor. We present \textbf{INSIGHT}, a learning framework for leveraging token-level uncertainty signals to predict when a VLA should request help. Using $\pi_0$-FAST as the underlying model, we extract per-token \emph{entropy}, \emph{log-probability}, and Dirichlet-based estimates of \emph{aleatoric and epistemic uncertainty}, and train compact transformer classifiers to map these sequences to help triggers. We explore supervision regimes for strong or weak supervision, and extensively compare them across in-distribution and out-of-distribution tasks. Our results show a trade-off: strong labels enable models to capture fine-grained uncertainty dynamics for reliable help detection, while weak labels, though noisier, still support competitive introspection when training and evaluation are aligned, offering a scalable path when dense annotation is impractical. Crucially, we find that modeling the temporal evolution of token-level uncertainty signals with transformers provides far greater predictive power than static sequence-level scores. This study provides the first systematic evaluation of uncertainty-based introspection in VLAs, opening future avenues for active learning and for real-time error mitigation through selective human intervention.
>
---
#### [replaced 025] PRISM-SLAM: Probabilistic Ray-Grounded Inference for Scale-aware Metric SLAM
- **分类: cs.RO**

- **简介: 该论文提出PRISM-SLAM，解决单目SLAM的尺度模糊和动态环境跟踪问题，通过融合视觉基础模型先验，实现精准、实时的度量定位与建图。**

- **链接: [https://arxiv.org/pdf/2605.19257](https://arxiv.org/pdf/2605.19257)**

> **作者:** Eunsoo Im
>
> **摘要:** Monocular SLAM historically suffers from scale ambiguity and tracking failure in dynamic environments. While recent vision foundation models (VFMs) provide remarkable zero-shot depth priors, naively integrating these deterministic predictions ignores predictive uncertainty and frame-to-frame scale inconsistencies. We propose PRISM-SLAM, a real-time framework that rigorously integrates VFM priors into a structured Bayesian factor graph to achieve scale-aware, metric-consistent localization and mapping. Specifically, we introduce a Plücker Ray-Distance Factor to anchor monocular observations in absolute space within a globally consistent metric coordinate system, mathematically resolving scale drift by making the metric scale Fisher-identifiable. To handle environmental dynamics, we derive an epistemic uncertainty proxy from temporal depth consistency and formulate a Dynamic Scene Uncertainty Gating (DSUG) mechanism. This soft-gating approach probabilistically down-weights dynamic distractors without incurring the heavy computational overhead associated with traditional semantic segmentation masks. By employing a multi-process architecture that asynchronously processes VFM inference and geometric tracking, PRISM-SLAM provides verified metric output at 30 FPS using solely RGB input, bridging the gap between foundation models and real-world robotic applications. Evaluated on the TUM RGB-D and 7-Scenes benchmarks, PRISM-SLAM achieves a metric $SE(3)$ Absolute Trajectory Error (ATE) nearly identical to its oracle-aligned $Sim(3)$ error. This demonstrates that our system can produce deployment-ready metric trajectories by delivering robust metric SLAM solutions without any post-hoc scale correction. Project page: this https URL
>
---
#### [replaced 026] OHP-RL: Online Human Preference as Guidance in Reinforcement Learning for Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文提出OHP-RL框架，用于机器人操作中的强化学习，解决人类干预信息利用不足的问题。通过在线人类偏好引导策略学习，提升任务成功率与稳定性。**

- **链接: [https://arxiv.org/pdf/2605.15971](https://arxiv.org/pdf/2605.15971)**

> **作者:** Yunyang Mo; Jian Li; Qiwei Wu; Yihang Kang; Renjing Xu
>
> **摘要:** While reinforcement learning (RL) enables robots to acquire skills autonomously, its real-world deployment is severely limited by inefficient and unsafe exploration. Human-in-the-loop interventions offer a practical solution, yet existing methods typically exploit these interventions as auxiliary training signals, without fully capturing the richer information they provide about when and how autonomy should be guided. Human interventions often encode relative preferences over behavior under safety and task constraints, rather than prescribing exact actions to imitate. Motivated by this perspective, we propose Online Human Preference as Guidance in Reinforcement Learning (OHP-RL), a framework that leverages human interventions as preference information to guide policy learning. OHP-RL introduces a state-dependent preference gate that adaptively regulates when and to what extent human interventions should shape policy learning. This design enables the agent to benefit from intermittent and imperfect human feedback while preserving autonomous exploration and stable policy optimization. We evaluate OHP-RL on three challenging real-world contact-rich manipulation tasks on a Franka robot. Across all tasks, OHP-RL consistently achieves strong success rates, faster convergence, and substantially lower human intervention effort than prior approaches. Moreover, the learned policies exhibit more stable and human-aligned behavior throughout training.
>
---
#### [replaced 027] Logic-Guided Socially-aware Robot Navigation World Model
- **分类: cs.RO**

- **简介: 该论文属于社会机器人导航任务，解决LLM在动态人类空间中导航时的不可预测和不安全问题。通过引入NaviWM，结合结构化世界模型与逻辑推理，提升导航的社交合规性和物理安全性。**

- **链接: [https://arxiv.org/pdf/2510.23509](https://arxiv.org/pdf/2510.23509)**

> **作者:** Weizheng Wang; Obi Ike; Soyun Choi; Sungeun Hong; Aniket Bera; Byung-Cheol Min
>
> **摘要:** Social robot navigation increasingly relies on large language models for reasoning, path planning, and enabling movement in dynamic human spaces. However, relying solely on LLMs for planning often leads to unpredictable and unsafe behaviors, especially in dynamic human spaces, due to limited physical grounding and weak logical consistency. In this work, we introduce NaviWM, a socially-aware robot Navigation World Model that augments LLM reasoning with a structured world model and a logic-driven chain-of-thought process. NaviWM consists of two main components: (1) a spatial-temporal world model that captures the positions, velocities, and activities of agents in the environment, and (2) a deductive reasoning module that guides LLMs through a multi-step, logic-based inference process. This integration enables the robot to generate navigation decisions that are both socially compliant and physically safe, under well-defined constraints such as personal space, collision avoidance, and timing. Unlike previous methods based on prompting or fine-tuning, NaviWM encodes social norms as first-order logic, enabling interpretable and verifiable reasoning. Experiments show that NaviWM improves success rates and reduces social violations, particularly in crowded environments. These results demonstrate the benefit of combining formal reasoning with LLMs for robust social navigation. Additional experimental details and demo videos for this work can be found at: this https URL.
>
---
#### [replaced 028] Approximating Safety Feedback Without a Safety Oracle via Model Predictive Control
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于机器人安全控制任务，解决无安全反馈时的安全决策问题。通过模拟器构建安全函数代理，利用MPPI算法验证动作安全性，避免手动标注。**

- **链接: [https://arxiv.org/pdf/2510.20955](https://arxiv.org/pdf/2510.20955)**

> **作者:** Jeff Pflueger; Michael Everett
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Safe decision-making algorithms for control of mobile robots often require the existence of feedback to verify the safety of proposed actions. This feedback is assumed to be directly available during the development or deployment of the control system. It can take the form of either an explicit constraint formulation or a set of hand-labeled safety data, both of which can be inaccurate or time consuming to produce. Many recently developed simulators can handle complex interactions and varied environments. These environments have implicit safety constraints that may be hard to model. By leveraging one of these simulators, we can construct a proxy for a safety function that bypasses the need for hand designed feedback in capturing these constraints. We present an algorithm that approximates safety by using reversibility and a positive-invariance assumption on the unsafe state space. This method employs the Model-Predictive Path Integral algorithm (MPPI) to establish this reversibility and verify a proposed action. First the action is projected via the simulator to a future state. Then if MPPI can find a path back to a previous state in the trajectory, that state is guaranteed to be outside the unsafe (positive invariant) set. Experimental results demonstrate that the proposed algorithm can approximate the performance of a safety oracle while avoiding classification of unsafe states as safe.
>
---
#### [replaced 029] NeuralTouch: Neural Descriptors for Precise Sim-to-Real Tactile Robot Control
- **分类: cs.RO**

- **简介: 该论文属于机器人抓取任务，解决sim-to-real tactile控制精度问题。通过融合NDF与触觉传感，提出NeuralTouch框架，提升抓取准确性和鲁棒性。**

- **链接: [https://arxiv.org/pdf/2510.20390](https://arxiv.org/pdf/2510.20390)**

> **作者:** Yijiong Lin; Bowen Deng; Keju Pu; Chenghua Lu; Max Yang; Efi Psomopoulou; Nathan F. Lepora
>
> **摘要:** Grasping accuracy is a critical prerequisite for precise object manipulation, often requiring careful alignment between the robot hand and object. Neural Descriptor Fields (NDF) offer a promising vision-based method to generate grasping poses that generalize across object categories. However, NDF alone can produce inaccurate poses due to imperfect camera calibration, incomplete point clouds, and object variability. Meanwhile, tactile sensing enables more precise contact, but existing approaches typically learn policies limited to simple, predefined contact geometries. In this work, we introduce NeuralTouch, a multimodal framework that integrates NDF and tactile sensing to enable accurate, generalizable grasping through gentle physical interaction. Our approach leverages NDF to implicitly represent the target contact geometry, from which a deep reinforcement learning (RL) policy is trained to refine the grasp using tactile feedback. This policy is conditioned on the neural descriptors and does not require explicit specification of contact types. We validate NeuralTouch through ablation studies in simulation and zero-shot transfer to real-world manipulation tasks--such as peg-out-in-hole and bottle lid opening--without additional fine-tuning. Results show that NeuralTouch significantly improves grasping accuracy and robustness over baseline methods, offering a general framework for precise, contact-rich robotic manipulation.
>
---
#### [replaced 030] HeLoM: Hierarchical Learning for Whole-Body Loco-Manipulation by a Hexapod Robot
- **分类: cs.RO**

- **简介: 该论文提出HeLoM框架，解决六足机器人整体协调推动物体的问题，通过分层学习实现稳定高效的全身体操控制。**

- **链接: [https://arxiv.org/pdf/2509.23651](https://arxiv.org/pdf/2509.23651)**

> **作者:** Xinrong Yang; Peizhuo Li; Hongyi Li; Yifeng Peng; Arhaan Jain; Junkai Lu; Linnan Chang; Yuhong Cao; Yifeng Zhang; Ge Sun; Guillaume Sartoretti
>
> **摘要:** In nature, animals often need to move/manipulate objects comparable in weight/size to their own bodies. Compared to grasping and carrying, pushing provides a more straightforward and efficient non-prehensile manipulation strategy, avoiding complex grasp design while leveraging direct contact to regulate an object's pose during interaction. Achieving effective pushing, however, requires both sufficient manipulation capability and stable whole-body coordination, which is particularly challenging when dealing with heavy or irregular objects. To address these challenges, we propose HeLoM, a learning-based hierarchical whole-body manipulation framework for hexapod robots that exploits coordinated multi-limb control and is applicable to multi-legged robotic systems. Inspired by the cooperative strategies of multi-legged insects, our framework leverages multiple contact points and high degrees of freedom to enable efficient and dynamic whole-body coordination during object interaction. HeLoM's high-level planner plans pushing behaviors, while its low-level controller maintains locomotion stability and generates dynamically consistent joint actions. This design enables the robot to maintain balance while executing continuous and controllable pushing behaviors through coordinated foreleg interaction and supportive hind-leg propulsion. We validate the effectiveness of HeLoM through both simulation and real-world experiments. Results show that our framework can stably push objects of varying sizes and unknown physical properties to designated goal poses in the real world.
>
---
#### [replaced 031] Learning from Trials and Errors: Reflective Test-Time Planning for Embodied LLMs
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.RO**

- **简介: 该论文研究 embodied LLMs 的任务规划问题，旨在提升机器人在部署中的反思与学习能力。通过引入反射式测试时规划，结合行动中与行动后的反思机制，提高任务执行效果与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.21198](https://arxiv.org/pdf/2602.21198)**

> **作者:** Yining Hong; Huang Huang; Manling Li; Li Fei-Fei; Leonidas Guibas; Jiajun Wu; Yejin Choi
>
> **摘要:** Embodied LLMs endow robots with high-level task reasoning, but they cannot reflect on what went wrong or why, turning deployment into a sequence of independent trials where mistakes repeat rather than accumulate into experience. Drawing upon human reflective practitioners, we introduce Reflective Test-Time Planning, which integrates two modes of reflection: \textit{reflection-in-action}, where the agent uses test-time scaling to generate and score multiple candidate actions using internal reflections before execution; and \textit{reflection-on-action}, which uses test-time training to update both its internal reflection model and its action policy based on external reflections after execution. We also include retrospective reflection, allowing the agent to re-evaluate earlier decisions and perform model updates with hindsight for proper long-horizon credit assignment. Experiments on our newly-designed Long-Horizon Household benchmark and MuJoCo Cupboard Fitting benchmark show significant gains over baseline models, with zero-shot generalization to photorealistic HM3D environments and real-robot experiments on a Franka Panda arm. Ablations confirm that reflection-in-action and reflection-on-action are mutually dependent, and that retrospective reflection achieves better credit assignment than step-wise external feedback at lower computational overhead. Qualitative analyses further highlight behavioral correction through reflection.
>
---
#### [replaced 032] World-VLA-Loop: Closed-Loop Learning of Video World Model and VLA Policy
- **分类: cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决VLA政策在真实世界中训练成本高、动作跟随不准确的问题。通过构建闭环学习框架，提升模型性能并减少物理交互依赖。**

- **链接: [https://arxiv.org/pdf/2602.06508](https://arxiv.org/pdf/2602.06508)**

> **作者:** Xiaokang Liu; Zechen Bai; Hai Ci; Kevin Yuchen Ma; Mike Zheng Shou
>
> **备注:** 16 pages, 9 figures
>
> **摘要:** Reinforcement learning (RL) can refine Vision-Language-Action (VLA) policies beyond behavior cloning, but real-world RL remains expensive due to extensive rollouts, resets, supervision, and safety risks. Action-conditioned video world models offer an option to train in virtual environments, yet they exhibit imprecise action following, particularly on subtle near-success failures. Besides, they lack native reward signals for RL. Computing rewards based on inaccurate visual predictions remain unreliable. We introduce World-VLA-Loop, structured around two foundational designs and a higher-level co-evolving paradigm. We first curate SANS, dedicatedly mixing successful and near-success trajectories to improve action-outcome alignment. Then, we train a state-aware video world model that jointly predicts future frames and binary rewards from diffusion latents. It couples reward estimation to the generator rather than a separate module, and in turn, benefits visual prediction. Since VLA behavior shifts during RL, a fixed simulator can misalign with the updated policy, World-VLA-Loop therefore closes the loop by using the refined world model for iterative VLA post-training while feeding rollouts from each improved policy back to augment and fine-tune the world model. Across simulation and real-robot experiments, World-VLA-Loop substantially improves VLA performance while reducing reliance on costly physical interaction.
>
---
#### [replaced 033] Generalizable Vision-Language Few-Shot Adaptation with Predictive Prompts and Negative Learning
- **分类: cs.CV; cs.AI; cs.GR; cs.RO**

- **简介: 该论文属于视觉-语言少样本适应任务，解决负类信号处理不足的问题。提出SCAN框架，通过自适应负样本路由、对比提示和自适应融合提升模型性能。**

- **链接: [https://arxiv.org/pdf/2505.11758](https://arxiv.org/pdf/2505.11758)**

> **作者:** Sriram Mandalika
>
> **摘要:** Few-shot adaptation of vision-language models remains fundamentally limited by how negative class signals are handled at inference. Existing methods apply uniform negative suppression across all queries, ignoring that the most damaging confusions are query-specific and shift with support-set geometry. We introduce SCAN (Selective Confusion-Aware Negatives), a framework that addresses this gap through three targeted contributions. In inference, query-adaptive negative routing restricts suppression to the top-K most confusable classes per query, requiring zero additional parameters. Generic negative text templates are replaced with LLM-bootstrapped contrastive prompts that describe discriminative attributes between confusable class pairs, sharpening the textual decision boundary where it matters most. A parameter-free adaptive fusion weight estimated from support-set Fisher discriminability removes the need for manual tuning of the vision-language trade-off. Evaluated across 11 standard benchmarks, SCAN consistently outperforms prior prompt-based and adapter-based methods by an average of 4.61% at 16-shot, with gains of up to 7.70% on fine-grained datasets where inter-class confusion is most severe. SCAN also generalizes strongly under distribution shift, improving by 2.95% on average across four ImageNet OOD variants, and maintains robust performance under significant label noise, with accuracy under 50% label corruption still exceeding the clean baseline of the strongest competing method.
>
---
#### [replaced 034] Language Movement Primitives: Grounding Language Models in Robot Motion
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决从自然语言指令到机器人运动的映射问题。通过结合视觉语言模型和动态运动基元，提出LMP框架，实现零样本机器人操作。**

- **链接: [https://arxiv.org/pdf/2602.02839](https://arxiv.org/pdf/2602.02839)**

> **作者:** Yinlong Dai; Benjamin A. Christie; Daniel J. Evans; Dylan P. Losey; Simon Stepputtis
>
> **摘要:** Enabling robots to perform novel manipulation tasks from natural language instructions remains a fundamental challenge in robotics, despite significant progress in generalized problem solving with foundational models. Large vision and language models (VLMs) are capable of processing high-dimensional input data for visual scene and language understanding, as well as decomposing tasks into a sequence of logical steps; however, they struggle to ground those steps in embodied robot motion. On the other hand, robotics foundation models output action commands, but require in-domain fine-tuning or experience before they are able to perform novel tasks successfully. At its core, there still remains the fundamental challenge of connecting abstract task reasoning with low-level motion control. To address this disconnect, we propose Language Movement Primitives (LMPs), a framework that grounds VLM reasoning in Dynamic Movement Primitive (DMP) parameterization. Our key insight is that DMPs provide a small number of interpretable parameters, and VLMs can set these parameters to specify diverse, continuous, and stable trajectories. Put another way: VLMs can reason over free-form natural language task descriptions, and semantically ground their desired motions into DMPs -- bridging the gap between high-level task reasoning and low-level position and velocity control. Building on this combination of VLMs and DMPs, we formulate our LMP pipeline for zero-shot robot manipulation that effectively completes tabletop manipulation problems by generating a sequence of DMP motions. Across 31 real-world manipulation tasks, we show that LMP achieves 65% task success as compared to 35% for the best performing baseline. See videos at our website: this https URL
>
---
#### [replaced 035] Kilometer-Scale GNSS-Denied UAV Navigation via Heightmap Gradients: A Winning System from the SPRIN-D Challenge
- **分类: cs.RO**

- **简介: 该论文属于GNSS-denied环境下无人机自主导航任务，解决长距离飞行中的定位漂移问题。通过融合LiDAR高度图与先验地图，实现精准定位与路径规划。**

- **链接: [https://arxiv.org/pdf/2510.01348](https://arxiv.org/pdf/2510.01348)**

> **作者:** Michal Werner; David Čapek; Tomáš Musil; Ondřej Franěk; Tomáš Báča; Martin Saska
>
> **备注:** 8 pages
>
> **摘要:** Reliable long-range flight of unmanned aerial vehicles (UAVs) in GNSS-denied environments is challenging: integrating odometry leads to drift, loop closures are unavailable in previously unseen areas and embedded platforms provide limited computational power. We present a fully onboard UAV system developed for the SPRIN-D Funke Fully Autonomous Flight Challenge, which required 9 km long-range waypoint navigation below 25 m AGL (Above Ground Level) without GNSS or prior dense mapping. The system integrates perception, mapping, planning, and control with a lightweight drift-correction method that matches LiDAR-derived local heightmaps to a prior geo-data heightmap via gradient-template matching and fuses the evidence with odometry in a clustered particle filter. Deployed during the competition, the system executed kilometer-scale flights across urban, forest, and open-field terrain and reduced drift substantially relative to raw odometry, while running in real time on CPU-only hardware. We describe the system architecture, the localization pipeline, and the competition evaluation, and we report practical insights from field deployment that inform the design of GNSS-denied UAV autonomy.
>
---
#### [replaced 036] RoboManipBaselines: A Unified Framework for Imitation Learning in Robotic Manipulation across Real and Simulation Environments
- **分类: cs.RO**

- **简介: 该论文提出RoboManipBaselines框架，用于机器人操作中的模仿学习。解决仿真与真实环境下的模仿学习问题，实现数据收集、训练与评估的统一流程。**

- **链接: [https://arxiv.org/pdf/2509.17057](https://arxiv.org/pdf/2509.17057)**

> **作者:** Masaki Murooka; Tomohiro Motoda; Ryoichi Nakajo; Hanbit Oh; Koshi Makihara; Keisuke Shirai; Tetsuya Ogata; Yukiyasu Domae
>
> **备注:** Added a Limitations section in response to comments from reviewers
>
> **摘要:** We present RoboManipBaselines, an open-source software framework for imitation learning research in robotic manipulation. The framework supports the entire imitation learning pipeline, including data collection, policy training, and rollout, across both simulation and real-world environments. Its design emphasizes integration through a consistent workflow, generality across diverse environments and robot platforms, extensibility for easily adding new robots, tasks, and policies, and reproducibility through evaluations using publicly available datasets. RoboManipBaselines systematically implements the core components of imitation learning: environment, dataset, and policy. Through a unified interface, the framework supports multiple simulators and real robot environments, as well as multimodal sensors and a wide variety of policy models. We further present benchmark evaluations in both simulation and real-world environments and introduce several research applications, including data augmentation, integration with tactile models, interactive robotic systems, 3D sensing evaluation, and hardware extensions. These results demonstrate that RoboManipBaselines provides a useful foundation for advancing research and experimental validation in robotic manipulation using imitation learning. this https URL
>
---
#### [replaced 037] Few-Shot Neural Differentiable Simulator: Real-to-Sim Rigid-Contact Modeling
- **分类: cs.RO**

- **简介: 该论文属于机器人学习任务，旨在解决物理模拟与真实世界数据不匹配的问题。通过结合分析模型与图神经网络，实现少量真实数据下的高精度仿真与优化。**

- **链接: [https://arxiv.org/pdf/2603.06218](https://arxiv.org/pdf/2603.06218)**

> **作者:** Zhenhao Huang; Siyuan Luo; Bingyang Zhou; Ziqiu Zeng; Jason Pho; Fan Shi
>
> **备注:** Accepted in ICRA 2026
>
> **摘要:** Accurate physics simulation is essential for robotic learning and control, yet analytical simulators often fail to capture complex contact dynamics, while learning-based simulators typically require large amounts of costly real-world data. To bridge this gap, we propose a few-shot real-to-sim approach that combines the physical consistency of analytical formulations with the representational capacity of graph neural network (GNN)-based models. Using only a small amount of real-world data, our method calibrates analytical simulators to generate large-scale synthetic datasets that capture diverse contact interactions. On this foundation, we introduce a mesh-based GNN that implicitly models rigid-body forward dynamics and derive surrogate gradients for collision detection, achieving full differentiability. Experimental results demonstrate that our approach enables learning-based simulators to outperform differentiable baselines in replicating real-world trajectories. In addition, the differentiable design supports gradient-based optimization, which we validate through simulation-based policy learning in multi-object interaction scenarios. Extensive experiments show that our framework not only improves simulation fidelity with minimal supervision but also increases the efficiency of policy learning. Taken together, these findings suggest that differentiable simulation with few-shot real-world grounding provides a powerful direction for advancing future robotic manipulation and control.
>
---
#### [replaced 038] LIBERO-PRO: Towards Robust and Fair Evaluation of Vision-Language-Action Models Beyond Memorization
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言-动作模型评估任务，旨在解决现有基准LIBERO因记忆效应导致性能高估的问题。通过引入LIBERO-PRO，系统性测试模型在多种扰动下的表现，揭示模型依赖记忆而非理解的缺陷。**

- **链接: [https://arxiv.org/pdf/2510.03827](https://arxiv.org/pdf/2510.03827)**

> **作者:** Xueyang Zhou; Yangming Xu; Guiyao Tie; Yongchao Chen; Guowen Zhang; Duanfeng Chu; Pan Zhou; Lichao Sun
>
> **备注:** 10 pages,7 figures, 0 tables
>
> **摘要:** LIBERO has emerged as a widely adopted benchmark for evaluating Vision-Language-Action (VLA) models; however, its current training and evaluation settings are problematic, often leading to inflated performance estimates and preventing fair model comparison. To address these issues, we introduce LIBERO-PRO, an extended LIBERO benchmark that systematically evaluates model performance under reasonable perturbations across four dimensions: manipulated objects, initial states, task instructions, and environments. Experimental results reveal that, although existing models achieve over 90% accuracy under the standard LIBERO evaluation, their performance collapses to 0.0% under our generalized setting. Crucially, this discrepancy exposes the models' reliance on rote memorization of action sequences and environment layouts from the training set, rather than genuine task understanding or environmental perception. For instance, models persist in executing grasping actions when the target object is replaced with irrelevant items, and their outputs remain unchanged even when given corrupted instructions or even messy tokens. These findings expose the severe flaws in current evaluation practices, and we call on the community to abandon misleading methodologies in favor of robust assessments of model generalization and comprehension. Our code is available at: this https URL.
>
---
#### [replaced 039] Altitude-Adaptive Vision-Only Geo-Localization for UAVs in GPS-Denied Environments
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于无人机视觉定位任务，解决GPS拒收环境下因高度变化导致的尺度不匹配问题。通过估计相对高度并调整图像尺度，提升视觉定位精度。**

- **链接: [https://arxiv.org/pdf/2602.23872](https://arxiv.org/pdf/2602.23872)**

> **作者:** Xingyu Shao; Mengfan He; Chunyu Li; Liangzheng Sun; Ziyang Meng
>
> **摘要:** To address the scale mismatch caused by large altitude variations in UAV visual place recognition, we propose a monocular vision-only altitude-adaptive geo-localization framework. The method first estimates relative altitude from a single downward-looking image by transforming the input into the frequency domain and formulating altitude estimation as a regression-as-classification (RAC) problem. The estimated altitude is then used to crop the query image to a canonical scale, after which a classification-then-retrieval visual place recognition module performs coarse localization. To improve retrieval robustness under varying image quality, we further introduce a quality-adaptive margin classifier (QAMC) and refine the final location by weighted coordinate estimation over the top retrieved candidates. Experiments on two synthetic datasets and two real-flight datasets show that the relative altitude estimation (RAE) module yields clear overall improvements in downstream retrieval performance under significant altitude changes. With our visual place recognition module, altitude adaptation improves average R@1 and R@5 by 41.50 and 56.83 percentage points, respectively, compared with using the same retrieval pipeline without altitude normalization, and the full system runs at 13.3 frames/s on the reported workstation hardware. These results indicate that relative altitude estimation provides an effective scale prior for cross-altitude UAV geo-localization and supports GPS-denied coarse initialization without auxiliary range sensors or temporal inputs.
>
---
#### [replaced 040] Lidar Scan Registration Robust to Extreme Motions
- **分类: cs.RO**

- **简介: 该论文属于激光雷达点云配准任务，旨在解决极端运动下配准失效的问题。通过考虑轨迹不确定性和环境几何，提升算法鲁棒性，在极端加速度下显著降低误差。**

- **链接: [https://arxiv.org/pdf/2105.01215](https://arxiv.org/pdf/2105.01215)**

> **作者:** Simon-Pierre Deschênes; Dominic Baril; Vladimír Kubelka; Philippe Giguère; François Pomerleau
>
> **备注:** 8 pages, 8 figures, published in 2021 18th Conference on Robots and Vision (CRV), Burnaby, Canada
>
> **摘要:** Registration algorithms, such as Iterative Closest Point (ICP), have proven effective in mobile robot localization algorithms over the last decades. However, they are susceptible to failure when a robot sustains extreme velocities and accelerations. For example, this kind of motion can happen after a collision, causing a point cloud to be heavily skewed. While point cloud de-skewing methods have been explored in the past to increase localization and mapping accuracy, these methods still rely on highly accurate odometry systems or ideal navigation conditions. In this paper, we present a method taking into account the remaining motion uncertainties of the trajectory used to de-skew a point cloud along with the environment geometry to increase the robustness of current registration algorithms. We compare our method to three other solutions in a test bench producing 3D maps with peak accelerations of 200 m/s^2 and 800 rad/s^2. In these extreme scenarios, we demonstrate that our method decreases the error by 9.26 % in translation and by 21.84 % in rotation. The proposed method is generic enough to be integrated to many variants of weighted ICP without adaptation and supports localization robustness in harsher terrains.
>
---
