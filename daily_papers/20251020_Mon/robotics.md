# 机器人 cs.RO

- **最新发布 28 篇**

- **更新 23 篇**

## 最新发布

#### [new 001] Few-Shot Demonstration-Driven Task Coordination and Trajectory Execution for Multi-Robot Systems
- **分类: cs.RO**

- **简介: 该论文研究多机器人系统的少样本任务协调与轨迹执行，旨在减少对大量演示数据的依赖。作者提出DDACE框架，解耦时空因素，结合时序图网络与高斯过程，实现跨任务泛化。实验验证了其在多种协作场景下的有效性。**

- **链接: [http://arxiv.org/pdf/2510.15686v1](http://arxiv.org/pdf/2510.15686v1)**

> **作者:** Taehyeon Kim; Vishnunandan L. N. Venkatesh; Byung-Cheol Min
>
> **摘要:** In this paper, we propose a novel few-shot learning framework for multi-robot systems that integrate both spatial and temporal elements: Few-Shot Demonstration-Driven Task Coordination and Trajectory Execution (DDACE). Our approach leverages temporal graph networks for learning task-agnostic temporal sequencing and Gaussian Processes for spatial trajectory modeling, ensuring modularity and generalization across various tasks. By decoupling temporal and spatial aspects, DDACE requires only a small number of demonstrations, significantly reducing data requirements compared to traditional learning from demonstration approaches. To validate our proposed framework, we conducted extensive experiments in task environments designed to assess various aspects of multi-robot coordination-such as multi-sequence execution, multi-action dynamics, complex trajectory generation, and heterogeneous configurations. The experimental results demonstrate that our approach successfully achieves task execution under few-shot learning conditions and generalizes effectively across dynamic and diverse settings. This work underscores the potential of modular architectures in enhancing the practicality and scalability of multi-robot systems in real-world applications. Additional materials are available at https://sites.google.com/view/ddace.
>
---
#### [new 002] DexCanvas: Bridging Human Demonstrations and Robot Learning for Dexterous Manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文提出DexCanvas，旨在解决机器人灵巧操作中缺乏大规模真实与物理一致数据的问题。通过结合真实人类示范与仿真，构建包含7000小时交互的大规模数据集，覆盖21类操作，提供精细的接触与力信息，推动机器人学习与技能迁移。**

- **链接: [http://arxiv.org/pdf/2510.15786v1](http://arxiv.org/pdf/2510.15786v1)**

> **作者:** Xinyue Xu; Jieqiang Sun; Jing; Dai; Siyuan Chen; Lanjie Ma; Ke Sun; Bin Zhao; Jianbo Yuan; Yiwen Lu
>
> **摘要:** We present DexCanvas, a large-scale hybrid real-synthetic human manipulation dataset containing 7,000 hours of dexterous hand-object interactions seeded from 70 hours of real human demonstrations, organized across 21 fundamental manipulation types based on the Cutkosky taxonomy. Each entry combines synchronized multi-view RGB-D, high-precision mocap with MANO hand parameters, and per-frame contact points with physically consistent force profiles. Our real-to-sim pipeline uses reinforcement learning to train policies that control an actuated MANO hand in physics simulation, reproducing human demonstrations while discovering the underlying contact forces that generate the observed object motion. DexCanvas is the first manipulation dataset to combine large-scale real demonstrations, systematic skill coverage based on established taxonomies, and physics-validated contact annotations. The dataset can facilitate research in robotic manipulation learning, contact-rich control, and skill transfer across different hand morphologies.
>
---
#### [new 003] LVI-Q: Robust LiDAR-Visual-Inertial-Kinematic Odometry for Quadruped Robots Using Tightly-Coupled and Efficient Alternating Optimization
- **分类: cs.RO**

- **简介: 该论文研究四足机器人在复杂环境中的自主导航，针对现有SLAM系统易漂移的问题，提出紧耦合的多传感器融合里程计LVI-Q。通过交替优化视觉-惯性-运动与激光-惯性-运动信息，提升定位鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.15220v1](http://arxiv.org/pdf/2510.15220v1)**

> **作者:** Kevin Christiansen Marsim; Minho Oh; Byeongho Yu; Seungjae Lee; I Made Aswin Nahrendra; Hyungtae Lim; Hyun Myung
>
> **备注:** 8 Pages, 9 Figures
>
> **摘要:** Autonomous navigation for legged robots in complex and dynamic environments relies on robust simultaneous localization and mapping (SLAM) systems to accurately map surroundings and localize the robot, ensuring safe and efficient operation. While prior sensor fusion-based SLAM approaches have integrated various sensor modalities to improve their robustness, these algorithms are still susceptible to estimation drift in challenging environments due to their reliance on unsuitable fusion strategies. Therefore, we propose a robust LiDAR-visual-inertial-kinematic odometry system that integrates information from multiple sensors, such as a camera, LiDAR, inertial measurement unit (IMU), and joint encoders, for visual and LiDAR-based odometry estimation. Our system employs a fusion-based pose estimation approach that runs optimization-based visual-inertial-kinematic odometry (VIKO) and filter-based LiDAR-inertial-kinematic odometry (LIKO) based on measurement availability. In VIKO, we utilize the footpreintegration technique and robust LiDAR-visual depth consistency using superpixel clusters in a sliding window optimization. In LIKO, we incorporate foot kinematics and employ a point-toplane residual in an error-state iterative Kalman filter (ESIKF). Compared with other sensor fusion-based SLAM algorithms, our approach shows robust performance across public and longterm datasets.
>
---
#### [new 004] VDRive: Leveraging Reinforced VLA and Diffusion Policy for End-to-end Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文提出VDRive，面向端到端自动驾驶任务，解决动态环境与极端场景下的决策鲁棒性问题。结合视觉语言动作模型与扩散策略，通过状态-动作建模实现可解释、上下文与几何感知的驾驶决策。**

- **链接: [http://arxiv.org/pdf/2510.15446v1](http://arxiv.org/pdf/2510.15446v1)**

> **作者:** Ziang Guo; Zufeng Zhang
>
> **备注:** 1st version
>
> **摘要:** In autonomous driving, dynamic environment and corner cases pose significant challenges to the robustness of ego vehicle's state understanding and decision making. We introduce VDRive, a novel pipeline for end-to-end autonomous driving that explicitly models state-action mapping to address these challenges, enabling interpretable and robust decision making. By leveraging the advancement of the state understanding of the Vision Language Action Model (VLA) with generative diffusion policy-based action head, our VDRive guides the driving contextually and geometrically. Contextually, VLA predicts future observations through token generation pre-training, where the observations are represented as discrete codes by a Conditional Vector Quantized Variational Autoencoder (CVQ-VAE). Geometrically, we perform reinforcement learning fine-tuning of the VLA to predict future trajectories and actions based on current driving conditions. VLA supplies the current state tokens and predicted state tokens for the action policy head to generate hierarchical actions and trajectories. During policy training, a learned critic evaluates the actions generated by the policy and provides gradient-based feedback, forming an actor-critic framework that enables a reinforcement-based policy learning pipeline. Experiments show that our VDRive achieves state-of-the-art performance in the Bench2Drive closed-loop benchmark and nuScenes open-loop planning.
>
---
#### [new 005] Freehand 3D Ultrasound Imaging: Sim-in-the-Loop Probe Pose Optimization via Visual Servoing
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文研究自由手三维超声成像中的探头位姿估计问题，旨在降低对昂贵设备的依赖并提升重建精度。作者提出一种基于轻量相机和视觉伺服的仿真闭环方法，通过纹理匹配修复遮挡区域，并在模拟环境中优化位姿估计，实现高精度三维重建。**

- **链接: [http://arxiv.org/pdf/2510.15668v1](http://arxiv.org/pdf/2510.15668v1)**

> **作者:** Yameng Zhang; Dianye Huang; Max Q. -H. Meng; Nassir Navab; Zhongliang Jiang
>
> **摘要:** Freehand 3D ultrasound (US) imaging using conventional 2D probes offers flexibility and accessibility for diverse clinical applications but faces challenges in accurate probe pose estimation. Traditional methods depend on costly tracking systems, while neural network-based methods struggle with image noise and error accumulation, compromising reconstruction precision. We propose a cost-effective and versatile solution that leverages lightweight cameras and visual servoing in simulated environments for precise 3D US imaging. These cameras capture visual feedback from a textured planar workspace. To counter occlusions and lighting issues, we introduce an image restoration method that reconstructs occluded regions by matching surrounding texture patterns. For pose estimation, we develop a simulation-in-the-loop approach, which replicates the system setup in simulation and iteratively minimizes pose errors between simulated and real-world observations. A visual servoing controller refines the alignment of camera views, improving translational estimation by optimizing image alignment. Validations on a soft vascular phantom, a 3D-printed conical model, and a human arm demonstrate the robustness and accuracy of our approach, with Hausdorff distances to the reference reconstructions of 0.359 mm, 1.171 mm, and 0.858 mm, respectively. These results confirm the method's potential for reliable freehand 3D US reconstruction.
>
---
#### [new 006] Adaptive Cost-Map-based Path Planning in Partially Unknown Environments with Movable Obstacles
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文针对未知环境中可移动障碍物的导航问题，提出一种基于自适应代价地图的路径规划方法。通过识别潜在可动物体并动态调整局部代价，结合进度监控实现重规划，提升了机器人在复杂室内场景下的通行成功率，适用于资源受限的救援机器人。**

- **链接: [http://arxiv.org/pdf/2510.15336v1](http://arxiv.org/pdf/2510.15336v1)**

> **作者:** Liviu-Mihai Stan; Ranulfo Bezerra; Shotaro Kojima; Tsige Tadesse Alemayoh; Satoshi Tadokoro; Masashi Konyo; Kazunori Ohno
>
> **摘要:** Reliable navigation in disaster-response and other unstructured indoor settings requires robots not only to avoid obstacles but also to recognise when those obstacles can be pushed aside. We present an adaptive, LiDAR and odometry-based path-planning framework that embeds this capability into the ROS2 Nav2 stack. A new Movable Obstacles Layer labels all LiDAR returns missing from a prior static map as tentatively movable and assigns a reduced traversal cost. A companion Slow-Pose Progress Checker monitors the ratio of commanded to actual velocity; when the robot slows appreciably, the local cost is raised from light to heavy, and on a stall to lethal, prompting the global planner to back out and re-route. Gazebo evaluations on a Scout Mini, spanning isolated objects and cluttered corridors, show higher goal-reach rates and fewer deadlocks than a no-layer baseline, with traversal times broadly comparable. Because the method relies only on planar scans and CPU-level computation, it suits resource-constrained search and rescue robots and integrates into heterogeneous platforms with minimal engineering. Overall, the results indicate that interaction-aware cost maps are a lightweight, ROS2-native extension for navigating among potentially movable obstacles in unstructured settings. The full implementation will be released as open source athttps://costmap-namo.github.io.
>
---
#### [new 007] A Generalized Sylvester-Fermat-Torricelli problem with application in disaster relief operations by UAVs
- **分类: cs.RO; math.OC**

- **简介: 该论文针对灾害救援中无人机部署问题，提出一种考虑风力和无人机异质性的移动站选址数学模型。通过推广Sylvester问题为SFT问题，优化数据采集效率，显著减少操作时间，提升救援效能。**

- **链接: [http://arxiv.org/pdf/2510.15229v1](http://arxiv.org/pdf/2510.15229v1)**

> **作者:** Sina Kazemdehbashi; Yanchao Liu; Boris S. Mordukhovich
>
> **摘要:** Natural and human-made disasters can cause severe devastation and claim thousands of lives worldwide. Therefore, developing efficient methods for disaster response and management is a critical task for relief teams. One of the most essential components of effective response is the rapid collection of information about affected areas, damages, and victims. More data translates into better coordination, faster rescue operations, and ultimately, more lives saved. However, in some disasters, such as earthquakes, the communication infrastructure is often partially or completely destroyed, making it extremely difficult for victims to send distress signals and for rescue teams to locate and assist them in time. Unmanned Aerial Vehicles (UAVs) have emerged as valuable tools in such scenarios. In particular, a fleet of UAVs can be dispatched from a mobile station to the affected area to facilitate data collection and establish temporary communication networks. Nevertheless, real-world deployment of UAVs faces several challenges, with adverse weather conditions--especially wind--being among the most significant. To address this, we develop a novel mathematical framework to determine the optimal location of a mobile UAV station while explicitly accounting for the heterogeneity of the UAVs and the effect of wind. In particular, we generalize the Sylvester problem to introduce the Sylvester-Fermat-Torricelli (SFT) problem, which captures complex factors such as wind influence, UAV heterogeneity, and back-and-forth motion within a unified framework. The proposed framework enhances the practicality of UAV-based disaster response planning by accounting for real-world factors such as wind and UAV heterogeneity. Experimental results demonstrate that it can reduce wasted operational time by up to 84%, making post-disaster missions significantly more efficient and effective.
>
---
#### [new 008] PolyFly: Polytopic Optimal Planning for Collision-Free Cable-Suspended Aerial Payload Transportation
- **分类: cs.RO**

- **简介: 该论文研究空中机器人通过悬索运载货物时的快速无碰撞路径规划。针对传统方法过于保守的问题，提出PolyFly方法，将飞行器、缆绳、负载及环境建模为带姿态的多面体，结合对偶理论优化轨迹，实现更快速、精确的避障飞行。**

- **链接: [http://arxiv.org/pdf/2510.15226v1](http://arxiv.org/pdf/2510.15226v1)**

> **作者:** Mrunal Sarvaiya; Guanrui Li; Giuseppe Loianno
>
> **摘要:** Aerial transportation robots using suspended cables have emerged as versatile platforms for disaster response and rescue operations. To maximize the capabilities of these systems, robots need to aggressively fly through tightly constrained environments, such as dense forests and structurally unsafe buildings, while minimizing flight time and avoiding obstacles. Existing methods geometrically over-approximate the vehicle and obstacles, leading to conservative maneuvers and increased flight times. We eliminate these restrictions by proposing PolyFly, an optimal global planner which considers a non-conservative representation for aerial transportation by modeling each physical component of the environment, and the robot (quadrotor, cable and payload), as independent polytopes. We further increase the model accuracy by incorporating the attitude of the physical components by constructing orientation-aware polytopes. The resulting optimal control problem is efficiently solved by converting the polytope constraints into smooth differentiable constraints via duality theory. We compare our method against the existing state-of-the-art approach in eight maze-like environments and show that PolyFly produces faster trajectories in each scenario. We also experimentally validate our proposed approach on a real quadrotor with a suspended payload, demonstrating the practical reliability and accuracy of our method.
>
---
#### [new 009] RM-RL: Role-Model Reinforcement Learning for Precise Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文研究精确机器人操作任务，解决高精度操作中专家示范难获取、离线强化学习数据效率低的问题。提出RM-RL框架，通过角色模型自动生成标签，结合在线与离线训练，提升学习效率与精度。**

- **链接: [http://arxiv.org/pdf/2510.15189v1](http://arxiv.org/pdf/2510.15189v1)**

> **作者:** Xiangyu Chen; Chuhao Zhou; Yuxi Liu; Jianfei Yang
>
> **摘要:** Precise robot manipulation is critical for fine-grained applications such as chemical and biological experiments, where even small errors (e.g., reagent spillage) can invalidate an entire task. Existing approaches often rely on pre-collected expert demonstrations and train policies via imitation learning (IL) or offline reinforcement learning (RL). However, obtaining high-quality demonstrations for precision tasks is difficult and time-consuming, while offline RL commonly suffers from distribution shifts and low data efficiency. We introduce a Role-Model Reinforcement Learning (RM-RL) framework that unifies online and offline training in real-world environments. The key idea is a role-model strategy that automatically generates labels for online training data using approximately optimal actions, eliminating the need for human demonstrations. RM-RL reformulates policy learning as supervised training, reducing instability from distribution mismatch and improving efficiency. A hybrid training scheme further leverages online role-model data for offline reuse, enhancing data efficiency through repeated sampling. Extensive experiments show that RM-RL converges faster and more stably than existing RL methods, yielding significant gains in real-world manipulation: 53% improvement in translation accuracy and 20% in rotation accuracy. Finally, we demonstrate the successful execution of a challenging task, precisely placing a cell plate onto a shelf, highlighting the framework's effectiveness where prior methods fail.
>
---
#### [new 010] Educational SoftHand-A: Building an Anthropomorphic Hand with Soft Synergies using LEGO MINDSTORMS
- **分类: cs.RO**

- **简介: 该论文设计了一款基于LEGO MINDSTORMS的仿人形软体机械手Educational SoftHand-A，旨在用于教育场景。它采用欠驱动腱传动与软协同机制，实现自适应抓取，仅用标准乐高零件和家用设备即可搭建测试，帮助儿童学习现代机器人技术。**

- **链接: [http://arxiv.org/pdf/2510.15638v1](http://arxiv.org/pdf/2510.15638v1)**

> **作者:** Jared K. Lepora; Haoran Li; Efi Psomopoulou; Nathan F. Lepora
>
> **备注:** 6 pages. Accepted at IROS 2025
>
> **摘要:** This paper introduces an anthropomorphic robot hand built entirely using LEGO MINDSTORMS: the Educational SoftHand-A, a tendon-driven, highly-underactuated robot hand based on the Pisa/IIT SoftHand and related hands. To be suitable for an educational context, the design is constrained to use only standard LEGO pieces with tests using common equipment available at home. The hand features dual motors driving an agonist/antagonist opposing pair of tendons on each finger, which are shown to result in reactive fine control. The finger motions are synchonized through soft synergies, implemented with a differential mechanism using clutch gears. Altogether, this design results in an anthropomorphic hand that can adaptively grasp a broad range of objects using a simple actuation and control mechanism. Since the hand can be constructed from LEGO pieces and uses state-of-the-art design concepts for robotic hands, it has the potential to educate and inspire children to learn about the frontiers of modern robotics.
>
---
#### [new 011] ASBI: Leveraging Informative Real-World Data for Active Black-Box Simulator Tuning
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究黑盒仿真器参数调优，解决真实数据信息不足导致的后验估计困难。提出ASBI框架，结合主动数据采集与神经后验估计，通过最大化信息增益提升参数估计精度，并在真实机器人任务中验证有效性。**

- **链接: [http://arxiv.org/pdf/2510.15331v1](http://arxiv.org/pdf/2510.15331v1)**

> **作者:** Gahee Kim; Takamitsu Matsubara
>
> **摘要:** Black-box simulators are widely used in robotics, but optimizing their parameters remains challenging due to inaccessible likelihoods. Simulation-Based Inference (SBI) tackles this issue using simulation-driven approaches, estimating the posterior from offline real observations and forward simulations. However, in black-box scenarios, preparing observations that contain sufficient information for parameter estimation is difficult due to the unknown relationship between parameters and observations. In this work, we present Active Simulation-Based Inference (ASBI), a parameter estimation framework that uses robots to actively collect real-world online data to achieve accurate black-box simulator tuning. Our framework optimizes robot actions to collect informative observations by maximizing information gain, which is defined as the expected reduction in Shannon entropy between the posterior and the prior. While calculating information gain requires the likelihood, which is inaccessible in black-box simulators, our method solves this problem by leveraging Neural Posterior Estimation (NPE), which leverages a neural network to learn the posterior estimator. Three simulation experiments quantitatively verify that our method achieves accurate parameter estimation, with posteriors sharply concentrated around the true parameters. Moreover, we show a practical application using a real robot to estimate the simulation parameters of cubic particles corresponding to two real objects, beads and gravel, with a bucket pouring action.
>
---
#### [new 012] Traversability-aware Consistent Situational Graphs for Indoor Localization and Mapping
- **分类: cs.RO**

- **简介: 该论文属室内定位与建图任务，旨在解决现有场景图在房间分割中因传感器视场限制导致的过分割与欠分割问题。作者提出一种考虑机器人可通行性的房间分割方法，提升语义一致性与优化效率。**

- **链接: [http://arxiv.org/pdf/2510.15319v1](http://arxiv.org/pdf/2510.15319v1)**

> **作者:** Jeewon Kim; Minho Oh; Hyun Myung
>
> **备注:** Accepted by RiTA 2024
>
> **摘要:** Scene graphs enhance 3D mapping capabilities in robotics by understanding the relationships between different spatial elements, such as rooms and objects. Recent research extends scene graphs to hierarchical layers, adding and leveraging constraints across these levels. This approach is tightly integrated with pose-graph optimization, improving both localization and mapping accuracy simultaneously. However, when segmenting spatial characteristics, consistently recognizing rooms becomes challenging due to variations in viewpoints and limited field of view (FOV) of sensors. For example, existing real-time approaches often over-segment large rooms into smaller, non-functional spaces that are not useful for localization and mapping due to the time-dependent method. Conversely, their voxel-based room segmentation method often under-segment in complex cases like not fully enclosed 3D space that are non-traversable for ground robots or humans, leading to false constraints in pose-graph optimization. We propose a traversability-aware room segmentation method that considers the interaction between robots and surroundings, with consistent feasibility of traversability information. This enhances both the semantic coherence and computational efficiency of pose-graph optimization. Improved performance is demonstrated through the re-detection frequency of the same rooms in a dataset involving repeated traversals of the same space along the same path, as well as the optimization time consumption.
>
---
#### [new 013] Towards Automated Chicken Deboning via Learning-based Dynamically-Adaptive 6-DoF Multi-Material Cutting
- **分类: cs.RO**

- **简介: 该论文研究自动化鸡肩去骨任务，解决多材料、易变形关节中精确6自由度切割难题。提出带力反馈的强化学习策略，开发开源模拟器与实物测试平台，实现从仿真到真实的零样本迁移，显著提升去骨成功率并减少碰骨风险。**

- **链接: [http://arxiv.org/pdf/2510.15376v1](http://arxiv.org/pdf/2510.15376v1)**

> **作者:** Zhaodong Yang; Ai-Ping Hu; Harish Ravichandar
>
> **备注:** 8 Pages, 8 figures
>
> **摘要:** Automating chicken shoulder deboning requires precise 6-DoF cutting through a partially occluded, deformable, multi-material joint, since contact with the bones presents serious health and safety risks. Our work makes both systems-level and algorithmic contributions to train and deploy a reactive force-feedback cutting policy that dynamically adapts a nominal trajectory and enables full 6-DoF knife control to traverse the narrow joint gap while avoiding contact with the bones. First, we introduce an open-source custom-built simulator for multi-material cutting that models coupling, fracture, and cutting forces, and supports reinforcement learning, enabling efficient training and rapid prototyping. Second, we design a reusable physical testbed to emulate the chicken shoulder: two rigid "bone" spheres with controllable pose embedded in a softer block, enabling rigorous and repeatable evaluation while preserving essential multi-material characteristics of the target problem. Third, we train and deploy a residual RL policy, with discretized force observations and domain randomization, enabling robust zero-shot sim-to-real transfer and the first demonstration of a learned policy that debones a real chicken shoulder. Our experiments in our simulator, on our physical testbed, and on real chicken shoulders show that our learned policy reliably navigates the joint gap and reduces undesired bone/cartilage contact, resulting in up to a 4x improvement over existing open-loop cutting baselines in terms of success rate and bone avoidance. Our results also illustrate the necessity of force feedback for safe and effective multi-material cutting. The project website is at https://sites.google.com/view/chickendeboning-2026.
>
---
#### [new 014] Dynamic Recalibration in LiDAR SLAM: Integrating AI and Geometric Methods with Real-Time Feedback Using INAF Fusion
- **分类: cs.RO**

- **简介: 该论文研究LiDAR SLAM任务，旨在提升复杂环境下的定位与建图精度。提出INAF融合模块，结合AI与几何里程计，通过实时环境反馈动态调整注意力权重，增强系统适应性与测量准确性。**

- **链接: [http://arxiv.org/pdf/2510.15803v1](http://arxiv.org/pdf/2510.15803v1)**

> **作者:** Zahra Arjmandi; Gunho Sohn
>
> **备注:** 9 pages, 9 figures
>
> **摘要:** This paper presents a novel fusion technique for LiDAR Simultaneous Localization and Mapping (SLAM), aimed at improving localization and 3D mapping using LiDAR sensor. Our approach centers on the Inferred Attention Fusion (INAF) module, which integrates AI with geometric odometry. Utilizing the KITTI dataset's LiDAR data, INAF dynamically adjusts attention weights based on environmental feedback, enhancing the system's adaptability and measurement accuracy. This method advances the precision of both localization and 3D mapping, demonstrating the potential of our fusion technique to enhance autonomous navigation systems in complex scenarios.
>
---
#### [new 015] Autonomous Reactive Masonry Construction using Collaborative Heterogeneous Aerial Robots with Experimental Demonstration
- **分类: cs.RO**

- **简介: 该论文研究无人机协同砌筑任务，解决自主空中砌砖与粘合剂涂覆问题。开发了两种异构无人机，实现砖块精准放置与粘合剂自动涂抹，结合反应式规划、动态分配与视觉定位，完成全自主空中砌筑实验验证。**

- **链接: [http://arxiv.org/pdf/2510.15114v1](http://arxiv.org/pdf/2510.15114v1)**

> **作者:** Marios-Nektarios Stamatopoulos; Elias Small; Shridhar Velhal; Avijit Banerjee; George Nikolakopoulos
>
> **摘要:** This article presents a fully autonomous aerial masonry construction framework using heterogeneous unmanned aerial vehicles (UAVs), supported by experimental validation. Two specialized UAVs were developed for the task: (i) a brick-carrier UAV equipped with a ball-joint actuation mechanism for precise brick manipulation, and (ii) an adhesion UAV integrating a servo-controlled valve and extruder nozzle for accurate adhesion application. The proposed framework employs a reactive mission planning unit that combines a dependency graph of the construction layout with a conflict graph to manage simultaneous task execution, while hierarchical state machines ensure robust operation and safe transitions during task execution. Dynamic task allocation allows real-time adaptation to environmental feedback, while minimum-jerk trajectory generation ensures smooth and precise UAV motion during brick pickup and placement. Additionally, the brick-carrier UAV employs an onboard vision system that estimates brick poses in real time using ArUco markers and a least-squares optimization filter, enabling accurate alignment during construction. To the best of the authors' knowledge, this work represents the first experimental demonstration of fully autonomous aerial masonry construction using heterogeneous UAVs, where one UAV precisely places the bricks while another autonomously applies adhesion material between them. The experimental results supported by the video showcase the effectiveness of the proposed framework and demonstrate its potential to serve as a foundation for future developments in autonomous aerial robotic construction.
>
---
#### [new 016] Integration of a Variable Stiffness Link for Long-Reach Aerial Manipulation
- **分类: cs.RO**

- **简介: 该论文针对长距离空中操作中刚性或缆索连接导致精度低和扰动传递的问题，提出一种可变刚度连接机构（VSL），实现飞行器与机械臂间的刚柔切换。通过实验验证其在抗扰和定位精度上的优势，提升系统灵活性与安全性。**

- **链接: [http://arxiv.org/pdf/2510.15639v1](http://arxiv.org/pdf/2510.15639v1)**

> **作者:** Manuel J. Fernandez; Alejandro Suarez; Anibal Ollero; Matteo Fumagalli
>
> **摘要:** This paper presents the integration of a Variable Stiffness Link (VSL) for long-reach aerial manipulation, enabling adaptable mechanical coupling between an aerial multirotor platform and a dual-arm manipulator. Conventional long-reach manipulation systems rely on rigid or cable connections, which limit precision or transmit disturbances to the aerial vehicle. The proposed VSL introduces an adjustable stiffness mechanism that allows the link to behave either as a flexible rope or as a rigid rod, depending on task requirements. The system is mounted on a quadrotor equipped with the LiCAS dual-arm manipulator and evaluated through teleoperated experiments, involving external disturbances and parcel transportation tasks. Results demonstrate that varying the link stiffness significantly modifies the dynamic interaction between the UAV and the payload. The flexible configuration attenuates external impacts and aerodynamic perturbations, while the rigid configuration improves positional accuracy during manipulation phases. These results confirm that VSL enhances versatility and safety, providing a controllable trade-off between compliance and precision. Future work will focus on autonomous stiffness regulation, multi-rope configurations, cooperative aerial manipulation and user studies to further assess its impact on teleoperated and semi-autonomous aerial tasks.
>
---
#### [new 017] Perfect Prediction or Plenty of Proposals? What Matters Most in Planning for Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文研究自动驾驶中预测与规划的集成问题，探讨预测对规划性能的影响。发现完美预测未必提升规划效果，而高质量轨迹提议更为关键。提出以提议生成为核心的方法，在复杂场景中显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2510.15505v1](http://arxiv.org/pdf/2510.15505v1)**

> **作者:** Aron Distelzweig; Faris Janjoš; Oliver Scheel; Sirish Reddy Varra; Raghu Rajan; Joschka Boedecker
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Traditionally, prediction and planning in autonomous driving (AD) have been treated as separate, sequential modules. Recently, there has been a growing shift towards tighter integration of these components, known as Integrated Prediction and Planning (IPP), with the aim of enabling more informed and adaptive decision-making. However, it remains unclear to what extent this integration actually improves planning performance. In this work, we investigate the role of prediction in IPP approaches, drawing on the widely adopted Val14 benchmark, which encompasses more common driving scenarios with relatively low interaction complexity, and the interPlan benchmark, which includes highly interactive and out-of-distribution driving situations. Our analysis reveals that even access to perfect future predictions does not lead to better planning outcomes, indicating that current IPP methods often fail to fully exploit future behavior information. Instead, we focus on high-quality proposal generation, while using predictions primarily for collision checks. We find that many imitation learning-based planners struggle to generate realistic and plausible proposals, performing worse than PDM - a simple lane-following approach. Motivated by this observation, we build on PDM with an enhanced proposal generation method, shifting the emphasis towards producing diverse but realistic and high-quality proposals. This proposal-centric approach significantly outperforms existing methods, especially in out-of-distribution and highly interactive settings, where it sets new state-of-the-art results.
>
---
#### [new 018] Adaptive Legged Locomotion via Online Learning for Model Predictive Control
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文研究自适应足式机器人 locomotion 控制，旨在应对未知负载、地形和外力扰动。提出结合在线学习与模型预测控制的方法，通过随机傅里叶特征在线估计残差动力学，实现轨迹跟踪，具备低动态遗憾性，并在仿真中验证有效性。**

- **链接: [http://arxiv.org/pdf/2510.15626v1](http://arxiv.org/pdf/2510.15626v1)**

> **作者:** Hongyu Zhou; Xiaoyu Zhang; Vasileios Tzoumas
>
> **备注:** 9 pages
>
> **摘要:** We provide an algorithm for adaptive legged locomotion via online learning and model predictive control. The algorithm is composed of two interacting modules: model predictive control (MPC) and online learning of residual dynamics. The residual dynamics can represent modeling errors and external disturbances. We are motivated by the future of autonomy where quadrupeds will autonomously perform complex tasks despite real-world unknown uncertainty, such as unknown payload and uneven terrains. The algorithm uses random Fourier features to approximate the residual dynamics in reproducing kernel Hilbert spaces. Then, it employs MPC based on the current learned model of the residual dynamics. The model is updated online in a self-supervised manner using least squares based on the data collected while controlling the quadruped. The algorithm enjoys sublinear \textit{dynamic regret}, defined as the suboptimality against an optimal clairvoyant controller that knows how the residual dynamics. We validate our algorithm in Gazebo and MuJoCo simulations, where the quadruped aims to track reference trajectories. The Gazebo simulations include constant unknown external forces up to $12\boldsymbol{g}$, where $\boldsymbol{g}$ is the gravity vector, in flat terrain, slope terrain with $20\degree$ inclination, and rough terrain with $0.25m$ height variation. The MuJoCo simulations include time-varying unknown disturbances with payload up to $8~kg$ and time-varying ground friction coefficients in flat terrain.
>
---
#### [new 019] Nauplius Optimisation for Autonomous Hydrodynamics
- **分类: cs.RO; cs.NE**

- **简介: 该论文提出NOAH算法，属水下群控优化任务，旨在解决自主水下航行器在强流、低带宽等复杂环境中的可靠协同问题。受藤壶幼虫行为启发，引入水流感知、不可逆锚定与群体通信机制，提升水下探测的稳定性与能效。**

- **链接: [http://arxiv.org/pdf/2510.15350v1](http://arxiv.org/pdf/2510.15350v1)**

> **作者:** Shyalan Ramesh; Scott Mann; Alex Stumpf
>
> **摘要:** Autonomous Underwater vehicles must operate in strong currents, limited acoustic bandwidth, and persistent sensing requirements where conventional swarm optimisation methods are unreliable. This paper presents NOAH, a novel nature-inspired swarm optimisation algorithm that combines current-aware drift, irreversible settlement in persistent sensing nodes, and colony-based communication. Drawing inspiration from the behaviour of barnacle nauplii, NOAH addresses the critical limitations of existing swarm algorithms by providing hydrodynamic awareness, irreversible anchoring mechanisms, and colony-based communication capabilities essential for underwater exploration missions. The algorithm establishes a comprehensive foundation for scalable and energy-efficient underwater swarm robotics with validated performance analysis. Validation studies demonstrate an 86% success rate for permanent anchoring scenarios, providing a unified formulation for hydrodynamic constraints and irreversible settlement behaviours with an empirical study under flow.
>
---
#### [new 020] Improved Extended Kalman Filter-Based Disturbance Observers for Exoskeletons
- **分类: cs.RO**

- **简介: 该论文针对外骨骼系统中未知扰动影响控制性能的问题，提出两种改进的扩展卡尔曼滤波扰动观测器，通过权衡跟踪速度与不确定性，提升扰动估计精度，实验验证了方法在关节误差上的显著改善。**

- **链接: [http://arxiv.org/pdf/2510.15533v1](http://arxiv.org/pdf/2510.15533v1)**

> **作者:** Shilei Li; Dawei Shi; Makoto Iwasaki; Yan Ning; Hongpeng Zhou; Ling Shi
>
> **摘要:** The nominal performance of mechanical systems is often degraded by unknown disturbances. A two-degree-of-freedom control structure can decouple nominal performance from disturbance rejection. However, perfect disturbance rejection is unattainable when the disturbance dynamic is unknown. In this work, we reveal an inherent trade-off in disturbance estimation subject to tracking speed and tracking uncertainty. Then, we propose two novel methods to enhance disturbance estimation: an interacting multiple model extended Kalman filter-based disturbance observer and a multi-kernel correntropy extended Kalman filter-based disturbance observer. Experiments on an exoskeleton verify that the proposed two methods improve the tracking accuracy $36.3\%$ and $16.2\%$ in hip joint error, and $46.3\%$ and $24.4\%$ in knee joint error, respectively, compared to the extended Kalman filter-based disturbance observer, in a time-varying interaction force scenario, demonstrating the superiority of the proposed method.
>
---
#### [new 021] VO-DP: Semantic-Geometric Adaptive Diffusion Policy for Vision-Only Robotic Manipulation
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文研究视觉-动作策略学习任务，解决纯视觉机械臂操控中语义与几何特征融合问题。提出VO-DP方法，结合DINOv2语义与交替注意力几何特征，通过跨模态融合提升性能，在仿真与真实场景均显著优于基线方法。**

- **链接: [http://arxiv.org/pdf/2510.15530v1](http://arxiv.org/pdf/2510.15530v1)**

> **作者:** Zehao Ni; Yonghao He; Lingfeng Qian; Jilei Mao; Fa Fu; Wei Sui; Hu Su; Junran Peng; Zhipeng Wang; Bin He
>
> **摘要:** In the context of imitation learning, visuomotor-based diffusion policy learning is one of the main directions in robotic manipulation. Most of these approaches rely on point clouds as observation inputs and construct scene representations through point clouds feature learning, which enables them to achieve remarkable accuracy. However, the existing literature lacks an in-depth exploration of vision-only solutions that have significant potential. In this paper, we propose a Vision-Only and single-view Diffusion Policy learning method (VO-DP) that leverages pretrained visual foundation models to achieve effective fusion of semantic and geometric features. We utilize intermediate features from VGGT incorporating semantic features from DINOv2 and geometric features from Alternating Attention blocks. Features are fused via cross-attention and spatially compressed with a CNN to form the input to the policy head. Extensive experiments demonstrate that VO-DP not only outperforms the vision-only baseline DP significantly but also exhibits distinct performance trends against the point cloud-based method DP3: in simulation tasks, VO-DP achieves an average success rate of 64.6% on par with DP3 64.0% and far higher than DP 34.8%, while in real-world tasks, it reaches 87.9%, outperforming both DP3 67.5% and DP 11.2% by a notable margin. Further robustness evaluations confirm that VO-DP remains highly stable under varying conditions including color, size, background, and lighting. Lastly, we open-source a training library for robotic manipulation. Built on Accelerate, this library supports multi-machine and multi-GPU parallel training, as well as mixed precision training. It is compatible with visuomotor policies such as DP, DP3 and VO-DP, and also supports the RoboTwin simulator.
>
---
#### [new 022] Lagrange-Poincaré-Kepler Equations of Disturbed Space-Manipulator Systems in Orbit
- **分类: cs.RO**

- **简介: 该论文提出拉格朗日-庞加莱-开普勒方程（LPKE），用于建模轨道非惯性系下航天器-机械臂系统的耦合动力学。旨在解决姿态、轨道与机械臂运动耦合及外部扰动影响问题，建立了包含扰动耦合的闭式结构矩阵，适用于空间机器人自主控制与仿真。**

- **链接: [http://arxiv.org/pdf/2510.15199v1](http://arxiv.org/pdf/2510.15199v1)**

> **作者:** Borna Monazzah Moghaddam; Robin Chhabra
>
> **摘要:** This article presents an extension of the Lagrange-Poincare Equations (LPE) to model the dynamics of spacecraft-manipulator systems operating within a non-inertial orbital reference frame. Building upon prior formulations of LPE for vehicle-manipulator systems, the proposed framework, termed the Lagrange-Poincare-Kepler Equations (LPKE), incorporates the coupling between spacecraft attitude dynamics, orbital motion, and manipulator kinematics. The formalism combines the Euler-Poincare equations for the base spacecraft, Keplerian orbital dynamics for the reference frame, and reduced Euler-Lagrange equations for the manipulator's shape space, using an exponential joint parametrization. Leveraging the Lagrange-d'Alembert principle on principal bundles, we derive novel closed-form structural matrices that explicitly capture the effects of orbital disturbances and their dynamic coupling with the manipulator system. The LPKE framework also systematically includes externally applied, symmetry-breaking wrenches, allowing for immediate integration into hardware-in-the-loop simulations and model-based control architectures for autonomous robotic operations in the orbital environment. To illustrate the effectiveness of the proposed model and its numerical superiority, we present a simulation study analyzing orbital effects on a 7-degree-of-freedom manipulator mounted on a spacecraft.
>
---
#### [new 023] HEADER: Hierarchical Robot Exploration via Attention-Based Deep Reinforcement Learning with Expert-Guided Reward
- **分类: cs.RO**

- **简介: 该论文研究机器人自主探索任务，旨在提升大尺度环境下的探索效率。提出HEADER方法，结合分层图表示与注意力机制的深度强化学习，设计社区化全局图构建算法和专家引导的免参数奖励，实现高效、可扩展的探索，在仿真与真实大型场景中均表现优越。**

- **链接: [http://arxiv.org/pdf/2510.15679v1](http://arxiv.org/pdf/2510.15679v1)**

> **作者:** Yuhong Cao; Yizhuo Wang; Jingsong Liang; Shuhao Liao; Yifeng Zhang; Peizhuo Li; Guillaume Sartoretti
>
> **摘要:** This work pushes the boundaries of learning-based methods in autonomous robot exploration in terms of environmental scale and exploration efficiency. We present HEADER, an attention-based reinforcement learning approach with hierarchical graphs for efficient exploration in large-scale environments. HEADER follows existing conventional methods to construct hierarchical representations for the robot belief/map, but further designs a novel community-based algorithm to construct and update a global graph, which remains fully incremental, shape-adaptive, and operates with linear complexity. Building upon attention-based networks, our planner finely reasons about the nearby belief within the local range while coarsely leveraging distant information at the global scale, enabling next-best-viewpoint decisions that consider multi-scale spatial dependencies. Beyond novel map representation, we introduce a parameter-free privileged reward that significantly improves model performance and produces near-optimal exploration behaviors, by avoiding training objective bias caused by handcrafted reward shaping. In simulated challenging, large-scale exploration scenarios, HEADER demonstrates better scalability than most existing learning and non-learning methods, while achieving a significant improvement in exploration efficiency (up to 20%) over state-of-the-art baselines. We also deploy HEADER on hardware and validate it in complex, large-scale real-life scenarios, including a 300m*230m campus environment.
>
---
#### [new 024] GaussGym: An open-source real-to-sim framework for learning locomotion from pixels
- **分类: cs.RO; cs.AI; cs.GR**

- **简介: 该论文提出GaussGym，一个开源的实拍到仿真框架，旨在解决机器人从图像学习运动时仿真速度与视觉真实感难以兼顾的问题。通过集成3D高斯点阵渲染，在保持高视觉保真的同时实现超高速仿真，并支持多种真实场景数据导入，推动可扩展的机器人学习。**

- **链接: [http://arxiv.org/pdf/2510.15352v1](http://arxiv.org/pdf/2510.15352v1)**

> **作者:** Alejandro Escontrela; Justin Kerr; Arthur Allshire; Jonas Frey; Rocky Duan; Carmelo Sferrazza; Pieter Abbeel
>
> **摘要:** We present a novel approach for photorealistic robot simulation that integrates 3D Gaussian Splatting as a drop-in renderer within vectorized physics simulators such as IsaacGym. This enables unprecedented speed -- exceeding 100,000 steps per second on consumer GPUs -- while maintaining high visual fidelity, which we showcase across diverse tasks. We additionally demonstrate its applicability in a sim-to-real robotics setting. Beyond depth-based sensing, our results highlight how rich visual semantics improve navigation and decision-making, such as avoiding undesirable regions. We further showcase the ease of incorporating thousands of environments from iPhone scans, large-scale scene datasets (e.g., GrandTour, ARKit), and outputs from generative video models like Veo, enabling rapid creation of realistic training worlds. This work bridges high-throughput simulation and high-fidelity perception, advancing scalable and generalizable robot learning. All code and data will be open-sourced for the community to build upon. Videos, code, and data available at https://escontrela.me/gauss_gym/.
>
---
#### [new 025] Towards Robust Zero-Shot Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文研究零样本强化学习（zero-shot RL），旨在解决现有方法因OOD动作导致的表示偏差与表达能力不足问题。作者提出BREEZE框架，引入行为正则化、扩散模型策略提取和注意力架构，提升表示质量、学习稳定性与策略性能，在离线零样本设置下实现更优表现。**

- **链接: [http://arxiv.org/pdf/2510.15382v1](http://arxiv.org/pdf/2510.15382v1)**

> **作者:** Kexin Zheng; Lauriane Teyssier; Yinan Zheng; Yu Luo; Xiayuan Zhan
>
> **备注:** Neurips 2025, 36 pages, 18 figures
>
> **摘要:** The recent development of zero-shot reinforcement learning (RL) has opened a new avenue for learning pre-trained generalist policies that can adapt to arbitrary new tasks in a zero-shot manner. While the popular Forward-Backward representations (FB) and related methods have shown promise in zero-shot RL, we empirically found that their modeling lacks expressivity and that extrapolation errors caused by out-of-distribution (OOD) actions during offline learning sometimes lead to biased representations, ultimately resulting in suboptimal performance. To address these issues, we propose Behavior-REgularizEd Zero-shot RL with Expressivity enhancement (BREEZE), an upgraded FB-based framework that simultaneously enhances learning stability, policy extraction capability, and representation learning quality. BREEZE introduces behavioral regularization in zero-shot RL policy learning, transforming policy optimization into a stable in-sample learning paradigm. Additionally, BREEZE extracts the policy using a task-conditioned diffusion model, enabling the generation of high-quality and multimodal action distributions in zero-shot RL settings. Moreover, BREEZE employs expressive attention-based architectures for representation modeling to capture the complex relationships between environmental dynamics. Extensive experiments on ExORL and D4RL Kitchen demonstrate that BREEZE achieves the best or near-the-best performance while exhibiting superior robustness compared to prior offline zero-shot RL methods. The official implementation is available at: https://github.com/Whiterrrrr/BREEZE.
>
---
#### [new 026] CuSfM: CUDA-Accelerated Structure-from-Motion
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于三维重建任务，旨在解决传统SfM方法效率与精度难以兼顾的问题。作者提出cuSfM，利用CUDA加速特征提取与匹配，实现高效精确的相机位姿估计和全局一致建图，并开源PyCuSfM供研究使用。**

- **链接: [http://arxiv.org/pdf/2510.15271v1](http://arxiv.org/pdf/2510.15271v1)**

> **作者:** Jingrui Yu; Jun Liu; Kefei Ren; Joydeep Biswas; Rurui Ye; Keqiang Wu; Chirag Majithia; Di Zeng
>
> **摘要:** Efficient and accurate camera pose estimation forms the foundational requirement for dense reconstruction in autonomous navigation, robotic perception, and virtual simulation systems. This paper addresses the challenge via cuSfM, a CUDA-accelerated offline Structure-from-Motion system that leverages GPU parallelization to efficiently employ computationally intensive yet highly accurate feature extractors, generating comprehensive and non-redundant data associations for precise camera pose estimation and globally consistent mapping. The system supports pose optimization, mapping, prior-map localization, and extrinsic refinement. It is designed for offline processing, where computational resources can be fully utilized to maximize accuracy. Experimental results demonstrate that cuSfM achieves significantly improved accuracy and processing speed compared to the widely used COLMAP method across various testing scenarios, while maintaining the high precision and global consistency essential for offline SfM applications. The system is released as an open-source Python wrapper implementation, PyCuSfM, available at https://github.com/nvidia-isaac/pyCuSFM, to facilitate research and applications in computer vision and robotics.
>
---
#### [new 027] UrbanVerse: Scaling Urban Simulation by Watching City-Tour Videos
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属城市仿真任务，旨在解决现有模拟场景缺乏真实性和扩展性的问题。作者提出UrbanVerse系统，通过城市场景视频自动生成具物理交互能力的3D仿真环境，包含大规模资产库与生成管线，显著提升导航策略训练效果及现实迁移能力。**

- **链接: [http://arxiv.org/pdf/2510.15018v1](http://arxiv.org/pdf/2510.15018v1)**

> **作者:** Mingxuan Liu; Honglin He; Elisa Ricci; Wayne Wu; Bolei Zhou
>
> **备注:** Technical report. Project page: https://urbanverseproject.github.io/
>
> **摘要:** Urban embodied AI agents, ranging from delivery robots to quadrupeds, are increasingly populating our cities, navigating chaotic streets to provide last-mile connectivity. Training such agents requires diverse, high-fidelity urban environments to scale, yet existing human-crafted or procedurally generated simulation scenes either lack scalability or fail to capture real-world complexity. We introduce UrbanVerse, a data-driven real-to-sim system that converts crowd-sourced city-tour videos into physics-aware, interactive simulation scenes. UrbanVerse consists of: (i) UrbanVerse-100K, a repository of 100k+ annotated urban 3D assets with semantic and physical attributes, and (ii) UrbanVerse-Gen, an automatic pipeline that extracts scene layouts from video and instantiates metric-scale 3D simulations using retrieved assets. Running in IsaacSim, UrbanVerse offers 160 high-quality constructed scenes from 24 countries, along with a curated benchmark of 10 artist-designed test scenes. Experiments show that UrbanVerse scenes preserve real-world semantics and layouts, achieving human-evaluated realism comparable to manually crafted scenes. In urban navigation, policies trained in UrbanVerse exhibit scaling power laws and strong generalization, improving success by +6.3% in simulation and +30.1% in zero-shot sim-to-real transfer comparing to prior methods, accomplishing a 300 m real-world mission with only two interventions.
>
---
#### [new 028] Exploring Conditions for Diffusion models in Robotic Control
- **分类: cs.CV; cs.RO**

- **简介: 该论文研究扩散模型在机器人控制中的视觉表征应用，旨在解决直接使用文本条件导致性能不佳的问题。提出ORCA方法，引入可学习任务提示和视觉提示，实现任务自适应表征，在多个基准上达到最优性能。**

- **链接: [http://arxiv.org/pdf/2510.15510v1](http://arxiv.org/pdf/2510.15510v1)**

> **作者:** Heeseong Shin; Byeongho Heo; Dongyoon Han; Seungryong Kim; Taekyung Kim
>
> **备注:** Project page: https://orca-rc.github.io/
>
> **摘要:** While pre-trained visual representations have significantly advanced imitation learning, they are often task-agnostic as they remain frozen during policy learning. In this work, we explore leveraging pre-trained text-to-image diffusion models to obtain task-adaptive visual representations for robotic control, without fine-tuning the model itself. However, we find that naively applying textual conditions - a successful strategy in other vision domains - yields minimal or even negative gains in control tasks. We attribute this to the domain gap between the diffusion model's training data and robotic control environments, leading us to argue for conditions that consider the specific, dynamic visual information required for control. To this end, we propose ORCA, which introduces learnable task prompts that adapt to the control environment and visual prompts that capture fine-grained, frame-specific details. Through facilitating task-adaptive representations with our newly devised conditions, our approach achieves state-of-the-art performance on various robotic control benchmarks, significantly surpassing prior methods.
>
---
## 更新

#### [replaced 001] Self-supervised Multi-future Occupancy Forecasting for Autonomous Driving
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2407.21126v3](http://arxiv.org/pdf/2407.21126v3)**

> **作者:** Bernard Lange; Masha Itkina; Jiachen Li; Mykel J. Kochenderfer
>
> **备注:** Accepted to Robotics: Science and Systems (RSS) 2025
>
> **摘要:** Environment prediction frameworks are critical for the safe navigation of autonomous vehicles (AVs) in dynamic settings. LiDAR-generated occupancy grid maps (L-OGMs) offer a robust bird's-eye view for the scene representation, enabling self-supervised joint scene predictions while exhibiting resilience to partial observability and perception detection failures. Prior approaches have focused on deterministic L-OGM prediction architectures within the grid cell space. While these methods have seen some success, they frequently produce unrealistic predictions and fail to capture the stochastic nature of the environment. Additionally, they do not effectively integrate additional sensor modalities present in AVs. Our proposed framework, Latent Occupancy Prediction (LOPR), performs stochastic L-OGM prediction in the latent space of a generative architecture and allows for conditioning on RGB cameras, maps, and planned trajectories. We decode predictions using either a single-step decoder, which provides high-quality predictions in real-time, or a diffusion-based batch decoder, which can further refine the decoded frames to address temporal consistency issues and reduce compression losses. Our experiments on the nuScenes and Waymo Open datasets show that all variants of our approach qualitatively and quantitatively outperform prior approaches.
>
---
#### [replaced 002] Onboard Mission Replanning for Adaptive Cooperative Multi-Robot Systems
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.06094v2](http://arxiv.org/pdf/2506.06094v2)**

> **作者:** Elim Kwan; Rehman Qureshi; Liam Fletcher; Colin Laganier; Victoria Nockles; Richard Walters
>
> **备注:** 9 pages, 5 figures, 1 table
>
> **摘要:** Cooperative autonomous robotic systems have significant potential for executing complex multi-task missions across space, air, ground, and maritime domains. But they commonly operate in remote, dynamic and hazardous environments, requiring rapid in-mission adaptation without reliance on fragile or slow communication links to centralised compute. Fast, on-board replanning algorithms are therefore needed to enhance resilience. Reinforcement Learning shows strong promise for efficiently solving mission planning tasks when formulated as Travelling Salesperson Problems (TSPs), but existing methods: 1) are unsuitable for replanning, where agents do not start at a single location; 2) do not allow cooperation between agents; 3) are unable to model tasks with variable durations; or 4) lack practical considerations for on-board deployment. Here we define the Cooperative Mission Replanning Problem as a novel variant of multiple TSP with adaptations to overcome these issues, and develop a new encoder/decoder-based model using Graph Attention Networks and Attention Models to solve it effectively and efficiently. Using a simple example of cooperative drones, we show our replanner consistently (90% of the time) maintains performance within 10% of the state-of-the-art LKH3 heuristic solver, whilst running 85-370 times faster on a Raspberry Pi. This work paves the way for increased resilience in autonomous multi-agent systems.
>
---
#### [replaced 003] CLOVER: Context-aware Long-term Object Viewpoint- and Environment- Invariant Representation Learning
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2407.09718v3](http://arxiv.org/pdf/2407.09718v3)**

> **作者:** Dongmyeong Lee; Amanda Adkins; Joydeep Biswas
>
> **备注:** 8 pages, 3 figures, 8 tables
>
> **摘要:** Mobile service robots can benefit from object-level understanding of their environments, including the ability to distinguish object instances and re-identify previously seen instances. Object re-identification is challenging across different viewpoints and in scenes with significant appearance variation arising from weather or lighting changes. Existing works on object re-identification either focus on specific classes or require foreground segmentation. Further, these methods, along with object re-identification datasets, have limited consideration of challenges such as outdoor scenes and illumination changes. To address this problem, we introduce CODa Re-ID: an in-the-wild object re-identification dataset containing 1,037,814 observations of 557 objects across 8 classes under diverse lighting conditions and viewpoints. Further, we propose CLOVER, a representation learning method for object observations that can distinguish between static object instances without requiring foreground segmentation. We also introduce MapCLOVER, a method for scalably summarizing CLOVER descriptors for use in object maps and matching new observations to summarized descriptors. Our results show that CLOVER achieves superior performance in static object re-identification under varying lighting conditions and viewpoint changes and can generalize to unseen instances and classes.
>
---
#### [replaced 004] Real-time Recognition of Human Interactions from a Single RGB-D Camera for Socially-Aware Robot Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.24907v2](http://arxiv.org/pdf/2509.24907v2)**

> **作者:** Thanh Long Nguyen; Duc Phu Nguyen; Thanh Thao Ton Nu; Quan Le; Thuan Hoang Tran; Manh Duong Phung
>
> **摘要:** {Recognizing human interactions is essential for social robots as it enables them to navigate safely and naturally in shared environments. Conventional robotic systems however often focus on obstacle avoidance, neglecting social cues necessary for seamless human-robot interaction. To address this gap, we propose a framework to recognize human group interactions for socially aware navigation. Our method utilizes color and depth frames from a monocular RGB-D camera to estimate 3D human keypoints and positions. Principal component analysis (PCA) is then used to determine dominant interaction directions. The shoelace formula is finally applied to compute interest points and engagement areas. Extensive experiments have been conducted to evaluate the validity of the proposed method. The results show that our method is capable of recognizing group interactions across different scenarios with varying numbers of individuals. It also achieves high-speed performance, processing each frame in approximately 4 ms on a single-board computer used in robotic systems. The method is implemented as a ROS 2 package making it simple to integrate into existing navigation systems. Source code is available at https://github.com/thanhlong103/social-interaction-detector
>
---
#### [replaced 005] Contact-Aware Safety in Soft Robots Using High-Order Control Barrier and Lyapunov Functions
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2505.03841v3](http://arxiv.org/pdf/2505.03841v3)**

> **作者:** Kiwan Wong; Maximilian Stölzle; Wei Xiao; Cosimo Della Santina; Daniela Rus; Gioele Zardini
>
> **备注:** 8 pages
>
> **摘要:** Robots operating alongside people, particularly in sensitive scenarios such as aiding the elderly with daily tasks or collaborating with workers in manufacturing, must guarantee safety and cultivate user trust. Continuum soft manipulators promise safety through material compliance, but as designs evolve for greater precision, payload capacity, and speed, and increasingly incorporate rigid elements, their injury risk resurfaces. In this letter, we introduce a comprehensive High-Order Control Barrier Function (HOCBF) + High-Order Control Lyapunov Function (HOCLF) framework that enforces strict contact force limits across the entire soft-robot body during environmental interactions. Our approach combines a differentiable Piecewise Cosserat-Segment (PCS) dynamics model with a convex-polygon distance approximation metric, named Differentiable Conservative Separating Axis Theorem (DCSAT), based on the soft robot geometry to enable real-time, whole-body collision detection, resolution, and enforcement of the safety constraints. By embedding HOCBFs into our optimization routine, we guarantee safety, allowing, for instance, safe navigation in operational space under HOCLF-driven motion objectives. Extensive planar simulations demonstrate that our method maintains safety-bounded contacts while achieving precise shape and task-space regulation. This work thus lays a foundation for the deployment of soft robots in human-centric environments with provable safety and performance.
>
---
#### [replaced 006] Towards smart and adaptive agents for active sensing on edge devices
- **分类: cs.RO; cs.AI; eess.IV**

- **链接: [http://arxiv.org/pdf/2501.06262v2](http://arxiv.org/pdf/2501.06262v2)**

> **作者:** Devendra Vyas; Nikola Pižurica; Nikola Milović; Igor Jovančević; Miguel de Prado; Tim Verbelen
>
> **摘要:** TinyML has made deploying deep learning models on low-power edge devices feasible, creating new opportunities for real-time perception in constrained environments. However, the adaptability of such deep learning methods remains limited to data drift adaptation, lacking broader capabilities that account for the environment's underlying dynamics and inherent uncertainty. Deep learning's scaling laws, which counterbalance this limitation by massively up-scaling data and model size, cannot be applied when deploying on the Edge, where deep learning limitations are further amplified as models are scaled down for deployment on resource-constrained devices. This paper presents an innovative agentic system capable of performing on-device perception and planning, enabling active sensing on the edge. By incorporating active inference into our solution, our approach extends beyond deep learning capabilities, allowing the system to plan in dynamic environments while operating in real-time with a compact memory footprint of as little as 300 MB. We showcase our proposed system by creating and deploying a saccade agent connected to an IoT camera with pan and tilt capabilities on an NVIDIA Jetson embedded device. The saccade agent controls the camera's field of view following optimal policies derived from the active inference principles, simulating human-like saccadic motion for surveillance and robotics applications.
>
---
#### [replaced 007] MLFM: Multi-Layered Feature Maps for Richer Language Understanding in Zero-Shot Semantic Navigation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2507.07299v2](http://arxiv.org/pdf/2507.07299v2)**

> **作者:** Sonia Raychaudhuri; Enrico Cancelli; Tommaso Campari; Lamberto Ballan; Manolis Savva; Angel X. Chang
>
> **摘要:** Recent progress in large vision-language models has driven improvements in language-based semantic navigation, where an embodied agent must reach a target object described in natural language. Yet we still lack a clear, language-focused evaluation framework to test how well agents ground the words in their instructions. We address this gap by proposing LangNav, an open-vocabulary multi-object navigation dataset with natural language goal descriptions (e.g. 'go to the red short candle on the table') and corresponding fine-grained linguistic annotations (e.g., attributes: color=red, size=short; relations: support=on). These labels enable systematic evaluation of language understanding. To evaluate on this setting, we extend multi-object navigation task setting to Language-guided Multi-Object Navigation (LaMoN), where the agent must find a sequence of goals specified using language. Furthermore, we propose Multi-Layered Feature Map (MLFM), a novel method that builds a queryable, multi-layered semantic map from pretrained vision-language features and proves effective for reasoning over fine-grained attributes and spatial relations in goal descriptions. Experiments on LangNav show that MLFM outperforms state-of-the-art zero-shot mapping-based navigation baselines.
>
---
#### [replaced 008] LOPR: Latent Occupancy PRediction using Generative Models
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2210.01249v4](http://arxiv.org/pdf/2210.01249v4)**

> **作者:** Bernard Lange; Masha Itkina; Mykel J. Kochenderfer
>
> **备注:** We recommend referring to the peer-reviewed and updated version of this approach, available at arXiv:2407.21126
>
> **摘要:** Environment prediction frameworks are integral for autonomous vehicles, enabling safe navigation in dynamic environments. LiDAR generated occupancy grid maps (L-OGMs) offer a robust bird's eye-view scene representation that facilitates joint scene predictions without relying on manual labeling unlike commonly used trajectory prediction frameworks. Prior approaches have optimized deterministic L-OGM prediction architectures directly in grid cell space. While these methods have achieved some degree of success in prediction, they occasionally grapple with unrealistic and incorrect predictions. We claim that the quality and realism of the forecasted occupancy grids can be enhanced with the use of generative models. We propose a framework that decouples occupancy prediction into: representation learning and stochastic prediction within the learned latent space. Our approach allows for conditioning the model on other available sensor modalities such as RGB-cameras and high definition maps. We demonstrate that our approach achieves state-of-the-art performance and is readily transferable between different robotic platforms on the real-world NuScenes, Waymo Open, and a custom dataset we collected on an experimental vehicle platform.
>
---
#### [replaced 009] MotionScript: Natural Language Descriptions for Expressive 3D Human Motions
- **分类: cs.CV; cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2312.12634v5](http://arxiv.org/pdf/2312.12634v5)**

> **作者:** Payam Jome Yazdian; Rachel Lagasse; Hamid Mohammadi; Eric Liu; Li Cheng; Angelica Lim
>
> **备注:** Project webpage: https://pjyazdian.github.io/MotionScript
>
> **摘要:** We introduce MotionScript, a novel framework for generating highly detailed, natural language descriptions of 3D human motions. Unlike existing motion datasets that rely on broad action labels or generic captions, MotionScript provides fine-grained, structured descriptions that capture the full complexity of human movement including expressive actions (e.g., emotions, stylistic walking) and interactions beyond standard motion capture datasets. MotionScript serves as both a descriptive tool and a training resource for text-to-motion models, enabling the synthesis of highly realistic and diverse human motions from text. By augmenting motion datasets with MotionScript captions, we demonstrate significant improvements in out-of-distribution motion generation, allowing large language models (LLMs) to generate motions that extend beyond existing data. Additionally, MotionScript opens new applications in animation, virtual human simulation, and robotics, providing an interpretable bridge between intuitive descriptions and motion synthesis. To the best of our knowledge, this is the first attempt to systematically translate 3D motion into structured natural language without requiring training data.
>
---
#### [replaced 010] U-ARM : Ultra low-cost general teleoperation interface for robot manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.02437v3](http://arxiv.org/pdf/2509.02437v3)**

> **作者:** Yanwen Zou; Zhaoye Zhou; Chenyang Shi; Zewei Ye; Junda Huang; Yan Ding; Bo Zhao
>
> **摘要:** We propose U-Arm, a low-cost and rapidly adaptable leader-follower teleoperation framework designed to interface with most of commercially available robotic arms. Our system supports teleoperation through three structurally distinct 3D-printed leader arms that share consistent control logic, enabling seamless compatibility with diverse commercial robot configurations. Compared with previous open-source leader-follower interfaces, we further optimized both the mechanical design and servo selection, achieving a bill of materials (BOM) cost of only \$50.5 for the 6-DoF leader arm and \$56.8 for the 7-DoF version. To enhance usability, we mitigate the common challenge in controlling redundant degrees of freedom by %engineering methods mechanical and control optimizations. Experimental results demonstrate that U-Arm achieves 39\% higher data collection efficiency and comparable task success rates across multiple manipulation scenarios compared with Joycon, another low-cost teleoperation interface. We have open-sourced all CAD models of three configs and also provided simulation support for validating teleoperation workflows. We also open-sourced real-world manipulation data collected with U-Arm. The project website is https://github.com/MINT-SJTU/LeRobot-Anything-U-Arm.
>
---
#### [replaced 011] Learning to Capture Rocks using an Excavator: A Reinforcement Learning Approach with Guiding Reward Formulation
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2510.04168v2](http://arxiv.org/pdf/2510.04168v2)**

> **作者:** Amirmasoud Molaei; Mohammad Heravi; Reza Ghabcheloo
>
> **摘要:** Rock capturing with standard excavator buckets is a challenging task typically requiring the expertise of skilled operators. Unlike soil digging, it involves manipulating large, irregular rocks in unstructured environments where complex contact interactions with granular material make model-based control impractical. Existing autonomous excavation methods focus mainly on continuous media or rely on specialized grippers, limiting their applicability to real-world construction sites. This paper introduces a fully data-driven control framework for rock capturing that eliminates the need for explicit modeling of rock or soil properties. A model-free reinforcement learning agent is trained in the AGX Dynamics simulator using the Proximal Policy Optimization (PPO) algorithm and a guiding reward formulation. The learned policy outputs joint velocity commands directly to the boom, arm, and bucket of a CAT365 excavator model. Robustness is enhanced through extensive domain randomization of rock geometry, density, and mass, as well as the initial configurations of the bucket, rock, and goal position. To the best of our knowledge, this is the first study to develop and evaluate an RL-based controller for the rock capturing task. Experimental results show that the policy generalizes well to unseen rocks and varying soil conditions, achieving high success rates comparable to those of human participants while maintaining machine stability. These findings demonstrate the feasibility of learning-based excavation strategies for discrete object manipulation without requiring specialized hardware or detailed material models.
>
---
#### [replaced 012] General-Purpose Robotic Navigation via LVLM-Orchestrated Perception, Reasoning, and Acting
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2506.17462v2](http://arxiv.org/pdf/2506.17462v2)**

> **作者:** Bernard Lange; Anil Yildiz; Mansur Arief; Shehryar Khattak; Mykel Kochenderfer; Georgios Georgakis
>
> **摘要:** Developing general-purpose navigation policies for unknown environments remains a core challenge in robotics. Most existing systems rely on task-specific neural networks and fixed information flows, limiting their generalizability. Large Vision-Language Models (LVLMs) offer a promising alternative by embedding human-like knowledge for reasoning and planning, but prior LVLM-robot integrations have largely depended on pre-mapped spaces, hard-coded representations, and rigid control logic. We introduce the Agentic Robotic Navigation Architecture (ARNA), a general-purpose framework that equips an LVLM-based agent with a library of perception, reasoning, and navigation tools drawn from modern robotic stacks. At runtime, the agent autonomously defines and executes task-specific workflows that iteratively query modules, reason over multimodal inputs, and select navigation actions. This agentic formulation enables robust navigation and reasoning in previously unmapped environments, offering a new perspective on robotic stack design. Evaluated in Habitat Lab on the HM-EQA benchmark, ARNA outperforms state-of-the-art EQA-specific approaches. Qualitative results on RxR and custom tasks further demonstrate its ability to generalize across a broad range of navigation challenges.
>
---
#### [replaced 013] Scaling Multi Agent Reinforcement Learning for Underwater Acoustic Tracking via Autonomous Vehicles
- **分类: cs.RO; cs.AI; cs.DC; cs.PF**

- **链接: [http://arxiv.org/pdf/2505.08222v2](http://arxiv.org/pdf/2505.08222v2)**

> **作者:** Matteo Gallici; Ivan Masmitja; Mario Martín
>
> **摘要:** Autonomous vehicles (AV) offer a cost-effective solution for scientific missions such as underwater tracking. Recently, reinforcement learning (RL) has emerged as a powerful method for controlling AVs in complex marine environments. However, scaling these techniques to a fleet--essential for multi-target tracking or targets with rapid, unpredictable motion--presents significant computational challenges. Multi-Agent Reinforcement Learning (MARL) is notoriously sample-inefficient, and while high-fidelity simulators like Gazebo's LRAUV provide 100x faster-than-real-time single-robot simulations, they offer no significant speedup for multi-vehicle scenarios, making MARL training impractical. To address these limitations, we propose an iterative distillation method that transfers high-fidelity simulations into a simplified, GPU-accelerated environment while preserving high-level dynamics. This approach achieves up to a 30,000x speedup over Gazebo through parallelization, enabling efficient training via end-to-end GPU acceleration. Additionally, we introduce a novel Transformer-based architecture (TransfMAPPO) that learns multi-agent policies invariant to the number of agents and targets, significantly improving sample efficiency. Following large-scale curriculum learning conducted entirely on GPU, we perform extensive evaluations in Gazebo, demonstrating that our method maintains tracking errors below 5 meters over extended durations, even in the presence of multiple fast-moving targets. This work bridges the gap between large-scale MARL training and high-fidelity deployment, providing a scalable framework for autonomous fleet control in real-world sea missions.
>
---
#### [replaced 014] TAS: A Transit-Aware Strategy for Embodied Navigation with Non-Stationary Targets
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2403.09905v4](http://arxiv.org/pdf/2403.09905v4)**

> **作者:** Vishnu Sashank Dorbala; Bhrij Patel; Amrit Singh Bedi; Dinesh Manocha
>
> **备注:** 15 pages
>
> **摘要:** Embodied navigation methods commonly operate in static environments with stationary targets. In this work, we present a new algorithm for navigation in dynamic scenarios with non-stationary targets. Our novel Transit-Aware Strategy (TAS) enriches embodied navigation policies with object path information. TAS improves performance in non-stationary environments by rewarding agents for synchronizing their routes with target routes. To evaluate TAS, we further introduce Dynamic Object Maps (DOMs), a dynamic variant of node-attributed topological graphs with structured object transitions. DOMs are inspired by human habits to simulate realistic object routes on a graph. Our experiments show that on average, TAS improves agent Success Rate (SR) by 21.1 in non-stationary environments, while also generalizing better from static environments by 44.5% when measured by Relative Change in Success (RCS). We qualitatively investigate TAS-agent performance on DOMs and draw various inferences to help better model generalist navigation policies. To the best of our knowledge, ours is the first work that quantifies the adaptability of embodied navigation methods in non-stationary environments. Code and data for our benchmark will be made publicly available.
>
---
#### [replaced 015] Spatial Forcing: Implicit Spatial Representation Alignment for Vision-language-action Model
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.12276v2](http://arxiv.org/pdf/2510.12276v2)**

> **作者:** Fuhao Li; Wenxuan Song; Han Zhao; Jingbo Wang; Pengxiang Ding; Donglin Wang; Long Zeng; Haoang Li
>
> **摘要:** Vision-language-action (VLA) models have recently shown strong potential in enabling robots to follow language instructions and execute precise actions. However, most VLAs are built upon vision-language models pretrained solely on 2D data, which lack accurate spatial awareness and hinder their ability to operate in the 3D physical world. Existing solutions attempt to incorporate explicit 3D sensor inputs such as depth maps or point clouds, but these approaches face challenges due to sensor noise, hardware heterogeneity, and incomplete depth coverage in existing datasets. Alternative methods that estimate 3D cues from 2D images also suffer from the limited performance of depth estimators. We propose Spatial Forcing (SF), a simple yet effective alignment strategy that implicitly forces VLA models to develop spatial comprehension capabilities without relying on explicit 3D inputs or depth estimators. SF aligns intermediate visual embeddings of VLAs with geometric representations produced by pretrained 3D foundation models. By enforcing alignment at intermediate layers, SF guides VLAs to encode richer spatial representations that enhance action precision. Extensive experiments in simulation and real-world environments demonstrate that SF achieves state-of-the-art results, surpassing both 2D- and 3D-based VLAs. SF further accelerates training by up to 3.8x and improves data efficiency across diverse robotic tasks. Project page is at https://spatial-forcing.github.io/
>
---
#### [replaced 016] Pseudo-Kinematic Trajectory Control and Planning of Tracked Vehicles
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2409.18641v3](http://arxiv.org/pdf/2409.18641v3)**

> **作者:** Michele Focchi; Daniele Fontanelli; Davide Stocco; Riccardo Bussola; Luigi Palopoli
>
> **摘要:** Tracked vehicles distribute their weight continuously over a large surface area (the tracks). This distinctive feature makes them the preferred choice for vehicles required to traverse soft and uneven terrain. From a robotics perspective, however, this flexibility comes at a cost: the complexity of modelling the system and the resulting difficulty in designing theoretically sound navigation solutions. In this paper, we aim to bridge this gap by proposing a framework for the navigation of tracked vehicles, built upon three key pillars. The first pillar comprises two models: a simulation model and a control-oriented model. The simulation model captures the intricate terramechanics dynamics arising from soil-track interaction and is employed to develop faithful digital twins of the system across a wide range of operating conditions. The control-oriented model is pseudo-kinematic and mathematically tractable, enabling the design of efficient and theoretically robust control schemes. The second pillar is a Lyapunov-based feedback trajectory controller that provides certifiable tracking guarantees. The third pillar is a portfolio of motion planning solutions, each offering different complexity-accuracy trade-offs. The various components of the proposed approach are validated through an extensive set of simulation and experimental data.
>
---
#### [replaced 017] Development and Adaptation of Robotic Vision in the Real-World: the Challenge of Door Detection
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2401.17996v2](http://arxiv.org/pdf/2401.17996v2)**

> **作者:** Michele Antonazzi; Matteo Luperto; N. Alberto Borghese; Nicola Basilico
>
> **摘要:** Mobile service robots are increasingly prevalent in human-centric, real-world domains, operating autonomously in unconstrained indoor environments. In such a context, robotic vision plays a central role in enabling service robots to perceive high-level environmental features from visual observations. Despite the data-driven approaches based on deep learning push the boundaries of vision systems, applying these techniques to real-world robotic scenarios presents unique methodological challenges. Traditional models fail to represent the challenging perception constraints typical of service robots and must be adapted for the specific environment where robots finally operate. We propose a method leveraging photorealistic simulations that balances data quality and acquisition costs for synthesizing visual datasets from the robot perspective used to train deep architectures. Then, we show the benefits in qualifying a general detector for the target domain in which the robot is deployed, showing also the trade-off between the effort for obtaining new examples from such a setting and the performance gain. In our extensive experimental campaign, we focus on the door detection task (namely recognizing the presence and the traversability of doorways) that, in dynamic settings, is useful to infer the topology of the map. Our findings are validated in a real-world robot deployment, comparing prominent deep-learning models and demonstrating the effectiveness of our approach in practical settings.
>
---
#### [replaced 018] From Language to Locomotion: Retargeting-free Humanoid Control via Motion Latent Guidance
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2510.14952v2](http://arxiv.org/pdf/2510.14952v2)**

> **作者:** Zhe Li; Cheng Chi; Yangyang Wei; Boan Zhu; Yibo Peng; Tao Huang; Pengwei Wang; Zhongyuan Wang; Shanghang Zhang; Chang Xu
>
> **摘要:** Natural language offers a natural interface for humanoid robots, but existing language-guided humanoid locomotion pipelines remain cumbersome and untrustworthy. They typically decode human motion, retarget it to robot morphology, and then track it with a physics-based controller. However, this multi-stage process is prone to cumulative errors, introduces high latency, and yields weak coupling between semantics and control. These limitations call for a more direct pathway from language to action, one that eliminates fragile intermediate stages. Therefore, we present RoboGhost, a retargeting-free framework that directly conditions humanoid policies on language-grounded motion latents. By bypassing explicit motion decoding and retargeting, RoboGhost enables a diffusion-based policy to denoise executable actions directly from noise, preserving semantic intent and supporting fast, reactive control. A hybrid causal transformer-diffusion motion generator further ensures long-horizon consistency while maintaining stability and diversity, yielding rich latent representations for precise humanoid behavior. Extensive experiments demonstrate that RoboGhost substantially reduces deployment latency, improves success rates and tracking precision, and produces smooth, semantically aligned locomotion on real humanoids. Beyond text, the framework naturally extends to other modalities such as images, audio, and music, providing a universal foundation for vision-language-action humanoid systems.
>
---
#### [replaced 019] MOBODY: Model Based Off-Dynamics Offline Reinforcement Learning
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.08460v2](http://arxiv.org/pdf/2506.08460v2)**

> **作者:** Yihong Guo; Yu Yang; Pan Xu; Anqi Liu
>
> **摘要:** We study off-dynamics offline reinforcement learning, where the goal is to learn a policy from offline source and limited target datasets with mismatched dynamics. Existing methods either penalize the reward or discard source transitions occurring in parts of the transition space with high dynamics shift. As a result, they optimize the policy using data from low-shift regions, limiting exploration of high-reward states in the target domain that do not fall within these regions. Consequently, such methods often fail when the dynamics shift is significant or the optimal trajectories lie outside the low-shift regions. To overcome this limitation, we propose MOBODY, a Model-Based Off-Dynamics Offline RL algorithm that optimizes a policy using learned target dynamics transitions to explore the target domain, rather than only being trained with the low dynamics-shift transitions. For the dynamics learning, built on the observation that achieving the same next state requires taking different actions in different domains, MOBODY employs separate action encoders for each domain to encode different actions to the shared latent space while sharing a unified representation of states and a common transition function. We further introduce a target Q-weighted behavior cloning loss in policy optimization to avoid out-of-distribution actions, which push the policy toward actions with high target-domain Q-values, rather than high source domain Q-values or uniformly imitating all actions in the offline dataset. We evaluate MOBODY on a wide range of MuJoCo and Adroit benchmarks, demonstrating that it outperforms state-of-the-art off-dynamics RL baselines as well as policy learning methods based on different dynamics learning baselines, with especially pronounced improvements in challenging scenarios where existing methods struggle.
>
---
#### [replaced 020] An Intention-driven Lane Change Framework Considering Heterogeneous Dynamic Cooperation in Mixed-traffic Environment
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.22550v3](http://arxiv.org/pdf/2509.22550v3)**

> **作者:** Xiaoyun Qiu; Haichao Liu; Yue Pan; Jun Ma; Xinhu Zheng
>
> **摘要:** In mixed-traffic environments, where autonomous vehicles (AVs) interact with diverse human-driven vehicles (HVs), unpredictable intentions and heterogeneous behaviors make safe and efficient lane change maneuvers highly challenging. Existing methods often oversimplify these interactions by assuming uniform patterns. We propose an intention-driven lane change framework that integrates driving-style recognition, cooperation-aware decision-making, and coordinated motion planning. A deep learning classifier trained on the NGSIM dataset identifies human driving styles in real time. A cooperation score with intrinsic and interactive components estimates surrounding drivers' intentions and quantifies their willingness to cooperate with the ego vehicle. Decision-making combines behavior cloning with inverse reinforcement learning to determine whether a lane change should be initiated. For trajectory generation, model predictive control is integrated with IRL-based intention inference to produce collision-free and socially compliant maneuvers. Experiments show that the proposed model achieves 94.2\% accuracy and 94.3\% F1-score, outperforming rule-based and learning-based baselines by 4-15\% in lane change recognition. These results highlight the benefit of modeling inter-driver heterogeneity and demonstrate the potential of the framework to advance context-aware and human-like autonomous driving in complex traffic environments.
>
---
#### [replaced 021] COMPASS: Cross-embodiment Mobility Policy via Residual RL and Skill Synthesis
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.16372v2](http://arxiv.org/pdf/2502.16372v2)**

> **作者:** Wei Liu; Huihua Zhao; Chenran Li; Yuchen Deng; Joydeep Biswas; Soha Pouya; Yan Chang
>
> **摘要:** As robots are increasingly deployed in diverse application domains, enabling robust mobility across different embodiments has become a critical challenge. Classical mobility stacks, though effective on specific platforms, require extensive per-robot tuning and do not scale easily to new embodiments. Learning-based approaches, such as imitation learning (IL), offer alternatives, but face significant limitations on the need for high-quality demonstrations for each embodiment. To address these challenges, we introduce COMPASS, a unified framework that enables scalable cross-embodiment mobility using expert demonstrations from only a single embodiment. We first pre-train a mobility policy on a single robot using IL, combining a world model with a policy model. We then apply residual reinforcement learning (RL) to efficiently adapt this policy to diverse embodiments through corrective refinements. Finally, we distill specialist policies into a single generalist policy conditioned on an embodiment embedding vector. This design significantly reduces the burden of collecting data while enabling robust generalization across a wide range of robot designs. Our experiments demonstrate that COMPASS scales effectively across diverse robot platforms while maintaining adaptability to various environment configurations, achieving a generalist policy with a success rate approximately 5X higher than the pre-trained IL policy on unseen embodiments, and further demonstrates zero-shot sim-to-real transfer.
>
---
#### [replaced 022] CLASP: General-Purpose Clothes Manipulation with Semantic Keypoints
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.19983v2](http://arxiv.org/pdf/2507.19983v2)**

> **作者:** Yuhong Deng; Chao Tang; Cunjun Yu; Linfeng Li; David Hsu
>
> **摘要:** Clothes manipulation, such as folding or hanging, is a critical capability for home service robots. Despite recent advances, most existing methods remain limited to specific clothes types and tasks, due to the complex, high-dimensional geometry of clothes. This paper presents CLothes mAnipulation with Semantic keyPoints (CLASP), which aims at general-purpose clothes manipulation over diverse clothes types, T-shirts, shorts, skirts, long dresses, ..., as well as different tasks, folding, flattening, hanging, .... The core idea of CLASP is semantic keypoints-e.g., ''left sleeve'' and ''right shoulder''-a sparse spatial-semantic representation, salient for both perception and action. Semantic keypoints of clothes can be reliably extracted from RGB-D images and provide an effective representation for a wide range of clothes manipulation policies. CLASP uses semantic keypoints as an intermediate representation to connect high-level task planning and low-level action execution. At the high level, it exploits vision language models (VLMs) to predict task plans over the semantic keypoints. At the low level, it executes the plans with the help of a set of pre-built manipulation skills conditioned on the keypoints. Extensive simulation experiments show that CLASP outperforms state-of-the-art baseline methods on multiple tasks across diverse clothes types, demonstrating strong performance and generalization. Further experiments with a Franka dual-arm system on four distinct tasks-folding, flattening, hanging, and placing-confirm CLASP's performance on real-life clothes manipulation.
>
---
#### [replaced 023] Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.05635v2](http://arxiv.org/pdf/2508.05635v2)**

> **作者:** Yue Liao; Pengfei Zhou; Siyuan Huang; Donglin Yang; Shengcong Chen; Yuxin Jiang; Yue Hu; Jingbin Cai; Si Liu; Jianlan Luo; Liliang Chen; Shuicheng Yan; Maoqing Yao; Guanghui Ren
>
> **备注:** https://genie-envisioner.github.io/
>
> **摘要:** We introduce Genie Envisioner (GE), a unified world foundation platform for robotic manipulation that integrates policy learning, evaluation, and simulation within a single video-generative framework. At its core, GE-Base is a large-scale, instruction-conditioned video diffusion model that captures the spatial, temporal, and semantic dynamics of real-world robotic interactions in a structured latent space. Built upon this foundation, GE-Act maps latent representations to executable action trajectories through a lightweight, flow-matching decoder, enabling precise and generalizable policy inference across diverse embodiments with minimal supervision. To support scalable evaluation and training, GE-Sim serves as an action-conditioned neural simulator, producing high-fidelity rollouts for closed-loop policy development. The platform is further equipped with EWMBench, a standardized benchmark suite measuring visual fidelity, physical consistency, and instruction-action alignment. Together, these components establish Genie Envisioner as a scalable and practical foundation for instruction-driven, general-purpose embodied intelligence. All code, models, and benchmarks will be released publicly.
>
---
