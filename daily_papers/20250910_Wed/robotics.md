# 机器人 cs.RO

- **最新发布 31 篇**

- **更新 18 篇**

## 最新发布

#### [new 001] Performance Characterization of a Point-Cloud-Based Path Planner in Off-Road Terrain
- **分类: cs.RO**

- **简介: 论文评估了基于点云的自主越野导航系统MUONS的性能，通过3万次仿真和实地测试分析其路径规划参数对成功率、路径长度和时间的影响，确定关键参数并支持使用蒙特卡洛方法进行参数调优。**

- **链接: [http://arxiv.org/pdf/2509.07321v1](http://arxiv.org/pdf/2509.07321v1)**

> **作者:** Casey D. Majhor; Jeremy P. Bos
>
> **备注:** This work has been published in the Journal of Field Robotics
>
> **摘要:** We present a comprehensive evaluation of a point-cloud-based navigation stack, MUONS, for autonomous off-road navigation. Performance is characterized by analyzing the results of 30,000 planning and navigation trials in simulation and validated through field testing. Our simulation campaign considers three kinematically challenging terrain maps and twenty combinations of seven path-planning parameters. In simulation, our MUONS-equipped AGV achieved a 0.98 success rate and experienced no failures in the field. By statistical and correlation analysis we determined that the Bi-RRT expansion radius used in the initial planning stages is most correlated with performance in terms of planning time and traversed path length. Finally, we observed that the proportional variation due to changes in the tuning parameters is remarkably well correlated to performance in field testing. This finding supports the use of Monte-Carlo simulation campaigns for performance assessment and parameter tuning.
>
---
#### [new 002] TransMPC: Transformer-based Explicit MPC with Variable Prediction Horizon
- **分类: cs.RO**

- **简介: 论文提出TransMPC，一种基于Transformer的显式MPC算法，用于复杂系统的实时控制。解决传统MPC计算复杂度高、精度不足的问题，通过Transformer编码器实现变预测时域的高效控制策略优化。**

- **链接: [http://arxiv.org/pdf/2509.07381v1](http://arxiv.org/pdf/2509.07381v1)**

> **作者:** Sichao Wu; Jiang Wu; Xingyu Cao; Fawang Zhang; Guangyuan Yu; Junjie Zhao; Yue Qu; Fei Ma; Jingliang Duan
>
> **摘要:** Traditional online Model Predictive Control (MPC) methods often suffer from excessive computational complexity, limiting their practical deployment. Explicit MPC mitigates online computational load by pre-computing control policies offline; however, existing explicit MPC methods typically rely on simplified system dynamics and cost functions, restricting their accuracy for complex systems. This paper proposes TransMPC, a novel Transformer-based explicit MPC algorithm capable of generating highly accurate control sequences in real-time for complex dynamic systems. Specifically, we formulate the MPC policy as an encoder-only Transformer leveraging bidirectional self-attention, enabling simultaneous inference of entire control sequences in a single forward pass. This design inherently accommodates variable prediction horizons while ensuring low inference latency. Furthermore, we introduce a direct policy optimization framework that alternates between sampling and learning phases. Unlike imitation-based approaches dependent on precomputed optimal trajectories, TransMPC directly optimizes the true finite-horizon cost via automatic differentiation. Random horizon sampling combined with a replay buffer provides independent and identically distributed (i.i.d.) training samples, ensuring robust generalization across varying states and horizon lengths. Extensive simulations and real-world vehicle control experiments validate the effectiveness of TransMPC in terms of solution accuracy, adaptability to varying horizons, and computational efficiency.
>
---
#### [new 003] Can SSD-Mamba2 Unlock Reinforcement Learning for End-to-End Motion Control?
- **分类: cs.RO; cs.AI; cs.CV; cs.SY; eess.IV; eess.SY**

- **简介: 论文提出基于SSD-Mamba2的跨模态强化学习框架，解决端到端运动控制中感知-动作策略的融合问题。通过高效状态空间模型实现低延迟、长依赖建模，提升控制性能与训练效率。**

- **链接: [http://arxiv.org/pdf/2509.07593v1](http://arxiv.org/pdf/2509.07593v1)**

> **作者:** Gavin Tao; Yinuo Wang; Jinzhao Zhou
>
> **备注:** 4 figures and 6 tables
>
> **摘要:** End-to-end reinforcement learning for motion control promises unified perception-action policies that scale across embodiments and tasks, yet most deployed controllers are either blind (proprioception-only) or rely on fusion backbones with unfavorable compute-memory trade-offs. Recurrent controllers struggle with long-horizon credit assignment, and Transformer-based fusion incurs quadratic cost in token length, limiting temporal and spatial context. We present a vision-driven cross-modal RL framework built on SSD-Mamba2, a selective state-space backbone that applies state-space duality (SSD) to enable both recurrent and convolutional scanning with hardware-aware streaming and near-linear scaling. Proprioceptive states and exteroceptive observations (e.g., depth tokens) are encoded into compact tokens and fused by stacked SSD-Mamba2 layers. The selective state-space updates retain long-range dependencies with markedly lower latency and memory use than quadratic self-attention, enabling longer look-ahead, higher token resolution, and stable training under limited compute. Policies are trained end-to-end under curricula that randomize terrain and appearance and progressively increase scene complexity. A compact, state-centric reward balances task progress, energy efficiency, and safety. Across diverse motion-control scenarios, our approach consistently surpasses strong state-of-the-art baselines in return, safety (collisions and falls), and sample efficiency, while converging faster at the same compute budget. These results suggest that SSD-Mamba2 provides a practical fusion backbone for scalable, foresightful, and efficient end-to-end motion control.
>
---
#### [new 004] Collaborative Exploration with a Marsupial Ground-Aerial Robot Team through Task-Driven Map Compression
- **分类: cs.RO**

- **简介: 论文提出一种协作探索框架，利用地面-空中机器人团队互补能力，通过任务驱动的地图压缩策略提升大范围未知环境下的探索效率，减少通信开销。属于自主机器人探索任务，解决通信受限下的高效地图共享与路径规划问题。**

- **链接: [http://arxiv.org/pdf/2509.07655v1](http://arxiv.org/pdf/2509.07655v1)**

> **作者:** Angelos Zacharia; Mihir Dharmadhikari; Kostas Alexis
>
> **备注:** Accepted for publication in IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Efficient exploration of unknown environments is crucial for autonomous robots, especially in confined and large-scale scenarios with limited communication. To address this challenge, we propose a collaborative exploration framework for a marsupial ground-aerial robot team that leverages the complementary capabilities of both platforms. The framework employs a graph-based path planning algorithm to guide exploration and deploy the aerial robot in areas where its expected gain significantly exceeds that of the ground robot, such as large open spaces or regions inaccessible to the ground platform, thereby maximizing coverage and efficiency. To facilitate large-scale spatial information sharing, we introduce a bandwidth-efficient, task-driven map compression strategy. This method enables each robot to reconstruct resolution-specific volumetric maps while preserving exploration-critical details, even at high compression rates. By selectively compressing and sharing key data, communication overhead is minimized, ensuring effective map integration for collaborative path planning. Simulation and real-world experiments validate the proposed approach, demonstrating its effectiveness in improving exploration efficiency while significantly reducing data transmission.
>
---
#### [new 005] OmniMap: A General Mapping Framework Integrating Optics, Geometry, and Semantics
- **分类: cs.RO**

- **简介: 该论文提出OmniMap，一种集成光学、几何与语义的实时三维地图框架，解决现有方法在视觉清晰度、几何精度与语义理解上的不足，通过混合表示与多模态融合实现高效精准的环境感知。**

- **链接: [http://arxiv.org/pdf/2509.07500v1](http://arxiv.org/pdf/2509.07500v1)**

> **作者:** Yinan Deng; Yufeng Yue; Jianyu Dou; Jingyu Zhao; Jiahui Wang; Yujie Tang; Yi Yang; Mengyin Fu
>
> **备注:** Accepted by IEEE Transactions on Robotics (TRO), project website: https://omni-map.github.io/
>
> **摘要:** Robotic systems demand accurate and comprehensive 3D environment perception, requiring simultaneous capture of photo-realistic appearance (optical), precise layout shape (geometric), and open-vocabulary scene understanding (semantic). Existing methods typically achieve only partial fulfillment of these requirements while exhibiting optical blurring, geometric irregularities, and semantic ambiguities. To address these challenges, we propose OmniMap. Overall, OmniMap represents the first online mapping framework that simultaneously captures optical, geometric, and semantic scene attributes while maintaining real-time performance and model compactness. At the architectural level, OmniMap employs a tightly coupled 3DGS-Voxel hybrid representation that combines fine-grained modeling with structural stability. At the implementation level, OmniMap identifies key challenges across different modalities and introduces several innovations: adaptive camera modeling for motion blur and exposure compensation, hybrid incremental representation with normal constraints, and probabilistic fusion for robust instance-level understanding. Extensive experiments show OmniMap's superior performance in rendering fidelity, geometric accuracy, and zero-shot semantic segmentation compared to state-of-the-art methods across diverse scenes. The framework's versatility is further evidenced through a variety of downstream applications, including multi-domain scene Q&A, interactive editing, perception-guided manipulation, and map-assisted navigation.
>
---
#### [new 006] Programmable Locking Cells (PLC) for Modular Robots with High Stiffness Tunability and Morphological Adaptability
- **分类: cs.RO**

- **简介: 论文提出可编程锁定单元（PLC），用于模块化机器人，解决传统变刚度方案复杂、不可扩展的问题。通过缆绳驱动实现离散刚度调节，具备高刚度可调性和形态适应性，适用于抓取、导航等任务。**

- **链接: [http://arxiv.org/pdf/2509.07916v1](http://arxiv.org/pdf/2509.07916v1)**

> **作者:** Jianshu Zhou; Wei Chen; Junda Huang; Boyuan Liang; Yunhui Liu; Masayoshi Tomizuka
>
> **摘要:** Robotic systems operating in unstructured environments require the ability to switch between compliant and rigid states to perform diverse tasks such as adaptive grasping, high-force manipulation, shape holding, and navigation in constrained spaces, among others. However, many existing variable stiffness solutions rely on complex actuation schemes, continuous input power, or monolithic designs, limiting their modularity and scalability. This paper presents the Programmable Locking Cell (PLC)-a modular, tendon-driven unit that achieves discrete stiffness modulation through mechanically interlocked joints actuated by cable tension. Each unit transitions between compliant and firm states via structural engagement, and the assembled system exhibits high stiffness variation-up to 950% per unit-without susceptibility to damage under high payload in the firm state. Multiple PLC units can be assembled into reconfigurable robotic structures with spatially programmable stiffness. We validate the design through two functional prototypes: (1) a variable-stiffness gripper capable of adaptive grasping, firm holding, and in-hand manipulation; and (2) a pipe-traversing robot composed of serial PLC units that achieves shape adaptability and stiffness control in confined environments. These results demonstrate the PLC as a scalable, structure-centric mechanism for programmable stiffness and motion, enabling robotic systems with reconfigurable morphology and task-adaptive interaction.
>
---
#### [new 007] Quantum Machine Learning and Grover's Algorithm for Quantum Optimization of Robotic Manipulators
- **分类: cs.RO**

- **简介: 论文提出一种量子机器学习与Grover算法结合的框架，用于优化高自由度机械臂的运动学问题。通过参数化量子电路近似正向运动学模型，并利用Grover算法加速搜索，实现比经典方法更高的效率，适用于多自由度机械臂任务。**

- **链接: [http://arxiv.org/pdf/2509.07216v1](http://arxiv.org/pdf/2509.07216v1)**

> **作者:** Hassen Nigatu; Shi Gaokun; Li Jituo; Wang Jin; Lu Guodong; Howard Li
>
> **摘要:** Optimizing high-degree of freedom robotic manipulators requires searching complex, high-dimensional configuration spaces, a task that is computationally challenging for classical methods. This paper introduces a quantum native framework that integrates quantum machine learning with Grover's algorithm to solve kinematic optimization problems efficiently. A parameterized quantum circuit is trained to approximate the forward kinematics model, which then constructs an oracle to identify optimal configurations. Grover's algorithm leverages this oracle to provide a quadratic reduction in search complexity. Demonstrated on 1-DoF, 2-DoF, and dual-arm manipulator tasks, the method achieves significant speedups-up to 93x over classical optimizers like Nelder Mead as problem dimensionality increases. This work establishes a foundational, quantum-native framework for robot kinematic optimization, effectively bridging quantum computing and robotics problems.
>
---
#### [new 008] Aerial-ground Cross-modal Localization: Dataset, Ground-truth, and Benchmark
- **分类: cs.RO**

- **简介: 该论文属于视觉定位任务，旨在解决城市环境中图像定位精度低的问题。通过构建融合地面图像与激光雷达点云的大规模数据集，提出基准以推动跨平台视觉定位算法的发展。**

- **链接: [http://arxiv.org/pdf/2509.07362v1](http://arxiv.org/pdf/2509.07362v1)**

> **作者:** Yandi Yang; Jianping Li; Youqi Liao; Yuhao Li; Yizhe Zhang; Zhen Dong; Bisheng Yang; Naser El-Sheimy
>
> **摘要:** Accurate visual localization in dense urban environments poses a fundamental task in photogrammetry, geospatial information science, and robotics. While imagery is a low-cost and widely accessible sensing modality, its effectiveness on visual odometry is often limited by textureless surfaces, severe viewpoint changes, and long-term drift. The growing public availability of airborne laser scanning (ALS) data opens new avenues for scalable and precise visual localization by leveraging ALS as a prior map. However, the potential of ALS-based localization remains underexplored due to three key limitations: (1) the lack of platform-diverse datasets, (2) the absence of reliable ground-truth generation methods applicable to large-scale urban environments, and (3) limited validation of existing Image-to-Point Cloud (I2P) algorithms under aerial-ground cross-platform settings. To overcome these challenges, we introduce a new large-scale dataset that integrates ground-level imagery from mobile mapping systems with ALS point clouds collected in Wuhan, Hong Kong, and San Francisco.
>
---
#### [new 009] Decoding RobKiNet: Insights into Efficient Training of Robotic Kinematics Informed Neural Network
- **分类: cs.RO**

- **简介: 论文提出RobKiNet，一种结合运动学知识的神经网络，用于机器人任务与运动规划中的高效配置空间采样。解决传统方法效率低的问题，实现快速准确采样，提升机器人整体控制与任务完成率。**

- **链接: [http://arxiv.org/pdf/2509.07646v1](http://arxiv.org/pdf/2509.07646v1)**

> **作者:** Yanlong Peng; Zhigang Wang; Ziwen He; Pengxu Chang; Chuangchuang Zhou; Yu Yan; Ming Chen
>
> **摘要:** In robots task and motion planning (TAMP), it is crucial to sample within the robot's configuration space to meet task-level global constraints and enhance the efficiency of subsequent motion planning. Due to the complexity of joint configuration sampling under multi-level constraints, traditional methods often lack efficiency. This paper introduces the principle of RobKiNet, a kinematics-informed neural network, for end-to-end sampling within the Continuous Feasible Set (CFS) under multiple constraints in configuration space, establishing its Optimization Expectation Model. Comparisons with traditional sampling and learning-based approaches reveal that RobKiNet's kinematic knowledge infusion enhances training efficiency by ensuring stable and accurate gradient optimization.Visualizations and quantitative analyses in a 2-DOF space validate its theoretical efficiency, while its application on a 9-DOF autonomous mobile manipulator robot(AMMR) demonstrates superior whole-body and decoupled control, excelling in battery disassembly tasks. RobKiNet outperforms deep reinforcement learning with a training speed 74.29 times faster and a sampling accuracy of up to 99.25%, achieving a 97.33% task completion rate in real-world scenarios.
>
---
#### [new 010] Text2Touch: Tactile In-Hand Manipulation with LLM-Designed Reward Functions
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Text2Touch，利用大语言模型设计奖励函数，解决多轴在手物体旋转任务中的触觉感知问题。通过提示工程和仿真到现实的知识蒸馏，实现触觉驱动的灵巧机械手控制，显著提升旋转速度与稳定性。**

- **链接: [http://arxiv.org/pdf/2509.07445v1](http://arxiv.org/pdf/2509.07445v1)**

> **作者:** Harrison Field; Max Yang; Yijiong Lin; Efi Psomopoulou; David Barton; Nathan F. Lepora
>
> **备注:** Accepted at CoRL 2025
>
> **摘要:** Large language models (LLMs) are beginning to automate reward design for dexterous manipulation. However, no prior work has considered tactile sensing, which is known to be critical for human-like dexterity. We present Text2Touch, bringing LLM-crafted rewards to the challenging task of multi-axis in-hand object rotation with real-world vision based tactile sensing in palm-up and palm-down configurations. Our prompt engineering strategy scales to over 70 environment variables, and sim-to-real distillation enables successful policy transfer to a tactile-enabled fully actuated four-fingered dexterous robot hand. Text2Touch significantly outperforms a carefully tuned human-engineered baseline, demonstrating superior rotation speed and stability while relying on reward functions that are an order of magnitude shorter and simpler. These results illustrate how LLM-designed rewards can significantly reduce the time from concept to deployable dexterous tactile skills, supporting more rapid and scalable multimodal robot learning. Project website: https://hpfield.github.io/text2touch-website
>
---
#### [new 011] DepthVision: Robust Vision-Language Understanding through GAN-Based LiDAR-to-RGB Synthesis
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出DepthVision框架，通过GAN生成RGB图像以增强视觉-语言理解，解决机器人在低光或视觉退化环境下的可靠操作问题。利用LiDAR点云合成RGB图像，并结合LAMA模块适应光照条件，提升模型在低光环境下的性能，无需微调下游模型。**

- **链接: [http://arxiv.org/pdf/2509.07463v1](http://arxiv.org/pdf/2509.07463v1)**

> **作者:** Sven Kirchner; Nils Purschke; Ross Greer; Alois C. Knoll
>
> **摘要:** Ensuring reliable robot operation when visual input is degraded or insufficient remains a central challenge in robotics. This letter introduces DepthVision, a framework for multimodal scene understanding designed to address this problem. Unlike existing Vision-Language Models (VLMs), which use only camera-based visual input alongside language, DepthVision synthesizes RGB images from sparse LiDAR point clouds using a conditional generative adversarial network (GAN) with an integrated refiner network. These synthetic views are then combined with real RGB data using a Luminance-Aware Modality Adaptation (LAMA), which blends the two types of data dynamically based on ambient lighting conditions. This approach compensates for sensor degradation, such as darkness or motion blur, without requiring any fine-tuning of downstream vision-language models. We evaluate DepthVision on real and simulated datasets across various models and tasks, with particular attention to safety-critical tasks. The results demonstrate that our approach improves performance in low-light conditions, achieving substantial gains over RGB-only baselines while preserving compatibility with frozen VLMs. This work highlights the potential of LiDAR-guided RGB synthesis for achieving robust robot operation in real-world environments.
>
---
#### [new 012] Flexible Morphing Aerial Robot with Inflatable Structure for Perching-based Human-Robot Interaction
- **分类: cs.RO**

- **简介: 该论文提出一种可变形空中机器人，用于与人类互动的抓握着陆。解决飞行稳定性与安全抓握问题，设计混合结构与气动控制系统，实现柔顺着陆与高效能量利用。**

- **链接: [http://arxiv.org/pdf/2509.07496v1](http://arxiv.org/pdf/2509.07496v1)**

> **作者:** Ayano Miyamichi; Moju Zhao; Kazuki Sugihara; Junichiro Sugihara; Masanori Konishi; Kunio Kojima; Kei Okada; Masayuki Inaba
>
> **摘要:** Birds in nature perform perching not only for rest but also for interaction with human such as the relationship with falconers. Recently, researchers achieve perching-capable aerial robots as a way to save energy, and deformable structure demonstrate significant advantages in efficiency of perching and compactness of configuration. However, ensuring flight stability remains challenging for deformable aerial robots due to the difficulty of controlling flexible arms. Furthermore, perching for human interaction requires high compliance along with safety. Thus, this study aims to develop a deformable aerial robot capable of perching on humans with high flexibility and grasping ability. To overcome the challenges of stability of both flight and perching, we propose a hybrid morphing structure that combines a unilateral flexible arm and a pneumatic inflatable actuators. This design allows the robot's arms to remain rigid during flight and soft while perching for more effective grasping. We also develop a pneumatic control system that optimizes pressure regulation while integrating shock absorption and adjustable grasping forces, enhancing interaction capabilities and energy efficiency. Besides, we focus on the structural characteristics of the unilateral flexible arm and identify sufficient conditions under which standard quadrotor modeling and control remain effective in terms of flight stability. Finally, the developed prototype demonstrates the feasibility of compliant perching maneuvers on humans, as well as the robust recovery even after arm deformation caused by thrust reductions during flight. To the best of our knowledge, this work is the first to achieve an aerial robot capable of perching on humans for interaction.
>
---
#### [new 013] Unlocking Stopped-Rotor Flight: Development and Validation of SPERO, a Novel UAV Platform
- **分类: cs.RO**

- **简介: 论文提出SPERO无人机，解决停转旋翼飞行器在垂直起降与前飞模式间切换的稳定性与效率问题。通过五旋翼结构、可翻转机翼等创新设计，实现稳定双向过渡，建立通用设计与控制框架。**

- **链接: [http://arxiv.org/pdf/2509.07812v1](http://arxiv.org/pdf/2509.07812v1)**

> **作者:** Kristan Hilby; Ian Hunter
>
> **备注:** 15 pages, 11 figures, 5 tables
>
> **摘要:** Stop-rotor aircraft have long been proposed as the ideal vertical takeoff and landing (VTOL) aircraft for missions with equal time spent in both flight regimes, such as agricultural monitoring, search and rescue, and last-mile delivery. Featuring a central lifting surface that rotates in VTOL to generate vertical thrust and locks in forward flight to generate passive lift, the stop-rotor offers the potential for high efficiency across both modes. However, practical implementation has remained infeasible due to aerodynamic and stability conflicts between flight modes. In this work, we present SPERO (Stopped-Penta Rotor), a stop-rotor uncrewed aerial vehicle (UAV) featuring a flipping and latching wing, an active center of pressure mechanism, thrust vectored counterbalances, a five-rotor architecture, and an eleven-state machine flight controller coordinating geometric and controller reconfiguration. Furthermore, SPERO establishes a generalizable design and control framework for stopped-rotor UAVs. Together, these innovations overcome longstanding challenges in stop-rotor flight and enable the first stable, bidirectional transition between VTOL and forward flight.
>
---
#### [new 014] Temporal Counterfactual Explanations of Behaviour Tree Decisions
- **分类: cs.RO; cs.HC**

- **简介: 该论文提出一种方法，为行为树决策生成反事实解释，解决机器人决策可解释性问题。通过构建因果模型，自动回答“为什么”问题，提升系统透明度与可信度。**

- **链接: [http://arxiv.org/pdf/2509.07674v1](http://arxiv.org/pdf/2509.07674v1)**

> **作者:** Tamlin Love; Antonio Andriella; Guillem Alenyà
>
> **备注:** 23 pages, 6 figures, submitted to Engineering Applications of Artificial Intelligence
>
> **摘要:** Explainability is a critical tool in helping stakeholders understand robots. In particular, the ability for robots to explain why they have made a particular decision or behaved in a certain way is useful in this regard. Behaviour trees are a popular framework for controlling the decision-making of robots and other software systems, and thus a natural question to ask is whether or not a system driven by a behaviour tree is capable of answering "why" questions. While explainability for behaviour trees has seen some prior attention, no existing methods are capable of generating causal, counterfactual explanations which detail the reasons for robot decisions and behaviour. Therefore, in this work, we introduce a novel approach which automatically generates counterfactual explanations in response to contrastive "why" questions. Our method achieves this by first automatically building a causal model from the structure of the behaviour tree as well as domain knowledge about the state and individual behaviour tree nodes. The resultant causal model is then queried and searched to find a set of diverse counterfactual explanations. We demonstrate that our approach is able to correctly explain the behaviour of a wide range of behaviour tree structures and states. By being able to answer a wide range of causal queries, our approach represents a step towards more transparent, understandable and ultimately trustworthy robotic systems.
>
---
#### [new 015] Timing the Message: Language-Based Notifications for Time-Critical Assistive Settings
- **分类: cs.RO; cs.HC**

- **简介: 论文研究时间敏感场景下的语言通知系统，解决传统警报信息不及时、不明确的问题。提出结合强化学习与生成数据集的框架，平衡时效性与信息量，提升任务成功率。属于人机协作中的通信优化任务。**

- **链接: [http://arxiv.org/pdf/2509.07438v1](http://arxiv.org/pdf/2509.07438v1)**

> **作者:** Ya-Chuan Hsu; Jonathan DeCastro; Andrew Silva; Guy Rosman
>
> **摘要:** In time-critical settings such as assistive driving, assistants often rely on alerts or haptic signals to prompt rapid human attention, but these cues usually leave humans to interpret situations and decide responses independently, introducing potential delays or ambiguity in meaning. Language-based assistive systems can instead provide instructions backed by context, offering more informative guidance. However, current approaches (e.g., social assistive robots) largely prioritize content generation while overlooking critical timing factors such as verbal conveyance duration, human comprehension delays, and subsequent follow-through duration. These timing considerations are crucial in time-critical settings, where even minor delays can substantially affect outcomes. We aim to study this inherent trade-off between timeliness and informativeness by framing the challenge as a sequential decision-making problem using an augmented-state Markov Decision Process. We design a framework combining reinforcement learning and a generated offline taxonomy dataset, where we balance the trade-off while enabling a scalable taxonomy dataset generation pipeline. Empirical evaluation with synthetic humans shows our framework improves success rates by over 40% compared to methods that ignore time delays, while effectively balancing timeliness and informativeness. It also exposes an often-overlooked trade-off between these two factors, opening new directions for optimizing communication in time-critical human-AI assistance.
>
---
#### [new 016] Improving Machine Learning-Based Robot Self-Collision Checking with Input Positional Encoding
- **分类: cs.RO**

- **简介: 该论文属于机器人自碰撞检测任务，旨在提升机器学习模型的检测精度。研究将位置编码引入输入向量，增强模型对复杂碰撞模式的捕捉能力，证明轻量MLP比传统几何方法更高效。**

- **链接: [http://arxiv.org/pdf/2509.07542v1](http://arxiv.org/pdf/2509.07542v1)**

> **作者:** Bartlomiej Kulecki; Dominik Belter
>
> **摘要:** This manuscript investigates the integration of positional encoding -- a technique widely used in computer graphics -- into the input vector of a binary classification model for self-collision detection. The results demonstrate the benefits of incorporating positional encoding, which enhances classification accuracy by enabling the model to better capture high-frequency variations, leading to a more detailed and precise representation of complex collision patterns. The manuscript shows that machine learning-based techniques, such as lightweight multilayer perceptrons (MLPs) operating in a low-dimensional feature space, offer a faster alternative for collision checking than traditional methods that rely on geometric approaches, such as triangle-to-triangle intersection tests and Bounding Volume Hierarchies (BVH) for mesh-based models.
>
---
#### [new 017] Robust Radar SLAM for Vehicle Parking Applications
- **分类: cs.RO**

- **简介: 该论文提出一种基于雷达的SLAM方法，用于自动驾驶停车场景中的高精度定位。解决传统方法依赖校准、成本高的问题，通过融合雷达特征与多普勒速度，实现无需校准的鲁棒定位，提升恶劣天气下的性能。**

- **链接: [http://arxiv.org/pdf/2509.07683v1](http://arxiv.org/pdf/2509.07683v1)**

> **作者:** Luis Diener; Jens Kalkkuhl; Markus Enzweiler
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** We address ego-motion estimation for automated parking, where centimeter-level accuracy is crucial due to tight spaces and nearby obstacles. Traditional methods using inertial-measurement units and wheel encoders require calibration, making them costly and time-consuming. To overcome this, we propose a radar-based simultaneous localization and mapping (SLAM) approach that leverages the robustness of radar to adverse weather and support for online calibration. Our robocentric formulation fuses feature positions and Doppler velocities for robust data association and filter convergence. Key contributions include a Doppler-augmented radar SLAM method, multi-radar support and an information-based feature-pruning strategy. Experiments demonstrate high-accuracy localization and improved robustness over state-of-the-art methods, meeting the demands of automated parking.
>
---
#### [new 018] Safe Gap-based Planning in Dynamic Settings
- **分类: cs.RO**

- **简介: 该论文提出动态间隙规划方法，解决动态环境下的避障问题。通过跟踪和预测自由空间区域（间隙），结合追击引导理论生成安全轨迹，并在真实机器人平台上验证其有效性。属于运动规划任务。**

- **链接: [http://arxiv.org/pdf/2509.07239v1](http://arxiv.org/pdf/2509.07239v1)**

> **作者:** Max Asselmeier; Abdel Zaro; Dhruv Ahuja; Ye Zhao; Patricio A. Vela
>
> **备注:** Accepted to Algorithms for Machine Vision in Navigation and Control - Springer Publishing House
>
> **摘要:** This chapter extends the family of perception-informed gap-based local planners to dynamic environments. Existing perception-informed local planners that operate in dynamic environments often rely on emergent or empirical robustness for collision avoidance as opposed to performing formal analysis of dynamic obstacles. This proposed planner, dynamic gap, explicitly addresses dynamic obstacles through several steps in the planning pipeline. First, polar regions of free space known as gaps are tracked and their dynamics are estimated in order to understand how the local environment evolves over time. Then, at planning time, gaps are propagated into the future through novel gap propagation algorithms to understand what regions are feasible for passage. Lastly, pursuit guidance theory is leveraged to generate local trajectories that are provably collision-free under ideal conditions. Additionally, obstacle-centric ungap processing is performed in situations where no gaps exist to robustify the overall planning framework. A set of gap-based planners are benchmarked against a series of classical and learned motion planners in dynamic environments, and dynamic gap is shown to outperform all other baselines in all environments. Furthermore, dynamic gap is deployed on a TurtleBot2 platform in several real-world experiments to validate collision avoidance behaviors.
>
---
#### [new 019] TA-VLA: Elucidating the Design Space of Torque-aware Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文研究如何将扭矩信号整合到视觉-语言-动作（VLA）模型中，以提升机器人操作任务的性能。通过设计扭矩感知模块，提出三种有效策略，改善模型对物理交互的理解与控制能力。**

- **链接: [http://arxiv.org/pdf/2509.07962v1](http://arxiv.org/pdf/2509.07962v1)**

> **作者:** Zongzheng Zhang; Haobo Xu; Zhuo Yang; Chenghao Yue; Zehao Lin; Huan-ang Gao; Ziwei Wang; Hao Zhao
>
> **备注:** Accepted to CoRL 2025, project page: \url{https://zzongzheng0918.github.io/Torque-Aware-VLA.github.io/}
>
> **摘要:** Many robotic manipulation tasks require sensing and responding to force signals such as torque to assess whether the task has been successfully completed and to enable closed-loop control. However, current Vision-Language-Action (VLA) models lack the ability to integrate such subtle physical feedback. In this work, we explore Torque-aware VLA models, aiming to bridge this gap by systematically studying the design space for incorporating torque signals into existing VLA architectures. We identify and evaluate several strategies, leading to three key findings. First, introducing torque adapters into the decoder consistently outperforms inserting them into the encoder.Third, inspired by joint prediction and planning paradigms in autonomous driving, we propose predicting torque as an auxiliary output, which further improves performance. This strategy encourages the model to build a physically grounded internal representation of interaction dynamics. Extensive quantitative and qualitative experiments across contact-rich manipulation benchmarks validate our findings.
>
---
#### [new 020] Fault Tolerant Control of a Quadcopter using Reinforcement Learning
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出基于强化学习的四旋翼飞行器容错控制框架，解决单旋翼故障下的安全飞行问题，采用DP和DDPG算法训练控制器，通过仿真验证其鲁棒性，适用于关键任务飞行器应用。**

- **链接: [http://arxiv.org/pdf/2509.07707v1](http://arxiv.org/pdf/2509.07707v1)**

> **作者:** Muzaffar Habib; Adnan Maqsood; Adnan Fayyaz ud Din
>
> **备注:** e-ISSN: 1946-3901, ISSN: 1946-3855, https://www.sae.org/publications/technical-papers/content/01-18-01-0006/
>
> **摘要:** This study presents a novel reinforcement learning (RL)-based control framework aimed at enhancing the safety and robustness of the quadcopter, with a specific focus on resilience to in-flight one propeller failure. Addressing the critical need of a robust control strategy for maintaining a desired altitude for the quadcopter to safe the hardware and the payload in physical applications. The proposed framework investigates two RL methodologies Dynamic Programming (DP) and Deep Deterministic Policy Gradient (DDPG), to overcome the challenges posed by the rotor failure mechanism of the quadcopter. DP, a model-based approach, is leveraged for its convergence guarantees, despite high computational demands, whereas DDPG, a model-free technique, facilitates rapid computation but with constraints on solution duration. The research challenge arises from training RL algorithms on large dimensions and action domains. With modifications to the existing DP and DDPG algorithms, the controllers were trained not only to cater for large continuous state and action domain and also achieve a desired state after an inflight propeller failure. To verify the robustness of the proposed control framework, extensive simulations were conducted in a MATLAB environment across various initial conditions and underscoring its viability for mission-critical quadcopter applications. A comparative analysis was performed between both RL algorithms and their potential for applications in faulty aerial systems.
>
---
#### [new 021] Safe and Non-Conservative Contingency Planning for Autonomous Vehicles via Online Learning-Based Reachable Set Barriers
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文提出一种基于在线学习的可达集屏障方法，用于自动驾驶车辆的实时应急轨迹优化。旨在解决动态不确定环境下安全与效率的平衡问题，通过量化人类车辆不确定性并嵌入安全约束，提升驾驶效率和舒适度。**

- **链接: [http://arxiv.org/pdf/2509.07464v1](http://arxiv.org/pdf/2509.07464v1)**

> **作者:** Rui Yang; Lei Zheng; Shuzhi Sam Ge; Jun Ma
>
> **备注:** 16 pages, 13 figures
>
> **摘要:** Autonomous vehicles must navigate dynamically uncertain environments while balancing the safety and driving efficiency. This challenge is exacerbated by the unpredictable nature of surrounding human-driven vehicles (HVs) and perception inaccuracies, which require planners to adapt to evolving uncertainties while maintaining safe trajectories. Overly conservative planners degrade driving efficiency, while deterministic approaches may encounter serious issues and risks of failure when faced with sudden and unexpected maneuvers. To address these issues, we propose a real-time contingency trajectory optimization framework in this paper. By employing event-triggered online learning of HV control-intent sets, our method dynamically quantifies multi-modal HV uncertainties and refines the forward reachable set (FRS) incrementally. Crucially, we enforce invariant safety through FRS-based barrier constraints that ensure safety without reliance on accurate trajectory prediction of HVs. These constraints are embedded in contingency trajectory optimization and solved efficiently through consensus alternative direction method of multipliers (ADMM). The system continuously adapts to the uncertainties in HV behaviors, preserving feasibility and safety without resorting to excessive conservatism. High-fidelity simulations on highway and urban scenarios, as well as a series of real-world experiments demonstrate significant improvements in driving efficiency and passenger comfort while maintaining safety under uncertainty. The project page is available at https://pathetiue.github.io/frscp.github.io/.
>
---
#### [new 022] Robust Docking Maneuvers for Autonomous Trolley Collection: An Optimization-Based Visual Servoing Scheme
- **分类: cs.RO**

- **简介: 论文提出基于优化的视觉伺服方案，解决自主推车收集中的鲁棒对接问题。通过红外标记和非完整运动学建模，提升对接精度与稳定性，适用于多机器人队形任务。**

- **链接: [http://arxiv.org/pdf/2509.07413v1](http://arxiv.org/pdf/2509.07413v1)**

> **作者:** Yuhan Pang; Bingyi Xia; Zhe Zhang; Zhirui Sun; Peijia Xie; Bike Zhu; Wenjun Xu; Jiankun Wang
>
> **摘要:** Service robots have demonstrated significant potential for autonomous trolley collection and redistribution in public spaces like airports or warehouses to improve efficiency and reduce cost. Usually, a fully autonomous system for the collection and transportation of multiple trolleys is based on a Leader-Follower formation of mobile manipulators, where reliable docking maneuvers of the mobile base are essential to align trolleys into organized queues. However, developing a vision-based robotic docking system faces significant challenges: high precision requirements, environmental disturbances, and inherent robot constraints. To address these challenges, we propose an optimization-based Visual Servoing scheme that incorporates active infrared markers for robust feature extraction across diverse lighting conditions. This framework explicitly models nonholonomic kinematics and visibility constraints within the Hybrid Visual Servoing problem, augmented with an observer for disturbance rejection to ensure precise and stable docking. Experimental results across diverse environments demonstrate the robustness of this system, with quantitative evaluations confirming high docking accuracy.
>
---
#### [new 023] First Plan Then Evaluate: Use a Vectorized Motion Planner for Grasping
- **分类: cs.RO**

- **简介: 论文提出一种基于向量化运动规划器的抓取框架，解决传统方法在轨迹优化与抓取成功率间的权衡问题。通过并行规划多目标轨迹并评估成功率，提升抓取效率与准确性。属于机器人抓取规划任务。**

- **链接: [http://arxiv.org/pdf/2509.07162v1](http://arxiv.org/pdf/2509.07162v1)**

> **作者:** Martin Matak; Mohanraj Devendran Ashanti; Karl Van Wyk; Tucker Hermans
>
> **摘要:** Autonomous multi-finger grasping is a fundamental capability in robotic manipulation. Optimization-based approaches show strong performance, but tend to be sensitive to initialization and are potentially time-consuming. As an alternative, the generator-evaluator-planner framework has been proposed. A generator generates grasp candidates, an evaluator ranks the proposed grasps, and a motion planner plans a trajectory to the highest-ranked grasp. If the planner doesn't find a trajectory, a new trajectory optimization is started with the next-best grasp as the target and so on. However, executing lower-ranked grasps means a lower chance of grasp success, and multiple trajectory optimizations are time-consuming. Alternatively, relaxing the threshold for motion planning accuracy allows for easier computation of a successful trajectory but implies lower accuracy in estimating grasp success likelihood. It's a lose-lose proposition: either spend more time finding a successful trajectory or have a worse estimate of grasp success. We propose a framework that plans trajectories to a set of generated grasp targets in parallel, the evaluator estimates the grasp success likelihood of the resulting trajectories, and the robot executes the trajectory most likely to succeed. To plan trajectories to different targets efficiently, we propose the use of a vectorized motion planner. Our experiments show our approach improves over the traditional generator-evaluator-planner framework across different objects, generators, and motion planners, and successfully generalizes to novel environments in the real world, including different shelves and table heights. Project website https://sites.google.com/view/fpte
>
---
#### [new 024] Graph-Fused Vision-Language-Action for Policy Reasoning in Multi-Arm Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文提出GF-VLA框架，解决双臂机器人从人类视频演示中学习复杂操作任务的问题。通过融合视觉、语言和动作信息，实现任务级推理与执行，提升跨物体与空间布局的泛化能力。**

- **链接: [http://arxiv.org/pdf/2509.07957v1](http://arxiv.org/pdf/2509.07957v1)**

> **作者:** Shunlei Li; Longsen Gao; Jiuwen Cao; Yingbai Hu
>
> **备注:** This paper is submitted to IEEE IROS 2025 Workshop AIR4S
>
> **摘要:** Acquiring dexterous robotic skills from human video demonstrations remains a significant challenge, largely due to conventional reliance on low-level trajectory replication, which often fails to generalize across varying objects, spatial layouts, and manipulator configurations. To address this limitation, we introduce Graph-Fused Vision-Language-Action (GF-VLA), a unified framework that enables dual-arm robotic systems to perform task-level reasoning and execution directly from RGB-D human demonstrations. GF-VLA employs an information-theoretic approach to extract task-relevant cues, selectively highlighting critical hand-object and object-object interactions. These cues are structured into temporally ordered scene graphs, which are subsequently integrated with a language-conditioned transformer to produce hierarchical behavior trees and interpretable Cartesian motion primitives. To enhance efficiency in bimanual execution, we propose a cross-arm allocation strategy that autonomously determines gripper assignment without requiring explicit geometric modeling. We validate GF-VLA on four dual-arm block assembly benchmarks involving symbolic structure construction and spatial generalization. Empirical results demonstrate that the proposed representation achieves over 95% graph accuracy and 93% subtask segmentation, enabling the language-action planner to generate robust, interpretable task policies. When deployed on a dual-arm robot, these policies attain 94% grasp reliability, 89% placement accuracy, and 90% overall task success across stacking, letter-formation, and geometric reconfiguration tasks, evidencing strong generalization and robustness under diverse spatial and semantic variations.
>
---
#### [new 025] Attention and Risk-Aware Decision Framework for Safe Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文提出一种改进的PPO算法，用于解决自动驾驶中的安全决策问题。通过引入风险感知机制、注意力网络和安全辅助机制，提升碰撞规避能力与训练效率，实现更安全高效的自主驾驶控制。**

- **链接: [http://arxiv.org/pdf/2509.07412v1](http://arxiv.org/pdf/2509.07412v1)**

> **作者:** Zhen Tian; Fujiang Yuan; Yangfan He; Qinghao Li; Changlin Chen; Huilin Chen; Tianxiang Xu; Jianyu Duan; Yanhong Peng; Zhihao Lin
>
> **摘要:** Autonomous driving has attracted great interest due to its potential capability in full-unsupervised driving. Model-based and learning-based methods are widely used in autonomous driving. Model-based methods rely on pre-defined models of the environment and may struggle with unforeseen events. Proximal policy optimization (PPO), an advanced learning-based method, can adapt to the above limits by learning from interactions with the environment. However, existing PPO faces challenges with poor training results, and low training efficiency in long sequences. Moreover, the poor training results are equivalent to collisions in driving tasks. To solve these issues, this paper develops an improved PPO by introducing the risk-aware mechanism, a risk-attention decision network, a balanced reward function, and a safety-assisted mechanism. The risk-aware mechanism focuses on highlighting areas with potential collisions, facilitating safe-driving learning of the PPO. The balanced reward function adjusts rewards based on the number of surrounding vehicles, promoting efficient exploration of the control strategy during training. Additionally, the risk-attention network enhances the PPO to hold channel and spatial attention for the high-risk areas of input images. Moreover, the safety-assisted mechanism supervises and prevents the actions with risks of collisions during the lane keeping and lane changing. Simulation results on a physical engine demonstrate that the proposed algorithm outperforms benchmark algorithms in collision avoidance, achieving higher peak reward with less training time, and shorter driving time remaining on the risky areas among multiple testing traffic flow scenarios.
>
---
#### [new 026] RaC: Robot Learning for Long-Horizon Tasks by Scaling Recovery and Correction
- **分类: cs.RO; cs.LG**

- **简介: 论文提出RaC方法，解决机器人长时序任务中因人类演示数据不足导致的性能瓶颈。通过引入人类干预轨迹进行微调，增强机器人重试与适应能力，提升复杂任务执行效率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2509.07953v1](http://arxiv.org/pdf/2509.07953v1)**

> **作者:** Zheyuan Hu; Robyn Wu; Naveen Enock; Jasmine Li; Riya Kadakia; Zackory Erickson; Aviral Kumar
>
> **摘要:** Modern paradigms for robot imitation train expressive policy architectures on large amounts of human demonstration data. Yet performance on contact-rich, deformable-object, and long-horizon tasks plateau far below perfect execution, even with thousands of expert demonstrations. This is due to the inefficiency of existing ``expert'' data collection procedures based on human teleoperation. To address this issue, we introduce RaC, a new phase of training on human-in-the-loop rollouts after imitation learning pre-training. In RaC, we fine-tune a robotic policy on human intervention trajectories that illustrate recovery and correction behaviors. Specifically, during a policy rollout, human operators intervene when failure appears imminent, first rewinding the robot back to a familiar, in-distribution state and then providing a corrective segment that completes the current sub-task. Training on this data composition expands the robotic skill repertoire to include retry and adaptation behaviors, which we show are crucial for boosting both efficiency and robustness on long-horizon tasks. Across three real-world bimanual control tasks: shirt hanging, airtight container lid sealing, takeout box packing, and a simulated assembly task, RaC outperforms the prior state-of-the-art using 10$\times$ less data collection time and samples. We also show that RaC enables test-time scaling: the performance of the trained RaC policy scales linearly in the number of recovery maneuvers it exhibits. Videos of the learned policy are available at https://rac-scaling-robot.github.io/.
>
---
#### [new 027] Efficient Multi-Agent Coordination via Dynamic Joint-State Graph Construction
- **分类: cs.MA; cs.RO**

- **简介: 该论文研究多智能体路径规划中的协作问题，提出动态联合状态图构建方法（Dynamic-HJSG），解决高风险边的协同优化问题，降低计算复杂度，提升大规模团队的规划效率。属于多智能体协作规划任务。**

- **链接: [http://arxiv.org/pdf/2509.07234v1](http://arxiv.org/pdf/2509.07234v1)**

> **作者:** Yanlin Zhou; Manshi Limbu; Xuesu Xiao
>
> **摘要:** Multi-agent pathfinding (MAPF) traditionally focuses on collision avoidance, but many real-world applications require active coordination between agents to improve team performance. This paper introduces Team Coordination on Graphs with Risky Edges (TCGRE), where agents collaborate to reduce traversal costs on high-risk edges via support from teammates. We reformulate TCGRE as a 3D matching problem-mapping robot pairs, support pairs, and time steps-and rigorously prove its NP-hardness via reduction from Minimum 3D Matching. To address this complexity, (in the conference version) we proposed efficient decomposition methods, reducing the problem to tractable subproblems: Joint-State Graph (JSG): Encodes coordination as a single-agent shortest-path problem. Coordination-Exhaustive Search (CES): Optimizes support assignments via exhaustive pairing. Receding-Horizon Optimistic Cooperative A* (RHOCA*): Balances optimality and scalability via horizon-limited planning. Further in this extension, we introduce a dynamic graph construction method (Dynamic-HJSG), leveraging agent homogeneity to prune redundant states and reduce computational overhead by constructing the joint-state graph dynamically. Theoretical analysis shows Dynamic-HJSG preserves optimality while lowering complexity from exponential to polynomial in key cases. Empirical results validate scalability for large teams and graphs, with HJSG outperforming baselines greatly in runtime in different sizes and types of graphs. This work bridges combinatorial optimization and multi-agent planning, offering a principled framework for collaborative pathfinding with provable guarantees, and the key idea of the solution can be widely extended to many other collaborative optimization problems, such as MAPF.
>
---
#### [new 028] A Robot That Listens: Enhancing Self-Disclosure and Engagement Through Sentiment-based Backchannels and Active Listening
- **分类: cs.HC; cs.RO**

- **简介: 该论文研究社会机器人通过主动倾听和情感反馈增强与人类交流的效果。任务是提升机器人的人际互动能力，解决如何促进人类自我披露和建立积极关系的问题。研究开发并比较了不同倾听行为的机器人效果，结果显示主动倾听显著提升互动质量和用户自我披露程度。**

- **链接: [http://arxiv.org/pdf/2509.07873v1](http://arxiv.org/pdf/2509.07873v1)**

> **作者:** Hieu Tran; Go-Eum Cha; Sooyeon Jeong
>
> **摘要:** As social robots get more deeply integrated intoour everyday lives, they will be expected to engage in meaningful conversations and exhibit socio-emotionally intelligent listening behaviors when interacting with people. Active listening and backchanneling could be one way to enhance robots' communicative capabilities and enhance their effectiveness in eliciting deeper self-disclosure, providing a sense of empathy,and forming positive rapport and relationships with people.Thus, we developed an LLM-powered social robot that can exhibit contextually appropriate sentiment-based backchannelingand active listening behaviors (active listening+backchanneling) and compared its efficacy in eliciting people's self-disclosurein comparison to robots that do not exhibit any of these listening behaviors (control) and a robot that only exhibitsbackchanneling behavior (backchanneling-only). Through ourexperimental study with sixty-five participants, we found theparticipants who conversed with the active listening robot per-ceived the interactions more positively, in which they exhibited the highest self-disclosures, and reported the strongest senseof being listened to. The results of our study suggest that the implementation of active listening behaviors in social robotshas the potential to improve human-robot communication andcould further contribute to the building of deeper human-robot relationships and rapport.
>
---
#### [new 029] Bio-inspired decision making in swarms under biases from stubborn robots, corrupted communication, and independent discovery
- **分类: cs.MA; cs.RO**

- **简介: 论文研究受生物启发的群体决策机制，解决机器人群在通信受限、存在偏见和独立发现下的协调问题。通过对比两种机制，提出更高效、鲁棒的跨抑制方法，提升群体决策性能。**

- **链接: [http://arxiv.org/pdf/2509.07561v1](http://arxiv.org/pdf/2509.07561v1)**

> **作者:** Raina Zakir; Timoteo Carletti; Marco Dorigo; Andreagiovanni Reina
>
> **摘要:** Minimalistic robot swarms offer a scalable, robust, and cost-effective approach to performing complex tasks with the potential to transform applications in healthcare, disaster response, and environmental monitoring. However, coordinating such decentralised systems remains a fundamental challenge, particularly when robots are constrained in communication, computation, and memory. In our study, individual robots frequently make errors when sensing the environment, yet the swarm can rapidly and reliably reach consensus on the best among $n$ discrete options. We compare two canonical mechanisms of opinion dynamics -- direct-switch and cross-inhibition -- which are simple yet effective rules for collective information processing observed in biological systems across scales, from neural populations to insect colonies. We generalise the existing mean-field models by considering asocial biases influencing the opinion dynamics. While swarms using direct-switch reliably select the best option in absence of asocial dynamics, their performance deteriorates once such biases are introduced, often resulting in decision deadlocks. In contrast, bio-inspired cross-inhibition enables faster, more cohesive, accurate, robust, and scalable decisions across a wide range of biased conditions. Our findings provide theoretical and practical insights into the coordination of minimal swarms and offer insights that extend to a broad class of decentralised decision-making systems in biology and engineering.
>
---
#### [new 030] Knowledge Isn't Power: The Ethics of Social Robots and the Difficulty of Informed Consent
- **分类: cs.HC; cs.RO**

- **简介: 论文探讨社会机器人引发的伦理问题，聚焦知情同意的挑战。通过法律与伦理视角，分析人机交互的独特性，提出更符合伦理的设计目标，以促进更负责任的机器人开发与使用。**

- **链接: [http://arxiv.org/pdf/2509.07942v1](http://arxiv.org/pdf/2509.07942v1)**

> **作者:** James M. Berzuk; Lauren Corcoran; Brannen McKenzie-Lefurgey; Katie Szilagyi; James E. Young
>
> **备注:** Submitted to the International Journal of Social Robotics. 18 pages, 1 figure
>
> **摘要:** Contemporary robots are increasingly mimicking human social behaviours to facilitate interaction, such as smiling to signal approachability, or hesitating before taking an action to allow people time to react. Such techniques can activate a person's entrenched social instincts, triggering emotional responses as though they are interacting with a fellow human, and can prompt them to treat a robot as if it truly possesses the underlying life-like processes it outwardly presents, raising significant ethical questions. We engage these issues through the lens of informed consent: drawing upon prevailing legal principles and ethics, we examine how social robots can influence user behaviour in novel ways, and whether under those circumstances users can be appropriately informed to consent to these heightened interactions. We explore the complex circumstances of human-robot interaction and highlight how it differs from more familiar interaction contexts, and we apply legal principles relating to informed consent to social robots in order to reconceptualize the current ethical debates surrounding the field. From this investigation, we synthesize design goals for robot developers to achieve more ethical and informed human-robot interaction.
>
---
#### [new 031] Adaptive Evolutionary Framework for Safe, Efficient, and Cooperative Autonomous Vehicle Interactions
- **分类: cs.MA; cs.RO**

- **简介: 该论文提出基于进化博弈论的框架，解决自动驾驶车辆间安全、高效、协作交互问题。通过引入因果评估模块优化演化速率，提升适应性与效率，优于传统方法。属于多智能体协作与交通优化任务。**

- **链接: [http://arxiv.org/pdf/2509.07411v1](http://arxiv.org/pdf/2509.07411v1)**

> **作者:** Zhen Tian; Zhihao Lin
>
> **摘要:** Modern transportation systems face significant challenges in ensuring road safety, given serious injuries caused by road accidents. The rapid growth of autonomous vehicles (AVs) has prompted new traffic designs that aim to optimize interactions among AVs. However, effective interactions between AVs remains challenging due to the absence of centralized control. Besides, there is a need for balancing multiple factors, including passenger demands and overall traffic efficiency. Traditional rule-based, optimization-based, and game-theoretic approaches each have limitations in addressing these challenges. Rule-based methods struggle with adaptability and generalization in complex scenarios, while optimization-based methods often require high computational resources. Game-theoretic approaches, such as Stackelberg and Nash games, suffer from limited adaptability and potential inefficiencies in cooperative settings. This paper proposes an Evolutionary Game Theory (EGT)-based framework for AV interactions that overcomes these limitations by utilizing a decentralized and adaptive strategy evolution mechanism. A causal evaluation module (CEGT) is introduced to optimize the evolutionary rate, balancing mutation and evolution by learning from historical interactions. Simulation results demonstrate the proposed CEGT outperforms EGT and popular benchmark games in terms of lower collision rates, improved safety distances, higher speeds, and overall better performance compared to Nash and Stackelberg games across diverse scenarios and parameter settings.
>
---
## 更新

#### [replaced 001] EmbodiedOneVision: Interleaved Vision-Text-Action Pretraining for General Robot Control
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.21112v3](http://arxiv.org/pdf/2508.21112v3)**

> **作者:** Delin Qu; Haoming Song; Qizhi Chen; Zhaoqing Chen; Xianqiang Gao; Xinyi Ye; Qi Lv; Modi Shi; Guanghui Ren; Cheng Ruan; Maoqing Yao; Haoran Yang; Jiacheng Bao; Bin Zhao; Dong Wang
>
> **摘要:** The human ability to seamlessly perform multimodal reasoning and physical interaction in the open world is a core goal for general-purpose embodied intelligent systems. Recent vision-language-action (VLA) models, which are co-trained on large-scale robot and visual-text data, have demonstrated notable progress in general robot control. However, they still fail to achieve human-level flexibility in interleaved reasoning and interaction. In this work, introduce EO-Robotics, consists of EO-1 model and EO-Data1.5M dataset. EO-1 is a unified embodied foundation model that achieves superior performance in multimodal embodied reasoning and robot control through interleaved vision-text-action pre-training. The development of EO-1 is based on two key pillars: (i) a unified architecture that processes multimodal inputs indiscriminately (image, text, video, and action), and (ii) a massive, high-quality multimodal embodied reasoning dataset, EO-Data1.5M, which contains over 1.5 million samples with emphasis on interleaved vision-text-action comprehension. EO-1 is trained through synergies between auto-regressive decoding and flow matching denoising on EO-Data1.5M, enabling seamless robot action generation and multimodal embodied reasoning. Extensive experiments demonstrate the effectiveness of interleaved vision-text-action learning for open-world understanding and generalization, validated through a variety of long-horizon, dexterous manipulation tasks across multiple embodiments. This paper details the architecture of EO-1, the data construction strategy of EO-Data1.5M, and the training methodology, offering valuable insights for developing advanced embodied foundation models.
>
---
#### [replaced 002] Monte Carlo Tree Search with Tensor Factorization for Robot Optimization
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.04949v2](http://arxiv.org/pdf/2507.04949v2)**

> **作者:** Teng Xue; Yan Zhang; Amirreza Razmjoo; Sylvain Calinon
>
> **备注:** 21 pages, 11 figures
>
> **摘要:** Many robotic tasks, such as inverse kinematics, motion planning, and optimal control, can be formulated as optimization problems. Solving these problems involves addressing nonlinear kinematics, complex contact dynamics, long-horizon correlation, and multi-modal landscapes, each posing distinct challenges for state-of-the-art optimization methods. Monte Carlo Tree Search is a powerful approach that can strategically explore the solution space and can be applied to a wide range of tasks across varying scenarios. However, it typically suffers from combinatorial complexity when applied to robotics, resulting in slow convergence and high memory demands. To address this limitation, we propose \emph{Tensor Train Tree Search} (TTTS), which leverages tensor factorization to exploit correlations among decision variables arising from common kinematic structures, dynamic constraints, and environmental interactions in robot decision-making. This yields a compact, linear-complexity representation that significantly reduces both computation time and storage requirements. We prove that TTTS can efficiently reach the bounded global optimum within a finite time. Experimental results across inverse kinematics, motion planning around obstacles, legged robot manipulation, multi-stage motion planning, and bimanual whole-body manipulation demonstrate the efficiency of TTTS on a diverse set of robotic tasks.
>
---
#### [replaced 003] PINGS: Gaussian Splatting Meets Distance Fields within a Point-Based Implicit Neural Map
- **分类: cs.RO; cs.CV; cs.GR**

- **链接: [http://arxiv.org/pdf/2502.05752v2](http://arxiv.org/pdf/2502.05752v2)**

> **作者:** Yue Pan; Xingguang Zhong; Liren Jin; Louis Wiesmann; Marija Popović; Jens Behley; Cyrill Stachniss
>
> **备注:** 15 pages, 8 figures, presented at RSS 2025
>
> **摘要:** Robots benefit from high-fidelity reconstructions of their environment, which should be geometrically accurate and photorealistic to support downstream tasks. While this can be achieved by building distance fields from range sensors and radiance fields from cameras, realising scalable incremental mapping of both fields consistently and at the same time with high quality is challenging. In this paper, we propose a novel map representation that unifies a continuous signed distance field and a Gaussian splatting radiance field within an elastic and compact point-based implicit neural map. By enforcing geometric consistency between these fields, we achieve mutual improvements by exploiting both modalities. We present a novel LiDAR-visual SLAM system called PINGS using the proposed map representation and evaluate it on several challenging large-scale datasets. Experimental results demonstrate that PINGS can incrementally build globally consistent distance and radiance fields encoded with a compact set of neural points. Compared to state-of-the-art methods, PINGS achieves superior photometric and geometric rendering at novel views by constraining the radiance field with the distance field. Furthermore, by utilizing dense photometric cues and multi-view consistency from the radiance field, PINGS produces more accurate distance fields, leading to improved odometry estimation and mesh reconstruction. We also provide an open-source implementation of PING at: https://github.com/PRBonn/PINGS.
>
---
#### [replaced 004] Visuospatial Cognitive Assistant
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.12312v4](http://arxiv.org/pdf/2505.12312v4)**

> **作者:** Qi Feng
>
> **备注:** 31 pages, 10 figures, 6 tables
>
> **摘要:** Video-based spatial cognition is vital for robotics and embodied AI but challenges current Vision-Language Models (VLMs). This paper makes two key contributions. First, we introduce ViCA (Visuospatial Cognitive Assistant)-322K, a diverse dataset of 322,003 QA pairs from real-world indoor videos (ARKitScenes, ScanNet, ScanNet++), offering supervision for 3D metadata-grounded queries and video-based complex reasoning. Second, we develop ViCA-7B, fine-tuned on ViCA-322K, which achieves new state-of-the-art on all eight VSI-Bench tasks, outperforming existing models, including larger ones (e.g., +26.1 on Absolute Distance). For interpretability, we present ViCA-Thinking-2.68K, a dataset with explicit reasoning chains, and fine-tune ViCA-7B to create ViCA-7B-Thinking, a model that articulates its spatial reasoning. Our work highlights the importance of targeted data and suggests paths for improved temporal-spatial modeling. We release all resources to foster research in robust visuospatial intelligence.
>
---
#### [replaced 005] Generalizable Humanoid Manipulation with 3D Diffusion Policies
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2410.10803v3](http://arxiv.org/pdf/2410.10803v3)**

> **作者:** Yanjie Ze; Zixuan Chen; Wenhao Wang; Tianyi Chen; Xialin He; Ying Yuan; Xue Bin Peng; Jiajun Wu
>
> **备注:** IROS 2025. Project website: https://humanoid-manipulation.github.io
>
> **摘要:** Humanoid robots capable of autonomous operation in diverse environments have long been a goal for roboticists. However, autonomous manipulation by humanoid robots has largely been restricted to one specific scene, primarily due to the difficulty of acquiring generalizable skills and the expensiveness of in-the-wild humanoid robot data. In this work, we build a real-world robotic system to address this challenging problem. Our system is mainly an integration of 1) a whole-upper-body robotic teleoperation system to acquire human-like robot data, 2) a 25-DoF humanoid robot platform with a height-adjustable cart and a 3D LiDAR sensor, and 3) an improved 3D Diffusion Policy learning algorithm for humanoid robots to learn from noisy human data. We run more than 2000 episodes of policy rollouts on the real robot for rigorous policy evaluation. Empowered by this system, we show that using only data collected in one single scene and with only onboard computing, a full-sized humanoid robot can autonomously perform skills in diverse real-world scenarios. Videos are available at https://humanoid-manipulation.github.io .
>
---
#### [replaced 006] SCIZOR: A Self-Supervised Approach to Data Curation for Large-Scale Imitation Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.22626v2](http://arxiv.org/pdf/2505.22626v2)**

> **作者:** Yu Zhang; Yuqi Xie; Huihan Liu; Rutav Shah; Michael Wan; Linxi Fan; Yuke Zhu
>
> **摘要:** Imitation learning advances robot capabilities by enabling the acquisition of diverse behaviors from human demonstrations. However, large-scale datasets used for policy training often introduce substantial variability in quality, which can negatively impact performance. As a result, automatically curating datasets by filtering low-quality samples to improve quality becomes essential. Existing robotic curation approaches rely on costly manual annotations and perform curation at a coarse granularity, such as the dataset or trajectory level, failing to account for the quality of individual state-action pairs. To address this, we introduce SCIZOR, a self-supervised data curation framework that filters out low-quality state-action pairs to improve the performance of imitation learning policies. SCIZOR targets two complementary sources of low-quality data: suboptimal data, which hinders learning with undesirable actions, and redundant data, which dilutes training with repetitive patterns. SCIZOR leverages a self-supervised task progress predictor for suboptimal data to remove samples lacking task progression, and a deduplication module operating on joint state-action representation for samples with redundant patterns. Empirically, we show that SCIZOR enables imitation learning policies to achieve higher performance with less data, yielding an average improvement of 15.4% across multiple benchmarks. More information is available at: https://ut-austin-rpl.github.io/SCIZOR/
>
---
#### [replaced 007] F1: A Vision-Language-Action Model Bridging Understanding and Generation to Actions
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2509.06951v2](http://arxiv.org/pdf/2509.06951v2)**

> **作者:** Qi Lv; Weijie Kong; Hao Li; Jia Zeng; Zherui Qiu; Delin Qu; Haoming Song; Qizhi Chen; Xiang Deng; Jiangmiao Pang
>
> **备注:** Homepage: https://aopolin-lv.github.io/F1-VLA/
>
> **摘要:** Executing language-conditioned tasks in dynamic visual environments remains a central challenge in embodied AI. Existing Vision-Language-Action (VLA) models predominantly adopt reactive state-to-action mappings, often leading to short-sighted behaviors and poor robustness in dynamic scenes. In this paper, we introduce F1, a pretrained VLA framework which integrates the visual foresight generation into decision-making pipeline. F1 adopts a Mixture-of-Transformer architecture with dedicated modules for perception, foresight generation, and control, thereby bridging understanding, generation, and actions. At its core, F1 employs a next-scale prediction mechanism to synthesize goal-conditioned visual foresight as explicit planning targets. By forecasting plausible future visual states, F1 reformulates action generation as a foresight-guided inverse dynamics problem, enabling actions that implicitly achieve visual goals. To endow F1 with robust and generalizable capabilities, we propose a three-stage training recipe on an extensive dataset comprising over 330k trajectories across 136 diverse tasks. This training scheme enhances modular reasoning and equips the model with transferable visual foresight, which is critical for complex and dynamic environments. Extensive evaluations on real-world tasks and simulation benchmarks demonstrate F1 consistently outperforms existing approaches, achieving substantial gains in both task success rate and generalization ability.
>
---
#### [replaced 008] DriveSOTIF: Advancing Perception SOTIF Through Multimodal Large Language Models
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.07084v3](http://arxiv.org/pdf/2505.07084v3)**

> **作者:** Shucheng Huang; Freda Shi; Chen Sun; Jiaming Zhong; Minghao Ning; Yufeng Yang; Yukun Lu; Hong Wang; Amir Khajepour
>
> **备注:** This work has been accepted to IEEE Transactions on Vehicular Technology. Please refer to the copyright notice for additional information
>
> **摘要:** Human drivers possess spatial and causal intelligence, enabling them to perceive driving scenarios, anticipate hazards, and react to dynamic environments. In contrast, autonomous vehicles lack these abilities, making it challenging to manage perception-related Safety of the Intended Functionality (SOTIF) risks, especially under complex or unpredictable driving conditions. To address this gap, we propose fine-tuning multimodal large language models (MLLMs) on a customized dataset specifically designed to capture perception-related SOTIF scenarios. Benchmarking results show that fine-tuned MLLMs achieve an 11.8\% improvement in close-ended VQA accuracy and a 12.0\% increase in open-ended VQA scores compared to baseline models, while maintaining real-time performance with a 0.59-second average inference time per image. We validate our approach through real-world case studies in Canada and China, where fine-tuned models correctly identify safety risks that challenge even experienced human drivers. This work represents the first application of domain-specific MLLM fine-tuning for SOTIF domain in autonomous driving. The dataset and related resources are available at github.com/s95huang/DriveSOTIF.git
>
---
#### [replaced 009] Prepared for the Worst: A Learning-Based Adversarial Attack for Resilience Analysis of the ICP Algorithm
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2403.05666v3](http://arxiv.org/pdf/2403.05666v3)**

> **作者:** Ziyu Zhang; Johann Laconte; Daniil Lisus; Timothy D. Barfoot
>
> **备注:** 9 pages (6 content, 1 reference, 2 appendix). 7 figures, accepted to 2025 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** This paper presents a novel method for assessing the resilience of the ICP algorithm via learning-based, worst-case attacks on lidar point clouds. For safety-critical applications such as autonomous navigation, ensuring the resilience of algorithms before deployments is crucial. The ICP algorithm is the standard for lidar-based localization, but its accuracy can be greatly affected by corrupted measurements from various sources, including occlusions, adverse weather, or mechanical sensor issues. Unfortunately, the complex and iterative nature of ICP makes assessing its resilience to corruption challenging. While there have been efforts to create challenging datasets and develop simulations to evaluate the resilience of ICP, our method focuses on finding the maximum possible ICP error that can arise from corrupted measurements at a location. We demonstrate that our perturbation-based adversarial attacks can be used pre-deployment to identify locations on a map where ICP is particularly vulnerable to corruptions in the measurements. With such information, autonomous robots can take safer paths when deployed, to mitigate against their measurements being corrupted. The proposed attack outperforms baselines more than 88% of the time across a wide range of scenarios.
>
---
#### [replaced 010] MoRPI-PINN: A Physics-Informed Framework for Mobile Robot Pure Inertial Navigation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.18206v2](http://arxiv.org/pdf/2507.18206v2)**

> **作者:** Arup Kumar Sahoo; Itzik Klein
>
> **备注:** 9 pages, 5 figures
>
> **摘要:** A fundamental requirement for full autonomy in mobile robots is accurate navigation even in situations where satellite navigation or cameras are unavailable. In such practical situations, relying only on inertial sensors will result in navigation solution drift due to the sensors' inherent noise and error terms. One of the emerging solutions to mitigate drift is to maneuver the robot in a snake-like slithering motion to increase the inertial signal-to-noise ratio, allowing the regression of the mobile robot position. In this work, we propose MoRPI-PINN as a physics-informed neural network framework for accurate inertial-based mobile robot navigation. By embedding physical laws and constraints into the training process, MoRPI-PINN is capable of providing an accurate and robust navigation solution. Using real-world experiments, we show accuracy improvements of over 85% compared to other approaches. MoRPI-PINN is a lightweight approach that can be implemented even on edge devices and used in any typical mobile robot application.
>
---
#### [replaced 011] Towards Visuospatial Cognition via Hierarchical Fusion of Visual Experts
- **分类: cs.CV; cs.AI; cs.CL; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.12363v4](http://arxiv.org/pdf/2505.12363v4)**

> **作者:** Qi Feng
>
> **备注:** 26 pages, 19 figures, 4 tables
>
> **摘要:** While Multimodal Large Language Models (MLLMs) excel at general vision-language tasks, visuospatial cognition - reasoning about spatial layouts, relations, and dynamics - remains a significant challenge. Existing models often lack the necessary architectural components and specialized training data for fine-grained spatial understanding. We introduce ViCA2 (Visuospatial Cognitive Assistant 2), a novel MLLM designed to enhance spatial reasoning. ViCA2 features a dual vision encoder architecture integrating SigLIP for semantics and Hiera for spatial structure, coupled with a token ratio control mechanism for efficiency. We also developed ViCA-322K, a new large-scale dataset with over 322,000 spatially grounded question-answer pairs for targeted instruction tuning. On the challenging VSI-Bench benchmark, our ViCA2-7B model achieves a state-of-the-art average score of 56.8, significantly surpassing larger open-source models (e.g., LLaVA-NeXT-Video-72B, 40.9) and leading proprietary models (Gemini-1.5 Pro, 45.4). This demonstrates the effectiveness of our approach in achieving strong visuospatial intelligence with a compact model. We release ViCA2, its codebase, and the ViCA-322K dataset to facilitate further research.
>
---
#### [replaced 012] GraspCoT: Integrating Physical Property Reasoning for 6-DoF Grasping under Flexible Language Instructions
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.16013v2](http://arxiv.org/pdf/2503.16013v2)**

> **作者:** Xiaomeng Chu; Jiajun Deng; Guoliang You; Wei Liu; Xingchen Li; Jianmin Ji; Yanyong Zhang
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** Flexible instruction-guided 6-DoF grasping is a significant yet challenging task for real-world robotic systems. Existing methods utilize the contextual understanding capabilities of the large language models (LLMs) to establish mappings between expressions and targets, allowing robots to comprehend users' intentions in the instructions. However, the LLM's knowledge about objects' physical properties remains underexplored despite its tight relevance to grasping. In this work, we propose GraspCoT, a 6-DoF grasp detection framework that integrates a Chain-of-Thought (CoT) reasoning mechanism oriented to physical properties, guided by auxiliary question-answering (QA) tasks. Particularly, we design a set of QA templates to enable hierarchical reasoning that includes three stages: target parsing, physical property analysis, and grasp action selection. Moreover, GraspCoT presents a unified multimodal LLM architecture, which encodes multi-view observations of 3D scenes into 3D-aware visual tokens, and then jointly embeds these visual tokens with CoT-derived textual tokens within LLMs to generate grasp pose predictions. Furthermore, we present IntentGrasp, a large-scale benchmark that fills the gap in public datasets for multi-object grasp detection under diverse and indirect verbal commands. Extensive experiments on IntentGrasp demonstrate the superiority of our method, with additional validation in real-world robotic applications confirming its practicality. The code is available at https://github.com/cxmomo/GraspCoT.
>
---
#### [replaced 013] VMGNet: A Low Computational Complexity Robotic Grasping Network Based on VMamba with Multi-Scale Feature Fusion
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.12520v2](http://arxiv.org/pdf/2411.12520v2)**

> **作者:** Yuhao Jin; Qizhong Gao; Xiaohui Zhu; Yong Yue; Eng Gee Lim; Yuqing Chen; Prudence Wong; Yijie Chu
>
> **备注:** This work is part of ongoing research, and we are further developing new techniques based on these results. To avoid premature disclosure of incomplete content, we request withdrawal of the current version and will resubmit once the study is more complete
>
> **摘要:** While deep learning-based robotic grasping technology has demonstrated strong adaptability, its computational complexity has also significantly increased, making it unsuitable for scenarios with high real-time requirements. Therefore, we propose a low computational complexity and high accuracy model named VMGNet for robotic grasping. For the first time, we introduce the Visual State Space into the robotic grasping field to achieve linear computational complexity, thereby greatly reducing the model's computational cost. Meanwhile, to improve the accuracy of the model, we propose an efficient and lightweight multi-scale feature fusion module, named Fusion Bridge Module, to extract and fuse information at different scales. We also present a new loss function calculation method to enhance the importance differences between subtasks, improving the model's fitting ability. Experiments show that VMGNet has only 8.7G Floating Point Operations and an inference time of 8.1 ms on our devices. VMGNet also achieved state-of-the-art performance on the Cornell and Jacquard public datasets. To validate VMGNet's effectiveness in practical applications, we conducted real grasping experiments in multi-object scenarios, and VMGNet achieved an excellent performance with a 94.4% success rate in real-world grasping tasks. The video for the real-world robotic grasping experiments is available at https://youtu.be/S-QHBtbmLc4.
>
---
#### [replaced 014] LiDARCrafter: Dynamic 4D World Modeling from LiDAR Sequences
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.03692v2](http://arxiv.org/pdf/2508.03692v2)**

> **作者:** Ao Liang; Youquan Liu; Yu Yang; Dongyue Lu; Linfeng Li; Lingdong Kong; Huaici Zhao; Wei Tsang Ooi
>
> **备注:** Preprint; 28 pages, 18 figures, 12 tables; Project Page at https://lidarcrafter.github.io
>
> **摘要:** Generative world models have become essential data engines for autonomous driving, yet most existing efforts focus on videos or occupancy grids, overlooking the unique LiDAR properties. Extending LiDAR generation to dynamic 4D world modeling presents challenges in controllability, temporal coherence, and evaluation standardization. To this end, we present LiDARCrafter, a unified framework for 4D LiDAR generation and editing. Given free-form natural language inputs, we parse instructions into ego-centric scene graphs, which condition a tri-branch diffusion network to generate object structures, motion trajectories, and geometry. These structured conditions enable diverse and fine-grained scene editing. Additionally, an autoregressive module generates temporally coherent 4D LiDAR sequences with smooth transitions. To support standardized evaluation, we establish a comprehensive benchmark with diverse metrics spanning scene-, object-, and sequence-level aspects. Experiments on the nuScenes dataset using this benchmark demonstrate that LiDARCrafter achieves state-of-the-art performance in fidelity, controllability, and temporal consistency across all levels, paving the way for data augmentation and simulation. The code and benchmark are released to the community.
>
---
#### [replaced 015] Interactive Shaping of Granular Media Using Reinforcement Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.06469v2](http://arxiv.org/pdf/2509.06469v2)**

> **作者:** Benedikt Kreis; Malte Mosbach; Anny Ripke; Muhammad Ehsan Ullah; Sven Behnke; Maren Bennewitz
>
> **备注:** Accepted to IEEE-RAS International Conference on Humanoid Robots (Humanoids) 2025
>
> **摘要:** Autonomous manipulation of granular media, such as sand, is crucial for applications in construction, excavation, and additive manufacturing. However, shaping granular materials presents unique challenges due to their high-dimensional configuration space and complex dynamics, where traditional rule-based approaches struggle without extensive engineering efforts. Reinforcement learning (RL) offers a promising alternative by enabling agents to learn adaptive manipulation strategies through trial and error. In this work, we present an RL framework that enables a robotic arm with a cubic end-effector and a stereo camera to shape granular media into desired target structures. We show the importance of compact observations and concise reward formulations for the large configuration space, validating our design choices with an ablation study. Our results demonstrate the effectiveness of the proposed approach for the training of visual policies that manipulate granular media including their real-world deployment, significantly outperforming two baseline approaches in terms of target shape accuracy.
>
---
#### [replaced 016] Semi-SMD: Semi-Supervised Metric Depth Estimation via Surrounding Cameras for Autonomous Driving
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.19713v3](http://arxiv.org/pdf/2503.19713v3)**

> **作者:** Yusen Xie; Zhengmin Huang; Shaojie Shen; Jun Ma
>
> **摘要:** In this paper, we introduce Semi-SMD, a novel metric depth estimation framework tailored for surrounding cameras equipment in autonomous driving. In this work, the input data consists of adjacent surrounding frames and camera parameters. We propose a unified spatial-temporal-semantic fusion module to construct the visual fused features. Cross-attention components for surrounding cameras and adjacent frames are utilized to focus on metric scale information refinement and temporal feature matching. Building on this, we propose a pose estimation framework using surrounding cameras, their corresponding estimated depths, and extrinsic parameters, which effectively address the scale ambiguity in multi-camera setups. Moreover, semantic world model and monocular depth estimation world model are integrated to supervised the depth estimation, which improve the quality of depth estimation. We evaluate our algorithm on DDAD and nuScenes datasets, and the results demonstrate that our method achieves state-of-the-art performance in terms of surrounding camera based depth estimation quality. The source code will be available on https://github.com/xieyuser/Semi-SMD.
>
---
#### [replaced 017] T-araVLN: Translator for Agricultural Robotic Agents on Vision-and-Language Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.06644v2](http://arxiv.org/pdf/2509.06644v2)**

> **作者:** Xiaobei Zhao; Xingqi Lyu; Xiang Li
>
> **摘要:** Agricultural robotic agents have been becoming powerful helpers in a wide range of agricultural tasks, nevertheless, still heavily rely on manual operation or untransportable railway for movement. The AgriVLN method and the A2A benchmark pioneeringly extend Vision-and-Language Navigation (VLN) to the agricultural domain, enabling agents navigate to the target position following the natural language instructions. AgriVLN effectively understands the simple instructions, however, often misunderstands the complicated instructions. To bridge this gap, we propose the method of Translator for Agricultural Robotic Agents on Vision-and-Language Navigation (T-araVLN), in which the Instruction Translator module translates the original instruction to be both refined and precise. Being evaluated on the A2A benchmark, our T-araVLN effectively improves SR from 0.47 to 0.63 and reduces NE from 2.91m to 2.28m, demonstrating the state-of-the-art performance in the agricultural domain. Code: https://github.com/AlexTraveling/T-araVLN.
>
---
#### [replaced 018] TrojanRobot: Physical-world Backdoor Attacks Against VLM-based Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.11683v4](http://arxiv.org/pdf/2411.11683v4)**

> **作者:** Xianlong Wang; Hewen Pan; Hangtao Zhang; Minghui Li; Shengshan Hu; Ziqi Zhou; Lulu Xue; Peijin Guo; Aishan Liu; Leo Yu Zhang; Xiaohua Jia
>
> **摘要:** Robotic manipulation in the physical world is increasingly empowered by \textit{large language models} (LLMs) and \textit{vision-language models} (VLMs), leveraging their understanding and perception capabilities. Recently, various attacks against such robotic policies have been proposed, with backdoor attacks drawing considerable attention for their high stealth and strong persistence capabilities. However, existing backdoor efforts are limited to simulators and suffer from physical-world realization. To address this, we propose \textit{TrojanRobot}, a highly stealthy and broadly effective robotic backdoor attack in the physical world. Specifically, we introduce a module-poisoning approach by embedding a backdoor module into the modular robotic policy, enabling backdoor control over the policy's visual perception module thereby backdooring the entire robotic policy. Our vanilla implementation leverages a backdoor-finetuned VLM to serve as the backdoor module. To enhance its generalization in physical environments, we propose a prime implementation, leveraging the LVLM-as-a-backdoor paradigm and developing three types of prime attacks, \ie, \textit{permutation}, \textit{stagnation}, and \textit{intentional} attacks, thus achieving finer-grained backdoors. Extensive experiments on the UR3e manipulator with 18 task instructions using robotic policies based on four VLMs demonstrate the broad effectiveness and physical-world stealth of TrojanRobot. Our attack's video demonstrations are available via a github link https://trojanrobot.github.io.
>
---
