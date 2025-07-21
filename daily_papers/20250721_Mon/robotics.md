# 机器人 cs.RO

- **最新发布 27 篇**

- **更新 15 篇**

## 最新发布

#### [new 001] Safety Certification in the Latent space using Control Barrier Functions and World Models
- **分类: cs.RO; cs.CV; cs.LG; cs.SY; eess.SY**

- **简介: 该论文属于安全控制任务，旨在解决从视觉数据中合成安全控制器时依赖大量标注安全关键数据的问题。论文提出了一种半监督框架，结合控制屏障函数与世界模型，在潜在空间中学习安全控制器，减少对标注数据的依赖，提升安全控制的可扩展性与数据效率。**

- **链接: [http://arxiv.org/pdf/2507.13871v1](http://arxiv.org/pdf/2507.13871v1)**

> **作者:** Mehul Anand; Shishir Kolathaya
>
> **备注:** 6 pages, 6 figures. arXiv admin note: text overlap with arXiv:2409.12616
>
> **摘要:** Synthesising safe controllers from visual data typically requires extensive supervised labelling of safety-critical data, which is often impractical in real-world settings. Recent advances in world models enable reliable prediction in latent spaces, opening new avenues for scalable and data-efficient safe control. In this work, we introduce a semi-supervised framework that leverages control barrier certificates (CBCs) learned in the latent space of a world model to synthesise safe visuomotor policies. Our approach jointly learns a neural barrier function and a safe controller using limited labelled data, while exploiting the predictive power of modern vision transformers for latent dynamics modelling.
>
---
#### [new 002] Context-Aware Behavior Learning with Heuristic Motion Memory for Underwater Manipulation
- **分类: cs.RO**

- **简介: 该论文属于水下机器人自主运动规划任务，旨在解决动态海洋环境中运动规划效率低、适应性差的问题。作者提出了一种结合启发式运动空间（HMS）与贝叶斯网络的自适应运动规划框架，利用先验经验与实时传感数据优化路径，提升计算效率与实时规划能力。**

- **链接: [http://arxiv.org/pdf/2507.14099v1](http://arxiv.org/pdf/2507.14099v1)**

> **作者:** Markus Buchholz; Ignacio Carlucho; Michele Grimaldi; Maria Koskinopoulou; Yvan R. Petillot
>
> **备注:** Accepted at 2025 IEEE International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Autonomous motion planning is critical for efficient and safe underwater manipulation in dynamic marine environments. Current motion planning methods often fail to effectively utilize prior motion experiences and adapt to real-time uncertainties inherent in underwater settings. In this paper, we introduce an Adaptive Heuristic Motion Planner framework that integrates a Heuristic Motion Space (HMS) with Bayesian Networks to enhance motion planning for autonomous underwater manipulation. Our approach employs the Probabilistic Roadmap (PRM) algorithm within HMS to optimize paths by minimizing a composite cost function that accounts for distance, uncertainty, energy consumption, and execution time. By leveraging HMS, our framework significantly reduces the search space, thereby boosting computational performance and enabling real-time planning capabilities. Bayesian Networks are utilized to dynamically update uncertainty estimates based on real-time sensor data and environmental conditions, thereby refining the joint probability of path success. Through extensive simulations and real-world test scenarios, we showcase the advantages of our method in terms of enhanced performance and robustness. This probabilistic approach significantly advances the capability of autonomous underwater robots, ensuring optimized motion planning in the face of dynamic marine challenges.
>
---
#### [new 003] Safe Robotic Capsule Cleaning with Integrated Transpupillary and Intraocular Optical Coherence Tomography
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于医疗机器人任务，旨在解决白内障术后后发性白内障的治疗问题。作者开发了一种结合经瞳孔和眼内光学相干断层扫描（OCT）的机器人系统，实现对晶状体囊的精确建模与实时反馈，从而安全地进行囊袋清洁手术。**

- **链接: [http://arxiv.org/pdf/2507.13650v1](http://arxiv.org/pdf/2507.13650v1)**

> **作者:** Yu-Ting Lai; Yasamin Foroutani; Aya Barzelay; Tsu-Chin Tsao
>
> **备注:** 12 pages, 27 figures
>
> **摘要:** Secondary cataract is one of the most common complications of vision loss due to the proliferation of residual lens materials that naturally grow on the lens capsule after cataract surgery. A potential treatment is capsule cleaning, a surgical procedure that requires enhanced visualization of the entire capsule and tool manipulation on the thin membrane. This article presents a robotic system capable of performing the capsule cleaning procedure by integrating a standard transpupillary and an intraocular optical coherence tomography probe on a surgical instrument for equatorial capsule visualization and real-time tool-to-tissue distance feedback. Using robot precision, the developed system enables complete capsule mapping in the pupillary and equatorial regions with in-situ calibration of refractive index and fiber offset, which are still current challenges in obtaining an accurate capsule model. To demonstrate effectiveness, the capsule mapping strategy was validated through five experimental trials on an eye phantom that showed reduced root-mean-square errors in the constructed capsule model, while the cleaning strategy was performed in three ex-vivo pig eyes without tissue damage.
>
---
#### [new 004] NeHMO: Neural Hamilton-Jacobi Reachability Learning for Decentralized Safe Multi-Agent Motion Planning
- **分类: cs.RO**

- **简介: 该论文属于多智能体运动规划任务，旨在解决分布式环境下智能体间避碰与安全规划问题。现有方法面临通信依赖或计算复杂度高的挑战，为此论文提出NeHMO方法，结合神经网络与哈密尔顿-雅可比可达性分析，实现高效、安全的多智能体实时轨迹规划。**

- **链接: [http://arxiv.org/pdf/2507.13940v1](http://arxiv.org/pdf/2507.13940v1)**

> **作者:** Qingyi Chen; Ahmed H. Qureshi
>
> **摘要:** Safe Multi-Agent Motion Planning (MAMP) is a significant challenge in robotics. Despite substantial advancements, existing methods often face a dilemma. Decentralized algorithms typically rely on predicting the behavior of other agents, sharing contracts, or maintaining communication for safety, while centralized approaches struggle with scalability and real-time decision-making. To address these challenges, we introduce Neural Hamilton-Jacobi Reachability Learning (HJR) for Decentralized Multi-Agent Motion Planning. Our method provides scalable neural HJR modeling to tackle high-dimensional configuration spaces and capture worst-case collision and safety constraints between agents. We further propose a decentralized trajectory optimization framework that incorporates the learned HJR solutions to solve MAMP tasks in real-time. We demonstrate that our method is both scalable and data-efficient, enabling the solution of MAMP problems in higher-dimensional scenarios with complex collision constraints. Our approach generalizes across various dynamical systems, including a 12-dimensional dual-arm setup, and outperforms a range of state-of-the-art techniques in successfully addressing challenging MAMP tasks. Video demonstrations are available at https://youtu.be/IZiePX0p1Mc.
>
---
#### [new 005] A Study of Teleoperation Methods in a Simulated Virtual Eye Surgery Environment
- **分类: cs.RO**

- **简介: 论文研究了虚拟现实环境中不同控制模式和缩放因子对遥操作眼科手术性能的影响。任务是评估遥操作方法在模拟玻璃体视网膜手术中的效果。通过让医生和工程师完成手术任务，发现内部控制模式在高缩放因子（20或30）下表现最佳，优化控制方式可提升手术效率和安全性。**

- **链接: [http://arxiv.org/pdf/2507.13654v1](http://arxiv.org/pdf/2507.13654v1)**

> **作者:** Haoran Wang; Yasamin Foroutani; Matthew Nepo; Mercedes Rodriguez; Ji Ma; Jean-Pierre Hubschman; Tsu-Chin Tsao; Jacob Rosen
>
> **备注:** 9 pages, 11 figures
>
> **摘要:** This paper examines the performance of Inside and Outside Control modes at various scaling factors in a simulated vitreoretinal surgical setting. The IRISS teleoperated surgical system's console (cockpit) was adapted to project a simulated microscope view of an intraocular setup to a virtual reality (VR) headset. Five experienced vitreoretinal surgeons and five engineers with no surgical experience used the system to perform tasks common to vitreoretinal surgery. Experimental results indicate that Inside Control methods at higher scaling factors (20 or 30) achieved the best performance overall, though the optimal scaling factor may vary by task and complexity. Optimizing control methods and scaling factors could lead to improvements in surgical efficiency and accuracy, as well as minimize risks in future robotic-assisted intraocular procedures.
>
---
#### [new 006] SaWa-ML: Structure-Aware Pose Correction and Weight Adaptation-Based Robust Multi-Robot Localization
- **分类: cs.RO**

- **简介: 该论文属于多机器人定位任务，旨在解决多机器人系统中因里程计误差累积导致的长期漂移问题。通过融合视觉-惯性-UWB传感器数据，提出SaWa-ML方法，实现结构感知的姿态校正与自适应权重优化，提升定位鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.13702v1](http://arxiv.org/pdf/2507.13702v1)**

> **作者:** Junho Choi; Kihwan Ryoo; Jeewon Kim; Taeyun Kim; Eungchang Lee; Myeongwoo Jeong; Kevin Christiansen Marsim; Hyungtae Lim; Hyun Myung
>
> **备注:** This paper has been accepted to the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Multi-robot localization is a crucial task for implementing multi-robot systems. Numerous researchers have proposed optimization-based multi-robot localization methods that use camera, IMU, and UWB sensors. Nevertheless, characteristics of individual robot odometry estimates and distance measurements between robots used in the optimization are not sufficiently considered. In addition, previous researches were heavily influenced by the odometry accuracy that is estimated from individual robots. Consequently, long-term drift error caused by error accumulation is potentially inevitable. In this paper, we propose a novel visual-inertial-range-based multi-robot localization method, named SaWa-ML, which enables geometric structure-aware pose correction and weight adaptation-based robust multi-robot localization. Our contributions are twofold: (i) we leverage UWB sensor data, whose range error does not accumulate over time, to first estimate the relative positions between robots and then correct the positions of each robot, thus reducing long-term drift errors, (ii) we design adaptive weights for robot pose correction by considering the characteristics of the sensor data and visual-inertial odometry estimates. The proposed method has been validated in real-world experiments, showing a substantial performance increase compared with state-of-the-art algorithms.
>
---
#### [new 007] SCOPE for Hexapod Gait Generation
- **分类: cs.RO; cs.NE**

- **简介: 该论文属于机器人控制任务，旨在解决六足机器人步态生成中输入空间复杂导致进化算法效果下降的问题。作者提出SCOPE方法，利用离散余弦变换压缩输入数据维度，从而提升策略进化的效率，并通过实验验证其有效性。**

- **链接: [http://arxiv.org/pdf/2507.13539v1](http://arxiv.org/pdf/2507.13539v1)**

> **作者:** Jim O'Connor; Jay B. Nash; Derin Gezgin; Gary B. Parker
>
> **备注:** IJCCI Conference on Evolutionary Computation and Theory and Applications, 2025
>
> **摘要:** Evolutionary methods have previously been shown to be an effective learning method for walking gaits on hexapod robots. However, the ability of these algorithms to evolve an effective policy rapidly degrades as the input space becomes more complex. This degradation is due to the exponential growth of the solution space, resulting from an increasing parameter count to handle a more complex input. In order to address this challenge, we introduce Sparse Cosine Optimized Policy Evolution (SCOPE). SCOPE utilizes the Discrete Cosine Transform (DCT) to learn directly from the feature coefficients of an input matrix. By truncating the coefficient matrix returned by the DCT, we can reduce the dimensionality of an input while retaining the highest energy features of the original input. We demonstrate the effectiveness of this method by using SCOPE to learn the gait of a hexapod robot. The hexapod controller is given a matrix input containing time-series information of previous poses, which are then transformed to gait parameters by an evolved policy. In this task, the addition of SCOPE to a reference algorithm achieves a 20% increase in efficacy. SCOPE achieves this result by reducing the total input size of the time-series pose data from 2700 to 54, a 98% decrease. Additionally, SCOPE is capable of compressing an input to any output shape, provided that each output dimension is no greater than the corresponding input dimension. This paper demonstrates that SCOPE is capable of significantly compressing the size of an input to an evolved controller, resulting in a statistically significant gain in efficacy.
>
---
#### [new 008] A Minimalist Controller for Autonomously Self-Aggregating Robotic Swarms: Enabling Compact Formations in Multitasking Scenarios
- **分类: cs.RO; cs.MA**

- **简介: 论文研究多任务环境下机器人集群的自主聚集控制。目标是实现多个机器人组同时形成紧凑集群，解决现有方法中形成的群体不够紧凑或缺乏完全自主的问题。工作包括设计仅依赖视线传感器的控制器，并通过仿真验证其在不同规模下的性能，提升了聚集效果与扩展性。**

- **链接: [http://arxiv.org/pdf/2507.13969v1](http://arxiv.org/pdf/2507.13969v1)**

> **作者:** Maria Eduarda Silva de Macedo; Ana Paula Chiarelli de Souza; Roberto Silvio Ubertino Rosso Jr.; Yuri Kaszubowski Lopes
>
> **备注:** 7 pages total (6 pages of content + 1 page of references). Short paper manuscript submitted to TAROS 2025
>
> **摘要:** The deployment of simple emergent behaviors in swarm robotics has been well-rehearsed in the literature. A recent study has shown how self-aggregation is possible in a multitask approach -- where multiple self-aggregation task instances occur concurrently in the same environment. The multitask approach poses new challenges, in special, how the dynamic of each group impacts the performance of others. So far, the multitask self-aggregation of groups of robots suffers from generating a circular formation -- that is not fully compact -- or is not fully autonomous. In this paper, we present a multitask self-aggregation where groups of homogeneous robots sort themselves into different compact clusters, relying solely on a line-of-sight sensor. Our multitask self-aggregation behavior was able to scale well and achieve a compact formation. We report scalability results from a series of simulation trials with different configurations in the number of groups and the number of robots per group. We were able to improve the multitask self-aggregation behavior performance in terms of the compactness of the clusters, keeping the proportion of clustered robots found in other studies.
>
---
#### [new 009] AGENTS-LLM: Augmentative GENeration of Challenging Traffic Scenarios with an Agentic LLM Framework
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶测试任务，旨在解决罕见关键交通场景生成困难的问题。现有方法依赖大量数据或手动设计，效率低且控制性差。论文提出AGENTS-LLM框架，利用小规模大语言模型通过自然语言描述生成高质量、可控的挑战性交通场景，提升自动驾驶系统的测试效率与质量。**

- **链接: [http://arxiv.org/pdf/2507.13729v1](http://arxiv.org/pdf/2507.13729v1)**

> **作者:** Yu Yao; Salil Bhatnagar; Markus Mazzola; Vasileios Belagiannis; Igor Gilitschenski; Luigi Palmieri; Simon Razniewski; Marcel Hallgarten
>
> **摘要:** Rare, yet critical, scenarios pose a significant challenge in testing and evaluating autonomous driving planners. Relying solely on real-world driving scenes requires collecting massive datasets to capture these scenarios. While automatic generation of traffic scenarios appears promising, data-driven models require extensive training data and often lack fine-grained control over the output. Moreover, generating novel scenarios from scratch can introduce a distributional shift from the original training scenes which undermines the validity of evaluations especially for learning-based planners. To sidestep this, recent work proposes to generate challenging scenarios by augmenting original scenarios from the test set. However, this involves the manual augmentation of scenarios by domain experts. An approach that is unable to meet the demands for scale in the evaluation of self-driving systems. Therefore, this paper introduces a novel LLM-agent based framework for augmenting real-world traffic scenarios using natural language descriptions, addressing the limitations of existing methods. A key innovation is the use of an agentic design, enabling fine-grained control over the output and maintaining high performance even with smaller, cost-effective LLMs. Extensive human expert evaluation demonstrates our framework's ability to accurately adhere to user intent, generating high quality augmented scenarios comparable to those created manually.
>
---
#### [new 010] Improving Low-Cost Teleoperation: Augmenting GELLO with Force
- **分类: cs.RO; cs.HC; cs.LG**

- **简介: 该论文属于机器人遥操作任务，旨在提升低成本GELLO系统的操作性能。论文通过增加力反馈和在数据收集与训练中引入力信息，改进原有仅依赖关节位置控制的系统。实验验证了改进在模拟和真实灵巧操作任务中的有效性，提升了任务成功率并获得用户偏好。**

- **链接: [http://arxiv.org/pdf/2507.13602v1](http://arxiv.org/pdf/2507.13602v1)**

> **作者:** Shivakanth Sujit; Luca Nunziante; Dan Ogawa Lillrank; Rousslan Fernand Julien Dossa; Kai Arulkumaran
>
> **备注:** Accepted at the 2025 IEEE/SICE International Symposium on System Integration
>
> **摘要:** In this work we extend the low-cost GELLO teleoperation system, initially designed for joint position control, with additional force information. Our first extension is to implement force feedback, allowing users to feel resistance when interacting with the environment. Our second extension is to add force information into the data collection process and training of imitation learning models. We validate our additions by implementing these on a GELLO system with a Franka Panda arm as the follower robot, performing a user study, and comparing the performance of policies trained with and without force information on a range of simulated and real dexterous manipulation tasks. Qualitatively, users with robotics experience preferred our controller, and the addition of force inputs improved task success on the majority of tasks.
>
---
#### [new 011] A segmented robot grasping perception neural network for edge AI
- **分类: cs.RO; cs.AI; I.2; I.2.9; I.2.10**

- **简介: 该论文属于机器人抓取感知任务，旨在解决边缘设备上实时、低功耗抓取识别问题。作者在RISC-V芯片上实现了基于热图引导的6-Dof抓取检测模型，并通过模型优化技术实现全片上推理，验证了低功耗MCU在自主操作中的可行性。**

- **链接: [http://arxiv.org/pdf/2507.13970v1](http://arxiv.org/pdf/2507.13970v1)**

> **作者:** Casper Bröcheler; Thomas Vroom; Derrick Timmermans; Alan van den Akker; Guangzhi Tang; Charalampos S. Kouzinopoulos; Rico Möckel
>
> **备注:** Accepted by SMC 2025
>
> **摘要:** Robotic grasping, the ability of robots to reliably secure and manipulate objects of varying shapes, sizes and orientations, is a complex task that requires precise perception and control. Deep neural networks have shown remarkable success in grasp synthesis by learning rich and abstract representations of objects. When deployed at the edge, these models can enable low-latency, low-power inference, making real-time grasping feasible in resource-constrained environments. This work implements Heatmap-Guided Grasp Detection, an end-to-end framework for the detection of 6-Dof grasp poses, on the GAP9 RISC-V System-on-Chip. The model is optimised using hardware-aware techniques, including input dimensionality reduction, model partitioning, and quantisation. Experimental evaluation on the GraspNet-1Billion benchmark validates the feasibility of fully on-chip inference, highlighting the potential of low-power MCUs for real-time, autonomous manipulation.
>
---
#### [new 012] Design Analysis of an Innovative Parallel Robot for Minimally Invasive Pancreatic Surgery
- **分类: cs.RO**

- **简介: 该论文设计并分析了一种用于微创胰腺手术的新型并联机器人，提出了两种4自由度结构（ATHENA-1和ATHENA-2），通过运动学分析、3D建模、有限元仿真和工作空间评估，比较其刚度与适用性，最终选择最优结构用于实验样机开发。**

- **链接: [http://arxiv.org/pdf/2507.13787v1](http://arxiv.org/pdf/2507.13787v1)**

> **作者:** Doina Pisla; Alexandru Pusca; Andrei Caprariu; Adrian Pisla; Bogdan Gherman; Calin Vaida; Damien Chablat
>
> **摘要:** This paper focuses on the design of a parallel robot designed for robotic assisted minimally invasive pancreatic surgery. Two alternative architectures, called ATHENA-1 and ATHENA-2, each with 4 degrees of freedom (DOF) are proposed. Their kinematic schemes are presented, and the conceptual 3D CAD models are illustrated. Based on these, two Finite Element Method (FEM) simulations were performed to determine which architecture has the higher stiffness. A workspace quantitative analysis is performed to further assess the usability of the two proposed parallel architectures related to the medical tasks. The obtained results are used to select the architecture which fit the required design criteria and will be used to develop the experimental model of the surgical robot.
>
---
#### [new 013] MorphIt: Flexible Spherical Approximation of Robot Morphology for Representation-driven Adaptation
- **分类: cs.RO**

- **简介: 论文提出MorphIt算法，用于机器人形态的球形近似，解决固定形态表示在不同任务中效率与精度不匹配的问题。通过自动优化框架，实现几何精度与计算成本的平衡，提升碰撞检测、交互模拟和狭窄空间导航能力。属于机器人形态表示与自适应优化任务。**

- **链接: [http://arxiv.org/pdf/2507.14061v1](http://arxiv.org/pdf/2507.14061v1)**

> **作者:** Nataliya Nechyporenko; Yutong Zhang; Sean Campbell; Alessandro Roncone
>
> **摘要:** What if a robot could rethink its own morphological representation to better meet the demands of diverse tasks? Most robotic systems today treat their physical form as a fixed constraint rather than an adaptive resource, forcing the same rigid geometric representation to serve applications with vastly different computational and precision requirements. We introduce MorphIt, a novel algorithm for approximating robot morphology using spherical primitives that balances geometric accuracy with computational efficiency. Unlike existing approaches that rely on either labor-intensive manual specification or inflexible computational methods, MorphIt implements an automatic gradient-based optimization framework with tunable parameters that provides explicit control over the physical fidelity versus computational cost tradeoff. Quantitative evaluations demonstrate that MorphIt outperforms baseline approaches (Variational Sphere Set Approximation and Adaptive Medial-Axis Approximation) across multiple metrics, achieving better mesh approximation with fewer spheres and reduced computational overhead. Our experiments show enhanced robot capabilities in collision detection accuracy, contact-rich interaction simulation, and navigation through confined spaces. By dynamically adapting geometric representations to task requirements, robots can now exploit their physical embodiment as an active resource rather than an inflexible parameter, opening new frontiers for manipulation in environments where physical form must continuously balance precision with computational tractability.
>
---
#### [new 014] AeroThrow: An Autonomous Aerial Throwing System for Precise Payload Delivery
- **分类: cs.RO**

- **简介: 论文提出AeroThrow系统，属于无人机精准投递任务。解决空中投递中控制模式切换突变、系统延迟和误差问题。通过引入主动自由度补偿误差，结合NMPC框架与扰动补偿策略，提升投掷轨迹精度和系统鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.13903v1](http://arxiv.org/pdf/2507.13903v1)**

> **作者:** Ziliang Li; Hongming Chen; Yiyang Lin; Biyu Ye; Ximin Lyu
>
> **摘要:** Autonomous aerial systems play an increasingly vital role in a wide range of applications, particularly for transport and delivery tasks in complex environments. In airdrop missions, these platforms face the dual challenges of abrupt control mode switching and inherent system delays along with control errors. To address these issues, this paper presents an autonomous airdrop system based on an aerial manipulator (AM). The introduction of additional actuated degrees of freedom enables active compensation for UAV tracking errors. By imposing smooth and continuous constraints on the parabolic landing point, the proposed approach generates aerial throwing trajectories that are less sensitive to the timing of payload release. A hierarchical disturbance compensation strategy is incorporated into the Nonlinear Model Predictive Control (NMPC) framework to mitigate the effects of sudden changes in system parameters, while the predictive capabilities of NMPC are further exploited to improve the precision of aerial throwing. Both simulation and real-world experimental results demonstrate that the proposed system achieves greater agility and precision in airdrop missions.
>
---
#### [new 015] Iteratively Learning Muscle Memory for Legged Robots to Master Adaptive and High Precision Locomotion
- **分类: cs.RO**

- **简介: 论文提出一种结合迭代学习控制（ILC）与类肌肉记忆扭矩库（TL）的框架，旨在提升足式机器人在复杂环境中的运动精度与适应能力。该研究属于机器人控制任务，解决非结构化环境中机器人步态控制与快速适应问题。通过融合模型预测与实时学习，实现高精度轨迹跟踪与低计算开销。**

- **链接: [http://arxiv.org/pdf/2507.13662v1](http://arxiv.org/pdf/2507.13662v1)**

> **作者:** Jing Cheng; Yasser G. Alqaham; Zhenyu Gan; Amit K. Sanyal
>
> **摘要:** This paper presents a scalable and adaptive control framework for legged robots that integrates Iterative Learning Control (ILC) with a biologically inspired torque library (TL), analogous to muscle memory. The proposed method addresses key challenges in robotic locomotion, including accurate trajectory tracking under unmodeled dynamics and external disturbances. By leveraging the repetitive nature of periodic gaits and extending ILC to nonperiodic tasks, the framework enhances accuracy and generalization across diverse locomotion scenarios. The control architecture is data-enabled, combining a physics-based model derived from hybrid-system trajectory optimization with real-time learning to compensate for model uncertainties and external disturbances. A central contribution is the development of a generalized TL that stores learned control profiles and enables rapid adaptation to changes in speed, terrain, and gravitational conditions-eliminating the need for repeated learning and significantly reducing online computation. The approach is validated on the bipedal robot Cassie and the quadrupedal robot A1 through extensive simulations and hardware experiments. Results demonstrate that the proposed framework reduces joint tracking errors by up to 85% within a few seconds and enables reliable execution of both periodic and nonperiodic gaits, including slope traversal and terrain adaptation. Compared to state-of-the-art whole-body controllers, the learned skills eliminate the need for online computation during execution and achieve control update rates exceeding 30x those of existing methods. These findings highlight the effectiveness of integrating ILC with torque memory as a highly data-efficient and practical solution for legged locomotion in unstructured and dynamic environments.
>
---
#### [new 016] Design of a Modular Mobile Inspection and Maintenance Robot for an Orbital Servicing Hub
- **分类: cs.RO**

- **简介: 论文设计了一种模块化移动检测与维护机器人（MIM），用于轨道服务枢纽的自主检测与维护任务，旨在解决空间硬件组件状态监测与设施维护问题。工作包括MIM的概念设计、机械电子结构、传感器配置及测试，支持多传感器扩展与机械臂协作维护操作。**

- **链接: [http://arxiv.org/pdf/2507.14059v1](http://arxiv.org/pdf/2507.14059v1)**

> **作者:** Tianyuan Wang; Mark A Post; Mathieu Deremetz
>
> **备注:** In proceedings of the Towards Autonomous Robotic Systems 2025 conference (TAROS 2025), York, UK 6 pages, one page of references, 6 figures
>
> **摘要:** The use of autonomous robots in space is an essential part of the "New Space" commercial ecosystem of assembly and re-use of space hardware components in Earth orbit and beyond. The STARFAB project aims to create a ground demonstration of an orbital automated warehouse as a hub for sustainable commercial operations and servicing. A critical part of this fully-autonomous robotic facility will be the capability to monitor, inspect, and assess the condition of both the components stored in the warehouse, and the STARFAB facility itself. This paper introduces ongoing work on the STARFAB Mobile Inspection Module (MIM). The MIM uses Standard Interconnects (SI) so that it can be carried by Walking Manipulators (WM) as an independently-mobile robot, and multiple MIMs can be stored and retrieved as needed for operations on STARFAB. The MIM carries high-resolution cameras, a 3D profilometer, and a thermal imaging sensor, with the capability to add other modular sensors. A grasping tool and torque wrench are stored within the modular body for use by an attached WM for maintenance operations. Implementation and testing is still ongoing at the time of writing. This paper details the concept of operations for the MIM as an on-orbit autonomous inspection and maintenance system, the mechanical and electronic design of the MIM, and the sensors package used for non-destructive testing.
>
---
#### [new 017] Improved particle swarm optimization algorithm: multi-target trajectory optimization for swarm drones
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于无人机实时轨迹规划任务，旨在解决传统算法在动态环境中计算效率低、适应性差的问题。作者提出PE-PSO算法，结合持久探索机制和熵调节策略，提升多无人机协同轨迹优化的实时性与质量，并通过仿真验证其优越性。**

- **链接: [http://arxiv.org/pdf/2507.13647v1](http://arxiv.org/pdf/2507.13647v1)**

> **作者:** Minze Li; Wei Zhao; Ran Chen; Mingqiang Wei
>
> **备注:** 8 papers,7 figures
>
> **摘要:** Real-time trajectory planning for unmanned aerial vehicles (UAVs) in dynamic environments remains a key challenge due to high computational demands and the need for fast, adaptive responses. Traditional Particle Swarm Optimization (PSO) methods, while effective for offline planning, often struggle with premature convergence and latency in real-time scenarios. To overcome these limitations, we propose PE-PSO, an enhanced PSO-based online trajectory planner. The method introduces a persistent exploration mechanism to preserve swarm diversity and an entropy-based parameter adjustment strategy to dynamically adapt optimization behavior. UAV trajectories are modeled using B-spline curves, which ensure path smoothness while reducing optimization complexity. To extend this capability to UAV swarms, we develop a multi-agent framework that combines genetic algorithm (GA)-based task allocation with distributed PE-PSO, supporting scalable and coordinated trajectory generation. The distributed architecture allows for parallel computation and decentralized control, enabling effective cooperation among agents while maintaining real-time performance. Comprehensive simulations demonstrate that the proposed framework outperforms conventional PSO and other swarm-based planners across several metrics, including trajectory quality, energy efficiency, obstacle avoidance, and computation time. These results confirm the effectiveness and applicability of PE-PSO in real-time multi-UAV operations under complex environmental conditions.
>
---
#### [new 018] ERR@HRI 2.0 Challenge: Multimodal Detection of Errors and Failures in Human-Robot Conversations
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文属于人机交互任务，旨在解决基于大语言模型的对话机器人在交流中易出现错误的问题。论文构建了一个包含16小时多模态数据的ERR@HRI 2.0数据集，标注了机器人错误及用户纠正意图，用于训练和评估多模态机器学习模型，以提升人机对话中错误检测能力。**

- **链接: [http://arxiv.org/pdf/2507.13468v1](http://arxiv.org/pdf/2507.13468v1)**

> **作者:** Shiye Cao; Maia Stiber; Amama Mahmood; Maria Teresa Parreira; Wendy Ju; Micol Spitale; Hatice Gunes; Chien-Ming Huang
>
> **摘要:** The integration of large language models (LLMs) into conversational robots has made human-robot conversations more dynamic. Yet, LLM-powered conversational robots remain prone to errors, e.g., misunderstanding user intent, prematurely interrupting users, or failing to respond altogether. Detecting and addressing these failures is critical for preventing conversational breakdowns, avoiding task disruptions, and sustaining user trust. To tackle this problem, the ERR@HRI 2.0 Challenge provides a multimodal dataset of LLM-powered conversational robot failures during human-robot conversations and encourages researchers to benchmark machine learning models designed to detect robot failures. The dataset includes 16 hours of dyadic human-robot interactions, incorporating facial, speech, and head movement features. Each interaction is annotated with the presence or absence of robot errors from the system perspective, and perceived user intention to correct for a mismatch between robot behavior and user expectation. Participants are invited to form teams and develop machine learning models that detect these failures using multimodal data. Submissions will be evaluated using various performance metrics, including detection accuracy and false positive rate. This challenge represents another key step toward improving failure detection in human-robot interaction through social signal analysis.
>
---
#### [new 019] Hard-Stop Synthesis for Multi-DOF Compliant Mechanisms
- **分类: cs.RO**

- **简介: 该论文属于机械设计任务，旨在解决多自由度柔性机构在复杂载荷下易疲劳失效的问题。通过提出一种硬限位综合设计方法，优化接触面几何以最大化工作空间，同时保证机构处于弹性范围内，并在骨科植入铰链机构中验证了设计的有效性。**

- **链接: [http://arxiv.org/pdf/2507.13455v1](http://arxiv.org/pdf/2507.13455v1)**

> **作者:** Dean Chen; Armin Pomeroy; Brandon T. Peterson; Will Flanagan; He Kai Lim; Alexandra Stavrakis; Nelson F. SooHoo; Jonathan B. Hopkins; Tyler R. Clites
>
> **备注:** 42 pages, 17 figures. Under review at ASME Journal of Mechanical Design
>
> **摘要:** Compliant mechanisms have significant potential in precision applications due to their ability to guide motion without contact. However, an inherent vulnerability to fatigue and mechanical failure has hindered the translation of compliant mechanisms to real-world applications. This is particularly challenging in service environments where loading is complex and uncertain, and the cost of failure is high. In such cases, mechanical hard stops are critical to prevent yielding and buckling. Conventional hard-stop designs, which rely on stacking single-DOF limits, must be overly restrictive in multi-DOF space to guarantee safety in the presence of unknown loads. In this study, we present a systematic design synthesis method to guarantee overload protection in compliant mechanisms by integrating coupled multi-DOF motion limits within a single pair of compact hard-stop surfaces. Specifically, we introduce a theoretical and practical framework for optimizing the contact surface geometry to maximize the mechanisms multi-DOF working space while still ensuring that the mechanism remains within its elastic regime. We apply this synthesis method to a case study of a caged-hinge mechanism for orthopaedic implants, and provide numerical and experimental validation that the derived design offers reliable protection against fatigue, yielding, and buckling. This work establishes a foundation for precision hard-stop design in compliant systems operating under uncertain loads, which is a crucial step toward enabling the application of compliant mechanisms in real-world systems.
>
---
#### [new 020] EdgeVLA: Efficient Vision-Language-Action Models
- **分类: cs.RO; cs.CL**

- **简介: 该论文属于机器人视觉-语言-动作模型任务，旨在解决大规模模型在边缘设备部署的效率问题。论文提出了EdgeVLA，通过非自回归预测和小语言模型提升推理速度，实现与OpenVLA相当的训练效果，同时显著提高推理速度和内存效率。**

- **链接: [http://arxiv.org/pdf/2507.14049v1](http://arxiv.org/pdf/2507.14049v1)**

> **作者:** Paweł Budzianowski; Wesley Maa; Matthew Freed; Jingxiang Mo; Winston Hsiao; Aaron Xie; Tomasz Młoduchowski; Viraj Tipnis; Benjamin Bolte
>
> **摘要:** Vision-Language Models (VLMs) have emerged as a promising approach to address the data scarcity challenge in robotics, enabling the development of generalizable visuomotor control policies. While models like OpenVLA showcase the potential of this paradigm, deploying large-scale VLMs on resource-constrained mobile manipulation systems remains a significant hurdle. This paper introduces Edge VLA (EVLA), a novel approach designed to significantly enhance the inference speed of Vision-Language-Action (VLA) models. EVLA maintains the representational power of these models while enabling real-time performance on edge devices. We achieve this through two key innovations: 1) Eliminating the autoregressive requirement for end-effector position prediction, leading to a 7x speedup in inference, and 2) Leveraging the efficiency of Small Language Models (SLMs), demonstrating comparable training performance to larger models with significantly reduced computational demands. Our early results demonstrate that EVLA achieves comparable training characteristics to OpenVLA while offering substantial gains in inference speed and memory efficiency. We release our model checkpoints and training \href{https://github.com/kscalelabs/evla }{codebase} to foster further research.
>
---
#### [new 021] A multi-strategy improved snake optimizer for three-dimensional UAV path planning and engineering problems
- **分类: cs.RO; cs.AI; cs.CE**

- **简介: 该论文属于优化算法研究任务，旨在解决传统Snake Optimizer（SO）算法收敛速度慢、易陷入局部最优的问题。作者提出了一种多策略改进的Snake Optimizer（MISO），通过引入自适应随机扰动、Levy飞行策略及精英领导与布朗运动结合的位置更新策略，提升了算法性能。实验验证了MISO在CEC测试函数、无人机三维路径规划和工程问题中的优越性。**

- **链接: [http://arxiv.org/pdf/2507.14043v1](http://arxiv.org/pdf/2507.14043v1)**

> **作者:** Genliang Li; Yaxin Cui; Jinyu Su
>
> **备注:** 59 pages, 22 figures
>
> **摘要:** Metaheuristic algorithms have gained widespread application across various fields owing to their ability to generate diverse solutions. One such algorithm is the Snake Optimizer (SO), a progressive optimization approach. However, SO suffers from the issues of slow convergence speed and susceptibility to local optima. In light of these shortcomings, we propose a novel Multi-strategy Improved Snake Optimizer (MISO). Firstly, we propose a new adaptive random disturbance strategy based on sine function to alleviate the risk of getting trapped in a local optimum. Secondly, we introduce adaptive Levy flight strategy based on scale factor and leader and endow the male snake leader with flight capability, which makes it easier for the algorithm to leap out of the local optimum and find the global optimum. More importantly, we put forward a position update strategy combining elite leadership and Brownian motion, effectively accelerating the convergence speed while ensuring precision. Finally, to demonstrate the performance of MISO, we utilize 30 CEC2017 test functions and the CEC2022 test suite, comparing it with 11 popular algorithms across different dimensions to validate its effectiveness. Moreover, Unmanned Aerial Vehicle (UAV) has been widely used in various fields due to its advantages of low cost, high mobility and easy operation. However, the UAV path planning problem is crucial for flight safety and efficiency, and there are still challenges in establishing and optimizing the path model. Therefore, we apply MISO to the UAV 3D path planning problem as well as 6 engineering design problems to assess its feasibility in practical applications. The experimental results demonstrate that MISO exceeds other competitive algorithms in terms of solution quality and stability, establishing its strong potential for application.
>
---
#### [new 022] Fixed time convergence guarantees for Higher Order Control Barrier Functions
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文属于控制系统理论任务，旨在解决高阶控制屏障函数（HOCBF）缺乏固定时间收敛保证的问题。作者提出新方法，通过构造具有重复根的特征多项式，实现安全集的精确固定时间收敛。该方法提升了传统HOCBF在时间敏感与安全关键应用（如自主导航）中的性能，并在多个机器人系统上验证了有效性。**

- **链接: [http://arxiv.org/pdf/2507.13888v1](http://arxiv.org/pdf/2507.13888v1)**

> **作者:** Janani S K; Shishir Kolathaya
>
> **备注:** 6 PAGES, 2 FIGURES
>
> **摘要:** We present a novel method for designing higher-order Control Barrier Functions (CBFs) that guarantee convergence to a safe set within a user-specified finite. Traditional Higher Order CBFs (HOCBFs) ensure asymptotic safety but lack mechanisms for fixed-time convergence, which is critical in time-sensitive and safety-critical applications such as autonomous navigation. In contrast, our approach imposes a structured differential constraint using repeated roots in the characteristic polynomial, enabling closed-form polynomial solutions with exact convergence at a prescribed time. We derive conditions on the barrier function and its derivatives that ensure forward invariance and fixed-time reachability, and we provide an explicit formulation for second-order systems. Our method is evaluated on three robotic systems - a point-mass model, a unicycle, and a bicycle model and benchmarked against existing HOCBF approaches. Results demonstrate that our formulation reliably enforces convergence within the desired time, even when traditional methods fail. This work provides a tractable and robust framework for real-time control with provable finite-time safety guarantees.
>
---
#### [new 023] Conformal Contraction for Robust Nonlinear Control with Distribution-Free Uncertainty Quantification
- **分类: math.OC; cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于鲁棒控制任务，旨在解决存在非线性不确定性的动力系统中轨迹跟踪误差的指数有界性问题。作者提出了一种基于共形预测的收缩控制框架，结合数据驱动的不确定性预测，实现了无需不确定性模型的分布无关鲁棒控制，并通过数值仿真验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.13613v1](http://arxiv.org/pdf/2507.13613v1)**

> **作者:** Sihang Wei; Melkior Ornik; Hiroyasu Tsukamoto
>
> **备注:** IEEE CDC 2025 submission (accepted)
>
> **摘要:** We present a novel robust control framework for continuous-time, perturbed nonlinear dynamical systems with uncertainty that depends nonlinearly on both the state and control inputs. Unlike conventional approaches that impose structural assumptions on the uncertainty, our framework enhances contraction-based robust control with data-driven uncertainty prediction, remaining agnostic to the models of the uncertainty and predictor. We statistically quantify how reliably the contraction conditions are satisfied under dynamics with uncertainty via conformal prediction, thereby obtaining a distribution-free and finite-time probabilistic guarantee for exponential boundedness of the trajectory tracking error. We further propose the probabilistically robust control invariant (PRCI) tube for distributionally robust motion planning, within which the perturbed system trajectories are guaranteed to stay with a finite probability, without explicit knowledge of the uncertainty model. Numerical simulations validate the effectiveness of the proposed robust control framework and the performance of the PRCI tube.
>
---
#### [new 024] Conceptual and Design Principles for a Self-Referential Algorithm Mimicking Neuronal Assembly Functions
- **分类: cs.NE; cs.RO**

- **简介: 论文提出了一种基于自我参照语言（E-language）的算法框架（EGO），用于模拟神经元集合的认知过程，旨在从生命系统内部视角形式化基于经验的认知模型，解决传统方法中观察者视角的问题。已实现并测试了原型系统（EGO-P）。**

- **链接: [http://arxiv.org/pdf/2507.14011v1](http://arxiv.org/pdf/2507.14011v1)**

> **作者:** Paolo Totaro; Alberto Mangiante
>
> **摘要:** This article proposes a method to formalise models of cognitive processes grounded in experience, considering experience from the perspective of a living system and not from that of an observer of the living system. The perspective of a living system is defined by the need of the system to preserve the vital equilibria. The method is based on an algorithmic schema that we call Environment Generative Operator (EGO) and uses a self-referential language developed for this purpose which we call E-language. EGO simulates cognitive processes as operations on neuron assemblies as understood by Hebb. In this article we present an EGO prototype (EGO-P) which has already been implemented and tested.
>
---
#### [new 025] Depth3DLane: Fusing Monocular 3D Lane Detection with Self-Supervised Monocular Depth Estimation
- **分类: cs.CV; cs.RO**

- **简介: 论文提出Depth3DLane，用于单目三维车道检测任务，旨在解决缺乏空间信息、依赖昂贵传感器或真实深度数据的问题。方法融合自监督深度估计，提取场景点云与语义信息，结合3D车道锚点预测几何结构，并可估计相机参数，提升在无标定场景下的适用性与稳定性。**

- **链接: [http://arxiv.org/pdf/2507.13857v1](http://arxiv.org/pdf/2507.13857v1)**

> **作者:** Max van den Hoven; Kishaan Jeeveswaran; Pieter Piscaer; Thijs Wensveen; Elahe Arani; Bahram Zonooz
>
> **摘要:** Monocular 3D lane detection is essential for autonomous driving, but challenging due to the inherent lack of explicit spatial information. Multi-modal approaches rely on expensive depth sensors, while methods incorporating fully-supervised depth networks rely on ground-truth depth data that is impractical to collect at scale. Additionally, existing methods assume that camera parameters are available, limiting their applicability in scenarios like crowdsourced high-definition (HD) lane mapping. To address these limitations, we propose Depth3DLane, a novel dual-pathway framework that integrates self-supervised monocular depth estimation to provide explicit structural information, without the need for expensive sensors or additional ground-truth depth data. Leveraging a self-supervised depth network to obtain a point cloud representation of the scene, our bird's-eye view pathway extracts explicit spatial information, while our front view pathway simultaneously extracts rich semantic information. Depth3DLane then uses 3D lane anchors to sample features from both pathways and infer accurate 3D lane geometry. Furthermore, we extend the framework to predict camera parameters on a per-frame basis and introduce a theoretically motivated fitting procedure to enhance stability on a per-segment basis. Extensive experiments demonstrate that Depth3DLane achieves competitive performance on the OpenLane benchmark dataset. Furthermore, experimental results show that using learned parameters instead of ground-truth parameters allows Depth3DLane to be applied in scenarios where camera calibration is infeasible, unlike previous methods.
>
---
#### [new 026] Safe and Performant Controller Synthesis using Gradient-based Model Predictive Control and Control Barrier Functions
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 论文提出了一种结合梯度模型预测控制（MPC）与控制屏障函数（CBF）的安全控制器合成方法。旨在解决自主系统在复杂环境中同时保证性能与安全性的难题。通过两阶段框架，先以梯度法优化带安全惩罚的控制器，再用CBF-QP确保硬安全约束，实现高效且安全的控制。**

- **链接: [http://arxiv.org/pdf/2507.13872v1](http://arxiv.org/pdf/2507.13872v1)**

> **作者:** Aditya Singh; Aastha Mishra; Manan Tayal; Shishir Kolathaya; Pushpak Jagtap
>
> **备注:** 6 Pages, 2 Figures. The first two authors contributed equally
>
> **摘要:** Ensuring both performance and safety is critical for autonomous systems operating in real-world environments. While safety filters such as Control Barrier Functions (CBFs) enforce constraints by modifying nominal controllers in real time, they can become overly conservative when the nominal policy lacks safety awareness. Conversely, solving State-Constrained Optimal Control Problems (SC-OCPs) via dynamic programming offers formal guarantees but is intractable in high-dimensional systems. In this work, we propose a novel two-stage framework that combines gradient-based Model Predictive Control (MPC) with CBF-based safety filtering for co-optimizing safety and performance. In the first stage, we relax safety constraints as penalties in the cost function, enabling fast optimization via gradient-based methods. This step improves scalability and avoids feasibility issues associated with hard constraints. In the second stage, we modify the resulting controller using a CBF-based Quadratic Program (CBF-QP), which enforces hard safety constraints with minimal deviation from the reference. Our approach yields controllers that are both performant and provably safe. We validate the proposed framework on two case studies, showcasing its ability to synthesize scalable, safe, and high-performance controllers for complex, high-dimensional autonomous systems.
>
---
#### [new 027] Generalist Bimanual Manipulation via Foundation Video Diffusion Models
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于机器人双臂操作任务，旨在解决数据稀缺和异构性问题。论文提出了VIDAR框架，结合视频扩散模型与掩码逆动力学模型，利用多视角视频预训练，实现跨任务和背景的动作预测。使用少量人类示范，即可在新任务中取得良好泛化效果。**

- **链接: [http://arxiv.org/pdf/2507.12898v1](http://arxiv.org/pdf/2507.12898v1)**

> **作者:** Yao Feng; Hengkai Tan; Xinyi Mao; Guodong Liu; Shuhe Huang; Chendong Xiang; Hang Su; Jun Zhu
>
> **摘要:** Bimanual robotic manipulation, which involves the coordinated control of two robotic arms, is foundational for solving challenging tasks. Despite recent progress in general-purpose manipulation, data scarcity and embodiment heterogeneity remain serious obstacles to further scaling up in bimanual settings. In this paper, we introduce VIdeo Diffusion for Action Reasoning (VIDAR), a two-stage framework that leverages large-scale, diffusion-based video pre-training and a novel masked inverse dynamics model for action prediction. We pre-train the video diffusion model on 750K multi-view videos from three real-world bimanual robot platforms, utilizing a unified observation space that encodes robot, camera, task, and scene contexts. Our masked inverse dynamics model learns masks to extract action-relevant information from generated trajectories without requiring pixel-level labels, and the masks can effectively generalize to unseen backgrounds. Our experiments demonstrate that with only 20 minutes of human demonstrations on an unseen robot platform (only 1% of typical data requirements), VIDAR generalizes to unseen tasks and backgrounds with strong semantic understanding, surpassing state-of-the-art methods. Our findings highlight the potential of video foundation models, coupled with masked action prediction, to enable scalable and generalizable robotic manipulation in diverse real-world settings.
>
---
## 更新

#### [replaced 001] Multi-Objective Reinforcement Learning for Adaptable Personalized Autonomous Driving
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.05223v2](http://arxiv.org/pdf/2505.05223v2)**

> **作者:** Hendrik Surmann; Jorge de Heuvel; Maren Bennewitz
>
> **摘要:** Human drivers exhibit individual preferences regarding driving style. Adapting autonomous vehicles to these preferences is essential for user trust and satisfaction. However, existing end-to-end driving approaches often rely on predefined driving styles or require continuous user feedback for adaptation, limiting their ability to support dynamic, context-dependent preferences. We propose a novel approach using multi-objective reinforcement learning (MORL) with preference-driven optimization for end-to-end autonomous driving that enables runtime adaptation to driving style preferences. Preferences are encoded as continuous weight vectors to modulate behavior along interpretable style objectives$\unicode{x2013}$including efficiency, comfort, speed, and aggressiveness$\unicode{x2013}$without requiring policy retraining. Our single-policy agent integrates vision-based perception in complex mixed-traffic scenarios and is evaluated in diverse urban environments using the CARLA simulator. Experimental results demonstrate that the agent dynamically adapts its driving behavior according to changing preferences while maintaining performance in terms of collision avoidance and route completion.
>
---
#### [replaced 002] Critiques of World Models
- **分类: cs.LG; cs.AI; cs.CL; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.05169v2](http://arxiv.org/pdf/2507.05169v2)**

> **作者:** Eric Xing; Mingkai Deng; Jinyu Hou; Zhiting Hu
>
> **摘要:** World Model, the supposed algorithmic surrogate of the real-world environment which biological agents experience with and act upon, has been an emerging topic in recent years because of the rising needs to develop virtual agents with artificial (general) intelligence. There has been much debate on what a world model really is, how to build it, how to use it, and how to evaluate it. In this essay, starting from the imagination in the famed Sci-Fi classic Dune, and drawing inspiration from the concept of "hypothetical thinking" in psychology literature, we offer critiques of several schools of thoughts on world modeling, and argue the primary goal of a world model to be simulating all actionable possibilities of the real world for purposeful reasoning and acting. Building on the critiques, we propose a new architecture for a general-purpose world model, based on hierarchical, multi-level, and mixed continuous/discrete representations, and a generative and self-supervision learning framework, with an outlook of a Physical, Agentic, and Nested (PAN) AGI system enabled by such a model.
>
---
#### [replaced 003] Lost in Tracking Translation: A Comprehensive Analysis of Visual SLAM in Human-Centered XR and IoT Ecosystems
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.07146v2](http://arxiv.org/pdf/2411.07146v2)**

> **作者:** Yasra Chandio; Khotso Selialia; Joseph DeGol; Luis Garcia; Fatima M. Anwar
>
> **摘要:** Advancements in tracking algorithms have empowered nascent applications across various domains, from steering autonomous vehicles to guiding robots to enhancing augmented reality experiences for users. However, these algorithms are application-specific and do not work across applications with different types of motion; even a tracking algorithm designed for a given application does not work in scenarios deviating from highly standard conditions. For example, a tracking algorithm designed for robot navigation inside a building will not work for tracking the same robot in an outdoor environment. To demonstrate this problem, we evaluate the performance of the state-of-the-art tracking methods across various applications and scenarios. To inform our analysis, we first categorize algorithmic, environmental, and locomotion-related challenges faced by tracking algorithms. We quantitatively evaluate the performance using multiple tracking algorithms and representative datasets for a wide range of Internet of Things (IoT) and Extended Reality (XR) applications, including autonomous vehicles, drones, and humans. Our analysis shows that no tracking algorithm works across different applications and scenarios within applications. Ultimately, using the insights generated from our analysis, we discuss multiple approaches to improving the tracking performance using input data characterization, leveraging intermediate information, and output evaluation.
>
---
#### [replaced 004] Stonefish: Supporting Machine Learning Research in Marine Robotics
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2502.11887v2](http://arxiv.org/pdf/2502.11887v2)**

> **作者:** Michele Grimaldi; Patryk Cieslak; Eduardo Ochoa; Vibhav Bharti; Hayat Rajani; Ignacio Carlucho; Maria Koskinopoulou; Yvan R. Petillot; Nuno Gracias
>
> **备注:** 2025 IEEE/RSJ International Conference on Robotics and Automation (ICRA)
>
> **摘要:** Simulations are highly valuable in marine robotics, offering a cost-effective and controlled environment for testing in the challenging conditions of underwater and surface operations. Given the high costs and logistical difficulties of real-world trials, simulators capable of capturing the operational conditions of subsea environments have become key in developing and refining algorithms for remotely-operated and autonomous underwater vehicles. This paper highlights recent enhancements to the Stonefish simulator, an advanced open-source platform supporting development and testing of marine robotics solutions. Key updates include a suite of additional sensors, such as an event-based camera, a thermal camera, and an optical flow camera, as well as, visual light communication, support for tethered operations, improved thruster modelling, more flexible hydrodynamics, and enhanced sonar accuracy. These developments and an automated annotation tool significantly bolster Stonefish's role in marine robotics research, especially in the field of machine learning, where training data with a known ground truth is hard or impossible to collect.
>
---
#### [replaced 005] GeoPF: Infusing Geometry into Potential Fields for Reactive Planning in Non-trivial Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.19688v2](http://arxiv.org/pdf/2505.19688v2)**

> **作者:** Yuhe Gong; Riddhiman Laha; Luis Figueredo
>
> **摘要:** Reactive intelligence remains one of the cornerstones of versatile robotics operating in cluttered, dynamic, and human-centred environments. Among reactive approaches, potential fields (PF) continue to be widely adopted due to their simplicity and real-time applicability. However, existing PF methods typically oversimplify environmental representations by relying on isotropic, point- or sphere-based obstacle approximations. In human-centred settings, this simplification results in overly conservative paths, cumbersome tuning, and computational overhead -- even breaking real-time requirements. In response, we propose the Geometric Potential Field (GeoPF), a reactive motion-planning framework that explicitly infuses geometric primitives -- points, lines, planes, cubes, and cylinders -- their structure and spatial relationship in modulating the real-time repulsive response. Extensive quantitative analyses consistently show GeoPF's higher success rates, reduced tuning complexity (a single parameter set across experiments), and substantially lower computational costs (up to 2 orders of magnitude) compared to traditional PF methods. Real-world experiments further validate GeoPF reliability, robustness, and practical ease of deployment, as well as its scalability to whole-body avoidance. GeoPF provides a fresh perspective on reactive planning problems driving geometric-aware temporal motion generation, enabling flexible and low-latency motion planning suitable for modern robotic applications.
>
---
#### [replaced 006] A Survey of Behavior Foundation Model: Next-Generation Whole-Body Control System of Humanoid Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.20487v3](http://arxiv.org/pdf/2506.20487v3)**

> **作者:** Mingqi Yuan; Tao Yu; Wenqi Ge; Xiuyong Yao; Huijiang Wang; Jiayu Chen; Xin Jin; Bo Li; Hua Chen; Wei Zhang; Wenjun Zeng
>
> **备注:** 18 pages, 9 figures
>
> **摘要:** Humanoid robots are drawing significant attention as versatile platforms for complex motor control, human-robot interaction, and general-purpose physical intelligence. However, achieving efficient whole-body control (WBC) in humanoids remains a fundamental challenge due to sophisticated dynamics, underactuation, and diverse task requirements. While learning-based controllers have shown promise for complex tasks, their reliance on labor-intensive and costly retraining for new scenarios limits real-world applicability. To address these limitations, behavior(al) foundation models (BFMs) have emerged as a new paradigm that leverages large-scale pre-training to learn reusable primitive skills and broad behavioral priors, enabling zero-shot or rapid adaptation to a wide range of downstream tasks. In this paper, we present a comprehensive overview of BFMs for humanoid WBC, tracing their development across diverse pre-training pipelines. Furthermore, we discuss real-world applications, current limitations, urgent challenges, and future opportunities, positioning BFMs as a key approach toward scalable and general-purpose humanoid intelligence. Finally, we provide a curated and long-term list of BFM papers and projects to facilitate more subsequent research, which is available at https://github.com/yuanmingqi/awesome-bfm-papers.
>
---
#### [replaced 007] Horticultural Temporal Fruit Monitoring via 3D Instance Segmentation and Re-Identification using Colored Point Clouds
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2411.07799v2](http://arxiv.org/pdf/2411.07799v2)**

> **作者:** Daniel Fusaro; Federico Magistri; Jens Behley; Alberto Pretto; Cyrill Stachniss
>
> **备注:** Submitted to Computers and Electronics in Agriculture
>
> **摘要:** Accurate and consistent fruit monitoring over time is a key step toward automated agricultural production systems. However, this task is inherently difficult due to variations in fruit size, shape, occlusion, orientation, and the dynamic nature of orchards where fruits may appear or disappear between observations. In this article, we propose a novel method for fruit instance segmentation and re-identification on 3D terrestrial point clouds collected over time. Our approach directly operates on dense colored point clouds, capturing fine-grained 3D spatial detail. We segment individual fruits using a learning-based instance segmentation method applied directly to the point cloud. For each segmented fruit, we extract a compact and discriminative descriptor using a 3D sparse convolutional neural network. To track fruits across different times, we introduce an attention-based matching network that associates fruits with their counterparts from previous sessions. Matching is performed using a probabilistic assignment scheme, selecting the most likely associations across time. We evaluate our approach on real-world datasets of strawberries and apples, demonstrating that it outperforms existing methods in both instance segmentation and temporal re-identification, enabling robust and precise fruit monitoring across complex and dynamic orchard environments.
>
---
#### [replaced 008] Chance-constrained Linear Quadratic Gaussian Games for Multi-robot Interaction under Uncertainty
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.06776v2](http://arxiv.org/pdf/2503.06776v2)**

> **作者:** Kai Ren; Giulio Salizzoni; Mustafa Emre Gürsoy; Maryam Kamgarpour
>
> **备注:** Published in IEEE Control Systems Letters
>
> **摘要:** We address safe multi-robot interaction under uncertainty. In particular, we formulate a chance-constrained linear quadratic Gaussian game with coupling constraints and system uncertainties. We find a tractable reformulation of the game and propose a dual ascent algorithm. We prove that the algorithm converges to a feedback generalized Nash equilibrium of the reformulated game, ensuring the satisfaction of the chance constraints. We test our method in driving simulations and real-world robot experiments. Our method ensures safety under uncertainty and generates less conservative trajectories than single-agent model predictive control.
>
---
#### [replaced 009] From Words to Collisions: LLM-Guided Evaluation and Adversarial Generation of Safety-Critical Driving Scenarios
- **分类: cs.AI; cs.CL; cs.RO**

- **链接: [http://arxiv.org/pdf/2502.02145v4](http://arxiv.org/pdf/2502.02145v4)**

> **作者:** Yuan Gao; Mattia Piccinini; Korbinian Moller; Amr Alanwar; Johannes Betz
>
> **备注:** Final Version and Paper Accepted at IEEE ITSC 2025
>
> **摘要:** Ensuring the safety of autonomous vehicles requires virtual scenario-based testing, which depends on the robust evaluation and generation of safety-critical scenarios. So far, researchers have used scenario-based testing frameworks that rely heavily on handcrafted scenarios as safety metrics. To reduce the effort of human interpretation and overcome the limited scalability of these approaches, we combine Large Language Models (LLMs) with structured scenario parsing and prompt engineering to automatically evaluate and generate safety-critical driving scenarios. We introduce Cartesian and Ego-centric prompt strategies for scenario evaluation, and an adversarial generation module that modifies trajectories of risk-inducing vehicles (ego-attackers) to create critical scenarios. We validate our approach using a 2D simulation framework and multiple pre-trained LLMs. The results show that the evaluation module effectively detects collision scenarios and infers scenario safety. Meanwhile, the new generation module identifies high-risk agents and synthesizes realistic, safety-critical scenarios. We conclude that an LLM equipped with domain-informed prompting techniques can effectively evaluate and generate safety-critical driving scenarios, reducing dependence on handcrafted metrics. We release our open-source code and scenarios at: https://github.com/TUM-AVS/From-Words-to-Collisions.
>
---
#### [replaced 010] Robotic Monitoring of Colorimetric Leaf Sensors for Precision Agriculture
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.13916v2](http://arxiv.org/pdf/2505.13916v2)**

> **作者:** Malakhi Hopkins; Alice Kate Li; Shobhita Kramadhati; Jackson Arnold; Akhila Mallavarapu; Chavez Lawrence; Varun Murali; Sanjeev J. Koppal; Cherie R. Kagan; Vijay Kumar
>
> **备注:** Revised version. Initial version was accepted to the Novel Approaches for Precision Agriculture and Forestry with Autonomous Robots IEEE ICRA Workshop - 2025
>
> **摘要:** Common remote sensing modalities (RGB, multispectral, hyperspectral imaging or LiDAR) are often used to indirectly measure crop health and do not directly capture plant stress indicators. Commercially available direct leaf sensors are bulky, powered electronics that are expensive and interfere with crop growth. In contrast, low-cost, passive and bio-degradable leaf sensors offer an opportunity to advance real-time monitoring as they directly interface with the crop surface while not interfering with crop growth. To this end, we co-design a sensor-detector system, where the sensor is a passive colorimetric leaf sensor that directly measures crop health in a precision agriculture setting, and the detector autonomously obtains optical signals from these leaf sensors. The detector comprises a low size weight and power (SWaP) mobile ground robot with an onboard monocular RGB camera and object detector to localize each leaf sensor, as well as a hyperspectral camera with a motorized mirror and halogen light to acquire hyperspectral images. The sensor's crop health-dependent optical signals can be extracted from the hyperspectral images. The proof-of-concept system is demonstrated in row-crop environments both indoors and outdoors where it is able to autonomously navigate, locate and obtain a hyperspectral image of all leaf sensors present, and acquire interpretable spectral resonance with 80 $\%$ accuracy within a required retrieval distance from the sensor.
>
---
#### [replaced 011] Robustness Evaluation of Offline Reinforcement Learning for Robot Control Against Action Perturbations
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2412.18781v2](http://arxiv.org/pdf/2412.18781v2)**

> **作者:** Shingo Ayabe; Takuto Otomo; Hiroshi Kera; Kazuhiko Kawamoto
>
> **备注:** 22 pages, 6 figures
>
> **摘要:** Offline reinforcement learning, which learns solely from datasets without environmental interaction, has gained attention. This approach, similar to traditional online deep reinforcement learning, is particularly promising for robot control applications. Nevertheless, its robustness against real-world challenges, such as joint actuator faults in robots, remains a critical concern. This study evaluates the robustness of existing offline reinforcement learning methods using legged robots from OpenAI Gym based on average episodic rewards. For robustness evaluation, we simulate failures by incorporating both random and adversarial perturbations, representing worst-case scenarios, into the joint torque signals. Our experiments show that existing offline reinforcement learning methods exhibit significant vulnerabilities to these action perturbations and are more vulnerable than online reinforcement learning methods, highlighting the need for more robust approaches in this field.
>
---
#### [replaced 012] VMTS: Vision-Assisted Teacher-Student Reinforcement Learning for Multi-Terrain Locomotion in Bipedal Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.07049v2](http://arxiv.org/pdf/2503.07049v2)**

> **作者:** Fu Chen; Rui Wan; Peidong Liu; Nanxing Zheng; Bo Zhou
>
> **摘要:** Bipedal robots, due to their anthropomorphic design, offer substantial potential across various applications, yet their control is hindered by the complexity of their structure. Currently, most research focuses on proprioception-based methods, which lack the capability to overcome complex terrain. While visual perception is vital for operation in human-centric environments, its integration complicates control further. Recent reinforcement learning (RL) approaches have shown promise in enhancing legged robot locomotion, particularly with proprioception-based methods. However, terrain adaptability, especially for bipedal robots, remains a significant challenge, with most research focusing on flat-terrain scenarios. In this paper, we introduce a novel mixture of experts teacher-student network RL strategy, which enhances the performance of teacher-student policies based on visual inputs through a simple yet effective approach. Our method combines terrain selection strategies with the teacher policy, resulting in superior performance compared to traditional models. Additionally, we introduce an alignment loss between the teacher and student networks, rather than enforcing strict similarity, to improve the student's ability to navigate diverse terrains. We validate our approach experimentally on the Limx Dynamic P1 bipedal robot, demonstrating its feasibility and robustness across multiple terrain types.
>
---
#### [replaced 013] EgoVLA: Learning Vision-Language-Action Models from Egocentric Human Videos
- **分类: cs.RO; cs.AI; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.12440v3](http://arxiv.org/pdf/2507.12440v3)**

> **作者:** Ruihan Yang; Qinxi Yu; Yecheng Wu; Rui Yan; Borui Li; An-Chieh Cheng; Xueyan Zou; Yunhao Fang; Xuxin Cheng; Ri-Zhao Qiu; Hongxu Yin; Sifei Liu; Song Han; Yao Lu; Xiaolong Wang
>
> **备注:** More videos can be found on our website: https://rchalyang.github.io/EgoVLA
>
> **摘要:** Real robot data collection for imitation learning has led to significant advancements in robotic manipulation. However, the requirement for robot hardware in the process fundamentally constrains the scale of the data. In this paper, we explore training Vision-Language-Action (VLA) models using egocentric human videos. The benefit of using human videos is not only for their scale but more importantly for the richness of scenes and tasks. With a VLA trained on human video that predicts human wrist and hand actions, we can perform Inverse Kinematics and retargeting to convert the human actions to robot actions. We fine-tune the model using a few robot manipulation demonstrations to obtain the robot policy, namely EgoVLA. We propose a simulation benchmark called Ego Humanoid Manipulation Benchmark, where we design diverse bimanual manipulation tasks with demonstrations. We fine-tune and evaluate EgoVLA with Ego Humanoid Manipulation Benchmark and show significant improvements over baselines and ablate the importance of human data. Videos can be found on our website: https://rchalyang.github.io/EgoVLA
>
---
#### [replaced 014] DiffAD: A Unified Diffusion Modeling Approach for Autonomous Driving
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.12170v2](http://arxiv.org/pdf/2503.12170v2)**

> **作者:** Tao Wang; Cong Zhang; Xingguang Qu; Kun Li; Weiwei Liu; Chang Huang
>
> **备注:** 8 pages, 6 figures; Code released
>
> **摘要:** End-to-end autonomous driving (E2E-AD) has rapidly emerged as a promising approach toward achieving full autonomy. However, existing E2E-AD systems typically adopt a traditional multi-task framework, addressing perception, prediction, and planning tasks through separate task-specific heads. Despite being trained in a fully differentiable manner, they still encounter issues with task coordination, and the system complexity remains high. In this work, we introduce DiffAD, a novel diffusion probabilistic model that redefines autonomous driving as a conditional image generation task. By rasterizing heterogeneous targets onto a unified bird's-eye view (BEV) and modeling their latent distribution, DiffAD unifies various driving objectives and jointly optimizes all driving tasks in a single framework, significantly reducing system complexity and harmonizing task coordination. The reverse process iteratively refines the generated BEV image, resulting in more robust and realistic driving behaviors. Closed-loop evaluations in Carla demonstrate the superiority of the proposed method, achieving a new state-of-the-art Success Rate and Driving Score.
>
---
#### [replaced 015] Non-Overlap-Aware Egocentric Pose Estimation for Collaborative Perception in Connected Autonomy
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.14180v2](http://arxiv.org/pdf/2506.14180v2)**

> **作者:** Hong Huang; Dongkuan Xu; Hao Zhang; Peng Gao
>
> **备注:** IROS 2025
>
> **摘要:** Egocentric pose estimation is a fundamental capability for multi-robot collaborative perception in connected autonomy, such as connected autonomous vehicles. During multi-robot operations, a robot needs to know the relative pose between itself and its teammates with respect to its own coordinates. However, different robots usually observe completely different views that contains similar objects, which leads to wrong pose estimation. In addition, it is unrealistic to allow robots to share their raw observations to detect overlap due to the limited communication bandwidth constraint. In this paper, we introduce a novel method for Non-Overlap-Aware Egocentric Pose Estimation (NOPE), which performs egocentric pose estimation in a multi-robot team while identifying the non-overlap views and satifying the communication bandwidth constraint. NOPE is built upon an unified hierarchical learning framework that integrates two levels of robot learning: (1) high-level deep graph matching for correspondence identification, which allows to identify if two views are overlapping or not, (2) low-level position-aware cross-attention graph learning for egocentric pose estimation. To evaluate NOPE, we conduct extensive experiments in both high-fidelity simulation and real-world scenarios. Experimental results have demonstrated that NOPE enables the novel capability for non-overlapping-aware egocentric pose estimation and achieves state-of-art performance compared with the existing methods. Our project page at https://hongh0.github.io/NOPE/.
>
---
