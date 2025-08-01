# 机器人 cs.RO

- **最新发布 40 篇**

- **更新 14 篇**

## 最新发布

#### [new 001] MCOO-SLAM: A Multi-Camera Omnidirectional Object SLAM System
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于SLAM任务，旨在解决大场景下对象建模不准确的问题。通过多视角相机和语义融合策略，提升对象关联与地图一致性。**

- **链接: [http://arxiv.org/pdf/2506.15402v1](http://arxiv.org/pdf/2506.15402v1)**

> **作者:** Miaoxin Pan; Jinnan Li; Yaowen Zhang; Yi Yang; Yufeng Yue
>
> **摘要:** Object-level SLAM offers structured and semantically meaningful environment representations, making it more interpretable and suitable for high-level robotic tasks. However, most existing approaches rely on RGB-D sensors or monocular views, which suffer from narrow fields of view, occlusion sensitivity, and limited depth perception-especially in large-scale or outdoor environments. These limitations often restrict the system to observing only partial views of objects from limited perspectives, leading to inaccurate object modeling and unreliable data association. In this work, we propose MCOO-SLAM, a novel Multi-Camera Omnidirectional Object SLAM system that fully leverages surround-view camera configurations to achieve robust, consistent, and semantically enriched mapping in complex outdoor scenarios. Our approach integrates point features and object-level landmarks enhanced with open-vocabulary semantics. A semantic-geometric-temporal fusion strategy is introduced for robust object association across multiple views, leading to improved consistency and accurate object modeling, and an omnidirectional loop closure module is designed to enable viewpoint-invariant place recognition using scene-level descriptors. Furthermore, the constructed map is abstracted into a hierarchical 3D scene graph to support downstream reasoning tasks. Extensive experiments in real-world demonstrate that MCOO-SLAM achieves accurate localization and scalable object-level mapping with improved robustness to occlusion, pose variation, and environmental complexity.
>
---
#### [new 002] Comparison of Innovative Strategies for the Coverage Problem: Path Planning, Search Optimization, and Applications in Underwater Robotics
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于路径规划任务，解决水下机器人覆盖问题，比较了TSP、MST和OCP三种策略，评估其效率与计算成本。**

- **链接: [http://arxiv.org/pdf/2506.15376v1](http://arxiv.org/pdf/2506.15376v1)**

> **作者:** Ahmed Ibrahim; Francisco F. C. Rego; Éric Busvelle
>
> **摘要:** In many applications, including underwater robotics, the coverage problem requires an autonomous vehicle to systematically explore a defined area while minimizing redundancy and avoiding obstacles. This paper investigates coverage path planning strategies to enhance the efficiency of underwater gliders, particularly in maximizing the probability of detecting a radioactive source while ensuring safe navigation. We evaluate three path-planning approaches: the Traveling Salesman Problem (TSP), Minimum Spanning Tree (MST), and Optimal Control Problem (OCP). Simulations were conducted in MATLAB, comparing processing time, uncovered areas, path length, and traversal time. Results indicate that OCP is preferable when traversal time is constrained, although it incurs significantly higher computational costs. Conversely, MST-based approaches provide faster but less optimal solutions. These findings offer insights into selecting appropriate algorithms based on mission priorities, balancing efficiency and computational feasibility.
>
---
#### [new 003] EmojiVoice: Towards long-term controllable expressivity in robot speech
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于语音合成任务，旨在解决机器人语音缺乏长期情感变化的问题。提出EmojiVoice工具，通过表情符号控制语音表达，提升机器人语音的自然性和表现力。**

- **链接: [http://arxiv.org/pdf/2506.15085v1](http://arxiv.org/pdf/2506.15085v1)**

> **作者:** Paige Tuttösí; Shivam Mehta; Zachary Syvenky; Bermet Burkanova; Gustav Eje Henter; Angelica Lim
>
> **备注:** Accepted to RO-MAN 2025, Demo at HRI 2025 : https://dl.acm.org/doi/10.5555/3721488.3721774
>
> **摘要:** Humans vary their expressivity when speaking for extended periods to maintain engagement with their listener. Although social robots tend to be deployed with ``expressive'' joyful voices, they lack this long-term variation found in human speech. Foundation model text-to-speech systems are beginning to mimic the expressivity in human speech, but they are difficult to deploy offline on robots. We present EmojiVoice, a free, customizable text-to-speech (TTS) toolkit that allows social roboticists to build temporally variable, expressive speech on social robots. We introduce emoji-prompting to allow fine-grained control of expressivity on a phase level and use the lightweight Matcha-TTS backbone to generate speech in real-time. We explore three case studies: (1) a scripted conversation with a robot assistant, (2) a storytelling robot, and (3) an autonomous speech-to-speech interactive agent. We found that using varied emoji prompting improved the perception and expressivity of speech over a long period in a storytelling task, but expressive voice was not preferred in the assistant use case.
>
---
#### [new 004] 3D Vision-tactile Reconstruction from Infrared and Visible Images for Robotic Fine-grained Tactile Perception
- **分类: cs.RO**

- **简介: 该论文属于机器人触觉感知任务，旨在解决平面触觉传感器在曲面应用中的重建难题。通过改进成像和算法，提升触觉精度与抓取性能。**

- **链接: [http://arxiv.org/pdf/2506.15087v1](http://arxiv.org/pdf/2506.15087v1)**

> **作者:** Yuankai Lin; Xiaofan Lu; Jiahui Chen; Hua Yang
>
> **摘要:** To achieve human-like haptic perception in anthropomorphic grippers, the compliant sensing surfaces of vision tactile sensor (VTS) must evolve from conventional planar configurations to biomimetically curved topographies with continuous surface gradients. However, planar VTSs have challenges when extended to curved surfaces, including insufficient lighting of surfaces, blurring in reconstruction, and complex spatial boundary conditions for surface structures. With an end goal of constructing a human-like fingertip, our research (i) develops GelSplitter3D by expanding imaging channels with a prism and a near-infrared (NIR) camera, (ii) proposes a photometric stereo neural network with a CAD-based normal ground truth generation method to calibrate tactile geometry, and (iii) devises a normal integration method with boundary constraints of depth prior information to correcting the cumulative error of surface integrals. We demonstrate better tactile sensing performance, a 40$\%$ improvement in normal estimation accuracy, and the benefits of sensor shapes in grasping and manipulation tasks.
>
---
#### [new 005] FEAST: A Flexible Mealtime-Assistance System Towards In-the-Wild Personalization
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人机交互任务，旨在解决家庭环境中个性化进食辅助问题。提出FEAST系统，通过模块化设计和参数化行为树实现灵活、安全的个性化适配。**

- **链接: [http://arxiv.org/pdf/2506.14968v1](http://arxiv.org/pdf/2506.14968v1)**

> **作者:** Rajat Kumar Jenamani; Tom Silver; Ben Dodson; Shiqin Tong; Anthony Song; Yuting Yang; Ziang Liu; Benjamin Howe; Aimee Whitneck; Tapomayukh Bhattacharjee
>
> **备注:** RSS 2025 - Outstanding Paper Award & Outstanding Systems Paper Award Finalist
>
> **摘要:** Physical caregiving robots hold promise for improving the quality of life of millions worldwide who require assistance with feeding. However, in-home meal assistance remains challenging due to the diversity of activities (e.g., eating, drinking, mouth wiping), contexts (e.g., socializing, watching TV), food items, and user preferences that arise during deployment. In this work, we propose FEAST, a flexible mealtime-assistance system that can be personalized in-the-wild to meet the unique needs of individual care recipients. Developed in collaboration with two community researchers and informed by a formative study with a diverse group of care recipients, our system is guided by three key tenets for in-the-wild personalization: adaptability, transparency, and safety. FEAST embodies these principles through: (i) modular hardware that enables switching between assisted feeding, drinking, and mouth-wiping, (ii) diverse interaction methods, including a web interface, head gestures, and physical buttons, to accommodate diverse functional abilities and preferences, and (iii) parameterized behavior trees that can be safely and transparently adapted using a large language model. We evaluate our system based on the personalization requirements identified in our formative study, demonstrating that FEAST offers a wide range of transparent and safe adaptations and outperforms a state-of-the-art baseline limited to fixed customizations. To demonstrate real-world applicability, we conduct an in-home user study with two care recipients (who are community researchers), feeding them three meals each across three diverse scenarios. We further assess FEAST's ecological validity by evaluating with an Occupational Therapist previously unfamiliar with the system. In all cases, users successfully personalize FEAST to meet their individual needs and preferences. Website: https://emprise.cs.cornell.edu/feast
>
---
#### [new 006] Real-Time Initialization of Unknown Anchors for UWB-aided Navigation
- **分类: cs.RO**

- **简介: 该论文属于UWB定位任务，解决未知锚点实时初始化问题。通过融合PDOP估计、异常检测和自适应鲁棒核，提升系统鲁棒性与定位精度。**

- **链接: [http://arxiv.org/pdf/2506.15518v1](http://arxiv.org/pdf/2506.15518v1)**

> **作者:** Giulio Delama; Igor Borowski; Roland Jung; Stephan Weiss
>
> **摘要:** This paper presents a framework for the real-time initialization of unknown Ultra-Wideband (UWB) anchors in UWB-aided navigation systems. The method is designed for localization solutions where UWB modules act as supplementary sensors. Our approach enables the automatic detection and calibration of previously unknown anchors during operation, removing the need for manual setup. By combining an online Positional Dilution of Precision (PDOP) estimation, a lightweight outlier detection method, and an adaptive robust kernel for non-linear optimization, our approach significantly improves robustness and suitability for real-world applications compared to state-of-the-art. In particular, we show that our metric which triggers an initialization decision is more conservative than current ones commonly based on initial linear or non-linear initialization guesses. This allows for better initialization geometry and subsequently lower initialization errors. We demonstrate the proposed approach on two different mobile robots: an autonomous forklift and a quadcopter equipped with a UWB-aided Visual-Inertial Odometry (VIO) framework. The results highlight the effectiveness of the proposed method with robust initialization and low positioning error. We open-source our code in a C++ library including a ROS wrapper.
>
---
#### [new 007] GRIM: Task-Oriented Grasping with Conditioning on Generative Examples
- **分类: cs.RO**

- **简介: 该论文属于任务导向抓取领域，旨在解决如何根据任务需求准确抓取物体的问题。提出GRIM框架，通过实例匹配与姿态优化实现高效抓取。**

- **链接: [http://arxiv.org/pdf/2506.15607v1](http://arxiv.org/pdf/2506.15607v1)**

> **作者:** Shailesh; Alok Raj; Nayan Kumar; Priya Shukla; Andrew Melnik; Micheal Beetz; Gora Chand Nandi
>
> **摘要:** Task-Oriented Grasping (TOG) presents a significant challenge, requiring a nuanced understanding of task semantics, object affordances, and the functional constraints dictating how an object should be grasped for a specific task. To address these challenges, we introduce GRIM (Grasp Re-alignment via Iterative Matching), a novel training-free framework for task-oriented grasping. Initially, a coarse alignment strategy is developed using a combination of geometric cues and principal component analysis (PCA)-reduced DINO features for similarity scoring. Subsequently, the full grasp pose associated with the retrieved memory instance is transferred to the aligned scene object and further refined against a set of task-agnostic, geometrically stable grasps generated for the scene object, prioritizing task compatibility. In contrast to existing learning-based methods, GRIM demonstrates strong generalization capabilities, achieving robust performance with only a small number of conditioning examples.
>
---
#### [new 008] Assigning Multi-Robot Tasks to Multitasking Robots
- **分类: cs.RO**

- **简介: 该论文属于多机器人任务分配问题，解决多任务机器人在物理约束下的高效任务分配问题。提出新框架并引入求解方法以提升效率。**

- **链接: [http://arxiv.org/pdf/2506.15032v1](http://arxiv.org/pdf/2506.15032v1)**

> **作者:** Winston Smith; Andrew Boateng; Taha Shaheen; Yu Zhang
>
> **摘要:** One simplifying assumption in existing and well-performing task allocation methods is that the robots are single-tasking: each robot operates on a single task at any given time. While this assumption is harmless to make in some situations, it can be inefficient or even infeasible in others. In this paper, we consider assigning multi-robot tasks to multitasking robots. The key contribution is a novel task allocation framework that incorporates the consideration of physical constraints introduced by multitasking. This is in contrast to the existing work where such constraints are largely ignored. After formulating the problem, we propose a compilation to weighted MAX-SAT, which allows us to leverage existing solvers for a solution. A more efficient greedy heuristic is then introduced. For evaluation, we first compare our methods with a modern baseline that is efficient for single-tasking robots to validate the benefits of multitasking in synthetic domains. Then, using a site-clearing scenario in simulation, we further illustrate the complex task interaction considered by the multitasking robots in our approach to demonstrate its performance. Finally, we demonstrate a physical experiment to show how multitasking enabled by our approach can benefit task efficiency in a realistic setting.
>
---
#### [new 009] DyNaVLM: Zero-Shot Vision-Language Navigation System with Dynamic Viewpoints and Self-Refining Graph Memory
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，旨在解决机器人在复杂环境中自主导航问题。提出DyNaVLM系统，通过动态视角和自修正图记忆实现高效导航。**

- **链接: [http://arxiv.org/pdf/2506.15096v1](http://arxiv.org/pdf/2506.15096v1)**

> **作者:** Zihe Ji; Huangxuan Lin; Yue Gao
>
> **摘要:** We present DyNaVLM, an end-to-end vision-language navigation framework using Vision-Language Models (VLM). In contrast to prior methods constrained by fixed angular or distance intervals, our system empowers agents to freely select navigation targets via visual-language reasoning. At its core lies a self-refining graph memory that 1) stores object locations as executable topological relations, 2) enables cross-robot memory sharing through distributed graph updates, and 3) enhances VLM's decision-making via retrieval augmentation. Operating without task-specific training or fine-tuning, DyNaVLM demonstrates high performance on GOAT and ObjectNav benchmarks. Real-world tests further validate its robustness and generalization. The system's three innovations: dynamic action space formulation, collaborative graph memory, and training-free deployment, establish a new paradigm for scalable embodied robot, bridging the gap between discrete VLN tasks and continuous real-world navigation.
>
---
#### [new 010] Context-Aware Deep Lagrangian Networks for Model Predictive Control
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，解决复杂环境中模型泛化问题。通过构建上下文感知的深度拉格朗日网络，结合在线系统辨识与MPC，提升轨迹跟踪精度。**

- **链接: [http://arxiv.org/pdf/2506.15249v1](http://arxiv.org/pdf/2506.15249v1)**

> **作者:** Lucas Schulze; Jan Peters; Oleg Arenz
>
> **备注:** Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Controlling a robot based on physics-informed dynamic models, such as deep Lagrangian networks (DeLaN), can improve the generalizability and interpretability of the resulting behavior. However, in complex environments, the number of objects to potentially interact with is vast, and their physical properties are often uncertain. This complexity makes it infeasible to employ a single global model. Therefore, we need to resort to online system identification of context-aware models that capture only the currently relevant aspects of the environment. While physical principles such as the conservation of energy may not hold across varying contexts, ensuring physical plausibility for any individual context-aware model can still be highly desirable, particularly when using it for receding horizon control methods such as Model Predictive Control (MPC). Hence, in this work, we extend DeLaN to make it context-aware, combine it with a recurrent network for online system identification, and integrate it with a MPC for adaptive, physics-informed control. We also combine DeLaN with a residual dynamics model to leverage the fact that a nominal model of the robot is typically available. We evaluate our method on a 7-DOF robot arm for trajectory tracking under varying loads. Our method reduces the end-effector tracking error by 39%, compared to a 21% improvement achieved by a baseline that uses an extended Kalman filter.
>
---
#### [new 011] Time-Optimized Safe Navigation in Unstructured Environments through Learning Based Depth Completion
- **分类: cs.RO**

- **简介: 该论文属于无人机自主导航任务，旨在解决在非结构化环境中安全、实时路径规划的问题。通过融合立体与单目深度学习方法构建密集3D地图，并生成时间最优轨迹。**

- **链接: [http://arxiv.org/pdf/2506.14975v1](http://arxiv.org/pdf/2506.14975v1)**

> **作者:** Jeffrey Mao; Raghuram Cauligi Srinivas; Steven Nogar; Giuseppe Loianno
>
> **摘要:** Quadrotors hold significant promise for several applications such as agriculture, search and rescue, and infrastructure inspection. Achieving autonomous operation requires systems to navigate safely through complex and unfamiliar environments. This level of autonomy is particularly challenging due to the complexity of such environments and the need for real-time decision making especially for platforms constrained by size, weight, and power (SWaP), which limits flight time and precludes the use of bulky sensors like Light Detection and Ranging (LiDAR) for mapping. Furthermore, computing globally optimal, collision-free paths and translating them into time-optimized, safe trajectories in real time adds significant computational complexity. To address these challenges, we present a fully onboard, real-time navigation system that relies solely on lightweight onboard sensors. Our system constructs a dense 3D map of the environment using a novel visual depth estimation approach that fuses stereo and monocular learning-based depth, yielding longer-range, denser, and less noisy depth maps than conventional stereo methods. Building on this map, we introduce a novel planning and trajectory generation framework capable of rapidly computing time-optimal global trajectories. As the map is incrementally updated with new depth information, our system continuously refines the trajectory to maintain safety and optimality. Both our planner and trajectory generator outperforms state-of-the-art methods in terms of computational efficiency and guarantee obstacle-free trajectories. We validate our system through robust autonomous flight experiments in diverse indoor and outdoor environments, demonstrating its effectiveness for safe navigation in previously unknown settings.
>
---
#### [new 012] Context Matters: Learning Generalizable Rewards via Calibrated Features
- **分类: cs.RO**

- **简介: 该论文属于强化学习中的奖励学习任务，解决上下文影响下奖励函数泛化的问题。通过显式建模偏好与特征显著性，提升样本效率和个性化适应能力。**

- **链接: [http://arxiv.org/pdf/2506.15012v1](http://arxiv.org/pdf/2506.15012v1)**

> **作者:** Alexandra Forsey-Smerek; Julie Shah; Andreea Bobu
>
> **备注:** 30 pages, 21 figures
>
> **摘要:** A key challenge in reward learning from human input is that desired agent behavior often changes based on context. Traditional methods typically treat each new context as a separate task with its own reward function. For example, if a previously ignored stove becomes too hot to be around, the robot must learn a new reward from scratch, even though the underlying preference for prioritizing safety over efficiency remains unchanged. We observe that context influences not the underlying preference itself, but rather the $\textit{saliency}$--or importance--of reward features. For instance, stove heat affects the importance of the robot's proximity, yet the human's safety preference stays the same. Existing multi-task and meta IRL methods learn context-dependent representations $\textit{implicitly}$--without distinguishing between preferences and feature importance--resulting in substantial data requirements. Instead, we propose $\textit{explicitly}$ modeling context-invariant preferences separately from context-dependent feature saliency, creating modular reward representations that adapt to new contexts. To achieve this, we introduce $\textit{calibrated features}$--representations that capture contextual effects on feature saliency--and present specialized paired comparison queries that isolate saliency from preference for efficient learning. Experiments with simulated users show our method significantly improves sample efficiency, requiring 10x fewer preference queries than baselines to achieve equivalent reward accuracy, with up to 15% better performance in low-data regimes (5-10 queries). An in-person user study (N=12) demonstrates that participants can effectively teach their unique personal contextual preferences using our method, enabling more adaptable and personalized reward learning.
>
---
#### [new 013] Feedback-MPPI: Fast Sampling-Based MPC via Rollout Differentiation -- Adios low-level controllers
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人控制任务，解决实时高频率控制计算需求高的问题。通过引入反馈机制改进MPPI算法，提升控制性能与稳定性。**

- **链接: [http://arxiv.org/pdf/2506.14855v1](http://arxiv.org/pdf/2506.14855v1)**

> **作者:** Tommaso Belvedere; Michael Ziegltrum; Giulio Turrisi; Valerio Modugno
>
> **摘要:** Model Predictive Path Integral control is a powerful sampling-based approach suitable for complex robotic tasks due to its flexibility in handling nonlinear dynamics and non-convex costs. However, its applicability in real-time, highfrequency robotic control scenarios is limited by computational demands. This paper introduces Feedback-MPPI (F-MPPI), a novel framework that augments standard MPPI by computing local linear feedback gains derived from sensitivity analysis inspired by Riccati-based feedback used in gradient-based MPC. These gains allow for rapid closed-loop corrections around the current state without requiring full re-optimization at each timestep. We demonstrate the effectiveness of F-MPPI through simulations and real-world experiments on two robotic platforms: a quadrupedal robot performing dynamic locomotion on uneven terrain and a quadrotor executing aggressive maneuvers with onboard computation. Results illustrate that incorporating local feedback significantly improves control performance and stability, enabling robust, high-frequency operation suitable for complex robotic systems.
>
---
#### [new 014] Booster Gym: An End-to-End Reinforcement Learning Framework for Humanoid Robot Locomotion
- **分类: cs.RO**

- **简介: 该论文属于人形机器人运动控制任务，旨在解决强化学习策略从仿真到现实的迁移难题。工作包括构建端到端框架，涵盖训练、部署及环境随机化等关键技术。**

- **链接: [http://arxiv.org/pdf/2506.15132v1](http://arxiv.org/pdf/2506.15132v1)**

> **作者:** Yushi Wang; Penghui Chen; Xinyu Han; Feng Wu; Mingguo Zhao
>
> **摘要:** Recent advancements in reinforcement learning (RL) have led to significant progress in humanoid robot locomotion, simplifying the design and training of motion policies in simulation. However, the numerous implementation details make transferring these policies to real-world robots a challenging task. To address this, we have developed a comprehensive code framework that covers the entire process from training to deployment, incorporating common RL training methods, domain randomization, reward function design, and solutions for handling parallel structures. This library is made available as a community resource, with detailed descriptions of its design and experimental results. We validate the framework on the Booster T1 robot, demonstrating that the trained policies seamlessly transfer to the physical platform, enabling capabilities such as omnidirectional walking, disturbance resistance, and terrain adaptability. We hope this work provides a convenient tool for the robotics community, accelerating the development of humanoid robots. The code can be found in https://github.com/BoosterRobotics/booster_gym.
>
---
#### [new 015] Towards Perception-based Collision Avoidance for UAVs when Guiding the Visually Impaired
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于无人机辅助视障人士导航任务，旨在解决无人机在复杂环境中避障问题。工作包括设计感知路径规划系统和多DNN避障框架。**

- **链接: [http://arxiv.org/pdf/2506.14857v1](http://arxiv.org/pdf/2506.14857v1)**

> **作者:** Suman Raj; Swapnil Padhi; Ruchi Bhoot; Prince Modi; Yogesh Simmhan
>
> **备注:** 16 pages, 7 figures; Accepted as Late-Breaking Results at the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2023
>
> **摘要:** Autonomous navigation by drones using onboard sensors combined with machine learning and computer vision algorithms is impacting a number of domains, including agriculture, logistics, and disaster management. In this paper, we examine the use of drones for assisting visually impaired people (VIPs) in navigating through outdoor urban environments. Specifically, we present a perception-based path planning system for local planning around the neighborhood of the VIP, integrated with a global planner based on GPS and maps for coarse planning. We represent the problem using a geometric formulation and propose a multi DNN based framework for obstacle avoidance of the UAV as well as the VIP. Our evaluations conducted on a drone human system in a university campus environment verifies the feasibility of our algorithms in three scenarios; when the VIP walks on a footpath, near parked vehicles, and in a crowded street.
>
---
#### [new 016] SHeRLoc: Synchronized Heterogeneous Radar Place Recognition for Cross-Modal Localization
- **分类: cs.RO**

- **简介: 该论文属于跨模态定位任务，解决异构雷达数据识别难题。提出SHeRLoc网络，通过极化匹配和特征聚合提升识别性能。**

- **链接: [http://arxiv.org/pdf/2506.15175v1](http://arxiv.org/pdf/2506.15175v1)**

> **作者:** Hanjun Kim; Minwoo Jung; Wooseong Yang; Ayoung Kim
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Despite the growing adoption of radar in robotics, the majority of research has been confined to homogeneous sensor types, overlooking the integration and cross-modality challenges inherent in heterogeneous radar technologies. This leads to significant difficulties in generalizing across diverse radar data types, with modality-aware approaches that could leverage the complementary strengths of heterogeneous radar remaining unexplored. To bridge these gaps, we propose SHeRLoc, the first deep network tailored for heterogeneous radar, which utilizes RCS polar matching to align multimodal radar data. Our hierarchical optimal transport-based feature aggregation method generates rotationally robust multi-scale descriptors. By employing FFT-similarity-based data mining and adaptive margin-based triplet loss, SHeRLoc enables FOV-aware metric learning. SHeRLoc achieves an order of magnitude improvement in heterogeneous radar place recognition, increasing recall@1 from below 0.1 to 0.9 on a public dataset and outperforming state of-the-art methods. Also applicable to LiDAR, SHeRLoc paves the way for cross-modal place recognition and heterogeneous sensor SLAM. The source code will be available upon acceptance.
>
---
#### [new 017] Robust Instant Policy: Leveraging Student's t-Regression Model for Robust In-context Imitation Learning of Robot Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人模仿学习任务，解决LLM生成轨迹中的幻觉问题。提出RIP算法，利用Student's t回归模型提升轨迹可靠性。**

- **链接: [http://arxiv.org/pdf/2506.15157v1](http://arxiv.org/pdf/2506.15157v1)**

> **作者:** Hanbit Oh; Andrea M. Salcedo-Vázquez; Ixchel G. Ramirez-Alpizar; Yukiyasu Domae
>
> **备注:** IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025 accepted
>
> **摘要:** Imitation learning (IL) aims to enable robots to perform tasks autonomously by observing a few human demonstrations. Recently, a variant of IL, called In-Context IL, utilized off-the-shelf large language models (LLMs) as instant policies that understand the context from a few given demonstrations to perform a new task, rather than explicitly updating network models with large-scale demonstrations. However, its reliability in the robotics domain is undermined by hallucination issues such as LLM-based instant policy, which occasionally generates poor trajectories that deviate from the given demonstrations. To alleviate this problem, we propose a new robust in-context imitation learning algorithm called the robust instant policy (RIP), which utilizes a Student's t-regression model to be robust against the hallucinated trajectories of instant policies to allow reliable trajectory generation. Specifically, RIP generates several candidate robot trajectories to complete a given task from an LLM and aggregates them using the Student's t-distribution, which is beneficial for ignoring outliers (i.e., hallucinations); thereby, a robust trajectory against hallucinations is generated. Our experiments, conducted in both simulated and real-world environments, show that RIP significantly outperforms state-of-the-art IL methods, with at least $26\%$ improvement in task success rates, particularly in low-data scenarios for everyday tasks. Video results available at https://sites.google.com/view/robustinstantpolicy.
>
---
#### [new 018] Human Locomotion Implicit Modeling Based Real-Time Gait Phase Estimation
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于运动阶段估计任务，旨在提高外骨骼对步态变化的适应性。通过结合时序卷积与Transformer的隐式建模方法，提升了模型在不同地形下的准确性和鲁棒性。**

- **链接: [http://arxiv.org/pdf/2506.15150v1](http://arxiv.org/pdf/2506.15150v1)**

> **作者:** Yuanlong Ji; Xingbang Yang; Ruoqi Zhao; Qihan Ye; Quan Zheng; Yubo Fan
>
> **摘要:** Gait phase estimation based on inertial measurement unit (IMU) signals facilitates precise adaptation of exoskeletons to individual gait variations. However, challenges remain in achieving high accuracy and robustness, particularly during periods of terrain changes. To address this, we develop a gait phase estimation neural network based on implicit modeling of human locomotion, which combines temporal convolution for feature extraction with transformer layers for multi-channel information fusion. A channel-wise masked reconstruction pre-training strategy is proposed, which first treats gait phase state vectors and IMU signals as joint observations of human locomotion, thus enhancing model generalization. Experimental results demonstrate that the proposed method outperforms existing baseline approaches, achieving a gait phase RMSE of $2.729 \pm 1.071%$ and phase rate MAE of $0.037 \pm 0.016%$ under stable terrain conditions with a look-back window of 2 seconds, and a phase RMSE of $3.215 \pm 1.303%$ and rate MAE of $0.050 \pm 0.023%$ under terrain transitions. Hardware validation on a hip exoskeleton further confirms that the algorithm can reliably identify gait cycles and key events, adapting to various continuous motion scenarios. This research paves the way for more intelligent and adaptive exoskeleton systems, enabling safer and more efficient human-robot interaction across diverse real-world environments.
>
---
#### [new 019] Particle-Grid Neural Dynamics for Learning Deformable Object Models from RGB-D Videos
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于变形物体建模任务，旨在从RGB-D视频中学习动态模型。通过结合粒子与网格的混合表示，解决物体状态估计和动态建模难题。**

- **链接: [http://arxiv.org/pdf/2506.15680v1](http://arxiv.org/pdf/2506.15680v1)**

> **作者:** Kaifeng Zhang; Baoyu Li; Kris Hauser; Yunzhu Li
>
> **备注:** Project page: https://kywind.github.io/pgnd
>
> **摘要:** Modeling the dynamics of deformable objects is challenging due to their diverse physical properties and the difficulty of estimating states from limited visual information. We address these challenges with a neural dynamics framework that combines object particles and spatial grids in a hybrid representation. Our particle-grid model captures global shape and motion information while predicting dense particle movements, enabling the modeling of objects with varied shapes and materials. Particles represent object shapes, while the spatial grid discretizes the 3D space to ensure spatial continuity and enhance learning efficiency. Coupled with Gaussian Splattings for visual rendering, our framework achieves a fully learning-based digital twin of deformable objects and generates 3D action-conditioned videos. Through experiments, we demonstrate that our model learns the dynamics of diverse objects -- such as ropes, cloths, stuffed animals, and paper bags -- from sparse-view RGB-D recordings of robot-object interactions, while also generalizing at the category level to unseen instances. Our approach outperforms state-of-the-art learning-based and physics-based simulators, particularly in scenarios with limited camera views. Furthermore, we showcase the utility of our learned models in model-based planning, enabling goal-conditioned object manipulation across a range of tasks. The project page is available at https://kywind.github.io/pgnd .
>
---
#### [new 020] I Know You're Listening: Adaptive Voice for HRI
- **分类: cs.RO; cs.HC; cs.SD; eess.AS**

- **简介: 该论文属于人机交互任务，旨在提升语言教学机器人的语音效果。解决语音表达不足和环境适应性差的问题，通过开发轻量级语音、环境自适应调整及优化L2清晰度等方法提高教学效果。**

- **链接: [http://arxiv.org/pdf/2506.15107v1](http://arxiv.org/pdf/2506.15107v1)**

> **作者:** Paige Tuttösí
>
> **备注:** PhD Thesis Simon Fraser University https://summit.sfu.ca/item/39353 Read the Room: Adapting a Robot's Voice to Ambient and Social Contexts IROS 23 Mmm whatcha say? Uncovering distal and proximal context effects in first and second-language word perception using psychophysical reverse correlation INTERSPEECH 24 Emojivoice: Towards long-term controllable expressivity in robot speech RO-MAN 25
>
> **摘要:** While the use of social robots for language teaching has been explored, there remains limited work on a task-specific synthesized voices for language teaching robots. Given that language is a verbal task, this gap may have severe consequences for the effectiveness of robots for language teaching tasks. We address this lack of L2 teaching robot voices through three contributions: 1. We address the need for a lightweight and expressive robot voice. Using a fine-tuned version of Matcha-TTS, we use emoji prompting to create an expressive voice that shows a range of expressivity over time. The voice can run in real time with limited compute resources. Through case studies, we found this voice more expressive, socially appropriate, and suitable for long periods of expressive speech, such as storytelling. 2. We explore how to adapt a robot's voice to physical and social ambient environments to deploy our voices in various locations. We found that increasing pitch and pitch rate in noisy and high-energy environments makes the robot's voice appear more appropriate and makes it seem more aware of its current environment. 3. We create an English TTS system with improved clarity for L2 listeners using known linguistic properties of vowels that are difficult for these listeners. We used a data-driven, perception-based approach to understand how L2 speakers use duration cues to interpret challenging words with minimal tense (long) and lax (short) vowels in English. We found that the duration of vowels strongly influences the perception for L2 listeners and created an "L2 clarity mode" for Matcha-TTS that applies a lengthening to tense vowels while leaving lax vowels unchanged. Our clarity mode was found to be more respectful, intelligible, and encouraging than base Matcha-TTS while reducing transcription errors in these challenging tense/lax minimal pairs.
>
---
#### [new 021] VIMS: A Visual-Inertial-Magnetic-Sonar SLAM System in Underwater Environments
- **分类: cs.RO**

- **简介: 该论文属于SLAM任务，旨在解决水下环境中的定位与建图问题。通过融合视觉、惯性、磁和声呐信息，提升系统的鲁棒性和精度。**

- **链接: [http://arxiv.org/pdf/2506.15126v1](http://arxiv.org/pdf/2506.15126v1)**

> **作者:** Bingbing Zhang; Huan Yin; Shuo Liu; Fumin Zhang; Wen Xu
>
> **备注:** This work has been accepted for publication at the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** In this study, we present a novel simultaneous localization and mapping (SLAM) system, VIMS, designed for underwater navigation. Conventional visual-inertial state estimators encounter significant practical challenges in perceptually degraded underwater environments, particularly in scale estimation and loop closing. To address these issues, we first propose leveraging a low-cost single-beam sonar to improve scale estimation. Then, VIMS integrates a high-sampling-rate magnetometer for place recognition by utilizing magnetic signatures generated by an economical magnetic field coil. Building on this, a hierarchical scheme is developed for visual-magnetic place recognition, enabling robust loop closure. Furthermore, VIMS achieves a balance between local feature tracking and descriptor-based loop closing, avoiding additional computational burden on the front end. Experimental results highlight the efficacy of the proposed VIMS, demonstrating significant improvements in both the robustness and accuracy of state estimation within underwater environments.
>
---
#### [new 022] Six-DoF Hand-Based Teleoperation for Omnidirectional Aerial Robots
- **分类: cs.RO**

- **简介: 该论文属于空中机械臂的遥操作任务，旨在解决多旋翼飞行器六自由度控制不足的问题。通过手部动作捕捉实现更灵活的空中操作。**

- **链接: [http://arxiv.org/pdf/2506.15009v1](http://arxiv.org/pdf/2506.15009v1)**

> **作者:** Jinjie Li; Jiaxuan Li; Kotaro Kaneko; Liming Shu; Moju Zhao
>
> **备注:** 7 pages, 9 figures. This work has been accepted to IROS 2025. The video will be released soon
>
> **摘要:** Omnidirectional aerial robots offer full 6-DoF independent control over position and orientation, making them popular for aerial manipulation. Although advancements in robotic autonomy, operating by human remains essential in complex aerial environments. Existing teleoperation approaches for multirotors fail to fully leverage the additional DoFs provided by omnidirectional rotation. Additionally, the dexterity of human fingers should be exploited for more engaged interaction. In this work, we propose an aerial teleoperation system that brings the omnidirectionality of human hands into the unbounded aerial workspace. Our system includes two motion-tracking marker sets -- one on the shoulder and one on the hand -- along with a data glove to capture hand gestures. Using these inputs, we design four interaction modes for different tasks, including Spherical Mode and Cartesian Mode for long-range moving as well as Operation Mode and Locking Mode for precise manipulation, where the hand gestures are utilized for seamless mode switching. We evaluate our system on a valve-turning task in real world, demonstrating how each mode contributes to effective aerial manipulation. This interaction framework bridges human dexterity with aerial robotics, paving the way for enhanced teleoperated aerial manipulation in unstructured environments.
>
---
#### [new 023] Aerial Grasping via Maximizing Delta-Arm Workspace Utilization
- **分类: cs.RO**

- **简介: 该论文属于空中抓取任务，旨在解决机械臂工作空间利用率低的问题。通过优化轨迹和引入MLP与RevNet方法，提升操作灵活性和效率。**

- **链接: [http://arxiv.org/pdf/2506.15539v1](http://arxiv.org/pdf/2506.15539v1)**

> **作者:** Haoran Chen; Weiliang Deng; Biyu Ye; Yifan Xiong; Ximin Lyu
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** The workspace limits the operational capabilities and range of motion for the systems with robotic arms. Maximizing workspace utilization has the potential to provide more optimal solutions for aerial manipulation tasks, increasing the system's flexibility and operational efficiency. In this paper, we introduce a novel planning framework for aerial grasping that maximizes workspace utilization. We formulate an optimization problem to optimize the aerial manipulator's trajectory, incorporating task constraints to achieve efficient manipulation. To address the challenge of incorporating the delta arm's non-convex workspace into optimization constraints, we leverage a Multilayer Perceptron (MLP) to map position points to feasibility probabilities.Furthermore, we employ Reversible Residual Networks (RevNet) to approximate the complex forward kinematics of the delta arm, utilizing efficient model gradients to eliminate workspace constraints. We validate our methods in simulations and real-world experiments to demonstrate their effectiveness.
>
---
#### [new 024] TACT: Humanoid Whole-body Contact Manipulation through Deep Imitation Learning with Tactile Modality
- **分类: cs.RO**

- **简介: 该论文属于人形机器人全身体接触操作任务，旨在解决接触感知与控制难题。通过深度模仿学习结合触觉信息，提升操作鲁棒性与稳定性。**

- **链接: [http://arxiv.org/pdf/2506.15146v1](http://arxiv.org/pdf/2506.15146v1)**

> **作者:** Masaki Murooka; Takahiro Hoshi; Kensuke Fukumitsu; Shimpei Masuda; Marwan Hamze; Tomoya Sasaki; Mitsuharu Morisawa; Eiichi Yoshida
>
> **摘要:** Manipulation with whole-body contact by humanoid robots offers distinct advantages, including enhanced stability and reduced load. On the other hand, we need to address challenges such as the increased computational cost of motion generation and the difficulty of measuring broad-area contact. We therefore have developed a humanoid control system that allows a humanoid robot equipped with tactile sensors on its upper body to learn a policy for whole-body manipulation through imitation learning based on human teleoperation data. This policy, named tactile-modality extended ACT (TACT), has a feature to take multiple sensor modalities as input, including joint position, vision, and tactile measurements. Furthermore, by integrating this policy with retargeting and locomotion control based on a biped model, we demonstrate that the life-size humanoid robot RHP7 Kaleido is capable of achieving whole-body contact manipulation while maintaining balance and walking. Through detailed experimental verification, we show that inputting both vision and tactile modalities into the policy contributes to improving the robustness of manipulation involving broad and delicate contact.
>
---
#### [new 025] Efficient and Real-Time Motion Planning for Robotics Using Projection-Based Optimization
- **分类: cs.RO**

- **简介: 该论文属于机器人运动规划任务，解决复杂环境下实时运动生成问题。提出ALSPG方法，利用几何投影提升效率与性能。**

- **链接: [http://arxiv.org/pdf/2506.14865v1](http://arxiv.org/pdf/2506.14865v1)**

> **作者:** Xuemin Chi; Hakan Girgin; Tobias Löw; Yangyang Xie; Teng Xue; Jihao Huang; Cheng Hu; Zhitao Liu; Sylvain Calinon
>
> **备注:** submitted to IROS 2025
>
> **摘要:** Generating motions for robots interacting with objects of various shapes is a complex challenge, further complicated by the robot geometry and multiple desired behaviors. While current robot programming tools (such as inverse kinematics, collision avoidance, and manipulation planning) often treat these problems as constrained optimization, many existing solvers focus on specific problem domains or do not exploit geometric constraints effectively. We propose an efficient first-order method, Augmented Lagrangian Spectral Projected Gradient Descent (ALSPG), which leverages geometric projections via Euclidean projections, Minkowski sums, and basis functions. We show that by using geometric constraints rather than full constraints and gradients, ALSPG significantly improves real-time performance. Compared to second-order methods like iLQR, ALSPG remains competitive in the unconstrained case. We validate our method through toy examples and extensive simulations, and demonstrate its effectiveness on a 7-axis Franka robot, a 6-axis P-Rob robot and a 1:10 scale car in real-world experiments. Source codes, experimental data and videos are available on the project webpage: https://sites.google.com/view/alspg-oc
>
---
#### [new 026] Efficient Navigation Among Movable Obstacles using a Mobile Manipulator via Hierarchical Policy Learning
- **分类: cs.RO**

- **简介: 该论文研究移动机械臂在可移动障碍物中高效导航的任务，通过分层强化学习框架解决动态障碍物处理问题，提升路径效率与成功率。**

- **链接: [http://arxiv.org/pdf/2506.15380v1](http://arxiv.org/pdf/2506.15380v1)**

> **作者:** Taegeun Yang; Jiwoo Hwang; Jeil Jeong; Minsung Yoon; Sung-Eui Yoon
>
> **备注:** 8 pages, 6 figures, Accepted to IROS 2025. Supplementary Video: https://youtu.be/sZ8_z7sYVP0
>
> **摘要:** We propose a hierarchical reinforcement learning (HRL) framework for efficient Navigation Among Movable Obstacles (NAMO) using a mobile manipulator. Our approach combines interaction-based obstacle property estimation with structured pushing strategies, facilitating the dynamic manipulation of unforeseen obstacles while adhering to a pre-planned global path. The high-level policy generates pushing commands that consider environmental constraints and path-tracking objectives, while the low-level policy precisely and stably executes these commands through coordinated whole-body movements. Comprehensive simulation-based experiments demonstrate improvements in performing NAMO tasks, including higher success rates, shortened traversed path length, and reduced goal-reaching times, compared to baselines. Additionally, ablation studies assess the efficacy of each component, while a qualitative analysis further validates the accuracy and reliability of the real-time obstacle property estimation.
>
---
#### [new 027] SurfAAV: Design and Implementation of a Novel Multimodal Surfing Aquatic-Aerial Vehicle
- **分类: cs.RO**

- **简介: 该论文属于水空多模态机器人任务，旨在解决水下、水面和空中运动协同问题。设计了SurfAAV原型，实现高效水面滑行与水下航行，并验证其性能优势。**

- **链接: [http://arxiv.org/pdf/2506.15450v1](http://arxiv.org/pdf/2506.15450v1)**

> **作者:** Kun Liu; Junhao Xiao; Hao Lin; Yue Cao; Hui Peng; Kaihong Huang; Huimin Lu
>
> **摘要:** Despite significant advancements in the research of aquatic-aerial robots, existing configurations struggle to efficiently perform underwater, surface, and aerial movement simultaneously. In this paper, we propose a novel multimodal surfing aquatic-aerial vehicle, SurfAAV, which efficiently integrates underwater navigation, surface gliding, and aerial flying capabilities. Thanks to the design of the novel differential thrust vectoring hydrofoil, SurfAAV can achieve efficient surface gliding and underwater navigation without the need for a buoyancy adjustment system. This design provides flexible operational capabilities for both surface and underwater tasks, enabling the robot to quickly carry out underwater monitoring activities. Additionally, when it is necessary to reach another water body, SurfAAV can switch to aerial mode through a gliding takeoff, flying to the target water area to perform corresponding tasks. The main contribution of this letter lies in proposing a new solution for underwater, surface, and aerial movement, designing a novel hybrid prototype concept, developing the required control laws, and validating the robot's ability to successfully perform surface gliding and gliding takeoff. SurfAAV achieves a maximum surface gliding speed of 7.96 m/s and a maximum underwater speed of 3.1 m/s. The prototype's surface gliding maneuverability and underwater cruising maneuverability both exceed those of existing aquatic-aerial vehicles.
>
---
#### [new 028] Vision in Action: Learning Active Perception from Human Demonstrations
- **分类: cs.RO**

- **简介: 该论文属于机器人视觉感知任务，旨在解决多阶段操作中的视觉遮挡问题。通过人类示范学习主动感知策略，并设计VR接口实现高效人机协作。**

- **链接: [http://arxiv.org/pdf/2506.15666v1](http://arxiv.org/pdf/2506.15666v1)**

> **作者:** Haoyu Xiong; Xiaomeng Xu; Jimmy Wu; Yifan Hou; Jeannette Bohg; Shuran Song
>
> **摘要:** We present Vision in Action (ViA), an active perception system for bimanual robot manipulation. ViA learns task-relevant active perceptual strategies (e.g., searching, tracking, and focusing) directly from human demonstrations. On the hardware side, ViA employs a simple yet effective 6-DoF robotic neck to enable flexible, human-like head movements. To capture human active perception strategies, we design a VR-based teleoperation interface that creates a shared observation space between the robot and the human operator. To mitigate VR motion sickness caused by latency in the robot's physical movements, the interface uses an intermediate 3D scene representation, enabling real-time view rendering on the operator side while asynchronously updating the scene with the robot's latest observations. Together, these design elements enable the learning of robust visuomotor policies for three complex, multi-stage bimanual manipulation tasks involving visual occlusions, significantly outperforming baseline systems.
>
---
#### [new 029] Offensive Robot Cybersecurity
- **分类: cs.RO; cs.CR**

- **简介: 该论文属于机器人网络安全任务，旨在提升机器人防御能力。通过自动化攻击方法和机器学习，识别漏洞并构建自主防御系统。**

- **链接: [http://arxiv.org/pdf/2506.15343v1](http://arxiv.org/pdf/2506.15343v1)**

> **作者:** Víctor Mayoral-Vilches
>
> **备注:** Doctoral thesis
>
> **摘要:** Offensive Robot Cybersecurity introduces a groundbreaking approach by advocating for offensive security methods empowered by means of automation. It emphasizes the necessity of understanding attackers' tactics and identifying vulnerabilities in advance to develop effective defenses, thereby improving robots' security posture. This thesis leverages a decade of robotics experience, employing Machine Learning and Game Theory to streamline the vulnerability identification and exploitation process. Intrinsically, the thesis uncovers a profound connection between robotic architecture and cybersecurity, highlighting that the design and creation aspect of robotics deeply intertwines with its protection against attacks. This duality -- whereby the architecture that shapes robot behavior and capabilities also necessitates a defense mechanism through offensive and defensive cybersecurity strategies -- creates a unique equilibrium. Approaching cybersecurity with a dual perspective of defense and attack, rooted in an understanding of systems architecture, has been pivotal. Through comprehensive analysis, including ethical considerations, the development of security tools, and executing cyber attacks on robot software, hardware, and industry deployments, this thesis proposes a novel architecture for cybersecurity cognitive engines. These engines, powered by advanced game theory and machine learning, pave the way for autonomous offensive cybersecurity strategies for robots, marking a significant shift towards self-defending robotic systems. This research not only underscores the importance of offensive measures in enhancing robot cybersecurity but also sets the stage for future advancements where robots are not just resilient to cyber threats but are equipped to autonomously safeguard themselves.
>
---
#### [new 030] Model Predictive Path-Following Control for a Quadrotor
- **分类: eess.SY; cs.RO; cs.SY; 93-XX**

- **简介: 该论文属于无人机路径跟踪控制任务，旨在解决传统方法在处理约束和实时性上的不足，提出基于模型预测控制的路径跟踪方法，并在四旋翼无人机上验证其有效性。**

- **链接: [http://arxiv.org/pdf/2506.15447v1](http://arxiv.org/pdf/2506.15447v1)**

> **作者:** David Leprich; Mario Rosenfelder; Mario Hermle; Jingshan Chen; Peter Eberhard
>
> **备注:** 15 pages, 11 figures, submitted to PAMM 2025
>
> **摘要:** Automating drone-assisted processes is a complex task. Many solutions rely on trajectory generation and tracking, whereas in contrast, path-following control is a particularly promising approach, offering an intuitive and natural approach to automate tasks for drones and other vehicles. While different solutions to the path-following problem have been proposed, most of them lack the capability to explicitly handle state and input constraints, are formulated in a conservative two-stage approach, or are only applicable to linear systems. To address these challenges, the paper is built upon a Model Predictive Control-based path-following framework and extends its application to the Crazyflie quadrotor, which is investigated in hardware experiments. A cascaded control structure including an underlying attitude controller is included in the Model Predictive Path-Following Control formulation to meet the challenging real-time demands of quadrotor control. The effectiveness of the proposed method is demonstrated through real-world experiments, representing, to the best of the authors' knowledge, a novel application of this MPC-based path-following approach to the quadrotor. Additionally, as an extension to the original method, to allow for deviations of the path in cases where the precise following of the path might be overly restrictive, a corridor path-following approach is presented.
>
---
#### [new 031] Multi-Agent Reinforcement Learning for Autonomous Multi-Satellite Earth Observation: A Realistic Case Study
- **分类: cs.AI; cs.MA; cs.RO**

- **简介: 该论文属于多卫星自主协同任务，解决动态地球观测中的资源管理和协调问题，通过MARL方法提升多卫星系统的决策能力。**

- **链接: [http://arxiv.org/pdf/2506.15207v1](http://arxiv.org/pdf/2506.15207v1)**

> **作者:** Mohamad A. Hady; Siyi Hu; Mahardhika Pratama; Jimmy Cao; Ryszard Kowalczyk
>
> **摘要:** The exponential growth of Low Earth Orbit (LEO) satellites has revolutionised Earth Observation (EO) missions, addressing challenges in climate monitoring, disaster management, and more. However, autonomous coordination in multi-satellite systems remains a fundamental challenge. Traditional optimisation approaches struggle to handle the real-time decision-making demands of dynamic EO missions, necessitating the use of Reinforcement Learning (RL) and Multi-Agent Reinforcement Learning (MARL). In this paper, we investigate RL-based autonomous EO mission planning by modelling single-satellite operations and extending to multi-satellite constellations using MARL frameworks. We address key challenges, including energy and data storage limitations, uncertainties in satellite observations, and the complexities of decentralised coordination under partial observability. By leveraging a near-realistic satellite simulation environment, we evaluate the training stability and performance of state-of-the-art MARL algorithms, including PPO, IPPO, MAPPO, and HAPPO. Our results demonstrate that MARL can effectively balance imaging and resource management while addressing non-stationarity and reward interdependency in multi-satellite coordination. The insights gained from this study provide a foundation for autonomous satellite operations, offering practical guidelines for improving policy learning in decentralised EO missions.
>
---
#### [new 032] RaCalNet: Radar Calibration Network for Sparse-Supervised Metric Depth Estimation
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于深度估计任务，解决稀疏监督下高精度深度图生成问题。提出RaCalNet框架，通过稀疏LiDAR监督提升雷达点精度，实现无密集标注的高质量深度预测。**

- **链接: [http://arxiv.org/pdf/2506.15560v1](http://arxiv.org/pdf/2506.15560v1)**

> **作者:** Xingrui Qin; Wentao Zhao; Chuan Cao; Yihe Niu; Houcheng Jiang; Jingchuan Wang
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Dense metric depth estimation using millimeter-wave radar typically requires dense LiDAR supervision, generated via multi-frame projection and interpolation, to guide the learning of accurate depth from sparse radar measurements and RGB images. However, this paradigm is both costly and data-intensive. To address this, we propose RaCalNet, a novel framework that eliminates the need for dense supervision by using sparse LiDAR to supervise the learning of refined radar measurements, resulting in a supervision density of merely around 1% compared to dense-supervised methods. Unlike previous approaches that associate radar points with broad image regions and rely heavily on dense labels, RaCalNet first recalibrates and refines sparse radar points to construct accurate depth priors. These priors then serve as reliable anchors to guide monocular depth prediction, enabling metric-scale estimation without resorting to dense supervision. This design improves structural consistency and preserves fine details. Despite relying solely on sparse supervision, RaCalNet surpasses state-of-the-art dense-supervised methods, producing depth maps with clear object contours and fine-grained textures. Extensive experiments on the ZJU-4DRadarCam dataset and real-world deployment scenarios demonstrate its effectiveness, reducing RMSE by 35.30% and 34.89%, respectively.
>
---
#### [new 033] Recent Advances in Multi-Agent Human Trajectory Prediction: A Comprehensive Review
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 本文属于多智能体人类轨迹预测任务，旨在提升对多主体交互的理解。论文综述了2020至2024年的深度学习方法，分析其架构、输入与策略，并探讨未来方向。**

- **链接: [http://arxiv.org/pdf/2506.14831v1](http://arxiv.org/pdf/2506.14831v1)**

> **作者:** Céline Finet; Stephane Da Silva Martins; Jean-Bernard Hayet; Ioannis Karamouzas; Javad Amirian; Sylvie Le Hégarat-Mascle; Julien Pettré; Emanuel Aldea
>
> **备注:** 30 pages
>
> **摘要:** With the emergence of powerful data-driven methods in human trajectory prediction (HTP), gaining a finer understanding of multi-agent interactions lies within hand's reach, with important implications in areas such as autonomous navigation and crowd modeling. This survey reviews some of the most recent advancements in deep learning-based multi-agent trajectory prediction, focusing on studies published between 2020 and 2024. We categorize the existing methods based on their architectural design, their input representations, and their overall prediction strategies, placing a particular emphasis on models evaluated using the ETH/UCY benchmark. Furthermore, we highlight key challenges and future research directions in the field of multi-agent HTP.
>
---
#### [new 034] Probabilistic Trajectory GOSPA: A Metric for Uncertainty-Aware Multi-Object Tracking Performance Evaluation
- **分类: eess.SP; cs.RO**

- **简介: 该论文属于多目标跟踪任务，解决轨迹估计中不确定性评估问题，提出一种考虑存在性和状态不确定性的概率轨迹GOSPA度量。**

- **链接: [http://arxiv.org/pdf/2506.15148v1](http://arxiv.org/pdf/2506.15148v1)**

> **作者:** Yuxuan Xia; Ángel F. García-Fernández; Johan Karlsson; Yu Ge; Lennart Svensson; Ting Yuan
>
> **备注:** 7 pages, 4 figures
>
> **摘要:** This paper presents a generalization of the trajectory general optimal sub-pattern assignment (GOSPA) metric for evaluating multi-object tracking algorithms that provide trajectory estimates with track-level uncertainties. This metric builds on the recently introduced probabilistic GOSPA metric to account for both the existence and state estimation uncertainties of individual object states. Similar to trajectory GOSPA (TGOSPA), it can be formulated as a multidimensional assignment problem, and its linear programming relaxation--also a valid metric--is computable in polynomial time. Additionally, this metric retains the interpretability of TGOSPA, and we show that its decomposition yields intuitive costs terms associated to expected localization error and existence probability mismatch error for properly detected objects, expected missed and false detection error, and track switch error. The effectiveness of the proposed metric is demonstrated through a simulation study.
>
---
#### [new 035] Minimizing Structural Vibrations via Guided Flow Matching Design Optimization
- **分类: cs.CE; cs.LG; cs.RO; math.OC; stat.ML**

- **简介: 该论文属于结构优化任务，旨在减少板状结构的振动。通过结合生成流匹配与代理模型，实现低振动且可制造的设计优化。**

- **链接: [http://arxiv.org/pdf/2506.15263v1](http://arxiv.org/pdf/2506.15263v1)**

> **作者:** Jan van Delden; Julius Schultz; Sebastian Rothe; Christian Libner; Sabine C. Langer; Timo Lüddecke
>
> **摘要:** Structural vibrations are a source of unwanted noise in engineering systems like cars, trains or airplanes. Minimizing these vibrations is crucial for improving passenger comfort. This work presents a novel design optimization approach based on guided flow matching for reducing vibrations by placing beadings (indentations) in plate-like structures. Our method integrates a generative flow matching model and a surrogate model trained to predict structural vibrations. During the generation process, the flow matching model pushes towards manufacturability while the surrogate model pushes to low-vibration solutions. The flow matching model and its training data implicitly define the design space, enabling a broader exploration of potential solutions as no optimization of manually-defined design parameters is required. We apply our method to a range of differentiable optimization objectives, including direct optimization of specific eigenfrequencies through careful construction of the objective function. Results demonstrate that our method generates diverse and manufacturable plate designs with reduced structural vibrations compared to designs from random search, a criterion-based design heuristic and genetic optimization. The code and data are available from https://github.com/ecker-lab/Optimizing_Vibrating_Plates.
>
---
#### [new 036] Advances in Compliance Detection: Novel Models Using Vision-Based Tactile Sensors
- **分类: cs.CV; cs.RO; I.2.9**

- **简介: 该论文属于机器人感知任务，旨在解决传统方法在检测物体柔顺性上的不足。通过引入LRCN和Transformer模型，提升基于视觉触觉传感器的柔顺性预测精度。**

- **链接: [http://arxiv.org/pdf/2506.14980v1](http://arxiv.org/pdf/2506.14980v1)**

> **作者:** Ziteng Li; Malte Kuhlmann; Ilana Nisky; Nicolás Navarro-Guerrero
>
> **备注:** Accepted in the IEEE International Conference on Development and Learning (ICDL). The paper contains 8 pages and 7 figures
>
> **摘要:** Compliance is a critical parameter for describing objects in engineering, agriculture, and biomedical applications. Traditional compliance detection methods are limited by their lack of portability and scalability, rely on specialized, often expensive equipment, and are unsuitable for robotic applications. Moreover, existing neural network-based approaches using vision-based tactile sensors still suffer from insufficient prediction accuracy. In this paper, we propose two models based on Long-term Recurrent Convolutional Networks (LRCNs) and Transformer architectures that leverage RGB tactile images and other information captured by the vision-based sensor GelSight to predict compliance metrics accurately. We validate the performance of these models using multiple metrics and demonstrate their effectiveness in accurately estimating compliance. The proposed models exhibit significant performance improvement over the baseline. Additionally, we investigated the correlation between sensor compliance and object compliance estimation, which revealed that objects that are harder than the sensor are more challenging to estimate.
>
---
#### [new 037] Embodied Web Agents: Bridging Physical-Digital Realms for Integrated Agent Intelligence
- **分类: cs.AI; cs.CL; cs.CV; cs.MM; cs.RO**

- **简介: 该论文提出Embodied Web Agents，解决物理与数字智能融合问题。构建了集成3D环境与网络接口的仿真平台，发布基准测试任务，评估跨领域智能。**

- **链接: [http://arxiv.org/pdf/2506.15677v1](http://arxiv.org/pdf/2506.15677v1)**

> **作者:** Yining Hong; Rui Sun; Bingxuan Li; Xingcheng Yao; Maxine Wu; Alexander Chien; Da Yin; Ying Nian Wu; Zhecan James Wang; Kai-Wei Chang
>
> **摘要:** AI agents today are mostly siloed - they either retrieve and reason over vast amount of digital information and knowledge obtained online; or interact with the physical world through embodied perception, planning and action - but rarely both. This separation limits their ability to solve tasks that require integrated physical and digital intelligence, such as cooking from online recipes, navigating with dynamic map data, or interpreting real-world landmarks using web knowledge. We introduce Embodied Web Agents, a novel paradigm for AI agents that fluidly bridge embodiment and web-scale reasoning. To operationalize this concept, we first develop the Embodied Web Agents task environments, a unified simulation platform that tightly integrates realistic 3D indoor and outdoor environments with functional web interfaces. Building upon this platform, we construct and release the Embodied Web Agents Benchmark, which encompasses a diverse suite of tasks including cooking, navigation, shopping, tourism, and geolocation - all requiring coordinated reasoning across physical and digital realms for systematic assessment of cross-domain intelligence. Experimental results reveal significant performance gaps between state-of-the-art AI systems and human capabilities, establishing both challenges and opportunities at the intersection of embodied cognition and web-scale knowledge access. All datasets, codes and websites are publicly available at our project page https://embodied-web-agent.github.io/.
>
---
#### [new 038] Designing Intent: A Multimodal Framework for Human-Robot Cooperation in Industrial Workspaces
- **分类: cs.HC; cs.RO**

- **简介: 该论文属于人机协作任务，旨在解决工业场景中人类与机器人之间的意图沟通问题。提出多模态设计框架，以提升协作的透明度与信任度。**

- **链接: [http://arxiv.org/pdf/2506.15293v1](http://arxiv.org/pdf/2506.15293v1)**

> **作者:** Francesco Chiossi; Julian Rasch; Robin Welsch; Albrecht Schmidt; Florian Michahelles
>
> **备注:** 9 pages
>
> **摘要:** As robots enter collaborative workspaces, ensuring mutual understanding between human workers and robotic systems becomes a prerequisite for trust, safety, and efficiency. In this position paper, we draw on the cooperation scenario of the AIMotive project in which a human and a cobot jointly perform assembly tasks to argue for a structured approach to intent communication. Building on the Situation Awareness-based Agent Transparency (SAT) framework and the notion of task abstraction levels, we propose a multidimensional design space that maps intent content (SAT1, SAT3), planning horizon (operational to strategic), and modality (visual, auditory, haptic). We illustrate how this space can guide the design of multimodal communication strategies tailored to dynamic collaborative work contexts. With this paper, we lay the conceptual foundation for a future design toolkit aimed at supporting transparent human-robot interaction in the workplace. We highlight key open questions and design challenges, and propose a shared agenda for multimodal, adaptive, and trustworthy robotic collaboration in hybrid work environments.
>
---
#### [new 039] FindingDory: A Benchmark to Evaluate Memory in Embodied Agents
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人记忆评估任务，旨在解决长时记忆在具身智能体中的应用问题。提出一个基准测试，评估智能体在复杂环境中的记忆与推理能力。**

- **链接: [http://arxiv.org/pdf/2506.15635v1](http://arxiv.org/pdf/2506.15635v1)**

> **作者:** Karmesh Yadav; Yusuf Ali; Gunshi Gupta; Yarin Gal; Zsolt Kira
>
> **备注:** Our dataset and code will be made available at: https://findingdory-benchmark.github.io/
>
> **摘要:** Large vision-language models have recently demonstrated impressive performance in planning and control tasks, driving interest in their application to real-world robotics. However, deploying these models for reasoning in embodied contexts is limited by their ability to incorporate long-term experience collected across multiple days and represented by vast collections of images. Current VLMs typically struggle to process more than a few hundred images concurrently, highlighting the need for more efficient mechanisms to handle long-term memory in embodied settings. To effectively evaluate these models for long-horizon control, a benchmark must specifically target scenarios where memory is crucial for success. Existing long-video QA benchmarks overlook embodied challenges like object manipulation and navigation, which demand low-level skills and fine-grained reasoning over past interactions. Moreover, effective memory integration in embodied agents involves both recalling relevant historical information and executing actions based on that information, making it essential to study these aspects together rather than in isolation. In this work, we introduce a new benchmark for long-range embodied tasks in the Habitat simulator. This benchmark evaluates memory-based capabilities across 60 tasks requiring sustained engagement and contextual awareness in an environment. The tasks can also be procedurally extended to longer and more challenging versions, enabling scalable evaluation of memory and reasoning. We also present baselines that integrate state-of-the-art VLMs with low level navigation policies, assessing their performance on these memory-intensive tasks and highlight areas for improvement.
>
---
#### [new 040] HEAL: An Empirical Study on Hallucinations in Embodied Agents Driven by Large Language Models
- **分类: cs.LG; cs.RO**

- **简介: 该论文研究基于大语言模型的具身代理中的幻觉问题，属于机器人导航任务。旨在解决场景与任务不一致时产生的错误行为，通过构建测试集分析模型表现并提出改进建议。**

- **链接: [http://arxiv.org/pdf/2506.15065v1](http://arxiv.org/pdf/2506.15065v1)**

> **作者:** Trishna Chakraborty; Udita Ghosh; Xiaopan Zhang; Fahim Faisal Niloy; Yue Dong; Jiachen Li; Amit K. Roy-Chowdhury; Chengyu Song
>
> **摘要:** Large language models (LLMs) are increasingly being adopted as the cognitive core of embodied agents. However, inherited hallucinations, which stem from failures to ground user instructions in the observed physical environment, can lead to navigation errors, such as searching for a refrigerator that does not exist. In this paper, we present the first systematic study of hallucinations in LLM-based embodied agents performing long-horizon tasks under scene-task inconsistencies. Our goal is to understand to what extent hallucinations occur, what types of inconsistencies trigger them, and how current models respond. To achieve these goals, we construct a hallucination probing set by building on an existing benchmark, capable of inducing hallucination rates up to 40x higher than base prompts. Evaluating 12 models across two simulation environments, we find that while models exhibit reasoning, they fail to resolve scene-task inconsistencies-highlighting fundamental limitations in handling infeasible tasks. We also provide actionable insights on ideal model behavior for each scenario, offering guidance for developing more robust and reliable planning strategies.
>
---
## 更新

#### [replaced 001] LLM-as-BT-Planner: Leveraging LLMs for Behavior Tree Generation in Robot Task Planning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.10444v3](http://arxiv.org/pdf/2409.10444v3)**

> **作者:** Jicong Ao; Fan Wu; Yansong Wu; Abdalla Swikir; Sami Haddadin
>
> **备注:** 7 pages. presented in ICRA 2025
>
> **摘要:** Robotic assembly tasks remain an open challenge due to their long horizon nature and complex part relations. Behavior trees (BTs) are increasingly used in robot task planning for their modularity and flexibility, but creating them manually can be effort-intensive. Large language models (LLMs) have recently been applied to robotic task planning for generating action sequences, yet their ability to generate BTs has not been fully investigated. To this end, we propose LLM-as-BT-Planner, a novel framework that leverages LLMs for BT generation in robotic assembly task planning. Four in-context learning methods are introduced to utilize the natural language processing and inference capabilities of LLMs for producing task plans in BT format, reducing manual effort while ensuring robustness and comprehensibility. Additionally, we evaluate the performance of fine-tuned smaller LLMs on the same tasks. Experiments in both simulated and real-world settings demonstrate that our framework enhances LLMs' ability to generate BTs, improving success rate through in-context learning and supervised fine-tuning.
>
---
#### [replaced 002] PP-Tac: Paper Picking Using Tactile Feedback in Dexterous Robotic Hands
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.16649v2](http://arxiv.org/pdf/2504.16649v2)**

> **作者:** Pei Lin; Yuzhe Huang; Wanlin Li; Jianpeng Ma; Chenxi Xiao; Ziyuan Jiao
>
> **备注:** accepted by Robotics: Science and Systems(RSS) 2025 url: https://peilin-666.github.io/projects/PP-Tac/
>
> **摘要:** Robots are increasingly envisioned as human companions, assisting with everyday tasks that often involve manipulating deformable objects. Although recent advances in robotic hardware and embodied AI have expanded their capabilities, current systems still struggle with handling thin, flat, and deformable objects such as paper and fabric. This limitation arises from the lack of suitable perception techniques for robust state estimation under diverse object appearances, as well as the absence of planning techniques for generating appropriate grasp motions. To bridge these gaps, this paper introduces PP-Tac, a robotic system for picking up paper-like objects. PP-Tac features a multi-fingered robotic hand with high-resolution omnidirectional tactile sensors \sensorname. This hardware configuration enables real-time slip detection and online frictional force control that mitigates such slips. Furthermore, grasp motion generation is achieved through a trajectory synthesis pipeline, which first constructs a dataset of finger's pinching motions. Based on this dataset, a diffusion-based policy is trained to control the hand-arm robotic system. Experiments demonstrate that PP-Tac can effectively grasp paper-like objects of varying material, thickness, and stiffness, achieving an overall success rate of 87.5\%. To our knowledge, this work is the first attempt to grasp paper-like deformable objects using a tactile dexterous hand. Our project webpage can be found at: https://peilin-666.github.io/projects/PP-Tac/
>
---
#### [replaced 003] A compact neuromorphic system for ultra-energy-efficient, on-device robot localization
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2408.16754v2](http://arxiv.org/pdf/2408.16754v2)**

> **作者:** Adam D. Hines; Michael Milford; Tobias Fischer
>
> **备注:** 42 pages, 5 main figures, 8 supplementary figures, 2 supplementary tables, and 1 movie
>
> **摘要:** Neuromorphic computing offers a transformative pathway to overcome the computational and energy challenges faced in deploying robotic localization and navigation systems at the edge. Visual place recognition, a critical component for navigation, is often hampered by the high resource demands of conventional systems, making them unsuitable for small-scale robotic platforms which still require accurate long-endurance localization. Although neuromorphic approaches offer potential for greater efficiency, real-time edge deployment remains constrained by the complexity of bio-realistic networks. In order to overcome this challenge, fusion of hardware and algorithms is critical to employ this specialized computing paradigm. Here, we demonstrate a neuromorphic localization system that performs competitive place recognition in up to 8 kilometers of traversal using models as small as 180 kilobytes with 44,000 parameters, while consuming less than 8% of the energy required by conventional methods. Our Locational Encoding with Neuromorphic Systems (LENS) integrates spiking neural networks, an event-based dynamic vision sensor, and a neuromorphic processor within a single SynSense Speck chip, enabling real-time, energy-efficient localization on a hexapod robot. When compared to a benchmark place recognition method, Sum-of-Absolute-Differences (SAD), LENS performs comparably in overall precision. LENS represents an accurate fully neuromorphic localization system capable of large-scale, on-device deployment for energy efficient robotic place recognition. Neuromorphic computing enables resource-constrained robots to perform energy efficient, accurate localization.
>
---
#### [replaced 004] Learning the Geometric Mechanics of Robot Motion Using Gaussian Mixtures
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.05309v2](http://arxiv.org/pdf/2502.05309v2)**

> **作者:** Ruizhen Hu; Shai Revzen
>
> **备注:** 16 pages, 10 figures
>
> **摘要:** Data-driven models of robot motion constructed using principles from Geometric Mechanics have been shown to produce useful predictions of robot motion for a variety of robots. For robots with a useful number of DoF, these geometric mechanics models can only be constructed in the neighborhood of a gait. Here we show how Gaussian Mixture Models (GMM) can be used as a form of manifold learning that learns the structure of the Geometric Mechanics "motility map" and demonstrate: [i] a sizable improvement in prediction quality when compared to the previously published methods; [ii] a method that can be applied to any motion dataset and not only periodic gait data; [iii] a way to pre-process the data-set to facilitate extrapolation in places where the motility map is known to be linear. Our results can be applied anywhere a data-driven geometric motion model might be useful.
>
---
#### [replaced 005] DreamGen: Unlocking Generalization in Robot Learning through Video World Models
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2505.12705v2](http://arxiv.org/pdf/2505.12705v2)**

> **作者:** Joel Jang; Seonghyeon Ye; Zongyu Lin; Jiannan Xiang; Johan Bjorck; Yu Fang; Fengyuan Hu; Spencer Huang; Kaushil Kundalia; Yen-Chen Lin; Loic Magne; Ajay Mandlekar; Avnish Narayan; You Liang Tan; Guanzhi Wang; Jing Wang; Qi Wang; Yinzhen Xu; Xiaohui Zeng; Kaiyuan Zheng; Ruijie Zheng; Ming-Yu Liu; Luke Zettlemoyer; Dieter Fox; Jan Kautz; Scott Reed; Yuke Zhu; Linxi Fan
>
> **备注:** See website for videos: https://research.nvidia.com/labs/gear/dreamgen
>
> **摘要:** We introduce DreamGen, a simple yet highly effective 4-stage pipeline for training robot policies that generalize across behaviors and environments through neural trajectories - synthetic robot data generated from video world models. DreamGen leverages state-of-the-art image-to-video generative models, adapting them to the target robot embodiment to produce photorealistic synthetic videos of familiar or novel tasks in diverse environments. Since these models generate only videos, we recover pseudo-action sequences using either a latent action model or an inverse-dynamics model (IDM). Despite its simplicity, DreamGen unlocks strong behavior and environment generalization: a humanoid robot can perform 22 new behaviors in both seen and unseen environments, while requiring teleoperation data from only a single pick-and-place task in one environment. To evaluate the pipeline systematically, we introduce DreamGen Bench, a video generation benchmark that shows a strong correlation between benchmark performance and downstream policy success. Our work establishes a promising new axis for scaling robot learning well beyond manual data collection. Code available at https://github.com/NVIDIA/GR00T-Dreams.
>
---
#### [replaced 006] SurgSora: Object-Aware Diffusion Model for Controllable Surgical Video Generation
- **分类: cs.CV; cs.AI; cs.MM; cs.RO**

- **链接: [http://arxiv.org/pdf/2412.14018v2](http://arxiv.org/pdf/2412.14018v2)**

> **作者:** Tong Chen; Shuya Yang; Junyi Wang; Long Bai; Hongliang Ren; Luping Zhou
>
> **摘要:** Surgical video generation can enhance medical education and research, but existing methods lack fine-grained motion control and realism. We introduce SurgSora, a framework that generates high-fidelity, motion-controllable surgical videos from a single input frame and user-specified motion cues. Unlike prior approaches that treat objects indiscriminately or rely on ground-truth segmentation masks, SurgSora leverages self-predicted object features and depth information to refine RGB appearance and optical flow for precise video synthesis. It consists of three key modules: (1) the Dual Semantic Injector, which extracts object-specific RGB-D features and segmentation cues to enhance spatial representations; (2) the Decoupled Flow Mapper, which fuses multi-scale optical flow with semantic features for realistic motion dynamics; and (3) the Trajectory Controller, which estimates sparse optical flow and enables user-guided object movement. By conditioning these enriched features within the Stable Video Diffusion, SurgSora achieves state-of-the-art visual authenticity and controllability in advancing surgical video synthesis, as demonstrated by extensive quantitative and qualitative comparisons. Our human evaluation in collaboration with expert surgeons further demonstrates the high realism of SurgSora-generated videos, highlighting the potential of our method for surgical training and education. Our project is available at https://surgsora.github.io/surgsora.github.io.
>
---
#### [replaced 007] Mass-Adaptive Admittance Control for Robotic Manipulators
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2504.16224v2](http://arxiv.org/pdf/2504.16224v2)**

> **作者:** Hossein Gholampour; Jonathon E. Slightam; Logan E. Beaver
>
> **备注:** 6 pages, 7 figures
>
> **摘要:** Handling objects with unknown or changing masses is a common challenge in robotics, often leading to errors or instability if the control system cannot adapt in real-time. In this paper, we present a novel approach that enables a six-degrees-of-freedom robotic manipulator to reliably follow waypoints while automatically estimating and compensating for unknown payload weight. Our method integrates an admittance control framework with a mass estimator, allowing the robot to dynamically update an excitation force to compensate for the payload mass. This strategy mitigates end-effector sagging and preserves stability when handling objects of unknown weights. We experimentally validated our approach in a challenging pick-and-place task on a shelf with a crossbar, improved accuracy in reaching waypoints and compliant motion compared to a baseline admittance-control scheme. By safely accommodating unknown payloads, our work enhances flexibility in robotic automation and represents a significant step forward in adaptive control for uncertain environments.
>
---
#### [replaced 008] Human-Robot Co-Transportation using Disturbance-Aware MPC with Pose Optimization
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2404.00514v3](http://arxiv.org/pdf/2404.00514v3)**

> **作者:** Al Jaber Mahmud; Amir Hossain Raj; Duc M. Nguyen; Weizi Li; Xuesu Xiao; Xuan Wang
>
> **备注:** 8 pages, 6 figures
>
> **摘要:** This paper proposes a new control algorithm for human-robot co-transportation using a robot manipulator equipped with a mobile base and a robotic arm. We integrate the regular Model Predictive Control (MPC) with a novel pose optimization mechanism to more efficiently mitigate disturbances (such as human behavioral uncertainties or robot actuation noise) during the task. The core of our methodology involves a two-step iterative design: At each planning horizon, we determine the optimal pose of the robotic arm (joint angle configuration) from a candidate set, aiming to achieve the lowest estimated control cost. This selection is based on solving a disturbance-aware Discrete Algebraic Ricatti Equation (DARE), which also determines the optimal inputs for the robot's whole body control (including both the mobile base and the robotic arm). To validate the effectiveness of the proposed approach, we provide theoretical derivation for the disturbance-aware DARE and perform simulated experiments and hardware demos using a Fetch robot under varying conditions, including different trajectories and different levels of disturbances. The results reveal that our proposed approach outperforms baseline algorithms.
>
---
#### [replaced 009] Semantic Mapping in Indoor Embodied AI -- A Survey on Advances, Challenges, and Future Directions
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.05750v2](http://arxiv.org/pdf/2501.05750v2)**

> **作者:** Sonia Raychaudhuri; Angel X. Chang
>
> **摘要:** Intelligent embodied agents (e.g. robots) need to perform complex semantic tasks in unfamiliar environments. Among many skills that the agents need to possess, building and maintaining a semantic map of the environment is most crucial in long-horizon tasks. A semantic map captures information about the environment in a structured way, allowing the agent to reference it for advanced reasoning throughout the task. While existing surveys in embodied AI focus on general advancements or specific tasks like navigation and manipulation, this paper provides a comprehensive review of semantic map-building approaches in embodied AI, specifically for indoor navigation. We categorize these approaches based on their structural representation (spatial grids, topological graphs, dense point-clouds or hybrid maps) and the type of information they encode (implicit features or explicit environmental data). We also explore the strengths and limitations of the map building techniques, highlight current challenges, and propose future research directions. We identify that the field is moving towards developing open-vocabulary, queryable, task-agnostic map representations, while high memory demands and computational inefficiency still remaining to be open challenges. This survey aims to guide current and future researchers in advancing semantic mapping techniques for embodied AI systems.
>
---
#### [replaced 010] An Advanced Framework for Ultra-Realistic Simulation and Digital Twinning for Autonomous Vehicles
- **分类: cs.RO; G.3**

- **链接: [http://arxiv.org/pdf/2405.01328v2](http://arxiv.org/pdf/2405.01328v2)**

> **作者:** Yuankai He; Hanlin Chen; Weisong Shi
>
> **备注:** 6 Pages. 5 Figures, 1 Table
>
> **摘要:** Simulation is a fundamental tool in developing autonomous vehicles, enabling rigorous testing without the logistical and safety challenges associated with real-world trials. As autonomous vehicle technologies evolve and public safety demands increase, advanced, realistic simulation frameworks are critical. Current testing paradigms employ a mix of general-purpose and specialized simulators, such as CARLA and IVRESS, to achieve high-fidelity results. However, these tools often struggle with compatibility due to differing platform, hardware, and software requirements, severely hampering their combined effectiveness. This paper introduces BlueICE, an advanced framework for ultra-realistic simulation and digital twinning, to address these challenges. BlueICE's innovative architecture allows for the decoupling of computing platforms, hardware, and software dependencies while offering researchers customizable testing environments to meet diverse fidelity needs. Key features include containerization to ensure compatibility across different systems, a unified communication bridge for seamless integration of various simulation tools, and synchronized orchestration of input and output across simulators. This framework facilitates the development of sophisticated digital twins for autonomous vehicle testing and sets a new standard in simulation accuracy and flexibility. The paper further explores the application of BlueICE in two distinct case studies: the ICAT indoor testbed and the STAR campus outdoor testbed at the University of Delaware. These case studies demonstrate BlueICE's capability to create sophisticated digital twins for autonomous vehicle testing and underline its potential as a standardized testbed for future autonomous driving technologies.
>
---
#### [replaced 011] Tailless Flapping-Wing Robot With Bio-Inspired Elastic Passive Legs for Multi-Modal Locomotion
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.00805v2](http://arxiv.org/pdf/2503.00805v2)**

> **作者:** Zhi Zheng; Xiangyu Xu; Jin Wang; Yikai Chen; Jingyang Huang; Ruixin Wu; Huan Yu; Guodong Lu
>
> **备注:** 8 pages, 11 figures, accepted by IEEE Robotics and Automation Letters (RAL)
>
> **摘要:** Flapping-wing robots offer significant versatility; however, achieving efficient multi-modal locomotion remains challenging. This paper presents the design, modeling, and experimentation of a novel tailless flapping-wing robot with three independently actuated pairs of wings. Inspired by the leg morphology of juvenile water striders, the robot incorporates bio-inspired elastic passive legs that convert flapping-induced vibrations into directional ground movement, enabling locomotion without additional actuators. This vibration-driven mechanism facilitates lightweight, mechanically simplified multi-modal mobility. An SE(3)-based controller coordinates flight and mode transitions with minimal actuation. To validate the robot's feasibility, a functional prototype was developed, and experiments were conducted to evaluate its flight, ground locomotion, and mode-switching capabilities. Results show satisfactory performance under constrained actuation, highlighting the potential of multi-modal flapping-wing designs for future aerial-ground robotic applications. These findings provide a foundation for future studies on frequency-based terrestrial control and passive yaw stabilization in hybrid locomotion systems.
>
---
#### [replaced 012] Map Space Belief Prediction for Manipulation-Enhanced Mapping
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2502.20606v3](http://arxiv.org/pdf/2502.20606v3)**

> **作者:** Joao Marcos Correia Marques; Nils Dengler; Tobias Zaenker; Jesper Mucke; Shenlong Wang; Maren Bennewitz; Kris Hauser
>
> **备注:** 14 pages, 10 figures; Published at RSS 2025 - this version contains a small fix to figure 6 which was missing a plot in the original submission
>
> **摘要:** Searching for objects in cluttered environments requires selecting efficient viewpoints and manipulation actions to remove occlusions and reduce uncertainty in object locations, shapes, and categories. In this work, we address the problem of manipulation-enhanced semantic mapping, where a robot has to efficiently identify all objects in a cluttered shelf. Although Partially Observable Markov Decision Processes~(POMDPs) are standard for decision-making under uncertainty, representing unstructured interactive worlds remains challenging in this formalism. To tackle this, we define a POMDP whose belief is summarized by a metric-semantic grid map and propose a novel framework that uses neural networks to perform map-space belief updates to reason efficiently and simultaneously about object geometries, locations, categories, occlusions, and manipulation physics. Further, to enable accurate information gain analysis, the learned belief updates should maintain calibrated estimates of uncertainty. Therefore, we propose Calibrated Neural-Accelerated Belief Updates (CNABUs) to learn a belief propagation model that generalizes to novel scenarios and provides confidence-calibrated predictions for unknown areas. Our experiments show that our novel POMDP planner improves map completeness and accuracy over existing methods in challenging simulations and successfully transfers to real-world cluttered shelves in zero-shot fashion.
>
---
#### [replaced 013] Semantic-Geometric-Physical-Driven Robot Manipulation Skill Transfer via Skill Library and Tactile Representation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.11714v2](http://arxiv.org/pdf/2411.11714v2)**

> **作者:** Mingchao Qi; Yuanjin Li; Xing Liu; Zhengxiong Liu; Panfeng Huang
>
> **摘要:** Developing general robotic systems capable of manipulating in unstructured environments is a significant challenge, particularly as the tasks involved are typically long-horizon and rich-contact, requiring efficient skill transfer across different task scenarios. To address these challenges, we propose knowledge graph-based skill library construction method. This method hierarchically organizes manipulation knowledge using "task graph" and "scene graph" to represent task-specific and scene-specific information, respectively. Additionally, we introduce "state graph" to facilitate the interaction between high-level task planning and low-level scene information. Building upon this foundation, we further propose a novel hierarchical skill transfer framework based on the skill library and tactile representation, which integrates high-level reasoning for skill transfer and low-level precision for execution. At the task level, we utilize large language models (LLMs) and combine contextual learning with a four-stage chain-of-thought prompting paradigm to achieve subtask sequence transfer. At the motion level, we develop an adaptive trajectory transfer method based on the skill library and the heuristic path planning algorithm. At the physical level, we propose an adaptive contour extraction and posture perception method based on tactile representation. This method dynamically acquires high-precision contour and posture information from visual-tactile images, adjusting parameters such as contact position and posture to ensure the effectiveness of transferred skills in new environments. Experiments demonstrate the skill transfer and adaptability capabilities of the proposed methods across different task scenarios. Project website: https://github.com/MingchaoQi/skill_transfer
>
---
#### [replaced 014] An Actionable Hierarchical Scene Representation Enhancing Autonomous Inspection Missions in Unknown Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2412.19582v2](http://arxiv.org/pdf/2412.19582v2)**

> **作者:** Vignesh Kottayam Viswanathan; Mario Alberto Valdes Saucedo; Sumeet Gajanan Satpute; Christoforos Kanellakis; George Nikolakopoulos
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** In this article, we present the Layered Semantic Graphs (LSG), a novel actionable hierarchical scene graph, fully integrated with a multi-modal mission planner, the FLIE: A First-Look based Inspection and Exploration planner. The novelty of this work stems from aiming to address the task of maintaining an intuitive and multi-resolution scene representation, while simultaneously offering a tractable foundation for planning and scene understanding during an ongoing inspection mission of apriori unknown targets-of-interest in an unknown environment. The proposed LSG scheme is composed of locally nested hierarchical graphs, at multiple layers of abstraction, with the abstract concepts grounded on the functionality of the integrated FLIE planner. Furthermore, LSG encapsulates real-time semantic segmentation models that offer extraction and localization of desired semantic elements within the hierarchical representation. This extends the capability of the inspection planner, which can then leverage LSG to make an informed decision to inspect a particular semantic of interest. We also emphasize the hierarchical and semantic path-planning capabilities of LSG, which could extend inspection missions by improving situational awareness for human operators in an unknown environment. The validity of the proposed scheme is proven through extensive evaluations of the proposed architecture in simulations, as well as experimental field deployments on a Boston Dynamics Spot quadruped robot in urban outdoor environment settings.
>
---
