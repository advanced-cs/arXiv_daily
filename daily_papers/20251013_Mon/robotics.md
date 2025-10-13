# 机器人 cs.RO

- **最新发布 44 篇**

- **更新 34 篇**

## 最新发布

#### [new 001] Bridging Research and Practice in Simulation-based Testing of Industrial Robot Navigation Systems
- **分类: cs.RO; cs.SE**

- **简介: 该论文属于机器人仿真测试任务，旨在解决工业机器人在动态环境中导航的鲁棒性验证问题。作者将原本用于无人机的仿真测试框架Surrealist扩展至四足机器人ANYmal的工业检测场景，采用基于搜索的算法自动生成挑战性避障测试用例，发现了实验算法中的关键缺陷，并在实际工业流程中验证了框架的有效性。**

- **链接: [http://arxiv.org/pdf/2510.09396v1](http://arxiv.org/pdf/2510.09396v1)**

> **作者:** Sajad Khatiri; Francisco Eli Vina Barrientos; Maximilian Wulf; Paolo Tonella; Sebastiano Panichella
>
> **备注:** 12 pages, accepted for publication at IEEE/ACM International Conference on Automated Software Engineering (ASE) 2025 - Industry Showcase Track
>
> **摘要:** Ensuring robust robotic navigation in dynamic environments is a key challenge, as traditional testing methods often struggle to cover the full spectrum of operational requirements. This paper presents the industrial adoption of Surrealist, a simulation-based test generation framework originally for UAVs, now applied to the ANYmal quadrupedal robot for industrial inspection. Our method uses a search-based algorithm to automatically generate challenging obstacle avoidance scenarios, uncovering failures often missed by manual testing. In a pilot phase, generated test suites revealed critical weaknesses in one experimental algorithm (40.3% success rate) and served as an effective benchmark to prove the superior robustness of another (71.2% success rate). The framework was then integrated into the ANYbotics workflow for a six-month industrial evaluation, where it was used to test five proprietary algorithms. A formal survey confirmed its value, showing it enhances the development process, uncovers critical failures, provides objective benchmarks, and strengthens the overall verification pipeline.
>
---
#### [new 002] PLEXUS Hand: Lightweight Four-Motor Prosthetic Hand Enabling Precision-Lateral Dexterous Manipulation
- **分类: cs.RO**

- **简介: 该论文设计了一款轻量四电机假手PLEXUS Hand，旨在实现精准侧向灵活操作。任务是解决现有电动假手过重、内部空间不足及操作复杂等问题。通过单轴拇指与优化设计，实现仅用四个电机完成姿态转换与旋转操作，成功率达90-100%，适用于日常精细操作任务。**

- **链接: [http://arxiv.org/pdf/2510.09209v1](http://arxiv.org/pdf/2510.09209v1)**

> **作者:** Yuki Kuroda; Tomoya Takahashi; Cristian C Beltran-Hernandez; Masashi Hamaya; Kazutoshi Tanaka
>
> **摘要:** Electric prosthetic hands should be lightweight to decrease the burden on the user, shaped like human hands for cosmetic purposes, and have motors inside to protect them from damage and dirt. In addition to the ability to perform daily activities, these features are essential for everyday use of the hand. In-hand manipulation is necessary to perform daily activities such as transitioning between different postures, particularly through rotational movements, such as reorienting cards before slot insertion and operating tools such as screwdrivers. However, currently used electric prosthetic hands only achieve static grasp postures, and existing manipulation approaches require either many motors, which makes the prosthesis heavy for daily use in the hand, or complex mechanisms that demand a large internal space and force external motor placement, complicating attachment and exposing the components to damage. Alternatively, we combine a single-axis thumb and optimized thumb positioning to achieve basic posture and in-hand manipulation, that is, the reorientation between precision and lateral grasps, using only four motors in a lightweight (311 g) prosthetic hand. Experimental validation using primitive objects of various widths (5-30 mm) and shapes (cylinders and prisms) resulted in success rates of 90-100% for reorientation tasks. The hand performed seal stamping and USB device insertion, as well as rotation to operate a screwdriver.
>
---
#### [new 003] A geometrical approach to solve the proximity of a point to an axisymmetric quadric in space
- **分类: cs.RO**

- **简介: 该论文属于几何计算任务，旨在解决三维空间中点到轴对称二次曲面的最近距离问题。作者将问题从三维简化至二维，并基于圆锥曲线几何特性提出新解法，进一步细分了抛物线、椭圆/双曲线的处理情况，提高了计算效率且适合编程实现。**

- **链接: [http://arxiv.org/pdf/2510.08973v1](http://arxiv.org/pdf/2510.08973v1)**

> **作者:** Bibekananda Patra; Aditya Mahesh Kolte; Sandipan Bandyopadhyay
>
> **摘要:** This paper presents the classification of a general quadric into an axisymmetric quadric (AQ) and the solution to the problem of the proximity of a given point to an AQ. The problem of proximity in $R^3$ is reduced to the same in $R^2$, which is not found in the literature. A new method to solve the problem in $R^2$ is used based on the geometrical properties of the conics, such as sub-normal, length of the semi-major axis, eccentricity, slope and radius. Furthermore, the problem in $R^2$ is categorised into two and three more sub-cases for parabola and ellipse/hyperbola, respectively, depending on the location of the point, which is a novel approach as per the authors' knowledge. The proposed method is suitable for implementation in a common programming language, such as C and proved to be faster than a commercial library, namely, Bullet.
>
---
#### [new 004] Zero-shot Structure Learning and Planning for Autonomous Robot Navigation using Active Inference
- **分类: cs.RO**

- **简介: 论文提出了一种基于主动推理的自主导航框架AIMAPP，用于机器人在未知环境中实现自监督的探索、定位与路径规划。该模型受海马导航机制启发，结合拓扑推理与状态转移学习，通过最小化预期自由能平衡探索与目标导向行为，具备良好的适应性与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2510.09574v1](http://arxiv.org/pdf/2510.09574v1)**

> **作者:** Daria de tinguy; Tim Verbelen; Emilio Gamba; Bart Dhoedt
>
> **备注:** yet to be submitted
>
> **摘要:** Autonomous navigation in unfamiliar environments requires robots to simultaneously explore, localise, and plan under uncertainty, without relying on predefined maps or extensive training. We present a biologically inspired, Active Inference-based framework, Active Inference MAPping and Planning (AIMAPP). This model unifies mapping, localisation, and decision-making within a single generative model. Inspired by hippocampal navigation, it uses topological reasoning, place-cell encoding, and episodic memory to guide behaviour. The agent builds and updates a sparse topological map online, learns state transitions dynamically, and plans actions by minimising Expected Free Energy. This allows it to balance goal-directed and exploratory behaviours. We implemented a ROS-compatible navigation system that is sensor and robot-agnostic, capable of integrating with diverse hardware configurations. It operates in a fully self-supervised manner, is resilient to drift, and supports both exploration and goal-directed navigation without any pre-training. We demonstrate robust performance in large-scale real and simulated environments against state-of-the-art planning models, highlighting the system's adaptability to ambiguous observations, environmental changes, and sensor noise. The model offers a biologically inspired, modular solution to scalable, self-supervised navigation in unstructured settings. AIMAPP is available at https://github.com/decide-ugent/AIMAPP.
>
---
#### [new 005] Glovity: Learning Dexterous Contact-Rich Manipulation via Spatial Wrench Feedback Teleoperation System
- **分类: cs.RO**

- **简介: 论文提出Glovity，一种低成本可穿戴遥操作系统，用于灵巧的接触丰富操作任务。它通过空间力反馈和触觉手套，解决遥操作中力觉与触觉缺失、灵巧性不足的问题。系统包含力反馈装置与指尖校准，提升抓取与操作性能，并结合模仿学习实现复杂接触任务的高效学习。**

- **链接: [http://arxiv.org/pdf/2510.09229v1](http://arxiv.org/pdf/2510.09229v1)**

> **作者:** Yuyang Gao; Haofei Ma; Pai Zheng
>
> **摘要:** We present Glovity, a novel, low-cost wearable teleoperation system that integrates a spatial wrench (force-torque) feedback device with a haptic glove featuring fingertip Hall sensor calibration, enabling feedback-rich dexterous manipulation. Glovity addresses key challenges in contact-rich tasks by providing intuitive wrench and tactile feedback, while overcoming embodiment gaps through precise retargeting. User studies demonstrate significant improvements: wrench feedback boosts success rates in book-flipping tasks from 48% to 78% and reduces completion time by 25%, while fingertip calibration enhances thin-object grasping success significantly compared to commercial glove. Furthermore, incorporating wrench signals into imitation learning (via DP-R3M) achieves high success rate in novel contact-rich scenarios, such as adaptive page flipping and force-aware handovers. All hardware designs, software will be open-sourced. Project website: https://glovity.github.io/
>
---
#### [new 006] Obstacle Avoidance using Dynamic Movement Primitives and Reinforcement Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人运动规划任务，旨在解决避障轨迹生成问题。现有方法依赖大量数据或人工示范，而本文提出结合动态运动基元（DMP）与强化学习的方法，仅需一次人工示范即可生成多样化的避障轨迹。通过编码DMP并用强化学习优化，生成用于训练神经网络的轨迹数据集，实现快速、平滑的三维避障路径生成。**

- **链接: [http://arxiv.org/pdf/2510.09254v1](http://arxiv.org/pdf/2510.09254v1)**

> **作者:** Dominik Urbaniak; Alejandro Agostini; Pol Ramon; Jan Rosell; Raúl Suárez; Michael Suppa
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Learning-based motion planning can quickly generate near-optimal trajectories. However, it often requires either large training datasets or costly collection of human demonstrations. This work proposes an alternative approach that quickly generates smooth, near-optimal collision-free 3D Cartesian trajectories from a single artificial demonstration. The demonstration is encoded as a Dynamic Movement Primitive (DMP) and iteratively reshaped using policy-based reinforcement learning to create a diverse trajectory dataset for varying obstacle configurations. This dataset is used to train a neural network that takes as inputs the task parameters describing the obstacle dimensions and location, derived automatically from a point cloud, and outputs the DMP parameters that generate the trajectory. The approach is validated in simulation and real-robot experiments, outperforming a RRT-Connect baseline in terms of computation and execution time, as well as trajectory length, while supporting multi-modal trajectory generation for different obstacle geometries and end-effector dimensions. Videos and the implementation code are available at https://github.com/DominikUrbaniak/obst-avoid-dmp-pi2.
>
---
#### [new 007] CDE: Concept-Driven Exploration for Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于视觉强化学习任务，旨在解决智能体在视觉控制任务中探索效率低的问题。通过引入概念驱动探索方法，利用预训练视觉-语言模型生成对象中心的视觉概念，以辅助策略学习，提升探索效率，并减少部署时对外部模型的依赖。**

- **链接: [http://arxiv.org/pdf/2510.08851v1](http://arxiv.org/pdf/2510.08851v1)**

> **作者:** Le Mao; Andrew H. Liu; Renos Zabounidis; Zachary Kingston; Joseph Campbell
>
> **备注:** Preprint
>
> **摘要:** Intelligent exploration remains a critical challenge in reinforcement learning (RL), especially in visual control tasks. Unlike low-dimensional state-based RL, visual RL must extract task-relevant structure from raw pixels, making exploration inefficient. We propose Concept-Driven Exploration (CDE), which leverages a pre-trained vision-language model (VLM) to generate object-centric visual concepts from textual task descriptions as weak, potentially noisy supervisory signals. Rather than directly conditioning on these noisy signals, CDE trains a policy to reconstruct the concepts via an auxiliary objective, using reconstruction accuracy as an intrinsic reward to guide exploration toward task-relevant objects. Because the policy internalizes these concepts, VLM queries are only needed during training, reducing dependence on external models during deployment. Across five challenging simulated visual manipulation tasks, CDE achieves efficient, targeted exploration and remains robust to noisy VLM predictions. Finally, we demonstrate real-world transfer by deploying CDE on a Franka Research 3 arm, attaining an 80\% success rate in a real-world manipulation task.
>
---
#### [new 008] Placeit! A Framework for Learning Robot Object Placement Skills
- **分类: cs.RO; cs.LG**

- **简介: 论文提出Placeit!框架，用于学习机器人物体放置技能，属机器人操作任务。解决生成多样有效放置姿态的数据瓶颈问题。采用进化计算与质量-多样性优化，自动生成高质量放置位姿，实验证明其优于现有方法，并实现90%真实场景抓取放置成功率。**

- **链接: [http://arxiv.org/pdf/2510.09267v1](http://arxiv.org/pdf/2510.09267v1)**

> **作者:** Amina Ferrad; Johann Huber; François Hélénon; Julien Gleyze; Mahdi Khoramshahi; Stéphane Doncieux
>
> **备注:** 8 pages, 8 figures. Draft version
>
> **摘要:** Robotics research has made significant strides in learning, yet mastering basic skills like object placement remains a fundamental challenge. A key bottleneck is the acquisition of large-scale, high-quality data, which is often a manual and laborious process. Inspired by Graspit!, a foundational work that used simulation to automatically generate dexterous grasp poses, we introduce Placeit!, an evolutionary-computation framework for generating valid placement positions for rigid objects. Placeit! is highly versatile, supporting tasks from placing objects on tables to stacking and inserting them. Our experiments show that by leveraging quality-diversity optimization, Placeit! significantly outperforms state-of-the-art methods across all scenarios for generating diverse valid poses. A pick&place pipeline built on our framework achieved a 90% success rate over 120 real-world deployments. This work positions Placeit! as a powerful tool for open-environment pick-and-place tasks and as a valuable engine for generating the data needed to train simulation-based foundation models in robotics.
>
---
#### [new 009] Point and Go: Intuitive Reference Frame Reallocation in Mode Switching for Assistive Robotics
- **分类: cs.RO**

- **简介: 论文提出“Point and Go”模式切换方法，用于辅助机器人操作。任务是提升轮椅安装机械臂的操控直观性和效率。解决了传统笛卡尔空间控制中参考系不直观、运动受限等问题。工作包括设计新参考系、结合指向动作与平移控制，实现更自然的人机交互。实验验证其在效率与用户体验上的优势。**

- **链接: [http://arxiv.org/pdf/2510.08753v1](http://arxiv.org/pdf/2510.08753v1)**

> **作者:** A. Wang; C. Jiang; M. Przystupa; J. Valentine; M. Jagersand
>
> **备注:** 7 Pages, 5 figures
>
> **摘要:** Operating high degree of freedom robots can be difficult for users of wheelchair mounted robotic manipulators. Mode switching in Cartesian space has several drawbacks such as unintuitive control reference frames, separate translation and orientation control, and limited movement capabilities that hinder performance. We propose Point and Go mode switching, which reallocates the Cartesian mode switching reference frames into a more intuitive action space comprised of new translation and rotation modes. We use a novel sweeping motion to point the gripper, which defines the new translation axis along the robot base frame's horizontal plane. This creates an intuitive `point and go' translation mode that allows the user to easily perform complex, human-like movements without switching control modes. The system's rotation mode combines position control with a refined end-effector oriented frame that provides precise and consistent robot actions in various end-effector poses. We verified its effectiveness through initial experiments, followed by a three-task user study that compared our method to Cartesian mode switching and a state of the art learning method. Results show that Point and Go mode switching reduced completion times by 31\%, pauses by 41\%, and mode switches by 33\%, while receiving significantly favorable responses in user surveys.
>
---
#### [new 010] Adaptive Science Operations in Deep Space Missions Using Offline Belief State Planning
- **分类: cs.RO; cs.AI**

- **简介: 论文研究深空任务中因通信延迟和环境不确定性导致的自主科学操作问题，提出一种基于离线信念状态规划的POMDP框架，结合贝叶斯网络处理高维不确定数据，优化科学仪器操作策略，应用于恩克拉多斯轨道着陆器的生命探测任务，有效降低样本识别错误率。**

- **链接: [http://arxiv.org/pdf/2510.08812v1](http://arxiv.org/pdf/2510.08812v1)**

> **作者:** Grace Ra Kim; Hailey Warner; Duncan Eddy; Evan Astle; Zachary Booth; Edward Balaban; Mykel J. Kochenderfer
>
> **备注:** 7 pages, 4 tables, 5 figures, accepted in IEEE ISPARO 2026
>
> **摘要:** Deep space missions face extreme communication delays and environmental uncertainty that prevent real-time ground operations. To support autonomous science operations in communication-constrained environments, we present a partially observable Markov decision process (POMDP) framework that adaptively sequences spacecraft science instruments. We integrate a Bayesian network into the POMDP observation space to manage the high-dimensional and uncertain measurements typical of astrobiology missions. This network compactly encodes dependencies among measurements and improves the interpretability and computational tractability of science data. Instrument operation policies are computed offline, allowing resource-aware plans to be generated and thoroughly validated prior to launch. We use the Enceladus Orbilander's proposed Life Detection Suite (LDS) as a case study, demonstrating how Bayesian network structure and reward shaping influence system performance. We compare our method against the mission's baseline Concept of Operations (ConOps), evaluating both misclassification rates and performance in off-nominal sample accumulation scenarios. Our approach reduces sample identification errors by nearly 40%
>
---
#### [new 011] Failure Prediction at Runtime for Generative Robot Policies
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人安全控制任务，旨在解决生成式模仿学习策略在运行时因环境变化或动作误差累积导致的不可预测失败问题。论文提出FIPER框架，通过检测观测数据的分布偏移和动作生成的不确定性，实现无需失败数据的在线失败预测，提升机器人部署的安全性与可解释性。**

- **链接: [http://arxiv.org/pdf/2510.09459v1](http://arxiv.org/pdf/2510.09459v1)**

> **作者:** Ralf Römer; Adrian Kobras; Luca Worbis; Angela P. Schoellig
>
> **备注:** Accepted to NeurIPS 2025
>
> **摘要:** Imitation learning (IL) with generative models, such as diffusion and flow matching, has enabled robots to perform complex, long-horizon tasks. However, distribution shifts from unseen environments or compounding action errors can still cause unpredictable and unsafe behavior, leading to task failure. Early failure prediction during runtime is therefore essential for deploying robots in human-centered and safety-critical environments. We propose FIPER, a general framework for Failure Prediction at Runtime for generative IL policies that does not require failure data. FIPER identifies two key indicators of impending failure: (i) out-of-distribution (OOD) observations detected via random network distillation in the policy's embedding space, and (ii) high uncertainty in generated actions measured by a novel action-chunk entropy score. Both failure prediction scores are calibrated using a small set of successful rollouts via conformal prediction. A failure alarm is triggered when both indicators, aggregated over short time windows, exceed their thresholds. We evaluate FIPER across five simulation and real-world environments involving diverse failure modes. Our results demonstrate that FIPER better distinguishes actual failures from benign OOD situations and predicts failures more accurately and earlier than existing methods. We thus consider this work an important step towards more interpretable and safer generative robot policies. Code, data and videos are available at https://tum-lsy.github.io/fiper_website.
>
---
#### [new 012] Adaptive Motion Planning via Contact-Based Intent Inference for Human-Robot Collaboration
- **分类: cs.RO**

- **简介: 该论文属于人机协作任务，旨在解决机器人在共享空间中准确理解人类意图并自适应调整运动的问题。论文提出了一种基于接触的意图推断框架，通过优化力估计、接触检测与在线运动规划，实现机器人对人类意图的理解与实时响应。**

- **链接: [http://arxiv.org/pdf/2510.08811v1](http://arxiv.org/pdf/2510.08811v1)**

> **作者:** Jiurun Song; Xiao Liang; Minghui Zheng
>
> **摘要:** Human-robot collaboration (HRC) requires robots to adapt their motions to human intent to ensure safe and efficient cooperation in shared spaces. Although large language models (LLMs) provide high-level reasoning for inferring human intent, their application to reliable motion planning in HRC remains challenging. Physical human-robot interaction (pHRI) is intuitive but often relies on continuous kinesthetic guidance, which imposes burdens on operators. To address these challenges, a contact-informed adaptive motion-planning framework is introduced to infer human intent directly from physical contact and employ the inferred intent for online motion correction in HRC. First, an optimization-based force estimation method is proposed to infer human-intended contact forces and locations from joint torque measurements and a robot dynamics model, thereby reducing cost and installation complexity while enabling whole-body sensitivity. Then, a torque-based contact detection mechanism with link-level localization is introduced to reduce the optimization search space and to enable real-time estimation. Subsequently, a contact-informed adaptive motion planner is developed to infer human intent from contacts and to replan robot motion online, while maintaining smoothness and adapting to human corrections. Finally, experiments on a 7-DOF manipulator are conducted to demonstrate the accuracy of the proposed force estimation method and the effectiveness of the contact-informed adaptive motion planner under perception uncertainty in HRC.
>
---
#### [new 013] Direct Data-Driven Predictive Control for a Three-dimensional Cable-Driven Soft Robotic Arm
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人控制任务，旨在解决3D软体机器人精确动态控制难题。作者设计并制作了一款3D缆索驱动软体机械臂，并应用数据驱动的DeePC控制方法，结合SVD降维技术，实现了比传统模型控制更优的控制精度与适应性。**

- **链接: [http://arxiv.org/pdf/2510.08953v1](http://arxiv.org/pdf/2510.08953v1)**

> **作者:** Cheng Ouyang; Moeen Ul Islam; Dong Chen; Kaixiang Zhang; Zhaojian Li; Xiaobo Tan
>
> **摘要:** Soft robots offer significant advantages in safety and adaptability, yet achieving precise and dynamic control remains a major challenge due to their inherently complex and nonlinear dynamics. Recently, Data-enabled Predictive Control (DeePC) has emerged as a promising model-free approach that bypasses explicit system identification by directly leveraging input-output data. While DeePC has shown success in other domains, its application to soft robots remains underexplored, particularly for three-dimensional (3D) soft robotic systems. This paper addresses this gap by developing and experimentally validating an effective DeePC framework on a 3D, cable-driven soft arm. Specifically, we design and fabricate a soft robotic arm with a thick tubing backbone for stability, a dense silicone body with large cavities for strength and flexibility, and rigid endcaps for secure termination. Using this platform, we implement DeePC with singular value decomposition (SVD)-based dimension reduction for two key control tasks: fixed-point regulation and trajectory tracking in 3D space. Comparative experiments with a baseline model-based controller demonstrate DeePC's superior accuracy, robustness, and adaptability, highlighting its potential as a practical solution for dynamic control of soft robots.
>
---
#### [new 014] Flow-Opt: Scalable Centralized Multi-Robot Trajectory Optimization with Flow Matching and Differentiable Optimization
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于多机器人轨迹优化任务，旨在解决传统集中式优化计算复杂度高、难以扩展的问题。作者提出Flow-Opt方法，结合流匹配生成模型与可微分安全过滤器，实现高效、平滑且多样化的轨迹规划，显著提升了计算效率与轨迹质量。**

- **链接: [http://arxiv.org/pdf/2510.09204v1](http://arxiv.org/pdf/2510.09204v1)**

> **作者:** Simon Idoko; Arun Kumar Singh
>
> **摘要:** Centralized trajectory optimization in the joint space of multiple robots allows access to a larger feasible space that can result in smoother trajectories, especially while planning in tight spaces. Unfortunately, it is often computationally intractable beyond a very small swarm size. In this paper, we propose Flow-Opt, a learning-based approach towards improving the computational tractability of centralized multi-robot trajectory optimization. Specifically, we reduce the problem to first learning a generative model to sample different candidate trajectories and then using a learned Safety-Filter(SF) to ensure fast inference-time constraint satisfaction. We propose a flow-matching model with a diffusion transformer (DiT) augmented with permutation invariant robot position and map encoders as the generative model. We develop a custom solver for our SF and equip it with a neural network that predicts context-specific initialization. The initialization network is trained in a self-supervised manner, taking advantage of the differentiability of the SF solver. We advance the state-of-the-art in the following respects. First, we show that we can generate trajectories of tens of robots in cluttered environments in a few tens of milliseconds. This is several times faster than existing centralized optimization approaches. Moreover, our approach also generates smoother trajectories orders of magnitude faster than competing baselines based on diffusion models. Second, each component of our approach can be batched, allowing us to solve a few tens of problem instances in a fraction of a second. We believe this is a first such result; no existing approach provides such capabilities. Finally, our approach can generate a diverse set of trajectories between a given set of start and goal locations, which can capture different collision-avoidance behaviors.
>
---
#### [new 015] When a Robot is More Capable than a Human: Learning from Constrained Demonstrators
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文研究机器人如何从受限示范中学习更优策略，解决因示范者受限导致策略次优的问题。通过仅使用状态奖励信号和时间插值自标注未知状态，使机器人自主探索更高效路径，提升学习效率与任务完成速度。**

- **链接: [http://arxiv.org/pdf/2510.09096v1](http://arxiv.org/pdf/2510.09096v1)**

> **作者:** Xinhu Li; Ayush Jain; Zhaojing Yang; Yigit Korkmaz; Erdem Bıyık
>
> **摘要:** Learning from demonstrations enables experts to teach robots complex tasks using interfaces such as kinesthetic teaching, joystick control, and sim-to-real transfer. However, these interfaces often constrain the expert's ability to demonstrate optimal behavior due to indirect control, setup restrictions, and hardware safety. For example, a joystick can move a robotic arm only in a 2D plane, even though the robot operates in a higher-dimensional space. As a result, the demonstrations collected by constrained experts lead to suboptimal performance of the learned policies. This raises a key question: Can a robot learn a better policy than the one demonstrated by a constrained expert? We address this by allowing the agent to go beyond direct imitation of expert actions and explore shorter and more efficient trajectories. We use the demonstrations to infer a state-only reward signal that measures task progress, and self-label reward for unknown states using temporal interpolation. Our approach outperforms common imitation learning in both sample efficiency and task completion time. On a real WidowX robotic arm, it completes the task in 12 seconds, 10x faster than behavioral cloning, as shown in real-robot videos on https://sites.google.com/view/constrainedexpert .
>
---
#### [new 016] HANDO: Hierarchical Autonomous Navigation and Dexterous Omni-loco-manipulation
- **分类: cs.RO**

- **简介: 该论文提出HANDO框架，用于腿臂机器人的人机交互任务。解决在非结构化环境中机器人自主导航与精准操作问题。分为两层：高层自主探索导航到指定目标，底层全身协调操作完成精细任务。**

- **链接: [http://arxiv.org/pdf/2510.09221v1](http://arxiv.org/pdf/2510.09221v1)**

> **作者:** Jingyuan Sun; Chaoran Wang; Mingyu Zhang; Cui Miao; Hongyu Ji; Zihan Qu; Han Sun; Bing Wang; Qingyi Si
>
> **备注:** 4 pages, 2 figures, this paper has been accepted for the workshop Perception and Planning for Mobile Manipulation in Changing Environments (PM2CE) at IROS 2025
>
> **摘要:** Seamless loco-manipulation in unstructured environments requires robots to leverage autonomous exploration alongside whole-body control for physical interaction. In this work, we introduce HANDO (Hierarchical Autonomous Navigation and Dexterous Omni-loco-manipulation), a two-layer framework designed for legged robots equipped with manipulators to perform human-centered mobile manipulation tasks. The first layer utilizes a goal-conditioned autonomous exploration policy to guide the robot to semantically specified targets, such as a black office chair in a dynamic environment. The second layer employs a unified whole-body loco-manipulation policy to coordinate the arm and legs for precise interaction tasks-for example, handing a drink to a person seated on the chair. We have conducted an initial deployment of the navigation module, and will continue to pursue finer-grained deployment of whole-body loco-manipulation.
>
---
#### [new 017] Robust Visual Teach-and-Repeat Navigation with Flexible Topo-metric Graph Map Representation
- **分类: cs.RO**

- **简介: 该论文属于移动机器人导航任务，旨在解决环境变化和动态物体影响下的鲁棒视觉教学与重复导航问题。论文提出了一种灵活的拓扑度量图地图表示方法，结合关键帧聚类与地图匹配策略，并设计了无需地图的局部轨迹控制算法，以提升导航系统的鲁棒性与有效性。**

- **链接: [http://arxiv.org/pdf/2510.09089v1](http://arxiv.org/pdf/2510.09089v1)**

> **作者:** Jikai Wang; Yunqi Cheng; Kezhi Wang; Zonghai Chen
>
> **摘要:** Visual Teach-and-Repeat Navigation is a direct solution for mobile robot to be deployed in unknown environments. However, robust trajectory repeat navigation still remains challenged due to environmental changing and dynamic objects. In this paper, we propose a novel visual teach-and-repeat navigation system, which consists of a flexible map representation, robust map matching and a map-less local navigation module. During the teaching process, the recorded keyframes are formulated as a topo-metric graph and each node can be further extended to save new observations. Such representation also alleviates the requirement of globally consistent mapping. To enhance the place recognition performance during repeating process, instead of using frame-to-frame matching, we firstly implement keyframe clustering to aggregate similar connected keyframes into local map and perform place recognition based on visual frame-tolocal map matching strategy. To promote the local goal persistent tracking performance, a long-term goal management algorithm is constructed, which can avoid the robot getting lost due to environmental changes or obstacle occlusion. To achieve the goal without map, a local trajectory-control candidate optimization algorithm is proposed. Extensively experiments are conducted on our mobile platform. The results demonstrate that our system is superior to the baselines in terms of robustness and effectiveness.
>
---
#### [new 018] ConPoSe: LLM-Guided Contact Point Selection for Scalable Cooperative Object Pushing
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多机器人协同操作任务，旨在解决复杂环境中多机器人合作推动物体时的接触点选择问题。传统方法因解空间过大而扩展性差，论文提出ConPoSe方法，结合大语言模型与局部搜索，实现高效接触点选择，提升了扩展性与适应性。**

- **链接: [http://arxiv.org/pdf/2510.08705v1](http://arxiv.org/pdf/2510.08705v1)**

> **作者:** Noah Steinkrüger; Nisarga Nilavadi; Wolfram Burgard; Tanja Katharina Kaiser
>
> **摘要:** Object transportation in cluttered environments is a fundamental task in various domains, including domestic service and warehouse logistics. In cooperative object transport, multiple robots must coordinate to move objects that are too large for a single robot. One transport strategy is pushing, which only requires simple robots. However, careful selection of robot-object contact points is necessary to push the object along a preplanned path. Although this selection can be solved analytically, the solution space grows combinatorially with the number of robots and object size, limiting scalability. Inspired by how humans rely on common-sense reasoning for cooperative transport, we propose combining the reasoning capabilities of Large Language Models with local search to select suitable contact points. Our LLM-guided local search method for contact point selection, ConPoSe, successfully selects contact points for a variety of shapes, including cuboids, cylinders, and T-shapes. We demonstrate that ConPoSe scales better with the number of robots and object size than the analytical approach, and also outperforms pure LLM-based selection.
>
---
#### [new 019] iMoWM: Taming Interactive Multi-Modal World Model for Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作中的世界建模任务，旨在解决现有2D视频模型缺乏几何空间推理能力的问题。作者提出iMoWM，一种基于多模态输入的交互式世界模型，结合MMTokenizer实现高效计算，支持未来视觉预测、模型强化学习和模仿学习。**

- **链接: [http://arxiv.org/pdf/2510.09036v1](http://arxiv.org/pdf/2510.09036v1)**

> **作者:** Chuanrui Zhang; Zhengxian Wu; Guanxing Lu; Yansong Tang; Ziwei Wang
>
> **摘要:** Learned world models hold significant potential for robotic manipulation, as they can serve as simulator for real-world interactions. While extensive progress has been made in 2D video-based world models, these approaches often lack geometric and spatial reasoning, which is essential for capturing the physical structure of the 3D world. To address this limitation, we introduce iMoWM, a novel interactive world model designed to generate color images, depth maps, and robot arm masks in an autoregressive manner conditioned on actions. To overcome the high computational cost associated with three-dimensional information, we propose MMTokenizer, which unifies multi-modal inputs into a compact token representation. This design enables iMoWM to leverage large-scale pretrained VideoGPT models while maintaining high efficiency and incorporating richer physical information. With its multi-modal representation, iMoWM not only improves the visual quality of future predictions but also serves as an effective simulator for model-based reinforcement learning (MBRL) and facilitates real-world imitation learning. Extensive experiments demonstrate the superiority of iMoWM across these tasks, showcasing the advantages of multi-modal world modeling for robotic manipulation. Homepage: https://xingyoujun.github.io/imowm/
>
---
#### [new 020] Geometry-aware Policy Imitation
- **分类: cs.RO**

- **简介: 该论文属于机器人模仿学习任务，旨在解决从专家示范中高效学习控制策略的问题。论文提出Geometry-aware Policy Imitation（GPI），将示范视为几何曲线，构建距离场生成控制策略，实现高效、鲁棒的机器人行为引导。**

- **链接: [http://arxiv.org/pdf/2510.08787v1](http://arxiv.org/pdf/2510.08787v1)**

> **作者:** Yiming Li; Nael Darwiche; Amirreza Razmjoo; Sichao Liu; Yilun Du; Auke Ijspeert; Sylvain Calinon
>
> **备注:** 21 pages, 13 figures. In submission
>
> **摘要:** We propose a Geometry-aware Policy Imitation (GPI) approach that rethinks imitation learning by treating demonstrations as geometric curves rather than collections of state-action samples. From these curves, GPI derives distance fields that give rise to two complementary control primitives: a progression flow that advances along expert trajectories and an attraction flow that corrects deviations. Their combination defines a controllable, non-parametric vector field that directly guides robot behavior. This formulation decouples metric learning from policy synthesis, enabling modular adaptation across low-dimensional robot states and high-dimensional perceptual inputs. GPI naturally supports multimodality by preserving distinct demonstrations as separate models and allows efficient composition of new demonstrations through simple additions to the distance field. We evaluate GPI in simulation and on real robots across diverse tasks. Experiments show that GPI achieves higher success rates than diffusion-based policies while running 20 times faster, requiring less memory, and remaining robust to perturbations. These results establish GPI as an efficient, interpretable, and scalable alternative to generative approaches for robotic imitation learning. Project website: https://yimingli1998.github.io/projects/GPI/
>
---
#### [new 021] Decentralized Multi-Robot Relative Navigation in Unknown, Structurally Constrained Environments under Limited Communication
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于多机器人导航任务，旨在解决未知、结构受限、无GPS且通信受限环境中的导航问题。论文提出了一种去中心化的分层相对导航框架，结合拓扑地图交换与局部轨迹规划，兼顾全局策略与局部避障，提升了导航成功率与效率。**

- **链接: [http://arxiv.org/pdf/2510.09188v1](http://arxiv.org/pdf/2510.09188v1)**

> **作者:** Zihao Mao; Yunheng Wang; Yunting Ji; Yi Yang; Wenjie Song
>
> **摘要:** Multi-robot navigation in unknown, structurally constrained, and GPS-denied environments presents a fundamental trade-off between global strategic foresight and local tactical agility, particularly under limited communication. Centralized methods achieve global optimality but suffer from high communication overhead, while distributed methods are efficient but lack the broader awareness to avoid deadlocks and topological traps. To address this, we propose a fully decentralized, hierarchical relative navigation framework that achieves both strategic foresight and tactical agility without a unified coordinate system. At the strategic layer, robots build and exchange lightweight topological maps upon opportunistic encounters. This process fosters an emergent global awareness, enabling the planning of efficient, trap-avoiding routes at an abstract level. This high-level plan then inspires the tactical layer, which operates on local metric information. Here, a sampling-based escape point strategy resolves dense spatio-temporal conflicts by generating dynamically feasible trajectories in real time, concurrently satisfying tight environmental and kinodynamic constraints. Extensive simulations and real-world experiments demonstrate that our system significantly outperforms in success rate and efficiency, especially in communication-limited environments with complex topological structures.
>
---
#### [new 022] Autonomous Soft Robotic Guidewire Navigation via Imitation Learning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于医疗机器人任务，旨在解决血管内手术中软体导丝导航的自动化问题。通过模仿学习与Transformer框架，实现导丝自主导航至动脉瘤位置，提升了精度与安全性。**

- **链接: [http://arxiv.org/pdf/2510.09497v1](http://arxiv.org/pdf/2510.09497v1)**

> **作者:** Noah Barnes; Ji Woong Kim; Lingyun Di; Hannah Qu; Anuruddha Bhattacharjee; Miroslaw Janowski; Dheeraj Gandhi; Bailey Felix; Shaopeng Jiang; Olivia Young; Mark Fuge; Ryan D. Sochol; Jeremy D. Brown; Axel Krieger
>
> **摘要:** In endovascular surgery, endovascular interventionists push a thin tube called a catheter, guided by a thin wire to a treatment site inside the patient's blood vessels to treat various conditions such as blood clots, aneurysms, and malformations. Guidewires with robotic tips can enhance maneuverability, but they present challenges in modeling and control. Automation of soft robotic guidewire navigation has the potential to overcome these challenges, increasing the precision and safety of endovascular navigation. In other surgical domains, end-to-end imitation learning has shown promising results. Thus, we develop a transformer-based imitation learning framework with goal conditioning, relative action outputs, and automatic contrast dye injections to enable generalizable soft robotic guidewire navigation in an aneurysm targeting task. We train the model on 36 different modular bifurcated geometries, generating 647 total demonstrations under simulated fluoroscopy, and evaluate it on three previously unseen vascular geometries. The model can autonomously drive the tip of the robot to the aneurysm location with a success rate of 83% on the unseen geometries, outperforming several baselines. In addition, we present ablation and baseline studies to evaluate the effectiveness of each design and data collection choice. Project website: https://softrobotnavigation.github.io/
>
---
#### [new 023] Online IMU-odometer Calibration using GNSS Measurements for Autonomous Ground Vehicle Localization
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶定位任务，旨在解决IMU与里程计在线联合标定问题。现有方法依赖GNSS定位或未解算模糊度，可观测性不明确。论文提出紧耦合标定方法，融合IMU、里程计和原始GNSS数据，在因子图框架下实现在线标定与定位，提升精度并分析参数可观测性。实验表明其定位误差显著降低。**

- **链接: [http://arxiv.org/pdf/2510.08880v1](http://arxiv.org/pdf/2510.08880v1)**

> **作者:** Baoshan Song; Xiao Xia; Penggao Yan; Yihan Zhong; Weisong Wen; Li-Ta Hsu
>
> **备注:** Submitted to IEEE Transactions on Intelligent Transportation Systems
>
> **摘要:** Accurate calibration of intrinsic (odometer scaling factors) and extrinsic parameters (IMU-odometer translation and rotation) is essential for autonomous ground vehicle localization. Existing GNSS-aided approaches often rely on positioning results or raw measurements without ambiguity resolution, and their observability properties remain underexplored. This paper proposes a tightly coupled online calibration method that fuses IMU, odometer, and raw GNSS measurements (pseudo-range, carrier-phase, and Doppler) within an extendable factor graph optimization (FGO) framework, incorporating outlier mitigation and ambiguity resolution. Observability analysis reveals that two horizontal translation and three rotation parameters are observable under general motion, while vertical translation remains unobservable. Simulation and real-world experiments demonstrate superior calibration and localization performance over state-of-the-art loosely coupled methods. Specifically, the IMU-odometer positioning using our calibrated parameters achieves the absolute maximum error of 17.75 m while the one of LC method is 61.51 m, achieving up to 71.14 percent improvement. To foster further research, we also release the first open-source dataset that combines IMU, 2D odometer, and raw GNSS measurements from both rover and base stations.
>
---
#### [new 024] Dynamic Quadrupedal Legged and Aerial Locomotion via Structure Repurposing
- **分类: cs.RO**

- **简介: 该论文属于机器人设计任务，旨在解决多模态地面-空中机器人在不同运动模式下结构冲突的问题。作者设计了可重构结构的四足机器人Husky v.2，通过姿态操控和推力矢量实现腿式运动与飞行的动态切换，并展示了其硬件设计及初步实验结果。**

- **链接: [http://arxiv.org/pdf/2510.09526v1](http://arxiv.org/pdf/2510.09526v1)**

> **作者:** Chenghao Wang; Kaushik Venkatesh Krishnamurthy; Shreyansh Pitroda; Adarsh Salagame; Ioannis Mandralis; Eric Sihite; Alireza Ramezani; Morteza Gharib
>
> **摘要:** Multi-modal ground-aerial robots have been extensively studied, with a significant challenge lying in the integration of conflicting requirements across different modes of operation. The Husky robot family, developed at Northeastern University, and specifically the Husky v.2 discussed in this study, addresses this challenge by incorporating posture manipulation and thrust vectoring into multi-modal locomotion through structure repurposing. This quadrupedal robot features leg structures that can be repurposed for dynamic legged locomotion and flight. In this paper, we present the hardware design of the robot and report primary results on dynamic quadrupedal legged locomotion and hovering.
>
---
#### [new 025] Training Models to Detect Successive Robot Errors from Human Reactions
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文属于人机交互任务，旨在解决机器人如何通过人类反应检测连续错误的问题。研究提取人类语音和非语言行为特征，训练机器学习模型识别机器人错误阶段，提升了错误检测与交互理解效果。**

- **链接: [http://arxiv.org/pdf/2510.09080v1](http://arxiv.org/pdf/2510.09080v1)**

> **作者:** Shannon Liu; Maria Teresa Parreira; Wendy Ju
>
> **备注:** Accepted to NERC '25
>
> **摘要:** As robots become more integrated into society, detecting robot errors is essential for effective human-robot interaction (HRI). When a robot fails repeatedly, how can it know when to change its behavior? Humans naturally respond to robot errors through verbal and nonverbal cues that intensify over successive failures-from confusion and subtle speech changes to visible frustration and impatience. While prior work shows that human reactions can indicate robot failures, few studies examine how these evolving responses reveal successive failures. This research uses machine learning to recognize stages of robot failure from human reactions. In a study with 26 participants interacting with a robot that made repeated conversational errors, behavioral features were extracted from video data to train models for individual users. The best model achieved 93.5% accuracy for detecting errors and 84.1% for classifying successive failures. Modeling the progression of human reactions enhances error detection and understanding of repeated interaction breakdowns in HRI.
>
---
#### [new 026] Guiding Energy-Efficient Locomotion through Impact Mitigation Rewards
- **分类: cs.RO**

- **简介: 该论文旨在提升机器人运动的能效。通过结合对抗运动先验（AMP）和强化学习（RL），并引入基于物理的冲击缓解因子（IMF）作为奖励项，使机器人不仅能模仿动物的运动模式，还能学习其被动动力学特性，从而降低运输成本（CoT），提升能量效率。**

- **链接: [http://arxiv.org/pdf/2510.09543v1](http://arxiv.org/pdf/2510.09543v1)**

> **作者:** Chenghao Wang; Arjun Viswanathan; Eric Sihite; Alireza Ramezani
>
> **摘要:** Animals achieve energy-efficient locomotion by their implicit passive dynamics, a marvel that has captivated roboticists for decades.Recently, methods incorporated Adversarial Motion Prior (AMP) and Reinforcement learning (RL) shows promising progress to replicate Animals' naturalistic motion. However, such imitation learning approaches predominantly capture explicit kinematic patterns, so-called gaits, while overlooking the implicit passive dynamics. This work bridges this gap by incorporating a reward term guided by Impact Mitigation Factor (IMF), a physics-informed metric that quantifies a robot's ability to passively mitigate impacts. By integrating IMF with AMP, our approach enables RL policies to learn both explicit motion trajectories from animal reference motion and the implicit passive dynamic. We demonstrate energy efficiency improvements of up to 32%, as measured by the Cost of Transport (CoT), across both AMP and handcrafted reward structure.
>
---
#### [new 027] Model-Based Lookahead Reinforcement Learning for in-hand manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决灵巧的手中操作问题。通过结合无模型与基于模型的强化学习方法，提出一种混合框架，利用动态模型和价值函数提升策略性能。实验验证了其在不同物体属性下的操控效果与泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.08884v1](http://arxiv.org/pdf/2510.08884v1)**

> **作者:** Alexandre Lopes; Catarina Barata; Plinio Moreno
>
> **摘要:** In-Hand Manipulation, as many other dexterous tasks, remains a difficult challenge in robotics by combining complex dynamic systems with the capability to control and manoeuvre various objects using its actuators. This work presents the application of a previously developed hybrid Reinforcement Learning (RL) Framework to In-Hand Manipulation task, verifying that it is capable of improving the performance of the task. The model combines concepts of both Model-Free and Model-Based Reinforcement Learning, by guiding a trained policy with the help of a dynamic model and value-function through trajectory evaluation, as done in Model Predictive Control. This work evaluates the performance of the model by comparing it with the policy that will be guided. To fully explore this, various tests are performed using both fully-actuated and under-actuated simulated robotic hands to manipulate different objects for a given task. The performance of the model will also be tested for generalization tests, by changing the properties of the objects in which both the policy and dynamic model were trained, such as density and size, and additionally by guiding a trained policy in a certain object to perform the same task in a different one. The results of this work show that, given a policy with high average reward and an accurate dynamic model, the hybrid framework improves the performance of in-hand manipulation tasks for most test cases, even when the object properties are changed. However, this improvement comes at the expense of increasing the computational cost, due to the complexity of trajectory evaluation.
>
---
#### [new 028] FOGMACHINE -- Leveraging Discrete-Event Simulation and Scene Graphs for Modeling Hierarchical, Interconnected Environments under Partial Observations from Mobile Agents
- **分类: cs.RO**

- **简介: 论文提出FOGMACHINE框架，属于动态场景建模任务，旨在解决部分可观测环境下多智能体交互建模问题。结合离散事件仿真与动态场景图，模拟复杂环境中的对象动态与智能体感知，支持不确定性推理与行为预测，推动具身智能在城市场景中的研究。**

- **链接: [http://arxiv.org/pdf/2510.09483v1](http://arxiv.org/pdf/2510.09483v1)**

> **作者:** Lars Ohnemus; Nils Hantke; Max Weißer; Kai Furmans
>
> **备注:** submitted to the IEEE for possible publication; 8 pages, 3 figures, 1 table
>
> **摘要:** Dynamic Scene Graphs (DSGs) provide a structured representation of hierarchical, interconnected environments, but current approaches struggle to capture stochastic dynamics, partial observability, and multi-agent activity. These aspects are critical for embodied AI, where agents must act under uncertainty and delayed perception. We introduce FOGMACHINE , an open-source framework that fuses DSGs with discrete-event simulation to model object dynamics, agent observations, and interactions at scale. This setup enables the study of uncertainty propagation, planning under limited perception, and emergent multi-agent behavior. Experiments in urban scenarios illustrate realistic temporal and spatial patterns while revealing the challenges of belief estimation under sparse observations. By combining structured representations with efficient simulation, FOGMACHINE establishes an effective tool for benchmarking, model training, and advancing embodied AI in complex, uncertain environments.
>
---
#### [new 029] Whole Body Model Predictive Control for Spin-Aware Quadrupedal Table Tennis
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人控制任务，旨在解决四足机器人打乒乓球时对球速、旋转和轨迹的快速感知与响应问题。作者提出了一种结合高速感知、轨迹预测和模型预测控制的方法，实现了机器人对不同旋转球的有效击打与回球，并成功在真实场景中与人类进行对打实验。**

- **链接: [http://arxiv.org/pdf/2510.08754v1](http://arxiv.org/pdf/2510.08754v1)**

> **作者:** David Nguyen; Zulfiqar Zaidi; Kevin Karol; Jessica Hodgins; Zhaoming Xie
>
> **备注:** Submitted to appear in IEEE ICRA 2026
>
> **摘要:** Developing table tennis robots that mirror human speed, accuracy, and ability to predict and respond to the full range of ball spins remains a significant challenge for legged robots. To demonstrate these capabilities we present a system to play dynamic table tennis for quadrupedal robots that integrates high speed perception, trajectory prediction, and agile control. Our system uses external cameras for high-speed ball localization, physical models with learned residuals to infer spin and predict trajectories, and a novel model predictive control (MPC) formulation for agile full-body control. Notably, a continuous set of stroke strategies emerge automatically from different ball return objectives using this control paradigm. We demonstrate our system in the real world on a Spot quadruped, evaluate accuracy of each system component, and exhibit coordination through the system's ability to aim and return balls with varying spin types. As a further demonstration, the system is able to rally with human players.
>
---
#### [new 030] Trust Modeling and Estimation in Human-Autonomy Interactions
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文研究人机交互中的信任建模与估计问题，属于人机协同任务。针对现有模型无法反映信任对系统表现的非对称响应及通信间断性的问题，论文提出一种基于切换线性系统和事件触发采样的信任模型，并通过51名参与者的数据进行参数识别，实现对监督者信任状态的估计。**

- **链接: [http://arxiv.org/pdf/2510.09013v1](http://arxiv.org/pdf/2510.09013v1)**

> **作者:** Daniel A. Williams; Airlie Chapman; Daniel R. Little; Chris Manzie
>
> **备注:** 10 pages. 13 figures
>
> **摘要:** Advances in the control of autonomous systems have accompanied an expansion in the potential applications for autonomous robotic systems. The success of applications involving humans depends on the quality of interaction between the autonomous system and the human supervisor, which is particularly affected by the degree of trust that the supervisor places in the autonomous system. Absent from the literature are models of supervisor trust dynamics that can accommodate asymmetric responses to autonomous system performance and the intermittent nature of supervisor-autonomous system communication. This paper focuses on formulating an estimated model of supervisor trust that incorporates both of these features by employing a switched linear system structure with event-triggered sampling of the model input and output. Trust response data collected in a user study with 51 participants were then used identify parameters for a switched linear model-based observer of supervisor trust.
>
---
#### [new 031] Humanoid Everyday: A Comprehensive Robotic Dataset for Open-World Humanoid Manipulation
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人学习与数据集构建任务，旨在解决现有数据集任务单一、缺乏人机交互和移动能力的问题。论文构建了大规模多样化的人形机器人操作数据集 Humanoid Everyday，包含多模态感知数据与语言标注，并提供了云端评估平台，以推动通用人形机器人研究。**

- **链接: [http://arxiv.org/pdf/2510.08807v1](http://arxiv.org/pdf/2510.08807v1)**

> **作者:** Zhenyu Zhao; Hongyi Jing; Xiawei Liu; Jiageng Mao; Abha Jha; Hanwen Yang; Rong Xue; Sergey Zakharor; Vitor Guizilini; Yue Wang
>
> **摘要:** From loco-motion to dextrous manipulation, humanoid robots have made remarkable strides in demonstrating complex full-body capabilities. However, the majority of current robot learning datasets and benchmarks mainly focus on stationary robot arms, and the few existing humanoid datasets are either confined to fixed environments or limited in task diversity, often lacking human-humanoid interaction and lower-body locomotion. Moreover, there are a few standardized evaluation platforms for benchmarking learning-based policies on humanoid data. In this work, we present Humanoid Everyday, a large-scale and diverse humanoid manipulation dataset characterized by extensive task variety involving dextrous object manipulation, human-humanoid interaction, locomotion-integrated actions, and more. Leveraging a highly efficient human-supervised teleoperation pipeline, Humanoid Everyday aggregates high-quality multimodal sensory data, including RGB, depth, LiDAR, and tactile inputs, together with natural language annotations, comprising 10.3k trajectories and over 3 million frames of data across 260 tasks across 7 broad categories. In addition, we conduct an analysis of representative policy learning methods on our dataset, providing insights into their strengths and limitations across different task categories. For standardized evaluation, we introduce a cloud-based evaluation platform that allows researchers to seamlessly deploy their policies in our controlled setting and receive performance feedback. By releasing Humanoid Everyday along with our policy learning analysis and a standardized cloud-based evaluation platform, we intend to advance research in general-purpose humanoid manipulation and lay the groundwork for more capable and embodied robotic agents in real-world scenarios. Our dataset, data collection code, and cloud evaluation website are made publicly available on our project website.
>
---
#### [new 032] Differential Analysis of Pseudo Haptic Feedback: Novel Comparative Study of Visual and Auditory Cue Integration for Psychophysical Evaluation
- **分类: cs.HC; cs.GR; cs.NE; cs.RO; physics.med-ph**

- **简介: 该论文研究伪触觉反馈中视觉与听觉线索的整合效果，旨在探索低成本设备实现触觉感知的方法。通过设计实验测量用户在不同视听刺激下的手指力度变化，分析多感官融合对伪触觉的影响，验证了视听线索能有效诱发触觉体验，为康复工具等应用提供新思路。**

- **链接: [http://arxiv.org/pdf/2510.09570v1](http://arxiv.org/pdf/2510.09570v1)**

> **作者:** Nishant Gautam; Somya Sharma; Peter Corcoran; Kaspar Althoefer
>
> **备注:** 17 Pages, 9 Figures
>
> **摘要:** Pseudo-haptics exploit carefully crafted visual or auditory cues to trick the brain into "feeling" forces that are never physically applied, offering a low-cost alternative to traditional haptic hardware. Here, we present a comparative psychophysical study that quantifies how visual and auditory stimuli combine to evoke pseudo-haptic pressure sensations on a commodity tablet. Using a Unity-based Rollball game, participants (n = 4) guided a virtual ball across three textured terrains while their finger forces were captured in real time with a Robotous RFT40 force-torque sensor. Each terrain was paired with a distinct rolling-sound profile spanning 440 Hz - 4.7 kHz, 440 Hz - 13.1 kHz, or 440 Hz - 8.9 kHz; crevice collisions triggered additional "knocking" bursts to heighten realism. Average tactile forces increased systematically with cue intensity: 0.40 N, 0.79 N and 0.88 N for visual-only trials and 0.41 N, 0.81 N and 0.90 N for audio-only trials on Terrains 1-3, respectively. Higher audio frequencies and denser visual textures both elicited stronger muscle activation, and their combination further reduced the force needed to perceive surface changes, confirming multisensory integration. These results demonstrate that consumer-grade isometric devices can reliably induce and measure graded pseudo-haptic feedback without specialized actuators, opening a path toward affordable rehabilitation tools, training simulators and assistive interfaces.
>
---
#### [new 033] Detecting spills using thermal imaging, pretrained deep learning models, and a robotic platform
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于计算机视觉与机器人任务，旨在解决实时检测液体泄漏问题。作者利用预训练深度学习模型，结合RGB和热成像数据，在消费级硬件上实现了高效的泄漏分类。实验表明，使用热成像的轻量模型如VGG19和NasNetMobile在不同光照条件下具有更高的准确性和推理速度，适用于安全关键场景的部署。**

- **链接: [http://arxiv.org/pdf/2510.08770v1](http://arxiv.org/pdf/2510.08770v1)**

> **作者:** Gregory Yeghiyan; Jurius Azar; Devson Butani; Chan-Jin Chung
>
> **备注:** 6 pages
>
> **摘要:** This paper presents a real-time spill detection system that utilizes pretrained deep learning models with RGB and thermal imaging to classify spill vs. no-spill scenarios across varied environments. Using a balanced binary dataset (4,000 images), our experiments demonstrate the advantages of thermal imaging in inference speed, accuracy, and model size. We achieve up to 100% accuracy using lightweight models like VGG19 and NasNetMobile, with thermal models performing faster and more robustly across different lighting conditions. Our system runs on consumer-grade hardware (RTX 4080) and achieves inference times as low as 44 ms with model sizes under 350 MB, highlighting its deployability in safety-critical contexts. Results from experiments with a real robot and test datasets indicate that a VGG19 model trained on thermal imaging performs best.
>
---
#### [new 034] Parametrized Topological Complexity for a Multi-Robot System with Variable Tasks
- **分类: math.AT; cs.RO; 55M30, 55R80**

- **简介: 该论文研究多机器人系统在未知障碍环境中的路径规划问题，属于拓扑复杂度理论任务。它旨在解决不同任务需求下，如何最小化运动规划算法的不稳定性。论文通过构建合适的纤维化，确定了广义参数拓扑复杂度，并分析了奇偶维空间中的上同调计算及算法构造。**

- **链接: [http://arxiv.org/pdf/2510.09323v1](http://arxiv.org/pdf/2510.09323v1)**

> **作者:** Gopal Chandra Dutta; Amit Kumar Paul; Subhankar Sau
>
> **备注:** 25 pages. All comments are welcome
>
> **摘要:** We study a generalized motion planning problem involving multiple autonomous robots navigating in a $d$-dimensional Euclidean space in the presence of a set of obstacles whose positions are unknown a priori. Each robot is required to visit sequentially a prescribed set of target states, with the number of targets varying between robots. This heterogeneous setting generalizes the framework considered in the prior works on sequential parametrized topological complexity by Farber and the second author of this article. To determine the topological complexity of our problem, we formulate it mathematically by constructing an appropriate fibration. Our main contribution is the determination of this invariant in the generalized setting, which captures the minimal algorithmic instability required for designing collision-free motion planning algorithms under parameter-dependent constraints. We provide a detailed analysis for both odd and even-dimensional ambient spaces, including the essential cohomological computations and explicit constructions of corresponding motion planning algorithms.
>
---
#### [new 035] Unified World Models: Memory-Augmented Planning and Foresight for Visual Navigation
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 论文提出UniWM，一种统一的具身导航模型，通过融合视觉世界建模与规划，解决现有方法中状态-动作不匹配和适应性差的问题，提升导航成功率与轨迹准确性，并展现良好泛化能力。**

- **链接: [http://arxiv.org/pdf/2510.08713v1](http://arxiv.org/pdf/2510.08713v1)**

> **作者:** Yifei Dong; Fengyi Wu; Guangyu Chen; Zhi-Qi Cheng; Qiyu Hu; Yuxuan Zhou; Jingdong Sun; Jun-Yan He; Qi Dai; Alexander G Hauptmann
>
> **备注:** 18 pages, 11 figures, code: https://github.com/F1y1113/UniWM
>
> **摘要:** Enabling embodied agents to effectively imagine future states is critical for robust and generalizable visual navigation. Current state-of-the-art approaches, however, adopt modular architectures that separate navigation planning from visual world modeling, leading to state-action misalignment and limited adaptability in novel or dynamic scenarios. To overcome this fundamental limitation, we propose UniWM, a unified, memory-augmented world model integrating egocentric visual foresight and planning within a single multimodal autoregressive backbone. Unlike modular frameworks, UniWM explicitly grounds action decisions in visually imagined outcomes, ensuring tight alignment between prediction and control. A hierarchical memory mechanism further integrates detailed short-term perceptual cues with longer-term trajectory context, enabling stable, coherent reasoning over extended horizons. Extensive experiments across four challenging benchmarks (Go Stanford, ReCon, SCAND, HuRoN) demonstrate that UniWM substantially improves navigation success rates by up to 30%, significantly reduces trajectory errors compared to strong baselines, and exhibits impressive zero-shot generalization on the unseen TartanDrive dataset. These results highlight UniWM as a principled step toward unified, imagination-driven embodied navigation.
>
---
#### [new 036] PhysToolBench: Benchmarking Physical Tool Understanding for MLLMs
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉-语言任务，旨在评估多模态大语言模型（MLLMs）对物理工具的理解能力。为解决当前缺乏量化工具理解能力的基准问题，作者构建了PhysToolBench，包含1000多个图像-文本对，覆盖工具识别、理解和创造三个层次。实验发现现有MLLMs在该任务上表现不佳，提出了改进方向。**

- **链接: [http://arxiv.org/pdf/2510.09507v1](http://arxiv.org/pdf/2510.09507v1)**

> **作者:** Zixin Zhang; Kanghao Chen; Xingwang Lin; Lutao Jiang; Xu Zheng; Yuanhuiyi Lyu; Litao Guo; Yinchuan Li; Ying-Cong Chen
>
> **摘要:** The ability to use, understand, and create tools is a hallmark of human intelligence, enabling sophisticated interaction with the physical world. For any general-purpose intelligent agent to achieve true versatility, it must also master these fundamental skills. While modern Multimodal Large Language Models (MLLMs) leverage their extensive common knowledge for high-level planning in embodied AI and in downstream Vision-Language-Action (VLA) models, the extent of their true understanding of physical tools remains unquantified. To bridge this gap, we present PhysToolBench, the first benchmark dedicated to evaluating the comprehension of physical tools by MLLMs. Our benchmark is structured as a Visual Question Answering (VQA) dataset comprising over 1,000 image-text pairs. It assesses capabilities across three distinct difficulty levels: (1) Tool Recognition: Requiring the recognition of a tool's primary function. (2) Tool Understanding: Testing the ability to grasp the underlying principles of a tool's operation. (3) Tool Creation: Challenging the model to fashion a new tool from surrounding objects when conventional options are unavailable. Our comprehensive evaluation of 32 MLLMs-spanning proprietary, open-source, specialized embodied, and backbones in VLAs-reveals a significant deficiency in tool understanding. Furthermore, we provide an in-depth analysis and propose preliminary solutions. Code and dataset are publicly available.
>
---
#### [new 037] Zero-Shot Policy Transfer in Reinforcement Learning using Buckingham's Pi Theorem
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决策略在不同机器人或环境间泛化能力差的问题。作者利用Buckingham的Pi定理，提出一种零样本策略迁移方法，通过量纲分析对策略的输入输出进行缩放，实现无需重新训练的跨环境部署。实验表明该方法在多个环境下优于基线，提升了策略的适用范围。**

- **链接: [http://arxiv.org/pdf/2510.08768v1](http://arxiv.org/pdf/2510.08768v1)**

> **作者:** Francisco Pascoa; Ian Lalonde; Alexandre Girard
>
> **摘要:** Reinforcement learning (RL) policies often fail to generalize to new robots, tasks, or environments with different physical parameters, a challenge that limits their real-world applicability. This paper presents a simple, zero-shot transfer method based on Buckingham's Pi Theorem to address this limitation. The method adapts a pre-trained policy to new system contexts by scaling its inputs (observations) and outputs (actions) through a dimensionless space, requiring no retraining. The approach is evaluated against a naive transfer baseline across three environments of increasing complexity: a simulated pendulum, a physical pendulum for sim-to-real validation, and the high-dimensional HalfCheetah. Results demonstrate that the scaled transfer exhibits no loss of performance on dynamically similar contexts. Furthermore, on non-similar contexts, the scaled policy consistently outperforms the naive transfer, significantly expanding the volume of contexts where the original policy remains effective. These findings demonstrate that dimensional analysis provides a powerful and practical tool to enhance the robustness and generalization of RL policies.
>
---
#### [new 038] BEAR: Benchmarking and Enhancing Multimodal Language Models for Atomic Embodied Capabilities
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于多模态语言模型评估与增强任务，旨在解决当前模型在具身能力上的评估不足与性能短板。作者构建了BEAR基准，包含4,469项跨模态任务，覆盖14个领域，用以全面评测多模态大模型的感知、理解与交互能力。同时提出BEAR-Agent框架，通过整合预训练视觉模型显著提升模型表现，验证了其在模拟环境中的应用潜力。**

- **链接: [http://arxiv.org/pdf/2510.08759v1](http://arxiv.org/pdf/2510.08759v1)**

> **作者:** Yu Qi; Haibo Zhao; Ziyu Guo; Siyuan Ma; Ziyan Chen; Yaokun Han; Renrui Zhang; Zitiantao Lin; Shiji Xin; Yijian Huang; Kai Cheng; Peiheng Wang; Jiazheng Liu; Jiayi Zhang; Yizhe Zhu; Wenqing Wang; Yiran Qin; Xupeng Zhu; Haojie Huang; Lawson L. S. Wong
>
> **摘要:** Embodied capabilities refer to a suite of fundamental abilities for an agent to perceive, comprehend, and interact with the physical world. While multimodal large language models (MLLMs) show promise as embodied agents, a thorough and systematic evaluation of their embodied capabilities remains underexplored, as existing benchmarks primarily focus on specific domains such as planning or spatial understanding. To bridge this gap, we introduce BEAR, a comprehensive and fine-grained benchmark that evaluates MLLMs on atomic embodied capabilities. BEAR comprises 4,469 interleaved image-video-text entries across 14 domains in 6 categories, including tasks from low-level pointing, trajectory understanding, spatial reasoning, to high-level planning. Extensive evaluation results of 20 representative MLLMs reveal their persistent limitations across all domains of embodied capabilities. To tackle the shortfall, we propose BEAR-Agent, a multimodal conversable agent that integrates pretrained vision models to strengthen MLLM perception, 3D understanding, and planning capabilities. It substantially enhances MLLM performance across diverse embodied capabilities on BEAR, yielding a 9.12% absolute gain and a relative improvement of 17.5% on GPT-5. Furthermore, our experiments indicate that improving MLLM embodied capabilities can benefit embodied tasks in simulated environments. Project website: https://bear-official66.github.io/
>
---
#### [new 039] Scalable Multi-Agent Path Finding using Collision-Aware Dynamic Alert Mask and a Hybrid Execution Strategy
- **分类: cs.MA; cs.AI; cs.RO**

- **简介: 该论文属于多智能体路径规划任务，旨在解决大规模场景下多机器人路径冲突问题。论文提出一种混合框架，结合分布式强化学习与轻量级中心协调器，通过动态冲突预警机制减少信息共享，提升算法扩展性，同时保证路径规划的可行性与安全性。**

- **链接: [http://arxiv.org/pdf/2510.09469v1](http://arxiv.org/pdf/2510.09469v1)**

> **作者:** Bharath Muppasani; Ritirupa Dey; Biplav Srivastava; Vignesh Narayanan
>
> **摘要:** Multi-agent pathfinding (MAPF) remains a critical problem in robotics and autonomous systems, where agents must navigate shared spaces efficiently while avoiding conflicts. Traditional centralized algorithms that have global information, such as Conflict-Based Search (CBS), provide high-quality solutions but become computationally expensive in large-scale scenarios due to the combinatorial explosion of conflicts that need resolution. Conversely, distributed approaches that have local information, particularly learning-based methods, offer better scalability by operating with relaxed information availability, yet often at the cost of solution quality. To address these limitations, we propose a hybrid framework that combines decentralized path planning with a lightweight centralized coordinator. Our framework leverages reinforcement learning (RL) for decentralized planning, enabling agents to adapt their planning based on minimal, targeted alerts--such as static conflict-cell flags or brief conflict tracks--that are dynamically shared information from the central coordinator for effective conflict resolution. We empirically study the effect of the information available to an agent on its planning performance. Our approach reduces the inter-agent information sharing compared to fully centralized and distributed methods, while still consistently finding feasible, collision-free solutions--even in large-scale scenarios having higher agent counts.
>
---
#### [new 040] Exploring Single Domain Generalization of LiDAR-based Semantic Segmentation under Imperfect Labels
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于LiDAR语义分割任务，旨在解决在标签不完美情况下实现跨域泛化的问题。作者提出了DGLSS-NL任务和DuNe方法，通过双视图框架提升鲁棒性，并在多个数据集上验证效果。**

- **链接: [http://arxiv.org/pdf/2510.09035v1](http://arxiv.org/pdf/2510.09035v1)**

> **作者:** Weitong Kong; Zichao Zeng; Di Wen; Jiale Wei; Kunyu Peng; June Moh Goo; Jan Boehm; Rainer Stiefelhagen
>
> **摘要:** Accurate perception is critical for vehicle safety, with LiDAR as a key enabler in autonomous driving. To ensure robust performance across environments, sensor types, and weather conditions without costly re-annotation, domain generalization in LiDAR-based 3D semantic segmentation is essential. However, LiDAR annotations are often noisy due to sensor imperfections, occlusions, and human errors. Such noise degrades segmentation accuracy and is further amplified under domain shifts, threatening system reliability. While noisy-label learning is well-studied in images, its extension to 3D LiDAR segmentation under domain generalization remains largely unexplored, as the sparse and irregular structure of point clouds limits direct use of 2D methods. To address this gap, we introduce the novel task Domain Generalization for LiDAR Semantic Segmentation under Noisy Labels (DGLSS-NL) and establish the first benchmark by adapting three representative noisy-label learning strategies from image classification to 3D segmentation. However, we find that existing noisy-label learning approaches adapt poorly to LiDAR data. We therefore propose DuNe, a dual-view framework with strong and weak branches that enforce feature-level consistency and apply cross-entropy loss based on confidence-aware filtering of predictions. Our approach shows state-of-the-art performance by achieving 56.86% mIoU on SemanticKITTI, 42.28% on nuScenes, and 52.58% on SemanticPOSS under 10% symmetric label noise, with an overall Arithmetic Mean (AM) of 49.57% and Harmonic Mean (HM) of 48.50%, thereby demonstrating robust domain generalization in DGLSS-NL tasks. The code is available on our project page.
>
---
#### [new 041] SilvaScenes: Tree Segmentation and Species Classification from Under-Canopy Images in Natural Forests
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于计算机视觉与森林管理交叉任务，旨在解决自然森林中树木实例分割与物种分类问题。现有数据集多针对城市环境或少数物种，难以满足复杂森林场景需求。论文构建了包含1476棵树、24个物种的新数据集SilvaScenes，并基于深度学习进行实例分割与分类实验，结果显示分类仍具挑战性。**

- **链接: [http://arxiv.org/pdf/2510.09458v1](http://arxiv.org/pdf/2510.09458v1)**

> **作者:** David-Alexandre Duclos; William Guimont-Martin; Gabriel Jeanson; Arthur Larochelle-Tremblay; Théo Defosse; Frédéric Moore; Philippe Nolet; François Pomerleau; Philippe Giguère
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Interest in robotics for forest management is growing, but perception in complex, natural environments remains a significant hurdle. Conditions such as heavy occlusion, variable lighting, and dense vegetation pose challenges to automated systems, which are essential for precision forestry, biodiversity monitoring, and the automation of forestry equipment. These tasks rely on advanced perceptual capabilities, such as detection and fine-grained species classification of individual trees. Yet, existing datasets are inadequate to develop such perception systems, as they often focus on urban settings or a limited number of species. To address this, we present SilvaScenes, a new dataset for instance segmentation of tree species from under-canopy images. Collected across five bioclimatic domains in Quebec, Canada, SilvaScenes features 1476 trees from 24 species with annotations from forestry experts. We demonstrate the relevance and challenging nature of our dataset by benchmarking modern deep learning approaches for instance segmentation. Our results show that, while tree segmentation is easy, with a top mean average precision (mAP) of 67.65%, species classification remains a significant challenge with an mAP of only 35.69%. Our dataset and source code will be available at https://github.com/norlab-ulaval/SilvaScenes.
>
---
#### [new 042] Toggling stiffness via multistability
- **分类: cond-mat.soft; cs.RO; physics.app-ph**

- **简介: 该论文设计了一种多稳态机械超材料，实现刚度切换。通过结构设计而非材料改变，利用弯曲与轴向变形平衡，调控支撑梁几何实现刚度比调节。实验验证了数值预测，并展示了可编程软离合器应用。属于机械超材料设计任务，旨在解决刚度动态调节问题。**

- **链接: [http://arxiv.org/pdf/2510.09511v1](http://arxiv.org/pdf/2510.09511v1)**

> **作者:** Hugo de Souza Oliveira; Michele Curatolo; Renate Sachse; Edoardo Milana
>
> **摘要:** Mechanical metamaterials enable unconventional and programmable mechanical responses through structural design rather than material composition. In this work, we introduce a multistable mechanical metamaterial that exhibits a toggleable stiffness effect, where the effective shear stiffness switches discretely between stable configurations. The mechanical analysis of surrogate beam models of the unit cell reveal that this behavior originates from the rotation transmitted by the support beams to the curved beam, which governs the balance between bending and axial deformation. The stiffness ratio between the two states of the unit cell can be tuned by varying the slenderness of the support beams or by incorporating localized hinges that modulate rotational transfer. Experiments on 3D-printed prototypes validate the numerical predictions, confirming consistent stiffness toggling across different geometries. Finally, we demonstrate a monolithic soft clutch that leverages this effect to achieve programmable, stepwise stiffness modulation. This work establishes a design strategy for toggleable stiffness using multistable metamaterials, paving the way for adaptive, lightweight, and autonomous systems in soft robotics and smart structures.
>
---
#### [new 043] Hamba: Single-view 3D Hand Reconstruction with Graph-guided Bi-Scanning Mamba
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于3D手部重建任务，旨在解决单张RGB图像中因关节运动、自遮挡和物体交互导致的重建难题。现有方法使用注意力机制但空间关系建模不足。作者提出Hamba，结合图引导与状态空间建模，设计GSS模块学习关节关系，大幅减少计算量，并融合全局与局部特征，显著提升性能。**

- **链接: [http://arxiv.org/pdf/2407.09646v2](http://arxiv.org/pdf/2407.09646v2)**

> **作者:** Haoye Dong; Aviral Chharia; Wenbo Gou; Francisco Vicente Carrasco; Fernando De la Torre
>
> **备注:** NeurIPS 2024; Project Website: https://humansensinglab.github.io/Hamba/
>
> **摘要:** 3D Hand reconstruction from a single RGB image is challenging due to the articulated motion, self-occlusion, and interaction with objects. Existing SOTA methods employ attention-based transformers to learn the 3D hand pose and shape, yet they do not fully achieve robust and accurate performance, primarily due to inefficiently modeling spatial relations between joints. To address this problem, we propose a novel graph-guided Mamba framework, named Hamba, which bridges graph learning and state space modeling. Our core idea is to reformulate Mamba's scanning into graph-guided bidirectional scanning for 3D reconstruction using a few effective tokens. This enables us to efficiently learn the spatial relationships between joints for improving reconstruction performance. Specifically, we design a Graph-guided State Space (GSS) block that learns the graph-structured relations and spatial sequences of joints and uses 88.5% fewer tokens than attention-based methods. Additionally, we integrate the state space features and the global features using a fusion module. By utilizing the GSS block and the fusion module, Hamba effectively leverages the graph-guided state space features and jointly considers global and local features to improve performance. Experiments on several benchmarks and in-the-wild tests demonstrate that Hamba significantly outperforms existing SOTAs, achieving the PA-MPVPE of 5.3mm and F@15mm of 0.992 on FreiHAND. At the time of this paper's acceptance, Hamba holds the top position, Rank 1 in two Competition Leaderboards on 3D hand reconstruction. Project Website: https://humansensinglab.github.io/Hamba/
>
---
#### [new 044] Visual Anomaly Detection for Reliable Robotic Implantation of Flexible Microelectrode Array
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉异常检测任务，旨在解决柔性微电极植入脑皮质过程中因材料形变和生物组织交互带来的可靠性与安全性问题。论文提出了一种基于显微图像的异常检测框架，利用视觉Transformer和渐进式特征采样方法，在四个关键检查点进行监测，以提高植入过程的稳定性和准确性。**

- **链接: [http://arxiv.org/pdf/2510.09071v1](http://arxiv.org/pdf/2510.09071v1)**

> **作者:** Yitong Chen; Xinyao Xu; Ping Zhu; Xinyong Han; Fangbo Qin; Shan Yu
>
> **备注:** Accept by IROS 2025
>
> **摘要:** Flexible microelectrode (FME) implantation into brain cortex is challenging due to the deformable fiber-like structure of FME probe and the interaction with critical bio-tissue. To ensure reliability and safety, the implantation process should be monitored carefully. This paper develops an image-based anomaly detection framework based on the microscopic cameras of the robotic FME implantation system. The unified framework is utilized at four checkpoints to check the micro-needle, FME probe, hooking result, and implantation point, respectively. Exploiting the existing object localization results, the aligned regions of interest (ROIs) are extracted from raw image and input to a pretrained vision transformer (ViT). Considering the task specifications, we propose a progressive granularity patch feature sampling method to address the sensitivity-tolerance trade-off issue at different locations. Moreover, we select a part of feature channels with higher signal-to-noise ratios from the raw general ViT features, to provide better descriptors for each specific scene. The effectiveness of the proposed methods is validated with the image datasets collected from our implantation system.
>
---
## 更新

#### [replaced 001] Real-time Human Finger Pointing Recognition and Estimation for Robot Directives Using a Single Web-Camera
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2307.02949v2](http://arxiv.org/pdf/2307.02949v2)**

> **作者:** Eran Bamani; Eden Nissinman; Lisa Koenigsberg; Inbar Meir; Yoav Matalon; Avishai Sintov
>
> **摘要:** Gestures play a pivotal role in human communication, often serving as a preferred or complementary medium to verbal expression due to their superior spatial reference capabilities. A finger-pointing gesture conveys vital information regarding some point of interest in the environment. In Human-Robot Interaction (HRI), users can easily direct robots to target locations, facilitating tasks in diverse domains such as search and rescue or factory assistance. State-of-the-art approaches for visual pointing estimation often rely on depth cameras, are limited to indoor environments, and provide discrete predictions between limited targets. In this paper, we explore the development of models that enable robots to understand pointing directives from humans using a single web camera, even in diverse indoor and outdoor environments. A novel perception framework is proposed which includes a designated data-based model termed PointingNet. PointingNet recognizes the occurrence of pointing through classification followed by approximating the position and direction of the index finger with an advanced regression model. The model relies on a novel segmentation model for masking any lifted arm. While state-of-the-art human pose estimation models provide poor pointing angle estimation error of 28deg, PointingNet exhibits a mean error of less than 2deg. With the pointing information, the target location is computed, followed by robot motion planning and execution. The framework is evaluated on two robotic systems, demonstrating accurate target reaching.
>
---
#### [replaced 002] NAMOUnc: Navigation Among Movable Obstacles with Decision Making on Uncertainty Interval
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.12723v2](http://arxiv.org/pdf/2509.12723v2)**

> **作者:** Kai Zhang; Eric Lucet; Julien Alexandre Dit Sandretto; Shoubin Chen; David Filliat
>
> **备注:** 11 pages, ICINCO2025
>
> **摘要:** Navigation among movable obstacles (NAMO) is a critical task in robotics, often challenged by real-world uncertainties such as observation noise, model approximations, action failures, and partial observability. Existing solutions frequently assume ideal conditions, leading to suboptimal or risky decisions. This paper introduces NAMOUnc, a novel framework designed to address these uncertainties by integrating them into the decision-making process. We first estimate them and compare the corresponding time cost intervals for removing and bypassing obstacles, optimizing both the success rate and time efficiency, ensuring safer and more efficient navigation. We validate our method through extensive simulations and real-world experiments, demonstrating significant improvements over existing NAMO frameworks. More details can be found in our website: https://kai-zhang-er.github.io/namo-uncertainty/
>
---
#### [replaced 003] The Impact of 2D Segmentation Backbones on Point Cloud Predictions Using 4D Radar
- **分类: cs.CV; cs.RO; I.4.6; I.2.9; I.2.10**

- **链接: [http://arxiv.org/pdf/2509.19644v2](http://arxiv.org/pdf/2509.19644v2)**

> **作者:** William Muckelroy III; Mohammed Alsakabi; John Dolan; Ozan Tonguz
>
> **摘要:** LiDAR's dense, sharp point cloud (PC) representations of the surrounding environment enable accurate perception and significantly improve road safety by offering greater scene awareness and understanding. However, LiDAR's high cost continues to restrict the broad adoption of high-level Autonomous Driving (AD) systems in commercially available vehicles. Prior research has shown progress towards circumventing the need for LiDAR by training a neural network, using LiDAR point clouds as ground truth (GT), to produce LiDAR-like 3D point clouds using only 4D Radars. One of the best examples is a neural network created to train a more efficient radar target detector with a modular 2D convolutional neural network (CNN) backbone and a temporal coherence network at its core that uses the RaDelft dataset for training (see arXiv:2406.04723). In this work, we investigate the impact of higher-capacity segmentation backbones on the quality of the produced point clouds. Our results show that while very high-capacity models may actually hurt performance, an optimal segmentation backbone can provide a 23.7% improvement over the state-of-the-art (SOTA).
>
---
#### [replaced 004] SwarmGPT: Combining Large Language Models with Safe Motion Planning for Drone Swarm Choreography
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2412.08428v2](http://arxiv.org/pdf/2412.08428v2)**

> **作者:** Martin Schuck; Dinushka Orrin Dahanaggamaarachchi; Ben Sprenger; Vedant Vyas; Siqi Zhou; Angela P. Schoellig
>
> **备注:** Accepted at RA-L 2025
>
> **摘要:** Drone swarm performances -- synchronized, expressive aerial displays set to music -- have emerged as a captivating application of modern robotics. Yet designing smooth, safe choreographies remains a complex task requiring expert knowledge. We present SwarmGPT, a language-based choreographer that leverages the reasoning power of large language models (LLMs) to streamline drone performance design. The LLM is augmented by a safety filter that ensures deployability by making minimal corrections when safety or feasibility constraints are violated. By decoupling high-level choreographic design from low-level motion planning, our system enables non-experts to iteratively refine choreographies using natural language without worrying about collisions or actuator limits. We validate our approach through simulations with swarms up to 200 drones and real-world experiments with up to 20 drones performing choreographies to diverse types of songs, demonstrating scalable, synchronized, and safe performances. Beyond entertainment, this work offers a blueprint for integrating foundation models into safety-critical swarm robotics applications.
>
---
#### [replaced 005] A Multimodal Depth-Aware Method For Embodied Reference Understanding
- **分类: cs.CV; cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2510.08278v2](http://arxiv.org/pdf/2510.08278v2)**

> **作者:** Fevziye Irem Eyiokur; Dogucan Yaman; Hazım Kemal Ekenel; Alexander Waibel
>
> **摘要:** Embodied Reference Understanding requires identifying a target object in a visual scene based on both language instructions and pointing cues. While prior works have shown progress in open-vocabulary object detection, they often fail in ambiguous scenarios where multiple candidate objects exist in the scene. To address these challenges, we propose a novel ERU framework that jointly leverages LLM-based data augmentation, depth-map modality, and a depth-aware decision module. This design enables robust integration of linguistic and embodied cues, improving disambiguation in complex or cluttered environments. Experimental results on two datasets demonstrate that our approach significantly outperforms existing baselines, achieving more accurate and reliable referent detection.
>
---
#### [replaced 006] Maximizing UAV Cellular Connectivity with Reinforcement Learning for BVLoS Path Planning
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2509.13336v2](http://arxiv.org/pdf/2509.13336v2)**

> **作者:** Mehran Behjati; Rosdiadee Nordin; Nor Fadzilah Abdullah
>
> **备注:** Submitted to an IEEE Conference
>
> **摘要:** This paper presents a reinforcement learning (RL) based approach for path planning of cellular connected unmanned aerial vehicles (UAVs) operating beyond visual line of sight (BVLoS). The objective is to minimize travel distance while maximizing the quality of cellular link connectivity by considering real world aerial coverage constraints and employing an empirical aerial channel model. The proposed solution employs RL techniques to train an agent, using the quality of communication links between the UAV and base stations (BSs) as the reward function. Simulation results demonstrate the effectiveness of the proposed method in training the agent and generating feasible UAV path plans. The proposed approach addresses the challenges due to limitations in UAV cellular communications, highlighting the need for investigations and considerations in this area. The RL algorithm efficiently identifies optimal paths, ensuring maximum connectivity with ground BSs to ensure safe and reliable BVLoS flight operation. Moreover, the solution can be deployed as an offline path planning module that can be integrated into future ground control systems (GCS) for UAV operations, enhancing their capabilities and safety. The method holds potential for complex long range UAV applications, advancing the technology in the field of cellular connected UAV path planning.
>
---
#### [replaced 007] Automating eHMI Action Design with LLMs for Automated Vehicle Communication
- **分类: cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2505.20711v2](http://arxiv.org/pdf/2505.20711v2)**

> **作者:** Ding Xia; Xinyue Gui; Fan Gao; Dongyuan Li; Mark Colley; Takeo Igarashi
>
> **备注:** Accepted as findings for EMNLP 2025
>
> **摘要:** The absence of explicit communication channels between automated vehicles (AVs) and other road users requires the use of external Human-Machine Interfaces (eHMIs) to convey messages effectively in uncertain scenarios. Currently, most eHMI studies employ predefined text messages and manually designed actions to perform these messages, which limits the real-world deployment of eHMIs, where adaptability in dynamic scenarios is essential. Given the generalizability and versatility of large language models (LLMs), they could potentially serve as automated action designers for the message-action design task. To validate this idea, we make three contributions: (1) We propose a pipeline that integrates LLMs and 3D renderers, using LLMs as action designers to generate executable actions for controlling eHMIs and rendering action clips. (2) We collect a user-rated Action-Design Scoring dataset comprising a total of 320 action sequences for eight intended messages and four representative eHMI modalities. The dataset validates that LLMs can translate intended messages into actions close to a human level, particularly for reasoning-enabled LLMs. (3) We introduce two automated raters, Action Reference Score (ARS) and Vision-Language Models (VLMs), to benchmark 18 LLMs, finding that the VLM aligns with human preferences yet varies across eHMI modalities.
>
---
#### [replaced 008] Multi-robot Rigid Formation Navigation via Synchronous Motion and Discrete-time Communication-Control Optimization
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.02624v2](http://arxiv.org/pdf/2510.02624v2)**

> **作者:** Qun Yang; Soung Chang Liew
>
> **摘要:** Rigid-formation navigation of multiple robots is essential for applications such as cooperative transportation. This process involves a team of collaborative robots maintaining a predefined geometric configuration, such as a square, while in motion. For untethered collaborative motion, inter-robot communication must be conducted through a wireless network. Notably, few existing works offer a comprehensive solution for multi-robot formation navigation executable on microprocessor platforms via wireless networks, particularly for formations that must traverse complex curvilinear paths. To address this gap, we introduce a novel "hold-and-hit" communication-control framework designed to work seamlessly with the widely-used Robotic Operating System (ROS) platform. The hold-and-hit framework synchronizes robot movements in a manner robust against wireless network delays and packet loss. It operates over discrete-time communication-control cycles, making it suitable for implementation on contemporary microprocessors. Complementary to hold-and-hit, we propose an intra-cycle optimization approach that enables rigid formations to closely follow desired curvilinear paths, even under the nonholonomic movement constraints inherent to most vehicular robots. The combination of hold-and-hit and intra-cycle optimization ensures precise and reliable navigation even in challenging scenarios. Simulations in a virtual environment demonstrate the superiority of our method in maintaining a four-robot square formation along an S-shaped path, outperforming two existing approaches. Furthermore, real-world experiments validate the effectiveness of our framework: the robots maintained an inter-distance error within $\pm 0.069m$ and an inter-angular orientation error within $\pm19.15^{\circ}$ while navigating along an S-shaped path at a fixed linear velocity of $0.1 m/s$.
>
---
#### [replaced 009] Safe Autonomous Environmental Contact for Soft Robots using Control Barrier Functions
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2504.14755v2](http://arxiv.org/pdf/2504.14755v2)**

> **作者:** Akua K. Dickson; Juan C. Pacheco Garcia; Meredith L. Anderson; Ran Jing; Sarah Alizadeh-Shabdiz; Audrey X. Wang; Charles DeLorey; Zach J. Patterson; Andrew P. Sabelhaus
>
> **备注:** 8 pages, 9 figures
>
> **摘要:** Robots built from soft materials will inherently apply lower environmental forces than their rigid counterparts, and therefore may be more suitable in sensitive settings with unintended contact. However, these robots' applied forces result from both their design and their control system in closed-loop, and therefore, ensuring bounds on these forces requires controller synthesis for safety as well. This article introduces the first feedback controller for a soft manipulator that formally meets a safety specification with respect to environmental contact. In our proof-of-concept setting, the robot's environment has known geometry and is deformable with a known elastic modulus. Our approach maps a bound on applied forces to a safe set of positions of the robot's tip via predicted deformations of the environment. Then, a quadratic program with Control Barrier Functions in its constraints is used to supervise a nominal feedback signal, verifiably maintaining the robot's tip within this safe set. Hardware experiments on a multi-segment soft pneumatic robot demonstrate that the proposed framework successfully maintains a positive safety margin. This framework represents a fundamental shift in perspective on control and safety for soft robots, implementing a formally verifiable logic specification on their pose and contact forces.
>
---
#### [replaced 010] Robo-Instruct: Simulator-Augmented Instruction Alignment For Finetuning Code LLMs
- **分类: cs.CL; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2405.20179v5](http://arxiv.org/pdf/2405.20179v5)**

> **作者:** Zichao Hu; Junyi Jessy Li; Arjun Guha; Joydeep Biswas
>
> **备注:** Conference on Language Modeling (COLM) 2025, Project site: https://amrl.cs.utexas.edu/robo-instruct/
>
> **摘要:** Code LLMs have shown promising results with converting tasks in natural language to programs that can be executed by service robots. We are interested in finetuning small, specialized LLMs for this purpose, but collecting datasets of task-program pairs specific to each robot is time-consuming and expensive. While approaches such as SELF-INSTRUCT and EVOL-INSTRUCT are capable of generating novel tasks given a few examples, they are unable to provide the corresponding programs that correctly abide by physical-world and robot-constraints using the provided programming interface. Using a simulator is a natural potential solution to checking for such constraints, but building simulation environments that can handle arbitrary tasks and their necessary objects and locations, is challenging. To address these challenges, we introduce ROBO-INSTRUCT, which synthesizes task-specific simulation environments on the fly during program execution, by opportunistically inferring entity properties and enforcing corresponding constraints based on how the entities are used in the task program. Additionally, ROBO-INSTRUCT integrates an LLM-aided post-processing procedure to refine instructions for better alignment with robot programs. We demonstrate the effectiveness of ROBO-INSTRUCT across multiple LLMs, showing that our fine-tuned models outperform all baseline methods and even match or surpass the performance of several larger and proprietary models.
>
---
#### [replaced 011] Artists' Views on Robotics Involvement in Painting Productions
- **分类: cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2510.07063v2](http://arxiv.org/pdf/2510.07063v2)**

> **作者:** Francesca Cocchella; Nilay Roy Choudhury; Eric Chen; Patrícia Alves-Oliveira
>
> **备注:** 10 pages, 9 figures, submitted to RAM special issue: Arts and Robotics
>
> **摘要:** As robotic technologies evolve, their potential in artistic creation becomes an increasingly relevant topic of inquiry. This study explores how professional abstract artists perceive and experience co-creative interactions with an autonomous painting robotic arm. Eight artists engaged in six painting sessions -- three with a human partner, followed by three with the robot -- and subsequently participated in semi-structured interviews analyzed through reflexive thematic analysis. Human-human interactions were described as intuitive, dialogic, and emotionally engaging, whereas human-robot sessions felt more playful and reflective, offering greater autonomy and prompting for novel strategies to overcome the system's limitations. This work offers one of the first empirical investigations into artists' lived experiences with a robot, highlighting the value of long-term engagement and a multidisciplinary approach to human-robot co-creation.
>
---
#### [replaced 012] Mitigating Suboptimality of Deterministic Policy Gradients in Complex Q-functions
- **分类: cs.LG; cs.AI; cs.RO; stat.ML**

- **链接: [http://arxiv.org/pdf/2410.11833v2](http://arxiv.org/pdf/2410.11833v2)**

> **作者:** Ayush Jain; Norio Kosaka; Xinhu Li; Kyung-Min Kim; Erdem Bıyık; Joseph J. Lim
>
> **备注:** Outstanding Paper Award on Empirical Reinforcement Learning Research, RLC 2025
>
> **摘要:** In reinforcement learning, off-policy actor-critic methods like DDPG and TD3 use deterministic policy gradients: the Q-function is learned from environment data, while the actor maximizes it via gradient ascent. We observe that in complex tasks such as dexterous manipulation and restricted locomotion with mobility constraints, the Q-function exhibits many local optima, making gradient ascent prone to getting stuck. To address this, we introduce SAVO, an actor architecture that (i) generates multiple action proposals and selects the one with the highest Q-value, and (ii) approximates the Q-function repeatedly by truncating poor local optima to guide gradient ascent more effectively. We evaluate tasks such as restricted locomotion, dexterous manipulation, and large discrete-action space recommender systems and show that our actor finds optimal actions more frequently and outperforms alternate actor architectures.
>
---
#### [replaced 013] An Introduction to Zero-Order Optimization Techniques for Robotics
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.22087v2](http://arxiv.org/pdf/2506.22087v2)**

> **作者:** Armand Jordana; Jianghan Zhang; Joseph Amigo; Ludovic Righetti
>
> **摘要:** Zero-order optimization techniques are becoming increasingly popular in robotics due to their ability to handle non-differentiable functions and escape local minima. These advantages make them particularly useful for trajectory optimization and policy optimization. In this work, we propose a mathematical tutorial on random search. It offers a simple and unifying perspective for understanding a wide range of algorithms commonly used in robotics. Leveraging this viewpoint, we classify many trajectory optimization methods under a common framework and derive novel competitive RL algorithms.
>
---
#### [replaced 014] SMapper: A Multi-Modal Data Acquisition Platform for SLAM Benchmarking
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2509.09509v2](http://arxiv.org/pdf/2509.09509v2)**

> **作者:** Pedro Miguel Bastos Soares; Ali Tourani; Miguel Fernandez-Cortizas; Asier Bikandi-Noya; Holger Voos; Jose Luis Sanchez-Lopez
>
> **备注:** 13 pages, 5 figures, 6 tables
>
> **摘要:** Advancing research in fields such as Simultaneous Localization and Mapping (SLAM) and autonomous navigation critically depends on the availability of reliable and reproducible multimodal datasets. While several influential datasets have driven progress in these domains, they often suffer from limitations in sensing modalities, environmental diversity, and the reproducibility of the underlying hardware setups. To address these challenges, this paper introduces SMapper, a novel open-hardware, multi-sensor platform designed explicitly for, though not limited to, SLAM research. The device integrates synchronized LiDAR, multi-camera, and inertial sensing, supported by a robust calibration and synchronization pipeline that ensures precise spatio-temporal alignment across modalities. Its open and replicable design allows researchers to extend its capabilities and reproduce experiments across both handheld and robot-mounted scenarios. To demonstrate its practicality, we additionally release SMapper-light, a publicly available SLAM dataset containing representative indoor and outdoor sequences. The dataset includes tightly synchronized multimodal data and ground truth trajectories derived from offline LiDAR-based SLAM with sub-centimeter accuracy, alongside dense 3D reconstructions. Furthermore, the paper contains benchmarking results on state-of-the-art LiDAR and visual SLAM frameworks using the SMapper-light dataset. By combining open-hardware design, reproducible data collection, and comprehensive benchmarking, SMapper establishes a robust foundation for advancing SLAM algorithm development, evaluation, and reproducibility. The project's documentation, including source code, CAD models, and dataset links, is publicly available at https://snt-arg.github.io/smapper_docs.
>
---
#### [replaced 015] An Imitative Reinforcement Learning Framework for Pursuit-Lock-Launch Missions
- **分类: cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2406.11562v3](http://arxiv.org/pdf/2406.11562v3)**

> **作者:** Siyuan Li; Rongchang Zuo; Bofei Liu; Yaoyu He; Peng Liu; Yingnan Zhao
>
> **摘要:** Unmanned Combat Aerial Vehicle (UCAV) Within-Visual-Range (WVR) engagement, referring to a fight between two or more UCAVs at close quarters, plays a decisive role on the aerial battlefields. With the development of artificial intelligence, WVR engagement progressively advances towards intelligent and autonomous modes. However, autonomous WVR engagement policy learning is hindered by challenges such as weak exploration capabilities, low learning efficiency, and unrealistic simulated environments. To overcome these challenges, we propose a novel imitative reinforcement learning framework, which efficiently leverages expert data while enabling autonomous exploration. The proposed framework not only enhances learning efficiency through expert imitation, but also ensures adaptability to dynamic environments via autonomous exploration with reinforcement learning. Therefore, the proposed framework can learn a successful policy of `pursuit-lock-launch' for UCAVs. To support data-driven learning, we establish an environment based on the Harfang3D sandbox. The extensive experiment results indicate that the proposed framework excels in this multistage task, and significantly outperforms state-of-the-art reinforcement learning and imitation learning methods. Thanks to the ability of imitating experts and autonomous exploration, our framework can quickly learn the critical knowledge in complex aerial combat tasks, achieving up to a 100% success rate and demonstrating excellent robustness.
>
---
#### [replaced 016] Navigation and Exploration with Active Inference: from Biology to Industry
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.07269v2](http://arxiv.org/pdf/2508.07269v2)**

> **作者:** Daria de Tinguy; Tim Verbelen; Bart Dhoedt
>
> **备注:** conference IWAI 2025 - accepted (in processing)
>
> **摘要:** By building and updating internal cognitive maps, animals exhibit extraordinary navigation abilities in complex, dynamic environments. Inspired by these biological mechanisms, we present a real time robotic navigation system grounded in the Active Inference Framework (AIF). Our model incrementally constructs a topological map, infers the agent's location, and plans actions by minimising expected uncertainty and fulfilling perceptual goals without any prior training. Integrated into the ROS2 ecosystem, we validate its adaptability and efficiency across both 2D and 3D environments (simulated and real world), demonstrating competitive performance with traditional and state of the art exploration approaches while offering a biologically inspired navigation approach.
>
---
#### [replaced 017] Through the Perspective of LiDAR: A Feature-Enriched and Uncertainty-Aware Annotation Pipeline for Terrestrial Point Cloud Segmentation
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2510.06582v2](http://arxiv.org/pdf/2510.06582v2)**

> **作者:** Fei Zhang; Rob Chancia; Josie Clapp; Amirhossein Hassanzadeh; Dimah Dera; Richard MacKenzie; Jan van Aardt
>
> **备注:** 40 pages (28 main text), 20 figures, 4 supplementary materials; links to 3D point animations are included in the last table
>
> **摘要:** Accurate semantic segmentation of terrestrial laser scanning (TLS) point clouds is limited by costly manual annotation. We propose a semi-automated, uncertainty-aware pipeline that integrates spherical projection, feature enrichment, ensemble learning, and targeted annotation to reduce labeling effort, while sustaining high accuracy. Our approach projects 3D points to a 2D spherical grid, enriches pixels with multi-source features, and trains an ensemble of segmentation networks to produce pseudo-labels and uncertainty maps, the latter guiding annotation of ambiguous regions. The 2D outputs are back-projected to 3D, yielding densely annotated point clouds supported by a three-tier visualization suite (2D feature maps, 3D colorized point clouds, and compact virtual spheres) for rapid triage and reviewer guidance. Using this pipeline, we build Mangrove3D, a semantic segmentation TLS dataset for mangrove forests. We further evaluate data efficiency and feature importance to address two key questions: (1) how much annotated data are needed and (2) which features matter most. Results show that performance saturates after ~12 annotated scans, geometric features contribute the most, and compact nine-channel stacks capture nearly all discriminative power, with the mean Intersection over Union (mIoU) plateauing at around 0.76. Finally, we confirm the generalization of our feature-enrichment strategy through cross-dataset tests on ForestSemantic and Semantic3D. Our contributions include: (i) a robust, uncertainty-aware TLS annotation pipeline with visualization tools; (ii) the Mangrove3D dataset; and (iii) empirical guidance on data efficiency and feature importance, thus enabling scalable, high-quality segmentation of TLS point clouds for ecological monitoring and beyond. The dataset and processing scripts are publicly available at https://fz-rit.github.io/through-the-lidars-eye/.
>
---
#### [replaced 018] USIM and U0: A Vision-Language-Action Dataset and Model for General Underwater Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.07869v2](http://arxiv.org/pdf/2510.07869v2)**

> **作者:** Junwen Gu; Zhiheng wu; Pengxuan Si; Shuang Qiu; Yukai Feng; Luoyang Sun; Laien Luo; Lianyi Yu; Jian Wang; Zhengxing Wu
>
> **备注:** Project Page: https://vincentgu2000.github.io/u0project/
>
> **摘要:** Underwater environments present unique challenges for robotic operation, including complex hydrodynamics, limited visibility, and constrained communication. Although data-driven approaches have advanced embodied intelligence in terrestrial robots and enabled task-specific autonomous underwater robots, developing underwater intelligence capable of autonomously performing multiple tasks remains highly challenging, as large-scale, high-quality underwater datasets are still scarce. To address these limitations, we introduce USIM, a simulation-based multi-task Vision-Language-Action (VLA) dataset for underwater robots. USIM comprises over 561K frames from 1,852 trajectories, totaling approximately 15.6 hours of BlueROV2 interactions across 20 tasks in 9 diverse scenarios, ranging from visual navigation to mobile manipulation. Building upon this dataset, we propose U0, a VLA model for general underwater robots, which integrates binocular vision and other sensor modalities through multimodal fusion, and further incorporates a convolution-attention-based perception focus enhancement module (CAP) to improve spatial understanding and mobile manipulation. Across tasks such as inspection, obstacle avoidance, scanning, and dynamic tracking, the framework achieves a success rate of 80%, while in challenging mobile manipulation tasks, it reduces the distance to the target by 21.2% compared with baseline methods, demonstrating its effectiveness. USIM and U0 show that VLA models can be effectively applied to underwater robotic applications, providing a foundation for scalable dataset construction, improved task autonomy, and the practical realization of intelligent general underwater robots.
>
---
#### [replaced 019] CCDP: Composition of Conditional Diffusion Policies with Guided Sampling
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2503.15386v2](http://arxiv.org/pdf/2503.15386v2)**

> **作者:** Amirreza Razmjoo; Sylvain Calinon; Michael Gienger; Fan Zhang
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** Imitation Learning offers a promising approach to learn directly from data without requiring explicit models, simulations, or detailed task definitions. During inference, actions are sampled from the learned distribution and executed on the robot. However, sampled actions may fail for various reasons, and simply repeating the sampling step until a successful action is obtained can be inefficient. In this work, we propose an enhanced sampling strategy that refines the sampling distribution to avoid previously unsuccessful actions. We demonstrate that by solely utilizing data from successful demonstrations, our method can infer recovery actions without the need for additional exploratory behavior or a high-level controller. Furthermore, we leverage the concept of diffusion model decomposition to break down the primary problem, which may require long-horizon history to manage failures, into multiple smaller, more manageable sub-problems in learning, data collection, and inference, thereby enabling the system to adapt to variable failure counts. Our approach yields a low-level controller that dynamically adjusts its sampling space to improve efficiency when prior samples fall short. We validate our method across several tasks, including door opening with unknown directions, object manipulation, and button-searching scenarios, demonstrating that our approach outperforms traditional baselines.
>
---
#### [replaced 020] Aegis: Automated Error Generation and Attribution for Multi-Agent Systems
- **分类: cs.RO; cs.MA**

- **链接: [http://arxiv.org/pdf/2509.14295v4](http://arxiv.org/pdf/2509.14295v4)**

> **作者:** Fanqi Kong; Ruijie Zhang; Huaxiao Yin; Guibin Zhang; Xiaofei Zhang; Ziang Chen; Zhaowei Zhang; Xiaoyuan Zhang; Song-Chun Zhu; Xue Feng
>
> **摘要:** Large language model based multi-agent systems (MAS) have unlocked significant advancements in tackling complex problems, but their increasing capability introduces a structural fragility that makes them difficult to debug. A key obstacle to improving their reliability is the severe scarcity of large-scale, diverse datasets for error attribution, as existing resources rely on costly and unscalable manual annotation. To address this bottleneck, we introduce Aegis, a novel framework for Automated error generation and attribution for multi-agent systems. Aegis constructs a large dataset of 9,533 trajectories with annotated faulty agents and error modes, covering diverse MAS architectures and task domains. This is achieved using a LLM-based manipulator that can adaptively inject context-aware errors into successful execution trajectories. Leveraging fine-grained labels and the structured arrangement of positive-negative sample pairs, Aegis supports three different learning paradigms: Supervised Fine-Tuning, Reinforcement Learning, and Contrastive Learning. We develop learning methods for each paradigm. Comprehensive experiments show that trained models consistently achieve substantial improvements in error attribution. Notably, several of our fine-tuned LLMs demonstrate performance competitive with or superior to proprietary models an order of magnitude larger, validating our automated data generation framework as a crucial resource for developing more robust and interpretable multi-agent systems. Our project website is available at https://kfq20.github.io/Aegis-Website/.
>
---
#### [replaced 021] Event-RGB Fusion for Spacecraft Pose Estimation Under Harsh Lighting
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.05698v2](http://arxiv.org/pdf/2507.05698v2)**

> **作者:** Mohsi Jawaid; Marcus Märtens; Tat-Jun Chin
>
> **摘要:** Spacecraft pose estimation is crucial for autonomous in-space operations, such as rendezvous, docking and on-orbit servicing. Vision-based pose estimation methods, which typically employ RGB imaging sensors, is a compelling solution for spacecraft pose estimation, but are challenged by harsh lighting conditions, which produce imaging artifacts such as glare, over-exposure, blooming and lens flare. Due to their much higher dynamic range, neuromorphic or event sensors are more resilient to extreme lighting conditions. However, event sensors generally have lower spatial resolution and suffer from reduced signal-to-noise ratio during periods of low relative motion. This work addresses these individual sensor limitations by introducing a sensor fusion approach combining RGB and event sensors. A beam-splitter prism was employed to achieve precise optical and temporal alignment. Then, a RANSAC-based technique was developed to fuse the information from the RGB and event channels to achieve pose estimation that leveraged the strengths of the two modalities. The pipeline was complemented by dropout uncertainty estimation to detect extreme conditions that affect either channel. To benchmark the performance of the proposed event-RGB fusion method, we collected a comprehensive real dataset of RGB and event data for satellite pose estimation in a laboratory setting under a variety of challenging illumination conditions. Encouraging results on the dataset demonstrate the efficacy of our event-RGB fusion approach and further supports the usage of event sensors for spacecraft pose estimation. To support community research on this topic, our dataset has been released publicly.
>
---
#### [replaced 022] DPL: Depth-only Perceptive Humanoid Locomotion via Realistic Depth Synthesis and Cross-Attention Terrain Reconstruction
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2510.07152v2](http://arxiv.org/pdf/2510.07152v2)**

> **作者:** Jingkai Sun; Gang Han; Pihai Sun; Wen Zhao; Jiahang Cao; Jiaxu Wang; Yijie Guo; Qiang Zhang
>
> **摘要:** Recent advancements in legged robot perceptive locomotion have shown promising progress. However, terrain-aware humanoid locomotion remains largely constrained to two paradigms: depth image-based end-to-end learning and elevation map-based methods. The former suffers from limited training efficiency and a significant sim-to-real gap in depth perception, while the latter depends heavily on multiple vision sensors and localization systems, resulting in latency and reduced robustness. To overcome these challenges, we propose a novel framework that tightly integrates three key components: (1) Terrain-Aware Locomotion Policy with a Blind Backbone, which leverages pre-trained elevation map-based perception to guide reinforcement learning with minimal visual input; (2) Multi-Modality Cross-Attention Transformer, which reconstructs structured terrain representations from noisy depth images; (3) Realistic Depth Images Synthetic Method, which employs self-occlusion-aware ray casting and noise-aware modeling to synthesize realistic depth observations, achieving over 30\% reduction in terrain reconstruction error. This combination enables efficient policy training with limited data and hardware resources, while preserving critical terrain features essential for generalization. We validate our framework on a full-sized humanoid robot, demonstrating agile and adaptive locomotion across diverse and challenging terrains.
>
---
#### [replaced 023] A Knowledge-Informed Deep Learning Paradigm for Generalizable and Stability-Optimized Car-Following Models
- **分类: cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.14241v2](http://arxiv.org/pdf/2504.14241v2)**

> **作者:** Chengming Wang; Dongyao Jia; Wei Wang; Dong Ngoduy; Bei Peng; Jianping Wang
>
> **摘要:** Car-following models (CFMs) are fundamental to traffic flow analysis and autonomous driving. Although calibrated physics-based and trained data-driven CFMs can replicate human driving behavior, their reliance on specific datasets limits generalization across diverse scenarios and reduces reliability in real-world deployment. Moreover, these models typically focus on behavioral fidelity and do not support the explicit optimization of local and string stability, which are increasingly important for the safe and efficient operation of autonomous vehicles (AVs). To address these limitations, we propose a Knowledge-Informed Deep Learning (KIDL) paradigm that distills the generalization capabilities of pre-trained Large Language Models (LLMs) into a lightweight and stability-aware neural architecture. LLMs are used to extract fundamental car-following knowledge beyond dataset-specific patterns, and this knowledge is transferred to a reliable, tractable, and computationally efficient model through knowledge distillation. KIDL also incorporates stability constraints directly into its training objective, ensuring that the resulting model not only emulates human-like behavior but also satisfies the local and string stability requirements essential for real-world AV deployment. We evaluate KIDL on the real-world NGSIM and HighD datasets, comparing its performance with representative physics-based, data-driven, and hybrid CFMs. Both empirical and theoretical results consistently demonstrate KIDL's superior behavioral generalization and traffic flow stability, offering a robust and scalable solution for next-generation traffic systems.
>
---
#### [replaced 024] Hybrid Feedback Control for Global Navigation with Locally Optimal Obstacle Avoidance in n-Dimensional Spaces
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2412.20320v3](http://arxiv.org/pdf/2412.20320v3)**

> **作者:** Ishak Cheniouni; Soulaimane Berkane; Abdelhamid Tayebi
>
> **摘要:** We present a hybrid feedback control framework for autonomous robot navigation in n-dimensional Euclidean spaces cluttered with spherical obstacles. The proposed approach ensures safe and global navigation towards a target location by dynamically switching between two operational modes: motion-to-destination and locally optimal obstacle-avoidance. It produces continuous velocity inputs, ensures collision-free trajectories and generates locally optimal obstacle avoidance maneuvers. Unlike existing methods, the proposed framework is compatible with range sensors, enabling navigation in both a priori known and unknown environments. Extensive simulations in 2D and 3D settings, complemented by experimental validation on a TurtleBot 4 platform, confirm the efficacy and robustness of the approach. Our results demonstrate shorter paths and smoother trajectories compared to state-of-the-art methods, while maintaining computational efficiency and real-world feasibility.
>
---
#### [replaced 025] IG-MCTS: Human-in-the-Loop Cooperative Navigation under Incomplete Information
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2502.01857v2](http://arxiv.org/pdf/2502.01857v2)**

> **作者:** Shenghui Chen; Ruihan Zhao; Sandeep Chinchali; Ufuk Topcu
>
> **摘要:** Human-robot cooperative navigation is challenging under incomplete information. We introduce CoNav-Maze, a simulated environment where a robot navigates with local perception while a human operator provides guidance based on an inaccurate map. The robot can share its onboard camera views to help the operator refine their understanding of the environment. To enable efficient cooperation, we propose Information Gain Monte Carlo Tree Search (IG-MCTS), an online planning algorithm that jointly optimizes autonomous movement and informative communication. IG-MCTS leverages a learned Neural Human Perception Model (NHPM) -- trained on a crowdsourced mapping dataset -- to predict how the human's internal map evolves as new observations are shared. User studies show that IG-MCTS significantly reduces communication demands and yields eye-tracking metrics indicative of lower cognitive load, while maintaining task performance comparable to teleoperation and instruction-following baselines. Finally, we illustrate generalization beyond discrete mazes through a continuous-space waterway navigation setting, in which NHPM benefits from deeper encoder-decoder architectures and IG-MCTS leverages a dynamically constructed Voronoi-partitioned traversability graph.
>
---
#### [replaced 026] Extending First-order Robotic Motion Planners to Second-order Robot Dynamics
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.17589v2](http://arxiv.org/pdf/2503.17589v2)**

> **作者:** Mayur Sawant; Abdelhamid Tayebi
>
> **备注:** 14 pages, 10 figures
>
> **摘要:** This paper extends first-order motion planners to robots governed by second-order dynamics. Two control schemes are proposed based on the knowledge of a scalar function whose negative gradient aligns with a given first-order motion planner. When such a function is known, the first-order motion planner is combined with a damping velocity vector with a dynamic gain to extend the safety and convergence guarantees of the first-order motion planner to second-order systems. If no such function is available, we propose an alternative control scheme ensuring that the error between the robot's velocity and the first-order motion planner converges to zero. The theoretical developments are supported by simulation results demonstrating the effectiveness of the proposed approaches.
>
---
#### [replaced 027] MP1: MeanFlow Tames Policy Learning in 1-step for Robotic Manipulation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.10543v4](http://arxiv.org/pdf/2507.10543v4)**

> **作者:** Juyi Sheng; Ziyi Wang; Peiming Li; Mengyuan Liu
>
> **摘要:** In robot manipulation, robot learning has become a prevailing approach. However, generative models within this field face a fundamental trade-off between the slow, iterative sampling of diffusion models and the architectural constraints of faster Flow-based methods, which often rely on explicit consistency losses. To address these limitations, we introduce MP1, which pairs 3D point-cloud inputs with the MeanFlow paradigm to generate action trajectories in one network function evaluation (1-NFE). By directly learning the interval-averaged velocity via the "MeanFlow Identity", our policy avoids any additional consistency constraints. This formulation eliminates numerical ODE-solver errors during inference, yielding more precise trajectories. MP1 further incorporates CFG for improved trajectory controllability while retaining 1-NFE inference without reintroducing structural constraints. Because subtle scene-context variations are critical for robot learning, especially in few-shot learning, we introduce a lightweight Dispersive Loss that repels state embeddings during training, boosting generalization without slowing inference. We validate our method on the Adroit and Meta-World benchmarks, as well as in real-world scenarios. Experimental results show MP1 achieves superior average task success rates, outperforming DP3 by 10.2% and FlowPolicy by 7.3%. Its average inference time is only 6.8 ms-19x faster than DP3 and nearly 2x faster than FlowPolicy. Our project page is available at https://mp1-2254.github.io/, and the code can be accessed at https://github.com/LogSSim/MP1.
>
---
#### [replaced 028] AirScape: An Aerial Generative World Model with Motion Controllability
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.08885v2](http://arxiv.org/pdf/2507.08885v2)**

> **作者:** Baining Zhao; Rongze Tang; Mingyuan Jia; Ziyou Wang; Fanghang Man; Xin Zhang; Yu Shang; Weichen Zhang; Wei Wu; Chen Gao; Xinlei Chen; Yong Li
>
> **摘要:** How to enable agents to predict the outcomes of their own motion intentions in three-dimensional space has been a fundamental problem in embodied intelligence. To explore general spatial imagination capability, we present AirScape, the first world model designed for six-degree-of-freedom aerial agents. AirScape predicts future observation sequences based on current visual inputs and motion intentions. Specifically, we construct a dataset for aerial world model training and testing, which consists of 11k video-intention pairs. This dataset includes first-person-view videos capturing diverse drone actions across a wide range of scenarios, with over 1,000 hours spent annotating the corresponding motion intentions. Then we develop a two-phase schedule to train a foundation model--initially devoid of embodied spatial knowledge--into a world model that is controllable by motion intentions and adheres to physical spatio-temporal constraints. Experimental results demonstrate that AirScape significantly outperforms existing foundation models in 3D spatial imagination capabilities, especially with over a 50% improvement in metrics reflecting motion alignment. The project is available at: https://embodiedcity.github.io/AirScape/.
>
---
#### [replaced 029] Nav-EE: Navigation-Guided Early Exiting for Efficient Vision-Language Models in Autonomous Driving
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2510.01795v2](http://arxiv.org/pdf/2510.01795v2)**

> **作者:** Haibo Hu; Lianming Huang; Xinyu Wang; Yufei Cui; Shangyu Wu; Nan Guan; Chun Jason Xue
>
> **摘要:** Vision-Language Models (VLMs) are increasingly applied in autonomous driving for unified perception and reasoning, but high inference latency hinders real-time deployment. Early-exit reduces latency by terminating inference at intermediate layers, yet its task-dependent nature limits generalization across diverse scenarios. We observe that this limitation aligns with autonomous driving: navigation systems can anticipate upcoming contexts (e.g., intersections, traffic lights), indicating which tasks will be required. We propose Nav-EE, a navigation-guided early-exit framework that precomputes task-specific exit layers offline and dynamically applies them online based on navigation priors. Experiments on CODA, Waymo, and BOSCH show that Nav-EE achieves accuracy comparable to full inference while reducing latency by up to 63.9%. Real-vehicle integration with Autoware Universe further demonstrates reduced inference latency (600ms to 300ms), supporting faster decision-making in complex scenarios. These results suggest that coupling navigation foresight with early-exit offers a viable path toward efficient deployment of large models in autonomous systems. Code and data are available at our anonymous repository: https://anonymous.4open.science/r/Nav-EE-BBC4
>
---
#### [replaced 030] Learning a Shape-adaptive Assist-as-needed Rehabilitation Policy from Therapist-informed Input
- **分类: eess.SY; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2510.04666v2](http://arxiv.org/pdf/2510.04666v2)**

> **作者:** Zhimin Hou; Jiacheng Hou; Xiao Chen; Hamid Sadeghian; Tianyu Ren; Sami Haddadin
>
> **摘要:** Therapist-in-the-loop robotic rehabilitation has shown great promise in enhancing rehabilitation outcomes by integrating the strengths of therapists and robotic systems. However, its broader adoption remains limited due to insufficient safe interaction and limited adaptation capability. This article proposes a novel telerobotics-mediated framework that enables therapists to intuitively and safely deliver assist-as-needed~(AAN) therapy based on two primary contributions. First, our framework encodes the therapist-informed corrective force into via-points in a latent space, allowing the therapist to provide only minimal assistance while encouraging patient maintaining own motion preferences. Second, a shape-adaptive ANN rehabilitation policy is learned to partially and progressively deform the reference trajectory for movement therapy based on encoded patient motion preferences and therapist-informed via-points. The effectiveness of the proposed shape-adaptive AAN strategy was validated on a telerobotic rehabilitation system using two representative tasks. The results demonstrate its practicality for remote AAN therapy and its superiority over two state-of-the-art methods in reducing corrective force and improving movement smoothness.
>
---
#### [replaced 031] HA-VLN 2.0: An Open Benchmark and Leaderboard for Human-Aware Navigation in Discrete and Continuous Environments with Dynamic Multi-Human Interactions
- **分类: cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.14229v3](http://arxiv.org/pdf/2503.14229v3)**

> **作者:** Yifei Dong; Fengyi Wu; Qi He; Zhi-Qi Cheng; Heng Li; Minghan Li; Zebang Cheng; Yuxuan Zhou; Jingdong Sun; Qi Dai; Alexander G Hauptmann
>
> **备注:** 33 pages, 20 figures, website: https://ha-vln-project.vercel.app/
>
> **摘要:** Vision-and-Language Navigation (VLN) has been studied mainly in either discrete or continuous settings, with little attention to dynamic, crowded environments. We present HA-VLN 2.0, a unified benchmark introducing explicit social-awareness constraints. Our contributions are: (i) a standardized task and metrics capturing both goal accuracy and personal-space adherence; (ii) HAPS 2.0 dataset and simulators modeling multi-human interactions, outdoor contexts, and finer language-motion alignment; (iii) benchmarks on 16,844 socially grounded instructions, revealing sharp performance drops of leading agents under human dynamics and partial observability; and (iv) real-world robot experiments validating sim-to-real transfer, with an open leaderboard enabling transparent comparison. Results show that explicit social modeling improves navigation robustness and reduces collisions, underscoring the necessity of human-centric approaches. By releasing datasets, simulators, baselines, and protocols, HA-VLN 2.0 provides a strong foundation for safe, socially responsible navigation research.
>
---
#### [replaced 032] A Real-Time System for Scheduling and Managing UAV Delivery in Urban Areas
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2412.11590v2](http://arxiv.org/pdf/2412.11590v2)**

> **作者:** Han Liu; Tian Liu; Kai Huang
>
> **摘要:** As urban logistics demand continues to grow, UAV delivery has become a key solution to improve delivery efficiency, reduce traffic congestion, and lower logistics costs. However, to fully leverage the potential of UAV delivery networks, efficient swarm scheduling and management are crucial. In this paper, we propose a real-time scheduling and management system based on the ``Airport-Unloading Station" model, aiming to bridge the gap between high-level scheduling algorithms and low-level execution systems. This system, acting as middleware, accurately translates the requirements from the scheduling layer into specific execution instructions, ensuring that the scheduling algorithms perform effectively in real-world environments. Additionally, we implement three collaborative scheduling schemes involving autonomous ground vehicles (AGVs), unmanned aerial vehicles (UAVs), and ground staff to further optimize overall delivery efficiency. Through extensive experiments, this study demonstrates the rationality and feasibility of the proposed management system, providing practical solution for the commercial application of UAVs delivery in urban. Code: https://github.com/chengji253/UAVDeliverySystem
>
---
#### [replaced 033] SHeRLoc: Synchronized Heterogeneous Radar Place Recognition for Cross-Modal Localization
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.15175v2](http://arxiv.org/pdf/2506.15175v2)**

> **作者:** Hanjun Kim; Minwoo Jung; Wooseong Yang; Ayoung Kim
>
> **备注:** 9 pages, 9 figures, accepted to RA-L
>
> **摘要:** Despite the growing adoption of radar in robotics, the majority of research has been confined to homogeneous sensor types, overlooking the integration and cross-modality challenges inherent in heterogeneous radar technologies. This leads to significant difficulties in generalizing across diverse radar data types, with modality-aware approaches that could leverage the complementary strengths of heterogeneous radar remaining unexplored. To bridge these gaps, we propose SHeRLoc, the first deep network tailored for heterogeneous radar, which utilizes RCS polar matching to align multimodal radar data. Our hierarchical optimal transport-based feature aggregation method generates rotationally robust multi-scale descriptors. By employing FFT-similarity-based data mining and adaptive margin-based triplet loss, SHeRLoc enables FOV-aware metric learning. SHeRLoc achieves an order of magnitude improvement in heterogeneous radar place recognition, increasing recall@1 from below 0.1 to 0.9 on a public dataset and outperforming state of-the-art methods. Also applicable to LiDAR, SHeRLoc paves the way for cross-modal place recognition and heterogeneous sensor SLAM. The supplementary materials and source code are available at https://sites.google.com/view/radar-sherloc.
>
---
#### [replaced 034] PeRoI: A Pedestrian-Robot Interaction Dataset for Learning Avoidance, Neutrality, and Attraction Behaviors in Social Navigation
- **分类: cs.HC; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.16481v2](http://arxiv.org/pdf/2503.16481v2)**

> **作者:** Subham Agrawal; Nico Ostermann-Myrau; Nils Dengler; Maren Bennewitz
>
> **摘要:** Robots are increasingly being deployed in public spaces such as shopping malls, sidewalks, and hospitals, where safe and socially aware navigation depends on anticipating how pedestrians respond to their presence. However, existing datasets rarely capture the full spectrum of robot-induced reactions, e.g., avoidance, neutrality, attraction, which limits progress in modeling these interactions. In this paper, we present the Pedestrian-Robot Interaction~(PeRoI) dataset that captures pedestrian motions categorized into attraction, neutrality, and repulsion across two outdoor sites under three controlled conditions: no robot present, with stationary robot, and with moving robot. This design explicitly reveals how pedestrian behavior varies across robot contexts, and we provide qualitative and quantitative comparisons to established state-of-the-art datasets. Building on these data, we propose the Neural Robot Social Force Model~(NeuRoSFM), an extension of the Social Force Model that integrates neural networks to augment inter-human dynamics with learned components and explicit robot-induced forces to better predict pedestrian motion in vicinity of robots. We evaluate NeuRoSFM by generating trajectories on multiple real-world datasets. The results demonstrate improved modeling of pedestrian-robot interactions, leading to better prediction accuracy, and highlight the value of our dataset and method for advancing socially aware navigation strategies in human-centered environments.
>
---
