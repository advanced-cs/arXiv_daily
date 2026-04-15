# 机器人 cs.RO

- **最新发布 48 篇**

- **更新 29 篇**

## 最新发布

#### [new 001] Scalable Trajectory Generation for Whole-Body Mobile Manipulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于移动操作任务，解决协调全身运动轨迹生成的问题。提出AutoMoMa框架，实现大规模、物理有效的轨迹数据生成，提升数据规模与多样性。**

- **链接: [https://arxiv.org/pdf/2604.12565](https://arxiv.org/pdf/2604.12565)**

> **作者:** Yida Niu; Xinhai Chang; Xin Liu; Ziyuan Jiao; Yixin Zhu
>
> **摘要:** Robots deployed in unstructured environments must coordinate whole-body motion -- simultaneously moving a mobile base and arm -- to interact with the physical world. This coupled mobility and dexterity yields a state space that grows combinatorially with scene and object diversity, demanding datasets far larger than those sufficient for fixed-base manipulation. Yet existing acquisition methods, including teleoperation and planning, are either labor-intensive or computationally prohibitive at scale. The core bottleneck is the lack of a scalable pipeline for generating large-scale, physically valid, coordinated trajectory data across diverse embodiments and environments. Here we introduce AutoMoMa, a GPU-accelerated framework that unifies AKR modeling, which consolidates base, arm, and object kinematics into a single chain, with parallelized trajectory optimization. AutoMoMa achieves 5,000 episodes per GPU-hour (over $80\times$ faster than CPU-based baselines), producing a dataset of over 500k physically valid trajectories spanning 330 scenes, diverse articulated objects, and multiple robot embodiments. Prior datasets were forced to compromise on scale, diversity, or kinematic fidelity; AutoMoMa addresses all three simultaneously. Training downstream IL policies further reveals that even a single articulated-object task requires tens of thousands of demonstrations for SOTA methods to reach $\approx 80\%$ success, confirming that data scarcity -- not algorithmic limitations -- has been the binding constraint. AutoMoMa thus bridges high-performance planning and reliable IL-based control, providing the infrastructure previously missing for coordinated mobile manipulation research. By making large-scale, kinematically valid training data practical, AutoMoMa showcases generalizable whole-body robot policies capable of operating in the diverse, unstructured settings of the real world.
>
---
#### [new 002] M2HRI: An LLM-Driven Multimodal Multi-Agent Framework for Personalized Human-Robot Interaction
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决多机器人系统中个体差异与协作问题。通过构建M2HRI框架，赋予机器人个性和记忆，并实现有效协调，提升交互质量与个性化体验。**

- **链接: [https://arxiv.org/pdf/2604.11975](https://arxiv.org/pdf/2604.11975)**

> **作者:** Shaid Hasan; Breenice Lee; Sujan Sarker; Tariq Iqbal
>
> **摘要:** Multi-robot systems hold significant promise for social environments such as homes and hospitals, yet existing multi-robot works treat robots as functionally identical, overlooking how robots individual identity shape user perception and how coordination shapes multi-robot behavior when such individuality is present. To address this, we introduce M2HRI, a multimodal multi-agent framework built on large language models that equips each robot with distinct personality and long-term memory, alongside a coordination mechanism conditioned on these differences. In a controlled user study (n = 105) in a multi-agent human-robot interaction (HRI) scenario, we find that LLM-driven personality traits are significantly distinguishable and enhance interaction quality, long-term memory improves personalization and preference awareness, and centralized coordination significantly reduces overlap while improving overall interaction quality. Together, these results demonstrate that both agent individuality and structured coordination are essential for coherent and socially appropriate multi-agent HRI. Project website and code are available at this https URL.
>
---
#### [new 003] Learning Versatile Humanoid Manipulation with Touch Dreaming
- **分类: cs.RO**

- **简介: 该论文属于人形机器人操作任务，解决接触频繁的复杂操作问题。通过强化学习控制器和触觉预测模型，提升人形机器人在真实环境中的操作能力。**

- **链接: [https://arxiv.org/pdf/2604.13015](https://arxiv.org/pdf/2604.13015)**

> **作者:** Yaru Niu; Zhenlong Fang; Binghong Chen; Shuai Zhou; Revanth Senthilkumaran; Hao Zhang; Bingqing Chen; Chen Qiu; H. Eric Tseng; Jonathan Francis; Ding Zhao
>
> **摘要:** Humanoid robots promise general-purpose assistance, yet real-world humanoid loco-manipulation remains challenging because it requires whole-body stability, dexterous hands, and contact-aware perception under frequent contact changes. In this work, we study dexterous, contact-rich humanoid loco-manipulation. We first develop an RL-based whole-body controller that provides stable lower-body and torso execution during complex manipulation. Built on this controller, we develop a whole-body humanoid data collection system that combines VR-based teleoperation with human-to-humanoid motion mapping, enabling efficient collection of real-world demonstrations. We then propose Humanoid Transformer with Touch Dreaming (HTD), a multimodal encoder--decoder Transformer that models touch as a core modality alongside multi-view vision and proprioception. HTD is trained in a single stage with behavioral cloning augmented by touch dreaming: in addition to predicting action chunks, the policy predicts future hand-joint forces and future tactile latents, encouraging the shared Transformer trunk to learn contact-aware representations for dexterous interaction. Across five contact-rich tasks, Insert-T, Book Organization, Towel Folding, Cat Litter Scooping, and Tea Serving, HTD achieves a 90.9% relative improvement in average success rate over the stronger baseline. Ablation results further show that latent-space tactile prediction is more effective than raw tactile prediction, yielding a 30% relative gain in success rate. These results demonstrate that combining robust whole-body execution, scalable humanoid data collection, and predictive touch-centered learning enables versatile, high-dexterity humanoid manipulation in the real world. Project webpage: this http URL.
>
---
#### [new 004] FastGrasp: Learning-based Whole-body Control method for Fast Dexterous Grasping with Mobile Manipulators
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人抓取任务，解决高速移动机械臂在复杂场景中快速灵巧抓取的问题。提出FastGrasp框架，结合强化学习与触觉反馈，实现高效抓取与实时调整。**

- **链接: [https://arxiv.org/pdf/2604.12879](https://arxiv.org/pdf/2604.12879)**

> **作者:** Heng Tao; Yiming Zhong; Zemin Yang; Yuexin Ma
>
> **摘要:** Fast grasping is critical for mobile robots in logistics, manufacturing, and service applications. Existing methods face fundamental challenges in impact stabilization under high-speed motion, real-time whole-body coordination, and generalization across diverse objects and scenarios, limited by fixed bases, simple grippers, or slow tactile response capabilities. We propose \textbf{FastGrasp}, a learning-based framework that integrates grasp guidance, whole-body control, and tactile feedback for mobile fast grasping. Our two-stage reinforcement learning strategy first generates diverse grasp candidates via conditional variational autoencoder conditioned on object point clouds, then executes coordinated movements of mobile base, arm, and hand guided by optimal grasp selection. Tactile sensing enables real-time grasp adjustments to handle impact effects and object variations. Extensive experiments demonstrate superior grasping performance in both simulation and real-world scenarios, achieving robust manipulation across diverse object geometries through effective sim-to-real transfer.
>
---
#### [new 005] HazardArena: Evaluating Semantic Safety in Vision-Language-Action Models
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型的安全评估任务，旨在解决模型在执行正确动作时可能引发语义风险的问题。工作包括构建HazardArena基准和提出安全增强方法。**

- **链接: [https://arxiv.org/pdf/2604.12447](https://arxiv.org/pdf/2604.12447)**

> **作者:** Zixing Chen; Yifeng Gao; Li Wang; Yunhan Zhao; Yi Liu; Jiayu Li; Xiang Zheng; Zuxuan Wu; Cong Wang; Xingjun Ma; Yu-Gang Jiang
>
> **备注:** Submitted to conference; 12 pages, 8 figures, including supplementary material
>
> **摘要:** Vision-Language-Action (VLA) models inherit rich world knowledge from vision-language backbones and acquire executable skills via action demonstrations. However, existing evaluations largely focus on action execution success, leaving action policies loosely coupled with visual-linguistic semantics. This decoupling exposes a systematic vulnerability whereby correct action execution may induce unsafe outcomes under semantic risk. To expose this vulnerability, we introduce HazardArena, a benchmark designed to evaluate semantic safety in VLAs under controlled yet risk-bearing contexts. HazardArena is constructed from safe/unsafe twin scenarios that share matched objects, layouts, and action requirements, differing only in the semantic context that determines whether an action is unsafe. We find that VLA models trained exclusively on safe scenarios often fail to behave safely when evaluated in their corresponding unsafe counterparts. HazardArena includes over 2,000 assets and 40 risk-sensitive tasks spanning 7 real-world risk categories grounded in established robotic safety standards. To mitigate this vulnerability, we propose a training-free Safety Option Layer that constrains action execution using semantic attributes or a vision-language judge, substantially reducing unsafe behaviors with minimal impact on task performance. We hope that HazardArena highlights the need to rethink how semantic safety is evaluated and enforced in VLAs as they scale toward real-world deployment.
>
---
#### [new 006] PAINT: Partner-Agnostic Intent-Aware Cooperative Transport with Legged Robots
- **分类: cs.RO**

- **简介: 该论文提出PAINT框架，解决腿部机器人协作运输中的伙伴意图感知问题。通过本体反馈实现高效、轻量的协同运输，适用于多种环境和机器人配置。**

- **链接: [https://arxiv.org/pdf/2604.12852](https://arxiv.org/pdf/2604.12852)**

> **作者:** Zhihao Cao; Tianxu An; Chenhao Li; Stelian Coros; Marco Hutter
>
> **摘要:** Collaborative transport requires robots to infer partner intent through physical interaction while maintaining stable loco-manipulation. This becomes particularly challenging in complex environments, where interaction signals are difficult to capture and model. We present PAINT, a lightweight yet efficient hierarchical learning framework for partner-agonistic intent-aware collaborative legged transport that infers partner intent directly from proprioceptive feedback. PAINT decouples intent understanding from terrain-robust locomotion: A high-level policy infers the partner interaction wrench using an intent estimator and a teacher-student training scheme, while a low-level locomotion backbone ensures robust execution. This enables lightweight deployment without external force-torque sensing or payload tracking. Extensive simulation and real-world experiments demonstrate compliant cooperative transport across diverse terrains, payloads, and partners. Furthermore, we show that PAINT naturally scales to decentralized multi-robot transport and transfers across robot embodiments by swapping the underlying locomotion backbone. Our results suggest that proprioceptive signals in payload-coupled interaction provide a scalable interface for partner-agnostic intent-aware collaborative transport.
>
---
#### [new 007] Habitat-GS: A High-Fidelity Navigation Simulator with Dynamic Gaussian Splatting
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于 embodied AI 导航任务，旨在提升模拟环境的视觉真实性和动态人类建模能力。工作包括引入 3D Gaussian Splatting 渲染和 gaussian avatars，增强代理对真实场景的适应能力。**

- **链接: [https://arxiv.org/pdf/2604.12626](https://arxiv.org/pdf/2604.12626)**

> **作者:** Ziyuan Xia; Jingyi Xu; Chong Cui; Yuanhong Yu; Jiazhao Zhang; Qingsong Yan; Tao Ni; Junbo Chen; Xiaowei Zhou; Hujun Bao; Ruizhen Hu; Sida Peng
>
> **备注:** Project page: this https URL
>
> **摘要:** Training embodied AI agents depends critically on the visual fidelity of simulation environments and the ability to model dynamic humans. Current simulators rely on mesh-based rasterization with limited visual realism, and their support for dynamic human avatars, where available, is constrained to mesh representations, hindering agent generalization to human-populated real-world scenarios. We present Habitat-GS, a navigation-centric embodied AI simulator extended from Habitat-Sim that integrates 3D Gaussian Splatting scene rendering and drivable gaussian avatars while maintaining full compatibility with the Habitat ecosystem. Our system implements a 3DGS renderer for real-time photorealistic rendering and supports scalable 3DGS asset import from diverse sources. For dynamic human modeling, we introduce a gaussian avatar module that enables each avatar to simultaneously serve as a photorealistic visual entity and an effective navigation obstacle, allowing agents to learn human-aware behaviors in realistic settings. Experiments on point-goal navigation demonstrate that agents trained on 3DGS scenes achieve stronger cross-domain generalization, with mixed-domain training being the most effective strategy. Evaluations on avatar-aware navigation further confirm that gaussian avatars enable effective human-aware navigation. Finally, performance benchmarks validate the system's scalability across varying scene complexity and avatar counts.
>
---
#### [new 008] Dynamic Modeling and Robust Gait Optimization of a Compliant Worm Robot
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人运动控制任务，旨在解决柔性蠕动机器人在复杂环境中的运动建模与优化问题。通过建立动态模型和能量模型，实现高效且稳定的步态优化。**

- **链接: [https://arxiv.org/pdf/2604.12031](https://arxiv.org/pdf/2604.12031)**

> **作者:** Xinyu Zhou; Yu Mei; Faith Thomson; Christian Luedtke; Xinda Qi; Xiaobo Tan
>
> **摘要:** Worm-inspired robots provide an effective locomotion strategy for constrained environments by combining cyclic body deformation with alternating anchoring. For compliant robots, however, the interaction between deformable anchoring structures and the environment makes predictive modeling and deployable gait optimization challenging. This paper presents an experimentally grounded modeling and optimization framework for a compliant worm robot capable of traversing corrugated pipes. First, a hybrid dynamic locomotion model is derived, in which the robot motion is represented by continuous dynamics within a corrugation groove and discrete switching of anchoring positions between adjacent grooves. A slack-aware actuation model is further introduced to map the commanded gait input to the realized body-length change, and an energy model is developed based on physics and calibrated with empirical power measurement. Based on these models, a multi-objective gait optimization problem is formulated to maximize average speed while minimizing average power. To reduce the fragility of nominal boundary-seeking solutions, a kinematic robustness margin is introduced into the anchoring-transition conditions, leading to a margin-based robust gait optimization framework. Experimental results show that the proposed framework captures the dominant locomotion and energy-consumption behavior of the robot over the tested conditions, and enables robust gait optimization for achieving speed-power trade-off.
>
---
#### [new 009] Uncertainty Guided Exploratory Trajectory Optimization for Sampling-Based Model Predictive Control
- **分类: cs.RO**

- **简介: 该论文属于轨迹优化任务，旨在解决采样方法易陷入局部最优的问题。提出UGE-TO算法，通过不确定性引导增强轨迹多样性与探索能力，提升模型预测控制的效率与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.12149](https://arxiv.org/pdf/2604.12149)**

> **作者:** O. Goktug Poyrazoglu; Yukang Cao; Rahul Moorthy; Volkan Isler
>
> **备注:** This paper has been accepted for presentation at the IEEE International Conference on Robotics and Automation (ICRA) 2026
>
> **摘要:** Trajectory optimization depends heavily on initialization. In particular, sampling-based approaches are highly sensitive to initial solutions, and limited exploration frequently leads them to converge to local minima in complex environments. We present Uncertainty Guided Exploratory Trajectory Optimization (UGE-TO), a trajectory optimization algorithm that generates well-separated samples to achieve a better coverage of the configuration space. UGE-TO represents trajectories as probability distributions induced by uncertainty ellipsoids. Unlike sampling-based approaches that explore only in the action space, this representation captures the effects of both system dynamics and action selection. By incorporating the impact of dynamics, in addition to the action space, into our distributions, our method enhances trajectory diversity by enforcing distributional separation via the Hellinger distance between them. It enables a systematic exploration of the configuration space and improves robustness against local minima. Further, we present UGE-MPC, which integrates UGE-TO into sampling-based model predictive controller methods. Experiments demonstrate that UGE-MPC achieves higher exploration and faster convergence in trajectory optimization compared to baselines under the same sampling budget, achieving 72.1% faster convergence in obstacle-free environments and 66% faster convergence with a 6.7% higher success rate in the cluttered environment compared to the best-performing baseline. Additionally, we validate the approach through a range of simulation scenarios and real-world experiments. Our results indicate that UGE-MPC has higher success rates and faster convergence, especially in environments that demand significant deviations from nominal trajectories to avoid failures. The project and code are available at this https URL.
>
---
#### [new 010] RMGS-SLAM: Real-time Multi-sensor Gaussian Splatting SLAM
- **分类: cs.RO**

- **简介: 该论文属于SLAM任务，旨在解决大场景下实时定位与建图中的低延迟、高精度和全局一致性问题。提出一种融合LiDAR、IMU和视觉的3DGS框架，实现高效、高质量的实时映射。**

- **链接: [https://arxiv.org/pdf/2604.12942](https://arxiv.org/pdf/2604.12942)**

> **作者:** Dongen Li; Yi Liu; Junqi Liu; Zewen Sun; Zefan Huang; Shuo Sun; Jiahui Liu; Chengran Yuan; Hongliang Guo; Francis E.H. Tay; Marcelo H. Ang Jr
>
> **摘要:** Real-time 3D Gaussian splatting (3DGS)-based Simultaneous Localization and Mapping (SLAM) in large-scale real-world environments remains challenging, as existing methods often struggle to jointly achieve low-latency pose estimation, 3D Gaussian reconstruction in step with incoming sensor streams, and long-term global consistency. In this paper, we present a tightly coupled LiDAR-Inertial-Visual (LIV) 3DGS-based SLAM framework for real-time pose estimation and photorealistic mapping in large-scale real-world scenes. The system executes state estimation and 3D Gaussian primitive initialization in parallel with global Gaussian optimization, thereby enabling continuous dense mapping. To improve Gaussian initialization quality and accelerate optimization convergence, we introduce a cascaded strategy that combines feed-forward predictions with voxel-based principal component analysis (voxel-PCA) geometric priors. To enhance global consistency in large scenes, we further perform loop closure directly on the optimized global Gaussian map by estimating loop constraints through Gaussian-based Generalized Iterative Closest Point (GICP) registration, followed by pose-graph optimization. In addition, we collected challenging large-scale looped outdoor SLAM sequences with hardware-synchronized LiDAR-camera-IMU and ground-truth trajectories to support realistic and comprehensive evaluation. Extensive experiments on both public datasets and our dataset demonstrate that the proposed method achieves a strong balance among real-time efficiency, localization accuracy, and rendering quality across diverse and challenging real-world scenes.
>
---
#### [new 011] OVAL: Open-Vocabulary Augmented Memory Model for Lifelong Object Goal Navigation
- **分类: cs.RO**

- **简介: 该论文属于对象导航任务，旨在解决长期记忆与持续目标导航的问题。提出OVAL框架，通过记忆描述符和概率探索策略提升导航效率与准确性。**

- **链接: [https://arxiv.org/pdf/2604.12872](https://arxiv.org/pdf/2604.12872)**

> **作者:** Jiahua Pei; Yi Liu; Guoping Pan; Yuanhao Jiang; Houde Liu; Xueqian Wang
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Object Goal Navigation (ObjectNav) refers to an agent navigating to an object in an unseen environment, which is an ability often required in the accomplishment of complex tasks. While existing methods demonstrate proficiency in isolated single object navigation, their limitations emerge in the restricted applicability of lifelong memory representations, which ultimately hinders effective navigation toward continual targets over extended periods. To address this problem, we propose OVAL, a novel lifelong open-vocabulary memory framework, which enables efficient and precise execution of long-term navigation in semantically open tasks. Within this framework, we introduce memory descriptors to facilitate structured management of the memory model. Additionally, we propose a novel probability-based exploration strategy, utilizing a multi-value frontier scoring to enhance lifelong exploration efficiency. Extensive experiments demonstrate the efficiency and robustness of the proposed system.
>
---
#### [new 012] Designing for Error Recovery in Human-Robot Interaction
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机交互任务，旨在解决机器人系统错误恢复问题。通过分析核设施机械臂案例，探讨如何设计具备自我检测与恢复能力的AI系统。**

- **链接: [https://arxiv.org/pdf/2604.12473](https://arxiv.org/pdf/2604.12473)**

> **作者:** Christopher D. Wallbridge; Erwin Jose Lopez Pulgarin
>
> **摘要:** This position paper looks briefly at the way we attempt to program robotic AI systems. Many AI systems are based on the idea of trying to improve the performance of one individual system to beyond so-called human baselines. However, these systems often look at one shot and one-way decisions, whereas the real world is more continuous and interactive. Humans, however, are often able to recover from and learn from errors - enabling a much higher rate of success. We look at the challenges of building a system that can detect/recover from its own errors, using the example of robotic nuclear gloveboxes as a use case to help illustrate examples. We then go on to talk about simple starting designs.
>
---
#### [new 013] Contextual Multi-Task Reinforcement Learning for Autonomous Reef Monitoring
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主水下监测任务，旨在解决水下环境动态不确定带来的控制难题。通过上下文多任务强化学习，提升策略的泛化与复用能力。**

- **链接: [https://arxiv.org/pdf/2604.12645](https://arxiv.org/pdf/2604.12645)**

> **作者:** Melvin Laux; Yi-Ling Liu; Rina Alo; Sören Töpper; Mariela De Lucas Alvarez; Frank Kirchner; Rebecca Adam
>
> **备注:** To be published in IEEE OCEANS 2026 (Sanya) conference proceedings
>
> **摘要:** Although autonomous underwater vehicles promise the capability of marine ecosystem monitoring, their deployment is fundamentally limited by the difficulty of controlling vehicles under highly uncertain and non-stationary underwater dynamics. To address these challenges, we employ a data-driven reinforcement learning approach to compensate for unknown dynamics and task this http URL single-task reinforcement learning has a tendency to overfit the training environment, thus, limit the long-term usefulness of the learnt policy. Hence, we propose to use a contextual multi-task reinforcement learning paradigm instead, allowing us to learn controllers that can be reused for various tasks, e.g., detecting oysters in one reef and detecting corals in another. We evaluate whether contextual multi-task reinforcement learning can efficiently learn robust and generalisable control policies for autonomous underwater reef monitoring. We train a single context-dependent policy that is able to solve multiple related monitoring tasks in a simulated reef environment in HoloOcean. In our experiments, we empirically evaluate the contextual policies regarding sample-efficiency, zero-shot generalisation to unseen tasks, and robustness to varying water currents. By utilising multi-task reinforcement learning, we aim to improve the training effectiveness, as well as the reusability of learnt policies to take a step towards more sustainable procedures in autonomous reef monitoring.
>
---
#### [new 014] Asymptotically Stable Gait Generation and Instantaneous Walkability Determination for Planar Almost Linear Biped with Knees
- **分类: cs.RO**

- **简介: 该论文属于机器人步态生成与行走稳定性分析任务，解决平面双足机器人稳定步态生成和即时行走可行性判断问题。通过建立线性化模型实现快速计算。**

- **链接: [https://arxiv.org/pdf/2604.12274](https://arxiv.org/pdf/2604.12274)**

> **作者:** Fumihiko Asano; Ning Lei; Taiki Sedoguchi
>
> **备注:** Accepted for presentation at the IEEE International Conference on Robotics and Automation (ICRA), 2026. This version includes a correction to a typographical error in one equation
>
> **摘要:** A class of planar bipedal robots with unique mechanical properties has been proposed, where all links are balanced around the hip joint, preventing natural swinging motion due to gravity. A common property of their equations of motion is that the inertia matrix is a constant matrix, there are no nonlinear velocity terms, and the gravity term contains simple nonlinear terms. By performing a Taylor expansion of the gravity term and making a linear approximation, it is easy to derive a linearized model, and calculations for future states or walkability determination can be performed instantaneously without the need for numerical integration. This paper extends the method to a planar biped robot model with knees. First, we derive the equations of motion, constraint conditions, and inelastic collisions for a planar 6-DOF biped robot, design its control system, and numerically generate a stable bipedal gait on a horizontal plane. Next, we reduce the equations of motion to a 3-DOF model, and derive a linearized model by approximating the gravity term as linear around the expansion point for the thigh frame angle. Through numerical simulations, we demonstrate that calculations for future states and walkability determination can be completed in negligible time. By applying control inputs to the obtained model, performing state-space realization, and then discretizing it, instantaneous walkability determination through iterative calculation becomes possible. Through detailed gait analysis, we discuss how the knee joint flexion angle and the expansion point affect the accuracy of the linear approximation, and the issues that arise when descending a small step.
>
---
#### [new 015] Complementarity by Construction: A Lie-Group Approach to Solving Quadratic Programs with Linear Complementarity Constraints
- **分类: cs.RO**

- **简介: 该论文研究机器人中的混合连续与离散问题，解决LCQPs求解难题。通过Lie群方法构造满足互补约束的优化框架，提出新求解器Marble，有效处理机器人应用中的复杂约束问题。**

- **链接: [https://arxiv.org/pdf/2604.11991](https://arxiv.org/pdf/2604.11991)**

> **作者:** Arun L. Bishop; Micah I. Reich; Zachary Manchester
>
> **摘要:** Many problems in robotics require reasoning over a mix of continuous dynamics and discrete events, such as making and breaking contact in manipulation and locomotion. These problems are locally well modeled by linear complementarity quadratic programs (LCQPs), an extension to QPs that introduce complementarity constraints. While very expressive, LCQPs are non-convex, and few solvers exist for computing good local solutions for use in planning pipelines. In this work, we observe that complementarity constraints form a Lie group under infinitesimal relaxation, and leverage this structure to perform on-manifold optimization. We introduce a retraction map that is numerically well behaved, and use it to parameterize the constraints so that they are satisfied by construction. The resulting solver avoids many of the classical issues with complementarity constraints. We provide an open-source solver, Marble, that is implemented in C++ with Julia and Python bindings. We demonstrate that Marble is competitive on a suite of benchmark problems, and solves a number of robotics problems where existing approaches fail to converge.
>
---
#### [new 016] Reliability-Guided Depth Fusion for Glare-Resilient Navigation Costmaps
- **分类: cs.RO**

- **简介: 该论文属于室内导航任务，解决反射表面导致的深度测量错误问题。通过建立可靠性地图和融合机制，提升成本图准确性与导航鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.12753](https://arxiv.org/pdf/2604.12753)**

> **作者:** Shang-En Tsai; Wei-Cheng Sun
>
> **摘要:** Specular glare on reflective floors and glass surfaces frequently corrupts RGB-D depth measurements, producing holes and spikes that accumulate as persistent phantom obstacles in occupancy-grid costmaps. This paper proposes a glare-resilient costmap construction method based on explicit depth-reliability modeling. A lightweight Depth Reliability Map (DRM) estimator predicts per-pixel measurement trustworthiness under specular interference, and a Reliability-Guided Fusion (RGF) mechanism uses this signal to modulate occupancy updates before corrupted measurements are accumulated into the map. Experiments on a real mobile robotic platform equipped with an Intel RealSense D435 and a Jetson Orin Nano show that the proposed method substantially reduces false obstacle insertion and improves free-space preservation under real reflective-floor and glass-surface conditions, while introducing only modest computational overhead. These results indicate that treating glare as a measurement-reliability problem provides a practical and lightweight solution for improving costmap correctness and navigation robustness in safety-critical indoor environments.
>
---
#### [new 017] Ternary Logic Encodings of Temporal Behavior Trees with Application to Control Synthesis
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于控制合成任务，旨在解决行为树的正确性验证问题。通过引入三值时序逻辑，建立混合整数编码，实现可靠控制策略。**

- **链接: [https://arxiv.org/pdf/2604.12092](https://arxiv.org/pdf/2604.12092)**

> **作者:** Ryan Matheu; John S. Baras; Calin Belta
>
> **备注:** 8 pages, 4 figures. This work has been submitted to the IEEE for possible publication
>
> **摘要:** Behavior Trees (BTs) provide designers an intuitive graphical interface to construct long-horizon plans for autonomous systems. To ensure their correctness and safety, rigorous formal models and verification techniques are essential. Temporal BTs (TBTs) offer a promising approach by leveraging existing temporal logic formalisms to specify and verify the executions of BTs. However, this analysis is currently limited to offline post hoc analysis and trace repair. In this paper, we reformulate TBTs using a ternary-valued Signal Temporal Logic (STL) amenable for control synthesis. Ternary logic introduces a third truth value \textit{Unknown}, formally capturing cases where a trajectory has neither fully satisfied or dissatisfied a specification. We propose mixed-integer linear encodings for partial trajectory STL and TBTs over ternary logic allowing for correct-by-construction control strategies for linear dynamical systems via mixed-integer optimization. We demonstrate the utility of our framework by solving optimal control problems.
>
---
#### [new 018] Evolving the Complete Muscle: Efficient Morphology-Control Co-design for Musculoskeletal Locomotion
- **分类: cs.RO**

- **简介: 该论文属于机器人学领域，解决肌肉骨骼系统形态与控制协同设计问题。通过进化方法优化肌肉力量、速度和刚度，提升机器人运动性能。**

- **链接: [https://arxiv.org/pdf/2604.12855](https://arxiv.org/pdf/2604.12855)**

> **作者:** Lidong Sun; Wentao Zhao; Ye Wang; Huaping Liu; Fuchun Sun
>
> **摘要:** Musculoskeletal robots offer intrinsic compliance and flexibility, providing a promising paradigm for versatile locomotion. However, existing research typically relies on models with fixed muscle physiological parameters. This static physical setting fails to accommodate the diverse dynamic demands of complex tasks, inherently limiting the robot's performance upper bound. In this work, we focus on the morphology and control co-design of musculoskeletal systems. Unlike previous studies that optimize single physiological attributes such as stiffness, we introduce a Complete Musculoskeletal Morphological Evolution Space that simultaneously evolves muscle strength, velocity, and stiffness. To overcome the exponential expansion of the exploration space caused by this comprehensive evolution, we propose Spectral Design Evolution (SDE), a high-efficiency co-optimization framework. By integrating a bilateral symmetry prior with Principal Component Analysis (PCA), SDE projects complex muscle parameters onto a low-dimensional spectral manifold, enabling efficient morphological exploration. Evaluated on the MyoSuite framework across four tasks (Walk, Stair, Hilly, and Rough terrains), our method demonstrates superior learning efficiency and locomotion stability compared to fixed-morphology and standard evolutionary baselines.
>
---
#### [new 019] A Foot Resistive Force Model for Legged Locomotion on Muddy Terrains
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，旨在解决泥地腿部机器人运动难题。提出了一种足-泥相互作用的阻力模型，并设计了新型可变形足部以提高移动性和能效。**

- **链接: [https://arxiv.org/pdf/2604.12006](https://arxiv.org/pdf/2604.12006)**

> **作者:** Xunjie Chen; Liuyin Wang; Xinyan Huang; Jerry Shan; Yantao Shen; Jingang Yi
>
> **备注:** IEEE/ASME Transactions on Mechatronics (under review)
>
> **摘要:** Legged robots face significant challenges in moving and navigating on deformable and highly yielding terrain such as mud. We present a resistive force model for legged foot-mud interactions. The model captures rheological behaviors such as visco-elasticity, thixotropy of the mud suspension and retractive suction. One attractive property of this new model lies in its effective, uniform formulation to provide underlying physical interpretation and accurate resistive force predictions. We further take advantage of the resistive force model to design a new morphing robotic foot for effective and efficient legged locomotion. We conduct extensive experiments to validate the force model, and the results demonstrate that the morphing foot enhances not only the locomotion mobility but also energy-efficiency of walking in mud. The new resistive force model can be further used to develop data-driven simulation and locomotion control of legged robots on muddy terrains.
>
---
#### [new 020] VULCAN: Vision-Language-Model Enhanced Multi-Agent Cooperative Navigation for Indoor Fire-Disaster Response
- **分类: cs.RO**

- **简介: 该论文属于多智能体协作导航任务，旨在解决室内火灾环境下自主搜救的感知与规划问题。提出VULCAN框架，结合多模态感知和视觉语言模型，提升灾害场景下的导航性能。**

- **链接: [https://arxiv.org/pdf/2604.12831](https://arxiv.org/pdf/2604.12831)**

> **作者:** Shengding Liu; Qiben Yan
>
> **备注:** INFOCOM EIN Workshop 2026
>
> **摘要:** Indoor fire disasters pose severe challenges to autonomous search and rescue due to dense smoke, high temperatures, and dynamically evolving indoor environments. In such time-critical scenarios, multi-agent cooperative navigation is particularly useful, as it enables faster and broader exploration than single-agent approaches. However, existing multi-agent navigation systems are primarily vision-based and designed for benign indoor settings, leading to significant performance degradation under fire-driven dynamic conditions. In this paper, we present VULCAN, a multi-agent cooperative navigation framework based on multi-modal perception and vision-language models (VLMs), tailored for indoor fire disaster response. We extend the Habitat-Matterport3D benchmark by simulating physically realistic fire scenarios, including smoke diffusion, thermal hazards, and sensor degradation. We evaluate representative multi-agent cooperative navigation baselines under both normal and fire-driven environments. Our results reveal critical failure modes of existing methods in fire scenarios and underscore the necessity of robust perception and hazard-aware planning for reliable multi-agent search and rescue.
>
---
#### [new 021] RACF: A Resilient Autonomous Car Framework with Object Distance Correction
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶感知任务，旨在解决传感器失效或攻击导致的距离估计错误问题。通过融合多传感器数据，提出RACF框架提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.12418](https://arxiv.org/pdf/2604.12418)**

> **作者:** Chieh Tsai; Hossein Rastgoftar; Salim Hariri
>
> **备注:** 8 pages, 9 figures, 5 tables. Submitted manuscript to IROS 2026
>
> **摘要:** Autonomous vehicles are increasingly deployed in safety-critical applications, where sensing failures or cyberphysical attacks can lead to unsafe operations resulting in human loss and/or severe physical damages. Reliable real-time perception is therefore critically important for their safe operations and acceptability. For example, vision-based distance estimation is vulnerable to environmental degradation and adversarial perturbations, and existing defenses are often reactive and too slow to promptly mitigate their impacts on safe operations. We present a Resilient Autonomous Car Framework (RACF) that incorporates an Object Distance Correction Algorithm (ODCA) to improve perception-layer robustness through redundancy and diversity across a depth camera, LiDAR, and physics-based kinematics. Within this framework, when obstacle distance estimation produced by depth camera is inconsistent, a cross-sensor gate activates the correction algorithm to fix the detected inconsistency. We have experiment with the proposed resilient car framework and evaluate its performance on a testbed implemented using the Quanser QCar 2 platform. The presented framework achieved up to 35% RMSE reduction under strong corruption and improves stop compliance and braking latency, while operating in real time. These results demonstrate a practical and lightweight approach to resilient perception for safety-critical autonomous driving
>
---
#### [new 022] Unveiling the Surprising Efficacy of Navigation Understanding in End-to-End Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主驾驶任务，解决系统过度依赖局部感知而忽视全局导航的问题。提出SNG框架和SNG-VLA模型，融合全局与局部规划，提升导航性能。**

- **链接: [https://arxiv.org/pdf/2604.12208](https://arxiv.org/pdf/2604.12208)**

> **作者:** Zhihua Hua; Junli Wang; Pengfei LI; Qihao Jin; Bo Zhang; Kehua Sheng; Yilun Chen; Zhongxue Gan; Wenchao Ding
>
> **备注:** 8 pages, 6 figures. ICRA 2026. Code available at this https URL
>
> **摘要:** Global navigation information and local scene understanding are two crucial components of autonomous driving systems. However, our experimental results indicate that many end-to-end autonomous driving systems tend to over-rely on local scene understanding while failing to utilize global navigation information. These systems exhibit weak correlation between their planning capabilities and navigation input, and struggle to perform navigation-following in complex scenarios. To overcome this limitation, we propose the Sequential Navigation Guidance (SNG) framework, an efficient representation of global navigation information based on real-world navigation patterns. The SNG encompasses both navigation paths for constraining long-term trajectories and turn-by-turn (TBT) information for real-time decision-making logic. We constructed the SNG-QA dataset, a visual question answering (VQA) dataset based on SNG that aligns global and local planning. Additionally, we introduce an efficient model SNG-VLA that fuses local planning with global planning. The SNG-VLA achieves state-of-the-art performance through precise navigation information modeling without requiring auxiliary loss functions from perception tasks. Project page: SNG-VLA
>
---
#### [new 023] Machine Learning-Based Real-Time Detection of Compensatory Trunk Movements Using Trunk-Wrist Inertial Measurement Units
- **分类: cs.RO**

- **简介: 该论文属于运动检测任务，旨在解决实时检测中风后代偿性躯干运动的问题。通过双惯性测量单元和机器学习方法，实现可靠、实时的CTM检测。**

- **链接: [https://arxiv.org/pdf/2604.12591](https://arxiv.org/pdf/2604.12591)**

> **作者:** Jannis Gabler; Clément Lhoste; Max Quast; Laura Mayrhuber; Andrea Ronco; Olivier Lambercy; Paulius Viskaitis; Dane Donegan
>
> **备注:** This manuscript has been submitted to IEEE Transactions on Neural Systems and Rehabilitation Engineering for possible publication. This version is a preprint and has not undergone peer review
>
> **摘要:** Compensatory trunk movements (CTMs) are commonly observed after stroke and can lead to maladaptive movement patterns, limiting targeted training of affected structures. Objective, continuous detection of CTMs during therapy and activities of daily living remains challenging due to the typically complex measurements setups required, as well as limited applicability for real-time use. This study investigates whether a two-inertial measurement unit configuration enables reliable, real-time CTM detection using machine learning. Data were collected from ten able-bodied participants performing activities of daily living under simulated impairment conditions (elbow brace restricting flexion-extension, resistance band inducing flexor-synergy-like patterns), with synchronized optical motion capture (OMC) and manually annotated video recordings serving as reference. A systematic location-reduction analysis using OMC identified wrist and trunk kinematics as a minimal yet sufficient set of anatomical sensing locations. Using an extreme gradient boosting classifier (XGBoost) evaluated with leave-one-subject-out cross-validation, our two-IMU model achieved strong discriminative performance (macro-F1 = 0.80 +/- 0.07, MCC = 0.73 +/- 0.08; ROC-AUC > 0.93), with performance comparable to an OMC-based model and prediction timing suitable for real-time applications. Explainability analysis revealed dominant contributions from trunk dynamics and wrist-trunk interaction features. In preliminary evaluation using recordings from four participants with neurological conditions, the model retained good discriminative capability (ROC-AUC ~ 0.78), but showed reduced and variable threshold-dependent performance, highlighting challenges in clinical generalization. These results support sparse wearable sensing as a viable pathway toward scalable, real-time monitoring of CTMs during therapy and daily living.
>
---
#### [new 024] Frequency-aware Decomposition Learning for Sensorless Wrench Forecasting on a Vibration-rich Hydraulic Manipulator
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于无传感器力/力矩预测任务，解决高速振动环境下 wrench 预测问题。提出 FDNet 模型，通过频域分解和自适应滤波提升预测精度。**

- **链接: [https://arxiv.org/pdf/2604.12905](https://arxiv.org/pdf/2604.12905)**

> **作者:** Hyeonbeen Lee; Min-Jae Jung; Tae-Kyeong Yeu; Jong-Boo Han; Daegil Park; Jin-Gyun Kim
>
> **备注:** 11 pages, 6 figures, submitted to IEEE/ASME Transactions on Mechatronics
>
> **摘要:** Force and torque (F/T) sensing is critical for robot-environment interaction, but physical F/T sensors impose constraints in size, cost, and fragility. To mitigate this, recent studies have estimated force/wrench sensorlessly from robot internal states. While existing methods generally target relatively slow interactions, tasks involving rapid interactions, such as grinding, can induce task-critical high-frequency vibrations, and estimation in such robotic settings remains underexplored. To address this gap, we propose a Frequency-aware Decomposition Network (FDN) for short-term forecasting of vibration-rich wrench from proprioceptive history. FDN predicts spectrally decomposed wrench with asymmetric deterministic and probabilistic heads, modeling the high-frequency residual as a learned conditional distribution. It further incorporates frequency-awareness to adaptively enhance input spectra with learned filtering and impose a frequency-band prior on the outputs. We pretrain FDN on a large-scale open-source robot dataset and transfer the learned proprioception-to-wrench representation to the downstream. On real-world grinding excavation data from a 6-DoF hydraulic manipulator and under a delayed estimation setting, FDN outperforms baseline estimators and forecasters in the high-frequency band and remains competitive in the low-frequency band. Transfer learning provides additional gains, suggesting the potential of large-scale pretraining and transfer learning for robotic wrench estimation. Code and data will be made available upon acceptance.
>
---
#### [new 025] Tree Learning: A Multi-Skill Continual Learning Framework for Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文属于多技能持续学习任务，旨在解决人形机器人在扩展新技能时的灾难性遗忘问题。提出Tree Learning框架，通过参数继承和多模态适应机制实现高效技能学习与切换。**

- **链接: [https://arxiv.org/pdf/2604.12909](https://arxiv.org/pdf/2604.12909)**

> **作者:** Yifei Yan; Linqi Ye
>
> **摘要:** As reinforcement learning for humanoid robots evolves from single-task to multi-skill paradigms, efficiently expanding new skills while avoiding catastrophic forgetting has become a key challenge in embodied intelligence. Existing approaches either rely on complex topology adjustments in Mixture-of-Experts (MoE) models or require training extremely large-scale models, making lightweight deployment difficult. To address this, we propose Tree Learning, a multi-skill continual learning framework for humanoid robots. The framework adopts a root-branch hierarchical parameter inheritance mechanism, providing motion priors for branch skills through parameter reuse to fundamentally prevent catastrophic forgetting. A multi-modal feedforward adaptation mechanism combining phase modulation and interpolation is designed to support both periodic and aperiodic motions. A task-level reward shaping strategy is also proposed to accelerate skill convergence. Unity-based simulation experiments show that, in contrast to simultaneous multi-task training, Tree Learning achieves higher rewards across various representative locomotion skills while maintaining a 100% skill retention rate, enabling seamless multi-skill switching and real-time interactive control. We further validate the performance and generalization capability of Tree Learning on two distinct Unity-simulated tasks: a Super Mario-inspired interactive scenario and autonomous navigation in a classical Chinese garden environment.
>
---
#### [new 026] 3DRO: Lidar-level SE(3) Direct Radar Odometry Using a 2D Imaging Radar and a Gyroscope
- **分类: cs.RO**

- **简介: 该论文属于机器人状态估计任务，解决2D雷达数据在SE(3)空间中的定位问题。通过融合惯性测量，实现高精度的三维里程计。**

- **链接: [https://arxiv.org/pdf/2604.12027](https://arxiv.org/pdf/2604.12027)**

> **作者:** Cedric Le Gentil; Daniil Lisus; Timothy D. Barfoot
>
> **摘要:** Recently, the robotics community has regained interest in radar-based perception and state estimation. A 2D imaging radar provides dense 360deg information about the environment. Despite the radar antenna's cone of emission and reception, the collected data is generally assumed to be limited to the plane orthogonal to the radar's spinning axis. Accordingly, most methods based on 2D imaging radars only perform SE(2) state estimation. This paper presents 3DRO, an extension of the SE(2) Direct Radar Odometry (DRO) framework to perform state estimation in SE(3). While still assuming planarity of the data through DRO's 2D velocity estimates, it integrates 3D gyroscope measurements over SO(3) to estimate SE(3) ego motion. While simple, this approach provides lidar-level odometry accuracy as demonstrated using 643km of data from the Boreas-RT dataset.
>
---
#### [new 027] MVAdapt: Zero-Shot Multi-Vehicle Adaptation for End-to-End Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自动驾驶任务，解决车辆域差距问题。通过MVAdapt框架，将驾驶策略与车辆物理特性结合，提升模型在不同车辆上的泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.11854](https://arxiv.org/pdf/2604.11854)**

> **作者:** Haesung Oh; Jaeheung Park
>
> **摘要:** End-to-End (E2E) autonomous driving models are usually trained and evaluated with a fixed ego-vehicle, even though their driving policy is implicitly tied to vehicle dynamics. When such a model is deployed on a vehicle with different size, mass, or drivetrain characteristics, its performance can degrade substantially; we refer to this problem as the vehicle-domain gap. To address it, we propose MVAdapt, a physics-conditioned adaptation framework for multi-vehicle E2E driving. MVAdapt combines a frozen TransFuser++ scene encoder with a lightweight physics encoder and a cross-attention module that conditions scene features on vehicle properties before waypoint decoding. In the CARLA Leaderboard 1.0 benchmark, MVAdapt improves over naive transfer and multi-embodiment adaptation baselines on both in-distribution and unseen vehicles. We further show two complementary behaviors: strong zero-shot transfer on many unseen vehicles, and data-efficient few-shot calibration for severe physical outliers. These results suggest that explicitly conditioning E2E driving policies on vehicle physics is an effective step toward more transferable autonomous driving models. All codes are available at this https URL
>
---
#### [new 028] Defining and Evaluation Method for External Human-Machine Interfaces
- **分类: cs.RO**

- **简介: 该论文属于人机交互任务，旨在解决自动驾驶车辆与道路其他参与者沟通的问题。提出了一种223项的评估方法，用于比较不同eHMI方案。**

- **链接: [https://arxiv.org/pdf/2604.12293](https://arxiv.org/pdf/2604.12293)**

> **作者:** Jose Gonzalez-Belmonte; Jaerock Kwon
>
> **备注:** 62 pages, 8 figures, 26 tables,
>
> **摘要:** As the number of fatalities involving Autonomous Vehicles increase, the need for a universal method of communicating between vehicles and other agents on the road has also increased. Over the past decade, numerous proposals of external Human-Machine Interfaces (eHMIs) have been brought forward with the purpose of bridging this communication gap, with none yet to be determined as the ideal one. This work proposes a universal evaluation method conformed of 223 questions to objectively evaluate and compare different proposals and arrive at a conclusion. The questionnaire is divided into 7 categories that evaluate different aspects of any given proposal that uses eHMIs: ease of standardization, cost effectiveness, accessibility, ease of understanding, multifacetedness in communication, positioning, and readability. In order to test the method it was used on four existing proposals, plus a baseline using only kinematic motions, in order to both exemplify the application of the evaluation method and offer a baseline score for future comparison. The result of this testing suggests that the ideal method of machine-human communication is a combination of intentionally-designed vehicle kinematics and distributed well-placed text-based displays, but it also reveals knowledge gaps in the readability of eHMIs and the speed at which different observers may learn their meaning. This paper proposes future work related to these uncertainties, along with future testing with the proposed method.
>
---
#### [new 029] Robotic Nanoparticle Synthesis via Solution-based Processes
- **分类: cs.RO**

- **简介: 该论文属于实验室自动化任务，旨在解决长周期、多步骤合成中的机器人操作问题。通过螺钉几何运动表示和示教编程，实现纳米粒子的自主合成。**

- **链接: [https://arxiv.org/pdf/2604.12169](https://arxiv.org/pdf/2604.12169)**

> **作者:** Dasharadhan Mahalingam; Michael Gallagher; Nilanjan Chakraborty; Stanislaus S. Wong
>
> **摘要:** We present a screw geometry-based manipulation planning framework for the robotic automation of solution-based synthesis, exemplified through the preparation of gold and magnetite nanoparticles. The synthesis protocols are inherently long-horizon, multi-step tasks, requiring skills such as pick-and-place, pouring, turning a knob, and periodic visual inspection to detect reaction completion. A central challenge is that some skills, notably pouring, transferring containers with solutions, and turning a knob, impose geometric and kinematic constraints on the end-effector motion. To address this, we use a programming by demonstration paradigm where the constraints can be extracted from a single demonstration. This combination of screw-based motion representation and demonstration-driven specification enables domain experts, such as chemists, to readily adapt and reprogram the system for new experimental protocols and laboratory setups without requiring expertise in robotics or motion planning. We extract sequences of constant screws from demonstrations, which compactly encode the motion constraints while remaining coordinate-invariant. This representation enables robust generalization across variations in grasp placement and allows parameterized reuse of a skill learned from a single example. By composing these screw-parameterized primitives according to the synthesis protocol, the robot autonomously generates motion plans that execute the complete experiment over repeated runs. Our results highlight that screw-theoretic planning, combined with programming by demonstration, provides a rigorous and generalizable foundation for long-horizon laboratory automation, thereby enabling fundamental kinematics to have a translational impact on the use of robots in developing scalable solution-based synthesis protocols.
>
---
#### [new 030] Bipedal-Walking-Dynamics Model on Granular Terrains
- **分类: cs.RO**

- **简介: 该论文属于机器人行走任务，旨在解决 bipedal 机器人在沙地等颗粒地形上的运动不稳定问题。通过建立新的动力学模型，预测足部沉降与滑移，提升行走效率与控制精度。**

- **链接: [https://arxiv.org/pdf/2604.11981](https://arxiv.org/pdf/2604.11981)**

> **作者:** Xunjie Chen; Xinyan Huang; Peter Shan; Jingang Yi; Tao Liu
>
> **备注:** Accepted paper in ICRA 2026
>
> **摘要:** Bipeds have demonstrated high agility and mobility in unstructured environments such as sand. The yielding of such granular media brings significant sinkage and slip of the bipedal feet, leading to uncertainty and instability of walking locomotion. We present a new dynamics-modeling approach to capture and predict bipedal-walking locomotion on granular media. A dynamic foot-terrain interaction model is integrated to compute the ground reaction force (GRF). The proposed granular dynamic model has three additional degree-of-freedom (DoF) to estimate foot sinkage and slip that are critical to capturing robot-walking kinematics and kinetics such as cost of transport (CoT). Using the new model, we analyze bipedal kinetics, CoT, and foot-terrain rolling and intrusion affects. Experiments are conducted using a biped robotic walker on sand to validate the proposed dynamic model with robot-gait profiles, media-intrusion prediction, and GRF estimations. This new dynamics model can further serve as an enabling tool for locomotion control and optimization of bipedal robots to efficiently walk on granular terrains.
>
---
#### [new 031] BIND-USBL: Bounding IMU Navigation Drift using USBL in Heterogeneous ASV-AUV Teams
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于水下机器人定位任务，解决AUV在无GPS环境下的导航漂移问题。通过ASVs提供的USBL定位信息，结合多船队形和调度算法，有效限制AUV的死区误差。**

- **链接: [https://arxiv.org/pdf/2604.11861](https://arxiv.org/pdf/2604.11861)**

> **作者:** Pranav Kedia; Rajini Makam; Heiko Hamann; Suresh Sundaram
>
> **备注:** Accepted at OCEANS 2026, Sanya, China
>
> **摘要:** Accurate and continuous localization of Autonomous Underwater Vehicles (AUVs) in GPS-denied environments is a persistent challenge in marine robotics. In the absence of external position fixes, AUVs rely on inertial dead-reckoning, which accumulates unbounded drift due to sensor bias and noise. This paper presents BIND-USBL, a cooperative localization framework in which a fleet of Autonomous Surface Vessels (ASVs) equipped with Ultra-Short Baseline (USBL) acoustic positioning systems provides intermittent fixes to bound AUV dead-reckoning error. The key insight is that long-duration navigation failure is driven not by the accuracy of individual USBL measurements, but by the temporal sparsity and geometric availability of those fixes. BIND-USBL combines a multi-ASV formation model linking survey scale and anchor placement to acoustic coverage, a conflict-graph-based TDMA uplink scheduler for shared-channel servicing, and delayed fusion of received USBL updates with drift-prone dead reckoning. The framework is evaluated in the HoloOcean simulator using heterogeneous ASV-AUV teams executing lawnmower coverage missions. The results show that localization performance is shaped by the interaction of survey scale, acoustic coverage, team composition, and ASV-formation geometry. Further, the spatial-reuse scheduler improves per-AUV fix delivery rate without violating the no-collision constraint, while maintaining low end-to-end fix latency.
>
---
#### [new 032] ReefMapGS: Enabling Large-Scale Underwater Reconstruction by Closing the Loop Between Multimodal SLAM and Gaussian Splatting
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于 underwater 3D重建任务，解决传统方法依赖计算密集型姿态估计的问题。提出ReefMapGS框架，结合多模态SLAM与3D高斯点云，实现高效、准确的水下场景重建与姿态估计。**

- **链接: [https://arxiv.org/pdf/2604.11992](https://arxiv.org/pdf/2604.11992)**

> **作者:** Daniel Yang; Jungseok Hong; John J. Leonard; Yogesh Girdhar
>
> **摘要:** 3D Gaussian Splatting is a powerful visual representation, providing high-quality and efficient 3D scene reconstruction, but it is crucially dependent on accurate camera poses typically obtained from computationally intensive processes like structure-from-motion that are unsuitable for field robot applications. However, in these domains, multimodal sensor data from acoustic, inertial, pressure, and visual sensors are available and suitable for pose-graph optimization-based SLAM methods that can estimate the vehicle's trajectory and thus our needed camera poses while providing uncertainty. We propose a 3DGS-based incremental reconstruction framework, ReefMapGS, that builds an initial model from a high certainty region and progressively expands to incorporate the whole scene. We reconstruct the scene incrementally by interleaving local tracking of new image observations with optimization of the underlying 3DGS scene. These refined poses are integrated back into the pose-graph to globally optimize the whole trajectory. We show COLMAP-free 3D reconstruction of two underwater reef sites with complex geometry as well as more accurate global pose estimation of our AUV over survey trajectories spanning up to 700 m.
>
---
#### [new 033] Social Learning Strategies for Evolved Virtual Soft Robots
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究虚拟软体机器人的协同学习策略，解决形态与控制联合优化问题。通过引入社会学习，机器人可借鉴同类经验提升性能。**

- **链接: [https://arxiv.org/pdf/2604.12482](https://arxiv.org/pdf/2604.12482)**

> **作者:** K. Ege de Bruin; Kyrre Glette; Kai Olav Ellefsen; Giorgia Nadizar; Eric Medvet
>
> **摘要:** Optimizing the body and brain of a robot is a coupled challenge: the morphology determines what control strategies are effective, while the control parameters influence how well the morphology performs. This joint optimization can be done through nested loops of evolutionary and learning processes, where the control parameters of each robot are learned independently. However, the control parameters learned by one robot may contain valuable information for others. Thus, we introduce a social learning approach in which robots can exploit optimized parameters from their peers to accelerate their own brain optimization. Within this framework, we systematically investigate how the selection of teachers, deciding which and how many robots to learn from, affects performance, experimenting with virtual soft robots in four tasks and environments. In particular, we study the effect of inheriting experience from morphologically similar robots due to the tightly coupled body and brain in robot optimization. Our results confirm the effectiveness of building on others' experience, as social learning clearly outperforms learning from scratch under equivalent computational budgets. In addition, while the optimal teacher selection strategy remains open, our findings suggest that incorporating knowledge from multiple teachers can yield more consistent and robust improvements.
>
---
#### [new 034] From Kinematics to Dynamics: Learning to Refine Hybrid Plans for Physically Feasible Execution
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人路径规划任务，解决混合离散-连续动作序列与物理约束不匹配的问题。通过强化学习优化轨迹，提升动态可行性。**

- **链接: [https://arxiv.org/pdf/2604.12474](https://arxiv.org/pdf/2604.12474)**

> **作者:** Lidor Erez; Shahaf S. Shperberg; Ayal Taitler
>
> **摘要:** In many robotic tasks, agents must traverse a sequence of spatial regions to complete a mission. Such problems are inherently mixed discrete-continuous: a high-level action sequence and a physically feasible continuous trajectory. The resulting trajectory and action sequence must also satisfy problem constraints such as deadlines, time windows, and velocity or acceleration limits. While hybrid temporal planners attempt to address this challenge, they typically model motion using linear (first-order) dynamics, which cannot guarantee that the resulting plan respects the robot's true physical constraints. Consequently, even when the high-level action sequence is fixed, producing a dynamically feasible trajectory becomes a bi-level optimization problem. We address this problem via reinforcement learning in continuous space. We define a Markov Decision Process that explicitly incorporates analytical second-order constraints and use it to refine first-order plans generated by a hybrid planner. Our results show that this approach can reliably recover physical feasibility and effectively bridge the gap between a planner's initial first-order trajectory and the dynamics required for real execution.
>
---
#### [new 035] Whole-Body Mobile Manipulation using Offline Reinforcement Learning on Sub-optimal Controllers
- **分类: cs.RO; cs.CV**

- **简介: 该论文研究移动操作任务，解决传统控制器依赖人工调优和学习方法数据成本高的问题。通过离线强化学习改进子优控制器，提升机器人操作性能。**

- **链接: [https://arxiv.org/pdf/2604.12509](https://arxiv.org/pdf/2604.12509)**

> **作者:** Snehal Jauhri; Vignesh Prasad; Georgia Chalvatzaki
>
> **备注:** PrePrint. Project website: this http URL
>
> **摘要:** Mobile Manipulation (MoMa) of articulated objects, such as opening doors, drawers, and cupboards, demands simultaneous, whole-body coordination between a robot's base and arms. Classical whole-body controllers (WBCs) can solve such problems via hierarchical optimization, but require extensive hand-tuned optimization and remain brittle. Learning-based methods, on the other hand, show strong generalization capabilities but typically rely on expensive whole-body teleoperation data or heavy reward engineering. We observe that even a sub-optimal WBC is a powerful structural prior: it can be used to collect data in a constrained, task-relevant region of the state-action space, and its behavior can still be improved upon using offline reinforcement learning. Building on this, we propose WHOLE-MoMa, a two-stage pipeline that first generates diverse demonstrations by randomizing a lightweight WBC, and then applies offline RL to identify and stitch together improved behaviors via a reward signal. To support the expressive action-chunked diffusion policies needed for complex coordination tasks, we extend offline implicit Q-learning with Q-chunking for chunk-level critic evaluation and advantage-weighted policy extraction. On three tasks of increasing difficulty using a TIAGo++ mobile manipulator in simulation, WHOLE-MoMa significantly outperforms WBC, behavior cloning, and several offline RL baselines. Policies transfer directly to the real robot without finetuning, achieving 80% success in bimanual drawer manipulation and 68% in simultaneous cupboard opening and object placement, all without any teleoperated or real-world training data.
>
---
#### [new 036] E2E-Fly: An Integrated Training-to-Deployment System for End-to-End Quadrotor Autonomy
- **分类: cs.RO**

- **简介: 该论文提出E2E-Fly系统，解决四旋翼无人机从仿真到现实的零样本迁移难题。整合训练、验证与部署流程，实现端到端控制任务的高效迁移。**

- **链接: [https://arxiv.org/pdf/2604.12916](https://arxiv.org/pdf/2604.12916)**

> **作者:** Fangyu Sun; Fanxing Li; Linzuo Zhang; Yu Hu; Renbiao Jin; Shuyu Wu; Wenxian Yu; Danping Zou
>
> **摘要:** Training and transferring learning-based policies for quadrotors from simulation to reality remains challenging due to inefficient visual rendering, physical modeling inaccuracies, unmodeled sensor discrepancies, and the absence of a unified platform integrating differentiable physics learning into end-to-end training. While recent work has demonstrated various end-to-end quadrotor control tasks, few systems provide a systematic, zero-shot transfer pipeline, hindering reproducibility and real-world deployment. To bridge this gap, we introduce E2E-Fly, an integrated framework featuring an agile quadrotor platform coupled with a full-stack training, validation, and deployment workflow. The training framework incorporates a high-performance simulator with support for differentiable physics learning and reinforcement learning, alongside structured reward design tailored to common quadrotor tasks. We further introduce a two-stage validation strategy using sim-to-sim transfer and hardware-in-the-loop testing, and deploy policies onto two physical quadrotor platforms via a dedicated low-level control interface and a comprehensive sim-to-real alignment methodology, encompassing system identification, domain randomization, latency compensation, and noise modeling. To the best of our knowledge, this is the first work to systematically unify differentiable physical learning with training, validation, and real-world deployment for quadrotors. Finally, we demonstrate the effectiveness of our framework for training six end-to-end control tasks and deploy them in the real world.
>
---
#### [new 037] D-BDM: A Direct and Efficient Boundary-Based Occupancy Grid Mapping Framework for LiDARs
- **分类: cs.RO**

- **简介: 该论文属于3D占用地图构建任务，解决传统方法内存消耗大和更新延迟高的问题。提出D-BDM框架，通过截断射线投射和直接边界更新，提升效率并减少内存使用。**

- **链接: [https://arxiv.org/pdf/2604.12436](https://arxiv.org/pdf/2604.12436)**

> **作者:** Benxu Tang; Yixi Cai; Fanze Kong; Longji Yin; Fu Zhang
>
> **摘要:** Efficient and scalable 3D occupancy mapping is essential for autonomous robot applications in unknown environments. However, traditional occupancy grid representations suffer from two fundamental limitations. First, explicitly storing all voxels in three-dimensional space leads to prohibitive memory consumption. Second, exhaustive ray casting incurs high update latency. A recent representation alleviate memory demands by maintaining only the voxels on the two-dimensional boundary, yet they still rely on full ray casting updates. This work advances the boundary-based framework with a highly efficient update scheme. We introduce a truncated ray casting strategy that restricts voxel traversal to the exterior of the boundary, which dramatically reduces the number of updated voxels. In addition, we propose a direct boundary update mechanism that removes the need for an auxiliary local 3D occupancy grid, further reducing memory usage and simplifying the map update pipeline. We name our framework as D-BDM. Extensive evaluations on public datasets demonstrate that our approach achieves significantly lower update time and reduced memory consumption compared with the baseline methods, as well as the prior boundary-based approach.
>
---
#### [new 038] Actuation space reduction to facilitate insightful shape matching in a novel reconfigurable tendon driven continuum manipulator
- **分类: cs.RO**

- **简介: 该论文研究可重构腱驱动连续机械臂的形状匹配问题，通过减少驱动空间提升形状匹配效率。**

- **链接: [https://arxiv.org/pdf/2604.12792](https://arxiv.org/pdf/2604.12792)**

> **作者:** Sabyasachi Dash; John Golden; Girish Krishnan
>
> **摘要:** In tendon driven continuum manipulators (TDCMs), reconfiguring the tendon routing enables tailored spatial deformation of the backbone. This work presents a design in which tendons can be rerouted either prior to or after actuation by actively rotating the individual spacer disks. Each disk rotation thus adds a degree of freedom to the actuation space, complicating the mapping from a desired backbone curve to the corresponding actuator inputs. However, when the backbone shape is projected into an intermediate space defined by curvature and torsion (C-T), patterns emerge that highlight which disks are most influential in achieving a global shape. This insight enables a simplified, sequential shape-matching strategy: first, the proximal and intermediate disks are rotated to approximate the global shape; then, the distal disks are adjusted to fine-tune the end-effector position with minimal impact on the overall shape. The proposed actuation framework offers a model-free alternative to conventional control approaches, bypassing the complexities of modeling reconfigurable TDCMs.
>
---
#### [new 039] DeCoNav: Dialog enhanced Long-Horizon Collaborative Vision-Language Navigation
- **分类: cs.RO**

- **简介: 该论文提出DeCoNav，解决多机器人协作导航中的动态协调问题，通过对话驱动的实时任务重分配提升协作效率。**

- **链接: [https://arxiv.org/pdf/2604.12486](https://arxiv.org/pdf/2604.12486)**

> **作者:** Sunyao Zhou; Yunzi Wu; Tianhang Wang; Xinhai Li; Guang Chen; Lizheng Liu; Chenjia Bai; Xuelong Li
>
> **摘要:** Long-horizon collaborative vision-language navigation (VLN) is critical for multi-robot systems to accomplish complex tasks beyond the capability of a single agent. CoNavBench takes a first step by introducing the first collaborative long-horizon VLN benchmark with relay-style multi-robot tasks, a collaboration taxonomy, along with graph-grounded generation and evaluation to model handoffs and rendezvous in shared environments. However, existing benchmarks and evaluations often do not enforce strictly synchronized dual-robot rollout on a shared world timeline, and they typically rely on static coordination policies that cannot adapt when new cross-agent evidence emerges. We present Dialog enhanced Long-Horizon Collaborative Vision-Language Navigation (DeCoNav), a decentralized framework that couples event-triggered dialogue with dynamic task allocation and replanning for real-time, adaptive coordination. In DeCoNav, robots exchange compact semantic states via dialogue without a central controller. When informative events such as new evidence, uncertainty, or conflicts arise, dialogue is triggered to dynamically reassign subgoals and replan under synchronized execution. Implemented in DeCoNavBench with 1,213 tasks across 176 HM3D scenes, DeCoNav improves the both-success rate (BSR) by 69.2%, demonstrating the effectiveness of dialogue-driven, dynamically reallocated planning for multi-robot collaboration.
>
---
#### [new 040] DINO-Explorer: Active Underwater Discovery via Ego-Motion Compensated Semantic Predictive Coding
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出DINO-Explorer，用于水下主动感知任务，解决AUV被动记录遗漏重要事件的问题，通过语义预测和自运动补偿实现高效环境监测。**

- **链接: [https://arxiv.org/pdf/2604.12933](https://arxiv.org/pdf/2604.12933)**

> **作者:** Yuhan Jin; Nayari Marie Lessa; Mariela De Lucas Alvarez; Melvin Laux; Lucas Amparo Barbosa; Frank Kirchner; Rebecca Adam
>
> **摘要:** Marine ecosystem degradation necessitates continuous, scientifically selective underwater monitoring. However, most autonomous underwater vehicles (AUVs) operate as passive data loggers, capturing exhaustive video for offline review and frequently missing transient events of high scientific value. Transitioning to active perception requires a causal, online signal that highlights significant phenomena while suppressing maneuver-induced visual changes. We propose DINO-Explorer, a novelty-aware perception framework driven by a continuous semantic surprise signal. Operating within the latent space of a frozen DINOv3 foundation model, it leverages a lightweight, action-conditioned recurrent predictor to anticipate short-horizon semantic evolution. An efference-copy-inspired module utilizes globally pooled optical flow to discount self-induced visual changes without suppressing genuine environmental novelty. We evaluate this signal on the downstream task of asynchronous event triage under variant telemetry constraints. Results demonstrate that DINO-Explorer provides a robust, bandwidth-efficient attention mechanism. At a fixed operating point, the system retains 78.8% of post-discovery human-reviewer consensus events with a 56.8% trigger confirmation rate, effectively surfacing mission-relevant phenomena. Crucially, ego-motion conditioning suppresses 45.5% of false positives relative to an uncompensated surprise signal baseline. In a replay-side Pareto ablation study, DINO-Explorer robustly dominates the validated peak F1 versus telemetry bandwidth frontier, reducing telemetry bandwidth by 48.2% at the selected operating point while maintaining a 62.2% peak F1 score, successfully concentrating data transmission around human-verified novelty events.
>
---
#### [new 041] Robotic Manipulation is Vision-to-Geometry Mapping ($f(v) \rightarrow G$): Vision-Geometry Backbones over Language and Video Models
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决视觉到几何的映射问题。通过构建视觉-几何-动作模型，直接基于3D表示生成动作，提升操作精度与泛化能力。**

- **链接: [https://arxiv.org/pdf/2604.12908](https://arxiv.org/pdf/2604.12908)**

> **作者:** Zijian Song; Qichang Li; Jiawei Zhou; Zhenlong Yuan; Tianshui Chen; Liang Lin; Guangrun Wang
>
> **备注:** 18 pages, 10 figures
>
> **摘要:** At its core, robotic manipulation is a problem of vision-to-geometry mapping ($f(v) \rightarrow G$). Physical actions are fundamentally defined by geometric properties like 3D positions and spatial relationships. Consequently, we argue that the foundation for generalizable robotic control should be a vision-geometry backbone, rather than the widely adopted vision-language or video models. Conventional VLA and video-predictive models rely on backbones pretrained on large-scale 2D image-text or temporal pixel data. While effective, their representations are largely shaped by semantic concepts or 2D priors, which do not intrinsically align with the precise 3D geometric nature required for physical manipulation. Driven by this insight, we propose the Vision-Geometry-Action (VGA) model, which directly conditions action generation on pretrained native 3D representations. Specifically, VGA replaces conventional language or video backbones with a pretrained 3D world model, establishing a seamless vision-to-geometry mapping that translates visual inputs directly into physical actions. To further enhance geometric consistency, we introduce a Progressive Volumetric Modulation module and adopt a joint training strategy. Extensive experiments validate the effectiveness of our approach. In simulation benchmarks, VGA outperforms top-tier VLA baselines including $\pi_{0.5}$ and GeoVLA, demonstrating its superiority in precise manipulation. More importantly, VGA exhibits remarkable zero-shot generalization to unseen viewpoints in real-world deployments, consistently outperforming $\pi_{0.5}$. These results highlight that operating on native 3D representations-rather than translating through language or 2D video priors-is a highly promising direction for achieving generalizable physical intelligence.
>
---
#### [new 042] XRZero-G0: Pushing the Frontier of Dexterous Robotic Manipulation with Interfaces, Quality and Ratios
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决高质量演示数据获取难题。通过设计XRZero-G0系统，提升数据收集效率与质量，并探索数据混合比例，实现低成本、高效果的机器人学习。**

- **链接: [https://arxiv.org/pdf/2604.13001](https://arxiv.org/pdf/2604.13001)**

> **作者:** Junming Wang; Teng Pu; Wingmun Fung; Jindong Wang; Shanchang Wang; Yuan Deng; Shuyuan Wang; Ziwei Liu; Kunhao Pan; Ping Yang; Peng Zhai; Yuxin Liang; Xiaofan Li; Jiabi Sun; Renchao Xu; Xiaotian Tian; Pengfei Yan; Guoqiang Ye; Liang Li; Qian Wang; Ruyi Gan; Hao Wang
>
> **备注:** Technical Report
>
> **摘要:** The acquisition of high-quality, action-aligned demonstration data remains a fundamental bottleneck in scaling foundation models for dexterous robot manipulation. Although robot-free human demonstrations (e.g., the UMI paradigm) offer a scalable alternative to traditional teleoperation, current systems are constrained by sub-optimal hardware ergonomics, open-loop workflows, and a lack of systematic data-mixing strategies. To address these limitations, we present XRZero-G0, a hardware-software co-designed system for embodied data collection and policy learning. The system features an ergonomic, virtual reality interface equipped with a top-view camera and dual specialized grippers to directly improve collection efficiency. To ensure dataset reliability, we propose a closed-loop collection, inspection, training, and evaluation pipeline for non-proprioceptive data. This workflow achieves an 85% data validity rate and establishes a transparent mechanism for quality control. Furthermore, we investigate the empirical scaling behaviors and optimal mixing ratios of robot-free data. Extensive experiments indicate that combining a minimal volume of real-robot data with large-scale robot-free data (e.g., a 10:1 ratio) achieves performance comparable to exclusively real-robot datasets, while reducing acquisition costs by a factor of twenty. Utilizing XRZero-G0, we construct a 2,000-hour robot-free dataset that enables zero-shot cross-embodiment transfer to a target physical robot, demonstrating a highly scalable methodology for generalized real-world this http URL project repository: this https URL
>
---
#### [new 043] FeaXDrive: Feasibility-aware Trajectory-Centric Diffusion Planning for End-to-End Autonomous Driving
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于自动驾驶轨迹规划任务，解决生成轨迹物理可行性不足的问题。提出FeaXDrive方法，通过轨迹中心建模提升几何、运动学及道路区域一致性。**

- **链接: [https://arxiv.org/pdf/2604.12656](https://arxiv.org/pdf/2604.12656)**

> **作者:** Baoyun Wang; Zhuoren Li; Ming Liu; Xinrui Zhang; Bo Leng; Lu Xiong
>
> **备注:** 21 pages, 6 figures
>
> **摘要:** End-to-end diffusion planning has shown strong potential for autonomous driving, but the physical feasibility of generated trajectories remains insufficiently addressed. In particular, generated trajectories may exhibit local geometric irregularities, violate trajectory-level kinematic constraints, or deviate from the drivable area, indicating that the commonly used noise-centric formulation in diffusion planning is not yet well aligned with the trajectory space where feasibility is more naturally characterized. To address this issue, we propose FeaXDrive, a feasibility-aware trajectory-centric diffusion planning method for end-to-end autonomous driving. The core idea is to treat the clean trajectory as the unified object for feasibility-aware modeling throughout the diffusion process. Built on this trajectory-centric formulation, FeaXDrive integrates adaptive curvature-constrained training to improve intrinsic geometric and kinematic feasibility, drivable-area guidance within reverse diffusion sampling to enhance consistency with the drivable area, and feasibility-aware GRPO post-training to further improve planning performance while balancing trajectory-space feasibility. Experiments on the NAVSIM benchmark show that FeaXDrive achieves strong closed-loop planning performance while substantially improving trajectory-space feasibility. These findings highlight the importance of explicitly modeling trajectory-space feasibility in end-to-end diffusion planning and provide a step toward more reliable and physically grounded autonomous driving planners.
>
---
#### [new 044] GGD-SLAM: Monocular 3DGS SLAM Powered by Generalizable Motion Model for Dynamic Environments
- **分类: cs.RO**

- **简介: 该论文属于视觉SLAM任务，解决动态环境中定位与建图问题。提出GGD-SLAM框架，利用通用运动模型分离静态与动态成分，提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2604.12837](https://arxiv.org/pdf/2604.12837)**

> **作者:** Yi Liu; Haoxuan Xu; Hongbo Duan; Keyu Fan; Zhengyang Zhang; Peiyu Zhuang; Pengting Luo; Houde Liu
>
> **备注:** 8 pages, Accepted by ICRA 2026
>
> **摘要:** Visual SLAM algorithms achieve significant improvements through the exploration of 3D Gaussian Splatting (3DGS) representations, particularly in generating high-fidelity dense maps. However, they depend on a static environment assumption and experience significant performance degradation in dynamic environments. This paper presents GGD-SLAM, a framework that employs a generalizable motion model to address the challenges of localization and dense mapping in dynamic environments - without predefined semantic annotations or depth input. Specifically, the proposed system employs a First-In-First-Out (FIFO) queue to manage incoming frames, facilitating dynamic semantic feature extraction through a sequential attention mechanism. This is integrated with a dynamic feature enhancer to separate static and dynamic components. Additionally, to minimize dynamic distractors' impact on the static components, we devise a method to fill occluded areas via static information sampling and design a distractor-adaptive Structure Similarity Index Measure (SSIM) loss tailored for dynamic environments, significantly enhancing the system's resilience. Experiments conducted on real-world dynamic datasets demonstrate that the proposed system achieves state-of-the-art performance in camera pose estimation and dense reconstruction in dynamic scenes.
>
---
#### [new 045] Dynamic Multi-Robot Task Allocation under Uncertainty and Communication Constraints: A Game-Theoretic Approach
- **分类: eess.SY; cs.GT; cs.RO**

- **简介: 该论文研究多机器人动态任务分配问题，解决在不确定性和通信限制下的任务调度难题。提出IBR算法，实现高效任务分配。**

- **链接: [https://arxiv.org/pdf/2604.11954](https://arxiv.org/pdf/2604.11954)**

> **作者:** Maria G. Mendoza; Pan-Yang Su; Bryce L. Ferguson; S. Shankar Sastry
>
> **备注:** 9 pages, 6 figures
>
> **摘要:** We study dynamic multi-robot task allocation under uncertain task completion, time-window constraints, and incomplete information. Tasks arrive online over a finite horizon and must be completed within specified deadlines, while agents operate from distributed hubs with limited sensing and communication. We model incomplete information through hub-based sensing regions that determine task visibility and a communication graph that governs inter-hub information exchange. Using this framework, we propose Iterative Best Response (IBR), a decentralized policy in which each agent selects the task that maximizes its marginal contribution to the locally observed welfare. We compare IBR against three baselines: Earliest Due Date first (EDD), Hungarian algorithm, and Stochastic Conflict-Based Allocation (SCoBA), on a city-scale package-delivery domain with up to 100 drones and varying task arrival scenarios. Under full and sparse communication, IBR achieves competitive task-completion performance with lower computation time.
>
---
#### [new 046] A Comparison of Reinforcement Learning and Optimal Control Methods for Path Planning
- **分类: math.OC; cs.RO**

- **简介: 该论文属于路径规划任务，旨在解决自主车辆在威胁环境中的实时路径规划问题。通过DDPG方法与传统最优控制方法对比，验证了学习方法在速度上的优势。**

- **链接: [https://arxiv.org/pdf/2604.12628](https://arxiv.org/pdf/2604.12628)**

> **作者:** Qiang Le; Yaguang Yang; Isaac E. Weintraub
>
> **备注:** 8 pages, 9 figures, submitted to AAAI Conference
>
> **摘要:** Path-planning for autonomous vehicles in threat-laden environments is a fundamental challenge. While traditional optimal control methods can find ideal paths, the computational time is often too slow for real-time decision-making. To solve this challenge, we propose a method based on Deep Deterministic Policy Gradient (DDPG) and model the threat as a simple, circular `no-go' zone. A mission failure is claimed if the vehicle enters this `no-go' zone at any time or does not reach a neighborhood of the destination. The DDPG agent is trained to learn a direct mapping from its current state (position and velocity) to a series of feasible actions that guide the agent to safely reach its goal. A reward function and two neural networks, critic and actor, are used to describe the environment and guide the control efforts. The DDPG trains the agent to find the largest possible set of starting points (``feasible set'') wherein a safe path to the goal is guaranteed. This provides critical information for mission planning, showing beforehand whether a task is achievable from a given starting point, assisting pre-mission planning activities. The approach is validated in simulation. A comparison between the DDPG method and a traditional optimal control (pseudo-spectral) method is carried out. The results show that the learning-based agent may produce effective paths while being significantly faster, making it a better fit for real-time applications. However, there are areas (``infeasible set'') where the DDPG agent cannot find paths to the destination, and the paths in the feasible set may not be optimal. These preliminary results guide our future research: (1) improve the reward function to enlarge the DDPG feasible set, (2) examine the feasible set obtained by the pseudo-spectral method, and (3) investigate the arc-search IPM method for the path planning problem.
>
---
#### [new 047] Artificial Intelligence for Modeling and Simulation of Mixed Automated and Human Traffic
- **分类: cs.AI; cs.RO; eess.SY**

- **简介: 该论文属于交通仿真领域，旨在解决混合交通中自动驾驶与人类驾驶行为建模的问题。通过综述AI方法，提出分类体系并分析现有工具不足，推动仿真技术发展。**

- **链接: [https://arxiv.org/pdf/2604.12857](https://arxiv.org/pdf/2604.12857)**

> **作者:** Saeed Rahmani; Shiva Rasouli; Daphne Cornelisse; Eugene Vinitsky; Bart van Arem; Simeon C. Calvert
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Autonomous vehicles (AVs) are now operating on public roads, which makes their testing and validation more critical than ever. Simulation offers a safe and controlled environment for evaluating AV performance in varied conditions. However, existing simulation tools mainly focus on graphical realism and rely on simple rule-based models and therefore fail to accurately represent the complexity of driving behaviors and interactions. Artificial intelligence (AI) has shown strong potential to address these limitations; however, despite the rapid progress across AI methodologies, a comprehensive survey of their application to mixed autonomy traffic simulation remains lacking. Existing surveys either focus on simulation tools without examining the AI methods behind them, or cover ego-centric decision-making without addressing the broader challenge of modeling surrounding traffic. Moreover, they do not offer a unified taxonomy of AI methods covering individual behavior modeling to full scene simulation. To address these gaps, this survey provides a structured review and synthesis of AI methods for modeling AV and human driving behavior in mixed autonomy traffic simulation. We introduce a taxonomy that organizes methods into three families: agent-level behavior models, environment-level simulation methods, and cognitive and physics-informed methods. The survey analyzes how existing simulation platforms fall short of the needs of mixed autonomy research and outlines directions to narrow this gap. It also provides a chronological overview of AI methods and reviews evaluation protocols and metrics, simulation tools, and datasets. By covering both traffic engineering and computer science perspectives, we aim to bridge the gap between these two communities.
>
---
#### [new 048] Learning step-level dynamic soaring in shear flow
- **分类: physics.flu-dyn; cs.RO**

- **简介: 该论文研究动态滑翔在剪切流中的控制问题，通过深度强化学习实现无需轨迹规划的局部反馈控制，提升飞行器在复杂环境中的自主导航能力。**

- **链接: [https://arxiv.org/pdf/2604.12413](https://arxiv.org/pdf/2604.12413)**

> **作者:** Lunbing Chen; Jixin Lu; Yufei Yin; Jinpeng Huang; Yang Xiang; Hong Liu
>
> **摘要:** Dynamic soaring enables sustained flight by extracting energy from wind shear, yet it is commonly understood as a cycle-level maneuver that assumes stable flow conditions. In realistic unsteady environments, however, such assumptions are often violated, raising the question of whether explicit cycle-level planning is necessary. Here, we show that dynamic soaring can emerge from step-level, state-feedback control using only local sensing, without explicit trajectory planning. Using deep reinforcement learning as a tool, we obtain policies that achieve robust omnidirectional navigation across diverse shear-flow conditions. The learned behavior organizes into a structured control law that coordinates turning and vertical motion, giving rise to a two-phase strategy governed by a trade-off between energy extraction and directional progress. The resulting policy generalizes across varying conditions and reproduces key features observed in biological flight and optimal-control solutions. These findings identify a feedback-based control structure underlying dynamic soaring, demonstrating that efficient energy-harvesting flight can emerge from local interactions with the flow without explicit planning, and providing insights for biological flight and autonomous systems in complex, flow-coupled environments.
>
---
## 更新

#### [replaced 001] LLM-Guided Task- and Affordance-Level Exploration in Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人强化学习任务，解决样本效率低和探索困难的问题。工作是提出LLM-TALE框架，利用大语言模型引导任务与可操作性层面的探索，提升学习效率与成功率。**

- **链接: [https://arxiv.org/pdf/2509.16615](https://arxiv.org/pdf/2509.16615)**

> **作者:** Jelle Luijkx; Runyu Ma; Zlatan Ajanović; Jens Kober
>
> **备注:** 8 pages, 7 figures, ICRA 2026
>
> **摘要:** Reinforcement learning (RL) is a promising approach for robotic manipulation, but it can suffer from low sample efficiency and requires extensive exploration of large state-action spaces. Recent methods leverage the commonsense knowledge and reasoning abilities of large language models (LLMs) to guide exploration toward more meaningful states. However, LLMs can produce plans that are semantically plausible yet physically infeasible, yielding unreliable behavior. We introduce LLM-TALE, a framework that uses LLMs' planning to directly steer RL exploration. LLM-TALE integrates planning at both the task level and the affordance level, improving learning efficiency by directing agents toward semantically meaningful actions. Unlike prior approaches that assume optimal LLM-generated plans or rewards, LLM-TALE corrects suboptimality online and explores multimodal affordance-level plans without human supervision. We evaluate LLM-TALE on pick-and-place tasks in standard RL benchmarks, observing improvements in both sample efficiency and success rates over strong baselines. Real-robot experiments indicate promising zero-shot sim-to-real transfer. Code and supplementary material are available at this http URL.
>
---
#### [replaced 002] Scalable Verification of Neural Control Barrier Functions Using Linear Bound Propagation
- **分类: cs.LG; cs.RO; eess.SY; math.OC**

- **简介: 该论文属于安全验证任务，解决神经网络作为控制屏障函数的验证问题。通过线性边界传播和 McCormick 放松，提升验证效率与规模。**

- **链接: [https://arxiv.org/pdf/2511.06341](https://arxiv.org/pdf/2511.06341)**

> **作者:** Nikolaus Vertovec; Frederik Baymler Mathiesen; Thom Badings; Luca Laurenti; Alessandro Abate
>
> **备注:** accepted at the 8th Annual Conference on Learning for Dynamics and Control (L4DC 2026)
>
> **摘要:** Control barrier functions (CBFs) are a popular tool for safety certification of nonlinear dynamical control systems. Recently, CBFs represented as neural networks have shown great promise due to their expressiveness and applicability to a broad class of dynamics and safety constraints. However, verifying that a trained neural network is indeed a valid CBF is a computational bottleneck that limits the size of the networks that can be used. To overcome this limitation, we present a novel framework for verifying neural CBFs based on piecewise linear upper and lower bounds on the conditions required for a neural network to be a CBF. Our approach is rooted in linear bound propagation (LBP) for neural networks, which we extend to compute bounds on the gradients of the network. Combined with McCormick relaxation, we derive linear upper and lower bounds on the CBF conditions, thereby eliminating the need for computationally expensive verification procedures. Our approach applies to arbitrary control-affine systems and a broad range of nonlinear activation functions. To reduce conservatism, we develop a parallelizable refinement strategy that adaptively refines the regions over which these bounds are computed. Our approach scales to larger neural networks than state-of-the-art verification procedures for CBFs, as demonstrated by our numerical experiments.
>
---
#### [replaced 003] Goal-Conditioned Neural ODEs with Guaranteed Safety and Stability for Learning-Based All-Pairs Motion Planning
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于运动规划任务，解决任意起点到目标的路径生成问题。通过构建安全的神经ODE模型，确保全局稳定性和安全性，提升学习方法的可靠性。**

- **链接: [https://arxiv.org/pdf/2604.02821](https://arxiv.org/pdf/2604.02821)**

> **作者:** Dechuan Liu; Ruigang Wang; Ian R. Manchester
>
> **摘要:** This paper presents a learning-based approach for all-pairs motion planning, where the initial and goal states are allowed to be arbitrary points in a safe set. We construct smooth goal-conditioned neural ordinary differential equations (neural ODEs) via bi-Lipschitz diffeomorphisms. Theoretical results show that the proposed model can provide guarantees of global exponential stability and safety (safe set forward invariance) regardless of goal location. Moreover, explicit bounds on convergence rate, tracking error, and vector field magnitude are established. Our approach admits a tractable learning implementation using bi-Lipschitz neural networks and can incorporate demonstration data. We illustrate the effectiveness of the proposed method on a 2D corridor navigation task.
>
---
#### [replaced 004] Deep QP Safety Filter: Model-free Learning for Reachability-based Safety Filter
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于安全控制任务，旨在解决黑箱动态系统中的安全问题。通过结合HJ可达性与无模型学习，提出Deep QP Safety Filter，实现安全过滤器的无模型学习。**

- **链接: [https://arxiv.org/pdf/2601.21297](https://arxiv.org/pdf/2601.21297)**

> **作者:** Byeongjun Kim; H. Jin Kim
>
> **备注:** Accepted to the 8th Annual Learning for Dynamics and Control Conference (L4DC 2026)
>
> **摘要:** We introduce Deep QP Safety Filter, a fully data-driven safety layer for black-box dynamical systems. Our method learns a Quadratic-Program (QP) safety filter without model knowledge by combining Hamilton-Jacobi (HJ) reachability with model-free learning. We construct contraction-based losses for both the safety value and its derivatives, and train two neural networks accordingly. In the exact setting, the learned critic converges to the viscosity solution (and its derivative), even for non-smooth values. Across diverse dynamical systems -- even including a hybrid system -- and multiple RL tasks, Deep QP Safety Filter substantially reduces pre-convergence failures while accelerating learning toward higher returns than strong baselines, offering a principled and practical route to safe, model-free control.
>
---
#### [replaced 005] iTeach: In the Wild Interactive Teaching for Failure-Driven Adaptation of Robot Perception
- **分类: cs.RO**

- **简介: 该论文提出iTeach框架，解决机器人感知在真实环境中的失败问题。通过人类交互收集失败样本，进行模型微调，提升分割与操作性能。属于机器人感知适应任务。**

- **链接: [https://arxiv.org/pdf/2410.09072](https://arxiv.org/pdf/2410.09072)**

> **作者:** Jishnu Jaykumar P; Cole Salvato; Vinaya Bomnale; Jikai Wang; Yu Xiang
>
> **摘要:** Robotic perception models often fail when deployed in real-world environments due to out-of-distribution conditions such as clutter, occlusion, and novel object instances. Existing approaches address this gap through offline data collection and retraining, which are slow and do not resolve deployment-time failures. We propose iTeach, a failure-driven interactive teaching framework for adapting robot perception in the wild. A co-located human observes model predictions during deployment, identifies failure cases, and performs short human-object interaction (HumanPlay) to expose informative object configurations while recording RGB-D sequences. To minimize annotation effort, iTeach employs a Few-Shot Semi- Supervised (FS3) labeling strategy, where only the final frame of a short interaction sequence is annotated using hands-free eye-gaze and voice commands, and labels are propagated across the video to produce dense supervision. The collected failure-driven samples are used for iterative fine-tuning, enabling progressive deployment-time adaptation of the perception model. We evaluate iTeach on unseen object instance segmentation (UOIS) starting from a pretrained MSMFormer model. Using a small number of failure-driven samples, our method significantly improves segmentation performance across diverse real-world scenes. These improvements directly translate to higher grasping and pick-and-place success on the SceneReplica benchmark and real robotic experiments. Our results demonstrate that failure-driven, co-located interactive teaching enables efficient in-the-wild adaptation of robot perception and improves downstream manipulation performance. Project page at this https URL
>
---
#### [replaced 006] Progress-Think: Semantic Progress Reasoning for Vision-Language Navigation
- **分类: cs.RO**

- **简介: 该论文属于视觉语言导航任务，解决长期导航中进度理解不足的问题。通过语义进度推理，提升导航一致性与准确性。**

- **链接: [https://arxiv.org/pdf/2511.17097](https://arxiv.org/pdf/2511.17097)**

> **作者:** Shuo Wang; Yucheng Wang; Guoxin Lian; Yongcai Wang; Maiyue Chen; Kaihui Wang; Bo Zhang; Zhizhong Su; Yutian Zhou; Wanting Li; Deying Li; Zhaoxin Fan
>
> **摘要:** Vision-Language Navigation requires agents to act coherently over long horizons by understanding not only local visual context but also how far they have advanced within a multi-step instruction. However, recent Vision-Language-Action models focus on direct action prediction and earlier progress methods predict numeric achievements; both overlook the monotonic co-progression property of the observation and instruction sequences. Building on this insight, Progress-Think introduces semantic progress reasoning, predicting instruction-style progress from visual observations to enable more accurate navigation. To achieve this without expensive annotations, we propose a three-stage framework. In the initial stage, Self-Aligned Progress Pretraining bootstraps a reasoning module via a novel differentiable alignment between visual history and instruction prefixes. Then, Progress-Guided Policy Pretraining injects learned progress states into the navigation context, guiding the policy toward consistent actions. Finally, Progress-Policy Co-Finetuning jointly optimizes both modules with tailored progress-aware reinforcement objectives. Experiments on R2R-CE and RxR-CE show state-of-the-art success and efficiency, demonstrating that semantic progress yields a more consistent representation of navigation advancement.
>
---
#### [replaced 007] Toward Efficient and Robust Behavior Models for Multi-Agent Driving Simulation
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于多智能体驾驶仿真任务，旨在提升行为模型的效率与鲁棒性。通过优化场景表示和交互建模，实现更高效的训练与推理。**

- **链接: [https://arxiv.org/pdf/2512.05812](https://arxiv.org/pdf/2512.05812)**

> **作者:** Fabian Konstantinidis; Moritz Sackmann; Ulrich Hofmann; Christoph Stiller
>
> **备注:** This is the author's accepted version of a paper to appear in the IEEE International Conference on Robotics & Automation (ICRA 2026)
>
> **摘要:** Scalable multi-agent driving simulation requires behavior models that are both realistic and computationally efficient. We address this by optimizing the behavior model that controls individual traffic participants. To improve efficiency, we adopt an instance-centric scene representation, where each traffic participant and map element is modeled in its own local coordinate frame. This design enables efficient, viewpoint-invariant scene encoding and allows static map tokens to be reused across simulation steps. To model interactions, we employ a query-centric symmetric context encoder with relative positional encodings between local frames. We use Adversarial Inverse Reinforcement Learning to learn the behavior model and propose an adaptive reward transformation that automatically balances robustness and realism during training. Experiments demonstrate that our approach scales efficiently with the number of tokens, significantly reducing training and inference times, while outperforming several agent-centric baselines in terms of positional accuracy and robustness.
>
---
#### [replaced 008] BINDER: Instantly Adaptive Mobile Manipulation with Open-Vocabulary Commands
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出BINDER框架，解决动态环境中机器人实时适应问题。通过结合战略规划与持续监控模块，提升任务成功率和效率，适用于移动操作任务。**

- **链接: [https://arxiv.org/pdf/2511.22364](https://arxiv.org/pdf/2511.22364)**

> **作者:** Seongwon Cho; Daechul Ahn; Donghyun Shin; Hyeonbeom Choi; San Kim; Jonghyun Choi
>
> **备注:** 12 pages, 8 figures
>
> **摘要:** Open-vocabulary mobile manipulation (OVMM) requires robots to follow language instructions, navigate, and manipulate while updating their world representation under dynamic environmental changes. However, most prior approaches update their world representation only at discrete update points such as navigation targets, waypoints, or the end of an action step, leaving robots blind between updates and causing cascading failures: overlooked objects, late error detection, and delayed replanning. To address this limitation, we propose BINDER (Bridging INstant and DEliberative Reasoning), a dual process framework that decouples strategic planning from continuous environment monitoring. Specifically, BINDER integrates a Deliberative Response Module (DRM, a multimodal LLM for task planning) with an Instant Response Module (IRM, a VideoLLM for continuous monitoring). The two modules play complementary roles: the DRM performs strategic planning with structured 3D scene updates and guides what the IRM attends to, while the IRM analyzes video streams to update memory, correct ongoing actions, and trigger replanning when necessary. Through this bidirectional coordination, the modules address the trade off between maintaining awareness and avoiding costly updates, enabling robust adaptation under dynamic conditions. Evaluated in three real world environments with dynamic object placement, BINDER achieves substantially higher success and efficiency than SoTA baselines, demonstrating its effectiveness for real world deployment.
>
---
#### [replaced 009] Learned Incremental Nonlinear Dynamic Inversion for Quadrotors with and without Slung Payloads
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于无人机控制任务，解决传统控制器难以准确建模高阶力的问题。通过引入神经网络替代专用传感器，实现更精确的残差计算，提升多旋翼飞行性能。**

- **链接: [https://arxiv.org/pdf/2503.09441](https://arxiv.org/pdf/2503.09441)**

> **作者:** Eckart Cobo-Briesewitz; Khaled Wahba; Wolfgang Hönig
>
> **备注:** Accepted to L4DC 2026
>
> **摘要:** The increasing complexity of multirotor applications demands flight controllers that can accurately account for all forces acting on the vehicle. Conventional controllers model most aerodynamic and dynamic effects but often neglect higher-order forces, as their accurate estimation is computationally expensive. Incremental Nonlinear Dynamic Inversion (INDI) offers an alternative by estimating residual forces from differences in sensor measurements; however, its reliance on specialized and often noisy sensors limits its applicability. Recent work has demonstrated that residual forces can be predicted using learning-based methods. In this paper, we show that a neural network can generate smooth approximations of INDI outputs without requiring specialized rotor RPM sensor inputs. We further propose a hybrid approach that integrates learning-based predictions with INDI and demonstrate both methods for multirotors and multirotors carrying slung payloads. Experimental results on trajectory tracking errors demonstrate that the specialized sensor measurements required by INDI can be eliminated by replacing the residual computation with a neural network.
>
---
#### [replaced 010] AnySlot: Goal-Conditioned Vision-Language-Action Policies for Zero-Shot Slot-Level Placement
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，解决复杂语言指令下的精准物体放置问题。提出AnySlot框架，通过视觉目标分解任务，提升零样本场景下的放置精度。**

- **链接: [https://arxiv.org/pdf/2604.10432](https://arxiv.org/pdf/2604.10432)**

> **作者:** Zhaofeng Hu; Sifan Zhou; Qinbo Zhang; Rongtao Xu; Qi Su; Ci-Jyun Liang
>
> **摘要:** Vision-Language-Action (VLA) policies have emerged as a versatile paradigm for generalist robotic manipulation. However, precise object placement under compositional language instructions remains a major challenge for modern monolithic VLA policies. Slot-level tasks require both reliable slot grounding and sub-centimeter execution accuracy. To this end, we propose AnySlot, a framework that reduces compositional complexity by introducing an explicit spatial visual goal as an intermediate representation between language grounding and control. AnySlot turns language into an explicit visual goal by generating a scene marker, then executes this goal with a goal-conditioned VLA policy. This hierarchical design effectively decouples high-level slot selection from low-level execution, ensuring both semantic accuracy and spatial robustness. Furthermore, recognizing the lack of existing benchmarks for such precision-demanding tasks, we introduce SlotBench, a comprehensive simulation benchmark featuring nine task categories tailored to evaluate structured spatial reasoning in slot-level placement. Extensive experiments show that AnySlot significantly outperforms flat VLA baselines and previous modular grounding methods in zero-shot slot-level placement.
>
---
#### [replaced 011] ACDC: Adaptive Curriculum Planning with Dynamic Contrastive Control for Goal-Conditioned Reinforcement Learning in Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人操作中的强化学习任务，旨在解决传统方法在多样任务中性能不佳的问题。提出ACDC框架，结合自适应课程规划与动态对比控制，提升学习效率和成功率。**

- **链接: [https://arxiv.org/pdf/2603.02104](https://arxiv.org/pdf/2603.02104)**

> **作者:** Xuerui Wang; Guangyu Ren; Tianhong Dai; Bintao Hu; Shuangyao Huang; Wenzhang Zhang; Hengyan Liu
>
> **备注:** 13 pages (including references and appendix), 12 figures. Accepted to ICAPS 2026. Code available at this https URL
>
> **摘要:** Goal-conditioned reinforcement learning has shown considerable potential in robotic manipulation; however, existing approaches remain limited by their reliance on prioritizing collected experience, resulting in suboptimal performance across diverse tasks. Inspired by human learning behaviors, we propose a more comprehensive learning paradigm, ACDC, which integrates multidimensional Adaptive Curriculum (AC) Planning with Dynamic Contrastive (DC) Control to guide the agent along a well-designed learning trajectory. More specifically, at the planning level, the AC component schedules the learning curriculum by dynamically balancing diversity-driven exploration and quality-driven exploitation based on the agent's success rate and training progress. At the control level, the DC component implements the curriculum plan through norm-constrained contrastive learning, enabling magnitude-guided experience selection aligned with the current curriculum focus. Extensive experiments on challenging robotic manipulation tasks demonstrate that ACDC consistently outperforms the state-of-the-art baselines in both sample efficiency and final task success rate.
>
---
#### [replaced 012] ABot-M0: VLA Foundation Model for Robotic Manipulation with Action Manifold Learning
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决多硬件通用智能体构建难题。通过数据标准化与动作流形学习，提升动作预测效率与泛化能力。**

- **链接: [https://arxiv.org/pdf/2602.11236](https://arxiv.org/pdf/2602.11236)**

> **作者:** Yandan Yang; Shuang Zeng; Tong Lin; Xinyuan Chang; Dekang Qi; Junjin Xiao; Haoyun Liu; Ronghan Chen; Yuzhi Chen; Dongjie Huo; Feng Xiong; Xing Wei; Zhiheng Ma; Mu Xu
>
> **备注:** Project website: this https URL . Code: this https URL . 22 pages, 10 figures, 10 tables
>
> **摘要:** Building general-purpose embodied agents across diverse hardware remains a central challenge in robotics, often framed as the ''one-brain, many-forms'' paradigm. Progress is hindered by fragmented data, inconsistent representations, and misaligned training objectives. We present ABot-M0, a framework that builds a systematic data curation pipeline while jointly optimizing model architecture and training strategies, enabling end-to-end transformation of heterogeneous raw data into unified, efficient representations. From six public datasets, we clean, standardize, and balance samples to construct UniACT-dataset, a large-scale dataset with over 6 million trajectories and 9,500 hours of data, covering diverse robot morphologies and task scenarios. Unified pre-training improves knowledge transfer and generalization across platforms and tasks, supporting general-purpose embodied intelligence. To improve action prediction efficiency and stability, we propose the Action Manifold Hypothesis: effective robot actions lie not in the full high-dimensional space but on a low-dimensional, smooth manifold governed by physical laws and task constraints. Based on this, we introduce Action Manifold Learning (AML), which uses a DiT backbone to predict clean, continuous action sequences directly. This shifts learning from denoising to projection onto feasible manifolds, improving decoding speed and policy stability. ABot-M0 supports modular perception via a dual-stream mechanism that integrates VLM semantics with geometric priors and multi-view inputs from plug-and-play 3D modules such as VGGT and Qwen-Image-Edit, enhancing spatial understanding without modifying the backbone and mitigating standard VLM limitations in 3D reasoning. Experiments show components operate independently with additive benefits. We will release all code and pipelines for reproducibility and future research.
>
---
#### [replaced 013] MR.ScaleMaster: Scale-Consistent Collaborative Mapping from Crowd-Sourced Monocular Videos
- **分类: cs.RO**

- **简介: 该论文属于多机器人协同定位与建图任务，解决单目视频在重复环境中的尺度崩溃和长期轨迹尺度漂移问题，提出Sim(3)框架和报警机制实现尺度一致的联合建图。**

- **链接: [https://arxiv.org/pdf/2604.11372](https://arxiv.org/pdf/2604.11372)**

> **作者:** Hyoseok Ju; Giseop Kim
>
> **备注:** 8 pages, 7 figures, submitted to IROS 2026
>
> **摘要:** Crowd-sourced cooperative mapping from monocular cameras promises scalable 3D reconstruction without specialized sensors, yet remains hindered by two scale-specific failure modes: abrupt scale collapse from false-positive loop closures in repetitive environments, and gradual scale drift over long trajectories and per-robot scale ambiguity that prevent direct multi-session fusion. We present \textbf{this http URL}, a cooperative mapping system for crowd-sourced monocular videos that addresses both failure modes. \textbf{this http URL} introduces three key mechanisms. First, a Scale Collapse Alarm rejects spurious loop closures before they corrupt the pose graph. Second, a Sim(3) anchor node formulation generalizes the classical SE(3) framework to explicitly estimate per-session scale, resolving per-robot scale ambiguity and enforcing global scale consistency. Third, a modular, open-source, plug-and-play interface enables any monocular reconstruction model to integrate without backend modification. On KITTI sequences with up to 15 agents, the Sim(3) formulation achieves a 7.2x ATE reduction over the SE(3) baseline, and the alarm rejects all false-positive loops while preserving every valid constraint. We further demonstrate heterogeneous multi-robot dense mapping fusing MASt3R-SLAM, pi3, and VGGT-SLAM 2.0 within a single unified map.
>
---
#### [replaced 014] Scalable and General Whole-Body Control for Cross-Humanoid Locomotion
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决跨人体模型的通用控制问题。通过XHugWBC框架，实现单一策略在多种机器人上的泛化控制。**

- **链接: [https://arxiv.org/pdf/2602.05791](https://arxiv.org/pdf/2602.05791)**

> **作者:** Yufei Xue; YunFeng Lin; Wentao Dong; Yang Tang; Jingbo Wang; Jiangmiao Pang; Ming Zhou; Minghuan Liu; Weinan Zhang
>
> **摘要:** Learning-based whole-body controllers have become a key driver for humanoid robots, yet most existing approaches require robot-specific training. In this paper, we study the problem of cross-embodiment humanoid control and show that a single policy can robustly generalize across a wide range of humanoid robot designs with one-time training. We introduce XHugWBC, a novel cross-embodiment training framework that enables generalist humanoid control through: (1) physics-consistent morphological randomization, (2) semantically aligned observation and action spaces across diverse humanoid robots, and (3) effective policy architectures modeling morphological and dynamical properties. XHugWBC is not tied to any specific robot. Instead, it internalizes a broad distribution of morphological and dynamical characteristics during training. By learning motion priors from diverse randomized embodiments, the policy acquires a strong structural bias that supports zero-shot transfer to previously unseen robots. Experiments on twelve simulated humanoids and seven real-world robots demonstrate the strong generalization and robustness of the resulting universal controller.
>
---
#### [replaced 015] Skill-informed Data-driven Haptic Nudges for High-dimensional Human Motor Learning
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于人机协同学习任务，旨在解决高维运动任务中学习效率低的问题。通过构建POMDP模型优化触觉提示策略，提升学习者运动表现与技能发展。**

- **链接: [https://arxiv.org/pdf/2603.12583](https://arxiv.org/pdf/2603.12583)**

> **作者:** Ankur Kamboj; Rajiv Ranganathan; Xiaobo Tan; Vaibhav Srivastava
>
> **摘要:** In this work, we propose a data-driven framework to design optimal haptic nudge feedback leveraging the learner's estimated skill to address the challenge of learning a novel motor task in a high-dimensional, redundant motor space. A nudge is a series of vibrotactile feedback delivered to the learner to encourage motor movements that aid in task completion. We first model the stochastic dynamics of human motor learning under haptic nudges using an Input-Output Hidden Markov Model (IOHMM), which explicitly decouples latent skill evolution from observable performance measures. Leveraging this predictive model, we formulate the haptic nudge feedback design problem as a Partially Observable Markov Decision Process (POMDP). This allows us to derive an optimal nudging policy that minimizes long-term performance cost and implicitly guides the learner toward superior skill states. We validate our approach through a human participant study (N=30) involving a high-dimensional motor task rendered through a hand exoskeleton. Results demonstrate that participants trained with the POMDP-derived policy exhibit significantly accelerated movement efficiency and endpoint accuracy compared to groups receiving heuristic-based feedback or no feedback. Furthermore, synergy analysis reveals that the POMDP group discovers efficient low-dimensional motor representations more rapidly.
>
---
#### [replaced 016] Multi-ORFT: Stable Online Reinforcement Fine-Tuning for Multi-Agent Diffusion Planning in Cooperative Driving
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多智能体协同驾驶任务，旨在提升规划器的安全性和效率。通过结合场景一致的扩散预训练与稳定在线强化后训练，解决场景一致性差和在线优化困难的问题。**

- **链接: [https://arxiv.org/pdf/2604.11734](https://arxiv.org/pdf/2604.11734)**

> **作者:** Haojie Bai; Aimin Li; Ruoyu Yao; Xiongwei Zhao; Tingting Zhang; Xing Zhang; Lin Gao; and Jun Ma
>
> **摘要:** Closed-loop cooperative driving requires planners that generate realistic multimodal multi-agent trajectories while improving safety and traffic efficiency. Existing diffusion planners can model multimodal behaviors from demonstrations, but they often exhibit weak scene consistency and remain poorly aligned with closed-loop objectives; meanwhile, stable online post-training in reactive multi-agent environments remains difficult. We present Multi-ORFT, which couples scene-conditioned diffusion pre-training with stable online reinforcement post-training. In pre-training, the planner uses inter-agent self-attention, cross-attention, and AdaLN-Zero-based scene conditioning to improve scene consistency and road adherence of joint trajectories. In post-training, we formulate a two-level MDP that exposes step-wise reverse-kernel likelihoods for online optimization, and combine dense trajectory-level rewards with variance-gated group-relative policy optimization (VG-GRPO) to stabilize training. On the WOMD closed-loop benchmark, Multi-ORFT reduces collision rate from 2.04% to 1.89% and off-road rate from 1.68% to 1.36%, while increasing average speed from 8.36 to 8.61 m/s relative to the pre-trained planner, and it outperforms strong open-source baselines including SMART-large, SMART-tiny-CLSFT, and VBD on the primary safety and efficiency metrics. These results show that coupling scene-consistent denoising with stable online diffusion-policy optimization improves the reliability of closed-loop cooperative driving.
>
---
#### [replaced 017] Ro-SLM: Onboard Small Language Models for Robot Task Planning and Operation Code Generation
- **分类: cs.RO**

- **简介: 该论文提出Ro-SLM框架，解决机器人在受限环境下无法使用大模型的问题，通过微调小语言模型实现任务规划和代码生成。**

- **链接: [https://arxiv.org/pdf/2604.10929](https://arxiv.org/pdf/2604.10929)**

> **作者:** Wenhao Wang; Yanyan Li; Long Jiao; Jiawei Yuan
>
> **备注:** 25 pages, 2 figures, ACL 2026
>
> **摘要:** Recent advances in large language models (LLMs) provide robots with contextual reasoning abilities to comprehend human instructions. Yet, current LLM-enabled robots typically depend on cloud-based models or high-performance computing infrastructure, which limit their deployment on robots under unreliable internet environments or with constrained computational resources, such as UAVs and small ground vehicles. Thus, deploying fine-tuned small language models (SLMs) that support onboard deployment offers a promising alternative. This paper introduces Ro-SLM, a framework that enables reliable SLM-driven robot operation by distilling LLMs' knowledge and reasoning. Ro-SLM starts from dataset synthesis by leveraging LLMs to generate diverse task instructions, produce corresponding ground truth code with minimal human assistance, and augment instructions into real-world application scenarios. Ro-SLM is then fine-tuned with the dataset, in which LLM serves as a reward function to guide the training. Extensive experiments on UAV operation tasks demonstrate that Ro-SLM improves the performance of SLM from being incapable of supporting robotic task planning and code generation to achieving performance that approaches LLM.
>
---
#### [replaced 018] Mixed-Integer vs. Continuous Model Predictive Control for Binary Thrusters: A Comparative Study
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于航天控制任务，解决二进制推进器的控制问题。比较了混合整数MPC与连续MPC结合调制方法，评估其性能并提出改进方案。**

- **链接: [https://arxiv.org/pdf/2603.19796](https://arxiv.org/pdf/2603.19796)**

> **作者:** Franek Stark; Jakob Middelberg; Shubham Vyas
>
> **备注:** Accepted to CEAS EuroGNC 2026
>
> **摘要:** Binary on/off thrusters are commonly used for spacecraft attitude and position control during proximity operations. However, their discrete nature poses challenges for conventional continuous control methods. The control of these discrete actuators is either explicitly formulated as a mixed-integer optimization problem or handled in a two-layer approach, where a continuous controller's output is converted to binary commands using analog-to digital modulation techniques such as Delta-Sigma-modulation. This paper provides the first systematic comparison between these two paradigms for binary thruster control, contrasting continuous Model Predictive Control (MPC) with Delta-Sigma modulation against direct Mixed-Integer MPC (MIMPC) approaches. Furthermore, we propose a new variant of MPC for binary actuated systems, which is informed using the state of the Delta-Sigma Modulator. The two variations for the continuous MPC along with the MIMPC are evaluated through extensive simulations using ESA's REACSA platform. Results demonstrate that while all approaches perform similarly in high-thrust regimes, MIMPC achieves superior fuel efficiency in low-thrust conditions. Continuous MPC with modulation shows instabilities at higher thrust levels, while binary informed MPC, which incorporates modulator dynamics, improves robustness and reduces the efficiency gap to the MIMPC. It can be seen from the simulated and real-system experiments that MIMPC offers complete stability and fuel efficiency benefits, particularly for resource-constrained missions, while continuous control methods remain attractive for computationally limited applications.
>
---
#### [replaced 019] Latent Chain-of-Thought World Modeling for End-to-End Driving
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在提升复杂场景下的驾驶性能与安全。通过引入隐式链式思维（Latent-CoT）模型，结合动作与世界模型进行推理决策，提高推理效率与轨迹质量。**

- **链接: [https://arxiv.org/pdf/2512.10226](https://arxiv.org/pdf/2512.10226)**

> **作者:** Shuhan Tan; Kashyap Chitta; Yuxiao Chen; Ran Tian; Yurong You; Yan Wang; Wenjie Luo; Yulong Cao; Philipp Krahenbuhl; Marco Pavone; Boris Ivanovic
>
> **备注:** Accepted to CVPR 2026
>
> **摘要:** Recent Vision-Language-Action (VLA) models for autonomous driving explore inference-time reasoning as a way to improve driving performance and safety in challenging scenarios. Most prior work uses natural language to express chain-of-thought (CoT) reasoning before producing driving actions. However, text may not be the most efficient representation for reasoning. In this work, we present Latent-CoT-Drive (LCDrive): a model that expresses CoT in a latent language that captures possible outcomes of the driving actions being considered. Our approach unifies CoT reasoning and decision making by representing both in an action-aligned latent space. Instead of natural language, the model reasons by interleaving (1) action-proposal tokens, which use the same vocabulary as the model's output actions; and (2) world model tokens, which are grounded in a learned latent world model and express future outcomes of these actions. We cold start latent CoT by supervising the model's action proposals and world model tokens based on ground-truth future rollouts of the scene. We then post-train with closed-loop reinforcement learning to strengthen reasoning capabilities. On a large-scale end-to-end driving benchmark, LCDrive achieves faster inference, better trajectory quality, and larger improvements from interactive reinforcement learning compared to both non-reasoning and text-reasoning baselines.
>
---
#### [replaced 020] Physically Accurate Rigid-Body Dynamics in Particle-Based Simulation
- **分类: cs.RO**

- **简介: 该论文属于物理仿真任务，旨在解决粒子模拟中刚体动力学物理准确性不足的问题。通过改进PBD方法，提出PBD-R，提升仿真精度与计算效率。**

- **链接: [https://arxiv.org/pdf/2603.14634](https://arxiv.org/pdf/2603.14634)**

> **作者:** Ava Abderezaei; Nataliya Nechyporenko; Joseph Miceli; Gilberto Briscoe-Martinez; Alessandro Roncone
>
> **备注:** Submitted to IROS 2026
>
> **摘要:** Robotics demands simulation that can reason about the diversity of real-world physical interactions, from rigid to deformable objects and fluids. Current simulators address this by stitching together multiple subsolvers for different material types, resulting in a compositional architecture that complicates physical reasoning. Particle-based simulators offer a compelling alternative, representing all materials through a single unified formulation that enables seamless cross-material interactions. Among particle-based simulators, position-based dynamics (PBD) is a popular solver known for its computational efficiency and visual plausibility. However, its lack of physical accuracy has limited its adoption in robotics. To leverage the benefits of particle-based solvers while meeting the physical fidelity demands of robotics, we introduce PBD-R, a revised PBD formulation that enforces physically accurate rigid-body dynamics through a novel momentum-conservation constraint and a modified velocity update. Additionally, we introduce a solver-agnostic benchmark with analytical solutions to evaluate physical accuracy. Using this benchmark, we show that PBD-R significantly outperforms PBD and achieves competitive accuracy with MuJoCo while requiring less computation.
>
---
#### [replaced 021] CLAW: Composable Language-Annotated Whole-body Motion Generation
- **分类: cs.RO**

- **简介: 该论文提出CLAW系统，用于生成语言标注的全身运动数据，解决人形机器人运动数据不足与物理可行性问题。**

- **链接: [https://arxiv.org/pdf/2604.11251](https://arxiv.org/pdf/2604.11251)**

> **作者:** Jianuo Cao; Yuxin Chen; Masayoshi Tomizuka
>
> **摘要:** Training language-conditioned whole-body controllers for humanoid robots requires large-scale datasets pairing motion trajectories with natural-language descriptions. Existing approaches based on motion capture are costly and limited in diversity, while text-to-motion generative models produce purely kinematic outputs that are not guaranteed to be physically feasible. Therefore, we present CLAW, an interactive web-based pipeline for scalable generation of language-annotated whole-body motion data for the Unitree G1 humanoid robot. CLAW treats the motion modes of a kinematic planner as composable building blocks, each parameterized by movement, heading, speed, pelvis height and duration, and provides two browser-based interfaces -- a real-time keyboard mode and a timeline-based sequence editor -- for exploratory and batch data collection. A low-level whole-body controller tracks the planner's kinematic references in MuJoCo simulation, producing physically grounded trajectories recorded at 50Hz. Simultaneously, a deterministic template-based annotation engine generates diverse natural-language descriptions at multiple stylistic registers for every segment and for the full trajectory. We release the system as open source to support scalable generation of language-motion paired data for humanoid robot learning.
>
---
#### [replaced 022] STRONG-VLA: Decoupled Robustness Learning for Vision-Language-Action Models under Multimodal Perturbations
- **分类: cs.RO**

- **简介: 该论文属于视觉-语言-动作模型的鲁棒性研究，解决多模态扰动下的任务执行问题。提出STRONG-VLA框架，通过分阶段训练提升模型鲁棒性与任务精度。**

- **链接: [https://arxiv.org/pdf/2604.10055](https://arxiv.org/pdf/2604.10055)**

> **作者:** Yuhan Xie; Yuping Yan; Yunqi Zhao; Handing Wang; Yaochu Jin
>
> **摘要:** Despite their strong performance in embodied tasks, recent Vision-Language-Action (VLA) models remain highly fragile under multimodal perturbations, where visual corruption and linguistic noise jointly induce distribution shifts that degrade task-level execution. Existing robustness approaches typically rely on joint training with perturbed data, treating robustness as a static objective, which leads to conflicting optimization between robustness and task fidelity. In this work, we propose STRONG-VLA, a decoupled fine-tuning framework that explicitly separates robustness acquisition from task-aligned refinement. In Stage I, the model is exposed to a curriculum of multimodal perturbations with increasing difficulty, enabling progressive robustness learning under controlled distribution shifts. In Stage II, the model is re-aligned with clean task distributions to recover execution fidelity while preserving robustness. We further establish a comprehensive benchmark with 28 perturbation types spanning both textual and visual modalities, grounded in realistic sources of sensor noise, occlusion, and instruction corruption. Extensive experiments on the LIBERO benchmark show that STRONG-VLA consistently improves task success rates across multiple VLA architectures. On OpenVLA, our method achieves gains of up to 12.60% under seen perturbations and 7.77% under unseen perturbations. Notably, similar or larger improvements are observed on OpenVLA-OFT (+14.48% / +13.81%) and pi0 (+16.49% / +5.58%), demonstrating strong cross-architecture generalization. Real-world experiments on an AIRBOT robotic platform further validate its practical effectiveness. These results highlight the importance of decoupled optimization for multimodal robustness and establish STRONG-VLA as a simple yet principled framework for robust embodied control.
>
---
#### [replaced 023] Iterative Compositional Data Generation for Robot Control
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人控制任务，旨在解决多任务数据生成难题。通过构建可组合的扩散Transformer模型，实现未见任务的零样本数据生成与策略学习。**

- **链接: [https://arxiv.org/pdf/2512.10891](https://arxiv.org/pdf/2512.10891)**

> **作者:** Anh-Quan Pham; Marcel Hussing; Shubhankar P. Patankar; Dani S. Bassett; Jorge Mendez-Mendez; Eric Eaton
>
> **摘要:** Collecting robotic manipulation data is expensive, making it impractical to acquire demonstrations for the combinatorially large space of tasks that arise in multi-object, multi-robot, and multi-environment settings. While recent generative models can synthesize useful data for individual tasks, they do not exploit the compositional structure of robotic domains and struggle to generalize to unseen task combinations. We propose a semantic compositional diffusion transformer that factorizes transitions into robot-, object-, obstacle-, and objective-specific components and learns their interactions through attention. Once trained on a limited subset of tasks, we show that our model can zero-shot generate high-quality transitions from which we can learn control policies for unseen task combinations. Then, we introduce an iterative self-improvement procedure in which synthetic data is validated via offline reinforcement learning and incorporated into subsequent training rounds. Our approach substantially improves zero-shot performance over monolithic and hard-coded compositional baselines, ultimately solving nearly all held-out tasks and demonstrating the emergence of meaningful compositional structure in the learned representations.
>
---
#### [replaced 024] BLaDA: Bridging Language to Functional Dexterous Actions within 3DGS Fields
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出BLaDA框架，解决功能性灵巧操作任务中的语义-姿态耦合问题，通过语言解析、三维定位和姿态生成实现零样本功能操作。**

- **链接: [https://arxiv.org/pdf/2604.08410](https://arxiv.org/pdf/2604.08410)**

> **作者:** Fan Yang; Wenrui Chen; Guorun Yan; Ruize Liao; Wanjun Jia; Dongsheng Luo; Jiacheng Lin; Kailun Yang; Zhiyong Li; Yaonan Wang
>
> **备注:** Code will be publicly available at this https URL
>
> **摘要:** In unstructured environments, functional dexterous grasping calls for the tight integration of semantic understanding, precise 3D functional localization, and physically interpretable execution. Modular hierarchical methods are more controllable and interpretable than end-to-end VLA approaches, but existing ones still rely on predefined affordance labels and lack the tight semantic--pose coupling needed for functional dexterous manipulation. To address this, we propose BLaDA (Bridging Language to Dexterous Actions in 3DGS fields), an interpretable zero-shot framework that grounds open-vocabulary instructions as perceptual and control constraints for functional dexterous manipulation. BLaDA establishes an interpretable reasoning chain by first parsing natural language into a structured sextuple of manipulation constraints via a Knowledge-guided Language Parsing (KLP) module. To achieve pose-consistent spatial reasoning, we introduce the Triangular Functional Point Localization (TriLocation) module, which utilizes 3D Gaussian Splatting as a continuous scene representation and identifies functional regions under triangular geometric constraints. Finally, the 3D Keypoint Grasp Matrix Transformation Execution (KGT3D+) module decodes these semantic-geometric constraints into physically plausible wrist poses and finger-level commands. Extensive experiments on complex benchmarks demonstrate that BLaDA significantly outperforms existing methods in both affordance grounding precision and the success rate of functional manipulation across diverse categories and tasks. Code will be publicly available at this https URL.
>
---
#### [replaced 025] Improved particle swarm optimization algorithm: multi-target trajectory optimization for swarm drones
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多无人机轨迹优化任务，旨在解决动态环境中实时路径规划问题。提出PE-PSO算法，提升收敛性和适应性，实现高效协同飞行。**

- **链接: [https://arxiv.org/pdf/2507.13647](https://arxiv.org/pdf/2507.13647)**

> **作者:** Minze Li; Wei Zhao; Ran Chen; Mingqiang Wei
>
> **备注:** New experiments have revealed systematic errors in the original data
>
> **摘要:** Real-time trajectory planning for unmanned aerial vehicles (UAVs) in dynamic environments remains a key challenge due to high computational demands and the need for fast, adaptive responses. Traditional Particle Swarm Optimization (PSO) methods, while effective for offline planning, often struggle with premature convergence and latency in real-time scenarios. To overcome these limitations, we propose PE-PSO, an enhanced PSO-based online trajectory planner. The method introduces a persistent exploration mechanism to preserve swarm diversity and an entropy-based parameter adjustment strategy to dynamically adapt optimization behavior. UAV trajectories are modeled using B-spline curves, which ensure path smoothness while reducing optimization complexity. To extend this capability to UAV swarms, we develop a multi-agent framework that combines genetic algorithm (GA)-based task allocation with distributed PE-PSO, supporting scalable and coordinated trajectory generation. The distributed architecture allows for parallel computation and decentralized control, enabling effective cooperation among agents while maintaining real-time performance. Comprehensive simulations demonstrate that the proposed framework outperforms conventional PSO and other swarm-based planners across several metrics, including trajectory quality, energy efficiency, obstacle avoidance, and computation time. These results confirm the effectiveness and applicability of PE-PSO in real-time multi-UAV operations under complex environmental conditions.
>
---
#### [replaced 026] Unconventional Hexacopters via Evolution and Learning: Performance Gains and New Insights
- **分类: cs.RO**

- **简介: 该论文属于机器人与进化计算领域，旨在通过演化与学习结合，设计性能更优的六旋翼无人机，解决传统结构性能不足的问题，并探索两者交互机制。**

- **链接: [https://arxiv.org/pdf/2505.14129](https://arxiv.org/pdf/2505.14129)**

> **作者:** Jed Muff; Keiichi Ito; Elijah H. W. Ang; Karine Miras; A.E. Eiben
>
> **备注:** 16 pages, 14 figures, Published in evostar2026. Code: this https URL. Videos: this https URL
>
> **摘要:** Evolution and learning have historically been interrelated topics, and their interplay is attracting increased interest lately. The emerging new factor in this trend is morphological evolution, the evolution of physical forms within embodied AI systems such as robots. In this study, we investigate a system of hexacopter-type drones with evolvable morphologies and learnable controllers and make contributions to two fields. For aerial robotics, we demonstrate that the combination of evolution and learning can deliver non-conventional drones that significantly outperform the traditional hexacopter on several tasks that are more complex than previously considered in the literature. For the field of Evolutionary Computing, we introduce novel metrics and perform new analyses into the interaction of morphological evolution and learning, uncovering hitherto unidentified effects. Our analysis tools are domain-agnostic, making a methodological contribution towards building solid foundations for embodied AI systems that integrate evolution and learning.
>
---
#### [replaced 027] Mixed-Density Diffuser: Efficient Planning with Non-Uniform Temporal Resolution
- **分类: cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决轨迹规划中时间密度不均的问题。提出Mixed-Density Diffuser模型，通过调整不同时间段的密度提升规划效率与效果。**

- **链接: [https://arxiv.org/pdf/2510.23026](https://arxiv.org/pdf/2510.23026)**

> **作者:** Crimson Stambaugh; Rajesh P. N. Rao
>
> **备注:** European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (ESANN, 2026)
>
> **摘要:** Recent studies demonstrate that diffusion planners benefit from sparse-step planning over single-step planning. Training models to skip steps in their trajectories helps capture long-term dependencies without additional memory or computational cost. However, predicting excessively sparse plans degrades performance. We hypothesize this temporal density threshold is non-uniform across a planning horizon and that certain parts of a predicted trajectory should be more densely generated. We propose Mixed-Density Diffuser (MDD), a diffusion planner where the densities throughout the horizon are tunable hyperparameters. We show that MDD surpasses the SOTA Diffusion Veteran (DV) framework across the Maze2D, Franka Kitchen, and Antmaze Datasets for Deep Data-Driven Reinforcement Learning (D4RL) task domains, achieving a new SOTA on the D4RL benchmark.
>
---
#### [replaced 028] TriDeliver: Cooperative Air-Ground Instant Delivery with UAVs, Couriers, and Crowdsourced Ground Vehicles
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于即时配送任务，旨在解决单一配送方式效率低的问题。通过整合无人机、快递员和众包车辆，提出TriDeliver框架，提升配送效率与降低成本。**

- **链接: [https://arxiv.org/pdf/2604.09049](https://arxiv.org/pdf/2604.09049)**

> **作者:** Junhui Gao; Yan Pan; Qianru Wang; Wenzhe Hou; Yiqin Deng; Liangliang Jiang; Yuguang Fang
>
> **摘要:** Instant delivery, shipping items before critical deadlines, is essential in daily life. While multiple delivery agents, such as couriers, Unmanned Aerial Vehicles (UAVs), and crowdsourced agents, have been widely employed, each of them faces inherent limitations (e.g., low efficiency/labor shortages, flight control, and dynamic capabilities, respectively), preventing them from meeting the surging demands alone. This paper proposes TriDeliver, the first hierarchical cooperative framework, integrating human couriers, UAVs, and crowdsourced ground vehicles (GVs) for efficient instant delivery. To obtain the initial scheduling knowledge for GVs and UAVs as well as improve the cooperative delivery performance, we design a Transfer Learning (TL)-based algorithm to extract delivery knowledge from couriers' behavioral history and transfer their knowledge to UAVs and GVs with fine-tunings, which is then used to dispatch parcels for efficient delivery. Evaluated on one-month real-world trajectory and delivery datasets, it has been demonstrated that 1) by integrating couriers, UAVs, and crowdsourced GVs, TriDeliver reduces the delivery cost by $65.8\%$ versus state-of-the-art cooperative delivery by UAVs and couriers; 2) TriDeliver achieves further improvements in terms of delivery time ($-17.7\%$), delivery cost ($-9.8\%$), and impacts on original tasks of crowdsourced GVs ($-43.6\%$), even with the representation of the transferred knowledge by simple neural networks, respectively.
>
---
#### [replaced 029] Relative Pose Estimation for Nonholonomic Robot Formation with UWB-IO Measurements (Extended version)
- **分类: cs.RO**

- **简介: 该论文属于多机器人编队控制任务，解决非完整机器人在无共同参考系下的相对位姿估计问题。通过UWB和IO数据实现局部相对定位，并设计分布式控制器验证方法有效性。**

- **链接: [https://arxiv.org/pdf/2411.05481](https://arxiv.org/pdf/2411.05481)**

> **作者:** Kunrui Ze; Wei Wang; Shuoyu Yue; Guibin Sun; Kexin Liu; Jinhu Lü
>
> **备注:** 17 pages, 26 figures
>
> **摘要:** This article studies the problem of distributed formation control for multiple robots by using onboard ultra wide band (UWB) distance and inertial odometer (IO) measurements. Although this problem has been widely studied, a fundamental limitation of most works is that they require each robot's pose and sensor measurements are expressed in a common reference frame. However, it is inapplicable for nonholonomic robot formations due to the practical difficulty of aligning IO measurements of individual robot in a common frame. To address this problem, firstly, a concurrent-learning based estimator is firstly proposed to achieve relative localization between neighboring robots in a local frame. Different from most relative localization methods in a global frame, both relative position and orientation in a local frame are estimated with only UWB ranging and IO measurements. Secondly, to deal with information loss caused by directed communication topology, a cooperative localization algorithm is introduced to estimate the relative pose to the leader robot. Thirdly, based on the theoretical results on relative pose estimation, a distributed formation tracking controller is proposed for nonholonomic robots. Both 3D and 2D real-world experiments conducted on aerial robots and grounded robots are provided to demonstrate the effectiveness of the proposed method.
>
---
