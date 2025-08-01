# 机器人 cs.RO

- **最新发布 35 篇**

- **更新 31 篇**

## 最新发布

#### [new 001] User Experience Estimation in Human-Robot Interaction Via Multi-Instance Learning of Multimodal Social Signals
- **分类: cs.RO; cs.CV; cs.HC**

- **简介: 该论文属于人机交互中的用户体验（UX）评估任务，旨在解决如何准确估计用户在与机器人互动中的多维度体验问题。论文通过融合多模态社交信号（如面部表情和语音），构建了一个基于Transformer的模型，并采用多实例学习框架捕捉交互的短期与长期模式，提升了UX估计的准确性，实验表明其效果优于人类评估者。**

- **链接: [http://arxiv.org/pdf/2507.23544v1](http://arxiv.org/pdf/2507.23544v1)**

> **作者:** Ryo Miyoshi; Yuki Okafuji; Takuya Iwamoto; Junya Nakanishi; Jun Baba
>
> **备注:** This paper has been accepted for presentation at IEEE/RSJ International Conference on Intelligent Robots and Systems 2025 (IROS 2025)
>
> **摘要:** In recent years, the demand for social robots has grown, requiring them to adapt their behaviors based on users' states. Accurately assessing user experience (UX) in human-robot interaction (HRI) is crucial for achieving this adaptability. UX is a multi-faceted measure encompassing aspects such as sentiment and engagement, yet existing methods often focus on these individually. This study proposes a UX estimation method for HRI by leveraging multimodal social signals. We construct a UX dataset and develop a Transformer-based model that utilizes facial expressions and voice for estimation. Unlike conventional models that rely on momentary observations, our approach captures both short- and long-term interaction patterns using a multi-instance learning framework. This enables the model to capture temporal dynamics in UX, providing a more holistic representation. Experimental results demonstrate that our method outperforms third-party human evaluators in UX estimation.
>
---
#### [new 002] DuLoc: Life-Long Dual-Layer Localization in Changing and Dynamic Expansive Scenarios
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶中的定位任务，旨在解决长期动态环境中基于激光雷达的定位漂移和鲁棒性不足问题。作者提出了DuLoc方法，融合离线地图与实时局部地图，并结合惯性里程计和速度模型，提升定位精度与适应性。**

- **链接: [http://arxiv.org/pdf/2507.23660v1](http://arxiv.org/pdf/2507.23660v1)**

> **作者:** Haoxuan Jiang; Peicong Qian; Yusen Xie; Xiaocong Li; Ming Liu; Jun Ma
>
> **摘要:** LiDAR-based localization serves as a critical component in autonomous systems, yet existing approaches face persistent challenges in balancing repeatability, accuracy, and environmental adaptability. Traditional point cloud registration methods relying solely on offline maps often exhibit limited robustness against long-term environmental changes, leading to localization drift and reliability degradation in dynamic real-world scenarios. To address these challenges, this paper proposes DuLoc, a robust and accurate localization method that tightly couples LiDAR-inertial odometry with offline map-based localization, incorporating a constant-velocity motion model to mitigate outlier noise in real-world scenarios. Specifically, we develop a LiDAR-based localization framework that seamlessly integrates a prior global map with dynamic real-time local maps, enabling robust localization in unbounded and changing environments. Extensive real-world experiments in ultra unbounded port that involve 2,856 hours of operational data across 32 Intelligent Guided Vehicles (IGVs) are conducted and reported in this study. The results attained demonstrate that our system outperforms other state-of-the-art LiDAR localization systems in large-scale changing outdoor environments.
>
---
#### [new 003] GSFusion:Globally Optimized LiDAR-Inertial-Visual Mapping for Gaussian Splatting
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于三维重建与SLAM任务，旨在解决3D高斯点绘制（3DGS）在计算负载高、弱纹理/光照环境失效及短程操作的局限性。作者提出GSFusion，融合LiDAR、惯性与视觉信息，通过全局位姿优化、像素感知初始化和有界Sigmoid约束，实现高精度、高效的实时地图构建。**

- **链接: [http://arxiv.org/pdf/2507.23273v1](http://arxiv.org/pdf/2507.23273v1)**

> **作者:** Jaeseok Park; Chanoh Park; Minsu Kim; Soohwan Kim
>
> **摘要:** While 3D Gaussian Splatting (3DGS) has revolutionized photorealistic mapping, conventional approaches based on camera sensor, even RGB-D, suffer from fundamental limitations such as high computational load, failure in environments with poor texture or illumination, and short operational ranges. LiDAR emerges as a robust alternative, but its integration with 3DGS introduces new challenges, such as the need for exceptional global alignment for photorealistic quality and prolonged optimization times caused by sparse data. To address these challenges, we propose GSFusion, an online LiDAR-Inertial-Visual mapping system that ensures high-precision map consistency through a surfel-to-surfel constraint in the global pose-graph optimization. To handle sparse data, our system employs a pixel-aware Gaussian initialization strategy for efficient representation and a bounded sigmoid constraint to prevent uncontrolled Gaussian growth. Experiments on public and our datasets demonstrate our system outperforms existing 3DGS SLAM systems in terms of rendering quality and map-building efficiency.
>
---
#### [new 004] H-RDT: Human Manipulation Enhanced Bimanual Robotic Manipulation
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于机器人操作任务，旨在解决机器人操作中缺乏高质量演示数据的问题。论文提出H-RDT模型，利用人类操作数据预训练，再通过跨形态微调提升双手机器人的操作能力，显著优于现有方法。**

- **链接: [http://arxiv.org/pdf/2507.23523v1](http://arxiv.org/pdf/2507.23523v1)**

> **作者:** Hongzhe Bi; Lingxuan Wu; Tianwei Lin; Hengkai Tan; Zhizhong Su; Hang Su; Jun Zhu
>
> **摘要:** Imitation learning for robotic manipulation faces a fundamental challenge: the scarcity of large-scale, high-quality robot demonstration data. Recent robotic foundation models often pre-train on cross-embodiment robot datasets to increase data scale, while they face significant limitations as the diverse morphologies and action spaces across different robot embodiments make unified training challenging. In this paper, we present H-RDT (Human to Robotics Diffusion Transformer), a novel approach that leverages human manipulation data to enhance robot manipulation capabilities. Our key insight is that large-scale egocentric human manipulation videos with paired 3D hand pose annotations provide rich behavioral priors that capture natural manipulation strategies and can benefit robotic policy learning. We introduce a two-stage training paradigm: (1) pre-training on large-scale egocentric human manipulation data, and (2) cross-embodiment fine-tuning on robot-specific data with modular action encoders and decoders. Built on a diffusion transformer architecture with 2B parameters, H-RDT uses flow matching to model complex action distributions. Extensive evaluations encompassing both simulation and real-world experiments, single-task and multitask scenarios, as well as few-shot learning and robustness assessments, demonstrate that H-RDT outperforms training from scratch and existing state-of-the-art methods, including Pi0 and RDT, achieving significant improvements of 13.9% and 40.5% over training from scratch in simulation and real-world experiments, respectively. The results validate our core hypothesis that human manipulation data can serve as a powerful foundation for learning bimanual robotic manipulation policies.
>
---
#### [new 005] Benchmarking Massively Parallelized Multi-Task Reinforcement Learning for Robotics Tasks
- **分类: cs.RO**

- **简介: 该论文属于机器人多任务强化学习任务，旨在解决现有方法在大规模并行训练中的局限性。作者构建了MTBench基准，包含50个操作任务和20个运动任务，基于GPU加速的IsaacGym仿真器，结合多种先进算法，提供统一评估框架，探索大规模并行化对多任务强化学习的影响。**

- **链接: [http://arxiv.org/pdf/2507.23172v1](http://arxiv.org/pdf/2507.23172v1)**

> **作者:** Vira Joshi; Zifan Xu; Bo Liu; Peter Stone; Amy Zhang
>
> **备注:** RLC 2025
>
> **摘要:** Multi-task Reinforcement Learning (MTRL) has emerged as a critical training paradigm for applying reinforcement learning (RL) to a set of complex real-world robotic tasks, which demands a generalizable and robust policy. At the same time, \emph{massively parallelized training} has gained popularity, not only for significantly accelerating data collection through GPU-accelerated simulation but also for enabling diverse data collection across multiple tasks by simulating heterogeneous scenes in parallel. However, existing MTRL research has largely been limited to off-policy methods like SAC in the low-parallelization regime. MTRL could capitalize on the higher asymptotic performance of on-policy algorithms, whose batches require data from the current policy, and as a result, take advantage of massive parallelization offered by GPU-accelerated simulation. To bridge this gap, we introduce a massively parallelized $\textbf{M}$ulti-$\textbf{T}$ask $\textbf{Bench}$mark for robotics (MTBench), an open-sourced benchmark featuring a broad distribution of 50 manipulation tasks and 20 locomotion tasks, implemented using the GPU-accelerated simulator IsaacGym. MTBench also includes four base RL algorithms combined with seven state-of-the-art MTRL algorithms and architectures, providing a unified framework for evaluating their performance. Our extensive experiments highlight the superior speed of evaluating MTRL approaches using MTBench, while also uncovering unique challenges that arise from combining massive parallelism with MTRL. Code is available at $\href{https://github.com/Viraj-Joshi/MTBench}{ https://github.com/Viraj-Joshi/MTBench}$
>
---
#### [new 006] Stereo 3D Gaussian Splatting SLAM for Outdoor Urban Scenes
- **分类: cs.RO**

- **简介: 该论文属于SLAM任务，旨在解决户外城市场景中缺乏高效、高精度三维重建与定位的问题。现有方法多限于室内或依赖主动传感器。论文提出BGS-SLAM，首个基于双目RGB图像的3D高斯点绘SLAM系统，无需LiDAR或主动传感器，通过预训练立体网络估计深度，优化三维高斯表示，提升户外复杂环境下的跟踪与建图性能。**

- **链接: [http://arxiv.org/pdf/2507.23677v1](http://arxiv.org/pdf/2507.23677v1)**

> **作者:** Xiaohan Li; Ziren Gong; Fabio Tosi; Matteo Poggi; Stefano Mattoccia; Dong Liu; Jun Wu
>
> **摘要:** 3D Gaussian Splatting (3DGS) has recently gained popularity in SLAM applications due to its fast rendering and high-fidelity representation. However, existing 3DGS-SLAM systems have predominantly focused on indoor environments and relied on active depth sensors, leaving a gap for large-scale outdoor applications. We present BGS-SLAM, the first binocular 3D Gaussian Splatting SLAM system designed for outdoor scenarios. Our approach uses only RGB stereo pairs without requiring LiDAR or active sensors. BGS-SLAM leverages depth estimates from pre-trained deep stereo networks to guide 3D Gaussian optimization with a multi-loss strategy enhancing both geometric consistency and visual quality. Experiments on multiple datasets demonstrate that BGS-SLAM achieves superior tracking accuracy and mapping performance compared to other 3DGS-based solutions in complex outdoor environments.
>
---
#### [new 007] Assessing the Alignment of Automated Vehicle Decisions with Human Reasons
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶决策评估任务，旨在解决自动驾驶车辆在日常驾驶中如何符合人类伦理期望的问题。论文提出了一种基于人类理由的轨迹评估框架，通过量化法律、舒适等因素，评估自动驾驶行为与人类决策的对齐程度，提升了决策透明度与可解释性。**

- **链接: [http://arxiv.org/pdf/2507.23324v1](http://arxiv.org/pdf/2507.23324v1)**

> **作者:** Lucas Elbert Suryana; Saeed Rahmani; Simeon Craig Calvert; Arkady Zgonnikov; Bart van Arem
>
> **备注:** This version incorporates revisions based on peer-review feedback from a prior submission. The work has not yet been accepted and is being prepared for resubmission
>
> **摘要:** A key challenge in deploying automated vehicles (AVs) is ensuring they make appropriate decisions in ethically challenging everyday driving situations. While much attention has been paid to rare, high-stakes dilemmas such as trolley problems, similar tensions also arise in routine scenarios, such as navigating empty intersections, where multiple human considerations, including legality and comfort, often conflict. Current AV planning systems typically rely on rigid rules, which struggle to balance these competing considerations and can lead to behaviour that misaligns with human expectations. This paper proposes a novel reasons-based trajectory evaluation framework that operationalises the tracking condition of Meaningful Human Control (MHC). The framework models the reasons of human agents, such as regulatory compliance, as quantifiable functions and evaluates how well candidate AV trajectories align with these reasons. By assigning adjustable weights to agent priorities and integrating a balance function to discourage the exclusion of any agent, the framework supports interpretable decision evaluation. Through a real-world-inspired overtaking scenario, we show how this approach reveals tensions, for instance between regulatory compliance, efficiency, and comfort. The framework functions as a modular evaluation layer over existing planning algorithms. It offers a transparent tool for assessing ethical alignment in everyday scenarios and provides a practical step toward implementing MHC in real-world AV deployment.
>
---
#### [new 008] Human-Exoskeleton Kinematic Calibration to Improve Hand Tracking for Dexterous Teleoperation
- **分类: cs.RO; cs.HC; cs.SY; eess.SY**

- **简介: 该论文属于机器人与人机交互任务，旨在解决手部外骨骼在灵巧遥操作中因个体差异和穿戴不一致导致的追踪不准问题。作者提出一种基于冗余关节传感与残差加权优化的个性化校准方法，并通过实验验证其在关节角度与指尖定位上的提升效果。**

- **链接: [http://arxiv.org/pdf/2507.23592v1](http://arxiv.org/pdf/2507.23592v1)**

> **作者:** Haiyun Zhang; Stefano Dalla Gasperina; Saad N. Yousaf; Toshimitsu Tsuboi; Tetsuya Narita; Ashish D. Deshpande
>
> **备注:** 8 pages, 10 figures, submitted to RA-L
>
> **摘要:** Hand exoskeletons are critical tools for dexterous teleoperation and immersive manipulation interfaces, but achieving accurate hand tracking remains a challenge due to user-specific anatomical variability and donning inconsistencies. These issues lead to kinematic misalignments that degrade tracking performance and limit applicability in precision tasks. We propose a subject-specific calibration framework for exoskeleton-based hand tracking that uses redundant joint sensing and a residual-weighted optimization strategy to estimate virtual link parameters. Implemented on the Maestro exoskeleton, our method improves joint angle and fingertip position estimation across users with varying hand geometries. We introduce a data-driven approach to empirically tune cost function weights using motion capture ground truth, enabling more accurate and consistent calibration across participants. Quantitative results from seven subjects show substantial reductions in joint and fingertip tracking errors compared to uncalibrated and evenly weighted models. Qualitative visualizations using a Unity-based virtual hand further confirm improvements in motion fidelity. The proposed framework generalizes across exoskeleton designs with closed-loop kinematics and minimal sensing, and lays the foundation for high-fidelity teleoperation and learning-from-demonstration applications.
>
---
#### [new 009] Simulation-based planning of Motion Sequences for Automated Procedure Optimization in Multi-Robot Assembly Cells
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于多机器人装配任务的运动规划与优化。旨在解决多机器人装配中配置规划与协调问题，提出一种基于仿真的分解式运动规划方法，优化核心操作调度与避障路径规划，提升装配效率。**

- **链接: [http://arxiv.org/pdf/2507.23270v1](http://arxiv.org/pdf/2507.23270v1)**

> **作者:** Loris Schneider; Marc Ungen; Elias Huber; Jan-Felix Klein
>
> **摘要:** Reconfigurable multi-robot cells offer a promising approach to meet fluctuating assembly demands. However, the recurrent planning of their configurations introduces new challenges, particularly in generating optimized, coordinated multi-robot motion sequences that minimize the assembly duration. This work presents a simulation-based method for generating such optimized sequences. The approach separates assembly steps into task-related core operations and connecting traverse operations. While core operations are constrained and predetermined, traverse operations offer substantial optimization potential. Scheduling the core operations is formulated as an optimization problem, requiring feasible traverse operations to be integrated using a decomposition-based motion planning strategy. Several solution techniques are explored, including a sampling heuristic, tree-based search and gradient-free optimization. For motion planning, a decomposition method is proposed that identifies specific areas in the schedule, which can be solved independently with modified centralized path planning algorithms. The proposed method generates efficient and collision-free multi-robot assembly procedures that outperform a baseline relying on decentralized, robot-individual motion planning. Its effectiveness is demonstrated through simulation experiments.
>
---
#### [new 010] Beyond Rigid AI: Towards Natural Human-Machine Symbiosis for Interoperative Surgical Assistance
- **分类: cs.RO; cs.AI; cs.HC**

- **简介: 该论文属于医疗机器人与人工智能任务，旨在解决手术环境中人机交互不自然、AI系统灵活性差的问题。作者提出了一种基于大语言模型、图像分割和跟踪模型的“感知代理”，可实现更自然、实时的人机协作手术辅助，提升对已知和新出现对象的分割灵活性。**

- **链接: [http://arxiv.org/pdf/2507.23088v1](http://arxiv.org/pdf/2507.23088v1)**

> **作者:** Lalithkumar Seenivasan; Jiru Xu; Roger D. Soberanis Mukul; Hao Ding; Grayson Byrd; Yu-Chun Ku; Jose L. Porras; Masaru Ishii; Mathias Unberath
>
> **摘要:** Emerging surgical data science and robotics solutions, especially those designed to provide assistance in situ, require natural human-machine interfaces to fully unlock their potential in providing adaptive and intuitive aid. Contemporary AI-driven solutions remain inherently rigid, offering limited flexibility and restricting natural human-machine interaction in dynamic surgical environments. These solutions rely heavily on extensive task-specific pre-training, fixed object categories, and explicit manual-prompting. This work introduces a novel Perception Agent that leverages speech-integrated prompt-engineered large language models (LLMs), segment anything model (SAM), and any-point tracking foundation models to enable a more natural human-machine interaction in real-time intraoperative surgical assistance. Incorporating a memory repository and two novel mechanisms for segmenting unseen elements, Perception Agent offers the flexibility to segment both known and unseen elements in the surgical scene through intuitive interaction. Incorporating the ability to memorize novel elements for use in future surgeries, this work takes a marked step towards human-machine symbiosis in surgical procedures. Through quantitative analysis on a public dataset, we show that the performance of our agent is on par with considerably more labor-intensive manual-prompting strategies. Qualitatively, we show the flexibility of our agent in segmenting novel elements (instruments, phantom grafts, and gauze) in a custom-curated dataset. By offering natural human-machine interaction and overcoming rigidity, our Perception Agent potentially brings AI-based real-time assistance in dynamic surgical environments closer to reality.
>
---
#### [new 011] Quantifying and Visualizing Sim-to-Real Gaps: Physics-Guided Regularization for Reproducibility
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人控制任务，旨在解决仿真到现实（sim-to-real）的性能差距问题。作者提出了一种基于物理引导的增益正则化方法，通过实际实验测量机器人等效比例增益，并在神经控制器训练中惩罚偏离该增益的行为，同时结合参数条件控制，提升了控制策略在真实硬件上的可重复性和性能。**

- **链接: [http://arxiv.org/pdf/2507.23445v1](http://arxiv.org/pdf/2507.23445v1)**

> **作者:** Yuta Kawachi
>
> **摘要:** Simulation-to-real transfer using domain randomization for robot control often relies on low-gear-ratio, backdrivable actuators, but these approaches break down when the sim-to-real gap widens. Inspired by the traditional PID controller, we reinterpret its gains as surrogates for complex, unmodeled plant dynamics. We then introduce a physics-guided gain regularization scheme that measures a robot's effective proportional gains via simple real-world experiments. Then, we penalize any deviation of a neural controller's local input-output sensitivities from these values during training. To avoid the overly conservative bias of naive domain randomization, we also condition the controller on the current plant parameters. On an off-the-shelf two-wheeled balancing robot with a 110:1 gearbox, our gain-regularized, parameter-conditioned RNN achieves angular settling times in hardware that closely match simulation. At the same time, a purely domain-randomized policy exhibits persistent oscillations and a substantial sim-to-real gap. These results demonstrate a lightweight, reproducible framework for closing sim-to-real gaps on affordable robotic hardware.
>
---
#### [new 012] DRACo-SLAM2: Distributed Robust Acoustic Communication-efficient SLAM for Imaging Sonar EquippedUnderwater Robot Teams with Object Graph Matching
- **分类: cs.RO**

- **简介: 论文提出DRACo-SLAM2，一种用于多水下机器人声呐的分布式SLAM框架。任务是提升水下多机器人系统的协同建图与定位能力，解决无几何先验信息下的闭环检测效率问题。工作包括引入对象图表示声呐地图、采用图匹配实现高效闭环检测，并改进扫描匹配方法处理相近误差场景。**

- **链接: [http://arxiv.org/pdf/2507.23629v1](http://arxiv.org/pdf/2507.23629v1)**

> **作者:** Yewei Huang; John McConnell; Xi Lin; Brendan Englot
>
> **摘要:** We present DRACo-SLAM2, a distributed SLAM framework for underwater robot teams equipped with multibeam imaging sonar. This framework improves upon the original DRACo-SLAM by introducing a novel representation of sonar maps as object graphs and utilizing object graph matching to achieve time-efficient inter-robot loop closure detection without relying on prior geometric information. To better-accommodate the needs and characteristics of underwater scan matching, we propose incremental Group-wise Consistent Measurement Set Maximization (GCM), a modification of Pairwise Consistent Measurement Set Maximization (PCM), which effectively handles scenarios where nearby inter-robot loop closures share similar registration errors. The proposed approach is validated through extensive comparative analyses on simulated and real-world datasets.
>
---
#### [new 013] Learning to Prune Branches in Modern Tree-Fruit Orchards
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于农业机器人任务，旨在解决现代高产果园中冬季树木修剪的难题。作者开发了一种闭环视觉控制器，通过仅使用腕部摄像头的光流图像，引导机械剪枝器在复杂环境中精准剪枝。在仿真和实际场景中实现了30%的成功率，约为理想规划器的一半。**

- **链接: [http://arxiv.org/pdf/2507.23015v1](http://arxiv.org/pdf/2507.23015v1)**

> **作者:** Abhinav Jain; Cindy Grimm; Stefan Lee
>
> **摘要:** Dormant tree pruning is labor-intensive but essential to maintaining modern highly-productive fruit orchards. In this work we present a closed-loop visuomotor controller for robotic pruning. The controller guides the cutter through a cluttered tree environment to reach a specified cut point and ensures the cutters are perpendicular to the branch. We train the controller using a novel orchard simulation that captures the geometric distribution of branches in a target apple orchard configuration. Unlike traditional methods requiring full 3D reconstruction, our controller uses just optical flow images from a wrist-mounted camera. We deploy our learned policy in simulation and the real-world for an example V-Trellis envy tree with zero-shot transfer, achieving a 30% success rate -- approximately half the performance of an oracle planner.
>
---
#### [new 014] In-between Motion Generation Based Multi-Style Quadruped Robot Locomotion
- **分类: cs.RO**

- **简介: 该论文属于机器人运动控制任务，旨在解决四足机器人运动风格单一和参考数据不足的问题。论文提出了一种基于中间运动生成的多风格运动框架，结合条件变分自编码器与对抗先验算法，实现多样化的运动生成与稳定控制。**

- **链接: [http://arxiv.org/pdf/2507.23053v1](http://arxiv.org/pdf/2507.23053v1)**

> **作者:** Yuanhao Chen; Liu Zhao; Ji Ma; Peng Lu
>
> **摘要:** Quadruped robots face persistent challenges in achieving versatile locomotion due to limitations in reference motion data diversity. To address these challenges, this approach introduces an in-between motion generation based multi-style quadruped robot locomotion framework, integrating synergistic advances in motion generation and imitation learning. Our approach establishes a unified pipeline addressing two fundamental aspects: First, we propose a CVAE based motion generator, synthesizing multi-style dynamically feasible locomotion sequences between arbitrary start and end states. By embedding physical constraints and leveraging joint poses based phase manifold continuity, this component produces physically plausible motions spanning multiple gait modalities while ensuring kinematic compatibility with robotic morphologies. Second, we adopt the adversarial motion priors algorithm. We validate the effectiveness of generated motion data in enhancing controller stability and improving velocity tracking performance. The proposed framework demonstrates significant improvements in velocity tracking and deployment stability. We successfully deploy the framework on a real-world quadruped robot, and the experimental validation confirms the framework's capability to generate and execute complex motion profiles, including gallop, tripod, trotting and pacing.
>
---
#### [new 015] A Certifably Correct Algorithm for Generalized Robot-World and Hand-Eye Calibration
- **分类: cs.RO**

- **简介: 该论文属于机器人传感器标定任务，旨在解决广义机器人-世界与手-眼标定（RWHEC）问题。作者提出了一种快速且全局最优的算法，支持多传感器和目标姿态估计，并适用于单目相机。论文还推导了新的可识别性条件，提供了全局最优性保证，并开发了局部求解器及开源实现。**

- **链接: [http://arxiv.org/pdf/2507.23045v1](http://arxiv.org/pdf/2507.23045v1)**

> **作者:** Emmett Wise; Pushyami Kaveti; Qilong Chen; Wenhao Wang; Hanumant Singh; Jonathan Kelly; David M. Rosen; Matthew Giamou
>
> **备注:** 25 pages, 10 figures, submitted to the International Journal of Robotics Research
>
> **摘要:** Automatic extrinsic sensor calibration is a fundamental problem for multi-sensor platforms. Reliable and general-purpose solutions should be computationally efficient, require few assumptions about the structure of the sensing environment, and demand little effort from human operators. Since the engineering effort required to obtain accurate calibration parameters increases with the number of sensors deployed, robotics researchers have pursued methods requiring few assumptions about the sensing environment and minimal effort from human operators. In this work, we introduce a fast and certifiably globally optimal algorithm for solving a generalized formulation of the $\textit{robot-world and hand-eye calibration}$ (RWHEC) problem. The formulation of RWHEC presented is "generalized" in that it supports the simultaneous estimation of multiple sensor and target poses, and permits the use of monocular cameras that, alone, are unable to measure the scale of their environments. In addition to demonstrating our method's superior performance over existing solutions, we derive novel identifiability criteria and establish $\textit{a priori}$ guarantees of global optimality for problem instances with bounded measurement errors. We also introduce a complementary Lie-algebraic local solver for RWHEC and compare its performance with our global method and prior art. Finally, we provide a free and open-source implementation of our algorithms and experiments.
>
---
#### [new 016] Whisker-based Active Tactile Perception for Contour Reconstruction
- **分类: cs.RO**

- **简介: 该论文属于机器人触觉感知任务，旨在解决基于胡须传感器的物体轮廓重建中缺乏主动控制的问题。通过设计磁感应胡须传感器和主动控制策略，结合梯度下降与贝叶斯滤波，实现了亚毫米级精度的物体表面跟踪与轮廓重建。**

- **链接: [http://arxiv.org/pdf/2507.23305v1](http://arxiv.org/pdf/2507.23305v1)**

> **作者:** Yixuan Dang; Qinyang Xu; Yu Zhang; Xiangtong Yao; Liding Zhang; Zhenshan Bing; Florian Roehrbein; Alois Knoll
>
> **摘要:** Perception using whisker-inspired tactile sensors currently faces a major challenge: the lack of active control in robots based on direct contact information from the whisker. To accurately reconstruct object contours, it is crucial for the whisker sensor to continuously follow and maintain an appropriate relative touch pose on the surface. This is especially important for localization based on tip contact, which has a low tolerance for sharp surfaces and must avoid slipping into tangential contact. In this paper, we first construct a magnetically transduced whisker sensor featuring a compact and robust suspension system composed of three flexible spiral arms. We develop a method that leverages a characterized whisker deflection profile to directly extract the tip contact position using gradient descent, with a Bayesian filter applied to reduce fluctuations. We then propose an active motion control policy to maintain the optimal relative pose of the whisker sensor against the object surface. A B-Spline curve is employed to predict the local surface curvature and determine the sensor orientation. Results demonstrate that our algorithm can effectively track objects and reconstruct contours with sub-millimeter accuracy. Finally, we validate the method in simulations and real-world experiments where a robot arm drives the whisker sensor to follow the surfaces of three different objects.
>
---
#### [new 017] Design of a bioinspired robophysical antenna for insect-scale tactile perception and navigation
- **分类: cs.RO**

- **简介: 该论文设计了一种受蟑螂启发的触觉机器人天线CITRAS，用于昆虫级机器人的感知与导航。论文属于机器人传感任务，旨在解决微型机器人在复杂环境中感知能力受限的问题。工作包括开发多段柔性传感器，实现距离、间隙和表面纹理的感知。**

- **链接: [http://arxiv.org/pdf/2507.23719v1](http://arxiv.org/pdf/2507.23719v1)**

> **作者:** Parker McDonnell; Lingsheng Meng; Hari Krishna Hariprasad; Alexander Hedrick; Eduardo Miscles; Samuel Gilinsky; Jean-Michel Mongeau; Kaushik Jayaram
>
> **摘要:** The American cockroach (Periplaneta americana) uses its soft antennae to guide decision making by extracting rich tactile information from tens of thousands of distributed mechanosensors. Although tactile sensors enable robust, autonomous perception and navigation in natural systems, replicating these capabilities in insect-scale robots remains challenging due to stringent size, weight, and power constraints that limit existing sensor technologies. To overcome these limitations, we introduce CITRAS (Cockroach Inspired Tactile Robotic Antenna Sensor), a bioinspired, multi-segmented, compliant laminate sensor with embedded capacitive angle sensors. CITRAS is compact (73.7x15.6x2.1 mm), lightweight (491 mg), and low-power (32 mW), enabling seamless integration with miniature robotic platforms. The segmented compliant structure passively bends in response to environmental stimuli, achieving accurate hinge angle measurements with maximum errors of just 0.79 degree (quasistatic bending) and 3.58 degree (dynamic bending). Experimental evaluations demonstrate CITRAS' multifunctional tactile perception capabilities: predicting base-to-tip distances with 7.75 % error, estimating environmental gap widths with 6.73 % error, and distinguishing surface textures through differential sensor response. The future integration of this bioinspired tactile antenna in insect-scale robots addresses critical sensing gaps, promising enhanced autonomous exploration, obstacle avoidance, and environmental mapping in complex, confined environments.
>
---
#### [new 018] Quadratic Programming-Based Posture Manipulation and Thrust-vectoring for Agile Dynamic Walking on Narrow Pathways
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决四足机器人在狭窄路径上稳定行走的问题。通过在Husky β机器人上应用姿态操控与推力矢量控制，结合基于质心动力学模型的QP控制器，实现对机器人前后动力的稳定控制，并模拟横向推力恢复以增强动态平衡能力。**

- **链接: [http://arxiv.org/pdf/2507.23203v1](http://arxiv.org/pdf/2507.23203v1)**

> **作者:** Chenghao Wang; Eric Sihite; Kaushik Venkatesh Krishnamurthy; Shreyansh Pitroda; Adarsh Salagame; Alireza Ramezani; Morteza Gharib
>
> **摘要:** There has been significant advancement in legged robot's agility where they can show impressive acrobatic maneuvers, such as parkour. These maneuvers rely heavily on posture manipulation. To expand the stability and locomotion plasticity, we use the multi-modal ability in our legged-aerial platform, the Husky Beta, to perform thruster-assisted walking. This robot has thrusters on each of its sagittal knee joints which can be used to stabilize its frontal dynamic as it walks. In this work, we perform a simulation study of quadruped narrow-path walking with Husky $\beta$, where the robot will utilize its thrusters to stably walk on a narrow path. The controller is designed based on a centroidal dynamics model with thruster and foot ground contact forces as inputs. These inputs are regulated using a QP solver to be used in a model predictive control framework. In addition to narrow-path walking, we also perform a lateral push-recovery simulation to study how the thrusters can be used to stabilize the frontal dynamics.
>
---
#### [new 019] Distributed AI Agents for Cognitive Underwater Robot Autonomy
- **分类: cs.RO; cs.AI; cs.MA**

- **简介: 该论文属于水下机器人认知自主性任务，旨在解决复杂环境中机器人自主决策能力不足的问题。论文提出UROSA架构，通过分布式AI代理实现多模态感知、动态任务规划与实时决策，结合ROS 2框架提升适应性与可靠性，验证了其在未知水下环境中的优越性能。**

- **链接: [http://arxiv.org/pdf/2507.23735v1](http://arxiv.org/pdf/2507.23735v1)**

> **作者:** Markus Buchholz; Ignacio Carlucho; Michele Grimaldi; Yvan R. Petillot
>
> **摘要:** Achieving robust cognitive autonomy in robots navigating complex, unpredictable environments remains a fundamental challenge in robotics. This paper presents Underwater Robot Self-Organizing Autonomy (UROSA), a groundbreaking architecture leveraging distributed Large Language Model AI agents integrated within the Robot Operating System 2 (ROS 2) framework to enable advanced cognitive capabilities in Autonomous Underwater Vehicles. UROSA decentralises cognition into specialised AI agents responsible for multimodal perception, adaptive reasoning, dynamic mission planning, and real-time decision-making. Central innovations include flexible agents dynamically adapting their roles, retrieval-augmented generation utilising vector databases for efficient knowledge management, reinforcement learning-driven behavioural optimisation, and autonomous on-the-fly ROS 2 node generation for runtime functional extensibility. Extensive empirical validation demonstrates UROSA's promising adaptability and reliability through realistic underwater missions in simulation and real-world deployments, showing significant advantages over traditional rule-based architectures in handling unforeseen scenarios, environmental uncertainties, and novel mission objectives. This work not only advances underwater autonomy but also establishes a scalable, safe, and versatile cognitive robotics framework capable of generalising to a diverse array of real-world applications.
>
---
#### [new 020] villa-X: Enhancing Latent Action Modeling in Vision-Language-Action Models
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人操控策略学习任务，旨在提升视觉-语言-动作模型（VLA）中的潜在动作建模。论文提出villa-X框架，改进潜在动作的学习与整合方法，增强模型泛化能力。论文在多个模拟环境和真实机器人设置中验证了方法的有效性。**

- **链接: [http://arxiv.org/pdf/2507.23682v1](http://arxiv.org/pdf/2507.23682v1)**

> **作者:** Xiaoyu Chen; Hangxing Wei; Pushi Zhang; Chuheng Zhang; Kaixin Wang; Yanjiang Guo; Rushuai Yang; Yucen Wang; Xinquan Xiao; Li Zhao; Jianyu Chen; Jiang Bian
>
> **备注:** Project page: https://aka.ms/villa-x
>
> **摘要:** Visual-Language-Action (VLA) models have emerged as a popular paradigm for learning robot manipulation policies that can follow language instructions and generalize to novel scenarios. Recent work has begun to explore the incorporation of latent actions, an abstract representation of visual change between two frames, into VLA pre-training. In this paper, we introduce villa-X, a novel Visual-Language-Latent-Action (ViLLA) framework that advances latent action modeling for learning generalizable robot manipulation policies. Our approach improves both how latent actions are learned and how they are incorporated into VLA pre-training. Together, these contributions enable villa-X to achieve superior performance across simulated environments including SIMPLER and LIBERO, as well as on two real-world robot setups including gripper and dexterous hand manipulation. We believe the ViLLA paradigm holds significant promise, and that our villa-X provides a strong foundation for future research.
>
---
#### [new 021] Can LLM-Reasoning Models Replace Classical Planning? A Benchmark Study
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于任务规划领域，探讨大语言模型（LLMs）能否替代传统规划方法。论文通过对比LLMs与经典规划器Fast Downward在多个基准任务上的表现，评估LLMs在复杂任务中的规划能力，发现其在简单任务上有效，但在资源管理、状态跟踪和约束满足方面存在不足，提出结合LLMs与传统规划的未来方向。**

- **链接: [http://arxiv.org/pdf/2507.23589v1](http://arxiv.org/pdf/2507.23589v1)**

> **作者:** Kai Goebel; Patrik Zips
>
> **摘要:** Recent advancements in Large Language Models have sparked interest in their potential for robotic task planning. While these models demonstrate strong generative capabilities, their effectiveness in producing structured and executable plans remains uncertain. This paper presents a systematic evaluation of a broad spectrum of current state of the art language models, each directly prompted using Planning Domain Definition Language domain and problem files, and compares their planning performance with the Fast Downward planner across a variety of benchmarks. In addition to measuring success rates, we assess how faithfully the generated plans translate into sequences of actions that can actually be executed, identifying both strengths and limitations of using these models in this setting. Our findings show that while the models perform well on simpler planning tasks, they continue to struggle with more complex scenarios that require precise resource management, consistent state tracking, and strict constraint compliance. These results underscore fundamental challenges in applying language models to robotic planning in real world environments. By outlining the gaps that emerge during execution, we aim to guide future research toward combined approaches that integrate language models with classical planners in order to enhance the reliability and scalability of planning in autonomous robotics.
>
---
#### [new 022] Learning to Drift with Individual Wheel Drive: Maneuvering Autonomous Vehicle at the Handling Limits
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于自动驾驶控制任务，旨在解决车辆在极限操控下的漂移动作控制问题。利用强化学习与GPU加速仿真，结合领域随机化方法，实现了仿真到实车的有效迁移，并在定制的四轮独立驱动RC车上验证了复杂漂移场景的轨迹跟踪性能。**

- **链接: [http://arxiv.org/pdf/2507.23339v1](http://arxiv.org/pdf/2507.23339v1)**

> **作者:** Yihan Zhou; Yiwen Lu; Bo Yang; Jiayun Li; Yilin Mo
>
> **摘要:** Drifting, characterized by controlled vehicle motion at high sideslip angles, is crucial for safely handling emergency scenarios at the friction limits. While recent reinforcement learning approaches show promise for drifting control, they struggle with the significant simulation-to-reality gap, as policies that perform well in simulation often fail when transferred to physical systems. In this paper, we present a reinforcement learning framework with GPU-accelerated parallel simulation and systematic domain randomization that effectively bridges the gap. The proposed approach is validated on both simulation and a custom-designed and open-sourced 1/10 scale Individual Wheel Drive (IWD) RC car platform featuring independent wheel speed control. Experiments across various scenarios from steady-state circular drifting to direction transitions and variable-curvature path following demonstrate that our approach achieves precise trajectory tracking while maintaining controlled sideslip angles throughout complex maneuvers in both simulated and real-world environments.
>
---
#### [new 023] A Unified Perception-Language-Action Framework for Adaptive Autonomous Driving
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 论文提出了一种统一的感知-语言-行动（PLA）框架，用于自动驾驶，融合多传感器数据与大语言模型（GPT-4.1），提升系统在复杂环境中的适应性、可解释性与安全性。属于自动驾驶任务，旨在解决现有系统在开放世界中适应性差、泛化能力弱及语义理解不足的问题。**

- **链接: [http://arxiv.org/pdf/2507.23540v1](http://arxiv.org/pdf/2507.23540v1)**

> **作者:** Yi Zhang; Erik Leo Haß; Kuo-Yi Chao; Nenad Petrovic; Yinglei Song; Chengdong Wu; Alois Knoll
>
> **摘要:** Autonomous driving systems face significant challenges in achieving human-like adaptability, robustness, and interpretability in complex, open-world environments. These challenges stem from fragmented architectures, limited generalization to novel scenarios, and insufficient semantic extraction from perception. To address these limitations, we propose a unified Perception-Language-Action (PLA) framework that integrates multi-sensor fusion (cameras, LiDAR, radar) with a large language model (LLM)-augmented Vision-Language-Action (VLA) architecture, specifically a GPT-4.1-powered reasoning core. This framework unifies low-level sensory processing with high-level contextual reasoning, tightly coupling perception with natural language-based semantic understanding and decision-making to enable context-aware, explainable, and safety-bounded autonomous driving. Evaluations on an urban intersection scenario with a construction zone demonstrate superior performance in trajectory tracking, speed prediction, and adaptive planning. The results highlight the potential of language-augmented cognitive frameworks for advancing the safety, interpretability, and scalability of autonomous driving systems.
>
---
#### [new 024] Multi-Waypoint Path Planning and Motion Control for Non-holonomic Mobile Robots in Agricultural Applications
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 论文研究农业环境中非完整移动机器人的多航点路径规划与运动控制。旨在解决无结构农田中高效导航问题，结合Dubins TSP与非线性模型预测控制，实现平滑、短路径规划与精确航点跟踪，减少16%路径长度，提升自主导航效率。**

- **链接: [http://arxiv.org/pdf/2507.23350v1](http://arxiv.org/pdf/2507.23350v1)**

> **作者:** Mahmoud Ghorab; Matthias Lorenzen
>
> **备注:** 6 pages
>
> **摘要:** There is a growing demand for autonomous mobile robots capable of navigating unstructured agricultural environments. Tasks such as weed control in meadows require efficient path planning through an unordered set of coordinates while minimizing travel distance and adhering to curvature constraints to prevent soil damage and protect vegetation. This paper presents an integrated navigation framework combining a global path planner based on the Dubins Traveling Salesman Problem (DTSP) with a Nonlinear Model Predictive Control (NMPC) strategy for local path planning and control. The DTSP generates a minimum-length, curvature-constrained path that efficiently visits all targets, while the NMPC leverages this path to compute control signals to accurately reach each waypoint. The system's performance was validated through comparative simulation analysis on real-world field datasets, demonstrating that the coupled DTSP-based planner produced smoother and shorter paths, with a reduction of about 16% in the provided scenario, compared to decoupled methods. Based thereon, the NMPC controller effectively steered the robot to the desired waypoints, while locally optimizing the trajectory and ensuring adherence to constraints. These findings demonstrate the potential of the proposed framework for efficient autonomous navigation in agricultural environments.
>
---
#### [new 025] Scalable Multi-Task Reinforcement Learning for Generalizable Spatial Intelligence in Visuomotor Agents
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多任务强化学习与空间智能研究，旨在解决视觉-运动智能体在不同环境中泛化能力差的问题。论文提出跨视角目标设定方法，并利用《Minecraft》环境实现自动化任务生成与大规模训练，验证了强化学习在提升空间推理与零样本迁移能力上的有效性。**

- **链接: [http://arxiv.org/pdf/2507.23698v1](http://arxiv.org/pdf/2507.23698v1)**

> **作者:** Shaofei Cai; Zhancun Mu; Haiwen Xia; Bowei Zhang; Anji Liu; Yitao Liang
>
> **摘要:** While Reinforcement Learning (RL) has achieved remarkable success in language modeling, its triumph hasn't yet fully translated to visuomotor agents. A primary challenge in RL models is their tendency to overfit specific tasks or environments, thereby hindering the acquisition of generalizable behaviors across diverse settings. This paper provides a preliminary answer to this challenge by demonstrating that RL-finetuned visuomotor agents in Minecraft can achieve zero-shot generalization to unseen worlds. Specifically, we explore RL's potential to enhance generalizable spatial reasoning and interaction capabilities in 3D worlds. To address challenges in multi-task RL representation, we analyze and establish cross-view goal specification as a unified multi-task goal space for visuomotor policies. Furthermore, to overcome the significant bottleneck of manual task design, we propose automated task synthesis within the highly customizable Minecraft environment for large-scale multi-task RL training, and we construct an efficient distributed RL framework to support this. Experimental results show RL significantly boosts interaction success rates by $4\times$ and enables zero-shot generalization of spatial reasoning across diverse environments, including real-world settings. Our findings underscore the immense potential of RL training in 3D simulated environments, especially those amenable to large-scale task generation, for significantly advancing visuomotor agents' spatial reasoning.
>
---
#### [new 026] Vision-Language Fusion for Real-Time Autonomous Driving: Goal-Centered Cross-Attention of Camera, HD-Map, & Waypoints
- **分类: cs.CV; cs.AI; cs.LG; cs.RO; I.4.8; I.2.10; I.2.6; C.3.3; I.4.9**

- **简介: 该论文属于自动驾驶任务，旨在解决复杂环境中几何精度与语义理解分离的问题。论文提出XYZ-Drive模型，通过视觉-语言融合与目标中心注意力机制，结合摄像头、高精地图和路径点，实现高效实时驾驶决策，提升了成功率与安全性。**

- **链接: [http://arxiv.org/pdf/2507.23064v1](http://arxiv.org/pdf/2507.23064v1)**

> **作者:** Santosh Patapati; Trisanth Srinivasan; Murari Ambati
>
> **备注:** 5 pages
>
> **摘要:** Autonomous cars need geometric accuracy and semantic understanding to navigate complex environments, yet most stacks handle them separately. We present XYZ-Drive, a single vision-language model that reads a front-camera frame, a 25m $\times$ 25m overhead map, and the next waypoint, then outputs steering and speed. A lightweight goal-centered cross-attention layer lets waypoint tokens highlight relevant image and map patches, supporting both action and textual explanations, before the fused tokens enter a partially fine-tuned LLaMA-3.2 11B model. On the MD-NEX Outdoor-Driving benchmark XYZ-Drive attains 95% success and 0.80 Success weighted by Path Length (SPL), surpassing PhysNav-DG by 15%. and halving collisions, all while significantly improving efficiency by using only a single branch. Sixteen ablations explain the gains. Removing any modality (vision, waypoint, map) drops success by up to 11%, confirming their complementary roles and rich connections. Replacing goal-centered attention with simple concatenation cuts 3% in performance, showing query-based fusion injects map knowledge more effectively. Keeping the transformer frozen loses 5%, showing the importance of fine-tuning when applying VLMs for specific tasks such as autonomous driving. Coarsening map resolution from 10 cm to 40 cm blurs lane edges and raises crash rate. Overall, these results demonstrate that early, token-level fusion of intent and map layout enables accurate, transparent, real-time driving.
>
---
#### [new 027] Online Estimation of Table-Top Grown Strawberry Mass in Field Conditions with Occlusions
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于计算机视觉与农业自动化任务，旨在解决田间条件下桌面种植草莓因遮挡和姿态变化导致的质量估计难题。作者提出了一种结合RGB-D传感与深度学习的在线质量估计算法，采用YOLOv8-Seg实例分割、CycleGAN修复遮挡区域、倾角校正及多项式回归模型，实现了高精度非接触式草莓质量估计。**

- **链接: [http://arxiv.org/pdf/2507.23487v1](http://arxiv.org/pdf/2507.23487v1)**

> **作者:** Jinshan Zhen; Yuanyue Ge; Tianxiao Zhu; Hui Zhao; Ya Xiong
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** Accurate mass estimation of table-top grown strawberries under field conditions remains challenging due to frequent occlusions and pose variations. This study proposes a vision-based pipeline integrating RGB-D sensing and deep learning to enable non-destructive, real-time and online mass estimation. The method employed YOLOv8-Seg for instance segmentation, Cycle-consistent generative adversarial network (CycleGAN) for occluded region completion, and tilt-angle correction to refine frontal projection area calculations. A polynomial regression model then mapped the geometric features to mass. Experiments demonstrated mean mass estimation errors of 8.11% for isolated strawberries and 10.47% for occluded cases. CycleGAN outperformed large mask inpainting (LaMa) model in occlusion recovery, achieving superior pixel area ratios (PAR) (mean: 0.978 vs. 1.112) and higher intersection over union (IoU) scores (92.3% vs. 47.7% in the [0.9-1] range). This approach addresses critical limitations of traditional methods, offering a robust solution for automated harvesting and yield monitoring with complex occlusion patterns.
>
---
#### [new 028] A blessing or a burden? Exploring worker perspectives of using a social robot in a church
- **分类: cs.HC; cs.RO**

- **简介: 该论文探讨在教堂中引入社交机器人对工人的影响，属社会机器人应用研究任务。旨在解决非营利组织引入机器人时的社会影响问题。通过访谈15名教堂工作人员，分析其对机器人使用的看法，发现其态度复杂，关注同理心责任与潜在问题，但也认可机器人在信息提供和减轻琐碎任务中的潜力。**

- **链接: [http://arxiv.org/pdf/2507.22903v1](http://arxiv.org/pdf/2507.22903v1)**

> **作者:** Andrew Blair; Peggy Gregory; Mary Ellen Foster
>
> **备注:** Accepted by the 2025 34th IEEE International Conference on Robot and Human Interactive Communication (ROMAN)
>
> **摘要:** Recent technological advances have allowed robots to assist in the service sector, and consequently accelerate job and sector transformation. Less attention has been paid to the use of robots in real-world organisations where social benefits, as opposed to profits, are the primary motivator. To explore these opportunities, we have partnered with a working church and visitor attraction. We conducted interviews with 15 participants from a range of stakeholder groups within the church to understand worker perspectives of introducing a social robot to the church and analysed the results using reflexive thematic analysis. Findings indicate mixed responses to the use of a robot, with participants highlighting the empathetic responsibility the church has towards people and the potential for unintended consequences. However, information provision and alleviation of menial or mundane tasks were identified as potential use cases. This highlights the need to consider not only the financial aspects of robot introduction, but also how social and intangible values shape what roles a robot should take on within an organisation.
>
---
#### [new 029] Policy Learning from Large Vision-Language Model Feedback without Reward Modeling
- **分类: cs.LG; cs.RO**

- **简介: 论文提出PLARE方法，属于离线强化学习任务，旨在解决无需手动设计奖励函数的机器人策略学习问题。利用大视觉语言模型（VLM）提供偏好反馈，通过对比学习目标直接训练策略，实验证明其性能优于现有方法，并在真实机器人任务中有效。**

- **链接: [http://arxiv.org/pdf/2507.23391v1](http://arxiv.org/pdf/2507.23391v1)**

> **作者:** Tung M. Luu; Donghoon Lee; Younghwan Lee; Chang D. Yoo
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** Offline reinforcement learning (RL) provides a powerful framework for training robotic agents using pre-collected, suboptimal datasets, eliminating the need for costly, time-consuming, and potentially hazardous online interactions. This is particularly useful in safety-critical real-world applications, where online data collection is expensive and impractical. However, existing offline RL algorithms typically require reward labeled data, which introduces an additional bottleneck: reward function design is itself costly, labor-intensive, and requires significant domain expertise. In this paper, we introduce PLARE, a novel approach that leverages large vision-language models (VLMs) to provide guidance signals for agent training. Instead of relying on manually designed reward functions, PLARE queries a VLM for preference labels on pairs of visual trajectory segments based on a language task description. The policy is then trained directly from these preference labels using a supervised contrastive preference learning objective, bypassing the need to learn explicit reward models. Through extensive experiments on robotic manipulation tasks from the MetaWorld, PLARE achieves performance on par with or surpassing existing state-of-the-art VLM-based reward generation methods. Furthermore, we demonstrate the effectiveness of PLARE in real-world manipulation tasks with a physical robot, further validating its practical applicability.
>
---
#### [new 030] iLearnRobot: An Interactive Learning-Based Multi-Modal Robot with Continuous Improvement
- **分类: cs.HC; cs.AI; cs.CV; cs.RO**

- **简介: 该论文属于机器人交互学习任务，旨在解决机器人部署后难以适应新场景的问题。论文提出iLearnRobot系统，基于多模态大语言模型，通过与非专家用户的自然对话进行持续学习，结合问题链和双模态检索模块，提升机器人适应性和性能，实现无缝用户体验。**

- **链接: [http://arxiv.org/pdf/2507.22896v1](http://arxiv.org/pdf/2507.22896v1)**

> **作者:** Kohou Wang; ZhaoXiang Liu; Lin Bai; Kun Fan; Xiang Liu; Huan Hu; Kai Wang; Shiguo Lian
>
> **备注:** 17 pages, 12 figures
>
> **摘要:** It is crucial that robots' performance can be improved after deployment, as they are inherently likely to encounter novel scenarios never seen before. This paper presents an innovative solution: an interactive learning-based robot system powered by a Multi-modal Large Language Model(MLLM). A key feature of our system is its ability to learn from natural dialogues with non-expert users. We also propose chain of question to clarify the exact intent of the question before providing an answer and dual-modality retrieval modules to leverage these interaction events to avoid repeating same mistakes, ensuring a seamless user experience before model updates, which is in contrast to current mainstream MLLM-based robotic systems. Our system marks a novel approach in robotics by integrating interactive learning, paving the way for superior adaptability and performance in diverse environments. We demonstrate the effectiveness and improvement of our method through experiments, both quantitively and qualitatively.
>
---
#### [new 031] Early Goal-Guided Multi-Scale Fusion for Real-Time Vision-Language Driving
- **分类: cs.CV; cs.AI; cs.LG; cs.MM; cs.RO; I.2.6; I.2.9; I.2.10; C.3.3**

- **简介: 该论文属于自动驾驶任务，旨在提升车辆在复杂场景中的实时导航能力与安全性。论文提出了NovaDrive，一种单分支视觉-语言架构，融合图像、高精地图、LiDAR和文本路径点信息，通过跨注意力机制与平滑损失优化路径，减少碰撞并提高行驶效率。**

- **链接: [http://arxiv.org/pdf/2507.23042v1](http://arxiv.org/pdf/2507.23042v1)**

> **作者:** Santosh Patapati; Trisanth Srinivasan
>
> **备注:** 6 pages
>
> **摘要:** Autonomous vehicles must react in milliseconds while reasoning about road geometry and traffic intent to navigate complex situations. We introduce NovaDrive, a single-branch vision-language architecture that processes front-camera images, HD-map tiles, LiDAR depth, and textual waypoints in a single branch. A lightweight, two-stage cross-attention block first aligns waypoint tokens with the HD map, then refines attention over fine-grained image and depth patches. Coupled with a novel smoothness loss that discourages abrupt steering and speed changes, this design eliminates the need for recurrent memory. We fine-tune the top 15 layers of an 11B LLaMA-3.2 vision-language backbone, enabling real-time inference. On the nuScenes / Waymo subset of the MD-NEX Outdoor benchmark, NovaDrive raises success rate to 84% (+4%), boosts path-efficiency (SPL) to 0.66 (+0.11), and reduces collision frequency from 2.6% to 1.2% (-1.4%) relative to the previous state-of-the-art. Our ablations confirm that waypoint tokens, partial VLM fine-tuning, and the cross-attention fusion each contribute the most to these gains. Beyond safety, NovaDrive's shorter routes (resulting from the novel smoothness loss) translate to lower fuel or battery usage, pointing toward leaner, more easily updated driving stacks. NovaDrive can be extended to other embodied-AI domains as well.
>
---
#### [new 032] SimuRA: Towards General Goal-Oriented Agent via Simulative Reasoning Architecture with LLM-Based World Model
- **分类: cs.AI; cs.CL; cs.LG; cs.RO**

- **简介: 该论文属于人工智能代理任务，旨在解决当前AI代理在通用性和可扩展性上的不足。论文提出SimuRA架构，通过基于大语言模型的模拟世界模型实现规划，提升目标导向任务的表现。实验表明其在复杂网页浏览任务中显著提高成功率。**

- **链接: [http://arxiv.org/pdf/2507.23773v1](http://arxiv.org/pdf/2507.23773v1)**

> **作者:** Mingkai Deng; Jinyu Hou; Yilin Shen; Hongxia Jin; Graham Neubig; Zhiting Hu; Eric Xing
>
> **摘要:** AI agents built on large language models (LLMs) hold enormous promise, but current practice focuses on a one-task-one-agent approach, which not only falls short of scalability and generality, but also suffers from the fundamental limitations of autoregressive LLMs. On the other hand, humans are general agents who reason by mentally simulating the outcomes of their actions and plans. Moving towards a more general and powerful AI agent, we introduce SimuRA, a goal-oriented architecture for generalized agentic reasoning. Based on a principled formulation of optimal agent in any environment, \modelname overcomes the limitations of autoregressive reasoning by introducing a world model for planning via simulation. The generalized world model is implemented using LLM, which can flexibly plan in a wide range of environments using the concept-rich latent space of natural language. Experiments on difficult web browsing tasks show that \modelname improves the success of flight search from 0\% to 32.2\%. World-model-based planning, in particular, shows consistent advantage of up to 124\% over autoregressive planning, demonstrating the advantage of world model simulation as a reasoning paradigm. We are excited about the possibility for training a single, general agent model based on LLMs that can act superintelligently in all environments. To start, we make SimuRA, a web-browsing agent built on \modelname with pretrained LLMs, available as a research demo for public testing.
>
---
#### [new 033] RAGNet: Large-scale Reasoning-based Affordance Segmentation Benchmark towards General Grasping
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于机器人抓取任务，旨在解决开放世界场景中缺乏大规模推理型可抓取区域预测数据的问题。作者构建了包含273k图像和26k推理指令的大规模基准数据集RAGNet，并提出基于可抓取区域感知的框架AffordanceNet，提升机器人根据语言指令在复杂环境中抓取物体的能力。**

- **链接: [http://arxiv.org/pdf/2507.23734v1](http://arxiv.org/pdf/2507.23734v1)**

> **作者:** Dongming Wu; Yanping Fu; Saike Huang; Yingfei Liu; Fan Jia; Nian Liu; Feng Dai; Tiancai Wang; Rao Muhammad Anwer; Fahad Shahbaz Khan; Jianbing Shen
>
> **备注:** Accepted by ICCV 2025. The code is at https://github.com/wudongming97/AffordanceNet
>
> **摘要:** General robotic grasping systems require accurate object affordance perception in diverse open-world scenarios following human instructions. However, current studies suffer from the problem of lacking reasoning-based large-scale affordance prediction data, leading to considerable concern about open-world effectiveness. To address this limitation, we build a large-scale grasping-oriented affordance segmentation benchmark with human-like instructions, named RAGNet. It contains 273k images, 180 categories, and 26k reasoning instructions. The images cover diverse embodied data domains, such as wild, robot, ego-centric, and even simulation data. They are carefully annotated with an affordance map, while the difficulty of language instructions is largely increased by removing their category name and only providing functional descriptions. Furthermore, we propose a comprehensive affordance-based grasping framework, named AffordanceNet, which consists of a VLM pre-trained on our massive affordance data and a grasping network that conditions an affordance map to grasp the target. Extensive experiments on affordance segmentation benchmarks and real-robot manipulation tasks show that our model has a powerful open-world generalization ability. Our data and code is available at https://github.com/wudongming97/AffordanceNet.
>
---
#### [new 034] Impact of a Lower Limb Exosuit Anchor Points on Energetics and Biomechanics
- **分类: physics.med-ph; cs.RO; eess.SP**

- **简介: 该论文研究下肢外骨骼锚点位置对能量消耗和生物力学的影响，属于外骨骼设计优化任务。通过六种实验配置，分析不同锚点对髋、膝、踝关节运动及肌肉激活的影响。结果显示锚点位置显著影响效果，最优位置因人而异，需个性化设计。**

- **链接: [http://arxiv.org/pdf/2507.23579v1](http://arxiv.org/pdf/2507.23579v1)**

> **作者:** Chiara Lambranzi; Giulia Oberti; Christian Di Natali; Darwin G. Caldwell; Manuela Galli; Elena De Momi; Jesùs Ortiz
>
> **备注:** 12 pages, 10 figures
>
> **摘要:** Anchor point placement is a crucial yet often overlooked aspect of exosuit design since it determines how forces interact with the human body. This work analyzes the impact of different anchor point positions on gait kinematics, muscular activation and energetic consumption. A total of six experiments were conducted with 11 subjects wearing the XoSoft exosuit, which assists hip flexion in five configurations. Subjects were instrumented with an IMU-based motion tracking system, EMG sensors, and a mask to measure metabolic consumption. The results show that positioning the knee anchor point on the posterior side while keeping the hip anchor on the anterior part can reduce muscle activation in the hip flexors by up to 10.21\% and metabolic expenditure by up to 18.45\%. Even if the only assisted joint was the hip, all the configurations introduced changes also in the knee and ankle kinematics. Overall, no single configuration was optimal across all subjects, suggesting that a personalized approach is necessary to transmit the assistance forces optimally. These findings emphasize that anchor point position does indeed have a significant impact on exoskeleton effectiveness and efficiency. However, these optimal positions are subject-specific to the exosuit design, and there is a strong need for future work to tailor musculoskeletal models to individual characteristics and validate these results in clinical populations.
>
---
#### [new 035] Experimentally-Driven Analysis of Stability in Connected Vehicle Platooning: Insights and Control Strategies
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文属于智能交通系统任务，旨在解决协同自适应巡航控制（CACC）在实际车辆编队中的稳定性问题。作者通过实验平台验证了CACC系统的有效性，填补了现有研究多依赖仿真而缺乏实物验证的空白。**

- **链接: [http://arxiv.org/pdf/2507.23078v1](http://arxiv.org/pdf/2507.23078v1)**

> **作者:** Niladri Dutta; Elham Abolfazli; Themistoklis Charalambous
>
> **摘要:** This paper presents the development of a tangible platform for demonstrating the practical implementation of cooperative adaptive cruise control (CACC) systems, an enhancement to the standard adaptive cruise control (ACC) concept by means of Vehicle-to-Everything (V2X) communication. It involves a detailed examination of existing longitudinal controllers and their performance in homogeneous vehicle platoons. Moreover, extensive tests are conducted using multiple autonomous experimental vehicle platform topologies to verify the effectiveness of the controller. The outcomes from both simulations and field tests affirm the substantial benefits of the proposed CACC platooning approach in longitudinal vehicle platooning scenarios. This research is crucial due to a notable gap in the existing literature; while numerous studies focus on simulated vehicle platooning systems, there is lack of research demonstrating these controllers on physical vehicle systems or robot platforms. This paper seeks to fill this gap by providing a practical demonstration of CACC systems in action, showcasing their potential for real-world application in intelligent transportation systems.
>
---
## 更新

#### [replaced 001] Tiny LiDARs for Manipulator Self-Awareness: Sensor Characterization and Initial Localization Experiments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.03449v2](http://arxiv.org/pdf/2503.03449v2)**

> **作者:** Giammarco Caroleo; Alessandro Albini; Daniele De Martini; Timothy D. Barfoot; Perla Maiolino
>
> **备注:** 7 pages, 6 figures, 3 tables, IEEE/RSJ International Conference on Intelligent Robots and Systems 2025 accepted paper
>
> **摘要:** For several tasks, ranging from manipulation to inspection, it is beneficial for robots to localize a target object in their surroundings. In this paper, we propose an approach that utilizes coarse point clouds obtained from miniaturized VL53L5CX Time-of-Flight (ToF) sensors (tiny LiDARs) to localize a target object in the robot's workspace. We first conduct an experimental campaign to calibrate the dependency of sensor readings on relative range and orientation to targets. We then propose a probabilistic sensor model, which we validate in an object pose estimation task using a Particle Filter (PF). The results show that the proposed sensor model improves the performance of the localization of the target object with respect to two baselines: one that assumes measurements are free from uncertainty and one in which the confidence is provided by the sensor datasheet.
>
---
#### [replaced 002] Humanoids in Hospitals: A Technical Study of Humanoid Robot Surrogates for Dexterous Medical Interventions
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.12725v2](http://arxiv.org/pdf/2503.12725v2)**

> **作者:** Soofiyan Atar; Xiao Liang; Calvin Joyce; Florian Richter; Wood Ricardo; Charles Goldberg; Preetham Suresh; Michael Yip
>
> **备注:** 8 pages
>
> **摘要:** The increasing demand for healthcare workers, driven by aging populations and labor shortages, presents a significant challenge for hospitals. Humanoid robots have the potential to alleviate these pressures by leveraging their human-like dexterity and adaptability to assist in medical procedures. This work conducted an exploratory study on the feasibility of humanoid robots performing direct clinical tasks through teleoperation. A bimanual teleoperation system was developed for the Unitree G1 Humanoid Robot, integrating high-fidelity pose tracking, custom grasping configurations, and an impedance controller to safely and precisely manipulate medical tools. The system is evaluated in seven diverse medical procedures, including physical examinations, emergency interventions, and precision needle tasks. Our results demonstrate that humanoid robots can successfully replicate critical aspects of human medical assessments and interventions, with promising quantitative performance in ventilation and ultrasound-guided tasks. However, challenges remain, including limitations in force output for procedures requiring high strength and sensor sensitivity issues affecting clinical accuracy. This study highlights the potential and current limitations of humanoid robots in hospital settings and lays the groundwork for future research on robotic healthcare integration.
>
---
#### [replaced 003] Estimating Scene Flow in Robot Surroundings with Distributed Miniaturized Time-of-Flight Sensors
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.02439v2](http://arxiv.org/pdf/2504.02439v2)**

> **作者:** Jack Sander; Giammarco Caroleo; Alessandro Albini; Perla Maiolino
>
> **备注:** 7 pages, 5 figures, 2 tables, 1 algorithm, IEEE RO-MAN 2025 accepted paper
>
> **摘要:** Tracking motions of humans or objects in the surroundings of the robot is essential to improve safe robot motions and reactions. In this work, we present an approach for scene flow estimation from low-density and noisy point clouds acquired from miniaturized Time of Flight (ToF) sensors distributed on the robot body. The proposed method clusters points from consecutive frames and applies Iterative Closest Point (ICP) to estimate a dense motion flow, with additional steps introduced to mitigate the impact of sensor noise and low-density data points. Specifically, we employ a fitness-based classification to distinguish between stationary and moving points and an inlier removal strategy to refine geometric correspondences. The proposed approach is validated in an experimental setup where 24 ToF are used to estimate the velocity of an object moving at different controlled speeds. Experimental results show that the method consistently approximates the direction of the motion and its magnitude with an error which is in line with sensor noise.
>
---
#### [replaced 004] Grasp EveryThing (GET): 1-DoF, 3-Fingered Gripper with Tactile Sensing for Robust Grasping
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.09771v3](http://arxiv.org/pdf/2505.09771v3)**

> **作者:** Michael Burgess; Edward H. Adelson
>
> **摘要:** We introduce the Grasp EveryThing (GET) gripper, a novel 1-DoF, 3-finger design for securely grasping objects of many shapes and sizes. Mounted on a standard parallel jaw actuator, the design features three narrow, tapered fingers arranged in a two-against-one configuration, where the two fingers converge into a V-shape. The GET gripper is more capable of conforming to object geometries and forming secure grasps than traditional designs with two flat fingers. Inspired by the principle of self-similarity, these V-shaped fingers enable secure grasping across a wide range of object sizes. Further to this end, fingers are parametrically designed for convenient resizing and interchangeability across robotic embodiments with a parallel jaw gripper. Additionally, we incorporate a rigid fingernail for ease in manipulating small objects. Tactile sensing can be integrated into the standalone finger via an externally-mounted camera. A neural network was trained to estimate normal force from tactile images with an average validation error of 1.3 N across a diverse set of geometries. In grasping 15 objects and performing 3 tasks via teleoperation, the GET fingers consistently outperformed standard flat fingers. All finger designs, compatible with multiple robotic embodiments, both incorporating and lacking tactile sensing, are available on GitHub.
>
---
#### [replaced 005] SHINE: Social Homology Identification for Navigation in Crowded Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2404.16705v3](http://arxiv.org/pdf/2404.16705v3)**

> **作者:** Diego Martinez-Baselga; Oscar de Groot; Luzia Knoedler; Luis Riazuelo; Javier Alonso-Mora; Luis Montano
>
> **备注:** This paper has been accepted for publication at The International Journal of Robotics Research. Please, when citing the paper, refer to the official manuscript with the following DOI: 10.1177/02783649251344639
>
> **摘要:** Navigating mobile robots in social environments remains a challenging task due to the intricacies of human-robot interactions. Most of the motion planners designed for crowded and dynamic environments focus on choosing the best velocity to reach the goal while avoiding collisions, but do not explicitly consider the high-level navigation behavior (avoiding through the left or right side, letting others pass or passing before others, etc.). In this work, we present a novel motion planner that incorporates topology distinct paths representing diverse navigation strategies around humans. The planner selects the topology class that imitates human behavior the best using a deep neural network model trained on real-world human motion data, ensuring socially intelligent and contextually aware navigation. Our system refines the chosen path through an optimization-based local planner in real time, ensuring seamless adherence to desired social behaviors. In this way, we decouple perception and local planning from the decision-making process. We evaluate the prediction accuracy of the network with real-world data. In addition, we assess the navigation capabilities in both simulation and a real-world platform, comparing it with other state-of-the-art planners. We demonstrate that our planner exhibits socially desirable behaviors and shows a smooth and remarkable performance.
>
---
#### [replaced 006] LaViPlan : Language-Guided Visual Path Planning with RLVR
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.12911v3](http://arxiv.org/pdf/2507.12911v3)**

> **作者:** Hayeon Oh
>
> **备注:** This paper has been withdrawn due to an internal institutional policy that prohibits preprint submissions to arXiv
>
> **摘要:** Out-of-distribution (OOD) scenarios in autonomous driving refer to situations that deviate from the training domain, often leading to unexpected and potentially hazardous behavior from planners that lack prior exposure to such cases. Recently, Vision-Language Models (VLMs) have been introduced into autonomous driving research for their promising generalization capabilities in OOD settings. Early studies demonstrated that VLMs could recognize OOD scenarios and generate user-level decisions such as "go straight" or "turn right." However, a new challenge has emerged due to the misalignment between the VLM's high-level decisions or visual reasoning expressed in language, and the low-level predicted trajectories interpreted as actions. In this paper, we propose LaViPlan, a framework that leverages Reinforcement Learning with Verifiable Rewards (RLVR) to optimize VLMs using planning-oriented metrics. This approach addresses the vision-language-action misalignment observed in existing VLMs fine-tuned via supervised learning, which can recognize driving scenarios but often produce context-unaware decisions. Experimental results demonstrate that our method improves situational awareness and decision-making under OOD conditions, highlighting its potential to mitigate the misalignment issue. This work introduces a promising post-training paradigm for VLM agents in the context of autonomous driving.
>
---
#### [replaced 007] Generalizable Motion Policies through Keypoint Parameterization and Transportation Maps
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2404.13458v2](http://arxiv.org/pdf/2404.13458v2)**

> **作者:** Giovanni Franzese; Ravi Prakash; Cosimo Della Santina; Jens Kober
>
> **备注:** This article was accepted at IEEE Transactions on Robotics (T-RO)
>
> **摘要:** Learning from Interactive Demonstrations has revolutionized the way non-expert humans teach robots. It is enough to kinesthetically move the robot around to teach pick-and-place, dressing, or cleaning policies. However, the main challenge is correctly generalizing to novel situations, e.g., different surfaces to clean or different arm postures to dress. This article proposes a novel task parameterization and generalization to transport the original robot policy, i.e., position, velocity, orientation, and stiffness. Unlike the state of the art, only a set of keypoints is tracked during the demonstration and the execution, e.g., a point cloud of the surface to clean. We then propose to fit a nonlinear transformation that would deform the space and then the original policy using the paired source and target point sets. The use of function approximators like Gaussian Processes allows us to generalize, or transport, the policy from every space location while estimating the uncertainty of the resulting policy due to the limited task keypoints and the reduced number of demonstrations. We compare the algorithm's performance with state-of-the-art task parameterization alternatives and analyze the effect of different function approximators. We also validated the algorithm on robot manipulation tasks, i.e., different posture arm dressing, different location product reshelving, and different shape surface cleaning.
>
---
#### [replaced 008] Controlling diverse robots by inferring Jacobian fields with deep networks
- **分类: cs.RO; cs.CV; cs.LG**

- **链接: [http://arxiv.org/pdf/2407.08722v2](http://arxiv.org/pdf/2407.08722v2)**

> **作者:** Sizhe Lester Li; Annan Zhang; Boyuan Chen; Hanna Matusik; Chao Liu; Daniela Rus; Vincent Sitzmann
>
> **备注:** Project Page: https://sizhe-li.github.io/publication/neural_jacobian_field
>
> **摘要:** Mirroring the complex structures and diverse functions of natural organisms is a long-standing challenge in robotics. Modern fabrication techniques have greatly expanded the feasible hardware, but using these systems requires control software to translate the desired motions into actuator commands. Conventional robots can easily be modeled as rigid links connected by joints, but it remains an open challenge to model and control biologically inspired robots that are often soft or made of several materials, lack sensing capabilities, and may change their material properties with use. Here, we introduce a method that uses deep neural networks to map a video stream of a robot to its visuomotor Jacobian field (the sensitivity of all 3D points to the robot's actuators). Our method enables the control of robots from only a single camera, makes no assumptions about the robots' materials, actuation, or sensing, and is trained without expert intervention by observing the execution of random commands. We demonstrate our method on a diverse set of robot manipulators that vary in actuation, materials, fabrication, and cost. Our approach achieves accurate closed-loop control and recovers the causal dynamic structure of each robot. Because it enables robot control using a generic camera as the only sensor, we anticipate that our work will broaden the design space of robotic systems and serve as a starting point for lowering the barrier to robotic automation.
>
---
#### [replaced 009] MaxInfoRL: Boosting exploration in reinforcement learning through information gain maximization
- **分类: cs.LG; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2412.12098v2](http://arxiv.org/pdf/2412.12098v2)**

> **作者:** Bhavya Sukhija; Stelian Coros; Andreas Krause; Pieter Abbeel; Carmelo Sferrazza
>
> **摘要:** Reinforcement learning (RL) algorithms aim to balance exploiting the current best strategy with exploring new options that could lead to higher rewards. Most common RL algorithms use undirected exploration, i.e., select random sequences of actions. Exploration can also be directed using intrinsic rewards, such as curiosity or model epistemic uncertainty. However, effectively balancing task and intrinsic rewards is challenging and often task-dependent. In this work, we introduce a framework, MaxInfoRL, for balancing intrinsic and extrinsic exploration. MaxInfoRL steers exploration towards informative transitions, by maximizing intrinsic rewards such as the information gain about the underlying task. When combined with Boltzmann exploration, this approach naturally trades off maximization of the value function with that of the entropy over states, rewards, and actions. We show that our approach achieves sublinear regret in the simplified setting of multi-armed bandits. We then apply this general formulation to a variety of off-policy model-free RL methods for continuous state-action spaces, yielding novel algorithms that achieve superior performance across hard exploration problems and complex scenarios such as visual control tasks.
>
---
#### [replaced 010] Optimizing Start Locations in Ergodic Search for Disaster Response
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.02708v3](http://arxiv.org/pdf/2507.02708v3)**

> **作者:** Ananya Rao; Alyssa Hargis; David Wettergreen; Howie Choset
>
> **摘要:** In disaster response scenarios, deploying robotic teams effectively is crucial for improving situational awareness and enhancing search and rescue operations. The use of robots in search and rescue has been studied but the question of where to start robot deployments has not been addressed. This work addresses the problem of optimally selecting starting locations for robots with heterogeneous capabilities by formulating a joint optimization problem. To determine start locations, this work adds a constraint to the ergodic optimization framework whose minimum assigns robots to start locations. This becomes a little more challenging when the robots are heterogeneous (equipped with different sensing and motion modalities) because not all robots start at the same location, and a more complex adaptation of the aforementioned constraint is applied. Our method assumes access to potential starting locations, which can be obtained from expert knowledge or aerial imagery. We experimentally evaluate the efficacy of our joint optimization approach by comparing it to baseline methods that use fixed starting locations for all robots. Our experimental results show significant gains in coverage performance, with average improvements of 35.98% on synthetic data and 31.91% on real-world data for homogeneous and heterogeneous teams, in terms of the ergodic metric.
>
---
#### [replaced 011] KineDepth: Utilizing Robot Kinematics for Online Metric Depth Estimation
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2409.19490v2](http://arxiv.org/pdf/2409.19490v2)**

> **作者:** Soofiyan Atar; Yuheng Zhi; Florian Richter; Michael Yip
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Depth perception is essential for a robot's spatial and geometric understanding of its environment, with many tasks traditionally relying on hardware-based depth sensors like RGB-D or stereo cameras. However, these sensors face practical limitations, including issues with transparent and reflective objects, high costs, calibration complexity, spatial and energy constraints, and increased failure rates in compound systems. While monocular depth estimation methods offer a cost-effective and simpler alternative, their adoption in robotics is limited due to their output of relative rather than metric depth, which is crucial for robotics applications. In this paper, we propose a method that utilizes a single calibrated camera, enabling the robot to act as a "measuring stick" to convert relative depth estimates into metric depth in real-time as tasks are performed. Our approach employs an LSTM-based metric depth regressor, trained online and refined through probabilistic filtering, to accurately restore the metric depth across the monocular depth map, particularly in areas proximal to the robot's motion. Experiments with real robots demonstrate that our method significantly outperforms current state-of-the-art monocular metric depth estimation techniques, achieving a 22.1% reduction in depth error and a 52% increase in success rate for a downstream task.
>
---
#### [replaced 012] ActSafe: Active Exploration with Safety Constraints for Reinforcement Learning
- **分类: cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2410.09486v3](http://arxiv.org/pdf/2410.09486v3)**

> **作者:** Yarden As; Bhavya Sukhija; Lenart Treven; Carmelo Sferrazza; Stelian Coros; Andreas Krause
>
> **摘要:** Reinforcement learning (RL) is ubiquitous in the development of modern AI systems. However, state-of-the-art RL agents require extensive, and potentially unsafe, interactions with their environments to learn effectively. These limitations confine RL agents to simulated environments, hindering their ability to learn directly in real-world settings. In this work, we present ActSafe, a novel model-based RL algorithm for safe and efficient exploration. ActSafe learns a well-calibrated probabilistic model of the system and plans optimistically w.r.t. the epistemic uncertainty about the unknown dynamics, while enforcing pessimism w.r.t. the safety constraints. Under regularity assumptions on the constraints and dynamics, we show that ActSafe guarantees safety during learning while also obtaining a near-optimal policy in finite time. In addition, we propose a practical variant of ActSafe that builds on latest model-based RL advancements and enables safe exploration even in high-dimensional settings such as visual control. We empirically show that ActSafe obtains state-of-the-art performance in difficult exploration tasks on standard safe deep RL benchmarks while ensuring safety during learning.
>
---
#### [replaced 013] Decentralized Uncertainty-Aware Multi-Agent Collision Avoidance with Model Predictive Path Integral
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.20293v2](http://arxiv.org/pdf/2507.20293v2)**

> **作者:** Stepan Dergachev; Konstantin Yakovlev
>
> **备注:** This is a pre-print of the paper accepted to IROS2025. The manuscript includes 8 pages, 4 figures, and 1 table. A supplementary video is available at https://youtu.be/_D4zDYJ4KCk Updated version: added link to source code in the abstract; updated experimental results description in Section VI.A; updated author affiliation and funding information; minor typo corrections
>
> **摘要:** Decentralized multi-agent navigation under uncertainty is a complex task that arises in numerous robotic applications. It requires collision avoidance strategies that account for both kinematic constraints, sensing and action execution noise. In this paper, we propose a novel approach that integrates the Model Predictive Path Integral (MPPI) with a probabilistic adaptation of Optimal Reciprocal Collision Avoidance. Our method ensures safe and efficient multi-agent navigation by incorporating probabilistic safety constraints directly into the MPPI sampling process via a Second-Order Cone Programming formulation. This approach enables agents to operate independently using local noisy observations while maintaining safety guarantees. We validate our algorithm through extensive simulations with differential-drive robots and benchmark it against state-of-the-art methods, including ORCA-DD and B-UAVC. Results demonstrate that our approach outperforms them while achieving high success rates, even in densely populated environments. Additionally, validation in the Gazebo simulator confirms its practical applicability to robotic platforms. A source code is available at http://github.com/PathPlanning/MPPI-Collision-Avoidance.
>
---
#### [replaced 014] Controllable Traffic Simulation through LLM-Guided Hierarchical Reasoning and Refinement
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.15135v2](http://arxiv.org/pdf/2409.15135v2)**

> **作者:** Zhiyuan Liu; Leheng Li; Yuning Wang; Haotian Lin; Hao Cheng; Zhizhe Liu; Lei He; Jianqiang Wang
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** Evaluating autonomous driving systems in complex and diverse traffic scenarios through controllable simulation is essential to ensure their safety and reliability. However, existing traffic simulation methods face challenges in their controllability. To address this, we propose a novel diffusion-based and LLM-enhanced traffic simulation framework. Our approach incorporates a high-level understanding module and a low-level refinement module, which systematically examines the hierarchical structure of traffic elements, guides LLMs to thoroughly analyze traffic scenario descriptions step by step, and refines the generation by self-reflection, enhancing their understanding of complex situations. Furthermore, we propose a Frenet-frame-based cost function framework that provides LLMs with geometrically meaningful quantities, improving their grasp of spatial relationships in a scenario and enabling more accurate cost function generation. Experiments on the Waymo Open Motion Dataset (WOMD) demonstrate that our method can handle more intricate descriptions and generate a broader range of scenarios in a controllable manner.
>
---
#### [replaced 015] KGN-Pro: Keypoint-Based Grasp Prediction through Probabilistic 2D-3D Correspondence Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.14820v2](http://arxiv.org/pdf/2507.14820v2)**

> **作者:** Bingran Chen; Baorun Li; Jian Yang; Yong Liu; Guangyao Zhai
>
> **摘要:** High-level robotic manipulation tasks demand flexible 6-DoF grasp estimation to serve as a basic function. Previous approaches either directly generate grasps from point-cloud data, suffering from challenges with small objects and sensor noise, or infer 3D information from RGB images, which introduces expensive annotation requirements and discretization issues. Recent methods mitigate some challenges by retaining a 2D representation to estimate grasp keypoints and applying Perspective-n-Point (PnP) algorithms to compute 6-DoF poses. However, these methods are limited by their non-differentiable nature and reliance solely on 2D supervision, which hinders the full exploitation of rich 3D information. In this work, we present KGN-Pro, a novel grasping network that preserves the efficiency and fine-grained object grasping of previous KGNs while integrating direct 3D optimization through probabilistic PnP layers. KGN-Pro encodes paired RGB-D images to generate Keypoint Map, and further outputs a 2D confidence map to weight keypoint contributions during re-projection error minimization. By modeling the weighted sum of squared re-projection errors probabilistically, the network effectively transmits 3D supervision to its 2D keypoint predictions, enabling end-to-end learning. Experiments on both simulated and real-world platforms demonstrate that KGN-Pro outperforms existing methods in terms of grasp cover rate and success rate.
>
---
#### [replaced 016] UniLGL: Learning Uniform Place Recognition for FOV-limited/Panoramic LiDAR Global Localization
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.12194v2](http://arxiv.org/pdf/2507.12194v2)**

> **作者:** Hongming Shen; Xun Chen; Yulin Hui; Zhenyu Wu; Wei Wang; Qiyang Lyu; Tianchen Deng; Danwei Wang
>
> **摘要:** Existing LGL methods typically consider only partial information (e.g., geometric features) from LiDAR observations or are designed for homogeneous LiDAR sensors, overlooking the uniformity in LGL. In this work, a uniform LGL method is proposed, termed UniLGL, which simultaneously achieves spatial and material uniformity, as well as sensor-type uniformity. The key idea of the proposed method is to encode the complete point cloud, which contains both geometric and material information, into a pair of BEV images (i.e., a spatial BEV image and an intensity BEV image). An end-to-end multi-BEV fusion network is designed to extract uniform features, equipping UniLGL with spatial and material uniformity. To ensure robust LGL across heterogeneous LiDAR sensors, a viewpoint invariance hypothesis is introduced, which replaces the conventional translation equivariance assumption commonly used in existing LPR networks and supervises UniLGL to achieve sensor-type uniformity in both global descriptors and local feature representations. Finally, based on the mapping between local features on the 2D BEV image and the point cloud, a robust global pose estimator is derived that determines the global minimum of the global pose on SE(3) without requiring additional registration. To validate the effectiveness of the proposed uniform LGL, extensive benchmarks are conducted in real-world environments, and the results show that the proposed UniLGL is demonstratively competitive compared to other State-of-the-Art LGL methods. Furthermore, UniLGL has been deployed on diverse platforms, including full-size trucks and agile Micro Aerial Vehicles (MAVs), to enable high-precision localization and mapping as well as multi-MAV collaborative exploration in port and forest environments, demonstrating the applicability of UniLGL in industrial and field scenarios.
>
---
#### [replaced 017] LoL-NMPC: Low-Level Dynamics Integration in Nonlinear Model Predictive Control for Unmanned Aerial Vehicles
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.02169v2](http://arxiv.org/pdf/2506.02169v2)**

> **作者:** Parakh M. Gupta; Ondřej Procházka; Jan Hřebec; Matej Novosad; Robert Pěnička; Martin Saska
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** [Accepted to IROS 2025] In this paper, we address the problem of tracking high-speed agile trajectories for Unmanned Aerial Vehicles(UAVs), where model inaccuracies can lead to large tracking errors. Existing Nonlinear Model Predictive Controller(NMPC) methods typically neglect the dynamics of the low-level flight controllers such as underlying PID controller present in many flight stacks, and this results in sub-optimal tracking performance at high speeds and accelerations. To this end, we propose a novel NMPC formulation, LoL-NMPC, which explicitly incorporates low-level controller dynamics and motor dynamics in order to minimize trajectory tracking errors while maintaining computational efficiency. By leveraging linear constraints inside low-level dynamics, our approach inherently accounts for actuator constraints without requiring additional reallocation strategies. The proposed method is validated in both simulation and real-world experiments, demonstrating improved tracking accuracy and robustness at speeds up to 98.57 km/h and accelerations of 3.5 g. Our results show an average 21.97 % reduction in trajectory tracking error over standard NMPC formulation, with LoL-NMPC maintaining real-time feasibility at 100 Hz on an embedded ARM-based flight computer.
>
---
#### [replaced 018] UniLegs: Universal Multi-Legged Robot Control through Morphology-Agnostic Policy Distillation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.22653v2](http://arxiv.org/pdf/2507.22653v2)**

> **作者:** Weijie Xi; Zhanxiang Cao; Chenlin Ming; Jianying Zheng; Guyue Zhou
>
> **备注:** 6 pages, 3 figures, IROS 2025
>
> **摘要:** Developing controllers that generalize across diverse robot morphologies remains a significant challenge in legged locomotion. Traditional approaches either create specialized controllers for each morphology or compromise performance for generality. This paper introduces a two-stage teacher-student framework that bridges this gap through policy distillation. First, we train specialized teacher policies optimized for individual morphologies, capturing the unique optimal control strategies for each robot design. Then, we distill this specialized expertise into a single Transformer-based student policy capable of controlling robots with varying leg configurations. Our experiments across five distinct legged morphologies demonstrate that our approach preserves morphology-specific optimal behaviors, with the Transformer architecture achieving 94.47% of teacher performance on training morphologies and 72.64% on unseen robot designs. Comparative analysis reveals that Transformer-based architectures consistently outperform MLP baselines by leveraging attention mechanisms to effectively model joint relationships across different kinematic structures. We validate our approach through successful deployment on a physical quadruped robot, demonstrating the practical viability of our morphology-agnostic control framework. This work presents a scalable solution for developing universal legged robot controllers that maintain near-optimal performance while generalizing across diverse morphologies.
>
---
#### [replaced 019] Learning Object Compliance via Young's Modulus from Single Grasps using Camera-Based Tactile Sensors
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2406.15304v4](http://arxiv.org/pdf/2406.15304v4)**

> **作者:** Michael Burgess; Jialiang Zhao; Laurence Willemet
>
> **摘要:** Compliance is a useful parametrization of tactile information that humans often utilize in manipulation tasks. It can be used to inform low-level contact-rich actions or characterize objects at a high-level. In robotic manipulation, existing approaches to estimate compliance have struggled to generalize across both object shape and material. Using camera-based tactile sensors, proprioception, and force measurements, we present a novel approach to estimate object compliance as Young's modulus (E) from parallel grasps. We evaluate our method over a novel dataset of 285 common objects, including a wide array of shapes and materials with Young's moduli ranging from 5.0 kPa to 250 GPa. Combining analytical and data-driven approaches, we develop a hybrid system using a multi-tower neural network to analyze a sequence of tactile images from grasping. This system is shown to estimate the Young's modulus of unseen objects within an order of magnitude at 74.2% accuracy across our dataset. This is an improvement over purely analytical and data-driven baselines which exhibit 28.9% and 65.0% accuracy respectively. Importantly, this estimation system performs irrespective of object geometry and demonstrates increased robustness across material types. Code is available on GitHub and collected data is available on HuggingFace.
>
---
#### [replaced 020] AKF-LIO: LiDAR-Inertial Odometry with Gaussian Map by Adaptive Kalman Filter
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.06891v2](http://arxiv.org/pdf/2503.06891v2)**

> **作者:** Xupeng Xie; Ruoyu Geng; Jun Ma; Boyu Zhou
>
> **备注:** Submitted to IROS 2025 Conference, https://github.com/xpxie/AKF-LIO.git
>
> **摘要:** Existing LiDAR-Inertial Odometry (LIO) systems typically use sensor-specific or environment-dependent measurement covariances during state estimation, leading to laborious parameter tuning and suboptimal performance in challenging conditions (e.g., sensor degeneracy and noisy observations). Therefore, we propose an Adaptive Kalman Filter (AKF) framework that dynamically estimates time-varying noise covariances of LiDAR and Inertial Measurement Unit (IMU) measurements, enabling context-aware confidence weighting between sensors. During LiDAR degeneracy, the system prioritizes IMU data while suppressing contributions from unreliable inputs like moving objects or noisy point clouds. Furthermore, a compact Gaussian-based map representation is introduced to model environmental planarity and spatial noise. A correlated registration strategy ensures accurate plane normal estimation via pseudo-merge, even in unstructured environments like forests. Extensive experiments validate the robustness of the proposed system across diverse environments, including dynamic scenes and geometrically degraded scenarios. Our method achieves reliable localization results across all MARS-LVIG sequences and ranks 8th on the KITTI Odometry Benchmark. The code will be released at https://github.com/xpxie/AKF-LIO.git.
>
---
#### [replaced 021] Model-Free and Real-Time Unicycle-Based Source Seeking with Differential Wheeled Robotic Experiments
- **分类: cs.RO; math.OC**

- **链接: [http://arxiv.org/pdf/2501.02184v4](http://arxiv.org/pdf/2501.02184v4)**

> **作者:** Ahmed A. Elgohary; Sameh A. Eisa; Shivam Bajpai
>
> **摘要:** Many autonomous robots aimed at source-seeking are studied, and their controls designed, using unicycle modeling and formulation. This is true not only for model-based controllers, but also for model-free, real-time control methods such as extremum seeking control (ESC). In this paper, we propose a unicycle-based ESC design applicable to differential wheeled robots that: (1) is very simple design, based on one simple control-affine law, and without state integrators; (2) attenuates oscillations known to persist in ESC designs (i.e., fully stop at the source); and (3) operates in a model-free, real-time setting, tolerating environmental/sensor noise. We provide simulation and real-world robotic experimental results for fixed and moving light source seeking by a differential wheeled robot using our proposed design. Results indicate clear advantages of our proposed design when compared to the literature, including attenuation of undesired oscillations, improved convergence speed, and better handling of noise.
>
---
#### [replaced 022] Generalizable Image Repair for Robust Visual Control
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.05911v2](http://arxiv.org/pdf/2503.05911v2)**

> **作者:** Carson Sobolewski; Zhenjiang Mao; Kshitij Maruti Vejre; Ivan Ruchkin
>
> **备注:** 8 pages, 4 figures, 2 tables, 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** Vision-based control relies on accurate perception to achieve robustness. However, image distribution changes caused by sensor noise, adverse weather, and dynamic lighting can degrade perception, leading to suboptimal control decisions. Existing approaches, including domain adaptation and adversarial training, improve robustness but struggle to generalize to unseen corruptions while introducing computational overhead. To address this challenge, we propose a real-time image repair module that restores corrupted images before they are used by the controller. Our method leverages generative adversarial models, specifically CycleGAN and pix2pix, for image repair. CycleGAN enables unpaired image-to-image translation to adapt to novel corruptions, while pix2pix exploits paired image data when available to improve the quality. To ensure alignment with control performance, we introduce a control-focused loss function that prioritizes perceptual consistency in repaired images. We evaluated our method in a simulated autonomous racing environment with various visual corruptions. The results show that our approach significantly improves performance compared to baselines, mitigating distribution shift and enhancing controller reliability.
>
---
#### [replaced 023] SmartPNT-MSF: A Multi-Sensor Fusion Dataset for Positioning and Navigation Research
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2507.19079v2](http://arxiv.org/pdf/2507.19079v2)**

> **作者:** Feng Zhu; Zihang Zhang; Kangcheng Teng; Abduhelil Yakup; Xiaohong Zhang
>
> **摘要:** High-precision navigation and positioning systems are critical for applications in autonomous vehicles and mobile mapping, where robust and continuous localization is essential. To test and enhance the performance of algorithms, some research institutions and companies have successively constructed and publicly released datasets. However, existing datasets still suffer from limitations in sensor diversity and environmental coverage. To address these shortcomings and advance development in related fields, the SmartPNT Multisource Integrated Navigation, Positioning, and Attitude Dataset has been developed. This dataset integrates data from multiple sensors, including Global Navigation Satellite Systems (GNSS), Inertial Measurement Units (IMU), optical cameras, and LiDAR, to provide a rich and versatile resource for research in multi-sensor fusion and high-precision navigation. The dataset construction process is thoroughly documented, encompassing sensor configurations, coordinate system definitions, and calibration procedures for both cameras and LiDAR. A standardized framework for data collection and processing ensures consistency and scalability, enabling large-scale analysis. Validation using state-of-the-art Simultaneous Localization and Mapping (SLAM) algorithms, such as VINS-Mono and LIO-SAM, demonstrates the dataset's applicability for advanced navigation research. Covering a wide range of real-world scenarios, including urban areas, campuses, tunnels, and suburban environments, the dataset offers a valuable tool for advancing navigation technologies and addressing challenges in complex environments. By providing a publicly accessible, high-quality dataset, this work aims to bridge gaps in sensor diversity, data accessibility, and environmental representation, fostering further innovation in the field.
>
---
#### [replaced 024] Diffusion Beats Autoregressive in Data-Constrained Settings
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.15857v4](http://arxiv.org/pdf/2507.15857v4)**

> **作者:** Mihir Prabhudesai; Mengning Wu; Amir Zadeh; Katerina Fragkiadaki; Deepak Pathak
>
> **备注:** Project Webpage: https://diffusion-scaling.github.io
>
> **摘要:** Autoregressive (AR) models have long dominated the landscape of large language models, driving progress across a wide range of tasks. Recently, diffusion-based language models have emerged as a promising alternative, though their advantages over AR models remain underexplored. In this paper, we systematically study masked diffusion models in data-constrained settings-where training involves repeated passes over limited data-and find that they significantly outperform AR models when compute is abundant but data is scarce. Diffusion models make better use of repeated data, achieving lower validation loss and superior downstream performance. We interpret this advantage as implicit data augmentation: masked diffusion exposes the model to a diverse distribution of token orderings and prediction tasks, unlike AR's fixed left-to-right factorization. We find new scaling laws for diffusion models and derive a closed-form expression for the critical compute threshold at which diffusion begins to outperform AR. These results suggest that when data, not compute, is the bottleneck, diffusion models offer a compelling alternative to the standard AR paradigm. Our code is available at: https://diffusion-scaling.github.io.
>
---
#### [replaced 025] SDHN: Skewness-Driven Hypergraph Networks for Enhanced Localized Multi-Robot Coordination
- **分类: cs.RO; cs.MA**

- **链接: [http://arxiv.org/pdf/2504.06684v2](http://arxiv.org/pdf/2504.06684v2)**

> **作者:** Delin Zhao; Yanbo Shan; Chang Liu; Shenghang Lin; Yingxin Shou; Bin Xu
>
> **摘要:** Multi-Agent Reinforcement Learning is widely used for multi-robot coordination, where simple graphs typically model pairwise interactions. However, such representations fail to capture higher-order collaborations, limiting effectiveness in complex tasks. While hypergraph-based approaches enhance cooperation, existing methods often generate arbitrary hypergraph structures and lack adaptability to environmental uncertainties. To address these challenges, we propose the Skewness-Driven Hypergraph Network (SDHN), which employs stochastic Bernoulli hyperedges to explicitly model higher-order multi-robot interactions. By introducing a skewness loss, SDHN promotes an efficient structure with Small-Hyperedge Dominant Hypergraph, allowing robots to prioritize localized synchronization while still adhering to the overall information, similar to human coordination. Extensive experiments on Moving Agents in Formation and Robotic Warehouse tasks validate SDHN's effectiveness, demonstrating superior performance over state-of-the-art baselines.
>
---
#### [replaced 026] Exploiting Local Observations for Robust Robot Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2309.14792v4](http://arxiv.org/pdf/2309.14792v4)**

> **作者:** Wenshuai Zhao; Eetu-Aleksi Rantala; Sahar Salimpour; Zhiyuan Li; Joni Pajarinen; Jorge Peña Queralta
>
> **备注:** 8 pages, 8 figures
>
> **摘要:** While many robotic tasks can be addressed using either centralized single-agent control with full state observation or decentralized multi-agent control, clear criteria for choosing between these approaches remain underexplored. This paper systematically investigates how multi-agent reinforcement learning (MARL) with local observations can improve robustness in complex robotic systems compared to traditional centralized control. Through theoretical analysis and empirical validation, we show that in certain tasks, decentralized MARL can achieve performance comparable to centralized methods while exhibiting greater resilience to perturbations and agent failures. By analytically demonstrating the equivalence of single-agent reinforcement learning (SARL) and MARL under full observability, we identify observability as the critical factor distinguishing the two paradigms. We further derive bounds quantifying performance degradation under external perturbations for locally observable policies. Empirical results on standard MARL benchmarks confirm that MARL with limited observations can maintain competitive performance. Finally, real-world experiments with a mobile manipulator demonstrate that decentralized MARL controllers achieve markedly improved robustness to agent malfunctions and environmental disturbances relative to centralized baselines. Together, these findings highlight MARL with local observations as a robust and practical alternative to conventional centralized control in complex robotic systems.
>
---
#### [replaced 027] Allocation for Omnidirectional Aerial Robots: Incorporating Power Dynamics
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2412.16107v2](http://arxiv.org/pdf/2412.16107v2)**

> **作者:** Eugenio Cuniato; Mike Allenspach; Thomas Stastny; Helen Oleynikova; Roland Siegwart; Michael Pantic
>
> **摘要:** Tilt-rotor aerial robots are more dynamic and versatile than fixed-rotor platforms, since the thrust vector and body orientation are decoupled. However, the coordination of servos and propellers (the allocation problem) is not trivial, especially accounting for overactuation and actuator dynamics. We incrementally build and present three novel allocation methods for tiltrotor aerial robots, comparing them to state-of-the-art methods on a real system performing dynamic maneuvers. We extend the state-of-the-art geometric allocation into a differential allocation, which uses the platform's redundancy and does not suffer from singularities. We expand it by incorporating actuator dynamics and propeller power dynamics. These allow us to model dynamic propeller acceleration limits, bringing two main advantages: balancing propeller speed without the need of nullspace goals and allowing the platform to selectively turn-off propellers during flight, opening the door to new manipulation possibilities. We also use actuator dynamics and limits to normalize the allocation problem, making it easier to tune and allowing it to track 70% faster trajectories than a geometric allocation.
>
---
#### [replaced 028] CoA-VLA: Improving Vision-Language-Action Models via Visual-Textual Chain-of-Affordance
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2412.20451v2](http://arxiv.org/pdf/2412.20451v2)**

> **作者:** Jinming Li; Yichen Zhu; Zhibin Tang; Junjie Wen; Minjie Zhu; Xiaoyu Liu; Chengmeng Li; Ran Cheng; Yaxin Peng; Yan Peng; Feifei Feng
>
> **备注:** Project webpage is available at https://chain-of-affordance.github.io
>
> **摘要:** Robot foundation models, particularly Vision-Language-Action (VLA) models, have garnered significant attention for their ability to enhance robot policy learning, greatly improving robot's generalization and robustness. OpenAI's recent model, O1, showcased impressive capabilities in solving complex problems by utilizing extensive reasoning chains. This prompts an important question: can robot models achieve better performance in multi-task , complex environments by reviewing prior observations and then providing task-specific reasoning to guide action prediction? In this paper, we introduce Chain-of-Affordance (CoA-VLA) , a novel approach to scaling robot models by incorporating reasoning in the format of sequential robot affordances to facilitate task completion. Specifically, we prompt the model to consider the following four types of affordances before taking action: (1) object affordance - what object to manipulate and where it is ; (2) grasp affordance - the specific object part to grasp ; (3) spatial affordance - the optimal space to place the object ; and (4) movement affordance-the collision - free path for movement. We further transform each affordance into two prompting formats: visual affordance and textual affordance. We introduce a novel vision-language co-injection module that integrates this knowledge into the policy network. This allows the robot to leverage essential contextual information during action inference, resulting in improved precision and robustness. Our experiments demonstrate that CoA-VLA outperforms state-of-the-art robot foundation models, including OpenVLA and Octo, on a variety of tasks. Furthermore, CoA-VLA exhibits strong generalization capabilities, including recognizing unseen object poses, identifying free space, and avoiding obstacles in novel environments.
>
---
#### [replaced 029] Line-Search Filter Differential Dynamic Programming for Optimal Control with Nonlinear Equality Constraints
- **分类: math.OC; cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2504.08278v4](http://arxiv.org/pdf/2504.08278v4)**

> **作者:** Ming Xu; Stephen Gould; Iman Shames
>
> **摘要:** We present FilterDDP, a differential dynamic programming algorithm for solving discrete-time, optimal control problems (OCPs) with nonlinear equality constraints. Unlike prior methods based on merit functions or the augmented Lagrangian class of algorithms, FilterDDP uses a step filter in conjunction with a line search to handle equality constraints. We identify two important design choices for the step filter criteria which lead to robust numerical performance: 1) we use the Lagrangian instead of the cost as one of the filter criterion and, 2) for the stopping criteria and backward pass Hessians, we replace the value function gradient with an estimated dual variable of the dynamics constraints. Both choices are rigorously justified, for 2) in particular by a formal proof of local quadratic convergence. We validate FilterDDP on three contact implicit trajectory optimisation problems which arise in robotics.
>
---
#### [replaced 030] iFANnpp: Nuclear Power Plant Digital Twin for Robots and Autonomous Intelligence
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.09213v3](http://arxiv.org/pdf/2410.09213v3)**

> **作者:** Youndo Do; Marc Zebrowitz; Jackson Stahl; Fan Zhang
>
> **摘要:** Robotics has gained attention in the nuclear industry due to its precision and ability to automate tasks. However, there is a critical need for advanced simulation and control methods to predict robot behavior and optimize plant performance, motivating the use of digital twins. Most existing digital twins do not offer a total design of a nuclear power plant. Moreover, they are designed for specific algorithms or tasks, making them unsuitable for broader research applications. In response, this work proposes a comprehensive nuclear power plant digital twin designed to improve real-time monitoring, operational efficiency, and predictive maintenance. A full nuclear power plant is modeled in Unreal Engine 5 and integrated with a high-fidelity Generic Pressurized Water Reactor Simulator to create a realistic model of a nuclear power plant and a real-time updated virtual environment. The virtual environment provides various features for researchers to easily test custom robot algorithms and frameworks.
>
---
#### [replaced 031] EP-Diffuser: An Efficient Diffusion Model for Traffic Scene Generation and Prediction via Polynomial Representations
- **分类: cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.05422v3](http://arxiv.org/pdf/2504.05422v3)**

> **作者:** Yue Yao; Mohamed-Khalil Bouzidi; Daniel Goehring; Joerg Reichardt
>
> **摘要:** As the prediction horizon increases, predicting the future evolution of traffic scenes becomes increasingly difficult due to the multi-modal nature of agent motion. Most state-of-the-art (SotA) prediction models primarily focus on forecasting the most likely future. However, for the safe operation of autonomous vehicles, it is equally important to cover the distribution for plausible motion alternatives. To address this, we introduce EP-Diffuser, a novel parameter-efficient diffusion-based generative model designed to capture the distribution of possible traffic scene evolutions. Conditioned on road layout and agent history, our model acts as a predictor and generates diverse, plausible scene continuations. We benchmark EP-Diffuser against two SotA models in terms of accuracy and plausibility of predictions on the Argoverse 2 dataset. Despite its significantly smaller model size, our approach achieves both highly accurate and plausible traffic scene predictions. We further evaluate model generalization ability in an out-of-distribution (OoD) test setting using Waymo Open dataset and show superior robustness of our approach. The code and model checkpoints are available at: https://github.com/continental/EP-Diffuser.
>
---
