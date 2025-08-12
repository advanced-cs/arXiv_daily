# 机器人 cs.RO

- **最新发布 70 篇**

- **更新 42 篇**

## 最新发布

#### [new 001] Verti-Arena: A Controllable and Standardized Indoor Testbed for Multi-Terrain Off-Road Autonomy
- **分类: cs.RO**

- **简介: 论文介绍了一种可控、标准化的室内测试平台Verti-Arena，用于多地形越野自主导航研究，解决传统测试环境不足的问题，通过传感器与运动捕捉提供精准数据并支持远程实验。**

- **链接: [http://arxiv.org/pdf/2508.08226v1](http://arxiv.org/pdf/2508.08226v1)**

> **作者:** Haiyue Chen; Aniket Datar; Tong Xu; Francesco Cancelliere; Harsh Rangwala; Madhan Balaji Rao; Daeun Song; David Eichinger; Xuesu Xiao
>
> **备注:** 6 pages
>
> **摘要:** Off-road navigation is an important capability for mobile robots deployed in environments that are inaccessible or dangerous to humans, such as disaster response or planetary exploration. Progress is limited due to the lack of a controllable and standardized real-world testbed for systematic data collection and validation. To fill this gap, we introduce Verti-Arena, a reconfigurable indoor facility designed specifically for off-road autonomy. By providing a repeatable benchmark environment, Verti-Arena supports reproducible experiments across a variety of vertically challenging terrains and provides precise ground truth measurements through onboard sensors and a motion capture system. Verti-Arena also supports consistent data collection and comparative evaluation of algorithms in off-road autonomy research. We also develop a web-based interface that enables research groups worldwide to remotely conduct standardized off-road autonomy experiments on Verti-Arena.
>
---
#### [new 002] EGS-SLAM: RGB-D Gaussian Splatting SLAM with Events
- **分类: cs.RO**

- **简介: 论文提出EGS-SLAM框架，通过融合事件与RGB-D数据缓解运动模糊，提升GS-SLAM在实时场景中的跟踪精度及高保真3D重建。核心创新包括：1）建模相机连续轨迹支持事件-模糊感知；2）引入可学习相机响应函数与无事件损失抑制畸变；3）在合成与真实场景中验证有效性。**

- **链接: [http://arxiv.org/pdf/2508.07003v1](http://arxiv.org/pdf/2508.07003v1)**

> **作者:** Siyu Chen; Shenghai Yuan; Thien-Minh Nguyen; Zhuyu Huang; Chenyang Shi; Jin Jing; Lihua Xie
>
> **备注:** Accepted by IEEE RAL
>
> **摘要:** Gaussian Splatting SLAM (GS-SLAM) offers a notable improvement over traditional SLAM methods, enabling photorealistic 3D reconstruction that conventional approaches often struggle to achieve. However, existing GS-SLAM systems perform poorly under persistent and severe motion blur commonly encountered in real-world scenarios, leading to significantly degraded tracking accuracy and compromised 3D reconstruction quality. To address this limitation, we propose EGS-SLAM, a novel GS-SLAM framework that fuses event data with RGB-D inputs to simultaneously reduce motion blur in images and compensate for the sparse and discrete nature of event streams, enabling robust tracking and high-fidelity 3D Gaussian Splatting reconstruction. Specifically, our system explicitly models the camera's continuous trajectory during exposure, supporting event- and blur-aware tracking and mapping on a unified 3D Gaussian Splatting scene. Furthermore, we introduce a learnable camera response function to align the dynamic ranges of events and images, along with a no-event loss to suppress ringing artifacts during reconstruction. We validate our approach on a new dataset comprising synthetic and real-world sequences with significant motion blur. Extensive experimental results demonstrate that EGS-SLAM consistently outperforms existing GS-SLAM systems in both trajectory accuracy and photorealistic 3D Gaussian Splatting reconstruction. The source code will be available at https://github.com/Chensiyu00/EGS-SLAM.
>
---
#### [new 003] A Learning-Based Framework for Collision-Free Motion Planning
- **分类: cs.RO**

- **简介: 论文提出一种基于学习的碰撞规避运动规划框架，通过深度神经网络从场景图像中学习最优规划参数，实现高效实时轨迹生成。该方法克服传统手动参数调整的局限，结合CUDA感知模块与贝叶斯优化，验证于仿真与机械臂实验，提升泛化能力。**

- **链接: [http://arxiv.org/pdf/2508.07502v1](http://arxiv.org/pdf/2508.07502v1)**

> **作者:** Mateus Salomão; Tianyü Ren; Alexander König
>
> **摘要:** This paper presents a learning-based extension to a Circular Field (CF)-based motion planner for efficient, collision-free trajectory generation in cluttered environments. The proposed approach overcomes the limitations of hand-tuned force field parameters by employing a deep neural network trained to infer optimal planner gains from a single depth image of the scene. The pipeline incorporates a CUDA-accelerated perception module, a predictive agent-based planning strategy, and a dataset generated through Bayesian optimization in simulation. The resulting framework enables real-time planning without manual parameter tuning and is validated both in simulation and on a Franka Emika Panda robot. Experimental results demonstrate successful task completion and improved generalization compared to classical planners.
>
---
#### [new 004] MonoMPC: Monocular Vision Based Navigation with Learned Collision Model and Risk-Aware Model Predictive Control
- **分类: cs.RO**

- **简介: 论文提出基于单目视觉的导航方法，通过学习碰撞模型与风险感知MPC联合训练，解决深度缺失导致的碰撞检测难题，提升复杂环境导航成功率。**

- **链接: [http://arxiv.org/pdf/2508.07387v1](http://arxiv.org/pdf/2508.07387v1)**

> **作者:** Basant Sharma; Prajyot Jadhav; Pranjal Paul; K. Madhava Krishna; Arun Kumar Singh
>
> **摘要:** Navigating unknown environments with a single RGB camera is challenging, as the lack of depth information prevents reliable collision-checking. While some methods use estimated depth to build collision maps, we found that depth estimates from vision foundation models are too noisy for zero-shot navigation in cluttered environments. We propose an alternative approach: instead of using noisy estimated depth for direct collision-checking, we use it as a rich context input to a learned collision model. This model predicts the distribution of minimum obstacle clearance that the robot can expect for a given control sequence. At inference, these predictions inform a risk-aware MPC planner that minimizes estimated collision risk. Our joint learning pipeline co-trains the collision model and risk metric using both safe and unsafe trajectories. Crucially, our joint-training ensures optimal variance in our collision model that improves navigation in highly cluttered environments. Consequently, real-world experiments show 9x and 7x improvements in success rates over NoMaD and the ROS stack, respectively. Ablation studies further validate the effectiveness of our design choices.
>
---
#### [new 005] LAURON VI: A Six-Legged Robot for Dynamic Walking
- **分类: cs.RO**

- **简介: 论文提出LAURON VI六足机器人，研究动态行走与自主性，通过三种控制方法及火星模拟测试，解决复杂地形下高效稳定行走问题。**

- **链接: [http://arxiv.org/pdf/2508.07689v1](http://arxiv.org/pdf/2508.07689v1)**

> **作者:** Christian Eichmann; Sabine Bellmann; Nicolas Hügel; Louis-Elias Enslin; Carsten Plasberg; Georg Heppner; Arne Roennau; Ruediger Dillmann
>
> **摘要:** Legged locomotion enables robotic systems to traverse extremely challenging terrains. In many real-world scenarios, the terrain is not that difficult and these mixed terrain types introduce the need for flexible use of different walking strategies to achieve mission goals in a fast, reliable, and energy-efficient way. Six-legged robots have a high degree of flexibility and inherent stability that aids them in traversing even some of the most difficult terrains, such as collapsed buildings. However, their lack of fast walking gaits for easier surfaces is one reason why they are not commonly applied in these scenarios. This work presents LAURON VI, a six-legged robot platform for research on dynamic walking gaits as well as on autonomy for complex field missions. The robot's 18 series elastic joint actuators offer high-frequency interfaces for Cartesian impedance and pure torque control. We have designed, implemented, and compared three control approaches: kinematic-based, model-predictive, and reinforcement-learned controllers. The robot hardware and the different control approaches were extensively tested in a lab environment as well as on a Mars analog mission. The introduction of fast locomotion strategies for LAURON VI makes six-legged robots vastly more suitable for a wide range of real-world applications.
>
---
#### [new 006] Automated Seam Folding and Sewing Machine on Pleated Pants for Apparel Manufacturing
- **分类: cs.RO**

- **简介: 论文设计自动化褶皱裤缝纫机，解决传统手工缝纫效率低、一致性差及成本高的问题，通过精密折叠与实时监控提升生产效率93%，降低劳动时间与浪费，推动服装制造业向高效可持续方向发展。**

- **链接: [http://arxiv.org/pdf/2508.06518v1](http://arxiv.org/pdf/2508.06518v1)**

> **作者:** Ray Wai Man Kong
>
> **备注:** 13 pages, 9 figures
>
> **摘要:** The applied research is the design and development of an automated folding and sewing machine for pleated pants. It represents a significant advancement in addressing the challenges associated with manual sewing processes. Traditional methods for creating pleats are labour-intensive, prone to inconsistencies, and require high levels of skill, making automation a critical need in the apparel industry. This research explores the technical feasibility and operational benefits of integrating advanced technologies into garment production, focusing on the creation of an automated machine capable of precise folding and sewing operations and eliminating the marking operation. The proposed machine incorporates key features such as a precision folding mechanism integrated into the automated sewing unit with real-time monitoring capabilities. The results demonstrate remarkable improvements: the standard labour time has been reduced by 93%, dropping from 117 seconds per piece to just 8 seconds with the automated system. Similarly, machinery time improved by 73%, and the total output rate increased by 72%. These enhancements translate into a cycle time reduction from 117 seconds per piece to an impressive 33 seconds, enabling manufacturers to meet customer demand more swiftly. By eliminating manual marking processes, the machine not only reduces labour costs but also minimizes waste through consistent pleat formation. This automation aligns with industry trends toward sustainability and efficiency, potentially reducing environmental impact by decreasing material waste and energy consumption.
>
---
#### [new 007] 3D Gaussian Representations with Motion Trajectory Field for Dynamic Scene Reconstruction
- **分类: cs.RO**

- **简介: 论文提出结合3D高斯表示与运动轨迹场的动态场景重建方法，解决传统技术在动态场景中运动轨迹建模不足的问题，通过解耦动态对象与静态背景，优化运动轨迹场以实现高效准确的动态场景重建。**

- **链接: [http://arxiv.org/pdf/2508.07182v1](http://arxiv.org/pdf/2508.07182v1)**

> **作者:** Xuesong Li; Lars Petersson; Vivien Rolland
>
> **摘要:** This paper addresses the challenge of novel-view synthesis and motion reconstruction of dynamic scenes from monocular video, which is critical for many robotic applications. Although Neural Radiance Fields (NeRF) and 3D Gaussian Splatting (3DGS) have demonstrated remarkable success in rendering static scenes, extending them to reconstruct dynamic scenes remains challenging. In this work, we introduce a novel approach that combines 3DGS with a motion trajectory field, enabling precise handling of complex object motions and achieving physically plausible motion trajectories. By decoupling dynamic objects from static background, our method compactly optimizes the motion trajectory field. The approach incorporates time-invariant motion coefficients and shared motion trajectory bases to capture intricate motion patterns while minimizing optimization complexity. Extensive experiments demonstrate that our approach achieves state-of-the-art results in both novel-view synthesis and motion trajectory recovery from monocular video, advancing the capabilities of dynamic scene reconstruction.
>
---
#### [new 008] Autonomous Navigation of Cloud-Controlled Quadcopters in Confined Spaces Using Multi-Modal Perception and LLM-Driven High Semantic Reasoning
- **分类: cs.RO; cs.AI; cs.CV; cs.SY; eess.SY**

- **简介: 论文提出一种基于云控四旋翼无人机在GPS禁用室内环境的自主导航方案，融合多模态感知与LLM驱动的高语义推理，解决复杂场景下的避障与导航难题，通过传感器融合与轻量化处理实现低延迟高精度导航。**

- **链接: [http://arxiv.org/pdf/2508.07885v1](http://arxiv.org/pdf/2508.07885v1)**

> **作者:** Shoaib Ahmmad; Zubayer Ahmed Aditto; Md Mehrab Hossain; Noushin Yeasmin; Shorower Hossain
>
> **摘要:** This paper introduces an advanced AI-driven perception system for autonomous quadcopter navigation in GPS-denied indoor environments. The proposed framework leverages cloud computing to offload computationally intensive tasks and incorporates a custom-designed printed circuit board (PCB) for efficient sensor data acquisition, enabling robust navigation in confined spaces. The system integrates YOLOv11 for object detection, Depth Anything V2 for monocular depth estimation, a PCB equipped with Time-of-Flight (ToF) sensors and an Inertial Measurement Unit (IMU), and a cloud-based Large Language Model (LLM) for context-aware decision-making. A virtual safety envelope, enforced by calibrated sensor offsets, ensures collision avoidance, while a multithreaded architecture achieves low-latency processing. Enhanced spatial awareness is facilitated by 3D bounding box estimation with Kalman filtering. Experimental results in an indoor testbed demonstrate strong performance, with object detection achieving a mean Average Precision (mAP50) of 0.6, depth estimation Mean Absolute Error (MAE) of 7.2 cm, only 16 safety envelope breaches across 42 trials over approximately 11 minutes, and end-to-end system latency below 1 second. This cloud-supported, high-intelligence framework serves as an auxiliary perception and navigation system, complementing state-of-the-art drone autonomy for GPS-denied confined spaces.
>
---
#### [new 009] A tutorial note on collecting simulated data for vision-language-action models
- **分类: cs.RO**

- **简介: 论文综述了三种模拟数据集构建方法（PyBullet、LIBERO、RT-X），解决视觉-语言-动作模型所需复杂数据关系问题，指导多机器人场景下的数据采集与标准化。**

- **链接: [http://arxiv.org/pdf/2508.06547v1](http://arxiv.org/pdf/2508.06547v1)**

> **作者:** Heran Wu; Zirun Zhou; Jingfeng Zhang
>
> **备注:** This is a tutorial note for educational purposes
>
> **摘要:** Traditional robotic systems typically decompose intelligence into independent modules for computer vision, natural language processing, and motion control. Vision-Language-Action (VLA) models fundamentally transform this approach by employing a single neural network that can simultaneously process visual observations, understand human instructions, and directly output robot actions -- all within a unified framework. However, these systems are highly dependent on high-quality training datasets that can capture the complex relationships between visual observations, language instructions, and robotic actions. This tutorial reviews three representative systems: the PyBullet simulation framework for flexible customized data generation, the LIBERO benchmark suite for standardized task definition and evaluation, and the RT-X dataset collection for large-scale multi-robot data acquisition. We demonstrated dataset generation approaches in PyBullet simulation and customized data collection within LIBERO, and provide an overview of the characteristics and roles of the RT-X dataset for large-scale multi-robot data acquisition.
>
---
#### [new 010] Efficient Safety Testing of Autonomous Vehicles via Adaptive Search over Crash-Derived Scenarios
- **分类: cs.RO; cs.AI**

- **简介: 论文提出一种基于事故场景的高效安全测试方法，通过提取真实事故特征并优化算法，提升自动驾驶在安全关键场景下的测试覆盖率，相较传统方法显著增强效率。**

- **链接: [http://arxiv.org/pdf/2508.06575v1](http://arxiv.org/pdf/2508.06575v1)**

> **作者:** Rui Zhou
>
> **摘要:** Ensuring the safety of autonomous vehicles (AVs) is paramount in their development and deployment. Safety-critical scenarios pose more severe challenges, necessitating efficient testing methods to validate AVs safety. This study focuses on designing an accelerated testing algorithm for AVs in safety-critical scenarios, enabling swift recognition of their driving capabilities. First, typical logical scenarios were extracted from real-world crashes in the China In-depth Mobility Safety Study-Traffic Accident (CIMSS-TA) database, obtaining pre-crash features through reconstruction. Second, Baidu Apollo, an advanced black-box automated driving system (ADS) is integrated to control the behavior of the ego vehicle. Third, we proposed an adaptive large-variable neighborhood-simulated annealing algorithm (ALVNS-SA) to expedite the testing process. Experimental results demonstrate a significant enhancement in testing efficiency when utilizing ALVNS-SA. It achieves an 84.00% coverage of safety-critical scenarios, with crash scenario coverage of 96.83% and near-crash scenario coverage of 92.07%. Compared to genetic algorithm (GA), adaptive large neighborhood-simulated annealing algorithm (ALNS-SA), and random testing, ALVNS-SA exhibits substantially higher coverage in safety-critical scenarios.
>
---
#### [new 011] MoRoCo: Multi-operator-robot Coordination, Interaction and Exploration under Restricted Communication
- **分类: cs.RO**

- **简介: 论文提出MoRoCo框架，解决受限通信下多操作员/机器人协同探索问题，通过三种模式切换及本地通信实现高效可靠交互。**

- **链接: [http://arxiv.org/pdf/2508.07657v1](http://arxiv.org/pdf/2508.07657v1)**

> **作者:** Zhuoli Tian; Yuyang Zhang; Jinsheng Wei; Meng Guo
>
> **备注:** 38 pages, 28 figures, Submitted to the International Journal of Robotics Research (IJRR). Project website: https://zl-tian.github.io/MoRoCo/
>
> **摘要:** Fleets of autonomous robots are increasingly deployed alongside multiple human operators to explore unknown environments, identify salient features, and perform complex tasks in scenarios such as subterranean exploration, reconnaissance, and search-and-rescue missions. In these contexts, communication is often severely limited to short-range exchanges via ad-hoc networks, posing challenges to coordination. While recent studies have addressed multi-robot exploration under communication constraints, they largely overlook the essential role of human operators and their real-time interaction with robotic teams. Operators may demand timely updates on the exploration progress and robot status, reprioritize or cancel tasks dynamically, or request live video feeds and control access. Conversely, robots may seek human confirmation for anomalous events or require help recovering from motion or planning failures. To enable such bilateral, context-aware interactions under restricted communication, this work proposes MoRoCo, a unified framework for online coordination and exploration in multi-operator, multi-robot systems. MoRoCo enables the team to adaptively switch among three coordination modes: spread mode for parallelized exploration with intermittent data sharing, migrate mode for coordinated relocation, and chain mode for maintaining high-bandwidth connectivity through multi-hop links. These transitions are managed through distributed algorithms via only local communication. Extensive large-scale human-in-the-loop simulations and hardware experiments validate the necessity of incorporating human robot interactions and demonstrate that MoRoCo enables efficient, reliable coordination under limited communication, marking a significant step toward robust human-in-the-loop multi-robot autonomy in challenging environments.
>
---
#### [new 012] Robust-Sub-Gaussian Model Predictive Control for Safe Ultrasound-Image-Guided Robotic Spinal Surgery
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出基于子高斯噪声建模的鲁棒模型预测控制框架，用于安全控制高维感知数据（如超声图像）的机器人脊柱手术，解决估计误差分布复杂导致的安全保障难题，并在仿真中验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.06744v1](http://arxiv.org/pdf/2508.06744v1)**

> **作者:** Yunke Ao; Manish Prajapat; Yarden As; Yassine Taoudi-Benchekroun; Fabio Carrillo; Hooman Esfandiari; Benjamin F. Grewe; Andreas Krause; Philipp Fürnstahl
>
> **摘要:** Safety-critical control using high-dimensional sensory feedback from optical data (e.g., images, point clouds) poses significant challenges in domains like autonomous driving and robotic surgery. Control can rely on low-dimensional states estimated from high-dimensional data. However, the estimation errors often follow complex, unknown distributions that standard probabilistic models fail to capture, making formal safety guarantees challenging. In this work, we introduce a novel characterization of these general estimation errors using sub-Gaussian noise with bounded mean. We develop a new technique for uncertainty propagation of proposed noise characterization in linear systems, which combines robust set-based methods with the propagation of sub-Gaussian variance proxies. We further develop a Model Predictive Control (MPC) framework that provides closed-loop safety guarantees for linear systems under the proposed noise assumption. We apply this MPC approach in an ultrasound-image-guided robotic spinal surgery pipeline, which contains deep-learning-based semantic segmentation, image-based registration, high-level optimization-based planning, and low-level robotic control. To validate the pipeline, we developed a realistic simulation environment integrating real human anatomy, robot dynamics, efficient ultrasound simulation, as well as in-vivo data of breathing motion and drilling force. Evaluation results in simulation demonstrate the potential of our approach for solving complex image-guided robotic surgery task while ensuring safety.
>
---
#### [new 013] An Evolutionary Game-Theoretic Merging Decision-Making Considering Social Acceptance for Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 论文提出基于进化博弈论的自动驾驶高速入口合并决策框架，解决现有方法难以应对动态复杂性和社会接受度的问题，通过多目标收益函数与实时反馈优化切入时机，提升效率、舒适与安全。**

- **链接: [http://arxiv.org/pdf/2508.07080v1](http://arxiv.org/pdf/2508.07080v1)**

> **作者:** Haolin Liu; Zijun Guo; Yanbo Chen; Jiaqi Chen; Huilong Yu; Junqiang Xi
>
> **摘要:** Highway on-ramp merging is of great challenge for autonomous vehicles (AVs), since they have to proactively interact with surrounding vehicles to enter the main road safely within limited time. However, existing decision-making algorithms fail to adequately address dynamic complexities and social acceptance of AVs, leading to suboptimal or unsafe merging decisions. To address this, we propose an evolutionary game-theoretic (EGT) merging decision-making framework, grounded in the bounded rationality of human drivers, which dynamically balances the benefits of both AVs and main-road vehicles (MVs). We formulate the cut-in decision-making process as an EGT problem with a multi-objective payoff function that reflects human-like driving preferences. By solving the replicator dynamic equation for the evolutionarily stable strategy (ESS), the optimal cut-in timing is derived, balancing efficiency, comfort, and safety for both AVs and MVs. A real-time driving style estimation algorithm is proposed to adjust the game payoff function online by observing the immediate reactions of MVs. Empirical results demonstrate that we improve the efficiency, comfort and safety of both AVs and MVs compared with existing game-theoretic and traditional planning approaches across multi-object metrics.
>
---
#### [new 014] Aerial Target Encirclement and Interception with Noisy Range Observations
- **分类: cs.RO**

- **简介: 论文提出基于抗同步轨迹和卡尔曼滤波的空中目标环绕与拦截方法，解决有噪声测距下的状态估计与动态控制问题，设计新型控制器实现目标切换，分析误差收敛性并验证系统有效性。**

- **链接: [http://arxiv.org/pdf/2508.08046v1](http://arxiv.org/pdf/2508.08046v1)**

> **作者:** Fen Liu; Shenghai Yuan; Thien-Minh Nguyen; Wei Meng; Lihua Xie
>
> **备注:** The paper has been accepted in Automatica
>
> **摘要:** This paper proposes a strategy to encircle and intercept a non-cooperative aerial point-mass moving target by leveraging noisy range measurements for state estimation. In this approach, the guardians actively ensure the observability of the target by using an anti-synchronization (AS), 3D ``vibrating string" trajectory, which enables rapid position and velocity estimation based on the Kalman filter. Additionally, a novel anti-target controller is designed for the guardians to enable adaptive transitions from encircling a protected target to encircling, intercepting, and neutralizing a hostile target, taking into consideration the input constraints of the guardians. Based on the guaranteed uniform observability, the exponentially bounded stability of the state estimation error and the convergence of the encirclement error are rigorously analyzed. Simulation results and real-world UAV experiments are presented to further validate the effectiveness of the system design.
>
---
#### [new 015] Improved Obstacle Avoidance for Autonomous Robots with ORCA-FLC
- **分类: cs.RO**

- **简介: 论文提出ORCA-FLC算法，通过模糊逻辑控制器优化ORCA，解决多智能体环境中避障效率低、计算复杂的问题，提升路径规划的鲁棒性和适应性。**

- **链接: [http://arxiv.org/pdf/2508.06722v1](http://arxiv.org/pdf/2508.06722v1)**

> **作者:** Justin London
>
> **摘要:** Obstacle avoidance enables autonomous agents and robots to operate safely and efficiently in dynamic and complex environments, reducing the risk of collisions and damage. For a robot or autonomous system to successfully navigate through obstacles, it must be able to detect such obstacles. While numerous collision avoidance algorithms like the dynamic window approach (DWA), timed elastic bands (TEB), and reciprocal velocity obstacles (RVO) have been proposed, they may lead to suboptimal paths due to fixed weights, be computationally expensive, or have limited adaptability to dynamic obstacles in multi-agent environments. Optimal reciprocal collision avoidance (ORCA), which improves on RVO, provides smoother trajectories and stronger collision avoidance guarantees. We propose ORCA-FL to improve on ORCA by using fuzzy logic controllers (FLCs) to better handle uncertainty and imprecision for obstacle avoidance in path planning. Numerous multi-agent experiments are conducted and it is shown that ORCA-FL can outperform ORCA in reducing the number of collision if the agent has a velocity that exceeds a certain threshold. In addition, a proposed algorithm for improving ORCA-FL using fuzzy Q reinforcement learning (FQL) is detailed for optimizing and tuning FLCs.
>
---
#### [new 016] Impact of Gaze-Based Interaction and Augmentation on Human-Robot Collaboration in Critical Tasks
- **分类: cs.RO**

- **简介: 本研究探讨注视基交互与增强在关键任务中的人机协作影响，通过模拟搜救任务验证聚焦增强提升任务性能并减少认知负荷。**

- **链接: [http://arxiv.org/pdf/2508.07244v1](http://arxiv.org/pdf/2508.07244v1)**

> **作者:** Ayesha Jena; Stefan Reitmann; Elin Anna Topp
>
> **摘要:** We present a user study analyzing head-gaze-based robot control and foveated visual augmentation in a simulated search-and-rescue task. Results show that foveated augmentation significantly improves task performance, reduces cognitive load by 38%, and shortens task time by over 60%. Head-gaze patterns analysed over both the entire task duration and shorter time segments show that near and far attention capture is essential to better understand user intention in critical scenarios. Our findings highlight the potential of foveation as an augmentation technique and the need to further study gaze measures to leverage them during critical tasks.
>
---
#### [new 017] Learning a Vision-Based Footstep Planner for Hierarchical Walking Control
- **分类: cs.RO**

- **简介: 论文提出基于视觉的分层行走控制框架，解决传统方法依赖 proprioception 和手动视觉管道的不足，通过强化学习生成足部规划指令并结合操作空间控制器实现动态地形导航，采用低维状态表示降低复杂度。**

- **链接: [http://arxiv.org/pdf/2508.06779v1](http://arxiv.org/pdf/2508.06779v1)**

> **作者:** Minku Kim; Brian Acosta; Pratik Chaudhari; Michael Posa
>
> **备注:** 8 pages, 8 figures, accepted to 2025 IEEE-RAS 24th International Conference on Humanoid Robots
>
> **摘要:** Bipedal robots demonstrate potential in navigating challenging terrains through dynamic ground contact. However, current frameworks often depend solely on proprioception or use manually designed visual pipelines, which are fragile in real-world settings and complicate real-time footstep planning in unstructured environments. To address this problem, we present a vision-based hierarchical control framework that integrates a reinforcement learning high-level footstep planner, which generates footstep commands based on a local elevation map, with a low-level Operational Space Controller that tracks the generated trajectories. We utilize the Angular Momentum Linear Inverted Pendulum model to construct a low-dimensional state representation to capture an informative encoding of the dynamics while reducing complexity. We evaluate our method across different terrain conditions using the underactuated bipedal robot Cassie and investigate the capabilities and challenges of our approach through simulation and hardware experiments.
>
---
#### [new 018] Feedback Control of a Single-Tail Bioinspired 59-mg Swimmer
- **分类: cs.RO**

- **简介: 论文提出一种基于形状记忆合金的单尾仿生机器人，通过改进结构设计实现反馈控制，提升亚克格规模下的运动速度和轨迹跟踪精度。**

- **链接: [http://arxiv.org/pdf/2508.07566v1](http://arxiv.org/pdf/2508.07566v1)**

> **作者:** Conor K. Trygstad; Cody R. Longwell; Francisco M. F. R. Gonçalves; Elijah K. Blankenship; Néstor O. Pérez-Arancibia
>
> **备注:** To be presented at the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** We present an evolved steerable version of the single-tail Fish-&-Ribbon-Inspired Small Swimming Harmonic roBot (FRISSHBot), a 59-mg biologically inspired swimmer, which is driven by a new shape-memory alloy (SMA)-based bimorph actuator. The new FRISSHBot is controllable in the two-dimensional (2D) space, which enabled the first demonstration of feedback-controlled trajectory tracking of a single-tail aquatic robot with onboard actuation at the subgram scale. These new capabilities are the result of a physics-informed design with an enlarged head and shortened tail relative to those of the original platform. Enhanced by its design, this new platform achieves forward swimming speeds of up to 13.6 mm/s (0.38 Bl/s), which is over four times that of the original platform. Furthermore, when following 2D references in closed loop, the tested FRISSHBot prototype attains forward swimming speeds of up to 9.1 mm/s, root-mean-square (RMS) tracking errors as low as 2.6 mm, turning rates of up to 13.1 {\deg}/s, and turning radii as small as 10 mm.
>
---
#### [new 019] Optimization of Flip-Landing Trajectories for Starship based on a Deep Learned Simulator
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出基于深度学习模拟器的端到端优化框架，针对可重复使用航天器翻转着陆问题，通过高保真CFD数据训练神经网络预测气动特性，并与可微分动力学求解器耦合，实现高精度、物理一致的控制序列优化，支持长时程滚动与复杂机动。**

- **链接: [http://arxiv.org/pdf/2508.06520v1](http://arxiv.org/pdf/2508.06520v1)**

> **作者:** Liwei Chen; Tong Qin; Zhenhua Huangfu; Li Li; Wei Wei
>
> **摘要:** We propose a differentiable optimization framework for flip-and-landing trajectory design of reusable spacecraft, exemplified by the Starship vehicle. A deep neural network surrogate, trained on high-fidelity CFD data, predicts aerodynamic forces and moments, and is tightly coupled with a differentiable rigid-body dynamics solver. This enables end-to-end gradient-based trajectory optimization without linearization or convex relaxation. The framework handles actuator limits and terminal landing constraints, producing physically consistent, optimized control sequences. Both standard automatic differentiation and Neural ODEs are applied to support long-horizon rollouts. Results demonstrate the framework's effectiveness in modeling and optimizing complex maneuvers with high nonlinearities. This work lays the groundwork for future extensions involving unsteady aerodynamics, plume interactions, and intelligent guidance design.
>
---
#### [new 020] PCHands: PCA-based Hand Pose Synergy Representation on Manipulators with N-DoF
- **分类: cs.RO**

- **简介: 论文提出基于PCA的手部姿态协同表示方法，解决不同机械臂结构与自由度下的通用表示问题，通过统一锚点描述实现高效RL应用。**

- **链接: [http://arxiv.org/pdf/2508.07945v1](http://arxiv.org/pdf/2508.07945v1)**

> **作者:** En Yen Puang; Federico Ceola; Giulia Pasquale; Lorenzo Natale
>
> **备注:** 2025 IEEE-RAS 24th International Conference on Humanoid Robots
>
> **摘要:** We consider the problem of learning a common representation for dexterous manipulation across manipulators of different morphologies. To this end, we propose PCHands, a novel approach for extracting hand postural synergies from a large set of manipulators. We define a simplified and unified description format based on anchor positions for manipulators ranging from 2-finger grippers to 5-finger anthropomorphic hands. This enables learning a variable-length latent representation of the manipulator configuration and the alignment of the end-effector frame of all manipulators. We show that it is possible to extract principal components from this latent representation that is universal across manipulators of different structures and degrees of freedom. To evaluate PCHands, we use this compact representation to encode observation and action spaces of control policies for dexterous manipulation tasks learned with RL. In terms of learning efficiency and consistency, the proposed representation outperforms a baseline that learns the same tasks in joint space. We additionally show that PCHands performs robustly in RL from demonstration, when demonstrations are provided from a different manipulator. We further support our results with real-world experiments that involve a 2-finger gripper and a 4-finger anthropomorphic hand. Code and additional material are available at https://hsp-iit.github.io/PCHands/.
>
---
#### [new 021] End-to-End Humanoid Robot Safe and Comfortable Locomotion Policy
- **分类: cs.RO**

- **简介: 论文提出端到端 humanoid 机器人运动策略，解决动态环境安全舒适问题，结合 CMDP 和 CBFs 实现安全约束，引入舒适奖励促进平滑运动，并通过 sim-to-real 验证。**

- **链接: [http://arxiv.org/pdf/2508.07611v1](http://arxiv.org/pdf/2508.07611v1)**

> **作者:** Zifan Wang; Xun Yang; Jianzhuang Zhao; Jiaming Zhou; Teli Ma; Ziyao Gao; Arash Ajoudani; Junwei Liang
>
> **摘要:** The deployment of humanoid robots in unstructured, human-centric environments requires navigation capabilities that extend beyond simple locomotion to include robust perception, provable safety, and socially aware behavior. Current reinforcement learning approaches are often limited by blind controllers that lack environmental awareness or by vision-based systems that fail to perceive complex 3D obstacles. In this work, we present an end-to-end locomotion policy that directly maps raw, spatio-temporal LiDAR point clouds to motor commands, enabling robust navigation in cluttered dynamic scenes. We formulate the control problem as a Constrained Markov Decision Process (CMDP) to formally separate safety from task objectives. Our key contribution is a novel methodology that translates the principles of Control Barrier Functions (CBFs) into costs within the CMDP, allowing a model-free Penalized Proximal Policy Optimization (P3O) to enforce safety constraints during training. Furthermore, we introduce a set of comfort-oriented rewards, grounded in human-robot interaction research, to promote motions that are smooth, predictable, and less intrusive. We demonstrate the efficacy of our framework through a successful sim-to-real transfer to a physical humanoid robot, which exhibits agile and safe navigation around both static and dynamic 3D obstacles.
>
---
#### [new 022] Progressive Bird's Eye View Perception for Safety-Critical Autonomous Driving: A Comprehensive Survey
- **分类: cs.RO; cs.CV**

- **简介: 该论文综述安全关键自动驾驶中的BEV感知任务，分析多阶段框架及挑战，评估数据集并提出未来方向。**

- **链接: [http://arxiv.org/pdf/2508.07560v1](http://arxiv.org/pdf/2508.07560v1)**

> **作者:** Yan Gong; Naibang Wang; Jianli Lu; Xinyu Zhang; Yongsheng Gao; Jie Zhao; Zifan Huang; Haozhi Bai; Nanxin Zeng; Nayu Su; Lei Yang; Ziying Song; Xiaoxi Hu; Xinmin Jiang; Xiaojuan Zhang; Susanto Rahardja
>
> **摘要:** Bird's-Eye-View (BEV) perception has become a foundational paradigm in autonomous driving, enabling unified spatial representations that support robust multi-sensor fusion and multi-agent collaboration. As autonomous vehicles transition from controlled environments to real-world deployment, ensuring the safety and reliability of BEV perception in complex scenarios - such as occlusions, adverse weather, and dynamic traffic - remains a critical challenge. This survey provides the first comprehensive review of BEV perception from a safety-critical perspective, systematically analyzing state-of-the-art frameworks and implementation strategies across three progressive stages: single-modality vehicle-side, multimodal vehicle-side, and multi-agent collaborative perception. Furthermore, we examine public datasets encompassing vehicle-side, roadside, and collaborative settings, evaluating their relevance to safety and robustness. We also identify key open-world challenges - including open-set recognition, large-scale unlabeled data, sensor degradation, and inter-agent communication latency - and outline future research directions, such as integration with end-to-end autonomous driving systems, embodied intelligence, and large language models.
>
---
#### [new 023] Learning Causal Structure Distributions for Robust Planning
- **分类: cs.RO; cs.AI; cs.LG; cs.SY; eess.SY**

- **简介: 论文提出通过学习因果结构分布，结合不确定性建模，构建稳健动力学模型以提升机器人规划鲁棒性，采用编码器-解码器框架采样因果图，验证在模拟与真实环境中有效。**

- **链接: [http://arxiv.org/pdf/2508.06742v1](http://arxiv.org/pdf/2508.06742v1)**

> **作者:** Alejandro Murillo-Gonzalez; Junhong Xu; Lantao Liu
>
> **摘要:** Structural causal models describe how the components of a robotic system interact. They provide both structural and functional information about the relationships that are present in the system. The structural information outlines the variables among which there is interaction. The functional information describes how such interactions work, via equations or learned models. In this paper we find that learning the functional relationships while accounting for the uncertainty about the structural information leads to more robust dynamics models which improves downstream planning, while using significantly lower computational resources. This in contrast with common model-learning methods that ignore the causal structure and fail to leverage the sparsity of interactions in robotic systems. We achieve this by estimating a causal structure distribution that is used to sample causal graphs that inform the latent-space representations in an encoder-multidecoder probabilistic model. We show that our model can be used to learn the dynamics of a robot, which together with a sampling-based planner can be used to perform new tasks in novel environments, provided an objective function for the new requirement is available. We validate our method using manipulators and mobile robots in both simulation and the real-world. Additionally, we validate the learned dynamics' adaptability and increased robustness to corrupted inputs and changes in the environment, which is highly desirable in challenging real-world robotics scenarios. Video: https://youtu.be/X6k5t7OOnNc.
>
---
#### [new 024] Integrating Neurosymbolic AI in Advanced Air Mobility: A Comprehensive Survey
- **分类: cs.RO; cs.AI; cs.NE**

- **简介: 论文综述神经符号AI在AAM中的整合，解决监管、运营、安全挑战，分析应用、分类进展及未来方向。**

- **链接: [http://arxiv.org/pdf/2508.07163v1](http://arxiv.org/pdf/2508.07163v1)**

> **作者:** Kamal Acharya; Iman Sharifi; Mehul Lad; Liang Sun; Houbing Song
>
> **备注:** 9 pages, 4 figures, IJCAI-2025 (accepted)
>
> **摘要:** Neurosymbolic AI combines neural network adaptability with symbolic reasoning, promising an approach to address the complex regulatory, operational, and safety challenges in Advanced Air Mobility (AAM). This survey reviews its applications across key AAM domains such as demand forecasting, aircraft design, and real-time air traffic management. Our analysis reveals a fragmented research landscape where methodologies, including Neurosymbolic Reinforcement Learning, have shown potential for dynamic optimization but still face hurdles in scalability, robustness, and compliance with aviation standards. We classify current advancements, present relevant case studies, and outline future research directions aimed at integrating these approaches into reliable, transparent AAM systems. By linking advanced AI techniques with AAM's operational demands, this work provides a concise roadmap for researchers and practitioners developing next-generation air mobility solutions.
>
---
#### [new 025] Risk Map As Middleware: Towards Interpretable Cooperative End-to-end Autonomous Driving for Risk-Aware Planning
- **分类: cs.RO**

- **简介: 论文提出RiskMM框架，解决单Agent端到端自动驾驶中遮挡、感知局限及可解释性问题，通过风险地图融合感知与环境交互，结合MPC实现风险-aware轨迹规划，提升协作驾驶透明度。**

- **链接: [http://arxiv.org/pdf/2508.07686v1](http://arxiv.org/pdf/2508.07686v1)**

> **作者:** Mingyue Lei; Zewei Zhou; Hongchen Li; Jiaqi Ma; Jia Hu
>
> **摘要:** End-to-end paradigm has emerged as a promising approach to autonomous driving. However, existing single-agent end-to-end pipelines are often constrained by occlusion and limited perception range, resulting in hazardous driving. Furthermore, their black-box nature prevents the interpretability of the driving behavior, leading to an untrustworthiness system. To address these limitations, we introduce Risk Map as Middleware (RiskMM) and propose an interpretable cooperative end-to-end driving framework. The risk map learns directly from the driving data and provides an interpretable spatiotemporal representation of the scenario from the upstream perception and the interactions between the ego vehicle and the surrounding environment for downstream planning. RiskMM first constructs a multi-agent spatiotemporal representation with unified Transformer-based architecture, then derives risk-aware representations by modeling interactions among surrounding environments with attention. These representations are subsequently fed into a learning-based Model Predictive Control (MPC) module. The MPC planner inherently accommodates physical constraints and different vehicle types and can provide interpretation by aligning learned parameters with explicit MPC elements. Evaluations conducted on the real-world V2XPnP-Seq dataset confirm that RiskMM achieves superior and robust performance in risk-aware trajectory planning, significantly enhancing the interpretability of the cooperative end-to-end driving framework. The codebase will be released to facilitate future research in this field.
>
---
#### [new 026] ODYSSEY: Open-World Quadrupeds Exploration and Manipulation for Long-Horizon Tasks
- **分类: cs.RO; cs.CV**

- **简介: 论文提出ODYSSEY框架，针对开放世界四足机器人探索与操作的长期任务，解决桌面场景限制、泛化不足及机动性与控制平衡问题，通过视觉语言模型实现分步规划与鲁棒控制，建立首个长程移动任务基准，验证其在真实环境中的泛化与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.08240v1](http://arxiv.org/pdf/2508.08240v1)**

> **作者:** Kaijun Wang; Liqin Lu; Mingyu Liu; Jianuo Jiang; Zeju Li; Bolin Zhang; Wancai Zheng; Xinyi Yu; Hao Chen; Chunhua Shen
>
> **摘要:** Language-guided long-horizon mobile manipulation has long been a grand challenge in embodied semantic reasoning, generalizable manipulation, and adaptive locomotion. Three fundamental limitations hinder progress: First, although large language models have improved spatial reasoning and task planning through semantic priors, existing implementations remain confined to tabletop scenarios, failing to address the constrained perception and limited actuation ranges of mobile platforms. Second, current manipulation strategies exhibit insufficient generalization when confronted with the diverse object configurations encountered in open-world environments. Third, while crucial for practical deployment, the dual requirement of maintaining high platform maneuverability alongside precise end-effector control in unstructured settings remains understudied. In this work, we present ODYSSEY, a unified mobile manipulation framework for agile quadruped robots equipped with manipulators, which seamlessly integrates high-level task planning with low-level whole-body control. To address the challenge of egocentric perception in language-conditioned tasks, we introduce a hierarchical planner powered by a vision-language model, enabling long-horizon instruction decomposition and precise action execution. At the control level, our novel whole-body policy achieves robust coordination across challenging terrains. We further present the first benchmark for long-horizon mobile manipulation, evaluating diverse indoor and outdoor scenarios. Through successful sim-to-real transfer, we demonstrate the system's generalization and robustness in real-world deployments, underscoring the practicality of legged manipulators in unstructured environments. Our work advances the feasibility of generalized robotic assistants capable of complex, dynamic tasks. Our project page: https://kaijwang.github.io/odyssey.github.io/
>
---
#### [new 027] Grasp-HGN: Grasping the Unexpected
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 论文提出Grasp-HGN任务，解决机器人假肢手在新环境下的泛化问题，通过语义投影、Grasp-LLaVA和HGN混合网络提升抓取准确性与性能。**

- **链接: [http://arxiv.org/pdf/2508.07648v1](http://arxiv.org/pdf/2508.07648v1)**

> **作者:** Mehrshad Zandigohar; Mallesham Dasari; Gunar Schirner
>
> **备注:** Paper accepted at ACM Transactions on Embedded Computing Systems
>
> **摘要:** For transradial amputees, robotic prosthetic hands promise to regain the capability to perform daily living activities. To advance next-generation prosthetic hand control design, it is crucial to address current shortcomings in robustness to out of lab artifacts, and generalizability to new environments. Due to the fixed number of object to interact with in existing datasets, contrasted with the virtually infinite variety of objects encountered in the real world, current grasp models perform poorly on unseen objects, negatively affecting users' independence and quality of life. To address this: (i) we define semantic projection, the ability of a model to generalize to unseen object types and show that conventional models like YOLO, despite 80% training accuracy, drop to 15% on unseen objects. (ii) we propose Grasp-LLaVA, a Grasp Vision Language Model enabling human-like reasoning to infer the suitable grasp type estimate based on the object's physical characteristics resulting in a significant 50.2% accuracy over unseen object types compared to 36.7% accuracy of an SOTA grasp estimation model. Lastly, to bridge the performance-latency gap, we propose Hybrid Grasp Network (HGN), an edge-cloud deployment infrastructure enabling fast grasp estimation on edge and accurate cloud inference as a fail-safe, effectively expanding the latency vs. accuracy Pareto. HGN with confidence calibration (DC) enables dynamic switching between edge and cloud models, improving semantic projection accuracy by 5.6% (to 42.3%) with 3.5x speedup over the unseen object types. Over a real-world sample mix, it reaches 86% average accuracy (12.2% gain over edge-only), and 2.2x faster inference than Grasp-LLaVA alone.
>
---
#### [new 028] From Data to Safe Mobile Robot Navigation: An Efficient and Modular Robust MPC Design Pipeline
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出基于数据的高效鲁棒MPC设计管道，用于安全移动机器人导航，通过闭环实验数据估计干扰边界并合成鲁棒输出反馈方案，实现实验验证。**

- **链接: [http://arxiv.org/pdf/2508.07045v1](http://arxiv.org/pdf/2508.07045v1)**

> **作者:** Dennis Benders; Johannes Köhler; Robert Babuška; Javier Alonso-Mora; Laura Ferranti
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** Model predictive control (MPC) is a powerful strategy for planning and control in autonomous mobile robot navigation. However, ensuring safety in real-world deployments remains challenging due to the presence of disturbances and measurement noise. Existing approaches often rely on idealized assumptions, neglect the impact of noisy measurements, and simply heuristically guess unrealistic bounds. In this work, we present an efficient and modular robust MPC design pipeline that systematically addresses these limitations. The pipeline consists of an iterative procedure that leverages closed-loop experimental data to estimate disturbance bounds and synthesize a robust output-feedback MPC scheme. We provide the pipeline in the form of deterministic and reproducible code to synthesize the robust output-feedback MPC from data. We empirically demonstrate robust constraint satisfaction and recursive feasibility in quadrotor simulations using Gazebo.
>
---
#### [new 029] Navigation and Exploration with Active Inference: from Biology to Industry
- **分类: cs.RO**

- **简介: 论文提出基于主动推理框架（AIF）的实时机器人导航系统，通过增量构建拓扑地图、最小化不确定性实现动态环境下的高效探索，无需先验训练，验证其在2D/3D环境中的适应性与生物启发式性能。**

- **链接: [http://arxiv.org/pdf/2508.07269v1](http://arxiv.org/pdf/2508.07269v1)**

> **作者:** Daria de Tinguy; Tim Verbelen; Bart Dhoedt
>
> **备注:** conference IWAI 2025 - accepted (in processing)
>
> **摘要:** By building and updating internal cognitive maps, animals exhibit extraordinary navigation abilities in complex, dynamic environments. Inspired by these biological mechanisms, we present a real time robotic navigation system grounded in the Active Inference Framework (AIF). Our model incrementally constructs a topological map, infers the agent's location, and plans actions by minimising expected uncertainty and fulfilling perceptual goals without any prior training. Integrated into the ROS2 ecosystem, we validate its adaptability and efficiency across both 2D and 3D environments (simulated and real world), demonstrating competitive performance with traditional and state of the art exploration approaches while offering a biologically inspired navigation approach.
>
---
#### [new 030] MolmoAct: Action Reasoning Models that can Reason in Space
- **分类: cs.RO**

- **简介: 论文提出一种基于视觉-语言-动作的结构化三阶段模型MolmoAct，解决机器人控制中感知与指令直接映射导致的适应性不足问题，通过深度感知令牌生成空间计划并预测动作，实现可解释可控行为，取得显著性能提升并开源数据集与代码。**

- **链接: [http://arxiv.org/pdf/2508.07917v1](http://arxiv.org/pdf/2508.07917v1)**

> **作者:** Jason Lee; Jiafei Duan; Haoquan Fang; Yuquan Deng; Shuo Liu; Boyang Li; Bohan Fang; Jieyu Zhang; Yi Ru Wang; Sangho Lee; Winson Han; Wilbert Pumacay; Angelica Wu; Rose Hendrix; Karen Farley; Eli VanderBilt; Ali Farhadi; Dieter Fox; Ranjay Krishna
>
> **备注:** Appendix on Blogpost: https://allenai.org/blog/molmoact
>
> **摘要:** Reasoning is central to purposeful action, yet most robotic foundation models map perception and instructions directly to control, which limits adaptability, generalization, and semantic grounding. We introduce Action Reasoning Models (ARMs), a class of vision-language-action models that integrate perception, planning, and control through a structured three-stage pipeline. Our model, MolmoAct, encodes observations and instructions into depth-aware perception tokens, generates mid-level spatial plans as editable trajectory traces, and predicts precise low-level actions, enabling explainable and steerable behavior. MolmoAct-7B-D achieves strong performance across simulation and real-world settings: 70.5% zero-shot accuracy on SimplerEnv Visual Matching tasks, surpassing closed-source Pi-0 and GR00T N1; 86.6% average success on LIBERO, including an additional 6.3% gain over ThinkAct on long-horizon tasks; and in real-world fine-tuning, an additional 10% (single-arm) and an additional 22.7% (bimanual) task progression over Pi-0-FAST. It also outperforms baselines by an additional 23.3% on out-of-distribution generalization and achieves top human-preference scores for open-ended instruction following and trajectory steering. Furthermore, we release, for the first time, the MolmoAct Dataset -- a mid-training robot dataset comprising over 10,000 high quality robot trajectories across diverse scenarios and tasks. Training with this dataset yields an average 5.5% improvement in general performance over the base model. We release all model weights, training code, our collected dataset, and our action reasoning dataset, establishing MolmoAct as both a state-of-the-art robotics foundation model and an open blueprint for building ARMs that transform perception into purposeful action through structured reasoning. Blogpost: https://allenai.org/blog/molmoact
>
---
#### [new 031] Touch Speaks, Sound Feels: A Multimodal Approach to Affective and Social Touch from Robots to Humans
- **分类: cs.RO**

- **简介: 论文提出多模态方法，利用振动与音频结合，解决机器人传达触觉社交手势问题，实验显示联合模态提升情感识别准确性，单通道有效但需多模态协同。**

- **链接: [http://arxiv.org/pdf/2508.07839v1](http://arxiv.org/pdf/2508.07839v1)**

> **作者:** Qiaoqiao Ren; Tony Belpaeme
>
> **摘要:** Affective tactile interaction constitutes a fundamental component of human communication. In natural human-human encounters, touch is seldom experienced in isolation; rather, it is inherently multisensory. Individuals not only perceive the physical sensation of touch but also register the accompanying auditory cues generated through contact. The integration of haptic and auditory information forms a rich and nuanced channel for emotional expression. While extensive research has examined how robots convey emotions through facial expressions and speech, their capacity to communicate social gestures and emotions via touch remains largely underexplored. To address this gap, we developed a multimodal interaction system incorporating a 5*5 grid of 25 vibration motors synchronized with audio playback, enabling robots to deliver combined haptic-audio stimuli. In an experiment involving 32 Chinese participants, ten emotions and six social gestures were presented through vibration, sound, or their combination. Participants rated each stimulus on arousal and valence scales. The results revealed that (1) the combined haptic-audio modality significantly enhanced decoding accuracy compared to single modalities; (2) each individual channel-vibration or sound-effectively supported certain emotions recognition, with distinct advantages depending on the emotional expression; and (3) gestures alone were generally insufficient for conveying clearly distinguishable emotions. These findings underscore the importance of multisensory integration in affective human-robot interaction and highlight the complementary roles of haptic and auditory cues in enhancing emotional communication.
>
---
#### [new 032] DETACH: Cross-domain Learning for Long-Horizon Tasks via Mixture of Disentangled Experts
- **分类: cs.RO; cs.AI**

- **简介: 论文提出DETACh框架，针对长时序任务（LH）在人类场景交互（HSI）中跨域泛化不足的问题，通过双流解耦环境与技能模块实现跨域学习，提升子任务成功率23%和执行效率29%。**

- **链接: [http://arxiv.org/pdf/2508.07842v1](http://arxiv.org/pdf/2508.07842v1)**

> **作者:** Yutong Shen; Hangxu Liu; Penghui Liu; Ruizhe Xia; Tianyi Yao; Yitong Sun; Tongtong Feng
>
> **备注:** 14 pages,8 figures. Submitted to AAAI'26
>
> **摘要:** Long-Horizon (LH) tasks in Human-Scene Interaction (HSI) are complex multi-step tasks that require continuous planning, sequential decision-making, and extended execution across domains to achieve the final goal. However, existing methods heavily rely on skill chaining by concatenating pre-trained subtasks, with environment observations and self-state tightly coupled, lacking the ability to generalize to new combinations of environments and skills, failing to complete various LH tasks across domains. To solve this problem, this paper presents DETACH, a cross-domain learning framework for LH tasks via biologically inspired dual-stream disentanglement. Inspired by the brain's "where-what" dual pathway mechanism, DETACH comprises two core modules: i) an environment learning module for spatial understanding, which captures object functions, spatial relationships, and scene semantics, achieving cross-domain transfer through complete environment-self disentanglement; ii) a skill learning module for task execution, which processes self-state information including joint degrees of freedom and motor patterns, enabling cross-skill transfer through independent motor pattern encoding. We conducted extensive experiments on various LH tasks in HSI scenes. Compared with existing methods, DETACH can achieve an average subtasks success rate improvement of 23% and average execution efficiency improvement of 29%.
>
---
#### [new 033] AgentWorld: An Interactive Simulation Platform for Scene Construction and Mobile Robotic Manipulation
- **分类: cs.RO**

- **简介: 论文提出AgentWorld平台，用于机器人场景构建与移动操控，解决仿真到现实迁移问题，通过数据集和方法对比验证有效性。**

- **链接: [http://arxiv.org/pdf/2508.07770v1](http://arxiv.org/pdf/2508.07770v1)**

> **作者:** Yizheng Zhang; Zhenjun Yu; Jiaxin Lai; Cewu Lu; Lei Han
>
> **备注:** Accepted by Conference on Robot Learning 2025
>
> **摘要:** We introduce AgentWorld, an interactive simulation platform for developing household mobile manipulation capabilities. Our platform combines automated scene construction that encompasses layout generation, semantic asset placement, visual material configuration, and physics simulation, with a dual-mode teleoperation system supporting both wheeled bases and humanoid locomotion policies for data collection. The resulting AgentWorld Dataset captures diverse tasks ranging from primitive actions (pick-and-place, push-pull, etc.) to multistage activities (serve drinks, heat up food, etc.) across living rooms, bedrooms, and kitchens. Through extensive benchmarking of imitation learning methods including behavior cloning, action chunking transformers, diffusion policies, and vision-language-action models, we demonstrate the dataset's effectiveness for sim-to-real transfer. The integrated system provides a comprehensive solution for scalable robotic skill acquisition in complex home environments, bridging the gap between simulation-based training and real-world deployment. The code, datasets will be available at https://yizhengzhang1.github.io/agent_world/
>
---
#### [new 034] Robot and Overhead Crane Collaboration Scheme to Enhance Payload Manipulation
- **分类: cs.RO**

- **简介: 论文提出机器人与吊车协作方案，解决传统手动操作风险与效率低的问题，通过交互力控制实现安全流畅的负载引导，设计双控制函数优化协作流程并进行仿真验证。**

- **链接: [http://arxiv.org/pdf/2508.07758v1](http://arxiv.org/pdf/2508.07758v1)**

> **作者:** Antonio Rosales; Alaa Abderrahim; Markku Suomalainen; Mikael Haag; Tapio Heikkilä
>
> **摘要:** This paper presents a scheme to enhance payload manipulation using a robot collaborating with an overhead crane. In the current industrial practice, when the crane's payload has to be accurately manipulated and located in a desired position, the task becomes laborious and risky since the operators have to guide the fine motions of the payload by hand. In the proposed collaborative scheme, the crane lifts the payload while the robot's end-effector guides it toward the desired position. The only link between the robot and the crane is the interaction force produced during the guiding of the payload. Two admittance transfer functions are considered to accomplish harmless and smooth contact with the payload. The first is used in a position-based admittance control integrated with the robot. The second one adds compliance to the crane by processing the interaction force through the admittance transfer function to generate a crane's velocity command that makes the crane follow the payload. Then the robot's end-effector and the crane move collaboratively to guide the payload to the desired location. A method is presented to design the admittance controllers that accomplish a fluent robot-crane collaboration. Simulations and experiments validating the scheme potential are shown.
>
---
#### [new 035] Triple-S: A Collaborative Multi-LLM Framework for Solving Long-Horizon Implicative Tasks in Robotics
- **分类: cs.RO**

- **简介: 论文提出Triple-S框架，通过多LLM协作解决机器人长时序隐含任务中的API错误问题，采用In-Context学习分阶段处理，提升成功率并支持泛化。**

- **链接: [http://arxiv.org/pdf/2508.07421v1](http://arxiv.org/pdf/2508.07421v1)**

> **作者:** Zixi Jia; Hongbin Gao; Fashe Li; Jiqiang Liu; Hexiao Li; Qinghua Liu
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** Leveraging Large Language Models (LLMs) to write policy code for controlling robots has gained significant attention. However, in long-horizon implicative tasks, this approach often results in API parameter, comments and sequencing errors, leading to task failure. To address this problem, we propose a collaborative Triple-S framework that involves multiple LLMs. Through In-Context Learning, different LLMs assume specific roles in a closed-loop Simplification-Solution-Summary process, effectively improving success rates and robustness in long-horizon implicative tasks. Additionally, a novel demonstration library update mechanism which learned from success allows it to generalize to previously failed tasks. We validate the framework in the Long-horizon Desktop Implicative Placement (LDIP) dataset across various baseline models, where Triple-S successfully executes 89% of tasks in both observable and partially observable scenarios. Experiments in both simulation and real-world robot settings further validated the effectiveness of Triple-S. Our code and dataset is available at: https://github.com/Ghbbbbb/Triple-S.
>
---
#### [new 036] DexFruit: Dexterous Manipulation and Gaussian Splatting Inspection of Fruit
- **分类: cs.RO**

- **简介: 论文提出DexFruit框架，通过触觉感知与扩散政策实现水果无损抓取，结合FruitSplat技术量化视觉损伤，提升抓取成功率并减少 bruising，验证其在三种水果上的有效性。**

- **链接: [http://arxiv.org/pdf/2508.07118v1](http://arxiv.org/pdf/2508.07118v1)**

> **作者:** Aiden Swann; Alex Qiu; Matthew Strong; Angelina Zhang; Samuel Morstein; Kai Rayle; Monroe Kennedy III
>
> **备注:** 8 pages, 5 figures
>
> **摘要:** DexFruit is a robotic manipulation framework that enables gentle, autonomous handling of fragile fruit and precise evaluation of damage. Many fruits are fragile and prone to bruising, thus requiring humans to manually harvest them with care. In this work, we demonstrate by using optical tactile sensing, autonomous manipulation of fruit with minimal damage can be achieved. We show that our tactile informed diffusion policies outperform baselines in both reduced bruising and pick-and-place success rate across three fruits: strawberries, tomatoes, and blackberries. In addition, we introduce FruitSplat, a novel technique to represent and quantify visual damage in high-resolution 3D representation via 3D Gaussian Splatting (3DGS). Existing metrics for measuring damage lack quantitative rigor or require expensive equipment. With FruitSplat, we distill a 2D strawberry mask as well as a 2D bruise segmentation mask into the 3DGS representation. Furthermore, this representation is modular and general, compatible with any relevant 2D model. Overall, we demonstrate a 92% grasping policy success rate, up to a 20% reduction in visual bruising, and up to an 31% improvement in grasp success rate on challenging fruit compared to our baselines across our three tested fruits. We rigorously evaluate this result with over 630 trials. Please checkout our website at https://dex-fruit.github.io .
>
---
#### [new 037] Vibration-Based Energy Metric for Restoring Needle Alignment in Autonomous Robotic Ultrasound
- **分类: cs.RO; cs.CV**

- **简介: 论文提出基于振动的能量指标方法，解决机器人超声引导中针头对齐难题，通过周期性振动恢复针头位置，降低因图像分辨率和噪声导致的误差。**

- **链接: [http://arxiv.org/pdf/2508.06921v1](http://arxiv.org/pdf/2508.06921v1)**

> **作者:** Zhongyu Chen; Chenyang Li; Xuesong Li; Dianye Huang; Zhongliang Jiang; Stefanie Speidel; Xiangyu Chu; K. W. Samuel Au
>
> **摘要:** Precise needle alignment is essential for percutaneous needle insertion in robotic ultrasound-guided procedures. However, inherent challenges such as speckle noise, needle-like artifacts, and low image resolution make robust needle detection difficult, particularly when visibility is reduced or lost. In this paper, we propose a method to restore needle alignment when the ultrasound imaging plane and the needle insertion plane are misaligned. Unlike many existing approaches that rely heavily on needle visibility in ultrasound images, our method uses a more robust feature by periodically vibrating the needle using a mechanical system. Specifically, we propose a vibration-based energy metric that remains effective even when the needle is fully out of plane. Using this metric, we develop a control strategy to reposition the ultrasound probe in response to misalignments between the imaging plane and the needle insertion plane in both translation and rotation. Experiments conducted on ex-vivo porcine tissue samples using a dual-arm robotic ultrasound-guided needle insertion system demonstrate the effectiveness of the proposed approach. The experimental results show the translational error of 0.41$\pm$0.27 mm and the rotational error of 0.51$\pm$0.19 degrees.
>
---
#### [new 038] Model Predictive Control for Crowd Navigation via Learning-Based Trajectory Prediction
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 论文提出基于学习的社交-隐含轨迹预测与MPC融合的群体导航方法，解决人群密集区域安全高效导航问题，通过降低轨迹误差并提升运动平滑性，验证了该框架在动态环境中的适应性。**

- **链接: [http://arxiv.org/pdf/2508.07079v1](http://arxiv.org/pdf/2508.07079v1)**

> **作者:** Mohamed Parvez Aslam; Bojan Derajic; Mohamed-Khalil Bouzidi; Sebastian Bernhard; Jan Oliver Ringert
>
> **摘要:** Safe navigation in pedestrian-rich environments remains a key challenge for autonomous robots. This work evaluates the integration of a deep learning-based Social-Implicit (SI) pedestrian trajectory predictor within a Model Predictive Control (MPC) framework on the physical Continental Corriere robot. Tested across varied pedestrian densities, the SI-MPC system is compared to a traditional Constant Velocity (CV) model in both open-loop prediction and closed-loop navigation. Results show that SI improves trajectory prediction - reducing errors by up to 76% in low-density settings - and enhances safety and motion smoothness in crowded scenes. Moreover, real-world deployment reveals discrepancies between open-loop metrics and closed-loop performance, as the SI model yields broader, more cautious predictions. These findings emphasize the importance of system-level evaluation and highlight the SI-MPC framework's promise for safer, more adaptive navigation in dynamic, human-populated environments.
>
---
#### [new 039] Imaginative World Modeling with Scene Graphs for Embodied Agent Navigation
- **分类: cs.RO**

- **简介: 论文提出一种基于场景图的想象建模框架SGImagineNav，用于自主导航任务，解决未知环境中快速定位目标的问题。通过符号世界建模与大语言模型预测，实现主动探索与语义上下文积累，提升导航成功率至65.4%。**

- **链接: [http://arxiv.org/pdf/2508.06990v1](http://arxiv.org/pdf/2508.06990v1)**

> **作者:** Yue Hu; Junzhe Wu; Ruihan Xu; Hang Liu; Avery Xi; Henry X. Liu; Ram Vasudevan; Maani Ghaffari
>
> **备注:** 23 pages
>
> **摘要:** Semantic navigation requires an agent to navigate toward a specified target in an unseen environment. Employing an imaginative navigation strategy that predicts future scenes before taking action, can empower the agent to find target faster. Inspired by this idea, we propose SGImagineNav, a novel imaginative navigation framework that leverages symbolic world modeling to proactively build a global environmental representation. SGImagineNav maintains an evolving hierarchical scene graphs and uses large language models to predict and explore unseen parts of the environment. While existing methods solely relying on past observations, this imaginative scene graph provides richer semantic context, enabling the agent to proactively estimate target locations. Building upon this, SGImagineNav adopts an adaptive navigation strategy that exploits semantic shortcuts when promising and explores unknown areas otherwise to gather additional context. This strategy continuously expands the known environment and accumulates valuable semantic contexts, ultimately guiding the agent toward the target. SGImagineNav is evaluated in both real-world scenarios and simulation benchmarks. SGImagineNav consistently outperforms previous methods, improving success rate to 65.4 and 66.8 on HM3D and HSSD, and demonstrating cross-floor and cross-room navigation in real-world environments, underscoring its effectiveness and generalizability.
>
---
#### [new 040] $\mathcal{P}^3$: Toward Versatile Embodied Agents
- **分类: cs.RO**

- **简介: 论文提出P³框架，解决具身智能代理在动态感知、工具灵活使用及多任务调度中的挑战，通过主动感知、无反馈工具插件和动态任务调度实现通用化部署。**

- **链接: [http://arxiv.org/pdf/2508.07033v1](http://arxiv.org/pdf/2508.07033v1)**

> **作者:** Shengli Zhou; Xiangchen Wang; Jinrui Zhang; Ruozai Tian; Rongtao Xu; Feng Zheng
>
> **备注:** 16 pages, 8 figures
>
> **摘要:** Embodied agents have shown promising generalization capabilities across diverse physical environments, making them essential for a wide range of real-world applications. However, building versatile embodied agents poses critical challenges due to three key issues: dynamic environment perception, open-ended tool usage, and complex multi-task planning. Most previous works rely solely on feedback from tool agents to perceive environmental changes and task status, which limits adaptability to real-time dynamics, causes error accumulation, and restricts tool flexibility. Furthermore, multi-task scheduling has received limited attention, primarily due to the inherent complexity of managing task dependencies and balancing competing priorities in dynamic and complex environments. To overcome these challenges, we introduce $\mathcal{P}^3$, a unified framework that integrates real-time perception and dynamic scheduling. Specifically, $\mathcal{P}^3$ enables 1) \textbf Perceive relevant task information actively from the environment, 2) \textbf Plug and utilize any tool without feedback requirement, and 3) \textbf Plan multi-task execution based on prioritizing urgent tasks and dynamically adjusting task order based on dependencies. Extensive real-world experiments show that our approach bridges the gap between benchmarks and practical deployment, delivering highly transferable, general-purpose embodied agents. Code and data will be released soon.
>
---
#### [new 041] A Hybrid Force-Position Strategy for Shape Control of Deformable Linear Objects With Graph Attention Networks
- **分类: cs.RO**

- **简介: 论文提出一种基于图注意力网络的混合力位移策略，用于解决可变形线性物体（DLO）形状控制中的复杂动力学问题，通过融合力与位置信息实现高效稳定控制。**

- **链接: [http://arxiv.org/pdf/2508.07319v1](http://arxiv.org/pdf/2508.07319v1)**

> **作者:** Yanzhao Yu; Haotian Yang; Junbo Tan; Xueqian Wang
>
> **摘要:** Manipulating deformable linear objects (DLOs) such as wires and cables is crucial in various applications like electronics assembly and medical surgeries. However, it faces challenges due to DLOs' infinite degrees of freedom, complex nonlinear dynamics, and the underactuated nature of the system. To address these issues, this paper proposes a hybrid force-position strategy for DLO shape control. The framework, combining both force and position representations of DLO, integrates state trajectory planning in the force space and Model Predictive Control (MPC) in the position space. We present a dynamics model with an explicit action encoder, a property extractor and a graph processor based on Graph Attention Networks. The model is used in the MPC to enhance prediction accuracy. Results from both simulations and real-world experiments demonstrate the effectiveness of our approach in achieving efficient and stable shape control of DLOs. Codes and videos are available at https://sites.google.com/view/dlom.
>
---
#### [new 042] BeyondMimic: From Motion Tracking to Versatile Humanoid Control via Guided Diffusion
- **分类: cs.RO**

- **简介: 论文提出BeyondMimic框架，通过引导扩散学习人类动作，解决运动追踪质量差与蒸馏不足问题，实现零样本灵活控制及多样化任务。**

- **链接: [http://arxiv.org/pdf/2508.08241v1](http://arxiv.org/pdf/2508.08241v1)**

> **作者:** Takara E. Truong; Qiayuan Liao; Xiaoyu Huang; Guy Tevet; C. Karen Liu; Koushil Sreenath
>
> **备注:** 9 pages, 1 figure
>
> **摘要:** Learning skills from human motions offers a promising path toward generalizable policies for whole-body humanoid control, yet two key cornerstones are missing: (1) a high-quality motion tracking framework that faithfully transforms large-scale kinematic references into robust and extremely dynamic motions on real hardware, and (2) a distillation approach that can effectively learn these motion primitives and compose them to solve downstream tasks. We address these gaps with BeyondMimic, the first real-world framework to learn from human motions for versatile and naturalistic humanoid control via guided diffusion. Our framework provides a motion tracking pipeline capable of challenging skills such as jumping spins, sprinting, and cartwheels with state-of-the-art motion quality. Moving beyond mimicking existing motions and synthesize novel ones, we further introduce a unified diffusion policy that enables zero-shot task-specific control at test time using simple cost functions. Deployed on hardware, BeyondMimic performs diverse tasks at test time, including waypoint navigation, joystick teleoperation, and obstacle avoidance, bridging sim-to-real motion tracking and flexible synthesis of human motion primitives for whole-body control. https://beyondmimic.github.io/.
>
---
#### [new 043] Stinger Robot: A Self-Bracing Robotic Platform for Autonomous Drilling in Confined Underground Environments
- **分类: cs.RO; 68T40, 93C85, 70E60**

- **简介: 论文提出一种自锁三腿结构的自主钻探机器人，解决受限地下环境中高力钻探难题，通过力感知闭环控制与ROS 2有限状态机实现动态腿部署，验证其在无外部支持下自主稳定钻探能力。**

- **链接: [http://arxiv.org/pdf/2508.06521v1](http://arxiv.org/pdf/2508.06521v1)**

> **作者:** H. Liu; L. S. Moreu; T. S. Andersen; V. V. Puche; M. Fumagalli
>
> **备注:** 7 pages, submitted
>
> **摘要:** The increasing demand for critical raw materials has revitalized interest in abandoned underground mines, which pose extreme challenges for conventional drilling machinery due to confined, unstructured, and infrastructure-less environments. This paper presents the Stinger Robot, a novel compact robotic platform specifically designed for autonomous high-force drilling in such settings. The robot features a mechanically self-locking tri-leg bracing mechanism that enables stable anchoring to irregular tunnel surfaces. A key innovation lies in its force-aware, closed-loop control strategy, which enables force interaction with unstructured environments during bracing and drilling. Implemented as a finite-state machine in ROS 2, the control policy dynamically adapts leg deployment based on real-time contact feedback and load thresholds, ensuring stability without external supports. We demonstrate, through simulation and preliminary hardware tests, that the Stinger Robot can autonomously stabilize and drill in conditions previously inaccessible to nowadays mining machines. This work constitutes the first validated robotic architecture to integrate distributed force-bracing and autonomous drilling in underground environments, laying the groundwork for future collaborative mining operations using modular robot systems.
>
---
#### [new 044] AgriVLN: Vision-and-Language Navigation for Agricultural Robots
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 论文提出AgriVLN框架，针对农业机器人导航中指令执行困难的问题，构建A2A基准数据集，结合视觉-语言模型与子任务分解模块，提升长指令下的导航成功率至0.47，实现农业场景下的高效自主导航。**

- **链接: [http://arxiv.org/pdf/2508.07406v1](http://arxiv.org/pdf/2508.07406v1)**

> **作者:** Xiaobei Zhao; Xingqi Lyu; Xiang Li
>
> **摘要:** Agricultural robots have emerged as powerful members in agricultural tasks, nevertheless, still heavily rely on manual operation or untransportable railway for movement, resulting in limited mobility and poor adaptability. Vision-and-Language Navigation (VLN) enables robots to navigate to the target destinations following natural language instructions, demonstrating strong performance on several domains. However, none of the existing benchmarks or methods is specifically designed for agricultural scenes. To bridge this gap, we propose Agriculture to Agriculture (A2A) benchmark, containing 1,560 episodes across six diverse agricultural scenes, in which all realistic RGB videos are captured by front-facing camera on a quadruped robot at a height of 0.38 meters, aligning with the practical deployment conditions. Meanwhile, we propose Vision-and-Language Navigation for Agricultural Robots (AgriVLN) baseline based on Vision-Language Model (VLM) prompted with carefully crafted templates, which can understand both given instructions and agricultural environments to generate appropriate low-level actions for robot control. When evaluated on A2A, AgriVLN performs well on short instructions but struggles with long instructions, because it often fails to track which part of the instruction is currently being executed. To address this, we further propose Subtask List (STL) instruction decomposition module and integrate it into AgriVLN, improving Success Rate (SR) from 0.33 to 0.47. We additionally compare AgriVLN with several existing VLN methods, demonstrating the state-of-the-art performance in the agricultural domain.
>
---
#### [new 045] AquaChat++: LLM-Assisted Multi-ROV Inspection for Aquaculture Net Pens with Integrated Battery Management and Thruster Fault Tolerance
- **分类: cs.RO**

- **简介: 论文提出基于LLM的多ROV协同检查框架AquaChat++，解决传统方法在实时性、能耗和动态环境适应性不足的问题，通过两层架构实现任务规划、调度与故障容错控制，集成电池管理与舵机故障补偿，提升水产养殖网箱检查效率与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.06554v1](http://arxiv.org/pdf/2508.06554v1)**

> **作者:** Abdelhaleem Saad; Waseem Akram; Irfan Hussain
>
> **摘要:** Inspection of aquaculture net pens is essential for ensuring the structural integrity and sustainable operation of offshore fish farming systems. Traditional methods, typically based on manually operated or single-ROV systems, offer limited adaptability to real-time constraints such as energy consumption, hardware faults, and dynamic underwater conditions. This paper introduces AquaChat++, a novel multi-ROV inspection framework that uses Large Language Models (LLMs) to enable adaptive mission planning, coordinated task execution, and fault-tolerant control in complex aquaculture environments. The proposed system consists of a two-layered architecture. The high-level plan generation layer employs an LLM, such as ChatGPT-4, to translate natural language user commands into symbolic, multi-agent inspection plans. A task manager dynamically allocates and schedules actions among ROVs based on their real-time status and operational constraints, including thruster faults and battery levels. The low-level control layer ensures accurate trajectory tracking and integrates thruster fault detection and compensation mechanisms. By incorporating real-time feedback and event-triggered replanning, AquaChat++ enhances system robustness and operational efficiency. Simulated experiments in a physics-based aquaculture environment demonstrate improved inspection coverage, energy-efficient behavior, and resilience to actuator failures. These findings highlight the potential of LLM-driven frameworks to support scalable, intelligent, and autonomous underwater robotic operations within the aquaculture sector.
>
---
#### [new 046] Robust and Agile Quadrotor Flight via Adaptive Unwinding-Free Quaternion Sliding Mode Control
- **分类: cs.RO**

- **简介: 论文提出一种自适应滑模控制框架，针对传统四旋翼飞行控制中的收敛慢、未缠绕、增益溢出等问题，通过非光滑稳定性分析实现鲁棒与敏捷飞行，控制器在资源受限设备上运行稳定，实现实验室与实际场景下的高性能控制。**

- **链接: [http://arxiv.org/pdf/2508.06568v1](http://arxiv.org/pdf/2508.06568v1)**

> **作者:** Amin Yazdanshenas; Reza Faieghi
>
> **摘要:** This paper presents a new adaptive sliding mode control (SMC) framework for quadrotors that achieves robust and agile flight under tight computational constraints. The proposed controller addresses key limitations of prior SMC formulations, including (i) the slow convergence and almost-global stability of $\mathrm{SO(3)}$-based methods, (ii) the oversimplification of rotational dynamics in Euler-based controllers, (iii) the unwinding phenomenon in quaternion-based formulations, and (iv) the gain overgrowth problem in adaptive SMC schemes. Leveraging nonsmooth stability analysis, we provide rigorous global stability proofs for both the nonsmooth attitude sliding dynamics defined on $\mathbb{S}^3$ and the position sliding dynamics. Our controller is computationally efficient and runs reliably on a resource-constrained nano quadrotor, achieving 250 Hz and 500 Hz refresh rates for position and attitude control, respectively. In an extensive set of hardware experiments with over 130 flight trials, the proposed controller consistently outperforms three benchmark methods, demonstrating superior trajectory tracking accuracy and robustness with relatively low control effort. The controller enables aggressive maneuvers such as dynamic throw launches, flip maneuvers, and accelerations exceeding 3g, which is remarkable for a 32-gram nano quadrotor. These results highlight promising potential for real-world applications, particularly in scenarios requiring robust, high-performance flight control under significant external disturbances and tight computational constraints.
>
---
#### [new 047] Symbolic Learning of Interpretable Reduced-Order Models for Jumping Quadruped Robots
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 论文提出结合SINDy与物理先验的符号学习方法，构建可解释的低维跳跃四足机器人动力学模型，解决复杂非线性动态建模问题，提升控制精度。**

- **链接: [http://arxiv.org/pdf/2508.06538v1](http://arxiv.org/pdf/2508.06538v1)**

> **作者:** Gioele Buriani; Jingyue Liu; Maximilian Stölzle; Cosimo Della Santina; Jiatao Ding
>
> **备注:** 8 pages, under review
>
> **摘要:** Reduced-order models are essential for motion planning and control of quadruped robots, as they simplify complex dynamics while preserving critical behaviors. This paper introduces a novel methodology for deriving such interpretable dynamic models, specifically for jumping. We capture the high-dimensional, nonlinear jumping dynamics in a low-dimensional latent space by proposing a learning architecture combining Sparse Identification of Nonlinear Dynamics (SINDy) with physical structural priors on the jump dynamics. Our approach demonstrates superior accuracy to the traditional actuated Spring-loaded Inverted Pendulum (aSLIP) model and is validated through simulation and hardware experiments across different jumping strategies.
>
---
#### [new 048] Collision-Free Trajectory Planning and control of Robotic Manipulator using Energy-Based Artificial Potential Field (E-APF)
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 论文提出基于能量的势场方法（E-APF）解决机器人轨迹规划中的碰撞与振荡问题，融合位置/速度依赖势函数与混合优化器，实现动态环境下的高效率、平滑轨迹控制，仿真验证其有效性。**

- **链接: [http://arxiv.org/pdf/2508.07323v1](http://arxiv.org/pdf/2508.07323v1)**

> **作者:** Adeetya Uppal; Rakesh Kumar Sahoo; Manoranjan Sinha
>
> **摘要:** Robotic trajectory planning in dynamic and cluttered environments remains a critical challenge, particularly when striving for both time efficiency and motion smoothness under actuation constraints. Traditional path planner, such as Artificial Potential Field (APF), offer computational efficiency but suffer from local minima issue due to position-based potential field functions and oscillatory motion near the obstacles due to Newtonian mechanics. To address this limitation, an Energy-based Artificial Potential Field (APF) framework is proposed in this paper that integrates position and velocity-dependent potential functions. E-APF ensures dynamic adaptability and mitigates local minima, enabling uninterrupted progression toward the goal. The proposed framework integrates E-APF with a hybrid trajectory optimizer that jointly minimizes jerk and execution time under velocity and acceleration constraints, ensuring geometric smoothness and time efficiency. The entire framework is validated in simulation using the 7-degree-of-freedom Kinova Gen3 robotic manipulator. The results demonstrate collision-free, smooth, time-efficient, and oscillation-free trajectory in the presence of obstacles, highlighting the efficacy of the combined trajectory optimization and real-time obstacle avoidance approach. This work lays the foundation for future integration with reactive control strategies and physical hardware deployment in real-world manipulation tasks.
>
---
#### [new 049] Capsizing-Guided Trajectory Optimization for Autonomous Navigation with Rough Terrain
- **分类: cs.RO**

- **简介: 论文提出基于翻倒稳定性的轨迹优化方法，解决机器人在崎岖地形自主导航中的安全避障问题，通过分析机器人倾覆稳定性定义可通行方向并纳入约束，采用图算法生成鲁棒轨迹。**

- **链接: [http://arxiv.org/pdf/2508.08108v1](http://arxiv.org/pdf/2508.08108v1)**

> **作者:** Wei Zhang; Yinchuan Wang; Wangtao Lu; Pengyu Zhang; Xiang Zhang; Yue Wang; Chaoqun Wang
>
> **摘要:** It is a challenging task for ground robots to autonomously navigate in harsh environments due to the presence of non-trivial obstacles and uneven terrain. This requires trajectory planning that balances safety and efficiency. The primary challenge is to generate a feasible trajectory that prevents robot from tip-over while ensuring effective navigation. In this paper, we propose a capsizing-aware trajectory planner (CAP) to achieve trajectory planning on the uneven terrain. The tip-over stability of the robot on rough terrain is analyzed. Based on the tip-over stability, we define the traversable orientation, which indicates the safe range of robot orientations. This orientation is then incorporated into a capsizing-safety constraint for trajectory optimization. We employ a graph-based solver to compute a robust and feasible trajectory while adhering to the capsizing-safety constraint. Extensive simulation and real-world experiments validate the effectiveness and robustness of the proposed method. The results demonstrate that CAP outperforms existing state-of-the-art approaches, providing enhanced navigation performance on uneven terrains.
>
---
#### [new 050] In-situ Value-aligned Human-Robot Interactions with Physical Constraints
- **分类: cs.RO**

- **简介: 论文提出结合人类偏好与物理约束的框架，解决机器人在复杂场景中平衡任务执行与人类需求的问题，通过ICLHF实现任务规划。**

- **链接: [http://arxiv.org/pdf/2508.07606v1](http://arxiv.org/pdf/2508.07606v1)**

> **作者:** Hongtao Li; Ziyuan Jiao; Xiaofeng Liu; Hangxin Liu; Zilong Zheng
>
> **备注:** 8 pages, 7 figures
>
> **摘要:** Equipped with Large Language Models (LLMs), human-centered robots are now capable of performing a wide range of tasks that were previously deemed challenging or unattainable. However, merely completing tasks is insufficient for cognitive robots, who should learn and apply human preferences to future scenarios. In this work, we propose a framework that combines human preferences with physical constraints, requiring robots to complete tasks while considering both. Firstly, we developed a benchmark of everyday household activities, which are often evaluated based on specific preferences. We then introduced In-Context Learning from Human Feedback (ICLHF), where human feedback comes from direct instructions and adjustments made intentionally or unintentionally in daily life. Extensive sets of experiments, testing the ICLHF to generate task plans and balance physical constraints with preferences, have demonstrated the efficiency of our approach.
>
---
#### [new 051] GraphCoT-VLA: A 3D Spatial-Aware Reasoning Vision-Language-Action Model for Robotic Manipulation with Ambiguous Instructions
- **分类: cs.RO**

- **简介: 论文提出基于3D空间感知的视觉-语言-动作模型GraphCoT-VLA，解决机器人操作中模糊指令与未知环境的挑战，通过结构化推理模块与3D姿态-物体图实现三维交互建模，提升任务规划与控制效率，实验显示其性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.07650v1](http://arxiv.org/pdf/2508.07650v1)**

> **作者:** Helong Huang; Min Cen; Kai Tan; Xingyue Quan; Guowei Huang; Hong Zhang
>
> **备注:** 10 pages, 6 figures
>
> **摘要:** Vision-language-action models have emerged as a crucial paradigm in robotic manipulation. However, existing VLA models exhibit notable limitations in handling ambiguous language instructions and unknown environmental states. Furthermore, their perception is largely constrained to static two-dimensional observations, lacking the capability to model three-dimensional interactions between the robot and its environment. To address these challenges, this paper proposes GraphCoT-VLA, an efficient end-to-end model. To enhance the model's ability to interpret ambiguous instructions and improve task planning, we design a structured Chain-of-Thought reasoning module that integrates high-level task understanding and planning, failed task feedback, and low-level imaginative reasoning about future object positions and robot actions. Additionally, we construct a real-time updatable 3D Pose-Object graph, which captures the spatial configuration of robot joints and the topological relationships between objects in 3D space, enabling the model to better understand and manipulate their interactions. We further integrates a dropout hybrid reasoning strategy to achieve efficient control outputs. Experimental results across multiple real-world robotic tasks demonstrate that GraphCoT-VLA significantly outperforms existing methods in terms of task success rate and response speed, exhibiting strong generalization and robustness in open environments and under uncertain instructions.
>
---
#### [new 052] Multimodal Spiking Neural Network for Space Robotic Manipulation
- **分类: cs.RO**

- **简介: 本文提出基于脉冲神经网络的多模态框架，解决空间机器人自主操作与资源限制问题，通过融合几何、触觉和语义信息及强化学习提升性能。**

- **链接: [http://arxiv.org/pdf/2508.07287v1](http://arxiv.org/pdf/2508.07287v1)**

> **作者:** Liwen Zhang; Dong Zhou; Shibo Shao; Zihao Su; Guanghui Sun
>
> **摘要:** This paper presents a multimodal control framework based on spiking neural networks (SNNs) for robotic arms aboard space stations. It is designed to cope with the constraints of limited onboard resources while enabling autonomous manipulation and material transfer in space operations. By combining geometric states with tactile and semantic information, the framework strengthens environmental awareness and contributes to more robust control strategies. To guide the learning process progressively, a dual-channel, three-stage curriculum reinforcement learning (CRL) scheme is further integrated into the system. The framework was tested across a range of tasks including target approach, object grasping, and stable lifting with wall-mounted robotic arms, demonstrating reliable performance throughout. Experimental evaluations demonstrate that the proposed method consistently outperforms baseline approaches in both task success rate and energy efficiency. These findings highlight its suitability for real-world aerospace applications.
>
---
#### [new 053] Optimal Planning and Machine Learning for Responsive Tracking and Enhanced Forecasting of Wildfires using a Spacecraft Constellation
- **分类: cs.RO**

- **简介: 论文提出利用卫星星座与优化规划、机器学习实现野火实时追踪与精准预测，解决传统方法滞后、数据不足问题，通过CYGNSS数据生成高分辨率烧毁区域地图并集成到气象模型中，提升预测精度与响应速度。**

- **链接: [http://arxiv.org/pdf/2508.06687v1](http://arxiv.org/pdf/2508.06687v1)**

> **作者:** Sreeja Roy-Singh; Vinay Ravindra; Richard Levinson; Mahta Moghaddam; Jan Mandel; Adam Kochanski; Angel Farguell Caus; Kurtis Nelson; Samira Alkaee Taleghan; Archana Kannan; Amer Melebari
>
> **摘要:** We propose a novel concept of operations using optimal planning methods and machine learning (ML) to collect spaceborne data that is unprecedented for monitoring wildfires, process it to create new or enhanced products in the context of wildfire danger or spread monitoring, and assimilate them to improve existing, wildfire decision support tools delivered to firefighters within latency appropriate for time-critical applications. The concept is studied with respect to NASA's CYGNSS Mission, a constellation of passive microwave receivers that measure specular GNSS-R reflections despite clouds and smoke. Our planner uses a Mixed Integer Program formulation to schedule joint observation data collection and downlink for all satellites. Optimal solutions are found quickly that collect 98-100% of available observation opportunities. ML-based fire predictions that drive the planner objective are greater than 40% more correlated with ground truth than existing state-of-art. The presented case study on the TX Smokehouse Creek fire in 2024 and LA fires in 2025 represents the first high-resolution data collected by CYGNSS of active fires. Creation of Burnt Area Maps (BAM) using ML applied to the data during active fires and BAM assimilation into NASA's Weather Research and Forecasting Model using ML to broadcast fire spread are novel outcomes. BAM and CYGNSS obtained soil moisture are integrated for the first time into USGS fire danger maps. Inclusion of CYGNSS data in ML-based burn predictions boosts accuracy by 13%, and inclusion of high-resolution data boosts ML recall by another 15%. The proposed workflow has an expected latency of 6-30h, improving on the current delivery time of multiple days. All components in the proposed concept are shown to be computationally scalable and globally generalizable, with sustainability considerations such as edge efficiency and low latency on small devices.
>
---
#### [new 054] MetAdv: A Unified and Interactive Adversarial Testing Platform for Autonomous Driving
- **分类: cs.RO; cs.AI**

- **简介: 论文提出MetAdv平台，解决自动驾驶对抗鲁棒性评估难题，通过虚拟仿真与物理反馈融合，构建三层闭环环境，支持多任务算法验证及人机交互，提升安全性和可靠性。**

- **链接: [http://arxiv.org/pdf/2508.06534v1](http://arxiv.org/pdf/2508.06534v1)**

> **作者:** Aishan Liu; Jiakai Wang; Tianyuan Zhang; Hainan Li; Jiangfan Liu; Siyuan Liang; Yilong Ren; Xianglong Liu; Dacheng Tao
>
> **备注:** Accepted by ACM MM 2025 Demo/Videos track
>
> **摘要:** Evaluating and ensuring the adversarial robustness of autonomous driving (AD) systems is a critical and unresolved challenge. This paper introduces MetAdv, a novel adversarial testing platform that enables realistic, dynamic, and interactive evaluation by tightly integrating virtual simulation with physical vehicle feedback. At its core, MetAdv establishes a hybrid virtual-physical sandbox, within which we design a three-layer closed-loop testing environment with dynamic adversarial test evolution. This architecture facilitates end-to-end adversarial evaluation, ranging from high-level unified adversarial generation, through mid-level simulation-based interaction, to low-level execution on physical vehicles. Additionally, MetAdv supports a broad spectrum of AD tasks, algorithmic paradigms (e.g., modular deep learning pipelines, end-to-end learning, vision-language models). It supports flexible 3D vehicle modeling and seamless transitions between simulated and physical environments, with built-in compatibility for commercial platforms such as Apollo and Tesla. A key feature of MetAdv is its human-in-the-loop capability: besides flexible environmental configuration for more customized evaluation, it enables real-time capture of physiological signals and behavioral feedback from drivers, offering new insights into human-machine trust under adversarial conditions. We believe MetAdv can offer a scalable and unified framework for adversarial assessment, paving the way for safer AD.
>
---
#### [new 055] Bio-Inspired Topological Autonomous Navigation with Active Inference in Robotics
- **分类: cs.RO**

- **简介: 论文提出基于主动推理的生物启发机器人自主导航方法，解决传统方法需预训练或静态假设的局限，实现动态环境下的拓扑地图构建与目标探索，具备鲁棒适应性和模块化架构。**

- **链接: [http://arxiv.org/pdf/2508.07267v1](http://arxiv.org/pdf/2508.07267v1)**

> **作者:** Daria de Tinguy; Tim Verbelen; Emilio Gamba; Bart Dhoedt
>
> **备注:** Conference ICCAS 2025 - accepted (in processing)
>
> **摘要:** Achieving fully autonomous exploration and navigation remains a critical challenge in robotics, requiring integrated solutions for localisation, mapping, decision-making and motion planning. Existing approaches either rely on strict navigation rules lacking adaptability or on pre-training, which requires large datasets. These AI methods are often computationally intensive or based on static assumptions, limiting their adaptability in dynamic or unknown environments. This paper introduces a bio-inspired agent based on the Active Inference Framework (AIF), which unifies mapping, localisation, and adaptive decision-making for autonomous navigation, including exploration and goal-reaching. Our model creates and updates a topological map of the environment in real-time, planning goal-directed trajectories to explore or reach objectives without requiring pre-training. Key contributions include a probabilistic reasoning framework for interpretable navigation, robust adaptability to dynamic changes, and a modular ROS2 architecture compatible with existing navigation systems. Our method was tested in simulated and real-world environments. The agent successfully explores large-scale simulated environments and adapts to dynamic obstacles and drift, proving to be comparable to other exploration strategies such as Gbplanner, FAEL and Frontiers. This approach offers a scalable and transparent approach for navigating complex, unstructured environments.
>
---
#### [new 056] Manipulator for people with limited abilities
- **分类: cs.RO**

- **简介: 论文任务：设计四自由度机械手以辅助残疾人。  
问题：解决残疾人操作困难。  
工作：开发机械结构、控制系统及ROS集成技术。**

- **链接: [http://arxiv.org/pdf/2508.06969v1](http://arxiv.org/pdf/2508.06969v1)**

> **作者:** Bingkun Huang; Evgeniy Kotov; Arkady Yuschenko
>
> **备注:** 105 pages, in Russian language
>
> **摘要:** The topic of this final qualification work was chosen due to the importance of developing robotic systems designed to assist people with disabilities. Advances in robotics and automation technologies have opened up new prospects for creating devices that can significantly improve the quality of life for these people. In this context, designing a robotic hand with a control system adapted to the needs of people with disabilities is a major scientific and practical challenge. This work addresses the problem of developing and manufacturing a four-degree-of-freedom robotic hand suitable for practical manipulation. Addressing this issue requires a comprehensive approach, encompassing the design of the hand's mechanical structure, the development of its control system, and its integration with a technical vision system and software based on the Robot Operating System (ROS).
>
---
#### [new 057] COMponent-Aware Pruning for Accelerated Control Tasks in Latent Space Models
- **分类: cs.RO; cs.AI; cs.SY; eess.SY**

- **简介: 论文提出基于组件感知的结构化剪枝方法，解决资源受限环境下神经网络控制器的计算复杂性问题，通过Lyapunov准则保障稳定性，建立安全压缩边界，实现高效控制任务部署。**

- **链接: [http://arxiv.org/pdf/2508.08144v1](http://arxiv.org/pdf/2508.08144v1)**

> **作者:** Ganesh Sundaram; Jonas Ulmen; Amjad Haider; Daniel Görges
>
> **备注:** Submitted in: The 2026 IEEE/SICE International Symposium on System Integration (SII 2026)
>
> **摘要:** The rapid growth of resource-constrained mobile platforms, including mobile robots, wearable systems, and Internet-of-Things devices, has increased the demand for computationally efficient neural network controllers (NNCs) that can operate within strict hardware limitations. While deep neural networks (DNNs) demonstrate superior performance in control applications, their substantial computational complexity and memory requirements present significant barriers to practical deployment on edge devices. This paper introduces a comprehensive model compression methodology that leverages component-aware structured pruning to determine the optimal pruning magnitude for each pruning group, ensuring a balance between compression and stability for NNC deployment. Our approach is rigorously evaluated on Temporal Difference Model Predictive Control (TD-MPC), a state-of-the-art model-based reinforcement learning algorithm, with a systematic integration of mathematical stability guarantee properties, specifically Lyapunov criteria. The key contribution of this work lies in providing a principled framework for determining the theoretical limits of model compression while preserving controller stability. Experimental validation demonstrates that our methodology successfully reduces model complexity while maintaining requisite control performance and stability characteristics. Furthermore, our approach establishes a quantitative boundary for safe compression ratios, enabling practitioners to systematically determine the maximum permissible model reduction before violating critical stability properties, thereby facilitating the confident deployment of compressed NNCs in resource-limited environments.
>
---
#### [new 058] D3P: Dynamic Denoising Diffusion Policy via Reinforcement Learning
- **分类: cs.RO**

- **简介: 论文提出D3P，一种基于强化学习的动态去噪扩散策略，针对机器人视觉运动任务中固定去噪步骤导致的实时性瓶颈，通过状态感知适配器动态分配去噪步骤，实现任务性能与推理效率的平衡。**

- **链接: [http://arxiv.org/pdf/2508.06804v1](http://arxiv.org/pdf/2508.06804v1)**

> **作者:** Shu-Ang Yu; Feng Gao; Yi Wu; Chao Yu; Yu Wang
>
> **摘要:** Diffusion policies excel at learning complex action distributions for robotic visuomotor tasks, yet their iterative denoising process poses a major bottleneck for real-time deployment. Existing acceleration methods apply a fixed number of denoising steps per action, implicitly treating all actions as equally important. However, our experiments reveal that robotic tasks often contain a mix of \emph{crucial} and \emph{routine} actions, which differ in their impact on task success. Motivated by this finding, we propose \textbf{D}ynamic \textbf{D}enoising \textbf{D}iffusion \textbf{P}olicy \textbf{(D3P)}, a diffusion-based policy that adaptively allocates denoising steps across actions at test time. D3P uses a lightweight, state-aware adaptor to allocate the optimal number of denoising steps for each action. We jointly optimize the adaptor and base diffusion policy via reinforcement learning to balance task performance and inference efficiency. On simulated tasks, D3P achieves an averaged 2.2$\times$ inference speed-up over baselines without degrading success. Furthermore, we demonstrate D3P's effectiveness on a physical robot, achieving a 1.9$\times$ acceleration over the baseline.
>
---
#### [new 059] SwarmVLM: VLM-Guided Impedance Control for Autonomous Navigation of Heterogeneous Robots in Dynamic Warehousing
- **分类: cs.RO**

- **简介: 论文提出基于VLM和RAG的阻抗控制方法，解决异构机器人动态仓储协作与避障问题，实现92%成功率。**

- **链接: [http://arxiv.org/pdf/2508.07814v1](http://arxiv.org/pdf/2508.07814v1)**

> **作者:** Malaika Zafar; Roohan Ahmed Khan; Faryal Batool; Yasheerah Yaqoot; Ziang Guo; Mikhail Litvinov; Aleksey Fedoseev; Dzmitry Tsetserukou
>
> **摘要:** With the growing demand for efficient logistics, unmanned aerial vehicles (UAVs) are increasingly being paired with automated guided vehicles (AGVs). While UAVs offer the ability to navigate through dense environments and varying altitudes, they are limited by battery life, payload capacity, and flight duration, necessitating coordinated ground support. Focusing on heterogeneous navigation, SwarmVLM addresses these limitations by enabling semantic collaboration between UAVs and ground robots through impedance control. The system leverages the Vision Language Model (VLM) and the Retrieval-Augmented Generation (RAG) to adjust impedance control parameters in response to environmental changes. In this framework, the UAV acts as a leader using Artificial Potential Field (APF) planning for real-time navigation, while the ground robot follows via virtual impedance links with adaptive link topology to avoid collisions with short obstacles. The system demonstrated a 92% success rate across 12 real-world trials. Under optimal lighting conditions, the VLM-RAG framework achieved 8% accuracy in object detection and selection of impedance parameters. The mobile robot prioritized short obstacle avoidance, occasionally resulting in a lateral deviation of up to 50 cm from the UAV path, which showcases safe navigation in a cluttered setting.
>
---
#### [new 060] AimBot: A Simple Auxiliary Visual Cue to Enhance Spatial Awareness of Visuomotor Policies
- **分类: cs.RO**

- **简介: 论文提出AimBot，通过在RGB图像叠加射线和瞄准镜提供空间引导，提升视觉运动策略的时空感知，实现低开销的辅助视觉反馈，有效改善性能。**

- **链接: [http://arxiv.org/pdf/2508.08113v1](http://arxiv.org/pdf/2508.08113v1)**

> **作者:** Yinpei Dai; Jayjun Lee; Yichi Zhang; Ziqiao Ma; Jed Yang; Amir Zadeh; Chuan Li; Nima Fazeli; Joyce Chai
>
> **备注:** CoRL 2025
>
> **摘要:** In this paper, we propose AimBot, a lightweight visual augmentation technique that provides explicit spatial cues to improve visuomotor policy learning in robotic manipulation. AimBot overlays shooting lines and scope reticles onto multi-view RGB images, offering auxiliary visual guidance that encodes the end-effector's state. The overlays are computed from depth images, camera extrinsics, and the current end-effector pose, explicitly conveying spatial relationships between the gripper and objects in the scene. AimBot incurs minimal computational overhead (less than 1 ms) and requires no changes to model architectures, as it simply replaces original RGB images with augmented counterparts. Despite its simplicity, our results show that AimBot consistently improves the performance of various visuomotor policies in both simulation and real-world settings, highlighting the benefits of spatially grounded visual feedback.
>
---
#### [new 061] PANAMA: A Network-Aware MARL Framework for Multi-Agent Path Finding in Digital Twin Ecosystems
- **分类: cs.LG; cs.AI; cs.DC; cs.MA; cs.RO**

- **简介: 论文提出PANAMA框架，针对数字孪生生态中的多智能体路径规划问题，通过网络感知的MARL算法解决数据共享与动态适应难题，采用CTDE架构实现高效训练与自主执行，提升路径规划精度、速度与扩展性。**

- **链接: [http://arxiv.org/pdf/2508.06767v1](http://arxiv.org/pdf/2508.06767v1)**

> **作者:** Arman Dogru; R. Irem Bor-Yaliniz; Nimal Gamini Senarath
>
> **摘要:** Digital Twins (DTs) are transforming industries through advanced data processing and analysis, positioning the world of DTs, Digital World, as a cornerstone of nextgeneration technologies including embodied AI. As robotics and automated systems scale, efficient data-sharing frameworks and robust algorithms become critical. We explore the pivotal role of data handling in next-gen networks, focusing on dynamics between application and network providers (AP/NP) in DT ecosystems. We introduce PANAMA, a novel algorithm with Priority Asymmetry for Network Aware Multi-agent Reinforcement Learning (MARL) based multi-agent path finding (MAPF). By adopting a Centralized Training with Decentralized Execution (CTDE) framework and asynchronous actor-learner architectures, PANAMA accelerates training while enabling autonomous task execution by embodied AI. Our approach demonstrates superior pathfinding performance in accuracy, speed, and scalability compared to existing benchmarks. Through simulations, we highlight optimized data-sharing strategies for scalable, automated systems, ensuring resilience in complex, real-world environments. PANAMA bridges the gap between network-aware decision-making and robust multi-agent coordination, advancing the synergy between DTs, wireless networks, and AI-driven automation.
>
---
#### [new 062] From Imitation to Optimization: A Comparative Study of Offline Learning for Autonomous Driving
- **分类: cs.LG; cs.AI; cs.RO; cs.SY; eess.SY**

- **简介: 论文探讨自主驾驶中从模仿学习到优化学习的转变，针对模仿学习的脆弱性提出基于Offline RL的解决方案，通过结构化状态表示与保守Q学习（CQL）提升鲁棒性，实现高成功率与低碰撞率。**

- **链接: [http://arxiv.org/pdf/2508.07029v1](http://arxiv.org/pdf/2508.07029v1)**

> **作者:** Antonio Guillen-Perez
>
> **摘要:** Learning robust driving policies from large-scale, real-world datasets is a central challenge in autonomous driving, as online data collection is often unsafe and impractical. While Behavioral Cloning (BC) offers a straightforward approach to imitation learning, policies trained with BC are notoriously brittle and suffer from compounding errors in closed-loop execution. This work presents a comprehensive pipeline and a comparative study to address this limitation. We first develop a series of increasingly sophisticated BC baselines, culminating in a Transformer-based model that operates on a structured, entity-centric state representation. While this model achieves low imitation loss, we show that it still fails in long-horizon simulations. We then demonstrate that by applying a state-of-the-art Offline Reinforcement Learning algorithm, Conservative Q-Learning (CQL), to the same data and architecture, we can learn a significantly more robust policy. Using a carefully engineered reward function, the CQL agent learns a conservative value function that enables it to recover from minor errors and avoid out-of-distribution states. In a large-scale evaluation on 1,000 unseen scenarios from the Waymo Open Motion Dataset, our final CQL agent achieves a 3.2x higher success rate and a 7.4x lower collision rate than the strongest BC baseline, proving that an offline RL approach is critical for learning robust, long-horizon driving policies from static expert data.
>
---
#### [new 063] AR-VRM: Imitating Human Motions for Visual Robot Manipulation with Analogical Reasoning
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出AR-VRM，通过类比推理从人类动作视频中学习动作知识，解决视觉机器人操作数据不足问题，实现显式模仿人类动作。**

- **链接: [http://arxiv.org/pdf/2508.07626v1](http://arxiv.org/pdf/2508.07626v1)**

> **作者:** Dejie Yang; Zijing Zhao; Yang Liu
>
> **备注:** Accepted by ICCV2025
>
> **摘要:** Visual Robot Manipulation (VRM) aims to enable a robot to follow natural language instructions based on robot states and visual observations, and therefore requires costly multi-modal data. To compensate for the deficiency of robot data, existing approaches have employed vision-language pretraining with large-scale data. However, they either utilize web data that differs from robotic tasks, or train the model in an implicit way (e.g., predicting future frames at the pixel level), thus showing limited generalization ability under insufficient robot data. In this paper, we propose to learn from large-scale human action video datasets in an explicit way (i.e., imitating human actions from hand keypoints), introducing Visual Robot Manipulation with Analogical Reasoning (AR-VRM). To acquire action knowledge explicitly from human action videos, we propose a keypoint Vision-Language Model (VLM) pretraining scheme, enabling the VLM to learn human action knowledge and directly predict human hand keypoints. During fine-tuning on robot data, to facilitate the robotic arm in imitating the action patterns of human motions, we first retrieve human action videos that perform similar manipulation tasks and have similar historical observations , and then learn the Analogical Reasoning (AR) map between human hand keypoints and robot components. Taking advantage of focusing on action keypoints instead of irrelevant visual cues, our method achieves leading performance on the CALVIN benchmark {and real-world experiments}. In few-shot scenarios, our AR-VRM outperforms previous methods by large margins , underscoring the effectiveness of explicitly imitating human actions under data scarcity.
>
---
#### [new 064] Noise-Aware Generative Microscopic Traffic Simulation
- **分类: eess.SY; cs.AI; cs.MA; cs.RO; cs.SY**

- **简介: 论文提出基于噪声感知的生成式微观交通仿真方法，解决现有数据集噪声大、标准差低的问题，构建I-24 MOTION Scenario Dataset并引入噪声感知损失函数，提升模型对现实噪声的鲁棒性与真实感。**

- **链接: [http://arxiv.org/pdf/2508.07453v1](http://arxiv.org/pdf/2508.07453v1)**

> **作者:** Vindula Jayawardana; Catherine Tang; Junyi Ji; Jonah Philion; Xue Bin Peng; Cathy Wu
>
> **摘要:** Accurately modeling individual vehicle behavior in microscopic traffic simulation remains a key challenge in intelligent transportation systems, as it requires vehicles to realistically generate and respond to complex traffic phenomena such as phantom traffic jams. While traditional human driver simulation models offer computational tractability, they do so by abstracting away the very complexity that defines human driving. On the other hand, recent advances in infrastructure-mounted camera-based roadway sensing have enabled the extraction of vehicle trajectory data, presenting an opportunity to shift toward generative, agent-based models. Yet, a major bottleneck remains: most existing datasets are either overly sanitized or lack standardization, failing to reflect the noisy, imperfect nature of real-world sensing. Unlike data from vehicle-mounted sensors-which can mitigate sensing artifacts like occlusion through overlapping fields of view and sensor fusion-infrastructure-based sensors surface a messier, more practical view of challenges that traffic engineers encounter. To this end, we present the I-24 MOTION Scenario Dataset (I24-MSD)-a standardized, curated dataset designed to preserve a realistic level of sensor imperfection, embracing these errors as part of the learning problem rather than an obstacle to overcome purely from preprocessing. Drawing from noise-aware learning strategies in computer vision, we further adapt existing generative models in the autonomous driving community for I24-MSD with noise-aware loss functions. Our results show that such models not only outperform traditional baselines in realism but also benefit from explicitly engaging with, rather than suppressing, data imperfection. We view I24-MSD as a stepping stone toward a new generation of microscopic traffic simulation that embraces the real-world challenges and is better aligned with practical needs.
>
---
#### [new 065] ForeSight: Multi-View Streaming Joint Object Detection and Trajectory Forecasting
- **分类: cs.CV; cs.RO**

- **简介: 论文提出ForeSight框架，针对传统分离检测与预测导致的时空信息利用不足问题，通过多任务流式学习实现联合目标检测与轨迹预测，共享记忆提升信息传播效率，达到优于现有方法的性能。**

- **链接: [http://arxiv.org/pdf/2508.07089v1](http://arxiv.org/pdf/2508.07089v1)**

> **作者:** Sandro Papais; Letian Wang; Brian Cheong; Steven L. Waslander
>
> **备注:** Accepted to ICCV 2025
>
> **摘要:** We introduce ForeSight, a novel joint detection and forecasting framework for vision-based 3D perception in autonomous vehicles. Traditional approaches treat detection and forecasting as separate sequential tasks, limiting their ability to leverage temporal cues. ForeSight addresses this limitation with a multi-task streaming and bidirectional learning approach, allowing detection and forecasting to share query memory and propagate information seamlessly. The forecast-aware detection transformer enhances spatial reasoning by integrating trajectory predictions from a multiple hypothesis forecast memory queue, while the streaming forecast transformer improves temporal consistency using past forecasts and refined detections. Unlike tracking-based methods, ForeSight eliminates the need for explicit object association, reducing error propagation with a tracking-free model that efficiently scales across multi-frame sequences. Experiments on the nuScenes dataset show that ForeSight achieves state-of-the-art performance, achieving an EPA of 54.9%, surpassing previous methods by 9.3%, while also attaining the best mAP and minADE among multi-view detection and forecasting models.
>
---
#### [new 066] Historical Prediction Attention Mechanism based Trajectory Forecasting for Proactive Work Zone Safety in a Digital Twin Environment
- **分类: cs.OH; cs.RO**

- **简介: 本论文提出一种基于历史预测注意力机制的数字孪生环境下的主动工作区安全预警系统，通过融合多传感器数据与HD地图，实现车辆轨迹预测与潜在冲突预警，有效降低工作区交通事故风险。**

- **链接: [http://arxiv.org/pdf/2508.06544v1](http://arxiv.org/pdf/2508.06544v1)**

> **作者:** Minhaj Uddin Ahmad; Mizanur Rahman; Alican Sevim; David Bodoh; Sakib Khan; Li Zhao; Nathan Huynh; Eren Erman Ozguven
>
> **摘要:** Proactive safety systems aim to mitigate risks by anticipating potential conflicts between vehicles and enabling early intervention to prevent work zone-related crashes. This study presents an infrastructure-enabled proactive work zone safety warning system that leverages a Digital Twin environment, integrating real-time multi-sensor data, detailed High-Definition (HD) maps, and a historical prediction attention mechanism-based trajectory prediction model. Using a co-simulation environment that combines Simulation of Urban MObility (SUMO) and CAR Learning to Act (CARLA) simulators, along with Lanelet2 HD maps and the Historical Prediction Network (HPNet) model, we demonstrate effective trajectory prediction and early warning generation for vehicle interactions in freeway work zones. To evaluate the accuracy of predicted trajectories, we use two standard metrics: Joint Average Displacement Error (ADE) and Joint Final Displacement Error (FDE). Specifically, the infrastructure-enabled HPNet model demonstrates superior performance on the work-zone datasets generated from the co-simulation environment, achieving a minimum Joint FDE of 0.3228 meters and a minimum Joint ADE of 0.1327 meters, lower than the benchmarks on the Argoverse (minJointFDE: 1.0986 m, minJointADE: 0.7612 m) and Interaction (minJointFDE: 0.8231 m, minJointADE: 0.2548 m) datasets. In addition, our proactive safety warning generation application, utilizing vehicle bounding boxes and probabilistic conflict modeling, demonstrates its capability to issue alerts for potential vehicle conflicts.
>
---
#### [new 067] Emergent morphogenesis via planar fabrication enabled by a reduced model of composites
- **分类: cs.GR; cs.RO**

- **简介: 论文提出基于简化复合材料模型的平面制造方法，通过耦合拉伸与弯曲力学实现3D形态生成，解决传统多层建模复杂度高的问题，实验验证了可重复的三维结构制造技术。**

- **链接: [http://arxiv.org/pdf/2508.08198v1](http://arxiv.org/pdf/2508.08198v1)**

> **作者:** Yupeng Zhang; Adam Alon; M. Khalid Jawed
>
> **备注:** GitHub repository: https://github.com/StructuresComp/discrete-shells-shrinky-dink/
>
> **摘要:** The ability to engineer complex three-dimensional shapes from planar sheets with precise, programmable control underpins emerging technologies in soft robotics, reconfigurable devices, and functional materials. Here, we present a reduced-order numerical and experimental framework for a bilayer system consisting of a stimuli-responsive thermoplastic sheet (Shrinky Dink) bonded to a kirigami-patterned, inert plastic layer. Upon uniform heating, the active layer contracts while the patterned layer constrains in-plane stretch but allows out-of-plane bending, yielding programmable 3D morphologies from simple planar precursors. Our approach enables efficient computational design and scalable manufacturing of 3D forms with a single-layer reduced model that captures the coupled mechanics of stretching and bending. Unlike traditional bilayer modeling, our framework collapses the multilayer composite into a single layer of nodes and elements, reducing the degrees of freedom and enabling simulation on a 2D geometry. This is achieved by introducing a novel energy formulation that captures the coupling between in-plane stretch mismatch and out-of-plane bending - extending beyond simple isotropic linear elastic models. Experimentally, we establish a fully planar, repeatable fabrication protocol using a stimuli-responsive thermoplastic and a laser-cut inert plastic layer. The programmed strain mismatch drives an array of 3D morphologies, such as bowls, canoes, and flower petals, all verified by both simulation and physical prototypes.
>
---
#### [new 068] IRL-VLA: Training an Vision-Language-Action Policy via Reward World Model
- **分类: cs.AI; cs.CV; cs.RO**

- **简介: 论文提出IRL-VLA框架，通过逆向强化学习构建奖励世界模型，解决开环模仿与闭环仿真难题，实现视觉-语言-动作政策的高效训练，提升自动驾驶的安全性、舒适性和效率。**

- **链接: [http://arxiv.org/pdf/2508.06571v1](http://arxiv.org/pdf/2508.06571v1)**

> **作者:** Anqing Jiang; Yu Gao; Yiru Wang; Zhigang Sun; Shuo Wang; Yuwen Heng; Hao Sun; Shichen Tang; Lijuan Zhu; Jinhao Chai; Jijun Wang; Zichong Gu; Hao Jiang; Li Sun
>
> **备注:** 9 pagres, 2 figures
>
> **摘要:** Vision-Language-Action (VLA) models have demonstrated potential in autonomous driving. However, two critical challenges hinder their development: (1) Existing VLA architectures are typically based on imitation learning in open-loop setup which tends to capture the recorded behaviors in the dataset, leading to suboptimal and constrained performance, (2) Close-loop training relies heavily on high-fidelity sensor simulation, where domain gaps and computational inefficiencies pose significant barriers. In this paper, we introduce IRL-VLA, a novel close-loop Reinforcement Learning via \textbf{I}nverse \textbf{R}einforcement \textbf{L}earning reward world model with a self-built VLA approach. Our framework proceeds in a three-stage paradigm: In the first stage, we propose a VLA architecture and pretrain the VLA policy via imitation learning. In the second stage, we construct a lightweight reward world model via inverse reinforcement learning to enable efficient close-loop reward computation. To further enhance planning performance, finally, we design specialized reward world model guidence reinforcement learning via PPO(Proximal Policy Optimization) to effectively balance the safety incidents, comfortable driving, and traffic efficiency. Our approach achieves state-of-the-art performance in NAVSIM v2 end-to-end driving benchmark, 1st runner up in CVPR2025 Autonomous Grand Challenge. We hope that our framework will accelerate VLA research in close-loop autonomous driving.
>
---
#### [new 069] The 2D+ Dynamic Articulatory Model DYNARTmo: Tongue-Palate Contact Area Estimation
- **分类: cs.CL; cs.RO**

- **简介: 论文提出2D+动态模型DYNARTmo，通过三维硬腭建模估算舌面-硬腭接触面积，解决传统2D模型对三维结构的不足，采用半椭圆与余弦曲线建模，生成多视角可视化结果，支持语音教学与治疗。**

- **链接: [http://arxiv.org/pdf/2508.07262v1](http://arxiv.org/pdf/2508.07262v1)**

> **作者:** Bernd J. Kröger
>
> **备注:** 11 pages, 9 figures, 14 references; supplementary material: python source code
>
> **摘要:** This paper describes an extension of the two-dimensional dynamic articulatory model DYNARTmo by integrating an internal three-dimensional representation of the palatal dome to estimate tongue-palate contact areas from midsagittal tongue contours. Two alternative dome geometries - a half-ellipse and a cosine based profile - are implemented to model lateral curvature in the coronal plane. Using these geometries, lateral contact points are analytically computed for each anterior-posterior position, enabling the generation of electropalatography-like visualizations within the 2D+ framework. The enhanced model supports three synchronized views (sagittal, glottal, and palatal) for static and dynamic (animated) articulation displays, suitable for speech science education and speech therapy. Future work includes adding a facial (lip) view and implementing articulatory-to-acoustic synthesis to quantitatively evaluate model realism.
>
---
#### [new 070] Multi-view Normal and Distance Guidance Gaussian Splatting for Surface Reconstruction
- **分类: cs.CV; cs.RO**

- **简介: 论文提出多视图法线和距离引导的Gaussian splatting方法，解决多视图场景下几何深度不一致和法线偏差问题，通过距离重投影正则化和法线增强模块实现多视图对齐与一致性。**

- **链接: [http://arxiv.org/pdf/2508.07701v1](http://arxiv.org/pdf/2508.07701v1)**

> **作者:** Bo Jia; Yanan Guo; Ying Chang; Benkui Zhang; Ying Xie; Kangning Du; Lin Cao
>
> **备注:** This paper has been accepted by IROS 2025
>
> **摘要:** 3D Gaussian Splatting (3DGS) achieves remarkable results in the field of surface reconstruction. However, when Gaussian normal vectors are aligned within the single-view projection plane, while the geometry appears reasonable in the current view, biases may emerge upon switching to nearby views. To address the distance and global matching challenges in multi-view scenes, we design multi-view normal and distance-guided Gaussian splatting. This method achieves geometric depth unification and high-accuracy reconstruction by constraining nearby depth maps and aligning 3D normals. Specifically, for the reconstruction of small indoor and outdoor scenes, we propose a multi-view distance reprojection regularization module that achieves multi-view Gaussian alignment by computing the distance loss between two nearby views and the same Gaussian surface. Additionally, we develop a multi-view normal enhancement module, which ensures consistency across views by matching the normals of pixel points in nearby views and calculating the loss. Extensive experimental results demonstrate that our method outperforms the baseline in both quantitative and qualitative evaluations, significantly enhancing the surface reconstruction capability of 3DGS.
>
---
## 更新

#### [replaced 001] Exploring Video-Based Driver Activity Recognition under Noisy Labels
- **分类: cs.CV; cs.LG; cs.RO; eess.IV**

- **链接: [http://arxiv.org/pdf/2504.11966v2](http://arxiv.org/pdf/2504.11966v2)**

> **作者:** Linjuan Fan; Di Wen; Kunyu Peng; Kailun Yang; Jiaming Zhang; Ruiping Liu; Yufan Chen; Junwei Zheng; Jiamin Wu; Xudong Han; Rainer Stiefelhagen
>
> **备注:** Accepted to SMC 2025. The source code is available at https://github.com/ilonafan/DAR-noisy-labels
>
> **摘要:** As an open research topic in the field of deep learning, learning with noisy labels has attracted much attention and grown rapidly over the past ten years. Learning with label noise is crucial for driver distraction behavior recognition, as real-world video data often contains mislabeled samples, impacting model reliability and performance. However, label noise learning is barely explored in the driver activity recognition field. In this paper, we propose the first label noise learning approach for the driver activity recognition task. Based on the cluster assumption, we initially enable the model to learn clustering-friendly low-dimensional representations from given videos and assign the resultant embeddings into clusters. We subsequently perform co-refinement within each cluster to smooth the classifier outputs. Furthermore, we propose a flexible sample selection strategy that combines two selection criteria without relying on any hyperparameters to filter clean samples from the training dataset. We also incorporate a self-adaptive parameter into the sample selection process to enforce balancing across classes. A comprehensive variety of experiments on the public Drive&Act dataset for all granularity levels demonstrates the superior performance of our method in comparison with other label-denoising methods derived from the image classification field. The source code is available at https://github.com/ilonafan/DAR-noisy-labels.
>
---
#### [replaced 002] TextInPlace: Indoor Visual Place Recognition in Repetitive Structures with Scene Text Spotting and Verification
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.06501v2](http://arxiv.org/pdf/2503.06501v2)**

> **作者:** Huaqi Tao; Bingxi Liu; Calvin Chen; Tingjun Huang; He Li; Jinqiang Cui; Hong Zhang
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** Visual Place Recognition (VPR) is a crucial capability for long-term autonomous robots, enabling them to identify previously visited locations using visual information. However, existing methods remain limited in indoor settings due to the highly repetitive structures inherent in such environments. We observe that scene texts frequently appear in indoor spaces and can help distinguish visually similar but different places. This inspires us to propose TextInPlace, a simple yet effective VPR framework that integrates Scene Text Spotting (STS) to mitigate visual perceptual ambiguity in repetitive indoor environments. Specifically, TextInPlace adopts a dual-branch architecture within a local parameter sharing network. The VPR branch employs attention-based aggregation to extract global descriptors for coarse-grained retrieval, while the STS branch utilizes a bridging text spotter to detect and recognize scene texts. Finally, the discriminative texts are filtered to compute text similarity and re-rank the top-K retrieved images. To bridge the gap between current text-based repetitive indoor scene datasets and the typical scenarios encountered in robot navigation, we establish an indoor VPR benchmark dataset, called Maze-with-Text. Extensive experiments on both custom and public datasets demonstrate that TextInPlace achieves superior performance over existing methods that rely solely on appearance information. The dataset, code, and trained models are publicly available at https://github.com/HqiTao/TextInPlace.
>
---
#### [replaced 003] BonnBeetClouds3D: A Dataset Towards Point Cloud-based Organ-level Phenotyping of Sugar Beet Plants under Field Conditions
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2312.14706v2](http://arxiv.org/pdf/2312.14706v2)**

> **作者:** Elias Marks; Jonas Bömer; Federico Magistri; Anurag Sah; Jens Behley; Cyrill Stachniss
>
> **摘要:** Agricultural production is facing severe challenges in the next decades induced by climate change and the need for sustainability, reducing its impact on the environment. Advancements in field management through non-chemical weeding by robots in combination with monitoring of crops by autonomous unmanned aerial vehicles (UAVs) and breeding of novel and more resilient crop varieties are helpful to address these challenges. The analysis of plant traits, called phenotyping, is an essential activity in plant breeding, it however involves a great amount of manual labor. With this paper, we address the problem of automatic fine-grained organ-level geometric analysis needed for precision phenotyping. As the availability of real-world data in this domain is relatively scarce, we propose a novel dataset that was acquired using UAVs capturing high-resolution images of a real breeding trial containing 48 plant varieties and therefore covering great morphological and appearance diversity. This enables the development of approaches for autonomous phenotyping that generalize well to different varieties. Based on overlapping high-resolution images from multiple viewing angles, we compute photogrammetric dense point clouds and provide detailed and accurate point-wise labels for plants, leaves, and salient points as the tip and the base. Additionally, we include measurements of phenotypic traits performed by experts from the German Federal Plant Variety Office on the real plants, allowing the evaluation of new approaches not only on segmentation and keypoint detection but also directly on the downstream tasks. The provided labeled point clouds enable fine-grained plant analysis and support further progress in the development of automatic phenotyping approaches, but also enable further research in surface reconstruction, point cloud completion, and semantic interpretation of point clouds.
>
---
#### [replaced 004] CARP: Visuomotor Policy Learning via Coarse-to-Fine Autoregressive Prediction
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2412.06782v3](http://arxiv.org/pdf/2412.06782v3)**

> **作者:** Zhefei Gong; Pengxiang Ding; Shangke Lyu; Siteng Huang; Mingyang Sun; Wei Zhao; Zhaoxin Fan; Donglin Wang
>
> **摘要:** In robotic visuomotor policy learning, diffusion-based models have achieved significant success in improving the accuracy of action trajectory generation compared to traditional autoregressive models. However, they suffer from inefficiency due to multiple denoising steps and limited flexibility from complex constraints. In this paper, we introduce Coarse-to-Fine AutoRegressive Policy (CARP), a novel paradigm for visuomotor policy learning that redefines the autoregressive action generation process as a coarse-to-fine, next-scale approach. CARP decouples action generation into two stages: first, an action autoencoder learns multi-scale representations of the entire action sequence; then, a GPT-style transformer refines the sequence prediction through a coarse-to-fine autoregressive process. This straightforward and intuitive approach produces highly accurate and smooth actions, matching or even surpassing the performance of diffusion-based policies while maintaining efficiency on par with autoregressive policies. We conduct extensive evaluations across diverse settings, including single-task and multi-task scenarios on state-based and image-based simulation benchmarks, as well as real-world tasks. CARP achieves competitive success rates, with up to a 10% improvement, and delivers 10x faster inference compared to state-of-the-art policies, establishing a high-performance, efficient, and flexible paradigm for action generation in robotic tasks.
>
---
#### [replaced 005] Semantic Mapping in Indoor Embodied AI -- A Survey on Advances, Challenges, and Future Directions
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.05750v3](http://arxiv.org/pdf/2501.05750v3)**

> **作者:** Sonia Raychaudhuri; Angel X. Chang
>
> **摘要:** Intelligent embodied agents (e.g. robots) need to perform complex semantic tasks in unfamiliar environments. Among many skills that the agents need to possess, building and maintaining a semantic map of the environment is most crucial in long-horizon tasks. A semantic map captures information about the environment in a structured way, allowing the agent to reference it for advanced reasoning throughout the task. While existing surveys in embodied AI focus on general advancements or specific tasks like navigation and manipulation, this paper provides a comprehensive review of semantic map-building approaches in embodied AI, specifically for indoor navigation. We categorize these approaches based on their structural representation (spatial grids, topological graphs, dense point-clouds or hybrid maps) and the type of information they encode (implicit features or explicit environmental data). We also explore the strengths and limitations of the map building techniques, highlight current challenges, and propose future research directions. We identify that the field is moving towards developing open-vocabulary, queryable, task-agnostic map representations, while high memory demands and computational inefficiency still remaining to be open challenges. This survey aims to guide current and future researchers in advancing semantic mapping techniques for embodied AI systems.
>
---
#### [replaced 006] A Differentiated Reward Method for Reinforcement Learning based Multi-Vehicle Cooperative Decision-Making Algorithms
- **分类: cs.AI; cs.MA; cs.RO**

- **链接: [http://arxiv.org/pdf/2502.00352v3](http://arxiv.org/pdf/2502.00352v3)**

> **作者:** Ye Han; Lijun Zhang; Dejian Meng; Zhuang Zhang
>
> **备注:** 10 pages, 3 figures
>
> **摘要:** Reinforcement learning (RL) shows great potential for optimizing multi-vehicle cooperative driving strategies through the state-action-reward feedback loop, but it still faces challenges such as low sample efficiency. This paper proposes a differentiated reward method based on steady-state transition systems, which incorporates state transition gradient information into the reward design by analyzing traffic flow characteristics, aiming to optimize action selection and policy learning in multi-vehicle cooperative decision-making. The performance of the proposed method is validated in RL algorithms such as MAPPO, MADQN, and QMIX under varying autonomous vehicle penetration. The results show that the differentiated reward method significantly accelerates training convergence and outperforms centering reward and others in terms of traffic efficiency, safety, and action rationality. Additionally, the method demonstrates strong scalability and environmental adaptability, providing a novel approach for multi-agent cooperative decision-making in complex traffic scenarios.
>
---
#### [replaced 007] Interactive Imitation Learning for Dexterous Robotic Manipulation: Challenges and Perspectives -- A Survey
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2506.00098v2](http://arxiv.org/pdf/2506.00098v2)**

> **作者:** Edgar Welte; Rania Rayyes
>
> **备注:** 27 pages, 4 figures, 3 tables
>
> **摘要:** Dexterous manipulation is a crucial yet highly complex challenge in humanoid robotics, demanding precise, adaptable, and sample-efficient learning methods. As humanoid robots are usually designed to operate in human-centric environments and interact with everyday objects, mastering dexterous manipulation is critical for real-world deployment. Traditional approaches, such as reinforcement learning and imitation learning, have made significant strides, but they often struggle due to the unique challenges of real-world dexterous manipulation, including high-dimensional control, limited training data, and covariate shift. This survey provides a comprehensive overview of these challenges and reviews existing learning-based methods for real-world dexterous manipulation, spanning imitation learning, reinforcement learning, and hybrid approaches. A promising yet underexplored direction is interactive imitation learning, where human feedback actively refines a robots behavior during training. While interactive imitation learning has shown success in various robotic tasks, its application to dexterous manipulation remains limited. To address this gap, we examine current interactive imitation learning techniques applied to other robotic tasks and discuss how these methods can be adapted to enhance dexterous manipulation. By synthesizing state-of-the-art research, this paper highlights key challenges, identifies gaps in current methodologies, and outlines potential directions for leveraging interactive imitation learning to improve dexterous robotic skills.
>
---
#### [replaced 008] In-between Motion Generation Based Multi-Style Quadruped Robot Locomotion
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.23053v2](http://arxiv.org/pdf/2507.23053v2)**

> **作者:** Yuanhao Chen; Liu Zhao; Ji Ma; Peng Lu
>
> **摘要:** Quadruped robots face persistent challenges in achieving versatile locomotion due to limitations in reference motion data diversity. To address these challenges, we introduce an in-between motion generation based multi-style quadruped robot locomotion framework. We propose a CVAE based motion generator, synthesizing multi-style dynamically feasible locomotion sequences between arbitrary start and end states. By embedding physical constraints and leveraging joint poses based phase manifold continuity, this component produces physically plausible motions spanning multiple gait modalities while ensuring kinematic compatibility with robotic morphologies. We train the imitation policy based on generated data, which validates the effectiveness of generated motion data in enhancing controller stability and improving velocity tracking performance. The proposed framework demonstrates significant improvements in velocity tracking and deployment stability. We successfully deploy the framework on a real-world quadruped robot, and the experimental validation confirms the framework's capability to generate and execute complex motion profiles, including gallop, tripod, trotting and pacing.
>
---
#### [replaced 009] Schema-Guided Scene-Graph Reasoning based on Multi-Agent Large Language Model System
- **分类: cs.LG; cs.AI; cs.MA; cs.RO**

- **链接: [http://arxiv.org/pdf/2502.03450v2](http://arxiv.org/pdf/2502.03450v2)**

> **作者:** Yiye Chen; Harpreet Sawhney; Nicholas Gydé; Yanan Jian; Jack Saunders; Patricio Vela; Ben Lundell
>
> **备注:** In submission
>
> **摘要:** Scene graphs have emerged as a structured and serializable environment representation for grounded spatial reasoning with Large Language Models (LLMs). In this work, we propose SG^2, an iterative Schema-Guided Scene-Graph reasoning framework based on multi-agent LLMs. The agents are grouped into two modules: a (1) Reasoner module for abstract task planning and graph information queries generation, and a (2) Retriever module for extracting corresponding graph information based on code-writing following the queries. Two modules collaborate iteratively, enabling sequential reasoning and adaptive attention to graph information. The scene graph schema, prompted to both modules, serves to not only streamline both reasoning and retrieval process, but also guide the cooperation between two modules. This eliminates the need to prompt LLMs with full graph data, reducing the chance of hallucination due to irrelevant information. Through experiments in multiple simulation environments, we show that our framework surpasses existing LLM-based approaches and baseline single-agent, tool-based Reason-while-Retrieve strategy in numerical Q\&A and planning tasks.
>
---
#### [replaced 010] Elastic Motion Policy: An Adaptive Dynamical System for Robust and Efficient One-Shot Imitation Learning
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.08029v2](http://arxiv.org/pdf/2503.08029v2)**

> **作者:** Tianyu Li; Sunan Sun; Shubhodeep Shiv Aditya; Nadia Figueroa
>
> **摘要:** Behavior cloning (BC) has become a staple imitation learning paradigm in robotics due to its ease of teaching robots complex skills directly from expert demonstrations. However, BC suffers from an inherent generalization issue. To solve this, the status quo solution is to gather more data. Yet, regardless of how much training data is available, out-of-distribution performance is still sub-par, lacks any formal guarantee of convergence and success, and is incapable of allowing and recovering from physical interactions with humans. These are critical flaws when robots are deployed in ever-changing human-centric environments. Thus, we propose Elastic Motion Policy (EMP), a one-shot imitation learning framework that allows robots to adjust their behavior based on the scene change while respecting the task specification. Trained from a single demonstration, EMP follows the dynamical systems paradigm where motion planning and control are governed by first-order differential equations with convergence guarantees. We leverage Laplacian editing in full end-effector space, $\mathbb{R}^3\times SO(3)$, and online convex learning of Lyapunov functions, to adapt EMP online to new contexts, avoiding the need to collect new demonstrations. We extensively validate our framework in real robot experiments, demonstrating its robust and efficient performance in dynamic environments, with obstacle avoidance and multi-step task capabilities. Project Website: https://elastic-motion-policy.github.io/EMP/
>
---
#### [replaced 011] POEX: Towards Policy Executable Jailbreak Attacks Against the LLM-based Robots
- **分类: cs.RO; cs.AI; cs.CY**

- **链接: [http://arxiv.org/pdf/2412.16633v3](http://arxiv.org/pdf/2412.16633v3)**

> **作者:** Xuancun Lu; Zhengxian Huang; Xinfeng Li; Chi Zhang; Xiaoyu ji; Wenyuan Xu
>
> **备注:** Homepage: https://poex-jailbreak.github.io/
>
> **摘要:** The integration of LLMs into robots has witnessed significant growth, where LLMs can convert instructions into executable robot policies. However, the inherent vulnerability of LLMs to jailbreak attacks brings critical security risks from the digital domain to the physical world. An attacked LLM-based robot could execute harmful policies and cause physical harm. In this paper, we investigate the feasibility and rationale of jailbreak attacks against LLM-based robots and answer three research questions: (1) How applicable are existing LLM jailbreak attacks against LLM-based robots? (2) What unique challenges arise if they are not directly applicable? (3) How to defend against such jailbreak attacks? To this end, we first construct a "human-object-environment" robot risks-oriented Harmful-RLbench and then conduct a measurement study on LLM-based robot systems. Our findings conclude that traditional LLM jailbreak attacks are inapplicable in robot scenarios, and we identify two unique challenges: determining policy-executable optimization directions and accurately evaluating robot-jailbroken policies. To enable a more thorough security analysis, we introduce POEX (POlicy EXecutable) jailbreak, a red-teaming framework that induces harmful yet executable policy to jailbreak LLM-based robots. POEX incorporates hidden layer gradient optimization to guarantee jailbreak success and policy execution as well as a multi-agent evaluator to accurately assess the practical executability of policies. Experiments conducted on the real-world robotic systems and in simulation demonstrate the efficacy of POEX, highlighting critical security vulnerabilities and its transferability across LLMs. Finally, we propose prompt-based and model-based defenses to mitigate attacks. Our findings underscore the urgent need for security measures to ensure the safe deployment of LLM-based robots in critical applications.
>
---
#### [replaced 012] Exploring Spatial Representation to Enhance LLM Reasoning in Aerial Vision-Language Navigation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.08500v3](http://arxiv.org/pdf/2410.08500v3)**

> **作者:** Yunpeng Gao; Zhigang Wang; Pengfei Han; Linglin Jing; Dong Wang; Bin Zhao
>
> **摘要:** Aerial Vision-and-Language Navigation (VLN) is a novel task enabling Unmanned Aerial Vehicles (UAVs) to navigate in outdoor environments through natural language instructions and visual cues. However, it remains challenging due to the complex spatial relationships in aerial scenes.In this paper, we propose a training-free, zero-shot framework for aerial VLN tasks, where the large language model (LLM) is leveraged as the agent for action prediction. Specifically, we develop a novel Semantic-Topo-Metric Representation (STMR) to enhance the spatial reasoning capabilities of LLMs. This is achieved by extracting and projecting instruction-related semantic masks onto a top-down map, which presents spatial and topological information about surrounding landmarks and grows during the navigation process. At each step, a local map centered at the UAV is extracted from the growing top-down map, and transformed into a ma trix representation with distance metrics, serving as the text prompt to LLM for action prediction in response to the given instruction. Experiments conducted in real and simulation environments have proved the effectiveness and robustness of our method, achieving absolute success rate improvements of 26.8% and 5.8% over current state-of-the-art methods on simple and complex navigation tasks, respectively. The dataset and code will be released soon.
>
---
#### [replaced 013] Learning 3D-Gaussian Simulators from RGB Videos
- **分类: cs.GR; cs.AI; cs.CV; cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.24009v2](http://arxiv.org/pdf/2503.24009v2)**

> **作者:** Mikel Zhobro; Andreas René Geist; Georg Martius
>
> **摘要:** Realistic simulation is critical for applications ranging from robotics to animation. Learned simulators have emerged as a possibility to capture real world physics directly from video data, but very often require privileged information such as depth information, particle tracks and hand-engineered features to maintain spatial and temporal consistency. These strong inductive biases or ground truth 3D information help in domains where data is sparse but limit scalability and generalization in data rich regimes. To overcome the key limitations, we propose 3DGSim, a learned 3D simulator that directly learns physical interactions from multi-view RGB videos. 3DGSim unifies 3D scene reconstruction, particle dynamics prediction and video synthesis into an end-to-end trained framework. It adopts MVSplat to learn a latent particle-based representation of 3D scenes, a Point Transformer for particle dynamics, a Temporal Merging module for consistent temporal aggregation and Gaussian Splatting to produce novel view renderings. By jointly training inverse rendering and dynamics forecasting, 3DGSim embeds the physical properties into point-wise latent features. This enables the model to capture diverse physical behaviors, from rigid to elastic, cloth-like dynamics, and boundary conditions (e.g. fixed cloth corner), along with realistic lighting effects that also generalize to unseen multibody interactions and novel scene edits.
>
---
#### [replaced 014] DRAMA-X: A Fine-grained Intent Prediction and Risk Reasoning Benchmark For Driving
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.17590v2](http://arxiv.org/pdf/2506.17590v2)**

> **作者:** Mihir Godbole; Xiangbo Gao; Zhengzhong Tu
>
> **备注:** 19 pages, 5 figures, Preprint under review. Code available at: https://github.com/taco-group/DRAMA-X
>
> **摘要:** Understanding the short-term motion of vulnerable road users (VRUs) like pedestrians and cyclists is critical for safe autonomous driving, especially in urban scenarios with ambiguous or high-risk behaviors. While vision-language models (VLMs) have enabled open-vocabulary perception, their utility for fine-grained intent reasoning remains underexplored. Notably, no existing benchmark evaluates multi-class intent prediction in safety-critical situations, To address this gap, we introduce DRAMA-X, a fine-grained benchmark constructed from the DRAMA dataset via an automated annotation pipeline. DRAMA-X contains 5,686 accident-prone frames labeled with object bounding boxes, a nine-class directional intent taxonomy, binary risk scores, expert-generated action suggestions for the ego vehicle, and descriptive motion summaries. These annotations enable a structured evaluation of four interrelated tasks central to autonomous decision-making: object detection, intent prediction, risk assessment, and action suggestion. As a reference baseline, we propose SGG-Intent, a lightweight, training-free framework that mirrors the ego vehicle's reasoning pipeline. It sequentially generates a scene graph from visual input using VLM-backed detectors, infers intent, assesses risk, and recommends an action using a compositional reasoning stage powered by a large language model. We evaluate a range of recent VLMs, comparing performance across all four DRAMA-X tasks. Our experiments demonstrate that scene-graph-based reasoning enhances intent prediction and risk assessment, especially when contextual cues are explicitly modeled.
>
---
#### [replaced 015] MultiNash-PF: A Particle Filtering Approach for Computing Multiple Local Generalized Nash Equilibria in Trajectory Games
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.05554v3](http://arxiv.org/pdf/2410.05554v3)**

> **作者:** Maulik Bhatt; Iman Askari; Yue Yu; Ufuk Topcu; Huazhen Fang; Negar Mehr
>
> **摘要:** Modern robotic systems frequently engage in complex multi-agent interactions, many of which are inherently multi-modal, i.e., they can lead to multiple distinct outcomes. To interact effectively, robots must recognize the possible interaction modes and adapt to the one preferred by other agents. In this work, we propose MultiNash-PF, an efficient algorithm for capturing the multimodality in multi-agent interactions. We model interaction outcomes as equilibria of a game-theoretic planner, where each equilibrium corresponds to a distinct interaction mode. Our framework formulates interactive planning as Constrained Potential Trajectory Games (CPTGs), in which local Generalized Nash Equilibria (GNEs) represent plausible interaction outcomes. We propose to integrate the potential game approach with implicit particle filtering, a sample-efficient method for non-convex trajectory optimization. We utilize implicit particle filtering to identify the coarse estimates of multiple local minimizers of the game's potential function. MultiNash-PF then refines these estimates with optimization solvers, obtaining different local GNEs. We show through numerical simulations that MultiNash-PF reduces computation time by up to 50\% compared to a baseline. We further demonstrate the effectiveness of our algorithm in real-world human-robot interaction scenarios, where it successfully accounts for the multi-modal nature of interactions and resolves potential conflicts in real-time.
>
---
#### [replaced 016] MAT-DiSMech: A Discrete Differential Geometry-based Computational Tool for Simulation of Rods, Shells, and Soft Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.17186v2](http://arxiv.org/pdf/2504.17186v2)**

> **作者:** Radha Lahoti; M. Khalid Jawed
>
> **备注:** Total 31 pages, 12 figures, open-source code available at https://github.com/StructuresComp/dismech-matlab
>
> **摘要:** Accurate and efficient simulation tools are essential in robotics, enabling the visualization of system dynamics and the validation of control laws before committing resources to physical experimentation. Developing physically accurate simulation tools is particularly challenging in soft robotics, largely due to the prevalence of geometrically nonlinear deformation. A variety of robot simulators tackle this challenge by using simplified modeling techniques -- such as lumped mass models -- which lead to physical inaccuracies in real-world applications. On the other hand, high-fidelity simulation methods for soft structures, like finite element analysis, offer increased accuracy but lead to higher computational costs. In light of this, we present a Discrete Differential Geometry-based simulator that provides a balance between physical accuracy and computational speed. Building on an extensive body of research on rod and shell-based representations of soft robots, our tool provides a pathway to accurately model soft robots in a computationally tractable manner. Our open-source MATLAB-based framework is capable of simulating the deformations of rods, shells, and their combinations, primarily utilizing implicit integration techniques. The software design is modular for the user to customize the code, for example, add new external forces and impose custom boundary conditions. The implementations for prevalent forces encountered in robotics, including gravity, contact, kinetic and viscous friction, and aerodynamic drag, have been provided. We provide several illustrative examples that showcase the capabilities and validate the physical accuracy of the simulator. The open-source code is available at https://github.com/StructuresComp/dismech-matlab.git. We anticipate that the proposed simulator can serve as an effective digital twin tool, enhancing the Sim2Real pathway in soft robotics research.
>
---
#### [replaced 017] Industrial Robot Motion Planning with GPUs: Integration of cuRobo for Extended DOF Systems
- **分类: cs.RO; I.2.9; I.2.10; J.7**

- **链接: [http://arxiv.org/pdf/2508.04146v2](http://arxiv.org/pdf/2508.04146v2)**

> **作者:** Luai Abuelsamen; Harsh Rana; Ho-Wei Lu; Wenhan Tang; Swati Priyadarshini; Gabriel Gomes
>
> **备注:** 8 pages, 2 figures, 2 tables
>
> **摘要:** Efficient motion planning remains a key challenge in industrial robotics, especially for multi-axis systems operating in complex environments. This paper addresses that challenge by integrating GPU-accelerated motion planning through NVIDIA's cuRobo library into Vention's modular automation platform. By leveraging accurate CAD-based digital twins and real-time parallel optimization, our system enables rapid trajectory generation and dynamic collision avoidance for pick-and-place tasks. We demonstrate this capability on robots equipped with additional degrees of freedom, including a 7th-axis gantry, and benchmark performance across various scenarios. The results show significant improvements in planning speed and robustness, highlighting the potential of GPU-based planning pipelines for scalable, adaptable deployment in modern industrial workflows.
>
---
#### [replaced 018] Optimizing Design and Control Methods for Using Collaborative Robots in Upper-Limb Rehabilitation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2407.18661v3](http://arxiv.org/pdf/2407.18661v3)**

> **作者:** Dario Onfiani; Marco Caramaschi; Luigi Biagiotti; Fabio Pini
>
> **摘要:** In this paper, we address the development of a robotic rehabilitation system for the upper limbs based on collaborative end-effector solutions. The use of commercial collaborative robots offers significant advantages for this task, as they are optimized from an engineering perspective and ensure safe physical interaction with humans. However, they also come with noticeable drawbacks, such as the limited range of sizes available on the market and the standard control modes, which are primarily oriented towards industrial or service applications. To address these limitations, we propose an optimization-based design method to fully exploit the capability of the cobot in performing rehabilitation tasks. Additionally, we introduce a novel control architecture based on an admittance-type Virtual Fixture method, which constrains the motion of the robot along a prescribed path. This approach allows for an intuitive definition of the task to be performed via Programming by Demonstration and enables the system to operate both passively and actively. In passive mode, the system supports the patient during task execution with additional force, while in active mode, it opposes the motion with a braking force. Experimental results demonstrate the effectiveness of the proposed method.
>
---
#### [replaced 019] A Step-by-step Guide on Nonlinear Model Predictive Control for Safe Mobile Robot Navigation
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2507.17856v3](http://arxiv.org/pdf/2507.17856v3)**

> **作者:** Dennis Benders; Laura Ferranti; Johannes Köhler
>
> **备注:** 51 pages, 3 figures
>
> **摘要:** Designing a model predictive control (MPC) scheme that enables a mobile robot to safely navigate through an obstacle-filled environment is a complicated yet essential task in robotics. In this technical report, safety refers to ensuring that the robot respects state and input constraints while avoiding collisions with obstacles despite the presence of disturbances and measurement noise. This report offers a step-by-step approach to implementing nonlinear model predictive control (NMPC) schemes addressing these safety requirements. Numerous books and survey papers provide comprehensive overviews of linear MPC (LMPC), NMPC, and their applications in various domains, including robotics. This report does not aim to replicate those exhaustive reviews. Instead, it focuses specifically on NMPC as a foundation for safe mobile robot navigation. The goal is to provide a practical and accessible path from theoretical concepts to mathematical proofs and implementation, emphasizing safety and performance guarantees. It is intended for researchers, robotics engineers, and practitioners seeking to bridge the gap between theoretical NMPC formulations and real-world robotic applications. This report is not necessarily meant to remain fixed over time. If someone finds an error in the presented theory, please reach out via the given email addresses. We are happy to update the document if necessary.
>
---
#### [replaced 020] Vehicle Top Tag Assisted Vehicle-Road Cooperative Localization For Autonomous Public Buses
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.00546v2](http://arxiv.org/pdf/2503.00546v2)**

> **作者:** Hao Li; Bo Liu; Linbin Wang
>
> **摘要:** Accurate vehicle localization is indispensable to autonomous vehicles, but is difficult to realize in complicated application scenarios. Intersection scenarios that suffer from environmental shielding and crowded dynamic objects are especially crucial and challenging. To handle difficult intersection scenarios, the methodology of vehicle top tag assisted vehicle-road cooperative localization or for short vehicle top tag assisted localization is proposed. The proposed methodology has merits of satisfying all the feasibility, reliability, explainability, society and economy concerns. Concrete solutions of vehicle top tag detection and vehicle top tag localization that instantiate the core part of the proposed methodology are presented. Simulation results are provided to demonstrate effectiveness of the presented solutions. The proposed methodology of vehicle top tag assisted localization also has the potential to be extended to a much wider range of practical applications than our intended ones involving autonomous public buses.
>
---
#### [replaced 021] Is Single-View Mesh Reconstruction Ready for Robotics?
- **分类: cs.RO; cs.CV; I.4.5; I.4.8; I.2.9; I.2.10**

- **链接: [http://arxiv.org/pdf/2505.17966v2](http://arxiv.org/pdf/2505.17966v2)**

> **作者:** Frederik Nolte; Andreas Geiger; Bernhard Schölkopf; Ingmar Posner
>
> **备注:** 20 pages, 18 figures
>
> **摘要:** This paper evaluates single-view mesh reconstruction models for their potential in enabling instant digital twin creation for real-time planning and dynamics prediction using physics simulators for robotic manipulation. Recent single-view 3D reconstruction advances offer a promising avenue toward an automated real-to-sim pipeline: directly mapping a single observation of a scene into a simulation instance by reconstructing scene objects as individual, complete, and physically plausible 3D meshes. However, their suitability for physics simulations and robotics applications under immediacy, physical fidelity, and simulation readiness remains underexplored. We establish robotics-specific benchmarking criteria for 3D reconstruction, including handling typical inputs, collision-free and stable geometry, occlusions robustness, and meeting computational constraints. Our empirical evaluation using realistic robotics datasets shows that despite success on computer vision benchmarks, existing approaches fail to meet robotics-specific requirements. We quantitively examine limitations of single-view reconstruction for practical robotics implementation, in contrast to prior work that focuses on multi-view approaches. Our findings highlight critical gaps between computer vision advances and robotics needs, guiding future research at this intersection.
>
---
#### [replaced 022] Learning Adaptive Dexterous Grasping from Single Demonstrations
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.20208v2](http://arxiv.org/pdf/2503.20208v2)**

> **作者:** Liangzhi Shi; Yulin Liu; Lingqi Zeng; Bo Ai; Zhengdong Hong; Hao Su
>
> **摘要:** How can robots learn dexterous grasping skills efficiently and apply them adaptively based on user instructions? This work tackles two key challenges: efficient skill acquisition from limited human demonstrations and context-driven skill selection. We introduce AdaDexGrasp, a framework that learns a library of grasping skills from a single human demonstration per skill and selects the most suitable one using a vision-language model (VLM). To improve sample efficiency, we propose a trajectory following reward that guides reinforcement learning (RL) toward states close to a human demonstration while allowing flexibility in exploration. To learn beyond the single demonstration, we employ curriculum learning, progressively increasing object pose variations to enhance robustness. At deployment, a VLM retrieves the appropriate skill based on user instructions, bridging low-level learned skills with high-level intent. We evaluate AdaDexGrasp in both simulation and real-world settings, showing that our approach significantly improves RL efficiency and enables learning human-like grasp strategies across varied object configurations. Finally, we demonstrate zero-shot transfer of our learned policies to a real-world PSYONIC Ability Hand, with a 90% success rate across objects, significantly outperforming the baseline.
>
---
#### [replaced 023] Dynamic Robot-Assisted Surgery with Hierarchical Class-Incremental Semantic Segmentation
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.01713v2](http://arxiv.org/pdf/2508.01713v2)**

> **作者:** Julia Hindel; Ema Mekic; Enamundram Naga Karthik; Rohit Mohan; Daniele Cattaneo; Maria Kalweit; Abhinav Valada
>
> **备注:** accepted at MICCAI AMAI 2025 workshop
>
> **摘要:** Robot-assisted surgeries rely on accurate and real-time scene understanding to safely guide surgical instruments. However, segmentation models trained on static datasets face key limitations when deployed in these dynamic and evolving surgical environments. Class-incremental semantic segmentation (CISS) allows models to continually adapt to new classes while avoiding catastrophic forgetting of prior knowledge, without training on previous data. In this work, we build upon the recently introduced Taxonomy-Oriented Poincar\'e-regularized Incremental Class Segmentation (TOPICS) approach and propose an enhanced variant, termed TOPICS+, specifically tailored for robust segmentation of surgical scenes. Concretely, we incorporate the Dice loss into the hierarchical loss formulation to handle strong class imbalances, introduce hierarchical pseudo-labeling, and design tailored label taxonomies for robotic surgery environments. We also propose six novel CISS benchmarks designed for robotic surgery environments including multiple incremental steps and several semantic categories to emulate realistic class-incremental settings in surgical environments. In addition, we introduce a refined set of labels with more than 144 classes on the Syn-Mediverse synthetic dataset, hosted online as an evaluation benchmark. We make the code and trained models publicly available at http://topics.cs.uni-freiburg.de.
>
---
#### [replaced 024] CDP: Towards Robust Autoregressive Visuomotor Policy Learning via Causal Diffusion
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.14769v2](http://arxiv.org/pdf/2506.14769v2)**

> **作者:** Jiahua Ma; Yiran Qin; Yixiong Li; Xuanqi Liao; Yulan Guo; Ruimao Zhang
>
> **摘要:** Diffusion Policy (DP) enables robots to learn complex behaviors by imitating expert demonstrations through action diffusion. However, in practical applications, hardware limitations often degrade data quality, while real-time constraints restrict model inference to instantaneous state and scene observations. These limitations seriously reduce the efficacy of learning from expert demonstrations, resulting in failures in object localization, grasp planning, and long-horizon task execution. To address these challenges, we propose Causal Diffusion Policy (CDP), a novel transformer-based diffusion model that enhances action prediction by conditioning on historical action sequences, thereby enabling more coherent and context-aware visuomotor policy learning. To further mitigate the computational cost associated with autoregressive inference, a caching mechanism is also introduced to store attention key-value pairs from previous timesteps, substantially reducing redundant computations during execution. Extensive experiments in both simulated and real-world environments, spanning diverse 2D and 3D manipulation tasks, demonstrate that CDP uniquely leverages historical action sequences to achieve significantly higher accuracy than existing methods. Moreover, even when faced with degraded input observation quality, CDP maintains remarkable precision by reasoning through temporal continuity, which highlights its practical robustness for robotic control under realistic, imperfect conditions.
>
---
#### [replaced 025] FunGraph: Functionality Aware 3D Scene Graphs for Language-Prompted Scene Interaction
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2503.07909v2](http://arxiv.org/pdf/2503.07909v2)**

> **作者:** Dennis Rotondi; Fabio Scaparro; Hermann Blum; Kai O. Arras
>
> **备注:** Paper accepted for IROS 2025
>
> **摘要:** The concept of 3D scene graphs is increasingly recognized as a powerful semantic and hierarchical representation of the environment. Current approaches often address this at a coarse, object-level resolution. In contrast, our goal is to develop a representation that enables robots to directly interact with their environment by identifying both the location of functional interactive elements and how these can be used. To achieve this, we focus on detecting and storing objects at a finer resolution, focusing on affordance-relevant parts. The primary challenge lies in the scarcity of data that extends beyond instance-level detection and the inherent difficulty of capturing detailed object features using robotic sensors. We leverage currently available 3D resources to generate 2D data and train a detector, which is then used to augment the standard 3D scene graph generation pipeline. Through our experiments, we demonstrate that our approach achieves functional element segmentation comparable to state-of-the-art 3D models and that our augmentation enables task-driven affordance grounding with higher accuracy than the current solutions. See our project page at https://fungraph.github.io.
>
---
#### [replaced 026] UniCalib: Targetless LiDAR-Camera Calibration via Probabilistic Flow on Unified Depth Representations
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2504.01416v2](http://arxiv.org/pdf/2504.01416v2)**

> **作者:** Shu Han; Xubo Zhu; Ji Wu; Ximeng Cai; Wen Yang; Huai Yu; Gui-Song Xia
>
> **备注:** 8 pages,5 figures
>
> **摘要:** Precise LiDAR-camera calibration is crucial for integrating these two sensors into robotic systems to achieve robust perception. In applications like autonomous driving, online targetless calibration enables a prompt sensor misalignment correction from mechanical vibrations without extra targets. However, existing methods exhibit limitations in effectively extracting consistent features from LiDAR and camera data and fail to prioritize salient regions, compromising cross-modal alignment robustness. To address these issues, we propose DF-Calib, a LiDAR-camera calibration method that reformulates calibration as an intra-modality depth flow estimation problem. DF-Calib estimates a dense depth map from the camera image and completes the sparse LiDAR projected depth map, using a shared feature encoder to extract consistent depth-to-depth features, effectively bridging the 2D-3D cross-modal gap. Additionally, we introduce a reliability map to prioritize valid pixels and propose a perceptually weighted sparse flow loss to enhance depth flow estimation. Experimental results across multiple datasets validate its accuracy and generalization,with DF-Calib achieving a mean translation error of 0.635cm and rotation error of 0.045 degrees on the KITTI dataset.
>
---
#### [replaced 027] EfficientEQA: An Efficient Approach to Open-Vocabulary Embodied Question Answering
- **分类: cs.RO; cs.AI; cs.CV**

- **链接: [http://arxiv.org/pdf/2410.20263v2](http://arxiv.org/pdf/2410.20263v2)**

> **作者:** Kai Cheng; Zhengyuan Li; Xingpeng Sun; Byung-Cheol Min; Amrit Singh Bedi; Aniket Bera
>
> **备注:** IROS 2025 Oral
>
> **摘要:** Embodied Question Answering (EQA) is an essential yet challenging task for robot assistants. Large vision-language models (VLMs) have shown promise for EQA, but existing approaches either treat it as static video question answering without active exploration or restrict answers to a closed set of choices. These limitations hinder real-world applicability, where a robot must explore efficiently and provide accurate answers in open-vocabulary settings. To overcome these challenges, we introduce EfficientEQA, a novel framework that couples efficient exploration with free-form answer generation. EfficientEQA features three key innovations: (1) Semantic-Value-Weighted Frontier Exploration (SFE) with Verbalized Confidence (VC) from a black-box VLM to prioritize semantically important areas to explore, enabling the agent to gather relevant information faster; (2) a BLIP relevancy-based mechanism to stop adaptively by flagging highly relevant observations as outliers to indicate whether the agent has collected enough information; and (3) a Retrieval-Augmented Generation (RAG) method for the VLM to answer accurately based on pertinent images from the agent's observation history without relying on predefined choices. Our experimental results show that EfficientEQA achieves over 15% higher answer accuracy and requires over 20% fewer exploration steps than state-of-the-art methods. Our code is available at: https://github.com/chengkaiAcademyCity/EfficientEQA
>
---
#### [replaced 028] Learn to Teach: Sample-Efficient Privileged Learning for Humanoid Locomotion over Diverse Terrains
- **分类: cs.RO; cs.LG**

- **链接: [http://arxiv.org/pdf/2402.06783v3](http://arxiv.org/pdf/2402.06783v3)**

> **作者:** Feiyang Wu; Xavier Nal; Jaehwi Jang; Wei Zhu; Zhaoyuan Gu; Anqi Wu; Ye Zhao
>
> **备注:** Accepted to IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** Humanoid robots promise transformative capabilities for industrial and service applications. While recent advances in Reinforcement Learning (RL) yield impressive results in locomotion, manipulation, and navigation, the proposed methods typically require enormous simulation samples to account for real-world variability. This work proposes a novel one-stage training framework-Learn to Teach (L2T)-which unifies teacher and student policy learning. Our approach recycles simulator samples and synchronizes the learning trajectories through shared dynamics, significantly reducing sample complexities and training time while achieving state-of-the-art performance. Furthermore, we validate the RL variant (L2T-RL) through extensive simulations and hardware tests on the Digit robot, demonstrating zero-shot sim-to-real transfer and robust performance over 12+ challenging terrains without depth estimation modules.
>
---
#### [replaced 029] Mapless Collision-Free Flight via MPC using Dual KD-Trees in Cluttered Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.10141v3](http://arxiv.org/pdf/2503.10141v3)**

> **作者:** Linzuo Zhang; Yu Hu; Yang Deng; Feng Yu; Danping Zou
>
> **摘要:** Collision-free flight in cluttered environments is a critical capability for autonomous quadrotors. Traditional methods often rely on detailed 3D map construction, trajectory generation, and tracking. However, this cascade pipeline can introduce accumulated errors and computational delays, limiting flight agility and safety. In this paper, we propose a novel method for enabling collision-free flight in cluttered environments without explicitly constructing 3D maps or generating and tracking collision-free trajectories. Instead, we leverage Model Predictive Control (MPC) to directly produce safe actions from sparse waypoints and point clouds from a depth camera. These sparse waypoints are dynamically adjusted online based on nearby obstacles detected from point clouds. To achieve this, we introduce a dual KD-Tree mechanism: the Obstacle KD-Tree quickly identifies the nearest obstacle for avoidance, while the Edge KD-Tree provides a robust initial guess for the MPC solver, preventing it from getting stuck in local minima during obstacle avoidance. We validate our approach through extensive simulations and real-world experiments. The results show that our approach significantly outperforms the mapping-based methods and is also superior to imitation learning-based methods, demonstrating reliable obstacle avoidance at up to 12 m/s in simulations and 6 m/s in real-world tests. Our method provides a simple and robust alternative to existing methods. The code is publicly available at https://github.com/SJTU-ViSYS-team/avoid-mpc.
>
---
#### [replaced 030] Language-Driven Policy Distillation for Cooperative Driving in Multi-Agent Reinforcement Learning
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.24152v2](http://arxiv.org/pdf/2410.24152v2)**

> **作者:** Jiaqi Liu; Chengkai Xu; Peng Hang; Jian Sun; Wei Zhan; Masayoshi Tomizuka; Mingyu Ding
>
> **摘要:** The cooperative driving technology of Connected and Autonomous Vehicles (CAVs) is crucial for improving the efficiency and safety of transportation systems. Learning-based methods, such as Multi-Agent Reinforcement Learning (MARL), have demonstrated strong capabilities in cooperative decision-making tasks. However, existing MARL approaches still face challenges in terms of learning efficiency and performance. In recent years, Large Language Models (LLMs) have rapidly advanced and shown remarkable abilities in various sequential decision-making tasks. To enhance the learning capabilities of cooperative agents while ensuring decision-making efficiency and cost-effectiveness, we propose LDPD, a language-driven policy distillation method for guiding MARL exploration. In this framework, a teacher agent based on LLM trains smaller student agents to achieve cooperative decision-making through its own decision-making demonstrations. The teacher agent enhances the observation information of CAVs and utilizes LLMs to perform complex cooperative decision-making reasoning, which also leverages carefully designed decision-making tools to achieve expert-level decisions, providing high-quality teaching experiences. The student agent then refines the teacher's prior knowledge into its own model through gradient policy updates. The experiments demonstrate that the students can rapidly improve their capabilities with minimal guidance from the teacher and eventually surpass the teacher's performance. Extensive experiments show that our approach demonstrates better performance and learning efficiency compared to baseline methods.
>
---
#### [replaced 031] Embodied intelligent industrial robotics: Concepts and techniques
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.09305v4](http://arxiv.org/pdf/2505.09305v4)**

> **作者:** Chaoran Zhang; Chenhao Zhang; Zhaobo Xu; Qinghongbing Xie; Jinliang Hou; Pingfa Feng; Long Zeng
>
> **备注:** 68 pages, 12 figures. The associated project can be found at https://github.com/jackyzengl/EIIR
>
> **摘要:** In order to work more efficiently, accurately, reliably, and safely in industrial scenarios, robots should have at least general knowledge, working-environment knowledge, and operating-object knowledge. These pose significant challenges to existing embodied intelligent robotics (EIR) techniques. Thus, this paper first briefly reviews the history of industrial robotics and analyzes the limitations of mainstream EIR frameworks. Then, a knowledge-driven technical framework of embodied intelligent industrial robotics (EIIR) is proposed for various industrial environments. It has five modules: a world model, a high-level task planner, a low-level skill controller, a simulator, and a physical system. The development of techniques related to each module are also thoroughly reviewed, and recent progress regarding their adaption to industrial applications are discussed. A case study is given to demonstrate the newly proposed EIIR framework's applicability to real-world assembly system. Finally, the key challenges that EIIR encounters in industrial scenarios are summarized and future research directions are suggested. The authors believe that EIIR technology is shaping the next generation of industrial robotics and EIIR-based industrial systems supply a new technological paradigm for intelligent manufacturing. It is expected that this review could serve as a valuable reference for scholars and engineers that are interested in industrial embodied intelligence. Together, scholars can use this research to drive their rapid advancement and application of EIIR techniques. The interested authors would continue to track and contribute new studies in the project page https://github.com/jackyzengl/EIIR.
>
---
#### [replaced 032] LifelongPR: Lifelong point cloud place recognition based on sample replay and prompt learning
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.10034v2](http://arxiv.org/pdf/2507.10034v2)**

> **作者:** Xianghong Zou; Jianping Li; Zhe Chen; Zhen Cao; Zhen Dong; Qiegen Liu; Bisheng Yang
>
> **摘要:** Point cloud place recognition (PCPR) determines the geo-location within a prebuilt map and plays a crucial role in geoscience and robotics applications such as autonomous driving, intelligent transportation, and augmented reality. In real-world large-scale deployments of a geographic positioning system, PCPR models must continuously acquire, update, and accumulate knowledge to adapt to diverse and dynamic environments, i.e., the ability known as continual learning (CL). However, existing PCPR models often suffer from catastrophic forgetting, leading to significant performance degradation in previously learned scenes when adapting to new environments or sensor types. This results in poor model scalability, increased maintenance costs, and system deployment difficulties, undermining the practicality of PCPR. To address these issues, we propose LifelongPR, a novel continual learning framework for PCPR, which effectively extracts and fuses knowledge from sequential point cloud data. First, to alleviate the knowledge loss, we propose a replay sample selection method that dynamically allocates sample sizes according to each dataset's information quantity and selects spatially diverse samples for maximal representativeness. Second, to handle domain shifts, we design a prompt learning-based CL framework with a lightweight prompt module and a two-stage training strategy, enabling domain-specific feature adaptation while minimizing forgetting. Comprehensive experiments on large-scale public and self-collected datasets are conducted to validate the effectiveness of the proposed method. Compared with state-of-the-art (SOTA) methods, our method achieves 6.50% improvement in mIR@1, 7.96% improvement in mR@1, and an 8.95% reduction in F. The code and pre-trained models are publicly available at https://github.com/zouxianghong/LifelongPR.
>
---
#### [replaced 033] Koopman Operator Based Time-Delay Embeddings and State History Augmented LQR for Periodic Hybrid Systems: Bouncing Pendulum and Bipedal Walking
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.14455v2](http://arxiv.org/pdf/2507.14455v2)**

> **作者:** Chun-Ming Yang; Pranav A. Bhounsule
>
> **摘要:** Time-delay embedding is a technique that uses snapshots of state history over time to build a linear state space model of a nonlinear smooth system. We demonstrate that periodic non-smooth or hybrid system can also be modeled as a linear state space system using this approach as long as its behavior is consistent in modes and timings. We extend time-delay embeddings to generate a linear model of two periodic hybrid systems: the bouncing pendulum and the simplest walker with control inputs. This leads to a state history augmented linear quadratic regulator (LQR) which uses current and past state history for feedback control. Example code can be found at https://github.com/Chun-MingYang/koopman-timeDelay-lqr.git
>
---
#### [replaced 034] DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2502.05855v3](http://arxiv.org/pdf/2502.05855v3)**

> **作者:** Junjie Wen; Yichen Zhu; Jinming Li; Zhibin Tang; Chaomin Shen; Feifei Feng
>
> **备注:** The webpage is at https://dex-vla.github.io/. DexVLA is accepted by CoRL 2025
>
> **摘要:** Enabling robots to perform diverse tasks across varied environments is a central challenge in robot learning. While vision-language-action (VLA) models have shown promise for generalizable robot skills, realizing their full potential requires addressing limitations in action representation and efficient training. Current VLA models often focus on scaling the vision-language model (VLM) component, while the action space representation remains a critical bottleneck. This paper introduces DexVLA, a novel framework designed to enhance the efficiency and generalization capabilities of VLAs for complex, long-horizon tasks across diverse robot embodiments. DexVLA features a novel diffusion-based action expert, scaled to one billion parameters, designed for cross-embodiment learning. A novel embodiment curriculum learning strategy facilitates efficient training: (1) pre-training the diffusion expert that is separable from the VLA on cross-embodiment data, (2) aligning the VLA model to specific embodiments, and (3) post-training for rapid adaptation to new tasks. We conduct comprehensive experiments across multiple embodiments, including single-arm, bimanual, and dexterous hand, demonstrating DexVLA's adaptability to challenging tasks without task-specific adaptation, its ability to learn dexterous skills on novel embodiments with limited data, and its capacity to complete complex, long-horizon tasks using only direct language prompting, such as laundry folding. In all settings, our method demonstrates superior performance compared to state-of-the-art models like Octo, OpenVLA, and Diffusion Policy.
>
---
#### [replaced 035] Unveiling the Potential of iMarkers: Invisible Fiducial Markers for Advanced Robotics
- **分类: cs.RO; cs.CV; I.2.10; I.2.9; I.4.8**

- **链接: [http://arxiv.org/pdf/2501.15505v4](http://arxiv.org/pdf/2501.15505v4)**

> **作者:** Ali Tourani; Deniz Isinsu Avsar; Hriday Bavle; Jose Luis Sanchez-Lopez; Jan Lagerwall; Holger Voos
>
> **备注:** 18 pages, 10 figures, 3 tables
>
> **摘要:** Fiducial markers are widely used in various robotics tasks, facilitating enhanced navigation, object recognition, and scene understanding. Despite their advantages for robots and Augmented Reality (AR) applications, they often disrupt the visual aesthetics of environments because they are visible to humans, making them unsuitable for non-intrusive use cases. To address this gap, this paper presents "iMarkers"-innovative, unobtrusive fiducial markers detectable exclusively by robots equipped with specialized sensors. These markers offer high flexibility in production, allowing customization of their visibility range and encoding algorithms to suit various demands. The paper also introduces the hardware designs and software algorithms developed for detecting iMarkers, highlighting their adaptability and robustness in the detection and recognition stages. Various evaluations have demonstrated the effectiveness of iMarkers compared to conventional (printed) and blended fiducial markers and confirmed their applicability in diverse robotics scenarios.
>
---
#### [replaced 036] Multimodal Visual Transformer for Sim2real Transfer in Visual Reinforcement Learning
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2507.09180v3](http://arxiv.org/pdf/2507.09180v3)**

> **作者:** Zichun Xu; Yuntao Li; Zhaomin Wang; Lei Zhuang; Guocai Yang; Jingdong Zhao
>
> **摘要:** Depth information is robust to scene appearance variations and inherently carries 3D spatial details. In this paper, a visual backbone based on the vision transformer is proposed to fuse RGB and depth modalities for enhancing generalization. Different modalities are first processed by separate CNN stems, and the combined convolutional features are delivered to the scalable vision transformer to obtain visual representations. Moreover, a contrastive unsupervised learning scheme is designed with masked and unmasked tokens to accelerate the sample efficiency during the reinforcement learning process. Simulation results demonstrate that our visual backbone can focus more on task-related regions and exhibit better generalization in unseen scenarios. For sim2real transfer, a flexible curriculum learning schedule is developed to deploy domain randomization over training processes. Finally, the feasibility of our model is validated to perform real-world manipulation tasks via zero-shot transfer.
>
---
#### [replaced 037] AORRTC: Almost-Surely Asymptotically Optimal Planning with RRT-Connect
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.10542v3](http://arxiv.org/pdf/2505.10542v3)**

> **作者:** Tyler Wilson; Wil Thomason; Zachary Kingston; Jonathan Gammell
>
> **备注:** In revision for IEEE Robotics and Automation Letters (RA-L). Manuscript #25-1915. 8 pages, 4 figures, 1 table. A video of AORRTC can be found at https://www.youtube.com/watch?v=j1itxP3KuiM . Information on the implementation of AORRTC is available at https://robotic-esp.com/code/aorrtc/
>
> **摘要:** Finding high-quality solutions quickly is an important objective in motion planning. This is especially true for high-degree-of-freedom robots. Satisficing planners have traditionally found feasible solutions quickly but provide no guarantees on their optimality, while almost-surely asymptotically optimal (a.s.a.o.) planners have probabilistic guarantees on their convergence towards an optimal solution but are more computationally expensive. This paper uses the AO-x meta-algorithm to extend the satisficing RRT-Connect planner to optimal planning. The resulting Asymptotically Optimal RRT-Connect (AORRTC) finds initial solutions in similar times as RRT-Connect and uses any additional planning time to converge towards the optimal solution in an anytime manner. It is proven to be probabilistically complete and a.s.a.o. AORRTC was tested with the Panda (7 DoF) and Fetch (8 DoF) robotic arms on the MotionBenchMaker dataset. These experiments show that AORRTC finds initial solutions as fast as RRT-Connect and faster than the tested state-of-the-art a.s.a.o. algorithms while converging to better solutions faster. AORRTC finds solutions to difficult high-DoF planning problems in milliseconds where the other a.s.a.o. planners could not consistently find solutions in seconds. This performance was demonstrated both with and without single instruction/multiple data (SIMD) acceleration.
>
---
#### [replaced 038] Designing Robots with, not for: A Co-Design Framework for Empowering Interactions in Forensic Psychiatry
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.14931v2](http://arxiv.org/pdf/2507.14931v2)**

> **作者:** Qiaoqiao Ren; Remko Proesmans; Arend Pissens; Lara Dehandschutter; William Denecker; Lotte Rouckhout; Joke Carrette; Peter Vanhopplinus; Tony Belpaeme; Francis wyffels
>
> **摘要:** Forensic mental health care involves the treatment of individuals with severe mental disorders who have committed violent offences. These settings are often characterized by high levels of bureaucracy, risk avoidance, and restricted autonomy. Patients frequently experience a profound loss of control over their lives, leading to heightened psychological stress-sometimes resulting in isolation as a safety measure. In this study, we explore how co-design can be used to collaboratively develop a companion robot that helps monitor and regulate stress while maintaining tracking of the patients' interaction behaviours for long-term intervention. We conducted four co-design workshops in a forensic psychiatric clinic with patients, caregivers, and therapists. Our process began with the presentation of an initial speculative prototype to therapists, enabling reflection on shared concerns, ethical risks, and desirable features. This was followed by a creative ideation session with patients, a third workshop focused on defining desired functions and emotional responses, and we are planning a final prototype demo to gather direct patient feedback. Our findings emphasize the importance of empowering patients in the design process and adapting proposals based on their current emotional state. The goal was to empower the patient in the design process and ensure each patient's voice was heard.
>
---
#### [replaced 039] Understanding and Imitating Human-Robot Motion with Restricted Visual Fields
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2410.05547v3](http://arxiv.org/pdf/2410.05547v3)**

> **作者:** Maulik Bhatt; HongHao Zhen; Monroe Kennedy III; Negar Mehr
>
> **摘要:** When working around other agents such as humans, it is important to model their perception capabilities to predict and make sense of their behavior. In this work, we consider agents whose perception capabilities are determined by their limited field of view, viewing range, and the potential to miss objects within their viewing range. By considering the perception capabilities and observation model of agents independently from their motion policy, we show that we can better predict the agents' behavior; i.e., by reasoning about the perception capabilities of other agents, one can better make sense of their actions. We perform a user study where human operators navigate a cluttered scene while scanning the region for obstacles with a limited field of view and range. We show that by reasoning about the limited observation space of humans, a robot can better learn a human's strategy for navigating an environment and navigate with minimal collision with dynamic and static obstacles. We also show that this learned model helps it successfully navigate a physical hardware vehicle in real-time. Code available at https://github.com/labicon/HRMotion-RestrictedView.
>
---
#### [replaced 040] OceanSim: A GPU-Accelerated Underwater Robot Perception Simulation Framework
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2503.01074v2](http://arxiv.org/pdf/2503.01074v2)**

> **作者:** Jingyu Song; Haoyu Ma; Onur Bagoren; Advaith V. Sethuraman; Yiting Zhang; Katherine A. Skinner
>
> **备注:** Accepted at IROS 2025; 8 pages, 6 figures
>
> **摘要:** Underwater simulators offer support for building robust underwater perception solutions. Significant work has recently been done to develop new simulators and to advance the performance of existing underwater simulators. Still, there remains room for improvement on physics-based underwater sensor modeling and rendering efficiency. In this paper, we propose OceanSim, a high-fidelity GPU-accelerated underwater simulator to address this research gap. We propose advanced physics-based rendering techniques to reduce the sim-to-real gap for underwater image simulation. We develop OceanSim to fully leverage the computing advantages of GPUs and achieve real-time imaging sonar rendering and fast synthetic data generation. We evaluate the capabilities and realism of OceanSim using real-world data to provide qualitative and quantitative results. The code and detailed documentation are made available on the project website to support the marine robotics community: https://umfieldrobotics.github.io/OceanSim.
>
---
#### [replaced 041] Dynamic Layer Detection of Thin Materials using DenseTact Optical Tactile Sensors
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2409.09849v3](http://arxiv.org/pdf/2409.09849v3)**

> **作者:** Ankush Kundan Dhawan; Camille Chungyoun; Karina Ting; Monroe Kennedy III
>
> **备注:** 7 pages, 9 figures, accepted to IROS 2025
>
> **摘要:** Manipulation of thin materials is critical for many everyday tasks and remains a significant challenge for robots. While existing research has made strides in tasks like material smoothing and folding, many studies struggle with common failure modes (crumpled corners/edges, incorrect grasp configurations) that a preliminary step of layer detection could solve. We present a novel method for classifying the number of grasped material layers using a custom gripper equipped with DenseTact 2.0 optical tactile sensors. After grasping, the gripper performs an anthropomorphic rubbing motion while collecting optical flow, 6-axis wrench, and joint state data. Using this data in a transformer-based network achieves a test accuracy of 98.21\% in classifying the number of grasped cloth layers, and 81.25\% accuracy in classifying layers of grasped paper, showing the effectiveness of our dynamic rubbing method. Evaluating different inputs and model architectures highlights the usefulness of tactile sensor information and a transformer model for this task. A comprehensive dataset of 568 labeled trials (368 for cloth and 200 for paper) was collected and made open-source along with this paper. Our project page is available at https://armlabstanford.github.io/dynamic-cloth-detection.
>
---
#### [replaced 042] Affordance-R1: Reinforcement Learning for Generalizable Affordance Reasoning in Multimodal Large Language Model
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.06206v2](http://arxiv.org/pdf/2508.06206v2)**

> **作者:** Hanqing Wang; Shaoyang Wang; Yiming Zhong; Zemin Yang; Jiamin Wang; Zhiqing Cui; Jiahao Yuan; Yifan Han; Mingyu Liu; Yuexin Ma
>
> **摘要:** Affordance grounding focuses on predicting the specific regions of objects that are associated with the actions to be performed by robots. It plays a vital role in the fields of human-robot interaction, human-object interaction, embodied manipulation, and embodied perception. Existing models often neglect the affordance shared among different objects because they lack the Chain-of-Thought(CoT) reasoning abilities, limiting their out-of-domain (OOD) generalization and explicit reasoning capabilities. To address these challenges, we propose Affordance-R1, the first unified affordance grounding framework that integrates cognitive CoT guided Group Relative Policy Optimization (GRPO) within a reinforcement learning paradigm. Specifically, we designed a sophisticated affordance function, which contains format, perception, and cognition rewards to effectively guide optimization directions. Furthermore, we constructed a high-quality affordance-centric reasoning dataset, ReasonAff, to support training. Trained exclusively via reinforcement learning with GRPO and without explicit reasoning data, Affordance-R1 achieves robust zero-shot generalization and exhibits emergent test-time reasoning capabilities. Comprehensive experiments demonstrate that our model outperforms well-established methods and exhibits open-world generalization. To the best of our knowledge, Affordance-R1 is the first to integrate GRPO-based RL with reasoning into affordance reasoning. The code of our method and our dataset is released on https://github.com/hq-King/Affordance-R1.
>
---
