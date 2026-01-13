# 机器人 cs.RO

- **最新发布 51 篇**

- **更新 33 篇**

## 最新发布

#### [new 001] Heterogeneous Multi-Expert Reinforcement Learning for Long-Horizon Multi-Goal Tasks in Autonomous Forklifts
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主叉车的长周期多目标任务，旨在解决导航与操作间的冲突问题。通过分层强化学习框架，分离导航与操作策略，提升任务成功率和精度。**

- **链接: [https://arxiv.org/pdf/2601.07304v1](https://arxiv.org/pdf/2601.07304v1)**

> **作者:** Yun Chen; Bowei Huang; Fan Guo; Kang Song
>
> **备注:** 9 pages
>
> **摘要:** Autonomous mobile manipulation in unstructured warehouses requires a balance between efficient large-scale navigation and high-precision object interaction. Traditional end-to-end learning approaches often struggle to handle the conflicting demands of these distinct phases. Navigation relies on robust decision-making over large spaces, while manipulation needs high sensitivity to fine local details. Forcing a single network to learn these different objectives simultaneously often causes optimization interference, where improving one task degrades the other. To address these limitations, we propose a Heterogeneous Multi-Expert Reinforcement Learning (HMER) framework tailored for autonomous forklifts. HMER decomposes long-horizon tasks into specialized sub-policies controlled by a Semantic Task Planner. This structure separates macro-level navigation from micro-level manipulation, allowing each expert to focus on its specific action space without interference. The planner coordinates the sequential execution of these experts, bridging the gap between task planning and continuous control. Furthermore, to solve the problem of sparse exploration, we introduce a Hybrid Imitation-Reinforcement Training Strategy. This method uses expert demonstrations to initialize the policy and Reinforcement Learning for fine-tuning. Experiments in Gazebo simulations show that HMER significantly outperforms sequential and end-to-end baselines. Our method achieves a task success rate of 94.2\% (compared to 62.5\% for baselines), reduces operation time by 21.4\%, and maintains placement error within 1.5 cm, validating its efficacy for precise material handling.
>
---
#### [new 002] CulinaryCut-VLAP: A Vision-Language-Action-Physics Framework for Food Cutting via a Force-Aware Material Point Method
- **分类: cs.RO; cs.CV**

- **简介: 该论文聚焦于食品切割任务，解决变形材料切割中的物理建模与视觉-语言-动作对齐问题，提出融合视觉语言与物理模拟的框架。**

- **链接: [https://arxiv.org/pdf/2601.06451v1](https://arxiv.org/pdf/2601.06451v1)**

> **作者:** Hyunseo Koh; Chang-Yong Song; Youngjae Choi; Misa Viveiros; David Hyde; Heewon Kim
>
> **备注:** 16 pages; 15 figures; 5 tables
>
> **摘要:** Food cutting is a highly practical yet underexplored application at the intersection of vision and robotic manipulation. The task remains challenging because interactions between the knife and deformable materials are highly nonlinear and often entail large deformations, frequent contact, and topological change, which in turn hinder stable and safe large-scale data collection. To address these challenges, we propose a unified framework that couples a vision-language-action (VLA) dataset with a physically realistic cutting simulator built on the material point method (MPM). Our simulator adopts MLS-MPM as its computational core, reducing numerical dissipation and energy drift while preserving rotational and shear responses even under topology-changing cuts. During cutting, forces and stress distributions are estimated from impulse exchanges between particles and the grid, enabling stable tracking of transient contact forces and energy transfer. We also provide a benchmark dataset that integrates diverse cutting trajectories, multi-view visual observations, and fine-grained language instructions, together with force--torque and tool--pose labels to provide physically consistent training signals. These components realize a learning--evaluation loop that respects the core physics of cutting and establishes a safe, reproducible, and scalable foundation for advancing VLA models in deformable object manipulation.
>
---
#### [new 003] UMLoc: Uncertainty-Aware Map-Constrained Inertial Localization with Quantified Bounds
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于室内定位任务，解决IMU在无GPS环境下的漂移问题。提出UMLoc框架，结合IMU数据与地图约束，实现精准且带有不确定性评估的定位。**

- **链接: [https://arxiv.org/pdf/2601.06602v1](https://arxiv.org/pdf/2601.06602v1)**

> **作者:** Mohammed S. Alharbi; Shinkyu Park
>
> **摘要:** Inertial localization is particularly valuable in GPS-denied environments such as indoors. However, localization using only Inertial Measurement Units (IMUs) suffers from drift caused by motion-process noise and sensor biases. This paper introduces Uncertainty-aware Map-constrained Inertial Localization (UMLoc), an end-to-end framework that jointly models IMU uncertainty and map constraints to achieve drift-resilient positioning. UMLoc integrates two coupled modules: (1) a Long Short-Term Memory (LSTM) quantile regressor, which estimates the specific quantiles needed to define 68%, 90%, and 95% prediction intervals serving as a measure of localization uncertainty and (2) a Conditioned Generative Adversarial Network (CGAN) with cross-attention that fuses IMU dynamic data with distance-based floor-plan maps to generate geometrically feasible trajectories. The modules are trained jointly, allowing uncertainty estimates to propagate through the CGAN during trajectory generation. UMLoc was evaluated on three datasets, including a newly collected 2-hour indoor benchmark with time-aligned IMU data, ground-truth poses and floor-plan maps. Results show that the method achieves a mean drift ratio of 5.9% over a 70 m travel distance and an average Absolute Trajectory Error (ATE) of 1.36 m, while maintaining calibrated prediction bounds.
>
---
#### [new 004] NanoCockpit: Performance-optimized Application Framework for AI-based Autonomous Nanorobotics
- **分类: cs.RO; cs.SE; eess.SY**

- **简介: 该论文提出NanoCockpit框架，解决纳米机器人中资源受限下的高效任务调度问题，通过优化图像处理与通信流程，提升控制性能。**

- **链接: [https://arxiv.org/pdf/2601.07476v1](https://arxiv.org/pdf/2601.07476v1)**

> **作者:** Elia Cereda; Alessandro Giusti; Daniele Palossi
>
> **备注:** Source code available on GitHub at https://github.com/idsia-robotics/crazyflie-nanocockpit
>
> **摘要:** Autonomous nano-drones, powered by vision-based tiny machine learning (TinyML) models, are a novel technology gaining momentum thanks to their broad applicability and pushing scientific advancement on resource-limited embedded systems. Their small form factor, i.e., a few 10s grams, severely limits their onboard computational resources to sub-\SI{100}{\milli\watt} microcontroller units (MCUs). The Bitcraze Crazyflie nano-drone is the \textit{de facto} standard, offering a rich set of programmable MCUs for low-level control, multi-core processing, and radio transmission. However, roboticists very often underutilize these onboard precious resources due to the absence of a simple yet efficient software layer capable of time-optimal pipelining of multi-buffer image acquisition, multi-core computation, intra-MCUs data exchange, and Wi-Fi streaming, leading to sub-optimal control performances. Our \textit{NanoCockpit} framework aims to fill this gap, increasing the throughput and minimizing the system's latency, while simplifying the developer experience through coroutine-based multi-tasking. In-field experiments on three real-world TinyML nanorobotics applications show our framework achieves ideal end-to-end latency, i.e. zero overhead due to serialized tasks, delivering quantifiable improvements in closed-loop control performance ($-$30\% mean position error, mission success rate increased from 40\% to 100\%).
>
---
#### [new 005] Model Reconciliation through Explainability and Collaborative Recovery in Assistive Robotics
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机协作任务，解决机器人与人类模型不一致问题。通过框架实现模型协调，利用大语言模型解释差异并允许人类修正。**

- **链接: [https://arxiv.org/pdf/2601.06552v1](https://arxiv.org/pdf/2601.06552v1)**

> **作者:** Britt Besch; Tai Mai; Jeremias Thun; Markus Huff; Jörn Vogel; Freek Stulp; Samuel Bustamante
>
> **摘要:** Whenever humans and robots work together, it is essential that unexpected robot behavior can be explained to the user. Especially in applications such as shared control the user and the robot must share the same model of the objects in the world, and the actions that can be performed on these objects. In this paper, we achieve this with a so-called model reconciliation framework. We leverage a Large Language Model to predict and explain the difference between the robot's and the human's mental models, without the need of a formal mental model of the user. Furthermore, our framework aims to solve the model divergence after the explanation by allowing the human to correct the robot. We provide an implementation in an assistive robotics domain, where we conduct a set of experiments with a real wheelchair-based mobile manipulator and its digital twin.
>
---
#### [new 006] Failure-Aware RL: Reliable Offline-to-Online Reinforcement Learning with Self-Recovery for Real-World Manipulation
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人强化学习任务，旨在解决真实世界中因失败导致的干预问题。通过引入FARL算法，结合安全评论器和恢复策略，减少失败并提升性能。**

- **链接: [https://arxiv.org/pdf/2601.07821v1](https://arxiv.org/pdf/2601.07821v1)**

> **作者:** Huanyu Li; Kun Lei; Sheng Zang; Kaizhe Hu; Yongyuan Liang; Bo An; Xiaoli Li; Huazhe Xu
>
> **备注:** Project page: https://failure-aware-rl.github.io
>
> **摘要:** Post-training algorithms based on deep reinforcement learning can push the limits of robotic models for specific objectives, such as generalizability, accuracy, and robustness. However, Intervention-requiring Failures (IR Failures) (e.g., a robot spilling water or breaking fragile glass) during real-world exploration happen inevitably, hindering the practical deployment of such a paradigm. To tackle this, we introduce Failure-Aware Offline-to-Online Reinforcement Learning (FARL), a new paradigm minimizing failures during real-world reinforcement learning. We create FailureBench, a benchmark that incorporates common failure scenarios requiring human intervention, and propose an algorithm that integrates a world-model-based safety critic and a recovery policy trained offline to prevent failures during online exploration. Extensive simulation and real-world experiments demonstrate the effectiveness of FARL in significantly reducing IR Failures while improving performance and generalization during online reinforcement learning post-training. FARL reduces IR Failures by 73.1% while elevating performance by 11.3% on average during real-world RL post-training. Videos and code are available at https://failure-aware-rl.github.io.
>
---
#### [new 007] Semilinear single-track vehicle models with distributed tyre friction dynamics
- **分类: cs.RO**

- **简介: 该论文属于车辆动力学建模任务，旨在解决轮胎瞬态动态与非线性摩擦效应的建模问题。提出一种基于分布摩擦与刷子动力学的半线性模型，用于描述车辆横向运动与轮胎变形的相互作用。**

- **链接: [https://arxiv.org/pdf/2601.06854v1](https://arxiv.org/pdf/2601.06854v1)**

> **作者:** Luigi Romano; Ole Morten Aamo; Jan Åslund; Erik Frisk
>
> **备注:** 37 pages, 12 figures. Accepted by Nonlinear Dynamics
>
> **摘要:** This paper introduces a novel family of single-track vehicle models that incorporate a distributed representation of transient tyre dynamics, whilst simultaneously accounting for nonlinear effects induced by friction. The core of the proposed framework is represented by the distributed Friction with Bristle Dynamics (FrBD) model, which unifies and extends classical formulations such as Dahl and LuGre by describing the rolling contact process as a spatially distributed system governed by semilinear partial differential equations (PDEs). This model is systematically integrated into a single-track vehicle framework, where the resulting semilinear ODE-PDE interconnection captures the interaction between lateral vehicle motion and tyre deformation. Two main variants are considered: one with rigid tyre carcass and another with flexible carcass, each admitting a compact state-space representation. Local and global well-posedness properties for the coupled system are established rigorously, highlighting the dissipative and physically consistent properties of the distributed FrBD model. A linearisation procedure is also presented, enabling spectral analysis and transfer function derivation, and potentially facilitating the synthesis of controllers and observers. Numerical simulations demonstrate the model's capability to capture micro-shimmy oscillations and transient lateral responses to advanced steering manoeuvres. The proposed formulation advances the state-of-the-art in vehicle dynamics modelling by providing a physically grounded, mathematically rigorous, and computationally tractable approach to incorporating transient tyre behaviour in lateral vehicle dynamics, when accounting for the effect of limited friction.
>
---
#### [new 008] Follow the Signs: Using Textual Cues and LLMs to Guide Efficient Robot Navigation
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，解决在未知环境中高效定位目标的问题。通过结合文本线索与大语言模型，提升导航效率。**

- **链接: [https://arxiv.org/pdf/2601.06652v1](https://arxiv.org/pdf/2601.06652v1)**

> **作者:** Jing Cao; Nishanth Kumar; Aidan Curtis
>
> **摘要:** Autonomous navigation in unfamiliar environments often relies on geometric mapping and planning strategies that overlook rich semantic cues such as signs, room numbers, and textual labels. We propose a novel semantic navigation framework that leverages large language models (LLMs) to infer patterns from partial observations and predict regions where the goal is most likely located. Our method combines local perceptual inputs with frontier-based exploration and periodic LLM queries, which extract symbolic patterns (e.g., room numbering schemes and building layout structures) and update a confidence grid used to guide exploration. This enables robots to move efficiently toward goal locations labeled with textual identifiers (e.g., "room 8") even before direct observation. We demonstrate that this approach enables more efficient navigation in sparse, partially observable grid environments by exploiting symbolic patterns. Experiments across environments modeled after real floor plans show that our approach consistently achieves near-optimal paths and outperforms baselines by over 25% in Success weighted by Path Length.
>
---
#### [new 009] A Sliding Mode Controller Based on Timoshenko Beam Theory Developed for a Tendon-Driven Robotic Wrist
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决腱驱动腕部的精确运动控制问题。通过建立基于Timoshenko梁理论的模型，并设计滑模控制器，提升控制精度与响应速度。**

- **链接: [https://arxiv.org/pdf/2601.07009v1](https://arxiv.org/pdf/2601.07009v1)**

> **作者:** Shifa Sulaiman; Mohammad Gohari; Francesco Schetter; Fanny Ficuciello
>
> **摘要:** Development of dexterous robotic joints is essential for advancing manipulation capabilities in robotic systems. This paper presents the design and implementation of a tendon-driven robotic wrist joint together with an efficient Sliding Mode Controller (SMC) for precise motion control. The wrist mechanism is modeled using a Timoshenko-based approach to accurately capture its kinematic and dynamic properties, which serve as the foundation for tendon force calculations within the controller. The proposed SMC is designed to deliver fast dynamic response and computational efficiency, enabling accurate trajectory tracking under varying operating conditions. The effectiveness of the controller is validated through comparative analyses with existing controllers for similar wrist mechanisms. The proposed SMC demonstrates superior performance in both simulation and experimental studies. The Root Mean Square Error (RMSE) in simulation is approximately 1.67e-2 radians, while experimental validation yields an error of 0.2 radians. Additionally, the controller achieves a settling time of less than 3 seconds and a steady-state error below 1e-1 radians, consistently observed across both simulation and experimental evaluations. Comparative analyses confirm that the developed SMC surpasses alternative control strategies in motion accuracy, rapid convergence, and steady-state precision. This work establishes a foundation for future exploration of tendon-driven wrist mechanisms and control strategies in robotic applications.
>
---
#### [new 010] Observability-Enhanced Target Motion Estimation via Bearing-Box: Theory and MAV Applications
- **分类: cs.RO**

- **简介: 该论文属于目标运动估计任务，解决单目视觉下目标运动与尺寸估计问题。提出bearing-box方法，利用3D边界框信息，无需复杂假设，提升MAV应用中的估计性能。**

- **链接: [https://arxiv.org/pdf/2601.06887v1](https://arxiv.org/pdf/2601.06887v1)**

> **作者:** Yin Zhang; Zian Ning; Shiyu Zhao
>
> **备注:** This paper is accepted by IEEE Transactions on Robotics (20 pages, 11 figures)
>
> **摘要:** Monocular vision-based target motion estimation is a fundamental challenge in numerous applications. This work introduces a novel bearing-box approach that fully leverages modern 3D detection measurements that are widely available nowadays but have not been well explored for motion estimation so far. Unlike existing methods that rely on restrictive assumptions such as isotropic target shape and lateral motion, our bearing-box estimator can estimate both the target's motion and its physical size without these assumptions by exploiting the information buried in a 3D bounding box. When applied to multi-rotor micro aerial vehicles (MAVs), the estimator yields an interesting advantage: it further removes the need for higher-order motion assumptions by exploiting the unique coupling between MAV's acceleration and thrust. This is particularly significant, as higher-order motion assumptions are widely believed to be necessary in state-of-the-art bearing-based estimators. We support our claims with rigorous observability analyses and extensive experimental validation, demonstrating the estimator's superior performance in real-world scenarios.
>
---
#### [new 011] THETA: Triangulated Hand-State Estimation for Teleoperation and Automation in Robotic Hand Control
- **分类: cs.RO**

- **简介: 该论文提出THETA系统，用于低成本的机械手遥操作。任务是估计手指关节角度，解决传统传感器成本高的问题。通过三摄像头和深度学习实现手部姿态估计，并控制低成本机械手。**

- **链接: [https://arxiv.org/pdf/2601.07768v1](https://arxiv.org/pdf/2601.07768v1)**

> **作者:** Alex Huang; Akshay Karthik
>
> **备注:** The 11th International Conference on Engineering and Emerging Technologies (ICEET) 2025
>
> **摘要:** The teleoperation of robotic hands is limited by the high costs of depth cameras and sensor gloves, commonly used to estimate hand relative joint positions (XYZ). We present a novel, cost-effective approach using three webcams for triangulation-based tracking to approximate relative joint angles (theta) of human fingers. We also introduce a modified DexHand, a low-cost robotic hand from TheRobotStudio, to demonstrate THETA's real-time application. Data collection involved 40 distinct hand gestures using three 640x480p webcams arranged at 120-degree intervals, generating over 48,000 RGB images. Joint angles were manually determined by measuring midpoints of the MCP, PIP, and DIP finger joints. Captured RGB frames were processed using a DeepLabV3 segmentation model with a ResNet-50 backbone for multi-scale hand segmentation. The segmented images were then HSV-filtered and fed into THETA's architecture, consisting of a MobileNetV2-based CNN classifier optimized for hierarchical spatial feature extraction and a 9-channel input tensor encoding multi-perspective hand representations. The classification model maps segmented hand views into discrete joint angles, achieving 97.18% accuracy, 98.72% recall, F1 Score of 0.9274, and a precision of 0.8906. In real-time inference, THETA captures simultaneous frames, segments hand regions, filters them, and compiles a 9-channel tensor for classification. Joint-angle predictions are relayed via serial to an Arduino, enabling the DexHand to replicate hand movements. Future research will increase dataset diversity, integrate wrist tracking, and apply computer vision techniques such as OpenAI-Vision. THETA potentially ensures cost-effective, user-friendly teleoperation for medical, linguistic, and manufacturing applications.
>
---
#### [new 012] PROTEA: Securing Robot Task Planning and Execution
- **分类: cs.RO**

- **简介: 该论文属于机器人安全任务，旨在解决任务规划中的安全漏洞问题。提出PROTEA机制，利用大模型评估任务计划安全性，提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.07186v1](https://arxiv.org/pdf/2601.07186v1)**

> **作者:** Zainab Altaweel; Mohaiminul Al Nahian; Jake Juettner; Adnan Siraj Rakin; Shiqi Zhang
>
> **摘要:** Robots need task planning methods to generate action sequences for complex tasks. Recent work on adversarial attacks has revealed significant vulnerabilities in existing robot task planners, especially those built on foundation models. In this paper, we aim to address these security challenges by introducing PROTEA, an LLM-as-a-Judge defense mechanism, to evaluate the security of task plans. PROTEA is developed to address the dimensionality and history challenges in plan safety assessment. We used different LLMs to implement multiple versions of PROTEA for comparison purposes. For systemic evaluations, we created a dataset containing both benign and malicious task plans, where the harmful behaviors were injected at varying levels of stealthiness. Our results provide actionable insights for robotic system practitioners seeking to enhance robustness and security of their task planning systems. Details, dataset and demos are provided: https://protea-secure.github.io/PROTEA/
>
---
#### [new 013] Precision Meets Art: Autonomous Multi-UAV System for Large Scale Mural Drawing
- **分类: cs.RO; cs.CV; eess.SY**

- **简介: 该论文属于多无人机协同绘画任务，解决大尺度壁画自动化绘制问题。设计了多无人机系统，结合定位与控制算法，实现高效精准的户外壁画创作。**

- **链接: [https://arxiv.org/pdf/2601.06508v1](https://arxiv.org/pdf/2601.06508v1)**

> **作者:** Andrei A. Korigodskii; Artem E. Vasiunik; Georgii A. Varin; Adilia M. Zukhurova; Matvei V. Urvantsev; Semen A. Osipenkov; Igor S. Efremov; Georgii E. Bondar
>
> **备注:** 6 pages, 9 figures
>
> **摘要:** The integration of autonomous unmanned aerial vehicles (UAVs) into large-scale artistic projects has emerged as a new application in robotics. This paper presents the design, deployment, and testing of a novel multi-drone system for automated mural painting in outdoor settings. This technology makes use of new software that coordinates multiple drones simultaneously, utilizing state-machine algorithms for task execution. Key advancements are the complex positioning system that combines 2D localization using a single motion tracking camera with onboard LiDAR for precise positioning, and a novel flight control algorithm, which works differently along the trajectory and normally to it, ensuring smoothness and high precision of the drawings at the same time. A 100 square meters mural was created using the developed multi-drone system, validating the system's efficacy. Compared to single-drone approaches, our multi-UAV solution significantly improves scalability and operational speed while maintaining high stability even in harsh weather conditions. The findings highlight the potential of autonomous robotic swarms in creative applications, paving the way for further advancements in large-scale robotic art.
>
---
#### [new 014] Data-driven control of hydraulic impact hammers under strict operational and control constraints
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决液压冲击锤在有限传感和离散控制下的位姿控制问题。通过数据驱动方法，利用遥操作数据进行系统建模与控制器设计，实现精准的末端定位。**

- **链接: [https://arxiv.org/pdf/2601.07813v1](https://arxiv.org/pdf/2601.07813v1)**

> **作者:** Francisco Leiva; Claudio Canales; Michelle Valenzuela; Javier Ruiz-del-Solar
>
> **备注:** 21 pages, 14 figures
>
> **摘要:** This paper presents a data-driven methodology for the control of static hydraulic impact hammers, also known as rock breakers, which are commonly used in the mining industry. The task addressed in this work is that of controlling the rock-breaker so its end-effector reaches arbitrary target poses, which is required in normal operation to place the hammer on top of rocks that need to be fractured. The proposed approach considers several constraints, such as unobserved state variables due to limited sensing and the strict requirement of using a discrete control interface at the joint level. First, the proposed methodology addresses the problem of system identification to obtain an approximate dynamic model of the hydraulic arm. This is done via supervised learning, using only teleoperation data. The learned dynamic model is then exploited to obtain a controller capable of reaching target end-effector poses. For policy synthesis, both reinforcement learning (RL) and model predictive control (MPC) algorithms are utilized and contrasted. As a case study, we consider the automation of a Bobcat E10 mini-excavator arm with a hydraulic impact hammer attached as end-effector. Using this machine, both the system identification and policy synthesis stages are studied in simulation and in the real world. The best RL-based policy consistently reaches target end-effector poses with position errors below 12 cm and pitch angle errors below 0.08 rad in the real world. Considering that the impact hammer has a 4 cm diameter chisel, this level of precision is sufficient for breaking rocks. Notably, this is accomplished by relying only on approximately 68 min of teleoperation data to train and 8 min to evaluate the dynamic model, and without performing any adjustments for a successful policy Sim2Real transfer. A demonstration of policy execution in the real world can be found in https://youtu.be/e-7tDhZ4ZgA.
>
---
#### [new 015] SPINE Gripper: A Twisted Underactuated Mechanism-based Passive Mode-Transition Gripper
- **分类: cs.RO**

- **简介: 该论文提出一种单驱动被动夹爪，解决多功能抓取与旋转切换问题。通过机械设计实现稳定抓取和双向旋转，无需传感器或控制。**

- **链接: [https://arxiv.org/pdf/2601.06833v1](https://arxiv.org/pdf/2601.06833v1)**

> **作者:** JaeHyung Jang; JunHyeong Park; Joong-Ku Lee; Jee-Hwan Ryu
>
> **备注:** 11 pages, 10 figures. Preprint version of a manuscript submitted to IEEE Transactions on Mechatronics
>
> **摘要:** This paper presents a single-actuator passive gripper that achieves both stable grasping and continuous bidirectional in-hand rotation through mechanically encoded power transmission logic. Unlike conventional multifunctional grippers that require multiple actuators, sensors, or control-based switching, the proposed gripper transitions between grasping and rotation solely according to the magnitude of the applied input torque. The key enabler of this behavior is a Twisted Underactuated Mechanism (TUM), which generates non-coplanar motions, namely axial contraction and rotation, from a single rotational input while producing identical contraction regardless of rotation direction. A friction generator mechanically defines torque thresholds that govern passive mode switching, enabling stable grasp establishment before autonomously transitioning to in-hand rotation without sensing or active control. Analytical models describing the kinematics, elastic force generation, and torque transmission of the TUM are derived and experimentally validated. The fabricated gripper is evaluated through quantitative experiments on grasp success, friction-based grasp force regulation, and bidirectional rotation performance. System-level demonstrations, including bolt manipulation, object reorientation, and manipulator-integrated tasks driven solely by wrist torque, confirm reliable grasp to rotate transitions in both rotational directions. These results demonstrate that non-coplanar multifunctional manipulation can be realized through mechanical design alone, establishing mechanically encoded power transmission logic as a robust alternative to actuator and control intensive gripper architectures.
>
---
#### [new 016] PALM: Progress-Aware Policy Learning via Affordance Reasoning for Long-Horizon Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于长期机器人操作任务，解决多步骤任务中执行错误问题。提出PALM框架，通过 affordance 理解和进度跟踪提升任务执行效果。**

- **链接: [https://arxiv.org/pdf/2601.07060v1](https://arxiv.org/pdf/2601.07060v1)**

> **作者:** Yuanzhe Liu; Jingyuan Zhu; Yuchen Mo; Gen Li; Xu Cao; Jin Jin; Yifan Shen; Zhengyuan Li; Tianjiao Yu; Wenzhen Yuan; Fangqiang Ding; Ismini Lourentzou
>
> **摘要:** Recent advancements in vision-language-action (VLA) models have shown promise in robotic manipulation, yet they continue to struggle with long-horizon, multi-step tasks. Existing methods lack internal reasoning mechanisms that can identify task-relevant interaction cues or track progress within a subtask, leading to critical execution errors such as repeated actions, missed steps, and premature termination. To address these challenges, we introduce PALM, a VLA framework that structures policy learning around interaction-centric affordance reasoning and subtask progress cues. PALM distills complementary affordance representations that capture object relevance, contact geometry, spatial placements, and motion dynamics, and serve as task-relevant anchors for visuomotor control. To further stabilize long-horizon execution, PALM predicts continuous within-subtask progress, enabling seamless subtask transitions. Across extensive simulation and real-world experiments, PALM consistently outperforms baselines, achieving a 91.8% success rate on LIBERO-LONG, a 12.5% improvement in average length on CALVIN ABC->D, and a 2x improvement over real-world baselines across three long-horizon generalization settings.
>
---
#### [new 017] On-the-Fly VLA Adaptation via Test-Time Reinforcement Learning
- **分类: cs.RO**

- **简介: 该论文提出TT-VLA，解决VLAs在动态环境中需实时适应的问题。通过测试时强化学习，提升模型的自适应能力与任务成功率。**

- **链接: [https://arxiv.org/pdf/2601.06748v1](https://arxiv.org/pdf/2601.06748v1)**

> **作者:** Changyu Liu; Yiyang Liu; Taowen Wang; Qiao Zhuang; James Chenhao Liang; Wenhao Yang; Renjing Xu; Qifan Wang; Dongfang Liu; Cheng Han
>
> **摘要:** Vision-Language-Action models have recently emerged as a powerful paradigm for general-purpose robot learning, enabling agents to map visual observations and natural-language instructions into executable robotic actions. Though popular, they are primarily trained via supervised fine-tuning or training-time reinforcement learning, requiring explicit fine-tuning phases, human interventions, or controlled data collection. Consequently, existing methods remain unsuitable for challenging simulated- or physical-world deployments, where robots must respond autonomously and flexibly to evolving environments. To address this limitation, we introduce a Test-Time Reinforcement Learning for VLAs (TT-VLA), a framework that enables on-the-fly policy adaptation during inference. TT-VLA formulates a dense reward mechanism that leverages step-by-step task-progress signals to refine action policies during test time while preserving the SFT/RL-trained priors, making it an effective supplement to current VLA models. Empirical results show that our approach enhances overall adaptability, stability, and task success in dynamic, previously unseen scenarios under simulated and real-world settings. We believe TT-VLA offers a principled step toward self-improving, deployment-ready VLAs.
>
---
#### [new 018] ObjSplat: Geometry-Aware Gaussian Surfels for Active Object Reconstruction
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出ObjSplat，用于主动物体重建任务，解决传统方法在几何复杂对象中重建不完整的问题。通过几何感知的视角评估和多步路径规划，提升重建精度与效率。**

- **链接: [https://arxiv.org/pdf/2601.06997v1](https://arxiv.org/pdf/2601.06997v1)**

> **作者:** Yuetao Li; Zhizhou Jia; Yu Zhang; Qun Hao; Shaohui Zhang
>
> **备注:** Project Page: https://li-yuetao.github.io/ObjSplat-page/
>
> **摘要:** Autonomous high-fidelity object reconstruction is fundamental for creating digital assets and bridging the simulation-to-reality gap in robotics. We present ObjSplat, an active reconstruction framework that leverages Gaussian surfels as a unified representation to progressively reconstruct unknown objects with both photorealistic appearance and accurate geometry. Addressing the limitations of conventional opacity or depth-based cues, we introduce a geometry-aware viewpoint evaluation pipeline that explicitly models back-face visibility and occlusion-aware multi-view covisibility, reliably identifying under-reconstructed regions even on geometrically complex objects. Furthermore, to overcome the limitations of greedy planning strategies, ObjSplat employs a next-best-path (NBP) planner that performs multi-step lookahead on a dynamically constructed spatial graph. By jointly optimizing information gain and movement cost, this planner generates globally efficient trajectories. Extensive experiments in simulation and on real-world cultural artifacts demonstrate that ObjSplat produces physically consistent models within minutes, achieving superior reconstruction fidelity and surface completeness while significantly reducing scan time and path length compared to state-of-the-art approaches. Project page: https://li-yuetao.github.io/ObjSplat-page/ .
>
---
#### [new 019] LOONG: Online Time-Optimal Autonomous Flight for MAVs in Cluttered Environments
- **分类: cs.RO**

- **简介: 该论文属于自主飞行任务，解决MAVs在复杂环境中高速、时间最优飞行的问题。提出集成规划与控制框架，结合模仿学习和模型预测控制，实现安全且激进的飞行。**

- **链接: [https://arxiv.org/pdf/2601.07434v1](https://arxiv.org/pdf/2601.07434v1)**

> **作者:** Xin Guan; Fangguo Zhao; Qianyi Wang; Chengcheng Zhao; Jiming Chen; Shuo Li
>
> **摘要:** Autonomous flight of micro air vehicles (MAVs) in unknown, cluttered environments remains challenging for time-critical missions due to conservative maneuvering strategies. This article presents an integrated planning and control framework for high-speed, time-optimal autonomous flight of MAVs in cluttered environments. In each replanning cycle (100 Hz), a time-optimal trajectory under polynomial presentation is generated as a reference, with the time-allocation process accelerated by imitation learning. Subsequently, a time-optimal model predictive contouring control (MPCC) incorporates safe flight corridor (SFC) constraints at variable horizon steps to enable aggressive yet safe maneuvering, while fully exploiting the MAV's dynamics. We validate the proposed framework extensively on a custom-built LiDAR-based MAV platform. Simulation results demonstrate superior aggressiveness compared to the state of the art, while real-world experiments achieve a peak speed of 18 m/s in a cluttered environment and succeed in 10 consecutive trials from diverse start points. The video is available at the following link: https://youtu.be/vexXXhv99oQ.
>
---
#### [new 020] FlyCo: Foundation Model-Empowered Drones for Autonomous 3D Structure Scanning in Open-World Environments
- **分类: cs.RO**

- **简介: 该论文提出FlyCo系统，用于解决开放环境中无人机自主3D结构扫描的问题。通过整合基础模型，实现高效、自适应的扫描任务。**

- **链接: [https://arxiv.org/pdf/2601.07558v1](https://arxiv.org/pdf/2601.07558v1)**

> **作者:** Chen Feng; Guiyong Zheng; Tengkai Zhuang; Yongqian Wu; Fangzhan He; Haojia Li; Juepeng Zheng; Shaojie Shen; Boyu Zhou
>
> **备注:** 34 pages, 24 figures, 9 tables. Video: https://www.youtube.com/playlist?list=PLqjZjnqsCyl40rw3y15Yzc7Mdo-z1y2j8
>
> **摘要:** Autonomous 3D scanning of open-world target structures via drones remains challenging despite broad applications. Existing paradigms rely on restrictive assumptions or effortful human priors, limiting practicality, efficiency, and adaptability. Recent foundation models (FMs) offer great potential to bridge this gap. This paper investigates a critical research problem: What system architecture can effectively integrate FM knowledge for this task? We answer it with FlyCo, a principled FM-empowered perception-prediction-planning loop enabling fully autonomous, prompt-driven 3D target scanning in diverse unknown open-world environments. FlyCo directly translates low-effort human prompts (text, visual annotations) into precise adaptive scanning flights via three coordinated stages: (1) perception fuses streaming sensor data with vision-language FMs for robust target grounding and tracking; (2) prediction distills FM knowledge and combines multi-modal cues to infer the partially observed target's complete geometry; (3) planning leverages predictive foresight to generate efficient and safe paths with comprehensive target coverage. Building on this, we further design key components to boost open-world target grounding efficiency and robustness, enhance prediction quality in terms of shape accuracy, zero-shot generalization, and temporal stability, and balance long-horizon flight efficiency with real-time computability and online collision avoidance. Extensive challenging real-world and simulation experiments show FlyCo delivers precise scene understanding, high efficiency, and real-time safety, outperforming existing paradigms with lower human effort and verifying the proposed architecture's practicality. Comprehensive ablations validate each component's contribution. FlyCo also serves as a flexible, extensible blueprint, readily leveraging future FM and robotics advances. Code will be released.
>
---
#### [new 021] HERE: Hierarchical Active Exploration of Radiance Field with Epistemic Uncertainty Minimization
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于3D场景重建任务，旨在解决如何高效获取数据并精确重建场景的问题。通过主动学习和不确定性量化，提升未探索区域的识别能力，实现更全面的重建。**

- **链接: [https://arxiv.org/pdf/2601.07242v1](https://arxiv.org/pdf/2601.07242v1)**

> **作者:** Taekbeom Lee; Dabin Kim; Youngseok Jang; H. Jin Kim
>
> **备注:** Accepted to IEEE RA-L. The first two authors contributed equally
>
> **摘要:** We present HERE, an active 3D scene reconstruction framework based on neural radiance fields, enabling high-fidelity implicit mapping. Our approach centers around an active learning strategy for camera trajectory generation, driven by accurate identification of unseen regions, which supports efficient data acquisition and precise scene reconstruction. The key to our approach is epistemic uncertainty quantification based on evidential deep learning, which directly captures data insufficiency and exhibits a strong correlation with reconstruction errors. This allows our framework to more reliably identify unexplored or poorly reconstructed regions compared to existing methods, leading to more informed and targeted exploration. Additionally, we design a hierarchical exploration strategy that leverages learned epistemic uncertainty, where local planning extracts target viewpoints from high-uncertainty voxels based on visibility for trajectory generation, and global planning uses uncertainty to guide large-scale coverage for efficient and comprehensive reconstruction. The effectiveness of the proposed method in active 3D reconstruction is demonstrated by achieving higher reconstruction completeness compared to previous approaches on photorealistic simulated scenes across varying scales, while a hardware demonstration further validates its real-world applicability.
>
---
#### [new 022] WaveMan: mmWave-Based Room-Scale Human Interaction Perception for Humanoid Robots
- **分类: cs.RO**

- **简介: 该论文提出WaveMan系统，解决家庭环境中人形机器人在非受限位置下的可靠交互问题，通过mmWave技术实现隐私保护和空间自适应感知。**

- **链接: [https://arxiv.org/pdf/2601.07454v1](https://arxiv.org/pdf/2601.07454v1)**

> **作者:** Yuxuan Hu; Kuangji Zuo; Boyu Ma; Shihao Li; Zhaoyang Xia; Feng Xu; Jianfei Yang
>
> **摘要:** Reliable humanoid-robot interaction (HRI) in household environments is constrained by two fundamental requirements, namely robustness to unconstrained user positions and preservation of user privacy. Millimeter-wave (mmWave) sensing inherently supports privacy-preserving interaction, making it a promising modality for room-scale HRI. However, existing mmWave-based interaction-sensing systems exhibit poor spatial generalization at unseen distances or viewpoints. To address this challenge, we introduce WaveMan, a spatially adaptive room-scale perception system that restores reliable human interaction sensing across arbitrary user positions. WaveMan integrates viewpoint alignment and spectrogram enhancement for spatial consistency, with dual-channel attention for robust feature extraction. Experiments across five participants show that, under fixed-position evaluation, WaveMan achieves the same cross-position accuracy as the baseline with five times fewer training positions. In random free-position testing, accuracy increases from 33.00% to 94.33%, enabled by the proposed method. These results demonstrate the feasibility of reliable, privacy-preserving interaction for household humanoid robots across unconstrained user positions.
>
---
#### [new 023] Large-Scale Autonomous Gas Monitoring for Volcanic Environments: A Legged Robot on Mount Etna
- **分类: cs.RO**

- **简介: 该论文属于自主气体监测任务，旨在解决火山环境中近地表气体测量的危险与困难。研究使用腿式机器人在埃特纳火山进行自主气体分析，提升监测效率与安全性。**

- **链接: [https://arxiv.org/pdf/2601.07362v1](https://arxiv.org/pdf/2601.07362v1)**

> **作者:** Julia Richter; Turcan Tuna; Manthan Patel; Takahiro Miki; Devon Higgins; James Fox; Cesar Cadena; Andres Diaz; Marco Hutter
>
> **备注:** 12 pages, 7 figures, submitted to IEEE Robotics & Automation Magazine (RAM)
>
> **摘要:** Volcanic gas emissions are key precursors of eruptive activity. Yet, obtaining accurate near-surface measurements remains hazardous and logistically challenging, motivating the need for autonomous solutions. Limited mobility in rough volcanic terrain has prevented wheeled systems from performing reliable in situ gas measurements, reducing their usefulness as sensing platforms. We present a legged robotic system for autonomous volcanic gas analysis, utilizing the quadruped ANYmal, equipped with a quadrupole mass spectrometer system. Our modular autonomy stack integrates a mission planning interface, global planner, localization framework, and terrain-aware local navigation. We evaluated the system on Mount Etna across three autonomous missions in varied terrain, achieving successful gas-source detections with autonomy rates of 93-100%. In addition, we conducted a teleoperated mission in which the robot measured natural fumaroles, detecting sulfur dioxide and carbon dioxide. We discuss lessons learned from the gas-analysis and autonomy perspectives, emphasizing the need for adaptive sensing strategies, tighter integration of global and local planning, and improved hardware design.
>
---
#### [new 024] Robust Evacuation for Multi-Drone Failure in Drone Light Shows
- **分类: cs.RO**

- **简介: 该论文属于多无人机系统可靠性任务，旨在解决无人机编队表演中多机故障导致的碰撞风险。通过预测故障无人机轨迹并规划避让路径，提升系统的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2601.06728v1](https://arxiv.org/pdf/2601.06728v1)**

> **作者:** Minhyuk Park; Aloysius K. Mok; Tsz-Chiu Au
>
> **备注:** Accepted to AAAI-26 Bridge Program B10: Making Embodied AI Reliable with Testing and Formal Verification
>
> **摘要:** Drone light shows have emerged as a popular form of entertainment in recent years. However, several high-profile incidents involving large-scale drone failures -- where multiple drones simultaneously fall from the sky -- have raised safety and reliability concerns. To ensure robustness, we propose a drone parking algorithm designed specifically for multiple drone failures in drone light shows, aimed at mitigating the risk of cascading collisions by drone evacuation and enabling rapid recovery from failures by leveraging strategically placed hidden drones. Our algorithm integrates a Social LSTM model with attention mechanisms to predict the trajectories of failing drones and compute near-optimal evacuation paths that minimize the likelihood of surviving drones being hit by fallen drones. In the recovery node, our system deploys hidden drones (operating with their LED lights turned off) to replace failed drones so that the drone light show can continue. Our experiments showed that our approach can greatly increase the robustness of a multi-drone system by leveraging deep learning to predict the trajectories of fallen drones.
>
---
#### [new 025] Deep Whole-body Parkour
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人形机器人控制任务，旨在解决传统方法在复杂地形中动态非移动任务的不足。通过整合感知与运动控制，实现高动态多接触动作。**

- **链接: [https://arxiv.org/pdf/2601.07701v1](https://arxiv.org/pdf/2601.07701v1)**

> **作者:** Ziwen Zhuang; Shaoting Zhu; Mengjie Zhao; Hang Zhao
>
> **摘要:** Current approaches to humanoid control generally fall into two paradigms: perceptive locomotion, which handles terrain well but is limited to pedal gaits, and general motion tracking, which reproduces complex skills but ignores environmental capabilities. This work unites these paradigms to achieve perceptive general motion control. We present a framework where exteroceptive sensing is integrated into whole-body motion tracking, permitting a humanoid to perform highly dynamic, non-locomotion tasks on uneven terrain. By training a single policy to perform multiple distinct motions across varied terrestrial features, we demonstrate the non-trivial benefit of integrating perception into the control loop. Our results show that this framework enables robust, highly dynamic multi-contact motions, such as vaulting and dive-rolling, on unstructured terrain, significantly expanding the robot's traversability beyond simple walking or running. https://project-instinct.github.io/deep-whole-body-parkour
>
---
#### [new 026] Stable In-hand Manipulation for a Lightweight Four-motor Prosthetic Hand
- **分类: cs.RO**

- **简介: 该论文属于康复机器人任务，旨在解决假肢手稳定抓取问题。通过电机电流反馈和手指协调，提升抓取稳定性，成功处理不同重量和形状的物体。**

- **链接: [https://arxiv.org/pdf/2601.07559v1](https://arxiv.org/pdf/2601.07559v1)**

> **作者:** Yuki Kuroda; Tomoya Takahashi; Cristian C. Beltran-Hernandez; Kazutoshi Tanaka; Masashi Hamaya
>
> **摘要:** Electric prosthetic hands should be lightweight to decrease the burden on the user, shaped like human hands for cosmetic purposes, and designed with motors enclosed inside to protect them from damage and dirt. Additionally, in-hand manipulation is necessary to perform daily activities such as transitioning between different postures, particularly through rotational movements, such as reorienting a pen into a writing posture after picking it up from a desk. We previously developed PLEXUS hand (Precision-Lateral dEXteroUS manipulation hand), a lightweight (311 g) prosthetic hand driven by four motors. This prosthetic performed reorientation between precision and lateral grasps with various objects. However, its controller required predefined object widths and was limited to handling lightweight objects (of weight up to 34 g). This study addresses these limitations by employing motor current feedback. Combined with the hand's previously optimized single-axis thumb, this approach achieves more stable manipulation by estimating the object's width and adjusting the index finger position to maintain stable object holding during the reorientation. Experimental validation using primitive objects of various widths (5-30 mm) and shapes (cylinders and prisms) resulted in a 100% success rate with lightweight objects and maintained a high success rate (>=80) even with heavy aluminum prisms (of weight up to 289 g). By contrast, the performance without index finger coordination dropped to just 40% on the heaviest 289 g prism. The hand also successfully executed several daily tasks, including closing bottle caps and orienting a pen for writing.
>
---
#### [new 027] Walk the PLANC: Physics-Guided RL for Agile Humanoid Locomotion on Constrained Footholds
- **分类: cs.RO**

- **简介: 该论文属于双足机器人步态规划任务，解决在有限落脚点上精准步态控制问题。通过结合物理引导的强化学习，提升机器人在复杂地形上的稳定性和可靠性。**

- **链接: [https://arxiv.org/pdf/2601.06286v1](https://arxiv.org/pdf/2601.06286v1)**

> **作者:** Min Dai; William D. Compton; Junheng Li; Lizhi Yang; Aaron D. Ames
>
> **摘要:** Bipedal humanoid robots must precisely coordinate balance, timing, and contact decisions when locomoting on constrained footholds such as stepping stones, beams, and planks -- even minor errors can lead to catastrophic failure. Classical optimization and control pipelines handle these constraints well but depend on highly accurate mathematical representations of terrain geometry, making them prone to error when perception is noisy or incomplete. Meanwhile, reinforcement learning has shown strong resilience to disturbances and modeling errors, yet end-to-end policies rarely discover the precise foothold placement and step sequencing required for discontinuous terrain. These contrasting limitations motivate approaches that guide learning with physics-based structure rather than relying purely on reward shaping. In this work, we introduce a locomotion framework in which a reduced-order stepping planner supplies dynamically consistent motion targets that steer the RL training process via Control Lyapunov Function (CLF) rewards. This combination of structured footstep planning and data-driven adaptation produces accurate, agile, and hardware-validated stepping-stone locomotion on a humanoid robot, substantially improving reliability compared to conventional model-free reinforcement-learning baselines.
>
---
#### [new 028] AdaMorph: Unified Motion Retargeting via Embodiment-Aware Adaptive Transformers
- **分类: cs.RO**

- **简介: 该论文属于机器人运动重定向任务，解决不同机器人形态间运动迁移问题。提出AdaMorph框架，通过统一模型适应多种机器人，提升泛化能力。**

- **链接: [https://arxiv.org/pdf/2601.07284v1](https://arxiv.org/pdf/2601.07284v1)**

> **作者:** Haoyu Zhang; Shibo Jin; Lvsong Li; Jun Li; Liang Lin; Xiaodong He; Zecui Zeng
>
> **摘要:** Retargeting human motion to heterogeneous robots is a fundamental challenge in robotics, primarily due to the severe kinematic and dynamic discrepancies between varying embodiments. Existing solutions typically resort to training embodiment-specific models, which scales poorly and fails to exploit shared motion semantics. To address this, we present AdaMorph, a unified neural retargeting framework that enables a single model to adapt human motion to diverse robot morphologies. Our approach treats retargeting as a conditional generation task. We map human motion into a morphology-agnostic latent intent space and utilize a dual-purpose prompting mechanism to condition the generation. Instead of simple input concatenation, we leverage Adaptive Layer Normalization (AdaLN) to dynamically modulate the decoder's feature space based on embodiment constraints. Furthermore, we enforce physical plausibility through a curriculum-based training objective that ensures orientation and trajectory consistency via integration. Experimental results on 12 distinct humanoid robots demonstrate that AdaMorph effectively unifies control across heterogeneous topologies, exhibiting strong zero-shot generalization to unseen complex motions while preserving the dynamic essence of the source behaviors.
>
---
#### [new 029] RSLCPP - Deterministic Simulations Using ROS 2
- **分类: cs.RO**

- **简介: 该论文属于机器人仿真任务，旨在解决ROS 2异步设计导致的不可复现问题。通过RSLCPP库实现确定性仿真，确保结果一致。**

- **链接: [https://arxiv.org/pdf/2601.07052v1](https://arxiv.org/pdf/2601.07052v1)**

> **作者:** Simon Sagmeister; Marcel Weinmann; Phillip Pitschi; Markus Lienkamp
>
> **备注:** Submitted to 'IEEE Robotics and Automation Practice' for possible publication
>
> **摘要:** Simulation is crucial in real-world robotics, offering safe, scalable, and efficient environments for developing applications, ranging from humanoid robots to autonomous vehicles and drones. While the Robot Operating System (ROS) has been widely adopted as the backbone of these robotic applications in both academia and industry, its asynchronous, multiprocess design complicates reproducibility, especially across varying hardware platforms. Deterministic callback execution cannot be guaranteed when computation times and communication delays vary. This lack of reproducibility complicates scientific benchmarking and continuous integration, where consistent results are essential. To address this, we present a methodology to create deterministic simulations using ROS 2 nodes. Our ROS Simulation Library for C++ (RSLCPP) implements this approach, enabling existing nodes to be combined into a simulation routine that yields reproducible results without requiring any code changes. We demonstrate that our approach yields identical results across various CPUs and architectures when testing both a synthetic benchmark and a real-world robotics system. RSLCPP is open-sourced at https://github.com/TUMFTM/rslcpp.
>
---
#### [new 030] Semantic Enrichment of CAD-Based Industrial Environments via Scene Graphs for Simulation and Reasoning
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于工业环境语义建模任务，旨在解决CAD文件缺乏语义信息的问题。通过构建3D场景图，增强功能元素的关联性，支持仿真与推理。**

- **链接: [https://arxiv.org/pdf/2601.06415v1](https://arxiv.org/pdf/2601.06415v1)**

> **作者:** Nathan Pascal Walus; Ranulfo Bezerra; Shotaro Kojima; Tsige Tadesse Alemayoh; Satoshi Tadokoro; Kazunori Ohno
>
> **备注:** Accepted to IEEE SSRR 2025
>
> **摘要:** Utilizing functional elements in an industrial environment, such as displays and interactive valves, provide effective possibilities for robot training. When preparing simulations for robots or applications that involve high-level scene understanding, the simulation environment must be equally detailed. Although CAD files for such environments deliver an exact description of the geometry and visuals, they usually lack semantic, relational and functional information, thus limiting the simulation and training possibilities. A 3D scene graph can organize semantic, spatial and functional information by enriching the environment through a Large Vision-Language Model (LVLM). In this paper we present an offline approach to creating detailed 3D scene graphs from CAD environments. This will serve as a foundation to include the relations of functional and actionable elements, which then can be used for dynamic simulation and reasoning. Key results of this research include both quantitative results of the generated semantic labels as well as qualitative results of the scene graph, especially in hindsight of pipe structures and identified functional relations. All code, results and the environment will be made available at https://cad-scenegraph.github.io
>
---
#### [new 031] Hiking in the Wild: A Scalable Perceptive Parkour Framework for Humanoids
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人运动控制任务，解决复杂环境中人形机器人稳健行走问题。提出一种可扩展的感知跳跃框架，结合安全机制与强化学习，实现高速稳定地形穿越。**

- **链接: [https://arxiv.org/pdf/2601.07718v1](https://arxiv.org/pdf/2601.07718v1)**

> **作者:** Shaoting Zhu; Ziwen Zhuang; Mengjie Zhao; Kun-Ying Lee; Hang Zhao
>
> **备注:** Project Page: https://project-instinct.github.io/hiking-in-the-wild
>
> **摘要:** Achieving robust humanoid hiking in complex, unstructured environments requires transitioning from reactive proprioception to proactive perception. However, integrating exteroception remains a significant challenge: mapping-based methods suffer from state estimation drift; for instance, LiDAR-based methods do not handle torso jitter well. Existing end-to-end approaches often struggle with scalability and training complexity; specifically, some previous works using virtual obstacles are implemented case-by-case. In this work, we present \textit{Hiking in the Wild}, a scalable, end-to-end parkour perceptive framework designed for robust humanoid hiking. To ensure safety and training stability, we introduce two key mechanisms: a foothold safety mechanism combining scalable \textit{Terrain Edge Detection} with \textit{Foot Volume Points} to prevent catastrophic slippage on edges, and a \textit{Flat Patch Sampling} strategy that mitigates reward hacking by generating feasible navigation targets. Our approach utilizes a single-stage reinforcement learning scheme, mapping raw depth inputs and proprioception directly to joint actions, without relying on external state estimation. Extensive field experiments on a full-size humanoid demonstrate that our policy enables robust traversal of complex terrains at speeds up to 2.5 m/s. The training and deployment code is open-sourced to facilitate reproducible research and deployment on real robots with minimal hardware modifications.
>
---
#### [new 032] Robotic Tele-Operation for Upper Aerodigestive Tract Microsurgery: System Design and Validation
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在解决UADT手术中手动操作力钳的局限性。设计了新型末端执行器和远程操作框架，提升手术精度与医生操作舒适度。**

- **链接: [https://arxiv.org/pdf/2601.06617v1](https://arxiv.org/pdf/2601.06617v1)**

> **作者:** Giovani Braglia; José Jair Alves Mendes Junior; Augusto Tetsuo Prado Inafuco; Federico Mariano; Leonardo S. Mattos
>
> **摘要:** Upper aerodigestive tract (UADT) treatments frequently employ transoral laser microsurgery (TLM) for procedures such as the removal of tumors or polyps. In TLM, a laser beam is used to cut target tissue, while forceps are employed to grasp, manipulate, and stabilize tissue within the UADT. Although TLM systems may rely on different technologies and interfaces, forceps manipulation is still predominantly performed manually, introducing limitations in ergonomics, precision, and controllability. This paper proposes a novel robotic system for tissue manipulation in UADT procedures, based on a novel end-effector designed for forceps control. The system is integrated within a teleoperation framework that employs a robotic manipulator with a programmed remote center of motion (RCM), enabling precise and constrained instrument motion while improving surgeon ergonomics. The proposed approach is validated through two experimental studies and a dedicated usability evaluation, demonstrating its effectiveness and suitability for UADT surgical applications.
>
---
#### [new 033] BlazeAIoT: A Modular Multi-Layer Platform for Real-Time Distributed Robotics Across Edge, Fog, and Cloud Infrastructures
- **分类: cs.RO; cs.DC**

- **简介: 该论文提出BlazeAIoT平台，解决分布式机器人在边缘、雾和云环境中的实时通信与计算问题。通过模块化架构实现动态服务分配与低延迟处理。**

- **链接: [https://arxiv.org/pdf/2601.06344v1](https://arxiv.org/pdf/2601.06344v1)**

> **作者:** Cedric Melancon; Julien Gascon-Samson; Maarouf Saad; Kuljeet Kaur; Simon Savard
>
> **备注:** 17 pages, 9 figures
>
> **摘要:** The increasing complexity of distributed robotics has driven the need for platforms that seamlessly integrate edge, fog, and cloud computing layers while meeting strict real-time constraints. This paper introduces BlazeAIoT, a modular multi-layer platform designed to unify distributed robotics across heterogeneous infrastructures. BlazeAIoT provides dynamic data transfer, configurable services, and integrated monitoring, while ensuring resilience, security, and programming language flexibility. The architecture leverages Kubernetes-based clusters, broker interoperability (DDS, Kafka, Redis, and ROS2), and adaptive data distribution mechanisms to optimize communication and computation across diverse environments. The proposed solution includes a multi-layer configuration service, dynamic and adaptive data bridging, and hierarchical rate limiting to handle large messages. The platform is validated through robotics scenarios involving navigation and artificial intelligence-driven large-scale message processing, demonstrating robust performance under real-time constraints. Results highlight BlazeAIoT's ability to dynamically allocate services across incomplete topologies, maintain system health, and minimize latency, making it a cost-aware, scalable solution for robotics and broader IoT applications, such as smart cities and smart factories.
>
---
#### [new 034] WHU-PCPR: A cross-platform heterogeneous point cloud dataset for place recognition in complex urban scenes
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出WHU-PCPR数据集，用于复杂城市场景下的位置识别任务，解决现有数据集多样性不足的问题，通过多平台、多传感器采集数据进行评估与分析。**

- **链接: [https://arxiv.org/pdf/2601.06442v1](https://arxiv.org/pdf/2601.06442v1)**

> **作者:** Xianghong Zou; Jianping Li; Yandi Yang; Weitong Wu; Yuan Wang; Qiegen Liu; Zhen Dong
>
> **摘要:** Point Cloud-based Place Recognition (PCPR) demonstrates considerable potential in applications such as autonomous driving, robot localization and navigation, and map update. In practical applications, point clouds used for place recognition are often acquired from different platforms and LiDARs across varying scene. However, existing PCPR datasets lack diversity in scenes, platforms, and sensors, which limits the effective development of related research. To address this gap, we establish WHU-PCPR, a cross-platform heterogeneous point cloud dataset designed for place recognition. The dataset differentiates itself from existing datasets through its distinctive characteristics: 1) cross-platform heterogeneous point clouds: collected from survey-grade vehicle-mounted Mobile Laser Scanning (MLS) systems and low-cost Portable helmet-mounted Laser Scanning (PLS) systems, each equipped with distinct mechanical and solid-state LiDAR sensors. 2) Complex localization scenes: encompassing real-time and long-term changes in both urban and campus road scenes. 3) Large-scale spatial coverage: featuring 82.3 km of trajectory over a 60-month period and an unrepeated route of approximately 30 km. Based on WHU-PCPR, we conduct extensive evaluation and in-depth analysis of several representative PCPR methods, and provide a concise discussion of key challenges and future research directions. The dataset and benchmark code are available at https://github.com/zouxianghong/WHU-PCPR.
>
---
#### [new 035] SpatialNav: Leveraging Spatial Scene Graphs for Zero-Shot Vision-and-Language Navigation
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于视觉与语言导航任务，旨在解决零样本导航中因缺乏训练数据导致的效率低和性能差问题。通过构建空间场景图，提出SpatialNav模型提升导航能力。**

- **链接: [https://arxiv.org/pdf/2601.06806v1](https://arxiv.org/pdf/2601.06806v1)**

> **作者:** Jiwen Zhang; Zejun Li; Siyuan Wang; Xiangyu Shi; Zhongyu Wei; Qi Wu
>
> **备注:** 11 pages, 4 figures, 6 tables
>
> **摘要:** Although learning-based vision-and-language navigation (VLN) agents can learn spatial knowledge implicitly from large-scale training data, zero-shot VLN agents lack this process, relying primarily on local observations for navigation, which leads to inefficient exploration and a significant performance gap. To deal with the problem, we consider a zero-shot VLN setting that agents are allowed to fully explore the environment before task execution. Then, we construct the Spatial Scene Graph (SSG) to explicitly capture global spatial structure and semantics in the explored environment. Based on the SSG, we introduce SpatialNav, a zero-shot VLN agent that integrates an agent-centric spatial map, a compass-aligned visual representation, and a remote object localization strategy for efficient navigation. Comprehensive experiments in both discrete and continuous environments demonstrate that SpatialNav significantly outperforms existing zero-shot agents and clearly narrows the gap with state-of-the-art learning-based methods. Such results highlight the importance of global spatial representations for generalizable navigation.
>
---
#### [new 036] TranSC: Hardware-Aware Design of Transcendental Functions Using Stochastic Logic
- **分类: cs.ET; cs.RO; eess.SY**

- **简介: 该论文属于硬件设计任务，解决 transcendental 函数高效实现问题。通过 stochastic computing 和 quasi-random 序列，提升计算精度与效率，降低硬件成本。**

- **链接: [https://arxiv.org/pdf/2601.07172v1](https://arxiv.org/pdf/2601.07172v1)**

> **作者:** Mehran Moghadam; Sercan Aygun; M. Hassan Najafi
>
> **备注:** 12 pages
>
> **摘要:** The hardware-friendly implementation of transcendental functions remains a longstanding challenge in design automation. These functions, which cannot be expressed as finite combinations of algebraic operations, pose significant complexity in digital circuit design. This study introduces a novel approach, TranSC, that utilizes stochastic computing (SC) for lightweight yet accurate implementation of transcendental functions. Building on established SC techniques, our method explores alternative random sources-specifically, quasi-random Van der Corput low-discrepancy (LD) sequences-instead of conventional pseudo-randomness. This shift enhances both the accuracy and efficiency of SC-based computations. We validate our approach through extensive experiments on various function types, including trigonometric, hyperbolic, and activation functions. The proposed design approach significantly reduces MSE by up to 98% compared to the state-of-the-art solutions while reducing hardware area, power consumption, and energy usage by 33%, 72%, and 64%, respectively.
>
---
#### [new 037] Affordable Data Collection System for UAVs Taxi Vibration Testing
- **分类: eess.SY; cs.RO; eess.SP**

- **简介: 该论文属于航空航天领域，旨在解决高成本数据采集系统的问题。通过设计低成本的DAQ系统，实现对无人机振动的有效测试与分析。**

- **链接: [https://arxiv.org/pdf/2601.07783v1](https://arxiv.org/pdf/2601.07783v1)**

> **作者:** Chaoyi Lin Yang; Gabriele Dessena; Oscar E. Bonilla-Manrique
>
> **摘要:** Structural vibration testing plays a key role in aerospace engineering for evaluating dynamic behaviour, ensuring reliability and verifying structural integrity. These tests rely on accurate and robust data acquisition systems (DAQ) to capture high-quality acceleration data. However, commercial DAQs that provide the required performance and features are often expensive and complex, limiting their accessibility for small-scale research and experimental applications. This work presents the design and experimental validation of an affordable and in-house-developed acceleration DAQ, tested on a small fixed-wing UAV through several Taxi Vibration Test (TVT) runs and ambient vibration measurements. The proposed system integrates several OrangePi 3 LTS single-board computers with multiple LSM6DS3TR-C MEMS inertial measurement units operating simultaneously via an Inter-Integrated Circuit (I2C) communication interface, managed under a Python-based master/slave architecture. Data is acquired at a stable sampling rate of approximately 208 Hz and post-processed using Welch's method to estimate their Power Spectral Density (PSD). Results confirm the system ability to provide consistent multi-sensor acceleration data and repeatable PSD profiles under the same test conditions; thus, demonstrating its reliability. With a total hardware cost below 600 EUR (approximately 690 USD), the developed DAQ offers a compact, scalable and cost-effective alternative for aerospace vibration analysis and structural testing.
>
---
#### [new 038] NAS-GS: Noise-Aware Sonar Gaussian Splatting
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于 underwater 3D重建与视图合成任务，针对声呐图像的噪声和缺乏高度信息问题，提出NAS-GS框架，结合双方向投影和GMM噪声模型提升重建精度与图像真实性。**

- **链接: [https://arxiv.org/pdf/2601.06285v1](https://arxiv.org/pdf/2601.06285v1)**

> **作者:** Shida Xu; Jingqi Jiang; Jonatan Scharff Willners; Sen Wang
>
> **摘要:** Underwater sonar imaging plays a crucial role in various applications, including autonomous navigation in murky water, marine archaeology, and environmental monitoring. However, the unique characteristics of sonar images, such as complex noise patterns and the lack of elevation information, pose significant challenges for 3D reconstruction and novel view synthesis. In this paper, we present NAS-GS, a novel Noise-Aware Sonar Gaussian Splatting framework specifically designed to address these challenges. Our approach introduces a Two-Ways Splatting technique that accurately models the dual directions for intensity accumulation and transmittance calculation inherent in sonar imaging, significantly improving rendering speed without sacrificing quality. Moreover, we propose a Gaussian Mixture Model (GMM) based noise model that captures complex sonar noise patterns, including side-lobes, speckle, and multi-path noise. This model enhances the realism of synthesized images while preventing 3D Gaussian overfitting to noise, thereby improving reconstruction accuracy. We demonstrate state-of-the-art performance on both simulated and real-world large-scale offshore sonar scenarios, achieving superior results in novel view synthesis and 3D reconstruction.
>
---
#### [new 039] Predefined-time One-Shot Cooperative Estimation, Guidance, and Control for Simultaneous Target Interception
- **分类: eess.SY; cs.MA; cs.RO; math.DS**

- **简介: 该论文属于协同拦截任务，解决多智能体在部分可观测条件下同时击中目标的问题。设计了统一的估计-制导-控制框架，实现快速、精确的协同拦截。**

- **链接: [https://arxiv.org/pdf/2601.07744v1](https://arxiv.org/pdf/2601.07744v1)**

> **作者:** Lohitvel Gopikannan; Shashi Ranjan Kumar; Abhinav Sinha
>
> **摘要:** This work develops a unified nonlinear estimation-guidance-control framework for cooperative simultaneous interception of a stationary target under a heterogeneous sensing topology, where sensing capabilities are non-uniform across interceptors. Specifically, only a subset of agents is instrumented with onboard seekers (informed/seeker-equipped agents), whereas the rest of them (seeker-less agents) acquire the information about the target indirectly via the informed agents and execute a distributed cooperative guidance for simultaneous target interception. To address the resulting partial observability, a predefined-time distributed observer is leveraged, guaranteeing convergence of the target state estimates for seeker-less agents through information exchange with seeker-equipped neighbors over a directed communication graph. Thereafter, an improved time-to-go estimate accounting for wide launch envelopes is utilized to design the distributed cooperative guidance commands. This estimate is coupled with a predefined-time consensus protocol, ensuring consensus in the agents' time-to-go values. The temporal upper bounds within which both observer error and time-to-go consensus error converge to zero can be prescribed as design parameters. Furthermore, the cooperative guidance commands are realized by means of an autopilot, wherein the interceptor is steered by canard actuation. The corresponding fin deflection commands are generated using a predefined-time convergent sliding mode control law. This enables the autopilot to precisely track the commanded lateral acceleration within a design-specified time, while maintaining non-singularity of the overall design. Theoretical guarantees are supported by numerical simulations across diverse engagement geometries, verifying the estimation accuracy, the cooperative interception performance, and the autopilot response using the proposed scheme.
>
---
#### [new 040] First Multi-Constellation Observations of Navigation Satellite Signals in the Lunar Domain by Post-Processing L1/L5 IQ Snapshots
- **分类: physics.space-ph; astro-ph.IM; cs.RO; eess.SP**

- **简介: 该论文属于月球导航任务，旨在评估多星座卫星信号在月球域的可用性。通过处理LuGRE接收器的IQ数据，验证了多系统信号在远距离下的可检测性，提升了月球任务的自主性。**

- **链接: [https://arxiv.org/pdf/2601.06081v1](https://arxiv.org/pdf/2601.06081v1)**

> **作者:** Lorenzo Sciacca; Alex Minetto; Andrea Nardin; Fabio Dovis; Luca Canzian; Mario Musmeci; Claudia Facchinetti; Giancarlo Varacalli
>
> **备注:** 13 pages, 9 figures, IEEE Transactions on Aerospace and Electronic Systems
>
> **摘要:** The use of Global Navigation Satellite Systems (GNSS) to increase spacecraft autonomy for orbit determination has gained renewed momentum following the Lunar GNSS Receiver Experiment (LuGRE), which demonstrated feasible onboard GPS and Galileo signal reception and tracking at lunar distances. This work processes in-phase and quadrature (IQ) snapshots collected by the LuGRE receiver in cis-lunar space and on the lunar surface to assess multi-frequency, multi-constellation signal availability. Signals from additional systems beyond GPS and Galileo, including RNSS and SBAS constellations, are observable and successfully acquired exclusively in the recorded IQ snapshots. These observations provide the first experimental evidence that signals from multiple constellations, including systems not supported by LuGRE realtime operations, are detectable at unprecedented distances from Earth. Useful observables can be extracted from the IQ snapshots, despite minimal sampling rates, 4-bit quantization, and short durations (200 ms-2 s), through a hybrid coherent/non-coherent acquisition stage compensating for code Doppler. These observations are exploited to tune simulation tools and to perform extended simulation campaigns, showing that the inclusion of additional constellations significantly improves availability; for a 26 dB-Hz acquisition threshold, the fraction of epochs with at least four visible satellites increases from 11% to 46% of the total epoch count. These findings indicate that BeiDou, RNSS, and SBAS signals can substantially enhance GNSS-based autonomy for lunar and cislunar missions.
>
---
#### [new 041] Object-Centric World Models Meet Monte Carlo Tree Search
- **分类: cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决动态环境建模问题。通过引入ObjectZero算法，利用对象级表示和图神经网络，提升环境建模效果，并结合蒙特卡洛树搜索进行有效决策。**

- **链接: [https://arxiv.org/pdf/2601.06604v1](https://arxiv.org/pdf/2601.06604v1)**

> **作者:** Rodion Vakhitov; Leonid Ugadiarov; Aleksandr Panov
>
> **摘要:** In this paper, we introduce ObjectZero, a novel reinforcement learning (RL) algorithm that leverages the power of object-level representations to model dynamic environments more effectively. Unlike traditional approaches that process the world as a single undifferentiated input, our method employs Graph Neural Networks (GNNs) to capture intricate interactions among multiple objects. These objects, which can be manipulated and interact with each other, serve as the foundation for our model's understanding of the environment. We trained the algorithm in a complex setting teeming with diverse, interactive objects, demonstrating its ability to effectively learn and predict object dynamics. Our results highlight that a structured world model operating on object-centric representations can be successfully integrated into a model-based RL algorithm utilizing Monte Carlo Tree Search as a planning module.
>
---
#### [new 042] Visible Light Communication using Led-Based AR Markers for Robot Localization
- **分类: cs.IT; cs.RO; eess.IV**

- **简介: 论文提出一种基于LED的AR标记方法，用于机器人定位。该任务旨在解决传统视觉标记在人机协作环境中不够自然的问题。通过LED闪烁频率编码信息，实现隐蔽的通信与定位。**

- **链接: [https://arxiv.org/pdf/2601.06527v1](https://arxiv.org/pdf/2601.06527v1)**

> **作者:** Wataru Uemura; Shogo Kawasaki
>
> **摘要:** A method of information transmission using visual markers has been widely studied. In this approach, information or identifiers (IDs) are encoded in the black-and-white pattern of each marker. By analyzing the geometric properties of the marker frame - such as its size, distortion, and coordinates - the relative position and orientation between the camera and the marker can be estimated. Furthermore, by associating the positional information of each marker with its corresponding ID, the position of the camera that takes the image picture can be calculated. In the field of mobile robotics, such markers are commonly utilized for robot localization. As mobile robots become more widely used in everyday environments, such visual markers are expected to be utilized across various contexts. In environments where robots collaborate with humans - such as in cell-based manufacturing systems in factories or in domestic settings with partner robots - it is desirable for such markers to be designed in a manner that appears natural and unobtrusive to humans. In this paper, we propose a method for implementing an ArUco marker in the form of illumination. In the proposed method, LEDs are arranged in accordance with the grid pattern of the marker, and the blinking frequency of each LED is determined based on the corresponding black or white cell. As a result, the illumination appears uniformly bright to the human eye, while the camera can capture variations in the blinking frequency. From these differences, the black-and-white pattern can be reconstructed, enabling the identification of the marker's tag information. We develop a prototype system, and conduct experiments which are conducted to evaluate its performance in terms of recognition accuracy under varying distances and viewing angles with respect to the ArUco marker.
>
---
#### [new 043] Benchmarking Autonomy in Scientific Experiments: A Hierarchical Taxonomy for Autonomous Large-Scale Facilities
- **分类: physics.ins-det; cond-mat.mtrl-sci; cs.AI; cs.RO**

- **简介: 该论文提出BASE Scale，解决大型科学设施自主性评估问题，建立六级分类体系，适应零样本部署需求。**

- **链接: [https://arxiv.org/pdf/2601.06978v1](https://arxiv.org/pdf/2601.06978v1)**

> **作者:** James Le Houx
>
> **备注:** 12 pages, 2 figures, 2 tables
>
> **摘要:** The transition from automated data collection to fully autonomous discovery requires a shared vocabulary to benchmark progress. While the automotive industry relies on the SAE J3016 standard, current taxonomies for autonomous science presuppose an owner-operator model that is incompatible with the operational rigidities of Large-Scale User Facilities. Here, we propose the Benchmarking Autonomy in Scientific Experiments (BASE) Scale, a 6-level taxonomy (Levels 0-5) specifically adapted for these unique constraints. Unlike owner-operator models, User Facilities require zero-shot deployment where agents must operate immediately without extensive training periods. We define the specific technical requirements for each tier, identifying the Inference Barrier (Level 3) as the critical latency threshold where decisions shift from scalar feedback to semantic digital twins. Fundamentally, this level extends the decision manifold from spatial exploration to temporal gating, enabling the agent to synchronise acquisition with the onset of transient physical events. By establishing these operational definitions, the BASE Scale provides facility directors, funding bodies, and beamline scientists with a standardised metric to assess risk, define liability, and quantify the intelligence of experimental workflows.
>
---
#### [new 044] Aggregating swarms through morphology handling design contingencies: from the sweet spot to a rich expressivity
- **分类: cond-mat.soft; cs.RO**

- **简介: 该论文研究群体机器人在光照任务中的行为，探讨形态设计与策略对集体行为的影响。旨在解决如何通过物理设计优化群体协作的问题。工作包括实验与仿真，分析不同对齐强度下的表现。**

- **链接: [https://arxiv.org/pdf/2601.07610v1](https://arxiv.org/pdf/2601.07610v1)**

> **作者:** Jeremy Fersula; Nicolas Bredeche; Olivier Dauchot
>
> **备注:** 8 pages, 3 figures
>
> **摘要:** Morphological computing, the use of the physical design of a robot to ease the realization of a given task has been proven to be a relevant concept in the context of swarm robotics. Here we demonstrate both experimentally and numerically, that the success of such a strategy may heavily rely on the type of policy adopted by the robots, as well as on the details of the physical design. To do so, we consider a swarm of robots, composed of Kilobots embedded in an exoskeleton, the design of which controls the propensity of the robots to align or anti-align with the direction of the external force they experience. We find experimentally that the contrast that was observed between the two morphologies in the success rate of a simple phototactic task, where the robots were programmed to stop when entering a light region, becomes dramatic, if the robots are not allowed to stop, and can only slow down. Building on a faithful physical model of the self-aligning dynamics of the robots, we perform numerical simulations and demonstrate on one hand that a precise tuning of the self-aligning strength around a sweet spot is required to achieve an efficient phototactic behavior, on the other hand that exploring a range of self-alignment strength allows for a rich expressivity of collective behaviors.
>
---
#### [new 045] OSCAR: Open-Set CAD Retrieval from a Language Prompt and a Single Image
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出OSCAR，解决开放集CAD模型检索问题，通过语言提示和单张图像匹配3D模型，提升6D物体姿态估计的效率与准确性。**

- **链接: [https://arxiv.org/pdf/2601.07333v1](https://arxiv.org/pdf/2601.07333v1)**

> **作者:** Tessa Pulli; Jean-Baptiste Weibel; Peter Hönig; Matthias Hirschmanner; Markus Vincze; Andreas Holzinger
>
> **摘要:** 6D object pose estimation plays a crucial role in scene understanding for applications such as robotics and augmented reality. To support the needs of ever-changing object sets in such context, modern zero-shot object pose estimators were developed to not require object-specific training but only rely on CAD models. Such models are hard to obtain once deployed, and a continuously changing and growing set of objects makes it harder to reliably identify the instance model of interest. To address this challenge, we introduce an Open-Set CAD Retrieval from a Language Prompt and a Single Image (OSCAR), a novel training-free method that retrieves a matching object model from an unlabeled 3D object database. During onboarding, OSCAR generates multi-view renderings of database models and annotates them with descriptive captions using an image captioning model. At inference, GroundedSAM detects the queried object in the input image, and multi-modal embeddings are computed for both the Region-of-Interest and the database captions. OSCAR employs a two-stage retrieval: text-based filtering using CLIP identifies candidate models, followed by image-based refinement using DINOv2 to select the most visually similar object. In our experiments we demonstrate that OSCAR outperforms all state-of-the-art methods on the cross-domain 3D model retrieval benchmark MI3DOR. Furthermore, we demonstrate OSCAR's direct applicability in automating object model sourcing for 6D object pose estimation. We propose using the most similar object model for pose estimation if the exact instance is not available and show that OSCAR achieves an average precision of 90.48\% during object retrieval on the YCB-V object dataset. Moreover, we demonstrate that the most similar object model can be utilized for pose estimation using Megapose achieving better results than a reconstruction-based approach.
>
---
#### [new 046] Video Generation Models in Robotics - Applications, Research Challenges, Future Directions
- **分类: eess.SY; cs.RO**

- **简介: 本文综述视频生成模型在机器人学中的应用，解决物理世界建模与交互问题，涵盖数据生成、动作预测、策略评估等任务。**

- **链接: [https://arxiv.org/pdf/2601.07823v1](https://arxiv.org/pdf/2601.07823v1)**

> **作者:** Zhiting Mei; Tenny Yin; Ola Shorinwa; Apurva Badithela; Zhonghe Zheng; Joseph Bruno; Madison Bland; Lihan Zha; Asher Hancock; Jaime Fernández Fisac; Philip Dames; Anirudha Majumdar
>
> **摘要:** Video generation models have emerged as high-fidelity models of the physical world, capable of synthesizing high-quality videos capturing fine-grained interactions between agents and their environments conditioned on multi-modal user inputs. Their impressive capabilities address many of the long-standing challenges faced by physics-based simulators, driving broad adoption in many problem domains, e.g., robotics. For example, video models enable photorealistic, physically consistent deformable-body simulation without making prohibitive simplifying assumptions, which is a major bottleneck in physics-based simulation. Moreover, video models can serve as foundation world models that capture the dynamics of the world in a fine-grained and expressive way. They thus overcome the limited expressiveness of language-only abstractions in describing intricate physical interactions. In this survey, we provide a review of video models and their applications as embodied world models in robotics, encompassing cost-effective data generation and action prediction in imitation learning, dynamics and rewards modeling in reinforcement learning, visual planning, and policy evaluation. Further, we highlight important challenges hindering the trustworthy integration of video models in robotics, which include poor instruction following, hallucinations such as violations of physics, and unsafe content generation, in addition to fundamental limitations such as significant data curation, training, and inference costs. We present potential future directions to address these open research challenges to motivate research and ultimately facilitate broader applications, especially in safety-critical settings.
>
---
#### [new 047] FMAC: a Fair Fiducial Marker Accuracy Comparison Software
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于视觉定位任务，旨在公平比较 fiducial 标记的位姿估计精度。通过生成高保真合成图像，分析六自由度误差关联，验证标记性能。**

- **链接: [https://arxiv.org/pdf/2601.07723v1](https://arxiv.org/pdf/2601.07723v1)**

> **作者:** Guillaume J. Laurent; Patrick Sandoz
>
> **摘要:** This paper presents a method for carrying fair comparisons of the accuracy of pose estimation using fiducial markers. These comparisons rely on large sets of high-fidelity synthetic images enabling deep exploration of the 6 degrees of freedom. A low-discrepancy sampling of the space allows to check the correlations between each degree of freedom and the pose errors by plotting the 36 pairs of combinations. The images are rendered using a physically based ray tracing code that has been specifically developed to use the standard calibration coefficients of any camera directly. The software reproduces image distortions, defocus and diffraction blur. Furthermore, sub-pixel sampling is applied to sharp edges to enhance the fidelity of the rendered image. After introducing the rendering algorithm and its experimental validation, the paper proposes a method for evaluating the pose accuracy. This method is applied to well-known markers, revealing their strengths and weaknesses for pose estimation. The code is open source and available on GitHub.
>
---
#### [new 048] Robust maximum hands-off optimal control: existence, maximum principle, and $L^{0}$-$L^1$ equivalence
- **分类: math.OC; cs.RO; eess.SY; math.NA**

- **简介: 该论文属于最优控制任务，解决不确定线性系统的稀疏控制问题。通过引入鲁棒最大放手原则，证明了$L^0$与$L^1$方法的等价性，并提出有效算法。**

- **链接: [https://arxiv.org/pdf/2601.07256v1](https://arxiv.org/pdf/2601.07256v1)**

> **作者:** Siddhartha Ganguly; Kenji Kashima
>
> **备注:** Revised version of a journal submission; comments are welcome
>
> **摘要:** This work advances the maximum hands-off sparse control framework by developing a robust counterpart for constrained linear systems with parametric uncertainties. The resulting optimal control problem minimizes an $L^{0}$ objective subject to an uncountable, compact family of constraints, and is therefore a nonconvex, nonsmooth robust optimization problem. To address this, we replace the $L^{0}$ objective with its convex $L^{1}$ surrogate and, using a nonsmooth variant of the robust Pontryagin maximum principle, show that the $L^{0}$ and $L^{1}$ formulations have identical sets of optimal solutions -- we call this the robust hands-off principle. Building on this equivalence, we propose an algorithmic framework -- drawing on numerically viable techniques from the semi-infinite robust optimization literature -- to solve the resulting problems. An illustrative example is provided to demonstrate the effectiveness of the approach.
>
---
#### [new 049] Optimizing the Design of a Simple Three-Sphere Magnetic Microswimmer
- **分类: cond-mat.soft; cs.RO; physics.app-ph; physics.flu-dyn; physics.med-ph**

- **简介: 该论文属于微机器人设计任务，旨在解决低雷诺数下实现定向运动的问题。通过优化三磁珠微泳器结构和磁场参数，实现高效可控的微尺度游泳。**

- **链接: [https://arxiv.org/pdf/2601.07370v1](https://arxiv.org/pdf/2601.07370v1)**

> **作者:** Theo Lequy; Andreas M. Menzel
>
> **备注:** 10 pages, 7 figures
>
> **摘要:** When swimming at low Reynolds numbers, inertial effects are negligible and reciprocal movements cannot induce net motion. Instead, symmetry breaking is necessary to achieve net propulsion. Directed swimming can be supported by magnetic fields, which simultaneously provide a versatile means of remote actuation. Thus, we analyze the motion of a straight microswimmer composed of three magnetizable beads connected by two elastic links. The swimming mechanism is based on oriented external magnetic fields that oscillate in magnitude. Through induced reversible hysteretic collapse of the two segments of the swimmer, the two pairs of beads jump into contact and separate nonreciprocally. Due to higher-order hydrodynamic interactions, net displacement results after each cycle. Different microswimmers can be tuned to different driving amplitudes and frequencies, allowing for simultaneous independent control by just one external magnetic field. The swimmer geometry and magnetic field shape are optimized for maximum swimming speed using an evolutionary optimization strategy. Thanks to the simple working principle, an experimental realization of such a microrobot seems feasible and may open new approaches for microinvasive medical interventions such as targeted drug delivery.
>
---
#### [new 050] Low-Back Pain Physical Rehabilitation by Movement Analysis in Clinical Trial
- **分类: cs.CV; cs.HC; cs.RO**

- **简介: 该论文属于康复运动分析任务，旨在解决康复训练中的运动评估、错误识别等问题。提出Keraal数据集，用于智能辅导系统的开发与评估。**

- **链接: [https://arxiv.org/pdf/2601.06138v1](https://arxiv.org/pdf/2601.06138v1)**

> **作者:** Sao Mai Nguyen
>
> **备注:** ICMST, Tokyo University of Science; Taiwanese Society of Movement Science and Technology; Research institute for Science and Technology, Nov 2025, Tokyo, Japan
>
> **摘要:** To allow the development and assessment of physical rehabilitation by an intelligent tutoring system, we propose a medical dataset of clinical patients carrying out low back-pain rehabilitation exercises and benchmark on state of the art human movement analysis algorithms. This dataset is valuable because it includes rehabilitation motions in a clinical setting with patients in their rehabilitation program. This paper introduces the Keraal dataset, a clinically collected dataset to enable intelligent tutoring systems (ITS) for rehabilitation. It addresses four challenges in exercise monitoring: motion assessment, error recognition, spatial localization, temporal localization
>
---
#### [new 051] A Review of Online Diffusion Policy RL Algorithms for Scalable Robotic Control
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决在线扩散策略强化学习的集成难题。通过分类与实验分析，提出算法选择指南并指出未来研究方向。**

- **链接: [https://arxiv.org/pdf/2601.06133v1](https://arxiv.org/pdf/2601.06133v1)**

> **作者:** Wonhyeok Choi; Minwoo Choi; Jungwan Woo; Kyumin Hwang; Jaeyeul Kim; Sunghoon Im
>
> **摘要:** Diffusion policies have emerged as a powerful approach for robotic control, demonstrating superior expressiveness in modeling multimodal action distributions compared to conventional policy networks. However, their integration with online reinforcement learning remains challenging due to fundamental incompatibilities between diffusion model training objectives and standard RL policy improvement mechanisms. This paper presents the first comprehensive review and empirical analysis of current Online Diffusion Policy Reinforcement Learning (Online DPRL) algorithms for scalable robotic control systems. We propose a novel taxonomy that categorizes existing approaches into four distinct families -- Action-Gradient, Q-Weighting, Proximity-Based, and Backpropagation Through Time (BPTT) methods -- based on their policy improvement mechanisms. Through extensive experiments on a unified NVIDIA Isaac Lab benchmark encompassing 12 diverse robotic tasks, we systematically evaluate representative algorithms across five critical dimensions: task diversity, parallelization capability, diffusion step scalability, cross-embodiment generalization, and environmental robustness. Our analysis identifies key findings regarding the fundamental trade-offs inherent in each algorithmic family, particularly concerning sample efficiency and scalability. Furthermore, we reveal critical computational and algorithmic bottlenecks that currently limit the practical deployment of online DPRL. Based on these findings, we provide concrete guidelines for algorithm selection tailored to specific operational constraints and outline promising future research directions to advance the field toward more general and scalable robotic learning systems.
>
---
## 更新

#### [replaced 001] Navigation Around Unknown Space Objects Using Visible-Thermal Image Fusion
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于空间导航任务，旨在解决未知目标物体周围精确定位问题。通过融合可见光与热红外图像，提升SLAM算法在复杂光照条件下的导航性能。**

- **链接: [https://arxiv.org/pdf/2512.12203v2](https://arxiv.org/pdf/2512.12203v2)**

> **作者:** Eric J. Elias; Michael Esswein; Jonathan P. How; David W. Miller
>
> **备注:** 18 pages, 11 figures. To be published in proceedings of AIAA SCITECH 2026 Forum
>
> **摘要:** As the popularity of on-orbit operations grows, so does the need for precise navigation around unknown resident space objects (RSOs) such as other spacecraft, orbital debris, and asteroids. The use of Simultaneous Localization and Mapping (SLAM) algorithms is often studied as a method to map out the surface of an RSO and find the inspector's relative pose using a lidar or conventional camera. However, conventional cameras struggle during eclipse or shadowed periods, and lidar, though robust to lighting conditions, tends to be heavier, bulkier, and more power-intensive. Thermal-infrared cameras can track the target RSO throughout difficult illumination conditions without these limitations. While useful, thermal-infrared imagery lacks the resolution and feature-richness of visible cameras. In this work, images of a target satellite in low Earth orbit are photo-realistically simulated in both visible and thermal-infrared bands. Pixel-level fusion methods are used to create visible/thermal-infrared composites that leverage the best aspects of each camera. Navigation errors from a monocular SLAM algorithm are compared between visible, thermal-infrared, and fused imagery in various lighting and trajectories. Fused imagery yields substantially improved navigation performance over visible-only and thermal-only methods.
>
---
#### [replaced 002] What Is The Best 3D Scene Representation for Robotics? From Geometric to Foundation Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人3D场景表示研究，旨在比较不同表示方法的优劣，探索最佳方案。工作包括分类核心模块、分析方法特点，并展望未来发展方向。**

- **链接: [https://arxiv.org/pdf/2512.03422v2](https://arxiv.org/pdf/2512.03422v2)**

> **作者:** Tianchen Deng; Yue Pan; Shenghai Yuan; Dong Li; Chen Wang; Mingrui Li; Long Chen; Lihua Xie; Danwei Wang; Jingchuan Wang; Javier Civera; Hesheng Wang; Weidong Chen
>
> **摘要:** In this paper, we provide a comprehensive overview of existing scene representation methods for robotics, covering traditional representations such as point clouds, voxels, signed distance functions (SDF), and scene graphs, as well as more recent neural representations like Neural Radiance Fields (NeRF), 3D Gaussian Splatting (3DGS), and the emerging Foundation Models. While current SLAM and localization systems predominantly rely on sparse representations like point clouds and voxels, dense scene representations are expected to play a critical role in downstream tasks such as navigation and obstacle avoidance. Moreover, neural representations such as NeRF, 3DGS, and foundation models are well-suited for integrating high-level semantic features and language-based priors, enabling more comprehensive 3D scene understanding and embodied intelligence. In this paper, we categorized the core modules of robotics into five parts (Perception, Mapping, Localization, Navigation, Manipulation). We start by presenting the standard formulation of different scene representation methods and comparing the advantages and disadvantages of scene representation across different modules. This survey is centered around the question: What is the best 3D scene representation for robotics? We then discuss the future development trends of 3D scene representations, with a particular focus on how the 3D Foundation Model could replace current methods as the unified solution for future robotic applications. The remaining challenges in fully realizing this model are also explored. We aim to offer a valuable resource for both newcomers and experienced researchers to explore the future of 3D scene representations and their application in robotics. We have published an open-source project on GitHub and will continue to add new works and technologies to this project.
>
---
#### [replaced 003] Aerial Robots Persistent Monitoring and Target Detection: Deployment and Assessment in the Field
- **分类: cs.RO**

- **简介: 该论文属于多机器人持续监控与目标检测任务，解决实际部署中的故障问题。提出融合多种模型的算法，提升系统鲁棒性与效率，并通过实验验证其有效性。**

- **链接: [https://arxiv.org/pdf/2504.18832v2](https://arxiv.org/pdf/2504.18832v2)**

> **作者:** Manuel Boldrer; Vit Kratky; Martin Saska
>
> **摘要:** In this article, we present a distributed algorithm for multi-robot persistent monitoring and target detection. In particular, we propose a novel solution that effectively integrates the Time-inverted Kuramoto model, three-dimensional Lissajous curves, and Model Predictive Control. We focus on the implementation of this algorithm on aerial robots, addressing the practical challenges involved in deploying our approach under real-world conditions. Our method ensures an effective and robust solution that maintains operational efficiency even in the presence of what we define as type I and type II failures. Type I failures refer to short-time disruptions, such as tracking errors and communication delays, while type II failures account for long-time disruptions, including malicious attacks, severe communication failures, and battery depletion. Our approach guarantees persistent monitoring and target detection despite these challenges. Furthermore, we validate our method with extensive field experiments involving up to eleven aerial robots, demonstrating the effectiveness, resilience, and scalability of our solution.
>
---
#### [replaced 004] Proprioception Enhances Vision Language Model in Generating Captions and Subtask Segmentations for Robot Task
- **分类: cs.RO**

- **简介: 该论文属于视觉语言模型任务，旨在解决机器人运动理解与任务分割问题。通过引入本体感觉数据，提升视频描述生成和子任务分割效果。**

- **链接: [https://arxiv.org/pdf/2512.20876v2](https://arxiv.org/pdf/2512.20876v2)**

> **作者:** Kanata Suzuki; Shota Shimizu; Tetsuya Ogata
>
> **摘要:** From the perspective of future developments in robotics, it is crucial to verify whether foundation models trained exclusively on offline data, such as images and language, can understand the robot motion. In particular, since Vision Language Models (VLMs) do not include low-level motion information from robots in their training datasets, video understanding including trajectory information remains a significant challenge. In this study, we assess two capabilities of VLMs through a video captioning task with low-level robot motion information: (1) automatic captioning of robot tasks and (2) segmentation of a series of tasks. Both capabilities are expected to enhance the efficiency of robot imitation learning by linking language and motion and serve as a measure of the foundation model's performance. The proposed method generates multiple "scene" captions using image captions and trajectory data from robot tasks. The full task caption is then generated by summarizing these individual captions. Additionally, the method performs subtask segmentation by comparing the similarity between text embeddings of image captions. In both captioning tasks, the proposed method aims to improve performance by providing the robot's motion data - joint and end-effector states - as input to the VLM. Simulator experiments were conducted to validate the effectiveness of the proposed method.
>
---
#### [replaced 005] Explicit World Models for Reliable Human-Robot Collaboration
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人机协作任务，旨在解决感知噪声和指令模糊带来的可靠性问题。通过构建显式世界模型，使机器人更好地理解并响应人类意图。**

- **链接: [https://arxiv.org/pdf/2601.01705v2](https://arxiv.org/pdf/2601.01705v2)**

> **作者:** Kenneth Kwok; Basura Fernando; Qianli Xu; Vigneshwaran Subbaraju; Dongkyu Choi; Boon Kiat Quek
>
> **备注:** Accepted to AAAI-26 Bridge Program B10: Making Embodied AI Reliable with Testing and Formal Verification
>
> **摘要:** This paper addresses the topic of robustness under sensing noise, ambiguous instructions, and human-robot interaction. We take a radically different tack to the issue of reliable embodied AI: instead of focusing on formal verification methods aimed at achieving model predictability and robustness, we emphasise the dynamic, ambiguous and subjective nature of human-robot interactions that requires embodied AI systems to perceive, interpret, and respond to human intentions in a manner that is consistent, comprehensible and aligned with human expectations. We argue that when embodied agents operate in human environments that are inherently social, multimodal, and fluid, reliability is contextually determined and only has meaning in relation to the goals and expectations of humans involved in the interaction. This calls for a fundamentally different approach to achieving reliable embodied AI that is centred on building and updating an accessible "explicit world model" representing the common ground between human and AI, that is used to align robot behaviours with human expectations.
>
---
#### [replaced 006] MG-SLAM: Structure Gaussian Splatting SLAM with Manhattan World Hypothesis
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于SLAM任务，旨在解决复杂室内环境中的重建不完整问题。通过引入曼哈顿世界假设，提升几何精度与完整性。**

- **链接: [https://arxiv.org/pdf/2405.20031v4](https://arxiv.org/pdf/2405.20031v4)**

> **作者:** Shuhong Liu; Tianchen Deng; Heng Zhou; Liuzhuozheng Li; Hongyu Wang; Danwei Wang; Mingrui Li
>
> **备注:** IEEE Transactions on Automation Science and Engineering
>
> **摘要:** Gaussian Splatting SLAMs have made significant advancements in improving the efficiency and fidelity of real-time reconstructions. However, these systems often encounter incomplete reconstructions in complex indoor environments, characterized by substantial holes due to unobserved geometry caused by obstacles or limited view angles. To address this challenge, we present Manhattan Gaussian SLAM, an RGB-D system that leverages the Manhattan World hypothesis to enhance geometric accuracy and completeness. By seamlessly integrating fused line segments derived from structured scenes, our method ensures robust tracking in textureless indoor areas. Moreover, The extracted lines and planar surface assumption allow strategic interpolation of new Gaussians in regions of missing geometry, enabling efficient scene completion. Extensive experiments conducted on both synthetic and real-world scenes demonstrate that these advancements enable our method to achieve state-of-the-art performance, marking a substantial improvement in the capabilities of Gaussian SLAM systems.
>
---
#### [replaced 007] A Vision-Language-Action Model with Visual Prompt for OFF-Road Autonomous Driving
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，旨在解决非结构化道路轨迹规划问题。提出OFF-EMMA框架，通过视觉提示和链式思考策略提升模型的感知与推理能力。**

- **链接: [https://arxiv.org/pdf/2601.03519v2](https://arxiv.org/pdf/2601.03519v2)**

> **作者:** Liangdong Zhang; Yiming Nie; Haoyang Li; Fanjie Kong; Baobao Zhang; Shunxin Huang; Kai Fu; Chen Min; Liang Xiao
>
> **摘要:** Efficient trajectory planning in off-road terrains presents a formidable challenge for autonomous vehicles, often necessitating complex multi-step pipelines. However, traditional approaches exhibit limited adaptability in dynamic environments. To address these limitations, this paper proposes OFF-EMMA, a novel end-to-end multimodal framework designed to overcome the deficiencies of insufficient spatial perception and unstable reasoning in visual-language-action (VLA) models for off-road autonomous driving scenarios. The framework explicitly annotates input images through the design of a visual prompt block and introduces a chain-of-thought with self-consistency (COT-SC) reasoning strategy to enhance the accuracy and robustness of trajectory planning. The visual prompt block utilizes semantic segmentation masks as visual prompts, enhancing the spatial understanding ability of pre-trained visual-language models for complex terrains. The COT- SC strategy effectively mitigates the error impact of outliers on planning performance through a multi-path reasoning mechanism. Experimental results on the RELLIS-3D off-road dataset demonstrate that OFF-EMMA significantly outperforms existing methods, reducing the average L2 error of the Qwen backbone model by 13.3% and decreasing the failure rate from 16.52% to 6.56%.
>
---
#### [replaced 008] RoboPanoptes: The All-seeing Robot with Whole-body Dexterity
- **分类: cs.RO**

- **简介: 该论文介绍RoboPanoptes，一种通过全身视觉实现全身体操能力的机器人，解决复杂环境中的操作与导航问题，通过分布式视觉和学习策略提升适应性与效率。**

- **链接: [https://arxiv.org/pdf/2501.05420v3](https://arxiv.org/pdf/2501.05420v3)**

> **作者:** Xiaomeng Xu; Dominik Bauer; Shuran Song
>
> **备注:** Project website: https://robopanoptes.github.io
>
> **摘要:** We present RoboPanoptes, a capable yet practical robot system that achieves whole-body dexterity through whole-body vision. Its whole-body dexterity allows the robot to utilize its entire body surface for manipulation, such as leveraging multiple contact points or navigating constrained spaces. Meanwhile, whole-body vision uses a camera system distributed over the robot's surface to provide comprehensive, multi-perspective visual feedback of its own and the environment's state. At its core, RoboPanoptes uses a whole-body visuomotor policy that learns complex manipulation skills directly from human demonstrations, efficiently aggregating information from the distributed cameras while maintaining resilience to sensor failures. Together, these design aspects unlock new capabilities and tasks, allowing RoboPanoptes to unbox in narrow spaces, sweep multiple or oversized objects, and succeed in multi-step stowing in cluttered environments, outperforming baselines in adaptability and efficiency. Results are best viewed on https://robopanoptes.github.io.
>
---
#### [replaced 009] AgriLiRa4D: A Multi-Sensor UAV Dataset for Robust SLAM in Challenging Agricultural Fields
- **分类: cs.RO; eess.SP**

- **简介: 该论文提出AgriLiRa4D数据集，用于解决农业环境下UAV的SLAM问题，包含多传感器数据和多种场景，支持鲁棒定位研究。**

- **链接: [https://arxiv.org/pdf/2512.01753v2](https://arxiv.org/pdf/2512.01753v2)**

> **作者:** Zhihao Zhan; Yuhang Ming; Shaobin Li; Jie Yuan
>
> **摘要:** Multi-sensor Simultaneous Localization and Mapping (SLAM) is essential for Unmanned Aerial Vehicles (UAVs) performing agricultural tasks such as spraying, surveying, and inspection. However, real-world, multi-modal agricultural UAV datasets that enable research on robust operation remain scarce. To address this gap, we present AgriLiRa4D, a multi-modal UAV dataset designed for challenging outdoor agricultural environments. AgriLiRa4D spans three representative farmland types-flat, hilly, and terraced-and includes both boundary and coverage operation modes, resulting in six flight sequence groups. The dataset provides high-accuracy ground-truth trajectories from a Fiber Optic Inertial Navigation System with Real-Time Kinematic capability (FINS_RTK), along with synchronized measurements from a 3D LiDAR, a 4D Radar, and an Inertial Measurement Unit (IMU), accompanied by complete intrinsic and extrinsic calibrations. Leveraging its comprehensive sensor suite and diverse real-world scenarios, AgriLiRa4D supports diverse SLAM and localization studies and enables rigorous robustness evaluation against low-texture crops, repetitive patterns, dynamic vegetation, and other challenges of real agricultural environments. To further demonstrate its utility, we benchmark four state-of-the-art multi-sensor SLAM algorithms across different sensor combinations, highlighting the difficulty of the proposed sequences and the necessity of multi-modal approaches for reliable UAV localization. By filling a critical gap in agricultural SLAM datasets, AgriLiRa4D provides a valuable benchmark for the research community and contributes to advancing autonomous navigation technologies for agricultural UAVs. The dataset can be downloaded from: https://zhan994.github.io/AgriLiRa4D.
>
---
#### [replaced 010] MimicKit: A Reinforcement Learning Framework for Motion Imitation and Control
- **分类: cs.GR; cs.LG; cs.RO**

- **简介: 该论文提出MimicKit，一个用于运动模仿与控制的强化学习框架，解决运动控制器训练问题，提供模块化代码和标准化结构，支持图形学与机器人应用。**

- **链接: [https://arxiv.org/pdf/2510.13794v3](https://arxiv.org/pdf/2510.13794v3)**

> **作者:** Xue Bin Peng
>
> **摘要:** MimicKit is an open-source framework for training motion controllers using motion imitation and reinforcement learning. The codebase provides implementations of commonly-used motion-imitation techniques and RL algorithms. This framework is intended to support research and applications in computer graphics and robotics by providing a unified training framework, along with standardized environment, agent, and data structures. The codebase is designed to be modular and easily configurable, enabling convenient modification and extension to new characters and tasks. The open-source codebase is available at: https://github.com/xbpeng/MimicKit.
>
---
#### [replaced 011] Certifying Stability of Reinforcement Learning Policies using Generalized Lyapunov Functions
- **分类: cs.LG; cs.RO; eess.SY; math.OC**

- **简介: 该论文属于强化学习稳定性分析任务，旨在解决RL策略稳定性证书构建难题。通过引入广义Lyapunov函数，提升稳定性验证能力。**

- **链接: [https://arxiv.org/pdf/2505.10947v4](https://arxiv.org/pdf/2505.10947v4)**

> **作者:** Kehan Long; Jorge Cortés; Nikolay Atanasov
>
> **备注:** NeurIPS 2025
>
> **摘要:** Establishing stability certificates for closed-loop systems under reinforcement learning (RL) policies is essential to move beyond empirical performance and offer guarantees of system behavior. Classical Lyapunov methods require a strict stepwise decrease in the Lyapunov function but such certificates are difficult to construct for learned policies. The RL value function is a natural candidate but it is not well understood how it can be adapted for this purpose. To gain intuition, we first study the linear quadratic regulator (LQR) problem and make two key observations. First, a Lyapunov function can be obtained from the value function of an LQR policy by augmenting it with a residual term related to the system dynamics and stage cost. Second, the classical Lyapunov decrease requirement can be relaxed to a generalized Lyapunov condition requiring only decrease on average over multiple time steps. Using this intuition, we consider the nonlinear setting and formulate an approach to learn generalized Lyapunov functions by augmenting RL value functions with neural network residual terms. Our approach successfully certifies the stability of RL policies trained on Gymnasium and DeepMind Control benchmarks. We also extend our method to jointly train neural controllers and stability certificates using a multi-step Lyapunov loss, resulting in larger certified inner approximations of the region of attraction compared to the classical Lyapunov approach. Overall, our formulation enables stability certification for a broad class of systems with learned policies by making certificates easier to construct, thereby bridging classical control theory and modern learning-based methods.
>
---
#### [replaced 012] First Experimental Demonstration of Natural Hovering Extremum Seeking: A New Paradigm in Flapping Flight Physics
- **分类: cs.RO; math.OC**

- **简介: 该论文属于飞行控制任务，旨在解决自主悬停问题。通过实验验证了一种无需模型的自然悬停极值搜索方法，使飞行器能自主稳定在光源上方。**

- **链接: [https://arxiv.org/pdf/2508.20836v2](https://arxiv.org/pdf/2508.20836v2)**

> **作者:** Ahmed A. Elgohary; Rohan Palanikumar; Sameh A. Eisa
>
> **摘要:** In this letter, we report the first experimental demonstration of the recently emerged new paradigm in flapping flight physics called (Natural Hovering Extremum Seeking (NH-ES)) [doi.org/10.1103/4dm4-kc4g], which theorized that hovering flight physics observed in nature by flapping insects and hummingbirds can be generated via a model-free, real-time, computationally basic, sensory-based feedback mechanism that only needs the built-in natural oscillations of the flapping wing as its propulsive input. We run experiments, including moth-like, light source-seeking, on a flapping-wing body in a total model-free setting that is agnostic to morphological parameters and body/aerodynamic models, and show that the flapping body gains altitude and stabilizes hovering about the light source autonomously needing only sensor measurements of light intensity.
>
---
#### [replaced 013] Multi-User Personalisation in Human-Robot Interaction: Resolving Preference Conflicts Using Gradual Argumentation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于多用户人机交互任务，解决多人偏好冲突问题。提出MUP-QBAF框架，通过动态论证机制协调不同用户偏好。**

- **链接: [https://arxiv.org/pdf/2511.03576v3](https://arxiv.org/pdf/2511.03576v3)**

> **作者:** Aniol Civit; Antonio Andriella; Carles Sierra; Guillem Alenyà
>
> **备注:** Preprint submitted to a journal
>
> **摘要:** While personalisation in Human-Robot Interaction (HRI) has advanced significantly, most existing approaches focus on single-user adaptation, overlooking scenarios involving multiple stakeholders with potentially conflicting preferences. To address this, we propose the Multi-User Preferences Quantitative Bipolar Argumentation Framework (MUP-QBAF), a novel multi-user personalisation framework based on Quantitative Bipolar Argumentation Frameworks (QBAFs) that explicitly models and resolves multi-user preference conflicts. Unlike prior work in Argumentation Frameworks, which typically assumes static inputs, our approach is tailored to robotics: it incorporates both users' arguments and the robot's dynamic observations of the environment, allowing the system to adapt over time and respond to changing contexts. Preferences, both positive and negative, are represented as arguments whose strength is recalculated iteratively based on new information. The framework's properties and capabilities are presented and validated through a realistic case study, where an assistive robot mediates between the conflicting preferences of a caregiver and a care recipient during a frailty assessment task. This evaluation further includes a sensitivity analysis of argument base scores, demonstrating how preference outcomes can be shaped by user input and contextual observations. By offering a transparent, structured, and context-sensitive approach to resolving competing user preferences, this work advances the field of multi-user HRI. It provides a principled alternative to data-driven methods, enabling robots to navigate conflicts in real-world environments.
>
---
#### [replaced 014] From Human Bias to Robot Choice: How Occupational Contexts and Racial Priming Shape Robot Selection
- **分类: cs.RO; cs.HC**

- **简介: 该论文研究人类偏见如何影响对机器人选择，属于社会心理学与人机交互任务，旨在解决职业情境和种族暗示对机器人偏好影响的问题。通过实验分析不同职业领域中的选择模式。**

- **链接: [https://arxiv.org/pdf/2512.20951v2](https://arxiv.org/pdf/2512.20951v2)**

> **作者:** Jiangen He; Wanqi Zhang; Jessica Barfield
>
> **备注:** HRI '26
>
> **摘要:** As artificial agents increasingly integrate into professional environments, fundamental questions have emerged about how societal biases influence human-robot selection decisions. We conducted two comprehensive experiments (N = 1,038) examining how occupational contexts and stereotype activation shape robotic agent choices across construction, healthcare, educational, and athletic domains. Participants made selections from artificial agents that varied systematically in skin tone and anthropomorphic characteristics. Our study revealed distinct context-dependent patterns. Healthcare and educational scenarios demonstrated strong favoritism toward lighter-skinned artificial agents, while construction and athletic contexts showed greater acceptance of darker-toned alternatives. Participant race was associated with systematic differences in selection patterns across professional domains. The second experiment demonstrated that exposure to human professionals from specific racial backgrounds systematically shifted later robotic agent preferences in stereotype-consistent directions. These findings show that occupational biases and color-based discrimination transfer directly from human-human to human-robot evaluation contexts. The results highlight mechanisms through which robotic deployment may unintentionally perpetuate existing social inequalities.
>
---
#### [replaced 015] E2-BKI: Evidential Ellipsoidal Bayesian Kernel Inference for Uncertainty-aware Gaussian Semantic Mapping
- **分类: cs.RO**

- **简介: 该论文属于语义地图构建任务，旨在解决户外环境中不确定性带来的映射性能下降问题。提出E2-BKI框架，结合证据深度学习与贝叶斯核推理，提升映射的鲁棒性和准确性。**

- **链接: [https://arxiv.org/pdf/2509.11964v2](https://arxiv.org/pdf/2509.11964v2)**

> **作者:** Junyoung Kim; Minsik Jeon; Jihong Min; Kiho Kwak; Junwon Seo
>
> **备注:** Accepted to IEEE RA-L. Our project website can be found at https://kjyoung.github.io/Homepage/#/Projects/E2-BKI
>
> **摘要:** Semantic mapping aims to construct a 3D semantic representation of the environment, providing essential knowledge for robots operating in complex outdoor settings. While Bayesian Kernel Inference (BKI) addresses discontinuities of map inference from sparse sensor data, existing semantic mapping methods suffer from various sources of uncertainties in challenging outdoor environments. To address these issues, we propose an uncertainty-aware semantic mapping framework that handles multiple sources of uncertainties, which significantly degrade mapping performance. Our method estimates uncertainties in semantic predictions using Evidential Deep Learning and incorporates them into BKI for robust semantic inference. It further aggregates noisy observations into coherent Gaussian representations to mitigate the impact of unreliable points, while employing geometry-aligned kernels that adapt to complex scene structures. These Gaussian primitives effectively fuse local geometric and semantic information, enabling robust, uncertainty-aware mapping in complex outdoor scenarios. Comprehensive evaluation across diverse off-road and urban outdoor environments demonstrates consistent improvements in mapping quality, uncertainty calibration, representational flexibility, and robustness, while maintaining real-time efficiency. Our project website: https://e2-bki.github.io
>
---
#### [replaced 016] LOST-3DSG: Lightweight Open-Vocabulary 3D Scene Graphs with Semantic Tracking in Dynamic Environments
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于动态环境中的3D场景图跟踪任务，旨在解决机器人跟踪动态物体的效率问题。提出LOST-3DSG方法，利用语义嵌入实现轻量级、开放词汇的物体跟踪。**

- **链接: [https://arxiv.org/pdf/2601.02905v2](https://arxiv.org/pdf/2601.02905v2)**

> **作者:** Sara Micol Ferraina; Michele Brienza; Francesco Argenziano; Emanuele Musumeci; Vincenzo Suriani; Domenico D. Bloisi; Daniele Nardi
>
> **摘要:** Tracking objects that move within dynamic environments is a core challenge in robotics. Recent research has advanced this topic significantly; however, many existing approaches remain inefficient due to their reliance on heavy foundation models. To address this limitation, we propose LOST-3DSG, a lightweight open-vocabulary 3D scene graph designed to track dynamic objects in real-world environments. Our method adopts a semantic approach to entity tracking based on word2vec and sentence embeddings, enabling an open-vocabulary representation while avoiding the necessity of storing dense CLIP visual features. As a result, LOST-3DSG achieves superior performance compared to approaches that rely on high-dimensional visual embeddings. We evaluate our method through qualitative and quantitative experiments conducted in a real 3D environment using a TIAGo robot. The results demonstrate the effectiveness and efficiency of LOST-3DSG in dynamic object tracking. Code and supplementary material are publicly available on the project website at https://lab-rococo-sapienza.github.io/lost-3dsg/.
>
---
#### [replaced 017] VibES: Induced Vibration for Persistent Event-Based Sensing
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出VibES方法，解决静态场景下事件相机无法生成事件的问题。通过振动装置和运动补偿算法，实现持续事件生成，提升图像重建与边缘检测效果。**

- **链接: [https://arxiv.org/pdf/2508.19094v2](https://arxiv.org/pdf/2508.19094v2)**

> **作者:** Vincenzo Polizzi; Stephen Yang; Quentin Clark; Jonathan Kelly; Igor Gilitschenski; David B. Lindell
>
> **备注:** Accepted to the IEEE International Conference on 3D Vision (3DV), Vancouver, BC, Canada, Mar 20-23, 2026
>
> **摘要:** Event cameras are a bio-inspired class of sensors that asynchronously measure per-pixel intensity changes. Under fixed illumination conditions in static or low-motion scenes, rigidly mounted event cameras are unable to generate any events and become unsuitable for most computer vision tasks. To address this limitation, recent work has investigated motion-induced event stimulation, which often requires complex hardware or additional optical components. In contrast, we introduce a lightweight approach to sustain persistent event generation by employing a simple rotating unbalanced mass to induce periodic vibrational motion. This is combined with a motion-compensation pipeline that removes the injected motion and yields clean, motion-corrected events for downstream perception tasks. We develop a hardware prototype to demonstrate our approach and evaluate it on real-world datasets. Our method reliably recovers motion parameters and improves both image reconstruction and edge detection compared to event-based sensing without motion induction.
>
---
#### [replaced 018] DiffPF: Differentiable Particle Filtering with Generative Sampling via Conditional Diffusion Models
- **分类: cs.RO**

- **简介: 该论文提出DiffPF，一种基于条件扩散模型的可微粒子滤波方法，用于动态系统中的状态估计。解决传统方法在复杂分布下采样效率低的问题，通过学习灵活后验采样器提升估计精度。**

- **链接: [https://arxiv.org/pdf/2507.15716v2](https://arxiv.org/pdf/2507.15716v2)**

> **作者:** Ziyu Wan; Lin Zhao
>
> **摘要:** This paper proposes DiffPF, a differentiable particle filter that leverages diffusion models for state estimation in dynamic systems. Unlike conventional differentiable particle filters, which require importance weighting and typically rely on predefined or low-capacity proposal distributions. DiffPF learns a flexible posterior sampler by conditioning a diffusion model on predicted particles and the current observation. This enables accurate, equally-weighted sampling from complex, high-dimensional, and multimodal filtering distributions. We evaluate DiffPF across a range of scenarios, including both unimodal and highly multimodal distributions, and test it on simulated as well as real-world tasks, where it consistently outperforms existing filtering baselines. In particular, DiffPF achieves an 82.8% improvement in estimation accuracy on a highly multimodal global localization benchmark, and a 26% improvement on the real-world KITTI visual odometry benchmark, compared to state-of-the-art differentiable filters. To the best of our knowledge, DiffPF is the first method to integrate conditional diffusion models into particle filtering, enabling high-quality posterior sampling that produces more informative particles and significantly improves state estimation.
>
---
#### [replaced 019] Out-of-Distribution Semantic Occupancy Prediction
- **分类: cs.CV; cs.RO; eess.IV**

- **简介: 该论文属于3D语义占用预测任务，旨在解决OoD物体检测问题。提出OccOoD框架和数据集，提升OoD检测性能。**

- **链接: [https://arxiv.org/pdf/2506.21185v2](https://arxiv.org/pdf/2506.21185v2)**

> **作者:** Yuheng Zhang; Mengfei Duan; Kunyu Peng; Yuhang Wang; Ruiping Liu; Fei Teng; Kai Luo; Zhiyong Li; Kailun Yang
>
> **备注:** The established datasets and source code will be made publicly available at https://github.com/7uHeng/OccOoD
>
> **摘要:** 3D semantic occupancy prediction is crucial for autonomous driving, providing a dense, semantically rich environmental representation. However, existing methods focus on in-distribution scenes, making them susceptible to Out-of-Distribution (OoD) objects and long-tail distributions, which increases the risk of undetected anomalies and misinterpretations, posing safety hazards. To address these challenges, we introduce Out-of-Distribution Semantic Occupancy Prediction, targeting OoD detection in 3D voxel space. To fill dataset gaps, we propose a Realistic Anomaly Augmentation that injects synthetic anomalies while preserving realistic spatial and occlusion patterns, enabling the creation of two datasets: VAA-KITTI and VAA-KITTI-360. Then, a novel framework that integrates OoD detection into 3D semantic occupancy prediction, OccOoD, is proposed, which uses Cross-Space Semantic Refinement (CSSR) to refine semantic predictions from complementary voxel and BEV representations, improving OoD detection. Experimental results demonstrate that OccOoD achieves state-of-the-art OoD detection with an AuROC of 65.50% and an AuPRCr of 31.83 within a 1.2m region, while maintaining competitive semantic occupancy prediction performance and generalization in real-world urban driving scenes. The established datasets and source code will be made publicly available at https://github.com/7uHeng/OccOoD.
>
---
#### [replaced 020] Scaffolding Dexterous Manipulation with Vision-Language Models
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决复杂抓取训练难题。通过视觉-语言模型生成轨迹指导强化学习，提升机器人操作的泛化能力与真实性。**

- **链接: [https://arxiv.org/pdf/2506.19212v3](https://arxiv.org/pdf/2506.19212v3)**

> **作者:** Vincent de Bakker; Joey Hejna; Tyler Ga Wei Lum; Onur Celik; Aleksandar Taranovic; Denis Blessing; Gerhard Neumann; Jeannette Bohg; Dorsa Sadigh
>
> **摘要:** Dexterous robotic hands are essential for performing complex manipulation tasks, yet remain difficult to train due to the challenges of demonstration collection and high-dimensional control. While reinforcement learning (RL) can alleviate the data bottleneck by generating experience in simulation, it typically relies on carefully designed, task-specific reward functions, which hinder scalability and generalization. Thus, contemporary works in dexterous manipulation have often bootstrapped from reference trajectories. These trajectories specify target hand poses that guide the exploration of RL policies and object poses that enable dense, task-agnostic rewards. However, sourcing suitable trajectories - particularly for dexterous hands - remains a significant challenge. Yet, the precise details in explicit reference trajectories are often unnecessary, as RL ultimately refines the motion. Our key insight is that modern vision-language models (VLMs) already encode the commonsense spatial and semantic knowledge needed to specify tasks and guide exploration effectively. Given a task description (e.g., "open the cabinet") and a visual scene, our method uses an off-the-shelf VLM to first identify task-relevant keypoints (e.g., handles, buttons) and then synthesize 3D trajectories for hand motion and object motion. Subsequently, we train a low-level residual RL policy in simulation to track these coarse trajectories or "scaffolds" with high fidelity. Across a number of simulated tasks involving articulated objects and semantic understanding, we demonstrate that our method is able to learn robust dexterous manipulation policies. Moreover, we showcase that our method transfers to real-world robotic hands without any human demonstrations or handcrafted rewards.
>
---
#### [replaced 021] Surface-Based Manipulation with Modular Foldable Robots
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人操作任务，旨在解决复杂物体抓取与操控问题。通过表面操控策略，使用平面末端执行器实现物体的移动、旋转和翻转，无需稳定抓握。**

- **链接: [https://arxiv.org/pdf/2502.19389v2](https://arxiv.org/pdf/2502.19389v2)**

> **作者:** Ziqiao Wang; Serhat Demirtas; Fabio Zuliani; Jamie Paik
>
> **备注:** This manuscript has been published in npj Robotics. Supplementary video: https://www.youtube.com/watch?v=2TPTBqp84BY
>
> **摘要:** Intelligence lies not only in the brain (decision-making processes) but in the body (physical morphology). The morphology of robots can significantly influence how they interact with the physical world, crucial for manipulating objects in real-life scenarios. Conventional robotic manipulation strategies mainly rely on finger-shaped end effectors. However, achieving stable grasps on fragile, deformable, irregularly shaped, or slippery objects is challenging due to difficulty in establishing stable forces or geometric constraints. Here, we present surface-based manipulation strategies that diverge from classical grasping approaches, using flat surfaces as minimalist end-effectors. By adjusting surfaces' position and orientation, objects can be translated, rotated, and flipped across the surface using closed-loop control strategies. Since this method does not rely on stable grasping, it can adapt to objects of various shapes, sizes, and stiffness levels and can even manipulate the shape of deformable objects. Our results provide a new perspective for solving complex manipulation problems.
>
---
#### [replaced 022] Integrating Symbolic RL Planning into a BDI-based Autonomous UAV Framework: System Integration and SIL Validation
- **分类: cs.RO; cs.AI**

- **简介: 论文提出AMAD-SRL框架，将符号化强化学习整合到BDI架构中，解决无人机任务规划与执行的动态复杂性问题，提升任务效率与安全性。**

- **链接: [https://arxiv.org/pdf/2508.11890v2](https://arxiv.org/pdf/2508.11890v2)**

> **作者:** Sangwoo Jeon; Juchul Shin; YeonJe Cho; Gyeong-Tae Kim; Seongwoo Kim
>
> **备注:** This submission has been withdrawn by the authors due to institutional and contractual requirements related to security and export-control review
>
> **摘要:** Modern autonomous drone missions increasingly require software frameworks capable of seamlessly integrating structured symbolic planning with adaptive reinforcement learning (RL). Although traditional rule-based architectures offer robust structured reasoning for drone autonomy, their capabilities fall short in dynamically complex operational environments that require adaptive symbolic planning. Symbolic RL (SRL), using the Planning Domain Definition Language (PDDL), explicitly integrates domain-specific knowledge and operational constraints, significantly improving the reliability and safety of unmanned aerial vehicle (UAV) decision making. In this study, we propose the AMAD-SRL framework, an extended and refined version of the Autonomous Mission Agents for Drones (AMAD) cognitive multi-agent architecture, enhanced with symbolic reinforcement learning for dynamic mission planning and execution. We validated our framework in a Software-in-the-Loop (SIL) environment structured identically to an intended Hardware-In-the-Loop Simulation (HILS) platform, ensuring seamless transition to real hardware. Experimental results demonstrate stable integration and interoperability of modules, successful transitions between BDI-driven and symbolic RL-driven planning phases, and consistent mission performance. Specifically, we evaluate a target acquisition scenario in which the UAV plans a surveillance path followed by a dynamic reentry path to secure the target while avoiding threat zones. In this SIL evaluation, mission efficiency improved by approximately 75% over a coverage-based baseline, measured by travel distance reduction. This study establishes a robust foundation for handling complex UAV missions and discusses directions for further enhancement and validation.
>
---
#### [replaced 023] SeePerSea: Multi-modal Perception Dataset of In-water Objects for Autonomous Surface Vehicles
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出SeePerSea数据集，用于解决水下障碍物感知问题，提升自主水面车辆的环境感知能力。通过多模态数据训练和测试算法，推动海洋自主技术发展。**

- **链接: [https://arxiv.org/pdf/2404.18411v3](https://arxiv.org/pdf/2404.18411v3)**

> **作者:** Mingi Jeong; Arihant Chadda; Ziang Ren; Luyang Zhao; Haowen Liu; Monika Roznere; Aiwei Zhang; Yitao Jiang; Sabriel Achong; Samuel Lensgraf; Alberto Quattrini Li
>
> **备注:** Topic: Special Issue on ICRA 2024 Workshop on Field Robotics
>
> **摘要:** This paper introduces the first publicly accessible labeled multi-modal perception dataset for autonomous maritime navigation, focusing on in-water obstacles within the aquatic environment to enhance situational awareness for Autonomous Surface Vehicles (ASVs). This dataset, collected over 4 years and consisting of diverse objects encountered under varying environmental conditions, aims to bridge the research gap in ASVs by providing a multi-modal, annotated, and ego-centric perception dataset, for object detection and classification. We also show the applicability of the proposed dataset by training and testing current deep learning-based open-source perception algorithms that have shown success in the autonomous ground vehicle domain. With the training and testing results, we discuss open challenges for existing datasets and methods, identifying future research directions. We expect that our dataset will contribute to the development of future marine autonomy pipelines and marine (field) robotics. This dataset is open source and found at https://seepersea.github.io/.
>
---
#### [replaced 024] Cross-Platform Learnable Fuzzy Gain-Scheduled Proportional-Integral-Derivative Controller Tuning via Physics-Constrained Meta-Learning and Reinforcement Learning Adaptation
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，解决PID控制器跨平台调参难题。提出一种结合元学习和强化学习的模糊增益调度方法，提升控制性能与泛化能力。**

- **链接: [https://arxiv.org/pdf/2511.06500v2](https://arxiv.org/pdf/2511.06500v2)**

> **作者:** JiaHao Wu; ShengWen Yu
>
> **备注:** 24 pages,15 tables, 6 figures
>
> **摘要:** Motivation and gap: PID-family controllers remain a pragmatic choice for many robotic systems due to their simplicity and interpretability, but tuning stable, high-performing gains is time-consuming and typically non-transferable across robot morphologies, payloads, and deployment conditions. Fuzzy gain scheduling can provide interpretable online adjustment, yet its per-joint scaling and consequent parameters are platform-dependent and difficult to tune systematically. Proposed approach: We propose a hierarchical framework for cross-platform tuning of a learnable fuzzy gain-scheduled PID (LF-PID). The controller uses shared fuzzy membership partitions to preserve common error semantics, while learning per-joint scaling and Takagi-Sugeno consequent parameters that schedule PID gains online. Combined with physics-constrained virtual robot synthesis, meta-learning provides cross-platform initialization from robot physical features, and a lightweight reinforcement learning (RL) stage performs deployment-specific refinement under dynamics mismatch. Starting from three base simulated platforms, we generate 232 physically valid training variants via bounded perturbations of mass (+/-10%), inertia (+/-15%), and friction (+/-20%). Results and insight: We evaluate cross-platform generalization on two distinct systems (a 9-DOF serial manipulator and a 12-DOF quadruped) under multiple disturbance scenarios. The RL adaptation stage improves tracking performance on top of the meta-initialized controller, with up to 80.4% error reduction in challenging high-load joints (12.36 degrees to 2.42 degrees) and 19.2% improvement under parameter uncertainty. We further identify an optimization ceiling effect: online refinement yields substantial gains when the meta-initialized baseline exhibits localized deficiencies, but provides limited improvement when baseline quality is already uniformly strong.
>
---
#### [replaced 025] ManiFeel: Benchmarking and Understanding Visuotactile Manipulation Policy Learning
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决视觉输入受限时的操控问题。通过构建ManiFeel基准，研究触觉感知对策略性能的影响，探索有效触觉模态与设计原则。**

- **链接: [https://arxiv.org/pdf/2505.18472v2](https://arxiv.org/pdf/2505.18472v2)**

> **作者:** Quan Khanh Luu; Pokuang Zhou; Zhengtong Xu; Zhiyuan Zhang; Qiang Qiu; Yu She
>
> **摘要:** Supervised visuomotor policies have shown strong performance in robotic manipulation but often struggle in tasks with limited visual inputs, such as operations in confined spaces and dimly lit environments, or tasks requiring precise perception of object properties and environmental interactions. In such cases, tactile feedback becomes essential for manipulation. While the rapid progress of supervised visuomotor policies has benefited greatly from high-quality, reproducible simulation benchmarks in visual imitation, the visuotactile domain still lacks a similarly comprehensive and reliable benchmark for large-scale and rigorous evaluation. To address this, we introduce ManiFeel, a reproducible and scalable simulation benchmark designed to systematically study supervised visuotactile policy learning. ManiFeel offers a diverse suite of contact-rich and visually challenging manipulation tasks, a modular evaluation pipeline spanning sensing modalities, tactile representations, and policy architectures, as well as real-world validation. Through extensive experiments, ManiFeel demonstrates how tactile sensing enhances policy performance across diverse manipulation scenarios, ranging from precise contact-driven operations to visually constrained settings. In addition, the results reveal task-dependent strengths of different tactile modalities and identify key design principles and open challenges for robust visuotactile policy learning. Real-world evaluations further confirm that ManiFeel provides a reliable and meaningful foundation for benchmarking and future visuotactile policy development. To foster reproducibility and future research, we will release our codebase, datasets, training logs, and pretrained checkpoints, aiming to accelerate progress toward generalizable visuotactile policy learning and manipulation.
>
---
#### [replaced 026] Modeling and Control for UAV with Off-center Slung Load
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于无人机控制任务，解决偏心悬挂负载带来的耦合动态问题。通过以悬挂点为模型基础，设计分层控制器，实现负载摆动抑制和姿态跟踪。**

- **链接: [https://arxiv.org/pdf/2601.03386v2](https://arxiv.org/pdf/2601.03386v2)**

> **作者:** Zongyang Lv; Yanmei Jia; Yongqing Liu; Alan F. Lynch; Qing Zhao; Yuhu Wu
>
> **摘要:** Unmanned aerial vehicle (UAV) with slung load system is a classic air transportation system. In practical applications, the suspension point of the slung load does not always align with the center of mass (CoM) of the UAV due to mission requirements or mechanical interference. This offset creates coupling in the system's nonlinear dynamics which leads to a complicated motion control problem. In existing research, modeling of the system are performed about the UAV's CoM. In this work we use the point of suspension instead. Based on the new model, a cascade control strategy is developed. In the middle-loop controller, the acceleration of the suspension point is used to regulate the swing angle of the slung load without the need for considering the coupling between the slung load and the UAV. An inner-loop controller is designed to track the UAV's attitude without the need of simplification on the coupling effects. We prove local exponential stability of the closed-loop using Lyapunov approach. Finally, simulations and experiments are conducted to validate the proposed control system.
>
---
#### [replaced 027] AURA-CVC: Autonomous Ultrasound-guided Robotic Assistance for Central Venous Catheterization
- **分类: cs.RO**

- **简介: 该论文属于医疗机器人任务，旨在解决CVC操作中的人为误差问题。提出一种端到端的机器人辅助系统，实现自主扫描、血管定位和针头插入。**

- **链接: [https://arxiv.org/pdf/2507.05979v2](https://arxiv.org/pdf/2507.05979v2)**

> **作者:** Deepak Raina; Lidia Al-Zogbi; Brian Teixeira; Vivek Singh; Ankur Kapoor; Thorsten Fleiter; Muyinatu A. Lediju Bell; Vinciya Pandian; Axel Krieger
>
> **备注:** Accepted in International Journal of Computer Assisted Radiology and Surgery (IJCARS) 2026
>
> **摘要:** Purpose: Central venous catheterization (CVC) is a critical medical procedure for vascular access, hemodynamic monitoring, and life-saving interventions. Its success remains challenging due to the need for continuous ultrasound-guided visualization of a target vessel and approaching needle, which is further complicated by anatomical variability and operator dependency. Errors in needle placement can lead to life-threatening complications. While robotic systems offer a potential solution, achieving full autonomy remains challenging. In this work, we propose an end-to-end robotic-ultrasound-guided CVC pipeline, from scan initialization to needle insertion. Methods: We introduce a deep-learning model to identify clinically relevant anatomical landmarks from a depth image of the patient's neck, obtained using RGB-D camera, to autonomously define the scanning region and paths. Then, a robot motion planning framework is proposed to scan, segment, reconstruct, and localize vessels (veins and arteries), followed by the identification of the optimal insertion zone. Finally, a needle guidance module plans the insertion under ultrasound guidance with operator's feedback. This pipeline was validated on a high-fidelity commercial phantom across 10 simulated clinical scenarios. Results: The proposed pipeline achieved 10 out of 10 successful needle placements on the first attempt. Vessels were reconstructed with a mean error of 2.15 \textit{mm}, and autonomous needle insertion was performed with an error less than or close to 1 \textit{mm}. Conclusion: To our knowledge, this is the first robotic CVC system demonstrated on a high-fidelity phantom with integrated planning, scanning, and insertion. Experimental results show its potential for clinical translation.
>
---
#### [replaced 028] Modern Middlewares for Automated Vehicles: A Tutorial
- **分类: cs.SE; cs.DC; cs.RO; eess.SY**

- **简介: 论文属于技术综述任务，旨在介绍自动驾驶车辆中的中间件技术，解决如何有效设计和应用中间件的问题。工作包括分析中间件类型、功能及未来应用方向。**

- **链接: [https://arxiv.org/pdf/2412.07817v2](https://arxiv.org/pdf/2412.07817v2)**

> **作者:** David Philipp Klüner; Marius Molz; Alexandru Kampmann; Stefan Kowalewski; Bassam Alrifaee
>
> **备注:** This work has been submitted and accepted to the IEEE for possible publication
>
> **摘要:** This paper offers a tutorial on current middlewares in automated vehicles. Our aim is to provide the reader with an overview of current middlewares and to identify open challenges in this field. We start by explaining the fundamentals of software architecture in distributed systems and the distinguishing requirements of Automated Vehicles. We then distinguish between communication middlewares and architecture platforms and highlight their key principles and differences. Next, we present five state-of-the-art middlewares as well as their capabilities and functions. We explore how these middlewares could be applied in the design of future vehicle software and their role in the automotive domain. Finally, we compare the five middlewares presented and discuss open research challenges.
>
---
#### [replaced 029] SegDAC: Improving Visual Reinforcement Learning by Extracting Dynamic Object-Centric Representations from Pretrained Vision Models
- **分类: cs.CV; cs.AI; cs.LG; cs.RO**

- **简介: 该论文属于视觉强化学习任务，旨在提升视觉泛化能力和样本效率。通过提取动态物体中心表示，提出SegDAC方法，有效整合预训练视觉模型。**

- **链接: [https://arxiv.org/pdf/2508.09325v3](https://arxiv.org/pdf/2508.09325v3)**

> **作者:** Alexandre Brown; Glen Berseth
>
> **摘要:** Visual reinforcement learning (RL) is challenging due to the need to extract useful representations from high-dimensional inputs while learning effective control from sparse and noisy rewards. Although large perception models exist, integrating them effectively into RL for visual generalization and improved sample efficiency remains difficult. We propose SegDAC, a Segmentation-Driven Actor-Critic method. SegDAC uses Segment Anything (SAM) for object-centric decomposition and YOLO-World to ground the image segmentation process via text inputs. It includes a novel transformer-based architecture that supports a dynamic number of segments at each time step and effectively learns which segments to focus on using online RL, without using human labels. By evaluating SegDAC over a challenging visual generalization benchmark using Maniskill3, which covers diverse manipulation tasks under strong visual perturbations, we demonstrate that SegDAC achieves significantly better visual generalization, doubling prior performance on the hardest setting and matching or surpassing prior methods in sample efficiency across all evaluated tasks. Project Page: https://segdac.github.io/
>
---
#### [replaced 030] Agile Tradespace Exploration for Space Rendezvous Mission Design via Transformers
- **分类: math.OC; cs.AI; cs.RO**

- **简介: 该论文属于航天任务设计领域，解决多目标优化难题。通过Transformer模型快速生成近似最优轨迹，提升任务设计效率与精度。**

- **链接: [https://arxiv.org/pdf/2510.03544v2](https://arxiv.org/pdf/2510.03544v2)**

> **作者:** Yuji Takubo; Daniele Gammelli; Marco Pavone; Simone D'Amico
>
> **备注:** 14 pages, 7 figures
>
> **摘要:** Spacecraft rendezvous enables on-orbit servicing, debris removal, and crewed docking, forming the foundation for a scalable space economy. Designing such missions requires rapid exploration of the tradespace between control cost and flight time across multiple candidate targets. However, multi-objective optimization in this setting is challenging, as the underlying constraints are often nonconvex, and mission designers must balance accuracy (e.g., solving the full problem) with efficiency (e.g., convex relaxations), slowing iteration and limiting design agility. To address these challenges, this paper proposes an AI-powered framework that enables agile and generalized rendezvous mission design. Given the orbital information of the target spacecraft, boundary conditions of the servicer, and a range of flight times, a transformer model generates a set of near-Pareto optimal trajectories across varying flight times in a single parallelized inference step, thereby enabling rapid mission trade studies. The model is further extended to accommodate variable flight times and perturbed orbital dynamics, supporting realistic multi-objective trade-offs. Validation on chance-constrained rendezvous problems in Earth orbits with passive safety constraints demonstrates that the model generalizes across both flight times and dynamics, consistently providing high-quality initial guesses that converge to superior solutions in fewer iterations. Moreover, the framework efficiently approximates the Pareto front, achieving runtimes comparable to convex relaxation by exploiting parallelized inference. Together, these results position the proposed framework as a practical surrogate for nonconvex trajectory generation and mark an important step toward AI-driven trajectory design for accelerating preliminary mission planning in real-world rendezvous applications.
>
---
#### [replaced 031] Adaptive Science Operations in Deep Space Missions Using Offline Belief State Planning
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于深空任务中的自主科学操作研究，解决通信延迟与环境不确定性问题。通过POMDP框架和贝叶斯网络，优化仪器序列，提升样本识别准确性。**

- **链接: [https://arxiv.org/pdf/2510.08812v2](https://arxiv.org/pdf/2510.08812v2)**

> **作者:** Grace Ra Kim; Hailey Warner; Duncan Eddy; Evan Astle; Zachary Booth; Edward Balaban; Mykel J. Kochenderfer
>
> **备注:** 7 pages, 4 tables, 5 figures, accepted in IEEE ISPARO 2025 (V2 - grammatical edits, also mispelled conference year)
>
> **摘要:** Deep space missions face extreme communication delays and environmental uncertainty that prevent real-time ground operations. To support autonomous science operations in communication-constrained environments, we present a partially observable Markov decision process (POMDP) framework that adaptively sequences spacecraft science instruments. We integrate a Bayesian network into the POMDP observation space to manage the high-dimensional and uncertain measurements typical of astrobiology missions. This network compactly encodes dependencies among measurements and improves the interpretability and computational tractability of science data. Instrument operation policies are computed offline, allowing resource-aware plans to be generated and thoroughly validated prior to launch. We use the Enceladus Orbilander's proposed Life Detection Suite (LDS) as a case study, demonstrating how Bayesian network structure and reward shaping influence system performance. We compare our method against the mission's baseline Concept of Operations (ConOps), evaluating both misclassification rates and performance in off-nominal sample accumulation scenarios. Our approach reduces sample identification errors by nearly 40%
>
---
#### [replaced 032] Meta-learning enhanced adaptive robot control strategy for automated PCB assembly
- **分类: cs.RO**

- **简介: 该论文属于机器人控制任务，旨在解决PCB装配中定位误差问题。提出一种无需视觉的元学习方法，提升机器人自适应能力，减少对专用夹具依赖。**

- **链接: [https://arxiv.org/pdf/2506.20445v2](https://arxiv.org/pdf/2506.20445v2)**

> **作者:** Jieyang Peng; Dongkun Wang; Junkai Zhao; Yunfei Teng; Andreas Kimmig; Xiaoming Tao; Jivka Ovtcharova
>
> **备注:** Pattern: CN 118960772 A
>
> **摘要:** The assembly of printed circuit boards (PCBs) is one of the standard processes in chip production, directly contributing to the quality and performance of the chips. In the automated PCB assembly process, machine vision and coordinate localization methods are commonly employed to guide the positioning of assembly units. However, occlusion or poor lighting conditions can affect the effectiveness of machine vision-based methods. Additionally, the assembly of odd-form components requires highly specialized fixtures for assembly unit positioning, leading to high costs and low flexibility, especially for multi-variety and small-batch production. Drawing on these considerations, a vision-free, model-agnostic meta-method for compensating robotic position errors is proposed, which maximizes the probability of accurate robotic positioning through interactive feedback, thereby reducing the dependency on visual feedback and mitigating the impact of occlusions or lighting variations. The proposed method endows the robot with the capability to learn and adapt to various position errors, inspired by the human instinct for grasping under uncertainties. Furthermore, it is a self-adaptive method that can accelerate the robotic positioning process as more examples are incorporated and learned. Empirical studies show that the proposed method can handle a variety of odd-form components without relying on specialized fixtures, while achieving similar assembly efficiency to highly dedicated automation equipment. As of the writing of this paper, the proposed meta-method has already been implemented in a robotic-based assembly line for odd-form electronic components. Since PCB assembly involves various electronic components with different sizes, shapes, and functions, subsequent studies can focus on assembly sequence and assembly route optimization to further enhance assembly efficiency.
>
---
#### [replaced 033] Autonomous Driving in Unstructured Environments: How Far Have We Come?
- **分类: cs.RO**

- **简介: 该论文属于自主驾驶任务，旨在解决非结构化环境中的自主驾驶问题。通过综述250余篇文献，探讨了相关技术与挑战，推动该领域发展。**

- **链接: [https://arxiv.org/pdf/2410.07701v4](https://arxiv.org/pdf/2410.07701v4)**

> **作者:** Chen Min; Shubin Si; Xu Wang; Hanzhang Xue; Weizhong Jiang; Zitong Chen; Mengmeng Li; Jilin Mei; Erke Shang; Zhipeng Xiao; Bin Dai; Qi Zhu; Hao Fu; Dawei Zhao; Liang Xiao; Yiming Nie; Yu Hu
>
> **备注:** Accepted by Journal of Field Robotics (JFR) 2025; Survey paper; 59 pages
>
> **摘要:** Research on autonomous driving in unstructured outdoor environments is less advanced than in structured urban settings due to challenges like environmental diversities and scene complexity. These environments-such as rural areas and rugged terrains-pose unique obstacles that are not common in structured urban areas. Despite these difficulties, autonomous driving in unstructured outdoor environments is crucial for applications in agriculture, mining, and military operations. Our survey reviews over 250 papers for autonomous driving in unstructured outdoor environments, covering offline mapping, pose estimation, environmental perception, path planning, end-to-end autonomous driving, datasets, and relevant challenges. We also discuss emerging trends and future research directions. This review aims to consolidate knowledge and encourage further research for autonomous driving in unstructured environments. To support ongoing work, we maintain an active repository with up-to-date literature and open-source projects at: https://github.com/chaytonmin/Survey-Autonomous-Driving-in-Unstructured-Environments.
>
---
