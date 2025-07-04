# 机器人 cs.RO

- **最新发布 24 篇**

- **更新 21 篇**

## 最新发布

#### [new 001] MISC: Minimal Intervention Shared Control with Guaranteed Safety under Non-Convex Constraints
- **分类: cs.RO; cs.HC; cs.SY; eess.SY**

- **简介: 该论文属于共享控制任务，解决非凸约束下的安全控制问题。提出一种基于最优控制的框架，确保安全、满足约束并最小干预用户意图。**

- **链接: [http://arxiv.org/pdf/2507.02438v1](http://arxiv.org/pdf/2507.02438v1)**

> **作者:** Shivam Chaubey; Francesco Verdoja; Shankar Deka; Ville Kyrki
>
> **备注:** This work has been submitted to the IEEE for possible publication
>
> **摘要:** Shared control combines human intention with autonomous decision-making, from low-level safety overrides to high-level task guidance, enabling systems that adapt to users while ensuring safety and performance. This enhances task effectiveness and user experience across domains such as assistive robotics, teleoperation, and autonomous driving. However, existing shared control methods, based on e.g. Model Predictive Control, Control Barrier Functions, or learning-based control, struggle with feasibility, scalability, or safety guarantees, particularly since the user input is unpredictable. To address these challenges, we propose an assistive controller framework based on Constrained Optimal Control Problem that incorporates an offline-computed Control Invariant Set, enabling online computation of control actions that ensure feasibility, strict constraint satisfaction, and minimal override of user intent. Moreover, the framework can accommodate structured class of non-convex constraints, which are common in real-world scenarios. We validate the approach through a large-scale user study with 66 participants--one of the most extensive in shared control research--using a computer game environment to assess task load, trust, and perceived control, in addition to performance. The results show consistent improvements across all these aspects without compromising safety and user intent.
>
---
#### [new 002] DigiT4TAF -- Bridging Physical and Digital Worlds for Future Transportation Systems
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于数字孪生任务，旨在构建交通系统的虚拟模型。通过整合传感器数据，实现真实与虚拟交通环境的实时交互，用于优化交通管理和安全模拟。**

- **链接: [http://arxiv.org/pdf/2507.02400v1](http://arxiv.org/pdf/2507.02400v1)**

> **作者:** Maximilian Zipfl; Pascal Zwick; Patrick Schulz; Marc Rene Zofka; Albert Schotschneider; Helen Gremmelmaier; Nikolai Polley; Ferdinand Mütsch; Kevin Simon; Fabian Gottselig; Michael Frey; Sergio Marschall; Akim Stark; Maximilian Müller; Marek Wehmer; Mihai Kocsis; Dominic Waldenmayer; Florian Schnepf; Erik Heinrich; Sabrina Pletz; Matthias Kölle; Karin Langbein-Euchner; Alexander Viehl; Raoul Zöllner; J. Marius Zöllner
>
> **备注:** Accepted at the IEEE IAVVC 2025 Conference
>
> **摘要:** In the future, mobility will be strongly shaped by the increasing use of digitalization. Not only will individual road users be highly interconnected, but also the road and associated infrastructure. At that point, a Digital Twin becomes particularly appealing because, unlike a basic simulation, it offers a continuous, bilateral connection linking the real and virtual environments. This paper describes the digital reconstruction used to develop the Digital Twin of the Test Area Autonomous Driving-Baden-W\"urttemberg (TAF-BW), Germany. The TAF-BW offers a variety of different road sections, from high-traffic urban intersections and tunnels to multilane motorways. The test area is equipped with a comprehensive Vehicle-to-Everything (V2X) communication infrastructure and multiple intelligent intersections equipped with camera sensors to facilitate real-time traffic flow monitoring. The generation of authentic data as input for the Digital Twin was achieved by extracting object lists at the intersections. This process was facilitated by the combined utilization of camera images from the intelligent infrastructure and LiDAR sensors mounted on a test vehicle. Using a unified interface, recordings from real-world detections of traffic participants can be resimulated. Additionally, the simulation framework's design and the reconstruction process is discussed. The resulting framework is made publicly available for download and utilization at: https://digit4taf-bw.fzi.de The demonstration uses two case studies to illustrate the application of the digital twin and its interfaces: the analysis of traffic signal systems to optimize traffic flow and the simulation of security-related scenarios in the communications sector.
>
---
#### [new 003] Integrating path-planning and control for robotic unicycles
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于机器人路径规划与控制任务，旨在解决 robotic unicycles 的路径规划与控制集成问题，通过优化路径分段和曲率来提升控制性能。**

- **链接: [http://arxiv.org/pdf/2507.02700v1](http://arxiv.org/pdf/2507.02700v1)**

> **作者:** Máté B. Vizi; Dénes Tákács; Gábor Stépán; Gábor Orosz
>
> **摘要:** This article focuses on integrating path-planning and control with specializing on the unique needs of robotic unicycles. A unicycle design is presented which is capable of accelerating/breaking and carrying out a variety of maneuvers. The proposed path-planning method segments the path into straight and curved path sections dedicated for accelerating/breaking and turning maneuvers, respectively. The curvature profiles of the curved sections are optimized while considering the control performance and the slipping limits of the wheel. The performance of the proposed integrated approach is demonstrated via numerical simulations.
>
---
#### [new 004] Optimizing Start Locations in Ergodic Search for Disaster Response
- **分类: cs.RO**

- **简介: 该论文属于机器人路径规划任务，解决灾难响应中机器人起始位置优化问题。通过联合优化方法提升覆盖性能，实验显示显著改进。**

- **链接: [http://arxiv.org/pdf/2507.02708v1](http://arxiv.org/pdf/2507.02708v1)**

> **作者:** Ananya Rao; Alyssa Hargis; David Wettergreen; Howie Choset
>
> **摘要:** In disaster response scenarios, deploying robotic teams effectively is crucial for improving situational awareness and enhancing search and rescue operations. The use of robots in search and rescue has been studied but the question of where to start robot deployments has not been addressed. This work addresses the problem of optimally selecting starting locations for robots with heterogeneous capabilities by formulating a joint optimization problem. To determine start locations, this work adds a constraint to the ergodic optimization framework whose minimum assigns robots to start locations. This becomes a little more challenging when the robots are heterogeneous (equipped with different sensing and motion modalities) because not all robots start at the same location, and a more complex adaptation of the aforementioned constraint is applied. Our method assumes access to potential starting locations, which can be obtained from expert knowledge or aerial imagery. We experimentally evaluate the efficacy of our joint optimization approach by comparing it to baseline methods that use fixed starting locations for all robots. Our experimental results show significant gains in coverage performance, with average improvements of 35.98% on synthetic data and 31.91% on real-world data for homogeneous and heterogeneous teams, in terms of the ergodic metric.
>
---
#### [new 005] A Late Collaborative Perception Framework for 3D Multi-Object and Multi-Source Association and Fusion
- **分类: cs.RO; eess.IV; eess.SP**

- **简介: 该论文属于多目标多源融合任务，解决自动驾驶中协作感知的通信与模型保护问题。提出一种基于共享3D边界框属性的晚期融合框架，提升精度并降低误差。**

- **链接: [http://arxiv.org/pdf/2507.02430v1](http://arxiv.org/pdf/2507.02430v1)**

> **作者:** Maryem Fadili; Mohamed Anis Ghaoui; Louis Lecrosnier; Steve Pechberti; Redouane Khemmar
>
> **摘要:** In autonomous driving, recent research has increasingly focused on collaborative perception based on deep learning to overcome the limitations of individual perception systems. Although these methods achieve high accuracy, they rely on high communication bandwidth and require unrestricted access to each agent's object detection model architecture and parameters. These constraints pose challenges real-world autonomous driving scenarios, where communication limitations and the need to safeguard proprietary models hinder practical implementation. To address this issue, we introduce a novel late collaborative framework for 3D multi-source and multi-object fusion, which operates solely on shared 3D bounding box attributes-category, size, position, and orientation-without necessitating direct access to detection models. Our framework establishes a new state-of-the-art in late fusion, achieving up to five times lower position error compared to existing methods. Additionally, it reduces scale error by a factor of 7.5 and orientation error by half, all while maintaining perfect 100% precision and recall when fusing detections from heterogeneous perception systems. These results highlight the effectiveness of our approach in addressing real-world collaborative perception challenges, setting a new benchmark for efficient and scalable multi-agent fusion.
>
---
#### [new 006] Effective Explanations for Belief-Desire-Intention Robots: When and What to Explain
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人机交互任务，旨在解决机器人在异常行为时如何有效解释的问题。研究用户对解释的需求与内容偏好，并提出算法以生成简洁有效的解释。**

- **链接: [http://arxiv.org/pdf/2507.02016v1](http://arxiv.org/pdf/2507.02016v1)**

> **作者:** Cong Wang; Roberto Calandra; Verena Klös
>
> **备注:** Paper accepted at IEEE RO-MAN 2025; 6 pages
>
> **摘要:** When robots perform complex and context-dependent tasks in our daily lives, deviations from expectations can confuse users. Explanations of the robot's reasoning process can help users to understand the robot intentions. However, when to provide explanations and what they contain are important to avoid user annoyance. We have investigated user preferences for explanation demand and content for a robot that helps with daily cleaning tasks in a kitchen. Our results show that users want explanations in surprising situations and prefer concise explanations that clearly state the intention behind the confusing action and the contextual factors that were relevant to this decision. Based on these findings, we propose two algorithms to identify surprising actions and to construct effective explanations for Belief-Desire-Intention (BDI) robots. Our algorithms can be easily integrated in the BDI reasoning process and pave the way for better human-robot interaction with context- and user-specific explanations.
>
---
#### [new 007] Vibration of Soft, Twisted Beams for Under-Actuated Quadrupedal Locomotion
- **分类: cs.RO**

- **简介: 该论文属于机器人学任务，旨在解决欠驱动机器人的运动控制问题。通过设计柔性腿部结构，利用振动实现多种移动模式，并验证其运动性能与鲁棒性。**

- **链接: [http://arxiv.org/pdf/2507.02547v1](http://arxiv.org/pdf/2507.02547v1)**

> **作者:** Yuhao Jiang; Fuchen Chen; Jamie Paik; Daniel M. Aukes
>
> **备注:** This manuscript is under revision for possible publication in the IEEE/ASME Transactions on Mechatronics. Copyright may be transferred to IEEE if the manuscript is accepted for publication, without further notice. Supplementary videos: https://youtu.be/T3d6FT3Rx-s, https://youtu.be/nPQrhKlN02E
>
> **摘要:** Under-actuated compliant robotic systems offer a promising approach to mitigating actuation and control challenges by harnessing pre-designed, embodied dynamic behaviors. This paper presents Flix-Walker, a novel, untethered, centimeter-scale quadrupedal robot inspired by compliant under-actuated mechanisms. Flix-Walker employs flexible, helix-shaped beams as legs, which are actuated by vibrations from just two motors to achieve three distinct mobility modes. We analyze the actuation parameters required to generate various locomotion modes through both simulation and prototype experiments. The effects of system and environmental variations on locomotion performance are examined, and we propose a generic metric for selecting control parameters that produce robust and functional motions. Experiments validate the effectiveness and robustness of these actuation parameters within a closed-loop control framework, demonstrating reliable trajectory-tracking and self-navigation capabilities.
>
---
#### [new 008] GPS-DRIFT: Marine Surface Robot Localization using IMU-GPS Fusion and Invariant Filtering
- **分类: cs.RO**

- **简介: 该论文属于机器人定位任务，解决GPS与IMU数据融合中的姿态和航向估计问题，提出一种基于不变滤波的传感器融合方法。**

- **链接: [http://arxiv.org/pdf/2507.02198v1](http://arxiv.org/pdf/2507.02198v1)**

> **作者:** Surya Pratap Singh; Tsimafei Lazouski; Maani Ghaffari
>
> **备注:** 6 pages
>
> **摘要:** This paper presents an extension of the DRIFT invariant state estimation framework, enabling robust fusion of GPS and IMU data for accurate pose and heading estimation. Originally developed for testing and usage on a marine autonomous surface vehicle (ASV), this approach can also be utilized on other mobile systems. Building upon the original proprioceptive only DRIFT algorithm, we develop a symmetry-preserving sensor fusion pipeline utilizing the invariant extended Kalman filter (InEKF) to integrate global position updates from GPS directly into the correction step. Crucially, we introduce a novel heading correction mechanism that leverages GPS course-over-ground information in conjunction with IMU orientation, overcoming the inherent unobservability of yaw in dead-reckoning. The system was deployed and validated on a customized Blue Robotics BlueBoat, but the methodological focus is on the algorithmic approach to fusing exteroceptive and proprioceptive sensors for drift-free localization and reliable orientation estimation. This work provides an open source solution for accurate yaw observation and localization in challenging or GPS-degraded conditions, and lays the groundwork for future experimental and comparative studies.
>
---
#### [new 009] Trajectory Optimization for Differential Drive Mobile Manipulators via Topological Paths Search and Arc Length-Yaw Parameterization
- **分类: cs.RO**

- **简介: 该论文属于移动机械臂轨迹优化任务，解决非完整动力学下的路径规划问题。通过拓扑路径搜索和弧长-偏航参数化方法，实现高效可行的运动规划。**

- **链接: [http://arxiv.org/pdf/2507.02761v1](http://arxiv.org/pdf/2507.02761v1)**

> **作者:** Long Xu; Choilam Wong; Mengke Zhang; Junxiao Lin; Fei Gao
>
> **备注:** Technical Report
>
> **摘要:** We present an efficient hierarchical motion planning pipeline for differential drive mobile manipulators. Our approach first searches for multiple collisionfree and topologically distinct paths for the mobile base to extract the space in which optimal solutions may exist. Further sampling and optimization are then conducted in parallel to explore feasible whole-body trajectories. For trajectory optimization, we employ polynomial trajectories and arc length-yaw parameterization, enabling efficient handling of the nonholonomic dynamics while ensuring optimality.
>
---
#### [new 010] CoInfra: A Large-Scale Cooperative Infrastructure Perception System and Dataset in Adverse Weather
- **分类: cs.RO**

- **简介: 该论文提出CoInfra系统及数据集，用于解决恶劣天气下多智能体感知问题，通过协同感知与实时数据融合提升自动驾驶可靠性。**

- **链接: [http://arxiv.org/pdf/2507.02245v1](http://arxiv.org/pdf/2507.02245v1)**

> **作者:** Minghao Ning; Yufeng Yang; Keqi Shu; Shucheng Huang; Jiaming Zhong; Maryam Salehi; Mahdi Rahmani; Yukun Lu; Chen Sun; Aladdin Saleh; Ehsan Hashemi; Amir Khajepour
>
> **备注:** This paper has been submitted to the IEEE Transactions on Robotics for review
>
> **摘要:** We present CoInfra, a large-scale cooperative infrastructure perception system and dataset designed to advance robust multi-agent perception under real-world and adverse weather conditions. The CoInfra system includes 14 fully synchronized sensor nodes, each equipped with dual RGB cameras and a LiDAR, deployed across a shared region and operating continuously to capture all traffic participants in real-time. A robust, delay-aware synchronization protocol and a scalable system architecture that supports real-time data fusion, OTA management, and remote monitoring are provided in this paper. On the other hand, the dataset was collected in different weather scenarios, including sunny, rainy, freezing rain, and heavy snow and includes 195k LiDAR frames and 390k camera images from 8 infrastructure nodes that are globally time-aligned and spatially calibrated. Furthermore, comprehensive 3D bounding box annotations for five object classes (i.e., car, bus, truck, person, and bicycle) are provided in both global and individual node frames, along with high-definition maps for contextual understanding. Baseline experiments demonstrate the trade-offs between early and late fusion strategies, the significant benefits of HD map integration are discussed. By openly releasing our dataset, codebase, and system documentation at https://github.com/NingMingHao/CoInfra, we aim to enable reproducible research and drive progress in infrastructure-supported autonomous driving, particularly in challenging, real-world settings.
>
---
#### [new 011] RoboBrain 2.0 Technical Report
- **分类: cs.RO**

- **简介: 该论文提出RoboBrain 2.0，属于具身智能任务，解决物理环境中感知、推理与规划问题，通过两种规模模型实现空间和时间决策能力。**

- **链接: [http://arxiv.org/pdf/2507.02029v1](http://arxiv.org/pdf/2507.02029v1)**

> **作者:** BAAI RoboBrain Team; Mingyu Cao; Huajie Tan; Yuheng Ji; Minglan Lin; Zhiyu Li; Zhou Cao; Pengwei Wang; Enshen Zhou; Yi Han; Yingbo Tang; Xiangqi Xu; Wei Guo; Yaoxu Lyu; Yijie Xu; Jiayu Shi; Cheng Chi; Mengdi Zhao; Xiaoshuai Hao; Shanyu Rong; Zhengliang Cai; Bolun Zhang; Shuyi Zhang; Huaihai Lyu; Mengfei Du; Lingfeng Zhang; Xi Feng; Xiaodan Liu; Yance Jiao; Chenrui He; Mengsi Lyu; Zhuo Chen; Yulong Ao; Xue Sun; Zheqi He; Jingshu Zheng; Xi Yang; Donghai Shi; Kunchang Xie; Bochao Zhang; Shaokai Nie; Chunlei Men; Yonghua Lin; Zhongyuan Wang; Tiejun Huang; Shanghang Zhang
>
> **摘要:** We introduce RoboBrain 2.0, our latest generation of embodied vision-language foundation models, designed to unify perception, reasoning, and planning for complex embodied tasks in physical environments. It comes in two variants: a lightweight 7B model and a full-scale 32B model, featuring a heterogeneous architecture with a vision encoder and a language model. Despite its compact size, RoboBrain 2.0 achieves strong performance across a wide spectrum of embodied reasoning tasks. On both spatial and temporal benchmarks, the 32B variant achieves leading results, surpassing prior open-source and proprietary models. In particular, it supports key real-world embodied AI capabilities, including spatial understanding (e.g., affordance prediction, spatial referring, trajectory forecasting) and temporal decision-making (e.g., closed-loop interaction, multi-agent long-horizon planning, and scene graph updating). This report details the model architecture, data construction, multi-stage training strategies, infrastructure and practical applications. We hope RoboBrain 2.0 advances embodied AI research and serves as a practical step toward building generalist embodied agents. The code, checkpoint and benchmark are available at https://superrobobrain.github.io.
>
---
#### [new 012] HAC-LOCO: Learning Hierarchical Active Compliance Control for Quadruped Locomotion under Continuous External Disturbances
- **分类: cs.RO**

- **简介: 该论文属于四足机器人控制任务，旨在解决外部扰动下的鲁棒与柔顺平衡问题。通过分阶段学习框架，实现对扰动的主动响应，提升运动性能。**

- **链接: [http://arxiv.org/pdf/2507.02447v1](http://arxiv.org/pdf/2507.02447v1)**

> **作者:** Xiang Zhou; Xinyu Zhang; Qingrui Zhang
>
> **备注:** 8 pages, 7 Figures
>
> **摘要:** Despite recent remarkable achievements in quadruped control, it remains challenging to ensure robust and compliant locomotion in the presence of unforeseen external disturbances. Existing methods prioritize locomotion robustness over compliance, often leading to stiff, high-frequency motions, and energy inefficiency. This paper, therefore, presents a two-stage hierarchical learning framework that can learn to take active reactions to external force disturbances based on force estimation. In the first stage, a velocity-tracking policy is trained alongside an auto-encoder to distill historical proprioceptive features. A neural network-based estimator is learned through supervised learning, which estimates body velocity and external forces based on proprioceptive measurements. In the second stage, a compliance action module, inspired by impedance control, is learned based on the pre-trained encoder and policy. This module is employed to actively adjust velocity commands in response to external forces based on real-time force estimates. With the compliance action module, a quadruped robot can robustly handle minor disturbances while appropriately yielding to significant forces, thus striking a balance between robustness and compliance. Simulations and real-world experiments have demonstrated that our method has superior performance in terms of robustness, energy efficiency, and safety. Experiment comparison shows that our method outperforms the state-of-the-art RL-based locomotion controllers. Ablation studies are given to show the critical roles of the compliance action module.
>
---
#### [new 013] MISCGrasp: Leveraging Multiple Integrated Scales and Contrastive Learning for Enhanced Volumetric Grasping
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人抓取任务，旨在解决不同形状和大小物体的自适应抓取问题。通过多尺度特征提取与对比学习，提升抓取性能。**

- **链接: [http://arxiv.org/pdf/2507.02672v1](http://arxiv.org/pdf/2507.02672v1)**

> **作者:** Qingyu Fan; Yinghao Cai; Chao Li; Chunting Jiao; Xudong Zheng; Tao Lu; Bin Liang; Shuo Wang
>
> **备注:** IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS), 2025
>
> **摘要:** Robotic grasping faces challenges in adapting to objects with varying shapes and sizes. In this paper, we introduce MISCGrasp, a volumetric grasping method that integrates multi-scale feature extraction with contrastive feature enhancement for self-adaptive grasping. We propose a query-based interaction between high-level and low-level features through the Insight Transformer, while the Empower Transformer selectively attends to the highest-level features, which synergistically strikes a balance between focusing on fine geometric details and overall geometric structures. Furthermore, MISCGrasp utilizes multi-scale contrastive learning to exploit similarities among positive grasp samples, ensuring consistency across multi-scale features. Extensive experiments in both simulated and real-world environments demonstrate that MISCGrasp outperforms baseline and variant methods in tabletop decluttering tasks. More details are available at https://miscgrasp.github.io/.
>
---
#### [new 014] A Vehicle-in-the-Loop Simulator with AI-Powered Digital Twins for Testing Automated Driving Controllers
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于自动驾驶控制器测试任务，旨在解决传统仿真成本高、精度低的问题。通过引入缩比车辆和AI数字孪生技术，构建高效可靠的测试平台。**

- **链接: [http://arxiv.org/pdf/2507.02313v1](http://arxiv.org/pdf/2507.02313v1)**

> **作者:** Zengjie Zhang; Giannis Badakis; Michalis Galanis; Adem Bavarşi; Edwin van Hassel; Mohsen Alirezaei; Sofie Haesaert
>
> **摘要:** Simulators are useful tools for testing automated driving controllers. Vehicle-in-the-loop (ViL) tests and digital twins (DTs) are widely used simulation technologies to facilitate the smooth deployment of controllers to physical vehicles. However, conventional ViL tests rely on full-size vehicles, requiring large space and high expenses. Also, physical-model-based DT suffers from the reality gap caused by modeling imprecision. This paper develops a comprehensive and practical simulator for testing automated driving controllers enhanced by scaled physical cars and AI-powered DT models. The scaled cars allow for saving space and expenses of simulation tests. The AI-powered DT models ensure superior simulation fidelity. Moreover, the simulator integrates well with off-the-shelf software and control algorithms, making it easy to extend. We use a filtered control benchmark with formal safety guarantees to showcase the capability of the simulator in validating automated driving controllers. Experimental studies are performed to showcase the efficacy of the simulator, implying its great potential in validating control solutions for autonomous vehicles and intelligent traffic.
>
---
#### [new 015] ArtGS:3D Gaussian Splatting for Interactive Visual-Physical Modeling and Manipulation of Articulated Objects
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决关节物体建模与操控问题。提出ArtGS框架，结合视觉-物理建模和3D高斯点云渲染，提升操控准确性与适应性。**

- **链接: [http://arxiv.org/pdf/2507.02600v1](http://arxiv.org/pdf/2507.02600v1)**

> **作者:** Qiaojun Yu; Xibin Yuan; Yu jiang; Junting Chen; Dongzhe Zheng; Ce Hao; Yang You; Yixing Chen; Yao Mu; Liu Liu; Cewu Lu
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** Articulated object manipulation remains a critical challenge in robotics due to the complex kinematic constraints and the limited physical reasoning of existing methods. In this work, we introduce ArtGS, a novel framework that extends 3D Gaussian Splatting (3DGS) by integrating visual-physical modeling for articulated object understanding and interaction. ArtGS begins with multi-view RGB-D reconstruction, followed by reasoning with a vision-language model (VLM) to extract semantic and structural information, particularly the articulated bones. Through dynamic, differentiable 3DGS-based rendering, ArtGS optimizes the parameters of the articulated bones, ensuring physically consistent motion constraints and enhancing the manipulation policy. By leveraging dynamic Gaussian splatting, cross-embodiment adaptability, and closed-loop optimization, ArtGS establishes a new framework for efficient, scalable, and generalizable articulated object modeling and manipulation. Experiments conducted in both simulation and real-world environments demonstrate that ArtGS significantly outperforms previous methods in joint estimation accuracy and manipulation success rates across a variety of articulated objects. Additional images and videos are available on the project website: https://sites.google.com/view/artgs/home
>
---
#### [new 016] Towards Bio-Inspired Robotic Trajectory Planning via Self-Supervised RNN
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于机器人轨迹规划任务，旨在解决传统方法计算复杂的问题。通过自监督RNN学习轨迹，无需大量标注数据即可生成有效路径。**

- **链接: [http://arxiv.org/pdf/2507.02171v1](http://arxiv.org/pdf/2507.02171v1)**

> **作者:** Miroslav Cibula; Kristína Malinovská; Matthias Kerzel
>
> **备注:** 12 pages, 4 figures, 2 tables. To be published in 2025 International Conference on Artificial Neural Networks (ICANN) proceedings. This research was funded by the Horizon Europe project TERAIS, GA no. 101079338, and in part by the Slovak Grant Agency for Science (VEGA), project 1/0373/23
>
> **摘要:** Trajectory planning in robotics is understood as generating a sequence of joint configurations that will lead a robotic agent, or its manipulator, from an initial state to the desired final state, thus completing a manipulation task while considering constraints like robot kinematics and the environment. Typically, this is achieved via sampling-based planners, which are computationally intensive. Recent advances demonstrate that trajectory planning can also be performed by supervised sequence learning of trajectories, often requiring only a single or fixed number of passes through a neural architecture, thus ensuring a bounded computation time. Such fully supervised approaches, however, perform imitation learning; they do not learn based on whether the trajectories can successfully reach a goal, but try to reproduce observed trajectories. In our work, we build on this approach and propose a cognitively inspired self-supervised learning scheme based on a recurrent architecture for building a trajectory model. We evaluate the feasibility of the proposed method on a task of kinematic planning for a robotic arm. The results suggest that the model is able to learn to generate trajectories only using given paired forward and inverse kinematics models, and indicate that this novel method could facilitate planning for more complex manipulation tasks requiring adaptive solutions.
>
---
#### [new 017] MultiGen: Using Multimodal Generation in Simulation to Learn Multimodal Policies in Real
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人学习任务，旨在解决多模态策略在现实中的迁移问题。通过整合生成模型到物理模拟器中，实现多感官仿真，提升机器人多模态感知与决策能力。**

- **链接: [http://arxiv.org/pdf/2507.02864v1](http://arxiv.org/pdf/2507.02864v1)**

> **作者:** Renhao Wang; Haoran Geng; Tingle Li; Feishi Wang; Gopala Anumanchipalli; Philipp Wu; Trevor Darrell; Boyi Li; Pieter Abbeel; Jitendra Malik; Alexei A. Efros
>
> **摘要:** Robots must integrate multiple sensory modalities to act effectively in the real world. Yet, learning such multimodal policies at scale remains challenging. Simulation offers a viable solution, but while vision has benefited from high-fidelity simulators, other modalities (e.g. sound) can be notoriously difficult to simulate. As a result, sim-to-real transfer has succeeded primarily in vision-based tasks, with multimodal transfer still largely unrealized. In this work, we tackle these challenges by introducing MultiGen, a framework that integrates large-scale generative models into traditional physics simulators, enabling multisensory simulation. We showcase our framework on the dynamic task of robot pouring, which inherently relies on multimodal feedback. By synthesizing realistic audio conditioned on simulation video, our method enables training on rich audiovisual trajectories -- without any real robot data. We demonstrate effective zero-shot transfer to real-world pouring with novel containers and liquids, highlighting the potential of generative modeling to both simulate hard-to-model modalities and close the multimodal sim-to-real gap.
>
---
#### [new 018] Safe and Socially Aware Multi-Robot Coordination in Multi-Human Social Care Settings
- **分类: cs.RO**

- **简介: 该论文属于多机器人协作任务，解决多人类环境下的路径规划、任务分配等问题，提出基于学习的协调方法。**

- **链接: [http://arxiv.org/pdf/2507.02521v1](http://arxiv.org/pdf/2507.02521v1)**

> **作者:** Ayodeji O. Abioye; Jayati Deshmukh; Athina Georgara; Dominic Price; Tuyen Nguyen; Aleksandra Landowska; Amel Bennaceur; Joel E. Fischer; Sarvapali D. Ramchurn
>
> **备注:** 3 pages, 1 figure. Accepted for poster presentation at the UK AI Research Symposium (UKAIR) 2025, themed "A Festival of Ideas", being held in Newcastle from 8th - 9th September, 2025. https://www.ukairs.ac.uk/
>
> **摘要:** This research investigates strategies for multi-robot coordination in multi-human environments. It proposes a multi-objective learning-based coordination approach to addressing the problem of path planning, navigation, task scheduling, task allocation, and human-robot interaction in multi-human multi-robot (MHMR) settings.
>
---
#### [new 019] cVLA: Towards Efficient Camera-Space VLAs
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于机器人操控任务，旨在解决VLA模型训练成本高的问题。提出cVLA方法，通过2D图像直接预测机械臂位姿，提升训练效率与泛化性。**

- **链接: [http://arxiv.org/pdf/2507.02190v1](http://arxiv.org/pdf/2507.02190v1)**

> **作者:** Max Argus; Jelena Bratulic; Houman Masnavi; Maxim Velikanov; Nick Heppert; Abhinav Valada; Thomas Brox
>
> **备注:** 20 pages, 10 figures
>
> **摘要:** Vision-Language-Action (VLA) models offer a compelling framework for tackling complex robotic manipulation tasks, but they are often expensive to train. In this paper, we propose a novel VLA approach that leverages the competitive performance of Vision Language Models (VLMs) on 2D images to directly infer robot end-effector poses in image frame coordinates. Unlike prior VLA models that output low-level controls, our model predicts trajectory waypoints, making it both more efficient to train and robot embodiment agnostic. Despite its lightweight design, our next-token prediction architecture effectively learns meaningful and executable robot trajectories. We further explore the underutilized potential of incorporating depth images, inference-time techniques such as decoding strategies, and demonstration-conditioned action generation. Our model is trained on a simulated dataset and exhibits strong sim-to-real transfer capabilities. We evaluate our approach using a combination of simulated and real data, demonstrating its effectiveness on a real robotic system.
>
---
#### [new 020] Path Planning using a One-shot-sampling Skeleton Map
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于路径规划任务，旨在解决移动机器人在结构化环境中快速安全导航的问题。通过提出SkelUnet方法，利用深度学习生成骨架图，提升路径规划效率与安全性。**

- **链接: [http://arxiv.org/pdf/2507.02328v1](http://arxiv.org/pdf/2507.02328v1)**

> **作者:** Gabriel O. Flores-Aquino; Octavio Gutierrez-Frias; Juan Irving Vasquez
>
> **摘要:** Path planning algorithms aim to compute a collision-free path, and many works focus on finding the optimal distance path. However, for some applications, a more suitable approach is to balance response time, safety of the paths, and path length. In this context, a skeleton map is a useful tool in graph-based schemes, as it provides an intrinsic representation of free configuration space. However, skeletonization algorithms are very resource-intensive, being primarily oriented towards image processing tasks. We propose an efficient path-planning methodology that finds safe paths within an acceptable processing time. This methodology leverages a Deep Denoising Auto-Encoder (DDAE) based on U-Net architecture to compute a skeletonized version of the navigation map, which we refer to as SkelUnet. The SkelUnet network facilitates exploration of the entire workspace through one-shot sampling (OSS), as opposed to the iterative process used by exact algorithms or the probabilistic sampling process. SkelUnet is trained and tested on a dataset consisting of 12,500 bi-dimensional dungeon maps. The motion planning methodology is evaluated in a simulation environment for an Unmanned Aerial Vehicle (UAV) using 250 previously unseen maps, and assessed with various navigation metrics to quantify the navigability of the computed paths. The results demonstrate that using SkelUnet to construct a roadmap offers significant advantages, such as connecting all regions of free workspace, providing safer paths, and reducing processing times. These characteristics make this method particularly suitable for mobile service robots in structured environments.
>
---
#### [new 021] Uncertainty-aware Reward Design Process
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决奖励函数设计效率与质量不足的问题。通过结合大语言模型和贝叶斯优化，提出URDP框架提升自动化奖励设计效果。**

- **链接: [http://arxiv.org/pdf/2507.02256v1](http://arxiv.org/pdf/2507.02256v1)**

> **作者:** Yang Yang; Xiaolu Zhou; Bosong Ding; Miao Xin
>
> **备注:** 34 pages, 9 figures
>
> **摘要:** Designing effective reward functions is a cornerstone of reinforcement learning (RL), yet it remains a challenging process due to the inefficiencies and inconsistencies inherent in conventional reward engineering methodologies. Recent advances have explored leveraging large language models (LLMs) to automate reward function design. However, their suboptimal performance in numerical optimization often yields unsatisfactory reward quality, while the evolutionary search paradigm demonstrates inefficient utilization of simulation resources, resulting in prohibitively lengthy design cycles with disproportionate computational overhead. To address these challenges, we propose the Uncertainty-aware Reward Design Process (URDP), a novel framework that integrates large language models to streamline reward function design and evaluation in RL environments. URDP quantifies candidate reward function uncertainty based on self-consistency analysis, enabling simulation-free identification of ineffective reward components while discovering novel reward components. Furthermore, we introduce uncertainty-aware Bayesian optimization (UABO), which incorporates uncertainty estimation to significantly enhance hyperparameter configuration efficiency. Finally, we construct a bi-level optimization architecture by decoupling the reward component optimization and the hyperparameter tuning. URDP orchestrates synergistic collaboration between the reward logic reasoning of the LLMs and the numerical optimization strengths of the Bayesian Optimization. We conduct a comprehensive evaluation of URDP across 35 diverse tasks spanning three benchmark environments. Our experimental results demonstrate that URDP not only generates higher-quality reward functions but also achieves significant improvements in the efficiency of automated reward design compared to existing approaches.
>
---
#### [new 022] Grounding Intelligence in Movement
- **分类: cs.AI; cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于AI与运动建模任务，旨在解决运动数据建模不足的问题，提出将运动作为核心建模目标，以提升智能系统对世界的理解能力。**

- **链接: [http://arxiv.org/pdf/2507.02771v1](http://arxiv.org/pdf/2507.02771v1)**

> **作者:** Melanie Segado; Felipe Parodi; Jordan K. Matelsky; Michael L. Platt; Eva B. Dyer; Konrad P. Kording
>
> **备注:** 9 pages, 2 figures
>
> **摘要:** Recent advances in machine learning have dramatically improved our ability to model language, vision, and other high-dimensional data, yet they continue to struggle with one of the most fundamental aspects of biological systems: movement. Across neuroscience, medicine, robotics, and ethology, movement is essential for interpreting behavior, predicting intent, and enabling interaction. Despite its core significance in our intelligence, movement is often treated as an afterthought rather than as a rich and structured modality in its own right. This reflects a deeper fragmentation in how movement data is collected and modeled, often constrained by task-specific goals and domain-specific assumptions. But movement is not domain-bound. It reflects shared physical constraints, conserved morphological structures, and purposeful dynamics that cut across species and settings. We argue that movement should be treated as a primary modeling target for AI. It is inherently structured and grounded in embodiment and physics. This structure, often allowing for compact, lower-dimensional representations (e.g., pose), makes it more interpretable and computationally tractable to model than raw, high-dimensional sensory inputs. Developing models that can learn from and generalize across diverse movement data will not only advance core capabilities in generative modeling and control, but also create a shared foundation for understanding behavior across biological and artificial systems. Movement is not just an outcome, it is a window into how intelligent systems engage with the world.
>
---
#### [new 023] Red grape detection with accelerated artificial neural networks in the FPGA's programmable logic
- **分类: cs.CV; cs.AI; cs.DC; cs.LG; cs.RO**

- **简介: 该论文属于目标检测任务，旨在解决机器人在移动中检测物体速度慢的问题。通过在FPGA中部署加速的神经网络模型，提升检测效率。**

- **链接: [http://arxiv.org/pdf/2507.02443v1](http://arxiv.org/pdf/2507.02443v1)**

> **作者:** Sandro Costa Magalhães; Marco Almeida; Filipe Neves dos Santos; António Paulo Moreira; Jorge Dias
>
> **备注:** Submitted to ROBOT'2025
>
> **摘要:** Robots usually slow down for canning to detect objects while moving. Additionally, the robot's camera is configured with a low framerate to track the velocity of the detection algorithms. This would be constrained while executing tasks and exploring, making robots increase the task execution time. AMD has developed the Vitis-AI framework to deploy detection algorithms into FPGAs. However, this tool does not fully use the FPGAs' PL. In this work, we use the FINN architecture to deploy three ANNs, MobileNet v1 with 4-bit quantisation, CNV with 2-bit quantisation, and CNV with 1-bit quantisation (BNN), inside an FPGA's PL. The models were trained on the RG2C dataset. This is a self-acquired dataset released in open access. MobileNet v1 performed better, reaching a success rate of 98 % and an inference speed of 6611 FPS. In this work, we proved that we can use FPGAs to speed up ANNs and make them suitable for attention mechanisms.
>
---
#### [new 024] DexVLG: Dexterous Vision-Language-Grasp Model at Scale
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出DexVLG模型，解决机器人基于语言指令进行精细抓取的任务。通过大规模数据训练，实现高精度的物体部分抓取。**

- **链接: [http://arxiv.org/pdf/2507.02747v1](http://arxiv.org/pdf/2507.02747v1)**

> **作者:** Jiawei He; Danshi Li; Xinqiang Yu; Zekun Qi; Wenyao Zhang; Jiayi Chen; Zhaoxiang Zhang; Zhizheng Zhang; Li Yi; He Wang
>
> **摘要:** As large models gain traction, vision-language-action (VLA) systems are enabling robots to tackle increasingly complex tasks. However, limited by the difficulty of data collection, progress has mainly focused on controlling simple gripper end-effectors. There is little research on functional grasping with large models for human-like dexterous hands. In this paper, we introduce DexVLG, a large Vision-Language-Grasp model for Dexterous grasp pose prediction aligned with language instructions using single-view RGBD input. To accomplish this, we generate a dataset of 170 million dexterous grasp poses mapped to semantic parts across 174,000 objects in simulation, paired with detailed part-level captions. This large-scale dataset, named DexGraspNet 3.0, is used to train a VLM and flow-matching-based pose head capable of producing instruction-aligned grasp poses for tabletop objects. To assess DexVLG's performance, we create benchmarks in physics-based simulations and conduct real-world experiments. Extensive testing demonstrates DexVLG's strong zero-shot generalization capabilities-achieving over 76% zero-shot execution success rate and state-of-the-art part-grasp accuracy in simulation-and successful part-aligned grasps on physical objects in real-world scenarios.
>
---
## 更新

#### [replaced 001] Aerial Vision-and-Language Navigation via Semantic-Topo-Metric Representation Guided LLM Reasoning
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2410.08500v2](http://arxiv.org/pdf/2410.08500v2)**

> **作者:** Yunpeng Gao; Zhigang Wang; Linglin Jing; Dong Wang; Xuelong Li; Bin Zhao
>
> **摘要:** Aerial Vision-and-Language Navigation (VLN) is a novel task enabling Unmanned Aerial Vehicles (UAVs) to navigate in outdoor environments through natural language instructions and visual cues. It remains challenging due to the complex spatial relationships in outdoor aerial scenes. In this paper, we propose an end-to-end zero-shot framework for aerial VLN tasks, where the large language model (LLM) is introduced as our agent for action prediction. Specifically, we develop a novel Semantic-Topo-Metric Representation (STMR) to enhance the spatial reasoning ability of LLMs. This is achieved by extracting and projecting instruction-related semantic masks of landmarks into a top-down map that contains the location information of surrounding landmarks. Further, this map is transformed into a matrix representation with distance metrics as the text prompt to the LLM, for action prediction according to the instruction. Experiments conducted in real and simulation environments have successfully proved the effectiveness and robustness of our method, achieving 15.9% and 12.5% improvements (absolute) in Oracle Success Rate (OSR) on AerialVLN-S dataset.
>
---
#### [replaced 002] HAPI: A Model for Learning Robot Facial Expressions from Human Preferences
- **分类: cs.RO; cs.AI; cs.CV; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.17046v2](http://arxiv.org/pdf/2503.17046v2)**

> **作者:** Dongsheng Yang; Qianying Liu; Wataru Sato; Takashi Minato; Chaoran Liu; Shin'ya Nishida
>
> **备注:** Accepted to IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** Automatic robotic facial expression generation is crucial for human-robot interaction, as handcrafted methods based on fixed joint configurations often yield rigid and unnatural behaviors. Although recent automated techniques reduce the need for manual tuning, they tend to fall short by not adequately bridging the gap between human preferences and model predictions-resulting in a deficiency of nuanced and realistic expressions due to limited degrees of freedom and insufficient perceptual integration. In this work, we propose a novel learning-to-rank framework that leverages human feedback to address this discrepancy and enhanced the expressiveness of robotic faces. Specifically, we conduct pairwise comparison annotations to collect human preference data and develop the Human Affective Pairwise Impressions (HAPI) model, a Siamese RankNet-based approach that refines expression evaluation. Results obtained via Bayesian Optimization and online expression survey on a 35-DOF android platform demonstrate that our approach produces significantly more realistic and socially resonant expressions of Anger, Happiness, and Surprise than those generated by baseline and expert-designed methods. This confirms that our framework effectively bridges the gap between human preferences and model predictions while robustly aligning robotic expression generation with human affective responses.
>
---
#### [replaced 003] High-Performance Reinforcement Learning on Spot: Optimizing Simulation Parameters with Distributional Measures
- **分类: cs.LG; cs.RO**

- **链接: [http://arxiv.org/pdf/2504.17857v3](http://arxiv.org/pdf/2504.17857v3)**

> **作者:** AJ Miller; Fangzhou Yu; Michael Brauckmann; Farbod Farshidian
>
> **摘要:** This work presents an overview of the technical details behind a high performance reinforcement learning policy deployment with the Spot RL Researcher Development Kit for low level motor access on Boston Dynamics Spot. This represents the first public demonstration of an end to end end reinforcement learning policy deployed on Spot hardware with training code publicly available through Nvidia IsaacLab and deployment code available through Boston Dynamics. We utilize Wasserstein Distance and Maximum Mean Discrepancy to quantify the distributional dissimilarity of data collected on hardware and in simulation to measure our sim2real gap. We use these measures as a scoring function for the Covariance Matrix Adaptation Evolution Strategy to optimize simulated parameters that are unknown or difficult to measure from Spot. Our procedure for modeling and training produces high quality reinforcement learning policies capable of multiple gaits, including a flight phase. We deploy policies capable of over 5.2ms locomotion, more than triple Spots default controller maximum speed, robustness to slippery surfaces, disturbance rejection, and overall agility previously unseen on Spot. We detail our method and release our code to support future work on Spot with the low level API.
>
---
#### [replaced 004] TiCoSS: Tightening the Coupling between Semantic Segmentation and Stereo Matching within A Joint Learning Framework
- **分类: cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2407.18038v4](http://arxiv.org/pdf/2407.18038v4)**

> **作者:** Guanfeng Tang; Zhiyuan Wu; Jiahang Li; Ping Zhong; We Ye; Xieyuanli Chen; Huiming Lu; Rui Fan
>
> **摘要:** Semantic segmentation and stereo matching, respectively analogous to the ventral and dorsal streams in our human brain, are two key components of autonomous driving perception systems. Addressing these two tasks with separate networks is no longer the mainstream direction in developing computer vision algorithms, particularly with the recent advances in large vision models and embodied artificial intelligence. The trend is shifting towards combining them within a joint learning framework, especially emphasizing feature sharing between the two tasks. The major contributions of this study lie in comprehensively tightening the coupling between semantic segmentation and stereo matching. Specifically, this study introduces three novelties: (1) a tightly coupled, gated feature fusion strategy, (2) a hierarchical deep supervision strategy, and (3) a coupling tightening loss function. The combined use of these technical contributions results in TiCoSS, a state-of-the-art joint learning framework that simultaneously tackles semantic segmentation and stereo matching. Through extensive experiments on the KITTI and vKITTI2 datasets, along with qualitative and quantitative analyses, we validate the effectiveness of our developed strategies and loss function, and demonstrate its superior performance compared to prior arts, with a notable increase in mIoU by over 9%. Our source code will be publicly available at mias.group/TiCoSS upon publication.
>
---
#### [replaced 005] Data Authorisation and Validation in Autonomous Vehicles: A Critical Review
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2405.17435v2](http://arxiv.org/pdf/2405.17435v2)**

> **作者:** Reem Alhabib; Poonam Yadav
>
> **备注:** 16 pages, 9 figures. This is the Author Version
>
> **摘要:** Autonomous systems are becoming increasingly prevalent in new vehicles. Due to their environmental friendliness and their remarkable capability to significantly enhance road safety, these vehicles have gained widespread recognition and acceptance in recent years. Automated Driving Systems (ADS) are intricate systems that incorporate a multitude of sensors and actuators to interact with the environment autonomously, pervasively, and interactively. Consequently, numerous studies are currently underway to keep abreast of these rapid developments. This paper aims to provide a comprehensive overview of recent advancements in ADS technologies. It provides in-depth insights into the detailed information about how data and information flow in the distributed system, including autonomous vehicles and other various supporting services and entities. Data validation and system requirements are emphasised, such as security, privacy, scalability, and data ownership, in accordance with regulatory standards. Finally, several current research directions in the AVs field will be discussed.
>
---
#### [replaced 006] ForceGrip: Reference-Free Curriculum Learning for Realistic Grip Force Control in VR Hand Manipulation
- **分类: cs.RO; cs.GR; cs.HC; cs.LG**

- **链接: [http://arxiv.org/pdf/2503.08061v4](http://arxiv.org/pdf/2503.08061v4)**

> **作者:** DongHeun Han; Byungmin Kim; RoUn Lee; KyeongMin Kim; Hyoseok Hwang; HyeongYeop Kang
>
> **备注:** 11 pages, 11 figures. Accepted to SIGGRAPH Conference Papers '25. Project page: https://han-dongheun.github.io/ForceGrip
>
> **摘要:** Realistic Hand manipulation is a key component of immersive virtual reality (VR), yet existing methods often rely on kinematic approach or motion-capture datasets that omit crucial physical attributes such as contact forces and finger torques. Consequently, these approaches prioritize tight, one-size-fits-all grips rather than reflecting users' intended force levels. We present ForceGrip, a deep learning agent that synthesizes realistic hand manipulation motions, faithfully reflecting the user's grip force intention. Instead of mimicking predefined motion datasets, ForceGrip uses generated training scenarios-randomizing object shapes, wrist movements, and trigger input flows-to challenge the agent with a broad spectrum of physical interactions. To effectively learn from these complex tasks, we employ a three-phase curriculum learning framework comprising Finger Positioning, Intention Adaptation, and Dynamic Stabilization. This progressive strategy ensures stable hand-object contact, adaptive force control based on user inputs, and robust handling under dynamic conditions. Additionally, a proximity reward function enhances natural finger motions and accelerates training convergence. Quantitative and qualitative evaluations reveal ForceGrip's superior force controllability and plausibility compared to state-of-the-art methods. Demo videos are available as supplementary material and the code is provided at https://han-dongheun.github.io/ForceGrip.
>
---
#### [replaced 007] Towards autonomous photogrammetric forest inventory using a lightweight under-canopy robotic drone
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2501.12073v2](http://arxiv.org/pdf/2501.12073v2)**

> **作者:** Väinö Karjalainen; Niko Koivumäki; Teemu Hakala; Jesse Muhojoki; Eric Hyyppä; Anand George; Juha Suomalainen; Eija Honkavaara
>
> **备注:** 36 pages, 11 Figures
>
> **摘要:** Drones are increasingly used in forestry to capture high-resolution remote sensing data, supporting enhanced monitoring, assessment, and decision-making processes. While operations above the forest canopy are already highly automated, flying inside forests remains challenging, primarily relying on manual piloting. Inside dense forests, reliance on the Global Navigation Satellite System (GNSS) for localization is not feasible. Additionally, the drone must autonomously adjust its flight path to avoid collisions. Recently, advancements in robotics have enabled autonomous drone flights in GNSS-denied obstacle-rich areas. In this article, a step towards autonomous forest data collection is taken by building a prototype of a robotic under-canopy drone utilizing state-of-the-art open-source methods and validating its performance for data collection inside forests. Specifically, the study focused on camera-based autonomous flight under the forest canopy and photogrammetric post-processing of the data collected with the low-cost onboard stereo camera. The autonomous flight capability of the prototype was evaluated through multiple test flights at boreal forests. The tree parameter estimation capability was studied by performing diameter at breast height (DBH) estimation. The prototype successfully carried out flights in selected challenging forest environments, and the experiments showed excellent performance in forest 3D modeling with a miniaturized stereoscopic photogrammetric system. The DBH estimation achieved a root mean square error (RMSE) of 3.33 cm (12.79 \%) across all trees. For trees with a DBH less than 30 cm, the RMSE was 1.16 cm (5.74 \%). The results provide valuable insights into autonomous under-canopy forest mapping and highlight the critical next steps for advancing lightweight robotic drone systems for mapping complex forest environments.
>
---
#### [replaced 008] AC-DiT: Adaptive Coordination Diffusion Transformer for Mobile Manipulation
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2507.01961v2](http://arxiv.org/pdf/2507.01961v2)**

> **作者:** Sixiang Chen; Jiaming Liu; Siyuan Qian; Han Jiang; Lily Li; Renrui Zhang; Zhuoyang Liu; Chenyang Gu; Chengkai Hou; Pengwei Wang; Zhongyuan Wang; Shanghang Zhang
>
> **备注:** Project website: https://ac-dit.github.io/
>
> **摘要:** Recently, mobile manipulation has attracted increasing attention for enabling language-conditioned robotic control in household tasks. However, existing methods still face challenges in coordinating mobile base and manipulator, primarily due to two limitations. On the one hand, they fail to explicitly model the influence of the mobile base on manipulator control, which easily leads to error accumulation under high degrees of freedom. On the other hand, they treat the entire mobile manipulation process with the same visual observation modality (e.g., either all 2D or all 3D), overlooking the distinct multimodal perception requirements at different stages during mobile manipulation. To address this, we propose the Adaptive Coordination Diffusion Transformer (AC-DiT), which enhances mobile base and manipulator coordination for end-to-end mobile manipulation. First, since the motion of the mobile base directly influences the manipulator's actions, we introduce a mobility-to-body conditioning mechanism that guides the model to first extract base motion representations, which are then used as context prior for predicting whole-body actions. This enables whole-body control that accounts for the potential impact of the mobile base's motion. Second, to meet the perception requirements at different stages of mobile manipulation, we design a perception-aware multimodal conditioning strategy that dynamically adjusts the fusion weights between various 2D visual images and 3D point clouds, yielding visual features tailored to the current perceptual needs. This allows the model to, for example, adaptively rely more on 2D inputs when semantic information is crucial for action prediction, while placing greater emphasis on 3D geometric information when precise spatial understanding is required. We validate AC-DiT through extensive experiments on both simulated and real-world mobile manipulation tasks.
>
---
#### [replaced 009] Image-Based Roadmaps for Vision-Only Planning and Control of Robotic Manipulators
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.19617v3](http://arxiv.org/pdf/2502.19617v3)**

> **作者:** Sreejani Chatterjee; Abhinav Gandhi; Berk Calli; Constantinos Chamzas
>
> **摘要:** This work presents a motion planning framework for robotic manipulators that computes collision-free paths directly in image space. The generated paths can then be tracked using vision-based control, eliminating the need for an explicit robot model or proprioceptive sensing. At the core of our approach is the construction of a roadmap entirely in image space. To achieve this, we explicitly define sampling, nearest-neighbor selection, and collision checking based on visual features rather than geometric models. We first collect a set of image-space samples by moving the robot within its workspace, capturing keypoints along its body at different configurations. These samples serve as nodes in the roadmap, which we construct using either learned or predefined distance metrics. At runtime, the roadmap generates collision-free paths directly in image space, removing the need for a robot model or joint encoders. We validate our approach through an experimental study in which a robotic arm follows planned paths using an adaptive vision-based control scheme to avoid obstacles. The results show that paths generated with the learned-distance roadmap achieved 100% success in control convergence, whereas the predefined image-space distance roadmap enabled faster transient responses but had a lower success rate in convergence.
>
---
#### [replaced 010] Observability Investigation for Rotational Calibration of (Global-pose aided) VIO under Straight Line Motion
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2503.00027v2](http://arxiv.org/pdf/2503.00027v2)**

> **作者:** Junlin Song; Antoine Richard; Miguel Olivares-Mendez
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** Online extrinsic calibration is crucial for building "power-on-and-go" moving platforms, like robots and AR devices. However, blindly performing online calibration for unobservable parameter may lead to unpredictable results. In the literature, extensive studies have been conducted on the extrinsic calibration between IMU and camera, from theory to practice. It is well-known that the observability of extrinsic parameter can be guaranteed under sufficient motion excitation. Furthermore, the impacts of degenerate motions are also investigated. Despite these successful analyses, we identify an issue with respect to the existing observability conclusion. This paper focuses on the observability investigation for straight line motion, which is a common-seen and fundamental degenerate motion in applications. We analytically prove that pure translational straight line motion can lead to the unobservability of the rotational extrinsic parameter between IMU and camera (at least one degree of freedom). By correcting the existing observability conclusion, our novel theoretical finding disseminates more precise principle to the research community and provides explainable calibration guideline for practitioners. Our analysis is validated by rigorous theory and experiments.
>
---
#### [replaced 011] AirV2X: Unified Air-Ground Vehicle-to-Everything Collaboration
- **分类: cs.CV; cs.AI; cs.RO**

- **链接: [http://arxiv.org/pdf/2506.19283v3](http://arxiv.org/pdf/2506.19283v3)**

> **作者:** Xiangbo Gao; Yuheng Wu; Fengze Yang; Xuewen Luo; Keshu Wu; Xinghao Chen; Yuping Wang; Chenxi Liu; Yang Zhou; Zhengzhong Tu
>
> **摘要:** While multi-vehicular collaborative driving demonstrates clear advantages over single-vehicle autonomy, traditional infrastructure-based V2X systems remain constrained by substantial deployment costs and the creation of "uncovered danger zones" in rural and suburban areas. We present AirV2X-Perception, a large-scale dataset that leverages Unmanned Aerial Vehicles (UAVs) as a flexible alternative or complement to fixed Road-Side Units (RSUs). Drones offer unique advantages over ground-based perception: complementary bird's-eye-views that reduce occlusions, dynamic positioning capabilities that enable hovering, patrolling, and escorting navigation rules, and significantly lower deployment costs compared to fixed infrastructure. Our dataset comprises 6.73 hours of drone-assisted driving scenarios across urban, suburban, and rural environments with varied weather and lighting conditions. The AirV2X-Perception dataset facilitates the development and standardized evaluation of Vehicle-to-Drone (V2D) algorithms, addressing a critical gap in the rapidly expanding field of aerial-assisted autonomous driving systems. The dataset and development kits are open-sourced at https://github.com/taco-group/AirV2X-Perception.
>
---
#### [replaced 012] Adaptive Koopman Model Predictive Control of Simple Serial Robots
- **分类: eess.SY; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2503.17902v2](http://arxiv.org/pdf/2503.17902v2)**

> **作者:** Adriano del Río; Christoph Stoeffler
>
> **备注:** Preprint. Planned for resubmission after revision; See supplementary material at https://github.com/adrianodelr/adaptive-koopman-mpc
>
> **摘要:** Approximating nonlinear systems as linear ones is a common workaround to apply control tools tailored for linear systems. This motivates our present work where we developed a data-driven model predictive controller (MPC) based on the Koopman operator framework, allowing the embedding of nonlinear dynamics in a higher dimensional, but linear function space. The controller, termed adaptive Koopman model predictive control (KMPC), uses online closed-loop feedback to learn and incrementally update a linear representation of nonlinear system dynamics, without the prior knowledge of a model. Adaptive KMPC differs from most other Koopman-based control frameworks that aim to identify high-validity-range models in advance and then enter closed-loop control without further model adaptations. To validate the controller, trajectory tracking experiments are conducted with 1R and 2R robots under force disturbances and changing model parameters. We compare the controller to classical linearization MPC and Koopman-based MPC without model updates, denoted static KMPC. The results show that adaptive KMPC can, opposed to static KMPC, generalize over unforeseen force disturbances and can, opposed to linearization MPC, handle varying dynamic parameters, while using a small set of basis functions to approximate the Koopman operator.
>
---
#### [replaced 013] CRESSim-MPM: A Material Point Method Library for Surgical Soft Body Simulation with Cutting and Suturing
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2502.18437v3](http://arxiv.org/pdf/2502.18437v3)**

> **作者:** Yafei Ou; Mahdi Tavakoli
>
> **备注:** 8 pages, 13 figures, accepted for IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS) 2025
>
> **摘要:** A number of recent studies have focused on developing surgical simulation platforms to train machine learning (ML) agents or models with synthetic data for surgical assistance. While existing platforms excel at tasks such as rigid body manipulation and soft body deformation, they struggle to simulate more complex soft body behaviors like cutting and suturing. A key challenge lies in modeling soft body fracture and splitting using the finite-element method (FEM), which is the predominant approach in current platforms. Additionally, the two-way suture needle/thread contact inside a soft body is further complicated when using FEM. In this work, we use the material point method (MPM) for such challenging simulations and propose new rigid geometries and soft-rigid contact methods specifically designed for them. We introduce CRESSim-MPM, a GPU-accelerated MPM library that integrates multiple MPM solvers and incorporates surgical geometries for cutting and suturing, serving as a specialized physics engine for surgical applications. It is further integrated into Unity, requiring minimal modifications to existing projects for soft body simulation. We demonstrate the simulator's capabilities in real-time simulation of cutting and suturing on soft tissue and provide an initial performance evaluation of different MPM solvers when simulating varying numbers of particles. The source code is available at https://github.com/yafei-ou/CRESSim-MPM.
>
---
#### [replaced 014] GRIP: A General Robotic Incremental Potential Contact Simulation Dataset for Unified Deformable-Rigid Coupled Grasping
- **分类: cs.RO; cs.GR**

- **链接: [http://arxiv.org/pdf/2503.05020v2](http://arxiv.org/pdf/2503.05020v2)**

> **作者:** Siyu Ma; Wenxin Du; Chang Yu; Ying Jiang; Zeshun Zong; Tianyi Xie; Yunuo Chen; Yin Yang; Xuchen Han; Chenfanfu Jiang
>
> **备注:** We release GRIP to advance research in robotic manipulation, soft-gripper control, and physics-driven simulation at: https://bell0o.github.io/GRIP/
>
> **摘要:** Grasping is fundamental to robotic manipulation, and recent advances in large-scale grasping datasets have provided essential training data and evaluation benchmarks, accelerating the development of learning-based methods for robust object grasping. However, most existing datasets exclude deformable bodies due to the lack of scalable, robust simulation pipelines, limiting the development of generalizable models for compliant grippers and soft manipulands. To address these challenges, we present GRIP, a General Robotic Incremental Potential contact simulation dataset for universal grasping. GRIP leverages an optimized Incremental Potential Contact (IPC)-based simulator for multi-environment data generation, achieving up to 48x speedup while ensuring efficient, intersection- and inversion-free simulations for compliant grippers and deformable objects. Our fully automated pipeline generates and evaluates diverse grasp interactions across 1,200 objects and 100,000 grasp poses, incorporating both soft and rigid grippers. The GRIP dataset enables applications such as neural grasp generation and stress field prediction.
>
---
#### [replaced 015] Bridging Deep Reinforcement Learning and Motion Planning for Model-Free Navigation in Cluttered Environments
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.07283v3](http://arxiv.org/pdf/2504.07283v3)**

> **作者:** Licheng Luo; Mingyu Cai
>
> **备注:** 16 pages
>
> **摘要:** Deep Reinforcement Learning (DRL) has emerged as a powerful model-free paradigm for learning optimal policies. However, in navigation tasks with cluttered environments, DRL methods often suffer from insufficient exploration, especially under sparse rewards or complex dynamics with system disturbances. To address this challenge, we bridge general graph-based motion planning with DRL, enabling agents to explore cluttered spaces more effectively and achieve desired navigation performance. Specifically, we design a dense reward function grounded in a graph structure that spans the entire state space. This graph provides rich guidance, steering the agent toward optimal strategies. We validate our approach in challenging environments, demonstrating substantial improvements in exploration efficiency and task success rates.
>
---
#### [replaced 016] TriVLA: A Triple-System-Based Unified Vision-Language-Action Model for General Robot Control
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.01424v2](http://arxiv.org/pdf/2507.01424v2)**

> **作者:** Zhenyang Liu; Yongchong Gu; Sixiao Zheng; Xiangyang Xue; Yanwei Fu
>
> **摘要:** Recent advancements in vision-language models (VLMs) for common-sense reasoning have led to the development of vision-language-action (VLA) models, enabling robots to perform generalized manipulation. Although existing autoregressive VLA methods design a specific architecture like dual-system to leverage large-scale pretrained knowledge, they tend to capture static information, often neglecting the dynamic aspects vital for embodied tasks. To this end, we propose TriVLA, a unified Vision-Language-Action model with a triple-system architecture for general robot control. The vision-language module (System 2) interprets the environment through vision and language instructions. The dynamics perception module (System 3) inherently produces visual representations that encompass both current static information and predicted future dynamics, thereby providing valuable guidance for policy learning. TriVLA utilizes pre-trained VLM model and fine-tunes pre-trained video foundation model on robot datasets along with internet human manipulation data. The subsequent policy learning module (System 1) generates fluid motor actions in real time. Experimental evaluation demonstrates that TriVLA operates at approximately 36 Hz and surpasses state-of-the-art imitation learning baselines on standard simulation benchmarks as well as challenging real-world manipulation tasks.
>
---
#### [replaced 017] Large Language Model-Driven Closed-Loop UAV Operation with Semantic Observations
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2507.01930v2](http://arxiv.org/pdf/2507.01930v2)**

> **作者:** Wenhao Wang; Yanyan Li; Long Jiao; Jiawei Yuan
>
> **备注:** 9 pages, 7 figures
>
> **摘要:** Recent advances in large Language Models (LLMs) have revolutionized mobile robots, including unmanned aerial vehicles (UAVs), enabling their intelligent operation within Internet of Things (IoT) ecosystems. However, LLMs still face challenges from logical reasoning and complex decision-making, leading to concerns about the reliability of LLM-driven UAV operations in IoT applications. In this paper, we propose a LLM-driven closed-loop control framework that enables reliable UAV operations powered by effective feedback and refinement using two LLM modules, i.e., a Code Generator and an Evaluator. Our framework transforms numerical state observations from UAV operations into natural language trajectory descriptions to enhance the evaluator LLM's understanding of UAV dynamics for precise feedback generation. Our framework also enables a simulation-based refinement process, and hence eliminates the risks to physical UAVs caused by incorrect code execution during the refinement. Extensive experiments on UAV control tasks with different complexities are conducted. The experimental results show that our framework can achieve reliable UAV operations using LLMs, which significantly outperforms baseline approaches in terms of success rate and completeness with the increase of task complexity.
>
---
#### [replaced 018] Modular Soft Wearable Glove for Real-Time Gesture Recognition and Dynamic 3D Shape Reconstruction
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.05983v2](http://arxiv.org/pdf/2504.05983v2)**

> **作者:** Huazhi Dong; Chunpeng Wang; Mingyuan Jiang; Francesco Giorgio-Serchi; Yunjie Yang
>
> **摘要:** With the increasing demand for human-computer interaction (HCI), flexible wearable gloves have emerged as a promising solution in virtual reality, medical rehabilitation, and industrial automation. However, the current technology still has problems like insufficient sensitivity and limited durability, which hinder its wide application. This paper presents a highly sensitive, modular, and flexible capacitive sensor based on line-shaped electrodes and liquid metal (EGaIn), integrated into a sensor module tailored to the human hand's anatomy. The proposed system independently captures bending information from each finger joint, while additional measurements between adjacent fingers enable the recording of subtle variations in inter-finger spacing. This design enables accurate gesture recognition and dynamic hand morphological reconstruction of complex movements using point clouds. Experimental results demonstrate that our classifier based on Convolution Neural Network (CNN) and Multilayer Perceptron (MLP) achieves an accuracy of 99.15% across 30 gestures. Meanwhile, a transformer-based Deep Neural Network (DNN) accurately reconstructs dynamic hand shapes with an Average Distance (AD) of 2.076\pm3.231 mm, with the reconstruction accuracy at individual key points surpassing SOTA benchmarks by 9.7% to 64.9%. The proposed glove shows excellent accuracy, robustness and scalability in gesture recognition and hand reconstruction, making it a promising solution for next-generation HCI systems.
>
---
#### [replaced 019] LUDO: Low-Latency Understanding of Deformable Objects using Point Cloud Occupancy Functions
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2411.08777v5](http://arxiv.org/pdf/2411.08777v5)**

> **作者:** Pit Henrich; Franziska Mathis-Ullrich; Paul Maria Scheikl
>
> **备注:** Published in IEEE Transactions on Robotics (T-RO)
>
> **摘要:** Accurately determining the shape of deformable objects and the location of their internal structures is crucial for medical tasks that require precise targeting, such as robotic biopsies. We introduce LUDO, a method for accurate low-latency understanding of deformable objects. LUDO reconstructs objects in their deformed state, including their internal structures, from a single-view point cloud observation in under 30 ms using occupancy networks. LUDO provides uncertainty estimates for its predictions. Additionally, it provides explainability by highlighting key features in its input observations. Both uncertainty and explainability are important for safety-critical applications such as surgery. We evaluate LUDO in real-world robotic experiments, achieving a success rate of 98.9% for puncturing various regions of interest (ROIs) inside deformable objects. We compare LUDO to a popular baseline and show its superior ROI localization accuracy, training time, and memory requirements. LUDO demonstrates the potential to interact with deformable objects without the need for deformable registration methods.
>
---
#### [replaced 020] SoccerDiffusion: Toward Learning End-to-End Humanoid Robot Soccer from Gameplay Recordings
- **分类: cs.RO; cs.AI; cs.LG**

- **链接: [http://arxiv.org/pdf/2504.20808v2](http://arxiv.org/pdf/2504.20808v2)**

> **作者:** Florian Vahl; Jörn Griepenburg; Jan Gutsche; Jasper Güldenstein; Jianwei Zhang
>
> **摘要:** This paper introduces SoccerDiffusion, a transformer-based diffusion model designed to learn end-to-end control policies for humanoid robot soccer directly from real-world gameplay recordings. Using data collected from RoboCup competitions, the model predicts joint command trajectories from multi-modal sensor inputs, including vision, proprioception, and game state. We employ a distillation technique to enable real-time inference on embedded platforms that reduces the multi-step diffusion process to a single step. Our results demonstrate the model's ability to replicate complex motion behaviors such as walking, kicking, and fall recovery both in simulation and on physical robots. Although high-level tactical behavior remains limited, this work provides a robust foundation for subsequent reinforcement learning or preference optimization methods. We release the dataset, pretrained models, and code under: https://bit-bots.github.io/SoccerDiffusion
>
---
#### [replaced 021] Benchmarking Generalizable Bimanual Manipulation: RoboTwin Dual-Arm Collaboration Challenge at CVPR 2025 MEIS Workshop
- **分类: cs.RO; cs.AI; cs.LG; cs.MA**

- **链接: [http://arxiv.org/pdf/2506.23351v2](http://arxiv.org/pdf/2506.23351v2)**

> **作者:** Tianxing Chen; Kaixuan Wang; Zhaohui Yang; Yuhao Zhang; Zanxin Chen; Baijun Chen; Wanxi Dong; Ziyuan Liu; Dong Chen; Tianshuo Yang; Haibao Yu; Xiaokang Yang; Yusen Qin; Zhiqiang Xie; Yao Mu; Ping Luo; Tian Nian; Weiliang Deng; Yiheng Ge; Yibin Liu; Zixuan Li; Dehui Wang; Zhixuan Liang; Haohui Xie; Rijie Zeng; Yunfei Ge; Peiqing Cong; Guannan He; Zhaoming Han; Ruocheng Yin; Jingxiang Guo; Lunkai Lin; Tianling Xu; Hongzhe Bi; Xuewu Lin; Tianwei Lin; Shujie Luo; Keyu Li; Ziyan Zhao; Ke Fan; Heyang Xu; Bo Peng; Wenlong Gao; Dongjiang Li; Feng Jin; Hui Shen; Jinming Li; Chaowei Cui; Yu Chen; Yaxin Peng; Lingdong Zeng; Wenlong Dong; Tengfei Li; Weijie Ke; Jun Chen; Erdemt Bao; Tian Lan; Tenglong Liu; Jin Yang; Huiping Zhuang; Baozhi Jia; Shuai Zhang; Zhengfeng Zou; Fangheng Guan; Tianyi Jia; Ke Zhou; Hongjiu Zhang; Yating Han; Cheng Fang; Yixian Zou; Chongyang Xu; Qinglun Zhang; Shen Cheng; Xiaohe Wang; Ping Tan; Haoqiang Fan; Shuaicheng Liu; Jiaheng Chen; Chuxuan Huang; Chengliang Lin; Kaijun Luo; Boyu Yue; Yi Liu; Jinyu Chen; Zichang Tan; Liming Deng; Shuo Xu; Zijian Cai; Shilong Yin; Hao Wang; Hongshan Liu; Tianyang Li; Long Shi; Ran Xu; Huilin Xu; Zhengquan Zhang; Congsheng Xu; Jinchang Yang; Feng Xu
>
> **备注:** Challenge Webpage: https://robotwin-benchmark.github.io/cvpr-2025-challenge/
>
> **摘要:** Embodied Artificial Intelligence (Embodied AI) is an emerging frontier in robotics, driven by the need for autonomous systems that can perceive, reason, and act in complex physical environments. While single-arm systems have shown strong task performance, collaborative dual-arm systems are essential for handling more intricate tasks involving rigid, deformable, and tactile-sensitive objects. To advance this goal, we launched the RoboTwin Dual-Arm Collaboration Challenge at the 2nd MEIS Workshop, CVPR 2025. Built on the RoboTwin Simulation platform (1.0 and 2.0) and the AgileX COBOT-Magic Robot platform, the competition consisted of three stages: Simulation Round 1, Simulation Round 2, and a final Real-World Round. Participants totally tackled 17 dual-arm manipulation tasks, covering rigid, deformable, and tactile-based scenarios. The challenge attracted 64 global teams and over 400 participants, producing top-performing solutions like SEM and AnchorDP3 and generating valuable insights into generalizable bimanual policy learning. This report outlines the competition setup, task design, evaluation methodology, key findings and future direction, aiming to support future research on robust and generalizable bimanual manipulation policies. The Challenge Webpage is available at https://robotwin-benchmark.github.io/cvpr-2025-challenge/.
>
---
