# 机器人 cs.RO

- **最新发布 17 篇**

- **更新 10 篇**

## 最新发布

#### [new 001] Take That for Me: Multimodal Exophora Resolution with Interactive Questioning for Ambiguous Out-of-View Instructions
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2508.16143v1](http://arxiv.org/pdf/2508.16143v1)**

> **作者:** Akira Oyama; Shoichi Hasegawa; Akira Taniguchi; Yoshinobu Hagiwara; Tadahiro Taniguchi
>
> **备注:** See website at https://emergentsystemlabstudent.github.io/MIEL/. Accepted at IEEE RO-MAN 2025
>
> **摘要:** Daily life support robots must interpret ambiguous verbal instructions involving demonstratives such as ``Bring me that cup,'' even when objects or users are out of the robot's view. Existing approaches to exophora resolution primarily rely on visual data and thus fail in real-world scenarios where the object or user is not visible. We propose Multimodal Interactive Exophora resolution with user Localization (MIEL), which is a multimodal exophora resolution framework leveraging sound source localization (SSL), semantic mapping, visual-language models (VLMs), and interactive questioning with GPT-4o. Our approach first constructs a semantic map of the environment and estimates candidate objects from a linguistic query with the user's skeletal data. SSL is utilized to orient the robot toward users who are initially outside its visual field, enabling accurate identification of user gestures and pointing directions. When ambiguities remain, the robot proactively interacts with the user, employing GPT-4o to formulate clarifying questions. Experiments in a real-world environment showed results that were approximately 1.3 times better when the user was visible to the robot and 2.0 times better when the user was not visible to the robot, compared to the methods without SSL and interactive questioning. The project website is https://emergentsystemlabstudent.github.io/MIEL/.
>
---
#### [new 002] GelSLAM: A Real-time, High-Fidelity, and Robust 3D Tactile SLAM System
- **分类: cs.RO; cs.CV**

- **简介: 论文提出GelSLAM，一种纯触觉驱动的实时3D SLAM系统，用于高精度物体位姿估计与形状重建。解决视觉方法在接触场景中易受遮挡的问题，通过触觉表面法向与曲率实现稳定跟踪和闭环，支持长时程、高保真感知。**

- **链接: [http://arxiv.org/pdf/2508.15990v1](http://arxiv.org/pdf/2508.15990v1)**

> **作者:** Hung-Jui Huang; Mohammad Amin Mirzaee; Michael Kaess; Wenzhen Yuan
>
> **备注:** 18 pages
>
> **摘要:** Accurately perceiving an object's pose and shape is essential for precise grasping and manipulation. Compared to common vision-based methods, tactile sensing offers advantages in precision and immunity to occlusion when tracking and reconstructing objects in contact. This makes it particularly valuable for in-hand and other high-precision manipulation tasks. In this work, we present GelSLAM, a real-time 3D SLAM system that relies solely on tactile sensing to estimate object pose over long periods and reconstruct object shapes with high fidelity. Unlike traditional point cloud-based approaches, GelSLAM uses tactile-derived surface normals and curvatures for robust tracking and loop closure. It can track object motion in real time with low error and minimal drift, and reconstruct shapes with submillimeter accuracy, even for low-texture objects such as wooden tools. GelSLAM extends tactile sensing beyond local contact to enable global, long-horizon spatial perception, and we believe it will serve as a foundation for many precise manipulation tasks involving interaction with objects in hand. The video demo is available on our website: https://joehjhuang.github.io/gelslam.
>
---
#### [new 003] Terrain Classification for the Spot Quadrupedal Mobile Robot Using Only Proprioceptive Sensing
- **分类: cs.RO; eess.SP**

- **简介: 该论文属于机器人地形分类任务，旨在解决四足机器人在复杂地形中易陷入或打滑的问题。作者利用波士顿动力Spot机器人的本体感觉信号，通过降维和分类技术，实现了三种地形的高精度识别（约97%），为路径规划提供 traversability 信息。**

- **链接: [http://arxiv.org/pdf/2508.16504v1](http://arxiv.org/pdf/2508.16504v1)**

> **作者:** Sophie Villemure; Jefferson Silveira; Joshua A. Marshall
>
> **摘要:** Quadrupedal mobile robots can traverse a wider range of terrain types than their wheeled counterparts but do not perform the same on all terrain types. These robots are prone to undesirable behaviours like sinking and slipping on challenging terrains. To combat this issue, we propose a terrain classifier that provides information on terrain type that can be used in robotic systems to create a traversability map to plan safer paths for the robot to navigate. The work presented here is a terrain classifier developed for a Boston Dynamics Spot robot. Spot provides over 100 measured proprioceptive signals describing the motions of the robot and its four legs (e.g., foot penetration, forces, joint angles, etc.). The developed terrain classifier combines dimensionality reduction techniques to extract relevant information from the signals and then applies a classification technique to differentiate terrain based on traversability. In representative field testing, the resulting terrain classifier was able to identify three different terrain types with an accuracy of approximately 97%
>
---
#### [new 004] GPL-SLAM: A Laser SLAM Framework with Gaussian Process Based Extended Landmarks
- **分类: cs.RO**

- **简介: 论文提出GPL-SLAM，一种基于高斯过程的激光SLAM框架，通过对象级轮廓表示实现高效、语义丰富的环境建模与定位，解决传统方法在结构化环境中精度与信息表达不足的问题。**

- **链接: [http://arxiv.org/pdf/2508.16459v1](http://arxiv.org/pdf/2508.16459v1)**

> **作者:** Ali Emre Balcı; Erhan Ege Keyvan; Emre Özkan
>
> **备注:** Authors Ali Emre Balc{\i} and Erhan Ege Keyvan contributed equally to this work
>
> **摘要:** We present a novel Simultaneous Localization and Mapping (SLAM) method that employs Gaussian Process (GP) based landmark (object) representations. Instead of conventional grid maps or point cloud registration, we model the environment on a per object basis using GP based contour representations. These contours are updated online through a recursive scheme, enabling efficient memory usage. The SLAM problem is formulated within a fully Bayesian framework, allowing joint inference over the robot pose and object based map. This representation provides semantic information such as the number of objects and their areas, while also supporting probabilistic measurement to object associations. Furthermore, the GP based contours yield confidence bounds on object shapes, offering valuable information for downstream tasks like safe navigation and exploration. We validate our method on synthetic and real world experiments, and show that it delivers accurate localization and mapping performance across diverse structured environments.
>
---
#### [new 005] Hierarchical Decision-Making for Autonomous Navigation: Integrating Deep Reinforcement Learning and Fuzzy Logic in Four-Wheel Independent Steering and Driving Systems
- **分类: cs.RO; cs.AI**

- **简介: 论文提出一种分层决策框架，用于四轮独立转向与驱动系统自主导航。结合深度强化学习与模糊逻辑，解决传统方法在复杂环境中稳定性差、物理可行性不足的问题，提升导航性能与安全性。**

- **链接: [http://arxiv.org/pdf/2508.16574v1](http://arxiv.org/pdf/2508.16574v1)**

> **作者:** Yizhi Wang; Degang Xu; Yongfang Xie; Shuzhong Tan; Xianan Zhou; Peng Chen
>
> **摘要:** This paper presents a hierarchical decision-making framework for autonomous navigation in four-wheel independent steering and driving (4WISD) systems. The proposed approach integrates deep reinforcement learning (DRL) for high-level navigation with fuzzy logic for low-level control to ensure both task performance and physical feasibility. The DRL agent generates global motion commands, while the fuzzy logic controller enforces kinematic constraints to prevent mechanical strain and wheel slippage. Simulation experiments demonstrate that the proposed framework outperforms traditional navigation methods, offering enhanced training efficiency and stability and mitigating erratic behaviors compared to purely DRL-based solutions. Real-world validations further confirm the framework's ability to navigate safely and effectively in dynamic industrial settings. Overall, this work provides a scalable and reliable solution for deploying 4WISD mobile robots in complex, real-world scenarios.
>
---
#### [new 006] On Kinodynamic Global Planning in a Simplicial Complex Environment: A Mixed Integer Approach
- **分类: cs.RO; math.OC**

- **简介: 论文研究车类车辆在单纯形复杂环境中的最优运动规划问题，旨在计算满足速度、加速度和转向约束的最短时间轨迹。通过将问题转化为混合整数线性规划求解，显著提升效率并避免局部最优。**

- **链接: [http://arxiv.org/pdf/2508.16511v1](http://arxiv.org/pdf/2508.16511v1)**

> **作者:** Otobong Jerome; Alexandr Klimchik; Alexander Maloletov; Geesara Kulathunga
>
> **摘要:** This work casts the kinodynamic planning problem for car-like vehicles as an optimization task to compute a minimum-time trajectory and its associated velocity profile, subject to boundary conditions on velocity, acceleration, and steering. The approach simultaneously optimizes both the spatial path and the sequence of acceleration and steering controls, ensuring continuous motion from a specified initial position and velocity to a target end position and velocity.The method analyzes the admissible control space and terrain to avoid local minima. The proposed method operates efficiently in simplicial complex environments, a preferred terrain representation for capturing intricate 3D landscapes. The problem is initially posed as a mixed-integer fractional program with quadratic constraints, which is then reformulated into a mixed-integer bilinear objective through a variable transformation and subsequently relaxed to a mixed-integer linear program using McCormick envelopes. Comparative simulations against planners such as MPPI and log-MPPI demonstrate that the proposed approach generates solutions 104 times faster while strictly adhering to the specified constraints
>
---
#### [new 007] Swarming Without an Anchor (SWA): Robot Swarms Adapt Better to Localization Dropouts Then a Single Robot
- **分类: cs.RO; cs.MA**

- **简介: 论文提出SWA方法，解决多无人机在局部定位失效时的状态估计问题。通过融合分布式估计与相对感知，实现无需全局定位的稳定编队，抑制个体扰动，仅允许整体平移漂移，提升系统鲁棒性。**

- **链接: [http://arxiv.org/pdf/2508.16460v1](http://arxiv.org/pdf/2508.16460v1)**

> **作者:** Jiri Horyna; Roland Jung; Stephan Weiss; Eliseo Ferrante; Martin Saska
>
> **备注:** Accepted to IEEE RA-L on April 1, 2025
>
> **摘要:** In this paper, we present the Swarming Without an Anchor (SWA) approach to state estimation in swarms of Unmanned Aerial Vehicles (UAVs) experiencing ego-localization dropout, where individual agents are laterally stabilized using relative information only. We propose to fuse decentralized state estimation with robust mutual perception and onboard sensor data to maintain accurate state awareness despite intermittent localization failures. Thus, the relative information used to estimate the lateral state of UAVs enables the identification of the unambiguous state of UAVs with respect to the local constellation. The resulting behavior reaches velocity consensus, as this task can be referred to as the double integrator synchronization problem. All disturbances and performance degradations except a uniform translation drift of the swarm as a whole is attenuated which is enabling new opportunities in using tight cooperation for increasing reliability and resilience of multi-UAV systems. Simulations and real-world experiments validate the effectiveness of our approach, demonstrating its capability to sustain cohesive swarm behavior in challenging conditions of unreliable or unavailable primary localization.
>
---
#### [new 008] UnPose: Uncertainty-Guided Diffusion Priors for Zero-Shot Pose Estimation
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出UnPose框架，用于零样本6D物体姿态估计与重建任务。针对缺乏CAD模型时的挑战，利用预训练扩散模型的3D先验和不确定性估计，通过多视角融合与姿态图优化，实现高精度姿态估计和高质量3D重建。**

- **链接: [http://arxiv.org/pdf/2508.15972v1](http://arxiv.org/pdf/2508.15972v1)**

> **作者:** Zhaodong Jiang; Ashish Sinha; Tongtong Cao; Yuan Ren; Bingbing Liu; Binbin Xu
>
> **备注:** Published at the Conference on Robot Learning (CoRL) 2025. For more details please visit https://frankzhaodong.github.io/UnPose
>
> **摘要:** Estimating the 6D pose of novel objects is a fundamental yet challenging problem in robotics, often relying on access to object CAD models. However, acquiring such models can be costly and impractical. Recent approaches aim to bypass this requirement by leveraging strong priors from foundation models to reconstruct objects from single or multi-view images, but typically require additional training or produce hallucinated geometry. To this end, we propose UnPose, a novel framework for zero-shot, model-free 6D object pose estimation and reconstruction that exploits 3D priors and uncertainty estimates from a pre-trained diffusion model. Specifically, starting from a single-view RGB-D frame, UnPose uses a multi-view diffusion model to estimate an initial 3D model using 3D Gaussian Splatting (3DGS) representation, along with pixel-wise epistemic uncertainty estimates. As additional observations become available, we incrementally refine the 3DGS model by fusing new views guided by the diffusion model's uncertainty, thereby continuously improving the pose estimation accuracy and 3D reconstruction quality. To ensure global consistency, the diffusion prior-generated views and subsequent observations are further integrated in a pose graph and jointly optimized into a coherent 3DGS field. Extensive experiments demonstrate that UnPose significantly outperforms existing approaches in both 6D pose estimation accuracy and 3D reconstruction quality. We further showcase its practical applicability in real-world robotic manipulation tasks.
>
---
#### [new 009] Spatial Policy: Guiding Visuomotor Robotic Manipulation with Spatial-Aware Modeling and Reasoning
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文提出Spatial Policy（SP），解决机器人操纵中视觉计划与动作执行间缺乏空间感知的问题。通过显式空间建模、动作预测与推理反馈机制，提升复杂环境下的控制精度与成功率。**

- **链接: [http://arxiv.org/pdf/2508.15874v1](http://arxiv.org/pdf/2508.15874v1)**

> **作者:** Yijun Liu; Yuwei Liu; Yuan Meng; Jieheng Zhang; Yuwei Zhou; Ye Li; Jiacheng Jiang; Kangye Ji; Shijia Ge; Zhi Wang; Wenwu Zhu
>
> **摘要:** Vision-centric hierarchical embodied models have demonstrated strong potential for long-horizon robotic control. However, existing methods lack spatial awareness capabilities, limiting their effectiveness in bridging visual plans to actionable control in complex environments. To address this problem, we propose Spatial Policy (SP), a unified spatial-aware visuomotor robotic manipulation framework via explicit spatial modeling and reasoning. Specifically, we first design a spatial-conditioned embodied video generation module to model spatially guided predictions through a spatial plan table. Then, we propose a spatial-based action prediction module to infer executable actions with coordination. Finally, we propose a spatial reasoning feedback policy to refine the spatial plan table via dual-stage replanning. Extensive experiments show that SP significantly outperforms state-of-the-art baselines, achieving a 33.0% average improvement over the best baseline. With an 86.7% average success rate across 11 diverse tasks, SP substantially enhances the practicality of embodied models for robotic control applications. Code and checkpoints are maintained at https://plantpotatoonmoon.github.io/SpatialPolicy/.
>
---
#### [new 010] Self-Aligning EPM Connector: A Versatile Solution for Adaptive and Multi-Modal Interfaces
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.16008v1](http://arxiv.org/pdf/2508.16008v1)**

> **作者:** Bingchao Wang; Adam A. Stokes
>
> **摘要:** This paper presents a multifunctional connector based on electro-permanent magnet (EPM) technology, integrating self-alignment, mechanical coupling, fluid transfer, and data communication within a compact SLA-3D printed structure. Experimental results demonstrate reliable self-alignment, efficient fluid transfer in single-loop and dual-channel modes, and robust data transmission via integrated electronic control. The connector exhibits high flexibility in accommodating axial, angular, and lateral misalignments while maintaining low energy consumption. These features make it highly suitable for modular robotics, electric vehicle charging, household robotic platforms, and aerospace docking applications.
>
---
#### [new 011] Comparative Analysis of UAV Path Planning Algorithms for Efficient Navigation in Urban 3D Environments
- **分类: cs.RO; cs.AI**

- **简介: 论文研究无人机在复杂城市3D环境中路径规划的问题，比较A*、RRT*和PSO三种算法的性能。通过多场景实验发现，A*效率与路径质量最优，PSO适合密集环境，RRT*表现均衡。**

- **链接: [http://arxiv.org/pdf/2508.16515v1](http://arxiv.org/pdf/2508.16515v1)**

> **作者:** Hichem Cheriet; Khellat Kihel Badra; Chouraqui Samira
>
> **摘要:** The most crucial challenges for UAVs are planning paths and avoiding obstacles in their way. In recent years, a wide variety of path-planning algorithms have been developed. These algorithms have successfully solved path-planning problems; however, they suffer from multiple challenges and limitations. To test the effectiveness and efficiency of three widely used algorithms, namely A*, RRT*, and Particle Swarm Optimization (PSO), this paper conducts extensive experiments in 3D urban city environments cluttered with obstacles. Three experiments were designed with two scenarios each to test the aforementioned algorithms. These experiments consider different city map sizes, different altitudes, and varying obstacle densities and sizes in the environment. According to the experimental results, the A* algorithm outperforms the others in both computation efficiency and path quality. PSO is especially suitable for tight turns and dense environments, and RRT* offers a balance and works well across all experiments due to its randomized approach to finding solutions.
>
---
#### [new 012] NeuralMeshing: Complete Object Mesh Extraction from Casual Captures
- **分类: cs.CV; cs.RO**

- **简介: 论文提出NeuralMeshing系统，用于从多视角视频自动重建日常物体的完整网格模型。解决无专业扫描设备时获取高质量3D模型的问题，通过已知点定位与结构光技术融合多视频数据生成完整几何模型。**

- **链接: [http://arxiv.org/pdf/2508.16026v1](http://arxiv.org/pdf/2508.16026v1)**

> **作者:** Floris Erich; Naoya Chiba; Abdullah Mustafa; Ryo Hanai; Noriaki Ando; Yusuke Yoshiyasu; Yukiyasu Domae
>
> **摘要:** How can we extract complete geometric models of objects that we encounter in our daily life, without having access to commercial 3D scanners? In this paper we present an automated system for generating geometric models of objects from two or more videos. Our system requires the specification of one known point in at least one frame of each video, which can be automatically determined using a fiducial marker such as a checkerboard or Augmented Reality (AR) marker. The remaining frames are automatically positioned in world space by using Structure-from-Motion techniques. By using multiple videos and merging results, a complete object mesh can be generated, without having to rely on hole filling. Code for our system is available from https://github.com/FlorisE/NeuralMeshing.
>
---
#### [new 013] Validating Terrain Models in Digital Twins for Trustworthy sUAS Operations
- **分类: cs.SE; cs.RO**

- **简介: 论文针对无人机在复杂环境中依赖数字孪生地形模型的可信性问题，提出三维验证流程，结合真实场景测试与仿真，解决模型精度与实际应用间的不确定性挑战。**

- **链接: [http://arxiv.org/pdf/2508.16104v1](http://arxiv.org/pdf/2508.16104v1)**

> **作者:** Arturo Miguel Russell Bernal; Maureen Petterson; Pedro Antonio Alarcon Granadeno; Michael Murphy; James Mason; Jane Cleland-Huang
>
> **备注:** Submitted to EDTconf 2025
>
> **摘要:** With the increasing deployment of small Unmanned Aircraft Systems (sUAS) in unfamiliar and complex environments, Environmental Digital Twins (EDT) that comprise weather, airspace, and terrain data are critical for safe flight planning and for maintaining appropriate altitudes during search and surveillance operations. With the expansion of sUAS capabilities through edge and cloud computing, accurate EDT are also vital for advanced sUAS capabilities, like geolocation. However, real-world sUAS deployment introduces significant sources of uncertainty, necessitating a robust validation process for EDT components. This paper focuses on the validation of terrain models, one of the key components of an EDT, for real-world sUAS tasks. These models are constructed by fusing U.S. Geological Survey (USGS) datasets and satellite imagery, incorporating high-resolution environmental data to support mission tasks. Validating both the terrain models and their operational use by sUAS under real-world conditions presents significant challenges, including limited data granularity, terrain discontinuities, GPS and sensor inaccuracies, visual detection uncertainties, as well as onboard resources and timing constraints. We propose a 3-Dimensions validation process grounded in software engineering principles, following a workflow across granularity of tests, simulation to real world, and the analysis of simple to edge conditions. We demonstrate our approach using a multi-sUAS platform equipped with a Terrain-Aware Digital Shadow.
>
---
#### [new 014] Active Prostate Phantom with Multiple Chambers
- **分类: physics.med-ph; cs.RO**

- **简介: 该论文属于医疗仿真领域，旨在解决现有前列腺phantom无法模拟动态病变的问题。作者设计了一种气动驱动的多腔体phantom，可精确模拟对称与非对称良性前列腺增生，通过MRI数据建模与FEM仿真验证，误差分别低至3.47%和1.41%，可用于机器人手术系统验证与训练。**

- **链接: [http://arxiv.org/pdf/2508.15873v1](http://arxiv.org/pdf/2508.15873v1)**

> **作者:** Sizhe Tian; Yinoussa Adagolodjo; Jeremie Dequidt
>
> **摘要:** Prostate cancer is a major global health concern, requiring advancements in robotic surgery and diagnostics to improve patient outcomes. A phantom is a specially designed object that simulates human tissues or organs. It can be used for calibrating and testing a medical process, as well as for training and research purposes. Existing prostate phantoms fail to simulate dynamic scenarios. This paper presents a pneumatically actuated prostate phantom with multiple independently controlled chambers, allowing for precise volumetric adjustments to replicate asymmetric and symmetric benign prostatic hyperplasia (BPH). The phantom is designed based on shape analysis of magnetic resonance imaging (MRI) datasets, modeled with finite element method (FEM), and validated through 3D reconstruction. The simulation results showed strong agreement with physical measurements, achieving average errors of 3.47% in forward modeling and 1.41% in inverse modeling. These results demonstrate the phantom's potential as a platform for validating robotic-assisted systems and for further development toward realistic simulation-based medical training.
>
---
#### [new 015] HOSt3R: Keypoint-free Hand-Object 3D Reconstruction from RGB images
- **分类: cs.CV; cs.AI; cs.HC; cs.LG; cs.RO**

- **简介: 论文提出HOSt3R，解决RGB图像下无关键点的手-物3D重建问题。通过无需关键点检测的单目视频运动估计与多视角重建结合，实现无需预扫描模板或相机参数的通用手-物3D形状恢复，性能优于现有方法。**

- **链接: [http://arxiv.org/pdf/2508.16465v1](http://arxiv.org/pdf/2508.16465v1)**

> **作者:** Anilkumar Swamy; Vincent Leroy; Philippe Weinzaepfel; Jean-Sébastien Franco; Grégory Rogez
>
> **备注:** 12 pages, 8 figures
>
> **摘要:** Hand-object 3D reconstruction has become increasingly important for applications in human-robot interaction and immersive AR/VR experiences. A common approach for object-agnostic hand-object reconstruction from RGB sequences involves a two-stage pipeline: hand-object 3D tracking followed by multi-view 3D reconstruction. However, existing methods rely on keypoint detection techniques, such as Structure from Motion (SfM) and hand-keypoint optimization, which struggle with diverse object geometries, weak textures, and mutual hand-object occlusions, limiting scalability and generalization. As a key enabler to generic and seamless, non-intrusive applicability, we propose in this work a robust, keypoint detector-free approach to estimating hand-object 3D transformations from monocular motion video/images. We further integrate this with a multi-view reconstruction pipeline to accurately recover hand-object 3D shape. Our method, named HOSt3R, is unconstrained, does not rely on pre-scanned object templates or camera intrinsics, and reaches state-of-the-art performance for the tasks of object-agnostic hand-object 3D transformation and shape estimation on the SHOWMe benchmark. We also experiment on sequences from the HO3D dataset, demonstrating generalization to unseen object categories.
>
---
#### [new 016] Sound and Solution-Complete CCBS
- **分类: cs.MA; cs.DM; cs.RO**

- **链接: [http://arxiv.org/pdf/2508.16410v1](http://arxiv.org/pdf/2508.16410v1)**

> **作者:** Alvin Combrink; Sabino Francesco Roselli; Martin Fabian
>
> **备注:** 15 pages
>
> **摘要:** Continuous-time Conflict Based-Search (CCBS) has long been viewed as the de-facto optimal solver for multi-agent path finding in continuous time (MAPFR). Recent findings, however, show that the original theoretical variant of CCBS can suffer from non-termination, while the widely used implementation can return sub-optimal solutions. We introduce an analytical framework that yields simple and sufficient conditions under which any CCBS-style algorithm is both sound, i.e., returns only optimal solutions, and solution complete, i.e., terminates on every solvable MAPFR instance. Investigating the publicly available implementation of CCBS reveals that it violates these conditions. Though this merely indicates that CCBS might be unsound, this indication is supported by counter-examples. Leveraging the analytical framework, we propose a novel branching rule and prove that it satisfies the sufficient conditions, thereby restoring soundness and termination guarantees. Consequently, the resulting CCBS variant is both sound and solution complete, matching the guarantees of the discrete-time CBS for the first time in the continuous domain. We experimentally apply standard CCBS and CCBS under our branching rule to an example problem, with our branching rule returning a solution with lower sum-of-costs than standard CCBS. Because the branching rule largely only affects the branching step, it can be adopted as a drop-in replacement in existing code-bases, as we show in our provided implementation. Beyond CCBS, the analytical framework and termination criterion provide a systematic way to evaluate other CCBS-like MAPFR solvers and future extensions.
>
---
#### [new 017] Do What? Teaching Vision-Language-Action Models to Reject the Impossible
- **分类: cs.AI; cs.RO**

- **简介: 该论文研究视觉-语言-动作模型在面对不可能指令时的处理能力，提出IVA框架，通过检测错误前提、语言澄清和提供可行替代方案，显著提升模型对不合理请求的识别与响应能力。**

- **链接: [http://arxiv.org/pdf/2508.16292v1](http://arxiv.org/pdf/2508.16292v1)**

> **作者:** Wen-Han Hsieh; Elvis Hsieh; Dantong Niu; Trevor Darrell; Roei Herzig; David M. Chan
>
> **备注:** 9 pages, 2 figures, 1 table
>
> **摘要:** Recently, Vision-Language-Action (VLA) models have demonstrated strong performance on a range of robotic tasks. These models rely on multimodal inputs, with language instructions playing a crucial role -- not only in predicting actions, but also in robustly interpreting user intent, even when the requests are impossible to fulfill. In this work, we investigate how VLAs can recognize, interpret, and respond to false-premise instructions: natural language commands that reference objects or conditions absent from the environment. We propose Instruct-Verify-and-Act (IVA), a unified framework that (i) detects when an instruction cannot be executed due to a false premise, (ii) engages in language-based clarification or correction, and (iii) grounds plausible alternatives in perception and action. Towards this end, we construct a large-scale instruction tuning setup with structured language prompts and train a VLA model capable of handling both accurate and erroneous requests. Our approach leverages a contextually augmented, semi-synthetic dataset containing paired positive and false-premise instructions, enabling robust detection and natural language correction. Our experiments show that IVA improves false premise detection accuracy by 97.56% over baselines, while increasing successful responses in false-premise scenarios by 50.78%.
>
---
## 更新

#### [replaced 001] B*: Efficient and Optimal Base Placement for Fixed-Base Manipulators
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.12719v2](http://arxiv.org/pdf/2504.12719v2)**

> **作者:** Zihang Zhao; Leiyao Cui; Sirui Xie; Saiyao Zhang; Zhi Han; Lecheng Ruan; Yixin Zhu
>
> **备注:** accepted for publication in the IEEE Robotics and Automation Letters (RA-L)
>
> **摘要:** B* is a novel optimization framework that addresses a critical challenge in fixed-base manipulator robotics: optimal base placement. Current methods rely on pre-computed kinematics databases generated through sampling to search for solutions. However, they face an inherent trade-off between solution optimality and computational efficiency when determining sampling resolution. To address these limitations, B* unifies multiple objectives without database dependence. The framework employs a two-layer hierarchical approach. The outer layer systematically manages terminal constraints through progressive tightening, particularly for base mobility, enabling feasible initialization and broad solution exploration. The inner layer addresses non-convexities in each outer-layer subproblem through sequential local linearization, converting the original problem into tractable sequential linear programming (SLP). Testing across multiple robot platforms demonstrates B*'s effectiveness. The framework achieves solution optimality five orders of magnitude better than sampling-based approaches while maintaining perfect success rates and reduced computational overhead. Operating directly in configuration space, B* enables simultaneous path planning with customizable optimization criteria. B* serves as a crucial initialization tool that bridges the gap between theoretical motion planning and practical deployment, where feasible trajectory existence is fundamental.
>
---
#### [replaced 002] Hyper Yoshimura: How a slight tweak on a classical folding pattern unleashes meta-stability for deployable robots
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2505.09919v2](http://arxiv.org/pdf/2505.09919v2)**

> **作者:** Ziyang Zhou; Yogesh Phalak; Vishrut Deshpande; Ethan O'Brien; Ian Walker; Suyi Li
>
> **摘要:** Deployable structures inspired by origami have provided lightweight, compact, and reconfigurable solutions for various robotic and architectural applications. However, creating an integrated structural system that can effectively balance the competing requirements of high packing efficiency, simple deployment, and precise morphing into multiple load-bearing configurations remains a significant challenge. This study introduces a new class of hyper-Yoshimura origami, which exhibits a wide range of kinematically admissible and locally metastable states, including newly discovered symmetric "self-packing" and asymmetric "pop-out" states. This metastability is achieved by breaking a design rule of Yoshimura origami that has been in place for many decades. To this end, this study derives a new set of mathematically rigorous design rules and geometric formulations. Based on this, forward and inverse kinematic strategies are developed to stack hyper-Yoshimura modules into deployable booms that can approximate complex 3D shapes. Finally, this study showcases the potential of hyper-Yoshimura with a meter-scale pop-up cellphone charging station deployed at our university's bus transit station, along with a 3D-printed, scaled prototype of a space crane that can function as an object manipulator, solar tracking device, or high-load-bearing structure. These results establish hyper-Yoshimura as a promising platform for deployable and adaptable robotic systems in both terrestrial and space environments.
>
---
#### [replaced 003] ScrewSplat: An End-to-End Method for Articulated Object Recognition
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.02146v2](http://arxiv.org/pdf/2508.02146v2)**

> **作者:** Seungyeon Kim; Junsu Ha; Young Hun Kim; Yonghyeon Lee; Frank C. Park
>
> **备注:** 26 pages, 12 figures, Conference on Robot Learning (CoRL) 2025
>
> **摘要:** Articulated object recognition -- the task of identifying both the geometry and kinematic joints of objects with movable parts -- is essential for enabling robots to interact with everyday objects such as doors and laptops. However, existing approaches often rely on strong assumptions, such as a known number of articulated parts; require additional inputs, such as depth images; or involve complex intermediate steps that can introduce potential errors -- limiting their practicality in real-world settings. In this paper, we introduce ScrewSplat, a simple end-to-end method that operates solely on RGB observations. Our approach begins by randomly initializing screw axes, which are then iteratively optimized to recover the object's underlying kinematic structure. By integrating with Gaussian Splatting, we simultaneously reconstruct the 3D geometry and segment the object into rigid, movable parts. We demonstrate that our method achieves state-of-the-art recognition accuracy across a diverse set of articulated objects, and further enables zero-shot, text-guided manipulation using the recovered kinematic model. See the project website at: https://screwsplat.github.io.
>
---
#### [replaced 004] ROS-related Robotic Systems Development with V-model-based Application of MeROS Metamodel
- **分类: cs.RO; cs.SE**

- **链接: [http://arxiv.org/pdf/2506.08706v2](http://arxiv.org/pdf/2506.08706v2)**

> **作者:** Tomasz Winiarski; Jan Kaniuka; Daniel Giełdowski; Jakub Ostrysz; Krystian Radlak; Dmytro Kushnir
>
> **备注:** 22 pages
>
> **摘要:** Systems built on the Robot Operating System (ROS) are increasingly easy to assemble, yet hard to govern and reliably coordinate. Beyond the sheer number of subsystems involved, the difficulty stems from their diversity and interaction depth. In this paper, we use a compact heterogeneous robotic system (HeROS), combining mobile and manipulation capabilities, as a demonstration vehicle under dynamically changing tasks. Notably, all its subsystems are powered by ROS. The use of compatible interfaces and other ROS integration capabilities simplifies the construction of such systems. However, this only addresses part of the complexity: the semantic coherence and structural traceability are even more important for precise coordination and call for deliberate engineering methods. The Model-Based Systems Engineering (MBSE) discipline, which emerged from the experience of complexity management in large-scale engineering domains, offers the methodological foundations needed. Despite their strengths in complementary aspects of robotics systems engineering, the lack of a unified approach to integrate ROS and MBSE hinders the full potential of these tools. Motivated by the anticipated impact of such a synergy in robotics practice, we propose a structured methodology based on MeROS - a SysML metamodel created specifically to put the ROS-based systems into the focus of the MBSE workflow. As its methodological backbone, we adapt the well-known V-model to this context, illustrating how complex robotic systems can be designed with traceability and validation capabilities embedded into their lifecycle using practices familiar to engineering teams.
>
---
#### [replaced 005] Adaptive Task Space Non-Singular Terminal Super-Twisting Sliding Mode Control of a 7-DOF Robotic Manipulator
- **分类: eess.SY; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2504.13056v2](http://arxiv.org/pdf/2504.13056v2)**

> **作者:** L. Wan; S. Smith; Y. -J. Pan; E. Witrant
>
> **备注:** Accepted for publication in IEEE Transactions on Industrial Electronics. 12 pages, 8 figures
>
> **摘要:** This paper presents a new task-space Non-singular Terminal Super-Twisting Sliding Mode (NT-STSM) controller with adaptive gains for robust trajectory tracking of a 7-DOF robotic manipulator. The proposed approach addresses the challenges of chattering, unknown disturbances, and rotational motion tracking, making it suited for high-DOF manipulators in dexterous manipulation tasks. A rigorous boundedness proof is provided, offering gain selection guidelines for practical implementation. Simulations and hardware experiments with external disturbances demonstrate the proposed controller's robust, accurate tracking with reduced control effort under unknown disturbances compared to other NT-STSM and conventional controllers. The results demonstrated that the proposed NT-STSM controller mitigates chattering and instability in complex motions, making it a viable solution for dexterous robotic manipulations and various industrial applications.
>
---
#### [replaced 006] UAV-ON: A Benchmark for Open-World Object Goal Navigation with Aerial Agents
- **分类: cs.RO; cs.CV**

- **链接: [http://arxiv.org/pdf/2508.00288v4](http://arxiv.org/pdf/2508.00288v4)**

> **作者:** Jianqiang Xiao; Yuexuan Sun; Yixin Shao; Boxi Gan; Rongqiang Liu; Yanjing Wu; Weili Guan; Xiang Deng
>
> **备注:** Accepted to ACM MM Dataset Track 2025
>
> **摘要:** Aerial navigation is a fundamental yet underexplored capability in embodied intelligence, enabling agents to operate in large-scale, unstructured environments where traditional navigation paradigms fall short. However, most existing research follows the Vision-and-Language Navigation (VLN) paradigm, which heavily depends on sequential linguistic instructions, limiting its scalability and autonomy. To address this gap, we introduce UAV-ON, a benchmark for large-scale Object Goal Navigation (ObjectNav) by aerial agents in open-world environments, where agents operate based on high-level semantic goals without relying on detailed instructional guidance as in VLN. UAV-ON comprises 14 high-fidelity Unreal Engine environments with diverse semantic regions and complex spatial layouts, covering urban, natural, and mixed-use settings. It defines 1270 annotated target objects, each characterized by an instance-level instruction that encodes category, physical footprint, and visual descriptors, allowing grounded reasoning. These instructions serve as semantic goals, introducing realistic ambiguity and complex reasoning challenges for aerial agents. To evaluate the benchmark, we implement several baseline methods, including Aerial ObjectNav Agent (AOA), a modular policy that integrates instruction semantics with egocentric observations for long-horizon, goal-directed exploration. Empirical results show that all baselines struggle in this setting, highlighting the compounded challenges of aerial navigation and semantic goal grounding. UAV-ON aims to advance research on scalable UAV autonomy driven by semantic goal descriptions in complex real-world environments.
>
---
#### [replaced 007] Optimized Lattice-Structured Flexible EIT Sensor for Tactile Reconstruction and Classification
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.00161v2](http://arxiv.org/pdf/2505.00161v2)**

> **作者:** Huazhi Dong; Sihao Teng; Xu Han; Xiaopeng Wu; Francesco Giorgio-Serchi; Yunjie Yang
>
> **备注:** Accepted by IEEE Transactions on Instrumentation & Measurement
>
> **摘要:** Flexible electrical impedance tomography (EIT) offers a promising alternative to traditional tactile sensing approaches, enabling low-cost, scalable, and deformable sensor designs. Here, we propose an optimized lattice-structured flexible EIT tactile sensor incorporating a hydrogel-based conductive layer, systematically designed through three-dimensional coupling field simulations to optimize structural parameters for enhanced sensitivity and robustness. By tuning the lattice channel width and conductive layer thickness, we achieve significant improvements in tactile reconstruction quality and classification performance. Experimental results demonstrate high-quality tactile reconstruction with correlation coefficients up to 0.9275, peak signal-to-noise ratios reaching 29.0303 dB, and structural similarity indexes up to 0.9660, while maintaining low relative errors down to 0.3798. Furthermore, the optimized sensor accurately classifies 12 distinct tactile stimuli with an accuracy reaching 99.6%. These results highlight the potential of simulation-guided structural optimization for advancing flexible EIT-based tactile sensors toward practical applications in wearable systems, robotics, and human-machine interfaces.
>
---
#### [replaced 008] TAGA: A Tangent-Based Reactive Approach for Socially Compliant Robot Navigation Around Human Groups
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.21168v2](http://arxiv.org/pdf/2503.21168v2)**

> **作者:** Utsha Kumar Roy; Sejuti Rahman
>
> **备注:** 6 pages, 3 figures. Preprint; intended for submission to IEEE International Conference on Robotics & Automation (ICRA), 2025
>
> **摘要:** Robot navigation in densely populated environments presents significant challenges, particularly regarding the interplay between individual and group dynamics. Current navigation models predominantly address interactions with individual pedestrians while failing to account for human groups that naturally form in real-world settings. Conversely, the limited models implementing group-aware navigation typically prioritize group dynamics at the expense of individual interactions, both of which are essential for socially appropriate navigation. This research extends an existing simulation framework to incorporate both individual pedestrians and human groups. We present Tangent Action for Group Avoidance (TAGA), a modular reactive mechanism that can be integrated with existing navigation frameworks to enhance their group-awareness capabilities. TAGA dynamically modifies robot trajectories using tangent action-based avoidance strategies while preserving the underlying model's capacity to navigate around individuals. Additionally, we introduce Group Collision Rate (GCR), a novel metric to quantitatively assess how effectively robots maintain group integrity during navigation. Through comprehensive simulation-based benchmarking, we demonstrate that integrating TAGA with state-of-the-art navigation models (ORCA, Social Force, DS-RNN, and AG-RL) reduces group intrusions by 45.7-78.6% while maintaining comparable success rates and navigation efficiency. Future work will focus on real-world implementation and validation of this approach.
>
---
#### [replaced 009] OmniVTLA: Vision-Tactile-Language-Action Model with Semantic-Aligned Tactile Sensing
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2508.08706v2](http://arxiv.org/pdf/2508.08706v2)**

> **作者:** Zhengxue Cheng; Yiqian Zhang; Wenkang Zhang; Haoyu Li; Keyu Wang; Li Song; Hengdi Zhang
>
> **备注:** 15 pages, 7 figures, 8 tables. ObjTac dataset: https://readerek.github.io/Objtac.github.io
>
> **摘要:** Recent vision-language-action (VLA) models build upon vision-language foundations, and have achieved promising results and exhibit the possibility of task generalization in robot manipulation. However, due to the heterogeneity of tactile sensors and the difficulty of acquiring tactile data, current VLA models significantly overlook the importance of tactile perception and fail in contact-rich tasks. To address this issue, this paper proposes OmniVTLA, a novel architecture involving tactile sensing. Specifically, our contributions are threefold. First, our OmniVTLA features a dual-path tactile encoder framework. This framework enhances tactile perception across diverse vision-based and force-based tactile sensors by using a pretrained vision transformer (ViT) and a semantically-aligned tactile ViT (SA-ViT). Second, we introduce ObjTac, a comprehensive force-based tactile dataset capturing textual, visual, and tactile information for 56 objects across 10 categories. With 135K tri-modal samples, ObjTac supplements existing visuo-tactile datasets. Third, leveraging this dataset, we train a semantically-aligned tactile encoder to learn a unified tactile representation, serving as a better initialization for OmniVTLA. Real-world experiments demonstrate substantial improvements over state-of-the-art VLA baselines, achieving 96.9% success rates with grippers, (21.9% higher over baseline) and 100% success rates with dexterous hands (6.2% higher over baseline) in pick-and-place tasks. Besides, OmniVTLA significantly reduces task completion time and generates smoother trajectories through tactile sensing compared to existing VLA. Our ObjTac dataset can be found at https://readerek.github.io/Objtac.github.io
>
---
#### [replaced 010] SIGMA: Sheaf-Informed Geometric Multi-Agent Pathfinding
- **分类: cs.RO; cs.AI; cs.MA**

- **链接: [http://arxiv.org/pdf/2502.06440v2](http://arxiv.org/pdf/2502.06440v2)**

> **作者:** Shuhao Liao; Weihang Xia; Yuhong Cao; Weiheng Dai; Chengyang He; Wenjun Wu; Guillaume Sartoretti
>
> **备注:** Accepted for presentation at the 2025 IEEE International Conference on Robotics and Automation (ICRA)
>
> **摘要:** The Multi-Agent Path Finding (MAPF) problem aims to determine the shortest and collision-free paths for multiple agents in a known, potentially obstacle-ridden environment. It is the core challenge for robotic deployments in large-scale logistics and transportation. Decentralized learning-based approaches have shown great potential for addressing the MAPF problems, offering more reactive and scalable solutions. However, existing learning-based MAPF methods usually rely on agents making decisions based on a limited field of view (FOV), resulting in short-sighted policies and inefficient cooperation in complex scenarios. There, a critical challenge is to achieve consensus on potential movements between agents based on limited observations and communications. To tackle this challenge, we introduce a new framework that applies sheaf theory to decentralized deep reinforcement learning, enabling agents to learn geometric cross-dependencies between each other through local consensus and utilize them for tightly cooperative decision-making. In particular, sheaf theory provides a mathematical proof of conditions for achieving global consensus through local observation. Inspired by this, we incorporate a neural network to approximately model the consensus in latent space based on sheaf theory and train it through self-supervised learning. During the task, in addition to normal features for MAPF as in previous works, each agent distributedly reasons about a learned consensus feature, leading to efficient cooperation on pathfinding and collision avoidance. As a result, our proposed method demonstrates significant improvements over state-of-the-art learning-based MAPF planners, especially in relatively large and complex scenarios, demonstrating its superiority over baselines in various simulations and real-world robot experiments. The code is available at https://github.com/marmotlab/SIGMA
>
---
