# 机器人 cs.RO

- **最新发布 50 篇**

- **更新 28 篇**

## 最新发布

#### [new 001] A Sequential Hermaphrodite Coupling Mechanism for Lattice-based Modular Robots
- **分类: cs.RO**

- **简介: 该论文属于机器人机械设计任务，旨在解决模块化机器人耦合机制的复杂问题。提出一种顺序雌雄耦合机制，实现单侧耦合与解耦，适用于多种机器人系统。**

- **链接: [https://arxiv.org/pdf/2512.23154v1](https://arxiv.org/pdf/2512.23154v1)**

> **作者:** Keigo Torii; Kentaro Uno; Shreya Santra; Kazuya Yoshida
>
> **备注:** Author's version of a manuscript accepted at the 2025 IEEE International Conference on Mechatronics (ICM). (c) IEEE. The final published version is available at https://doi.org/10.1109/ICM62621.2025.10934866
>
> **摘要:** Lattice-based modular robot systems are envisioned for large-scale construction in extreme environments, such as space. Coupling mechanisms for heterogeneous structural modules should meet all of the following requirements: single-sided coupling and decoupling, flat surfaces when uncoupled, and coupling to passive coupling interfaces as well as coupling behavior between coupling mechanisms. The design requirements for such a coupling mechanism are complex. We propose a novel shape-matching mechanical coupling mechanism that satisfies these design requirements. This mechanism enables controlled, sequential transitions between male and female states. When uncoupled, all mechanisms are in the female state. To enable single-sided coupling, one side of the mechanisms switches to the male state during the coupling process. Single-sided decoupling is possible not only from the male side but also from the female side by forcibly switching the opposite mechanism's male state to the female state. This coupling mechanism can be applied to various modular robot systems and robot arm tool changers.
>
---
#### [new 002] Beyond Coverage Path Planning: Can UAV Swarms Perfect Scattered Regions Inspections?
- **分类: cs.RO**

- **简介: 该论文研究多无人机协同巡检任务，解决分散区域高效覆盖问题。提出mUDAI方法，优化拍摄位置与路径，提升效率并减少冗余。**

- **链接: [https://arxiv.org/pdf/2512.23257v1](https://arxiv.org/pdf/2512.23257v1)**

> **作者:** Socratis Gkelios; Savvas D. Apostolidis; Pavlos Ch. Kapoutsis; Elias B. Kosmatopoulos; Athanasios Ch. Kapoutsis
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) have revolutionized inspection tasks by offering a safer, more efficient, and flexible alternative to traditional methods. However, battery limitations often constrain their effectiveness, necessitating the development of optimized flight paths and data collection techniques. While existing approaches like coverage path planning (CPP) ensure comprehensive data collection, they can be inefficient, especially when inspecting multiple non connected Regions of Interest (ROIs). This paper introduces the Fast Inspection of Scattered Regions (FISR) problem and proposes a novel solution, the multi UAV Disjoint Areas Inspection (mUDAI) method. The introduced approach implements a two fold optimization procedure, for calculating the best image capturing positions and the most efficient UAV trajectories, balancing data resolution and operational time, minimizing redundant data collection and resource consumption. The mUDAI method is designed to enable rapid, efficient inspections of scattered ROIs, making it ideal for applications such as security infrastructure assessments, agricultural inspections, and emergency site evaluations. A combination of simulated evaluations and real world deployments is used to validate and quantify the method's ability to improve operational efficiency while preserving high quality data capture, demonstrating its effectiveness in real world operations. An open source Python implementation of the mUDAI method can be found on GitHub (https://github.com/soc12/mUDAI) and the collected and processed data from the real world experiments are all hosted on Zenodo (https://zenodo.org/records/13866483). Finally, this online platform (https://sites.google.com/view/mudai-platform/) allows interested readers to interact with the mUDAI method and generate their own multi UAV FISR missions.
>
---
#### [new 003] Active Constraint Learning in High Dimensions from Demonstrations
- **分类: cs.RO; cs.AI; cs.LG; eess.SY; math.OC**

- **简介: 该论文属于强化学习中的约束学习任务，旨在从演示中准确推断未知约束。通过迭代主动学习，利用高斯过程建模约束，提升约束推断效果。**

- **链接: [https://arxiv.org/pdf/2512.22757v1](https://arxiv.org/pdf/2512.22757v1)**

> **作者:** Zheng Qiu; Chih-Yuan Chiu; Glen Chou
>
> **备注:** Under review, 25 pages, 11 figures
>
> **摘要:** We present an iterative active constraint learning (ACL) algorithm, within the learning from demonstrations (LfD) paradigm, which intelligently solicits informative demonstration trajectories for inferring an unknown constraint in the demonstrator's environment. Our approach iteratively trains a Gaussian process (GP) on the available demonstration dataset to represent the unknown constraints, uses the resulting GP posterior to query start/goal states, and generates informative demonstrations which are added to the dataset. Across simulation and hardware experiments using high-dimensional nonlinear dynamics and unknown nonlinear constraints, our method outperforms a baseline, random-sampling based method at accurately performing constraint inference from an iteratively generated set of sparse but informative demonstrations.
>
---
#### [new 004] RoboMirror: Understand Before You Imitate for Video to Humanoid Locomotion
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出RoboMirror，解决视频到人形机器人运动的控制问题，通过视觉理解实现精准模仿，无需姿态重建，提升任务成功率与控制效率。**

- **链接: [https://arxiv.org/pdf/2512.23649v1](https://arxiv.org/pdf/2512.23649v1)**

> **作者:** Zhe Li; Cheng Chi; Yangyang Wei; Boan Zhu; Tao Huang; Zhenguo Sun; Yibo Peng; Pengwei Wang; Zhongyuan Wang; Fangzhou Liu; Chang Xu; Shanghang Zhang
>
> **摘要:** Humans learn locomotion through visual observation, interpreting visual content first before imitating actions. However, state-of-the-art humanoid locomotion systems rely on either curated motion capture trajectories or sparse text commands, leaving a critical gap between visual understanding and control. Text-to-motion methods suffer from semantic sparsity and staged pipeline errors, while video-based approaches only perform mechanical pose mimicry without genuine visual understanding. We propose RoboMirror, the first retargeting-free video-to-locomotion framework embodying "understand before you imitate". Leveraging VLMs, it distills raw egocentric/third-person videos into visual motion intents, which directly condition a diffusion-based policy to generate physically plausible, semantically aligned locomotion without explicit pose reconstruction or retargeting. Extensive experiments validate the effectiveness of RoboMirror, it enables telepresence via egocentric videos, drastically reduces third-person control latency by 80%, and achieves a 3.7% higher task success rate than baselines. By reframing humanoid control around video understanding, we bridge the visual understanding and action gap.
>
---
#### [new 005] Embodied Learning of Reward for Musculoskeletal Control with Vision Language Models
- **分类: cs.RO**

- **简介: 该论文属于强化学习任务，旨在解决高维肌肉骨骼系统中奖励函数设计困难的问题。通过引入视觉语言模型，构建框架MoVLR，实现从语言描述到运动控制的映射与优化。**

- **链接: [https://arxiv.org/pdf/2512.23077v1](https://arxiv.org/pdf/2512.23077v1)**

> **作者:** Saraswati Soedarmadji; Yunyue Wei; Chen Zhang; Yisong Yue; Yanan Sui
>
> **备注:** 18 pages, 8 figures
>
> **摘要:** Discovering effective reward functions remains a fundamental challenge in motor control of high-dimensional musculoskeletal systems. While humans can describe movement goals explicitly such as "walking forward with an upright posture," the underlying control strategies that realize these goals are largely implicit, making it difficult to directly design rewards from high-level goals and natural language descriptions. We introduce Motion from Vision-Language Representation (MoVLR), a framework that leverages vision-language models (VLMs) to bridge the gap between goal specification and movement control. Rather than relying on handcrafted rewards, MoVLR iteratively explores the reward space through iterative interaction between control optimization and VLM feedback, aligning control policies with physically coordinated behaviors. Our approach transforms language and vision-based assessments into structured guidance for embodied learning, enabling the discovery and refinement of reward functions for high-dimensional musculoskeletal locomotion and manipulation. These results suggest that VLMs can effectively ground abstract motion descriptions in the implicit principles governing physiological motor control.
>
---
#### [new 006] A New Software Tool for Generating and Visualizing Robot Self-Collision Matrices
- **分类: cs.RO**

- **简介: 该论文提出一种交互式工具，用于生成和可视化机器人自碰撞矩阵，解决传统工具在动态检查、多形状支持和灵活性方面的不足。**

- **链接: [https://arxiv.org/pdf/2512.23140v1](https://arxiv.org/pdf/2512.23140v1)**

> **作者:** Roshan Klein-Seetharama; Daniel Rakita
>
> **摘要:** In robotics, it is common to check whether a given robot state results in self-intersection (i.e., a self-collision query) or to assess its distance from such an intersection (i.e., a self-proximity query). These checks are typically performed between pairs of shapes attached to different robot links. However, many of these shape pairs can be excluded in advance, as their configurations are known to always or never result in contact. This information is typically encoded in a self-collision matrix, where each entry (i, j) indicates whether a check should be performed between shape i and shape j. While the MoveIt Setup Assistant is widely used to generate such matrices, current tools are limited by static visualization, lack of proximity support, rigid single-geometry assumptions, and tedious refinement workflows, hindering flexibility and reuse in downstream robotics applications. In this work, we introduce an interactive tool that overcomes these limitations by generating and visualizing self-collision matrices across multiple shape representations, enabling dynamic inspection, filtering, and refinement of shape pairs. Outputs are provided in both JSON and YAML for easy integration. The system is implemented in Rust and uses the Bevy game engine to deliver high-quality visualizations. We demonstrate its effectiveness on multiple robot platforms, showing that matrices generated using diverse shape types yield faster and more accurate self-collision and self-proximity queries.
>
---
#### [new 007] ParaMaP: Parallel Mapping and Collision-free Motion Planning for Reactive Robot Manipulation
- **分类: cs.RO**

- **简介: 该论文提出ParaMaP框架，解决未知环境中机器人操作的实时、无碰撞运动规划问题。整合EDT环境表示与SMPC规划，实现并行映射与规划，提升响应速度与准确性。**

- **链接: [https://arxiv.org/pdf/2512.22575v1](https://arxiv.org/pdf/2512.22575v1)**

> **作者:** Xuewei Zhang; Bailing Tian; Kai Zheng; Yulin Hui; Junjie Lu; Zhiyu Li
>
> **摘要:** Real-time and collision-free motion planning remains challenging for robotic manipulation in unknown environments due to continuous perception updates and the need for frequent online replanning. To address these challenges, we propose a parallel mapping and motion planning framework that tightly integrates Euclidean Distance Transform (EDT)-based environment representation with a sampling-based model predictive control (SMPC) planner. On the mapping side, a dense distance-field-based representation is constructed using a GPU-based EDT and augmented with a robot-masked update mechanism to prevent false self-collision detections during online perception. On the planning side, motion generation is formulated as a stochastic optimization problem with a unified objective function and efficiently solved by evaluating large batches of candidate rollouts in parallel within a SMPC framework, in which a geometrically consistent pose tracking metric defined on SE(3) is incorporated to ensure fast and accurate convergence to the target pose. The entire mapping and planning pipeline is implemented on the GPU to support high-frequency replanning. The effectiveness of the proposed framework is validated through extensive simulations and real-world experiments on a 7-DoF robotic manipulator. More details are available at: https://zxw610.github.io/ParaMaP.
>
---
#### [new 008] Sistema de navegación de cobertura para vehículos no holonómicos en ambientes de exterior
- **分类: cs.RO**

- **简介: 该论文属于移动机器人覆盖导航任务，旨在解决非全向机器人在户外环境中的全面覆盖问题，通过路径规划与避障机制实现高效覆盖。**

- **链接: [https://arxiv.org/pdf/2512.22734v1](https://arxiv.org/pdf/2512.22734v1)**

> **作者:** Michelle Valenzuela; Francisco Leiva; Javier Ruiz-del-Solar
>
> **备注:** 13 pages, in Spanish language, 12 figures, accepted at Tercer Congreso Iberoamericano de Minería Subterranea y a Cielo Abierto, UMining 2024
>
> **摘要:** In mobile robotics, coverage navigation refers to the deliberate movement of a robot with the purpose of covering a certain area or volume. Performing this task properly is fundamental for the execution of several activities, for instance, cleaning a facility with a robotic vacuum cleaner. In the mining industry, it is required to perform coverage in several unit processes related with material movement using industrial machinery, for example, in cleaning tasks, in dumps, and in the construction of tailings dam walls. The automation of these processes is fundamental to enhance the security associated with their execution. In this work, a coverage navigation system for a non-holonomic robot is presented. This work is intended to be a proof of concept for the potential automation of various unit processes that require coverage navigation like the ones mentioned before. The developed system includes the calculation of routes that allow a mobile platform to cover a specific area, and incorporates recovery behaviors in case that an unforeseen event occurs, such as the arising of dynamic or previously unmapped obstacles in the terrain to be covered, e.g., other machines or pedestrians passing through the area, being able to perform evasive maneuvers and post-recovery to ensure a complete coverage of the terrain. The system was tested in different simulated and real outdoor environments, obtaining results near 90% of coverage in the majority of experiments. The next step of development is to scale up the utilized robot to a mining machine/vehicle whose operation will be validated in a real environment. The result of one of the tests performed in the real world can be seen in the video available in https://youtu.be/gK7_3bK1P5g.
>
---
#### [new 009] Bugs with Features: Vision-Based Fault-Tolerant Collective Motion Inspired by Nature
- **分类: cs.RO; cs.MA**

- **简介: 该论文研究群体运动的鲁棒性问题，针对视觉感知带来的不确定性，提出距离估计和间歇运动机制，提升机器人集群的抗故障能力。**

- **链接: [https://arxiv.org/pdf/2512.22448v1](https://arxiv.org/pdf/2512.22448v1)**

> **作者:** Peleg Shefi; Amir Ayali; Gal A. Kaminka
>
> **摘要:** In collective motion, perceptually-limited individuals move in an ordered manner, without centralized control. The perception of each individual is highly localized, as is its ability to interact with others. While natural collective motion is robust, most artificial swarms are brittle. This particularly occurs when vision is used as the sensing modality, due to ambiguities and information-loss inherent in visual perception. This paper presents mechanisms for robust collective motion inspired by studies of locusts. First, we develop a robust distance estimation method that combines visually perceived horizontal and vertical sizes of neighbors. Second, we introduce intermittent locomotion as a mechanism that allows robots to reliably detect peers that fail to keep up, and disrupt the motion of the swarm. We show how such faulty robots can be avoided in a manner that is robust to errors in classifying them as faulty. Through extensive physics-based simulation experiments, we show dramatic improvements to swarm resilience when using these techniques. We show these are relevant to both distance-based Avoid-Attract models, as well as to models relying on Alignment, in a wide range of experiment settings.
>
---
#### [new 010] PCR-ORB: Enhanced ORB-SLAM3 with Point Cloud Refinement Using Deep Learning-Based Dynamic Object Filtering
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于视觉SLAM任务，旨在解决动态环境中移动物体对定位与建图的干扰问题。通过结合深度学习和点云优化，提升系统鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.23318v1](https://arxiv.org/pdf/2512.23318v1)**

> **作者:** Sheng-Kai Chen; Jie-Yu Chao; Jr-Yu Chang; Po-Lien Wu; Po-Chiang Lin
>
> **备注:** 17 pages, 2 figures, 1 table
>
> **摘要:** Visual Simultaneous Localization and Mapping (vSLAM) systems encounter substantial challenges in dynamic environments where moving objects compromise tracking accuracy and map consistency. This paper introduces PCR-ORB (Point Cloud Refinement ORB), an enhanced ORB-SLAM3 framework that integrates deep learning-based point cloud refinement to mitigate dynamic object interference. Our approach employs YOLOv8 for semantic segmentation combined with CUDA-accelerated processing to achieve real-time performance. The system implements a multi-stage filtering strategy encompassing ground plane estimation, sky region removal, edge filtering, and temporal consistency validation. Comprehensive evaluation on the KITTI dataset (sequences 00-09) demonstrates performance characteristics across different environmental conditions and scene types. Notable improvements are observed in specific sequences, with sequence 04 achieving 25.9% improvement in ATE RMSE and 30.4% improvement in ATE median. However, results show mixed performance across sequences, indicating scenario-dependent effectiveness. The implementation provides insights into dynamic object filtering challenges and opportunities for robust navigation in complex environments.
>
---
#### [new 011] Do You Have Freestyle? Expressive Humanoid Locomotion via Audio Control
- **分类: cs.RO**

- **简介: 该论文提出RoboPerform框架，解决机器人缺乏音乐与语音驱动的表达性运动问题。通过音频直接生成动作，无需显式重建运动，实现低延迟、高保真响应。**

- **链接: [https://arxiv.org/pdf/2512.23650v1](https://arxiv.org/pdf/2512.23650v1)**

> **作者:** Zhe Li; Cheng Chi; Yangyang Wei; Boan Zhu; Tao Huang; Zhenguo Sun; Yibo Peng; Pengwei Wang; Zhongyuan Wang; Fangzhou Liu; Chang Xu; Shanghang Zhang
>
> **摘要:** Humans intuitively move to sound, but current humanoid robots lack expressive improvisational capabilities, confined to predefined motions or sparse commands. Generating motion from audio and then retargeting it to robots relies on explicit motion reconstruction, leading to cascaded errors, high latency, and disjointed acoustic-actuation mapping. We propose RoboPerform, the first unified audio-to-locomotion framework that can directly generate music-driven dance and speech-driven co-speech gestures from audio. Guided by the core principle of "motion = content + style", the framework treats audio as implicit style signals and eliminates the need for explicit motion reconstruction. RoboPerform integrates a ResMoE teacher policy for adapting to diverse motion patterns and a diffusion-based student policy for audio style injection. This retargeting-free design ensures low latency and high fidelity. Experimental validation shows that RoboPerform achieves promising results in physical plausibility and audio alignment, successfully transforming robots into responsive performers capable of reacting to audio.
>
---
#### [new 012] SurgWorld: Learning Surgical Robot Policies from Videos via World Modeling
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于手术机器人领域，旨在解决数据稀缺问题。通过构建SurgWorld模型和SATA数据集，生成合成视频与动作数据，提升手术机器人自主学习能力。**

- **链接: [https://arxiv.org/pdf/2512.23162v1](https://arxiv.org/pdf/2512.23162v1)**

> **作者:** Yufan He; Pengfei Guo; Mengya Xu; Zhaoshuo Li; Andriy Myronenko; Dillan Imans; Bingjie Liu; Dongren Yang; Mingxue Gu; Yongnan Ji; Yueming Jin; Ren Zhao; Baiyong Shen; Daguang Xu
>
> **摘要:** Data scarcity remains a fundamental barrier to achieving fully autonomous surgical robots. While large scale vision language action (VLA) models have shown impressive generalization in household and industrial manipulation by leveraging paired video action data from diverse domains, surgical robotics suffers from the paucity of datasets that include both visual observations and accurate robot kinematics. In contrast, vast corpora of surgical videos exist, but they lack corresponding action labels, preventing direct application of imitation learning or VLA training. In this work, we aim to alleviate this problem by learning policy models from SurgWorld, a world model designed for surgical physical AI. We curated the Surgical Action Text Alignment (SATA) dataset with detailed action description specifically for surgical robots. Then we built SurgeWorld based on the most advanced physical AI world model and SATA. It's able to generate diverse, generalizable and realistic surgery videos. We are also the first to use an inverse dynamics model to infer pseudokinematics from synthetic surgical videos, producing synthetic paired video action data. We demonstrate that a surgical VLA policy trained with these augmented data significantly outperforms models trained only on real demonstrations on a real surgical robot platform. Our approach offers a scalable path toward autonomous surgical skill acquisition by leveraging the abundance of unlabeled surgical video and generative world modeling, thus opening the door to generalizable and data efficient surgical robot policies.
>
---
#### [new 013] Joint UAV-UGV Positioning and Trajectory Planning via Meta A3C for Reliable Emergency Communications
- **分类: cs.RO; cs.ET; eess.SY**

- **简介: 该论文属于无人机与无人车协同定位与轨迹规划任务，旨在优化应急通信中的QoS，减少无人机使用数量。通过引入道路图和Meta-A3C算法解决动态环境下的路径优化问题。**

- **链接: [https://arxiv.org/pdf/2512.22187v1](https://arxiv.org/pdf/2512.22187v1)**

> **作者:** Ndagijimana Cyprien; Mehdi Sookhak; Hosein Zarini; Chandra N Sekharan; Mohammed Atiquzzaman
>
> **摘要:** Joint deployment of unmanned aerial vehicles (UAVs) and unmanned ground vehicles (UGVs) has been shown to be an effective method to establish communications in areas affected by disasters. However, ensuring good Quality of Services (QoS) while using as few UAVs as possible also requires optimal positioning and trajectory planning for UAVs and UGVs. This paper proposes a joint UAV-UGV-based positioning and trajectory planning framework for UAVs and UGVs deployment that guarantees optimal QoS for ground users. To model the UGVs' mobility, we introduce a road graph, which directs their movement along valid road segments and adheres to the road network constraints. To solve the sum rate optimization problem, we reformulate the problem as a Markov Decision Process (MDP) and propose a novel asynchronous Advantage Actor Critic (A3C) incorporated with meta-learning for rapid adaptation to new environments and dynamic conditions. Numerical results demonstrate that our proposed Meta-A3C approach outperforms A3C and DDPG, delivering 13.1\% higher throughput and 49\% faster execution while meeting the QoS requirements.
>
---
#### [new 014] Unsupervised Learning for Detection of Rare Driving Scenarios
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于异常检测任务，旨在解决自动驾驶中罕见危险场景的识别问题。通过无监督学习方法，结合深度特征与隔离森林，有效检测复杂异常场景。**

- **链接: [https://arxiv.org/pdf/2512.23585v1](https://arxiv.org/pdf/2512.23585v1)**

> **作者:** Dat Le; Thomas Manhardt; Moritz Venator; Johannes Betz
>
> **摘要:** The detection of rare and hazardous driving scenarios is a critical challenge for ensuring the safety and reliability of autonomous systems. This research explores an unsupervised learning framework for detecting rare and extreme driving scenarios using naturalistic driving data (NDD). We leverage the recently proposed Deep Isolation Forest (DIF), an anomaly detection algorithm that combines neural network-based feature representations with Isolation Forests (IFs), to identify non-linear and complex anomalies. Data from perception modules, capturing vehicle dynamics and environmental conditions, is preprocessed into structured statistical features extracted from sliding windows. The framework incorporates t-distributed stochastic neighbor embedding (t-SNE) for dimensionality reduction and visualization, enabling better interpretability of detected anomalies. Evaluation is conducted using a proxy ground truth, combining quantitative metrics with qualitative video frame inspection. Our results demonstrate that the proposed approach effectively identifies rare and hazardous driving scenarios, providing a scalable solution for anomaly detection in autonomous driving systems. Given the study's methodology, it was unavoidable to depend on proxy ground truth and manually defined feature combinations, which do not encompass the full range of real-world driving anomalies or their nuanced contextual dependencies.
>
---
#### [new 015] APOLLO Blender: A Robotics Library for Visualization and Animation in Blender
- **分类: cs.RO**

- **简介: 该论文属于机器人可视化任务，旨在解决研究人员难以高效使用Blender的问题。通过开发APOLLO Blender库，实现机器人模型导入、状态关键帧 scripting 和3D形状生成，提升可视化效率。**

- **链接: [https://arxiv.org/pdf/2512.23103v1](https://arxiv.org/pdf/2512.23103v1)**

> **作者:** Peter Messina; Daniel Rakita
>
> **摘要:** High-quality visualizations are an essential part of robotics research, enabling clear communication of results through figures, animations, and demonstration videos. While Blender is a powerful and freely available 3D graphics platform, its steep learning curve and lack of robotics-focused integrations make it difficult and time-consuming for researchers to use effectively. In this work, we introduce a lightweight software library that bridges this gap by providing simple scripting interfaces for common robotics visualization tasks. The library offers three primary capabilities: (1) importing robots and environments directly from standardized descriptions such as URDF; (2) Python-based scripting tools for keyframing robot states and visual attributes; and (3) convenient generation of primitive 3D shapes for schematic figures and animations. Together, these features allow robotics researchers to rapidly create publication-ready images, animations, and explanatory schematics without needing extensive Blender expertise. We demonstrate the library through a series of proof-of-concept examples and conclude with a discussion of current limitations and opportunities for future extensions.
>
---
#### [new 016] The body is not there to compute: Comment on "Informational embodiment: Computational role of information structure in codes and robots" by Pitti et al
- **分类: cs.RO; cs.AI; q-bio.NC; q-bio.QM**

- **简介: 该论文属于评论类任务，针对Pitti等人的研究提出不同观点，认为身体的主要功能不是计算，而是其他作用。**

- **链接: [https://arxiv.org/pdf/2512.22868v1](https://arxiv.org/pdf/2512.22868v1)**

> **作者:** Matej Hoffmann
>
> **备注:** Comment on Pitti, A., Austin, M., Nakajima, K., & Kuniyoshi, Y. (2025). Informational Embodiment: Computational role of information structure in codes and robots. Physics of Life Reviews 53, 262-276. https://doi.org/10.1016/j.plrev.2025.03.018. Also available as arXiv:2408.12950
>
> **摘要:** Applying the lens of computation and information has been instrumental in driving the technological progress of our civilization as well as in empowering our understanding of the world around us. The digital computer was and for many still is the leading metaphor for how our mind operates. Information theory (IT) has also been important in our understanding of how nervous systems encode and process information. The target article deploys information and computation to bodies: to understand why they have evolved in particular ways (animal bodies) and to design optimal bodies (robots). In this commentary, I argue that the main role of bodies is not to compute.
>
---
#### [new 017] Explainable Neural Inverse Kinematics for Obstacle-Aware Robotic Manipulation: A Comparative Analysis of IKNet Variants
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于机器人逆运动学任务，旨在解决深度学习模型不透明带来的安全问题。通过改进IKNet结构并结合XAI技术，提升模型可解释性与避障能力。**

- **链接: [https://arxiv.org/pdf/2512.23312v1](https://arxiv.org/pdf/2512.23312v1)**

> **作者:** Sheng-Kai Chen; Yi-Ling Tsai; Chun-Chih Chang; Yan-Chen Chen; Po-Chiang Lin
>
> **备注:** 27 pages, 16 figures
>
> **摘要:** Deep neural networks have accelerated inverse-kinematics (IK) inference to the point where low cost manipulators can execute complex trajectories in real time, yet the opaque nature of these models contradicts the transparency and safety requirements emerging in responsible AI regulation. This study proposes an explainability centered workflow that integrates Shapley-value attribution with physics-based obstacle avoidance evaluation for the ROBOTIS OpenManipulator-X. Building upon the original IKNet, two lightweight variants-Improved IKNet with residual connections and Focused IKNet with position-orientation decoupling are trained on a large, synthetically generated pose-joint dataset. SHAP is employed to derive both global and local importance rankings, while the InterpretML toolkit visualizes partial-dependence patterns that expose non-linear couplings between Cartesian poses and joint angles. To bridge algorithmic insight and robotic safety, each network is embedded in a simulator that subjects the arm to randomized single and multi-obstacle scenes; forward kinematics, capsule-based collision checks, and trajectory metrics quantify the relationship between attribution balance and physical clearance. Qualitative heat maps reveal that architectures distributing importance more evenly across pose dimensions tend to maintain wider safety margins without compromising positional accuracy. The combined analysis demonstrates that explainable AI(XAI) techniques can illuminate hidden failure modes, guide architectural refinements, and inform obstacle aware deployment strategies for learning based IK. The proposed methodology thus contributes a concrete path toward trustworthy, data-driven manipulation that aligns with emerging responsible-AI standards.
>
---
#### [new 018] Embodied Robot Manipulation in the Era of Foundation Models: Planning and Learning Perspectives
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决复杂环境下的机械臂操控问题。通过高阶规划与低阶控制的统一框架，整合学习方法，提升机器人操作能力。**

- **链接: [https://arxiv.org/pdf/2512.22983v1](https://arxiv.org/pdf/2512.22983v1)**

> **作者:** Shuanghao Bai; Wenxuan Song; Jiayi Chen; Yuheng Ji; Zhide Zhong; Jin Yang; Han Zhao; Wanqi Zhou; Zhe Li; Pengxiang Ding; Cheng Chi; Chang Xu; Xiaolong Zheng; Donglin Wang; Haoang Li; Shanghang Zhang; Badong Chen
>
> **备注:** This work is a re-architected core derived from the full survey (arXiv:2510.10903) , refined to highlight the most central themes and representative studies
>
> **摘要:** Recent advances in vision, language, and multimodal learning have substantially accelerated progress in robotic foundation models, with robot manipulation remaining a central and challenging problem. This survey examines robot manipulation from an algorithmic perspective and organizes recent learning-based approaches within a unified abstraction of high-level planning and low-level control. At the high level, we extend the classical notion of task planning to include reasoning over language, code, motion, affordances, and 3D representations, emphasizing their role in structured and long-horizon decision making. At the low level, we propose a training-paradigm-oriented taxonomy for learning-based control, organizing existing methods along input modeling, latent representation learning, and policy learning. Finally, we identify open challenges and prospective research directions related to scalability, data efficiency, multimodal physical interaction, and safety. Together, these analyses aim to clarify the design space of modern foundation models for robotic manipulation.
>
---
#### [new 019] A Unified AI, Embedded, Simulation, and Mechanical Design Approach to an Autonomous Delivery Robot
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于自主配送机器人开发任务，解决资源受限平台上的AI算法优化与实时控制问题，整合AI、嵌入式系统和机械设计实现可靠自主系统。**

- **链接: [https://arxiv.org/pdf/2512.22408v1](https://arxiv.org/pdf/2512.22408v1)**

> **作者:** Amro Gamar; Ahmed Abduljalil; Alargam Mohammed; Ali Elhenidy; Abeer Tawakol
>
> **摘要:** This paper presents the development of a fully autonomous delivery robot integrating mechanical engineering, embedded systems, and artificial intelligence. The platform employs a heterogeneous computing architecture, with RPi 5 and ROS 2 handling AI-based perception and path planning, while ESP32 running FreeRTOS ensures real-time motor control. The mechanical design was optimized for payload capacity and mobility through precise motor selection and material engineering. Key technical challenges addressed include optimizing computationally intensive AI algorithms on a resource-constrained platform and implementing a low-latency, reliable communication link between the ROS 2 host and embedded controller. Results demonstrate deterministic, PID-based motor control through rigorous memory and task management, and enhanced system reliability via AWS IoT monitoring and a firmware-level motor shutdown failsafe. This work highlights a unified, multi-disciplinary methodology, resulting in a robust and operational autonomous delivery system capable of real-world deployment.
>
---
#### [new 020] Theory of Mind for Explainable Human-Robot Interaction
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于人机交互任务，旨在解决机器人解释能力不足的问题。通过将心智理论融入可解释AI框架，提升机器人对用户心理状态的理解与解释能力。**

- **链接: [https://arxiv.org/pdf/2512.23482v1](https://arxiv.org/pdf/2512.23482v1)**

> **作者:** Marie Bauer; Julia Gachot; Matthias Kerzel; Cornelius Weber; Stefan Wermter
>
> **摘要:** Within the context of human-robot interaction (HRI), Theory of Mind (ToM) is intended to serve as a user-friendly backend to the interface of robotic systems, enabling robots to infer and respond to human mental states. When integrated into robots, ToM allows them to adapt their internal models to users' behaviors, enhancing the interpretability and predictability of their actions. Similarly, Explainable Artificial Intelligence (XAI) aims to make AI systems transparent and interpretable, allowing humans to understand and interact with them effectively. Since ToM in HRI serves related purposes, we propose to consider ToM as a form of XAI and evaluate it through the eValuation XAI (VXAI) framework and its seven desiderata. This paper identifies a critical gap in the application of ToM within HRI, as existing methods rarely assess the extent to which explanations correspond to the robot's actual internal reasoning. To address this limitation, we propose to integrate ToM within XAI frameworks. By embedding ToM principles inside XAI, we argue for a shift in perspective, as current XAI research focuses predominantly on the AI system itself and often lacks user-centered explanations. Incorporating ToM would enable a change in focus, prioritizing the user's informational needs and perspective.
>
---
#### [new 021] VL-LN Bench: Towards Long-horizon Goal-oriented Navigation with Active Dialogs
- **分类: cs.RO**

- **简介: 该论文提出IION任务，解决真实场景下导航指令模糊的问题，通过主动对话增强导航模型，构建VL-LN基准进行评估。**

- **链接: [https://arxiv.org/pdf/2512.22342v1](https://arxiv.org/pdf/2512.22342v1)**

> **作者:** Wensi Huang; Shaohao Zhu; Meng Wei; Jinming Xu; Xihui Liu; Hanqing Wang; Tai Wang; Feng Zhao; Jiangmiao Pang
>
> **摘要:** In most existing embodied navigation tasks, instructions are well-defined and unambiguous, such as instruction following and object searching. Under this idealized setting, agents are required solely to produce effective navigation outputs conditioned on vision and language inputs. However, real-world navigation instructions are often vague and ambiguous, requiring the agent to resolve uncertainty and infer user intent through active dialog. To address this gap, we propose Interactive Instance Object Navigation (IION), a task that requires agents not only to generate navigation actions but also to produce language outputs via active dialog, thereby aligning more closely with practical settings. IION extends Instance Object Navigation (ION) by allowing agents to freely consult an oracle in natural language while navigating. Building on this task, we present the Vision Language-Language Navigation (VL-LN) benchmark, which provides a large-scale, automatically generated dataset and a comprehensive evaluation protocol for training and assessing dialog-enabled navigation models. VL-LN comprises over 41k long-horizon dialog-augmented trajectories for training and an automatic evaluation protocol with an oracle capable of responding to agent queries. Using this benchmark, we train a navigation model equipped with dialog capabilities and show that it achieves significant improvements over the baselines. Extensive experiments and analyses further demonstrate the effectiveness and reliability of VL-LN for advancing research on dialog-enabled embodied navigation. Code and dataset: https://0309hws.github.io/VL-LN.github.io/
>
---
#### [new 022] P-FABRIK: A General Intuitive and Robust Inverse Kinematics Method for Parallel Mechanisms Using FABRIK Approach
- **分类: cs.RO**

- **简介: 该论文属于逆运动学任务，解决并联机构在冗余和非工作空间目标下的求解难题。提出P-FABRIK方法，通过分解机制并迭代计算，实现高效鲁棒的逆运动学解。**

- **链接: [https://arxiv.org/pdf/2512.22927v1](https://arxiv.org/pdf/2512.22927v1)**

> **作者:** Daqian Cao; Quan Yuan; Weibang Bai
>
> **备注:** 7 pages, 8 figures, and 2 tables
>
> **摘要:** Traditional geometric inverse kinematics methods for parallel mechanisms rely on specific spatial geometry constraints. However, their application to redundant parallel mechanisms is challenged due to the increased constraint complexity. Moreover, it will output no solutions and cause unpredictable control problems when the target pose lies outside its workspace. To tackle these challenging issues, this work proposes P-FABRIK, a general, intuitive, and robust inverse kinematics method to find one feasible solution for diverse parallel mechanisms based on the FABRIK algorithm. By decomposing the general parallel mechanism into multiple serial sub-chains using a new topological decomposition strategy, the end targets of each sub-chain can be subsequently revised to calculate the inverse kinematics solutions iteratively. Multiple case studies involving planar, standard, and redundant parallel mechanisms demonstrated the proposed method's generality across diverse parallel mechanisms. Furthermore, numerical simulation studies verified its efficacy and computational efficiency, as well as its robustness ability to handle out-of-workspace targets.
>
---
#### [new 023] Pole-centric Descriptors for Robust Robot Localization: Evaluation under Pole-at-Distance (PaD) Observations using the Small Pole Landmark (SPL) Dataset
- **分类: cs.RO**

- **简介: 该论文属于机器人定位任务，解决远距离杆状地标识别可靠性问题。通过构建SPL数据集，对比分析对比学习与监督学习的鲁棒性，提升稀疏几何下的定位性能。**

- **链接: [https://arxiv.org/pdf/2512.23141v1](https://arxiv.org/pdf/2512.23141v1)**

> **作者:** Wuhao Xie; Kanji Tanaka
>
> **备注:** 3 pages, technical report
>
> **摘要:** While pole-like structures are widely recognized as stable geometric anchors for long-term robot localization, their identification reliability degrades significantly under Pole-at-Distance (Pad) observations typical of large-scale urban environments. This paper shifts the focus from descriptor design to a systematic investigation of descriptor robustness. Our primary contribution is the establishment of a specialized evaluation framework centered on the Small Pole Landmark (SPL) dataset. This dataset is constructed via an automated tracking-based association pipeline that captures multi-view, multi-distance observations of the same physical landmarks without manual annotation. Using this framework, we present a comparative analysis of Contrastive Learning (CL) and Supervised Learning (SL) paradigms. Our findings reveal that CL induces a more robust feature space for sparse geometry, achieving superior retrieval performance particularly in the 5--10m range. This work provides an empirical foundation and a scalable methodology for evaluating landmark distinctiveness in challenging real-world scenarios.
>
---
#### [new 024] Asymmetric Friction in Geometric Locomotion
- **分类: cs.RO; math.DG**

- **简介: 该论文属于机器人学与运动建模任务，研究非对称摩擦下的几何运动问题。通过引入Finsler度量扩展传统方法，分析系统运动能力。**

- **链接: [https://arxiv.org/pdf/2512.22484v1](https://arxiv.org/pdf/2512.22484v1)**

> **作者:** Ross L. Hatton; Yousef Salaman; Shai Revzen
>
> **备注:** 23 pages, 15 figures
>
> **摘要:** Geometric mechanics models of locomotion have provided insight into how robots and animals use environmental interactions to convert internal shape changes into displacement through the world, encoding this relationship in a ``motility map''. A key class of such motility maps arises from (possibly anisotropic) linear drag acting on the system's individual body parts, formally described via Riemannian metrics on the motions of the system's individual body parts. The motility map can then be generated by invoking a sub-Riemannian constraint on the aggregate system motion under which the position velocity induced by a given shape velocity is that which minimizes the power dissipated via friction. The locomotion of such systems is ``geometric'' in the sense that the final position reached by the system depends only on the sequence of shapes that the system passes through, but not on the rate with which the shape changes are made. In this paper, we consider a far more general class of systems in which the drag may be not only anisotropic (with different coefficients for forward/backward and left/right motions), but also asymmetric (with different coefficients for forward and backward motions). Formally, including asymmetry in the friction replaces the Riemannian metrics on the body parts with Finsler metrics. We demonstrate that the sub-Riemannian approach to constructing the system motility map extends naturally to a sub-Finslerian approach and identify system properties analogous to the constraint curvature of sub-Riemannian systems that allow for the characterization of the system motion capabilities.
>
---
#### [new 025] The Bulldozer Technique: Efficient Elimination of Local Minima Traps for APF-Based Robot Navigation
- **分类: cs.RO**

- **简介: 该论文属于路径规划任务，解决APF方法中的局部极小值问题。提出Bulldozer技术，通过增强机制有效消除陷阱，提升导航效率与路径质量。**

- **链接: [https://arxiv.org/pdf/2512.23672v1](https://arxiv.org/pdf/2512.23672v1)**

> **作者:** Mohammed Baziyad; Manal Al Shohna; Tamer Rabie
>
> **摘要:** Path planning is a fundamental component in autonomous mobile robotics, enabling a robot to navigate from its current location to a desired goal while avoiding obstacles. Among the various techniques, Artificial Potential Field (APF) methods have gained popularity due to their simplicity, real-time responsiveness, and low computational requirements. However, a major limitation of conventional APF approaches is the local minima trap problem, where the robot becomes stuck in a position with no clear direction toward the goal. This paper proposes a novel path planning technique, termed the Bulldozer, which addresses the local minima issue while preserving the inherent advantages of APF. The Bulldozer technique introduces a backfilling mechanism that systematically identifies and eliminates local minima regions by increasing their potential values, analogous to a bulldozer filling potholes in a road. Additionally, a ramp-based enhancement is incorporated to assist the robot in escaping trap areas when starting within a local minimum. The proposed technique is experimentally validated using a physical mobile robot across various maps with increasing complexity. Comparative analyses are conducted against standard APF, adaptive APF, and well-established planning algorithms such as A*, PRM, and RRT. Results demonstrate that the Bulldozer technique effectively resolves the local minima problem while achieving superior execution speed and competitive path quality. Furthermore, a kinematic tracking controller is employed to assess the smoothness and traceability of the planned paths, confirming their suitability for real-world execution.
>
---
#### [new 026] Optimal Scalability-Aware Allocation of Swarm Robots: From Linear to Retrograde Performance via Marginal Gains
- **分类: cs.RO; cs.MA**

- **简介: 该论文研究多机器人系统中的任务分配问题，旨在优化有限资源的高效利用。通过边际增益算法，解决不同任务性能非线性变化下的机器人分配难题。**

- **链接: [https://arxiv.org/pdf/2512.23431v1](https://arxiv.org/pdf/2512.23431v1)**

> **作者:** Simay Atasoy Bingöl; Tobias Töpfer; Sven Kosub; Heiko Hamann; Andreagiovanni Reina
>
> **备注:** 14 pages, 11 figures, Accepted for publication in IEEE Transactions on Systems, Man, and Cybernetics: Systems
>
> **摘要:** In collective systems, the available agents are a limited resource that must be allocated among tasks to maximize collective performance. Computing the optimal allocation of several agents to numerous tasks through a brute-force approach can be infeasible, especially when each task's performance scales differently with the increase of agents. For example, difficult tasks may require more agents to achieve similar performances compared to simpler tasks, but performance may saturate nonlinearly as the number of allocated agents increases. We propose a computationally efficient algorithm, based on marginal performance gains, for optimally allocating agents to tasks with concave scalability functions, including linear, saturating, and retrograde scaling, to achieve maximum collective performance. We test the algorithm by allocating a simulated robot swarm among collective decision-making tasks, where embodied agents sample their environment and exchange information to reach a consensus on spatially distributed environmental features. We vary task difficulties by different geometrical arrangements of environmental features in space (patchiness). In this scenario, decision performance in each task scales either as a saturating curve (following the Condorcet's Jury Theorem in an interference-free setup) or as a retrograde curve (when physical interference among robots restricts their movement). Using simple robot simulations, we show that our algorithm can be useful in allocating robots among tasks. Our approach aims to advance the deployment of future real-world multi-robot systems.
>
---
#### [new 027] Act2Goal: From World Model To General Goal-conditioned Policy
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出Act2Goal，解决机器人长距离操作任务中的目标条件策略问题。通过整合视觉世界模型与多尺度时间控制，实现高效、鲁棒的长期操作。**

- **链接: [https://arxiv.org/pdf/2512.23541v1](https://arxiv.org/pdf/2512.23541v1)**

> **作者:** Pengfei Zhou; Liliang Chen; Shengcong Chen; Di Chen; Wenzhi Zhao; Rongjun Jin; Guanghui Ren; Jianlan Luo
>
> **摘要:** Specifying robotic manipulation tasks in a manner that is both expressive and precise remains a central challenge. While visual goals provide a compact and unambiguous task specification, existing goal-conditioned policies often struggle with long-horizon manipulation due to their reliance on single-step action prediction without explicit modeling of task progress. We propose Act2Goal, a general goal-conditioned manipulation policy that integrates a goal-conditioned visual world model with multi-scale temporal control. Given a current observation and a target visual goal, the world model generates a plausible sequence of intermediate visual states that captures long-horizon structure. To translate this visual plan into robust execution, we introduce Multi-Scale Temporal Hashing (MSTH), which decomposes the imagined trajectory into dense proximal frames for fine-grained closed-loop control and sparse distal frames that anchor global task consistency. The policy couples these representations with motor control through end-to-end cross-attention, enabling coherent long-horizon behavior while remaining reactive to local disturbances. Act2Goal achieves strong zero-shot generalization to novel objects, spatial layouts, and environments. We further enable reward-free online adaptation through hindsight goal relabeling with LoRA-based finetuning, allowing rapid autonomous improvement without external supervision. Real-robot experiments demonstrate that Act2Goal improves success rates from 30% to 90% on challenging out-of-distribution tasks within minutes of autonomous interaction, validating that goal-conditioned world models with multi-scale temporal control provide structured guidance necessary for robust long-horizon manipulation. Project page: https://act2goal.github.io/
>
---
#### [new 028] PreGME: Prescribed Performance Control of Aerial Manipulators based on Variable-Gain ESO
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于无人机机械臂控制任务，旨在解决动态耦合带来的高精度控制问题。提出PreGME方法，结合变增益ESO和预设性能控制，提升跟踪精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.22957v1](https://arxiv.org/pdf/2512.22957v1)**

> **作者:** Mengyu Ji; Shiliang Guo; Zhengzhen Li; Jiahao Shen; Huazi Cao; Shiyu Zhao
>
> **备注:** 12 pages, 6 figures
>
> **摘要:** An aerial manipulator, comprising a multirotor base and a robotic arm, is subject to significant dynamic coupling between these two components. Therefore, achieving precise and robust motion control is a challenging yet important objective. Here, we propose a novel prescribed performance motion control framework based on variable-gain extended state observers (ESOs), referred to as PreGME. The method includes variable-gain ESOs for real-time estimation of dynamic coupling and a prescribed performance flight control that incorporates error trajectory constraints. Compared with existing methods, the proposed approach exhibits the following two characteristics. First, the adopted variable-gain ESOs can accurately estimate rapidly varying dynamic coupling. This enables the proposed method to handle manipulation tasks that require aggressive motion of the robotic arm. Second, by prescribing the performance, a preset error trajectory is generated to guide the system evolution along this trajectory. This strategy allows the proposed method to ensure the tracking error remains within the prescribed performance envelope, thereby achieving high-precision control. Experiments on a real platform, including aerial staff twirling, aerial mixology, and aerial cart-pulling experiments, are conducted to validate the effectiveness of the proposed method. Experimental results demonstrate that even under the dynamic coupling caused by rapid robotic arm motion (end-effector velocity: 1.02 m/s, acceleration: 5.10 m/s$^2$), the proposed method achieves high tracking performance.
>
---
#### [new 029] Topology-Preserving Scalar Field Optimization for Boundary-Conforming Spiral Toolpaths on Multiply Connected Freeform Surfaces
- **分类: cs.RO; cs.GR**

- **简介: 该论文属于数控加工路径规划任务，解决多连通自由曲面边界一致的螺旋刀具路径优化问题，通过拓扑保持方法提升加工效率与路径连续性。**

- **链接: [https://arxiv.org/pdf/2512.22502v1](https://arxiv.org/pdf/2512.22502v1)**

> **作者:** Shen Changqing; Xu Bingzhou; Qi Bosong; Zhang Xiaojian; Yan Sijie; Ding Han
>
> **备注:** 24Pages,12Figures
>
> **摘要:** Ball-end milling path planning on multiply connected freeform surfaces is pivotal for high-quality and efficient machining of components in automotive and aerospace manufacturing. Although scalar-field-based optimization provides a unified framework for multi-objective toolpath generation, maintaining boundary conformity while eliminating zero-gradient singularities that cause iso-curve branching or termination and disrupt toolpath continuity remains challenging on multiply connected surfaces. We propose an efficient strategy to robustly enforce these constraints throughout optimization. Conformal slit mapping is employed to construct a feasible, singularity-free initial scalar field. The optimization is reformulated as a topology-preserving mesh deformation governed by boundary-synchronous updates, enabling globally optimized spacing, scallop-height uniformity, and smooth trajectory transitions. Consequently, the toolpaths are continuous, boundary-conforming, and free of self-intersections. Milling experiments demonstrate that, compared with a state-of-the-art conformal slit mapping-based method, the proposed approach increases machining efficiency by 14.24%, improves scallop-height uniformity by 5.70%, and reduces milling impact-induced vibrations by over 10%. The strategy offers broad applicability in high-performance machining scenarios.
>
---
#### [new 030] Robust Deep Learning Control with Guaranteed Performance for Safe and Reliable Robotization in Heavy-Duty Machinery
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人控制任务，旨在解决重型机械电气化与自主化中的安全控制问题，提出一种模块化、可扩展的控制框架，确保系统稳定与性能。**

- **链接: [https://arxiv.org/pdf/2512.23505v1](https://arxiv.org/pdf/2512.23505v1)**

> **作者:** Mehdi Heydari Shahna
>
> **备注:** Doctoral Dissertation, Tampere University
>
> **摘要:** Today's heavy-duty mobile machines (HDMMs) face two transitions: from diesel-hydraulic actuation to clean electric systems driven by climate goals, and from human supervision toward greater autonomy. Diesel-hydraulic systems have long dominated, so full electrification, via direct replacement or redesign, raises major technical and economic challenges. Although advanced artificial intelligence (AI) could enable higher autonomy, adoption in HDMMs is limited by strict safety requirements, and these machines still rely heavily on human supervision. This dissertation develops a control framework that (1) simplifies control design for electrified HDMMs through a generic modular approach that is energy-source independent and supports future modifications, and (2) defines hierarchical control policies that partially integrate AI while guaranteeing safety-defined performance and stability. Five research questions align with three lines of investigation: a generic robust control strategy for multi-body HDMMs with strong stability across actuation types and energy sources; control solutions that keep strict performance under uncertainty and faults while balancing robustness and responsiveness; and methods to interpret and trust black-box learning strategies so they can be integrated stably and verified against international safety standards. The framework is validated in three case studies spanning different actuators and conditions, covering heavy-duty mobile robots and robotic manipulators. Results appear in five peer-reviewed publications and one unpublished manuscript, advancing nonlinear control and robotics and supporting both transitions.
>
---
#### [new 031] Towards the Automation in the Space Station: Feasibility Study and Ground Tests of a Multi-Limbed Intra-Vehicular Robot
- **分类: cs.RO**

- **简介: 该论文属于空间站自动化任务，旨在解决 astronauts 重复性物流工作问题。通过仿真与原型测试，研究多臂机器人自主操作的可行性。**

- **链接: [https://arxiv.org/pdf/2512.23153v1](https://arxiv.org/pdf/2512.23153v1)**

> **作者:** Seiko Piotr Yamaguchi; Kentaro Uno; Yasumaru Fujii; Masazumi Imai; Kazuki Takada; Taku Okawara; Kazuya Yoshida
>
> **备注:** Author's version of a manuscript accepted at the 2025 IEEE/SICE International Symposium on System Integration (SII). (c) IEEE. The final published version is available at https://doi.org/10.1109/SII59315.2025.10870890
>
> **摘要:** This paper presents a feasibility study, including simulations and prototype tests, on the autonomous operation of a multi-limbed intra-vehicular robot (mobile manipulator), shortly MLIVR, designed to assist astronauts with logistical tasks on the International Space Station (ISS). Astronauts spend significant time on tasks such as preparation, close-out, and the collection and transportation of goods, reducing the time available for critical mission activities. Our study explores the potential for a mobile manipulator to support these operations, emphasizing the need for autonomous functionality to minimize crew and ground operator effort while enabling real-time task execution. We focused on the robot's transportation capabilities, simulating its motion planning in 3D space. The actual motion execution was tested with a prototype on a 2D table to mimic a microgravity environment. The results demonstrate the feasibility of performing these tasks with minimal human intervention, offering a promising solution to enhance operational efficiency on the ISS.
>
---
#### [new 032] The N-5 Scaling Law: Topological Dimensionality Reduction in the Optimal Design of Fully-actuated Multirotors
- **分类: cs.RO; math.GT; math.OC**

- **简介: 该论文研究多旋翼飞行器的最优设计问题，通过拓扑分析揭示其解空间结构，提出N-5 Scaling Law，解决几何优化与控制性能的关系。**

- **链接: [https://arxiv.org/pdf/2512.23619v1](https://arxiv.org/pdf/2512.23619v1)**

> **作者:** Antonio Franchi
>
> **摘要:** The geometric design of fully-actuated and omnidirectional N-rotor aerial vehicles is conventionally formulated as a parametric optimization problem, seeking a single optimal set of N orientations within a fixed architectural family. This work departs from that paradigm to investigate the intrinsic topological structure of the optimization landscape itself. We formulate the design problem on the product manifold of Projective Lines \RP^2^N, fixing the rotor positions to the vertices of polyhedral chassis while varying their lines of action. By minimizing a coordinate-invariant Log-Volume isotropy metric, we reveal that the topology of the global optima is governed strictly by the symmetry of the chassis. For generic (irregular) vertex arrangements, the solutions appear as a discrete set of isolated points. However, as the chassis geometry approaches regularity, the solution space undergoes a critical phase transition, collapsing onto an N-dimensional Torus of the lines tangent at the vertexes to the circumscribing sphere of the chassis, and subsequently reducing to continuous 1-dimensional curves driven by Affine Phase Locking. We synthesize these observations into the N-5 Scaling Law: an empirical relationship holding for all examined regular planar polygons and Platonic solids (N <= 10), where the space of optimal configurations consists of K=N-5 disconnected 1D topological branches. We demonstrate that these locking patterns correspond to a sequence of admissible Star Polygons {N/q}, allowing for the exact prediction of optimal phases for arbitrary N. Crucially, this topology reveals a design redundancy that enables optimality-preserving morphing: the vehicle can continuously reconfigure along these branches while preserving optimal isotropic control authority.
>
---
#### [new 033] Robo-Dopamine: General Process Reward Modeling for High-Precision Robotic Manipulation
- **分类: cs.RO**

- **简介: 该论文属于机器人强化学习任务，旨在解决奖励函数设计难题。通过构建多视角的通用奖励模型和理论稳健的策略框架，提升机器人操作的精确性和学习效率。**

- **链接: [https://arxiv.org/pdf/2512.23703v1](https://arxiv.org/pdf/2512.23703v1)**

> **作者:** Huajie Tan; Sixiang Chen; Yijie Xu; Zixiao Wang; Yuheng Ji; Cheng Chi; Yaoxu Lyu; Zhongxia Zhao; Xiansheng Chen; Peterson Co; Shaoxuan Xie; Guocai Yao; Pengwei Wang; Zhongyuan Wang; Shanghang Zhang
>
> **备注:** 27 pages, 11 figures
>
> **摘要:** The primary obstacle for applying reinforcement learning (RL) to real-world robotics is the design of effective reward functions. While recently learning-based Process Reward Models (PRMs) are a promising direction, they are often hindered by two fundamental limitations: their reward models lack step-aware understanding and rely on single-view perception, leading to unreliable assessments of fine-grained manipulation progress; and their reward shaping procedures are theoretically unsound, often inducing a semantic trap that misguides policy optimization. To address these, we introduce Dopamine-Reward, a novel reward modeling method for learning a general-purpose, step-aware process reward model from multi-view inputs. At its core is our General Reward Model (GRM), trained on a vast 3,400+ hour dataset, which leverages Step-wise Reward Discretization for structural understanding and Multi-Perspective Reward Fusion to overcome perceptual limitations. Building upon Dopamine-Reward, we propose Dopamine-RL, a robust policy learning framework that employs a theoretically-sound Policy-Invariant Reward Shaping method, which enables the agent to leverage dense rewards for efficient self-improvement without altering the optimal policy, thereby fundamentally avoiding the semantic trap. Extensive experiments across diverse simulated and real-world tasks validate our approach. GRM achieves state-of-the-art accuracy in reward assessment, and Dopamine-RL built on GRM significantly improves policy learning efficiency. For instance, after GRM is adapted to a new task in a one-shot manner from a single expert trajectory, the resulting reward model enables Dopamine-RL to improve the policy from near-zero to 95% success with only 150 online rollouts (approximately 1 hour of real robot interaction), while retaining strong generalization across tasks. Project website: https://robo-dopamine.github.io
>
---
#### [new 034] A Human-Oriented Cooperative Driving Approach: Integrating Driving Intention, State, and Conflict
- **分类: cs.RO**

- **简介: 该论文属于人车协同驾驶任务，旨在减少人机冲突，提升驾驶性能。通过融合驾驶员意图与状态，设计轨迹规划和控制权分配策略。**

- **链接: [https://arxiv.org/pdf/2512.23220v1](https://arxiv.org/pdf/2512.23220v1)**

> **作者:** Qin Wang; Shanmin Pang; Jianwu Fang; Shengye Dong; Fuhao Liu; Jianru Xue; Chen Lv
>
> **摘要:** Human-vehicle cooperative driving serves as a vital bridge to fully autonomous driving by improving driving flexibility and gradually building driver trust and acceptance of autonomous technology. To establish more natural and effective human-vehicle interaction, we propose a Human-Oriented Cooperative Driving (HOCD) approach that primarily minimizes human-machine conflict by prioritizing driver intention and state. In implementation, we take both tactical and operational levels into account to ensure seamless human-vehicle cooperation. At the tactical level, we design an intention-aware trajectory planning method, using intention consistency cost as the core metric to evaluate the trajectory and align it with driver intention. At the operational level, we develop a control authority allocation strategy based on reinforcement learning, optimizing the policy through a designed reward function to achieve consistency between driver state and authority allocation. The results of simulation and human-in-the-loop experiments demonstrate that our proposed approach not only aligns with driver intention in trajectory planning but also ensures a reasonable authority allocation. Compared to other cooperative driving approaches, the proposed HOCD approach significantly enhances driving performance and mitigates human-machine conflict.The code is available at https://github.com/i-Qin/HOCD.
>
---
#### [new 035] Soft Robotic Technological Probe for Speculative Fashion Futures
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于人机交互领域，探讨软体机器人服装的设计与社会影响。研究提出Sumbrella，通过设计和实验，分析其在时尚与技术融合中的作用及伦理问题。**

- **链接: [https://arxiv.org/pdf/2512.23570v1](https://arxiv.org/pdf/2512.23570v1)**

> **作者:** Amy Ingold; Loong Yi Lee; Richard Suphapol Diteesawat; Ajmal Roshan; Yael Zekaria; Edith-Clare Hall; Enrico Werner; Nahian Rahman; Elaine Czech; Jonathan Rossiter
>
> **摘要:** Emerging wearable robotics demand design approaches that address not only function, but also social meaning. In response, we present Sumbrella, a soft robotic garment developed as a speculative fashion probe. We first detail the design and fabrication of the Sumbrella, including sequenced origami-inspired bistable units, fabric pneumatic actuation chambers, cable driven shape morphing mechanisms, computer vision components, and an integrated wearable system comprising a hat and bolero jacket housing power and control electronics. Through a focus group with twelve creative technologists, we then used Sumbrella as a technological probe to explore how people interpreted, interacted, and imagined future relationships with soft robotic wearables. While Sumbrella allowed our participants to engage in rich discussion around speculative futures and expressive potential, it also surfaced concerns about exploitation, surveillance, and the personal risks and societal ethics of embedding biosensing technologies in public life. We contribute to the Human-Robot Interaction (HRI) field key considerations and recommendations for designing soft robotic garments, including the potential for kinesic communication, the impact of such technologies on social dynamics, and the importance of ethical guidelines. Finally, we provide a reflection on our application of speculative design; proposing that it allows HRI researchers to not only consider functionality, but also how wearable robots influence definitions of what is considered acceptable or desirable in public settings.
>
---
#### [new 036] Interactive Robot Programming for Surface Finishing via Task-Centric Mixed Reality Interfaces
- **分类: cs.RO**

- **简介: 该论文属于机器人编程任务，旨在解决非专家用户在小批量生产中难以编程机器人进行表面处理的问题。通过交互式混合现实接口和表面分割算法，简化机器人编程流程。**

- **链接: [https://arxiv.org/pdf/2512.23616v1](https://arxiv.org/pdf/2512.23616v1)**

> **作者:** Christoph Willibald; Lugh Martensen; Thomas Eiband; Dongheui Lee
>
> **备注:** Currently under review at Intelligent Service Robotics
>
> **摘要:** Lengthy setup processes that require robotics expertise remain a major barrier to deploying robots for tasks involving high product variability and small batch sizes. As a result, collaborative robots, despite their advanced sensing and control capabilities, are rarely used for surface finishing in small-scale craft and manufacturing settings. To address this gap, we propose a novel robot programming approach that enables non-experts to intuitively program robots through interactive, task-focused workflows. For that, we developed a new surface segmentation algorithm that incorporates human input to identify and refine workpiece regions for processing. Throughout the programming process, users receive continuous visual feedback on the robot's learned model, enabling them to iteratively refine the segmentation result. Based on the segmented surface model, a robot trajectory is generated to cover the desired processing area. We evaluated multiple interaction designs across two comprehensive user studies to derive an optimal interface that significantly reduces user workload, improves usability and enables effective task programming even for users with limited practical experience.
>
---
#### [new 037] VLA-Arena: An Open-Source Framework for Benchmarking Vision-Language-Action Models
- **分类: cs.RO; cs.CV**

- **简介: 该论文提出VLA-Arena，一个用于评估视觉-语言-动作模型的基准框架。旨在解决模型能力量化与局限性分析问题，通过结构化任务设计和多维度难度评估，促进通用机器人策略研究。**

- **链接: [https://arxiv.org/pdf/2512.22539v1](https://arxiv.org/pdf/2512.22539v1)**

> **作者:** Borong Zhang; Jiahao Li; Jiachen Shen; Yishuai Cai; Yuhao Zhang; Yuanpei Chen; Juntao Dai; Jiaming Ji; Yaodong Yang
>
> **摘要:** While Vision-Language-Action models (VLAs) are rapidly advancing towards generalist robot policies, it remains difficult to quantitatively understand their limits and failure modes. To address this, we introduce a comprehensive benchmark called VLA-Arena. We propose a novel structured task design framework to quantify difficulty across three orthogonal axes: (1) Task Structure, (2) Language Command, and (3) Visual Observation. This allows us to systematically design tasks with fine-grained difficulty levels, enabling a precise measurement of model capability frontiers. For Task Structure, VLA-Arena's 170 tasks are grouped into four dimensions: Safety, Distractor, Extrapolation, and Long Horizon. Each task is designed with three difficulty levels (L0-L2), with fine-tuning performed exclusively on L0 to assess general capability. Orthogonal to this, language (W0-W4) and visual (V0-V4) perturbations can be applied to any task to enable a decoupled analysis of robustness. Our extensive evaluation of state-of-the-art VLAs reveals several critical limitations, including a strong tendency toward memorization over generalization, asymmetric robustness, a lack of consideration for safety constraints, and an inability to compose learned skills for long-horizon tasks. To foster research addressing these challenges and ensure reproducibility, we provide the complete VLA-Arena framework, including an end-to-end toolchain from task definition to automated evaluation and the VLA-Arena-S/M/L datasets for fine-tuning. Our benchmark, data, models, and leaderboard are available at https://vla-arena.github.io.
>
---
#### [new 038] Emergence of Human to Robot Transfer in Vision-Language-Action Models
- **分类: cs.RO; cs.AI**

- **简介: 该论文研究视觉-语言-动作模型的人到机器人的迁移问题，旨在解决仅用人类视频训练VLAs的困难。通过预训练提升模型泛化能力，实现有效迁移。**

- **链接: [https://arxiv.org/pdf/2512.22414v1](https://arxiv.org/pdf/2512.22414v1)**

> **作者:** Simar Kareer; Karl Pertsch; James Darpinian; Judy Hoffman; Danfei Xu; Sergey Levine; Chelsea Finn; Suraj Nair
>
> **摘要:** Vision-language-action (VLA) models can enable broad open world generalization, but require large and diverse datasets. It is appealing to consider whether some of this data can come from human videos, which cover diverse real-world situations and are easy to obtain. However, it is difficult to train VLAs with human videos alone, and establishing a mapping between humans and robots requires manual engineering and presents a major research challenge. Drawing inspiration from advances in large language models, where the ability to learn from diverse supervision emerges with scale, we ask whether a similar phenomenon holds for VLAs that incorporate human video data. We introduce a simple co-training recipe, and find that human-to-robot transfer emerges once the VLA is pre-trained on sufficient scenes, tasks, and embodiments. Our analysis suggests that this emergent capability arises because diverse pretraining produces embodiment-agnostic representations for human and robot data. We validate these findings through a series of experiments probing human to robot skill transfer and find that with sufficiently diverse robot pre-training our method can nearly double the performance on generalization settings seen only in human data.
>
---
#### [new 039] A Kalman Filter-Based Disturbance Observer for Steer-by-Wire Systems
- **分类: cs.RO**

- **简介: 该论文属于控制任务，旨在解决Steer-by-Wire系统中高频率扰动的估计问题。通过设计基于卡尔曼滤波的扰动观测器，利用电机状态测量估计驾驶员扭矩，提升系统性能。**

- **链接: [https://arxiv.org/pdf/2512.23593v1](https://arxiv.org/pdf/2512.23593v1)**

> **作者:** Nikolai Beving; Jonas Marxen; Steffen Mueller; Johannes Betz
>
> **摘要:** Steer-by-Wire systems replace mechanical linkages, which provide benefits like weight reduction, design flexibility, and compatibility with autonomous driving. However, they are susceptible to high-frequency disturbances from unintentional driver torque, known as driver impedance, which can degrade steering performance. Existing approaches either rely on direct torque sensors, which are costly and impractical, or lack the temporal resolution to capture rapid, high-frequency driver-induced disturbances. We address this limitation by designing a Kalman filter-based disturbance observer that estimates high-frequency driver torque using only motor state measurements. We model the drivers passive torque as an extended state using a PT1-lag approximation and integrate it into both linear and nonlinear Steer-by-Wire system models. In this paper, we present the design, implementation and simulation of this disturbance observer with an evaluation of different Kalman filter variants. Our findings indicate that the proposed disturbance observer accurately reconstructs driver-induced disturbances with only minimal delay 14ms. We show that a nonlinear extended Kalman Filter outperforms its linear counterpart in handling frictional nonlinearities, improving estimation during transitions from static to dynamic friction. Given the study's methodology, it was unavoidable to rely on simulation-based validation rather than real-world experimentation. Further studies are needed to investigate the robustness of the observers under real-world driving conditions.
>
---
#### [new 040] Two-Robot Computational Landscape: A Complete Characterization of Model Power in Minimal Mobile Robot Systems
- **分类: cs.RO; cs.DC**

- **简介: 该论文研究两机器人系统的计算能力，解决其模型间关系问题。通过分析不同模型，揭示其计算格局，完成对两机器人系统的完整表征。**

- **链接: [https://arxiv.org/pdf/2512.22770v1](https://arxiv.org/pdf/2512.22770v1)**

> **作者:** Naoki Kitamura; Yuichi Sudo; Koichi Wada
>
> **备注:** 23 pages, 3 figures
>
> **摘要:** The computational power of autonomous mobile robots under the Look-Compute-Move (LCM) model has been widely studied through an extensive hierarchy of robot models defined by the presence of memory, communication, and synchrony assumptions. While the general n-robot landscape has been largely established, the exact structure for two robots has remained unresolved. This paper presents the first complete characterization of the computational power of two autonomous robots across all major models, namely OBLOT, FSTA, FCOM, and LUMI, under the full spectrum of schedulers (FSYNCH, SSYNCH, ASYNCH, and their atomic variants). Our results reveal a landscape that fundamentally differs from the general case. Most notably, we prove that FSTA^F and LUMI^F coincide under full synchrony, a surprising collapse indicating that perfect synchrony can substitute both memory and communication when only two robots exist. We also show that FSTA and FCOM are orthogonal: there exists a problem solvable in the weakest communication model but impossible even in the strongest finite-state model, completing the bidirectional incomparability. All equivalence and separation results are derived through a novel simulation-free method, providing a unified and constructive view of the two-robot hierarchy. This yields the first complete and exact computational landscape for two robots, highlighting the intrinsic challenges of coordination at the minimal scale.
>
---
#### [new 041] Beyond URDF: The Universal Robot Description Directory for Shared, Extensible, and Standardized Robot Models
- **分类: cs.RO**

- **简介: 该论文提出URDD，解决机器人模型描述不统一问题，通过结构化数据提升信息共享与标准制定。**

- **链接: [https://arxiv.org/pdf/2512.23135v1](https://arxiv.org/pdf/2512.23135v1)**

> **作者:** Roshan Klein-Seetharaman; Daniel Rakita
>
> **摘要:** Robots are typically described in software by specification files (e.g., URDF, SDF, MJCF, USD) that encode only basic kinematic, dynamic, and geometric information. As a result, downstream applications such as simulation, planning, and control must repeatedly re-derive richer data, leading to redundant computations, fragmented implementations, and limited standardization. In this work, we introduce the Universal Robot Description Directory (URDD), a modular representation that organizes derived robot information into structured, easy-to-parse JSON and YAML modules. Our open-source toolkit automatically generates URDDs from URDFs, with a Rust implementation supporting Bevy-based visualization. Additionally, we provide a JavaScript/Three.js viewer for web-based inspection of URDDs. Experiments on multiple robot platforms show that URDDs can be generated efficiently, encapsulate substantially richer information than standard specification files, and directly enable the construction of core robotics subroutines. URDD provides a unified, extensible resource for reducing redundancy and establishing shared standards across robotics frameworks. We conclude with a discussion on the limitations and implications of our work.
>
---
#### [new 042] Clutter-Resistant Vision-Language-Action Models through Object-Centric and Geometry Grounding
- **分类: cs.RO**

- **简介: 该论文属于机器人操作任务，旨在解决VLA模型在复杂环境中的感知与控制混淆问题。通过引入基于物体和几何的感知模块，提升模型的鲁棒性与泛化能力。**

- **链接: [https://arxiv.org/pdf/2512.22519v1](https://arxiv.org/pdf/2512.22519v1)**

> **作者:** Khoa Vo; Taisei Hanyu; Yuki Ikebe; Trong Thang Pham; Nhat Chung; Minh Nhat Vu; Duy Nguyen Ho Minh; Anh Nguyen; Anthony Gunderman; Chase Rainwater; Ngan Le
>
> **备注:** Under review. Project website: https://uark-aicv.github.io/OBEYED_VLA
>
> **摘要:** Recent Vision-Language-Action (VLA) models have made impressive progress toward general-purpose robotic manipulation by post-training large Vision-Language Models (VLMs) for action prediction. Yet most VLAs entangle perception and control in a monolithic pipeline optimized purely for action, which can erode language-conditioned grounding. In our real-world tabletop tests, policies over-grasp when the target is absent, are distracted by clutter, and overfit to background appearance. To address these issues, we propose OBEYED-VLA (OBject-centric and gEometrY groundED VLA), a framework that explicitly disentangles perceptual grounding from action reasoning. Instead of operating directly on raw RGB, OBEYED-VLA augments VLAs with a perception module that grounds multi-view inputs into task-conditioned, object-centric, and geometry-aware observations. This module includes a VLM-based object-centric grounding stage that selects task-relevant object regions across camera views, along with a complementary geometric grounding stage that emphasizes the 3D structure of these objects over their appearance. The resulting grounded views are then fed to a pretrained VLA policy, which we fine-tune exclusively on single-object demonstrations collected without environmental clutter or non-target objects. On a real-world UR10e tabletop setup, OBEYED-VLA substantially improves robustness over strong VLA baselines across four challenging regimes and multiple difficulty levels: distractor objects, absent-target rejection, background appearance changes, and cluttered manipulation of unseen objects. Ablation studies confirm that both semantic grounding and geometry-aware grounding are critical to these gains. Overall, the results indicate that making perception an explicit, object-centric component is an effective way to strengthen and generalize VLA-based robotic manipulation.
>
---
#### [new 043] Modeling of UAV Tether Aerodynamics for Real-Time Simulation
- **分类: cs.RO**

- **简介: 论文研究无人机系绳动力学建模，解决系绳力实时计算问题。提出两种方法：解析法和数值法，用于提高仿真效率与精度，适用于控制与规划任务。**

- **链接: [https://arxiv.org/pdf/2512.22588v1](https://arxiv.org/pdf/2512.22588v1)**

> **作者:** Max Beffert; Andreas Zell
>
> **摘要:** One of the main limitations of multirotor UAVs is their short flight time due to battery constraints. A practical solution for continuous operation is to power the drone from the ground via a tether. While this approach has been demonstrated for stationary systems, scenarios with a fast-moving base vehicle or strong wind conditions require modeling the tether forces, including aerodynamic effects. In this work, we propose two complementary approaches for real-time quasi-static tether modeling with aerodynamics. The first is an analytical method based on catenary theory with a uniform drag assumption, achieving very fast solve times below 1ms. The second is a numerical method that discretizes the tether into segments and lumped masses, solving the equilibrium equations using CasADi and IPOPT. By leveraging initialization strategies, such as warm starting and analytical initialization, real-time performance was achieved with a solve time of 5ms, while allowing for flexible force formulations. Both approaches were validated in real-world tests using a load cell to measure the tether force. The results show that the analytical method provides sufficient accuracy for most tethered UAV applications with minimal computational cost, while the numerical method offers higher flexibility and physical accuracy when required. These approaches form a lightweight and extensible framework for real-time tether simulation, applicable to both offline optimization and online tasks such as simulation, control, and trajectory planning.
>
---
#### [new 044] Breaking Symmetry-Induced Degeneracy in Multi-Agent Ergodic Coverage via Stochastic Spectral Control
- **分类: eess.SY; cs.RO**

- **简介: 该论文属于多智能体覆盖任务，旨在解决对称性导致的梯度消失问题。通过引入随机扰动和收缩项，确保智能体有效逃离对称区域并保持轨迹稳定。**

- **链接: [https://arxiv.org/pdf/2512.23158v1](https://arxiv.org/pdf/2512.23158v1)**

> **作者:** Kooktae Lee; Julian Martinez
>
> **摘要:** Multi-agent ergodic coverage via Spectral Multiscale Coverage (SMC) provides a principled framework for driving a team of agents so that their collective time-averaged trajectories match a prescribed spatial distribution. While classical SMC has demonstrated empirical success, it can suffer from gradient cancellation, particularly when agents are initialized near symmetry points of the target distribution, leading to undesirable behaviors such as stalling or motion constrained along symmetry axes. In this work, we rigorously characterize the initial conditions and symmetry-induced invariant manifolds that give rise to such directional degeneracy in first-order agent dynamics. To address this, we introduce a stochastic perturbation combined with a contraction term and prove that the resulting dynamics ensure almost-sure escape from zero-gradient manifolds while maintaining mean-square boundedness of agent trajectories. Simulations on symmetric multi-modal reference distributions demonstrate that the proposed stochastic SMC effectively mitigates transient stalling and axis-constrained motion, while ensuring that all agent trajectories remain bounded within the domain.
>
---
#### [new 045] Assessing behaviour coverage in a multi-agent system simulation for autonomous vehicle testing
- **分类: cs.MA; cs.LG; cs.RO**

- **简介: 该论文属于自动驾驶测试任务，旨在解决仿真环境中行为覆盖不足的问题。通过分析驾驶场景和代理交互，提出改进的行人代理模型以提升测试有效性。**

- **链接: [https://arxiv.org/pdf/2512.23445v1](https://arxiv.org/pdf/2512.23445v1)**

> **作者:** Manuel Franco-Vivo
>
> **摘要:** As autonomous vehicle technology advances, ensuring the safety and reliability of these systems becomes paramount. Consequently, comprehensive testing methodologies are essential to evaluate the performance of autonomous vehicles in diverse and complex real-world scenarios. This study focuses on the behaviour coverage analysis of a multi-agent system simulation designed for autonomous vehicle testing, and provides a systematic approach to measure and assess behaviour coverage within the simulation environment. By defining a set of driving scenarios, and agent interactions, we evaluate the extent to which the simulation encompasses a broad range of behaviours relevant to autonomous driving. Our findings highlight the importance of behaviour coverage in validating the effectiveness and robustness of autonomous vehicle systems. Through the analysis of behaviour coverage metrics and coverage-based testing, we identify key areas for improvement and optimization in the simulation framework. Thus, a Model Predictive Control (MPC) pedestrian agent is proposed, where its objective function is formulated to encourage \textit{interesting} tests while promoting a more realistic behaviour than other previously studied pedestrian agents. This research contributes to advancing the field of autonomous vehicle testing by providing insights into the comprehensive evaluation of system behaviour in simulated environments. The results offer valuable implications for enhancing the safety, reliability, and performance of autonomous vehicles through rigorous testing methodologies.
>
---
#### [new 046] Pose-Guided Residual Refinement for Interpretable Text-to-Motion Generation and Editing
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于文本到动作生成与编辑任务，解决Pose-code框架在时间动态和细节上的不足。提出PGR$^2$M，结合姿态码与残差码，提升生成质量和可编辑性。**

- **链接: [https://arxiv.org/pdf/2512.22464v1](https://arxiv.org/pdf/2512.22464v1)**

> **作者:** Sukhyun Jeong; Yong-Hoon Choi
>
> **摘要:** Text-based 3D motion generation aims to automatically synthesize diverse motions from natural-language descriptions to extend user creativity, whereas motion editing modifies an existing motion sequence in response to text while preserving its overall structure. Pose-code-based frameworks such as CoMo map quantifiable pose attributes into discrete pose codes that support interpretable motion control, but their frame-wise representation struggles to capture subtle temporal dynamics and high-frequency details, often degrading reconstruction fidelity and local controllability. To address this limitation, we introduce pose-guided residual refinement for motion (PGR$^2$M), a hybrid representation that augments interpretable pose codes with residual codes learned via residual vector quantization (RVQ). A pose-guided RVQ tokenizer decomposes motion into pose latents that encode coarse global structure and residual latents that model fine-grained temporal variations. Residual dropout further discourages over-reliance on residuals, preserving the semantic alignment and editability of the pose codes. On top of this tokenizer, a base Transformer autoregressively predicts pose codes from text, and a refine Transformer predicts residual codes conditioned on text, pose codes, and quantization stage. Experiments on HumanML3D and KIT-ML show that PGR$^2$M improves Fréchet inception distance and reconstruction metrics for both generation and editing compared with CoMo and recent diffusion- and tokenization-based baselines, while user studies confirm that it enables intuitive, structure-preserving motion edits.
>
---
#### [new 047] On Extending Semantic Abstraction for Efficient Search of Hidden Objects
- **分类: cs.CV; cs.AI; cs.RO**

- **简介: 该论文属于3D目标定位任务，旨在解决隐藏物体的高效搜索问题。通过扩展语义抽象方法，提升机器人查找被遮挡物体的效率。**

- **链接: [https://arxiv.org/pdf/2512.22220v1](https://arxiv.org/pdf/2512.22220v1)**

> **作者:** Tasha Pais; Nikhilesh Belulkar
>
> **摘要:** Semantic Abstraction's key observation is that 2D VLMs' relevancy activations roughly correspond to their confidence of whether and where an object is in the scene. Thus, relevancy maps are treated as "abstract object" representations. We use this framework for learning 3D localization and completion for the exclusive domain of hidden objects, defined as objects that cannot be directly identified by a VLM because they are at least partially occluded. This process of localizing hidden objects is a form of unstructured search that can be performed more efficiently using historical data of where an object is frequently placed. Our model can accurately identify the complete 3D location of a hidden object on the first try significantly faster than a naive random search. These extensions to semantic abstraction hope to provide household robots with the skills necessary to save time and effort when looking for lost objects.
>
---
#### [new 048] MUSON: A Reasoning-oriented Multimodal Dataset for Socially Compliant Navigation in Urban Environments
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出MUSON数据集，用于解决城市环境中社交合规导航任务中的安全行为学习问题。通过结构化标注和平衡动作空间，提升模型决策准确性。**

- **链接: [https://arxiv.org/pdf/2512.22867v1](https://arxiv.org/pdf/2512.22867v1)**

> **作者:** Zhuonan Liu; Xinyu Zhang; Zishuo Wang; Tomohito Kawabata; Xuesu Xiao; Ling Xiao
>
> **摘要:** Socially compliant navigation requires structured reasoning over dynamic pedestrians and physical constraints to ensure safe and interpretable decisions. However, existing social navigation datasets often lack explicit reasoning supervision and exhibit highly long-tailed action distributions, limiting models' ability to learn safety-critical behaviors. To address these issues, we introduce MUSON, a multimodal dataset for short-horizon social navigation collected across diverse indoor and outdoor campus scenes. MUSON adopts a structured five-step Chain-of-Thought annotation consisting of perception, prediction, reasoning, action, and explanation, with explicit modeling of static physical constraints and a rationally balanced discrete action space. Compared to SNEI, MUSON provides consistent reasoning, action, and explanation. Benchmarking multiple state-of-the-art Small Vision Language Models on MUSON shows that Qwen2.5-VL-3B achieves the highest decision accuracy of 0.8625, demonstrating that MUSON serves as an effective and reusable benchmark for socially compliant navigation. The dataset is publicly available at https://huggingface.co/datasets/MARSLab/MUSON
>
---
#### [new 049] HLS4PC: A Parametrizable Framework For Accelerating Point-Based 3D Point Cloud Models on FPGA
- **分类: cs.DC; cs.AI; cs.AR; cs.RO**

- **简介: 该论文针对3D点云分类任务，解决GPU性能不足问题，提出HLS4PC框架，通过FPGA加速优化模型，提升实时性。**

- **链接: [https://arxiv.org/pdf/2512.22139v1](https://arxiv.org/pdf/2512.22139v1)**

> **作者:** Amur Saqib Pal; Muhammad Mohsin Ghaffar; Faisal Shafait; Christian Weis; Norbert Wehn
>
> **备注:** Accepted for publication by 25th International Conference on Embedded Computer Systems: Architectures, Modeling and Simulation (SAMOS 2025)
>
> **摘要:** Point-based 3D point cloud models employ computation and memory intensive mapping functions alongside NN layers for classification/segmentation, and are executed on server-grade GPUs. The sparse, and unstructured nature of 3D point cloud data leads to high memory and computational demand, hindering real-time performance in safety critical applications due to GPU under-utilization. To address this challenge, we present HLS4PC, a parameterizable HLS framework for FPGA acceleration. Our approach leverages FPGA parallelization and algorithmic optimizations to enable efficient fixed-point implementations of both mapping and NN functions. We explore several hardware-aware compression techniques on a state-of-the-art PointMLP-Elite model, including replacing FPS with URS, parameter quantization, layer fusion, and input-points pruning, yielding PointMLP-Lite, a 4x less complex variant with only 2% accuracy drop on ModelNet40. Secondly, we demonstrate that the FPGA acceleration of the PointMLP-Lite results in 3.56x higher throughput than previous works. Furthermore, our implementation achieves 2.3x and 22x higher throughput compared to the GPU and CPU implementations, respectively.
>
---
#### [new 050] Evaluating an Adaptive Multispectral Turret System for Autonomous Tracking Across Variable Illumination Conditions
- **分类: cs.CV; cs.LG; cs.RO**

- **简介: 该论文属于目标检测任务，旨在解决光照变化下自主机器人视觉性能下降的问题。通过融合RGB与LWIR图像，训练多模型并动态选择最佳方案，提升检测准确率。**

- **链接: [https://arxiv.org/pdf/2512.22263v1](https://arxiv.org/pdf/2512.22263v1)**

> **作者:** Aahan Sachdeva; Dhanvinkumar Ganeshkumar; James E. Gallagher; Tyler Treat; Edward J. Oughton
>
> **摘要:** Autonomous robotic platforms are playing a growing role across the emergency services sector, supporting missions such as search and rescue operations in disaster zones and reconnaissance. However, traditional red-green-blue (RGB) detection pipelines struggle in low-light environments, and thermal-based systems lack color and texture information. To overcome these limitations, we present an adaptive framework that fuses RGB and long-wave infrared (LWIR) video streams at multiple fusion ratios and dynamically selects the optimal detection model for each illumination condition. We trained 33 You Only Look Once (YOLO) models on over 22,000 annotated images spanning three light levels: no-light (<10 lux), dim-light (10-1000 lux), and full-light (>1000 lux). To integrate both modalities, fusion was performed by blending aligned RGB and LWIR frames at eleven ratios, from full RGB (100/0) to full LWIR (0/100) in 10% increments. Evaluation showed that the best full-light model (80/20 RGB-LWIR) and dim-light model (90/10 fusion) achieved 92.8% and 92.0% mean confidence; both significantly outperformed the YOLOv5 nano (YOLOv5n) and YOLOv11 nano (YOLOv11n) baselines. Under no-light conditions, the top 40/60 fusion reached 71.0%, exceeding baselines though not statistically significant. Adaptive RGB-LWIR fusion improved detection confidence and reliability across all illumination conditions, enhancing autonomous robotic vision performance.
>
---
## 更新

#### [replaced 001] Symphony: A Heuristic Normalized Calibrated Advantage Actor and Critic Algorithm in application for Humanoid Robots
- **分类: cs.RO; cs.NE**

- **简介: 该论文提出Symphony算法，用于解决人形机器人训练中的样本效率、动作安全和稳定性问题。通过引入Swaddling正则化和Fading Replay Buffer提升训练安全性与效果。**

- **链接: [https://arxiv.org/pdf/2512.10477v4](https://arxiv.org/pdf/2512.10477v4)**

> **作者:** Timur Ishuov; Michele Folgheraiter; Madi Nurmanov; Goncalo Gordo; Richárd Farkas; József Dombi
>
> **备注:** https://github.com/SuspensionRailway/symphony
>
> **摘要:** In our work we not explicitly hint that it is a misconception to think that humans learn fast. Learning process takes time. Babies start learning to move in the restricted liquid area called placenta. Children often are limited by underdeveloped body. Even adults are not allowed to participate in complex competitions right away. However, with robots, when learning from scratch, we often don't have the privilege of waiting for dozen millions of steps. "Swaddling" regularization is responsible for restraining an agent in rapid but unstable development penalizing action strength in a specific way not affecting actions directly. The Symphony, Transitional-policy Deterministic Actor and Critic algorithm, is a concise combination of different ideas for possibility of training humanoid robots from scratch with Sample Efficiency, Sample Proximity and Safety of Actions in mind. It is no secret that continuous increase in Gaussian noise without appropriate smoothing is harmful for motors and gearboxes. Compared to Stochastic algorithms, we set a limited parametric noise and promote a reduced strength of actions, safely increasing entropy, since the actions are kind of immersed in weaker noise. When actions require more extreme values, actions rise above the weak noise. Training becomes empirically much safer for both the environment around and the robot's mechanisms. We use Fading Replay Buffer: using a fixed formula containing the hyperbolic tangent, we adjust the batch sampling probability: the memory contains a recent memory and a long-term memory trail. Fading Replay Buffer allows us to use Temporal Advantage when we improve the current Critic Network prediction compared to the exponential moving average. Temporal Advantage allows us to update Actor and Critic in one pass, as well as combine Actor and Critic in one Object and implement their Losses in one line.
>
---
#### [replaced 002] Developing a Fundamental Diagram for Urban Air Mobility Based on Physical Experiments
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于交通流建模任务，旨在解决UAM交通特性不明确的问题。通过理论分析与物理实验，构建UAM基本图，验证其适用性并提供实际应用见解。**

- **链接: [https://arxiv.org/pdf/2512.21425v2](https://arxiv.org/pdf/2512.21425v2)**

> **作者:** Hang Zhou; Yuhui Zhai; Shiyu Shen; Yanfeng Ouyang; Xiaowei Shi; Xiaopeng Li
>
> **摘要:** Urban Air Mobility (UAM) is an emerging application of unmanned aerial vehicles (UAVs) that promises to reduce travel time and alleviate congestion in urban transportation systems. As drone density increases, UAM operations are expected to experience congestion similar to that in ground traffic. However, the fundamental characteristics of UAM traffic flow, particularly under real-world operating conditions, remain poorly understood. This study proposes a general framework for constructing the fundamental diagram (FD) of UAM traffic by integrating theoretical analysis with physical experiments. To the best of our knowledge, this is the first study to derive a UAM FD using real-world physical test data. On the theoretical side, we design two drone control laws for collision avoidance and develop simulation-based traffic generation methods to produce diverse UAM traffic scenarios. Based on Edie's definition, traffic flow theory is then applied to construct the FD and characterize the macroscopic properties of UAM traffic. To account for real-world disturbances and modeling uncertainties, we further conduct physical experiments on a reduced-scale testbed using Bitcraze Crazyflie drones. Both simulation and physical test trajectory data are collected and organized into the UAMTra2Flow dataset, which is analyzed using the proposed framework. Preliminary results indicate that classical FD structures for ground transportation are also applicable to UAM systems. Notably, FD curves obtained from physical experiments exhibit deviations from simulation-based results, highlighting the importance of experimental validation. Finally, results from the reduced-scale testbed are scaled to realistic operating conditions to provide practical insights for future UAM traffic systems. The dataset and code for this paper are publicly available at https://github.com/CATS-Lab/UAM-FD.
>
---
#### [replaced 003] Contingency Model-based Control (CMC) for Communicationless Cooperative Collision Avoidance in Robot Swarms
- **分类: math.OC; cs.RO; eess.SY**

- **简介: 该论文属于机器人协同避撞任务，解决无通信环境下多机器人碰撞问题。提出CMC方法，基于预设规则实现安全避撞，确保系统稳定与可扩展性。**

- **链接: [https://arxiv.org/pdf/2512.20391v2](https://arxiv.org/pdf/2512.20391v2)**

> **作者:** Georg Schildbach
>
> **摘要:** Cooperative collision avoidance between robots in swarm operations remains an open challenge. Assuming a decentralized architecture, each robot is responsible for making its own control decisions, including motion planning. To this end, most existing approaches mostly rely some form of (wireless) communication between the agents of the swarm. In reality, however, communication is brittle. It may be affected by latency, further delays and packet losses, transmission faults, and is subject to adversarial attacks, such as jamming or spoofing. This paper proposes Contingency Model-based Control (CMC) as a communicationless alternative. It follows the implicit cooperation paradigm, under which the design of the robots is based on consensual (offline) rules, similar to traffic rules. They include the definition of a contingency trajectory for each robot, and a method for construction of mutual collision avoidance constraints. The setup is shown to guarantee the recursive feasibility and collision avoidance between all swarm members in closed-loop operation. Moreover, CMC naturally satisfies the Plug \& Play paradigm, i.e., for new robots entering the swarm. Two numerical examples demonstrate that the collision avoidance guarantee is intact and that the robot swarm operates smoothly under the CMC regime.
>
---
#### [replaced 004] Relative Localization System Design for SnailBot: A Modular Self-reconfigurable Robot
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人定位任务，解决模块化机器人SnailBot的相对定位问题，通过融合视觉与惯性数据实现精准实时定位。**

- **链接: [https://arxiv.org/pdf/2512.21226v2](https://arxiv.org/pdf/2512.21226v2)**

> **作者:** Shuhan Zhang; Tin Lun Lam
>
> **备注:** The design presented in the article does not correspond to the actual situation
>
> **摘要:** This paper presents the design and implementation of a relative localization system for SnailBot, a modular self reconfigurable robot. The system integrates ArUco marker recognition, optical flow analysis, and IMU data processing into a unified fusion framework, enabling robust and accurate relative positioning for collaborative robotic tasks. Experimental validation demonstrates the effectiveness of the system in realtime operation, with a rule based fusion strategy ensuring reliability across dynamic scenarios. The results highlight the potential for scalable deployment in modular robotic systems.
>
---
#### [replaced 005] ReSemAct: Advancing Fine-Grained Robotic Manipulation via Semantic Structuring and Affordance Refinement
- **分类: cs.RO; cs.AI; cs.CV; cs.HC; cs.LG**

- **简介: 该论文提出ReSemAct，解决细粒度机器人操作中语义与功能匹配的问题。通过语义结构和可操作性优化，提升动态环境下的操作精度与鲁棒性。**

- **链接: [https://arxiv.org/pdf/2507.18262v4](https://arxiv.org/pdf/2507.18262v4)**

> **作者:** Chenyu Su; Weiwei Shang; Chen Qian; Fei Zhang; Shuang Cong
>
> **备注:** Code and videos: https://github.com/scy-v/ReSemAct and https://resemact.github.io
>
> **摘要:** Fine-grained robotic manipulation requires grounding natural language into appropriate affordance targets. However, most existing methods driven by foundation models often compress rich semantics into oversimplified affordances, preventing exploitation of implicit semantic information. To address these challenges, we present ReSemAct, a novel unified manipulation framework that introduces Semantic Structuring and Affordance Refinement (SSAR), powered by the automated synergistic reasoning between Multimodal Large Language Models (MLLMs) and Vision Foundation Models (VFMs). Specifically, the Semantic Structuring module derives a unified semantic affordance description from natural language and RGB observations, organizing affordance regions, implicit functional intent, and coarse affordance anchors into a structured representation for downstream refinement. Building upon this specification, the Affordance Refinement strategy instantiates two complementary flows that separately specialize geometry and position, yielding fine-grained affordance targets. These refined targets are then encoded as real-time joint-space optimization objectives, enabling reactive and robust manipulation in dynamic environments. Extensive simulation and real-world experiments are conducted in semantically rich household and sparse chemical lab environments. The results demonstrate that ReSemAct performs diverse tasks under zero-shot conditions, showcasing the robustness of SSAR with foundation models in fine-grained manipulation. Code and videos at https://github.com/scy-v/ReSemAct and https://resemact.github.io.
>
---
#### [replaced 006] InDRiVE: Reward-Free World-Model Pretraining for Autonomous Driving via Latent Disagreement
- **分类: cs.RO**

- **简介: 该论文属于自主驾驶领域，解决MBRL依赖任务奖励的问题。提出InDRiVE，通过潜在分歧进行无奖励预训练，提升模型泛化与适应能力。**

- **链接: [https://arxiv.org/pdf/2512.18850v2](https://arxiv.org/pdf/2512.18850v2)**

> **作者:** Feeza Khan Khanzada; Jaerock Kwon
>
> **摘要:** Model-based reinforcement learning (MBRL) can reduce interaction cost for autonomous driving by learning a predictive world model, but it typically still depends on task-specific rewards that are difficult to design and often brittle under distribution shift. This paper presents InDRiVE, a DreamerV3-style MBRL agent that performs reward-free pretraining in CARLA using only intrinsic motivation derived from latent ensemble disagreement. Disagreement acts as a proxy for epistemic uncertainty and drives the agent toward under-explored driving situations, while an imagination-based actor-critic learns a planner-free exploration policy directly from the learned world model. After intrinsic pretraining, we evaluate zero-shot transfer by freezing all parameters and deploying the pretrained exploration policy in unseen towns and routes. We then study few-shot adaptation by training a task policy with limited extrinsic feedback for downstream objectives (lane following and collision avoidance). Experiments in CARLA across towns, routes, and traffic densities show that disagreement-based pretraining yields stronger zero-shot robustness and robust few-shot collision avoidance under town shift and matched interaction budgets, supporting the use of intrinsic disagreement as a practical reward-free pretraining signal for reusable driving world models.
>
---
#### [replaced 007] EMMA: Scaling Mobile Manipulation via Egocentric Human Data
- **分类: cs.RO**

- **简介: 该论文提出EMMA框架，解决移动操作模仿学习中依赖昂贵遥控的问题，通过人类数据训练机器人，实现高效、可扩展的现实环境机器人学习。**

- **链接: [https://arxiv.org/pdf/2509.04443v3](https://arxiv.org/pdf/2509.04443v3)**

> **作者:** Lawrence Y. Zhu; Pranav Kuppili; Ryan Punamiya; Patcharapong Aphiwetsa; Dhruv Patel; Simar Kareer; Sehoon Ha; Danfei Xu
>
> **摘要:** Scaling mobile manipulation imitation learning is bottlenecked by expensive mobile robot teleoperation. We present Egocentric Mobile MAnipulation (EMMA), an end-to-end framework training mobile manipulation policies from human mobile manipulation data with static robot data, sidestepping mobile teleoperation. To accomplish this, we co-train human full-body motion data with static robot data. In our experiments across three real-world tasks, EMMA demonstrates comparable performance to baselines trained on teleoperated mobile robot data (Mobile ALOHA), achieving higher or equivalent task performance in full task success. We find that EMMA is able to generalize to new spatial configurations and scenes, and we observe positive performance scaling as we increase the hours of human data, opening new avenues for scalable robotic learning in real-world environments. Details of this project can be found at https://ego-moma.github.io/.
>
---
#### [replaced 008] RefAV: Towards Planning-Centric Scenario Mining
- **分类: cs.CV; cs.CL; cs.RO**

- **简介: 该论文属于场景挖掘任务，旨在从驾驶日志中识别安全关键场景。通过引入RefAV数据集和视觉语言模型，解决传统方法效率低、错误率高的问题。**

- **链接: [https://arxiv.org/pdf/2505.20981v3](https://arxiv.org/pdf/2505.20981v3)**

> **作者:** Cainan Davidson; Deva Ramanan; Neehar Peri
>
> **备注:** Project Page: https://cainand.github.io/RefAV/
>
> **摘要:** Autonomous Vehicles (AVs) collect and pseudo-label terabytes of multi-modal data localized to HD maps during normal fleet testing. However, identifying interesting and safety-critical scenarios from uncurated driving logs remains a significant challenge. Traditional scenario mining techniques are error-prone and prohibitively time-consuming, often relying on hand-crafted structured queries. In this work, we revisit spatio-temporal scenario mining through the lens of recent vision-language models (VLMs) to detect whether a described scenario occurs in a driving log and, if so, precisely localize it in both time and space. To address this problem, we introduce RefAV, a large-scale dataset of 10,000 diverse natural language queries that describe complex multi-agent interactions relevant to motion planning derived from 1000 driving logs in the Argoverse 2 Sensor dataset. We evaluate several referential multi-object trackers and present an empirical analysis of our baselines. Notably, we find that naively repurposing off-the-shelf VLMs yields poor performance, suggesting that scenario mining presents unique challenges. Lastly, we discuss our recently held competition and share insights from the community. Our code and dataset are available at https://github.com/CainanD/RefAV/ and https://argoverse.github.io/user-guide/tasks/scenario_mining.html
>
---
#### [replaced 009] On The Computational Complexity of Minimum Aerial Photographs for Planar Region Coverage
- **分类: cs.RO; cs.CG**

- **简介: 该论文研究在平面区域覆盖中使用最少航拍照片的计算复杂性，解决如何高效覆盖目标区域的问题，分析了正方形和圆形覆盖的近似难度并提出近似算法。**

- **链接: [https://arxiv.org/pdf/2512.18268v2](https://arxiv.org/pdf/2512.18268v2)**

> **作者:** Si Wei Feng
>
> **摘要:** With the popularity of drone technologies, aerial photography has become prevalent in many daily scenarios such as environment monitoring, structure inspection, law enforcement etc. A central challenge in this domain is the efficient coverage of a target area with photographs that can entirely capture the region, while respecting constraints such as the image resolution, and limited number of pictures that can be taken. This work investigates the computational complexity of covering a simple planar polygon using squares and circles. Specifically, it shows inapproximability gaps of $1.165$ (for squares) and $1.25$ (for restricted square centers) and develops a $2.828$-optimal approximation algorithm, demonstrating that these problems are computationally intractable to approximate. The intuitions of this work can extend beyond aerial photography to broader applications such as pesticide spraying and strategic sensor placement.
>
---
#### [replaced 010] Gaussian Process Implicit Surfaces as Control Barrier Functions for Safe Robot Navigation
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于机器人安全导航任务，旨在解决碰撞避免问题。通过将高斯过程隐表面作为控制屏障函数，实现安全轨迹规划。**

- **链接: [https://arxiv.org/pdf/2510.12919v2](https://arxiv.org/pdf/2510.12919v2)**

> **作者:** Mouhyemen Khan; Tatsuya Ibuki; Abhijit Chatterjee
>
> **备注:** 8 pages, 7 figures, under review
>
> **摘要:** Level set methods underpin modern safety techniques such as control barrier functions (CBFs), while also serving as implicit surface representations for geometric shapes via distance fields. Inspired by these two paradigms, we propose a unified framework where the implicit surface itself acts as a CBF. We leverage Gaussian process (GP) implicit surface (GPIS) to represent the safety boundaries, using safety samples which are derived from sensor measurements to condition the GP. The GP posterior mean defines the implicit safety surface (safety belief), while the posterior variance provides a robust safety margin. Although GPs have favorable properties such as uncertainty estimation and analytical tractability, they scale cubically with data. To alleviate this issue, we develop a sparse solution called sparse Gaussian CBFs. To the best of our knowledge, GPIS have not been explicitly used to synthesize CBFs. We validate the approach on collision avoidance tasks in two settings: a simulated 7-DOF manipulator operating around the Stanford bunny, and a quadrotor navigating in 3D around a physical chair. In both cases, Gaussian CBFs (with and without sparsity) enable safe interaction and collision-free execution of trajectories that would otherwise intersect the objects.
>
---
#### [replaced 011] Effective Game-Theoretic Motion Planning via Nested Search
- **分类: cs.RO; cs.MA**

- **简介: 该论文属于机器人运动规划任务，解决多智能体交互中的安全决策问题。提出GTNS方法，通过博弈论计算纳什均衡，提升自主系统在复杂环境中的决策能力。**

- **链接: [https://arxiv.org/pdf/2511.08001v2](https://arxiv.org/pdf/2511.08001v2)**

> **作者:** Avishav Engle; Andrey Zhitnikov; Oren Salzman; Omer Ben-Porat; Kiril Solovey
>
> **备注:** Updated version. Additional experiment included. Cosmetic/formatting changes made
>
> **摘要:** To facilitate effective, safe deployment in the real world, individual robots must reason about interactions with other agents, which often occur without explicit communication. Recent work has identified game theory, particularly the concept of Nash Equilibrium (NE), as a key enabler for behavior-aware decision-making. Yet, existing work falls short of fully unleashing the power of game-theoretic reasoning. Specifically, popular optimization-based methods require simplified robot dynamics and tend to get trapped in local minima due to convexification. Other works that rely on payoff matrices suffer from poor scalability due to the explicit enumeration of all possible trajectories. To bridge this gap, we introduce Game-Theoretic Nested Search (GTNS), a novel, scalable, and provably correct approach for computing NEs in general dynamical systems. GTNS efficiently searches the action space of all agents involved, while discarding trajectories that violate the NE constraint (no unilateral deviation) through an inner search over a lower-dimensional space. Our algorithm enables explicit selection among equilibria by utilizing a user-specified global objective, thereby capturing a rich set of realistic interactions. We demonstrate the approach on a variety of autonomous driving and racing scenarios where we achieve solutions in mere seconds on commodity hardware.
>
---
#### [replaced 012] Never-Ending Behavior-Cloning Agent for Robotic Manipulation
- **分类: cs.RO; cs.AI**

- **简介: 该论文提出NBAgent，解决机器人操作中的3D场景表示和任务学习问题，通过语义渲染与知识解耦实现持续学习。**

- **链接: [https://arxiv.org/pdf/2403.00336v3](https://arxiv.org/pdf/2403.00336v3)**

> **作者:** Wenqi Liang; Gan Sun; Yao He; Yu Ren; Jiahua Dong; Yang Cong
>
> **备注:** 17 pages, 6 figures, 9 tables
>
> **摘要:** Relying on multi-modal observations, embodied robots (e.g., humanoid robots) could perform multiple robotic manipulation tasks in unstructured real-world environments. However, most language-conditioned behavior-cloning agents in robots still face existing long-standing challenges, i.e., 3D scene representation and human-level task learning, when adapting into a series of new tasks in practical scenarios. We here investigate these above challenges with NBAgent in embodied robots, a pioneering language-conditioned Never-ending Behavior-cloning Agent, which can continually learn observation knowledge of novel 3D scene semantics and robot manipulation skills from skill-shared and skill-specific attributes, respectively. Specifically, we propose a skill-shared semantic rendering module and a skill-shared representation distillation module to effectively learn 3D scene semantics from skill-shared attribute, further tackling 3D scene representation overlooking. Meanwhile, we establish a skill-specific evolving planner to perform manipulation knowledge decoupling, which can continually embed novel skill-specific knowledge like human from latent and low-rank space. Finally, we design a never-ending embodied robot manipulation benchmark, and expensive experiments demonstrate the significant performance of our method.
>
---
#### [replaced 013] Forecasting in Offline Reinforcement Learning for Non-stationary Environments
- **分类: cs.LG; cs.AI; cs.RO**

- **简介: 该论文属于强化学习任务，解决非平稳环境中离线RL的性能下降问题。提出FORL框架，结合扩散生成和零样本时间序列模型，提升代理在非平稳环境中的鲁棒性。**

- **链接: [https://arxiv.org/pdf/2512.01987v3](https://arxiv.org/pdf/2512.01987v3)**

> **作者:** Suzan Ece Ada; Georg Martius; Emre Ugur; Erhan Oztop
>
> **备注:** The Thirty-Ninth Annual Conference on Neural Information Processing Systems, NeurIPS 2025
>
> **摘要:** Offline Reinforcement Learning (RL) provides a promising avenue for training policies from pre-collected datasets when gathering additional interaction data is infeasible. However, existing offline RL methods often assume stationarity or only consider synthetic perturbations at test time, assumptions that often fail in real-world scenarios characterized by abrupt, time-varying offsets. These offsets can lead to partial observability, causing agents to misperceive their true state and degrade performance. To overcome this challenge, we introduce Forecasting in Non-stationary Offline RL (FORL), a framework that unifies (i) conditional diffusion-based candidate state generation, trained without presupposing any specific pattern of future non-stationarity, and (ii) zero-shot time-series foundation models. FORL targets environments prone to unexpected, potentially non-Markovian offsets, requiring robust agent performance from the onset of each episode. Empirical evaluations on offline RL benchmarks, augmented with real-world time-series data to simulate realistic non-stationarity, demonstrate that FORL consistently improves performance compared to competitive baselines. By integrating zero-shot forecasting with the agent's experience, we aim to bridge the gap between offline RL and the complexities of real-world, non-stationary environments.
>
---
#### [replaced 014] Adaptive Keyframe Selection for Scalable 3D Scene Reconstruction in Dynamic Environments
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于3D场景重建任务，解决动态环境中关键帧选择问题。通过集成误差和动量模块，提升重建质量与实时性。**

- **链接: [https://arxiv.org/pdf/2510.23928v3](https://arxiv.org/pdf/2510.23928v3)**

> **作者:** Raman Jha; Yang Zhou; Giuseppe Loianno
>
> **备注:** Accepted at ROBOVIS 2026
>
> **摘要:** In this paper, we propose an adaptive keyframe selection method for improved 3D scene reconstruction in dynamic environments. The proposed method integrates two complementary modules: an error-based selection module utilizing photometric and structural similarity (SSIM) errors, and a momentum-based update module that dynamically adjusts keyframe selection thresholds according to scene motion dynamics. By dynamically curating the most informative frames, our approach addresses a key data bottleneck in real-time perception. This allows for the creation of high-quality 3D world representations from a compressed data stream, a critical step towards scalable robot learning and deployment in complex, dynamic environments. Experimental results demonstrate significant improvements over traditional static keyframe selection strategies, such as fixed temporal intervals or uniform frame skipping. These findings highlight a meaningful advancement toward adaptive perception systems that can dynamically respond to complex and evolving visual scenes. We evaluate our proposed adaptive keyframe selection module on two recent state-of-the-art 3D reconstruction networks, Spann3r and CUT3R, and observe consistent improvements in reconstruction quality across both frameworks. Furthermore, an extensive ablation study confirms the effectiveness of each individual component in our method, underlining their contribution to the overall performance gains.
>
---
#### [replaced 015] Anti-Slip AI-Driven Model-Free Control with Global Exponential Stability in Skid-Steering Robots
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于控制任务，旨在解决滑移转向重型机器人在复杂地形下的滑移问题。通过设计基于神经网络的模型无关控制方法，实现系统全局指数稳定。**

- **链接: [https://arxiv.org/pdf/2504.08831v2](https://arxiv.org/pdf/2504.08831v2)**

> **作者:** Mehdi Heydari Shahna; Pauli Mustalahti; Jouni Mattila
>
> **备注:** This paper has been published in 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)
>
> **摘要:** Undesired lateral and longitudinal wheel slippage can disrupt a mobile robot's heading angle, traction, and, eventually, desired motion. This issue makes the robotization and accurate modeling of heavy-duty machinery very challenging because the application primarily involves off-road terrains, which are susceptible to uneven motion and severe slippage. As a step toward robotization in skid-steering heavy-duty robot (SSHDR), this paper aims to design an innovative robust model-free control system developed by neural networks to strongly stabilize the robot dynamics in the presence of a broad range of potential wheel slippages. Before the control design, the dynamics of the SSHDR are first investigated by mathematically incorporating slippage effects, assuming that all functional modeling terms of the system are unknown to the control system. Then, a novel tracking control framework to guarantee global exponential stability of the SSHDR is designed as follows: 1) the unknown modeling of wheel dynamics is approximated using radial basis function neural networks (RBFNNs); and 2) a new adaptive law is proposed to compensate for slippage effects and tune the weights of the RBFNNs online during execution. Simulation and experimental results verify the proposed tracking control performance of a 4,836 kg SSHDR operating on slippery terrain.
>
---
#### [replaced 016] Think, Act, Learn: A Framework for Autonomous Robotic Agents using Closed-Loop Large Language Models
- **分类: cs.RO; cs.HC**

- **简介: 该论文提出T-A-L框架，解决机器人在动态环境中适应性不足的问题。通过闭环交互，使机器人自主学习并优化策略，提升任务成功率和泛化能力。**

- **链接: [https://arxiv.org/pdf/2507.19854v3](https://arxiv.org/pdf/2507.19854v3)**

> **作者:** Anjali R. Menon; Rohit K. Sharma; Priya Singh; Chengyu Wang; Aurora M. Ferreira; Mateja Novak
>
> **备注:** 13 pages, 7 figures
>
> **摘要:** The integration of Large Language Models (LLMs) into robotics has unlocked unprecedented capabilities in high-level task planning. However, most current systems operate in an open-loop fashion, where LLMs act as one-shot planners, rendering them brittle and unable to adapt to unforeseen circumstances in dynamic physical environments. To overcome this limitation, this paper introduces the "Think, Act, Learn" (T-A-L) framework, a novel architecture that enables an embodied agent to autonomously learn and refine its policies through continuous interaction. Our framework establishes a closed-loop cycle where an LLM first "thinks" by decomposing high-level commands into actionable plans. The robot then "acts" by executing these plans while gathering rich, multimodal sensory feedback. Critically, the "learn" module processes this feedback to facilitate LLM-driven self-reflection, allowing the agent to perform causal analysis on its failures and generate corrective strategies. These insights are stored in an experiential memory to guide future planning cycles. We demonstrate through extensive experiments in both simulation and the real world that our T-A-L agent significantly outperforms baseline methods, including open-loop LLMs, Behavioral Cloning, and traditional Reinforcement Learning. Our framework achieves over a 97% success rate on complex, long-horizon tasks, converges to a stable policy in an average of just 9 trials, and exhibits remarkable generalization to unseen tasks. This work presents a significant step towards developing more robust, adaptive, and truly autonomous robotic agents.
>
---
#### [replaced 017] MambaIO: Global-Coordinate Inertial Odometry for Pedestrians via Multi-Scale Frequency-Decoupled Modeling
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于惯性里程计任务，旨在提升行人定位精度。针对传统全局坐标系的局限性，提出MambaIO方法，通过多尺度分解和Mamba架构优化运动特征提取。**

- **链接: [https://arxiv.org/pdf/2511.15645v2](https://arxiv.org/pdf/2511.15645v2)**

> **作者:** Shanshan Zhang; Liqin Wu; Wenying Cao; Siyue Wang; Tianshui Wen; Qi Zhang; Xuemin Hong; Ao Peng; Lingxiang Zheng; Yu Yang
>
> **摘要:** Inertial Odometry (IO) enables real-time localization using only acceleration and angular velocity measurements from an Inertial Measurement Unit (IMU), making it a promising solution for localization in consumer-grade applications. Traditionally, researchers have routinely transformed IMU measurements into the global frame to obtain smoother motion representations. However, recent studies in drone scenarios have demonstrated that the body frame can significantly improve localization accuracy, prompting a re-evaluation of the suitability of the global frame for pedestrian IO. To address this issue, this paper systematically evaluates the effectiveness of the global frame in pedestrian IO through theoretical analysis, qualitative inspection, and quantitative experiments. Building upon these findings, we further propose MambaIO, which decomposes IMU measurements into high-frequency and low-frequency components using a Laplacian pyramid. The low-frequency component is processed by a Mamba architecture to extract implicit contextual motion cues, while the high-frequency component is handled by a convolutional structure to capture fine-grained local motion details. Experiments on multiple public datasets show that MambaIO substantially reduces localization error and achieves state-of-the-art (SOTA) performance. To the best of our knowledge, this is the first application of the Mamba architecture to the IO task.
>
---
#### [replaced 018] Enhancing Fatigue Detection through Heterogeneous Multi-Source Data Integration and Cross-Domain Modality Imputation
- **分类: cs.RO; cs.AI**

- **简介: 该论文属于疲劳检测任务，旨在解决真实场景中传感器性能下降的问题。通过融合多源异构数据和跨域模态补全，提升检测效果。**

- **链接: [https://arxiv.org/pdf/2507.16859v4](https://arxiv.org/pdf/2507.16859v4)**

> **作者:** Luobin Cui; Yanlai Wu; Tang Ying; Weikai Li
>
> **备注:** 4figures,14pages
>
> **摘要:** Fatigue detection for human operators plays a key role in safety critical applications such as aviation, mining, and long haul transport. While numerous studies have demonstrated the effectiveness of high fidelity sensors in controlled laboratory environments, their performance often degrades when ported to real world settings due to noise, lighting conditions, and field of view constraints, thereby limiting their practicality. This paper formalizes a deployment oriented setting for real world fatigue detection, where high quality sensors are often unavailable in practical applications. To address this challenge, we propose leveraging knowledge from heterogeneous source domains, including high fidelity sensors that are difficult to deploy in the field but commonly used in controlled environments, to assist fatigue detection in the real world target domain. Building on this idea, we design a heterogeneous and multiple source fatigue detection framework that adaptively utilizes the available modalities in the target domain while exploiting diverse configurations in the source domains through alignment across domains and modality imputation. Our experiments, conducted using a field deployed sensor setup and two publicly available human fatigue datasets, demonstrate the practicality, robustness, and improved generalization of our approach across subjects and domains. The proposed method achieves consistent gains over strong baselines in sensor constrained scenarios. This work has been submitted to the IEEE for possible publication. Copyright may be transferred without notice, after which this version may no longer be accessible.
>
---
#### [replaced 019] Mechanically Programming the Cross-Sectional Shape of Soft Growing Robotic Structures for Patient Transfer
- **分类: cs.RO**

- **简介: 该论文属于医疗辅助机器人领域，解决软体机器人在患者转移任务中形状控制问题。通过柔性条带实现结构横截面的机械编程，提升其适应性和灵活性。**

- **链接: [https://arxiv.org/pdf/2505.11593v3](https://arxiv.org/pdf/2505.11593v3)**

> **作者:** O. Godson Osele; Kentaro Barhydt; Teagan Sullivan; H. Harry Asada; Allison M. Okamura
>
> **摘要:** Pneumatic soft everting robotic structures have the potential to facilitate human transfer tasks due to their ability to grow underneath humans without sliding friction and their utility as a flexible sling when deflated. Tubular structures naturally yield circular cross-sections when inflated, whereas a robotic sling must be both thin enough to grow between them and their resting surface and wide enough to cradle the human. Recent works have achieved flattened cross-sections by including rigid components into the structure, but this reduces conformability to the human. We present a method of mechanically programming the cross-section of soft everting robotic structures using flexible strips that constrain radial expansion between points along the outer membrane. Our method enables simultaneously wide and thin profiles while maintaining the full multi-axis flexibility of traditional slings. We develop and validate a model relating the geometric design specifications to the fabrication parameters, and experimentally characterize their effects on growth rate. Finally, we prototype a soft growing robotic sling system and demonstrate its use for assisting a single caregiver in bed-to-chair patient transfer.
>
---
#### [replaced 020] Never too Cocky to Cooperate: An FIM and RL-based USV-AUV Collaborative System for Underwater Tasks in Extreme Sea Conditions
- **分类: cs.RO; eess.SY**

- **简介: 该论文属于水下任务协作系统研究，解决极端海况下USV与AUV协同作业问题。提出融合FIM和强化学习的协作方案，提升定位与任务执行性能。**

- **链接: [https://arxiv.org/pdf/2504.14894v3](https://arxiv.org/pdf/2504.14894v3)**

> **作者:** Jingzehua Xu; Guanwen Xie; Jiwei Tang; Yimian Ding; Weiyi Liu; Junhao Huang; Shuai Zhang; Yi Li
>
> **备注:** This paper has been accepted by IEEE Transactions on Mobile Computing
>
> **摘要:** This paper develops a novel unmanned surface vehicle (USV)-autonomous underwater vehicle (AUV) collaborative system designed to enhance underwater task performance in extreme sea conditions. The system integrates a dual strategy: (1) high-precision multi-AUV localization enabled by Fisher information matrix-optimized USV path planning, and (2) reinforcement learning-based cooperative planning and control method for multi-AUV task execution. Extensive experimental evaluations in the underwater data collection task demonstrate the system's operational feasibility, with quantitative results showing significant performance improvements over baseline methods. The proposed system exhibits robust coordination capabilities between USV and AUVs while maintaining stability in extreme sea conditions. To facilitate reproducibility and community advancement, we provide an open-source simulation toolkit available at: https://github.com/360ZMEM/USV-AUV-colab .
>
---
#### [replaced 021] Driving Beyond Privilege: Distilling Dense-Reward Knowledge into Sparse-Reward Policies
- **分类: cs.RO**

- **简介: 该论文属于自动驾驶任务，解决稀疏奖励下策略泛化问题。通过知识蒸馏，将密集奖励的动态模型知识转移到稀疏奖励策略中，提升性能与安全性。**

- **链接: [https://arxiv.org/pdf/2512.04279v2](https://arxiv.org/pdf/2512.04279v2)**

> **作者:** Feeza Khan Khanzada; Jaerock Kwon
>
> **摘要:** We study how to exploit dense simulator-defined rewards in vision-based autonomous driving without inheriting their misalignment with deployment metrics. In realistic simulators such as CARLA, privileged state (e.g., lane geometry, infractions, time-to-collision) can be converted into dense rewards that stabilize and accelerate model-based reinforcement learning, but policies trained directly on these signals often overfit and fail to generalize when evaluated on sparse objectives such as route completion and collision-free overtaking. We propose reward-privileged world model distillation, a two-stage framework in which a teacher DreamerV3-style agent is first trained with a dense privileged reward, and only its latent dynamics are distilled into a student trained solely on sparse task rewards. Teacher and student share the same observation space (semantic bird's-eye-view images); privileged information enters only through the teacher's reward, and the student does not imitate the teacher's actions or value estimates. Instead, the student's world model is regularized to match the teacher's latent dynamics while its policy is learned from scratch on sparse success/failure signals. In CARLA lane-following and overtaking benchmarks, sparse-reward students outperform both dense-reward teachers and sparse-from-scratch baselines. On unseen lane-following routes, reward-privileged distillation improves success by about 23 percent relative to the dense teacher while maintaining comparable or better safety. On overtaking, students retain near-perfect performance on training routes and achieve up to a 27x improvement in success on unseen routes, with improved lane keeping. These results show that dense rewards can be leveraged to learn richer dynamics models while keeping the deployed policy optimized strictly for sparse, deployment-aligned objectives.
>
---
#### [replaced 022] Model-free source seeking of exponentially convergent unicycle: theoretical and robotic experimental results
- **分类: math.OC; cs.RO**

- **简介: 该论文属于自主导航任务，解决未知信号源的定位问题。提出一种无需模型的实时控制方法，使无人车指数收敛至信号极值点，验证了其在不同条件下的鲁棒性与有效性。**

- **链接: [https://arxiv.org/pdf/2511.00752v2](https://arxiv.org/pdf/2511.00752v2)**

> **作者:** Rohan Palanikumar; Ahmed A. Elgohary; Victoria Grushkovskaya; Sameh A. Eisa
>
> **摘要:** This paper introduces a novel model-free, real-time unicycle-based source seeking design. This design autonomously steers the unicycle dynamic system towards the extremum point of an objective function or physical/scalar signal that is unknown expression-wise, but accessible via measurements. A key contribution of this paper is that the introduced design converges exponentially to the extremum point of objective functions (or scalar signals) that behave locally like a higher-degree power function (e.g., fourth-degree polynomial function) as opposed to locally quadratic objective functions, the usual case in literature. We provide theoretical results and design characterization, supported by a variety of simulation results that demonstrate the robustness of the proposed design, including cases with different initial conditions and measurement delays/noise. Also, for the first time in the literature, we provide experimental robotic results that demonstrate the effectiveness of the proposed design and its exponential convergence ability.
>
---
#### [replaced 023] LidarDM: Generative LiDAR Simulation in a Generated World
- **分类: cs.CV; cs.RO**

- **简介: 该论文提出LidarDM，属于LiDAR生成任务，解决生成真实、时序一致的LiDAR视频问题。通过4D世界生成框架，结合扩散模型与动态物体，生成高质量LiDAR数据。**

- **链接: [https://arxiv.org/pdf/2404.02903v2](https://arxiv.org/pdf/2404.02903v2)**

> **作者:** Vlas Zyrianov; Henry Che; Zhijian Liu; Shenlong Wang
>
> **摘要:** We present LidarDM, a novel LiDAR generative model capable of producing realistic, layout-aware, physically plausible, and temporally coherent LiDAR videos. LidarDM stands out with two unprecedented capabilities in LiDAR generative modeling: (i) LiDAR generation guided by driving scenarios, offering significant potential for autonomous driving simulations, and (ii) 4D LiDAR point cloud generation, enabling the creation of realistic and temporally coherent sequences. At the heart of our model is a novel integrated 4D world generation framework. Specifically, we employ latent diffusion models to generate the 3D scene, combine it with dynamic actors to form the underlying 4D world, and subsequently produce realistic sensory observations within this virtual environment. Our experiments indicate that our approach outperforms competing algorithms in realism, temporal coherency, and layout consistency. We additionally show that LidarDM can be used as a generative world model simulator for training and testing perception models.
>
---
#### [replaced 024] UniTacHand: Unified Spatio-Tactile Representation for Human to Robotic Hand Skill Transfer
- **分类: cs.RO**

- **简介: 该论文属于机器人技能迁移任务，解决人类与机器人触觉数据不匹配的问题。通过构建统一的触觉表示，实现人类到机器人的零样本触觉策略迁移。**

- **链接: [https://arxiv.org/pdf/2512.21233v3](https://arxiv.org/pdf/2512.21233v3)**

> **作者:** Chi Zhang; Penglin Cai; Haoqi Yuan; Chaoyi Xu; Zongqing Lu
>
> **备注:** The first two authors contributed equally
>
> **摘要:** Tactile sensing is crucial for robotic hands to achieve human-level dexterous manipulation, especially in scenarios with visual occlusion. However, its application is often hindered by the difficulty of collecting large-scale real-world robotic tactile data. In this study, we propose to collect low-cost human manipulation data using haptic gloves for tactile-based robotic policy learning. The misalignment between human and robotic tactile data makes it challenging to transfer policies learned from human data to robots. To bridge this gap, we propose UniTacHand, a unified representation to align robotic tactile information captured by dexterous hands with human hand touch obtained from gloves. First, we project tactile signals from both human hands and robotic hands onto a morphologically consistent 2D surface space of the MANO hand model. This unification standardizes the heterogeneous data structures and inherently embeds the tactile signals with spatial context. Then, we introduce a contrastive learning method to align them into a unified latent space, trained on only 10 minutes of paired data from our data collection system. Our approach enables zero-shot tactile-based policy transfer from humans to a real robot, generalizing to objects unseen in the pre-training data. We also demonstrate that co-training on mixed data, including both human and robotic demonstrations via UniTacHand, yields better performance and data efficiency compared with using only robotic data. UniTacHand paves a path toward general, scalable, and data-efficient learning for tactile-based dexterous hands.
>
---
#### [replaced 025] RiemanLine: Riemannian Manifold Representation of 3D Lines for Factor Graph Optimization
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于三维线表示任务，解决机器人定位与结构映射中线特征的参数化问题。提出RiemanLine，统一表示单线与平行线组，减少参数空间并提升精度与稳定性。**

- **链接: [https://arxiv.org/pdf/2508.04335v2](https://arxiv.org/pdf/2508.04335v2)**

> **作者:** Yan Li; Ze Yang; Keisuke Tateno; Federico Tombari; Liang Zhao; Gim Hee Lee
>
> **摘要:** Minimal parametrization of 3D lines plays a critical role in camera localization and structural mapping. Existing representations in robotics and computer vision predominantly handle independent lines, overlooking structural regularities such as sets of parallel lines that are pervasive in man-made environments. This paper introduces \textbf{RiemanLine}, a unified minimal representation for 3D lines formulated on Riemannian manifolds that jointly accommodates both individual lines and parallel-line groups. Our key idea is to decouple each line landmark into global and local components: a shared vanishing direction optimized on the unit sphere $\mathcal{S}^2$, and scaled normal vectors constrained on orthogonal subspaces, enabling compact encoding of structural regularities. For $n$ parallel lines, the proposed representation reduces the parameter space from $4n$ (orthonormal form) to $2n+2$, naturally embedding parallelism without explicit constraints. We further integrate this parameterization into a factor graph framework, allowing global direction alignment and local reprojection optimization within a unified manifold-based bundle adjustment. Extensive experiments on ICL-NUIM, TartanAir, and synthetic benchmarks demonstrate that our method achieves significantly more accurate pose estimation and line reconstruction, while reducing parameter dimensionality and improving convergence stability.
>
---
#### [replaced 026] CHARM: Considering Human Attributes for Reinforcement Modeling
- **分类: cs.RO; cs.HC**

- **简介: 该论文属于强化学习任务，旨在解决人类反馈质量受个体特征影响的问题。通过实验分析人类特征与反馈模式的关系，提升反馈预测准确性。**

- **链接: [https://arxiv.org/pdf/2506.13079v2](https://arxiv.org/pdf/2506.13079v2)**

> **作者:** Qidi Fang; Hang Yu; Shijie Fang; Jindan Huang; Qiuyu Chen; Reuben M. Aronson; Elaine S. Short
>
> **摘要:** Reinforcement Learning from Human Feedback has recently achieved significant success in various fields, and its performance is highly related to feedback quality. While much prior work acknowledged that human teachers' characteristics would affect human feedback patterns, there is little work that has closely investigated the actual effects. In this work, we designed an exploratory study investigating how human feedback patterns are associated with human characteristics. We conducted a public space study with two long horizon tasks and 46 participants. We found that feedback patterns are not only correlated with task statistics, such as rewards, but also correlated with participants' characteristics, especially robot experience and educational background. Additionally, we demonstrated that human feedback value can be more accurately predicted with human characteristics compared to only using task statistics. All human feedback and characteristics we collected, and codes for our data collection and predicting more accurate human feedback are available at https://github.com/AABL-Lab/CHARM
>
---
#### [replaced 027] GrOMP: Grasped Object Manifold Projection for Multimodal Imitation Learning of Manipulation
- **分类: cs.RO**

- **简介: 该论文提出GrOMP方法，用于解决模仿学习中的轨迹精度问题。属于机械操作任务，通过约束物体到低维流形提升装配精度。**

- **链接: [https://arxiv.org/pdf/2512.03347v2](https://arxiv.org/pdf/2512.03347v2)**

> **作者:** William van den Bogert; Gregory Linkowski; Nima Fazeli
>
> **备注:** 8 pages, 8 figures, 2 tables
>
> **摘要:** Imitation Learning (IL) holds great potential for learning repetitive manipulation tasks, such as those in industrial assembly. However, its effectiveness is often limited by insufficient trajectory precision due to compounding errors. In this paper, we introduce Grasped Object Manifold Projection (GrOMP), an interactive method that mitigates these errors by constraining a non-rigidly grasped object to a lower-dimensional manifold. GrOMP assumes a precise task in which a manipulator holds an object that may shift within the grasp in an observable manner and must be mated with a grounded part. Crucially, all GrOMP enhancements are learned from the same expert dataset used to train the base IL policy, and are adjusted with an n-arm bandit-based interactive component. We propose a theoretical basis for GrOMP's improvement upon the well-known compounding error bound in IL literature. We demonstrate the framework on four precise assembly tasks using tactile feedback, and note that the approach remains modality-agnostic. Data and videos are available at williamvdb.github.io/GrOMPsite.
>
---
#### [replaced 028] How Much Progress Did I Make? An Unexplored Human Feedback Signal for Teaching Robots
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文属于人机协作学习任务，旨在通过“进度”信号提升机器人学习效果。研究验证了进度信号的有效性，并提供了一个非专家演示数据集。**

- **链接: [https://arxiv.org/pdf/2407.06459v2](https://arxiv.org/pdf/2407.06459v2)**

> **作者:** Hang Yu; Qidi Fang; Shijie Fang; Reuben M. Aronson; Elaine Schaertl Short
>
> **备注:** 8 pages. RO-MAN 2024
>
> **摘要:** Enhancing the expressiveness of human teaching is vital for both improving robots' learning from humans and the human-teaching-robot experience. In this work, we characterize and test a little-used teaching signal: \textit{progress}, designed to represent the completion percentage of a task. We conducted two online studies with 76 crowd-sourced participants and one public space study with 40 non-expert participants to validate the capability of this progress signal. We find that progress indicates whether the task is successfully performed, reflects the degree of task completion, identifies unproductive but harmless behaviors, and is likely to be more consistent across participants. Furthermore, our results show that giving progress does not require extra workload and time. An additional contribution of our work is a dataset of 40 non-expert demonstrations from the public space study through an ice cream topping-adding task, which we observe to be multi-policy and sub-optimal, with sub-optimality not only from teleoperation errors but also from exploratory actions and attempts. The dataset is available at https://github.com/TeachingwithProgress/Non-Expert\_Demonstrations.
>
---
