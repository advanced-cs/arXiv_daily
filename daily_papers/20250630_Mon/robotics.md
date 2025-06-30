# 机器人 cs.RO

- **最新发布 27 篇**

- **更新 17 篇**

## 最新发布

#### [new 001] An Introduction to Zero-Order Optimization Techniques for Robotics
- **分类: cs.RO**

- **简介: 该论文属于机器人学领域，旨在解决轨迹和策略优化问题。通过随机搜索方法，提出统一框架并开发新型强化学习算法。**

- **链接: [http://arxiv.org/pdf/2506.22087v1](http://arxiv.org/pdf/2506.22087v1)**

> **作者:** Armand Jordana; Jianghan Zhang; Joseph Amigo; Ludovic Righetti
>
> **摘要:** Zero-order optimization techniques are becoming increasingly popular in robotics due to their ability to handle non-differentiable functions and escape local minima. These advantages make them particularly useful for trajectory optimization and policy optimization. In this work, we propose a mathematical tutorial on random search. It offers a simple and unifying perspective for understanding a wide range of algorithms commonly used in robotics. Leveraging this viewpoint, we classify many trajectory optimization methods under a common framework and derive novel competitive RL algorithms.
>
---
#### [new 002] Real-Time 3D Guidewire Reconstruction from Intraoperative DSA Images for Robot-Assisted Endovascular Interventions
- **分类: cs.RO**

- **简介: 该论文属于医学影像处理任务，旨在解决机器人辅助血管介入中 guidewire 三维重建问题。通过融合术前 CTA 和术中 DSA 数据，实现高精度实时重建。**

- **链接: [http://arxiv.org/pdf/2506.21631v1](http://arxiv.org/pdf/2506.21631v1)**

> **作者:** Tianliang Yao; Bingrui Li; Bo Lu; Zhiqiang Pei; Yixuan Yuan; Peng Qi
>
> **备注:** This paper has been accepted by IEEE/RSJ IROS 2025
>
> **摘要:** Accurate three-dimensional (3D) reconstruction of guidewire shapes is crucial for precise navigation in robot-assisted endovascular interventions. Conventional 2D Digital Subtraction Angiography (DSA) is limited by the absence of depth information, leading to spatial ambiguities that hinder reliable guidewire shape sensing. This paper introduces a novel multimodal framework for real-time 3D guidewire reconstruction, combining preoperative 3D Computed Tomography Angiography (CTA) with intraoperative 2D DSA images. The method utilizes robust feature extraction to address noise and distortion in 2D DSA data, followed by deformable image registration to align the 2D projections with the 3D CTA model. Subsequently, the inverse projection algorithm reconstructs the 3D guidewire shape, providing real-time, accurate spatial information. This framework significantly enhances spatial awareness for robotic-assisted endovascular procedures, effectively bridging the gap between preoperative planning and intraoperative execution. The system demonstrates notable improvements in real-time processing speed, reconstruction accuracy, and computational efficiency. The proposed method achieves a projection error of 1.76$\pm$0.08 pixels and a length deviation of 2.93$\pm$0.15\%, with a frame rate of 39.3$\pm$1.5 frames per second (FPS). These advancements have the potential to optimize robotic performance and increase the precision of complex endovascular interventions, ultimately contributing to better clinical outcomes.
>
---
#### [new 003] Evaluating Pointing Gestures for Target Selection in Human-Robot Collaboration
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于人机协作任务，旨在解决目标选择中的指向手势识别问题。通过姿态估计和几何模型提取手势数据，并集成到机器人系统中进行验证。**

- **链接: [http://arxiv.org/pdf/2506.22116v1](http://arxiv.org/pdf/2506.22116v1)**

> **作者:** Noora Sassali; Roel Pieters
>
> **备注:** Accepted by the 2025 34th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN). Preprint
>
> **摘要:** Pointing gestures are a common interaction method used in Human-Robot Collaboration for various tasks, ranging from selecting targets to guiding industrial processes. This study introduces a method for localizing pointed targets within a planar workspace. The approach employs pose estimation, and a simple geometric model based on shoulder-wrist extension to extract gesturing data from an RGB-D stream. The study proposes a rigorous methodology and comprehensive analysis for evaluating pointing gestures and target selection in typical robotic tasks. In addition to evaluating tool accuracy, the tool is integrated into a proof-of-concept robotic system, which includes object detection, speech transcription, and speech synthesis to demonstrate the integration of multiple modalities in a collaborative application. Finally, a discussion over tool limitations and performance is provided to understand its role in multimodal robotic systems. All developments are available at: https://github.com/NMKsas/gesture_pointer.git.
>
---
#### [new 004] Ark: An Open-source Python-based Framework for Robot Learning
- **分类: cs.RO; cs.AI; cs.LG**

- **简介: 该论文提出ARK框架，解决机器人软件开发复杂问题，通过Python实现高效仿真与真实机器人交互，促进自主机器人研究与应用。**

- **链接: [http://arxiv.org/pdf/2506.21628v1](http://arxiv.org/pdf/2506.21628v1)**

> **作者:** Magnus Dierking; Christopher E. Mower; Sarthak Das; Huang Helong; Jiacheng Qiu; Cody Reading; Wei Chen; Huidong Liang; Huang Guowei; Jan Peters; Quan Xingyue; Jun Wang; Haitham Bou-Ammar
>
> **摘要:** Robotics has made remarkable hardware strides-from DARPA's Urban and Robotics Challenges to the first humanoid-robot kickboxing tournament-yet commercial autonomy still lags behind progress in machine learning. A major bottleneck is software: current robot stacks demand steep learning curves, low-level C/C++ expertise, fragmented tooling, and intricate hardware integration, in stark contrast to the Python-centric, well-documented ecosystems that propelled modern AI. We introduce ARK, an open-source, Python-first robotics framework designed to close that gap. ARK presents a Gym-style environment interface that allows users to collect data, preprocess it, and train policies using state-of-the-art imitation-learning algorithms (e.g., ACT, Diffusion Policy) while seamlessly toggling between high-fidelity simulation and physical robots. A lightweight client-server architecture provides networked publisher-subscriber communication, and optional C/C++ bindings ensure real-time performance when needed. ARK ships with reusable modules for control, SLAM, motion planning, system identification, and visualization, along with native ROS interoperability. Comprehensive documentation and case studies-from manipulation to mobile navigation-demonstrate rapid prototyping, effortless hardware swapping, and end-to-end pipelines that rival the convenience of mainstream machine-learning workflows. By unifying robotics and AI practices under a common Python umbrella, ARK lowers entry barriers and accelerates research and commercial deployment of autonomous robots.
>
---
#### [new 005] A MILP-Based Solution to Multi-Agent Motion Planning and Collision Avoidance in Constrained Environments
- **分类: cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于多智能体路径规划任务，解决受限环境下的碰撞避免问题。通过MILP方法结合区域序列与超平面模型，提升规划效率与轨迹平滑性。**

- **链接: [http://arxiv.org/pdf/2506.21982v1](http://arxiv.org/pdf/2506.21982v1)**

> **作者:** Akshay Jaitly; Jack Cline; Siavash Farzan
>
> **备注:** Accepted to 2025 IEEE International Conference on Automation Science and Engineering (CASE 2025)
>
> **摘要:** We propose a mixed-integer linear program (MILP) for multi-agent motion planning that embeds Polytopic Action-based Motion Planning (PAAMP) into a sequence-then-solve pipeline. Region sequences confine each agent to adjacent convex polytopes, while a big-M hyperplane model enforces inter-agent separation. Collision constraints are applied only to agents sharing or neighboring a region, which reduces binary variables exponentially compared with naive formulations. An L1 path-length-plus-acceleration cost yields smooth trajectories. We prove finite-time convergence and demonstrate on representative multi-agent scenarios with obstacles that our formulation produces collision-free trajectories an order of magnitude faster than an unstructured MILP baseline.
>
---
#### [new 006] Multi-Robot Assembly of Deformable Linear Objects Using Multi-Modal Perception
- **分类: cs.RO**

- **简介: 该论文属于多机器人协作任务，旨在解决工业中柔性线性物体（如电缆）的装配问题。通过多模态感知与规划框架，实现从抓取到安装的全流程自动化。**

- **链接: [http://arxiv.org/pdf/2506.22034v1](http://arxiv.org/pdf/2506.22034v1)**

> **作者:** Kejia Chen; Celina Dettmering; Florian Pachler; Zhuo Liu; Yue Zhang; Tailai Cheng; Jonas Dirr; Zhenshan Bing; Alois Knoll; Rüdiger Daub
>
> **摘要:** Industrial assembly of deformable linear objects (DLOs) such as cables offers great potential for many industries. However, DLOs pose several challenges for robot-based automation due to the inherent complexity of deformation and, consequentially, the difficulties in anticipating the behavior of DLOs in dynamic situations. Although existing studies have addressed isolated subproblems like shape tracking, grasping, and shape control, there has been limited exploration of integrated workflows that combine these individual processes. To address this gap, we propose an object-centric perception and planning framework to achieve a comprehensive DLO assembly process throughout the industrial value chain. The framework utilizes visual and tactile information to track the DLO's shape as well as contact state across different stages, which facilitates effective planning of robot actions. Our approach encompasses robot-based bin picking of DLOs from cluttered environments, followed by a coordinated handover to two additional robots that mount the DLOs onto designated fixtures. Real-world experiments employing a setup with multiple robots demonstrate the effectiveness of the approach and its relevance to industrial scenarios.
>
---
#### [new 007] Embodied Domain Adaptation for Object Detection
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于目标检测任务，解决室内环境下的域适应问题。通过无源域适应方法提升模型在动态环境中的泛化能力。**

- **链接: [http://arxiv.org/pdf/2506.21860v1](http://arxiv.org/pdf/2506.21860v1)**

> **作者:** Xiangyu Shi; Yanyuan Qiao; Lingqiao Liu; Feras Dayoub
>
> **备注:** Accepted by IROS 2025
>
> **摘要:** Mobile robots rely on object detectors for perception and object localization in indoor environments. However, standard closed-set methods struggle to handle the diverse objects and dynamic conditions encountered in real homes and labs. Open-vocabulary object detection (OVOD), driven by Vision Language Models (VLMs), extends beyond fixed labels but still struggles with domain shifts in indoor environments. We introduce a Source-Free Domain Adaptation (SFDA) approach that adapts a pre-trained model without accessing source data. We refine pseudo labels via temporal clustering, employ multi-scale threshold fusion, and apply a Mean Teacher framework with contrastive learning. Our Embodied Domain Adaptation for Object Detection (EDAOD) benchmark evaluates adaptation under sequential changes in lighting, layout, and object diversity. Our experiments show significant gains in zero-shot detection performance and flexible adaptation to dynamic indoor conditions.
>
---
#### [new 008] AeroLite-MDNet: Lightweight Multi-task Deviation Detection Network for UAV Landing
- **分类: cs.RO; cs.AI; cs.CV**

- **简介: 该论文属于无人机着陆偏差检测任务，旨在解决GPS干扰下的精准着陆问题。提出AeroLite-MDNet模型，结合多尺度融合与分割分支，提升检测精度与响应速度。**

- **链接: [http://arxiv.org/pdf/2506.21635v1](http://arxiv.org/pdf/2506.21635v1)**

> **作者:** Haiping Yang; Huaxing Liu; Wei Wu; Zuohui Chen; Ning Wu
>
> **摘要:** Unmanned aerial vehicles (UAVs) are increasingly employed in diverse applications such as land surveying, material transport, and environmental monitoring. Following missions like data collection or inspection, UAVs must land safely at docking stations for storage or recharging, which is an essential requirement for ensuring operational continuity. However, accurate landing remains challenging due to factors like GPS signal interference. To address this issue, we propose a deviation warning system for UAV landings, powered by a novel vision-based model called AeroLite-MDNet. This model integrates a multiscale fusion module for robust cross-scale object detection and incorporates a segmentation branch for efficient orientation estimation. We introduce a new evaluation metric, Average Warning Delay (AWD), to quantify the system's sensitivity to landing deviations. Furthermore, we contribute a new dataset, UAVLandData, which captures real-world landing deviation scenarios to support training and evaluation. Experimental results show that our system achieves an AWD of 0.7 seconds with a deviation detection accuracy of 98.6\%, demonstrating its effectiveness in enhancing UAV landing reliability. Code will be available at https://github.com/ITTTTTI/Maskyolo.git
>
---
#### [new 009] LMPVC and Policy Bank: Adaptive voice control for industrial robots with code generating LLMs and reusable Pythonic policies
- **分类: cs.RO**

- **简介: 该论文属于人机协作任务，旨在解决工业机器人语音控制的适应性问题。通过结合大语言模型和可重用策略库，实现灵活高效的语音指令执行。**

- **链接: [http://arxiv.org/pdf/2506.22028v1](http://arxiv.org/pdf/2506.22028v1)**

> **作者:** Ossi Parikka; Roel Pieters
>
> **备注:** Accepted by the 2025 34th IEEE International Conference on Robot and Human Interactive Communication (RO-MAN). For further information, videos and code, see https://github.com/ozzyuni/LMPVC
>
> **摘要:** Modern industry is increasingly moving away from mass manufacturing, towards more specialized and personalized products. As manufacturing tasks become more complex, full automation is not always an option, human involvement may be required. This has increased the need for advanced human robot collaboration (HRC), and with it, improved methods for interaction, such as voice control. Recent advances in natural language processing, driven by artificial intelligence (AI), have the potential to answer this demand. Large language models (LLMs) have rapidly developed very impressive general reasoning capabilities, and many methods of applying this to robotics have been proposed, including through the use of code generation. This paper presents Language Model Program Voice Control (LMPVC), an LLM-based prototype voice control architecture with integrated policy programming and teaching capabilities, built for use with Robot Operating System 2 (ROS2) compatible robots. The architecture builds on prior works using code generation for voice control by implementing an additional programming and teaching system, the Policy Bank. We find this system can compensate for the limitations of the underlying LLM, and allow LMPVC to adapt to different downstream tasks without a slow and costly training process. The architecture and additional results are released on GitHub (https://github.com/ozzyuni/LMPVC).
>
---
#### [new 010] RM-Dijkstra: A surface optimal path planning algorithm based on Riemannian metric
- **分类: cs.RO; math.OC; 00A69, 93C85, 14H55; I.2.9**

- **简介: 该论文属于路径规划任务，解决移动机器人在表面的最优路径问题。提出RM-Dijkstra算法，基于黎曼度量模型，提升路径精度与平滑性。**

- **链接: [http://arxiv.org/pdf/2506.22170v1](http://arxiv.org/pdf/2506.22170v1)**

> **作者:** Yu Zhang; Xiao-Song Yang
>
> **备注:** 7 pages
>
> **摘要:** The Dijkstra algorithm is a classic path planning method, which operates in a discrete graph space to determine the shortest path from a specified source point to a target node or all other nodes based on non-negative edge weights. Numerous studies have focused on the Dijkstra algorithm due to its potential application. However, its application in surface path planning for mobile robots remains largely unexplored. In this letter, a surface optimal path planning algorithm called RM-Dijkstra is proposed, which is based on Riemannian metric model. By constructing a new Riemannian metric on the 2D projection plane, the surface optimal path planning problem is therefore transformed into a geometric problem on the 2D plane with new Riemannian metric. Induced by the standard Euclidean metric on surface, the constructed new metric reflects environmental information of the robot and ensures that the projection map is an isometric immersion. By conducting a series of simulation tests, the experimental results demonstrate that the RM-Dijkstra algorithm not only effectively solves the optimal path planning problem on surfaces, but also outperforms traditional path planning algorithms in terms of path accuracy and smoothness, particularly in complex scenarios.
>
---
#### [new 011] Robotic Multimodal Data Acquisition for In-Field Deep Learning Estimation of Cover Crop Biomass
- **分类: cs.RO**

- **简介: 该论文属于农业遥感任务，旨在解决覆盖作物生物量估算问题。通过多模态传感器融合与机器学习，提升生物量预测精度，支持精准农业管理。**

- **链接: [http://arxiv.org/pdf/2506.22364v1](http://arxiv.org/pdf/2506.22364v1)**

> **作者:** Joe Johnson; Phanender Chalasani; Arnav Shah; Ram L. Ray; Muthukumar Bagavathiannan
>
> **备注:** Accepted in the Extended Abstract, The 22nd International Conference on Ubiquitous Robots (UR 2025), Texas, USA
>
> **摘要:** Accurate weed management is essential for mitigating significant crop yield losses, necessitating effective weed suppression strategies in agricultural systems. Integrating cover crops (CC) offers multiple benefits, including soil erosion reduction, weed suppression, decreased nitrogen requirements, and enhanced carbon sequestration, all of which are closely tied to the aboveground biomass (AGB) they produce. However, biomass production varies significantly due to microsite variability, making accurate estimation and mapping essential for identifying zones of poor weed suppression and optimizing targeted management strategies. To address this challenge, developing a comprehensive CC map, including its AGB distribution, will enable informed decision-making regarding weed control methods and optimal application rates. Manual visual inspection is impractical and labor-intensive, especially given the extensive field size and the wide diversity and variation of weed species and sizes. In this context, optical imagery and Light Detection and Ranging (LiDAR) data are two prominent sources with unique characteristics that enhance AGB estimation. This study introduces a ground robot-mounted multimodal sensor system designed for agricultural field mapping. The system integrates optical and LiDAR data, leveraging machine learning (ML) methods for data fusion to improve biomass predictions. The best ML-based model for dry AGB estimation achieved a coefficient of determination value of 0.88, demonstrating robust performance in diverse field conditions. This approach offers valuable insights for site-specific management, enabling precise weed suppression strategies and promoting sustainable farming practices.
>
---
#### [new 012] KnotDLO: Toward Interpretable Knot Tying
- **分类: cs.RO; cs.CV**

- **简介: 该论文属于机器人操作任务，解决单手打结问题。提出KnotDLO方法，无需示教即可生成可解释的运动策略，实现鲁棒打结。**

- **链接: [http://arxiv.org/pdf/2506.22176v1](http://arxiv.org/pdf/2506.22176v1)**

> **作者:** Holly Dinkel; Raghavendra Navaratna; Jingyi Xiang; Brian Coltin; Trey Smith; Timothy Bretl
>
> **备注:** 4 pages, 5 figures, presented at the Workshop on 3D Visual Representations for Manipulation at the 2023 IEEE International Conference on Robotics and Automation in Yokohama, Japan. Video presentation [https://youtu.be/mg30uCUtpOk]. Poster [https://hollydinkel.github.io/assets/pdf/ICRA20243DVRM_poster.pdf] 3DVRM Workshop [https://3d-manipulation-workshop.github.io/]
>
> **摘要:** This work presents KnotDLO, a method for one-handed Deformable Linear Object (DLO) knot tying that is robust to occlusion, repeatable for varying rope initial configurations, interpretable for generating motion policies, and requires no human demonstrations or training. Grasp and target waypoints for future DLO states are planned from the current DLO shape. Grasp poses are computed from indexing the tracked piecewise linear curve representing the DLO state based on the current curve shape and are piecewise continuous. KnotDLO computes intermediate waypoints from the geometry of the current DLO state and the desired next state. The system decouples visual reasoning from control. In 16 trials of knot tying, KnotDLO achieves a 50% success rate in tying an overhand knot from previously unseen configurations.
>
---
#### [new 013] ASVSim (AirSim for Surface Vehicles): A High-Fidelity Simulation Framework for Autonomous Surface Vehicle Research
- **分类: cs.RO; cs.LG**

- **简介: 该论文属于自主水面船舶研究任务，旨在解决缺乏高保真仿真框架的问题，提出ASVSim平台以支持算法开发与数据生成。**

- **链接: [http://arxiv.org/pdf/2506.22174v1](http://arxiv.org/pdf/2506.22174v1)**

> **作者:** Bavo Lesy; Siemen Herremans; Robin Kerstens; Jan Steckel; Walter Daems; Siegfried Mercelis; Ali Anwar
>
> **备注:** 14 Pages, 11 Figures
>
> **摘要:** The transport industry has recently shown significant interest in unmanned surface vehicles (USVs), specifically for port and inland waterway transport. These systems can improve operational efficiency and safety, which is especially relevant in the European Union, where initiatives such as the Green Deal are driving a shift towards increased use of inland waterways. At the same time, a shortage of qualified personnel is accelerating the adoption of autonomous solutions. However, there is a notable lack of open-source, high-fidelity simulation frameworks and datasets for developing and evaluating such solutions. To address these challenges, we introduce AirSim For Surface Vehicles (ASVSim), an open-source simulation framework specifically designed for autonomous shipping research in inland and port environments. The framework combines simulated vessel dynamics with marine sensor simulation capabilities, including radar and camera systems and supports the generation of synthetic datasets for training computer vision models and reinforcement learning agents. Built upon Cosys-AirSim, ASVSim provides a comprehensive platform for developing autonomous navigation algorithms and generating synthetic datasets. The simulator supports research of both traditional control methods and deep learning-based approaches. Through limited experiments, we demonstrate the potential of the simulator in these research areas. ASVSim is provided as an open-source project under the MIT license, making autonomous navigation research accessible to a larger part of the ocean engineering community.
>
---
#### [new 014] Experimental investigation of pose informed reinforcement learning for skid-steered visual navigation
- **分类: cs.RO; cs.AI; cs.CV; cs.LG; cs.SY; eess.SY**

- **简介: 该论文属于视觉导航任务，旨在解决滑移车辆在复杂地形中的自主导航问题。通过提出一种姿态感知的强化学习方法，提升导航性能。**

- **链接: [http://arxiv.org/pdf/2506.21732v1](http://arxiv.org/pdf/2506.21732v1)**

> **作者:** Ameya Salvi; Venkat Krovi
>
> **摘要:** Vision-based lane keeping is a topic of significant interest in the robotics and autonomous ground vehicles communities in various on-road and off-road applications. The skid-steered vehicle architecture has served as a useful vehicle platform for human controlled operations. However, systematic modeling, especially of the skid-slip wheel terrain interactions (primarily in off-road settings) has created bottlenecks for automation deployment. End-to-end learning based methods such as imitation learning and deep reinforcement learning, have gained prominence as a viable deployment option to counter the lack of accurate analytical models. However, the systematic formulation and subsequent verification/validation in dynamic operation regimes (particularly for skid-steered vehicles) remains a work in progress. To this end, a novel approach for structured formulation for learning visual navigation is proposed and investigated in this work. Extensive software simulations, hardware evaluations and ablation studies now highlight the significantly improved performance of the proposed approach against contemporary literature.
>
---
#### [new 015] Skill-Nav: Enhanced Navigation with Versatile Quadrupedal Locomotion via Waypoint Interface
- **分类: cs.RO**

- **简介: 该论文属于机器人导航任务，旨在解决四足机器人在复杂地形中自主导航的问题。通过引入基于路径点的层次化框架，提升其运动技能与导航的结合能力。**

- **链接: [http://arxiv.org/pdf/2506.21853v1](http://arxiv.org/pdf/2506.21853v1)**

> **作者:** Dewei Wang; Chenjia Ba; Chenhui Li; Jiyuan Shi; Yan Ding; Chi Zhang; Bin Zhao
>
> **备注:** 17pages, 6 figures
>
> **摘要:** Quadrupedal robots have demonstrated exceptional locomotion capabilities through Reinforcement Learning (RL), including extreme parkour maneuvers. However, integrating locomotion skills with navigation in quadrupedal robots has not been fully investigated, which holds promise for enhancing long-distance movement capabilities. In this paper, we propose Skill-Nav, a method that incorporates quadrupedal locomotion skills into a hierarchical navigation framework using waypoints as an interface. Specifically, we train a waypoint-guided locomotion policy using deep RL, enabling the robot to autonomously adjust its locomotion skills to reach targeted positions while avoiding obstacles. Compared with direct velocity commands, waypoints offer a simpler yet more flexible interface for high-level planning and low-level control. Utilizing waypoints as the interface allows for the application of various general planning tools, such as large language models (LLMs) and path planning algorithms, to guide our locomotion policy in traversing terrains with diverse obstacles. Extensive experiments conducted in both simulated and real-world scenarios demonstrate that Skill-Nav can effectively traverse complex terrains and complete challenging navigation tasks.
>
---
#### [new 016] FrankenBot: Brain-Morphic Modular Orchestration for Robotic Manipulation with Vision-Language Models
- **分类: cs.RO; cs.AI; F.4.3; I.2.9**

- **简介: 该论文提出FrankenBot框架，解决机器人在复杂环境中高效执行任务的问题。通过整合视觉语言模型与脑结构模拟，实现任务规划、策略生成等功能的统一协调。**

- **链接: [http://arxiv.org/pdf/2506.21627v1](http://arxiv.org/pdf/2506.21627v1)**

> **作者:** Shiyi Wang; Wenbo Li; Yiteng Chen; Qingyao Wu; Huiping Zhuang
>
> **备注:** 15 pages, 4 figures, under review of NeurIPS
>
> **摘要:** Developing a general robot manipulation system capable of performing a wide range of tasks in complex, dynamic, and unstructured real-world environments has long been a challenging task. It is widely recognized that achieving human-like efficiency and robustness manipulation requires the robotic brain to integrate a comprehensive set of functions, such as task planning, policy generation, anomaly monitoring and handling, and long-term memory, achieving high-efficiency operation across all functions. Vision-Language Models (VLMs), pretrained on massive multimodal data, have acquired rich world knowledge, exhibiting exceptional scene understanding and multimodal reasoning capabilities. However, existing methods typically focus on realizing only a single function or a subset of functions within the robotic brain, without integrating them into a unified cognitive architecture. Inspired by a divide-and-conquer strategy and the architecture of the human brain, we propose FrankenBot, a VLM-driven, brain-morphic robotic manipulation framework that achieves both comprehensive functionality and high operational efficiency. Our framework includes a suite of components, decoupling a part of key functions from frequent VLM calls, striking an optimal balance between functional completeness and system efficiency. Specifically, we map task planning, policy generation, memory management, and low-level interfacing to the cortex, cerebellum, temporal lobe-hippocampus complex, and brainstem, respectively, and design efficient coordination mechanisms for the modules. We conducted comprehensive experiments in both simulation and real-world robotic environments, demonstrating that our method offers significant advantages in anomaly detection and handling, long-term memory, operational efficiency, and stability -- all without requiring any fine-tuning or retraining.
>
---
#### [new 017] Optimal Motion Scaling for Delayed Telesurgery
- **分类: cs.RO**

- **简介: 该论文属于机器人遥操作任务，解决长延迟下手术精度问题。通过用户实验研究延迟与缩放系数的关系，提出个性化模型以优化性能。**

- **链接: [http://arxiv.org/pdf/2506.21689v1](http://arxiv.org/pdf/2506.21689v1)**

> **作者:** Jason Lim; Florian Richter; Zih-Yun Chiu; Jaeyon Lee; Ethan Quist; Nathan Fisher; Jonathan Chambers; Steven Hong; Michael C. Yip
>
> **备注:** Accepted to IROS 2025
>
> **摘要:** Robotic teleoperation over long communication distances poses challenges due to delays in commands and feedback from network latency. One simple yet effective strategy to reduce errors and increase performance under delay is to downscale the relative motion between the operating surgeon and the robot. The question remains as to what is the optimal scaling factor, and how this value changes depending on the level of latency as well as operator tendencies. We present user studies investigating the relationship between latency, scaling factor, and performance. The results of our studies demonstrate a statistically significant difference in performance between users and across scaling factors for certain levels of delay. These findings indicate that the optimal scaling factor for a given level of delay is specific to each user, motivating the need for personalized models for optimal performance. We present techniques to model the user-specific mapping of latency level to scaling factor for optimal performance, leading to an efficient and effective solution to optimizing performance of robotic teleoperation and specifically telesurgery under large communication delay.
>
---
#### [new 018] TOMD: A Trail-based Off-road Multimodal Dataset for Traversable Pathway Segmentation under Challenging Illumination Conditions
- **分类: cs.RO; cs.CV; cs.LG**

- **简介: 该论文属于自主导航任务，解决复杂光照下非结构化地形可通行路径分割问题。构建了TOMD数据集并提出多尺度融合模型。**

- **链接: [http://arxiv.org/pdf/2506.21630v1](http://arxiv.org/pdf/2506.21630v1)**

> **作者:** Yixin Sun; Li Li; Wenke E; Amir Atapour-Abarghouei; Toby P. Breckon
>
> **备注:** 8 pages, 9 figures, 2025 IJCNN
>
> **摘要:** Detecting traversable pathways in unstructured outdoor environments remains a significant challenge for autonomous robots, especially in critical applications such as wide-area search and rescue, as well as incident management scenarios like forest fires. Existing datasets and models primarily target urban settings or wide, vehicle-traversable off-road tracks, leaving a substantial gap in addressing the complexity of narrow, trail-like off-road scenarios. To address this, we introduce the Trail-based Off-road Multimodal Dataset (TOMD), a comprehensive dataset specifically designed for such environments. TOMD features high-fidelity multimodal sensor data -- including 128-channel LiDAR, stereo imagery, GNSS, IMU, and illumination measurements -- collected through repeated traversals under diverse conditions. We also propose a dynamic multiscale data fusion model for accurate traversable pathway prediction. The study analyzes the performance of early, cross, and mixed fusion strategies under varying illumination levels. Results demonstrate the effectiveness of our approach and the relevance of illumination in segmentation performance. We publicly release TOMD at https://github.com/yyyxs1125/TMOD to support future research in trail-based off-road navigation.
>
---
#### [new 019] The DevSafeOps Dilemma: A Systematic Literature Review on Rapidity in Safe Autonomous Driving Development and Operation
- **分类: cs.SE; cs.RO**

- **简介: 该论文属于系统综述任务，旨在探讨DevOps在自动驾驶安全开发中的应用，分析其挑战与解决方案，以推动安全高效的自动驾驶发展。**

- **链接: [http://arxiv.org/pdf/2506.21693v1](http://arxiv.org/pdf/2506.21693v1)**

> **作者:** Ali Nouri; Beatriz Cabrero-Daniel; Fredrik Törner; Christian Berger
>
> **备注:** Accepted for publication in the Journal of Systems and Software (JSS)
>
> **摘要:** Developing autonomous driving (AD) systems is challenging due to the complexity of the systems and the need to assure their safe and reliable operation. The widely adopted approach of DevOps seems promising to support the continuous technological progress in AI and the demand for fast reaction to incidents, which necessitate continuous development, deployment, and monitoring. We present a systematic literature review meant to identify, analyse, and synthesise a broad range of existing literature related to usage of DevOps in autonomous driving development. Our results provide a structured overview of challenges and solutions, arising from applying DevOps to safety-related AI-enabled functions. Our results indicate that there are still several open topics to be addressed to enable safe DevOps for the development of safe AD.
>
---
#### [new 020] Reinforcement Learning with Physics-Informed Symbolic Program Priors for Zero-Shot Wireless Indoor Navigation
- **分类: cs.LG; cs.RO**

- **简介: 该论文属于无线室内导航任务，旨在解决RL训练效率低和泛化能力差的问题。通过引入物理符号程序先验，提升RL性能并减少训练时间。**

- **链接: [http://arxiv.org/pdf/2506.22365v1](http://arxiv.org/pdf/2506.22365v1)**

> **作者:** Tao Li; Haozhe Lei; Mingsheng Yin; Yaqi Hu
>
> **备注:** Spotlight paper at Reinforcement Learning Conference 2025, Workshop on Inductive Biases in Reinforcement Learning
>
> **摘要:** When using reinforcement learning (RL) to tackle physical control tasks, inductive biases that encode physics priors can help improve sample efficiency during training and enhance generalization in testing. However, the current practice of incorporating these helpful physics-informed inductive biases inevitably runs into significant manual labor and domain expertise, making them prohibitive for general users. This work explores a symbolic approach to distill physics-informed inductive biases into RL agents, where the physics priors are expressed in a domain-specific language (DSL) that is human-readable and naturally explainable. Yet, the DSL priors do not translate directly into an implementable policy due to partial and noisy observations and additional physical constraints in navigation tasks. To address this gap, we develop a physics-informed program-guided RL (PiPRL) framework with applications to indoor navigation. PiPRL adopts a hierarchical and modularized neuro-symbolic integration, where a meta symbolic program receives semantically meaningful features from a neural perception module, which form the bases for symbolic programming that encodes physics priors and guides the RL process of a low-level neural controller. Extensive experiments demonstrate that PiPRL consistently outperforms purely symbolic or neural policies and reduces training time by over 26% with the help of the program-based inductive biases.
>
---
#### [new 021] SceneDiffuser++: City-Scale Traffic Simulation via a Generative World Model
- **分类: cs.LG; cs.AI; cs.CV; cs.MA; cs.RO**

- **简介: 该论文属于交通仿真任务，旨在通过生成模型实现城市级交通模拟，解决真实数据不足问题，提出SceneDiffuser++模型完成端到端仿真。**

- **链接: [http://arxiv.org/pdf/2506.21976v1](http://arxiv.org/pdf/2506.21976v1)**

> **作者:** Shuhan Tan; John Lambert; Hong Jeon; Sakshum Kulshrestha; Yijing Bai; Jing Luo; Dragomir Anguelov; Mingxing Tan; Chiyu Max Jiang
>
> **备注:** Accepted to CVPR 2025
>
> **摘要:** The goal of traffic simulation is to augment a potentially limited amount of manually-driven miles that is available for testing and validation, with a much larger amount of simulated synthetic miles. The culmination of this vision would be a generative simulated city, where given a map of the city and an autonomous vehicle (AV) software stack, the simulator can seamlessly simulate the trip from point A to point B by populating the city around the AV and controlling all aspects of the scene, from animating the dynamic agents (e.g., vehicles, pedestrians) to controlling the traffic light states. We refer to this vision as CitySim, which requires an agglomeration of simulation technologies: scene generation to populate the initial scene, agent behavior modeling to animate the scene, occlusion reasoning, dynamic scene generation to seamlessly spawn and remove agents, and environment simulation for factors such as traffic lights. While some key technologies have been separately studied in various works, others such as dynamic scene generation and environment simulation have received less attention in the research community. We propose SceneDiffuser++, the first end-to-end generative world model trained on a single loss function capable of point A-to-B simulation on a city scale integrating all the requirements above. We demonstrate the city-scale traffic simulation capability of SceneDiffuser++ and study its superior realism under long simulation conditions. We evaluate the simulation quality on an augmented version of the Waymo Open Motion Dataset (WOMD) with larger map regions to support trip-level simulation.
>
---
#### [new 022] Robust and Accurate Multi-view 2D/3D Image Registration with Differentiable X-ray Rendering and Dual Cross-view Constraints
- **分类: cs.CV; cs.RO**

- **简介: 该论文属于2D/3D图像配准任务，旨在解决术中多视角图像与术前模型的精准对齐问题。通过设计联合损失函数和跨视角约束，提升配准鲁棒性与准确性。**

- **链接: [http://arxiv.org/pdf/2506.22191v1](http://arxiv.org/pdf/2506.22191v1)**

> **作者:** Yuxin Cui; Rui Song; Yibin Li; Max Q. -H. Meng; Zhe Min
>
> **备注:** ICRA 2025
>
> **摘要:** Robust and accurate 2D/3D registration, which aligns preoperative models with intraoperative images of the same anatomy, is crucial for successful interventional navigation. To mitigate the challenge of a limited field of view in single-image intraoperative scenarios, multi-view 2D/3D registration is required by leveraging multiple intraoperative images. In this paper, we propose a novel multi-view 2D/3D rigid registration approach comprising two stages. In the first stage, a combined loss function is designed, incorporating both the differences between predicted and ground-truth poses and the dissimilarities (e.g., normalized cross-correlation) between simulated and observed intraoperative images. More importantly, additional cross-view training loss terms are introduced for both pose and image losses to explicitly enforce cross-view constraints. In the second stage, test-time optimization is performed to refine the estimated poses from the coarse stage. Our method exploits the mutual constraints of multi-view projection poses to enhance the robustness of the registration process. The proposed framework achieves a mean target registration error (mTRE) of $0.79 \pm 2.17$ mm on six specimens from the DeepFluoro dataset, demonstrating superior performance compared to state-of-the-art registration algorithms.
>
---
#### [new 023] Stochastic Neural Control Barrier Functions
- **分类: eess.SY; cs.RO; cs.SY**

- **简介: 该论文属于安全控制任务，解决随机神经控制屏障函数（SNCBF）的验证问题，提出两种合成框架并验证其有效性。**

- **链接: [http://arxiv.org/pdf/2506.21697v1](http://arxiv.org/pdf/2506.21697v1)**

> **作者:** Hongchao Zhang; Manan Tayal; Jackson Cox; Pushpak Jagtap; Shishir Kolathaya; Andrew Clark
>
> **摘要:** Control Barrier Functions (CBFs) are utilized to ensure the safety of control systems. CBFs act as safety filters in order to provide safety guarantees without compromising system performance. These safety guarantees rely on the construction of valid CBFs. Due to their complexity, CBFs can be represented by neural networks, known as neural CBFs (NCBFs). Existing works on the verification of the NCBF focus on the synthesis and verification of NCBFs in deterministic settings, leaving the stochastic NCBFs (SNCBFs) less studied. In this work, we propose a verifiably safe synthesis for SNCBFs. We consider the cases of smooth SNCBFs with twice-differentiable activation functions and SNCBFs that utilize the Rectified Linear Unit or ReLU activation function. We propose a verification-free synthesis framework for smooth SNCBFs and a verification-in-the-loop synthesis framework for both smooth and ReLU SNCBFs. and we validate our frameworks in three cases, namely, the inverted pendulum, Darboux, and the unicycle model.
>
---
#### [new 024] ARMOR: Robust Reinforcement Learning-based Control for UAVs under Physical Attacks
- **分类: cs.LG; cs.CR; cs.RO**

- **简介: 该论文属于无人机控制任务，解决传感器受物理攻击导致的安全问题。提出ARMOR方法，通过鲁棒状态表示提升无人机在对抗性攻击下的稳定性与安全性。**

- **链接: [http://arxiv.org/pdf/2506.22423v1](http://arxiv.org/pdf/2506.22423v1)**

> **作者:** Pritam Dash; Ethan Chan; Nathan P. Lawrence; Karthik Pattabiraman
>
> **摘要:** Unmanned Aerial Vehicles (UAVs) depend on onboard sensors for perception, navigation, and control. However, these sensors are susceptible to physical attacks, such as GPS spoofing, that can corrupt state estimates and lead to unsafe behavior. While reinforcement learning (RL) offers adaptive control capabilities, existing safe RL methods are ineffective against such attacks. We present ARMOR (Adaptive Robust Manipulation-Optimized State Representations), an attack-resilient, model-free RL controller that enables robust UAV operation under adversarial sensor manipulation. Instead of relying on raw sensor observations, ARMOR learns a robust latent representation of the UAV's physical state via a two-stage training framework. In the first stage, a teacher encoder, trained with privileged attack information, generates attack-aware latent states for RL policy training. In the second stage, a student encoder is trained via supervised learning to approximate the teacher's latent states using only historical sensor data, enabling real-world deployment without privileged information. Our experiments show that ARMOR outperforms conventional methods, ensuring UAV safety. Additionally, ARMOR improves generalization to unseen attacks and reduces training cost by eliminating the need for iterative adversarial training.
>
---
#### [new 025] Advanced System Engineering Approaches to Emerging Challenges in Planetary and Deep-Space Exploration
- **分类: astro-ph.IM; astro-ph.EP; cs.RO; cs.SY; eess.SY**

- **简介: 该论文属于深空探测任务，解决行星与深空电子系统挑战，提出五项创新技术，包括定位、通信、热防护、CubeSat设计和电源管理。**

- **链接: [http://arxiv.org/pdf/2506.21648v1](http://arxiv.org/pdf/2506.21648v1)**

> **作者:** J. de Curtò; Cristina LiCalzi; Julien Tubiana Warin; Jack Gehlert; Brian Langbein; Alexandre Gamboa; Chris Sixbey; William Maguire; Santiago Fernández; Álvaro Maestroarena; Alex Brenchley; Logan Maroclo; Philemon Mercado; Joshua DeJohn; Cesar Velez; Ethan Dahmus; Taylor Steinys; David Fritz; I. de Zarzà
>
> **摘要:** This paper presents innovative solutions to critical challenges in planetary and deep-space exploration electronics. We synthesize findings across diverse mission profiles, highlighting advances in: (1) MARTIAN positioning systems with dual-frequency transmission to achieve $\pm$1m horizontal accuracy; (2) artificial reef platforms for Titan's hydrocarbon seas utilizing specialized sensor arrays and multi-stage communication chains; (3) precision orbital rendezvous techniques demonstrating novel thermal protection solutions; (4) miniaturized CubeSat architectures for asteroid exploration with optimized power-to-mass ratios; and (5) next-generation power management systems for MARS rovers addressing dust accumulation challenges. These innovations represent promising directions for future space exploration technologies, particularly in environments where traditional Earth-based electronic solutions prove inadequate. The interdisciplinary nature of these developments highlights the critical intersection of aerospace engineering, electrical engineering, and planetary science in advancing human exploration capabilities beyond Earth orbit.
>
---
#### [new 026] M3PO: Massively Multi-Task Model-Based Policy Optimization
- **分类: cs.LG; cs.RO**

- **简介: 该论文提出M3PO，属于强化学习领域，解决单任务样本效率低和多任务泛化差的问题。通过结合模型预测与探索策略，提升性能。**

- **链接: [http://arxiv.org/pdf/2506.21782v1](http://arxiv.org/pdf/2506.21782v1)**

> **作者:** Aditya Narendra; Dmitry Makarov; Aleksandr Panov
>
> **备注:** 6 pages, 4 figures. Accepted at IEEE/RSJ IROS 2025. Full version, including appendix and implementation details
>
> **摘要:** We introduce Massively Multi-Task Model-Based Policy Optimization (M3PO), a scalable model-based reinforcement learning (MBRL) framework designed to address sample inefficiency in single-task settings and poor generalization in multi-task domains. Existing model-based approaches like DreamerV3 rely on pixel-level generative models that neglect control-centric representations, while model-free methods such as PPO suffer from high sample complexity and weak exploration. M3PO integrates an implicit world model, trained to predict task outcomes without observation reconstruction, with a hybrid exploration strategy that combines model-based planning and model-free uncertainty-driven bonuses. This eliminates the bias-variance trade-off in prior methods by using discrepancies between model-based and model-free value estimates to guide exploration, while maintaining stable policy updates through a trust-region optimizer. M3PO provides an efficient and robust alternative to existing model-based policy optimization approaches and achieves state-of-the-art performance across multiple benchmarks.
>
---
#### [new 027] Integrating Multi-Modal Sensors: A Review of Fusion Techniques for Intelligent Vehicles
- **分类: cs.CV; cs.MM; cs.RO**

- **简介: 该论文属于多模态传感器融合任务，旨在提升自动驾驶环境感知能力，解决传感器局限性问题，通过深度学习方法进行系统综述与分析。**

- **链接: [http://arxiv.org/pdf/2506.21885v1](http://arxiv.org/pdf/2506.21885v1)**

> **作者:** Chuheng Wei; Ziye Qin; Ziyan Zhang; Guoyuan Wu; Matthew J. Barth
>
> **备注:** Accepted by IEEE IV 2025
>
> **摘要:** Multi-sensor fusion plays a critical role in enhancing perception for autonomous driving, overcoming individual sensor limitations, and enabling comprehensive environmental understanding. This paper first formalizes multi-sensor fusion strategies into data-level, feature-level, and decision-level categories and then provides a systematic review of deep learning-based methods corresponding to each strategy. We present key multi-modal datasets and discuss their applicability in addressing real-world challenges, particularly in adverse weather conditions and complex urban environments. Additionally, we explore emerging trends, including the integration of Vision-Language Models (VLMs), Large Language Models (LLMs), and the role of sensor fusion in end-to-end autonomous driving, highlighting its potential to enhance system adaptability and robustness. Our work offers valuable insights into current methods and future directions for multi-sensor fusion in autonomous driving.
>
---
## 更新

#### [replaced 001] AirLine: Efficient Learnable Line Detection with Local Edge Voting
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2303.16500v3](http://arxiv.org/pdf/2303.16500v3)**

> **作者:** Xiao Lin; Chen Wang
>
> **摘要:** Line detection is widely used in many robotic tasks such as scene recognition, 3D reconstruction, and simultaneous localization and mapping (SLAM). Compared to points, lines can provide both low-level and high-level geometrical information for downstream tasks. In this paper, we propose a novel learnable edge-based line detection algorithm, AirLine, which can be applied to various tasks. In contrast to existing learnable endpoint-based methods, which are sensitive to the geometrical condition of environments, AirLine can extract line segments directly from edges, resulting in a better generalization ability for unseen environments. To balance efficiency and accuracy, we introduce a region-grow algorithm and a local edge voting scheme for line parameterization. To the best of our knowledge, AirLine is one of the first learnable edge-based line detection methods. Our extensive experiments have shown that it retains state-of-the-art-level precision, yet with a 3 to 80 times runtime acceleration compared to other learning-based methods, which is critical for low-power robots.
>
---
#### [replaced 002] TritonZ: A Remotely Operated Underwater Rover with Manipulator Arm for Exploration and Rescue Operations
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.18343v2](http://arxiv.org/pdf/2506.18343v2)**

> **作者:** Kawser Ahmed; Mir Shahriar Fardin; Md Arif Faysal Nayem; Fahim Hafiz; Swakkhar Shatabda
>
> **备注:** 7 pages, 5 figures
>
> **摘要:** The increasing demand for underwater exploration and rescue operations enforces the development of advanced wireless or semi-wireless underwater vessels equipped with manipulator arms. This paper presents the implementation of a semi-wireless underwater vehicle, "TritonZ" equipped with a manipulator arm, tailored for effective underwater exploration and rescue operations. The vehicle's compact design enables deployment in different submarine surroundings, addressing the need for wireless systems capable of navigating challenging underwater terrains. The manipulator arm can interact with the environment, allowing the robot to perform sophisticated tasks during exploration and rescue missions in emergency situations. TritonZ is equipped with various sensors such as Pi-Camera, Humidity, and Temperature sensors to send real-time environmental data. Our underwater vehicle controlled using a customized remote controller can navigate efficiently in the water where Pi-Camera enables live streaming of the surroundings. Motion control and video capture are performed simultaneously using this camera. The manipulator arm is designed to perform various tasks, similar to grasping, manipulating, and collecting underwater objects. Experimental results shows the efficacy of the proposed remotely operated vehicle in performing a variety of underwater exploration and rescue tasks. Additionally, the results show that TritonZ can maintain an average of 13.5cm/s with a minimal delay of 2-3 seconds. Furthermore, the vehicle can sustain waves underwater by maintaining its position as well as average velocity. The full project details and source code can be accessed at this link: https://github.com/kawser-ahmed-byte/TritonZ
>
---
#### [replaced 003] Cooperative Bearing-Only Target Pursuit via Multiagent Reinforcement Learning: Design and Experiment
- **分类: cs.MA; cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2503.08740v2](http://arxiv.org/pdf/2503.08740v2)**

> **作者:** Jianan Li; Zhikun Wang; Susheng Ding; Shiliang Guo; Shiyu Zhao
>
> **备注:** To appear in the 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)
>
> **摘要:** This paper addresses the multi-robot pursuit problem for an unknown target, encompassing both target state estimation and pursuit control. First, in state estimation, we focus on using only bearing information, as it is readily available from vision sensors and effective for small, distant targets. Challenges such as instability due to the nonlinearity of bearing measurements and singularities in the two-angle representation are addressed through a proposed uniform bearing-only information filter. This filter integrates multiple 3D bearing measurements, provides a concise formulation, and enhances stability and resilience to target loss caused by limited field of view (FoV). Second, in target pursuit control within complex environments, where challenges such as heterogeneity and limited FoV arise, conventional methods like differential games or Voronoi partitioning often prove inadequate. To address these limitations, we propose a novel multiagent reinforcement learning (MARL) framework, enabling multiple heterogeneous vehicles to search, localize, and follow a target while effectively handling those challenges. Third, to bridge the sim-to-real gap, we propose two key techniques: incorporating adjustable low-level control gains in training to replicate the dynamics of real-world autonomous ground vehicles (AGVs), and proposing spectral-normalized RL algorithms to enhance policy smoothness and robustness. Finally, we demonstrate the successful zero-shot transfer of the MARL controllers to AGVs, validating the effectiveness and practical feasibility of our approach. The accompanying video is available at https://youtu.be/HO7FJyZiJ3E.
>
---
#### [replaced 004] Hierarchical Intention-Aware Expressive Motion Generation for Humanoid Robots
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.01563v3](http://arxiv.org/pdf/2506.01563v3)**

> **作者:** Lingfan Bao; Yan Pan; Tianhu Peng; Dimitrios Kanoulas; Chengxu Zhou
>
> **备注:** 7 pages, 2 figures, IEEE conference paper
>
> **摘要:** Effective human-robot interaction requires robots to identify human intentions and generate expressive, socially appropriate motions in real-time. Existing approaches often rely on fixed motion libraries or computationally expensive generative models. We propose a hierarchical framework that combines intention-aware reasoning via in-context learning (ICL) with real-time motion generation using diffusion models. Our system introduces structured prompting with confidence scoring, fallback behaviors, and social context awareness to enable intention refinement and adaptive response. Leveraging large-scale motion datasets and efficient latent-space denoising, the framework generates diverse, physically plausible gestures suitable for dynamic humanoid interactions. Experimental validation on a physical platform demonstrates the robustness and social alignment of our method in realistic scenarios.
>
---
#### [replaced 005] eCAV: An Edge-Assisted Evaluation Platform for Connected Autonomous Vehicles
- **分类: cs.RO; cs.MA; cs.NI; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2506.16535v2](http://arxiv.org/pdf/2506.16535v2)**

> **作者:** Tyler Landle; Jordan Rapp; Dean Blank; Chandramouli Amarnath; Abhijit Chatterjee; Alexandros Daglis; Umakishore Ramachandran
>
> **摘要:** As autonomous vehicles edge closer to widespread adoption, enhancing road safety through collision avoidance and minimization of collateral damage becomes imperative. Vehicle-to-everything (V2X) technologies, which include vehicle-to-vehicle (V2V), vehicle-to-infrastructure (V2I), and vehicle-to-cloud (V2C), are being proposed as mechanisms to achieve this safety improvement. Simulation-based testing is crucial for early-stage evaluation of Connected Autonomous Vehicle (CAV) control systems, offering a safer and more cost-effective alternative to real-world tests. However, simulating large 3D environments with many complex single- and multi-vehicle sensors and controllers is computationally intensive. There is currently no evaluation framework that can effectively evaluate realistic scenarios involving large numbers of autonomous vehicles. We propose eCAV -- an efficient, modular, and scalable evaluation platform to facilitate both functional validation of algorithmic approaches to increasing road safety, as well as performance prediction of algorithms of various V2X technologies, including a futuristic Vehicle-to-Edge control plane and correspondingly designed control algorithms. eCAV can model up to 256 vehicles running individual control algorithms without perception enabled, which is $8\times$ more vehicles than what is possible with state-of-the-art alternatives.
>
---
#### [replaced 006] Efficient Reconfiguration of Tile Arrangements by a Single Active Robot
- **分类: cs.CG; cs.DS; cs.RO; F.2.2**

- **链接: [http://arxiv.org/pdf/2502.09299v2](http://arxiv.org/pdf/2502.09299v2)**

> **作者:** Aaron T. Becker; Sándor P. Fekete; Jonas Friemel; Ramin Kosfeld; Peter Kramer; Harm Kube; Christian Rieck; Christian Scheffer; Arne Schmidt
>
> **备注:** 19 pages, 15 figures, to appear in the proceedings of the 37th Canadian Conference on Computational Geometry (CCCG 2025)
>
> **摘要:** We consider the problem of reconfiguring a two-dimensional connected grid arrangement of passive building blocks from a start configuration to a goal configuration, using a single active robot that can move on the tiles, remove individual tiles from a given location and physically move them to a new position by walking on the remaining configuration. The objective is to determine a schedule that minimizes the overall makespan, while keeping the tile configuration connected. We provide both negative and positive results. (1) We generalize the problem by introducing weighted movement costs, which can vary depending on whether tiles are carried or not, and prove that this variant is NP-hard. (2) We give a polynomial-time constant-factor approximation algorithm for the case of disjoint start and target bounding boxes, which additionally yields optimal carry distance for 2-scaled instances.
>
---
#### [replaced 007] FEAST: A Flexible Mealtime-Assistance System Towards In-the-Wild Personalization
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2506.14968v2](http://arxiv.org/pdf/2506.14968v2)**

> **作者:** Rajat Kumar Jenamani; Tom Silver; Ben Dodson; Shiqin Tong; Anthony Song; Yuting Yang; Ziang Liu; Benjamin Howe; Aimee Whitneck; Tapomayukh Bhattacharjee
>
> **备注:** RSS 2025 - Best Paper Award
>
> **摘要:** Physical caregiving robots hold promise for improving the quality of life of millions worldwide who require assistance with feeding. However, in-home meal assistance remains challenging due to the diversity of activities (e.g., eating, drinking, mouth wiping), contexts (e.g., socializing, watching TV), food items, and user preferences that arise during deployment. In this work, we propose FEAST, a flexible mealtime-assistance system that can be personalized in-the-wild to meet the unique needs of individual care recipients. Developed in collaboration with two community researchers and informed by a formative study with a diverse group of care recipients, our system is guided by three key tenets for in-the-wild personalization: adaptability, transparency, and safety. FEAST embodies these principles through: (i) modular hardware that enables switching between assisted feeding, drinking, and mouth-wiping, (ii) diverse interaction methods, including a web interface, head gestures, and physical buttons, to accommodate diverse functional abilities and preferences, and (iii) parameterized behavior trees that can be safely and transparently adapted using a large language model. We evaluate our system based on the personalization requirements identified in our formative study, demonstrating that FEAST offers a wide range of transparent and safe adaptations and outperforms a state-of-the-art baseline limited to fixed customizations. To demonstrate real-world applicability, we conduct an in-home user study with two care recipients (who are community researchers), feeding them three meals each across three diverse scenarios. We further assess FEAST's ecological validity by evaluating with an Occupational Therapist previously unfamiliar with the system. In all cases, users successfully personalize FEAST to meet their individual needs and preferences. Website: https://emprise.cs.cornell.edu/feast
>
---
#### [replaced 008] Estimating Spatially-Dependent GPS Errors Using a Swarm of Robots
- **分类: cs.RO; cs.SY; eess.SY**

- **链接: [http://arxiv.org/pdf/2506.19712v2](http://arxiv.org/pdf/2506.19712v2)**

> **作者:** Praneeth Somisetty; Robert Griffin; Victor M. Baez; Miguel F. Arevalo-Castiblanco; Aaron T. Becker; Jason M. O'Kane
>
> **备注:** 6 pages, 7 figures, 2025 IEEE 21st International Conference on Automation Science and Engineering
>
> **摘要:** External factors, including urban canyons and adversarial interference, can lead to Global Positioning System (GPS) inaccuracies that vary as a function of the position in the environment. This study addresses the challenge of estimating a static, spatially-varying error function using a team of robots. We introduce a State Bias Estimation Algorithm (SBE) whose purpose is to estimate the GPS biases. The central idea is to use sensed estimates of the range and bearing to the other robots in the team to estimate changes in bias across the environment. A set of drones moves in a 2D environment, each sampling data from GPS, range, and bearing sensors. The biases calculated by the SBE at estimated positions are used to train a Gaussian Process Regression (GPR) model. We use a Sparse Gaussian process-based Informative Path Planning (IPP) algorithm that identifies high-value regions of the environment for data collection. The swarm plans paths that maximize information gain in each iteration, further refining their understanding of the environment's positional bias landscape. We evaluated SBE and IPP in simulation and compared the IPP methodology to an open-loop strategy.
>
---
#### [replaced 009] QT-DoG: Quantization-aware Training for Domain Generalization
- **分类: cs.LG; cs.AI; cs.CV; cs.RO**

- **链接: [http://arxiv.org/pdf/2410.06020v2](http://arxiv.org/pdf/2410.06020v2)**

> **作者:** Saqib Javed; Hieu Le; Mathieu Salzmann
>
> **备注:** Accepted at International Conference on Machine Learning (ICML) 2025. Project website: https://saqibjaved1.github.io/QT_DoG/
>
> **摘要:** A key challenge in Domain Generalization (DG) is preventing overfitting to source domains, which can be mitigated by finding flatter minima in the loss landscape. In this work, we propose Quantization-aware Training for Domain Generalization (QT-DoG) and demonstrate that weight quantization effectively leads to flatter minima in the loss landscape, thereby enhancing domain generalization. Unlike traditional quantization methods focused on model compression, QT-DoG exploits quantization as an implicit regularizer by inducing noise in model weights, guiding the optimization process toward flatter minima that are less sensitive to perturbations and overfitting. We provide both an analytical perspective and empirical evidence demonstrating that quantization inherently encourages flatter minima, leading to better generalization across domains. Moreover, with the benefit of reducing the model size through quantization, we demonstrate that an ensemble of multiple quantized models further yields superior accuracy than the state-of-the-art DG approaches with no computational or memory overheads. Code is released at: https://saqibjaved1.github.io/QT_DoG/.
>
---
#### [replaced 010] Large-Scale Multirobot Coverage Path Planning on Grids With Path Deconfliction
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2411.01707v3](http://arxiv.org/pdf/2411.01707v3)**

> **作者:** Jingtao Tang; Zining Mao; Hang Ma
>
> **备注:** accepted to T-RO
>
> **摘要:** We study Multi-Robot Coverage Path Planning (MCPP) on a 4-neighbor 2D grid G, which aims to compute paths for multiple robots to cover all cells of G. Traditional approaches are limited as they first compute coverage trees on a quadrant coarsened grid H and then employ the Spanning Tree Coverage (STC) paradigm to generate paths on G, making them inapplicable to grids with partially obstructed 2x2 blocks. To address this limitation, we reformulate the problem directly on G, revolutionizing grid-based MCPP solving and establishing new NP-hardness results. We introduce Extended-STC (ESTC), a novel paradigm that extends STC to ensure complete coverage with bounded suboptimality, even when H includes partially obstructed blocks. Furthermore, we present LS-MCPP, a new algorithmic framework that integrates ESTC with three novel types of neighborhood operators within a local search strategy to optimize coverage paths directly on G. Unlike prior grid-based MCPP work, our approach also incorporates a versatile post-processing procedure that applies Multi-Agent Path Finding (MAPF) techniques to MCPP for the first time, enabling a fusion of these two important fields in multi-robot coordination. This procedure effectively resolves inter-robot conflicts and accommodates turning costs by solving a MAPF variant, making our MCPP solutions more practical for real-world applications. Extensive experiments demonstrate that our approach significantly improves solution quality and efficiency, managing up to 100 robots on grids as large as 256x256 within minutes of runtime. Validation with physical robots confirms the feasibility of our solutions under real-world conditions.
>
---
#### [replaced 011] Mitigating Metropolitan Carbon Emissions with Dynamic Eco-driving at Scale
- **分类: eess.SY; cs.AI; cs.LG; cs.MA; cs.RO; cs.SY**

- **链接: [http://arxiv.org/pdf/2408.05609v2](http://arxiv.org/pdf/2408.05609v2)**

> **作者:** Vindula Jayawardana; Baptiste Freydt; Ao Qu; Cameron Hickert; Edgar Sanchez; Catherine Tang; Mark Taylor; Blaine Leonard; Cathy Wu
>
> **备注:** Accepted for publication at Transportation Research Part C: Emerging Technologies
>
> **摘要:** The sheer scale and diversity of transportation make it a formidable sector to decarbonize. Here, we consider an emerging opportunity to reduce carbon emissions: the growing adoption of semi-autonomous vehicles, which can be programmed to mitigate stop-and-go traffic through intelligent speed commands and, thus, reduce emissions. But would such dynamic eco-driving move the needle on climate change? A comprehensive impact analysis has been out of reach due to the vast array of traffic scenarios and the complexity of vehicle emissions. We address this challenge with large-scale scenario modeling efforts and by using multi-task deep reinforcement learning with a carefully designed network decomposition strategy. We perform an in-depth prospective impact assessment of dynamic eco-driving at 6,011 signalized intersections across three major US metropolitan cities, simulating a million traffic scenarios. Overall, we find that vehicle trajectories optimized for emissions can cut city-wide intersection carbon emissions by 11-22%, without harming throughput or safety, and with reasonable assumptions, equivalent to the national emissions of Israel and Nigeria, respectively. We find that 10% eco-driving adoption yields 25%-50% of the total reduction, and nearly 70% of the benefits come from 20% of intersections, suggesting near-term implementation pathways. However, the composition of this high-impact subset of intersections varies considerably across different adoption levels, with minimal overlap, calling for careful strategic planning for eco-driving deployments. Moreover, the impact of eco-driving, when considered jointly with projections of vehicle electrification and hybrid vehicle adoption remains significant. More broadly, this work paves the way for large-scale analysis of traffic externalities, such as time, safety, and air quality, and the potential impact of solution strategies.
>
---
#### [replaced 012] Personalized Robotic Object Rearrangement from Scene Context
- **分类: cs.RO; cs.AI**

- **链接: [http://arxiv.org/pdf/2505.11108v2](http://arxiv.org/pdf/2505.11108v2)**

> **作者:** Kartik Ramachandruni; Sonia Chernova
>
> **备注:** Accepted at IEEE ROMAN 2025
>
> **摘要:** Object rearrangement is a key task for household robots requiring personalization without explicit instructions, meaningful object placement in environments occupied with objects, and generalization to unseen objects and new environments. To facilitate research addressing these challenges, we introduce PARSEC, an object rearrangement benchmark for learning user organizational preferences from observed scene context to place objects in a partially arranged environment. PARSEC is built upon a novel dataset of 110K rearrangement examples crowdsourced from 72 users, featuring 93 object categories and 15 environments. To better align with real-world organizational habits, we propose ContextSortLM, an LLM-based personalized rearrangement model that handles flexible user preferences by explicitly accounting for objects with multiple valid placement locations when placing items in partially arranged environments. We evaluate ContextSortLM and existing personalized rearrangement approaches on the PARSEC benchmark and complement these findings with a crowdsourced evaluation of 108 online raters ranking model predictions based on alignment with user preferences. Our results indicate that personalized rearrangement models leveraging multiple scene context sources perform better than models relying on a single context source. Moreover, ContextSortLM outperforms other models in placing objects to replicate the target user's arrangement and ranks among the top two in all three environment categories, as rated by online evaluators. Importantly, our evaluation highlights challenges associated with modeling environment semantics across different environment categories and provides recommendations for future work.
>
---
#### [replaced 013] Aux-Think: Exploring Reasoning Strategies for Data-Efficient Vision-Language Navigation
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2505.11886v3](http://arxiv.org/pdf/2505.11886v3)**

> **作者:** Shuo Wang; Yongcai Wang; Wanting Li; Xudong Cai; Yucheng Wang; Maiyue Chen; Kaihui Wang; Zhizhong Su; Deying Li; Zhaoxin Fan
>
> **摘要:** Vision-Language Navigation (VLN) is a critical task for developing embodied agents that can follow natural language instructions to navigate in complex real-world environments. Recent advances in VLN by large pretrained models have significantly improved generalization and instruction grounding compared to traditional approaches. However, the role of reasoning strategies in navigation-an action-centric, long-horizon task-remains underexplored, despite Chain-of-Thought (CoT) reasoning's demonstrated success in static tasks like visual question answering. To address this gap, we conduct the first systematic evaluation of reasoning strategies for VLN, including No-Think (direct action prediction), Pre-Think (reason before action), and Post-Think (reason after action). Surprisingly, our findings reveal the Inference-time Reasoning Collapse issue, where inference-time reasoning degrades navigation accuracy, highlighting the challenges of integrating reasoning into VLN. Based on this insight, we propose Aux-Think, a framework that trains models to internalize structured reasoning patterns through CoT supervision, while inferring action directly without reasoning in online prediction. To support this framework, we release R2R-CoT-320k, the first Chain-of-Thought annotated dataset for VLN. Extensive experiments show that Aux-Think reduces training effort greatly and achieves the best performance under the same data scale.
>
---
#### [replaced 014] TrajFlow: Learning Distributions over Trajectories for Human Behavior Prediction
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2304.05166v5](http://arxiv.org/pdf/2304.05166v5)**

> **作者:** Anna Mészáros; Julian F. Schumann; Javier Alonso-Mora; Arkady Zgonnikov; Jens Kober
>
> **摘要:** Predicting the future behavior of human road users is an important aspect for the development of risk-aware autonomous vehicles. While many models have been developed towards this end, effectively capturing and predicting the variability inherent to human behavior still remains an open challenge. This paper proposes TrajFlow - a new approach for probabilistic trajectory prediction based on Normalizing Flows. We reformulate the problem of capturing distributions over trajectories into capturing distributions over abstracted trajectory features using an autoencoder, simplifying the learning task of the Normalizing Flows. TrajFlow outperforms state-of-the-art behavior prediction models in capturing full trajectory distributions in two synthetic benchmarks with known true distributions, and is competitive on the naturalistic datasets ETH/UCY, rounD, and nuScenes. Our results demonstrate the effectiveness of TrajFlow in probabilistic prediction of human behavior.
>
---
#### [replaced 015] Haptic-ACT -- Pseudo Oocyte Manipulation by a Robot Using Multimodal Information and Action Chunking with Transformers
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2506.18212v2](http://arxiv.org/pdf/2506.18212v2)**

> **作者:** Pedro Miguel Uriguen Eljuri; Hironobu Shibata; Maeyama Katsuyoshi; Yuanyuan Jia; Tadahiro Taniguchi
>
> **备注:** Accepted at IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS2025) Project website https://tanichu-laboratory.github.io/pedro_haptic_act_iros2025/
>
> **摘要:** In this paper we introduce Haptic-ACT, an advanced robotic system for pseudo oocyte manipulation, integrating multimodal information and Action Chunking with Transformers (ACT). Traditional automation methods for oocyte transfer rely heavily on visual perception, often requiring human supervision due to biological variability and environmental disturbances. Haptic-ACT enhances ACT by incorporating haptic feedback, enabling real-time grasp failure detection and adaptive correction. Additionally, we introduce a 3D-printed TPU soft gripper to facilitate delicate manipulations. Experimental results demonstrate that Haptic-ACT improves the task success rate, robustness, and adaptability compared to conventional ACT, particularly in dynamic environments. These findings highlight the potential of multimodal learning in robotics for biomedical automation.
>
---
#### [replaced 016] UAV-based path planning for efficient localization of non-uniformly distributed weeds using prior knowledge: A reinforcement-learning approach
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2412.11717v2](http://arxiv.org/pdf/2412.11717v2)**

> **作者:** Rick van Essen; Eldert van Henten; Gert Kootstra
>
> **摘要:** UAVs are becoming popular in agriculture, however, they usually use time-consuming row-by-row flight paths. This paper presents a deep-reinforcement-learning-based approach for path planning to efficiently localize weeds in agricultural fields using UAVs with minimal flight-path length. The method combines prior knowledge about the field containing uncertain, low-resolution weed locations with in-flight weed detections. The search policy was learned using deep Q-learning. We trained the agent in simulation, allowing a thorough evaluation of the weed distribution, typical errors in the perception system, prior knowledge, and different stopping criteria on the planner's performance. When weeds were non-uniformly distributed over the field, the agent found them faster than a row-by-row path, showing its capability to learn and exploit the weed distribution. Detection errors and prior knowledge quality had a minor effect on the performance, indicating that the learned search policy was robust to detection errors and did not need detailed prior knowledge. The agent also learned to terminate the search. To test the transferability of the learned policy to a real-world scenario, the planner was tested on real-world image data without further training, which showed a 66% shorter path compared to a row-by-row path at the cost of a 10% lower percentage of found weeds. Strengths and weaknesses of the planner for practical application are comprehensively discussed, and directions for further development are provided. Overall, it is concluded that the learned search policy can improve the efficiency of finding non-uniformly distributed weeds using a UAV and shows potential for use in agricultural practice.
>
---
#### [replaced 017] RESPLE: Recursive Spline Estimation for LiDAR-Based Odometry
- **分类: cs.RO**

- **链接: [http://arxiv.org/pdf/2504.11580v2](http://arxiv.org/pdf/2504.11580v2)**

> **作者:** Ziyu Cao; William Talbot; Kailai Li
>
> **摘要:** We present a novel recursive Bayesian estimation framework using B-splines for continuous-time 6-DoF dynamic motion estimation. The state vector consists of a recurrent set of position control points and orientation control point increments, enabling efficient estimation via a modified iterated extended Kalman filter without involving error-state formulations. The resulting recursive spline estimator (RESPLE) is further leveraged to develop a versatile suite of direct LiDAR-based odometry solutions, supporting the integration of one or multiple LiDARs and an IMU. We conduct extensive real-world evaluations using public datasets and our own experiments, covering diverse sensor setups, platforms, and environments. Compared to existing systems, RESPLE achieves comparable or superior estimation accuracy and robustness, while attaining real-time efficiency. Our results and analysis demonstrate RESPLE's strength in handling highly dynamic motions and complex scenes within a lightweight and flexible design, showing strong potential as a universal framework for multi-sensor motion estimation. We release the source code and experimental datasets at https://github.com/ASIG-X/RESPLE.
>
---
